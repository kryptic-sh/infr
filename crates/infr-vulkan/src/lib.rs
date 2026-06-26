//! Vulkan backend (`ash` + SPIR-V). The MVP `Backend` impl.
//!
//! Reference: `~/Projects/llama.cpp/ggml/src/ggml-vulkan/` and its `vulkan-shaders/*.comp`
//! (reuse the tuned quant matmul / dequant / attention shaders). Enable device features
//! `VK_KHR_cooperative_matrix`, `shaderFloat16`, `VK_KHR_16bit_storage`,
//! `VK_KHR_shader_subgroup_extended_types`. See PLAN.md.
#![allow(dead_code)]

mod linear;
mod matmul;

use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::MemoryLocation;

use infr_core::{
    backend::{Buffer, BufferUsage, Capabilities, Plan},
    error::{Error, Result},
    graph::{Bindings, Graph},
    Backend,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn be(s: impl std::fmt::Display) -> Error {
    Error::Backend(s.to_string())
}

/// Downcast `&dyn Buffer` → `&VkBuffer`.
///
/// # Safety
/// Must only be called with buffers returned by `VulkanBackend::alloc`.
unsafe fn as_vk_buf(b: &dyn Buffer) -> &VkBuffer {
    // Fat pointer (data_ptr, vtable_ptr) → thin data_ptr → &VkBuffer.
    &*(b as *const dyn Buffer as *const () as *const VkBuffer)
}

// ── shared GPU state ──────────────────────────────────────────────────────────

struct VulkanShared {
    // NOTE: field declaration order matters for drop.
    // Rust drops struct fields in *declaration order*.  We keep `allocator`
    // in a `ManuallyDrop` so we can drop it explicitly before calling
    // `destroy_device` in the `Drop` impl.
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    /// Serialises all one-shot command-buffer submissions.
    cmd_pool: Mutex<vk::CommandPool>,
    /// Must be dropped before the device is destroyed.
    allocator: ManuallyDrop<Mutex<Allocator>>,
    caps: Capabilities,
}

// ash Instances/Devices/handles are Send+Sync per the Vulkan spec when
// accessed through our Mutexes.
unsafe impl Send for VulkanShared {}
unsafe impl Sync for VulkanShared {}

impl Drop for VulkanShared {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            // Destroy command pool.
            let pool = *self.cmd_pool.lock().unwrap();
            self.device.destroy_command_pool(pool, None);
            // Drop the allocator *before* destroying the device.
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ── VkBuffer ──────────────────────────────────────────────────────────────────

struct VkBuffer {
    shared: Arc<VulkanShared>,
    buffer: vk::Buffer,
    /// Wrapped in ManuallyDrop so we control the drop order in `Drop`.
    allocation: ManuallyDrop<Allocation>,
    size: usize,
    location: MemoryLocation,
}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        unsafe {
            let alloc = ManuallyDrop::take(&mut self.allocation);
            // Free the gpu-allocator sub-allocation first.
            self.shared.allocator.lock().unwrap().free(alloc).ok();
            // Then destroy the Vulkan buffer object.
            self.shared.device.destroy_buffer(self.buffer, None);
        }
    }
}

unsafe impl Send for VkBuffer {}
unsafe impl Sync for VkBuffer {}

impl Buffer for VkBuffer {
    fn len_bytes(&self) -> usize {
        self.size
    }
}

// ── VkPlan (stub) ─────────────────────────────────────────────────────────────

struct VkPlan;
impl Plan for VkPlan {}

// ── VulkanBackend ─────────────────────────────────────────────────────────────

/// Vulkan device + allocator + pipeline cache.
pub struct VulkanBackend {
    shared: Arc<VulkanShared>,
}

impl VulkanBackend {
    /// Initialize Vulkan: create instance, pick a GPU (prefer discrete), create a logical
    /// device + compute queue with the required extensions/features, set up the allocator.
    pub fn new() -> Result<Self> {
        // ── entry ──────────────────────────────────────────────────────────────
        let entry =
            unsafe { ash::Entry::load() }.map_err(|e| be(format!("ash::Entry::load: {e}")))?;

        // ── instance (Vulkan 1.3) ──────────────────────────────────────────────
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"infr")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"infr-vulkan")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )
        }
        .map_err(|e| be(format!("create_instance: {e}")))?;

        // ── physical device: prefer discrete ──────────────────────────────────
        let pdevices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| be(format!("enumerate_physical_devices: {e}")))?;
        if pdevices.is_empty() {
            return Err(be("no Vulkan physical devices"));
        }
        let physical_device = pdevices
            .iter()
            .copied()
            .find(|&pd| {
                let p = unsafe { instance.get_physical_device_properties(pd) };
                p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .unwrap_or(pdevices[0]);

        // ── compute queue family ───────────────────────────────────────────────
        let qf_props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = qf_props
            .iter()
            .enumerate()
            .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| be("no compute queue family found"))?;

        // ── probe device extensions ────────────────────────────────────────────
        let avail_exts = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map_err(|e| be(format!("enumerate device extensions: {e}")))?;

        let has_ext = |name: &CStr| -> bool {
            avail_exts
                .iter()
                .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) == name })
        };

        let has_coop_matrix = has_ext(c"VK_KHR_cooperative_matrix");
        let has_16bit_storage = has_ext(c"VK_KHR_16bit_storage");
        let has_subgroup_ext = has_ext(c"VK_KHR_shader_subgroup_extended_types");

        // ── probe f16 feature (via VK 1.1 get_physical_device_features2) ──────
        let mut f16_feat = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut feat2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f16_feat);
        unsafe { instance.get_physical_device_features2(physical_device, &mut feat2) };
        let has_f16 = f16_feat.shader_float16 != 0;

        // ── build extension name list (only available ones) ────────────────────
        let mut ext_ptrs: Vec<*const i8> = Vec::new();
        if has_coop_matrix {
            ext_ptrs.push(c"VK_KHR_cooperative_matrix".as_ptr());
        }
        if has_16bit_storage {
            ext_ptrs.push(c"VK_KHR_16bit_storage".as_ptr());
        }
        if has_subgroup_ext {
            ext_ptrs.push(c"VK_KHR_shader_subgroup_extended_types".as_ptr());
        }

        // ── logical device ─────────────────────────────────────────────────────
        let priorities = [1.0f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        // Enable float16 if available; setting it to false is a no-op.
        let mut shader_f16_ci =
            vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(has_f16);

        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .enabled_extension_names(&ext_ptrs)
            .push_next(&mut shader_f16_ci);

        let device = unsafe { instance.create_device(physical_device, &device_ci, None) }
            .map_err(|e| be(format!("create_device: {e}")))?;

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // ── command pool ───────────────────────────────────────────────────────
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| be(format!("create_command_pool: {e}")))?;

        // ── capabilities ───────────────────────────────────────────────────────
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let caps = Capabilities {
            name: device_name,
            f16: has_f16,
            cooperative_matrix: has_coop_matrix,
            max_buffer_bytes: props.limits.max_storage_buffer_range as u64,
            unified_memory: false, // discrete GPU
        };

        // ── gpu-allocator ──────────────────────────────────────────────────────
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| be(format!("gpu_allocator::Allocator::new: {e}")))?;

        Ok(Self {
            shared: Arc::new(VulkanShared {
                _entry: entry,
                instance,
                physical_device,
                device,
                queue,
                queue_family_index,
                cmd_pool: Mutex::new(cmd_pool),
                allocator: ManuallyDrop::new(Mutex::new(allocator)),
                caps,
            }),
        })
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    /// Create a `vk::Buffer` + gpu-allocator sub-allocation of the requested size/location.
    fn make_buf(&self, size: usize, location: MemoryLocation, label: &str) -> Result<VkBuffer> {
        let buf_ci = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.shared.device.create_buffer(&buf_ci, None) }
            .map_err(|e| be(format!("create_buffer: {e}")))?;

        let requirements = unsafe { self.shared.device.get_buffer_memory_requirements(buffer) };

        let allocation = {
            let mut alloc = self.shared.allocator.lock().unwrap();
            alloc
                .allocate(&AllocationCreateDesc {
                    name: label,
                    requirements,
                    location,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| {
                    // Clean up the buffer we created if allocation fails.
                    unsafe { self.shared.device.destroy_buffer(buffer, None) };
                    be(format!("gpu_allocator::allocate: {e}"))
                })?
        };

        unsafe {
            self.shared
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .map_err(|e| be(format!("bind_buffer_memory: {e}")))?;

        Ok(VkBuffer {
            shared: Arc::clone(&self.shared),
            buffer,
            allocation: ManuallyDrop::new(allocation),
            size,
            location,
        })
    }

    /// Record a single command into a one-shot command buffer, submit it to the
    /// compute queue, and block until idle.
    ///
    /// The closure receives the command buffer handle to record into.
    /// All operations are serialised through the `cmd_pool` mutex.
    fn one_shot(&self, f: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let device = &self.shared.device;
        let pool = *self.shared.cmd_pool.lock().unwrap();

        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| be(format!("allocate_command_buffers: {e}")))?[0];

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|e| be(format!("begin_command_buffer: {e}")))?;

        f(cmd);

        unsafe { device.end_command_buffer(cmd) }
            .map_err(|e| be(format!("end_command_buffer: {e}")))?;

        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        unsafe { device.queue_submit(self.shared.queue, &[submit], vk::Fence::null()) }
            .map_err(|e| be(format!("queue_submit: {e}")))?;

        unsafe { device.queue_wait_idle(self.shared.queue) }
            .map_err(|e| be(format!("queue_wait_idle: {e}")))?;

        unsafe { device.free_command_buffers(pool, &cmds) };
        Ok(())
    }
}

// ── Backend impl ──────────────────────────────────────────────────────────────

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn capabilities(&self) -> Capabilities {
        self.shared.caps.clone()
    }

    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        let (location, label) = match usage {
            BufferUsage::Weights => (MemoryLocation::GpuOnly, "weights"),
            BufferUsage::Activations => (MemoryLocation::GpuOnly, "activations"),
            BufferUsage::Staging => (MemoryLocation::CpuToGpu, "staging"),
        };
        let buf = self.make_buf(bytes, location, label)?;
        Ok(Box::new(buf))
    }

    /// Copy `src` (host slice) into `dst` (device buffer).
    ///
    /// If `dst` is host-visible (`CpuToGpu`), writes directly through the
    /// persistent mapped pointer.  Otherwise, creates a temporary staging buffer,
    /// writes there, then submits a `cmd_copy_buffer` to the compute queue.
    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        // Safety: every buffer from this backend is a VkBuffer.
        let vk_dst = unsafe { as_vk_buf(dst) };

        if vk_dst.location == MemoryLocation::CpuToGpu {
            // Direct write through the persistently-mapped pointer.
            let ptr = vk_dst
                .allocation
                .mapped_ptr()
                .ok_or_else(|| be("CpuToGpu buffer is not persistently mapped"))?
                .as_ptr() as *mut u8;
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()) };
        } else {
            // Staging path: CPU → staging → device-local.
            let staging = self.make_buf(src.len(), MemoryLocation::CpuToGpu, "upload_staging")?;
            let stg_ptr = staging
                .allocation
                .mapped_ptr()
                .ok_or_else(|| be("staging buffer is not mapped"))?
                .as_ptr() as *mut u8;
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), stg_ptr, src.len()) };

            let stg_buf = staging.buffer;
            let dst_buf = vk_dst.buffer;
            let size = src.len() as u64;
            // Clone the Arc so the closure is independent of `self`.
            let shared = Arc::clone(&self.shared);
            self.one_shot(move |cmd| {
                let region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                };
                unsafe {
                    shared
                        .device
                        .cmd_copy_buffer(cmd, stg_buf, dst_buf, &[region])
                };
            })?;
            // `staging` is dropped here → frees vk::Buffer + gpu-allocator sub-allocation.
        }
        Ok(())
    }

    /// Copy `src` (device buffer) into `dst` (host slice).
    ///
    /// If `src` is host-visible (`GpuToCpu`), reads directly from the mapped
    /// pointer.  Otherwise, copies via a temporary readback staging buffer.
    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        // Safety: every buffer from this backend is a VkBuffer.
        let vk_src = unsafe { as_vk_buf(src) };

        if vk_src.location == MemoryLocation::GpuToCpu {
            // Direct read from the persistently-mapped pointer.
            let ptr = vk_src
                .allocation
                .mapped_ptr()
                .ok_or_else(|| be("GpuToCpu buffer is not persistently mapped"))?
                .as_ptr() as *const u8;
            unsafe { std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len()) };
        } else {
            // Readback path: device-local → staging → host.
            let staging = self.make_buf(dst.len(), MemoryLocation::GpuToCpu, "download_staging")?;

            let src_buf = vk_src.buffer;
            let stg_buf = staging.buffer;
            let size = dst.len() as u64;
            let shared = Arc::clone(&self.shared);
            self.one_shot(move |cmd| {
                let region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                };
                unsafe {
                    shared
                        .device
                        .cmd_copy_buffer(cmd, src_buf, stg_buf, &[region])
                };
            })?;

            // GPU→staging transfer is complete (queue_wait_idle returned).
            let ptr = staging
                .allocation
                .mapped_ptr()
                .ok_or_else(|| be("readback staging is not mapped"))?
                .as_ptr() as *const u8;
            unsafe { std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len()) };
            // `staging` dropped here.
        }
        Ok(())
    }

    fn compile(&self, _graph: &Graph) -> Result<Box<dyn Plan>> {
        todo!("lower Graph ops to SPIR-V pipelines + record command buffers")
    }

    fn execute(&self, _plan: &dyn Plan, _bindings: &mut Bindings) -> Result<()> {
        todo!("bind buffers, submit command buffer")
    }

    fn sync(&self) -> Result<()> {
        unsafe { self.shared.device.device_wait_idle() }
            .map_err(|e| be(format!("device_wait_idle: {e}")))
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::Backend;

    /// GPU f32 matmul correctness: compares `VulkanBackend::matmul_f32` against a CPU
    /// reference; asserts max relative error < 1e-3.
    ///
    /// Run with: `cargo test -p infr-vulkan -- --ignored --nocapture`
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn test_matmul_f32() {
        let backend = VulkanBackend::new().expect("VulkanBackend::new failed");
        let caps = backend.capabilities();
        println!("device: {}", caps.name);

        let (m, k, n) = (32usize, 32usize, 32usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

        // CPU reference
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c_ref[i * n + j] = sum;
            }
        }

        let c_gpu = backend
            .matmul_f32(&a, &b, m, k, n)
            .expect("matmul_f32 failed");

        let max_abs = c_gpu
            .iter()
            .zip(c_ref.iter())
            .map(|(g, r)| (*g - r).abs())
            .fold(0.0f32, f32::max);
        let max_ref = c_ref.iter().map(|r| r.abs()).fold(0.0f32, f32::max);
        let rel_err = if max_ref > 1e-6 {
            max_abs / max_ref
        } else {
            max_abs
        };

        println!("matmul {m}×{k}×{n}: max_rel_err = {rel_err:.2e}");
        assert!(rel_err < 1e-3, "matmul rel error too large: {rel_err:.2e}");
        println!("matmul GPU test PASS");
    }

    /// End-to-end roundtrip: init → alloc (device-local) → upload → download → assert.
    ///
    /// Marked `#[ignore]` so CI without a GPU passes; run manually with:
    /// ```text
    /// cargo test -p infr-vulkan -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn roundtrip_upload_download() {
        let backend = VulkanBackend::new().expect("VulkanBackend::new failed");

        let caps = backend.capabilities();
        println!("=== Capabilities ===\n{caps:#?}\n");

        const N: usize = 1024;
        // Pattern: bytes 0x00..0xFF repeating.
        let pattern: Vec<u8> = (0..N).map(|i| (i % 256) as u8).collect();

        // Alloc a device-local buffer (exercises the staging copy path).
        let buf = backend
            .alloc(N, BufferUsage::Weights)
            .expect("alloc Weights buffer");

        backend
            .upload(buf.as_ref(), &pattern)
            .expect("upload host→device");

        let mut got = vec![0u8; N];
        backend
            .download(buf.as_ref(), &mut got)
            .expect("download device→host");

        assert_eq!(pattern, got, "roundtrip data mismatch at 1024 bytes");

        backend.sync().expect("sync");

        println!("roundtrip OK — {N} bytes match");
    }
}
