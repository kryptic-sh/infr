//! Vulkan backend (`ash` + SPIR-V). The MVP `Backend` impl.
//!
//! Reference: `~/Projects/llama.cpp/ggml/src/ggml-vulkan/` and its `vulkan-shaders/*.comp`
//! (reuse the tuned quant matmul / dequant / attention shaders). Enable device features
//! `VK_KHR_cooperative_matrix`, `shaderFloat16`, `VK_KHR_16bit_storage`,
//! `VK_KHR_shader_subgroup_extended_types`. See docs/PLAN.md.
#![allow(dead_code)]
// GPU kernel record/dispatch APIs bind many distinct buffers (weights, scales, activations,
// scratch) — wide signatures are inherent here, not a refactor smell.
#![allow(clippy::too_many_arguments)]

mod adapter;
mod expert_pool;
mod gemm;
pub mod linear;
mod matmul;
mod ops;
mod recorder;

pub use expert_pool::ExpertPool;
pub use recorder::{RecordedCmd, Recorder};

/// Shared-memory bytes consumed per query row of a flash-attention prefill tile
/// (`Ss` + `Ps` + `Os` + softmax state, at `BN=64` / `HD=128`). The tile height is chosen so
/// `rows * FLASH_SHARED_PER_ROW <= maxComputeSharedMemorySize`; `use_flash` needs the smallest
/// tile (`BM=32`) to fit. Keep in sync with `attn_flash{,_warp,_partial}.comp`.
pub const FLASH_SHARED_PER_ROW: u32 = 908;
/// Same, for the register-O flash tile (`sfsh` + `Psh` + `pvsh` + state); smallest tile is `BR=64`.
/// Keep in sync with `attn_flash_reg.comp`.
pub const FLASH_REG_SHARED_PER_ROW: u32 = 460;

use std::collections::HashMap;
use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::MemoryLocation;

use infr_core::{
    backend::{Bindings, Buffer, BufferUsage, Capabilities, Plan},
    error::{Error, Result},
    graph::Graph,
    Backend,
};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Terse local shorthand for the shared [`Error::backend`] constructor.
fn be(s: impl std::fmt::Display) -> Error {
    Error::backend(s)
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

/// Device-local VRAM snapshot from [`VulkanBackend::vram`]. `available` is live free bytes when
/// `live` is true (VK_EXT_memory_budget present), otherwise it equals `total` (best-effort).
#[derive(Clone, Copy, Debug)]
pub struct VramInfo {
    pub total: u64,
    pub available: u64,
    pub live: bool,
}

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
    /// VK_EXT_memory_budget enabled → `vram()` can report live free bytes (else total only).
    has_mem_budget: bool,
    /// Pre-reserved bump arena for load-once weights (see `reserve_weights`). `None` until reserved;
    /// weight allocs then sub-allocate from it instead of the gpu-allocator.
    weight_arena: Mutex<Option<WeightArena>>,
    /// Lazily-built, reused compute pipeline for the linear op (see `linear.rs`).
    linear_kernel: std::sync::OnceLock<crate::linear::LinearKernel>,
    /// Generic cache of compute kernels by name (see `ops.rs`).
    kernels: Mutex<HashMap<&'static str, crate::ops::ComputeKernel>>,
    /// Active weight-load progress bar (see [`VulkanBackend::weight_progress`]). Every
    /// `BufferUsage::Weights` allocation advances it, so no model loader can forget to tick it.
    weight_pb: Mutex<Option<indicatif::ProgressBar>>,
}

// ash Instances/Devices/handles are Send+Sync per the Vulkan spec when
// accessed through our Mutexes.
unsafe impl Send for VulkanShared {}
unsafe impl Sync for VulkanShared {}

impl Drop for VulkanShared {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            // Destroy the cached linear kernel (pipeline/layouts/shader/pool) if built.
            if let Some(k) = self.linear_kernel.get() {
                crate::linear::destroy_linear_kernel(&self.device, k);
            }
            if let Ok(map) = self.kernels.lock() {
                for k in map.values() {
                    crate::ops::destroy_compute_kernel(&self.device, k);
                }
            }
            // Destroy command pool.
            let pool = *self.cmd_pool.lock().unwrap();
            self.device.destroy_command_pool(pool, None);
            // Free the weight arena's device memory before destroying the device.
            if let Some(arena) = self.weight_arena.lock().unwrap().as_mut() {
                arena.destroy(&self.device);
            }
            // Drop the allocator *before* destroying the device.
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ── VkBuffer ──────────────────────────────────────────────────────────────────

/// How a `VkBuffer`'s device memory is owned.
enum Backing {
    /// A gpu-allocator sub-allocation — freed back to the allocator on drop (transient buffers,
    /// host-visible staging/readback, and weights when no arena is reserved).
    Pooled(ManuallyDrop<Allocation>),
    /// Bump-allocated from the [`WeightArena`]. The arena block owns the memory; on drop the buffer
    /// only destroys its own handle (the block frees the memory when the arena drops).
    Arena,
}

struct VkBuffer {
    shared: Arc<VulkanShared>,
    buffer: vk::Buffer,
    backing: Backing,
    size: usize,
    location: MemoryLocation,
}

impl VkBuffer {
    /// Persistently-mapped host pointer for host-visible (pooled) buffers; `None` for device-local
    /// or arena buffers (which are never mapped — they're filled via a staging copy).
    fn mapped_ptr(&self) -> Option<*mut u8> {
        match &self.backing {
            Backing::Pooled(a) => a.mapped_ptr().map(|p| p.as_ptr() as *mut u8),
            Backing::Arena => None,
        }
    }
}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        unsafe {
            if let Backing::Pooled(alloc) = &mut self.backing {
                let alloc = ManuallyDrop::take(alloc);
                self.shared.allocator.lock().unwrap().free(alloc).ok();
            }
            // Arena memory belongs to the arena block; only destroy the buffer handle here.
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ── weight arena ────────────────────────────────────────────────────────────────

/// Buffer usage flags for every device buffer (must match across the arena probe and all
/// allocations so their memory-type bits / alignment agree).
const BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);

/// Overflow arena blocks (only allocated if the reserved block underflows the estimate) stay modest
/// so a tiny estimate miss can't waste a whole second model-sized block.
const ARENA_OVERFLOW_BLOCK: u64 = 64 * 1024 * 1024;

/// One large device-local memory block the weight arena bump-allocates from.
struct ArenaBlock {
    memory: vk::DeviceMemory,
    size: u64,
    cursor: u64,
}

/// Pre-reserved VRAM for load-once weights: one big block sized to the model (plus modest overflow
/// blocks if the estimate underflows), bump-allocated since weights are never individually freed.
/// Reserving the whole model up front guarantees it fits contiguously and frees in one shot.
/// MoE-ready: a future expert-streaming mode can hold a second arena/pool and evict experts into it
/// without disturbing the dense arena.
struct WeightArena {
    mem_type: u32,
    blocks: Vec<ArenaBlock>,
}

impl WeightArena {
    /// Bump-allocate `size` bytes at `align` from the newest block, growing with an overflow block
    /// if it won't fit. Returns the device memory + offset to bind a buffer to.
    fn bump(
        &mut self,
        device: &ash::Device,
        size: u64,
        align: u64,
    ) -> Result<(vk::DeviceMemory, u64)> {
        if let Some(b) = self.blocks.last_mut() {
            let off = b.cursor.div_ceil(align) * align;
            if off + size <= b.size {
                b.cursor = off + size;
                return Ok((b.memory, off));
            }
        }
        let bs = size.max(ARENA_OVERFLOW_BLOCK);
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(bs)
                    .memory_type_index(self.mem_type),
                None,
            )
        }
        .map_err(|e| be(format!("arena overflow allocate_memory({bs}): {e}")))?;
        self.blocks.push(ArenaBlock {
            memory,
            size: bs,
            cursor: size,
        });
        Ok((memory, 0))
    }

    /// Free all blocks. Must be called before `destroy_device`.
    unsafe fn destroy(&mut self, device: &ash::Device) {
        for b in self.blocks.drain(..) {
            device.free_memory(b.memory, None);
        }
    }
}

// ── VulkanBackend ─────────────────────────────────────────────────────────────

/// Vulkan device + allocator + pipeline cache.
pub struct VulkanBackend {
    shared: Arc<VulkanShared>,
}

/// RAII scope for a weight-load progress bar (see [`VulkanBackend::weight_progress`]). While alive,
/// `BufferUsage::Weights` allocations advance the bar; on drop it finishes and clears it.
pub struct WeightProgress {
    shared: Arc<VulkanShared>,
}

impl Drop for WeightProgress {
    fn drop(&mut self) {
        if let Some(pb) = self.shared.weight_pb.lock().unwrap().take() {
            pb.finish_and_clear();
        }
    }
}

impl VulkanBackend {
    /// `maxComputeSharedMemorySize` for the active device — the per-workgroup shared-memory budget
    /// the flash-attention tile height is sized against (cheap accessor; avoids cloning caps).
    pub fn max_shared_memory_bytes(&self) -> u32 {
        self.shared.caps.max_shared_memory_bytes
    }

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
        let has_8bit_storage = has_ext(c"VK_KHR_8bit_storage");
        let has_subgroup_ext = has_ext(c"VK_KHR_shader_subgroup_extended_types");
        let has_mem_budget = has_ext(c"VK_EXT_memory_budget");

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
        if has_8bit_storage {
            ext_ptrs.push(c"VK_KHR_8bit_storage".as_ptr());
        }
        if has_subgroup_ext {
            ext_ptrs.push(c"VK_KHR_shader_subgroup_extended_types".as_ptr());
        }
        if has_mem_budget {
            ext_ptrs.push(c"VK_EXT_memory_budget".as_ptr());
        }

        // ── logical device ─────────────────────────────────────────────────────
        let priorities = [1.0f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        // Feature chain — needed for cooperative-matrix kernels:
        //   shaderFloat16 (f16 math), 16-bit storage (f16 SSBOs), Vulkan memory model
        //   (required by coopmat), cooperativeMatrix itself.
        let mut shader_f16_ci = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(has_f16)
            .shader_int8(true);
        let mut storage16_ci = vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(has_16bit_storage);
        let mut storage8_ci = vk::PhysicalDevice8BitStorageFeatures::default()
            .storage_buffer8_bit_access(has_8bit_storage);
        let mut memmodel_ci = vk::PhysicalDeviceVulkanMemoryModelFeatures::default()
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true);
        let mut coopmat_ci =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        // Lets us pin the subgroup size to 32 (RDNA3 coopmat is wave32) for the tiled GEMM.
        let mut sgsize_ci = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
            .subgroup_size_control(true)
            .compute_full_subgroups(true);

        let mut device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .enabled_extension_names(&ext_ptrs)
            .push_next(&mut shader_f16_ci)
            .push_next(&mut storage16_ci)
            .push_next(&mut storage8_ci)
            .push_next(&mut memmodel_ci)
            .push_next(&mut sgsize_ci);
        if has_coop_matrix {
            device_ci = device_ci.push_next(&mut coopmat_ci);
        }

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
            max_shared_memory_bytes: props.limits.max_compute_shared_memory_size,
            unified_memory: false, // discrete GPU
            // The seam adapter records the decode graph once and replays it (params-driven `_dyn`
            // kernels); the runner compiles the eligible qwen3 decode graph once.
            decode_replay: true,
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
                has_mem_budget,
                weight_arena: Mutex::new(None),
                linear_kernel: std::sync::OnceLock::new(),
                kernels: Mutex::new(HashMap::new()),
                weight_pb: Mutex::new(None),
            }),
        })
    }

    /// Begin a "loading weights" progress bar covering `total_bytes` (pass `None` for an
    /// indeterminate byte spinner when the total isn't known up front). Every subsequent
    /// `BufferUsage::Weights` allocation advances it automatically — the ticking lives in `alloc`,
    /// so a model loader cannot forget it; it only has to open the scope once. The returned guard
    /// finishes and clears the bar on drop, so the bar's lifetime is the loader's scope.
    pub fn weight_progress(&self, total_bytes: Option<u64>) -> WeightProgress {
        let pb = infr_core::progress::bar(
            total_bytes,
            "loading weights",
            infr_core::progress::Unit::Bytes,
        );
        *self.shared.weight_pb.lock().unwrap() = Some(pb);
        WeightProgress {
            shared: self.shared.clone(),
        }
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    /// Create a `vk::Buffer` + gpu-allocator sub-allocation of the requested size/location.
    /// Device-local VRAM: total heap size and currently-available bytes. `available` comes from
    /// VK_EXT_memory_budget (live, accounts for other processes + our own allocations) when the
    /// extension is present; otherwise it falls back to the total heap size (best effort).
    pub fn vram(&self) -> VramInfo {
        let s = &self.shared;
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceMemoryProperties2::default();
        if s.has_mem_budget {
            props2 = props2.push_next(&mut budget);
        }
        unsafe {
            s.instance
                .get_physical_device_memory_properties2(s.physical_device, &mut props2)
        };
        let mp = props2.memory_properties;
        let mut total = 0u64;
        let mut available = 0u64;
        for i in 0..mp.memory_heap_count as usize {
            if mp.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            {
                total += mp.memory_heaps[i].size;
                available += if s.has_mem_budget {
                    budget.heap_budget[i]
                } else {
                    mp.memory_heaps[i].size
                };
            }
        }
        VramInfo {
            total,
            available,
            live: s.has_mem_budget,
        }
    }

    /// First device-local memory type compatible with `type_bits` (from a buffer's requirements).
    fn find_memory_type(&self, type_bits: u32, props: vk::MemoryPropertyFlags) -> Option<u32> {
        let mp = unsafe {
            self.shared
                .instance
                .get_physical_device_memory_properties(self.shared.physical_device)
        };
        (0..mp.memory_type_count).find(|&i| {
            (type_bits & (1 << i)) != 0
                && mp.memory_types[i as usize].property_flags.contains(props)
        })
    }

    /// Pre-reserve `total` bytes of device-local VRAM as a bump arena for load-once weights, so the
    /// whole model's weight memory is committed up front (one contiguous block, freed in one shot)
    /// instead of dribbled out per-tensor. Subsequent `BufferUsage::Weights` allocs sub-allocate
    /// from it. Call once after the footprint check, before uploading weights. On failure (e.g. no
    /// contiguous block available) leaves no arena → callers fall back to the per-tensor path.
    pub fn reserve_weights(&self, total: u64) -> Result<()> {
        // Probe a weight-shaped buffer for its memory-type bits + alignment (identical for every
        // weight buffer, since they all share BUFFER_USAGE).
        let probe = unsafe {
            self.shared.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(4096)
                    .usage(BUFFER_USAGE)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|e| be(format!("arena probe buffer: {e}")))?;
        let req = unsafe { self.shared.device.get_buffer_memory_requirements(probe) };
        unsafe { self.shared.device.destroy_buffer(probe, None) };

        let mem_type = self
            .find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .ok_or_else(|| be("no DEVICE_LOCAL memory type for weights"))?;
        let block = total.max(req.alignment).next_multiple_of(req.alignment);
        let memory = unsafe {
            self.shared.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(block)
                    .memory_type_index(mem_type),
                None,
            )
        }
        .map_err(|e| be(format!("reserve_weights {block} bytes: {e}")))?;

        *self.shared.weight_arena.lock().unwrap() = Some(WeightArena {
            mem_type,
            blocks: vec![ArenaBlock {
                memory,
                size: block,
                cursor: 0,
            }],
        });
        Ok(())
    }

    fn make_buf(&self, size: usize, location: MemoryLocation, label: &str) -> Result<VkBuffer> {
        let buf_ci = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(BUFFER_USAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.shared.device.create_buffer(&buf_ci, None) }
            .map_err(|e| be(format!("create_buffer: {e}")))?;

        let requirements = unsafe { self.shared.device.get_buffer_memory_requirements(buffer) };

        // Load-once weights (label "weights") bind into the pre-reserved bump arena when one exists
        // — the whole model's VRAM is reserved up front (see `reserve_weights`). Everything else
        // (transient activations, host-visible staging/readback, and weights with no arena) uses the
        // gpu-allocator below.
        if label == "weights" {
            let mut arena = self.shared.weight_arena.lock().unwrap();
            if let Some(a) = arena.as_mut() {
                match a.bump(
                    &self.shared.device,
                    requirements.size,
                    requirements.alignment,
                ) {
                    Ok((memory, offset)) => {
                        unsafe {
                            self.shared
                                .device
                                .bind_buffer_memory(buffer, memory, offset)
                        }
                        .map_err(|e| {
                            unsafe { self.shared.device.destroy_buffer(buffer, None) };
                            be(format!("arena bind_buffer_memory: {e}"))
                        })?;
                        return Ok(VkBuffer {
                            shared: Arc::clone(&self.shared),
                            buffer,
                            backing: Backing::Arena,
                            size,
                            location,
                        });
                    }
                    Err(e) => {
                        unsafe { self.shared.device.destroy_buffer(buffer, None) };
                        return Err(e);
                    }
                }
            }
        }

        // Large buffers (KV cache, big weights) get a DEDICATED exact-size VkDeviceMemory; otherwise
        // they sub-allocate into gpu-allocator's 256MB blocks and waste the remainder (e.g. 3×67MB
        // KV buffers per block leave ~55MB unused — ~0.7GB across a long-context KV cache). Small/
        // transient buffers stay sub-allocated (cheap, pooled).
        const DEDICATED_MIN: u64 = 32 * 1024 * 1024;
        let scheme = if requirements.size >= DEDICATED_MIN {
            AllocationScheme::DedicatedBuffer(buffer)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };
        let allocation = {
            let mut alloc = self.shared.allocator.lock().unwrap();
            alloc
                .allocate(&AllocationCreateDesc {
                    name: label,
                    requirements,
                    location,
                    linear: true,
                    allocation_scheme: scheme,
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
            backing: Backing::Pooled(ManuallyDrop::new(allocation)),
            size,
            location,
        })
    }

    /// Fill a buffer with the repeated byte `byte` (0x00 = zero-init, 0xFF = poison). Host-visible
    /// buffers are memset through the mapped pointer (no submit); device-local buffers use
    /// `vkCmdFillBuffer` via a one-shot submit. Each `VkBuffer` owns a distinct `vk::Buffer` handle
    /// addressing its region from offset 0 (arena buffers included), so filling `[0, size)` is correct.
    fn fill_buf(&self, buf: &VkBuffer, byte: u8) -> Result<()> {
        if let Some(ptr) = buf.mapped_ptr() {
            unsafe { std::ptr::write_bytes(ptr, byte, buf.size) };
        } else {
            let word = u32::from_ne_bytes([byte; 4]);
            let size = (buf.size / 4 * 4) as u64; // vkCmdFillBuffer requires a 4-byte multiple
            if size > 0 {
                let vkbuf = buf.buffer;
                let shared = Arc::clone(&self.shared);
                self.one_shot(move |cmd| unsafe {
                    shared.device.cmd_fill_buffer(cmd, vkbuf, 0, size, word);
                })?;
            }
        }
        Ok(())
    }

    /// The shared body of `alloc`/`alloc_uninit`: pick the memory location + tick the weight-load
    /// progress bar. Zero/poison filling is applied by the callers.
    fn make_alloc(&self, bytes: usize, usage: BufferUsage) -> Result<VkBuffer> {
        let (location, label) = match usage {
            BufferUsage::Weights => (MemoryLocation::GpuOnly, "weights"),
            BufferUsage::Activations => (MemoryLocation::GpuOnly, "activations"),
            BufferUsage::Staging => (MemoryLocation::CpuToGpu, "staging"),
            BufferUsage::Readback => (MemoryLocation::GpuToCpu, "readback"),
        };
        let buf = self.make_buf(bytes, location, label)?;
        // Advance the weight-load progress bar (if active) — the single funnel every weight upload
        // passes through, so no loader can forget to account for a tensor.
        if matches!(usage, BufferUsage::Weights) {
            if let Some(pb) = self.shared.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
        }
        Ok(buf)
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
        // calloc contract: zero-init so recycled/uninitialized VRAM can't leak into a read-before-write.
        let buf = self.make_alloc(bytes, usage)?;
        self.fill_buf(&buf, 0x00)?;
        Ok(Box::new(buf))
    }

    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        // Opt-out: skip the zero-fill (caller guarantees the full extent is written before any read).
        // Debug builds poison with 0xFF (= NaN as f32) so a misuse surfaces loudly in tests;
        // INFR_POISON_UNINIT=1 forces the poison in release too — for hunting layout-sensitive
        // read-before-write bugs whose output shifts with unrelated code changes.
        let buf = self.make_alloc(bytes, usage)?;
        #[cfg(debug_assertions)]
        self.fill_buf(&buf, 0xFF)?;
        #[cfg(not(debug_assertions))]
        if std::env::var("INFR_POISON_UNINIT").is_ok() {
            self.fill_buf(&buf, 0xFF)?;
        }
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
                .mapped_ptr()
                .ok_or_else(|| be("CpuToGpu buffer is not persistently mapped"))?;
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()) };
        } else {
            // Staging path: CPU → staging → device-local.
            let staging = self.make_buf(src.len(), MemoryLocation::CpuToGpu, "upload_staging")?;
            let stg_ptr = staging
                .mapped_ptr()
                .ok_or_else(|| be("staging buffer is not mapped"))?;
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
                .mapped_ptr()
                .ok_or_else(|| be("GpuToCpu buffer is not persistently mapped"))?
                as *const u8;
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
                .mapped_ptr()
                .ok_or_else(|| be("readback staging is not mapped"))?
                as *const u8;
            unsafe { std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len()) };
            // `staging` dropped here.
        }
        Ok(())
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        adapter::compile(graph)
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        adapter::execute(self, plan, bindings)
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

    /// Weight arena: reserve a small arena, allocate several `Weights` buffers from it (forcing both
    /// the reserved block and at least one overflow block), and verify each round-trips bytes through
    /// the staging copy path — proving arena buffers bind to valid, distinct memory regions.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn weight_arena_roundtrip() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        // Reserve only 1 MB so the later allocations spill into an overflow block.
        be.reserve_weights(1024 * 1024).expect("reserve_weights");
        let sizes = [4096usize, 256 * 1024, 4 * 1024 * 1024]; // last forces an overflow block
        let mut bufs = Vec::new();
        for (bi, &sz) in sizes.iter().enumerate() {
            let data: Vec<u8> = (0..sz)
                .map(|i| (i as u8).wrapping_add(bi as u8 * 31))
                .collect();
            let buf = be
                .alloc(sz, BufferUsage::Weights)
                .expect("arena weight alloc");
            be.upload(buf.as_ref(), &data).expect("upload");
            let mut back = vec![0u8; sz];
            be.download(buf.as_ref(), &mut back).expect("download");
            assert_eq!(
                back, data,
                "arena buffer {bi} (size {sz}) round-trip mismatch"
            );
            bufs.push(buf);
        }
        // All three buffers coexist (distinct memory) — re-download the first and re-check.
        let mut back0 = vec![0u8; sizes[0]];
        be.download(bufs[0].as_ref(), &mut back0)
            .expect("re-download");
        assert_eq!(
            back0[1], 1u8,
            "first arena buffer corrupted by later allocs"
        );
    }

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

// Qwen3-Next SSM kernels: the GPU conv1d+SiLU and gated-DeltaNet recurrence must match the CPU
// reference. Self-skip without a GPU (so CI passes, runs locally with a device).
#[cfg(test)]
mod ssm_tests {
    use super::*;
    use infr_core::backend::{Buffer, BufferUsage};

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    fn softplus(x: f32) -> f32 {
        x.max(0.0) + (-x.abs()).exp().ln_1p()
    }
    fn det(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.137 + seed).sin()).collect()
    }
    fn dev(be: &VulkanBackend, data: &[f32]) -> Box<dyn Buffer> {
        let b = be
            .alloc((data.len() * 4).max(4), BufferUsage::Activations)
            .unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(data)).unwrap();
        b
    }
    fn read(be: &VulkanBackend, buf: &dyn Buffer, n: usize) -> Vec<f32> {
        let mut bytes = vec![0u8; n * 4];
        be.download(buf, &mut bytes).unwrap();
        bytemuck::cast_slice(&bytes).to_vec()
    }
    fn maxerr(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn softcap_matches_cpu() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let (n, cap) = (100usize, 30.0f32);
        let x = det(n, 0.5);
        let out_cpu: Vec<f32> = x.iter().map(|&v| cap * (v / cap).tanh()).collect();
        let xb = dev(&be, &x);
        let ob = be.alloc(n * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.softcap(xb.as_ref(), ob.as_ref(), cap, n);
        rec.finish().unwrap();
        let out_gpu = read(&be, ob.as_ref(), n);
        let e = maxerr(&out_cpu, &out_gpu);
        assert!(e < 1e-4, "softcap err {e}");
    }

    #[test]
    fn deltanet_matches_cpu() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let (nv, nk, kd, vd) = (4usize, 2usize, 8usize, 8usize);
        let eps = 1e-6f32;
        let q = det(nk * kd, 0.1);
        let k = det(nk * kd, 0.7);
        let v = det(nv * vd, 1.3);
        let blog = det(nv, 2.0);
        let alpha = det(nv, 0.5);
        let acoef: Vec<f32> = (0..nv).map(|i| -(0.2 + 0.1 * i as f32)).collect();
        let dtbias = det(nv, -0.3);
        let state0 = det(nv * kd * vd, 0.05);

        // CPU reference (mirrors shaders/deltanet.comp + the qwen35 CPU mixer).
        let qscale = 1.0 / (kd as f32).sqrt();
        let mut s = state0.clone();
        let mut out_cpu = vec![0f32; nv * vd];
        for h in 0..nv {
            let khid = h % nk;
            let mut qh = q[khid * kd..khid * kd + kd].to_vec();
            let mut kh = k[khid * kd..khid * kd + kd].to_vec();
            let qn = (qh.iter().map(|x| x * x).sum::<f32>() + eps).sqrt();
            let kn = (kh.iter().map(|x| x * x).sum::<f32>() + eps).sqrt();
            for x in qh.iter_mut() {
                *x = *x / qn * qscale;
            }
            for x in kh.iter_mut() {
                *x /= kn;
            }
            let beta = sigmoid(blog[h]);
            let decay = (acoef[h] * softplus(alpha[h] + dtbias[h])).exp();
            let sb = h * kd * vd;
            for d in 0..vd {
                let mut kvv = 0.0;
                for kk in 0..kd {
                    let sv = s[sb + kk * vd + d] * decay;
                    s[sb + kk * vd + d] = sv;
                    kvv += kh[kk] * sv;
                }
                let delta = (v[h * vd + d] - kvv) * beta;
                let mut o = 0.0;
                for kk in 0..kd {
                    let sv = s[sb + kk * vd + d] + kh[kk] * delta;
                    s[sb + kk * vd + d] = sv;
                    o += qh[kk] * sv;
                }
                out_cpu[h * vd + d] = o;
            }
        }

        let (qb, kb, vb) = (dev(&be, &q), dev(&be, &k), dev(&be, &v));
        let (bb, ab) = (dev(&be, &blog), dev(&be, &alpha));
        let (acb, dtb) = (dev(&be, &acoef), dev(&be, &dtbias));
        let sbuf = dev(&be, &state0);
        let ob = be.alloc(nv * vd * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.deltanet(
            qb.as_ref(),
            kb.as_ref(),
            vb.as_ref(),
            bb.as_ref(),
            ab.as_ref(),
            acb.as_ref(),
            dtb.as_ref(),
            sbuf.as_ref(),
            ob.as_ref(),
            1, // rows: single-token bespoke path
            nv,
            nk,
            kd,
            vd,
            eps,
        );
        rec.finish().unwrap();
        let out_gpu = read(&be, ob.as_ref(), nv * vd);
        let s_gpu = read(&be, sbuf.as_ref(), nv * kd * vd);
        assert!(
            maxerr(&out_cpu, &out_gpu) < 1e-4,
            "deltanet out err {}",
            maxerr(&out_cpu, &out_gpu)
        );
        assert!(
            maxerr(&s, &s_gpu) < 1e-4,
            "deltanet state err {}",
            maxerr(&s, &s_gpu)
        );
    }

    #[test]
    fn conv1d_silu_matches_cpu() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let (cc, kconv) = (40usize, 4usize);
        let qkv = det(cc, 0.2);
        let w = det(cc * kconv, 1.1);
        let state0 = det((kconv - 1) * cc, 0.3);

        // CPU reference (mirrors shaders/conv1d_silu.comp).
        let mut st = state0.clone();
        let mut out_cpu = vec![0f32; cc];
        let km1 = kconv - 1;
        for ch in 0..cc {
            let mut acc = 0.0;
            for k in 0..km1 {
                acc += st[k * cc + ch] * w[ch * kconv + k];
            }
            acc += qkv[ch] * w[ch * kconv + km1];
            out_cpu[ch] = acc * sigmoid(acc);
            for k in 0..km1 - 1 {
                st[k * cc + ch] = st[(k + 1) * cc + ch];
            }
            st[(km1 - 1) * cc + ch] = qkv[ch];
        }

        let xb = dev(&be, &qkv);
        let wb = dev(&be, &w);
        let sbuf = dev(&be, &state0);
        let ob = be.alloc(cc * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.conv1d_silu(
            xb.as_ref(),
            wb.as_ref(),
            sbuf.as_ref(),
            ob.as_ref(),
            1, // rows: single-token bespoke path
            cc,
            kconv,
        );
        rec.finish().unwrap();
        let out_gpu = read(&be, ob.as_ref(), cc);
        let s_gpu = read(&be, sbuf.as_ref(), (kconv - 1) * cc);
        assert!(
            maxerr(&out_cpu, &out_gpu) < 1e-5,
            "conv out err {}",
            maxerr(&out_cpu, &out_gpu)
        );
        assert!(
            maxerr(&st, &s_gpu) < 1e-5,
            "conv state err {}",
            maxerr(&st, &s_gpu)
        );
    }
}
