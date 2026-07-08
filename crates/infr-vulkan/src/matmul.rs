//! Naive f32 matrix multiplication via a WGSL compute shader compiled to SPIR-V at runtime.
//!
//! Shader strategy: GLSL source → SPIR-V at build time (glslc, see build.rs), cached via
//! `OnceLock`). No native `glslangValidator`/`shaderc` dependency.
//!
//! Kernel: C[M,N] = A[M,K] × B[K,N].  One invocation per output element; workgroup 16×16×1.
//! Push constants carry M, N, K as `u32` (12 bytes total).

use std::sync::OnceLock;
use std::time::Instant;

use ash::vk;

use infr_core::{backend::BufferUsage, error::Result, Backend};

use super::{as_vk_buf, be, VulkanBackend};

// ── SPIR-V compilation (once) ─────────────────────────────────────────────────

static MATMUL_SPV: OnceLock<Vec<u32>> = OnceLock::new();

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn matmul_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32.spv"));
    MATMUL_SPV.get_or_init(|| {
        BYTES
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    })
}

// ── workgroup tile size ───────────────────────────────────────────────────────

const WG: u32 = 16;

// ── VulkanBackend::matmul_f32 ─────────────────────────────────────────────────

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanBackend {
    /// GPU f32 matmul: C[m×n] = A[m×k] × B[k×n].
    ///
    /// Compiles the WGSL shader to SPIR-V once (cached), then:
    /// 1. Creates a compute pipeline + descriptor set for this call.
    /// 2. Uploads A and B to device-local storage buffers (via staging).
    /// 3. Dispatches with `ceil(m/16) × ceil(n/16) × 1` workgroups.
    /// 4. Downloads C to host.
    /// 5. Destroys all transient Vulkan objects (pool, pipeline, layouts, shader module).
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        assert_eq!(a.len(), m * k, "A must have m×k = {} elements", m * k);
        assert_eq!(b.len(), k * n, "B must have k×n = {} elements", k * n);

        let device = &self.shared.device;
        let spv = matmul_spv();
        let t0 = Instant::now();

        // ── shader module ──────────────────────────────────────────────────────
        let shader_module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spv), None)
        }
        .map_err(|e| be(format!("create_shader_module: {e}")))?;

        // ── descriptor set layout (3 storage buffers: A, B, C) ────────────────
        let ds_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let ds_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&ds_bindings),
                None,
            )
        }
        .map_err(|e| be(format!("create_descriptor_set_layout: {e}")))?;

        // ── pipeline layout (push constants: M, N, K as u32 = 12 bytes) ───────
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(12); // 3 × u32
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&ds_layout))
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
                None,
            )
        }
        .map_err(|e| be(format!("create_pipeline_layout: {e}")))?;

        // ── compute pipeline ───────────────────────────────────────────────────
        let entry_name = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_name);
        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    self.shared.pipeline_cache, // disk-persisted device cache (see pcache.rs)
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(pipeline_layout)],
                    None,
                )
                .map_err(|(_, e)| be(format!("create_compute_pipelines: {e}")))?[0]
        };
        // A driver can return VK_SUCCESS with a VK_NULL_HANDLE pipeline (some ICDs use this to
        // signal a per-pipeline compile failure inside an otherwise-successful batch call). Using
        // that handle in a later vkCmdBindPipeline/vkCmdDispatch is how the segfault manifested —
        // check here and fail with a clear error instead of dereferencing garbage downstream.
        if pipeline == vk::Pipeline::null() {
            return Err(be(
                "create_compute_pipelines: driver returned VK_SUCCESS with a null pipeline handle \
                 (matmul_f32)",
            ));
        }

        // ── descriptor pool + set ──────────────────────────────────────────────
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        }];
        let desc_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|e| be(format!("create_descriptor_pool: {e}")))?;
        let desc_set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(desc_pool)
                        .set_layouts(std::slice::from_ref(&ds_layout)),
                )
                .map_err(|e| be(format!("allocate_descriptor_sets: {e}")))?[0]
        };

        // ── upload A and B to device-local buffers ─────────────────────────────
        let a_bytes: &[u8] = bytemuck::cast_slice(a);
        let b_bytes: &[u8] = bytemuck::cast_slice(b);

        let buf_a = self.alloc(a_bytes.len(), BufferUsage::Weights)?;
        let buf_b = self.alloc(b_bytes.len(), BufferUsage::Weights)?;
        let buf_c = self.alloc(m * n * 4, BufferUsage::Activations)?;

        self.upload(buf_a.as_ref(), a_bytes)?;
        self.upload(buf_b.as_ref(), b_bytes)?;

        // Raw Vulkan buffer handles (needed for descriptor writes and barrier).
        let vk_a = unsafe { as_vk_buf(buf_a.as_ref()) }.buffer;
        let vk_b = unsafe { as_vk_buf(buf_b.as_ref()) }.buffer;
        let vk_c = unsafe { as_vk_buf(buf_c.as_ref()) }.buffer;

        // ── update descriptor set ──────────────────────────────────────────────
        let buf_infos = [
            vk::DescriptorBufferInfo {
                buffer: vk_a,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: vk_b,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: vk_c,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
        ];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_infos[0..1]),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_infos[1..2]),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_infos[2..3]),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        // ── push constant bytes (M, N, K as little-endian u32) ────────────────
        let push_bytes: [u8; 12] = {
            let mut b = [0u8; 12];
            b[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
            b[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
            b[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
            b
        };

        // ── record + dispatch ──────────────────────────────────────────────────
        let groups_x = (m as u32).div_ceil(WG);
        let groups_y = (n as u32).div_ceil(WG);

        // Clone Arc so the closure owns shared state independently.
        let shared = std::sync::Arc::clone(&self.shared);
        self.one_shot(move |cmd| unsafe {
            // Pipeline barrier: ensure transfer writes to A and B are visible to
            // the compute shader.  (queue_wait_idle in one_shot already serialises
            // the submissions, but we add a correct barrier to be explicit.)
            let buf_barriers = [
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(vk_a)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(vk_b)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
            ];
            shared.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &buf_barriers,
                &[],
            );

            shared
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            shared.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            shared.device.cmd_push_constants(
                cmd,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push_bytes,
            );
            shared.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
        })?;

        let dispatch_ms = t0.elapsed().as_millis();

        // ── download C ─────────────────────────────────────────────────────────
        let mut c_bytes = vec![0u8; m * n * 4];
        self.download(buf_c.as_ref(), &mut c_bytes)?;
        let c: Vec<f32> = bytemuck::cast_slice(&c_bytes).to_vec();

        let total_ms = t0.elapsed().as_millis();
        println!("[matmul_f32] {m}×{k}×{n}: dispatch={dispatch_ms}ms total={total_ms}ms");

        // ── free transient Vulkan objects ──────────────────────────────────────
        // buf_a / buf_b / buf_c drop here (VkBuffer + gpu-allocator sub-alloc freed).
        drop((buf_a, buf_b, buf_c));
        unsafe {
            device.destroy_descriptor_pool(desc_pool, None); // frees desc_set implicitly
            device.destroy_pipeline(pipeline, None);
            device.destroy_pipeline_layout(pipeline_layout, None);
            device.destroy_descriptor_set_layout(ds_layout, None);
            device.destroy_shader_module(shader_module, None);
        }

        Ok(c)
    }
}
