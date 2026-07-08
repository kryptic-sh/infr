//! Persistent-weight linear layer: `y = W · x` where `W` is stored `[out, in]` row-major
//! (the GGUF layout: data index `o*in + i`). The weight buffer is uploaded once
//! (`upload_weight`) and reused; the compute pipeline is built once (cached in
//! `VulkanShared.linear_kernel`) and reused across all calls — only the (small) activation
//! buffers are created per call.
//!
//! Build-compiled GLSL → SPIR-V (see build.rs / shaders/).

use std::sync::OnceLock;

use ash::vk;

use infr_core::{
    backend::{Buffer, BufferUsage},
    error::Result,
    Backend,
};

use super::{as_vk_buf, be, VulkanBackend};

/// Unified quant dequant GEMV with fused residual add: `y = residual + x·Wᵀ`.
// ─── Native-block dequant GEMV shaders (Phase 0-2) ─────────────────────────
//
// Each shader reads raw GGUF block bytes (uploaded padded to a u32-multiple)
// from `w_buf: array<u32>` and dequantizes elements in-shader. The outer GEMV
// cooperative-over-K structure matches LINEAR_F16_WGSL: one workgroup per
// output element, 64 threads stride K, tree-reduce.
//
/// Return the static kernel name for a native-block GEMV (Phase 0-2).
/// Kernel cache name for the id-indexed native GEMV (one per affine quant format); `None` for
/// formats without an id variant.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn native_id_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Q8_0 => "native_id_q8_0",
        Q4_0 => "native_id_q4_0",
        Q4_1 => "native_id_q4_1",
        Q5_0 => "native_id_q5_0",
        Q5_1 => "native_id_q5_1",
        Q2K => "native_id_q2k",
        Q3K => "native_id_q3k",
        Q4K => "native_id_q4k",
        Q5K => "native_id_q5k",
        Q6K => "native_id_q6k",
        _ => return None,
    })
}

/// Kernel cache name for the multi-slot id-indexed native GEMV; `None` for formats without it.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn native_idm_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Q8_0 => "native_idm_q8_0",
        Q4_0 => "native_idm_q4_0",
        Q4_1 => "native_idm_q4_1",
        Q5_0 => "native_idm_q5_0",
        Q5_1 => "native_idm_q5_1",
        Q2K => "native_idm_q2k",
        Q3K => "native_idm_q3k",
        Q4K => "native_idm_q4k",
        Q5K => "native_idm_q5k",
        Q6K => "native_idm_q6k",
        _ => return None,
    })
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn native_kernel_name(dtype: infr_core::DType, residual: bool) -> &'static str {
    use infr_core::DType::*;
    match (dtype, residual) {
        (Q8_0, false) => "native_q8_0",
        (Q8_0, true) => "native_q8_0_res",
        (Bf16, false) => "native_bf16",
        (Bf16, true) => "native_bf16_res",
        (Q4_0, false) => "native_q4_0",
        (Q4_0, true) => "native_q4_0_res",
        (Q4_1, false) => "native_q4_1",
        (Q4_1, true) => "native_q4_1_res",
        (Q5_0, false) => "native_q5_0",
        (Q5_0, true) => "native_q5_0_res",
        (Q5_1, false) => "native_q5_1",
        (Q5_1, true) => "native_q5_1_res",
        (Q2K, false) => "native_q2k",
        (Q2K, true) => "native_q2k_res",
        (Q3K, false) => "native_q3k",
        (Q3K, true) => "native_q3k_res",
        (Q4K, false) => "native_q4k",
        (Q4K, true) => "native_q4k_res",
        (Q5K, false) => "native_q5k",
        (Q5K, true) => "native_q5k_res",
        (Q6K, false) => "native_q6k",
        (Q6K, true) => "native_q6k_res",
        (Iq4Nl, false) => "native_iq4nl",
        (Iq4Nl, true) => "native_iq4nl_res",
        (Iq4Xs, false) => "native_iq4xs",
        (Iq4Xs, true) => "native_iq4xs_res",
        (Mxfp4, false) => "native_mxfp4",
        (Mxfp4, true) => "native_mxfp4_res",
        (Nvfp4, false) => "native_nvfp4",
        (Nvfp4, true) => "native_nvfp4_res",
        (Tq1_0, false) => "native_tq1_0",
        (Tq1_0, true) => "native_tq1_0_res",
        (Tq2_0, false) => "native_tq2_0",
        (Tq2_0, true) => "native_tq2_0_res",
        (Iq2Xxs, false) => "native_iq2xxs",
        (Iq2Xxs, true) => "native_iq2xxs_res",
        (Iq2Xs, false) => "native_iq2xs",
        (Iq2Xs, true) => "native_iq2xs_res",
        (Iq2S, false) => "native_iq2s",
        (Iq2S, true) => "native_iq2s_res",
        (Iq3Xxs, false) => "native_iq3xxs",
        (Iq3Xxs, true) => "native_iq3xxs_res",
        (Iq3S, false) => "native_iq3s",
        (Iq3S, true) => "native_iq3s_res",
        (Iq1S, false) => "native_iq1s",
        (Iq1S, true) => "native_iq1s_res",
        (Iq1M, false) => "native_iq1m",
        (Iq1M, true) => "native_iq1m_res",
        _ => panic!("no native GEMV for {:?}", dtype),
    }
}

/// Kernel-cache key for the native-block prefill GEMM (one coopmat pipeline per quant format).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn native_gemm_kernel_name(dtype: infr_core::DType) -> &'static str {
    use infr_core::DType::*;
    match dtype {
        Q8_0 => "native_gemm_q8_0",
        Bf16 => "native_gemm_bf16",
        Q4_0 => "native_gemm_q4_0",
        Q4_1 => "native_gemm_q4_1",
        Q5_0 => "native_gemm_q5_0",
        Q5_1 => "native_gemm_q5_1",
        Q2K => "native_gemm_q2k",
        Q3K => "native_gemm_q3k",
        Q4K => "native_gemm_q4k",
        Q5K => "native_gemm_q5k",
        Q6K => "native_gemm_q6k",
        Iq4Nl => "native_gemm_iq4nl",
        Iq4Xs => "native_gemm_iq4xs",
        Mxfp4 => "native_gemm_mxfp4",
        Nvfp4 => "native_gemm_nvfp4",
        Tq1_0 => "native_gemm_tq1_0",
        Tq2_0 => "native_gemm_tq2_0",
        Iq2Xxs => "native_gemm_iq2xxs",
        Iq2Xs => "native_gemm_iq2xs",
        Iq2S => "native_gemm_iq2s",
        Iq3Xxs => "native_gemm_iq3xxs",
        Iq3S => "native_gemm_iq3s",
        Iq1S => "native_gemm_iq1s",
        Iq1M => "native_gemm_iq1m",
        _ => panic!("no native GEMM for {:?}", dtype),
    }
}

/// True if `dtype` has the full dense native-block pipeline — a decode GEMV (`native_*`, see
/// [`native_kernel_name`]) AND a prefill coopmat GEMM (`native_gemm_*`, see
/// [`native_gemm_kernel_name`]). When true, the weight can be uploaded as raw GGUF block bytes and
/// run on the GPU with in-shader dequant — no host dequant → f16. Covers every quant format
/// (affine k-quants, legacy round, codebook i-quants, fp4, ternary, and grid i-quants). Float types
/// (F16/F32/BF16) are not quants and stay on the plain f16 GEMV.
///
/// The MoE *stacked/id-indexed* path (`native_id_*`/`native_idm_*`) is narrower — affine only; use
/// [`native_id_kernel_name`] for that.
/// Formats the `embed_gather` kernel family covers (`Op::EmbedGather` — see
/// `gemm::embed_gather_build_spv`). The runner gates the token-ids input path on this.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn embed_gather_supported(dtype: infr_core::DType) -> bool {
    crate::gemm::embed_gather_build_spv(dtype).is_some()
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn native_dense_supported(dtype: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(
        dtype,
        Bf16 | Q8_0
            | Q4_0
            | Q4_1
            | Q5_0
            | Q5_1
            | Q2K
            | Q3K
            | Q4K
            | Q5K
            | Q6K
            | Iq4Nl
            | Iq4Xs
            | Mxfp4
            | Nvfp4
            | Tq1_0
            | Tq2_0
            | Iq2Xxs
            | Iq2Xs
            | Iq2S
            | Iq3Xxs
            | Iq3S
            | Iq1S
            | Iq1M
    )
}

/// Pad raw GGUF block bytes to the next multiple of 4 for upload as `array<u32>`.
/// Appends zero bytes; the final u32 word's padding bytes are never read (they
/// contain only out-of-block data which the shader never accesses for valid g).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn pad_to_u32_align(bytes: &[u8]) -> Vec<u8> {
    let padded = (bytes.len() + 3) & !3;
    let mut v = bytes.to_vec();
    v.resize(padded, 0u8);
    v
}

static LINEAR_SPV: OnceLock<Vec<u32>> = OnceLock::new();

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn linear_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_f32.spv"));
    LINEAR_SPV.get_or_init(|| {
        BYTES
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    })
}

/// Cached, reusable compute objects for the linear kernel (built once per device).
pub(crate) struct LinearKernel {
    pub shader: vk::ShaderModule,
    pub ds_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub desc_pool: vk::DescriptorPool,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn create_linear_kernel(
    device: &ash::Device,
    pcache: vk::PipelineCache,
) -> LinearKernel {
    let spv = linear_spv();
    let shader = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spv), None)
    }
    .expect("create linear shader module");

    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let ds_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .expect("create linear ds layout");

    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(12);
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&ds_layout))
                .push_constant_ranges(std::slice::from_ref(&push_range)),
            None,
        )
    }
    .expect("create linear pipeline layout");

    let entry = c"main";
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(entry);
    let pipeline = unsafe {
        device.create_compute_pipelines(
            pcache, // disk-persisted device cache (see pcache.rs)
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout)],
            None,
        )
    }
    .unwrap_or_else(|(_, e)| panic!("create_compute_pipelines failed for linear kernel: {e}"))[0];
    // See ops.rs::make_compute_kernel for why this is checked explicitly: a driver can return
    // VK_SUCCESS with a null pipeline handle, and using it later is the actual crash.
    assert!(
        pipeline != vk::Pipeline::null(),
        "create_compute_pipelines returned VK_SUCCESS with a null pipeline handle for the linear \
         kernel"
    );

    // Pool holds one set; we reset + reallocate it each call (single-stream gen).
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
    .expect("create linear desc pool");

    LinearKernel {
        shader,
        ds_layout,
        pipeline_layout,
        pipeline,
        desc_pool,
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn destroy_linear_kernel(device: &ash::Device, k: &LinearKernel) {
    unsafe {
        device.destroy_descriptor_pool(k.desc_pool, None);
        device.destroy_pipeline(k.pipeline, None);
        device.destroy_pipeline_layout(k.pipeline_layout, None);
        device.destroy_descriptor_set_layout(k.ds_layout, None);
        device.destroy_shader_module(k.shader, None);
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanBackend {
    fn linear_kernel(&self) -> &LinearKernel {
        let first = self.shared.linear_kernel.get().is_none();
        let k = self
            .shared
            .linear_kernel
            .get_or_init(|| create_linear_kernel(&self.shared.device, self.shared.pipeline_cache));
        if first {
            self.shared.persist_pipeline_cache(); // new pipeline -> debounced disk save
        }
        k
    }

    /// Upload an `[out, in]` f32 weight to a persistent device buffer.
    pub fn upload_weight(&self, data: &[f32]) -> Result<Box<dyn Buffer>> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let buf = self.alloc(bytes.len(), BufferUsage::Weights)?;
        self.upload(buf.as_ref(), bytes)?;
        Ok(buf)
    }

    /// Upload an `[out, in]` weight as f16 (halves device bandwidth for the GEMV/matmul kernels
    /// that read weights). Source stays f32; converted on the host.
    pub fn upload_weight_f16(&self, data: &[f32]) -> Result<Box<dyn Buffer>> {
        let f16: Vec<u16> = data
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        self.upload_weight_bytes(bytemuck::cast_slice(&f16))
    }

    /// Upload an `[out, in]` weight as bf16 (truncate-round of f32; bf16 is the top 16 bits of f32).
    /// Read back losslessly to f32 in-shader by `LINEAR_BF16_WGSL`. Preserves f32's exponent range
    /// (unlike f16), so it's the correct GPU storage for bf16-source tensors that would overflow f16.
    pub fn upload_weight_bf16(&self, data: &[f32]) -> Result<Box<dyn Buffer>> {
        let bf16: Vec<u16> = data
            .iter()
            .map(|&x| {
                // round-to-nearest-even on the f32→bf16 truncation
                let bits = x.to_bits();
                let round = 0x7fffu32 + ((bits >> 16) & 1);
                ((bits.wrapping_add(round)) >> 16) as u16
            })
            .collect();
        self.upload_weight_bytes(bytemuck::cast_slice(&bf16))
    }

    /// Upload raw weight bytes (already in the target dtype) to a persistent device buffer.
    /// Use for f16 GGUF tensors to skip the f16→f32→f16 round-trip.
    pub fn upload_weight_bytes(&self, bytes: &[u8]) -> Result<Box<dyn Buffer>> {
        let buf = self.alloc(bytes.len(), BufferUsage::Weights)?;
        self.upload(buf.as_ref(), bytes)?;
        Ok(buf)
    }

    /// Compute `y[rows, out] = x[rows, in] · Wᵀ` where `w_buf` holds `W[out, in]`.
    /// Reuses the cached pipeline; only the per-call x/y buffers + descriptor set are fresh.
    pub fn linear(
        &self,
        w_buf: &dyn Buffer,
        x: &[f32],
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<Vec<f32>> {
        assert_eq!(x.len(), rows * in_f, "x must be rows*in");
        let device = self.shared.device.clone();
        let k = self.linear_kernel();

        // fresh descriptor set from the cached pool
        unsafe {
            device
                .reset_descriptor_pool(k.desc_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| be(format!("reset_descriptor_pool: {e}")))?;
        }
        let desc_set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(k.desc_pool)
                        .set_layouts(std::slice::from_ref(&k.ds_layout)),
                )
                .map_err(|e| be(format!("allocate_descriptor_sets: {e}")))?[0]
        };

        // Host-visible activation buffers: upload/download become direct memcpy (no extra
        // submit+wait), leaving the dispatch as the only GPU round-trip in this call.
        let x_bytes: &[u8] = bytemuck::cast_slice(x);
        let buf_x = self.alloc(x_bytes.len(), BufferUsage::Staging)?;
        let buf_y = self.alloc(rows * out_f * 4, BufferUsage::Readback)?;
        self.upload(buf_x.as_ref(), x_bytes)?;

        let vk_w = unsafe { as_vk_buf(w_buf) }.buffer;
        let vk_x = unsafe { as_vk_buf(buf_x.as_ref()) }.buffer;
        let vk_y = unsafe { as_vk_buf(buf_y.as_ref()) }.buffer;

        let infos = [
            vk::DescriptorBufferInfo {
                buffer: vk_w,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: vk_x,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: vk_y,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
        ];
        let writes: Vec<vk::WriteDescriptorSet> = (0..3)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i..i + 1])
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());

        let groups = ((rows * out_f) as u32).div_ceil(64);
        let shared = std::sync::Arc::clone(&self.shared);
        let (pipeline, pipeline_layout) = (k.pipeline, k.pipeline_layout);
        self.one_shot(move |cmd| unsafe {
            let barriers = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(vk_x)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            shared.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
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
                &push,
            );
            shared.device.cmd_dispatch(cmd, groups, 1, 1);
        })?;

        let mut y_bytes = vec![0u8; rows * out_f * 4];
        self.download(buf_y.as_ref(), &mut y_bytes)?;
        Ok(bytemuck::cast_slice(&y_bytes).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn linear_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (rows, in_f, out_f) = (3usize, 5usize, 4usize);
        let w: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32) * 0.01).collect();
        let x: Vec<f32> = (0..rows * in_f).map(|i| (i as f32) * 0.02).collect();
        let wbuf = be.upload_weight(&w).unwrap();
        // run twice to exercise the cached pipeline path
        let _ = be.linear(wbuf.as_ref(), &x, rows, in_f, out_f).unwrap();
        let y = be.linear(wbuf.as_ref(), &x, rows, in_f, out_f).unwrap();
        let mut want = vec![0.0f32; rows * out_f];
        for r in 0..rows {
            for o in 0..out_f {
                let mut acc = 0.0;
                for i in 0..in_f {
                    acc += x[r * in_f + i] * w[o * in_f + i];
                }
                want[r * out_f + o] = acc;
            }
        }
        for (g, w) in y.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-3, "{g} vs {w}");
        }
    }

    // CPU reference GEMV for the f16/bf16 eager-path tests (odd in_f exercises bf16 packing).
    fn cpu_gemv(w: &[f32], x: &[f32], rows: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * out_f];
        for r in 0..rows {
            for o in 0..out_f {
                let mut acc = 0.0;
                for i in 0..in_f {
                    acc += x[r * in_f + i] * w[o * in_f + i];
                }
                y[r * out_f + o] = acc;
            }
        }
        y
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn linear_f16_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (rows, in_f, out_f) = (2usize, 70usize, 5usize);
        let w: Vec<f32> = (0..out_f * in_f)
            .map(|i| (i as f32 % 9.0) * 0.05 - 0.2)
            .collect();
        let x: Vec<f32> = (0..rows * in_f).map(|i| (i as f32 % 7.0) * 0.03).collect();
        let wbuf = be.upload_weight_f16(&w).unwrap();
        let _ = be.linear_f16(wbuf.as_ref(), &x, rows, in_f, out_f).unwrap();
        let y = be.linear_f16(wbuf.as_ref(), &x, rows, in_f, out_f).unwrap();
        for (g, c) in y.iter().zip(cpu_gemv(&w, &x, rows, in_f, out_f).iter()) {
            assert!((g - c).abs() < 1e-2, "{g} vs {c}");
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn linear_bf16_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        // in_f odd → rows are NOT u32-aligned in the packed bf16 stream (exercises global addressing)
        let (rows, in_f, out_f) = (3usize, 65usize, 4usize);
        let w: Vec<f32> = (0..out_f * in_f)
            .map(|i| (i as f32 % 11.0) * 0.04 - 0.2)
            .collect();
        let x: Vec<f32> = (0..rows * in_f).map(|i| (i as f32 % 5.0) * 0.06).collect();
        let wbuf = be.upload_weight_bf16(&w).unwrap();
        let _ = be
            .linear_bf16(wbuf.as_ref(), &x, rows, in_f, out_f)
            .unwrap();
        let y = be
            .linear_bf16(wbuf.as_ref(), &x, rows, in_f, out_f)
            .unwrap();
        // bf16 has 8 mantissa bits → looser tolerance than f16
        for (g, c) in y.iter().zip(cpu_gemv(&w, &x, rows, in_f, out_f).iter()) {
            assert!((g - c).abs() < 5e-2, "{g} vs {c}");
        }
    }
}
