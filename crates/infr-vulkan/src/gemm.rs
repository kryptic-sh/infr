//! Cooperative-matrix GEMM (the production matmul primitive). Uses the GLSL coopmat shader
//! compiled by build.rs. f16 inputs, f32 accumulate/output. v1 requires m,n,k multiples of 16.

use std::sync::OnceLock;

use ash::vk;
use half::f16;

use infr_core::{backend::BufferUsage, error::Result, Backend};

use super::{as_vk_buf, be, VulkanBackend};

fn spv_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

const GEMM_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat.spv"));
const GEMM_TILED_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat_tiled.spv"));
const GEMM_PROJ_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_proj.spv"));
const ATTN_PARTIAL_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_partial.spv"));
const ATTN_QK_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_qk.spv"));
const ATTN_SM_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_softmax.spv"));
const ATTN_PV_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_pv.spv"));
const MMV_Q4_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q4.spv"));
const MMV_Q8_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q8.spv"));
const MMV_Q4_RES_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q4_res.spv"));
const MMV_Q8_RES_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q8_res.spv"));
static GEMM_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static GEMM_TILED_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static GEMM_PROJ_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_PARTIAL_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_QK_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_SM_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_PV_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static MMV_Q4_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static MMV_Q8_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static MMV_Q4_RES_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static MMV_Q8_RES_SPV: OnceLock<Vec<u32>> = OnceLock::new();

fn gemm_spv() -> &'static [u32] {
    GEMM_SPV.get_or_init(|| spv_words(GEMM_SPV_BYTES))
}
fn gemm_tiled_spv() -> &'static [u32] {
    GEMM_TILED_SPV.get_or_init(|| spv_words(GEMM_TILED_SPV_BYTES))
}
/// SPIR-V for the prefill projection GEMM (`C=A·Wᵀ`, f16/quant W). Used by the recorder.
pub(crate) fn gemm_proj_spv() -> &'static [u32] {
    GEMM_PROJ_SPV.get_or_init(|| spv_words(GEMM_PROJ_SPV_BYTES))
}
/// SPIR-V for the subgroup-reduction flash-decoding pass-1 (split-K) kernel. Used by the recorder.
pub(crate) fn attn_partial_spv() -> &'static [u32] {
    ATTN_PARTIAL_SPV.get_or_init(|| spv_words(ATTN_PARTIAL_SPV_BYTES))
}
/// SPIR-V for the non-FA prefill attention kernels (QK scores / row softmax / PV). Recorder use.
pub(crate) fn attn_qk_spv() -> &'static [u32] {
    ATTN_QK_SPV.get_or_init(|| spv_words(ATTN_QK_SPV_BYTES))
}
pub(crate) fn attn_softmax_spv() -> &'static [u32] {
    ATTN_SM_SPV.get_or_init(|| spv_words(ATTN_SM_SPV_BYTES))
}
pub(crate) fn attn_pv_spv() -> &'static [u32] {
    ATTN_PV_SPV.get_or_init(|| spv_words(ATTN_PV_SPV_BYTES))
}
/// SPIR-V for the subgroup decode GEMV (`y=x·Wᵀ`). `bits`=4/8 picks the quant variant; `res` adds
/// a fused residual. Used by the recorder's `linear_q` / `linear_add_q`.
pub(crate) fn mul_mat_vec_q_spv(bits: u32, res: bool) -> &'static [u32] {
    match (bits, res) {
        (4, false) => MMV_Q4_SPV.get_or_init(|| spv_words(MMV_Q4_SPV_BYTES)),
        (8, false) => MMV_Q8_SPV.get_or_init(|| spv_words(MMV_Q8_SPV_BYTES)),
        (4, true) => MMV_Q4_RES_SPV.get_or_init(|| spv_words(MMV_Q4_RES_SPV_BYTES)),
        (8, true) => MMV_Q8_RES_SPV.get_or_init(|| spv_words(MMV_Q8_RES_SPV_BYTES)),
        _ => panic!("mul_mat_vec_q: unsupported bits={bits}"),
    }
}

impl VulkanBackend {
    /// Untiled coopmat GEMM (m,n,k multiples of 16). Correct but memory-bound; use `matmul_f16`
    /// (tiled) for throughput.
    pub fn matmul_f16_untiled(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        assert!(m % 16 == 0 && n % 16 == 0 && k % 16 == 0);
        let kern = self.kernel_spv("gemm_coopmat", gemm_spv(), 3, 12);
        self.run_gemm(kern, a, b, m, k, n, (n / 16) as u32, (m / 16) as u32)
    }

    /// Tiled cooperative-matrix GEMM (shared-memory, 64x64 tiles): `C[m,n]=A[m,k]*B[k,n]`.
    /// f16 inputs, f32 output. v1 requires m,n multiples of 64 and k multiple of 32.
    pub fn matmul_f16(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        assert!(
            m % 64 == 0 && n % 64 == 0 && k % 32 == 0,
            "tiled coopmat GEMM needs m,n %64 and k %32 (got {m},{k},{n})"
        );
        let kern = self.kernel_spv_sg("gemm_coopmat_tiled", gemm_tiled_spv(), 3, 12, 32);
        self.run_gemm(kern, a, b, m, k, n, (n / 64) as u32, (m / 64) as u32)
    }

    /// Benchmark ONLY the tiled GEMM dispatch (weights pre-uploaded as f16; no host
    /// conversion / transfer in the loop). Returns avg seconds per dispatch.
    #[doc(hidden)]
    pub fn bench_tiled_gemm(&self, m: usize, k: usize, n: usize, iters: usize) -> f64 {
        let kern = self.kernel_spv_sg("gemm_coopmat_tiled", gemm_tiled_spv(), 3, 12, 32);
        let a16 = vec![0u16; m * k];
        let b16 = vec![0u16; k * n];
        let buf_a = self.alloc(a16.len() * 2, BufferUsage::Staging).unwrap();
        let buf_b = self.alloc(b16.len() * 2, BufferUsage::Staging).unwrap();
        let buf_c = self.alloc(m * n * 4, BufferUsage::Activations).unwrap();
        self.upload(buf_a.as_ref(), bytemuck::cast_slice(&a16))
            .unwrap();
        self.upload(buf_b.as_ref(), bytemuck::cast_slice(&b16))
            .unwrap();

        let device = self.shared.device.clone();
        unsafe {
            device
                .reset_descriptor_pool(kern.desc_pool, vk::DescriptorPoolResetFlags::empty())
                .unwrap();
        }
        let set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(kern.desc_pool)
                        .set_layouts(std::slice::from_ref(&kern.ds_layout)),
                )
                .unwrap()[0]
        };
        let bufs = [
            unsafe { as_vk_buf(buf_a.as_ref()) }.buffer,
            unsafe { as_vk_buf(buf_b.as_ref()) }.buffer,
            unsafe { as_vk_buf(buf_c.as_ref()) }.buffer,
        ];
        let infos: Vec<vk::DescriptorBufferInfo> = bufs
            .iter()
            .map(|&buffer| vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = (0..3)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i..i + 1])
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        let (gx, gy) = ((n / 64) as u32, (m / 64) as u32);

        let dispatch = || {
            let shared = std::sync::Arc::clone(&self.shared);
            self.one_shot(move |cmd| unsafe {
                shared
                    .device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kern.pipeline);
                shared.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    kern.pipeline_layout,
                    0,
                    &[set],
                    &[],
                );
                shared.device.cmd_push_constants(
                    cmd,
                    kern.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push,
                );
                shared.device.cmd_dispatch(cmd, gx, gy, 1);
            })
            .unwrap();
        };
        dispatch(); // warm
        let t = std::time::Instant::now();
        for _ in 0..iters {
            dispatch();
        }
        t.elapsed().as_secs_f64() / iters as f64
    }

    fn run_gemm(
        &self,
        kern: super::ops::ComputeKernel,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        gx: u32,
        gy: u32,
    ) -> Result<Vec<f32>> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        let device = self.shared.device.clone();

        let a16: Vec<u16> = a.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
        let b16: Vec<u16> = b.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
        let buf_a = self.alloc(a16.len() * 2, BufferUsage::Staging)?;
        let buf_b = self.alloc(b16.len() * 2, BufferUsage::Staging)?;
        let buf_c = self.alloc(m * n * 4, BufferUsage::Readback)?;
        self.upload(buf_a.as_ref(), bytemuck::cast_slice(&a16))?;
        self.upload(buf_b.as_ref(), bytemuck::cast_slice(&b16))?;

        unsafe {
            device
                .reset_descriptor_pool(kern.desc_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| be(format!("reset pool: {e}")))?;
        }
        let set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(kern.desc_pool)
                        .set_layouts(std::slice::from_ref(&kern.ds_layout)),
                )
                .map_err(|e| be(format!("alloc set: {e}")))?[0]
        };
        let vk_a = unsafe { as_vk_buf(buf_a.as_ref()) }.buffer;
        let vk_b = unsafe { as_vk_buf(buf_b.as_ref()) }.buffer;
        let vk_c = unsafe { as_vk_buf(buf_c.as_ref()) }.buffer;
        let infos = [
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
        let writes: Vec<vk::WriteDescriptorSet> = (0..3)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i..i + 1])
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());

        let shared = std::sync::Arc::clone(&self.shared);
        self.one_shot(move |cmd| unsafe {
            shared
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kern.pipeline);
            shared.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                kern.pipeline_layout,
                0,
                &[set],
                &[],
            );
            shared.device.cmd_push_constants(
                cmd,
                kern.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            shared.device.cmd_dispatch(cmd, gx, gy, 1);
        })?;

        let mut c_bytes = vec![0u8; m * n * 4];
        self.download(buf_c.as_ref(), &mut c_bytes)?;
        Ok(bytemuck::cast_slice(&c_bytes).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn check(got: &[f32], want: &[f32], label: &str) {
        let mut max_rel = 0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            max_rel = max_rel.max((g - w).abs() / w.abs().max(1.0));
        }
        println!("{label} max_rel_err = {max_rel:.4e}");
        assert!(max_rel < 2e-2, "{label} rel err {max_rel} too high");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU with cooperative matrix"]
    fn coopmat_gemm_untiled_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (m, k, n) = (64usize, 48usize, 32usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let got = be.matmul_f16_untiled(&a, &b, m, k, n).unwrap();
        check(&got, &cpu(&a, &b, m, k, n), "untiled");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU with cooperative matrix"]
    fn coopmat_gemm_tiled_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (m, k, n) = (128usize, 96usize, 64usize); // m,n %64, k %32
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        let got = be.matmul_f16(&a, &b, m, k, n).unwrap();
        check(&got, &cpu(&a, &b, m, k, n), "tiled");
    }

    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn coopmat_gemm_bench() {
        let be = VulkanBackend::new().unwrap();
        for s in [1024usize, 2048, 4096] {
            let dt = be.bench_tiled_gemm(s, s, s, 20);
            let flops = 2.0 * (s as f64).powi(3);
            println!(
                "tiled coopmat GEMM {s}^3: {:.3} ms, {:.0} GFLOP/s",
                dt * 1e3,
                flops / dt / 1e9
            );
        }
    }
}
