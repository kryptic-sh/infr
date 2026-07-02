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

/// Build-compiled native-block dequant GEMV SPIR-V for `(dtype, residual)`, or `None` if `dtype`
/// is not a native-block quant format. One match arm per quant format.
pub(crate) fn native_build_spv(dtype: infr_core::DType, res: bool) -> Option<&'static [u32]> {
    use infr_core::DType::*;
    // Each arm lazily decodes its own build-compiled .spv (a fresh `static` per block).
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            S.get_or_init(|| {
                spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
            })
            .as_slice()
        }};
    }
    Some(match (dtype, res) {
        (Q8_0, false) => v!("native_q8_0"),
        (Q8_0, true) => v!("native_q8_0_res"),
        (Q4_0, false) => v!("native_q4_0"),
        (Q4_0, true) => v!("native_q4_0_res"),
        (Q4_1, false) => v!("native_q4_1"),
        (Q4_1, true) => v!("native_q4_1_res"),
        (Q5_0, false) => v!("native_q5_0"),
        (Q5_0, true) => v!("native_q5_0_res"),
        (Q5_1, false) => v!("native_q5_1"),
        (Q5_1, true) => v!("native_q5_1_res"),
        (Q2K, false) => v!("native_q2k"),
        (Q2K, true) => v!("native_q2k_res"),
        (Q3K, false) => v!("native_q3k"),
        (Q3K, true) => v!("native_q3k_res"),
        (Q4K, false) => v!("native_q4k"),
        (Q4K, true) => v!("native_q4k_res"),
        (Q5K, false) => v!("native_q5k"),
        (Q5K, true) => v!("native_q5k_res"),
        (Q6K, false) => v!("native_q6k"),
        (Q6K, true) => v!("native_q6k_res"),
        (Iq4Nl, false) => v!("native_iq4nl"),
        (Iq4Nl, true) => v!("native_iq4nl_res"),
        (Iq4Xs, false) => v!("native_iq4xs"),
        (Iq4Xs, true) => v!("native_iq4xs_res"),
        (Mxfp4, false) => v!("native_mxfp4"),
        (Mxfp4, true) => v!("native_mxfp4_res"),
        (Nvfp4, false) => v!("native_nvfp4"),
        (Nvfp4, true) => v!("native_nvfp4_res"),
        (Tq1_0, false) => v!("native_tq1_0"),
        (Tq1_0, true) => v!("native_tq1_0_res"),
        (Tq2_0, false) => v!("native_tq2_0"),
        (Tq2_0, true) => v!("native_tq2_0_res"),
        (Iq2Xxs, false) => v!("native_iq2xxs"),
        (Iq2Xxs, true) => v!("native_iq2xxs_res"),
        (Iq2Xs, false) => v!("native_iq2xs"),
        (Iq2Xs, true) => v!("native_iq2xs_res"),
        (Iq2S, false) => v!("native_iq2s"),
        (Iq2S, true) => v!("native_iq2s_res"),
        (Iq3Xxs, false) => v!("native_iq3xxs"),
        (Iq3Xxs, true) => v!("native_iq3xxs_res"),
        (Iq3S, false) => v!("native_iq3s"),
        (Iq3S, true) => v!("native_iq3s_res"),
        (Iq1S, false) => v!("native_iq1s"),
        (Iq1S, true) => v!("native_iq1s_res"),
        (Iq1M, false) => v!("native_iq1m"),
        (Iq1M, true) => v!("native_iq1m_res"),
        _ => return None,
    })
}

/// SPIR-V for the id-indexed native GEMV (expert chosen from a GPU buffer). One specialization per
/// affine quant format; `None` for formats without an id variant (caller falls back to host top-k).
pub(crate) fn native_id_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            S.get_or_init(|| {
                spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
            })
            .as_slice()
        }};
    }
    Some(match dtype {
        Q8_0 => v!("native_id_q8_0"),
        Q4_0 => v!("native_id_q4_0"),
        Q4_1 => v!("native_id_q4_1"),
        Q5_0 => v!("native_id_q5_0"),
        Q5_1 => v!("native_id_q5_1"),
        Q2K => v!("native_id_q2k"),
        Q3K => v!("native_id_q3k"),
        Q4K => v!("native_id_q4k"),
        Q5K => v!("native_id_q5k"),
        Q6K => v!("native_id_q6k"),
        _ => return None,
    })
}
/// SPIR-V for the multi-slot id-indexed native GEMV (all n_used experts in one dispatch).
pub(crate) fn native_idm_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            S.get_or_init(|| {
                spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
            })
            .as_slice()
        }};
    }
    Some(match dtype {
        Q8_0 => v!("native_idm_q8_0"),
        Q4_0 => v!("native_idm_q4_0"),
        Q4_1 => v!("native_idm_q4_1"),
        Q5_0 => v!("native_idm_q5_0"),
        Q5_1 => v!("native_idm_q5_1"),
        Q2K => v!("native_idm_q2k"),
        Q3K => v!("native_idm_q3k"),
        Q4K => v!("native_idm_q4k"),
        Q5K => v!("native_idm_q5k"),
        Q6K => v!("native_idm_q6k"),
        _ => return None,
    })
}
/// SPIR-V for the multi-slot id-indexed Q4_K dp4a (mmq) GEMV.
pub(crate) fn native_mmv_id_q4k_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_mmv_id_q4k.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the tiled Q4_K dp4a (mmq) GEMM.
pub(crate) fn native_gemm_mmq_q4k_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the tiled Q6_K dp4a (mmq) GEMM (the MoE down projection).
pub(crate) fn native_gemm_mmq_q6k_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the MoE weighted-accumulate (sum of selected experts' down outputs into hidden).
pub(crate) fn moe_accumulate_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_accumulate.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the GPU MoE router top-k.
pub(crate) fn moe_topk_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_topk.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the greedy argmax (vocab logits → token id).
pub(crate) fn argmax_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for GPU stochastic sampling (radix top-k + temp + top-p → token id).
pub(crate) fn moe_sample_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_sample.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// Max `top_k` the GPU sampler supports (matches the shader's KMAX); larger falls back to host.
pub const SAMPLE_KMAX: usize = 64;
/// SPIR-V for the MoE expert-bucketing passes (count / exclusive-scan / scatter).
pub(crate) fn moe_bucket_count_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_bucket_count.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
pub(crate) fn moe_bucket_scan_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_bucket_scan.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
pub(crate) fn moe_bucket_scatter_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_bucket_scatter.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the indexed axpy (`acc += wts[slot]*x`).
pub(crate) fn add_scaled_id_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_scaled_id.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the LARGE-WARPTILE native-block prefill GEMM (8-warp BM=64×BN=256, gemm_proj_warp
/// structure with in-shader native dequant). Only the hot formats are compiled; `None` falls back
/// to the 64×64 `native_gemm_build_spv` kernel.
pub(crate) fn native_gemm_warp_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            S.get_or_init(|| {
                spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
            })
            .as_slice()
        }};
    }
    Some(match dtype {
        Q4K => v!("native_gemm_warp_q4k"),
        Q6K => v!("native_gemm_warp_q6k"),
        Q8_0 => v!("native_gemm_warp_q8_0"),
        _ => return None,
    })
}

/// SPIR-V for the native-block prefill GEMM (`C=A·Wᵀ`, raw GGUF blocks dequantized in-shader via the
/// coopmat tiled kernel). One specialization per quant format; `None` for unsupported dtypes.
pub(crate) fn native_gemm_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            S.get_or_init(|| {
                spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
            })
            .as_slice()
        }};
    }
    Some(match dtype {
        Q8_0 => v!("native_gemm_q8_0"),
        Q4_0 => v!("native_gemm_q4_0"),
        Q4_1 => v!("native_gemm_q4_1"),
        Q5_0 => v!("native_gemm_q5_0"),
        Q5_1 => v!("native_gemm_q5_1"),
        Q2K => v!("native_gemm_q2k"),
        Q3K => v!("native_gemm_q3k"),
        Q4K => v!("native_gemm_q4k"),
        Q5K => v!("native_gemm_q5k"),
        Q6K => v!("native_gemm_q6k"),
        Iq4Nl => v!("native_gemm_iq4nl"),
        Iq4Xs => v!("native_gemm_iq4xs"),
        Mxfp4 => v!("native_gemm_mxfp4"),
        Nvfp4 => v!("native_gemm_nvfp4"),
        Tq1_0 => v!("native_gemm_tq1_0"),
        Tq2_0 => v!("native_gemm_tq2_0"),
        Iq2Xxs => v!("native_gemm_iq2xxs"),
        Iq2Xs => v!("native_gemm_iq2xs"),
        Iq2S => v!("native_gemm_iq2s"),
        Iq3Xxs => v!("native_gemm_iq3xxs"),
        Iq3S => v!("native_gemm_iq3s"),
        Iq1S => v!("native_gemm_iq1s"),
        Iq1M => v!("native_gemm_iq1m"),
        _ => return None,
    })
}

const GEMM_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat.spv"));
const GEMM_TILED_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat_tiled.spv"));
const GEMM_WARP_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_warp.spv"));
const GEMM_DP4A_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_dp4a.spv"));
const QUANT_Q8_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quant_q8.spv"));
const GEMM_PROJ_MMQ_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_proj_mmq.spv"));
const GEMM_PROJ_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_proj.spv"));
const GEMM_PROJ_WARP_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_proj_warp.spv"));
const ATTN_PARTIAL_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_partial.spv"));
const ATTN_QK_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_qk.spv"));
const ATTN_QK_WARP_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_qk_warp.spv"));
const ATTN_FLASH_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash.spv"));
const ATTN_FLASH_BM32_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_bm32.spv"));
const ATTN_FLASH_PARTIAL_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_partial.spv"));
const ATTN_FLASH_PARTIAL_BM32_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_partial_bm32.spv"));
const ATTN_FLASH_WARP_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_warp.spv"));
const ATTN_FLASH_WARP_BM32_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_warp_bm32.spv"));
const ATTN_FLASH_REG_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_reg.spv"));
const ATTN_FLASH_REG_BR64_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_reg_br64.spv"));
const ATTN_FLASH_COMBINE_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_combine.spv"));
const ATTN_SM_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_softmax.spv"));
const ATTN_PV_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_pv.spv"));
const ATTN_PV_WARP_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_pv_warp.spv"));
const ATTN_PV_REDUCE_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_pv_reduce.spv"));
const RMSNORM_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.spv"));
const DELTANET_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deltanet.spv"));
const DELTANET_CHUNKED_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_chunked.spv"));
const DELTANET_PREP_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_prep.spv"));
const DELTANET_GATES_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_gates.spv"));
const DELTANET_SCAN_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_scan.spv"));
const CONV1D_SILU_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv1d_silu.spv"));
const CONV1D_SILU_PAR_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/conv1d_silu_par.spv"));
const CONV1D_SHIFT_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv1d_shift.spv"));
const COPY_STRIDED_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy_strided.spv"));
const MUL_SIGMOID_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_sigmoid.spv"));
const ADD_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add.spv"));
const SILU_MUL_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silu_mul.spv"));
const GELU_MUL_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gelu_mul.spv"));
const SILU_MUL_FUSED_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/silu_mul_fused.spv"));
const GELU_MUL_FUSED_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gelu_mul_fused.spv"));
const STORE_F16_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/store_f16.spv"));
const ROPE_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv"));
const LINEAR_F16_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_f16.spv"));
const LINEAR_BF16_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_bf16.spv"));
const LINEAR_Q_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_q.spv"));
const LINEAR_RES_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_res.spv"));
const LINEAR_RES_Q_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_res_q.spv"));
const ATTENTION_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv"));
const ATTN_COMBINE_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_combine.spv"));
const ATTENTION_KV_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention_kv.spv"));
const QK_NORM_ROPE_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qk_norm_rope.spv"));
const QK_NORM_ROPE_FF_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/qk_norm_rope_ff.spv"));
const ATTN_IN_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_in.spv"));
const FFN_IN_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ffn_in.spv"));
const FFN_IN_Q_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ffn_in_q.spv"));
const ATTN_IN_Q_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_in_q.spv"));
const MMV_Q4_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q4.spv"));
const MMV_Q8_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q8.spv"));
const MMV_Q4_RES_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q4_res.spv"));
const MMV_Q8_RES_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q8_res.spv"));
static GEMM_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static GEMM_TILED_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static GEMM_WARP_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static GEMM_DP4A_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static QUANT_Q8_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static GEMM_PROJ_MMQ_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static GEMM_PROJ_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static GEMM_PROJ_WARP_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_PARTIAL_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_QK_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_QK_WARP_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_REG_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_REG_BR64_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_COMBINE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_SM_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_PV_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_PV_WARP_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_PV_REDUCE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static RMSNORM_SPV: OnceLock<Vec<u32>> = OnceLock::new();
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
fn gemm_warp_spv() -> &'static [u32] {
    GEMM_WARP_SPV.get_or_init(|| spv_words(GEMM_WARP_SPV_BYTES))
}
fn gemm_dp4a_spv() -> &'static [u32] {
    GEMM_DP4A_SPV.get_or_init(|| spv_words(GEMM_DP4A_SPV_BYTES))
}
/// SPIR-V for the activation int8 quantize pass (Q8 per block) feeding the dp4a mmq matmul.
pub(crate) fn quant_q8_spv() -> &'static [u32] {
    QUANT_Q8_SPV.get_or_init(|| spv_words(QUANT_Q8_SPV_BYTES))
}
/// SPIR-V for the integer (dp4a) u4 projection GEMM. Weights stay quantized; no per-GEMM dequant.
pub(crate) fn gemm_proj_mmq_spv() -> &'static [u32] {
    GEMM_PROJ_MMQ_SPV.get_or_init(|| spv_words(GEMM_PROJ_MMQ_SPV_BYTES))
}
/// SPIR-V for the prefill projection GEMM (`C=A·Wᵀ`, f16/quant W). Used by the recorder.
pub(crate) fn gemm_proj_spv() -> &'static [u32] {
    GEMM_PROJ_SPV.get_or_init(|| spv_words(GEMM_PROJ_SPV_BYTES))
}
/// Warp-tiled projection GEMM (BM=64,BN=128). Faster for large M (low/mid-ctx prefill); the recorder
/// falls back to `gemm_proj_spv` for small M (high ctx) where its fewer workgroups lose occupancy.
pub(crate) fn gemm_proj_warp_spv() -> &'static [u32] {
    GEMM_PROJ_WARP_SPV.get_or_init(|| spv_words(GEMM_PROJ_WARP_SPV_BYTES))
}
/// SPIR-V for the subgroup-reduction flash-decoding pass-1 (split-K) kernel. Used by the recorder.
pub(crate) fn attn_partial_spv() -> &'static [u32] {
    ATTN_PARTIAL_SPV.get_or_init(|| spv_words(ATTN_PARTIAL_SPV_BYTES))
}
/// SPIR-V for the non-FA prefill attention kernels (QK scores / row softmax / PV). Recorder use.
pub(crate) fn attn_qk_spv() -> &'static [u32] {
    ATTN_QK_SPV.get_or_init(|| spv_words(ATTN_QK_SPV_BYTES))
}
/// 8-warp/256-thread QK GEMM (kv_pad % 256). Matches ollama's mul_mm warptile; the recorder uses it
/// over the 4-warp attn_qk unless INFR_NO_QK_WARP is set.
pub(crate) fn attn_qk_warp_spv() -> &'static [u32] {
    ATTN_QK_WARP_SPV.get_or_init(|| spv_words(ATTN_QK_WARP_SPV_BYTES))
}
/// Fused flash-attention prefill (QK→softmax→PV, no materialized S). Recorder `attention_prefill_flash`.
pub(crate) fn attn_flash_spv() -> &'static [u32] {
    ATTN_FLASH_SPV.get_or_init(|| spv_words(ATTN_FLASH_SPV_BYTES))
}
/// BM=32 build of the fused flash prefill (29056 B shared) for sub-64 KB shared devices.
pub(crate) fn attn_flash_bm32_spv() -> &'static [u32] {
    ATTN_FLASH_BM32_SPV.get_or_init(|| spv_words(ATTN_FLASH_BM32_SPV_BYTES))
}
/// Flash-attention split-K partial pass (per kv-split online-softmax partials). Recorder use.
pub(crate) fn attn_flash_partial_spv() -> &'static [u32] {
    ATTN_FLASH_PARTIAL_SPV.get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_SPV_BYTES))
}
/// BM=32 build of the split-K flash partial (29056 B shared) for sub-64 KB shared devices.
pub(crate) fn attn_flash_partial_bm32_spv() -> &'static [u32] {
    ATTN_FLASH_PARTIAL_BM32_SPV.get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_BM32_SPV_BYTES))
}
/// Register-blocked flash partial (hd=128). Used over attn_flash_partial when hd==128.
pub(crate) fn attn_flash_warp_spv() -> &'static [u32] {
    ATTN_FLASH_WARP_SPV.get_or_init(|| spv_words(ATTN_FLASH_WARP_SPV_BYTES))
}
/// BM=32 build of the flash partial (29056 B shared vs 58112 B): for devices whose
/// maxComputeSharedMemorySize is under the 64 KB the default BM=64 tile needs (NVIDIA, MoltenVK).
pub(crate) fn attn_flash_warp_bm32_spv() -> &'static [u32] {
    ATTN_FLASH_WARP_BM32_SPV.get_or_init(|| spv_words(ATTN_FLASH_WARP_BM32_SPV_BYTES))
}
/// FlashAttention-2 register-O flash partial (Br=128, per-thread register accumulator). hd=128.
pub(crate) fn attn_flash_reg_spv() -> &'static [u32] {
    ATTN_FLASH_REG_SPV.get_or_init(|| spv_words(ATTN_FLASH_REG_SPV_BYTES))
}
/// BR=64 build of the register-O flash partial (29440 B shared) for sub-64 KB shared devices.
pub(crate) fn attn_flash_reg_br64_spv() -> &'static [u32] {
    ATTN_FLASH_REG_BR64_SPV.get_or_init(|| spv_words(ATTN_FLASH_REG_BR64_SPV_BYTES))
}
/// Flash-attention split-K combine (merge partials → final O). Recorder use.
pub(crate) fn attn_flash_combine_spv() -> &'static [u32] {
    ATTN_FLASH_COMBINE_SPV.get_or_init(|| spv_words(ATTN_FLASH_COMBINE_SPV_BYTES))
}
pub(crate) fn attn_softmax_spv() -> &'static [u32] {
    ATTN_SM_SPV.get_or_init(|| spv_words(ATTN_SM_SPV_BYTES))
}
pub(crate) fn attn_pv_spv() -> &'static [u32] {
    ATTN_PV_SPV.get_or_init(|| spv_words(ATTN_PV_SPV_BYTES))
}
/// 8-warp/256-thread PV GEMM (BN=128=hd, hd % 128). The recorder uses it over the 4-warp attn_pv
/// when hd % 128 == 0 and INFR_NO_PV_WARP is unset.
pub(crate) fn attn_pv_warp_spv() -> &'static [u32] {
    ATTN_PV_WARP_SPV.get_or_init(|| spv_words(ATTN_PV_WARP_SPV_BYTES))
}
/// SPIR-V for the attn_pv split-K partial reducer (sums n_splits partial-O buffers).
pub(crate) fn attn_pv_reduce_spv() -> &'static [u32] {
    ATTN_PV_REDUCE_SPV.get_or_init(|| spv_words(ATTN_PV_REDUCE_SPV_BYTES))
}
/// SPIR-V for the 256-thread subgroup RMSNorm (`y=rmsnorm(x,w)`). Used by the recorder's `rmsnorm`.
pub(crate) fn rmsnorm_spv() -> &'static [u32] {
    RMSNORM_SPV.get_or_init(|| spv_words(RMSNORM_SPV_BYTES))
}
/// SPIR-V for the elementwise add (`y=a+b`).
pub(crate) fn add_spv() -> &'static [u32] {
    static ADD_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ADD_SPV.get_or_init(|| spv_words(ADD_SPV_BYTES))
}
/// SPIR-V for the scaled add / axpy (`acc += scale*x`).
pub(crate) fn add_scaled_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_scaled.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the in-place scalar multiply (`y *= scale`). gemma4 per-layer output scale.
pub(crate) fn scale_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scale.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for elementwise softcap `y = cap·tanh(x/cap)` (gemma logit softcap).
pub(crate) fn softcap_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softcap.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the row gather (`dst[j]=src[idx[j]]`).
pub(crate) fn gather_rows_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gather_rows.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the weighted row scatter-add (`dst[idx[j]] += w[j]*y[j]`).
pub(crate) fn scatter_add_rows_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scatter_add_rows.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the SwiGLU activation (`y=silu(gate)*up`).
pub(crate) fn silu_mul_spv() -> &'static [u32] {
    static SILU_MUL_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    SILU_MUL_SPV.get_or_init(|| spv_words(SILU_MUL_SPV_BYTES))
}
/// SPIR-V for the gated-DeltaNet recurrence step (Qwen3-Next SSM).
pub(crate) fn deltanet_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_SPV_BYTES))
}
/// SPIR-V for the chunked-DeltaNet PREP pass (normalize + intra-chunk dot matrices).
pub(crate) fn deltanet_prep_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_PREP_SPV_BYTES))
}
/// SPIR-V for the chunked-DeltaNet GATES pass (β + prefix log-decay per chunk/head).
pub(crate) fn deltanet_gates_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_GATES_SPV_BYTES))
}
/// SPIR-V for the chunked-DeltaNet SCAN pass (the sequential state-coupled part).
pub(crate) fn deltanet_scan_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_SCAN_SPV_BYTES))
}
/// SPIR-V for the CHUNKED gated-DeltaNet prefill (chunkwise delta rule, C=32).
pub(crate) fn deltanet_chunked_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_CHUNKED_SPV_BYTES))
}
/// SPIR-V for the causal depthwise conv1d + SiLU step (Qwen3-Next SSM input conv).
pub(crate) fn conv1d_silu_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(CONV1D_SILU_SPV_BYTES))
}
/// SPIR-V for the BATCH depthwise conv1d+SiLU (pass 1: all outputs in parallel).
pub(crate) fn conv1d_silu_par_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(CONV1D_SILU_PAR_SPV_BYTES))
}
/// SPIR-V for the BATCH depthwise conv1d history rebuild (pass 2).
pub(crate) fn conv1d_shift_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(CONV1D_SHIFT_SPV_BYTES))
}
/// SPIR-V for the batched strided row copy (Op::CopyStrided in one dispatch).
pub(crate) fn copy_strided_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(COPY_STRIDED_SPV_BYTES))
}
/// SPIR-V for the elementwise sigmoid gate `y = a * sigmoid(b)`.
pub(crate) fn mul_sigmoid_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(MUL_SIGMOID_SPV_BYTES))
}
/// SPIR-V for the GeGLU activation with separate gate/up buffers (`y=gelu(gate)*up`). gemma4's
/// per-layer-embd gate.
pub(crate) fn gelu_mul_spv() -> &'static [u32] {
    static GELU_MUL_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    GELU_MUL_SPV.get_or_init(|| spv_words(GELU_MUL_SPV_BYTES))
}
/// SPIR-V for the fused SwiGLU over a combined gate||up buffer.
pub(crate) fn silu_mul_fused_spv() -> &'static [u32] {
    static SILU_MUL_FUSED_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    SILU_MUL_FUSED_SPV.get_or_init(|| spv_words(SILU_MUL_FUSED_SPV_BYTES))
}
/// SPIR-V for the fused GeGLU (GELU tanh-approx gate) over a combined gate||up buffer (gemma).
pub(crate) fn gelu_mul_fused_spv() -> &'static [u32] {
    static GELU_MUL_FUSED_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    GELU_MUL_FUSED_SPV.get_or_init(|| spv_words(GELU_MUL_FUSED_SPV_BYTES))
}
/// SPIR-V for the f32→f16 cast-store into an f16 cache.
pub(crate) fn store_f16_spv() -> &'static [u32] {
    static STORE_F16_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    STORE_F16_SPV.get_or_init(|| spv_words(STORE_F16_SPV_BYTES))
}
/// SPIR-V for RoPE (ggml NORM, interleaved pairs).
pub(crate) fn rope_spv() -> &'static [u32] {
    static ROPE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ROPE_SPV.get_or_init(|| spv_words(ROPE_SPV_BYTES))
}
/// SPIR-V for the f16-weight GEMV (`y=x·Wᵀ`).
pub(crate) fn linear_f16_spv() -> &'static [u32] {
    static LINEAR_F16_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    LINEAR_F16_SPV.get_or_init(|| spv_words(LINEAR_F16_SPV_BYTES))
}
/// SPIR-V for the bf16-weight GEMV (`y=x·Wᵀ`).
pub(crate) fn linear_bf16_spv() -> &'static [u32] {
    static LINEAR_BF16_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    LINEAR_BF16_SPV.get_or_init(|| spv_words(LINEAR_BF16_SPV_BYTES))
}
/// SPIR-V for the unified affine-quant dequant GEMV (`y=x·Wᵀ`).
pub(crate) fn linear_q_spv() -> &'static [u32] {
    static LINEAR_Q_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    LINEAR_Q_SPV.get_or_init(|| spv_words(LINEAR_Q_SPV_BYTES))
}
/// SPIR-V for the f16-weight GEMV with fused residual.
pub(crate) fn linear_res_spv() -> &'static [u32] {
    static LINEAR_RES_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    LINEAR_RES_SPV.get_or_init(|| spv_words(LINEAR_RES_SPV_BYTES))
}
/// SPIR-V for the affine-quant dequant GEMV with fused residual.
pub(crate) fn linear_res_q_spv() -> &'static [u32] {
    static LINEAR_RES_Q_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    LINEAR_RES_Q_SPV.get_or_init(|| spv_words(LINEAR_RES_Q_SPV_BYTES))
}
/// SPIR-V for the online-softmax GQA attention (hd<=128).
pub(crate) fn attention_spv() -> &'static [u32] {
    static ATTENTION_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ATTENTION_SPV.get_or_init(|| spv_words(ATTENTION_SPV_BYTES))
}
/// SPIR-V for flash-decode combine (merge split-K partials).
pub(crate) fn attn_combine_spv() -> &'static [u32] {
    static ATTN_COMBINE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ATTN_COMBINE_SPV.get_or_init(|| spv_words(ATTN_COMBINE_SPV_BYTES))
}
/// SPIR-V for tiled online-softmax attention over an f16 KV cache.
pub(crate) fn attention_kv_spv() -> &'static [u32] {
    static ATTENTION_KV_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ATTENTION_KV_SPV.get_or_init(|| spv_words(ATTENTION_KV_SPV_BYTES))
}
/// SPIR-V for fused per-head QK-norm + NEOX RoPE (f16 out).
pub(crate) fn qk_norm_rope_spv() -> &'static [u32] {
    static QK_NORM_ROPE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    QK_NORM_ROPE_SPV.get_or_init(|| spv_words(QK_NORM_ROPE_SPV_BYTES))
}
/// SPIR-V for QK-norm + RoPE with proportional-rope freq_factors (gemma4 full-attention layers).
pub(crate) fn qk_norm_rope_ff_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(QK_NORM_ROPE_FF_SPV_BYTES))
}
// Record-once decode variants (`-DUSE_PARAMS`): read the per-token pos/kv_len from a host-updated
// params SSBO instead of push constants, so the decode command buffer can be replayed across tokens.
macro_rules! dyn_spv {
    ($f:ident, $name:literal) => {
        pub(crate) fn $f() -> &'static [u32] {
            const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"));
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            S.get_or_init(|| spv_words(BYTES))
        }
    };
}
dyn_spv!(qk_norm_rope_dyn_spv, "qk_norm_rope_dyn");
dyn_spv!(store_f16_dyn_spv, "store_f16_dyn");
dyn_spv!(attention_kv_dyn_spv, "attention_kv_dyn");
dyn_spv!(attn_partial_dyn_spv, "attn_partial_dyn");
dyn_spv!(attn_in_dyn_spv, "attn_in_dyn");
/// SPIR-V for fused attention input (RMSNorm + QKV proj + RoPE).
pub(crate) fn attn_in_spv() -> &'static [u32] {
    static ATTN_IN_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ATTN_IN_SPV.get_or_init(|| spv_words(ATTN_IN_SPV_BYTES))
}
/// SPIR-V for fused FFN input (RMSNorm + gate/up proj + SwiGLU).
pub(crate) fn ffn_in_spv() -> &'static [u32] {
    static FFN_IN_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    FFN_IN_SPV.get_or_init(|| spv_words(FFN_IN_SPV_BYTES))
}
/// SPIR-V for the quant variant of fused FFN input.
pub(crate) fn ffn_in_q_spv() -> &'static [u32] {
    static FFN_IN_Q_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    FFN_IN_Q_SPV.get_or_init(|| spv_words(FFN_IN_Q_SPV_BYTES))
}
/// SPIR-V for the quant variant of fused attention input (RMSNorm + QKV proj).
pub(crate) fn attn_in_q_spv() -> &'static [u32] {
    static ATTN_IN_Q_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ATTN_IN_Q_SPV.get_or_init(|| spv_words(ATTN_IN_Q_SPV_BYTES))
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
        assert!(m.is_multiple_of(16) && n.is_multiple_of(16) && k.is_multiple_of(16));
        let kern = self.kernel("gemm_coopmat", gemm_spv(), 3, 12);
        self.run_gemm(kern, a, b, m, k, n, (n / 16) as u32, (m / 16) as u32)
    }

    /// mul_mm-style warp-tiled coopmat GEMM `C[m,n]=A[m,k]·B[k,n]`. m,n %128, k %16.
    pub fn matmul_warp(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(16));
        let kern = self.kernel_sg("gemm_warp", gemm_warp_spv(), 3, 12, 32);
        self.run_gemm(kern, a, b, m, k, n, (n / 128) as u32, (m / 128) as u32)
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
            m.is_multiple_of(64) && n.is_multiple_of(64) && k.is_multiple_of(32),
            "tiled coopmat GEMM needs m,n %64 and k %32 (got {m},{k},{n})"
        );
        let kern = self.kernel_sg("gemm_coopmat_tiled", gemm_tiled_spv(), 3, 12, 32);
        self.run_gemm(kern, a, b, m, k, n, (n / 64) as u32, (m / 64) as u32)
    }

    /// Benchmark ONLY the tiled GEMM dispatch (weights pre-uploaded as f16; no host
    /// conversion / transfer in the loop). Returns avg seconds per dispatch.
    #[doc(hidden)]
    pub fn bench_tiled_gemm(&self, m: usize, k: usize, n: usize, iters: usize) -> f64 {
        let kern = self.kernel_sg("gemm_coopmat_tiled", gemm_tiled_spv(), 3, 12, 32);
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

    /// Benchmark the mul_mm-style warp-tiled GEMM (m,n %128, k %16). Returns avg sec/dispatch.
    #[doc(hidden)]
    pub fn bench_warp_gemm(&self, m: usize, k: usize, n: usize, iters: usize) -> f64 {
        let kern = self.kernel_sg("gemm_warp", gemm_warp_spv(), 3, 12, 32);
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
        let (gx, gy) = ((n / 128) as u32, (m / 128) as u32);
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

    /// Benchmark the RAW dp4a scalar GEMM (m,n %64, k %32). Ceiling probe. Returns avg sec/dispatch.
    #[doc(hidden)]
    pub fn bench_dp4a_gemm(&self, m: usize, k: usize, n: usize, iters: usize) -> f64 {
        let kp = k / 4;
        let kern = self.kernel_sg("gemm_dp4a", gemm_dp4a_spv(), 3, 12, 32);
        let buf_a = self.alloc(m * kp * 4, BufferUsage::Staging).unwrap();
        let buf_b = self.alloc(n * kp * 4, BufferUsage::Staging).unwrap();
        let buf_c = self.alloc(m * n * 4, BufferUsage::Activations).unwrap();
        self.upload(buf_a.as_ref(), &vec![0u8; m * kp * 4]).unwrap();
        self.upload(buf_b.as_ref(), &vec![0u8; n * kp * 4]).unwrap();
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
        push[8..12].copy_from_slice(&(kp as u32).to_ne_bytes());
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
    #[ignore = "requires a Vulkan GPU"]
    fn warp_gemm_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        for &(m, k, n) in &[
            (128usize, 16usize, 128usize),
            (256, 128, 256),
            (128, 512, 128),
        ] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
            let got = be.matmul_warp(&a, &b, m, k, n).unwrap();
            check(&got, &cpu(&a, &b, m, k, n), "warp");
        }
    }

    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn dp4a_ceiling() {
        use std::io::Write as _;
        let be = VulkanBackend::new().unwrap();
        for &(m, k, n, label) in &[
            (2048usize, 2048usize, 2048usize, "dp4a 2048^3"),
            (2048, 1024, 2048, "dp4a proj m2048 k1024 n2048"),
            (512, 1024, 2048, "dp4a proj-smallM m512 k1024 n2048"),
            (2048, 1024, 6144, "dp4a ffn m2048 k1024 n6144"),
        ] {
            print!("running {label}... ");
            std::io::stdout().flush().ok();
            let dt = be.bench_dp4a_gemm(m, k, n, 30);
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            println!("{:.3} ms, {:.0} GFLOP/s", dt * 1e3, flops / dt / 1e9);
            std::io::stdout().flush().ok();
        }
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
        // Attention shapes (per head, 32k ctx): QK=[512,128]·[128,32768], PV=[512,32768]·[32768,128]
        for &(m, k, n, label) in &[
            (512usize, 128usize, 32768usize, "QK m512 k128 n32k"),
            (512, 32768, 128, "PV m512 k32k n128"),
        ] {
            let dt = be.bench_tiled_gemm(m, k, n, 20);
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            println!(
                "tiled coopmat GEMM {label}: {:.3} ms, {:.0} GFLOP/s",
                dt * 1e3,
                flops / dt / 1e9
            );
        }
        // mul_mm-style warp-tiled GEMM at the same shapes (m,n %128, k %16)
        for &(m, k, n, label) in &[
            (2048usize, 2048usize, 2048usize, "warp 2048^3"),
            (512, 128, 32768, "warp QK m512 k128 n32k"),
            (512, 32768, 128, "warp PV m512 k32k n128"),
        ] {
            let dt = be.bench_warp_gemm(m, k, n, 20);
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            println!(
                "{label}: {:.3} ms, {:.0} GFLOP/s",
                dt * 1e3,
                flops / dt / 1e9
            );
        }
        // RAW dp4a scalar ceiling (int8 WMMA hangs on RADV). GFLOP/s comparable to the f16 numbers.
        for &(m, k, n, label) in &[
            (2048usize, 2048usize, 2048usize, "dp4a 2048^3"),
            (2048, 1024, 2048, "dp4a proj m2048 k1024 n2048"),
            (512, 1024, 2048, "dp4a proj-smallM m512 k1024 n2048"),
        ] {
            let dt = be.bench_dp4a_gemm(m, k, n, 20);
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            println!(
                "{label}: {:.3} ms, {:.0} GFLOP/s",
                dt * 1e3,
                flops / dt / 1e9
            );
        }
    }
}
