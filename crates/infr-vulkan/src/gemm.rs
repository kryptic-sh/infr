//! Cooperative-matrix GEMM (the production matmul primitive). Uses the GLSL coopmat shader
//! compiled by build.rs. f16 inputs, f32 accumulate/output. v1 requires m,n,k multiples of 16.

use std::sync::OnceLock;

use ash::vk;
use half::f16;

use infr_core::{backend::BufferUsage, error::Result, Backend};

use super::{as_vk_buf, VulkanBackend};

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn spv_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Build-compiled multi-row native GEMV SPIR-V (m = 2..8, weight streamed once — the spec-decode
/// verify / short-suffix-prefill shape). `None` for formats without an mrow build (they keep the
/// tiled GEMM route).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mrow_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
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
        Q8_0 => v!("native_mrow_q8_0"),
        Bf16 => v!("native_mrow_bf16"),
        Q4_0 => v!("native_mrow_q4_0"),
        Q4_1 => v!("native_mrow_q4_1"),
        Q5_0 => v!("native_mrow_q5_0"),
        Q5_1 => v!("native_mrow_q5_1"),
        Q2K => v!("native_mrow_q2k"),
        Q3K => v!("native_mrow_q3k"),
        Q4K => v!("native_mrow_q4k"),
        Q5K => v!("native_mrow_q5k"),
        Q6K => v!("native_mrow_q6k"),
        Iq4Nl => v!("native_mrow_iq4nl"),
        Iq4Xs => v!("native_mrow_iq4xs"),
        Q2_0 => v!("native_mrow_q2_0"),
        _ => return None,
    })
}

/// Kernel-cache name for the multi-row native GEMV (must pair with [`native_mrow_build_spv`]).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mrow_kernel_name(dtype: infr_core::DType) -> &'static str {
    use infr_core::DType::*;
    match dtype {
        Q8_0 => "native_mrow_q8_0",
        Bf16 => "native_mrow_bf16",
        Q4_0 => "native_mrow_q4_0",
        Q4_1 => "native_mrow_q4_1",
        Q5_0 => "native_mrow_q5_0",
        Q5_1 => "native_mrow_q5_1",
        Q2K => "native_mrow_q2k",
        Q3K => "native_mrow_q3k",
        Q4K => "native_mrow_q4k",
        Q5K => "native_mrow_q5k",
        Q6K => "native_mrow_q6k",
        Iq4Nl => "native_mrow_iq4nl",
        Iq4Xs => "native_mrow_iq4xs",
        Q2_0 => "native_mrow_q2_0",
        _ => "native_mrow_unsupported",
    }
}

/// Build-compiled native-block dequant GEMV SPIR-V for `(dtype, residual)`, or `None` if `dtype`
/// is not a native-block quant format. One match arm per quant format.
#[cfg_attr(infr_profile, infr_prof::instrument)]
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
        (Bf16, false) => v!("native_bf16"),
        (Bf16, true) => v!("native_bf16_res"),
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
        (Q2_0, false) => v!("native_q2_0"),
        (Q2_0, true) => v!("native_q2_0_res"),
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

/// SPIR-V + kernel-cache name for the multi-output-row decode GEMV (`RM` rows/workgroup, bit-
/// identical per row to the RM=1 native GEMV). Only the K-quant formats that dominate decode
/// (Q4_K/Q6_K) have RM builds; everything else stays on the RM=1 path. `rm` is 2 or 4.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_rm_build_spv(
    dtype: infr_core::DType,
    res: bool,
    rm: u32,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (dtype, res, rm) {
        (Q4K, false, 2) => v!("native_q4k_rm2"),
        (Q4K, true, 2) => v!("native_q4k_rm2_res"),
        (Q4K, false, 4) => v!("native_q4k_rm4"),
        (Q4K, true, 4) => v!("native_q4k_rm4_res"),
        (Q6K, false, 2) => v!("native_q6k_rm2"),
        (Q6K, true, 2) => v!("native_q6k_rm2_res"),
        (Q6K, false, 4) => v!("native_q6k_rm4"),
        (Q6K, true, 4) => v!("native_q6k_rm4_res"),
        _ => None,
    }
}

/// SPIR-V + kernel name for an experimental RM kernel variant (env-gated, default OFF).
/// Returns (kernel_name, spv) for the given variant + dtype + res combination.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_rm_variant_spv(
    variant: &str,
    dtype: infr_core::DType,
    res: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (variant, dtype, res) {
        ("sg", Q4K, false) => v!("native_q4k_rm2_sg"),
        ("sg", Q4K, true) => v!("native_q4k_rm2_sg_res"),
        ("sg", Q6K, false) => v!("native_q6k_rm2_sg"),
        ("sg", Q6K, true) => v!("native_q6k_rm2_sg_res"),
        ("dbuf", Q4K, false) => v!("native_q4k_rm2_dbuf"),
        ("dbuf", Q4K, true) => v!("native_q4k_rm2_dbuf_res"),
        ("wg128", Q4K, false) => v!("native_q4k_rm2_wg128"),
        ("wg128", Q4K, true) => v!("native_q4k_rm2_wg128_res"),
        ("reg", Q4K, false) => v!("native_q4k_rm2_reg"),
        ("reg", Q4K, true) => v!("native_q4k_rm2_reg_res"),
        _ => None,
    }
}

/// SPIR-V + kernel-cache name for the reassociation-tolerant subgroup+NUM_ROWS decode GEMV
/// (`native_gemv_sg.comp`, one pinned subgroup per workgroup + subgroupAdd). NOT bit-identical to
/// the tree GEMV — reordered accumulation; the caller must gate to the projection band and
/// re-bless any changed golden. Only Q6_K has an SG build (on Q4_K the tree/RM kernel already
/// saturates — SG regressed). `nr` ∈ {2,4,8}. `sg16` selects the `-DSG=16` twin (Intel,
/// `caps.sg_pref == 16` — pass false everywhere else for the byte-identical SG=32 build).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_sg_build_spv(
    dtype: infr_core::DType,
    res: bool,
    nr: u32,
    sg16: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (dtype, res, nr, sg16) {
        (Q6K, false, 2, false) => v!("native_q6k_sg2"),
        (Q6K, true, 2, false) => v!("native_q6k_sg2_res"),
        (Q6K, false, 4, false) => v!("native_q6k_sg4"),
        (Q6K, true, 4, false) => v!("native_q6k_sg4_res"),
        (Q6K, false, 8, false) => v!("native_q6k_sg8"),
        (Q6K, true, 8, false) => v!("native_q6k_sg8_res"),
        (Q6K, false, 2, true) => v!("native_q6k_sg2_sg16"),
        (Q6K, true, 2, true) => v!("native_q6k_sg2_res_sg16"),
        (Q6K, false, 4, true) => v!("native_q6k_sg4_sg16"),
        (Q6K, true, 4, true) => v!("native_q6k_sg4_res_sg16"),
        (Q6K, false, 8, true) => v!("native_q6k_sg8_sg16"),
        (Q6K, true, 8, true) => v!("native_q6k_sg8_res_sg16"),
        _ => None,
    }
}

/// SPIR-V for the id-indexed native GEMV (expert chosen from a GPU buffer). One specialization per
/// weight format an expert bank can hold — the FULL dense native-GEMV format set plus F16/F32 for
/// float banks (resident float banks bind as effective f16; paged ones stage raw GGUF bytes, so
/// both float variants exist). `None` only for non-weight dtypes (I32/U32/...).
#[cfg_attr(infr_profile, infr_prof::instrument)]
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
        Iq4Nl => v!("native_id_iq4nl"),
        Iq4Xs => v!("native_id_iq4xs"),
        Mxfp4 => v!("native_id_mxfp4"),
        Nvfp4 => v!("native_id_nvfp4"),
        Tq1_0 => v!("native_id_tq1_0"),
        Tq2_0 => v!("native_id_tq2_0"),
        Q2_0 => v!("native_id_q2_0"),
        Iq2Xxs => v!("native_id_iq2xxs"),
        Iq2Xs => v!("native_id_iq2xs"),
        Iq2S => v!("native_id_iq2s"),
        Iq3Xxs => v!("native_id_iq3xxs"),
        Iq3S => v!("native_id_iq3s"),
        Iq1S => v!("native_id_iq1s"),
        Iq1M => v!("native_id_iq1m"),
        Bf16 => v!("native_id_bf16"),
        F16 => v!("native_id_f16"),
        F32 => v!("native_id_f32"),
        _ => return None,
    })
}
/// SPIR-V for the multi-slot id-indexed native GEMV (all n_used experts in one dispatch).
#[cfg_attr(infr_profile, infr_prof::instrument)]
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
        Iq4Nl => v!("native_idm_iq4nl"),
        Iq4Xs => v!("native_idm_iq4xs"),
        Mxfp4 => v!("native_idm_mxfp4"),
        Nvfp4 => v!("native_idm_nvfp4"),
        Tq1_0 => v!("native_idm_tq1_0"),
        Tq2_0 => v!("native_idm_tq2_0"),
        Q2_0 => v!("native_idm_q2_0"),
        Iq2Xxs => v!("native_idm_iq2xxs"),
        Iq2Xs => v!("native_idm_iq2xs"),
        Iq2S => v!("native_idm_iq2s"),
        Iq3Xxs => v!("native_idm_iq3xxs"),
        Iq3S => v!("native_idm_iq3s"),
        Iq1S => v!("native_idm_iq1s"),
        Iq1M => v!("native_idm_iq1m"),
        Bf16 => v!("native_idm_bf16"),
        F16 => v!("native_idm_f16"),
        F32 => v!("native_idm_f32"),
        _ => return None,
    })
}
/// [`native_id_build_spv`]'s paged twin (`infr_vulkan::pager`) — one extra LUT-buffer binding.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_id_paged_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
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
        Q8_0 => v!("native_id_q8_0_paged"),
        Q4_0 => v!("native_id_q4_0_paged"),
        Q4_1 => v!("native_id_q4_1_paged"),
        Q5_0 => v!("native_id_q5_0_paged"),
        Q5_1 => v!("native_id_q5_1_paged"),
        Q2K => v!("native_id_q2k_paged"),
        Q3K => v!("native_id_q3k_paged"),
        Q4K => v!("native_id_q4k_paged"),
        Q5K => v!("native_id_q5k_paged"),
        Q6K => v!("native_id_q6k_paged"),
        Iq4Nl => v!("native_id_iq4nl_paged"),
        Iq4Xs => v!("native_id_iq4xs_paged"),
        Mxfp4 => v!("native_id_mxfp4_paged"),
        Nvfp4 => v!("native_id_nvfp4_paged"),
        Tq1_0 => v!("native_id_tq1_0_paged"),
        Tq2_0 => v!("native_id_tq2_0_paged"),
        Q2_0 => v!("native_id_q2_0_paged"),
        Iq2Xxs => v!("native_id_iq2xxs_paged"),
        Iq2Xs => v!("native_id_iq2xs_paged"),
        Iq2S => v!("native_id_iq2s_paged"),
        Iq3Xxs => v!("native_id_iq3xxs_paged"),
        Iq3S => v!("native_id_iq3s_paged"),
        Iq1S => v!("native_id_iq1s_paged"),
        Iq1M => v!("native_id_iq1m_paged"),
        Bf16 => v!("native_id_bf16_paged"),
        F16 => v!("native_id_f16_paged"),
        F32 => v!("native_id_f32_paged"),
        _ => return None,
    })
}
/// [`native_idm_build_spv`]'s paged twin (`infr_vulkan::pager`) — one extra LUT-buffer binding.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_idm_paged_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
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
        Q8_0 => v!("native_idm_q8_0_paged"),
        Q4_0 => v!("native_idm_q4_0_paged"),
        Q4_1 => v!("native_idm_q4_1_paged"),
        Q5_0 => v!("native_idm_q5_0_paged"),
        Q5_1 => v!("native_idm_q5_1_paged"),
        Q2K => v!("native_idm_q2k_paged"),
        Q3K => v!("native_idm_q3k_paged"),
        Q4K => v!("native_idm_q4k_paged"),
        Q5K => v!("native_idm_q5k_paged"),
        Q6K => v!("native_idm_q6k_paged"),
        Iq4Nl => v!("native_idm_iq4nl_paged"),
        Iq4Xs => v!("native_idm_iq4xs_paged"),
        Mxfp4 => v!("native_idm_mxfp4_paged"),
        Nvfp4 => v!("native_idm_nvfp4_paged"),
        Tq1_0 => v!("native_idm_tq1_0_paged"),
        Tq2_0 => v!("native_idm_tq2_0_paged"),
        Q2_0 => v!("native_idm_q2_0_paged"),
        Iq2Xxs => v!("native_idm_iq2xxs_paged"),
        Iq2Xs => v!("native_idm_iq2xs_paged"),
        Iq2S => v!("native_idm_iq2s_paged"),
        Iq3Xxs => v!("native_idm_iq3xxs_paged"),
        Iq3S => v!("native_idm_iq3s_paged"),
        Iq1S => v!("native_idm_iq1s_paged"),
        Iq1M => v!("native_idm_iq1m_paged"),
        Bf16 => v!("native_idm_bf16_paged"),
        F16 => v!("native_idm_f16_paged"),
        F32 => v!("native_idm_f32_paged"),
        _ => return None,
    })
}
/// SPIR-V + kernel-cache name for the reassociation-tolerant subgroup+NR variant of the multi-slot
/// id GEMV (`native_gemv_id_multi_sg.comp`, wave32 + subgroupAdd). NOT bit-identical to
/// `native_idm_*` — reordered accumulation; the caller gates to the Q6_K projection band (see
/// `native_id_sg_choice`). Q6_K/Q5_K/IQ3_S only (Q4_K idm already saturates; IQ2_S's gate/up
/// shape REGRESSES on this tier). `nr` ∈ {2,4,8}.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_idm_sg_build_spv(
    dtype: infr_core::DType,
    nr: u32,
    sg16: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (dtype, nr, sg16) {
        (Q6K, 2, false) => v!("native_idm_q6k_sg2"),
        (Q6K, 4, false) => v!("native_idm_q6k_sg4"),
        (Q6K, 8, false) => v!("native_idm_q6k_sg8"),
        (Q5K, 2, false) => v!("native_idm_q5k_sg2"),
        (Q5K, 4, false) => v!("native_idm_q5k_sg4"),
        (Q5K, 8, false) => v!("native_idm_q5k_sg8"),
        (Q6K, 2, true) => v!("native_idm_q6k_sg2_sg16"),
        (Q6K, 4, true) => v!("native_idm_q6k_sg4_sg16"),
        (Q6K, 8, true) => v!("native_idm_q6k_sg8_sg16"),
        (Q5K, 2, true) => v!("native_idm_q5k_sg2_sg16"),
        (Q5K, 4, true) => v!("native_idm_q5k_sg4_sg16"),
        (Q5K, 8, true) => v!("native_idm_q5k_sg8_sg16"),
        (Iq3S, 2, false) => v!("native_idm_iq3s_sg2"),
        (Iq3S, 4, false) => v!("native_idm_iq3s_sg4"),
        (Iq3S, 8, false) => v!("native_idm_iq3s_sg8"),
        (Iq3S, 2, true) => v!("native_idm_iq3s_sg2_sg16"),
        (Iq3S, 4, true) => v!("native_idm_iq3s_sg4_sg16"),
        (Iq3S, 8, true) => v!("native_idm_iq3s_sg8_sg16"),
        _ => None,
    }
}

/// SPIR-V for the int8 dp4a decode GEMV (m=1, NUM_ROWS=2, `native_mmv.comp`). `None` = format
/// has no int-dot build (falls back to the dequant `native_gemv`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_build_spv(dtype: infr_core::DType, res: bool) -> Option<&'static [u32]> {
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
    Some(match (dtype, res) {
        (Q4K, false) => v!("native_mmv_q4k"),
        (Q4K, true) => v!("native_mmv_q4k_res"),
        (Q6K, false) => v!("native_mmv_q6k"),
        (Q6K, true) => v!("native_mmv_q6k_res"),
        (Iq4Xs, false) => v!("native_mmv_iq4xs"),
        (Iq4Xs, true) => v!("native_mmv_iq4xs_res"),
        _ => return None,
    })
}
/// SPIR-V + cache name for the multi-warp int8 dp4a decode GEMV (`native_mmv_mw.comp`, warp-per-row
/// subgroupAdd, `warps` rows/block). Gated by the adapter's `mmv_mw_choice` (default-on Intel,
/// opt-in elsewhere). `warps` ∈ {4, 8}. `sg16` selects the `-DSG=16` twin (`caps.sg_pref == 16`).
/// `None` for formats/warp counts without a build.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mw_build_spv(
    dtype: infr_core::DType,
    res: bool,
    warps: u32,
    sg16: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (dtype, res, warps, sg16) {
        // Dispatch-shape sweep builds (Q4_K only, WARPS ∈ {1,2,16} — no sg16 twin, AMD-only probe).
        (Q4K, false, 1, false) => v!("native_mmv_mw_q4k_w1"),
        (Q4K, true, 1, false) => v!("native_mmv_mw_q4k_w1_res"),
        (Q4K, false, 2, false) => v!("native_mmv_mw_q4k_w2"),
        (Q4K, true, 2, false) => v!("native_mmv_mw_q4k_w2_res"),
        (Q4K, false, 16, false) => v!("native_mmv_mw_q4k_w16"),
        (Q4K, true, 16, false) => v!("native_mmv_mw_q4k_w16_res"),
        (Q4K, false, 4, false) => v!("native_mmv_mw_q4k_w4"),
        (Q4K, true, 4, false) => v!("native_mmv_mw_q4k_w4_res"),
        (Q4K, false, 8, false) => v!("native_mmv_mw_q4k_w8"),
        (Q4K, true, 8, false) => v!("native_mmv_mw_q4k_w8_res"),
        (Q6K, false, 4, false) => v!("native_mmv_mw_q6k_w4"),
        (Q6K, true, 4, false) => v!("native_mmv_mw_q6k_w4_res"),
        (Q6K, false, 8, false) => v!("native_mmv_mw_q6k_w8"),
        (Q6K, true, 8, false) => v!("native_mmv_mw_q6k_w8_res"),
        (Q2K, false, 4, false) => v!("native_mmv_mw_q2k_w4"),
        (Q2K, true, 4, false) => v!("native_mmv_mw_q2k_w4_res"),
        (Q2K, false, 8, false) => v!("native_mmv_mw_q2k_w8"),
        (Q2K, true, 8, false) => v!("native_mmv_mw_q2k_w8_res"),
        (Q3K, false, 4, false) => v!("native_mmv_mw_q3k_w4"),
        (Q3K, true, 4, false) => v!("native_mmv_mw_q3k_w4_res"),
        (Q3K, false, 8, false) => v!("native_mmv_mw_q3k_w8"),
        (Q3K, true, 8, false) => v!("native_mmv_mw_q3k_w8_res"),
        (Q4K, false, 4, true) => v!("native_mmv_mw_q4k_w4_sg16"),
        (Q4K, true, 4, true) => v!("native_mmv_mw_q4k_w4_res_sg16"),
        (Q4K, false, 8, true) => v!("native_mmv_mw_q4k_w8_sg16"),
        (Q4K, true, 8, true) => v!("native_mmv_mw_q4k_w8_res_sg16"),
        (Q6K, false, 4, true) => v!("native_mmv_mw_q6k_w4_sg16"),
        (Q6K, true, 4, true) => v!("native_mmv_mw_q6k_w4_res_sg16"),
        (Q6K, false, 8, true) => v!("native_mmv_mw_q6k_w8_sg16"),
        (Q6K, true, 8, true) => v!("native_mmv_mw_q6k_w8_res_sg16"),
        (Q2K, false, 4, true) => v!("native_mmv_mw_q2k_w4_sg16"),
        (Q2K, true, 4, true) => v!("native_mmv_mw_q2k_w4_res_sg16"),
        (Q2K, false, 8, true) => v!("native_mmv_mw_q2k_w8_sg16"),
        (Q2K, true, 8, true) => v!("native_mmv_mw_q2k_w8_res_sg16"),
        (Q3K, false, 4, true) => v!("native_mmv_mw_q3k_w4_sg16"),
        (Q3K, true, 4, true) => v!("native_mmv_mw_q3k_w4_res_sg16"),
        (Q3K, false, 8, true) => v!("native_mmv_mw_q3k_w8_sg16"),
        (Q3K, true, 8, true) => v!("native_mmv_mw_q3k_w8_res_sg16"),
        // Q5_K: NEW int8 decode arm — {plain,res} x WARPS{4,8}, no sg16 twin (AMD-only
        // measurement; add an Intel SG=16 build before ever defaulting it on there).
        (Q5K, false, 4, false) => v!("native_mmv_mw_q5k_w4"),
        (Q5K, true, 4, false) => v!("native_mmv_mw_q5k_w4_res"),
        (Q5K, false, 8, false) => v!("native_mmv_mw_q5k_w8"),
        (Q5K, true, 8, false) => v!("native_mmv_mw_q5k_w8_res"),
        _ => None,
    }
}
/// SPIR-V for the multi-row int8 dp4a GEMV (m = 2..8, `native_mmv_mrow.comp`). `None` = format
/// has no int-dot build (falls back to the dequant `native_gemv_mrow`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mrow_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
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
        Q4K => v!("native_mmv_mrow_q4k"),
        Q6K => v!("native_mmv_mrow_q6k"),
        Iq4Xs => v!("native_mmv_mrow_iq4xs"),
        Q2K => v!("native_mmv_mrow_q2k"),
        Q3K => v!("native_mmv_mrow_q3k"),
        Q5K => v!("native_mmv_mrow_q5k"),
        Q8_0 => v!("native_mmv_mrow_q8_0"),
        Q4_0 => v!("native_mmv_mrow_q4_0"),
        Q5_0 => v!("native_mmv_mrow_q5_0"),
        Q4_1 => v!("native_mmv_mrow_q4_1"),
        Q5_1 => v!("native_mmv_mrow_q5_1"),
        Iq4Nl => v!("native_mmv_mrow_iq4nl"),
        _ => return None,
    })
}
/// Kernel-cache name for the multi-row int8 dp4a GEMV.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mrow_kernel_name(dtype: infr_core::DType) -> &'static str {
    use infr_core::DType::*;
    match dtype {
        Q4K => "native_mmv_mrow_q4k",
        Q6K => "native_mmv_mrow_q6k",
        Iq4Xs => "native_mmv_mrow_iq4xs",
        Q2K => "native_mmv_mrow_q2k",
        Q3K => "native_mmv_mrow_q3k",
        Q5K => "native_mmv_mrow_q5k",
        Q8_0 => "native_mmv_mrow_q8_0",
        Q4_0 => "native_mmv_mrow_q4_0",
        Q5_0 => "native_mmv_mrow_q5_0",
        Q4_1 => "native_mmv_mrow_q4_1",
        Q5_1 => "native_mmv_mrow_q5_1",
        Iq4Nl => "native_mmv_mrow_iq4nl",
        _ => unreachable!("native_mmv_mrow_kernel_name: gated by native_mmv_mrow_build_spv"),
    }
}
/// SPIR-V for a multi-row int8 dp4a GEMV layout variant: `o4` = the small-in_f 4-outputs ×
/// 16-K-lanes workgroup split (-DOUTS4), `m4` = the rows<=4 MR specialization (-DMRV=4), `res` =
/// the fused-residual decode build (-DUSE_RES, only ever paired with rows=1, hence requires
/// `m4=true`; see `Recorder::linear_mmv_mrow`'s gates and its doc on the rows=1 decode/verify
/// unification).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mrow_variant_spv(
    dtype: infr_core::DType,
    o4: bool,
    m4: bool,
    res: bool,
) -> Option<&'static [u32]> {
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
    Some(match (dtype, o4, m4, res) {
        (Q4K, false, false, false) => v!("native_mmv_mrow_q4k"),
        (Q4K, false, true, false) => v!("native_mmv_mrow_q4k_m4"),
        (Q4K, true, false, false) => v!("native_mmv_mrow_q4k_o4"),
        (Q4K, true, true, false) => v!("native_mmv_mrow_q4k_o4_m4"),
        (Q4K, false, true, true) => v!("native_mmv_mrow_q4k_m4_res"),
        (Q4K, true, true, true) => v!("native_mmv_mrow_q4k_o4_m4_res"),
        (Q6K, false, false, false) => v!("native_mmv_mrow_q6k"),
        (Q6K, false, true, false) => v!("native_mmv_mrow_q6k_m4"),
        (Q6K, true, false, false) => v!("native_mmv_mrow_q6k_o4"),
        (Q6K, true, true, false) => v!("native_mmv_mrow_q6k_o4_m4"),
        (Q6K, false, true, true) => v!("native_mmv_mrow_q6k_m4_res"),
        (Q6K, true, true, true) => v!("native_mmv_mrow_q6k_o4_m4_res"),
        (Iq4Xs, false, false, false) => v!("native_mmv_mrow_iq4xs"),
        (Iq4Xs, false, true, false) => v!("native_mmv_mrow_iq4xs_m4"),
        (Iq4Xs, true, false, false) => v!("native_mmv_mrow_iq4xs_o4"),
        (Iq4Xs, true, true, false) => v!("native_mmv_mrow_iq4xs_o4_m4"),
        (Q2K, false, false, false) => v!("native_mmv_mrow_q2k"),
        (Q2K, false, true, false) => v!("native_mmv_mrow_q2k_m4"),
        (Q2K, true, false, false) => v!("native_mmv_mrow_q2k_o4"),
        (Q2K, true, true, false) => v!("native_mmv_mrow_q2k_o4_m4"),
        (Q2K, false, true, true) => v!("native_mmv_mrow_q2k_m4_res"),
        (Q2K, true, true, true) => v!("native_mmv_mrow_q2k_o4_m4_res"),
        (Q3K, false, false, false) => v!("native_mmv_mrow_q3k"),
        (Q3K, false, true, false) => v!("native_mmv_mrow_q3k_m4"),
        (Q3K, true, false, false) => v!("native_mmv_mrow_q3k_o4"),
        (Q3K, true, true, false) => v!("native_mmv_mrow_q3k_o4_m4"),
        (Q3K, false, true, true) => v!("native_mmv_mrow_q3k_m4_res"),
        (Q3K, true, true, true) => v!("native_mmv_mrow_q3k_o4_m4_res"),
        (Q5K, false, false, false) => v!("native_mmv_mrow_q5k"),
        (Q5K, false, true, false) => v!("native_mmv_mrow_q5k_m4"),
        (Q5K, true, false, false) => v!("native_mmv_mrow_q5k_o4"),
        (Q5K, true, true, false) => v!("native_mmv_mrow_q5k_o4_m4"),
        (Q5K, false, true, true) => v!("native_mmv_mrow_q5k_m4_res"),
        (Q5K, true, true, true) => v!("native_mmv_mrow_q5k_o4_m4_res"),
        (Q8_0, false, false, false) => v!("native_mmv_mrow_q8_0"),
        (Q8_0, false, true, false) => v!("native_mmv_mrow_q8_0_m4"),
        (Q8_0, true, false, false) => v!("native_mmv_mrow_q8_0_o4"),
        (Q8_0, true, true, false) => v!("native_mmv_mrow_q8_0_o4_m4"),
        (Q8_0, false, true, true) => v!("native_mmv_mrow_q8_0_m4_res"),
        (Q8_0, true, true, true) => v!("native_mmv_mrow_q8_0_o4_m4_res"),
        (Q4_0, false, false, false) => v!("native_mmv_mrow_q4_0"),
        (Q4_0, false, true, false) => v!("native_mmv_mrow_q4_0_m4"),
        (Q4_0, true, false, false) => v!("native_mmv_mrow_q4_0_o4"),
        (Q4_0, true, true, false) => v!("native_mmv_mrow_q4_0_o4_m4"),
        (Q4_0, false, true, true) => v!("native_mmv_mrow_q4_0_m4_res"),
        (Q4_0, true, true, true) => v!("native_mmv_mrow_q4_0_o4_m4_res"),
        (Q5_0, false, false, false) => v!("native_mmv_mrow_q5_0"),
        (Q5_0, false, true, false) => v!("native_mmv_mrow_q5_0_m4"),
        (Q5_0, true, false, false) => v!("native_mmv_mrow_q5_0_o4"),
        (Q5_0, true, true, false) => v!("native_mmv_mrow_q5_0_o4_m4"),
        (Q5_0, false, true, true) => v!("native_mmv_mrow_q5_0_m4_res"),
        (Q5_0, true, true, true) => v!("native_mmv_mrow_q5_0_o4_m4_res"),
        (Q4_1, false, false, false) => v!("native_mmv_mrow_q4_1"),
        (Q4_1, false, true, false) => v!("native_mmv_mrow_q4_1_m4"),
        (Q4_1, true, false, false) => v!("native_mmv_mrow_q4_1_o4"),
        (Q4_1, true, true, false) => v!("native_mmv_mrow_q4_1_o4_m4"),
        (Q4_1, false, true, true) => v!("native_mmv_mrow_q4_1_m4_res"),
        (Q4_1, true, true, true) => v!("native_mmv_mrow_q4_1_o4_m4_res"),
        (Q5_1, false, false, false) => v!("native_mmv_mrow_q5_1"),
        (Q5_1, false, true, false) => v!("native_mmv_mrow_q5_1_m4"),
        (Q5_1, true, false, false) => v!("native_mmv_mrow_q5_1_o4"),
        (Q5_1, true, true, false) => v!("native_mmv_mrow_q5_1_o4_m4"),
        (Q5_1, false, true, true) => v!("native_mmv_mrow_q5_1_m4_res"),
        (Q5_1, true, true, true) => v!("native_mmv_mrow_q5_1_o4_m4_res"),
        (Iq4Nl, false, false, false) => v!("native_mmv_mrow_iq4nl"),
        (Iq4Nl, false, true, false) => v!("native_mmv_mrow_iq4nl_m4"),
        (Iq4Nl, true, false, false) => v!("native_mmv_mrow_iq4nl_o4"),
        (Iq4Nl, true, true, false) => v!("native_mmv_mrow_iq4nl_o4_m4"),
        (Iq4Nl, false, true, true) => v!("native_mmv_mrow_iq4nl_m4_res"),
        (Iq4Nl, true, true, true) => v!("native_mmv_mrow_iq4nl_o4_m4_res"),
        _ => return None,
    })
}
/// Kernel-cache name for a multi-row int8 dp4a GEMV layout variant.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mrow_variant_name(
    dtype: infr_core::DType,
    o4: bool,
    m4: bool,
    res: bool,
) -> &'static str {
    use infr_core::DType::*;
    match (dtype, o4, m4, res) {
        (Q4K, false, false, false) => "native_mmv_mrow_q4k",
        (Q4K, false, true, false) => "native_mmv_mrow_q4k_m4",
        (Q4K, true, false, false) => "native_mmv_mrow_q4k_o4",
        (Q4K, true, true, false) => "native_mmv_mrow_q4k_o4_m4",
        (Q4K, false, true, true) => "native_mmv_mrow_q4k_m4_res",
        (Q4K, true, true, true) => "native_mmv_mrow_q4k_o4_m4_res",
        (Q6K, false, false, false) => "native_mmv_mrow_q6k",
        (Q6K, false, true, false) => "native_mmv_mrow_q6k_m4",
        (Q6K, true, false, false) => "native_mmv_mrow_q6k_o4",
        (Q6K, true, true, false) => "native_mmv_mrow_q6k_o4_m4",
        (Q6K, false, true, true) => "native_mmv_mrow_q6k_m4_res",
        (Q6K, true, true, true) => "native_mmv_mrow_q6k_o4_m4_res",
        (Iq4Xs, false, false, false) => "native_mmv_mrow_iq4xs",
        (Iq4Xs, false, true, false) => "native_mmv_mrow_iq4xs_m4",
        (Iq4Xs, true, false, false) => "native_mmv_mrow_iq4xs_o4",
        (Iq4Xs, true, true, false) => "native_mmv_mrow_iq4xs_o4_m4",
        (Q2K, false, false, false) => "native_mmv_mrow_q2k",
        (Q2K, false, true, false) => "native_mmv_mrow_q2k_m4",
        (Q2K, true, false, false) => "native_mmv_mrow_q2k_o4",
        (Q2K, true, true, false) => "native_mmv_mrow_q2k_o4_m4",
        (Q2K, false, true, true) => "native_mmv_mrow_q2k_m4_res",
        (Q2K, true, true, true) => "native_mmv_mrow_q2k_o4_m4_res",
        (Q3K, false, false, false) => "native_mmv_mrow_q3k",
        (Q3K, false, true, false) => "native_mmv_mrow_q3k_m4",
        (Q3K, true, false, false) => "native_mmv_mrow_q3k_o4",
        (Q3K, true, true, false) => "native_mmv_mrow_q3k_o4_m4",
        (Q3K, false, true, true) => "native_mmv_mrow_q3k_m4_res",
        (Q3K, true, true, true) => "native_mmv_mrow_q3k_o4_m4_res",
        (Q5K, false, false, false) => "native_mmv_mrow_q5k",
        (Q5K, false, true, false) => "native_mmv_mrow_q5k_m4",
        (Q5K, true, false, false) => "native_mmv_mrow_q5k_o4",
        (Q5K, true, true, false) => "native_mmv_mrow_q5k_o4_m4",
        (Q5K, false, true, true) => "native_mmv_mrow_q5k_m4_res",
        (Q5K, true, true, true) => "native_mmv_mrow_q5k_o4_m4_res",
        (Q8_0, false, false, false) => "native_mmv_mrow_q8_0",
        (Q8_0, false, true, false) => "native_mmv_mrow_q8_0_m4",
        (Q8_0, true, false, false) => "native_mmv_mrow_q8_0_o4",
        (Q8_0, true, true, false) => "native_mmv_mrow_q8_0_o4_m4",
        (Q8_0, false, true, true) => "native_mmv_mrow_q8_0_m4_res",
        (Q8_0, true, true, true) => "native_mmv_mrow_q8_0_o4_m4_res",
        (Q4_0, false, false, false) => "native_mmv_mrow_q4_0",
        (Q4_0, false, true, false) => "native_mmv_mrow_q4_0_m4",
        (Q4_0, true, false, false) => "native_mmv_mrow_q4_0_o4",
        (Q4_0, true, true, false) => "native_mmv_mrow_q4_0_o4_m4",
        (Q4_0, false, true, true) => "native_mmv_mrow_q4_0_m4_res",
        (Q4_0, true, true, true) => "native_mmv_mrow_q4_0_o4_m4_res",
        (Q5_0, false, false, false) => "native_mmv_mrow_q5_0",
        (Q5_0, false, true, false) => "native_mmv_mrow_q5_0_m4",
        (Q5_0, true, false, false) => "native_mmv_mrow_q5_0_o4",
        (Q5_0, true, true, false) => "native_mmv_mrow_q5_0_o4_m4",
        (Q5_0, false, true, true) => "native_mmv_mrow_q5_0_m4_res",
        (Q5_0, true, true, true) => "native_mmv_mrow_q5_0_o4_m4_res",
        (Q4_1, false, false, false) => "native_mmv_mrow_q4_1",
        (Q4_1, false, true, false) => "native_mmv_mrow_q4_1_m4",
        (Q4_1, true, false, false) => "native_mmv_mrow_q4_1_o4",
        (Q4_1, true, true, false) => "native_mmv_mrow_q4_1_o4_m4",
        (Q4_1, false, true, true) => "native_mmv_mrow_q4_1_m4_res",
        (Q4_1, true, true, true) => "native_mmv_mrow_q4_1_o4_m4_res",
        (Q5_1, false, false, false) => "native_mmv_mrow_q5_1",
        (Q5_1, false, true, false) => "native_mmv_mrow_q5_1_m4",
        (Q5_1, true, false, false) => "native_mmv_mrow_q5_1_o4",
        (Q5_1, true, true, false) => "native_mmv_mrow_q5_1_o4_m4",
        (Q5_1, false, true, true) => "native_mmv_mrow_q5_1_m4_res",
        (Q5_1, true, true, true) => "native_mmv_mrow_q5_1_o4_m4_res",
        (Iq4Nl, false, false, false) => "native_mmv_mrow_iq4nl",
        (Iq4Nl, false, true, false) => "native_mmv_mrow_iq4nl_m4",
        (Iq4Nl, true, false, false) => "native_mmv_mrow_iq4nl_o4",
        (Iq4Nl, true, true, false) => "native_mmv_mrow_iq4nl_o4_m4",
        (Iq4Nl, false, true, true) => "native_mmv_mrow_iq4nl_m4_res",
        (Iq4Nl, true, true, true) => "native_mmv_mrow_iq4nl_o4_m4_res",
        _ => unreachable!("native_mmv_mrow_variant_name: gated by native_mmv_mrow_build_spv"),
    }
}
/// SPIR-V for the rows 9..=16 multi-row int8 dp4a GEMV tier (`-DMRV=16`, 2-output layout — no
/// OUTS4 twin: the tier is gated to >= 8M-element weights, whose in_f is comfortably >= 2048).
/// `None` = format not covered (same coverage as [`native_mmv_mrow_build_spv`]; those shapes stay
/// on the split-K GEMM tile).
pub(crate) fn native_mmv_mrow_m16_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
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
        Q4K => v!("native_mmv_mrow_q4k_m16"),
        Q6K => v!("native_mmv_mrow_q6k_m16"),
        Iq4Xs => v!("native_mmv_mrow_iq4xs_m16"),
        Q2K => v!("native_mmv_mrow_q2k_m16"),
        Q3K => v!("native_mmv_mrow_q3k_m16"),
        Q5K => v!("native_mmv_mrow_q5k_m16"),
        Q8_0 => v!("native_mmv_mrow_q8_0_m16"),
        Q4_0 => v!("native_mmv_mrow_q4_0_m16"),
        Q5_0 => v!("native_mmv_mrow_q5_0_m16"),
        Q4_1 => v!("native_mmv_mrow_q4_1_m16"),
        Q5_1 => v!("native_mmv_mrow_q5_1_m16"),
        Iq4Nl => v!("native_mmv_mrow_iq4nl_m16"),
        _ => return None,
    })
}
/// Kernel-cache name for the rows 9..=16 multi-row int8 dp4a GEMV tier.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mrow_m16_name(dtype: infr_core::DType) -> &'static str {
    use infr_core::DType::*;
    match dtype {
        Q4K => "native_mmv_mrow_q4k_m16",
        Q6K => "native_mmv_mrow_q6k_m16",
        Iq4Xs => "native_mmv_mrow_iq4xs_m16",
        Q2K => "native_mmv_mrow_q2k_m16",
        Q3K => "native_mmv_mrow_q3k_m16",
        Q5K => "native_mmv_mrow_q5k_m16",
        Q8_0 => "native_mmv_mrow_q8_0_m16",
        Q4_0 => "native_mmv_mrow_q4_0_m16",
        Q5_0 => "native_mmv_mrow_q5_0_m16",
        Q4_1 => "native_mmv_mrow_q4_1_m16",
        Q5_1 => "native_mmv_mrow_q5_1_m16",
        Iq4Nl => "native_mmv_mrow_iq4nl_m16",
        _ => unreachable!("native_mmv_mrow_m16_name: gated by native_mmv_mrow_m16_spv"),
    }
}
/// Kernel-cache name for the int8 dp4a decode GEMV.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_kernel_name(dtype: infr_core::DType, res: bool) -> &'static str {
    use infr_core::DType::*;
    match (dtype, res) {
        (Q4K, false) => "native_mmv_q4k",
        (Q4K, true) => "native_mmv_q4k_res",
        (Q6K, false) => "native_mmv_q6k",
        (Q6K, true) => "native_mmv_q6k_res",
        (Iq4Xs, false) => "native_mmv_iq4xs",
        (Iq4Xs, true) => "native_mmv_iq4xs_res",
        _ => unreachable!("native_mmv_kernel_name: gated by native_mmv_build_spv"),
    }
}
/// SPIR-V for the multi-slot id-indexed Q4_K dp4a (mmq) GEMV.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_id_q4k_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_mmv_id_q4k.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the tiled Q4_K dp4a (mmq) GEMM.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the tiled Q6_K dp4a (mmq) GEMM (the MoE down projection).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// Dense (non-expert-grid) dp4a mmq GEMM SPIR-V for every `infr_core::tensor::MOE_MMQ_DTYPES`
/// member — the non-coopmat prefill tier's kernel table (see adapter.rs `nc_mmq`: devices
/// without a usable 16x16x16 f16 coopmat but WITH packed int8 dot, e.g. Intel Arc/ANV). Returns
/// `(kernel_cache_name, spv)`; `None` for non-mmq dtypes. Same shader sources as the
/// parity-proven `_xp` expert-grid builds — base compile, dense grid. Whether the kernel binds
/// the activation `sact` buffer follows `infr_core::tensor::moe_mmq_needs_sact` (the SSOT the
/// `_xp` dispatch uses too).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_dense_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
    macro_rules! spv {
        ($name:literal) => {{
            const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"));
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            ($name, S.get_or_init(|| spv_words(BYTES)).as_slice())
        }};
    }
    use infr_core::DType::*;
    Some(match dtype {
        Q4K => ("native_gemm_mmq_q4k", native_gemm_mmq_q4k_spv()),
        Q6K => ("native_gemm_mmq_q6k", native_gemm_mmq_q6k_spv()),
        Q8_0 => spv!("native_gemm_mmq_q8_0"),
        Q5_0 => spv!("native_gemm_mmq_q5_0"),
        Q5K => spv!("native_gemm_mmq_q5k"),
        Q5_1 => spv!("native_gemm_mmq_q5_1"),
        Q2K => spv!("native_gemm_mmq_q2_k"),
        Q3K => spv!("native_gemm_mmq_q3_k"),
        Q4_0 => spv!("native_gemm_mmq_q4_0"),
        Q4_1 => spv!("native_gemm_mmq_q4_1"),
        Iq4Nl => spv!("native_gemm_mmq_iq4_nl"),
        Iq4Xs => spv!("native_gemm_mmq_iq4_xs"),
        Iq2S => spv!("native_gemm_mmq_iq2_s"),
        Iq3S => spv!("native_gemm_mmq_iq3_s"),
        Mxfp4 => spv!("native_gemm_mmq_mxfp4"),
        Nvfp4 => spv!("native_gemm_mmq_nvfp4"),
        Q2_0 => spv!("native_gemm_mmq_q2_0"),
        _ => return None,
    })
}
/// SPIR-V for the non-coopmat float-weight prefill GEMM (the "fma-warp" tier, see
/// `native_gemm_fma.comp`): shared-memory fma warptile for f16/bf16/f32 weights on devices
/// without a usable f16 coopmat (adapter.rs `nc_fma`). Returns `(kernel_cache_name, spv)`;
/// `None` for any other dtype.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_fma_build_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
    macro_rules! spv {
        ($name:literal) => {{
            const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"));
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            ($name, S.get_or_init(|| spv_words(BYTES)).as_slice())
        }};
    }
    use infr_core::DType::*;
    Some(match dtype {
        F16 => spv!("native_gemm_fma_f16"),
        Bf16 => spv!("native_gemm_fma_bf16"),
        F32 => spv!("native_gemm_fma_f32"),
        _ => return None,
    })
}
/// SPIR-V for the non-coopmat shared-memory fma flash-attention prefill (the "nc_fa" tier, see
/// `attn_nc_fa.comp`): fused QK^T → online softmax → PV for devices without a usable f16 coopmat
/// (adapter.rs `nc_fa_ok`). Three shared-Os builds; returns the smallest that fits `hd` as
/// `(kernel_cache_name, spv, BM row tile)`, or `None` for hd > 512.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_nc_fa_spv(hd: usize) -> Option<(&'static str, &'static [u32], usize)> {
    macro_rules! spv {
        ($name:literal, $bm:literal) => {{
            const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"));
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            ($name, S.get_or_init(|| spv_words(BYTES)).as_slice(), $bm)
        }};
    }
    Some(if hd <= 128 {
        spv!("attn_nc_fa_hd128", 32)
    } else if hd <= 256 {
        spv!("attn_nc_fa_hd256", 32)
    } else if hd <= 512 {
        spv!("attn_nc_fa_hd512", 16)
    } else {
        return None;
    })
}
/// SPIR-V for the int8 cooperative-matrix (WMMA) prefill GEMM, Q8_0 only — measurement kernel
/// gated behind `INFR_I8_COOPMAT=1` (see `native_gemm_i8cm_q8_0.comp` for the design doc).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_i8cm_q8_0_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_i8cm_q8_0.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V + cache name for the row-wise (whole-K) activation quant pass — int8-coopmat GEMM
/// "Idea 2" measurement variant (see `quant_q8_row.comp`), gated behind `INFR_I8_ROW_SCALE=1`.
/// `sg16` selects the `-DSG=16` twin (`caps.sg_pref == 16`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quant_q8_row_build_spv(sg16: bool) -> (&'static str, &'static [u32]) {
    if sg16 {
        const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quant_q8_row_sg16.spv"));
        static S: OnceLock<Vec<u32>> = OnceLock::new();
        ("quant_q8_row_sg16", S.get_or_init(|| spv_words(BYTES)))
    } else {
        const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quant_q8_row.spv"));
        static S: OnceLock<Vec<u32>> = OnceLock::new();
        ("quant_q8_row", S.get_or_init(|| spv_words(BYTES)))
    }
}
/// SPIR-V for the fp8 (E4M3) cooperative-matrix (WMMA) prefill GEMM, Q8_0 only, WIDE tile
/// (BM=64xBN=256, same warptile shape as `native_gemm_warp`) — gated behind `INFR_F8_COOPMAT=1` +
/// `caps.f8_coopmat` (see `native_gemm_f8cm_q8_0.comp` for the design doc).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_f8cm_q8_0_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_f8cm_q8_0.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the fp8-coopmat GEMM's NARROW_N tile (BM=64xBN=128, BK=64) — the occupancy fix for
/// n%128 (not n%256) shapes, mirroring `native_gemm_warp_n128_build_spv`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_f8cm_q8_0_n128_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_f8cm_q8_0_n128.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the row-wise (whole-K) fp8 activation quant pass — the activation-scaling step for
/// `native_gemm_f8cm_q8_0` (see `quant_f8_row.comp`), gated behind `INFR_F8_COOPMAT=1` +
/// `caps.f8_coopmat`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quant_f8_row_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quant_f8_row.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the `-DPREPACK` fp8-coopmat GEMM WIDE tile: reads a pre-packed E4M3 weight buffer
/// directly (no in-shader Q8_0 dequant) — the measurement variant for whether removing the dqblk
/// bottleneck lets fp8 beat f16 (see `native_gemm_f8cm_q8_0.comp` header + `repack_q8_to_f8.comp`).
/// Gated behind `INFR_F8_COOPMAT=1` + `INFR_F8_PREPACK=1` + `caps.f8_coopmat`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_f8cm_q8_0_prepack_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_f8cm_q8_0_prepack.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the `-DPREPACK` fp8-coopmat GEMM's NARROW_N tile (BM=64xBN=128, BK=64) — the n%128
/// occupancy fix, mirroring `native_gemm_f8cm_q8_0_n128_spv` but reading pre-packed E4M3 weights.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_f8cm_q8_0_prepack_n128_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_f8cm_q8_0_prepack_n128.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the NATIVE bf16 cooperative-matrix (WMMA) variant of the production
/// `native_gemm_warp` kernel, WIDE tile (BM=64xBN=256) — `-DFMT_BF16 -DBF16CM`, gated behind
/// `INFR_BF16_COOPMAT=1` + `caps.bf16_coopmat` (see `native_gemm_warp.comp`'s BF16CM doc). Same
/// tuned warptile as `native_gemm_warp_bf16_spv`'s f16-clamped path, just bf16-typed coopmat
/// operands — replaces the old standalone `native_gemm_bf16cm.comp` measurement kernel.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_bf16cm_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_warp_bf16cm.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the bf16-coopmat warptile's NARROW_N tile (BM=64xBN=128, BK=64) — the occupancy fix
/// for n%128 (not n%256) shapes, mirroring `native_gemm_warp_bf16cm_spv`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_bf16cm_n128_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_warp_bf16cm_n128.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for `repack_q8_to_f8.comp`: bakes each Q8_0 32-block's scale into an E4M3 output
/// (decode-once via `dqblk`), producing the pre-packed weight buffer the PREPACK GEMM variants
/// above read directly. Gated behind `INFR_F8_COOPMAT=1` + `INFR_F8_PREPACK=1`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn repack_q8_to_f8_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/repack_q8_to_f8.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the int8-coopmat GEMM's "Idea 2" whole-row-activation-scale measurement variant
/// (see `native_gemm_i8cm_q8_0.comp` #ifdef ROW_SCALE), gated behind `INFR_I8_ROW_SCALE=1`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_i8cm_q8_0_rowscale_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_i8cm_q8_0_rowscale.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the MoE weighted-accumulate (sum of selected experts' down outputs into hidden).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_accumulate_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_accumulate.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the MoE weighted-accumulate with a per-expert down-output scale (diffusion-gemma
/// `ffn_down_exps.scale`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_accumulate_scaled_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_accumulate_scaled.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the GPU MoE router top-k.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_topk_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_topk.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the embedding-row gather+dequant (`Op::EmbedGather`). `None` = format has no
/// build (grid-table IQ formats) — the runner then keeps the host embed path.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn embed_gather_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
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
        Q8_0 => v!("embed_gather_q8_0"),
        Bf16 => v!("embed_gather_bf16"),
        F16 => v!("embed_gather_f16"),
        Q4_0 => v!("embed_gather_q4_0"),
        Q4_1 => v!("embed_gather_q4_1"),
        Q5_0 => v!("embed_gather_q5_0"),
        Q5_1 => v!("embed_gather_q5_1"),
        Q2K => v!("embed_gather_q2k"),
        Q3K => v!("embed_gather_q3k"),
        Q4K => v!("embed_gather_q4k"),
        Q5K => v!("embed_gather_q5k"),
        Q6K => v!("embed_gather_q6k"),
        Iq4Nl => v!("embed_gather_iq4nl"),
        Iq4Xs => v!("embed_gather_iq4xs"),
        Q2_0 => v!("embed_gather_q2_0"),
        _ => return None,
    })
}
/// Kernel-cache name for the embedding-row gather.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn embed_gather_kernel_name(dtype: infr_core::DType) -> &'static str {
    use infr_core::DType::*;
    match dtype {
        Q8_0 => "embed_gather_q8_0",
        Bf16 => "embed_gather_bf16",
        F16 => "embed_gather_f16",
        Q4_0 => "embed_gather_q4_0",
        Q4_1 => "embed_gather_q4_1",
        Q5_0 => "embed_gather_q5_0",
        Q5_1 => "embed_gather_q5_1",
        Q2K => "embed_gather_q2k",
        Q3K => "embed_gather_q3k",
        Q4K => "embed_gather_q4k",
        Q5K => "embed_gather_q5k",
        Q6K => "embed_gather_q6k",
        Iq4Nl => "embed_gather_iq4nl",
        Iq4Xs => "embed_gather_iq4xs",
        Q2_0 => "embed_gather_q2_0",
        _ => unreachable!("embed_gather_kernel_name: gated by embed_gather_build_spv"),
    }
}
/// SPIR-V for the chained-decode id ring log (ring[pos & 63] = sampled id).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn id_log_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/id_log.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the decode-replay params advance (device-side [pos, kv_len] increment).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn params_advance_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/params_advance.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the vocab sampler's slice pass (256 workgroups → 256*k (val, idx) candidates).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn sample_topk_part_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sample_topk_part.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the vocab sampler's select+softmax+nucleus+CDF pass (candidates → token id).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn sample_topk_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sample_topk.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the vocab sampler's chained-decode select+softmax+nucleus+CDF pass: `u` is read
/// from the 64-slot ring at `u_buf[params[0] & 63]` instead of `u_buf[0]` (see sample_topk.comp).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn sample_topk_chain_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sample_topk_chain.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the greedy-argmax slice pass (256 workgroups → 256 (val, idx) partials).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn argmax_part_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax_part.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the greedy-argmax reduce pass (256 partials → token id).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn argmax_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the fused argmax+top1-prob slice pass (256 workgroups → 256 (max, idx, sum_exp)
/// partials) — `Op::ArgmaxProb`, the MTP draft-loop accept.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn argmax_prob_part_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax_prob_part.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the fused argmax+top1-prob reduce pass (256 partials → token id + top1 probability).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn argmax_prob_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax_prob.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for GPU stochastic sampling (radix top-k + temp + top-p → token id).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_sample_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_sample.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the DiffusionGemma entropy-bound sampler reduction (perf slice 3): per-canvas-row
/// argmax/entropy/CDF-sample over `[rows, vocab]` logits — see `shaders/dg_eb_sample.comp`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn dg_eb_sample_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dg_eb_sample.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// Max `top_k` the GPU sampler supports (matches the shader's KMAX); larger falls back to host.
pub const SAMPLE_KMAX: usize = 64;
/// SPIR-V for the MoE expert-bucketing passes (count / exclusive-scan / scatter).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_bucket_count_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_bucket_count.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_bucket_scan_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_bucket_scan.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_bucket_scatter_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_bucket_scatter.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for `moe_bucket_scatter`'s per-expert-`dscale`-baking variant (diffusion-gemma).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_bucket_scatter_scaled_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_bucket_scatter_scaled.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the indexed axpy (`acc += wts[slot]*x`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn add_scaled_id_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_scaled_id.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for llama4's weight-before-FFN in-place per-row scale (`Op::MoeFfn`'s `weight_before`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_weight_scale_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_weight_scale.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the LARGE-WARPTILE native-block prefill GEMM (8-warp BM=64×BN=256, gemm_proj_warp
/// structure with in-shader native dequant). Only the hot formats are compiled; `None` falls back
/// to the 64×64 `native_gemm_build_spv` kernel.
#[cfg_attr(infr_profile, infr_prof::instrument)]
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
        Bf16 => v!("native_gemm_warp_bf16"),
        Iq4Nl => v!("native_gemm_warp_iq4nl"),
        Iq4Xs => v!("native_gemm_warp_iq4xs"),
        Q2K => v!("native_gemm_warp_q2k"),
        Q3K => v!("native_gemm_warp_q3k"),
        Q5_0 => v!("native_gemm_warp_q5_0"),
        Q5_1 => v!("native_gemm_warp_q5_1"),
        Q4_0 => v!("native_gemm_warp_q4_0"),
        Q4K => v!("native_gemm_warp_q4k"),
        Q5K => v!("native_gemm_warp_q5k"),
        Q6K => v!("native_gemm_warp_q6k"),
        Q8_0 => v!("native_gemm_warp_q8_0"),
        _ => return None,
    })
}

/// SPIR-V for the NARROW-N warptile (BN=128/BK=64) — same math per thread, 2× the workgroups; the
/// occupancy fix for n=1024/2048 GEMMs. `None` for formats without a warp build.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_n128_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
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
        Bf16 => v!("native_gemm_warp_bf16_n128"),
        Iq4Nl => v!("native_gemm_warp_iq4nl_n128"),
        Iq4Xs => v!("native_gemm_warp_iq4xs_n128"),
        Q2K => v!("native_gemm_warp_q2k_n128"),
        Q3K => v!("native_gemm_warp_q3k_n128"),
        Q5_0 => v!("native_gemm_warp_q5_0_n128"),
        Q5_1 => v!("native_gemm_warp_q5_1_n128"),
        Q4_0 => v!("native_gemm_warp_q4_0_n128"),
        Q4K => v!("native_gemm_warp_q4k_n128"),
        Q5K => v!("native_gemm_warp_q5k_n128"),
        Q6K => v!("native_gemm_warp_q6k_n128"),
        Q8_0 => v!("native_gemm_warp_q8_0_n128"),
        _ => return None,
    })
}

/// 8x8x16-fragment `_cm8` warptile variants (Intel Arc/ANV XMX; `-DCM_M=8 -DCM_N=8` on the
/// NARROW_N+BM32 tile, 128 threads / pinned subgroup 16) — dispatched ONLY when the device's
/// selected f16 coopmat shape is 8x8x16 (`caps.f16_coopmat_8x8()`, which itself requires the
/// `INFR_CM_8X8=1` opt-in; see lib.rs `select_coopmat_shape`). Name+SPIR-V so `kernel_sg` caches
/// distinct pipelines. Hot k-quants + Q8_0 only; `None` routes the shape to the nc_mmq/nc_fma
/// non-coopmat tier (the Arc default).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_cm8_build_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            (
                $name,
                S.get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice(),
            )
        }};
    }
    Some(match dtype {
        Q4K => v!("native_gemm_warp_q4k_cm8"),
        Q6K => v!("native_gemm_warp_q6k_cm8"),
        Q8_0 => v!("native_gemm_warp_q8_0_cm8"),
        _ => return None,
    })
}

/// A_GLOBAL warptile variants: A pre-converted to f16 by the caller and coopMatLoad'd straight
/// from global memory — no As staging, no As LDS. Shrinking LDS to Bs-only lifts occupancy from
/// 2 to 3 workgroups/WGP, which is worth ~1.5x on the 8B prefill shapes (29→44 TF on the o
/// projection). Name+SPIR-V per tile so `kernel_sg` caches distinct pipelines.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_ag_build_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
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
        Iq4Nl => ("native_gemm_warp_iq4nl_ag", v!("native_gemm_warp_iq4nl_ag")),
        Iq4Xs => ("native_gemm_warp_iq4xs_ag", v!("native_gemm_warp_iq4xs_ag")),
        Q2K => ("native_gemm_warp_q2k_ag", v!("native_gemm_warp_q2k_ag")),
        Q3K => ("native_gemm_warp_q3k_ag", v!("native_gemm_warp_q3k_ag")),
        Q4_0 => ("native_gemm_warp_q4_0_ag", v!("native_gemm_warp_q4_0_ag")),
        Q4K => ("native_gemm_warp_q4k_ag", v!("native_gemm_warp_q4k_ag")),
        Q5K => ("native_gemm_warp_q5k_ag", v!("native_gemm_warp_q5k_ag")),
        Q6K => ("native_gemm_warp_q6k_ag", v!("native_gemm_warp_q6k_ag")),
        Q8_0 => ("native_gemm_warp_q8_0_ag", v!("native_gemm_warp_q8_0_ag")),
        _ => return None,
    })
}

/// NARROW_N (BN=128) + A_GLOBAL.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_n128_ag_build_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
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
        Iq4Nl => (
            "native_gemm_warp_iq4nl_n128_ag",
            v!("native_gemm_warp_iq4nl_n128_ag"),
        ),
        Iq4Xs => (
            "native_gemm_warp_iq4xs_n128_ag",
            v!("native_gemm_warp_iq4xs_n128_ag"),
        ),
        Q2K => (
            "native_gemm_warp_q2k_n128_ag",
            v!("native_gemm_warp_q2k_n128_ag"),
        ),
        Q3K => (
            "native_gemm_warp_q3k_n128_ag",
            v!("native_gemm_warp_q3k_n128_ag"),
        ),
        Q4_0 => (
            "native_gemm_warp_q4_0_n128_ag",
            v!("native_gemm_warp_q4_0_n128_ag"),
        ),
        Q4K => (
            "native_gemm_warp_q4k_n128_ag",
            v!("native_gemm_warp_q4k_n128_ag"),
        ),
        Q5K => (
            "native_gemm_warp_q5k_n128_ag",
            v!("native_gemm_warp_q5k_n128_ag"),
        ),
        Q6K => (
            "native_gemm_warp_q6k_n128_ag",
            v!("native_gemm_warp_q6k_n128_ag"),
        ),
        Q8_0 => (
            "native_gemm_warp_q8_0_n128_ag",
            v!("native_gemm_warp_q8_0_n128_ag"),
        ),
        _ => return None,
    })
}

/// SPLIT_K (NARROW_N tile) + A_GLOBAL.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_sk_ag_build_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
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
        Iq4Nl => (
            "native_gemm_warp_iq4nl_sk_ag",
            v!("native_gemm_warp_iq4nl_sk_ag"),
        ),
        Iq4Xs => (
            "native_gemm_warp_iq4xs_sk_ag",
            v!("native_gemm_warp_iq4xs_sk_ag"),
        ),
        Q2K => (
            "native_gemm_warp_q2k_sk_ag",
            v!("native_gemm_warp_q2k_sk_ag"),
        ),
        Q3K => (
            "native_gemm_warp_q3k_sk_ag",
            v!("native_gemm_warp_q3k_sk_ag"),
        ),
        Q4_0 => (
            "native_gemm_warp_q4_0_sk_ag",
            v!("native_gemm_warp_q4_0_sk_ag"),
        ),
        Q4K => (
            "native_gemm_warp_q4k_sk_ag",
            v!("native_gemm_warp_q4k_sk_ag"),
        ),
        Q5K => (
            "native_gemm_warp_q5k_sk_ag",
            v!("native_gemm_warp_q5k_sk_ag"),
        ),
        Q6K => (
            "native_gemm_warp_q6k_sk_ag",
            v!("native_gemm_warp_q6k_sk_ag"),
        ),
        Q8_0 => (
            "native_gemm_warp_q8_0_sk_ag",
            v!("native_gemm_warp_q8_0_sk_ag"),
        ),
        _ => return None,
    })
}

/// BM=32 row-tile variant of [`native_gemm_warp_n128_ag_build_spv`] — see the recorder's
/// `DENSE_SMALL_TILE_MAX_M` doc. Only the formats MTP verify's qwen35-4B-UD-Q4_K_XL run actually
/// hits (Q4_K/Q5_K/Q6_K/Q8_0) are built. NOT built for the sk_ag (split-K) family —
/// `dense_small_m_row_tile_bench` measured a net LOSS there (split-K's own `splits` dimension
/// already fills the device; see `matmul_native_splitk`'s doc).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_n128_ag_bm32_build_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
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
        Q4K => (
            "native_gemm_warp_q4k_n128_ag_bm32",
            v!("native_gemm_warp_q4k_n128_ag_bm32"),
        ),
        Q5K => (
            "native_gemm_warp_q5k_n128_ag_bm32",
            v!("native_gemm_warp_q5k_n128_ag_bm32"),
        ),
        Q6K => (
            "native_gemm_warp_q6k_n128_ag_bm32",
            v!("native_gemm_warp_q6k_n128_ag_bm32"),
        ),
        Q8_0 => (
            "native_gemm_warp_q8_0_n128_ag_bm32",
            v!("native_gemm_warp_q8_0_n128_ag_bm32"),
        ),
        _ => return None,
    })
}

/// BM=16 row-tile variant of [`native_gemm_warp_n128_ag_build_spv`] — one coopmat M-frag floor,
/// see the recorder's `DENSE_SMALL_TILE_MAX_M16` doc. Same format coverage as the BM=32 variant
/// (Q4_K/Q5_K/Q6_K/Q8_0), no sk_ag family.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_n128_ag_bm16_build_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
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
        Q4K => (
            "native_gemm_warp_q4k_n128_ag_bm16",
            v!("native_gemm_warp_q4k_n128_ag_bm16"),
        ),
        Q5K => (
            "native_gemm_warp_q5k_n128_ag_bm16",
            v!("native_gemm_warp_q5k_n128_ag_bm16"),
        ),
        Q6K => (
            "native_gemm_warp_q6k_n128_ag_bm16",
            v!("native_gemm_warp_q6k_n128_ag_bm16"),
        ),
        Q8_0 => (
            "native_gemm_warp_q8_0_n128_ag_bm16",
            v!("native_gemm_warp_q8_0_n128_ag_bm16"),
        ),
        _ => return None,
    })
}

/// SPIR-V for the SPLIT-K narrow warptile (NARROW_N + a k-split grid dimension writing partials).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_sk_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
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
        // F16 weights are floats, not quants (native_dense_supported is false) — this sk build
        // exists ONLY for the adapter's targeted deep-k narrow-n F16 route (the DiffusionGemma
        // SC soft-embedding GEMM); see the matching arm in `matmul_native_splitk`.
        F16 => v!("native_gemm_warp_f16_sk"),
        Iq4Nl => v!("native_gemm_warp_iq4nl_sk"),
        Iq4Xs => v!("native_gemm_warp_iq4xs_sk"),
        Q2K => v!("native_gemm_warp_q2k_sk"),
        Q3K => v!("native_gemm_warp_q3k_sk"),
        Q4_0 => v!("native_gemm_warp_q4_0_sk"),
        Q4K => v!("native_gemm_warp_q4k_sk"),
        Q5K => v!("native_gemm_warp_q5k_sk"),
        Q6K => v!("native_gemm_warp_q6k_sk"),
        Q8_0 => v!("native_gemm_warp_q8_0_sk"),
        _ => return None,
    })
}

/// SPIR-V for the split-K reduce (sum the partials planes).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn splitk_reduce_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/splitk_reduce.spv"
        )))
    })
}

/// SPIR-V for the native-block prefill GEMM (`C=A·Wᵀ`, raw GGUF blocks dequantized in-shader via the
/// coopmat tiled kernel). One specialization per quant format; `None` for unsupported dtypes.
#[cfg_attr(infr_profile, infr_prof::instrument)]
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
        Bf16 => v!("native_gemm_bf16"),
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
        Q2_0 => v!("native_gemm_q2_0"),
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
const ATTN_PARTIAL_DYNAC_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_partial_dynac.spv"));
const ATTN_COMBINE_LIVE_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_combine_live.spv"));
const ATTN_LIVE_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_live.spv"));
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
const RMSNORM_GATE_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm_gate.spv"));
const DELTANET_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deltanet.spv"));
const DELTANET_CHUNKED_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_chunked.spv"));
const DELTANET_PREP_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_prep.spv"));
const DELTANET_GATES_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_gates.spv"));
const DELTANET_SCAN_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_scan.spv"));
const DELTANET_NORM_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_norm.spv"));
const DELTANET_GATES_SEQ_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_gates_seq.spv"));
const DELTANET_SEQ_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_seq.spv"));
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
const LINEAR_F16_NOEXT_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/linear_f16_noext.spv"));
const LINEAR_F32_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_f32.spv"));
const LINEAR_F32R_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_f32r.spv"));
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

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn gemm_spv() -> &'static [u32] {
    GEMM_SPV.get_or_init(|| spv_words(GEMM_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn gemm_tiled_spv() -> &'static [u32] {
    GEMM_TILED_SPV.get_or_init(|| spv_words(GEMM_TILED_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn gemm_warp_spv() -> &'static [u32] {
    GEMM_WARP_SPV.get_or_init(|| spv_words(GEMM_WARP_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn gemm_dp4a_spv() -> &'static [u32] {
    GEMM_DP4A_SPV.get_or_init(|| spv_words(GEMM_DP4A_SPV_BYTES))
}
/// SPIR-V for the activation int8 quantize pass (Q8 per block) feeding the dp4a mmq matmul.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quant_q8_spv() -> &'static [u32] {
    QUANT_Q8_SPV.get_or_init(|| spv_words(QUANT_Q8_SPV_BYTES))
}
/// SPIR-V for the integer (dp4a) u4 projection GEMM. Weights stay quantized; no per-GEMM dequant.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn gemm_proj_mmq_spv() -> &'static [u32] {
    GEMM_PROJ_MMQ_SPV.get_or_init(|| spv_words(GEMM_PROJ_MMQ_SPV_BYTES))
}
/// SPIR-V for the prefill projection GEMM (`C=A·Wᵀ`, f16/quant W). Used by the recorder.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn gemm_proj_spv() -> &'static [u32] {
    GEMM_PROJ_SPV.get_or_init(|| spv_words(GEMM_PROJ_SPV_BYTES))
}
/// Warp-tiled projection GEMM (BM=64,BN=128). Faster for large M (low/mid-ctx prefill); the recorder
/// falls back to `gemm_proj_spv` for small M (high ctx) where its fewer workgroups lose occupancy.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn gemm_proj_warp_spv() -> &'static [u32] {
    GEMM_PROJ_WARP_SPV.get_or_init(|| spv_words(GEMM_PROJ_WARP_SPV_BYTES))
}
/// SPIR-V for the subgroup-reduction flash-decoding pass-1 (split-K) kernel. Used by the recorder.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_partial_spv() -> &'static [u32] {
    ATTN_PARTIAL_SPV.get_or_init(|| spv_words(ATTN_PARTIAL_SPV_BYTES))
}
/// Rows-batched split pass 1 (K/V streamed once per 4-row group; chunk <= 256).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_partial_mrows_c256_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_partial_mrows_c256.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the non-FA prefill attention kernels (QK scores / row softmax / PV). Recorder use.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_qk_spv() -> &'static [u32] {
    ATTN_QK_SPV.get_or_init(|| spv_words(ATTN_QK_SPV_BYTES))
}
/// 8-warp/256-thread QK GEMM (kv_pad % 256). Matches ollama's mul_mm warptile; the recorder uses it
/// over the 4-warp attn_qk unless INFR_NO_QK_WARP is set.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_qk_warp_spv() -> &'static [u32] {
    ATTN_QK_WARP_SPV.get_or_init(|| spv_words(ATTN_QK_WARP_SPV_BYTES))
}
/// Fused flash-attention prefill (QK→softmax→PV, no materialized S). Recorder `attention_prefill_flash`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_spv() -> &'static [u32] {
    ATTN_FLASH_SPV.get_or_init(|| spv_words(ATTN_FLASH_SPV_BYTES))
}
/// BM=32 build of the fused flash prefill (29056 B shared) for sub-64 KB shared devices.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_bm32_spv() -> &'static [u32] {
    ATTN_FLASH_BM32_SPV.get_or_init(|| spv_words(ATTN_FLASH_BM32_SPV_BYTES))
}
/// Flash-attention split-K partial pass (per kv-split online-softmax partials). Recorder use.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_partial_spv() -> &'static [u32] {
    ATTN_FLASH_PARTIAL_SPV.get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_SPV_BYTES))
}
/// BM=32 build of the split-K flash partial (29056 B shared) for sub-64 KB shared devices.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_partial_bm32_spv() -> &'static [u32] {
    ATTN_FLASH_PARTIAL_BM32_SPV.get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_BM32_SPV_BYTES))
}
/// Register-blocked flash partial (hd=128). Used over attn_flash_partial when hd==128.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_warp_spv() -> &'static [u32] {
    ATTN_FLASH_WARP_SPV.get_or_init(|| spv_words(ATTN_FLASH_WARP_SPV_BYTES))
}
/// BM=32 build of the flash partial (29056 B shared vs 58112 B): for devices whose
/// maxComputeSharedMemorySize is under the 64 KB the default BM=64 tile needs (NVIDIA, MoltenVK).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_warp_bm32_spv() -> &'static [u32] {
    ATTN_FLASH_WARP_BM32_SPV.get_or_init(|| spv_words(ATTN_FLASH_WARP_BM32_SPV_BYTES))
}
/// FlashAttention-2 register-O flash partial (Br=128, per-thread register accumulator). hd=128.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_reg_spv() -> &'static [u32] {
    ATTN_FLASH_REG_SPV.get_or_init(|| spv_words(ATTN_FLASH_REG_SPV_BYTES))
}
/// BR=64 build of the register-O flash partial (29440 B shared) for sub-64 KB shared devices.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_reg_br64_spv() -> &'static [u32] {
    ATTN_FLASH_REG_BR64_SPV.get_or_init(|| spv_words(ATTN_FLASH_REG_BR64_SPV_BYTES))
}
/// Flash-attention split-K combine (merge partials → final O). Recorder use.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_combine_spv() -> &'static [u32] {
    ATTN_FLASH_COMBINE_SPV.get_or_init(|| spv_words(ATTN_FLASH_COMBINE_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_softmax_spv() -> &'static [u32] {
    ATTN_SM_SPV.get_or_init(|| spv_words(ATTN_SM_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_pv_spv() -> &'static [u32] {
    ATTN_PV_SPV.get_or_init(|| spv_words(ATTN_PV_SPV_BYTES))
}
/// 8-warp/256-thread PV GEMM (BN=128=hd, hd % 128). The recorder uses it over the 4-warp attn_pv
/// when hd % 128 == 0 and INFR_NO_PV_WARP is unset.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_pv_warp_spv() -> &'static [u32] {
    ATTN_PV_WARP_SPV.get_or_init(|| spv_words(ATTN_PV_WARP_SPV_BYTES))
}
/// SPIR-V for the attn_pv split-K partial reducer (sums n_splits partial-O buffers).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_pv_reduce_spv() -> &'static [u32] {
    ATTN_PV_REDUCE_SPV.get_or_init(|| spv_words(ATTN_PV_REDUCE_SPV_BYTES))
}
/// SPIR-V for the 256-thread subgroup RMSNorm (`y=rmsnorm(x,w)`). Used by the recorder's `rmsnorm`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rmsnorm_spv() -> &'static [u32] {
    RMSNORM_SPV.get_or_init(|| spv_words(RMSNORM_SPV_BYTES))
}
/// SPIR-V for the 1024-thread vec4 RMSNorm (`rmsnorm.comp`'s -DWIDE build) — the decode-shaped
/// twin of [`rmsnorm_spv`], picked by the recorder when `rows==1 && dim>=2048 && dim%4==0`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rmsnorm_wide_spv() -> &'static [u32] {
    static RMSNORM_WIDE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    RMSNORM_WIDE_SPV.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/rmsnorm_wide.spv"
        )))
    })
}
/// SPIR-V for the fused per-head RMSNorm + SiLU gate multiply (`rmsnorm.comp`'s -DGATE build,
/// `Op::GatedRmsNorm`) — the qwen35 DeltaNet z-gate, one dispatch instead of `rmsnorm`+`silu_mul`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rmsnorm_gate_spv() -> &'static [u32] {
    static RMSNORM_GATE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    RMSNORM_GATE_SPV.get_or_init(|| spv_words(RMSNORM_GATE_SPV_BYTES))
}
/// SPIR-V for fused RMSNorm + in-place add (`rmsnorm.comp`'s -DADD build, `Op::RmsNormAdd`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rmsnorm_add_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm_add.spv"))))
}
/// SPIR-V for f16-in/f16-out RMSNorm (`rmsnorm.comp`'s -DF16IO build) — llama4's post-rope
/// weightless Q/K L2-norm (`Op::QkNorm` on the f16 rope scratch).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rmsnorm_f16_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm_f16.spv"))))
}
/// SPIR-V for the 256-thread subgroup row-softmax (`y=softmax(x*scale)`). Used by the recorder's
/// `softmax` (diffusion-gemma's in-graph self-conditioning).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn softmax_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softmax.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the elementwise add (`y=a+b`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn add_spv() -> &'static [u32] {
    static ADD_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ADD_SPV.get_or_init(|| spv_words(ADD_SPV_BYTES))
}
/// SPIR-V for the scaled add / axpy (`acc += scale*x`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn add_scaled_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_scaled.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the broadcast bias add (`dst[i] = x[i] + bias[i % n]`; Qwen2 q/k/v projections).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn add_bias_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_bias.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the broadcast vector multiply (`dst[i] = x[i] * vec[i % n]`; diffusion-gemma's
/// router input scale — the multiplicative twin of `add_bias`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn mul_vec_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_vec.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the qwen35moe shared-expert combine (`dst[r,c] = moe[r,c] + sigmoid(gate[r]) *
/// shexp[r,c]`; row-broadcast gate — the shared-expert twin of `mul_vec`'s column broadcast).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_shared_expert_add_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_shared_expert_add.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the in-place scalar multiply (`y *= scale`). gemma4 per-layer output scale.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn scale_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scale.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for elementwise softcap `y = cap·tanh(x/cap)` (gemma logit softcap).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn softcap_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softcap.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled Q8_0 dp4a (mmq) GEMM, expert-grid variant (a diffusion-gemma MoE down
/// projection quant option).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q8_0_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q8_0_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled Q5_0 dp4a (mmq) GEMM, expert-grid variant (the shipped
/// diffusiongemma-26B-A4B-it-GGUF quantizes its MoE down projection as Q5_0).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_0_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_0_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled Q5_K dp4a (mmq) GEMM, expert-grid variant (unsloth-dynamic Qwen3.6-MoE
/// quants mix Q5_K into the MoE down-projection banks; carries Q4_K's min term → binds `sact`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5k_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5k_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled Q5_1 dp4a (mmq) GEMM, expert-grid variant (the shipped
/// gemma-4-26B-A4B-it-GGUF quantizes its MoE down projection as Q5_1 on 29/30 layers; min-carrying
/// like Q4_K/Q5_K → binds `sact`, but no superblock sub-scale — one d/m pair per 32-block).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_1_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_1_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled Q2_K dp4a (mmq) GEMM, expert-grid variant (Llama-4-Scout's shipped
/// MoE gate/up banks; min-carrying like Q4_K/Q5_K → binds `sact`, 16-elem sub-block granularity).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_k_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_k_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled Q3_K dp4a (mmq) GEMM, expert-grid variant (Llama-4-Scout's shipped
/// MoE down-projection bank; symmetric like Q6_K — no `sact`, no min term).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q3_k_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q3_k_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

// BM=32 row-tile expert-grid variants — see build.rs's `_xp32` entries and
// `matmul_mmq_experts`'s small-rows-per-expert doc.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
// BM=32 + BN=128 (wide-N) expert-grid variants — see build.rs's `_xp32w` entries and
// `matmul_mmq_experts`'s wide-BN doc. Only the two dtypes real small-row MoE pools use.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_xp32w_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k_xp32w.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_xp32w_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k_xp32w.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
// BM=64 (default row tile) + BN=128 (wide-N) expert-grid variants — see build.rs's `_xp128`
// entries and `matmul_mmq_experts`'s wide-BN doc. Same two dtypes as the `_xp32w` small-tile pair.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_xp128_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k_xp128.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_xp128_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k_xp128.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q8_0_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q8_0_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_0_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_0_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5k_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5k_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_1_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_1_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_k_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_k_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q3_k_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q3_k_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

// PAGED expert-grid variants — the batched-MoE mmq GEMM reading a GpuPager arena through the
// word-base LUT (Scout's batched paged prefill; see the shaders' PAGED doc).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_k_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_k_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q3_k_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q3_k_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_k_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_k_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q3_k_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q3_k_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled Q4_0 dp4a (mmq) GEMM, expert-grid variant (symmetric, no shipped MoE GGUF
/// in the audited cache uses it for expert banks — trivial family member, synthetic parity only).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_0_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_0_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_0_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_0_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_0_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_0_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_0_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_0_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled Q2_0 dp4a (mmq) GEMM, expert-grid variants (Bonsai ternary — symmetric
/// trivial family member like Q4_0; no shipped MoE GGUF uses it for expert banks, synthetic
/// parity only).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_0_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_0_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_0_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_0_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_0_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_0_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_0_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_0_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled IQ2_S / IQ3_S dp4a (mmq) GEMMs, expert-grid variants (the grid-codebook
/// pair Qwen3.6-35B-A3B-UD-IQ3_S ships for its expert banks: IQ2_S gate/up, IQ3_S down). Both
/// symmetric (sign-flipped grid codes are the signed dp4a operand directly) — no `sact`. The
/// grid LUT is staged into shared memory per workgroup (see the shaders' doc comments).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq2_s_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq2_s_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq2_s_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq2_s_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq2_s_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq2_s_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq2_s_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq2_s_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq3_s_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq3_s_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq3_s_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq3_s_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq3_s_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq3_s_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq3_s_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq3_s_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled Q4_1 dp4a (mmq) GEMM, expert-grid variant (min-carrying, Q5_1's pattern
/// minus the highbit — binds `sact`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_1_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_1_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_1_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_1_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_1_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_1_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_1_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_1_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled IQ4_NL dp4a (mmq) GEMM, expert-grid variant (codebook, symmetric — the
/// LUT value itself is the signed dp4a operand, no `sact`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_nl_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq4_nl_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_nl_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq4_nl_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_nl_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq4_nl_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_nl_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_mmq_iq4_nl_xpg32.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled IQ4_XS dp4a (mmq) GEMM, expert-grid variant (codebook + Q4_K-shaped
/// superblock, symmetric — no `sact`; unsloth's UD quants mix this into most of
/// Qwen3.6-35B-A3B's gate/up expert banks).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_xs_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq4_xs_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_xs_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq4_xs_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_xs_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq4_xs_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_xs_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_mmq_iq4_xs_xpg32.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

// PAGED expert-grid variants for the remaining mmq dtype set (Q8_0/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K)
// — `MOE_MMQ_PAGED_DTYPES` now mirrors `MOE_MMQ_DTYPES` in full (fused gemma-4 MoE /
// DiffusionGemma banks and UD mixed-dtype roles page through these).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q8_0_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q8_0_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q8_0_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q8_0_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_0_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_0_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_0_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_0_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_1_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_1_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_1_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_1_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5k_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5k_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5k_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5k_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled MXFP4 dp4a (mmq) GEMM, expert-grid variant (the MXFP4_MOE quant family's
/// expert banks — signed E2M1 codebook → dp4a, IQ4_NL's treatment; symmetric, no `sact`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_mxfp4_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_mxfp4_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_mxfp4_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_mxfp4_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_mxfp4_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_mxfp4_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_mxfp4_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_mxfp4_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// SPIR-V for the tiled NVFP4 dp4a (mmq) GEMM, expert-grid variant (shares MXFP4's E2M1 codebook;
/// per-16 UE4M3 sub-block scales split each 32-block into two dp4a halves — see the shader doc).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_nvfp4_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_nvfp4_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_nvfp4_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_nvfp4_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_nvfp4_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_nvfp4_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_nvfp4_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_nvfp4_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quant_q8_gather_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quant_q8_gather.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_scatter_reduce_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_scatter_reduce.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the row gather (`dst[j]=src[idx[j]]`).
/// SPIR-V for the weighted row scatter-add (`dst[idx[j]] += w[j]*y[j]`).
/// SPIR-V for the SwiGLU activation (`y=silu(gate)*up`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn silu_mul_spv() -> &'static [u32] {
    static SILU_MUL_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    SILU_MUL_SPV.get_or_init(|| spv_words(SILU_MUL_SPV_BYTES))
}
/// SPIR-V for the gated-DeltaNet recurrence step (qwen35/Qwen3.5 SSM).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_SPV_BYTES))
}
/// SPIR-V for DeltaNet reading q/k/v from a single strided source buffer (env-gated).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_strided_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/deltanet_strided.spv"
        )))
    })
}
/// SPIR-V for the chunked-DeltaNet PREP pass (normalize + intra-chunk dot matrices).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_prep_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_PREP_SPV_BYTES))
}
/// SPIR-V for the chunked-DeltaNet GATES pass (β + prefix log-decay per chunk/head).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_gates_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_GATES_SPV_BYTES))
}
/// SPIR-V for the chunked-DeltaNet SCAN pass (the sequential state-coupled part).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_scan_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_SCAN_SPV_BYTES))
}
/// SPIR-V for the sequential-DeltaNet NORM pass (q/k L2-normalize; prep without the D/Dq dots).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_norm_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_NORM_SPV_BYTES))
}
/// SPIR-V for the sequential-DeltaNet GATES pass (β + per-token decay, flat rows·nv grid).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_gates_seq_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_GATES_SEQ_SPV_BYTES))
}
/// SPIR-V for the token-serial DeltaNet SCAN with the state column register-resident.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_seq_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_SEQ_SPV_BYTES))
}
/// SPIR-V for the CHUNKED gated-DeltaNet prefill (chunkwise delta rule, C=32).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_chunked_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_CHUNKED_SPV_BYTES))
}
/// SPIR-V for the causal depthwise conv1d + SiLU step (qwen35/Qwen3.5 SSM input conv).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn conv1d_silu_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(CONV1D_SILU_SPV_BYTES))
}
/// SPIR-V for the BATCH depthwise conv1d+SiLU (pass 1: all outputs in parallel).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn conv1d_silu_par_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(CONV1D_SILU_PAR_SPV_BYTES))
}
/// SPIR-V for the BATCH depthwise conv1d history rebuild (pass 2).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn conv1d_shift_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(CONV1D_SHIFT_SPV_BYTES))
}
/// SPIR-V for the batched strided row copy (Op::CopyStrided in one dispatch).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn copy_strided_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(COPY_STRIDED_SPV_BYTES))
}
/// SPIR-V for the elementwise sigmoid gate `y = a * sigmoid(b)`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn mul_sigmoid_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(MUL_SIGMOID_SPV_BYTES))
}
/// SPIR-V for the GeGLU activation with separate gate/up buffers (`y=gelu(gate)*up`). gemma4's
/// per-layer-embd gate.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn gelu_mul_spv() -> &'static [u32] {
    static GELU_MUL_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    GELU_MUL_SPV.get_or_init(|| spv_words(GELU_MUL_SPV_BYTES))
}
/// SPIR-V for the fused SwiGLU over a combined gate||up buffer.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn silu_mul_fused_spv() -> &'static [u32] {
    static SILU_MUL_FUSED_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    SILU_MUL_FUSED_SPV.get_or_init(|| spv_words(SILU_MUL_FUSED_SPV_BYTES))
}
/// SPIR-V for the fused GeGLU (GELU tanh-approx gate) over a combined gate||up buffer (gemma).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn gelu_mul_fused_spv() -> &'static [u32] {
    static GELU_MUL_FUSED_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    GELU_MUL_FUSED_SPV.get_or_init(|| spv_words(GELU_MUL_FUSED_SPV_BYTES))
}
/// SPIR-V for the f32→f16 cast-store into an f16 cache.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn store_f16_spv() -> &'static [u32] {
    static STORE_F16_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    STORE_F16_SPV.get_or_init(|| spv_words(STORE_F16_SPV_BYTES))
}
/// SPIR-V for RoPE (ggml NORM, interleaved pairs).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rope_spv() -> &'static [u32] {
    static ROPE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ROPE_SPV.get_or_init(|| spv_words(ROPE_SPV_BYTES))
}
/// SPIR-V for the f16-weight GEMV (`y=x·Wᵀ`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f16_spv() -> &'static [u32] {
    static LINEAR_F16_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    LINEAR_F16_SPV.get_or_init(|| spv_words(LINEAR_F16_SPV_BYTES))
}
/// SPIR-V for the f16-weight GEMV, `!caps.f16` tier (no shaderFloat16 — `unpackHalf2x16` dequant).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f16_noext_spv() -> &'static [u32] {
    static LINEAR_F16_NOEXT_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    LINEAR_F16_NOEXT_SPV.get_or_init(|| spv_words(LINEAR_F16_NOEXT_SPV_BYTES))
}
/// SPIR-V for the f32-weight GEMV (full-precision projection weights, e.g. gemma4 E2B per-layer).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f32_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(LINEAR_F32_SPV_BYTES))
}
/// SPIR-V for the reduction-shape f32 GEMV (workgroup per output — decode-hot narrow GEMVs).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f32r_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(LINEAR_F32R_SPV_BYTES))
}
/// SPIR-V for the ROW-TILED f32 GEMM (8 rows/workgroup — prefill weight reuse). Bit-identical to
/// `linear_f32r_spv` per output (same K-accumulation order); grid = out_f·ceil(rows/8).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f32r_mrow8_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/linear_f32r_mrow8.spv"
        )))
    })
}
/// SPIR-V for the vec4 ROW-TILED f32 GEMM (4 rows/workgroup, vec4 K stream — the small-m
/// prefill shape; requires in_f % 4 == 0). vec4-lane accumulation → NOT bit-identical to the
/// scalar kernels (f32 tolerance-level shift).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f32r_mrow4_v4_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/linear_f32r_mrow4_v4.spv"
        )))
    })
}
/// SPIR-V for the vec4 ROW-TILED f32 GEMM, 8 rows/workgroup (chunked-prefill rows>4 shape;
/// requires in_f % 4 == 0). Same vec4 accumulation caveat as the 4-row variant.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f32r_mrow8_v4_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/linear_f32r_mrow8_v4.spv"
        )))
    })
}
/// SPIR-V for E2B per-layer inp_gate fused GEMV+GELU+strided-multiply kernel.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn e2b_gate_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/e2b_gate.spv"))))
}
/// SPIR-V for E2B per-layer proj: fused f32 GEMV+RMSNorm+Add kernel.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn e2b_proj_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/e2b_proj.spv"))))
}
/// SPIR-V for fused QkNormRope reading from interleaved q+g buffer (qwen35 CopyStrided elim).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn qk_norm_rope_interleaved_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/qk_norm_rope_interleaved.spv"
        )))
    })
}
/// SPIR-V for interleaved QkNormRope with USE_PARAMS (record-once decode replay).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn qk_norm_rope_interleaved_dyn_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/qk_norm_rope_interleaved_dyn.spv"
        )))
    })
}
/// SPIR-V for the bf16-weight GEMV (`y=x·Wᵀ`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_bf16_spv() -> &'static [u32] {
    static LINEAR_BF16_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    LINEAR_BF16_SPV.get_or_init(|| spv_words(LINEAR_BF16_SPV_BYTES))
}
/// SPIR-V for the unified affine-quant dequant GEMV (`y=x·Wᵀ`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_q_spv() -> &'static [u32] {
    static LINEAR_Q_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    LINEAR_Q_SPV.get_or_init(|| spv_words(LINEAR_Q_SPV_BYTES))
}
/// SPIR-V for the f16-weight GEMV with fused residual.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_res_spv() -> &'static [u32] {
    static LINEAR_RES_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    LINEAR_RES_SPV.get_or_init(|| spv_words(LINEAR_RES_SPV_BYTES))
}
/// SPIR-V for the affine-quant dequant GEMV with fused residual.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_res_q_spv() -> &'static [u32] {
    static LINEAR_RES_Q_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    LINEAR_RES_Q_SPV.get_or_init(|| spv_words(LINEAR_RES_Q_SPV_BYTES))
}
/// SPIR-V for the online-softmax GQA attention (hd<=128).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attention_spv() -> &'static [u32] {
    static ATTENTION_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ATTENTION_SPV.get_or_init(|| spv_words(ATTENTION_SPV_BYTES))
}
/// SPIR-V for flash-decode combine (merge split-K partials).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_combine_spv() -> &'static [u32] {
    static ATTN_COMBINE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ATTN_COMBINE_SPV.get_or_init(|| spv_words(ATTN_COMBINE_SPV_BYTES))
}
/// SPIR-V for tiled online-softmax attention over an f16 KV cache.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attention_kv_spv() -> &'static [u32] {
    static ATTENTION_KV_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ATTENTION_KV_SPV.get_or_init(|| spv_words(ATTENTION_KV_SPV_BYTES))
}
/// SPIR-V for fused per-head QK-norm + NEOX RoPE (f16 out).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn qk_norm_rope_spv() -> &'static [u32] {
    static QK_NORM_ROPE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    QK_NORM_ROPE_SPV.get_or_init(|| spv_words(QK_NORM_ROPE_SPV_BYTES))
}
/// SPIR-V for QK-norm + RoPE with proportional-rope freq_factors (gemma4 full-attention layers).
#[cfg_attr(infr_profile, infr_prof::instrument)]
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
dyn_spv!(qk_norm_rope_dyn_ff_spv, "qk_norm_rope_dyn_ff");
// Same "read a scalar from a device buffer instead of a push constant" idea, but for
// DiffusionGemma denoise self-conditioning's softmax scale (`-DUSE_SCALE_BUF`) rather than the
// record-once decode replay's pos/kv_len — see `Op::Softmax::scale_buf`'s doc and `Recorder::
// softmax_dyn`.
dyn_spv!(softmax_dyn_spv, "softmax_dyn");
dyn_spv!(rope_f16_spv, "rope_f16");
dyn_spv!(rope_f16_dyn_spv, "rope_f16_dyn");
dyn_spv!(store_f16_dyn_spv, "store_f16_dyn");
dyn_spv!(attention_kv_dyn_spv, "attention_kv_dyn");
dyn_spv!(attn_partial_dyn_spv, "attn_partial_dyn");
// A/B escape for the hd=256/512 attn_partial fast paths (INFR_NO_ATTN_HD=1): the same three f16
// form-factors compiled with -DNO_HD_SPEC, so a regression on those shapes is diagnosable
// against the general per-key loops.
dyn_spv!(attn_partial_nohd_spv, "attn_partial_nohd");
dyn_spv!(attn_partial_dyn_nohd_spv, "attn_partial_dyn_nohd");
dyn_spv!(attn_partial_dynac_nohd_spv, "attn_partial_dynac_nohd");
/// `INFR_NO_ATTN_HD=1` — select the `-DNO_HD_SPEC` attn_partial variants (general loops only).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_hd_spec_disabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("INFR_NO_ATTN_HD").is_ok())
}
// Coupled Q8_0 KV cache: scalar dequant-on-read attention (static + record-once) and the row
// quantize-store kernels (f32 V + f16 K sources, each with a params/decode variant).
dyn_spv!(attention_kv_q8_spv, "attention_kv_q8");
dyn_spv!(attention_kv_kq8_spv, "attention_kv_kq8");
dyn_spv!(attention_kv_vq8_spv, "attention_kv_vq8");
dyn_spv!(attention_kv_dyn_q8_spv, "attention_kv_dyn_q8");
dyn_spv!(store_q8_spv, "store_q8");
dyn_spv!(store_q8_dyn_spv, "store_q8_dyn");
dyn_spv!(store_q8_f16_spv, "store_q8_f16");
dyn_spv!(store_q8_f16_dyn_spv, "store_q8_f16_dyn");
// Mainline low-bit KV quants: per-format quantize (f32 V / f16 K) + dequant→f16 prefix expander.
dyn_spv!(quant_kv_q4_0_spv, "quant_kv_q4_0");
dyn_spv!(quant_kv_q4_0_f16_spv, "quant_kv_q4_0_f16");
dyn_spv!(quant_kv_q4_1_spv, "quant_kv_q4_1");
dyn_spv!(quant_kv_q4_1_f16_spv, "quant_kv_q4_1_f16");
dyn_spv!(quant_kv_q5_0_spv, "quant_kv_q5_0");
dyn_spv!(quant_kv_q5_0_f16_spv, "quant_kv_q5_0_f16");
dyn_spv!(quant_kv_q5_1_spv, "quant_kv_q5_1");
dyn_spv!(quant_kv_q5_1_f16_spv, "quant_kv_q5_1_f16");
dyn_spv!(quant_kv_iq4_nl_spv, "quant_kv_iq4_nl");
dyn_spv!(quant_kv_iq4_nl_f16_spv, "quant_kv_iq4_nl_f16");
dyn_spv!(dequant_kv_q4_0_spv, "dequant_kv_q4_0");
dyn_spv!(dequant_kv_q4_1_spv, "dequant_kv_q4_1");
dyn_spv!(dequant_kv_q5_0_spv, "dequant_kv_q5_0");
dyn_spv!(dequant_kv_q5_1_spv, "dequant_kv_q5_1");
dyn_spv!(dequant_kv_iq4_nl_spv, "dequant_kv_iq4_nl");
dyn_spv!(dequant_kv_bf16_spv, "dequant_kv_bf16");
// Dense KV cast-store (f32 / bf16 cache, from f16 K or f32 V).
dyn_spv!(store_kv_f32_spv, "store_kv_f32");
dyn_spv!(store_kv_f32_from_f16_spv, "store_kv_f32_from_f16");
dyn_spv!(store_kv_bf16_spv, "store_kv_bf16");
dyn_spv!(store_kv_bf16_from_f16_spv, "store_kv_bf16_from_f16");

dyn_spv!(quant_turbo_t2_spv, "quant_turbo_t2");
dyn_spv!(quant_turbo_t2_f16_spv, "quant_turbo_t2_f16");
dyn_spv!(quant_turbo_t3_spv, "quant_turbo_t3");
dyn_spv!(quant_turbo_t3_f16_spv, "quant_turbo_t3_f16");
dyn_spv!(quant_turbo_t4_spv, "quant_turbo_t4");
dyn_spv!(quant_turbo_t4_f16_spv, "quant_turbo_t4_f16");
dyn_spv!(dequant_turbo_t2_spv, "dequant_turbo_t2");
dyn_spv!(dequant_turbo_t3_spv, "dequant_turbo_t3");
dyn_spv!(dequant_turbo_t4_spv, "dequant_turbo_t4");

/// (kernel name, SPIR-V) for the TurboQuant KV quantize of `dt` (`src_f16` = f16 K, else f32 V).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quant_turbo_kernel(
    dt: infr_core::DType,
    src_f16: bool,
) -> (&'static str, &'static [u32]) {
    use infr_core::DType::*;
    match (dt, src_f16) {
        (Turbo2, false) => ("quant_turbo_t2", quant_turbo_t2_spv()),
        (Turbo2, true) => ("quant_turbo_t2_f16", quant_turbo_t2_f16_spv()),
        (Turbo3, false) => ("quant_turbo_t3", quant_turbo_t3_spv()),
        (Turbo3, true) => ("quant_turbo_t3_f16", quant_turbo_t3_f16_spv()),
        (Turbo4, false) => ("quant_turbo_t4", quant_turbo_t4_spv()),
        (Turbo4, true) => ("quant_turbo_t4_f16", quant_turbo_t4_f16_spv()),
        _ => unreachable!("quant_turbo_kernel for non-turbo dtype {dt:?}"),
    }
}

/// (kernel name, SPIR-V) for the TurboQuant KV dequant→f16 of `dt`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn dequant_turbo_kernel(dt: infr_core::DType) -> (&'static str, &'static [u32]) {
    use infr_core::DType::*;
    match dt {
        Turbo2 => ("dequant_turbo_t2", dequant_turbo_t2_spv()),
        Turbo3 => ("dequant_turbo_t3", dequant_turbo_t3_spv()),
        Turbo4 => ("dequant_turbo_t4", dequant_turbo_t4_spv()),
        _ => unreachable!("dequant_turbo_kernel for non-turbo dtype {dt:?}"),
    }
}

/// (kernel name, SPIR-V) for the dense KV cast-store into `dst_dt` (F32/Bf16), `src_f16` = f16 K.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn store_kv_dense_kernel(
    dst_dt: infr_core::DType,
    src_f16: bool,
) -> (&'static str, &'static [u32]) {
    use infr_core::DType::*;
    match (dst_dt, src_f16) {
        (F32, false) => ("store_kv_f32", store_kv_f32_spv()),
        (F32, true) => ("store_kv_f32_from_f16", store_kv_f32_from_f16_spv()),
        (Bf16, false) => ("store_kv_bf16", store_kv_bf16_spv()),
        (Bf16, true) => ("store_kv_bf16_from_f16", store_kv_bf16_from_f16_spv()),
        _ => unreachable!("store_kv_dense_kernel for non-dense KV dtype {dst_dt:?}"),
    }
}

/// (kernel name, SPIR-V) for the KV quantize kernel of `dt` (`src_f16` = f16 K source, else f32 V).
/// Distinct names per variant so the recorder's name-keyed pipeline cache never collides.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quant_kv_kernel(
    dt: infr_core::DType,
    src_f16: bool,
) -> (&'static str, &'static [u32]) {
    use infr_core::DType::*;
    match (dt, src_f16) {
        (Q4_0, false) => ("quant_kv_q4_0", quant_kv_q4_0_spv()),
        (Q4_0, true) => ("quant_kv_q4_0_f16", quant_kv_q4_0_f16_spv()),
        (Q4_1, false) => ("quant_kv_q4_1", quant_kv_q4_1_spv()),
        (Q4_1, true) => ("quant_kv_q4_1_f16", quant_kv_q4_1_f16_spv()),
        (Q5_0, false) => ("quant_kv_q5_0", quant_kv_q5_0_spv()),
        (Q5_0, true) => ("quant_kv_q5_0_f16", quant_kv_q5_0_f16_spv()),
        (Q5_1, false) => ("quant_kv_q5_1", quant_kv_q5_1_spv()),
        (Q5_1, true) => ("quant_kv_q5_1_f16", quant_kv_q5_1_f16_spv()),
        (Iq4Nl, false) => ("quant_kv_iq4_nl", quant_kv_iq4_nl_spv()),
        (Iq4Nl, true) => ("quant_kv_iq4_nl_f16", quant_kv_iq4_nl_f16_spv()),
        _ => unreachable!("quant_kv_kernel for non-KV-quant dtype {dt:?}"),
    }
}

/// (kernel name, SPIR-V) for the KV dequant→f16 prefix expander of `dt`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn dequant_kv_kernel(dt: infr_core::DType) -> (&'static str, &'static [u32]) {
    use infr_core::DType::*;
    match dt {
        Q4_0 => ("dequant_kv_q4_0", dequant_kv_q4_0_spv()),
        Q4_1 => ("dequant_kv_q4_1", dequant_kv_q4_1_spv()),
        Q5_0 => ("dequant_kv_q5_0", dequant_kv_q5_0_spv()),
        Q5_1 => ("dequant_kv_q5_1", dequant_kv_q5_1_spv()),
        Iq4Nl => ("dequant_kv_iq4_nl", dequant_kv_iq4_nl_spv()),
        Bf16 => ("dequant_kv_bf16", dequant_kv_bf16_spv()),
        _ => unreachable!("dequant_kv_kernel for non-prepass KV dtype {dt:?}"),
    }
}
// Coupled Q8_0 KV: coalesced split-K decode partials reading Q8 blocks (static / dyn / self-chunk).
dyn_spv!(attn_partial_q8_spv, "attn_partial_q8");
dyn_spv!(attn_partial_kq8_spv, "attn_partial_kq8");
dyn_spv!(attn_partial_vq8_spv, "attn_partial_vq8");
dyn_spv!(attn_partial_dyn_q8_spv, "attn_partial_dyn_q8");
dyn_spv!(attn_partial_dynac_q8_spv, "attn_partial_dynac_q8");
dyn_spv!(dequant_q8_f16_spv, "dequant_q8_f16");
/// SPIR-V for the SELF-CHUNKING record-once split-K decode partial (adaptive chunk from the live
/// kv_len; workgroups past the live range early-exit with a zero-weight header).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_partial_dynac_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(ATTN_PARTIAL_DYNAC_SPV_BYTES))
}
/// SPIR-V for the live-count combine (record-once replay; loops the prologue's live chunks).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_combine_live_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(ATTN_COMBINE_LIVE_SPV_BYTES))
}
/// SPIR-V for the split-K replay prologue (indirect args + live count from kv_len).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_live_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(ATTN_LIVE_SPV_BYTES))
}
/// SPIR-V + cache name for the subgroup decode GEMV (`y=x·Wᵀ`). `bits`=4/8 picks the quant
/// variant; `res` adds a fused residual; `sg16` selects the `-DSG=16` twin (`caps.sg_pref == 16`).
/// Used by the recorder's `linear_q` / `linear_add_q`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn mul_mat_vec_q_spv(
    bits: u32,
    res: bool,
    sg16: bool,
) -> (&'static str, &'static [u32]) {
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            ($name, s)
        }};
    }
    match (bits, res, sg16) {
        (4, false, false) => (
            "mul_mat_vec_q4",
            MMV_Q4_SPV
                .get_or_init(|| spv_words(MMV_Q4_SPV_BYTES))
                .as_slice(),
        ),
        (8, false, false) => (
            "mul_mat_vec_q8",
            MMV_Q8_SPV
                .get_or_init(|| spv_words(MMV_Q8_SPV_BYTES))
                .as_slice(),
        ),
        (4, true, false) => (
            "mul_mat_vec_q4_res",
            MMV_Q4_RES_SPV
                .get_or_init(|| spv_words(MMV_Q4_RES_SPV_BYTES))
                .as_slice(),
        ),
        (8, true, false) => (
            "mul_mat_vec_q8_res",
            MMV_Q8_RES_SPV
                .get_or_init(|| spv_words(MMV_Q8_RES_SPV_BYTES))
                .as_slice(),
        ),
        (4, false, true) => v!("mul_mat_vec_q4_sg16"),
        (8, false, true) => v!("mul_mat_vec_q8_sg16"),
        (4, true, true) => v!("mul_mat_vec_q4_res_sg16"),
        (8, true, true) => v!("mul_mat_vec_q8_res_sg16"),
        _ => panic!("mul_mat_vec_q: unsupported bits={bits}"),
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
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
        // Pin subgroup size 32: the shader computes one subgroup-scope coopmat tile per
        // 32-thread workgroup, and an unpinned compute pipeline may get wave64 on RDNA3 — a
        // workgroup that's a fraction of a subgroup, which the spec forbids for subgroup-scope
        // cooperative matrices (VUID-VkPipelineShaderStageCreateInfo-module-08987).
        let kern = self.kernel_sg("gemm_coopmat", gemm_spv(), 3, 12, 32);
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

        let bufs = [
            unsafe { as_vk_buf(buf_a.as_ref()) }.buffer,
            unsafe { as_vk_buf(buf_b.as_ref()) }.buffer,
            unsafe { as_vk_buf(buf_c.as_ref()) }.buffer,
        ];
        let binding = self.eager_bind(&kern, &bufs).unwrap();

        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        let (gx, gy) = ((n / 64) as u32, (m / 64) as u32);

        let dispatch = || {
            let shared = &self.shared;
            self.one_shot(|cmd| unsafe {
                shared
                    .device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kern.pipeline);
                binding.bind(shared, cmd, kern.pipeline_layout);
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
        let bufs = [
            unsafe { as_vk_buf(buf_a.as_ref()) }.buffer,
            unsafe { as_vk_buf(buf_b.as_ref()) }.buffer,
            unsafe { as_vk_buf(buf_c.as_ref()) }.buffer,
        ];
        let binding = self.eager_bind(&kern, &bufs).unwrap();
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        let (gx, gy) = ((n / 128) as u32, (m / 128) as u32);
        let dispatch = || {
            let shared = &self.shared;
            self.one_shot(|cmd| unsafe {
                shared
                    .device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kern.pipeline);
                binding.bind(shared, cmd, kern.pipeline_layout);
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
        let bufs = [
            unsafe { as_vk_buf(buf_a.as_ref()) }.buffer,
            unsafe { as_vk_buf(buf_b.as_ref()) }.buffer,
            unsafe { as_vk_buf(buf_c.as_ref()) }.buffer,
        ];
        let binding = self.eager_bind(&kern, &bufs).unwrap();
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(kp as u32).to_ne_bytes());
        let (gx, gy) = ((n / 64) as u32, (m / 64) as u32);
        let dispatch = || {
            let shared = &self.shared;
            self.one_shot(|cmd| unsafe {
                shared
                    .device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kern.pipeline);
                binding.bind(shared, cmd, kern.pipeline_layout);
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

        let a16: Vec<u16> = a.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
        let b16: Vec<u16> = b.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
        let buf_a = self.alloc(a16.len() * 2, BufferUsage::Staging)?;
        let buf_b = self.alloc(b16.len() * 2, BufferUsage::Staging)?;
        let buf_c = self.alloc(m * n * 4, BufferUsage::Readback)?;
        self.upload(buf_a.as_ref(), bytemuck::cast_slice(&a16))?;
        self.upload(buf_b.as_ref(), bytemuck::cast_slice(&b16))?;

        let bufs = [
            unsafe { as_vk_buf(buf_a.as_ref()) }.buffer,
            unsafe { as_vk_buf(buf_b.as_ref()) }.buffer,
            unsafe { as_vk_buf(buf_c.as_ref()) }.buffer,
        ];
        let binding = self.eager_bind(&kern, &bufs)?;

        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());

        let shared = &self.shared;
        self.one_shot(|cmd| unsafe {
            shared
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kern.pipeline);
            binding.bind(shared, cmd, kern.pipeline_layout);
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
