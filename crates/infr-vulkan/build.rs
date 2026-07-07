use std::process::Command;

/// Compile GLSL compute shaders (which need features WGSL/naga can't express, e.g.
/// cooperative matrix) to SPIR-V via `glslc` at build time. Output to OUT_DIR.
fn main() {
    println!("cargo:rerun-if-changed=shaders");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    gen_grids(&out);
    // (source stem, output stem, extra glslc defines)
    let builds: &[(&str, &str, &[&str])] = &[
        ("gemm_coopmat", "gemm_coopmat", &[]),
        ("gemm_coopmat_tiled", "gemm_coopmat_tiled", &[]),
        ("gemm_warp", "gemm_warp", &[]),
        ("gemm_dp4a", "gemm_dp4a", &[]),
        ("quant_q8", "quant_q8", &[]),
        ("gemm_proj_mmq", "gemm_proj_mmq", &[]),
        ("gemm_proj", "gemm_proj", &[]),
        ("gemm_proj_warp", "gemm_proj_warp", &[]),
        ("attn_partial", "attn_partial", &[]),
        // PROBE (INFR_MROWS_ATTN=1): RB=4 rows-batched pass 1 — occupancy-hypothesis experiment.
        ("attn_partial_mrows", "attn_partial_mrows", &[]),
        (
            "attn_partial_mrows",
            "attn_partial_mrows_c256",
            &["-DSC_MAX=256u"],
        ),
        // A/B escape (INFR_NO_ATTN_HD=1): compile out the hd=256/512 fast paths so a regression
        // on those shapes is diagnosable against the general loops (f16 form-factors only).
        ("attn_partial", "attn_partial_nohd", &["-DNO_HD_SPEC"]),
        ("attn_partial", "attn_partial_dyn", &["-DUSE_PARAMS"]),
        (
            "attn_partial",
            "attn_partial_dyn_nohd",
            &["-DUSE_PARAMS", "-DNO_HD_SPEC"],
        ),
        (
            "attn_partial",
            "attn_partial_dynac",
            &["-DUSE_PARAMS", "-DSELF_CHUNK"],
        ),
        (
            "attn_partial",
            "attn_partial_dynac_nohd",
            &["-DUSE_PARAMS", "-DSELF_CHUNK", "-DNO_HD_SPEC"],
        ),
        // Planar Q8_0 KV cache: coalesced split-K decode reading Q8 blocks (halves KV read traffic).
        // K and V decouple (-DKQ8 / -DVQ8) → 3 quant combos (kq8, vq8, both) for the STATIC split.
        ("attn_partial", "attn_partial_q8", &["-DKQ8", "-DVQ8"]),
        ("attn_partial", "attn_partial_kq8", &["-DKQ8"]),
        ("attn_partial", "attn_partial_vq8", &["-DVQ8"]),
        // Record-once (USE_PARAMS) is disabled for a Q8 cache (forced static), so only the coupled
        // variant is kept for the dyn/dynac form-factors (referenced but unreachable at runtime).
        (
            "attn_partial",
            "attn_partial_dyn_q8",
            &["-DKQ8", "-DVQ8", "-DUSE_PARAMS"],
        ),
        (
            "attn_partial",
            "attn_partial_dynac_q8",
            &["-DKQ8", "-DVQ8", "-DUSE_PARAMS", "-DSELF_CHUNK"],
        ),
        ("attn_qk", "attn_qk", &[]),
        ("attn_qk_warp", "attn_qk_warp", &[]),
        ("attn_flash", "attn_flash", &[]),
        ("attn_flash", "attn_flash_bm32", &["-DBM_TILE=32"]),
        ("attn_flash_partial", "attn_flash_partial", &[]),
        (
            "attn_flash_partial",
            "attn_flash_partial_bm32",
            &["-DBM_TILE=32"],
        ),
        ("attn_flash_warp", "attn_flash_warp", &[]),
        // BM=32 tile: 29056 B shared (vs 58112 B), fits NVIDIA (48 KB) / MoltenVK (32 KB) devices
        // whose maxComputeSharedMemorySize is under the 64 KB the default BM=64 tile needs.
        ("attn_flash_warp", "attn_flash_warp_bm32", &["-DBM_TILE=32"]),
        ("attn_flash_reg", "attn_flash_reg", &[]),
        // BR=64 tile: 29440 B shared (vs 58880 B) for sub-64 KB shared devices (NVIDIA, MoltenVK).
        ("attn_flash_reg", "attn_flash_reg_br64", &["-DBR_TILE=64"]),
        ("attn_flash_combine", "attn_flash_combine", &[]),
        ("attn_softmax", "attn_softmax", &[]),
        ("attn_pv", "attn_pv", &[]),
        ("attn_pv_warp", "attn_pv_warp", &[]),
        ("attn_pv_reduce", "attn_pv_reduce", &[]),
        ("rmsnorm", "rmsnorm", &[]),
        ("softmax", "softmax", &[]),
        // DiffusionGemma denoise self-conditioning perf: scale read from a device buffer instead
        // of a push constant (see `Op::Softmax::scale_buf`'s doc + `Recorder::softmax_dyn`).
        ("softmax", "softmax_dyn", &["-DUSE_SCALE_BUF"]),
        ("deltanet", "deltanet", &[]),
        ("deltanet_chunked", "deltanet_chunked", &[]),
        ("deltanet_prep", "deltanet_prep", &[]),
        ("deltanet_gates", "deltanet_gates", &[]),
        ("deltanet_scan", "deltanet_scan", &[]),
        ("conv1d_silu", "conv1d_silu", &[]),
        ("conv1d_silu_par", "conv1d_silu_par", &[]),
        ("conv1d_shift", "conv1d_shift", &[]),
        ("copy_strided", "copy_strided", &[]),
        ("mul_sigmoid", "mul_sigmoid", &[]),
        ("add", "add", &[]),
        ("add_bias", "add_bias", &[]),
        ("mul_vec", "mul_vec", &[]),
        ("moe_shared_expert_add", "moe_shared_expert_add", &[]),
        ("add_scaled", "add_scaled", &[]),
        ("scale", "scale", &[]),
        ("softcap", "softcap", &[]),
        ("silu_mul", "silu_mul", &[]),
        ("silu_mul", "gelu_mul", &["-DGELU"]),
        ("silu_mul_fused", "silu_mul_fused", &[]),
        ("silu_mul_fused", "gelu_mul_fused", &["-DGELU"]),
        ("store_f16", "store_f16", &[]),
        ("store_f16", "store_f16_dyn", &["-DUSE_PARAMS"]),
        // Quantize a KV row into the Q8_0 cache. f32 source (V) + f16 source (un-fused roped K),
        // each with a record-once (pos from params) variant.
        ("store_q8", "store_q8", &[]),
        ("store_q8", "store_q8_dyn", &["-DUSE_PARAMS"]),
        ("store_q8", "store_q8_f16", &["-DSRC_F16"]),
        // Expand a Q8_0 KV prefix → f16 scratch so the f16 flash/non-FA prefill kernels can run.
        ("dequant_q8_f16", "dequant_q8_f16", &[]),
        (
            "store_q8",
            "store_q8_f16_dyn",
            &["-DSRC_F16", "-DUSE_PARAMS"],
        ),
        // Mainline low-bit KV quants (standard GGUF blocks): a quantize kernel per format (f32 V +
        // f16 K sources) and a dequant→f16 prefix expander per format (reuses native_decode `dq()`).
        // Static only — a quantized KV cache forces per-execute static decode (see decode_eligible).
        ("quant_kv", "quant_kv_q4_0", &["-DFMT_Q4_0"]),
        (
            "quant_kv",
            "quant_kv_q4_0_f16",
            &["-DFMT_Q4_0", "-DSRC_F16"],
        ),
        ("quant_kv", "quant_kv_q4_1", &["-DFMT_Q4_1"]),
        (
            "quant_kv",
            "quant_kv_q4_1_f16",
            &["-DFMT_Q4_1", "-DSRC_F16"],
        ),
        ("quant_kv", "quant_kv_q5_0", &["-DFMT_Q5_0"]),
        (
            "quant_kv",
            "quant_kv_q5_0_f16",
            &["-DFMT_Q5_0", "-DSRC_F16"],
        ),
        ("quant_kv", "quant_kv_q5_1", &["-DFMT_Q5_1"]),
        (
            "quant_kv",
            "quant_kv_q5_1_f16",
            &["-DFMT_Q5_1", "-DSRC_F16"],
        ),
        ("quant_kv", "quant_kv_iq4_nl", &["-DFMT_IQ4NL"]),
        (
            "quant_kv",
            "quant_kv_iq4_nl_f16",
            &["-DFMT_IQ4NL", "-DSRC_F16"],
        ),
        ("dequant_kv_f16", "dequant_kv_q4_0", &["-DFMT_Q4_0"]),
        ("dequant_kv_f16", "dequant_kv_q4_1", &["-DFMT_Q4_1"]),
        ("dequant_kv_f16", "dequant_kv_q5_0", &["-DFMT_Q5_0"]),
        ("dequant_kv_f16", "dequant_kv_q5_1", &["-DFMT_Q5_1"]),
        ("dequant_kv_f16", "dequant_kv_iq4_nl", &["-DFMT_IQ4NL"]),
        // Dense KV caches (f32/bf16): a cast-store per (dst, src) + the bf16→f16 read (native_decode
        // FMT_BF16). f32→f16 read reuses store_f16. K = f16 source, V = f32 source.
        ("store_kv_dense", "store_kv_f32", &["-DDST_F32"]),
        (
            "store_kv_dense",
            "store_kv_f32_from_f16",
            &["-DDST_F32", "-DSRC_F16"],
        ),
        ("store_kv_dense", "store_kv_bf16", &["-DDST_BF16"]),
        (
            "store_kv_dense",
            "store_kv_bf16_from_f16",
            &["-DDST_BF16", "-DSRC_F16"],
        ),
        ("dequant_kv_f16", "dequant_kv_bf16", &["-DFMT_BF16"]),
        // TurboQuant KV (WHT-rotated): quantize (f32 V + f16 K) + dequant→f16, per width.
        ("quant_turbo", "quant_turbo_t2", &["-DTURBO2"]),
        (
            "quant_turbo",
            "quant_turbo_t2_f16",
            &["-DTURBO2", "-DSRC_F16"],
        ),
        ("quant_turbo", "quant_turbo_t3", &["-DTURBO3"]),
        (
            "quant_turbo",
            "quant_turbo_t3_f16",
            &["-DTURBO3", "-DSRC_F16"],
        ),
        ("quant_turbo", "quant_turbo_t4", &["-DTURBO4"]),
        (
            "quant_turbo",
            "quant_turbo_t4_f16",
            &["-DTURBO4", "-DSRC_F16"],
        ),
        ("dequant_turbo_f16", "dequant_turbo_t2", &["-DTURBO2"]),
        ("dequant_turbo_f16", "dequant_turbo_t3", &["-DTURBO3"]),
        ("dequant_turbo_f16", "dequant_turbo_t4", &["-DTURBO4"]),
        ("rope", "rope", &[]),
        ("rope", "rope_f16", &["-DOUT_F16"]),
        ("rope", "rope_f16_dyn", &["-DOUT_F16", "-DUSE_PARAMS"]),
        ("linear_f16", "linear_f16", &[]),
        ("linear_bf16", "linear_bf16", &[]),
        ("linear_f32", "linear_f32", &[]),
        ("linear_f32r", "linear_f32r", &[]),
        ("linear_f32r", "linear_f32r_mrow8", &["-DMROW=8"]),
        (
            "linear_f32r",
            "linear_f32r_mrow4_v4",
            &["-DMROW=4", "-DVEC4"],
        ),
        (
            "linear_f32r",
            "linear_f32r_mrow8_v4",
            &["-DMROW=8", "-DVEC4"],
        ),
        ("matmul_f32", "matmul_f32", &[]),
        ("linear_q", "linear_q", &[]),
        ("linear_res", "linear_res", &[]),
        ("linear_res_q", "linear_res_q", &[]),
        ("attention", "attention", &[]),
        ("attn_combine", "attn_combine", &[]),
        ("attn_combine", "attn_combine_live", &["-DUSE_LIVE"]),
        ("attn_live", "attn_live", &[]),
        ("attention_kv", "attention_kv", &[]),
        ("attention_kv", "attention_kv_dyn", &["-DUSE_PARAMS"]),
        // Planar Q8_0 KV cache: scalar dequant-on-read variants. K/V decouple (-DKQ8 / -DVQ8) → 3
        // quant combos for the STATIC scalar path; coupled-only for the record-once (dead for Q8).
        ("attention_kv", "attention_kv_q8", &["-DKQ8", "-DVQ8"]),
        ("attention_kv", "attention_kv_kq8", &["-DKQ8"]),
        ("attention_kv", "attention_kv_vq8", &["-DVQ8"]),
        (
            "attention_kv",
            "attention_kv_dyn_q8",
            &["-DKQ8", "-DVQ8", "-DUSE_PARAMS"],
        ),
        ("qk_norm_rope", "qk_norm_rope", &[]),
        ("qk_norm_rope", "qk_norm_rope_dyn", &["-DUSE_PARAMS"]),
        ("qk_norm_rope", "qk_norm_rope_ff", &["-DFREQ_FACTORS"]),
        (
            "qk_norm_rope",
            "qk_norm_rope_dyn_ff",
            &["-DUSE_PARAMS", "-DFREQ_FACTORS"],
        ),
        // Native-block dequant GEMVs: one .spv per (quant format, residual) from one source.
        ("native_gemv", "native_q8_0", &["-DFMT_Q8_0"]),
        ("native_gemv", "native_bf16", &["-DFMT_BF16"]),
        (
            "native_gemv",
            "native_bf16_res",
            &["-DFMT_BF16", "-DUSE_RES"],
        ),
        (
            "native_gemv",
            "native_iq2xxs",
            &["-DFMT_IQ2XXS", "-DUSE_GRID"],
        ),
        (
            "native_gemv",
            "native_iq2xxs_res",
            &["-DFMT_IQ2XXS", "-DUSE_GRID", "-DUSE_RES"],
        ),
        (
            "native_gemv",
            "native_iq2xs",
            &["-DFMT_IQ2XS", "-DUSE_GRID"],
        ),
        (
            "native_gemv",
            "native_iq2xs_res",
            &["-DFMT_IQ2XS", "-DUSE_GRID", "-DUSE_RES"],
        ),
        ("native_gemv", "native_iq2s", &["-DFMT_IQ2S", "-DUSE_GRID"]),
        (
            "native_gemv",
            "native_iq2s_res",
            &["-DFMT_IQ2S", "-DUSE_GRID", "-DUSE_RES"],
        ),
        (
            "native_gemv",
            "native_iq3xxs",
            &["-DFMT_IQ3XXS", "-DUSE_GRID"],
        ),
        (
            "native_gemv",
            "native_iq3xxs_res",
            &["-DFMT_IQ3XXS", "-DUSE_GRID", "-DUSE_RES"],
        ),
        ("native_gemv", "native_iq3s", &["-DFMT_IQ3S", "-DUSE_GRID"]),
        (
            "native_gemv",
            "native_iq3s_res",
            &["-DFMT_IQ3S", "-DUSE_GRID", "-DUSE_RES"],
        ),
        ("native_gemv", "native_iq1s", &["-DFMT_IQ1S", "-DUSE_GRID"]),
        (
            "native_gemv",
            "native_iq1s_res",
            &["-DFMT_IQ1S", "-DUSE_GRID", "-DUSE_RES"],
        ),
        ("native_gemv", "native_iq1m", &["-DFMT_IQ1M", "-DUSE_GRID"]),
        (
            "native_gemv",
            "native_iq1m_res",
            &["-DFMT_IQ1M", "-DUSE_GRID", "-DUSE_RES"],
        ),
        ("native_gemv", "native_iq4nl", &["-DFMT_IQ4NL"]),
        (
            "native_gemv",
            "native_iq4nl_res",
            &["-DFMT_IQ4NL", "-DUSE_RES"],
        ),
        ("native_gemv", "native_iq4xs", &["-DFMT_IQ4XS"]),
        (
            "native_gemv",
            "native_iq4xs_res",
            &["-DFMT_IQ4XS", "-DUSE_RES"],
        ),
        ("native_gemv", "native_mxfp4", &["-DFMT_MXFP4"]),
        (
            "native_gemv",
            "native_mxfp4_res",
            &["-DFMT_MXFP4", "-DUSE_RES"],
        ),
        ("native_gemv", "native_nvfp4", &["-DFMT_NVFP4"]),
        (
            "native_gemv",
            "native_nvfp4_res",
            &["-DFMT_NVFP4", "-DUSE_RES"],
        ),
        ("native_gemv", "native_tq1_0", &["-DFMT_TQ1_0"]),
        (
            "native_gemv",
            "native_tq1_0_res",
            &["-DFMT_TQ1_0", "-DUSE_RES"],
        ),
        ("native_gemv", "native_tq2_0", &["-DFMT_TQ2_0"]),
        (
            "native_gemv",
            "native_tq2_0_res",
            &["-DFMT_TQ2_0", "-DUSE_RES"],
        ),
        ("native_gemv", "native_q4_0", &["-DFMT_Q4_0"]),
        (
            "native_gemv",
            "native_q4_0_res",
            &["-DFMT_Q4_0", "-DUSE_RES"],
        ),
        ("native_gemv", "native_q4_1", &["-DFMT_Q4_1"]),
        (
            "native_gemv",
            "native_q4_1_res",
            &["-DFMT_Q4_1", "-DUSE_RES"],
        ),
        ("native_gemv", "native_q5_0", &["-DFMT_Q5_0"]),
        (
            "native_gemv",
            "native_q5_0_res",
            &["-DFMT_Q5_0", "-DUSE_RES"],
        ),
        ("native_gemv", "native_q5_1", &["-DFMT_Q5_1"]),
        (
            "native_gemv",
            "native_q5_1_res",
            &["-DFMT_Q5_1", "-DUSE_RES"],
        ),
        ("native_gemv", "native_q2k", &["-DFMT_Q2K"]),
        ("native_gemv", "native_q2k_res", &["-DFMT_Q2K", "-DUSE_RES"]),
        ("native_gemv", "native_q3k", &["-DFMT_Q3K"]),
        ("native_gemv", "native_q3k_res", &["-DFMT_Q3K", "-DUSE_RES"]),
        ("native_gemv", "native_q4k", &["-DFMT_Q4K"]),
        ("native_gemv", "native_q4k_res", &["-DFMT_Q4K", "-DUSE_RES"]),
        ("native_gemv", "native_q5k", &["-DFMT_Q5K"]),
        ("native_gemv", "native_q5k_res", &["-DFMT_Q5K", "-DUSE_RES"]),
        ("native_gemv", "native_q6k", &["-DFMT_Q6K"]),
        ("native_gemv", "native_q6k_res", &["-DFMT_Q6K", "-DUSE_RES"]),
        // Multi-row GEMV (m = 2..8: spec verify / short suffix prefill) — mainstream dense
        // projection formats only; the rest fall back to the tiled GEMM.
        ("native_gemv_mrow", "native_mrow_q8_0", &["-DFMT_Q8_0"]),
        ("native_gemv_mrow", "native_mrow_bf16", &["-DFMT_BF16"]),
        ("native_gemv_mrow", "native_mrow_q4_0", &["-DFMT_Q4_0"]),
        ("native_gemv_mrow", "native_mrow_q4_1", &["-DFMT_Q4_1"]),
        ("native_gemv_mrow", "native_mrow_q5_0", &["-DFMT_Q5_0"]),
        ("native_gemv_mrow", "native_mrow_q5_1", &["-DFMT_Q5_1"]),
        ("native_gemv_mrow", "native_mrow_q2k", &["-DFMT_Q2K"]),
        ("native_gemv_mrow", "native_mrow_q3k", &["-DFMT_Q3K"]),
        ("native_gemv_mrow", "native_mrow_q4k", &["-DFMT_Q4K"]),
        ("native_gemv_mrow", "native_mrow_q5k", &["-DFMT_Q5K"]),
        ("native_gemv_mrow", "native_mrow_q6k", &["-DFMT_Q6K"]),
        ("native_gemv_mrow", "native_mrow_iq4nl", &["-DFMT_IQ4NL"]),
        ("native_gemv_mrow", "native_mrow_iq4xs", &["-DFMT_IQ4XS"]),
        (
            "native_gemv",
            "native_q8_0_res",
            &["-DFMT_Q8_0", "-DUSE_RES"],
        ),
        // Id-indexed native GEMVs for GPU-resident MoE decode (expert chosen from a GPU buffer): one
        // .spv per affine quant format experts are stored in. Codebook/grid formats fall back to the
        // host-top-k path, so they're omitted here.
        ("native_gemv_id", "native_id_q8_0", &["-DFMT_Q8_0"]),
        ("native_gemv_id", "native_id_q4_0", &["-DFMT_Q4_0"]),
        ("native_gemv_id", "native_id_q4_1", &["-DFMT_Q4_1"]),
        ("native_gemv_id", "native_id_q5_0", &["-DFMT_Q5_0"]),
        ("native_gemv_id", "native_id_q5_1", &["-DFMT_Q5_1"]),
        ("native_gemv_id", "native_id_q2k", &["-DFMT_Q2K"]),
        ("native_gemv_id", "native_id_q3k", &["-DFMT_Q3K"]),
        ("native_gemv_id", "native_id_q4k", &["-DFMT_Q4K"]),
        ("native_gemv_id", "native_id_q5k", &["-DFMT_Q5K"]),
        ("native_gemv_id", "native_id_q6k", &["-DFMT_Q6K"]),
        // Multi-slot id GEMV: all n_used experts in one dispatch (concurrent, no inter-expert barrier).
        ("native_gemv_id_multi", "native_idm_q8_0", &["-DFMT_Q8_0"]),
        ("native_gemv_id_multi", "native_idm_q4_0", &["-DFMT_Q4_0"]),
        ("native_gemv_id_multi", "native_idm_q4_1", &["-DFMT_Q4_1"]),
        ("native_gemv_id_multi", "native_idm_q5_0", &["-DFMT_Q5_0"]),
        ("native_gemv_id_multi", "native_idm_q5_1", &["-DFMT_Q5_1"]),
        ("native_gemv_id_multi", "native_idm_q2k", &["-DFMT_Q2K"]),
        ("native_gemv_id_multi", "native_idm_q3k", &["-DFMT_Q3K"]),
        ("native_gemv_id_multi", "native_idm_q4k", &["-DFMT_Q4K"]),
        ("native_gemv_id_multi", "native_idm_q5k", &["-DFMT_Q5K"]),
        ("native_gemv_id_multi", "native_idm_q6k", &["-DFMT_Q6K"]),
        ("moe_accumulate", "moe_accumulate", &[]),
        ("moe_accumulate_scaled", "moe_accumulate_scaled", &[]),
        ("native_mmv_id_q4k", "native_mmv_id_q4k", &[]),
        // Int8 dp4a decode GEMV (m=1, NUM_ROWS=2): one .spv per (format, residual).
        ("native_mmv", "native_mmv_q4k", &["-DFMT_Q4K"]),
        (
            "native_mmv",
            "native_mmv_q4k_res",
            &["-DFMT_Q4K", "-DUSE_RES"],
        ),
        ("native_mmv", "native_mmv_q6k", &["-DFMT_Q6K"]),
        (
            "native_mmv",
            "native_mmv_q6k_res",
            &["-DFMT_Q6K", "-DUSE_RES"],
        ),
        // Multi-row int8 dp4a GEMV (m=2..8): weight sub-block unpacked once, dp4a per row.
        ("native_mmv_mrow", "native_mmv_mrow_q4k", &["-DFMT_Q4K"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_m4",
            &["-DFMT_Q4K", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_m4",
            &["-DFMT_Q6K", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_o4_m4",
            &["-DFMT_Q4K", "-DOUTS4", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_o4_m4",
            &["-DFMT_Q6K", "-DOUTS4", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_o4",
            &["-DFMT_Q4K", "-DOUTS4"],
        ),
        ("native_mmv_mrow", "native_mmv_mrow_q6k", &["-DFMT_Q6K"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_o4",
            &["-DFMT_Q6K", "-DOUTS4"],
        ),
        ("native_gemm_mmq_q4k", "native_gemm_mmq_q4k", &[]),
        ("native_gemm_mmq_q6k", "native_gemm_mmq_q6k", &[]),
        (
            "native_gemm_mmq_q4k",
            "native_gemm_mmq_q4k_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_q6k",
            "native_gemm_mmq_q6k_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_q8_0",
            "native_gemm_mmq_q8_0_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_q5_0",
            "native_gemm_mmq_q5_0_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_q5k",
            "native_gemm_mmq_q5k_xp",
            &["-DEXPERT_GRID"],
        ),
        ("quant_q8", "quant_q8_gather", &["-DGATHER"]),
        ("moe_scatter_reduce", "moe_scatter_reduce", &[]),
        ("moe_topk", "moe_topk", &[]),
        // Embedding-row gather+dequant (Op::EmbedGather): one .spv per table format.
        ("embed_gather", "embed_gather_q8_0", &["-DFMT_Q8_0"]),
        ("embed_gather", "embed_gather_bf16", &["-DFMT_BF16"]),
        ("embed_gather", "embed_gather_f16", &["-DFMT_F16"]),
        ("embed_gather", "embed_gather_q4_0", &["-DFMT_Q4_0"]),
        ("embed_gather", "embed_gather_q4_1", &["-DFMT_Q4_1"]),
        ("embed_gather", "embed_gather_q5_0", &["-DFMT_Q5_0"]),
        ("embed_gather", "embed_gather_q5_1", &["-DFMT_Q5_1"]),
        ("embed_gather", "embed_gather_q2k", &["-DFMT_Q2K"]),
        ("embed_gather", "embed_gather_q3k", &["-DFMT_Q3K"]),
        ("embed_gather", "embed_gather_q4k", &["-DFMT_Q4K"]),
        ("embed_gather", "embed_gather_q5k", &["-DFMT_Q5K"]),
        ("embed_gather", "embed_gather_q6k", &["-DFMT_Q6K"]),
        ("embed_gather", "embed_gather_iq4nl", &["-DFMT_IQ4NL"]),
        ("embed_gather", "embed_gather_iq4xs", &["-DFMT_IQ4XS"]),
        // Chained-decode id ring log (ring[pos & 63] = sampled id) — see id_log.comp.
        ("id_log", "id_log", &[]),
        // Device-side decode-replay params advance ([pos, kv_len] += 1) — see params_advance.comp.
        ("params_advance", "params_advance", &[]),
        // Two-stage vocab-scale stochastic sampler (Op::Sample): per-slice top-k candidates,
        // then select+softmax+nucleus+CDF over the union.
        ("sample_topk", "sample_topk_part", &[]),
        ("sample_topk", "sample_topk", &["-DPASS2"]),
        // Chained-decode variant: stage 2 reads `u` from a 64-slot ring keyed by the self-advancing
        // `params[0]` instead of a 1-float buffer — see sample_topk.comp's CHAIN doc.
        ("sample_topk", "sample_topk_chain", &["-DPASS2", "-DCHAIN"]),
        // Two-stage greedy argmax (Op::Argmax): slice partials, then a one-workgroup reduce.
        ("argmax", "argmax_part", &[]),
        ("argmax", "argmax", &["-DPASS2"]),
        ("moe_sample", "moe_sample", &[]),
        // DiffusionGemma perf slice 3 (docs/DIFFUSIONGEMMA.md): fused per-row entropy-bound
        // sampler reduction — argmax/entropy/CDF-sample over [rows, vocab] logits on-GPU.
        ("dg_eb_sample", "dg_eb_sample", &[]),
        ("moe_bucket_count", "moe_bucket_count", &[]),
        ("moe_bucket_scan", "moe_bucket_scan", &[]),
        ("moe_bucket_scatter", "moe_bucket_scatter", &[]),
        (
            "moe_bucket_scatter",
            "moe_bucket_scatter_scaled",
            &["-DDSCALE"],
        ),
        ("add_scaled_id", "add_scaled_id", &[]),
        // Native-block prefill GEMMs: one .spv per quant format (coopmat tiled, no residual).
        ("native_gemm_warp", "native_gemm_warp_q4k", &["-DFMT_Q4K"]),
        ("native_gemm_warp", "native_gemm_warp_q6k", &["-DFMT_Q6K"]),
        ("native_gemm_warp", "native_gemm_warp_q8_0", &["-DFMT_Q8_0"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_n128",
            &["-DFMT_Q4K", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_n128",
            &["-DFMT_Q6K", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_n128",
            &["-DFMT_Q8_0", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_sk",
            &["-DFMT_Q4K", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_sk",
            &["-DFMT_Q6K", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_sk",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_ag",
            &["-DFMT_Q4K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_n128_ag",
            &["-DFMT_Q4K", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_sk_ag",
            &["-DFMT_Q4K", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_ag",
            &["-DFMT_Q6K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_n128_ag",
            &["-DFMT_Q6K", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_sk_ag",
            &["-DFMT_Q6K", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_ag",
            &["-DFMT_Q8_0", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_n128_ag",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_sk_ag",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        // F16-weight SPLIT-K warptile (DG slice-7): the SC soft-embedding GEMM (m=canvas,
        // k=vocab=262144, n=ne=2816) sat on the legacy `gemm_proj` BN=64 tile at ~6.5 TFLOPS —
        // 4x slower than llama.cpp's same-shape f16 GEMM. Only the sk variant is instantiated:
        // the adapter routes F16 here ONLY when the narrow-grid split-K policy fires (deep-k,
        // underfilled grid); every other f16 GEMM keeps its existing `matmul_proj` route.
        (
            "native_gemm_warp",
            "native_gemm_warp_f16_sk",
            &["-DFMT_F16", "-DNARROW_N", "-DSPLIT_K"],
        ),
        ("native_gemm_warp", "native_gemm_warp_bf16", &["-DFMT_BF16"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_bf16_n128",
            &["-DFMT_BF16", "-DNARROW_N"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q3k", &["-DFMT_Q3K"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q3k_n128",
            &["-DFMT_Q3K", "-DNARROW_N"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q5_0", &["-DFMT_Q5_0"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5_0_n128",
            &["-DFMT_Q5_0", "-DNARROW_N"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q5_1", &["-DFMT_Q5_1"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5_1_n128",
            &["-DFMT_Q5_1", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4xs",
            &["-DFMT_IQ4XS"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4xs_n128",
            &["-DFMT_IQ4XS", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4xs_sk",
            &["-DFMT_IQ4XS", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4xs_ag",
            &["-DFMT_IQ4XS", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4xs_n128_ag",
            &["-DFMT_IQ4XS", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4xs_sk_ag",
            &["-DFMT_IQ4XS", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q2k", &["-DFMT_Q2K"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q2k_n128",
            &["-DFMT_Q2K", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q2k_sk",
            &["-DFMT_Q2K", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q2k_ag",
            &["-DFMT_Q2K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q2k_n128_ag",
            &["-DFMT_Q2K", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q2k_sk_ag",
            &["-DFMT_Q2K", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q4_0", &["-DFMT_Q4_0"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4_0_n128",
            &["-DFMT_Q4_0", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4_0_sk",
            &["-DFMT_Q4_0", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4_0_ag",
            &["-DFMT_Q4_0", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4_0_n128_ag",
            &["-DFMT_Q4_0", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4_0_sk_ag",
            &["-DFMT_Q4_0", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q5k", &["-DFMT_Q5K"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_n128",
            &["-DFMT_Q5K", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_sk",
            &["-DFMT_Q5K", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_ag",
            &["-DFMT_Q5K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_n128_ag",
            &["-DFMT_Q5K", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_sk_ag",
            &["-DFMT_Q5K", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        ("splitk_reduce", "splitk_reduce", &[]),
        ("native_gemm", "native_gemm_q8_0", &["-DFMT_Q8_0"]),
        ("native_gemm", "native_gemm_bf16", &["-DFMT_BF16"]),
        ("native_gemm", "native_gemm_q4_0", &["-DFMT_Q4_0"]),
        ("native_gemm", "native_gemm_q4_1", &["-DFMT_Q4_1"]),
        ("native_gemm", "native_gemm_q5_0", &["-DFMT_Q5_0"]),
        ("native_gemm", "native_gemm_q5_1", &["-DFMT_Q5_1"]),
        ("native_gemm", "native_gemm_q2k", &["-DFMT_Q2K"]),
        ("native_gemm", "native_gemm_q3k", &["-DFMT_Q3K"]),
        ("native_gemm", "native_gemm_q4k", &["-DFMT_Q4K"]),
        ("native_gemm", "native_gemm_q5k", &["-DFMT_Q5K"]),
        ("native_gemm", "native_gemm_q6k", &["-DFMT_Q6K"]),
        ("native_gemm", "native_gemm_iq4nl", &["-DFMT_IQ4NL"]),
        ("native_gemm", "native_gemm_iq4xs", &["-DFMT_IQ4XS"]),
        ("native_gemm", "native_gemm_mxfp4", &["-DFMT_MXFP4"]),
        ("native_gemm", "native_gemm_nvfp4", &["-DFMT_NVFP4"]),
        ("native_gemm", "native_gemm_tq1_0", &["-DFMT_TQ1_0"]),
        ("native_gemm", "native_gemm_tq2_0", &["-DFMT_TQ2_0"]),
        (
            "native_gemm",
            "native_gemm_iq2xxs",
            &["-DFMT_IQ2XXS", "-DUSE_GRID"],
        ),
        (
            "native_gemm",
            "native_gemm_iq2xs",
            &["-DFMT_IQ2XS", "-DUSE_GRID"],
        ),
        (
            "native_gemm",
            "native_gemm_iq2s",
            &["-DFMT_IQ2S", "-DUSE_GRID"],
        ),
        (
            "native_gemm",
            "native_gemm_iq3xxs",
            &["-DFMT_IQ3XXS", "-DUSE_GRID"],
        ),
        (
            "native_gemm",
            "native_gemm_iq3s",
            &["-DFMT_IQ3S", "-DUSE_GRID"],
        ),
        (
            "native_gemm",
            "native_gemm_iq1s",
            &["-DFMT_IQ1S", "-DUSE_GRID"],
        ),
        (
            "native_gemm",
            "native_gemm_iq1m",
            &["-DFMT_IQ1M", "-DUSE_GRID"],
        ),
        // Decode GEMV: q4/q8 × plain/residual specializations from one source.
        ("mul_mat_vec_q", "mul_mat_vec_q4", &["-DQBITS=4"]),
        ("mul_mat_vec_q", "mul_mat_vec_q8", &["-DQBITS=8"]),
        (
            "mul_mat_vec_q",
            "mul_mat_vec_q4_res",
            &["-DQBITS=4", "-DUSE_RES"],
        ),
        (
            "mul_mat_vec_q",
            "mul_mat_vec_q8_res",
            &["-DQBITS=8", "-DUSE_RES"],
        ),
    ];
    for (src_stem, dst_stem, defines) in builds {
        let src = format!("shaders/{src_stem}.comp");
        let dst = format!("{out}/{dst_stem}.spv");
        println!("cargo:rerun-if-changed={src}");
        let mut args: Vec<String> = vec![
            "-fshader-stage=comp".into(),
            "--target-env=vulkan1.3".into(),
            "-O".into(),
            format!("-I{out}"),
        ];
        for d in *defines {
            args.push((*d).to_string());
        }
        args.push(src.clone());
        args.push("-o".into());
        args.push(dst);
        let status = Command::new("glslc")
            .args(&args)
            .status()
            .expect("failed to run glslc — install shaderc (provides glslc)");
        assert!(status.success(), "glslc failed for {src}");
    }
    // Shader-set fingerprint for the on-disk vkPipelineCache (see src/pcache.rs): FNV-1a over
    // every compiled SPIR-V blob in the (stable) build-list order. Any shader edit, new variant,
    // or glslc/define change flips it, and the persisted cache file is discarded wholesale —
    // stale pipeline entries never accumulate across shader-set changes.
    let mut h: u64 = 0xcbf29ce484222325;
    for (_, dst_stem, _) in builds {
        let bytes = std::fs::read(format!("{out}/{dst_stem}.spv")).expect("read spv for hash");
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    std::fs::write(
        format!("{out}/shader_fingerprint.rs"),
        format!("pub(crate) const SHADER_SET_FINGERPRINT: u64 = {h:#x};\n"),
    )
    .expect("write shader_fingerprint.rs");
}

/// Generate `native_grids.glsl` (the i-quant lookup tables as GLSL `const` arrays) from
/// `infr_core::iquant_grids` — single source of truth, baked into the per-format SPIR-V at build
/// time. Each table is `#if`-guarded by the formats that use it. WGSL/GLSL has no u64, so u64 grids
/// are emitted as u32 pairs (lo, hi).
fn gen_grids(out: &str) {
    use infr_core::iquant_grids as g;
    let u64arr = |name: &str, guard: &str, grid: &[u64]| -> String {
        let mut t = format!(
            "#if {guard}\nconst uint {name}[{}] = uint[](",
            grid.len() * 2
        );
        for &v in grid {
            t += &format!("{}u,{}u,", v as u32, (v >> 32) as u32);
        }
        t.pop();
        t += ");\n#endif\n";
        t
    };
    let u32arr = |name: &str, guard: &str, grid: &[u32]| -> String {
        let mut t = format!("#if {guard}\nconst uint {name}[{}] = uint[](", grid.len());
        for &v in grid {
            t += &format!("{v}u,");
        }
        t.pop();
        t += ");\n#endif\n";
        t
    };
    let mut s =
        String::from("// Generated by build.rs from infr_core::iquant_grids. Do not edit.\n");
    let mut ks = String::from(
        "#if defined(FMT_IQ2XXS) || defined(FMT_IQ2XS) || defined(FMT_IQ3XXS)\nconst uint KSIGNS[128] = uint[](",
    );
    for &v in g::KSIGNS_IQ2XS.iter() {
        ks += &format!("{v}u,");
    }
    ks.pop();
    ks += ");\n#endif\n";
    s += &ks;
    s += &u64arr("G_IQ2XXS", "defined(FMT_IQ2XXS)", &g::IQ2XXS_GRID);
    s += &u64arr("G_IQ2XS", "defined(FMT_IQ2XS)", &g::IQ2XS_GRID);
    s += &u64arr("G_IQ2S", "defined(FMT_IQ2S)", &g::IQ2S_GRID);
    s += &u32arr("G_IQ3XXS", "defined(FMT_IQ3XXS)", &g::IQ3XXS_GRID);
    s += &u32arr("G_IQ3S", "defined(FMT_IQ3S)", &g::IQ3S_GRID);
    s += &u64arr(
        "G_IQ1S",
        "defined(FMT_IQ1S) || defined(FMT_IQ1M)",
        &g::IQ1S_GRID,
    );
    std::fs::write(format!("{out}/native_grids.glsl"), s).expect("write native_grids.glsl");
}
