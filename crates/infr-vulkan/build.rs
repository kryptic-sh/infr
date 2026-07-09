use std::process::Command;

/// Compile GLSL compute shaders (which need features WGSL/naga can't express, e.g.
/// cooperative matrix) to SPIR-V via `glslc` at build time. Output to OUT_DIR.
fn main() {
    // INFR_PROFILE=1 at build time -> cfg(infr_profile) -> #[cfg_attr(infr_profile,
    // infr_prof::instrument)] annotations become live and inject profiling spans into every fn.
    // Default builds get NO cfg and zero profiling code. See docs/PERF.md.
    println!("cargo:rerun-if-env-changed=INFR_PROFILE");
    println!("cargo:rustc-check-cfg=cfg(infr_profile)");
    if std::env::var("INFR_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0") {
        println!("cargo:rustc-cfg=infr_profile");
    }
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
        // Fused per-head RMSNorm + SiLU gate multiply (qwen35 DeltaNet z-gate, Op::GatedRmsNorm) —
        // same reduction as `rmsnorm`, one extra buffer + the gate multiply on store.
        ("rmsnorm", "rmsnorm_gate", &["-DGATE"]),
        ("rmsnorm", "rmsnorm_add", &["-DADD"]),
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
        // !caps.f16 fallback (no shaderFloat16 ext): reads the f16 weight buffer as packed u32 +
        // unpackHalf2x16 (core GLSL) instead of a float16_t SSBO read.
        ("linear_f16_noext", "linear_f16_noext", &[]),
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
        ("e2b_gate", "e2b_gate", &[]),
        ("e2b_proj", "e2b_proj", &[]),
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
        // Interleaved q+g variant: reads query elements from strided buffer, eliminates per-head
        // CopyStrided dispatches (qwen35 attention layers).
        ("qk_norm_rope_interleaved", "qk_norm_rope_interleaved", &[]),
        (
            "qk_norm_rope_interleaved",
            "qk_norm_rope_interleaved_dyn",
            &["-DUSE_PARAMS"],
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
        // Multi-output-row decode GEMV (RM rows/workgroup) — bit-identical to the RM=1 GEMV per
        // row, but RM× the in-flight weight streams per wave to saturate DRAM on the low-out_f
        // (out_f≈4096) projection GEMVs. K-quant dense/MoE-hot formats only.
        ("native_gemv_rm", "native_q4k_rm2", &["-DFMT_Q4K", "-DRM=2"]),
        (
            "native_gemv_rm",
            "native_q4k_rm2_res",
            &["-DFMT_Q4K", "-DRM=2", "-DUSE_RES"],
        ),
        ("native_gemv_rm", "native_q4k_rm4", &["-DFMT_Q4K", "-DRM=4"]),
        (
            "native_gemv_rm",
            "native_q4k_rm4_res",
            &["-DFMT_Q4K", "-DRM=4", "-DUSE_RES"],
        ),
        ("native_gemv_rm", "native_q6k_rm2", &["-DFMT_Q6K", "-DRM=2"]),
        (
            "native_gemv_rm",
            "native_q6k_rm2_res",
            &["-DFMT_Q6K", "-DRM=2", "-DUSE_RES"],
        ),
        ("native_gemv_rm", "native_q6k_rm4", &["-DFMT_Q6K", "-DRM=4"]),
        (
            "native_gemv_rm",
            "native_q6k_rm4_res",
            &["-DFMT_Q6K", "-DRM=4", "-DUSE_RES"],
        ),
        // ── Experimental RM kernel variants (env-gated, default OFF) ──────────────────────────
        // Subgroup shuffle tree-reduce (replaces shared-mem + 2 barriers per row)
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_sg",
            &["-DFMT_Q4K", "-DRM=2", "-DVARIANT_SG"],
        ),
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_sg_res",
            &["-DFMT_Q4K", "-DRM=2", "-DUSE_RES", "-DVARIANT_SG"],
        ),
        // Double-buffered dequant (pre-load next dqblk during current dot product)
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_dbuf",
            &["-DFMT_Q4K", "-DRM=2", "-DVARIANT_DBUF"],
        ),
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_dbuf_res",
            &["-DFMT_Q4K", "-DRM=2", "-DUSE_RES", "-DVARIANT_DBUF"],
        ),
        // 128-thread workgroup (2x memory requests in flight)
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_wg128",
            &["-DFMT_Q4K", "-DRM=2", "-DVARIANT_WG128"],
        ),
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_wg128_res",
            &["-DFMT_Q4K", "-DRM=2", "-DUSE_RES", "-DVARIANT_WG128"],
        ),
        // Register-only reduce via subgroup ops (no shared memory at all)
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_reg",
            &["-DFMT_Q4K", "-DRM=2", "-DVARIANT_REG"],
        ),
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_reg_res",
            &["-DFMT_Q4K", "-DRM=2", "-DUSE_RES", "-DVARIANT_REG"],
        ),
        // Q6K subgroup variant
        (
            "native_gemv_rm_v2",
            "native_q6k_rm2_sg",
            &["-DFMT_Q6K", "-DRM=2", "-DVARIANT_SG"],
        ),
        (
            "native_gemv_rm_v2",
            "native_q6k_rm2_sg_res",
            &["-DFMT_Q6K", "-DRM=2", "-DUSE_RES", "-DVARIANT_SG"],
        ),
        // Reassociation-tolerant subgroup + NUM_ROWS decode GEMV (wave32, subgroupAdd, no shared
        // reduce) for the latency-STARVED out_f≈2048-8192 Q6_K projections (ffn_down / o / attn_qkv).
        // NOT bit-identical to the tree GEMV (reordered accumulation); gated to that band only. Q6_K
        // ONLY — on Q4_K the tree/RM kernel already saturates and SG regressed at every measured shape
        // (decode_gemv_bw A/B), so no Q4_K SG build exists. NR ∈ {2,4,8} × {plain,res}.
        ("native_gemv_sg", "native_q6k_sg2", &["-DFMT_Q6K", "-DNR=2"]),
        (
            "native_gemv_sg",
            "native_q6k_sg2_res",
            &["-DFMT_Q6K", "-DNR=2", "-DUSE_RES"],
        ),
        ("native_gemv_sg", "native_q6k_sg4", &["-DFMT_Q6K", "-DNR=4"]),
        (
            "native_gemv_sg",
            "native_q6k_sg4_res",
            &["-DFMT_Q6K", "-DNR=4", "-DUSE_RES"],
        ),
        ("native_gemv_sg", "native_q6k_sg8", &["-DFMT_Q6K", "-DNR=8"]),
        (
            "native_gemv_sg",
            "native_q6k_sg8_res",
            &["-DFMT_Q6K", "-DNR=8", "-DUSE_RES"],
        ),
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
        // Reassociation-tolerant subgroup+NR variant of the multi-slot id GEMV (wave32, subgroupAdd,
        // no shared reduce) for the latency-STARVED Q6_K MoE expert down-projection (out_f≈2048 — the
        // largest SG win of any decode shape). NOT bit-identical (reordered accumulation); gated to
        // the Q6_K projection band in the recorder. Q6_K ONLY (mirrors the dense native_gemv_sg
        // discipline — Q4_K idm already saturates). NR ∈ {2,4,8}.
        (
            "native_gemv_id_multi_sg",
            "native_idm_q6k_sg2",
            &["-DFMT_Q6K", "-DNR=2"],
        ),
        (
            "native_gemv_id_multi_sg",
            "native_idm_q6k_sg4",
            &["-DFMT_Q6K", "-DNR=4"],
        ),
        (
            "native_gemv_id_multi_sg",
            "native_idm_q6k_sg8",
            &["-DFMT_Q6K", "-DNR=8"],
        ),
        // Q5_K SG id variant: the Qwen3.6-A3B UD-quant stores most expert down-projections as Q5_K
        // (not Q6_K) at the same out_f≈2048 down shape — the heavy K-quant decode still nets out on
        // wave32+subgroupAdd (A/B-confirmed a win; Q4_K stays on the tree). NR ∈ {2,4,8}.
        (
            "native_gemv_id_multi_sg",
            "native_idm_q5k_sg2",
            &["-DFMT_Q5K", "-DNR=2"],
        ),
        (
            "native_gemv_id_multi_sg",
            "native_idm_q5k_sg4",
            &["-DFMT_Q5K", "-DNR=4"],
        ),
        (
            "native_gemv_id_multi_sg",
            "native_idm_q5k_sg8",
            &["-DFMT_Q5K", "-DNR=8"],
        ),
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
        // Multi-warp int8 dp4a decode GEMV (llama mul_mat_vec_q block, warp-per-row subgroupAdd) for
        // wave32-native GPUs. WARPS ∈ {4,8} rows/block × {plain,res}. Gated to subgroup_max==32.
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w4",
            &["-DFMT_Q4K", "-DWARPS=4"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w4_res",
            &["-DFMT_Q4K", "-DWARPS=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w8",
            &["-DFMT_Q4K", "-DWARPS=8"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w8_res",
            &["-DFMT_Q4K", "-DWARPS=8", "-DUSE_RES"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q6k_w4",
            &["-DFMT_Q6K", "-DWARPS=4"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q6k_w4_res",
            &["-DFMT_Q6K", "-DWARPS=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q6k_w8",
            &["-DFMT_Q6K", "-DWARPS=8"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q6k_w8_res",
            &["-DFMT_Q6K", "-DWARPS=8", "-DUSE_RES"],
        ),
        // IQ4_XS: codebook-gather-then-dp4a (the 4-bit code indexes KV_IQ4NL -> int8 before the dot).
        ("native_mmv", "native_mmv_iq4xs", &["-DFMT_IQ4XS"]),
        (
            "native_mmv",
            "native_mmv_iq4xs_res",
            &["-DFMT_IQ4XS", "-DUSE_RES"],
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
        // IQ4_XS multi-row (m=2..8): codebook-gather-then-dp4a, Q4_K-style single-dp accumulation.
        ("native_mmv_mrow", "native_mmv_mrow_iq4xs", &["-DFMT_IQ4XS"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4xs_m4",
            &["-DFMT_IQ4XS", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4xs_o4",
            &["-DFMT_IQ4XS", "-DOUTS4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4xs_o4_m4",
            &["-DFMT_IQ4XS", "-DOUTS4", "-DMRV=4"],
        ),
        ("native_gemm_mmq_q4k", "native_gemm_mmq_q4k", &[]),
        ("native_gemm_mmq_q6k", "native_gemm_mmq_q6k", &[]),
        // int8 coopmat (WMMA) prefill GEMM, Q8_0 only — measurement kernel gated behind
        // INFR_I8_COOPMAT=1 (see adapter.rs / docs in the .comp file). Default-off; correctness
        // validated against native_gemm_mmq_q8_0/native_gemm_warp_q8_0.
        ("native_gemm_i8cm_q8_0", "native_gemm_i8cm_q8_0", &[]),
        // "Idea 2" measurement variant: whole-row (block-invariant) activation scale instead of
        // per-32-block, gated behind INFR_I8_ROW_SCALE=1 (see native_gemm_i8cm_q8_0.comp #ifdef
        // ROW_SCALE + quant_q8_row.comp).
        (
            "native_gemm_i8cm_q8_0",
            "native_gemm_i8cm_q8_0_rowscale",
            &["-DROW_SCALE"],
        ),
        // Row-wise activation quant for the int8-coopmat GEMM's "Idea 2" measurement (whole-K
        // scale instead of per-32-block; see quant_q8_row.comp). Gated by INFR_I8_ROW_SCALE=1.
        ("quant_q8_row", "quant_q8_row", &[]),
        // fp8 (E4M3) coopmat prefill GEMM, Q8_0 only — gated behind INFR_F8_COOPMAT=1 +
        // caps.f8_coopmat (RDNA4 native fp8 WMMA; see adapter.rs `f8cm_ok` / the .comp file's
        // design doc). Default-off; correctness UNVALIDATED on this box (no fp8 coopmat hardware
        // here — compile-checked only), pending an RDNA4 run. Same 256-thread/8-warp warptile as
        // native_gemm_warp (BM=64xBN=256 wide); -DNARROW_N mirrors native_gemm_warp's n%128
        // occupancy-fix variant (BN=128/BK=64).
        ("native_gemm_f8cm_q8_0", "native_gemm_f8cm_q8_0", &[]),
        (
            "native_gemm_f8cm_q8_0",
            "native_gemm_f8cm_q8_0_n128",
            &["-DNARROW_N"],
        ),
        // Row-wise activation quant for the fp8-coopmat GEMM (whole-K amax -> scale = amax/448,
        // range-scales activations into E4M3 before the cast; see quant_f8_row.comp). Gated by
        // the same INFR_F8_COOPMAT=1 + caps.f8_coopmat as native_gemm_f8cm_q8_0 above.
        ("quant_f8_row", "quant_f8_row", &[]),
        // -DPREPACK measurement variant: reads a pre-packed E4M3 weight buffer directly (no
        // in-shader Q8_0 dequant) — tests whether removing the dqblk bottleneck lets fp8 beat f16
        // (see native_gemm_f8cm_q8_0.comp header). Gated behind INFR_F8_COOPMAT=1 +
        // INFR_F8_PREPACK=1 (adapter.rs `f8cm_ok`'s INFR_F8_PREPACK arm). Default path (unset) is
        // completely unaffected — these are new, additive SPIR-V entries.
        (
            "native_gemm_f8cm_q8_0",
            "native_gemm_f8cm_q8_0_prepack",
            &["-DPREPACK"],
        ),
        (
            "native_gemm_f8cm_q8_0",
            "native_gemm_f8cm_q8_0_prepack_n128",
            &["-DPREPACK", "-DNARROW_N"],
        ),
        // Bakes each Q8_0 32-block's scale into an E4M3 output (decode-once via dqblk), producing
        // the pre-packed weight buffer the PREPACK GEMM variants above read directly. Gated by the
        // same INFR_F8_COOPMAT=1 + INFR_F8_PREPACK=1.
        ("repack_q8_to_f8", "repack_q8_to_f8", &[]),
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
        // BM=32 row-tile variants of the expert-grid GEMM (see matmul_mmq_experts' `n_used` doc):
        // at small rows-per-expert (Qwen3.6-MoE's 256-expert pool averages ~16/expert at pp512)
        // the default BM=64 tile is ~75% masked waste — a BM=32 tile halves that. Selected
        // per-dispatch by the recorder based on the caller's actual avg rows/expert; qwen3-30B-A3B
        // (128 experts, ~32/expert) still gets BM=64 by default at pp512, so this is additive, not
        // a blanket swap.
        (
            "native_gemm_mmq_q4k",
            "native_gemm_mmq_q4k_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q6k",
            "native_gemm_mmq_q6k_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q8_0",
            "native_gemm_mmq_q8_0_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q5_0",
            "native_gemm_mmq_q5_0_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q5k",
            "native_gemm_mmq_q5k_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
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
        // Two-stage fused argmax+top1-prob (Op::ArgmaxProb, MTP draft-loop accept): same shape as
        // argmax, plus an online-softmax sum_exp carried alongside the (max, idx) reduction.
        ("argmax_prob", "argmax_prob_part", &[]),
        ("argmax_prob", "argmax_prob", &["-DPASS2"]),
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
            // BK64W (double k-stage, half the barriers) measured +14% on the qwen35-4B wide
            // prefill GEMMs (pp512 3811→4330) — Q4K's cheap dqblk leaves the tile barrier-bound.
            // Q5K/Q6K/Q8_0 measured a NET LOSS with it (all-formats build 4330→4000: their
            // heavier decoders collide with the doubled stage), so Q4K only.
            "native_gemm_warp",
            "native_gemm_warp_q4k_ag",
            &["-DFMT_Q4K", "-DA_GLOBAL", "-DBK64W"],
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
        // NATIVE bf16 cooperative-matrix (WMMA) variant of the SAME production warptile above —
        // gated behind INFR_BF16_COOPMAT=1 + caps.bf16_coopmat (RDNA4 native bf16 WMMA; see
        // adapter.rs `bf16cm_ok` / native_gemm_warp.comp's BF16CM doc). -DBF16CM swaps ONLY the
        // coopmat A/B operand type (float16_t -> bfloat16_t, dropping the f16 clamp on FMT_BF16
        // weights) — every other structural choice (staging, tiling, dqblk, epilogue) is the
        // IDENTICAL tuned production kernel `native_gemm_warp_bf16` uses, so this should match its
        // speed while keeping bf16's full exponent range. Replaces the old standalone
        // native_gemm_bf16cm.comp measurement kernel (retired: this variant subsumes it).
        // Default-off; correctness UNVALIDATED on this box (no bf16 coopmat hardware here —
        // compile-checked only), pending an RDNA4 run. No A_GLOBAL variant: the existing
        // `native_gemm_warp_bf16` f16-clamp path doesn't use A_GLOBAL either (Bf16 isn't in
        // `native_gemm_warp_ag_build_spv`'s dtype match), so there's nothing to mirror there.
        (
            "native_gemm_warp",
            "native_gemm_warp_bf16cm",
            &["-DFMT_BF16", "-DBF16CM"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_bf16cm_n128",
            &["-DFMT_BF16", "-DBF16CM", "-DNARROW_N"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q3k", &["-DFMT_Q3K"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q3k_n128",
            &["-DFMT_Q3K", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q3k_sk",
            &["-DFMT_Q3K", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            // A_GLOBAL family (occupancy 2→3 wgs/WGP) + split-K (grid-fill for narrow-n
            // down/o/kv projections). Q3_K was the only warp K-quant missing these, so its GEMMs
            // ran the plain n128 tile at ~9.6 TF (vs 30-52 TF for the ag siblings) — the dominant
            // cost in unsloth's Q2_K mixed quant (down/o/kv are Q3_K). NO BK64W: like Q6K, the
            // heavy Q3_K decoder collides with the doubled k-stage (BK64W is a Q4K-only win).
            "native_gemm_warp",
            "native_gemm_warp_q3k_ag",
            &["-DFMT_Q3K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q3k_n128_ag",
            &["-DFMT_Q3K", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q3k_sk_ag",
            &["-DFMT_Q3K", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
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
        // BM=32 row-tile variants of the A_GLOBAL dense warp GEMM's n128_ag (non-split-K) family
        // — MTP verify's batched-prefill draft window (m≈6-24, growing under the no-rewind
        // fallback) on the default BM=64 tile is mostly masked waste (m=8 → 87.5%). Only the
        // formats the qwen35-4B-UD-Q4_K_XL verify GEMMs actually hit (Q4_K/Q5_K/Q6_K/Q8_0, per
        // INFR_PROF2_SHAPES profiling) get a variant; selected per-dispatch by the recorder from
        // `m` (see `DENSE_SMALL_TILE_MAX_M` in recorder.rs). Same K-accumulation order as BM=64 —
        // tile GRANULARITY only, bit-identical. NO sk_ag (split-K) variants: the split-K family's
        // own `splits` dimension already fills the device at these shapes, so a smaller row tile
        // there measured a net LOSS (`dense_small_m_row_tile_bench`) — BM=64 stays unconditional
        // on that path (see `matmul_native_splitk`'s doc).
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_n128_ag_bm32",
            &["-DFMT_Q4K", "-DNARROW_N", "-DA_GLOBAL", "-DBM32"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_n128_ag_bm32",
            &["-DFMT_Q5K", "-DNARROW_N", "-DA_GLOBAL", "-DBM32"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_n128_ag_bm32",
            &["-DFMT_Q6K", "-DNARROW_N", "-DA_GLOBAL", "-DBM32"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_n128_ag_bm32",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DA_GLOBAL", "-DBM32"],
        ),
        // BM=16 row-tile variants — one coopmat M-frag floor, mirroring the BM=32 family above but
        // for the m<=~12-16 sub-band where BM=32's own tile is still ~half masked waste (see
        // `DENSE_SMALL_TILE_MAX_M16` in recorder.rs for the measured crossover). Same formats as
        // BM32 (Q4_K/Q5_K/Q6_K/Q8_0), same NARROW_N + A_GLOBAL, no split-K variant (unmeasured;
        // BM32 already skips sk_ag for the same reason — see `matmul_native_splitk`'s doc).
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_n128_ag_bm16",
            &["-DFMT_Q4K", "-DNARROW_N", "-DA_GLOBAL", "-DBM16"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_n128_ag_bm16",
            &["-DFMT_Q5K", "-DNARROW_N", "-DA_GLOBAL", "-DBM16"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_n128_ag_bm16",
            &["-DFMT_Q6K", "-DNARROW_N", "-DA_GLOBAL", "-DBM16"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_n128_ag_bm16",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DA_GLOBAL", "-DBM16"],
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
