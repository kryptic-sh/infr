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
        ("attn_partial", "attn_partial_dyn", &["-DUSE_PARAMS"]),
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
        ("add_scaled", "add_scaled", &[]),
        ("scale", "scale", &[]),
        ("softcap", "softcap", &[]),
        ("gather_rows", "gather_rows", &[]),
        ("scatter_add_rows", "scatter_add_rows", &[]),
        ("silu_mul", "silu_mul", &[]),
        ("silu_mul", "gelu_mul", &["-DGELU"]),
        ("silu_mul_fused", "silu_mul_fused", &[]),
        ("silu_mul_fused", "gelu_mul_fused", &["-DGELU"]),
        ("store_f16", "store_f16", &[]),
        ("store_f16", "store_f16_dyn", &["-DUSE_PARAMS"]),
        ("rope", "rope", &[]),
        ("linear_f16", "linear_f16", &[]),
        ("linear_bf16", "linear_bf16", &[]),
        ("linear_f32", "linear_f32", &[]),
        ("matmul_f32", "matmul_f32", &[]),
        ("linear_q", "linear_q", &[]),
        ("linear_res", "linear_res", &[]),
        ("linear_res_q", "linear_res_q", &[]),
        ("attention", "attention", &[]),
        ("attn_combine", "attn_combine", &[]),
        ("attention_kv", "attention_kv", &[]),
        ("attention_kv", "attention_kv_dyn", &["-DUSE_PARAMS"]),
        ("qk_norm_rope", "qk_norm_rope", &[]),
        ("qk_norm_rope", "qk_norm_rope_dyn", &["-DUSE_PARAMS"]),
        ("qk_norm_rope", "qk_norm_rope_ff", &["-DFREQ_FACTORS"]),
        ("attn_in", "attn_in", &[]),
        ("attn_in", "attn_in_dyn", &["-DUSE_PARAMS"]),
        ("ffn_in", "ffn_in", &[]),
        ("ffn_in_q", "ffn_in_q", &[]),
        ("attn_in_q", "attn_in_q", &[]),
        // Native-block dequant GEMVs: one .spv per (quant format, residual) from one source.
        ("native_gemv", "native_q8_0", &["-DFMT_Q8_0"]),
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
        ("native_mmv_id_q4k", "native_mmv_id_q4k", &[]),
        ("native_gemm_mmq_q4k", "native_gemm_mmq_q4k", &[]),
        ("native_gemm_mmq_q6k", "native_gemm_mmq_q6k", &[]),
        ("moe_topk", "moe_topk", &[]),
        ("argmax", "argmax", &[]),
        ("moe_sample", "moe_sample", &[]),
        ("moe_bucket_count", "moe_bucket_count", &[]),
        ("moe_bucket_scan", "moe_bucket_scan", &[]),
        ("moe_bucket_scatter", "moe_bucket_scatter", &[]),
        ("add_scaled_id", "add_scaled_id", &[]),
        // Native-block prefill GEMMs: one .spv per quant format (coopmat tiled, no residual).
        ("native_gemm_warp", "native_gemm_warp_q4k", &["-DFMT_Q4K"]),
        ("native_gemm_warp", "native_gemm_warp_q6k", &["-DFMT_Q6K"]),
        ("native_gemm_warp", "native_gemm_warp_q8_0", &["-DFMT_Q8_0"]),
        ("native_gemm", "native_gemm_q8_0", &["-DFMT_Q8_0"]),
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
