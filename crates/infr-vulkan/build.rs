use std::process::Command;

/// Compile GLSL compute shaders (which need features WGSL/naga can't express, e.g.
/// cooperative matrix) to SPIR-V via `glslc` at build time. Output to OUT_DIR.
fn main() {
    println!("cargo:rerun-if-changed=shaders");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
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
        ("attn_qk", "attn_qk", &[]),
        ("attn_qk_warp", "attn_qk_warp", &[]),
        ("attn_flash", "attn_flash", &[]),
        ("attn_flash_partial", "attn_flash_partial", &[]),
        ("attn_flash_warp", "attn_flash_warp", &[]),
        ("attn_flash_reg", "attn_flash_reg", &[]),
        ("attn_flash_combine", "attn_flash_combine", &[]),
        ("attn_softmax", "attn_softmax", &[]),
        ("attn_pv", "attn_pv", &[]),
        ("attn_pv_warp", "attn_pv_warp", &[]),
        ("attn_pv_reduce", "attn_pv_reduce", &[]),
        ("rmsnorm", "rmsnorm", &[]),
        ("add", "add", &[]),
        ("silu_mul", "silu_mul", &[]),
        ("silu_mul_fused", "silu_mul_fused", &[]),
        ("store_f16", "store_f16", &[]),
        ("rope", "rope", &[]),
        ("linear_f16", "linear_f16", &[]),
        ("linear_bf16", "linear_bf16", &[]),
        ("linear_f32", "linear_f32", &[]),
        ("matmul_f32", "matmul_f32", &[]),
        ("linear_q", "linear_q", &[]),
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
