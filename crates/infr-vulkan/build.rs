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
        ("gemm_proj", "gemm_proj", &[]),
        ("attn_prefill", "attn_prefill", &[]),
        ("attn_prefill_partial", "attn_prefill_partial", &[]),
        ("attn_prefill_combine", "attn_prefill_combine", &[]),
        ("attn_partial", "attn_partial", &[]),
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
