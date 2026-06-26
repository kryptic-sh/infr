use std::process::Command;

/// Compile GLSL compute shaders (which need features WGSL/naga can't express, e.g.
/// cooperative matrix) to SPIR-V via `glslc` at build time. Output to OUT_DIR.
fn main() {
    println!("cargo:rerun-if-changed=shaders");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    for name in ["gemm_coopmat", "gemm_coopmat_tiled", "gemm_proj"] {
        let src = format!("shaders/{name}.comp");
        let dst = format!("{out}/{name}.spv");
        println!("cargo:rerun-if-changed={src}");
        let status = Command::new("glslc")
            .args([
                "-fshader-stage=comp",
                "--target-env=vulkan1.3",
                "-O",
                &src,
                "-o",
                &dst,
            ])
            .status()
            .expect("failed to run glslc — install shaderc (provides glslc)");
        assert!(status.success(), "glslc failed for {src}");
    }
}
