//! Emit the ROCm/HIP library search path so `-lamdhip64` / `-lhiprtc` (declared
//! in `src/ffi.rs`) resolve at link time. Only fires when the `rocm` feature is
//! active — the default (empty-lib) build links nothing HIP and stays portable.
//!
//! The path is taken from `$ROCM_PATH` (the standard ROCm env var, e.g.
//! `/opt/rocm`), falling back to `/opt/rocm`. Override with `ROCM_PATH` if HIP
//! lives elsewhere.

fn main() {
    // `CARGO_FEATURE_ROCM` is set by cargo iff the `rocm` feature is enabled.
    if std::env::var_os("CARGO_FEATURE_ROCM").is_none() {
        return;
    }
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    println!("cargo:rustc-link-search=native={rocm_path}/lib");
}
