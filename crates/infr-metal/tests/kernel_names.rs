#![cfg(target_os = "macos")] // like the lib: empty on other targets (metal dep is cfg-gated)

//! Tripwire for the silent-missing-kernel class: every kernel name the executor can dispatch
//! must exist in the compiled MSL library.
//!
//! Why this exists: the executor's pipeline cap-checks treat a missing function as "capability
//! absent" and fall back gracefully — correct behavior for genuinely capped devices, but it
//! also means a kernel that silently VANISHES from the library (e.g. a rebase restoring a stale
//! copy of the shader source, as happened once: the native-read KV kernels disappeared and
//! q4_0 decode at depth ran 3x slower with zero errors anywhere) degrades performance invisibly
//! instead of failing. This test turns that class red: it scrapes every kernel-shaped string
//! literal out of exec.rs and asserts each one resolves in the library.

/// A string literal in exec.rs is "kernel-shaped" when it starts with one of the kernel-family
/// prefixes. Names are scraped from the SOURCE (include_str! at compile time), so a new dispatch
/// site is covered automatically — no list to maintain. If a literal matches a prefix but is
/// not meant to be a kernel, extend the exclusion below with a comment.
const KERNEL_PREFIXES: &[&str] = &[
    "add_",
    "argmax",
    "attention",
    "attnflash",
    "attnvec",
    "cast_",
    "cmm_",
    "conv1d_",
    "copy_",
    "deltanet_",
    "dequant_",
    "embed_gather_",
    "gatedact",
    "linear_",
    "moe_",
    "qknorm",
    "rmsnorm",
    "rope_",
    "sample_f32",
    "scale_f32",
    "softcap",
    "writekv_",
];

#[test]
fn iq4nl_has_a_native_four_row_decode_body() {
    let src = include_str!("../shaders/linear.metal");
    assert!(src.contains("inline void linear_iq4nl_body"));
    assert!(src.contains("kernel void linear_iq4nl_add"));
}

#[test]
#[ignore = "requires a Metal GPU"]
fn every_dispatchable_kernel_exists_in_the_library() {
    let src = include_str!("../src/exec.rs");
    // Scrape "..." literals. exec.rs kernel names are plain [a-z0-9_] identifiers.
    let mut names: Vec<String> = Vec::new();
    let mut rest = src;
    while let Some(q) = rest.find('"') {
        rest = &rest[q + 1..];
        let Some(end) = rest.find('"') else { break };
        let lit = &rest[..end];
        rest = &rest[end + 1..];
        if !lit.is_empty()
            && lit
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            && KERNEL_PREFIXES.iter().any(|p| lit.starts_with(p))
            && !names.iter().any(|n| n == lit)
        {
            names.push(lit.to_string());
        }
    }
    assert!(
        names.len() > 40,
        "kernel-name scrape looks broken: only {} names found",
        names.len()
    );

    let dev = metal::Device::system_default().expect("no Metal device");
    // Resolve each name against EXACTLY the source the runtime compiles (infr_metal::msl_source
    // is the backend's own assembly — no separately-maintained file list to drift).
    let lib = dev
        .new_library_with_source(&infr_metal::msl_source(), &metal::CompileOptions::new())
        .expect("MSL library compile");
    let have: std::collections::HashSet<String> = lib.function_names().into_iter().collect();

    let missing: Vec<&String> = names.iter().filter(|n| !have.contains(*n)).collect();
    assert!(
        missing.is_empty(),
        "exec.rs dispatches kernels that do NOT exist in the compiled MSL library \
         (the runtime would silently treat them as capability-absent and fall back): {missing:?}"
    );
}
