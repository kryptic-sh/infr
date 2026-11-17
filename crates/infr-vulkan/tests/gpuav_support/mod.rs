//! GPU-assisted validation for tests whose bug is an out-of-bounds read that cannot change any
//! output (padding workgroups, rows past `m`): the Khronos layer is the only thing that sees it.
//!
//! [`enable`] turns the layer and GPU-AV on through the loader's environment and points its log at
//! a per-process file, so it must run before the first Vulkan call. `set_var` races any other
//! thread reading the environment, so a binary using this holds exactly one test.

use std::io::Write;
use std::path::{Path, PathBuf};

/// Enable the layer with GPU-AV for every instance this process creates; returns the log path.
pub fn enable(name: &str) -> PathBuf {
    let log = std::env::temp_dir().join(format!("infr-vk-{name}-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&log);
    std::env::set_var("VK_LOADER_LAYERS_ENABLE", "VK_LAYER_KHRONOS_validation");
    std::env::set_var("VK_KHRONOS_VALIDATION_GPUAV_ENABLE", "1");
    std::env::set_var("VK_KHRONOS_VALIDATION_LOG_FILENAME", &log);
    log
}

/// Assert the layer logged no error to `log`, and say whether it could. Call after the backend is
/// dropped so every queued report has been written. GPU-AV always logs a settings warning at
/// instance creation, so a missing file means the layer never loaded rather than no findings: then
/// this returns `false` and writes `test` did NOT verify `unverified` (what the caller's check
/// depends on the layer for) to stderr. That notice goes to the stderr handle directly, which the
/// test harness does not capture, so it shows without `--nocapture`.
pub fn assert_clean(log: &Path, test: &str, unverified: &str) -> bool {
    let Ok(text) = std::fs::read_to_string(log) else {
        let _ = writeln!(
            std::io::stderr().lock(),
            "WARNING: {test} did NOT verify {unverified}: the Khronos validation layer is not \
             installed"
        );
        return false;
    };
    let _ = std::fs::remove_file(log);
    let errors: Vec<&str> = text
        .split("\n\n")
        .filter(|m| m.contains("Validation Error"))
        .collect();
    assert!(
        errors.is_empty(),
        "{} validation errors, first:\n{}",
        errors.len(),
        errors[0]
    );
    true
}
