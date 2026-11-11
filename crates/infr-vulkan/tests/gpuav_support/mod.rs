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

/// Assert the layer logged no error to `log`. Call after the backend is dropped so every queued
/// report has been written. GPU-AV always logs a settings warning at instance creation, so a missing
/// file means the layer never loaded rather than no findings: the test then still passes on its
/// numeric half, and says on stderr that it did not verify the bound. That notice is written to the
/// stderr handle directly, which the test harness does not capture, so it shows without
/// `--nocapture`.
pub fn assert_clean(log: &Path, test: &str) {
    let Ok(text) = std::fs::read_to_string(log) else {
        let _ = writeln!(
            std::io::stderr().lock(),
            "WARNING: {test} did NOT verify the out-of-bounds bound: the Khronos validation layer \
             is not installed, so only the numeric result was checked"
        );
        return;
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
}
