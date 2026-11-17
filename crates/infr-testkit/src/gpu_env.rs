//! The test-environment knobs every GPU test helper must honour, in one place.
//!
//! `INFR_FORCE_WAVE64` (`kernels.vulkan.force_wave64`) runs the Vulkan backend's wave64 path, the
//! one AMD GCN takes, on a GPU that could pin 32 — so a run with it set is a test OF that path.
//! Helpers that build a backend through `VulkanBackend::new` get it from the environment layer
//! already; helpers that start from `Config::default()` must apply it with [`honour_env`]. And a
//! helper that SKIPS when no backend opens must not skip when the knob is set: there the failure
//! to open IS the result, so [`gpu_or_skip`] panics with it instead.

use std::fmt::Display;

use infr_core::config::{Config, ConfigLayer};

/// Whether `INFR_FORCE_WAVE64` is set, read through the config's environment layer like every
/// other `INFR_*` knob.
pub fn force_wave64() -> bool {
    let env = ConfigLayer::env().expect("test environment config");
    Config::load_from_layers(&[env]).kernels.vulkan.force_wave64
}

/// Apply the GPU-path knobs of the test environment to a config a helper built by hand: the forced
/// wave64 path, and `INFR_TUNE_NO_ROWS` (`kernels.vulkan.tuning.rows`), which runs every tuning
/// parameter at its generic default, so a run with it set tests the kernels a device with no table
/// row gets.
pub fn honour_env(cfg: &mut Config) {
    let env = Config::load_from_layers(&[ConfigLayer::env().expect("test environment config")]);
    cfg.kernels.vulkan.force_wave64 = env.kernels.vulkan.force_wave64;
    cfg.kernels.vulkan.tuning.rows = env.kernels.vulkan.tuning.rows;
}

/// A backend for a test that skips when there is none: `Some` on success; on failure, `None` (the
/// caller prints its own skip line) — unless `INFR_FORCE_WAVE64` is set, where a backend that does not open means the
/// forced path is unavailable on this device, which panics with the reason rather than letting
/// every test report a pass it never ran.
pub fn gpu_or_skip<B, E: Display>(opened: Result<B, E>) -> Option<B> {
    match opened {
        Ok(be) => Some(be),
        Err(e) if force_wave64() => {
            panic!("INFR_FORCE_WAVE64 is set but the Vulkan backend did not open: {e}")
        }
        Err(_) => None,
    }
}
