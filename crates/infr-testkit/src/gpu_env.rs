//! The test-environment knobs every GPU test helper must honour, in one place.
//!
//! `INFR_FORCE_WAVE64` (`kernels.vulkan.force_wave64`) runs the Vulkan backend's wave64 path, the
//! one AMD GCN takes, on a GPU that could pin 32 — so a run with it set is a test OF that path.
//! Helpers that build a backend through `VulkanBackend::new` get it from the environment layer
//! already; helpers that start from `Config::default()` must apply it with [`honour_env`]. And a
//! helper that SKIPS when no backend opens must not skip when the knob is set, nor when `INFR_DEV`
//! names a Vulkan device: there the failure to open IS the result, so [`gpu_or_skip`] panics with
//! it instead.

use std::fmt::Display;

use infr_core::config::{Config, ConfigLayer};

/// Whether `INFR_FORCE_WAVE64` is set, read through the config's environment layer like every
/// other `INFR_*` knob.
pub fn force_wave64() -> bool {
    let env = ConfigLayer::env().expect("test environment config");
    Config::load_from_layers(&[env]).kernels.vulkan.force_wave64
}

/// Apply the GPU-path knobs of the test environment to a config a helper built by hand: the forced
/// wave64 path; `INFR_TUNE_NO_ROWS` (`kernels.vulkan.tuning.rows`), which runs every tuning
/// parameter at its generic default, so a run with it set tests the kernels a device with no table
/// row gets; and `INFR_TUNE` (`kernels.vulkan.tune`), which forces single tuning parameters, so
/// a run with `INFR_TUNE=gemv.kq_fma=true` tests the kernels a device without an accelerated int8
/// dot gets. A test that pins a parameter itself adds its setting after this, at a higher level
/// (see [`env_config_with_tune`]).
pub fn honour_env(cfg: &mut Config) {
    let env = Config::load_from_layers(&[ConfigLayer::env().expect("test environment config")]);
    cfg.kernels.vulkan.force_wave64 = env.kernels.vulkan.force_wave64;
    cfg.kernels.vulkan.tuning.rows = env.kernels.vulkan.tuning.rows;
    cfg.kernels
        .vulkan
        .tune
        .entries
        .extend(env.kernels.vulkan.tune.entries);
}

/// The Vulkan device `INFR_DEV` names (`device.dev`, `VulkanN`), if the test environment names
/// one. A run that names another backend's device expects no Vulkan backend.
fn named_vulkan_device() -> Option<String> {
    let env = ConfigLayer::env().expect("test environment config");
    Config::load_from_layers(&[env])
        .device
        .dev
        .filter(|d| d.to_ascii_lowercase().starts_with("vulkan"))
}

/// A backend for a test that skips when there is none: `Some` on success; on failure, `None` (the
/// caller prints its own skip line) — unless the environment says a backend was expected, where
/// the failure to open IS the result and this panics with it rather than letting the test report
/// a pass it never ran: `INFR_FORCE_WAVE64` is set (the forced path is unavailable on this
/// device), or `INFR_DEV` names a Vulkan device (every GPU gate run does).
pub fn gpu_or_skip<B, E: Display>(opened: Result<B, E>) -> Option<B> {
    match opened {
        Ok(be) => Some(be),
        Err(e) if force_wave64() => {
            panic!("INFR_FORCE_WAVE64 is set but the Vulkan backend did not open: {e}")
        }
        Err(e) => {
            if let Some(dev) = named_vulkan_device() {
                panic!("INFR_DEV={dev} is set but the Vulkan backend did not open: {e}");
            }
            None
        }
    }
}

/// The test environment's config with Vulkan tuning parameters forced on top of it, at
/// command-line level, so each wins over the table, a tuning file and `INFR_TUNE`. For a test
/// that must run one side of a tuning parameter on every device, whatever the device's default:
/// `env_config_with_tune(&[("gemv.kq_fma", "true")])`. Every other knob of the environment
/// (`INFR_DEV`, the capability masks, `INFR_FORCE_WAVE64`) still applies.
pub fn env_config_with_tune(settings: &[(&str, &str)]) -> Config {
    use infr_core::config::tune::{TuneEntry, TuneLevel};
    let mut cfg = Config::load_from_layers(&[ConfigLayer::env().expect("test environment config")]);
    for (key, value) in settings {
        cfg.kernels.vulkan.tune.entries.push(TuneEntry {
            key: (*key).to_string(),
            value: (*value).to_string(),
            level: TuneLevel::Cli,
        });
    }
    cfg
}
