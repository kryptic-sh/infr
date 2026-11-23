//! The test-environment knobs every GPU test helper must honour, in one place.
//!
//! `INFR_FORCE_WAVE64` (`kernels.vulkan.force_wave64`) runs the Vulkan backend's wave64 path, the
//! one AMD GCN takes, on a GPU that could pin 32 — so a run with it set is a test OF that path.
//! Helpers that build a backend through `VulkanBackend::new` get it from the environment layer
//! already; helpers that start from `Config::default()` must apply it with [`honour_env`]. And a
//! helper that SKIPS when no backend opens must not skip when the knob is set, nor when `INFR_DEV`
//! names a Vulkan device: there the failure to open IS the result, so [`gpu_or_skip`] panics with
//! it instead. [`gpu_or_skip_noted`] also writes the skip line, and [`gpu_devices_or_skip_noted`]
//! is its twin for a test that lists the devices.

use std::fmt::Display;
use std::io::Write;

use infr_core::config::tune::TUNE_ALIASES;
use infr_core::config::{Config, ConfigLayer, PartialConfig};

/// Whether `INFR_FORCE_WAVE64` is set, read through the config's environment layer like every
/// other `INFR_*` knob.
pub fn force_wave64() -> bool {
    let env = ConfigLayer::env().expect("test environment config");
    Config::load_from_layers(&[env]).kernels.vulkan.force_wave64
}

/// Apply the GPU-path knobs of the test environment to a config a helper built by hand: the forced
/// wave64 path; `INFR_TUNE_NO_ROWS` (`kernels.vulkan.tuning.rows`), which runs every tuning
/// parameter at its generic default, so a run with it set tests the kernels a device with no table
/// row gets; `INFR_TUNE` (`kernels.vulkan.tune`), which forces single tuning parameters, so
/// a run with `INFR_TUNE=gemv.kq_fma=true` tests the kernels a device without an accelerated int8
/// dot gets; and every older knob that is an alias of a tuning parameter ([`TUNE_ALIASES`]:
/// `INFR_KQ_FMA_GEMV`, `INFR_GFX_QUEUE` and the rest), so the alias spelling selects the same
/// kernels here as it does in a backend opened from the environment. A test that pins a parameter
/// itself adds its setting after this, at a higher level (see [`env_config_with_tune`]).
pub fn honour_env(cfg: &mut Config) {
    carry_gpu_knobs(&ConfigLayer::env().expect("test environment config"), cfg);
}

/// [`honour_env`] over an environment layer the caller parsed.
fn carry_gpu_knobs(layer: &PartialConfig, cfg: &mut Config) {
    let env = Config::load_from_layers(std::slice::from_ref(layer));
    cfg.kernels.vulkan.force_wave64 = env.kernels.vulkan.force_wave64;
    cfg.kernels.vulkan.tuning.rows = env.kernels.vulkan.tuning.rows;
    // The alias knobs the environment set, each by its config path, as typed values.
    let mut aliases = PartialConfig::default();
    for &path in TUNE_ALIASES {
        layer.copy_path_to(path, &mut aliases);
    }
    aliases.apply(cfg);
    // The tuning settings, and the record of which aliases this layer set, which the tuning
    // resolver orders them by.
    let tune = &mut cfg.kernels.vulkan.tune;
    tune.entries.extend(env.kernels.vulkan.tune.entries);
    tune.aliases.extend(env.kernels.vulkan.tune.aliases);
}

/// The Vulkan device the environment layer names (`INFR_DEV`, `device.dev`), if it names one,
/// decided by the predicate the backend's own resolver is built on
/// ([`infr_core::config::vulkan_dev_spec`]): a bare index and a padded value name a device as
/// much as `VulkanN` does, and `cpu`, `metal` and an empty value name none.
fn named_vulkan_device_in(layer: &PartialConfig) -> Option<String> {
    let cfg = Config::load_from_layers(std::slice::from_ref(layer));
    infr_core::config::vulkan_dev_spec(cfg.device.dev.as_deref()).map(str::to_string)
}

/// [`named_vulkan_device_in`] the process environment.
fn named_vulkan_device() -> Option<String> {
    named_vulkan_device_in(&ConfigLayer::env().expect("test environment config"))
}

/// Panic with `e` if the test environment says a backend was expected (the cases
/// [`gpu_or_skip`] lists). The skip policy of every test that goes through this module's
/// helpers; a test that classifies the failure itself has its own.
fn refuse_an_expected_backend(e: &dyn Display) {
    if force_wave64() {
        panic!("INFR_FORCE_WAVE64 is set but the Vulkan backend did not open: {e}");
    }
    if let Some(dev) = named_vulkan_device() {
        panic!("INFR_DEV={dev} is set but the Vulkan backend did not open: {e}");
    }
}

/// Write a test's `skip:` line. It goes to stderr with `writeln!`: the libtest harness shows it
/// beside a passing test, where it would capture an `eprintln!`; nextest hides a passing test's
/// output either way, and shows it with `--success-output immediate` or `--no-capture`.
fn note_skip(why: &dyn Display) {
    writeln!(std::io::stderr(), "skip: {why}").expect("stderr");
}

/// A backend for a test that skips when there is none: `Some` on success; on failure, `None` (the
/// caller prints its own skip line) — unless the environment says a backend was expected, where
/// the failure to open IS the result and this panics with it rather than letting the test report
/// a pass it never ran: `INFR_FORCE_WAVE64` is set (the forced path is unavailable on this
/// device), or `INFR_DEV` names a Vulkan device (every GPU gate run does).
pub fn gpu_or_skip<B, E: Display>(opened: Result<B, E>) -> Option<B> {
    match opened {
        Ok(be) => Some(be),
        Err(e) => {
            refuse_an_expected_backend(&e);
            None
        }
    }
}

/// [`gpu_or_skip`] that writes the `skip:` line itself, with the reason the backend did not open,
/// so every test that skips for want of a backend says so in the same words.
pub fn gpu_or_skip_noted<B, E: Display>(opened: Result<B, E>) -> Option<B> {
    match opened {
        Ok(be) => Some(be),
        Err(e) => {
            refuse_an_expected_backend(&e);
            note_skip(&format_args!("the Vulkan backend did not open: {e}"));
            None
        }
    }
}

/// [`gpu_or_skip_noted`] for a test that lists the devices instead of opening one. A listing
/// succeeds even when `INFR_DEV` names a device that is not there (it only marks no device as
/// the pick), so with a Vulkan device named this also panics when `is_pick` holds for none of
/// the listed devices, where a backend would not have opened.
pub fn gpu_devices_or_skip_noted<D, E: Display>(
    listed: Result<Vec<D>, E>,
    is_pick: impl Fn(&D) -> bool,
) -> Option<Vec<D>> {
    match listed {
        Ok(devs) => {
            if let Some(dev) = named_vulkan_device() {
                assert!(
                    devs.iter().any(is_pick),
                    "INFR_DEV={dev} is set but names none of the {} Vulkan devices listed",
                    devs.len()
                );
            }
            Some(devs)
        }
        Err(e) => {
            refuse_an_expected_backend(&e);
            note_skip(&format_args!("no Vulkan devices were listed: {e}"));
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

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::config::manifest::KEYS;
    use infr_core::config::{TuneEntry, TuneLevel};

    /// The environment layer of `vars`, parsed as the process environment is.
    fn env_layer(vars: &[(&str, &str)]) -> PartialConfig {
        infr_core::config::env::parse(&|k| {
            vars.iter()
                .find(|(name, _)| *name == k)
                .map(|(_, v)| (*v).to_string())
        })
        .expect("environment layer")
    }

    /// Every alias of a tuning parameter that the environment can set reaches a hand-built
    /// config with the value a config loaded from that environment holds, and with the record
    /// that the environment set it. The variables come from the config manifest, so an alias
    /// added there or to [`TUNE_ALIASES`] is held to this without a list here.
    #[test]
    fn honour_env_carries_every_tuning_alias() {
        let mut carried = Vec::new();
        for key in KEYS.iter().filter(|k| TUNE_ALIASES.contains(&k.path)) {
            let layer = env_layer(&[(key.env, key.sample)]);
            assert!(layer.is_path_set(key.path), "{} sets {}", key.env, key.path);
            let want = Config::load_from_layers(std::slice::from_ref(&layer));
            let mut cfg = Config::default();
            carry_gpu_knobs(&layer, &mut cfg);
            assert_eq!(
                cfg.get_path(key.path),
                want.get_path(key.path),
                "{}={}",
                key.env,
                key.sample
            );
            assert_eq!(
                cfg.kernels.vulkan.tune.alias_level(key.path),
                Some(TuneLevel::Env),
                "{}",
                key.env
            );
            // Nothing else moved: the whole config is the one the environment loads.
            assert_eq!(cfg, want, "{}={}", key.env, key.sample);
            carried.push(key.path);
        }
        for path in TUNE_ALIASES {
            assert!(
                carried.contains(path),
                "{path}: no environment variable in the manifest sets this alias"
            );
        }
    }

    /// An alias arrives as the typed value the environment gave it: an optional text holding
    /// the empty string or the word `none` stays set, which a copy through its display string
    /// reads back as unset.
    #[test]
    fn honour_env_carries_an_alias_as_its_typed_value() {
        let layer = env_layer(&[("INFR_GEMV_VARIANT", "")]);
        let mut cfg = Config::default();
        carry_gpu_knobs(&layer, &mut cfg);
        assert_eq!(cfg.kernels.vulkan.gemv.variant.as_deref(), Some(""));
        for key in KEYS.iter().filter(|k| TUNE_ALIASES.contains(&k.path)) {
            for value in ["", "none", "None", " "] {
                let get = |k: &str| (k == key.env).then(|| value.to_string());
                let Ok(layer) = infr_core::config::env::parse(&get) else {
                    continue; // this variable refuses the value outright
                };
                let want = Config::load_from_layers(std::slice::from_ref(&layer));
                let mut cfg = Config::default();
                carry_gpu_knobs(&layer, &mut cfg);
                assert_eq!(cfg, want, "{}={value:?}", key.env);
            }
        }
    }

    /// A Vulkan device is named by whatever the backend's resolver takes as an index, however it
    /// is spelled, and by nothing else.
    #[test]
    fn a_vulkan_device_is_named_as_the_backend_reads_it() {
        let named = |v: &str| named_vulkan_device_in(&env_layer(&[("INFR_DEV", v)]));
        assert_eq!(named("Vulkan1").as_deref(), Some("Vulkan1"));
        assert_eq!(named("vulkan1").as_deref(), Some("vulkan1"));
        assert_eq!(named("9").as_deref(), Some("9"), "a bare index");
        assert_eq!(named(" Vulkan9 ").as_deref(), Some("Vulkan9"), "padded");
        assert_eq!(named(" 9\t").as_deref(), Some("9"), "a padded bare index");
        // A value the resolver refuses is still a Vulkan device asked for.
        assert_eq!(named("VulkanX").as_deref(), Some("VulkanX"));
        for none in ["", "  ", "cpu", "CPU", "metal", " Metal "] {
            assert_eq!(named(none), None, "{none:?}");
        }
        assert_eq!(named_vulkan_device_in(&env_layer(&[])), None, "unset");
    }

    /// The aliases the PR follow-ups added select their side of the parameter: a value that
    /// differs from the default arrives, not only the record of it.
    #[test]
    fn honour_env_carries_the_kq_fma_and_queue_aliases() {
        let mut cfg = Config::default();
        carry_gpu_knobs(
            &env_layer(&[("INFR_KQ_FMA_GEMV", "0"), ("INFR_GFX_QUEUE", "1")]),
            &mut cfg,
        );
        assert_eq!(cfg.kernels.vulkan.kq_fma_gemv, Some(false));
        assert!(cfg.kernels.vulkan.gfx_queue);
    }

    /// `INFR_TUNE`, `INFR_TUNE_NO_ROWS` and `INFR_FORCE_WAVE64` arrive, after what the helper set
    /// itself, and a knob that is not a GPU-path one does not.
    #[test]
    fn honour_env_keeps_the_helpers_own_settings() {
        let mut cfg = Config::default();
        cfg.kernels.vulkan.tune.entries.push(TuneEntry {
            key: "gemv.kq_fma".to_string(),
            value: "true".to_string(),
            level: TuneLevel::Config,
        });
        carry_gpu_knobs(
            &env_layer(&[
                ("INFR_TUNE", "gemv_sg.rows=4"),
                ("INFR_TUNE_NO_ROWS", "1"),
                ("INFR_FORCE_WAVE64", "1"),
                ("INFR_DEV", "Vulkan7"),
            ]),
            &mut cfg,
        );
        let keys: Vec<&str> = cfg
            .kernels
            .vulkan
            .tune
            .entries
            .iter()
            .map(|e| e.key.as_str())
            .collect();
        assert_eq!(keys, ["gemv.kq_fma", "gemv_sg.rows"]);
        assert!(!cfg.kernels.vulkan.tuning.rows);
        assert!(cfg.kernels.vulkan.force_wave64);
        assert_eq!(cfg.device.dev, None);
    }
}
