//! Where `infr` keeps its configuration and its caches.

use std::path::PathBuf;

/// The invoking user's home directory, or `None` when it cannot be determined.
///
/// The one spelling: `$HOME` on Unix, and the `FOLDERID_Profile` Known Folder on Windows, where
/// `$HOME` is normally unset — a bare `env::var("HOME")` finds nothing there and silently yields
/// an empty string, which then builds a RELATIVE path rather than failing.
pub fn home_dir() -> Option<PathBuf> {
    dirs::home_dir()
}

/// `$XDG_CONFIG_HOME/infr` → `~/.config/infr`, or `None` when the home directory is unknown.
///
/// **Not `dirs::config_dir()`**, which would resolve to `~/Library/Application Support` on macOS
/// and `%APPDATA%` on Windows and so silently relocate the file for anyone already using one. The
/// XDG layout is the same on every platform because infr ships a CLI, not a signed `.app` bundle,
/// and the resolution differs from a hand-rolled `$HOME` lookup in exactly one place that matters:
/// `dirs::home_dir()` reads the `FOLDERID_Profile` Known Folder on Windows, where `$HOME` is
/// normally unset and a `var_os("HOME")` lookup therefore finds nothing at all.
///
/// Does NOT create the directory.
pub fn config_dir() -> Option<PathBuf> {
    xdg_base(std::env::var_os("XDG_CONFIG_HOME"), ".config").map(|b| b.join("infr"))
}

/// `$XDG_CACHE_HOME/infr` → `~/.cache/infr`, or `None` when the home directory is unknown.
///
/// See [`config_dir`] for why this is spelled out rather than delegated to `dirs::cache_dir()`.
/// Does NOT create the directory — callers that want one no matter what apply their own fallback.
pub fn cache_dir() -> Option<PathBuf> {
    cache_home().map(|b| b.join("infr"))
}

/// The XDG cache BASE — `$XDG_CACHE_HOME` when absolute, else `~/.cache` — without any
/// application name appended.
///
/// Separate from [`cache_dir`] because not everything under it belongs to infr: the Hugging Face
/// hub cache is `<base>/huggingface/hub`, a layout owned by `huggingface_hub` and shared with
/// llama.cpp, and infr must agree with it byte for byte or it re-downloads models another tool
/// already has. Like the rest of this module it is the SAME shape on every platform, which is what
/// `huggingface_hub` itself does: it reads `$XDG_CACHE_HOME` and falls back to `~/.cache`
/// unconditionally, with no `%LOCALAPPDATA%` or `~/Library/Caches` arm.
pub fn cache_home() -> Option<PathBuf> {
    xdg_base(std::env::var_os("XDG_CACHE_HOME"), ".cache")
}

/// The operating system's OWN cache directory, which is a different place from [`cache_home`] on
/// everything but Linux: `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows.
///
/// Exposed for ONE purpose — finding data an earlier version of infr wrote there before it agreed
/// with `huggingface_hub`'s layout — so that upgrading does not silently re-download multi-gigabyte
/// models. New data should not be written here; use [`cache_home`] or [`cache_dir`].
pub fn os_cache_home() -> Option<PathBuf> {
    dirs::cache_dir()
}

/// The XDG base directory, given the raw environment value and the `~`-relative fallback.
///
/// Takes the value rather than the variable NAME so the policy is testable without mutating the
/// process environment — `set_var` races every other thread in the test binary.
fn xdg_base(from_env: Option<std::ffi::OsString>, default_subdir: &str) -> Option<PathBuf> {
    // An empty or relative value counts as unset, per the XDG spec (§ Basics): the variable must
    // hold an absolute path, and honouring a relative one would put the config wherever the
    // process happened to be started from.
    match from_env.map(PathBuf::from).filter(|p| p.is_absolute()) {
        Some(p) => Some(p),
        None => Some(home_dir()?.join(default_subdir)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the crate-level move: both helpers answer on every platform this
    /// compiles for, including the one where `$HOME` is not a thing.
    #[test]
    fn both_directories_resolve_on_this_platform() {
        let config = config_dir().expect("no config dir on this platform");
        let cache = cache_dir().expect("no cache dir on this platform");
        assert!(
            config.is_absolute(),
            "config dir must be absolute: {config:?}"
        );
        assert!(cache.is_absolute(), "cache dir must be absolute: {cache:?}");
        assert!(config.ends_with("infr"));
        assert!(cache.ends_with("infr"));
        // ...and they are not the same directory, which a copy-paste slip would make them.
        assert_ne!(config, cache);
    }

    /// An absolute `XDG_*` value wins; a relative or empty one is ignored rather than honoured,
    /// which would otherwise put the config wherever the process happened to be started from.
    #[test]
    fn only_an_absolute_xdg_value_is_a_base() {
        let home_fallback = home_dir().map(|h| h.join(".config"));
        let absolute = if cfg!(windows) {
            "C:\\xdg-test"
        } else {
            "/xdg-test"
        };

        assert_eq!(
            xdg_base(Some(absolute.into()), ".config"),
            Some(PathBuf::from(absolute)),
        );
        assert_eq!(
            xdg_base(Some("not/absolute".into()), ".config"),
            home_fallback
        );
        assert_eq!(xdg_base(Some("".into()), ".config"), home_fallback);
        assert_eq!(xdg_base(None, ".config"), home_fallback);
        // The absolute case must not be the fallback in disguise — otherwise every assertion
        // above passes for a `xdg_base` that ignores its argument entirely.
        assert_ne!(xdg_base(Some(absolute.into()), ".config"), home_fallback);
    }
}
