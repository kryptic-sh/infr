//! The Vulkan kernel tuning settings as the config layers see them: `[kernels.vulkan.tune]`,
//! `INFR_TUNE` and `--set kernels.vulkan.tune.<key>=<value>`, plus the record of which layer set
//! each older knob that is now an ALIAS of a tuning parameter.
//!
//! The config layers only check the SHAPE of a key (`family.param[@qualifier]`). Whether that
//! parameter exists, and whether its value is in range, is decided by the Vulkan backend's tuning
//! resolver (`infr-vulkan`'s `tuning` module), which owns the parameter list; an unknown key or a
//! bad value is a hard error there, naming the key.
//!
//! Why the layer is recorded here at all: a resolved `Config` is a fold of every layer, so it no
//! longer says whether `flash_splits` came from the file or from `INFR_FLASH_SPLITS`. The tuning
//! resolver has to know, because the file, the environment and the command line are three
//! different levels of the tuning order and a tuning key at one level must lose to an alias at a
//! higher one. Each layer parser therefore stamps its own level onto its entries
//! ([`stamp_layer`]).

use super::partial::ConfigValue;
use super::{PartialConfig, SetPathError};

/// Which config layer a tuning setting came from, lowest precedence first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum TuneLevel {
    /// The config file (and any layer that was not stamped, such as a hand-built test layer).
    #[default]
    Config,
    /// The `INFR_*` environment.
    Env,
    /// The command line (`--set`).
    Cli,
}

impl std::fmt::Display for TuneLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Config => "config",
            Self::Env => "env",
            Self::Cli => "cli",
        })
    }
}

/// One `family.param[@qualifier] = value` setting, in the order its layer gave it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuneEntry {
    /// `family.param` or `family.param@qualifier`, as typed (shape-checked, not resolved).
    pub key: String,
    /// The value as typed; the resolver parses it into the parameter's type.
    pub value: String,
    /// The layer it came from.
    pub level: TuneLevel,
}

/// One alias knob (see [`TUNE_ALIASES`]) that a layer set, and which layer that was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AliasSet {
    /// The alias's config path, e.g. `kernels.vulkan.flash_splits`.
    pub path: String,
    /// The layer that set it.
    pub level: TuneLevel,
}

/// The `kernels.vulkan.tune` leaf: an OPEN map, whose keys the schema does not fix.
///
/// Folding two layers keeps both layers' entries (a later layer adds to an earlier one), in
/// precedence order, so the resolver can apply them in order and let the last one win.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TuneMap {
    /// Explicit tuning settings, lowest-precedence layer first, each layer in typed order.
    pub entries: Vec<TuneEntry>,
    /// The alias knobs each layer set (see the module docs for why this is recorded).
    pub aliases: Vec<AliasSet>,
}

impl TuneMap {
    /// Nothing set, by either spelling.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.aliases.is_empty()
    }

    /// The highest layer that set the alias at `path`, if any layer recorded setting it.
    pub fn alias_level(&self, path: &str) -> Option<TuneLevel> {
        self.aliases
            .iter()
            .filter(|a| a.path == path)
            .map(|a| a.level)
            .max()
    }
}

/// The config paths whose knobs are now aliases of a Vulkan tuning parameter. The backend's
/// alias table names the parameter each one writes; its test holds the two lists equal.
pub const TUNE_ALIASES: &[&str] = &[
    "device.subgroup_pref",
    "kernels.vulkan.gemv.no_rm",
    "kernels.vulkan.gemv.rm",
    "kernels.vulkan.gemv.rm_maxout",
    "kernels.vulkan.gemv.rm_minout",
    "kernels.vulkan.gemv.no_sg",
    "kernels.vulkan.gemv.no_id_sg",
    "kernels.vulkan.gemv.sg_minout",
    "kernels.vulkan.gemv.sg_maxout",
    "kernels.vulkan.gemv.sg_nr",
    "kernels.vulkan.gemv.variant",
    "kernels.vulkan.i8cm_kq",
    "kernels.vulkan.mmv_mw_warps",
    "kernels.vulkan.moe_small_m",
    "kernels.vulkan.canvas_chunk_n",
    "kernels.vulkan.flash_splits",
    "kernels.vulkan.flash_bm32",
    "kernels.vulkan.flash_min_rows",
    "kernels.vulkan.pv_splits",
];

/// Check a tuning key's shape: `family.param` or `family.param@qualifier`, each part made of
/// lowercase ASCII letters, digits and `_`, and not empty.
pub fn check_key(key: &str) -> Result<(), String> {
    let ident = |s: &str| {
        !s.is_empty()
            && s.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    };
    let (name, qual) = match key.split_once('@') {
        Some((n, q)) => (n, Some(q)),
        None => (key, None),
    };
    let ok = match name.split_once('.') {
        Some((family, param)) => ident(family) && ident(param) && qual.is_none_or(ident),
        None => false,
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "`{key}` is not a tuning key: expected `family.param` or `family.param@qualifier` \
             (lowercase letters, digits and `_`)"
        ))
    }
}

/// Parse the list grammar `INFR_TUNE` and `--set kernels.vulkan.tune=…` share:
/// `family.param[@qualifier]=value`, comma-separated. Spaces around items are ignored and an empty
/// item is skipped, so `""` is the empty list.
pub fn parse_list(raw: &str) -> Result<Vec<TuneEntry>, String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|item| {
            let (key, value) = item
                .split_once('=')
                .ok_or_else(|| format!("`{item}`: expected `family.param=value`"))?;
            let (key, value) = (key.trim(), value.trim());
            check_key(key)?;
            if value.is_empty() {
                return Err(format!("`{key}`: empty value"));
            }
            Ok(TuneEntry {
                key: key.to_string(),
                value: value.to_string(),
                level: TuneLevel::default(),
            })
        })
        .collect()
}

impl ConfigValue for TuneMap {
    fn parse_set(raw: &str) -> Result<Self, String> {
        Ok(TuneMap {
            entries: parse_list(raw)?,
            aliases: Vec::new(),
        })
    }

    fn to_display(&self) -> String {
        self.entries
            .iter()
            .map(|e| format!("{}={}", e.key, e.value))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn set_below(slot: &mut Option<Self>, rest: &str, raw: &str) -> Result<(), SetPathError> {
        check_key(rest).map_err(SetPathError::Value)?;
        let value = raw.trim();
        if value.is_empty() {
            return Err(SetPathError::Value(format!("`{rest}`: empty value")));
        }
        slot.get_or_insert_with(TuneMap::default)
            .entries
            .push(TuneEntry {
                key: rest.to_string(),
                value: value.to_string(),
                level: TuneLevel::default(),
            });
        Ok(())
    }

    fn fold(slot: &mut Option<Self>, other: Self) {
        let map = slot.get_or_insert_with(TuneMap::default);
        map.entries.extend(other.entries);
        map.aliases.extend(other.aliases);
    }
}

/// Stamp a freshly parsed layer with its level: every tuning entry it holds, plus a record of each
/// [`TUNE_ALIASES`] knob it set. Each layer parser calls this once, on its own layer, before the
/// layers are folded.
pub fn stamp_layer(layer: &mut PartialConfig, level: TuneLevel) {
    let had_map = layer.kernels.vulkan.tune.is_some();
    let mut map = layer.kernels.vulkan.tune.take().unwrap_or_default();
    for e in &mut map.entries {
        e.level = level;
    }
    for &path in TUNE_ALIASES {
        if layer.is_path_set(path) {
            map.aliases.push(AliasSet {
                path: path.to_string(),
                level,
            });
        }
    }
    if had_map || !map.is_empty() {
        layer.kernels.vulkan.tune = Some(map);
    }
}
