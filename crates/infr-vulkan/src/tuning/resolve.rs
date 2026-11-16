//! Resolution: most specific wins, per parameter (`docs/plan/10-tunable-kernels.md` §3.2).
//!
//! Levels, lowest first: the generic default (derived from the device's facts), class rows,
//! device rows, the user's tuning file, config, environment, command line. A setting replaces the
//! value below it only for the parameter (and qualifier) it names. Within one level:
//!
//! - rows apply in order of how many match fields they name, so the more specific row wins; an
//!   unqualified setting (`mmv_mw.warps`) applies before a qualified one (`mmv_mw.warps@q4_k`);
//! - at a config level, an alias knob (`flash_splits`) applies before a tuning key
//!   (`kernels.vulkan.tune`), so the new spelling wins over the old one at the same level, and
//!   a higher level wins over both.
//!
//! A built-in row value that fails validation is dropped with a warning and the parameter falls
//! through to the level below; a value the USER wrote (tuning file, config, environment, command
//! line) that fails is a hard error naming where it came from.

use std::path::Path;

use infr_core::config::{Config, TuneLevel};

use super::families::{self, Kind, Param, Qual, Value, ALIASES, PARAMS};
use super::report::{ParamReport, RowReport, TuningReport};
use super::table::{self, Row, Verified};
use super::{DeviceFacts, Tuning};

/// The resolved set and its report.
#[derive(Debug)]
pub(crate) struct Resolved {
    pub tuning: Tuning,
    pub report: TuningReport,
}

/// Where a setting sits in the order. Later stages win.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    ClassRow,
    DeviceRow,
    File,
    User(TuneLevel),
}

/// One setting of one parameter, from any level.
struct Setting {
    stage: Stage,
    /// The order within the stage (see the module docs); later wins.
    rank: (usize, bool, usize),
    param: &'static Param,
    qual: Qual,
    /// `None` resets the parameter to what the levels below config gave it (an alias set to a
    /// value with no effect, at a level above a tuning key).
    value: Option<Value>,
    label: String,
}

/// Resolve against the built-in table and the tuning file `cfg` names.
pub(crate) fn resolve(facts: &DeviceFacts, cfg: &Config) -> Result<Resolved, String> {
    let file = match &cfg.kernels.vulkan.tuning.file {
        Some(path) => Some((path.as_path(), read_file(path)?)),
        None => None,
    };
    resolve_with(facts, cfg, table::builtin(), file)
}

fn read_file(path: &Path) -> Result<Vec<Row>, String> {
    let label = format!("tuning file {}", path.display());
    let text = std::fs::read_to_string(path).map_err(|e| format!("{label}: {e}"))?;
    table::parse(&text, &label)
}

/// [`resolve`] over explicit rows, for the tests: `builtin` stands in for the compiled-in table.
pub(crate) fn resolve_with(
    facts: &DeviceFacts,
    cfg: &Config,
    builtin: &[Row],
    file: Option<(&Path, Vec<Row>)>,
) -> Result<Resolved, String> {
    let tcfg = &cfg.kernels.vulkan.tuning;
    let disabled = |id: &str| tcfg.disable.iter().any(|d| d == id);
    let mut settings: Vec<Setting> = Vec::new();
    let mut rows_report: Vec<RowReport> = Vec::new();

    // ── rows: built-in (class, device), then the user's file ─────────────────
    let builtin_rows = if tcfg.rows { builtin } else { &[] };
    for (i, row) in builtin_rows.iter().enumerate() {
        if disabled(&row.id) || !row.matcher.matches(facts) {
            continue;
        }
        let stage = if row.matcher.is_device() {
            Stage::DeviceRow
        } else {
            Stage::ClassRow
        };
        let set = row_settings(row, stage, i, "row", &mut settings, |msg| {
            tracing::warn!("[infr] tuning: {msg} — value dropped, the level below applies");
            Ok(())
        })?;
        note_row(row, &set, &mut rows_report, stage);
    }
    if let Some((path, rows)) = &file {
        for c in table::conflicts(rows) {
            tracing::warn!(
                "[infr] tuning file {}: {c} — the later row wins",
                path.display()
            );
        }
        for (i, row) in rows.iter().enumerate() {
            if disabled(&row.id) || !row.matcher.matches(facts) {
                continue;
            }
            let label = format!("tuning file {} row", path.display());
            let set = row_settings(row, Stage::File, i, &label, &mut settings, Err)?;
            note_row(row, &set, &mut rows_report, Stage::File);
        }
    }
    for id in &tcfg.disable {
        let known = builtin.iter().any(|r| &r.id == id)
            || file
                .as_ref()
                .is_some_and(|(_, rows)| rows.iter().any(|r| &r.id == id));
        if !known {
            tracing::warn!("[infr] tuning: kernels.vulkan.tuning.disable names no row `{id}`");
        }
    }

    // ── config, environment, command line ────────────────────────────────────
    let tune = &cfg.kernels.vulkan.tune;
    for (i, alias) in ALIASES.iter().enumerate() {
        let recorded = tune.alias_level(alias.path);
        let Some(writes) = (alias.read)(cfg, recorded.is_some()) else {
            // Set by a layer, but to a value with no effect: that layer's word is "the default",
            // which must still beat a tuning key from a LOWER layer.
            if let Some(level) = recorded {
                let label = format!("{} (unset)", alias_label(level, alias.path));
                for key in alias.params {
                    let (param, qual) =
                        families::lookup(key).expect("alias names a real parameter");
                    settings.push(Setting {
                        stage: Stage::User(level),
                        rank: (0, qual.is_some(), i),
                        param,
                        qual,
                        value: None,
                        label: label.clone(),
                    });
                }
            }
            continue;
        };
        let level = recorded.unwrap_or_default();
        let label = alias_label(level, alias.path);
        for (key, value) in writes {
            let (param, qual) = families::lookup(key).expect("alias names a real parameter");
            if param.kind == Kind::Fixed || !param.ty.admits(&value) {
                return Err(format!(
                    "{label}: `{key}` {}",
                    param
                        .ty
                        .parse(&value.to_string())
                        .err()
                        .unwrap_or_else(|| "is fixed".to_string())
                ));
            }
            settings.push(Setting {
                stage: Stage::User(level),
                rank: (0, qual.is_some(), i),
                param,
                qual,
                value: Some(value),
                label: label.clone(),
            });
        }
    }
    for (i, e) in tune.entries.iter().enumerate() {
        let label = key_label(e.level, &e.key);
        let (param, qual) = families::lookup(&e.key).map_err(|m| format!("{label}: {m}"))?;
        let value = validate(param, &e.value).map_err(|m| format!("{label}: `{}` {m}", e.key))?;
        settings.push(Setting {
            stage: Stage::User(e.level),
            rank: (1, qual.is_some(), i),
            param,
            qual,
            value: Some(value),
            label,
        });
    }

    // ── apply, per parameter and qualifier ───────────────────────────────────
    settings.sort_by_key(|s| (s.stage, s.rank));
    let mut tuning = Tuning::default();
    let mut params = Vec::new();
    for p in PARAMS {
        for slot in families::slots(p) {
            let mut value = (p.default)(facts, slot);
            let mut source = if p.kind == Kind::Fixed {
                "fixed"
            } else {
                "default"
            }
            .to_string();
            // What the levels below config give: an alias's "unset" goes back to it.
            let mut base: Option<(Value, String)> = None;
            for s in settings
                .iter()
                .filter(|s| std::ptr::eq(s.param, p) && (s.qual.is_none() || s.qual == slot))
            {
                if matches!(s.stage, Stage::User(_)) && base.is_none() {
                    base = Some((value.clone(), source.clone()));
                }
                match &s.value {
                    Some(v) => {
                        value = v.clone();
                        source = s.label.clone();
                    }
                    None => {
                        let (v, src) = base.clone().expect("set on entering the user levels");
                        value = v;
                        source = format!("{src}; {}", s.label);
                    }
                }
            }
            if let Some((used, why)) = families::in_use(p, facts, &value) {
                tracing::warn!("[infr] tuning: `{}` {why} — using {used}", p.key());
                source = format!("{source}; {why}, so {used} is used");
                value = used;
            }
            if p.kind != Kind::Fixed {
                (p.set)(&mut tuning, slot, &value);
            }
            let key = match families::qual_name(p, slot) {
                Some(q) => format!("{}@{q}", p.key()),
                None => p.key(),
            };
            params.push(ParamReport {
                key,
                value: value.to_string(),
                source,
                kind: p.kind.name(),
                range: p.ty.describe(),
                axis: p.axis.name(),
                rule: p.rule,
                doc: p.doc,
            });
        }
    }
    for r in &rows_report {
        if r.verified == "contributor" && !r.params.is_empty() {
            tracing::info!(
                "[infr] tuning: contributed row `{}` is in use (unverified here); \
                 `--set kernels.vulkan.tuning.disable={}` turns it off",
                r.id,
                r.id
            );
        }
    }
    Ok(Resolved {
        tuning,
        report: TuningReport::new(facts, rows_report, params, !tcfg.rows),
    })
}

/// Validate one written value for `p`: the parameter must not be fixed, and the value must parse
/// into its type and range.
pub(crate) fn validate(p: &Param, raw: &str) -> Result<Value, String> {
    if p.kind == Kind::Fixed {
        return Err(format!(
            "is fixed: it must equal the literal its shader is compiled with ({})",
            p.rule
        ));
    }
    p.ty.parse(raw)
}

/// Turn one matched row's params into settings. `on_bad` decides what an unknown key or a value
/// that fails validation does: a built-in row's is warned about and skipped (`Ok`), a user
/// file's is fatal (`Err`). Returns the keys the row set.
fn row_settings(
    row: &Row,
    stage: Stage,
    index: usize,
    what: &str,
    out: &mut Vec<Setting>,
    on_bad: impl Fn(String) -> Result<(), String>,
) -> Result<Vec<String>, String> {
    let mut set = Vec::new();
    let label = match stage {
        Stage::DeviceRow => format!("device {what} {}", row.id),
        _ => format!("{what} {}", row.id),
    };
    for (key, raw) in &row.params {
        let resolved = families::lookup(key)
            .and_then(|(p, q)| validate(p, raw).map(|v| (p, q, v)))
            .map_err(|m| format!("{label}: `{key}` {m}"));
        match resolved {
            Ok((param, qual, value)) => {
                out.push(Setting {
                    stage,
                    rank: (row.matcher.fields(), qual.is_some(), index),
                    param,
                    qual,
                    value: Some(value),
                    label: label.clone(),
                });
                set.push(key.clone());
            }
            Err(msg) => on_bad(msg)?,
        }
    }
    Ok(set)
}

fn note_row(row: &Row, set: &[String], out: &mut Vec<RowReport>, stage: Stage) {
    out.push(RowReport {
        id: row.id.clone(),
        level: match stage {
            Stage::ClassRow => "class",
            Stage::DeviceRow => "device",
            _ => "file",
        },
        verified: match row.verified {
            Verified::Maintainer => "maintainer",
            Verified::Contributor => "contributor",
        },
        params: set.to_vec(),
    });
}

/// The source label of an alias knob set at `level`.
fn alias_label(level: TuneLevel, path: &str) -> String {
    let env = families::alias_env_names(path);
    match level {
        TuneLevel::Config => format!("config alias {path} ({env})"),
        TuneLevel::Env => format!("env alias {env}"),
        TuneLevel::Cli => format!("cli alias --set {path}"),
    }
}

/// The source label of a tuning key set at `level`.
fn key_label(level: TuneLevel, key: &str) -> String {
    match level {
        TuneLevel::Config => format!("config kernels.vulkan.tune.{key}"),
        TuneLevel::Env => format!("env INFR_TUNE ({key})"),
        TuneLevel::Cli => format!("cli --set kernels.vulkan.tune.{key}"),
    }
}
