//! The tuning table's tests (`docs/plan/10-tunable-kernels.md` §3.6): the schema, the rows'
//! hygiene, parameter ranges, the resolution order and failure handling, the aliases, the shader
//! inventory, and the snapshot of every fixture device's resolved set.

use std::path::{Path, PathBuf};

use infr_core::config::{Config, ConfigLayer, ConfigOverrides, PartialConfig};

use super::classes::{ShaderClass, SHADER_CLASSES};
use super::families::{self, Kind, Value, ALIASES, PARAMS};
use super::resolve::{resolve_with, Resolved};
use super::table::{self, Row};
use super::{DeviceFacts, Tuning};

// ── fixtures ─────────────────────────────────────────────────────────────────

fn tuning_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tuning")
}

/// Every fixture device, in file order.
pub(crate) fn fixtures() -> Vec<(String, DeviceFacts)> {
    let text = std::fs::read_to_string(tuning_dir().join("fixtures.toml")).expect("fixtures.toml");
    let doc: toml::Table = toml::from_str(&text).expect("fixtures.toml parses");
    doc["device"]
        .as_array()
        .expect("[[device]]")
        .iter()
        .map(|d| {
            let t = d.as_table().expect("a device is a table");
            let s = |k: &str| t[k].as_str().unwrap_or_else(|| panic!("{k}")).to_string();
            let n = |k: &str| u32::try_from(t[k].as_integer().expect(k)).expect(k);
            let b = |k: &str| t[k].as_bool().expect(k);
            assert!(
                t.contains_key("measured") && t.contains_key("source"),
                "{}",
                s("id")
            );
            let arch = super::arch_from_name(&s("arch")).expect("a known arch name");
            (
                s("id"),
                DeviceFacts {
                    name: s("name"),
                    vendor_id: n("vendor_id"),
                    device_id: n("device_id"),
                    driver: t
                        .get("driver")
                        .map(|v| v.as_str().expect("driver").to_string()),
                    driver_version: n("driver_version"),
                    arch: super::arch_name(arch),
                    integrated: b("integrated"),
                    compute_units: n("compute_units"),
                    subgroup_min: n("subgroup_min"),
                    subgroup_max: n("subgroup_max"),
                    wave64: b("wave64"),
                    max_shared_memory: n("max_shared_memory"),
                    matrix_units: b("matrix_units"),
                    dot_accelerated: b("dot_accelerated"),
                },
            )
        })
        .collect()
}

pub(crate) fn fixture(id: &str) -> DeviceFacts {
    fixtures()
        .into_iter()
        .find(|(i, _)| i == id)
        .unwrap_or_else(|| panic!("no fixture `{id}`"))
        .1
}

fn resolve_fixture(id: &str, cfg: &Config) -> Resolved {
    resolve_with(&fixture(id), cfg, table::builtin(), None).expect("resolves")
}

/// The resolved set of fixture `id` under the default config — what a backend on that device
/// runs with. For other modules' tests.
pub(crate) fn fixture_tuning(id: &str) -> Tuning {
    resolve_fixture(id, &Config::default()).tuning
}

/// [`fixture_tuning`] under an environment layer of `pairs` (`INFR_*` name, value).
pub(crate) fn fixture_tuning_env(id: &str, pairs: &[(&str, &str)]) -> Tuning {
    resolve_fixture(id, &env_cfg(pairs)).tuning
}

fn value_of(r: &Resolved, key: &str) -> (String, String) {
    let p = r
        .report
        .params
        .iter()
        .find(|p| p.key == key)
        .unwrap_or_else(|| panic!("no parameter `{key}` in the report"));
    (p.value.clone(), p.source.clone())
}

fn env_cfg(pairs: &[(&str, &str)]) -> Config {
    let get = |k: &str| {
        pairs
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.to_string())
    };
    let layer = infr_core::config::env::parse(&get).expect("env layer");
    Config::load_from_layers(&[layer])
}

fn file_layer(text: &str) -> PartialConfig {
    infr_core::config::file::parse_str(text, Path::new("infr.toml"))
        .expect("config file")
        .0
}

fn cli_layer(sets: &[&str]) -> PartialConfig {
    ConfigLayer::cli(&ConfigOverrides {
        sets: sets.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    })
    .expect("cli layer")
}

fn env_layer(pairs: &[(&str, &str)]) -> PartialConfig {
    let get = |k: &str| {
        pairs
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.to_string())
    };
    infr_core::config::env::parse(&get).expect("env layer")
}

// ── the table ────────────────────────────────────────────────────────────────

/// The built-in table parses; every row has a unique id, names a vendor plus at least one more
/// match field, cites a complete source, and sets only real parameters to valid values.
#[test]
fn builtin_rows_are_well_formed() {
    let rows = table::parse(table::BUILTIN, "table.toml").expect("table.toml parses");
    assert!(!rows.is_empty(), "the table has rows to check");
    let mut ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    ids.sort_unstable();
    let n = ids.len();
    ids.dedup();
    assert_eq!(n, ids.len(), "duplicate row ids");
    for r in &rows {
        assert!(r.matcher.vendor.is_some(), "row `{}` names no vendor", r.id);
        assert!(
            r.matcher.fields() >= 2,
            "row `{}` names only a vendor: it would retune every card of it",
            r.id
        );
        for f in table::SOURCE_FIELDS {
            assert!(
                r.source.iter().any(|(k, v)| k == f && !v.trim().is_empty()),
                "row `{}` cites no `source.{f}`",
                r.id
            );
        }
        assert!(!r.params.is_empty(), "row `{}` sets nothing", r.id);
        for (k, v) in &r.params {
            let (p, _) = families::lookup(k).unwrap_or_else(|e| panic!("row `{}`: {e}", r.id));
            super::resolve::validate(p, v).unwrap_or_else(|e| panic!("row `{}`: `{k}` {e}", r.id));
        }
    }
}

/// No built-in row may match the device the table knows nothing about.
#[test]
fn no_row_matches_the_unknown_fixture() {
    let unknown = fixture("unknown");
    for r in table::builtin() {
        assert!(
            !r.matcher.matches(&unknown),
            "row `{}` matches `unknown`",
            r.id
        );
    }
    let resolved = resolve_fixture("unknown", &Config::default());
    assert!(resolved.report.rows.is_empty());
    assert!(resolved
        .report
        .params
        .iter()
        .all(|p| p.source == "default" || p.source == "fixed"));
}

/// Two built-in rows at the same level and specificity that both match a fixture must not set
/// the same parameter to different values: which one won would be file order, not intent.
#[test]
fn no_ambiguous_ties_on_any_fixture() {
    check_no_ties(table::builtin());
}

fn check_no_ties(rows: &[Row]) {
    for (id, facts) in fixtures() {
        let matched: Vec<&Row> = rows.iter().filter(|r| r.matcher.matches(&facts)).collect();
        for (i, a) in matched.iter().enumerate() {
            for b in &matched[i + 1..] {
                if a.matcher.is_device() != b.matcher.is_device()
                    || a.matcher.fields() != b.matcher.fields()
                {
                    continue;
                }
                for (k, va) in &a.params {
                    if let Some((_, vb)) = b.params.iter().find(|(kb, _)| kb == k) {
                        assert_eq!(
                            va, vb,
                            "rows `{}` and `{}` tie on fixture `{id}` and disagree on `{k}`",
                            a.id, b.id
                        );
                    }
                }
            }
        }
    }
}

/// The tie check is not vacuous: two equally specific rows that disagree fail it.
#[test]
#[should_panic(expected = "tie on fixture")]
fn a_disagreeing_tie_fails_the_tie_check() {
    let rows = table::parse(
        r#"
[[row]]
id = "a"
match = { vendor = "amd", driver = "AMD_PROPRIETARY" }
params = { "gemv_sg.rows" = 4 }
[[row]]
id = "b"
match = { vendor = "amd", integrated = false }
params = { "gemv_sg.rows" = 8 }
"#,
        "test",
    )
    .unwrap();
    check_no_ties(&rows);
}

/// A row that would match the unknown device (vendor `other` alone is not enough, but a row on
/// a field `unknown` shares — here its subgroup range — is) is caught.
#[test]
fn a_row_matching_unknown_is_caught() {
    let rows = table::parse(
        r#"
[[row]]
id = "too-broad"
match = { vendor = "other", subgroup_min = 32 }
params = { "gemv_sg.rows" = 4 }
"#,
        "test",
    )
    .unwrap();
    assert!(rows[0].matcher.matches(&fixture("unknown")));
}

// ── parameters ───────────────────────────────────────────────────────────────

/// Every parameter resolves to a value of its type in its declared range on every fixture, and
/// its setter and getter agree (a parameter wired to the wrong field fails here).
#[test]
fn every_parameter_is_in_range_and_round_trips() {
    let mut keys: Vec<String> = PARAMS.iter().map(|p| p.key()).collect();
    let n = keys.len();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(n, keys.len(), "duplicate parameter keys");
    for (id, facts) in fixtures() {
        let r = resolve_with(&facts, &Config::default(), table::builtin(), None).unwrap();
        let mut i = 0;
        for p in PARAMS {
            for slot in families::slots(p) {
                let got = (p.get)(&r.tuning, slot);
                assert!(
                    p.ty.admits(&got),
                    "{id}: `{}` = {got} out of range",
                    p.key()
                );
                assert_eq!(
                    r.report.params[i].value,
                    got.to_string(),
                    "{id}: {}",
                    p.key()
                );
                // Write a different in-range value and read it back.
                if p.kind != Kind::Fixed {
                    let other = match (&got, p.ty) {
                        (Value::Bool(b), _) => Value::Bool(!b),
                        (Value::Uint(v), families::Ty::Set(s)) => {
                            Value::Uint(*s.iter().find(|&&x| x != *v).unwrap_or(v))
                        }
                        (Value::Uint(v), families::Ty::Uint { min, max }) => {
                            Value::Uint(if *v < max { v + 1 } else { min })
                        }
                        (Value::Text(t), families::Ty::Choice(words)) => Value::Text(
                            words
                                .iter()
                                .find(|w| **w != t.as_str())
                                .expect("a choice has more than one word")
                                .to_string(),
                        ),
                        (Value::Text(_), _) => Value::Text("other".to_string()),
                        _ => unreachable!(),
                    };
                    let mut t = r.tuning.clone();
                    (p.set)(&mut t, slot, &other);
                    assert_eq!(
                        (p.get)(&t, slot),
                        other,
                        "{id}: `{}` setter/getter",
                        p.key()
                    );
                }
                i += 1;
            }
        }
        assert_eq!(i, r.report.params.len());
    }
}

// ── the defaults are today's rules, verbatim ─────────────────────────────────

/// The generic defaults and rows reproduce, on every fixture, exactly the value each migrated
/// rule computed before the table existed (copied here from the pre-table code).
#[test]
fn defaults_are_the_pre_table_rules() {
    for (id, f) in fixtures() {
        let t = fixture_tuning(&id);
        // lib.rs's sg_pref rule.
        let sg = if f.subgroup_min <= 16 {
            16u32.max(f.subgroup_min)
        } else {
            32
        };
        assert_eq!(t.sg_pref, sg, "{id}");
        // adapter.rs's split_k_cu_for_driver + amd_proprietary_driver, and i8cm_kq_wanted's gate.
        let amd_prop = f.driver.as_deref() == Some("AMD_PROPRIETARY");
        assert_eq!(t.split_k_scale_by_cu, amd_prop, "{id}");
        assert_eq!(t.i8cm_kq_enabled, amd_prop, "{id}");
        assert_eq!(
            (t.split_k_fixed_grid, t.split_k_fill_per_cu),
            (128, 4),
            "{id}"
        );
        assert_eq!(t.i8cm_kq_l_fill_per_cu, 2, "{id}");
        // recorder.rs's mmv_o4.
        assert_eq!(
            (t.mmv_o4_max_in_f, t.mmv_o4_all_widths),
            (2048, f.integrated),
            "{id}"
        );
        // adapter.rs's per-dtype mmv WARPS default.
        for &(_, dt) in super::MMV_MW_DTYPES {
            let want = if dt == infr_core::DType::Q4K { 1 } else { 8 };
            assert_eq!(t.mmv_mw_warps(dt), Some(want), "{id} {dt:?}");
        }
        // The config defaults the old read sites used.
        let d = Config::default();
        let g = &d.kernels.vulkan.gemv;
        assert_eq!(t.gemv_variant, g.variant, "{id}");
        assert_eq!(t.gemv_rm_enabled, !g.no_rm);
        assert_eq!(
            (t.gemv_rm_rows, t.gemv_rm_min_out, t.gemv_rm_max_out),
            (g.rm, g.rm_minout, g.rm_maxout)
        );
        assert_eq!(
            (t.gemv_sg_enabled, t.gemv_id_sg_enabled),
            (!g.no_sg, !g.no_id_sg)
        );
        assert_eq!(
            (t.gemv_sg_rows, t.gemv_sg_min_out, t.gemv_sg_max_out),
            (g.sg_nr, g.sg_minout, g.sg_maxout)
        );
        let v = &d.kernels.vulkan;
        assert_eq!(t.flash_min_rows, v.flash_min_rows);
        assert_eq!(t.flash_force_bm32, v.flash_bm32);
        assert_eq!(t.moe_small_m, v.moe_small_m);
        assert_eq!(t.canvas_chunks, v.canvas_chunk_n);
        assert_eq!((t.flash.splits, t.pv.splits), (0, 0));
        assert!(t.rmsnorm_wide && t.rmsnorm_wide_min_dim == 2048);
    }
}

/// `AttnSplit::count` at the defaults is the pre-table flash and PV split formulas, at every
/// shape in a sweep that crosses each of their thresholds.
#[test]
fn split_policy_matches_the_pre_table_formulas() {
    let t = fixture_tuning("rx7900xtx-amdwin");
    let old_flash = |base_wg: u32, kv_len: usize| -> u32 {
        if base_wg >= 1024 || kv_len < 4096 {
            1u32
        } else {
            (2048 / base_wg.max(1))
                .clamp(2, 8)
                .min((kv_len / 2048).max(1) as u32)
        }
    };
    let old_pv = |pv_base_wg: u32, kv_pad: u32| -> u32 {
        if pv_base_wg >= 1024 || kv_pad < 4096 {
            1u32
        } else {
            (2048 / pv_base_wg.max(1))
                .clamp(2, 8)
                .min(kv_pad / 2048)
                .max(1)
        }
    };
    for base in [0u32, 1, 2, 7, 64, 255, 256, 257, 511, 1023, 1024, 4096] {
        for kv in [
            0u32,
            1,
            2047,
            2048,
            4095,
            4096,
            4097,
            6144,
            8192,
            16384,
            65536,
            1 << 20,
        ] {
            assert_eq!(
                t.flash.count(base, kv),
                old_flash(base, kv as usize),
                "{base} {kv}"
            );
            assert_eq!(t.pv.count(base, kv), old_pv(base, kv), "{base} {kv}");
        }
    }
    // A fixed count wins over the policy.
    let fixed = super::AttnSplit {
        splits: 3,
        ..t.flash
    };
    assert_eq!(fixed.count(1, 1), 3);
}

// ── resolution order and failure handling ────────────────────────────────────

fn rows(text: &str) -> Vec<Row> {
    table::parse(text, "test").unwrap()
}

/// Per parameter, the most specific level wins: class row < device row < tuning file < config <
/// env < cli; a row sets only what it names.
#[test]
fn levels_apply_in_order() {
    let facts = fixture("rx7900xtx-amdwin");
    let builtin = rows(
        r#"
[[row]]
id = "class"
match = { vendor = "amd", arch = "AmdRdna3" }
params = { "gemv_sg.rows" = 4, "gemv_sg.min_out" = 1024, "moe.small_m" = 16 }
[[row]]
id = "class-more-specific"
match = { vendor = "amd", arch = "AmdRdna3", integrated = false }
params = { "gemv_sg.min_out" = 512 }
[[row]]
id = "dev"
match = { vendor = "amd", device_id = 0x744c }
params = { "gemv_sg.rows" = 8 }
"#,
    );
    let r = resolve_with(&facts, &Config::default(), &builtin, None).unwrap();
    assert_eq!(
        value_of(&r, "gemv_sg.rows"),
        ("8".into(), "device row dev".into())
    );
    assert_eq!(
        value_of(&r, "gemv_sg.min_out"),
        ("512".into(), "row class-more-specific".into())
    );
    assert_eq!(
        value_of(&r, "moe.small_m"),
        ("16".into(), "row class".into())
    );
    assert_eq!(
        value_of(&r, "gemv_sg.max_out"),
        ("8192".into(), "default".into())
    );

    // The user's tuning file beats every row.
    let file = rows(
        r#"
[[row]]
id = "mine"
params = { "gemv_sg.rows" = 2, "attn_canvas.chunks" = 5 }
"#,
    );
    let path = Path::new("my-tune.toml");
    let r = resolve_with(
        &facts,
        &Config::default(),
        &builtin,
        Some((path, file.clone())),
    )
    .unwrap();
    assert_eq!(
        value_of(&r, "gemv_sg.rows"),
        ("2".into(), "tuning file my-tune.toml row mine".into())
    );

    // Config < env < cli, through the real layer parsers.
    let cfg = Config::load_from_layers(&[
        file_layer("[kernels.vulkan.tune]\n\"gemv_sg.rows\" = 4\n\"moe.small_m\" = 3\n"),
        env_layer(&[("INFR_TUNE", "moe.small_m=5,attn_canvas.chunks=7")]),
        cli_layer(&["kernels.vulkan.tune.attn_canvas.chunks=9"]),
    ]);
    let r = resolve_with(&facts, &cfg, &builtin, Some((path, file))).unwrap();
    assert_eq!(
        value_of(&r, "gemv_sg.rows"),
        ("4".into(), "config kernels.vulkan.tune.gemv_sg.rows".into())
    );
    assert_eq!(
        value_of(&r, "moe.small_m"),
        ("5".into(), "env INFR_TUNE (moe.small_m)".into())
    );
    assert_eq!(
        value_of(&r, "attn_canvas.chunks"),
        (
            "9".into(),
            "cli --set kernels.vulkan.tune.attn_canvas.chunks".into()
        )
    );

    // `rows = false` resolves from defaults alone; `disable` drops one row by id.
    let mut off = Config::default();
    off.kernels.vulkan.tuning.rows = false;
    let r = resolve_with(&facts, &off, &builtin, None).unwrap();
    assert_eq!(value_of(&r, "gemv_sg.rows").1, "default");
    let mut drop_dev = Config::default();
    drop_dev.kernels.vulkan.tuning.disable = vec!["dev".to_string()];
    let r = resolve_with(&facts, &drop_dev, &builtin, None).unwrap();
    assert_eq!(
        value_of(&r, "gemv_sg.rows"),
        ("4".into(), "row class".into())
    );
}

/// A qualified setting applies to its qualifier only and beats an unqualified one at its level.
#[test]
fn qualifiers_are_per_slot() {
    let cfg = env_cfg(&[("INFR_TUNE", "mmv_mw.warps@q6_k=4,mmv_mw.warps=2")]);
    let r = resolve_with(&fixture("unknown"), &cfg, &[], None).unwrap();
    assert_eq!(value_of(&r, "mmv_mw.warps@q6_k").0, "4");
    assert_eq!(value_of(&r, "mmv_mw.warps@q4_k").0, "2");
    assert_eq!(r.tuning.mmv_mw_warps(infr_core::DType::Q6K), Some(4));
    assert_eq!(r.tuning.mmv_mw_warps(infr_core::DType::Q5K), Some(2));
}

/// A built-in row value that fails validation is dropped and the parameter falls through; a
/// user value that fails is a hard error naming where it came from.
#[test]
fn bad_row_values_fall_through_and_bad_user_values_fail() {
    let facts = fixture("rx7900xtx-amdwin");
    let bad = rows(
        r#"
[[row]]
id = "bad"
match = { vendor = "amd", arch = "AmdRdna3" }
params = { "moe.small_m" = 65, "gemv_sg.rows" = 4, "no_such.param" = 1, "quant_q8.blocks_per_wg" = 8 }
"#,
    );
    let r = resolve_with(&facts, &Config::default(), &bad, None).unwrap();
    assert_eq!(value_of(&r, "moe.small_m"), ("8".into(), "default".into()));
    assert_eq!(value_of(&r, "gemv_sg.rows"), ("4".into(), "row bad".into()));
    assert_eq!(r.report.rows[0].params, vec!["gemv_sg.rows".to_string()]);

    let err = |r: Result<Resolved, String>| r.expect_err("must refuse");
    let e = err(resolve_with(
        &facts,
        &Config::default(),
        &[],
        Some((Path::new("t.toml"), bad)),
    ));
    assert!(e.contains("tuning file t.toml row bad"), "{e}");
    let e = err(resolve_with(
        &facts,
        &env_cfg(&[("INFR_TUNE", "moe.small_m=65")]),
        &[],
        None,
    ));
    assert!(
        e.contains("INFR_TUNE") && e.contains("moe.small_m") && e.contains("0..=64"),
        "{e}"
    );
    let e = err(resolve_with(
        &facts,
        &env_cfg(&[("INFR_TUNE", "gemv_sg.rowz=4")]),
        &[],
        None,
    ));
    assert!(
        e.contains("gemv_sg.rowz") && e.contains("not a tuning parameter"),
        "{e}"
    );
    let e = err(resolve_with(
        &facts,
        &env_cfg(&[("INFR_TUNE", "mla.sg_width=32")]),
        &[],
        None,
    ));
    assert!(e.contains("is fixed"), "{e}");
    let e = err(resolve_with(
        &facts,
        &env_cfg(&[("INFR_TUNE", "gemv_sg.rows@q4_k=4")]),
        &[],
        None,
    ));
    assert!(e.contains("takes no qualifier"), "{e}");
    let e = err(resolve_with(
        &facts,
        &env_cfg(&[("INFR_SG", "16")]).tap_sg(17),
        &[],
        None,
    ));
    assert!(
        e.contains("must be 16 or 32") && e.contains("INFR_SG"),
        "{e}"
    );
}

trait TapSg {
    fn tap_sg(self, v: u32) -> Self;
}
impl TapSg for Config {
    /// Put a value the env layer would refuse onto the field directly, as a TOML file can.
    fn tap_sg(mut self, v: u32) -> Self {
        self.device.subgroup_pref = Some(v);
        self
    }
}

/// `(name, value)` pairs: an environment, or the parameter values it must resolve to.
type Pairs<'a> = &'a [(&'a str, &'a str)];

// ── aliases ──────────────────────────────────────────────────────────────────

/// The alias table names exactly the config paths the config layers record as aliases.
#[test]
fn alias_table_matches_the_config_list() {
    let mut ours: Vec<&str> = ALIASES.iter().map(|a| a.path).collect();
    let mut theirs: Vec<&str> = infr_core::config::tune::TUNE_ALIASES.to_vec();
    ours.sort_unstable();
    theirs.sort_unstable();
    assert_eq!(ours, theirs);
    for a in ALIASES {
        assert!(
            !families::alias_env_names(a.path).is_empty(),
            "alias `{}` has no INFR_* name",
            a.path
        );
    }
}

/// Every alias, set through the env layer exactly as a user sets it today, writes the parameter
/// the old read site read, with the value it computed (its clamp and its fallback included), and
/// labels the source with the alias's name.
#[test]
fn every_alias_writes_the_old_value() {
    let facts = fixture("rx7900xtx-amdwin");
    // (env pairs, [(parameter key, resolved value)]).
    let cases: &[(Pairs, Pairs)] = &[
        // The 7900 XTX cannot pin 16: the width in use is 32, as the backend always fell back to.
        (&[("INFR_SG", "16")], &[("device.sg_pref", "32")]),
        (&[("INFR_SG", "32")], &[("device.sg_pref", "32")]),
        (&[("INFR_NO_GEMV_RM", "")], &[("gemv_rm.enabled", "false")]),
        (&[("INFR_GEMV_RM", "4")], &[("gemv_rm.rows", "4")]),
        (&[("INFR_GEMV_RM", "3")], &[("gemv_rm.rows", "3")]),
        (
            &[("INFR_GEMV_RM_MAXOUT", "9000")],
            &[("gemv_rm.max_out", "9000")],
        ),
        (
            &[("INFR_GEMV_RM_MINOUT", "1000")],
            &[("gemv_rm.min_out", "1000")],
        ),
        (&[("INFR_NO_GEMV_SG", "")], &[("gemv_sg.enabled", "false")]),
        (
            &[("INFR_NO_GEMV_ID_SG", "")],
            &[("gemv_id_sg.enabled", "false")],
        ),
        (
            &[("INFR_GEMV_SG_MINOUT", "512")],
            &[("gemv_sg.min_out", "512")],
        ),
        (
            &[("INFR_GEMV_SG_MAXOUT", "4096")],
            &[("gemv_sg.max_out", "4096")],
        ),
        (&[("INFR_GEMV_SG_NR", "8")], &[("gemv_sg.rows", "8")]),
        (&[("INFR_GEMV_VARIANT", "sg")], &[("gemv.variant", "sg")]),
        (&[("INFR_NO_GEMV_REG", "")], &[("gemv.variant", "none")]),
        (
            &[("INFR_NO_GEMV_REG", ""), ("INFR_GEMV_VARIANT", "sg")],
            &[("gemv.variant", "none")],
        ),
        (
            &[("INFR_NO_I8CM_KQ", "")],
            &[("gemm_i8cm_kq.enabled", "false")],
        ),
        (
            &[("INFR_MMV_MW_WARPS", "4")],
            &[("mmv_mw.warps@q4_k", "4"), ("mmv_mw.warps@q6_k", "4")],
        ),
        (&[("INFR_GFX_QUEUE", "1")], &[("queue.family", "graphics")]),
        (&[("INFR_KQ_FMA_GEMV", "1")], &[("gemv.kq_fma", "true")]),
        (&[("INFR_KQ_FMA_GEMV", "0")], &[("gemv.kq_fma", "false")]),
        (
            &[("INFR_MMV_Q4K_META", "1")],
            &[("mmv_mrow.q4k_packed_meta", "true")],
        ),
        (
            &[("INFR_MMV_Q4K_META", "0")],
            &[("mmv_mrow.q4k_packed_meta", "false")],
        ),
        (&[("INFR_MOE_SMALL_M", "16")], &[("moe.small_m", "16")]),
        (&[("INFR_MOE_SMALL_M", "100000")], &[("moe.small_m", "64")]),
        (
            &[("INFR_CANVAS_CHUNK_N", "5")],
            &[("attn_canvas.chunks", "5")],
        ),
        (
            &[("INFR_CANVAS_CHUNK_N", "0")],
            &[("attn_canvas.chunks", "1")],
        ),
        (&[("INFR_FLASH_SPLITS", "4")], &[("attn_flash.splits", "4")]),
        (&[("INFR_FLASH_SPLITS", "0")], &[("attn_flash.splits", "0")]),
        (
            &[("INFR_FLASH_SPLITS", "5000000000")],
            &[("attn_flash.splits", "0")],
        ),
        (
            &[("INFR_FLASH_BM", "32")],
            &[("attn_flash.force_bm32", "true")],
        ),
        (
            &[("INFR_FLASH_BM", "64")],
            &[("attn_flash.force_bm32", "false")],
        ),
        (
            &[("INFR_FLASH_MIN_ROWS", "8")],
            &[("attn_flash.min_rows", "8")],
        ),
        (&[("INFR_PV_SPLITS", "2")], &[("attn_pv.splits", "2")]),
    ];
    let mut covered: Vec<&str> = Vec::new();
    for (env, want) in cases {
        let cfg = env_cfg(env);
        let r = resolve_with(&facts, &cfg, table::builtin(), None)
            .unwrap_or_else(|e| panic!("{env:?}: {e}"));
        for (key, v) in *want {
            let (got, src) = value_of(&r, key);
            assert_eq!(&got, v, "{env:?} → {key}");
            assert!(
                src.starts_with("env alias ") && env.iter().any(|(k, _)| src.contains(k)),
                "{env:?} → {key}: source {src:?} does not name the alias"
            );
        }
        for (k, _) in *env {
            covered.push(k);
        }
    }
    // Every alias's every env spelling is exercised above.
    for a in ALIASES {
        for env in families::alias_env_names(a.path).split('/') {
            assert!(covered.contains(&env), "alias env `{env}` has no case");
        }
    }
    // Unsettable spellings stay no-ops, as they were: `i8cm_kq = true` does not force the
    // tier on where the driver row is off, `INFR_MMV_MW_WARPS` past u32 keeps the defaults.
    let r = resolve_with(
        &fixture("rx7900xtx-radv"),
        &Config::load_from_layers(&[cli_layer(&["kernels.vulkan.i8cm_kq=true"])]),
        table::builtin(),
        None,
    )
    .unwrap();
    // (An explicitly set no-op spelling is recorded as an "unset" at its level.)
    let (v, src) = value_of(&r, "gemm_i8cm_kq.enabled");
    assert_eq!(v, "false");
    assert!(
        src.starts_with("default; cli alias") && src.ends_with("(unset)"),
        "{src}"
    );
    let r = resolve_with(
        &facts,
        &env_cfg(&[("INFR_MMV_MW_WARPS", "5000000000")]),
        &[],
        None,
    )
    .unwrap();
    let (v, src) = value_of(&r, "mmv_mw.warps@q4_k");
    assert_eq!(v, "1");
    assert!(src.starts_with("default;"), "{src}");
}

/// An alias set by a hand-built config (no layer recorded it, as `be_with` builds them) still
/// applies, labelled as a config alias; one set to its own default through a layer still counts
/// as set.
#[test]
fn aliases_apply_without_a_layer_record() {
    let mut cfg = Config::default();
    cfg.kernels.vulkan.flash_splits = Some(4);
    cfg.kernels.vulkan.gemv.sg_nr = 4;
    let r = resolve_with(&fixture("unknown"), &cfg, &[], None).unwrap();
    assert_eq!(
        value_of(&r, "attn_flash.splits"),
        (
            "4".into(),
            "config alias kernels.vulkan.flash_splits (INFR_FLASH_SPLITS)".into()
        )
    );
    assert_eq!(value_of(&r, "gemv_sg.rows").0, "4");
    // Set to the default value through a layer: still that layer's setting, and it outranks a
    // lower layer's tuning key.
    let cfg = Config::load_from_layers(&[
        file_layer("[kernels.vulkan.tune]\n\"gemv_sg.rows\" = 4\n"),
        env_layer(&[("INFR_GEMV_SG_NR", "2")]),
    ]);
    let r = resolve_with(&fixture("unknown"), &cfg, &[], None).unwrap();
    assert_eq!(
        value_of(&r, "gemv_sg.rows"),
        ("2".into(), "env alias INFR_GEMV_SG_NR".into())
    );
}

/// Precedence between an alias and the new key: at the SAME level the new key wins; across
/// levels the higher level wins whichever spelling it uses.
#[test]
fn alias_and_key_precedence() {
    let facts = fixture("unknown");
    let same_level = env_cfg(&[
        ("INFR_FLASH_SPLITS", "4"),
        ("INFR_TUNE", "attn_flash.splits=2"),
    ]);
    let r = resolve_with(&facts, &same_level, &[], None).unwrap();
    assert_eq!(value_of(&r, "attn_flash.splits").0, "2");
    let same_file = Config::load_from_layers(&[file_layer(
        "[kernels.vulkan]\nflash_splits = 4\n[kernels.vulkan.tune]\n\"attn_flash.splits\" = 2\n",
    )]);
    let r = resolve_with(&facts, &same_file, &[], None).unwrap();
    assert_eq!(value_of(&r, "attn_flash.splits").0, "2");
    // Env alias over a config-file key.
    let cfg = Config::load_from_layers(&[
        file_layer("[kernels.vulkan.tune]\n\"attn_flash.splits\" = 2\n"),
        env_layer(&[("INFR_FLASH_SPLITS", "4")]),
    ]);
    let r = resolve_with(&facts, &cfg, &[], None).unwrap();
    assert_eq!(
        value_of(&r, "attn_flash.splits"),
        ("4".into(), "env alias INFR_FLASH_SPLITS".into())
    );
    // Cli key over an env alias.
    let cfg = Config::load_from_layers(&[
        env_layer(&[("INFR_FLASH_SPLITS", "4")]),
        cli_layer(&["kernels.vulkan.tune.attn_flash.splits=8"]),
    ]);
    let r = resolve_with(&facts, &cfg, &[], None).unwrap();
    assert_eq!(value_of(&r, "attn_flash.splits").0, "8");
    // Cli alias over an env key.
    let cfg = Config::load_from_layers(&[
        env_layer(&[("INFR_TUNE", "attn_flash.splits=8")]),
        cli_layer(&["kernels.vulkan.flash_splits=3"]),
    ]);
    let r = resolve_with(&facts, &cfg, &[], None).unwrap();
    assert_eq!(
        value_of(&r, "attn_flash.splits"),
        (
            "3".into(),
            "cli alias --set kernels.vulkan.flash_splits".into()
        )
    );
}

/// The `kernels.vulkan.gemv` group resolves, for any env state, to tuning values that make the
/// GEMV dispatchers choose exactly what the pre-config per-call env reads chose (the same states
/// as `recorder.rs`'s `gemv_knobs_resolve_matches_env_reads`).
#[test]
fn gemv_aliases_resolve_like_the_old_env_reads() {
    let states: &[&[(&str, &str)]] = &[
        &[],
        &[
            ("INFR_NO_GEMV_RM", ""),
            ("INFR_NO_GEMV_SG", ""),
            ("INFR_NO_GEMV_ID_SG", ""),
            ("INFR_NO_GEMV_REG", ""),
        ],
        &[
            ("INFR_GEMV_RM", "4"),
            ("INFR_GEMV_RM_MAXOUT", "9000"),
            ("INFR_GEMV_RM_MINOUT", "1000"),
            ("INFR_GEMV_SG_MINOUT", "512"),
            ("INFR_GEMV_SG_MAXOUT", "4096"),
            ("INFR_GEMV_SG_NR", "8"),
            ("INFR_GEMV_VARIANT", "o4"),
        ],
        &[("INFR_GEMV_RM", "xx"), ("INFR_GEMV_SG_NR", "yy")],
    ];
    for st in states {
        let get = |k: &str| st.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
        let t = resolve_with(&fixture("unknown"), &env_cfg(st), &[], None)
            .unwrap()
            .tuning;
        let num = |k: &str, d: u64| get(k).and_then(|s| s.parse::<u64>().ok()).unwrap_or(d);
        let variant = if get("INFR_NO_GEMV_REG").is_some() {
            None
        } else {
            get("INFR_GEMV_VARIANT").or_else(|| Some("reg".to_string()))
        };
        // A name with no build (`o4` here) found nothing and took the RM path: `None`.
        let variant = variant.filter(|v| crate::gemm::GEMV_RM_V2_VARIANTS.contains(&v.as_str()));
        assert_eq!(t.gemv_variant, variant, "{st:?}");
        assert_eq!(t.gemv_rm_enabled, get("INFR_NO_GEMV_RM").is_none());
        assert_eq!(u64::from(t.gemv_rm_rows), num("INFR_GEMV_RM", 2));
        assert_eq!(
            t.gemv_rm_max_out as u64,
            num("INFR_GEMV_RM_MAXOUT", u64::MAX)
        );
        assert_eq!(t.gemv_rm_min_out as u64, num("INFR_GEMV_RM_MINOUT", 2048));
        assert_eq!(t.gemv_sg_enabled, get("INFR_NO_GEMV_SG").is_none());
        assert_eq!(t.gemv_id_sg_enabled, get("INFR_NO_GEMV_ID_SG").is_none());
        assert_eq!(t.gemv_sg_min_out as u64, num("INFR_GEMV_SG_MINOUT", 2048));
        assert_eq!(t.gemv_sg_max_out as u64, num("INFR_GEMV_SG_MAXOUT", 8192));
        assert_eq!(u64::from(t.gemv_sg_rows), num("INFR_GEMV_SG_NR", 2));
    }
}

// ── the report ───────────────────────────────────────────────────────────────

/// Equal sets hash equal and a one-value change moves the hash.
#[test]
fn hash_tracks_the_values() {
    let a = resolve_fixture("rx7900xtx-amdwin", &Config::default()).report;
    let b = resolve_fixture("rx7900xtx-amdwin", &Config::default()).report;
    assert_eq!(a.hash(), b.hash());
    let c = resolve_fixture("rx7900xtx-amdwin", &env_cfg(&[("INFR_GEMV_SG_NR", "4")])).report;
    assert_ne!(a.hash(), c.hash());
    assert_eq!(a.hash().len(), 8);
}

/// Each fixture's resolved set (`infr tune show`'s text under the default config) matches its
/// checked-in snapshot. A table change that moves a value on hardware it does not name fails
/// here. Regenerate with `TUNING_SNAPSHOT_UPDATE=1` and review the diff.
#[test]
fn fixture_snapshots() {
    let dir = tuning_dir().join("snapshots");
    let update = std::env::var_os("TUNING_SNAPSHOT_UPDATE").is_some();
    let mut failed = Vec::new();
    for (id, _) in fixtures() {
        let got = resolve_fixture(&id, &Config::default()).report.show();
        let path = dir.join(format!("{id}.txt"));
        if update {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&path, &got).unwrap();
            continue;
        }
        let want = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .replace("\r\n", "\n");
        if want != got {
            failed.push(format!(
                "--- {id} (snapshot)\n{want}\n+++ {id} (now)\n{got}"
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "tuning snapshots changed:\n{}",
        failed.join("\n")
    );
}

// ── the shader inventory ─────────────────────────────────────────────────────

/// Every `.comp` source has a class, every class entry names a source, and each assignment
/// agrees with what the source uses: class 4 exactly the cooperative-matrix kernels, class 2
/// only kernels with subgroup operations, class 5 none with any.
#[test]
fn every_shader_is_classified() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let mut found: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension()? == "comp").then(|| p.file_stem()?.to_str().map(str::to_string))?
        })
        .collect();
    found.sort_unstable();
    assert!(
        found.len() > 100,
        "the shader scan found {} files",
        found.len()
    );
    let listed: Vec<&str> = SHADER_CLASSES.iter().map(|(n, _)| *n).collect();
    let missing: Vec<&String> = found
        .iter()
        .filter(|f| !listed.contains(&f.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "shaders with no class in tuning/classes.rs: {missing:?}"
    );
    let stale: Vec<&&str> = listed
        .iter()
        .filter(|l| !found.iter().any(|f| f == *l))
        .collect();
    assert!(
        stale.is_empty(),
        "classes.rs names shaders that do not exist: {stale:?}"
    );
    for (name, class) in SHADER_CLASSES {
        let src = std::fs::read_to_string(dir.join(format!("{name}.comp"))).unwrap();
        let code = strip_comments(&src);
        let coopmat = code.contains("coopmat") || code.contains("GL_KHR_cooperative_matrix");
        let subgroup = uses_subgroup_ops(&code);
        match class {
            ShaderClass::C4 => assert!(coopmat, "{name}: class 4 but no cooperative matrix"),
            _ => assert!(
                !coopmat,
                "{name}: uses cooperative matrix but is not class 4"
            ),
        }
        match class {
            ShaderClass::C2 => assert!(subgroup, "{name}: class 2 but no subgroup operation"),
            ShaderClass::C5a | ShaderClass::C5b => {
                assert!(!subgroup, "{name}: class 5 but uses subgroup operations")
            }
            _ => {}
        }
    }
}

fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix("//") {
            rest = r.find('\n').map_or("", |i| &r[i..]);
        } else if let Some(r) = rest.strip_prefix("/*") {
            rest = r.find("*/").map_or("", |i| &r[i + 2..]);
        } else {
            let c = rest.chars().next().unwrap();
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

/// A subgroup builtin, a `subgroup*(` operation, or one of `wave64.glsl`'s `w64_*(` wrappers.
fn uses_subgroup_ops(code: &str) -> bool {
    if code.contains("gl_Subgroup") || code.contains("gl_NumSubgroups") {
        return true;
    }
    let call_after = |prefix: &str| {
        code.match_indices(prefix).any(|(i, _)| {
            let before_ok = i == 0
                || !code[..i]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
            let tail = &code[i + prefix.len()..];
            let ident_end = tail
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(tail.len());
            before_ok && ident_end > 0 && tail[ident_end..].trim_start().starts_with('(')
        })
    };
    call_after("subgroup") || call_after("w64_")
}

/// The inventory check is not vacuous: a class-5 assignment for a kernel that uses subgroup
/// operations, and a class-4 assignment for a kernel with no cooperative matrix, are both caught.
#[test]
fn the_inventory_checks_can_fail() {
    let rms = strip_comments(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/rmsnorm.comp"),
        )
        .unwrap(),
    );
    assert!(uses_subgroup_ops(&rms));
    let add = strip_comments(
        &std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/add.comp"))
            .unwrap(),
    );
    assert!(!uses_subgroup_ops(&add) && !add.contains("coopmat"));
}

// ── declared ranges against what the code downstream can hold ────────────────

/// Every bounded parameter refuses the first value past its bound, naming the key: the bounds
/// are what the code after it can hold (the 8-split partials scratch, the watchdog-safe MoE
/// small-batch ceiling, divisors that must not be 0, the builds that exist).
#[test]
fn values_past_each_bound_are_refused() {
    let facts = fixture("rx7900xtx-amdwin");
    let cases: &[(&str, &str, &str)] = &[
        ("attn_flash.splits", "9", "0..=8"),
        ("attn_pv.splits", "9", "0..=8"),
        ("attn_flash.max_splits", "9", "2..=8"),
        ("attn_pv.max_splits", "9", "2..=8"),
        ("attn_pv.max_splits", "64", "2..=8"),
        ("attn_flash.max_splits", "1", "2..=8"),
        ("attn_pv.max_splits", "1", "2..=8"),
        ("attn_flash.kv_per_split", "0", "1..="),
        ("attn_pv.kv_per_split", "0", "1..="),
        ("attn_canvas.chunks", "0", "1..="),
        ("moe.small_m", "65", "0..=64"),
        ("device.sg_pref", "64", "16 or 32"),
        ("mmv_mw.warps", "3", "1|2|4|8|16"),
        ("mmv_mw.warps@q4_k", "32", "1|2|4|8|16"),
        ("gemv.variant", "bogus", "reg|sg|dbuf|wg128|none"),
        ("gemv.variant", "REG", "reg|sg|dbuf|wg128|none"),
    ];
    for (key, bad, range) in cases {
        let e = resolve_with(
            &facts,
            &env_cfg(&[("INFR_TUNE", &format!("{key}={bad}"))]),
            &[],
            None,
        )
        .err()
        .unwrap_or_else(|| panic!("`{key}={bad}` was accepted"));
        assert!(e.contains(key) && e.contains(range), "{key}={bad}: {e}");
    }
    // The values at each bound are accepted.
    for ok in [
        "attn_flash.splits=8",
        "attn_pv.splits=8",
        "attn_pv.max_splits=8",
        "moe.small_m=64",
        "mmv_mw.warps=16",
        "gemv.variant=none",
        "gemv.variant=wg128",
    ] {
        resolve_with(&facts, &env_cfg(&[("INFR_TUNE", ok)]), &[], None)
            .unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
    // The split bounds ARE the scratch the adapter sizes.
    for key in ["attn_flash.splits", "attn_pv.splits"] {
        assert_eq!(
            families::find(key).unwrap().ty.describe(),
            format!("0..={}", super::ATTN_PREFILL_MAX_SPLITS)
        );
    }
}

/// With every parameter at its most aggressive in-range value, the split policy still never
/// asks for more partials than the scratch holds (the review's `max_splits=64,target_wg=100000`
/// is now refused, and the largest accepted set stays inside).
#[test]
fn split_policy_stays_inside_the_scratch_at_the_extremes() {
    let cfg = env_cfg(&[(
        "INFR_TUNE",
        "attn_pv.max_splits=8,attn_pv.target_wg=4294967295,attn_pv.min_kv=0,attn_pv.kv_per_split=1,\
         attn_pv.full_wg=4294967295,attn_flash.max_splits=8,attn_flash.target_wg=4294967295",
    )]);
    let t = resolve_with(&fixture("unknown"), &cfg, &[], None)
        .unwrap()
        .tuning;
    for base in [0u32, 1, 2, 100, 1 << 20] {
        for kv in [0u32, 1, 4096, 1 << 30] {
            assert!(t.pv.count(base, kv) <= super::ATTN_PREFILL_MAX_SPLITS);
            assert!(t.flash.count(base, kv) <= super::ATTN_PREFILL_MAX_SPLITS);
        }
    }
    let e = resolve_with(
        &fixture("unknown"),
        &env_cfg(&[(
            "INFR_TUNE",
            "attn_pv.max_splits=64,attn_pv.target_wg=100000",
        )]),
        &[],
        None,
    )
    .unwrap_err();
    assert!(e.contains("attn_pv.max_splits"), "{e}");
}

/// The dispatch-side guard: a split count past the scratch panics instead of writing past it.
#[test]
#[should_panic(expected = "outside 1..=8")]
fn a_split_count_past_the_scratch_panics() {
    let t = fixture_tuning("unknown");
    super::AttnSplit {
        splits: super::ATTN_PREFILL_MAX_SPLITS + 1,
        ..t.flash
    }
    .count(1, 1);
}

// ── old knobs at their edges: exactly the old behaviour ──────────────────────

/// What a resolved alias must give, for one raw environment value.
#[derive(Debug, PartialEq)]
enum Old {
    /// The parameter's value (what the old read site used, or an equivalent the kernels treat
    /// identically).
    Is(String),
    /// Refused with an error naming the knob (only values that wrote out of bounds before).
    Refused,
}

/// For every old environment knob and the edge inputs — empty, whitespace, malformed, negative,
/// zero, at and past each bound, past `u32` and past `u64` — the resolved parameter is exactly
/// what the old read site at the merge base computed from the same input. Each `old` closure
/// transcribes that read site: the env layer's `num`/`opt_num`/presence grammar (`s.parse()`,
/// untrimmed), then the site's own fallback or clamp.
#[test]
fn old_knobs_at_their_edges_behave_as_before() {
    let inputs = [
        "",
        " ",
        " 4",
        "abc",
        "-1",
        "0",
        "1",
        "2",
        "3",
        "4",
        "8",
        "9",
        "32",
        "64",
        "65",
        "100000",
        "4294967296",
        "99999999999999999999",
        "sg",
        "none",
        "reg",
    ];
    let num32 = |raw: &str, d: u64| raw.parse::<u32>().map(u64::from).ok().unwrap_or(d);
    let numz = |raw: &str, d: u64| raw.parse::<usize>().map(|v| v as u64).ok().unwrap_or(d);
    let splits = |raw: &str| match raw.parse::<usize>() {
        Err(_) => Old::Is("0".into()),
        Ok(v) => match u32::try_from(v).ok().filter(|&n| n > 0) {
            None => Old::Is("0".into()),
            Some(n) if n <= super::ATTN_PREFILL_MAX_SPLITS => Old::Is(n.to_string()),
            Some(_) => Old::Refused,
        },
    };
    let show = |v: u64| {
        if v == u64::MAX {
            "max".to_string()
        } else {
            v.to_string()
        }
    };
    type OldFn<'a> = Box<dyn Fn(&str) -> Old + 'a>;
    let specs: Vec<(&str, &str, OldFn)> = vec![
        (
            "INFR_GEMV_RM",
            "gemv_rm.rows",
            Box::new(|r| Old::Is(num32(r, 2).to_string())),
        ),
        (
            "INFR_GEMV_RM_MAXOUT",
            "gemv_rm.max_out",
            Box::new(|r| Old::Is(show(numz(r, u64::MAX)))),
        ),
        (
            "INFR_GEMV_RM_MINOUT",
            "gemv_rm.min_out",
            Box::new(|r| Old::Is(numz(r, 2048).to_string())),
        ),
        (
            "INFR_GEMV_SG_MINOUT",
            "gemv_sg.min_out",
            Box::new(|r| Old::Is(numz(r, 2048).to_string())),
        ),
        (
            "INFR_GEMV_SG_MAXOUT",
            "gemv_sg.max_out",
            Box::new(|r| Old::Is(numz(r, 8192).to_string())),
        ),
        (
            "INFR_GEMV_SG_NR",
            "gemv_sg.rows",
            Box::new(|r| Old::Is(num32(r, 2).to_string())),
        ),
        // An unknown name, the empty one included, found no build: the RM path, as `none`.
        (
            "INFR_GEMV_VARIANT",
            "gemv.variant",
            Box::new(|r| {
                Old::Is(
                    if crate::gemm::GEMV_RM_V2_VARIANTS.contains(&r) {
                        r
                    } else {
                        "none"
                    }
                    .into(),
                )
            }),
        ),
        (
            "INFR_NO_GEMV_REG",
            "gemv.variant",
            Box::new(|_| Old::Is("none".into())),
        ),
        (
            "INFR_NO_GEMV_RM",
            "gemv_rm.enabled",
            Box::new(|_| Old::Is("false".into())),
        ),
        (
            "INFR_NO_GEMV_SG",
            "gemv_sg.enabled",
            Box::new(|_| Old::Is("false".into())),
        ),
        (
            "INFR_NO_GEMV_ID_SG",
            "gemv_id_sg.enabled",
            Box::new(|_| Old::Is("false".into())),
        ),
        (
            "INFR_NO_I8CM_KQ",
            "gemm_i8cm_kq.enabled",
            Box::new(|_| Old::Is("false".into())),
        ),
        (
            "INFR_MOE_SMALL_M",
            "moe.small_m",
            Box::new(|r| Old::Is(numz(r, 8).min(64).to_string())),
        ),
        (
            "INFR_CANVAS_CHUNK_N",
            "attn_canvas.chunks",
            Box::new(|r| Old::Is(numz(r, 3).max(1).to_string())),
        ),
        ("INFR_FLASH_SPLITS", "attn_flash.splits", Box::new(splits)),
        ("INFR_PV_SPLITS", "attn_pv.splits", Box::new(splits)),
        (
            "INFR_FLASH_BM",
            "attn_flash.force_bm32",
            Box::new(|r| Old::Is((r == "32").to_string())),
        ),
        (
            "INFR_FLASH_MIN_ROWS",
            "attn_flash.min_rows",
            Box::new(|r| Old::Is(numz(r, 24).to_string())),
        ),
        // The multi-warp GEMV is unreachable (`unified_mmv_row1`), so the old WARPS value reached
        // no dispatch whatever it was; a count with no build now leaves the default instead of
        // carrying a number nothing reads. A built count is kept as before.
        (
            "INFR_MMV_MW_WARPS",
            "mmv_mw.warps@q6_k",
            Box::new(|r| {
                let w = r.parse::<usize>().ok().and_then(|w| u32::try_from(w).ok());
                Old::Is(match w {
                    Some(w @ (1 | 2 | 4 | 8 | 16)) => w.to_string(),
                    _ => "8".into(),
                })
            }),
        ),
    ];
    let facts = fixture("rx7900xtx-amdwin");
    for (env, key, old) in &specs {
        for raw in inputs {
            let want = old(raw);
            let got = match resolve_with(&facts, &env_cfg(&[(env, raw)]), table::builtin(), None) {
                Ok(r) => Old::Is(value_of(&r, key).0),
                Err(e) => {
                    assert!(e.contains(env), "{env}={raw:?}: error names no knob: {e}");
                    Old::Refused
                }
            };
            assert_eq!(got, want, "{env}={raw:?} → `{key}`");
        }
    }
    // `INFR_SG` keeps its loud env grammar: anything but 16 or 32 is refused by the env layer.
    for raw in ["", " ", "16 ", "17", "64", "abc"] {
        let get = |k: &str| (k == "INFR_SG").then(|| raw.to_string());
        assert!(
            infr_core::config::env::parse(&get).is_err(),
            "INFR_SG={raw:?}"
        );
    }
}

/// The subgroup width the report (and its hash) carries is the one IN USE: `INFR_SG=16` on a
/// device that cannot pin 16 resolves to 32, and says why; where 16 is pinnable it stays 16.
#[test]
fn an_unpinnable_sg_pref_reports_the_width_in_use() {
    let r = resolve_with(
        &fixture("unknown"),
        &env_cfg(&[("INFR_SG", "16")]),
        &[],
        None,
    )
    .unwrap();
    let (v, src) = value_of(&r, "device.sg_pref");
    assert_eq!(v, "32");
    assert!(
        src.contains("INFR_SG") && src.contains("cannot pin 16"),
        "{src}"
    );
    assert_eq!(r.tuning.sg_pref, 32);
    let plain = resolve_with(&fixture("unknown"), &Config::default(), &[], None).unwrap();
    assert_eq!(
        r.report.hash(),
        plain.report.hash(),
        "the hash is of the width in use"
    );
    let r = resolve_with(
        &fixture("intel-arc-a770-anv"),
        &env_cfg(&[("INFR_SG", "16")]),
        &[],
        None,
    )
    .unwrap();
    assert_eq!(value_of(&r, "device.sg_pref").0, "16");
}

// ── precedence edge cases ────────────────────────────────────────────────────

/// An alias set at a higher level to a value with no effect (its unset spelling) resets the
/// parameter to what the table gives, beating a lower level's tuning key: the higher level wins
/// whichever spelling it uses.
#[test]
fn an_unset_alias_at_a_higher_level_wins() {
    let facts = fixture("rx7900xtx-amdwin");
    let cfg = Config::load_from_layers(&[
        file_layer("[kernels.vulkan.tune]\n\"mmv_mw.warps\" = 4\n"),
        cli_layer(&["kernels.vulkan.mmv_mw_warps=none"]),
    ]);
    let r = resolve_with(&facts, &cfg, table::builtin(), None).unwrap();
    let (v, src) = value_of(&r, "mmv_mw.warps@q4_k");
    assert_eq!(v, "1");
    assert!(
        src.contains("cli alias") && src.contains("(unset)"),
        "{src}"
    );
    // `i8cm_kq = true` was always "let the driver decide": over a lower key turning the GEMM off,
    // it restores the row's value on the driver the row names.
    let cfg = Config::load_from_layers(&[
        env_layer(&[("INFR_TUNE", "gemm_i8cm_kq.enabled=false")]),
        cli_layer(&["kernels.vulkan.i8cm_kq=true"]),
    ]);
    let r = resolve_with(&facts, &cfg, table::builtin(), None).unwrap();
    assert_eq!(value_of(&r, "gemm_i8cm_kq.enabled").0, "true");
    // At the SAME level the new key still wins over the alias's unset.
    let cfg = Config::load_from_layers(&[cli_layer(&[
        "kernels.vulkan.mmv_mw_warps=none",
        "kernels.vulkan.tune.mmv_mw.warps=4",
    ])]);
    let r = resolve_with(&facts, &cfg, table::builtin(), None).unwrap();
    assert_eq!(value_of(&r, "mmv_mw.warps@q4_k").0, "4");
}

/// Within one command line the whole-map spelling and per-key spellings of `tune` merge.
#[test]
fn tune_spellings_on_one_command_line_merge() {
    let cfg = Config::load_from_layers(&[cli_layer(&[
        "kernels.vulkan.tune.gemv_sg.rows=8",
        "kernels.vulkan.tune=moe.small_m=4",
    ])]);
    let r = resolve_with(&fixture("unknown"), &cfg, &[], None).unwrap();
    assert_eq!(value_of(&r, "gemv_sg.rows").0, "8");
    assert_eq!(value_of(&r, "moe.small_m").0, "4");
}

/// A `Config` built in code records no layers, so a numeric alias field equal to its config
/// default cannot be told from an unset one and counts as unset (a tuning key in the same config
/// then wins); a different value counts as set at the config level.
#[test]
fn a_code_built_alias_at_its_default_is_unset() {
    let mut cfg = Config::default();
    cfg.kernels.vulkan.gemv.sg_nr = 2;
    cfg.kernels
        .vulkan
        .tune
        .entries
        .push(infr_core::config::TuneEntry {
            key: "gemv_sg.rows".into(),
            value: "4".into(),
            level: infr_core::config::TuneLevel::Config,
        });
    let r = resolve_with(&fixture("unknown"), &cfg, &[], None).unwrap();
    assert_eq!(value_of(&r, "gemv_sg.rows").0, "4");
    cfg.kernels.vulkan.gemv.sg_nr = 8;
    let r = resolve_with(&fixture("unknown"), &cfg, &[], None).unwrap();
    // Same level: the key beats the alias.
    assert_eq!(value_of(&r, "gemv_sg.rows").0, "4");
    cfg.kernels.vulkan.tune.entries.clear();
    let r = resolve_with(&fixture("unknown"), &cfg, &[], None).unwrap();
    assert_eq!(value_of(&r, "gemv_sg.rows").0, "8");
}

// ── row match keys ───────────────────────────────────────────────────────────

/// A row's driver must be a Vulkan driver id name (a typo would silently match nothing); the
/// vendor and arch were already checked.
#[test]
fn row_match_values_are_checked() {
    let bad = |m: &str| {
        table::parse(&format!("[[row]]\nid = \"x\"\nmatch = {{ {m} }}\n"), "t").unwrap_err()
    };
    assert!(bad("vendor = \"amd\", driver = \"AMD_PROPIETARY\"").contains("driver id name"));
    assert!(bad("vendor = \"amd\", arch = \"Rdna3\"").contains("architecture"));
    assert!(bad("vendor = \"ati\"").contains("vendor"));
    for ok in [
        "AMD_PROPRIETARY",
        "MESA_RADV",
        "NVIDIA_PROPRIETARY",
        "unreported",
    ] {
        table::parse(
            &format!("[[row]]\nid = \"x\"\nmatch = {{ vendor = \"amd\", driver = \"{ok}\" }}\n"),
            "t",
        )
        .unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
}

/// The structural tie check over the rows themselves: no two built-in rows that one device could
/// both match, at the same level and specificity, set a parameter differently — whether or not
/// any fixture is that device.
#[test]
fn no_structural_ties_in_the_table() {
    let c = table::conflicts(table::builtin());
    assert!(c.is_empty(), "{c:#?}");
}

/// The structural check catches a tie no fixture exercises, and leaves rows that cannot meet.
#[test]
fn the_structural_tie_check_can_fail() {
    let rows = |a: &str, b: &str| {
        table::parse(
            &format!(
                "[[row]]\nid = \"a\"\nmatch = {{ {a} }}\nparams = {{ \"gemv_sg.rows\" = 4 }}\n\
                 [[row]]\nid = \"b\"\nmatch = {{ {b} }}\nparams = {{ \"gemv_sg.rows\" = 8 }}\n"
            ),
            "t",
        )
        .unwrap()
    };
    // No fixture is an integrated AMD on RADV with 40+ CUs, but one device could be both.
    assert_eq!(
        table::conflicts(&rows(
            "vendor = \"amd\", driver = \"MESA_RADV\"",
            "vendor = \"amd\", cu_min = 40"
        ))
        .len(),
        1
    );
    // Different drivers can never meet; different specificity is ordered, not tied.
    assert!(table::conflicts(&rows(
        "vendor = \"amd\", driver = \"MESA_RADV\"",
        "vendor = \"amd\", driver = \"AMD_PROPRIETARY\""
    ))
    .is_empty());
    assert!(table::conflicts(&rows(
        "vendor = \"amd\", driver = \"MESA_RADV\"",
        "vendor = \"amd\", driver = \"MESA_RADV\", integrated = true"
    ))
    .is_empty());
    assert!(table::conflicts(&rows(
        "vendor = \"amd\", cu_min = 40",
        "vendor = \"amd\", cu_max = 20"
    ))
    .is_empty());
}

/// `rows = false` is "generic defaults only": on AMD's proprietary driver it also turns off the
/// two rows that driver's GEMM choices come from, and the report says so.
#[test]
fn rows_off_drops_the_driver_rows_and_says_so() {
    let mut cfg = Config::default();
    cfg.kernels.vulkan.tuning.rows = false;
    let r = resolve_fixture("rx7900xtx-amdwin", &cfg);
    assert!(!r.tuning.i8cm_kq_enabled && !r.tuning.split_k_scale_by_cu);
    assert!(r.report.rows_off);
    let text = r.report.show();
    assert!(
        text.contains("rows: off") && text.contains("int8 cooperative-matrix"),
        "{text}"
    );
    assert!(r.report.list(None).contains("INFR_TUNE_NO_ROWS"));
}

/// The first forward's submit cap follows the device facts (issue 94): the GCN fixture, a
/// discrete card with neither matrix units nor an accelerated int8 dot, starts split at 64
/// dispatches; the integrated GPU keeps its 128 and the RX 7900 XTX stays unsplit. Before the
/// facts existed every discrete device started unsplit, the GCN card included.
#[test]
fn initial_submit_cap_follows_the_device_facts() {
    let cap = |id: &str| fixture_tuning(id).submit_initial_dispatches;
    assert_eq!(cap("rx580-gcn-radv"), 64);
    assert_eq!(cap("raphael-igpu-amdwin"), 128);
    assert_eq!(cap("rx7900xtx-amdwin"), 0);
    assert_eq!(cap("rx7900xtx-radv"), 0);
    assert!(fixture("rx580-gcn-radv").slow_class());
    assert!(!fixture("rx7900xtx-amdwin").slow_class());
    // Either fact alone keeps a discrete device out of the slow class; wave64 alone puts it in.
    let mut f = fixture("rx7900xtx-amdwin");
    f.matrix_units = false;
    assert!(!f.slow_class(), "an accelerated dot is enough");
    f.dot_accelerated = false;
    assert!(f.slow_class());
    let mut w = fixture("rx7900xtx-amdwin");
    w.wave64 = true;
    assert!(w.slow_class());
    // A user value wins over the derived default.
    let r = resolve_fixture(
        "rx580-gcn-radv",
        &env_cfg(&[("INFR_TUNE", "submit.initial_dispatches=16")]),
    );
    assert_eq!(r.tuning.submit_initial_dispatches, 16);
}

/// `queue.family` is `auto` by default on every discrete device, so inference leaves the ring
/// the compositor renders on wherever a usable compute-only family exists. An integrated GPU is
/// `graphics` by rule (the compute ring's hang budget measured tighter there under RADV,
/// `docs/igpu.md`) and `auto` only through the row of a driver it was measured on.
#[test]
fn queue_family_follows_the_device_facts() {
    use super::QueueFamily::{Auto, Compute, Graphics};
    let of = |id: &str| fixture_tuning(id).queue_family;
    for discrete in [
        "rx580-gcn-radv",
        "rx7900xtx-amdwin",
        "rx7900xtx-radv",
        "nvidia-rtx2080ti",
        "intel-arc-a770-anv",
        "unknown",
    ] {
        assert_eq!(of(discrete), Auto, "{discrete}");
    }
    // The integrated GPU: `auto` by the AMD Windows driver's row, `graphics` without the rows
    // and under any other driver.
    let r = resolve_fixture("raphael-igpu-amdwin", &Config::default());
    assert_eq!(r.tuning.queue_family, Auto);
    assert_eq!(
        value_of(&r, "queue.family"),
        (
            "auto".into(),
            "row amd-proprietary-integrated-compute-queue".into()
        )
    );
    let igpu = fixture("raphael-igpu-amdwin");
    let r = resolve_with(&igpu, &Config::default(), &[], None).unwrap();
    assert_eq!(r.tuning.queue_family, Graphics);
    let mut radv = igpu.clone();
    radv.driver = Some("MESA_RADV".to_string());
    let r = resolve_with(&radv, &Config::default(), table::builtin(), None).unwrap();
    assert_eq!(r.tuning.queue_family, Graphics);
    // The alias, and a user's value on an integrated device.
    let r = resolve_fixture("rx580-gcn-radv", &env_cfg(&[("INFR_GFX_QUEUE", "1")]));
    assert_eq!(r.tuning.queue_family, Graphics);
    let r = resolve_with(
        &radv,
        &env_cfg(&[("INFR_TUNE", "queue.family=compute")]),
        table::builtin(),
        None,
    )
    .unwrap();
    assert_eq!(r.tuning.queue_family, Compute);
}

/// The three surfaces a user sets `queue.family` from all reach the resolver, each labelled
/// with where it came from and the later one winning: the config file's `[kernels.vulkan.tune]`
/// table, `INFR_TUNE`, and `--set kernels.vulkan.tune.queue.family=...` on the command line.
#[test]
fn queue_family_is_set_from_config_env_and_command_line() {
    use super::QueueFamily::{Compute, Graphics};
    let facts = fixture("rx7900xtx-amdwin");
    let file = file_layer("[kernels.vulkan.tune]\n\"queue.family\" = \"graphics\"\n");
    let env = env_layer(&[("INFR_TUNE", "queue.family=compute")]);
    let cli = cli_layer(&["kernels.vulkan.tune.queue.family=graphics"]);
    let resolve = |layers: &[PartialConfig]| {
        resolve_with(
            &facts,
            &Config::load_from_layers(layers),
            table::builtin(),
            None,
        )
        .unwrap()
    };
    let r = resolve(std::slice::from_ref(&file));
    assert_eq!(r.tuning.queue_family, Graphics);
    assert!(
        value_of(&r, "queue.family").1.starts_with("config"),
        "{:?}",
        value_of(&r, "queue.family")
    );
    let r = resolve(&[file.clone(), env.clone()]);
    assert_eq!(r.tuning.queue_family, Compute);
    assert!(
        value_of(&r, "queue.family").1.starts_with("env INFR_TUNE"),
        "{:?}",
        value_of(&r, "queue.family")
    );
    let r = resolve(&[file, env, cli]);
    assert_eq!(r.tuning.queue_family, Graphics);
    assert!(
        value_of(&r, "queue.family").1.starts_with("cli --set"),
        "{:?}",
        value_of(&r, "queue.family")
    );
}

/// `queue.family` takes exactly its three words: every one parses to its value and prints back
/// as itself, and anything else (the old on/off spellings, `none`, another case) is an error
/// that names the three.
#[test]
fn queue_family_takes_its_three_words_only() {
    let facts = fixture("rx7900xtx-amdwin");
    for (word, value) in super::QueueFamily::ALL {
        let cfg = env_cfg(&[("INFR_TUNE", &format!("queue.family={word}"))]);
        let r = resolve_with(&facts, &cfg, &[], None).unwrap();
        assert_eq!(r.tuning.queue_family, value, "{word}");
        assert_eq!(value_of(&r, "queue.family").0, word);
        assert_eq!(value.name(), word);
        assert_eq!(super::QueueFamily::from_name(word), Some(value));
    }
    assert_eq!(super::QueueFamily::from_name("gfx"), None);
    for bad in ["true", "on", "none", "Auto", "gfx"] {
        let cfg = env_cfg(&[("INFR_TUNE", &format!("queue.family={bad}"))]);
        let e = resolve_with(&facts, &cfg, &[], None).unwrap_err();
        assert!(
            e.contains("queue.family") && e.contains("auto|compute|graphics"),
            "{bad:?}: {e}"
        );
    }
}

/// `attn_split.decode_min_chunk` keeps PR 105's floor on the class it was measured on, a discrete
/// device in the slow class (`SLOW_CLASS_DECODE_MIN_CHUNK`), and elsewhere the one every device
/// ran before it (`DECODE_MIN_CHUNK`); `infr tune show` reads the same field the attention
/// lowering does. A floor above the policy's ceiling is refused.
#[test]
fn decode_chunk_floor_follows_the_device_facts() {
    let floor = |id: &str| fixture_tuning(id).attn_decode_min_chunk;
    assert_eq!(floor("rx580-gcn-radv"), 16);
    assert_eq!(floor("rx7900xtx-amdwin"), 64);
    assert_eq!(floor("rx7900xtx-radv"), 64);
    assert_eq!(floor("raphael-igpu-amdwin"), 64);
    let r = resolve_fixture(
        "rx7900xtx-amdwin",
        &env_cfg(&[("INFR_TUNE", "attn_split.decode_min_chunk=16")]),
    );
    assert_eq!(r.tuning.attn_decode_min_chunk, 16);
    assert_eq!(value_of(&r, "attn_split.decode_min_chunk").0, "16");
    let cfg = env_cfg(&[("INFR_TUNE", "attn_split.decode_min_chunk=513")]);
    let e = resolve_with(&fixture("rx7900xtx-amdwin"), &cfg, &[], None).unwrap_err();
    assert!(e.contains("out of range"), "{e}");
}

/// `gemv.kq_fma` is on only where the driver does not report an accelerated packed int8 dot,
/// which is where the fp32 decode GEMVs were measured (an RX 580 under RADV). The RX 7900 XTX and
/// the integrated GPU keep the kernels they were tuned with; `INFR_KQ_FMA_GEMV` and `INFR_TUNE`
/// set it either way.
#[test]
fn kq_fma_follows_the_dot_fact() {
    let on = |id: &str| fixture_tuning(id).gemv_kq_fma;
    assert!(on("rx580-gcn-radv"));
    assert!(!on("rx7900xtx-amdwin"));
    assert!(!on("rx7900xtx-radv"));
    assert!(!on("raphael-igpu-amdwin"));
    // The dot fact alone decides: matrix units do not turn it on or off.
    let mut f = fixture("rx7900xtx-amdwin");
    f.matrix_units = false;
    assert!(
        !resolve_with(&f, &Config::default(), &[], None)
            .unwrap()
            .tuning
            .gemv_kq_fma
    );
    f.dot_accelerated = false;
    assert!(
        resolve_with(&f, &Config::default(), &[], None)
            .unwrap()
            .tuning
            .gemv_kq_fma
    );
    let r = resolve_fixture("rx7900xtx-amdwin", &env_cfg(&[("INFR_KQ_FMA_GEMV", "1")]));
    assert!(r.tuning.gemv_kq_fma);
    let r = resolve_fixture(
        "rx580-gcn-radv",
        &env_cfg(&[("INFR_TUNE", "gemv.kq_fma=false")]),
    );
    assert!(!r.tuning.gemv_kq_fma);
}

/// `gemm.kq_fma` is off by default on every device. Where int8 arms run, a prefill with only its
/// Q4_0/Q4_K/Q6_K Linears on f32 measured further from the CPU reference than all-int8 (gemma-3);
/// where none run, the tile put gemma-3 outside its FloatNoCoopmat row on the integrated GPU.
/// `INFR_TUNE` turns it on, on any device; whether the device can then take the kernel (shared
/// memory, grid width) is `kq_fma_plan`'s check.
#[test]
fn the_kq_fma_gemm_is_off_unless_set() {
    let all = fixtures();
    assert!(
        all.iter().any(|(_, f)| !f.dot_accelerated),
        "no fixture without an accelerated dot"
    );
    for (id, f) in &all {
        let r = resolve_with(f, &Config::default(), &[], None).unwrap();
        assert!(!r.tuning.gemm_kq_fma, "{id}: on by default");
        let r = resolve_with(f, &env_cfg(&[("INFR_TUNE", "gemm.kq_fma=true")]), &[], None).unwrap();
        assert!(r.tuning.gemm_kq_fma, "{id}: INFR_TUNE did not set it");
    }
}

/// `device.i8_dot` is on everywhere but the class its table row names, integrated Xe1 (Xe-LP and
/// Xe-LPG) under ANV, where the packed int8 dot kernels failed gemma-3's tolerance rows that the
/// float kernels pass. `rows = false` and `INFR_TUNE` turn it back on there.
#[test]
fn i8_dot_is_off_only_on_the_class_its_row_names() {
    const IGPU: &str = "intel-rplu-igpu-anv";
    for (id, f) in fixtures() {
        let r = resolve_with(&f, &Config::default(), table::builtin(), None).unwrap();
        assert_eq!(r.tuning.i8_dot, id != IGPU, "{id}");
    }
    assert_eq!(
        value_of(&resolve_fixture(IGPU, &Config::default()), "device.i8_dot"),
        (
            "false".into(),
            "row intel-xe-lp-anv-integrated-i8-dot".into()
        )
    );
    // The discrete Xe1 part under the same driver keeps it: the row names `integrated`.
    assert!(fixture_tuning("intel-arc-a770-anv").i8_dot);
    let mut cfg = Config::default();
    cfg.kernels.vulkan.tuning.rows = false;
    assert!(resolve_fixture(IGPU, &cfg).tuning.i8_dot);
    let r = resolve_fixture(IGPU, &env_cfg(&[("INFR_TUNE", "device.i8_dot=true")]));
    assert!(r.tuning.i8_dot);
}
