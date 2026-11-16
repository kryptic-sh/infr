//! Every tunable parameter, one descriptor each: what it is, what values it takes, its generic
//! default as a function of the device's facts, and how it reaches the typed [`Tuning`] struct.
//!
//! S1 of `docs/plan/10-tunable-kernels.md`: every value here is a runtime choice that existed
//! before the table did, migrated VERBATIM — the generic defaults below reproduce exactly what the
//! code chose before, on every device. Moving a default is a later, separate step.

use infr_core::config::Config;

use super::{DeviceFacts, Tuning, MMV_MW_DTYPES};

/// How a parameter acts on the kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A number the host uses to choose a dispatch: a threshold, a split count, a band.
    Host,
    /// Selects which BUILD of a kernel runs (no new build appears; it picks among existing ones).
    Variant,
    /// A pinned subgroup width.
    Pin,
    /// Must equal a literal compiled into a shader: reported, not settable, held by a drift test.
    Fixed,
}

impl Kind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Kind::Host => "host",
            Kind::Variant => "variant",
            Kind::Pin => "pin",
            Kind::Fixed => "fixed",
        }
    }
}

/// A parameter's type and declared range.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Ty {
    /// An unsigned integer in `min..=max`. `max` may be spelled `max`.
    Uint { min: u64, max: u64 },
    /// `true`/`false` (also `1`/`0`, `yes`/`no`, `on`/`off`).
    Bool,
    /// One of a fixed set of integers.
    Set(&'static [u64]),
    /// A build name; `none` means no build.
    Text,
}

impl Ty {
    /// The declared range in words, for `infr tune list`.
    pub(crate) fn describe(self) -> String {
        match self {
            Ty::Uint { min, max } => format!("{min}..={}", show_uint(max)),
            Ty::Bool => "true|false".to_string(),
            Ty::Set(v) => v.iter().map(u64::to_string).collect::<Vec<_>>().join("|"),
            Ty::Text => "build name|none".to_string(),
        }
    }

    /// Parse a value as typed by a user or a row, and check it is in range.
    pub(crate) fn parse(self, raw: &str) -> Result<Value, String> {
        let raw = raw.trim();
        match self {
            Ty::Uint { min, max } => {
                let v = if raw.eq_ignore_ascii_case("max") {
                    max
                } else {
                    raw.parse::<u64>()
                        .map_err(|_| format!("expected an unsigned integer (got {raw:?})"))?
                };
                if v < min || v > max {
                    return Err(format!(
                        "{} is out of range {min}..={}",
                        show_uint(v),
                        show_uint(max)
                    ));
                }
                Ok(Value::Uint(v))
            }
            Ty::Bool => match raw {
                "1" | "true" | "yes" | "on" => Ok(Value::Bool(true)),
                "0" | "false" | "no" | "off" => Ok(Value::Bool(false)),
                other => Err(format!("expected true or false (got {other:?})")),
            },
            Ty::Set(set) => {
                let v = raw
                    .parse::<u64>()
                    .map_err(|_| format!("expected an unsigned integer (got {raw:?})"))?;
                if set.contains(&v) {
                    Ok(Value::Uint(v))
                } else {
                    Err(match set {
                        [a, b] => format!("must be {a} or {b} (got {v})"),
                        _ => format!("must be one of {} (got {v})", self.describe()),
                    })
                }
            }
            Ty::Text => {
                if raw.is_empty() {
                    Err("empty build name".to_string())
                } else {
                    Ok(Value::Text(raw.to_string()))
                }
            }
        }
    }

    /// Whether `v` is a value of this type in range (what a resolved value must satisfy).
    pub(crate) fn admits(self, v: &Value) -> bool {
        match (self, v) {
            (Ty::Uint { min, max }, Value::Uint(x)) => (min..=max).contains(x),
            (Ty::Set(set), Value::Uint(x)) => set.contains(x),
            (Ty::Bool, Value::Bool(_)) => true,
            (Ty::Text, Value::Text(s)) => !s.is_empty(),
            _ => false,
        }
    }
}

/// `u64::MAX` (and `usize::MAX`, the same on every target this builds for) prints as `max`.
fn show_uint(v: u64) -> String {
    if v == u64::MAX || v == usize::MAX as u64 {
        "max".to_string()
    } else {
        v.to_string()
    }
}

/// A parameter value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Value {
    Uint(u64),
    Bool(bool),
    Text(String),
}

impl Value {
    fn uint(&self) -> u64 {
        match self {
            Value::Uint(v) => *v,
            other => unreachable!("validated as an integer, got {other:?}"),
        }
    }
    fn flag(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            other => unreachable!("validated as a boolean, got {other:?}"),
        }
    }
    fn text(&self) -> &str {
        match self {
            Value::Text(s) => s,
            other => unreachable!("validated as text, got {other:?}"),
        }
    }
    fn u32(&self) -> u32 {
        u32::try_from(self.uint()).expect("range-checked to fit u32")
    }
    fn usize(&self) -> usize {
        usize::try_from(self.uint()).expect("range-checked to fit usize")
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Uint(v) => f.write_str(&show_uint(*v)),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Text(s) => f.write_str(s),
        }
    }
}

/// What a parameter is keyed on beyond the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Axis {
    /// One value per device.
    None,
    /// One value per quant format of [`MMV_MW_DTYPES`] (`family.param@q4_k`).
    Dtype,
}

impl Axis {
    /// The qualifier spellings, in storage order (empty for [`Axis::None`]).
    pub(crate) fn qualifiers(self) -> Vec<&'static str> {
        match self {
            Axis::None => Vec::new(),
            Axis::Dtype => MMV_MW_DTYPES.iter().map(|&(n, _)| n).collect(),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Axis::None => "-",
            Axis::Dtype => "dtype",
        }
    }
}

/// The qualifier index a parameter's accessors take: `None` on an unqualified parameter, the
/// position in [`Axis::qualifiers`] otherwise.
pub(crate) type Qual = Option<usize>;

/// One tunable parameter.
pub(crate) struct Param {
    pub family: &'static str,
    pub name: &'static str,
    pub kind: Kind,
    pub ty: Ty,
    pub axis: Axis,
    /// What it controls, one line.
    pub doc: &'static str,
    /// The generic default's rule, in words.
    pub rule: &'static str,
    /// The generic default on a device.
    pub default: fn(&DeviceFacts, Qual) -> Value,
    /// Read it back out of the typed set (the report and the round-trip test use this).
    pub get: fn(&Tuning, Qual) -> Value,
    /// Write it into the typed set. Never called for a [`Kind::Fixed`] parameter.
    pub set: fn(&mut Tuning, Qual, &Value),
}

impl Param {
    /// `family.param`.
    pub(crate) fn key(&self) -> String {
        format!("{}.{}", self.family, self.name)
    }
}

const fn uint(min: u64, max: u64) -> Ty {
    Ty::Uint { min, max }
}

const U32: u64 = u32::MAX as u64;
const USIZE: u64 = usize::MAX as u64;

fn fixed_set(_: &mut Tuning, _: Qual, _: &Value) {
    unreachable!("a fixed parameter is never set")
}

/// Every parameter. `infr tune list` prints them in this order.
pub(crate) static PARAMS: &[Param] = &[
    // ── device ───────────────────────────────────────────────────────────────
    Param {
        family: "device",
        name: "sg_pref",
        kind: Kind::Pin,
        ty: Ty::Set(&[16, 32]),
        axis: Axis::None,
        doc: "subgroup width the decode GEMV/reduction kernels that take one are pinned to",
        rule: "16 (or the minimum, if above 16) when the device can pin 16 or less, else 32; a \
               16 the device cannot pin falls back to 32",
        default: |f, _| {
            Value::Uint(if f.subgroup_min <= 16 {
                16u64.max(u64::from(f.subgroup_min))
            } else {
                32
            })
        },
        get: |t, _| Value::Uint(u64::from(t.sg_pref)),
        set: |t, _, v| t.sg_pref = v.u32(),
    },
    // ── decode GEMV ──────────────────────────────────────────────────────────
    Param {
        family: "gemv",
        name: "variant",
        kind: Kind::Variant,
        ty: Ty::Text,
        axis: Axis::None,
        doc: "multi-row decode GEMV build tried before the RM path (reg, sg, dbuf, wg128, ...)",
        rule: "reg",
        default: |_, _| Value::Text("reg".to_string()),
        get: |t, _| Value::Text(t.gemv_variant.clone().unwrap_or_else(|| "none".to_string())),
        set: |t, _, v| {
            t.gemv_variant = (v.text() != "none").then(|| v.text().to_string());
        },
    },
    Param {
        family: "gemv_rm",
        name: "enabled",
        kind: Kind::Variant,
        ty: Ty::Bool,
        axis: Axis::None,
        doc: "multi-output-row Q4_K/Q6_K decode GEMV (false: one row per workgroup)",
        rule: "true",
        default: |_, _| Value::Bool(true),
        get: |t, _| Value::Bool(t.gemv_rm_enabled),
        set: |t, _, v| t.gemv_rm_enabled = v.flag(),
    },
    Param {
        family: "gemv_rm",
        name: "rows",
        kind: Kind::Variant,
        ty: uint(0, U32),
        axis: Axis::None,
        doc: "output rows per workgroup of the RM GEMV; only 2 and 4 have builds, any other \
              value keeps one row per workgroup",
        rule: "2",
        default: |_, _| Value::Uint(2),
        get: |t, _| Value::Uint(u64::from(t.gemv_rm_rows)),
        set: |t, _, v| t.gemv_rm_rows = v.u32(),
    },
    Param {
        family: "gemv_rm",
        name: "min_out",
        kind: Kind::Host,
        ty: uint(0, USIZE),
        axis: Axis::None,
        doc: "smallest output width that takes the RM GEMV",
        rule: "2048",
        default: |_, _| Value::Uint(2048),
        get: |t, _| Value::Uint(t.gemv_rm_min_out as u64),
        set: |t, _, v| t.gemv_rm_min_out = v.usize(),
    },
    Param {
        family: "gemv_rm",
        name: "max_out",
        kind: Kind::Host,
        ty: uint(0, USIZE),
        axis: Axis::None,
        doc: "largest output width that takes the RM GEMV",
        rule: "max (no ceiling)",
        default: |_, _| Value::Uint(USIZE),
        get: |t, _| Value::Uint(t.gemv_rm_max_out as u64),
        set: |t, _, v| t.gemv_rm_max_out = v.usize(),
    },
    Param {
        family: "gemv_sg",
        name: "enabled",
        kind: Kind::Variant,
        ty: Ty::Bool,
        axis: Axis::None,
        doc: "subgroup decode GEMV for Q6_K projections in its band (false also turns off \
              gemv_id_sg)",
        rule: "true",
        default: |_, _| Value::Bool(true),
        get: |t, _| Value::Bool(t.gemv_sg_enabled),
        set: |t, _, v| t.gemv_sg_enabled = v.flag(),
    },
    Param {
        family: "gemv_sg",
        name: "rows",
        kind: Kind::Variant,
        ty: uint(0, U32),
        axis: Axis::None,
        doc: "output rows per workgroup of the subgroup GEMVs (dense and expert); only 2, 4 and \
              8 have builds, any other value keeps the tree kernel",
        rule: "2",
        default: |_, _| Value::Uint(2),
        get: |t, _| Value::Uint(u64::from(t.gemv_sg_rows)),
        set: |t, _, v| t.gemv_sg_rows = v.u32(),
    },
    Param {
        family: "gemv_sg",
        name: "min_out",
        kind: Kind::Host,
        ty: uint(0, USIZE),
        axis: Axis::None,
        doc: "smallest output width the subgroup GEMVs take",
        rule: "2048",
        default: |_, _| Value::Uint(2048),
        get: |t, _| Value::Uint(t.gemv_sg_min_out as u64),
        set: |t, _, v| t.gemv_sg_min_out = v.usize(),
    },
    Param {
        family: "gemv_sg",
        name: "max_out",
        kind: Kind::Host,
        ty: uint(0, USIZE),
        axis: Axis::None,
        doc: "largest output width the subgroup GEMVs take",
        rule: "8192",
        default: |_, _| Value::Uint(8192),
        get: |t, _| Value::Uint(t.gemv_sg_max_out as u64),
        set: |t, _, v| t.gemv_sg_max_out = v.usize(),
    },
    Param {
        family: "gemv_id_sg",
        name: "enabled",
        kind: Kind::Variant,
        ty: Ty::Bool,
        axis: Axis::None,
        doc: "subgroup expert (id-indexed) decode GEMV for Q6_K/Q5_K/IQ3_S, in gemv_sg's band",
        rule: "true",
        default: |_, _| Value::Bool(true),
        get: |t, _| Value::Bool(t.gemv_id_sg_enabled),
        set: |t, _, v| t.gemv_id_sg_enabled = v.flag(),
    },
    Param {
        family: "mmv_mw",
        name: "warps",
        kind: Kind::Variant,
        ty: uint(0, U32),
        axis: Axis::Dtype,
        doc: "warps per workgroup of the multi-warp int8 decode GEMV (native_mmv_mw), when the \
              decode does not take the unified rows=1 kernel",
        rule: "1 for q4_k, 8 for every other format",
        default: |_, q| Value::Uint(if q == Some(0) { 1 } else { 8 }),
        get: |t, q| Value::Uint(u64::from(t.mmv_mw_warps[q.expect("qualified")])),
        set: |t, q, v| t.mmv_mw_warps[q.expect("qualified")] = v.u32(),
    },
    Param {
        family: "mmv_mrow",
        name: "o4_max_in_f",
        kind: Kind::Variant,
        ty: uint(0, USIZE),
        axis: Axis::None,
        doc: "the int8 multi-row GEMV takes its four-outputs-per-workgroup build below this \
              input width",
        rule: "2048",
        default: |_, _| Value::Uint(2048),
        get: |t, _| Value::Uint(t.mmv_o4_max_in_f as u64),
        set: |t, _, v| t.mmv_o4_max_in_f = v.usize(),
    },
    Param {
        family: "mmv_mrow",
        name: "o4_all_widths",
        kind: Kind::Variant,
        ty: Ty::Bool,
        axis: Axis::None,
        doc: "the four-outputs build at every input width",
        rule: "true on an integrated GPU, else false",
        default: |f, _| Value::Bool(f.integrated),
        get: |t, _| Value::Bool(t.mmv_o4_all_widths),
        set: |t, _, v| t.mmv_o4_all_widths = v.flag(),
    },
    // ── attention ────────────────────────────────────────────────────────────
    Param {
        family: "attn_flash",
        name: "min_rows",
        kind: Kind::Host,
        ty: uint(0, USIZE),
        axis: Axis::None,
        doc: "fewest query rows that take flash attention (and the non-coopmat flash tier) at \
              8192+ keys; 64+ rows always do",
        rule: "24",
        default: |_, _| Value::Uint(24),
        get: |t, _| Value::Uint(t.flash_min_rows as u64),
        set: |t, _, v| t.flash_min_rows = v.usize(),
    },
    Param {
        family: "attn_flash",
        name: "force_bm32",
        kind: Kind::Variant,
        ty: Ty::Bool,
        axis: Axis::None,
        doc: "always take the 32-row flash tile (else 64 rows when shared memory allows and the \
              tile is full)",
        rule: "false",
        default: |_, _| Value::Bool(false),
        get: |t, _| Value::Bool(t.flash_force_bm32),
        set: |t, _, v| t.flash_force_bm32 = v.flag(),
    },
    Param {
        family: "attn_flash",
        name: "splits",
        kind: Kind::Host,
        ty: uint(0, U32),
        axis: Axis::None,
        doc: "flash-attention prefill split-K count; 0 = the policy below",
        rule: "0 (policy)",
        default: |_, _| Value::Uint(0),
        get: |t, _| Value::Uint(u64::from(t.flash.splits)),
        set: |t, _, v| t.flash.splits = v.u32(),
    },
    Param {
        family: "attn_flash",
        name: "full_wg",
        kind: Kind::Host,
        ty: uint(0, U32),
        axis: Axis::None,
        doc: "policy: no split once the plain grid has this many workgroups",
        rule: "1024",
        default: |_, _| Value::Uint(1024),
        get: |t, _| Value::Uint(u64::from(t.flash.full_wg)),
        set: |t, _, v| t.flash.full_wg = v.u32(),
    },
    Param {
        family: "attn_flash",
        name: "target_wg",
        kind: Kind::Host,
        ty: uint(0, U32),
        axis: Axis::None,
        doc: "policy: split to about this many workgroups",
        rule: "2048",
        default: |_, _| Value::Uint(2048),
        get: |t, _| Value::Uint(u64::from(t.flash.target_wg)),
        set: |t, _, v| t.flash.target_wg = v.u32(),
    },
    Param {
        family: "attn_flash",
        name: "max_splits",
        kind: Kind::Host,
        ty: uint(2, 8),
        axis: Axis::None,
        doc: "policy: most splits (the partials scratch is sized for 8)",
        rule: "8",
        default: |_, _| Value::Uint(8),
        get: |t, _| Value::Uint(u64::from(t.flash.max_splits)),
        set: |t, _, v| t.flash.max_splits = v.u32(),
    },
    Param {
        family: "attn_flash",
        name: "min_kv",
        kind: Kind::Host,
        ty: uint(0, U32),
        axis: Axis::None,
        doc: "policy: no split below this many keys",
        rule: "4096",
        default: |_, _| Value::Uint(4096),
        get: |t, _| Value::Uint(u64::from(t.flash.min_kv)),
        set: |t, _, v| t.flash.min_kv = v.u32(),
    },
    Param {
        family: "attn_flash",
        name: "kv_per_split",
        kind: Kind::Host,
        ty: uint(1, U32),
        axis: Axis::None,
        doc: "policy: at least this many keys per split",
        rule: "2048",
        default: |_, _| Value::Uint(2048),
        get: |t, _| Value::Uint(u64::from(t.flash.kv_per_split)),
        set: |t, _, v| t.flash.kv_per_split = v.u32(),
    },
    Param {
        family: "attn_pv",
        name: "splits",
        kind: Kind::Host,
        ty: uint(0, U32),
        axis: Axis::None,
        doc: "non-flash coopmat attention P·V split-K count; 0 = the policy below",
        rule: "0 (policy)",
        default: |_, _| Value::Uint(0),
        get: |t, _| Value::Uint(u64::from(t.pv.splits)),
        set: |t, _, v| t.pv.splits = v.u32(),
    },
    Param {
        family: "attn_pv",
        name: "full_wg",
        kind: Kind::Host,
        ty: uint(0, U32),
        axis: Axis::None,
        doc: "policy: no split once the plain grid has this many workgroups",
        rule: "1024",
        default: |_, _| Value::Uint(1024),
        get: |t, _| Value::Uint(u64::from(t.pv.full_wg)),
        set: |t, _, v| t.pv.full_wg = v.u32(),
    },
    Param {
        family: "attn_pv",
        name: "target_wg",
        kind: Kind::Host,
        ty: uint(0, U32),
        axis: Axis::None,
        doc: "policy: split to about this many workgroups",
        rule: "2048",
        default: |_, _| Value::Uint(2048),
        get: |t, _| Value::Uint(u64::from(t.pv.target_wg)),
        set: |t, _, v| t.pv.target_wg = v.u32(),
    },
    Param {
        family: "attn_pv",
        name: "max_splits",
        kind: Kind::Host,
        ty: uint(2, U32),
        axis: Axis::None,
        doc: "policy: most splits",
        rule: "8",
        default: |_, _| Value::Uint(8),
        get: |t, _| Value::Uint(u64::from(t.pv.max_splits)),
        set: |t, _, v| t.pv.max_splits = v.u32(),
    },
    Param {
        family: "attn_pv",
        name: "min_kv",
        kind: Kind::Host,
        ty: uint(0, U32),
        axis: Axis::None,
        doc: "policy: no split below this many (padded) keys",
        rule: "4096",
        default: |_, _| Value::Uint(4096),
        get: |t, _| Value::Uint(u64::from(t.pv.min_kv)),
        set: |t, _, v| t.pv.min_kv = v.u32(),
    },
    Param {
        family: "attn_pv",
        name: "kv_per_split",
        kind: Kind::Host,
        ty: uint(1, U32),
        axis: Axis::None,
        doc: "policy: at least this many keys per split",
        rule: "2048",
        default: |_, _| Value::Uint(2048),
        get: |t, _| Value::Uint(u64::from(t.pv.kv_per_split)),
        set: |t, _, v| t.pv.kv_per_split = v.u32(),
    },
    Param {
        family: "attn_canvas",
        name: "chunks",
        kind: Kind::Host,
        ty: uint(1, USIZE),
        axis: Axis::None,
        doc: "the canvas-mask (DiffusionGemma) split-K cuts its span into about this many chunks",
        rule: "3",
        default: |_, _| Value::Uint(3),
        get: |t, _| Value::Uint(t.canvas_chunks as u64),
        set: |t, _, v| t.canvas_chunks = v.usize(),
    },
    Param {
        family: "attn_split",
        name: "target_chunks",
        kind: Kind::Fixed,
        ty: uint(32, 32),
        axis: Axis::None,
        doc: "decode split-K attention: chunks per head (attn_live.comp computes the same policy)",
        rule: "32 (fixed: attn_live.comp)",
        default: |_, _| Value::Uint(crate::adapter::ATTN_SPLIT.target_chunks as u64),
        get: |_, _| Value::Uint(crate::adapter::ATTN_SPLIT.target_chunks as u64),
        set: fixed_set,
    },
    Param {
        family: "attn_split",
        name: "min_chunk",
        kind: Kind::Fixed,
        ty: uint(64, 64),
        axis: Axis::None,
        doc: "decode split-K attention: fewest keys per chunk",
        rule: "64 (fixed: attn_live.comp)",
        default: |_, _| Value::Uint(crate::adapter::ATTN_SPLIT.min_chunk as u64),
        get: |_, _| Value::Uint(crate::adapter::ATTN_SPLIT.min_chunk as u64),
        set: fixed_set,
    },
    Param {
        family: "attn_split",
        name: "max_chunk",
        kind: Kind::Fixed,
        ty: uint(512, 512),
        axis: Axis::None,
        doc: "decode split-K attention: most keys per chunk",
        rule: "512 (fixed: attn_live.comp)",
        default: |_, _| Value::Uint(crate::adapter::ATTN_SPLIT.max_chunk as u64),
        get: |_, _| Value::Uint(crate::adapter::ATTN_SPLIT.max_chunk as u64),
        set: fixed_set,
    },
    Param {
        family: "mla",
        name: "sg_width",
        kind: Kind::Fixed,
        ty: Ty::Set(&[32]),
        axis: Axis::None,
        doc: "subgroup width the MLA subgroup tier pins (mla.comp's MLA_SG_WIDTH)",
        rule: "32 (fixed: mla.comp)",
        default: |_, _| Value::Uint(u64::from(crate::gemm::MLA_SG_WIDTH)),
        get: |_, _| Value::Uint(u64::from(crate::gemm::MLA_SG_WIDTH)),
        set: fixed_set,
    },
    // ── MoE ──────────────────────────────────────────────────────────────────
    Param {
        family: "moe",
        name: "small_m",
        kind: Kind::Host,
        ty: uint(0, 64),
        axis: Axis::None,
        doc: "largest batch that takes the small-batch MoE path (per-row GEMVs); larger batches \
              take the batched expert GEMM. Capped at 64: past that one dispatch can outlast \
              the GPU hang watchdog",
        rule: "8",
        default: |_, _| Value::Uint(8),
        get: |t, _| Value::Uint(t.moe_small_m as u64),
        set: |t, _, v| t.moe_small_m = v.usize(),
    },
    // ── prefill GEMM ─────────────────────────────────────────────────────────
    Param {
        family: "split_k",
        name: "fixed_grid",
        kind: Kind::Host,
        ty: uint(0, USIZE),
        axis: Axis::None,
        doc: "coopmat prefill GEMM: split K when the plain tile grid is under this many \
              workgroups (the window's floor)",
        rule: "128",
        default: |_, _| Value::Uint(128),
        get: |t, _| Value::Uint(t.split_k_fixed_grid as u64),
        set: |t, _, v| t.split_k_fixed_grid = v.usize(),
    },
    Param {
        family: "split_k",
        name: "fill_per_cu",
        kind: Kind::Host,
        ty: uint(0, U32),
        axis: Axis::None,
        doc: "workgroups per compute unit the window widens to when scale_by_cu is on",
        rule: "4",
        default: |_, _| Value::Uint(4),
        get: |t, _| Value::Uint(t.split_k_fill_per_cu as u64),
        set: |t, _, v| t.split_k_fill_per_cu = v.usize(),
    },
    Param {
        family: "split_k",
        name: "scale_by_cu",
        kind: Kind::Host,
        ty: Ty::Bool,
        axis: Axis::None,
        doc: "widen the split-K window with the compute-unit count",
        rule: "false (the measured driver's row turns it on)",
        default: |_, _| Value::Bool(false),
        get: |t, _| Value::Bool(t.split_k_scale_by_cu),
        set: |t, _, v| t.split_k_scale_by_cu = v.flag(),
    },
    Param {
        family: "gemm_i8cm_kq",
        name: "enabled",
        kind: Kind::Variant,
        ty: Ty::Bool,
        axis: Axis::None,
        doc: "int8 cooperative-matrix K-quant prefill GEMM (still needs int8 coopmat and the \
              init-time layout check)",
        rule: "false (the measured driver's row turns it on)",
        default: |_, _| Value::Bool(false),
        get: |t, _| Value::Bool(t.i8cm_kq_enabled),
        set: |t, _, v| t.i8cm_kq_enabled = v.flag(),
    },
    Param {
        family: "gemm_i8cm_kq",
        name: "l_fill_per_cu",
        kind: Kind::Variant,
        ty: uint(0, U32),
        axis: Axis::None,
        doc: "the large tile once its grid has this many workgroups per compute unit",
        rule: "2",
        default: |_, _| Value::Uint(2),
        get: |t, _| Value::Uint(t.i8cm_kq_l_fill_per_cu as u64),
        set: |t, _, v| t.i8cm_kq_l_fill_per_cu = v.usize(),
    },
    // ── norms and quantization ───────────────────────────────────────────────
    Param {
        family: "rmsnorm",
        name: "wide",
        kind: Kind::Variant,
        ty: Ty::Bool,
        axis: Axis::None,
        doc: "decode RMSNorm (one row) on the 1024-wide build, where the device's workgroup \
              limits hold it",
        rule: "true",
        default: |_, _| Value::Bool(true),
        get: |t, _| Value::Bool(t.rmsnorm_wide),
        set: |t, _, v| t.rmsnorm_wide = v.flag(),
    },
    Param {
        family: "rmsnorm",
        name: "wide_min_dim",
        kind: Kind::Variant,
        ty: uint(0, USIZE),
        axis: Axis::None,
        doc: "smallest row width the wide RMSNorm build takes",
        rule: "2048",
        default: |_, _| Value::Uint(2048),
        get: |t, _| Value::Uint(t.rmsnorm_wide_min_dim as u64),
        set: |t, _, v| t.rmsnorm_wide_min_dim = v.usize(),
    },
    Param {
        family: "quant_q8",
        name: "blocks_per_wg",
        kind: Kind::Fixed,
        ty: uint(8, 8),
        axis: Axis::None,
        doc: "32-element blocks quant_q8 quantizes per workgroup (quant_q8.comp's \
              QUANT_Q8_BLOCKS_PER_WG)",
        rule: "8 (fixed: quant_q8.comp)",
        default: |_, _| Value::Uint(crate::recorder::QUANT_Q8_BLOCKS_PER_WG as u64),
        get: |_, _| Value::Uint(crate::recorder::QUANT_Q8_BLOCKS_PER_WG as u64),
        set: fixed_set,
    },
];

/// The parameter named `family.param`, if there is one.
pub(crate) fn find(key: &str) -> Option<&'static Param> {
    PARAMS.iter().find(|p| p.key() == key)
}

/// A setting's key split into its parameter and qualifier index.
pub(crate) fn lookup(key: &str) -> Result<(&'static Param, Qual), String> {
    let (name, qual) = match key.split_once('@') {
        Some((n, q)) => (n, Some(q)),
        None => (key, None),
    };
    let p = find(name).ok_or_else(|| {
        format!("`{name}` is not a tuning parameter (`infr tune list` names every one)")
    })?;
    let q = match (p.axis, qual) {
        (_, None) => None,
        (Axis::None, Some(q)) => {
            return Err(format!("`{name}` takes no qualifier (got `@{q}`)"));
        }
        (axis, Some(q)) => Some(axis.qualifiers().iter().position(|&n| n == q).ok_or_else(
            || {
                format!(
                    "`{name}` has no qualifier `@{q}` (one of: {})",
                    axis.qualifiers().join(", ")
                )
            },
        )?),
    };
    Ok((p, q))
}

/// The parameter writes an alias makes: `(parameter key, value)`.
pub(crate) type Writes = Vec<(&'static str, Value)>;

/// An older config knob that is now an alias of one or more tuning parameters.
pub(crate) struct Alias {
    /// Its config path (`infr_core::config::tune::TUNE_ALIASES` lists the same paths).
    pub path: &'static str,
    /// The parameter(s) it writes, and the values, for the knob's current setting in `cfg` —
    /// exactly the value the old read site computed from it (its clamp, its fallback). `None`
    /// when the setting has no effect on the parameter (unset, or a value the old site treated
    /// as "use the default").
    pub read: fn(&Config, bool) -> Option<Writes>,
}

fn one(key: &'static str, v: Value) -> Option<Writes> {
    Some(vec![(key, v)])
}

/// A split-count knob (`Option<usize>`): a value that fits a `u32` and is not 0 is the count;
/// 0 or a value past `u32` meant "the adaptive policy" at the old read site, which is 0 here.
fn splits_alias(v: Option<usize>, key: &'static str) -> Option<Writes> {
    let v = v?;
    let n = u32::try_from(v).ok().filter(|&n| n > 0).unwrap_or(0);
    one(key, Value::Uint(u64::from(n)))
}

/// Every alias. The `bool` passed to `read` says whether a config layer recorded setting the
/// knob, so a knob set to its own default still counts as set (and keeps its level).
pub(crate) static ALIASES: &[Alias] = &[
    Alias {
        path: "device.subgroup_pref",
        read: |c, _| {
            c.device
                .subgroup_pref
                .and_then(|v| one("device.sg_pref", Value::Uint(u64::from(v))))
        },
    },
    Alias {
        path: "kernels.vulkan.gemv.no_rm",
        read: |c, _| {
            c.kernels
                .vulkan
                .gemv
                .no_rm
                .then(|| vec![("gemv_rm.enabled", Value::Bool(false))])
        },
    },
    Alias {
        path: "kernels.vulkan.gemv.rm",
        read: |c, set| {
            let v = c.kernels.vulkan.gemv.rm;
            (set || v != 2).then(|| vec![("gemv_rm.rows", Value::Uint(u64::from(v)))])
        },
    },
    Alias {
        path: "kernels.vulkan.gemv.rm_maxout",
        read: |c, set| {
            let v = c.kernels.vulkan.gemv.rm_maxout;
            (set || v != usize::MAX).then(|| vec![("gemv_rm.max_out", Value::Uint(v as u64))])
        },
    },
    Alias {
        path: "kernels.vulkan.gemv.rm_minout",
        read: |c, set| {
            let v = c.kernels.vulkan.gemv.rm_minout;
            (set || v != 2048).then(|| vec![("gemv_rm.min_out", Value::Uint(v as u64))])
        },
    },
    Alias {
        path: "kernels.vulkan.gemv.no_sg",
        read: |c, _| {
            c.kernels
                .vulkan
                .gemv
                .no_sg
                .then(|| vec![("gemv_sg.enabled", Value::Bool(false))])
        },
    },
    Alias {
        path: "kernels.vulkan.gemv.no_id_sg",
        read: |c, _| {
            c.kernels
                .vulkan
                .gemv
                .no_id_sg
                .then(|| vec![("gemv_id_sg.enabled", Value::Bool(false))])
        },
    },
    Alias {
        path: "kernels.vulkan.gemv.sg_minout",
        read: |c, set| {
            let v = c.kernels.vulkan.gemv.sg_minout;
            (set || v != 2048).then(|| vec![("gemv_sg.min_out", Value::Uint(v as u64))])
        },
    },
    Alias {
        path: "kernels.vulkan.gemv.sg_maxout",
        read: |c, set| {
            let v = c.kernels.vulkan.gemv.sg_maxout;
            (set || v != 8192).then(|| vec![("gemv_sg.max_out", Value::Uint(v as u64))])
        },
    },
    Alias {
        path: "kernels.vulkan.gemv.sg_nr",
        read: |c, set| {
            let v = c.kernels.vulkan.gemv.sg_nr;
            (set || v != 2).then(|| vec![("gemv_sg.rows", Value::Uint(u64::from(v)))])
        },
    },
    Alias {
        path: "kernels.vulkan.gemv.variant",
        read: |c, set| {
            let v = c.kernels.vulkan.gemv.variant.as_deref();
            (set || v != Some("reg"))
                .then(|| vec![("gemv.variant", Value::Text(v.unwrap_or("none").to_string()))])
        },
    },
    Alias {
        path: "kernels.vulkan.i8cm_kq",
        // Only OFF ever meant anything: `true` (the default) left the driver gate in charge.
        read: |c, _| {
            (!c.kernels.vulkan.i8cm_kq).then(|| vec![("gemm_i8cm_kq.enabled", Value::Bool(false))])
        },
    },
    Alias {
        path: "kernels.vulkan.mmv_mw_warps",
        // Every format at once; a value past `u32` fell back to the per-format default.
        read: |c, _| {
            let w = u32::try_from(c.kernels.vulkan.mmv_mw_warps?).ok()?;
            one("mmv_mw.warps", Value::Uint(u64::from(w)))
        },
    },
    Alias {
        path: "kernels.vulkan.moe_small_m",
        // The old site clamped to 0..=64 (`tier::EnvRows`); the parameter's range is that clamp.
        read: |c, set| {
            let v = c.kernels.vulkan.moe_small_m;
            (set || v != 8).then(|| vec![("moe.small_m", Value::Uint(v.min(64) as u64))])
        },
    },
    Alias {
        path: "kernels.vulkan.canvas_chunk_n",
        // The old site floored at 1 (`tier::EnvRows`).
        read: |c, set| {
            let v = c.kernels.vulkan.canvas_chunk_n;
            (set || v != 3).then(|| vec![("attn_canvas.chunks", Value::Uint(v.max(1) as u64))])
        },
    },
    Alias {
        path: "kernels.vulkan.flash_splits",
        read: |c, _| splits_alias(c.kernels.vulkan.flash_splits, "attn_flash.splits"),
    },
    Alias {
        path: "kernels.vulkan.flash_bm32",
        read: |c, set| {
            let v = c.kernels.vulkan.flash_bm32;
            (set || v).then(|| vec![("attn_flash.force_bm32", Value::Bool(v))])
        },
    },
    Alias {
        path: "kernels.vulkan.flash_min_rows",
        read: |c, set| {
            let v = c.kernels.vulkan.flash_min_rows;
            (set || v != 24).then(|| vec![("attn_flash.min_rows", Value::Uint(v as u64))])
        },
    },
    Alias {
        path: "kernels.vulkan.pv_splits",
        read: |c, _| splits_alias(c.kernels.vulkan.pv_splits, "attn_pv.splits"),
    },
];

/// The `INFR_*` name(s) of the alias at `path`, for a source label.
pub(crate) fn alias_env_names(path: &str) -> String {
    infr_core::config::manifest::KEYS
        .iter()
        .filter(|k| k.path == path)
        .map(|k| k.env)
        .collect::<Vec<_>>()
        .join("/")
}

/// The qualifier spelled for index `q` of `p`'s axis (`None` for an unqualified slot).
pub(crate) fn qual_name(p: &Param, q: Qual) -> Option<&'static str> {
    q.map(|i| p.axis.qualifiers()[i])
}

/// The slots a parameter resolves: one for an unqualified parameter, one per qualifier otherwise.
pub(crate) fn slots(p: &Param) -> Vec<Qual> {
    match p.axis {
        Axis::None => vec![None],
        axis => (0..axis.qualifiers().len()).map(Some).collect(),
    }
}

/// The `DType` a `mmv_mw` qualifier index stands for.
#[cfg(test)]
pub(crate) fn dtype_of(q: usize) -> infr_core::DType {
    MMV_MW_DTYPES[q].1
}
