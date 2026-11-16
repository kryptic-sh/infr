//! What `infr tune show` and `infr tune list` print, and the short hash of a resolved set that
//! the device banner carries.

use super::DeviceFacts;

/// One matched table row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowReport {
    /// The row's id.
    pub id: String,
    /// `class`, `device` or `file` (the user's tuning file).
    pub level: &'static str,
    /// `maintainer` or `contributor`.
    pub verified: &'static str,
    /// The parameter keys the row set (a key that failed validation is not listed).
    pub params: Vec<String>,
}

/// One resolved parameter (one per qualifier for a qualified parameter).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamReport {
    /// `family.param`, or `family.param@qualifier`.
    pub key: String,
    /// The value, as a setting would spell it.
    pub value: String,
    /// The level it came from: `default`, `row <id>`, `device row <id>`, `tuning file … row
    /// <id>`, or the config / environment / command-line setting (an alias names the old knob).
    pub source: String,
    /// `host`, `variant`, `pin` or `fixed`.
    pub kind: &'static str,
    /// The declared range or set.
    pub range: String,
    /// The qualifier axis (`dtype`), or `-`.
    pub axis: &'static str,
    /// The generic default's rule, in words.
    pub rule: &'static str,
    /// What the parameter controls.
    pub doc: &'static str,
}

/// A resolved parameter set, as the tools print it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuningReport {
    /// The device it was resolved for.
    pub device: DeviceFacts,
    /// The rows that matched, in table order.
    pub rows: Vec<RowReport>,
    /// Every parameter.
    pub params: Vec<ParamReport>,
    /// The built-in table's rows were switched off (`kernels.vulkan.tuning.rows = false`).
    pub rows_off: bool,
}

/// What switching the table's rows off means, for `show` and `list`.
pub const ROWS_OFF_NOTE: &str = "the built-in table's rows are off (`kernels.vulkan.tuning.rows = false` / `INFR_TUNE_NO_ROWS`): every value is its generic default. On AMD's proprietary driver that also turns off the int8 cooperative-matrix K-quant prefill GEMM and the compute-unit-scaled split-K window those rows enable, so prefill takes the f16 coopmat GEMM (about a third slower pp512 on the RX 7900 XTX).";

impl TuningReport {
    pub(crate) fn new(
        device: &DeviceFacts,
        rows: Vec<RowReport>,
        params: Vec<ParamReport>,
        rows_off: bool,
    ) -> Self {
        Self {
            device: device.clone(),
            rows,
            params,
            rows_off,
        }
    }

    /// The facts a row can match on, as `key=value` pairs.
    pub fn match_key(&self) -> String {
        let d = &self.device;
        format!(
            "vendor={} arch={} driver={} integrated={} device_id={:#06x} cu={} subgroup={}-{} \
             driver_version={:#x}",
            d.vendor(),
            d.arch,
            d.driver.as_deref().unwrap_or("unreported"),
            d.integrated,
            d.device_id,
            d.compute_units,
            d.subgroup_min,
            d.subgroup_max,
            d.driver_version,
        )
    }

    /// A short hash of the resolved values (every key and value, in table order): equal sets hash
    /// equal on every machine, so a bug report's banner says whether two runs tuned alike. The
    /// first 8 hex digits of FNV-1a 64.
    pub fn hash(&self) -> String {
        let mut text = String::new();
        for p in &self.params {
            text.push_str(&p.key);
            text.push('=');
            text.push_str(&p.value);
            text.push('\n');
        }
        let h = infr_core::kernel_cache::fnv1a(text.as_bytes());
        format!("{:08x}", h >> 32)
    }

    /// `infr tune show`'s text: the device, the match key, the matched rows, then one line per
    /// parameter with its value and the level it came from.
    pub fn show(&self) -> String {
        let d = &self.device;
        let mut out = format!(
            "device:  {} ({}, {}{}{})\nmatch:   {}\nhash:    {}\nrows:",
            d.name,
            d.arch,
            d.driver.as_deref().unwrap_or("driver unreported"),
            if d.integrated { ", integrated" } else { "" },
            if d.wave64 { ", wave64 emulation" } else { "" },
            self.match_key(),
            self.hash(),
        );
        if self.rows_off {
            out.push_str(&format!(" off — {ROWS_OFF_NOTE}\n"));
        } else if self.rows.is_empty() {
            out.push_str(" none\n");
        } else {
            out.push('\n');
            for r in &self.rows {
                out.push_str(&format!(
                    "  {} [{} row, {}] sets {}\n",
                    r.id,
                    r.level,
                    r.verified,
                    if r.params.is_empty() {
                        "nothing".to_string()
                    } else {
                        r.params.join(", ")
                    }
                ));
            }
        }
        out.push('\n');
        let w = self.params.iter().map(|p| p.key.len()).max().unwrap_or(0);
        let wv = self.params.iter().map(|p| p.value.len()).max().unwrap_or(0);
        for p in &self.params {
            out.push_str(&format!("{:<w$}  {:<wv$}  {}\n", p.key, p.value, p.source));
        }
        out
    }

    /// `infr tune list`'s text: every parameter (of `family`, if given) with its kind, declared
    /// range, qualifier axis, default rule, what it controls, and its value on this device.
    pub fn list(&self, family: Option<&str>) -> String {
        let mut out = String::from(
            "Every tunable Vulkan kernel parameter, with its value on this device.\n\
             Set one with `--set kernels.vulkan.tune.<key>=<value>`, `INFR_TUNE=<key>=<value>,...`\n\
             or `[kernels.vulkan.tune]` in infr.toml; `infr tune show` says where each value \
             came from.\n\
             kind: host = a dispatch threshold or count, variant = picks a kernel build, pin = a \
             subgroup width, fixed = equals a shader literal (reported, not settable).\n\
             `kernels.vulkan.tuning.rows = false` (`INFR_TUNE_NO_ROWS`) drops the built-in table's \
             rows and leaves the generic defaults; on AMD's proprietary driver that turns off the \
             int8 cooperative-matrix prefill GEMM and the CU-scaled split-K window (slower \
             prefill).\n\n",
        );
        if self.rows_off {
            out.push_str(&format!("Note: {ROWS_OFF_NOTE}\n\n"));
        }
        let mut last_family = "";
        for p in self
            .params
            .iter()
            .filter(|p| family.is_none_or(|f| p.key.split('.').next() == Some(f)))
        {
            let fam = p.key.split('.').next().unwrap_or("");
            if fam != last_family {
                out.push_str(&format!("[{fam}]\n"));
                last_family = fam;
            }
            let base = p.key.split('@').next().unwrap_or(&p.key);
            // A qualified parameter prints its description once, then one value per qualifier.
            let first = !p.key.contains('@')
                || self
                    .params
                    .iter()
                    .find(|q| q.key.split('@').next() == Some(base))
                    .is_some_and(|q| q.key == p.key);
            if first {
                out.push_str(&format!(
                    "  {base}  ({}, {}{})\n      {}\n      default: {}\n",
                    p.kind,
                    p.range,
                    if p.axis == "-" {
                        String::new()
                    } else {
                        format!(", per {}", p.axis)
                    },
                    p.doc,
                    p.rule,
                ));
            }
            out.push_str(&format!("      here: {} = {}\n", p.key, p.value));
        }
        out
    }
}
