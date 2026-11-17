//! The row schema shared by the built-in table (`table.toml`) and a user's tuning file, and the
//! parser for both.
//!
//! ```toml
//! [[row]]
//! id = "amd-proprietary-split-k"
//! match = { vendor = "amd", driver = "AMD_PROPRIETARY" }
//! verified = "maintainer"            # or "contributor"
//! source = { by = "…", device = "…", driver = "…", date = "…", command = "…", evidence = "…" }
//! params = { "split_k.scale_by_cu" = true }
//! ```
//!
//! `match` keys: `vendor` (`amd`/`intel`/`nvidia`/`other`), `arch` (a `DeviceArch` name such as
//! `AmdRdna3`), `driver` (a Vulkan driver id name such as `AMD_PROPRIETARY` or `MESA_RADV`;
//! `unreported` matches a driver that reports none), `integrated`, `device_id` (one PCI id or a
//! list, decimal or `"0x…"`), `name` (a case-insensitive substring of the device name;
//! discouraged), `driver_version_min`/`driver_version_max` (raw `driverVersion`),
//! `subgroup_min`/`subgroup_max` (the exact reported range), `cu_min`/`cu_max` (compute units).
//! A row matching `device_id` or `name` is a DEVICE row; any other is a CLASS row.

use super::DeviceFacts;

/// Who stands behind a row's values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Verified {
    /// Measured by the project's maintainers on the hardware the row names.
    Maintainer,
    /// Contributed from hardware the maintainers do not have; logged when in use.
    Contributor,
}

/// What a row matches. Every field present must hold.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Match {
    pub vendor: Option<String>,
    pub arch: Option<String>,
    pub driver: Option<String>,
    pub integrated: Option<bool>,
    pub device_id: Option<Vec<u32>>,
    pub name: Option<String>,
    pub driver_version_min: Option<u32>,
    pub driver_version_max: Option<u32>,
    pub subgroup_min: Option<u32>,
    pub subgroup_max: Option<u32>,
    pub cu_min: Option<u32>,
    pub cu_max: Option<u32>,
}

impl Match {
    /// How many key fields the row names (its specificity within a level).
    pub(crate) fn fields(&self) -> usize {
        [
            self.vendor.is_some(),
            self.arch.is_some(),
            self.driver.is_some(),
            self.integrated.is_some(),
            self.device_id.is_some(),
            self.name.is_some(),
            self.driver_version_min.is_some(),
            self.driver_version_max.is_some(),
            self.subgroup_min.is_some(),
            self.subgroup_max.is_some(),
            self.cu_min.is_some(),
            self.cu_max.is_some(),
        ]
        .iter()
        .filter(|&&b| b)
        .count()
    }

    /// A device row (PCI ids or a name) rather than a class row.
    pub(crate) fn is_device(&self) -> bool {
        self.device_id.is_some() || self.name.is_some()
    }

    pub(crate) fn matches(&self, f: &DeviceFacts) -> bool {
        let driver = f.driver.as_deref().unwrap_or("unreported");
        self.vendor.as_deref().is_none_or(|v| v == f.vendor())
            && self.arch.as_deref().is_none_or(|a| a == f.arch)
            && self.driver.as_deref().is_none_or(|d| d == driver)
            && self.integrated.is_none_or(|i| i == f.integrated)
            && self
                .device_id
                .as_ref()
                .is_none_or(|ids| ids.contains(&f.device_id))
            && self
                .name
                .as_deref()
                .is_none_or(|n| f.name.to_lowercase().contains(&n.to_lowercase()))
            && self
                .driver_version_min
                .is_none_or(|v| f.driver_version >= v)
            && self
                .driver_version_max
                .is_none_or(|v| f.driver_version <= v)
            && self.subgroup_min.is_none_or(|v| f.subgroup_min == v)
            && self.subgroup_max.is_none_or(|v| f.subgroup_max == v)
            && self.cu_min.is_none_or(|v| f.compute_units >= v)
            && self.cu_max.is_none_or(|v| f.compute_units <= v)
    }
}

/// One table row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Row {
    pub id: String,
    pub matcher: Match,
    pub verified: Verified,
    /// `source`'s fields, in file order (`by`, `device`, `driver`, `date`, `command`, `evidence`).
    pub source: Vec<(String, String)>,
    /// `family.param[@qualifier]` → the value as written, in file order.
    pub params: Vec<(String, String)>,
}

/// The fields a row's `source` must carry.
pub(crate) const SOURCE_FIELDS: &[&str] =
    &["by", "device", "driver", "date", "command", "evidence"];

/// The built-in table, as compiled in.
pub(crate) const BUILTIN: &str = include_str!("table.toml");

/// Parse a table (`label` names it in errors). Structure errors are errors here; whether each
/// parameter exists and its value is in range is the resolver's call, because a bad value in a
/// built-in row is dropped with a warning while one in a user file is fatal.
pub(crate) fn parse(text: &str, label: &str) -> Result<Vec<Row>, String> {
    let doc: toml::Table = toml::from_str(text).map_err(|e| format!("{label}: {}", e.message()))?;
    for key in doc.keys() {
        if key != "row" {
            return Err(format!(
                "{label}: unknown top-level key `{key}` (rows are `[[row]]`)"
            ));
        }
    }
    let Some(rows) = doc.get("row") else {
        return Ok(Vec::new());
    };
    let rows = rows
        .as_array()
        .ok_or_else(|| format!("{label}: `row` must be an array of tables (`[[row]]`)"))?;
    rows.iter()
        .enumerate()
        .map(|(i, r)| parse_row(r, &format!("{label} row {}", i + 1)))
        .collect()
}

fn parse_row(v: &toml::Value, label: &str) -> Result<Row, String> {
    let t = v
        .as_table()
        .ok_or_else(|| format!("{label}: a row must be a table"))?;
    for key in t.keys() {
        if !matches!(
            key.as_str(),
            "id" | "match" | "verified" | "source" | "params"
        ) {
            return Err(format!("{label}: unknown key `{key}`"));
        }
    }
    let id = t
        .get("id")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| format!("{label}: missing string `id`"))?
        .to_string();
    let label = format!("{label} (`{id}`)");
    let matcher = match t.get("match") {
        Some(m) => parse_match(
            m.as_table()
                .ok_or_else(|| format!("{label}: `match` must be a table"))?,
            &label,
        )?,
        None => Match::default(),
    };
    let verified = match t.get("verified").map(|v| v.as_str()) {
        None => Verified::Contributor,
        Some(Some("maintainer")) => Verified::Maintainer,
        Some(Some("contributor")) => Verified::Contributor,
        Some(_) => {
            return Err(format!(
                "{label}: `verified` must be \"maintainer\" or \"contributor\""
            ))
        }
    };
    let source = match t.get("source") {
        None => Vec::new(),
        Some(s) => s
            .as_table()
            .ok_or_else(|| format!("{label}: `source` must be a table"))?
            .iter()
            .map(|(k, v)| {
                v.as_str()
                    .map(|s| (k.clone(), s.to_string()))
                    .ok_or_else(|| format!("{label}: `source.{k}` must be a string"))
            })
            .collect::<Result<_, _>>()?,
    };
    let params = match t.get("params") {
        None => Vec::new(),
        Some(p) => p
            .as_table()
            .ok_or_else(|| format!("{label}: `params` must be a table"))?
            .iter()
            .map(|(k, v)| {
                scalar(v).map(|s| (k.clone(), s)).ok_or_else(|| {
                    format!("{label}: `params.\"{k}\"` must be a number, boolean or string")
                })
            })
            .collect::<Result<_, _>>()?,
    };
    Ok(Row {
        id,
        matcher,
        verified,
        source,
        params,
    })
}

fn scalar(v: &toml::Value) -> Option<String> {
    Some(match v {
        toml::Value::String(s) => s.clone(),
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::Boolean(b) => b.to_string(),
        _ => return None,
    })
}

fn parse_match(t: &toml::Table, label: &str) -> Result<Match, String> {
    let mut m = Match::default();
    let text = |k: &str, v: &toml::Value| {
        v.as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("{label}: `match.{k}` must be a string"))
    };
    let num = |k: &str, v: &toml::Value| -> Result<u32, String> {
        let n = match v {
            toml::Value::Integer(i) => u32::try_from(*i).ok(),
            toml::Value::String(s) => match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                Some(hex) => u32::from_str_radix(hex, 16).ok(),
                None => s.parse().ok(),
            },
            _ => None,
        };
        n.ok_or_else(|| format!("{label}: `match.{k}` must be an unsigned 32-bit number"))
    };
    for (k, v) in t {
        match k.as_str() {
            "vendor" => {
                let s = text(k, v)?;
                if !matches!(s.as_str(), "amd" | "intel" | "nvidia" | "other") {
                    return Err(format!(
                        "{label}: `match.vendor` must be amd, intel, nvidia or other (got {s:?})"
                    ));
                }
                m.vendor = Some(s);
            }
            "arch" => {
                let s = text(k, v)?;
                if super::arch_from_name(&s).is_none() {
                    return Err(format!(
                        "{label}: `match.arch` {s:?} is not an architecture name"
                    ));
                }
                m.arch = Some(s);
            }
            "driver" => {
                let d = text(k, v)?;
                if d != "unreported" && !driver_names().contains(&d) {
                    return Err(format!(
                        "{label}: `match.driver` {d:?} is not a Vulkan driver id name (one of: \
                         unreported, {})",
                        driver_names().join(", ")
                    ));
                }
                m.driver = Some(d);
            }
            "integrated" => {
                m.integrated = Some(
                    v.as_bool()
                        .ok_or_else(|| format!("{label}: `match.integrated` must be a boolean"))?,
                );
            }
            "device_id" => {
                m.device_id = Some(match v {
                    toml::Value::Array(a) => {
                        a.iter().map(|x| num(k, x)).collect::<Result<_, _>>()?
                    }
                    one => vec![num(k, one)?],
                });
            }
            "name" => m.name = Some(text(k, v)?),
            "driver_version_min" => m.driver_version_min = Some(num(k, v)?),
            "driver_version_max" => m.driver_version_max = Some(num(k, v)?),
            "subgroup_min" => m.subgroup_min = Some(num(k, v)?),
            "subgroup_max" => m.subgroup_max = Some(num(k, v)?),
            "cu_min" => m.cu_min = Some(num(k, v)?),
            "cu_max" => m.cu_max = Some(num(k, v)?),
            other => return Err(format!("{label}: unknown match key `{other}`")),
        }
    }
    Ok(m)
}

/// Every `VkDriverId` name ash knows (`AMD_PROPRIETARY`, `MESA_RADV`, …), as a row's
/// `match.driver` and the device facts spell it.
pub(crate) fn driver_names() -> &'static [String] {
    static NAMES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    NAMES.get_or_init(|| {
        // The registry numbers driver ids from 1 and has fewer than 64; an id ash does not know
        // prints as its number, which the filter drops.
        (1..64)
            .map(|i| super::driver_name(ash::vk::DriverId::from_raw(i)))
            .filter(|n| n.starts_with(|c: char| c.is_ascii_alphabetic()))
            .collect()
    })
}

impl Match {
    /// Whether one device could satisfy both `self` and `other` (each key both name must be
    /// satisfiable at once; a name pattern is assumed to be).
    pub(crate) fn compatible(&self, other: &Match) -> bool {
        fn eq<T: PartialEq>(a: &Option<T>, b: &Option<T>) -> bool {
            match (a, b) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            }
        }
        fn overlap(lo: [Option<u32>; 2], hi: [Option<u32>; 2]) -> bool {
            let lo = lo.iter().flatten().max().copied().unwrap_or(0);
            let hi = hi.iter().flatten().min().copied().unwrap_or(u32::MAX);
            lo <= hi
        }
        let ids = match (&self.device_id, &other.device_id) {
            (Some(a), Some(b)) => a.iter().any(|x| b.contains(x)),
            _ => true,
        };
        eq(&self.vendor, &other.vendor)
            && eq(&self.arch, &other.arch)
            && eq(&self.driver, &other.driver)
            && eq(&self.integrated, &other.integrated)
            && eq(&self.subgroup_min, &other.subgroup_min)
            && eq(&self.subgroup_max, &other.subgroup_max)
            && ids
            && overlap(
                [self.driver_version_min, other.driver_version_min],
                [self.driver_version_max, other.driver_version_max],
            )
            && overlap([self.cu_min, other.cu_min], [self.cu_max, other.cu_max])
    }
}

/// Pairs of rows that tie — same level (class or device) and the same number of match fields —
/// that one device could match both of, and that set the same parameter key to different values:
/// which one wins would be file order, not intent. One line per conflict.
pub(crate) fn conflicts(rows: &[Row]) -> Vec<String> {
    let mut out = Vec::new();
    for (i, a) in rows.iter().enumerate() {
        for b in &rows[i + 1..] {
            if a.matcher.is_device() != b.matcher.is_device()
                || a.matcher.fields() != b.matcher.fields()
                || !a.matcher.compatible(&b.matcher)
            {
                continue;
            }
            for (k, va) in &a.params {
                if let Some((_, vb)) = b.params.iter().find(|(kb, _)| kb == k) {
                    if va != vb {
                        out.push(format!(
                            "rows `{}` and `{}` can match the same device with equal \
                             specificity and set `{k}` to {va} and {vb}",
                            a.id, b.id
                        ));
                    }
                }
            }
        }
    }
    out
}

/// The built-in rows. A parse failure is a build defect the table tests catch, so this panics.
pub(crate) fn builtin() -> &'static [Row] {
    static ROWS: std::sync::OnceLock<Vec<Row>> = std::sync::OnceLock::new();
    ROWS.get_or_init(|| parse(BUILTIN, "built-in tuning table").expect("table.toml parses"))
}
