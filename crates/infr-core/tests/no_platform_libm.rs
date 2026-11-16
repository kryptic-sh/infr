//! **No platform math in production code.** `f32::exp`, `ln`, `tanh`, `powf`, `sin`, `cos` and
//! their relatives lower to the C runtime, whose last bits differ between operating systems,
//! C library versions and architectures — so any output computed through one differs too. The
//! workspace uses `infr_core::math` (correctly rounded, the same bits everywhere) instead; see
//! `docs/math.md` and `docs/plan/11-cpu-bit-identical.md`.
//!
//! What counts as a call: a method `.name(` or a path `f32::name` / `f64::name` for every name
//! in [`FORBIDDEN`], in the production source of every workspace crate (`crates/*/src`, minus
//! `#[cfg(test)]` items and the files of `#[cfg(test)] mod x;` declarations). Comments and
//! string literals are not code and are not scanned.
//!
//! `powi`, `sqrt`, `abs`, `floor`, `round`, `mul_add` and friends are NOT forbidden: they are
//! exact (`sqrt` is correctly rounded by IEEE 754, `mul_add` is a single rounding everywhere) or,
//! for `powi`, a pure-Rust compiler builtin (`__powisf2`/`__powidf2` in compiler-builtins,
//! repeated squaring in the operand's own type) that is the same code on every target.
//!
//! **The allowlist.** A call that has not moved yet is marked at the site with a
//! `// libm-ok(S2): <reason>` comment (on its line or in the comment block directly above), and
//! its file is listed in [`ALLOWLIST`]. A marker outside a listed file, and a listed file with no
//! marked call left, both fail — so the list only shrinks, and slice S2 of the plan empties it.

use std::path::{Path, PathBuf};

/// The platform math functions (`f32`/`f64` methods that lower to the C runtime).
const FORBIDDEN: &[&str] = &[
    "exp", "exp2", "exp_m1", "ln", "ln_1p", "log", "log2", "log10", "powf", "sin", "cos", "tan",
    "sin_cos", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "cbrt", "hypot",
];

const MARKER: &str = "libm-ok(S2):";

/// Files allowed to carry `libm-ok(S2)` markers, crate-relative. Every entry is a file whose
/// platform-math calls have not been migrated yet; S2 removes them all.
const ALLOWLIST: &[&str] = &[
    "infr-core/src/decode_spec.rs",
    "infr-core/src/pager/shapes.rs",
    "infr-cpu/src/kernels.rs",
    "infr-cpu/src/lib.rs",
    "infr-llama/src/config.rs",
    "infr-llama/src/diffusion.rs",
    "infr-llama/src/mtp/mod.rs",
    "infr-llama/src/sampling.rs",
    "infr-llama/src/seam/runner.rs",
    "infr-llama/src/seam/sc.rs",
    "infr-metal/src/exec.rs",
];

#[test]
fn no_platform_libm_outside_the_allowlist() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .to_path_buf();
    let mut files = Vec::new();
    collect_rs(&crates, &mut files);
    let parsed: Vec<(PathBuf, Source)> = files
        .iter()
        .filter(|p| p.components().any(|c| c.as_os_str() == "src"))
        .map(|p| {
            (
                p.clone(),
                Source::parse(&std::fs::read_to_string(p).unwrap()),
            )
        })
        .collect();
    // Files that are test-only because a `#[cfg(test)] mod x;` declares them.
    let mut test_files = std::collections::HashSet::new();
    for (path, src) in &parsed {
        for m in &src.test_mods {
            for c in module_files(path, m) {
                test_files.insert(c);
            }
        }
    }
    let mut offenders = Vec::new();
    let mut marked_files: std::collections::BTreeMap<String, usize> = Default::default();
    let mut scanned = 0usize;
    // A test-only module's own submodules are test-only too: everything under its directory.
    let test_dirs: Vec<PathBuf> = test_files
        .iter()
        .map(|f| {
            if f.file_name().is_some_and(|n| n == "mod.rs") {
                f.parent().unwrap().to_path_buf()
            } else {
                f.with_extension("")
            }
        })
        .collect();
    for (path, src) in &parsed {
        if test_files.contains(path) || test_dirs.iter().any(|d| path.starts_with(d)) {
            continue;
        }
        scanned += 1;
        let rel = path
            .strip_prefix(&crates)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let listed = ALLOWLIST.contains(&rel.as_str());
        for (line, name) in src.calls() {
            if src.marked(line) {
                if listed {
                    *marked_files.entry(rel.clone()).or_default() += 1;
                    continue;
                }
                offenders.push(format!(
                    "{rel}:{}: `{name}` carries a `{MARKER}` marker, but {rel} is not in \
                     ALLOWLIST",
                    line + 1
                ));
                continue;
            }
            offenders.push(format!("{rel}:{}: platform `{name}`", line + 1));
        }
    }
    // The scan must have RUN: a moved `crates/` would otherwise pass with nothing scanned.
    assert!(
        scanned >= 100,
        "scanned only {scanned} production files; the scanner matches nothing"
    );
    for listed in ALLOWLIST {
        if !marked_files.contains_key(*listed) {
            offenders.push(format!(
                "{listed} is in ALLOWLIST but has no marked call left: remove it from the list"
            ));
        }
    }
    offenders.sort();
    let summary: Vec<String> = marked_files
        .iter()
        .map(|(f, n)| format!("{f}: {n}"))
        .collect();
    println!("allowlisted calls:\n{}", summary.join("\n"));
    assert!(
        offenders.is_empty(),
        "use `infr_core::math` instead of the platform's math functions (see docs/math.md):\n{}",
        offenders.join("\n")
    );
}

/// The detector on a sample: every form of call is found, nothing in a comment, a string, a
/// `#[cfg(test)]` item or an exact function, and a marker covers its call.
#[test]
fn the_detector_finds_each_form_and_nothing_else() {
    let text = r##"
fn a(x: f32) -> f32 { x.exp() + f32::ln(x) + x . tanh () }
fn b(v: &[f32]) -> Vec<f32> { v.iter().map(|x| x.sin()).map(f64::cos).collect() }
// x.exp() in a comment
/* x.powf(2.0) in a block comment */
fn c() -> &'static str { "x.ln_1p()" }
fn d() -> &'static str { r#"x.sin()"# }
fn e(x: f32, c: char) -> f32 { let _ = '"'; x.sqrt().powi(2).mul_add(x, x).abs() + x.expensive() }
fn f(x: f32) -> f32 {
    // libm-ok(S2): kept for the sample
    x.powf(0.5)
}
#[cfg(test)]
mod tests {
    fn t(x: f32) -> f32 { x.exp() }
}
#[cfg(test)]
mod more_tests;
fn g(x: f64) -> f64 { x.log2() + x.ln_1p() }
"##;
    let src = Source::parse(text);
    let calls: Vec<(usize, &str)> = src.calls();
    let names: Vec<&str> = calls.iter().map(|c| c.1).collect();
    assert_eq!(
        names,
        ["exp", "ln", "tanh", "sin", "cos", "powf", "log2", "ln_1p"],
        "{calls:?}"
    );
    let marked: Vec<bool> = calls.iter().map(|&(l, _)| src.marked(l)).collect();
    assert_eq!(
        marked,
        [false, false, false, false, false, true, false, false]
    );
    assert_eq!(src.test_mods, ["more_tests"]);
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            collect_rs(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// The files `mod name;` in `decl` can refer to.
fn module_files(decl: &Path, name: &str) -> Vec<PathBuf> {
    let dir = decl.parent().unwrap();
    let stem = decl.file_stem().unwrap().to_string_lossy();
    let base = if matches!(&*stem, "lib" | "main" | "mod") {
        dir.to_path_buf()
    } else {
        dir.join(&*stem)
    };
    vec![
        base.join(format!("{name}.rs")),
        base.join(name).join("mod.rs"),
    ]
}

/// A source file split into its code (comments and literals blanked, `#[cfg(test)]` items
/// removed, line structure kept) and its original lines (for markers).
struct Source {
    code: Vec<String>,
    original: Vec<String>,
    /// Names of `#[cfg(test)] mod name;` declarations.
    test_mods: Vec<String>,
}

impl Source {
    fn parse(text: &str) -> Source {
        let blank = blank_comments_and_literals(text);
        let (code, test_mods) = remove_cfg_test_items(&blank);
        Source {
            code: code.lines().map(str::to_owned).collect(),
            original: text.lines().map(str::to_owned).collect(),
            test_mods,
        }
    }

    /// `(line, name)` of every forbidden call, in order.
    fn calls(&self) -> Vec<(usize, &'static str)> {
        let mut out = Vec::new();
        for (i, line) in self.code.iter().enumerate() {
            let b = line.as_bytes();
            let mut k = 0;
            while k < b.len() {
                let hit = if b[k] == b'.' && (k == 0 || b[k - 1] != b'.') {
                    // `.name (` — method call, whitespace allowed around the name.
                    let j = skip_ws(b, k + 1);
                    ident_at(b, j).and_then(|(name, end)| {
                        let after = skip_ws(b, end);
                        (after < b.len() && b[after] == b'(')
                            .then(|| forbidden(name))
                            .flatten()
                    })
                } else if b[k..].starts_with(b"f32::") || b[k..].starts_with(b"f64::") {
                    let ok_before = k == 0 || !is_ident(b[k - 1]);
                    ident_at(b, k + 5)
                        .and_then(|(name, _)| ok_before.then(|| forbidden(name)).flatten())
                } else {
                    None
                };
                if let Some(name) = hit {
                    out.push((i, name));
                }
                k += 1;
            }
        }
        out
    }

    /// Does line `i` carry the marker, on itself or in the `//` block directly above?
    fn marked(&self, i: usize) -> bool {
        if self.original[i].contains(MARKER) {
            return true;
        }
        let mut j = i;
        while j > 0 {
            let above = self.original[j - 1].trim_start();
            if !above.starts_with("//") {
                return false;
            }
            if above.contains(MARKER) {
                return true;
            }
            j -= 1;
        }
        false
    }
}

fn forbidden(name: &str) -> Option<&'static str> {
    FORBIDDEN.iter().copied().find(|f| *f == name)
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn skip_ws(b: &[u8], mut k: usize) -> usize {
    while k < b.len() && b[k].is_ascii_whitespace() {
        k += 1;
    }
    k
}

/// The identifier starting at `k`, and the index after it.
fn ident_at(b: &[u8], k: usize) -> Option<(&str, usize)> {
    let mut e = k;
    while e < b.len() && is_ident(b[e]) {
        e += 1;
    }
    (e > k && !b[k].is_ascii_digit()).then(|| (std::str::from_utf8(&b[k..e]).unwrap(), e))
}

/// `text` with every comment and string/char literal replaced by spaces (newlines kept).
fn blank_comments_and_literals(text: &str) -> String {
    let c: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let blank = |out: &mut String, ch: char| out.push(if ch == '\n' { '\n' } else { ' ' });
    let mut i = 0;
    while i < c.len() {
        let at = |k: usize| c.get(k).copied().unwrap_or('\0');
        if at(i) == '/' && at(i + 1) == '/' {
            while i < c.len() && c[i] != '\n' {
                blank(&mut out, c[i]);
                i += 1;
            }
        } else if at(i) == '/' && at(i + 1) == '*' {
            let mut depth = 0;
            loop {
                if at(i) == '/' && at(i + 1) == '*' {
                    depth += 1;
                    blank(&mut out, c[i]);
                    blank(&mut out, c[i + 1]);
                    i += 2;
                } else if at(i) == '*' && at(i + 1) == '/' {
                    depth -= 1;
                    blank(&mut out, c[i]);
                    blank(&mut out, c[i + 1]);
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else if i >= c.len() {
                    break;
                } else {
                    blank(&mut out, c[i]);
                    i += 1;
                }
            }
        } else if (at(i) == 'r' || (at(i) == 'b' && at(i + 1) == 'r'))
            && (i == 0 || !(c[i - 1].is_alphanumeric() || c[i - 1] == '_'))
            && {
                let mut k = i + if at(i) == 'b' { 2 } else { 1 };
                while at(k) == '#' {
                    k += 1;
                }
                at(k) == '"'
            }
        {
            // Raw string: r"..." / r#"..."# (and b-prefixed).
            let mut k = i + if at(i) == 'b' { 2 } else { 1 };
            let mut hashes = 0;
            while at(k) == '#' {
                hashes += 1;
                k += 1;
            }
            k += 1; // opening quote
            loop {
                if k >= c.len() {
                    break;
                }
                if c[k] == '"' && (1..=hashes).all(|h| at(k + h) == '#') {
                    k += 1 + hashes;
                    break;
                }
                k += 1;
            }
            for &ch in &c[i..k.min(c.len())] {
                blank(&mut out, ch);
            }
            i = k;
        } else if at(i) == '"' {
            blank(&mut out, c[i]);
            i += 1;
            while i < c.len() && c[i] != '"' {
                if c[i] == '\\' {
                    blank(&mut out, c[i]);
                    i += 1;
                }
                if i < c.len() {
                    blank(&mut out, c[i]);
                    i += 1;
                }
            }
            if i < c.len() {
                blank(&mut out, c[i]);
                i += 1;
            }
        } else if at(i) == '\'' && (at(i + 1) == '\\' || at(i + 2) == '\'') {
            // A char literal ('x', '\n', '\u{..}'); a lifetime has no closing quote here.
            let mut k = i + 1;
            if at(k) == '\\' {
                k += 2;
                while k < c.len() && c[k] != '\'' {
                    k += 1;
                }
            } else {
                k += 1;
            }
            k += 1;
            for &ch in &c[i..k.min(c.len())] {
                blank(&mut out, ch);
            }
            i = k;
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    out
}

/// `code` with every `#[cfg(test)]`-attributed item blanked, and the names of the
/// `#[cfg(test)] mod name;` declarations among them.
fn remove_cfg_test_items(code: &str) -> (String, Vec<String>) {
    let mut b: Vec<u8> = code.as_bytes().to_vec();
    let mut mods = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let Some(end) = match_cfg_test(&b, i) else {
            i += 1;
            continue;
        };
        // Skip further attributes, then the item: up to its `;` or its matching `}`.
        let mut k = skip_ws(&b, end);
        while k < b.len() && b[k] == b'#' {
            let mut depth = 0;
            while k < b.len() {
                match b[k] {
                    b'[' => depth += 1,
                    b']' => {
                        depth -= 1;
                        if depth == 0 {
                            k += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                k += 1;
            }
            k = skip_ws(&b, k);
        }
        let item_start = k;
        let mut depth = 0;
        while k < b.len() {
            match b[k] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        k += 1;
                        break;
                    }
                }
                b';' if depth == 0 => {
                    k += 1;
                    let item = String::from_utf8_lossy(&b[item_start..k]).to_string();
                    let words: Vec<&str> = item
                        .split(|ch: char| ch.is_whitespace() || ch == ';')
                        .filter(|w| !w.is_empty())
                        .collect();
                    if let Some(p) = words.iter().position(|w| *w == "mod") {
                        if let Some(name) = words.get(p + 1) {
                            mods.push(name.to_string());
                        }
                    }
                    break;
                }
                _ => {}
            }
            k += 1;
        }
        let k = k.min(b.len());
        for ch in &mut b[i..k] {
            if *ch != b'\n' {
                *ch = b' ';
            }
        }
        i = k;
    }
    (String::from_utf8(b).unwrap(), mods)
}

/// If `#[cfg(test)]` (whitespace allowed between tokens) starts at `i`, the index after it.
fn match_cfg_test(b: &[u8], i: usize) -> Option<usize> {
    let mut k = i;
    for tok in ["#", "[", "cfg", "(", "test", ")", "]"] {
        k = skip_ws(b, k);
        if !b[k..].starts_with(tok.as_bytes()) {
            return None;
        }
        k += tok.len();
        if (tok == "cfg" || tok == "test") && k < b.len() && is_ident(b[k]) {
            return None;
        }
    }
    Some(k)
}
