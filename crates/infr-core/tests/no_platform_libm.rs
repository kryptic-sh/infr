//! **No platform math in production code.** `f32::exp`, `ln`, `tanh`, `powf`, `sin`, `cos` and
//! their relatives lower to the C runtime, whose last bits differ between operating systems,
//! C library versions and architectures — so any output computed through one differs too. The
//! workspace uses `infr_core::math` (correctly rounded, the same bits everywhere) instead; see
//! `docs/math.md` and `docs/plan/11-cpu-bit-identical.md`.
//!
//! **What it scans:** the production source of every workspace crate — `crates/<crate>/src/**`,
//! anchored at the crate roots — minus `#[cfg(test)]` code: the one element the attribute
//! applies to (an item, a field, a variant, a match arm, a struct-literal field, a statement)
//! and the files of `#[cfg(test)] mod x;` declarations with everything under them. Comments and
//! string and char literals are not code and are not scanned.
//!
//! **What counts as a call**, for every name in [`FORBIDDEN`]:
//! - a method call `.name(` (spaces allowed), which also covers calls on generic values;
//! - a path `Q::name`, with or without a call, so a function passed as a value
//!   (`.map(f32::exp)`) counts: `Q` is anything but this module (`math::name`) — `f32`, `f64`,
//!   `core::f32`, a type alias, a type parameter — or a qualified type (`<f32>::exp`,
//!   `<F as Float>::tanh`), spaces allowed around `::`, `<` and `>`.
//!
//! `powi` is checked separately ([`powi_is_exact`]). It lowers to compiler-builtins'
//! `__powisf2`/`__powidf2` (repeated squaring in the operand's own type, the same code on every
//! target), but LLVM may constant-fold a call whose operands are both constants using the HOST's
//! `pow`, which need not round the same way as repeated squaring. Two shapes are exact either
//! way and are allowed: an exponent of `-1`, `0`, `1` or `2` written as a literal (each a single
//! rounding, `x*x` for 2), and a base of literal 2 (`2f32.powi(n)`, `f32::powi(2.0, n)`: a power
//! of two, exact while in range). Any other `powi` counts as a call.
//!
//! **What it cannot see**, being a source scan without name resolution: a method name built by
//! a macro (`paste!`, `concat_idents!`), a call through a re-export or wrapper function defined
//! outside the scanned crates, the C names reached through FFI (`extern "C" { fn expf(..) }`),
//! `core::intrinsics`, and anything in a dependency's source. None exists in the tree today;
//! review has to catch them.
//!
//! **No exceptions.** Every call site moved to `infr_core::math` in slice S2 of the plan, and
//! there is no allowlist or marker: a platform call anywhere in production code outside the
//! math module fails the scan. A new use needs a function in `infr_core::math` instead.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// The platform math functions (`f32`/`f64` methods that lower to the C runtime).
const FORBIDDEN: &[&str] = &[
    "exp", "exp2", "exp_m1", "ln", "ln_1p", "log", "log2", "log10", "powf", "sin", "cos", "tan",
    "sin_cos", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "cbrt", "hypot",
];

/// `infr_core::math` itself, which is not scanned: its own paths (`pub use pow::powf`) name
/// its replacements, and its determinism rule (no platform math at all, stated in its module
/// doc) is stricter than this lint.
const MATH_MODULE: &str = "infr-core/src/math/";

#[test]
fn no_platform_libm_in_production_code() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .to_path_buf();
    let report = scan(&crates);
    // The scan must have RUN: a moved `crates/` would otherwise pass with nothing scanned.
    assert!(
        report.scanned >= 100,
        "scanned only {} production files; the scanner matches nothing",
        report.scanned
    );
    assert!(
        report.offenders.is_empty(),
        "use `infr_core::math` instead of the platform's math functions (see docs/math.md):\n{}",
        report.offenders.join("\n")
    );
}

/// What a scan of a `crates/` directory found.
struct Report {
    scanned: usize,
    offenders: Vec<String>,
}

/// Scan `crates/<crate>/src/**` under `crates`.
fn scan(crates: &Path) -> Report {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(crates).expect("read crates/").flatten() {
        let src = entry.path().join("src");
        if src.is_dir() {
            collect_rs(&src, &mut files);
        }
    }
    let parsed: Vec<(PathBuf, Source)> = files
        .iter()
        .map(|p| {
            (
                p.clone(),
                Source::parse(&std::fs::read_to_string(p).unwrap()),
            )
        })
        .collect();
    // Files that are test-only because a `#[cfg(test)] mod x;` declares them, and (for a module
    // with submodules) everything under the module's directory.
    let mut test_files = HashSet::new();
    for (path, src) in &parsed {
        for m in &src.test_mods {
            for c in module_files(path, m) {
                test_files.insert(c);
            }
        }
    }
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
    let mut offenders = Vec::new();
    let mut scanned = 0usize;
    for (path, src) in &parsed {
        if test_files.contains(path) || test_dirs.iter().any(|d| path.starts_with(d)) {
            continue;
        }
        scanned += 1;
        let rel = path
            .strip_prefix(crates)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        if rel.starts_with(MATH_MODULE) {
            continue;
        }
        offenders.append(&mut src.check(&rel));
    }
    offenders.sort();
    Report { scanned, offenders }
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
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

/// A source file split into its code (comments and literals blanked, `#[cfg(test)]` code
/// removed, line structure kept).
struct Source {
    code: Vec<String>,
    /// Names of `#[cfg(test)] mod name;` declarations.
    test_mods: Vec<String>,
}

impl Source {
    fn parse(text: &str) -> Source {
        let blank = blank_comments_and_literals(text);
        let (code, test_mods) = remove_cfg_test_items(&blank);
        Source {
            code: code.lines().map(str::to_owned).collect(),
            test_mods,
        }
    }

    /// `(line, name)` of every platform call, in order.
    fn calls(&self) -> Vec<(usize, &'static str)> {
        let mut out = Vec::new();
        for (i, line) in self.code.iter().enumerate() {
            let b = line.as_bytes();
            let mut k = 0;
            while k < b.len() {
                if let Some(name) = call_at(b, k) {
                    out.push((i, name));
                }
                k += 1;
            }
        }
        out
    }

    /// One error per platform call.
    fn check(&self, rel: &str) -> Vec<String> {
        self.calls()
            .into_iter()
            .map(|(line, name)| format!("{rel}:{}: platform `{name}`", line + 1))
            .collect()
    }
}

/// The platform call starting at byte `k` of a code line, if one does.
fn call_at(b: &[u8], k: usize) -> Option<&'static str> {
    if b[k] == b'.' && (k == 0 || b[k - 1] != b'.') {
        // `.name (` — a method call; `powi` only when not exact.
        let j = skip_ws(b, k + 1);
        let (name, end) = ident_at(b, j)?;
        let after = skip_ws(b, end);
        if after >= b.len() || b[after] != b'(' {
            return None;
        }
        if name == "powi" {
            let arg = paren_arg(b, after)?;
            return (!(powi_is_exact(receiver_before(b, k), &arg))).then_some("powi");
        }
        return forbidden(name);
    }
    if b[k] == b':' && b.get(k + 1) == Some(&b':') && (k == 0 || b[k - 1] != b':') {
        // `Q :: name` — a path; Q is what precedes the `::`.
        let j = skip_ws(b, k + 2);
        let (name, end) = ident_at(b, j)?;
        // A module segment (`tracing::log::warn`), not a function.
        if b[skip_ws(b, end)..].starts_with(b"::") {
            return None;
        }
        let before = skip_ws_back(b, k);
        let qualifier = if before > 0 && b[before - 1] == b'>' {
            "<qualified>"
        } else {
            ident_before(b, before)?
        };
        if qualifier == "math" {
            return None;
        }
        if name == "powi" {
            let after = skip_ws(b, end);
            let arg = b
                .get(after)
                .filter(|&&c| c == b'(')
                .and_then(|_| paren_arg(b, after));
            let first = arg
                .as_deref()
                .and_then(|a| a.split(',').next())
                .unwrap_or("");
            let second = arg
                .as_deref()
                .and_then(|a| a.split(',').nth(1))
                .unwrap_or("");
            return (!powi_is_exact(first.trim(), second.trim())).then_some("powi");
        }
        return forbidden(name);
    }
    None
}

/// Is `base.powi(exp)` exact on every target and under constant folding? See the module doc.
fn powi_is_exact(base: &str, exp: &str) -> bool {
    let base = base.trim().replace('_', "");
    let base_two = ["2f32", "2f64", "2.0f32", "2.0f64", "2.0", "2."].contains(&base.as_str());
    base_two || ["-1", "0", "1", "2"].contains(&exp.trim())
}

/// The literal or identifier token ending right before byte `k` (a method's receiver), or "".
fn receiver_before(b: &[u8], k: usize) -> &str {
    let mut s = k;
    while s > 0 && (is_ident(b[s - 1]) || b[s - 1] == b'.') {
        s -= 1;
    }
    std::str::from_utf8(&b[s..k]).unwrap_or("")
}

/// The text inside the parentheses opening at `open`, if they close on this line.
fn paren_arg(b: &[u8], open: usize) -> Option<String> {
    let mut depth = 0;
    for (i, &c) in b.iter().enumerate().skip(open) {
        match c {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(String::from_utf8_lossy(&b[open + 1..i]).into_owned());
                }
            }
            _ => {}
        }
    }
    None
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

/// The index after skipping whitespace backwards from just before `k`.
fn skip_ws_back(b: &[u8], mut k: usize) -> usize {
    while k > 0 && b[k - 1].is_ascii_whitespace() {
        k -= 1;
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

/// The identifier ending right before `k`.
fn ident_before(b: &[u8], k: usize) -> Option<&str> {
    let mut s = k;
    while s > 0 && is_ident(b[s - 1]) {
        s -= 1;
    }
    (s < k).then(|| std::str::from_utf8(&b[s..k]).unwrap())
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

/// Item keywords whose item ends at its body's closing brace (or a `;` before any body).
const BRACE_ITEMS: &[&str] = &[
    "fn",
    "mod",
    "impl",
    "struct",
    "enum",
    "union",
    "trait",
    "extern",
    "macro_rules",
];
/// Item and statement keywords that end at a `;`.
const SEMI_ITEMS: &[&str] = &["const", "static", "use", "type", "let"];
/// Words that may precede an item keyword.
const QUALIFIERS: &[&str] = &["pub", "unsafe", "async", "default", "const"];

/// `code` with every `#[cfg(test)]`-attributed element blanked, and the names of the
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
        // Further attributes on the same element.
        let mut k = skip_ws(&b, end);
        while k < b.len() && b[k] == b'#' {
            k = skip_ws(
                &b,
                matching(&b, skip_ws(&b, k + 1)).map_or(b.len(), |e| e + 1),
            );
        }
        let start = k;
        let stop = element_end(&b, start);
        let words = leading_words(&b, start);
        if words.first().map(String::as_str) == Some("mod")
            || (words.len() >= 2 && words[0] == "pub" && words[1] == "mod")
        {
            let item = String::from_utf8_lossy(&b[start..stop]).to_string();
            if item.trim_end().ends_with(';') {
                if let Some(name) = item
                    .split(|ch: char| ch.is_whitespace() || ch == ';')
                    .filter(|w| !w.is_empty())
                    .skip_while(|w| *w != "mod")
                    .nth(1)
                {
                    mods.push(name.to_string());
                }
            }
        }
        for ch in &mut b[i..stop] {
            if *ch != b'\n' {
                *ch = b' ';
            }
        }
        i = stop;
    }
    (String::from_utf8(b).unwrap(), mods)
}

/// The identifiers at the start of an element, past `pub(...)`, until the first non-identifier.
fn leading_words(b: &[u8], mut k: usize) -> Vec<String> {
    let mut out = Vec::new();
    loop {
        k = skip_ws(b, k);
        let Some((w, e)) = ident_at(b, k) else {
            break;
        };
        out.push(w.to_string());
        k = skip_ws(b, e);
        if w == "pub" && k < b.len() && b[k] == b'(' {
            k = matching(b, k).map_or(b.len(), |e| e + 1);
        }
        if out.len() > 6 {
            break;
        }
    }
    out
}

/// The index just past the element starting at `k`: an item or statement, or an inner element
/// (field, variant, arm, argument) that ends at its `,`, at its enclosing group's close, or —
/// for a match arm with a block body — at the block's close.
fn element_end(b: &[u8], k: usize) -> usize {
    if k >= b.len() {
        return b.len();
    }
    if b[k] == b'{' {
        return matching(b, k).map_or(b.len(), |e| e + 1);
    }
    let words = leading_words(b, k);
    let kw = words
        .iter()
        .find(|w| !QUALIFIERS.contains(&w.as_str()))
        .map(String::as_str);
    let first = words.first().map(String::as_str);
    if kw.is_some_and(|w| BRACE_ITEMS.contains(&w)) {
        // Up to a `;` before any body, or through the body's closing brace.
        let mut j = k;
        while j < b.len() {
            match b[j] {
                b'(' | b'[' => j = matching(b, j).map_or(b.len(), |e| e + 1),
                b';' => return j + 1,
                b'{' => return matching(b, j).map_or(b.len(), |e| e + 1),
                _ => j += 1,
            }
        }
        return b.len();
    }
    if kw.is_some_and(|w| SEMI_ITEMS.contains(&w)) || first == Some("const") {
        let mut j = k;
        while j < b.len() {
            match b[j] {
                b'(' | b'[' | b'{' => j = matching(b, j).map_or(b.len(), |e| e + 1),
                b';' => return j + 1,
                _ => j += 1,
            }
        }
        return b.len();
    }
    // An inner element.
    let mut j = k;
    while j < b.len() {
        match b[j] {
            b'(' | b'[' | b'{' => j = matching(b, j).map_or(b.len(), |e| e + 1),
            b')' | b']' | b'}' => return j,
            b',' | b';' => return j + 1,
            b'=' if b.get(j + 1) == Some(&b'>') => {
                let body = skip_ws(b, j + 2);
                if body < b.len() && b[body] == b'{' {
                    let close = matching(b, body).map_or(b.len(), |e| e + 1);
                    let after = skip_ws(b, close);
                    return if after < b.len() && b[after] == b',' {
                        after + 1
                    } else {
                        close
                    };
                }
                j += 2;
            }
            _ => j += 1,
        }
    }
    b.len()
}

/// The index of the bracket closing the one at `open` (any of `()[]{}`), if any.
fn matching(b: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (j, &c) in b.iter().enumerate().skip(open) {
        match c {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(j);
                }
            }
            _ => {}
        }
    }
    None
}

/// If `#[cfg(test)]` (whitespace allowed between tokens) starts at `i`, the index after it.
fn match_cfg_test(b: &[u8], i: usize) -> Option<usize> {
    if b[i] != b'#' {
        return None;
    }
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

// ── the scanner's own tests ──────────────────────────────────────────────────────────────────

/// The detector on a sample: every form of call is found, nothing in a comment, a string, a
/// `#[cfg(test)]` item or an exact function.
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
    // A comment does not excuse the call under it.
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
    let names: Vec<&str> = src.calls().iter().map(|c| c.1).collect();
    assert_eq!(
        names,
        ["exp", "ln", "tanh", "sin", "cos", "powf", "log2", "ln_1p"]
    );
    let errs = src.check("sample.rs");
    assert_eq!(errs.len(), 8, "{errs:#?}");
    assert_eq!(src.test_mods, ["more_tests"]);
}

/// `#[cfg(test)]` on a field, a variant, a match arm, a struct-literal field or a statement
/// blanks only that element: the production code after it is still scanned.
#[test]
fn cfg_test_on_an_inner_element_skips_only_that_element() {
    let cases: &[(&str, &str)] = &[
        (
            "field",
            "struct S { #[cfg(test)] t: u8, v: u8 } impl S { fn p(x: f32) -> f32 { x.exp() } }",
        ),
        (
            "last field",
            "struct S { v: u8, #[cfg(test)] t: u8 }\nfn p(x: f32) -> f32 { x.exp() }",
        ),
        (
            "variant",
            "enum E { A, #[cfg(test)] B(u8), C }\nfn p(x: f32) -> f32 { x.exp() }",
        ),
        (
            "match arm with a block",
            "fn p(e: u8, x: f32) -> f32 { match e { #[cfg(test)] 0 => { 1.0 } _ => x.exp() } }",
        ),
        (
            "match arm with an expression",
            "fn p(e: u8, x: f32) -> f32 { match e { #[cfg(test)] 0 => 1.0, _ => x.exp() } }",
        ),
        (
            "struct-literal field",
            "fn p(x: f32) -> S { S { #[cfg(test)] t: 0, v: x.exp() } }",
        ),
        (
            "statement",
            "fn p(x: f32) -> f32 { #[cfg(test)] let _a = 1; x.exp() }",
        ),
        (
            "block statement",
            "fn p(x: f32) -> f32 { #[cfg(test)] { let _a = 1; } x.exp() }",
        ),
        (
            "fn in an impl",
            "impl S { #[cfg(test)] fn t() {} fn p(x: f32) -> f32 { x.exp() } }",
        ),
        (
            "const with a block",
            "#[cfg(test)] const T: S = S { a: 1 };\nfn p(x: f32) -> f32 { x.exp() }",
        ),
    ];
    for (what, text) in cases {
        let names: Vec<&str> = Source::parse(text).calls().iter().map(|c| c.1).collect();
        assert_eq!(names, ["exp"], "{what}: {text}");
    }
    // And the attributed element itself IS skipped.
    let text = "impl S { #[cfg(test)] fn t(x: f32) -> f32 { x.exp() } fn p() {} }";
    assert!(Source::parse(text).calls().is_empty());
}

/// The qualified, spaced, value and generic forms of a platform call are found.
#[test]
fn qualified_spaced_and_value_forms_are_found() {
    let cases: &[(&str, &str)] = &[
        ("qualified", "fn p(x: f32) -> f32 { <f32>::exp(x) }"),
        (
            "qualified with spaces",
            "fn p(x: f32) -> f32 { < f32 > :: exp(x) }",
        ),
        (
            "trait-qualified",
            "fn p<F: Float>(x: F) -> F { <F as Float>::tanh(x) }",
        ),
        ("spaced path", "fn p(x: f32) -> f32 { f32 :: exp(x) }"),
        (
            "function value",
            "fn p(v: &[f32]) -> Vec<f32> { v.iter().copied().map(f32::exp).collect() }",
        ),
        (
            "qualified value",
            "fn p(v: &[f32]) -> Vec<f32> { v.iter().copied().map(<f32>::tanh).collect() }",
        ),
        ("type alias", "type R = f32;\nfn p(x: R) -> R { R::exp(x) }"),
        ("type parameter", "fn p<F: Float>(x: F) -> F { F::ln(x) }"),
        ("core path", "fn p(x: f64) -> f64 { core::f64::exp(x) }"),
    ];
    for (what, text) in cases {
        let calls = Source::parse(text).calls();
        assert_eq!(calls.len(), 1, "{what}: {text} -> {calls:?}");
    }
    // This module's own functions are the replacement, not a platform call.
    let text = "fn p(x: f32) -> f32 { math::exp(x) + infr_core::math::tanh(x) }";
    assert!(Source::parse(text).calls().is_empty());
}

/// Only `crates/<crate>/src` is scanned, even when the checkout itself sits under a directory
/// named `src`: a crate's `tests/` stays out.
#[test]
fn the_scan_is_anchored_at_the_crate_roots() {
    let root = std::env::temp_dir().join(format!("libm-lint-{}", std::process::id()));
    let crates = root.join("src").join("crates");
    let lib = crates.join("a").join("src");
    let tests = crates.join("a").join("tests");
    std::fs::create_dir_all(&lib).unwrap();
    std::fs::create_dir_all(&tests).unwrap();
    std::fs::write(lib.join("lib.rs"), "pub fn p(x: f32) -> f32 { x }\n").unwrap();
    std::fs::write(tests.join("t.rs"), "fn t(x: f32) -> f32 { x.exp() }\n").unwrap();
    let report = scan(&crates);
    std::fs::remove_dir_all(&root).unwrap();
    assert_eq!(report.scanned, 1);
    let ours: Vec<&String> = report
        .offenders
        .iter()
        .filter(|o| o.starts_with("a/"))
        .collect();
    assert!(ours.is_empty(), "{ours:?}");
}

/// `powi` is allowed only in its exact shapes: a literal exponent of -1, 0, 1 or 2, or a literal
/// base of 2.
#[test]
fn powi_is_allowed_only_where_exact() {
    let names = |text: &str| -> Vec<&'static str> {
        Source::parse(text).calls().iter().map(|c| c.1).collect()
    };
    for ok in [
        "fn p(x: f32) -> f32 { x.powi(2) }",
        "fn p(x: f32) -> f32 { (x - 1.0).powi(2) }",
        "fn p(x: f32) -> f32 { x.powi(-1) }",
        "fn p(e: i32) -> f32 { 2f32.powi(e - 7) }",
        "fn p(e: i32) -> f32 { 2.0_f32.powi(e) }",
        "fn p(e: i32) -> f32 { f32::powi(2.0, e - 7) }",
    ] {
        assert!(names(ok).is_empty(), "{ok}");
    }
    for bad in [
        "fn p(x: f32) -> f32 { x.powi(3) }",
        "fn p(x: f32, n: i32) -> f32 { x.powi(n) }",
        "fn p(x: f32) -> f32 { 3f32.powi(-2) }",
        "fn p(x: f64, n: i32) -> f64 { f64::powi(x, n) }",
    ] {
        assert_eq!(names(bad), ["powi"], "{bad}");
    }
}
