//! Turning a template value into text the way Python's Jinja2 does, which is what Hugging Face's
//! `apply_chat_template` renders with and so what a model was trained on.
//!
//! Jinja2 prints a value with Python's `str()`: `True` / `False` / `None`, a float as
//! `repr(float)` (`1e-07`, `1e+16`, `2.0`), and a container as its `repr()`, which quotes the
//! strings inside it with single quotes (`{'a': [1, 'x', True]}`). minijinja's own `Display` writes
//! JSON-like containers and never writes a float in exponent form, so infr routes every conversion
//! it can reach through [`py_str`]: bare `{{ }}` ([`formatter`]), `| string`, `| join`, the
//! string filters ([`upper_filter`] and its neighbours), the arguments of `| format` and
//! `str.format` (`crate::pyformat`), and the request's lists and dicts wherever minijinja prints
//! them itself (`crate::pyvalue`). The `~` operator on a float is what it cannot reach
//! (`KNOWN_GAPS` in `tests/jinja2_text.rs`). `tests/fixtures/jinja2_text.json` is the oracle
//! every rule here is checked against.

use std::fmt::Write;

use minijinja::value::{StringInput, ValueKind};
use minijinja::{Error, ErrorKind, Output, State, Value};

use crate::tojson::NESTING_MAX;

/// Python's `str(value)`: a string as it is, an undefined value as nothing, anything else as
/// [`write_repr`] writes it.
pub(crate) fn py_str(value: &Value) -> Result<String, Error> {
    if let Some(s) = value.as_str() {
        return Ok(s.to_owned());
    }
    let mut out = String::new();
    if !value.is_undefined() {
        write_repr(&mut out, value, 0)?;
    }
    Ok(out)
}

/// Append Python's `repr(value)` at container nesting `level`. A lazy sequence of a known length
/// (minijinja's slice of a list, where Python has a list) is written as a list; a value Python has
/// no counterpart for (a macro, a namespace) is written as minijinja writes it.
fn write_repr(out: &mut String, value: &Value, level: usize) -> Result<(), Error> {
    match value.kind() {
        ValueKind::Undefined => out.push_str("Undefined"),
        ValueKind::None => out.push_str("None"),
        ValueKind::Bool => out.push_str(if value.is_true() { "True" } else { "False" }),
        ValueKind::Number if value.is_integer() => write!(out, "{value}").expect("String write"),
        ValueKind::Number => out.push_str(&float_repr(f64::try_from(value.clone())?)),
        ValueKind::String => write_str_repr(out, value.as_str().unwrap_or_default()),
        ValueKind::Seq | ValueKind::Map | ValueKind::Iterable
            if value.as_object().and_then(|o| o.enumerator_len()).is_some() =>
        {
            if level >= NESTING_MAX {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("value is nested more than {NESTING_MAX} levels deep"),
                ));
            }
            let is_map = value.kind() == ValueKind::Map;
            out.push(if is_map { '{' } else { '[' });
            for (i, item) in value.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_repr(out, &item, level + 1)?;
                if is_map {
                    out.push_str(": ");
                    write_repr(out, &value.get_item(&item)?, level + 1)?;
                }
            }
            out.push(if is_map { '}' } else { ']' });
        }
        _ => write!(out, "{value}").expect("String write"),
    }
    Ok(())
}

/// Python's `repr(str)` (`unicode_repr` in CPython's `Objects/unicodeobject.c`): single quotes,
/// or double quotes when the string holds a `'` and no `"`; the chosen quote and `\` escaped;
/// `\t`, `\n`, `\r` by name; any other character that is not printable as `\xhh`, `\uhhhh` or
/// `\Uhhhhhhhh` by its size.
fn write_str_repr(out: &mut String, s: &str) {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if !is_printable(c) => {
                let n = u32::from(c);
                if n <= 0xff {
                    write!(out, "\\x{n:02x}")
                } else if n <= 0xffff {
                    write!(out, "\\u{n:04x}")
                } else {
                    write!(out, "\\U{n:08x}")
                }
                .expect("String write");
            }
            c => out.push(c),
        }
    }
    out.push(quote);
}

/// Python's `str.isprintable` for one character: false for the general categories Cc (control),
/// Cf (format), Co (private use), Zs (space separators other than the ASCII space), Zl and Zp.
///
/// Python also escapes Cn (unassigned code points) and Cs (surrogates). Cs cannot occur in a
/// Rust `str`; Cn is not covered: telling it apart needs the whole Unicode table, and which code
/// points are unassigned moves with Python's Unicode version. An unassigned code point inside a
/// container is therefore written raw here and escaped by Python.
fn is_printable(c: char) -> bool {
    if c == ' ' {
        return true;
    }
    // `is_whitespace` (the White_Space property) is the Cc whitespace plus every Zs, Zl and Zp
    // character; `is_control` is exactly Cc.
    !(c.is_control() || c.is_whitespace() || is_private_use(c) || is_format(c))
}

/// General category Co: the three private-use ranges, fixed since Unicode 2.0.
fn is_private_use(c: char) -> bool {
    matches!(u32::from(c), 0xE000..=0xF8FF | 0xF_0000..=0xF_FFFD | 0x10_0000..=0x10_FFFD)
}

/// General category Cf, as Python 3.15's `unicodedata` (Unicode 17.0) lists it.
fn is_format(c: char) -> bool {
    matches!(
        u32::from(c),
        0xAD | 0x600..=0x605
            | 0x61C
            | 0x6DD
            | 0x70F
            | 0x890..=0x891
            | 0x8E2
            | 0x180E
            | 0x200B..=0x200F
            | 0x202A..=0x202E
            | 0x2060..=0x2064
            | 0x2066..=0x206F
            | 0xFEFF
            | 0xFFF9..=0xFFFB
            | 0x110BD
            | 0x110CD
            | 0x13430..=0x1343F
            | 0x1BCA0..=0x1BCA3
            | 0x1D173..=0x1D17A
            | 0xE0001
            | 0xE0020..=0xE007F
    )
}

/// `x` as Python's `repr(float)` writes it: the shortest digits that read back as `x`, in fixed
/// notation when the decimal point falls at `-4 < point <= 16` (a whole number gets `.0`),
/// otherwise as `d.ddde±XX`; `nan`, `inf`, `-inf` for the rest.
pub(crate) fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_owned();
    }
    let mut out = String::from(if x.is_sign_negative() { "-" } else { "" });
    if x == 0.0 {
        out.push_str("0.0");
        return out;
    }
    let (digits, point) = shortest_digits(x.abs());
    let n = digits.len() as i32;
    if point > -4 && point <= 16 {
        if point <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', point.unsigned_abs() as usize));
            out.push_str(&digits);
        } else if point >= n {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', (point - n) as usize));
            out.push_str(".0");
        } else {
            let (int, frac) = digits.split_at(point as usize);
            write!(out, "{int}.{frac}").expect("String write");
        }
    } else {
        let (first, rest) = digits.split_at(1);
        out.push_str(first);
        if !rest.is_empty() {
            write!(out, ".{rest}").expect("String write");
        }
        let exp = point - 1;
        let sign = if exp < 0 { '-' } else { '+' };
        write!(out, "e{sign}{:02}", exp.abs()).expect("String write");
    }
    out
}

/// The shortest digits that read back as `x` (finite, positive), without leading or trailing
/// zeros, and how many of them come before the decimal point (`0.0015` is `("15", -2)`).
///
/// Taken from serde_json, which writes a float with `ryu`: when two shortest digit strings are
/// equally close to `x`, `ryu` keeps the even one, as CPython's `repr` does. Rust's own `{}` and
/// `{:e}` round such a tie up (`1125899906842624.25` comes out `…4.3`, Python writes `…4.2`).
fn shortest_digits(x: f64) -> (String, i32) {
    let text = serde_json::to_string(&x).expect("a finite float serializes");
    let (mantissa, exp) = match text.split_once(['e', 'E']) {
        Some((m, e)) => (
            m,
            e.parse::<i32>()
                .expect("serde_json writes an integer exponent"),
        ),
        None => (text.as_str(), 0),
    };
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all = format!("{int}{frac}");
    let digits = all.trim_start_matches('0');
    let point = int.len() as i32 + exp - (all.len() - digits.len()) as i32;
    (digits.trim_end_matches('0').to_owned(), point)
}

/// The `{{ }}` formatter: Jinja2's default `finalize`, which is `str()`. Strings and undefined
/// values go through minijinja's own formatter unchanged (the chat template is compiled without
/// auto-escaping, so that writes them as they are).
pub(crate) fn formatter(out: &mut Output, state: &State, value: &Value) -> Result<(), Error> {
    if value.is_undefined() || value.kind() == ValueKind::String {
        return minijinja::escape_formatter(out, state, value);
    }
    out.write_str(&py_str(value)?).map_err(Error::from)
}

/// Jinja2's `string` filter: `str(value)`. A string, safe or not, is returned as it is.
pub(crate) fn string_filter(value: &Value) -> Result<Value, Error> {
    if value.kind() == ValueKind::String {
        return Ok(value.clone());
    }
    py_str(value).map(Value::from)
}

/// Jinja2's `join` filter: `str(d).join(map(str, value))`.
pub(crate) fn join_filter(value: &Value, joiner: Option<String>) -> Result<Value, Error> {
    let joiner = joiner.unwrap_or_default();
    let mut out = String::new();
    for (i, item) in value.try_iter()?.enumerate() {
        if i > 0 {
            out.push_str(&joiner);
        }
        out.push_str(&py_str(&item)?);
    }
    Ok(Value::from(out))
}

/// Jinja2's `upper`, `lower`, `capitalize`, `trim` and `replace` start from `str(value)` (its
/// `soft_str`), so a list or a float is cased as Python writes it: a tool parameter typed
/// `["string", "null"]` through Gemma 4's `| upper` is `['STRING', 'NULL']`. minijinja's own
/// filters take a non-string through its `Display`; these hand them [`string_filter`]'s text.
pub(crate) fn upper_filter(state: &State, value: &Value) -> Result<Value, Error> {
    let text = string_filter(value)?;
    Ok(minijinja::filters::upper(StringInput::new(state, &text)?))
}

/// See [`upper_filter`].
pub(crate) fn lower_filter(state: &State, value: &Value) -> Result<Value, Error> {
    let text = string_filter(value)?;
    Ok(minijinja::filters::lower(StringInput::new(state, &text)?))
}

/// Jinja2's `capitalize`: `str(value).capitalize()`, the first character upper-cased and the
/// rest lower-cased in the context of the whole string, so a final `Σ` becomes `ς` (`'aΣ.'` is
/// `Aς.`). Python title-cases the first character where Rust can only upper-case it
/// (`'ß'`: Python `Ss`, here `SS`); the standard library has no titlecase mapping.
pub(crate) fn capitalize_filter(value: &Value) -> Result<Value, Error> {
    let text = py_str(value)?;
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return Ok(Value::from(""));
    };
    let lower = text.to_lowercase();
    let mut out: String = first.to_uppercase().collect();
    out.extend(lower.chars().skip(first.to_lowercase().count()));
    Ok(Value::from(out))
}

/// Jinja2's `trim`: `str(value).strip(chars)`, which without `chars` strips what
/// [`is_py_space`] calls whitespace.
pub(crate) fn trim_filter(value: &Value, chars: Option<String>) -> Result<Value, Error> {
    let text = py_str(value)?;
    let trimmed = match chars {
        Some(chars) => text.trim_matches(|c| chars.contains(c)),
        None => text.trim_matches(is_py_space),
    };
    Ok(Value::from(trimmed))
}

/// Python's `str.isspace` for one character (also what its regex `\s` matches): the White_Space
/// property `char::is_whitespace` tests, plus the separators U+001C..=U+001F, which Python counts
/// for their bidirectional class.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

/// See [`upper_filter`]; Jinja2 also takes `str()` of what to replace and its replacement.
pub(crate) fn replace_filter(
    state: &State,
    value: &Value,
    from: &Value,
    to: &Value,
) -> Result<Value, Error> {
    let (text, from, to) = (
        string_filter(value)?,
        string_filter(from)?,
        string_filter(to)?,
    );
    minijinja::filters::replace(
        state,
        StringInput::new(state, &text)?,
        StringInput::new(state, &from)?,
        StringInput::new(state, &to)?,
    )
}

/// Jinja2's `title` (`do_title`): `str(value)` split before each run of `-`, whitespace, `(`,
/// `{`, `[` or `<`, and each piece upper-cased in its first character and lower-cased in the
/// rest. minijinja's own `title` also starts a word after any ASCII punctuation (`It'S`).
///
/// The rest of a piece is lower-cased as one string, as Python does, so a final `Σ` becomes `ς`.
pub(crate) fn title_filter(value: &Value) -> Result<Value, Error> {
    let text = py_str(value)?;
    let is_separator = |c: char| matches!(c, '-' | '(' | '{' | '[' | '<') || is_py_space(c);
    let mut out = String::with_capacity(text.len());
    let mut rest = text.as_str();
    while let Some(first) = rest.chars().next() {
        if is_separator(first) {
            let end = rest.find(|c| !is_separator(c)).unwrap_or(rest.len());
            out.push_str(&rest[..end]);
            rest = &rest[end..];
            continue;
        }
        let end = rest.find(is_separator).unwrap_or(rest.len());
        let (word, tail) = rest.split_at(end);
        let after = first.len_utf8();
        out.extend(first.to_uppercase());
        out.push_str(&word[after..].to_lowercase());
        rest = tail;
    }
    Ok(Value::from(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`is_py_space`] is exactly Python 3.15's `str.isspace`: every character it is true for,
    /// as `[hex(c) for c in range(0x110000) if chr(c).isspace()]` listed them.
    #[test]
    fn py_space_is_pythons_isspace() {
        let python: Vec<u32> = [
            0x9, 0xa, 0xb, 0xc, 0xd, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000,
            0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028,
            0x2029, 0x202f, 0x205f, 0x3000,
        ]
        .to_vec();
        let ours: Vec<u32> = (0..=0x10_FFFF)
            .filter_map(char::from_u32)
            .filter(|&c| is_py_space(c))
            .map(u32::from)
            .collect();
        assert_eq!(ours, python);
    }

    /// Python's `repr(float)` at the notation thresholds and the edges of the range. Each expected
    /// string is what Python 3.15 printed for `repr(x)`.
    #[test]
    fn float_repr_matches_python() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (2.0, "2.0"),
            (2.5, "2.5"),
            (0.1 + 0.2, "0.30000000000000004"),
            (1e-4, "0.0001"),
            (9.999999999999999e-05, "9.999999999999999e-05"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (9999999999999998.0, "9999999999999998.0"),
            (1e16, "1e+16"),
            (1e22, "1e+22"),
            (1e100, "1e+100"),
            (-1.5e-7, "-1.5e-07"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (1.0 / 3.0, "0.3333333333333333"),
            // Ties: two shortest digit strings are equally close, and Python keeps the even one.
            (1125899906842624.0 + 0.25, "1125899906842624.2"),
            (1324350514534.0 + 0.90625, "1324350514534.9062"),
            (73637923426838.0 + 0.625, "73637923426838.62"),
            (-(1125899906842624.0 + 0.25), "-1125899906842624.2"),
            (f64::NAN, "nan"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ];
        for &(x, want) in cases {
            assert_eq!(float_repr(x), want, "{x:?}");
        }
    }
}
