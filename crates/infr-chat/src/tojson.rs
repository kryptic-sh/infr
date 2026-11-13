//! `tojson` as llama.cpp's jinja engine writes it (`tojson` and `value_to_json_internal` in
//! `common/jinja/value.cpp`), which is Python's `json.dumps` as the Hugging Face chat-template
//! environment calls it (`ensure_ascii=False`).
//!
//! minijinja's own filter writes compact JSON and HTML-escapes `<`, `>`, `&` and `'`; the
//! reference writes `", "` / `": "` separators and leaves those characters alone, so every tool
//! block a Qwen-family template printed differed from llama.cpp's in its spaces.

use std::fmt::Write;

use minijinja::value::{from_args, Kwargs, Rest, ValueKind};
use minijinja::{Error, ErrorKind, Value};

use crate::template::check_width;

/// Deepest nesting the writer descends into before refusing, as the reference does: llama.cpp's
/// `JINJA_RECURSIVE_LIMIT` (`common/jinja/utils.h`). It also keeps a deep value from overflowing
/// the stack.
const NESTING_MAX: usize = 128;

/// The reference's separators when the template names none: Python's `json.dumps` defaults, with
/// the item separator losing its space once `indent` breaks the lines.
const ITEM_SEP: &str = ", ";
const ITEM_SEP_INDENTED: &str = ",";
const KEY_SEP: &str = ": ";

/// How one `tojson` call lays out its output.
struct Layout {
    /// `None` writes one line; `Some(n)` breaks after every item and indents each level by `n`.
    indent: Option<usize>,
    item_sep: String,
    key_sep: String,
    ensure_ascii: bool,
}

/// The `tojson` filter. On an undefined value it fails, as the reference's filter does (it has no
/// `tojson` for that type); everything else is [`tojson_function`].
pub(crate) fn tojson_filter(value: &Value, args: Rest<Value>) -> Result<Value, Error> {
    if value.is_undefined() {
        return Err(Error::new(
            ErrorKind::UndefinedError,
            "tojson: the value is undefined",
        ));
    }
    tojson_function(value, args)
}

/// The global `tojson(x, ...)` function, with the reference's arguments: `ensure_ascii`,
/// `indent`, `separators` and `sort_keys`, by keyword or in that order by position (the order of
/// Hugging Face's `tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)`).
/// An undefined value writes `null` here, as the reference's function does.
pub(crate) fn tojson_function(value: &Value, args: Rest<Value>) -> Result<Value, Error> {
    // Taken from `Rest`: a native minijinja function has at most five parameters.
    type Args = (
        Option<Value>,
        Option<Value>,
        Option<Value>,
        Option<Value>,
        Kwargs,
    );
    let (ensure_ascii, indent, separators, sort_keys, kwargs): Args = from_args(&args)?;
    let arg = |pos: Option<Value>, name: &str| -> Result<Value, Error> {
        Ok(match kwargs.get::<Option<Value>>(name)? {
            Some(v) => v,
            None => pos.unwrap_or(Value::UNDEFINED),
        })
    };
    let ensure_ascii = arg(ensure_ascii, "ensure_ascii")?.is_true();
    let indent = arg(indent, "indent")?;
    let separators = arg(separators, "separators")?;
    let sort_keys = arg(sort_keys, "sort_keys")?.is_true();
    // Other keyword arguments are ignored, not refused: the reference's leniency, copied so a
    // template llama.cpp renders does not fail here.
    if sort_keys {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "tojson: sort_keys=true is not supported (llama.cpp's jinja engine refuses it too)",
        ));
    }
    // Only an integer indents (a bool or a string does not, there); a negative one writes one line.
    let indent = if indent.is_integer() {
        match i128::try_from(indent)? {
            n if n < 0 => None,
            n => {
                let n = usize::try_from(n).unwrap_or(usize::MAX);
                check_width("tojson indent", n)?;
                Some(n)
            }
        }
    } else {
        None
    };
    let separators: Vec<String> = if separators.kind() == ValueKind::Seq {
        separators.try_iter()?.map(|s| s.to_string()).collect()
    } else {
        Vec::new()
    };
    let default_item_sep = if indent.is_some() {
        ITEM_SEP_INDENTED
    } else {
        ITEM_SEP
    };
    let layout = Layout {
        indent,
        item_sep: separators
            .first()
            .cloned()
            .unwrap_or_else(|| default_item_sep.to_owned()),
        key_sep: separators
            .get(1)
            .cloned()
            .unwrap_or_else(|| KEY_SEP.to_owned()),
        ensure_ascii,
    };
    let mut out = String::new();
    write_value(&mut out, value, 0, &layout)?;
    Ok(Value::from_safe_string(out))
}

/// Append `value` at nesting `level`. Anything JSON has no form for (a macro, bytes, an undefined
/// inside a container) is written `null`, as the reference writes it.
fn write_value(
    out: &mut String,
    value: &Value,
    level: usize,
    layout: &Layout,
) -> Result<(), Error> {
    match value.kind() {
        ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number if value.is_integer() => write!(out, "{value}").expect("String write"),
        ValueKind::Number => out.push_str(&python_float_repr(f64::try_from(value.clone())?)),
        ValueKind::String => write_string(out, value.as_str().unwrap_or_default(), layout),
        ValueKind::Seq | ValueKind::Iterable | ValueKind::Map => {
            if level >= NESTING_MAX {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("tojson: value is nested more than {NESTING_MAX} levels deep"),
                ));
            }
            let is_map = value.kind() == ValueKind::Map;
            let (open, close) = if is_map { ('{', '}') } else { ('[', ']') };
            out.push(open);
            let items: Vec<Value> = value.try_iter()?.collect();
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&layout.item_sep);
                }
                line_break(out, level + 1, layout);
                if is_map {
                    // A key that is not a string is written as the reference names it:
                    // Python's spelling for a bool or none, the float repr for a float.
                    let key = match item.kind() {
                        ValueKind::String => item.as_str().unwrap_or_default().to_owned(),
                        ValueKind::Bool if item.is_true() => "True".to_owned(),
                        ValueKind::Bool => "False".to_owned(),
                        ValueKind::None => "None".to_owned(),
                        ValueKind::Number if !item.is_integer() => {
                            python_float_repr(f64::try_from(item.clone())?)
                        }
                        _ => item.to_string(),
                    };
                    write_string(out, &key, layout);
                    out.push_str(&layout.key_sep);
                    write_value(out, &value.get_item(item)?, level + 1, layout)?;
                } else {
                    write_value(out, item, level + 1, layout)?;
                }
            }
            if !items.is_empty() {
                line_break(out, level, layout);
            }
            out.push(close);
        }
        _ => out.push_str("null"),
    }
    Ok(())
}

/// With an indent, a newline and `level` steps of it; without, nothing.
fn line_break(out: &mut String, level: usize, layout: &Layout) {
    if let Some(n) = layout.indent {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', n * level));
    }
}

/// A JSON string literal. serde_json escapes the set the reference does (`"`, `\`, `\b`, `\f`,
/// `\n`, `\r`, `\t`, other control characters as lowercase `\u00XX`, nothing else); with
/// `ensure_ascii` everything past ASCII then becomes UTF-16 `\uXXXX` escapes.
fn write_string(out: &mut String, s: &str, layout: &Layout) {
    let quoted = serde_json::to_string(s).expect("a str always serializes");
    if !layout.ensure_ascii {
        out.push_str(&quoted);
        return;
    }
    for c in quoted.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            for unit in c.encode_utf16(&mut [0; 2]) {
                write!(out, "\\u{unit:04x}").expect("String write");
            }
        }
    }
}

/// `x` as Python's `repr(float)` writes it, which `json.dumps` and the reference's `float_repr`
/// follow: the shortest digits that read back as `x`, in fixed notation when the decimal point
/// falls at `-4 < point <= 16` (a whole number gets `.0`), otherwise as `d.ddde±XX`. `NaN` and
/// `Infinity` are what `json.dumps` writes for the non-finite values.
fn python_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_owned();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-Infinity" } else { "Infinity" }.to_owned();
    }
    // Rust's `{:e}` writes the same shortest round-trip digits, as `d[.ddd]e<exp>`.
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` writes an exponent");
    let exp: i32 = exp.parse().expect("`{:e}` writes an integer exponent");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let n = digits.len() as i32;
    let point = exp + 1; // digits before the decimal point
    let mut out = String::from(if x.is_sign_negative() { "-" } else { "" });
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
        let sign = if exp < 0 { '-' } else { '+' };
        write!(out, "e{sign}{:02}", exp.abs()).expect("String write");
    }
    out
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use crate::render_template;

    /// Render `src` with `messages` = `msgs` and nothing else of note.
    fn render_with(src: &str, msgs: Vec<Value>) -> Result<String, minijinja::Error> {
        render_template(src, msgs, Value::Null, "", "", false, true)
    }

    fn render(src: &str) -> Result<String, minijinja::Error> {
        render_with(src, vec![])
    }

    /// Render each `(template, want)` and list every mismatch (or error) in one failure.
    fn assert_renders(cases: &[(&str, &str)]) {
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|&(src, want)| {
                let got = render(src).unwrap_or_else(|e| format!("error: {e:#}"));
                (got != want).then(|| format!("{src}\n  got  {got:?}\n  want {want:?}"))
            })
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// Every expected string here is what llama.cpp's `test-chat-template --no-common` (at
    /// 5676dfbe0) wrote for the same template and variables.
    #[test]
    fn tojson_matches_the_reference() {
        let cases = [
            // Python's `json.dumps` separators, keys in insertion order.
            (
                "{{ {'b': 1, 'a': [1, 2], 'c': {}, 'd': []} | tojson }}",
                r#"{"b": 1, "a": [1, 2], "c": {}, "d": []}"#,
            ),
            // `indent=` drops the space after the item separator; empty containers stay inline.
            (
                "{{ {'b': 1, 'a': [1, 2], 'c': {}, 'd': []} | tojson(indent=2) }}",
                "{\n  \"b\": 1,\n  \"a\": [\n    1,\n    2\n  ],\n  \"c\": {},\n  \"d\": []\n}",
            ),
            // `indent=0` still breaks lines.
            (
                "{{ {'b': 1, 'a': [1, 2]} | tojson(indent=0) }}",
                "{\n\"b\": 1,\n\"a\": [\n1,\n2\n]\n}",
            ),
            // Only an integer indents: a bool or a negative number leaves the one-line form.
            (
                "{{ {'b': 1, 'a': [1, 2]} | tojson(indent=true) }}",
                r#"{"b": 1, "a": [1, 2]}"#,
            ),
            (
                "{{ {'b': 1, 'a': [1, 2]} | tojson(indent=-1) }}",
                r#"{"b": 1, "a": [1, 2]}"#,
            ),
            (
                "{{ {'a': 1} | tojson(ensure_ascii=false, indent=none) }}",
                r#"{"a": 1}"#,
            ),
            // `separators=(item, key)`.
            (
                "{{ {'b': 1, 'a': [1, 2]} | tojson(separators=(',', ':')) }}",
                r#"{"b":1,"a":[1,2]}"#,
            ),
            (
                "{{ {'b': 1, 'a': [1, 2]} | tojson(indent=1, separators=(';', '=')) }}",
                "{\n \"b\"=1;\n \"a\"=[\n  1;\n  2\n ]\n}",
            ),
            // Non-string keys are written as strings; none and booleans as JSON literals.
            (
                "{{ {1: 2, 'x': none, 'y': true, 'z': false} | tojson }}",
                r#"{"1": 2, "x": null, "y": true, "z": false}"#,
            ),
            // `tojson` is a global function too.
            ("{{ tojson({'b': 1}) }}", r#"{"b": 1}"#),
        ];
        assert_renders(&cases);
    }

    /// Strings: raw UTF-8 and no HTML escaping by default (Jinja2's own `tojson` and minijinja's
    /// escape `<` as `\u003c`; the reference does not); `ensure_ascii` (keyword, or the first
    /// positional argument) writes UTF-16 escapes. Expected bytes from the reference tool.
    #[test]
    fn tojson_strings_match_the_reference() {
        let text = "é😀<>&'\"\\/\u{1}\u{1f}\u{7f}\u{8}\u{c}\n\r\t";
        let msgs = vec![json!({"role": "user", "content": text})];
        let plain = "\"é😀<>&'\\\"\\\\/\\u0001\\u001f\u{7f}\\b\\f\\n\\r\\t\"";
        let ascii = "\"\\u00e9\\ud83d\\ude00<>&'\\\"\\\\/\\u0001\\u001f\u{7f}\\b\\f\\n\\r\\t\"";
        for (src, want) in [
            ("{{ messages[0].content | tojson }}", plain),
            (
                "{{ messages[0].content | tojson(ensure_ascii=true) }}",
                ascii,
            ),
            ("{{ messages[0].content | tojson(true) }}", ascii),
        ] {
            assert_eq!(render_with(src, msgs.clone()).unwrap(), want, "{src}");
        }
    }

    /// Floats are written as Python's `repr` writes them (shortest round trip, `.0` on whole
    /// numbers, exponent form outside `1e-4 <= |x| < 1e16` with a signed two-digit exponent);
    /// integers stay integers. Expected string from the reference tool.
    #[test]
    fn tojson_numbers_match_the_reference() {
        let nums = json!([
            1.0,
            0.1,
            0.3333333333333333,
            1e16,
            1e15,
            0.0001,
            0.00001,
            -0.0,
            1.5e300,
            5e-324,
            123456789012345678.0,
            -7,
            2.5
        ]);
        let out = render_with(
            "{{ messages[0].content | tojson }}",
            vec![json!({"role": "user", "content": nums})],
        )
        .unwrap();
        assert_eq!(
            out,
            "[1.0, 0.1, 0.3333333333333333, 1e+16, 1000000000000000.0, 0.0001, 1e-05, -0.0, \
             1.5e+300, 5e-324, 1.2345678901234568e+17, -7, 2.5]"
        );
    }

    /// Where the reference is more lenient than Jinja2, or names keys its own way. Expected strings
    /// from the reference tool.
    #[test]
    fn tojson_leniencies_match_the_reference() {
        let cases = [
            // Bool and none keys are spelled as Python spells them.
            (
                "{{ {true: 1, none: 2, false: 3} | tojson }}",
                r#"{"True": 1, "None": 2, "False": 3}"#,
            ),
            // An unknown keyword argument is ignored.
            ("{{ {'a': 1} | tojson(foo=1) }}", r#"{"a": 1}"#),
            // The function form writes an undefined value as null (the filter form refuses it).
            ("{{ tojson(missing) }}", "null"),
            ("{{ tojson(missing, indent=2) }}", "null"),
            ("{{ [1, missing] | tojson }}", "[1, null]"),
        ];
        assert_renders(&cases);
    }

    /// What the reference refuses, infr refuses: an undefined value, and `sort_keys=true` (which
    /// llama.cpp's engine does not implement).
    #[test]
    fn tojson_refuses_what_the_reference_refuses() {
        for src in [
            "{{ missing | tojson }}",
            "{{ {'b': 1, 'a': 2} | tojson(sort_keys=true) }}",
        ] {
            assert!(render(src).is_err(), "{src} rendered");
        }
    }
}
