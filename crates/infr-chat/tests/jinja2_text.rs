//! Chat templates turn values into text as Python's Jinja2 does: Hugging Face's
//! `apply_chat_template` renders with it, so it is what a model was trained on. Every row of
//! `fixtures/jinja2_text.json` is a template, the variables it is rendered with, and what Jinja2
//! printed (or that it raised); `fixtures/jinja2_text.py` generated it, and says how to regenerate.
//! Each row is rendered here through `render_template`, so no Python is needed at test time.

use infr_chat::render_template;
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/jinja2_text.json");

/// How the generator hands a row its variables: they travel as `tools`, read as `v`.
const VARS_PREFIX: &str = "{% set v = tools %}";

/// Rows where infr does not print what Jinja2 prints, with what infr prints today (`None`: it
/// refuses the template). Each is asserted as it stands, so a change to any of them is noticed:
/// a row that starts matching Python fails here until its entry is deleted.
const KNOWN_GAPS: &[(&str, Option<&str>)] = &[
    // `~` concatenates with minijinja's own `Display` (`ops::string_concat` in its
    // `value/ops.rs`), which has no hook. A list or dict from the request prints as Python's
    // `repr` there (`src/pyvalue.rs`); a float cannot be wrapped that way, since it must stay a
    // number: outside `1e-4 <= |x| < 1e16` it is written in fixed notation, `nan` as `NaN`, and a
    // tie rounds up (Rust's `{}`). A list or dict the template builds itself is written
    // JSON-style.
    ("{{ 'p' ~ 1e16 }}", Some("p10000000000000000.0")),
    ("{{ 1e16 ~ 'p' }}", Some("10000000000000000.0p")),
    ("{{ 'p' ~ 1.5e-7 }}", Some("p0.00000015")),
    ("{{ 1.5e-7 ~ 'p' }}", Some("0.00000015p")),
    ("{{ 'p' ~ v.tiny }}", Some("p0.0000001")),
    ("{{ v.tiny ~ 'p' }}", Some("0.0000001p")),
    ("{{ 'p' ~ v.smaller }}", Some("p0.00001")),
    ("{{ v.smaller ~ 'p' }}", Some("0.00001p")),
    ("{{ 'p' ~ v.e16 }}", Some("p10000000000000000.0")),
    ("{{ v.e16 ~ 'p' }}", Some("10000000000000000.0p")),
    ("{{ 'p' ~ v.e22 }}", Some("p10000000000000000000000.0")),
    ("{{ v.e22 ~ 'p' }}", Some("10000000000000000000000.0p")),
    ("{{ 'p' ~ v.tie1 }}", Some("p1125899906842624.3")),
    ("{{ v.tie1 ~ 'p' }}", Some("1125899906842624.3p")),
    ("{{ 'p' ~ v.tie2 }}", Some("p1324350514534.9063")),
    ("{{ v.tie2 ~ 'p' }}", Some("1324350514534.9063p")),
    ("{{ 'p' ~ v.tie3 }}", Some("p73637923426838.63")),
    ("{{ v.tie3 ~ 'p' }}", Some("73637923426838.63p")),
    ("{{ 'p' ~ ('nan' | float) }}", Some("pNaN")),
    (
        "{{ 'p' ~ [1, 'a', true, none] }}",
        Some(r#"p[1, "a", True, None]"#),
    ),
    (
        "{{ [1, 'a', true, none] ~ 'p' }}",
        Some(r#"[1, "a", True, None]p"#),
    ),
    (
        "{{ 'p' ~ {'a': 1, 'b': [none]} }}",
        Some(r#"p{"a": 1, "b": [None]}"#),
    ),
    (
        "{{ {'a': 1, 'b': [none]} ~ 'p' }}",
        Some(r#"{"a": 1, "b": [None]}p"#),
    ),
    // A list or dict the template builds itself (`| list`, a slice, `+`, `dict(...)`) is
    // minijinja's own value, which `~` writes JSON-style: only the request's containers are
    // wrapped (`src/pyvalue.rs`).
    ("{{ (v.ptype | list) ~ '' }}", Some(r#"["string", "null"]"#)),
    ("{{ v.ptype[0:1] ~ '' }}", Some(r#"["string"]"#)),
    (
        "{{ (v.ptype + ['x']) ~ '' }}",
        Some(r#"["string", "null", "x"]"#),
    ),
    ("{{ dict(a=none) ~ '' }}", Some(r#"{"a": None}"#)),
    // Python title-cases the first character in `capitalize`; the standard library can only
    // upper-case it and has no titlecase table.
    ("{{ 'ß' | capitalize }}", Some("SS")),
    ("{{ 'ǆ' | capitalize }}", Some("Ǆ")),
    // minijinja's `map` gives a list and its slice a lazy iterable, where Python has an iterator
    // and a list: `is sequence` answers the other way round for these two.
    (
        "{{ (v.list | reverse) is sequence }}|{{ (v.list | map('string')) is sequence }}",
        Some("False|True"),
    ),
    (
        "{{ undefined_name is sequence }}|{{ ('a b'.split())[1:] is sequence }}|{{ range(2) is iterable }}",
        Some("True|False|True"),
    ),
    // `{% generation %}`, transformers' tag: minijinja has no custom tags (B172).
    (
        "{% set x = 1 %}{% generation %}{% set x = 2 %}in={{ x }}{% endgeneration %} out={{ x }}",
        None,
    ),
    // minijinja's formatter applies a spec its own way: a bool under `,` stays a bool (Python
    // formats it as the integer), and a precision on an integer is accepted (Python refuses it).
    ("{{ '{:,}'.format(false) }}", Some("False")),
    ("{{ '{:.3}'.format(5) }}", Some("5")),
    // minijinja has no tuple: a tuple literal, `dictsort` and `items` give lists; and no range
    // type: `range(3)` is a list.
    ("{{ (1, 2) }}", Some("[1, 2]")),
    ("{{ (1,) }}", Some("[1]")),
    ("{{ range(3) }}", Some("[0, 1, 2]")),
    (
        "{{ {'b': 1, 'a': 2} | dictsort }}",
        Some("[['a', 2], ['b', 1]]"),
    ),
    (
        "{{ v.obj | dictsort }}",
        Some(r#"[['a', "it's"], ['b', {'c': [1, 'x', True, None, 2.5]}], ['z', {}]]"#),
    ),
    (
        "{{ v.args.filters | items | list }}",
        Some("[['min', 1.0], ['tags', ['a', 'b']]]"),
    ),
    // An unassigned code point (category Cn) is written raw inside a container; Python escapes
    // it. See `is_printable` in `src/pystr.rs`.
    ("{{ [v.unassigned] }}", Some("['\u{378}']")),
    // minijinja's formatter (`format_utils.rs`) has no `r` conversion and no `!` conversions.
    ("{{ '%r' | format('x') }}", None),
    ("{{ '{!r}'.format('x') }}", None),
    ("{{ '{!s}'.format(none) }}", None),
    // A float in a field with a spec but no type or precision: minijinja's formatter writes `%g`'s
    // six significant digits, Python writes `repr`. Only an argument that meets nothing but
    // spec-less fields is converted first (`src/pyformat.rs`).
    (
        "{{ '{:>20}'.format(0.1 + 0.2) }}",
        Some("                 0.3"),
    ),
    // A deliberate limit: a value nested deeper than `tojson::NESTING_MAX` is refused rather than
    // walked (it would otherwise recurse on the stack as deep as the template likes).
    (
        "{% set ns = namespace(x=[]) %}{% for i in range(200) %}{% set ns.x = [ns.x] %}{% endfor %}{{ ns.x | string | length }}",
        None,
    ),
    // Also deliberate, and minijinja's: a string repeated past 100,000,000 bytes
    // (`MAX_REPEATED_STRING_LEN` in its `value/ops.rs`) is refused, bounding what one `*` of an
    // untrusted template can allocate.
    ("{{ ('x' * 100000001) | length }}", None),
    // Not a conversion to text; found by this table and listed in `docs/backlog.md`.
    // minijinja parses `-v.e16` as `(-v).e16`.
    ("{{ -v.e16 }}", None),
    ("{{ 2 ** -1 }}", None),
    ("{{ 1 / 0 }}", Some("inf")),
    ("{{ 1 < ('nan' | float) }}", Some("True")),
    ("{{ 'abc' | float }}", None),
    ("{{ 'abc' | int }}", None),
    ("{{ 2.5 | round }}", Some("3.0")),
    ("{{ 2.5 | round(0, 'floor') }}", None),
    ("{{ true | center(8) }}", None),
    ("{{ none | wordcount }}", None),
    ("{{ true | indent(2) }}", Some("True")),
];

/// Whole templates from `fixtures/`, rendered by the generator through Jinja2 (its
/// `TEMPLATE_RENDERS`, the fixture's `renders`) and here through infr with the same variables.
/// Qwen3.5 prints each scalar argument of a replayed tool call with `| string`: a bool, none, an
/// exponent-range float and a tie float are where infr used to differ.
#[test]
fn whole_templates_render_as_python_jinja2_renders_them() {
    let doc: Value = serde_json::from_str(FIXTURE).expect("fixture parses");
    let renders = doc["renders"].as_array().expect("renders");
    assert!(!renders.is_empty());
    for case in renders {
        let name = case["template"].as_str().expect("template name");
        let source = match name {
            "qwen35_chat_template.jinja" => include_str!("fixtures/qwen35_chat_template.jinja"),
            other => panic!("no include for {other}"),
        };
        let messages = case["messages"].as_array().expect("messages").clone();
        let got = render_template(
            source,
            messages,
            case["tools"].clone(),
            "<s>",
            "</s>",
            true,
            true,
        )
        .expect("renders");
        assert_eq!(got, case["python"].as_str().expect("python text"), "{name}");
    }
}

/// What a row expects: Jinja2's text, or `None` when Jinja2 raised.
fn python(row: &Value) -> Option<&str> {
    row.get("python").map(|p| p.as_str().expect("text"))
}

#[test]
fn values_become_text_as_python_jinja2_writes_them() {
    let doc: Value = serde_json::from_str(FIXTURE).expect("fixture parses");
    let vars = doc["vars"].clone();
    let rows = doc["rows"].as_array().expect("rows");
    assert!(rows.len() > 600, "only {} rows", rows.len());
    let mut wrong = Vec::new();
    let mut gaps_seen = 0;
    for row in rows {
        let template = row["template"].as_str().expect("template");
        let expr = template.strip_prefix(VARS_PREFIX).unwrap_or(template);
        let got = render_template(template, vec![], vars.clone(), "", "", false, true).ok();
        let want = python(row);
        match KNOWN_GAPS.iter().find(|(gap, _)| *gap == expr) {
            Some(&(_, today)) => {
                gaps_seen += 1;
                if got.as_deref() != today {
                    wrong.push(format!(
                        "{expr}\n  known gap changed: got {got:?}, recorded {today:?}"
                    ));
                } else if got.as_deref() == want {
                    wrong.push(format!(
                        "{expr}\n  matches Python now: delete it from KNOWN_GAPS"
                    ));
                }
            }
            None if got.as_deref() != want => {
                wrong.push(format!("{expr}\n  got  {got:?}\n  want {want:?}"));
            }
            None => {}
        }
    }
    assert_eq!(
        gaps_seen,
        KNOWN_GAPS.len(),
        "a KNOWN_GAPS entry names no fixture row"
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
