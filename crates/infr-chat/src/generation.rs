//! Hugging Face's `{% generation %}` tag, which minijinja has no way to register.
//!
//! transformers adds the tag to the environment it renders chat templates with (`AssistantTracker`
//! in `transformers/utils/chat_template_utils.py`). It parses the body up to `{% endgeneration %}`
//! into a `CallBlock`: the body runs as a macro's would, in a scope of its own, and its text is
//! returned unchanged (the tracker only records where it landed when an assistant mask is asked
//! for). minijinja's parser has a fixed set of statements and no hook for another one, so
//! [`rewrite_generation_tags`] turns each tag into `{% with %}` / `{% endwith %}`, which scopes a
//! `{% set %}` in its body the same way, before the template is compiled. A `with` is not a macro
//! body, though: `break` / `continue` inside the tag, and `caller`, `varargs` and `kwargs` read
//! there, behave as in the enclosing block, where Jinja2 refuses the first two and gives the body
//! its own values for the rest (`docs/backlog.md`, B187). A template Hugging Face renders is
//! unaffected.
//!
//! The rewrite finds tags with the default delimiters (infr never configures others): comments
//! are skipped from `{#` to the first `#}` as Jinja2 does, `{% raw %}` blocks are skipped whole,
//! and a tag ends at the first `%}` (or `}}`) outside a string literal and outside brackets. Only the statement name is
//! replaced, padded with spaces to its old length, so whitespace control (`{%-`, `{%+`, `-%}`,
//! `+%}`), lines and columns are all left as they were.

use std::borrow::Cow;

/// A block the rewrite tracks: a `{% generation %}` turns into a `{% with %}`, so the two must
/// close in the order they opened for the rewritten template to fail where Python's does.
#[derive(Clone, Copy, PartialEq)]
enum Open {
    With,
    Generation,
}

/// `template` with every `{% generation %}` / `{% endgeneration %}` written as `{% with %}` /
/// `{% endwith %}`; borrowed unchanged when it has none.
///
/// Refused, as Jinja2 refuses them: a tag with anything but whitespace control between its name
/// and its end, an `endgeneration` that closes nothing or closes a `with`, an `endwith` that
/// closes a `generation`, and a `generation` left open. Every other error is left to minijinja,
/// which sees the same template text in the same places.
pub(crate) fn rewrite_generation_tags(template: &str) -> Result<Cow<'_, str>, minijinja::Error> {
    let bytes = template.as_bytes();
    let mut out: Option<String> = None;
    // How far `template` has been copied into `out`.
    let mut copied = 0;
    let mut open: Vec<(Open, usize)> = Vec::new();
    let mut pos = 0;
    while let Some(found) = template[pos..].find('{') {
        let start = pos + found;
        let Some(&kind) = bytes.get(start + 1) else {
            break;
        };
        match kind {
            b'#' => match template[start + 2..].find("#}") {
                Some(end) => pos = start + 2 + end + 2,
                // An unclosed comment: minijinja reports it.
                None => break,
            },
            b'{' => match tag_end(template, start + 2, "}}") {
                Some(end) => pos = end,
                None => break,
            },
            b'%' => {
                let mut name_at = start + 2;
                if matches!(bytes.get(name_at), Some(b'-' | b'+')) {
                    name_at += 1;
                }
                name_at += leading_ws(&template[name_at..]);
                let name_len = template[name_at..]
                    .find(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .unwrap_or(template.len() - name_at);
                let name = &template[name_at..name_at + name_len];
                if name == "raw" {
                    // A `raw` that is not a bare tag, or is never closed, is an error minijinja
                    // reports; stopping here also keeps the scan linear in the template's length.
                    let Some(end) = raw_end(template, name_at + name_len) else {
                        break;
                    };
                    pos = end;
                    continue;
                }
                let Some(end) = tag_end(template, name_at + name_len, "%}") else {
                    break;
                };
                pos = end;
                let replacement = match name {
                    "with" => {
                        open.push((Open::With, start));
                        None
                    }
                    "endwith" => {
                        if let Some(&(Open::Generation, opened)) = open.last() {
                            return Err(nesting_error(template, start, "endwith", opened));
                        }
                        open.pop();
                        None
                    }
                    "generation" => {
                        open.push((Open::Generation, start));
                        Some("with")
                    }
                    "endgeneration" => match open.pop() {
                        Some((Open::Generation, _)) => Some("endwith"),
                        Some((Open::With, opened)) => {
                            return Err(nesting_error(template, start, "endgeneration", opened))
                        }
                        None => {
                            return Err(syntax_error(format!(
                                "`endgeneration` on line {} closes no `generation` block",
                                line_of(template, start)
                            )))
                        }
                    },
                    _ => None,
                };
                let Some(replacement) = replacement else {
                    continue;
                };
                if !is_bare_tag_tail(&template[name_at + name_len..end - 2]) {
                    return Err(syntax_error(format!(
                        "`{name}` on line {} takes no arguments",
                        line_of(template, start)
                    )));
                }
                let out = out.get_or_insert_with(|| String::with_capacity(template.len()));
                out.push_str(&template[copied..name_at]);
                let width = name.len();
                out.push_str(&format!("{replacement:<width$}"));
                copied = name_at + name_len;
            }
            _ => pos = start + 1,
        }
    }
    if let Some(&(_, opened)) = open.iter().rev().find(|(o, _)| *o == Open::Generation) {
        return Err(syntax_error(format!(
            "`generation` on line {} is never closed by `endgeneration`",
            line_of(template, opened)
        )));
    }
    Ok(match out {
        Some(mut out) => {
            out.push_str(&template[copied..]);
            Cow::Owned(out)
        }
        None => Cow::Borrowed(template),
    })
}

/// Where the tag whose body starts at `from` ends (just past `end`): the first `end` outside a
/// string literal and outside brackets, as minijinja's lexer reads it. `None` when it never ends.
fn tag_end(template: &str, from: usize, end: &str) -> Option<usize> {
    let bytes = template.as_bytes();
    // Signed, like the lexer's: an unmatched closer leaves it below zero and the tag unended.
    let mut depth: isize = 0;
    let mut i = from;
    while i < bytes.len() {
        match bytes[i] {
            quote @ (b'\'' | b'"') => {
                i += 1;
                while i < bytes.len() && bytes[i] != quote {
                    i += if bytes[i] == b'\\' { 2 } else { 1 };
                }
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' => depth -= 1,
            b'}' | b'%' if depth == 0 && template[i..].starts_with(end) => {
                return Some(i + end.len())
            }
            b'}' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    None
}

/// Past the `{% endraw %}` that closes a `{% raw %}` whose name ends at `after_name`, or `None`
/// when the tag is not a bare `raw` (the lexer then reads it as a statement, which does not exist)
/// or is never closed.
fn raw_end(template: &str, after_name: usize) -> Option<usize> {
    let mut pos = after_name + leading_ws(&template[after_name..]);
    if template[pos..].starts_with(['-', '+']) {
        pos += 1;
    }
    template[pos..].strip_prefix("%}")?;
    pos += 2;
    while let Some(found) = template[pos..].find("{%") {
        let mut at = pos + found + 2;
        if template[at..].starts_with(['-', '+']) {
            at += 1;
        }
        at += leading_ws(&template[at..]);
        if let Some(rest) = template[at..].strip_prefix("endraw") {
            let mut tail = rest.trim_start_matches(|c: char| c.is_ascii_whitespace());
            tail = tail.strip_prefix(['-', '+']).unwrap_or(tail);
            if let Some(after) = tail.strip_prefix("%}") {
                return Some(template.len() - after.len());
            }
        }
        pos = at;
    }
    None
}

/// Whether what follows a tag's name, up to its `%}`, is only whitespace and an optional
/// whitespace-control marker: the `generation` tags take nothing else.
fn is_bare_tag_tail(tail: &str) -> bool {
    let tail = tail.strip_suffix(['-', '+']).unwrap_or(tail);
    tail.bytes().all(|b| b.is_ascii_whitespace())
}

fn leading_ws(s: &str) -> usize {
    s.len()
        - s.trim_start_matches(|c: char| c.is_ascii_whitespace())
            .len()
}

/// 1-based line of byte offset `at`.
fn line_of(template: &str, at: usize) -> usize {
    template[..at].matches('\n').count() + 1
}

fn nesting_error(template: &str, at: usize, closer: &str, opened: usize) -> minijinja::Error {
    syntax_error(format!(
        "`{closer}` on line {} does not close the block opened on line {}",
        line_of(template, at),
        line_of(template, opened)
    ))
}

fn syntax_error(detail: String) -> minijinja::Error {
    minijinja::Error::new(minijinja::ErrorKind::SyntaxError, detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(src: &str) -> Result<String, minijinja::Error> {
        crate::render_template(src, vec![], serde_json::Value::Null, "", "", false, true)
    }

    /// Each template beside what Python Jinja2 3.1.6 printed for it in transformers' environment
    /// (the oracle `tests/fixtures/jinja2_text.py` builds), with no variables.
    #[test]
    fn generation_renders_as_python_renders_it() {
        let cases = [
            // The body's `set` stays inside it, as in a `with` block.
            (
                "{% set x = 1 %}{% generation %}{% set x = 2 %}in={{ x }}{% endgeneration %} out={{ x }}",
                "in=2 out=1",
            ),
            // Whitespace control on either tag, and trim_blocks / lstrip_blocks around them.
            ("A\n  {%- generation -%}\n B \n{%- endgeneration -%}\nC", "ABC"),
            ("A\n  {% generation %}\nB\n  {% endgeneration %}\nC", "A\nB\nC"),
            (
                "A\n  {%+ generation +%}\nB\n  {%+ endgeneration +%}\nC",
                "A\n  \nB\n  \nC",
            ),
            ("A\n  {%+ generation %}\nB{% endgeneration -%}\n  C", "A\n  BC"),
            // A tag split across lines.
            ("{%\n  generation\n%}x{%-\nendgeneration\n-%}\ny", "xy"),
            // Nested.
            (
                "{% generation %}a{% generation %}b{% endgeneration %}c{% endgeneration %}",
                "abc",
            ),
            (
                "{% generation %}{% with %}{% set x = 1 %}{% endwith %}{{ x }}{% endgeneration %}",
                "",
            ),
            // Text that only looks like the tag: raw blocks, comments, string literals.
            (
                "{% raw %}{% generation %}{%- endgeneration -%}{% endraw %}",
                "{% generation %}{%- endgeneration -%}",
            ),
            (
                "{%- raw -%} {% generation %} {%- endraw %}|{% generation %}x{% endgeneration %}",
                "{% generation %}|x",
            ),
            ("{# {% generation %} #}x", "x"),
            ("{#- {% endgeneration %} -#}y", "y"),
            ("{{ '{% generation %}' }}", "{% generation %}"),
            (r#"{{ "}}{% generation %}" }}"#, "}}{% generation %}"),
            (
                r#"{% set s = "%}{% endgeneration %}" %}{{ s }}"#,
                "%}{% endgeneration %}",
            ),
            (
                r#"{{ "{% endgeneration %}" }}|{{ "%}{% endgeneration %}" }}"#,
                "{% endgeneration %}|%}{% endgeneration %}",
            ),
            (
                r"{% set s = 'it\'s {% generation %}' %}{{ s }}",
                "it's {% generation %}",
            ),
            (
                "{% set d = {'k': '%}'} %}{% generation %}{{ d.k }}{% endgeneration %}",
                "%}",
            ),
            // A `}}` inside a dict literal does not end the expression.
            (
                "{{ {'a': {'b': 1}}['a']['b'] ~ '{% generation %}' }}",
                "1{% generation %}",
            ),
            // `generation` as an ordinary name.
            (
                "{% set generation = 1 %}{% if generation %}{{ generation }}{% endif %}",
                "1",
            ),
            (
                "{% for generation in [1, 2] %}{{ generation }}{% endfor %}",
                "12",
            ),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|&(src, want)| {
                let got = render(src);
                (got.as_deref().ok() != Some(want))
                    .then(|| format!("{src:?}: got {got:?}, want {want:?}"))
            })
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// Every template Python refuses for a misused tag is refused here too: unclosed, closing
    /// nothing, crossed with a `with` or an `if`, or given arguments.
    #[test]
    fn misused_generation_is_refused_as_python_refuses_it() {
        for src in [
            "{% generation %}x",
            "{% generation %}a\n\n{% generation %}b",
            "x{% endgeneration %}",
            "{% with %}x{% endgeneration %}",
            "{% generation %}x{% endwith %}",
            "{% generation x %}y{% endgeneration %}",
            "{% generation x = 1 %}y{% endgeneration %}",
            "{% generation %}y{% endgeneration x %}",
            "{% generation %}{% if true %}y{% endgeneration %}{% endif %}",
        ] {
            let err = render(src).expect_err(src);
            assert_eq!(
                err.kind(),
                minijinja::ErrorKind::SyntaxError,
                "{src}: {err:#}"
            );
        }
    }

    /// Only the statement name changes, padded to its old length, so every whitespace-control
    /// marker, line and column of the template stays where it was; a template without the tag is
    /// handed back as it came.
    #[test]
    fn rewrite_replaces_only_the_name() {
        let src = "{%- generation +%}\n{%+\tendgeneration -%}";
        let out = rewrite_generation_tags(src).unwrap();
        assert_eq!(out, "{%- with       +%}\n{%+\tendwith       -%}");
        let plain = "{% if generation %}{{ '{% generation %}' }}{% endif %}";
        assert!(matches!(
            rewrite_generation_tags(plain).unwrap(),
            Cow::Borrowed(s) if s == plain
        ));
    }

    /// An error names the line of the tag at fault.
    #[test]
    fn errors_name_the_line() {
        let err = rewrite_generation_tags("a\n{% with %}\nb\n{% endgeneration %}").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("line 4") && msg.contains("line 2"), "{msg}");
        let err = rewrite_generation_tags("a\n\n{% generation %}").unwrap_err();
        assert!(err.to_string().contains("line 3"), "{err}");
    }
}
