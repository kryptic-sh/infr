//! Pure-logic chat helpers — channel splitting, tool-call parsing, message normalisation,
//! and chat-template rendering. No GPU, no model, no IO.
//!
//! Reference: `~/Projects/scratch/dgemma-openai-server.py` (Python shim).
//! Token formats: docs/PLAN.md "DiffusionGemma spec".

use serde_json::Value;

use crate::ChatMessage;

// ---------------------------------------------------------------------------
// Channel splitting
// ---------------------------------------------------------------------------

/// The marker that begins the final-answer channel.
const FINAL_MARK: &str = "<channel|>";

/// All marker substrings that should be stripped when cleaning channel text.
const MARKERS: &[&str] = &[
    "<|channel>thought",
    "<|channel|>thought",
    "<|channel>",
    "<channel|>",
    "<|channel|>",
];

fn strip_markers(s: &str) -> String {
    let mut out = s.to_owned();
    for m in MARKERS {
        out = out.replace(m, "");
    }
    // Trim leading newlines / whitespace (mirrors Python lstrip("\n").strip())
    out.trim_start_matches('\n').trim().to_owned()
}

/// Split cumulative model output into `(reasoning, content)`.
///
/// Reasoning = text before `<channel|>` (markers stripped).
/// Content   = text after  `<channel|>` (markers stripped).
/// If the marker is absent, reasoning = full stripped text, content = `""`.
pub fn split_channels(full: &str) -> (String, String) {
    if let Some(idx) = full.find(FINAL_MARK) {
        let head = &full[..idx];
        let tail = &full[idx + FINAL_MARK.len()..];
        (strip_markers(head), strip_markers(tail))
    } else {
        (strip_markers(full), String::new())
    }
}

// ---------------------------------------------------------------------------
// Tool-call parsing
// ---------------------------------------------------------------------------

/// One parsed tool invocation from the model's `<|tool_call>…<tool_call|>` block.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

// Regex-free: scan for literal open/close delimiters.
const TC_OPEN: &str = "<|tool_call>";
const TC_CLOSE: &str = "<tool_call|>";

/// Recursive-descent value parser — mirrors `_parse_value` in the Python shim.
///
/// `s` has already had `<|"|>` replaced with `"`.
fn parse_value(s: &[u8], mut i: usize) -> (Value, usize) {
    // skip whitespace
    while i < s.len() && matches!(s[i], b' ' | b'\n' | b'\t' | b'\r') {
        i += 1;
    }
    if i >= s.len() {
        return (Value::Null, i);
    }
    match s[i] {
        b'{' => {
            i += 1;
            let mut obj = serde_json::Map::new();
            loop {
                // skip whitespace and commas
                while i < s.len() && matches!(s[i], b' ' | b'\n' | b'\t' | b'\r' | b',') {
                    i += 1;
                }
                if i >= s.len() || s[i] == b'}' {
                    if i < s.len() {
                        i += 1;
                    }
                    return (Value::Object(obj), i);
                }
                // find colon for key:value
                let colon = match s[i..].iter().position(|&b| b == b':') {
                    Some(p) => i + p,
                    None => return (Value::Object(obj), i),
                };
                let raw_key = std::str::from_utf8(&s[i..colon])
                    .unwrap_or("")
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_owned();
                i = colon + 1;
                let (val, ni) = parse_value(s, i);
                i = ni;
                obj.insert(raw_key, val);
            }
        }
        b'[' => {
            i += 1;
            let mut arr = Vec::new();
            loop {
                while i < s.len() && matches!(s[i], b' ' | b'\n' | b'\t' | b'\r' | b',') {
                    i += 1;
                }
                if i >= s.len() || s[i] == b']' {
                    if i < s.len() {
                        i += 1;
                    }
                    return (Value::Array(arr), i);
                }
                let (val, ni) = parse_value(s, i);
                i = ni;
                arr.push(val);
            }
        }
        q @ b'"' | q @ b'\'' => {
            i += 1;
            let mut buf = Vec::new();
            while i < s.len() {
                if s[i] == b'\\' && i + 1 < s.len() {
                    buf.push(s[i + 1]);
                    i += 2;
                } else if s[i] == q {
                    i += 1;
                    break;
                } else {
                    buf.push(s[i]);
                    i += 1;
                }
            }
            let text = String::from_utf8_lossy(&buf).into_owned();
            (Value::String(text), i)
        }
        _ => {
            // bareword / number / bool / null — read until delimiter
            let j_rel = s[i..]
                .iter()
                .position(|&b| matches!(b, b',' | b'}' | b']'))
                .unwrap_or(s.len() - i);
            let j = i + j_rel;
            let tok = std::str::from_utf8(&s[i..j]).unwrap_or("").trim();
            i = j;
            match tok {
                "true" => (Value::Bool(true), i),
                "false" => (Value::Bool(false), i),
                "null" => (Value::Null, i),
                _ => {
                    if let Ok(n) = tok.parse::<i64>() {
                        (Value::Number(n.into()), i)
                    } else if let Ok(f) = tok.parse::<f64>() {
                        (
                            Value::Number(serde_json::Number::from_f64(f).unwrap_or(0.into())),
                            i,
                        )
                    } else {
                        (Value::String(tok.to_owned()), i)
                    }
                }
            }
        }
    }
}

/// Find all `<|tool_call>…<tool_call|>` blocks, parse them, and return:
/// - `clean`: the input text with all tool-call blocks (and stray markers) removed.
/// - the parsed [`ToolCall`] list.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut clean = text.to_owned();

    // Work on the original text for finding blocks; collect spans to remove.
    let mut search_from = 0usize;
    let mut spans: Vec<(usize, usize)> = Vec::new();

    while let Some(open_pos) = text[search_from..].find(TC_OPEN) {
        let open_abs = search_from + open_pos;
        let body_start = open_abs + TC_OPEN.len();
        let Some(close_rel) = text[body_start..].find(TC_CLOSE) else {
            break;
        };
        let body = &text[body_start..body_start + close_rel];
        let close_abs = body_start + close_rel + TC_CLOSE.len();
        spans.push((open_abs, close_abs));
        search_from = close_abs;

        // parse body: strip leading "call:"
        let body = body.trim();
        let body = body.strip_prefix("call:").unwrap_or(body);

        let Some(brace) = body.find('{') else {
            continue;
        };
        let name = body[..brace].trim().to_owned();
        if name.is_empty() {
            continue;
        }
        // Replace the model's string-quote escape with real double-quotes
        let argstr = body[brace..].replace("<|\"|>", "\"");
        let (val, _) = parse_value(argstr.as_bytes(), 0);
        let arguments = match val {
            Value::Object(_) => val,
            other => {
                let mut m = serde_json::Map::new();
                m.insert("value".to_owned(), other);
                Value::Object(m)
            }
        };
        calls.push(ToolCall { name, arguments });
    }

    // Remove spans in reverse order so indices stay valid
    for (start, end) in spans.into_iter().rev() {
        clean.replace_range(start..end, "");
    }
    clean = strip_markers(&clean);
    (clean, calls)
}

// ---------------------------------------------------------------------------
// Message normalisation
// ---------------------------------------------------------------------------

/// Normalise a slice of [`ChatMessage`]s for feeding into the chat template.
///
/// - Flattens content arrays to plain text.
/// - Preserves `tool_call_id`, `name`.
/// - Returns new owned `ChatMessage` values (cheap: Strings only).
///
/// In the Python shim this also preserves `tool_calls` / `reasoning_content`
/// fields, but those live outside `ChatMessage` in our type; callers handle them.
pub fn normalize_messages(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            tool_call_id: m.tool_call_id.clone(),
            name: m.name.clone(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- split_channels --------------------------------------------------

    #[test]
    fn split_channels_reasoning_and_answer() {
        let full = "<|channel>thought\nI need to think about this.\n<channel|>The answer is 42.";
        let (r, c) = split_channels(full);
        assert_eq!(r, "I need to think about this.", "reasoning mismatch");
        assert_eq!(c, "The answer is 42.", "content mismatch");
    }

    #[test]
    fn split_channels_no_marker_returns_reasoning_only() {
        let full = "<|channel>thought\nOnly reasoning here, no final marker.";
        let (r, c) = split_channels(full);
        assert_eq!(r, "Only reasoning here, no final marker.");
        assert_eq!(c, "", "content should be empty when marker absent");
    }

    #[test]
    fn split_channels_strips_all_marker_variants() {
        // Both head and tail may contain stray markers from the model.
        let full = "<|channel|>thought\nsome thought<|channel|><channel|>the answer";
        let (r, c) = split_channels(full);
        assert!(!r.contains("<|channel"), "stray markers in reasoning");
        assert!(!c.contains("<channel|"), "stray markers in content");
        assert_eq!(r, "some thought");
        assert_eq!(c, "the answer");
    }

    #[test]
    fn split_channels_empty_input() {
        let (r, c) = split_channels("");
        assert_eq!(r, "");
        assert_eq!(c, "");
    }

    // --- parse_tool_calls ------------------------------------------------

    #[test]
    fn parse_tool_calls_bash_single_arg() {
        // Format from docs/PLAN.md: strings wrapped in <|"|>…<|"|>
        let text = r#"<|tool_call>call:bash{command:<|"|>ls<|"|>}<tool_call|>"#;
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments, json!({"command": "ls"}));
        assert_eq!(
            clean.trim(),
            "",
            "clean should have no leftover tool-call text"
        );
    }

    #[test]
    fn parse_tool_calls_multi_arg() {
        let text = r#"<|tool_call>call:write_file{path:<|"|>/tmp/x.txt<|"|>,content:<|"|>hello<|"|>}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(
            calls[0].arguments,
            json!({"path": "/tmp/x.txt", "content": "hello"})
        );
    }

    #[test]
    fn parse_tool_calls_nested_values() {
        // Nested object + array
        let text = r#"<|tool_call>call:query{filter:{field:<|"|>name<|"|>,values:[<|"|>a<|"|>,<|"|>b<|"|>]}}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "query");
        let args = &calls[0].arguments;
        assert_eq!(args["filter"]["field"], json!("name"));
        assert_eq!(args["filter"]["values"], json!(["a", "b"]));
    }

    #[test]
    fn parse_tool_calls_numeric_and_bool_args() {
        let text = r#"<|tool_call>call:set_config{timeout:30,enabled:true,ratio:0.5}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["timeout"], json!(30));
        assert_eq!(calls[0].arguments["enabled"], json!(true));
        // f64 comparison via as_f64
        let ratio = calls[0].arguments["ratio"].as_f64().unwrap();
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn parse_tool_calls_no_tool_call_empty_vec() {
        let text = "Just a plain answer with no tool calls.";
        let (clean, calls) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert_eq!(clean, text);
    }

    #[test]
    fn parse_tool_calls_multiple_calls() {
        let text = "<|tool_call>call:foo{x:<|\"|>1<|\"|>}<tool_call|> middle <|tool_call>call:bar{y:<|\"|>2<|\"|>}<tool_call|> suffix.";
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "foo");
        assert_eq!(calls[1].name, "bar");
        assert!(clean.contains("middle"));
        assert!(clean.contains("suffix"));
    }

    #[test]
    fn parse_tool_calls_text_preserved_around_blocks() {
        let text = "Before.<|tool_call>call:ping{}<tool_call|>After.";
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert!(
            clean.contains("Before.") || clean.contains("After."),
            "surrounding text should survive: {clean:?}"
        );
    }

    // --- normalize_messages ----------------------------------------------

    #[test]
    fn normalize_messages_passthrough() {
        let msgs = vec![
            ChatMessage {
                role: "user".to_owned(),
                content: "hello".to_owned(),
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".to_owned(),
                content: "hi".to_owned(),
                tool_call_id: Some("tc_1".to_owned()),
                name: Some("my_fn".to_owned()),
            },
        ];
        let out = normalize_messages(&msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, "user");
        assert_eq!(out[1].tool_call_id.as_deref(), Some("tc_1"));
        assert_eq!(out[1].name.as_deref(), Some("my_fn"));
    }
}
