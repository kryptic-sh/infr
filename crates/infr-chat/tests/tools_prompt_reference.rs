//! Regression tests: a conversation with tool definitions, a replayed assistant tool call and a
//! tool result renders, through real chat templates, byte for byte what llama.cpp renders.
//!
//! The templates are `tokenizer.chat_template` extracted verbatim from the GGUFs
//! (unsloth/Qwen3-0.6B-GGUF, unsloth/Qwen3.5-0.8B-GGUF, unsloth/Llama-3.2-1B-Instruct-GGUF). Each
//! `*_tools_reference.txt` is what llama.cpp's `test-chat-template --no-common` (at 5676dfbe0)
//! wrote for that template with the same variables as [`render`] passes, `\r\n` turned back into
//! `\n` (the tool writes its output file in text mode on Windows). The tool definitions use the
//! key order llama.cpp's server gives them (`type`, then `function` with `name`, `description`,
//! `parameters`), and their `tojson` output is where infr used to differ: compact JSON where the
//! reference writes Python's `", "` and `": "` separators.

use infr_chat::render_template;
use serde_json::{json, Value};

fn tools() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city — Zürich, 東京 <b>bold</b> & \"quoted\" 'single'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        "days": {"type": "integer", "minimum": 1, "maximum": 7, "default": 1.0},
                        "detail": {
                            "type": "object",
                            "properties": {
                                "hourly": {"type": "boolean"},
                                "threshold": {"type": "number"}
                            }
                        }
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"}
                    },
                    "required": ["query"]
                }
            }
        }
    ])
}

fn messages() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "You are a helpful assistant."}),
        json!({"role": "user", "content": "What's the weather in Zürich?"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {
                        "city": "Zürich",
                        "unit": "celsius",
                        "days": 3,
                        "detail": {"hourly": true, "threshold": 0.5},
                        "tags": ["a", "b"]
                    }
                },
                "id": "call_1"
            }]
        }),
        json!({
            "role": "tool",
            "content": "{\"temp\": 21.5, \"sky\": \"clear\"}",
            "tool_call_id": "call_1",
            "name": "get_weather"
        }),
        json!({"role": "user", "content": "Thanks"}),
    ]
}

fn render(template: &str) -> String {
    render_template(template, messages(), tools(), "<s>", "</s>", true, true).unwrap()
}

#[test]
fn qwen3_tools_prompt_matches_the_reference() {
    assert_eq!(
        render(include_str!("fixtures/qwen3_chat_template.jinja")),
        include_str!("fixtures/qwen3_tools_reference.txt")
    );
}

#[test]
fn qwen35_tools_prompt_matches_the_reference() {
    assert_eq!(
        render(include_str!("fixtures/qwen35_chat_template.jinja")),
        include_str!("fixtures/qwen35_tools_reference.txt")
    );
}

/// The Llama 3.x template stamps today's date through `strftime_now` (`render_template` passes no
/// `date_string` to override it); the reference render was taken on the date written in it, so
/// that line is moved to the render's date before comparing. The date is read on both sides of the
/// render, so a run that crosses midnight still compares against the date the render used.
#[test]
fn llama3_tools_prompt_matches_the_reference() {
    const CAPTURED: &str = "Today Date: 13 Nov 2026\n";
    let want = include_str!("fixtures/llama3_tools_reference.txt");
    assert!(want.contains(CAPTURED), "the fixture's date line moved");
    let want_on = |date: chrono::DateTime<chrono::Local>| {
        want.replace(CAPTURED, &date.format("Today Date: %d %b %Y\n").to_string())
    };
    let before = chrono::Local::now();
    let got = render(include_str!("fixtures/llama3_chat_template.jinja"));
    let after = chrono::Local::now();
    if got != want_on(after) {
        assert_eq!(got, want_on(before));
    }
}
