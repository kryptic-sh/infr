//! Regression tests: the STOCK Llama-3.x chat template (extracted verbatim from
//! unsloth/Llama-3.2-1B-Instruct-GGUF `tokenizer.chat_template` — same template family as
//! Llama-3.1/3.3) must render through `render_template` with tools present. It exercises jinja
//! constructs the Qwen-family templates don't: `tojson(indent=4)`, `strftime_now`, list slicing
//! (`messages[1:]`), and the ipython/tool role branch. A render failure here is exactly the
//! serve-side 500 on `/v1/chat/completions` with `tools`.

use infr_chat::render_template;
use serde_json::{json, Value};

const LLAMA3_TEMPLATE: &str = include_str!("fixtures/llama3_chat_template.jinja");

/// The OpenAI-shaped `tools` array serve passes through (one function with parameters).
fn weather_tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }
        }
    }])
}

fn render(msgs: Vec<Value>, tools: Value) -> Result<String, minijinja::Error> {
    render_template(
        LLAMA3_TEMPLATE,
        msgs,
        tools,
        "<|begin_of_text|>",
        "<|eot_id|>",
        true,
        true,
    )
}

/// Tools request → must render (this was the serve 500: `tojson(indent=4)` rejected the kwarg) and
/// carry the Llama-3.x structure: header markers, ipython environment line, the function JSON in
/// the first user turn, and a trailing assistant header.
#[test]
fn llama3_renders_with_tools() {
    let msgs = vec![
        json!({ "role": "system", "content": "You are a helpful assistant." }),
        json!({ "role": "user", "content": "What's the weather in Paris?" }),
    ];
    let out = render(msgs, weather_tools()).expect("Llama-3.x template must render with tools");
    assert!(out.starts_with("<|begin_of_text|>"), "bos first: {out:?}");
    assert!(out.contains("<|start_header_id|>system<|end_header_id|>"));
    // tools present → the template turns on the ipython environment.
    assert!(out.contains("Environment: ipython"), "{out}");
    // Function JSON lands in the FIRST USER message (tools_in_user_message defaults true), pretty
    // printed (indent=4) — key markers of the tool block.
    assert!(out.contains("<|start_header_id|>user<|end_header_id|>"));
    assert!(out.contains("\"name\": \"get_weather\""), "{out}");
    assert!(
        out.contains("Given the following functions, please respond with a JSON"),
        "{out}"
    );
    // indent=4: the nested "function" key sits at one 4-space level.
    assert!(out.contains("\n    \"function\": {"), "{out}");
    // Original user text is inlined after the tool block; generation prompt closes the render.
    assert!(out.contains("What's the weather in Paris?<|eot_id|>"));
    assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
}

/// The full agentic round-trip: assistant `tool_calls` replayed as the bare-JSON Llama-3.x call
/// form, tool result rendered under the `ipython` header.
#[test]
fn llama3_renders_tool_call_roundtrip() {
    let msgs = vec![
        json!({ "role": "user", "content": "What's the weather in Paris?" }),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": { "name": "get_weather", "arguments": { "city": "Paris" } }
            }]
        }),
        json!({ "role": "tool", "content": "sunny, 24C" }),
    ];
    let out = render(msgs, weather_tools()).expect("tool round-trip must render");
    // Assistant tool call: {"name": "get_weather", "parameters": {"city": "Paris"}} (bare JSON form).
    assert!(
        out.contains("{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}"),
        "{out}"
    );
    // Tool result comes back under the ipython role header.
    assert!(
        out.contains("<|start_header_id|>ipython<|end_header_id|>"),
        "{out}"
    );
    assert!(out.contains("sunny, 24C"), "{out}");
}

/// No tools → plain chat render keeps working (guards the shared path).
#[test]
fn llama3_renders_without_tools() {
    let msgs = vec![json!({ "role": "user", "content": "Hi" })];
    let out = render(msgs, Value::Null).expect("plain chat must render");
    assert!(out.contains("<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"));
    assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
}

/// Today's date as a bare `strftime_now` call with the template's own format writes it. What the
/// function answers in a known time zone is `localtime`'s unit tests' business.
fn today() -> String {
    render_template(
        "{{ strftime_now('%d %b %Y') }}",
        vec![],
        Value::Null,
        "",
        "",
        false,
        false,
    )
    .expect("a bare strftime_now renders")
}

/// `strftime_now` is defined (llama.cpp-minja parity), so the template stamps TODAY's date — not
/// its hardcoded "26 Jul 2024" fallback. The date is read on both sides of the render, so a run
/// that crosses midnight still finds the date the render used.
#[test]
fn llama3_date_string_uses_strftime_now() {
    let msgs = vec![json!({ "role": "user", "content": "Hi" })];
    let before = today();
    let out = render(msgs, Value::Null).expect("plain chat must render");
    let after = today();
    assert!(
        [&before, &after]
            .iter()
            .any(|today| out.contains(&format!("Today Date: {today}\n"))),
        "expected today's date ({before} or {after}) in render: {out}"
    );
    assert!(!out.contains("26 Jul 2024"), "fallback date used: {out}");
    // A date, not an empty or half-formatted string: two digits, a month name, four digits.
    let parts: Vec<&str> = after.split(' ').collect();
    let [day, month, year] = parts[..] else {
        panic!("{after}");
    };
    assert!(day.len() == 2 && day.parse::<u8>().is_ok(), "{after}");
    assert!(month.len() == 3 && month.is_ascii(), "{after}");
    assert!(year.len() == 4 && year.parse::<u16>().is_ok(), "{after}");
}

/// `tojson` WITHOUT `indent` writes Python's `json.dumps` separators, as llama.cpp does — the form
/// the Qwen-family templates render tools with.
#[test]
fn tojson_without_indent_uses_python_separators() {
    let out = render_template(
        "{% set t = {\"a\": 1, \"b\": [1, 2]} %}{{ t | tojson }}",
        vec![],
        Value::Null,
        "",
        "",
        false,
        false,
    )
    .expect("one-line tojson");
    assert_eq!(out, "{\"a\": 1, \"b\": [1, 2]}");
}
