//! Chat-template rendering: turn `(role, content)` messages into a prompt string via the GGUF's
//! embedded `tokenizer.chat_template` (a Jinja2 string). The single source of truth — every prompt
//! path funnels through [`render_chat_jinja`].

use infr_core::loader::MetaValue;
use infr_core::WeightSource; // brings `Gguf::metadata()` into scope
use infr_gguf::Gguf;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::ChatMessage;

/// THE jinja chat renderer — turns `(role, content)` messages into a prompt via the GGUF's embedded
/// `tokenizer.chat_template`. Template handling (pycompat, `enable_thinking`, bos/eos, tools) lives
/// here so every caller (single-turn, multi-turn, CPU + GPU backends) shares it. Returns `None` if
/// there's no template or it fails to render (caller falls back to [`chatml`]).
pub fn render_chat_jinja(
    gguf: &Gguf,
    tokenizer: &Tokenizer,
    eos: u32,
    messages: &[(&str, &str)],
    add_generation_prompt: bool,
) -> Option<String> {
    let msgs: Vec<Value> = messages
        .iter()
        .map(|(r, c)| serde_json::json!({ "role": r, "content": c }))
        .collect();
    render_core(
        gguf,
        tokenizer,
        eos,
        msgs,
        Value::Null,
        add_generation_prompt,
    )
}

/// Render OpenAI-shaped [`ChatMessage`]s (full multi-turn history WITH tool calls + results) plus an
/// optional `tools` spec through the GGUF's embedded chat template. This is the tool-calling entry
/// point: the model's OWN template renders the tool definitions and wraps prior `tool_calls` /
/// `tool` results in its native format — so infr never hardcodes a per-model tool syntax.
///
/// `tools` is the request's OpenAI `tools` array (`[{type:"function", function:{name, parameters}}]`)
/// or `None`. Assistant `tool_calls` are emitted as `{type:"function", function:{name, arguments}}`
/// with `arguments` as a JSON object (templates `| tojson` it).
pub fn render_chat_oai(
    gguf: &Gguf,
    tokenizer: &Tokenizer,
    eos: u32,
    messages: &[ChatMessage],
    tools: Option<&Value>,
    add_generation_prompt: bool,
) -> Option<String> {
    let msgs: Vec<Value> = messages.iter().map(message_to_json).collect();
    let tools = tools.cloned().unwrap_or(Value::Null);
    render_core(gguf, tokenizer, eos, msgs, tools, add_generation_prompt)
}

/// Build the template's per-message dict, preserving the tool round-trip fields the HF chat templates
/// read (`tool_calls`, `tool_call_id`, `name`).
fn message_to_json(m: &ChatMessage) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("role".into(), m.role.clone().into());
    obj.insert("content".into(), m.content.clone().into());
    if let Some(calls) = &m.tool_calls {
        let arr: Vec<Value> = calls
            .iter()
            .map(|c| {
                serde_json::json!({
                    "type": "function",
                    "function": { "name": c.name, "arguments": c.arguments },
                })
            })
            .collect();
        obj.insert("tool_calls".into(), Value::Array(arr));
    }
    if let Some(id) = &m.tool_call_id {
        obj.insert("tool_call_id".into(), id.clone().into());
    }
    if let Some(name) = &m.name {
        obj.insert("name".into(), name.clone().into());
    }
    Value::Object(obj)
}

/// Core renderer: set up the minijinja env (pycompat, `raise_exception`, `tojson`), bind the prompt
/// context (`messages`, `tools`, bos/eos, `enable_thinking`), and render. Shared by every entry
/// point so template handling lives in ONE place.
fn render_core(
    gguf: &Gguf,
    tokenizer: &Tokenizer,
    eos: u32,
    msgs: Vec<Value>,
    tools: Value,
    add_generation_prompt: bool,
) -> Option<String> {
    let template = gguf.metadata().str("tokenizer.chat_template")?;
    let bos_id = gguf
        .metadata()
        .get("tokenizer.ggml.bos_token_id")
        .and_then(MetaValue::as_u64)
        .unwrap_or(2) as u32;
    let bos = tokenizer.id_to_token(bos_id).unwrap_or_default();
    let eos_s = tokenizer.id_to_token(eos).unwrap_or_default();
    let mut env = minijinja::Environment::new();
    // HF chat templates lean on Python str/dict/list methods (`.get`, `.items`, `.strip`, …) that
    // minijinja core doesn't implement — pycompat supplies them (e.g. gemma4's tool-calling template).
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_filter("tojson", |v: minijinja::Value| {
        serde_json::to_string(&v).unwrap_or_else(|_| "null".to_owned())
    });
    if let Err(e) = env.add_template("chat", template) {
        if std::env::var("INFR_DEBUG_CHAT").is_ok() {
            eprintln!("[chat-template] parse error: {e:#}");
        }
        return None;
    }
    let tmpl = env.get_template("chat").ok()?;
    let mut ctx = serde_json::Map::new();
    ctx.insert("messages".into(), Value::Array(msgs));
    ctx.insert("tools".into(), tools);
    ctx.insert("add_generation_prompt".into(), add_generation_prompt.into());
    ctx.insert("bos_token".into(), bos.into());
    ctx.insert("eos_token".into(), eos_s.into());
    // Thinking is ON by default for every model whose template supports it — the key is simply
    // ignored by non-thinking templates, and thinking-capable models (Qwen3, Qwen3.5/Qwen3-Next)
    // then behave the same under `infr run`/`serve` regardless of what their template's own
    // default is (Qwen3.5 defaults itself OFF via `enable_thinking is defined and is true`).
    // INFR_NO_THINK=1 turns thinking off (INFR_NO_THINK=0 is a no-op, matching the other INFR_NO_*
    // toggles).
    let think = !std::env::var("INFR_NO_THINK").is_ok_and(|v| v != "0");
    ctx.insert("enable_thinking".into(), think.into());
    match tmpl.render(serde_json::Value::Object(ctx)) {
        Ok(s) => {
            if std::env::var("INFR_DEBUG_CHAT").is_ok() {
                eprintln!("[chat-template] rendered:\n{s}\n[/chat-template]");
            }
            Some(s)
        }
        Err(e) => {
            if std::env::var("INFR_DEBUG_CHAT").is_ok() {
                eprintln!("[chat-template] render error: {e:#}");
            }
            None
        }
    }
}

/// Single user turn through [`render_chat_jinja`] (`add_generation_prompt=true`). Shared by the GPU
/// and CPU one-shot paths so an instruct model answers coherently.
pub fn render_chat_user(
    gguf: &Gguf,
    tokenizer: &Tokenizer,
    eos: u32,
    user: &str,
) -> Option<String> {
    render_chat_jinja(gguf, tokenizer, eos, &[("user", user)], true)
}
