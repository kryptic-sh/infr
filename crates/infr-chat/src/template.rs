//! Chat-template rendering: turn `(role, content)` messages into a prompt string via the GGUF's
//! embedded `tokenizer.chat_template` (a Jinja2 string). The single source of truth — every prompt
//! path funnels through [`render_chat_jinja`].

use infr_core::loader::MetaValue;
use infr_core::WeightSource; // brings `Gguf::metadata()` into scope
use infr_gguf::Gguf;
use tokenizers::Tokenizer;

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
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|(r, c)| serde_json::json!({ "role": r, "content": c }))
        .collect();
    let mut ctx = serde_json::Map::new();
    ctx.insert("messages".into(), serde_json::Value::Array(msgs));
    ctx.insert("tools".into(), serde_json::Value::Null);
    ctx.insert("add_generation_prompt".into(), add_generation_prompt.into());
    ctx.insert("bos_token".into(), bos.into());
    ctx.insert("eos_token".into(), eos_s.into());
    // `enable_thinking` is only set when the user opts in via INFR_THINK — otherwise the key is
    // ABSENT so each template applies its OWN default (e.g. Qwen3 thinks; Qwen3-Next prefills an
    // empty `<think></think>` to stay non-thinking, which is what keeps its greedy decode from
    // degenerating). INFR_THINK=1 forces thinking on, INFR_THINK=0 forces it off.
    if let Ok(v) = std::env::var("INFR_THINK") {
        ctx.insert("enable_thinking".into(), (v != "0").into());
    }
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
