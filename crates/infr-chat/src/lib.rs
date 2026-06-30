//! Shared chat text logic — turning conversations into model prompts and parsing model output back
//! out. **No GPU, no model, no inference.** Used by `infr-llama` (to feed the forward pass) and by
//! `infr-engine`/`infr-server` (the OpenAI-shaped API) so prompt rendering lives in ONE place instead
//! of being re-implemented next to each backend.
//!
//! - [`render_chat_jinja`] / [`render_chat_user`] — render a GGUF's embedded `tokenizer.chat_template`
//!   (jinja, via minijinja) into a prompt string. Return `None` when the GGUF has no template (or it
//!   fails to render); callers fail loud rather than fabricate a default — infr only supports
//!   models that ship a chat template.
//! - [`split_channels`] / [`parse_tool_calls`] / [`normalize_messages`] — parse model output
//!   (reasoning vs answer, `<|tool_call>` blocks) and tidy inbound messages.

mod template;
mod tools;

pub use template::{render_chat_jinja, render_chat_user};
pub use tools::{normalize_messages, parse_tool_calls, split_channels, ToolCall};

/// One chat message (OpenAI-shaped; tool fields preserved for the agentic round-trip).
#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}
