//! Shared chat text logic — turning conversations into model prompts and parsing model output back
//! out. **No GPU, no model, no inference.** Used by `infr-llama` (to feed the forward pass) and by
//! `infr-engine`/`infr-server` (the OpenAI-shaped API) so prompt rendering lives in ONE place instead
//! of being re-implemented next to each backend.
//!
//! - [`render_chat_jinja`] / [`render_chat_user`] — render a GGUF's embedded `tokenizer.chat_template`
//!   (jinja, via minijinja) into a prompt string. Return `None` when the GGUF has no template (or it
//!   fails to render); callers fail loud rather than fabricate a default — infr only supports
//!   models that ship a chat template.
//! - [`split_channels`] / [`parse_tool_calls`] — parse model output (reasoning vs answer,
//!   `<|tool_call>` blocks).

mod stream;
mod template;
mod tools;

pub use stream::{prompt_prefills_think, ChatStream, Delta};
pub use template::{
    render_chat_jinja, render_chat_oai, render_chat_user, render_template, TemplateError,
};
pub use tools::{
    parse_any_tool_calls, parse_hermes_tool_calls, parse_tool_calls, split_channels,
    split_reasoning, split_think, ToolCall,
};

/// Stands in for one image in a message's text: `infr serve` renders each OpenAI `image_url` part
/// as this, and the multimodal prompt encoder (`infr_llama::media`) splits the rendered prompt on it
/// and splices the image's soft tokens in its place. Private-use code points, so ordinary text never
/// contains it by accident and no chat template rewrites it.
pub const IMAGE_MARKER: &str = "\u{E000}infr:image\u{E001}";
/// Shared request limits, checked before allocating decoded data-URI bytes.
pub const MAX_REQUEST_IMAGES: usize = 4;
/// Maximum encoded image file size (after base64 decoding).
pub const MAX_IMAGE_BYTES: usize = 16 << 20;

/// One chat message (OpenAI-shaped; tool fields preserved for the agentic round-trip).
#[derive(Clone, Debug, Default)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// The assistant's OUTGOING tool calls (OpenAI `message.tool_calls`), replayed into the prompt on
    /// the next turn so the model sees its own prior calls. Empty/None for non-assistant messages.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For a `tool`-role result message: which call it answers (OpenAI `tool_call_id`).
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}
