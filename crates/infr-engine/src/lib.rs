//! Load pipeline + session orchestration. Backend-agnostic: holds `Arc<dyn Backend>` and
//! never names a GPU API. Shared by `infr run` and `infr serve`.
//!
//! Reference for chat/channel/tool behavior: `~/Projects/scratch/dgemma-openai-server.py`.
//! See docs/PLAN.md "engine".
#![allow(dead_code, unused_variables)]

// Chat text logic (prompt rendering, channel/tool-call parsing, message types) lives in the shared
// `infr-chat` crate so backends and the server share ONE implementation. Re-exported here for the
// existing `infr_engine::{ChatMessage, …}` call sites.
pub use infr_chat::{
    normalize_messages, parse_hermes_tool_calls, parse_tool_calls, split_channels, split_reasoning,
    split_think, ChatMessage, ToolCall,
};

use std::path::Path;
use std::sync::Arc;

use infr_core::{error::Result, Backend};

/// A streamed piece of a response (re-exported from the single splitter in `infr-chat`).
pub use infr_chat::{prompt_prefills_think, ChatStream, Delta};

/// Owns the loaded model + compiled plan + decoder, over an opaque backend.
pub struct Engine {
    backend: Arc<dyn Backend>,
    // TODO(sonnet): Box<dyn Model>, compiled plan(s), Box<dyn DecodeStrategy>, tokenizer,
    // chat template (minijinja Environment), tool definitions.
}

impl Engine {
    /// Load a GGUF model onto the given backend and prepare it for generation.
    ///
    /// TODO(sonnet): Gguf::open -> DiffusionGemma::from_weights -> build_graph ->
    /// backend.compile -> construct DiffusionDecoder; load tokenizer + chat template.
    pub fn load(model_path: &Path, backend: Arc<dyn Backend>) -> Result<Self> {
        todo!("load model onto backend")
    }

    /// Run one chat turn, streaming deltas (reasoning vs content vs tool calls).
    ///
    /// TODO(sonnet): apply chat template (+ tool defs) -> tokenize -> decode; split
    /// `<|channel>thought`/`<channel|>`; parse `<|tool_call>call:NAME{…}<tool_call|>` into
    /// `Delta::ToolCall`. Mirror the shim's logic.
    pub fn chat(
        &mut self,
        messages: &[ChatMessage],
        tools_json: Option<&str>,
        on_delta: &mut dyn FnMut(Delta),
    ) -> Result<()> {
        todo!("apply template, tokenize, decode, stream channels + tool calls")
    }
}
