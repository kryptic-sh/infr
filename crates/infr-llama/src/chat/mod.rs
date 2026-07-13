//! Agnostic chat orchestration: ONE shared [`Chat`] over a per-backend [`ChatModel`], so every
//! architecture runs the identical multi-turn REPL instead of 5 bespoke chat loops.
//!
//! Each backend keeps its own bespoke forward (dense `ChatSession` with a persistent KV cache, the
//! eager `generate_moe`, qwen35's per-token loop, the CPU compute-graph runner). A backend only
//! implements two primitives — [`render`](ChatModel::render) a message list through the model's OWN
//! chat template, and [`generate`](ChatModel::generate) a reply for an already-rendered prompt. The
//! shared [`Chat`] owns the conversation history, re-renders it every turn, strips the model's
//! `<think>` reasoning from what it stores, and reports [`GenStats`]. Turn orchestration lives in
//! ONE place; only the two primitives differ per architecture.
//!
//! The boxed trait object is lifetime-bounded (`Box<dyn ChatModel + '_>`) so the borrow-based dense
//! `ChatSession` (which borrows `&Llama`) needs no ownership change — the caller owns the `Llama`,
//! the box borrows it.
//!
//! Per-backend [`ChatModel`] impls live one file per backend (module split): [`vulkan`] (dense +
//! qwen35 on the Vulkan seam), [`metal`] (macOS-only dense + speculative), [`cpu`] (the CPU/Metal
//! reference dense path), [`diffusion`] (diffusion-gemma). This module keeps the agnostic core —
//! the trait, the shared REPL, and the OpenAI-shaped renderer.

use crate::{no_template_err, GenStats};
use anyhow::Result;

mod cpu;
mod diffusion;
mod metal;
mod vulkan;

pub use cpu::CpuDenseChat;
pub use diffusion::DiffusionGemmaChat;
#[cfg(target_os = "macos")]
pub use metal::{MetalSeamChat, SpecMetalChat};
pub use vulkan::DenseSeamChat;

/// The two arch-specific primitives the shared [`Chat`] drives. Object-safe: no generics, callbacks
/// are `&mut dyn FnMut`. A stateful backend (dense) may keep a KV cache warm across `generate` calls;
/// stateless ones prefill the whole prompt each turn.
pub trait ChatModel {
    /// Render a conversation `(role, content)` into a prompt string via the model's embedded chat
    /// template. All backends funnel to `infr_chat::render_chat_jinja` (the single prompt renderer).
    fn render(&self, messages: &[(&str, &str)]) -> Result<String>;

    /// Generate a completion for the already-rendered `prompt`, streaming decoded text to `on_piece`.
    ///
    /// `req` is the in-flight SEQUENCE's own state (`infr serve`): its sampling overrides, its
    /// stop-sequence abort latch, and its turn on the GPU baton. `None` — `infr run`, `bench`, every
    /// test — means "inherit the process default", i.e. exactly the pre-existing behavior. It is an
    /// explicit parameter rather than ambient state precisely because a server steps several
    /// sequences at once; see [`crate::sampling::RequestCtx`].
    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats>;

    /// Like [`generate`](Self::generate), plus an optional per-denoise-step observable view for
    /// architectures with an internal multi-step decode inside ONE `generate` call (currently only
    /// diffusion-gemma's block-diffusion loop — [`crate::diffusion::StepView`], driving
    /// `infr run`'s `INFR_DIFFUSION_VISUAL` live canvas view). Default: ignore the hook and
    /// delegate to `generate` — every autoregressive backend already streams token-by-token via
    /// `on_piece` and has no internal notion of a "step", so this default keeps every existing
    /// `ChatModel` impl byte-identical with zero code change.
    fn generate_with_step_hook(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
        _on_step: Option<&mut dyn FnMut(crate::diffusion::StepView)>,
    ) -> Result<GenStats> {
        self.generate(prompt, max_new, req, on_piece)
    }

    /// Optional REPL status (e.g. dense returns `ctx N/MAX`); `None` for stateless backends.
    fn status(&self) -> Option<String> {
        None
    }

    /// Run a tiny throwaway generation through the real forward so every lazily-built pipeline
    /// compiles NOW instead of on the first user request (a cold Vulkan seam pays seconds of
    /// pipeline builds otherwise). INFR_PROF2 is suppressed for the duration — recorders read it
    /// at construction, and warmup submits would pollute a later bench's per-op aggregate.
    /// Default: no-op (stateless/CPU backends have nothing to warm).
    fn warmup(&mut self) -> Result<()> {
        Ok(())
    }

    /// Like [`generate`](Self::generate), with an llguidance grammar constraint applied to the
    /// decode (serve's FORCED tool_choice). Backends without constraint support return an error —
    /// the caller falls back to unconstrained generation.
    fn generate_constrained(
        &mut self,
        _prompt: &str,
        _max_new: usize,
        _constraint: &mut crate::grammar::Constraint,
        _req: Option<&crate::sampling::RequestCtx>,
        _on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        Err(anyhow::anyhow!(
            "grammar-constrained generation is not supported by this backend"
        ))
    }
}

/// Store only the ANSWER, dropping the model's reasoning (Qwen3 excludes prior-turn thinking;
/// keeping it degrades the model). Delegates to `infr-chat`'s splitter — the SAME reasoning
/// grammar `infr run`'s display and `infr serve`'s deltas use — so what history keeps always
/// matches what the surfaces call "content". Unterminated reasoning (truncated turn) → empty.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn strip_think(reply: &str) -> String {
    infr_chat::split_reasoning(reply).1
}

/// The SINGLE arch-agnostic chat: owns the conversation history + `<think>`-stripping, drives any
/// [`ChatModel`] backend. Every architecture uses this one turn implementation and REPL, so a working
/// multi-turn REPL is uniform (no per-backend "one-shot only" special cases).
pub struct Chat<'a> {
    model: Box<dyn ChatModel + 'a>,
    history: Vec<(String, String)>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl<'a> Chat<'a> {
    /// Wrap a per-backend [`ChatModel`] in the shared multi-turn chat.
    pub fn new(model: Box<dyn ChatModel + 'a>) -> Self {
        Self {
            model,
            history: Vec::new(),
        }
    }

    /// Run one user turn: append the message, re-render the FULL history through the model's chat
    /// template, generate the reply (streamed to `on_piece`), strip the `<think>` reasoning from what
    /// we store, and keep the answer for the next turn. Returns per-turn [`GenStats`].
    pub fn turn(
        &mut self,
        message: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.turn_with_step_hook(message, max_new, on_piece, None)
    }

    /// Like [`turn`](Self::turn), plus an optional per-denoise-step hook
    /// ([`crate::diffusion::StepView`]) forwarded to the backend's
    /// [`generate_with_step_hook`](ChatModel::generate_with_step_hook) — `infr run`'s
    /// `INFR_DIFFUSION_VISUAL` live canvas view is the only caller today. `on_step: None` behaves
    /// EXACTLY like `turn` (same default no-op path in every `ChatModel` impl).
    pub fn turn_with_step_hook(
        &mut self,
        message: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
        on_step: Option<&mut dyn FnMut(crate::diffusion::StepView)>,
    ) -> Result<GenStats> {
        self.history.push(("user".into(), message.to_string()));
        let prompt = {
            let refs: Vec<(&str, &str)> = self
                .history
                .iter()
                .map(|(r, c)| (r.as_str(), c.as_str()))
                .collect();
            match self.model.render(&refs) {
                Ok(p) => p,
                Err(e) => {
                    self.history.pop();
                    return Err(e);
                }
            }
        };
        let mut answer = String::new();
        let mut emit = |p: &str| {
            answer.push_str(p);
            on_piece(p);
        };
        // Template-prefilled thinking (Qwen3.5-style: the PROMPT ends with the `<think>` opener,
        // so the output starts mid-reasoning): inject a synthetic opener so the display styler
        // and the history stripper see a well-formed block.
        if infr_chat::prompt_prefills_think(&prompt) {
            emit("<think>");
        }
        // `req: None` — the interactive REPL is a sole sequence: sampling comes from the env/CLI
        // defaults, there is no stop-sequence latch, and there is no GPU to share.
        let stats = self
            .model
            .generate_with_step_hook(&prompt, max_new, None, &mut emit, on_step)?;
        self.history
            .push(("assistant".into(), strip_think(&answer)));
        Ok(stats)
    }

    /// Every backend now supports the interactive multi-turn REPL (kept for the CLI's call site).
    pub fn supports_repl(&self) -> bool {
        true
    }

    /// Optional status line (e.g. `ctx 12/8192`) for the REPL prompt.
    pub fn repl_status(&self) -> Option<String> {
        self.model.status()
    }
}

/// Standalone OpenAI-shaped prompt renderer over a GGUF's own chat template — tool specs and prior
/// tool calls/results included (`infr_chat::render_chat_oai`). Model-independent: the serve-side
/// render for seam-backed [`ChatModel`]s, so serve never needs the bespoke `Llama` just to render.
pub struct OaiRenderer {
    gguf: infr_gguf::Gguf,
    tokenizer: tokenizers::Tokenizer,
    eos: u32,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl OaiRenderer {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let gguf = infr_gguf::Gguf::open(path).map_err(|e| anyhow::anyhow!("open gguf: {e}"))?;
        let tokenizer = crate::build_tokenizer(&gguf)?;
        // Raw metadata (NOT Config::from_gguf — that parser is dense-only and rejects qwen35).
        use infr_core::WeightSource;
        let eos = gguf
            .metadata()
            .u64("tokenizer.ggml.eos_token_id")
            .unwrap_or(2) as u32;
        Ok(Self {
            gguf,
            tokenizer,
            eos,
        })
    }

    pub fn render(
        &self,
        messages: &[infr_chat::ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        infr_chat::render_chat_oai(&self.gguf, &self.tokenizer, self.eos, messages, tools, true)
            .map_err(|e| match e {
                // No embedded template at all → the standard "infr requires an instruct model" error.
                infr_chat::TemplateError::NoTemplate => no_template_err(),
                // The template EXISTS but failed to render → surface the actual jinja error so
                // serve's 500 body says what broke (not a generic "no usable template").
                e @ infr_chat::TemplateError::Render(_) => anyhow::anyhow!("{e}"),
            })
    }

    /// Build the FORCED tool-call grammar constraint for this model's tokenizer (see
    /// [`crate::grammar::tool_constraint_for`]); `None` for auto/none/absent tool_choice.
    pub fn tool_constraint(
        &self,
        tools: Option<&serde_json::Value>,
        tool_choice: Option<&str>,
    ) -> Result<Option<crate::grammar::Constraint>> {
        let vocab = self.tokenizer.get_vocab_size(true);
        crate::grammar::tool_constraint_for(&self.tokenizer, vocab, &[self.eos], tools, tool_choice)
    }

    /// Detokenize ids (markers preserved) — the serve-side parse of a constrained tool-call body.
    pub fn decode_ids(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scripted [`ChatModel`] that records every rendered prompt and replies with a canned string
    /// (optionally wrapped in `<think>`). Proves the shared [`Chat`] orchestration WITHOUT a model:
    /// history accumulates, both prior turns render into turn 2's prompt, and `<think>` is stripped
    /// from stored history but streamed to the caller.
    struct ScriptModel {
        rendered: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
        reply: String,
    }

    impl ChatModel for ScriptModel {
        fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
            // A trivial deterministic "template": one line per message.
            let s = messages
                .iter()
                .map(|(r, c)| format!("{r}: {c}"))
                .collect::<Vec<_>>()
                .join("\n");
            self.rendered.borrow_mut().push(s.clone());
            Ok(s)
        }

        fn generate(
            &mut self,
            _prompt: &str,
            _max_new: usize,
            _req: Option<&crate::sampling::RequestCtx>,
            on_piece: &mut dyn FnMut(&str),
        ) -> Result<GenStats> {
            on_piece(&self.reply);
            Ok(GenStats::default())
        }
    }

    #[test]
    fn shared_chat_accumulates_history_and_strips_think() {
        let rendered = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let model = ScriptModel {
            rendered: rendered.clone(),
            reply: "<think>reason</think>\n\nParis".into(),
        };
        let mut chat = Chat::new(Box::new(model));

        // Turn 1.
        let mut streamed = String::new();
        chat.turn("capital of France?", 8, &mut |p| streamed.push_str(p))
            .unwrap();
        // The caller sees the full reply (think included — the CLI's ThinkRender dims it live).
        assert_eq!(streamed, "<think>reason</think>\n\nParis");
        // Stored history keeps only the answer after </think>.
        assert_eq!(
            chat.history[0],
            ("user".into(), "capital of France?".into())
        );
        assert_eq!(chat.history[1], ("assistant".into(), "Paris".into()));

        // Turn 2: the rendered prompt must include BOTH prior turns (multi-turn context) with the
        // think-free assistant answer, then the new user message.
        chat.turn("and of Italy?", 8, &mut |_| {}).unwrap();
        let second = &rendered.borrow()[1];
        assert!(second.contains("user: capital of France?"), "{second}");
        assert!(second.contains("assistant: Paris"), "{second}");
        assert!(
            !second.contains("reason"),
            "prior think must be excluded: {second}"
        );
        assert!(second.contains("user: and of Italy?"), "{second}");
    }

    #[test]
    fn strip_think_variants() {
        // Reasoning spans stripped (ALL of them), content concatenated and trimmed.
        assert_eq!(strip_think("<think>a</think>b"), "b");
        assert_eq!(strip_think("<think>a</think>x<think>c</think>y"), "xy");
        // Unterminated think → nothing.
        assert_eq!(strip_think("<think>a"), "");
        // No think → the whole reply.
        assert_eq!(strip_think("plain"), "plain");
    }
}
