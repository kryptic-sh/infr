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

use crate::{no_template_err, ChatSession, CpuModel, GenStats, Llama};
use anyhow::Result;
use std::time::Instant;

/// The two arch-specific primitives the shared [`Chat`] drives. Object-safe: no generics, callbacks
/// are `&mut dyn FnMut`. A stateful backend (dense) may keep a KV cache warm across `generate` calls;
/// stateless ones prefill the whole prompt each turn.
pub trait ChatModel {
    /// Render a conversation `(role, content)` into a prompt string via the model's embedded chat
    /// template. All backends funnel to `infr_chat::render_chat_jinja` (the single prompt renderer).
    fn render(&self, messages: &[(&str, &str)]) -> Result<String>;

    /// Generate a completion for the already-rendered `prompt`, streaming decoded text to `on_piece`.
    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats>;

    /// Optional REPL status (e.g. dense returns `ctx N/MAX`); `None` for stateless backends.
    fn status(&self) -> Option<String> {
        None
    }
}

/// Store only the ANSWER, dropping the model's `<think>…</think>` reasoning (Qwen3 excludes
/// prior-turn thinking; keeping it degrades the model). Mirrors the dense token-based logic: text
/// after the LAST `</think>`; a `<think>` with no `</think>` (unterminated) → empty; no `<think>` at
/// all → the whole reply.
fn strip_think(reply: &str) -> String {
    const CLOSE: &str = "</think>";
    if let Some(pos) = reply.rfind(CLOSE) {
        reply[pos + CLOSE.len()..].to_string()
    } else if reply.contains("<think>") {
        String::new()
    } else {
        reply.to_string()
    }
}

/// The SINGLE arch-agnostic chat: owns the conversation history + `<think>`-stripping, drives any
/// [`ChatModel`] backend. Every architecture uses this one turn implementation and REPL, so a working
/// multi-turn REPL is uniform (no per-backend "one-shot only" special cases).
pub struct Chat<'a> {
    model: Box<dyn ChatModel + 'a>,
    history: Vec<(String, String)>,
}

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
        let stats = self.model.generate(&prompt, max_new, &mut |p| {
            answer.push_str(p);
            on_piece(p);
        })?;
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

/// Wrap a callback-driven generate: measure time-to-first-piece + decode time, count pieces. Used by
/// the stateless backends (dense computes its own timing inside `ChatSession::generate`).
fn timed(
    on_piece: &mut dyn FnMut(&str),
    n_prompt: usize,
    run: impl FnOnce(&mut dyn FnMut(&str)) -> Result<()>,
) -> Result<GenStats> {
    let t0 = Instant::now();
    let mut t_first: Option<Instant> = None;
    let mut n_gen = 0usize;
    let mut cb = |p: &str| {
        if t_first.is_none() {
            t_first = Some(Instant::now());
        }
        n_gen += 1;
        on_piece(p);
    };
    run(&mut cb)?;
    let now = Instant::now();
    let tf = t_first.unwrap_or(now);
    Ok(GenStats {
        n_prompt,
        prompt_secs: tf.duration_since(t0).as_secs_f64(),
        n_gen,
        decode_secs: now.duration_since(tf).as_secs_f64(),
    })
}

/// Dense Qwen3/Llama/Gemma: the stateful multi-turn `ChatSession` over a persistent KV cache. Its
/// `generate` keeps the cache warm — only the token suffix that differs from the cached prefix is
/// prefilled (incremental prefill), so KV-cache reuse survives the shared-orchestration refactor.
impl ChatModel for ChatSession<'_> {
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        ChatSession::render(self, messages)
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        ChatSession::generate(self, prompt, max_new, |p| on_piece(p))
    }

    fn status(&self) -> Option<String> {
        Some(format!("ctx {}/{}", self.ctx_len(), self.max_ctx()))
    }
}

/// qwen3moe: the eager MoE forward (`generate_moe`) with a GPU KV cache. Stateless full-prefill each
/// turn (MoE KV-reuse via `forward_moe_chunk` is a separate future item) — acceptable, the point is
/// the shared history-based REPL.
pub struct MoeChat<'a> {
    llama: &'a Llama,
}

impl<'a> MoeChat<'a> {
    pub fn new(llama: &'a Llama) -> Self {
        Self { llama }
    }
}

impl ChatModel for MoeChat<'_> {
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        self.llama
            .render_chat_messages(messages, true)
            .ok_or_else(no_template_err)
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        let n_prompt = self
            .llama
            .tokenizer
            .encode(prompt, false)
            .map(|e| e.get_ids().len())
            .unwrap_or(0);
        timed(on_piece, n_prompt, |cb| {
            self.llama.generate_moe(prompt, max_new, cb).map(|_| ())
        })
    }
}

/// qwen35 / Qwen3-Next GPU: the bespoke per-token hybrid forward. Stateless full-prefill each turn.
/// FOLLOW-UP: `qwen35::generate_chat` reloads the GGUF + rebuilds the model per turn (pre-existing
/// wart); a load-once persistent qwen35 model behind this trait would drop that reload.
pub struct Qwen35Chat {
    path: std::path::PathBuf,
}

impl Qwen35Chat {
    pub fn new(path: std::path::PathBuf) -> Self {
        Self { path }
    }
}

impl ChatModel for Qwen35Chat {
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        crate::qwen35::render_chat_messages(&self.path, messages)
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        let mut n_prompt = 0usize;
        timed(on_piece, 0, |cb| {
            let (np, _ng) = crate::qwen35::generate_chat(&self.path, prompt, max_new, cb)?;
            n_prompt = np;
            Ok(())
        })
        .map(|mut s| {
            s.n_prompt = n_prompt;
            s
        })
    }
}

/// CPU reference backend (`INFR_CPU=1`) for dense/MoE: the agnostic compute-graph forward, no GPU.
/// Stateless full-prefill each turn (no cross-turn KV yet), but the shared `Chat` now feeds the FULL
/// rendered history in every turn, so multi-turn context works.
pub struct CpuDenseChat {
    model: CpuModel,
    /// Run the dense forward on the reference Metal backend instead of the CPU interpreter.
    metal: bool,
}

impl CpuDenseChat {
    pub fn new(model: CpuModel) -> Self {
        Self {
            model,
            metal: false,
        }
    }

    /// Same dense model, but driven through the reference Metal backend (`INFR_METAL`).
    pub fn new_metal(model: CpuModel) -> Self {
        Self { model, metal: true }
    }
}

impl ChatModel for CpuDenseChat {
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        self.model.render_chat_messages(messages)
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.metal {
            #[cfg(target_os = "macos")]
            return self.model.generate_metal(prompt, max_new, |p| on_piece(p));
            #[cfg(not(target_os = "macos"))]
            return Err(anyhow::anyhow!(
                "the Metal backend is only available on macOS"
            ));
        }
        self.model.generate_cpu(prompt, max_new, |p| on_piece(p))
    }
}

/// CPU reference backend (`INFR_CPU=1`) for qwen35 / Qwen3-Next (the bespoke per-token oracle).
/// Same follow-up as [`Qwen35Chat`]: `qwen35::generate_cpu` reloads the GGUF per turn.
pub struct CpuQwen35Chat {
    path: std::path::PathBuf,
    /// Run the qwen35 decode on the reference Metal backend instead of the CPU interpreter.
    metal: bool,
}

impl CpuQwen35Chat {
    pub fn new(path: std::path::PathBuf) -> Self {
        Self { path, metal: false }
    }

    /// Same qwen35 decode, driven through the reference Metal backend (`INFR_METAL`).
    pub fn new_metal(path: std::path::PathBuf) -> Self {
        Self { path, metal: true }
    }
}

impl ChatModel for CpuQwen35Chat {
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        crate::qwen35::render_chat_messages(&self.path, messages)
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.metal {
            #[cfg(target_os = "macos")]
            return crate::qwen35::generate_metal(&self.path, prompt, max_new, |p| on_piece(p));
            #[cfg(not(target_os = "macos"))]
            return Err(anyhow::anyhow!(
                "the Metal backend is only available on macOS"
            ));
        }
        crate::qwen35::generate_cpu(&self.path, prompt, max_new, |p| on_piece(p))
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
        assert_eq!(chat.history[1], ("assistant".into(), "\n\nParis".into()));

        // Turn 2: the rendered prompt must include BOTH prior turns (multi-turn context) with the
        // think-free assistant answer, then the new user message.
        chat.turn("and of Italy?", 8, &mut |_| {}).unwrap();
        let second = &rendered.borrow()[1];
        assert!(second.contains("user: capital of France?"), "{second}");
        assert!(second.contains("assistant: \n\nParis"), "{second}");
        assert!(
            !second.contains("reason"),
            "prior think must be excluded: {second}"
        );
        assert!(second.contains("user: and of Italy?"), "{second}");
    }

    #[test]
    fn strip_think_variants() {
        // After the last </think>.
        assert_eq!(strip_think("<think>a</think>b"), "b");
        assert_eq!(strip_think("<think>a</think>x<think>c</think>y"), "y");
        // Unterminated think → nothing.
        assert_eq!(strip_think("<think>a"), "");
        // No think → the whole reply.
        assert_eq!(strip_think("plain"), "plain");
    }
}
