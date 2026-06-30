//! Agnostic chat orchestration: one [`ChatTurn`] interface over every architecture's generation
//! path, so the CLI dispatches uniformly instead of branching per arch.
//!
//! Each backend keeps its own bespoke forward (dense `ChatSession` with a persistent KV cache, the
//! eager `generate_moe`, qwen35's per-token loop); this trait just unifies "run one chat turn from a
//! message, stream decoded pieces, report stats". The boxed trait object is lifetime-bounded so the
//! borrow-based `ChatSession` (which borrows `&Llama`) needs no ownership change — the caller owns
//! the `Llama`, the box borrows it.

use crate::{ChatSession, Llama};
use anyhow::Result;
use std::time::Instant;

/// Per-turn generation stats (mirrors `cpu_backend::CpuStats`).
pub struct GenStats {
    pub n_prompt: usize,
    pub prompt_secs: f64,
    pub n_gen: usize,
    pub decode_secs: f64,
}

/// One chat turn, arch-agnostic: render the prompt, generate, stream decoded text pieces, report
/// stats. The boxed form is `Box<dyn ChatTurn + '_>` so it may borrow a caller-owned `Llama`.
pub trait ChatTurn {
    /// Run a turn from `message`; `on_piece` receives raw decoded text deltas.
    fn turn(
        &mut self,
        message: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats>;

    /// Whether this backend supports an interactive multi-turn REPL (else one-shot only).
    fn supports_repl(&self) -> bool {
        false
    }

    /// Optional status (e.g. `ctx 12/8192`) shown in the REPL prompt.
    fn repl_status(&self) -> Option<String> {
        None
    }
}

/// Wrap a callback-driven generate: measure time-to-first-piece + decode time, count pieces.
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

/// Dense Qwen3/Llama/Gemma: the stateful multi-turn `ChatSession` over a persistent KV cache.
impl ChatTurn for ChatSession<'_> {
    fn turn(
        &mut self,
        message: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        let mut stats = timed(on_piece, 0, |cb| {
            ChatSession::turn(self, message, max_new, cb).map(|_| ())
        })?;
        stats.n_prompt = self.last_prompt_tokens;
        Ok(stats)
    }

    fn supports_repl(&self) -> bool {
        true
    }

    fn repl_status(&self) -> Option<String> {
        Some(format!("ctx {}/{}", self.ctx_len(), self.max_ctx()))
    }
}

/// qwen3moe: the eager MoE forward (`generate_moe`) with a host KV cache. One-shot.
pub struct MoeChat<'a> {
    llama: &'a Llama,
}

impl<'a> MoeChat<'a> {
    pub fn new(llama: &'a Llama) -> Self {
        Self { llama }
    }
}

impl ChatTurn for MoeChat<'_> {
    fn turn(
        &mut self,
        message: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        let prompt = self.llama.render_chat(message)?;
        let n_prompt = self
            .llama
            .tokenizer
            .encode(prompt.as_str(), false)
            .map(|e| e.get_ids().len())
            .unwrap_or(0);
        timed(on_piece, n_prompt, |cb| {
            self.llama.generate_moe(&prompt, max_new, cb).map(|_| ())
        })
    }
}

/// qwen35 / Qwen3-Next: the bespoke per-token hybrid/GPU forward (`qwen35::generate_chat`). One-shot.
pub struct Qwen35Chat {
    path: std::path::PathBuf,
}

impl Qwen35Chat {
    pub fn new(path: std::path::PathBuf) -> Self {
        Self { path }
    }
}

impl ChatTurn for Qwen35Chat {
    fn turn(
        &mut self,
        message: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        let mut n_prompt = 0usize;
        let mut stats = timed(on_piece, 0, |cb| {
            let (np, _ng) = crate::qwen35::generate_chat(&self.path, message, max_new, cb)?;
            n_prompt = np;
            Ok(())
        })?;
        stats.n_prompt = n_prompt;
        Ok(stats)
    }
}
