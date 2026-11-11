//! Minimal autoregressive **Llama** inference for GGUF models, for fast GPU bring-up.
//!
//! Strategy (bring-up): the heavy linear projections run on the GPU (`infr-vulkan` eager
//! `linear`, weights uploaded once); the cheap ops (embedding gather, RMSNorm, RoPE, GQA
//! attention, SwiGLU, residual, sampling) run on the host. No KV cache yet — each step does a
//! full-prefix forward (fine for a tiny model). Validated on SmolLM2-135M.
//!
//! TODO(next): move host ops to GPU; add a KV cache; fold into the `Model`/`Backend` seams.
#![allow(clippy::needless_range_loop)]

pub mod arch;
pub mod chat;
mod config;
pub mod seam;
pub use seam::model::{BenchPlacement, SeamModel};
pub mod diffusion;
mod util;
pub(crate) use util::*;
mod kv;
pub use kv::MoeConfig;
pub mod grammar;
/// N-slot concurrent generation (`infr serve --parallel N`) — see [`parallel::ParallelSeam`].
pub mod parallel;
pub mod sampling;
pub use config::Config;

/// The ENGINE configuration ([`infr_core::config::Config`]), re-exported under a name that cannot
/// be confused with this crate's own MODEL [`Config`] (the GGUF-derived architecture description).
///
/// Since the config migration, every `INFR_*` knob this crate used to read out of the process
/// environment — `kv.*`, `spec.*`, `sampling.*`, `multi.*`, `device.ubatch*`, `paging.cache`, and
/// the seam's `prof.*` diagnostics — is read from a value of this type, owned by [`SeamModel`] and
/// by each [`seam::model::DenseSession`] and BORROWED at every read site (R4/R6: no globals, no
/// per-token clone). Two models in one process can hold different configurations.
pub use infr_core::config::Config as EngineConfig;
mod weights;
pub use weights::{weight_footprint, WeightFootprint};

/// Notices a weight file being overwritten under a live mapping — see [`infr_gguf::watch`].
///
/// Re-exported here because the CLI reaches every GGUF concern through this crate rather than
/// depending on `infr-gguf` directly, and a watch is opened beside each model load.
pub use infr_gguf::watch::WeightWatch;
pub mod mtp;
mod quant;
pub mod qwen35;
mod tokenizer;
pub(crate) use quant::*;
pub(crate) use tokenizer::*;

/// Metadata-only peek: is this GGUF a Llama 4 model (`arch::LLAMA4`)? llama4's MoE routing
/// (sigmoid / no-renorm / weight-before-FFN) + iRoPE are CPU-only for now — the GPU `MoeFfn`
/// lowerings assert on the non-default routing — so the CLI uses this to force the CPU backend
/// instead of panicking mid-lowering. Mirrors `diffusion::is_diffusion_gemma`.
pub fn is_llama4(path: &std::path::Path) -> bool {
    use infr_core::WeightSource;
    infr_gguf::Gguf::open(path)
        .ok()
        .map(|g| g.metadata().str("general.architecture") == Some(arch::LLAMA4))
        .unwrap_or(false)
}

/// The prompt plus the reply budget does not fit the KV context. Typed rather than a bare message
/// so `infr serve` can answer it as an over-long REQUEST (the client's to fix) instead of a server
/// fault the client would retry unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextOverflow {
    pub prompt: usize,
    pub gen: usize,
    pub max_ctx: usize,
}

impl std::fmt::Display for ContextOverflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "prompt {} + gen {} exceeds the session KV capacity {}",
            self.prompt, self.gen, self.max_ctx
        )
    }
}

impl std::error::Error for ContextOverflow {}

/// Per-turn generation timing — backend-agnostic (shared by the CPU runner and the GPU
/// `ChatTurn` path). `prompt_secs` = prefill (time to first token), `decode_secs` = generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenStats {
    /// Prompt tokens PREFILLED by this call. On a KV prefix-cache hit that is only the un-cached
    /// suffix; the whole prompt is `n_prompt + n_cached`.
    pub n_prompt: usize,
    pub prompt_secs: f64,
    pub n_gen: usize,
    pub decode_secs: f64,
    /// Prompt tokens reused from the KV prefix cache instead of prefilled.
    pub n_cached: usize,
    /// The reply ran out of context, not out of the caller's budget: the engine lowered `max_new`
    /// to the room the context had left, and the generation used all of that lowered budget. The
    /// caller's own `n_gen >= max_new` test cannot see this case.
    pub ctx_full: bool,
    /// The generation ended itself — it sampled EOS, or a grammar-constrained span completed — so
    /// it did not run out of budget even when `n_gen` reached it (the EOS token is counted).
    /// Reported by the dense seam runner; `false` from paths that do not track it.
    pub natural_stop: bool,
}

/// Whether a generation that asked for `asked` tokens, was given only `budget` because the
/// context had no more room, and ended with `stats`, was cut off by the context.
pub(crate) fn ran_out_of_context(asked: usize, budget: usize, stats: &GenStats) -> bool {
    budget < asked && stats.n_gen >= budget && !stats.natural_stop
}

#[cfg(test)]
mod tests {
    use super::{ran_out_of_context, GenStats};

    fn stats(n_gen: usize, natural_stop: bool) -> GenStats {
        GenStats {
            n_gen,
            natural_stop,
            ..Default::default()
        }
    }

    /// The reply used every token the context had left: cut off. The same count ending on EOS is
    /// the model finishing (the EOS token is counted in `n_gen`), and a budget the context did not
    /// lower is the caller's own `max_tokens`, not the context.
    #[test]
    fn only_a_lowered_budget_used_up_without_eos_is_a_context_cut() {
        assert!(ran_out_of_context(2048, 95, &stats(95, false)));
        assert!(!ran_out_of_context(2048, 95, &stats(95, true)));
        assert!(!ran_out_of_context(2048, 95, &stats(40, false)));
        assert!(!ran_out_of_context(2048, 2048, &stats(2048, false)));
    }
}
