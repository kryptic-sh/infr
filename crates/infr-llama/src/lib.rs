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
pub use seam::model::SeamModel;
pub mod diffusion;
mod util;
pub use util::gpu_available;
pub(crate) use util::*;
mod kv;
pub use kv::MoeConfig;
pub mod grammar;
/// N-slot concurrent generation (`infr serve --parallel N`) — see [`parallel::ParallelSeam`].
pub mod parallel;
pub mod sampling;
pub use config::Config;
mod weights;
pub use weights::{weight_footprint, WeightFootprint};
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

/// Per-turn generation timing — backend-agnostic (shared by the CPU runner and the GPU
/// `ChatTurn` path). `prompt_secs` = prefill (time to first token), `decode_secs` = generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenStats {
    pub n_prompt: usize,
    pub prompt_secs: f64,
    pub n_gen: usize,
    pub decode_secs: f64,
}
