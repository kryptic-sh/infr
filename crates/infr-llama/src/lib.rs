//! Minimal autoregressive **Llama** inference for GGUF models, for fast GPU bring-up.
//!
//! Strategy (bring-up): the heavy linear projections run on the GPU (`infr-vulkan` eager
//! `linear`, weights uploaded once); the cheap ops (embedding gather, RMSNorm, RoPE, GQA
//! attention, SwiGLU, residual, sampling) run on the host. No KV cache yet — each step does a
//! full-prefix forward (fine for a tiny model). Validated on SmolLM2-135M.
//!
//! TODO(next): move host ops to GPU; add a KV cache; fold into the `Model`/`Backend` seams.
#![allow(clippy::needless_range_loop)]

mod config;
pub mod cpu_backend;
mod cpu_model;
pub use cpu_model::CpuModel;
mod util;
pub use util::gpu_available;
pub(crate) use util::*;
mod transformer;
pub(crate) use transformer::PerLayerEmbd;
pub use transformer::{ChatSession, Llama, ServeCache};
mod kv;
pub(crate) use kv::{DecodeScratch, DenseDecodeScratch, PrefillScratch, QBufs};
pub use kv::{KvCache, MoeConfig, MoeKv};
pub mod grammar;
mod sampling;
pub use config::Config;
pub(crate) use sampling::*;
mod weights;
pub(crate) use weights::*;
pub use weights::{weight_footprint, WeightFootprint};
mod mixers;
pub mod model;
mod quant;
pub mod qwen35;
mod tokenizer;
pub(crate) use quant::*;
pub(crate) use tokenizer::*;

/// Per-turn generation timing — backend-agnostic (shared by the CPU runner and the GPU
/// `ChatTurn` path). `prompt_secs` = prefill (time to first token), `decode_secs` = generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenStats {
    pub n_prompt: usize,
    pub prompt_secs: f64,
    pub n_gen: usize,
    pub decode_secs: f64,
}
