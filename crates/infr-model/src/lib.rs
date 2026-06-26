//! Model seam + DiffusionGemma. A `Model` reads weights/metadata and builds the
//! backend-agnostic forward `Graph`.
//!
//! Reference: `~/Projects/llama.cpp/src/llama-model.cpp` (arch `diffusion-gemma`).
//! Spec (from GGUF): 30 layers, hidden 2816, vocab 262144, head_count 16, head_count_kv
//! `[8,8,8,8,8,2]×5` (layers 5/11/17/23/29 full, rest sliding-window), key/value_length
//! 512 (full) / 256 (swa), canvas_length 256, mask_token 4. See PLAN.md "DiffusionGemma spec".
#![allow(dead_code, unused_variables)]

use infr_core::{backend::Capabilities, error::Result, graph::Graph, WeightSource};

/// Static model shape read from GGUF metadata.
#[derive(Clone, Debug, Default)]
pub struct ModelConfig {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: Vec<usize>, // per-layer (2 == full attn, else sliding-window)
    pub hidden: usize,
    pub vocab: usize,
    pub head_dim_full: usize,
    pub head_dim_swa: usize,
    pub swa_window: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub canvas_len: usize,
    pub mask_token: u32,
}

/// Inputs needed to build a forward graph for one decode step/batch.
pub struct BuildCtx<'a> {
    pub weights: &'a dyn WeightSource,
    pub caps: &'a Capabilities,
    /// Total sequence length (prompt prefix + canvas) for this graph.
    pub seq_len: usize,
}

pub trait Model: Send + Sync {
    fn config(&self) -> &ModelConfig;
    /// Build the transformer forward graph (embed → layers → norm → logits).
    fn build_graph(&self, ctx: &BuildCtx) -> Result<Graph>;
}

pub struct DiffusionGemma {
    cfg: ModelConfig,
}

impl DiffusionGemma {
    /// Read `diffusion-gemma.*` metadata into a [`ModelConfig`].
    ///
    /// TODO(sonnet): pull keys from `weights.metadata()` (block_count, embedding_length,
    /// attention.head_count[_kv], attention.key/value_length[_swa], rope freq, rms eps,
    /// diffusion.canvas_length, mask token).
    pub fn from_weights(weights: &dyn WeightSource) -> Result<Self> {
        todo!("parse diffusion-gemma metadata into ModelConfig")
    }
}

impl Model for DiffusionGemma {
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }
    fn build_graph(&self, ctx: &BuildCtx) -> Result<Graph> {
        // TODO(sonnet): embed -> 30 × (RMSNorm, GQA attn [full|swa], RMSNorm, MoE FFN) ->
        // final RMSNorm -> output projection. Use ctx.weights tensors + ctx.caps to pick
        // fast vs fallback ops.
        todo!("build DiffusionGemma forward graph")
    }
}

#[cfg(test)]
mod tests {
    // TODO(sonnet): once GGUF load works, assert from_weights() parses the known config
    // (n_layer 30, full-attn layers at indices 5/11/17/23/29, mask_token 4).
}
