//! Model seam + DiffusionGemma. A `Model` reads weights/metadata and builds the
//! backend-agnostic forward `Graph`.
//!
//! Reference: `~/Projects/llama.cpp/src/llama-model.cpp` (arch `diffusion-gemma`).
//! Spec (from GGUF): 30 layers, hidden 2816, vocab 262144, head_count 16, head_count_kv
//! `[8,8,8,8,8,2]×5` (layers 5/11/17/23/29 full, rest sliding-window), key/value_length
//! 512 (full) / 256 (swa), canvas_length 256, mask_token 4. See docs/PLAN.md "DiffusionGemma spec".
#![allow(dead_code, unused_variables)]

use infr_core::{
    backend::Capabilities,
    error::{Error, Result},
    graph::Graph,
    loader::MetaValue,
    WeightSource,
};

// ─── constants (confirmed from real GGUF dump) ────────────────────────────────

const ARCH: &str = "diffusion-gemma";

// Metadata key strings — all confirmed against the Q4_K_M GGUF file.
const KEY_ARCH: &str = "general.architecture";
const KEY_BLOCK_COUNT: &str = "diffusion-gemma.block_count";
const KEY_EMBEDDING_LENGTH: &str = "diffusion-gemma.embedding_length";
const KEY_HEAD_COUNT: &str = "diffusion-gemma.attention.head_count";
const KEY_HEAD_COUNT_KV: &str = "diffusion-gemma.attention.head_count_kv";
const KEY_KEY_LENGTH: &str = "diffusion-gemma.attention.key_length";
const KEY_KEY_LENGTH_SWA: &str = "diffusion-gemma.attention.key_length_swa";
const KEY_SLIDING_WINDOW: &str = "diffusion-gemma.attention.sliding_window";
const KEY_ROPE_FREQ_BASE: &str = "diffusion-gemma.rope.freq_base";
const KEY_RMS_EPS: &str = "diffusion-gemma.attention.layer_norm_rms_epsilon";
const KEY_CANVAS_LENGTH: &str = "diffusion.canvas_length"; // no arch prefix
const KEY_MASK_TOKEN: &str = "tokenizer.ggml.mask_token_id";
const KEY_VOCAB_SIZE: &str = "diffusion-gemma.vocab_size"; // may not exist
const KEY_TOKENS: &str = "tokenizer.ggml.tokens"; // fallback for vocab count

// ─── model types ──────────────────────────────────────────────────────────────

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

impl ModelConfig {
    /// Indices (0-based) of layers that use full (global) attention.
    ///
    /// Layers where `n_head_kv[i] == 2` are full-attention; the rest are SWA.
    pub fn full_attn_layers(&self) -> Vec<usize> {
        self.n_head_kv
            .iter()
            .enumerate()
            .filter(|(_, &kv)| kv == 2)
            .map(|(i, _)| i)
            .collect()
    }
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

#[derive(Debug)]
pub struct DiffusionGemma {
    cfg: ModelConfig,
}

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Extract an f32 from a MetaValue::F64 (all GGUF f32s are widened to F64 by the reader).
#[inline]
fn meta_f32(v: &MetaValue) -> Option<f32> {
    match v {
        MetaValue::F64(f) => Some(*f as f32),
        MetaValue::U64(u) => Some(*u as f32),
        MetaValue::I64(i) => Some(*i as f32),
        _ => None,
    }
}

// ─── DiffusionGemma ───────────────────────────────────────────────────────────

impl DiffusionGemma {
    /// Parse `diffusion-gemma.*` metadata into a [`ModelConfig`].
    ///
    /// # Keys read (confirmed from Q4_K_M GGUF dump)
    ///
    /// | Key                                              | Type        | Field          |
    /// |--------------------------------------------------|-------------|----------------|
    /// | `general.architecture`                           | str         | required check |
    /// | `diffusion-gemma.block_count`                    | u32→U64     | n_layer        |
    /// | `diffusion-gemma.embedding_length`               | u32→U64     | hidden         |
    /// | `diffusion-gemma.attention.head_count`           | u32→U64     | n_head         |
    /// | `diffusion-gemma.attention.head_count_kv`        | arr\[i32\]  | n_head_kv      |
    /// | `diffusion-gemma.attention.key_length`           | u32→U64     | head_dim_full  |
    /// | `diffusion-gemma.attention.key_length_swa`       | u32→U64     | head_dim_swa   |
    /// | `diffusion-gemma.attention.sliding_window`       | u32→U64     | swa_window     |
    /// | `diffusion-gemma.rope.freq_base`                 | f32→F64     | rope_theta     |
    /// | `diffusion-gemma.attention.layer_norm_rms_epsilon`| f32→F64   | rms_eps        |
    /// | `diffusion.canvas_length`                        | u32→U64     | canvas_len     |
    /// | `tokenizer.ggml.mask_token_id`                   | u32→U64     | mask_token     |
    pub fn from_weights(weights: &dyn WeightSource) -> Result<Self> {
        let meta = weights.metadata();

        // ── architecture guard ────────────────────────────────────────────────
        let arch = meta
            .str(KEY_ARCH)
            .ok_or_else(|| Error::Model(format!("missing '{KEY_ARCH}' in metadata")))?;
        if arch != ARCH {
            return Err(Error::Model(format!(
                "expected architecture '{ARCH}', got '{arch}'"
            )));
        }

        // ── required: block_count ─────────────────────────────────────────────
        let n_layer = meta
            .u64(KEY_BLOCK_COUNT)
            .ok_or_else(|| Error::Model(format!("missing '{KEY_BLOCK_COUNT}' in metadata")))?
            as usize;

        // ── required: per-layer head_count_kv ────────────────────────────────
        // In practice this is always an array of INT32 values (stored as I64 by the reader).
        let n_head_kv: Vec<usize> = match meta.get(KEY_HEAD_COUNT_KV) {
            Some(MetaValue::Arr(arr)) => arr
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect(),
            Some(scalar) => {
                // Fall back: scalar repeated for all layers.
                let kv = scalar.as_u64().unwrap_or(0) as usize;
                vec![kv; n_layer]
            }
            None => {
                return Err(Error::Model(format!(
                    "missing '{KEY_HEAD_COUNT_KV}' in metadata"
                )))
            }
        };

        if n_head_kv.len() != n_layer {
            return Err(Error::Model(format!(
                "head_count_kv length {} != block_count {}",
                n_head_kv.len(),
                n_layer
            )));
        }

        // ── scalars with sensible fallbacks ───────────────────────────────────
        let hidden = meta.u64(KEY_EMBEDDING_LENGTH).unwrap_or(2816) as usize;
        let n_head = meta.u64(KEY_HEAD_COUNT).unwrap_or(16) as usize;
        let head_dim_full = meta.u64(KEY_KEY_LENGTH).unwrap_or(512) as usize;
        let head_dim_swa = meta.u64(KEY_KEY_LENGTH_SWA).unwrap_or(256) as usize;
        let swa_window = meta.u64(KEY_SLIDING_WINDOW).unwrap_or(1024) as usize;
        let canvas_len = meta.u64(KEY_CANVAS_LENGTH).unwrap_or(256) as usize;
        let mask_token = meta.u64(KEY_MASK_TOKEN).unwrap_or(4) as u32;

        // ── floats (GGUF stores f32 as F64 in MetaValue) ─────────────────────
        let rope_theta = meta
            .get(KEY_ROPE_FREQ_BASE)
            .and_then(meta_f32)
            .unwrap_or(1_000_000.0_f32);

        let rms_eps = meta.get(KEY_RMS_EPS).and_then(meta_f32).unwrap_or(1e-6_f32);

        // ── vocab size: arch key (may not exist) → count token list ──────────
        let vocab = meta
            .u64(KEY_VOCAB_SIZE)
            .map(|v| v as usize)
            .or_else(|| {
                meta.get(KEY_TOKENS)
                    .and_then(|v| v.as_arr())
                    .map(|a| a.len())
            })
            .unwrap_or(262_144);

        Ok(Self {
            cfg: ModelConfig {
                n_layer,
                n_head,
                n_head_kv,
                hidden,
                vocab,
                head_dim_full,
                head_dim_swa,
                swa_window,
                rope_theta,
                rms_eps,
                canvas_len,
                mask_token,
            },
        })
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

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::{
        error::Result,
        loader::{MetaValue, Metadata, TensorInfo},
    };
    use std::collections::HashMap;

    // ── mock WeightSource backed by a hand-built Metadata ────────────────────

    struct MockWeights {
        meta: Metadata,
    }

    impl WeightSource for MockWeights {
        fn metadata(&self) -> &Metadata {
            &self.meta
        }
        fn tensors(&self) -> &[TensorInfo] {
            &[]
        }
        fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
            Err(Error::Loader(format!("mock: no tensor '{name}'")))
        }
        fn chat_template(&self) -> Option<&str> {
            None
        }
    }

    /// Build a Metadata map that matches the real GGUF's structure.
    ///
    /// head_count_kv pattern: `[8, 8, 8, 8, 8, 2]` × 5 = 30 layers.
    /// Full-attention at indices 5, 11, 17, 23, 29 (kv == 2).
    fn mock_meta(n_layer: usize) -> Metadata {
        let pattern = [8i64, 8, 8, 8, 8, 2];
        let kv_arr: Vec<MetaValue> = (0..n_layer)
            .map(|i| MetaValue::I64(pattern[i % pattern.len()]))
            .collect();

        let mut kv: HashMap<String, MetaValue> = HashMap::new();
        kv.insert(KEY_ARCH.into(), MetaValue::Str("diffusion-gemma".into()));
        kv.insert(KEY_BLOCK_COUNT.into(), MetaValue::U64(n_layer as u64));
        kv.insert(KEY_EMBEDDING_LENGTH.into(), MetaValue::U64(2816));
        kv.insert(KEY_HEAD_COUNT.into(), MetaValue::U64(16));
        kv.insert(KEY_HEAD_COUNT_KV.into(), MetaValue::Arr(kv_arr));
        kv.insert(KEY_KEY_LENGTH.into(), MetaValue::U64(512));
        kv.insert(KEY_KEY_LENGTH_SWA.into(), MetaValue::U64(256));
        kv.insert(KEY_SLIDING_WINDOW.into(), MetaValue::U64(1024));
        kv.insert(KEY_ROPE_FREQ_BASE.into(), MetaValue::F64(1_000_000.0));
        kv.insert(KEY_RMS_EPS.into(), MetaValue::F64(1e-6));
        kv.insert(KEY_CANVAS_LENGTH.into(), MetaValue::U64(256));
        kv.insert(KEY_MASK_TOKEN.into(), MetaValue::U64(4));
        // No KEY_VOCAB_SIZE → falls back to KEY_TOKENS count; skip both → default 262144
        Metadata { kv }
    }

    // ── unit: mock WeightSource, 30 layers ───────────────────────────────────

    #[test]
    fn from_weights_mock() {
        let weights = MockWeights {
            meta: mock_meta(30),
        };
        let model = DiffusionGemma::from_weights(&weights).expect("from_weights failed");
        let cfg = model.config();

        assert_eq!(cfg.n_layer, 30, "n_layer");
        assert_eq!(cfg.hidden, 2816, "hidden");
        assert_eq!(cfg.n_head, 16, "n_head");
        assert_eq!(cfg.n_head_kv.len(), 30, "n_head_kv length");
        assert_eq!(cfg.head_dim_full, 512, "head_dim_full");
        assert_eq!(cfg.head_dim_swa, 256, "head_dim_swa");
        assert_eq!(cfg.swa_window, 1024, "swa_window");
        assert_eq!(cfg.canvas_len, 256, "canvas_len");
        assert_eq!(cfg.mask_token, 4, "mask_token");
        assert_eq!(cfg.vocab, 262_144, "vocab (default)");

        // rope_theta: stored as F64(1e6) → cast to f32
        assert!(
            (cfg.rope_theta - 1_000_000.0_f32).abs() < 1.0,
            "rope_theta ≈ 1e6, got {}",
            cfg.rope_theta
        );

        // rms_eps: F64(1e-6) → f32
        assert!(
            (cfg.rms_eps - 1e-6_f32).abs() < 1e-9,
            "rms_eps ≈ 1e-6, got {}",
            cfg.rms_eps
        );

        // Full-attention layers: indices where kv == 2 in pattern [8,8,8,8,8,2]×5
        let full = cfg.full_attn_layers();
        assert_eq!(
            full,
            vec![5, 11, 17, 23, 29],
            "full-attention layer indices"
        );

        // Spot-check the per-layer values
        assert_eq!(cfg.n_head_kv[0], 8, "layer 0 is SWA (kv=8)");
        assert_eq!(cfg.n_head_kv[5], 2, "layer 5 is full-attn (kv=2)");
        assert_eq!(cfg.n_head_kv[29], 2, "layer 29 is full-attn (kv=2)");
    }

    // ── unit: architecture mismatch returns Err ───────────────────────────────

    #[test]
    fn from_weights_wrong_arch() {
        let mut kv: HashMap<String, MetaValue> = HashMap::new();
        kv.insert(KEY_ARCH.into(), MetaValue::Str("llama".into()));
        kv.insert(KEY_BLOCK_COUNT.into(), MetaValue::U64(30));
        let weights = MockWeights {
            meta: Metadata { kv },
        };
        let err = DiffusionGemma::from_weights(&weights).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("expected architecture"),
            "error should mention architecture mismatch: {msg}"
        );
    }

    // ── unit: missing block_count returns Err ─────────────────────────────────

    #[test]
    fn from_weights_missing_block_count() {
        let mut kv: HashMap<String, MetaValue> = HashMap::new();
        kv.insert(KEY_ARCH.into(), MetaValue::Str("diffusion-gemma".into()));
        // no block_count
        let weights = MockWeights {
            meta: Metadata { kv },
        };
        let err = DiffusionGemma::from_weights(&weights).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("block_count"),
            "error should mention block_count: {msg}"
        );
    }

    // ── unit: scalar head_count_kv is expanded to per-layer vec ──────────────

    #[test]
    fn from_weights_scalar_kv() {
        let mut kv: HashMap<String, MetaValue> = HashMap::new();
        kv.insert(KEY_ARCH.into(), MetaValue::Str("diffusion-gemma".into()));
        kv.insert(KEY_BLOCK_COUNT.into(), MetaValue::U64(4));
        kv.insert(KEY_HEAD_COUNT_KV.into(), MetaValue::U64(4)); // scalar
        let weights = MockWeights {
            meta: Metadata { kv },
        };
        let model = DiffusionGemma::from_weights(&weights).expect("from_weights failed");
        let cfg = model.config();
        assert_eq!(cfg.n_head_kv, vec![4, 4, 4, 4]);
    }

    // ── gated: open the real GGUF file and parse metadata ────────────────────
    // Run with: cargo test -p infr-model -- --ignored

    #[test]
    #[ignore]
    fn real_model_from_weights() {
        use infr_gguf::Gguf;
        use std::path::Path;

        let path = Path::new(
            "/home/mxaddict/Projects/models/diffusiongemma-26B-A4B-it-GGUF/\
             diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("GGUF file not found at {path:?}, skipping");
            return;
        }

        let gguf = Gguf::open(path).expect("Gguf::open failed");
        let model = DiffusionGemma::from_weights(&gguf).expect("from_weights failed");
        let cfg = model.config();

        assert_eq!(cfg.n_layer, 30, "n_layer");
        assert_eq!(cfg.hidden, 2816, "hidden");
        assert_eq!(cfg.n_head, 16, "n_head");
        assert_eq!(cfg.head_dim_full, 512, "head_dim_full");
        assert_eq!(cfg.head_dim_swa, 256, "head_dim_swa");
        assert_eq!(cfg.swa_window, 1024, "swa_window (from GGUF)");
        assert_eq!(cfg.canvas_len, 256, "canvas_len");
        assert_eq!(cfg.mask_token, 4, "mask_token");
        assert_eq!(
            cfg.full_attn_layers(),
            vec![5, 11, 17, 23, 29],
            "full-attention layer indices"
        );
        assert!(cfg.vocab >= 262_144, "vocab >= 262144, got {}", cfg.vocab);
        assert!(
            (cfg.rope_theta - 1_000_000.0_f32).abs() < 1.0,
            "rope_theta ≈ 1e6, got {}",
            cfg.rope_theta
        );
    }
}
