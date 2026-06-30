//! Model hyper-parameters parsed from GGUF metadata. Mechanically split out of `lib.rs`.
use crate::{meta_u64, MoeConfig};
use anyhow::{bail, Context, Result};
use infr_core::loader::MetaValue;
use infr_core::WeightSource;
use infr_gguf::Gguf;

#[derive(Clone, Debug)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub n_embd: usize,
    /// Dense FFN inner width. For models with a uniform FFN this is the width every layer uses; for
    /// gemma4 E2B (per-layer FFN array) it's the MAX over layers, used to size shared FFN scratch.
    pub n_ff: usize,
    /// Per-layer FFN inner width. gemma4 E2B stores `feed_forward_length` as an array (most 6144, the
    /// late layers 12288); every other model is uniform (all entries equal `n_ff`).
    pub n_ff_layers: Vec<usize>,
    /// gemma4 gemma3n-style per-layer input embeddings: the width of each layer's extra input vector
    /// (`embedding_length_per_layer_input`, 256 for E2B). `0` = the model has no per-layer embeddings.
    pub n_embd_per_layer: usize,
    /// gemma4 E2B KV sharing (gemma3n): only the first `n_layer_kv_from_start` layers compute + cache
    /// their own K/V; later layers reuse an earlier layer's cache (SWA→`from_start-2`, full→`-1`).
    /// Equal to `n_layer` (every layer owns its KV) for models without sharing.
    pub n_layer_kv_from_start: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub vocab: usize,
    pub eos: u32,
    /// All tokens that end generation (the GGUF eos plus `<|im_end|>` / `<|endoftext|>` when present
    /// in the vocab). A chat model can emit any of these; stopping only on `eos` lets it ramble.
    pub eos_ids: Vec<u32>,
    /// Qwen3-style per-head RMSNorm on Q and K before RoPE.
    pub qk_norm: bool,
    /// gemma family: scale input embeddings by √n_embd, sandwich norms (post-attn / post-ffw RMSNorm
    /// before the residual add), and a GeGLU (GELU) FFN instead of SwiGLU.
    pub gemma: bool,
    /// gemma4: adds per-layer heterogeneous head dims (the `*_swa` fields), a weightless RMSNorm on V,
    /// attention scale 1.0 (no 1/√d — QK-norm handles magnitude), a final logit softcap, and
    /// proportional RoPE (freq_factors) on the full-attention layers.
    pub gemma4: bool,
    /// Per-layer dims for the SWA (local) layers when they differ from the full (global) layers
    /// (gemma4). Equal to `head_dim` / `n_kv` / `rope_dim` for uniform-dim models.
    pub head_dim_swa: usize,
    pub n_kv_swa: usize,
    pub rope_dim_swa: usize,
    /// Final logit softcap (gemma2/gemma4): `logits = cap * tanh(logits / cap)`. `0` = no softcap.
    pub final_softcap: f32,
    /// Sliding-window attention size (gemma); `0` = full causal attention everywhere. SWA layers
    /// only attend to the last `swa_window` keys.
    pub swa_window: usize,
    /// SWA layer pattern (gemma): every `swa_pattern`-th layer uses FULL attention, the rest SWA.
    /// `0`/`1` = no pattern. llama.cpp `set_swa_pattern(p)`: layer `il` is full iff `(il+1) % p == 0`.
    pub swa_pattern: usize,
    /// RoPE base for the SWA (local) layers (gemma3 dual-rope): SWA layers use this, full layers use
    /// `rope_theta`. Defaults to 10000 (llama.cpp's `rope_freq_base_train_swa` default) when gemma's
    /// GGUF omits an explicit `rope.freq_base_swa`. Equal to `rope_theta` for non-SWA models.
    pub swa_rope_theta: f32,
    /// MoE config (qwen3moe): `Some` enables the routed-expert FFN. `None` = dense FFN.
    pub moe: Option<MoeConfig>,
}

impl Config {
    /// Whether layer `il` uses sliding-window (vs full) attention. gemma interleaves SWA with full
    /// attention on a fixed period; non-gemma models are always full.
    pub fn is_swa_layer(&self, il: usize) -> bool {
        self.swa_window > 0 && self.swa_pattern > 1 && !(il + 1).is_multiple_of(self.swa_pattern)
    }

    /// RoPE base for layer `il`: gemma3 SWA (local) layers use the smaller `swa_rope_theta`, full
    /// (global) layers use `rope_theta`. Non-gemma models return `rope_theta` for every layer.
    pub fn layer_rope_theta(&self, il: usize) -> f32 {
        if self.is_swa_layer(il) {
            self.swa_rope_theta
        } else {
            self.rope_theta
        }
    }

    /// Head dim for layer `il`. gemma4 SWA layers are narrower than full layers; uniform elsewhere.
    pub fn layer_head_dim(&self, il: usize) -> usize {
        if self.is_swa_layer(il) {
            self.head_dim_swa
        } else {
            self.head_dim
        }
    }

    /// KV-head count for layer `il` (gemma4 SWA vs full GQA grouping; uniform elsewhere).
    pub fn layer_n_kv(&self, il: usize) -> usize {
        if self.is_swa_layer(il) {
            self.n_kv_swa
        } else {
            self.n_kv
        }
    }

    /// FFN inner width for layer `il`. gemma4 E2B's late layers are wider (12288 vs 6144); uniform
    /// (`n_ff`) for every other model.
    pub fn layer_n_ff(&self, il: usize) -> usize {
        self.n_ff_layers.get(il).copied().unwrap_or(self.n_ff)
    }

    /// Whether layer `il` computes + caches its own K/V. gemma4 E2B's later layers (`il >=
    /// n_layer_kv_from_start`) reuse an earlier layer's cache instead. `true` for every layer of a
    /// non-sharing model.
    pub fn has_own_kv(&self, il: usize) -> bool {
        il < self.n_layer_kv_from_start
    }

    /// The cache layer whose K/V layer `il` attends to. For an own-KV layer that's `il` itself; for a
    /// gemma4 E2B shared layer it's `n_layer_kv_from_start - (2 if SWA else 1)` (matching llama.cpp's
    /// gemma3n/gemma4 reuse: SWA shared layers reuse the last own SWA layer, full the last own full).
    pub fn kv_src_layer(&self, il: usize) -> usize {
        if self.has_own_kv(il) {
            il
        } else {
            self.n_layer_kv_from_start - if self.is_swa_layer(il) { 2 } else { 1 }
        }
    }

    /// RoPE rotation dim for layer `il` (gemma4 SWA vs full; uniform elsewhere).
    pub fn layer_rope_dim(&self, il: usize) -> usize {
        if self.is_swa_layer(il) {
            self.rope_dim_swa
        } else {
            self.rope_dim
        }
    }

    /// The largest per-layer head_dim / n_kv across all layers — used to size shared activation and
    /// KV scratch that's reused across layers of differing width (gemma4).
    pub fn max_head_dim(&self) -> usize {
        self.head_dim.max(self.head_dim_swa)
    }
    pub fn max_n_kv(&self) -> usize {
        self.n_kv.max(self.n_kv_swa)
    }

    /// Parse the model config purely from GGUF metadata + tensor shapes — no GPU/Vulkan, no weight
    /// upload. The single source of truth for both the GPU loader ([`Llama::load_opt`]) and the
    /// CPU-only loader ([`CpuModel::load`]). `eos_ids` holds only the GGUF `eos` here; chat-end
    /// markers (`<|im_end|>` …) are appended once a tokenizer exists (see [`add_chat_eos`]).
    pub fn from_gguf(g: &Gguf) -> Result<Config> {
        let arch = g
            .metadata()
            .str("general.architecture")
            .unwrap_or("")
            .to_string();
        let qk_norm = match arch.as_str() {
            "llama" => false,
            "qwen3" | "qwen3moe" | "gemma3" | "gemma4" => true,
            other => bail!(
                "infr-llama supports architecture=llama|qwen3|qwen3moe|gemma3|gemma4, got {other:?}"
            ),
        };
        let gemma4 = arch == "gemma4";
        let gemma = arch == "gemma3" || gemma4;
        let mk = |k: &str| format!("{arch}.{k}");
        let n_layer = meta_u64(g, &mk("block_count")).context("block_count")? as usize;
        let n_embd = meta_u64(g, &mk("embedding_length")).context("embedding_length")? as usize;
        let n_head = meta_u64(g, &mk("attention.head_count")).context("head_count")? as usize;
        let n_kv = meta_u64(g, &mk("attention.head_count_kv")).unwrap_or(n_head as u64) as usize;
        let n_ff_layers: Vec<usize> = if let Some(arr) = g
            .metadata()
            .get(&mk("feed_forward_length"))
            .and_then(MetaValue::as_arr)
        {
            arr.iter()
                .filter_map(MetaValue::as_u64)
                .map(|v| v as usize)
                .collect()
        } else {
            let ff =
                meta_u64(g, &mk("feed_forward_length")).context("feed_forward_length")? as usize;
            vec![ff; n_layer]
        };
        let n_ff = n_ff_layers.iter().copied().max().unwrap_or(0);
        let moe = if arch == "qwen3moe" {
            let n_expert = meta_u64(g, &mk("expert_count")).context("expert_count")? as usize;
            let n_used =
                meta_u64(g, &mk("expert_used_count")).context("expert_used_count")? as usize;
            let n_ff_exp = meta_u64(g, &mk("expert_feed_forward_length"))
                .map(|v| v as usize)
                .unwrap_or(n_ff / n_used.max(1));
            Some(MoeConfig {
                n_expert,
                n_used,
                n_ff_exp,
                scale: 1.0,
            })
        } else {
            None
        };
        let head_dim =
            meta_u64(g, &mk("attention.key_length")).unwrap_or((n_embd / n_head) as u64) as usize;
        let rope_dim = meta_u64(g, &mk("rope.dimension_count")).unwrap_or(head_dim as u64) as usize;
        let rope_theta = g
            .metadata()
            .get(&mk("rope.freq_base"))
            .and_then(|v| match v {
                MetaValue::F64(f) => Some(*f as f32),
                MetaValue::U64(u) => Some(*u as f32),
                _ => None,
            })
            .unwrap_or(10000.0);
        let rms_eps = g
            .metadata()
            .get(&mk("attention.layer_norm_rms_epsilon"))
            .and_then(|v| match v {
                MetaValue::F64(f) => Some(*f as f32),
                _ => None,
            })
            .unwrap_or(1e-5);
        let swa_window = if gemma {
            meta_u64(g, &mk("attention.sliding_window")).unwrap_or(0) as usize
        } else {
            0
        };
        let swa_pattern = if swa_window == 0 {
            0
        } else if let Some(arr) = g
            .metadata()
            .get(&mk("attention.sliding_window_pattern"))
            .and_then(MetaValue::as_arr)
        {
            arr.iter()
                .position(|v| matches!(v, MetaValue::Bool(false)))
                .map(|i| i + 1)
                .unwrap_or(6)
        } else {
            meta_u64(g, &mk("attention.sliding_window_pattern")).unwrap_or(6) as usize
        };
        let swa_rope_theta = if swa_window > 0 {
            g.metadata()
                .get(&mk("rope.freq_base_swa"))
                .and_then(|v| match v {
                    MetaValue::F64(f) => Some(*f as f32),
                    MetaValue::U64(u) => Some(*u as f32),
                    _ => None,
                })
                .unwrap_or(10000.0)
        } else {
            rope_theta
        };
        let (head_dim, n_kv, rope_dim, head_dim_swa, n_kv_swa, rope_dim_swa) = if gemma4 {
            let hk = g
                .metadata()
                .get(&mk("attention.head_count_kv"))
                .and_then(MetaValue::as_arr);
            let kv_at = |i: usize| {
                hk.and_then(|a| a.get(i))
                    .and_then(MetaValue::as_u64)
                    .map(|v| v as usize)
            };
            let full_idx = swa_pattern.saturating_sub(1);
            let hd_full =
                meta_u64(g, &mk("attention.key_length")).unwrap_or(head_dim as u64) as usize;
            let hd_swa =
                meta_u64(g, &mk("attention.key_length_swa")).unwrap_or(hd_full as u64) as usize;
            let rd_full =
                meta_u64(g, &mk("rope.dimension_count")).unwrap_or(hd_full as u64) as usize;
            let rd_swa =
                meta_u64(g, &mk("rope.dimension_count_swa")).unwrap_or(hd_swa as u64) as usize;
            (
                hd_full,
                kv_at(full_idx).unwrap_or(n_kv),
                rd_full,
                hd_swa,
                kv_at(0).unwrap_or(n_kv),
                rd_swa,
            )
        } else {
            (head_dim, n_kv, rope_dim, head_dim, n_kv, rope_dim)
        };
        let final_softcap = if gemma4 {
            g.metadata()
                .get(&mk("final_logit_softcapping"))
                .and_then(MetaValue::as_f64)
                .unwrap_or(0.0) as f32
        } else {
            0.0
        };
        let n_embd_per_layer = if gemma4 {
            meta_u64(g, &mk("embedding_length_per_layer_input")).unwrap_or(0) as usize
        } else {
            0
        };
        let n_layer_kv_from_start = if gemma4 {
            let shared = meta_u64(g, &mk("attention.shared_kv_layers")).unwrap_or(0) as usize;
            n_layer.saturating_sub(shared)
        } else {
            n_layer
        };
        let eos = meta_u64(g, "tokenizer.ggml.eos_token_id").unwrap_or(2) as u32;
        // vocab = token_embd rows (GGUF shape `[n_embd, vocab]`) — read from the tensor header, no load.
        let vocab = g
            .tensors()
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.shape.last().copied())
            .context("token_embd.weight shape")?;
        Ok(Config {
            n_layer,
            n_head,
            n_kv,
            n_embd,
            n_ff,
            n_ff_layers,
            n_embd_per_layer,
            n_layer_kv_from_start,
            head_dim,
            rope_dim,
            rope_theta,
            rms_eps,
            vocab,
            eos,
            eos_ids: vec![eos],
            qk_norm,
            gemma,
            gemma4,
            head_dim_swa,
            n_kv_swa,
            rope_dim_swa,
            final_softcap,
            swa_window,
            swa_pattern,
            swa_rope_theta,
            moe,
        })
    }
}
