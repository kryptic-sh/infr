//! Qwen4-Exp text configuration. The HF identifier is not a canonical GGUF architecture.
//!
//! Loading is supported as a foundation for inference; HC, PLE, QSA and the sigmoid GDN
//! output gate must be emitted together before the shared runner can execute this architecture.

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

/// Layer kinds accepted by both the tiny checkpoint and newer Transformers configs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    LinearAttention,
    IndexedAttention,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputGate {
    Sigmoid,
    Silu,
}

#[derive(Clone, Debug)]
pub struct IndexerConfig {
    pub n_heads: usize,
    pub head_dim: usize,
    pub budget: usize,
    pub compress_ratio: usize,
}

#[derive(Clone, Debug)]
pub struct PleConfig {
    /// One-indexed layer IDs, as in HF. Sorted and deduplicated.
    pub layer_ids: Vec<usize>,
    pub embed_dim: usize,
    pub conv_kernel_size: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub vocab_size_base: usize,
    pub vocab_divisor: usize,
    pub seed: u64,
    pub split_parts: usize,
}

// Bound public parsing too, not just file loading, before allocating per-layer geometry.
// Far above current checkpoints (48 layers), but rejects hostile allocation sizes.
const MAX_LAYERS: usize = 4096;

#[derive(Clone, Debug)]
pub struct Cfg {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_dim: usize,
    pub rope_sections: [u32; 4],
    pub mrope_interleaved: bool,
    pub mamba_ssm_dtype: infr_core::DType,
    pub number_of_conv_states: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub layer_types: Vec<LayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub output_gate: OutputGate,
    pub indexer: Option<IndexerConfig>,
    pub ple: Option<PleConfig>,
    pub eos_ids: Vec<u32>,
    pub bos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub tie_word_embeddings: bool,
}

fn uint(v: &Value, key: &str, default: Option<usize>) -> Result<usize> {
    let value = match v.get(key) {
        None | Some(Value::Null) => default,
        Some(n) => n.as_u64().and_then(|n| usize::try_from(n).ok()),
    };
    value
        .filter(|&n| n > 0)
        .with_context(|| format!("qwen4_exp: {key} must be a positive usize"))
}

fn positive_float(v: &Value, key: &str, default: f32) -> Result<f32> {
    let n = match v.get(key) {
        None | Some(Value::Null) => default,
        Some(n) => n.as_f64().context(format!("qwen4_exp: invalid {key}"))? as f32,
    };
    ensure!(
        n.is_finite() && n > 0.0,
        "qwen4_exp: {key} must be positive and finite"
    );
    Ok(n)
}

fn boolean(v: &Value, key: &str, default: bool) -> Result<bool> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(b) => b
            .as_bool()
            .context(format!("qwen4_exp: {key} must be boolean")),
    }
}

fn string<'a>(v: &'a Value, key: &str, default: &'a str) -> Result<&'a str> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(s) => s
            .as_str()
            .with_context(|| format!("qwen4_exp: {key} must be a string")),
    }
}

pub(crate) fn ensure_forward_supported(c: &crate::Config) -> Result<()> {
    ensure!(c.qwen4_exp.is_none(), "{FORWARD_UNSUPPORTED}");
    Ok(())
}

fn token_id(v: &Value, vocab: usize) -> Result<u32> {
    let id = v
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .context("qwen4_exp: token ID must fit u32")?;
    ensure!(
        (id as usize) < vocab,
        "qwen4_exp: token ID {id} outside vocabulary {vocab}"
    );
    Ok(id)
}

impl Cfg {
    /// Parse either `Qwen4ExpConfig` or standalone `Qwen4ExpTextConfig`.
    pub fn from_hf(root: &Value) -> Result<Self> {
        ensure!(root.is_object(), "qwen4_exp: config must be an object");
        let text = match root["model_type"].as_str() {
            Some("qwen4_exp") => root
                .get("text_config")
                .filter(|v| v.is_object())
                .context("qwen4_exp: missing text_config object")?,
            Some("qwen4_exp_text") => root,
            other => bail!("qwen4_exp: unsupported model_type {other:?}"),
        };
        ensure!(
            text["model_type"].is_null() || text["model_type"] == "qwen4_exp_text",
            "qwen4_exp: unsupported text_config.model_type"
        );
        ensure!(
            root["quantization_config"].is_null() && text["quantization_config"].is_null(),
            "qwen4_exp: only dense HF safetensors are supported, not quantized checkpoints"
        );
        ensure!(
            string(text, "hidden_act", "silu")? == "silu",
            "qwen4_exp: MoE hidden_act must be silu"
        );
        ensure!(
            !boolean(text, "attention_bias", false)?,
            "qwen4_exp: attention_bias is unsupported"
        );
        let hidden = uint(text, "hidden_size", None)?;
        let layers = uint(text, "num_hidden_layers", None)?;
        ensure!(
            layers <= MAX_LAYERS,
            "qwen4_exp: num_hidden_layers exceeds loader limit {MAX_LAYERS}"
        );
        let vocab = uint(text, "vocab_size", None)?;
        ensure!(
            vocab <= u32::MAX as usize,
            "qwen4_exp: vocabulary exceeds u32 token IDs"
        );
        let heads = uint(text, "num_attention_heads", None)?;
        let kv_heads = uint(text, "num_key_value_heads", None)?;
        ensure!(
            heads.is_multiple_of(kv_heads),
            "qwen4_exp: invalid attention head geometry"
        );
        let head_dim = uint(text, "head_dim", Some(hidden / heads))?;
        let hc_count = uint(text, "hc_count", Some(4))?;
        ensure!(hc_count > 1, "qwen4_exp: hc_count must be > 1");
        hidden
            .checked_mul(hc_count)
            .context("qwen4_exp: HC width overflow")?;
        // Check every projected width before constructing tensor shapes or allocating state.
        heads
            .checked_mul(head_dim)
            .and_then(|n| n.checked_mul(2))
            .context("qwen4_exp: query/gate width overflow")?;
        kv_heads
            .checked_mul(head_dim)
            .context("qwen4_exp: KV width overflow")?;
        let rope = text
            .get("rope_parameters")
            .filter(|v| !v.is_null())
            .unwrap_or(text);
        ensure!(
            rope.is_object(),
            "qwen4_exp: rope_parameters must be an object"
        );
        ensure!(
            rope["rope_type"].is_null() || rope["rope_type"] == "default",
            "qwen4_exp: only default RoPE is supported"
        );
        ensure!(
            text["rope_scaling"].is_null(),
            "qwen4_exp: rope_scaling is unsupported"
        );
        let factor = positive_float(rope, "partial_rotary_factor", 1.0)?;
        ensure!(
            factor <= 1.0,
            "qwen4_exp: partial_rotary_factor must be <= 1"
        );
        let rope_dim = (head_dim as f64 * factor as f64) as usize;
        ensure!(
            rope_dim > 0 && rope_dim.is_multiple_of(2),
            "qwen4_exp: rotary dimension must be positive and even"
        );
        let mut rope_sections = [0; 4];
        if let Some(sections) = rope.get("mrope_section") {
            let sections = sections
                .as_array()
                .context("qwen4_exp: mrope_section must be an array")?;
            ensure!(
                sections.len() <= 4,
                "qwen4_exp: at most four RoPE sections are supported"
            );
            for (dst, src) in rope_sections.iter_mut().zip(sections) {
                *dst = src
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .context("qwen4_exp: invalid RoPE section")?;
            }
            ensure!(
                rope_sections.iter().map(|&n| n as u64).sum::<u64>() == (rope_dim / 2) as u64,
                "qwen4_exp: mrope_section must sum to half the rotary dimension"
            );
        }
        let layer_types = if let Some(types) = text.get("layer_types").filter(|v| !v.is_null()) {
            let types = types
                .as_array()
                .context("qwen4_exp: layer_types must be an array")?;
            ensure!(
                types.len() == layers,
                "qwen4_exp: layer_types length must equal num_hidden_layers"
            );
            types
                .iter()
                .map(|v| match v.as_str() {
                    Some("linear_attention") => Ok(LayerType::LinearAttention),
                    Some("qwen_sparse_attention" | "indexed_attention" | "full_attention") => {
                        Ok(LayerType::IndexedAttention)
                    }
                    other => bail!("qwen4_exp: unsupported layer type {other:?}"),
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            let interval = uint(text, "full_attention_interval", Some(4))?;
            // Bound the layer list by the checkpoint in the adapter before calling this parser.
            (0..layers)
                .map(|i| {
                    if (i + 1).is_multiple_of(interval) {
                        LayerType::IndexedAttention
                    } else {
                        LayerType::LinearAttention
                    }
                })
                .collect()
        };
        let index_keys = [
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        ];
        let needs_indexer = layer_types.contains(&LayerType::IndexedAttention)
            || index_keys.iter().any(|key| !text[*key].is_null());
        let indexer = if needs_indexer {
            let n_heads = uint(text, index_keys[0], None)?;
            ensure!(
                uint(text, index_keys[1], None)? == 1,
                "qwen4_exp: QSA requires indexer_kv_heads=1"
            );
            let dim = uint(text, index_keys[2], None)?;
            let budget = uint(text, index_keys[3], None)?;
            let ratio = uint(text, index_keys[4], None)?;
            ensure!(
                budget.is_multiple_of(ratio),
                "qwen4_exp: indexer_budget must be divisible by indexer_compress_ratio"
            );
            ensure!(
                rope_dim <= dim,
                "qwen4_exp: rotary dimension exceeds indexer_head_dim"
            );
            n_heads
                .checked_add(1)
                .and_then(|n| n.checked_mul(dim))
                .context("qwen4_exp: indexer projection width overflow")?;
            budget
                .checked_add(ratio - 1)
                .context("qwen4_exp: indexer selection width overflow")?;
            Some(IndexerConfig {
                n_heads,
                head_dim: dim,
                budget,
                compress_ratio: ratio,
            })
        } else {
            None
        };
        let experts = uint(text, "num_experts", None)?;
        let topk = uint(text, "num_experts_per_tok", None)?;
        ensure!(
            topk <= experts,
            "qwen4_exp: num_experts_per_tok exceeds num_experts"
        );
        let eos = &text["eos_token_id"];
        let eos_ids = match eos.as_array() {
            Some(ids) => ids
                .iter()
                .map(|id| token_id(id, vocab))
                .collect::<Result<Vec<_>>>()?,
            None => vec![token_id(eos, vocab)?],
        };
        ensure!(
            !eos_ids.is_empty(),
            "qwen4_exp: eos_token_id must not be empty"
        );
        let ple_ids = match text.get("ple_layer_ids").filter(|v| !v.is_null()) {
            Some(ids) => ids
                .as_array()
                .context("qwen4_exp: ple_layer_ids must be an array")?
                .iter()
                .map(|n| {
                    n.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .context("qwen4_exp: invalid PLE layer ID")
                })
                .collect::<Result<Vec<_>>>()?,
            None => Vec::new(),
        };
        let ple = if ple_ids.is_empty() {
            None
        } else {
            let mut ids = ple_ids;
            ids.sort_unstable();
            ids.dedup();
            for &id in &ids {
                ensure!(
                    id > 0 && id <= layers,
                    "qwen4_exp: PLE layer IDs must be one-indexed and within the trunk"
                );
                ensure!(
                    layer_types[id - 1] == LayerType::LinearAttention,
                    "qwen4_exp: PLE requires a linear_attention layer"
                );
            }
            let ngram = uint(text, "ngram_size", Some(3))?;
            let heads = uint(text, "heads_per_ngram", Some(8))?;
            let ngram_heads = ngram
                .checked_sub(1)
                .and_then(|n| n.checked_mul(heads))
                .filter(|&n| n > 0)
                .context("qwen4_exp: invalid n-gram head count")?;
            let dim = uint(text, "ple_embed_dim", Some(hidden))?;
            ensure!(
                dim.is_multiple_of(ngram_heads),
                "qwen4_exp: ple_embed_dim must be divisible by n-gram head count"
            );
            Some(PleConfig {
                layer_ids: ids,
                embed_dim: dim,
                conv_kernel_size: uint(text, "ple_conv_kernel_size", Some(4))?,
                ngram_size: ngram,
                heads_per_ngram: heads,
                vocab_size_base: uint(text, "ngram_vocab_size_base", Some(20_000_000))?,
                vocab_divisor: uint(text, "make_ngram_vocab_size_divisible_by", Some(128))?,
                seed: text
                    .get("seed")
                    .map(|v| v.as_u64().context("qwen4_exp: seed must be u64"))
                    .transpose()?
                    .unwrap_or(1234),
                split_parts: uint(text, "split_ngram_parts", Some(512))?,
            })
        };
        let number_of_conv_states = uint(
            text,
            "number_of_conv_states",
            Some(if ple.is_some() { 3 } else { 1 }),
        )?;
        ensure!(
            number_of_conv_states == if ple.is_some() { 3 } else { 1 },
            "qwen4_exp: number_of_conv_states must be 3 with PLE, otherwise 1"
        );
        let mamba_ssm_dtype = match string(text, "mamba_ssm_dtype", "float32")? {
            "float32" => infr_core::DType::F32,
            "float16" => infr_core::DType::F16,
            "bfloat16" => infr_core::DType::Bf16,
            other => bail!("qwen4_exp: unsupported mamba_ssm_dtype {other:?}"),
        };
        let output_gate = match string(text, "output_gate_type", "silu")? {
            "sigmoid" => OutputGate::Sigmoid,
            "silu" => OutputGate::Silu,
            other => bail!("qwen4_exp: unsupported output gate {other:?}"),
        };
        let k_heads = uint(text, "linear_num_key_heads", Some(16))?;
        let v_heads = uint(text, "linear_num_value_heads", Some(32))?;
        ensure!(
            v_heads.is_multiple_of(k_heads),
            "qwen4_exp: GDN value heads must be divisible by key heads"
        );
        let k_dim = uint(text, "linear_key_head_dim", Some(128))?;
        let v_dim = uint(text, "linear_value_head_dim", Some(128))?;
        let k_width = k_heads
            .checked_mul(k_dim)
            .context("qwen4_exp: GDN key width overflow")?;
        let v_width = v_heads
            .checked_mul(v_dim)
            .context("qwen4_exp: GDN value width overflow")?;
        k_width
            .checked_mul(2)
            .and_then(|n| n.checked_add(v_width))
            .context("qwen4_exp: GDN QKV width overflow")?;
        Ok(Self {
            hidden_size: hidden,
            num_hidden_layers: layers,
            vocab_size: vocab,
            num_attention_heads: heads,
            num_key_value_heads: kv_heads,
            head_dim,
            max_position_embeddings: uint(text, "max_position_embeddings", None)?,
            rms_norm_eps: positive_float(text, "rms_norm_eps", 1e-6)?,
            rope_theta: positive_float(rope, "rope_theta", 10_000.0)?,
            rope_dim,
            rope_sections,
            mrope_interleaved: boolean(rope, "mrope_interleaved", false)?,
            mamba_ssm_dtype,
            number_of_conv_states,
            hc_count,
            hc_lowrank: uint(text, "hc_lowrank", Some(320))?,
            layer_types,
            linear_conv_kernel_dim: uint(text, "linear_conv_kernel_dim", Some(4))?,
            linear_key_head_dim: k_dim,
            linear_value_head_dim: v_dim,
            linear_num_key_heads: k_heads,
            linear_num_value_heads: v_heads,
            num_experts: experts,
            num_experts_per_tok: topk,
            moe_intermediate_size: uint(text, "moe_intermediate_size", None)?,
            shared_expert_intermediate_size: uint(text, "shared_expert_intermediate_size", None)?,
            norm_topk_prob: boolean(text, "norm_topk_prob", true)?,
            output_gate,
            indexer,
            ple,
            eos_ids,
            bos_token_id: text
                .get("bos_token_id")
                .filter(|v| !v.is_null())
                .map(|v| token_id(v, vocab))
                .transpose()?,
            pad_token_id: text
                .get("pad_token_id")
                .filter(|v| !v.is_null())
                .map(|v| token_id(v, vocab))
                .transpose()?,
            tie_word_embeddings: boolean(
                text,
                "tie_word_embeddings",
                boolean(root, "tie_word_embeddings", false)?,
            )?,
        })
    }

    pub(crate) fn shared_config(self) -> crate::Config {
        crate::Config {
            n_layer: self.num_hidden_layers,
            n_head: self.num_attention_heads,
            n_kv: self.num_key_value_heads,
            n_embd: self.hidden_size,
            n_ff: self
                .moe_intermediate_size
                .max(self.shared_expert_intermediate_size),
            n_ff_layers: vec![
                self.moe_intermediate_size
                    .max(self.shared_expert_intermediate_size);
                self.num_hidden_layers
            ],
            n_layer_kv_from_start: self.num_hidden_layers,
            head_dim: self.head_dim,
            rope_dim: self.rope_dim,
            rope_theta: self.rope_theta,
            rms_eps: self.rms_norm_eps,
            vocab: self.vocab_size,
            eos: self.eos_ids[0],
            eos_ids: self.eos_ids.clone(),
            qk_norm: true,
            head_dim_swa: self.head_dim,
            n_kv_swa: self.num_key_value_heads,
            rope_dim_swa: self.rope_dim,
            swa_rope_theta: self.rope_theta,
            n_ctx_train: self.max_position_embeddings,
            qwen35: true,
            attn_out_gate: true,
            ssm_d_conv: self.linear_conv_kernel_dim,
            ssm_d_state: self.linear_key_head_dim,
            ssm_d_inner: self.linear_num_value_heads * self.linear_value_head_dim,
            ssm_n_group: self.linear_num_key_heads,
            ssm_dt_rank: self.linear_num_value_heads,
            rope_sections: self.rope_sections,
            shexp_ff: self.shared_expert_intermediate_size,
            shexp_gated: true,
            moe: Some(crate::MoeConfig {
                n_expert: self.num_experts,
                n_used: self.num_experts_per_tok,
                n_ff_exp: self.moe_intermediate_size,
                scale: 1.0,
                gating: infr_core::graph::MoeGating::Softmax,
                norm_w: self.norm_topk_prob,
                weight_before: false,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
            }),
            qwen4_exp: Some(self),
            ..Default::default()
        }
    }
}

pub(crate) const FORWARD_UNSUPPORTED: &str = "arch=qwen4_exp: checkpoint loading is supported, but inference is not yet implemented (requires Qwen4-Exp HC, PLE, QSA and GDN output-gate semantics)";
