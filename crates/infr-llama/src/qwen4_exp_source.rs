//! HF Qwen4-Exp text adapter over the EXISTING mmap safetensors reader.
//!
//! No quantizer, file-format parser, or pretend GGUF converter here. Dense tensors are
//! borrowed unchanged, with GGUF-style fastest-dimension-first shapes. HC and PLE keep
//! explicit names (there are no ordinary residual/final RMSNorm weights in this model).
//! Individual and packed expert layouts are retained, not eagerly stacked/copied.

#[cfg(test)]
#[path = "qwen4_exp_source_tests.rs"]
mod tests;

use crate::qwen4_exp::{Cfg, LayerType};
use anyhow::{ensure, Context, Result};
use infr_core::loader::{MetaValue, Metadata, TensorInfo, WeightSource};
use infr_core::{DType, Error};
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub(crate) struct Qwen4ExpSource {
    st: infr_safetensors::Safetensors,
    pub(crate) cfg: Cfg,
    metadata: Metadata,
    infos: Vec<TensorInfo>,
    names: HashMap<String, String>,
    source_index: HashMap<String, usize>,
    consumed: HashSet<String>,
    chat_template: Option<String>,
    // Host-only PLE hash buffers, decoded without narrowing or binding as graph weights.
    _ple_buffers: HashMap<String, Vec<i64>>,
}

fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .context("qwen4_exp: tensor dimension overflow")
}

// Deterministic u64 Miller–Rabin avoids sqrt(value) work for hostile vocabulary bases.
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        let mut base = a % n;
        if base == 0 {
            continue;
        }
        let mut exp = d;
        let mut x = 1u64;
        while exp > 0 {
            if exp & 1 != 0 {
                x = ((x as u128 * base as u128) % n as u128) as u64;
            }
            base = ((base as u128 * base as u128) % n as u128) as u64;
            exp >>= 1;
        }
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut witness = true;
        for _ in 1..s {
            x = ((x as u128 * x as u128) % n as u128) as u64;
            if x == n - 1 {
                witness = false;
                break;
            }
        }
        if witness {
            return false;
        }
    }
    true
}

fn validate_ple_hashes(
    cfg: &Cfg,
    ple: &crate::qwen4_exp::PleConfig,
    ordinal: usize,
    multipliers: &[i64],
    sizes: &[i64],
) -> Result<()> {
    const GAMMA: u64 = 0x9E3779B97F4A7C15;
    let half_bound = ((i64::MAX as u64 / cfg.vocab_size as u64) / 2).max(1);
    let seed = ple.seed.wrapping_add(10007 * ordinal as u64);
    for (i, &actual) in multipliers.iter().enumerate() {
        let mut value = seed
            .wrapping_add(GAMMA.wrapping_mul(i as u64 + 1))
            .wrapping_add(GAMMA);
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^= value >> 31;
        let expected = (2 * (value % half_bound) + 1) as i64;
        ensure!(
            actual == expected,
            "qwen4_exp: PLE multipliers do not match seed/vocabulary/ordinal"
        );
    }
    let count = mul(ordinal + 1, sizes.len())?;
    let start = count - sizes.len();
    let mut prime = ple.vocab_size_base as u64 - 1;
    let mut attempts = 0usize;
    // Resource bound independent of malicious numeric values. Real/tiny configurations
    // need only hundreds of candidates, not an unbounded prime search.
    for i in 0..count {
        loop {
            attempts += 1;
            ensure!(
                attempts <= 65536,
                "qwen4_exp: PLE prime search exceeds loader limit"
            );
            prime = prime
                .checked_add(1)
                .filter(|&n| n <= i64::MAX as u64)
                .context("qwen4_exp: PLE prime vocabulary exceeds I64")?;
            if is_prime(prime) {
                break;
            }
        }
        if i >= start {
            ensure!(
                sizes[i - start] == prime as i64,
                "qwen4_exp: PLE head sizes do not match vocabulary base/ordinal"
            );
        }
    }
    Ok(())
}

impl Qwen4ExpSource {
    pub(crate) fn ple_buffer(&self, layer: usize, suffix: &str) -> Result<&[i64]> {
        let key = format!("blk.{layer}.ple_key.weight");
        let prefix = self
            .names
            .get(&key)
            .and_then(|name| name.strip_suffix(".key_proj.weight"))
            .context("qwen4_exp CPU: missing PLE prefix")?;
        self._ple_buffers
            .get(&format!("{prefix}.ple_embedding.{suffix}"))
            .map(Vec::as_slice)
            .context("qwen4_exp CPU: missing PLE buffer")
    }

    pub(crate) fn open(path: &Path) -> Result<Self> {
        let dir = path.parent().unwrap_or(Path::new("."));
        let root: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json"))
                .context("qwen4_exp: read adjacent config.json")?,
        )
        .context("qwen4_exp: parse config.json")?;
        Self::open_config(path, &root)
    }

    fn open_config(path: &Path, root: &serde_json::Value) -> Result<Self> {
        let st = infr_safetensors::Safetensors::open_with_i64_buffers(path)?;
        let text = if root["model_type"] == "qwen4_exp" {
            &root["text_config"]
        } else {
            root
        };
        // Before Cfg creates a default layer list, bound it by the actual file inventory.
        // Missing shards fail here or at the required-tensor check, not by allocating a huge list.
        let layers = text["num_hidden_layers"]
            .as_u64()
            .context("qwen4_exp: missing num_hidden_layers")?;
        ensure!(layers <= st.tensors().len() as u64, "qwen4_exp: layer count exceeds tensor count (incomplete/sharded checkpoints are unsupported)");
        let cfg = Cfg::from_hf(root)?;
        let source_index: HashMap<_, _> = st
            .tensors()
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        let prefix = ["model.language_model.", "model.", ""]
            .into_iter()
            .find(|p| source_index.contains_key(&format!("{p}embed_tokens.weight")))
            .context("qwen4_exp: missing text embed_tokens.weight in this file")?;
        let mut metadata = Metadata { kv: HashMap::new() };
        metadata.kv.insert(
            "general.architecture".into(),
            MetaValue::Str(crate::arch::QWEN4_EXP.into()),
        );
        metadata.kv.insert(
            "tokenizer.ggml.eos_token_id".into(),
            MetaValue::U64(cfg.eos_ids[0] as u64),
        );
        metadata.kv.insert(
            "tokenizer.ggml.add_bos_token".into(),
            MetaValue::Bool(false),
        );
        if let Some(bos) = cfg.bos_token_id {
            metadata.kv.insert(
                "tokenizer.ggml.bos_token_id".into(),
                MetaValue::U64(bos as u64),
            );
        }
        let tc_path = path
            .parent()
            .unwrap_or(Path::new("."))
            .join("tokenizer_config.json");
        let chat_template = if tc_path.exists() {
            let tc: serde_json::Value = serde_json::from_slice(&std::fs::read(tc_path)?)
                .context("qwen4_exp: parse tokenizer_config.json")?;
            tc["chat_template"].as_str().map(str::to_owned)
        } else {
            None
        };
        let mut s = Self {
            st,
            cfg: cfg.clone(),
            metadata,
            infos: Vec::new(),
            names: HashMap::new(),
            source_index,
            consumed: HashSet::new(),
            chat_template,
            _ple_buffers: HashMap::new(),
        };
        let (h, c, r) = (cfg.hidden_size, cfg.hc_count, cfg.hc_lowrank);
        let ch = mul(c, h)?;
        s.add(
            "token_embd.weight",
            &format!("{prefix}embed_tokens.weight"),
            &[cfg.vocab_size, h],
        )?;
        if s.has("lm_head.weight") {
            s.add("output.weight", "lm_head.weight", &[cfg.vocab_size, h])?;
        } else {
            ensure!(
                cfg.tie_word_embeddings,
                "qwen4_exp: missing lm_head.weight with untied embeddings"
            );
        }
        s.hc(
            "output_hc",
            &format!("{prefix}hyper_connection_mixer"),
            ch,
            r,
            None,
        )?;
        let qwidth = mul(cfg.num_attention_heads, cfg.head_dim)?;
        let kvwidth = mul(cfg.num_key_value_heads, cfg.head_dim)?;
        let kwidth = mul(cfg.linear_num_key_heads, cfg.linear_key_head_dim)?;
        let vwidth = mul(cfg.linear_num_value_heads, cfg.linear_value_head_dim)?;
        let convwidth = mul(2, kwidth)?
            .checked_add(vwidth)
            .context("qwen4_exp: conv width overflow")?;
        for (l, kind) in cfg.layer_types.iter().enumerate() {
            let hf = format!("{prefix}layers.{l}");
            let b = format!("blk.{l}");
            s.hc(
                &format!("{b}.attn_hc"),
                &format!("{hf}.attn_hyper_connection"),
                ch,
                r,
                Some(c),
            )?;
            s.hc(
                &format!("{b}.ffn_hc"),
                &format!("{hf}.mlp_hyper_connection"),
                ch,
                r,
                Some(c),
            )?;
            match kind {
                LayerType::LinearAttention => {
                    let a = format!("{hf}.linear_attn");
                    for (name, suffix, shape) in [
                        ("ssm_a", "A_log", vec![cfg.linear_num_value_heads]),
                        ("ssm_dt.bias", "dt_bias", vec![cfg.linear_num_value_heads]),
                        (
                            "ssm_conv1d.weight",
                            "conv1d.weight",
                            vec![convwidth, 1, cfg.linear_conv_kernel_dim],
                        ),
                        ("attn_qkv.weight", "in_proj_qkv.weight", vec![convwidth, h]),
                        ("attn_gate.weight", "in_proj_z.weight", vec![vwidth, h]),
                        (
                            "ssm_alpha.weight",
                            "in_proj_a.weight",
                            vec![cfg.linear_num_value_heads, h],
                        ),
                        (
                            "ssm_beta.weight",
                            "in_proj_b.weight",
                            vec![cfg.linear_num_value_heads, h],
                        ),
                        (
                            "ssm_norm.weight",
                            "norm.weight",
                            vec![cfg.linear_value_head_dim],
                        ),
                        ("ssm_out.weight", "out_proj.weight", vec![h, vwidth]),
                    ] {
                        s.add(&format!("{b}.{name}"), &format!("{a}.{suffix}"), &shape)?;
                    }
                }
                LayerType::IndexedAttention => {
                    let a = format!("{hf}.self_attn");
                    for (name, suffix, shape) in [
                        ("attn_q.weight", "q_proj.weight", vec![mul(2, qwidth)?, h]),
                        ("attn_k.weight", "k_proj.weight", vec![kvwidth, h]),
                        ("attn_v.weight", "v_proj.weight", vec![kvwidth, h]),
                        ("attn_output.weight", "o_proj.weight", vec![h, qwidth]),
                        ("attn_q_norm.weight", "q_norm.weight", vec![cfg.head_dim]),
                        ("attn_k_norm.weight", "k_norm.weight", vec![cfg.head_dim]),
                    ] {
                        s.add(&format!("{b}.{name}"), &format!("{a}.{suffix}"), &shape)?;
                    }
                    let idx = cfg
                        .indexer
                        .as_ref()
                        .context("qwen4_exp: missing indexer config")?;
                    let heads = idx
                        .n_heads
                        .checked_add(1)
                        .context("qwen4_exp: indexer head count overflow")?;
                    s.add(
                        &format!("{b}.indexer_qk.weight"),
                        &format!("{a}.indexer.index_qk_proj.weight"),
                        &[mul(heads, idx.head_dim)?, h],
                    )?;
                    s.add(
                        &format!("{b}.indexer_q_norm.weight"),
                        &format!("{a}.indexer.q_layernorm.weight"),
                        &[idx.head_dim],
                    )?;
                    s.add(
                        &format!("{b}.indexer_k_norm.weight"),
                        &format!("{a}.indexer.k_layernorm.weight"),
                        &[idx.head_dim],
                    )?;
                }
            }
            let ff = cfg.moe_intermediate_size;
            let n = cfg.num_experts;
            s.add(
                &format!("{b}.ffn_gate_inp.weight"),
                &format!("{hf}.mlp.gate.weight"),
                &[n, h],
            )?;
            if s.has(&format!("{hf}.mlp.experts.gate_up_proj")) {
                s.add(
                    &format!("{b}.ffn_gate_up_exps.weight"),
                    &format!("{hf}.mlp.experts.gate_up_proj"),
                    &[n, mul(2, ff)?, h],
                )?;
                s.add(
                    &format!("{b}.ffn_down_exps.weight"),
                    &format!("{hf}.mlp.experts.down_proj"),
                    &[n, h, ff],
                )?;
            } else {
                ensure!(
                    n <= s.st.tensors().len() / 3,
                    "qwen4_exp: expert count exceeds tensor inventory"
                );
                for e in 0..n {
                    s.ffn(
                        &format!("{b}.expert.{e}"),
                        &format!("{hf}.mlp.experts.{e}"),
                        h,
                        ff,
                    )?;
                }
            }
            s.ffn(
                &format!("{b}.shexp"),
                &format!("{hf}.mlp.shared_expert"),
                h,
                cfg.shared_expert_intermediate_size,
            )?;
            s.add(
                &format!("{b}.ffn_gate_inp_shexp.weight"),
                &format!("{hf}.mlp.shared_expert_gate.weight"),
                &[1, h],
            )?;
            if let Some(ple) = &cfg.ple {
                if let Some(ordinal) = ple.layer_ids.iter().position(|&id| id == l + 1) {
                    let p = format!("{hf}.ple");
                    for (name, suffix, shape) in [
                        ("ple_key.weight", "key_proj.weight", vec![ch, ple.embed_dim]),
                        (
                            "ple_value.weight",
                            "value_proj.weight",
                            vec![h, ple.embed_dim],
                        ),
                        ("ple_key_norm.weight", "norm_key.weight", vec![ch]),
                        ("ple_query_norm.weight", "norm_query.weight", vec![ch]),
                        ("ple_conv_norm.weight", "norm_conv.weight", vec![ch]),
                        (
                            "ple_conv.weight",
                            "conv1d.weight",
                            vec![ch, 1, ple.conv_kernel_size],
                        ),
                    ] {
                        s.add(&format!("{b}.{name}"), &format!("{p}.{suffix}"), &shape)?;
                    }
                    let p = format!("{hf}.ple.ple_embedding");
                    let heads = mul(ple.ngram_size - 1, ple.heads_per_ngram)?;
                    let multipliers =
                        s.i64(&format!("{p}.layer_multipliers"), &[ple.ngram_size])?;
                    ensure!(
                        multipliers.iter().all(|&n| n > 0 && n % 2 == 1),
                        "qwen4_exp: PLE multipliers must be positive odd I64 values"
                    );
                    let sizes = s.i64(&format!("{p}.ngram_heads_vocab_sizes"), &[heads])?;
                    let offsets = s.i64(&format!("{p}.ngram_heads_offsets"), &[heads])?;
                    validate_ple_hashes(&cfg, ple, ordinal, &multipliers, &sizes)?;
                    let mut rows = 0usize;
                    for (&size, &offset) in sizes.iter().zip(&offsets) {
                        let size = usize::try_from(size)
                            .ok()
                            .filter(|&n| n > 0)
                            .context("qwen4_exp: invalid PLE vocabulary size")?;
                        ensure!(
                            offset >= 0 && usize::try_from(offset).ok() == Some(rows),
                            "qwen4_exp: PLE offsets must be cumulative"
                        );
                        rows = rows
                            .checked_add(size)
                            .context("qwen4_exp: PLE vocabulary overflow")?;
                    }
                    let padded_rows = rows
                        .checked_add(ple.vocab_divisor - 1)
                        .context("qwen4_exp: PLE padding overflow")?
                        / ple.vocab_divisor
                        * ple.vocab_divisor;
                    if s.has(&format!("{p}.ngram_embedding.weight")) {
                        s.add(
                            &format!("{b}.ple_embd.weight"),
                            &format!("{p}.ngram_embedding.weight"),
                            &[padded_rows, ple.embed_dim / heads],
                        )?;
                    } else {
                        ensure!(
                            ple.split_parts <= padded_rows
                                && ple.split_parts <= s.st.tensors().len(),
                            "qwen4_exp: invalid PLE shard count"
                        );
                        ensure!(
                            padded_rows.is_multiple_of(ple.split_parts),
                            "qwen4_exp: PLE rows must divide evenly into shards"
                        );
                        for part in 0..ple.split_parts {
                            s.add(
                                &format!("{b}.ple_embd.{part}.weight"),
                                &format!("{p}.ngram_embedding.shard_{part}.weight"),
                                &[padded_rows / ple.split_parts, ple.embed_dim / heads],
                            )?;
                        }
                    }
                }
            }
        }
        // Refuse unknown TEXT weights/buffers instead of silently losing architecture features.
        // Vision is deliberately excluded: this slice supports the text tower only.
        for t in s.st.tensors() {
            ensure!(s.consumed.contains(&t.name) || t.name.starts_with("model.visual.") || t.name.starts_with("visual."),
                "qwen4_exp: unrecognized tensor {} (only single-file text checkpoints are supported)", t.name);
        }
        for t in s.st.i64_buffers() {
            ensure!(
                s._ple_buffers.contains_key(&t.name),
                "qwen4_exp: unrecognized I64 buffer {}",
                t.name
            );
        }
        Ok(s)
    }

    fn has(&self, hf: &str) -> bool {
        self.source_index.contains_key(hf)
    }

    fn add(&mut self, name: &str, hf: &str, shape: &[usize]) -> Result<()> {
        let &index = self.source_index.get(hf).with_context(|| {
            format!("qwen4_exp: missing tensor {hf} (all text weights must be in one file)")
        })?;
        let t = &self.st.tensors()[index];
        ensure!(
            matches!(t.dtype, DType::F32 | DType::F16 | DType::Bf16) && t.shape == shape,
            "qwen4_exp: {hf} must be dense float {shape:?}, got {:?} {:?}",
            t.dtype,
            t.shape
        );
        ensure!(
            !self.names.contains_key(name),
            "qwen4_exp: duplicate mapped tensor {name}"
        );
        self.infos.push(TensorInfo {
            name: name.into(),
            shape: shape.iter().rev().copied().collect(),
            dtype: t.dtype,
            offset: t.offset,
            nbytes: t.nbytes,
        });
        self.names.insert(name.into(), hf.into());
        self.consumed.insert(hf.into());
        Ok(())
    }

    fn hc(
        &mut self,
        name: &str,
        hf: &str,
        ch: usize,
        r: usize,
        inject: Option<usize>,
    ) -> Result<()> {
        self.add(
            &format!("{name}_norm.weight"),
            &format!("{hf}.hc_norm.weight"),
            &[ch],
        )?;
        self.add(
            &format!("{name}_down.weight"),
            &format!("{hf}.input_mix_weight_down.weight"),
            &[r, ch],
        )?;
        self.add(
            &format!("{name}_up.weight"),
            &format!("{hf}.input_mix_weight_up.weight"),
            &[ch, r],
        )?;
        if let Some(c) = inject {
            self.add(
                &format!("{name}_inject.weight"),
                &format!("{hf}.block_inject_weight.weight"),
                &[c, ch],
            )?;
        }
        Ok(())
    }

    fn ffn(&mut self, name: &str, hf: &str, h: usize, ff: usize) -> Result<()> {
        self.add(
            &format!("{name}.gate.weight"),
            &format!("{hf}.gate_proj.weight"),
            &[ff, h],
        )?;
        self.add(
            &format!("{name}.up.weight"),
            &format!("{hf}.up_proj.weight"),
            &[ff, h],
        )?;
        self.add(
            &format!("{name}.down.weight"),
            &format!("{hf}.down_proj.weight"),
            &[h, ff],
        )?;
        Ok(())
    }

    fn i64(&mut self, hf: &str, shape: &[usize]) -> Result<Vec<i64>> {
        let t = self
            .st
            .i64_buffers()
            .iter()
            .find(|t| t.name == hf)
            .with_context(|| format!("qwen4_exp: missing persistent I64 PLE buffer {hf}"))?;
        ensure!(
            t.shape == shape,
            "qwen4_exp: {hf} expected I64 {shape:?}, got {:?}",
            t.shape
        );
        let values: Vec<_> = self
            .st
            .i64_buffer_bytes(hf)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&b| i64::from_le_bytes(b))
            .collect();
        self._ple_buffers.insert(hf.into(), values.clone());
        Ok(values)
    }
}

impl WeightSource for Qwen4ExpSource {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    fn tensors(&self) -> &[TensorInfo] {
        &self.infos
    }
    fn tensor_bytes(&self, name: &str) -> infr_core::error::Result<&[u8]> {
        let hf = self
            .names
            .get(name)
            .ok_or_else(|| Error::Loader(format!("tensor not found: '{name}'")))?;
        self.st.tensor_bytes(hf)
    }
    fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
