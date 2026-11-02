//! EXL3 checkpoint names with native trellis or folded reference projections.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{anyhow, ensure, Context, Result};
use infr_core::loader::{MetaValue, Metadata, TensorInfo};
use infr_core::{
    DType, Error, Exl3Codebook, Exl3EmbeddingSpec, Exl3Layout, Exl3Spec, WeightSource,
};

#[path = "exl3_source_dense.rs"]
mod dense;
#[path = "exl3_source_gemma4.rs"]
mod gemma4;
#[path = "exl3_source_qwen35.rs"]
mod qwen35;
use dense::{DenseOp, DenseOptions};
use infr_cpu::exl3::{decode_tile, hadamard128, tile_perm, Codebook};

struct LinearSpec {
    hf_prefix: String,
    k: usize,
    n: usize,
    bits: u32,
    codebook: Exl3Codebook,
    input_layout: Exl3Layout,
    output_layout: Exl3Layout,
    cache: OnceLock<std::result::Result<Vec<u8>, String>>,
}

impl LinearSpec {
    fn projection(&self) -> Exl3Spec {
        Exl3Spec {
            in_f: self.k as u32,
            out_f: self.n as u32,
            bits: self.bits,
            codebook: self.codebook,
            input_layout: self.input_layout,
            output_layout: self.output_layout,
        }
    }
}

struct DenseSpec {
    hf_name: String,
    dtype: DType,
    convert: bool,
    op: DenseOp,
    layout: Exl3Layout,
    cache: OnceLock<std::result::Result<Vec<u8>, String>>,
}

enum EmbeddingSpec {
    Packed {
        hf_prefix: String,
        geometry: Exl3EmbeddingSpec,
    },
    Dense {
        hf_name: String,
        width: usize,
        vocab: usize,
        dtype: DType,
    },
}

pub(crate) struct Exl3Source {
    st: infr_safetensors::Safetensors,
    metadata: Metadata,
    infos: Vec<TensorInfo>,
    linears: HashMap<String, LinearSpec>,
    passthrough: HashMap<String, DenseSpec>,
    embeddings: HashMap<String, EmbeddingSpec>,
    owned: HashMap<String, Vec<u8>>,
    chat_template: Option<String>,
    native: bool,
}

fn find_hf<'a>(st: &'a infr_safetensors::Safetensors, name: &str) -> Result<&'a TensorInfo> {
    st.tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("exl3: missing tensor {name}"))
}

fn dense_size(shape: &[usize], width: usize) -> Result<usize> {
    shape
        .iter()
        .try_fold(width, |n, &d| n.checked_mul(d))
        .ok_or_else(|| anyhow!("exl3: tensor size overflows usize: {shape:?}"))
}

impl Exl3Source {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let st = infr_safetensors::Safetensors::open(path)?;
        let dir = path.parent().unwrap_or(Path::new("."));
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).context("exl3: read adjacent config.json")?,
        )
        .context("exl3: parse config.json")?;
        ensure!(
            cfg["quantization_config"]["quant_method"] == "exl3",
            "exl3: quant_method must be exl3"
        );
        if matches!(
            cfg["model_type"].as_str(),
            Some("gemma4" | "gemma4_unified")
        ) {
            return Self::open_gemma4(st, &cfg, dir);
        }
        ensure!(
            cfg["tie_word_embeddings"] == false,
            "exl3: tied embeddings are unsupported"
        );
        if cfg["model_type"] == "qwen3_5" {
            return Self::open_qwen35(st, &cfg, dir);
        }
        ensure!(
            cfg["model_type"] == "qwen2",
            "exl3: only qwen2 and qwen3_5 are supported"
        );
        ensure!(
            cfg["hidden_act"].as_str().unwrap_or("silu") == "silu",
            "exl3: hidden_act must be silu"
        );
        ensure!(
            !cfg["use_sliding_window"].as_bool().unwrap_or(false),
            "exl3: sliding-window attention is unsupported"
        );
        ensure!(
            cfg["rope_scaling"].is_null(),
            "exl3: rope_scaling is unsupported"
        );
        let uint = |key: &str| -> Result<usize> {
            cfg[key]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| anyhow!("exl3: {key} must be a positive usize"))
        };
        let float = |key: &str| -> Result<f64> {
            cfg[key]
                .as_f64()
                .filter(|n| n.is_finite() && *n > 0.0)
                .ok_or_else(|| anyhow!("exl3: {key} must be positive and finite"))
        };
        let (layers, hidden, ff, heads, kv_heads, vocab) = (
            uint("num_hidden_layers")?,
            uint("hidden_size")?,
            uint("intermediate_size")?,
            uint("num_attention_heads")?,
            uint("num_key_value_heads")?,
            uint("vocab_size")?,
        );
        ensure!(
            hidden.is_multiple_of(heads) && heads.is_multiple_of(kv_heads),
            "exl3: invalid attention head geometry"
        );
        let head_dim = hidden / heads;
        ensure!(
            head_dim.is_multiple_of(2),
            "exl3: rotary head dimension must be even"
        );
        if let Some(explicit) = cfg["head_dim"].as_u64() {
            ensure!(
                explicit == head_dim as u64,
                "exl3: custom head_dim is unsupported"
            );
        }
        let kv_width = kv_heads
            .checked_mul(head_dim)
            .context("exl3: KV width overflow")?;
        ensure!(
            layers <= st.tensors().len(),
            "exl3: layer count exceeds tensor count"
        );
        let eos = cfg["eos_token_id"]
            .as_u64()
            .filter(|&n| n < vocab as u64)
            .context("exl3: eos_token_id must be a vocabulary index")?;
        let mut metadata = Metadata { kv: HashMap::new() };
        metadata.kv.insert(
            "general.architecture".into(),
            MetaValue::Str("qwen2".into()),
        );
        for (key, value) in [
            ("block_count", layers),
            ("embedding_length", hidden),
            ("feed_forward_length", ff),
            ("attention.head_count", heads),
            ("attention.head_count_kv", kv_heads),
            ("context_length", uint("max_position_embeddings")?),
        ] {
            metadata
                .kv
                .insert(format!("qwen2.{key}"), MetaValue::U64(value as u64));
        }
        metadata.kv.insert(
            "qwen2.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F64(float("rms_norm_eps")?),
        );
        metadata.kv.insert(
            "qwen2.rope.freq_base".into(),
            MetaValue::F64(float("rope_theta")?),
        );
        metadata
            .kv
            .insert("tokenizer.ggml.eos_token_id".into(), MetaValue::U64(eos));
        metadata.kv.insert(
            "tokenizer.ggml.add_bos_token".into(),
            MetaValue::Bool(false),
        );
        if let Some(bos) = cfg["bos_token_id"].as_u64() {
            ensure!(bos < vocab as u64, "exl3: bos_token_id outside vocabulary");
            metadata
                .kv
                .insert("tokenizer.ggml.bos_token_id".into(), MetaValue::U64(bos));
        }
        let mut source = Self::with_metadata(st, metadata, dir)?;
        source.add_dense(
            "token_embd.weight",
            "model.embed_tokens.weight",
            &[hidden, vocab],
        )?;
        source.add_dense("output_norm.weight", "model.norm.weight", &[hidden])?;
        for l in 0..layers {
            let p = format!("model.layers.{l}");
            source.add_dense(
                &format!("blk.{l}.attn_norm.weight"),
                &format!("{p}.input_layernorm.weight"),
                &[hidden],
            )?;
            source.add_dense(
                &format!("blk.{l}.ffn_norm.weight"),
                &format!("{p}.post_attention_layernorm.weight"),
                &[hidden],
            )?;
            for (hf, gguf, out) in [
                ("q_proj", "attn_q", hidden),
                ("k_proj", "attn_k", kv_width),
                ("v_proj", "attn_v", kv_width),
            ] {
                source.add_linear(
                    &format!("blk.{l}.{gguf}.weight"),
                    &format!("{p}.self_attn.{hf}"),
                    hidden,
                    out,
                )?;
                source.add_dense(
                    &format!("blk.{l}.{gguf}.bias"),
                    &format!("{p}.self_attn.{hf}.bias"),
                    &[out],
                )?;
            }
            source.add_linear(
                &format!("blk.{l}.attn_output.weight"),
                &format!("{p}.self_attn.o_proj"),
                hidden,
                hidden,
            )?;
            for (hf, gguf, k, n) in [
                ("gate_proj", "ffn_gate", hidden, ff),
                ("up_proj", "ffn_up", hidden, ff),
                ("down_proj", "ffn_down", ff, hidden),
            ] {
                source.add_linear(
                    &format!("blk.{l}.{gguf}.weight"),
                    &format!("{p}.mlp.{hf}"),
                    k,
                    n,
                )?;
            }
        }
        source.add_linear("output.weight", "lm_head", hidden, vocab)?;
        Ok(source)
    }

    fn with_metadata(
        st: infr_safetensors::Safetensors,
        metadata: Metadata,
        dir: &Path,
    ) -> Result<Self> {
        let tokenizer_cfg = dir.join("tokenizer_config.json");
        let mut chat_template = if tokenizer_cfg.exists() {
            let tc: serde_json::Value = serde_json::from_slice(&std::fs::read(&tokenizer_cfg)?)
                .context("exl3: parse tokenizer_config.json")?;
            tc["chat_template"].as_str().map(str::to_owned)
        } else {
            None
        };
        let file = dir.join("chat_template.jinja");
        if chat_template.is_none() && file.exists() {
            chat_template = Some(std::fs::read_to_string(file)?);
        }
        Ok(Self {
            st,
            metadata,
            infos: Vec::new(),
            linears: HashMap::new(),
            passthrough: HashMap::new(),
            embeddings: HashMap::new(),
            owned: HashMap::new(),
            chat_template,
            native: false,
        })
    }

    pub(crate) fn open_native(path: &Path) -> Result<Self> {
        let mut source = Self::open(path)?;
        let mut scales = Vec::new();
        for info in &mut source.infos {
            let Some(linear) = source.linears.get(&info.name) else {
                continue;
            };
            let spec = linear.projection();
            info.nbytes = spec.trellis_bytes()?;
            info.dtype = DType::I16;
            info.shape = vec![linear.k / 16, linear.n / 16, 16 * linear.bits as usize];
            for (suffix, size) in [("suh", linear.k), ("svh", linear.n)] {
                let bytes = source
                    .st
                    .tensor_bytes(&format!("{}.{suffix}", linear.hf_prefix))?;
                ensure!(
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .all(|p| half::f16::from_le_bytes(*p).is_finite()),
                    "exl3: {}.{suffix} contains non-finite scales",
                    linear.hf_prefix
                );
                scales.push(TensorInfo {
                    name: format!("{}.{suffix}", info.name),
                    shape: vec![size],
                    dtype: DType::F16,
                    offset: 0,
                    nbytes: size * 2,
                });
            }
        }
        source.infos.extend(scales);
        for dense in source.passthrough.values().filter(|dense| dense.convert) {
            source.dense_bytes(dense)?;
        }
        source.native = true;
        Ok(source)
    }

    pub(crate) fn native_spec(&self, name: &str) -> Option<infr_core::Exl3Spec> {
        if !self.native {
            return None;
        }
        self.linears.get(name).map(LinearSpec::projection)
    }

    pub(crate) fn is_native(&self) -> bool {
        self.native
    }

    fn add_dense(&mut self, name: &str, hf: &str, shape: &[usize]) -> Result<()> {
        self.add_dense_with(name, hf, shape, None, DenseOptions::default())
    }

    fn add_linear(&mut self, name: &str, prefix: &str, k: usize, n: usize) -> Result<()> {
        self.add_linear_with(
            name,
            prefix,
            k,
            n,
            Exl3Layout::Identity,
            Exl3Layout::Identity,
        )
    }

    fn add_linear_with(
        &mut self,
        name: &str,
        prefix: &str,
        k: usize,
        n: usize,
        input_layout: Exl3Layout,
        output_layout: Exl3Layout,
    ) -> Result<()> {
        input_layout.validate(u32::try_from(k)?)?;
        output_layout.validate(u32::try_from(n)?)?;
        ensure!(
            k.is_multiple_of(128) && n.is_multiple_of(128),
            "exl3: {prefix} dimensions ({k},{n}) must be 128-divisible"
        );
        let info = find_hf(&self.st, &format!("{prefix}.trellis"))?;
        ensure!(
            info.dtype == DType::I16 && info.shape.len() == 3,
            "exl3: {prefix}.trellis must be rank-3 I16"
        );
        let words = info.shape[2];
        ensure!(
            words.is_multiple_of(16),
            "exl3: {prefix} fractional rates are unsupported"
        );
        let bits = u32::try_from(words / 16).context("exl3: rate overflows u32")?;
        ensure!(
            (1..=8).contains(&bits),
            "exl3: {prefix} rate {bits} is unsupported"
        );
        ensure!(
            info.shape == [k / 16, n / 16, words],
            "exl3: {prefix}.trellis shape {:?} disagrees with config ({k},{n})",
            info.shape
        );
        let mut codebook = Exl3Codebook::Cb0;
        for (suffix, kind, multiplier) in [
            ("mcg", Exl3Codebook::Mcg, 0xcbac1fed_u32),
            ("mul1", Exl3Codebook::Mul1, 0x83dcd12d_u32),
        ] {
            let marker = format!("{prefix}.{suffix}");
            if let Some(info) = self.st.tensors().iter().find(|t| t.name == marker) {
                ensure!(
                    codebook == Exl3Codebook::Cb0,
                    "exl3: {prefix} has conflicting codebook markers"
                );
                ensure!(
                    info.dtype == DType::I32 && info.shape.is_empty(),
                    "exl3: {marker} must be scalar I32"
                );
                ensure!(
                    self.st.tensor_bytes(&marker)? == multiplier.to_le_bytes(),
                    "exl3: {marker} has an unsupported multiplier"
                );
                codebook = kind;
            }
        }
        for (scale, size) in [("suh", k), ("svh", n)] {
            let info = find_hf(&self.st, &format!("{prefix}.{scale}"))?;
            ensure!(
                info.dtype == DType::F16 && info.shape == [size],
                "exl3: {prefix}.{scale} must be F16 [{size}]"
            );
        }
        self.infos.push(TensorInfo {
            name: name.into(),
            shape: vec![k, n],
            dtype: DType::F16,
            offset: 0,
            nbytes: dense_size(&[k, n], 2)?,
        });
        self.linears.insert(
            name.into(),
            LinearSpec {
                hf_prefix: prefix.into(),
                k,
                n,
                bits,
                codebook,
                input_layout,
                output_layout,
                cache: OnceLock::new(),
            },
        );
        Ok(())
    }

    fn fold_uncached(&self, spec: &LinearSpec) -> Result<Vec<u8>> {
        let trellis: Vec<_> = self
            .st
            .tensor_bytes(&format!("{}.trellis", spec.hf_prefix))?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u16::from_le_bytes(*p))
            .collect();
        let scales = |suffix: &str| -> Result<Vec<f32>> {
            self.st
                .tensor_bytes(&format!("{}.{suffix}", spec.hf_prefix))?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| {
                    let value = half::f16::from_le_bytes(*p).to_f32();
                    ensure!(
                        value.is_finite(),
                        "exl3: {}.{suffix} contains non-finite scales",
                        spec.hf_prefix
                    );
                    Ok(value)
                })
                .collect()
        };
        fold_trellis(
            &trellis,
            &scales("suh")?,
            &scales("svh")?,
            spec.projection(),
        )
    }

    // Loader-owned folds bypass the borrow cache to avoid retaining a second full model.
    pub(crate) fn take_fold(&self, name: &str) -> Result<Option<Vec<u8>>> {
        ensure!(
            !self.native || !self.linears.contains_key(name),
            "exl3: native projection cannot be folded"
        );
        self.linears
            .get(name)
            .map(|spec| self.fold_uncached(spec))
            .transpose()
    }
}

fn fold_trellis(trellis: &[u16], suh: &[f32], svh: &[f32], spec: Exl3Spec) -> Result<Vec<u8>> {
    use rayon::prelude::*;
    spec.trellis_bytes()?;
    let (k, n, bits, codebook) = (
        spec.in_f as usize,
        spec.out_f as usize,
        spec.bits,
        spec.codebook,
    );
    let w = 16 * bits as usize;
    ensure!(
        trellis.len() == dense_size(&[k / 16, n / 16], w)?,
        "exl3: invalid trellis length"
    );
    ensure!(
        suh.len() == k && svh.len() == n,
        "exl3: invalid scale lengths"
    );
    ensure!(
        suh.iter().chain(svh).all(|v| v.is_finite()),
        "exl3: non-finite scales"
    );
    let perm = tile_perm(bits);
    let mut out = vec![0u8; dense_size(&[k, n], 2)?];
    // Each parallel task owns 128 output rows, including every input-column segment.
    out.par_chunks_mut(128 * k * 2)
        .enumerate()
        .try_for_each(|(bj, band)| -> Result<()> {
            let mut d = vec![0f32; 128 * 128];
            let mut tile32 = vec![0u32; w / 2];
            let mut col = [0f32; 128];
            for bi in 0..k / 128 {
                for ti in 0..8 {
                    for tj in 0..8 {
                        let base = ((bi * 8 + ti) * (n / 16) + bj * 8 + tj) * w;
                        for (dst, pair) in tile32
                            .iter_mut()
                            .zip(trellis[base..base + w].as_chunks::<2>().0)
                        {
                            *dst = pair[0] as u32 | ((pair[1] as u32) << 16);
                        }
                        for (s, v) in decode_tile(Codebook::from(codebook), bits, &tile32)
                            .iter()
                            .enumerate()
                        {
                            let p = perm[s] as usize;
                            d[(ti * 16 + p / 16) * 128 + tj * 16 + p % 16] = *v;
                        }
                    }
                }
                for row in d.as_chunks_mut::<128>().0 {
                    hadamard128(row);
                }
                for c in 0..128 {
                    for r in 0..128 {
                        col[r] = d[r * 128 + c];
                    }
                    hadamard128(&mut col);
                    for r in 0..128 {
                        d[r * 128 + c] = col[r];
                    }
                }
                for rr in 0..128 {
                    for cc in 0..128 {
                        let value = half::f16::from_f32(
                            d[rr * 128 + cc] * suh[bi * 128 + rr] * svh[bj * 128 + cc],
                        );
                        ensure!(value.is_finite(), "exl3: folded weight overflows F16");
                        let o = (cc * k + bi * 128 + rr) * 2;
                        band[o..o + 2].copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
            Ok(())
        })?;
    let swaps = |layout: Exl3Layout, width: usize| {
        let mut map: Vec<_> = (0..width).map(|i| layout.index(i)).collect();
        let mut swaps = Vec::new();
        for i in 0..width {
            while map[i] != i {
                let j = map[i];
                map.swap(i, j);
                swaps.push((i, j));
            }
        }
        swaps
    };
    for (i, j) in swaps(spec.output_layout, n) {
        let (left, right) = out.split_at_mut(j * k * 2);
        left[i * k * 2..(i + 1) * k * 2].swap_with_slice(&mut right[..k * 2]);
    }
    let col_swaps = swaps(spec.input_layout, k);
    if !col_swaps.is_empty() {
        out.par_chunks_mut(k * 2).for_each(|row| {
            let elements = row.as_chunks_mut::<2>().0;
            for &(i, j) in &col_swaps {
                elements.swap(i, j);
            }
        });
    }
    Ok(out)
}

impl WeightSource for Exl3Source {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    fn tensors(&self) -> &[TensorInfo] {
        &self.infos
    }
    fn tensor_rows(&self, name: &str, ids: &[u32]) -> infr_core::Result<Option<Vec<f32>>> {
        let Some(spec) = self.embeddings.get(name) else {
            return Ok(None);
        };
        match spec {
            EmbeddingSpec::Packed {
                hf_prefix,
                geometry,
            } => {
                let trellis = self.st.tensor_bytes(&format!("{hf_prefix}.trellis"))?;
                let signs = self.st.tensor_bytes(&format!("{hf_prefix}.signs"))?;
                infr_cpu::exl3_embedding::lookup(trellis, signs, ids, *geometry).map(Some)
            }
            EmbeddingSpec::Dense {
                hf_name,
                width,
                vocab,
                dtype,
            } => {
                if ids.iter().any(|&id| id as usize >= *vocab) {
                    return Err(Error::Loader("embedding token id out of range".into()));
                }
                let bytes = self.st.tensor_bytes(hf_name)?;
                let mut rows = Vec::with_capacity(
                    ids.len()
                        .checked_mul(*width)
                        .ok_or_else(|| Error::Loader("embedding rows size overflow".into()))?,
                );
                for &id in ids {
                    for word in bytes[id as usize * width * 2..(id as usize + 1) * width * 2]
                        .as_chunks::<2>()
                        .0
                    {
                        let raw = if *dtype == DType::Bf16 {
                            half::bf16::from_le_bytes(*word).to_f32()
                        } else {
                            half::f16::from_le_bytes(*word).to_f32()
                        };
                        let value = half::f16::from_f32(raw).to_f32();
                        if !value.is_finite() {
                            return Err(Error::Loader("nonfinite embedding row".into()));
                        }
                        rows.push(value);
                    }
                }
                Ok(Some(rows))
            }
        }
    }
    fn tensor_bytes(&self, name: &str) -> infr_core::error::Result<&[u8]> {
        if let Some(bytes) = self.owned.get(name) {
            return Ok(bytes);
        }
        if self.embeddings.contains_key(name) {
            return Err(Error::Loader(format!(
                "{name}: row-only embedding cannot be expanded"
            )));
        }
        if self.native {
            if let Some(spec) = self.linears.get(name) {
                return self.st.tensor_bytes(&format!("{}.trellis", spec.hf_prefix));
            }
            if let Some((base, suffix)) = name.rsplit_once('.') {
                if matches!(suffix, "suh" | "svh") {
                    if let Some(spec) = self.linears.get(base) {
                        return self
                            .st
                            .tensor_bytes(&format!("{}.{suffix}", spec.hf_prefix));
                    }
                }
            }
        }
        if let Some(spec) = self.linears.get(name) {
            return match spec
                .cache
                .get_or_init(|| self.fold_uncached(spec).map_err(|e| e.to_string()))
            {
                Ok(bytes) => Ok(bytes),
                Err(error) => Err(Error::Loader(error.clone())),
            };
        }
        let spec = self
            .passthrough
            .get(name)
            .ok_or_else(|| Error::Loader(format!("tensor not found: '{name}'")))?;
        self.dense_bytes(spec)
            .map_err(|e| Error::Loader(e.to_string()))
    }
    fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
#[path = "exl3_source_tests.rs"]
mod tests;
