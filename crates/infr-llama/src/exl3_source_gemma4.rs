use super::*;

impl Exl3Source {
    pub(super) fn open_gemma4(
        st: infr_safetensors::Safetensors,
        cfg: &serde_json::Value,
        dir: &Path,
    ) -> Result<Self> {
        let text = &cfg["text_config"];
        ensure!(
            matches!(
                text["model_type"].as_str(),
                Some("gemma4_text" | "gemma4_unified_text")
            ),
            "exl3: unsupported Gemma4 text configuration"
        );
        let positive = |key: &str| -> Result<usize> {
            let value = text[key]
                .as_u64()
                .filter(|&v| v > 0 && v <= u32::MAX as u64)
                .with_context(|| format!("exl3: Gemma4 {key} must be a positive u32"))?;
            Ok(value as usize)
        };
        let (ne, nf, nl, nh, nk, vocab) = (
            positive("hidden_size")?,
            positive("intermediate_size")?,
            positive("num_hidden_layers")?,
            positive("num_attention_heads")?,
            positive("num_key_value_heads")?,
            positive("vocab_size")?,
        );
        let hd_swa = positive("head_dim")?;
        let hd_full = positive("global_head_dim")?;
        let nk_full = if text["num_global_key_value_heads"].is_null() {
            nk
        } else {
            positive("num_global_key_value_heads")?
        };
        let npl = text["hidden_size_per_layer_input"].as_u64().unwrap_or(0) as usize;
        let shared = text["num_kv_shared_layers"].as_u64().unwrap_or(0) as usize;
        ensure!(
            shared < nl && (shared == 0 || nl - shared >= 2),
            "exl3: invalid Gemma4 KV sharing"
        );
        ensure!(
            nh.is_multiple_of(nk) && nh.is_multiple_of(nk_full),
            "exl3: invalid Gemma4 GQA heads"
        );
        ensure!(
            text["hidden_activation"] == "gelu_pytorch_tanh"
                && !text["attention_bias"].as_bool().unwrap_or(false),
            "exl3: unsupported Gemma4 activation or attention bias"
        );
        ensure!(
            !text["enable_moe_block"].as_bool().unwrap_or(false)
                && !text["use_double_wide_mlp"].as_bool().unwrap_or(false)
                && text["num_experts"].as_u64().unwrap_or(0) == 0,
            "exl3: Gemma4 EXL3 supports dense, uniform-width FFNs only"
        );
        let types = text["layer_types"]
            .as_array()
            .context("exl3: Gemma4 layer_types missing")?;
        ensure!(
            types.len() == nl,
            "exl3: Gemma4 layer_types length mismatch"
        );
        let period = types
            .iter()
            .position(|t| t == "full_attention")
            .map(|i| i + 1)
            .context("exl3: Gemma4 needs full attention layers")?;
        ensure!(
            period > 1 && nl.is_multiple_of(period) && (nl - shared).is_multiple_of(period),
            "exl3: unsupported Gemma4 attention period or shared-KV boundary"
        );
        for (i, kind) in types.iter().enumerate() {
            ensure!(
                kind == if (i + 1).is_multiple_of(period) {
                    "full_attention"
                } else {
                    "sliding_attention"
                },
                "exl3: nonperiodic Gemma4 attention layout"
            );
        }
        let float = |value: &serde_json::Value, name: &str| -> Result<f64> {
            let f = value
                .as_f64()
                .with_context(|| format!("exl3: missing Gemma4 {name}"))?;
            ensure!(f.is_finite() && f > 0.0, "exl3: invalid Gemma4 {name}");
            Ok(f)
        };
        let full_rope = &text["rope_parameters"]["full_attention"];
        let swa_rope = &text["rope_parameters"]["sliding_attention"];
        ensure!(
            full_rope["rope_type"] == "proportional" && swa_rope["rope_type"] == "default",
            "exl3: unsupported Gemma4 RoPE type"
        );
        let fraction = float(&full_rope["partial_rotary_factor"], "partial_rotary_factor")?;
        let rotated_pairs = (hd_full / 2) as f64 * fraction;
        ensure!(
            hd_full.is_multiple_of(2) && fraction <= 1.0 && rotated_pairs.fract() == 0.0,
            "exl3: invalid Gemma4 proportional RoPE dimensions"
        );
        let mut metadata = Metadata::default();
        metadata.kv.insert(
            "general.architecture".into(),
            MetaValue::Str("gemma4".into()),
        );
        for (key, value) in [
            ("block_count", nl),
            ("embedding_length", ne),
            ("feed_forward_length", nf),
            ("context_length", positive("max_position_embeddings")?),
            ("attention.head_count", nh),
            ("attention.key_length", hd_full),
            ("attention.value_length", hd_full),
            ("attention.key_length_swa", hd_swa),
            ("attention.value_length_swa", hd_swa),
            ("rope.dimension_count", hd_full),
            ("rope.dimension_count_swa", hd_swa),
            ("attention.sliding_window", positive("sliding_window")?),
            ("attention.shared_kv_layers", shared),
            ("embedding_length_per_layer_input", npl),
        ] {
            metadata
                .kv
                .insert(format!("gemma4.{key}"), MetaValue::U64(value as u64));
        }
        metadata.kv.insert(
            "gemma4.attention.head_count_kv".into(),
            MetaValue::Arr(
                (0..nl)
                    .map(|i| {
                        MetaValue::U64(if (i + 1).is_multiple_of(period) {
                            nk_full
                        } else {
                            nk
                        } as u64)
                    })
                    .collect(),
            ),
        );
        metadata.kv.insert(
            "gemma4.attention.sliding_window_pattern".into(),
            MetaValue::Arr(
                (0..nl)
                    .map(|i| MetaValue::Bool(!(i + 1).is_multiple_of(period)))
                    .collect(),
            ),
        );
        for (key, value) in [
            (
                "attention.layer_norm_rms_epsilon",
                float(&text["rms_norm_eps"], "rms_norm_eps")?,
            ),
            (
                "rope.freq_base",
                float(&full_rope["rope_theta"], "full rope_theta")?,
            ),
            (
                "rope.freq_base_swa",
                float(&swa_rope["rope_theta"], "sliding rope_theta")?,
            ),
            (
                "final_logit_softcapping",
                float(&text["final_logit_softcapping"], "final_logit_softcapping")?,
            ),
        ] {
            metadata
                .kv
                .insert(format!("gemma4.{key}"), MetaValue::F64(value));
        }
        for (key, source) in [
            ("bos_token_id", "bos_token_id"),
            ("eos_token_id", "eos_token_id"),
            ("padding_token_id", "pad_token_id"),
        ] {
            let id = text[source]
                .as_u64()
                .filter(|&id| id < vocab as u64)
                .with_context(|| format!("exl3: invalid Gemma4 {source}"))?;
            metadata
                .kv
                .insert(format!("tokenizer.ggml.{key}"), MetaValue::U64(id));
        }
        if let Some(ids) = cfg["eos_token_id"].as_array() {
            let ids = ids
                .iter()
                .map(|id| {
                    let id = id
                        .as_u64()
                        .filter(|&id| id < vocab as u64)
                        .context("exl3: invalid Gemma4 EOS id")?;
                    Ok(MetaValue::U64(id))
                })
                .collect::<Result<Vec<_>>>()?;
            metadata
                .kv
                .insert("tokenizer.ggml.eos_token_ids".into(), MetaValue::Arr(ids));
        }
        metadata
            .kv
            .insert("tokenizer.ggml.add_bos_token".into(), MetaValue::Bool(true));
        metadata.kv.insert(
            "infr.exl3.embedding_fp16_scale".into(),
            MetaValue::Bool(true),
        );
        let mut source = Self::with_metadata(st, metadata, dir)?;
        source.add_embedding(
            "token_embd.weight",
            "model.language_model.embed_tokens",
            ne,
            vocab,
        )?;
        source.add_dense_with(
            "output_norm.weight",
            "model.language_model.norm.weight",
            &[ne],
            None,
            DenseOptions {
                f32: true,
                ..Default::default()
            },
        )?;
        source.add_linear("output.weight", "lm_head", ne, vocab)?;
        let frequencies: Vec<u8> = (0..hd_full / 2)
            .flat_map(|i| {
                (if i < rotated_pairs as usize {
                    1.0f32
                } else {
                    1e30f32
                })
                .to_le_bytes()
            })
            .collect();
        source.infos.push(TensorInfo {
            name: "rope_freqs.weight".into(),
            shape: vec![hd_full / 2],
            dtype: DType::F32,
            offset: 0,
            nbytes: frequencies.len(),
        });
        source.owned.insert("rope_freqs.weight".into(), frequencies);
        if npl > 0 {
            let total = nl
                .checked_mul(npl)
                .context("exl3: Gemma4 per-layer width overflow")?;
            let ple_vocab = text["vocab_size_per_layer_input"]
                .as_u64()
                .unwrap_or(vocab as u64) as usize;
            ensure!(
                ple_vocab == vocab,
                "exl3: unsupported Gemma4 per-layer vocabulary"
            );
            source.add_embedding(
                "per_layer_token_embd.weight",
                "model.language_model.embed_tokens_per_layer",
                total,
                vocab,
            )?;
            source.add_linear(
                "per_layer_model_proj.weight",
                "model.language_model.per_layer_model_projection",
                ne,
                total,
            )?;
            source.add_dense_with(
                "per_layer_proj_norm.weight",
                "model.language_model.per_layer_projection_norm.weight",
                &[npl],
                None,
                DenseOptions {
                    f32: true,
                    ..Default::default()
                },
            )?;
        }
        for layer in 0..nl {
            let prefix = format!("model.language_model.layers.{layer}");
            let full = (layer + 1).is_multiple_of(period);
            let (hd, kv) = if full {
                (hd_full, nk_full)
            } else {
                (hd_swa, nk)
            };
            let qw = nh
                .checked_mul(hd)
                .context("exl3: Gemma4 Q dimension overflow")?;
            let kw = kv
                .checked_mul(hd)
                .context("exl3: Gemma4 KV dimension overflow")?;
            for (name, hf, k, n) in [
                ("attn_q", "self_attn.q_proj", ne, qw),
                ("attn_output", "self_attn.o_proj", qw, ne),
                ("ffn_gate", "mlp.gate_proj", ne, nf),
                ("ffn_up", "mlp.up_proj", ne, nf),
                ("ffn_down", "mlp.down_proj", nf, ne),
            ] {
                source.add_linear(
                    &format!("blk.{layer}.{name}.weight"),
                    &format!("{prefix}.{hf}"),
                    k,
                    n,
                )?;
            }
            if layer < nl - shared {
                source.add_linear(
                    &format!("blk.{layer}.attn_k.weight"),
                    &format!("{prefix}.self_attn.k_proj"),
                    ne,
                    kw,
                )?;
                if source
                    .st
                    .tensors()
                    .iter()
                    .any(|t| t.name == format!("{prefix}.self_attn.v_proj.trellis"))
                {
                    source.add_linear(
                        &format!("blk.{layer}.attn_v.weight"),
                        &format!("{prefix}.self_attn.v_proj"),
                        ne,
                        kw,
                    )?;
                } else {
                    ensure!(
                        full && text["attention_k_eq_v"] == true,
                        "exl3: Gemma4 V projection missing"
                    );
                }
                source.add_dense_with(
                    &format!("blk.{layer}.attn_k_norm.weight"),
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    &[hd],
                    None,
                    DenseOptions {
                        f32: true,
                        ..Default::default()
                    },
                )?;
            }
            for (name, hf, width) in [
                ("attn_norm", "input_layernorm.weight", ne),
                ("post_attention_norm", "post_attention_layernorm.weight", ne),
                ("ffn_norm", "pre_feedforward_layernorm.weight", ne),
                ("post_ffw_norm", "post_feedforward_layernorm.weight", ne),
                ("attn_q_norm", "self_attn.q_norm.weight", hd),
                ("layer_output_scale", "layer_scalar", 1),
            ] {
                source.add_dense_with(
                    &format!("blk.{layer}.{name}.weight"),
                    &format!("{prefix}.{hf}"),
                    &[width],
                    None,
                    DenseOptions {
                        f32: true,
                        ..Default::default()
                    },
                )?;
            }
            if npl > 0 {
                source.add_linear(
                    &format!("blk.{layer}.inp_gate.weight"),
                    &format!("{prefix}.per_layer_input_gate"),
                    ne,
                    npl,
                )?;
                source.add_linear(
                    &format!("blk.{layer}.proj.weight"),
                    &format!("{prefix}.per_layer_projection"),
                    npl,
                    ne,
                )?;
                source.add_dense_with(
                    &format!("blk.{layer}.post_norm.weight"),
                    &format!("{prefix}.post_per_layer_input_norm.weight"),
                    &[ne],
                    None,
                    DenseOptions {
                        f32: true,
                        ..Default::default()
                    },
                )?;
            }
        }
        Ok(source)
    }

    fn add_embedding(
        &mut self,
        name: &str,
        prefix: &str,
        width: usize,
        vocab: usize,
    ) -> Result<()> {
        let packed = format!("{prefix}.trellis");
        let (dtype, nbytes, spec) = if self.st.tensors().iter().any(|t| t.name == packed) {
            let info = find_hf(&self.st, &packed)?;
            let groups = width.div_ceil(256);
            ensure!(
                info.dtype == DType::I16
                    && info.shape.len() == 2
                    && info.shape[0] == vocab
                    && info.shape[1].is_multiple_of(groups),
                "exl3: invalid row embedding shape or dtype"
            );
            let group_words = info.shape[1] / groups;
            ensure!(
                group_words > 1 && (group_words - 1).is_multiple_of(16),
                "exl3: invalid embedding group stride"
            );
            let geometry = Exl3EmbeddingSpec {
                width: u32::try_from(width)?,
                vocab: u32::try_from(vocab)?,
                bits: u32::try_from((group_words - 1) / 16)?,
            };
            ensure!(
                geometry.trellis_bytes()? == info.nbytes,
                "exl3: invalid packed embedding byte length"
            );
            let signs = find_hf(&self.st, &format!("{prefix}.signs"))?;
            ensure!(
                signs.dtype == DType::F16
                    && signs.shape == [groups, 256]
                    && signs.nbytes == geometry.sign_bytes()?,
                "exl3: invalid embedding signs"
            );
            ensure!(
                self.st
                    .tensor_bytes(&signs.name)?
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|p| half::f16::from_le_bytes(*p).is_finite()),
                "exl3: nonfinite embedding signs"
            );
            (
                DType::I16,
                info.nbytes,
                EmbeddingSpec::Packed {
                    hf_prefix: prefix.into(),
                    geometry,
                },
            )
        } else {
            let hf_name = format!("{prefix}.weight");
            let info = find_hf(&self.st, &hf_name)?;
            ensure!(
                matches!(info.dtype, DType::F16 | DType::Bf16) && info.shape == [vocab, width],
                "exl3: unsupported dense embedding dtype or shape"
            );
            (
                info.dtype,
                info.nbytes,
                EmbeddingSpec::Dense {
                    hf_name,
                    width,
                    vocab,
                    dtype: info.dtype,
                },
            )
        };
        self.infos.push(TensorInfo {
            name: name.into(),
            shape: vec![width, vocab],
            dtype,
            offset: 0,
            nbytes,
        });
        self.embeddings.insert(name.into(), spec);
        Ok(())
    }
}
