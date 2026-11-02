use super::*;

impl Exl3Source {
    pub(super) fn open_qwen35(
        st: infr_safetensors::Safetensors,
        outer: &serde_json::Value,
        dir: &Path,
    ) -> Result<Self> {
        let cfg = &outer["text_config"];
        ensure!(
            cfg["model_type"] == "qwen3_5_text",
            "exl3: missing Qwen3.5 text_config"
        );
        let uint = |key: &str| -> Result<usize> {
            cfg[key]
                .as_u64()
                .filter(|&n| n > 0 && n <= u32::MAX as u64)
                .map(|n| n as usize)
                .ok_or_else(|| anyhow!("exl3: {key} must be a positive u32"))
        };
        let float = |value: &serde_json::Value, name: &str| -> Result<f64> {
            value
                .as_f64()
                .filter(|n| n.is_finite() && *n > 0.0)
                .ok_or_else(|| anyhow!("exl3: {name} must be positive and finite"))
        };
        ensure!(
            !cfg["attention_bias"].as_bool().unwrap_or(false),
            "exl3: Qwen3.5 attention bias is unsupported"
        );
        ensure!(
            cfg["attn_output_gate"] == true,
            "exl3: Qwen3.5 output gate is required"
        );
        ensure!(
            cfg["hidden_act"].as_str().unwrap_or("silu") == "silu",
            "exl3: hidden_act must be silu"
        );
        ensure!(
            !cfg["tie_word_embeddings"].as_bool().unwrap_or(false),
            "exl3: tied text embeddings are unsupported"
        );
        ensure!(
            !cfg["use_sliding_window"].as_bool().unwrap_or(false) && cfg["rope_scaling"].is_null(),
            "exl3: custom attention scaling is unsupported"
        );
        ensure!(
            cfg["mlp_only_layers"].as_array().is_none_or(Vec::is_empty),
            "exl3: MLP-only layers are unsupported"
        );
        ensure!(
            cfg["mamba_ssm_dtype"].as_str().unwrap_or("float32") == "float32",
            "exl3: SSM state must be float32"
        );
        let (layers, hidden, ff, heads, kv_heads, hd, vocab, interval) = (
            uint("num_hidden_layers")?,
            uint("hidden_size")?,
            uint("intermediate_size")?,
            uint("num_attention_heads")?,
            uint("num_key_value_heads")?,
            uint("head_dim")?,
            uint("vocab_size")?,
            uint("full_attention_interval")?,
        );
        ensure!(
            layers <= st.tensors().len() && heads.is_multiple_of(kv_heads) && hd.is_multiple_of(2),
            "exl3: invalid Qwen3.5 layer or head geometry"
        );
        if let Some(types) = cfg["layer_types"].as_array() {
            ensure!(
                types.len() == layers
                    && types.iter().enumerate().all(|(i, kind)| {
                        kind.as_str()
                            == Some(if (i + 1).is_multiple_of(interval) {
                                "full_attention"
                            } else {
                                "linear_attention"
                            })
                    }),
                "exl3: layer_types disagrees with full_attention_interval"
            );
        }
        let (nk, nv, state, vd, conv) = (
            uint("linear_num_key_heads")?,
            uint("linear_num_value_heads")?,
            uint("linear_key_head_dim")?,
            uint("linear_value_head_dim")?,
            uint("linear_conv_kernel_dim")?,
        );
        ensure!(
            nv.is_multiple_of(nk) && state == vd && state == 128,
            "exl3: unsupported DeltaNet head geometry"
        );
        let key_width = nk
            .checked_mul(state)
            .context("exl3: DeltaNet key width overflow")?;
        let inner = nv
            .checked_mul(vd)
            .context("exl3: DeltaNet value width overflow")?;
        let prefix = key_width
            .checked_mul(2)
            .context("exl3: DeltaNet prefix overflow")?;
        let qkv = prefix
            .checked_add(inner)
            .context("exl3: DeltaNet qkv width overflow")?;
        let rope = &cfg["rope_parameters"];
        ensure!(
            rope["rope_type"].as_str() == Some("default"),
            "exl3: non-default Qwen3.5 RoPE is unsupported"
        );
        let factor = float(&rope["partial_rotary_factor"], "partial_rotary_factor")?;
        let rotary = factor * hd as f64;
        ensure!(
            factor <= 1.0
                && rotary.fract() == 0.0
                && rotary >= 2.0
                && (rotary as usize).is_multiple_of(2),
            "exl3: invalid partial rotary dimension"
        );
        let rotary = rotary as usize;
        let sections = rope["mrope_section"]
            .as_array()
            .context("exl3: mrope_section is required")?;
        let sections: Vec<_> = sections
            .iter()
            .map(|v| {
                v.as_u64()
                    .filter(|&n| n <= u32::MAX as u64)
                    .ok_or_else(|| anyhow!("exl3: invalid mrope_section"))
            })
            .collect::<Result<_>>()?;
        ensure!(
            sections.len() == 3 && sections.iter().sum::<u64>() == rotary as u64 / 2,
            "exl3: mrope_section disagrees with rotary dimension"
        );
        let eos = cfg["eos_token_id"]
            .as_u64()
            .filter(|&n| n < vocab as u64)
            .context("exl3: invalid eos_token_id")?;
        let mut metadata = Metadata { kv: HashMap::new() };
        metadata.kv.insert(
            "general.architecture".into(),
            MetaValue::Str("qwen35".into()),
        );
        for (key, value) in [
            ("block_count", layers),
            ("embedding_length", hidden),
            ("feed_forward_length", ff),
            ("context_length", uint("max_position_embeddings")?),
            ("attention.head_count", heads),
            ("attention.head_count_kv", kv_heads),
            ("attention.key_length", hd),
            ("attention.value_length", hd),
            ("full_attention_interval", interval),
            ("ssm.conv_kernel", conv),
            ("ssm.state_size", state),
            ("ssm.inner_size", inner),
            ("ssm.group_count", nk),
            ("ssm.time_step_rank", nv),
            ("rope.dimension_count", rotary),
        ] {
            metadata
                .kv
                .insert(format!("qwen35.{key}"), MetaValue::U64(value as u64));
        }
        metadata.kv.insert(
            "qwen35.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F64(float(&cfg["rms_norm_eps"], "rms_norm_eps")?),
        );
        metadata.kv.insert(
            "qwen35.rope.freq_base".into(),
            MetaValue::F64(float(&rope["rope_theta"], "rope_theta")?),
        );
        metadata.kv.insert(
            "qwen35.rope.dimension_sections".into(),
            MetaValue::Arr(
                sections
                    .into_iter()
                    .chain([0])
                    .map(MetaValue::U64)
                    .collect(),
            ),
        );
        metadata
            .kv
            .insert("tokenizer.ggml.eos_token_id".into(), MetaValue::U64(eos));
        metadata.kv.insert(
            "tokenizer.ggml.add_bos_token".into(),
            MetaValue::Bool(false),
        );
        let mut source = Self::with_metadata(st, metadata, dir)?;
        let normal = DenseOptions {
            f32: true,
            op: DenseOp::AddOne,
            ..Default::default()
        };
        let head_layout = |prefix: usize, head_dim: usize| -> Result<Exl3Layout> {
            Ok(Exl3Layout::GroupedValueHeads {
                prefix: u32::try_from(prefix)?,
                key_heads: u32::try_from(nk)?,
                value_heads: u32::try_from(nv)?,
                head_dim: u32::try_from(head_dim)?,
            })
        };
        source.add_dense(
            "token_embd.weight",
            "model.language_model.embed_tokens.weight",
            &[hidden, vocab],
        )?;
        source.add_dense_with(
            "output_norm.weight",
            "model.language_model.norm.weight",
            &[hidden],
            None,
            normal,
        )?;
        for l in 0..layers {
            let p = format!("model.language_model.layers.{l}");
            let g = format!("blk.{l}");
            for (hf, dest) in [
                ("input_layernorm", "attn_norm"),
                ("post_attention_layernorm", "post_attention_norm"),
            ] {
                source.add_dense_with(
                    &format!("{g}.{dest}.weight"),
                    &format!("{p}.{hf}.weight"),
                    &[hidden],
                    None,
                    normal,
                )?;
            }
            if (l + 1).is_multiple_of(interval) {
                let q_width = heads
                    .checked_mul(hd)
                    .and_then(|n| n.checked_mul(2))
                    .context("exl3: attention width overflow")?;
                let kv_width = kv_heads
                    .checked_mul(hd)
                    .context("exl3: KV width overflow")?;
                for (hf, dest, width) in [
                    ("q_proj", "attn_q", q_width),
                    ("k_proj", "attn_k", kv_width),
                    ("v_proj", "attn_v", kv_width),
                ] {
                    source.add_linear(
                        &format!("{g}.{dest}.weight"),
                        &format!("{p}.self_attn.{hf}"),
                        hidden,
                        width,
                    )?;
                }
                for (hf, dest) in [("q_norm", "attn_q_norm"), ("k_norm", "attn_k_norm")] {
                    source.add_dense_with(
                        &format!("{g}.{dest}.weight"),
                        &format!("{p}.self_attn.{hf}.weight"),
                        &[hd],
                        None,
                        normal,
                    )?;
                }
                source.add_linear(
                    &format!("{g}.attn_output.weight"),
                    &format!("{p}.self_attn.o_proj"),
                    q_width / 2,
                    hidden,
                )?;
            } else {
                let p = format!("{p}.linear_attn");
                source.add_linear_with(
                    &format!("{g}.attn_qkv.weight"),
                    &format!("{p}.in_proj_qkv"),
                    hidden,
                    qkv,
                    Exl3Layout::Identity,
                    head_layout(prefix, vd)?,
                )?;
                source.add_linear_with(
                    &format!("{g}.attn_gate.weight"),
                    &format!("{p}.in_proj_z"),
                    hidden,
                    inner,
                    Exl3Layout::Identity,
                    head_layout(0, vd)?,
                )?;
                source.add_linear_with(
                    &format!("{g}.ssm_out.weight"),
                    &format!("{p}.out_proj"),
                    inner,
                    hidden,
                    head_layout(0, vd)?,
                    Exl3Layout::Identity,
                )?;
                let scalar = DenseOptions {
                    f32: true,
                    layout: head_layout(0, 1)?,
                    ..Default::default()
                };
                source.add_dense_with(
                    &format!("{g}.ssm_a"),
                    &format!("{p}.A_log"),
                    &[nv],
                    None,
                    DenseOptions {
                        op: DenseOp::NegativeExp,
                        ..scalar
                    },
                )?;
                source.add_dense_with(
                    &format!("{g}.ssm_dt.bias"),
                    &format!("{p}.dt_bias"),
                    &[nv],
                    None,
                    scalar,
                )?;
                source.add_dense_with(
                    &format!("{g}.ssm_norm.weight"),
                    &format!("{p}.norm.weight"),
                    &[vd],
                    None,
                    DenseOptions {
                        f32: true,
                        ..Default::default()
                    },
                )?;
                for (hf, dest) in [("in_proj_a", "ssm_alpha"), ("in_proj_b", "ssm_beta")] {
                    source.add_dense_with(
                        &format!("{g}.{dest}.weight"),
                        &format!("{p}.{hf}.weight"),
                        &[hidden, nv],
                        None,
                        DenseOptions {
                            layout: head_layout(0, hidden)?,
                            ..Default::default()
                        },
                    )?;
                }
                source.add_dense_with(
                    &format!("{g}.ssm_conv1d.weight"),
                    &format!("{p}.conv1d.weight"),
                    &[conv, qkv],
                    Some(&[qkv, 1, conv]),
                    DenseOptions {
                        f32: true,
                        layout: head_layout(
                            prefix
                                .checked_mul(conv)
                                .context("exl3: conv prefix overflow")?,
                            vd.checked_mul(conv)
                                .context("exl3: conv head width overflow")?,
                        )?,
                        ..Default::default()
                    },
                )?;
            }
            for (hf, dest, input, output) in [
                ("gate_proj", "ffn_gate", hidden, ff),
                ("up_proj", "ffn_up", hidden, ff),
                ("down_proj", "ffn_down", ff, hidden),
            ] {
                source.add_linear(
                    &format!("{g}.{dest}.weight"),
                    &format!("{p}.mlp.{hf}"),
                    input,
                    output,
                )?;
            }
        }
        source.add_linear("output.weight", "lm_head", hidden, vocab)?;
        Ok(source)
    }
}
