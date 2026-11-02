use super::*;

fn raw_dense(
    f: &mut Fixture,
    name: &str,
    shape: &[usize],
    dtype: &str,
    value: impl Fn(usize) -> f32,
) {
    let bytes = (0..shape.iter().product())
        .flat_map(|i| {
            let v = value(i);
            match dtype {
                "F32" => v.to_le_bytes().to_vec(),
                "BF16" => half::bf16::from_f32(v).to_le_bytes().to_vec(),
                _ => half::f16::from_f32(v).to_le_bytes().to_vec(),
            }
        })
        .collect();
    f.tensors
        .push((name.into(), dtype.into(), shape.into(), bytes));
}

fn mcg(f: &mut Fixture, name: &str, k: usize, n: usize, bits: usize) {
    f.linear(name, k, n, bits);
    f.tensors.push((
        format!("{name}.mcg"),
        "I32".into(),
        vec![],
        0xcbac1fed_u32.to_le_bytes().to_vec(),
    ));
}

fn fixture() -> Fixture {
    let mut f = Fixture::new();
    f.tensors.clear();
    f.config = json!({
        "model_type":"qwen3_5", "tie_word_embeddings":false,
        "quantization_config":{"quant_method":"exl3", "codebook":"mcg"},
        "text_config":{
            "model_type":"qwen3_5_text", "hidden_size":256, "intermediate_size":384,
            "num_hidden_layers":4, "num_attention_heads":2, "num_key_value_heads":1, "head_dim":128,
            "vocab_size":256, "eos_token_id":255, "max_position_embeddings":128, "rms_norm_eps":1e-6,
            "hidden_act":"silu", "attention_bias":false, "attn_output_gate":true,
            "full_attention_interval":4, "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
            "linear_num_key_heads":3, "linear_num_value_heads":6, "linear_key_head_dim":128,
            "linear_value_head_dim":128, "linear_conv_kernel_dim":4, "mamba_ssm_dtype":"float32",
            "mtp_num_hidden_layers":1, "mlp_only_layers":[],
            "rope_parameters":{"rope_type":"default", "rope_theta":10000000,
                "partial_rotary_factor":0.25, "mrope_section":[5,5,6], "mrope_interleaved":true},
        }, "vision_config":{"hidden_size":32}
    });
    raw_dense(
        &mut f,
        "model.language_model.embed_tokens.weight",
        &[256, 256],
        "F16",
        |i| (i as f32 * 0.11).cos() * 0.1,
    );
    raw_dense(
        &mut f,
        "model.language_model.norm.weight",
        &[256],
        "BF16",
        |_| 0.125,
    );
    raw_dense(&mut f, "model.visual.unused.weight", &[4], "F32", |_| {
        f32::NAN
    });
    for l in 0..4 {
        let p = format!("model.language_model.layers.{l}");
        for name in ["input_layernorm", "post_attention_layernorm"] {
            raw_dense(
                &mut f,
                &format!("{p}.{name}.weight"),
                &[256],
                "BF16",
                |_| 0.125,
            );
        }
        if l == 3 {
            for (name, width) in [("q_proj", 512), ("k_proj", 128), ("v_proj", 128)] {
                mcg(&mut f, &format!("{p}.self_attn.{name}"), 256, width, 2);
            }
            mcg(&mut f, &format!("{p}.self_attn.o_proj"), 256, 256, 2);
            for name in ["q_norm", "k_norm"] {
                raw_dense(
                    &mut f,
                    &format!("{p}.self_attn.{name}.weight"),
                    &[128],
                    "BF16",
                    |_| 0.125,
                );
            }
        } else {
            let p = format!("{p}.linear_attn");
            mcg(&mut f, &format!("{p}.in_proj_qkv"), 256, 1536, 2);
            mcg(&mut f, &format!("{p}.in_proj_z"), 256, 768, 2);
            mcg(&mut f, &format!("{p}.out_proj"), 768, 256, 2);
            for name in ["in_proj_a", "in_proj_b"] {
                raw_dense(
                    &mut f,
                    &format!("{p}.{name}.weight"),
                    &[6, 256],
                    "F16",
                    |i| (i / 256) as f32 * 0.03125 + (i % 7) as f32 * 0.00025,
                );
            }
            raw_dense(&mut f, &format!("{p}.A_log"), &[6], "F32", |i| {
                i as f32 * 0.25
            });
            raw_dense(&mut f, &format!("{p}.dt_bias"), &[6], "BF16", |i| {
                i as f32 * 0.125
            });
            raw_dense(&mut f, &format!("{p}.norm.weight"), &[128], "F32", |_| 0.75);
            raw_dense(
                &mut f,
                &format!("{p}.conv1d.weight"),
                &[1536, 1, 4],
                "BF16",
                |i| 0.01 * (1 + i % 4) as f32 + 0.005 * (i / (128 * 4)) as f32,
            );
        }
        for (name, k, n) in [
            ("gate_proj", 256, 384),
            ("up_proj", 256, 384),
            ("down_proj", 384, 256),
        ] {
            mcg(&mut f, &format!("{p}.mlp.{name}"), k, n, 2);
        }
    }
    mcg(&mut f, "lm_head", 256, 256, 6);
    for tensor in f
        .tensors
        .iter_mut()
        .filter(|t| t.0.ends_with(".suh") || t.0.ends_with(".svh"))
    {
        for (i, pair) in tensor.3.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let v = (((i * 7) % 11) as f32 - 5.0) * 0.03;
            pair.copy_from_slice(
                &half::f16::from_f32(if v == 0.0 { 0.02 } else { v }).to_le_bytes(),
            );
        }
    }
    f
}

#[test]
fn maps_text_only_hybrid_metadata_and_native_layouts() {
    let f = fixture();
    f.open().unwrap();
    let source = Exl3Source::open_native(&f.dir.path().join("model.safetensors")).unwrap();
    let cfg = crate::Config::from_gguf(&source).unwrap();
    assert_eq!(
        (cfg.n_layer, cfg.n_embd, cfg.n_ff, cfg.vocab),
        (4, 256, 384, 256)
    );
    assert!(cfg.qwen35 && cfg.qk_norm && !cfg.qkv_bias && !cfg.permute_qk_neox);
    assert_eq!(
        (
            cfg.ssm_d_inner,
            cfg.ssm_n_group,
            cfg.ssm_d_state,
            cfg.ssm_dt_rank
        ),
        (768, 3, 128, 6)
    );
    assert_eq!(cfg.rope_sections, [5, 5, 6, 0]);
    assert!(!cfg.is_qwen35_attn_layer(0));
    assert!(cfg.is_qwen35_attn_layer(3));
    assert!(source
        .metadata()
        .get("qwen35.nextn_predict_layers")
        .is_none());
    assert!(source.tensors().iter().all(|t| !t.name.contains("visual")));
    let info = |name| source.tensors().iter().find(|t| t.name == name).unwrap();
    assert_eq!(info("blk.0.ssm_conv1d.weight").shape, [4, 1536]);
    assert_eq!(info("blk.0.ssm_alpha.weight").shape, [256, 6]);
    assert_eq!(info("blk.0.ssm_alpha.weight").dtype, DType::F16);
    assert_eq!(info("blk.0.ssm_conv1d.weight").dtype, DType::F32);
    let qkv = source.native_spec("blk.0.attn_qkv.weight").unwrap();
    assert_eq!(
        qkv.output_layout,
        Exl3Layout::GroupedValueHeads {
            prefix: 768,
            key_heads: 3,
            value_heads: 6,
            head_dim: 128
        }
    );
    assert_eq!(
        source
            .native_spec("blk.0.ssm_out.weight")
            .unwrap()
            .input_layout,
        Exl3Layout::GroupedValueHeads {
            prefix: 0,
            key_heads: 3,
            value_heads: 6,
            head_dim: 128
        }
    );
    assert_eq!(
        source
            .native_spec("blk.3.attn_q.weight")
            .unwrap()
            .output_layout,
        Exl3Layout::Identity
    );
    assert_eq!(
        source.native_spec("blk.3.attn_q.weight").unwrap().out_f,
        512
    );
}

#[test]
fn dense_preprocessing_and_unequal_value_heads_match_exporter() {
    let source = fixture().open().unwrap();
    let floats = |name| {
        source
            .tensor_bytes(name)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| f32::from_le_bytes(*p))
            .collect::<Vec<_>>()
    };
    assert_eq!(floats("output_norm.weight")[0], 1.125);
    assert_eq!(floats("blk.0.attn_norm.weight")[0], 1.125);
    assert_eq!(floats("blk.3.attn_q_norm.weight")[0], 1.125);
    assert_eq!(floats("blk.0.ssm_norm.weight")[0], 0.75);
    assert_eq!(floats("blk.0.ssm_a")[0], -1.0);
    let order = [0, 2, 4, 1, 3, 5];
    let a = floats("blk.0.ssm_a");
    let dt = floats("blk.0.ssm_dt.bias");
    let alpha = source
        .tensor_bytes("blk.0.ssm_alpha.weight")
        .unwrap()
        .as_chunks::<2>()
        .0;
    let beta = source
        .tensor_bytes("blk.0.ssm_beta.weight")
        .unwrap()
        .as_chunks::<2>()
        .0;
    let conv = floats("blk.0.ssm_conv1d.weight");
    for (dest, src) in order.into_iter().enumerate() {
        assert_eq!(a[dest], -(src as f32 * 0.25).exp());
        assert_eq!(dt[dest], src as f32 * 0.125);
        for col in [0, 1, 255] {
            let expected = half::f16::from_f32(
                src as f32 * 0.03125 + ((src * 256 + col) % 7) as f32 * 0.00025,
            );
            assert_eq!(half::f16::from_le_bytes(alpha[dest * 256 + col]), expected);
            assert_eq!(half::f16::from_le_bytes(beta[dest * 256 + col]), expected);
        }
        for channel in [0, 1, 127] {
            for tap in 0..4 {
                let expected =
                    half::bf16::from_f32(0.01 * (1 + tap) as f32 + 0.005 * (6 + src) as f32)
                        .to_f32();
                assert_eq!(conv[(768 + dest * 128 + channel) * 4 + tap], expected);
            }
        }
    }
    for i in 0..768 * 4 {
        let expected =
            half::bf16::from_f32(0.01 * (1 + i % 4) as f32 + 0.005 * (i / (128 * 4)) as f32)
                .to_f32();
        assert_eq!(conv[i], expected);
    }
}

#[test]
fn native_hybrid_logits_match_folded_reference_and_causal_prefixes() {
    let f = fixture();
    f.open().unwrap();
    let path = f.dir.path().join("model.safetensors");
    let ecfg = std::sync::Arc::new(crate::EngineConfig::default());
    let native = crate::SeamModel::load_with(&path, None, ecfg.clone()).unwrap();
    let folded = crate::SeamModel::from_source(
        Box::new(Exl3Source::open(&path).unwrap()),
        true,
        &path,
        None,
        ecfg,
    )
    .unwrap();
    let tokens = [3, 1, 4, 1, 5, 9, 2, 6, 5];
    let (all, _) = native.verify_logits_and_h_cpu(&tokens).unwrap();
    let (reference, _) = folded.verify_logits_and_h_cpu(&tokens).unwrap();
    let scale = reference.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
    for (i, (a, b)) in all.iter().zip(&reference).enumerate() {
        assert!(
            (a - b).abs() <= scale * 0.01,
            "logit {i}: native={a}, reference={b}"
        );
    }
    for end in 1..=tokens.len() {
        let (prefix, _) = native.verify_logits_and_h_cpu(&tokens[..end]).unwrap();
        for j in 0..256 {
            let index = (end - 1) * 256 + j;
            assert!(
                (all[index] - prefix[index]).abs() < 2e-5,
                "prefix={end}, logit={j}"
            );
        }
    }
    let mut changed = tokens;
    changed[8] = 8;
    let (suffix, _) = native.verify_logits_and_h_cpu(&changed).unwrap();
    assert_eq!(&all[..8 * 256], &suffix[..8 * 256]);
    assert_ne!(&all[8 * 256..], &suffix[8 * 256..]);
    let source = native
        .weights()
        .as_any()
        .downcast_ref::<Exl3Source>()
        .unwrap();
    assert!(source
        .linears
        .values()
        .all(|linear| linear.cache.get().is_none()));
}

#[test]
fn rejects_inconsistent_hybrid_config_shapes_and_nonfinite_parameters() {
    for (key, value) in [
        ("linear_num_value_heads", json!(5)),
        ("linear_value_head_dim", json!(64)),
        ("attn_output_gate", json!(false)),
        ("attention_bias", json!(true)),
        ("full_attention_interval", json!(2)),
    ] {
        let mut f = fixture();
        f.config["text_config"][key] = value;
        assert!(f.open().is_err(), "{key}");
    }
    for (name, dtype, shape) in [
        (
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            "BF16",
            vec![1536, 4],
        ),
        (
            "model.language_model.layers.3.self_attn.q_norm.weight",
            "I16",
            vec![128],
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            "BF16",
            vec![6, 256],
        ),
    ] {
        let mut f = fixture();
        let t = f.tensors.iter_mut().find(|t| t.0 == name).unwrap();
        t.1 = dtype.into();
        t.2 = shape;
        assert!(f.open().is_err());
    }
    for value in [f32::NAN, 1000.0] {
        let mut f = fixture();
        let t = f
            .tensors
            .iter_mut()
            .find(|t| t.0.ends_with(".A_log"))
            .unwrap();
        t.3[..4].copy_from_slice(&value.to_le_bytes());
        f.open().unwrap();
        assert!(Exl3Source::open_native(&f.dir.path().join("model.safetensors")).is_err());
    }
}

#[test]
fn reads_adjacent_jinja_when_tokenizer_config_omits_template() {
    let f = fixture();
    f.open().unwrap();
    std::fs::write(f.dir.path().join("tokenizer_config.json"), "{}").unwrap();
    std::fs::write(
        f.dir.path().join("chat_template.jinja"),
        "{{ messages[0].content }}",
    )
    .unwrap();
    let source = Exl3Source::open(&f.dir.path().join("model.safetensors")).unwrap();
    assert_eq!(source.chat_template(), Some("{{ messages[0].content }}"));
}
