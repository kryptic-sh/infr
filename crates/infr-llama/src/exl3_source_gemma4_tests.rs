use super::*;

fn fixture(packed: bool) -> Fixture {
    let mut f = Fixture::new();
    f.tensors.clear();
    f.config = json!({
        "model_type": if packed { "gemma4" } else { "gemma4_unified" },
        "tie_word_embeddings": !packed, "eos_token_id":[1,106], "quantization_config":{"quant_method":"exl3"},
        "text_config":{
            "model_type":if packed {"gemma4_text"} else {"gemma4_unified_text"},
            "hidden_size":256,"intermediate_size":512,"num_hidden_layers":4,
            "num_attention_heads":2,"num_key_value_heads":2,"num_global_key_value_heads":1,
            "head_dim":128,"global_head_dim":256,"vocab_size":256,"max_position_embeddings":256,
            "hidden_size_per_layer_input":if packed {128} else {0},
            "vocab_size_per_layer_input":256,"num_kv_shared_layers":if packed {2} else {0},
            "hidden_activation":"gelu_pytorch_tanh","attention_bias":false,
            "attention_k_eq_v":true,"enable_moe_block":false,"use_double_wide_mlp":false,
            "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
            "sliding_window":32,"rms_norm_eps":1e-6,"final_logit_softcapping":30.0,
            "rope_parameters":{"full_attention":{"rope_type":"proportional","partial_rotary_factor":0.25,"rope_theta":1e6},"sliding_attention":{"rope_type":"default","rope_theta":1e4}},
            "bos_token_id":2,"eos_token_id":1,"pad_token_id":0
        }
    });
    embedding(&mut f, "model.language_model.embed_tokens", 256, packed);
    dense(&mut f, "model.language_model.norm.weight", 256, 0.75);
    linear(&mut f, "lm_head", 256, 256, 6, packed);
    if packed {
        embedding(
            &mut f,
            "model.language_model.embed_tokens_per_layer",
            512,
            true,
        );
        linear(
            &mut f,
            "model.language_model.per_layer_model_projection",
            256,
            512,
            4,
            true,
        );
        dense(
            &mut f,
            "model.language_model.per_layer_projection_norm.weight",
            128,
            0.875,
        );
    }
    for layer in 0..4 {
        let p = format!("model.language_model.layers.{layer}");
        let full = layer % 2 == 1;
        let (hd, nk) = if full { (256, 1) } else { (128, 2) };
        let bits = if packed { 4 } else { 2 };
        for (hf, k, n) in [
            ("self_attn.q_proj", 256, 2 * hd),
            ("self_attn.o_proj", 2 * hd, 256),
            ("mlp.gate_proj", 256, 512),
            ("mlp.up_proj", 256, 512),
            ("mlp.down_proj", 512, 256),
        ] {
            linear(&mut f, &format!("{p}.{hf}"), k, n, bits, packed);
        }
        if !packed || layer < 2 {
            linear(
                &mut f,
                &format!("{p}.self_attn.k_proj"),
                256,
                nk * hd,
                bits,
                packed,
            );
            if !full {
                linear(
                    &mut f,
                    &format!("{p}.self_attn.v_proj"),
                    256,
                    nk * hd,
                    bits,
                    packed,
                );
            }
            dense(&mut f, &format!("{p}.self_attn.k_norm.weight"), hd, 0.875);
        }
        for name in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            dense(&mut f, &format!("{p}.{name}"), 256, 0.75);
        }
        dense(&mut f, &format!("{p}.self_attn.q_norm.weight"), hd, 0.875);
        dense(&mut f, &format!("{p}.layer_scalar"), 1, 1.0);
        if packed {
            linear(
                &mut f,
                &format!("{p}.per_layer_input_gate"),
                256,
                128,
                4,
                true,
            );
            linear(
                &mut f,
                &format!("{p}.per_layer_projection"),
                128,
                256,
                4,
                true,
            );
            dense(
                &mut f,
                &format!("{p}.post_per_layer_input_norm.weight"),
                256,
                0.75,
            );
        }
    }
    f
}

fn dense(f: &mut Fixture, name: &str, width: usize, value: f32) {
    f.tensors.push((
        name.into(),
        "BF16".into(),
        vec![width],
        half::bf16::from_f32(value).to_le_bytes().repeat(width),
    ));
}

fn linear(f: &mut Fixture, prefix: &str, k: usize, n: usize, bits: usize, mul1: bool) {
    f.linear(prefix, k, n, bits);
    f.tensors.push((
        format!("{prefix}.{}", if mul1 { "mul1" } else { "mcg" }),
        "I32".into(),
        vec![],
        (if mul1 { 0x83dcd12d_u32 } else { 0xcbac1fed_u32 })
            .to_le_bytes()
            .to_vec(),
    ));
    for t in f
        .tensors
        .iter_mut()
        .filter(|t| t.0 == format!("{prefix}.suh") || t.0 == format!("{prefix}.svh"))
    {
        t.3 = half::f16::from_f32(0.015625).to_le_bytes().repeat(t.2[0]);
    }
}

fn embedding(f: &mut Fixture, prefix: &str, width: usize, packed: bool) {
    if packed {
        let geometry = Exl3EmbeddingSpec {
            width: width as u32,
            vocab: 256,
            bits: 4,
        };
        let mut bytes = vec![0; geometry.trellis_bytes().unwrap()];
        for row in 0..256 {
            for group in 0..width.div_ceil(256) {
                let at = row * geometry.row_bytes().unwrap() + group * 130;
                bytes[at..at + 2].copy_from_slice(&half::f16::from_f32(0.005).to_le_bytes());
                for (i, v) in bytes[at + 2..at + 130].iter_mut().enumerate() {
                    *v = (i * 97 + row * 31 + group * 7) as u8;
                }
            }
        }
        f.tensors.push((
            format!("{prefix}.trellis"),
            "I16".into(),
            vec![256, geometry.row_bytes().unwrap() / 2],
            bytes,
        ));
        f.dense(&format!("{prefix}.signs"), &[width.div_ceil(256), 256], 1.0);
    } else {
        let bytes = (0..width * 256)
            .flat_map(|i| {
                half::bf16::from_f32(((i * 29 % 257) as f32 * 0.041).sin() * 0.01).to_le_bytes()
            })
            .collect();
        f.tensors.push((
            format!("{prefix}.weight"),
            "BF16".into(),
            vec![256, width],
            bytes,
        ));
    }
}

#[test]
fn maps_gemma4_variants_norms_head_sizes_sharing_and_proportional_rope() {
    for packed in [false, true] {
        let f = fixture(packed);
        f.open().unwrap();
        let source = Exl3Source::open_native(&f.dir.path().join("model.safetensors")).unwrap();
        let cfg = crate::Config::from_gguf(&source).unwrap();
        assert!(cfg.gemma4);
        assert!(cfg.moe.is_none());
        assert_eq!(cfg.eos_ids, [1, 106]);
        let resident: u64 = source
            .tensors()
            .iter()
            .filter(|t| !source.embeddings.contains_key(&t.name))
            .map(|t| ((t.nbytes + 3) & !3) as u64)
            .sum();
        let all: u64 = source
            .tensors()
            .iter()
            .map(|t| ((t.nbytes + 3) & !3) as u64)
            .sum();
        assert_eq!(crate::weights::weight_footprint(&source).total(), resident);
        assert!(all > resident);
        assert_eq!(
            (
                cfg.n_embd,
                cfg.n_layer,
                cfg.n_head,
                cfg.head_dim,
                cfg.head_dim_swa,
                cfg.n_kv,
                cfg.n_kv_swa,
                cfg.rope_dim
            ),
            (256, 4, 2, 256, 128, 1, 2, 256)
        );
        assert_eq!(cfg.n_embd_per_layer, if packed { 128 } else { 0 });
        assert_eq!(cfg.n_layer_kv_from_start, if packed { 2 } else { 4 });
        assert_eq!(
            source.native_spec("blk.1.attn_q.weight").unwrap().out_f,
            512
        );
        assert_eq!(
            source
                .native_spec("blk.0.attn_q.weight")
                .unwrap()
                .output_layout,
            Exl3Layout::Identity
        );
        assert_eq!(
            source.native_spec("output.weight").unwrap().codebook,
            if packed {
                Exl3Codebook::Mul1
            } else {
                Exl3Codebook::Mcg
            }
        );
        assert!(source.native_spec("blk.1.attn_v.weight").is_none());
        assert_eq!(source.native_spec("blk.3.attn_k.weight").is_some(), !packed);
        assert_eq!(
            f32::from_le_bytes(
                source.tensor_bytes("output_norm.weight").unwrap()[..4]
                    .try_into()
                    .unwrap()
            ),
            0.75
        );
        let frequencies: Vec<_> = source
            .tensor_bytes("rope_freqs.weight")
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| f32::from_le_bytes(*p))
            .collect();
        assert_eq!(&frequencies[..32], &[1.0; 32]);
        assert!(frequencies[32..].iter().all(|&f| f == 1e30));
        let rows = source
            .tensor_rows("token_embd.weight", &[7, 3, 7])
            .unwrap()
            .unwrap();
        assert_eq!(&rows[..256], &rows[512..]);
        assert_ne!(&rows[..256], &rows[256..512]);
        assert!(source.tensor_bytes("token_embd.weight").is_err());
        if packed {
            assert_eq!(cfg.kv_src_layer(2), 0);
            assert_eq!(cfg.kv_src_layer(3), 1);
        }
    }
}

#[test]
fn gemma4_native_logits_match_folded_and_causal_prefixes_for_both_embedding_formats() {
    for packed in [false, true] {
        let f = fixture(packed);
        f.open().unwrap();
        let path = f.dir.path().join("model.safetensors");
        let ec = std::sync::Arc::new(crate::EngineConfig::default());
        let native = crate::SeamModel::load_with(&path, None, ec.clone()).unwrap();
        let folded = crate::SeamModel::from_source(
            Box::new(Exl3Source::open(&path).unwrap()),
            true,
            &path,
            None,
            ec,
        )
        .unwrap();
        let tokens = [3, 1, 4, 7, 2];
        let (all, _) = native.verify_logits_and_h_cpu(&tokens).unwrap();
        let (reference, _) = folded.verify_logits_and_h_cpu(&tokens).unwrap();
        let scale = reference.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
        for (i, (a, b)) in all.iter().zip(&reference).enumerate() {
            assert!(
                (a - b).abs() <= scale * 0.01,
                "packed={packed}, logit={i}: {a} != {b}"
            );
        }
        for end in 1..=tokens.len() {
            let (prefix, _) = native.verify_logits_and_h_cpu(&tokens[..end]).unwrap();
            for j in 0..256 {
                assert!(
                    (all[(end - 1) * 256 + j] - prefix[(end - 1) * 256 + j]).abs() < 2e-5,
                    "packed={packed} prefix={end}, j={j}"
                );
            }
        }
        let changed = [3, 1, 4, 7, 8];
        let (suffix, _) = native.verify_logits_and_h_cpu(&changed).unwrap();
        assert_eq!(&all[..4 * 256], &suffix[..4 * 256]);
        assert_ne!(&all[4 * 256..], &suffix[4 * 256..]);
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
}

#[test]
fn rejects_unsupported_gemma4_configs_and_embedding_encodings() {
    for (key, value) in [
        ("num_global_key_value_heads", json!(0)),
        ("num_kv_shared_layers", json!(4)),
        ("hidden_activation", json!("silu")),
        ("enable_moe_block", json!(true)),
        ("num_experts", json!(2)),
        ("attention_bias", json!(true)),
        ("use_double_wide_mlp", json!(true)),
    ] {
        let mut f = fixture(true);
        f.config["text_config"][key] = value;
        assert!(f.open().is_err(), "{key}");
    }
    let mut f = fixture(true);
    f.config["eos_token_id"][1] = json!(256);
    assert!(f.open().is_err());
    let mut f = fixture(true);
    f.config["text_config"]["layer_types"][2] = json!("full_attention");
    assert!(f.open().is_err());
    let mut f = fixture(true);
    f.config["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] =
        json!(0.251);
    assert!(f.open().is_err());
    for (name, dtype, shape) in [
        ("model.language_model.embed_tokens.signs", "F16", vec![256]),
        (
            "model.language_model.embed_tokens.trellis",
            "I16",
            vec![256, 64],
        ),
        (
            "model.language_model.layers.1.self_attn.q_norm.weight",
            "BF16",
            vec![128],
        ),
    ] {
        let mut f = fixture(true);
        let t = f.tensors.iter_mut().find(|t| t.0 == name).unwrap();
        t.1 = dtype.into();
        t.2 = shape;
        t.3.resize(t.2.iter().product::<usize>() * 2, 0);
        assert!(f.open().is_err(), "{name}");
    }
    let mut f = fixture(true);
    let t = f
        .tensors
        .iter_mut()
        .find(|t| t.0 == "model.language_model.embed_tokens.signs")
        .unwrap();
    t.3[..2].copy_from_slice(&half::f16::NAN.to_le_bytes());
    assert!(f.open().is_err());
    let mut f = fixture(false);
    let t = f
        .tensors
        .iter_mut()
        .find(|t| t.0 == "model.language_model.embed_tokens.weight")
        .unwrap();
    t.3[7 * 256 * 2..7 * 256 * 2 + 2].copy_from_slice(&half::bf16::INFINITY.to_le_bytes());
    let source = f.open().unwrap();
    assert!(source.tensor_rows("token_embd.weight", &[7]).is_err());
}
