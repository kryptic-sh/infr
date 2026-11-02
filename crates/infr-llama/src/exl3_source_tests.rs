use super::*;
use serde_json::{json, Value};

#[path = "exl3_source_gemma4_tests.rs"]
mod gemma4;
#[path = "exl3_source_qwen35_tests.rs"]
mod qwen35;

struct Fixture {
    dir: tempfile::TempDir,
    config: Value,
    tensors: Vec<(String, String, Vec<usize>, Vec<u8>)>,
}

impl Fixture {
    fn new() -> Self {
        let mut f = Self {
            dir: tempfile::tempdir().unwrap(),
            config: json!({
                "model_type":"qwen2", "quantization_config":{"quant_method":"exl3"},
                "tie_word_embeddings":false, "hidden_act":"silu", "num_hidden_layers":1,
                "hidden_size":128, "intermediate_size":256, "num_attention_heads":2,
                "num_key_value_heads":2, "vocab_size":256, "max_position_embeddings":512,
                "rms_norm_eps":1e-6, "rope_theta":1e6, "eos_token_id":255, "bos_token_id":254,
            }),
            tensors: Vec::new(),
        };
        for (name, shape) in [
            ("model.embed_tokens.weight", vec![256, 128]),
            ("model.norm.weight", vec![128]),
            ("model.layers.0.input_layernorm.weight", vec![128]),
            ("model.layers.0.post_attention_layernorm.weight", vec![128]),
        ] {
            f.dense(name, &shape, 0.125);
        }
        for (proj, k, n, bits) in [
            ("self_attn.q_proj", 128, 128, 4),
            ("self_attn.k_proj", 128, 128, 4),
            ("self_attn.v_proj", 128, 128, 4),
            ("self_attn.o_proj", 128, 128, 4),
            ("mlp.gate_proj", 128, 256, 3),
            ("mlp.up_proj", 128, 256, 3),
            ("mlp.down_proj", 256, 128, 3),
        ] {
            let prefix = format!("model.layers.0.{proj}");
            f.linear(&prefix, k, n, bits);
            if proj.ends_with("q_proj") || proj.ends_with("k_proj") || proj.ends_with("v_proj") {
                f.dense(&format!("{prefix}.bias"), &[n], 0.25);
            }
        }
        f.linear("lm_head", 128, 256, 6);
        f
    }

    fn dense(&mut self, name: &str, shape: &[usize], value: f32) {
        let bytes = half::f16::from_f32(value)
            .to_le_bytes()
            .repeat(shape.iter().product());
        self.tensors
            .push((name.into(), "F16".into(), shape.into(), bytes));
    }

    fn linear(&mut self, prefix: &str, k: usize, n: usize, bits: usize) {
        let shape = vec![k / 16, n / 16, 16 * bits];
        let bytes: Vec<_> = (0..shape.iter().product::<usize>())
            .flat_map(|i| (i as u16).wrapping_mul(379).wrapping_add(31).to_le_bytes())
            .collect();
        self.tensors
            .push((format!("{prefix}.trellis"), "I16".into(), shape, bytes));
        self.dense(&format!("{prefix}.suh"), &[k], 0.125);
        self.dense(&format!("{prefix}.svh"), &[n], 0.25);
    }

    fn open(&self) -> Result<Exl3Source> {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::<u8>::new();
        for (name, dt, shape, bytes) in &self.tensors {
            let start = payload.len();
            payload.extend_from_slice(bytes);
            header.insert(
                name.clone(),
                json!({"dtype":dt,"shape":shape,"data_offsets":[start,payload.len()]}),
            );
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut file = (h.len() as u64).to_le_bytes().to_vec();
        file.extend(h);
        file.extend(payload);
        std::fs::write(self.dir.path().join("model.safetensors"), file).unwrap();
        std::fs::write(self.dir.path().join("config.json"), self.config.to_string()).unwrap();
        std::fs::write(
            self.dir.path().join("tokenizer_config.json"),
            json!({"chat_template":"{{ messages }}"}).to_string(),
        )
        .unwrap();
        std::fs::write(
            self.dir.path().join("tokenizer.json"),
            json!({
                "model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}
            })
            .to_string(),
        )
        .unwrap();
        Exl3Source::open(&self.dir.path().join("model.safetensors"))
    }
}

#[test]
fn row_only_embeddings_feed_scaled_host_inputs_without_a_table_cache() {
    let mut fixture = Fixture::new();
    let geometry = Exl3EmbeddingSpec {
        width: 256,
        vocab: 2,
        bits: 4,
    };
    let mut packed = vec![0; geometry.trellis_bytes().unwrap()];
    for row in 0..2 {
        packed[row * 130..row * 130 + 2].copy_from_slice(&half::f16::ONE.to_le_bytes());
    }
    for prefix in ["row_emb", "row_ple"] {
        fixture.tensors.push((
            format!("{prefix}.trellis"),
            "I16".into(),
            vec![2, 65],
            packed.clone(),
        ));
        fixture.dense(&format!("{prefix}.signs"), &[1, 256], 1.0);
    }
    let mut source = fixture.open().unwrap();
    for (name, prefix) in [
        ("token_embd.weight", "row_emb"),
        ("per_layer_token_embd.weight", "row_ple"),
    ] {
        source.embeddings.insert(
            name.into(),
            EmbeddingSpec::Packed {
                hf_prefix: prefix.into(),
                geometry,
            },
        );
    }
    source.metadata.kv.insert(
        "infr.exl3.embedding_fp16_scale".into(),
        MetaValue::Bool(true),
    );
    let cache = OnceLock::new();
    let handle = crate::seam::TokenEmbd::new(&cache, &source);
    let rows = handle.rows(&[1, 0, 1], 256, (2560f32).sqrt()).unwrap();
    let scaled = half::f16::from_f32(-55.25 * 50.5).to_f32();
    for row in rows.as_chunks::<256>().0 {
        assert_eq!(row[0], scaled);
        assert!(row[1..].iter().all(|&v| v == 0.0));
    }
    assert!(cache.get().is_none());
    assert!(handle.get().is_err());
    assert!(handle.rows(&[2], 256, 1.0).is_err());
    assert!(source.tensor_bytes("token_embd.weight").is_err());
    let ple = crate::util::PerLayerEmbd {
        npl: 128,
        n_layer: 2,
        tok_embd_dtype: DType::I16,
        tok_embd_row_bytes: 130,
    };
    let rows = crate::seam::e2b_ipl_rows(&source, &ple, &[1, 0]).unwrap();
    let scale = 128f32.sqrt();
    assert_eq!(rows[0], half::f16::from_f32(-55.25 * scale).to_f32());
    assert_eq!(rows[256], rows[0]);
    let mut main = [0.05078125f32];
    let mut per_layer = main;
    crate::seam::scale_embedding_rows(&source, &mut main, scale, true).unwrap();
    crate::seam::scale_embedding_rows(&source, &mut per_layer, scale, false).unwrap();
    assert_eq!(main[0], 0.57421875);
    assert_eq!(per_layer[0], 0.57470703125_f64 as f32);
}

#[test]
fn native_source_exposes_only_compressed_projection_bytes_and_scales() {
    let fixture = Fixture::new();
    fixture.open().unwrap();
    let source = Exl3Source::open_native(&fixture.dir.path().join("model.safetensors")).unwrap();
    assert!(source.is_native());
    let name = "blk.0.attn_q.weight";
    let spec = source.native_spec(name).unwrap();
    let info = source
        .tensors()
        .iter()
        .find(|info| info.name == name)
        .unwrap();
    assert_eq!(info.dtype, DType::I16);
    assert_eq!(info.shape, [8, 8, 64]);
    assert_eq!(
        source.tensor_bytes(name).unwrap().len(),
        spec.trellis_bytes().unwrap()
    );
    assert_eq!(
        source.tensor_bytes(&format!("{name}.suh")).unwrap().len(),
        256
    );
    assert!(source.take_fold(name).is_err());
    assert!(source
        .linears
        .values()
        .all(|linear| linear.cache.get().is_none()));
    let expected: u64 = source
        .tensors()
        .iter()
        .map(|info| ((info.nbytes + 3) & !3) as u64)
        .sum();
    assert_eq!(crate::weights::weight_footprint(&source).total(), expected);
}

#[test]
fn native_and_folded_reference_logits_agree() {
    let mut fixture = Fixture::new();
    let embed = fixture
        .tensors
        .iter_mut()
        .find(|t| t.0 == "model.embed_tokens.weight")
        .unwrap();
    for (i, pair) in embed.3.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        pair.copy_from_slice(&half::f16::from_f32((i as f32 * 0.11).cos() * 0.1).to_le_bytes());
    }
    for t in fixture
        .tensors
        .iter_mut()
        .filter(|t| t.0.ends_with(".suh") || t.0.ends_with(".svh"))
    {
        for (i, pair) in t.3.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let v = (((i * 7) % 11) as f32 - 5.0) * 0.03;
            pair.copy_from_slice(
                &half::f16::from_f32(if v == 0.0 { 0.02 } else { v }).to_le_bytes(),
            );
        }
    }
    fixture.open().unwrap();
    let path = fixture.dir.path().join("model.safetensors");
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
    let (a, _) = native.verify_logits_and_h_cpu(&tokens).unwrap();
    let (b, _) = folded.verify_logits_and_h_cpu(&tokens).unwrap();
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(
            (x - y).abs() <= scale * 0.01,
            "logit {i}: native {x} vs folded {y}"
        );
    }
}

#[test]
fn native_source_rejects_nonfinite_scales_before_binding() {
    let mut fixture = Fixture::new();
    let scale = fixture
        .tensors
        .iter_mut()
        .find(|t| t.0 == "lm_head.svh")
        .unwrap();
    scale.3[..2].copy_from_slice(&half::f16::NAN.to_le_bytes());
    fixture.open().unwrap();
    assert!(Exl3Source::open_native(&fixture.dir.path().join("model.safetensors")).is_err());
}

#[test]
fn maps_qwen_metadata_names_shapes_and_biases() {
    let source = Fixture::new().open().unwrap();
    let cfg = crate::Config::from_gguf(&source).unwrap();
    assert_eq!(
        (cfg.n_layer, cfg.n_embd, cfg.n_ff, cfg.vocab),
        (1, 128, 256, 256)
    );
    assert_eq!(source.metadata().u64("qwen2.context_length"), Some(512));
    assert!(
        matches!(source.metadata().get("qwen2.rope.freq_base"), Some(MetaValue::F64(v)) if *v == 1e6)
    );
    assert!(
        matches!(source.metadata().get("qwen2.attention.layer_norm_rms_epsilon"), Some(MetaValue::F64(v)) if *v == 1e-6)
    );
    assert_eq!(source.chat_template(), Some("{{ messages }}"));
    let embed = source
        .tensors()
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .unwrap();
    assert_eq!(embed.shape, [128, 256]);
    assert_eq!(
        source.tensor_bytes("blk.0.attn_q.bias").unwrap().len(),
        128 * 4
    );
    for name in [
        "blk.0.attn_q.bias",
        "blk.0.attn_norm.weight",
        "output_norm.weight",
    ] {
        assert_eq!(
            source
                .tensors()
                .iter()
                .find(|t| t.name == name)
                .unwrap()
                .dtype,
            DType::F32
        );
    }
    assert_eq!(
        &source.tensor_bytes("blk.0.attn_q.bias").unwrap()[..4],
        &0.25f32.to_le_bytes()
    );
    assert_eq!(
        &source.tensor_bytes("token_embd.weight").unwrap()[..2],
        &half::f16::from_f32(0.125).to_le_bytes()
    );
    assert!(source.tensor_bytes("missing").is_err());
}

#[test]
fn loader_folds_do_not_populate_borrow_cache() {
    let source = Fixture::new().open().unwrap();
    let fresh = source.take_fold("blk.0.ffn_gate.weight").unwrap().unwrap();
    assert_eq!(fresh.len(), 128 * 256 * 2);
    assert!(source.linears.values().all(|s| s.cache.get().is_none()));
    assert_eq!(source.tensor_bytes("blk.0.ffn_gate.weight").unwrap(), fresh);
    assert!(source.linears["blk.0.ffn_gate.weight"]
        .cache
        .get()
        .is_some());
    assert!(source.take_fold("output_norm.weight").unwrap().is_none());
}

#[test]
fn rejects_wrong_dtype_even_when_byte_width_matches() {
    for name in [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.suh",
        "model.layers.0.self_attn.q_proj.trellis",
    ] {
        let mut f = Fixture::new();
        f.tensors.iter_mut().find(|t| t.0 == name).unwrap().1 = "BF16".into();
        assert!(f.open().is_err(), "accepted {name} BF16");
    }
}

#[test]
fn rejects_inconsistent_shapes_and_fractional_rates() {
    for name in [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.suh",
        "model.layers.0.mlp.down_proj.trellis",
    ] {
        let mut f = Fixture::new();
        let tensor = f.tensors.iter_mut().find(|t| t.0 == name).unwrap();
        if name.ends_with("suh") {
            tensor.2 = vec![16, 8];
        } else {
            tensor.2.reverse();
        }
        assert!(f.open().is_err(), "accepted incompatible {name}");
    }
    let mut f = Fixture::new();
    let t = f
        .tensors
        .iter_mut()
        .find(|t| t.0 == "lm_head.trellis")
        .unwrap();
    t.2[2] += 8;
    t.3.resize(t.2.iter().product::<usize>() * 2, 0);
    assert!(f.open().is_err());
}

#[test]
fn rejects_unsupported_configs_and_codebooks() {
    for (key, value) in [
        ("num_attention_heads", json!(0)),
        ("num_key_value_heads", json!(3)),
        ("hidden_size", json!(129)),
        ("eos_token_id", json!(256)),
        ("rope_scaling", json!({"type":"linear","factor":2})),
        ("use_sliding_window", json!(true)),
        ("tie_word_embeddings", json!(true)),
        ("hidden_act", json!("gelu")),
        ("rms_norm_eps", json!(0)),
    ] {
        let mut f = Fixture::new();
        f.config[key] = value;
        assert!(f.open().is_err(), "accepted {key}");
    }
    for marker in ["mcg", "mul1"] {
        let mut f = Fixture::new();
        f.dense(&format!("lm_head.{marker}"), &[1], 1.0);
        assert!(f.open().is_err());
    }
}

#[test]
fn accepts_codebook_markers_and_rejects_unsupported_parameters() {
    for (marker, codebook, multiplier) in [
        ("mcg", Exl3Codebook::Mcg, 0xcbac1fed_u32),
        ("mul1", Exl3Codebook::Mul1, 0x83dcd12d_u32),
    ] {
        let mut fixture = Fixture::new();
        let cb0 = fixture
            .open()
            .unwrap()
            .take_fold("output.weight")
            .unwrap()
            .unwrap();
        fixture.tensors.push((
            format!("lm_head.{marker}"),
            "I32".into(),
            vec![],
            multiplier.to_le_bytes().to_vec(),
        ));
        let source = fixture.open().unwrap();
        let folded = source.take_fold("output.weight").unwrap().unwrap();
        assert_ne!(folded, cb0);
        let native =
            Exl3Source::open_native(&fixture.dir.path().join("model.safetensors")).unwrap();
        assert_eq!(
            native.native_spec("output.weight").unwrap().codebook,
            codebook
        );
        assert_eq!(
            native.native_spec("blk.0.attn_q.weight").unwrap().codebook,
            Exl3Codebook::Cb0
        );
        assert!(native
            .linears
            .values()
            .all(|linear| linear.cache.get().is_none()));
        for (dtype, shape, value) in [
            ("F32", vec![], multiplier),
            ("I32", vec![1], multiplier),
            ("I32", vec![], 1),
        ] {
            let mut invalid = Fixture::new();
            invalid.tensors.push((
                format!("lm_head.{marker}"),
                dtype.into(),
                shape,
                value.to_le_bytes().to_vec(),
            ));
            assert!(invalid.open().is_err());
        }
    }
    let mut conflicting = Fixture::new();
    for (marker, multiplier) in [("mcg", 0xcbac1fed_u32), ("mul1", 0x83dcd12d_u32)] {
        conflicting.tensors.push((
            format!("lm_head.{marker}"),
            "I32".into(),
            vec![],
            multiplier.to_le_bytes().to_vec(),
        ));
    }
    assert!(conflicting.open().is_err());
}

#[test]
fn cached_fold_errors_are_returned_without_panics() {
    let mut f = Fixture::new();
    let t = f.tensors.iter_mut().find(|t| t.0 == "lm_head.svh").unwrap();
    t.3[..2].copy_from_slice(&half::f16::NAN.to_le_bytes());
    let source = f.open().unwrap();
    assert!(source.take_fold("output.weight").is_err());
    assert!(source.tensor_bytes("output.weight").is_err());
    assert!(source.tensor_bytes("output.weight").is_err());
}

#[test]
fn synthetic_qwen_batched_logits_match_causal_prefix_forwards() {
    let mut fixture = Fixture::new();
    let embed = fixture
        .tensors
        .iter_mut()
        .find(|t| t.0 == "model.embed_tokens.weight")
        .unwrap();
    for (i, pair) in embed.3.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        pair.copy_from_slice(&half::f16::from_f32((i as f32 * 0.07).sin() * 0.1).to_le_bytes());
    }
    fixture.open().unwrap();
    let model = crate::SeamModel::load_with(
        &fixture.dir.path().join("model.safetensors"),
        None,
        std::sync::Arc::new(crate::EngineConfig::default()),
    )
    .unwrap();
    let tokens = [1, 2, 3, 4, 5];
    let (all, _) = model.verify_logits_and_h_cpu(&tokens).unwrap();
    let vocab = model.config().vocab;
    for i in 0..tokens.len() {
        let prefix = model.prefill_logits_cpu(&tokens[..=i]).unwrap();
        for (&a, &b) in all[i * vocab..(i + 1) * vocab].iter().zip(&prefix) {
            assert!(
                (a - b).abs() < 0.00002,
                "row {i}: batched {a} != causal {b}"
            );
        }
    }
    let (changed, _) = model.verify_logits_and_h_cpu(&[1, 2, 3, 19, 23]).unwrap();
    assert_eq!(&all[..3 * vocab], &changed[..3 * vocab]);
    assert_ne!(&all[4 * vocab..], &changed[4 * vocab..]);
    let source = model
        .weights()
        .as_any()
        .downcast_ref::<Exl3Source>()
        .unwrap();
    assert!(source.is_native());
    assert!(source
        .linears
        .values()
        .all(|linear| linear.cache.get().is_none()));
}

fn walsh_reference(x: &[f32]) -> Vec<f32> {
    x.as_chunks::<128>()
        .0
        .iter()
        .flat_map(|chunk| {
            (0usize..128)
                .map(|r| {
                    chunk
                        .iter()
                        .enumerate()
                        .map(|(c, &v)| {
                            if (r & c).count_ones().is_multiple_of(2) {
                                v
                            } else {
                                -v
                            }
                        })
                        .sum::<f32>()
                        / 128f32.sqrt()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn folded_linear_matches_unfused_hadamard_forward_for_every_integer_rate() {
    let (k, n) = (256, 256);
    for (codebook, bits) in [Exl3Codebook::Cb0, Exl3Codebook::Mcg]
        .into_iter()
        .flat_map(|codebook| (1..=8).map(move |bits| (codebook, bits)))
    {
        let packed: Vec<_> = (0..(k / 16) * (n / 16) * 16 * bits as usize)
            .map(|i| (i as u16).wrapping_mul(313).wrapping_add(117))
            .collect();
        let suh: Vec<_> = (0..k).map(|i| (i as f32 % 13.0 - 6.0) * 0.017).collect();
        let svh: Vec<_> = (0..n).map(|i| (i as f32 % 7.0 - 3.0) * 0.023).collect();
        let x: Vec<_> = (0..k).map(|i| (i as f32 * 0.23).sin()).collect();
        let d = infr_cpu::exl3::decode_matrix(Codebook::from(codebook), bits, &packed, k, n);
        let scaled: Vec<_> = x.iter().zip(&suh).map(|(a, b)| a * b).collect();
        let xh = walsh_reference(&scaled);
        let raw: Vec<_> = (0..n)
            .map(|j| (0..k).map(|i| xh[i] * d[i * n + j]).sum())
            .collect();
        let expected: Vec<_> = walsh_reference(&raw)
            .iter()
            .zip(&svh)
            .map(|(a, b)| a * b)
            .collect();
        let folded = fold_trellis(
            &packed,
            &suh,
            &svh,
            Exl3Spec {
                in_f: k as u32,
                out_f: n as u32,
                bits,
                codebook,
                input_layout: Exl3Layout::Identity,
                output_layout: Exl3Layout::Identity,
            },
        )
        .unwrap();
        for j in 0..n {
            let actual: f32 = (0..k)
                .map(|i| {
                    let offset = (j * k + i) * 2;
                    half::f16::from_le_bytes(folded[offset..offset + 2].try_into().unwrap())
                        .to_f32()
                        * x[i]
                })
                .sum();
            let tolerance = 0.0002 + expected[j].abs() * 0.002;
            assert!(
                (actual - expected[j]).abs() < tolerance,
                "codebook={codebook:?} K={bits} out={j}: {actual} != {}",
                expected[j]
            );
        }
    }
}
