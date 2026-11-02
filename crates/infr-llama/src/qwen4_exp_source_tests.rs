//! Network-free fixtures reproduce the tiny model's schema, not its random weights.
use super::*;
use crate::qwen4_exp::{LayerType, OutputGate, FORWARD_UNSUPPORTED};
use serde_json::{json, Value};
use std::io::Write;

fn config() -> Value {
    serde_json::from_str(include_str!("../tests/fixtures/qwen4_exp/config.json")).unwrap()
}
fn refuses<T>(result: Result<T>) {
    let e = match result {
        Ok(_) => panic!("unsupported inference must refuse"),
        Err(e) => e,
    };
    assert_eq!(e.to_string(), FORWARD_UNSUPPORTED);
}

fn manifest() -> Value {
    serde_json::from_str(include_str!("../tests/fixtures/qwen4_exp/tensors.json")).unwrap()
}

struct Fixture {
    dir: tempfile::TempDir,
    path: std::path::PathBuf,
}
impl Fixture {
    fn new(config: &Value, manifest: &Value) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (i, (name, spec)) in manifest.as_object().unwrap().iter().enumerate() {
            let numel: usize = spec["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .product();
            let start = data.len();
            if spec["dtype"] == "I64" {
                let values = spec["values"].as_array().unwrap();
                assert_eq!(values.len(), numel);
                for n in values {
                    data.extend_from_slice(&n.as_i64().unwrap().to_le_bytes());
                }
            } else {
                let value = (i + 1) as f32 / 256.0;
                let bytes = match spec["dtype"].as_str().unwrap() {
                    "BF16" => half::bf16::from_f32(value).to_le_bytes().to_vec(),
                    "F16" => half::f16::from_f32(value).to_le_bytes().to_vec(),
                    "F32" => value.to_le_bytes().to_vec(),
                    "I32" => 0i32.to_le_bytes().to_vec(),
                    _ => panic!("unsupported synthetic dtype"),
                };
                for _ in 0..numel {
                    data.extend_from_slice(&bytes);
                }
            }
            header.insert(name.clone(), json!({ "dtype":spec["dtype"], "shape":spec["shape"], "data_offsets":[start,data.len()] }));
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(&data).unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(config).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            br#"{"chat_template":"{{ messages }}"}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), serde_json::to_vec(&json!({
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null, "pre_tokenizer":null, "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel", "vocab":{"[UNK]":0,"a":1,"<|im_end|>":2}, "unk_token":"[UNK]"}
        })).unwrap()).unwrap();
        Self { dir, path }
    }
    fn load(&self) -> Result<Qwen4ExpSource> {
        Qwen4ExpSource::open(&self.path)
    }
    fn error(&self) -> String {
        match self.load() {
            Ok(_) => panic!("fixture should fail"),
            Err(e) => format!("{e:#}"),
        }
    }
}

#[test]
fn tiny_config_captures_all_new_semantics() {
    let c = Cfg::from_hf(&config()).unwrap();
    assert_eq!(
        (c.hidden_size, c.num_hidden_layers, c.hc_count, c.hc_lowrank),
        (64, 4, 4, 8)
    );
    assert_eq!(
        c.layer_types,
        [
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::IndexedAttention
        ]
    );
    assert_eq!(c.output_gate, OutputGate::Sigmoid);
    assert_eq!((c.rope_dim, c.rope_sections), (8, [2, 1, 1, 0]));
    assert!(c.mrope_interleaved);
    assert_eq!(c.mamba_ssm_dtype, DType::F32);
    assert_eq!(c.number_of_conv_states, 3);
    assert_eq!(c.pad_token_id, Some(0));
    let idx = c.indexer.as_ref().unwrap();
    assert_eq!(
        (idx.n_heads, idx.head_dim, idx.budget, idx.compress_ratio),
        (2, 8, 8, 4)
    );
    let ple = c.ple.as_ref().unwrap();
    assert_eq!(ple.layer_ids, [2]);
    assert_eq!(ple.seed, 20260907);
    let shared = c.shared_config();
    assert!(shared.qwen4_exp.is_some());
    assert!(shared.qwen35 && shared.qk_norm && shared.attn_out_gate);
    assert_eq!(
        (shared.n_layer, shared.n_embd, shared.ssm_d_inner),
        (4, 64, 96)
    );
    let moe = shared.moe.as_ref().unwrap();
    assert_eq!((moe.n_expert, moe.n_used, moe.n_ff_exp), (4, 2, 32));
    assert!(shared.is_qwen35_attn_layer(3));
    assert!(!shared.is_qwen35_attn_layer(2));
    assert!(!crate::arch::ALL.contains(&crate::arch::QWEN4_EXP));
}

#[test]
fn tiny_source_is_zero_copy_and_has_no_fake_norms() {
    let f = Fixture::new(&config(), &manifest());
    let s = f.load().unwrap();
    assert_eq!(s.infos.len(), 149);
    assert_eq!(s._ple_buffers.len(), 3);
    assert_eq!(s.chat_template(), Some("{{ messages }}"));
    assert!(s.infos.iter().all(|t| !t.name.contains("visual")));
    assert!(!s.has("model.language_model.norm.weight"));
    assert!(s.tensor_bytes("output_norm.weight").is_err());
    for t in &s.infos {
        let hf = &s.names[&t.name];
        let bytes = s.tensor_bytes(&t.name).unwrap();
        let raw = s.st.tensor_bytes(hf).unwrap();
        assert_eq!(bytes.as_ptr(), raw.as_ptr());
        assert_eq!(bytes.len(), t.nbytes);
        let original = s.st.tensors().iter().find(|x| &x.name == hf).unwrap();
        assert_eq!(
            t.shape,
            original.shape.iter().rev().copied().collect::<Vec<_>>()
        );
        assert_eq!(t.dtype, original.dtype);
    }
    let p = "model.language_model.layers.1.ple.ple_embedding";
    assert_eq!(
        s._ple_buffers[&format!("{p}.layer_multipliers")],
        [29891111472640045, 20407171196159859, 31567350566385267]
    );
    assert!(s.tensor_bytes(&format!("{p}.layer_multipliers")).is_err());
    assert_eq!(crate::Config::from_gguf(&s).unwrap().eos_ids, [2]);
}

#[test]
fn load_with_routes_hf_and_refuses_before_cpu_or_gpu_work() {
    let f = Fixture::new(&config(), &manifest());
    let model = crate::SeamModel::load_with(
        &f.path,
        None,
        std::sync::Arc::new(crate::EngineConfig::default()),
    )
    .unwrap();
    assert_eq!(model.path(), f.path);
    assert_eq!(
        model.weights().metadata().str("general.architecture"),
        Some(crate::arch::QWEN4_EXP)
    );
    assert_eq!(model.weights().chat_template(), Some("{{ messages }}"));
    let mut called = false;
    let e = model
        .generate_cpu_ids(&[1, 2], 1, |_| called = true)
        .unwrap_err();
    assert_eq!(e.to_string(), FORWARD_UNSUPPORTED);
    assert!(!called);
    // No Vulkan enumeration, backend creation, placement or tensor binding is allowed.
    for result in [
        model.vulkan_session_on(Some(usize::MAX), 8),
        model.vulkan_session_default_on(Some(usize::MAX)),
        model.vulkan_session_frac_on(Some(usize::MAX), 0.5),
    ] {
        let err = match result {
            Ok(_) => panic!("inference must refuse"),
            Err(e) => e,
        };
        assert_eq!(err.to_string(), FORWARD_UNSUPPORTED);
    }
    refuses(model.prefill_logits_vulkan(&[1, 2]));
    refuses(model.generate_dense_vulkan("a", 1));
    refuses(model.generate_vulkan_ids(&[1, 2], 1, |_| panic!("callback")));
    refuses(model.generate_dense_vulkan_pipeline(&[], "a", 1));
    refuses(model.generate_pipeline_ids(&[], &[1, 2], 1, |_| panic!("callback")));
    refuses(model.generate_dense_vulkan_tp(&[], "a", 1));
    refuses(model.generate_tp_ids(&[], &[1, 2], 1, |_| panic!("callback")));
    refuses(model.generate_moe_vulkan_ep(&[], "a", 1));
    refuses(model.generate_ep_ids(&[], &[1, 2], 1, |_| panic!("callback")));
    refuses(model.diffusion_gemma_vulkan_session(8));
    refuses(model.bench_vulkan(1, 1, 1, None, 1));
    refuses(model.perplexity(&[1, 2], 8, Some(4), true, |_, _, _| panic!("callback")));
    #[cfg(target_os = "macos")]
    {
        refuses(model.metal_session(8));
        refuses(model.diffusion_gemma_metal_session(8));
        refuses(model.generate_metal("a", 1, None, |_| panic!("callback")));
    }
    // A supplied tokenizer path still wins over adjacent HF tokenizer.json.
    let override_path = f.dir.path().join("override.json");
    std::fs::copy(f.dir.path().join("tokenizer.json"), &override_path).unwrap();
    std::fs::remove_file(f.dir.path().join("tokenizer.json")).unwrap();
    assert!(crate::SeamModel::load_with(
        &f.path,
        Some(&override_path),
        std::sync::Arc::new(crate::EngineConfig::default())
    )
    .is_ok());
    let e = match crate::SeamModel::load_with(
        &f.path,
        None,
        std::sync::Arc::new(crate::EngineConfig::default()),
    ) {
        Ok(_) => panic!("missing tokenizer"),
        Err(e) => e,
    };
    assert!(e.to_string().contains("adjacent HF tokenizer.json"));
}

#[test]
fn config_aliases_nonuniform_layers_tied_head_and_float_dtypes() {
    for alias in ["indexed_attention", "full_attention"] {
        let mut c = config()["text_config"].clone();
        c["layer_types"][3] = json!(alias);
        c["tie_word_embeddings"] = json!(true);
        c["eos_token_id"] = json!([2, 3]);
        let mut m = manifest();
        m.as_object_mut().unwrap().remove("lm_head.weight");
        let m: serde_json::Map<_, _> = m
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, val)| {
                let mut spec = val.clone();
                if spec["dtype"] != "I64" {
                    spec["dtype"] = json!("F32");
                }
                (key.replace("model.language_model.", "model."), spec)
            })
            .collect();
        let f = Fixture::new(&c, &Value::Object(m));
        let s = f.load().unwrap();
        assert!(!s.names.contains_key("output.weight"));
        assert_eq!(s.cfg.eos_ids, [2, 3]);
    }
    let mut c = config();
    c["text_config"]["layer_types"] = json!([
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "linear_attention"
    ]);
    c["text_config"]["ple_layer_ids"] = json!([3]);
    let shared = Cfg::from_hf(&c).unwrap().shared_config();
    assert!(shared.is_qwen35_attn_layer(1));
    assert!(!shared.is_qwen35_attn_layer(3));
    c["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("layer_types");
    let c = Cfg::from_hf(&c).unwrap();
    assert_eq!(c.layer_types[3], LayerType::IndexedAttention);
}

#[test]
fn packed_experts_and_consolidated_ple_are_retained() {
    let mut m = manifest();
    let map = m.as_object_mut().unwrap();
    for l in 0..4 {
        let p = format!("model.language_model.layers.{l}.mlp.experts");
        for e in 0..4 {
            for suffix in ["gate_proj", "up_proj", "down_proj"] {
                map.remove(&format!("{p}.{e}.{suffix}.weight"));
            }
        }
        map.insert(
            format!("{p}.gate_up_proj"),
            json!({"dtype":"F16","shape":[4,64,64]}),
        );
        map.insert(
            format!("{p}.down_proj"),
            json!({"dtype":"F16","shape":[4,64,32]}),
        );
    }
    let p = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
    for part in 0..2 {
        map.remove(&format!("{p}.shard_{part}.weight"));
    }
    map.insert(
        format!("{p}.weight"),
        json!({"dtype":"BF16","shape":[296,8]}),
    );
    let f = Fixture::new(&config(), &m);
    let s = f.load().unwrap();
    assert_eq!(s.infos.len(), 108);
    assert_eq!(
        s.infos
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_up_exps.weight")
            .unwrap()
            .shape,
        [64, 64, 4]
    );
    assert_eq!(
        s.infos
            .iter()
            .find(|t| t.name == "blk.1.ple_embd.weight")
            .unwrap()
            .shape,
        [8, 296]
    );
}

#[test]
fn malformed_configs_are_rejected() {
    for (key, val) in [
        ("hc_count", json!(1)),
        ("hidden_size", json!(0)),
        ("num_hidden_layers", json!(0)),
        ("num_experts_per_tok", json!(5)),
        ("indexer_kv_heads", json!(2)),
        ("indexer_budget", json!(7)),
        ("indexer_head_dim", json!(4)),
        ("linear_num_value_heads", json!(5)),
        ("ngram_size", json!(1)),
        ("ple_layer_ids", json!([0])),
        ("ple_embed_dim", json!(31)),
        ("ple_layer_ids", json!([4])),
        ("attention_bias", json!(true)),
        ("output_gate_type", json!("relu")),
        ("output_gate_type", json!(42)),
        ("mamba_ssm_dtype", json!("garbage")),
        ("mamba_ssm_dtype", json!(false)),
        ("number_of_conv_states", json!(2)),
        ("pad_token_id", json!(264)),
        ("rope_parameters", json!("bad")),
        ("eos_token_id", json!([])),
        ("eos_token_id", json!(264)),
        ("linear_num_key_heads", json!(u64::MAX)),
        ("hc_count", json!(u64::MAX)),
        ("hidden_act", json!(false)),
        ("layer_types", json!(["unknown"])),
    ] {
        let mut c = config();
        c["text_config"][key] = val;
        assert!(Cfg::from_hf(&c).is_err(), "accepted {key}");
    }
    let mut c = config();
    c["text_config"]["rope_parameters"]["mrope_interleaved"] = json!("bad");
    assert!(Cfg::from_hf(&c).is_err());
    for layers in [4097, 10_000_000, u64::MAX] {
        let mut c = config();
        c["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("layer_types");
        c["text_config"]["num_hidden_layers"] = json!(layers);
        assert!(Cfg::from_hf(&c).is_err());
    }
    let mut c = config();
    c["quantization_config"] = json!({"quant_method":"exl3"});
    assert!(Cfg::from_hf(&c)
        .unwrap_err()
        .to_string()
        .contains("dense HF"));
    let mut c = config();
    c["text_config"]["num_hidden_layers"] = json!(u64::MAX);
    let f = Fixture::new(&c, &manifest());
    assert!(f.error().contains("layer count exceeds tensor count"));
}

#[test]
fn missing_wrong_shaped_unknown_weights_and_bad_ple_buffers_refuse() {
    let p = "model.language_model.layers.1.ple.ple_embedding";
    for (key, replacement, expected) in [
        ("lm_head.weight".into(), None, "missing lm_head"),
        (
            "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight".into(),
            Some(json!({"dtype":"BF16","shape":[255]})),
            "dense float [256]",
        ),
        (
            "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight".into(),
            None,
            "missing tensor",
        ),
        (
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight".into(),
            Some(json!({"dtype":"I32","shape":[32,64]})),
            "dense float",
        ),
        (
            format!("{p}.ngram_heads_offsets"),
            Some(json!({"dtype":"I64","shape":[4],"values":[0,67,138,210]})),
            "offsets must be cumulative",
        ),
        (
            format!("{p}.layer_multipliers"),
            None,
            "missing persistent I64",
        ),
        (
            "model.language_model.unexpected.weight".into(),
            Some(json!({"dtype":"BF16","shape":[1]})),
            "unrecognized tensor",
        ),
    ] {
        let mut m = manifest();
        if let Some(spec) = replacement {
            m[key] = spec;
        } else {
            m.as_object_mut().unwrap().remove(&key);
        }
        let f = Fixture::new(&config(), &m);
        assert!(
            f.error().contains(expected),
            "expected {expected}, got {}",
            f.error()
        );
    }
}

#[test]
fn ple_hash_buffers_must_match_config_not_just_shapes() {
    for (key, value, expected) in [
        ("seed", json!(20260908), "multipliers do not match"),
        (
            "ngram_vocab_size_base",
            json!(69),
            "head sizes do not match",
        ),
        ("ngram_vocab_size_base", json!(usize::MAX), "PLE"),
    ] {
        let mut c = config();
        c["text_config"][key] = value;
        let f = Fixture::new(&c, &manifest());
        assert!(f.error().contains(expected), "{}", f.error());
    }
    let mut m = manifest();
    let p = "model.language_model.layers.1.ple.ple_embedding";
    m[format!("{p}.layer_multipliers")]["values"][0] = json!(29891111472640047i64);
    let f = Fixture::new(&config(), &m);
    assert!(f.error().contains("multipliers do not match"));
    let c = Cfg::from_hf(&config()).unwrap();
    let ple = c.ple.as_ref().unwrap();
    validate_ple_hashes(
        &c,
        ple,
        1,
        &[9249548184588039, 31563857923877717, 31603134332648951],
        &[83, 89, 97, 101],
    )
    .unwrap();
}

#[test]
fn bounded_prime_validation_handles_large_values() {
    for p in [
        2,
        3,
        67,
        20_000_003,
        2_305_843_009_213_693_951,
        18_446_744_073_709_551_557,
    ] {
        assert!(is_prime(p), "{p}");
    }
    for n in [0, 1, 4, 561, 3_215_031_751, 341_550_071_728_321, u64::MAX] {
        assert!(!is_prime(n), "{n}");
    }
}

#[test]
#[ignore = "set INFR_TEST_MODEL to cached model.safetensors with adjacent config.json"]
fn cached_tiny_checkpoint_schema() {
    let path = std::env::var_os("INFR_TEST_MODEL").expect("INFR_TEST_MODEL");
    let s = Qwen4ExpSource::open(Path::new(&path)).unwrap();
    assert_eq!(s.cfg.hidden_size, 64);
    assert_eq!(s.infos.len(), 149);
    assert_eq!(s._ple_buffers.len(), 3);
    assert!(crate::Config::from_gguf(&s).unwrap().qwen4_exp.is_some());
}
