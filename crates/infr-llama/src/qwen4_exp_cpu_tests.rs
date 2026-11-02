use super::*;
use serde_json::{json, Value};
use std::io::Write;

fn config() -> Value {
    serde_json::from_str(include_str!("../tests/fixtures/qwen4_exp/config.json")).unwrap()
}
fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../tests/fixtures/qwen4_exp/cpu_reference.json"
    ))
    .unwrap()
}

// Matches generate_qwen4_exp_reference.py exactly, including BF16 rounding.
fn pattern(tensor: usize, element: usize, name: &str) -> f32 {
    let mut v = (tensor as u32 + 1).wrapping_mul(0x9e3779b9)
        ^ (element as u32 + 1).wrapping_mul(0x85ebca6b);
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb352d);
    v ^= v >> 15;
    let x = ((v % 2049) as i32 - 1024) as f32 / 32768.0;
    x + if name.ends_with("linear_attn.norm.weight") {
        1.0
    } else {
        0.0
    }
}

struct Fixture {
    dir: tempfile::TempDir,
    path: std::path::PathBuf,
}
impl Fixture {
    fn new(cfg: &Value) -> Self {
        let manifest: Value =
            serde_json::from_str(include_str!("../tests/fixtures/qwen4_exp/tensors.json")).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut ordered: Vec<_> = manifest.as_object().unwrap().iter().collect();
        ordered.sort_by(|a, b| a.0.cmp(b.0));
        for (ordinal, (name, spec)) in ordered.into_iter().enumerate() {
            if name == "lm_head.weight" && cfg["text_config"]["tie_word_embeddings"] == true {
                continue;
            }
            let count: usize = spec["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .product();
            let start = data.len();
            if spec["dtype"] == "I64" {
                for v in spec["values"].as_array().unwrap() {
                    data.extend_from_slice(&v.as_i64().unwrap().to_le_bytes());
                }
            } else {
                for j in 0..count {
                    let v = pattern(ordinal, j, name);
                    match spec["dtype"].as_str().unwrap() {
                        "BF16" => data.extend_from_slice(&bf16::from_f32(v).to_le_bytes()),
                        "F16" => data.extend_from_slice(&f16::from_f32(v).to_le_bytes()),
                        "F32" => data.extend_from_slice(&v.to_le_bytes()),
                        _ => panic!("unexpected fixture dtype"),
                    }
                }
            }
            header.insert(name.clone(), json!({"dtype":spec["dtype"],"shape":spec["shape"],"data_offsets":[start,data.len()]}));
        }
        let mut h = serde_json::to_vec(&header).unwrap();
        while !h.len().is_multiple_of(8) {
            h.push(b' ');
        }
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&h).unwrap();
        file.write_all(&data).unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(cfg).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), serde_json::to_vec(&json!({
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null, "pre_tokenizer":null, "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel", "vocab":{"[UNK]":0,"a":1,"<|im_end|>":2}, "unk_token":"[UNK]"}
        })).unwrap()).unwrap();
        Self { dir, path }
    }
    fn source(&self) -> Qwen4ExpSource {
        Qwen4ExpSource::open(&self.path).unwrap()
    }
}
fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let (i, error) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(i, (a, b))| (i, (a - b).abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    assert!(
        actual.iter().all(|v| v.is_finite()),
        "{label}: nonfinite value"
    );
    assert!(
        error <= 2e-5,
        "{label}: max abs {error} at {i}: {} vs {}",
        actual[i],
        expected[i]
    );
}

#[test]
fn hf_causal_hidden_logits_and_recurrent_cache_parity() {
    let fixture = Fixture::new(&config());
    let source = fixture.source();
    let mut model = CpuModel::new(&source).unwrap();
    compare_oracle(&mut model, &reference());
}

fn compare_oracle(model: &mut CpuModel<'_>, r: &Value) {
    let tokens: Vec<u32> = r["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    for (position, &id) in tokens.iter().enumerate() {
        let step = model.step(id).unwrap();
        let expected = &r["steps"][position];
        close(
            &step.logits,
            &floats(&expected["logits"]),
            &format!("logits token {position}"),
        );
        close(
            &step.hidden,
            &floats(&expected["hidden"]),
            &format!("hidden token {position}"),
        );
        if let Attention::Delta(state) = &model.layers[0].attention {
            if !expected["gdn_state_0"].is_null() {
                close(
                    &state.recurrent,
                    &floats(&expected["gdn_state_0"]),
                    "GDN state",
                );
                close(
                    &state.raw,
                    &floats(&expected["gdn_raw_0"]),
                    "GDN raw history",
                );
            }
        }
    }
    assert_eq!(model.position(), tokens.len());
    model.reset();
    assert_eq!(model.position(), 0);
    let mut whole = Vec::new();
    for chunk in [&tokens[..1], &tokens[1..5], &tokens[5..]] {
        whole.extend(model.prefill(chunk).unwrap());
    }
    for (pos, row) in whole.iter().enumerate() {
        close(
            row,
            &floats(&r["prefill_logits"][pos]),
            &format!("prefill {pos}"),
        );
    }
}

#[test]
fn invalid_prefill_is_atomic_and_context_is_bounded() {
    let mut c = config();
    c["text_config"]["max_position_embeddings"] = json!(4);
    let fixture = Fixture::new(&c);
    let source = fixture.source();
    let mut model = CpuModel::new(&source).unwrap();
    assert!(model
        .prefill(&[1, 264])
        .unwrap_err()
        .to_string()
        .contains("vocabulary"));
    assert_eq!(model.position(), 0);
    assert!(model
        .prefill(&[1; 5])
        .unwrap_err()
        .to_string()
        .contains("context limit"));
    assert_eq!(model.position(), 0);
    model.prefill(&[1; 4]).unwrap();
    assert!(model.step(1).is_err());
    assert_eq!(model.position(), 4);
    model.reset();
    assert_eq!(model.step(1).unwrap().logits.len(), 264);
    // Keep temp directory live through all source/mmap use.
    assert!(fixture.dir.path().exists());
}

#[test]
fn giant_reference_cache_refuses_before_allocating() {
    let mut c = config();
    c["text_config"]["max_position_embeddings"] = json!(1_000_000_000);
    let fixture = Fixture::new(&c);
    let source = fixture.source();
    assert!(CpuModel::new(&source)
        .err()
        .unwrap()
        .to_string()
        .contains("256 MiB"));
}

#[test]
fn packed_experts_consolidated_ple_and_indexer_ties() {
    let fixture = Fixture::new(&config());
    let source = fixture.source();
    let mut original = CpuModel::new(&source).unwrap();
    let mut packed_bytes = Vec::new();
    for l in 0..4 {
        let mut gu = Vec::new();
        let mut down = Vec::new();
        for e in 0..4 {
            gu.extend_from_slice(
                weight(
                    &original.weights,
                    &format!("blk.{l}.expert.{e}.gate.weight"),
                )
                .bytes,
            );
            gu.extend_from_slice(
                weight(&original.weights, &format!("blk.{l}.expert.{e}.up.weight")).bytes,
            );
            down.extend_from_slice(
                weight(
                    &original.weights,
                    &format!("blk.{l}.expert.{e}.down.weight"),
                )
                .bytes,
            );
        }
        packed_bytes.push((gu, down));
    }
    let mut table = Vec::new();
    for part in 0..2 {
        table.extend_from_slice(
            weight(&original.weights, &format!("blk.1.ple_embd.{part}.weight")).bytes,
        );
    }
    let mut packed = CpuModel::new(&source).unwrap();
    for (l, (gu, down)) in packed_bytes.iter().enumerate() {
        packed.weights.insert(
            format!("blk.{l}.ffn_gate_up_exps.weight"),
            Dense {
                bytes: gu,
                shape: &[64, 64, 4],
                dtype: DType::Bf16,
            },
        );
        packed.weights.insert(
            format!("blk.{l}.ffn_down_exps.weight"),
            Dense {
                bytes: down,
                shape: &[32, 64, 4],
                dtype: DType::Bf16,
            },
        );
    }
    packed.weights.insert(
        "blk.1.ple_embd.weight".into(),
        Dense {
            bytes: &table,
            shape: &[8, 296],
            dtype: DType::Bf16,
        },
    );
    for id in [
        1, 17, 23, 2, 31, 7, 0, 48, 5, 79, 3, 98, 67, 122, 255, 8, 131,
    ] {
        let a = original.step(id).unwrap();
        let b = packed.step(id).unwrap();
        assert_eq!(a.logits, b.logits);
        assert_eq!(a.hidden, b.hidden);
    }
    let qk = weight(&original.weights, "blk.3.indexer_qk.weight");
    let zeros = vec![0; qk.bytes.len()];
    let mut weights = original.weights.clone();
    weights.insert(
        "blk.3.indexer_qk.weight".into(),
        Dense {
            bytes: &zeros,
            ..qk
        },
    );
    let mut s = AttentionState {
        index_keys: vec![vec![1.0; 8]; 16],
        ..AttentionState::default()
    };
    let selected = index_selection(&weights, "blk.3", &[0.0; 64], &original.cfg, 16, &mut s);
    assert_eq!(selected, (0..8).chain([16]).collect::<Vec<_>>());
}

#[test]
fn grammar_and_seeded_sampling_use_causal_state() {
    use crate::sampling::{Penalties, RequestCtx, RequestSampling, Sampler};
    let fixture = Fixture::new(&config());
    let source = fixture.source();
    let ec = crate::EngineConfig::default();
    let request = RequestCtx::new(RequestSampling {
        temp: Some(0.8),
        top_k: Some(8),
        seed: Some(42),
        presence_penalty: 0.3,
        frequency_penalty: 0.2,
        repeat_penalty: 1.1,
        ..RequestSampling::default()
    });
    let mut direct = CpuModel::new(&source).unwrap();
    let mut logits = direct.step(1).unwrap().logits;
    let mut rng = 42;
    let mut penalties = Penalties::resolve(Some(&request)).unwrap();
    let sampler = Sampler::resolve(Some(&request), &ec.sampling);
    let mut expected = Vec::new();
    for _ in 0..8 {
        penalties.apply(&mut logits);
        let id = crate::sampling::sample_logits(&logits, sampler, &mut rng);
        penalties.observe(id);
        expected.push(id);
        logits = direct.step(id).unwrap().logits;
    }
    let (actual, _) = generate(&source, &ec, &[1], 8, &[], None, Some(&request), |_| {}).unwrap();
    assert_eq!(actual, expected);
    let mut tok =
        tokenizers::Tokenizer::from_file(fixture.path.with_file_name("tokenizer.json")).unwrap();
    tok.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    let env = crate::grammar::build_tok_env(&tok, 264, &[2]).unwrap();
    let grammar = llguidance::api::TopLevelGrammar::from_lark("start: \"aaa\"".into());
    let mut cst = crate::grammar::Constraint::new(env, grammar).unwrap();
    let (out, _) = generate(&source, &ec, &[1], 8, &[2], Some(&mut cst), None, |_| {}).unwrap();
    assert_eq!(out, [1, 1, 1]);
}

#[test]
fn tied_head_and_allocated_cache_bounds() {
    let mut c = config();
    c["text_config"]["tie_word_embeddings"] = json!(true);
    let fixture = Fixture::new(&c);
    let source = fixture.source();
    assert!(!source.tensors().iter().any(|t| t.name == "output.weight"));
    let mut model = CpuModel::new(&source).unwrap();
    let step = model.step(1).unwrap();
    assert_eq!(
        step.logits,
        weight(&model.weights, "token_embd.weight").linear(&step.hidden)
    );
    if let Attention::Indexed(s) = &model.layers[3].attention {
        assert_eq!(s.keys.capacity(), model.cfg.max_position_embeddings);
        assert_eq!(s.values.capacity(), model.cfg.max_position_embeddings);
        assert_eq!(s.index_keys.capacity(), model.cfg.max_position_embeddings);
    } else {
        panic!("expected indexed attention");
    }
    assert!(output_bound(67_108_864, 1, std::mem::size_of::<Vec<f32>>()).is_err());
    assert!(output_bound(usize::MAX, 2, 0).is_err());
}

#[test]
fn cpu_wrappers_generation_eos_and_cancellation() {
    use crate::sampling::{RequestCtx, RequestSampling};
    let fixture = Fixture::new(&config());
    let source = fixture.source();
    let cfg = source.cfg.clone().shared_config();
    let mut ec = crate::EngineConfig::default();
    ec.sampling.temp = 0.0;
    let ec = std::sync::Arc::new(ec);
    let cell = std::sync::OnceLock::new();
    let emb = crate::seam::TokenEmbd::new(&cell, &source); // Dispatch must precede ordinary binding.
    let ids = [1, 17, 23, 2, 31];
    let mut direct = CpuModel::new(&source).unwrap();
    let rows = direct.prefill(&ids).unwrap();
    let last = crate::seam::verify_dense_cpu(&source, &cfg, &ec, emb, None, &ids).unwrap();
    assert_eq!(last, *rows.last().unwrap());
    let (all, hidden) =
        crate::seam::verify_rows_cpu_with_h(&source, &cfg, &ec, emb, None, &ids).unwrap();
    assert_eq!(all, rows.into_iter().flatten().collect::<Vec<_>>());
    assert_eq!(hidden.len(), ids.len() * 64);
    let (_, last_h) =
        crate::seam::verify_dense_cpu_with_h(&source, &cfg, &ec, emb, None, &ids).unwrap();
    assert_eq!(last_h, hidden[hidden.len() - 64..]);
    let greedy = crate::sampling::sample_logits(
        &last,
        crate::sampling::Sampler::resolve(None, &ec.sampling),
        &mut 1,
    );
    let mut called = Vec::new();
    // A runtime-added EOS is not the source HF EOS used by the PLE hash.
    let (out, stats) = generate(&source, &ec, &ids, 5, &[greedy], None, None, |id| {
        called.push(id)
    })
    .unwrap();
    assert_eq!(out, [greedy]);
    assert!(called.is_empty());
    assert_eq!(stats.n_prompt, 5);
    assert_eq!(stats.n_gen, 1);
    let request = RequestCtx::new(RequestSampling::default());
    let (out, _) = generate(&source, &ec, &ids, 5, &[], None, Some(&request), |_| {
        request.abort()
    })
    .unwrap();
    assert_eq!(out.len(), 1);
    let request = RequestCtx::new(RequestSampling::default());
    request.abort();
    assert!(generate(
        &source,
        &ec,
        &ids,
        1,
        &[],
        None,
        Some(&request),
        |_| panic!("callback")
    )
    .unwrap_err()
    .to_string()
    .contains("cancelled during prompt"));
}

#[test]
#[ignore = "requires INFR_TEST_MODEL and adjacent generated cpu_reference.json"]
fn cached_tiny_cpu_reference_parity() {
    let path = std::env::var("INFR_TEST_MODEL")
        .expect("set INFR_TEST_MODEL to the cached tiny checkpoint");
    let source = Qwen4ExpSource::open(std::path::Path::new(&path)).unwrap();
    let mut model = CpuModel::new(&source).unwrap();
    let oracle_path = std::path::Path::new(&path).with_file_name("cpu_reference.json");
    let r = serde_json::from_slice(
        &std::fs::read(oracle_path).expect("generate the local HF oracle first"),
    )
    .unwrap();
    compare_oracle(&mut model, &r);
}
