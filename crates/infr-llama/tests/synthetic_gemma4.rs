//! Synthetic-GGUF harness for `gemma4` KV-layer sharing (`attention.shared_kv_layers`): a tiny
//! model whose last layers read their K/V from an earlier layer's cache, written to a temp file
//! and driven through the REAL loader and seam — on CPU unconditionally, and on Vulkan behind
//! `#[ignore]`.
//!
//! llama.cpp's `llama_model_gemma4::load_arch_tensors` loads a shared layer's `attn_k`,
//! `attn_k_norm` (both `TENSOR_NOT_REQUIRED` past `n_layer_kv_from_start`) and `attn_v` (optional
//! on every layer) as optional, and its graph never reads them there. Some conversions drop them
//! from the shared layers; others ship them anyway. Both must load, and since a shared layer never
//! reads its own K/V the two must give the same logits.
//!
//! ```text
//! cargo test -p infr-llama --test synthetic_gemma4
//! cargo test -p infr-llama --test synthetic_gemma4 -- --include-ignored   # + Vulkan
//! ```

use infr_testkit::gguf::{Fill, Meta, SyntheticModel, TempGguf, TensorSpec};

const N_LAYER: usize = 6;
/// `attention.shared_kv_layers`: layers `N_LAYER - SHARED..` read an earlier layer's cache.
const SHARED: usize = 3;
const NE: usize = 64;
const N_HEAD: usize = 4;
const N_KV: usize = 2;
/// Full-attention head width (`attention.key_length`).
const HD: usize = 32;
/// Sliding-window head width (`attention.key_length_swa`).
const HD_SWA: usize = 16;
const NFF: usize = 96;
const VOCAB: usize = 64;
const WINDOW: usize = 4;
/// `attention.sliding_window_pattern`: every third layer is full attention, so the last layer
/// that owns its KV (layer 2) is a full one and the one before it a window one — the layout
/// `Config::kv_src_layer` maps the shared layers onto.
const SWA: [bool; N_LAYER] = [true, true, false, true, true, false];
/// Longer than the window, so the shared window layers read past it.
const PROMPT: &[u32] = &[3, 11, 5, 40, 27, 8, 61, 14, 33, 20];
/// The tensors llama.cpp treats as optional on a layer that shares its KV.
const SHARED_OPTIONAL: [&str; 3] = ["attn_k", "attn_v", "attn_k_norm"];

fn token_text(id: u32) -> char {
    char::from(b'!' + u8::try_from(id).expect("a vocab id"))
}

/// The model with the K/V tensors on every layer, shared or not.
fn gemma4_model() -> SyntheticModel {
    let k = |key: &str| format!("gemma4.{key}");
    let u = |key: &str, v: usize| (k(key), Meta::U32(v as u32));
    let meta = vec![
        (
            "general.architecture".to_string(),
            Meta::Str("gemma4".to_string()),
        ),
        u("block_count", N_LAYER),
        u("context_length", 64),
        u("embedding_length", NE),
        u("feed_forward_length", NFF),
        u("attention.head_count", N_HEAD),
        u("attention.head_count_kv", N_KV),
        u("attention.key_length", HD),
        u("attention.value_length", HD),
        u("attention.key_length_swa", HD_SWA),
        u("attention.value_length_swa", HD_SWA),
        u("rope.dimension_count", HD),
        u("rope.dimension_count_swa", HD_SWA),
        (k("rope.freq_base"), Meta::F32(1.0e6)),
        (k("rope.freq_base_swa"), Meta::F32(1.0e4)),
        (k("attention.layer_norm_rms_epsilon"), Meta::F32(1.0e-6)),
        u("attention.sliding_window", WINDOW),
        (
            k("attention.sliding_window_pattern"),
            Meta::BoolArr(SWA.to_vec()),
        ),
        u("attention.shared_kv_layers", SHARED),
        (k("final_logit_softcapping"), Meta::F32(30.0)),
        // The minimum tokenizer `build_tokenizer` accepts; the tests hand token ids to the seam.
        (
            "tokenizer.ggml.model".to_string(),
            Meta::Str("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Meta::StrArr(
                (0..VOCAB as u32)
                    .map(|i| token_text(i).to_string())
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            Meta::StrArr(Vec::new()),
        ),
        ("tokenizer.ggml.eos_token_id".to_string(), Meta::U32(2)),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            Meta::Bool(false),
        ),
    ];
    // Proportional RoPE on the full layers: the first half of the pairs rope, the rest stand
    // still, as in the shipped files.
    let rope_freqs: Vec<f32> = (0..HD / 2)
        .map(|p| if p < HD / 4 { 1.0 } else { 1.0e30 })
        .collect();
    let mut tensors = vec![
        TensorSpec::new("token_embd.weight", vec![NE, VOCAB], Fill::Rand(0.5)),
        TensorSpec::new("output_norm.weight", vec![NE], Fill::Gain),
        TensorSpec::new("rope_freqs.weight", vec![HD / 2], Fill::Exact(rope_freqs)),
    ];
    let w = |l: usize, name: &str, shape: Vec<usize>, fill: Fill| {
        TensorSpec::new(format!("blk.{l}.{name}.weight"), shape, fill)
    };
    for (l, &swa) in SWA.iter().enumerate() {
        let hd = if swa { HD_SWA } else { HD };
        for norm in [
            "attn_norm",
            "ffn_norm",
            "post_attention_norm",
            "post_ffw_norm",
        ] {
            tensors.push(w(l, norm, vec![NE], Fill::Gain));
        }
        tensors.push(w(l, "attn_q", vec![NE, N_HEAD * hd], Fill::Rand(0.3)));
        tensors.push(w(l, "attn_k", vec![NE, N_KV * hd], Fill::Rand(0.3)));
        tensors.push(w(l, "attn_v", vec![NE, N_KV * hd], Fill::Rand(0.3)));
        tensors.push(w(l, "attn_q_norm", vec![hd], Fill::Gain));
        tensors.push(w(l, "attn_k_norm", vec![hd], Fill::Gain));
        tensors.push(w(l, "attn_output", vec![N_HEAD * hd, NE], Fill::Rand(0.2)));
        tensors.push(w(l, "ffn_gate", vec![NE, NFF], Fill::Rand(0.2)));
        tensors.push(w(l, "ffn_up", vec![NE, NFF], Fill::Rand(0.2)));
        tensors.push(w(l, "ffn_down", vec![NFF, NE], Fill::Rand(0.2)));
        tensors.push(w(
            l,
            "layer_output_scale",
            vec![1],
            Fill::Exact(vec![0.5 + 0.1 * l as f32]),
        ));
    }
    SyntheticModel { meta, tensors }
}

fn is_shared(l: usize) -> bool {
    l >= N_LAYER - SHARED
}

/// `model` without the tensors named `blk.{l}.{role}.weight` for each `(l, role)` in `drop`.
fn without(model: &SyntheticModel, drop: &[(usize, &str)]) -> SyntheticModel {
    let names: Vec<String> = drop
        .iter()
        .map(|(l, role)| format!("blk.{l}.{role}.weight"))
        .collect();
    let mut m = SyntheticModel {
        meta: model.meta.clone(),
        tensors: model.tensors.clone(),
    };
    let before = m.tensors.len();
    m.tensors.retain(|t| !names.contains(&t.name));
    assert_eq!(before - m.tensors.len(), names.len(), "dropped {names:?}");
    m
}

/// The model as a conversion that drops the shared layers' optional tensors writes it.
fn gemma4_stripped() -> SyntheticModel {
    let drop: Vec<(usize, &str)> = (0..N_LAYER)
        .filter(|&l| is_shared(l))
        .flat_map(|l| SHARED_OPTIONAL.iter().map(move |&r| (l, r)))
        .collect();
    without(&gemma4_model(), &drop)
}

fn load(tmp: &TempGguf) -> anyhow::Result<infr_llama::SeamModel> {
    let mut cfg = infr_llama::EngineConfig::default();
    infr_testkit::honour_env(&mut cfg);
    infr_llama::SeamModel::load_with(tmp.path(), None, std::sync::Arc::new(cfg))
}

fn cpu_logits(tag: &str, model: &SyntheticModel) -> Vec<f32> {
    let tmp = TempGguf::write(tag, model);
    load(&tmp)
        .expect("synthetic model load")
        .prefill_logits_cpu(PROMPT)
        .unwrap_or_else(|e| panic!("{tag}: cpu prefill: {e:#}"))
}

/// Max |a - b|, NaN if any difference is NaN (`f32::max` would drop a NaN operand).
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit vector lengths");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, |m, d| {
            if m.is_nan() || d.is_nan() {
                f32::NAN
            } else {
                m.max(d)
            }
        })
}

/// The fixture is laid out the way the test means it: the metadata marks layers 3.. shared, each
/// pointing at an earlier layer of its own kind.
#[test]
fn synthetic_gemma4_parses_its_shared_layers() {
    let tmp = TempGguf::write("gemma4-kv-parse", &gemma4_stripped());
    let model = load(&tmp).expect("synthetic model load");
    let c = model.config();
    assert!(c.gemma4 && c.qk_norm);
    assert_eq!(c.n_layer_kv_from_start, N_LAYER - SHARED);
    for (l, &swa) in SWA.iter().enumerate() {
        assert_eq!(c.is_swa_layer(l), swa, "layer {l} window");
        assert_eq!(c.has_own_kv(l), !is_shared(l), "layer {l} owns its KV");
        let src = c.kv_src_layer(l);
        assert_eq!(c.is_swa_layer(src), swa, "layer {l} reads layer {src}");
        assert!(c.has_own_kv(src), "layer {l} reads shared layer {src}");
    }
}

/// A file whose shared layers carry no `attn_k` / `attn_v` / `attn_k_norm` loads, and its CPU
/// prefill gives the same bits as the same model with those tensors present: a shared layer
/// reads only the cache of the layer it shares, so its own K/V weights never reach the logits.
/// The model with them present takes the long-standing load path, which makes it the reference.
#[test]
fn synthetic_gemma4_cpu_shared_layers_need_no_kv_tensors() {
    let full = cpu_logits("gemma4-kv-full", &gemma4_model());
    assert_eq!(full.len(), VOCAB);
    assert!(full.iter().all(|v| v.is_finite()), "non-finite logits");
    let stripped = cpu_logits("gemma4-kv-stripped", &gemma4_stripped());
    assert_eq!(
        full,
        stripped,
        "dropping the shared layers' K/V tensors moved the logits by {}",
        max_abs_diff(&full, &stripped)
    );
    // Each tensor alone on each shared layer, window and full, so a loader whose weight order
    // depends on which of them a layer carries is covered too.
    for l in (0..N_LAYER).filter(|&l| is_shared(l)) {
        for role in SHARED_OPTIONAL {
            let tag = format!("gemma4-kv-{l}-no-{role}");
            let got = cpu_logits(&tag, &without(&gemma4_model(), &[(l, role)]));
            assert_eq!(full, got, "dropping blk.{l}.{role} moved the logits");
        }
    }
    // The comparison has teeth: the shared layers do reach the logits, through their queries.
    // A rescale would not show (the query is RMS-normed), so the new values are a different
    // draw: the ones the same fill gives under another tensor's name.
    let mut moved = gemma4_model();
    let name = format!("blk.{}.attn_q.weight", N_LAYER - 1);
    let t = moved
        .tensors
        .iter_mut()
        .find(|t| t.name == name)
        .expect("shared layer query");
    let redraw = TensorSpec::new("redraw", t.shape.clone(), Fill::Rand(0.3)).values();
    t.fill = Fill::Exact(redraw);
    let other = cpu_logits("gemma4-kv-moved-q", &moved);
    assert!(
        max_abs_diff(&full, &other) > 1e-3,
        "a shared layer's query projection does not reach the logits"
    );
}

/// A layer that computes its own K/V still needs every one of those tensors: dropping one is
/// refused at load, naming the tensor, rather than run as a model that silently lacks it.
/// (`attn_v` is not on the list: a gemma4 layer without it reads its V from the K projection.)
#[test]
fn synthetic_gemma4_refuses_missing_kv_on_an_own_kv_layer() {
    let own = N_LAYER - SHARED - 1;
    for role in ["attn_k", "attn_k_norm"] {
        let m = without(&gemma4_stripped(), &[(own, role)]);
        let tmp = TempGguf::write(&format!("gemma4-kv-own-no-{role}"), &m);
        let name = format!("blk.{own}.{role}.weight");
        let err = match load(&tmp).and_then(|model| model.prefill_logits_cpu(PROMPT)) {
            Ok(_) => panic!("a missing {name} on an own-KV layer must be refused"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains(&name), "refusal does not name {name}: {err}");
    }
}

/// The Vulkan seam against the CPU on the stripped model.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_gemma4_vulkan_shared_layers_match_cpu() {
    let model = gemma4_stripped();
    let want = cpu_logits("gemma4-kv-cpu", &model);
    let tmp = TempGguf::write("gemma4-kv-vk", &model);
    let got = load(&tmp)
        .expect("synthetic model load")
        .prefill_logits_vulkan(PROMPT)
        .expect("vulkan prefill");
    let err = max_abs_diff(&want, &got);
    assert!(err < 2e-2, "Vulkan misses the CPU by {err}");
}
