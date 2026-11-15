//! Dense layer streaming of FLOAT weights, end to end on a synthetic llama: an all-f16 and an
//! all-f32 checkpoint, with the dense cache forced far below the model's projections
//! (`paging.cache`), must stream their per-layer projections through the arena and generate the
//! same tokens as the all-resident run.
//!
//! A float checkpoint used to have no streamable dense weights at all — the placement planner took
//! only the native quant formats — so a capped budget was silently ignored and the whole model
//! went resident. Both assertions below fail on that: no streaming session exists to report
//! counters.
//!
//! The model is shaped so the projections take every float route: a prefill long enough for the
//! tiled GEMMs (the f16 coopmat GEMM, the split-K warptile on the deep `ffn_down`, the fma
//! warptile without coopmat), decode through the GEMVs, and three layers so each pool holds more
//! blocks than slots and recycles them. Two streamed runs: one with the host DRAM tier off, so
//! misses read the mapping, and one with a host arena under the VRAM one (`paging.dram`), so the
//! float blocks are also read back off the model file.
//!
//! ```text
//! INFR_DEV=Vulkan0 cargo test -p infr-llama --release --test synthetic_float_stream -- --ignored
//! ```

use infr_testkit::gguf::{Fill, Meta, SyntheticModel, TempGguf, TensorSpec};

const N_LAYER: usize = 3;
const NE: usize = 128;
const N_HEAD: usize = 4;
const N_KV: usize = 2;
const HD: usize = NE / N_HEAD;
/// Deep enough that `ffn_down`'s narrow GEMM takes the f16 split-K warptile.
const NFF: usize = 1024;
const VOCAB: usize = 64;
/// Tokens to generate.
const N_GEN: usize = 12;

/// Token `id`'s text: the printable ASCII characters from `!` on, each its own GPT-2 byte-level
/// symbol, so a string of them encodes back to exactly these ids.
fn token_text(id: usize) -> char {
    char::from(b'!' + u8::try_from(id).expect("a vocab id"))
}

/// A llama whose every projection (and embedding and output) is `float`; the norms stay f32, as
/// in a real f16 checkpoint.
fn llama(float: Float) -> SyntheticModel {
    let k = |key: &str| format!("llama.{key}");
    let u = |key: &str, v: usize| (k(key), Meta::U32(v as u32));
    let meta = vec![
        (
            "general.architecture".to_string(),
            Meta::Str("llama".to_string()),
        ),
        u("block_count", N_LAYER),
        u("context_length", 256),
        u("embedding_length", NE),
        u("feed_forward_length", NFF),
        u("attention.head_count", N_HEAD),
        u("attention.head_count_kv", N_KV),
        u("rope.dimension_count", HD),
        (k("rope.freq_base"), Meta::F32(1.0e4)),
        (k("attention.layer_norm_rms_epsilon"), Meta::F32(1.0e-5)),
        (
            "tokenizer.ggml.model".to_string(),
            Meta::Str("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Meta::StrArr((0..VOCAB).map(|i| token_text(i).to_string()).collect()),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            Meta::StrArr(Vec::new()),
        ),
        ("tokenizer.ggml.eos_token_id".to_string(), Meta::U32(0)),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            Meta::Bool(false),
        ),
    ];
    let w = |name: String, shape: Vec<usize>, amp: f32| float.tensor(name, shape, amp);
    let mut tensors = vec![
        w("token_embd.weight".into(), vec![NE, VOCAB], 1.0),
        TensorSpec::new("output_norm.weight", vec![NE], Fill::Gain),
        w("output.weight".into(), vec![NE, VOCAB], 0.5),
    ];
    for l in 0..N_LAYER {
        let n = |s: &str| format!("blk.{l}.{s}.weight");
        tensors.push(TensorSpec::new(n("attn_norm"), vec![NE], Fill::Gain));
        tensors.push(TensorSpec::new(n("ffn_norm"), vec![NE], Fill::Gain));
        tensors.push(w(n("attn_q"), vec![NE, N_HEAD * HD], 0.3));
        tensors.push(w(n("attn_k"), vec![NE, N_KV * HD], 0.3));
        tensors.push(w(n("attn_v"), vec![NE, N_KV * HD], 0.3));
        tensors.push(w(n("attn_output"), vec![N_HEAD * HD, NE], 0.2));
        tensors.push(w(n("ffn_gate"), vec![NE, NFF], 0.2));
        tensors.push(w(n("ffn_up"), vec![NE, NFF], 0.2));
        tensors.push(w(n("ffn_down"), vec![NFF, NE], 0.1));
    }
    SyntheticModel { meta, tensors }
}

#[derive(Clone, Copy, Debug)]
enum Float {
    F16,
    F32,
}

impl Float {
    /// A pseudo-random tensor in `[-amp, amp]`, written as this dtype.
    fn tensor(self, name: String, shape: Vec<usize>, amp: f32) -> TensorSpec {
        let t = TensorSpec::new(name, shape, Fill::Rand(amp));
        match self {
            Float::F32 => t,
            Float::F16 => {
                let v = t.values();
                TensorSpec::new(t.name, t.shape, Fill::ExactF16(v))
            }
        }
    }
}

fn load(tmp: &TempGguf, edit: impl FnOnce(&mut infr_llama::EngineConfig)) -> infr_llama::SeamModel {
    let mut cfg = infr_llama::EngineConfig::default();
    edit(&mut cfg);
    infr_llama::SeamModel::load_with(tmp.path(), None, std::sync::Arc::new(cfg))
        .expect("synthetic model load")
}

/// Greedy text for `prompt` on a fresh Vulkan session of `model`, and the dense streaming
/// session's counters after it (`None` when nothing streamed).
fn generate(
    model: &infr_llama::SeamModel,
    prompt: &str,
) -> (String, Option<infr_vulkan::pager::DenseTotals>) {
    let mut session = model.vulkan_session(256).expect("vulkan session");
    let mut text = String::new();
    model
        .generate_vulkan_session(&mut session, prompt, N_GEN, None, |p| text.push_str(p))
        .expect("vulkan generation");
    (text, session.dense_pager_totals())
}

fn check(float: Float) {
    let tag = format!("float-stream-{float:?}");
    let tmp = TempGguf::write(&tag, &llama(float));
    // Long enough for a tiled prefill GEMM (well past the multi-row GEMV bands).
    let prompt: String = (0..40).map(|i| token_text((i * 7 + 3) % VOCAB)).collect();

    // The fused-gate/up capability the seam's group enumeration reads (`fuse_gu_decision`).
    let combined_gu = {
        use infr_core::Backend;
        infr_vulkan::VulkanBackend::new()
            .expect("vulkan device")
            .capabilities()
            .combined_gu
    };
    let (resident, none) = generate(&load(&tmp, |_| {}), &prompt);
    assert!(
        none.is_none(),
        "{float:?}: the resident run streamed ({none:?})"
    );
    assert!(
        resident.chars().count() >= N_GEN / 2,
        "{float:?}: the resident run stopped early ({resident:?}) — too few tokens to compare"
    );

    // 64 KiB is below the staging ring's own 256 MiB floor (`infr_core::pager::ring_bytes`), so
    // nothing is left for the arenas and every pool runs at its 2-slot floor — below the three
    // or more blocks each pool holds — so blocks are evicted and re-staged every forward.
    let cap = |c: &mut infr_llama::EngineConfig| {
        c.paging.cache = Some(infr_core::SizeSpec::Bytes(64 * 1024));
    };
    // `paging.dram = 0` turns the host tier off (misses read the mapping); 1 MiB is below the
    // model's streamed bytes, so the host arena evicts too.
    for (leg, dram) in [("over the mmap", 0), ("over the host tier", 1024 * 1024)] {
        let streamed_model = load(&tmp, |c| {
            cap(c);
            c.paging.dram = Some(infr_core::SizeSpec::Bytes(dram));
        });
        let (streamed, totals) = generate(&streamed_model, &prompt);
        let t = totals.unwrap_or_else(|| panic!("{float:?} ({leg}): no weight streamed"));
        println!("{float:?} ({leg}): {t:?}");
        // Every layer's q, k, v, output, gate and up (one group when the backend uploads them
        // combined, as `fuse_gu_decision` asks of it — every layer here shares one dtype) and
        // down.
        let gate_up_groups = if combined_gu { 1 } else { 2 };
        assert_eq!(
            t.groups,
            (5 + gate_up_groups) * N_LAYER,
            "{float:?} ({leg}): streamed groups"
        );
        assert!(
            t.vram.evictions > 0,
            "{float:?} ({leg}): nothing was evicted — the cap did not bite ({t:?})"
        );
        assert_eq!(
            t.file_reads > 0,
            dram > 0,
            "{float:?} ({leg}): file reads with the host tier, none without it ({t:?})"
        );
        assert_eq!(
            streamed, resident,
            "{float:?} ({leg}): the streamed run diverged from the resident one"
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_f16_llama_streams_and_matches_resident() {
    check(Float::F16);
}

#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_f32_llama_streams_and_matches_resident() {
    check(Float::F32);
}
