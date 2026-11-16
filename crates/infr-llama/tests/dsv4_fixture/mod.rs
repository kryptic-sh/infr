//! The synthetic `deepseek4` (V4) GGUF, shared by the test binaries that load it:
//! `synthetic_deepseek2.rs` (the loader, the refusals and the ratio-0 tier) and
//! `synthetic_deepseek4_llama_cpp.rs` (the llama.cpp oracle over the same file).
//!
//! [`IndexerDims`] is here because [`dsv4_model`] reads it; `deepseek32`'s fixture in
//! `synthetic_deepseek2.rs` uses it too.

use infr_testkit::gguf::{Fill, Meta, SyntheticModel, TensorSpec};

/// The `deepseek32` lightning indexer's three hyperparameters — the only metadata V3.2 adds to
/// `Ds2Dims` (synthetic_deepseek2.rs). See `docs/deepseek.md` § Stage 3.
#[derive(Clone, Debug)]
pub struct IndexerDims {
    /// `attention.indexer.head_count` — the indexer's QUERY heads. One key head serves all of them.
    pub n_head: usize,
    /// `attention.indexer.key_length` — the per-head key/query width, and the width of `k_norm`.
    pub head_size: usize,
    /// `attention.indexer.top_k` — how many keys survive to the real attention.
    pub top_k: usize,
}

/// The `deepseek4` lightning indexer. V4's differs structurally from V3.2's — no `indexer.attn_k`,
/// no `indexer.k_norm`, its keys come out of the compressor — but the three metadata keys are
/// shared, so [`IndexerDims`] describes it too.
///
/// `head_size` is deliberately neither `head_dim` nor `rope_dim` and `n_head` is not the model's
/// `n_head`, so the indexer's four shaped tensors have four shapes no other dimension of
/// [`dsv4_dims`] would fit. llama.cpp does not load a width under 64 (it forces a Hadamard rotation
/// on the indexer's cache and has the matrices only from 64 up), so the files it runs widen this in
/// their own [`Dsv4Dims::indexer`].
pub fn dsv4_indexer() -> IndexerDims {
    IndexerDims {
        n_head: 4,
        head_size: 24,
        top_k: 5,
    }
}

/// Every dimension of the synthetic `deepseek4` model.
#[derive(Clone, Debug)]
pub struct Dsv4Dims {
    /// One entry per layer; `n_layer` IS `compress_ratios.len()`, because the ratio is what decides
    /// a layer's tensor set and a fixture with a layer count independent of it could not be
    /// consistent. See [`dsv4_dims`] for why this particular sequence.
    pub compress_ratios: Vec<usize>,
    /// `hash_layer_count` — layers `< this` carry `ffn_gate_tid2eid` and NO `exp_probs_b`.
    pub hash_layer_count: usize,
    pub n_embd: usize,
    pub n_head: usize,
    /// `attention.head_count_kv`. V4 is MQA: one KV head serves every query head, which is why
    /// `attn_kv` is `[n_embd, head_dim]` rather than a per-head bank.
    pub n_kv: usize,
    /// `attention.key_length`. Deliberately NOT `n_embd / n_head`, so a loader that took the
    /// generic default instead of reading the key would size `attn_q_b`/`attn_kv` wrong.
    pub head_dim: usize,
    pub rope_dim: usize,
    /// `attention.sliding_window`. Every V4 layer uses it — `set_swa_pattern(0)`.
    pub swa_window: usize,
    pub vocab: usize,
    pub q_lora_rank: usize,
    /// `attention.output_group_count` / `attention.output_lora_rank` — the low-rank GROUPED output
    /// projection that replaces `attn_output`. `n_head * head_dim` must divide by the group count.
    pub o_group_count: usize,
    pub o_lora_rank: usize,
    /// `hyper_connection.count` — the number of parallel residual streams.
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    /// `attention.compress_rope_freq_base` — the rope base the COMPRESSED tiers use. Distinct from
    /// `rope.freq_base` so the two cannot be confused.
    pub compress_rope_theta: f32,
    pub n_expert: usize,
    pub n_used: usize,
    pub n_ff_exp: usize,
    pub n_expert_shared: usize,
    /// `swiglu_clamp_exp`, one entry per layer (the ARRAY form of the key).
    pub swiglu_clamp_exp: Vec<f32>,
    /// `swiglu_clamp_shexp`. `None` omits the key entirely, which is the case the reference falls
    /// back to `swiglu_clamp_exp` for.
    pub swiglu_clamp_shexp: Option<Vec<f32>>,
    /// `expert_gating_func`. V4 accepts only sqrt-softplus (4); a case below sets it to something
    /// else to pin the refusal.
    pub expert_gating_func: u32,
    /// The lightning indexer's three hyperparameters (model-level keys; tensors on ratio-4 layers).
    pub indexer: IndexerDims,
}

impl Dsv4Dims {
    pub fn n_layer(&self) -> usize {
        self.compress_ratios.len()
    }

    /// `hc_dim` — the width of a hyper-connection mixing matmul's input (all `hc_mult` streams
    /// concatenated).
    pub fn hc_dim(&self) -> usize {
        self.hc_mult * self.n_embd
    }

    /// `hc_mix_dim` — that matmul's output: the `pre` and `post` chunks (one per stream each) plus
    /// the `hc × hc` stream-mixing matrix.
    pub fn hc_mix_dim(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }

    pub fn shexp_ff(&self) -> usize {
        self.n_ff_exp * self.n_expert_shared
    }

    /// The compressor's channel multiplier for a layer at `ratio`: 2 for the CSA tier, 1 otherwise
    /// (`deepseek4.cpp`: `const int64_t coff = ratio == 4 ? 2 : 1`).
    pub fn coff(ratio: usize) -> usize {
        if ratio == 4 {
            2
        } else {
            1
        }
    }
}

/// The synthetic V4 model's dimensions.
///
/// The ratio sequence is the point of the fixture: **five layers covering every (compress_ratio,
/// routing) pair the loader branches on.** With `hash_layer_count = 2`:
///
/// | layer | ratio | tensors added by the ratio     | routing              |
/// | ----: | ----: | ------------------------------ | -------------------- |
/// |     0 |     0 | none                           | hash (`tid2eid`)     |
/// |     1 |     4 | compressor + indexer + its own | hash (`tid2eid`)     |
/// |     2 |   128 | compressor only                | bias (`exp_probs_b`) |
/// |     3 |     4 | compressor + indexer + its own | bias (`exp_probs_b`) |
/// |     4 |     0 | none                           | bias (`exp_probs_b`) |
///
/// A fixture where every layer had the same ratio would load identically whether the switch was
/// read or hard-coded to that one value; this one cannot.
pub fn dsv4_dims() -> Dsv4Dims {
    Dsv4Dims {
        compress_ratios: vec![0, 4, 128, 4, 0],
        hash_layer_count: 2,
        n_embd: 64,
        n_head: 2,
        n_kv: 1,
        head_dim: 48,
        rope_dim: 16,
        swa_window: 64,
        vocab: 64,
        q_lora_rank: 32,
        o_group_count: 2,
        o_lora_rank: 16,
        hc_mult: 2,
        hc_sinkhorn_iters: 2,
        hc_eps: 1e-6,
        compress_rope_theta: 40000.0,
        n_expert: 4,
        n_used: 2,
        n_ff_exp: 32,
        n_expert_shared: 1,
        // Per-layer, and all different, so a loader that broadcast entry 0 over the array would
        // land on values no layer but the first actually has.
        swiglu_clamp_exp: vec![7.0, 7.5, 8.0, 8.5, 9.0],
        swiglu_clamp_shexp: None,
        expert_gating_func: SQRT_SOFTPLUS,
        indexer: dsv4_indexer(),
    }
}

/// Token `id`'s text in the V4 fixture's vocabulary: the printable ASCII characters from `!` on,
/// each its own GPT-2 byte-level symbol, so with no merges llama.cpp's tokenizer reads a string of
/// them back as exactly these ids (the scheme `synthetic_lfm3.rs`'s `token_text` uses).
pub fn dsv4_token_text(id: u32) -> char {
    char::from(b'!' + u8::try_from(id).expect("a vocab id"))
}

/// `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS` — the only value `deepseek4.cpp` accepts.
pub const SQRT_SOFTPLUS: u32 = 4;

/// Build the whole GGUF description for a `deepseek4` model of `d`. Metadata keys are the ones
/// `deepseek4.cpp::load_arch_hparams` reads; tensor names and shapes are
/// `deepseek4.cpp::load_arch_tensors`', via `llama-arch.cpp`'s format strings.
///
/// A sibling of `mla_model` rather than a parameterisation of it — see synthetic_deepseek2.rs's
/// module doc.
pub fn dsv4_model(d: &Dsv4Dims) -> SyntheticModel {
    let arch = infr_llama::arch::DEEPSEEK4;
    assert!(
        (d.n_head * d.head_dim).is_multiple_of(d.o_group_count),
        "attention.output_group_count must divide n_head * key_length"
    );
    let u = |k: &str, v: usize| (format!("{arch}.{k}"), Meta::U32(v as u32));
    let f = |k: &str, v: f32| (format!("{arch}.{k}"), Meta::F32(v));
    let mut meta = vec![
        (
            "general.architecture".to_string(),
            Meta::Str(arch.to_string()),
        ),
        u("block_count", d.n_layer()),
        u("embedding_length", d.n_embd),
        u("attention.head_count", d.n_head),
        u("attention.head_count_kv", d.n_kv),
        u("attention.key_length", d.head_dim),
        // infr reads no V width (the cache row is its own V), but llama.cpp's
        // `build_attention_impl` asserts it equals the key length, as the real file declares.
        u("attention.value_length", d.head_dim),
        u("rope.dimension_count", d.rope_dim),
        f("rope.freq_base", 10000.0),
        u("context_length", 256),
        f("attention.layer_norm_rms_epsilon", 1e-5),
        u("attention.sliding_window", d.swa_window),
        u("attention.q_lora_rank", d.q_lora_rank),
        // The grouped low-rank output projection and the compressed tiers' rope.
        u("attention.output_group_count", d.o_group_count),
        u("attention.output_lora_rank", d.o_lora_rank),
        f("attention.compress_rope_freq_base", d.compress_rope_theta),
        (
            format!("{arch}.attention.compress_ratios"),
            Meta::I32Arr(d.compress_ratios.iter().map(|r| *r as i32).collect()),
        ),
        // Hyper-connections.
        u("hyper_connection.count", d.hc_mult),
        u("hyper_connection.sinkhorn_iterations", d.hc_sinkhorn_iters),
        f("hyper_connection.epsilon", d.hc_eps),
        // The lightning indexer (ratio-4 layers only, but the hparams are model-level).
        u("attention.indexer.head_count", d.indexer.n_head),
        u("attention.indexer.key_length", d.indexer.head_size),
        u("attention.indexer.top_k", d.indexer.top_k),
        // MoE. No `leading_dense_block_count`: V4 has no dense-lead layers at all, and no
        // `expert_group_count`: it does not use group-limited routing.
        u("hash_layer_count", d.hash_layer_count),
        u("expert_count", d.n_expert),
        u("expert_used_count", d.n_used),
        u("expert_feed_forward_length", d.n_ff_exp),
        u("expert_shared_count", d.n_expert_shared),
        u("expert_gating_func", d.expert_gating_func as usize),
        (format!("{arch}.expert_weights_norm"), Meta::Bool(true)),
        f("expert_weights_scale", 2.5),
        (
            format!("{arch}.swiglu_clamp_exp"),
            Meta::F32Arr(d.swiglu_clamp_exp.clone()),
        ),
        // The minimum tokenizer `build_tokenizer` accepts (`mla_model` in synthetic_deepseek2.rs
        // explains why), spelled so llama.cpp reads a prompt back as chosen ids: see
        // [`dsv4_token_text`]. No BOS, as on the real file, so a prompt is exactly its ids.
        (
            "tokenizer.ggml.model".to_string(),
            Meta::Str("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Meta::StrArr(
                (0..d.vocab as u32)
                    .map(|i| dsv4_token_text(i).to_string())
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
    if let Some(shexp) = &d.swiglu_clamp_shexp {
        meta.push((
            format!("{arch}.swiglu_clamp_shexp"),
            Meta::F32Arr(shexp.clone()),
        ));
    }
    meta.sort_by(|a, b| a.0.cmp(&b.0));

    let w = Fill::Rand(0.25);
    let ix = &d.indexer;
    let mut tensors = vec![
        TensorSpec::new("token_embd.weight", vec![d.n_embd, d.vocab], w.clone()),
        TensorSpec::new("output_norm.weight", vec![d.n_embd], Fill::Gain),
        TensorSpec::new("output.weight", vec![d.n_embd, d.vocab], w.clone()),
        // The hyper-connection HEAD: model-level, not per-layer (no `blk.` prefix), collapsing the
        // `hc_mult` residual streams back to one vector before `output_norm`.
        TensorSpec::new(
            "output_hc_fn.weight",
            vec![d.hc_dim(), d.hc_mult],
            w.clone(),
        ),
        TensorSpec::new("output_hc_base.weight", vec![d.hc_mult], Fill::Rand(0.1)),
        TensorSpec::new("output_hc_scale.weight", vec![1], Fill::Gain),
    ];
    for (l, &ratio) in d.compress_ratios.iter().enumerate() {
        let p = |s: &str| format!("blk.{l}.{s}");
        tensors.push(TensorSpec::new(
            p("attn_norm.weight"),
            vec![d.n_embd],
            Fill::Gain,
        ));
        // Attention sinks: one learned logit per query head, added to every row's softmax.
        tensors.push(TensorSpec::new(
            p("attn_sinks.weight"),
            vec![d.n_head],
            Fill::Rand(0.1),
        ));
        // The Q-LoRA triple — the one piece of deepseek2's attention V4 keeps.
        tensors.push(TensorSpec::new(
            p("attn_q_a.weight"),
            vec![d.n_embd, d.q_lora_rank],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("attn_q_a_norm.weight"),
            vec![d.q_lora_rank],
            Fill::Gain,
        ));
        tensors.push(TensorSpec::new(
            p("attn_q_b.weight"),
            vec![d.q_lora_rank, d.n_head * d.head_dim],
            w.clone(),
        ));
        // Single-head MQA KV: ONE key/value head for every query head.
        tensors.push(TensorSpec::new(
            p("attn_kv.weight"),
            vec![d.n_embd, d.head_dim],
            w.clone(),
        ));
        // `LLM_TENSOR_ATTN_KV_NORM`, which shares `blk.%d.attn_kv_a_norm` with deepseek2's
        // `LLM_TENSOR_ATTN_KV_A_NORM` — two enum values, one on-disk name.
        tensors.push(TensorSpec::new(
            p("attn_kv_a_norm.weight"),
            vec![d.head_dim],
            Fill::Gain,
        ));
        tensors.push(TensorSpec::new(
            p("attn_output_a.weight"),
            vec![
                d.n_head * d.head_dim / d.o_group_count,
                d.o_lora_rank * d.o_group_count,
            ],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("attn_output_b.weight"),
            vec![d.o_group_count * d.o_lora_rank, d.n_embd],
            w.clone(),
        ));
        for kind in ["attn", "ffn"] {
            tensors.push(TensorSpec::new(
                p(&format!("hc_{kind}_fn.weight")),
                vec![d.hc_dim(), d.hc_mix_dim()],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p(&format!("hc_{kind}_base.weight")),
                vec![d.hc_mix_dim()],
                Fill::Rand(0.1),
            ));
            tensors.push(TensorSpec::new(
                p(&format!("hc_{kind}_scale.weight")),
                vec![3],
                Fill::Gain,
            ));
        }
        if ratio != 0 {
            let coff = Dsv4Dims::coff(ratio);
            tensors.push(TensorSpec::new(
                p("attn_compressor_kv.weight"),
                vec![d.n_embd, coff * d.head_dim],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("attn_compressor_gate.weight"),
                vec![d.n_embd, coff * d.head_dim],
                w.clone(),
            ));
            // The absolute position-in-block embedding: one row per position within a block, which
            // is why `ratio` is a DIMENSION here and not just a switch.
            tensors.push(TensorSpec::new(
                p("attn_compressor_ape.weight"),
                vec![coff * d.head_dim, ratio],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("attn_compressor_norm.weight"),
                vec![d.head_dim],
                Fill::Gain,
            ));
        }
        if ratio == 4 {
            tensors.push(TensorSpec::new(
                p("indexer.proj.weight"),
                vec![d.n_embd, ix.n_head],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("indexer.attn_q_b.weight"),
                vec![d.q_lora_rank, ix.n_head * ix.head_size],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("indexer_compressor_kv.weight"),
                vec![d.n_embd, 2 * ix.head_size],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("indexer_compressor_gate.weight"),
                vec![d.n_embd, 2 * ix.head_size],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("indexer_compressor_ape.weight"),
                vec![2 * ix.head_size, ratio],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("indexer_compressor_norm.weight"),
                vec![ix.head_size],
                Fill::Gain,
            ));
        }
        // Every layer has a router; the first `hash_layer_count` replace the router BIAS with a
        // token-id → expert-id table.
        tensors.push(TensorSpec::new(
            p("ffn_gate_inp.weight"),
            vec![d.n_embd, d.n_expert],
            w.clone(),
        ));
        if l < d.hash_layer_count {
            // `[n_expert_used, n_vocab]` of PLAIN INTEGERS, i32 on disk — the real file's dtype
            // (see [`Fill::ExpertIds`]), not the f32 every other tensor here uses.
            tensors.push(TensorSpec::new(
                p("ffn_gate_tid2eid.weight"),
                vec![d.n_used, d.vocab],
                Fill::ExpertIds(d.n_expert),
            ));
        } else {
            tensors.push(TensorSpec::new(
                p("exp_probs_b.bias"),
                vec![d.n_expert],
                Fill::Rand(0.5),
            ));
        }
        tensors.push(TensorSpec::new(
            p("ffn_norm.weight"),
            vec![d.n_embd],
            Fill::Gain,
        ));
        tensors.push(TensorSpec::new(
            p("ffn_gate_exps.weight"),
            vec![d.n_embd, d.n_ff_exp, d.n_expert],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("ffn_up_exps.weight"),
            vec![d.n_embd, d.n_ff_exp, d.n_expert],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("ffn_down_exps.weight"),
            vec![d.n_ff_exp, d.n_embd, d.n_expert],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("ffn_gate_shexp.weight"),
            vec![d.n_embd, d.shexp_ff()],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("ffn_up_shexp.weight"),
            vec![d.n_embd, d.shexp_ff()],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("ffn_down_shexp.weight"),
            vec![d.shexp_ff(), d.n_embd],
            w.clone(),
        ));
    }
    SyntheticModel { meta, tensors }
}
