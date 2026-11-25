//! Synthetic-GGUF harness: a tiny but structurally complete `deepseek2` model, built in memory,
//! written to a temp file, and driven through the REAL loader (`Config::from_gguf`) and the REAL
//! seam — on CPU unconditionally, and on Vulkan behind `#[ignore]`.
//!
//! # Why this exists
//!
//! `docs/deepseek.md` § "Why this order": stages 3 (`deepseek32`/V3.2) and 4 (`deepseek4`/V4) have
//! **no model small enough to develop against** — V3.2 is 671B and V4-Flash's smallest quant is
//! 82.5 GB — so "stages 1–2 must leave behind MLA and MoE-routing pieces that are independently
//! tested". `docs/backlog.md` B46 names the two pieces stage 3 inherits verbatim that no test on a
//! real model reaches: **group-limited routing** (`n_expert_groups > 1`) and the **`exp_probs_b`
//! router bias**. V2-Lite — the only DeepSeek small enough to run here — ships
//! `expert_group_count = 1` and carries no bias tensor, so both branches were exercised only by
//! `seam_op_parity.rs`'s op-level `moe_groups_bias_parity`, never through a model load.
//!
//! This file closes that: the routing metadata and the bias tensor come off DISK, through
//! `Config::from_gguf` and the `wload`/`wpush`/emit lockstep in `seam/runner.rs`, into a real
//! prefill on both backends.
//!
//! # Adding an architecture
//!
//! The arch-agnostic part lives in `infr_testkit::gguf`: [`Meta`] (the GGUF value types it
//! writes), [`TensorSpec`] + [`Fill`] (name, ggml-order shape, deterministic values),
//! [`SyntheticModel`] (the whole file as data) and its writer, and [`TempGguf`].
//!
//! Stage 3 (`deepseek32`/V3.2) took that offer and found it could reuse more: V3.2 IS deepseek2's
//! absorbed MLA plus a lightning indexer, so rather than a second model function listing the same
//! thirty tensors, [`mla_model`] gained an `arch` parameter (every model key is `{arch}.`-prefixed)
//! and an optional [`IndexerDims`]. `synthetic_deepseek32_is_deepseek2_plus_the_indexer` then
//! ASSERTS the containment the sharing assumes. A future arch that is not a DeepSeek-MLA variant
//! should write its own function instead — the `infr_testkit::gguf` harness is still what it reuses.
//!
//! Stage 4 (`deepseek4`/V4) is that arch, and it took the second offer: [`dsv4_model`] is a sibling
//! of [`mla_model`], not a parameterisation of it. V4 is not MLA — no `kv_lora_rank`, no
//! `wk_b`/`wv_b` — and its per-layer tensor set is decided by `compress_ratios[il]`, so threading it
//! through the shared builder would mean a parameter per difference and an `Option` on most of the
//! shapes. What it DOES reuse is that harness plus the damaged-fixture helpers
//! ([`config_err`], [`without_tensor`], [`without_meta`]) and [`PROMPT`]. It lives in
//! `dsv4_fixture/mod.rs`, because the llama.cpp oracle in `synthetic_deepseek4_llama_cpp.rs`
//! builds the same file.
//!
//! The fill is seeded by the tensor NAME, not by its position in the file, which is what makes the
//! differential tests below valid: adding or removing `exp_probs_b.bias` changes that tensor and
//! nothing else, so two variants differ only in the thing under test.
//!
//! ```text
//! cargo test -p infr-llama --test synthetic_deepseek2
//! cargo test -p infr-llama --test synthetic_deepseek2 -- --include-ignored   # + Vulkan
//! ```

mod dsv4_fixture;

use dsv4_fixture::{dsv4_dims, dsv4_indexer, dsv4_model, Dsv4Dims, IndexerDims};
use infr_core::WeightSource;
use infr_testkit::gguf::{Fill, Meta, SyntheticModel, TempGguf, TensorSpec};

// ─── the deepseek2 model description ─────────────────────────────────────────────

/// Every dimension of the synthetic `deepseek2` model. The MLA relationships `config.rs` derives
/// are spelled out on the fields that bind them; violating one is refused by the loader, which
/// `deepseek2_mla_head_dims_match_reference` (config.rs) already pins.
#[derive(Clone, Debug)]
struct Ds2Dims {
    n_layer: usize,
    /// `nextn_predict_layers` — trailing NextN/MTP blocks that `block_count` INCLUDES and that the
    /// trunk loop must not walk. `0` for every deepseek2 case (V2 ships none).
    n_layer_nextn: usize,
    /// `attention.layer_norm_rms_epsilon`. Deliberately NOT `deepseek32`'s hardcoded LayerNorm
    /// epsilon, so `Config::norm_eps` and `Config::rms_eps` can be told apart.
    rms_eps: f32,
    /// `leading_dense_block_count` — layers `< this` run a plain dense SwiGLU at `n_ff`; the rest
    /// are MoE. DeepSeek's "first N dense, rest MoE" threshold, not a periodic step.
    n_dense_lead: usize,
    n_embd: usize,
    n_head: usize,
    n_ff: usize,
    vocab: usize,
    /// Non-zero ⇒ the `wq_a → q_a_norm → wq_b` LoRA query path. V2-Lite is the LITE variant
    /// (direct `attn_q`), so this is the branch a real model has never exercised here — and it is
    /// the only one stage 3 has (`deepseek32` makes `q_lora_rank` mandatory).
    q_lora_rank: usize,
    kv_lora_rank: usize,
    /// `rope.dimension_count` — the decoupled-rope width, shared by `q_pe` and the single `k_pe`.
    qk_rope_dim: usize,
    /// `attention.key_length_mla`; `head_k_mla` (the NOPE width) is this MINUS `qk_rope_dim`.
    key_length_mla: usize,
    /// `attention.value_length_mla`, which IS `v_head_dim`.
    value_length_mla: usize,
    n_expert: usize,
    n_used: usize,
    n_ff_exp: usize,
    /// `expert_shared_count`; the shared expert's width is `n_ff_exp * this` (llama.cpp fuses the
    /// shared experts into one wider branch).
    n_expert_shared: usize,
    /// `expert_group_count` / `expert_group_used_count`. `n_expert` must divide by the count.
    n_groups: usize,
    n_groups_used: usize,
    /// `blk.*.exp_probs_b.bias` — omitted entirely when `None`, like every V2 GGUF.
    exp_probs_b: Option<Vec<f32>>,
}

impl Ds2Dims {
    /// The NOPE width, derived exactly as `config.rs` does.
    fn qk_nope(&self) -> usize {
        self.key_length_mla - self.qk_rope_dim
    }

    /// The cache row width: ONE row per token per layer, `[latent | rope]`. V is a prefix VIEW of
    /// it, never a second cache.
    fn kv_row(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_dim
    }

    fn shexp_ff(&self) -> usize {
        self.n_ff_exp * self.n_expert_shared
    }
}

/// The synthetic model's dimensions. Small, but every MLA relationship holds and every width the
/// Vulkan kernels constrain is legal: the expert id-GEMV decodes 32-element sub-blocks, so `n_embd`
/// and `n_ff_exp` are ≥ 32 and multiples of 32; `mla.comp` packs the f16 KV row two-per-`uint`, so
/// `kv_row()` is even. `qk_nope`, `kv_lora_rank` and `value_length_mla` are deliberately three
/// DIFFERENT numbers, so `wk_b` `[qk_nope, kv_lora, n_head]` and `wv_b` `[kv_lora, v_head, n_head]`
/// have different shapes — swapping them (the classic MLA porting bug, `docs/deepseek.md` § Stage
/// 2) cannot go unnoticed the way it would if the dims coincided as they do on V2-Lite/V3.
fn ds2_dims(n_groups: usize, n_groups_used: usize, exp_probs_b: Option<Vec<f32>>) -> Ds2Dims {
    Ds2Dims {
        n_layer: 3,
        n_layer_nextn: 0,
        rms_eps: 1e-6,
        n_dense_lead: 1,
        n_embd: 64,
        n_head: 2,
        n_ff: 64,
        vocab: 64,
        q_lora_rank: 32,
        kv_lora_rank: 32,
        qk_rope_dim: 16,
        key_length_mla: 64, // ⇒ qk_nope = 48
        value_length_mla: 48,
        n_expert: N_EXPERT,
        n_used: N_USED,
        n_ff_exp: 32,
        n_expert_shared: 1,
        n_groups,
        n_groups_used,
        exp_probs_b,
    }
}

/// Build the whole GGUF description for an absorbed-MLA DeepSeek model of `d` under `arch`
/// (`deepseek2` or `deepseek32` — every model key is `{arch}.`-prefixed, which is the whole reason
/// the arch string is a parameter rather than a literal).
///
/// `indexer` is `Some` only for `deepseek32`: it adds the three `attention.indexer.*` keys and five
/// per-layer tensors and changes NOTHING else, because V3.2 genuinely is V2's absorbed MLA plus the
/// lightning indexer. Keeping one builder is what makes "the deepseek32 model is the deepseek2 one
/// plus the indexer" a property the file can assert rather than a claim in a comment.
fn mla_model(arch: &str, d: &Ds2Dims, indexer: Option<&IndexerDims>) -> SyntheticModel {
    assert!(
        d.n_expert.is_multiple_of(d.n_groups.max(1)),
        "expert_group_count must divide expert_count"
    );
    let u = |k: &str, v: usize| (format!("{arch}.{k}"), Meta::U32(v as u32));
    let f = |k: &str, v: f32| (format!("{arch}.{k}"), Meta::F32(v));
    let mut meta = vec![
        (
            "general.architecture".to_string(),
            Meta::Str(arch.to_string()),
        ),
        // `block_count` COUNTS the NextN blocks; the trunk is `block_count - nextn_predict_layers`.
        u("block_count", d.n_layer + d.n_layer_nextn),
        u("embedding_length", d.n_embd),
        u("feed_forward_length", d.n_ff),
        u("attention.head_count", d.n_head),
        u("context_length", 256),
        f("attention.layer_norm_rms_epsilon", d.rms_eps),
        // MLA geometry.
        u("attention.q_lora_rank", d.q_lora_rank),
        u("attention.kv_lora_rank", d.kv_lora_rank),
        u("attention.key_length_mla", d.key_length_mla),
        u("attention.value_length_mla", d.value_length_mla),
        u("rope.dimension_count", d.qk_rope_dim),
        f("rope.freq_base", 10000.0),
        // YaRN, in the shape the V2-Lite GGUF declares it (`docs/deepseek.md` open question 6):
        // `type = yarn` is what makes llama.cpp run the FULL ramp at every context length, and the
        // convert script writes `0.1 * mscale_all_dim`, which the loader divides back out.
        (
            format!("{arch}.rope.scaling.type"),
            Meta::Str("yarn".to_string()),
        ),
        f("rope.scaling.factor", 40.0),
        u("rope.scaling.original_context_length", 128),
        f("rope.scaling.yarn_log_multiplier", 0.0707),
        // MoE.
        u("expert_count", d.n_expert),
        u("expert_used_count", d.n_used),
        u("expert_feed_forward_length", d.n_ff_exp),
        u("expert_shared_count", d.n_expert_shared),
        u("leading_dense_block_count", d.n_dense_lead),
        u("expert_gating_func", 2), // sigmoid — V3's scoring_func
        (format!("{arch}.expert_weights_norm"), Meta::Bool(true)),
        f("expert_weights_scale", 2.5),
        u("expert_group_count", d.n_groups),
        u("expert_group_used_count", d.n_groups_used),
        // The minimum tokenizer `build_tokenizer` accepts: a gpt2-model vocab. `.merges` may be
        // empty (nothing here encodes text — the tests hand token ids straight to the seam), but
        // the key must exist or the build fails. `.token_type` is optional and omitted.
        (
            "tokenizer.ggml.model".to_string(),
            Meta::Str("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Meta::StrArr((0..d.vocab).map(|i| format!("t{i}")).collect()),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            Meta::StrArr(Vec::new()),
        ),
        ("tokenizer.ggml.eos_token_id".to_string(), Meta::U32(2)),
    ];
    if d.n_layer_nextn > 0 {
        meta.push(u("nextn_predict_layers", d.n_layer_nextn));
    }
    if let Some(ix) = indexer {
        meta.push(u("attention.indexer.head_count", ix.n_head));
        meta.push(u("attention.indexer.key_length", ix.head_size));
        meta.push(u("attention.indexer.top_k", ix.top_k));
    }
    meta.sort_by(|a, b| a.0.cmp(&b.0));

    let w = Fill::Rand(0.25);
    let mut tensors = vec![
        TensorSpec::new("token_embd.weight", vec![d.n_embd, d.vocab], w.clone()),
        TensorSpec::new("output_norm.weight", vec![d.n_embd], Fill::Gain),
        TensorSpec::new("output.weight", vec![d.n_embd, d.vocab], w.clone()),
    ];
    for l in 0..d.n_layer {
        let p = |s: &str| format!("blk.{l}.{s}");
        tensors.push(TensorSpec::new(
            p("attn_norm.weight"),
            vec![d.n_embd],
            Fill::Gain,
        ));
        // MLA, non-lite: the LoRA query path plus the compressed KV path.
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
            vec![d.q_lora_rank, d.n_head * d.key_length_mla],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("attn_kv_a_mqa.weight"),
            vec![d.n_embd, d.kv_row()],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("attn_kv_a_norm.weight"),
            vec![d.kv_lora_rank],
            Fill::Gain,
        ));
        // `wk_b` is TRANSPOSED relative to the HF weight and `wv_b` is not — the conversion script
        // calls `.transpose(1, 2)` on `k_b` only (docs/deepseek.md § Stage 2).
        tensors.push(TensorSpec::new(
            p("attn_k_b.weight"),
            vec![d.qk_nope(), d.kv_lora_rank, d.n_head],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("attn_v_b.weight"),
            vec![d.kv_lora_rank, d.value_length_mla, d.n_head],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("attn_output.weight"),
            vec![d.n_head * d.value_length_mla, d.n_embd],
            w.clone(),
        ));
        // The lightning indexer sits on EVERY layer, dense-lead included — `deepseek32.cpp` creates
        // these five outside the dense/MoE branch. `k_norm` carries a weight AND a bias under one
        // GGUF name because it is a mean-centred LayerNorm, not an RMSNorm.
        if let Some(ix) = indexer {
            tensors.push(TensorSpec::new(
                p("indexer.k_norm.weight"),
                vec![ix.head_size],
                Fill::Gain,
            ));
            tensors.push(TensorSpec::new(
                p("indexer.k_norm.bias"),
                vec![ix.head_size],
                Fill::Rand(0.1),
            ));
            tensors.push(TensorSpec::new(
                p("indexer.proj.weight"),
                vec![d.n_embd, ix.n_head],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("indexer.attn_k.weight"),
                vec![d.n_embd, ix.head_size],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("indexer.attn_q_b.weight"),
                vec![d.q_lora_rank, ix.n_head * ix.head_size],
                w.clone(),
            ));
        }
        tensors.push(TensorSpec::new(
            p("ffn_norm.weight"),
            vec![d.n_embd],
            Fill::Gain,
        ));
        if l < d.n_dense_lead {
            tensors.push(TensorSpec::new(
                p("ffn_gate.weight"),
                vec![d.n_embd, d.n_ff],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("ffn_up.weight"),
                vec![d.n_embd, d.n_ff],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("ffn_down.weight"),
                vec![d.n_ff, d.n_embd],
                w.clone(),
            ));
            continue;
        }
        tensors.push(TensorSpec::new(
            p("ffn_gate_inp.weight"),
            vec![d.n_embd, d.n_expert],
            w.clone(),
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
        if let Some(bias) = &d.exp_probs_b {
            tensors.push(TensorSpec::new(
                p("exp_probs_b.bias"),
                vec![d.n_expert],
                Fill::Exact(bias.clone()),
            ));
        }
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

/// [`mla_model`] as `deepseek2` — the arch every test above this line uses.
fn deepseek2_model(d: &Ds2Dims) -> SyntheticModel {
    mla_model(infr_llama::arch::DEEPSEEK2, d, None)
}

// ─── the routing cases ───────────────────────────────────────────────────────────

const N_EXPERT: usize = 16;
const N_USED: usize = 2;
const N_GROUPS: usize = 4;
const N_GROUPS_USED: usize = 2;

/// The router bias that FORCES group-limited routing to disagree with a flat top-k, for every
/// token, whatever the router logits are.
///
/// Gating is sigmoid, so every unbiased prob `p` is strictly inside `(0, 1)`; the selection score
/// is `p + bias` (bias affects SELECTION only). With four groups of four:
///
/// | group | experts | biased scores                | group score = top-2 sum |
/// | ----: | ------- | ---------------------------- | ----------------------- |
/// |     0 | 0–3     | e0 ∈ (9,10), rest ∈ (0,1)    | (9, 11)                 |
/// |     1 | 4–7     | e4,e5 ∈ (8,9), rest ∈ (0,1)  | (16, 18)                |
/// |     2 | 8–11    | e8,e9 ∈ (7,8), rest ∈ (0,1)  | (14, 16)                |
/// |     3 | 12–15   | all ∈ (0,1)                  | (0, 2)                  |
///
/// The four ranges are disjoint, so the group order is `1 > 2 > 0 > 3` unconditionally. The top
/// `expert_group_used_count = 2` groups are therefore `{1, 2}` — and group 0, which holds the
/// single highest-scoring expert in the whole layer, is MASKED OUT. So:
///
/// * grouped `top-2` = `{4, 5}` (their scores dominate e8/e9 and everything unbiased),
/// * flat `top-2` = `{0, 4 or 5}`.
///
/// **Expert 0 is routed if and only if group routing is off.** That is what makes
/// `synthetic_deepseek2_group_routing_changes_output` a test of the group mask rather than of the
/// weather: it cannot pass with `n_expert_groups = 1`, and it cannot pass if the mask is dropped.
///
/// It also pins the routing SET across backends, so the CPU-vs-Vulkan check measures arithmetic
/// rather than a routing near-tie flipping.
const FORCE_BIAS: [f32; N_EXPERT] = [
    9.0, 0.0, 0.0, 0.0, //
    8.0, 8.0, 0.0, 0.0, //
    7.0, 7.0, 0.0, 0.0, //
    0.0, 0.0, 0.0, 0.0,
];

/// A bias that is the SAME on every expert. It cannot change any selection — a flat top-k order is
/// invariant under a uniform shift, and every group's top-2 sum shifts by exactly `2×` the same
/// constant — so a correct implementation, which reads the returned weights from the UNBIASED
/// probs, must produce byte-identical logits to a model with no bias tensor at all. An
/// implementation that weights from the biased probs cannot: `(p+c)/Σ(p+c) ≠ p/Σp`.
const UNIFORM_BIAS: f32 = 0.25;

/// The prompt every case is prefilled with: raw token ids, no tokenizer round-trip (the synthetic
/// vocab spells nothing).
const PROMPT: &[u32] = &[3, 11, 5, 40, 27, 8, 61, 14];

fn force_bias() -> Vec<f32> {
    FORCE_BIAS.to_vec()
}

/// The canonical model: V3's routing shape (group-limited routing ON, a non-uniform router bias) at
/// toy dimensions.
fn grouped_model() -> SyntheticModel {
    deepseek2_model(&ds2_dims(N_GROUPS, N_GROUPS_USED, Some(force_bias())))
}

/// The canonical model with group routing DISABLED, exactly as a V2-era GGUF declares it
/// (`expert_group_count = 1`). Byte-identical in every other respect, bias included.
fn flat_model() -> SyntheticModel {
    deepseek2_model(&ds2_dims(1, 1, Some(force_bias())))
}

/// The canonical model with a UNIFORM bias — see [`UNIFORM_BIAS`].
fn uniform_bias_model() -> SyntheticModel {
    deepseek2_model(&ds2_dims(
        N_GROUPS,
        N_GROUPS_USED,
        Some(vec![UNIFORM_BIAS; N_EXPERT]),
    ))
}

/// The canonical model with NO `exp_probs_b` tensor at all — what V2-Lite ships.
fn no_bias_model() -> SyntheticModel {
    deepseek2_model(&ds2_dims(N_GROUPS, N_GROUPS_USED, None))
}

// ─── running them ────────────────────────────────────────────────────────────────

/// Serialize the Vulkan tests against each other: each `prefill_logits_vulkan` opens its own
/// device session, and cargo runs a binary's tests in parallel. Mirrors `cpu_backend.rs`'s
/// `test_serial_lock`, and poison-tolerant for the same reason.
fn gpu_serial_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The shipped defaults plus the GPU knobs of the environment: which GPU the Vulkan cases run on
/// (`INFR_DEV`; unset keeps the backend's default) and [`infr_testkit::honour_env`]'s.
fn env_cfg() -> infr_llama::EngineConfig {
    let env = infr_llama::EngineConfig::load_from_layers(&[
        infr_core::config::ConfigLayer::env().expect("test environment config")
    ]);
    let mut cfg = infr_llama::EngineConfig::default();
    cfg.device.dev = env.device.dev;
    infr_testkit::honour_env(&mut cfg);
    cfg
}

fn load_with(tmp: &TempGguf, cfg: infr_llama::EngineConfig) -> infr_llama::SeamModel {
    infr_llama::SeamModel::load_with(tmp.path(), None, std::sync::Arc::new(cfg))
        .expect("synthetic model load")
}

fn load(tmp: &TempGguf) -> infr_llama::SeamModel {
    load_with(tmp, env_cfg())
}

/// Prefill `prompt` on the CPU reference backend and return the last row's logits.
fn cpu_logits_of(tag: &str, model: &SyntheticModel, prompt: &[u32]) -> Vec<f32> {
    let tmp = TempGguf::write(tag, model);
    load(&tmp).prefill_logits_cpu(prompt).expect("cpu prefill")
}

/// [`cpu_logits_of`] at the shared [`PROMPT`].
fn cpu_logits(tag: &str, model: &SyntheticModel) -> Vec<f32> {
    cpu_logits_of(tag, model, PROMPT)
}

/// [`cpu_logits`]'s Vulkan twin.
fn vulkan_logits(tag: &str, model: &SyntheticModel) -> Vec<f32> {
    let tmp = TempGguf::write(tag, model);
    load(&tmp)
        .prefill_logits_vulkan(PROMPT)
        .expect("vulkan prefill")
}

/// Max |a - b|, NaN if any difference is NaN: `f32::max` returns its non-NaN operand, so folding
/// it would score an all-NaN output as an exact match.
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

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / v.len() as f64).sqrt() as f32
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    dot / (na * nb)
}

/// Assert the two runs produced materially different logits: a difference that is a real fraction
/// of the signal, not float noise on an identical computation. `mechanism` names what the two runs
/// were built to separate, so the failure says which code path did not execute.
fn assert_moves(what: &str, a: &[f32], b: &[f32], mechanism: &str) {
    let d = max_abs_diff(a, b);
    let scale = rms(a).max(rms(b));
    println!("{what}: max|Δ| = {d:e}  (logit rms {scale:e})");
    assert!(
        d > 0.01 * scale,
        "{what}: the two runs are indistinguishable (max|Δ| = {d:e}, logit rms {scale:e}) — \
         {mechanism} did not execute"
    );
}

/// Assert the two runs routed differently.
fn assert_routing_differs(what: &str, a: &[f32], b: &[f32]) {
    assert_moves(what, a, b, "the routing path under test");
}

// ─── tests ───────────────────────────────────────────────────────────────────────

/// Every variant's tensors as `(name, values)`, for comparing two variants' weights directly.
fn weights_of(m: &SyntheticModel) -> Vec<(String, Vec<f32>)> {
    m.tensors
        .iter()
        .map(|t| (t.name.clone(), t.values()))
        .collect()
}

/// The premise every differential test below rests on: two variants differ in exactly the thing
/// under test and in nothing else. If the fill ever became order- or run-dependent — a `HashMap`
/// walk, a clock seed — `grouped vs flat` would still "differ", but for the wrong reason, and
/// nothing else here would notice.
#[test]
fn synthetic_deepseek2_variants_differ_only_in_the_thing_under_test() {
    // Same description twice ⇒ byte-identical files.
    assert_eq!(
        grouped_model().to_gguf_bytes(),
        grouped_model().to_gguf_bytes(),
        "the synthetic GGUF is not reproducible"
    );

    // Grouped vs flat: identical tensors end to end; only the group metadata moves.
    assert_eq!(
        weights_of(&grouped_model()),
        weights_of(&flat_model()),
        "grouped and flat must share every weight — only expert_group_count differs"
    );
    let (gm, fm) = (grouped_model(), flat_model());
    let diff: Vec<_> = gm
        .meta
        .iter()
        .zip(&fm.meta)
        .filter(|(a, b)| a != b)
        .map(|(a, _)| a.0.clone())
        .collect();
    assert_eq!(
        diff,
        vec![
            "deepseek2.expert_group_count".to_string(),
            "deepseek2.expert_group_used_count".to_string()
        ],
        "grouped and flat must differ in the group metadata and nothing else"
    );

    // Uniform-bias vs no-bias: same metadata, and the ONLY tensor difference is the bias itself.
    let (um, nm) = (uniform_bias_model(), no_bias_model());
    assert_eq!(um.meta, nm.meta, "the bias is a TENSOR, not metadata");
    let (uw, nw) = (weights_of(&um), weights_of(&nm));
    let extra: Vec<_> = uw
        .iter()
        .filter(|(n, _)| !nw.iter().any(|(m, _)| m == n))
        .map(|(n, _)| n.clone())
        .collect();
    assert_eq!(
        extra,
        vec![
            "blk.1.exp_probs_b.bias".to_string(),
            "blk.2.exp_probs_b.bias".to_string()
        ],
        "the no-bias variant must be the uniform-bias one minus exactly the bias tensors"
    );
    for (n, v) in &nw {
        let (_, u) = uw.iter().find(|(m, _)| m == n).expect("shared tensor");
        assert_eq!(u, v, "{n} must be identical across the two bias variants");
    }
}

/// The harness itself: the bytes it writes must survive `Gguf::open` and be read back by the REAL
/// loader as the model that was described. Without this, every test below could be asserting about
/// a file whose dims silently drifted from the ones the comments reason over.
#[test]
fn synthetic_deepseek2_round_trips_through_the_real_loader() {
    let d = ds2_dims(N_GROUPS, N_GROUPS_USED, Some(force_bias()));
    let tmp = TempGguf::write("roundtrip", &deepseek2_model(&d));
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");

    // Tensor directory: names, shapes and dtypes are what was written.
    let find = |name: &str| {
        g.tensors()
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} missing from the synthetic GGUF"))
    };
    assert_eq!(find("token_embd.weight").shape, vec![d.n_embd, d.vocab]);
    assert_eq!(
        find("blk.1.attn_k_b.weight").shape,
        vec![d.qk_nope(), d.kv_lora_rank, d.n_head],
        "wk_b is [qk_nope, kv_lora, n_head] on disk"
    );
    assert_eq!(
        find("blk.1.attn_v_b.weight").shape,
        vec![d.kv_lora_rank, d.value_length_mla, d.n_head],
        "wv_b is [kv_lora, v_head_dim, n_head] on disk — NOT wk_b's orientation"
    );
    assert_eq!(find("blk.1.exp_probs_b.bias").shape, vec![d.n_expert]);
    assert!(
        !g.tensors()
            .iter()
            .any(|t| t.name == "blk.0.exp_probs_b.bias"),
        "layer 0 is dense-lead: no router, no bias"
    );
    assert!(
        !g.tensors().iter().any(|t| t.name == "blk.0.attn_q.weight"),
        "the synthetic model is NON-lite (wq_a/q_a_norm/wq_b), the branch V2-Lite never takes"
    );
    assert_eq!(find("blk.0.attn_k_b.weight").dtype, infr_core::DType::F32);

    // The bias tensor's VALUES survive the write — the whole forcing argument rests on them.
    let raw = g
        .tensor_bytes_arc("blk.2.exp_probs_b.bias")
        .expect("bias bytes");
    let got: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect();
    assert_eq!(got, FORCE_BIAS.to_vec());

    // And the loader derives the config the tests reason about.
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    assert!(cfg.deepseek2);
    assert!(!cfg.is_lite);
    assert_eq!(cfg.n_layer, d.n_layer);
    assert_eq!(cfg.vocab, d.vocab);
    assert_eq!(cfg.head_k_mla, d.qk_nope());
    assert_eq!(cfg.v_head_dim, d.value_length_mla);
    assert_eq!(cfg.kv_lora_rank, d.kv_lora_rank);
    assert_eq!(cfg.qk_rope_dim, d.qk_rope_dim);
    assert_eq!(cfg.head_dim, d.key_length_mla);
    assert_eq!(cfg.n_kv, 1, "MLA caches one key head for every query head");
    assert_eq!(cfg.q_lora_rank, d.q_lora_rank);
    assert_eq!(cfg.n_layer_dense_lead, d.n_dense_lead);
    assert!(!cfg.is_moe_layer(0), "layer 0 is the dense lead");
    assert!(cfg.is_moe_layer(1) && cfg.is_moe_layer(2));
    assert_eq!(cfg.shexp_ff, d.shexp_ff());
    assert!(cfg.rope_scaling_yarn);
    // The convert script writes `0.1 * mscale_all_dim`; the loader divides it back out.
    assert!((cfg.rope_yarn_log_mul - 0.707).abs() < 1e-4);
    let moe = cfg.moe.expect("the synthetic model has experts");
    assert_eq!(moe.n_expert, N_EXPERT);
    assert_eq!(moe.n_used, N_USED);
    assert_eq!(moe.n_expert_groups, N_GROUPS as u32);
    assert_eq!(moe.n_expert_groups_used, N_GROUPS_USED as u32);
    assert_eq!(moe.gating, infr_core::graph::MoeGating::Sigmoid);
    assert!(moe.norm_w);
    assert_eq!(moe.scale, 2.5);
}

/// The whole model runs: MLA (non-lite LoRA query path, YaRN ramp, compressed single-row KV) plus a
/// dense-lead layer and two group-routed MoE layers, end to end on the CPU reference backend.
#[test]
fn synthetic_deepseek2_cpu_prefill_is_finite() {
    let logits = cpu_logits("finite", &grouped_model());
    assert_eq!(logits.len(), 64, "logits are [vocab]");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "non-finite logit in the synthetic deepseek2 CPU prefill: {logits:?}"
    );
    assert!(rms(&logits) > 1e-3, "logits are degenerate: {logits:?}");
}

/// **Group-limited routing executes and changes the answer.** The same weights, the same bias, the
/// same prompt — only `expert_group_count`/`expert_group_used_count` differ. By [`FORCE_BIAS`]'s
/// construction the grouped run routes `{4, 5}` and the flat run routes `{0, 4|5}` for every token,
/// so the outputs cannot agree unless the group mask never ran.
#[test]
fn synthetic_deepseek2_group_routing_changes_output() {
    let grouped = cpu_logits("grouped", &grouped_model());
    let flat = cpu_logits("flat", &flat_model());
    assert_routing_differs("cpu grouped vs flat", &grouped, &flat);
}

/// **The router bias participates in selection.** Group routing is on in both runs; only the bias
/// tensor differs (present-and-forcing vs absent). With the bias, `{4, 5}` are routed on every
/// token; without it the router's own probs decide.
///
/// This is also the only proof that the LOAD path ran: `wload`'s `exp_probs_b` arm is
/// presence-gated (`layer_has_epb`), so a tensor it failed to find would leave `exp_probs_b: None`
/// on the op and make this run identical to the no-bias one — which is exactly the failure this
/// asserts against.
#[test]
fn synthetic_deepseek2_router_bias_changes_output() {
    let biased = cpu_logits("biased", &grouped_model());
    let unbiased = cpu_logits("unbiased", &no_bias_model());
    assert_routing_differs("cpu forcing-bias vs no-bias", &biased, &unbiased);
}

/// **The bias affects SELECTION only; the weights come from the unbiased probs.** A uniform bias
/// cannot move any selection (see [`UNIFORM_BIAS`]), so the logits must match a no-bias model
/// EXACTLY. They do not if the returned weight is read from the biased probs — that is the one
/// perturbation this case exists to catch, and the reason it asserts equality rather than
/// difference.
#[test]
fn synthetic_deepseek2_uniform_router_bias_is_a_no_op() {
    let uniform = cpu_logits("uniform-bias", &uniform_bias_model());
    let none = cpu_logits("no-bias", &no_bias_model());
    let d = max_abs_diff(&uniform, &none);
    println!("cpu uniform-bias vs no-bias: max|Δ| = {d:e}");
    assert_eq!(
        d, 0.0,
        "a uniform router bias changed the output — the returned expert weights are being read \
         from the BIASED probs instead of the unbiased ones"
    );
}

/// One dense MLA layer and nothing else — the shape the residual test below can compute by hand.
/// `n_dense_lead = n_layer = 1` keeps the single layer off the MoE branch, and the MoE metadata
/// stays declared (no layer reads it) so the file is still an ordinary deepseek2.
fn ds2_one_dense_layer_dims() -> Ds2Dims {
    Ds2Dims {
        n_layer: 1,
        n_dense_lead: 1,
        ..ds2_dims(1, 1, None)
    }
}

/// **The attention sublayer enters the residual stream exactly ONCE.**
///
/// A SINGLE-token prefill makes a whole MLA layer closed-form: with one key the softmax is `1`
/// whatever the query is, so the attention output is just `wv_b · kv_cmpr` and every query, rope,
/// mask and score term drops out. The rest of the layer is a norm, a matvec and a SwiGLU. So this
/// test recomputes the forward itself and compares — which is the ONLY way to catch a doubled
/// residual: adding `wo·attn` to the stream twice is exactly a model whose `attn_output.weight` is
/// scaled by 2, so no test that varies weights and compares outputs can tell the two apart. The
/// emit did add it twice for the whole life of `MixerW::Mla`, on both backends, and nothing here
/// or in `cpu_backend.rs` noticed — it cost V2-Lite 0.33 probability cosine against llama.cpp at a
/// ten-token prompt, and nothing short of an external oracle flagged it.
///
/// The tolerance is set by the KV cache, which is f16: `kv_cmpr` makes the round trip through it
/// before `Op::Mla` reads it back as V. Both bounds below were measured, and the doubled-residual
/// arm is asserted too — a reference that matched BOTH would be measuring nothing.
#[test]
fn synthetic_deepseek2_attention_enters_the_residual_once() {
    let d = ds2_one_dense_layer_dims();
    let model = deepseek2_model(&d);
    let w: std::collections::HashMap<String, Vec<f32>> = weights_of(&model).into_iter().collect();
    let get = |n: &str| {
        w.get(n)
            .unwrap_or_else(|| panic!("{n} is not in the model"))
            .as_slice()
    };

    let (ne, nff, eps) = (d.n_embd, d.n_ff, d.rms_eps);
    let (kl, vh) = (d.kv_lora_rank, d.value_length_mla);
    let tok = 3usize;

    // GGUF stores a matrix `[in, out]` with `in` fastest, so it is out-major rows of `in` elements
    // — the same indexing `ggml_mul_mat(w, x)` reads.
    let mv = |wm: &[f32], x: &[f32], inf: usize, outf: usize| -> Vec<f32> {
        assert_eq!(wm.len(), inf * outf, "matvec shape");
        assert_eq!(x.len(), inf, "matvec input shape");
        (0..outf)
            .map(|o| (0..inf).map(|i| wm[o * inf + i] * x[i]).sum())
            .collect()
    };
    let norm = |x: &[f32], g: &[f32]| -> Vec<f32> {
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let s = 1.0 / (ss + eps).sqrt();
        x.iter().zip(g).map(|(v, gain)| v * s * gain).collect()
    };

    let h = get("token_embd.weight")[tok * ne..][..ne].to_vec();
    let x = norm(&h, get("blk.0.attn_norm.weight"));
    let kv = mv(get("blk.0.attn_kv_a_mqa.weight"), &x, ne, d.kv_row());
    let c = norm(&kv[..kl], get("blk.0.attn_kv_a_norm.weight"));
    // One key ⇒ the attention weight is 1 ⇒ each head's output is its slice of `wv_b` times the
    // compressed latent. `wv_b` is `[kv_lora, v_head, n_head]`, so `(head, v)` is row `head*vh + v`.
    let vb = get("blk.0.attn_v_b.weight");
    let attn: Vec<f32> = (0..d.n_head * vh)
        .map(|r| (0..kl).map(|l| vb[r * kl + l] * c[l]).sum())
        .collect();
    let sub = mv(get("blk.0.attn_output.weight"), &attn, d.n_head * vh, ne);

    // Everything downstream of the attention residual, so both arms below go through it.
    let rest = |resid: &[f32]| -> Vec<f32> {
        let y = norm(resid, get("blk.0.ffn_norm.weight"));
        let gate = mv(get("blk.0.ffn_gate.weight"), &y, ne, nff);
        let up = mv(get("blk.0.ffn_up.weight"), &y, ne, nff);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let down = mv(get("blk.0.ffn_down.weight"), &act, nff, ne);
        let out: Vec<f32> = resid.iter().zip(&down).map(|(a, b)| a + b).collect();
        let o = norm(&out, get("output_norm.weight"));
        mv(get("output.weight"), &o, ne, d.vocab)
    };
    let once = rest(&h.iter().zip(&sub).map(|(a, b)| a + b).collect::<Vec<_>>());
    let twice = rest(
        &h.iter()
            .zip(&sub)
            .map(|(a, b)| a + 2.0 * b)
            .collect::<Vec<_>>(),
    );

    let ours = cpu_logits_of("ds2-residual-once", &model, &[tok as u32]);
    let scale = rms(&once);
    let (d_once, d_twice) = (
        max_abs_diff(&ours, &once) / scale,
        max_abs_diff(&ours, &twice) / scale,
    );
    println!("cpu vs one residual add: {d_once:e};  vs two: {d_twice:e}  (logit rms {scale:e})");
    assert!(
        d_once < 0.01,
        "the CPU forward does not match a hand-computed MLA layer with ONE residual add \
         (relative max|Δ| = {d_once:e}); against a DOUBLED attention residual it is {d_twice:e}"
    );
    assert!(
        d_twice > 0.1,
        "the two references are indistinguishable (relative max|Δ| = {d_twice:e}) — this fixture \
         cannot tell a doubled attention residual from a single one, so it guards nothing"
    );
}

// ─── deepseek32 (V3.2) ───────────────────────────────────────────────────────────
//
// Stage 3. There is no V3.2 GGUF anywhere — the model is 671B — so this is the only place the arch
// is exercised at all: `Config::from_gguf`'s `deepseek32` branch, `wload`'s five extra per-layer
// tensors, and the lightning indexer's emit arm, all driven off a real file through the real seam.

/// How many keys the indexer keeps, when the case wants the top-k to actually RESTRICT attention.
/// [`PROMPT`] is 8 tokens, so at the prefill graph 5 of 8 keys survive, and the last prompt row —
/// the one the logits come from — sees 5 of its 8 causally-eligible keys.
const RESTRICT_TOP_K: usize = 5;

/// The blessed value of [`synthetic_deepseek32_indexer_selection_is_locked`] — see that test for
/// what it does and does not prove.
///
/// Re-blessed 2026-08-12, when the doubled MLA residual was removed from `MixerW::Mla` — the emit
/// pushed its own `hidden += sub` and then fell through to the shared post-mixer one, so every
/// layer ran `x + 2·Wo·attn`. That is upstream of the indexer on every layer but the first (its
/// queries and keys are both projections of the attn-normed residual), so the selected key set was
/// blessed off a wrong residual stream. `docs/deepseek.md` § Stage 2 records the fix and the
/// llama.cpp-oracle numbers behind it.
///
/// Re-blessed 2026-11-25, when deepseek32's YaRN attention factor took the reference's own rule
/// (`config::yarn_mscale_ratio`). The fixture declares V2-Lite's multiplier `L`, and
/// `llama-context.cpp`'s `mscale = mscale_all_dims` special case applies to `LLM_ARCH_DEEPSEEK2`
/// only: the old values were blessed under that `deepseek2` rule (rope magnitude 1, score scale
/// from `get_mscale(f, L)`), the new ones under deepseek32's (magnitude
/// `get_mscale(f, 1) / get_mscale(f, L)`, score scale from `get_mscale(f, 1)`).
/// `synthetic_deepseek32_yarn_rule_matches_the_reference` checks that rule against the formula.
#[rustfmt::skip]
const GOLDEN_DS32_TOPK5: [f32; 64] = [
    1.5307678, 0.05498229, -0.23287037, -3.0614324,
    -1.2109104, 0.47288695, -0.72883505, -0.9374795,
    0.3698898, 0.5519588, 1.9441997, -0.47557783,
    -0.53684807, -1.716015, -0.059505038, 0.49732724,
    1.2705402, 1.6521858, 0.21998231, -1.1757166,
    0.62528974, -1.674165, -0.18168652, 0.5950237,
    -2.112706, -1.0566213, 0.8070792, -0.36124665,
    0.5057566, -1.4730296, 0.21873015, 0.5567526,
    -0.39121675, 1.3978524, 0.044012412, 1.3416146,
    0.63064986, 0.5527649, -1.1742871, 0.1410883,
    -0.58230364, 0.9851058, -0.22624849, -0.27751318,
    0.52736294, 0.056262612, 0.8925078, 0.90824944,
    -0.26945868, 0.5959414, 0.6791754, 2.2842422,
    -0.3421495, 1.0795772, 2.2031047, -0.4541933,
    1.2699249, 1.4044219, 1.7401822, -0.058567047,
    -0.8061923, -0.16286373, 0.6237036, 1.2303762,
];

/// A `top_k` past every `kv_len` this harness can produce (prompt 8 + the one decode step). The
/// seam clamps to `min(top_k, kv_len)` exactly as `deepseek32.cpp` does, so every key is selected
/// and the mask is all-zero — the case that must reproduce the unrestricted answer.
const UNRESTRICTED_TOP_K: usize = 64;

/// V3.2's indexer at toy dimensions.
///
/// `head_size` is DELIBERATELY neither `qk_rope_dim` nor `2 * qk_rope_dim`, and `n_head` is not the
/// model's `n_head`: the indexer's four shaped tensors then have four distinct shapes, so a loader
/// that reached for the MLA head's dims instead of the indexer's could not still fit. `head_size =
/// 24` against `qk_rope_dim = 16` also puts the `[rope | nope]` split at 16/8 — llama.cpp writes the
/// nope view's offset as `row_size(nope)` where the layout means `row_size(rope)`, and the two
/// coincide only on V3.2's own 64/64 head, so running here at 16/8 exercises a geometry where the
/// two readings genuinely differ.
fn ds32_indexer() -> IndexerDims {
    IndexerDims {
        n_head: 4,
        head_size: 24,
        top_k: RESTRICT_TOP_K,
    }
}

/// The canonical `deepseek32` dims: the grouped/biased deepseek2 model, with a DIFFERENT
/// `rms_eps` from the 1e-6 `deepseek32.cpp` hardcodes for the indexer's LayerNorm, so
/// `Config::rms_eps` and `Config::norm_eps` cannot be confused for one another.
fn ds32_dims() -> Ds2Dims {
    Ds2Dims {
        rms_eps: 1e-5,
        ..ds2_dims(N_GROUPS, N_GROUPS_USED, Some(force_bias()))
    }
}

/// The canonical `deepseek32` model — indexer top-k = [`RESTRICT_TOP_K`], so the indexer really
/// does restrict attention.
fn deepseek32_model() -> SyntheticModel {
    deepseek32_model_top_k(RESTRICT_TOP_K)
}

/// [`deepseek32_model`] with a chosen `attention.indexer.top_k` and NOTHING else changed — the
/// fill is seeded by tensor NAME, so two `top_k` variants share every weight byte for byte and
/// differ only in one metadata u32.
fn deepseek32_model_top_k(top_k: usize) -> SyntheticModel {
    mla_model(
        infr_llama::arch::DEEPSEEK32,
        &ds32_dims(),
        Some(&IndexerDims {
            top_k,
            ..ds32_indexer()
        }),
    )
}

/// The `deepseek2` twin of [`deepseek32_model`]: the SAME dims and the same (name-seeded) weights,
/// minus the five indexer tensors and the three indexer keys.
///
/// This is the "mask disabled" reference the two top-k cases below compare against, and it is an
/// exact one rather than an approximation: the indexer's ONLY effect on the model is the additive
/// score mask it hands `Op::Mla`, so an all-zero mask must leave the arithmetic bit-identical to a
/// model that has no indexer at all. That holds only where the two arch strings give the same YaRN
/// attention factor, so a comparison of the two runs both at [`at_v32_multiplier`].
fn deepseek32_without_indexer_model() -> SyntheticModel {
    mla_model(infr_llama::arch::DEEPSEEK2, &ds32_dims(), None)
}

/// `m` (of `arch`) at V3.2's own YaRN multiplier (`mscale_all_dim` 1, written as 0.1). There the
/// `deepseek2` and `deepseek32` rules for the YaRN attention factor agree exactly
/// (`llama-context.cpp` special-cases only `deepseek2`, and only for a multiplier other than 1),
/// where the canonical fixture's V2-Lite multiplier gives the two arches different score scales
/// and rope magnitudes.
fn at_v32_multiplier(m: SyntheticModel, arch: &str) -> SyntheticModel {
    with_meta(
        m,
        &format!("{arch}.rope.scaling.yarn_log_multiplier"),
        Meta::F32(0.1),
    )
}

/// The five per-layer tensor names V3.2 adds, as they appear on disk (`llama-arch.cpp`'s
/// `LLM_TENSOR_INDEXER_*` format strings).
const INDEXER_TENSORS: [&str; 5] = [
    "indexer.k_norm.weight",
    "indexer.k_norm.bias",
    "indexer.proj.weight",
    "indexer.attn_k.weight",
    "indexer.attn_q_b.weight",
];

/// `m` minus one tensor. The count check is the point: a typo in `name` would otherwise remove
/// nothing and leave a "differential" test comparing a model with itself.
fn without_tensor(mut m: SyntheticModel, name: &str) -> SyntheticModel {
    let before = m.tensors.len();
    m.tensors.retain(|t| t.name != name);
    assert_eq!(
        m.tensors.len() + 1,
        before,
        "{name} was not in the model — nothing was removed"
    );
    m
}

/// [`without_tensor`]'s metadata twin.
fn without_meta(mut m: SyntheticModel, key: &str) -> SyntheticModel {
    let before = m.meta.len();
    m.meta.retain(|(k, _)| k != key);
    assert_eq!(
        m.meta.len() + 1,
        before,
        "{key} was not in the model — nothing was removed"
    );
    m
}

/// `Config::from_gguf`'s error for `m`, as a full `{:#}` chain. Panics if the file parses.
fn config_err(tag: &str, m: &SyntheticModel) -> String {
    let tmp = TempGguf::write(tag, m);
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");
    let err = infr_llama::Config::from_gguf(&g).expect_err("this fixture must be refused");
    format!("{err:#}")
}

/// The error a CPU prefill of `m` fails with, as a full `{:#}` chain. Panics if the run succeeds —
/// every caller is a DAMAGED fixture that must be refused.
fn prefill_err(tag: &str, m: &SyntheticModel) -> String {
    let tmp = TempGguf::write(tag, m);
    let model = load(&tmp);
    let err = model
        .prefill_logits_cpu(PROMPT)
        .expect_err("this fixture is incomplete and must be refused, not silently run");
    format!("{err:#}")
}

/// Every gate boolean and every derived dim of a `deepseek32` config, including where the
/// `deepseek2`-only gates land. `deepseek2` is TRUE here on purpose: MLA, the one-compressed-row KV
/// geometry, the group-limited MoE and the dense-lead threshold are all shared verbatim, so they
/// read one flag. `deepseek32` gates only what V3.2 adds.
#[test]
fn synthetic_deepseek32_config_gates() {
    let d = ds32_dims();
    let ix = ds32_indexer();
    let tmp = TempGguf::write("ds32-config", &deepseek32_model());
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");

    assert!(cfg.deepseek32, "deepseek32 gate");
    assert!(
        cfg.deepseek2,
        "deepseek32 must ALSO set deepseek2 — that flag is what gates MLA, the compressed KV row, \
         the MoE shape and the dense-lead threshold, all of which V3.2 shares verbatim"
    );
    assert!(!cfg.deepseek, "the V1 gate must stay false");
    assert!(
        !cfg.is_lite,
        "V3.2 has no lite variant — the LoRA query path is the only one it has"
    );
    assert!(!cfg.qk_norm, "no learned q/k-norm");
    assert!(!cfg.qkv_bias, "no attention biases");
    assert!(!cfg.permute_qk_neox);
    assert!(!cfg.sub_norm);
    assert!(!cfg.llama4);
    assert!(!cfg.qwen35);
    assert!(!cfg.gemma && !cfg.gemma4);
    assert!(!cfg.shexp_gated, "DeepSeek's shared expert is summed plain");

    // Indexer hparams, and the LayerNorm epsilon that is NOT the RMSNorm one.
    assert_eq!(cfg.indexer_n_head, ix.n_head);
    assert_eq!(cfg.indexer_head_size, ix.head_size);
    assert_eq!(cfg.indexer_top_k, ix.top_k);
    assert_eq!(cfg.norm_eps, 1e-6, "deepseek32.cpp hardcodes f_norm_eps");
    assert_eq!(cfg.rms_eps, d.rms_eps, "f_norm_rms_eps comes off the GGUF");
    assert_ne!(
        cfg.norm_eps, cfg.rms_eps,
        "the indexer's LayerNorm epsilon and the RMSNorm epsilon are separate hparams"
    );

    // MLA geometry, derived by the same code deepseek2 uses.
    assert_eq!(cfg.n_layer, d.n_layer);
    assert_eq!(cfg.vocab, d.vocab);
    assert_eq!(cfg.n_kv, 1, "MLA caches one key head for every query head");
    assert_eq!(cfg.head_dim, d.key_length_mla);
    assert_eq!(cfg.head_k_mla, d.qk_nope());
    assert_eq!(cfg.v_head_dim, d.value_length_mla);
    assert_eq!(cfg.kv_lora_rank, d.kv_lora_rank);
    assert_eq!(cfg.qk_rope_dim, d.qk_rope_dim);
    assert_eq!(cfg.q_lora_rank, d.q_lora_rank);
    assert_eq!(cfg.key_length, d.key_length_mla);

    // MoE, dense lead, YaRN — every one of them a `deepseek2` code path.
    assert_eq!(cfg.n_layer_dense_lead, d.n_dense_lead);
    assert!(!cfg.is_moe_layer(0), "layer 0 is the dense lead");
    assert!(cfg.is_moe_layer(1) && cfg.is_moe_layer(2));
    assert_eq!(cfg.shexp_ff, d.shexp_ff());
    assert!(cfg.rope_scaling_yarn);
    assert!((cfg.rope_yarn_log_mul - 0.707).abs() < 1e-4);
    let moe = cfg.moe.expect("deepseek32 is a MoE arch");
    assert_eq!(moe.gating, infr_core::graph::MoeGating::Sigmoid);
    assert_eq!(moe.n_expert_groups, N_GROUPS as u32);
    assert_eq!(moe.n_expert_groups_used, N_GROUPS_USED as u32);

    // And the flag really is arch-keyed: the deepseek2 twin leaves every V3.2 field at zero.
    let tmp2 = TempGguf::write("ds2-not-32", &grouped_model());
    let g2 = infr_gguf::Gguf::open(tmp2.path()).expect("open synthetic GGUF");
    let cfg2 = infr_llama::Config::from_gguf(&g2).expect("Config::from_gguf");
    assert!(cfg2.deepseek2 && !cfg2.deepseek32);
    assert_eq!(
        (
            cfg2.indexer_n_head,
            cfg2.indexer_head_size,
            cfg2.indexer_top_k
        ),
        (0, 0, 0)
    );
    assert_eq!(cfg2.norm_eps, 0.0, "deepseek2 emits no LayerNorm");
}

/// **The weight loader consumes all five indexer tensors, on every layer.**
///
/// A complete model prefills to finite logits, which is only possible once `wload` has walked every
/// layer without complaint. Remove any ONE of the five from any layer and the SAME run stops
/// earlier, inside `wload`, naming the tensor it wanted — so a loader that quietly ignored these
/// files' extra tensors would fail this test on all ten of its cases.
#[test]
fn synthetic_deepseek32_load_consumes_every_indexer_tensor() {
    let complete = cpu_logits("ds32-complete", &deepseek32_model());
    assert!(
        complete.iter().all(|v| v.is_finite()),
        "a complete deepseek32 model must load and run, got: {complete:?}"
    );

    // Layer 0 is the DENSE-lead layer and layer 2 is a MoE layer: the indexer is unconditional, so
    // both must demand all five.
    for l in [0usize, 2] {
        for suffix in INDEXER_TENSORS {
            let name = format!("blk.{l}.{suffix}");
            let err = prefill_err(
                &format!("ds32-no-{}-{l}", suffix.replace('.', "-")),
                &without_tensor(deepseek32_model(), &name),
            );
            println!("deepseek32 without {name}: {err}");
            assert!(
                err.contains(&format!("tensor not found: {name}")),
                "removing {name} must fail the weight load naming it, got: {err}"
            );
        }
    }
}

/// A misnamed tensor is the same failure as a missing one, and this is the case that would catch a
/// loader asking for llama.cpp's ENUM name (`indexer_k_norm`) instead of the on-disk one
/// (`indexer.k_norm`). Renaming is remove-plus-add, so the file stays otherwise complete.
#[test]
fn synthetic_deepseek32_misnamed_indexer_tensor_is_refused() {
    let mut m = deepseek32_model();
    let ix = ds32_indexer();
    m = without_tensor(m, "blk.1.indexer.proj.weight");
    m.tensors.push(TensorSpec::new(
        "blk.1.indexer_proj.weight",
        vec![ds32_dims().n_embd, ix.n_head],
        Fill::Rand(0.25),
    ));
    let err = prefill_err("ds32-misnamed", &m);
    println!("deepseek32 with a misnamed indexer.proj: {err}");
    assert!(
        err.contains("tensor not found: blk.1.indexer.proj.weight"),
        "a misnamed indexer tensor must be refused, got: {err}"
    );
}

/// A weight whose dims disagree with the model's own metadata is refused at load, naming the
/// tensor and both shapes — not uploaded under its file size and indexed out of bounds by the
/// first forward pass. Each tensor below is one the loader derives a shape for (a dense-lead
/// layer, a MoE layer, MLA and the globals); each is widened along its contiguous axis in turn,
/// and the undamaged model must still run so the list cannot pass by refusing everything.
#[test]
fn synthetic_deepseek2_misshaped_weight_is_refused() {
    let names = [
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q_a.weight",
        "blk.0.attn_q_a_norm.weight",
        "blk.0.attn_q_b.weight",
        "blk.0.attn_kv_a_mqa.weight",
        "blk.0.attn_kv_a_norm.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.1.ffn_gate_inp.weight",
        "blk.1.ffn_gate_exps.weight",
        "blk.1.ffn_up_exps.weight",
        "blk.1.ffn_down_exps.weight",
        "blk.1.exp_probs_b.bias",
        "blk.1.ffn_gate_shexp.weight",
        "blk.1.ffn_up_shexp.weight",
        "blk.1.ffn_down_shexp.weight",
    ];
    assert!(
        cpu_logits("ds2-shape-ok", &flat_model())
            .iter()
            .all(|v| v.is_finite()),
        "the undamaged model must load and run"
    );
    for name in names {
        let mut m = flat_model();
        let t = m
            .tensors
            .iter_mut()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is not in the fixture"));
        let want = t.shape.clone();
        t.shape[0] += 2;
        t.fill = Fill::Rand(0.25);
        let got = t.shape.clone();
        let err = prefill_err(&format!("ds2-shape-{name}"), &m);
        assert!(
            err.contains(name)
                && err.contains(&format!("{got:?}"))
                && err.contains(&format!("{want:?}")),
            "{name} widened to {got:?} must be refused naming it and {want:?}, got: {err}"
        );
    }
}

/// **MLA is mandatory.** `deepseek32.cpp::load_arch_tensors` opens with
/// `if (!hparams.is_mla()) throw "DEEPSEEK32 architecture requires MLA"`, and `is_mla()` is exactly
/// "both MLA head-length keys are non-zero". There is no unabsorbed fallback for V3.2 at all.
#[test]
fn synthetic_deepseek32_requires_mla() {
    for key in ["attention.key_length_mla", "attention.value_length_mla"] {
        let full = format!("deepseek32.{key}");
        let err = config_err(
            &format!("ds32-no-{}", key.replace('.', "-")),
            &without_meta(deepseek32_model(), &full),
        );
        println!("deepseek32 without {full}: {err}");
        assert_eq!(
            err,
            format!("deepseek32 architecture requires MLA: {full} is missing or zero")
        );
    }
}

/// The keys `deepseek32.cpp` reads UNCONDITIONALLY where `deepseek2.cpp` tolerates their absence.
/// Both defaults would be silently wrong here: a missing `q_lora_rank` would read as the lite
/// variant V3.2 does not have, and a missing `expert_gating_func` would route softmax where V3.2 is
/// sigmoid.
#[test]
fn synthetic_deepseek32_mandatory_keys_are_required() {
    for key in [
        "attention.q_lora_rank",
        "expert_gating_func",
        "attention.indexer.head_count",
        "attention.indexer.key_length",
        "attention.indexer.top_k",
    ] {
        let full = format!("deepseek32.{key}");
        let err = config_err(
            &format!("ds32-nokey-{}", key.replace('.', "-")),
            &without_meta(deepseek32_model(), &full),
        );
        println!("deepseek32 without {full}: {err}");
        assert!(
            err.contains(&full),
            "a deepseek32 GGUF without {full} must be refused naming it, got: {err}"
        );
    }
}

/// **NextN/MTP blocks are split off and skipped**, which is what `deepseek32.cpp` does with its
/// `i >= n_layer` → `TENSOR_SKIP` arm. `block_count` counts them, so the trunk is the difference;
/// the fixture carries no `blk.3` tensors at all and must still load, proving nothing walks them.
#[test]
fn synthetic_deepseek32_nextn_blocks_are_skipped() {
    let d = Ds2Dims {
        n_layer_nextn: 1,
        ..ds32_dims()
    };
    let m = mla_model(infr_llama::arch::DEEPSEEK32, &d, Some(&ds32_indexer()));
    assert!(
        !m.tensors.iter().any(|t| t.name.starts_with("blk.3.")),
        "the fixture must carry no NextN-block tensors — that is what makes the skip observable"
    );
    let tmp = TempGguf::write("ds32-nextn", &m);
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    assert_eq!(cfg.n_layer_nextn, 1);
    assert_eq!(cfg.n_layer, d.n_layer, "the trunk excludes the NextN block");

    let logits = cpu_logits("ds32-nextn-load", &m);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "the NextN fixture must load fully and run the trunk, got: {logits:?}"
    );
}

/// **The whole V3.2 model runs**: MLA plus the lightning indexer on every layer (its own LayerNorm,
/// its NEOX rope, its second KV cache, its top-k, and the score mask that feeds `Op::Mla`), a
/// dense-lead layer and two group-routed MoE layers, end to end on the CPU reference backend.
#[test]
fn synthetic_deepseek32_cpu_prefill_is_finite() {
    let logits = cpu_logits("ds32-finite", &deepseek32_model());
    assert_eq!(logits.len(), 64, "logits are [vocab]");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "non-finite logit in the synthetic deepseek32 CPU prefill: {logits:?}"
    );
    assert!(rms(&logits) > 1e-3, "logits are degenerate: {logits:?}");
}

/// **`top_k >= kv_len` reproduces the unrestricted answer, EXACTLY.** This is the property that
/// makes the mask faithful: with every key selected the mask is all-zero, `score + 0.0` is exact,
/// and the run must therefore be bit-identical to the same model with no indexer at all.
///
/// Bit-identity holds on the CPU because its `Op::Mla` arm adds the mask as a SEPARATE statement,
/// which Rust does not contract. The Vulkan twin
/// (`gpu_synthetic_deepseek32_top_k_paths_execute`) can only assert it to ~1 ULP, for a reason
/// spelled out there.
///
/// It is also the check that the indexer perturbs NOTHING else — not the KV geometry it shares a
/// buffer with, not the residual stream, not the MoE routing. A single non-zero digit here means
/// some part of the indexer is leaking into the attention path. Both files sit at
/// [`at_v32_multiplier`], where the two arch strings' YaRN rules agree.
#[test]
fn synthetic_deepseek32_full_top_k_matches_no_indexer_at_all() {
    let full = cpu_logits(
        "ds32-topk-full",
        &at_v32_multiplier(
            deepseek32_model_top_k(UNRESTRICTED_TOP_K),
            infr_llama::arch::DEEPSEEK32,
        ),
    );
    let none = cpu_logits(
        "ds32-no-indexer",
        &at_v32_multiplier(
            deepseek32_without_indexer_model(),
            infr_llama::arch::DEEPSEEK2,
        ),
    );
    let d = max_abs_diff(&full, &none);
    println!("cpu deepseek32 top_k>=kv_len vs no indexer: max|Δ| = {d:e}");
    assert_eq!(
        d, 0.0,
        "an all-selected top-k must add an all-zero mask and leave the attention untouched"
    );
}

/// **The indexer actually RESTRICTS attention.** Same weights, same prompt, same everything — only
/// `attention.indexer.top_k` moves, from "keep every key" to "keep 5 of 8". The 8-token prompt's
/// last row has 8 causally-eligible keys and may see only 5 of them, so the outputs cannot agree
/// unless the mask never reached the MLA kernel.
///
/// A test that passed with the mask absent would be testing nothing, which is why this asserts
/// against the [`UNRESTRICTED_TOP_K`] run rather than against "is finite".
#[test]
fn synthetic_deepseek32_top_k_restricts_attention() {
    let restricted = cpu_logits("ds32-topk-5", &deepseek32_model());
    let full = cpu_logits(
        "ds32-topk-full-b",
        &deepseek32_model_top_k(UNRESTRICTED_TOP_K),
    );
    assert_routing_differs(
        "cpu deepseek32 top_k=5 vs top_k>=kv_len",
        &restricted,
        &full,
    );
}

/// The two `top_k` fixtures differ in exactly one metadata u32 — the premise the case above rests
/// on. Without this, "restricted vs full differ" could be true for some reason that has nothing to
/// do with the top-k.
#[test]
fn synthetic_deepseek32_top_k_variants_differ_only_in_top_k() {
    let (a, b) = (
        deepseek32_model(),
        deepseek32_model_top_k(UNRESTRICTED_TOP_K),
    );
    assert_eq!(
        weights_of(&a),
        weights_of(&b),
        "the two top_k variants must share every weight"
    );
    let diff: Vec<_> = a
        .meta
        .iter()
        .zip(&b.meta)
        .filter(|(x, y)| x != y)
        .map(|(x, _)| x.0.clone())
        .collect();
    assert_eq!(diff, vec!["deepseek32.attention.indexer.top_k".to_string()]);

    // And the no-indexer reference is the deepseek32 model minus exactly the indexer.
    let (ix, no_ix) = (deepseek32_model(), deepseek32_without_indexer_model());
    let extra: Vec<String> = ix
        .tensors
        .iter()
        .map(|t| t.name.clone())
        .filter(|n| !no_ix.tensors.iter().any(|t| &t.name == n))
        .collect();
    assert_eq!(extra.len(), INDEXER_TENSORS.len() * ds32_dims().n_layer);
    for (n, v) in weights_of(&no_ix) {
        let (_, w) = weights_of(&ix)
            .into_iter()
            .find(|(m, _)| *m == n)
            .expect("shared tensor");
        assert_eq!(w, v, "{n} must be identical across the two");
    }
}

/// **Regression lock on the indexer's SELECTION.** The restricted run's logits are a sharp function
/// of *which* 5 keys the indexer picked, and that pick depends on every detail of the indexer's
/// arithmetic: its LayerNorm (mean-centred, with bias), its NEOX rope pairing where the MLA rope
/// beside it is NORM, and the `[rope | nope]` head order that is the opposite of the MLA head's.
/// None of those has a cheap independent oracle — llama.cpp cannot run a 671B model here, and a
/// from-scratch host reference for this fixture would be a second implementation of the model.
///
/// So this is a LOCK, not a proof: the numbers were blessed from this implementation, and what they
/// buy is that any later change to the indexer's arithmetic is visible instead of silent. Each of
/// those three details was individually shown to move it (`docs/deepseek.md` § Stage 3).
///
/// The tolerance is deliberately relative and loose: a different rope pairing or head order changes
/// the selected key set, which moves the logits by a real fraction of their own scale, while
/// cross-machine float noise (SIMD width, reduction order) is many orders below it.
#[test]
fn synthetic_deepseek32_indexer_selection_is_locked() {
    /// `cpu_logits("ds32-topk-5", &deepseek32_model())`, blessed 2026-08-10,
    /// re-blessed 2026-08-12 and 2026-11-25 — see [`GOLDEN_DS32_TOPK5`].
    const GOLDEN: [f32; 64] = GOLDEN_DS32_TOPK5;
    let got = cpu_logits("ds32-lock", &deepseek32_model());
    let d = max_abs_diff(&got, &GOLDEN);
    let scale = rms(&GOLDEN);
    println!("deepseek32 selection lock: max|Δ| = {d:e}  (logit rms {scale:e})\n  got = {got:?}");
    assert!(
        d < 1e-4 * scale,
        "the synthetic deepseek32 logits moved (max|Δ| = {d:e}, logit rms {scale:e}) — the \
         lightning indexer selected a different key set. If that was intended, say why in \
         docs/deepseek.md before re-blessing; do not re-bless to make this green.\n  got  = \
         {got:?}\n  want = {GOLDEN:?}"
    );
}

/// The premise the load tests rest on: the deepseek32 fixture IS the deepseek2 one plus the
/// indexer. If the two builders drifted, "removing an indexer tensor breaks the load" could pass
/// for a reason that has nothing to do with the indexer.
#[test]
fn synthetic_deepseek32_is_deepseek2_plus_the_indexer() {
    let d = ds32_dims();
    let ds2 = mla_model(infr_llama::arch::DEEPSEEK2, &d, None);
    let ds32 = deepseek32_model();

    let extra: Vec<String> = ds32
        .tensors
        .iter()
        .map(|t| t.name.clone())
        .filter(|n| !ds2.tensors.iter().any(|t| &t.name == n))
        .collect();
    let mut want: Vec<String> = Vec::new();
    for l in 0..d.n_layer {
        for suffix in INDEXER_TENSORS {
            want.push(format!("blk.{l}.{suffix}"));
        }
    }
    assert_eq!(
        extra, want,
        "deepseek32 must add exactly the five indexer tensors, on every layer"
    );

    let strip = |m: &SyntheticModel| -> Vec<(String, Meta)> {
        m.meta
            .iter()
            .filter(|(k, _)| !k.contains("indexer"))
            .map(|(k, v)| (k.replacen("deepseek32.", "deepseek2.", 1), v.clone()))
            .collect()
    };
    assert_eq!(
        strip(&ds32),
        strip(&ds2)
            .into_iter()
            .map(|(k, v)| if k == "general.architecture" {
                (k, Meta::Str("deepseek32".to_string()))
            } else {
                (k, v)
            })
            .collect::<Vec<_>>(),
        "apart from the arch string and the three indexer keys, the two models' metadata is equal"
    );
}

// ─── Vulkan ──────────────────────────────────────────────────────────────────────

/// CPU vs Vulkan on the same synthetic file — the differential oracle. [`FORCE_BIAS`] pins the
/// routed set identically on both backends, so any divergence here is arithmetic (f16 KV, f16
/// weights, a different reduction order), not a routing near-tie.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_deepseek2_matches_cpu() {
    let _lk = gpu_serial_lock();
    let model = grouped_model();
    let cpu = cpu_logits("gpu-parity-cpu", &model);
    let gpu = vulkan_logits("gpu-parity-vk", &model);
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "non-finite logit in the Vulkan prefill: {gpu:?}"
    );
    let cos = cosine(&cpu, &gpu);
    println!("synthetic deepseek2 cpu-vs-vulkan cosine = {cos}");
    let (cpu_top, gpu_top) = (argmax(&cpu), argmax(&gpu));
    println!("cpu argmax = {cpu_top}, vulkan argmax = {gpu_top}");
    assert_eq!(cpu_top, gpu_top, "CPU and Vulkan disagree on the top token");
    // Tighter than the real-model seam tests' `> 0.5` on purpose: those run 27 layers of Q4_K
    // weights and tolerate routing near-ties flipping, while here the routed set is pinned and the
    // whole model is three f32 layers, so the two backends agree to ~1e-13. A threshold anywhere
    // near the observed value would be a driver-noise tripwire; this one still fails on any real
    // kernel divergence.
    assert!(
        cos > 0.9999,
        "CPU/Vulkan logits diverged on the synthetic deepseek2 model: cosine = {cos}"
    );
}

/// The CPU routing cases, repeated on Vulkan: `moe_topk.comp`'s group-mask and bias branches have
/// their own implementation of every rule the CPU arm implements, so proving them on one backend
/// proves nothing about the other.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_deepseek2_routing_paths_execute() {
    let _lk = gpu_serial_lock();
    let grouped = vulkan_logits("vk-grouped", &grouped_model());
    let flat = vulkan_logits("vk-flat", &flat_model());
    assert_routing_differs("vulkan grouped vs flat", &grouped, &flat);

    let unbiased = vulkan_logits("vk-unbiased", &no_bias_model());
    assert_routing_differs("vulkan forcing-bias vs no-bias", &grouped, &unbiased);

    let uniform = vulkan_logits("vk-uniform-bias", &uniform_bias_model());
    let d = max_abs_diff(&uniform, &unbiased);
    println!("vulkan uniform-bias vs no-bias: max|Δ| = {d:e}");
    assert_eq!(
        d, 0.0,
        "a uniform router bias changed the Vulkan output — moe_topk.comp is weighting from the \
         BIASED probs instead of the unbiased ones"
    );
}

/// [`gpu_synthetic_deepseek2_matches_cpu`]'s V3.2 twin — the only cross-backend check the lightning
/// indexer will ever get, since no real `deepseek32` file exists. Everything the indexer adds runs
/// on the GPU here: `layernorm.comp`, `rope.comp`'s `-DNEOX` build, the second `WriteKv` into the
/// indexer's own cache, `lightning_indexer.comp`, `topk_mask.comp`, and `mla.comp`'s `-DKEY_BIAS`
/// build. `RESTRICT_TOP_K` is in force, so the two backends must also agree on WHICH keys the
/// indexer picked — an integer decision, where a divergence is a whole key swapping in or out
/// rather than a rounding difference.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_deepseek32_matches_cpu() {
    let _lk = gpu_serial_lock();
    let model = deepseek32_model();
    let cpu = cpu_logits("ds32-gpu-parity-cpu", &model);
    let gpu = vulkan_logits("ds32-gpu-parity-vk", &model);
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "non-finite logit in the Vulkan deepseek32 prefill: {gpu:?}"
    );
    let cos = cosine(&cpu, &gpu);
    println!("synthetic deepseek32 cpu-vs-vulkan cosine = {cos}");
    let (cpu_top, gpu_top) = (argmax(&cpu), argmax(&gpu));
    println!("cpu argmax = {cpu_top}, vulkan argmax = {gpu_top}");
    assert_eq!(cpu_top, gpu_top, "CPU and Vulkan disagree on the top token");
    // Same threshold and same reasoning as the deepseek2 twin: three f32 layers with a pinned
    // routed set, so the two backends agree far tighter than this.
    assert!(
        cos > 0.9999,
        "CPU/Vulkan logits diverged on the synthetic deepseek32 model: cosine = {cos}"
    );
}

/// The two top-k properties, repeated on Vulkan: `topk_mask.comp` and `mla.comp`'s `-DKEY_BIAS`
/// build are their own implementations of the rule the CPU arm implements, so proving it on one
/// backend proves nothing about the other.
///
/// The all-selected case is NOT bit-exact here, unlike on the CPU, and the reason is worth stating
/// because it looks like a leak: the two runs take DIFFERENT pipelines (`mla_ff_bias` vs `mla_ff`),
/// and the masked build's score is `red[0] * scale + kbias[j]` — a fused-multiply-add candidate the
/// compiler takes, so the product is rounded ONCE where the unmasked build rounds it twice. With
/// `kbias[j] == 0.0` that is still a ≤1-ULP difference in the score, which the softmax and three
/// layers carry out to ~1e-6 on the logits. The mask really is all-zero: the CPU pair of the same
/// two files agrees to 0.0 exactly, and `topk_mask_parity` pins the Vulkan mask equal to the CPU's.
///
/// So the bound is relative and ~10× above what is observed, which is still four orders below the
/// difference a mask that actually removed keys makes (the case below).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_deepseek32_top_k_paths_execute() {
    let _lk = gpu_serial_lock();
    let full = vulkan_logits(
        "vk-ds32-topk-full",
        &at_v32_multiplier(
            deepseek32_model_top_k(UNRESTRICTED_TOP_K),
            infr_llama::arch::DEEPSEEK32,
        ),
    );
    let none = vulkan_logits(
        "vk-ds32-no-indexer",
        &at_v32_multiplier(
            deepseek32_without_indexer_model(),
            infr_llama::arch::DEEPSEEK2,
        ),
    );
    let d = max_abs_diff(&full, &none);
    let scale = rms(&none);
    println!("vulkan deepseek32 top_k>=kv_len vs no indexer: max|Δ| = {d:e} (logit rms {scale:e})");
    assert!(
        d < 1e-5 * scale,
        "an all-selected top-k must add an all-zero mask and leave mla.comp's attention untouched \
         (max|Δ| = {d:e}, logit rms {scale:e})"
    );
    assert_eq!(
        argmax(&full),
        argmax(&none),
        "an all-zero mask changed the top token"
    );

    let restricted = vulkan_logits(
        "vk-ds32-topk-5",
        &at_v32_multiplier(deepseek32_model(), infr_llama::arch::DEEPSEEK32),
    );
    assert_routing_differs(
        "vulkan deepseek32 top_k=5 vs top_k>=kv_len",
        &restricted,
        &full,
    );
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

// ─── deepseek4 (V4) ──────────────────────────────────────────────────────────────
//
// Stage 4's LOAD path. V4-Flash's smallest quant is 82.5 GB, so as with V3.2 there is no model file
// to develop against and this is the only place the arch is exercised at all: `Config::from_gguf`'s
// `deepseek4` block and `wload`'s `is_dsv4` arm, driven off a real file through the real loader.
//
// The canonical fixture covers every tier, and every tier generates on the CPU, so "the complete
// fixture generates" and "the fixture minus any one tensor is refused naming it" together say the
// loader consumed all of them.

/// The canonical `deepseek4` fixture.
fn deepseek4_model() -> SyntheticModel {
    dsv4_model(&dsv4_dims())
}

/// What a `deepseek4` fixture's CPU prefill of [`PROMPT`] comes to: its logits, or the error it
/// fails with, from EITHER the model load or the prefill, as a full `{:#}` chain. Both are damage
/// sites for this arch — a missing `token_embd` is refused by `SeamModel::load_with` and a missing
/// `blk.2.attn_kv` by `wload` inside the prefill — and a sweep over "remove any one tensor" has to
/// catch both without pre-judging which.
fn dsv4_run(tag: &str, m: &SyntheticModel) -> Result<Vec<f32>, String> {
    let tmp = TempGguf::write(tag, m);
    let model = infr_llama::SeamModel::load_with(
        tmp.path(),
        None,
        std::sync::Arc::new(infr_llama::EngineConfig::default()),
    )
    .map_err(|e| format!("{e:#}"))?;
    model
        .prefill_logits_cpu(PROMPT)
        .map_err(|e| format!("{e:#}"))
}

/// [`dsv4_run`]'s error, for a fixture that must be refused.
fn dsv4_err(tag: &str, m: &SyntheticModel) -> String {
    match dsv4_run(tag, m) {
        Ok(logits) => panic!("this fixture must be refused, but it produced logits: {logits:?}"),
        Err(e) => e,
    }
}

/// [`dsv4_run`]'s logits, for a fixture that must generate: finite ones.
fn dsv4_generates(tag: &str, m: &SyntheticModel) -> Vec<f32> {
    let logits = dsv4_run(tag, m).unwrap_or_else(|e| panic!("this fixture must generate: {e}"));
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "non-finite logits: {logits:?}"
    );
    logits
}

/// Every gate boolean and every derived value of a `deepseek4` config — including, first, the three
/// that must be FALSE. V4 is not MLA, and `Config::deepseek2` is what gates the MLA mixer, the
/// 576-wide compressed cache row, the f16-only KV rule and group-limited routing; inheriting it
/// (the way `deepseek32` deliberately does) would point every one of those at a model that has none
/// of them.
#[test]
fn synthetic_deepseek4_config_gates() {
    let d = dsv4_dims();
    let ix = dsv4_indexer();
    let tmp = TempGguf::write("ds4-config", &deepseek4_model());
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");

    assert!(cfg.deepseek4, "deepseek4 gate");
    assert!(!cfg.deepseek2, "V4 is NOT MLA — deepseek2 must stay false");
    assert!(!cfg.deepseek32, "deepseek32 must stay false");
    assert!(!cfg.deepseek, "the V1 gate must stay false");
    assert!(
        !cfg.is_lite,
        "`is_lite` is a deepseek2 tensor-presence test and must not fire here"
    );
    // The MLA fields `deepseek2` would have filled in. All zero: V4 has no `kv_lora_rank`, no
    // `wk_b`/`wv_b`, no MLA head lengths and no YaRN mscale.
    assert_eq!(
        (
            cfg.kv_lora_rank,
            cfg.key_length,
            cfg.head_k_mla,
            cfg.v_head_dim
        ),
        (0, 0, 0, 0),
        "V4 must carry none of deepseek2's MLA geometry"
    );
    assert!(!cfg.rope_scaling_yarn);
    assert_eq!(cfg.rope_yarn_log_mul, 0.0);
    // No `rope.scaling.*` keys at all: the reference's defaults are type "linear" with factor 0,
    // i.e. `freq_scale = 1` — the compressed tiers rope unscaled.
    assert_eq!(
        cfg.dsv4_compressed_rope_divisors(),
        Some(vec![1.0; d.rope_dim / 2]),
        "a V4 file without scaling keys has no YaRN"
    );
    assert_eq!(cfg.n_expert_groups, 0, "V4 does not group-limit routing");
    assert_eq!(cfg.n_layer_dense_lead, 0, "V4 has no dense-lead layers");
    assert_eq!(cfg.n_layer_nextn, 0, "V4 has no NextN blocks");
    assert_eq!(cfg.norm_eps, 0.0, "that epsilon is deepseek32's LayerNorm");
    assert!(!cfg.qk_norm && !cfg.qkv_bias && !cfg.permute_qk_neox);
    assert!(!cfg.sub_norm && !cfg.llama4 && !cfg.qwen35);
    assert!(!cfg.gemma && !cfg.gemma4);
    assert!(!cfg.shexp_gated, "DeepSeek's shared expert is summed plain");

    // Geometry read off the file, not defaulted: `head_dim` is 48 where `n_embd / n_head` is 32.
    assert_eq!(cfg.n_layer, d.n_layer());
    assert_eq!(cfg.vocab, d.vocab);
    assert_eq!(cfg.n_embd, d.n_embd);
    assert_eq!(cfg.n_head, d.n_head);
    assert_eq!(cfg.n_kv, d.n_kv, "MQA: one KV head");
    assert_eq!(cfg.head_dim, d.head_dim);
    assert_ne!(
        cfg.head_dim,
        d.n_embd / d.n_head,
        "the fixture's whole point"
    );
    assert_eq!(cfg.rope_dim, d.rope_dim);
    assert_eq!(cfg.q_lora_rank, d.q_lora_rank);

    // V4's own hyperparameters.
    assert_eq!(cfg.o_group_count, d.o_group_count);
    assert_eq!(cfg.o_lora_rank, d.o_lora_rank);
    assert_eq!(cfg.hc_mult, d.hc_mult);
    assert_eq!(cfg.hc_sinkhorn_iters, d.hc_sinkhorn_iters);
    assert_eq!(cfg.hc_eps, d.hc_eps);
    assert_eq!(cfg.compress_rope_theta, d.compress_rope_theta);
    assert_ne!(
        cfg.compress_rope_theta, cfg.rope_theta,
        "the compressed tiers' rope base is a separate hparam from the plain one"
    );
    assert_eq!(cfg.hash_layer_count, d.hash_layer_count);
    assert_eq!(cfg.compress_ratios, d.compress_ratios);
    assert_eq!(cfg.indexer_n_head, ix.n_head);
    assert_eq!(cfg.indexer_head_size, ix.head_size);
    assert_eq!(cfg.indexer_top_k, ix.top_k);

    // The per-layer switches, through the accessors the loader itself uses.
    for (l, &ratio) in d.compress_ratios.iter().enumerate() {
        assert_eq!(cfg.layer_compress_ratio(l), ratio, "layer {l} ratio");
        assert_eq!(
            cfg.is_hash_moe_layer(l),
            l < d.hash_layer_count,
            "layer {l} routing kind"
        );
        assert!(cfg.is_moe_layer(l), "every V4 layer is routed");
        // `set_swa_pattern(0)`: EVERY layer is sliding-window, which `swa_pattern` cannot express.
        assert!(cfg.is_swa_layer(l), "layer {l} must be sliding-window");
    }
    assert_eq!(cfg.swa_window, d.swa_window);
    assert_eq!(
        cfg.swa_pattern, 0,
        "V4 has no full-attention layers to space"
    );
    assert_eq!(
        cfg.swa_rope_theta, cfg.rope_theta,
        "V4's sliding-window layers rope at the plain base, not gemma's 10000 fallback"
    );

    // The SwiGLU clamps: per-layer, and `_shexp` absent from this fixture, so it mirrors `_exp`.
    assert_eq!(cfg.swiglu_clamp_exp, d.swiglu_clamp_exp);
    assert_eq!(
        cfg.swiglu_clamp_shexp, d.swiglu_clamp_exp,
        "an absent swiglu_clamp_shexp falls back to swiglu_clamp_exp"
    );

    // MoE.
    assert_eq!(cfg.shexp_ff, d.shexp_ff());
    let moe = cfg.moe.expect("deepseek4 is a MoE arch");
    assert_eq!(moe.gating, infr_core::graph::MoeGating::SqrtSoftplus);
    assert_eq!(moe.n_expert, d.n_expert);
    assert_eq!(moe.n_used, d.n_used);
    assert_eq!(moe.n_ff_exp, d.n_ff_exp);
    assert!(moe.norm_w, "expert_weights_norm came off the file");
    assert!(!moe.weight_before);
    assert_eq!(moe.scale, 2.5);
}

/// `swiglu_clamp_shexp`, when the file DOES declare it, is read rather than shadowed by `_exp` —
/// and a scalar is broadcast over the layers (`get_key_or_arr`'s other half).
#[test]
fn synthetic_deepseek4_shared_expert_clamp_is_read_when_present() {
    let d = Dsv4Dims {
        swiglu_clamp_shexp: Some(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        ..dsv4_dims()
    };
    let tmp = TempGguf::write("ds4-shexp-clamp", &dsv4_model(&d));
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    assert_eq!(cfg.swiglu_clamp_exp, d.swiglu_clamp_exp);
    assert_eq!(cfg.swiglu_clamp_shexp, d.swiglu_clamp_shexp.unwrap());

    // The SCALAR form of the same key: `get_key_or_arr` accepts one value and broadcasts it.
    let mut m = dsv4_model(&dsv4_dims());
    m.meta
        .push(("deepseek4.swiglu_clamp_shexp".to_string(), Meta::F32(2.5)));
    m.meta.sort_by(|a, b| a.0.cmp(&b.0));
    let tmp = TempGguf::write("ds4-shexp-scalar", &m);
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    assert_eq!(cfg.swiglu_clamp_shexp, vec![2.5; dsv4_dims().n_layer()]);
}

/// **The CPU host tier pages a V4 expert bank per expert.** With `paging.dram` set, a generation
/// reads from the file only the experts its routing picks, and its tokens are the mapped run's.
///
/// The banks are widened to exactly `MIN_PAGED_BYTES` of f32 so that they page at all. The
/// pager's trace is what shows the granularity: it must hold an expert pool, and no bank may have
/// had more of its experts pinned than the generation's tokens could route to. A bank paged whole
/// leaves no expert pool; an op that pinned every expert of a bank breaks the bound.
#[test]
fn synthetic_deepseek4_cpu_pages_experts_per_expert() {
    let d = Dsv4Dims {
        n_expert: 64,
        n_ff_exp: 64,
        ..dsv4_dims()
    };
    assert_eq!(
        d.n_embd * d.n_ff_exp * d.n_expert * 4,
        infr_cpu::paged::MIN_PAGED_BYTES,
        "the banks must be exactly pageable"
    );
    let tmp = TempGguf::write("ds4-cpu-paged-experts", &dsv4_model(&d));
    let trace = std::env::temp_dir().join(format!(
        "infr-ds4-cpu-paged-experts-{}.trace",
        std::process::id()
    ));
    let load = |edit: &dyn Fn(&mut infr_llama::EngineConfig)| {
        let mut ec = infr_llama::EngineConfig::default();
        edit(&mut ec);
        infr_llama::SeamModel::load_with(tmp.path(), None, std::sync::Arc::new(ec))
            .expect("load the synthetic V4 file")
    };
    const N_GEN: usize = 4;
    let want = load(&|_| {})
        .generate_cpu_ids(PROMPT, N_GEN, |_| {})
        .expect("mapped generation");
    let got = load(&|ec| {
        ec.paging.dram = Some(infr_core::SizeSpec::Bytes(64 << 20));
        ec.paging.trace = Some(trace.clone());
    })
    .generate_cpu_ids(PROMPT, N_GEN, |_| {})
    .expect("paged generation");
    let text = std::fs::read(&trace).expect("read the pager trace");
    std::fs::remove_file(&trace).expect("remove the pager trace");
    assert_eq!(got, want, "paging the experts changed the generated tokens");

    let parsed = infr_core::pager::trace::parse(&text[..]).expect("parse the pager trace");
    let experts = parsed
        .pools
        .iter()
        .find(|p| p.label.starts_with("cpu expert"))
        .unwrap_or_else(|| panic!("no expert pool in the trace: the banks were paged whole"));
    let mut pinned: std::collections::BTreeMap<u32, std::collections::BTreeSet<u32>> =
        Default::default();
    for call in &experts.calls {
        if let infr_core::pager::trace::Call::ResolveAndPin(id, _) = *call {
            let e = id % d.n_expert as u32;
            pinned.entry(id / d.n_expert as u32).or_default().insert(e);
        }
    }
    let banks = 3 * d.n_layer();
    assert_eq!(pinned.len(), banks, "a bank was never read: {pinned:?}");
    // Every token of the prompt and of the output routes `n_used` experts per layer, and nothing
    // else may be read.
    let bound = (PROMPT.len() + N_GEN) * d.n_used;
    for (bank, es) in &pinned {
        assert!(
            es.len() <= bound,
            "bank {bank} pinned {} experts, more than {bound} tokens x routes could pick",
            es.len()
        );
    }
}

/// **The hash-routing table is `GGML_TYPE_I32` on disk, and the loader carries it as integers.**
///
/// Every other tensor this harness writes is f32, and so was this one until a real V4 file was
/// opened: `ggml-org/DeepSeek-V4-Flash-0731-GGUF` writes `blk.{0,1,2}.ffn_gate_tid2eid.weight` as
/// ggml type 26, which `Gguf::open` refused outright as an unsupported type. The whole fixture is
/// now unopenable if that mapping goes away, so this test is the one that names WHY.
#[test]
fn synthetic_deepseek4_routing_table_is_i32_on_disk() {
    let d = dsv4_dims();
    let tmp = TempGguf::write("ds4-tid2eid", &deepseek4_model());
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic deepseek4 GGUF");
    let t = g
        .tensors()
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate_tid2eid.weight")
        .expect("the hash layer's routing table");
    assert_eq!(t.dtype, infr_core::DType::I32, "the table is i32, not f32");
    assert_eq!(t.shape, vec![d.n_used, d.vocab]);

    // The table's entries are expert IDS: they must decode as integers a router can index a bank
    // with, not as whatever the bytes happen to be. A fill that ran off the end of `0..n_expert`
    // would read as an out-of-range expert and index past the bank.
    let raw = g
        .tensor_bytes_arc(&t.name)
        .expect("routing table bytes")
        .to_vec();
    let ids: Vec<i32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| i32::from_le_bytes(c))
        .collect();
    assert_eq!(ids.len(), d.n_used * d.vocab);
    assert!(
        ids.iter().all(|&e| (0..d.n_expert as i32).contains(&e)),
        "every entry must be a valid expert id in 0..{}",
        d.n_expert
    );
    assert!(
        ids.iter().any(|&e| e != ids[0]),
        "a constant table would route every token identically"
    );
}

/// **The weight loader consumes every tensor the file declares.**
///
/// Not "every tensor of a list written down twice": the sweep is over the FIXTURE's own tensor
/// list, so a tensor the builder emits and `wload` never asks for fails here, and so does the
/// reverse. The complete model generates; with any tensor removed the load must fail naming it.
///
/// The fixture's five layers cover every (compress_ratio, routing) pair (see [`dsv4_dims`]), so the
/// per-layer switch is exercised in both directions: a loader that skipped the ratio-4 indexer set
/// or that asked for `exp_probs_b` on a hash layer fails on those layers' tensors specifically.
#[test]
fn synthetic_deepseek4_load_consumes_every_tensor() {
    dsv4_generates("ds4-complete", &deepseek4_model());

    let names: Vec<String> = deepseek4_model()
        .tensors
        .iter()
        .map(|t| t.name.clone())
        .collect();
    assert!(
        names.len() > 100,
        "fixture got smaller: {} tensors",
        names.len()
    );
    for name in &names {
        // `output.weight` is the one tensor whose absence is not damage: `wload` falls back to the
        // (tied) `token_embd.weight` for the LM head, exactly as it does for every tied model. Its
        // removal must therefore still generate — asserting that, rather than skipping it, keeps
        // the sweep exhaustive.
        let tag = format!("ds4-no-{}", name.replace('.', "-"));
        let m = without_tensor(deepseek4_model(), name);
        if name == "output.weight" {
            dsv4_generates(&tag, &m);
            continue;
        }
        let err = match dsv4_run(&tag, &m) {
            Ok(_) => panic!("removing {name} still generated: nothing asked for it"),
            Err(e) => e,
        };
        assert!(
            err.contains(name.as_str()),
            "removing {name} must fail the load naming it, got: {err}"
        );
    }
}

/// **Every V4 tensor is held to the shape its metadata implies.** The sweep above shows each tensor
/// is asked for; this one shows each is MEASURED: one dim of one tensor grown by one, everything
/// else intact, must be refused at load naming that tensor — not read out of bounds, or silently
/// mis-strided, by the first forward pass.
///
/// Two exceptions. `ffn_gate_tid2eid` is checked, by the hash gather's own vocabulary check and
/// with its own message (`synthetic_deepseek4_refuses_a_mis_shaped_routing_table`).
/// `token_embd` is not: with an untied `output.weight` on the CPU backend only the host embedding
/// path reads it, and that path does not go through the loader's shape check.
#[test]
fn synthetic_deepseek4_refuses_every_mis_shaped_tensor() {
    let names: Vec<String> = deepseek4_model()
        .tensors
        .iter()
        .map(|t| t.name.clone())
        .filter(|n| !n.ends_with("ffn_gate_tid2eid.weight") && n != "token_embd.weight")
        .collect();
    assert!(
        names.len() > 100,
        "fixture got smaller: {} tensors",
        names.len()
    );
    let mut missed = Vec::new();
    for name in &names {
        let mut m = deepseek4_model();
        let t = m
            .tensors
            .iter_mut()
            .find(|t| &t.name == name)
            .expect("listed above");
        t.shape[0] += 1;
        let err = dsv4_err(&format!("ds4-shape-{}", name.replace('.', "-")), &m);
        if !(err.contains(&format!("weight {name} has shape")) && err.contains("metadata implies"))
        {
            missed.push(format!("{name}: {err}"));
        }
    }
    assert!(
        missed.is_empty(),
        "growing each of these by one row was not refused at load naming it:\n{}",
        missed.join("\n")
    );
}

/// llama.cpp shapes V4's tensors with layer 0's head width, which on an all-sliding-window model is
/// `attention.key_length_swa` when the file declares one. infr's V4 graph indexes by
/// `attention.key_length`, so a file whose tensors follow a DIFFERENT swa width must be refused by
/// name at load, not run at the wrong stride.
#[test]
fn synthetic_deepseek4_refuses_tensors_shaped_by_a_different_swa_head_width() {
    let d = dsv4_ratio0_dims();
    let swa_head = d.head_dim - d.rope_dim;
    // Tensors built at the swa width; metadata claims the full width AND the swa one.
    let mut m = dsv4_model(&Dsv4Dims {
        head_dim: swa_head,
        ..dsv4_ratio0_dims()
    });
    m = with_meta(
        m,
        "deepseek4.attention.key_length",
        Meta::U32(d.head_dim as u32),
    );
    m.meta.push((
        "deepseek4.attention.key_length_swa".to_string(),
        Meta::U32(swa_head as u32),
    ));
    m.meta.sort_by(|a, b| a.0.cmp(&b.0));
    let err = dsv4_err("ds4-swa-head", &m);
    assert!(
        err.contains("has shape") && err.contains("metadata implies"),
        "tensors at a different swa head width must be refused at load, got: {err}"
    );
}

/// A file that stores `attn_output_a` the way llama.cpp creates it — `[rows, o_lora_rank,
/// o_group_count]` — is the same bytes as the 2-D form the shipped GGUF uses, and loads.
#[test]
fn synthetic_deepseek4_accepts_a_three_d_output_projection() {
    let d = dsv4_dims();
    let mut m = deepseek4_model();
    for t in m.tensors.iter_mut() {
        if t.name.ends_with("attn_output_a.weight") {
            t.shape = vec![
                d.n_head * d.head_dim / d.o_group_count,
                d.o_lora_rank,
                d.o_group_count,
            ];
        }
    }
    // The same bytes, so the same logits as the 2-D form.
    assert_eq!(
        dsv4_generates("ds4-wo-a-3d", &m),
        dsv4_generates("ds4-wo-a-2d", &deepseek4_model())
    );
}

/// The three `compress_ratio` tiers really do change which tensors a layer carries — the direction
/// the sweep above cannot show, because it only ever removes what the fixture already declared.
///
/// Flipping one layer's ratio while leaving its tensors alone makes the file inconsistent in a way
/// only a loader that READS the ratio can notice, and the tensor it then names says which tier it
/// switched to.
#[test]
fn synthetic_deepseek4_compress_ratio_picks_the_layer_tensor_set() {
    let base = dsv4_dims();
    // Each case relabels ONE layer by exactly one tier and names the tensor that tier adds:
    // 0 → 128 adds the attention compressor (layer 0 carries none), and 128 → 4 adds the indexer
    // set on top of a compressor layer 2 already has — so each failure names the tier's OWN tensor
    // rather than one it shares with the tier below.
    for (l, ratio, want) in [
        (0usize, 128usize, "blk.0.attn_compressor_kv.weight"),
        (2, 4, "blk.2.indexer.proj.weight"),
    ] {
        let mut ratios = base.compress_ratios.clone();
        ratios[l] = ratio;
        // Keep the TENSORS the original ratio gave that layer by building the file from the
        // unmodified dims and rewriting only the metadata array.
        let mut m = deepseek4_model();
        for (k, v) in m.meta.iter_mut() {
            if k == "deepseek4.attention.compress_ratios" {
                *v = Meta::I32Arr(ratios.iter().map(|r| *r as i32).collect());
            }
        }
        // A ratio-4 compressor is twice as wide as a ratio-128 one, so a 128 → 4 relabel is
        // refused at the compressor's width before the indexer is ever asked for. Give layer 2
        // the ratio-4 compressor shapes, so the indexer set is the ONLY thing the file lacks.
        if ratio == 4 {
            let coff = Dsv4Dims::coff(ratio);
            for t in m.tensors.iter_mut() {
                let Some(role) = t.name.strip_prefix(&format!("blk.{l}.")) else {
                    continue;
                };
                match role {
                    "attn_compressor_kv.weight" | "attn_compressor_gate.weight" => {
                        t.shape = vec![base.n_embd, coff * base.head_dim];
                    }
                    "attn_compressor_ape.weight" => t.shape = vec![coff * base.head_dim, ratio],
                    _ => {}
                }
            }
        }
        let err = dsv4_err(&format!("ds4-relabel-{l}-as-{ratio}"), &m);
        println!("deepseek4 layer {l} relabelled ratio {ratio}: {err}");
        assert!(
            err.contains(want),
            "a layer at ratio {ratio} must demand {want}, got: {err}"
        );
    }
    // Without that rewrite the relabel is caught one tensor earlier, at the compressor's width —
    // the ratio decides tensor SHAPES as well as which tensors a layer has.
    let mut ratios = base.compress_ratios.clone();
    ratios[2] = 4;
    let mut m = deepseek4_model();
    for (k, v) in m.meta.iter_mut() {
        if k == "deepseek4.attention.compress_ratios" {
            *v = Meta::I32Arr(ratios.iter().map(|r| *r as i32).collect());
        }
    }
    let err = dsv4_err("ds4-relabel-2-as-4-narrow", &m);
    assert!(
        err.contains("weight blk.2.attn_compressor_kv.weight has shape")
            && err.contains("metadata implies"),
        "a ratio-128 compressor under a ratio-4 label must be refused at its width, got: {err}"
    );
    // And the converse: relabel a ratio-4 layer as 0 and its indexer set is no longer asked for, so
    // the model generates even though those tensors are still sitting in the file unread.
    let mut ratios = base.compress_ratios.clone();
    ratios[1] = 0;
    let mut m = deepseek4_model();
    for (k, v) in m.meta.iter_mut() {
        if k == "deepseek4.attention.compress_ratios" {
            *v = Meta::I32Arr(ratios.iter().map(|r| *r as i32).collect());
        }
    }
    dsv4_generates("ds4-ratio4-as-0", &m);
}

/// `hash_layer_count` decides `ffn_gate_tid2eid` vs `exp_probs_b`, per layer and exclusively.
/// Widening it by one makes the loader ask a bias-carrying layer for the table it does not have.
#[test]
fn synthetic_deepseek4_hash_layer_count_picks_the_router_table() {
    let d = Dsv4Dims {
        hash_layer_count: dsv4_dims().hash_layer_count + 1,
        ..dsv4_dims()
    };
    // Build the file at the ORIGINAL count so layer 2 still carries `exp_probs_b`, then claim the
    // wider count in metadata.
    let mut m = deepseek4_model();
    for (k, v) in m.meta.iter_mut() {
        if k == "deepseek4.hash_layer_count" {
            *v = Meta::U32(d.hash_layer_count as u32);
        }
    }
    let err = dsv4_err("ds4-hash-wide", &m);
    println!("deepseek4 with hash_layer_count widened by one: {err}");
    assert!(
        err.contains("blk.2.ffn_gate_tid2eid.weight"),
        "a hash-routed layer must demand its tid2eid table, got: {err}"
    );

    // Narrowing it the other way: layer 1 then wants the bias it does not carry.
    let mut m = deepseek4_model();
    for (k, v) in m.meta.iter_mut() {
        if k == "deepseek4.hash_layer_count" {
            *v = Meta::U32(1);
        }
    }
    let err = dsv4_err("ds4-hash-narrow", &m);
    println!("deepseek4 with hash_layer_count narrowed by one: {err}");
    assert!(
        err.contains("blk.1.exp_probs_b.bias"),
        "a non-hash layer must demand its router bias, got: {err}"
    );
}

/// **Only `{0, 4, 128}` are accepted.** `deepseek4.cpp::load_arch_tensors` throws on anything else,
/// and it has to: the ratio decides which tensors the layer HAS, so an unknown value is not a
/// wider variant of a known one to be rounded toward.
#[test]
fn synthetic_deepseek4_rejects_an_unknown_compress_ratio() {
    for bad in [1i32, 8, 64, 127, 129, -4] {
        let mut m = deepseek4_model();
        for (k, v) in m.meta.iter_mut() {
            if k == "deepseek4.attention.compress_ratios" {
                let mut ratios: Vec<i32> = dsv4_dims()
                    .compress_ratios
                    .iter()
                    .map(|r| *r as i32)
                    .collect();
                ratios[2] = bad;
                *v = Meta::I32Arr(ratios);
            }
        }
        let err = config_err(&format!("ds4-ratio-{bad}"), &m);
        println!("deepseek4 with compress_ratios[2] = {bad}: {err}");
        assert!(
            err.contains("deepseek4.attention.compress_ratios[2]")
                && err.contains("is not one of 0, 4, 128"),
            "an unknown compression ratio must be refused naming the layer, got: {err}"
        );
    }
}

/// **`compress_ratios` must cover every layer.** The reference reads the array length first and
/// throws when it is shorter than `block_count`, because `load_arch_tensors` then indexes `[il]`
/// for every layer — a short array would leave the tail of the model reading past the end.
#[test]
fn synthetic_deepseek4_rejects_a_short_compress_ratios() {
    let mut m = deepseek4_model();
    for (k, v) in m.meta.iter_mut() {
        if k == "deepseek4.attention.compress_ratios" {
            *v = Meta::I32Arr(vec![0, 4, 128]); // 3 entries for 5 layers
        }
    }
    let err = config_err("ds4-ratios-short", &m);
    println!("deepseek4 with a short compress_ratios: {err}");
    assert_eq!(
        err,
        "deepseek4.attention.compress_ratios is shorter than block_count: 3 entries for 5 layers"
    );

    // A LONGER array is fine — the extra entries are simply unused, matching the reference's `<`.
    let mut m = deepseek4_model();
    for (k, v) in m.meta.iter_mut() {
        if k == "deepseek4.attention.compress_ratios" {
            *v = Meta::I32Arr(vec![0, 4, 128, 4, 0, 128, 0]);
        }
    }
    let tmp = TempGguf::write("ds4-ratios-long", &m);
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");
    let cfg = infr_llama::Config::from_gguf(&g).expect("a longer compress_ratios is accepted");
    assert_eq!(cfg.compress_ratios, dsv4_dims().compress_ratios);
}

/// **sqrt-softplus gating is mandatory.** Every other `expert_gating_func` names a DIFFERENT
/// routing function, and defaulting or approximating one would re-route silently — which is why
/// `deepseek4.cpp` throws rather than falling back.
#[test]
fn synthetic_deepseek4_requires_sqrt_softplus_gating() {
    for gf in [0u32, 1, 2, 3, 5] {
        let d = Dsv4Dims {
            expert_gating_func: gf,
            ..dsv4_dims()
        };
        let err = config_err(&format!("ds4-gating-{gf}"), &dsv4_model(&d));
        println!("deepseek4 with expert_gating_func = {gf}: {err}");
        assert_eq!(
            err,
            format!(
                "deepseek4 requires sqrt-softplus MoE scoring \
                 (deepseek4.expert_gating_func=4); got {gf}"
            )
        );
    }
}

/// The keys `deepseek4.cpp::load_arch_hparams` reads with a plain `get_key`. Every one of them is
/// either a tensor dimension or a routing input whose absence the shared parses below would paper
/// over with a default that is wrong in a way nothing downstream can detect — `expert_weights_scale`
/// would default to unscaled.
#[test]
fn synthetic_deepseek4_mandatory_keys_are_required() {
    for key in [
        "attention.layer_norm_rms_epsilon",
        "attention.q_lora_rank",
        "attention.sliding_window",
        "attention.output_group_count",
        "attention.output_lora_rank",
        "attention.compress_rope_freq_base",
        "attention.compress_ratios",
        "attention.indexer.head_count",
        "attention.indexer.key_length",
        "attention.indexer.top_k",
        "hyper_connection.count",
        "hyper_connection.sinkhorn_iterations",
        "hyper_connection.epsilon",
        "hash_layer_count",
        "expert_gating_func",
        "expert_feed_forward_length",
        "expert_shared_count",
        "expert_weights_scale",
        "expert_weights_norm",
        "swiglu_clamp_exp",
    ] {
        let full = format!("deepseek4.{key}");
        let err = config_err(
            &format!("ds4-nokey-{}", key.replace('.', "-")),
            &without_meta(deepseek4_model(), &full),
        );
        println!("deepseek4 without {full}: {err}");
        assert!(
            err.contains(&full),
            "a deepseek4 GGUF without {full} must be refused naming it, got: {err}"
        );
    }
}

/// `deepseek4` with `rope.scaling.*` keys — the canonical fixture plus the keys, rather than the
/// keys in the fixture itself, so every other V4 test keeps reading a file without them.
fn dsv4_model_with_scaling(keys: Vec<(&str, Meta)>) -> SyntheticModel {
    let mut m = deepseek4_model();
    for (k, v) in keys {
        m.meta.push((format!("deepseek4.{k}"), v));
    }
    m.meta.sort_by(|a, b| a.0.cmp(&b.0));
    m
}

/// The rope-scaling keys are read for `deepseek4`, by the generic loader's rules: `type`,
/// `factor` and `original_context_length` are used, while the beta corners and
/// `rope.scaling.attn_factor` are not (no V4 loader key reads them, and `deepseek4.cpp` passes
/// its own attention factor). The fixture's rope is 16 dims at compress base 40000, so with an
/// original context of 65536 ggml's truncated corr dims are `[floor 4.37…, ceil 6.99…] = [4, 7]`
/// at the default betas, and the divisors are `1 / (fs + (1 − fs)·ramp)` with `fs = 1/16` and
/// `ramp = 1 − clamp((p − 4)/3, 0, 1)`: 1 up to pair 4, then 16/11, 8/3, 16. A `yarn_beta_fast`
/// of 4 would move the start corner to 5 (floor 5.94…) and make pair 5 unscaled, so pair 5 is
/// what shows the declared beta was ignored.
#[test]
fn synthetic_deepseek4_rope_scaling_keys_are_read() {
    let yarn = || {
        vec![
            ("rope.scaling.type", Meta::Str("yarn".into())),
            ("rope.scaling.factor", Meta::F32(16.0)),
            ("rope.scaling.original_context_length", Meta::U32(65536)),
        ]
    };
    let cfg_of = |tag: &str, m: &SyntheticModel| {
        let tmp = TempGguf::write(tag, m);
        let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");
        infr_llama::Config::from_gguf(&g).expect("Config::from_gguf")
    };

    let mut keys = yarn();
    keys.push(("rope.scaling.yarn_beta_fast", Meta::F32(4.0)));
    keys.push(("rope.scaling.attn_factor", Meta::F32(3.0)));
    let cfg = cfg_of("ds4-yarn", &dsv4_model_with_scaling(keys));
    assert!(cfg.rope_scaling_yarn);
    assert_eq!(cfg.rope_scaling_factor, 16.0);
    assert_eq!(cfg.rope_scaling_orig_ctx, 65536);
    assert_eq!(
        (cfg.rope_yarn_beta_fast, cfg.rope_yarn_beta_slow),
        (32.0, 1.0)
    );
    assert_eq!(cfg.rope_attn_factor, 1.0);
    let want = [1.0, 1.0, 1.0, 1.0, 1.0, 16.0 / 11.0, 8.0 / 3.0, 16.0];
    let got = cfg.dsv4_compressed_rope_divisors().expect("deepseek4");
    assert_eq!(got.len(), want.len());
    for (p, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-6 * w, "pair {p}: got {g}, want {w}");
    }

    // "none" forces `freq_scale = 1` whatever the factor; no type at all means "linear", whose
    // factor divides every pair alike; and a type the reference does not know is refused.
    let mut none = yarn();
    none[0].1 = Meta::Str("none".into());
    let cfg = cfg_of("ds4-rope-none", &dsv4_model_with_scaling(none));
    assert_eq!(cfg.dsv4_compressed_rope_divisors(), Some(vec![1.0; 8]));
    let linear = yarn().split_off(1);
    let cfg = cfg_of("ds4-rope-linear", &dsv4_model_with_scaling(linear));
    assert!(!cfg.rope_scaling_yarn);
    assert_eq!(cfg.dsv4_compressed_rope_divisors(), Some(vec![16.0; 8]));
    let mut odd = yarn();
    odd[0].1 = Meta::Str("ntk".into());
    let err = config_err("ds4-rope-ntk", &dsv4_model_with_scaling(odd));
    assert!(
        err.contains("deepseek4.rope.scaling.type"),
        "an unknown rope scaling type must be refused naming the key, got: {err}"
    );
}

/// **Every tier generates on the CPU, and a CSA layer whose indexer cannot rope is refused by
/// name.** The canonical fixture (ratios 0, 4, 128, 4, 0) generates; `synthetic_deepseek4_llama_cpp.rs`
/// scores every tier against llama.cpp. An indexer head narrower than the rope tail every head of
/// a compressed layer rotates would otherwise reach a rope that asserts, so it is refused at the
/// graph build, naming the first ratio-4 layer.
#[test]
fn synthetic_deepseek4_csa_layer_generates_unless_its_indexer_cannot_rope() {
    dsv4_generates("ds4-csa", &deepseek4_model());
    let d = dsv4_dims();
    let err = dsv4_err(
        "ds4-indexer-narrow",
        &dsv4_model(&Dsv4Dims {
            indexer: IndexerDims {
                head_size: d.rope_dim / 2,
                ..dsv4_indexer()
            },
            ..dsv4_dims()
        }),
    );
    println!("deepseek4 narrow indexer: {err}");
    assert_eq!(
        err,
        format!(
            "arch=deepseek4 (DeepSeek V4) layer 1 has compress_ratio 4, but the lightning \
             indexer's attention.indexer.key_length {} is narrower than rope.dimension_count {}, \
             the tail every head of a compressed layer ropes.",
            d.rope_dim / 2,
            d.rope_dim
        )
    );
}

/// **A compressed V4 layer is refused on Vulkan, loudly.** The compressed tiers' graphs have been
/// checked against llama.cpp on the CPU only; until a GPU run is checked against them (plan 09 S9)
/// a GPU backend must refuse the model by name rather than run it unverified, and say how to
/// select the CPU. The refusal comes before the weight upload, which the GPU binder would
/// otherwise run. Both tiers: an HCA-only file, and the canonical one, whose first compressed
/// layer is a CSA layer.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_deepseek4_refuses_a_compressed_layer_on_vulkan() {
    let _lk = gpu_serial_lock();
    let mut hca = dsv4_dims();
    hca.compress_ratios = vec![0, 0, 128, 0, 0];
    hca.hash_layer_count = 0;
    for (tag, d, want) in [
        ("ds4-hca-vk", hca, "layer 2 has compress_ratio 128"),
        ("ds4-csa-vk", dsv4_dims(), "layer 1 has compress_ratio 4"),
    ] {
        let tmp = TempGguf::write(tag, &dsv4_model(&d));
        let err = match load(&tmp).prefill_logits_vulkan(PROMPT) {
            Ok(logits) => panic!("Vulkan ran a compressed V4 layer: {logits:?}"),
            Err(e) => format!("{e:#}"),
        };
        println!("deepseek4 {tag}: {err}");
        assert!(
            err.contains(&format!("{want}: the compressed tiers run on the CPU"))
                && err.contains("this is the vulkan backend")
                && err.contains("`--dev cpu`, `INFR_DEV=cpu`"),
            "the Vulkan refusal must name the layer and the backend, got: {err}"
        );
    }
}

/// **A `ffn_gate_tid2eid` that is not one row per vocabulary entry is refused.** The gather indexes
/// the table by TOKEN ID with no clamp, exactly as `Op::EmbedGather` indexes `token_embd`, so a
/// short table would read the wrong row — or past the end of the tensor.
#[test]
fn synthetic_deepseek4_refuses_a_mis_shaped_routing_table() {
    let mut d = dsv4_hash_dims();
    // Ratio 0 everywhere, so the compressed-tier refusal cannot be what fires.
    d.compress_ratios = vec![0; d.n_layer()];
    let mut m = dsv4_model(&d);
    // Halve the row count: still a valid i32 tensor, still loads, still `n_used`-wide rows.
    for t in m.tensors.iter_mut() {
        if t.name.ends_with("ffn_gate_tid2eid.weight") {
            t.shape = vec![d.n_used, d.vocab / 2];
            t.fill = Fill::ExpertIds(d.n_expert);
        }
    }
    let err = dsv4_err("ds4-short-tid2eid", &m);
    println!("deepseek4 short routing table: {err}");
    assert!(
        err.contains("ffn_gate_tid2eid.weight` holds")
            && err.contains("expert_used_count*vocab")
            && err.contains("TOKEN ID"),
        "a short routing table must be refused naming the shape it needs, got: {err}"
    );
}

// ─── deepseek4 ratio 0: the tier that GENERATES ──────────────────────────────────
//
// `generate_dense_backend` emits `compress_ratio == 0` layers end to end: hyper-connections
// wrapping both sublayers, MQA attention with a weightless per-head Q norm, a `[nope | rope]` tail
// rope, attention sinks, a de-roped grouped low-rank output projection, and sqrt-softplus MoE with
// the per-layer SwiGLU clamps under EITHER routing — the `exp_probs_b` router bias, or the
// `ffn_gate_tid2eid` hash table gathered by `Op::GatherI32`. Everything below drives that through
// the real loader and the real seam, on CPU unconditionally and on Vulkan behind `#[ignore]`.

/// The all-ratio-0 V4 fixture. Everything is [`dsv4_dims`] verbatim except the two things the emit
/// refuses: every layer's compression ratio is 0, and no layer is hash-routed.
///
/// What it deliberately does NOT simplify, because each would leave a path untested:
/// `o_group_count` stays 2 (a 1 would make the grouped output projection's `w_off` a check that
/// cannot fail), `hc_mult` stays 2 (a 1 makes almost every Sinkhorn detail inert — see
/// `docs/backlog.md` § B-DSV4-HC), the per-layer SwiGLU clamps stay all-different, `head_dim`
/// stays 48 where `n_embd / n_head` is 32, and `rope_dim` (16) stays well below `head_dim` so the
/// `[nope | rope]` split is a real split.
fn dsv4_ratio0_dims() -> Dsv4Dims {
    let base = dsv4_dims();
    Dsv4Dims {
        compress_ratios: vec![0; base.n_layer()],
        hash_layer_count: 0,
        ..base
    }
}

fn dsv4_ratio0_model() -> SyntheticModel {
    dsv4_model(&dsv4_ratio0_dims())
}

/// Replace the fill of every tensor whose name ends with `suffix`. Panics when nothing matched: a
/// perturbation that silently applied to no tensor is a check that cannot fail.
fn with_fill(mut m: SyntheticModel, suffix: &str, fill: Fill) -> SyntheticModel {
    let mut n = 0;
    for t in m.tensors.iter_mut() {
        if t.name.ends_with(suffix) {
            t.fill = fill.clone();
            n += 1;
        }
    }
    assert!(n > 0, "{suffix} matched no tensor — nothing was perturbed");
    m
}

/// [`with_fill`]'s metadata twin: overwrite an existing key's value (panics if absent).
fn with_meta(mut m: SyntheticModel, key: &str, v: Meta) -> SyntheticModel {
    let mut found = false;
    for (k, mv) in m.meta.iter_mut() {
        if k == key {
            *mv = v.clone();
            found = true;
        }
    }
    assert!(found, "{key} is not in the model — nothing was overwritten");
    m
}

// ─── the MLA YaRN magnitude ──────────────────────────────────────────────────────
//
// ggml's rope multiplies `q_pe` and `k_pe` (and V3.2's indexer q/k) by the YaRN magnitude `m`
// (`Config::mla_rope_magnitude`), so the reference's MLA score is
// `kq_scale · (q_nope·k_nope + m²·q_pe·k_pe)`. The runner applies `m²` to the query's rotary
// columns alone. Each case below checks that against a model with `m = 1` whose query projection
// rows were scaled on the host so that the formula gives the same scores: with powers of two the
// two runs are the same arithmetic up to exact scalings, so they agree to the bit.

/// `m` with `key` added (it must not be there yet), kept in the sorted order `mla_model` writes.
fn with_new_meta(mut m: SyntheticModel, key: &str, v: Meta) -> SyntheticModel {
    assert!(
        m.meta.iter().all(|(k, _)| k != key),
        "{key} is already in the model"
    );
    m.meta.push((key.to_string(), v));
    m.meta.sort_by(|a, b| a.0.cmp(&b.0));
    m
}

/// `m` with every layer's query projection `blk.N.{role}` rescaled per output row: an output
/// column `c` of a `head`-wide head takes `rope_gain` when `c` is in `rope`, `nope_gain` otherwise.
/// Each output row is `in_f` values (the ggml shape's first dimension). The role is matched
/// whole: `attn_q_b.weight` is a suffix of `indexer.attn_q_b.weight`.
fn with_scaled_q_heads(
    mut m: SyntheticModel,
    role: &str,
    in_f: usize,
    head: usize,
    rope: std::ops::Range<usize>,
    (nope_gain, rope_gain): (f32, f32),
) -> SyntheticModel {
    let mut n = 0;
    let is_role = |name: &str| {
        name.strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
            .is_some_and(|(_, r)| r == role)
    };
    for t in m.tensors.iter_mut().filter(|t| is_role(&t.name)) {
        let mut v = t.values();
        for (j, row) in v.chunks_mut(in_f).enumerate() {
            let gain = if rope.contains(&(j % head)) {
                rope_gain
            } else {
                nope_gain
            };
            row.iter_mut().for_each(|x| *x *= gain);
        }
        t.fill = Fill::Exact(v);
        n += 1;
    }
    assert!(n > 0, "{role} matched no tensor — nothing was scaled");
    m
}

/// `m` with a YaRN magnitude of 2, from `rope.scaling.attn_factor`: the score scale is
/// `kq_scale · 4` and the rotary dot `· 4` more. [`unit_magnitude_twin`] is the same model at a
/// magnitude of 1 with its query rows scaled to reproduce those scores.
fn magnitude_two(arch: &str, m: SyntheticModel) -> SyntheticModel {
    with_new_meta(
        m,
        &format!("{arch}.rope.scaling.attn_factor"),
        Meta::F32(2.0),
    )
}

/// [`magnitude_two`]'s reference: magnitude 1, the MLA query's NOPE rows `× 4` (the score scale's
/// factor) and its rotary rows `× 16` (`m²` on top), and, when `indexer` is set, the indexer
/// query's rotary rows `× 4` (its score scale does not involve the magnitude).
fn unit_magnitude_twin(
    m: SyntheticModel,
    d: &Ds2Dims,
    indexer: Option<&IndexerDims>,
) -> SyntheticModel {
    let m = with_scaled_q_heads(
        m,
        "attn_q_b.weight",
        d.q_lora_rank,
        d.key_length_mla,
        d.qk_nope()..d.key_length_mla,
        (4.0, 16.0),
    );
    match indexer {
        None => m,
        Some(ix) => with_scaled_q_heads(
            m,
            "indexer.attn_q_b.weight",
            d.q_lora_rank,
            ix.head_size,
            0..d.qk_rope_dim,
            (1.0, 4.0),
        ),
    }
}

/// The MLA and MulVec ops a CPU prefill of `prompt` ran, in order.
fn mla_and_gain_ops(tag: &str, model: &SyntheticModel) -> Vec<infr_cpu::op_tap::TappedOp> {
    use infr_cpu::op_tap::{self, TappedOp};
    let tmp = TempGguf::write(tag, model);
    let m = load(&tmp);
    op_tap::start();
    let logits = m.prefill_logits_cpu(PROMPT);
    let ops = op_tap::take();
    logits.expect("cpu prefill");
    ops.into_iter()
        .filter(|o| matches!(o, TappedOp::Mla { .. } | TappedOp::MulVec { .. }))
        .collect()
}

/// **A YaRN magnitude of 1 emits nothing new; any other one multiplies the query's rotary
/// columns once per layer, right before the attention.** The canonical deepseek2 model (V2-Lite's
/// YaRN multiplier) and the deepseek32 one at V3.2's ([`at_v32_multiplier`]) have magnitude 1 and
/// must run no `MulVec` at all; at magnitude 2 every `Mla` over `r` rows comes right after one
/// `MulVec` over the MLA query (`[r·n_head, key_length_mla]`) and, on deepseek32, one over the
/// indexer query (`[r·indexer n_head, indexer head_size]`) before that. Both magnitudes run the
/// same attentions. The canonical deepseek32 model, at V2-Lite's multiplier, is at a magnitude
/// other than 1 under its own arch's rule, so it emits the gains too.
#[test]
fn synthetic_deepseek2_mla_magnitude_emits_only_where_it_is_not_one() {
    use infr_cpu::op_tap::TappedOp;
    let d = ds2_dims(N_GROUPS, N_GROUPS_USED, Some(force_bias()));
    let ix = ds32_indexer();
    let mla_rows = |ops: &[TappedOp]| -> Vec<u32> {
        ops.iter()
            .filter_map(|o| match *o {
                TappedOp::Mla { rows } => Some(rows),
                _ => None,
            })
            .collect()
    };
    let with_gains = |rows: &[u32], indexer: bool| -> Vec<TappedOp> {
        rows.iter()
            .flat_map(|&r| {
                let ix_gain = TappedOp::MulVec {
                    rows: r * ix.n_head as u32,
                    n: ix.head_size as u32,
                };
                let mla_gain = TappedOp::MulVec {
                    rows: r * d.n_head as u32,
                    n: d.key_length_mla as u32,
                };
                let mut v = if indexer { vec![ix_gain] } else { vec![] };
                v.extend([mla_gain, TappedOp::Mla { rows: r }]);
                v
            })
            .collect()
    };
    for (arch, unit_model) in [
        (infr_llama::arch::DEEPSEEK2, grouped_model()),
        (
            infr_llama::arch::DEEPSEEK32,
            at_v32_multiplier(deepseek32_model(), infr_llama::arch::DEEPSEEK32),
        ),
    ] {
        let indexer = arch == infr_llama::arch::DEEPSEEK32;
        let unit = mla_and_gain_ops(&format!("{arch}-mag1-ops"), &unit_model);
        let rows = mla_rows(&unit);
        assert!(
            rows.len() >= d.n_layer,
            "{arch}: the prefill ran {} Mla ops",
            rows.len()
        );
        assert_eq!(
            unit,
            with_gains(&rows, indexer)
                .into_iter()
                .filter(|o| matches!(o, TappedOp::Mla { .. }))
                .collect::<Vec<_>>(),
            "{arch} at magnitude 1 ran a MulVec"
        );
        let two = mla_and_gain_ops(
            &format!("{arch}-mag2-ops"),
            &magnitude_two(arch, unit_model),
        );
        assert_eq!(
            mla_rows(&two),
            rows,
            "{arch}: the two magnitudes attend alike"
        );
        assert_eq!(two, with_gains(&rows, indexer), "{arch} at magnitude 2");
    }
    let v2lite = mla_and_gain_ops("ds32-v2lite-ops", &deepseek32_model());
    assert_eq!(
        v2lite,
        with_gains(&mla_rows(&v2lite), true),
        "deepseek32 at V2-Lite's multiplier"
    );
}

/// **The magnitude reaches the rotary half of the score squared, and nothing else.** A deepseek2
/// model at magnitude 2 must reproduce, to the bit, the magnitude-1 model whose query rows give
/// the reference formula's scores (see [`unit_magnitude_twin`]), and must differ from the same
/// file without the magnitude.
#[test]
fn synthetic_deepseek2_mla_magnitude_matches_the_reference_scores() {
    let d = ds2_dims(N_GROUPS, N_GROUPS_USED, Some(force_bias()));
    let two = cpu_logits(
        "ds2-mag2",
        &magnitude_two(infr_llama::arch::DEEPSEEK2, grouped_model()),
    );
    let twin = cpu_logits(
        "ds2-mag2-twin",
        &unit_magnitude_twin(grouped_model(), &d, None),
    );
    let dv = max_abs_diff(&two, &twin);
    println!("deepseek2 magnitude 2 vs its unit twin: max|Δ| = {dv:e}");
    assert_eq!(dv, 0.0, "deepseek2 magnitude 2 differs from its unit twin");
    assert_moves(
        "deepseek2 magnitude 2 vs 1",
        &two,
        &cpu_logits("ds2-mag1", &grouped_model()),
        "the query rotary gain",
    );
}

/// The bound, relative to the logit rms, within which
/// [`synthetic_deepseek2_mla_magnitude_without_a_multiplier_matches_the_reference`]'s two runs
/// must agree. Measured on that file: moving the twin's rotary gain by one f32 ulp either way moved
/// its logits by up to 1.12e-4 of their rms, so a bound under that would fail on rounding alone;
/// the gain itself moves them by about 0.6 of the rms.
const NO_MULTIPLIER_TOL: f32 = 2e-4;

/// YaRN WITHOUT a multiplier leaves the magnitude `1 + 0.1·ln(factor)`, from `get_mscale(factor,
/// 1)` — the other branch of the rule. Its score scale is the one a multiplier of 1 gives, at
/// magnitude 1, so the twin is that file with only its rotary query rows scaled by `m²`. `m` is
/// not a power of two, so the two runs round differently and agree to [`NO_MULTIPLIER_TOL`].
#[test]
fn synthetic_deepseek2_mla_magnitude_without_a_multiplier_matches_the_reference() {
    let d = ds2_dims(N_GROUPS, N_GROUPS_USED, Some(force_bias()));
    let log_mul = "deepseek2.rope.scaling.yarn_log_multiplier";
    let m = 1.0 + 0.1 * 40f64.ln();
    let bare = cpu_logits("ds2-nolm", &without_meta(grouped_model(), log_mul));
    let twin = with_meta(grouped_model(), log_mul, Meta::F32(0.1));
    let twin = with_scaled_q_heads(
        twin,
        "attn_q_b.weight",
        d.q_lora_rank,
        d.key_length_mla,
        d.qk_nope()..d.key_length_mla,
        (1.0, (m * m) as f32),
    );
    let twin = cpu_logits("ds2-nolm-twin", &twin);
    let dv = max_abs_diff(&bare, &twin);
    let scale = rms(&bare);
    println!("deepseek2 magnitude {m} vs its unit twin: max|Δ| = {dv:e} (rms {scale:e})");
    assert!(
        dv < NO_MULTIPLIER_TOL * scale,
        "deepseek2 without a YaRN multiplier differs from its unit twin: max|Δ| = {dv:e}"
    );
    assert_moves(
        &format!("deepseek2 magnitude {m} vs 1"),
        &bare,
        &cpu_logits("ds2-nolm-unit", &grouped_model()),
        "the query rotary gain",
    );
}

/// The deepseek32 case: the gain reaches the lightning indexer's query too, whose rotary half the
/// reference ropes with the same magnitude — so the twin scales both queries, and the indexer must
/// pick the same keys for the logits to agree to the bit. Both files declare V3.2's own YaRN
/// multiplier (1), where the deepseek2 and deepseek32 attention-factor rules agree.
#[test]
fn synthetic_deepseek32_mla_magnitude_matches_the_reference_scores() {
    let d = ds32_dims();
    let ix = ds32_indexer();
    let base = || {
        with_meta(
            deepseek32_model(),
            "deepseek32.rope.scaling.yarn_log_multiplier",
            Meta::F32(0.1),
        )
    };
    let two = cpu_logits(
        "ds32-mag2",
        &magnitude_two(infr_llama::arch::DEEPSEEK32, base()),
    );
    let twin = cpu_logits(
        "ds32-mag2-twin",
        &unit_magnitude_twin(base(), &d, Some(&ix)),
    );
    let dv = max_abs_diff(&two, &twin);
    println!("deepseek32 magnitude 2 vs its unit twin: max|Δ| = {dv:e}");
    assert_eq!(dv, 0.0, "deepseek32 magnitude 2 differs from its unit twin");
    assert_moves(
        "deepseek32 magnitude 2 vs 1",
        &two,
        &cpu_logits("ds32-mag1", &base()),
        "the query rotary gain",
    );
}

/// **deepseek32 keeps the YaRN ratio that `deepseek2` cancels.** `llama-context.cpp` sets
/// `mscale = mscale_all_dims` for `LLM_ARCH_DEEPSEEK2` alone, so the canonical deepseek32 fixture,
/// at V2-Lite's multiplier `L`, runs at the rope magnitude `r = get_mscale(f, 1) /
/// get_mscale(f, L)` (`yarn_attn_factor` times the `1 + 0.1·ln f` its graph multiplies back), and
/// its score scale `(r · get_mscale(f, L))² / √d` is the one a multiplier of 1 gives. Its twin is
/// therefore the same file at V3.2's multiplier ([`at_v32_multiplier`], magnitude 1) with the
/// rotary rows of both queries scaled by `r²`, `r` computed here in f64 from that formula. `r` is
/// not a power of two, so the two runs round differently and agree to [`NO_MULTIPLIER_TOL`]; the
/// twin's file without the scaled rows (magnitude 1) is far outside that bound.
#[test]
fn synthetic_deepseek32_yarn_rule_matches_the_reference() {
    let d = ds32_dims();
    let ix = ds32_indexer();
    let model = deepseek32_model();
    let meta_f32 = |key: &str| match model.meta.iter().find(|(k, _)| k == key) {
        Some((_, Meta::F32(v))) => f64::from(*v),
        other => panic!("{key}: {other:?}"),
    };
    let factor = meta_f32("deepseek32.rope.scaling.factor");
    // The file stores `0.1 · mscale_all_dim`; the loader divides the 0.1 back out.
    let log_mul = meta_f32("deepseek32.rope.scaling.yarn_log_multiplier") / 0.1;
    let get_mscale = |m: f64| 0.1 * m * factor.ln() + 1.0;
    let r = get_mscale(1.0) / get_mscale(log_mul);
    let gain = (r * r) as f32;
    let twin = at_v32_multiplier(deepseek32_model(), infr_llama::arch::DEEPSEEK32);
    let twin = with_scaled_q_heads(
        twin,
        "attn_q_b.weight",
        d.q_lora_rank,
        d.key_length_mla,
        d.qk_nope()..d.key_length_mla,
        (1.0, gain),
    );
    let twin = with_scaled_q_heads(
        twin,
        "indexer.attn_q_b.weight",
        d.q_lora_rank,
        ix.head_size,
        0..d.qk_rope_dim,
        (1.0, gain),
    );
    let got = cpu_logits("ds32-v2lite", &model);
    let want = cpu_logits("ds32-v2lite-twin", &twin);
    let dv = max_abs_diff(&got, &want);
    let scale = rms(&got);
    println!("deepseek32 YaRN ratio {r} vs its unit twin: max|Δ| = {dv:e} (rms {scale:e})");
    assert!(
        dv < NO_MULTIPLIER_TOL * scale,
        "deepseek32 at V2-Lite's multiplier differs from its unit twin: max|Δ| = {dv:e}"
    );
    assert_moves(
        &format!("deepseek32 YaRN ratio {r} vs 1"),
        &got,
        &cpu_logits(
            "ds32-v2lite-unit",
            &at_v32_multiplier(deepseek32_model(), infr_llama::arch::DEEPSEEK32),
        ),
        "the deepseek32 YaRN ratio",
    );
}

/// [`synthetic_deepseek32_mla_magnitude_matches_the_reference_scores`] on Vulkan: the gain is an
/// in-place `MulVec` on each query there too. The two runs round alike only up to the device's
/// reassociation, so they agree to a tolerance.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn vulkan_synthetic_deepseek32_mla_magnitude_matches_the_reference_scores() {
    let _g = gpu_serial_lock();
    let d = ds32_dims();
    let ix = ds32_indexer();
    let base = || {
        with_meta(
            deepseek32_model(),
            "deepseek32.rope.scaling.yarn_log_multiplier",
            Meta::F32(0.1),
        )
    };
    let two = vulkan_logits(
        "vk-ds32-mag2",
        &magnitude_two(infr_llama::arch::DEEPSEEK32, base()),
    );
    let twin = vulkan_logits(
        "vk-ds32-mag2-twin",
        &unit_magnitude_twin(base(), &d, Some(&ix)),
    );
    let dv = max_abs_diff(&two, &twin);
    let scale = rms(&two);
    println!("vulkan deepseek32 magnitude 2 vs its unit twin: max|Δ| = {dv:e} (rms {scale:e})");
    assert!(
        dv < 1e-3 * scale,
        "vulkan deepseek32 magnitude 2 differs from its unit twin: max|Δ| = {dv:e}"
    );
    assert_moves(
        "vulkan deepseek32 magnitude 2 vs 1",
        &two,
        &vulkan_logits("vk-ds32-mag1", &base()),
        "the query rotary gain",
    );
}

// ─── deepseek4 hash-routed MoE ────────────────────────────────────────────────────
//
// A hash-routed layer takes its experts from `blk.N.ffn_gate_tid2eid` — an i32
// `[n_expert_used, n_vocab]` table indexed by TOKEN ID — instead of the router's top-k.
// `Op::GatherI32` produces the `[batch, n_expert_used]` selection and `Op::MoeFfn::expert_ids`
// runs it. The router still runs: the routing WEIGHTS are its own sqrt-softplus probs at the
// hash-chosen experts (`seam_op_parity.rs`'s `moe_hash_routing_parity` pins that arithmetic; what
// is pinned HERE is that the whole-model emit reads the right table row for the right token).
//
// Every test below is built so that it CANNOT pass if the emit fell back to `expert_ids: None`:
// each pair of files differs in the routing table and in nothing else, so an emit that never reads
// the table produces two bit-identical runs.

/// The all-ratio-0, ALL-HASH-ROUTED V4 fixture: [`dsv4_ratio0_dims`] with every layer taking its
/// experts from `ffn_gate_tid2eid`. `hash_layer_count == n_layer` means no layer carries an
/// `exp_probs_b` at all, so nothing here can be passing on the bias-routed path by accident.
fn dsv4_hash_dims() -> Dsv4Dims {
    let base = dsv4_ratio0_dims();
    Dsv4Dims {
        hash_layer_count: base.n_layer(),
        ..base
    }
}

/// A `ffn_gate_tid2eid` fill of `d` whose row for token `t` names `pick(t)` — the same
/// `[n_used, vocab]` table every hash layer of the fixture gets.
fn dsv4_tid2eid(d: &Dsv4Dims, pick: impl Fn(usize) -> Vec<i32>) -> Fill {
    let mut v = Vec::with_capacity(d.n_used * d.vocab);
    for t in 0..d.vocab {
        let row = pick(t);
        assert_eq!(row.len(), d.n_used, "a table row is n_expert_used wide");
        assert!(
            row.iter().all(|&e| (0..d.n_expert as i32).contains(&e)),
            "expert id out of range: {row:?}"
        );
        v.extend(row);
    }
    Fill::ExactIds(v)
}

/// The three all-hash V4 files the token-id-indexing check compares, as
/// `(inside, outside, [base, bumped_inside, bumped_outside])`. They differ in `ffn_gate_tid2eid`
/// and in NOTHING else, which is what makes the comparison a check on the gather rather than on
/// the model: an emit that never reads the table returns three identical runs.
///
/// The baseline routes token `t` to experts `{t, t+1} mod n_expert`, so the selection is
/// token-dependent before anything is perturbed; each perturbation rotates ONE token's row by one
/// expert — `inside` is a token the shared [`PROMPT`] contains, `outside` one it never does.
fn dsv4_hash_row_variants(d: &Dsv4Dims) -> (usize, usize, [SyntheticModel; 3]) {
    let inside = PROMPT[PROMPT.len() - 1] as usize;
    let outside = (0..d.vocab)
        .find(|t| !PROMPT.contains(&(*t as u32)))
        .expect("the fixture's vocab is wider than the prompt");
    let row = |t: usize, target: Option<usize>| -> Vec<i32> {
        let bump = i32::from(target == Some(t));
        (0..d.n_used)
            .map(|k| ((t + k) as i32 + bump).rem_euclid(d.n_expert as i32))
            .collect()
    };
    let model = |target: Option<usize>| {
        with_fill(
            dsv4_model(d),
            "ffn_gate_tid2eid.weight",
            dsv4_tid2eid(d, |t| row(t, target)),
        )
    };
    (
        inside,
        outside,
        [model(None), model(Some(inside)), model(Some(outside))],
    )
}

/// **The hash table drives the selection, and it is read BY TOKEN ID.** Two halves, and the second
/// is what makes the first mean something:
///
/// 1. Changing the table row of a token that IS in the prompt moves the logits — the gather ran and
///    its value reached the experts.
/// 2. Changing the table row of a token that is NOT in the prompt is EXACTLY inert — the gather
///    read that token's row and no other. A gather that ignored `ids` (a constant row, row 0, a
///    whole-table reduction) fails this half while passing the first.
///
/// Neither half can pass on a `expert_ids: None` emit: the three files differ only in
/// `ffn_gate_tid2eid`, so an emit that never reads it makes all three runs bit-identical and
/// half (1) goes red.
#[test]
fn synthetic_deepseek4_hash_table_is_read_by_token_id() {
    let d = dsv4_hash_dims();
    let (inside, outside, m) = dsv4_hash_row_variants(&d);
    let plain = cpu_logits("ds4-hash-base", &m[0]);
    let moved = cpu_logits("ds4-hash-inside", &m[1]);
    let inert = cpu_logits("ds4-hash-outside", &m[2]);
    assert!(
        plain.iter().all(|v| v.is_finite()),
        "non-finite logit in the hash-routed V4 prefill: {plain:?}"
    );
    assert_moves(
        &format!("deepseek4 tid2eid row for prompt token {inside}"),
        &plain,
        &moved,
        "the ffn_gate_tid2eid gather (a top-k fallback never reads the table at all)",
    );
    let dv = max_abs_diff(&plain, &inert);
    println!(
        "deepseek4 tid2eid row for non-prompt token {outside}: max|Δ| = {dv:e} (logit rms {:e})",
        rms(&plain)
    );
    assert_eq!(
        dv, 0.0,
        "changing the table row of token {outside}, which the prompt never contains, moved the \
         logits — the gather is not indexing the table by token id"
    );
}

/// **Hash routing picks experts the router's own top-k does not.** Three tables naming three
/// different constant expert pairs, plus a top-k control: the control is the same model with
/// `hash_layer_count = 0` and an all-zero `exp_probs_b`, i.e. plain top-k over the same router.
///
/// Top-k picks ONE pair per token, so at most one of the three hash variants can agree with the
/// control — hence the `>= 2` bound, which is exactly what "the selection does not come from the
/// router" means here and is falsifiable: on a `expert_ids: None` emit all three variants ARE the
/// control and the count is 0.
#[test]
fn synthetic_deepseek4_hash_routing_overrides_the_router_topk() {
    let d = dsv4_hash_dims();
    let pairs: [[i32; 2]; 3] = [[0, 1], [2, 3], [1, 2]];
    assert_eq!(
        d.n_used, 2,
        "the constant pairs below are n_expert_used wide"
    );
    let runs: Vec<Vec<f32>> = pairs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let fill = dsv4_tid2eid(&d, |_| p.to_vec());
            cpu_logits(
                &format!("ds4-hash-pair{i}"),
                &with_fill(dsv4_model(&d), "ffn_gate_tid2eid.weight", fill),
            )
        })
        .collect();

    // Every pair of variants must differ: the tables are the only difference between the files.
    for (i, a) in runs.iter().enumerate() {
        for (j, b) in runs.iter().enumerate().skip(i + 1) {
            assert_moves(
                &format!("deepseek4 hash pair {:?} vs {:?}", pairs[i], pairs[j]),
                a,
                b,
                "the ffn_gate_tid2eid selection",
            );
        }
    }

    // The top-k control: same dims, bias routing with a ZERO bias, so selection is the router's
    // own ranking with nothing added.
    let mut ctl_dims = d.clone();
    ctl_dims.hash_layer_count = 0;
    let control = cpu_logits(
        "ds4-hash-topk-control",
        &with_fill(
            dsv4_model(&ctl_dims),
            "exp_probs_b.bias",
            Fill::Exact(vec![0.0; d.n_expert]),
        ),
    );
    let differing = runs
        .iter()
        .enumerate()
        .filter(|(i, r)| {
            let dv = max_abs_diff(r, &control);
            println!(
                "deepseek4 hash pair {:?} vs router top-k: max|Δ| = {dv:e}",
                pairs[*i]
            );
            dv > 0.01 * rms(&control).max(rms(r))
        })
        .count();
    assert!(
        differing >= 2,
        "only {differing} of the three constant hash tables routed differently from the router's \
         own top-k — top-k picks one pair per token, so at least two must differ; this is what \
         goes to 0 if the emit ignores the table and re-runs top-k"
    );
    assert!(
        rms(&control) > 1e-3,
        "the top-k control produced degenerate logits, so the comparison proves nothing"
    );
}

/// **A ratio-0 V4 model generates.** The end-to-end claim: the real loader, the real `wload`/`wpush`
/// lockstep and the real emit produce finite, non-degenerate logits on the CPU reference backend.
///
/// The all-equal check matters as much as the finiteness one: a hyper-connection wrap that zeroed
/// the residual, or a SwiGLU clamp applied with a `Some(0.0)` limit, both produce a perfectly
/// finite constant vector.
#[test]
fn synthetic_deepseek4_ratio0_prefill_is_finite() {
    let d = dsv4_ratio0_dims();
    let logits = cpu_logits("ds4-r0", &dsv4_ratio0_model());
    assert_eq!(logits.len(), d.vocab, "one logit per vocab entry");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "non-finite logit in the ratio-0 V4 prefill: {logits:?}"
    );
    let lo = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "deepseek4 ratio-0 logits: min {lo:e} max {hi:e} rms {:e}",
        rms(&logits)
    );
    assert!(
        hi - lo > 1e-3,
        "the ratio-0 V4 logits are constant (min {lo:e}, max {hi:e}) — a collapsed residual and a \
         zero-limit SwiGLU clamp both look exactly like this"
    );
}

/// **The V KV-cache type does not reach deepseek4.** V4 has no V cache (its attention reads the
/// K cache as V), so `INFR_KV_TYPE_V` must change nothing: these runs set it to formats the CPU
/// backend accepts at this head width — `turbo3` (a 128-wide WHT group), `q4_0` (32-blocks) and
/// `f32` — and each must reproduce the default run bit for bit. Head dim 128 so every one of
/// them passes the runner's format gates and is really requested, not quietly downgraded.
///
/// Before the V side followed K, a V format still declared a per-layer placeholder Input in
/// that dtype, which the CPU executor loaded and dequantized each step: `turbo3` panicked in
/// the host dequant.
#[test]
fn synthetic_deepseek4_v_kv_type_is_ignored() {
    let d = Dsv4Dims {
        head_dim: 128,
        ..dsv4_ratio0_dims()
    };
    let tmp = TempGguf::write("ds4-vtype", &dsv4_model(&d));
    let run = |v: Option<infr_core::DType>| {
        let ec = infr_llama::EngineConfig {
            kv: infr_core::config::KvCfg {
                type_v: v,
                type_v_specified: v.is_some(),
                ..Default::default()
            },
            ..Default::default()
        };
        infr_llama::SeamModel::load_with(tmp.path(), None, std::sync::Arc::new(ec))
            .expect("synthetic model load")
            .prefill_logits_cpu(PROMPT)
            .expect("cpu prefill")
    };
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let base = run(None);
    assert!(
        rms(&base) > 1e-3,
        "degenerate logits (rms {:e})",
        rms(&base)
    );
    for dt in [
        infr_core::DType::Turbo3,
        infr_core::DType::Q4_0,
        infr_core::DType::F32,
    ] {
        assert_eq!(
            bits(&run(Some(dt))),
            bits(&base),
            "INFR_KV_TYPE_V={dt:?} changed deepseek4's logits"
        );
    }
}

/// **The cross-stream Sinkhorn mixing is load-bearing.** Two files identical in every tensor except
/// the `comb` chunk of the per-layer hyper-connection `base` vectors, which is the ONLY part of the
/// wrap that has anything to mix: with the residual stream collapsed to one, `comb` is a 1x1 matrix
/// whose Sinkhorn normalisation divides it by itself, so the perturbation would be exactly inert
/// and the two runs identical.
///
/// The baseline is a flat all-zero `base` rather than the fixture's random one, so the two variants
/// differ in `hc_mult*hc_mult` numbers and in nothing else — including the `pre` and `post` chunks,
/// which are byte-identical in both.
///
/// The `hc_mult = 1` arm is the CONTROL that makes the assertion falsifiable rather than decorative:
/// the same perturbation, on the same emit, with one residual stream, must be EXACTLY inert. If it
/// moved anything there, the `hc_mult = 2` arm would be measuring something other than stream
/// mixing; if the emit collapsed the streams, the `hc_mult = 2` arm would go inert too.
#[test]
fn synthetic_deepseek4_stream_mixing_is_load_bearing() {
    // `comb`'s chunk starts at `2*hc` and its flat index is `dst + hc*src`. At hc=2, bump ONE
    // off-diagonal entry (dst=1, src=0) — the softmax over `dst` then pushes stream 0's mass onto
    // stream 1. At hc=1 the chunk is the single entry `2*hc + 0`, and Sinkhorn's normalisation of a
    // 1x1 matrix divides it by itself, so no value there can reach the output.
    let pair = |hc_mult: usize| -> (SyntheticModel, SyntheticModel) {
        let d = Dsv4Dims {
            hc_mult,
            ..dsv4_ratio0_dims()
        };
        let base_fill = |bump: bool| {
            let mut v = vec![0.0f32; d.hc_mix_dim()];
            if bump {
                v[2 * hc_mult + usize::from(hc_mult > 1)] = 4.0;
            }
            Fill::Exact(v)
        };
        let variant = |bump: bool| {
            let m = with_fill(dsv4_model(&d), "hc_attn_base.weight", base_fill(bump));
            with_fill(m, "hc_ffn_base.weight", base_fill(bump))
        };
        (variant(false), variant(true))
    };

    let (flat, bumped) = pair(dsv4_ratio0_dims().hc_mult);
    // The premise: the two files differ in the comb chunk and nowhere else.
    let diffs: Vec<String> = weights_of(&flat)
        .into_iter()
        .zip(weights_of(&bumped))
        .filter(|((_, a), (_, b))| a != b)
        .map(|((n, _), _)| n)
        .collect();
    assert_eq!(
        diffs.len(),
        2 * dsv4_ratio0_dims().n_layer(),
        "only the per-layer hc base vectors may differ, got: {diffs:?}"
    );
    let a = cpu_logits("ds4-hc-flat", &flat);
    let b = cpu_logits("ds4-hc-comb", &bumped);
    assert_moves(
        "deepseek4 comb bump",
        &a,
        &b,
        "the cross-stream Sinkhorn mixing (`comb`), which is inert on a single residual stream",
    );

    // The control: one stream, same perturbation, bit-identical output.
    let (flat1, bumped1) = pair(1);
    let a1 = cpu_logits("ds4-hc1-flat", &flat1);
    let b1 = cpu_logits("ds4-hc1-comb", &bumped1);
    let d1 = max_abs_diff(&a1, &b1);
    println!(
        "deepseek4 comb bump at hc_mult=1: max|Δ| = {d1:e} (logit rms {:e})",
        rms(&a1)
    );
    assert_eq!(
        d1, 0.0,
        "at hc_mult=1 the comb chunk has nothing to mix and Sinkhorn normalises it away, so this \
         perturbation must be EXACTLY inert — a non-zero difference means the hc_mult=2 arm above \
         is measuring some other sensitivity to the same bytes"
    );
    assert!(
        rms(&a1) > 1e-3,
        "the hc_mult=1 control produced degenerate logits, so its inertness proves nothing"
    );
}

/// The ratio-0 fixture twice over, differing ONLY in how every `ffn_gate_inp` router is stored:
/// f32, and bf16 — the shipped V4 file's router dtype. Both carry the same bf16-representable
/// values, so any difference between the two runs is the bf16 read, not the rounding.
fn dsv4_router_dtype_pair() -> (SyntheticModel, SyntheticModel) {
    let (mut f32_router, mut bf16_router) = (dsv4_ratio0_model(), dsv4_ratio0_model());
    for (a, b) in f32_router
        .tensors
        .iter_mut()
        .zip(bf16_router.tensors.iter_mut())
    {
        if !a.name.ends_with("ffn_gate_inp.weight") {
            continue;
        }
        let v: Vec<f32> = a
            .values()
            .iter()
            .map(|x| half::bf16::from_f32(*x).to_f32())
            .collect();
        a.fill = Fill::Exact(v.clone());
        b.fill = Fill::ExactBf16(v);
    }
    assert!(
        bf16_router
            .tensors
            .iter()
            .any(|t| matches!(t.fill, Fill::ExactBf16(_))),
        "no router was rewritten — the comparison would be a model against itself"
    );
    (f32_router, bf16_router)
}

/// **A bf16 router reads as the numbers it holds.** Every `ffn_gate_inp` in the shipped V4 file is
/// bf16, and no other V4 test stores one that way. Same values, two encodings: the routing — and
/// so the logits — must agree to accumulation-order noise, where a bf16 tensor misread as anything
/// else routes different experts.
#[test]
fn synthetic_deepseek4_bf16_router_matches_f32() {
    let (f32_router, bf16_router) = dsv4_router_dtype_pair();
    let a = cpu_logits("ds4-router-f32", &f32_router);
    let b = cpu_logits("ds4-router-bf16", &bf16_router);
    let dv = max_abs_diff(&a, &b);
    println!(
        "deepseek4 bf16 vs f32 router: max|Δ| = {dv:e} (logit rms {:e})",
        rms(&a)
    );
    assert!(
        dv <= 1e-4 * rms(&a),
        "a bf16 router holding the f32 router's values moved the logits by {dv:e}"
    );
}

/// [`synthetic_deepseek4_bf16_router_matches_f32`] on Vulkan, against the CPU's f32 router: the GPU
/// binder may re-encode float weights on upload, which is a second place a bf16 router can go wrong.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_deepseek4_bf16_router_matches_cpu() {
    let _lk = gpu_serial_lock();
    let (f32_router, bf16_router) = dsv4_router_dtype_pair();
    let cpu = cpu_logits("ds4-router-f32-cpu", &f32_router);
    let gpu = vulkan_logits("ds4-router-bf16-vk", &bf16_router);
    let cos = cosine(&cpu, &gpu);
    println!("synthetic deepseek4 bf16 router cpu-vs-vulkan cosine = {cos}");
    assert_eq!(
        argmax(&cpu),
        argmax(&gpu),
        "CPU and Vulkan top tokens differ"
    );
    assert!(
        cos > 0.9999,
        "CPU/Vulkan logits diverged with a bf16 router: cosine = {cos}"
    );
}

/// **The de-rope is load-bearing, and it is an exact inverse of the forward rope.**
///
/// Two halves, because either alone is passable by an implementation that ropes nothing at all:
///
/// 1. `rope.freq_base` moves the logits — the forward rope really runs.
/// 2. At `sliding_window = 1` every query attends ONLY its own position, so `q·k` is a dot of two
///    vectors rotated by the SAME angle and is position-independent, and the attention output is a
///    position-independent scalar times the (roped) KV row of that position. The de-rope then
///    rotates that row back by the query position, cancelling exactly — so on a prompt of a REPEATED
///    token, where every position's layer input is identical, the last row's logits must not depend
///    on the prompt LENGTH. Drop the de-rope (or the forward q rope) and each position keeps a
///    different rotation, and they do.
///
/// The tolerance in (2) is not float noise: the cancellation is exact in real arithmetic but the
/// cache holds the ROPED row in f16, so what comes back is `R(-t)·round_f16(R(t)·kv)` and the
/// rounding is itself position-dependent. Measured on this fixture: **1.5e-6** with an f32 KV
/// cache, **1.6e-3** with the default f16 one, and **1.0e0** — the whole signal — with the de-rope
/// deleted. The threshold sits between the second and the third.
#[test]
fn synthetic_deepseek4_derope_inverts_the_query_rope() {
    // (1) the forward rope is not a no-op.
    let plain = cpu_logits("ds4-theta-10k", &dsv4_ratio0_model());
    let retuned = cpu_logits(
        "ds4-theta-500",
        &with_meta(
            dsv4_ratio0_model(),
            "deepseek4.rope.freq_base",
            Meta::F32(500.0),
        ),
    );
    assert_moves(
        "deepseek4 rope theta",
        &plain,
        &retuned,
        "the [nope|rope] tail rope",
    );

    // (2) window 1 ⇒ rope and de-rope cancel ⇒ a repeated-token prompt is position-independent.
    let d = Dsv4Dims {
        swa_window: 1,
        ..dsv4_ratio0_dims()
    };
    let m = dsv4_model(&d);
    let short = cpu_logits_of("ds4-w1-short", &m, &[7, 7, 7]);
    let long = cpu_logits_of("ds4-w1-long", &m, &[7, 7, 7, 7, 7, 7, 7, 7, 7]);
    let dv = max_abs_diff(&short, &long);
    let scale = rms(&short).max(rms(&long));
    println!("deepseek4 window-1 position independence: max|Δ| = {dv:e} (logit rms {scale:e})");
    assert!(
        dv < 1e-2 * scale,
        "the last row's logits moved with the prompt LENGTH (max|Δ| = {dv:e}, logit rms {scale:e}) \
         on a repeated-token prompt at sliding_window=1 — the only position dependence left there \
         is the attention output's rope slice, so the backward de-rope is missing or is not the \
         inverse of the forward query rope"
    );
    // ...and the run it is being compared against is a real one, not two copies of a degenerate
    // constant that would satisfy any tolerance.
    assert!(
        scale > 1e-3,
        "window-1 logits are degenerate (rms {scale:e})"
    );
}

/// The grouped output projection's `w_off`: **CPU vs Vulkan on the same file.** `o_group_count = 2`
/// means group 1 reads `wo_a` rows `[o_lora_rank, 2*o_lora_rank)` through `Op::Linear::w_off`, and
/// the fixture's `wo_a` is f32 — the dtype the Vulkan adapter refused a weight offset on until this
/// slice taught its f32 GEMV to take the slice as a shifted device address. A Vulkan run that
/// ignored `w_off` would read group 0's rows twice and land here as a logit mismatch.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_deepseek4_ratio0_matches_cpu() {
    let _lk = gpu_serial_lock();
    let model = dsv4_ratio0_model();
    let cpu = cpu_logits("ds4-r0-parity-cpu", &model);
    let gpu = vulkan_logits("ds4-r0-parity-vk", &model);
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "non-finite logit in the Vulkan ratio-0 V4 prefill: {gpu:?}"
    );
    let cos = cosine(&cpu, &gpu);
    println!("synthetic deepseek4 ratio-0 cpu-vs-vulkan cosine = {cos}");
    let (cpu_top, gpu_top) = (argmax(&cpu), argmax(&gpu));
    println!("cpu argmax = {cpu_top}, vulkan argmax = {gpu_top}");
    assert_eq!(cpu_top, gpu_top, "CPU and Vulkan disagree on the top token");
    // Same reasoning as `gpu_synthetic_deepseek2_matches_cpu`: five f32 layers with a pinned
    // routed set, so the two backends agree far past this. It still fails on any real divergence
    // — a dropped `w_off`, a missing sink, a de-rope that ran forwards.
    assert!(
        cos > 0.9999,
        "CPU/Vulkan logits diverged on the synthetic ratio-0 deepseek4 model: cosine = {cos}"
    );
}

/// **Hash routing on Vulkan: the same numbers as the CPU, and the same token-id indexing.**
/// `gather_i32.comp` is a separate implementation of the gather, so the CPU test above says nothing
/// about it. Three claims on one fixture:
///
/// 1. the logits agree with the CPU reference (cosine + top token) — a gather that read a
///    neighbouring row would run a different expert and land here;
/// 2. perturbing the table row of a prompt token moves the Vulkan logits;
/// 3. perturbing the row of a non-prompt token is EXACTLY inert on Vulkan too.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_deepseek4_hash_routing_matches_cpu() {
    let _lk = gpu_serial_lock();
    let d = dsv4_hash_dims();
    let (inside, outside, m) = dsv4_hash_row_variants(&d);
    let cpu = cpu_logits("ds4-hash-parity-cpu", &m[0]);
    let gpu = vulkan_logits("ds4-hash-parity-vk", &m[0]);
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "non-finite logit in the Vulkan hash-routed V4 prefill: {gpu:?}"
    );
    let cos = cosine(&cpu, &gpu);
    let (cpu_top, gpu_top) = (argmax(&cpu), argmax(&gpu));
    println!("synthetic deepseek4 hash-routed cpu-vs-vulkan cosine = {cos}");
    println!("cpu argmax = {cpu_top}, vulkan argmax = {gpu_top}");
    assert_eq!(
        cpu_top, gpu_top,
        "CPU and Vulkan disagree on the top token of a hash-routed model — the two gathers \
         selected different experts"
    );
    assert!(
        cos > 0.9999,
        "CPU/Vulkan logits diverged on the hash-routed deepseek4 model: cosine = {cos}"
    );

    let moved = vulkan_logits("ds4-hash-inside-vk", &m[1]);
    let inert = vulkan_logits("ds4-hash-outside-vk", &m[2]);
    assert_moves(
        &format!("vulkan deepseek4 tid2eid row for prompt token {inside}"),
        &gpu,
        &moved,
        "the ffn_gate_tid2eid gather",
    );
    let dv = max_abs_diff(&gpu, &inert);
    println!(
        "vulkan deepseek4 tid2eid row for non-prompt token {outside}: max|Δ| = {dv:e} (rms {:e})",
        rms(&gpu)
    );
    assert_eq!(
        dv, 0.0,
        "vulkan: changing the table row of token {outside}, which the prompt never contains, moved \
         the logits — gather_i32.comp is not indexing the table by token id"
    );
}

/// The two mechanism tests above, repeated on Vulkan: `hyper_mix.comp`'s Sinkhorn and `rope_back`
/// are separate implementations of the same rules, so proving them on the CPU proves nothing about
/// the GPU.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_deepseek4_mechanisms_execute() {
    let _lk = gpu_serial_lock();
    let d = dsv4_ratio0_dims();
    let (hc, mix_dim) = (d.hc_mult, d.hc_mix_dim());
    let base_fill = |bump: bool| {
        let mut v = vec![0.0f32; mix_dim];
        if bump {
            v[2 * hc + 1] = 4.0;
        }
        Fill::Exact(v)
    };
    let variant = |bump: bool| {
        let m = with_fill(dsv4_ratio0_model(), "hc_attn_base.weight", base_fill(bump));
        with_fill(m, "hc_ffn_base.weight", base_fill(bump))
    };
    let a = vulkan_logits("vk-ds4-hc-flat", &variant(false));
    let b = vulkan_logits("vk-ds4-hc-comb", &variant(true));
    assert_moves(
        "vulkan deepseek4 comb bump",
        &a,
        &b,
        "the cross-stream Sinkhorn mixing (`comb`), which is inert on a single residual stream",
    );

    let dw = Dsv4Dims {
        swa_window: 1,
        ..dsv4_ratio0_dims()
    };
    let m = dsv4_model(&dw);
    let tmp_s = TempGguf::write("vk-ds4-w1-short", &m);
    let tmp_l = TempGguf::write("vk-ds4-w1-long", &m);
    let short = load(&tmp_s)
        .prefill_logits_vulkan(&[7, 7, 7])
        .expect("vulkan prefill");
    let long = load(&tmp_l)
        .prefill_logits_vulkan(&[7, 7, 7, 7, 7, 7, 7, 7, 7])
        .expect("vulkan prefill");
    let dv = max_abs_diff(&short, &long);
    let scale = rms(&short).max(rms(&long));
    println!("vulkan deepseek4 window-1 position independence: max|Δ| = {dv:e} (rms {scale:e})");
    assert!(
        dv < 1e-2 * scale,
        "vulkan: the last row's logits moved with the prompt LENGTH (max|Δ| = {dv:e}, rms \
         {scale:e}) — the backward de-rope is missing or is not the forward rope's inverse"
    );
    assert!(
        scale > 1e-3,
        "window-1 logits are degenerate (rms {scale:e})"
    );
}

// ─── Mistral-Large-3's attention temperature, against llama.cpp ───────────────────
//
// Mistral-Large-3 is a `deepseek2` file with `attention.temperature_scale` and
// `attention.temperature_length`: the reference multiplies every layer's query at position `p` by
// `ln(floor(p / length) + 1) · scale + 1` (`llm_graph_input_attn_temp`), and the runner emits
// that as `Op::AttnTemp` on the MLA query. The oracle is llama.cpp itself, on a synthetic file of
// that shape: `fixtures/synthetic_mistral_large3_llama_cpp.txt` holds the logits it computed at
// several prompt lengths, below and past the temperature length.
//
// ```text
// INFR_LLAMA_DEBUG=<llama.cpp>/build/bin/llama-debug [INFR_LLAMA_FIXTURE_OUT=<dir>] \
//   cargo test -p infr-llama --test synthetic_deepseek2 -- --ignored \
//   synthetic_mistral_large3_regenerate_llama_cpp_fixture
// ```

/// The temperature length of the Mistral-Large-3-shaped file, which its converter writes as the
/// YaRN original context (`MistralMoeModel::set_gguf_parameters`), so both keys carry it. Short,
/// so that a prompt of a few dozen tokens crosses it more than once.
const ML3_TEMP_LENGTH: u32 = 16;

/// `attention.temperature_scale` (the converter's `llama_4_scaling.beta`): the scale `llama4.cpp`
/// hardcodes for the same temperature, since the real file's was not read here.
const ML3_TEMP_SCALE: f32 = 0.1;

/// The prompt lengths scored against llama.cpp: every position below the temperature length (the
/// context the file was capped at before the temperature was applied), the first position at it,
/// and two more steps of the floor past it.
const ML3_PROMPT_LENS: [usize; 3] = [ML3_TEMP_LENGTH as usize, ML3_TEMP_LENGTH as usize + 1, 40];

/// The llama.cpp logits for [`mistral_large3_model`] at each of [`ML3_PROMPT_LENS`].
const ML3_ORACLE_FIXTURE: &str = "tests/fixtures/synthetic_mistral_large3_llama_cpp.txt";

/// The expert-bank format of [`mistral_large3_quantized_model`]: one the batched MoE prefill has a
/// kernel for.
const ML3_EXPERT_DTYPE: infr_core::tensor::DType = infr_core::tensor::DType::Q8_0;

/// `expert_gating_func` softmax, the scoring Mistral-Large-3's converter declares.
const SOFTMAX_GATING: u32 = 1;

/// Token `id`'s text in the oracle file's vocabulary: the printable ASCII characters from `!` on,
/// each its own GPT-2 byte-level symbol, so with no merges llama.cpp's tokenizer reads a string of
/// them back as exactly these ids (the scheme `synthetic_lfm3.rs`'s `token_text` uses).
fn ml3_token_text(id: u32) -> char {
    char::from(b'!' + u8::try_from(id).expect("a vocab id"))
}

/// A Mistral-Large-3-shaped `deepseek2` file: the canonical MLA model with softmax routing and no
/// router bias, YaRN with the multiplier its converter writes (`mscale_all_dim` 1), the attention
/// temperature, and a vocabulary llama.cpp tokenizes back to chosen ids, with no BOS.
fn mistral_large3_model() -> SyntheticModel {
    let k = |s: &str| format!("deepseek2.{s}");
    let d = ds2_dims(N_GROUPS, N_GROUPS_USED, None);
    let mut m = deepseek2_model(&d);
    m = with_meta(m, &k("expert_gating_func"), Meta::U32(SOFTMAX_GATING));
    m = with_meta(m, &k("rope.scaling.yarn_log_multiplier"), Meta::F32(0.1));
    m = with_meta(
        m,
        &k("rope.scaling.original_context_length"),
        Meta::U32(ML3_TEMP_LENGTH),
    );
    m = with_meta(
        m,
        "tokenizer.ggml.tokens",
        Meta::StrArr(
            (0..d.vocab as u32)
                .map(|i| ml3_token_text(i).to_string())
                .collect(),
        ),
    );
    m = with_new_meta(m, "tokenizer.ggml.add_bos_token", Meta::Bool(false));
    // The absorbed-MLA cache shape every DeepSeek converter writes (`DeepseekV2Model`): one KV
    // head of one `[latent | rope]` key row, with the latent as the value. llama.cpp sizes its
    // cache from them; infr derives all three from the MLA keys.
    m = with_new_meta(m, &k("attention.head_count_kv"), Meta::U32(1));
    m = with_new_meta(m, &k("attention.key_length"), Meta::U32(d.kv_row() as u32));
    m = with_new_meta(
        m,
        &k("attention.value_length"),
        Meta::U32(d.kv_lora_rank as u32),
    );
    m = with_new_meta(
        m,
        &k("attention.temperature_length"),
        Meta::U32(ML3_TEMP_LENGTH),
    );
    with_new_meta(
        m,
        &k("attention.temperature_scale"),
        Meta::F32(ML3_TEMP_SCALE),
    )
}

/// [`mistral_large3_model`] with the temperature off (scale 0, the reference's "off") and nothing
/// else changed.
fn mistral_large3_without_temperature() -> SyntheticModel {
    with_meta(
        mistral_large3_model(),
        "deepseek2.attention.temperature_scale",
        Meta::F32(0.0),
    )
}

/// [`mistral_large3_model`] with its expert banks in [`ML3_EXPERT_DTYPE`]. The f32 banks of the
/// oracle file have no batched MoE kernel, so every prompt token of it runs as its own one-row
/// graph; these do, so a prompt prefills in multi-row chunks.
fn mistral_large3_quantized_model() -> SyntheticModel {
    let mut m = mistral_large3_model();
    for bank in [
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    ] {
        m = with_fill(m, bank, Fill::Quant(ML3_EXPERT_DTYPE));
    }
    m
}

/// The first `len` ids of the oracle prompt: drawn from the vocabulary minus the double quote and
/// the backslash, which the command line and llama.cpp's prompt escaping would rewrite, and minus
/// the EOS id.
fn ml3_prompt(len: usize) -> Vec<u32> {
    let vocab = ds2_dims(N_GROUPS, N_GROUPS_USED, None).vocab as u32;
    let skip = |id: u32| matches!(ml3_token_text(id), '"' | '\\') || id == 2;
    let ids: Vec<u32> = (0..vocab).filter(|&id| !skip(id)).collect();
    (0..len as u64)
        .map(|i| ids[(infr_testkit::gguf::splitmix64(i) % ids.len() as u64) as usize])
        .collect()
}

/// The byte length and FNV-1a digest of `model`'s GGUF, as the fixture's `# gguf:` line records.
fn ml3_gguf_identity(model: &SyntheticModel) -> String {
    let bytes = model.to_gguf_bytes();
    format!(
        "{} bytes, fnv1a64 {:#018x}",
        bytes.len(),
        infr_testkit::gguf::fnv1a64(&bytes)
    )
}

/// One llama.cpp run in the fixture: the prompt ids and the last position's logits.
struct Ml3Run {
    tokens: Vec<u32>,
    logits: Vec<f32>,
}

/// The fixture's `# gguf:` identity and its runs: `# tokens:` lists a run's prompt ids, and the
/// value lines after it are its logits.
fn read_ml3_fixture() -> (String, Vec<Ml3Run>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(ML3_ORACLE_FIXTURE);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut identity = None;
    let mut runs: Vec<Ml3Run> = Vec::new();
    for line in text.lines() {
        if let Some(id) = line.strip_prefix("# gguf: ") {
            identity = Some(id.to_string());
        } else if let Some(ids) = line.strip_prefix("# tokens:") {
            runs.push(Ml3Run {
                tokens: ids
                    .split_whitespace()
                    .map(|t| t.parse().expect("a token id"))
                    .collect(),
                logits: Vec::new(),
            });
        } else if !line.starts_with('#') && !line.trim().is_empty() {
            runs.last_mut()
                .unwrap_or_else(|| panic!("{}: a logit before any `# tokens:`", path.display()))
                .logits
                .push(line.trim().parse().expect("a logit"));
        }
    }
    let identity = identity.unwrap_or_else(|| panic!("{} names no `# gguf:`", path.display()));
    (identity, runs)
}

/// The fixture's runs, checked to be for [`ML3_PROMPT_LENS`]' prompts and the model's vocabulary.
fn ml3_oracle_runs() -> Vec<Ml3Run> {
    let (_, runs) = read_ml3_fixture();
    let lens: Vec<usize> = runs.iter().map(|r| r.tokens.len()).collect();
    assert_eq!(lens, ML3_PROMPT_LENS, "the fixture is for other lengths");
    let vocab = ds2_dims(N_GROUPS, N_GROUPS_USED, None).vocab;
    for Ml3Run { tokens, logits } in &runs {
        assert_eq!(
            *tokens,
            ml3_prompt(tokens.len()),
            "the fixture is for another prompt"
        );
        assert_eq!(logits.len(), vocab, "the fixture is for another vocabulary");
    }
    runs
}

/// The KV cache format the oracle runs at, on both engines, by the name llama.cpp's
/// `--cache-type-k`/`--cache-type-v` and infr's `kv.type_k`/`kv.type_v` share: f32, so that what
/// separates them is summation order and not where each rounds its cache to f16. infr's CPU MLA
/// arm reads it natively; llama.cpp also runs with flash attention off, whose CPU path rounds the
/// query to f16.
const ML3_ORACLE_KV: &str = "f32";

/// [`load`] with both KV sides pinned to the format named `kv`.
fn load_kv(tmp: &TempGguf, kv: &str) -> infr_llama::SeamModel {
    let mut cfg = env_cfg();
    let dt = infr_core::budget::parse_kv_dtype(kv).expect("a KV format name");
    cfg.kv.type_k = Some(dt);
    cfg.kv.type_k_specified = true;
    cfg.kv.type_v = Some(dt);
    cfg.kv.type_v_specified = true;
    load_with(tmp, cfg)
}

/// The bound on infr's CPU logits against llama.cpp's on the Mistral-Large-3-shaped file, both at
/// [`ML3_ORACLE_KV`]. The fixture header records the gaps measured when the data was made, and the
/// gap the temperature itself makes, which this bound separates by orders of magnitude.
const ML3_ORACLE_MAX_ABS: f32 = 1e-4;

/// **The external oracle for the attention temperature, in CI.** infr's CPU prefill of the
/// Mistral-Large-3-shaped file against llama.cpp's logits at every stored length: below the
/// temperature length, at it, and two floor steps past it. Past it the temperature moves the
/// logits by orders of magnitude more than the bound, so a temperature at the wrong position or
/// with the wrong offset fails here. The file's f32 expert banks have no batched MoE kernel, so
/// every token here is a one-row graph: a chunk of several rows across the length is checked
/// against this one-row path by
/// [`synthetic_mistral_large3_straddling_chunk_matches_one_row_at_a_time`].
#[test]
fn synthetic_mistral_large3_cpu_matches_llama_cpp() {
    let tmp = TempGguf::write("ml3-oracle-cpu", &mistral_large3_model());
    let model = load_kv(&tmp, ML3_ORACLE_KV);
    for Ml3Run {
        tokens,
        logits: want,
    } in ml3_oracle_runs()
    {
        let got = model.prefill_logits_cpu(&tokens).expect("cpu prefill");
        let err = max_abs_diff(&got, &want);
        println!(
            "synthetic mistral-large-3 cpu at {} tokens: max |logit diff| {err:e} against \
             llama.cpp",
            tokens.len()
        );
        assert!(
            err < ML3_ORACLE_MAX_ABS,
            "infr's CPU misses llama.cpp by {err} at {} tokens on the synthetic Mistral-Large-3",
            tokens.len()
        );
    }
}

/// The `Op::AttnTemp` ops a CPU prefill of `prompt` on `tmp`'s model ran, as `(rows, pos)`, with
/// its logits, at the prefill chunk height `ubatch` (`None`: the default).
fn ml3_attn_temps(
    tmp: &TempGguf,
    prompt: &[u32],
    ubatch: Option<usize>,
) -> (Vec<f32>, Vec<(u32, u32)>) {
    use infr_cpu::op_tap::{self, TappedOp};
    let mut cfg = env_cfg();
    if let Some(rows) = ubatch {
        cfg.device.ubatch = Some(rows);
        cfg.device.ubatch_specified = true;
    }
    let model = load_with(tmp, cfg);
    op_tap::start();
    let logits = model.prefill_logits_cpu(prompt);
    let temps = op_tap::take()
        .into_iter()
        .filter_map(|o| match o {
            TappedOp::AttnTemp { rows, pos } => Some((rows, pos)),
            _ => None,
        })
        .collect();
    (logits.expect("cpu prefill"), temps)
}

/// **A chunk that straddles the temperature length computes what one row at a time does, to the
/// bit.** The runner gives `Op::AttnTemp` a chunk's first position and the kernel steps it per row,
/// so a chunk whose rows sit on both sides of the length must give each row its own factor. The
/// llama.cpp oracle cannot see that mapping: its f32 expert banks have no batched MoE kernel, so
/// every token there is a one-row graph, and with quantized banks llama.cpp and infr round the
/// expert activations differently, too far apart to judge it. So the quantized model's default
/// prefill (one multi-row chunk across the length, checked through the op tap) is held to the same
/// model prefilled one row per graph (`device.ubatch` 1), the path the oracle does check. The CPU
/// computes every row the same way whatever the chunk, so the two agree exactly unless the rows
/// got the wrong positions.
#[test]
fn synthetic_mistral_large3_straddling_chunk_matches_one_row_at_a_time() {
    let tmp = TempGguf::write("ml3-straddle", &mistral_large3_quantized_model());
    let prompt = ml3_prompt(*ML3_PROMPT_LENS.last().expect("lengths"));
    let len = ML3_TEMP_LENGTH;
    let (chunked, temps) = ml3_attn_temps(&tmp, &prompt, None);
    println!("mistral-large-3 chunked AttnTemp (rows, pos): {temps:?}");
    assert!(
        temps.iter().all(|&(rows, pos)| pos + rows > len),
        "an AttnTemp ran wholly before the temperature length: {temps:?}"
    );
    assert!(
        temps.iter().any(|&(rows, pos)| rows > 1 && pos < len),
        "no multi-row AttnTemp straddled the temperature length: {temps:?}"
    );
    let (rows, temps) = ml3_attn_temps(&tmp, &prompt, Some(1));
    assert!(
        temps.iter().all(|&(rows, _)| rows == 1),
        "the one-row prefill ran a multi-row AttnTemp: {temps:?}"
    );
    let dv = max_abs_diff(&chunked, &rows);
    println!("mistral-large-3 straddling chunk vs one row at a time: max|Δ| = {dv:e}");
    assert_eq!(dv, 0.0, "a straddling chunk moved the logits");
}

/// The prefill chunk height [`gpu_synthetic_mistral_large3_straddling_chunk_matches_one_row_at_a_time`]
/// runs at:
/// one whose chunks do not align with [`ML3_TEMP_LENGTH`], so one of them straddles it, and short
/// enough to stay under the Vulkan batched-MoE row count at which this synthetic file's quantized
/// experts give NaN logits (backlog B-ATTN-TEMP).
const ML3_VULKAN_STRADDLE_UBATCH: usize = 5;

/// [`synthetic_mistral_large3_straddling_chunk_matches_one_row_at_a_time`] on Vulkan, where
/// `attn_temp.comp` steps the position per row: the quantized model prefilled in chunks of
/// [`ML3_VULKAN_STRADDLE_UBATCH`], one of which straddles the length (checked on the CPU through the
/// op tap; the tap is CPU-only), against the same model one row per graph, at
/// [`ML3_VULKAN_MAX_ABS`]: the two agree exactly on the discrete GPU and to reassociation on the
/// integrated one (measured). Against the CPU they are int8-activation rounding apart, too far to
/// judge the mapping.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_mistral_large3_straddling_chunk_matches_one_row_at_a_time() {
    let _lk = gpu_serial_lock();
    let tmp = TempGguf::write("ml3-straddle-vk", &mistral_large3_quantized_model());
    let prompt = ml3_prompt(*ML3_PROMPT_LENS.last().expect("lengths"));
    let (_, temps) = ml3_attn_temps(&tmp, &prompt, Some(ML3_VULKAN_STRADDLE_UBATCH));
    assert!(
        temps
            .iter()
            .any(|&(rows, pos)| rows > 1 && pos < ML3_TEMP_LENGTH && pos + rows > ML3_TEMP_LENGTH),
        "no chunk straddles the temperature length: {temps:?}"
    );
    let mut cfg = env_cfg();
    cfg.device.ubatch = Some(ML3_VULKAN_STRADDLE_UBATCH);
    cfg.device.ubatch_specified = true;
    let chunked = load_with(&tmp, cfg.clone())
        .prefill_logits_vulkan(&prompt)
        .expect("vulkan prefill");
    cfg.device.ubatch = Some(1);
    let rows = load_with(&tmp, cfg)
        .prefill_logits_vulkan(&prompt)
        .expect("vulkan prefill");
    let dv = max_abs_diff(&chunked, &rows);
    println!("mistral-large-3 vulkan straddling chunk vs one row at a time: max|Δ| = {dv:e}");
    assert!(
        dv < ML3_VULKAN_MAX_ABS,
        "a straddling chunk moved the Vulkan logits by {dv}"
    );
}

/// The bound on Vulkan's logits against the CPU's at the f16 KV cache Vulkan's MLA kernel reads,
/// and against its own at another chunk height: the same arithmetic and the same cache roundings,
/// so only reassociation separates them.
const ML3_VULKAN_MAX_ABS: f32 = 2e-4;

/// The bound on Vulkan's logits against llama.cpp's: Vulkan's MLA cache is f16 and the oracle's
/// f32, so this is the f16 cache's rounding, recorded in the fixture header by the CPU at f16.
const ML3_VULKAN_ORACLE_MAX_ABS: f32 = 1e-2;

/// [`synthetic_mistral_large3_cpu_matches_llama_cpp`] on Vulkan, where `Op::AttnTemp` runs
/// `attn_temp.comp` on the prefill chunk and on every decode step. Vulkan's MLA reads an f16 cache
/// only, so it is held tightly to the CPU at that cache, and to llama.cpp at the f16 cache's own
/// distance from it.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_mistral_large3_matches_llama_cpp() {
    let _lk = gpu_serial_lock();
    let tmp = TempGguf::write("ml3-oracle-vk", &mistral_large3_model());
    let model = load_kv(&tmp, "f16");
    for Ml3Run {
        tokens,
        logits: want,
    } in ml3_oracle_runs()
    {
        let len = tokens.len();
        let got = model
            .prefill_logits_vulkan(&tokens)
            .expect("vulkan prefill");
        let cpu = model.prefill_logits_cpu(&tokens).expect("cpu prefill");
        let (to_cpu, to_oracle) = (max_abs_diff(&got, &cpu), max_abs_diff(&got, &want));
        println!(
            "synthetic mistral-large-3 vulkan at {len} tokens: max |logit diff| {to_cpu:e} against \
             the CPU at f16 KV, {to_oracle:e} against llama.cpp"
        );
        assert!(
            to_cpu < ML3_VULKAN_MAX_ABS,
            "Vulkan misses the CPU by {to_cpu} at {len} tokens on the synthetic Mistral-Large-3"
        );
        assert!(
            to_oracle < ML3_VULKAN_ORACLE_MAX_ABS,
            "Vulkan misses llama.cpp by {to_oracle} at {len} tokens on the synthetic \
             Mistral-Large-3"
        );
    }
}

/// **Below its length the temperature changes nothing, to the bit; past it, it changes the
/// output.** Every factor is exactly 1 until a position reaches the temperature length, and the
/// runner emits no `Op::AttnTemp` for a batch wholly before it, so a prompt that ends there must
/// give the same logits as the file with the temperature off: the context the file used to be
/// capped at keeps the numbers it had. Past it the two must differ, or the temperature never ran.
fn temperature_is_inert_before_its_length(
    logits: impl Fn(&str, &SyntheticModel, &[u32]) -> Vec<f32>,
) {
    let below = ml3_prompt(ML3_TEMP_LENGTH as usize);
    let on = logits("ml3-on-below", &mistral_large3_model(), &below);
    let off = logits(
        "ml3-off-below",
        &mistral_large3_without_temperature(),
        &below,
    );
    let dv = max_abs_diff(&on, &off);
    println!("mistral-large-3 temperature on vs off below its length: max|Δ| = {dv:e}");
    assert_eq!(
        dv, 0.0,
        "the temperature moved the logits before its length"
    );

    let past = ml3_prompt(*ML3_PROMPT_LENS.last().expect("lengths"));
    assert_moves(
        "mistral-large-3 temperature on vs off past its length",
        &logits("ml3-on-past", &mistral_large3_model(), &past),
        &logits("ml3-off-past", &mistral_large3_without_temperature(), &past),
        "the attention temperature",
    );
}

#[test]
fn synthetic_mistral_large3_temperature_is_inert_before_its_length() {
    temperature_is_inert_before_its_length(cpu_logits_of);
}

#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_mistral_large3_temperature_is_inert_before_its_length() {
    let _lk = gpu_serial_lock();
    temperature_is_inert_before_its_length(|tag, model, prompt| {
        let tmp = TempGguf::write(tag, model);
        load(&tmp)
            .prefill_logits_vulkan(prompt)
            .expect("vulkan prefill")
    });
}

/// The command that regenerates [`ML3_ORACLE_FIXTURE`].
const ML3_REGENERATE: &str = "INFR_LLAMA_DEBUG=<llama-debug> INFR_LLAMA_FIXTURE_OUT=<dir> cargo \
                              test -p infr-llama --test synthetic_deepseek2 -- --ignored \
                              synthetic_mistral_large3_regenerate_llama_cpp_fixture";

/// **The checked-in llama.cpp data was made from the file [`mistral_large3_model`] writes today.**
/// A change to the builder or to the testkit's GGUF writer that moves a byte of the file fails
/// here, rather than leaving infr scored against data for another model.
#[test]
fn synthetic_mistral_large3_llama_cpp_fixture_matches_its_gguf() {
    let (identity, _) = read_ml3_fixture();
    assert_eq!(
        identity,
        ml3_gguf_identity(&mistral_large3_model()),
        "{ML3_ORACLE_FIXTURE} was made from another GGUF; regenerate it with `{ML3_REGENERATE}`"
    );
}

/// Recompute [`ML3_ORACLE_FIXTURE`]: write the synthetic GGUF, run `llama-debug --save-logits` on
/// each prompt's text (the binary `INFR_LLAMA_DEBUG` names), check llama.cpp tokenized it to
/// exactly those ids, and record its last-position logits with the build's version line, the
/// command, the file's identity, and what infr's CPU measured against them. The result is compared
/// with the checked-in file, or written into the directory `INFR_LLAMA_FIXTURE_OUT` names, so
/// running every ignored test with llama.cpp configured never rewrites the tree. Without
/// `INFR_LLAMA_DEBUG` it returns at once.
#[test]
#[ignore = "recomputes the llama.cpp fixture: needs INFR_LLAMA_DEBUG (a llama-debug binary)"]
fn synthetic_mistral_large3_regenerate_llama_cpp_fixture() {
    use infr_testkit::llama_cpp::{
        llama_cpp_version, llama_debug_logits, run_headless, write_or_compare_fixture,
    };
    let Ok(exe) = std::env::var("INFR_LLAMA_DEBUG") else {
        eprintln!("skip: INFR_LLAMA_DEBUG not set");
        return;
    };
    let model = mistral_large3_model();
    let tmp = TempGguf::write("ml3-oracle", &model);
    let version = llama_cpp_version(&exe);
    let ours = load_kv(&tmp, ML3_ORACLE_KV);
    let ours_f16 = load_kv(&tmp, "f16");
    let tmp_off = TempGguf::write("ml3-oracle-off", &mistral_large3_without_temperature());
    let ours_off = load_kv(&tmp_off, ML3_ORACLE_KV);
    let kv = ML3_ORACLE_KV;
    // Flash attention off: llama.cpp's CPU flash path rounds the query to f16, which put the two
    // engines 4e-3 apart where the f32 path leaves them 4e-6 apart (measured at generation). No
    // device: a build with a GPU backend would otherwise offload and compute other bits than the
    // CPU-only build the data was made with, and compare mode needs identical data lines.
    let args: [&std::ffi::OsStr; 8] = [
        "--cache-type-k".as_ref(),
        kv.as_ref(),
        "--cache-type-v".as_ref(),
        kv.as_ref(),
        "--flash-attn".as_ref(),
        "off".as_ref(),
        "--device".as_ref(),
        "none".as_ref(),
    ];
    // The build's own device list, recorded so a regeneration on another build can be told apart.
    let devices: String = run_headless(&exe, &["--list-devices".as_ref()])
        .lines()
        .skip_while(|l| !l.starts_with("Available devices"))
        .map(|l| format!("#   {}\n", l.trim()))
        .collect();
    let mut header = format!(
        "# llama.cpp logits for the synthetic Mistral-Large-3-shaped deepseek2 model\n\
         # (`mistral_large3_model` in synthetic_deepseek2.rs) at the last position of each prompt,\n\
         # one per vocabulary id.\n\
         # llama.cpp {version}, run on the CPU (`--device none`); `llama-debug --list-devices`:\n\
         {devices}\
         # Regenerate: {ML3_REGENERATE}\n\
         # which runs: llama-debug -m <synthetic gguf> -p <prompt as text> --save-logits \
         --logits-output-dir <dir> --cache-type-k {kv} --cache-type-v {kv} \
         --flash-attn off --device none\n\
         # gguf: {}\n",
        ml3_gguf_identity(&model)
    );
    let mut body = String::new();
    for len in ML3_PROMPT_LENS {
        let ids = ml3_prompt(len);
        let prompt: String = ids.iter().map(|&id| ml3_token_text(id)).collect();
        let (tokens, logits) = llama_debug_logits(&exe, tmp.path(), &prompt, &args);
        assert_eq!(tokens, ids, "llama.cpp tokenized the prompt differently");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite llama.cpp logits"
        );
        let cpu = ours.prefill_logits_cpu(&ids).expect("cpu prefill");
        let off = ours_off.prefill_logits_cpu(&ids).expect("cpu prefill");
        let f16 = ours_f16.prefill_logits_cpu(&ids).expect("cpu prefill");
        header.push_str(&format!(
            "# infr CPU at {len} tokens: max |logit diff| {:e}; with the temperature off {:e}; \
             at an f16 KV cache {:e}\n",
            max_abs_diff(&cpu, &logits),
            max_abs_diff(&off, &logits),
            max_abs_diff(&f16, &logits)
        ));
        let ids: Vec<String> = tokens.iter().map(u32::to_string).collect();
        body.push_str(&format!("# tokens: {}\n", ids.join(" ")));
        for v in &logits {
            body.push_str(&format!("{v}\n"));
        }
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(ML3_ORACLE_FIXTURE);
    let text = format!("{header}{body}");
    if let Some(written) = write_or_compare_fixture(&path, &text) {
        println!("wrote {}", written.display());
    }
    print!("{header}");
}
