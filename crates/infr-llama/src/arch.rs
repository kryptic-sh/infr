//! The `general.architecture` strings infr supports — the single source of truth, replacing
//! magic strings scattered through dispatch code.
//!
//! These strings live INSIDE the GGUF (written by llama.cpp's `convert_hf_to_gguf.py`) and are
//! load-bearing twice over: the engine dispatches on the exact string, and the string namespaces
//! every model-metadata key (`{arch}.block_count`, `{arch}.embedding_length`, …). They must match
//! llama.cpp's canonical `LLM_ARCH_NAMES` table verbatim — never invent a name; audit new entries
//! against `llama.cpp/src/llama-arch.cpp` and a real converted GGUF.
//!
//! Beware the neighboring GGUF namespaces that reuse the same literals with DIFFERENT meanings:
//! `tokenizer.ggml.model == "llama"` means SentencePiece (a tokenizer family, not the arch), and
//! `tokenizer.ggml.pre == "qwen2"` names a pre-tokenizer regex. Those sites intentionally do NOT
//! use these constants.

/// Llama family (Llama 3.x, Mistral dense, TinyLlama, R1-Distill-Llama). NORM (interleaved) rope
/// from the converter's q/k permute; no qk-norm, no attention bias.
pub const LLAMA: &str = "llama";
/// Llama 4 (Scout 17B-16E etc.): the llama skeleton (NORM/interleaved rope, no attention bias)
/// plus a 16-expert top-1 SIGMOID-gated MoE FFN (weight-before-FFN, no top-k renorm) with a
/// Qwen2-MoE-style DENSE shared expert summed IN (no per-token gate, unlike qwen35moe), and iRoPE:
/// every `no_rope_layer_step`-th layer is NoPE (rope skipped, global attention) while rope layers
/// carry chunked local attention; rope layers also apply a WEIGHTLESS per-head L2-norm to Q/K
/// AFTER rope (`Llama4TextL2Norm`). Chunked-attention masking and the NoPE attention temperature
/// are not implemented; the context is capped where the first of them would change the output —
/// see `Config::faithful_ctx_limit`.
pub const LLAMA4: &str = "llama4";
/// Qwen2/2.5 (incl. Qwen2.5-Coder, R1-Distill-Qwen). Llama path + q/k/v projection biases; ships
/// HF rotate-half q/k order, so the loader permutes rows (`Config::permute_qk_neox`).
pub const QWEN2: &str = "qwen2";
/// Qwen3 dense. QK-norm, NEOX rope (fused `QkNormRope`), no biases.
pub const QWEN3: &str = "qwen3";
/// Qwen3 MoE (30B-A3B etc.): qwen3 + router/expert FFN banks.
pub const QWEN3_MOE: &str = "qwen3moe";
/// Gemma 3: SWA + dual-rope, hd=256, GeGLU, sandwich norms, SPM tokenizer.
pub const GEMMA3: &str = "gemma3";
/// Gemma 4 (incl. E2B): per-layer heterogeneous dims, V-norm, freq_factors, softcap,
/// per-layer output scale; E2B adds KV-sharing + per-layer input embeddings. The 26B-A4B variant
/// carries routed-expert tensors on the same string — `Config::gemma4_moe` detects them and swaps
/// in diffusion-gemma's dual FFN (`FfnW::DiffusionMoe`) via `Config::dual_moe`, autoregressive.
pub const GEMMA4: &str = "gemma4";
/// Qwen3.5/3.6 gated-DeltaNet hybrid (DENSE FFN) — runs through `Config::from_gguf`'s `qwen35`
/// fields + `MixerW::DeltaNet` (see `crate::qwen35::Cfg` for the raw metadata parse). NOT
/// llama.cpp's `qwen3next` (a sibling arch with a different DeltaNet V-broadcast pattern) — that
/// one is still rejected on purpose. `QWEN35_MOE` below is the routed-expert sibling.
pub const QWEN35: &str = "qwen35";
/// Qwen3.6 MoE (35B-A3B etc. — same gated-DeltaNet hybrid as `QWEN35`, but a routed-expert MoE
/// FFN on EVERY layer, delta or full-attention, plus a Qwen2-MoE-style shared expert gated by a
/// per-token sigmoid). Shares every `qwen35` field (`Config::from_gguf`'s `qwen35` gate,
/// `MixerW::DeltaNet`); only the FFN differs (`FfnW::Moe`'s `shexp` branch — see `seam::weights`).
pub const QWEN35_MOE: &str = "qwen35moe";
/// DiffusionGemma: block text-diffusion MoE on a Gemma-4 backbone (shares gemma4's heterogeneous
/// per-layer dims, V-norm, freq_factors, softcap, sandwich norms), plus a per-layer DUAL FFN
/// (dense GeGLU ∥ 128-expert MoE, summed) and encoder/decoder per-layer output scalars. See
/// `docs/diffusion-gemma.md`.
pub const DIFFUSION_GEMMA: &str = "diffusion-gemma";
/// BitNet b1.58 (1bitLLM): the llama skeleton (NEOX rope like qwen2 — GGUF q/k stay in HF order,
/// so the loader permutes rows via `Config::permute_qk_neox`; no qk-norm, no attention bias, tied
/// lm_head, gated SwiGLU FFN) plus SubLN's two extra RMSNorms (`Config::sub_norm`): `attn_sub_norm`
/// on the concatenated-heads attention output BEFORE the o-projection, and `ffn_sub_norm` on the
/// FFN intermediate BEFORE `ffn_down`. Confirmed against llama.cpp's `build_bitnet`
/// (`src/models/bitnet.cpp`): the FFN activation is SiLU (`LLM_FFN_SILU`/`LLM_FFN_PAR`), NOT
/// squared-ReLU. Ships TQ2_0 ternary weights (native on CPU + Vulkan) — this is an arch-only add.
pub const BITNET: &str = "bitnet";
/// DeepSeek V1 (DeepSeek-LLM 7B/67B, DeepSeek-MoE-16B): the llama skeleton (NORM/interleaved rope,
/// no qk-norm, no attention bias) plus a routed-expert softmax-gated MoE FFN (no top-k
/// renormalization, no group-limited routing) with an ungated shared expert summed in plain.
/// First `n_layer_dense_lead` layers are dense FFN; the rest are MoE. See docs/deepseek.md § Stage 1.
pub const DEEPSEEK: &str = "deepseek";
/// DeepSeek V2/V2-Lite/V3/V3.1: adds MLA attention (absorbed form, K 576 wide, V 512 wide aliased
/// from K, n_kv=1), YaRN rope scaling, group-limited MoE routing, router bias correction
/// (`exp_probs_b`), and multi-function gating (softmax/sigmoid/sqrt-softplus/softmax-on-weights).
/// The `deepseek2` GGUF arch string covers ALL of V2, V2-Lite, V3 (non-0324), and V3.1 —
/// llama.cpp keeps them in one model class (`src/models/deepseek2.cpp`). See docs/deepseek.md
/// § Stage 2.
pub const DEEPSEEK2: &str = "deepseek2";
/// DeepSeek V3.2: [`DEEPSEEK2`]'s absorbed MLA + group-limited MoE verbatim, plus the per-layer
/// "lightning indexer" that scores every (query, key) pair and keeps the top
/// `attention.indexer.top_k` keys for the real attention. Non-MLA is rejected outright and
/// `attention.q_lora_rank` is mandatory (there is no lite variant), so `Config::deepseek2` — the
/// MLA/KV-geometry/MoE-shape gate — is TRUE for this string too and `Config::deepseek32` gates only
/// what is new. llama.cpp keeps it in its own model class (`src/models/deepseek32.cpp`). The
/// indexer's five per-layer tensors load and the seam emits the indexer, so this arch generates on
/// the CPU and Vulkan backends. See docs/deepseek.md § Stage 3.
pub const DEEPSEEK32: &str = "deepseek32";
/// DeepSeek V4 (V4-Flash / V4-Pro): a genuinely different architecture, NOT an increment on
/// [`DEEPSEEK2`]. There is no MLA here — no `kv_lora_rank`, no `wk_b`/`wv_b` — so `Config::deepseek2`
/// is FALSE for this string and `Config::deepseek4` gates everything. What it keeps from the family
/// is the Q-LoRA projection (`wq_a`/`q_a_norm`/`wq_b`), the routed-expert MoE block with its shared
/// expert, the norms and the generic rope/embedding plumbing. What it adds: single-head MQA KV, a
/// low-rank grouped output projection, attention sinks, hyper-connections (`hc_mult` parallel
/// residual streams with Sinkhorn-normalised mixing) replacing every `x = x + f(x)`, three-tier
/// per-layer attention keyed on `attention.compress_ratios[il] ∈ {0, 4, 128}`, compressor blocks,
/// hash-routed MoE on the first `hash_layer_count` layers, and mandatory `sqrt(softplus)` gating.
/// Every layer is sliding-window (`set_swa_pattern(0)`), and there are no dense-lead layers and no
/// NextN. llama.cpp keeps it in its own model class (`src/models/deepseek4.cpp`).
///
/// The load path is complete — config, every tensor. A file whose `attention.compress_ratios` are
/// all 0 (every layer the pure sliding-window tier) generates; a layer with ratio 4 or 128 is
/// refused by name in `generate_dense_backend`, since the compressed tiers are not implemented.
/// See docs/deepseek.md § Stage 4.
pub const DEEPSEEK4: &str = "deepseek4";
/// Microsoft's official BitNet-b1.58 GGUFs (`microsoft/bitnet-b1.58-2B-4T-gguf`) declare
/// `general.architecture = "bitnet-b1.58"` and prefix EVERY metadata key with it
/// (`bitnet-b1.58.block_count`, …). Behaviorally identical to [`BITNET`] (same llama+SubLN
/// skeleton); the only reason it's a distinct string is that the metadata prefix must match the
/// file, so `Config::from_gguf` keeps `arch` verbatim for key lookups and treats this as bitnet via
/// [`is_bitnet`]. These files ship i2_s ternary weights (host-dequant → f16 at load).
pub const BITNET_B158: &str = "bitnet-b1.58";

/// True for either BitNet arch string ([`BITNET`] or [`BITNET_B158`]) — the two are behaviorally
/// identical (llama skeleton + SubLN, NEOX rope); they differ only in the metadata key prefix.
pub fn is_bitnet(arch: &str) -> bool {
    arch == BITNET || arch == BITNET_B158
}

/// What `Config::from_gguf` — the shared TRANSFORMER-skeleton path — accepts, MINUS `QWEN35`
/// (kept as its own match arm since `is_qwen35` gates a handful of qwen35-only fields — see
/// `Config::from_gguf`). The split is by code path, not FFN sparsity: this list includes the
/// routed-MoE `qwen3moe` (same attention/KV machinery, an expert FFN swapped in).
pub const TRANSFORMER: &[&str] = &[
    LLAMA,
    LLAMA4,
    QWEN2,
    QWEN3,
    QWEN3_MOE,
    GEMMA3,
    GEMMA4,
    DIFFUSION_GEMMA,
    BITNET,
    BITNET_B158,
    DEEPSEEK,
];

/// Read `general.architecture` from a GGUF WITHOUT a full model load (mirrors
/// [`crate::diffusion::is_diffusion_gemma`]'s cheap peek) — lets `infr run`/`serve` pick
/// architecture-aware sampling defaults before paying a `SeamModel::load`. `None` if the file
/// can't be opened or carries no architecture key.
pub fn arch_of(path: &std::path::Path) -> Option<String> {
    use infr_core::WeightSource;
    infr_gguf::Gguf::open(path).ok().and_then(|g| {
        g.metadata()
            .str("general.architecture")
            .map(|s| s.to_string())
    })
}

/// The `general.sampling.*` keys [`declared_sampling`] applies. llama.cpp also applies the rest of
/// the family (`min_p`, the penalties, `xtc_*`, `mirostat*`, `sequence`); see
/// [`unapplied_sampling_keys`].
pub const APPLIED_SAMPLING_KEYS: [&str; 3] = [
    "general.sampling.temp",
    "general.sampling.top_k",
    "general.sampling.top_p",
];

/// Every `general.sampling.*` key the GGUF at `path` carries that infr does not apply, sorted;
/// empty when there are none or the file cannot be opened. A file that pairs a hot temperature
/// with `min_p` or a repeat penalty gets the temperature without its counterweight, so the caller
/// warns with this list.
pub fn unapplied_sampling_keys(path: &std::path::Path) -> Vec<String> {
    use infr_core::WeightSource;
    let Ok(g) = infr_gguf::Gguf::open(path) else {
        return Vec::new();
    };
    let mut keys: Vec<String> = g
        .metadata()
        .kv
        .keys()
        .filter(|k| {
            k.starts_with("general.sampling.") && !APPLIED_SAMPLING_KEYS.contains(&k.as_str())
        })
        .cloned()
        .collect();
    keys.sort();
    keys
}

/// The sampler the GGUF itself declares: `general.sampling.temp` / `.top_k` / `.top_p`, which
/// llama.cpp's converter seeds from the source checkout's `generation_config.json` (and which
/// llama.cpp applies under any value the user set). Each field is independent; an absent or
/// mistyped one is `None`. A negative `top_k` is llama.cpp's "keep all" and maps to infr's `0`.
/// Cheap like [`arch_of`]: header only, no model load.
pub fn declared_sampling(path: &std::path::Path) -> (Option<f32>, Option<usize>, Option<f32>) {
    use infr_core::loader::MetaValue;
    use infr_core::WeightSource;
    let Ok(g) = infr_gguf::Gguf::open(path) else {
        return (None, None, None);
    };
    let meta = g.metadata();
    let [temp, top_k, top_p] = APPLIED_SAMPLING_KEYS;
    let float = |key| meta.get(key).and_then(MetaValue::as_f64).map(|x| x as f32);
    let top_k = match meta.get(top_k) {
        Some(MetaValue::I64(k)) if *k < 0 => Some(0),
        v => v
            .and_then(MetaValue::as_u64)
            .and_then(|k| usize::try_from(k).ok()),
    };
    (float(temp), top_k, float(top_p))
}

/// Every architecture infr runs, across both paths.
pub const ALL: &[&str] = &[
    LLAMA,
    LLAMA4,
    QWEN2,
    QWEN3,
    QWEN3_MOE,
    GEMMA3,
    GEMMA4,
    DIFFUSION_GEMMA,
    BITNET,
    BITNET_B158,
    DEEPSEEK,
    DEEPSEEK2,
    DEEPSEEK32,
    DEEPSEEK4,
    QWEN35,
    QWEN35_MOE,
];
