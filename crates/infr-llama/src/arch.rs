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
/// AFTER rope (`Llama4TextL2Norm`). Chunked-attention masking + NoPE attention-temperature scaling
/// (both no-ops below the 8192-token chunk size) are CPU follow-ups — see `Config::from_gguf`.
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
/// `docs/DIFFUSIONGEMMA.md`.
pub const DIFFUSION_GEMMA: &str = "diffusion-gemma";

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
];

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
    QWEN35,
    QWEN35_MOE,
];
