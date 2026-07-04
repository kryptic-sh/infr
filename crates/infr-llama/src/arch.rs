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
/// per-layer output scale; E2B adds KV-sharing + per-layer input embeddings.
pub const GEMMA4: &str = "gemma4";
/// Qwen3.5/3.6 gated-DeltaNet hybrid — routed to its OWN seam (`crate::qwen35`), never through
/// `Config::from_gguf`. NOT llama.cpp's `qwen3next` (a sibling arch with a different DeltaNet
/// V-broadcast pattern) and not `qwen35moe` — both are rejected on purpose.
pub const QWEN35: &str = "qwen35";

/// What `Config::from_gguf` (the dense/MoE transformer path) accepts. `QWEN35` is deliberately
/// absent — the runners route it to `crate::qwen35::SeamModel` before Config is ever built.
pub const DENSE: &[&str] = &[LLAMA, QWEN2, QWEN3, QWEN3_MOE, GEMMA3, GEMMA4];
