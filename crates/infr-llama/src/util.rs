//! Crate-level glue: GGUF metadata helpers, chat-eos detection, streaming detok, the GPU probe,
//! per-layer-embedding loader, and test helpers. Split out of `lib.rs` (no logic change).
use crate::*;
use anyhow::{anyhow, Result};
use infr_core::WeightSource;
use infr_gguf::Gguf;
use infr_vulkan::VulkanBackend;
use tokenizers::Tokenizer;

/// Qwen2/Qwen3 pre-tokenizer regex (same string the HF `tokenizer.json` uses) — applied via a
/// Split before ByteLevel. Differs from the default GPT-2 ByteLevel regex (punctuation/number runs),
/// which is what made a naive ByteLevel produce different token ids.
pub(crate) const QWEN2_PRE_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Llama 4 pre-tokenizer regex (`tokenizer.ggml.pre == "llama4"` → llama.cpp's `GPT4O` pre-type,
/// the original split from the model's `tokenizer.json`). Applied via a Split before ByteLevel,
/// exactly like `QWEN2_PRE_RE`. Numbers group in runs of up to 3 digits (`\p{N}{1,3}`), and the
/// letter runs are split by case (upper-run then lower-run) — distinct from the Qwen regex.
pub(crate) const LLAMA4_PRE_RE: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Build the gemma4 E2B per-layer-embedding host-gather metadata from the GGUF. `None` for models
/// without per-layer embeddings. The big `per_layer_token_embd` table stays quantized in the mmap
/// and is gathered + dequanted per token at forward time (mirrors llama.cpp, which classifies
/// input embeddings as CPU-resident: "very little benefit to offloading the input layer"). The
/// `per_layer_model_proj` / `per_layer_proj_norm` weights are NOT loaded here — they're small
/// enough to native-upload to VRAM like any other weight (see `wload`/`wpush` in `seam.rs`)
/// and the projection + RMSNorm now run as GPU graph ops instead of a host GEMV. Shared by the GPU
/// and CPU loaders.
pub(crate) struct PerLayerEmbd {
    pub(crate) npl: usize,                       // per-layer embedding width (256)
    pub(crate) n_layer: usize,                   // number of layers (35)
    pub(crate) tok_embd_dtype: infr_core::DType, // per_layer_token_embd dtype (gathered per token from the gguf)
    pub(crate) tok_embd_row_bytes: usize,        // bytes per token row (npl*n_layer elements)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn build_per_layer_embd(g: &Gguf, cfg: &Config) -> Result<Option<PerLayerEmbd>> {
    if cfg.n_embd_per_layer == 0 {
        return Ok(None);
    }
    let te = g
        .tensors()
        .iter()
        .find(|t| t.name == "per_layer_token_embd.weight")
        .ok_or_else(|| anyhow!("per_layer_token_embd.weight not found"))?;
    // Bytes per token row = total bytes / vocab (te shape is GGUF [npl*n_layer, vocab]).
    let te_vocab = *te.shape.last().unwrap();
    Ok(Some(PerLayerEmbd {
        npl: cfg.n_embd_per_layer,
        n_layer: cfg.n_layer,
        tok_embd_dtype: te.dtype,
        tok_embd_row_bytes: te.nbytes / te_vocab,
    }))
}

/// UTF-8-safe incremental detokenizer for streaming: appends `id` to `acc`, decodes the whole
/// sequence so far, and emits the newly-completed suffix past `printed` — holding back a trailing
/// `�` (a multi-byte char split across tokens) until it completes. Mirrors the GPU path's streamer.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn stream_token(
    tokenizer: &Tokenizer,
    acc: &mut Vec<u32>,
    printed: &mut usize,
    id: u32,
    on_piece: &mut impl FnMut(&str),
) {
    acc.push(id);
    if let Ok(full) = tokenizer.decode(acc, true) {
        if !full.ends_with('\u{FFFD}') && full.len() > *printed && full.is_char_boundary(*printed) {
            on_piece(&full[*printed..]);
            *printed = full.len();
        }
    }
}

// Chat-template rendering (`render_chat_jinja`, `render_chat_user`) lives in the shared `infr-chat`
// crate — imported at the top of this module. There is NO fabricated-ChatML fallback: infr supports
// only models that ship a `tokenizer.chat_template`, so a missing/broken template is a hard error.

/// The error surfaced when a GGUF has no usable chat template (none embedded, or it failed to render).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn no_template_err() -> anyhow::Error {
    anyhow!(
        "model GGUF has no usable chat template (no `tokenizer.chat_template`, or it failed to \
         render — set INFR_DEBUG_CHAT=1 for details). infr requires an instruct model with an \
         embedded chat template."
    )
}

/// Whether a Vulkan device is available — a cheap probe (creates and drops a backend). Lets callers
/// (and tests) decide between the GPU and CPU paths, or skip GPU-only work when there's no device.
///
/// PANICS when `INFR_DEV` is set and that device cannot be opened. An EXPLICIT device request is a
/// demand, not a hint: the GPU seam tests gate on this probe and self-skip when it is false, so a
/// failing/absent `INFR_DEV` device used to make the whole GPU suite skip SILENTLY and still report
/// "ok" — `INFR_DEV=Vulkan9 cargo test` returned "1 passed" in 0.02 s. A vacuous green on the exact
/// runs we most need to trust (a survey pinning a specific GPU) is worse than a crash, so refuse.
/// An UNSET `INFR_DEV` on a GPU-less box keeps returning false and skipping, exactly as before.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn gpu_available() -> bool {
    match VulkanBackend::new() {
        Ok(_) => true,
        Err(e) => {
            if let Ok(dev) = std::env::var("INFR_DEV") {
                panic!(
                    "INFR_DEV={dev} was requested but that Vulkan device could not be opened: {e}\n\
                     Refusing to silently fall back / skip: an explicit device is a demand, not a \
                     hint. Unset INFR_DEV to run on the default device (or on the CPU)."
                );
            }
            false
        }
    }
}

/// Locate the Qwen3-0.6B Q4_K_M GGUF in the HF Hub cache (or `INFR_TEST_MODEL`) for the model-backed
/// unit tests; `None` → the test self-skips. We use the shared HF cache everywhere now (no bespoke
/// local model dir).
#[cfg(test)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn test_qwen3_06b() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
        return Some(std::path::PathBuf::from(p));
    }
    let hub = std::env::var("HOME").ok()? + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--unsloth--Qwen3-0.6B-GGUF/snapshots");
    std::fs::read_dir(&base).ok()?.find_map(|e| {
        let f = e.ok()?.path().join("Qwen3-0.6B-Q4_K_M.gguf");
        f.exists().then_some(f)
    })
}

/// Append chat-end markers in the vocab (`<|im_end|>` / `<|endoftext|>` / `<|eot_id|>`) to
/// `cfg.eos_ids` so generation stops on any of them, not just the GGUF `eos`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn add_chat_eos(cfg: &mut Config, tokenizer: &Tokenizer) {
    for name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"] {
        if let Some(id) = tokenizer.token_to_id(name) {
            if !cfg.eos_ids.contains(&id) {
                cfg.eos_ids.push(id);
            }
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn meta_u64(g: &Gguf, key: &str) -> Option<u64> {
    g.metadata().u64(key)
}

/// Float metadata lookup (diffusion-gemma's `eb_*` sampler params are stored as GGUF strings —
/// same KV store, no typed float accessor on `Metadata` — so parse through `as_f64`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn meta_f64(g: &Gguf, key: &str) -> Option<f64> {
    g.metadata().get(key).and_then(|v| v.as_f64())
}
