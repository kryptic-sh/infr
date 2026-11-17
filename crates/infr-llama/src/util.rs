//! Crate-level glue: GGUF metadata helpers, chat-eos detection, streaming detok, the GPU probe,
//! per-layer-embedding loader, and test helpers. Split out of `lib.rs` (no logic change).
use crate::*;
use anyhow::{anyhow, Result};
use infr_core::WeightSource;
use infr_gguf::Gguf;
use tokenizers::Tokenizer;

/// Qwen2/Qwen3 pre-tokenizer regex (same string the HF `tokenizer.json` uses) — applied via a
/// Split before ByteLevel. Differs from the default GPT-2 ByteLevel regex (punctuation/number runs),
/// which is what made a naive ByteLevel produce different token ids.
pub(crate) const QWEN2_PRE_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Llama-3 pre-tokenizer regex (llama.cpp's `LLAMA3` pre-type: `llama-bpe` and its aliases), the
/// original split from the model's `tokenizer.json`. Same shape as `QWEN2_PRE_RE` except that
/// numbers group in runs of up to three digits rather than one.
pub(crate) const LLAMA3_PRE_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Qwen3.5 pre-tokenizer regex (llama.cpp's `QWEN35` pre-type), the original split from the model's
/// `tokenizer.json`. `QWEN2_PRE_RE` with combining marks (`\p{M}`) kept inside letter runs.
pub(crate) const QWEN35_PRE_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// DeepSeek-LLM (V1) pre-tokenizer patterns — 6 successive splits applied in order
/// (llama.cpp's `DEEPSEEK_LLM` pre-type, `src/llama-vocab.cpp`). See docs/deepseek.md §0.2.
pub(crate) const DEEPSEEK_LLM_PRE_RES: [&str; 6] = [
    r"[\r\n]",
    r"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
    r"\s?[!-/:-~！-／：-～‘-‟　-。]+",
    r"\s+$",
    r"[一-龥ࠀ-一가-퟿]+",
    r"\p{N}+",
];

/// DeepSeek-Coder pre-tokenizer patterns — 5 successive splits
/// (llama.cpp's `DEEPSEEK_CODER` pre-type).
pub(crate) const DEEPSEEK_CODER_PRE_RES: [&str; 5] = [
    r"[\r\n]",
    r"\s?\p{L}+",
    r"\s?\p{P}+",
    r"[一-龥ࠀ-一가-퟿]+",
    r"\p{N}",
];

/// DeepSeek-V3 pre-tokenizer patterns — 3 successive splits
/// (llama.cpp's `DEEPSEEK3_LLM` pre-type, also used by V4).
pub(crate) const DEEPSEEK_V3_PRE_RES: [&str; 3] = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
];

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

/// Where a streaming decode stands in its accumulator: the HF `TextStreamer` / vLLM
/// `prefix_offset` / `read_offset` pair. `acc[read..]` has not been emitted yet; `acc[prefix..read]`
/// is the last emitted stretch, decoded again as left context so that a decoder whose output
/// depends on the tokens before it (a Metaspace leading space, a byte-fallback sequence) produces
/// the same text for the new tokens that decoding the whole sequence would.
#[derive(Debug, Default)]
pub(crate) struct StreamOffsets {
    prefix: usize,
    read: usize,
}

/// The one decoder call streaming makes, behind a trait so a test can count the ids it is handed —
/// the work this module exists to keep linear.
pub(crate) trait StreamDecode {
    /// `ids` decoded with special tokens skipped, or `None` when the decoder fails.
    fn decode_ids(&self, ids: &[u32]) -> Option<String>;
}

impl StreamDecode for Tokenizer {
    fn decode_ids(&self, ids: &[u32]) -> Option<String> {
        self.decode(ids, true).ok()
    }
}

/// UTF-8-safe incremental detokenizer for streaming: appends `id` to `acc` and emits the text it
/// completes — holding back a trailing `�` (a multi-byte char split across tokens) until it
/// completes. Mirrors the GPU path's streamer.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn stream_token(
    tokenizer: &impl StreamDecode,
    acc: &mut Vec<u32>,
    at: &mut StreamOffsets,
    id: u32,
    on_piece: &mut impl FnMut(&str),
) {
    acc.push(id);
    emit_decoded_suffix(tokenizer, acc, at, on_piece);
}

/// Like [`stream_token`], but commits a whole SPAN of ids at once. Callers that already hold every
/// id of a chunk (diffusion's per-block `on_block` — the whole block's tokens are decided together)
/// use this so the new ids are decoded ONCE per span instead of once per id. Same UTF-8 holdback
/// semantics and same emitted bytes as feeding the ids through [`stream_token`] one at a time —
/// only the number of `decode` calls (and `on_piece` chunk boundaries) differs.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn stream_tokens(
    tokenizer: &impl StreamDecode,
    acc: &mut Vec<u32>,
    at: &mut StreamOffsets,
    ids: &[u32],
    on_piece: &mut impl FnMut(&str),
) {
    if ids.is_empty() {
        return;
    }
    acc.extend_from_slice(ids);
    emit_decoded_suffix(tokenizer, acc, at, on_piece);
}

/// Shared tail of [`stream_token`]/[`stream_tokens`]: decode the unemitted tokens together with the
/// last emitted stretch, emit what they add to it, and hold back a trailing `�` (a multi-byte char
/// split across the token boundary) until it completes.
///
/// Only `acc[at.prefix..]` is decoded, never the whole accumulator. Re-decoding everything per
/// token made a reply's detokenization quadratic in its length, paid while the GPU waits.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn emit_decoded_suffix(
    tokenizer: &impl StreamDecode,
    acc: &[u32],
    at: &mut StreamOffsets,
    on_piece: &mut impl FnMut(&str),
) {
    let Some(text) = tokenizer.decode_ids(&acc[at.prefix..]) else {
        return;
    };
    if text.ends_with('\u{FFFD}') {
        return;
    }
    let seen = if at.read > at.prefix {
        match tokenizer.decode_ids(&acc[at.prefix..at.read]) {
            Some(s) => s.len(),
            None => return,
        }
    } else {
        0
    };
    if text.len() > seen && text.is_char_boundary(seen) {
        on_piece(&text[seen..]);
        at.prefix = at.read;
        at.read = acc.len();
    }
}

/// Run `f` with per-op GPU profiling suppressed for its duration, restoring the prior state
/// afterward. Every backend's `warmup()` / startup path (`chat::vulkan`, `chat::diffusion`,
/// `parallel::init_slots`, `bench_vulkan`) needs this: the per-op profiling recorders read
/// `INFR_PROF_OPS` at CONSTRUCTION, and a throwaway warmup submit would otherwise pollute a later
/// bench's per-op aggregate. Extracted so the call sites can't drift (the Metal path deliberately
/// does NOT use it — the Metal backend reads its profile env at construction inside the warmup
/// generate itself; see `chat::metal`).
///
/// Toggles the process-global [`infr_prof_rt::set_profiling_suppressed`] flag (which the Vulkan
/// recorder ANDs into its `INFR_PROF_OPS` check) rather than mutating the `INFR_PROF_OPS` env var —
/// mutating the env table around a rayon-parallel forward is a data race, and `set_var` is
/// `unsafe` under edition 2024.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn with_profiling_suppressed<T>(f: impl FnOnce() -> T) -> T {
    let prev = infr_prof_rt::set_profiling_suppressed(true);
    let r = f();
    infr_prof_rt::set_profiling_suppressed(prev);
    r
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

/// Locate the Qwen3-0.6B Q4_K_M GGUF in the HF Hub cache (or `INFR_TEST_MODEL`) for the model-backed
/// unit tests; `None` → the test self-skips. We use the shared HF cache everywhere now (no bespoke
/// local model dir).
#[cfg(test)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn test_qwen3_06b() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
        return Some(std::path::PathBuf::from(p));
    }
    test_hub_gguf("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf")
}

/// `file` from `repo` in the HF Hub cache, resolved exactly as `infr` itself finds the cache
/// (`HF_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME`, the home directory on every platform), or `None`
/// when it is not downloaded.
#[cfg(test)]
pub(crate) fn test_hub_gguf(repo: &str, file: &str) -> Option<std::path::PathBuf> {
    infr_hub::Store::discover().ok()?.snapshot_file(repo, file)
}

/// The token texts llama.cpp treats as end-of-generation wherever a vocabulary has them
/// (`llama_vocab::impl::load`, the `special_eog_ids` loop), verbatim. A declared EOS that is not
/// the turn terminator is common, and without the terminator in the stop set the model runs on
/// into an invented next turn.
const EOG_TOKEN_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<eos>",
    "<turn|>",
    "<|tool_response>",
    "<｜end▁of▁sentence｜>",
    "[e~[",
];

/// The stop names infr used before it followed [`EOG_TOKEN_TEXTS`]: what a GGUF without
/// `tokenizer.ggml.token_type` still gets, since nothing there says which matches are control
/// tokens (see [`eog_ids`]).
const UNTYPED_EOG_TOKEN_TEXTS: &[&str] = &["<|im_end|>", "<|endoftext|>", "<|eot_id|>"];

/// Append the end-of-generation ids to `cfg.eos_ids` so generation stops on any of them, not just
/// the GGUF `eos`: the GGUF's own `eot` / `eom` ids, and the [`EOG_TOKEN_TEXTS`] entries the
/// vocabulary has as control tokens.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn add_chat_eos(cfg: &mut Config, tokenizer: &Tokenizer, md: &infr_core::Metadata) {
    for id in eog_ids(tokenizer, md, cfg.vocab) {
        if !cfg.eos_ids.contains(&id) {
            cfg.eos_ids.push(id);
        }
    }
}

/// The ids [`add_chat_eos`] adds, before de-duplication against the GGUF `eos`. `vocab` is the
/// model's vocabulary (its logit width): an `eot` / `eom` id at or past it can never be sampled, and
/// is ignored as llama.cpp ignores it.
///
/// Deliberately narrower than llama.cpp, which matches [`EOG_TOKEN_TEXTS`] by text alone and then
/// forces the matched token to CONTROL. Here a text match counts only when the GGUF already types
/// that token CONTROL or USER_DEFINED (`tokenizer.ggml.token_type` 3 / 4). A NORMAL token is
/// ordinary text the model prints — Qwen's `</s>` is one — and stopping on it would cut a reply off
/// at a literal `</s>` (HTML, or a reply about tokens). Without a `token_type` array no match can be
/// told apart from text, so only [`UNTYPED_EOG_TOKEN_TEXTS`] is used: exactly the stop set such a
/// file had before, so it neither gains nor loses a stop. Ids the GGUF declares in its `eot` /
/// `eom` keys count whatever their type.
///
/// llama.cpp's two exceptions to its own text list are kept: `<|end|>` ends a MESSAGE, not the
/// generation, in a vocabulary that also stops on `<|return|>`/`<|flush|>` and `<|call|>`/
/// `<|calls|>` (o200k_harmony, solar-open), and `</s>` is not a stop in one that stops on
/// `<|tool_response>` (gemma4). Unlike llama.cpp they only ever drop a text-list match, never an id
/// the GGUF declares in its `eos` / `eot` / `eom` keys.
fn eog_ids(tokenizer: &Tokenizer, md: &infr_core::Metadata, vocab: usize) -> Vec<u32> {
    let types = md
        .get("tokenizer.ggml.token_type")
        .and_then(infr_core::MetaValue::as_arr);
    let names = if types.is_some() {
        EOG_TOKEN_TEXTS
    } else {
        UNTYPED_EOG_TOKEN_TEXTS
    };
    let is_control = |id: u32| {
        types.is_none_or(|t| {
            t.get(id as usize)
                .and_then(infr_core::MetaValue::as_u64)
                .is_some_and(|ty| ty == 3 || ty == 4)
        })
    };
    let matched: Vec<(&str, u32)> = names
        .iter()
        .filter_map(|&name| Some((name, tokenizer.token_to_id(name)?)))
        .filter(|&(_, id)| is_control(id))
        .collect();
    let has = |name: &str| matched.iter().any(|&(n, _)| n == name);
    let harmony = (has("<|call|>") || has("<|calls|>")) && (has("<|return|>") || has("<|flush|>"));
    let gemma4 = has("<|tool_response>");
    let mut ids: Vec<u32> = matched
        .iter()
        .filter(|&&(name, _)| !(harmony && name == "<|end|>") && !(gemma4 && name == "</s>"))
        .map(|&(_, id)| id)
        .collect();
    for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
        if let Some(id) = md.u64(key).filter(|&id| (id as usize) < vocab) {
            ids.push(id as u32);
        }
    }
    ids
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

#[cfg(test)]
mod eog_tests {
    use super::*;
    use infr_core::{MetaValue, Metadata};
    use tokenizers::models::bpe::BPE;

    /// A tokenizer whose vocabulary is exactly `names`, id = position.
    fn vocab(names: &[&str]) -> Tokenizer {
        let v = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.to_string(), i as u32))
            .collect();
        Tokenizer::new(BPE::builder().vocab_and_merges(v, vec![]).build().unwrap())
    }

    fn sorted(mut v: Vec<u32>) -> Vec<u32> {
        v.sort_unstable();
        v
    }

    /// Metadata typing every token of a `n`-token vocabulary NORMAL (1) except `control`.
    fn typed(n: usize, control: &[usize]) -> Metadata {
        let mut md = Metadata::default();
        let types = (0..n)
            .map(|i| MetaValue::U64(if control.contains(&i) { 3 } else { 1 }))
            .collect();
        md.kv
            .insert("tokenizer.ggml.token_type".into(), MetaValue::Arr(types));
        md
    }

    /// The stop set follows llama.cpp's end-of-generation rules (`llama_vocab::impl::load`):
    /// its whole text list rather than three names, the GGUF's own `eot` / `eom` ids, and its two
    /// exceptions. A Gemma `<end_of_turn>` or a Phi-3 `<|end|>` under a different declared EOS
    /// used to be no stop at all.
    #[test]
    fn eog_ids_follow_the_reference_rules() {
        // Gemma-3 / Phi-3 / Llama-3.1 terminators — none of them in the old three-name list.
        let tok = vocab(&["a", "<end_of_turn>", "<|end|>", "<|eom_id|>", "b"]);
        assert_eq!(sorted(eog_ids(&tok, &typed(5, &[1, 2, 3]), 5)), [1, 2, 3]);

        // `eot` / `eom` keys name ids whatever their type; one past the MODEL's vocabulary is
        // ignored even when the tokenizer has a token there.
        let tok = vocab(&["a", "b", "<my_eot>", "<my_eom>", "c"]);
        let mut md = typed(5, &[]);
        md.kv
            .insert("tokenizer.ggml.eot_token_id".into(), MetaValue::U64(2));
        md.kv
            .insert("tokenizer.ggml.eom_token_id".into(), MetaValue::U64(3));
        assert_eq!(sorted(eog_ids(&tok, &md, 5)), [2, 3]);
        md.kv
            .insert("tokenizer.ggml.eom_token_id".into(), MetaValue::U64(4));
        assert_eq!(eog_ids(&tok, &md, 4), [2]);

        // o200k_harmony: `<|end|>` closes a message, `<|return|>` / `<|call|>` end generation.
        let tok = vocab(&["<|end|>", "<|return|>", "<|call|>"]);
        assert_eq!(sorted(eog_ids(&tok, &typed(3, &[0, 1, 2]), 3)), [1, 2]);
        // gemma4: `<|tool_response>` stops, `</s>` does not.
        let tok = vocab(&["</s>", "<|tool_response>", "<turn|>"]);
        let md = typed(3, &[0, 1, 2]);
        assert_eq!(sorted(eog_ids(&tok, &md, 3)), [1, 2]);
        // …but an exception never drops an id the GGUF declares.
        let mut md = md;
        md.kv
            .insert("tokenizer.ggml.eot_token_id".into(), MetaValue::U64(0));
        assert_eq!(sorted(eog_ids(&tok, &md, 3)), [0, 1, 2]);
    }

    /// A text-list match stops generation only when the GGUF types it as a control token. Qwen's
    /// `</s>` is a NORMAL token the model prints as text; llama.cpp stops on it anyway (it forces
    /// every text match to CONTROL), which would end a reply at a literal `</s>`.
    #[test]
    fn a_normal_token_on_the_list_is_not_a_stop() {
        let tok = vocab(&["a", "</s>", "<|im_end|>"]);
        assert_eq!(eog_ids(&tok, &typed(3, &[2]), 3), [2]);
        // No `token_type` at all: the stop names a file like that always had, nothing more.
        assert_eq!(eog_ids(&tok, &Metadata::default(), 3), [2]);
        let tok = vocab(&["a", "<end_of_turn>"]);
        assert!(eog_ids(&tok, &Metadata::default(), 2).is_empty());
    }
}

#[cfg(test)]
mod stream_tests {
    use super::*;
    use std::cell::Cell;
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::decoders::fuse::Fuse;
    use tokenizers::decoders::sequence::Sequence as DecoderSequence;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::DecoderWrapper;

    /// A byte-level BPE with one token per byte and no merges, so every multi-byte character is
    /// split across tokens.
    fn byte_level() -> Tokenizer {
        let mut chars: Vec<char> = ByteLevel::alphabet().into_iter().collect();
        chars.sort_unstable();
        let vocab = chars
            .iter()
            .enumerate()
            .map(|(i, c)| (c.to_string(), i as u32))
            .collect();
        let mut tok = Tokenizer::new(
            BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .build()
                .unwrap(),
        );
        tok.with_pre_tokenizer(Some(ByteLevel::default()));
        tok.with_decoder(Some(ByteLevelDecoder::default()));
        tok
    }

    /// A SentencePiece vocabulary as a GGUF describes it: `▁`-prefixed pieces, `<0xXX>` byte
    /// pieces (so a multi-byte character can span several tokens) and a CONTROL `<s>` that decoding
    /// skips.
    fn spm_metadata(add_space_prefix: bool) -> infr_core::Metadata {
        use infr_core::MetaValue;
        let mut pieces: Vec<String> = ["<unk>", "<s>", "▁", "▁Hello", "▁world", "!", "a"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        pieces.extend((0..=255u8).map(|b| format!("<0x{b:02X}>")));
        let types = pieces
            .iter()
            .map(|p| {
                MetaValue::U64(match p.as_str() {
                    "<s>" => 3,
                    p if p.starts_with("<0x") => 6,
                    _ => 1,
                })
            })
            .collect();
        let mut md = infr_core::Metadata::default();
        for (k, v) in [
            ("tokenizer.ggml.model", MetaValue::Str("llama".into())),
            ("tokenizer.ggml.token_type", MetaValue::Arr(types)),
            (
                "tokenizer.ggml.scores",
                MetaValue::Arr(pieces.iter().map(|_| MetaValue::F64(0.0)).collect()),
            ),
            (
                "tokenizer.ggml.tokens",
                MetaValue::Arr(pieces.into_iter().map(MetaValue::Str).collect()),
            ),
            (
                "tokenizer.ggml.add_space_prefix",
                MetaValue::Bool(add_space_prefix),
            ),
        ] {
            md.kv.insert(k.into(), v);
        }
        md
    }

    /// That vocabulary built by the loader itself, so the decoder chain is the one an SPM GGUF
    /// really gets.
    fn spm_gguf(add_space_prefix: bool) -> Tokenizer {
        crate::tokenizer_from_metadata(&spm_metadata(add_space_prefix)).unwrap()
    }

    /// The same vocabulary with the decoder an HF Llama-2 `tokenizer.json` ships: `▁` → space,
    /// byte fallback, fuse, then strip ONE leading space. The strip runs on the fused text, so the
    /// first token's space is dropped and every later one is kept — a decoder whose output depends
    /// on where the decode starts, which is what a windowed decode has to get right.
    fn spm_sidecar() -> Tokenizer {
        use tokenizers::decoders::strip::Strip;
        use tokenizers::normalizers::replace::Replace;
        let mut tok = spm_gguf(false);
        tok.with_decoder(Some(DecoderSequence::new(vec![
            DecoderWrapper::Replace(Replace::new("▁", " ").unwrap()),
            DecoderWrapper::ByteFallback(ByteFallback::default()),
            DecoderWrapper::Fuse(Fuse::default()),
            DecoderWrapper::Strip(Strip::new(' ', 1, 0)),
        ])));
        tok
    }

    /// The previous algorithm, kept as the oracle: decode the whole accumulator on every token and
    /// emit what lies past the last emitted byte.
    fn full_decode_stream(tok: &Tokenizer, ids: &[u32]) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut printed = 0usize;
        for n in 1..=ids.len() {
            let full = tok.decode(&ids[..n], true).unwrap();
            if !full.ends_with('\u{FFFD}') && full.len() > printed && full.is_char_boundary(printed)
            {
                chunks.push(full[printed..].to_string());
                printed = full.len();
            }
        }
        chunks
    }

    fn incremental_stream(tok: &impl StreamDecode, ids: &[u32]) -> Vec<String> {
        let mut chunks = Vec::new();
        let (mut acc, mut at) = (Vec::new(), StreamOffsets::default());
        for &id in ids {
            stream_token(tok, &mut acc, &mut at, id, &mut |p: &str| {
                chunks.push(p.to_string())
            });
        }
        chunks
    }

    /// Deterministic ids in `0..n` (xorshift), so a failure reproduces.
    fn random_ids(n: u32, len: usize, mut seed: u64) -> Vec<u32> {
        (0..len)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % n as u64) as u32
            })
            .collect()
    }

    /// The incremental decode emits exactly what decoding the whole accumulator on every token
    /// did — the same chunks, so the same bytes — on the cases where it could differ: multi-byte
    /// characters split across tokens (held back until whole), the decoders SPM GGUFs get with and
    /// without `add_space_prefix`, a decoder whose output depends on where the decode starts (the
    /// Llama-2 `tokenizer.json` chain, see [`spm_sidecar`]), skipped special tokens, invalid byte
    /// runs, and long random streams.
    #[test]
    fn incremental_stream_matches_full_decode() {
        let bl = byte_level();
        let spms = [spm_gguf(false), spm_gguf(true), spm_sidecar()];
        let text = "héllo 😁 世界, naïve café — ok";
        let mut cases: Vec<(&Tokenizer, Vec<u32>)> =
            vec![(&bl, bl.encode(text, false).unwrap().get_ids().to_vec())];
        for spm in &spms {
            cases.push((spm, spm.encode(text, false).unwrap().get_ids().to_vec()));
            cases.push((spm, vec![2, 3, 1, 4, 1, 1, 5, 3]));
        }
        for seed in 1..=6u64 {
            cases.push((&bl, random_ids(bl.get_vocab_size(true) as u32, 600, seed)));
            for spm in &spms {
                cases.push((spm, random_ids(spm.get_vocab_size(true) as u32, 600, seed)));
            }
        }
        for (i, (tok, ids)) in cases.iter().enumerate() {
            let want = full_decode_stream(tok, ids);
            assert!(!want.is_empty(), "case {i} emitted nothing to compare");
            assert_eq!(incremental_stream(*tok, ids), want, "case {i}");
        }
    }

    /// Counts the ids handed to the decoder.
    struct Counting<'a>(&'a Tokenizer, Cell<usize>);

    impl StreamDecode for Counting<'_> {
        fn decode_ids(&self, ids: &[u32]) -> Option<String> {
            self.1.set(self.1.get() + ids.len());
            self.0.decode_ids(ids)
        }
    }

    /// Streaming a reply decodes a bounded number of ids per token, not the whole reply so far.
    /// At this length the whole-accumulator decode handed the decoder n(n+1)/2 ids — about two
    /// thousand per token on average.
    #[test]
    fn streaming_decode_work_is_linear() {
        let tok = byte_level();
        let n = 4000;
        let text = "fn main() { println!(\"héllo, 世界\"); } ".repeat(n / 30 + 1);
        let enc = tok.encode(text.as_str(), false).unwrap();
        let ids = &enc.get_ids()[..n];
        let counting = Counting(&tok, Cell::new(0));
        let out: String = incremental_stream(&counting, ids).concat();
        assert_eq!(
            out,
            tok.decode(ids, true).unwrap().trim_end_matches('\u{FFFD}')
        );
        let decoded = counting.1.get();
        assert!(
            decoded <= 8 * n,
            "{decoded} ids decoded to stream {n} tokens"
        );
    }
}
