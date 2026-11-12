//! GGUF-embedded tokenizer construction (byte-level BPE + SentencePiece), and the one loader that
//! picks between it and a `tokenizer.json` sidecar ([`load_tokenizer`]).
use crate::{
    DEEPSEEK_CODER_PRE_RES, DEEPSEEK_LLM_PRE_RES, DEEPSEEK_V3_PRE_RES, LLAMA3_PRE_RE,
    LLAMA4_PRE_RE, QWEN2_PRE_RE, QWEN35_PRE_RE,
};
use anyhow::{anyhow, bail, Context, Result};
use infr_core::loader::{MetaValue, Metadata};
use infr_core::WeightSource;
use infr_gguf::Gguf;
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::decoders::sequence::Sequence as DecoderSequence;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::{Prepend, Replace};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{AddedToken, DecoderWrapper, SplitDelimiterBehavior, Tokenizer};

/// The `tokenizer.ggml.merges` line grammar: each entry is `"left right"`, ONE space, and the
/// right half may itself contain spaces (SPM pieces are `▁`-prefixed, but a byte-level BPE piece
/// can be a literal space) — hence `splitn(2, ' ')`, not `split`. Entries that are not strings, or
/// that have no space at all, are dropped rather than erroring: a merge we cannot read is one the
/// BPE simply never applies, which degrades tokenization instead of refusing to load the model.
///
/// Single-sources the grammar for both tokenizer families. `build_tokenizer` (gpt2 BPE) and
/// `build_spm_tokenizer` (gemma4's explicit merges) each spelled this expression out; a fix to one
/// copy — the `splitn` bound in particular — silently did not reach the other, and the two families
/// would then disagree about what a merge line means while both claiming to read the same key.
fn parse_merges(arr: &[MetaValue]) -> Vec<(String, String)> {
    arr.iter()
        .filter_map(|m| {
            let s = m.as_str()?;
            let mut it = s.splitn(2, ' ');
            Some((it.next()?.to_string(), it.next()?.to_string()))
        })
        .collect()
}

/// Register `tokenizer.ggml.token_type` 2 (UNKNOWN), 3 (CONTROL) and 4 (USER_DEFINED) tokens on
/// `tok` — the set llama.cpp splits out before tokenizing (`cache_special_tokens`). Unknown and
/// control tokens are SPECIAL (`AddedToken::from(s, true)`), user-defined ones are normal added
/// tokens. All encode atomically; only the special ones are dropped by
/// `decode(.., skip_special=true)`, which is why `<think>`/`</think>` (type 4) must NOT be special —
/// the reasoning block has to stay visible in the output for the reasoning split to find it. Each
/// also takes the whitespace stripping [`SpecialStrip`] gives it.
///
/// Absent `token_type` ⇒ nothing to register. The empty-vec guards are kept because `add_tokens` /
/// `add_special_tokens` on an empty slice still bump the tokenizer's added-vocab bookkeeping, and
/// the ORDER (added before specials) is load-bearing: a string appearing in both lists must end up
/// special, and the later call wins.
///
/// This is the single copy of a policy that used to be written out verbatim in both
/// [`build_tokenizer`] and [`build_spm_tokenizer`]. The two copies were already drifting in their
/// comments about which type meant what, and any new token_type the GGUF spec grows would have had
/// to be added twice — with the SPM family (which loads llama/gemma) the easy one to miss.
fn register_token_types(md: &Metadata, toks: &[MetaValue], tok: &mut Tokenizer) {
    let Some(types) = md
        .get("tokenizer.ggml.token_type")
        .and_then(MetaValue::as_arr)
    else {
        return;
    };
    let strip = SpecialStrip::of(md);
    let mut specials = Vec::new();
    let mut added = Vec::new();
    for (i, ty) in types.iter().enumerate() {
        let Some(s) = toks.get(i).and_then(MetaValue::as_str) else {
            continue;
        };
        let (lstrip, rstrip) = strip.around(s);
        let token = |special| {
            AddedToken::from(s.to_string(), special)
                .lstrip(lstrip)
                .rstrip(rstrip)
        };
        match ty.as_u64() {
            Some(2 | 3) => specials.push(token(true)),
            // Matched on the raw text, before any normalizer, as llama.cpp's
            // `tokenizer_st_partition` does: the SPM space-prefix normalizer must see the text
            // AFTER a user-defined token as its own stretch, not prefix the stretch around it.
            Some(4) => added.push(token(false).normalized(false)),
            _ => {}
        }
    }
    if !added.is_empty() {
        tok.add_tokens(&added);
    }
    if !specials.is_empty() {
        tok.add_special_tokens(&specials);
    }
}

/// Which whitespace next to a special token is dropped when the token is split out of the input:
/// llama.cpp's per-model LSTRIP / RSTRIP attributes (`llama_vocab::impl::load`, "Handle per token
/// attributes"). Decided once per model from the same metadata, matched the same way — a
/// substring of `tokenizer.ggml.pre` or `general.architecture`, or of `general.name` lowercased —
/// and applied per token by [`SpecialStrip::around`]. Phi-3's chat template puts a newline after
/// every turn marker, which its tokenizer drops. Whitespace is `char::is_whitespace` here and
/// ASCII `isspace` there, so the two differ only on non-ASCII whitespace next to such a token.
enum SpecialStrip {
    /// Whitespace before this token (jina-v2 / nomic-bert-moe / jina-bert-v3 `<mask>`,
    /// modern-bert `[MASK]`).
    LeftOf(&'static str),
    /// Whitespace after every special token except these (Phi-3).
    RightExcept(&'static [&'static str]),
    None,
}

impl SpecialStrip {
    fn of(md: &Metadata) -> Self {
        let name = md.str("general.name").unwrap_or("").to_lowercase();
        let pre = md.str("tokenizer.ggml.pre").unwrap_or("");
        let arch = md.str("general.architecture").unwrap_or("");
        let any = |s: &str, subs: &[&str]| subs.iter().any(|sub| s.contains(sub));
        if any(pre, &["jina-v2-de", "jina-v2-es", "jina-v2-code"])
            || any(arch, &["nomic-bert-moe", "jina-bert-v3"])
        {
            Self::LeftOf("<mask>")
        } else if any(&name, &["phi-3", "phi3"]) {
            Self::RightExcept(&["<unk>", "<s>", "<|endoftext|>"])
        } else if any(&name, &["modern-bert"]) {
            Self::LeftOf("[MASK]")
        } else {
            Self::None
        }
    }

    /// `(lstrip, rstrip)` for the special token `text`.
    fn around(&self, text: &str) -> (bool, bool) {
        match self {
            Self::LeftOf(t) => (text == *t, false),
            Self::RightExcept(keep) => (false, !keep.contains(&text)),
            Self::None => (false, false),
        }
    }
}

/// Build an HF `Tokenizer` from the GGUF's embedded vocab (`tokenizer.ggml.*`). Supports the
/// GPT-2 byte-level BPE family (Qwen/Llama-3/SmolLM etc., `tokenizer.ggml.model == "gpt2"`):
/// vocab from `.tokens`, merges from `.merges`, ByteLevel pre-tokenizer + decoder, and control /
/// user-defined tokens (token_type 3/4, e.g. `<|im_start|>`) registered as special so they encode
/// atomically. SentencePiece (`model == "llama"`) isn't built here — pass a `tokenizer.json`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn build_tokenizer(g: &Gguf) -> Result<Tokenizer> {
    tokenizer_from_metadata(g.metadata())
}

/// About how many GGUF token ids [`sidecar_mismatches`] compares with a `tokenizer.json`.
const SIDECAR_CHECK_SAMPLES: usize = 64;

/// The model's tokenizer: the `tokenizer.json` at `sidecar` when the caller found one beside the
/// GGUF, else the one embedded in the GGUF. Every consumer of one model — generation, the chat
/// renderer, the tool-call grammar — loads through here, so they cannot disagree on the vocabulary.
pub(crate) fn load_tokenizer(g: &Gguf, sidecar: Option<&std::path::Path>) -> Result<Tokenizer> {
    let Some(path) = sidecar else {
        return build_tokenizer(g);
    };
    let tok = Tokenizer::from_file(path).map_err(|e| anyhow!("load tokenizer: {e}"))?;
    check_sidecar(g.metadata(), &tok, path);
    Ok(tok)
}

/// Warn when a sidecar does not spell the GGUF's own vocabulary. A `tokenizer.json` is adopted
/// from the GGUF's directory with nothing tying it to this model, so a stray one in a folder of
/// several models encodes every prompt with the wrong ids and the output is garbage with no error.
/// It is a warning, not a refusal: a sidecar is also how a model the embedded tokenizer gets wrong
/// is made to work, and that must keep working.
fn check_sidecar(md: &Metadata, sidecar: &Tokenizer, path: &std::path::Path) {
    let (checked, bad) = sidecar_mismatches(md, sidecar);
    if checked == 0 {
        tracing::warn!(
            "{}: cannot be checked against the GGUF's vocabulary (no token ids in common); \
             using it unverified",
            path.display()
        );
    } else if let Some((id, gguf, side)) = bad.first() {
        tracing::warn!(
            "{}: disagrees with the GGUF's vocabulary on {} of {checked} sampled token ids \
             (id {id}: GGUF {gguf:?}, tokenizer.json {side:?}); it may belong to another model. \
             Remove it to use the GGUF's embedded tokenizer.",
            path.display(),
            bad.len(),
        );
    }
}

/// Compare a sample of the GGUF's token ids, spread over the ids both vocabularies have, with the
/// sidecar's spelling of the same ids. Returns how many were compared and the ones that differ
/// as `(id, GGUF text, sidecar text)`.
///
/// What it does NOT detect: a different vocabulary that spells every sampled id the same — one
/// that extends or truncates the model's own (the sizes are deliberately not compared, since a
/// GGUF may pad its vocabulary), or one that differs only at ids between the samples. It also says
/// nothing about the sidecar's pre-tokenizer or merges, which change ids without changing the
/// vocabulary.
fn sidecar_mismatches(md: &Metadata, sidecar: &Tokenizer) -> (usize, Vec<(u32, String, String)>) {
    let toks = md
        .get("tokenizer.ggml.tokens")
        .and_then(MetaValue::as_arr)
        .unwrap_or(&[]);
    let n = toks.len().min(sidecar.get_vocab_size(true));
    if n == 0 {
        return (0, Vec::new());
    }
    let step = n.div_ceil(SIDECAR_CHECK_SAMPLES);
    let mut bad = Vec::new();
    let mut ids: Vec<usize> = (0..n).step_by(step).collect();
    if ids.last() != Some(&(n - 1)) {
        ids.push(n - 1);
    }
    for &i in &ids {
        let gguf = toks[i].as_str().unwrap_or("");
        let side = sidecar.id_to_token(i as u32).unwrap_or_default();
        if gguf != side {
            bad.push((i as u32, gguf.to_string(), side));
        }
    }
    (ids.len(), bad)
}

/// [`build_tokenizer`] over the metadata alone, so a test can describe a vocabulary without
/// writing a GGUF file.
pub(crate) fn tokenizer_from_metadata(md: &Metadata) -> Result<Tokenizer> {
    let model = md.str("tokenizer.ggml.model").unwrap_or("");
    match model {
        "gpt2" => {}
        // SentencePiece (llama/gemma3/gemma4): a byte-fallback BPE with a Metaspace (▁) word-boundary
        // scheme. gemma4 ships explicit merges; llama/gemma3 reconstruct them from the token scores.
        "llama" | "gemma4" => return build_spm_tokenizer(md),
        other => bail!(
            "can't derive a tokenizer from tokenizer.ggml.model={other:?} \
             (only gpt2 BPE / llama SPM); pass a tokenizer.json sidecar instead"
        ),
    }
    let toks = md
        .get("tokenizer.ggml.tokens")
        .and_then(MetaValue::as_arr)
        .context("gguf missing tokenizer.ggml.tokens")?;
    let vocab: std::collections::HashMap<String, u32> = toks
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.as_str().map(|s| (s.to_string(), i as u32)))
        .collect();
    let merges = parse_merges(
        md.get("tokenizer.ggml.merges")
            .and_then(MetaValue::as_arr)
            .context("gguf missing tokenizer.ggml.merges")?,
    );
    let pre = md.str("tokenizer.ggml.pre").unwrap_or("default");
    // llama.cpp's `LLAMA3` pre-type names (`llama_vocab::impl::load`), which also turn on
    // `ignore_merges`: a pre-token that is itself a vocabulary entry is emitted whole.
    let llama3 = matches!(
        pre,
        "llama3"
            | "llama-v3"
            | "llama-bpe"
            | "falcon3"
            | "falcon-h1"
            | "pixtral"
            | "midm-2.0"
            | "lfm2"
            | "jina-v5-nano"
    );
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .ignore_merges(llama3)
        .build()
        .map_err(|e| anyhow!("build bpe: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    let add_prefix = matches!(
        md.get("tokenizer.ggml.add_space_prefix"),
        Some(MetaValue::Bool(true))
    );
    // The qwen2/qwen35/llama3/llama4 families split on a SINGLE regex before ByteLevel (matching
    // HF); only the pattern differs. DeepSeek pre-types apply a LIST of successive Split pre-tokenizers (see §0.2 of
    // docs/deepseek.md, and [`build_multi_split_seq`] for the token-id evidence that N successive
    // Isolated splits reproduce llama.cpp's sequential application).
    // The sequences below follow llama.cpp's `src/llama-vocab.cpp` verbatim.
    //
    // llama.cpp additionally sets `clean_spaces = false` for all three DeepSeek pre-types. Nothing
    // is needed here to match that: `clean_spaces` is read in exactly one place, the DETOKENIZER
    // (`llama_vocab::impl::detokenize`), where it drops the space before `?!.,`, strips a lone
    // apostrophe between spaces, and closes up `'s`/`'m`/`'re`/`'ve`. It never runs during
    // encoding. llama.cpp turns it ON for every BPE vocab and the DeepSeek arms turn it back off;
    // infr detokenizes through `tokenizers::Tokenizer::decode`, which has no such pass at all, so
    // infr is unconditionally in the `clean_spaces = false` state the DeepSeek arms ask for.
    match pre {
        "deepseek-llm" => {
            let seq = build_multi_split_seq(DEEPSEEK_LLM_PRE_RES.as_slice())?;
            tok.with_pre_tokenizer(Some(seq));
        }
        "deepseek-coder" => {
            let seq = build_multi_split_seq(DEEPSEEK_CODER_PRE_RES.as_slice())?;
            tok.with_pre_tokenizer(Some(seq));
        }
        "deepseek-v3" => {
            let seq = build_multi_split_seq(DEEPSEEK_V3_PRE_RES.as_slice())?;
            tok.with_pre_tokenizer(Some(seq));
        }
        // Grouped by the regex llama.cpp's `llm_tokenizer_bpe` gives each pre-type. QWEN2's arm
        // also serves its `STABLELM2`, `HUNYUAN` and `SOLAR_OPEN`; GPT-4o's (= `LLAMA4_PRE_RE`) its
        // `MINIMAX_M2`.
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "megrez" | "stablelm2" | "hunyuan"
        | "solar-open" | "qwen35" | "llama4" | "gpt-4o" | "kanana2" | "talkie" | "minimax-m2" => {
            let re = match pre {
                "qwen35" => QWEN35_PRE_RE,
                "llama4" | "gpt-4o" | "kanana2" | "talkie" | "minimax-m2" => LLAMA4_PRE_RE,
                _ => QWEN2_PRE_RE,
            };
            single_split_byte_level(&mut tok, re)?;
        }
        // `DBRX`, `SMAUG` and `CHATGLM4` split with the Llama-3 regex but, unlike `LLAMA3`, keep
        // the merges for every pre-token (no `ignore_merges`).
        _ if llama3 => single_split_byte_level(&mut tok, LLAMA3_PRE_RE)?,
        "dbrx" | "smaug-bpe" | "glm4" | "chatglm-bpe" => {
            single_split_byte_level(&mut tok, LLAMA3_PRE_RE)?
        }
        // The pre-types llama.cpp splits with the GPT-2 regex itself (its `GPT2`, `MPT`, `OLMO`,
        // `JAIS`, `TRILLION` and `GRANITE_DOCLING` pre-types), which is what ByteLevel applies.
        "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es" | "jina-v2-de"
        | "a.x-4.0" | "mellum" | "modern-bert" | "jina-v1-en" | "jina-v2-code" | "roberta-bpe"
        | "exaone4" | "mpt" | "olmo" | "jais" | "trillion" | "granite-docling" => {
            tok.with_pre_tokenizer(Some(ByteLevel::new(add_prefix, true, true)));
        }
        // llama.cpp refuses a pre-type it does not know, and splits its `default` with a regex
        // list of its own. Either way the GPT-2 regex is a guess that can give different ids from
        // the reference, so say so instead of tokenizing differently in silence.
        other => {
            tracing::warn!(
                "tokenizer.ggml.pre = {other:?} has no pre-tokenizer mapping here; using the \
                 GPT-2 regex, so token ids may differ from llama.cpp's. Place the model's \
                 tokenizer.json beside the GGUF to tokenize exactly."
            );
            tok.with_pre_tokenizer(Some(ByteLevel::new(add_prefix, true, true)));
        }
    }
    tok.with_decoder(Some(ByteLevelDecoder::default()));
    // Control (type 3, e.g. <|im_end|>) as SPECIAL, user-defined (type 4, e.g. <think>) as normal
    // added tokens — see [`register_token_types`].
    register_token_types(md, toks, &mut tok);
    Ok(tok)
}

/// Install `Sequence[ Split(re, Isolated), ByteLevel(use_regex=false) ]` — one pre-split regex
/// ahead of the byte mapping, which is how HF's `tokenizer.json` spells it.
fn single_split_byte_level(tok: &mut Tokenizer, re: &str) -> Result<()> {
    let split = Split::new(
        SplitPattern::Regex(re.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| anyhow!("split pretokenizer: {e}"))?;
    let seq = PreSequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ]);
    tok.with_pre_tokenizer(Some(seq));
    Ok(())
}

/// The global ordering of reconstructed SPM merge candidates, keyed by `(score, id_l, id_r)`:
/// score DESCENDING (a higher GGUF score is an EARLIER merge), ties broken by `(id_l, id_r)`
/// ASCENDING. That exact ordering is load-bearing — it is what reproduces HF `SpmConverter`'s
/// merge ranking, and flipping either half silently re-tokenizes every prompt on a llama/gemma3
/// GGUF (no merges array, so the ranks come from here). Factored out of
/// [`build_spm_tokenizer`] only so a test can pin it without a model file.
///
/// Scores compare with [`f64::total_cmp`], NOT `partial_cmp().unwrap_or(Equal)`. A NaN score — a
/// corrupt or hand-edited `tokenizer.ggml.scores` — makes the `unwrap_or(Equal)` comparator
/// non-transitive (NaN is "equal" to everything, but the things it ties with are not equal to each
/// other), and Rust's sort DETECTS that and panics with "user-provided comparison function does not
/// correctly implement a total order". So a bad score used to abort model load inside the sort with
/// an error about our comparator instead of about the file. `total_cmp` is a genuine total order
/// over every f64 including NaN, so the sort completes and the corrupt entry merely ranks
/// consistently (NaN sorts to one end) rather than taking the process down.
fn spm_merge_order(a: (f64, u32, u32), b: (f64, u32, u32)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2)))
}

/// Build a `PreSequence` for a list of successive `Split` pre-tokenizers (DeepSeek's multi-regex
/// approach — see docs/deepseek.md §0.2). Each regex splits the output of the previous one in
/// order, followed by `ByteLevel`; mirrors llama.cpp's `unicode_regex_split` pipeline.
///
/// That N successive `Isolated` splits reproduce llama.cpp's sequential `unicode_regex_split` is
/// no longer an assumption: it was checked against `llama-tokenize` token ids on the V2-Lite-Chat
/// Q4_K_M GGUF (`tokenizer.ggml.pre == "deepseek-llm"`, so the six-regex list), agreeing exactly on
/// every text of a battery covering digits, CJK, Hangul, Greek, Hebrew, Arabic, Devanagari,
/// punctuation runs, mixed scripts, emoji, CRLF and non-ASCII whitespace. The comparison has teeth:
/// re-breaking `DEEPSEEK_LLM_PRE_RES[2]` made it disagree with llama.cpp. `deepseek-coder` and
/// `deepseek-v3` are structurally identical but were NOT checked against ids — no GGUF with either
/// pre-type was available. [`build_tokenizer`]'s tests pin all three at the boundary level.
fn build_multi_split_seq(patterns: &[&str]) -> Result<PreSequence> {
    let mut parts: Vec<PreTokenizerWrapper> = patterns
        .iter()
        .map(|re| {
            Split::new(
                SplitPattern::Regex(re.to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map(PreTokenizerWrapper::Split)
            .map_err(|e| anyhow!("split pretokenizer: {e}"))
        })
        .collect::<Result<_, _>>()?;
    parts.push(PreTokenizerWrapper::ByteLevel(ByteLevel::new(
        false, false, false,
    )));
    Ok(PreSequence::new(parts))
}

/// Build a SentencePiece (Unigram) tokenizer from a GGUF's embedded vocab (`tokenizer.ggml.model
/// == "llama"`, used by llama/gemma). The token strings + `scores` become the Unigram lattice;
/// `<0xXX>` byte tokens (token_type 6) are handled by Unigram byte-fallback; CONTROL tokens
/// (type 3, e.g. `<bos>`/`<start_of_turn>`) register as special so they encode atomically. The
/// Metaspace replacement (▁) maps spaces; `add_space_prefix` controls the leading dummy space.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn build_spm_tokenizer(md: &Metadata) -> Result<Tokenizer> {
    let toks = md
        .get("tokenizer.ggml.tokens")
        .and_then(MetaValue::as_arr)
        .context("gguf missing tokenizer.ggml.tokens")?;
    let token_strs: Vec<String> = toks
        .iter()
        .map(|t| t.as_str().unwrap_or("").to_string())
        .collect();
    let vocab: std::collections::HashMap<String, u32> = token_strs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as u32))
        .collect();
    // Merge list for the greedy BPE. gemma4 ships explicit `merges` ("left right", ▁ for spaces);
    // llama/gemma3 don't, so reconstruct them from the token scores the same way HF's SpmConverter
    // builds `LlamaTokenizerFast` (the GGUF scores are negative merge RANKS, not unigram log-probs —
    // a Unigram model would maximize their sum and wrongly split common words). For every piece, each
    // split into two existing pieces is a candidate merge, globally ordered by the merged piece's
    // score (descending = earliest), ties broken by piece ids; greedy BPE over these reproduces SPM.
    let merges: Vec<(String, String)> =
        if let Some(arr) = md.get("tokenizer.ggml.merges").and_then(MetaValue::as_arr) {
            parse_merges(arr)
        } else {
            let scores = md
                .get("tokenizer.ggml.scores")
                .and_then(MetaValue::as_arr)
                .context("gguf needs tokenizer.ggml.merges or .scores for the SPM tokenizer")?;
            // (score, id_l, id_r, l, r) per candidate — global sort by score desc, then (id_l, id_r).
            let mut cand: Vec<(f64, u32, u32, &str, &str)> = Vec::new();
            for (i, piece) in token_strs.iter().enumerate() {
                if piece.len() < 2 {
                    continue;
                }
                let score = scores.get(i).and_then(MetaValue::as_f64).unwrap_or(0.0);
                for (b, _) in piece.char_indices().skip(1) {
                    let (l, r) = piece.split_at(b);
                    if let (Some(&il), Some(&ir)) = (vocab.get(l), vocab.get(r)) {
                        cand.push((score, il, ir, l, r));
                    }
                }
            }
            cand.sort_by(|a, b| spm_merge_order((a.0, a.1, a.2), (b.0, b.1, b.2)));
            cand.into_iter()
                .map(|(_, _, _, l, r)| (l.to_string(), r.to_string()))
                .collect()
        };
    let unk = md
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(MetaValue::as_u64)
        .and_then(|i| token_strs.get(i as usize).cloned())
        .unwrap_or_else(|| "<unk>".to_string());
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .unk_token(unk)
        .byte_fallback(true)
        .fuse_unk(true)
        .build()
        .map_err(|e| anyhow!("build spm bpe: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    // SPM: spaces → ▁. `split` stays OFF: llama.cpp merges over the whole escaped text
    // (`llm_tokenizer_spm_session::tokenize`; gemma4 splits only at newline runs), and splitting at
    // every ▁ would make whitespace-run pieces (`▁▁`, `▁▁▁▁`) unreachable.
    //
    // `add_space_prefix` follows `llama_vocab::impl::load`: an absent key means true for an SPM
    // vocab (`model == "llama"`, so Llama-2, Mistral v0.x, Phi-3) and false for gemma4, which
    // llama.cpp loads as BPE; gemma3/gemma4 GGUFs set it false explicitly.
    let add_prefix = match md.get("tokenizer.ggml.add_space_prefix") {
        Some(MetaValue::Bool(b)) => *b,
        _ => md.str("tokenizer.ggml.model") == Some("llama"),
    };
    tok.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Never, false)));
    if add_prefix {
        // llama.cpp (`llama_vocab::impl::tokenize`, SPM arm) puts a space before every raw-text
        // fragment that starts the input or follows a special or user-defined token, even one
        // that already starts with a space. The normalizer runs once per stretch between added
        // tokens, which is exactly that; Metaspace's own prepend skips text already starting
        // with ▁, so " Hello" would get one ▁ where the reference has two.
        tok.with_normalizer(Some(Prepend::new("▁".into())));
    }
    // Decode: map every ▁ to a space, reassemble byte-fallback bytes, fuse — stripping nothing:
    // the concatenation of llama.cpp's `token_to_piece` with no `lstrip`, which is how its server
    // turns a reply into text, so a reply keeps its leading space. Every caller here decodes
    // generated ids. (`detokenize` additionally drops the first piece's leading space when the
    // sequence does not start with a skipped BOS; HF's `decode` cannot tell a sequence's start
    // from a window into a reply, and nothing here decodes a whole BOS-less sequence.)
    //
    // The ▁ → space step is a `Replace` decoder rather than Metaspace's because the grammar
    // engine reads the decoder to learn each token's bytes: `toktrie_hf_tokenizers` takes the
    // space character only from a `Replace("▁", " ")`, and otherwise hands the constraint every
    // ▁ as its UTF-8 bytes, so a forced JSON tool call could not contain a space.
    tok.with_decoder(Some(DecoderSequence::new(vec![
        DecoderWrapper::Replace(Replace::new("▁", " ").map_err(|e| anyhow!("replace: {e}"))?),
        DecoderWrapper::ByteFallback(ByteFallback::default()),
        DecoderWrapper::Fuse(Fuse::default()),
    ])));
    // CONTROL tokens (type 3, e.g. <bos>/<start_of_turn>/<end_of_turn>) encode atomically as
    // special; USER_DEFINED (type 4) as normal added tokens — see [`register_token_types`].
    register_token_types(md, toks, &mut tok);
    Ok(tok)
}

#[cfg(test)]
mod tokenizer_tests {
    use super::*;
    use crate::sampling::{sample_logits, Sampler};

    /// The chunk boundaries a DeepSeek pre-tokenizer sequence yields, as slices of the ORIGINAL
    /// text. [`build_multi_split_seq`] ends the sequence with `ByteLevel`, which rewrites each
    /// chunk's CONTENT (a space becomes `\u{0120}`, every non-ASCII byte a printable stand-in),
    /// so this reads the original-referential offsets instead of the chunk strings. Those offsets
    /// are the boundaries the BPE then merges within, and they are exactly what a wrong character
    /// class moves — which is the thing under test.
    fn pre_chunks(patterns: &[&str], text: &str) -> Vec<String> {
        use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};
        let seq = build_multi_split_seq(patterns).expect("pattern list must compile");
        let mut pre = PreTokenizedString::from(text);
        seq.pre_tokenize(&mut pre).expect("pre-tokenize");
        pre.get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(_, (start, end), _)| text[start..end].to_string())
            .collect()
    }

    /// The three DeepSeek pre-tokenizer regex lists, pinned by the chunk boundaries they produce.
    ///
    /// These constants are transcribed from llama.cpp's `src/llama-vocab.cpp`, and a wrong
    /// character inside a class does not fail to compile, does not error and does not crash — it
    /// silently moves a chunk boundary, which silently changes token ids, which silently changes
    /// the model's output. Two such slips shipped (`docs/backlog.md` B43), and the `deepseek-llm`
    /// table below has rows that go red under each of them:
    ///
    /// * The Hebrew rows catch `DEEPSEEK_LLM_PRE_RES[2]` opening its quote range with `\u{0027}`
    ///   instead of `\u{2018}`. That runs the class from U+0027 to U+201F — most of the BMP,
    ///   including Hebrew, Arabic, Devanagari and the ASCII digits — so split #3 swallowed the
    ///   Hebrew word together with the full stop after it. Nothing after split #3 matches Hebrew,
    ///   so no later split puts that boundary back.
    /// * The `\u{2139}\u{213A}\u{213F}` row catches `DEEPSEEK_LLM_PRE_RES[1]` having
    ///   `\u{2139}-\u{213F}` where the reference has `\u{2139}\u{213C}-\u{213F}`.
    ///   The two extra letters made split #2 run all three characters together as one word.
    ///
    /// The `"a \u{00A0}. b"` row is the one that was shown to move real TOKEN IDS, not just
    /// boundaries: with the broken class, `\u{00A0}` (NO-BREAK SPACE, and NBSP before punctuation
    /// is ordinary French typography and ordinary scraped HTML) is pulled into split #3 together
    /// with the space before it and the full stop after it. On the V2-Lite vocab that turns
    /// llama.cpp's `[64, 207, 1202, 13, 270]` into `[64, 30683, 13, 270]`. The Hebrew and
    /// letterlike rows above move boundaries but happen NOT to move ids on that vocab, because it
    /// has no merge spanning the boundary they lose — so they guard the pre-tokenizer, while this
    /// row also guards the observable end of it.
    ///
    /// `deepseek-coder` and `deepseek-v3` were byte-identical to the reference when this was
    /// written (verified codepoint by codepoint against `llama-vocab.cpp`); their tables are here
    /// so the next slip lands on a red test rather than on a user. The differences between the
    /// three tables are real behaviour, not noise: `deepseek-coder` splits digits one at a time
    /// (`\p{N}`, no `+`) and matches Hebrew through `\p{L}`, while `deepseek-v3` keeps
    /// `(x` together because its final pattern binds an opening delimiter to the letters after it.
    #[test]
    fn deepseek_pre_split_boundaries_match_the_reference_lists() {
        fn check(name: &str, patterns: &[&str], cases: &[(&str, &[&str])]) {
            for (text, want) in cases {
                let got = pre_chunks(patterns, text);
                assert_eq!(
                    got, *want,
                    "{name} chunk boundaries moved for {text:?}\n  got  {got:?}\n  want {want:?}"
                );
            }
        }

        check(
            "deepseek-llm",
            DEEPSEEK_LLM_PRE_RES.as_slice(),
            &[
                ("a \u{00A0}. b", &["a", " ", "\u{00A0}.", " b"]),
                ("שלום.", &["שלום", "."]),
                ("עברית 42 test.", &["עברית ", "42", " test", "."]),
                ("ℹ℺ℿ", &["ℹ", "℺", "ℿ"]),
                ("3.14", &["3", ".", "14"]),
                (
                    "50% off, 2 items?",
                    &["50", "%", " off", ",", " ", "2", " items", "?"],
                ),
                ("中文 汉字。", &["中文", " ", "汉字", "。"]),
                (
                    "def f(x): return x*2\n",
                    &["def", " f", "(", "x", "):", " return", " x", "*", "2", "\n"],
                ),
            ],
        );
        check(
            "deepseek-coder",
            DEEPSEEK_CODER_PRE_RES.as_slice(),
            &[
                ("a \u{00A0}. b", &["a", " ", "\u{00A0}.", " b"]),
                ("שלום.", &["שלום", "."]),
                ("עברית 42 test.", &["עברית", " ", "4", "2", " test", "."]),
                ("ℹ℺ℿ", &["ℹ", "℺", "ℿ"]),
                ("3.14", &["3", ".", "1", "4"]),
                (
                    "50% off, 2 items?",
                    &["5", "0", "%", " off", ",", " ", "2", " items", "?"],
                ),
                ("中文 汉字。", &["中文", " ", "汉字", "。"]),
                (
                    "def f(x): return x*2\n",
                    &["def", " f", "(", "x", "):", " return", " x", "*", "2", "\n"],
                ),
            ],
        );
        check(
            "deepseek-v3",
            DEEPSEEK_V3_PRE_RES.as_slice(),
            &[
                ("a \u{00A0}. b", &["a", " ", "\u{00A0}", ".", " b"]),
                ("שלום.", &["שלום", "."]),
                ("עברית 42 test.", &["עברית", " ", "42", " test", "."]),
                ("ℹ℺ℿ", &["ℹ", "℺", "ℿ"]),
                ("3.14", &["3", ".", "14"]),
                (
                    "50% off, 2 items?",
                    &["50", "%", " off", ",", " ", "2", " items", "?"],
                ),
                ("中文 汉字。", &["中文", " ", "汉字", "。"]),
                (
                    "def f(x): return x*2\n",
                    &["def", " f", "(x", "):", " return", " x", "*", "2", "\n"],
                ),
            ],
        );
    }

    /// The SPM merge ranking, pinned in both halves and made NaN-proof.
    ///
    /// Two things are asserted, and they pull in opposite directions:
    ///
    /// 1. The ORDER must not move. Score descending, `(id_l, id_r)` ascending on ties — that is
    ///    HF `SpmConverter`'s ranking, and every llama/gemma3 GGUF (no `tokenizer.ggml.merges`)
    ///    tokenizes through it. A flip here is a silent re-tokenization of every prompt.
    /// 2. A NaN score must not panic the sort. `partial_cmp().unwrap_or(Equal)` is non-transitive
    ///    in the presence of NaN, and `sort_by` detects that and panics with "user-provided
    ///    comparison function does not correctly implement a total order" — so one corrupt entry
    ///    in `tokenizer.ggml.scores` used to abort model load. The full sort below is the actual
    ///    regression test: it is exactly the call site's `sort_by`, and it panics under the old
    ///    comparator.
    #[test]
    fn spm_merge_order_is_a_total_order_and_keeps_its_direction() {
        use std::cmp::Ordering::{Greater, Less};
        // Higher score sorts FIRST (earlier merge), regardless of ids.
        assert_eq!(spm_merge_order((-1.0, 9, 9), (-2.0, 0, 0)), Less);
        assert_eq!(spm_merge_order((-2.0, 0, 0), (-1.0, 9, 9)), Greater);
        // Equal scores: lower (id_l, id_r) sorts first, id_l dominating id_r.
        assert_eq!(spm_merge_order((-1.0, 3, 7), (-1.0, 4, 0)), Less);
        assert_eq!(spm_merge_order((-1.0, 3, 7), (-1.0, 3, 8)), Less);
        assert_eq!(
            spm_merge_order((-1.0, 3, 7), (-1.0, 3, 7)),
            std::cmp::Ordering::Equal
        );

        // A candidate list with NaN scores sprinkled through it. Two properties of this data are
        // deliberate and BOTH are needed to reproduce the panic:
        //
        //   * scores are not monotone in `(id_l, id_r)` — a real candidate list isn't either, since
        //     a candidate carries the MERGED piece's score alongside the two SPLIT pieces' ids. On
        //     monotone data the old comparator's id tiebreak happens to agree with the scores and
        //     no cycle forms, so the sort completes and the bug hides.
        //   * enough elements that the sort actually runs its merge (n >= ~32); the insertion sort
        //     used for tiny slices does not detect the violation.
        //
        // Under `partial_cmp().unwrap_or(Equal)` a NaN ties with every finite score, the id
        // tiebreak then decides those pairs, and `NaN > y`, `y > z`, `z > NaN` becomes reachable —
        // a cycle. `sort_by` detects it and panics. Verified: this exact shape panics with the old
        // comparator at every n >= 32.
        let n = 128u32;
        let mut cand: Vec<(f64, u32, u32)> = (0..n)
            .map(|i| {
                let score = if i % 16 == 7 {
                    f64::NAN
                } else {
                    -(((i as f64) * 7.0) % 31.0)
                };
                (score, i, n - i)
            })
            .collect();
        cand.sort_by(|a, b| spm_merge_order(*a, *b));
        // Nothing dropped, and the finite entries keep their descending-score order among
        // themselves (NaN sorts to one end and does not interleave).
        assert_eq!(cand.len(), n as usize, "no candidate may be dropped");
        let finite: Vec<f64> = cand.iter().map(|c| c.0).filter(|s| !s.is_nan()).collect();
        assert!(
            finite.windows(2).all(|w| w[0] >= w[1]),
            "finite scores must stay descending: {finite:?}"
        );
    }

    /// Tokenizer metadata for a synthetic vocabulary: `model`, `tokens`, and whatever else `extra`
    /// adds — enough for [`tokenizer_from_metadata`] without a GGUF file.
    fn vocab_md(model: &str, tokens: &[&str], extra: Vec<(&str, MetaValue)>) -> Metadata {
        let mut md = Metadata::default();
        md.kv
            .insert("tokenizer.ggml.model".into(), MetaValue::Str(model.into()));
        md.kv
            .insert("tokenizer.ggml.tokens".into(), str_arr(tokens));
        for (k, v) in extra {
            md.kv.insert(k.into(), v);
        }
        md
    }

    fn str_arr(v: &[&str]) -> MetaValue {
        MetaValue::Arr(v.iter().map(|t| MetaValue::Str(t.to_string())).collect())
    }

    /// A run of spaces reaches the vocabulary's whitespace-run pieces, as llama.cpp's SPM tokenizer
    /// (`llm_tokenizer_spm_session::tokenize`, run over the whole escaped text) and its gemma4 BPE
    /// path both do. With Metaspace's `split` on, every `▁` became its own pre-token first, so
    /// `▁▁▁▁` could never be produced and indented text got one id per space. Checked against
    /// llama.cpp's `ggml-vocab-gemma-4.gguf` / `ggml-vocab-llama-spm.gguf` fixtures, whose
    /// multi-space rows went from mismatching to matching with this change.
    #[test]
    fn spm_space_runs_reach_multi_space_pieces() {
        let tokens = ["<unk>", "▁", "▁▁", "▁▁▁▁", "a"];
        // llama/gemma3 rebuild their merges from the scores; gemma4 ships them.
        // The prefix is off so the input's own spaces are the whole run.
        let scored = vocab_md(
            "llama",
            &tokens,
            vec![
                (
                    "tokenizer.ggml.scores",
                    MetaValue::Arr([0.0, -1.0, -2.0, -3.0, -1.0].map(MetaValue::F64).to_vec()),
                ),
                ("tokenizer.ggml.add_space_prefix", MetaValue::Bool(false)),
            ],
        );
        let merged = vocab_md(
            "gemma4",
            &tokens,
            vec![("tokenizer.ggml.merges", str_arr(&["▁ ▁", "▁▁ ▁▁"]))],
        );
        for md in [scored, merged] {
            let tok = tokenizer_from_metadata(&md).expect("synthetic SPM vocab");
            let ids = tok.encode("    a", false).unwrap().get_ids().to_vec();
            assert_eq!(ids, [3, 4], "{:?}", md.str("tokenizer.ggml.model"));
        }
    }

    /// `add_space_prefix` as `llama_vocab::impl::load` and `tokenize` apply it: an SPM vocab
    /// without the key gets the prefix, before every stretch of text that starts the input or
    /// follows a special, unknown or user-defined token, even one already starting with a space.
    /// Decoding maps every `▁` to a space and strips none, as llama.cpp's server builds a reply
    /// from `token_to_piece`. An explicit false (gemma3) and gemma4's absent key mean no prefix.
    #[test]
    fn spm_space_prefix_follows_the_reference() {
        // 0 <unk> (UNKNOWN), 1 <s> (CONTROL), 2 ▁, 3 a, 4 ▁a, 5 b, 6 ▁b, 7 !, 8 <x> (USER_DEFINED)
        let tokens = ["<unk>", "<s>", "▁", "a", "▁a", "b", "▁b", "!", "<x>"];
        let types = [2, 3, 1, 1, 1, 1, 1, 1, 4];
        let md = |model: &str, prefix: Option<bool>| {
            let mut extra = vec![
                (
                    "tokenizer.ggml.scores",
                    MetaValue::Arr(tokens.iter().map(|_| MetaValue::F64(0.0)).collect()),
                ),
                (
                    "tokenizer.ggml.token_type",
                    MetaValue::Arr(types.map(MetaValue::U64).to_vec()),
                ),
            ];
            if let Some(p) = prefix {
                extra.push(("tokenizer.ggml.add_space_prefix", MetaValue::Bool(p)));
            }
            tokenizer_from_metadata(&vocab_md(model, &tokens, extra)).expect("synthetic SPM")
        };
        let ids = |tok: &Tokenizer, text: &str| tok.encode(text, false).unwrap().get_ids().to_vec();

        for tok in [md("llama", None), md("llama", Some(true))] {
            assert_eq!(ids(&tok, "a b!"), [4, 6, 7]);
            assert_eq!(ids(&tok, " a"), [2, 4]);
            assert_eq!(ids(&tok, "a<s>b"), [4, 1, 6]);
            assert_eq!(ids(&tok, "<s>a<x>b"), [1, 4, 8, 6]);
            assert_eq!(ids(&tok, "a<unk>b"), [4, 0, 6]);
            assert_eq!(tok.decode(&[4, 6, 7], true).unwrap(), " a b!");
            for text in ["a b!", " a", "  b"] {
                assert_eq!(
                    tok.decode(&ids(&tok, text), true).unwrap(),
                    format!(" {text}")
                );
            }
        }
        for tok in [md("llama", Some(false)), md("gemma4", None)] {
            assert_eq!(ids(&tok, " a b"), [4, 6]);
            assert_eq!(tok.decode(&[4, 6, 7], true).unwrap(), " a b!");
        }
    }

    /// The grammar engine learns each token's bytes from the decoder chain. An SPM piece must reach
    /// it with `▁` as a space, or a constrained (forced tool-call) JSON reply could never contain
    /// one; with the prefix on and off alike.
    #[test]
    fn spm_tokens_reach_the_grammar_engine_with_spaces() {
        let tokens = ["<unk>", "<s>", "▁", "a", "▁a"];
        for prefix in [None, Some(false)] {
            let mut extra = vec![
                (
                    "tokenizer.ggml.scores",
                    MetaValue::Arr(tokens.iter().map(|_| MetaValue::F64(0.0)).collect()),
                ),
                (
                    "tokenizer.ggml.token_type",
                    MetaValue::Arr([2, 3, 1, 1, 1].map(MetaValue::U64).to_vec()),
                ),
            ];
            if let Some(p) = prefix {
                extra.push(("tokenizer.ggml.add_space_prefix", MetaValue::Bool(p)));
            }
            let tok = tokenizer_from_metadata(&vocab_md("llama", &tokens, extra)).unwrap();
            let env = crate::grammar::build_tok_env(&tok, tokens.len(), &[1]).expect("tok env");
            assert_eq!(env.tok_trie().token(4), b" a", "prefix {prefix:?}");
            assert_eq!(env.tok_trie().token(2), b" ", "prefix {prefix:?}");
        }
    }

    /// llama.cpp's per-model whitespace stripping around special tokens: under a `general.name`
    /// naming Phi-3, whitespace after every special token but `<s>` (and `<unk>`,
    /// `<|endoftext|>`) is dropped before the next stretch is prefixed; any other name keeps it.
    /// Under `tokenizer.ggml.pre` jina-v2-* or a `general.name` naming modern-bert, whitespace
    /// before `<mask>` / `[MASK]` is dropped.
    #[test]
    fn special_token_whitespace_strips_like_the_reference() {
        // 0 <unk>, 1 <s>, 2 ▁, 3 a, 4 ▁a, 5 <|user|>, 6 <|end|>
        let tokens = ["<unk>", "<s>", "▁", "a", "▁a", "<|user|>", "<|end|>"];
        let tok = |name: &str| {
            tokenizer_from_metadata(&vocab_md(
                "llama",
                &tokens,
                vec![
                    (
                        "tokenizer.ggml.scores",
                        MetaValue::Arr(tokens.iter().map(|_| MetaValue::F64(0.0)).collect()),
                    ),
                    (
                        "tokenizer.ggml.token_type",
                        MetaValue::Arr([2, 3, 1, 1, 1, 3, 3].map(MetaValue::U64).to_vec()),
                    ),
                    ("general.name", MetaValue::Str(name.into())),
                ],
            ))
            .expect("synthetic SPM")
        };
        let ids = |tok: &Tokenizer, text: &str| tok.encode(text, false).unwrap().get_ids().to_vec();
        let phi = tok("Phi3");
        assert_eq!(ids(&phi, "<|user|>\n a<|end|> \n"), [5, 4, 6]);
        assert_eq!(ids(&phi, "<s> a"), [1, 2, 4]);
        let other = tok("llama-spm");
        assert_eq!(ids(&other, "<s> a"), [1, 2, 4]);
        assert_eq!(ids(&other, "<|user|> a"), [5, 2, 4]);

        let mask = |pre: &str, name: &str, mask: &str| {
            let md = vocab_md(
                "gpt2",
                &["a", "\u{0120}", mask],
                vec![
                    ("tokenizer.ggml.merges", str_arr(&[])),
                    ("tokenizer.ggml.pre", MetaValue::Str(pre.into())),
                    ("general.name", MetaValue::Str(name.into())),
                    (
                        "tokenizer.ggml.token_type",
                        MetaValue::Arr([1, 1, 3].map(MetaValue::U64).to_vec()),
                    ),
                ],
            );
            let tok = tokenizer_from_metadata(&md).expect("synthetic BPE vocab");
            tok.encode(format!("a {mask}"), false)
                .unwrap()
                .get_ids()
                .to_vec()
        };
        assert_eq!(mask("jina-v2-es", "x", "<mask>"), [0, 2]);
        assert_eq!(mask("gpt-2", "modern-bert-base", "[MASK]"), [0, 2]);
        assert_eq!(mask("gpt-2", "x", "[MASK]"), [0, 1, 2]);
    }

    /// The pre-split chunks `pre` produces for `text` on a byte-level BPE vocabulary, as slices of
    /// the original text (see [`pre_chunks`] for why offsets rather than the chunk strings).
    fn bpe_pre_chunks(pre: &str, text: &str) -> Vec<String> {
        use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};
        let md = vocab_md(
            "gpt2",
            &["a"],
            vec![
                ("tokenizer.ggml.merges", str_arr(&[])),
                ("tokenizer.ggml.pre", MetaValue::Str(pre.into())),
            ],
        );
        let tok = tokenizer_from_metadata(&md).expect("synthetic BPE vocab");
        let mut s = PreTokenizedString::from(text);
        tok.get_pre_tokenizer()
            .expect("a pre-tokenizer")
            .pre_tokenize(&mut s)
            .expect("pre-tokenize");
        s.get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(_, (a, b), _)| text[a..b].to_string())
            .collect()
    }

    /// `llama-bpe` (Llama-3) and `qwen35` get llama.cpp's own regexes (`llm_tokenizer_bpe`'s
    /// `LLAMA3` and `QWEN35` arms), not the GPT-2 one they used to fall through to: Llama-3
    /// groups digits in threes, Qwen3.5 keeps a combining mark inside its letter run. Against
    /// llama.cpp's `ggml-vocab-llama-bpe.gguf` and `ggml-vocab-qwen35.gguf` fixtures this took
    /// the mismatching rows to none. `gpt-2` is the control: it IS the GPT-2 regex.
    #[test]
    fn mapped_pre_types_split_like_the_reference() {
        assert_eq!(bpe_pre_chunks("llama-bpe", "3333 x"), ["333", "3", " x"]);
        assert_eq!(bpe_pre_chunks("qwen35", "e\u{301}x"), ["e\u{301}x"]);
        assert_eq!(bpe_pre_chunks("gpt-2", "3333 x"), ["3333", " x"]);
        // The pre-types that share one of those regexes in llama.cpp's table.
        for pre in ["dbrx", "smaug-bpe", "glm4", "chatglm-bpe"] {
            assert_eq!(bpe_pre_chunks(pre, "3333 x"), ["333", "3", " x"], "{pre}");
        }
        for pre in ["stablelm2", "hunyuan", "solar-open"] {
            assert_eq!(bpe_pre_chunks(pre, "33 x"), ["3", "3", " x"], "{pre}");
        }
        assert_eq!(bpe_pre_chunks("minimax-m2", "3333 x"), ["333", "3", " x"]);
    }

    /// A sidecar that spells the GGUF's vocabulary passes the check; one from another model is
    /// caught, by the ids it spells differently.
    #[test]
    fn a_foreign_sidecar_is_detected() {
        let tokens: Vec<String> = (0..300).map(|i| format!("t{i}")).collect();
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let md = vocab_md("gpt2", &refs, vec![]);
        let sidecar = |names: &[String]| {
            let v = names
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), i as u32))
                .collect();
            Tokenizer::new(BPE::builder().vocab_and_merges(v, vec![]).build().unwrap())
        };
        let (checked, bad) = sidecar_mismatches(&md, &sidecar(&tokens));
        assert!(checked > 1, "nothing was compared");
        assert!(
            bad.is_empty(),
            "the model's own vocabulary flagged: {bad:?}"
        );

        let foreign: Vec<String> = (0..300).map(|i| format!("u{i}")).collect();
        let (checked, bad) = sidecar_mismatches(&md, &sidecar(&foreign));
        assert_eq!(bad.len(), checked, "every sampled id differs");
        assert_eq!(bad[0], (0, "t0".to_string(), "u0".to_string()));

        // A vocabulary smaller than the sample compares each id once — its last one included.
        let md = vocab_md("gpt2", &refs[..10], vec![]);
        assert_eq!(sidecar_mismatches(&md, &sidecar(&tokens[..10])).0, 10);
    }

    /// `ignore_merges` follows llama.cpp: on for its `LLAMA3` names, off for the pre-types that
    /// share the Llama-3 regex without it. Here `abc` is a vocabulary entry the merges cannot reach
    /// (there is no `ab c` merge), so only `ignore_merges` emits it whole.
    #[test]
    fn ignore_merges_follows_the_llama3_pre_types() {
        let ids = |pre: &str| {
            let md = vocab_md(
                "gpt2",
                &["a", "b", "c", "ab", "abc"],
                vec![
                    ("tokenizer.ggml.merges", str_arr(&["a b"])),
                    ("tokenizer.ggml.pre", MetaValue::Str(pre.into())),
                ],
            );
            let tok = tokenizer_from_metadata(&md).expect("synthetic BPE vocab");
            tok.encode("abc", false).unwrap().get_ids().to_vec()
        };
        assert_eq!(ids("llama-bpe"), [4]);
        assert_eq!(ids("dbrx"), [3, 2]);
        assert_eq!(ids("glm4"), [3, 2]);
    }

    // Validate the GGUF-derived tokenizer against the HF tokenizer.json sidecar (same model).
    // Skips if the test model isn't present.
    #[test]
    fn embedded_tokenizer_matches_sidecar() {
        let Some(gguf) = crate::test_qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not in the HF cache");
            return;
        };
        // The sidecar tokenizer.json must sit beside the GGUF (HF cache blobs are content-addressed
        // with no sidecar, so this runs only where a snapshot ships tokenizer.json).
        let side = gguf.with_file_name("tokenizer.json");
        if !side.exists() {
            eprintln!("skip: no tokenizer.json sidecar beside the GGUF");
            return;
        }
        let g = Gguf::open(&gguf).unwrap();
        let derived = build_tokenizer(&g).unwrap();
        let sidecar = Tokenizer::from_file(&side).unwrap();
        for s in [
            "Hello world",
            "The quick brown fox.",
            "<|im_start|>user\nWhat is two plus two?<|im_end|>\n<|im_start|>assistant\n",
            "café déjà vu — 123 + 456 = 579",
            "def f(x):\n    return x * 2\n",
        ] {
            let a = derived.encode(s, false).unwrap();
            let b = sidecar.encode(s, false).unwrap();
            assert_eq!(a.get_ids(), b.get_ids(), "token id mismatch on {s:?}");
        }
        // <think>/</think> are user-defined (non-special): skip_special must KEEP them, while real
        // special tokens (<|im_end|>) are dropped — matching the sidecar.
        let think = "<think>\nreasoning\n</think>\n\nanswer<|im_end|>";
        let ids = derived.encode(think, false).unwrap();
        let d = derived.decode(ids.get_ids(), true).unwrap();
        assert!(
            d.contains("<think>") && d.contains("</think>"),
            "think tags dropped: {d:?}"
        );
        assert!(!d.contains("<|im_end|>"), "special token kept: {d:?}");
        assert_eq!(
            d,
            sidecar.decode(ids.get_ids(), true).unwrap(),
            "decode differs from sidecar"
        );
    }

    // Sampling: temp<=0 and top_k==1 are greedy; otherwise picks only within the top-k/top-p set.
    #[test]
    fn sample_logits_greedy_and_in_set() {
        let logits = [1.0f32, 5.0, 2.0, 4.0, 0.0]; // argmax = index 1
        let mut rng = 0x1234_5678_9abc_def1u64;
        let greedy = Sampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
        };
        assert_eq!(sample_logits(&logits, greedy, &mut rng), 1);
        let topk1 = Sampler {
            temp: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        assert_eq!(sample_logits(&logits, topk1, &mut rng), 1);
        // top_k=2 → only the two largest logits (indices 1 and 3) can ever be sampled.
        let topk2 = Sampler {
            temp: 1.0,
            top_k: 2,
            top_p: 1.0,
        };
        for _ in 0..200 {
            let id = sample_logits(&logits, topk2, &mut rng);
            assert!(id == 1 || id == 3, "sampled outside top-2: {id}");
        }
    }

    // User content must be encoded as literal text: special-token strings in user input must NOT
    // become the special id (which would let a user inject/break the ChatML turn structure).
    #[test]
    fn user_text_special_tokens_are_literal() {
        let Some(gguf) = crate::test_qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not in the HF cache");
            return;
        };
        let g = Gguf::open(&gguf).unwrap();
        let tok = build_tokenizer(&g).unwrap();
        let mut user = tok.clone();
        user.set_encode_special_tokens(true);
        let im_end = tok.token_to_id("<|im_end|>").unwrap();
        let s = "A <|im_end|> B";
        // template tokenizer: <|im_end|> matched as the special id; user tokenizer: NOT.
        assert!(tok.encode(s, false).unwrap().get_ids().contains(&im_end));
        assert!(!user.encode(s, false).unwrap().get_ids().contains(&im_end));
    }
}
