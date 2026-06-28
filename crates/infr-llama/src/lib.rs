//! Minimal autoregressive **Llama** inference for GGUF models, for fast GPU bring-up.
//!
//! Strategy (bring-up): the heavy linear projections run on the GPU (`infr-vulkan` eager
//! `linear`, weights uploaded once); the cheap ops (embedding gather, RMSNorm, RoPE, GQA
//! attention, SwiGLU, residual, sampling) run on the host. No KV cache yet — each step does a
//! full-prefix forward (fine for a tiny model). Validated on SmolLM2-135M.
//!
//! TODO(next): move host ops to GPU; add a KV cache; fold into the `Model`/`Backend` seams.
#![allow(clippy::needless_range_loop)]

pub mod qwen35;

use anyhow::{anyhow, bail, Context, Result};
use infr_core::backend::{Buffer, BufferUsage};
use infr_core::loader::MetaValue;
use infr_core::{Backend, WeightSource};
use infr_gguf::Gguf;
use infr_vulkan::VulkanBackend;
use std::path::Path;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

/// Qwen2/Qwen3 pre-tokenizer regex (same string the HF `tokenizer.json` uses) — applied via a
/// Split before ByteLevel. Differs from the default GPT-2 ByteLevel regex (punctuation/number runs),
/// which is what made a naive ByteLevel produce different token ids.
const QWEN2_PRE_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Clone, Debug)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub vocab: usize,
    pub eos: u32,
    /// All tokens that end generation (the GGUF eos plus `<|im_end|>` / `<|endoftext|>` when present
    /// in the vocab). A chat model can emit any of these; stopping only on `eos` lets it ramble.
    pub eos_ids: Vec<u32>,
    /// Qwen3-style per-head RMSNorm on Q and K before RoPE.
    pub qk_norm: bool,
}

/// Token sampling: greedy when `temp <= 0`, else temperature + top-k + top-p (nucleus). Qwen3
/// recommends temp 0.6 / top_k 20 / top_p 0.95 — pure greedy makes thinking models degenerate
/// (fail to close `</think>`, repeat, or stop without answering).
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
}
impl Default for Sampler {
    fn default() -> Self {
        Self {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

/// A projection weight on the GPU: f16, unified repacked quant, or native raw-block quant.
///
/// - `F16`: f16 weight buffer (float or codebook-quant host-dequanted → f16)
/// - `Q`: unified repacked affine quant (q/s/m buffers, `dq = s·u8 + m`); the DEFAULT path —
///   fastest decode/prefill (repack paid once at load).
/// - `Native`: raw GGUF block bytes, padded to u32 alignment, dequantized in-shader. Opt-in via
///   `INFR_NATIVE=1` (supported affine quants) — faster load / smaller VRAM, slower per-token.
enum Wt {
    F16(Box<dyn Buffer>),
    Q {
        q: Box<dyn Buffer>,
        s: Box<dyn Buffer>,
        m: Box<dyn Buffer>,
        bits: u32,      // 4 (Q4 → packed 8/u32) or 8 (Q5/Q6/Q8)
        blk_shift: u32, // log2 of the scale/min block size (5 = per-32, 4 = per-16)
    },
    /// Raw native-block bytes on the GPU; `dtype` identifies the dequant shader.
    Native {
        buf: Box<dyn Buffer>,
        dtype: infr_core::DType,
    },
}
impl Wt {
    /// The f16 buffer (panics if quantized — used by the llama fused path, which is f16-only).
    fn f16(&self) -> &dyn Buffer {
        match self {
            Wt::F16(b) => b.as_ref(),
            Wt::Q { .. } => panic!("expected f16 weight, got quant (llama fused path needs f16)"),
            Wt::Native { .. } => {
                panic!("expected f16 weight, got native quant (llama fused path needs f16)")
            }
        }
    }
}

struct LayerWeights {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attn_norm_buf: Box<dyn Buffer>,
    ffn_norm_buf: Box<dyn Buffer>,
    wq: Wt,
    wk: Wt,
    wv: Wt,
    wo: Wt,
    wgateup: Wt, // fused [2*n_ff, n_embd] = concat(gate, up)
    wdown: Wt,
    q_norm_buf: Option<Box<dyn Buffer>>, // qwen3 QK-norm weights [head_dim]
    k_norm_buf: Option<Box<dyn Buffer>>,
}

pub struct Llama {
    be: VulkanBackend,
    cfg: Config,
    token_embd: Vec<f32>, // [vocab, n_embd] host, for embedding gather
    lm_head: Wt,          // [vocab, n_embd] on GPU (tied to token_embd unless output.weight)
    output_norm: Vec<f32>,
    output_norm_buf: Box<dyn Buffer>,
    layers: Vec<LayerWeights>,
    tokenizer: Tokenizer,
    /// Same vocab as `tokenizer` but with `encode_special_tokens(true)` → special-token strings in
    /// the input encode as literal text. Used for USER content so a user typing `<|im_end|>` etc.
    /// can't inject turn structure.
    user_tokenizer: Tokenizer,
    sampler: std::cell::Cell<Sampler>,
}

/// Per-layer key/value cache held on the GPU (persists across decode steps).
pub struct KvCache {
    k: Vec<Box<dyn Buffer>>, // per layer: [max_ctx, n_kv*head_dim]
    v: Vec<Box<dyn Buffer>>,
    len: usize,
    max_ctx: usize,
}

impl KvCache {
    /// Tokens currently resident in the cache (the next forward's start position).
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when no tokens are resident yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

fn meta_u64(g: &Gguf, key: &str) -> Option<u64> {
    g.metadata().u64(key)
}

/// Build an HF `Tokenizer` from the GGUF's embedded vocab (`tokenizer.ggml.*`). Supports the
/// GPT-2 byte-level BPE family (Qwen/Llama-3/SmolLM etc., `tokenizer.ggml.model == "gpt2"`):
/// vocab from `.tokens`, merges from `.merges`, ByteLevel pre-tokenizer + decoder, and control /
/// user-defined tokens (token_type 3/4, e.g. `<|im_start|>`) registered as special so they encode
/// atomically. SentencePiece (`model == "llama"`) isn't built here — pass a `tokenizer.json`.
fn build_tokenizer(g: &Gguf) -> Result<Tokenizer> {
    let md = g.metadata();
    let model = md.str("tokenizer.ggml.model").unwrap_or("");
    if model != "gpt2" {
        bail!(
            "can't derive a tokenizer from tokenizer.ggml.model={model:?} \
             (only gpt2 byte-level BPE); pass a tokenizer.json sidecar instead"
        );
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
    let merges: Vec<(String, String)> = md
        .get("tokenizer.ggml.merges")
        .and_then(MetaValue::as_arr)
        .context("gguf missing tokenizer.ggml.merges")?
        .iter()
        .filter_map(|m| {
            let s = m.as_str()?;
            let mut it = s.splitn(2, ' ');
            Some((it.next()?.to_string(), it.next()?.to_string()))
        })
        .collect();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow!("build bpe: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    let add_prefix = matches!(
        md.get("tokenizer.ggml.add_space_prefix"),
        Some(MetaValue::Bool(true))
    );
    let pre = md.str("tokenizer.ggml.pre").unwrap_or("default");
    if pre == "qwen2" {
        // Sequence[ Split(qwen regex, Isolated), ByteLevel(use_regex=false) ] — matches HF Qwen.
        let split = Split::new(
            SplitPattern::Regex(QWEN2_PRE_RE.to_string()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| anyhow!("split pretokenizer: {e}"))?;
        let seq = PreSequence::new(vec![
            PreTokenizerWrapper::Split(split),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        ]);
        tok.with_pre_tokenizer(Some(seq));
    } else {
        tok.with_pre_tokenizer(Some(ByteLevel::new(add_prefix, true, true)));
    }
    tok.with_decoder(Some(ByteLevelDecoder::default()));
    // Add control (type 3, e.g. <|im_end|>) as SPECIAL tokens and user-defined (type 4, e.g.
    // <think>) as NORMAL added tokens — matching HF. Both encode atomically, but only special ones
    // are dropped by `decode(.., skip_special=true)`; keeping <think>/</think> non-special means
    // the reasoning block stays visible (and markable) in the output.
    if let Some(types) = md
        .get("tokenizer.ggml.token_type")
        .and_then(MetaValue::as_arr)
    {
        let mut specials = Vec::new();
        let mut added = Vec::new();
        for (i, ty) in types.iter().enumerate() {
            let Some(s) = toks.get(i).and_then(MetaValue::as_str) else {
                continue;
            };
            match ty.as_u64() {
                Some(3) => specials.push(AddedToken::from(s.to_string(), true)),
                Some(4) => added.push(AddedToken::from(s.to_string(), false)),
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
    Ok(tok)
}

/// Dequantize a tensor's raw `bytes` of `dtype` into host f32. Handles plain floats
/// (F32/F16/BF16), affine quants (via [`dequant_unified`]), and codebook quants (via
/// [`dequant_codebook`]). The single host-side dequant entry point.
fn dequant_block(dtype: infr_core::DType, bytes: &[u8]) -> Result<Vec<f32>> {
    use infr_core::DType::*;
    Ok(match dtype {
        F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Bf16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        d if is_quant(d) => {
            let (qv, sc, mn) = dequant_unified(d, bytes);
            (0..qv.len())
                .map(|g| sc[g] * qv[g] as f32 + mn[g])
                .collect()
        }
        d if is_codebook_quant(d) => dequant_codebook(d, bytes),
        other => bail!("unsupported dtype {other:?} (host dequant wants F16/F32/BF16/quant)"),
    })
}

/// Load a named tensor and dequantize it to host f32, returning (data, shape in GGUF ne order
/// `[in, out]`). The host/CPU-side dequant path — it does NOT load the bulk projection weights for
/// the GPU (those upload quantized/f16 in-VRAM). It feeds: the host embedding gather, the CPU norm
/// + SSM recurrence math (qwen35), the `Q35_CPU=1` oracle, and serves as the f32 source we convert
/// into f16/bf16/quant GPU weights. Survives even with full GPU format coverage.
fn load_tensor_dequant(g: &Gguf, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let info = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor not found: {name}"))?
        .clone();
    let bytes = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
    let v = dequant_block(info.dtype, bytes).with_context(|| format!("tensor {name}"))?;
    Ok((v, info.shape))
}

/// Return a tensor's data as raw f16 bytes (little-endian u16). F16 tensors pass through with no
/// conversion (fast path for f16 GGUFs); F32 tensors are converted on the host.
fn f16_bytes(g: &Gguf, name: &str) -> Result<Vec<u8>> {
    let info = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor not found: {name}"))?
        .clone();
    let bytes = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
    match info.dtype {
        infr_core::DType::F16 => Ok(bytes.to_vec()),
        infr_core::DType::F32 => {
            let f16: Vec<u16> = bytemuck::cast_slice::<u8, f32>(bytes)
                .iter()
                .map(|&x| f32_to_f16_sat(x).to_bits())
                .collect();
            Ok(bytemuck::cast_slice(&f16).to_vec())
        }
        infr_core::DType::Bf16 => {
            // bf16 → f32 → f16 (bf16 is the top 16 bits of f32). f16 has MORE mantissa than bf16
            // (10 vs 7 bits) but a far smaller exponent range, so the only loss is overflow — which
            // saturates (see `f32_to_f16_sat`) instead of becoming inf. Values that fit (≈all) are
            // exact. A native-bf16 fused path (no clip at all) is a follow-on; the eager qwen35 path
            // already stores bf16 natively via `upload_weight_bf16`.
            let f16: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| {
                    let f = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                    f32_to_f16_sat(f).to_bits()
                })
                .collect();
            Ok(bytemuck::cast_slice(&f16).to_vec())
        }
        other => bail!("unsupported dtype {other:?} for {name} (bring-up wants F16/F32/BF16)"),
    }
}

/// f32 → f16, saturating to ±65504 (f16 max) instead of overflowing to ±inf. Preserves NaN. Used
/// when down-converting bf16/f32 weights for the f16 fused path so a large-magnitude weight clips
/// to the largest finite f16 rather than corrupting the matmul with inf/NaN.
fn f32_to_f16_sat(x: f32) -> half::f16 {
    const F16_MAX: f32 = 65504.0;
    if x.is_nan() {
        half::f16::NAN
    } else {
        half::f16::from_f32(x.clamp(-F16_MAX, F16_MAX))
    }
}

fn rdf16(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}
fn k4(j: usize, q: &[u8]) -> (u32, u32) {
    // get_scale_min_k4: extract 6-bit scale `d` and min `m` for sub-block j (0..8) from scales[12]
    if j < 4 {
        ((q[j] & 63) as u32, (q[j + 4] & 63) as u32)
    } else {
        (
            ((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)) as u32,
            ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as u32,
        )
    }
}

/// Dequant any supported quant into the UNIFIED form: per-element u8 index + per-element
/// (scale, min) such that `weight = scale*u8 + min` (filled in natural tensor order). Scale/min are
/// constant across each consecutive 16-element block, which the kernel exploits.
fn dequant_unified(dtype: infr_core::DType, bytes: &[u8]) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    use infr_core::DType::*;
    let (qpb, bpb) = match dtype {
        Q4_0 => (32, 18),
        Q4_1 => (32, 20),
        Q5_0 => (32, 22),
        Q5_1 => (32, 24),
        Q8_0 => (32, 34),
        Q2K => (256, 84),
        Q3K => (256, 110),
        Q4K => (256, 144),
        Q5K => (256, 176),
        Q6K => (256, 210),
        _ => unreachable!(),
    };
    let nblk = bytes.len() / bpb;
    let numel = nblk * qpb;
    let mut qv = vec![0u8; numel];
    let mut sc = vec![0f32; numel];
    let mut mn = vec![0f32; numel];
    let mut set = |g: usize, q: u8, s: f32, m: f32| {
        qv[g] = q;
        sc[g] = s;
        mn[g] = m;
    };
    for b in 0..nblk {
        let blk = &bytes[b * bpb..(b + 1) * bpb];
        match dtype {
            // ── Q4_0: y = d*(q4 - 8), q4 ∈ 0..15 ──────────────────────────────
            // Ref: llama.cpp dequantize_row_q4_0 (ggml-quants.c l.401)
            // Block: [half d][uint8 qs[16]]
            // Unified: scale=d, index=q4 (0..15), min=-8*d
            Q4_0 => {
                let d = rdf16(blk);
                let min = -8.0 * d;
                let qs = &blk[2..18];
                for j in 0..16 {
                    set(b * 32 + j, qs[j] & 0x0F, d, min);
                    set(b * 32 + j + 16, qs[j] >> 4, d, min);
                }
            }
            // ── Q4_1: y = d*q4 + m, q4 ∈ 0..15 ─────────────────────────────────
            // Ref: llama.cpp dequantize_row_q4_1 (ggml-quants.c l.421)
            // Block: [half d][half m][uint8 qs[16]]
            // Unified: scale=d, index=q4 (0..15), min=m
            Q4_1 => {
                let d = rdf16(blk);
                let m = rdf16(&blk[2..4]);
                let qs = &blk[4..20];
                for j in 0..16 {
                    set(b * 32 + j, qs[j] & 0x0F, d, m);
                    set(b * 32 + j + 16, qs[j] >> 4, d, m);
                }
            }
            // ── Q5_0: y = d*(q5 - 16), q5 ∈ 0..31 ──────────────────────────────
            // Ref: llama.cpp dequantize_row_q5_0 (ggml-quants.c l.442)
            // Block: [half d][uint8 qh[4]][uint8 qs[16]]
            // Unified: scale=d, index=q5 (0..31), min=-16*d
            Q5_0 => {
                let d = rdf16(blk);
                let min = -16.0 * d;
                let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
                let qs = &blk[6..22];
                for j in 0..16 {
                    let xh0 = ((qh >> j) << 4) & 0x10;
                    let xh1 = (qh >> (j + 12)) & 0x10;
                    let q0 = (qs[j] as u32 & 0x0F) | xh0;
                    let q1 = (qs[j] as u32 >> 4) | xh1;
                    set(b * 32 + j, q0 as u8, d, min);
                    set(b * 32 + j + 16, q1 as u8, d, min);
                }
            }
            // ── Q5_1: y = d*q5 + m, q5 ∈ 0..31 ─────────────────────────────────
            // Ref: llama.cpp dequantize_row_q5_1 (ggml-quants.c l.468)
            // Block: [half d][half m][uint8 qh[4]][uint8 qs[16]]
            // Unified: scale=d, index=q5 (0..31), min=m
            Q5_1 => {
                let d = rdf16(blk);
                let m = rdf16(&blk[2..4]);
                let qh = u32::from_le_bytes(blk[4..8].try_into().unwrap());
                let qs = &blk[8..24];
                for j in 0..16 {
                    let xh0 = ((qh >> j) << 4) & 0x10;
                    let xh1 = (qh >> (j + 12)) & 0x10;
                    let q0 = (qs[j] as u32 & 0x0F) | xh0;
                    let q1 = (qs[j] as u32 >> 4) | xh1;
                    set(b * 32 + j, q0 as u8, d, m);
                    set(b * 32 + j + 16, q1 as u8, d, m);
                }
            }
            Q8_0 => {
                let d = rdf16(blk);
                for i in 0..32 {
                    set(
                        b * 32 + i,
                        (blk[2 + i] as i8 as i16 + 128) as u8,
                        d,
                        -128.0 * d,
                    );
                }
            }
            // ── Q2_K: y = dl*(q2) - ml; per-16-elem sub-block scale/min ─────────
            // Ref: llama.cpp dequantize_row_q2_K (ggml-quants.c l.903)
            // Block (84 bytes): [uint8 scales[16]][uint8 qs[64]][half d][half dmin]
            // x = d*(sc&0xF)*q2 - dmin*(sc>>4) → scale=d*(sc&0xF), index=q2, min=-dmin*(sc>>4)
            Q2K => {
                // Memory layout: scales[0..16], qs[16..80], d[80..82], dmin[82..84]
                let scales = &blk[0..16];
                let qs = &blk[16..80];
                let d = rdf16(&blk[80..82]);
                let dmin = rdf16(&blk[82..84]);
                let base = b * 256;
                let mut out = 0usize;
                let mut is = 0usize; // scale index
                let mut qoff = 0usize; // offset into qs
                for _n in 0..2 {
                    // n=0: elements 0..127, n=1: elements 128..255
                    let mut shift = 0u32;
                    for _j in 0..4 {
                        let sc = scales[is];
                        is += 1;
                        let dl = d * (sc & 0xF) as f32;
                        let ml = dmin * (sc >> 4) as f32;
                        for l in 0..16 {
                            let q2 = (qs[qoff + l] >> shift) & 3;
                            set(base + out, q2, dl, -ml);
                            out += 1;
                        }
                        let sc = scales[is];
                        is += 1;
                        let dl = d * (sc & 0xF) as f32;
                        let ml = dmin * (sc >> 4) as f32;
                        for l in 0..16 {
                            let q2 = (qs[qoff + l + 16] >> shift) & 3;
                            set(base + out, q2, dl, -ml);
                            out += 1;
                        }
                        shift += 2;
                    }
                    qoff += 32;
                }
            }
            // ── Q3_K: y = dl*(q3u - 4); per-16-elem 6-bit sub-block scale ────────
            // Ref: llama.cpp dequantize_row_q3_K (ggml-quants.c l.1247)
            // Block (110 bytes): [uint8 hmask[32]][uint8 qs[64]][uint8 scales[12]][half d]
            // q3u = (q_low2 | (hmask_bit << 2)) ∈ 0..7; y = d*(sc6-32)*q3u + d*(sc6-32)*(-4)
            Q3K => {
                // Memory layout: hmask[0..32], qs[32..96], scales[96..108], d[108..110]
                let hmask = &blk[0..32];
                let qs = &blk[32..96];
                let scales_raw = &blk[96..108];
                let d_all = rdf16(&blk[108..110]);
                let base = b * 256;

                // Decode 6-bit scales: port of the llama.cpp bit manipulation exactly.
                // scales_raw is 12 bytes encoding 16 × 6-bit values.
                // kmask1=0x03030303, kmask2=0x0f0f0f0f
                let mut aux = [0u32; 4];
                aux[0] = u32::from_le_bytes(scales_raw[0..4].try_into().unwrap());
                aux[1] = u32::from_le_bytes(scales_raw[4..8].try_into().unwrap());
                aux[2] = u32::from_le_bytes(scales_raw[8..12].try_into().unwrap());
                // Note: aux[3] starts as 0 (no 4th word in scales_raw)
                let kmask1: u32 = 0x0303_0303;
                let kmask2: u32 = 0x0f0f_0f0f;
                let tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                // Now aux as i8[16] gives decoded 6-bit scales; subtract 32 to get signed scale.
                let sc6 = |i: usize| -> f32 {
                    let byte_idx = i / 4;
                    let bit_shift = (i % 4) * 8;
                    let sc = ((aux[byte_idx] >> bit_shift) & 0xFF) as u8 as i8;
                    (sc as i32 - 32) as f32
                };

                let mut out = 0usize;
                let mut is = 0usize; // scale index
                let mut qoff = 0usize; // offset into qs
                let mut m = 1u8;
                for _n in 0..2 {
                    let mut shift = 0u32;
                    for _j in 0..4 {
                        let dl = d_all * sc6(is);
                        is += 1;
                        for l in 0..16 {
                            let low2 = (qs[qoff + l] >> shift) & 3;
                            let high = if hmask[l] & m != 0 { 1u8 } else { 0u8 };
                            let q3u = low2 | (high << 2); // 0..7
                            set(base + out, q3u, dl, -4.0 * dl);
                            out += 1;
                        }
                        let dl = d_all * sc6(is);
                        is += 1;
                        for l in 0..16 {
                            let low2 = (qs[qoff + l + 16] >> shift) & 3;
                            let high = if hmask[l + 16] & m != 0 { 1u8 } else { 0u8 };
                            let q3u = low2 | (high << 2); // 0..7
                            set(base + out, q3u, dl, -4.0 * dl);
                            out += 1;
                        }
                        shift += 2;
                        m <<= 1;
                    }
                    qoff += 32;
                }
            }
            Q4K => {
                let d = rdf16(&blk[0..2]);
                let dmin = rdf16(&blk[2..4]);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                let base = b * 256;
                for j in 0..4 {
                    let (sc1, m1) = k4(2 * j, scales);
                    let (sc2, m2) = k4(2 * j + 1, scales);
                    let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
                    let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
                    for l in 0..32 {
                        let v = qs[j * 32 + l];
                        set(base + j * 64 + l, v & 0xF, d1, -mm1);
                        set(base + j * 64 + 32 + l, v >> 4, d2, -mm2);
                    }
                }
            }
            Q5K => {
                let d = rdf16(&blk[0..2]);
                let dmin = rdf16(&blk[2..4]);
                let scales = &blk[4..16];
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                let base = b * 256;
                let (mut u1, mut u2) = (1u8, 2u8);
                for j in 0..4 {
                    let (sc1, m1) = k4(2 * j, scales);
                    let (sc2, m2) = k4(2 * j + 1, scales);
                    let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
                    let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
                    for l in 0..32 {
                        let v = qs[j * 32 + l];
                        let lo = (v & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 };
                        let hi = (v >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
                        set(base + j * 64 + l, lo, d1, -mm1);
                        set(base + j * 64 + 32 + l, hi, d2, -mm2);
                    }
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            Q6K => {
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let scales = &blk[192..208]; // 16 × int8
                let d = rdf16(&blk[208..210]);
                let sc_i8 = |i: usize| scales[i] as i8 as f32;
                for half in 0..2 {
                    let (qlo, qho, sco, base) =
                        (half * 64, half * 32, half * 8, b * 256 + half * 128);
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = (ql[qlo + l] & 0xF) | ((qh[qho + l] & 3) << 4);
                        let q2 = (ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4);
                        let q3 = (ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4);
                        let q4 = (ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4);
                        for (off, q, sci) in [(0, q1, 0), (32, q2, 2), (64, q3, 4), (96, q4, 6)] {
                            let s = d * sc_i8(sco + is + sci);
                            set(base + l + off, q, s, -32.0 * s); // y = s*(q-32)
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    (qv, sc, mn)
}

// ─── codebook (non-affine) dequant ───────────────────────────────────────────

/// IQ1_S / IQ1_M delta offset applied to each grid element.
/// Ref: llama.cpp ggml-common.h l.1121: `#define IQ1S_DELTA 0.125f`
const IQ1S_DELTA: f32 = 0.125;

/// MXFP4 / NVFP4 signed 4-bit codebook (E2M1 format × 2).
/// Ref: llama.cpp ggml-common.h l.1116: `kvalues_mxfp4`
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Decode an E8M0 exponent byte to float32, halved (= 2^(x-128)).
/// Matches llama.cpp `ggml_e8m0_to_fp32_half` (ggml-impl.h l.477).
#[inline(always)]
fn e8m0_to_fp32_half(x: u8) -> f32 {
    if x < 2 {
        f32::from_bits(0x0020_0000u32 << x)
    } else {
        f32::from_bits(((x as u32) - 1) << 23)
    }
}

/// Decode a UE4M3 byte (unsigned, 4 exp bits bias=7, 3 mantissa bits) to float32, halved.
/// Matches llama.cpp `ggml_ue4m3_to_fp32` (ggml-impl.h l.502).
#[inline(always)]
fn ue4m3_to_fp32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as f32;
    let raw = if exp == 0 {
        man * f32::powi(2.0, -9)
    } else {
        (1.0 + man / 8.0) * f32::powi(2.0, exp - 7)
    };
    raw * 0.5
}

/// IQ4_NL / IQ4_XS 16-entry signed-integer codebook.
/// Ref: llama.cpp ggml-common.h `kvalues_iq4nl` (l.1110)
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// True for codebook quants (IQ*/TQ*/fp4) that go host-dequant → f16, NOT the GPU affine path.
fn is_codebook_quant(d: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(
        d,
        Iq1S | Iq1M
            | Iq2Xxs
            | Iq2Xs
            | Iq2S
            | Iq3Xxs
            | Iq3S
            | Iq4Nl
            | Iq4Xs
            | Tq1_0
            | Tq2_0
            | Mxfp4
            | Nvfp4
    )
}

/// Dequantize a codebook (non-affine) quant to f32. Ported from llama.cpp `ggml-quants.c`.
/// Returns a `Vec<f32>` of length `numel` in natural tensor order.
fn dequant_codebook(dtype: infr_core::DType, bytes: &[u8]) -> Vec<f32> {
    use infr_core::DType::*;
    match dtype {
        // ── IQ4_NL: y = d * kvalues_iq4nl[q4], QK4_NL=32 ───────────────────────
        // Block: [half d][uint8 qs[16]], 18 bytes
        // Ref: llama.cpp dequantize_row_iq4_nl (ggml-quants.c l.2653)
        Iq4Nl => {
            // Ref: llama.cpp dequantize_row_iq4_nl (ggml-quants.c l.2653)
            // y[j]    = d * kv[qs[j] & 0xF]  for j in 0..16 → elements  0..15
            // y[j+16] = d * kv[qs[j] >>  4]  for j in 0..16 → elements 16..31
            let bpb = 18usize;
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 32];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let qs = &blk[2..18];
                let base = b * 32;
                for j in 0..16 {
                    out[base + j] = d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
                    out[base + j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                }
            }
            out
        }
        // ── IQ4_XS: y = d*(ls-32) * kvalues_iq4nl[q4], QK_K=256 ────────────────
        // Block: [half d][uint16 scales_h][uint8 scales_l[4]][uint8 qs[128]], 136 bytes
        // Ref: llama.cpp dequantize_row_iq4_xs (ggml-quants.c l.2671)
        Iq4Xs => {
            // Ref: llama.cpp dequantize_row_iq4_xs (ggml-quants.c l.2671)
            // Block: [half d][uint16 scales_h][uint8 scales_l[4]][uint8 qs[128]], 136 bytes
            // 8 sub-blocks of 32 elements each; ls = 6-bit scale per sub-block
            // y[j+0]  = dl * kv[qs[j] & 0xF] for j in 0..16 → elements  0..15 of sub-block
            // y[j+16] = dl * kv[qs[j] >>  4] for j in 0..16 → elements 16..31 of sub-block
            let bpb = 136usize;
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let scales_h = u16::from_le_bytes(blk[2..4].try_into().unwrap());
                let scales_l = &blk[4..8];
                let qs = &blk[8..136];
                let base = b * 256;
                let mut qoff = 0usize;
                let mut outoff = 0usize;
                for ib in 0..8usize {
                    // 6-bit ls: lower 4 bits from scales_l, upper 2 bits from scales_h
                    let lo = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as u32;
                    let hi = ((scales_h >> (2 * ib)) & 3) as u32;
                    let ls = lo | (hi << 4);
                    let dl = d * (ls as i32 - 32) as f32;
                    for j in 0..16 {
                        out[base + outoff + j] =
                            dl * KVALUES_IQ4NL[(qs[qoff + j] & 0xF) as usize] as f32;
                        out[base + outoff + j + 16] =
                            dl * KVALUES_IQ4NL[(qs[qoff + j] >> 4) as usize] as f32;
                    }
                    qoff += 16;
                    outoff += 32;
                }
            }
            out
        }
        // ── IQ2_XXS: block = [half d][uint16 qs[32]], 66 bytes, QK_K=256 ────────
        // Each super-block has 8 sub-blocks of 32 elements.  For each sub-block:
        //   aux32[0] = 4 grid indices (one byte each, into iq2xxs_grid[256])
        //   aux32[1] = sign pack: bits[6:0]*4 = four 7-bit sign indices + bits[31:28] = scale
        //   db = d * (0.5 + (aux32[1] >> 28)) * 0.25
        //   grid[j] = ((iq2xxs_grid[idx] >> (8*j)) & 0xFF) as i8  (for j in 0..8)
        //   y[j] = db * grid[j] * (if ksigns[sign_idx] & (1<<j) { -1 } else { 1 })
        // Ref: llama.cpp dequantize_row_iq2_xxs (ggml-quants.c l.2416)
        Iq2Xxs => {
            use infr_core::iquant_grids::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
            let bpb = 66usize; // 2 + 32*2
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                // qs as raw bytes (64 bytes = 32 uint16s)
                let qs = &blk[2..66];
                let base = b * 256;
                let mut outoff = 0usize;
                for ib32 in 0..8usize {
                    // read 8 bytes at offset 8*ib32 within qs
                    let off = ib32 * 8;
                    let aux0 = u32::from_le_bytes(qs[off..off + 4].try_into().unwrap());
                    let aux1 = u32::from_le_bytes(qs[off + 4..off + 8].try_into().unwrap());
                    let scale_mag = aux1 >> 28;
                    let db = d * (0.5 + scale_mag as f32) * 0.25;
                    let aux0_bytes = aux0.to_le_bytes();
                    for l in 0..4usize {
                        let grid_idx = aux0_bytes[l] as usize;
                        let sign_idx = ((aux1 >> (7 * l)) & 127) as usize;
                        let grid_u64 = IQ2XXS_GRID[grid_idx];
                        let signs = KSIGNS_IQ2XS[sign_idx];
                        for j in 0..8usize {
                            let gv = ((grid_u64 >> (8 * j)) & 0xFF) as i8;
                            let sign = if signs & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            out[base + outoff + j] = db * gv as f32 * sign;
                        }
                        outoff += 8;
                    }
                }
            }
            out
        }
        // ── IQ2_XS: block = [half d][uint16 qs[32]][uint8 scales[8]], 74 bytes ──
        // 8 sub-blocks of 32 elements each (4 groups of 8 per sub-block).
        //   db[0] = d * (0.5 + (scales[ib32] & 0xf)) * 0.25
        //   db[1] = d * (0.5 + (scales[ib32] >> 4)) * 0.25
        //   For l in 0..4: grid = iq2xs_grid[qs16[l] & 511]  (9-bit index)
        //                  signs = ksigns[qs16[l] >> 9]        (7-bit sign)
        //                  dl = db[l/2]
        // Ref: llama.cpp dequantize_row_iq2_xs (ggml-quants.c l.2444)
        Iq2Xs => {
            use infr_core::iquant_grids::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
            let bpb = 74usize; // 2 + 32*2 + 8
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                // qs as uint16 array (64 bytes = 32 entries)
                let qs_raw = &blk[2..66];
                let scales = &blk[66..74];
                let base = b * 256;
                let mut outoff = 0usize;
                for ib32 in 0..8usize {
                    let sc = scales[ib32];
                    let db0 = d * (0.5 + (sc & 0xf) as f32) * 0.25;
                    let db1 = d * (0.5 + (sc >> 4) as f32) * 0.25;
                    for l in 0..4usize {
                        let qoff = (ib32 * 4 + l) * 2;
                        let qs16 = u16::from_le_bytes(qs_raw[qoff..qoff + 2].try_into().unwrap());
                        let grid_idx = (qs16 & 511) as usize;
                        let sign_idx = (qs16 >> 9) as usize;
                        let grid_u64 = IQ2XS_GRID[grid_idx];
                        let signs = KSIGNS_IQ2XS[sign_idx];
                        let dl = if l < 2 { db0 } else { db1 };
                        for j in 0..8usize {
                            let gv = ((grid_u64 >> (8 * j)) & 0xFF) as i8;
                            let sign = if signs & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            out[base + outoff + j] = dl * gv as f32 * sign;
                        }
                        outoff += 8;
                    }
                }
            }
            out
        }
        // ── IQ2_S: block = [half d][u8 qs[64]][u8 qh[8]][u8 scales[8]], 82 bytes ─
        // First 32 bytes of qs = 8-bit grid indices (low bits);
        // next 32 bytes = per-group sign bytes. qh = 2-bit high bits per entry.
        //   grid_idx = qs[l] | ((qh[ib32] << (8-2*l)) & 0x300)  (10-bit → iq2s_grid[1024])
        // Ref: llama.cpp dequantize_row_iq2_s (ggml-quants.c l.2471)
        Iq2S => {
            use infr_core::iquant_grids::{IQ2S_GRID, KMASK_IQ2XS};
            let bpb = 82usize; // 2 + 64 + 8 + 8
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let qs_all = &blk[2..66]; // 64 bytes
                let qh = &blk[66..74]; // 8 bytes
                let scales = &blk[74..82]; // 8 bytes
                let base = b * 256;
                let mut outoff = 0usize;
                for ib32 in 0..8usize {
                    let sc = scales[ib32];
                    let db0 = d * (0.5 + (sc & 0xf) as f32) * 0.25;
                    let db1 = d * (0.5 + (sc >> 4) as f32) * 0.25;
                    let qh_byte = qh[ib32];
                    for l in 0..4usize {
                        let qs_idx = ib32 * 4 + l;
                        let sgn_idx = ib32 * 4 + l + 32; // signs start at qs_all[32]
                        let qs_byte = qs_all[qs_idx];
                        let sign_byte = qs_all[sgn_idx];
                        // high 2 bits: shift = 8 - 2*l; mask 0x300
                        let hi = ((qh_byte as u32).wrapping_shl((8 - 2 * l) as u32)) & 0x300;
                        let grid_idx = (qs_byte as usize) | (hi as usize);
                        let grid_u64 = IQ2S_GRID[grid_idx];
                        let dl = if l < 2 { db0 } else { db1 };
                        for j in 0..8usize {
                            let gv = ((grid_u64 >> (8 * j)) & 0xFF) as i8;
                            let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            out[base + outoff + j] = dl * gv as f32 * sign;
                        }
                        outoff += 8;
                    }
                }
            }
            out
        }
        // ── IQ3_XXS: block = [half d][u8 qs[96]], 98 bytes, QK_K=256 ─────────────
        // qs[0..64] = 8-bit indices (two per group of 8: grid1, grid2)
        // qs[64..96] = scales_and_signs: 4 bytes per sub-block (aux32)
        //   db = d * (0.5 + (aux32 >> 28)) * 0.5
        //   For l in 0..4: signs = ksigns[(aux32 >> 7*l) & 127]
        //     grid1 = iq3xxs_grid[qs[2*l+0]];  grid2 = iq3xxs_grid[qs[2*l+1]]
        //     y[j+0] = db * grid1[j] * sign;  y[j+4] = db * grid2[j] * sign
        // Ref: llama.cpp dequantize_row_iq3_xxs (ggml-quants.c l.2503)
        Iq3Xxs => {
            use infr_core::iquant_grids::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
            let bpb = 98usize; // 2 + 96
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let qs = &blk[2..66]; // first 64 bytes = grid indices
                let sas = &blk[66..98]; // scales_and_signs (32 bytes = 8 × u32)
                let base = b * 256;
                let mut outoff = 0usize;
                let mut qs_off = 0usize;
                for ib32 in 0..8usize {
                    let aux32 = u32::from_le_bytes(sas[4 * ib32..4 * ib32 + 4].try_into().unwrap());
                    let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                    for l in 0..4usize {
                        let sign_idx = ((aux32 >> (7 * l)) & 127) as usize;
                        let signs = KSIGNS_IQ2XS[sign_idx];
                        let g1 = IQ3XXS_GRID[qs[qs_off + 2 * l] as usize];
                        let g2 = IQ3XXS_GRID[qs[qs_off + 2 * l + 1] as usize];
                        for j in 0..4usize {
                            let s1 = if signs & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            let s2 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            let gv1 = ((g1 >> (8 * j)) & 0xFF) as i8;
                            let gv2 = ((g2 >> (8 * j)) & 0xFF) as i8;
                            out[base + outoff + j] = db * gv1 as f32 * s1;
                            out[base + outoff + j + 4] = db * gv2 as f32 * s2;
                        }
                        outoff += 8;
                    }
                    qs_off += 8;
                }
            }
            out
        }
        // ── IQ3_S: block = [half d][u8 qs[64]][u8 qh[8]][u8 signs[32]][u8 scales[4]], 110 bytes
        // Outer loop steps ib32 in 0,2,4,6 (pairs → two groups of 32 each = 64 elements/iter).
        //   db1 = d * (1 + 2*(scales[ib32/2] & 0xf))
        //   db2 = d * (1 + 2*(scales[ib32/2] >> 4))
        //   For first group (l in 0..4, using qh[0], db1):
        //     grid1 = iq3s_grid[qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)]
        //     grid2 = iq3s_grid[qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)]
        //   For second group (l in 0..4, using qh[1], db2): similarly
        // Ref: llama.cpp dequantize_row_iq3_s (ggml-quants.c l.2535)
        Iq3S => {
            use infr_core::iquant_grids::{IQ3S_GRID, KMASK_IQ2XS};
            let bpb = 110usize; // 2 + 64 + 8 + 32 + 4
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let qs_arr = &blk[2..66]; // 64 bytes
                let qh_arr = &blk[66..74]; // 8 bytes
                let signs_arr = &blk[74..106]; // 32 bytes
                let scales = &blk[106..110]; // 4 bytes
                let base = b * 256;
                let mut outoff = 0usize;
                let mut qs_off = 0usize;
                let mut signs_off = 0usize;
                let mut qh_off = 0usize;
                // outer loop: ib32 steps 0,2,4,6 → 4 pairs
                for pair in 0..4usize {
                    let db1 = d * (1.0 + 2.0 * (scales[pair] & 0xf) as f32);
                    let db2 = d * (1.0 + 2.0 * (scales[pair] >> 4) as f32);
                    // first group of 32 elements (using qh[qh_off], db1)
                    let qh0 = qh_arr[qh_off];
                    for l in 0..4usize {
                        let g1_idx = qs_arr[qs_off + 2 * l] as usize
                            | (((qh0 as u32).wrapping_shl((8 - 2 * l) as u32)) & 256) as usize;
                        let g2_idx = qs_arr[qs_off + 2 * l + 1] as usize
                            | (((qh0 as u32).wrapping_shl((7 - 2 * l) as u32)) & 256) as usize;
                        let g1 = IQ3S_GRID[g1_idx];
                        let g2 = IQ3S_GRID[g2_idx];
                        let sb = signs_arr[signs_off + l];
                        for j in 0..4usize {
                            let s1 = if sb & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            let s2 = if sb & KMASK_IQ2XS[j + 4] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            out[base + outoff + j] =
                                db1 * ((g1 >> (8 * j)) & 0xFF) as i8 as f32 * s1;
                            out[base + outoff + j + 4] =
                                db1 * ((g2 >> (8 * j)) & 0xFF) as i8 as f32 * s2;
                        }
                        outoff += 8;
                    }
                    qs_off += 8;
                    signs_off += 4;
                    // second group of 32 elements (using qh[qh_off+1], db2)
                    let qh1 = qh_arr[qh_off + 1];
                    for l in 0..4usize {
                        let g1_idx = qs_arr[qs_off + 2 * l] as usize
                            | (((qh1 as u32).wrapping_shl((8 - 2 * l) as u32)) & 256) as usize;
                        let g2_idx = qs_arr[qs_off + 2 * l + 1] as usize
                            | (((qh1 as u32).wrapping_shl((7 - 2 * l) as u32)) & 256) as usize;
                        let g1 = IQ3S_GRID[g1_idx];
                        let g2 = IQ3S_GRID[g2_idx];
                        let sb = signs_arr[signs_off + l];
                        for j in 0..4usize {
                            let s1 = if sb & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            let s2 = if sb & KMASK_IQ2XS[j + 4] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            out[base + outoff + j] =
                                db2 * ((g1 >> (8 * j)) & 0xFF) as i8 as f32 * s1;
                            out[base + outoff + j + 4] =
                                db2 * ((g2 >> (8 * j)) & 0xFF) as i8 as f32 * s2;
                        }
                        outoff += 8;
                    }
                    qh_off += 2;
                    qs_off += 8;
                    signs_off += 4;
                }
            }
            out
        }
        // ── IQ1_S: block = [half d][u8 qs[32]][u16 qh[8]], 50 bytes, QK_K=256 ────
        // 8 sub-blocks of 32 elements (4 groups of 8 each).
        //   dl = d * (2*((qh[ib] >> 12) & 7) + 1)
        //   delta = if qh[ib] & 0x8000 { -IQ1S_DELTA } else { IQ1S_DELTA }
        //   For l in 0..4: grid_idx = qs[l] | (((qh[ib] >> 3*l) & 7) << 8)
        //     y[j] = dl * (grid[j] as f32 + delta)
        // Ref: llama.cpp dequantize_row_iq1_s (ggml-quants.c l.2578)
        Iq1S => {
            use infr_core::iquant_grids::IQ1S_GRID;
            let bpb = 50usize; // 2 + 32 + 16
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let qs = &blk[2..34]; // 32 bytes
                let qh_raw = &blk[34..50]; // 16 bytes = 8 × u16
                let base = b * 256;
                let mut outoff = 0usize;
                let mut qs_off = 0usize;
                for ib in 0..8usize {
                    let qh = u16::from_le_bytes(qh_raw[2 * ib..2 * ib + 2].try_into().unwrap());
                    let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
                    let delta = if qh & 0x8000 != 0 {
                        -IQ1S_DELTA
                    } else {
                        IQ1S_DELTA
                    };
                    for l in 0..4usize {
                        let grid_idx =
                            qs[qs_off + l] as usize | (((qh >> (3 * l)) & 7) as usize) << 8;
                        let grid_u64 = IQ1S_GRID[grid_idx];
                        for j in 0..8usize {
                            let gv = ((grid_u64 >> (8 * j)) & 0xFF) as i8 as f32;
                            out[base + outoff + j] = dl * (gv + delta);
                        }
                        outoff += 8;
                    }
                    qs_off += 4;
                }
            }
            out
        }
        // ── IQ1_M: block = [u8 qs[32]][u8 qh[16]][u8 scales[8]], 56 bytes, QK_K=256
        // No separate `d` field — d is a f16 packed into the high 4 bits of each u16 of scales.
        //   sc[i] = scales reinterpreted as u16[4]; d extracted via:
        //   scale.u16 = (sc[0]>>12) | ((sc[1]>>8)&0xf0) | ((sc[2]>>4)&0xf00) | (sc[3]&0xf000)
        //   dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1)
        //   dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1)
        //   idx[0..4] from qs[0..3] and qh[0..1]:
        //     idx[0] = qs[0] | ((qh[0] << 8) & 0x700);  delta[0] = qh[0]&0x08 ? neg : pos
        //     idx[1] = qs[1] | ((qh[0] << 4) & 0x700);  delta[1] = qh[0]&0x80 ? neg : pos
        //     idx[2] = qs[2] | ((qh[1] << 8) & 0x700);  delta[2] = qh[1]&0x08 ? neg : pos
        //     idx[3] = qs[3] | ((qh[1] << 4) & 0x700);  delta[3] = qh[1]&0x80 ? neg : pos
        //   l=0,1: y[j] = dl1*(grid[j]+delta[l]); l=2,3: y[j] = dl2*(grid[j]+delta[l])
        // Ref: llama.cpp dequantize_row_iq1_m (ggml-quants.c l.2603)
        Iq1M => {
            use infr_core::iquant_grids::IQ1S_GRID;
            let bpb = 56usize; // 32 + 16 + 8
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let qs_arr = &blk[0..32]; // 32 bytes
                let qh_arr = &blk[32..48]; // 16 bytes
                let scales_raw = &blk[48..56]; // 8 bytes = 4 × u16
                                               // Extract d from high-nibble packing of the 4 scale u16s
                let sc0 = u16::from_le_bytes(scales_raw[0..2].try_into().unwrap());
                let sc1 = u16::from_le_bytes(scales_raw[2..4].try_into().unwrap());
                let sc2 = u16::from_le_bytes(scales_raw[4..6].try_into().unwrap());
                let sc3 = u16::from_le_bytes(scales_raw[6..8].try_into().unwrap());
                let sc = [sc0, sc1, sc2, sc3];
                let d_bits: u16 =
                    (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
                let d = half::f16::from_bits(d_bits).to_f32();
                let base = b * 256;
                let mut outoff = 0usize;
                let mut qs_off = 0usize;
                let mut qh_off = 0usize;
                for ib in 0..8usize {
                    let dl1 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) as f32 + 1.0);
                    let dl2 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) as f32 + 1.0);
                    let qh0 = qh_arr[qh_off];
                    let qh1 = qh_arr[qh_off + 1];
                    let idx = [
                        qs_arr[qs_off + 0] as usize | (((qh0 as usize) << 8) & 0x700),
                        qs_arr[qs_off + 1] as usize | (((qh0 as usize) << 4) & 0x700),
                        qs_arr[qs_off + 2] as usize | (((qh1 as usize) << 8) & 0x700),
                        qs_arr[qs_off + 3] as usize | (((qh1 as usize) << 4) & 0x700),
                    ];
                    let delta = [
                        if qh0 & 0x08 != 0 {
                            -IQ1S_DELTA
                        } else {
                            IQ1S_DELTA
                        },
                        if qh0 & 0x80 != 0 {
                            -IQ1S_DELTA
                        } else {
                            IQ1S_DELTA
                        },
                        if qh1 & 0x08 != 0 {
                            -IQ1S_DELTA
                        } else {
                            IQ1S_DELTA
                        },
                        if qh1 & 0x80 != 0 {
                            -IQ1S_DELTA
                        } else {
                            IQ1S_DELTA
                        },
                    ];
                    // l=0,1: use dl1; l=2,3: use dl2
                    for l in 0..2usize {
                        let grid_u64 = IQ1S_GRID[idx[l]];
                        for j in 0..8usize {
                            let gv = ((grid_u64 >> (8 * j)) & 0xFF) as i8 as f32;
                            out[base + outoff + j] = dl1 * (gv + delta[l]);
                        }
                        outoff += 8;
                    }
                    for l in 2..4usize {
                        let grid_u64 = IQ1S_GRID[idx[l]];
                        for j in 0..8usize {
                            let gv = ((grid_u64 >> (8 * j)) & 0xFF) as i8 as f32;
                            out[base + outoff + j] = dl2 * (gv + delta[l]);
                        }
                        outoff += 8;
                    }
                    qs_off += 4;
                    qh_off += 2;
                }
            }
            out
        }
        // ── TQ1_0: block = [u8 qs[48]][u8 qh[4]][half d], 54 bytes, QK_K=256 ────
        // 5-ternary-digits-per-byte encoding.
        //   Main loop: j=0..31 (32 bytes), 5 passes → 32*5=160 elements
        //   Second:   j=32..47 (16 bytes), 5 passes → 16*5=80 elements
        //   qh loop:  4 bytes, 4 passes →  4*4=16 elements
        //   Total 256.
        //   digit_n(b) = ((b * pow3[n] as u16) * 3 >> 8) as i16 → 0,1, or 2
        //   y = (digit - 1) * d
        // Ref: llama.cpp dequantize_row_tq1_0 (ggml-quants.c l.2356)
        Tq1_0 => {
            let bpb = 54usize; // 48 + 4 + 2
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let qs = &blk[0..48];
                let qh = &blk[48..52];
                let d = rdf16(&blk[52..54]);
                let base = b * 256;
                let mut outoff = 0usize;
                // qs[0..32]: 32 bytes, 5 digit passes
                for n in 0..5usize {
                    let p3 = POW3[n];
                    for m in 0..32usize {
                        let q = qs[m].wrapping_mul(p3);
                        let xi = (((q as u16) * 3) >> 8) as i16;
                        out[base + outoff + n * 32 + m] = (xi - 1) as f32 * d;
                    }
                }
                outoff += 5 * 32; // 160
                                  // qs[32..48]: 16 bytes, 5 digit passes
                for n in 0..5usize {
                    let p3 = POW3[n];
                    for m in 0..16usize {
                        let q = qs[32 + m].wrapping_mul(p3);
                        let xi = (((q as u16) * 3) >> 8) as i16;
                        out[base + outoff + n * 16 + m] = (xi - 1) as f32 * d;
                    }
                }
                outoff += 5 * 16; // 80
                                  // qh[0..4]: 4 bytes, 4 digit passes
                for n in 0..4usize {
                    let p3 = POW3[n];
                    for j in 0..4usize {
                        let q = qh[j].wrapping_mul(p3);
                        let xi = (((q as u16) * 3) >> 8) as i16;
                        out[base + outoff + n * 4 + j] = (xi - 1) as f32 * d;
                    }
                }
                // outoff += 16 (unused but for clarity)
                let _ = outoff;
            }
            out
        }
        // ── TQ2_0: block = [u8 qs[64]][half d], 66 bytes, QK_K=256 ──────────────
        // 2 bits per element; 4 elements packed per byte; two 32-byte passes.
        //   For each 32-byte chunk j, for l in 0..4, for m in 0..32:
        //     q = (qs[j+m] >> (l*2)) & 3  ∈ {0,1,2,3}
        //     y = (q - 1) * d
        // Ref: llama.cpp dequantize_row_tq2_0 (ggml-quants.c l.2395)
        Tq2_0 => {
            let bpb = 66usize; // 64 + 2
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let qs = &blk[0..64];
                let d = rdf16(&blk[64..66]);
                let base = b * 256;
                let mut outoff = 0usize;
                // Two 32-byte chunks (j=0, j=32)
                for chunk in 0..2usize {
                    let j = chunk * 32;
                    for l in 0..4usize {
                        for m in 0..32usize {
                            let q = ((qs[j + m] >> (l * 2)) & 3) as i32;
                            out[base + outoff] = (q - 1) as f32 * d;
                            outoff += 1;
                        }
                    }
                }
            }
            out
        }
        // ── MXFP4: block = [u8 e][u8 qs[16]], 17 bytes, QK_MXFP4=32 ─────────────
        // E8M0 shared exponent + nibble-packed E2M1 4-bit values.
        //   d = e8m0_to_fp32_half(e) = 2^(e-128)
        //   x0 = kvalues_mxfp4[qs[j] & 0xF]; x1 = kvalues_mxfp4[qs[j] >> 4]
        //   y[j+0] = x0*d; y[j+16] = x1*d
        // Ref: llama.cpp dequantize_row_mxfp4 (ggml-quants.c l.511)
        Mxfp4 => {
            let bpb = 17usize; // 1 + 16
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 32];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = e8m0_to_fp32_half(blk[0]);
                let qs = &blk[1..17];
                let base = b * 32;
                for j in 0..16usize {
                    let x0 = KVALUES_MXFP4[(qs[j] & 0xF) as usize] as f32;
                    let x1 = KVALUES_MXFP4[(qs[j] >> 4) as usize] as f32;
                    out[base + j] = x0 * d;
                    out[base + j + 16] = x1 * d;
                }
            }
            out
        }
        // ── NVFP4: block = [u8 d[4]][u8 qs[32]], 36 bytes, QK_NVFP4=64 ──────────
        // 4 sub-blocks of 16 elements each; UE4M3 scale per sub-block.
        //   For s in 0..4: d = ue4m3_to_fp32(scales[s])
        //     For j in 0..7: v0 = kvalues_mxfp4[qs[s*8+j] & 0xF]; v1 = qs[s*8+j] >> 4
        //     yb[j] = v0*d; yb[j+8] = v1*d
        // Ref: llama.cpp dequantize_row_nvfp4 (ggml-quants.c l.531)
        Nvfp4 => {
            let bpb = 36usize; // 4 + 32
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 64];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let scales = &blk[0..4];
                let qs = &blk[4..36];
                let base = b * 64;
                for s in 0..4usize {
                    let d = ue4m3_to_fp32(scales[s]);
                    let ybase = base + s * 16;
                    for j in 0..8usize {
                        let v0 = KVALUES_MXFP4[(qs[s * 8 + j] & 0xF) as usize] as f32;
                        let v1 = KVALUES_MXFP4[(qs[s * 8 + j] >> 4) as usize] as f32;
                        out[ybase + j] = v0 * d;
                        out[ybase + j + 8] = v1 * d;
                    }
                }
            }
            out
        }
        other => unimplemented!("codebook dequant for {other:?} not yet implemented"),
    }
}

/// True for types that go through the GPU in-kernel affine dequant path (`Wt::Q`).
/// All affine quants: legacy round quants + k-quants (Q2K–Q6K).
/// Codebook quants (IQ*/TQ*/fp4) are NOT included — they go host-dequant → f16.
fn is_quant(d: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(
        d,
        Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q2K | Q3K | Q4K | Q5K | Q6K
    )
}

/// In-VRAM packing per source quant: (bits, scale/min block size). Q4 packs to native 4-bit (8×
/// smaller index than f16); Q5/Q6/Q8 keep 8-bit. Block size matches the quant's sub-block.
fn quant_params(d: infr_core::DType) -> (u32, usize) {
    use infr_core::DType::*;
    match d {
        Q4_0 => (4, 32), // 4-bit index (0..15), per-32-elem scale/min
        Q4_1 => (4, 32), // 4-bit index (0..15), per-32-elem scale/min
        Q5_0 => (8, 32), // 5-bit index (0..31) stored in 8-bit, per-32-elem scale/min
        Q5_1 => (8, 32), // 5-bit index (0..31) stored in 8-bit, per-32-elem scale/min
        Q8_0 => (8, 32), // 8-bit index, per-32-elem scale/min
        Q2K => (4, 16),  // 2-bit quant (0..3) in 4-bit packing; per-16-elem sub-scale/min
        Q3K => (4, 16),  // 3-bit quant (0..7) in 4-bit packing; per-16-elem sub-scale/min
        Q4K => (4, 32),
        Q5K => (8, 32),
        Q6K => (8, 16),
        _ => unreachable!(),
    }
}

/// Pack a unified-dequant result into the GPU layout: indices at `bits` (4 → 8/u32, else 4/u32),
/// scales/mins one f16 per `blk`-element block.
fn pack_unified(
    qv: &[u8],
    sc: &[f32],
    mn: &[f32],
    bits: u32,
    blk: usize,
) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    let numel = qv.len();
    let quants = if bits == 4 {
        let mut q = vec![0u32; numel / 8];
        for (g, &v) in qv.iter().enumerate() {
            q[g / 8] |= ((v & 0xF) as u32) << (4 * (g % 8));
        }
        q
    } else {
        let mut q = vec![0u32; numel / 4];
        for (g, &v) in qv.iter().enumerate() {
            q[g / 4] |= (v as u32) << (8 * (g % 4));
        }
        q
    };
    let scales: Vec<u16> = (0..numel / blk)
        .map(|b| half::f16::from_f32(sc[b * blk]).to_bits())
        .collect();
    let mins: Vec<u16> = (0..numel / blk)
        .map(|b| half::f16::from_f32(mn[b * blk]).to_bits())
        .collect();
    (quants, scales, mins)
}

/// True when `INFR_NATIVE=1` opts into the raw-block / in-shader-dequant path (`Wt::Native`).
///
/// Default is the unified-repack path (`Wt::Q`): it pays a one-time host dequant at load to get a
/// GPU-friendly layout, which is ~1.75× faster decode and ~3× faster prefill than re-extracting
/// native blocks every matmul (measured, qwen3-0.6b Q4_K_M on 7900 XTX). Decode speed is the north
/// star, so native is opt-in — for its load-time / native-VRAM wins, and as the basis for in-kernel
/// i-quant support (which the unified path can't do).
fn use_native() -> bool {
    std::env::var("INFR_NATIVE").is_ok()
}

/// True for quant types that have a native-block GEMV shader (affine + codebook formats with no
/// grid-table dependency). Grid-based i-quants (IQ2*/IQ3*/IQ1*) are not yet native — they stay on
/// the host-dequant → f16 path.
fn is_native_supported(d: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(
        d,
        Q8_0 | Q4_0
            | Q4_1
            | Q5_0
            | Q5_1
            | Q2K
            | Q3K
            | Q4K
            | Q5K
            | Q6K
            | Iq4Nl
            | Iq4Xs
            | Mxfp4
            | Nvfp4
            | Tq1_0
            | Tq2_0
            | Iq2Xxs
            | Iq2Xs
            | Iq2S
            | Iq3Xxs
            | Iq3S
            | Iq1S
            | Iq1M
    )
}

/// Upload a projection weight, keeping quantized weights quantized in-VRAM (else convert to f16).
///
/// Default path:
/// - Affine quants → `Wt::Q` (host dequant + repack, GPU in-kernel via `LINEAR_Q_WGSL`; fastest
///   decode/prefill — the repack cost is paid once at load)
/// - Codebook quants (IQ*/TQ*/fp4) → host dequant → f16 → `Wt::F16`
/// - Float types (F16/F32/BF16) → `Wt::F16` directly
///
/// Native path (`INFR_NATIVE=1`, supported affine quants only):
/// - → `Wt::Native` (raw block bytes, in-shader dequant, no host work — faster load / smaller VRAM,
///   slower per-token; see [`use_native`])
fn upload_wt(be: &VulkanBackend, g: &Gguf, name: &str) -> Result<Wt> {
    let info = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor not found: {name}"))?
        .clone();
    // Native-block path: raw upload + in-shader dequant (opt-in via INFR_NATIVE).
    if use_native() && is_native_supported(info.dtype) {
        let bytes = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
        let padded = infr_vulkan::linear::pad_to_u32_align(bytes);
        return Ok(Wt::Native {
            buf: be
                .upload_weight_bytes(&padded)
                .map_err(|e| anyhow!("native upload {name}: {e}"))?,
            dtype: info.dtype,
        });
    }
    if is_quant(info.dtype) {
        // Affine quants: GPU in-kernel dequant path (legacy)
        let bytes = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
        let (bits, blk) = quant_params(info.dtype);
        let (qv, sc, mn) = dequant_unified(info.dtype, bytes);
        let (q, s, m) = pack_unified(&qv, &sc, &mn, bits, blk);
        Ok(Wt::Q {
            q: be.upload_weight_bytes(bytemuck::cast_slice(&q))?,
            s: be.upload_weight_bytes(bytemuck::cast_slice(&s))?,
            m: be.upload_weight_bytes(bytemuck::cast_slice(&m))?,
            bits,
            blk_shift: blk.trailing_zeros(),
        })
    } else if is_codebook_quant(info.dtype) {
        // Codebook quants: host dequant to f32, then upload as f16
        let bytes = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
        let f32_vals = dequant_codebook(info.dtype, bytes);
        let f16_bytes: Vec<u8> = f32_vals
            .iter()
            .flat_map(|&v| f32_to_f16_sat(v).to_bits().to_le_bytes())
            .collect();
        Ok(Wt::F16(be.upload_weight_bytes(&f16_bytes)?))
    } else {
        Ok(Wt::F16(be.upload_weight_bytes(&f16_bytes(g, name)?)?))
    }
}

impl Llama {
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Load with an explicit HF `tokenizer.json` sidecar.
    pub fn load(gguf_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        Self::load_opt(gguf_path, Some(tokenizer_path))
    }

    /// Load deriving the tokenizer from the GGUF's embedded vocab (`tokenizer.ggml.*`) — no
    /// sidecar needed (e.g. for `ollama:` refs, whose content-addressed blobs have no
    /// `tokenizer.json` beside them).
    pub fn load_embedded(gguf_path: &Path) -> Result<Self> {
        Self::load_opt(gguf_path, None)
    }

    /// Load with an optional sidecar tokenizer; falls back to the GGUF's embedded vocab.
    pub fn load_opt(gguf_path: &Path, tokenizer_path: Option<&Path>) -> Result<Self> {
        let g = Gguf::open(gguf_path).map_err(|e| anyhow!("open gguf: {e}"))?;
        let arch = g
            .metadata()
            .str("general.architecture")
            .unwrap_or("")
            .to_string();
        // llama and qwen3 share the transformer; qwen3 adds QK-norm + explicit head_dim.
        let qk_norm = match arch.as_str() {
            "llama" | "qwen3" => arch == "qwen3",
            other => bail!("infr-llama supports architecture=llama|qwen3, got {other:?}"),
        };
        let mk = |k: &str| format!("{arch}.{k}");
        let n_layer = meta_u64(&g, &mk("block_count")).context("block_count")? as usize;
        let n_embd = meta_u64(&g, &mk("embedding_length")).context("embedding_length")? as usize;
        let n_head = meta_u64(&g, &mk("attention.head_count")).context("head_count")? as usize;
        let n_kv = meta_u64(&g, &mk("attention.head_count_kv")).unwrap_or(n_head as u64) as usize;
        let n_ff =
            meta_u64(&g, &mk("feed_forward_length")).context("feed_forward_length")? as usize;
        // head_dim: explicit (qwen3 key_length) or n_embd/n_head (llama). Note q_dim = n_head*head_dim
        // may differ from n_embd (qwen3-0.6B: 16*128=2048 vs embd 1024).
        let head_dim =
            meta_u64(&g, &mk("attention.key_length")).unwrap_or((n_embd / n_head) as u64) as usize;
        let rope_dim =
            meta_u64(&g, &mk("rope.dimension_count")).unwrap_or(head_dim as u64) as usize;
        let rope_theta = g
            .metadata()
            .get(&mk("rope.freq_base"))
            .and_then(|v| match v {
                infr_core::MetaValue::F64(f) => Some(*f as f32),
                infr_core::MetaValue::U64(u) => Some(*u as f32),
                _ => None,
            })
            .unwrap_or(10000.0);
        let rms_eps = g
            .metadata()
            .get(&mk("attention.layer_norm_rms_epsilon"))
            .and_then(|v| match v {
                infr_core::MetaValue::F64(f) => Some(*f as f32),
                _ => None,
            })
            .unwrap_or(1e-5);
        let eos = meta_u64(&g, "tokenizer.ggml.eos_token_id").unwrap_or(2) as u32;

        let be = VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;

        // token embeddings (host) + lm head (GPU). tied unless output.weight present.
        let (token_embd, te_shape) = load_tensor_dequant(&g, "token_embd.weight")?;
        let vocab = te_shape[1];
        let lm_head = if g.tensors().iter().any(|t| t.name == "output.weight") {
            upload_wt(&be, &g, "output.weight")?
        } else {
            // tied; token_embd already dequantized to f32 for the host gather → f16 lm head
            Wt::F16(
                be.upload_weight_f16(&token_embd)
                    .map_err(|e| anyhow!("upload lm_head: {e}"))?,
            )
        };

        let (output_norm, _) = load_tensor_dequant(&g, "output_norm.weight")?;
        let output_norm_buf = be
            .upload_weight(&output_norm)
            .map_err(|e| anyhow!("upload output_norm: {e}"))?;

        // Loading the per-layer weights (dequant + GPU upload) dominates startup, especially for
        // big models — show a byte-progress bar so it reports copy speed + ETA (same shared style as
        // the download bar). Total/inc are GGUF source bytes; per-layer = sum of that layer's tensors.
        let layer_bytes = |l: usize| -> u64 {
            let prefix = format!("blk.{l}.");
            g.tensors()
                .iter()
                .filter(|t| t.name.starts_with(&prefix))
                .map(|t| t.nbytes as u64)
                .sum()
        };
        let total_bytes: u64 = (0..n_layer).map(layer_bytes).sum();
        let pb = infr_core::progress::bar(
            Some(total_bytes),
            "loading weights",
            infr_core::progress::Unit::Bytes,
        );
        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let up = |be: &VulkanBackend, name: String| -> Result<Wt> { upload_wt(be, &g, &name) };
            let attn_norm = load_tensor_dequant(&g, &p("attn_norm.weight"))?.0;
            let ffn_norm = load_tensor_dequant(&g, &p("ffn_norm.weight"))?.0;
            let attn_norm_buf = be
                .upload_weight(&attn_norm)
                .map_err(|e| anyhow!("upload attn_norm {l}: {e}"))?;
            let ffn_norm_buf = be
                .upload_weight(&ffn_norm)
                .map_err(|e| anyhow!("upload ffn_norm {l}: {e}"))?;
            // fuse gate + up into one [2*n_ff, n_embd] weight (concat rows). Quant stays quantized.
            let gate_dtype = g
                .tensors()
                .iter()
                .find(|t| t.name == p("ffn_gate.weight"))
                .map(|t| t.dtype);
            let wgateup = if gate_dtype.map(is_quant).unwrap_or(false) {
                let dt = gate_dtype.unwrap();
                let gb = g
                    .tensor_bytes(&p("ffn_gate.weight"))
                    .map_err(|e| anyhow!("{e}"))?;
                let ub = g
                    .tensor_bytes(&p("ffn_up.weight"))
                    .map_err(|e| anyhow!("{e}"))?;
                let (bits, blk) = quant_params(dt);
                let (mut qv, mut sc, mut mn) = dequant_unified(dt, gb);
                let (qu, scu, mnu) = dequant_unified(dt, ub);
                qv.extend(qu);
                sc.extend(scu);
                mn.extend(mnu);
                let (q, s, m) = pack_unified(&qv, &sc, &mn, bits, blk);
                Wt::Q {
                    q: be
                        .upload_weight_bytes(bytemuck::cast_slice(&q))
                        .map_err(|e| anyhow!("wgateup {l}: {e}"))?,
                    s: be
                        .upload_weight_bytes(bytemuck::cast_slice(&s))
                        .map_err(|e| anyhow!("wgateup {l}: {e}"))?,
                    m: be
                        .upload_weight_bytes(bytemuck::cast_slice(&m))
                        .map_err(|e| anyhow!("wgateup {l}: {e}"))?,
                    bits,
                    blk_shift: blk.trailing_zeros(),
                }
            } else if gate_dtype.map(is_codebook_quant).unwrap_or(false) {
                // Codebook quants: host dequant gate+up → f16, fuse into one buffer
                let dt = gate_dtype.unwrap();
                let to_f16_bytes = |bytes: &[u8]| -> Vec<u8> {
                    dequant_codebook(dt, bytes)
                        .iter()
                        .flat_map(|&v| f32_to_f16_sat(v).to_bits().to_le_bytes())
                        .collect()
                };
                let gb = g
                    .tensor_bytes(&p("ffn_gate.weight"))
                    .map_err(|e| anyhow!("{e}"))?;
                let ub = g
                    .tensor_bytes(&p("ffn_up.weight"))
                    .map_err(|e| anyhow!("{e}"))?;
                let mut gateup = to_f16_bytes(gb);
                gateup.extend_from_slice(&to_f16_bytes(ub));
                Wt::F16(
                    be.upload_weight_bytes(&gateup)
                        .map_err(|e| anyhow!("upload wgateup codebook {l}: {e}"))?,
                )
            } else {
                let mut gateup = f16_bytes(&g, &p("ffn_gate.weight"))?;
                gateup.extend_from_slice(&f16_bytes(&g, &p("ffn_up.weight"))?);
                Wt::F16(
                    be.upload_weight_bytes(&gateup)
                        .map_err(|e| anyhow!("upload wgateup {l}: {e}"))?,
                )
            };
            let (q_norm_buf, k_norm_buf) = if qk_norm {
                (
                    Some(
                        be.upload_weight(&load_tensor_dequant(&g, &p("attn_q_norm.weight"))?.0)
                            .map_err(|e| anyhow!("upload q_norm {l}: {e}"))?,
                    ),
                    Some(
                        be.upload_weight(&load_tensor_dequant(&g, &p("attn_k_norm.weight"))?.0)
                            .map_err(|e| anyhow!("upload k_norm {l}: {e}"))?,
                    ),
                )
            } else {
                (None, None)
            };
            layers.push(LayerWeights {
                attn_norm,
                ffn_norm,
                attn_norm_buf,
                ffn_norm_buf,
                wq: up(&be, p("attn_q.weight"))?,
                wk: up(&be, p("attn_k.weight"))?,
                wv: up(&be, p("attn_v.weight"))?,
                wo: up(&be, p("attn_output.weight"))?,
                wgateup,
                wdown: up(&be, p("ffn_down.weight"))?,
                q_norm_buf,
                k_norm_buf,
            });
            pb.inc(layer_bytes(l));
        }
        pb.finish_and_clear();

        let tokenizer = match tokenizer_path {
            Some(p) => Tokenizer::from_file(p).map_err(|e| anyhow!("load tokenizer: {e}"))?,
            None => build_tokenizer(&g)?,
        };
        // A variant that encodes special-token strings as literal text, for untrusted user content.
        let mut user_tokenizer = tokenizer.clone();
        user_tokenizer.set_encode_special_tokens(true);

        // Stop on the GGUF eos plus any chat-end markers in the vocab — a chat model can emit
        // <|endoftext|> mid-turn, and stopping only on <|im_end|> lets it ramble past the answer.
        let mut eos_ids = vec![eos];
        for name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"] {
            if let Some(id) = tokenizer.token_to_id(name) {
                if !eos_ids.contains(&id) {
                    eos_ids.push(id);
                }
            }
        }

        let cfg = Config {
            n_layer,
            n_head,
            n_kv,
            n_embd,
            n_ff,
            head_dim,
            rope_dim,
            rope_theta,
            rms_eps,
            vocab,
            eos,
            eos_ids,
            qk_norm,
        };
        Ok(Self {
            be,
            cfg,
            token_embd,
            lm_head,
            output_norm,
            output_norm_buf,
            layers,
            tokenizer,
            user_tokenizer,
            sampler: std::cell::Cell::new(Sampler::default()),
        })
    }

    /// Encode one chat turn: ChatML markers as real special tokens, USER content as literal text
    /// (so a user typing `<|im_end|>`/`<think>`/etc. can't inject or break the turn structure).
    /// `started` closes the previous assistant turn first.
    fn turn_tokens(&self, user: &str, started: bool) -> Result<Vec<u32>> {
        let pre = if started {
            "<|im_end|>\n<|im_start|>user\n"
        } else {
            "<|im_start|>user\n"
        };
        let post = "<|im_end|>\n<|im_start|>assistant\n";
        let enc = |t: &Tokenizer, s: &str| -> Result<Vec<u32>> {
            t.encode(s, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| anyhow!("encode: {e}"))
        };
        let mut ids = enc(&self.tokenizer, pre)?;
        ids.extend(enc(&self.user_tokenizer, user)?);
        ids.extend(enc(&self.tokenizer, post)?);
        Ok(ids)
    }

    /// Set token sampling (temp ≤ 0 → greedy). Applies to subsequent `generate`/`ChatSession::turn`.
    pub fn set_sampling(&self, temp: f32, top_k: usize, top_p: f32) {
        self.sampler.set(Sampler { temp, top_k, top_p });
    }

    fn lin(&self, w: &dyn Buffer, x: &[f32], rows: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        self.be.linear(w, x, rows, in_f, out_f).expect("gpu linear")
    }

    /// Full forward over `tokens`; returns logits (`vocab`) for the LAST position.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let c = &self.cfg;
        let t = tokens.len();
        let (ne, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_kv, c.head_dim);

        // embedding gather -> hidden [T, n_embd]
        let mut hidden = vec![0f32; t * ne];
        for (i, &tok) in tokens.iter().enumerate() {
            let src = &self.token_embd[tok as usize * ne..(tok as usize + 1) * ne];
            hidden[i * ne..(i + 1) * ne].copy_from_slice(src);
        }

        for layer in &self.layers {
            // --- attention ---
            let hn = rmsnorm_rows(&hidden, &layer.attn_norm, t, ne, c.rms_eps);
            let mut q = self.lin(layer.wq.f16(), &hn, t, ne, nh * hd);
            let mut k = self.lin(layer.wk.f16(), &hn, t, ne, nkv * hd);
            let v = self.lin(layer.wv.f16(), &hn, t, ne, nkv * hd);
            rope_rows(&mut q, t, nh, hd, c.rope_dim, c.rope_theta);
            rope_rows(&mut k, t, nkv, hd, c.rope_dim, c.rope_theta);
            let attn = attention(&q, &k, &v, t, nh, nkv, hd);
            let ao = self.lin(layer.wo.f16(), &attn, t, nh * hd, ne);
            for i in 0..t * ne {
                hidden[i] += ao[i];
            }

            // --- ffn (SwiGLU) ---
            let hn2 = rmsnorm_rows(&hidden, &layer.ffn_norm, t, ne, c.rms_eps);
            let gu = self.lin(layer.wgateup.f16(), &hn2, t, ne, 2 * c.n_ff);
            let mut act = vec![0f32; t * c.n_ff];
            for r in 0..t {
                for i in 0..c.n_ff {
                    let g = gu[r * 2 * c.n_ff + i];
                    act[r * c.n_ff + i] = silu(g) * gu[r * 2 * c.n_ff + c.n_ff + i];
                }
            }
            let down = self.lin(layer.wdown.f16(), &act, t, c.n_ff, ne);
            for i in 0..t * ne {
                hidden[i] += down[i];
            }
        }

        // final norm on the last row, then lm_head
        let last = &hidden[(t - 1) * ne..t * ne];
        let normed = rmsnorm_rows(last, &self.output_norm, 1, ne, c.rms_eps);
        self.lin(self.lm_head.f16(), &normed, 1, ne, c.vocab)
    }

    /// GPU-resident forward: records the whole stack into one command buffer (one submit),
    /// all ops on the GPU. Returns logits (`vocab`) for the last position. Much fewer
    /// GPU round-trips than `forward` (which submits per linear).
    pub fn forward_resident(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let t = tokens.len();
        let (ne, nh, nkv, hd, nff) = (c.n_embd, c.n_head, c.n_kv, c.head_dim, c.n_ff);

        let mut hidden_host = vec![0f32; t * ne];
        for (i, &tok) in tokens.iter().enumerate() {
            hidden_host[i * ne..(i + 1) * ne]
                .copy_from_slice(&self.token_embd[tok as usize * ne..(tok as usize + 1) * ne]);
        }

        let alloc = |n: usize, usage: BufferUsage| -> Result<Box<dyn Buffer>> {
            self.be
                .alloc((n * 4).max(4), usage)
                .map_err(|e| anyhow!("{e}"))
        };
        let hidden = alloc(t * ne, BufferUsage::Staging)?;
        self.be
            .upload(hidden.as_ref(), bytemuck::cast_slice(&hidden_host))
            .map_err(|e| anyhow!("{e}"))?;
        let hn = alloc(t * ne, BufferUsage::Activations)?;
        let q = alloc(t * nh * hd, BufferUsage::Activations)?;
        let k = alloc(t * nkv * hd, BufferUsage::Activations)?;
        let v = alloc(t * nkv * hd, BufferUsage::Activations)?;
        let attn = alloc(t * nh * hd, BufferUsage::Activations)?;
        let ao = alloc(t * ne, BufferUsage::Activations)?;
        let hn2 = alloc(t * ne, BufferUsage::Activations)?;
        let gu = alloc(t * 2 * nff, BufferUsage::Activations)?;
        let act = alloc(t * nff, BufferUsage::Activations)?;
        let down = alloc(t * ne, BufferUsage::Activations)?;
        let logits = alloc(t * c.vocab, BufferUsage::Readback)?;

        let prof = std::env::var("INFR_PROFILE").is_ok();
        let t_rec0 = std::time::Instant::now();
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        for layer in &self.layers {
            rec.rmsnorm(
                hidden.as_ref(),
                layer.attn_norm_buf.as_ref(),
                hn.as_ref(),
                t,
                ne,
                c.rms_eps,
            );
            rec.linear(layer.wq.f16(), hn.as_ref(), q.as_ref(), t, ne, nh * hd);
            rec.linear(layer.wk.f16(), hn.as_ref(), k.as_ref(), t, ne, nkv * hd);
            rec.linear(layer.wv.f16(), hn.as_ref(), v.as_ref(), t, ne, nkv * hd);
            rec.rope(
                q.as_ref(),
                q.as_ref(),
                t,
                nh,
                hd,
                c.rope_dim,
                c.rope_theta,
                0,
            );
            rec.rope(
                k.as_ref(),
                k.as_ref(),
                t,
                nkv,
                hd,
                c.rope_dim,
                c.rope_theta,
                0,
            );
            rec.attention(
                q.as_ref(),
                k.as_ref(),
                v.as_ref(),
                attn.as_ref(),
                t,
                nh,
                nkv,
                hd,
            );
            rec.linear(layer.wo.f16(), attn.as_ref(), ao.as_ref(), t, nh * hd, ne);
            rec.add(hidden.as_ref(), ao.as_ref(), hidden.as_ref(), t * ne);
            rec.rmsnorm(
                hidden.as_ref(),
                layer.ffn_norm_buf.as_ref(),
                hn2.as_ref(),
                t,
                ne,
                c.rms_eps,
            );
            rec.linear(
                layer.wgateup.f16(),
                hn2.as_ref(),
                gu.as_ref(),
                t,
                ne,
                2 * nff,
            );
            rec.silu_mul_fused(gu.as_ref(), act.as_ref(), t, nff);
            rec.linear(layer.wdown.f16(), act.as_ref(), down.as_ref(), t, nff, ne);
            rec.add(hidden.as_ref(), down.as_ref(), hidden.as_ref(), t * ne);
        }
        rec.rmsnorm(
            hidden.as_ref(),
            self.output_norm_buf.as_ref(),
            hn.as_ref(),
            t,
            ne,
            c.rms_eps,
        );
        rec.linear(
            self.lm_head.f16(),
            hn.as_ref(),
            logits.as_ref(),
            t,
            ne,
            c.vocab,
        );
        let t_rec = t_rec0.elapsed();
        let t_gpu0 = std::time::Instant::now();
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        let t_gpu = t_gpu0.elapsed();

        let mut bytes = vec![0u8; t * c.vocab * 4];
        self.be
            .download(logits.as_ref(), &mut bytes)
            .map_err(|e| anyhow!("{e}"))?;
        if prof {
            eprintln!("[prof] t={t} record={t_rec:?} gpu_submit_wait={t_gpu:?}");
        }
        let all: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        Ok(all[(t - 1) * c.vocab..].to_vec())
    }

    /// Allocate a KV cache with room for `max_ctx` tokens.
    pub fn new_kv(&self, max_ctx: usize) -> Result<KvCache> {
        let c = &self.cfg;
        let row = c.n_kv * c.head_dim;
        let mut k = Vec::with_capacity(c.n_layer);
        let mut v = Vec::with_capacity(c.n_layer);
        for _ in 0..c.n_layer {
            // f16 KV cache: 2 bytes/elem (half the f32 footprint that grows with context).
            k.push(
                self.be
                    .alloc((max_ctx + 64) * row * 2, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            );
            v.push(
                self.be
                    .alloc((max_ctx + 64) * row * 2, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            );
        }
        Ok(KvCache {
            k,
            v,
            len: 0,
            max_ctx,
        })
    }

    /// KV-cached resident forward: processes only `new_tokens` (n rows), appends their K/V to
    /// the cache, and attends over the whole cache. Returns logits for the last new token.
    pub fn forward_resident_kv(&self, new_tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let n = new_tokens.len();
        let pos = kv.len;
        let (ne, nh, nkv, hd, nff) = (c.n_embd, c.n_head, c.n_kv, c.head_dim, c.n_ff);
        if pos + n > kv.max_ctx {
            bail!("KV cache overflow: {} > {}", pos + n, kv.max_ctx);
        }

        let mut hidden_host = vec![0f32; n * ne];
        for (i, &tok) in new_tokens.iter().enumerate() {
            hidden_host[i * ne..(i + 1) * ne]
                .copy_from_slice(&self.token_embd[tok as usize * ne..(tok as usize + 1) * ne]);
        }
        let alloc = |m: usize, u: BufferUsage| -> Result<Box<dyn Buffer>> {
            self.be.alloc((m * 4).max(4), u).map_err(|e| anyhow!("{e}"))
        };
        // Prefill (many tokens) reuses each weight across all rows → a coopmat GEMM (matmul_proj)
        // beats the per-row GEMV and lets one submit cover a big chunk without tripping the GPU
        // watchdog. Decode (n==1) and Llama stay on the fused GEMV path. GEMM writes ceil(n/64)*64
        // rows (extra rows are 0), so its output buffers are M-padded to mpad.
        let use_gemm = c.qk_norm && n >= 64 && std::env::var("INFR_NOGEMM").is_err();
        // Register-O flash (FlashAttention-2 layout, Br=128) is opt-in (INFR_FLASH_REG) while it's
        // A/B'd vs the BM=64 flash; it needs mpad padded to 128 (q/attn/scratch).
        let use_flash_reg = use_gemm && hd == 128 && std::env::var("INFR_FLASH_REG").is_ok();
        let mpad = if use_flash_reg {
            n.div_ceil(128) * 128
        } else if use_gemm {
            n.div_ceil(64) * 64
        } else {
            n
        };
        // Prefill attention has TWO interchangeable algorithms — keep BOTH; which one wins is
        // HARDWARE-dependent (the card's compute:bandwidth ratio):
        //  • flash (attention_prefill_flash, split-K, 8-warp register-blocked for hd=128): never
        //    materializes the S=[m,kv] scores buffer → far less HBM. After warptile-izing its GEMMs
        //    it now also wins on this bandwidth-rich card (+8-12% across ctx, 32k 2351→2620) AND is
        //    the right choice on bandwidth-starved cards (APUs, cut-down GPUs) / very long context.
        //  • non-FA (attn_qk → softmax → attn_pv): materializes S (more HBM) but uses big-tile
        //    (BN=256) warptile GEMMs. Fallback for hd≠128 (the flash warptile is hd=128-specialized)
        //    or via INFR_NO_FLASH.
        // Both are correctness-tested (attention_prefill_{nonfa,flash}_matches_cpu) so neither rots.
        // DEFAULT = flash for hd=128. TODO: auto-select from device bandwidth/FLOP caps.
        let use_flash = use_gemm && hd == 128 && std::env::var("INFR_NO_FLASH").is_err();
        let nonfa = use_gemm && !use_flash;
        let hidden = alloc(n * ne, BufferUsage::Staging)?;
        self.be
            .upload(hidden.as_ref(), bytemuck::cast_slice(&hidden_host))
            .map_err(|e| anyhow!("{e}"))?;
        let hn = alloc(n * ne, BufferUsage::Activations)?;
        // q is f16 (read by the f16 attention kernels), like the KV cache. q and attn are M-padded
        // to mpad rows so the coopmat prefill attention can read/write whole 64-row tiles.
        let q = self
            .be
            .alloc((mpad * nh * hd * 2).max(4), BufferUsage::Activations)
            .map_err(|e| anyhow!("{e}"))?;
        let attn = alloc(mpad * nh * hd, BufferUsage::Activations)?;
        let act = alloc(n * nff, BufferUsage::Activations)?;
        // gate+up intermediate for the un-fused decode FFN (rmsnorm → gate/up GEMV → SwiGLU). The
        // GEMM path uses its own `gu` in `gemm_bufs`; this serves the small-batch (decode) path.
        let gu_ffn = alloc(n * 2 * nff, BufferUsage::Activations)?;
        // Only the LAST position's logits are needed → compute lm_head for one row. (Computing all n
        // rows at long context is a huge wasted dispatch + ~n*vocab buffer that can exceed the GPU
        // watchdog and lose the device.)
        let hlast = alloc(ne, BufferUsage::Activations)?;
        let logits = alloc(c.vocab, BufferUsage::Readback)?;
        let kvrow = nkv * hd;
        // qwen3 (QK-norm) uses an un-fused attention input: raw Q/K/V projections then a separate
        // per-head RMSNorm+RoPE. (Llama uses the single fused attn_in instead.)
        let qkv_raw = if c.qk_norm {
            Some((
                alloc(mpad * nh * hd, BufferUsage::Activations)?,
                alloc(mpad * kvrow, BufferUsage::Activations)?,
                alloc(mpad * kvrow, BufferUsage::Activations)?,
            ))
        } else {
            None
        };
        // GEMM-prefill scratch: o-proj out (ao), gate/up out (gu), down out (down), all M-padded;
        // plus a tiny dummy buffer bound as scales/mins when the weight is f16 (unused there).
        let gemm_bufs = if use_gemm {
            Some((
                alloc(mpad * ne, BufferUsage::Activations)?,
                alloc(mpad * 2 * nff, BufferUsage::Activations)?,
                alloc(mpad * ne, BufferUsage::Activations)?,
                alloc(1, BufferUsage::Activations)?,
            ))
        } else {
            None
        };
        // mmq (dp4a integer) prefill scratch: int8 activations + per-32-block f16 scale/sum, sized
        // for the largest projection K. Reused across all u4 projections. (q6/q8/f16 stay on the
        // f16 warp matmul_proj.) DEFAULT for u4 prefill (INFR_NOMMQ to disable).
        let use_mmq_alloc = use_gemm && std::env::var("INFR_NOMMQ").is_err();
        let mmq_bufs = if use_mmq_alloc {
            let maxk = ne.max(nh * hd).max(nff);
            let nblk = maxk / 32;
            Some((
                alloc(mpad * maxk, BufferUsage::Activations)?, // qa int8 (1 byte/elem)
                alloc(mpad * nblk * 2, BufferUsage::Activations)?, // dact f16
                alloc(mpad * nblk * 2, BufferUsage::Activations)?, // sact f16
            ))
        } else {
            None
        };

        // Flash-decoding: for single-token decode, split each head's KV range across many
        // workgroups (partials in pm/pl/pacc), so attention isn't stuck on `nh` workgroups. The
        // chunk size is adaptive: a coarse fixed chunk leaves too few workgroups at low/mid context,
        // so size it to ~`nchunk_div` chunks/head (≈nh*nchunk_div workgroups) with a 64-key floor.
        // ~32 chunks/head saturates pass-1's KV bandwidth on the 7900 XTX (nh*32=512 workgroups ≫ 96
        // CUs) while HALVING pass-2 (attn_combine) work vs the old 64 — combine is a serial scan over
        // n_chunks, so fewer chunks is a pure win once pass-1 is bandwidth-bound (decode +3..6% at
        // d4k-16k, no shallow regression). Override with INFR_DECODE_NCHUNK. Reused across layers.
        let kv_len = pos + n;
        let nchunk_div = std::env::var("INFR_DECODE_NCHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&d| d > 0)
            .unwrap_or(32);
        let chunk = (kv_len / nchunk_div).clamp(64, 512);
        // Non-FA scores scratch: [nh, mpad, kv_pad] f16 (kv padded to 256 — the 8-warp attn_qk's BN;
        // the recorder uses the same padding).
        let nonfa_s = if nonfa && !use_flash {
            let kv_pad = kv_len.div_ceil(256) * 256;
            Some(
                self.be
                    .alloc(nh * mpad * kv_pad * 2, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            None
        };
        // Split-K PV partials: [max_splits, mpad, nh*hd] f32 (summed by attn_pv_reduce). Max 8 splits.
        // Flash split-K scratch: po=[≤8, mpad, nh, hd] f32 partials + pm/pl=[≤8, mpad, nh] f32.
        let flash_bufs = if use_flash || use_flash_reg {
            Some((
                alloc(8 * mpad * nh * hd, BufferUsage::Activations)?,
                alloc(8 * mpad * nh, BufferUsage::Activations)?,
                alloc(8 * mpad * nh, BufferUsage::Activations)?,
            ))
        } else {
            None
        };
        let nonfa_pv = if nonfa && !use_flash {
            Some(
                self.be
                    .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            None
        };
        let use_split = n == 1 && kv_len > chunk;
        let n_chunks = if use_split { kv_len.div_ceil(chunk) } else { 0 };
        let split_bufs = if use_split {
            Some((
                alloc(nh * n_chunks, BufferUsage::Activations)?,
                alloc(nh * n_chunks, BufferUsage::Activations)?,
                alloc(nh * n_chunks * hd, BufferUsage::Activations)?,
            ))
        } else {
            None
        };

        let prof = std::env::var("INFR_PROF").is_ok();
        let t_rec = std::time::Instant::now();
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        // weight-op dispatchers: pick the f16, quant, or native kernel based on how the weight is stored.
        let lin = |w: &Wt, x: &dyn Buffer, y: &dyn Buffer, rows: usize, inf: usize, outf: usize| {
            match w {
                Wt::F16(b) => rec.linear(b.as_ref(), x, y, rows, inf, outf),
                Wt::Q {
                    q,
                    s,
                    m,
                    bits,
                    blk_shift,
                } => rec.linear_q(
                    q.as_ref(),
                    s.as_ref(),
                    m.as_ref(),
                    x,
                    y,
                    rows,
                    inf,
                    outf,
                    *bits,
                    *blk_shift,
                ),
                Wt::Native { buf, dtype } => {
                    rec.linear_native(*dtype, buf.as_ref(), x, y, rows, inf, outf)
                }
            }
        };
        let lin_add = |w: &Wt,
                       x: &dyn Buffer,
                       res: &dyn Buffer,
                       y: &dyn Buffer,
                       rows: usize,
                       inf: usize,
                       outf: usize| match w {
            Wt::F16(b) => rec.linear_add(b.as_ref(), x, res, y, rows, inf, outf),
            Wt::Q {
                q,
                s,
                m,
                bits,
                blk_shift,
            } => rec.linear_add_q(
                q.as_ref(),
                s.as_ref(),
                m.as_ref(),
                x,
                res,
                y,
                rows,
                inf,
                outf,
                *bits,
                *blk_shift,
            ),
            Wt::Native { buf, dtype } => {
                rec.linear_add_native(*dtype, buf.as_ref(), x, res, y, rows, inf, outf)
            }
        };
        let ffn = |hidden: &dyn Buffer,
                   nw: &dyn Buffer,
                   w: &Wt,
                   act: &dyn Buffer,
                   rows: usize,
                   ne: usize,
                   nff: usize,
                   eps: f32| match w {
            Wt::F16(b) => rec.ffn_in(hidden, nw, b.as_ref(), act, rows, ne, nff, eps),
            Wt::Q {
                q,
                s,
                m,
                bits,
                blk_shift,
            } => rec.ffn_in_q(
                hidden,
                nw,
                q.as_ref(),
                s.as_ref(),
                m.as_ref(),
                act,
                rows,
                ne,
                nff,
                eps,
                *bits,
                *blk_shift,
            ),
            // wgateup is always built via the Q/F16 fusion path, so Native won't appear here
            // for Phases 0-2. If it does (future), fall through to the unfused path.
            Wt::Native { .. } => unimplemented!("native ffn_in not yet implemented"),
        };
        // coopmat GEMM `c = a · Wᵀ` for prefill; binds the dummy buffer as scales/mins for f16.
        // Integer dp4a mmq path is DEFAULT for u4 projections (INFR_NOMMQ to disable). It keeps the
        // weight quantized (no per-GEMM dequant), which is the win at SMALL ubatch where the f16 path
        // falls back to the dequant-bound BN=64 gemm_proj and re-dequantizes the whole weight once
        // per BM-row-tile: mmq is +26..50% at ub≤512 and still +3..5% at ub=4096 (where the f16 warp
        // matmul is compute-bound). Adds a cheap quant_q8 activation pass amortized across projections.
        let use_mmq = use_mmq_alloc;
        let mm = |w: &Wt, a: &dyn Buffer, cbuf: &dyn Buffer, rows: usize, k: usize, outf: usize| {
            let dummy = gemm_bufs.as_ref().unwrap().3.as_ref();
            match w {
                Wt::F16(b) => {
                    rec.matmul_proj(a, b.as_ref(), dummy, dummy, cbuf, rows, k, outf, 16, 0)
                }
                Wt::Q {
                    q,
                    s,
                    m,
                    bits,
                    blk_shift,
                } => {
                    // u4 → dp4a integer mmq (no per-GEMM dequant; weights stay quantized). q6/q8
                    // keep the f16 warp matmul_proj. Requires k%32, outf%64 (all projections satisfy).
                    if *bits == 4 && k % 32 == 0 && outf % 64 == 0 && use_mmq {
                        let (qa, dact, sact) = mmq_bufs.as_ref().unwrap();
                        rec.matmul_proj_mmq(
                            a,
                            q.as_ref(),
                            s.as_ref(),
                            m.as_ref(),
                            cbuf,
                            qa.as_ref(),
                            dact.as_ref(),
                            sact.as_ref(),
                            rows,
                            k,
                            outf,
                        );
                    } else {
                        rec.matmul_proj(
                            a,
                            q.as_ref(),
                            s.as_ref(),
                            m.as_ref(),
                            cbuf,
                            rows,
                            k,
                            outf,
                            *bits,
                            *blk_shift,
                        );
                    }
                }
                // Native-block prefill: use the native GEMV (correct but slower than GEMM;
                // a native GEMM path is Phase 6 follow-on). Falls back gracefully.
                Wt::Native { buf, dtype } => {
                    rec.linear_native(*dtype, buf.as_ref(), a, cbuf, rows, k, outf)
                }
            }
        };
        for (li, layer) in self.layers.iter().enumerate() {
            if let Some((qr, kr, vr)) = &qkv_raw {
                // qwen3: rmsnorm → Q/K/V projections → per-head QK-norm+RoPE (K/V into the cache)
                let rmsnorm_qkv = || {
                    rec.rmsnorm(
                        hidden.as_ref(),
                        layer.attn_norm_buf.as_ref(),
                        hn.as_ref(),
                        n,
                        ne,
                        c.rms_eps,
                    );
                };
                // Un-fused (rmsnorm + 3× subgroup GEMV) beats the fused attn_in_q: the fused kernel
                // recomputes the RMS sum-of-squares per output row (~2× compute), and the standalone
                // GEMV is the fast subgroup mul_mat_vec_q. Opt back into fusion with INFR_FUSE.
                let fuse_qkv = std::env::var("INFR_FUSE").is_ok()
                    && matches!(
                        (&layer.wq, &layer.wk, &layer.wv),
                        (Wt::Q { .. }, Wt::Q { .. }, Wt::Q { .. })
                    );
                if use_gemm {
                    rmsnorm_qkv();
                    mm(&layer.wq, hn.as_ref(), qr.as_ref(), n, ne, nh * hd);
                    mm(&layer.wk, hn.as_ref(), kr.as_ref(), n, ne, kvrow);
                    mm(&layer.wv, hn.as_ref(), vr.as_ref(), n, ne, kvrow);
                } else if fuse_qkv {
                    // decode: fuse rmsnorm + Q/K/V quant projections into one dispatch.
                    let (
                        Wt::Q {
                            q: qq,
                            s: qs,
                            m: qm,
                            bits: qb,
                            blk_shift: qbs,
                        },
                        Wt::Q {
                            q: kq,
                            s: ks,
                            m: km,
                            bits: kb,
                            blk_shift: kbs,
                        },
                        Wt::Q {
                            q: vq,
                            s: vs,
                            m: vm,
                            bits: vb,
                            blk_shift: vbs,
                        },
                    ) = (&layer.wq, &layer.wk, &layer.wv)
                    else {
                        unreachable!()
                    };
                    rec.attn_in_q(
                        hidden.as_ref(),
                        layer.attn_norm_buf.as_ref(),
                        (qq.as_ref(), qs.as_ref(), qm.as_ref()),
                        (kq.as_ref(), ks.as_ref(), km.as_ref()),
                        (vq.as_ref(), vs.as_ref(), vm.as_ref()),
                        qr.as_ref(),
                        kr.as_ref(),
                        vr.as_ref(),
                        n,
                        ne,
                        nh * hd,
                        kvrow,
                        c.rms_eps,
                        (*qb, *qbs),
                        (*kb, *kbs),
                        (*vb, *vbs),
                    );
                } else {
                    rmsnorm_qkv();
                    lin(&layer.wq, hn.as_ref(), qr.as_ref(), n, ne, nh * hd);
                    lin(&layer.wk, hn.as_ref(), kr.as_ref(), n, ne, kvrow);
                    lin(&layer.wv, hn.as_ref(), vr.as_ref(), n, ne, kvrow);
                }
                rec.qk_norm_rope(
                    qr.as_ref(),
                    layer.q_norm_buf.as_ref().unwrap().as_ref(),
                    q.as_ref(),
                    n,
                    nh,
                    hd,
                    c.rope_dim,
                    c.rope_theta,
                    pos,
                    0,
                    c.rms_eps,
                );
                rec.qk_norm_rope(
                    kr.as_ref(),
                    layer.k_norm_buf.as_ref().unwrap().as_ref(),
                    kv.k[li].as_ref(),
                    n,
                    nkv,
                    hd,
                    c.rope_dim,
                    c.rope_theta,
                    pos,
                    pos,
                    c.rms_eps,
                );
                // v_raw is f32; cast into the f16 V cache at row offset `pos`.
                rec.store_f16(vr.as_ref(), kv.v[li].as_ref(), n * kvrow, pos * kvrow);
            } else {
                rec.attn_in(
                    hidden.as_ref(),
                    layer.attn_norm_buf.as_ref(),
                    layer.wq.f16(),
                    layer.wk.f16(),
                    layer.wv.f16(),
                    q.as_ref(),
                    kv.k[li].as_ref(),
                    kv.v[li].as_ref(),
                    n,
                    ne,
                    nh,
                    nkv,
                    hd,
                    c.rope_dim,
                    c.rope_theta,
                    pos,
                    c.rms_eps,
                );
            }
            if use_flash_reg {
                // prefill: FlashAttention-2 register-O (Br=128) — opt-in A/B vs the BM=64 flash.
                let (po, pm, pl) = flash_bufs.as_ref().unwrap();
                rec.attention_prefill_flash_reg(
                    q.as_ref(),
                    kv.k[li].as_ref(),
                    kv.v[li].as_ref(),
                    attn.as_ref(),
                    po.as_ref(),
                    pm.as_ref(),
                    pl.as_ref(),
                    n,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                );
            } else if use_flash {
                // prefill: fused flash attention (no materialized S buffer), split-K for occupancy.
                let (po, pm, pl) = flash_bufs.as_ref().unwrap();
                rec.attention_prefill_flash(
                    q.as_ref(),
                    kv.k[li].as_ref(),
                    kv.v[li].as_ref(),
                    attn.as_ref(),
                    po.as_ref(),
                    pm.as_ref(),
                    pl.as_ref(),
                    n,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                );
            } else if let Some(s) = &nonfa_s {
                // prefill: non-FA clean GEMMs (QK → softmax → PV).
                rec.attention_prefill_nonfa(
                    q.as_ref(),
                    kv.k[li].as_ref(),
                    kv.v[li].as_ref(),
                    attn.as_ref(),
                    s.as_ref(),
                    nonfa_pv.as_ref().unwrap().as_ref(),
                    n,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                );
            } else if let Some((pm, pl, pacc)) = &split_bufs {
                rec.attention_kv_split(
                    q.as_ref(),
                    kv.k[li].as_ref(),
                    kv.v[li].as_ref(),
                    attn.as_ref(),
                    pm.as_ref(),
                    pl.as_ref(),
                    pacc.as_ref(),
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    chunk,
                    n_chunks,
                );
            } else {
                rec.attention_kv(
                    q.as_ref(),
                    kv.k[li].as_ref(),
                    kv.v[li].as_ref(),
                    attn.as_ref(),
                    n,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                );
            }
            if use_gemm {
                let (ao, gu, down, _) = gemm_bufs.as_ref().unwrap();
                // o-proj via GEMM then residual add (matmul_proj can't fuse the residual).
                mm(&layer.wo, attn.as_ref(), ao.as_ref(), n, nh * hd, ne);
                rec.add(hidden.as_ref(), ao.as_ref(), hidden.as_ref(), n * ne);
                // FFN un-fused: rmsnorm → gate/up GEMM → SwiGLU → down GEMM → residual add.
                rec.rmsnorm(
                    hidden.as_ref(),
                    layer.ffn_norm_buf.as_ref(),
                    hn.as_ref(),
                    n,
                    ne,
                    c.rms_eps,
                );
                mm(&layer.wgateup, hn.as_ref(), gu.as_ref(), n, ne, 2 * nff);
                rec.silu_mul_fused(gu.as_ref(), act.as_ref(), n, nff);
                mm(&layer.wdown, act.as_ref(), down.as_ref(), n, nff, ne);
                rec.add(hidden.as_ref(), down.as_ref(), hidden.as_ref(), n * ne);
            } else {
                lin_add(
                    &layer.wo,
                    attn.as_ref(),
                    hidden.as_ref(),
                    hidden.as_ref(),
                    n,
                    nh * hd,
                    ne,
                );
                if std::env::var("INFR_FUSE").is_ok() {
                    ffn(
                        hidden.as_ref(),
                        layer.ffn_norm_buf.as_ref(),
                        &layer.wgateup,
                        act.as_ref(),
                        n,
                        ne,
                        nff,
                        c.rms_eps,
                    );
                } else {
                    // Un-fused FFN: rmsnorm → gate/up subgroup GEMV → SwiGLU (no per-output redundant
                    // RMS sum-of-squares; reuses the fast mul_mat_vec_q).
                    rec.rmsnorm(
                        hidden.as_ref(),
                        layer.ffn_norm_buf.as_ref(),
                        hn.as_ref(),
                        n,
                        ne,
                        c.rms_eps,
                    );
                    lin(&layer.wgateup, hn.as_ref(), gu_ffn.as_ref(), n, ne, 2 * nff);
                    rec.silu_mul_fused(gu_ffn.as_ref(), act.as_ref(), n, nff);
                }
                lin_add(
                    &layer.wdown,
                    act.as_ref(),
                    hidden.as_ref(),
                    hidden.as_ref(),
                    n,
                    nff,
                    ne,
                );
            }
        }
        // final norm + lm_head on the LAST row only: copy hidden[n-1] → hlast, norm it, project.
        rec.copy(hidden.as_ref(), (n - 1) * ne * 4, hlast.as_ref(), 0, ne * 4);
        rec.rmsnorm(
            hlast.as_ref(),
            self.output_norm_buf.as_ref(),
            hn.as_ref(),
            1,
            ne,
            c.rms_eps,
        );
        lin(&self.lm_head, hn.as_ref(), logits.as_ref(), 1, ne, c.vocab);
        let rec_us = t_rec.elapsed().as_micros();
        let t_gpu = std::time::Instant::now();
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        if prof {
            eprintln!(
                "[prof] n={n} record={rec_us}us submit+wait={}us",
                t_gpu.elapsed().as_micros()
            );
        }

        let mut bytes = vec![0u8; c.vocab * 4];
        self.be
            .download(logits.as_ref(), &mut bytes)
            .map_err(|e| anyhow!("{e}"))?;
        kv.len += n;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Prefill chunk size at cache position `pos`. One chunk = one GPU submit; its cost grows with
    /// chunk×context, so a fixed chunk trips the watchdog (device-lost) at long context. Keep the
    /// per-submit work roughly constant by shrinking the chunk as context grows. Qwen3 (coopmat
    /// GEMM, cheap projections) gets a bigger budget than Llama (GEMV). Rounded to a multiple of 64
    /// for the GEMM tiling, floored at 64.
    pub fn prefill_chunk(&self, pos: usize) -> usize {
        // The coopmat prefill attention launches nh*ceil(chunk/64) workgroups; a too-small chunk
        // starves GPU occupancy (only nh=16 workgroups at chunk=64), which dominates at depth.
        // Keep chunks large — bigger chunks are more efficient PER QUERY despite re-reading KV —
        // with a min that holds occupancy up, while the budget still bounds per-submit work to stay
        // under the GPU watchdog at very long context.
        // Budget bumped 16M→32M: keeps the chunk at the 2048 cap through ~pos 15k and ~1000 at 32k
        // (was ~1000 at 16k, ~500 at 32k). Bigger chunks at depth are a free win now that prefill is
        // mmq + flash (lower per-token work) — a coding-agent turn ingests at depth, so its chunks
        // were the over-shrunk ones. 2048 chunks warmed to 32k run without tripping the watchdog on
        // this model; the budget still tapers for very long context / bigger models.
        let budget = if self.cfg.qk_norm {
            32_000_000
        } else {
            256 * 64
        };
        let raw = (budget / (pos + 1)).clamp(256, 2048);
        (raw / 64 * 64).max(64)
    }

    /// Prefill `new_tokens` into `kv` in watchdog-sized chunks, returning the last-token logits.
    fn prefill(&self, new_tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>> {
        let len = new_tokens.len();
        let mut logits = Vec::new();
        let mut i = 0;
        while i < len {
            let end = (i + self.prefill_chunk(kv.len)).min(len);
            logits = self.forward_resident_kv(&new_tokens[i..end], kv)?;
            i = end;
        }
        Ok(logits)
    }

    /// Prefill `new_tokens` into `kv`, then decode up to `max_new` tokens (stop at any EOS), streaming
    /// each decoded piece to `on_token`. Returns the generated token ids. `kv` carries the context, so
    /// repeated calls continue one conversation. The EOS token is not appended to the cache.
    fn run_in_cache(
        &self,
        new_tokens: &[u32],
        kv: &mut KvCache,
        max_new: usize,
        on_token: impl FnMut(&str),
    ) -> Result<Vec<u32>> {
        let logits = self.prefill(new_tokens, kv)?;
        self.decode_loop(logits, kv, max_new, on_token)
    }

    /// Greedy/sampled decode loop from `logits` (the next-token distribution), appending to `kv`.
    fn decode_loop(
        &self,
        mut logits: Vec<f32>,
        kv: &mut KvCache,
        max_new: usize,
        mut on_token: impl FnMut(&str),
    ) -> Result<Vec<u32>> {
        let sampler = self.sampler.get();
        // xorshift64 seed (non-zero); varies per call so sampling isn't fixed across turns.
        let mut rng = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9e3779b97f4a7c15)
            | 1;
        let mut generated: Vec<u32> = Vec::new();
        // Stream UTF-8-safely: decode the whole reply each step and emit only the newly-completed
        // suffix. A multi-byte char (e.g. an emoji) is split across byte-level BPE tokens; decoding a
        // single token would yield a partial sequence → U+FFFD (the `�`). Holding until the decode no
        // longer ends in the replacement char emits whole characters only. `on_token` fires once per
        // generated token (delta may be empty while a char is mid-completion), so callers can count.
        let mut stream = StreamDecoder::default();
        for _ in 0..max_new {
            let next = sample_logits(&logits, sampler, &mut rng);
            if self.cfg.eos_ids.contains(&next) {
                break;
            }
            generated.push(next);
            let full = self.tokenizer.decode(&generated, true).unwrap_or_default();
            on_token(&stream.step(&full));
            logits = self.forward_resident_kv(&[next], kv)?;
        }
        Ok(generated)
    }

    /// Greedy generate up to `max_new` tokens after `prompt` (already a chat-formatted string).
    /// One-shot: uses a fresh KV cache. For multi-turn context use [`Llama::chat_session`].
    pub fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        on_token: impl FnMut(&str),
    ) -> Result<String> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        // Size the KV cache to exactly what this run needs — bounded only by VRAM, not a fixed cap.
        let mut kv = self.new_kv(prompt_tokens.len() + max_new + 8)?;
        let generated = self.run_in_cache(&prompt_tokens, &mut kv, max_new, on_token)?;
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Start a stateful multi-turn chat with a KV cache sized for `max_ctx` tokens. Each turn keeps
    /// prior context resident, so only the new tokens are prefilled.
    pub fn chat_session(&self, max_ctx: usize) -> Result<ChatSession<'_>> {
        Ok(ChatSession {
            llama: self,
            kv: self.new_kv(max_ctx)?,
            started: false,
            last_prompt_tokens: 0,
        })
    }

    /// Wrap a user message in the SmolLM2 ChatML template.
    pub fn chatml(&self, user: &str) -> String {
        format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    }
}

/// A stateful multi-turn chat over a persistent KV cache (so the model sees prior turns). Create via
/// [`Llama::chat_session`]; call [`ChatSession::turn`] per user message.
pub struct ChatSession<'a> {
    llama: &'a Llama,
    kv: KvCache,
    started: bool,
    last_prompt_tokens: usize,
}

impl ChatSession<'_> {
    /// Tokens of context currently held (all prior turns + their replies).
    pub fn ctx_len(&self) -> usize {
        self.kv.len
    }

    /// Prompt tokens prefilled in the most recent [`turn`](Self::turn) (the ChatML-wrapped user
    /// message, including any turn-open markers). Use for prefill-rate stats.
    pub fn last_prompt_tokens(&self) -> usize {
        self.last_prompt_tokens
    }

    /// KV-cache capacity in tokens.
    pub fn max_ctx(&self) -> usize {
        self.kv.max_ctx
    }

    /// Run one user turn: append the message in ChatML, decode the assistant reply (streamed to
    /// `on_token`), and keep it all in the cache for the next turn. Returns the reply text.
    pub fn turn(
        &mut self,
        user: &str,
        max_new: usize,
        on_token: impl FnMut(&str),
    ) -> Result<String> {
        // Open this user turn (closing the prior assistant turn first if started); user content is
        // encoded as literal text so it can't inject ChatML markers.
        let toks = self.llama.turn_tokens(user, self.started)?;
        self.last_prompt_tokens = toks.len();
        // Cap generation by whatever context room remains (don't bail) — `max_new` is just a ceiling.
        let room = self.kv.max_ctx.saturating_sub(self.kv.len + toks.len() + 1);
        if room == 0 {
            bail!(
                "context full: {} held + {} prompt = {} cap — start a new session",
                self.kv.len,
                toks.len(),
                self.kv.max_ctx
            );
        }
        let max_new = max_new.min(room);
        self.started = true;
        // Prefill the user turn; remember where the assistant's generation begins.
        let logits = self.llama.prefill(&toks, &mut self.kv)?;
        let answer_start = self.kv.len;
        let generated = self
            .llama
            .decode_loop(logits, &mut self.kv, max_new, on_token)?;

        // Keep only the ANSWER in history, not the model's <think>…</think> reasoning: Qwen3
        // explicitly excludes prior-turn thinking from context, and keeping it accumulates and
        // degrades the model (it starts emitting only-think then stopping). Rewind the cache to
        // before this generation and re-prefill just the answer (recomputed without the think in
        // context). Answer = tokens after the last </think>; only-think (unterminated) → keep none;
        // no <think> at all → keep everything (a direct answer).
        let tk = &self.llama.tokenizer;
        let close = tk.token_to_id("</think>");
        let open = tk.token_to_id("<think>");
        let answer: Vec<u32> = match close.and_then(|c| generated.iter().rposition(|&t| t == c)) {
            Some(pos) => generated[pos + 1..].to_vec(),
            None if open.is_some_and(|o| generated.contains(&o)) => Vec::new(),
            None => generated.clone(),
        };
        self.kv.len = answer_start; // drop the just-generated think+answer from the cache
        if !answer.is_empty() {
            self.llama.prefill(&answer, &mut self.kv)?;
        }
        self.llama
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }
}

// ---- host ops ----

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Incremental UTF-8-safe detokenizer: fed the FULL decoded text each step, returns the newly
/// completed suffix. Byte-level BPE splits a multi-byte char (e.g. an emoji) across tokens, so a
/// step's decode can end mid-character as U+FFFD (`�`); we hold output until it completes (decode no
/// longer ends in `�`), emitting whole characters only.
#[derive(Default)]
struct StreamDecoder {
    printed: usize,
}
impl StreamDecoder {
    fn step(&mut self, full: &str) -> String {
        if !full.ends_with('\u{FFFD}')
            && full.len() > self.printed
            && full.is_char_boundary(self.printed)
        {
            let delta = full[self.printed..].to_string();
            self.printed = full.len();
            delta
        } else {
            String::new()
        }
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

/// Sample a token id from `logits` per `s`. Greedy if `temp<=0`/`top_k==1`; else temperature +
/// top-k + top-p (nucleus). `rng` is an xorshift64 state advanced in place.
fn sample_logits(logits: &[f32], s: Sampler, rng: &mut u64) -> u32 {
    if s.temp <= 0.0 || s.top_k == 1 {
        return argmax(logits) as u32;
    }
    let n = logits.len();
    let k = if s.top_k == 0 { n } else { s.top_k.min(n) };
    let cmp = |a: &usize, b: &usize| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    let mut idx: Vec<usize> = (0..n).collect();
    if k < n {
        idx.select_nth_unstable_by(k - 1, cmp); // top-k at the front (unordered)
        idx.truncate(k);
    }
    idx.sort_unstable_by(cmp); // descending by logit
    let maxl = logits[idx[0]];
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - maxl) / s.temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    // nucleus: smallest prefix whose cumulative prob reaches top_p
    let mut cum = 0.0;
    let mut cutoff = probs.len();
    for (j, &p) in probs.iter().enumerate() {
        cum += p;
        if cum >= s.top_p {
            cutoff = j + 1;
            break;
        }
    }
    let total: f32 = probs[..cutoff].iter().sum();
    // xorshift64 → uniform [0, total)
    let mut x = *rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng = x;
    let u = (x >> 40) as f32 / (1u64 << 24) as f32;
    let r = u * total;
    let mut acc = 0.0;
    for j in 0..cutoff {
        acc += probs[j];
        if r <= acc {
            return idx[j] as u32;
        }
    }
    idx[cutoff - 1] as u32
}

fn rmsnorm_rows(x: &[f32], w: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        for i in 0..dim {
            y[r * dim + i] = row[i] * scale * w[i];
        }
    }
    y
}

/// ggml NORM rope (interleaved pairs (2i, 2i+1)), applied per head over the first `rope_dim` dims.
fn rope_rows(x: &mut [f32], t: usize, n_heads: usize, hd: usize, rope_dim: usize, theta: f32) {
    for pos in 0..t {
        for h in 0..n_heads {
            let base = (pos * n_heads + h) * hd;
            for i in 0..rope_dim / 2 {
                let freq = (theta as f64).powf(-2.0 * i as f64 / rope_dim as f64) as f32;
                let ang = pos as f32 * freq;
                let (s, co) = ang.sin_cos();
                let a = x[base + 2 * i];
                let b = x[base + 2 * i + 1];
                x[base + 2 * i] = a * co - b * s;
                x[base + 2 * i + 1] = a * s + b * co;
            }
        }
    }
}

/// Causal GQA attention. q [T, nh*hd], k/v [T, nkv*hd] -> out [T, nh*hd].
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
) -> Vec<f32> {
    let scale = 1.0 / (hd as f32).sqrt();
    let group = nh / nkv;
    let mut out = vec![0f32; t * nh * hd];
    for ti in 0..t {
        for h in 0..nh {
            let kvh = h / group;
            let qv = &q[(ti * nh + h) * hd..(ti * nh + h) * hd + hd];
            // scores over j in 0..=ti (causal)
            let mut scores = vec![0f32; ti + 1];
            let mut maxs = f32::NEG_INFINITY;
            for j in 0..=ti {
                let kv = &k[(j * nkv + kvh) * hd..(j * nkv + kvh) * hd + hd];
                let mut dot = 0f32;
                for d in 0..hd {
                    dot += qv[d] * kv[d];
                }
                dot *= scale;
                scores[j] = dot;
                if dot > maxs {
                    maxs = dot;
                }
            }
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - maxs).exp();
                sum += *s;
            }
            let ob = (ti * nh + h) * hd;
            for j in 0..=ti {
                let p = scores[j] / sum;
                let vv = &v[(j * nkv + kvh) * hd..(j * nkv + kvh) * hd + hd];
                for d in 0..hd {
                    out[ob + d] += p * vv[d];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod dequant_tests {
    use super::*;

    // ── IQ4_NL ──────────────────────────────────────────────────────────────────
    // Block: [half d][uint8 qs[16]], 32 elements, 18 bytes
    // y[j] = d * KVALUES_IQ4NL[qs[j] & 0xF]; y[j+16] = d * KVALUES_IQ4NL[qs[j] >> 4]
    // Reference: llama.cpp dequantize_row_iq4_nl (ggml-quants.c l.2653)
    #[test]
    fn iq4nl_single_block() {
        // d=1.0, qs[0]=0x80 (lo=0, hi=8)
        // KVALUES_IQ4NL[0] = -127, KVALUES_IQ4NL[8] = 1
        // y[0] = 1.0 * (-127) = -127.0
        // y[16] = 1.0 * 1 = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&d_bytes);
        block[2] = 0x80; // lo=0→-127, hi=8→1
        let y = dequant_codebook(infr_core::DType::Iq4Nl, &block);
        assert_eq!(y.len(), 32);
        assert!(
            (y[0] - (-127.0)).abs() < 1e-3,
            "iq4nl y[0] expected -127.0, got {}",
            y[0]
        );
        assert!(
            (y[16] - 1.0).abs() < 1e-3,
            "iq4nl y[16] expected 1.0, got {}",
            y[16]
        );
    }

    // ── IQ4_XS ──────────────────────────────────────────────────────────────────
    // Block: [half d][uint16 scales_h][uint8 scales_l[4]][uint8 qs[128]], 256 elements, 136 bytes
    // y = d*(ls-32) * KVALUES_IQ4NL[q4], ls is 6-bit per 32-elem sub-block
    // Reference: llama.cpp dequantize_row_iq4_xs (ggml-quants.c l.2671)
    #[test]
    fn iq4xs_single_block() {
        // d=1.0, scales: all sub-blocks have ls=32 → dl=d*(32-32)=0 → y=0
        // Verify: all 256 outputs are 0.0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 136];
        block[0..2].copy_from_slice(&d_bytes);
        // scales_h=0, scales_l=[0x00,0x00,0x00,0x00]: all lo=0, all hi=0 → ls=0 → dl=-32
        // Wait: ls=lo|(hi<<4). With scales_h=0 and scales_l=0, ls=0. dl=1.0*(0-32)=-32.
        // qs all 0: qs[j]&0xF=0 → KVALUES_IQ4NL[0]=-127; qs[j]>>4=0 → -127
        // y = -32 * (-127) = 4064.0 (all elements)
        let y = dequant_codebook(infr_core::DType::Iq4Xs, &block);
        assert_eq!(y.len(), 256);
        let expected = -32.0_f32 * KVALUES_IQ4NL[0] as f32; // 4064.0
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 0.5,
                "iq4xs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ1_S ───────────────────────────────────────────────────────────────────
    // Block: [half d][u8 qs[32]][u16 qh[8]], 50 bytes, QK_K=256
    // All-zero block: d=1.0, qh=0 → dl=1.0*(2*0+1)=1.0, delta=+0.125, grid_idx=0
    //   IQ1S_GRID[0] = 0xffffffffffffffff → gv=-1 for all j
    //   y[j] = 1.0 * (-1.0 + 0.125) = -0.875 for all 256 elements
    // Ref: llama.cpp dequantize_row_iq1_s (ggml-quants.c l.2578)
    #[test]
    fn iq1s_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 50];
        block[0..2].copy_from_slice(&d_bytes);
        // qs=0, qh=0 → grid_idx=0, dl=1.0, delta=+0.125
        let y = dequant_codebook(infr_core::DType::Iq1S, &block);
        assert_eq!(y.len(), 256);
        // IQ1S_GRID[0] = 0xffffffffffffffff → all bytes 0xFF = -1i8
        let expected = 1.0_f32 * (-1.0_f32 + IQ1S_DELTA);
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq1s y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── MXFP4 ───────────────────────────────────────────────────────────────────
    // Block: [u8 e][u8 qs[16]], 17 bytes, QK_MXFP4=32
    // e=128 → d=e8m0_to_fp32_half(128)=2^(128-128)=1.0; qs[0]=0x21 → lo=1, hi=2
    //   y[0] = KVALUES_MXFP4[1]*1.0 = 1.0; y[16] = KVALUES_MXFP4[2]*1.0 = 2.0
    // Ref: llama.cpp dequantize_row_mxfp4 (ggml-quants.c l.511)
    #[test]
    fn mxfp4_single_block() {
        let mut block = vec![0u8; 17];
        block[0] = 128; // e=128 → d=1.0
        block[1] = 0x21; // lo nibble=1→1, hi nibble=2→2
        let y = dequant_codebook(infr_core::DType::Mxfp4, &block);
        assert_eq!(y.len(), 32);
        assert!(
            (y[0] - 1.0).abs() < 1e-5,
            "mxfp4 y[0] expected 1.0, got {}",
            y[0]
        );
        assert!(
            (y[16] - 2.0).abs() < 1e-5,
            "mxfp4 y[16] expected 2.0, got {}",
            y[16]
        );
        // rest of qs=0 → x0=x1=0 → y=0.0
        for i in 1..16 {
            assert!(y[i].abs() < 1e-5, "mxfp4 y[{i}] expected 0.0, got {}", y[i]);
        }
    }

    // ── NVFP4 ───────────────────────────────────────────────────────────────────
    // Block: [u8 d[4]][u8 qs[32]], 36 bytes, QK_NVFP4=64
    // All-zero scales: d=ue4m3_to_fp32(0)=0.0 → all y=0.0
    // Ref: llama.cpp dequantize_row_nvfp4 (ggml-quants.c l.531)
    #[test]
    fn nvfp4_single_block() {
        let block = vec![0u8; 36];
        let y = dequant_codebook(infr_core::DType::Nvfp4, &block);
        assert_eq!(y.len(), 64);
        for i in 0..64 {
            assert!(y[i].abs() < 1e-5, "nvfp4 y[{i}] expected 0.0, got {}", y[i]);
        }
    }

    // ── IQ1_M ───────────────────────────────────────────────────────────────────
    // Block: [u8 qs[32]][u8 qh[16]][u8 scales[8]], 56 bytes, QK_K=256
    // All-zero: scales=0 → d_bits=0 → d=0.0 → all y=0.0
    // Ref: llama.cpp dequantize_row_iq1_m (ggml-quants.c l.2603)
    #[test]
    fn iq1m_single_block() {
        let block = vec![0u8; 56];
        let y = dequant_codebook(infr_core::DType::Iq1M, &block);
        assert_eq!(y.len(), 256);
        for i in 0..256 {
            assert!(y[i].abs() < 1e-4, "iq1m y[{i}] expected 0.0, got {}", y[i]);
        }
    }

    // ── TQ1_0 ───────────────────────────────────────────────────────────────────
    // Block: [u8 qs[48]][u8 qh[4]][half d], 54 bytes, QK_K=256
    // All-zero qs/qh: q=0 → xi=0 → y=(0-1)*d = -d for all 256 elements
    // Ref: llama.cpp dequantize_row_tq1_0 (ggml-quants.c l.2356)
    #[test]
    fn tq1_0_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 54];
        block[52..54].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Tq1_0, &block);
        assert_eq!(y.len(), 256);
        for i in 0..256 {
            assert!(
                (y[i] - (-1.0)).abs() < 1e-4,
                "tq1_0 y[{i}] expected -1.0, got {}",
                y[i]
            );
        }
    }

    // ── TQ2_0 ───────────────────────────────────────────────────────────────────
    // Block: [u8 qs[64]][half d], 66 bytes, QK_K=256
    // All-zero qs: q=(0>>l*2)&3=0 → y=(0-1)*d = -d for all 256 elements
    // Ref: llama.cpp dequantize_row_tq2_0 (ggml-quants.c l.2395)
    #[test]
    fn tq2_0_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 66];
        block[64..66].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Tq2_0, &block);
        assert_eq!(y.len(), 256);
        for i in 0..256 {
            assert!(
                (y[i] - (-1.0)).abs() < 1e-4,
                "tq2_0 y[{i}] expected -1.0, got {}",
                y[i]
            );
        }
    }

    // ── IQ2_XXS ─────────────────────────────────────────────────────────────────
    // Block: [half d][uint16 qs[32]], 66 bytes, QK_K=256
    // Sub-block 0: aux0=0 → 4 grid indices all 0; aux1=0 → scale_mag=0, sign_idx=0
    //   IQ2XXS_GRID[0] = 0x0808080808080808 → 8 bytes all 0x08
    //   KSIGNS_IQ2XS[0] = 0 → no negations
    //   db = 1.0*(0.5+0)*0.25 = 0.125
    //   y = 0.125 * 8 = 1.0 for each of 8 elements × 4 groups = 32 elements
    // Ref: llama.cpp dequantize_row_iq2_xxs (ggml-quants.c l.2416)
    #[test]
    fn iq2xxs_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 66];
        block[0..2].copy_from_slice(&d_bytes);
        // all qs = 0: aux0=0 (grid_idx=0), aux1=0 (scale_mag=0, sign_idx=0)
        let y = dequant_codebook(infr_core::DType::Iq2Xxs, &block);
        assert_eq!(y.len(), 256);
        // first sub-block, first element
        let expected = 0.125 * 8.0_f32;
        for i in 0..32 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq2xxs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
        // remaining sub-blocks: same pattern (all zeros)
        for i in 32..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq2xxs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ2_XS ──────────────────────────────────────────────────────────────────
    // Block: [half d][uint16 qs[32]][uint8 scales[8]], 74 bytes, QK_K=256
    // All zeros: scales[0]=0 → db0=db1=0.125; qs16=0 → grid_idx=0, sign_idx=0
    //   IQ2XS_GRID[0] = 0x0808080808080808 → gv=8; KSIGNS[0]=0 → +1
    //   y = 0.125 * 8 = 1.0 for first 32 elements
    // Ref: llama.cpp dequantize_row_iq2_xs (ggml-quants.c l.2444)
    #[test]
    fn iq2xs_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 74];
        block[0..2].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Iq2Xs, &block);
        assert_eq!(y.len(), 256);
        let expected = 0.125 * 8.0_f32;
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq2xs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ2_S ───────────────────────────────────────────────────────────────────
    // Block: [half d][u8 qs[64]][u8 qh[8]][u8 scales[8]], 82 bytes, QK_K=256
    // All zeros: scales=0 → db0=db1=0.125; qs_all[0]=0, qh[0]=0 → grid_idx=0
    //   IQ2S_GRID[0] = 0x0808080808080808 → gv=8; signs[32]=0 → +1
    //   y = 0.125 * 8 = 1.0 for all 256 elements
    // Ref: llama.cpp dequantize_row_iq2_s (ggml-quants.c l.2471)
    #[test]
    fn iq2s_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 82];
        block[0..2].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Iq2S, &block);
        assert_eq!(y.len(), 256);
        let expected = 0.125 * 8.0_f32;
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq2s y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ3_XXS ─────────────────────────────────────────────────────────────────
    // Block: [half d][u8 qs[96]], 98 bytes, QK_K=256
    // qs[0..64]=0 → grid_idx=0 for all; qs[64..96]=0 → aux32=0 → scale_mag=0, sign_idx=0
    //   IQ3XXS_GRID[0] = 0x04040404 → gv for j=0..3: 4; KSIGNS[0]=0 → +1
    //   db = 1.0*(0.5+0)*0.5 = 0.25
    //   y = 0.25 * 4 = 1.0 for first 32 elements
    // Ref: llama.cpp dequantize_row_iq3_xxs (ggml-quants.c l.2503)
    #[test]
    fn iq3xxs_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 98];
        block[0..2].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Iq3Xxs, &block);
        assert_eq!(y.len(), 256);
        let expected = 0.25 * 4.0_f32;
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq3xxs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ3_S ───────────────────────────────────────────────────────────────────
    // Block: [half d][u8 qs[64]][u8 qh[8]][u8 signs[32]][u8 scales[4]], 110 bytes
    // All zeros: scales=0 → db1=db2=1.0*(1+2*0)=1.0; qs=0, qh=0 → grid_idx=0
    //   IQ3S_GRID[0] = 0x01010101 → gv for j=0..3: 1; signs[0]=0 → +1
    //   y = 1.0 * 1 = 1.0 for all 256 elements
    // Ref: llama.cpp dequantize_row_iq3_s (ggml-quants.c l.2535)
    #[test]
    fn iq3s_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Iq3S, &block);
        assert_eq!(y.len(), 256);
        let expected = 1.0_f32;
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq3s y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── Q2_K ────────────────────────────────────────────────────────────────────
    // Block: [uint8 scales[16]][uint8 qs[64]][half d][half dmin]
    // y = d*(sc&0xF)*q2 - dmin*(sc>>4), q2 ∈ 0..3
    // Reference: llama.cpp dequantize_row_q2_K (ggml-quants.c l.903)
    #[test]
    fn q2k_single_block() {
        // d=1.0, dmin=2.0
        // scales[0]=0x23 → lo=3, hi=2 → first sub-block: dl=3.0, ml=4.0
        // scales[1]=0x23 → second 16-elem sub-block (qs[16..32]): same dl/ml
        // qs[0..16]=0x55 → q2 (shift=0) = 0x55 & 3 = 1
        // Expected y[0] = 3.0*1 - 4.0 = -1.0
        let mut block = vec![0u8; 84];
        // scales[0..16]
        block[0] = 0x23; // lo=3, hi=2
        block[1] = 0x23; // same for second sub-block
                         // qs[16..80]
        for b in &mut block[16..80] {
            *b = 0x55; // any bits; q2 at shift=0 for first 16 = 1
        }
        // d[80..82] = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        block[80..82].copy_from_slice(&d_bytes);
        // dmin[82..84] = 2.0
        let dmin_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        block[82..84].copy_from_slice(&dmin_bytes);

        let y = dequant_block(infr_core::DType::Q2K, &block).unwrap();
        assert_eq!(y.len(), 256);
        // First sub-block, first element: q2=1, y=3.0*1-4.0=-1.0
        assert!(
            (y[0] - (-1.0)).abs() < 1e-4,
            "q2k y[0] expected -1.0, got {}",
            y[0]
        );
        // All elements in first sub-block same q2=1 → same y
        for i in 0..16 {
            assert!(
                (y[i] - (-1.0)).abs() < 1e-4,
                "q2k y[{i}] expected -1.0, got {}",
                y[i]
            );
        }
        // Second sub-block (16..32): same scales, qs=0x55, q2=(0x55>>2)&3=(0x15)&3=1
        // Wait: shift=0 for j=0 applies to BOTH first and second 16-elem groups of the
        // same j-iteration. Let me re-check the llama logic.
        // In the llama code, for j=0 (shift=0):
        //   sc=scales[0], dl=d*(sc&0xF)=3, ml=dmin*(sc>>4)=4
        //   for l in 0..16: q2 = (q[l] >> 0) & 3 = qs[l] & 3 = 0x55 & 3 = 1
        //   sc=scales[1], dl=d*(sc&0xF)=3, ml=dmin*(sc>>4)=4
        //   for l in 0..16: q2 = (q[l+16] >> 0) & 3 = qs[l+16] & 3 = 1
        // So elements 16..32 also have dl=3, ml=4, q2=1 → y=-1.0
        assert!(
            (y[16] - (-1.0)).abs() < 1e-4,
            "q2k y[16] expected -1.0, got {}",
            y[16]
        );
    }

    // ── Q3_K ────────────────────────────────────────────────────────────────────
    // Block: [uint8 hmask[32]][uint8 qs[64]][uint8 scales[12]][half d]
    // y = d*(sc6-32)*(q3u - 4), q3u = (low2 | high_bit<<2) ∈ 0..7
    // Reference: llama.cpp dequantize_row_q3_K (ggml-quants.c l.1247)
    #[test]
    fn q3k_single_block() {
        // d=1.0
        // Choose scales to decode as sc6=36 for all sub-blocks → sc6-32=4 → dl=4.0
        // Encode sc6=36 for first sub-block in scales_raw:
        //   After decode, aux bytes give sc6 values. Simpler: set all scales[0..12]=0
        //   so that after bit manipulation aux has all-zero lower nibbles → sc6=0 for all.
        //   Then dl=0 → y=0 everywhere. That's a trivial test.
        //
        // Better: set scales bytes to give sc6=32 for first two sub-blocks (dl=0, y=0)
        // and verify that y[0..32]=0. Then set hmask and qs to anything.
        //
        // Even simpler: set scales_raw all-zero. After bit manipulation:
        //   aux[0]=0, aux[1]=0, aux[2]=0, aux[3]=0
        //   sc6(0)= aux[0] byte0 = 0 → sc6-32 = -32 → dl=-32
        //   hmask[0..16]=0 (high bit=0), qs[0..16]=0x00 (low2=0 at shift=0)
        //   q3u = 0 | (0<<2) = 0. y = -32*0 + (-4)*(-32) = 128... wait
        //   y = dl*q3u + (-4*dl) = -32*0 + (-4*(-32)) = 128
        //
        // Let me verify this explicitly:
        //   q3u=0, dl=-32, min=-4*dl=128. y = -32*0 + 128 = 128. ✓
        //
        // Alternatively: set scales_raw to encode sc6=32 for sub-block 0.
        //   When tmp=aux[2]=0, aux[0]=scales_bytes[0..4] as u32.
        //   For sc6=32 after decode:
        //     sc6(0) = (aux[0] & 0xFF) = 32 → need aux[0] byte 0 = 32 = 0x20
        //     After bit manip (tmp=0): aux[0] = (orig_aux0 & 0x0F0F0F0F) | ...
        //     So (orig_aux0 & 0xF) = 32? 32 > 15, so the lower 4 bits can't encode 32.
        //
        // The scale decoding is complex. Let me just use all-zero scales (sc6=0, dl=-32*1=-32)
        // with hmask[0..16]=0 and qs[0..16]=0x00:
        // y = -32*0 + (-4*(-32)) = 128.0
        let mut block = vec![0u8; 110];
        // hmask[0..32] = all 0 (high bit not set for any elem)
        // qs[32..96] = all 0 (low2=0 at any shift)
        // scales[96..108] = all 0 (encodes sc6=0 after bit manipulation → dl=-32)
        // d[108..110] = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        block[108..110].copy_from_slice(&d_bytes);

        let y = dequant_block(infr_core::DType::Q3K, &block).unwrap();
        assert_eq!(y.len(), 256);
        // sc6=0 → dl = 1.0*(0-32) = -32.0
        // q3u = 0 (hmask=0, qs=0), min = -4*(-32) = 128
        // y[0] = -32*0 + 128 = 128.0
        assert!(
            (y[0] - 128.0).abs() < 1e-3,
            "q3k y[0] expected 128.0, got {}",
            y[0]
        );
        // All elements should be 128.0 (same scale, q3u=0 everywhere)
        for i in 0..256 {
            assert!(
                (y[i] - 128.0).abs() < 1e-3,
                "q3k y[{i}] expected 128.0, got {}",
                y[i]
            );
        }
    }

    // ── Q4_0 ────────────────────────────────────────────────────────────────────
    // Block: [half d][uint8 qs[16]]; y = d * (q4 - 8), q4 ∈ 0..15
    // Reference: llama.cpp dequantize_row_q4_0 (ggml-quants.c l.401)
    #[test]
    fn q4_0_single_block() {
        // d = 2.0 (f16 = 0x4000), qs[0] = 0x89 (lo=9, hi=8), rest = 0x88 (lo=8, hi=8)
        let d_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&d_bytes);
        block[2] = 0x89; // qs[0]: lo=9, hi=8
        for b in &mut block[3..18] {
            *b = 0x88; // lo=8, hi=8 → y = d*(8-8) = 0
        }
        let y = dequant_block(infr_core::DType::Q4_0, &block).unwrap();
        assert_eq!(y.len(), 32);
        // y[0] = 2.0*(9-8) = 2.0
        assert!(
            (y[0] - 2.0).abs() < 1e-5,
            "q4_0 y[0] expected 2.0, got {}",
            y[0]
        );
        // y[16] = 2.0*(8-8) = 0.0
        assert!(y[16].abs() < 1e-5, "q4_0 y[16] expected 0.0, got {}", y[16]);
        // y[1] = 2.0*(8-8) = 0.0
        assert!(y[1].abs() < 1e-5, "q4_0 y[1] expected 0.0, got {}", y[1]);
    }

    // ── Q4_1 ────────────────────────────────────────────────────────────────────
    // Block: [half d][half m][uint8 qs[16]]; y = d*q4 + m, q4 ∈ 0..15
    // Reference: llama.cpp dequantize_row_q4_1 (ggml-quants.c l.421)
    #[test]
    fn q4_1_single_block() {
        // d=1.0, m=0.5, qs[0]=0x30 (lo=0, hi=3), rest=0x00
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let m_bytes = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&d_bytes);
        block[2..4].copy_from_slice(&m_bytes);
        block[4] = 0x30; // lo=0, hi=3
        let y = dequant_block(infr_core::DType::Q4_1, &block).unwrap();
        assert_eq!(y.len(), 32);
        // y[0] = 1.0*0 + 0.5 = 0.5
        assert!(
            (y[0] - 0.5).abs() < 1e-4,
            "q4_1 y[0] expected 0.5, got {}",
            y[0]
        );
        // y[16] = 1.0*3 + 0.5 = 3.5
        assert!(
            (y[16] - 3.5).abs() < 1e-4,
            "q4_1 y[16] expected 3.5, got {}",
            y[16]
        );
    }

    // ── Q5_0 ────────────────────────────────────────────────────────────────────
    // Block: [half d][uint8 qh[4]][uint8 qs[16]]; y = d*(q5 - 16), q5 ∈ 0..31
    // Reference: llama.cpp dequantize_row_q5_0 (ggml-quants.c l.442)
    #[test]
    fn q5_0_single_block() {
        // d=1.0, qh=[0x01,0,0,0] (bit 0 → element 0 gets high bit → q5=15|16=31)
        // qs[0]=0x0F (lo=15, hi=0), rest=0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&d_bytes);
        block[2] = 0x01; // qh[0]: bit 0 set
        block[6] = 0x0F; // qs[0]: lo=15, hi=0
        let y = dequant_block(infr_core::DType::Q5_0, &block).unwrap();
        assert_eq!(y.len(), 32);
        // j=0: xh0 = ((1>>0)<<4)&0x10 = 16. q5 = 15|16=31. y[0] = 1.0*(31-16) = 15.0
        assert!(
            (y[0] - 15.0).abs() < 1e-5,
            "q5_0 y[0] expected 15.0, got {}",
            y[0]
        );
        // j=0: xh1 = (1>>12)&0x10 = 0. q5 = 0. y[16] = 1.0*(0-16) = -16.0
        assert!(
            (y[16] - (-16.0)).abs() < 1e-5,
            "q5_0 y[16] expected -16.0, got {}",
            y[16]
        );
    }

    // ── Q5_1 ────────────────────────────────────────────────────────────────────
    // Block: [half d][half m][uint8 qh[4]][uint8 qs[16]]; y = d*q5 + m, q5 ∈ 0..31
    // Reference: llama.cpp dequantize_row_q5_1 (ggml-quants.c l.468)
    #[test]
    fn q5_1_single_block() {
        // d=2.0, m=-1.0, qh=[0,0,0,0], qs[0]=0x1F (lo=15, hi=1)
        let d_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let m_bytes = half::f16::from_f32(-1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&d_bytes);
        block[2..4].copy_from_slice(&m_bytes);
        // qh[4] all zero → no high bits
        block[8] = 0x1F; // qs[0]: lo=15, hi=1
        let y = dequant_block(infr_core::DType::Q5_1, &block).unwrap();
        assert_eq!(y.len(), 32);
        // y[0] = 2.0*15 + (-1.0) = 29.0
        assert!(
            (y[0] - 29.0).abs() < 1e-4,
            "q5_1 y[0] expected 29.0, got {}",
            y[0]
        );
        // y[16] = 2.0*1 + (-1.0) = 1.0
        assert!(
            (y[16] - 1.0).abs() < 1e-4,
            "q5_1 y[16] expected 1.0, got {}",
            y[16]
        );
    }
}

/// Phase 3: validate that the full dequant_unified → pack_unified → GPU linear_q pipeline
/// produces the same result as the CPU dequant for each new affine quant type.
#[cfg(test)]
mod gpu_affine_tests {
    use super::*;
    use infr_core::backend::BufferUsage;
    use infr_vulkan::VulkanBackend;

    /// Run `linear_q` on the GPU for a single-block weight, compare to CPU.
    fn check_gpu_affine(dtype: infr_core::DType, block_bytes: &[u8]) {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        // dequant + pack on CPU
        let (bits, blk) = quant_params(dtype);
        let (qv, sc, mn) = dequant_unified(dtype, block_bytes);
        let (q_packed, s_packed, m_packed) = pack_unified(&qv, &sc, &mn, bits, blk);
        let numel = qv.len();

        // input: one row of `numel` f32 activations, all 1.0 (sum = dot(w, 1) = sum of weights)
        let x: Vec<f32> = vec![1.0f32; numel];
        let bq = be
            .upload_weight_bytes(bytemuck::cast_slice(&q_packed))
            .unwrap();
        let bs = be
            .upload_weight_bytes(bytemuck::cast_slice(&s_packed))
            .unwrap();
        let bm = be
            .upload_weight_bytes(bytemuck::cast_slice(&m_packed))
            .unwrap();
        let upx = be.alloc(x.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(upx.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let by = be.alloc(1 * 4, BufferUsage::Readback).unwrap(); // 1 output row, 1 out feature

        // rows=1, in_f=numel, out_f=1 → single dot product
        let rec = be.recorder().unwrap();
        rec.linear_q(
            bq.as_ref(),
            bs.as_ref(),
            bm.as_ref(),
            upx.as_ref(),
            by.as_ref(),
            1,
            numel,
            1,
            bits,
            blk.trailing_zeros(),
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; 4];
        be.download(by.as_ref(), &mut bytes).unwrap();
        let gpu_out: f32 = bytemuck::cast_slice(&bytes)[0];

        // CPU reference: sum of all dequantized weights (dot with all-ones)
        let cpu_out: f32 = (0..numel)
            .map(|g| sc[g] * qv[g] as f32 + mn[g])
            .sum::<f32>();
        let err = (gpu_out - cpu_out).abs();
        let rel = err / (cpu_out.abs() + 1e-6);
        assert!(
            rel < 5e-3,
            "{dtype:?} GPU vs CPU: gpu={gpu_out} cpu={cpu_out} err={err} rel={rel}"
        );
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q4_0_gpu_matches_cpu() {
        // d=2.0, qs all=0x89 (lo=9,hi=8) → y[0..16]=d*(9-8)=2, y[16..32]=d*(8-8)=0
        let d_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&d_bytes);
        for b in &mut block[2..18] {
            *b = 0x89;
        }
        check_gpu_affine(infr_core::DType::Q4_0, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q4_1_gpu_matches_cpu() {
        // d=1.0, m=0.5, qs all=0x31 (lo=1,hi=3) → y[0..16]=1.5, y[16..32]=3.5
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let m_bytes = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&d_bytes);
        block[2..4].copy_from_slice(&m_bytes);
        for b in &mut block[4..20] {
            *b = 0x31;
        }
        check_gpu_affine(infr_core::DType::Q4_1, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q5_0_gpu_matches_cpu() {
        // d=1.0, qh=0, qs all=0x0A (lo=10,hi=0) → y=d*(10-16)=-6, y[16..]=d*(0-16)=-16
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&d_bytes);
        for b in &mut block[6..22] {
            *b = 0x0A;
        }
        check_gpu_affine(infr_core::DType::Q5_0, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q5_1_gpu_matches_cpu() {
        // d=1.0, m=2.0, qh=0, qs all=0x1F (lo=15,hi=1) → y[0..16]=17, y[16..32]=3
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let m_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&d_bytes);
        block[2..4].copy_from_slice(&m_bytes);
        for b in &mut block[8..24] {
            *b = 0x1F;
        }
        check_gpu_affine(infr_core::DType::Q5_1, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q2k_gpu_matches_cpu() {
        // Minimal 256-elem block: d=1.0, dmin=0, scales[0..2]=0x03 (lo=3,hi=0), qs=0x55
        let mut block = vec![0u8; 84];
        block[0] = 0x03;
        block[1] = 0x03;
        for b in &mut block[16..80] {
            *b = 0x55;
        }
        block[80..82].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        check_gpu_affine(infr_core::DType::Q2K, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q3k_gpu_matches_cpu() {
        // All-zero block except d=1.0 → all elements are 128.0 (see q3k unit test)
        let mut block = vec![0u8; 110];
        block[108..110].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        check_gpu_affine(infr_core::DType::Q3K, &block);
    }

    // ── Native-block GPU-vs-CPU parity tests (Phase 0-2) ────────────────────
    //
    // Each test: build a known raw block, run `linear_native` GEMV with x=all-1.0,
    // compare to `dequant_unified`/`dequant_codebook` CPU sum (dot with 1.0 = weight sum).
    // Also compare against `linear_q` (Wt::Q) for affine quants to prove parity.

    fn check_native(dtype: infr_core::DType, block_bytes: &[u8]) {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        use infr_vulkan::linear::pad_to_u32_align;

        // CPU reference: sum of dequantized weights (dot with all-1.0 input)
        let (qv, sc, mn) = dequant_unified(dtype, block_bytes);
        let numel = qv.len();
        let cpu_out: f32 = (0..numel).map(|g| sc[g] * qv[g] as f32 + mn[g]).sum();

        // Upload native raw block bytes (padded to u32)
        let padded = pad_to_u32_align(block_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let x: Vec<f32> = vec![1.0f32; numel];
        let xbuf = be.alloc(x.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(4, BufferUsage::Readback).unwrap();

        let rec = be.recorder().unwrap();
        rec.linear_native(
            dtype,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            numel,
            1,
        );
        rec.finish().unwrap();

        let mut out_bytes = vec![0u8; 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_out: f32 = bytemuck::cast_slice(&out_bytes)[0];

        let err = (gpu_out - cpu_out).abs();
        let rel = err / (cpu_out.abs() + 1e-6);
        assert!(
            rel < 5e-3,
            "{dtype:?} native GPU vs CPU: gpu={gpu_out} cpu={cpu_out} err={err} rel={rel}"
        );

        // Parity with Wt::Q unified path (for affine quants that support it)
        if is_quant(dtype) {
            let (bits, blk) = quant_params(dtype);
            let (qv2, sc2, mn2) = dequant_unified(dtype, block_bytes);
            let (q_packed, s_packed, m_packed) = pack_unified(&qv2, &sc2, &mn2, bits, blk);
            let bq = be
                .upload_weight_bytes(bytemuck::cast_slice(&q_packed))
                .unwrap();
            let bs = be
                .upload_weight_bytes(bytemuck::cast_slice(&s_packed))
                .unwrap();
            let bm = be
                .upload_weight_bytes(bytemuck::cast_slice(&m_packed))
                .unwrap();
            let xbuf2 = be.alloc(x.len() * 4, BufferUsage::Staging).unwrap();
            be.upload(xbuf2.as_ref(), bytemuck::cast_slice(&x)).unwrap();
            let ybuf2 = be.alloc(4, BufferUsage::Readback).unwrap();
            let rec2 = be.recorder().unwrap();
            rec2.linear_q(
                bq.as_ref(),
                bs.as_ref(),
                bm.as_ref(),
                xbuf2.as_ref(),
                ybuf2.as_ref(),
                1,
                numel,
                1,
                bits,
                blk.trailing_zeros(),
            );
            rec2.finish().unwrap();
            let mut out2 = vec![0u8; 4];
            be.download(ybuf2.as_ref(), &mut out2).unwrap();
            let q_out: f32 = bytemuck::cast_slice(&out2)[0];
            let err2 = (gpu_out - q_out).abs();
            let rel2 = err2 / (q_out.abs() + 1e-6);
            assert!(
                rel2 < 5e-3,
                "{dtype:?} native vs unified-Q: native={gpu_out} q={q_out} err={err2} rel={rel2}"
            );
        }
    }

    // ── Phase 0: Q8_0 ────────────────────────────────────────────────────────

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q8_0_native_matches_cpu() {
        // d=1.5, qs: bytes 0..32 = signed values -128..127 cycling
        let d_bits = half::f16::from_f32(1.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&d_bits);
        for i in 0..32u8 {
            // values: 0,1,..,127,-128,-127,...,-97 → will cycle through positive and negative
            block[2 + i as usize] = i.wrapping_add(100); // e.g. 100,101,..,127,-128,...
        }
        check_native(infr_core::DType::Q8_0, &block);
    }

    // ── Phase 1: Q4_0, Q4_1, Q5_0, Q5_1 ─────────────────────────────────────

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q4_0_native_matches_cpu() {
        // d=2.0, qs all=0x89 (lo=9,hi=8) → mix of positive/negative after -8
        let d_bits = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&d_bits);
        for b in &mut block[2..18] {
            *b = 0x89;
        }
        check_native(infr_core::DType::Q4_0, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q4_1_native_matches_cpu() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let m_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&m_bits);
        for b in &mut block[4..20] {
            *b = 0x31;
        }
        check_native(infr_core::DType::Q4_1, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q5_0_native_matches_cpu() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&d_bits);
        // qh=0 (no high bits), qs all=0x0A → q5 values 10 (lo) and 0 (hi)
        for b in &mut block[6..22] {
            *b = 0x0A;
        }
        check_native(infr_core::DType::Q5_0, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q5_1_native_matches_cpu() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let m_bits = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&m_bits);
        for b in &mut block[8..24] {
            *b = 0x1F;
        }
        check_native(infr_core::DType::Q5_1, &block);
    }

    // ── Phase 2: k-quants ─────────────────────────────────────────────────────

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q2k_native_matches_cpu() {
        let mut block = vec![0u8; 84];
        block[0] = 0x03;
        block[1] = 0x03;
        for b in &mut block[16..80] {
            *b = 0x55;
        }
        block[80..82].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        check_native(infr_core::DType::Q2K, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q3k_native_matches_cpu() {
        let mut block = vec![0u8; 110];
        block[108..110].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        check_native(infr_core::DType::Q3K, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q4k_native_matches_cpu() {
        // d=1.0, dmin=0.5, scales[0]=0x33 → sc=3, mn=3
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let dmin_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&dmin_bits);
        // scales[4..16]: all 0x33 → k4(0)=(3,3) for first sub-block
        for b in &mut block[4..16] {
            *b = 0x33;
        }
        // qs: alternating 0xAB
        for b in &mut block[16..144] {
            *b = 0xAB;
        }
        check_native(infr_core::DType::Q4K, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q5k_native_matches_cpu() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let dmin_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&dmin_bits);
        for b in &mut block[4..16] {
            *b = 0x33;
        }
        for b in &mut block[48..176] {
            *b = 0xAB;
        }
        check_native(infr_core::DType::Q5K, &block);
    }

    /// Non-uniform Q5K block: distinct scales per sub-block + non-zero qh.
    /// The uniform tests above are insensitive to indexing bugs; this one is not.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q5k_native_nonuniform() {
        // Build a block where each sub-block has a different scale and qh is varied.
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let dmin_bits = half::f16::from_f32(0.25).to_bits().to_le_bytes();
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&dmin_bits);
        // scales[0..12]: encode 8 distinct 6-bit (scale,min) pairs via k4 encoding.
        // Use simple encoding: first 4 bytes = low bits of sc (i=0..3), bytes 4..8 = low bits of mn,
        // bytes 8..12 = upper bits mixed.
        // Set them to varied values so each sub-block has a different scale.
        block[4] = 0x20; // k4(0): sc=0x20&0x3F=32, mn=block[8]&0x3F
        block[5] = 0x10; // k4(2): sc=16, mn=...
        block[6] = 0x08; // k4(4): sc computed via else branch
        block[7] = 0x04; // k4(6): sc computed via else branch
        block[8] = 0x3F; // k4(0): mn=63
        block[9] = 0x2A; // k4(2): mn=42
        block[10] = 0x15; // k4(4): (used in else branch)
        block[11] = 0x09; // k4(6): (used in else branch)
                          // block[12..16] could affect k4(4..7) upper bits; set to varied pattern
        block[12] = 0xC0; // affects k4(4): sc upper bits from (block[8]>>6)<<4 = (0x3F>>6)<<4=0
        block[13] = 0x80;
        block[14] = 0x40;
        block[15] = 0x20;
        // qh: set to varied pattern so high bits vary
        for i in 0..32usize {
            block[16 + i] = (i as u8).wrapping_mul(17).wrapping_add(1);
        }
        // qs: set to varied pattern
        for i in 0..128usize {
            block[48 + i] = (i as u8).wrapping_mul(13).wrapping_add(7);
        }
        check_native(infr_core::DType::Q5K, &block);
    }

    /// Non-uniform Q6K block: distinct scales per sub-block.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q6k_native_nonuniform() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 210];
        // ql: varied
        for i in 0..128usize {
            block[i] = (i as u8).wrapping_mul(11).wrapping_add(3);
        }
        // qh: varied
        for i in 0..64usize {
            block[128 + i] = (i as u8).wrapping_mul(7).wrapping_add(5);
        }
        // scales: varied signed int8 values (avoid extreme negatives to keep sums finite)
        for i in 0..16usize {
            block[192 + i] = ((i as u8).wrapping_mul(5) + 8) & 0x7F;
        } // positive only
        block[208..210].copy_from_slice(&d_bits);
        check_native(infr_core::DType::Q6K, &block);
    }

    /// Multi-block Q5K test: 4 blocks (in_f=1024), out_f=2. Tests cross-block access.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q5k_native_multiblock() {
        use infr_vulkan::linear::pad_to_u32_align;
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan");
                return;
            }
        };
        // Build 8 distinct Q5K blocks (in_f=2048, out_f=2 → weight matrix [2, 2048])
        const N_BLOCKS: usize = 8;
        const BLOCK_SZ: usize = 176;
        const NELEMS: usize = 256;
        const IN_F: usize = N_BLOCKS * NELEMS;
        const OUT_F: usize = 2;
        // Total weight elements: OUT_F * IN_F = 2 * 2048 = 4096 = 16 blocks
        const TOTAL_BLOCKS: usize = OUT_F * IN_F / NELEMS; // = OUT_F * N_BLOCKS
        let mut w_bytes = vec![0u8; TOTAL_BLOCKS * BLOCK_SZ];
        // Fill blocks with distinct, varied data
        for b in 0..TOTAL_BLOCKS {
            let off = b * BLOCK_SZ;
            let d_bits = half::f16::from_f32(0.5 + b as f32 * 0.1)
                .to_bits()
                .to_le_bytes();
            let dmin_bits = half::f16::from_f32(0.1).to_bits().to_le_bytes();
            w_bytes[off..off + 2].copy_from_slice(&d_bits);
            w_bytes[off + 2..off + 4].copy_from_slice(&dmin_bits);
            for i in 0..12 {
                w_bytes[off + 4 + i] = ((b * 12 + i) as u8).wrapping_mul(3) | 0x20;
            }
            for i in 0..32 {
                w_bytes[off + 16 + i] = ((b * 32 + i) as u8).wrapping_mul(17);
            }
            for i in 0..128 {
                w_bytes[off + 48 + i] = ((b * 128 + i) as u8).wrapping_mul(7).wrapping_add(3);
            }
        }
        // CPU reference: compute expected outputs using dequant_unified
        let mut cpu_outputs = vec![0f32; OUT_F];
        let x: Vec<f32> = (0..IN_F).map(|i| 1.0f32 + i as f32 * 0.001f32).collect();
        for o in 0..OUT_F {
            let w_row_bytes = &w_bytes[o * N_BLOCKS * BLOCK_SZ..(o + 1) * N_BLOCKS * BLOCK_SZ];
            let (qv, sc, mn) = dequant_unified(infr_core::DType::Q5K, w_row_bytes);
            let sum: f32 = (0..IN_F)
                .map(|i| (sc[i] * qv[i] as f32 + mn[i]) * x[i])
                .sum();
            cpu_outputs[o] = sum;
        }
        // GPU: upload and run
        let padded = pad_to_u32_align(&w_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let xbuf = be.alloc(IN_F * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(OUT_F * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q5K,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            IN_F,
            OUT_F,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; OUT_F * 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_outputs: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec();
        for o in 0..OUT_F {
            let err = (gpu_outputs[o] - cpu_outputs[o]).abs();
            let rel = err / (cpu_outputs[o].abs() + 1e-3);
            assert!(
                rel < 5e-3,
                "Q5K out[{o}]: gpu={} cpu={} err={err} rel={rel}",
                gpu_outputs[o],
                cpu_outputs[o]
            );
        }
    }

    /// Full-scale Q6K test matching ffn_down dimensions: out_f=1024, in_f=3072.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q6k_native_fullscale() {
        use infr_vulkan::linear::pad_to_u32_align;
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan");
                return;
            }
        };
        const BLOCK_SZ: usize = 210;
        const NELEMS: usize = 256;
        const IN_F: usize = 3072;
        const OUT_F: usize = 1024;
        let n_blocks_per_row = IN_F / NELEMS; // 12
        let total_blocks = OUT_F * n_blocks_per_row;
        let mut w_bytes = vec![0u8; total_blocks * BLOCK_SZ];
        for b in 0..total_blocks {
            let off = b * BLOCK_SZ;
            let d_bits = half::f16::from_f32(0.1 + (b % 16) as f32 * 0.05)
                .to_bits()
                .to_le_bytes();
            for i in 0..128 {
                w_bytes[off + i] = ((b * 7 + i) as u8).wrapping_mul(11);
            }
            for i in 0..64 {
                w_bytes[off + 128 + i] = ((b * 3 + i) as u8).wrapping_mul(7);
            }
            for i in 0..16 {
                w_bytes[off + 192 + i] = (((b + i) as u8).wrapping_mul(5) + 8) & 0x7F;
            }
            w_bytes[off + 208..off + 210].copy_from_slice(&d_bits);
        }
        let x: Vec<f32> = (0..IN_F).map(|i| 1.0f32 + i as f32 * 0.001f32).collect();
        // Only check a few output elements to keep test fast
        let check_rows = [0usize, 1, 100, 1023];
        let padded = pad_to_u32_align(&w_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let xbuf = be.alloc(IN_F * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(OUT_F * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q6K,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            IN_F,
            OUT_F,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; OUT_F * 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_outputs: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec();
        for &o in &check_rows {
            let w_row_bytes =
                &w_bytes[o * n_blocks_per_row * BLOCK_SZ..(o + 1) * n_blocks_per_row * BLOCK_SZ];
            let (qv, sc, mn) = dequant_unified(infr_core::DType::Q6K, w_row_bytes);
            let cpu: f32 = (0..IN_F)
                .map(|i| (sc[i] * qv[i] as f32 + mn[i]) * x[i])
                .sum();
            let err = (gpu_outputs[o] - cpu).abs();
            let rel = err / (cpu.abs() + 1e-3);
            assert!(
                rel < 5e-3,
                "Q6K fullscale out[{o}]: gpu={} cpu={cpu} err={err} rel={rel}",
                gpu_outputs[o]
            );
        }
    }

    /// Multi-block Q6K test: 8 blocks, out_f=2. Tests cross-block access.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q6k_native_multiblock() {
        use infr_vulkan::linear::pad_to_u32_align;
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan");
                return;
            }
        };
        const N_BLOCKS: usize = 4;
        const BLOCK_SZ: usize = 210;
        const NELEMS: usize = 256;
        const IN_F: usize = N_BLOCKS * NELEMS;
        const OUT_F: usize = 2;
        const TOTAL_BLOCKS: usize = OUT_F * N_BLOCKS;
        let mut w_bytes = vec![0u8; TOTAL_BLOCKS * BLOCK_SZ];
        for b in 0..TOTAL_BLOCKS {
            let off = b * BLOCK_SZ;
            let d_bits = half::f16::from_f32(0.5 + b as f32 * 0.1)
                .to_bits()
                .to_le_bytes();
            for i in 0..128 {
                w_bytes[off + i] = ((b * 128 + i) as u8).wrapping_mul(11).wrapping_add(3);
            }
            for i in 0..64 {
                w_bytes[off + 128 + i] = ((b * 64 + i) as u8).wrapping_mul(7).wrapping_add(5);
            }
            for i in 0..16 {
                w_bytes[off + 192 + i] = (((b * 16 + i) as u8).wrapping_mul(5) + 8) & 0x7F;
            }
            w_bytes[off + 208..off + 210].copy_from_slice(&d_bits);
        }
        let mut cpu_outputs = vec![0f32; OUT_F];
        let x: Vec<f32> = (0..IN_F).map(|i| 1.0f32 + i as f32 * 0.001f32).collect();
        for o in 0..OUT_F {
            let w_row_bytes = &w_bytes[o * N_BLOCKS * BLOCK_SZ..(o + 1) * N_BLOCKS * BLOCK_SZ];
            let (qv, sc, mn) = dequant_unified(infr_core::DType::Q6K, w_row_bytes);
            let sum: f32 = (0..IN_F)
                .map(|i| (sc[i] * qv[i] as f32 + mn[i]) * x[i])
                .sum();
            cpu_outputs[o] = sum;
        }
        let padded = pad_to_u32_align(&w_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let xbuf = be.alloc(IN_F * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(OUT_F * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q6K,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            IN_F,
            OUT_F,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; OUT_F * 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_outputs: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec();
        for o in 0..OUT_F {
            let err = (gpu_outputs[o] - cpu_outputs[o]).abs();
            let rel = err / (cpu_outputs[o].abs() + 1e-3);
            assert!(
                rel < 5e-3,
                "Q6K out[{o}]: gpu={} cpu={} err={err} rel={rel}",
                gpu_outputs[o],
                cpu_outputs[o]
            );
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q6k_native_matches_cpu() {
        // d=0.5, scales[0..16]=0x20 (i8=32), ql=0xFF, qh=0xFF → q6=63
        let d_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 210];
        for b in &mut block[0..128] {
            *b = 0xFF;
        } // ql
        for b in &mut block[128..192] {
            *b = 0xFF;
        } // qh
        for b in &mut block[192..208] {
            *b = 0x20;
        } // scales = +32
        block[208..210].copy_from_slice(&d_bits);
        check_native(infr_core::DType::Q6K, &block);
    }

    /// Verify Q6K native shader handles f16 subnormal d values correctly.
    /// Real model weights use subnormal d (e.g. d_bits=0x0140 ≈ 1.9e-5), which
    /// naive f16→f32 that maps e=0 to 0 will silently zero out every output.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q6k_native_subnormal_d() {
        // d_bits = 0x0140 (e=0, m=0x140=320): subnormal f16 ≈ 1.9073e-5
        let d_bits: u16 = 0x0140;
        let mut block = vec![0u8; 210];
        for b in &mut block[0..128] {
            *b = 0xFF;
        } // ql all-1
        for b in &mut block[128..192] {
            *b = 0xFF;
        } // qh all-1
        for b in &mut block[192..208] {
            *b = 0x20;
        } // scales = i8 +32
        block[208..210].copy_from_slice(&d_bits.to_le_bytes());
        check_native(infr_core::DType::Q6K, &block);
    }

    /// Load a real Q6K tensor from the model and verify GPU vs CPU.
    #[test]
    #[ignore = "requires a Vulkan GPU and model file"]
    fn q6k_real_model_tensor() {
        use infr_vulkan::linear::pad_to_u32_align;
        let model_path = std::path::Path::new(
            "/home/mxaddict/Projects/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf",
        );
        if !model_path.exists() {
            eprintln!("skip: model not found");
            return;
        }
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan");
                return;
            }
        };
        let g = infr_gguf::Gguf::open(model_path).unwrap();
        // attn_v.weight blk.0: Q6K, [1024, 1024] → in_f=1024, out_f=1024
        let tensor_name = "blk.0.attn_v.weight";
        let bytes = g.tensor_bytes(tensor_name).unwrap();
        let in_f = 1024usize;
        let out_f = 1024usize;
        // CPU ref: dot each output row against x=all-1.0
        let (qv, sc, mn) = dequant_unified(infr_core::DType::Q6K, bytes);
        let numel = in_f * out_f;
        assert_eq!(qv.len(), numel, "element count mismatch");
        let x: Vec<f32> = vec![1.0f32; in_f];
        let mut cpu_out = vec![0f32; out_f];
        for o in 0..out_f {
            cpu_out[o] = (0..in_f)
                .map(|i| sc[o * in_f + i] * qv[o * in_f + i] as f32 + mn[o * in_f + i])
                .sum();
        }
        // GPU
        let padded = pad_to_u32_align(bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let xbuf = be.alloc(in_f * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(out_f * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q6K,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            in_f,
            out_f,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; out_f * 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec();
        let mut max_err = 0f32;
        let mut max_idx = 0;
        let mut n_zero = 0usize;
        for o in 0..out_f {
            let err = (gpu_out[o] - cpu_out[o]).abs();
            if gpu_out[o] == 0.0 && cpu_out[o].abs() > 0.1 {
                n_zero += 1;
            }
            if err > max_err {
                max_err = err;
                max_idx = o;
            }
        }
        // Print first 5 failing elements
        let mut n_print = 0;
        for o in 0..out_f {
            let rel = (gpu_out[o] - cpu_out[o]).abs() / (cpu_out[o].abs() + 1e-3);
            if rel > 5e-3 && n_print < 5 {
                eprintln!("FAIL out[{o}]: gpu={} cpu={}", gpu_out[o], cpu_out[o]);
                n_print += 1;
            }
        }
        eprintln!("Real Q6K: n_zero={n_zero}/{out_f}, max_err={max_err} at out[{max_idx}]");
        let rel = max_err / (cpu_out[max_idx].abs() + 1e-3);
        assert!(
            rel < 5e-3,
            "Real Q6K tensor: max_err={max_err} at out[{max_idx}]: gpu={} cpu={} rel={rel}",
            gpu_out[max_idx],
            cpu_out[max_idx]
        );
    }

    // ── Native-block codebook formats (IQ4_NL/XS, MXFP4, NVFP4, TQ1_0, TQ2_0) ────
    //
    // CPU reference is `dequant_codebook` (the verified host port). GPU runs `linear_native`
    // with x=all-1.0 so the output is the sum of dequantized weights.

    fn check_native_cb(dtype: infr_core::DType, block_bytes: &[u8]) {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        use infr_vulkan::linear::pad_to_u32_align;
        let cpu = dequant_codebook(dtype, block_bytes);
        let numel = cpu.len();
        let cpu_out: f32 = cpu.iter().sum();

        let padded = pad_to_u32_align(block_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let x: Vec<f32> = vec![1.0f32; numel];
        let xbuf = be.alloc(x.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            dtype,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            numel,
            1,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_out: f32 = bytemuck::cast_slice(&out_bytes)[0];
        let rel = (gpu_out - cpu_out).abs() / (cpu_out.abs() + 1e-4);
        assert!(
            rel < 5e-3,
            "{dtype:?} native cb GPU vs CPU: gpu={gpu_out} cpu={cpu_out} rel={rel}"
        );
    }

    // varied non-trivial byte pattern
    fn fill(buf: &mut [u8], mul: u8, add: u8) {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(mul).wrapping_add(add);
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn iq4nl_native_matches_cpu() {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.5).to_bits().to_le_bytes());
        fill(&mut block[2..18], 23, 7);
        check_native_cb(infr_core::DType::Iq4Nl, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn iq4xs_native_matches_cpu() {
        let mut block = vec![0u8; 136];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&0x9ce3u16.to_le_bytes()); // scales_h varied
        fill(&mut block[4..8], 53, 11); // scales_l
        fill(&mut block[8..136], 13, 3); // qs
        check_native_cb(infr_core::DType::Iq4Xs, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn mxfp4_native_matches_cpu() {
        let mut block = vec![0u8; 17];
        block[0] = 128; // e8m0 → d=1.0
        fill(&mut block[1..17], 29, 5);
        check_native_cb(infr_core::DType::Mxfp4, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn nvfp4_native_matches_cpu() {
        let mut block = vec![0u8; 36];
        block[0..4].copy_from_slice(&[0x38, 0x40, 0x48, 0x30]); // valid ue4m3 scales
        fill(&mut block[4..36], 19, 9);
        check_native_cb(infr_core::DType::Nvfp4, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn tq1_0_native_matches_cpu() {
        let mut block = vec![0u8; 54];
        fill(&mut block[0..52], 17, 1); // qs + qh
        block[52..54].copy_from_slice(&half::f16::from_f32(0.75).to_bits().to_le_bytes());
        check_native_cb(infr_core::DType::Tq1_0, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn tq2_0_native_matches_cpu() {
        let mut block = vec![0u8; 66];
        fill(&mut block[0..64], 11, 3); // qs
        block[64..66].copy_from_slice(&half::f16::from_f32(1.25).to_bits().to_le_bytes());
        check_native_cb(infr_core::DType::Tq2_0, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn iq2xxs_native_matches_cpu() {
        // 2 blocks (in_f=512) to exercise cross-block + grid/sign decode.
        let mut blocks = vec![0u8; 2 * 66];
        for (bi, blk) in blocks.chunks_mut(66).enumerate() {
            blk[0..2].copy_from_slice(
                &half::f16::from_f32(1.0 + bi as f32 * 0.5)
                    .to_bits()
                    .to_le_bytes(),
            );
            fill(&mut blk[2..66], 31, (bi as u8) * 7 + 13); // qs (grid idx + signs + scale)
        }
        check_native_cb(infr_core::DType::Iq2Xxs, &blocks);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn iq2xs_native_matches_cpu() {
        let mut block = vec![0u8; 74];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..66], 29, 5); // qs (u16 grid idx + sign)
        fill(&mut block[66..74], 17, 1); // scales
        check_native_cb(infr_core::DType::Iq2Xs, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn iq2s_native_matches_cpu() {
        let mut block = vec![0u8; 82];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..66], 23, 7); // qs (idx low) + sign bytes
        fill(&mut block[66..74], 13, 2); // qh
        fill(&mut block[74..82], 19, 1); // scales
        check_native_cb(infr_core::DType::Iq2S, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn iq3xxs_native_matches_cpu() {
        let mut block = vec![0u8; 98];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..66], 7, 1); // qs (grid indices)
        fill(&mut block[66..98], 13, 3); // sas (scale+signs)
        check_native_cb(infr_core::DType::Iq3Xxs, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn iq3s_native_matches_cpu() {
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..66], 11, 2); // qs
        fill(&mut block[66..74], 5, 1); // qh
        fill(&mut block[74..106], 17, 3); // signs
        fill(&mut block[106..110], 3, 1); // scales
        check_native_cb(infr_core::DType::Iq3S, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn iq1s_native_matches_cpu() {
        let mut block = vec![0u8; 50];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..34], 13, 1); // qs
        fill(&mut block[34..50], 23, 7); // qh (u16: grid hi bits + scale + delta)
        check_native_cb(infr_core::DType::Iq1S, &block);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn iq1m_native_matches_cpu() {
        let mut block = vec![0u8; 56];
        fill(&mut block[0..32], 17, 3); // qs
        fill(&mut block[32..48], 11, 1); // qh
                                         // scales: nonzero so packed d != 0
        block[48..56].copy_from_slice(&[0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a, 0xf0, 0x3d]);
        check_native_cb(infr_core::DType::Iq1M, &block);
    }
}

#[cfg(test)]
mod tokenizer_tests {
    use super::*;

    // Validate the GGUF-derived tokenizer against the HF tokenizer.json sidecar (same model).
    // Skips if the test model isn't present.
    #[test]
    fn embedded_tokenizer_matches_sidecar() {
        let dir = Path::new("/home/mxaddict/Projects/models/qwen3-0.6b");
        let gguf = dir.join("Qwen3-0.6B-Q4_K_M.gguf");
        let side = dir.join("tokenizer.json");
        if !gguf.exists() || !side.exists() {
            eprintln!("skip: test model not present");
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

    // Streaming must hold a multi-byte char (emoji) split across tokens instead of emitting `�`.
    #[test]
    fn stream_decoder_holds_partial_utf8() {
        let mut s = StreamDecoder::default();
        // Simulate the per-step FULL decode of "Hi😄" where the emoji's bytes arrive across 2 tokens.
        assert_eq!(s.step("Hi"), "Hi");
        assert_eq!(s.step("Hi\u{FFFD}"), ""); // emoji half-decoded → hold, no `�` emitted
        assert_eq!(s.step("Hi😄"), "😄"); // completes → emit the whole char
        assert_eq!(s.step("Hi😄!"), "!");
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
        let gguf = Path::new("/home/mxaddict/Projects/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf");
        if !gguf.exists() {
            eprintln!("skip: test model not present");
            return;
        }
        let g = Gguf::open(gguf).unwrap();
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
