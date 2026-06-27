//! Minimal autoregressive **Llama** inference for GGUF models, for fast GPU bring-up.
//!
//! Strategy (bring-up): the heavy linear projections run on the GPU (`infr-vulkan` eager
//! `linear`, weights uploaded once); the cheap ops (embedding gather, RMSNorm, RoPE, GQA
//! attention, SwiGLU, residual, sampling) run on the host. No KV cache yet — each step does a
//! full-prefix forward (fine for a tiny model). Validated on SmolLM2-135M.
//!
//! TODO(next): move host ops to GPU; add a KV cache; fold into the `Model`/`Backend` seams.
#![allow(clippy::needless_range_loop)]

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

/// A projection weight on the GPU: either f16, or a unified in-kernel-dequant quant. Every
/// supported quant (Q8_0/Q4_K/Q5_K/Q6_K) is repacked at load into ONE form — `q` = u8 indices
/// packed 4-per-u32, `s`/`m` = one f16 scale/min per 16-element block — so `dq = s·u8 + m`.
enum Wt {
    F16(Box<dyn Buffer>),
    Q {
        q: Box<dyn Buffer>,
        s: Box<dyn Buffer>,
        m: Box<dyn Buffer>,
        bits: u32,      // 4 (Q4 → packed 8/u32) or 8 (Q5/Q6/Q8)
        blk_shift: u32, // log2 of the scale/min block size (5 = per-32, 4 = per-16)
    },
}
impl Wt {
    /// The f16 buffer (panics if quantized — used by the llama fused path, which is f16-only).
    fn f16(&self) -> &dyn Buffer {
        match self {
            Wt::F16(b) => b.as_ref(),
            Wt::Q { .. } => panic!("expected f16 weight, got quant (llama fused path needs f16)"),
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

/// Load a tensor as f32, returning (data, shape) where shape is GGUF ne order ([in, out]).
fn load_f32(g: &Gguf, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let info = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor not found: {name}"))?
        .clone();
    let bytes = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
    let v: Vec<f32> = match info.dtype {
        infr_core::DType::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        infr_core::DType::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        infr_core::DType::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        d if is_quant(d) => {
            let (qv, sc, mn) = dequant_unified(d, bytes);
            (0..qv.len())
                .map(|g| sc[g] * qv[g] as f32 + mn[g])
                .collect()
        }
        other => {
            bail!("unsupported dtype {other:?} for {name} (host load wants F16/F32/BF16/quant)")
        }
    };
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
                .map(|&x| half::f16::from_f32(x).to_bits())
                .collect();
            Ok(bytemuck::cast_slice(&f16).to_vec())
        }
        infr_core::DType::Bf16 => {
            // bf16 → f32 → f16 (bf16 is the top 16 bits of f32)
            let f16: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| {
                    let f = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                    half::f16::from_f32(f).to_bits()
                })
                .collect();
            Ok(bytemuck::cast_slice(&f16).to_vec())
        }
        other => bail!("unsupported dtype {other:?} for {name} (bring-up wants F16/F32/BF16)"),
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
        Q8_0 => (32, 34),
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

fn is_quant(d: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(d, Q8_0 | Q4K | Q5K | Q6K)
}

/// In-VRAM packing per source quant: (bits, scale/min block size). Q4 packs to native 4-bit (8×
/// smaller index than f16); Q5/Q6/Q8 keep 8-bit. Block size matches the quant's sub-block.
fn quant_params(d: infr_core::DType) -> (u32, usize) {
    use infr_core::DType::*;
    match d {
        Q4K => (4, 32),
        Q5K => (8, 32),
        Q6K => (8, 16),
        Q8_0 => (8, 32),
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

/// Upload a projection weight, keeping quantized weights quantized in-VRAM (else convert to f16).
fn upload_wt(be: &VulkanBackend, g: &Gguf, name: &str) -> Result<Wt> {
    let info = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor not found: {name}"))?
        .clone();
    if is_quant(info.dtype) {
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
        let (token_embd, te_shape) = load_f32(&g, "token_embd.weight")?;
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

        let (output_norm, _) = load_f32(&g, "output_norm.weight")?;
        let output_norm_buf = be
            .upload_weight(&output_norm)
            .map_err(|e| anyhow!("upload output_norm: {e}"))?;

        // Loading the per-layer weights (dequant + GPU upload) dominates startup, especially for
        // big models — show a progress bar (hidden when stderr isn't a TTY, e.g. piped/served).
        let pb = {
            use std::io::IsTerminal;
            let pb = if std::io::stderr().is_terminal() {
                indicatif::ProgressBar::new(n_layer as u64)
            } else {
                indicatif::ProgressBar::hidden()
            };
            pb.set_style(
                indicatif::ProgressStyle::with_template(
                    "  {spinner:.green} loading weights [{bar:32.cyan/blue}] {pos}/{len} layers ({elapsed})",
                )
                .unwrap()
                .progress_chars("━━╾─"),
            );
            pb
        };
        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let up = |be: &VulkanBackend, name: String| -> Result<Wt> { upload_wt(be, &g, &name) };
            let attn_norm = load_f32(&g, &p("attn_norm.weight"))?.0;
            let ffn_norm = load_f32(&g, &p("ffn_norm.weight"))?.0;
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
                        be.upload_weight(&load_f32(&g, &p("attn_q_norm.weight"))?.0)
                            .map_err(|e| anyhow!("upload q_norm {l}: {e}"))?,
                    ),
                    Some(
                        be.upload_weight(&load_f32(&g, &p("attn_k_norm.weight"))?.0)
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
            pb.inc(1);
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
        // weight-op dispatchers: pick the f16 or quant kernel based on how the weight is stored.
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
