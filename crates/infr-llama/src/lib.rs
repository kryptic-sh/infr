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
    /// MoE config (qwen3moe): `Some` enables the routed-expert FFN. `None` = dense FFN.
    pub moe: Option<MoeConfig>,
}

/// Mixture-of-experts FFN parameters (qwen3moe): a softmax router picks `n_used` of `n_expert`
/// experts per token, each a SwiGLU FFN of inner size `n_ff_exp`, summed by renormalized top-k
/// weights (`scale` applied). Attention is identical to dense qwen3.
#[derive(Clone, Copy, Debug)]
pub struct MoeConfig {
    pub n_expert: usize,
    pub n_used: usize,
    pub n_ff_exp: usize,
    pub scale: f32,
}

/// State for the eager MoE generation: a GPU KV cache (so context competes for VRAM, like the dense
/// path) + the streaming `ExpertPool` for `INFR_MOE_STREAM` (lazily created on first streamed layer).
pub struct MoeKv {
    kv: KvCache,
    pool: Option<infr_vulkan::ExpertPool>,
    /// Persistent decode scratch (Tier 0): the per-token activation buffers, allocated once and
    /// reused every decode step instead of created/freed per token.
    dec: Option<DecodeScratch>,
}

/// Reusable GPU scratch for one decode step's forward (all buffers sized for a single token; the
/// split-K attention buffers are sized for the cache's worst-case chunk count). Held by [`MoeKv`]
/// so decode doesn't churn ~22 buffer create/free calls per token.
struct DecodeScratch {
    hidden: Box<dyn Buffer>,
    hn: Box<dyn Buffer>,
    hn2: Box<dyn Buffer>,
    ao: Box<dyn Buffer>,
    qr: Box<dyn Buffer>,
    kr: Box<dyn Buffer>,
    vr: Box<dyn Buffer>,
    q_f16: Box<dyn Buffer>,
    attn: Box<dyn Buffer>,
    g: Box<dyn Buffer>,
    u: Box<dyn Buffer>,
    act: Box<dyn Buffer>,
    y: Box<dyn Buffer>,
    logits: Box<dyn Buffer>,
    ids: Box<dyn Buffer>,
    wts: Box<dyn Buffer>,
    qa: Box<dyn Buffer>,
    dact: Box<dyn Buffer>,
    sact: Box<dyn Buffer>,
    pm: Box<dyn Buffer>,
    pl: Box<dyn Buffer>,
    pacc: Box<dyn Buffer>,
}

impl MoeKv {
    /// Tokens currently resident in the cache (the next chunk's start position).
    pub fn len(&self) -> usize {
        self.kv.len
    }
    /// True when no tokens are resident yet.
    pub fn is_empty(&self) -> bool {
        self.kv.len == 0
    }
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
/// - `Q`: unified repacked affine quant (q/s/m buffers, `dq = s·u8 + m`); fallback when native is
///   disabled (`INFR_NONATIVE=1`) or for grid/codebook quants under `INFR_NATIVE=1`.
/// - `Native`: raw GGUF block bytes, padded to u32 alignment, dequantized in-shader (decode-once
///   GEMV + tiled coopmat GEMM). The DEFAULT for optimized affine quants — faster decode + prefill
///   and smaller VRAM (see [`is_native_default`]); `INFR_NATIVE=1` extends it to all formats.
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

/// One expert weight: resident on the GPU (`Gpu`) or host-backed (`Cpu`) — for host-backed experts
/// the bytes are read on demand from the kept-alive GGUF mmap (no host-RAM copy), then computed on
/// the CPU or streamed to a VRAM pool (`INFR_NCMOE` / `INFR_MOE_STREAM`, cf. `--n-cpu-moe`).
enum ExpertW {
    Gpu(Wt),
    Cpu { dtype: infr_core::DType },
}
impl ExpertW {
    fn is_cpu(&self) -> bool {
        matches!(self, ExpertW::Cpu { .. })
    }
    /// The GPU weight (panics for host experts — callers branch on [`is_cpu`] first).
    fn gpu(&self) -> &Wt {
        match self {
            ExpertW::Gpu(w) => w,
            ExpertW::Cpu { .. } => panic!("expected GPU expert"),
        }
    }
}

/// One routed expert's SwiGLU weights (gate/up [n_embd→n_ff_exp], down [n_ff_exp→n_embd]).
struct ExpertWt {
    gate: ExpertW,
    up: ExpertW,
    down: ExpertW,
}

/// Stacked GPU expert bank: one Native buffer per role holding all experts contiguously, addressed
/// by element offset (`expert_id * stride`). Lets the GPU-resident decode/prefill dispatch every
/// expert from a single buffer (so an on-GPU router id can pick the expert — no host round-trip).
/// Built only for fully-GPU, native-quant, non-offloaded models; offloaded models keep `experts`.
struct MoeStacked {
    gate: Wt,
    up: Wt,
    down: Wt,
    stride: usize, // elements per expert (n_ff_exp * n_embd), identical for gate/up/down
}

/// A layer's FFN: dense fused gate‖up + down, or a routed MoE bank (router + per-expert weights).
enum FfnWt {
    Dense {
        wgateup: Wt,
        wdown: Wt,
    },
    Moe {
        gate_inp: Wt,
        experts: Vec<ExpertWt>, // empty when `stacked` is Some (per-expert buffers dropped)
        stacked: Option<MoeStacked>,
    },
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
    ffn: FfnWt,
    q_norm_buf: Option<Box<dyn Buffer>>, // qwen3 QK-norm weights [head_dim]
    k_norm_buf: Option<Box<dyn Buffer>>,
}

impl LayerWeights {
    fn wgateup(&self) -> &Wt {
        match &self.ffn {
            FfnWt::Dense { wgateup, .. } => wgateup,
            FfnWt::Moe { .. } => panic!("MoE layer has no dense wgateup"),
        }
    }
    fn wdown(&self) -> &Wt {
        match &self.ffn {
            FfnWt::Dense { wdown, .. } => wdown,
            FfnWt::Moe { .. } => panic!("MoE layer has no dense wdown"),
        }
    }
    fn moe(&self) -> (&Wt, &[ExpertWt]) {
        match &self.ffn {
            FfnWt::Moe {
                gate_inp, experts, ..
            } => (gate_inp, experts),
            FfnWt::Dense { .. } => panic!("dense layer has no MoE bank"),
        }
    }
    /// The router weight + stacked expert bank, when this layer is a fully-GPU native MoE layer
    /// (the GPU-resident decode/prefill path). `None` for offloaded / per-expert layers.
    fn moe_stacked(&self) -> Option<(&Wt, &MoeStacked)> {
        match &self.ffn {
            FfnWt::Moe {
                gate_inp,
                stacked: Some(s),
                ..
            } => Some((gate_inp, s)),
            _ => None,
        }
    }
}

/// A forward step's output: the sampled token (chosen on the GPU — only 4 bytes cross the bus) or
/// the full vocab logits (host samples them, when GPU sampling can't handle the config).
enum GenOut {
    Token(u32),
    Logits(Vec<f32>),
}

/// Per-step sampling config for on-GPU token selection. `u` is the host-drawn uniform in [0,1).
#[derive(Clone, Copy)]
struct SampleParams {
    temp: f32,
    top_k: usize,
    top_p: f32,
    u: f32,
}
impl SampleParams {
    /// Greedy (argmax) when temperature is off or only one candidate is kept.
    fn greedy(&self) -> bool {
        self.temp <= 0.0 || self.top_k == 1
    }
    /// The GPU sampler handles temp/top-k/top-p only for a bounded top_k; else host samples logits.
    fn gpu_capable(&self) -> bool {
        !self.greedy() && self.top_k >= 2 && self.top_k <= infr_vulkan::Recorder::SAMPLE_KMAX
    }
}

/// Advance an xorshift64 RNG and return a uniform in [0,1) — the per-step random draw for sampling.
fn draw_u(rng: &mut u64) -> f32 {
    let mut x = *rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng = x;
    (x >> 40) as f32 / (1u64 << 24) as f32
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
    /// MoE: `INFR_MOE_STREAM` makes host-offloaded (`INFR_NCMOE`) layers stream their active experts
    /// into a VRAM pool + GPU-compute instead of CPU matvec.
    moe_stream: bool,
    /// The model's GGUF, kept mmap-alive so host-backed MoE experts can read their bytes on demand
    /// (zero-copy from the OS page cache) instead of duplicating them into RAM.
    gguf: Gguf,
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
/// and SSM recurrence math (qwen35), the `Q35_CPU=1` oracle, and serves as the f32 source we
/// convert into f16/bf16/quant GPU weights. Survives even with full GPU format coverage.
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
                aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
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
                    let dl1 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2))) & 7) as f32 + 1.0);
                    let dl2 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) as f32 + 1.0);
                    let qh0 = qh_arr[qh_off];
                    let qh1 = qh_arr[qh_off + 1];
                    let idx = [
                        qs_arr[qs_off] as usize | (((qh0 as usize) << 8) & 0x700),
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

/// VRAM the model's weights will occupy once resident, split dense vs MoE-expert. Experts are
/// tracked separately so a future expert-streaming / partial-offload mode can budget them apart
/// from the always-resident dense weights — for a dense model `expert` is 0.
#[derive(Clone, Copy, Debug)]
pub struct WeightFootprint {
    /// Always-resident weights: projections, embeddings, norms.
    pub dense: u64,
    /// MoE expert weights (GGUF `*_exps` stacked tensors). 0 for dense models.
    pub expert: u64,
}
impl WeightFootprint {
    /// All-resident footprint: dense + every expert kept in VRAM.
    pub fn total(&self) -> u64 {
        self.dense + self.expert
    }

    /// Footprint if experts are STREAMED through an `n_slots`-slot pool of `stride`-byte slots
    /// (`infr_vulkan::ExpertPool`) instead of all kept resident: `dense + n_slots·stride`, bounded
    /// regardless of the model's expert count. The MoE loader picks all-resident ([`total`]) when it
    /// fits VRAM, else reserves this and streams. (`stride` = one expert's max packed weight bytes.)
    pub fn streaming_total(&self, n_slots: usize, stride: usize) -> u64 {
        self.dense + n_slots as u64 * stride as u64
    }
}

/// Resident VRAM bytes for one tensor, mirroring [`upload_wt`]'s path so the estimate matches what
/// actually gets allocated: native raw blocks (padded to u32), unified repack (index@bits + f16
/// scale + f16 min per block), or f16 (codebook/float/norms dequanted to half).
fn tensor_resident_bytes(dtype: infr_core::DType, numel: usize, nbytes: usize) -> u64 {
    if use_native_for(dtype) && is_native_supported(dtype) {
        ((nbytes + 3) & !3) as u64 // raw blocks, padded to u32 alignment
    } else if is_quant(dtype) {
        let (bits, blk) = quant_params(dtype);
        let q = numel * bits as usize / 8;
        let sm = 2 * (numel / blk.max(1)) * 2; // scale + min, one f16 each per block
        (q + sm) as u64
    } else {
        (numel * 2) as u64 // f16
    }
}

/// Sum the resident weight footprint across all tensors (MoE-aware). Enumerating every tensor means
/// stacked expert tensors are counted in full, so this is correct for MoE the moment the arch is
/// supported. `token_embd` is excluded (it lives in host RAM for the CPU embedding gather) unless
/// the lm head is tied to it (no `output.weight`), where an f16 copy is uploaded to VRAM.
pub fn weight_footprint(g: &Gguf) -> WeightFootprint {
    let has_output = g.tensors().iter().any(|t| t.name == "output.weight");
    let mut dense = 0u64;
    let mut expert = 0u64;
    for t in g.tensors() {
        let numel: usize = t.shape.iter().product();
        if t.name == "token_embd.weight" {
            if !has_output {
                dense += (numel * 2) as u64; // tied lm head, uploaded as f16
            }
            continue;
        }
        let bytes = tensor_resident_bytes(t.dtype, numel, t.nbytes);
        if t.name.contains("_exps") {
            expert += bytes;
        } else {
            dense += bytes;
        }
    }
    WeightFootprint { dense, expert }
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

/// Affine quants with an optimized decode-once native path: sub-block-major GEMV (`dqblk`) +
/// tiled coopmat GEMM (`native_gemm`). For these, native beats the unified repack (`Wt::Q`) on BOTH
/// decode (e.g. Q4_K 506 vs 480 t/s) and prefill (Q8_0/Q6_K/Q5_K +27..40%; Q4_K within ~4% of
/// unified's dp4a mmq) AND uses less VRAM — so native is the DEFAULT here. Grid/codebook i-quants
/// (IQ2*/IQ3*/IQ1*, IQ4*, fp4, TQ*) only have the per-element fallback `dqblk` and their unified
/// alternative is fast f16, so they stay opt-in (decode is the north star).
fn is_native_default(d: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(
        d,
        Q8_0 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q2K | Q3K | Q4K | Q5K | Q6K
    )
}

/// Whether to use the raw-block / in-shader-dequant path (`Wt::Native`) for `dtype`.
/// - `INFR_NONATIVE=1` → never (force unified repack / f16, the pre-GEMM behavior).
/// - `INFR_NATIVE=1` → always, for every native-supported format (incl. grid/codebook).
/// - default → only the optimized affine formats ([`is_native_default`]).
fn use_native_for(d: infr_core::DType) -> bool {
    if std::env::var("INFR_NONATIVE").is_ok() {
        return false;
    }
    if std::env::var("INFR_NATIVE").is_ok() {
        return true;
    }
    is_native_default(d)
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
/// - Optimized affine quants (Q4_K/Q5_K/Q6_K/Q8_0/Q4_0…) → `Wt::Native` by default (raw block
///   bytes, in-shader decode-once dequant — faster decode + prefill, smaller VRAM; see
///   [`use_native_for`]). `INFR_NONATIVE=1` falls back to `Wt::Q` (host dequant + repack).
/// - Other affine quants → `Wt::Q`; `INFR_NATIVE=1` extends native to all supported formats.
/// - Codebook quants (IQ*/TQ*/fp4) → host dequant → f16 → `Wt::F16` (native via `INFR_NATIVE=1`)
/// - Float types (F16/F32/BF16) → `Wt::F16` directly
fn upload_wt(be: &VulkanBackend, g: &Gguf, name: &str) -> Result<Wt> {
    let dtype = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor not found: {name}"))?
        .dtype;
    let bytes = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
    upload_wt_bytes(be, dtype, bytes)
}

/// Like [`upload_wt`] but from a raw byte slice + dtype — lets a stacked MoE expert tensor be sliced
/// per expert (each expert is a contiguous block of the `*_exps` tensor) and uploaded individually.
fn upload_wt_bytes(be: &VulkanBackend, dtype: infr_core::DType, bytes: &[u8]) -> Result<Wt> {
    // Native-block path: raw upload + in-shader dequant. Default for optimized affine quants;
    // INFR_NATIVE forces all supported formats, INFR_NONATIVE disables (see use_native_for).
    if use_native_for(dtype) && is_native_supported(dtype) {
        let padded = infr_vulkan::linear::pad_to_u32_align(bytes);
        return Ok(Wt::Native {
            buf: be
                .upload_weight_bytes(&padded)
                .map_err(|e| anyhow!("native upload: {e}"))?,
            dtype,
        });
    }
    if is_quant(dtype) {
        // Affine quants: GPU in-kernel dequant path (legacy / INFR_NONATIVE)
        let (bits, blk) = quant_params(dtype);
        let (qv, sc, mn) = dequant_unified(dtype, bytes);
        let (q, s, m) = pack_unified(&qv, &sc, &mn, bits, blk);
        Ok(Wt::Q {
            q: be.upload_weight_bytes(bytemuck::cast_slice(&q))?,
            s: be.upload_weight_bytes(bytemuck::cast_slice(&s))?,
            m: be.upload_weight_bytes(bytemuck::cast_slice(&m))?,
            bits,
            blk_shift: blk.trailing_zeros(),
        })
    } else {
        // Codebook quants and float types → host dequant to f32 → f16.
        let f16_bytes: Vec<u8> = dequant_block(dtype, bytes)?
            .iter()
            .flat_map(|&v| f32_to_f16_sat(v).to_bits().to_le_bytes())
            .collect();
        Ok(Wt::F16(be.upload_weight_bytes(&f16_bytes)?))
    }
}

/// Build a dense layer's fused gate‖up weight (`[2*n_ff, n_embd]`, gate rows then up rows). Quant
/// stays quantized (unified repack); codebook/float → f16. `prefix` is e.g. `"blk.3."`.
fn build_wgateup(be: &VulkanBackend, g: &Gguf, prefix: &str) -> Result<Wt> {
    let gate_name = format!("{prefix}ffn_gate.weight");
    let up_name = format!("{prefix}ffn_up.weight");
    let gate_dtype = g
        .tensors()
        .iter()
        .find(|t| t.name == gate_name)
        .map(|t| t.dtype);
    let gb = g.tensor_bytes(&gate_name).map_err(|e| anyhow!("{e}"))?;
    let ub = g.tensor_bytes(&up_name).map_err(|e| anyhow!("{e}"))?;
    if gate_dtype.map(is_quant).unwrap_or(false) {
        let dt = gate_dtype.unwrap();
        let (bits, blk) = quant_params(dt);
        let (mut qv, mut sc, mut mn) = dequant_unified(dt, gb);
        let (qu, scu, mnu) = dequant_unified(dt, ub);
        qv.extend(qu);
        sc.extend(scu);
        mn.extend(mnu);
        let (q, s, m) = pack_unified(&qv, &sc, &mn, bits, blk);
        Ok(Wt::Q {
            q: be.upload_weight_bytes(bytemuck::cast_slice(&q))?,
            s: be.upload_weight_bytes(bytemuck::cast_slice(&s))?,
            m: be.upload_weight_bytes(bytemuck::cast_slice(&m))?,
            bits,
            blk_shift: blk.trailing_zeros(),
        })
    } else if gate_dtype.map(is_codebook_quant).unwrap_or(false) {
        let dt = gate_dtype.unwrap();
        let to_f16 = |bytes: &[u8]| -> Vec<u8> {
            dequant_codebook(dt, bytes)
                .iter()
                .flat_map(|&v| f32_to_f16_sat(v).to_bits().to_le_bytes())
                .collect()
        };
        let mut gateup = to_f16(gb);
        gateup.extend_from_slice(&to_f16(ub));
        Ok(Wt::F16(be.upload_weight_bytes(&gateup)?))
    } else {
        let mut gateup = f16_bytes(g, &gate_name)?;
        gateup.extend_from_slice(&f16_bytes(g, &up_name)?);
        Ok(Wt::F16(be.upload_weight_bytes(&gateup)?))
    }
}

/// Load a layer's MoE expert bank: the router `ffn_gate_inp` + the `n_expert` per-expert SwiGLU
/// weights sliced from the stacked `ffn_{gate,up,down}_exps` tensors (each expert is one contiguous
/// `1/n_expert` block of the stacked tensor — quant blocks never cross expert boundaries).
fn load_moe(
    be: &VulkanBackend,
    g: &Gguf,
    prefix: &str,
    n_expert: usize,
    on_cpu: bool,
    build_stacked: bool,
    stride_elems: usize,
) -> Result<FfnWt> {
    let gate_inp = upload_wt(be, g, &format!("{prefix}ffn_gate_inp.weight"))?;
    let stacked = |role: &str| -> Result<(infr_core::DType, &[u8])> {
        let name = format!("{prefix}ffn_{role}_exps.weight");
        let dt = g
            .tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?
            .dtype;
        let bytes = g.tensor_bytes(&name).map_err(|e| anyhow!("{e}"))?;
        Ok((dt, bytes))
    };
    let (gdt, gbytes) = stacked("gate")?;
    let (udt, ubytes) = stacked("up")?;
    let (ddt, dbytes) = stacked("down")?;

    // Fully-GPU native model: upload each role's whole `*_exps` tensor as ONE Native buffer and
    // address experts by element offset. Per-expert buffers are dropped (same VRAM, one allocation),
    // and the on-GPU router can index experts without a host round-trip.
    let native_ok = [gdt, udt, ddt]
        .iter()
        .all(|&d| use_native_for(d) && is_native_supported(d));
    if build_stacked && !on_cpu && native_ok {
        let mk = |dt, b| upload_wt_bytes(be, dt, b);
        return Ok(FfnWt::Moe {
            gate_inp,
            experts: Vec::new(),
            stacked: Some(MoeStacked {
                gate: mk(gdt, gbytes)?,
                up: mk(udt, ubytes)?,
                down: mk(ddt, dbytes)?,
                stride: stride_elems,
            }),
        });
    }

    let (gstride, ustride, dstride) = (
        gbytes.len() / n_expert,
        ubytes.len() / n_expert,
        dbytes.len() / n_expert,
    );
    // GPU experts upload to VRAM; host experts store only the dtype — their bytes are read on demand
    // from the kept-alive GGUF mmap at forward time (no host-RAM copy).
    let place = |dt: infr_core::DType, b: &[u8]| -> Result<ExpertW> {
        if on_cpu {
            Ok(ExpertW::Cpu { dtype: dt })
        } else {
            Ok(ExpertW::Gpu(upload_wt_bytes(be, dt, b)?))
        }
    };
    let mut experts = Vec::with_capacity(n_expert);
    for e in 0..n_expert {
        experts.push(ExpertWt {
            gate: place(gdt, &gbytes[e * gstride..(e + 1) * gstride])?,
            up: place(udt, &ubytes[e * ustride..(e + 1) * ustride])?,
            down: place(ddt, &dbytes[e * dstride..(e + 1) * dstride])?,
        });
    }
    Ok(FfnWt::Moe {
        gate_inp,
        experts,
        stacked: None,
    })
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
        // llama / qwen3 / qwen3moe share the transformer; qwen3* add QK-norm + explicit head_dim,
        // qwen3moe replaces the dense FFN with a routed-expert bank.
        let qk_norm = match arch.as_str() {
            "llama" => false,
            "qwen3" | "qwen3moe" => true,
            other => bail!("infr-llama supports architecture=llama|qwen3|qwen3moe, got {other:?}"),
        };
        let mk = |k: &str| format!("{arch}.{k}");
        let n_layer = meta_u64(&g, &mk("block_count")).context("block_count")? as usize;
        let n_embd = meta_u64(&g, &mk("embedding_length")).context("embedding_length")? as usize;
        let n_head = meta_u64(&g, &mk("attention.head_count")).context("head_count")? as usize;
        let n_kv = meta_u64(&g, &mk("attention.head_count_kv")).unwrap_or(n_head as u64) as usize;
        let n_ff =
            meta_u64(&g, &mk("feed_forward_length")).context("feed_forward_length")? as usize;
        // MoE (qwen3moe): softmax router over `n_expert`, top-`n_used`, per-expert SwiGLU of inner
        // size `n_ff_exp` (the GGUF `expert_feed_forward_length`).
        let moe = if arch == "qwen3moe" {
            let n_expert = meta_u64(&g, &mk("expert_count")).context("expert_count")? as usize;
            let n_used =
                meta_u64(&g, &mk("expert_used_count")).context("expert_used_count")? as usize;
            let n_ff_exp = meta_u64(&g, &mk("expert_feed_forward_length"))
                .map(|v| v as usize)
                .unwrap_or(n_ff / n_used.max(1));
            Some(MoeConfig {
                n_expert,
                n_used,
                n_ff_exp,
                scale: 1.0, // qwen3moe: renormalize top-k softmax weights, no extra scale
            })
        } else {
            None
        };
        // INFR_NCMOE=N: keep the experts of the first N layers in host RAM, saving their VRAM so a
        // too-big MoE still fits (cf. llama.cpp --n-cpu-moe). Explicit value disables auto-fit below.
        let ncmoe_explicit = std::env::var("INFR_NCMOE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let mut n_cpu_moe = ncmoe_explicit.unwrap_or(0).min(n_layer);
        let mut moe_stream = std::env::var("INFR_MOE_STREAM").is_ok();
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

        // Pre-flight VRAM check: size the resident weights up front and verify they fit before
        // uploading any tensor — turns a cryptic mid-load allocator OOM into a clear early error.
        // (KV cache + activation scratch are allocated later by `new_kv`/the forward, not here.)
        let fp = weight_footprint(&g);
        let vram = be.vram();
        let gb = |b: u64| b as f64 / 1e9;
        // GPU KV cache footprint at the target context (`INFR_MAX_CTX`, default 8192): f16 K+V per
        // layer. MoE attention now stores KV in VRAM, so it competes with experts for space.
        let target_ctx = std::env::var("INFR_MAX_CTX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8192);
        let kv_bytes =
            (n_kv * head_dim * 2/*K+V*/ * 2/*f16*/ * n_layer) as u64 * (target_ctx + 64) as u64;
        const ACT_HEADROOM: u64 = 512 * 1024 * 1024; // activation scratch + streaming pool slack
                                                     // MoE auto-fit (default; skipped if INFR_NCMOE is set): keep as many whole expert-layers on
                                                     // the GPU as fit alongside the dense weights, the ctx KV cache, and scratch — offload the
                                                     // overflow. Forced offload defaults to streaming (GPU-via-pool, ~10x the CPU path).
        if moe.is_some() && ncmoe_explicit.is_none() {
            let per_layer = (fp.expert / n_layer.max(1) as u64).max(1);
            let budget = vram
                .available
                .saturating_sub(fp.dense + kv_bytes + ACT_HEADROOM);
            let gpu_layers = (budget / per_layer).min(n_layer as u64) as usize;
            n_cpu_moe = n_layer - gpu_layers;
            if n_cpu_moe > 0 {
                moe_stream = true;
            }
            eprintln!(
                "MoE auto-fit: {gpu_layers}/{n_layer} expert layers on GPU, {n_cpu_moe} {} \
                 (ctx={target_ctx} → KV {:.2} GB)",
                if n_cpu_moe == 0 {
                    "all resident"
                } else if moe_stream {
                    "streamed"
                } else {
                    "on CPU"
                },
                gb(kv_bytes),
            );
        }
        // Experts of the first `n_cpu_moe` layers live in host RAM → subtract their
        // (uniform-per-layer) share from the VRAM total. The router/dense weights stay on GPU.
        let cpu_expert_bytes = if n_layer > 0 {
            fp.expert * n_cpu_moe as u64 / n_layer as u64
        } else {
            0
        };
        let gpu_total = fp.total() - cpu_expert_bytes;
        let experts = if fp.expert > 0 {
            let cpu = if n_cpu_moe > 0 {
                format!(
                    ", {n_cpu_moe} layers' experts on CPU = -{:.2} GB",
                    gb(cpu_expert_bytes)
                )
            } else {
                String::new()
            };
            format!(", experts {:.2} GB{cpu}", gb(fp.expert))
        } else {
            String::new()
        };
        // KV reservation only applies once the model has a GPU KV cache (MoE here; dense uses its own
        // path). Reserve it so the later `new_kv` allocation fits alongside the weights.
        let kv_reserve = if moe.is_some() { kv_bytes } else { 0 };
        eprintln!(
            "weights {:.2} GB on GPU (dense {:.2} GB{}) + KV {:.2} GB (ctx={target_ctx}) | \
             VRAM {:.2} GB {} / {:.2} GB total",
            gb(gpu_total),
            gb(fp.dense),
            experts,
            gb(kv_reserve),
            gb(vram.available),
            if vram.live { "free" } else { "total*" },
            gb(vram.total),
        );
        if gpu_total + kv_reserve + ACT_HEADROOM > vram.available {
            bail!(
                "weights {:.2} GB + KV {:.2} GB + {:.0} MB scratch exceed the {:.2} GB VRAM available \
                 (total {:.2} GB) — use a smaller quant/ctx, free GPU memory, or set INFR_NCMOE",
                gb(gpu_total),
                gb(kv_reserve),
                ACT_HEADROOM as f64 / 1e6,
                gb(vram.available),
                gb(vram.total),
            );
        }
        // Reserve the GPU-resident weight VRAM up front as one contiguous bump arena (frees in one
        // shot, no per-tensor fragmentation). Best-effort: if the contiguous block can't be obtained,
        // fall back to per-tensor allocation rather than failing the load.
        if let Err(e) = be.reserve_weights(gpu_total) {
            eprintln!("note: weight arena reservation failed ({e}); using per-tensor allocation");
        }

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
            // MoE layer: router + per-expert bank. Dense layer: fused gate‖up + down.
            let ffn = if let Some(mc) = moe {
                load_moe(
                    &be,
                    &g,
                    &format!("blk.{l}."),
                    mc.n_expert,
                    l < n_cpu_moe,
                    n_cpu_moe == 0,
                    mc.n_ff_exp * n_embd,
                )?
            } else {
                FfnWt::Dense {
                    wgateup: build_wgateup(&be, &g, &format!("blk.{l}."))?,
                    wdown: up(&be, p("ffn_down.weight"))?,
                }
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
                ffn,
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
            moe,
        };
        let llama = Self {
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
            moe_stream,
            gguf: g,
        };
        // Compile all GPU pipelines / first-touch state up front so any later timing (run / bench /
        // serve) measures compute, not one-time setup. Failures here would also fail real inference.
        llama.warmup()?;
        Ok(llama)
    }

    /// Run a tiny prefill + decode (+ both sampler paths) through the real forward to compile every
    /// VkPipeline and first-touch GPU state. The first use of each compute kernel lazily builds its
    /// pipeline (seconds across the whole MoE kernel set); doing it here keeps it out of timed paths.
    pub fn warmup(&self) -> Result<()> {
        // Suppress per-op profiling (INFR_PROF2) during warmup: recorders read the env at
        // construction, so without this the warmup forwards' submits pollute a subsequent bench's
        // [prof2] aggregate with prefill labels (the stage profiler does the same dance).
        let prof2 = std::env::var_os("INFR_PROF2");
        if prof2.is_some() {
            std::env::remove_var("INFR_PROF2");
        }
        let r = self.warmup_inner();
        if let Some(v) = prof2 {
            std::env::set_var("INFR_PROF2", v);
        }
        r
    }

    fn warmup_inner(&self) -> Result<()> {
        let prompt: Vec<u32> = (0..64).map(|i| (i % 64) as u32).collect();
        if self.cfg.moe.is_some() {
            let mut kv = self.new_moe_kv(96)?;
            self.forward_moe_chunk(&[1u32], &mut kv)?; // shallow decode → non-split attention
            self.forward_moe_chunk(&prompt, &mut kv)?; // prefill: flash attn, routing, gather/scatter, mmq/gemv, accumulate
            self.forward_moe_chunk(&[1u32], &mut kv)?; // deep decode → split-K attn, multi-slot FFN, top-k
            let greedy = SampleParams {
                temp: 0.0,
                top_k: 1,
                top_p: 1.0,
                u: 0.0,
            };
            self.forward_moe_chunk_g(&[1u32], &mut kv, Some(greedy))?; // argmax
            let stoch = SampleParams {
                temp: 0.6,
                top_k: 20,
                top_p: 0.95,
                u: 0.5,
            };
            self.forward_moe_chunk_g(&[1u32], &mut kv, Some(stoch))?; // moe_sample (radix top-k)
        } else {
            let mut kv = self.new_kv(96)?;
            self.forward_resident_kv(&[1u32], &mut kv)?;
            self.forward_resident_kv(&prompt, &mut kv)?;
            self.forward_resident_kv(&[1u32], &mut kv)?;
        }
        Ok(())
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
            let gu = self.lin(layer.wgateup().f16(), &hn2, t, ne, 2 * c.n_ff);
            let mut act = vec![0f32; t * c.n_ff];
            for r in 0..t {
                for i in 0..c.n_ff {
                    let g = gu[r * 2 * c.n_ff + i];
                    act[r * c.n_ff + i] = silu(g) * gu[r * 2 * c.n_ff + c.n_ff + i];
                }
            }
            let down = self.lin(layer.wdown().f16(), &act, t, c.n_ff, ne);
            for i in 0..t * ne {
                hidden[i] += down[i];
            }
        }

        // final norm on the last row, then lm_head
        let last = &hidden[(t - 1) * ne..t * ne];
        let normed = rmsnorm_rows(last, &self.output_norm, 1, ne, c.rms_eps);
        self.lin(self.lm_head.f16(), &normed, 1, ne, c.vocab)
    }

    /// Eager GPU GEMV `y = x·Wᵀ` for any weight kind (f16 / unified-Q / native), one submit. Uploads
    /// `x`, runs the matching recorder op, reads back `y`. Used by the MoE forward (many small,
    /// data-dependent matmuls that can't be baked into one resident command buffer).
    fn gemv_wt(
        &self,
        w: &Wt,
        x: &[f32],
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(x.len(), rows * in_f);
        let xb = self
            .be
            .alloc((rows * in_f).max(1) * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        self.be
            .upload(xb.as_ref(), bytemuck::cast_slice(x))
            .map_err(|e| anyhow!("{e}"))?;
        let yb = self
            .be
            .alloc((rows * out_f).max(1) * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        match w {
            Wt::F16(b) => rec.linear(b.as_ref(), xb.as_ref(), yb.as_ref(), rows, in_f, out_f),
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
                xb.as_ref(),
                yb.as_ref(),
                rows,
                in_f,
                out_f,
                *bits,
                *blk_shift,
            ),
            Wt::Native { buf, dtype } => rec.linear_native(
                *dtype,
                buf.as_ref(),
                xb.as_ref(),
                yb.as_ref(),
                rows,
                in_f,
                out_f,
            ),
        }
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        let mut out = vec![0u8; rows * out_f * 4];
        self.be
            .download(yb.as_ref(), &mut out)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(bytemuck::cast_slice(&out).to_vec())
    }

    /// Batched eager GEMV: record many independent `y = x·Wᵀ` into ONE command buffer / submit and
    /// read them all back. Cuts per-op submit+wait latency (the MoE bottleneck — ~1400 tiny matmuls
    /// per token). Each op is `(weight, x, rows, in_f, out_f)`; returns one output vec per op.
    fn gemv_wt_many(&self, ops: &[(&Wt, &[f32], usize, usize, usize)]) -> Result<Vec<Vec<f32>>> {
        let mut xbufs = Vec::with_capacity(ops.len());
        let mut ybufs = Vec::with_capacity(ops.len());
        for &(_, x, rows, in_f, _) in ops {
            debug_assert_eq!(x.len(), rows * in_f);
            let xb = self
                .be
                .alloc((x.len()).max(1) * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            self.be
                .upload(xb.as_ref(), bytemuck::cast_slice(x))
                .map_err(|e| anyhow!("{e}"))?;
            xbufs.push(xb);
        }
        for &(_, _, rows, _, out_f) in ops {
            let yb = self
                .be
                .alloc((rows * out_f).max(1) * 4, BufferUsage::Readback)
                .map_err(|e| anyhow!("{e}"))?;
            ybufs.push(yb);
        }
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        for (i, &(w, _, rows, in_f, out_f)) in ops.iter().enumerate() {
            let (xb, yb) = (xbufs[i].as_ref(), ybufs[i].as_ref());
            match w {
                Wt::F16(b) => rec.linear(b.as_ref(), xb, yb, rows, in_f, out_f),
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
                    xb,
                    yb,
                    rows,
                    in_f,
                    out_f,
                    *bits,
                    *blk_shift,
                ),
                Wt::Native { buf, dtype } => {
                    rec.linear_native(*dtype, buf.as_ref(), xb, yb, rows, in_f, out_f)
                }
            }
        }
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        let mut outs = Vec::with_capacity(ops.len());
        for (i, &(_, _, rows, _, out_f)) in ops.iter().enumerate() {
            let mut o = vec![0u8; rows * out_f * 4];
            self.be
                .download(ybufs[i].as_ref(), &mut o)
                .map_err(|e| anyhow!("{e}"))?;
            outs.push(bytemuck::cast_slice(&o).to_vec());
        }
        Ok(outs)
    }

    /// One-shot MoE forward over `tokens` (fresh cache) — returns last-position logits. Thin wrapper
    /// over [`forward_moe_chunk`](Self::forward_moe_chunk); used for tests / single-logit checks.
    pub fn forward_moe(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut kv = self.new_moe_kv(tokens.len() + 8)?;
        self.forward_moe_chunk(tokens, &mut kv)
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
                layer.wgateup().f16(),
                hn2.as_ref(),
                gu.as_ref(),
                t,
                ne,
                2 * nff,
            );
            rec.silu_mul_fused(gu.as_ref(), act.as_ref(), t, nff);
            rec.linear(layer.wdown().f16(), act.as_ref(), down.as_ref(), t, nff, ne);
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
                    if *bits == 4 && k.is_multiple_of(32) && outf.is_multiple_of(64) && use_mmq {
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
                // Native-block prefill: coopmat tiled GEMM with in-shader dequant (decode-once per
                // weight element, reused across the row tile). Needs n%64, k%32 (all projections
                // satisfy); else fall back to the native GEMV.
                Wt::Native { buf, dtype } => {
                    if outf.is_multiple_of(64) && k.is_multiple_of(32) {
                        rec.matmul_native(*dtype, a, buf.as_ref(), cbuf, rows, k, outf)
                    } else {
                        rec.linear_native(*dtype, buf.as_ref(), a, cbuf, rows, k, outf)
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
                mm(layer.wgateup(), hn.as_ref(), gu.as_ref(), n, ne, 2 * nff);
                rec.silu_mul_fused(gu.as_ref(), act.as_ref(), n, nff);
                mm(layer.wdown(), act.as_ref(), down.as_ref(), n, nff, ne);
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
                        layer.wgateup(),
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
                    lin(
                        layer.wgateup(),
                        hn.as_ref(),
                        gu_ffn.as_ref(),
                        n,
                        ne,
                        2 * nff,
                    );
                    rec.silu_mul_fused(gu_ffn.as_ref(), act.as_ref(), n, nff);
                }
                lin_add(
                    layer.wdown(),
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

    /// True for MoE models (qwen3moe) — use [`generate_moe`](Self::generate_moe), not the
    /// KV-resident path (which is dense-only).
    pub fn is_moe(&self) -> bool {
        self.cfg.moe.is_some()
    }

    /// Fresh MoE generation state with a GPU KV cache sized for `max_ctx` tokens.
    pub fn new_moe_kv(&self, max_ctx: usize) -> Result<MoeKv> {
        Ok(MoeKv {
            kv: self.new_kv(max_ctx)?,
            pool: None,
            dec: Some(self.build_decode_scratch(max_ctx)?),
        })
    }

    /// Allocate the persistent decode scratch (Tier 0). Split-K attention buffers are sized for the
    /// worst-case chunk count (`chunk` is clamped to ≥64, so `n_chunks ≤ ceil(max_ctx/64)`).
    fn build_decode_scratch(&self, max_ctx: usize) -> Result<DecodeScratch> {
        let c = &self.cfg;
        let mc = c.moe.expect("moe");
        let (ne, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_kv, c.head_dim);
        let nblk = ne / 32;
        let ncm = max_ctx.div_ceil(64); // worst-case split-K chunk count
        let af = |n: usize| {
            self.be
                .alloc((n * 4).max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };
        let ab = |bytes: usize| {
            self.be
                .alloc(bytes.max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };
        Ok(DecodeScratch {
            hidden: af(ne)?,
            hn: af(ne)?,
            hn2: af(ne)?,
            ao: af(ne)?,
            qr: af(nh * hd)?,
            kr: af(nkv * hd)?,
            vr: af(nkv * hd)?,
            q_f16: ab(nh * hd * 2)?,
            attn: af(nh * hd)?,
            g: af(mc.n_used * mc.n_ff_exp)?,
            u: af(mc.n_used * mc.n_ff_exp)?,
            act: af(mc.n_used * mc.n_ff_exp)?,
            y: af(mc.n_used * ne)?,
            logits: af(mc.n_expert)?,
            ids: af(mc.n_used)?,
            wts: af(mc.n_used)?,
            qa: ab(ne)?,
            dact: ab(nblk * 2)?,
            sact: ab(nblk * 2)?,
            pm: af(nh * ncm)?,
            pl: af(nh * ncm)?,
            pacc: af(nh * ncm * hd)?,
        })
    }

    /// GPU attention for one MoE layer: upload the raw Q/K/V projections, then record QK-norm + RoPE
    /// (Q → f16, K → the f16 KV cache at `pos`), V → cache, and causal GQA over the cache — reusing
    /// the dense path's kernels. Returns the attention output `[n, nh*hd]` (host f32).
    #[allow(clippy::too_many_arguments)]
    fn moe_attention(
        &self,
        layer: &LayerWeights,
        q_raw: &[f32],
        k_raw: &[f32],
        v_raw: &[f32],
        kv: &KvCache,
        li: usize,
        n: usize,
        pos: usize,
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (nh, nkv, hd) = (c.n_head, c.n_kv, c.head_dim);
        let kvrow = nkv * hd;
        let up = |data: &[f32]| -> Result<Box<dyn Buffer>> {
            let b = self
                .be
                .alloc(data.len() * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            self.be
                .upload(b.as_ref(), bytemuck::cast_slice(data))
                .map_err(|e| anyhow!("{e}"))?;
            Ok(b)
        };
        let qr = up(q_raw)?;
        let kr = up(k_raw)?;
        let vr = up(v_raw)?;
        let q_f16 = self
            .be
            .alloc(n * nh * hd * 2, BufferUsage::Activations)
            .map_err(|e| anyhow!("{e}"))?;
        let attn = self
            .be
            .alloc(n * nh * hd * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        let (qn, kn) = (
            layer.q_norm_buf.as_ref().unwrap().as_ref(),
            layer.k_norm_buf.as_ref().unwrap().as_ref(),
        );
        rec.qk_norm_rope(
            qr.as_ref(),
            qn,
            q_f16.as_ref(),
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
            kn,
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
        rec.store_f16(vr.as_ref(), kv.v[li].as_ref(), n * kvrow, pos * kvrow);
        // Single-token decode (n==1) at depth: split each head's KV range across many workgroups
        // (flash-decode split-K, partials in pm/pl/pacc) so attention isn't stuck on `nh` workgroups
        // grinding the whole cache serially — the dense path's decode kernel. Prefill (n>1) uses the
        // basic per-(token,head) attention_kv. ~32 chunks/head saturates pass-1's KV bandwidth.
        let kv_len = pos + n;
        let chunk = (kv_len / 32).clamp(64, 512);
        if n == 1 && kv_len > chunk {
            let n_chunks = kv_len.div_ceil(chunk);
            let al = |elems: usize| -> Result<Box<dyn Buffer>> {
                self.be
                    .alloc((elems * 4).max(4), BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))
            };
            let pm = al(nh * n_chunks)?;
            let pl = al(nh * n_chunks)?;
            let pacc = al(nh * n_chunks * hd)?;
            rec.attention_kv_split(
                q_f16.as_ref(),
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
                q_f16.as_ref(),
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
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        let mut out = vec![0u8; n * nh * hd * 4];
        self.be
            .download(attn.as_ref(), &mut out)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(bytemuck::cast_slice(&out).to_vec())
    }

    /// Eager native GEMV `y = x·Wᵀ` against an already-resident GPU weight buffer (a streaming
    /// `ExpertPool` slot holding raw native blocks), one submit. Like `gemv_wt` but the weight is a
    /// borrowed buffer + dtype rather than an owned `Wt`.
    fn gemv_native_one(
        &self,
        w: &dyn Buffer,
        dtype: infr_core::DType,
        x: &[f32],
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<Vec<f32>> {
        let xb = self
            .be
            .alloc((x.len()).max(1) * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        self.be
            .upload(xb.as_ref(), bytemuck::cast_slice(x))
            .map_err(|e| anyhow!("{e}"))?;
        let yb = self
            .be
            .alloc((rows * out_f).max(1) * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        rec.linear_native(dtype, w, xb.as_ref(), yb.as_ref(), rows, in_f, out_f);
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        let mut out = vec![0u8; rows * out_f * 4];
        self.be
            .download(yb.as_ref(), &mut out)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(bytemuck::cast_slice(&out).to_vec())
    }

    /// Final norm + lm head from a single resident hidden row `src` [n_embd]. With a sampling spec
    /// the token is chosen on the GPU — argmax for greedy, or temp/top-k/top-p sampling — and only
    /// the 4-byte token id reads back; without one (or for an unsupported top_k) the full vocab
    /// logits read back for host sampling.
    fn lm_head_out(&self, src: &dyn Buffer, sample: Option<SampleParams>) -> Result<GenOut> {
        let c = &self.cfg;
        let al = |n: usize| {
            self.be
                .alloc((n * 4).max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };
        let (normed, final_logits) = (al(c.n_embd)?, al(c.vocab)?);
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        rec.rmsnorm(
            src,
            self.output_norm_buf.as_ref(),
            normed.as_ref(),
            1,
            c.n_embd,
            c.rms_eps,
        );
        rec.label_next("vocab");
        rec_linear(
            &rec,
            &self.lm_head,
            normed.as_ref(),
            final_logits.as_ref(),
            1,
            c.n_embd,
            c.vocab,
        );
        let tok = self
            .be
            .alloc(4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        // GPU-sample when possible: greedy → argmax; temp/top-k/top-p (2 ≤ top_k ≤ KMAX) → sample.
        let gpu_tok = match sample {
            Some(sp) if sp.greedy() => {
                rec.argmax(final_logits.as_ref(), tok.as_ref(), c.vocab);
                true
            }
            Some(sp) if sp.gpu_capable() => {
                rec.sample(
                    final_logits.as_ref(),
                    tok.as_ref(),
                    c.vocab,
                    sp.top_k,
                    sp.temp,
                    sp.top_p,
                    sp.u,
                );
                true
            }
            _ => false,
        };
        if gpu_tok {
            rec.finish().map_err(|e| anyhow!("{e}"))?;
            let mut tb = [0u8; 4];
            self.be
                .download(tok.as_ref(), &mut tb)
                .map_err(|e| anyhow!("{e}"))?;
            Ok(GenOut::Token(u32::from_ne_bytes(tb)))
        } else {
            rec.finish().map_err(|e| anyhow!("{e}"))?;
            let mut out = vec![0u8; c.vocab * 4];
            self.be
                .download(final_logits.as_ref(), &mut out)
                .map_err(|e| anyhow!("{e}"))?;
            Ok(GenOut::Logits(bytemuck::cast_slice(&out).to_vec()))
        }
    }

    /// GPU-resident single-token decode (qwen3moe, all experts on GPU): the residual stream stays in
    /// VRAM the whole layer — rmsnorm / QKV / attention / O / residual / ffn-norm / router are one
    /// recorder, then (after reading back only the router logits for top-k) the selected experts'
    /// gate/up/SiLU/down + weighted accumulate (`hidden += w_e·y_e`) are a second recorder. Only the
    /// `n_expert` logits cross the PCIe bus per layer — no per-matmul host round-trip. When `greedy`,
    /// samples on the GPU and returns just the token; else returns the vocab logits.
    fn forward_moe_chunk_gpu(
        &self,
        token: u32,
        kv: &mut MoeKv,
        sample: Option<SampleParams>,
    ) -> Result<GenOut> {
        let c = &self.cfg;
        let mc = c.moe.expect("moe");
        let (ne, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_kv, c.head_dim);
        let kvrow = nkv * hd;
        let pos = kv.kv.len;
        let kv_len = pos + 1;
        // Tier 0: persistent decode scratch — reused every token (no per-token alloc/free). Bound as
        // `&Box<dyn Buffer>` so the existing `.as_ref()` call sites are unchanged.
        let dec = kv
            .dec
            .as_ref()
            .expect("decode scratch (built in new_moe_kv)");
        let hidden = &dec.hidden;
        let emb = &self.token_embd[token as usize * ne..(token as usize + 1) * ne];
        self.be
            .upload(hidden.as_ref(), bytemuck::cast_slice(emb))
            .map_err(|e| anyhow!("{e}"))?;
        let (hn, hn2, ao) = (&dec.hn, &dec.hn2, &dec.ao);
        let (qr, kr, vr) = (&dec.qr, &dec.kr, &dec.vr);
        let q_f16 = &dec.q_f16;
        let attn = &dec.attn;
        let (g, u, act, y) = (&dec.g, &dec.u, &dec.act, &dec.y);
        let logits = &dec.logits;
        // GPU-resident routing when the expert format has an id-indexed GEMV: top-k + expert ids and
        // weights stay in VRAM (one submit/layer). Else fall back to host top-k (two submits/layer).
        let (gate_dtype, _) = native_parts(&self.layers[0].moe_stacked().expect("stacked").1.gate);
        let gpu_route =
            infr_vulkan::Recorder::native_id_supported(gate_dtype) && mc.n_expert <= 128;
        let (ids_buf, wts_buf) = if gpu_route {
            (Some(&dec.ids), Some(&dec.wts))
        } else {
            (None, None)
        };
        // Q4_K experts → mmq (dp4a): quantize the ffn-normed row to int8 once (shared by gate+up).
        let mmq = gpu_route && matches!(gate_dtype, infr_core::DType::Q4K);
        let (qa, dact, sact) = if mmq {
            (Some(&dec.qa), Some(&dec.dact), Some(&dec.sact))
        } else {
            (None, None, None)
        };
        // split-K decode attention scratch (parallelize the KV reduction at depth)
        let chunk = (kv_len / 32).clamp(64, 512);
        let use_split = kv_len > chunk;
        let n_chunks = if use_split { kv_len.div_ceil(chunk) } else { 0 };
        let (pm, pl, pacc) = if use_split {
            (Some(&dec.pm), Some(&dec.pl), Some(&dec.pacc))
        } else {
            (None, None, None)
        };

        // Tier 1: the GPU-resident (gpu_route) path records ALL 48 layers into ONE command buffer and
        // submits once — vs a recorder + `queue_submit`/`queue_wait_idle` (a full GPU drain) per layer.
        // Inter-layer hazards on the shared scratch are serialized by the recorder's barrier tracking,
        // so a single submit is correct. The host-topk fallback still finishes per layer (it needs a
        // mid-layer logits readback), swapping in a fresh recorder via `mem::replace`.
        let mut rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        for (li, layer) in self.layers.iter().enumerate() {
            rec.rmsnorm(
                hidden.as_ref(),
                layer.attn_norm_buf.as_ref(),
                hn.as_ref(),
                1,
                ne,
                c.rms_eps,
            );
            rec_linear(&rec, &layer.wq, hn.as_ref(), qr.as_ref(), 1, ne, nh * hd);
            rec_linear(&rec, &layer.wk, hn.as_ref(), kr.as_ref(), 1, ne, nkv * hd);
            rec_linear(&rec, &layer.wv, hn.as_ref(), vr.as_ref(), 1, ne, nkv * hd);
            let (qn, kn) = (
                layer.q_norm_buf.as_ref().unwrap().as_ref(),
                layer.k_norm_buf.as_ref().unwrap().as_ref(),
            );
            rec.qk_norm_rope(
                qr.as_ref(),
                qn,
                q_f16.as_ref(),
                1,
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
                kn,
                kv.kv.k[li].as_ref(),
                1,
                nkv,
                hd,
                c.rope_dim,
                c.rope_theta,
                pos,
                pos,
                c.rms_eps,
            );
            rec.store_f16(vr.as_ref(), kv.kv.v[li].as_ref(), kvrow, pos * kvrow);
            if use_split {
                rec.attention_kv_split(
                    q_f16.as_ref(),
                    kv.kv.k[li].as_ref(),
                    kv.kv.v[li].as_ref(),
                    attn.as_ref(),
                    pm.as_ref().unwrap().as_ref(),
                    pl.as_ref().unwrap().as_ref(),
                    pacc.as_ref().unwrap().as_ref(),
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    chunk,
                    n_chunks,
                );
            } else {
                rec.attention_kv(
                    q_f16.as_ref(),
                    kv.kv.k[li].as_ref(),
                    kv.kv.v[li].as_ref(),
                    attn.as_ref(),
                    1,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                );
            }
            rec_linear(&rec, &layer.wo, attn.as_ref(), ao.as_ref(), 1, nh * hd, ne);
            rec.add(hidden.as_ref(), ao.as_ref(), hidden.as_ref(), ne); // residual
            rec.rmsnorm(
                hidden.as_ref(),
                layer.ffn_norm_buf.as_ref(),
                hn2.as_ref(),
                1,
                ne,
                c.rms_eps,
            );
            let (gate_inp, st) = layer.moe_stacked().expect("stacked experts");
            rec_linear(
                &rec,
                gate_inp,
                hn2.as_ref(),
                logits.as_ref(),
                1,
                ne,
                mc.n_expert,
            );

            if let (Some(ids), Some(wts)) = (&ids_buf, &wts_buf) {
                // Fully GPU-resident: top-k on the GPU writes expert ids + weights to VRAM, then the
                // selected experts' FFN (id-indexed gather of the stacked weights) accumulates into
                // hidden — all in this one recorder. No readback, one submit/layer.
                rec.moe_topk(
                    logits.as_ref(),
                    ids.as_ref(),
                    wts.as_ref(),
                    1,
                    mc.n_expert,
                    mc.n_used,
                    mc.scale,
                );
                // Fused: all n_used experts per role in ONE dispatch (concurrent, no inter-expert
                // barrier). gate/up read the shared ffn-normed row; down reads each slot's activation.
                let (gd, gb) = native_parts(&st.gate);
                let (ud, ub) = native_parts(&st.up);
                let (dd, db) = native_parts(&st.down);
                let nu = mc.n_used;
                if let (Some(qa), Some(da), Some(sa)) = (&qa, &dact, &sact) {
                    // Q4_K gate/up via dp4a (mmq): quantize the ffn-normed row to int8 once, shared.
                    rec.quant_q8(hn2.as_ref(), qa.as_ref(), da.as_ref(), sa.as_ref(), 1, ne);
                    rec.linear_mmv_id_multi_q4k(
                        gb,
                        qa.as_ref(),
                        da.as_ref(),
                        sa.as_ref(),
                        ids.as_ref(),
                        nu,
                        st.stride,
                        g.as_ref(),
                        ne,
                        mc.n_ff_exp,
                    );
                    rec.linear_mmv_id_multi_q4k(
                        ub,
                        qa.as_ref(),
                        da.as_ref(),
                        sa.as_ref(),
                        ids.as_ref(),
                        nu,
                        st.stride,
                        u.as_ref(),
                        ne,
                        mc.n_ff_exp,
                    );
                } else {
                    rec.linear_native_id_multi(
                        gd,
                        gb,
                        ids.as_ref(),
                        nu,
                        st.stride,
                        hn2.as_ref(),
                        false,
                        g.as_ref(),
                        ne,
                        mc.n_ff_exp,
                    );
                    rec.linear_native_id_multi(
                        ud,
                        ub,
                        ids.as_ref(),
                        nu,
                        st.stride,
                        hn2.as_ref(),
                        false,
                        u.as_ref(),
                        ne,
                        mc.n_ff_exp,
                    );
                }
                rec.silu_mul(g.as_ref(), u.as_ref(), act.as_ref(), nu * mc.n_ff_exp);
                rec.linear_native_id_multi(
                    dd,
                    db,
                    ids.as_ref(),
                    nu,
                    st.stride,
                    act.as_ref(),
                    true,
                    y.as_ref(),
                    mc.n_ff_exp,
                    ne,
                );
                rec.moe_accumulate(y.as_ref(), wts.as_ref(), hidden.as_ref(), ne, nu);
                // Tier 1: do NOT finish — keep recording the next layer into the same buffer.
            } else {
                // Fallback (non-id-capable expert format): host top-k needs this layer's logits, so
                // finish here and continue the next layer in a fresh recorder.
                let done =
                    std::mem::replace(&mut rec, self.be.recorder().map_err(|e| anyhow!("{e}"))?);
                done.finish().map_err(|e| anyhow!("{e}"))?;
                let mut lb = vec![0u8; mc.n_expert * 4];
                self.be
                    .download(logits.as_ref(), &mut lb)
                    .map_err(|e| anyhow!("{e}"))?;
                let (idx, weights) = moe_topk(bytemuck::cast_slice(&lb), &mc);
                let rec2 = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
                for (ki, &e) in idx.iter().enumerate() {
                    rec_linear_expert(
                        &rec2,
                        &st.gate,
                        e,
                        st.stride,
                        hn2.as_ref(),
                        g.as_ref(),
                        1,
                        ne,
                        mc.n_ff_exp,
                    );
                    rec_linear_expert(
                        &rec2,
                        &st.up,
                        e,
                        st.stride,
                        hn2.as_ref(),
                        u.as_ref(),
                        1,
                        ne,
                        mc.n_ff_exp,
                    );
                    rec2.silu_mul(g.as_ref(), u.as_ref(), act.as_ref(), mc.n_ff_exp);
                    rec_linear_expert(
                        &rec2,
                        &st.down,
                        e,
                        st.stride,
                        act.as_ref(),
                        y.as_ref(),
                        1,
                        mc.n_ff_exp,
                        ne,
                    );
                    rec2.add_scaled(y.as_ref(), hidden.as_ref(), weights[ki], ne);
                }
                rec2.finish().map_err(|e| anyhow!("{e}"))?;
            }
        }
        // gpu_route: the single submit for all 48 layers. host fallback: a trailing empty recorder.
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        kv.kv.len += 1;

        // final norm + lm head (on the GPU); greedy → GPU argmax + 4-byte token readback.
        self.lm_head_out(hidden.as_ref(), sample)
    }

    /// GPU-resident grouped prefill (qwen3moe, all experts on GPU): like [`forward_moe_chunk_gpu`]
    /// but for a multi-token chunk. The residual stream stays in VRAM; recorder #1 does
    /// rmsnorm → QKV → attention → O → residual → ffn-norm → router for all `t` tokens; only the
    /// `t*n_expert` router logits read back for host top-k. Recorder #2 runs the FFN grouped by
    /// expert — for each active expert: gather its token rows on the GPU, one SwiGLU GEMM, then a
    /// weighted scatter-add back into the resident hidden. No per-expert host round-trip. Returns
    /// last-token logits.
    fn forward_moe_chunk_gpu_prefill(
        &self,
        tokens: &[u32],
        kv: &mut MoeKv,
        sample: Option<SampleParams>,
    ) -> Result<GenOut> {
        let c = &self.cfg;
        let mc = c.moe.expect("moe");
        let t = tokens.len();
        let (ne, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_kv, c.head_dim);
        let nff = mc.n_ff_exp;
        let kvrow = nkv * hd;
        let pos = kv.kv.len;
        let kv_len = pos + t;
        let al = |n: usize| {
            self.be
                .alloc((n * 4).max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };
        let ab = |bytes: usize| {
            self.be
                .alloc(bytes.max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };

        // resident scratch (reused across all layers)
        let hidden = al(t * ne)?;
        let mut emb = vec![0f32; t * ne];
        for (i, &tok) in tokens.iter().enumerate() {
            emb[i * ne..(i + 1) * ne]
                .copy_from_slice(&self.token_embd[tok as usize * ne..(tok as usize + 1) * ne]);
        }
        self.be
            .upload(hidden.as_ref(), bytemuck::cast_slice(&emb))
            .map_err(|e| anyhow!("{e}"))?;
        // Projections (QKV/O) run as tiled GEMMs → outputs are M-padded to gmp = ceil(t/64)*64.
        let gmp = t.div_ceil(64) * 64;
        let (hn, hn2) = (al(t * ne)?, al(t * ne)?);
        let ao = al(gmp * ne)?;
        let (qr, kr, vr) = (al(gmp * nh * hd)?, al(gmp * nkv * hd)?, al(gmp * nkv * hd)?);
        // Q4_K Q/K/O projections use dp4a (mmq): quantize the projection inputs (hn for Q/K, attn for
        // O) to int8 once each. q4_proj gates on Q (q/k/o are Q4_K in this model; v is Q6_K → coopmat).
        let q4_proj = matches!(native_parts(&self.layers[0].wq).0, infr_core::DType::Q4K);
        let qbufs = |in_f: usize| -> Result<(Box<dyn Buffer>, Box<dyn Buffer>, Box<dyn Buffer>)> {
            Ok((
                ab(gmp * in_f)?,
                ab(gmp * (in_f / 32) * 2)?,
                ab(gmp * (in_f / 32) * 2)?,
            ))
        };
        let (qa_h, da_h, sa_h, qa_o, da_o, sa_o) = if q4_proj {
            let (a, b, c2) = qbufs(ne)?;
            let (d, e, f) = qbufs(nh * hd)?;
            (Some(a), Some(b), Some(c2), Some(d), Some(e), Some(f))
        } else {
            (None, None, None, None, None, None)
        };
        // Flash prefill attention (split-K, register-blocked, never materializes the score matrix) is
        // hd=128-specialized and wants 64-row tiles → pad q/attn to mpad rows. Small chunks (t<64) or
        // other head dims fall back to the basic per-query attention_kv. INFR_NO_FLASH forces fallback.
        let use_flash = hd == 128 && t >= 64 && std::env::var("INFR_NO_FLASH").is_err();
        let mpad = if use_flash { t.div_ceil(64) * 64 } else { t };
        let q_f16 = self
            .be
            .alloc(mpad * nh * hd * 2, BufferUsage::Activations)
            .map_err(|e| anyhow!("{e}"))?;
        let attn = al(mpad * nh * hd)?;
        // Flash split-K scratch: po=[≤8,mpad,nh,hd] partials, pm/pl=[≤8,mpad,nh] (reused across layers).
        let flash = if use_flash {
            Some((
                al(8 * mpad * nh * hd)?,
                al(8 * mpad * nh)?,
                al(8 * mpad * nh)?,
            ))
        } else {
            None
        };
        let logits = al(t * mc.n_expert)?;
        // GPU routing (n_expert ≤ 128 for the top-k workgroup): per-token top-k → bucket tokens by
        // expert entirely on the GPU. Only the per-expert counts/offsets (n_expert u32 each) read
        // back, to size the per-expert GEMM dispatches. Else fall back to host top-k + index uploads.
        let gpu_route = mc.n_expert <= 128;
        let n_pairs = t * mc.n_used;
        let rb = |n: usize| {
            self.be
                .alloc((n * 4).max(4), BufferUsage::Readback)
                .map_err(|e| anyhow!("{e}"))
        };
        let route = if gpu_route {
            Some((
                al(n_pairs)?,     // tok_ids
                al(n_pairs)?,     // tok_wts
                rb(mc.n_expert)?, // counts (downloaded)
                rb(mc.n_expert)?, // offsets (downloaded + used on GPU by scatter)
                al(mc.n_expert)?, // fill
                al(n_pairs)?,     // bucket_rows
                al(n_pairs)?,     // bucket_wts
            ))
        } else {
            None
        };

        for (li, layer) in self.layers.iter().enumerate() {
            // recorder 1: attention + router for all t tokens, on the GPU.
            let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
            rec.rmsnorm(
                hidden.as_ref(),
                layer.attn_norm_buf.as_ref(),
                hn.as_ref(),
                t,
                ne,
                c.rms_eps,
            );
            if let (Some(qa), Some(da), Some(sa)) = (&qa_h, &da_h, &sa_h) {
                // Q4_K Q/K via dp4a (quantize hn once); V (Q6_K) via coopmat.
                rec.quant_q8(hn.as_ref(), qa.as_ref(), da.as_ref(), sa.as_ref(), t, ne);
                let (_, wqb) = native_parts(&layer.wq);
                let (_, wkb) = native_parts(&layer.wk);
                rec.matmul_mmq_q4k(
                    qa.as_ref(),
                    da.as_ref(),
                    sa.as_ref(),
                    wqb,
                    0,
                    qr.as_ref(),
                    t,
                    ne,
                    nh * hd,
                );
                rec.matmul_mmq_q4k(
                    qa.as_ref(),
                    da.as_ref(),
                    sa.as_ref(),
                    wkb,
                    0,
                    kr.as_ref(),
                    t,
                    ne,
                    nkv * hd,
                );
            } else {
                rec_proj(&rec, &layer.wq, hn.as_ref(), qr.as_ref(), t, ne, nh * hd);
                rec_proj(&rec, &layer.wk, hn.as_ref(), kr.as_ref(), t, ne, nkv * hd);
            }
            rec_proj(&rec, &layer.wv, hn.as_ref(), vr.as_ref(), t, ne, nkv * hd);
            let (qn, kn) = (
                layer.q_norm_buf.as_ref().unwrap().as_ref(),
                layer.k_norm_buf.as_ref().unwrap().as_ref(),
            );
            rec.qk_norm_rope(
                qr.as_ref(),
                qn,
                q_f16.as_ref(),
                t,
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
                kn,
                kv.kv.k[li].as_ref(),
                t,
                nkv,
                hd,
                c.rope_dim,
                c.rope_theta,
                pos,
                pos,
                c.rms_eps,
            );
            rec.store_f16(vr.as_ref(), kv.kv.v[li].as_ref(), t * kvrow, pos * kvrow);
            if let Some((po, pm, pl)) = &flash {
                rec.attention_prefill_flash(
                    q_f16.as_ref(),
                    kv.kv.k[li].as_ref(),
                    kv.kv.v[li].as_ref(),
                    attn.as_ref(),
                    po.as_ref(),
                    pm.as_ref(),
                    pl.as_ref(),
                    t,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                );
            } else {
                rec.attention_kv(
                    q_f16.as_ref(),
                    kv.kv.k[li].as_ref(),
                    kv.kv.v[li].as_ref(),
                    attn.as_ref(),
                    t,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                );
            }
            if let (Some(qa), Some(da), Some(sa)) = (&qa_o, &da_o, &sa_o) {
                rec.quant_q8(
                    attn.as_ref(),
                    qa.as_ref(),
                    da.as_ref(),
                    sa.as_ref(),
                    t,
                    nh * hd,
                );
                let (_, wob) = native_parts(&layer.wo);
                rec.matmul_mmq_q4k(
                    qa.as_ref(),
                    da.as_ref(),
                    sa.as_ref(),
                    wob,
                    0,
                    ao.as_ref(),
                    t,
                    nh * hd,
                    ne,
                );
            } else {
                rec_proj(&rec, &layer.wo, attn.as_ref(), ao.as_ref(), t, nh * hd, ne);
            }
            rec.add(hidden.as_ref(), ao.as_ref(), hidden.as_ref(), t * ne); // residual
            rec.rmsnorm(
                hidden.as_ref(),
                layer.ffn_norm_buf.as_ref(),
                hn2.as_ref(),
                t,
                ne,
                c.rms_eps,
            );
            let (gate_inp, st) = layer.moe_stacked().expect("stacked experts");
            rec_linear(
                &rec,
                gate_inp,
                hn2.as_ref(),
                logits.as_ref(),
                t,
                ne,
                mc.n_expert,
            );

            #[allow(clippy::type_complexity, unused_assignments)]
            let mut fallback_bufs: Option<(Box<dyn Buffer>, Box<dyn Buffer>)> = None;
            let (counts_h, offs_h, bucket_rows, bucket_wts) =
                if let Some((tok_ids, tok_wts, counts, offsets, fill, bucket_rows, bucket_wts)) =
                    &route
                {
                    // GPU routing: per-token top-k → count/scan/scatter buckets, all on the GPU.
                    rec.moe_topk(
                        logits.as_ref(),
                        tok_ids.as_ref(),
                        tok_wts.as_ref(),
                        t,
                        mc.n_expert,
                        mc.n_used,
                        mc.scale,
                    );
                    rec.zero(counts.as_ref(), mc.n_expert);
                    rec.moe_bucket_count(tok_ids.as_ref(), counts.as_ref(), n_pairs);
                    rec.moe_bucket_scan(
                        counts.as_ref(),
                        offsets.as_ref(),
                        fill.as_ref(),
                        mc.n_expert,
                    );
                    rec.moe_bucket_scatter(
                        tok_ids.as_ref(),
                        tok_wts.as_ref(),
                        offsets.as_ref(),
                        fill.as_ref(),
                        bucket_rows.as_ref(),
                        bucket_wts.as_ref(),
                        n_pairs,
                        mc.n_used,
                    );
                    rec.finish().map_err(|e| anyhow!("{e}"))?;
                    // Read back only the per-expert counts + offsets (n_expert u32 each) to size dispatches.
                    let mut cb = vec![0u8; mc.n_expert * 4];
                    let mut ob = vec![0u8; mc.n_expert * 4];
                    self.be
                        .download(counts.as_ref(), &mut cb)
                        .map_err(|e| anyhow!("{e}"))?;
                    self.be
                        .download(offsets.as_ref(), &mut ob)
                        .map_err(|e| anyhow!("{e}"))?;
                    (
                        bytemuck::cast_slice::<u8, u32>(&cb).to_vec(),
                        bytemuck::cast_slice::<u8, u32>(&ob).to_vec(),
                        Some(bucket_rows),
                        Some(bucket_wts),
                    )
                } else {
                    // Fallback: host top-k → per-expert index buffers uploaded to GPU.
                    rec.finish().map_err(|e| anyhow!("{e}"))?;
                    let mut lb = vec![0u8; t * mc.n_expert * 4];
                    self.be
                        .download(logits.as_ref(), &mut lb)
                        .map_err(|e| anyhow!("{e}"))?;
                    let lh: &[f32] = bytemuck::cast_slice(&lb);
                    let mut rows_of: Vec<Vec<u32>> = vec![Vec::new(); mc.n_expert];
                    let mut wts_of: Vec<Vec<f32>> = vec![Vec::new(); mc.n_expert];
                    for r in 0..t {
                        let (idx, w) = moe_topk(&lh[r * mc.n_expert..(r + 1) * mc.n_expert], &mc);
                        for (ki, &e) in idx.iter().enumerate() {
                            rows_of[e].push(r as u32);
                            wts_of[e].push(w[ki]);
                        }
                    }
                    // Concatenate into the shared bucket layout (offsets = prefix sum) and upload once.
                    let mut offs = vec![0u32; mc.n_expert];
                    let mut acc = 0u32;
                    for e in 0..mc.n_expert {
                        offs[e] = acc;
                        acc += rows_of[e].len() as u32;
                    }
                    let mut rows_flat = Vec::with_capacity(n_pairs);
                    let mut wts_flat = Vec::with_capacity(n_pairs);
                    for e in 0..mc.n_expert {
                        rows_flat.extend_from_slice(&rows_of[e]);
                        wts_flat.extend_from_slice(&wts_of[e]);
                    }
                    let br = al(rows_flat.len().max(1))?;
                    let bw = al(wts_flat.len().max(1))?;
                    self.be
                        .upload(br.as_ref(), bytemuck::cast_slice(&rows_flat))
                        .map_err(|e| anyhow!("{e}"))?;
                    self.be
                        .upload(bw.as_ref(), bytemuck::cast_slice(&wts_flat))
                        .map_err(|e| anyhow!("{e}"))?;
                    let counts: Vec<u32> =
                        (0..mc.n_expert).map(|e| rows_of[e].len() as u32).collect();
                    fallback_bufs = Some((br, bw));
                    let (br, bw) = fallback_bufs.as_ref().unwrap();
                    (counts, offs, Some(br), Some(bw))
                };

            // recorder 2: per active expert, gather its bucket slice → SwiGLU GEMM → weighted
            // scatter-add into hidden. m/offset come from the GPU-built (or host) routing.
            let (bucket_rows, bucket_wts) = (bucket_rows.unwrap(), bucket_wts.unwrap());
            let rec2 = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
            let mut keep: Vec<Box<dyn Buffer>> = Vec::new();
            for e in 0..mc.n_expert {
                let m = counts_h[e] as usize;
                if m == 0 {
                    continue;
                }
                let off = offs_h[e] as usize;
                // Tiled GEMM: gate/up/down decode each expert weight ONCE and reuse across the 64-row
                // tile (vs the per-row GEMV re-reading the weight m times). GEMM outputs are M-padded
                // to ceil(m/64)*64 rows (extra rows are zero, ignored by silu/scatter on the first m).
                let mpad = m.div_ceil(64) * 64;
                let (xe, ge, ue, ae, ye) = (
                    al(m * ne)?,
                    al(mpad * nff)?,
                    al(mpad * nff)?,
                    al(m * nff)?,
                    al(mpad * ne)?,
                );
                rec2.gather_rows(hn2.as_ref(), bucket_rows.as_ref(), off, xe.as_ref(), m, ne);
                // gate/up: Q4_K → dp4a (mmq) GEMM (int8 dot, faster than coopmat-f16); quantize the
                // gathered batch to int8 once, shared by both. down (Q6_K) stays on the coopmat GEMM.
                if matches!(native_parts(&st.gate).0, infr_core::DType::Q4K) {
                    let nblk = ne / 32;
                    let (qa, da, sa) = (ab(mpad * ne)?, ab(mpad * nblk * 2)?, ab(mpad * nblk * 2)?);
                    rec2.quant_q8(xe.as_ref(), qa.as_ref(), da.as_ref(), sa.as_ref(), m, ne);
                    let (_, gb) = native_parts(&st.gate);
                    let (_, ub) = native_parts(&st.up);
                    let base = e * st.stride;
                    rec2.label_next("expert_gateup");
                    rec2.matmul_mmq_q4k(
                        qa.as_ref(),
                        da.as_ref(),
                        sa.as_ref(),
                        gb,
                        base,
                        ge.as_ref(),
                        m,
                        ne,
                        nff,
                    );
                    rec2.label_next("expert_gateup");
                    rec2.matmul_mmq_q4k(
                        qa.as_ref(),
                        da.as_ref(),
                        sa.as_ref(),
                        ub,
                        base,
                        ue.as_ref(),
                        m,
                        ne,
                        nff,
                    );
                    keep.extend([qa, da, sa]);
                } else {
                    rec2.label_next("expert_gateup");
                    rec_gemm_expert(
                        &rec2,
                        &st.gate,
                        e,
                        st.stride,
                        xe.as_ref(),
                        ge.as_ref(),
                        m,
                        ne,
                        nff,
                    );
                    rec2.label_next("expert_gateup");
                    rec_gemm_expert(
                        &rec2,
                        &st.up,
                        e,
                        st.stride,
                        xe.as_ref(),
                        ue.as_ref(),
                        m,
                        ne,
                        nff,
                    );
                }
                rec2.silu_mul(ge.as_ref(), ue.as_ref(), ae.as_ref(), m * nff);
                // down: Q6_K → dp4a (mmq) GEMM (int8 dot, faster than coopmat-f16); quantize the
                // SwiGLU activations to int8 per 32-block first. Else coopmat-f16 fallback.
                if matches!(native_parts(&st.down).0, infr_core::DType::Q6K) {
                    let nblk = nff / 32;
                    let (qa, da, sa) =
                        (ab(mpad * nff)?, ab(mpad * nblk * 2)?, ab(mpad * nblk * 2)?);
                    rec2.quant_q8(ae.as_ref(), qa.as_ref(), da.as_ref(), sa.as_ref(), m, nff);
                    let (_, db) = native_parts(&st.down);
                    rec2.label_next("expert_down");
                    rec2.matmul_mmq_q6k(
                        qa.as_ref(),
                        da.as_ref(),
                        db,
                        e * st.stride,
                        ye.as_ref(),
                        m,
                        nff,
                        ne,
                    );
                    keep.extend([qa, da, sa]);
                } else {
                    rec2.label_next("expert_down");
                    rec_gemm_expert(
                        &rec2,
                        &st.down,
                        e,
                        st.stride,
                        ae.as_ref(),
                        ye.as_ref(),
                        m,
                        nff,
                        ne,
                    );
                }
                rec2.scatter_add_rows(
                    ye.as_ref(),
                    bucket_rows.as_ref(),
                    bucket_wts.as_ref(),
                    off,
                    hidden.as_ref(),
                    m,
                    ne,
                );
                keep.extend([xe, ge, ue, ae, ye]);
            }
            rec2.finish().map_err(|e| anyhow!("{e}"))?;
            drop(keep);
        }
        kv.kv.len += t;

        // Gather hidden's last row on the GPU, then final norm + lm head (+ greedy GPU argmax).
        let last_idx = al(1)?;
        self.be
            .upload(last_idx.as_ref(), bytemuck::cast_slice(&[(t - 1) as u32]))
            .map_err(|e| anyhow!("{e}"))?;
        let hlast = al(ne)?;
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        rec.gather_rows(hidden.as_ref(), last_idx.as_ref(), 0, hlast.as_ref(), 1, ne);
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        self.lm_head_out(hlast.as_ref(), sample)
    }

    /// Eager MoE forward for one chunk of `tokens` at positions `kv.pos..`, appending K/V to the
    /// cache (so decode steps process only the new token, not the whole sequence). Returns logits
    /// (`vocab`) for the last token. Same math as [`forward_moe`] but cached.
    pub fn forward_moe_chunk(&self, tokens: &[u32], kv: &mut MoeKv) -> Result<Vec<f32>> {
        match self.forward_moe_chunk_g(tokens, kv, None)? {
            GenOut::Logits(l) => Ok(l),
            GenOut::Token(_) => unreachable!("no sampler always returns logits"),
        }
    }

    /// As [`forward_moe_chunk`] but with on-GPU greedy sampling: when `greedy`, the GPU argmaxes the
    /// vocab logits and only the 4-byte token id crosses the bus (no vocab-logits download).
    fn forward_moe_chunk_g(
        &self,
        tokens: &[u32],
        kv: &mut MoeKv,
        sample: Option<SampleParams>,
    ) -> Result<GenOut> {
        // Stacked GPU expert bank → fully GPU-resident path (no per-matmul host round-trip):
        // single-token decode, or grouped-by-expert prefill for a multi-token chunk. Offloaded /
        // per-expert layers use the eager path.
        if self.layers[0].moe_stacked().is_some() {
            return if tokens.len() == 1 {
                self.forward_moe_chunk_gpu(tokens[0], kv, sample)
            } else {
                self.forward_moe_chunk_gpu_prefill(tokens, kv, sample)
            };
        }
        let c = &self.cfg;
        let mc = c.moe.expect("forward_moe_chunk requires a MoE model");
        let t = tokens.len();
        let (ne, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_kv, c.head_dim);
        let pos0 = kv.kv.len;

        let mut hidden = vec![0f32; t * ne];
        for (i, &tok) in tokens.iter().enumerate() {
            hidden[i * ne..(i + 1) * ne]
                .copy_from_slice(&self.token_embd[tok as usize * ne..(tok as usize + 1) * ne]);
        }

        for (li, layer) in self.layers.iter().enumerate() {
            // attention with GPU KV cache — Q/K/V projections batched into one submit, then QK-norm /
            // RoPE / KV-append / attention on the GPU (reusing the dense kernels via moe_attention).
            let hn = rmsnorm_rows(&hidden, &layer.attn_norm, t, ne, c.rms_eps);
            let mut qkv = self.gemv_wt_many(&[
                (&layer.wq, hn.as_slice(), t, ne, nh * hd),
                (&layer.wk, hn.as_slice(), t, ne, nkv * hd),
                (&layer.wv, hn.as_slice(), t, ne, nkv * hd),
            ])?;
            let vnew = qkv.pop().unwrap();
            let knew = qkv.pop().unwrap();
            let q = qkv.pop().unwrap();
            let attn = self.moe_attention(layer, &q, &knew, &vnew, &kv.kv, li, t, pos0)?;
            let ao = self.gemv_wt(&layer.wo, &attn, t, nh * hd, ne)?;
            for i in 0..t * ne {
                hidden[i] += ao[i];
            }

            // MoE FFN: route each token to top-k experts, weighted SwiGLU sum
            let hn2 = rmsnorm_rows(&hidden, &layer.ffn_norm, t, ne, c.rms_eps);
            let (gate_inp, experts) = layer.moe();
            let logits = self.gemv_wt(gate_inp, &hn2, t, ne, mc.n_expert)?;
            if !experts[0].gate.is_cpu() {
                // All experts GPU-resident → group tokens by expert and run one SwiGLU GEMM per
                // expert (tiled coopmat) instead of `t × n_used` per-token GEMVs.
                let ffn = self.moe_ffn_grouped(&hn2, &logits, experts, &mc, t)?;
                for i in 0..t * ne {
                    hidden[i] += ffn[i];
                }
            } else {
                // Host-offloaded / streamed experts: per-token path (CPU or VRAM pool).
                for r in 0..t {
                    let out_row = self.moe_ffn_token(
                        &hn2[r * ne..(r + 1) * ne],
                        &logits[r * mc.n_expert..(r + 1) * mc.n_expert],
                        experts,
                        &mc,
                        li,
                        &mut kv.pool,
                    )?;
                    for i in 0..ne {
                        hidden[r * ne + i] += out_row[i];
                    }
                }
            }
        }
        kv.kv.len += t;

        // Eager (offloaded) path always returns logits; the caller samples on the host.
        let _ = sample;
        let last = &hidden[(t - 1) * ne..t * ne];
        let normed = rmsnorm_rows(last, &self.output_norm, 1, ne, c.rms_eps);
        Ok(GenOut::Logits(self.gemv_wt(
            &self.lm_head,
            &normed,
            1,
            ne,
            c.vocab,
        )?))
    }

    /// Raw quantized bytes of a host-backed expert's `role` weight ("gate"/"up"/"down"), read
    /// zero-copy from the GGUF mmap. Each expert is a contiguous `1/n_expert` slice of the stacked
    /// `ffn_{role}_exps` tensor.
    fn expert_bytes(&self, li: usize, role: &str, e: usize) -> Result<&[u8]> {
        let name = format!("blk.{li}.ffn_{role}_exps.weight");
        let all = self
            .gguf
            .tensor_bytes(&name)
            .map_err(|er| anyhow!("{er}"))?;
        let n_expert = self.cfg.moe.expect("moe").n_expert;
        let stride = all.len() / n_expert;
        Ok(&all[e * stride..(e + 1) * stride])
    }

    /// (dtype, mmap bytes) for a host-backed expert role — the inputs to a CPU/stream matmul.
    fn host_expert(
        &self,
        ew: &ExpertW,
        li: usize,
        role: &str,
        e: usize,
    ) -> Result<(infr_core::DType, &[u8])> {
        let ExpertW::Cpu { dtype } = ew else {
            unreachable!("host_expert on a GPU expert");
        };
        Ok((*dtype, self.expert_bytes(li, role, e)?))
    }

    /// One token's MoE FFN: softmax router → renormalized top-k → weighted SwiGLU sum over the
    /// selected experts. `x` is the (already ffn-normed) token `[n_embd]`, `rl` its router logits.
    /// `li` = layer index (for streaming-pool keys); `pool` = the streaming VRAM pool (lazily built).
    fn moe_ffn_token(
        &self,
        x: &[f32],
        rl: &[f32],
        experts: &[ExpertWt],
        mc: &MoeConfig,
        li: usize,
        pool: &mut Option<infr_vulkan::ExpertPool>,
    ) -> Result<Vec<f32>> {
        let ne = self.cfg.n_embd;
        let maxl = rl.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = rl.iter().map(|&v| (v - maxl).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for pr in probs.iter_mut() {
            *pr /= sum;
        }
        let mut idx: Vec<usize> = (0..mc.n_expert).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        idx.truncate(mc.n_used);
        let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);

        // Each expert's SwiGLU → `ys[ki]` (down output). Expert placement is per-layer uniform:
        // host-offloaded layers (`INFR_NCMOE`) run on the CPU, or — with `INFR_MOE_STREAM` and a
        // native-supported quant — stream the active experts into a VRAM pool and GPU-compute them;
        // otherwise the experts are GPU-resident and batched.
        let host_layer = !idx.is_empty() && experts[idx[0]].gate.is_cpu();
        let stream_layer = host_layer
            && self.moe_stream
            && matches!(&experts[idx[0]].gate, ExpertW::Cpu { dtype } if is_native_supported(*dtype));
        let ys: Vec<Vec<f32>> = if stream_layer {
            self.stream_experts(x, &idx, experts, mc, li, pool)?
        } else if host_layer {
            idx.iter()
                .map(|&e| {
                    let (gdt, gb) = self.host_expert(&experts[e].gate, li, "gate", e)?;
                    let gate = cpu_expert_matvec(gdt, gb, x, ne, mc.n_ff_exp)?;
                    let (udt, ub) = self.host_expert(&experts[e].up, li, "up", e)?;
                    let up = cpu_expert_matvec(udt, ub, x, ne, mc.n_ff_exp)?;
                    let act: Vec<f32> = (0..mc.n_ff_exp).map(|i| silu(gate[i]) * up[i]).collect();
                    let (ddt, db) = self.host_expert(&experts[e].down, li, "down", e)?;
                    cpu_expert_matvec(ddt, db, &act, mc.n_ff_exp, ne)
                })
                .collect::<Result<_>>()?
        } else {
            // Phase 1: all gate+up matmuls in ONE submit (they all read `x`).
            let mut gu_ops: Vec<(&Wt, &[f32], usize, usize, usize)> =
                Vec::with_capacity(idx.len() * 2);
            for &e in &idx {
                gu_ops.push((experts[e].gate.gpu(), x, 1, ne, mc.n_ff_exp));
                gu_ops.push((experts[e].up.gpu(), x, 1, ne, mc.n_ff_exp));
            }
            let gu = self.gemv_wt_many(&gu_ops)?;
            let acts: Vec<Vec<f32>> = (0..idx.len())
                .map(|ki| {
                    let (gate, up) = (&gu[2 * ki], &gu[2 * ki + 1]);
                    (0..mc.n_ff_exp).map(|i| silu(gate[i]) * up[i]).collect()
                })
                .collect();
            // Phase 2: all down matmuls in ONE submit.
            let down_ops: Vec<(&Wt, &[f32], usize, usize, usize)> = idx
                .iter()
                .enumerate()
                .map(|(ki, &e)| {
                    (
                        experts[e].down.gpu(),
                        acts[ki].as_slice(),
                        1,
                        mc.n_ff_exp,
                        ne,
                    )
                })
                .collect();
            self.gemv_wt_many(&down_ops)?
        };

        // Host weighted accumulate over the renormalized top-k softmax weights.
        let mut out = vec![0f32; ne];
        for (ki, &e) in idx.iter().enumerate() {
            let w_e = probs[e] / wsum * mc.scale;
            for i in 0..ne {
                out[i] += w_e * ys[ki][i];
            }
        }
        Ok(out)
    }

    /// Group-by-expert MoE FFN over a whole chunk of `t` tokens (all experts GPU-resident — the
    /// prefill path). Routes every token to its top-k experts on the host, then for each expert
    /// gathers all of its assigned token rows into one contiguous batch and runs **one** SwiGLU
    /// per expert as a tiled GEMM (`[m_e×ne]·Wᵀ`) — gate+up batched into a single submit, down into
    /// a second — instead of `t × n_used` per-token GEMVs. Scatter-adds the weighted expert outputs
    /// back to each token's row. Returns the `[t*ne]` FFN output to add into the residual stream.
    fn moe_ffn_grouped(
        &self,
        hn2: &[f32],    // [t*ne], ffn-normed token rows
        logits: &[f32], // [t*n_expert], router logits
        experts: &[ExpertWt],
        mc: &MoeConfig,
        t: usize,
    ) -> Result<Vec<f32>> {
        let ne = self.cfg.n_embd;
        let nff = mc.n_ff_exp;

        // Route: per expert, the token rows it must process and their renormalized weights.
        let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); mc.n_expert];
        let mut wts_of: Vec<Vec<f32>> = vec![Vec::new(); mc.n_expert];
        for r in 0..t {
            let (idx, weights) = moe_topk(&logits[r * mc.n_expert..(r + 1) * mc.n_expert], mc);
            for (ki, &e) in idx.iter().enumerate() {
                rows_of[e].push(r);
                wts_of[e].push(weights[ki]);
            }
        }
        let active: Vec<usize> = (0..mc.n_expert)
            .filter(|&e| !rows_of[e].is_empty())
            .collect();

        // Gather each active expert's token rows into a contiguous [m_e*ne] batch.
        let xs: Vec<Vec<f32>> = active
            .iter()
            .map(|&e| {
                let mut x = vec![0f32; rows_of[e].len() * ne];
                for (j, &r) in rows_of[e].iter().enumerate() {
                    x[j * ne..(j + 1) * ne].copy_from_slice(&hn2[r * ne..(r + 1) * ne]);
                }
                x
            })
            .collect();

        // Phase 1: every active expert's gate+up GEMM in ONE submit (both read its batch `xs[ai]`).
        let mut gu_ops: Vec<(&Wt, &[f32], usize, usize, usize)> =
            Vec::with_capacity(active.len() * 2);
        for (ai, &e) in active.iter().enumerate() {
            let m = rows_of[e].len();
            gu_ops.push((experts[e].gate.gpu(), xs[ai].as_slice(), m, ne, nff));
            gu_ops.push((experts[e].up.gpu(), xs[ai].as_slice(), m, ne, nff));
        }
        let gu = self.gemv_wt_many(&gu_ops)?;

        // SwiGLU on host, then Phase 2: every active expert's down GEMM in ONE submit.
        let acts: Vec<Vec<f32>> = (0..active.len())
            .map(|ai| {
                let (g, u) = (&gu[2 * ai], &gu[2 * ai + 1]);
                (0..g.len()).map(|i| silu(g[i]) * u[i]).collect()
            })
            .collect();
        let down_ops: Vec<(&Wt, &[f32], usize, usize, usize)> = active
            .iter()
            .enumerate()
            .map(|(ai, &e)| {
                (
                    experts[e].down.gpu(),
                    acts[ai].as_slice(),
                    rows_of[e].len(),
                    nff,
                    ne,
                )
            })
            .collect();
        let ys = self.gemv_wt_many(&down_ops)?;

        // Scatter-add each expert's weighted down output back to its token rows.
        let mut out = vec![0f32; t * ne];
        for (ai, &e) in active.iter().enumerate() {
            let y = &ys[ai];
            for (j, &r) in rows_of[e].iter().enumerate() {
                let w = wts_of[e][j];
                for i in 0..ne {
                    out[r * ne + i] += w * y[j * ne + i];
                }
            }
        }
        Ok(out)
    }

    /// Stream a host-offloaded layer's active experts through the VRAM `ExpertPool` and GPU-compute
    /// them (`INFR_MOE_STREAM`): for each selected expert, make its gate/up/down resident in a pool
    /// slot (upload-on-miss, LRU-evict) and run the native GEMV against the slot. Returns each
    /// expert's down output. Faster than the CPU path (GPU matmul), VRAM bounded to the pool.
    fn stream_experts(
        &self,
        x: &[f32],
        idx: &[usize],
        experts: &[ExpertWt],
        mc: &MoeConfig,
        li: usize,
        pool: &mut Option<infr_vulkan::ExpertPool>,
    ) -> Result<Vec<Vec<f32>>> {
        use infr_vulkan::linear::pad_to_u32_align;
        let ne = self.cfg.n_embd;
        // (dtype, native-padded mmap bytes) for an expert role — bytes read zero-copy then padded.
        let parts = |ew: &ExpertW, role: &str, ex: usize| -> Result<(infr_core::DType, Vec<u8>)> {
            let (dt, b) = self.host_expert(ew, li, role, ex)?;
            Ok((dt, pad_to_u32_align(b)))
        };
        // Lazily size the pool: one slot per expert-role's native-padded bytes, enough for a layer's
        // active set (n_used × 3 roles) plus headroom — bounded VRAM regardless of expert count.
        if pool.is_none() {
            let stride = parts(&experts[idx[0]].gate, "gate", idx[0])?
                .1
                .len()
                .max(parts(&experts[idx[0]].down, "down", idx[0])?.1.len());
            let n_slots = (mc.n_used * 3 + mc.n_used).max(8);
            *pool = Some(
                infr_vulkan::ExpertPool::new(&self.be, stride, n_slots)
                    .map_err(|e| anyhow!("{e}"))?,
            );
        }
        let pool = pool.as_mut().unwrap();
        let mut ys = Vec::with_capacity(idx.len());
        for &ex in idx {
            let key = |role: usize| li * mc.n_expert * 3 + ex * 3 + role;
            let (gdt, gb) = parts(&experts[ex].gate, "gate", ex)?;
            let gbuf = pool
                .resident(&self.be, key(0), &gb)
                .map_err(|e| anyhow!("{e}"))?;
            let gate = self.gemv_native_one(gbuf, gdt, x, 1, ne, mc.n_ff_exp)?;
            let (udt, ub) = parts(&experts[ex].up, "up", ex)?;
            let ubuf = pool
                .resident(&self.be, key(1), &ub)
                .map_err(|e| anyhow!("{e}"))?;
            let up = self.gemv_native_one(ubuf, udt, x, 1, ne, mc.n_ff_exp)?;
            let act: Vec<f32> = (0..mc.n_ff_exp).map(|i| silu(gate[i]) * up[i]).collect();
            let (ddt, db) = parts(&experts[ex].down, "down", ex)?;
            let dbuf = pool
                .resident(&self.be, key(2), &db)
                .map_err(|e| anyhow!("{e}"))?;
            ys.push(self.gemv_native_one(dbuf, ddt, &act, 1, mc.n_ff_exp, ne)?);
        }
        Ok(ys)
    }

    /// MoE generation (qwen3moe) with a host KV cache — prefill the prompt once, then decode one
    /// token per step (no O(n²) recompute). `prompt` is chat-formatted; `on_token` fires per token.
    pub fn generate_moe(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let tokens: Vec<u32> = enc.get_ids().to_vec();
        let sampler = self.sampler.get();
        let mut rng = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9e3779b97f4a7c15)
            | 1;
        // Sample on the GPU when possible (only the 4-byte token id reads back); the forward falls
        // back to returning logits for configs the GPU sampler can't handle, which we sample here.
        let sp = |rng: &mut u64| {
            Some(SampleParams {
                temp: sampler.temp,
                top_k: sampler.top_k,
                top_p: sampler.top_p,
                u: draw_u(rng),
            })
        };
        let resolve = |out: GenOut, rng: &mut u64| match out {
            GenOut::Token(t) => t,
            GenOut::Logits(l) => sample_logits(&l, sampler, rng),
        };
        let mut kv = self.new_moe_kv(tokens.len() + max_new + 8)?;
        let s = sp(&mut rng);
        let mut out = self.forward_moe_chunk_g(&tokens, &mut kv, s)?; // prefill
        let mut stream = StreamDecoder::default();
        let mut generated: Vec<u32> = Vec::new();
        for _ in 0..max_new {
            let next = resolve(out, &mut rng);
            if self.cfg.eos_ids.contains(&next) {
                break;
            }
            generated.push(next);
            let full = self.tokenizer.decode(&generated, true).unwrap_or_default();
            on_token(&stream.step(&full));
            let s = sp(&mut rng);
            out = self.forward_moe_chunk_g(&[next], &mut kv, s)?; // 1-token decode
        }
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

/// Dispatch a [`Wt`] linear (`y = x·Wᵀ`) into a recorder, picking the f16 / quant / native op.
/// The (dtype, buffer) of a Native weight — for the stacked MoE expert dispatch (native-only).
fn native_parts(w: &Wt) -> (infr_core::DType, &dyn Buffer) {
    match w {
        Wt::Native { buf, dtype } => (*dtype, buf.as_ref()),
        _ => unreachable!("stacked MoE experts are native-only"),
    }
}

/// Prefill projection (`y = X·Wᵀ`, X = [m,in_f], m≥64): tiled coopmat GEMM for native-quant weights
/// (decode-once, reused across the 64-row tile) instead of the per-row GEMV that re-reads the weight
/// m times. `y` is allocated `ceil(m/64)*64` rows. Non-native weights (the small f16 router) fall
/// back to the GEMV.
#[allow(clippy::too_many_arguments)]
fn rec_proj(
    rec: &infr_vulkan::Recorder,
    w: &Wt,
    x: &dyn Buffer,
    y: &dyn Buffer,
    m: usize,
    in_f: usize,
    out_f: usize,
) {
    match w {
        Wt::Native { buf, dtype } => rec.matmul_native(*dtype, x, buf.as_ref(), y, m, in_f, out_f),
        _ => rec_linear(rec, w, x, y, m, in_f, out_f),
    }
}

/// Dispatch a stacked MoE expert as a tiled coopmat GEMM (`y = X·W_eᵀ`, X = [m,in_f]): the weight is
/// `expert*stride` elements into the stacked Native buffer, decoded ONCE and reused across the 64-row
/// tile (vs the per-row GEMV re-read). `y` is allocated `ceil(m/64)*64` rows. Native-only.
#[allow(clippy::too_many_arguments)]
fn rec_gemm_expert(
    rec: &infr_vulkan::Recorder,
    w: &Wt,
    expert: usize,
    stride: usize,
    x: &dyn Buffer,
    y: &dyn Buffer,
    m: usize,
    in_f: usize,
    out_f: usize,
) {
    let (dtype, buf) = native_parts(w);
    rec.matmul_native_off(dtype, x, buf, expert * stride, y, m, in_f, out_f);
}

/// Dispatch a stacked MoE expert's linear (`y = x·W_eᵀ`): the weight is `expert * stride` elements
/// into the role's stacked Native buffer. Stacked experts are native-only (see [`load_moe`]).
#[allow(clippy::too_many_arguments)]
fn rec_linear_expert(
    rec: &infr_vulkan::Recorder,
    w: &Wt,
    expert: usize,
    stride: usize,
    x: &dyn Buffer,
    y: &dyn Buffer,
    rows: usize,
    in_f: usize,
    out_f: usize,
) {
    match w {
        Wt::Native { buf, dtype } => rec.linear_native_off(
            *dtype,
            buf.as_ref(),
            expert * stride,
            x,
            y,
            rows,
            in_f,
            out_f,
        ),
        _ => unreachable!("stacked MoE experts are native-only"),
    }
}

fn rec_linear(
    rec: &infr_vulkan::Recorder,
    w: &Wt,
    x: &dyn Buffer,
    y: &dyn Buffer,
    rows: usize,
    in_f: usize,
    out_f: usize,
) {
    match w {
        Wt::F16(b) => rec.linear(b.as_ref(), x, y, rows, in_f, out_f),
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
            in_f,
            out_f,
            *bits,
            *blk_shift,
        ),
        Wt::Native { buf, dtype } => {
            rec.linear_native(*dtype, buf.as_ref(), x, y, rows, in_f, out_f)
        }
    }
}

/// MoE router top-k on host: softmax the `n_expert` logits, take the `n_used` highest, renormalize
/// their probs and apply the routing `scale`. Returns (expert indices, per-expert weights).
fn moe_topk(rl: &[f32], mc: &MoeConfig) -> (Vec<usize>, Vec<f32>) {
    let maxl = rl.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = rl.iter().map(|&v| (v - maxl).exp()).collect();
    let mut idx: Vec<usize> = (0..mc.n_expert).collect();
    idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    idx.truncate(mc.n_used);
    let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
    let weights: Vec<f32> = idx.iter().map(|&e| probs[e] / wsum * mc.scale).collect();
    (idx, weights)
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

/// Host matvec `y = x·Wᵀ` for a host-backed expert weight: dequant the quantized `[out_f, in_f]`
/// `bytes` (read zero-copy from the GGUF mmap) to f32, then dot each row with `x`. Correctness-first
/// — the CPU path is the VRAM/speed tradeoff; not micro-optimized (full dequant per call).
fn cpu_expert_matvec(
    dtype: infr_core::DType,
    bytes: &[u8],
    x: &[f32],
    in_f: usize,
    out_f: usize,
) -> Result<Vec<f32>> {
    let w = dequant_block(dtype, bytes)?; // [out_f * in_f] row-major (out rows)
    let mut y = vec![0f32; out_f];
    for o in 0..out_f {
        let row = &w[o * in_f..(o + 1) * in_f];
        y[o] = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
    Ok(y)
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
        let by = be.alloc(4, BufferUsage::Readback).unwrap(); // 1 output row, 1 out feature

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
        let mut cpu_outputs = [0f32; OUT_F];
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
        let mut cpu_outputs = [0f32; OUT_F];
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

    // ── Native-block prefill GEMM parity (matmul_native vs trusted linear_native) ──
    //
    // The tiled coopmat GEMM reuses the same per-format dqblk decode as the GEMV, so the decode is
    // already covered by the *_native_matches_cpu tests. This guards the NEW code — the 64x64 tile,
    // shared staging, and coopmat accumulation — by checking that C[m,:] from matmul_native equals
    // the GEMV linear_native(weight, A[m]) for every row m, across M spanning multiple row-tiles.
    // Weight blocks vary their f16 d per block so columns are distinguishable (catches col mixups).

    // Build one valid native block of `dtype` with f16 scale `d` and a varied payload from `seed`.
    fn native_block(dtype: infr_core::DType, d: f32, seed: u8) -> Vec<u8> {
        use infr_core::DType::*;
        let dbits = half::f16::from_f32(d).to_bits().to_le_bytes();
        match dtype {
            Q8_0 => {
                let mut b = vec![0u8; 34];
                b[0..2].copy_from_slice(&dbits);
                fill(&mut b[2..34], 17, seed);
                b
            }
            Q4K => {
                let mut b = vec![0u8; 144];
                b[0..2].copy_from_slice(&dbits); // d
                b[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes()); // dmin
                fill(&mut b[4..16], 13, seed); // 6-bit scales
                fill(&mut b[16..144], 7, seed); // qs
                b
            }
            Q6K => {
                let mut b = vec![0u8; 210];
                fill(&mut b[0..128], 7, seed); // ql
                fill(&mut b[128..192], 11, seed); // qh
                fill(&mut b[192..208], 3, seed); // i8 scales
                b[208..210].copy_from_slice(&dbits); // d
                b
            }
            other => panic!("native_block: add {other:?}"),
        }
    }

    fn check_native_gemm(dtype: infr_core::DType, m: usize) {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        use infr_vulkan::linear::pad_to_u32_align;
        let n = 64usize;
        let k = 256usize;
        let belems = if dtype == infr_core::DType::Q8_0 {
            32
        } else {
            256
        };
        let blocks_per_row = k / belems;

        // Weight [N, K] as native blocks (row-major). d varies per block → distinguishable columns.
        let mut wbytes: Vec<u8> = Vec::new();
        for o in 0..n {
            for bk in 0..blocks_per_row {
                let d = 0.005 * ((o % 7) as f32 + 1.0) + 0.001 * bk as f32;
                wbytes.extend_from_slice(&native_block(dtype, d, (o * 3 + bk * 5) as u8));
            }
        }
        let wbuf = be.upload_weight_bytes(&pad_to_u32_align(&wbytes)).unwrap();

        // Activations [M, K], varied per (row, col).
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05 + ((i / k) as f32) * 0.001)
            .collect();
        let abuf = be.alloc(a.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(abuf.as_ref(), bytemuck::cast_slice(&a)).unwrap();

        // GPU GEMM → C [ceil(m/64)*64, N]. Device-local (coopmat store needs it), download via copy.
        let crows = m.div_ceil(64) * 64;
        let cbuf = be.alloc(crows * n * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.matmul_native(dtype, abuf.as_ref(), wbuf.as_ref(), cbuf.as_ref(), m, k, n);
        rec.finish().unwrap();
        let mut cbytes = vec![0u8; crows * n * 4];
        be.download(cbuf.as_ref(), &mut cbytes).unwrap();
        let cgemm: &[f32] = bytemuck::cast_slice(&cbytes);

        // Reference: one GEMV per row → C[m,:]
        for row in 0..m {
            let xbuf = be.alloc(k * 4, BufferUsage::Staging).unwrap();
            be.upload(
                xbuf.as_ref(),
                bytemuck::cast_slice(&a[row * k..row * k + k]),
            )
            .unwrap();
            let ybuf = be.alloc(n * 4, BufferUsage::Readback).unwrap();
            let rec2 = be.recorder().unwrap();
            rec2.linear_native(dtype, wbuf.as_ref(), xbuf.as_ref(), ybuf.as_ref(), 1, k, n);
            rec2.finish().unwrap();
            let mut ybytes = vec![0u8; n * 4];
            be.download(ybuf.as_ref(), &mut ybytes).unwrap();
            let yref: &[f32] = bytemuck::cast_slice(&ybytes);
            // The GEMM rounds activations+weights to f16 for coopmat (GEMV keeps f32 activations), so
            // compare error against the row's largest magnitude (standard GEMM metric) — near-zero
            // outputs from cancellation otherwise blow up a pure relative error.
            let rmax = yref.iter().fold(0f32, |a, &v| a.max(v.abs()));
            for col in 0..n {
                let g = cgemm[row * n + col];
                let r = yref[col];
                let err = (g - r).abs();
                assert!(
                    err < 0.02 * rmax + 1e-4,
                    "{dtype:?} GEMM vs GEMV at [{row},{col}]: gemm={g} gemv={r} err={err} rmax={rmax}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q8_0_native_gemm_matches_gemv() {
        check_native_gemm(infr_core::DType::Q8_0, 70);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q4k_native_gemm_matches_gemv() {
        check_native_gemm(infr_core::DType::Q4K, 70);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn q6k_native_gemm_matches_gemv() {
        check_native_gemm(infr_core::DType::Q6K, 70);
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
