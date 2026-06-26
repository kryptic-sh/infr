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
use infr_core::{Backend, WeightSource};
use infr_gguf::Gguf;
use infr_vulkan::VulkanBackend;
use std::path::Path;
use tokenizers::Tokenizer;

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
    /// Qwen3-style per-head RMSNorm on Q and K before RoPE.
    pub qk_norm: bool,
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
}

/// Per-layer key/value cache held on the GPU (persists across decode steps).
pub struct KvCache {
    k: Vec<Box<dyn Buffer>>, // per layer: [max_ctx, n_kv*head_dim]
    v: Vec<Box<dyn Buffer>>,
    len: usize,
    max_ctx: usize,
}

fn meta_u64(g: &Gguf, key: &str) -> Option<u64> {
    g.metadata().u64(key)
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

    pub fn load(gguf_path: &Path, tokenizer_path: &Path) -> Result<Self> {
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
        }

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!("load tokenizer: {e}"))?;

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
        })
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
        let mpad = if use_gemm { n.div_ceil(64) * 64 } else { n };
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

        // Flash-decoding: for single-token decode at long context, split each head's KV range
        // across many workgroups (partials in pm/pl/pacc), so attention isn't stuck on `nh`
        // workgroups. Reused across layers.
        const CHUNK: usize = 256;
        let kv_len = pos + n;
        let use_split = n == 1 && kv_len > CHUNK;
        let n_chunks = if use_split { kv_len.div_ceil(CHUNK) } else { 0 };
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
                } => rec.matmul_proj(
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
                ),
            }
        };
        for (li, layer) in self.layers.iter().enumerate() {
            if let Some((qr, kr, vr)) = &qkv_raw {
                // qwen3: rmsnorm → Q/K/V projections → per-head QK-norm+RoPE (K/V into the cache)
                rec.rmsnorm(
                    hidden.as_ref(),
                    layer.attn_norm_buf.as_ref(),
                    hn.as_ref(),
                    n,
                    ne,
                    c.rms_eps,
                );
                if use_gemm {
                    mm(&layer.wq, hn.as_ref(), qr.as_ref(), n, ne, nh * hd);
                    mm(&layer.wk, hn.as_ref(), kr.as_ref(), n, ne, kvrow);
                    mm(&layer.wv, hn.as_ref(), vr.as_ref(), n, ne, kvrow);
                } else {
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
            if use_gemm {
                // prefill: coopmat flash attention (reads each K/V block once per 64-query tile).
                rec.attention_prefill(
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
                    CHUNK,
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
        let budget = if self.cfg.qk_norm {
            16_000_000
        } else {
            256 * 64
        };
        let raw = (budget / (pos + 1)).clamp(256, 2048);
        (raw / 64 * 64).max(64)
    }

    /// Greedy generate up to `max_new` tokens after `prompt` (already a chat-formatted string).
    pub fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        let mut generated: Vec<u32> = Vec::new();

        // Size the KV cache to exactly what this run needs — bounded only by VRAM, not a fixed cap.
        let mut kv = self.new_kv(prompt_tokens.len() + max_new + 8)?;
        // Prefill: one submit over a very long prompt can exceed the GPU watchdog (TDR /
        // device-lost), since each attention/matmul dispatch grows with tokens × context. Short
        // prompts go in one fast pass; long ones are split into small chunks that stay well under
        // the watchdog. (A real GEMM prefill path would let chunks be large; GEMV is the current
        // limit — see coopmat-prefill TODO.) Only the last chunk's logits matter.
        let len = prompt_tokens.len();
        let mut logits = Vec::new();
        let mut i = 0;
        while i < len {
            let end = (i + self.prefill_chunk(i)).min(len);
            logits = self.forward_resident_kv(&prompt_tokens[i..end], &mut kv)?;
            i = end;
        }
        for _ in 0..max_new {
            let next = argmax(&logits) as u32;
            if next == self.cfg.eos {
                break;
            }
            generated.push(next);
            if let Ok(piece) = self.tokenizer.decode(&[next], false) {
                on_token(&piece);
            }
            // decode step: single new token, attends over the cache
            logits = self.forward_resident_kv(&[next], &mut kv)?;
        }
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Wrap a user message in the SmolLM2 ChatML template.
    pub fn chatml(&self, user: &str) -> String {
        format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    }
}

// ---- host ops ----

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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
