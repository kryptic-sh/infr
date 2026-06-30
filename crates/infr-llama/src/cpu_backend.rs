//! CPU reference backend — a correctness-first interpreter of the backend-agnostic
//! [`infr_core`] compute [`Graph`]. No SIMD, no threading yet: every op is a plain scalar loop.
//! Weights are read **zero-copy from the GGUF mmap** (no `memcpy`, no owned RAM): the bulk
//! projection weights (`Op::Linear`) are dequantized one row at a time straight from the mapping
//! inside the dot, so 12B / MoE models cost only their on-disk size in page cache. Only the tiny
//! norm weights are dequant-cached; the model writes (KV / conv / recurrent state, per-step IO) use
//! small owned buffers. It exists to (a) run every model without a GPU and (b) serve as the oracle
//! the GPU backends are validated against.
//!
//! Lives in `infr-llama` for now (next to [`crate::dequant_block`] + the qwen35 CPU oracle) to
//! avoid a circular crate dep; it implements the agnostic `infr_core::Backend` trait, so it can be
//! extracted to an `infr-cpu` crate later without touching callers.

use crate::{dequant_block, Config, PerLayerEmbd};
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Capabilities, Plan};
use infr_core::error::Result;
use infr_core::graph::{Activation, AttnMask, Graph, Op, TensorKind};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::WeightSource;
use infr_gguf::{Gguf, TensorBytes};
use rayon::prelude::*;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Timing/counts from a CPU generation, for the caller's stats line.
#[derive(Debug, Clone, Copy)]
pub struct CpuStats {
    pub n_prompt: usize,
    pub prompt_secs: f64,
    pub n_gen: usize,
    pub decode_secs: f64,
}

/// Activation quantized to Q8 over 256-element super-blocks: `qs[i] = round(x[i]/d[blk])` (int8),
/// `d[blk] = max|x|/127`. Quantize the activation ONCE per matvec, then integer-dot it against the
/// quantized weight rows (llama.cpp's q8_K path) — no per-row f32 weight expansion.
struct Q8 {
    qs: Vec<i8>,
    d: Vec<f32>,
}

fn quantize_q8(x: &[f32]) -> Q8 {
    let nb = x.len() / 256;
    let mut qs = vec![0i8; nb * 256];
    let mut d = vec![0f32; nb];
    for b in 0..nb {
        let blk = &x[b * 256..b * 256 + 256];
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let dd = amax / 127.0;
        let id = if dd > 0.0 { 1.0 / dd } else { 0.0 };
        d[b] = dd;
        for (i, &v) in blk.iter().enumerate() {
            qs[b * 256 + i] = (v * id).round().clamp(-127.0, 127.0) as i8;
        }
    }
    Q8 { qs, d }
}

/// `Σ weight·x` for one Q4_K row (144 bytes / 256 elems) against the Q8 activation. Weight value is
/// `d·sc_s·q4 − dmin·m_s` over 8 sub-blocks of 32; the integer sub-block dot `Σ q4·q8` autovectorizes.
fn vec_dot_q4k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let nb = in_f / 256;
    let mut sumf = 0f32;
    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        let d = crate::rdf16(&blk[0..2]);
        let dmin = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        for s in 0..8 {
            let (sc, m) = crate::k4(s, scales);
            let (half, hi) = (s / 2, s % 2 == 1);
            let qbyte = &qs[half * 32..half * 32 + 32];
            let q8s = &q8b[s * 32..s * 32 + 32];
            let (mut iprod, mut isum) = (0i32, 0i32);
            for l in 0..32 {
                let q4 = if hi {
                    (qbyte[l] >> 4) as i32
                } else {
                    (qbyte[l] & 0xF) as i32
                };
                let v = q8s[l] as i32;
                iprod += q4 * v;
                isum += v;
            }
            sd += sc as i32 * iprod;
            sm += m as i32 * isum;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// `Σ weight·x` for one Q6_K row (210 bytes / 256 elems). Weight value is `d·sc·(q6−32)` over 16
/// sub-blocks of 16 (int8 scales); accumulate `Σ q6·q8` and `Σ q8` per sub-block.
fn vec_dot_q6k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let nb = in_f / 256;
    let mut sumf = 0f32;
    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        let d = crate::rdf16(&blk[208..210]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut sumi = [0i32; 16];
        let mut bsum = [0i32; 16];
        for half in 0..2 {
            let (qlo, qho, sco, base) = (half * 64, half * 32, half * 8, half * 128);
            for l in 0..32 {
                let is = l / 16;
                let q1 = (ql[qlo + l] & 0xF) | ((qh[qho + l] & 3) << 4);
                let q2 = (ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4);
                let q3 = (ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4);
                let q4 = (ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4);
                for (off, q, sci) in [(0, q1, 0), (32, q2, 2), (64, q3, 4), (96, q4, 6)] {
                    let sub = sco + is + sci;
                    let v = q8b[base + l + off] as i32;
                    sumi[sub] += q as i32 * v;
                    bsum[sub] += v;
                }
            }
        }
        let mut s = 0f32;
        for sub in 0..16 {
            s += scales[sub] as i8 as f32 * (sumi[sub] - 32 * bsum[sub]) as f32;
        }
        sumf += d * q8.d[b] * s;
    }
    sumf
}

/// `Σ f16_weight·x` (weight is 2 bytes/elem). `target-cpu=native` lowers the f16→f32 to F16C.
fn dot_f16(w: &[u8], x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n = x.len();
    let chunks = n / 8;
    for c in 0..chunks {
        for (j, ac) in acc.iter_mut().enumerate() {
            let i = c * 8 + j;
            let wv = half::f16::from_le_bytes([w[i * 2], w[i * 2 + 1]]).to_f32();
            *ac += wv * x[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..n {
        s += half::f16::from_le_bytes([w[i * 2], w[i * 2 + 1]]).to_f32() * x[i];
    }
    s
}

/// `Σ bf16_weight·x` (bf16 = top 16 bits of f32).
fn dot_bf16(w: &[u8], x: &[f32]) -> f32 {
    let mut s = 0f32;
    for (i, &xi) in x.iter().enumerate() {
        let wv = f32::from_bits((u16::from_le_bytes([w[i * 2], w[i * 2 + 1]]) as u32) << 16);
        s += wv * xi;
    }
    s
}

/// Dot product with 8 independent accumulators so the reduction isn't latency-bound — lets the
/// autovectorizer (with `target-cpu=native`) keep several AVX FMA lanes in flight. `a`/`b` equal len.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc = [0f32; 8];
    for c in 0..chunks {
        let base = c * 8;
        for (j, ac) in acc.iter_mut().enumerate() {
            *ac += a[base + j] * b[base + j];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// A host buffer. Weights are **mapped** — a zero-copy [`TensorBytes`] view straight into the GGUF
/// mmap (read-only, no `memcpy`, no owned RAM). Everything the model writes (KV / conv / recurrent
/// state, per-step IO) is **owned** — a plain byte vec behind a `Mutex` (so `&dyn Buffer` stays
/// `Send + Sync` and writes are safe). `&dyn Buffer` reads go through [`CpuBuffer::read`].
pub enum CpuBuffer {
    Owned(Mutex<Vec<u8>>),
    Mapped(TensorBytes),
}

/// A uniform read view over either storage (a `MutexGuard` for owned, the slice for mapped); both
/// deref to `[u8]`.
enum CpuRead<'a> {
    Owned(std::sync::MutexGuard<'a, Vec<u8>>),
    Mapped(&'a TensorBytes),
}

impl std::ops::Deref for CpuRead<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            CpuRead::Owned(g) => g,
            CpuRead::Mapped(t) => t,
        }
    }
}

impl CpuBuffer {
    /// Read view of the bytes (zero-copy for mapped weights; mutex guard for owned buffers).
    fn read(&self) -> CpuRead<'_> {
        match self {
            CpuBuffer::Owned(m) => CpuRead::Owned(m.lock().unwrap()),
            CpuBuffer::Mapped(t) => CpuRead::Mapped(t),
        }
    }
    /// Mutable owned storage; panics for mapped (read-only) weights.
    fn owned(&self) -> std::sync::MutexGuard<'_, Vec<u8>> {
        match self {
            CpuBuffer::Owned(m) => m.lock().unwrap(),
            CpuBuffer::Mapped(_) => {
                panic!("cpu backend: write to a mapped (read-only) weight buffer")
            }
        }
    }
}

impl Buffer for CpuBuffer {
    fn len_bytes(&self) -> usize {
        match self {
            CpuBuffer::Owned(m) => m.lock().unwrap().len(),
            CpuBuffer::Mapped(t) => t.len(),
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A compiled plan = the owned graph (the CPU "compiles" nothing; it interprets at execute time).
pub struct CpuPlan {
    graph: Graph,
}

impl Plan for CpuPlan {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Default)]
pub struct CpuBackend {
    /// Dequantized-weight cache keyed by the bound buffer's address (weights are bound the same
    /// every step, so dequant once and reuse). Only the small norm weights (`RmsNorm` / `QkNorm`)
    /// land here — the large `Op::Linear` weights are streamed row-by-row instead (see that arm),
    /// so this never holds the whole model in f32.
    weight_cache: Mutex<HashMap<usize, Arc<Vec<f32>>>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Wrap a zero-copy GGUF mmap view as a read-only weight buffer (no allocation, no `memcpy`).
    pub fn map_weight(&self, bytes: TensorBytes) -> Box<dyn Buffer> {
        Box::new(CpuBuffer::Mapped(bytes))
    }
}

/// Reinterpret raw buffer bytes as `f32` values per `dtype` (dequantizing quant/f16/bf16, widening
/// integer position tensors). The universal "read a tensor's value on the host".
fn bytes_to_f32(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        DType::I32 => bytemuck::cast_slice::<u8, i32>(bytes)
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DType::U32 => bytemuck::cast_slice::<u8, u32>(bytes)
            .iter()
            .map(|&v| v as f32)
            .collect(),
        // F16 / Bf16 / all quant + codebook types go through the shared host dequant.
        other => dequant_block(other, bytes).expect("cpu backend: host dequant"),
    }
}

/// Gated-FFN activation applied to the gate value.
fn act_fn(act: Activation, g: f32) -> f32 {
    match act {
        Activation::Silu => g / (1.0 + (-g).exp()),
        // gelu_pytorch_tanh: 0.5 g (1 + tanh(√(2/π)·(g + 0.044715 g³)))
        Activation::Gelu => 0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()),
        Activation::Sigmoid => 1.0 / (1.0 + (-g).exp()),
    }
}

fn cpu_buf(b: &dyn Buffer) -> &CpuBuffer {
    b.as_any()
        .downcast_ref::<CpuBuffer>()
        .expect("cpu backend: buffer is not a CpuBuffer (mixed backends?)")
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            name: "cpu-reference".into(),
            f16: true,
            cooperative_matrix: false,
            max_buffer_bytes: u64::MAX,
            unified_memory: true,
        }
    }

    fn alloc(&self, bytes: usize, _usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        Ok(Box::new(CpuBuffer::Owned(Mutex::new(vec![
            0u8;
            bytes.max(4)
        ]))))
    }

    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        let mut d = cpu_buf(dst).owned();
        d[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        let s = cpu_buf(src).read();
        dst.copy_from_slice(&s[..dst.len()]);
        Ok(())
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        Ok(Box::new(CpuPlan {
            graph: graph.clone(),
        }))
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        let g = &plan
            .as_any()
            .downcast_ref::<CpuPlan>()
            .expect("cpu backend: plan is not a CpuPlan")
            .graph;

        // f32 working store for every Input/Internal/Output handle (weights are read on demand:
        // norms via the small dequant cache, `Op::Linear` weights streamed row-by-row).
        let mut vals: Vec<Vec<f32>> = vec![Vec::new(); g.tensors.len()];
        // KV-cache tensors (the `cache` of `WriteKv`, the `k_cache`/`v_cache` of `Attention`) are
        // accessed straight from their bound buffers — `WriteKv` writes one row, `Attention` reads
        // `kv_len` rows. They're sized for the WHOLE context (`max_ctx`), so loading them into `vals`
        // (and writing them back) each token would cost O(max_ctx) memory traffic per token instead of
        // O(kv_len) — catastrophic at a large `max_new`. Skip the round-trip for them.
        let mut direct: HashSet<u32> = HashSet::new();
        for op in &g.ops {
            match op {
                Op::WriteKv { cache, .. } => {
                    direct.insert(cache.0);
                }
                Op::Attention {
                    k_cache, v_cache, ..
                } => {
                    direct.insert(k_cache.0);
                    direct.insert(v_cache.0);
                }
                _ => {}
            }
        }
        for (i, decl) in g.tensors.iter().enumerate() {
            match decl.kind {
                TensorKind::Internal | TensorKind::Output => {
                    vals[i] = vec![0f32; decl.desc.numel()]
                }
                TensorKind::Input if direct.contains(&(i as u32)) => {} // read/written in place
                TensorKind::Input => {
                    let buf = bindings
                        .get(TensorId(i as u32))
                        .expect("cpu backend: unbound Input");
                    let bytes = cpu_buf(buf).read();
                    vals[i] = bytes_to_f32(&bytes, decl.desc.dtype);
                }
                TensorKind::Weight => {} // lazily dequantized in `weight()`
            }
        }

        // Fetch a (cached) dequantized weight.
        let weight = |id: TensorId| -> Arc<Vec<f32>> {
            let buf = bindings.get(id).expect("cpu backend: unbound Weight");
            let key = cpu_buf(buf) as *const CpuBuffer as usize;
            if let Some(w) = self.weight_cache.lock().unwrap().get(&key) {
                return w.clone();
            }
            let bytes = cpu_buf(buf).read();
            let w = Arc::new(bytes_to_f32(&bytes, g.desc(id).dtype));
            self.weight_cache.lock().unwrap().insert(key, w.clone());
            w
        };

        for op in &g.ops {
            match *op {
                Op::RmsNorm {
                    x,
                    weight: w,
                    dst,
                    rows,
                    dim,
                    eps,
                } => {
                    let (rows, dim) = (rows as usize, dim as usize);
                    let xs = &vals[x.0 as usize];
                    let ws = weight(w);
                    let mut out = vec![0f32; rows * dim];
                    for r in 0..rows {
                        let b = r * dim;
                        let ss: f32 =
                            (0..dim).map(|i| xs[b + i] * xs[b + i]).sum::<f32>() / dim as f32;
                        let s = 1.0 / (ss + eps).sqrt();
                        for i in 0..dim {
                            out[b + i] = xs[b + i] * s * ws[i];
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::QkNorm {
                    x,
                    weight: w,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    eps,
                } => {
                    let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                    let xs = &vals[x.0 as usize];
                    let ws = weight(w);
                    let mut out = vec![0f32; rows * nh * hd];
                    for r in 0..rows {
                        for h in 0..nh {
                            let b = (r * nh + h) * hd;
                            let ss: f32 =
                                (0..hd).map(|i| xs[b + i] * xs[b + i]).sum::<f32>() / hd as f32;
                            let s = 1.0 / (ss + eps).sqrt();
                            for i in 0..hd {
                                out[b + i] = xs[b + i] * s * ws[i];
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Linear {
                    x,
                    weight: w,
                    dst,
                    m,
                    in_f,
                    out_f,
                } => {
                    let (m, in_f, out_f) = (m as usize, in_f as usize, out_f as usize);
                    let xs = &vals[x.0 as usize];
                    // Stream the (row-major [out_f, in_f]) weight one row at a time straight from the
                    // mmap, dequantizing inside the dot — no full f32 materialization. GGUF rows are
                    // block-aligned, so each row is an equal `bytes/out_f` slice. Output rows are
                    // independent → fan out over the 32 cores with rayon.
                    let buf = bindings.get(w).expect("cpu backend: unbound Weight");
                    let bytes = cpu_buf(buf).read();
                    let wbytes: &[u8] = &bytes;
                    let dt = g.desc(w).dtype;
                    let bpr = wbytes.len() / out_f; // bytes per weight row
                    let mut out = vec![0f32; m * out_f];
                    // One token (decode) is the hot path. Dispatch on the weight dtype to the fastest
                    // per-row kernel: integer Q8×Q4_K/Q6_K dots (quantize the activation once), direct
                    // f16/bf16/f32 dots, else fall back to dequant-to-f32 + dot. All fan out over rows.
                    if m == 1 {
                        let xrow = &xs[..in_f];
                        let q8 = matches!(dt, DType::Q4K | DType::Q6K).then(|| quantize_q8(xrow));
                        out.par_iter_mut().enumerate().for_each(|(o, dst_o)| {
                            let row = &wbytes[o * bpr..o * bpr + bpr];
                            *dst_o = match dt {
                                DType::Q4K => vec_dot_q4k(row, q8.as_ref().unwrap(), in_f),
                                DType::Q6K => vec_dot_q6k(row, q8.as_ref().unwrap(), in_f),
                                DType::F32 => dot(bytemuck::cast_slice(row), xrow),
                                DType::F16 => dot_f16(row, xrow),
                                DType::Bf16 => dot_bf16(row, xrow),
                                _ => dot(&bytes_to_f32(row, dt), xrow),
                            };
                        });
                    } else {
                        for o in 0..out_f {
                            let row = bytes_to_f32(&wbytes[o * bpr..o * bpr + bpr], dt);
                            for r in 0..m {
                                out[r * out_f + o] = dot(&row, &xs[r * in_f..r * in_f + in_f]);
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Rope {
                    x,
                    positions,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    rope_dim,
                    theta,
                    freq_factors,
                } => {
                    let (rows, nh, hd, rd) = (
                        rows as usize,
                        n_head as usize,
                        head_dim as usize,
                        rope_dim as usize,
                    );
                    let xs = vals[x.0 as usize].clone();
                    let pos = vals[positions.0 as usize].clone();
                    let ff = freq_factors.map(|f| vals[f.0 as usize].clone());
                    let mut out = xs.clone(); // dims beyond rope_dim pass through unchanged
                    let hf = rd / 2;
                    for r in 0..rows {
                        let p0 = pos[r];
                        for h in 0..nh {
                            let b = (r * nh + h) * hd;
                            for p in 0..hf {
                                let (i0, i1) = (p, p + hf);
                                let mut ang = p0 * theta.powf(-2.0 * p as f32 / rd as f32);
                                if let Some(ff) = &ff {
                                    ang /= ff[p];
                                }
                                let (s, c) = (ang.sin(), ang.cos());
                                let a = xs[b + i0];
                                let bb = xs[b + i1];
                                out[b + i0] = a * c - bb * s;
                                out[b + i1] = a * s + bb * c;
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::QkNormRope {
                    x,
                    weight: w,
                    positions,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    rope_dim,
                    theta,
                    eps,
                    freq_factors,
                } => {
                    // Fused QkNorm + Rope: one pass per head — rmsnorm (× weight), then rotate the
                    // first `rope_dim` in place (dims beyond pass through normed). Output-identical to
                    // the separate QkNorm→Rope pair; maps 1:1 to the GPU `qk_norm_rope` kernel.
                    let (rows, nh, hd, rd) = (
                        rows as usize,
                        n_head as usize,
                        head_dim as usize,
                        rope_dim as usize,
                    );
                    let xs = vals[x.0 as usize].clone();
                    let ws = weight(w);
                    let pos = vals[positions.0 as usize].clone();
                    let ff = freq_factors.map(|f| vals[f.0 as usize].clone());
                    let mut out = vec![0f32; rows * nh * hd];
                    let hf = rd / 2;
                    for r in 0..rows {
                        let p0 = pos[r];
                        for h in 0..nh {
                            let b = (r * nh + h) * hd;
                            let ss: f32 =
                                (0..hd).map(|i| xs[b + i] * xs[b + i]).sum::<f32>() / hd as f32;
                            let s = 1.0 / (ss + eps).sqrt();
                            for i in 0..hd {
                                out[b + i] = xs[b + i] * s * ws[i];
                            }
                            for p in 0..hf {
                                let (i0, i1) = (p, p + hf);
                                let mut ang = p0 * theta.powf(-2.0 * p as f32 / rd as f32);
                                if let Some(ff) = &ff {
                                    ang /= ff[p];
                                }
                                let (sn, c) = (ang.sin(), ang.cos());
                                let a = out[b + i0];
                                let bb = out[b + i1];
                                out[b + i0] = a * c - bb * sn;
                                out[b + i1] = a * sn + bb * c;
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::WriteKv {
                    src,
                    cache,
                    rows,
                    row_stride,
                    pos,
                } => {
                    let (rows, rs, pos) = (rows as usize, row_stride as usize, pos as usize);
                    let s = &vals[src.0 as usize];
                    // Write the new row(s) straight into the persistent KV buffer — only `rows` rows
                    // touched, not the whole `max_ctx`-sized cache. The cache dtype (f16 to match the
                    // GPU and halve memory, or f32) is read from the graph; cast on write.
                    let buf = bindings.get(cache).expect("cpu backend: unbound KV cache");
                    let mut d = cpu_buf(buf).owned();
                    let base = pos * rs;
                    let n = rows * rs;
                    match g.desc(cache).dtype {
                        DType::F16 => {
                            let df: &mut [u16] = bytemuck::cast_slice_mut(&mut d);
                            for i in 0..n {
                                df[base + i] = half::f16::from_f32(s[i]).to_bits();
                            }
                        }
                        _ => {
                            let df: &mut [f32] = bytemuck::cast_slice_mut(&mut d);
                            df[base..base + n].copy_from_slice(&s[..n]);
                        }
                    }
                }
                Op::Attention {
                    q,
                    k_cache,
                    v_cache,
                    dst,
                    rows,
                    kv_len,
                    n_head,
                    n_kv,
                    head_dim,
                    scale,
                    mask,
                    pos,
                } => {
                    let (rows, kv_len, nh, nkv, hd) = (
                        rows as usize,
                        kv_len as usize,
                        n_head as usize,
                        n_kv as usize,
                        head_dim as usize,
                    );
                    let qs = &vals[q.0 as usize];
                    // K/V live in their persistent buffers (f32); borrow them — attention reads only
                    // the first `kv_len` rows, never the whole `max_ctx` cache.
                    let kbuf = bindings.get(k_cache).expect("cpu backend: unbound k_cache");
                    let vbuf = bindings.get(v_cache).expect("cpu backend: unbound v_cache");
                    let kguard = cpu_buf(kbuf).read();
                    let vguard = cpu_buf(vbuf).read();
                    // Materialize the valid KV prefix (`kv_len` rows) as f32, dequantizing an f16
                    // cache (matches the GPU's f16 KV) — the inner dot then runs in f32 either way.
                    let need = kv_len * nkv * hd;
                    let (ks, vs): (Vec<f32>, Vec<f32>) = match g.desc(k_cache).dtype {
                        DType::F16 => {
                            let f = |b: &[u8]| -> Vec<f32> {
                                bytemuck::cast_slice::<u8, u16>(b)[..need]
                                    .iter()
                                    .map(|&x| half::f16::from_bits(x).to_f32())
                                    .collect()
                            };
                            (f(&kguard), f(&vguard))
                        }
                        _ => (
                            bytemuck::cast_slice::<u8, f32>(&kguard)[..need].to_vec(),
                            bytemuck::cast_slice::<u8, f32>(&vguard)[..need].to_vec(),
                        ),
                    };
                    let group = nh / nkv;
                    let window = match mask {
                        AttnMask::Causal => 0usize,
                        AttnMask::SlidingWindow(w) => w,
                    };
                    let mut out = vec![0f32; rows * nh * hd];
                    for ti in 0..rows {
                        let abs = pos as usize + ti; // absolute position of this query
                        for h in 0..nh {
                            let kvh = h / group;
                            let qb = (ti * nh + h) * hd;
                            // visible keys: [lo, abs] (causal); SWA clips lo to abs-window+1.
                            let lo = if window > 0 && abs + 1 > window {
                                abs + 1 - window
                            } else {
                                0
                            };
                            let mut sc = vec![0f32; abs + 1 - lo];
                            let mut mx = f32::NEG_INFINITY;
                            for (jj, scj) in sc.iter_mut().enumerate() {
                                let j = lo + jj;
                                let kb = (j * nkv + kvh) * hd;
                                let d: f32 = (0..hd).map(|x| qs[qb + x] * ks[kb + x]).sum();
                                *scj = d * scale;
                                mx = mx.max(*scj);
                            }
                            let mut l = 0f32;
                            for s in &sc {
                                l += (s - mx).exp();
                            }
                            let ob = (ti * nh + h) * hd;
                            for (jj, s) in sc.iter().enumerate() {
                                let j = lo + jj;
                                let p = (s - mx).exp() / l;
                                let vb = (j * nkv + kvh) * hd;
                                for x in 0..hd {
                                    out[ob + x] += p * vs[vb + x];
                                }
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::GatedAct {
                    gate,
                    up,
                    dst,
                    rows,
                    nff,
                    act,
                    up_off,
                } => {
                    let (rows, nff, up_off) = (rows as usize, nff as usize, up_off as usize);
                    let gs = &vals[gate.0 as usize];
                    let us = &vals[up.0 as usize];
                    // `up` may be a wider layer-major buffer (E2B); the per-row stride stays `nff`
                    // but the read is shifted by `up_off` (0 for the normal [rows, nff] case).
                    let mut out = vec![0f32; rows * nff];
                    for r in 0..rows {
                        let gb = r * nff;
                        let ub = r * nff + up_off;
                        for i in 0..nff {
                            out[gb + i] = act_fn(act, gs[gb + i]) * us[ub + i];
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Add { a, b, dst, n } => {
                    let n = n as usize;
                    let av = vals[a.0 as usize].clone();
                    let bv = &vals[b.0 as usize];
                    let mut out = vec![0f32; n];
                    for i in 0..n {
                        out[i] = av[i] + bv[i];
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Scale { x, dst, s, n } => {
                    let n = n as usize;
                    let xs = vals[x.0 as usize].clone();
                    let mut out = vec![0f32; n];
                    for i in 0..n {
                        out[i] = xs[i] * s;
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Softcap { x, dst, cap, n } => {
                    let n = n as usize;
                    let xs = vals[x.0 as usize].clone();
                    let mut out = vec![0f32; n];
                    for i in 0..n {
                        out[i] = cap * (xs[i] / cap).tanh();
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Copy {
                    src,
                    src_off,
                    dst,
                    dst_off,
                    n,
                } => {
                    let (so, dof, n) = (src_off as usize, dst_off as usize, n as usize);
                    let s = vals[src.0 as usize].clone();
                    vals[dst.0 as usize][dof..dof + n].copy_from_slice(&s[so..so + n]);
                }
                Op::MoeFfn {
                    x,
                    router,
                    gate_exps,
                    up_exps,
                    down_exps,
                    dst,
                    ne,
                    n_expert,
                    n_used,
                    n_ff_exp,
                    scale,
                    act,
                } => {
                    let (ne, n_expert, n_used, nffx) = (
                        ne as usize,
                        n_expert as usize,
                        n_used as usize,
                        n_ff_exp as usize,
                    );
                    let xs = vals[x.0 as usize].clone();
                    // Stream a (row-major [out_f, in_f]) weight slice and matvec it against `v` —
                    // dequant per row, exactly like `Op::Linear`, parallel over rows.
                    let matvec = |bytes: &[u8], dt: DType, v: &[f32], in_f: usize, out_f: usize| {
                        let bpr = bytes.len() / out_f;
                        (0..out_f)
                            .into_par_iter()
                            .map(|r| {
                                let row = bytes_to_f32(&bytes[r * bpr..r * bpr + bpr], dt);
                                dot(&row, &v[..in_f])
                            })
                            .collect::<Vec<f32>>()
                    };
                    // Router softmax over all experts.
                    let rbuf = bindings.get(router).expect("cpu backend: unbound router");
                    let rbytes = cpu_buf(rbuf).read();
                    let logits = matvec(&rbytes, g.desc(router).dtype, &xs, ne, n_expert);
                    drop(rbytes);
                    let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                    let psum: f32 = probs.iter().sum();
                    for p in probs.iter_mut() {
                        *p /= psum;
                    }
                    // Top-`n_used` experts, renormalized weights.
                    let mut idx: Vec<usize> = (0..n_expert).collect();
                    idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                    idx.truncate(n_used);
                    let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                    // Per-expert stacked-weight byte slices.
                    let gbuf = bindings
                        .get(gate_exps)
                        .expect("cpu backend: unbound gate_exps");
                    let ubuf = bindings.get(up_exps).expect("cpu backend: unbound up_exps");
                    let dbuf = bindings
                        .get(down_exps)
                        .expect("cpu backend: unbound down_exps");
                    let gb = cpu_buf(gbuf).read();
                    let ub = cpu_buf(ubuf).read();
                    let db = cpu_buf(dbuf).read();
                    let (gdt, udt, ddt) = (
                        g.desc(gate_exps).dtype,
                        g.desc(up_exps).dtype,
                        g.desc(down_exps).dtype,
                    );
                    let (gst, ust, dst_) = (
                        gb.len() / n_expert,
                        ub.len() / n_expert,
                        db.len() / n_expert,
                    );
                    let mut out = vec![0f32; ne];
                    for &e in &idx {
                        let gate = matvec(&gb[e * gst..(e + 1) * gst], gdt, &xs, ne, nffx);
                        let up = matvec(&ub[e * ust..(e + 1) * ust], udt, &xs, ne, nffx);
                        let actv: Vec<f32> =
                            (0..nffx).map(|i| act_fn(act, gate[i]) * up[i]).collect();
                        let y = matvec(&db[e * dst_..(e + 1) * dst_], ddt, &actv, nffx, ne);
                        let w_e = probs[e] / wsum * scale;
                        for i in 0..ne {
                            out[i] += w_e * y[i];
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Conv1dSilu {
                    x,
                    weight: w,
                    state,
                    dst,
                    channels,
                    kernel,
                } => {
                    let (cc, kk) = (channels as usize, kernel as usize);
                    let xs = vals[x.0 as usize].clone();
                    let ws = weight(w); // [channels, kernel] row-major (per-channel kernel)
                    let st = &mut vals[state.0 as usize]; // [(kernel-1), channels], oldest row first
                    let mut out = vec![0f32; cc];
                    for ch in 0..cc {
                        // window = [history rows.. , current x]; tap j uses weight[ch*kk + j].
                        let mut acc = 0f32;
                        for j in 0..kk - 1 {
                            acc += st[j * cc + ch] * ws[ch * kk + j];
                        }
                        acc += xs[ch] * ws[ch * kk + (kk - 1)];
                        out[ch] = acc / (1.0 + (-acc).exp()); // silu
                    }
                    // shift history (drop oldest, append raw x).
                    for j in 0..kk.saturating_sub(2) {
                        for ch in 0..cc {
                            st[j * cc + ch] = st[(j + 1) * cc + ch];
                        }
                    }
                    if kk >= 2 {
                        for ch in 0..cc {
                            st[(kk - 2) * cc + ch] = xs[ch];
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::DeltaNet {
                    q,
                    k,
                    v,
                    b,
                    a,
                    a_coef,
                    dt_bias,
                    state,
                    dst,
                    n_vhead,
                    n_khead,
                    head_k,
                    head_v,
                    eps,
                } => {
                    let (nv, nk, kd, vd) = (
                        n_vhead as usize,
                        n_khead as usize,
                        head_k as usize,
                        head_v as usize,
                    );
                    let qf = vals[q.0 as usize].clone();
                    let kf = vals[k.0 as usize].clone();
                    let vf = vals[v.0 as usize].clone();
                    let bf = vals[b.0 as usize].clone();
                    let af = vals[a.0 as usize].clone();
                    let acoef = weight(a_coef);
                    let dtb = weight(dt_bias);
                    let st = &mut vals[state.0 as usize]; // [nv, kd, vd]
                    let mut out = vec![0f32; nv * vd];
                    let qscale = 1.0 / (kd as f32).sqrt();
                    let l2 = |slice: &[f32]| -> f32 {
                        (slice.iter().map(|x| x * x).sum::<f32>() + eps).sqrt()
                    };
                    for h in 0..nv {
                        // GQA: q/k heads are TILED to nv value heads → v-head h uses q/k head h % nk.
                        let kh_idx = h % nk;
                        let mut qh = qf[kh_idx * kd..kh_idx * kd + kd].to_vec();
                        let mut kh = kf[kh_idx * kd..kh_idx * kd + kd].to_vec();
                        let vh = &vf[h * vd..h * vd + vd];
                        let qn = l2(&qh);
                        let kn = l2(&kh);
                        for x in qh.iter_mut() {
                            *x = *x / qn * qscale;
                        }
                        for x in kh.iter_mut() {
                            *x /= kn;
                        }
                        let beta = 1.0 / (1.0 + (-bf[h]).exp());
                        // softplus(a + dt_bias), then g = a_coef * softplus (≤ 0); decay = exp(g).
                        let sp = {
                            let z = af[h] + dtb[h];
                            z.max(0.0) + (-z.abs()).exp().ln_1p()
                        };
                        let decay = (acoef[h] * sp).exp();
                        let sh = &mut st[h * kd * vd..(h + 1) * kd * vd]; // [kd, vd]
                        for x in sh.iter_mut() {
                            *x *= decay;
                        }
                        // kv = kᵀS  [vd]
                        let mut kv = vec![0f32; vd];
                        for kk in 0..kd {
                            let kkv = kh[kk];
                            let row = &sh[kk * vd..kk * vd + vd];
                            for d in 0..vd {
                                kv[d] += kkv * row[d];
                            }
                        }
                        // delta = (v - kv)*beta ; S += k ⊗ delta
                        let delta: Vec<f32> = (0..vd).map(|d| (vh[d] - kv[d]) * beta).collect();
                        for kk in 0..kd {
                            let kkv = kh[kk];
                            let row = &mut sh[kk * vd..kk * vd + vd];
                            for d in 0..vd {
                                row[d] += kkv * delta[d];
                            }
                        }
                        // out = qᵀS  [vd]
                        let oh = &mut out[h * vd..h * vd + vd];
                        for kk in 0..kd {
                            let qv = qh[kk];
                            let row = &sh[kk * vd..kk * vd + vd];
                            for d in 0..vd {
                                oh[d] += qv * row[d];
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
            }
        }

        // Write back the buffers the model reads after execute: Outputs (logits) and mutated f32
        // Inputs (conv/recurrent state). KV caches (`direct`) were written in place by `WriteKv`, so
        // they're skipped — no full-cache copy. Weights are read-only; positions are I32, unchanged.
        for (i, decl) in g.tensors.iter().enumerate() {
            let write_back = matches!(decl.kind, TensorKind::Output)
                || (decl.kind == TensorKind::Input
                    && decl.desc.dtype == DType::F32
                    && !direct.contains(&(i as u32)));
            if !write_back {
                continue;
            }
            if let Some(buf) = bindings.get(TensorId(i as u32)) {
                let mut d = cpu_buf(buf).owned();
                d.copy_from_slice(bytemuck::cast_slice(&vals[i]));
            }
        }
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}

// ─── Qwen3 dense CPU decode runner ───────────────────────────────────────────────
//
// Builds the n=1 decode Graph and drives it through `CpuBackend`, one token at a time, for BOTH
// prompt ingestion (looped) and generation — so no GEMM/flash prefill kernels are needed on CPU.
// The KV cache grows one row per step. Validates the agnostic seam end-to-end against the GPU path.

/// FFN weight handles: a dense gated FFN, or a qwen3moe routed-expert bank (router + stacked
/// per-expert gate/up/down).
enum FfnW {
    Dense {
        wgate: TensorId,
        wup: TensorId,
        wdown: TensorId,
    },
    Moe {
        router: TensorId,
        gate_exps: TensorId,
        up_exps: TensorId,
        down_exps: TensorId,
    },
}

/// Per-layer weight handles captured while building one decode graph (q/k-norm + the gemma
/// sandwich norms are optional; `wv` is absent on gemma4 full-attention layers, which reuse the raw
/// K projection as V). The order they're declared in MUST match the upload order so `weights[i]`
/// binds to `wbufs[i]`.
struct LayerW {
    attn_norm: TensorId,
    wq: TensorId,
    wk: TensorId,
    wv: Option<TensorId>,
    q_norm: Option<TensorId>,
    k_norm: Option<TensorId>,
    wo: TensorId,
    post_attn: Option<TensorId>,
    ffn_norm: TensorId,
    ffn: FfnW,
    post_ffw: Option<TensorId>,
    // gemma4 E2B per-layer input embedding: inp_gate, proj, post_norm.
    pl_inp_gate: Option<TensorId>,
    pl_proj: Option<TensorId>,
    pl_post_norm: Option<TensorId>,
}

/// Handles into one freshly-built decode graph that the driver re-binds each step.
struct DecodeHandles {
    hidden: TensorId,
    positions: TensorId,
    rope_freqs: Option<TensorId>, // gemma4 proportional-RoPE divisors (full-attention layers)
    per_layer_inp: Option<TensorId>, // gemma4 E2B per-(token,layer) input vector `[n_layer*npl]`
    logits: TensorId,
    k_cache: Vec<TensorId>,
    v_cache: Vec<TensorId>,
    weights: Vec<TensorId>, // flat, in declaration == upload order
}

/// Greedy CPU generation for a decoder (Qwen3 / Llama / Gemma 3 / Gemma 4 dense+E2B / qwen3moe). The
/// attention block is shared; the FFN is either a dense gated FFN or a routed-expert MoE bank; gemma4
/// E2B adds per-layer input embeddings + KV-layer sharing. `prompt` is the full token prefix; returns
/// the generated continuation. Stops at EOS or `max_new`.
pub(crate) fn generate_dense_cpu(
    g: &Gguf,
    cfg: &Config,
    token_embd: &[f32],
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    mut on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, CpuStats)> {
    let c = cfg;
    let be = CpuBackend::new();
    let (ne, nh) = (c.n_embd, c.n_head);
    // gemma4: per-layer SWA/full dims differ; size shared scratch + KV by the max over layers.
    let max_hd = c.max_head_dim();
    let max_kvrow = c.max_n_kv() * max_hd;
    let max_qrow = nh * max_hd;
    let nff = c.n_ff; // max FFN width
    let gemma = c.gemma;
    let gemma4 = c.gemma4;
    let qk_norm = c.qk_norm;
    let act = if gemma {
        Activation::Gelu
    } else {
        Activation::Silu
    };
    let max_ctx = prompt.len() + max_new + 1;
    // gemma4 E2B (gemma3n): per-layer input embeddings + KV-layer sharing.
    let e2b = c.n_embd_per_layer > 0;
    let npl = c.n_embd_per_layer;

    // Per-layer presence of an explicit V projection. gemma4 full-attention layers omit it (V = the
    // raw K projection); every layer of every other model has one.
    let has_wv: Vec<bool> = (0..c.n_layer)
        .map(|l| {
            g.tensors()
                .iter()
                .any(|t| t.name == format!("blk.{l}.attn_v.weight"))
        })
        .collect();
    // gemma4 per-layer output scale (`layer_output_scale.weight`, a single scalar multiplying the
    // layer output before the next layer). Read host-side; applied as an `Op::Scale`.
    let out_scale: Vec<Option<f32>> = (0..c.n_layer)
        .map(|l| {
            let name = format!("blk.{l}.layer_output_scale.weight");
            if g.tensors().iter().any(|t| t.name == name) {
                crate::load_tensor_dequant(g, &name)
                    .ok()
                    .and_then(|(v, _)| v.first().copied())
            } else {
                None
            }
        })
        .collect();
    // gemma4 proportional-RoPE frequency divisors (`rope_freqs.weight`, `[rope_dim/2]`): applied on
    // full-attention layers only (SWA layers use plain RoPE). Bound as a per-step f32 Input.
    let rope_freqs: Option<Vec<f32>> =
        if gemma4 && g.tensors().iter().any(|t| t.name == "rope_freqs.weight") {
            Some(crate::load_tensor_dequant(g, "rope_freqs.weight").map(|(v, _)| v)?)
        } else {
            None
        };

    // ── upload weights in their NATIVE GGUF dtype (no host pre-dequant — the backend dequants
    //    lazily in `bytes_to_f32`, so a quant weight occupies ~quant size, not 8× f32). `wspecs`
    //    records each (dtype, numel) so `build` can declare the handle with the matching dtype; its
    //    order MUST equal the `g.weight()` order in `build` below. ──────────────────────────────────
    let mut wbufs: Vec<Box<dyn Buffer>> = Vec::new();
    let mut wspecs: Vec<(DType, usize)> = Vec::new();
    // Map a weight tensor zero-copy from the GGUF mmap (no alloc, no memcpy); record its native dtype
    // + element count so `build` declares the handle to match.
    let mut wraw = |name: &str| -> AResult<()> {
        let info = g
            .tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?
            .clone();
        let numel: usize = info.shape.iter().product();
        let tb = g.tensor_bytes_arc(name).map_err(|e| anyhow!("{e}"))?;
        wbufs.push(be.map_weight(tb));
        wspecs.push((info.dtype, numel));
        Ok(())
    };
    for l in 0..c.n_layer {
        let p = |s: &str| format!("blk.{l}.{s}");
        wraw(&p("attn_norm.weight"))?;
        wraw(&p("attn_q.weight"))?;
        wraw(&p("attn_k.weight"))?;
        if has_wv[l] {
            wraw(&p("attn_v.weight"))?;
        }
        if qk_norm {
            wraw(&p("attn_q_norm.weight"))?;
            wraw(&p("attn_k_norm.weight"))?;
        }
        wraw(&p("attn_output.weight"))?;
        if gemma {
            wraw(&p("post_attention_norm.weight"))?;
        }
        wraw(&p("ffn_norm.weight"))?;
        if c.moe.is_some() {
            // qwen3moe: router + stacked per-expert gate/up/down banks.
            wraw(&p("ffn_gate_inp.weight"))?;
            wraw(&p("ffn_gate_exps.weight"))?;
            wraw(&p("ffn_up_exps.weight"))?;
            wraw(&p("ffn_down_exps.weight"))?;
        } else {
            wraw(&p("ffn_gate.weight"))?;
            wraw(&p("ffn_up.weight"))?;
            wraw(&p("ffn_down.weight"))?;
        }
        if gemma {
            wraw(&p("post_ffw_norm.weight"))?;
        }
        if e2b {
            // gemma4 E2B per-layer input-embedding application weights.
            wraw(&p("inp_gate.weight"))?;
            wraw(&p("proj.weight"))?;
            wraw(&p("post_norm.weight"))?;
        }
    }
    // Globals: output_norm, lm_head. lm_head = `output.weight`, or (tied) the quantized
    // `token_embd.weight` mapped from the mmap and dequantized per-row by `Op::Linear` — same f32
    // values as the host `token_embd`, but zero-copy.
    wraw("output_norm.weight")?;
    if g.tensors().iter().any(|t| t.name == "output.weight") {
        wraw("output.weight")?;
    } else {
        wraw("token_embd.weight")?;
    }
    // gemma4 weightless per-head V-norm = `QkNorm` with a unit weight (out = x/rms). One ones-vector
    // of the max head dim serves every layer (a narrower layer reads its leading prefix).
    if gemma4 {
        let ones = vec![1.0f32; max_hd];
        let b = be
            .alloc(ones.len() * 4, BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(b.as_ref(), bytemuck::cast_slice(&ones))
            .map_err(|e| anyhow!("{e}"))?;
        wbufs.push(b);
        wspecs.push((DType::F32, max_hd));
    }

    // ── persistent KV cache buffers (f32), sized per-layer (gemma4 SWA layers are narrower) ───────
    let mut kbufs: Vec<Box<dyn Buffer>> = Vec::new();
    let mut vbufs: Vec<Box<dyn Buffer>> = Vec::new();
    for l in 0..c.n_layer {
        let kvrow_l = c.layer_n_kv(l) * c.layer_head_dim(l);
        // f16 KV cache (2 bytes/elem) — matches the graph's f16 k_cache/v_cache decls.
        kbufs.push(
            be.alloc(max_ctx * kvrow_l * 2, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?,
        );
        vbufs.push(
            be.alloc(max_ctx * kvrow_l * 2, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?,
        );
    }

    // ── per-step IO buffers ────────────────────────────────────────────────────────
    let hidden_buf = be
        .alloc(ne * 4, BufferUsage::Staging)
        .map_err(|e| anyhow!("{e}"))?;
    let pos_buf = be
        .alloc(4, BufferUsage::Staging)
        .map_err(|e| anyhow!("{e}"))?;
    let rf_buf = match &rope_freqs {
        Some(rf) => {
            let b = be
                .alloc(rf.len() * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(b.as_ref(), bytemuck::cast_slice(rf))
                .map_err(|e| anyhow!("{e}"))?;
            Some((b, rf.len()))
        }
        None => None,
    };
    // gemma4 E2B per-(token,layer) input vector `[n_layer*npl]`, recomputed + re-uploaded each step.
    let ipl_buf = if e2b {
        Some(
            be.alloc(c.n_layer * npl * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?,
        )
    } else {
        None
    };
    let logits_buf = be
        .alloc(c.vocab * 4, BufferUsage::Readback)
        .map_err(|e| anyhow!("{e}"))?;

    // Build the decode graph for a given absolute position (kv_len = pos+1).
    let build = |pos: usize| -> (Graph, DecodeHandles) {
        let mut g = Graph::new();
        let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
        // KV cache is f16 — matches the GPU's f16 cache (halves memory, tightens CPU↔GPU parity).
        let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16);
        let hidden = g.input(f32d(ne));
        let positions = g.input(TensorDesc::new(vec![1], DType::I32));
        let rope_freqs = rf_buf.as_ref().map(|(_, n)| g.input(f32d(*n)));
        // gemma4 E2B per-(token,layer) input vector `[n_layer*npl]` (computed host-side each step).
        let per_layer_inp = if e2b {
            Some(g.input(f32d(c.n_layer * npl)))
        } else {
            None
        };
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        for l in 0..c.n_layer {
            let kvrow_l = c.layer_n_kv(l) * c.layer_head_dim(l);
            k_cache.push(g.input(f16d(max_ctx * kvrow_l)));
            v_cache.push(g.input(f16d(max_ctx * kvrow_l)));
        }

        // Weights — declared in the SAME order as the upload loop, pulling (dtype, numel) from
        // `wspecs` so each handle carries its native GGUF dtype (the backend dequants on read).
        // `wpush` records the handle in the flat `weights` list (for binding) and returns it.
        let mut weights: Vec<TensorId> = Vec::new();
        let mut wi = 0usize;
        let mut wpush = |g: &mut Graph, weights: &mut Vec<TensorId>| -> TensorId {
            let (dt, n) = wspecs[wi];
            wi += 1;
            let id = g.weight(TensorDesc::new(vec![n], dt));
            weights.push(id);
            id
        };
        let mut lw: Vec<LayerW> = Vec::new();
        for l in 0..c.n_layer {
            let attn_norm = wpush(&mut g, &mut weights);
            let wq = wpush(&mut g, &mut weights);
            let wk = wpush(&mut g, &mut weights);
            let wv = if has_wv[l] {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let (q_norm, k_norm) = if qk_norm {
                (
                    Some(wpush(&mut g, &mut weights)),
                    Some(wpush(&mut g, &mut weights)),
                )
            } else {
                (None, None)
            };
            let wo = wpush(&mut g, &mut weights);
            let post_attn = if gemma {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let ffn_norm = wpush(&mut g, &mut weights);
            let ffn = if c.moe.is_some() {
                FfnW::Moe {
                    router: wpush(&mut g, &mut weights),
                    gate_exps: wpush(&mut g, &mut weights),
                    up_exps: wpush(&mut g, &mut weights),
                    down_exps: wpush(&mut g, &mut weights),
                }
            } else {
                FfnW::Dense {
                    wgate: wpush(&mut g, &mut weights),
                    wup: wpush(&mut g, &mut weights),
                    wdown: wpush(&mut g, &mut weights),
                }
            };
            let post_ffw = if gemma {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let (pl_inp_gate, pl_proj, pl_post_norm) = if e2b {
                (
                    Some(wpush(&mut g, &mut weights)),
                    Some(wpush(&mut g, &mut weights)),
                    Some(wpush(&mut g, &mut weights)),
                )
            } else {
                (None, None, None)
            };
            lw.push(LayerW {
                attn_norm,
                wq,
                wk,
                wv,
                q_norm,
                k_norm,
                wo,
                post_attn,
                ffn_norm,
                ffn,
                post_ffw,
                pl_inp_gate,
                pl_proj,
                pl_post_norm,
            });
        }
        let w_out_norm = wpush(&mut g, &mut weights);
        let w_lm = wpush(&mut g, &mut weights);
        let v_ones = if gemma4 {
            Some(wpush(&mut g, &mut weights))
        } else {
            None
        };
        let logits = g.output(f32d(c.vocab));

        // scratch (sized to the per-layer max; ops reallocate dst, so these are upper bounds)
        let hn = g.internal(f32d(ne));
        let q = g.internal(f32d(max_qrow));
        let k = g.internal(f32d(max_kvrow));
        let v = g.internal(f32d(max_kvrow));
        let attn = g.internal(f32d(max_qrow));
        let gbuf = g.internal(f32d(nff));
        let ubuf = g.internal(f32d(nff));
        let actbuf = g.internal(f32d(nff));
        let sub = g.internal(f32d(ne));
        // E2B per-layer embed scratch: gate `[npl]` and projected `[ne]`.
        let plg = g.internal(f32d(npl.max(1)));
        let plp = g.internal(f32d(ne));

        let eps = c.rms_eps;
        for (l, lw) in lw.iter().enumerate() {
            // Per-layer dims (gemma4 SWA vs full; uniform for every other model).
            let hd = c.layer_head_dim(l);
            let nkv = c.layer_n_kv(l);
            let kvrow = nkv * hd;
            let qrow = nh * hd;
            let nff_l = c.layer_n_ff(l);
            let theta = c.layer_rope_theta(l); // gemma dual-rope (SWA 1e4 / full 1e6); uniform else
            let rope_dim = c.layer_rope_dim(l);
            let swa = gemma && c.is_swa_layer(l);
            let mask = if swa {
                AttnMask::SlidingWindow(c.swa_window)
            } else {
                AttnMask::Causal
            };
            // gemma4: attn scale 1.0 (QK-norm controls magnitude); everyone else 1/√hd.
            let scale = if gemma4 {
                1.0
            } else {
                1.0 / (hd as f32).sqrt()
            };
            // gemma4 proportional-RoPE applies only on full-attention layers.
            let layer_ff = if gemma4 && !swa { rope_freqs } else { None };
            // attn input norm
            g.push(Op::RmsNorm {
                x: hidden,
                weight: lw.attn_norm,
                dst: hn,
                rows: 1,
                dim: ne as u32,
                eps,
            });
            g.push(Op::Linear {
                x: hn,
                weight: lw.wq,
                dst: q,
                m: 1,
                in_f: ne as u32,
                out_f: qrow as u32,
            });
            // gemma4 E2B KV-layer sharing: shared layers compute Q only and attend to an earlier
            // layer's cache. `own_kv`/`kv_src` are `true`/`l` for every layer of a non-sharing model.
            let own_kv = c.has_own_kv(l);
            let kv_src = c.kv_src_layer(l);
            if own_kv {
                g.push(Op::Linear {
                    x: hn,
                    weight: lw.wk,
                    dst: k,
                    m: 1,
                    in_f: ne as u32,
                    out_f: kvrow as u32,
                });
                // V projection, or (gemma4 full layers) V = the raw K projection, copied BEFORE K is
                // QK-normed + RoPE'd.
                match lw.wv {
                    Some(wv) => g.push(Op::Linear {
                        x: hn,
                        weight: wv,
                        dst: v,
                        m: 1,
                        in_f: ne as u32,
                        out_f: kvrow as u32,
                    }),
                    None => g.push(Op::Copy {
                        src: k,
                        src_off: 0,
                        dst: v,
                        dst_off: 0,
                        n: kvrow as u32,
                    }),
                }
                // K: fused QkNorm+RoPE when the model has K-norm (qwen3/gemma), else RoPE alone (llama).
                match lw.k_norm {
                    Some(kn) => g.push(Op::QkNormRope {
                        x: k,
                        weight: kn,
                        positions,
                        dst: k,
                        rows: 1,
                        n_head: nkv as u32,
                        head_dim: hd as u32,
                        rope_dim: rope_dim as u32,
                        theta,
                        eps,
                        freq_factors: layer_ff,
                    }),
                    None => g.push(Op::Rope {
                        x: k,
                        positions,
                        dst: k,
                        rows: 1,
                        n_head: nkv as u32,
                        head_dim: hd as u32,
                        rope_dim: rope_dim as u32,
                        theta,
                        freq_factors: layer_ff,
                    }),
                }
                // gemma4 weightless per-head RMSNorm on V (= x/rms) before caching.
                if let Some(ones) = v_ones {
                    g.push(Op::QkNorm {
                        x: v,
                        weight: ones,
                        dst: v,
                        rows: 1,
                        n_head: nkv as u32,
                        head_dim: hd as u32,
                        eps,
                    });
                }
                g.push(Op::WriteKv {
                    src: k,
                    cache: k_cache[l],
                    rows: 1,
                    row_stride: kvrow as u32,
                    pos: pos as u32,
                });
                g.push(Op::WriteKv {
                    src: v,
                    cache: v_cache[l],
                    rows: 1,
                    row_stride: kvrow as u32,
                    pos: pos as u32,
                });
            }
            // Q: fused QkNorm+RoPE when the model has Q-norm (qwen3/gemma), else RoPE alone (llama).
            match lw.q_norm {
                Some(qn) => g.push(Op::QkNormRope {
                    x: q,
                    weight: qn,
                    positions,
                    dst: q,
                    rows: 1,
                    n_head: nh as u32,
                    head_dim: hd as u32,
                    rope_dim: rope_dim as u32,
                    theta,
                    eps,
                    freq_factors: layer_ff,
                }),
                None => g.push(Op::Rope {
                    x: q,
                    positions,
                    dst: q,
                    rows: 1,
                    n_head: nh as u32,
                    head_dim: hd as u32,
                    rope_dim: rope_dim as u32,
                    theta,
                    freq_factors: layer_ff,
                }),
            }
            g.push(Op::Attention {
                q,
                k_cache: k_cache[kv_src],
                v_cache: v_cache[kv_src],
                dst: attn,
                rows: 1,
                kv_len: (pos + 1) as u32,
                n_head: nh as u32,
                n_kv: nkv as u32,
                head_dim: hd as u32,
                scale,
                mask,
                pos: pos as u32,
            });
            g.push(Op::Linear {
                x: attn,
                weight: lw.wo,
                dst: sub,
                m: 1,
                in_f: qrow as u32,
                out_f: ne as u32,
            });
            // gemma sandwich: post-attention norm on the sublayer output BEFORE the residual add.
            if let Some(pa) = lw.post_attn {
                g.push(Op::RmsNorm {
                    x: sub,
                    weight: pa,
                    dst: sub,
                    rows: 1,
                    dim: ne as u32,
                    eps,
                });
            }
            g.push(Op::Add {
                a: hidden,
                b: sub,
                dst: hidden,
                n: ne as u32,
            });
            // ffn
            g.push(Op::RmsNorm {
                x: hidden,
                weight: lw.ffn_norm,
                dst: hn,
                rows: 1,
                dim: ne as u32,
                eps,
            });
            match lw.ffn {
                FfnW::Dense { wgate, wup, wdown } => {
                    g.push(Op::Linear {
                        x: hn,
                        weight: wgate,
                        dst: gbuf,
                        m: 1,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                    });
                    g.push(Op::Linear {
                        x: hn,
                        weight: wup,
                        dst: ubuf,
                        m: 1,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                    });
                    g.push(Op::GatedAct {
                        gate: gbuf,
                        up: ubuf,
                        dst: actbuf,
                        rows: 1,
                        nff: nff_l as u32,
                        act,
                        up_off: 0,
                    });
                    g.push(Op::Linear {
                        x: actbuf,
                        weight: wdown,
                        dst: sub,
                        m: 1,
                        in_f: nff_l as u32,
                        out_f: ne as u32,
                    });
                }
                FfnW::Moe {
                    router,
                    gate_exps,
                    up_exps,
                    down_exps,
                } => {
                    let mc = c.moe.expect("moe layer without MoeConfig");
                    g.push(Op::MoeFfn {
                        x: hn,
                        router,
                        gate_exps,
                        up_exps,
                        down_exps,
                        dst: sub,
                        ne: ne as u32,
                        n_expert: mc.n_expert as u32,
                        n_used: mc.n_used as u32,
                        n_ff_exp: mc.n_ff_exp as u32,
                        scale: mc.scale,
                        act, // qwen3moe: SwiGLU (act == Silu)
                    });
                }
            }
            if let Some(pf) = lw.post_ffw {
                g.push(Op::RmsNorm {
                    x: sub,
                    weight: pf,
                    dst: sub,
                    rows: 1,
                    dim: ne as u32,
                    eps,
                });
            }
            g.push(Op::Add {
                a: hidden,
                b: sub,
                dst: hidden,
                n: ne as u32,
            });
            // gemma4 E2B per-layer input embedding (gemma3n): mix this layer's input vector into
            // `hidden` after the FFN residual. `g = gelu(inp_gate·hidden) * inp_per_layer[l]`,
            // `p = post_norm(proj·g)`, `hidden += p`.
            if let (Some(gate_w), Some(proj_w), Some(post_norm), Some(ipl)) =
                (lw.pl_inp_gate, lw.pl_proj, lw.pl_post_norm, per_layer_inp)
            {
                g.push(Op::Linear {
                    x: hidden,
                    weight: gate_w,
                    dst: plg,
                    m: 1,
                    in_f: ne as u32,
                    out_f: npl as u32,
                });
                // gelu(plg) * ipl[l*npl .. l*npl+npl]  (the layer's slice of the input vector).
                g.push(Op::GatedAct {
                    gate: plg,
                    up: ipl,
                    dst: plg,
                    rows: 1,
                    nff: npl as u32,
                    act: Activation::Gelu,
                    up_off: (l * npl) as u32,
                });
                g.push(Op::Linear {
                    x: plg,
                    weight: proj_w,
                    dst: plp,
                    m: 1,
                    in_f: npl as u32,
                    out_f: ne as u32,
                });
                g.push(Op::RmsNorm {
                    x: plp,
                    weight: post_norm,
                    dst: plp,
                    rows: 1,
                    dim: ne as u32,
                    eps,
                });
                g.push(Op::Add {
                    a: hidden,
                    b: plp,
                    dst: hidden,
                    n: ne as u32,
                });
            }
            // gemma4: scale the whole layer output by the per-layer scalar before the next layer.
            if let Some(s) = out_scale[l] {
                g.push(Op::Scale {
                    x: hidden,
                    dst: hidden,
                    s,
                    n: ne as u32,
                });
            }
        }
        g.push(Op::RmsNorm {
            x: hidden,
            weight: w_out_norm,
            dst: hn,
            rows: 1,
            dim: ne as u32,
            eps,
        });
        g.push(Op::Linear {
            x: hn,
            weight: w_lm,
            dst: logits,
            m: 1,
            in_f: ne as u32,
            out_f: c.vocab as u32,
        });
        if c.final_softcap > 0.0 {
            g.push(Op::Softcap {
                x: logits,
                dst: logits,
                cap: c.final_softcap,
                n: c.vocab as u32,
            });
        }
        (
            g,
            DecodeHandles {
                hidden,
                positions,
                rope_freqs,
                per_layer_inp,
                logits,
                k_cache,
                v_cache,
                weights,
            },
        )
    };

    // ── drive ───────────────────────────────────────────────────────────────────────
    let embed_scale = if gemma { (ne as f32).sqrt() } else { 1.0 };
    let mut out = Vec::new();
    let mut cur = prompt.to_vec();
    let mut logits = vec![0f32; c.vocab];
    // INFR_PROF=1: report prompt-ingest + decode tok/s to stderr (CPU perf iteration).
    let prof = std::env::var("INFR_PROF").is_ok();
    let mut prompt_t = std::time::Duration::ZERO;
    let mut decode_t = std::time::Duration::ZERO;
    let mut decode_n = 0usize;
    for pos in 0..(prompt.len() + max_new) {
        let step_t0 = std::time::Instant::now();
        let tok = cur[pos] as usize;
        // embed (gemma scales by √n_embd; qwen3/llama identity)
        let emb: Vec<f32> = token_embd[tok * ne..tok * ne + ne]
            .iter()
            .map(|&x| x * embed_scale)
            .collect();
        be.upload(hidden_buf.as_ref(), bytemuck::cast_slice(&emb))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(pos_buf.as_ref(), bytemuck::cast_slice(&[pos as i32]))
            .map_err(|e| anyhow!("{e}"))?;

        // gemma4 E2B: build this token's per-layer input vector on the host (mirrors the GPU forward):
        // `ipl[l] = ((model_proj_l·emb)/√n_embd, RMSNorm'd over npl) + (per_layer_tok_embd_row × √npl)) / √2`.
        if let (Some(ple), Some(ipl_buf)) = (ple, &ipl_buf) {
            let (npl, nl, nem) = (ple.npl, ple.n_layer, ple.n_embd);
            let inv_sqrt_ne = 1.0 / (nem as f32).sqrt();
            let sqrt_npl = (npl as f32).sqrt();
            let inv_sqrt2 = 1.0 / 2f32.sqrt();
            let te_bytes = g
                .tensor_bytes("per_layer_token_embd.weight")
                .map_err(|e| anyhow!("{e}"))?;
            let r0 = tok * ple.tok_embd_row_bytes;
            let pl_tok = dequant_block(
                ple.tok_embd_dtype,
                &te_bytes[r0..r0 + ple.tok_embd_row_bytes],
            )
            .map_err(|e| anyhow!("{e}"))?;
            let mut ipl = vec![0f32; nl * npl];
            for layer in 0..nl {
                let mut proj = vec![0f32; npl];
                let mut ss = 0f32;
                for (j, pj) in proj.iter_mut().enumerate() {
                    let wrow =
                        &ple.model_proj[(layer * npl + j) * nem..(layer * npl + j) * nem + nem];
                    let acc: f32 = wrow.iter().zip(&emb).map(|(a, b)| a * b).sum();
                    let v = acc * inv_sqrt_ne;
                    *pj = v;
                    ss += v * v;
                }
                let rms = 1.0 / (ss / npl as f32 + c.rms_eps).sqrt();
                for j in 0..npl {
                    let normed = proj[j] * rms * ple.proj_norm[j];
                    let tokv = pl_tok[layer * npl + j] * sqrt_npl;
                    ipl[layer * npl + j] = (normed + tokv) * inv_sqrt2;
                }
            }
            be.upload(ipl_buf.as_ref(), bytemuck::cast_slice(&ipl))
                .map_err(|e| anyhow!("{e}"))?;
        }

        let (g, h) = build(pos);
        let plan = be.compile(&g).map_err(|e| anyhow!("{e}"))?;
        let mut b = Bindings::new();
        b.bind(h.hidden, hidden_buf.as_ref());
        b.bind(h.positions, pos_buf.as_ref());
        if let (Some(rid), Some((rb, _))) = (h.rope_freqs, &rf_buf) {
            b.bind(rid, rb.as_ref());
        }
        if let (Some(pid), Some(ib)) = (h.per_layer_inp, &ipl_buf) {
            b.bind(pid, ib.as_ref());
        }
        for l in 0..c.n_layer {
            b.bind(h.k_cache[l], kbufs[l].as_ref());
            b.bind(h.v_cache[l], vbufs[l].as_ref());
        }
        for (i, wid) in h.weights.iter().enumerate() {
            b.bind(*wid, wbufs[i].as_ref());
        }
        b.bind(h.logits, logits_buf.as_ref());
        be.execute(plan.as_ref(), &b).map_err(|e| anyhow!("{e}"))?;

        // Only sample once we're past the prompt (decode position = last prompt token onward).
        let is_decode = pos + 1 >= prompt.len();
        if is_decode {
            be.download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
                .map_err(|e| anyhow!("{e}"))?;
            let next = argmax(&logits) as u32;
            let is_eos = c.eos_ids.contains(&next) || next == c.eos;
            out.push(next);
            decode_t += step_t0.elapsed();
            decode_n += 1;
            if !is_eos {
                on_token(next); // stream the token (EOS is not emitted)
            }
            if is_eos || out.len() >= max_new {
                break;
            }
            if cur.len() <= pos + 1 {
                cur.push(next);
            }
        } else {
            prompt_t += step_t0.elapsed();
        }
    }
    if prof {
        let ts = |d: std::time::Duration, n: usize| {
            if d.as_secs_f64() > 0.0 {
                n as f64 / d.as_secs_f64()
            } else {
                0.0
            }
        };
        eprintln!(
            "[cpu prof] prompt {} tok in {:.2}s ({:.1} tok/s) | decode {} tok in {:.2}s ({:.2} tok/s)",
            prompt.len(),
            prompt_t.as_secs_f64(),
            ts(prompt_t, prompt.len()),
            decode_n,
            decode_t.as_secs_f64(),
            ts(decode_t, decode_n),
        );
    }
    let stats = CpuStats {
        n_prompt: prompt.len(),
        prompt_secs: prompt_t.as_secs_f64(),
        n_gen: decode_n,
        decode_secs: decode_t.as_secs_f64(),
    };
    Ok((out, stats))
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

#[cfg(test)]
mod kernel_tests {
    //! CPU-only, no GPU, no model file: the optimized quant/f16 dot kernels must match the trusted
    //! f32 reference (`dequant_block` → naive `dot`) on the SAME bytes. We dot against the *quantized*
    //! activation (`d8 * q8`) so the only difference is f32 summation order — i.e. this isolates
    //! kernel correctness from the (separate, expected) Q8 activation-quant error.
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed
    }
    fn det_bytes(n: usize, mut seed: u64) -> Vec<u8> {
        (0..n).map(|_| (lcg(&mut seed) >> 33) as u8).collect()
    }
    fn det_x(n: usize, mut seed: u64) -> Vec<f32> {
        (0..n)
            .map(|_| ((lcg(&mut seed) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
            .collect()
    }
    fn put_f16(b: &mut [u8], v: f32) {
        b.copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    /// The reference activation the integer kernels actually see: `d8 * q8` per super-block.
    fn dequant_q8(q8: &Q8) -> Vec<f32> {
        let mut x = vec![0f32; q8.qs.len()];
        for (b, &d) in q8.d.iter().enumerate() {
            for i in 0..256 {
                x[b * 256 + i] = d * q8.qs[b * 256 + i] as f32;
            }
        }
        x
    }
    fn rel_err(got: f32, want: f32) -> f32 {
        (got - want).abs() / want.abs().max(1.0)
    }

    #[test]
    fn q4k_dot_matches_dequant_reference() {
        let in_f = 768; // 3 super-blocks
        let nb = in_f / 256;
        let mut w = det_bytes(nb * 144, 1);
        for k in 0..nb {
            put_f16(&mut w[k * 144..k * 144 + 2], 0.05); // d
            put_f16(&mut w[k * 144 + 2..k * 144 + 4], 0.015); // dmin
        }
        let wref = crate::dequant_block(DType::Q4K, &w).unwrap();
        let q8 = quantize_q8(&det_x(in_f, 2));
        let got = vec_dot_q4k(&w, &q8, in_f);
        let want = dot(&wref, &dequant_q8(&q8));
        assert!(rel_err(got, want) < 1e-3, "q4k: got {got}, want {want}");
    }

    #[test]
    fn q6k_dot_matches_dequant_reference() {
        let in_f = 768;
        let nb = in_f / 256;
        let mut w = det_bytes(nb * 210, 3);
        for k in 0..nb {
            put_f16(&mut w[k * 210 + 208..k * 210 + 210], 0.04); // d
        }
        let wref = crate::dequant_block(DType::Q6K, &w).unwrap();
        let q8 = quantize_q8(&det_x(in_f, 4));
        let got = vec_dot_q6k(&w, &q8, in_f);
        let want = dot(&wref, &dequant_q8(&q8));
        assert!(rel_err(got, want) < 1e-3, "q6k: got {got}, want {want}");
    }

    #[test]
    fn f16_dot_matches_reference() {
        let n = 257; // odd, exercises the tail past the 8-wide chunks
        let x = det_x(n, 5);
        let wf = det_x(n, 6);
        let wbytes: Vec<u8> = wf
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let wref: Vec<f32> = wf
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        assert!(rel_err(dot_f16(&wbytes, &x), dot(&wref, &x)) < 1e-4);
    }

    #[test]
    fn bf16_dot_matches_reference() {
        let n = 130;
        let x = det_x(n, 7);
        let wf = det_x(n, 8);
        let wbytes: Vec<u8> = wf
            .iter()
            .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes()) // bf16 = top 16 bits
            .collect();
        let wref: Vec<f32> = wf
            .iter()
            .map(|&v| f32::from_bits((v.to_bits() >> 16) << 16))
            .collect();
        assert!(rel_err(dot_bf16(&wbytes, &x), dot(&wref, &x)) < 1e-4);
    }
}
