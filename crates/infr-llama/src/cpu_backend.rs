//! CPU reference backend — a correctness-first interpreter of the backend-agnostic
//! [`infr_core`] compute [`Graph`]. No SIMD, no threading yet: every op is a plain scalar loop,
//! quantized weights are dequantized on the host (cached). It exists to (a) run every model
//! without a GPU and (b) serve as the oracle the GPU backends are validated against.
//!
//! Lives in `infr-llama` for now (next to [`crate::dequant_block`] + the qwen35 CPU oracle) to
//! avoid a circular crate dep; it implements the agnostic `infr_core::Backend` trait, so it can be
//! extracted to an `infr-cpu` crate later without touching callers.

use crate::{dequant_block, load_tensor_dequant, Llama};
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Capabilities, Plan};
use infr_core::error::Result;
use infr_core::graph::{Activation, AttnMask, Graph, Op, TensorKind};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::WeightSource;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A host buffer: a plain byte vector behind a `Mutex` (so `&dyn Buffer` stays `Send + Sync` and
/// `upload`/`download`/in-place writes are safe single-threaded).
pub struct CpuBuffer {
    data: Mutex<Vec<u8>>,
}

impl Buffer for CpuBuffer {
    fn len_bytes(&self) -> usize {
        self.data.lock().unwrap().len()
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
    /// every step, so dequant once and reuse — otherwise we'd dequant the whole model per token).
    weight_cache: Mutex<HashMap<usize, Arc<Vec<f32>>>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
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
        Ok(Box::new(CpuBuffer {
            data: Mutex::new(vec![0u8; bytes.max(4)]),
        }))
    }

    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        let mut d = cpu_buf(dst).data.lock().unwrap();
        d[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        let s = cpu_buf(src).data.lock().unwrap();
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

        // f32 working store for every Input/Internal/Output handle (weights are read on demand from
        // the dequant cache, never materialized here — that would re-dequant the model each step).
        let mut vals: Vec<Vec<f32>> = vec![Vec::new(); g.tensors.len()];
        for (i, decl) in g.tensors.iter().enumerate() {
            match decl.kind {
                TensorKind::Internal | TensorKind::Output => {
                    vals[i] = vec![0f32; decl.desc.numel()]
                }
                TensorKind::Input => {
                    let buf = bindings
                        .get(TensorId(i as u32))
                        .expect("cpu backend: unbound Input");
                    let bytes = cpu_buf(buf).data.lock().unwrap();
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
            let bytes = cpu_buf(buf).data.lock().unwrap();
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
                    let ws = weight(w); // row-major [out_f, in_f]: row o = ws[o*in_f .. o*in_f+in_f]
                    let mut out = vec![0f32; m * out_f];
                    for r in 0..m {
                        let xb = r * in_f;
                        for o in 0..out_f {
                            let wb = o * in_f;
                            let mut acc = 0f32;
                            for k in 0..in_f {
                                acc += ws[wb + k] * xs[xb + k];
                            }
                            out[r * out_f + o] = acc;
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
                Op::WriteKv {
                    src,
                    cache,
                    rows,
                    row_stride,
                    pos,
                } => {
                    let (rows, rs, pos) = (rows as usize, row_stride as usize, pos as usize);
                    let s = vals[src.0 as usize].clone();
                    let dst = &mut vals[cache.0 as usize];
                    let base = pos * rs;
                    dst[base..base + rows * rs].copy_from_slice(&s[..rows * rs]);
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
                    let ks = &vals[k_cache.0 as usize];
                    let vs = &vals[v_cache.0 as usize];
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
                    let _ = kv_len;
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
                            let g = gs[gb + i];
                            let a = match act {
                                Activation::Silu => g / (1.0 + (-g).exp()),
                                // gelu_pytorch_tanh: 0.5 g (1 + tanh(√(2/π)·(g + 0.044715 g³)))
                                Activation::Gelu => {
                                    0.5 * g
                                        * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh())
                                }
                            };
                            out[gb + i] = a * us[ub + i];
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
            }
        }

        // Write back the buffers the model reads after execute: Outputs (logits) and mutated
        // Inputs (the KV cache). Weights are read-only; positions are I32 and unchanged.
        for (i, decl) in g.tensors.iter().enumerate() {
            let write_back = matches!(decl.kind, TensorKind::Output)
                || (decl.kind == TensorKind::Input && decl.desc.dtype == DType::F32);
            if !write_back {
                continue;
            }
            if let Some(buf) = bindings.get(TensorId(i as u32)) {
                let mut d = cpu_buf(buf).data.lock().unwrap();
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

/// Per-layer weight handles captured while building one decode graph (q/k-norm + the gemma
/// sandwich norms are optional). The order they're declared in MUST match the upload order so
/// `weights[i]` binds to `wbufs[i]`.
struct LayerW {
    attn_norm: TensorId,
    wq: TensorId,
    wk: TensorId,
    wv: TensorId,
    q_norm: Option<TensorId>,
    k_norm: Option<TensorId>,
    wo: TensorId,
    post_attn: Option<TensorId>,
    ffn_norm: TensorId,
    wgate: TensorId,
    wup: TensorId,
    wdown: TensorId,
    post_ffw: Option<TensorId>,
}

/// Handles into one freshly-built decode graph that the driver re-binds each step.
struct DecodeHandles {
    hidden: TensorId,
    positions: TensorId,
    logits: TensorId,
    k_cache: Vec<TensorId>,
    v_cache: Vec<TensorId>,
    weights: Vec<TensorId>, // flat, in declaration == upload order
}

/// Greedy CPU generation for a dense decoder (Qwen3 / Llama / Gemma 3). `prompt` is the full token
/// prefix; returns the generated continuation. Stops at EOS or `max_new`. gemma4 (per-layer head
/// dims, V-norm, layer-output scale, E2B) and MoE are not handled yet.
pub fn generate_dense_cpu(llama: &Llama, prompt: &[u32], max_new: usize) -> AResult<Vec<u32>> {
    let c = &llama.cfg;
    if c.moe.is_some() || c.gemma4 {
        return Err(anyhow!(
            "cpu runner: dense Qwen3/Llama/Gemma3 only (no MoE/gemma4 yet)"
        ));
    }
    let be = CpuBackend::new();
    let (ne, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_kv, c.head_dim);
    let kvrow = nkv * hd;
    let qrow = nh * hd;
    let nff = c.n_ff;
    let gemma = c.gemma;
    let qk_norm = c.qk_norm;
    let act = if gemma {
        Activation::Gelu
    } else {
        Activation::Silu
    };
    let max_ctx = prompt.len() + max_new + 1;

    // ── upload weights (all pre-dequantized to f32 for correctness-first). The order here MUST
    //    match the `g.weight()` declaration order in `build` below. ───────────────────────────────
    let mut wbufs: Vec<Box<dyn Buffer>> = Vec::new();
    let mut wf32 = |name: &str| -> AResult<()> {
        let (v, _) = load_tensor_dequant(&llama.gguf, name)?;
        let b = be
            .alloc(v.len() * 4, BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(b.as_ref(), bytemuck::cast_slice(&v))
            .map_err(|e| anyhow!("{e}"))?;
        wbufs.push(b);
        Ok(())
    };
    for l in 0..c.n_layer {
        let p = |s: &str| format!("blk.{l}.{s}");
        wf32(&p("attn_norm.weight"))?;
        wf32(&p("attn_q.weight"))?;
        wf32(&p("attn_k.weight"))?;
        wf32(&p("attn_v.weight"))?;
        if qk_norm {
            wf32(&p("attn_q_norm.weight"))?;
            wf32(&p("attn_k_norm.weight"))?;
        }
        wf32(&p("attn_output.weight"))?;
        if gemma {
            wf32(&p("post_attention_norm.weight"))?;
        }
        wf32(&p("ffn_norm.weight"))?;
        wf32(&p("ffn_gate.weight"))?;
        wf32(&p("ffn_up.weight"))?;
        wf32(&p("ffn_down.weight"))?;
        if gemma {
            wf32(&p("post_ffw_norm.weight"))?;
        }
    }
    // Globals: output_norm, lm_head (output.weight, or tied to token_embd f32).
    wf32("output_norm.weight")?;
    if llama
        .gguf
        .tensors()
        .iter()
        .any(|t| t.name == "output.weight")
    {
        wf32("output.weight")?;
    } else {
        let b = be
            .alloc(llama.token_embd.len() * 4, BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(b.as_ref(), bytemuck::cast_slice(&llama.token_embd))
            .map_err(|e| anyhow!("{e}"))?;
        wbufs.push(b);
    }

    // ── persistent KV cache buffers (f32) ──────────────────────────────────────────
    let mut kbufs: Vec<Box<dyn Buffer>> = Vec::new();
    let mut vbufs: Vec<Box<dyn Buffer>> = Vec::new();
    for _ in 0..c.n_layer {
        kbufs.push(
            be.alloc(max_ctx * kvrow * 4, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?,
        );
        vbufs.push(
            be.alloc(max_ctx * kvrow * 4, BufferUsage::Activations)
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
    let logits_buf = be
        .alloc(c.vocab * 4, BufferUsage::Readback)
        .map_err(|e| anyhow!("{e}"))?;

    // Build the decode graph for a given absolute position (kv_len = pos+1).
    let build = |pos: usize| -> (Graph, DecodeHandles) {
        let mut g = Graph::new();
        let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
        let hidden = g.input(f32d(ne));
        let positions = g.input(TensorDesc::new(vec![1], DType::I32));
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        for _ in 0..c.n_layer {
            k_cache.push(g.input(f32d(max_ctx * kvrow)));
            v_cache.push(g.input(f32d(max_ctx * kvrow)));
        }

        // Weights — declared in the SAME order as the upload loop. `wpush` records each handle in
        // the flat `weights` list (for binding) while we also keep the named handle.
        let mut weights: Vec<TensorId> = Vec::new();
        let mut lw: Vec<LayerW> = Vec::new();
        for _ in 0..c.n_layer {
            let mut wpush = |g: &mut Graph, n: usize| {
                let id = g.weight(f32d(n));
                weights.push(id);
                id
            };
            let attn_norm = wpush(&mut g, ne);
            let wq = wpush(&mut g, qrow * ne);
            let wk = wpush(&mut g, kvrow * ne);
            let wv = wpush(&mut g, kvrow * ne);
            let (q_norm, k_norm) = if qk_norm {
                (Some(wpush(&mut g, hd)), Some(wpush(&mut g, hd)))
            } else {
                (None, None)
            };
            let wo = wpush(&mut g, ne * qrow);
            let post_attn = if gemma { Some(wpush(&mut g, ne)) } else { None };
            let ffn_norm = wpush(&mut g, ne);
            let wgate = wpush(&mut g, nff * ne);
            let wup = wpush(&mut g, nff * ne);
            let wdown = wpush(&mut g, ne * nff);
            let post_ffw = if gemma { Some(wpush(&mut g, ne)) } else { None };
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
                wgate,
                wup,
                wdown,
                post_ffw,
            });
        }
        let w_out_norm = {
            let id = g.weight(f32d(ne));
            weights.push(id);
            id
        };
        let w_lm = {
            let id = g.weight(f32d(c.vocab * ne));
            weights.push(id);
            id
        };
        let logits = g.output(f32d(c.vocab));

        // scratch
        let hn = g.internal(f32d(ne));
        let q = g.internal(f32d(qrow));
        let k = g.internal(f32d(kvrow));
        let v = g.internal(f32d(kvrow));
        let attn = g.internal(f32d(qrow));
        let gbuf = g.internal(f32d(nff));
        let ubuf = g.internal(f32d(nff));
        let actbuf = g.internal(f32d(nff));
        let sub = g.internal(f32d(ne));

        let eps = c.rms_eps;
        // gemma3 uses 1/√hd like Qwen; gemma4 (1.0) is rejected above.
        let scale = 1.0 / (hd as f32).sqrt();
        for (l, lw) in lw.iter().enumerate() {
            let theta = c.layer_rope_theta(l); // gemma dual-rope (SWA 1e4 / full 1e6); uniform else
            let mask = if gemma && c.is_swa_layer(l) {
                AttnMask::SlidingWindow(c.swa_window)
            } else {
                AttnMask::Causal
            };
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
            g.push(Op::Linear {
                x: hn,
                weight: lw.wk,
                dst: k,
                m: 1,
                in_f: ne as u32,
                out_f: kvrow as u32,
            });
            g.push(Op::Linear {
                x: hn,
                weight: lw.wv,
                dst: v,
                m: 1,
                in_f: ne as u32,
                out_f: kvrow as u32,
            });
            if let (Some(qn), Some(kn)) = (lw.q_norm, lw.k_norm) {
                g.push(Op::QkNorm {
                    x: q,
                    weight: qn,
                    dst: q,
                    rows: 1,
                    n_head: nh as u32,
                    head_dim: hd as u32,
                    eps,
                });
                g.push(Op::QkNorm {
                    x: k,
                    weight: kn,
                    dst: k,
                    rows: 1,
                    n_head: nkv as u32,
                    head_dim: hd as u32,
                    eps,
                });
            }
            g.push(Op::Rope {
                x: q,
                positions,
                dst: q,
                rows: 1,
                n_head: nh as u32,
                head_dim: hd as u32,
                rope_dim: c.rope_dim as u32,
                theta,
                freq_factors: None,
            });
            g.push(Op::Rope {
                x: k,
                positions,
                dst: k,
                rows: 1,
                n_head: nkv as u32,
                head_dim: hd as u32,
                rope_dim: c.rope_dim as u32,
                theta,
                freq_factors: None,
            });
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
            g.push(Op::Attention {
                q,
                k_cache: k_cache[l],
                v_cache: v_cache[l],
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
            g.push(Op::Linear {
                x: hn,
                weight: lw.wgate,
                dst: gbuf,
                m: 1,
                in_f: ne as u32,
                out_f: nff as u32,
            });
            g.push(Op::Linear {
                x: hn,
                weight: lw.wup,
                dst: ubuf,
                m: 1,
                in_f: ne as u32,
                out_f: nff as u32,
            });
            g.push(Op::GatedAct {
                gate: gbuf,
                up: ubuf,
                dst: actbuf,
                rows: 1,
                nff: nff as u32,
                act,
                up_off: 0,
            });
            g.push(Op::Linear {
                x: actbuf,
                weight: lw.wdown,
                dst: sub,
                m: 1,
                in_f: nff as u32,
                out_f: ne as u32,
            });
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
    for pos in 0..(prompt.len() + max_new) {
        let tok = cur[pos] as usize;
        // embed (gemma scales by √n_embd; qwen3/llama identity)
        let emb: Vec<f32> = llama.token_embd[tok * ne..tok * ne + ne]
            .iter()
            .map(|&x| x * embed_scale)
            .collect();
        be.upload(hidden_buf.as_ref(), bytemuck::cast_slice(&emb))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(pos_buf.as_ref(), bytemuck::cast_slice(&[pos as i32]))
            .map_err(|e| anyhow!("{e}"))?;

        let (g, h) = build(pos);
        let plan = be.compile(&g).map_err(|e| anyhow!("{e}"))?;
        let mut b = Bindings::new();
        b.bind(h.hidden, hidden_buf.as_ref());
        b.bind(h.positions, pos_buf.as_ref());
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
        if pos + 1 >= prompt.len() {
            be.download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
                .map_err(|e| anyhow!("{e}"))?;
            let next = argmax(&logits) as u32;
            out.push(next);
            if c.eos_ids.contains(&next) || next == c.eos || out.len() >= max_new {
                break;
            }
            if cur.len() <= pos + 1 {
                cur.push(next);
            }
        }
    }
    Ok(out)
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
