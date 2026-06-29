//! Qwen3.5 / Qwen3.6 (`qwen35`, aka Qwen3-Next): hybrid gated-DeltaNet linear-attention + gated
//! full-attention. See `docs/QWEN35.md`.
//!
//! **Hybrid GPU execution (the general infr pattern):** infr has no automatic graph scheduler —
//! every model's forward is hand-written eager code. So "GPU where a kernel exists, CPU where it
//! doesn't" is expressed directly: the heavy linear projections run through `VulkanBackend::linear`
//! (see [`Lin`]), while the ops with no GPU kernel — the SSM depthwise conv, the gated-delta
//! recurrence, and the hd=256 gated full-attention — stay on the CPU. When we later add those GPU
//! kernels, each CPU block is swapped for a `be.<op>` call with no structural change. Set
//! `Q35_CPU=1` to force the pure-CPU path (the correctness oracle, no Vulkan init).
#![allow(dead_code)] // forward pass is built up incrementally on this loader

use crate::cpu_backend::CpuBackend;
use crate::load_tensor_dequant;
use anyhow::{anyhow, bail, Context, Result};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::tensor::TensorDesc;
use infr_core::{DType, TensorId, WeightSource};
use infr_gguf::Gguf;
use infr_vulkan::VulkanBackend;

/// GPU weight storage dtype. Chosen from the GGUF source dtype by the shared loader policy
/// ([`Model::upload_lin`]): quant/F16 → `F16` (dequant once, half the bandwidth — decode is
/// bandwidth-bound); BF16 → `Bf16` (native, preserves f32 exponent range — no f16 overflow clip);
/// F32 → `F32` (full precision, the rare large-magnitude tensor). Each maps to a GPU GEMV kernel.
#[derive(Clone, Copy)]
enum WDtype {
    F16,
    Bf16,
    F32,
}

/// A linear-projection weight on whichever device has a kernel for it: GPU (dtype-tagged, the fast
/// path) or CPU f32 (the fallback / `Q35_CPU=1` oracle). One call site, [`Lin::mul`], hides which.
/// This is how a hand-written forward gets per-op CPU fallback without a graph scheduler.
enum Lin {
    Cpu(Vec<f32>),
    Gpu { buf: Box<dyn Buffer>, dt: WDtype },
}

impl Lin {
    /// `y[out] = W[out,in] · x[in]` (single row). GPU path is a GEMV in the weight's dtype; CPU path
    /// the naive matvec.
    fn mul(&self, be: Option<&VulkanBackend>, x: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
        match self {
            Lin::Cpu(w) => matvec(w, in_f, out_f, x),
            Lin::Gpu { buf, dt } => {
                let be = be.expect("gpu Lin without backend");
                let w = buf.as_ref();
                match dt {
                    WDtype::F16 => be.linear_f16(w, x, 1, in_f, out_f),
                    WDtype::Bf16 => be.linear_bf16(w, x, 1, in_f, out_f),
                    WDtype::F32 => be.linear(w, x, 1, in_f, out_f),
                }
                .expect("gpu linear")
            }
        }
    }
}

/// Parsed `qwen35` hyper-parameters (subset needed for the 0.8B dense model).
#[derive(Debug, Clone)]
pub struct Cfg {
    pub n_layer: usize,
    pub n_embd: usize,
    pub vocab: usize,
    pub eps: f32,
    // attention layers
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize, // key_length == value_length (256)
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rope_sections: [u32; 4],
    pub full_attn_interval: usize,
    // linear (gated DeltaNet) layers
    pub d_conv: usize,  // ssm conv kernel (4)
    pub d_state: usize, // head_k_dim (128)
    pub d_inner: usize, // value_dim (2048)
    pub n_group: usize, // num_k_heads (16)
    pub dt_rank: usize, // num_v_heads (16)
}

impl Cfg {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.metadata().str("general.architecture").unwrap_or("");
        if arch != "qwen35" {
            bail!("not a qwen35 model (arch={arch:?})");
        }
        let u = |k: &str| g.metadata().u64(&format!("qwen35.{k}"));
        let req = |k: &str| u(k).ok_or_else(|| anyhow!("missing qwen35.{k}"));
        let f = |k: &str| -> Option<f32> {
            g.metadata()
                .get(&format!("qwen35.{k}"))
                .and_then(|v| match v {
                    infr_core::MetaValue::F64(x) => Some(*x as f32),
                    infr_core::MetaValue::U64(x) => Some(*x as f32),
                    infr_core::MetaValue::I64(x) => Some(*x as f32),
                    _ => None,
                })
        };
        // rope.dimension_sections is an array [11,11,10,0]
        let sections: [u32; 4] = {
            let mut s = [0u32; 4];
            if let Some(arr) = g
                .metadata()
                .get("qwen35.rope.dimension_sections")
                .and_then(|v| v.as_arr())
            {
                for (i, v) in arr.iter().take(4).enumerate() {
                    s[i] = v.as_u64().unwrap_or(0) as u32;
                }
            }
            s
        };
        Ok(Cfg {
            n_layer: req("block_count")? as usize,
            n_embd: req("embedding_length")? as usize,
            vocab: 0, // filled from token_embd shape
            eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            n_head: req("attention.head_count")? as usize,
            n_kv: req("attention.head_count_kv")? as usize,
            head_dim: req("attention.key_length")? as usize,
            rope_dim: u("rope.dimension_count").unwrap_or(64) as usize,
            rope_theta: f("rope.freq_base").unwrap_or(1e7),
            rope_sections: sections,
            full_attn_interval: u("full_attention_interval").unwrap_or(4) as usize,
            d_conv: req("ssm.conv_kernel")? as usize,
            d_state: req("ssm.state_size")? as usize,
            d_inner: req("ssm.inner_size")? as usize,
            n_group: req("ssm.group_count")? as usize,
            dt_rank: req("ssm.time_step_rank")? as usize,
        })
    }

    /// Attention (vs linear/SSM) layer test: every `full_attn_interval`-th layer is full attention.
    pub fn is_attn_layer(&self, i: usize) -> bool {
        (i + 1).is_multiple_of(self.full_attn_interval)
    }
    pub fn num_k_heads(&self) -> usize {
        self.n_group
    }
    pub fn num_v_heads(&self) -> usize {
        self.dt_rank
    }
    pub fn head_k_dim(&self) -> usize {
        self.d_state
    }
    pub fn head_v_dim(&self) -> usize {
        self.d_inner / self.dt_rank
    }
    pub fn conv_channels(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }
}

/// A linear (gated DeltaNet) layer's weights. Big matmuls are [`Lin`] (GPU/CPU); the small SSM
/// tensors stay CPU f32 (no GPU kernel — they feed the recurrence, which runs on the CPU).
struct LinearLayer {
    attn_norm: Vec<f32>, // [n_embd]
    qkv: Lin,            // [conv_channels, n_embd]  (out,in)
    gate: Lin,           // [d_inner, n_embd]  (z)
    conv1d: Vec<f32>,    // [conv_channels, d_conv]  (per-channel kernel)
    alpha: Vec<f32>,     // [dt_rank, n_embd]
    beta: Vec<f32>,      // [dt_rank, n_embd]
    a: Vec<f32>,         // [dt_rank]  (= -exp(A_log))
    dt_bias: Vec<f32>,   // [dt_rank]
    ssm_norm: Vec<f32>,  // [head_v_dim]
    out: Lin,            // [n_embd, d_inner]
    post_norm: Vec<f32>, // [n_embd]
    ffn_gate: Lin,       // [n_ff, n_embd]
    ffn_up: Lin,         // [n_ff, n_embd]
    ffn_down: Lin,       // [n_embd, n_ff]
    n_ff: usize,
}

/// A full-attention layer's weights. Projections are [`Lin`]; q/k norms stay CPU (per-head, hd=256).
struct AttnLayer {
    attn_norm: Vec<f32>, // [n_embd]
    q: Lin,              // [n_head*head_dim + d_inner(gate), n_embd]
    k: Lin,              // [n_kv*head_dim, n_embd]
    v: Lin,              // [n_kv*head_dim, n_embd]
    q_norm: Vec<f32>,    // [head_dim]
    k_norm: Vec<f32>,    // [head_dim]
    out: Lin,            // [n_embd, n_head*head_dim]
    post_norm: Vec<f32>,
    ffn_gate: Lin,
    ffn_up: Lin,
    ffn_down: Lin,
    n_ff: usize,
}

enum Layer {
    Linear(LinearLayer),
    Attn(AttnLayer),
}

/// Full model weights. Linear projections live on the GPU (f16) unless `Q35_CPU=1`; the SSM /
/// attention math stays on the CPU either way (see module docs).
pub struct Model {
    pub cfg: Cfg,
    token_embd: Vec<f32>, // [vocab, n_embd] host (embedding gather)
    output_norm: Vec<f32>,
    lm_head: Lin, // [vocab, n_embd]
    layers: Vec<Layer>,
    be: Option<VulkanBackend>,
}

impl Model {
    /// Shared loader policy: map a GGUF projection tensor to a GPU [`Lin`] in the dtype matching its
    /// source, or CPU f32 when `be` is `None` (`Q35_CPU=1`). quant/F16 → f16 (dequant once); BF16 →
    /// native bf16 (range-preserving, round-trips exactly through the f32 host buffer); F32 → f32.
    /// This is the single place the "correct dtype per source" decision lives.
    fn upload_lin(g: &Gguf, be: Option<&VulkanBackend>, name: &str) -> Result<Lin> {
        let w = load_tensor_dequant(g, name)
            .map(|x| x.0)
            .with_context(|| name.to_string())?;
        let be = match be {
            Some(b) => b,
            None => return Ok(Lin::Cpu(w)),
        };
        let src = g.tensors().iter().find(|t| t.name == name).map(|t| t.dtype);
        let up =
            |r: infr_core::Result<Box<dyn Buffer>>| r.map_err(|e| anyhow!("upload {name}: {e}"));
        let (buf, dt) = match src {
            Some(DType::Bf16) => (up(be.upload_weight_bf16(&w))?, WDtype::Bf16),
            Some(DType::F32) => (up(be.upload_weight(&w))?, WDtype::F32),
            _ => (up(be.upload_weight_f16(&w))?, WDtype::F16), // quant, F16, others → f16
        };
        Ok(Lin::Gpu { buf, dt })
    }

    pub fn load(g: &Gguf) -> Result<Self> {
        let mut cfg = Cfg::from_gguf(g)?;
        // CPU-only oracle skips Vulkan entirely; otherwise upload the matmul weights to the GPU.
        let be = if std::env::var("Q35_CPU").is_ok() {
            None
        } else {
            Some(VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?)
        };
        let (token_embd, te_shape) = load_tensor_dequant(g, "token_embd.weight")?;
        cfg.vocab = te_shape[1];
        let output_norm = load_tensor_dequant(g, "output_norm.weight")?.0;

        // Shared dtype policy: load a projection weight to a GPU `Lin` in the dtype matching its
        // GGUF source (see `upload_lin`), or keep it CPU f32 (`Q35_CPU=1`).
        let lin = |name: &str| -> Result<Lin> { Self::upload_lin(g, be.as_ref(), name) };
        let lm_head = if g.tensors().iter().any(|t| t.name == "output.weight") {
            lin("output.weight")?
        } else {
            // tied embeddings: lm head = token_embd (already dequantized to f32 for the host gather)
            match &be {
                Some(be) => Lin::Gpu {
                    buf: be
                        .upload_weight_f16(&token_embd)
                        .map_err(|e| anyhow!("upload lm_head: {e}"))?,
                    dt: WDtype::F16,
                },
                None => Lin::Cpu(token_embd.clone()),
            }
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let get = |s: &str| -> Result<Vec<f32>> {
                load_tensor_dequant(g, &p(s))
                    .map(|x| x.0)
                    .with_context(|| p(s))
            };
            let glin = |s: &str| -> Result<Lin> { lin(&p(s)) };
            let ffn_up_shape = g
                .tensors()
                .iter()
                .find(|t| t.name == p("ffn_up.weight"))
                .map(|t| t.shape.clone())
                .context("ffn_up")?;
            let n_ff = ffn_up_shape[1];
            if cfg.is_attn_layer(i) {
                layers.push(Layer::Attn(AttnLayer {
                    attn_norm: get("attn_norm.weight")?,
                    q: glin("attn_q.weight")?,
                    k: glin("attn_k.weight")?,
                    v: glin("attn_v.weight")?,
                    q_norm: get("attn_q_norm.weight")?,
                    k_norm: get("attn_k_norm.weight")?,
                    out: glin("attn_output.weight")?,
                    post_norm: get("post_attention_norm.weight")?,
                    ffn_gate: glin("ffn_gate.weight")?,
                    ffn_up: glin("ffn_up.weight")?,
                    ffn_down: glin("ffn_down.weight")?,
                    n_ff,
                }));
            } else {
                layers.push(Layer::Linear(LinearLayer {
                    attn_norm: get("attn_norm.weight")?,
                    qkv: glin("attn_qkv.weight")?,
                    gate: glin("attn_gate.weight")?,
                    conv1d: get("ssm_conv1d.weight")?,
                    alpha: get("ssm_alpha.weight")?,
                    beta: get("ssm_beta.weight")?,
                    a: get("ssm_a")?,
                    dt_bias: get("ssm_dt.bias")?,
                    ssm_norm: get("ssm_norm.weight")?,
                    out: glin("ssm_out.weight")?,
                    post_norm: get("post_attention_norm.weight")?,
                    ffn_gate: glin("ffn_gate.weight")?,
                    ffn_up: glin("ffn_up.weight")?,
                    ffn_down: glin("ffn_down.weight")?,
                    n_ff,
                }));
            }
        }
        Ok(Model {
            cfg,
            token_embd,
            output_norm,
            lm_head,
            layers,
            be,
        })
    }
}

// ── math helpers (CPU, f32) ─────────────────────────────────────────────────

/// `y[o] = Σ_j W[o*in+j] * x[j]`  (W is row-major [out, in], the ggml weight layout).
fn matvec(w: &[f32], in_f: usize, out_f: usize, x: &[f32]) -> Vec<f32> {
    (0..out_f)
        .map(|o| {
            let row = &w[o * in_f..o * in_f + in_f];
            row.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}
fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * s * g).collect()
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    // numerically stable ln(1+e^x)
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
fn l2norm(v: &mut [f32], eps: f32) {
    let n = (v.iter().map(|x| x * x).sum::<f32>() + eps).sqrt();
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// Per-layer recurrent state for the CPU reference.
enum LayerState {
    Linear {
        conv: Vec<f32>, // [d_conv-1, conv_channels] rolling history (oldest first)
        s: Vec<f32>,    // [num_v_heads, head_k_dim, head_v_dim]
    },
    Attn {
        k: Vec<f32>, // [pos, n_kv*head_dim]
        v: Vec<f32>,
    },
}

pub struct State {
    layers: Vec<LayerState>,
    pos: usize,
}

impl Model {
    pub fn new_state(&self) -> State {
        let c = &self.cfg;
        let layers = (0..c.n_layer)
            .map(|i| {
                if c.is_attn_layer(i) {
                    LayerState::Attn {
                        k: vec![],
                        v: vec![],
                    }
                } else {
                    LayerState::Linear {
                        conv: vec![0.0; (c.d_conv - 1) * c.conv_channels()],
                        s: vec![0.0; c.num_v_heads() * c.head_k_dim() * c.head_v_dim()],
                    }
                }
            })
            .collect();
        State { layers, pos: 0 }
    }

    /// One token through the whole stack; returns logits over the vocab. `st` carries the recurrent
    /// conv/SSM state and the attention KV cache across calls.
    pub fn forward(&self, token: u32, st: &mut State) -> Vec<f32> {
        let c = &self.cfg;
        let ne = c.n_embd;
        let mut hidden = self.token_embd[token as usize * ne..(token as usize + 1) * ne].to_vec();
        let pos = st.pos;
        for (li, layer) in self.layers.iter().enumerate() {
            match (layer, &mut st.layers[li]) {
                (Layer::Linear(w), LayerState::Linear { conv, s }) => {
                    let y = self.linear_mixer(w, &hidden, conv, s);
                    if std::env::var("Q35_NOLIN").is_err() {
                        for (h, yi) in hidden.iter_mut().zip(&y) {
                            *h += yi;
                        }
                    }
                    let d = self.ffn(
                        &hidden,
                        &w.post_norm,
                        &w.ffn_gate,
                        &w.ffn_up,
                        &w.ffn_down,
                        w.n_ff,
                    );
                    for (h, di) in hidden.iter_mut().zip(&d) {
                        *h += di;
                    }
                }
                (Layer::Attn(w), LayerState::Attn { k, v }) => {
                    let y = self.attn_mixer(w, &hidden, k, v, pos);
                    if std::env::var("Q35_NOATTN").is_err() {
                        for (h, yi) in hidden.iter_mut().zip(&y) {
                            *h += yi;
                        }
                    }
                    let d = self.ffn(
                        &hidden,
                        &w.post_norm,
                        &w.ffn_gate,
                        &w.ffn_up,
                        &w.ffn_down,
                        w.n_ff,
                    );
                    for (h, di) in hidden.iter_mut().zip(&d) {
                        *h += di;
                    }
                }
                _ => unreachable!("layer/state kind mismatch"),
            }
            if std::env::var("Q35_DBG").is_ok() {
                let kind = if c.is_attn_layer(li) { "attn" } else { "lin " };
                let nrm = (hidden.iter().map(|x| x * x).sum::<f32>() / ne as f32).sqrt();
                let fin = hidden.iter().all(|x| x.is_finite());
                eprintln!("  L{li:02} {kind} rms={nrm:.4} finite={fin}");
            }
        }
        st.pos += 1;
        // final norm + lm head (only this token)
        let hn = rmsnorm(&hidden, &self.output_norm, c.eps);
        self.lm_head.mul(self.be.as_ref(), &hn, ne, c.vocab)
    }

    fn ffn(
        &self,
        hidden: &[f32],
        norm: &[f32],
        gate: &Lin,
        up: &Lin,
        down: &Lin,
        n_ff: usize,
    ) -> Vec<f32> {
        let ne = self.cfg.n_embd;
        let be = self.be.as_ref();
        let h2 = rmsnorm(hidden, norm, self.cfg.eps);
        let g = gate.mul(be, &h2, ne, n_ff);
        let u = up.mul(be, &h2, ne, n_ff);
        let act: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
        down.mul(be, &act, n_ff, ne)
    }

    /// Gated DeltaNet linear-attention mixer (one token).
    fn linear_mixer(
        &self,
        w: &LinearLayer,
        hidden: &[f32],
        conv: &mut [f32],
        s: &mut [f32],
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (ne, kd, vd) = (c.n_embd, c.head_k_dim(), c.head_v_dim());
        let (nk, nv) = (c.num_k_heads(), c.num_v_heads());
        let cc = c.conv_channels();
        let be = self.be.as_ref();
        let xn = rmsnorm(hidden, &w.attn_norm, c.eps);
        let qkv = w.qkv.mul(be, &xn, ne, cc); // [6144]  GPU
        let z = w.gate.mul(be, &xn, ne, c.d_inner); // [2048]  GPU

        // causal depthwise conv over the cc channels: out[ch] = Σ_k tap_k[ch]*weight[ch*d_conv+k]
        // taps oldest→newest; window = [conv history.., current]
        let k_conv = c.d_conv;
        let mut conv_out = vec![0.0f32; cc];
        for ch in 0..cc {
            let mut acc = 0.0;
            for k in 0..k_conv - 1 {
                acc += conv[k * cc + ch] * w.conv1d[ch * k_conv + k];
            }
            acc += qkv[ch] * w.conv1d[ch * k_conv + (k_conv - 1)];
            conv_out[ch] = silu(acc);
        }
        // shift conv history (drop oldest, append current raw qkv)
        for k in 0..k_conv - 2 {
            for ch in 0..cc {
                conv[k * cc + ch] = conv[(k + 1) * cc + ch];
            }
        }
        for ch in 0..cc {
            conv[(k_conv - 2) * cc + ch] = qkv[ch];
        }

        // split conv_out → q,k,v
        let key_dim = nk * kd;
        let (q_all, rest) = conv_out.split_at(key_dim);
        let (k_all, v_all) = rest.split_at(key_dim);

        // beta / decay gates (per v-head)
        let b = matvec(&w.beta, ne, nv, &xn);
        let a = matvec(&w.alpha, ne, nv, &xn);

        let mut out = vec![0.0f32; nv * vd];
        let qscale = 1.0 / (kd as f32).sqrt();
        for h in 0..nv {
            // num_v_heads == num_k_heads here (1:1)
            let mut qh = q_all[h * kd..h * kd + kd].to_vec();
            let mut kh = k_all[h * kd..h * kd + kd].to_vec();
            let vh = &v_all[h * vd..h * vd + vd];
            l2norm(&mut qh, 1e-6);
            l2norm(&mut kh, 1e-6);
            for x in qh.iter_mut() {
                *x *= qscale;
            }
            let beta = sigmoid(b[h]);
            let g = w.a[h] * softplus(a[h] + w.dt_bias[h]); // ≤ 0
            let decay = g.exp();
            let sh = &mut s[h * kd * vd..(h + 1) * kd * vd]; // [kd, vd]
                                                             // S *= decay
            for x in sh.iter_mut() {
                *x *= decay;
            }
            // kv = kᵀS  [vd]
            let mut kv = vec![0.0f32; vd];
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

        // silu-gated RMSNorm per v-head, gate = z
        for h in 0..nv {
            let oh = &mut out[h * vd..h * vd + vd];
            let n = rmsnorm(oh, &w.ssm_norm, c.eps);
            let zh = &z[h * vd..h * vd + vd];
            for d in 0..vd {
                oh[d] = n[d] * silu(zh[d]);
            }
        }
        w.out.mul(be, &out, c.d_inner, ne) // GPU
    }

    /// Gated full attention (one token), GQA, head_dim 256, partial sectioned RoPE, sigmoid out-gate.
    fn attn_mixer(
        &self,
        w: &AttnLayer,
        hidden: &[f32],
        kc: &mut Vec<f32>,
        vc: &mut Vec<f32>,
        pos: usize,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (ne, hd) = (c.n_embd, c.head_dim);
        let (nh, nkv) = (c.n_head, c.n_kv);
        let be = self.be.as_ref();
        let xn = rmsnorm(hidden, &w.attn_norm, c.eps);
        // attn_q outputs query+gate INTERLEAVED PER HEAD: [h0 q(hd) h0 gate(hd) | h1 q gate | …].
        let qg = w.q.mul(be, &xn, ne, nh * 2 * hd); // GPU
        let mut q = vec![0f32; nh * hd];
        let mut gate = vec![0f32; nh * hd];
        for h in 0..nh {
            q[h * hd..h * hd + hd].copy_from_slice(&qg[h * 2 * hd..h * 2 * hd + hd]);
            gate[h * hd..h * hd + hd].copy_from_slice(&qg[h * 2 * hd + hd..h * 2 * hd + 2 * hd]);
        }
        let mut k = w.k.mul(be, &xn, ne, nkv * hd); // GPU
        let v = w.v.mul(be, &xn, ne, nkv * hd); // GPU
                                                // per-head q/k norm then RoPE
        for h in 0..nh {
            let qh = &mut q[h * hd..h * hd + hd];
            let nq = rmsnorm(qh, &w.q_norm, c.eps);
            qh.copy_from_slice(&nq);
            rope(qh, pos, c.rope_dim, c.rope_theta);
        }
        for h in 0..nkv {
            let kh = &mut k[h * hd..h * hd + hd];
            let nk = rmsnorm(kh, &w.k_norm, c.eps);
            kh.copy_from_slice(&nk);
            rope(kh, pos, c.rope_dim, c.rope_theta);
        }
        kc.extend_from_slice(&k);
        vc.extend_from_slice(&v);
        let t = pos + 1; // cached length
        let scale = 1.0 / (hd as f32).sqrt();
        let g = nh / nkv;
        let mut out = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let kvh = h / g;
            let qh = &q[h * hd..h * hd + hd];
            let mut scores = vec![0.0f32; t];
            for j in 0..t {
                let kj = &kc[j * nkv * hd + kvh * hd..j * nkv * hd + kvh * hd + hd];
                scores[j] = qh.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let mut den = 0.0;
            for sj in scores.iter_mut() {
                *sj = (*sj - m).exp();
                den += *sj;
            }
            let oh = &mut out[h * hd..h * hd + hd];
            for j in 0..t {
                let p = scores[j] / den;
                let vj = &vc[j * nkv * hd + kvh * hd..j * nkv * hd + kvh * hd + hd];
                for d in 0..hd {
                    oh[d] += p * vj[d];
                }
            }
            // per-head sigmoid output gate
            let gh = &gate[h * hd..h * hd + hd];
            for d in 0..hd {
                oh[d] *= sigmoid(gh[d]);
            }
        }
        w.out.mul(be, &out, nh * hd, ne) // GPU
    }
}

/// Partial NEOX-style RoPE over the first `rope_dim` dims (rest pass through). Text-only sectioned
/// RoPE reduces to standard RoPE since all position components equal the token position.
fn rope(x: &mut [f32], pos: usize, rope_dim: usize, theta: f32) {
    let half = rope_dim / 2;
    for i in 0..half {
        let freq = theta.powf(-2.0 * i as f32 / rope_dim as f32);
        let ang = pos as f32 * freq;
        let (s, co) = (ang.sin(), ang.cos());
        let a = x[i];
        let b = x[i + half];
        x[i] = a * co - b * s;
        x[i + half] = a * s + b * co;
    }
}

/// Greedy-generate `n` tokens from `prompt` (raw, no chat template) for CPU-reference validation.
pub fn generate(g: &Gguf, prompt: &str, n: usize) -> Result<String> {
    let m = Model::load(g)?;
    let tok = crate::build_tokenizer(g)?;
    let enc = tok
        .encode(prompt, false)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let ids = enc.get_ids();
    let mut st = m.new_state();
    let mut last = 0u32;
    for (i, &id) in ids.iter().enumerate() {
        let logits = m.forward(id, &mut st);
        if i == ids.len() - 1 {
            last = argmax(&logits);
        }
    }
    let mut outs = vec![last];
    for _ in 1..n {
        let logits = m.forward(last, &mut st);
        last = argmax(&logits);
        outs.push(last);
    }
    tok.decode(&outs, false).map_err(|e| anyhow!("decode: {e}"))
}

// ─── qwen35 pure-CPU runner on the backend-agnostic seam ─────────────────────────
//
// Builds an n=1 decode `Graph` (composite ops over typed handles) and runs it through `CpuBackend`
// — no Vulkan, no GPU. The gated-DeltaNet recurrence + depthwise conv state and the attention KV
// cache are model-owned `Input` buffers, mutated in place each step (exactly like the dense runner's
// KV cache). Validates the seam's qwen35 ops end-to-end against the bespoke CPU oracle.

/// Per-layer weight + state handles into one decode graph (declared in upload order so each binds to
/// the matching uploaded buffer).
struct Q35LinH {
    attn_norm: TensorId,
    qkv: TensorId,
    gate: TensorId,
    conv1d: TensorId,
    alpha: TensorId,
    beta: TensorId,
    ssm_a: TensorId,
    dt_bias: TensorId,
    ssm_norm: TensorId,
    out: TensorId,
    post_norm: TensorId,
    ffn_gate: TensorId,
    ffn_up: TensorId,
    ffn_down: TensorId,
    n_ff: usize,
    conv_state: TensorId,
    s_state: TensorId,
}
struct Q35AttnH {
    attn_norm: TensorId,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_norm: TensorId,
    k_norm: TensorId,
    out: TensorId,
    post_norm: TensorId,
    ffn_gate: TensorId,
    ffn_up: TensorId,
    ffn_down: TensorId,
    n_ff: usize,
    k_cache: TensorId,
    v_cache: TensorId,
}
enum Q35LayerH {
    Lin(Q35LinH),
    Attn(Q35AttnH),
}

/// Render a plain user message through the qwen35 GGUF's own jinja chat template (falls back to
/// ChatML — qwen35's native format — if there's no template). So `infr run` / tests pass plain text.
pub fn render_chat(path: &std::path::Path, user: &str) -> Result<String> {
    let g = Gguf::open(path).map_err(|e| anyhow!("open gguf: {e}"))?;
    let tok = crate::build_tokenizer(&g)?;
    let eos = g.metadata().u64("tokenizer.ggml.eos_token_id").unwrap_or(2) as u32;
    Ok(crate::render_chat_user(&g, &tok, eos, user)
        .unwrap_or_else(|| format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")))
}

/// Greedy pure-CPU generation for qwen35 / Qwen3-Next on the agnostic seam (no Vulkan). `prompt` is
/// the already-formatted text (see [`render_chat`]); returns timing/counts, text streams via `on_piece`.
pub fn generate_cpu(
    path: &std::path::Path,
    prompt: &str,
    n: usize,
    mut on_piece: impl FnMut(&str),
) -> Result<crate::cpu_backend::CpuStats> {
    let gg = Gguf::open(path).map_err(|e| anyhow!("open gguf: {e}"))?;
    let g = &gg;
    let c = Cfg::from_gguf(g)?;
    let (token_embd, te_shape) = load_tensor_dequant(g, "token_embd.weight")?;
    let vocab = te_shape[1];
    let ne = c.n_embd;
    let cc = c.conv_channels();
    let di = c.d_inner;
    let (nk, kd) = (c.num_k_heads(), c.head_k_dim());
    let (nv, vd) = (c.num_v_heads(), c.head_v_dim());
    let key_dim = nk * kd;
    let (nh, nkv, hd) = (c.n_head, c.n_kv, c.head_dim);
    let eps = c.eps;
    let kk = c.d_conv;
    let tok = crate::build_tokenizer(g)?;
    let enc = tok
        .encode(prompt, false)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    let max_ctx = prompt_ids.len() + n + 1;

    let be = CpuBackend::new();
    let attn = |i: usize| c.is_attn_layer(i);
    let n_ff_of = |i: usize| -> Result<usize> {
        Ok(g.tensors()
            .iter()
            .find(|t| t.name == format!("blk.{i}.ffn_up.weight"))
            .context("ffn_up")?
            .shape[1])
    };

    // ── upload weights in native GGUF dtype (the backend dequants on read). Order MUST equal the
    //    `wpush` order in `build`. ──────────────────────────────────────────────────────────────────
    let mut wbufs: Vec<Box<dyn Buffer>> = Vec::new();
    let mut wspecs: Vec<(DType, usize)> = Vec::new();
    let mut wraw = |name: &str| -> Result<()> {
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
    for i in 0..c.n_layer {
        let p = |s: &str| format!("blk.{i}.{s}");
        if attn(i) {
            for nm in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "attn_output.weight",
                "post_attention_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                wraw(&p(nm))?;
            }
        } else {
            for nm in [
                "attn_norm.weight",
                "attn_qkv.weight",
                "attn_gate.weight",
                "ssm_conv1d.weight",
                "ssm_alpha.weight",
                "ssm_beta.weight",
                "ssm_a",
                "ssm_dt.bias",
                "ssm_norm.weight",
                "ssm_out.weight",
                "post_attention_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                wraw(&p(nm))?;
            }
        }
    }
    wraw("output_norm.weight")?;
    // lm_head: `output.weight`, or (tied) the quantized `token_embd.weight` mapped zero-copy and
    // dequantized per-row by `Op::Linear`.
    if g.tensors().iter().any(|t| t.name == "output.weight") {
        wraw("output.weight")?;
    } else {
        wraw("token_embd.weight")?;
    }

    // ── persistent state buffers (f32, zero-init), one set per layer by kind ──────────────────────
    let mut conv_bufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
    let mut s_bufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
    let mut k_bufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
    let mut v_bufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
    for i in 0..c.n_layer {
        if attn(i) {
            k_bufs.push(Some(
                be.alloc(max_ctx * nkv * hd * 4, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            ));
            v_bufs.push(Some(
                be.alloc(max_ctx * nkv * hd * 4, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            ));
            conv_bufs.push(None);
            s_bufs.push(None);
        } else {
            conv_bufs.push(Some(
                be.alloc((kk - 1) * cc * 4, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            ));
            s_bufs.push(Some(
                be.alloc(nv * kd * vd * 4, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            ));
            k_bufs.push(None);
            v_bufs.push(None);
        }
    }

    // ── per-step IO ───────────────────────────────────────────────────────────────
    let hidden_buf = be
        .alloc(ne * 4, BufferUsage::Staging)
        .map_err(|e| anyhow!("{e}"))?;
    let pos_buf = be
        .alloc(4, BufferUsage::Staging)
        .map_err(|e| anyhow!("{e}"))?;
    let logits_buf = be
        .alloc(vocab * 4, BufferUsage::Readback)
        .map_err(|e| anyhow!("{e}"))?;

    let f32d = |x: usize| TensorDesc::new(vec![x], DType::F32);
    let scale = 1.0 / (hd as f32).sqrt();

    // Build the decode graph for absolute position `pos` (kv_len = pos+1).
    let build = |pos: usize| -> Result<(Graph, TensorId, TensorId, Vec<TensorId>, TensorId)> {
        let mut gr = Graph::new();
        let hidden = gr.input(f32d(ne));
        let positions = gr.input(TensorDesc::new(vec![1], DType::I32));
        // weights in upload order
        let mut weights: Vec<TensorId> = Vec::new();
        let mut wi = 0usize;
        let mut wpush = |gr: &mut Graph, weights: &mut Vec<TensorId>| -> TensorId {
            let (dt, num) = wspecs[wi];
            wi += 1;
            let id = gr.weight(TensorDesc::new(vec![num], dt));
            weights.push(id);
            id
        };
        let mut layers: Vec<Q35LayerH> = Vec::new();
        for i in 0..c.n_layer {
            if attn(i) {
                let attn_norm = wpush(&mut gr, &mut weights);
                let q = wpush(&mut gr, &mut weights);
                let k = wpush(&mut gr, &mut weights);
                let v = wpush(&mut gr, &mut weights);
                let q_norm = wpush(&mut gr, &mut weights);
                let k_norm = wpush(&mut gr, &mut weights);
                let out = wpush(&mut gr, &mut weights);
                let post_norm = wpush(&mut gr, &mut weights);
                let ffn_gate = wpush(&mut gr, &mut weights);
                let ffn_up = wpush(&mut gr, &mut weights);
                let ffn_down = wpush(&mut gr, &mut weights);
                let k_cache = gr.input(f32d(max_ctx * nkv * hd));
                let v_cache = gr.input(f32d(max_ctx * nkv * hd));
                layers.push(Q35LayerH::Attn(Q35AttnH {
                    attn_norm,
                    q,
                    k,
                    v,
                    q_norm,
                    k_norm,
                    out,
                    post_norm,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    n_ff: n_ff_of(i)?,
                    k_cache,
                    v_cache,
                }));
            } else {
                let attn_norm = wpush(&mut gr, &mut weights);
                let qkv = wpush(&mut gr, &mut weights);
                let gate = wpush(&mut gr, &mut weights);
                let conv1d = wpush(&mut gr, &mut weights);
                let alpha = wpush(&mut gr, &mut weights);
                let beta = wpush(&mut gr, &mut weights);
                let ssm_a = wpush(&mut gr, &mut weights);
                let dt_bias = wpush(&mut gr, &mut weights);
                let ssm_norm = wpush(&mut gr, &mut weights);
                let out = wpush(&mut gr, &mut weights);
                let post_norm = wpush(&mut gr, &mut weights);
                let ffn_gate = wpush(&mut gr, &mut weights);
                let ffn_up = wpush(&mut gr, &mut weights);
                let ffn_down = wpush(&mut gr, &mut weights);
                let conv_state = gr.input(f32d((kk - 1) * cc));
                let s_state = gr.input(f32d(nv * kd * vd));
                layers.push(Q35LayerH::Lin(Q35LinH {
                    attn_norm,
                    qkv,
                    gate,
                    conv1d,
                    alpha,
                    beta,
                    ssm_a,
                    dt_bias,
                    ssm_norm,
                    out,
                    post_norm,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    n_ff: n_ff_of(i)?,
                    conv_state,
                    s_state,
                }));
            }
        }
        let w_out_norm = wpush(&mut gr, &mut weights);
        let w_lm = wpush(&mut gr, &mut weights);
        let logits = gr.output(f32d(vocab));

        // scratch
        let xn = gr.internal(f32d(ne));
        let hn = gr.internal(f32d(ne));
        let sub = gr.internal(f32d(ne));
        let max_ff = (0..c.n_layer)
            .map(|i| n_ff_of(i).unwrap_or(0))
            .max()
            .unwrap_or(0);
        let gbuf = gr.internal(f32d(max_ff));
        let ubuf = gr.internal(f32d(max_ff));
        let actbuf = gr.internal(f32d(max_ff));
        // linear-mixer scratch
        let qkvbuf = gr.internal(f32d(cc));
        let zbuf = gr.internal(f32d(di));
        let convout = gr.internal(f32d(cc));
        let qbuf = gr.internal(f32d(key_dim));
        let kbuf = gr.internal(f32d(key_dim));
        let vbuf = gr.internal(f32d(nv * vd));
        let bbuf = gr.internal(f32d(nv));
        let abuf = gr.internal(f32d(nv));
        let dnout = gr.internal(f32d(nv * vd));
        // attn scratch
        let qg = gr.internal(f32d(nh * 2 * hd));
        let qa = gr.internal(f32d(nh * hd));
        let gate_a = gr.internal(f32d(nh * hd));
        let ka = gr.internal(f32d(nkv * hd));
        let va = gr.internal(f32d(nkv * hd));
        let attno = gr.internal(f32d(nh * hd));

        let rmsn = |gr: &mut Graph, x: TensorId, w: TensorId, dst: TensorId| {
            gr.push(Op::RmsNorm {
                x,
                weight: w,
                dst,
                rows: 1,
                dim: ne as u32,
                eps,
            });
        };
        let lin =
            |gr: &mut Graph, x: TensorId, w: TensorId, dst: TensorId, inf: usize, outf: usize| {
                gr.push(Op::Linear {
                    x,
                    weight: w,
                    dst,
                    m: 1,
                    in_f: inf as u32,
                    out_f: outf as u32,
                });
            };

        for (li, lh) in layers.iter().enumerate() {
            match lh {
                Q35LayerH::Lin(w) => {
                    rmsn(&mut gr, hidden, w.attn_norm, xn);
                    lin(&mut gr, xn, w.qkv, qkvbuf, ne, cc);
                    lin(&mut gr, xn, w.gate, zbuf, ne, di);
                    gr.push(Op::Conv1dSilu {
                        x: qkvbuf,
                        weight: w.conv1d,
                        state: w.conv_state,
                        dst: convout,
                        channels: cc as u32,
                        kernel: kk as u32,
                    });
                    // split conv_out → q | k | v
                    gr.push(Op::Copy {
                        src: convout,
                        src_off: 0,
                        dst: qbuf,
                        dst_off: 0,
                        n: key_dim as u32,
                    });
                    gr.push(Op::Copy {
                        src: convout,
                        src_off: key_dim as u32,
                        dst: kbuf,
                        dst_off: 0,
                        n: key_dim as u32,
                    });
                    gr.push(Op::Copy {
                        src: convout,
                        src_off: 2 * key_dim as u32,
                        dst: vbuf,
                        dst_off: 0,
                        n: (nv * vd) as u32,
                    });
                    lin(&mut gr, xn, w.beta, bbuf, ne, nv);
                    lin(&mut gr, xn, w.alpha, abuf, ne, nv);
                    gr.push(Op::DeltaNet {
                        q: qbuf,
                        k: kbuf,
                        v: vbuf,
                        b: bbuf,
                        a: abuf,
                        a_coef: w.ssm_a,
                        dt_bias: w.dt_bias,
                        state: w.s_state,
                        dst: dnout,
                        n_vhead: nv as u32,
                        head_k: kd as u32,
                        head_v: vd as u32,
                        eps: 1e-6,
                    });
                    // silu-gated RMSNorm per v-head: rmsnorm(out, ssm_norm) then * silu(z)
                    gr.push(Op::QkNorm {
                        x: dnout,
                        weight: w.ssm_norm,
                        dst: dnout,
                        rows: 1,
                        n_head: nv as u32,
                        head_dim: vd as u32,
                        eps,
                    });
                    gr.push(Op::GatedAct {
                        gate: zbuf,
                        up: dnout,
                        dst: dnout,
                        rows: 1,
                        nff: (nv * vd) as u32,
                        act: Activation::Silu,
                        up_off: 0,
                    });
                    lin(&mut gr, dnout, w.out, sub, di, ne);
                    gr.push(Op::Add {
                        a: hidden,
                        b: sub,
                        dst: hidden,
                        n: ne as u32,
                    });
                    // FFN
                    rmsn(&mut gr, hidden, w.post_norm, hn);
                    lin(&mut gr, hn, w.ffn_gate, gbuf, ne, w.n_ff);
                    lin(&mut gr, hn, w.ffn_up, ubuf, ne, w.n_ff);
                    gr.push(Op::GatedAct {
                        gate: gbuf,
                        up: ubuf,
                        dst: actbuf,
                        rows: 1,
                        nff: w.n_ff as u32,
                        act: Activation::Silu,
                        up_off: 0,
                    });
                    lin(&mut gr, actbuf, w.ffn_down, sub, w.n_ff, ne);
                    gr.push(Op::Add {
                        a: hidden,
                        b: sub,
                        dst: hidden,
                        n: ne as u32,
                    });
                }
                Q35LayerH::Attn(w) => {
                    rmsn(&mut gr, hidden, w.attn_norm, xn);
                    // q proj outputs q+gate interleaved per head [h: q(hd), gate(hd)].
                    lin(&mut gr, xn, w.q, qg, ne, nh * 2 * hd);
                    for h in 0..nh {
                        gr.push(Op::Copy {
                            src: qg,
                            src_off: (h * 2 * hd) as u32,
                            dst: qa,
                            dst_off: (h * hd) as u32,
                            n: hd as u32,
                        });
                        gr.push(Op::Copy {
                            src: qg,
                            src_off: (h * 2 * hd + hd) as u32,
                            dst: gate_a,
                            dst_off: (h * hd) as u32,
                            n: hd as u32,
                        });
                    }
                    lin(&mut gr, xn, w.k, ka, ne, nkv * hd);
                    lin(&mut gr, xn, w.v, va, ne, nkv * hd);
                    // per-head q/k norm then RoPE
                    gr.push(Op::QkNorm {
                        x: qa,
                        weight: w.q_norm,
                        dst: qa,
                        rows: 1,
                        n_head: nh as u32,
                        head_dim: hd as u32,
                        eps,
                    });
                    gr.push(Op::QkNorm {
                        x: ka,
                        weight: w.k_norm,
                        dst: ka,
                        rows: 1,
                        n_head: nkv as u32,
                        head_dim: hd as u32,
                        eps,
                    });
                    gr.push(Op::Rope {
                        x: qa,
                        positions,
                        dst: qa,
                        rows: 1,
                        n_head: nh as u32,
                        head_dim: hd as u32,
                        rope_dim: c.rope_dim as u32,
                        theta: c.rope_theta,
                        freq_factors: None,
                    });
                    gr.push(Op::Rope {
                        x: ka,
                        positions,
                        dst: ka,
                        rows: 1,
                        n_head: nkv as u32,
                        head_dim: hd as u32,
                        rope_dim: c.rope_dim as u32,
                        theta: c.rope_theta,
                        freq_factors: None,
                    });
                    gr.push(Op::WriteKv {
                        src: ka,
                        cache: w.k_cache,
                        rows: 1,
                        row_stride: (nkv * hd) as u32,
                        pos: pos as u32,
                    });
                    gr.push(Op::WriteKv {
                        src: va,
                        cache: w.v_cache,
                        rows: 1,
                        row_stride: (nkv * hd) as u32,
                        pos: pos as u32,
                    });
                    gr.push(Op::Attention {
                        q: qa,
                        k_cache: w.k_cache,
                        v_cache: w.v_cache,
                        dst: attno,
                        rows: 1,
                        kv_len: (pos + 1) as u32,
                        n_head: nh as u32,
                        n_kv: nkv as u32,
                        head_dim: hd as u32,
                        scale,
                        mask: AttnMask::Causal,
                        pos: pos as u32,
                    });
                    // per-head sigmoid output gate
                    gr.push(Op::GatedAct {
                        gate: gate_a,
                        up: attno,
                        dst: attno,
                        rows: 1,
                        nff: (nh * hd) as u32,
                        act: Activation::Sigmoid,
                        up_off: 0,
                    });
                    lin(&mut gr, attno, w.out, sub, nh * hd, ne);
                    gr.push(Op::Add {
                        a: hidden,
                        b: sub,
                        dst: hidden,
                        n: ne as u32,
                    });
                    // FFN
                    rmsn(&mut gr, hidden, w.post_norm, hn);
                    lin(&mut gr, hn, w.ffn_gate, gbuf, ne, w.n_ff);
                    lin(&mut gr, hn, w.ffn_up, ubuf, ne, w.n_ff);
                    gr.push(Op::GatedAct {
                        gate: gbuf,
                        up: ubuf,
                        dst: actbuf,
                        rows: 1,
                        nff: w.n_ff as u32,
                        act: Activation::Silu,
                        up_off: 0,
                    });
                    lin(&mut gr, actbuf, w.ffn_down, sub, w.n_ff, ne);
                    gr.push(Op::Add {
                        a: hidden,
                        b: sub,
                        dst: hidden,
                        n: ne as u32,
                    });
                    let _ = li;
                }
            }
        }
        rmsn(&mut gr, hidden, w_out_norm, hn);
        lin(&mut gr, hn, w_lm, logits, ne, vocab);

        // collect state-input handles per layer for binding (interleaved by kind)
        let mut state_ids: Vec<TensorId> = Vec::new();
        for lh in &layers {
            match lh {
                Q35LayerH::Lin(w) => {
                    state_ids.push(w.conv_state);
                    state_ids.push(w.s_state);
                }
                Q35LayerH::Attn(w) => {
                    state_ids.push(w.k_cache);
                    state_ids.push(w.v_cache);
                }
            }
        }
        Ok((gr, hidden, positions, [weights, state_ids].concat(), logits))
    };

    // ── drive ───────────────────────────────────────────────────────────────────
    let mut cur = prompt_ids.clone();
    let mut outs: Vec<u32> = Vec::new();
    let mut logits = vec![0f32; vocab];
    let mut prompt_t = std::time::Duration::ZERO;
    let mut decode_t = std::time::Duration::ZERO;
    let mut decode_n = 0usize;
    let mut printed = 0usize; // streaming detok cursor
    for pos in 0..(prompt_ids.len() + n) {
        let step_t0 = std::time::Instant::now();
        let t = cur[pos] as usize;
        be.upload(
            hidden_buf.as_ref(),
            bytemuck::cast_slice(&token_embd[t * ne..t * ne + ne]),
        )
        .map_err(|e| anyhow!("{e}"))?;
        be.upload(pos_buf.as_ref(), bytemuck::cast_slice(&[pos as i32]))
            .map_err(|e| anyhow!("{e}"))?;

        let (gr, h_hidden, h_pos, h_bind, h_logits) = build(pos)?;
        let plan = be.compile(&gr).map_err(|e| anyhow!("{e}"))?;
        let mut b = Bindings::new();
        b.bind(h_hidden, hidden_buf.as_ref());
        b.bind(h_pos, pos_buf.as_ref());
        b.bind(h_logits, logits_buf.as_ref());
        // h_bind = [weights.., state_ids..]; bind weights then state in declaration order.
        let nw = wbufs.len();
        for (i, id) in h_bind.iter().take(nw).enumerate() {
            b.bind(*id, wbufs[i].as_ref());
        }
        // state handles in layer order: per layer (conv,s) for linear, (k,v) for attn.
        let mut si = nw;
        for i in 0..c.n_layer {
            if attn(i) {
                b.bind(h_bind[si], k_bufs[i].as_ref().unwrap().as_ref());
                b.bind(h_bind[si + 1], v_bufs[i].as_ref().unwrap().as_ref());
            } else {
                b.bind(h_bind[si], conv_bufs[i].as_ref().unwrap().as_ref());
                b.bind(h_bind[si + 1], s_bufs[i].as_ref().unwrap().as_ref());
            }
            si += 2;
        }
        be.execute(plan.as_ref(), &b).map_err(|e| anyhow!("{e}"))?;

        if pos + 1 >= prompt_ids.len() {
            be.download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
                .map_err(|e| anyhow!("{e}"))?;
            let next = argmax(&logits);
            crate::stream_token(&tok, &mut outs, &mut printed, next, &mut on_piece);
            decode_t += step_t0.elapsed();
            decode_n += 1;
            if outs.len() >= n {
                break;
            }
            if cur.len() <= pos + 1 {
                cur.push(next);
            }
        } else {
            prompt_t += step_t0.elapsed();
        }
    }
    // The text streamed out via `on_piece`; return only timing/counts.
    Ok(crate::cpu_backend::CpuStats {
        n_prompt: prompt_ids.len(),
        prompt_secs: prompt_t.as_secs_f64(),
        n_gen: decode_n,
        decode_secs: decode_t.as_secs_f64(),
    })
}

/// True if the GGUF at `path` is a `qwen35` (Qwen3-Next) model.
pub fn is_qwen35(path: &std::path::Path) -> bool {
    Gguf::open(path)
        .ok()
        .map(|g| g.metadata().str("general.architecture") == Some("qwen35"))
        .unwrap_or(false)
}

/// One-shot chat generation on the CPU reference: applies the Qwen chat template, greedy-decodes
/// until `<|im_end|>`/eos or `max_new`, streaming each decoded piece to `on_piece`. Returns
/// (prompt_tokens, generated_tokens). For Qwen3.5/3.6 (no GPU path — see docs/QWEN35.md).
pub fn generate_chat(
    path: &std::path::Path,
    message: &str,
    max_new: usize,
    mut on_piece: impl FnMut(&str),
) -> Result<(usize, usize)> {
    let g = Gguf::open(path).map_err(|e| anyhow!("{e}"))?;
    let m = Model::load(&g)?;
    let tok = crate::build_tokenizer(&g)?;
    let eos = g
        .metadata()
        .u64("tokenizer.ggml.eos_token_id")
        .map(|x| x as u32);
    let im_end = tok.token_to_id("<|im_end|>");
    let prompt = format!("<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n");
    let enc = tok
        .encode(prompt, false)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let ids = enc.get_ids();
    let n_prompt = ids.len();

    let mut st = m.new_state();
    let mut last = 0u32;
    for (i, &id) in ids.iter().enumerate() {
        let logits = m.forward(id, &mut st);
        if i == n_prompt - 1 {
            last = argmax(&logits);
        }
    }
    // incremental detokenization: decode the growing id list, emit only the new suffix
    let mut gen_ids: Vec<u32> = Vec::new();
    let mut shown = String::new();
    let mut n_gen = 0usize;
    for _ in 0..max_new {
        if Some(last) == eos || (im_end.is_some() && Some(last) == im_end) {
            break;
        }
        gen_ids.push(last);
        n_gen += 1;
        let full = tok.decode(&gen_ids, false).unwrap_or_default();
        if full.len() > shown.len() && full.is_char_boundary(shown.len()) {
            on_piece(&full[shown.len()..]);
            shown = full;
        }
        let logits = m.forward(last, &mut st);
        last = argmax(&logits);
    }
    Ok((n_prompt, n_gen))
}

fn argmax(v: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::MIN;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> std::path::PathBuf {
        if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
            return std::path::PathBuf::from(p);
        }
        // the 0.8B pulled into our store
        dirs_cache().join("infr/models/blobs/sha256-bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517")
    }
    fn dirs_cache() -> std::path::PathBuf {
        std::env::var("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache")
            })
    }

    #[test]
    #[ignore = "needs the Qwen3.5-0.8B gguf in the local store"]
    fn loads_and_dims() {
        let g = Gguf::open(&model_path()).unwrap();
        let m = Model::load(&g).unwrap();
        let c = &m.cfg;
        println!("cfg: {c:?}");
        println!(
            "k_heads={} head_k={} v_heads={} head_v={} conv_ch={}",
            c.num_k_heads(),
            c.head_k_dim(),
            c.num_v_heads(),
            c.head_v_dim(),
            c.conv_channels()
        );
        assert_eq!(c.n_layer, 24);
        assert_eq!(c.conv_channels(), 6144);
        assert_eq!(c.head_v_dim(), 128);
        let n_attn = (0..c.n_layer).filter(|&i| c.is_attn_layer(i)).count();
        assert_eq!(n_attn, 6, "expected 6 full-attention layers");
        assert_eq!(m.layers.len(), 24);
    }

    #[test]
    #[ignore = "needs the Qwen3.5-0.8B gguf in the local store"]
    fn greedy_generate() {
        let g = Gguf::open(&model_path()).unwrap();
        let prompt =
            std::env::var("Q35_PROMPT").unwrap_or_else(|_| "The capital of France is".to_string());
        let n = std::env::var("Q35_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);
        let out = generate(&g, &prompt, n).unwrap();
        println!("=== qwen35 CPU greedy ===\n{out}");
    }

    /// The seam-based pure-CPU runner (`generate_cpu`, via `CpuBackend` + the new Conv1dSilu/DeltaNet
    /// ops, no Vulkan) must match the bespoke CPU oracle (`generate` w/ `Q35_CPU=1`) token-for-token —
    /// both are f32, so the match is exact.
    #[test]
    #[ignore = "needs the Qwen3.5-0.8B gguf; run --test-threads=1"]
    fn seam_cpu_matches_oracle() {
        let g = Gguf::open(&model_path()).unwrap();
        let prompt = "The capital of France is";
        let n = 16;
        std::env::set_var("Q35_CPU", "1");
        let oracle = generate(&g, prompt, n).unwrap();
        let mut seam = String::new();
        generate_cpu(&model_path(), prompt, n, |p| seam.push_str(p)).unwrap();
        println!("ORACLE: {oracle:?}\nSEAM:   {seam:?}");
        assert_eq!(
            seam, oracle,
            "qwen35 seam CPU must match the bespoke CPU oracle"
        );
    }

    /// qwen35 (Qwen3-Next hybrid: gated DeltaNet + gated full-attention) pure-CPU greedy must match
    /// the GPU-hybrid greedy token-for-token. `Q35_CPU=1` forces every linear projection onto the CPU
    /// (f32); unset, they run on the GPU (f16) — so this validates the CPU path end-to-end against the
    /// GPU one. The SSM recurrence / conv / gated attention run on the CPU in both modes.
    #[test]
    #[ignore = "needs a Vulkan GPU + the Qwen3.5-0.8B gguf; run --test-threads=1"]
    fn cpu_matches_hybrid() {
        let g = Gguf::open(&model_path()).unwrap();
        let prompt = "The capital of France is";
        let n = 16;
        std::env::set_var("Q35_CPU", "1");
        let cpu = generate(&g, prompt, n).unwrap();
        std::env::remove_var("Q35_CPU");
        let hybrid = generate(&g, prompt, n).unwrap();
        println!("CPU:    {cpu:?}\nHYBRID: {hybrid:?}");
        assert_eq!(
            cpu, hybrid,
            "qwen35 CPU must match GPU-hybrid greedy output"
        );
    }
}
