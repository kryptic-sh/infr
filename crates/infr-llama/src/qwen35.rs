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

use crate::load_tensor_dequant;
use anyhow::{anyhow, bail, Context, Result};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::tensor::TensorDesc;
use infr_core::{DType, TensorId, WeightSource};
use infr_cpu::CpuBackend;
use infr_gguf::{Gguf, TensorBytes};
use infr_vulkan::VulkanBackend;

/// A linear-projection weight on whichever device has a kernel for it: GPU (the shared
/// [`crate::Wt`] — native-quant blocks or f16, with the same dispatch the dense `Llama` path uses)
/// or CPU f32 (the `Q35_CPU=1` oracle). One call site, [`Lin::mul`]/[`Lin::record`], hides which.
enum Lin {
    Cpu(Vec<f32>),
    Gpu(crate::Wt),
}

impl Lin {
    /// `y[out] = W[out,in] · x[in]` (single row), CPU reference only. The GPU forward records via
    /// [`Lin::record`]; a GPU weight never reaches here (all `mul` call sites are the oracle).
    fn mul(&self, _be: Option<&VulkanBackend>, x: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
        match self {
            Lin::Cpu(w) => matvec(w, in_f, out_f, x),
            Lin::Gpu(_) => unreachable!("qwen35 GPU weights record via Lin::record, not mul"),
        }
    }

    /// Record a single-row GEMV `y = x·Wᵀ` into a command buffer via the shared [`crate::rec_linear`]
    /// dispatch (native quant + f16). Requires a GPU weight.
    fn record(
        &self,
        rec: &infr_vulkan::Recorder,
        x: &dyn Buffer,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        match self {
            Lin::Gpu(w) => crate::rec_linear(rec, w, x, y, 1, in_f, out_f),
            Lin::Cpu(_) => panic!("Lin::record requires a GPU weight (not the Q35_CPU oracle)"),
        }
    }

    /// The shared GPU weight (panics on the Q35_CPU oracle) — for passing into shared mixer blocks.
    fn gpu_wt(&self) -> &crate::Wt {
        match self {
            Lin::Gpu(w) => w,
            Lin::Cpu(_) => panic!("gpu_wt requires a GPU weight (not the Q35_CPU oracle)"),
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
    /// GPU copies of the non-`Lin` SSM/norm weights, for the GPU-resident forward (`Some` when a
    /// backend is present). The host `Vec<f32>` above stay for the `Q35_CPU=1` oracle.
    gpu: Option<LinearGpuW>,
}

/// GPU-resident DeltaNet/norm weights for one linear layer. Norms, conv1d, the `a`/`dt_bias` gates
/// are f32 buffers (read directly by rmsnorm/conv1d_silu/deltanet); `alpha`/`beta` are f16 GEMV
/// weights (recorded via `Recorder::linear`).
struct LinearGpuW {
    attn_norm: Box<dyn Buffer>,
    conv1d: Box<dyn Buffer>,
    alpha: Box<dyn Buffer>,
    beta: Box<dyn Buffer>,
    a: Box<dyn Buffer>,
    dt_bias: Box<dyn Buffer>,
    ssm_norm: Box<dyn Buffer>,
    post_norm: Box<dyn Buffer>,
}

/// GPU-resident norm weights for one attention layer (`Some` when a backend is present).
struct AttnGpuW {
    attn_norm: Box<dyn Buffer>,
    q_norm: Box<dyn Buffer>,
    k_norm: Box<dyn Buffer>,
    post_norm: Box<dyn Buffer>,
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
    /// GPU copies of the attention norm weights (`Some` when a backend is present).
    gpu: Option<AttnGpuW>,
}

enum Layer {
    Linear(LinearLayer),
    Attn(AttnLayer),
}

/// Full model weights. With a backend the forward runs fully on the GPU; `Q35_CPU=1` forces the
/// pure-CPU oracle (host `Vec<f32>` weights, no Vulkan).
pub struct Model {
    pub cfg: Cfg,
    token_embd: Vec<f32>, // [vocab, n_embd] host (embedding gather)
    output_norm: Vec<f32>,
    output_norm_buf: Option<Box<dyn Buffer>>, // GPU copy for the final norm
    lm_head: Lin,                             // [vocab, n_embd]
    layers: Vec<Layer>,
    be: Option<VulkanBackend>,
}

impl Model {
    /// Map a GGUF projection tensor to a GPU [`Lin`] via the SHARED dense loader ([`crate::upload_wt`]:
    /// native-quant blocks stay in-VRAM, float → f16), or CPU f32 when `be` is `None` (`Q35_CPU=1`).
    fn upload_lin(g: &Gguf, be: Option<&VulkanBackend>, name: &str) -> Result<Lin> {
        match be {
            None => Ok(Lin::Cpu(
                load_tensor_dequant(g, name)
                    .map(|x| x.0)
                    .with_context(|| name.to_string())?,
            )),
            Some(be) => Ok(Lin::Gpu(
                crate::upload_wt(be, g, name).with_context(|| name.to_string())?,
            )),
        }
    }

    pub fn load(g: &Gguf) -> Result<Self> {
        let mut cfg = Cfg::from_gguf(g)?;
        // CPU-only oracle skips Vulkan entirely; otherwise upload the matmul weights to the GPU.
        let be = if std::env::var("Q35_CPU").is_ok() {
            None
        } else {
            Some(VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?)
        };
        // Weight-load progress bar: every BufferUsage::Weights alloc advances it automatically (the
        // ticking lives in VulkanBackend::alloc), so we just open the scope. Guard drops at end of
        // `load`. Total = sum of GGUF tensor bytes (the bar's denominator); inert on the CPU oracle.
        let _wp = be.as_ref().map(|be| {
            let total: u64 = g.tensors().iter().map(|t| t.nbytes as u64).sum();
            be.weight_progress(Some(total))
        });
        let (token_embd, te_shape) = load_tensor_dequant(g, "token_embd.weight")?;
        cfg.vocab = te_shape[1];
        let output_norm = load_tensor_dequant(g, "output_norm.weight")?.0;

        // Shared dtype policy: load a projection weight to a GPU `Lin` in the dtype matching its
        // GGUF source (see `upload_lin`), or keep it CPU f32 (`Q35_CPU=1`).
        let lin = |name: &str| -> Result<Lin> { Self::upload_lin(g, be.as_ref(), name) };
        // On a GPU run the norm/gate weights live in VRAM (the `gpu` structs below); drop the host
        // f32 copy so we load straight into VRAM like the other model impls. The host Vec is kept
        // only for the `Q35_CPU=1` oracle (no backend), whose CPU forward reads it.
        let host_only = be.is_some();
        let take = |v: Vec<f32>| if host_only { Vec::new() } else { v };
        let lm_head = if g.tensors().iter().any(|t| t.name == "output.weight") {
            lin("output.weight")?
        } else {
            // tied: the lm head IS token_embd — load it like any projection (raw native blocks for
            // quant, in-shader dequant; the host keeps its own f32 `token_embd` for the gather).
            lin("token_embd.weight")?
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
                let attn_norm = get("attn_norm.weight")?;
                let q_norm = get("attn_q_norm.weight")?;
                let k_norm = get("attn_k_norm.weight")?;
                let post_norm = get("post_attention_norm.weight")?;
                let gpu = be
                    .as_ref()
                    .map(|be| -> Result<AttnGpuW> {
                        let f32b = |v: &[f32]| be.upload_weight(v).map_err(|e| anyhow!("{e}"));
                        Ok(AttnGpuW {
                            attn_norm: f32b(&attn_norm)?,
                            q_norm: f32b(&q_norm)?,
                            k_norm: f32b(&k_norm)?,
                            post_norm: f32b(&post_norm)?,
                        })
                    })
                    .transpose()?;
                layers.push(Layer::Attn(AttnLayer {
                    attn_norm: take(attn_norm),
                    q: glin("attn_q.weight")?,
                    k: glin("attn_k.weight")?,
                    v: glin("attn_v.weight")?,
                    q_norm: take(q_norm),
                    k_norm: take(k_norm),
                    out: glin("attn_output.weight")?,
                    post_norm: take(post_norm),
                    ffn_gate: glin("ffn_gate.weight")?,
                    ffn_up: glin("ffn_up.weight")?,
                    ffn_down: glin("ffn_down.weight")?,
                    n_ff,
                    gpu,
                }));
            } else {
                let attn_norm = get("attn_norm.weight")?;
                let conv1d = get("ssm_conv1d.weight")?;
                let alpha = get("ssm_alpha.weight")?;
                let beta = get("ssm_beta.weight")?;
                let a = get("ssm_a")?;
                let dt_bias = get("ssm_dt.bias")?;
                let ssm_norm = get("ssm_norm.weight")?;
                let post_norm = get("post_attention_norm.weight")?;
                // Upload the non-Lin SSM/norm weights to GPU for the resident forward (norms/conv1d/
                // gates as f32; alpha/beta as f16 GEMV weights). `None` for the Q35_CPU oracle.
                let gpu = be
                    .as_ref()
                    .map(|be| -> Result<LinearGpuW> {
                        let f32b = |v: &[f32]| be.upload_weight(v).map_err(|e| anyhow!("{e}"));
                        let f16b = |v: &[f32]| be.upload_weight_f16(v).map_err(|e| anyhow!("{e}"));
                        Ok(LinearGpuW {
                            attn_norm: f32b(&attn_norm)?,
                            conv1d: f32b(&conv1d)?,
                            alpha: f16b(&alpha)?,
                            beta: f16b(&beta)?,
                            a: f32b(&a)?,
                            dt_bias: f32b(&dt_bias)?,
                            ssm_norm: f32b(&ssm_norm)?,
                            post_norm: f32b(&post_norm)?,
                        })
                    })
                    .transpose()?;
                layers.push(Layer::Linear(LinearLayer {
                    attn_norm: take(attn_norm),
                    qkv: glin("attn_qkv.weight")?,
                    gate: glin("attn_gate.weight")?,
                    conv1d: take(conv1d),
                    alpha: take(alpha),
                    beta: take(beta),
                    a: take(a),
                    dt_bias: take(dt_bias),
                    ssm_norm: take(ssm_norm),
                    out: glin("ssm_out.weight")?,
                    post_norm: take(post_norm),
                    ffn_gate: glin("ffn_gate.weight")?,
                    ffn_up: glin("ffn_up.weight")?,
                    ffn_down: glin("ffn_down.weight")?,
                    gpu,
                    n_ff,
                }));
            }
        }
        let output_norm_buf = be
            .as_ref()
            .map(|be| be.upload_weight(&output_norm).map_err(|e| anyhow!("{e}")))
            .transpose()?;
        Ok(Model {
            cfg,
            token_embd,
            output_norm: take(output_norm),
            output_norm_buf,
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

// ── GPU SSM helpers (migrating the qwen35 SSM off the CPU) ──────────────────────

/// Upload an f32 slice to a fresh device buffer.
fn dev(be: &VulkanBackend, data: &[f32]) -> Box<dyn Buffer> {
    let b = be
        .alloc((data.len() * 4).max(4), BufferUsage::Activations)
        .expect("alloc");
    be.upload(b.as_ref(), bytemuck::cast_slice(data))
        .expect("upload");
    b
}
/// Download `n` f32s from a device buffer.
fn read(be: &VulkanBackend, buf: &dyn Buffer, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    be.download(buf, &mut bytes).expect("download");
    bytemuck::cast_slice(&bytes).to_vec()
}
/// On-device per-attention-layer KV cache: f16 `[max_ctx, n_kv*head_dim]` buffers appended in place
/// (qk_norm_rope writes K at `out_base=pos`, store_f16 writes V at `pos`), so the whole attention
/// runs in one command buffer with no host round-trip. `None` on the Q35_CPU oracle (host Vecs).
struct DevKv {
    k: Box<dyn Buffer>,
    v: Box<dyn Buffer>,
}

/// Per-layer recurrent state.
enum LayerState {
    Linear {
        conv: Vec<f32>, // [d_conv-1, conv_channels] rolling history (oldest first)
        s: Vec<f32>,    // [num_v_heads, head_k_dim, head_v_dim]
    },
    Attn {
        k: Vec<f32>, // [pos, n_kv*head_dim] — Q35_CPU oracle only
        v: Vec<f32>,
        dev: Option<DevKv>, // GPU path: on-device f16 KV cache
    },
}

pub struct State {
    layers: Vec<LayerState>,
    pos: usize,
}

/// Max decode context for the on-device KV cache (positions). `INFR_MAX_CTX`, default 8192.
fn max_ctx() -> usize {
    std::env::var("INFR_MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192)
}

impl Model {
    pub fn new_state(&self) -> State {
        let c = &self.cfg;
        let kvrow = c.n_kv * c.head_dim;
        let layers = (0..c.n_layer)
            .map(|i| {
                if c.is_attn_layer(i) {
                    // On the GPU path, alloc the device f16 KV cache (capacity = max_ctx positions).
                    let dev = self.be.as_ref().map(|be| {
                        let bytes = (max_ctx() * kvrow * 2).max(4);
                        DevKv {
                            k: be.alloc(bytes, BufferUsage::Activations).expect("kv k"),
                            v: be.alloc(bytes, BufferUsage::Activations).expect("kv v"),
                        }
                    });
                    LayerState::Attn {
                        k: vec![],
                        v: vec![],
                        dev,
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
                        w.gpu.as_ref().map(|g| g.post_norm.as_ref()),
                        &w.ffn_gate,
                        &w.ffn_up,
                        &w.ffn_down,
                        w.n_ff,
                    );
                    for (h, di) in hidden.iter_mut().zip(&d) {
                        *h += di;
                    }
                }
                (Layer::Attn(w), LayerState::Attn { k, v, dev }) => {
                    let y = self.attn_mixer(w, &hidden, k, v, dev.as_ref(), pos);
                    if std::env::var("Q35_NOATTN").is_err() {
                        for (h, yi) in hidden.iter_mut().zip(&y) {
                            *h += yi;
                        }
                    }
                    let d = self.ffn(
                        &hidden,
                        &w.post_norm,
                        w.gpu.as_ref().map(|g| g.post_norm.as_ref()),
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
        if let (Some(be), Some(onb)) = (self.be.as_ref(), self.output_norm_buf.as_ref()) {
            let hb = dev(be, &hidden);
            let nm = be
                .alloc(ne * 4, BufferUsage::Activations)
                .expect("alloc norm");
            let logits = be
                .alloc(c.vocab * 4, BufferUsage::Activations)
                .expect("alloc logits");
            {
                let rec = be.recorder().expect("recorder");
                rec.rmsnorm(hb.as_ref(), onb.as_ref(), nm.as_ref(), 1, ne, c.eps);
                self.lm_head
                    .record(&rec, nm.as_ref(), logits.as_ref(), ne, c.vocab);
                rec.finish().expect("finish");
            }
            return read(be, logits.as_ref(), c.vocab);
        }
        let hn = rmsnorm(&hidden, &self.output_norm, c.eps);
        self.lm_head.mul(self.be.as_ref(), &hn, ne, c.vocab)
    }

    /// SwiGLU FFN (one token). Fully on the GPU when a backend + the GPU norm buffer are present; the
    /// CPU reference otherwise. `norm`/`norm_buf` are the post-attention norm (host / GPU copy).
    #[allow(clippy::too_many_arguments)]
    fn ffn(
        &self,
        hidden: &[f32],
        norm: &[f32],
        norm_buf: Option<&dyn Buffer>,
        gate: &Lin,
        up: &Lin,
        down: &Lin,
        n_ff: usize,
    ) -> Vec<f32> {
        let ne = self.cfg.n_embd;
        if let (Some(be), Some(nb)) = (self.be.as_ref(), norm_buf) {
            return self.ffn_gpu(be, hidden, nb, gate, up, down, n_ff);
        }
        let h2 = rmsnorm(hidden, norm, self.cfg.eps);
        let g = gate.mul(None, &h2, ne, n_ff);
        let u = up.mul(None, &h2, ne, n_ff);
        let act: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
        down.mul(None, &act, n_ff, ne)
    }

    /// Fully-GPU SwiGLU FFN: rmsnorm → gate/up GEMV → silu_mul → down GEMV, one command buffer.
    #[allow(clippy::too_many_arguments)]
    fn ffn_gpu(
        &self,
        be: &VulkanBackend,
        hidden: &[f32],
        norm_buf: &dyn Buffer,
        gate: &Lin,
        up: &Lin,
        down: &Lin,
        n_ff: usize,
    ) -> Vec<f32> {
        let (ne, eps) = (self.cfg.n_embd, self.cfg.eps);
        let hb = dev(be, hidden);
        let al = |n: usize| {
            be.alloc((n * 4).max(4), BufferUsage::Activations)
                .expect("alloc")
        };
        let (h2, g, u, act, out) = (al(ne), al(n_ff), al(n_ff), al(n_ff), al(ne));
        {
            let rec = be.recorder().expect("recorder");
            rec.rmsnorm(hb.as_ref(), norm_buf, h2.as_ref(), 1, ne, eps);
            // SwiGLU channel-mixer via the shared block (qwen35 carries split gate/up weights).
            crate::mixers::ffn::record_swiglu(
                &rec,
                h2.as_ref(),
                crate::mixers::ffn::GateUp::Split {
                    gate: gate.gpu_wt(),
                    up: up.gpu_wt(),
                },
                down.gpu_wt(),
                u.as_ref(), // gu unused for Split
                g.as_ref(),
                u.as_ref(),
                act.as_ref(),
                out.as_ref(),
                None, // qwen35 adds the residual itself in forward()
                1,
                ne,
                n_ff,
            );
            rec.finish().expect("finish");
        }
        read(be, out.as_ref(), ne)
    }

    /// Fully-GPU gated-DeltaNet mixer for one token: the ENTIRE mixer (rmsnorm → qkv/gate GEMV →
    /// conv1d+SiLU → beta/alpha gate GEMVs → DeltaNet recurrence → silu-gated rmsnorm → out GEMV)
    /// recorded into ONE command buffer — no CPU math. hidden + the conv/S state round-trip the host
    /// (perf: a resident forward keeps them on-device); returns the mixer output `[ne]`.
    fn linear_mixer_gpu(
        &self,
        w: &LinearLayer,
        gpu: &LinearGpuW,
        be: &VulkanBackend,
        hidden: &[f32],
        conv: &mut [f32],
        s: &mut [f32],
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (ne, kd, vd, cc, di) = (
            c.n_embd,
            c.head_k_dim(),
            c.head_v_dim(),
            c.conv_channels(),
            c.d_inner,
        );
        let (nk, nv) = (c.num_k_heads(), c.num_v_heads());
        let (kconv, eps) = (c.d_conv, c.eps);
        let hb = dev(be, hidden);
        let convb = dev(be, conv);
        let sb = dev(be, s);
        let al = |n: usize| {
            be.alloc((n * 4).max(4), BufferUsage::Activations)
                .expect("alloc scratch")
        };
        let (xn, qkv, z, conv_out) = (al(ne), al(cc), al(di), al(cc));
        let (bg, ag) = (al(nv), al(nv));
        let (qd, kdd, vdd) = (al(nk * kd), al(nk * kd), al(nv * vd));
        let (dn, nm, gated, out) = (al(nv * vd), al(nv * vd), al(di), al(ne));
        {
            let rec = be.recorder().expect("recorder");
            rec.rmsnorm(hb.as_ref(), gpu.attn_norm.as_ref(), xn.as_ref(), 1, ne, eps);
            w.qkv.record(&rec, xn.as_ref(), qkv.as_ref(), ne, cc);
            w.gate.record(&rec, xn.as_ref(), z.as_ref(), ne, di);
            rec.conv1d_silu(
                qkv.as_ref(),
                gpu.conv1d.as_ref(),
                convb.as_ref(),
                conv_out.as_ref(),
                1, // rows: single-token bespoke path
                cc,
                kconv,
            );
            rec.linear(gpu.beta.as_ref(), xn.as_ref(), bg.as_ref(), 1, ne, nv);
            rec.linear(gpu.alpha.as_ref(), xn.as_ref(), ag.as_ref(), 1, ne, nv);
            // split conv_out [q | k | v] into the deltanet inputs.
            rec.copy(conv_out.as_ref(), 0, qd.as_ref(), 0, nk * kd * 4);
            rec.copy(conv_out.as_ref(), nk * kd * 4, kdd.as_ref(), 0, nk * kd * 4);
            rec.copy(
                conv_out.as_ref(),
                2 * nk * kd * 4,
                vdd.as_ref(),
                0,
                nv * vd * 4,
            );
            rec.deltanet(
                qd.as_ref(),
                kdd.as_ref(),
                vdd.as_ref(),
                bg.as_ref(),
                ag.as_ref(),
                gpu.a.as_ref(),
                gpu.dt_bias.as_ref(),
                sb.as_ref(),
                dn.as_ref(),
                1, // rows: single-token bespoke path
                nv,
                nk,
                kd,
                vd,
                1e-6,
            );
            // silu-gated rmsnorm per v-head: rmsnorm(dn, ssm_norm) * silu(z).
            rec.rmsnorm(dn.as_ref(), gpu.ssm_norm.as_ref(), nm.as_ref(), nv, vd, eps);
            rec.silu_mul(z.as_ref(), nm.as_ref(), gated.as_ref(), di);
            w.out.record(&rec, gated.as_ref(), out.as_ref(), di, ne);
            rec.finish().expect("finish");
        }
        conv.copy_from_slice(&read(be, convb.as_ref(), conv.len()));
        s.copy_from_slice(&read(be, sb.as_ref(), s.len()));
        read(be, out.as_ref(), ne)
    }

    /// Gated-DeltaNet mixer (one token). Fully on the GPU when a backend is present; the CPU
    /// reference (`Q35_CPU=1` oracle) otherwise.
    fn linear_mixer(
        &self,
        w: &LinearLayer,
        hidden: &[f32],
        conv: &mut [f32],
        s: &mut [f32],
    ) -> Vec<f32> {
        if let (Some(be), Some(gpu)) = (self.be.as_ref(), w.gpu.as_ref()) {
            return self.linear_mixer_gpu(w, gpu, be, hidden, conv, s);
        }
        // ── CPU reference (oracle) ──────────────────────────────────────────────
        let c = &self.cfg;
        let (ne, kd, vd) = (c.n_embd, c.head_k_dim(), c.head_v_dim());
        let (nk, nv) = (c.num_k_heads(), c.num_v_heads());
        let cc = c.conv_channels();
        let xn = rmsnorm(hidden, &w.attn_norm, c.eps);
        let qkv = w.qkv.mul(None, &xn, ne, cc);
        let z = w.gate.mul(None, &xn, ne, c.d_inner);
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
        for k in 0..k_conv - 2 {
            for ch in 0..cc {
                conv[k * cc + ch] = conv[(k + 1) * cc + ch];
            }
        }
        for ch in 0..cc {
            conv[(k_conv - 2) * cc + ch] = qkv[ch];
        }
        let key_dim = nk * kd;
        let (q_all, rest) = conv_out.split_at(key_dim);
        let (k_all, v_all) = rest.split_at(key_dim);
        let b = matvec(&w.beta, ne, nv, &xn);
        let a = matvec(&w.alpha, ne, nv, &xn);
        let mut out = vec![0.0f32; nv * vd];
        let qscale = 1.0 / (kd as f32).sqrt();
        for h in 0..nv {
            let kh_idx = h % nk;
            let mut qh = q_all[kh_idx * kd..kh_idx * kd + kd].to_vec();
            let mut kh = k_all[kh_idx * kd..kh_idx * kd + kd].to_vec();
            let vh = &v_all[h * vd..h * vd + vd];
            l2norm(&mut qh, 1e-6);
            l2norm(&mut kh, 1e-6);
            for x in qh.iter_mut() {
                *x *= qscale;
            }
            let beta = sigmoid(b[h]);
            let decay = (w.a[h] * softplus(a[h] + w.dt_bias[h])).exp();
            let sh = &mut s[h * kd * vd..(h + 1) * kd * vd];
            for x in sh.iter_mut() {
                *x *= decay;
            }
            let mut kv = vec![0.0f32; vd];
            for kk in 0..kd {
                let kkv = kh[kk];
                let row = &sh[kk * vd..kk * vd + vd];
                for d in 0..vd {
                    kv[d] += kkv * row[d];
                }
            }
            let delta: Vec<f32> = (0..vd).map(|d| (vh[d] - kv[d]) * beta).collect();
            for kk in 0..kd {
                let kkv = kh[kk];
                let row = &mut sh[kk * vd..kk * vd + vd];
                for d in 0..vd {
                    row[d] += kkv * delta[d];
                }
            }
            let oh = &mut out[h * vd..h * vd + vd];
            for kk in 0..kd {
                let qv = qh[kk];
                let row = &sh[kk * vd..kk * vd + vd];
                for d in 0..vd {
                    oh[d] += qv * row[d];
                }
            }
        }
        for h in 0..nv {
            let oh = &mut out[h * vd..h * vd + vd];
            let n = rmsnorm(oh, &w.ssm_norm, c.eps);
            let zh = &z[h * vd..h * vd + vd];
            for d in 0..vd {
                oh[d] = n[d] * silu(zh[d]);
            }
        }
        w.out.mul(None, &out, c.d_inner, ne)
    }

    /// Fully-recorded GPU attention mixer (no host compute): rmsnorm → q/k/v GEMV → deinterleave
    /// q/gate (strided copies) → qk-norm+rope → GQA softmax (f16 cache) → sigmoid output gate → out
    /// proj. Two command buffers: stage 1 produces the normed k/v read back into the host f32 cache
    /// (memory bookkeeping, not compute); stage 2 runs the attention + gate + out proj. The host f32
    /// KV cache (re-uploaded f16 per token) is a perf wart, not host compute — the resident forward
    /// will make it GPU-resident.
    #[allow(clippy::too_many_arguments)]
    /// Fully on-device attention for one token (no host round-trip): rmsnorm → q/k/v GEMV →
    /// deinterleave q/gate → qk-norm+RoPE (Q→`qf16`, K→`kv.k` at row `pos`) → store V into `kv.v` at
    /// row `pos` → GQA softmax over the device cache (`attention_kv`, kv_len `pos+1`) → sigmoid output
    /// gate → out proj. One command buffer; the f16 KV cache is appended in place (no per-token
    /// re-upload), which is what the dense `Llama` decode path does.
    fn attn_mixer_gpu(
        &self,
        be: &VulkanBackend,
        w: &AttnLayer,
        gpu: &AttnGpuW,
        kv: &DevKv,
        hidden: &[f32],
        pos: usize,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (ne, hd) = (c.n_embd, c.head_dim);
        let (nh, nkv) = (c.n_head, c.n_kv);
        let kvrow = nkv * hd;
        let alloc = |n: usize| {
            be.alloc((n * 4).max(4), BufferUsage::Activations)
                .expect("alloc")
        };
        // qk_norm_rope writes f16 (feeds attention_kv / the f16 KV cache directly): Q→qf16, K straight
        // into the device cache at row `pos`. qd/kd are the f32 projection outputs it reads.
        let alloc_f16 = |n: usize| {
            be.alloc((n * 2).max(4), BufferUsage::Activations)
                .expect("alloc f16")
        };
        let hb = dev(be, hidden);
        let xn = alloc(ne);
        let qg = alloc(nh * 2 * hd);
        let qd = alloc(nh * hd);
        let qf16 = alloc_f16(nh * hd);
        let gateb = alloc(nh * hd);
        let kd = alloc(nkv * hd);
        let vb = alloc(nkv * hd);
        let ob = alloc(nh * hd);
        let og = alloc(nh * hd);
        let res = alloc(ne);
        let t = pos + 1; // cached length after appending this token
        {
            let rec = be.recorder().expect("recorder");
            rec.rmsnorm(
                hb.as_ref(),
                gpu.attn_norm.as_ref(),
                xn.as_ref(),
                1,
                ne,
                c.eps,
            );
            // attn_q outputs query+gate INTERLEAVED per head: [h q(hd) | h gate(hd) | …].
            w.q.record(&rec, xn.as_ref(), qg.as_ref(), ne, nh * 2 * hd);
            w.k.record(&rec, xn.as_ref(), kd.as_ref(), ne, nkv * hd);
            w.v.record(&rec, xn.as_ref(), vb.as_ref(), ne, nkv * hd);
            for h in 0..nh {
                let qoff = h * 2 * hd * 4;
                rec.copy(qg.as_ref(), qoff, qd.as_ref(), h * hd * 4, hd * 4);
                rec.copy(
                    qg.as_ref(),
                    qoff + hd * 4,
                    gateb.as_ref(),
                    h * hd * 4,
                    hd * 4,
                );
            }
            // Q → qf16 at row 0; K → the device cache at row `pos` (out_base=pos).
            rec.qk_norm_rope(
                qd.as_ref(),
                gpu.q_norm.as_ref(),
                qf16.as_ref(),
                1,
                nh,
                hd,
                c.rope_dim,
                c.rope_theta,
                pos,
                0,
                c.eps,
                None,
            );
            rec.qk_norm_rope(
                kd.as_ref(),
                gpu.k_norm.as_ref(),
                kv.k.as_ref(),
                1,
                nkv,
                hd,
                c.rope_dim,
                c.rope_theta,
                pos,
                pos,
                c.eps,
                None,
            );
            // V (f32) → device cache (f16) at row `pos`.
            rec.store_f16(vb.as_ref(), kv.v.as_ref(), kvrow, pos * kvrow);
            // GQA softmax over the device cache (kv_len = t; pos_offset = pos; full causal; 1/√hd).
            rec.attention_kv(
                qf16.as_ref(),
                kv.k.as_ref(),
                kv.v.as_ref(),
                ob.as_ref(),
                1,
                t,
                nh,
                nkv,
                hd,
                pos,
                0,
                0.0,
            );
            rec.mul_sigmoid(ob.as_ref(), gateb.as_ref(), og.as_ref(), nh * hd);
            w.out.record(&rec, og.as_ref(), res.as_ref(), nh * hd, ne);
            rec.finish().expect("finish");
        }
        read(be, res.as_ref(), ne)
    }

    fn attn_mixer(
        &self,
        w: &AttnLayer,
        hidden: &[f32],
        kc: &mut Vec<f32>,
        vc: &mut Vec<f32>,
        dev: Option<&DevKv>,
        pos: usize,
    ) -> Vec<f32> {
        if let (Some(be), Some(gpu), Some(dev)) = (self.be.as_ref(), w.gpu.as_ref(), dev) {
            return self.attn_mixer_gpu(be, w, gpu, dev, hidden, pos);
        }
        // --- CPU reference oracle (Q35_CPU=1; no backend) ---
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
                         // GQA softmax attention over the cache (CPU reference; the GPU path lives in
                         // attn_mixer_gpu). Output gate (sigmoid) applied after.
        let mut out = {
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
            }
            out
        };
        // per-head sigmoid output gate (host; tiny)
        for h in 0..nh {
            let oh = &mut out[h * hd..h * hd + hd];
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
    render_chat_messages(path, &[("user", user)])
}

/// Render a multi-turn conversation `(role, content)` through the qwen35 GGUF's own jinja chat
/// template — the [`crate::model::ChatModel::render`] primitive for the qwen35 GPU + CPU paths, so
/// the shared [`crate::model::Chat`] can drive a history-based REPL. Errors if the GGUF has no usable
/// `tokenizer.chat_template`.
pub fn render_chat_messages(path: &std::path::Path, messages: &[(&str, &str)]) -> Result<String> {
    let g = Gguf::open(path).map_err(|e| anyhow!("open gguf: {e}"))?;
    let tok = crate::build_tokenizer(&g)?;
    let eos = g.metadata().u64("tokenizer.ggml.eos_token_id").unwrap_or(2) as u32;
    infr_chat::render_chat_jinja(&g, &tok, eos, messages, true).ok_or_else(|| {
        anyhow!(
            "model GGUF has no usable chat template (no `tokenizer.chat_template`, or it failed to \
             render — set INFR_DEBUG_CHAT=1 for details)."
        )
    })
}

/// Greedy pure-CPU generation for qwen35 / Qwen3-Next on the agnostic seam (no Vulkan). `prompt` is
/// the already-formatted text (see [`render_chat`]); returns timing/counts, text streams via `on_piece`.
/// Turns a native-dtype GGUF tensor into a backend weight buffer (upload or zero-copy map).
pub type BindWeight<'a> = &'a dyn Fn(&dyn Backend, TensorBytes) -> Result<Box<dyn Buffer>>;

/// Load-once Qwen3-Next (qwen35) seam model: a backend + the model's native-dtype weight buffers +
/// tokenizer/config, constructed ONCE and reused across `generate` calls. This is the SINGLE
/// generation engine behind `infr run` (via [`crate::model::Qwen35Chat`] /
/// on any backend) AND `infr bench` — bench times exactly what run executes, so a
/// production-path change can never silently leave the bench measuring a dead path again. Only the
/// per-conversation state (conv history, DeltaNet S, KV cache, sized to prompt+n) is allocated per
/// call.
pub struct SeamModel {
    be: Box<dyn Backend>,
    g: Gguf,
    c: Cfg,
    token_embd: Vec<f32>,
    vocab: usize,
    tok: tokenizers::Tokenizer,
    eos: Option<u32>,
    im_end: Option<u32>,
    wbufs: Vec<Box<dyn Buffer>>,
    wspecs: Vec<(DType, usize)>,
}

impl SeamModel {
    /// Open the GGUF and upload every weight tensor in its native dtype through `bind_weight`
    /// (Vulkan/Metal: alloc+upload; CPU: zero-copy mmap). The upload order MUST equal the `wpush`
    /// order in the graph build.
    pub fn load(
        be: Box<dyn Backend>,
        bind_weight: BindWeight,
        path: &std::path::Path,
    ) -> Result<Self> {
        let gg = Gguf::open(path).map_err(|e| anyhow!("open gguf: {e}"))?;
        let g = &gg;
        let c = Cfg::from_gguf(g)?;
        let (token_embd, te_shape) = load_tensor_dequant(g, "token_embd.weight")?;
        let vocab = te_shape[1];
        let tok = crate::build_tokenizer(g)?;
        let eos = g
            .metadata()
            .u64("tokenizer.ggml.eos_token_id")
            .map(|x| x as u32);
        let im_end = tok.token_to_id("<|im_end|>");
        let attn = |i: usize| c.is_attn_layer(i);
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
            wbufs.push(bind_weight(be.as_ref(), tb)?);
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

        Ok(SeamModel {
            be,
            g: gg,
            c,
            token_embd,
            vocab,
            tok,
            eos,
            im_end,
            wbufs,
            wspecs,
        })
    }

    /// Vulkan (the production GPU path): weights uploaded raw to VRAM, dequantized in-kernel.
    pub fn load_vulkan(path: &std::path::Path) -> Result<Self> {
        let be = VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        Self::load(
            Box::new(be),
            &|be, tb| {
                let buf = be
                    .alloc(tb.len().max(1), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                be.upload(buf.as_ref(), &tb).map_err(|e| anyhow!("{e}"))?;
                Ok(buf)
            },
            path,
        )
    }

    /// CPU reference backend: weights mapped zero-copy from the GGUF mmap.
    pub fn load_cpu(path: &std::path::Path) -> Result<Self> {
        Self::load(
            Box::new(CpuBackend::new()),
            &|_, tb| Ok(CpuBackend::new().map_weight(tb)),
            path,
        )
    }

    /// Reference Metal backend (Apple GPU).
    #[cfg(target_os = "macos")]
    pub fn load_metal(path: &std::path::Path) -> Result<Self> {
        let be = infr_metal::MetalBackend::new().map_err(|e| anyhow!("metal init: {e}"))?;
        Self::load(
            Box::new(be),
            &|be, tb| {
                let buf = be
                    .alloc(tb.len().max(1), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                be.upload(buf.as_ref(), &tb).map_err(|e| anyhow!("{e}"))?;
                Ok(buf)
            },
            path,
        )
    }

    /// Generate a completion for the already-rendered `prompt` on the loaded model: batched/chunked
    /// prefill + per-token decode (greedy), streaming text to `on_piece`. Per-conversation state is
    /// allocated fresh (sized to prompt+n).
    pub fn generate(
        &self,
        prompt: &str,
        n: usize,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let be = self.be.as_ref();
        let g = &self.g;
        let c = &self.c;
        let token_embd = &self.token_embd;
        let vocab = self.vocab;
        let tok = &self.tok;
        let (eos, im_end) = (self.eos, self.im_end);
        let wspecs = &self.wspecs;
        let wbufs = &self.wbufs;
        let ne = c.n_embd;
        let cc = c.conv_channels();
        let di = c.d_inner;
        let (nk, kd) = (c.num_k_heads(), c.head_k_dim());
        let (nv, vd) = (c.num_v_heads(), c.head_v_dim());
        let key_dim = nk * kd;
        let (nh, nkv, hd) = (c.n_head, c.n_kv, c.head_dim);
        let eps = c.eps;
        let kk = c.d_conv;
        let enc = tok
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
        let max_ctx = prompt_ids.len() + n + 1;

        if std::env::var("INFR_Q35_TIMING").is_ok() {
            eprintln!("[q35dims] ne={ne} nv={nv} nk={nk} kd={kd} vd={vd} cc={cc} di={di} n_layer={} nh={nh} nkv={nkv} hd={hd}", c.n_layer);
        }
        let attn = |i: usize| c.is_attn_layer(i);
        let n_ff_of = |i: usize| -> Result<usize> {
            Ok(g.tensors()
                .iter()
                .find(|t| t.name == format!("blk.{i}.ffn_up.weight"))
                .context("ffn_up")?
                .shape[1])
        };

        // ── persistent state buffers, one set per layer by kind ──────────────────────
        // The recurrence REQUIRES these start at zero (conv history, DeltaNet S, KV cache). `be.alloc`
        // returns uninitialized memory on the GPU backends (only the CPU's Vec happens to be zeroed), so
        // zero them explicitly via an upload — else the first token attends/accumulates garbage state.
        let zalloc = |n: usize| -> Result<Box<dyn Buffer>> {
            let buf = be
                .alloc(n * 4, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(buf.as_ref(), bytemuck::cast_slice(&vec![0f32; n]))
                .map_err(|e| anyhow!("{e}"))?;
            Ok(buf)
        };
        let mut conv_bufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
        let mut s_bufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
        let mut k_bufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
        let mut v_bufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
        for i in 0..c.n_layer {
            if attn(i) {
                k_bufs.push(Some(zalloc(max_ctx * nkv * hd)?));
                v_bufs.push(Some(zalloc(max_ctx * nkv * hd)?));
                conv_bufs.push(None);
                s_bufs.push(None);
            } else {
                conv_bufs.push(Some(zalloc((kk - 1) * cc)?));
                s_bufs.push(Some(zalloc(nv * kd * vd)?));
                k_bufs.push(None);
                v_bufs.push(None);
            }
        }

        // ── per-step IO ───────────────────────────────────────────────────────────────
        // Prompt tokens are ingested in chunks of up to CHUNK through ONE graph build each (instead of
        // one build per token); the recurrent conv/DeltaNet ops scan the chunk internally and the KV
        // cache is written for the whole chunk. Decode then runs one token (rows=1) at a time. The
        // hidden/pos input buffers are allocated per call sized to the chunk (the CPU backend writes
        // F32 inputs back, so their length must match the graph input exactly).
        // 512 beats 128 by ~27% end-to-end: the recurrent scans (DeltaNet/conv1d) cost the same total
        // either way, but bigger chunks give the GEMMs more row-tiles (occupancy) and amortize the
        // per-chunk build+compile+submit. INFR_Q35_CHUNK overrides for experiments.
        let chunk: usize = std::env::var("INFR_Q35_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&c: &usize| c > 0)
            .unwrap_or(512);
        let logits_buf = be
            .alloc(vocab * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;

        let f32d = |x: usize| TensorDesc::new(vec![x], DType::F32);
        let scale = 1.0 / (hd as f32).sqrt();

        // Build the graph that ingests `rows` tokens starting at absolute position `pos`
        // (kv_len = pos+rows). rows=1 is one decode token; rows>1 is a prefill chunk. Only the LAST
        // row's logits are produced (the next-token prediction).
        let build = |pos: usize,
                     rows: usize|
         -> Result<(Graph, TensorId, TensorId, Vec<TensorId>, TensorId)> {
            let mut gr = Graph::new();
            let hidden = gr.input(f32d(rows * ne));
            let positions = gr.input(TensorDesc::new(vec![rows], DType::I32));
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

            // scratch — every activation buffer holds the whole `rows`-token chunk.
            let xn = gr.internal(f32d(rows * ne));
            let hn = gr.internal(f32d(rows * ne));
            let sub = gr.internal(f32d(rows * ne));
            let max_ff = (0..c.n_layer)
                .map(|i| n_ff_of(i).unwrap_or(0))
                .max()
                .unwrap_or(0);
            let gbuf = gr.internal(f32d(rows * max_ff));
            let ubuf = gr.internal(f32d(rows * max_ff));
            let actbuf = gr.internal(f32d(rows * max_ff));
            // linear-mixer scratch
            let qkvbuf = gr.internal(f32d(rows * cc));
            let zbuf = gr.internal(f32d(rows * di));
            let convout = gr.internal(f32d(rows * cc));
            let qbuf = gr.internal(f32d(rows * key_dim));
            let kbuf = gr.internal(f32d(rows * key_dim));
            let vbuf = gr.internal(f32d(rows * nv * vd));
            let bbuf = gr.internal(f32d(rows * nv));
            let abuf = gr.internal(f32d(rows * nv));
            let dnout = gr.internal(f32d(rows * nv * vd));
            // attn scratch
            let qg = gr.internal(f32d(rows * nh * 2 * hd));
            let qa = gr.internal(f32d(rows * nh * hd));
            let gate_a = gr.internal(f32d(rows * nh * hd));
            let ka = gr.internal(f32d(rows * nkv * hd));
            // K-norm output MUST be a dedicated F16 scratch so the Vulkan `kv_write_peephole` fuses the
            // QkNormRope+WriteKv into a direct-to-cache write (the peephole only fires on an F16 Internal
            // dst). Reusing the F32 `ka` here would make QkNormRope write f16 into an f32 buffer with NO
            // fusion, so the following WriteKv's `store_f16` reads those f16 bytes AS f32 → garbage cache.
            let k16 = gr.internal(TensorDesc::new(vec![rows * nkv * hd], DType::F16));
            let va = gr.internal(f32d(rows * nkv * hd));
            let attno = gr.internal(f32d(rows * nh * hd));

            // rmsn/lin run over the whole `rows`-token chunk (RmsNorm.rows / Linear.m = rows).
            let rmsn = |gr: &mut Graph, x: TensorId, w: TensorId, dst: TensorId| {
                gr.push(Op::RmsNorm {
                    x,
                    weight: w,
                    dst,
                    rows: rows as u32,
                    dim: ne as u32,
                    eps,
                });
            };
            let lin = |gr: &mut Graph,
                       x: TensorId,
                       w: TensorId,
                       dst: TensorId,
                       inf: usize,
                       outf: usize| {
                gr.push(Op::Linear {
                    x,
                    weight: w,
                    dst,
                    m: rows as u32,
                    in_f: inf as u32,
                    out_f: outf as u32,
                });
            };

            let __maxl = std::env::var("INFR_Q35_MAXLAYERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(usize::MAX);
            for (li, lh) in layers.iter().enumerate() {
                if li >= __maxl {
                    break;
                }
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
                            rows: rows as u32,
                            channels: cc as u32,
                            kernel: kk as u32,
                        });
                        // split conv_out [rows, cc=q|k|v] → packed [rows, *] q / k / v (strided per token).
                        gr.push(Op::CopyStrided {
                            src: convout,
                            src_off: 0,
                            src_stride: cc as u32,
                            dst: qbuf,
                            dst_off: 0,
                            dst_stride: key_dim as u32,
                            rows: rows as u32,
                            n: key_dim as u32,
                        });
                        gr.push(Op::CopyStrided {
                            src: convout,
                            src_off: key_dim as u32,
                            src_stride: cc as u32,
                            dst: kbuf,
                            dst_off: 0,
                            dst_stride: key_dim as u32,
                            rows: rows as u32,
                            n: key_dim as u32,
                        });
                        gr.push(Op::CopyStrided {
                            src: convout,
                            src_off: 2 * key_dim as u32,
                            src_stride: cc as u32,
                            dst: vbuf,
                            dst_off: 0,
                            dst_stride: (nv * vd) as u32,
                            rows: rows as u32,
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
                            rows: rows as u32,
                            n_vhead: nv as u32,
                            n_khead: nk as u32,
                            head_k: kd as u32,
                            head_v: vd as u32,
                            eps: 1e-6,
                        });
                        // silu-gated RMSNorm per v-head: rmsnorm(out, ssm_norm) then * silu(z)
                        gr.push(Op::QkNorm {
                            x: dnout,
                            weight: w.ssm_norm,
                            dst: dnout,
                            rows: rows as u32,
                            n_head: nv as u32,
                            head_dim: vd as u32,
                            eps,
                        });
                        gr.push(Op::GatedAct {
                            gate: zbuf,
                            up: dnout,
                            dst: dnout,
                            rows: rows as u32,
                            nff: (nv * vd) as u32,
                            act: Activation::Silu,
                            up_off: 0,
                        });
                        lin(&mut gr, dnout, w.out, sub, di, ne);
                        gr.push(Op::Add {
                            a: hidden,
                            b: sub,
                            dst: hidden,
                            n: (rows * ne) as u32,
                        });
                        // FFN
                        rmsn(&mut gr, hidden, w.post_norm, hn);
                        lin(&mut gr, hn, w.ffn_gate, gbuf, ne, w.n_ff);
                        lin(&mut gr, hn, w.ffn_up, ubuf, ne, w.n_ff);
                        gr.push(Op::GatedAct {
                            gate: gbuf,
                            up: ubuf,
                            dst: actbuf,
                            rows: rows as u32,
                            nff: w.n_ff as u32,
                            act: Activation::Silu,
                            up_off: 0,
                        });
                        lin(&mut gr, actbuf, w.ffn_down, sub, w.n_ff, ne);
                        gr.push(Op::Add {
                            a: hidden,
                            b: sub,
                            dst: hidden,
                            n: (rows * ne) as u32,
                        });
                    }
                    Q35LayerH::Attn(w) => {
                        rmsn(&mut gr, hidden, w.attn_norm, xn);
                        // q proj outputs q+gate interleaved per head [h: q(hd), gate(hd)].
                        lin(&mut gr, xn, w.q, qg, ne, nh * 2 * hd);
                        // split interleaved per-head [q(hd) | gate(hd)] → packed qa / gate_a (strided).
                        for h in 0..nh {
                            gr.push(Op::CopyStrided {
                                src: qg,
                                src_off: (h * 2 * hd) as u32,
                                src_stride: (nh * 2 * hd) as u32,
                                dst: qa,
                                dst_off: (h * hd) as u32,
                                dst_stride: (nh * hd) as u32,
                                rows: rows as u32,
                                n: hd as u32,
                            });
                            gr.push(Op::CopyStrided {
                                src: qg,
                                src_off: (h * 2 * hd + hd) as u32,
                                src_stride: (nh * 2 * hd) as u32,
                                dst: gate_a,
                                dst_off: (h * hd) as u32,
                                dst_stride: (nh * hd) as u32,
                                rows: rows as u32,
                                n: hd as u32,
                            });
                        }
                        lin(&mut gr, xn, w.k, ka, ne, nkv * hd);
                        lin(&mut gr, xn, w.v, va, ne, nkv * hd);
                        // per-head q/k norm + RoPE — fused (qwen35 always has q/k-norm).
                        gr.push(Op::QkNormRope {
                            x: qa,
                            weight: w.q_norm,
                            positions,
                            dst: qa,
                            rows: rows as u32,
                            n_head: nh as u32,
                            head_dim: hd as u32,
                            rope_dim: c.rope_dim as u32,
                            theta: c.rope_theta,
                            eps,
                            freq_factors: None,
                        });
                        gr.push(Op::QkNormRope {
                            x: ka,
                            weight: w.k_norm,
                            positions,
                            dst: k16, // F16 scratch → peephole fuses this + the WriteKv below into a
                            // single direct-to-cache qk-norm+rope (see the k16 decl).
                            rows: rows as u32,
                            n_head: nkv as u32,
                            head_dim: hd as u32,
                            rope_dim: c.rope_dim as u32,
                            theta: c.rope_theta,
                            eps,
                            freq_factors: None,
                        });
                        gr.push(Op::WriteKv {
                            src: k16,
                            cache: w.k_cache,
                            rows: rows as u32,
                            row_stride: (nkv * hd) as u32,
                            pos: pos as u32,
                        });
                        gr.push(Op::WriteKv {
                            src: va,
                            cache: w.v_cache,
                            rows: rows as u32,
                            row_stride: (nkv * hd) as u32,
                            pos: pos as u32,
                        });
                        gr.push(Op::Attention {
                            q: qa,
                            k_cache: w.k_cache,
                            v_cache: w.v_cache,
                            dst: attno,
                            rows: rows as u32,
                            kv_len: (pos + rows) as u32,
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
                            rows: rows as u32,
                            nff: (nh * hd) as u32,
                            act: Activation::Sigmoid,
                            up_off: 0,
                        });
                        lin(&mut gr, attno, w.out, sub, nh * hd, ne);
                        gr.push(Op::Add {
                            a: hidden,
                            b: sub,
                            dst: hidden,
                            n: (rows * ne) as u32,
                        });
                        // FFN
                        rmsn(&mut gr, hidden, w.post_norm, hn);
                        lin(&mut gr, hn, w.ffn_gate, gbuf, ne, w.n_ff);
                        lin(&mut gr, hn, w.ffn_up, ubuf, ne, w.n_ff);
                        gr.push(Op::GatedAct {
                            gate: gbuf,
                            up: ubuf,
                            dst: actbuf,
                            rows: rows as u32,
                            nff: w.n_ff as u32,
                            act: Activation::Silu,
                            up_off: 0,
                        });
                        lin(&mut gr, actbuf, w.ffn_down, sub, w.n_ff, ne);
                        gr.push(Op::Add {
                            a: hidden,
                            b: sub,
                            dst: hidden,
                            n: (rows * ne) as u32,
                        });
                        let _ = li;
                    }
                }
            }
            // Only the LAST token's logits are needed (the next-token prediction). Extract its hidden
            // row, then output-norm + lm_head at rows=1 (avoids a wasteful [rows, vocab] projection).
            let last = gr.internal(f32d(ne));
            gr.push(Op::Copy {
                src: hidden,
                src_off: ((rows - 1) * ne) as u32,
                dst: last,
                dst_off: 0,
                n: ne as u32,
            });
            gr.push(Op::RmsNorm {
                x: last,
                weight: w_out_norm,
                dst: hn,
                rows: 1,
                dim: ne as u32,
                eps,
            });
            gr.push(Op::Linear {
                x: hn,
                weight: w_lm,
                dst: logits,
                m: 1,
                in_f: ne as u32,
                out_f: vocab as u32,
            });

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
        // Build + compile + bind + execute the graph for the `rows` tokens whose embeddings/positions
        // are `emb` (rows*ne) / `posv` (rows), at absolute start position `pos`. Binds fresh hidden/pos
        // input buffers (sized to the chunk), the shared logits buffer, weights, and recurrent state.
        let run_graph = |pos: usize, emb: &[f32], posv: &[i32]| -> Result<()> {
            let rows = posv.len();
            let hidden_buf = be
                .alloc(emb.len() * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(hidden_buf.as_ref(), bytemuck::cast_slice(emb))
                .map_err(|e| anyhow!("{e}"))?;
            let pos_buf = be
                .alloc(posv.len() * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(pos_buf.as_ref(), bytemuck::cast_slice(posv))
                .map_err(|e| anyhow!("{e}"))?;
            let _t_build = std::time::Instant::now();
            let (gr, h_hidden, h_pos, h_bind, h_logits) = build(pos, rows)?;
            let plan = be.compile(&gr).map_err(|e| anyhow!("{e}"))?;
            let _bc = _t_build.elapsed();
            let _t_exec = std::time::Instant::now();
            let mut b = Bindings::new();
            b.bind(h_hidden, hidden_buf.as_ref());
            b.bind(h_pos, pos_buf.as_ref());
            b.bind(h_logits, logits_buf.as_ref());
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
            let r = be.execute(plan.as_ref(), &b).map_err(|e| anyhow!("{e}"));
            if std::env::var("INFR_Q35_TIMING").is_ok() {
                eprintln!(
                    "[q35timing] rows={rows} build+compile={:.2}ms execute={:.2}ms",
                    _bc.as_secs_f64() * 1e3,
                    _t_exec.elapsed().as_secs_f64() * 1e3
                );
            }
            r
        };

        let mut outs: Vec<u32> = Vec::new();
        let mut logits = vec![0f32; vocab];
        let mut printed = 0usize; // streaming detok cursor
        let plen = prompt_ids.len();

        // ── prefill: ingest the prompt in chunks of ≤CHUNK (one graph per chunk) ──
        let prompt_t0 = std::time::Instant::now();
        let mut cpos = 0usize;
        while cpos < plen {
            let rows = (plen - cpos).min(chunk);
            let mut emb = vec![0f32; rows * ne];
            for r in 0..rows {
                let tid = prompt_ids[cpos + r] as usize;
                emb[r * ne..r * ne + ne].copy_from_slice(&token_embd[tid * ne..tid * ne + ne]);
            }
            let posv: Vec<i32> = (0..rows).map(|r| (cpos + r) as i32).collect();
            run_graph(cpos, &emb, &posv)?;
            cpos += rows;
        }
        let prompt_t = prompt_t0.elapsed();
        // The last chunk's last row predicts the first generated token.
        be.download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
            .map_err(|e| anyhow!("{e}"))?;
        if std::env::var("INFR_Q35_DUMP").is_ok() {
            let am = argmax(&logits);
            let s: f32 = logits.iter().map(|x| x.abs()).sum::<f32>() / logits.len() as f32;
            eprintln!(
                "[q35dump] argmax={am} logit[am]={:.4} mean|logit|={s:.4} first6={:?}",
                logits[am as usize],
                &logits[..6]
            );
        }

        // ── decode: one token at a time (rows=1), feeding the last prediction back ──
        let decode_t0 = std::time::Instant::now();
        let mut pos = plen;
        let mut decode_n = 0usize;
        loop {
            let next = argmax(&logits);
            // Stop on EOS / <|im_end|> before emitting the stop token (chat turn boundary).
            // INFR_Q35_IGNORE_EOS keeps generating to the cap (benchmarks need a fixed tg count).
            if (Some(next) == eos || (im_end.is_some() && Some(next) == im_end))
                && std::env::var("INFR_Q35_IGNORE_EOS").is_err()
            {
                break;
            }
            crate::stream_token(tok, &mut outs, &mut printed, next, &mut on_piece);
            decode_n += 1;
            if outs.len() >= n {
                break;
            }
            // feed `next` at absolute position `pos`, predict the following token.
            let emb = &token_embd[next as usize * ne..next as usize * ne + ne];
            run_graph(pos, emb, &[pos as i32])?;
            be.download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
                .map_err(|e| anyhow!("{e}"))?;
            pos += 1;
        }
        let decode_t = decode_t0.elapsed();

        // The text streamed out via `on_piece`; return only timing/counts.
        Ok(crate::GenStats {
            n_prompt: plen,
            prompt_secs: prompt_t.as_secs_f64(),
            n_gen: decode_n,
            decode_secs: decode_t.as_secs_f64(),
        })
    }
}

/// Qwen3-Next decode on the CPU reference backend (weights mapped zero-copy from the GGUF mmap).
pub fn generate_cpu(
    path: &std::path::Path,
    prompt: &str,
    n: usize,
    on_piece: impl FnMut(&str),
) -> Result<crate::GenStats> {
    SeamModel::load_cpu(path)?.generate(prompt, n, on_piece)
}

/// Qwen3-Next decode on the reference Metal backend (weights uploaded to Metal buffers in their
/// native GGUF dtype; the backend dequantizes lazily). The Apple-GPU twin of [`generate_cpu`].
#[cfg(target_os = "macos")]
pub fn generate_metal(
    path: &std::path::Path,
    prompt: &str,
    n: usize,
    on_piece: impl FnMut(&str),
) -> Result<crate::GenStats> {
    SeamModel::load_metal(path)?.generate(prompt, n, on_piece)
}

/// Qwen3-Next on the Vulkan GPU through the AGNOSTIC SEAM (weights uploaded to VRAM in their native
/// GGUF dtype; the backend dequantizes in-kernel). Unlike the bespoke [`Model::forward`], this uses
/// the batched/chunked prefill (the whole prompt flows through a few graph builds instead of one per
/// token). The Vulkan twin of [`generate_cpu`].
///
/// EXPERIMENTAL / WIP: this runs and is FAST (~590 pp / ~280 tg t/s) but currently returns
/// INCORRECT output — a qwen35-specific correctness bug in the Vulkan seam path (the dense seam and
/// the bespoke qwen35 GPU path both work). Not wired into production. See `gpu_seam_qwen35` (ignored)
/// and the qwen35-batched-prefill memory for the debug plan.
pub fn generate_vulkan(
    path: &std::path::Path,
    prompt: &str,
    n: usize,
    on_piece: impl FnMut(&str),
) -> Result<crate::GenStats> {
    SeamModel::load_vulkan(path)?.generate(prompt, n, on_piece)
}

/// Open the GGUF at `path` and build the bespoke qwen35 [`Model`] (GPU-resident forward, or the
/// `Q35_CPU=1` oracle). CLI/bench convenience so callers can drive the per-token [`Model::forward`]
/// for timing without depending on `infr_gguf` directly.
pub fn load_path(path: &std::path::Path) -> Result<Model> {
    let g = Gguf::open(path).map_err(|e| anyhow!("{e}"))?;
    Model::load(&g)
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
/// Generate the assistant reply for an already-rendered `prompt` on the qwen35 / Qwen3-Next GPU path
/// (the bespoke per-token hybrid forward). Returns `(n_prompt, n_gen)`; text streams via `on_piece`.
/// This is the [`crate::model::ChatModel::generate`] primitive for the qwen35 GPU backend — rendering
/// happens in [`render_chat_messages`], so the shared [`crate::model::Chat`] owns the history.
///
/// FOLLOW-UP: this reloads the GGUF + rebuilds the model on every turn (a pre-existing wart carried
/// over from the one-shot path). A load-once persistent qwen35 model behind the trait would drop the
/// per-turn reload; deferred to keep this refactor scoped to the shared orchestration.
pub fn generate_chat(
    path: &std::path::Path,
    prompt: &str,
    max_new: usize,
    mut on_piece: impl FnMut(&str),
) -> Result<(usize, usize)> {
    // Fast path: the batched/chunked GPU seam (full-GPU forward incl. attention) — ~7x prefill over
    // the bespoke per-token loop below. Escape hatch INFR_Q35_BESPOKE=1 forces the per-token path.
    if std::env::var("INFR_Q35_BESPOKE").is_err() {
        if let Ok(m) = SeamModel::load_vulkan(path) {
            let stats = m.generate(prompt, max_new, on_piece)?;
            return Ok((stats.n_prompt, stats.n_gen));
        }
    }
    let g = Gguf::open(path).map_err(|e| anyhow!("{e}"))?;
    let m = Model::load(&g)?;
    let tok = crate::build_tokenizer(&g)?;
    let eos = g
        .metadata()
        .u64("tokenizer.ggml.eos_token_id")
        .map(|x| x as u32);
    let im_end = tok.token_to_id("<|im_end|>");
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

    /// Locate the Qwen3.5-0.8B GGUF in the HF Hub cache (or `INFR_TEST_MODEL`), or `None` if it isn't
    /// present (the test self-skips).
    fn model_path() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
            return Some(std::path::PathBuf::from(p));
        }
        let hub = std::env::var("HOME").ok()? + "/.cache/huggingface/hub";
        let base = format!("{hub}/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots");
        std::fs::read_dir(&base).ok()?.find_map(|e| {
            let f = e.ok()?.path().join("Qwen3.5-0.8B-Q4_K_M.gguf");
            f.exists().then_some(f)
        })
    }

    #[test]
    fn loads_and_dims() {
        let Some(path) = model_path() else {
            eprintln!("skip: Qwen3.5-0.8B not present");
            return;
        };
        let g = Gguf::open(&path).unwrap();
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
    fn greedy_generate() {
        let Some(path) = model_path() else {
            eprintln!("skip: Qwen3.5-0.8B not present");
            return;
        };
        let g = Gguf::open(&path).unwrap();
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
    fn seam_cpu_matches_oracle() {
        let Some(path) = model_path() else {
            eprintln!("skip: Qwen3.5-0.8B not present");
            return;
        };
        let g = Gguf::open(&path).unwrap();
        let prompt = "The capital of France is";
        let n = 16;
        std::env::set_var("Q35_CPU", "1");
        let oracle = generate(&g, prompt, n).unwrap();
        let mut seam = String::new();
        generate_cpu(&path, prompt, n, |p| seam.push_str(p)).unwrap();
        println!("ORACLE: {oracle:?}\nSEAM:   {seam:?}");
        assert_eq!(
            seam, oracle,
            "qwen35 seam CPU must match the bespoke CPU oracle"
        );
    }

    /// Stable FNV-1a-64 (std `DefaultHasher` isn't stable across toolchains).
    fn fnv1a(s: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// GPU forward golden-hash lock (GPU linear projections + GPU conv1d/DeltaNet SSM; attention is
    /// still on the CPU). We DON'T compare to the CPU oracle — that's precision-brittle (f32 CPU vs
    /// the GPU f16/native kernels). Instead the GPU output is locked by hash and read for coherence;
    /// refresh with `QWEN35_BLESS=1` (prints the hash + text).
    #[test]
    fn gpu_golden_qwen35() {
        let Some(path) = model_path() else {
            eprintln!("skip: Qwen3.5-0.8B not present");
            return;
        };
        if !crate::gpu_available() {
            eprintln!("skip: no Vulkan GPU");
            return;
        }
        let g = Gguf::open(&path).unwrap();
        // Render through the model's chat template so the instruct model answers coherently (a raw
        // completion degenerates on the 0.8B). Greedy via the GPU forward.
        let prompt = render_chat(&path, "What is bash? Answer briefly.").unwrap();
        let out = generate(&g, &prompt, 48).unwrap();
        let h = fnv1a(&out);
        if std::env::var("QWEN35_BLESS").is_ok() {
            println!("gpu golden: 0x{h:016x}  // {out:?}");
        } else {
            assert_eq!(
                h, 0x8628d34de5890fb9,
                "qwen35 GPU golden changed\n  out: {out:?}"
            );
        }
    }

    /// Bisection: run the seam on CPU (correct) then Vulkan with INFR_Q35_DUMP=1 and a chosen
    /// INFR_Q35_MAXLAYERS=N — compare the two `[q35dump]` argmax/logit lines to find the first layer
    /// count at which Vulkan diverges from CPU (localizes the wiring bug).
    #[test]
    #[ignore = "manual bisection (set INFR_Q35_DUMP + INFR_Q35_MAXLAYERS)"]
    fn q35_bisect() {
        let Some(path) = model_path() else {
            return;
        };
        if !crate::gpu_available() {
            return;
        }
        let prompt = render_chat(&path, "Hi").unwrap();
        eprintln!("=== CPU seam ===");
        let _ = generate_cpu(&path, &prompt, 1, |_| {});
        eprintln!("=== Vulkan seam ===");
        let _ = generate_vulkan(&path, &prompt, 1, |_| {});
    }

    /// The batched/chunked-prefill SEAM on Vulkan (vs the bespoke per-token `generate` above). FAST
    /// (~450 pp / ~250 tg t/s vs the bespoke ~65/71) AND correct — the K-norm output now writes an
    /// F16 scratch so the Vulkan kv_write_peephole fuses QkNormRope+WriteKv (an F32 K-norm dst broke
    /// fusion → store_f16 read the f16 bytes as f32 → garbage K cache). This is the production GPU
    /// path (see `generate_chat`). See the qwen35-batched-prefill memory.
    #[test]
    #[ignore = "requires the Qwen3.5-0.8B GGUF + a Vulkan GPU"]
    fn gpu_seam_qwen35() {
        let Some(path) = model_path() else {
            eprintln!("skip: Qwen3.5-0.8B not present");
            return;
        };
        if !crate::gpu_available() {
            eprintln!("skip: no Vulkan GPU");
            return;
        }
        // Larger-prompt bench sample (~hundreds of tokens) so prefill t/s isn't dominated by fixed
        // per-chunk overhead. INFR_Q35_BENCH=1 blows the prompt up + skips the coherence assert.
        let (msg, ntok): (String, usize) = if std::env::var("INFR_Q35_BENCH").is_ok() {
            let filler = "The quick brown fox jumps over the lazy dog. ".repeat(80);
            (format!("Summarize this text in one word:\n{filler}"), 128)
        } else {
            ("What is bash? Answer briefly.".into(), 48)
        };
        let prompt = render_chat(&path, &msg).unwrap();
        let mut out = String::new();
        let stats = generate_vulkan(&path, &prompt, ntok, |p| out.push_str(p)).unwrap();
        eprintln!(
            "seam-vulkan: prefill {} tok @ {:.0} t/s | decode {} tok @ {:.1} t/s",
            stats.n_prompt,
            stats.n_prompt as f64 / stats.prompt_secs.max(1e-9),
            stats.n_gen,
            stats.n_gen as f64 / stats.decode_secs.max(1e-9),
        );
        eprintln!("seam-vulkan out: {out:?}");
        if std::env::var("INFR_Q35_BENCH").is_ok() {
            return;
        }
        let distinct = out
            .trim()
            .chars()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(distinct > 5, "seam-vulkan output degenerate: {out:?}");
    }
}
