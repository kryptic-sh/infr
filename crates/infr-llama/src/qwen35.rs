//! Qwen3.5 / Qwen3.6 (`qwen35`, aka Qwen3-Next): hybrid gated-DeltaNet linear-attention + gated
//! full-attention. See `docs/QWEN35.md`. This module is a **CPU reference** (correctness first);
//! the GPU path comes after the math is locked against llama.cpp.
#![allow(dead_code)] // forward pass is built up incrementally on this loader

use crate::load_f32;
use anyhow::{anyhow, bail, Context, Result};
use infr_core::WeightSource;
use infr_gguf::Gguf;

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
        (i + 1) % self.full_attn_interval == 0
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

/// A linear (gated DeltaNet) layer's weights, all dequantized to f32.
struct LinearLayer {
    attn_norm: Vec<f32>, // [n_embd]
    qkv: Vec<f32>,       // [conv_channels, n_embd]  (out,in)
    gate: Vec<f32>,      // [d_inner, n_embd]  (z)
    conv1d: Vec<f32>,    // [conv_channels, d_conv]  (per-channel kernel)
    alpha: Vec<f32>,     // [dt_rank, n_embd]
    beta: Vec<f32>,      // [dt_rank, n_embd]
    a: Vec<f32>,         // [dt_rank]  (= -exp(A_log))
    dt_bias: Vec<f32>,   // [dt_rank]
    ssm_norm: Vec<f32>,  // [head_v_dim]
    out: Vec<f32>,       // [n_embd, d_inner]
    post_norm: Vec<f32>, // [n_embd]
    ffn_gate: Vec<f32>,  // [n_ff, n_embd]
    ffn_up: Vec<f32>,    // [n_ff, n_embd]
    ffn_down: Vec<f32>,  // [n_embd, n_ff]
    n_ff: usize,
}

/// A full-attention layer's weights.
struct AttnLayer {
    attn_norm: Vec<f32>, // [n_embd]
    q: Vec<f32>,         // [n_head*head_dim + d_inner(gate), n_embd]
    k: Vec<f32>,         // [n_kv*head_dim, n_embd]
    v: Vec<f32>,         // [n_kv*head_dim, n_embd]
    q_norm: Vec<f32>,    // [head_dim]
    k_norm: Vec<f32>,    // [head_dim]
    out: Vec<f32>,       // [n_embd, n_head*head_dim]
    post_norm: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
    n_ff: usize,
}

enum Layer {
    Linear(LinearLayer),
    Attn(AttnLayer),
}

/// Full model weights (CPU, f32).
pub struct Model {
    pub cfg: Cfg,
    token_embd: Vec<f32>, // [vocab, n_embd]
    output_norm: Vec<f32>,
    lm_head: Vec<f32>, // [vocab, n_embd]
    layers: Vec<Layer>,
}

impl Model {
    pub fn load(g: &Gguf) -> Result<Self> {
        let mut cfg = Cfg::from_gguf(g)?;
        let (token_embd, te_shape) = load_f32(g, "token_embd.weight")?;
        cfg.vocab = te_shape[1];
        let output_norm = load_f32(g, "output_norm.weight")?.0;
        let lm_head = if g.tensors().iter().any(|t| t.name == "output.weight") {
            load_f32(g, "output.weight")?.0
        } else {
            token_embd.clone() // tied
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let get = |s: &str| -> Result<Vec<f32>> {
                load_f32(g, &p(s)).map(|x| x.0).with_context(|| p(s))
            };
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
                    q: get("attn_q.weight")?,
                    k: get("attn_k.weight")?,
                    v: get("attn_v.weight")?,
                    q_norm: get("attn_q_norm.weight")?,
                    k_norm: get("attn_k_norm.weight")?,
                    out: get("attn_output.weight")?,
                    post_norm: get("post_attention_norm.weight")?,
                    ffn_gate: get("ffn_gate.weight")?,
                    ffn_up: get("ffn_up.weight")?,
                    ffn_down: get("ffn_down.weight")?,
                    n_ff,
                }));
            } else {
                layers.push(Layer::Linear(LinearLayer {
                    attn_norm: get("attn_norm.weight")?,
                    qkv: get("attn_qkv.weight")?,
                    gate: get("attn_gate.weight")?,
                    conv1d: get("ssm_conv1d.weight")?,
                    alpha: get("ssm_alpha.weight")?,
                    beta: get("ssm_beta.weight")?,
                    a: get("ssm_a")?,
                    dt_bias: get("ssm_dt.bias")?,
                    ssm_norm: get("ssm_norm.weight")?,
                    out: get("ssm_out.weight")?,
                    post_norm: get("post_attention_norm.weight")?,
                    ffn_gate: get("ffn_gate.weight")?,
                    ffn_up: get("ffn_up.weight")?,
                    ffn_down: get("ffn_down.weight")?,
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
        matvec(&self.lm_head, ne, c.vocab, &hn)
    }

    fn ffn(
        &self,
        hidden: &[f32],
        norm: &[f32],
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        n_ff: usize,
    ) -> Vec<f32> {
        let ne = self.cfg.n_embd;
        let h2 = rmsnorm(hidden, norm, self.cfg.eps);
        let g = matvec(gate, ne, n_ff, &h2);
        let u = matvec(up, ne, n_ff, &h2);
        let act: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
        matvec(down, n_ff, ne, &act)
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
        let xn = rmsnorm(hidden, &w.attn_norm, c.eps);
        let qkv = matvec(&w.qkv, ne, cc, &xn); // [6144]
        let z = matvec(&w.gate, ne, c.d_inner, &xn); // [2048]

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
        matvec(&w.out, c.d_inner, ne, &out)
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
        let xn = rmsnorm(hidden, &w.attn_norm, c.eps);
        // attn_q outputs query+gate INTERLEAVED PER HEAD: [h0 q(hd) h0 gate(hd) | h1 q gate | …].
        let qg = matvec(&w.q, ne, nh * 2 * hd, &xn);
        let mut q = vec![0f32; nh * hd];
        let mut gate = vec![0f32; nh * hd];
        for h in 0..nh {
            q[h * hd..h * hd + hd].copy_from_slice(&qg[h * 2 * hd..h * 2 * hd + hd]);
            gate[h * hd..h * hd + hd].copy_from_slice(&qg[h * 2 * hd + hd..h * 2 * hd + 2 * hd]);
        }
        let mut k = matvec(&w.k, ne, nkv * hd, &xn);
        let v = matvec(&w.v, ne, nkv * hd, &xn);
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
        matvec(&w.out, nh * hd, ne, &out)
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
    Ok(tok
        .decode(&outs, false)
        .map_err(|e| anyhow!("decode: {e}"))?)
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
}
