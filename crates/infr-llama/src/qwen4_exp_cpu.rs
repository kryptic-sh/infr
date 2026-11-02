//! Scalar f32 causal oracle for Qwen4-Exp. Weights stay borrowed from the
//! safetensors mmap (including the PLE table); this is not a GPU graph fallback.

use anyhow::{ensure, Context, Result};
use half::{bf16, f16};
use infr_core::{loader::WeightSource, DType};
use std::collections::HashMap;

use crate::qwen4_exp::{Cfg, LayerType, OutputGate};
use crate::qwen4_exp_source::Qwen4ExpSource;

#[cfg(test)]
#[path = "qwen4_exp_cpu_tests.rs"]
mod tests;

/// One causal token's vocabulary logits and final HC-collapsed hidden vector.
#[derive(Debug)]
pub struct CpuStep {
    pub logits: Vec<f32>,
    pub hidden: Vec<f32>,
}

#[derive(Clone, Copy)]
struct Dense<'a> {
    bytes: &'a [u8],
    shape: &'a [usize], // fastest axis first; HF [out,in] becomes [in,out]
    dtype: DType,
}

impl Dense<'_> {
    fn at(&self, i: usize) -> f32 {
        match self.dtype {
            DType::F32 => f32::from_le_bytes(self.bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::F16 => f16::from_bits(u16::from_le_bytes(
                self.bytes[i * 2..i * 2 + 2].try_into().unwrap(),
            ))
            .to_f32(),
            DType::Bf16 => bf16::from_bits(u16::from_le_bytes(
                self.bytes[i * 2..i * 2 + 2].try_into().unwrap(),
            ))
            .to_f32(),
            _ => unreachable!("validated dense Qwen4-Exp dtype"),
        }
    }

    fn row(&self, row: usize) -> Vec<f32> {
        (0..self.shape[0])
            .map(|i| self.at(row * self.shape[0] + i))
            .collect()
    }

    fn linear_rows(&self, x: &[f32], first: usize, rows: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.shape[0]);
        (first..first + rows)
            .map(|row| {
                x.iter()
                    .enumerate()
                    .map(|(i, &v)| v * self.at(row * x.len() + i))
                    .sum()
            })
            .collect()
    }

    fn linear(&self, x: &[f32]) -> Vec<f32> {
        self.linear_rows(x, 0, self.shape[1..].iter().product())
    }
}

type Weights<'a> = HashMap<String, Dense<'a>>;
fn weight<'a>(w: &Weights<'a>, name: &str) -> Dense<'a> {
    *w.get(name).expect("validated Qwen4-Exp tensor")
}
fn linear(w: &Weights<'_>, name: &str, x: &[f32]) -> Vec<f32> {
    weight(w, name).linear(x)
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}
fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in x.iter_mut() {
        *v = (*v - max).exp();
    }
    let sum: f32 = x.iter().sum();
    for v in x {
        *v /= sum;
    }
}

// Qwen RMSNorm: separate H-wide groups, additive learned weights. GDN's gated
// output norm is different and uses direct weights shared over value heads.
fn norm(x: &[f32], w: Dense<'_>, group: usize, eps: f32, additive: bool, shared: bool) -> Vec<f32> {
    let mut out = x.to_vec();
    for (g, chunk) in out.chunks_exact_mut(group).enumerate() {
        let scale = (chunk.iter().map(|v| v * v).sum::<f32>() / group as f32 + eps)
            .sqrt()
            .recip();
        for (d, v) in chunk.iter_mut().enumerate() {
            let learned = w.at(if shared { d } else { g * group + d });
            *v *= scale * (learned + if additive { 1.0 } else { 0.0 });
        }
    }
    out
}

fn hc(w: &Weights<'_>, prefix: &str, x: &[f32], c: &Cfg, inject: bool) -> (Vec<f32>, Vec<f32>) {
    let n = norm(
        x,
        weight(w, &format!("{prefix}_norm.weight")),
        c.hidden_size,
        c.rms_norm_eps,
        true,
        false,
    );
    let mut low = linear(w, &format!("{prefix}_down.weight"), &n);
    for v in &mut low {
        *v = silu(*v / c.hc_count as f32);
    }
    let gates = linear(w, &format!("{prefix}_up.weight"), &low);
    let mut mixed = vec![0.0; c.hidden_size];
    for stream in 0..c.hc_count {
        for d in 0..c.hidden_size {
            let i = stream * c.hidden_size + d;
            mixed[d] += sigmoid(gates[i]) * n[i] / c.hc_count as f32;
        }
    }
    let injection = if inject {
        linear(w, &format!("{prefix}_inject.weight"), &n)
            .into_iter()
            .map(|v| 2.0 * sigmoid(v / c.hc_count as f32))
            .collect()
    } else {
        Vec::new()
    };
    (mixed, injection)
}

fn inject(x: &mut [f32], y: &[f32], gates: &[f32]) {
    for (stream, chunk) in x.chunks_exact_mut(y.len()).enumerate() {
        for (d, v) in chunk.iter_mut().enumerate() {
            *v += gates[stream] * y[d];
        }
    }
}

struct DeltaState {
    raw: Vec<f32>,
    recurrent: Vec<f32>,
}
#[derive(Default)]
struct AttentionState {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    index_keys: Vec<Vec<f32>>,
}
struct PleState<'a> {
    raw: Vec<f32>,
    tokens: Vec<u32>,
    multipliers: &'a [i64],
    sizes: &'a [i64],
    offsets: &'a [i64],
}
enum Attention {
    Delta(DeltaState),
    Indexed(AttentionState),
}
struct Layer<'a> {
    prefix: String,
    attention: Attention,
    ple: Option<PleState<'a>>,
}

/// Stateful, borrowed f32 CPU reference for a validated Qwen4-Exp HF source.
///
/// Every call to `step` advances ALL layer caches. `prefill` appends to the same
/// causal state; `reset` clears GDN, PLE and indexed-attention history together.
/// Activations/recurrent arithmetic are f32, even for BF16/F16 stored weights.
/// Cache geometry is capped at 256 MiB; this intentionally targets small oracles,
/// not production-size checkpoints. No full weight/table dequantization occurs.
pub struct CpuModel<'a> {
    cfg: Cfg,
    weights: Weights<'a>,
    layers: Vec<Layer<'a>>,
    position: usize,
}

impl<'a> CpuModel<'a> {
    pub fn new(source: &'a dyn WeightSource) -> Result<Self> {
        let source = source
            .as_any()
            .downcast_ref::<Qwen4ExpSource>()
            .context("qwen4_exp CPU: expected a validated dense HF checkpoint")?;
        let c = source.cfg.clone();
        ensure!(
            c.mamba_ssm_dtype == DType::F32,
            "qwen4_exp CPU: only float32 recurrent cache is implemented"
        );
        // Bound the complete possible cache before allocating even its initial
        // recurrent state. Projection geometry alone does not bound Dk * Dv.
        let mut bytes = 0usize;
        let add = |total: &mut usize, dims: &[usize], element: usize| -> Result<()> {
            let count = dims
                .iter()
                .try_fold(element, |a, &b| a.checked_mul(b))
                .context("qwen4_exp CPU: cache size overflow")?;
            *total = total
                .checked_add(count)
                .context("qwen4_exp CPU: cache size overflow")?;
            Ok(())
        };
        let channels = 2 * c.linear_num_key_heads * c.linear_key_head_dim
            + c.linear_num_value_heads * c.linear_value_head_dim;
        for (l, ty) in c.layer_types.iter().enumerate() {
            match ty {
                LayerType::LinearAttention => {
                    add(
                        &mut bytes,
                        &[
                            c.linear_num_value_heads,
                            c.linear_key_head_dim,
                            c.linear_value_head_dim,
                        ],
                        4,
                    )?;
                    add(&mut bytes, &[c.linear_conv_kernel_dim - 1, channels], 4)?;
                }
                LayerType::IndexedAttention => {
                    let i = c.indexer.as_ref().unwrap();
                    add(
                        &mut bytes,
                        &[
                            c.max_position_embeddings,
                            2,
                            c.num_key_value_heads,
                            c.head_dim,
                        ],
                        4,
                    )?;
                    add(&mut bytes, &[c.max_position_embeddings, i.head_dim], 4)?;
                    add(
                        &mut bytes,
                        &[c.max_position_embeddings, 3],
                        std::mem::size_of::<Vec<f32>>(),
                    )?;
                }
            }
            if let Some(p) = c.ple.as_ref().filter(|p| p.layer_ids.contains(&(l + 1))) {
                add(
                    &mut bytes,
                    &[
                        p.conv_kernel_size - 1,
                        p.ngram_size,
                        c.hc_count,
                        c.hidden_size,
                    ],
                    4,
                )?;
                add(&mut bytes, &[p.ngram_size - 1], 4)?;
            }
        }
        ensure!(
            bytes <= 256 * 1024 * 1024,
            "qwen4_exp CPU: reference cache exceeds 256 MiB"
        );
        output_bound(
            1,
            c.vocab_size
                .checked_add(c.hidden_size)
                .context("qwen4_exp CPU: output size overflow")?,
            0,
        )?;
        let mut weights = Weights::new();
        for t in source.tensors() {
            weights.insert(
                t.name.clone(),
                Dense {
                    bytes: source.tensor_bytes(&t.name)?,
                    shape: &t.shape,
                    dtype: t.dtype,
                },
            );
        }
        // The loader legitimately omits lm_head for tied embeddings.
        if !weights.contains_key("output.weight") {
            weights.insert(
                "output.weight".into(),
                weight(&weights, "token_embd.weight"),
            );
        }
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for (l, ty) in c.layer_types.iter().enumerate() {
            let attention = match ty {
                LayerType::LinearAttention => Attention::Delta(DeltaState {
                    raw: vec![0.0; (c.linear_conv_kernel_dim - 1) * channels],
                    recurrent: vec![
                        0.0;
                        c.linear_num_value_heads
                            * c.linear_key_head_dim
                            * c.linear_value_head_dim
                    ],
                }),
                LayerType::IndexedAttention => Attention::Indexed(AttentionState {
                    keys: Vec::with_capacity(c.max_position_embeddings),
                    values: Vec::with_capacity(c.max_position_embeddings),
                    index_keys: Vec::with_capacity(c.max_position_embeddings),
                }),
            };
            let ple = if let Some(p) = c.ple.as_ref().filter(|p| p.layer_ids.contains(&(l + 1))) {
                Some(PleState {
                    raw: vec![
                        0.0;
                        (p.conv_kernel_size - 1) * p.ngram_size * c.hc_count * c.hidden_size
                    ],
                    tokens: Vec::with_capacity(p.ngram_size - 1),
                    multipliers: source.ple_buffer(l, "layer_multipliers")?,
                    sizes: source.ple_buffer(l, "ngram_heads_vocab_sizes")?,
                    offsets: source.ple_buffer(l, "ngram_heads_offsets")?,
                })
            } else {
                None
            };
            layers.push(Layer {
                prefix: format!("blk.{l}"),
                attention,
                ple,
            });
        }
        Ok(Self {
            cfg: c,
            weights,
            layers,
            position: 0,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn reset(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            match &mut layer.attention {
                Attention::Delta(s) => {
                    s.raw.fill(0.0);
                    s.recurrent.fill(0.0);
                }
                Attention::Indexed(s) => {
                    s.keys.clear();
                    s.values.clear();
                    s.index_keys.clear();
                }
            }
            if let Some(s) = &mut layer.ple {
                s.raw.fill(0.0);
                s.tokens.clear();
            }
        }
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.validate_tokens(tokens)?;
        output_bound(
            tokens.len(),
            self.cfg.vocab_size,
            std::mem::size_of::<Vec<f32>>(),
        )?;
        tokens
            .iter()
            .map(|&id| self.step(id).map(|s| s.logits))
            .collect()
    }

    fn validate_tokens(&self, tokens: &[u32]) -> Result<()> {
        ensure!(
            tokens.len()
                <= self
                    .cfg
                    .max_position_embeddings
                    .saturating_sub(self.position),
            "qwen4_exp CPU: context limit exceeded"
        );
        ensure!(
            tokens.iter().all(|&id| (id as usize) < self.cfg.vocab_size),
            "qwen4_exp CPU: token ID is outside the vocabulary"
        );
        Ok(())
    }

    pub fn step(&mut self, token: u32) -> Result<CpuStep> {
        self.validate_tokens(&[token])?;
        let c = &self.cfg;
        let w = &self.weights;
        let embedding = weight(w, "token_embd.weight").row(token as usize);
        let mut x = embedding.repeat(c.hc_count);
        for layer in &mut self.layers {
            let b = &layer.prefix;
            if let Some(p) = &mut layer.ple {
                let contribution = ple_step(w, b, &x, token, c, p);
                for (v, y) in x.iter_mut().zip(contribution) {
                    *v += y;
                }
            }
            let (mixed, gates) = hc(w, &format!("{b}.attn_hc"), &x, c, true);
            let y = match &mut layer.attention {
                Attention::Delta(s) => delta_step(w, b, &mixed, c, s),
                Attention::Indexed(s) => attention_step(w, b, &mixed, c, self.position, s),
            };
            inject(&mut x, &y, &gates);
            let (mixed, gates) = hc(w, &format!("{b}.ffn_hc"), &x, c, true);
            inject(&mut x, &moe(w, b, &mixed, c), &gates);
        }
        let (hidden, _) = hc(w, "output_hc", &x, c, false);
        let logits = linear(w, "output.weight", &hidden);
        self.position += 1;
        Ok(CpuStep { logits, hidden })
    }
}

fn output_bound(rows: usize, columns: usize, row_metadata: usize) -> Result<()> {
    let bytes = columns
        .checked_mul(4)
        .and_then(|n| n.checked_add(row_metadata))
        .and_then(|n| rows.checked_mul(n))
        .context("qwen4_exp CPU: output size overflow")?;
    ensure!(
        bytes <= 256 * 1024 * 1024,
        "qwen4_exp CPU: reference output exceeds 256 MiB"
    );
    Ok(())
}

pub(crate) fn verify(
    source: &dyn WeightSource,
    tokens: &[u32],
    all_rows: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    ensure!(!tokens.is_empty(), "qwen4_exp CPU: prompt is empty");
    let mut model = CpuModel::new(source)?;
    model.validate_tokens(tokens)?;
    output_bound(
        if all_rows { tokens.len() } else { 1 },
        model
            .cfg
            .vocab_size
            .checked_add(model.cfg.hidden_size)
            .context("qwen4_exp CPU: output size overflow")?,
        0,
    )?;
    let mut logits = Vec::with_capacity(if all_rows {
        tokens.len() * model.cfg.vocab_size
    } else {
        0
    });
    let mut hidden = Vec::with_capacity(if all_rows {
        tokens.len() * model.cfg.hidden_size
    } else {
        0
    });
    for &token in tokens {
        let step = model.step(token)?;
        if all_rows {
            logits.extend(step.logits);
            hidden.extend(step.hidden);
        } else {
            logits = step.logits;
            hidden = step.hidden;
        }
    }
    Ok((logits, hidden))
}

#[allow(clippy::too_many_arguments)] // Mirrors concrete generation wrappers, plus effective runtime EOS.
pub(crate) fn generate(
    source: &dyn WeightSource,
    ec: &crate::EngineConfig,
    prompt: &[u32],
    max_new: usize,
    eos_ids: &[u32],
    mut constraint: Option<&mut crate::grammar::Constraint>,
    req: Option<&crate::sampling::RequestCtx>,
    mut on_token: impl FnMut(u32),
) -> Result<(Vec<u32>, crate::GenStats)> {
    use crate::sampling::{abort_requested, Penalties, Sampler};
    ensure!(!prompt.is_empty(), "qwen4_exp CPU: prompt is empty");
    let mut model = CpuModel::new(source)?;
    model.validate_tokens(prompt)?;
    ensure!(
        max_new.saturating_sub(1) <= model.cfg.max_position_embeddings - prompt.len(),
        "qwen4_exp CPU: context limit exceeded"
    );
    let sampler = Sampler::resolve(req, &ec.sampling);
    let mut rng = crate::sampling::resolve_seed(req, &ec.sampling);
    let mut penalties = Penalties::resolve(req);
    let start = std::time::Instant::now();
    let mut logits = Vec::new();
    for &token in prompt {
        ensure!(
            !abort_requested(req),
            "aborted: request cancelled during prompt prefill"
        );
        logits = model.step(token)?.logits;
    }
    let prompt_secs = start.elapsed().as_secs_f64();
    let decode = std::time::Instant::now();
    output_bound(max_new, 1, 0)?;
    let mut out = Vec::with_capacity(max_new);
    while out.len() < max_new && !abort_requested(req) {
        if let Some(cst) = constraint.as_deref_mut() {
            let (step, done) = crate::grammar::constrained_step(
                cst,
                &mut logits,
                eos_ids,
                sampler,
                &mut rng,
                max_new - out.len(),
            )?;
            if step.is_empty() {
                break;
            }
            // A forced span can contain several tokens. Feed EVERY preceding
            // token before sampling again, never just the last token of a span.
            for &id in &step {
                out.push(id);
                on_token(id);
            }
            if done || out.len() >= max_new || abort_requested(req) {
                break;
            }
            for id in step {
                if abort_requested(req) {
                    break;
                }
                logits = model.step(id)?.logits;
            }
        } else {
            if let Some(p) = penalties.as_ref() {
                p.apply(&mut logits);
            }
            let id = crate::sampling::sample_logits(&logits, sampler, &mut rng);
            if let Some(p) = penalties.as_mut() {
                p.observe(id);
            }
            let eos = !ec.sampling.ignore_eos && eos_ids.contains(&id);
            out.push(id);
            if !eos {
                on_token(id);
            }
            if eos || out.len() >= max_new || abort_requested(req) {
                break;
            }
            logits = model.step(id)?.logits;
        }
    }
    let n_gen = out.len();
    Ok((
        out,
        crate::GenStats {
            n_prompt: prompt.len(),
            prompt_secs,
            n_gen,
            decode_secs: decode.elapsed().as_secs_f64(),
        },
    ))
}

// Time-major RAW history, channel-major oldest-to-current convolution weights.
// PLE uses dilation=ngram_size; GDN uses dilation=1 and only the QKV projection.
fn convolution(
    w: Dense<'_>,
    history: &mut [f32],
    current: &[f32],
    kernel: usize,
    dilation: usize,
) -> Vec<f32> {
    let width = current.len();
    let mut out = vec![0.0; width];
    for ch in 0..width {
        let mut v = current[ch] * w.at(ch * kernel + kernel - 1);
        for tap in 0..kernel - 1 {
            v += history[tap * dilation * width + ch] * w.at(ch * kernel + tap);
        }
        out[ch] = silu(v);
    }
    if !history.is_empty() {
        let len = history.len();
        history.copy_within(width.., 0);
        history[len - width..].copy_from_slice(current);
    }
    out
}

fn delta_step(w: &Weights<'_>, b: &str, x: &[f32], c: &Cfg, s: &mut DeltaState) -> Vec<f32> {
    let raw = linear(w, &format!("{b}.attn_qkv.weight"), x);
    let z = linear(w, &format!("{b}.attn_gate.weight"), x);
    let a = linear(w, &format!("{b}.ssm_alpha.weight"), x);
    let beta = linear(w, &format!("{b}.ssm_beta.weight"), x);
    let conv = convolution(
        weight(w, &format!("{b}.ssm_conv1d.weight")),
        &mut s.raw,
        &raw,
        c.linear_conv_kernel_dim,
        1,
    );
    let dk = c.linear_key_head_dim;
    let dv = c.linear_value_head_dim;
    let qk_width = c.linear_num_key_heads * dk;
    let mut core = vec![0.0; c.linear_num_value_heads * dv];
    for h in 0..c.linear_num_value_heads {
        // HF repeat_interleave, NOT the existing CPU graph op's modulo tiling.
        let kh = h / (c.linear_num_value_heads / c.linear_num_key_heads);
        let q = &conv[kh * dk..(kh + 1) * dk];
        let k = &conv[qk_width + kh * dk..qk_width + (kh + 1) * dk];
        let qscale = (dot(q, q) + 1e-6).sqrt().recip() / (dk as f32).sqrt();
        let kscale = (dot(k, k) + 1e-6).sqrt().recip();
        let decay = (-weight(w, &format!("{b}.ssm_a")).at(h).exp()
            * softplus(a[h] + weight(w, &format!("{b}.ssm_dt.bias")).at(h)))
        .exp();
        let state = &mut s.recurrent[h * dk * dv..(h + 1) * dk * dv];
        for v in state.iter_mut() {
            *v *= decay;
        }
        for d in 0..dv {
            let prediction: f32 = (0..dk).map(|j| k[j] * kscale * state[j * dv + d]).sum();
            let delta = sigmoid(beta[h]) * (conv[2 * qk_width + h * dv + d] - prediction);
            for j in 0..dk {
                state[j * dv + d] += k[j] * kscale * delta;
            }
            core[h * dv + d] = (0..dk).map(|j| q[j] * qscale * state[j * dv + d]).sum();
        }
    }
    let mut out = norm(
        &core,
        weight(w, &format!("{b}.ssm_norm.weight")),
        dv,
        c.rms_norm_eps,
        false,
        true,
    );
    for (v, z) in out.iter_mut().zip(z) {
        *v *= match c.output_gate {
            OutputGate::Sigmoid => sigmoid(z),
            OutputGate::Silu => silu(z),
        };
    }
    linear(w, &format!("{b}.ssm_out.weight"), &out)
}

fn moe(w: &Weights<'_>, b: &str, x: &[f32], c: &Cfg) -> Vec<f32> {
    let mut probabilities = linear(w, &format!("{b}.ffn_gate_inp.weight"), x);
    softmax(&mut probabilities);
    let mut experts: Vec<usize> = (0..c.num_experts).collect();
    experts.sort_by(|&a, &b| {
        probabilities[b]
            .total_cmp(&probabilities[a])
            .then(a.cmp(&b))
    });
    experts.truncate(c.num_experts_per_tok);
    let denominator = if c.norm_topk_prob {
        experts.iter().map(|&e| probabilities[e]).sum()
    } else {
        1.0
    };
    // HF's reference dispatch accumulates in ascending expert ID, not rank order.
    experts.sort_unstable();
    let mut out = vec![0.0; c.hidden_size];
    for e in experts {
        let result = if let Some(packed) = w.get(&format!("{b}.ffn_gate_up_exps.weight")) {
            let mut gate =
                packed.linear_rows(x, e * 2 * c.moe_intermediate_size, c.moe_intermediate_size);
            let up = packed.linear_rows(
                x,
                e * 2 * c.moe_intermediate_size + c.moe_intermediate_size,
                c.moe_intermediate_size,
            );
            for (g, u) in gate.iter_mut().zip(up) {
                *g = silu(*g) * u;
            }
            weight(w, &format!("{b}.ffn_down_exps.weight")).linear_rows(
                &gate,
                e * c.hidden_size,
                c.hidden_size,
            )
        } else {
            expert(w, &format!("{b}.expert.{e}"), x)
        };
        for (v, y) in out.iter_mut().zip(result) {
            *v += probabilities[e] / denominator * y;
        }
    }
    let shared = expert(w, &format!("{b}.shexp"), x);
    let gate = sigmoid(linear(w, &format!("{b}.ffn_gate_inp_shexp.weight"), x)[0]);
    for (v, y) in out.iter_mut().zip(shared) {
        *v += gate * y;
    }
    out
}
fn expert(w: &Weights<'_>, prefix: &str, x: &[f32]) -> Vec<f32> {
    let mut gate = linear(w, &format!("{prefix}.gate.weight"), x);
    let up = linear(w, &format!("{prefix}.up.weight"), x);
    for (g, u) in gate.iter_mut().zip(up) {
        *g = silu(*g) * u;
    }
    linear(w, &format!("{prefix}.down.weight"), &gate)
}

fn rotary(x: &mut [f32], head_dim: usize, position: usize, c: &Cfg) {
    let half = c.rope_dim / 2;
    for head in x.chunks_exact_mut(head_dim) {
        for j in 0..half {
            let angle = position as f32 / c.rope_theta.powf(2.0 * j as f32 / c.rope_dim as f32);
            let (sin, cos) = angle.sin_cos();
            let (a, b) = (head[j], head[j + half]);
            head[j] = a * cos - b * sin;
            head[j + half] = b * cos + a * sin;
        }
    }
    // Text-only positions are identical across all three MRoPE axes, so both
    // mrope_interleaved layouts reduce to the same partial ordinary RoPE.
}

fn index_selection(
    w: &Weights<'_>,
    b: &str,
    x: &[f32],
    c: &Cfg,
    pos: usize,
    s: &mut AttentionState,
) -> Vec<usize> {
    let i = c.indexer.as_ref().unwrap();
    let projection = linear(w, &format!("{b}.indexer_qk.weight"), x);
    let mut query = norm(
        &projection[..i.n_heads * i.head_dim],
        weight(w, &format!("{b}.indexer_q_norm.weight")),
        i.head_dim,
        c.rms_norm_eps,
        true,
        true,
    );
    rotary(&mut query, i.head_dim, pos, c);
    s.index_keys
        .push(projection[i.n_heads * i.head_dim..].to_vec());
    let blocks = s.index_keys.len() / i.compress_ratio;
    let mut scores = Vec::with_capacity(blocks);
    for block in 0..blocks {
        let mut key = vec![0.0; i.head_dim];
        for raw in &s.index_keys[block * i.compress_ratio..(block + 1) * i.compress_ratio] {
            for (k, v) in key.iter_mut().zip(raw) {
                *k += *v / i.compress_ratio as f32;
            }
        }
        let mut key = norm(
            &key,
            weight(w, &format!("{b}.indexer_k_norm.weight")),
            i.head_dim,
            c.rms_norm_eps,
            true,
            true,
        );
        rotary(&mut key, i.head_dim, block * i.compress_ratio, c);
        let score = query
            .chunks_exact(i.head_dim)
            .map(|q| dot(q, &key).max(0.0))
            .sum::<f32>()
            / (i.head_dim as f32).sqrt();
        scores.push((block, score));
    }
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scores.truncate((i.budget / i.compress_ratio).min(blocks));
    let mut selected = Vec::new();
    for (block, _) in scores {
        selected.extend(block * i.compress_ratio..(block + 1) * i.compress_ratio);
    }
    selected.extend(blocks * i.compress_ratio..s.index_keys.len());
    // Attention reductions follow token order, independent of top-k score order.
    selected.sort_unstable();
    selected
}

fn attention_step(
    w: &Weights<'_>,
    b: &str,
    x: &[f32],
    c: &Cfg,
    pos: usize,
    s: &mut AttentionState,
) -> Vec<f32> {
    let projection = linear(w, &format!("{b}.attn_q.weight"), x);
    let mut q = Vec::with_capacity(c.num_attention_heads * c.head_dim);
    let mut gate = Vec::with_capacity(q.capacity());
    for head in projection.chunks_exact(2 * c.head_dim) {
        q.extend_from_slice(&head[..c.head_dim]);
        gate.extend_from_slice(&head[c.head_dim..]);
    }
    q = norm(
        &q,
        weight(w, &format!("{b}.attn_q_norm.weight")),
        c.head_dim,
        c.rms_norm_eps,
        true,
        true,
    );
    let mut k = norm(
        &linear(w, &format!("{b}.attn_k.weight"), x),
        weight(w, &format!("{b}.attn_k_norm.weight")),
        c.head_dim,
        c.rms_norm_eps,
        true,
        true,
    );
    rotary(&mut q, c.head_dim, pos, c);
    rotary(&mut k, c.head_dim, pos, c);
    s.keys.push(k);
    s.values.push(linear(w, &format!("{b}.attn_v.weight"), x));
    let selected = index_selection(w, b, x, c, pos, s);
    let mut out = vec![0.0; c.num_attention_heads * c.head_dim];
    for h in 0..c.num_attention_heads {
        let kh = h / (c.num_attention_heads / c.num_key_value_heads);
        let query = &q[h * c.head_dim..(h + 1) * c.head_dim];
        let mut scores: Vec<f32> = selected
            .iter()
            .map(|&t| {
                dot(query, &s.keys[t][kh * c.head_dim..(kh + 1) * c.head_dim])
                    / (c.head_dim as f32).sqrt()
            })
            .collect();
        softmax(&mut scores);
        for (&t, probability) in selected.iter().zip(scores) {
            for d in 0..c.head_dim {
                out[h * c.head_dim + d] += probability * s.values[t][kh * c.head_dim + d];
            }
        }
    }
    for (v, gate) in out.iter_mut().zip(gate) {
        *v *= sigmoid(gate);
    }
    linear(w, &format!("{b}.attn_output.weight"), &out)
}

fn ple_step(
    w: &Weights<'_>,
    b: &str,
    x: &[f32],
    token: u32,
    c: &Cfg,
    s: &mut PleState<'_>,
) -> Vec<f32> {
    let p = c.ple.as_ref().unwrap();
    let eos = c.eos_ids[0]; // HF PLE uses the first EOS even when generation has several.
    let mut shifted = vec![token];
    let mut boundary = false;
    for back in 1..p.ngram_size {
        let prior = s
            .tokens
            .len()
            .checked_sub(back)
            .map(|i| s.tokens[i])
            .unwrap_or(eos);
        boundary |= prior == eos;
        shifted.push(if boundary { eos } else { prior });
    }
    if s.tokens.len() == p.ngram_size - 1 {
        s.tokens.remove(0);
    }
    s.tokens.push(token);
    let heads = s.sizes.len();
    let head_width = p.embed_dim / heads;
    let mut embedding = Vec::with_capacity(p.embed_dim);
    let mut mixed = shifted[0] as i64 * s.multipliers[0];
    for n in 1..p.ngram_size {
        mixed ^= shifted[n] as i64 * s.multipliers[n];
        for h in 0..p.heads_per_ngram {
            let head = (n - 1) * p.heads_per_ngram + h;
            let row = (mixed % s.sizes[head] + s.offsets[head]) as usize;
            if let Some(table) = w.get(&format!("{b}.ple_embd.weight")) {
                embedding.extend(table.row(row));
            } else {
                let rows = weight(w, &format!("{b}.ple_embd.0.weight")).shape[1];
                embedding.extend(
                    weight(w, &format!("{b}.ple_embd.{}.weight", row / rows)).row(row % rows),
                );
            }
        }
    }
    debug_assert_eq!(embedding.len(), heads * head_width);
    let key = norm(
        &linear(w, &format!("{b}.ple_key.weight"), &embedding),
        weight(w, &format!("{b}.ple_key_norm.weight")),
        c.hidden_size,
        c.rms_norm_eps,
        true,
        false,
    );
    let query = norm(
        x,
        weight(w, &format!("{b}.ple_query_norm.weight")),
        c.hidden_size,
        c.rms_norm_eps,
        true,
        false,
    );
    let value = linear(w, &format!("{b}.ple_value.weight"), &embedding);
    let mut gated = vec![0.0; c.hc_count * c.hidden_size];
    for stream in 0..c.hc_count {
        let range = stream * c.hidden_size..(stream + 1) * c.hidden_size;
        let score = dot(&key[range.clone()], &query[range]) / (c.hidden_size as f32).sqrt();
        let transformed = if score == 0.0 {
            0.0
        } else {
            score.signum() * score.abs().max(1e-6).sqrt()
        };
        let gate = sigmoid(transformed);
        for d in 0..c.hidden_size {
            gated[stream * c.hidden_size + d] = gate * value[d];
        }
    }
    let input = norm(
        &gated,
        weight(w, &format!("{b}.ple_conv_norm.weight")),
        c.hidden_size,
        c.rms_norm_eps,
        true,
        false,
    );
    let conv = convolution(
        weight(w, &format!("{b}.ple_conv.weight")),
        &mut s.raw,
        &input,
        p.conv_kernel_size,
        p.ngram_size,
    );
    for (v, y) in gated.iter_mut().zip(conv) {
        *v += y;
    }
    gated
}
