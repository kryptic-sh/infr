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
}

struct LayerWeights {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attn_norm_buf: Box<dyn Buffer>,
    ffn_norm_buf: Box<dyn Buffer>,
    wq: Box<dyn Buffer>,
    wk: Box<dyn Buffer>,
    wv: Box<dyn Buffer>,
    wo: Box<dyn Buffer>,
    wgateup: Box<dyn Buffer>, // fused [2*n_ff, n_embd] = concat(gate, up)
    wdown: Box<dyn Buffer>,
}

pub struct Llama {
    be: VulkanBackend,
    cfg: Config,
    token_embd: Vec<f32>,     // [vocab, n_embd] host, for embedding gather
    lm_head: Box<dyn Buffer>, // [vocab, n_embd] on GPU (tied to token_embd unless output.weight)
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
        other => bail!("unsupported dtype {other:?} for {name} (bring-up wants F16/F32 weights)"),
    };
    Ok((v, info.shape))
}

impl Llama {
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn load(gguf_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let g = Gguf::open(gguf_path).map_err(|e| anyhow!("open gguf: {e}"))?;
        let arch = g.metadata().str("general.architecture").unwrap_or("");
        if arch != "llama" {
            bail!("infr-llama expects architecture=llama, got {arch:?}");
        }
        let n_layer = meta_u64(&g, "llama.block_count").context("block_count")? as usize;
        let n_embd = meta_u64(&g, "llama.embedding_length").context("embedding_length")? as usize;
        let n_head = meta_u64(&g, "llama.attention.head_count").context("head_count")? as usize;
        let n_kv = meta_u64(&g, "llama.attention.head_count_kv").unwrap_or(n_head as u64) as usize;
        let n_ff =
            meta_u64(&g, "llama.feed_forward_length").context("feed_forward_length")? as usize;
        let head_dim = n_embd / n_head;
        let rope_dim =
            meta_u64(&g, "llama.rope.dimension_count").unwrap_or(head_dim as u64) as usize;
        let rope_theta = g
            .metadata()
            .get("llama.rope.freq_base")
            .and_then(|v| match v {
                infr_core::MetaValue::F64(f) => Some(*f as f32),
                infr_core::MetaValue::U64(u) => Some(*u as f32),
                _ => None,
            })
            .unwrap_or(10000.0);
        let rms_eps = g
            .metadata()
            .get("llama.attention.layer_norm_rms_epsilon")
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
        let lm_head_data = if g.tensors().iter().any(|t| t.name == "output.weight") {
            load_f32(&g, "output.weight")?.0
        } else {
            token_embd.clone()
        };
        let lm_head = be
            .upload_weight(&lm_head_data)
            .map_err(|e| anyhow!("upload lm_head: {e}"))?;

        let (output_norm, _) = load_f32(&g, "output_norm.weight")?;
        let output_norm_buf = be
            .upload_weight(&output_norm)
            .map_err(|e| anyhow!("upload output_norm: {e}"))?;

        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let up = |be: &VulkanBackend, name: String| -> Result<Box<dyn Buffer>> {
                let (d, _) = load_f32(&g, &name)?;
                be.upload_weight(&d)
                    .map_err(|e| anyhow!("upload {name}: {e}"))
            };
            let attn_norm = load_f32(&g, &p("attn_norm.weight"))?.0;
            let ffn_norm = load_f32(&g, &p("ffn_norm.weight"))?.0;
            let attn_norm_buf = be
                .upload_weight(&attn_norm)
                .map_err(|e| anyhow!("upload attn_norm {l}: {e}"))?;
            let ffn_norm_buf = be
                .upload_weight(&ffn_norm)
                .map_err(|e| anyhow!("upload ffn_norm {l}: {e}"))?;
            // fuse gate + up into one [2*n_ff, n_embd] weight (concat rows)
            let mut gateup = load_f32(&g, &p("ffn_gate.weight"))?.0;
            gateup.extend_from_slice(&load_f32(&g, &p("ffn_up.weight"))?.0);
            let wgateup = be
                .upload_weight(&gateup)
                .map_err(|e| anyhow!("upload wgateup {l}: {e}"))?;
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
            let mut q = self.lin(layer.wq.as_ref(), &hn, t, ne, nh * hd);
            let mut k = self.lin(layer.wk.as_ref(), &hn, t, ne, nkv * hd);
            let v = self.lin(layer.wv.as_ref(), &hn, t, ne, nkv * hd);
            rope_rows(&mut q, t, nh, hd, c.rope_dim, c.rope_theta);
            rope_rows(&mut k, t, nkv, hd, c.rope_dim, c.rope_theta);
            let attn = attention(&q, &k, &v, t, nh, nkv, hd);
            let ao = self.lin(layer.wo.as_ref(), &attn, t, nh * hd, ne);
            for i in 0..t * ne {
                hidden[i] += ao[i];
            }

            // --- ffn (SwiGLU) ---
            let hn2 = rmsnorm_rows(&hidden, &layer.ffn_norm, t, ne, c.rms_eps);
            let gu = self.lin(layer.wgateup.as_ref(), &hn2, t, ne, 2 * c.n_ff);
            let mut act = vec![0f32; t * c.n_ff];
            for r in 0..t {
                for i in 0..c.n_ff {
                    let g = gu[r * 2 * c.n_ff + i];
                    act[r * c.n_ff + i] = silu(g) * gu[r * 2 * c.n_ff + c.n_ff + i];
                }
            }
            let down = self.lin(layer.wdown.as_ref(), &act, t, c.n_ff, ne);
            for i in 0..t * ne {
                hidden[i] += down[i];
            }
        }

        // final norm on the last row, then lm_head
        let last = &hidden[(t - 1) * ne..t * ne];
        let normed = rmsnorm_rows(last, &self.output_norm, 1, ne, c.rms_eps);
        self.lin(self.lm_head.as_ref(), &normed, 1, ne, c.vocab)
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
            rec.linear(layer.wq.as_ref(), hn.as_ref(), q.as_ref(), t, ne, nh * hd);
            rec.linear(layer.wk.as_ref(), hn.as_ref(), k.as_ref(), t, ne, nkv * hd);
            rec.linear(layer.wv.as_ref(), hn.as_ref(), v.as_ref(), t, ne, nkv * hd);
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
            rec.linear(
                layer.wo.as_ref(),
                attn.as_ref(),
                ao.as_ref(),
                t,
                nh * hd,
                ne,
            );
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
                layer.wgateup.as_ref(),
                hn2.as_ref(),
                gu.as_ref(),
                t,
                ne,
                2 * nff,
            );
            rec.silu_mul_fused(gu.as_ref(), act.as_ref(), t, nff);
            rec.linear(
                layer.wdown.as_ref(),
                act.as_ref(),
                down.as_ref(),
                t,
                nff,
                ne,
            );
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
            self.lm_head.as_ref(),
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
            k.push(
                self.be
                    .alloc(max_ctx * row * 4, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            );
            v.push(
                self.be
                    .alloc(max_ctx * row * 4, BufferUsage::Activations)
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
        let kvrow = nkv * hd;
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
        let hidden = alloc(n * ne, BufferUsage::Staging)?;
        self.be
            .upload(hidden.as_ref(), bytemuck::cast_slice(&hidden_host))
            .map_err(|e| anyhow!("{e}"))?;
        let hn = alloc(n * ne, BufferUsage::Activations)?;
        let q = alloc(n * nh * hd, BufferUsage::Activations)?;
        let k_new = alloc(n * kvrow, BufferUsage::Activations)?;
        let v_new = alloc(n * kvrow, BufferUsage::Activations)?;
        let attn = alloc(n * nh * hd, BufferUsage::Activations)?;
        let ao = alloc(n * ne, BufferUsage::Activations)?;
        let hn2 = alloc(n * ne, BufferUsage::Activations)?;
        let gu = alloc(n * 2 * nff, BufferUsage::Activations)?;
        let act = alloc(n * nff, BufferUsage::Activations)?;
        let down = alloc(n * ne, BufferUsage::Activations)?;
        let logits = alloc(n * c.vocab, BufferUsage::Readback)?;

        let off = pos * kvrow * 4; // byte offset into the cache for the new rows
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        for (li, layer) in self.layers.iter().enumerate() {
            rec.rmsnorm(
                hidden.as_ref(),
                layer.attn_norm_buf.as_ref(),
                hn.as_ref(),
                n,
                ne,
                c.rms_eps,
            );
            rec.linear(layer.wq.as_ref(), hn.as_ref(), q.as_ref(), n, ne, nh * hd);
            rec.linear(layer.wk.as_ref(), hn.as_ref(), k_new.as_ref(), n, ne, kvrow);
            rec.linear(layer.wv.as_ref(), hn.as_ref(), v_new.as_ref(), n, ne, kvrow);
            rec.rope(
                q.as_ref(),
                q.as_ref(),
                n,
                nh,
                hd,
                c.rope_dim,
                c.rope_theta,
                pos,
            );
            rec.rope(
                k_new.as_ref(),
                k_new.as_ref(),
                n,
                nkv,
                hd,
                c.rope_dim,
                c.rope_theta,
                pos,
            );
            rec.copy(k_new.as_ref(), kv.k[li].as_ref(), off, n * kvrow * 4);
            rec.copy(v_new.as_ref(), kv.v[li].as_ref(), off, n * kvrow * 4);
            rec.attention_kv(
                q.as_ref(),
                kv.k[li].as_ref(),
                kv.v[li].as_ref(),
                attn.as_ref(),
                n,
                pos + n,
                nh,
                nkv,
                hd,
                pos,
            );
            rec.linear(
                layer.wo.as_ref(),
                attn.as_ref(),
                ao.as_ref(),
                n,
                nh * hd,
                ne,
            );
            rec.add(hidden.as_ref(), ao.as_ref(), hidden.as_ref(), n * ne);
            rec.rmsnorm(
                hidden.as_ref(),
                layer.ffn_norm_buf.as_ref(),
                hn2.as_ref(),
                n,
                ne,
                c.rms_eps,
            );
            rec.linear(
                layer.wgateup.as_ref(),
                hn2.as_ref(),
                gu.as_ref(),
                n,
                ne,
                2 * nff,
            );
            rec.silu_mul_fused(gu.as_ref(), act.as_ref(), n, nff);
            rec.linear(
                layer.wdown.as_ref(),
                act.as_ref(),
                down.as_ref(),
                n,
                nff,
                ne,
            );
            rec.add(hidden.as_ref(), down.as_ref(), hidden.as_ref(), n * ne);
        }
        rec.rmsnorm(
            hidden.as_ref(),
            self.output_norm_buf.as_ref(),
            hn.as_ref(),
            n,
            ne,
            c.rms_eps,
        );
        rec.linear(
            self.lm_head.as_ref(),
            hn.as_ref(),
            logits.as_ref(),
            n,
            ne,
            c.vocab,
        );
        rec.finish().map_err(|e| anyhow!("{e}"))?;

        let mut bytes = vec![0u8; n * c.vocab * 4];
        self.be
            .download(logits.as_ref(), &mut bytes)
            .map_err(|e| anyhow!("{e}"))?;
        kv.len += n;
        let all: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        Ok(all[(n - 1) * c.vocab..].to_vec())
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

        let mut kv = self.new_kv((prompt_tokens.len() + max_new + 8).min(8192))?;
        // prefill the whole prompt in one resident pass
        let mut logits = self.forward_resident_kv(&prompt_tokens, &mut kv)?;
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
