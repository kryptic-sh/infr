//! Gemma 4 MTP: the `gemma4-assistant` draft model (the sibling `mtp-gemma-4-*.gguf`) driving
//! speculative decode on a gemma4 target.
//!
//! Unlike the qwen35 head, the assistant owns no KV cache: its layers project Q only and attend the
//! TARGET's cache (SWA layers read the target's last SWA layer, full layers its last full layer —
//! llama.cpp's `gemma4-assistant` memory sharing). Every draft step runs at the same position, so a
//! draft step is one single-row forward with nothing to catch up afterwards.

use anyhow::{anyhow, bail, Result};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::loader::MetaValue;
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::WeightSource;
use infr_gguf::Gguf;

const ARCH: &str = "gemma4-assistant";

struct AsstLayer {
    swa: bool,
    n_head: usize,
    head_dim: usize,
    n_ff: usize,
    out_scale: f32,
}

/// A loaded `gemma4-assistant` GGUF: hyperparameters plus the file the weights are read from.
pub struct Gemma4Assistant {
    gguf: Gguf,
    n_embd: usize,
    n_embd_out: usize,
    vocab: usize,
    eps: f32,
    theta: f32,
    theta_swa: f32,
    layers: Vec<AsstLayer>,
}

fn meta_usize_at(g: &Gguf, key: &str, il: usize) -> Option<usize> {
    match g.metadata().get(key)? {
        MetaValue::Arr(a) => a.get(il)?.as_u64().map(|v| v as usize),
        v => v.as_u64().map(|v| v as usize),
    }
}

fn meta_f32(g: &Gguf, key: &str) -> Option<f32> {
    g.metadata().get(key)?.as_f64().map(|v| v as f32)
}

impl Gemma4Assistant {
    /// Open and validate an assistant GGUF against its `target` model config.
    pub fn open(path: &std::path::Path, target: &crate::Config) -> Result<Self> {
        let gguf = Gguf::open(path).map_err(|e| anyhow!("open {}: {e}", path.display()))?;
        let md = gguf.metadata();
        let arch = md.str("general.architecture").unwrap_or("");
        if arch != ARCH {
            bail!(
                "{}: architecture `{arch}`, expected `{ARCH}`",
                path.display()
            );
        }
        let key = |k: &str| format!("{ARCH}.{k}");
        let req = |k: &str| -> Result<usize> {
            meta_usize_at(&gguf, &key(k), 0).ok_or_else(|| anyhow!("assistant: missing {k}"))
        };
        let n_layer = req("block_count")?;
        let n_embd = req("embedding_length")?;
        let n_embd_out = req("embedding_length_out")?;
        if n_embd_out != target.n_embd {
            bail!(
                "assistant expects a {n_embd_out}-wide target, this model is {}",
                target.n_embd
            );
        }
        let hd_full = req("attention.key_length")?;
        let hd_swa = req("attention.key_length_swa")?;
        let pattern: Vec<bool> = match md.get(&key("attention.sliding_window_pattern")) {
            Some(MetaValue::Arr(a)) => a
                .iter()
                .map(|v| matches!(v, MetaValue::Bool(true)))
                .collect(),
            _ => bail!("assistant: missing sliding_window_pattern"),
        };
        if pattern.len() != n_layer {
            bail!(
                "assistant: sliding_window_pattern has {} entries",
                pattern.len()
            );
        }
        let mut layers = Vec::with_capacity(n_layer);
        for (il, &swa) in pattern.iter().enumerate() {
            let n_head = meta_usize_at(&gguf, &key("attention.head_count"), il)
                .ok_or_else(|| anyhow!("assistant: missing head_count"))?;
            let n_ff = meta_usize_at(&gguf, &key("feed_forward_length"), il)
                .ok_or_else(|| anyhow!("assistant: missing feed_forward_length"))?;
            let name = format!("blk.{il}.layer_output_scale.weight");
            let tb = gguf
                .tensor_bytes(&name)
                .map_err(|e| anyhow!("{name}: {e}"))?;
            let out_scale = f32::from_le_bytes(
                tb[..4]
                    .try_into()
                    .map_err(|_| anyhow!("{name}: short tensor"))?,
            );
            layers.push(AsstLayer {
                swa,
                n_head,
                head_dim: if swa { hd_swa } else { hd_full },
                n_ff,
                out_scale,
            });
        }
        let src_swa = target.kv_src_layer(target.n_layer - 2);
        let src_full = target.kv_src_layer(target.n_layer - 1);
        if !target.is_swa_layer(src_swa) || target.is_swa_layer(src_full) {
            bail!("target's last two layers are not one SWA + one full layer");
        }
        for l in &layers {
            let src = if l.swa { src_swa } else { src_full };
            if target.layer_head_dim(src) != l.head_dim {
                bail!(
                    "assistant head_dim {} does not match target layer {src} ({})",
                    l.head_dim,
                    target.layer_head_dim(src)
                );
            }
        }
        let vocab = gguf
            .tensors()
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .map(|t| t.shape[1])
            .ok_or_else(|| anyhow!("assistant: missing token_embd.weight"))?;
        if vocab != target.vocab {
            bail!("assistant vocab {vocab} != target vocab {}", target.vocab);
        }
        Ok(Self {
            eps: meta_f32(&gguf, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
            theta: meta_f32(&gguf, &key("rope.freq_base")).unwrap_or(1e6),
            theta_swa: meta_f32(&gguf, &key("rope.freq_base_swa")).unwrap_or(1e4),
            gguf,
            n_embd,
            n_embd_out,
            vocab,
            layers,
        })
    }

    /// The sibling `mtp-*.gguf` next to `model`, if one exists.
    pub fn sibling_of(model: &std::path::Path) -> Option<std::path::PathBuf> {
        let dir = model.parent()?.to_path_buf();
        let mut hits: Vec<_> = std::fs::read_dir(&dir)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("mtp-") && n.ends_with(".gguf"))
            })
            .collect();
        hits.sort();
        hits.into_iter().next()
    }

    fn weight_names(&self) -> Vec<String> {
        let mut v = Vec::new();
        for il in 0..self.layers.len() {
            for t in [
                "attn_norm",
                "attn_q",
                "attn_q_norm",
                "attn_output",
                "post_attention_norm",
                "ffn_norm",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
                "post_ffw_norm",
            ] {
                v.push(format!("blk.{il}.{t}.weight"));
            }
        }
        for t in [
            "nextn.pre_projection.weight",
            "output_norm.weight",
            "token_embd.weight",
            "nextn.post_projection.weight",
            "rope_freqs.weight",
        ] {
            v.push(t.to_string());
        }
        v
    }
}

/// Uploaded assistant weights plus the reusable per-step IO buffers.
pub(crate) struct AssistantWeights {
    wbufs: Vec<Box<dyn Buffer>>,
    wspecs: Vec<(DType, usize)>,
    xh_buf: Box<dyn Buffer>,
    pos_buf: Box<dyn Buffer>,
    id_buf: Box<dyn Buffer>,
    prob_buf: Box<dyn Buffer>,
    h_buf: Box<dyn Buffer>,
}

struct StepHandles {
    xh: TensorId,
    positions: TensorId,
    k_swa: TensorId,
    v_swa: TensorId,
    k_full: TensorId,
    v_full: TensorId,
    weights: Vec<TensorId>,
    id: TensorId,
    prob: TensorId,
    h_next: TensorId,
}

/// Where the assistant's attention reads the target's cache from.
struct TargetKv {
    n_kv_swa: usize,
    n_kv_full: usize,
    rows_swa: usize,
    rows_full: usize,
    k_fmt: DType,
    v_fmt: DType,
    swa_window: usize,
}

impl AssistantWeights {
    /// Upload the assistant's weights to `be` through `bind`.
    pub(crate) fn new(
        be: &dyn Backend,
        bind: &crate::seam::BindWeight<'_>,
        asst: &Gemma4Assistant,
    ) -> Result<Self> {
        let mut wbufs = Vec::new();
        let mut wspecs = Vec::new();
        for name in asst.weight_names() {
            let t = asst
                .gguf
                .tensors()
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| anyhow!("assistant: missing tensor {name}"))?
                .clone();
            let tb = asst
                .gguf
                .tensor_bytes_arc(&name)
                .map_err(|e| anyhow!("{name}: {e}"))?;
            let numel: usize = t.shape.iter().product();
            let (buf, dt) = bind(&name, crate::seam::WBytes::Mmap(tb), t.dtype, numel)?;
            wbufs.push(buf);
            wspecs.push((dt, numel));
        }
        let alloc = |n: usize, u: BufferUsage| be.alloc(n, u).map_err(|e| anyhow!("{e}"));
        Ok(Self {
            wbufs,
            wspecs,
            xh_buf: alloc(2 * asst.n_embd_out * 4, BufferUsage::Staging)?,
            pos_buf: alloc(4, BufferUsage::Staging)?,
            id_buf: alloc(4, BufferUsage::Readback)?,
            prob_buf: alloc(4, BufferUsage::Readback)?,
            h_buf: alloc(asst.n_embd_out * 4, BufferUsage::Readback)?,
        })
    }

    fn build(&self, a: &Gemma4Assistant, kv: &TargetKv, pos: usize) -> (Graph, StepHandles) {
        let ne = a.n_embd;
        let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
        let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16);
        let mut g = Graph::new();
        let xh = g.input(f32d(2 * a.n_embd_out));
        let positions = g.input(TensorDesc::new(vec![1], DType::I32));
        let kv_decl = |g: &mut Graph, n_kv: usize, hd: usize, rows: usize| {
            let elems = crate::seam::kv_side_elems(rows * n_kv * hd);
            (
                g.input(TensorDesc::new(vec![elems], kv.k_fmt)),
                g.input(TensorDesc::new(vec![elems], kv.v_fmt)),
            )
        };
        let hd_swa = a.layers.iter().find(|l| l.swa).map_or(0, |l| l.head_dim);
        let hd_full = a.layers.iter().find(|l| !l.swa).map_or(0, |l| l.head_dim);
        let (k_swa, v_swa) = kv_decl(&mut g, kv.n_kv_swa, hd_swa, kv.rows_swa);
        let (k_full, v_full) = kv_decl(&mut g, kv.n_kv_full, hd_full, kv.rows_full);
        let weights: Vec<TensorId> = self
            .wspecs
            .iter()
            .map(|&(dt, n)| g.weight(TensorDesc::new(vec![n], dt)))
            .collect();
        let id = g.output(f32d(1));
        let prob = g.output(f32d(1));
        let h_next = g.output(f32d(a.n_embd_out));

        let max_q = a
            .layers
            .iter()
            .map(|l| l.n_head * l.head_dim)
            .max()
            .unwrap_or(0);
        let max_ff = a.layers.iter().map(|l| l.n_ff).max().unwrap_or(0);
        let resid = g.internal(f32d(ne));
        let hn = g.internal(f32d(ne));
        let sub = g.internal(f32d(ne));
        let q = g.internal(f32d(max_q));
        let q16 = g.internal(f16d(max_q));
        let attn = g.internal(f32d(max_q));
        let gbuf = g.internal(f32d(max_ff));
        let ubuf = g.internal(f32d(max_ff));
        let actbuf = g.internal(f32d(max_ff));
        let logits = g.internal(f32d(a.vocab));

        let n_globals = 5;
        let gw = &weights[weights.len() - n_globals..];
        let (w_pre, w_out_norm, w_lm, w_post, w_rope_ff) = (gw[0], gw[1], gw[2], gw[3], gw[4]);

        g.push(Op::Linear {
            x: xh,
            weight: w_pre,
            dst: resid,
            m: 1,
            in_f: (2 * a.n_embd_out) as u32,
            out_f: ne as u32,
            w_off: 0,
        });
        for (il, l) in a.layers.iter().enumerate() {
            let w = &weights[il * 10..il * 10 + 10];
            let qrow = l.n_head * l.head_dim;
            let (k_cache, v_cache, n_kv) = if l.swa {
                (k_swa, v_swa, kv.n_kv_swa)
            } else {
                (k_full, v_full, kv.n_kv_full)
            };
            g.push(Op::RmsNorm {
                x: resid,
                weight: w[0],
                dst: hn,
                rows: 1,
                dim: ne as u32,
                eps: a.eps,
            });
            g.push(Op::Linear {
                x: hn,
                weight: w[1],
                dst: q,
                m: 1,
                in_f: ne as u32,
                out_f: qrow as u32,
                w_off: 0,
            });
            g.push(Op::QkNormRope {
                x: q,
                weight: w[2],
                positions,
                dst: q16,
                rows: 1,
                n_head: l.n_head as u32,
                head_dim: l.head_dim as u32,
                rope_dim: l.head_dim as u32,
                theta: if l.swa { a.theta_swa } else { a.theta },
                eps: a.eps,
                freq_factors: (!l.swa).then_some(w_rope_ff),
                x_stride: 0,
            });
            g.push(Op::Attention {
                q: q16,
                k_cache,
                v_cache,
                dst: attn,
                rows: 1,
                kv_len: pos as u32,
                n_head: l.n_head as u32,
                n_kv: n_kv as u32,
                head_dim: l.head_dim as u32,
                scale: 1.0,
                mask: if l.swa {
                    AttnMask::SlidingWindow(kv.swa_window)
                } else {
                    AttnMask::Causal
                },
                pos: pos as u32,
                sinks: None,
                key_bias: None,
            });
            g.push(Op::Linear {
                x: attn,
                weight: w[3],
                dst: sub,
                m: 1,
                in_f: qrow as u32,
                out_f: ne as u32,
                w_off: 0,
            });
            g.push(Op::RmsNorm {
                x: sub,
                weight: w[4],
                dst: sub,
                rows: 1,
                dim: ne as u32,
                eps: a.eps,
            });
            g.push(Op::Add {
                a: resid,
                b: sub,
                dst: resid,
                n: ne as u32,
            });
            g.push(Op::RmsNorm {
                x: resid,
                weight: w[5],
                dst: hn,
                rows: 1,
                dim: ne as u32,
                eps: a.eps,
            });
            for (wt, dst) in [(w[6], gbuf), (w[7], ubuf)] {
                g.push(Op::Linear {
                    x: hn,
                    weight: wt,
                    dst,
                    m: 1,
                    in_f: ne as u32,
                    out_f: l.n_ff as u32,
                    w_off: 0,
                });
            }
            g.push(Op::GatedAct {
                gate: gbuf,
                up: ubuf,
                dst: actbuf,
                rows: 1,
                nff: l.n_ff as u32,
                act: Activation::Gelu,
                up_off: 0,
                up_stride: 0,
                gate_stride: 0,
                gate_block_width: 0,
                swiglu_clamp: None,
            });
            g.push(Op::Linear {
                x: actbuf,
                weight: w[8],
                dst: sub,
                m: 1,
                in_f: l.n_ff as u32,
                out_f: ne as u32,
                w_off: 0,
            });
            g.push(Op::RmsNorm {
                x: sub,
                weight: w[9],
                dst: sub,
                rows: 1,
                dim: ne as u32,
                eps: a.eps,
            });
            g.push(Op::Add {
                a: resid,
                b: sub,
                dst: resid,
                n: ne as u32,
            });
            g.push(Op::Scale {
                x: resid,
                dst: resid,
                s: l.out_scale,
                n: ne as u32,
            });
        }
        g.push(Op::RmsNorm {
            x: resid,
            weight: w_out_norm,
            dst: hn,
            rows: 1,
            dim: ne as u32,
            eps: a.eps,
        });
        g.push(Op::Linear {
            x: hn,
            weight: w_lm,
            dst: logits,
            m: 1,
            in_f: ne as u32,
            out_f: a.vocab as u32,
            w_off: 0,
        });
        g.push(Op::ArgmaxProb {
            x: logits,
            dst_id: id,
            dst_prob: prob,
            n: a.vocab as u32,
        });
        g.push(Op::Linear {
            x: hn,
            weight: w_post,
            dst: h_next,
            m: 1,
            in_f: ne as u32,
            out_f: a.n_embd_out as u32,
            w_off: 0,
        });
        (
            g,
            StepHandles {
                xh,
                positions,
                k_swa,
                v_swa,
                k_full,
                v_full,
                weights,
                id,
                prob,
                h_next,
            },
        )
    }

    /// One draft step: `(token, h)` at position `pos` against the target cache in `state`.
    /// Returns the drafted id, its probability, and the next step's `h`.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        be: &dyn Backend,
        a: &Gemma4Assistant,
        target: &crate::Config,
        ec: &crate::EngineConfig,
        state: &crate::seam::SeamKv,
        embd_row: &[f32],
        h: &[f32],
        pos: usize,
    ) -> Result<(u32, f32, Vec<f32>)> {
        let src_swa = target.kv_src_layer(target.n_layer - 2);
        let src_full = target.kv_src_layer(target.n_layer - 1);
        let swa_view = state.kv_layer(src_swa);
        let full_view = state.kv_layer(src_full);
        let max_ctx = state.max_ctx();
        let kv = TargetKv {
            n_kv_swa: target.layer_n_kv(src_swa),
            n_kv_full: target.layer_n_kv(src_full),
            rows_swa: crate::seam::kv_rows(target, src_swa, max_ctx, swa_view.ring, ec),
            rows_full: crate::seam::kv_rows(target, src_full, max_ctx, full_view.ring, ec),
            k_fmt: swa_view.k_fmt,
            v_fmt: swa_view.v_fmt,
            swa_window: target.swa_window,
        };
        let (graph, hs) = self.build(a, &kv, pos);
        let plan = be.compile(&graph).map_err(|e| anyhow!("{e}"))?;

        let scale = (target.n_embd as f32).sqrt();
        let mut xh = Vec::with_capacity(2 * a.n_embd_out);
        xh.extend(embd_row.iter().map(|&x| x * scale));
        xh.extend_from_slice(h);
        be.upload(self.xh_buf.as_ref(), bytemuck::cast_slice(&xh))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(self.pos_buf.as_ref(), bytemuck::bytes_of(&(pos as i32)))
            .map_err(|e| anyhow!("{e}"))?;

        let mut b = Bindings::new();
        b.bind(hs.xh, self.xh_buf.as_ref());
        b.bind(hs.positions, self.pos_buf.as_ref());
        b.bind(hs.k_swa, swa_view.k);
        b.bind(hs.v_swa, swa_view.v);
        b.bind(hs.k_full, full_view.k);
        b.bind(hs.v_full, full_view.v);
        for (id, buf) in hs.weights.iter().zip(&self.wbufs) {
            b.bind(*id, buf.as_ref());
        }
        b.bind(hs.id, self.id_buf.as_ref());
        b.bind(hs.prob, self.prob_buf.as_ref());
        b.bind(hs.h_next, self.h_buf.as_ref());
        be.execute(plan.as_ref(), &b).map_err(|e| anyhow!("{e}"))?;

        let mut id = [0u32; 1];
        let mut prob = [0f32; 1];
        let mut h_next = vec![0f32; a.n_embd_out];
        be.download(self.id_buf.as_ref(), bytemuck::cast_slice_mut(&mut id))
            .map_err(|e| anyhow!("{e}"))?;
        be.download(self.prob_buf.as_ref(), bytemuck::cast_slice_mut(&mut prob))
            .map_err(|e| anyhow!("{e}"))?;
        be.download(self.h_buf.as_ref(), bytemuck::cast_slice_mut(&mut h_next))
            .map_err(|e| anyhow!("{e}"))?;
        Ok((id[0], prob[0], h_next))
    }
}

/// Counters from one [`generate_spec`] run.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecStats {
    pub cycles: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub draft_secs: f64,
    pub verify_secs: f64,
}

/// Greedy speculative decode of `prompt` on `be`: `n_draft` assistant tokens per cycle, verified
/// in one batched target forward. Calls `on_token` for every committed token.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_spec(
    be: &dyn Backend,
    bind: &crate::seam::BindWeight<'_>,
    model: &crate::SeamModel,
    asst: &Gemma4Assistant,
    asst_w: &AssistantWeights,
    state: &mut Option<crate::seam::SeamKv>,
    want_ctx: usize,
    prompt: &[u32],
    max_new: usize,
    n_draft: usize,
    p_min: f32,
    mut on_token: impl FnMut(u32),
) -> Result<(crate::GenStats, SpecStats)> {
    anyhow::ensure!(!prompt.is_empty(), "MTP: empty prompt");
    anyhow::ensure!(n_draft >= 1, "MTP: n_draft must be at least 1");
    anyhow::ensure!(
        prompt.len() + max_new + n_draft < want_ctx,
        "MTP: prompt ({}) + max_new ({max_new}) + drafts exceed the {want_ctx}-token context",
        prompt.len()
    );
    let cfg = model.config();
    let ec = model.engine_cfg();
    let ne = cfg.n_embd;
    let ignore_eos = ec.sampling.ignore_eos;
    let is_eos = |t: u32| !ignore_eos && (cfg.eos_ids.contains(&t) || t == cfg.eos);
    let mut stats = SpecStats::default();

    let t_prompt = std::time::Instant::now();
    let mut h = Vec::new();
    let (ids, _) = crate::seam::generate_dense_backend(
        be,
        bind,
        model.gguf(),
        cfg,
        ec,
        model.embd(),
        model.per_layer_embd(),
        prompt,
        1,
        |_| {},
        state,
        want_ctx,
        None,
        None,
        None,
        None,
        Some(&mut h),
        None,
        None,
    )?;
    let prompt_secs = t_prompt.elapsed().as_secs_f64();
    anyhow::ensure!(
        ids.len() == 1 && h.len() == ne,
        "MTP prime: expected 1 id and {ne} h floats, got {} and {}",
        ids.len(),
        h.len()
    );
    let mut seq: Vec<u32> = prompt.to_vec();
    let mut last = ids[0];
    seq.push(last);
    on_token(last);
    let mut n_gen = 1;
    let t_decode = std::time::Instant::now();
    let embd = model.embd().get()?;

    while n_gen < max_new && !is_eos(last) {
        let pos = seq.len() - 1;
        let budget = n_draft.min(max_new - n_gen);
        let t0 = std::time::Instant::now();
        let mut drafts = Vec::with_capacity(budget);
        {
            let st = state
                .as_ref()
                .ok_or_else(|| anyhow!("MTP: target session missing"))?;
            let (mut tok, mut hh) = (last, h.clone());
            for _ in 0..budget {
                let row = &embd[tok as usize * ne..(tok as usize + 1) * ne];
                let (d, p, hn) = asst_w.step(be, asst, cfg, ec, st, row, &hh, pos)?;
                if p < p_min && !drafts.is_empty() {
                    break;
                }
                drafts.push(d);
                if is_eos(d) {
                    break;
                }
                tok = d;
                hh = hn;
            }
        }
        stats.draft_secs += t0.elapsed().as_secs_f64();

        let t1 = std::time::Instant::now();
        let mut tokens = seq.clone();
        tokens.extend_from_slice(&drafts);
        let mut logits = Vec::new();
        let mut vids = Vec::new();
        let mut vh = Vec::new();
        crate::seam::generate_dense_backend(
            be,
            bind,
            model.gguf(),
            cfg,
            ec,
            model.embd(),
            model.per_layer_embd(),
            &tokens,
            0,
            |_| {},
            state,
            want_ctx,
            None,
            Some(&mut logits),
            Some(&mut vids),
            None,
            Some(&mut vh),
            None,
            None,
        )?;
        if vids.is_empty() {
            vids = logits
                .chunks_exact(cfg.vocab)
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
                            if x > b.1 {
                                (i, x)
                            } else {
                                b
                            }
                        })
                        .0 as u32
                })
                .collect();
        }
        stats.verify_secs += t1.elapsed().as_secs_f64();
        anyhow::ensure!(
            vids.len() == drafts.len() + 1 && vh.len() == vids.len() * ne,
            "MTP verify: expected {} rows, got {} ids / {} h floats",
            drafts.len() + 1,
            vids.len(),
            vh.len()
        );

        let n_acc = drafts.iter().zip(&vids).take_while(|(d, v)| d == v).count();
        stats.cycles += 1;
        stats.drafted += drafts.len();
        stats.accepted += n_acc;
        for &t in drafts[..n_acc].iter().chain(std::iter::once(&vids[n_acc])) {
            seq.push(t);
            on_token(t);
            n_gen += 1;
            last = t;
            if is_eos(t) || n_gen >= max_new {
                break;
            }
        }
        h = vh[n_acc * ne..(n_acc + 1) * ne].to_vec();
        tracing::debug!(
            cycle = stats.cycles,
            drafted = drafts.len(),
            accepted = n_acc,
            "mtp cycle"
        );
    }
    Ok((
        crate::GenStats {
            n_prompt: prompt.len(),
            prompt_secs,
            n_gen,
            decode_secs: t_decode.elapsed().as_secs_f64(),
        },
        stats,
    ))
}

/// A persistent Gemma 4 MTP session: the target's KV and weights plus the assistant's weights,
/// kept across turns so each turn prefills only the changed suffix.
#[derive(Default)]
pub struct Gemma4Mtp {
    trunk: Option<crate::seam::SeamKv>,
    asst_w: Option<AssistantWeights>,
    max_ctx: usize,
}

impl Gemma4Mtp {
    /// Forget the cached tokens; weights and buffers stay resident.
    pub fn reset(&mut self) {
        if let Some(t) = self.trunk.as_mut() {
            t.reset();
        }
    }
}

/// [`generate_spec`] on `vk` for a rendered chat `prompt`, streaming decoded text to `on_piece`.
pub fn generate_vulkan(
    vk: &infr_vulkan::VulkanBackend,
    model: &crate::SeamModel,
    asst: &Gemma4Assistant,
    mtp: &mut Gemma4Mtp,
    prompt: &str,
    max_new: usize,
    mut on_piece: impl FnMut(&str),
) -> Result<(crate::GenStats, SpecStats)> {
    let ec = model.engine_cfg();
    if ec.sampling.temp > 0.0 {
        tracing::warn!("Gemma 4 MTP decodes greedily; sampling.temp is ignored");
    }
    let tokens = model.encode(prompt)?;
    let n_draft = ec.spec.k.max(1);
    if mtp.trunk.is_none() {
        let want = crate::chat::cfg_ctx_spec(ec);
        mtp.max_ctx = model.vulkan_slot_ctx(vk, 1, want)?;
    }
    let cold_bind;
    let warm_bind = |name: &str, _: crate::seam::WBytes, _: DType, _: usize| -> Result<_> {
        Err(anyhow!("warm MTP session must not re-bind {name}"))
    };
    let bind: &crate::seam::BindWeight<'_> = if mtp.trunk.is_none() {
        cold_bind = crate::seam::vulkan_moe_binder(
            vk,
            model.gguf(),
            model.config(),
            ec,
            true,
            mtp.max_ctx,
        )?;
        &*cold_bind
    } else {
        &warm_bind
    };
    if mtp.asst_w.is_none() {
        let asst_bind = |_name: &str, tb: crate::seam::WBytes, dt: DType, _n: usize| {
            let bytes = tb.materialize();
            let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
            let buf = vk
                .alloc(padded.len(), BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            vk.upload(buf.as_ref(), &padded)
                .map_err(|e| anyhow!("{e}"))?;
            Ok((buf, dt))
        };
        mtp.asst_w = Some(AssistantWeights::new(vk, &asst_bind, asst)?);
    }
    let max_new = max_new.min(mtp.max_ctx.saturating_sub(tokens.len() + n_draft + 1));
    let mut acc = Vec::new();
    let mut printed = 0;
    let (gs, st) = generate_spec(
        vk,
        bind,
        model,
        asst,
        mtp.asst_w.as_ref().expect("uploaded above"),
        &mut mtp.trunk,
        mtp.max_ctx,
        &tokens,
        max_new,
        n_draft,
        0.0,
        |t| crate::stream_token(model.tokenizer(), &mut acc, &mut printed, t, &mut on_piece),
    )?;
    tracing::info!(
        cycles = st.cycles,
        drafted = st.drafted,
        accepted = st.accepted,
        accept_rate = st.accepted as f64 / st.drafted.max(1) as f64,
        tokens_per_cycle = gs.n_gen as f64 / st.cycles.max(1) as f64,
        draft_ms = st.draft_secs * 1e3,
        verify_ms = st.verify_secs * 1e3,
        "gemma4 mtp"
    );
    Ok((gs, st))
}
