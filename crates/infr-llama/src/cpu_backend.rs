//! CPU model runner — builds and drives the agnostic decode [`Graph`] through [`CpuBackend`].
//! The backend itself lives in `infr-cpu`; this module is the model-specific "glue" that
//! assembles the layer graph, uploads weights, and steps the KV cache.
#![allow(clippy::too_many_arguments)]

use crate::{dequant_block, Config, GenStats, PerLayerEmbd};
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::WeightSource;
use infr_cpu::CpuBackend;
use infr_gguf::{Gguf, TensorBytes};

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
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    // Thin CPU wrapper over the backend-generic runner: a CpuBackend + a zero-copy weight binder
    // (maps each tensor straight from the GGUF mmap — no alloc, no memcpy).
    let cpu_be = CpuBackend::new();
    generate_dense_backend(
        &cpu_be,
        &|tb, dt, _n| Ok((cpu_be.map_weight(tb), dt)),
        g,
        cfg,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
    )
}

/// GPU seam runner: the SAME dense forward as [`generate_dense_cpu`], but on the Vulkan backend
/// through the agnostic [`Graph`] adapter (weights padded + uploaded to VRAM instead of mmap-mapped).
/// This is the end-to-end GPU parity/perf path — running it and diffing the CPU oracle proves the
/// adapter, and its decode tok/s (still recompiling the graph per token) is the baseline
/// record-once replay must close. Prefill's batched attention is decode-only on the seam, so the
/// caller may pass short prompts to force the per-token path.
pub(crate) fn generate_dense_gpu(
    vk: &infr_vulkan::VulkanBackend,
    g: &Gguf,
    cfg: &Config,
    token_embd: &[f32],
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_backend(
        vk,
        &|tb, dt, _n| {
            // Convert ONLY f16/bf16 weights → f16 in VRAM (mirrors the production loader): the
            // adapter's Linear then runs the f16 coopmat GEMM for prefill instead of the slow per-row
            // GEMV, and the declared dtype becomes F16 so the graph handle matches. F32 is left native
            // — the norm weights are F32 and rmsnorm/qk_norm_rope read f32, so converting them would
            // corrupt the norms. Quant weights → raw native blocks (u32-padded, in-shader dequant).
            match dt {
                DType::F16 | DType::Bf16 => {
                    let f32v = crate::dequant_block(dt, &tb).map_err(|e| anyhow!("{e}"))?;
                    let mut f16b = Vec::with_capacity(f32v.len() * 2);
                    for &v in &f32v {
                        f16b.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
                    }
                    let buf = vk
                        .alloc(f16b.len(), BufferUsage::Weights)
                        .map_err(|e| anyhow!("{e}"))?;
                    vk.upload(buf.as_ref(), &f16b).map_err(|e| anyhow!("{e}"))?;
                    Ok((buf, DType::F16))
                }
                _ => {
                    let padded = infr_vulkan::linear::pad_to_u32_align(&tb);
                    let buf = vk
                        .alloc(padded.len(), BufferUsage::Weights)
                        .map_err(|e| anyhow!("{e}"))?;
                    vk.upload(buf.as_ref(), &padded)
                        .map_err(|e| anyhow!("{e}"))?;
                    Ok((buf, dt))
                }
            }
        },
        g,
        cfg,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
    )
}

/// Metal seam runner: the SAME dense forward as [`generate_dense_cpu`], on the reference Metal
/// backend through the agnostic [`Graph`]. Weights are uploaded to Metal buffers in their NATIVE
/// GGUF dtype (the backend dequantizes lazily in its own `bytes_to_f32`, exactly like the CPU
/// interpreter — so a quant weight occupies ~quant size, not 8× f32).
#[cfg(target_os = "macos")]
pub(crate) fn generate_dense_metal(
    mtl: &infr_metal::MetalBackend,
    g: &Gguf,
    cfg: &Config,
    token_embd: &[f32],
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_backend(
        mtl,
        &|tb, dt, _n| {
            let buf = mtl
                .alloc(tb.len().max(1), BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            mtl.upload(buf.as_ref(), &tb).map_err(|e| anyhow!("{e}"))?;
            Ok((buf, dt))
        },
        g,
        cfg,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
    )
}

/// Backend-generic dense decode runner. Builds the agnostic decode [`Graph`] per token and runs it
/// on `be` (CPU reference or Vulkan). `bind_weight` turns each native-dtype GGUF tensor into a
/// backend buffer: the CPU maps it zero-copy from the mmap; the GPU pads + uploads it to VRAM. This
/// is the single forward both backends share — running it on Vulkan and diffing the CPU oracle is
/// the end-to-end dense parity check.
/// Turns a native-dtype GGUF tensor into a backend buffer + the EFFECTIVE dtype it now holds (the
/// GPU binder may convert float weights to f16), so the graph declares the handle to match.
type BindWeight<'a> = dyn Fn(TensorBytes, DType, usize) -> AResult<(Box<dyn Buffer>, DType)> + 'a;

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_dense_backend(
    be: &dyn Backend,
    bind_weight: &BindWeight,
    g: &Gguf,
    cfg: &Config,
    token_embd: &[f32],
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    mut on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    let c = cfg;
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
        // bind_weight returns the EFFECTIVE dtype the buffer holds (the GPU binder may convert float
        // weights to f16), so the graph declares the handle to match what the backend will read.
        let (buf, eff_dt) = bind_weight(tb, info.dtype, numel)?;
        wbufs.push(buf);
        wspecs.push((eff_dt, numel));
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

    // Build a forward graph for `batch` tokens starting at absolute position `start_pos`.
    // `batch = 1` is the normal decode path; `batch > 1` is the batched-prefill path.
    // Scratch tensors scale by `batch`; the LM head always runs on the LAST token only
    // (extracted via Op::Copy for batch > 1) so the logits output is always [vocab].
    let build = |batch: usize, start_pos: usize| -> (Graph, DecodeHandles) {
        let mut g = Graph::new();
        let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
        // KV cache is f16 — matches the GPU's f16 cache (halves memory, tightens CPU↔GPU parity).
        let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16);
        let hidden = g.input(f32d(batch * ne));
        let positions = g.input(TensorDesc::new(vec![batch], DType::I32));
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

        // scratch (sized to the per-layer max × batch; ops reallocate dst, so these are upper bounds)
        let hn = g.internal(f32d(batch * ne));
        let q = g.internal(f32d(batch * max_qrow));
        let k = g.internal(f32d(batch * max_kvrow));
        let v = g.internal(f32d(batch * max_kvrow));
        // QkNorm+RoPE writes f16 (the GPU `qk_norm_rope` is f32-in→f16-out, can't be in place; the GPU
        // attention reads f16 q). q16/k16 hold the f16 normed+roped q/k for the q/k-norm (qwen3/gemma)
        // path; the llama RoPE-only path stays in f32 q/k. Free on the CPU (its store is f32 regardless).
        let q16 = g.internal(f16d(batch * max_qrow));
        let k16 = g.internal(f16d(batch * max_kvrow));
        let attn = g.internal(f32d(batch * max_qrow));
        let gbuf = g.internal(f32d(batch * nff));
        let ubuf = g.internal(f32d(batch * nff));
        let actbuf = g.internal(f32d(batch * nff));
        let sub = g.internal(f32d(batch * ne));
        // E2B per-layer embed scratch: gate `[npl]` and projected `[ne]`.
        let plg = g.internal(f32d(batch * npl.max(1)));
        let plp = g.internal(f32d(batch * ne));

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
                rows: batch as u32,
                dim: ne as u32,
                eps,
            });
            g.push(Op::Linear {
                x: hn,
                weight: lw.wq,
                dst: q,
                m: batch as u32,
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
                    m: batch as u32,
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
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: kvrow as u32,
                    }),
                    None => g.push(Op::Copy {
                        src: k,
                        src_off: 0,
                        dst: v,
                        dst_off: 0,
                        n: (batch * kvrow) as u32,
                    }),
                }
                // K: fused QkNorm+RoPE (qwen3/gemma) → f16 `k16`, else RoPE alone (llama) in-place f32.
                let k_write = match lw.k_norm {
                    Some(kn) => {
                        g.push(Op::QkNormRope {
                            x: k,
                            weight: kn,
                            positions,
                            dst: k16,
                            rows: batch as u32,
                            n_head: nkv as u32,
                            head_dim: hd as u32,
                            rope_dim: rope_dim as u32,
                            theta,
                            eps,
                            freq_factors: layer_ff,
                        });
                        k16
                    }
                    None => {
                        g.push(Op::Rope {
                            x: k,
                            positions,
                            dst: k,
                            rows: batch as u32,
                            n_head: nkv as u32,
                            head_dim: hd as u32,
                            rope_dim: rope_dim as u32,
                            theta,
                            freq_factors: layer_ff,
                        });
                        k
                    }
                };
                // gemma4 weightless per-head RMSNorm on V (= x/rms) before caching.
                if let Some(ones) = v_ones {
                    g.push(Op::QkNorm {
                        x: v,
                        weight: ones,
                        dst: v,
                        rows: batch as u32,
                        n_head: nkv as u32,
                        head_dim: hd as u32,
                        eps,
                    });
                }
                g.push(Op::WriteKv {
                    src: k_write,
                    cache: k_cache[l],
                    rows: batch as u32,
                    row_stride: kvrow as u32,
                    pos: start_pos as u32,
                });
                g.push(Op::WriteKv {
                    src: v,
                    cache: v_cache[l],
                    rows: batch as u32,
                    row_stride: kvrow as u32,
                    pos: start_pos as u32,
                });
            }
            // Q: fused QkNorm+RoPE (qwen3/gemma) → f16 `q16`, else RoPE alone (llama) in-place f32.
            let q_attn = match lw.q_norm {
                Some(qn) => {
                    g.push(Op::QkNormRope {
                        x: q,
                        weight: qn,
                        positions,
                        dst: q16,
                        rows: batch as u32,
                        n_head: nh as u32,
                        head_dim: hd as u32,
                        rope_dim: rope_dim as u32,
                        theta,
                        eps,
                        freq_factors: layer_ff,
                    });
                    q16
                }
                None => {
                    g.push(Op::Rope {
                        x: q,
                        positions,
                        dst: q,
                        rows: batch as u32,
                        n_head: nh as u32,
                        head_dim: hd as u32,
                        rope_dim: rope_dim as u32,
                        theta,
                        freq_factors: layer_ff,
                    });
                    q
                }
            };
            g.push(Op::Attention {
                q: q_attn,
                k_cache: k_cache[kv_src],
                v_cache: v_cache[kv_src],
                dst: attn,
                rows: batch as u32,
                kv_len: (start_pos + batch) as u32,
                n_head: nh as u32,
                n_kv: nkv as u32,
                head_dim: hd as u32,
                scale,
                mask,
                pos: start_pos as u32,
            });
            g.push(Op::Linear {
                x: attn,
                weight: lw.wo,
                dst: sub,
                m: batch as u32,
                in_f: qrow as u32,
                out_f: ne as u32,
            });
            // gemma sandwich: post-attention norm on the sublayer output BEFORE the residual add.
            if let Some(pa) = lw.post_attn {
                g.push(Op::RmsNorm {
                    x: sub,
                    weight: pa,
                    dst: sub,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
            }
            g.push(Op::Add {
                a: hidden,
                b: sub,
                dst: hidden,
                n: (batch * ne) as u32,
            });
            // ffn
            g.push(Op::RmsNorm {
                x: hidden,
                weight: lw.ffn_norm,
                dst: hn,
                rows: batch as u32,
                dim: ne as u32,
                eps,
            });
            match lw.ffn {
                FfnW::Dense { wgate, wup, wdown } => {
                    g.push(Op::Linear {
                        x: hn,
                        weight: wgate,
                        dst: gbuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                    });
                    g.push(Op::Linear {
                        x: hn,
                        weight: wup,
                        dst: ubuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                    });
                    g.push(Op::GatedAct {
                        gate: gbuf,
                        up: ubuf,
                        dst: actbuf,
                        rows: batch as u32,
                        nff: nff_l as u32,
                        act,
                        up_off: 0,
                    });
                    g.push(Op::Linear {
                        x: actbuf,
                        weight: wdown,
                        dst: sub,
                        m: batch as u32,
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
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
            }
            g.push(Op::Add {
                a: hidden,
                b: sub,
                dst: hidden,
                n: (batch * ne) as u32,
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
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: npl as u32,
                });
                // gelu(plg) * ipl[l*npl .. l*npl+npl]  (the layer's slice of the input vector).
                g.push(Op::GatedAct {
                    gate: plg,
                    up: ipl,
                    dst: plg,
                    rows: batch as u32,
                    nff: npl as u32,
                    act: Activation::Gelu,
                    up_off: (l * npl) as u32,
                });
                g.push(Op::Linear {
                    x: plg,
                    weight: proj_w,
                    dst: plp,
                    m: batch as u32,
                    in_f: npl as u32,
                    out_f: ne as u32,
                });
                g.push(Op::RmsNorm {
                    x: plp,
                    weight: post_norm,
                    dst: plp,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
                g.push(Op::Add {
                    a: hidden,
                    b: plp,
                    dst: hidden,
                    n: (batch * ne) as u32,
                });
            }
            // gemma4: scale the whole layer output by the per-layer scalar before the next layer.
            if let Some(s) = out_scale[l] {
                g.push(Op::Scale {
                    x: hidden,
                    dst: hidden,
                    s,
                    n: (batch * ne) as u32,
                });
            }
        }
        g.push(Op::RmsNorm {
            x: hidden,
            weight: w_out_norm,
            dst: hn,
            rows: batch as u32,
            dim: ne as u32,
            eps,
        });
        // For batch > 1: the LM head runs only on the LAST token's hidden state — extract it
        // via Op::Copy before the projection so the logits output is always [vocab] regardless
        // of batch size. (For batch = 1, `hn` is already the single token's hidden state.)
        let lm_in = if batch > 1 {
            let hn_last = g.internal(f32d(ne));
            g.push(Op::Copy {
                src: hn,
                src_off: ((batch - 1) * ne) as u32,
                dst: hn_last,
                dst_off: 0,
                n: ne as u32,
            });
            hn_last
        } else {
            hn
        };
        g.push(Op::Linear {
            x: lm_in,
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
    // INFR_PROF_DEC: split decode per-token wall time into host setup (build graph + compile + bind)
    // vs execute (record + submit + GPU + wait) to guide the record-once-replay decision.
    let mut dec_setup = std::time::Duration::ZERO;
    let mut dec_exec = std::time::Duration::ZERO;

    // ── batched prefill (dense non-MoE non-E2B models only) ──────────────────────────────────
    // Process all-but-the-last prompt tokens in a single graph execution: each Op::Linear runs
    // m=(N-1) activations against every weight row in parallel (O(out_f) rayon tasks, N-1 dots
    // each), reading each weight row ONCE and reusing it across all tokens. This fills the KV
    // cache for positions 0..N-2. The last prompt token is left for the normal decode loop so
    // that the "decode" stats (tok/s) remain meaningful and the first generated token is sampled
    // in the canonical way.
    //
    // Guard: MoE uses Op::MoeFfn (per-token expert routing, no batched variant yet); E2B/gemma4
    // requires a per-(token,layer) host-side input vector that is computed in the per-step loop.
    // Both fall through to the original token-by-token loop below unchanged.
    let decode_start = if prompt.len() > 2 && c.moe.is_none() && !e2b {
        let pf_m = prompt.len() - 1; // process all but the last prompt token
                                     // Concatenate embeddings for the pf_m tokens: [pf_m × ne] row-major.
        let mut pf_hidden: Vec<f32> = Vec::with_capacity(pf_m * ne);
        for &tok in &prompt[..pf_m] {
            let base = tok as usize * ne;
            pf_hidden.extend(token_embd[base..base + ne].iter().map(|&x| x * embed_scale));
        }
        // Absolute positions [0, 1, ..., pf_m-1].
        let pf_positions: Vec<i32> = (0..pf_m as i32).collect();
        // Allocate staging buffers sized for the prefill batch.
        let pf_hidden_buf = be
            .alloc(pf_m * ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let pf_pos_buf = be
            .alloc(pf_m * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(pf_hidden_buf.as_ref(), bytemuck::cast_slice(&pf_hidden))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(pf_pos_buf.as_ref(), bytemuck::cast_slice(&pf_positions))
            .map_err(|e| anyhow!("{e}"))?;

        let pf_t0 = std::time::Instant::now();
        let (pf_g, pf_h) = build(pf_m, 0);
        let pf_plan = be.compile(&pf_g).map_err(|e| anyhow!("{e}"))?;
        let mut pf_b = Bindings::new();
        pf_b.bind(pf_h.hidden, pf_hidden_buf.as_ref());
        pf_b.bind(pf_h.positions, pf_pos_buf.as_ref());
        // gemma4's proportional-RoPE divisors are a graph input too — bind them (the per-token decode
        // loop below does the same). Without this the batched graph has an unbound `rope_freqs` Input
        // and panics. (E2B's per-layer input is excluded by the `!e2b` guard above.)
        if let (Some(rid), Some((rb, _))) = (pf_h.rope_freqs, &rf_buf) {
            pf_b.bind(rid, rb.as_ref());
        }
        for l in 0..c.n_layer {
            pf_b.bind(pf_h.k_cache[l], kbufs[l].as_ref());
            pf_b.bind(pf_h.v_cache[l], vbufs[l].as_ref());
        }
        for (i, wid) in pf_h.weights.iter().enumerate() {
            pf_b.bind(*wid, wbufs[i].as_ref());
        }
        pf_b.bind(pf_h.logits, logits_buf.as_ref());
        be.execute(pf_plan.as_ref(), &pf_b)
            .map_err(|e| anyhow!("{e}"))?;
        prompt_t += pf_t0.elapsed();

        // KV cache is filled for positions 0..pf_m-1.
        // The last prompt token (position pf_m) is handled by the decode loop below,
        // which will write its KV, get the correct logits, and sample the first generated token.
        pf_m
    } else {
        0 // fall through to per-token loop for MoE / E2B / short prompts
    };

    // Record-once decode: for an eligible qwen3-style dense decode on a backend that supports replay
    // (the Vulkan seam), build+compile+bind ONE plan here and reuse it across the whole decode loop.
    // The adapter records the graph once and replays it per token, reading `pos` from the bound
    // positions buffer + a params SSBO — so the baked pos=0 here is irrelevant, and the per-token host
    // cost drops to just the emb/pos uploads. The gate is a strict subset of the adapter's graph
    // eligibility (qwen3 dense: qk-norm, causal 1/√hd attention, no softcap / SWA / MoE / E2B /
    // proportional-RoPE), so an eligible plan here is guaranteed to take the adapter's replay path.
    // Backends without `decode_replay` (CPU interpreter, which reads the baked `pos`) and every
    // ineligible model keep rebuilding + recompiling per token below.
    let dyn_replay = be.capabilities().decode_replay
        && qk_norm
        && !gemma
        && !gemma4
        && c.moe.is_none()
        && !e2b
        && rope_freqs.is_none()
        && c.final_softcap <= 0.0;
    let ro = if dyn_replay {
        let (g, h) = build(1, 0);
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
        Some((plan, b))
    } else {
        None
    };

    for pos in decode_start..(prompt.len() + max_new) {
        if out.len() >= max_new {
            break;
        }
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

        let t_setup = std::time::Instant::now();
        let (setup_el, exec_el);
        if let Some((plan, b)) = &ro {
            // Record-once path: reuse the single compiled plan + bindings (no per-token rebuild).
            setup_el = t_setup.elapsed();
            let t_exec = std::time::Instant::now();
            be.execute(plan.as_ref(), b).map_err(|e| anyhow!("{e}"))?;
            exec_el = t_exec.elapsed();
        } else {
            let (g, h) = build(1, pos);
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
            setup_el = t_setup.elapsed();
            let t_exec = std::time::Instant::now();
            be.execute(plan.as_ref(), &b).map_err(|e| anyhow!("{e}"))?;
            exec_el = t_exec.elapsed();
        }
        if std::env::var("INFR_PROF_DEC").is_ok() && pos + 1 >= prompt.len() {
            dec_setup += setup_el;
            dec_exec += exec_el;
        }

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
    if std::env::var("INFR_PROF_DEC").is_ok() && decode_n > 0 {
        eprintln!(
            "[dec prof] {} decode tok | setup(build+compile+bind) {:.3}ms/tok | exec(record+submit+gpu) {:.3}ms/tok",
            decode_n,
            dec_setup.as_secs_f64() * 1e3 / decode_n as f64,
            dec_exec.as_secs_f64() * 1e3 / decode_n as f64,
        );
    }
    let stats = GenStats {
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
