//! Backend-generic dense decode runner: builds the agnostic decode [`Graph`] per token/batch and
//! drives it through a [`Backend`]. This is the giant `generate_dense_backend` — the single
//! forward every entry point in `super` (CPU/Vulkan/Metal, one-shot/session/verify/denoise) funnels
//! through. Pure-move split of `seam.rs` — see `super` for the module overview.
use super::sc::{build_sc_embt, diffusion_self_cond, DenoiseCache, DenoiseReq, SelfCondWeights};
use super::weights::{AttnW, DeltaW, FfnW, LayerW, MixerW, SeamKv, SeamWeights};
use super::{common_prefix_len, e2b_ipl_rows, kv_fmt_bytes, kv_forces_static, BindWeight, WBytes};
use crate::{Config, GenStats, PerLayerEmbd};
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::WeightSource;
use infr_gguf::Gguf;

/// Handles into one freshly-built decode graph that the driver re-binds each step.
pub(super) struct DecodeHandles {
    hidden: TensorId,
    positions: TensorId,
    rope_freqs: Option<TensorId>, // gemma4 proportional-RoPE divisors (full-attention layers)
    // gemma4 E2B host-gathered per-layer TOKEN embedding rows `[n_layer*npl]` — the graph Input
    // the driver binds `ipl_buf` to; the GPU prologue turns this into the layer loop's actual
    // per-layer input vector (see `per_layer_inp` inside `build`).
    pl_tok_in: Option<TensorId>,
    // Phase-B perf: DiffusionGemma in-graph self-conditioning inputs/weight — `Some` only when
    // `build` was called with `gpu_sc: Some(true)` (see `build`'s doc). `sc_logits` is the
    // per-step Input (host-premultiplied previous canvas logits); `sc_embt` is the one-time
    // device weight bound from `SeamKv::sc_embt`, NOT from the ordinary `weights` upload loop.
    sc_logits: Option<TensorId>,
    sc_embt: Option<TensorId>,
    // Vulkan-only perf (`dyn_sc_scale`'s doc on `build`): the 1-element Input the SC subgraph's
    // `Op::Softmax` reads its scale from instead of a baked constant. `Some` only when `build` was
    // called with `gpu_sc: Some(true)` AND `dyn_sc_scale: true`; `None` otherwise (Metal's SC
    // subgraph, and every non-SC build) — mirrors `sc_logits`/`sc_embt`'s gating.
    temp_inv: Option<TensorId>,
    logits: TensorId,
    // MTP Phase 1 (issue #33, docs/MTP.md): the LM-head INPUT — the same rows `logits` was
    // computed from, one op earlier (post-`output_norm`, pre-`w_lm`). `Some` only when `build`
    // was called with `h_tap: true`; `None` for every ordinary caller (no extra op, no extra
    // download). This is the primitive Phase 2's MTP head needs (`h_p` in `docs/MTP.md`'s forward
    // pseudocode) — Phase 1 only exposes the tap, no head graph reads it yet.
    h_out: Option<TensorId>,
    k_cache: Vec<TensorId>,
    v_cache: Vec<TensorId>,
    weights: Vec<TensorId>, // flat, in declaration == upload order
}

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
    state: &mut Option<SeamKv>,
    want_ctx: usize,
    mut constraint: Option<&mut crate::grammar::Constraint>,
    verify: Option<&mut Vec<f32>>,
    // Phase-1 DiffusionGemma validation hook: captures the LAST prompt token's raw logits (the
    // per-token loop's first `is_decode` row, i.e. the causal-prefill result) without disturbing
    // the sampled continuation. `None` everywhere else. Unlike `verify` (a batched m-row forward,
    // MoE-incompatible — see its guard below) this rides the existing rows==1 per-token loop, so
    // it works for MoE/diffusion-gemma models too.
    mut logits_out: Option<&mut Vec<f32>>,
    // MTP Phase 1 (issue #33, docs/MTP.md): captures the LM-head INPUT rows (post-`output_norm`,
    // pre-`w_lm` — `DecodeHandles::h_out`'s doc) for the SAME row(s) `logits_out`/`verify` came
    // from: `[ne]` for the per-token decode loop's frontier row, `[m * ne]` for speculative
    // VERIFY's `m` rows. `None` everywhere else (no extra op, no extra download — see `h_tap`'s
    // doc on `build`). This is Phase 2's MTP driver primitive (`h_p` in `docs/MTP.md`); Phase 1
    // only exposes it for validation (`lm_head(h_row) == logits_row`).
    mut h_out: Option<&mut Vec<f32>>,
    // Phase-2 DiffusionGemma canvas denoise (see `DenoiseReq`'s doc). `None` everywhere else.
    denoise_req: Option<DenoiseReq>,
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
    // layer output before the next layer). Read host-side; applied as an `Op::Scale`. diffusion-
    // gemma ships TWO per-layer scalars (encoder for the prompt, decoder for the canvas); Phase 1
    // is the encoder-only causal prefill, so it reads `enc_layer_output_scale` — the decoder
    // scalar is unused until the canvas denoise graph (Phase 2+).
    let out_scale_name = if c.diffusion_gemma {
        "enc_layer_output_scale"
    } else {
        "layer_output_scale"
    };
    let out_scale: Vec<Option<f32>> = (0..c.n_layer)
        .map(|l| {
            let name = format!("blk.{l}.{out_scale_name}.weight");
            if g.tensors().iter().any(|t| t.name == name) {
                crate::load_tensor_dequant(g, &name)
                    .ok()
                    .and_then(|(v, _)| v.first().copied())
            } else {
                None
            }
        })
        .collect();
    // diffusion-gemma's DECODER per-layer scalar (`layer_output_scale`, the canvas-denoise twin
    // of `out_scale`'s encoder-named array above) — read unconditionally alongside it (both are
    // tiny [1]-tensors, negligible host cost) so the denoise graph (`build`'s `denoise` flag) can
    // select it without re-deriving the name. `None`/empty for every non-diffusion-gemma model
    // (never read there).
    let dec_out_scale: Vec<Option<f32>> = (0..c.n_layer)
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
    // Combined gate+up FFN weights (one GEMV/GEMM + GatedActFused instead of two Linears +
    // GatedAct — the bespoke path's fused-gu shape, ~1 dispatch/layer off the decode hot loop).
    // Requires the backend to opt in (Vulkan; the CPU keeps zero-copy separate tensors) AND every
    // dense layer's gate/up to share a dtype (the concat is one [2*nff, ne] tensor). The decision
    // is global so the upload order and `build`'s handle declarations always agree.
    let fuse_gu = be.capabilities().combined_gu
        && c.moe.is_none()
        && (0..c.n_layer).all(|l| {
            let dt = |s: &str| {
                let name = format!("blk.{l}.{s}");
                g.tensors().iter().find(|t| t.name == name).map(|t| t.dtype)
            };
            dt("ffn_gate.weight").is_some() && dt("ffn_gate.weight") == dt("ffn_up.weight")
        });
    // Combined QKV: one [qrow+2·kvrow, ne] weight → prefill runs ONE wide GEMM (the separate
    // q/k/v GEMMs are narrow-n and underfill a big GPU — the pp512 sweep's dominant cost), and
    // decode keeps three offset GEMVs into the same buffer (`Op::Linear.w_off`), so its dispatch
    // count is unchanged. Needs every layer to own all three projections in ONE native-supported
    // dtype (gemma4's V-less full layers keep the split form), uniform dims (the offsets are
    // baked once), and a backend that opted into combined weights.
    let fuse_qkv = be.capabilities().combined_gu
        && (0..c.n_layer).all(|l| {
            let dt = |s: &str| {
                let name = format!("blk.{l}.{s}");
                g.tensors().iter().find(|t| t.name == name).map(|t| t.dtype)
            };
            let q = dt("attn_q.weight");
            q.is_some()
                && q == dt("attn_k.weight")
                && q == dt("attn_v.weight")
                && q.is_some_and(|d| {
                    infr_vulkan::linear::native_dense_supported(d) && d != DType::F16
                })
                && c.layer_head_dim(l) == c.head_dim
                && c.layer_n_kv(l) == c.n_kv
                && c.has_own_kv(l)
        });

    // KV cache dtype, chosen PER-SIDE (K and V independent, like llama's --cache-type-k /
    // --cache-type-v). Q8_0 stores 34 bytes / 32 elems — half the f16 footprint and bandwidth.
    //   INFR_KV_TYPE_K / INFR_KV_TYPE_V ∈ {f16, q8_0}  (per-side override)
    //   INFR_KV_Q8=1                                    legacy alias: any side not otherwise set → q8_0
    // Per-side KV dtype, chosen from INFR_KV_TYPE_K/V (llama's --cache-type-k/-v). The graph decl
    // carries the dtype and the env is stable for the process, so a warm session and its rebuilt
    // graphs always agree. Gates: Q8_0 needs each layer's KV row (n_kv*head_dim) 32-block-aligned and
    // a backend with the Q8 read/write (cpu/vulkan/metal). TurboQuant (turbo2/3/4) is WHT-rotated,
    // 128-elem blocks = head_dim slices — CPU-only, needs head_dim%128. The mainline low-bit quants
    // (q4_0/q4_1/q5_0/q5_1/iq4_nl) + f32/bf16 are CPU-only too (no GPU KV kernel yet); the block
    // quants need 32-alignment. All of these are footprint knobs (quantized KV is slower on CPU).
    let kv_align_ok =
        (0..c.n_layer).all(|l| (c.layer_n_kv(l) * c.layer_head_dim(l)).is_multiple_of(32));
    let kv_q8_backend = matches!(be.name(), "metal" | "cpu" | "vulkan");
    // TurboQuant (turbo2/3/4): CPU + Vulkan + Metal (both GPUs use a dequant→f16 prepass); needs
    // head_dim % 128 (a WHT group is a 128-elem head_dim slice).
    let kv_turbo_ok = matches!(be.name(), "cpu" | "vulkan" | "metal")
        && (0..c.n_layer).all(|l| c.layer_head_dim(l).is_multiple_of(128));
    // Mainline low-bit block quants (q4_0/…/iq4_nl): CPU + Vulkan + Metal (both GPUs dequant→f16
    // prepass); need 32-block alignment.
    let blk_ok = matches!(be.name(), "cpu" | "vulkan" | "metal") && kv_align_ok;
    // Dense f32/bf16 KV: CPU + Vulkan + Metal. Vulkan/Metal store dense; f32 reads natively (its
    // own f32 attention), bf16 reads via a cast→f16 prepass.
    let dense_ok = matches!(be.name(), "cpu" | "vulkan" | "metal");
    let parse_kv_fmt = |var: &str| -> DType {
        let side = std::env::var(var).ok();
        match side.as_deref() {
            Some("turbo2") if kv_turbo_ok => DType::Turbo2,
            Some("turbo3") if kv_turbo_ok => DType::Turbo3,
            Some("turbo4") if kv_turbo_ok => DType::Turbo4,
            Some("q8_0") | Some("q8") | Some("Q8_0") if kv_align_ok && kv_q8_backend => DType::Q8_0,
            Some("q4_0") if blk_ok => DType::Q4_0,
            Some("q4_1") if blk_ok => DType::Q4_1,
            Some("q5_0") if blk_ok => DType::Q5_0,
            Some("q5_1") if blk_ok => DType::Q5_1,
            Some("iq4_nl") if blk_ok => DType::Iq4Nl,
            Some("bf16") if dense_ok => DType::Bf16,
            Some("f32") if dense_ok => DType::F32,
            Some("f16") | Some("F16") => DType::F16,
            // unset/unknown → legacy INFR_KV_Q8 alias (both sides q8) or f16.
            _ if std::env::var("INFR_KV_Q8").is_ok() && kv_align_ok && kv_q8_backend => DType::Q8_0,
            _ => DType::F16,
        }
    };
    let mut k_fmt = parse_kv_fmt("INFR_KV_TYPE_K");
    let mut v_fmt = parse_kv_fmt("INFR_KV_TYPE_V");
    // Metal's Q8 and F32 KV use native-read attention that reads BOTH sides as one dtype, so a
    // mixed request with q8/f32 on one side would misread the other — clamp those to coupled f16.
    // The prepass formats (block quants / bf16 / turbo) expand each side to its own f16 scratch, so
    // they compose freely with each other and with a native-f16 side (per-side, like Vulkan/CPU).
    if be.name() == "metal" && k_fmt != v_fmt {
        let native_read = |dt| matches!(dt, DType::Q8_0 | DType::F32);
        if native_read(k_fmt) || native_read(v_fmt) {
            k_fmt = DType::F16;
            v_fmt = DType::F16;
        }
    }

    // ── one-time session init: weights, KV cache, per-step IO (skipped when `state` is warm) ──
    if state.is_none() {
        // Weight-load progress: opened HERE — the single weight-upload funnel every runner path
        // (CPU/Vulkan/Metal × one-shot/session/bench/serve) goes through — so no entry point can
        // load without it. The ticking lives in each backend's `alloc` (Weights/HostWeights);
        // backends without a display return a no-op scope. Guard drops when the init block ends.
        let fp = crate::weights::weight_footprint(g);
        let _weight_pb = be.weight_progress(Some(fp.dense + fp.expert));
        // ── upload weights in their NATIVE GGUF dtype (no host pre-dequant — the backend dequants
        //    lazily in `bytes_to_f32`, so a quant weight occupies ~quant size, not 8× f32). `wspecs`
        //    records each (dtype, numel) so `build` can declare the handle with the matching dtype; its
        //    order MUST equal the `g.weight()` order in `build` below. ──────────────────────────────────
        let mut wbufs: Vec<Box<dyn Buffer>> = Vec::new();
        let mut wspecs: Vec<(DType, usize)> = Vec::new();
        // Load one weight (zero-copy mmap slice — no alloc, no memcpy) or CONCATENATE several into
        // one owned buffer (the combined gate+up upload; same dtype, row-major concat of [nff, ne]
        // tensors = a valid [k*nff, ne] tensor). Records the native dtype + element count so
        // `build` declares the handle to match.
        // NEOX→NORM row permute (qwen2, `Config::permute_qk_neox`): qwen2's GGUF keeps attn_q/attn_k
        // in the HF rotate-half order (the converter only permutes llama-arch), but the no-qknorm
        // path's `Op::Rope` is the INTERLEAVED rotation. Reordering each head's rows at load —
        // new[2p] = old[p], new[2p+1] = old[p + rd/2], dims past rope_dim pass through — makes NORM
        // rope over the permuted projections equal NEOX over the originals (llama.cpp's convert-time
        // permute), with no kernel variant on any backend. Row reorder is quant-block-safe (blocks
        // run along the input dim, whole rows move). Returns the head count for a q/k tensor, or
        // None (no permute).
        let qk_perm_heads = |name: &str| -> Option<usize> {
            if !c.permute_qk_neox {
                return None;
            }
            if name.ends_with("attn_q.weight") || name.ends_with("attn_q.bias") {
                Some(c.n_head)
            } else if name.ends_with("attn_k.weight") || name.ends_with("attn_k.bias") {
                Some(c.n_kv)
            } else {
                None
            }
        };
        let permute_rows = |src: &[u8], heads: usize, row_b: usize| -> Vec<u8> {
            let (hd, rd) = (c.head_dim, c.rope_dim);
            let mut out = vec![0u8; src.len()];
            for h in 0..heads {
                for j in 0..hd {
                    let sj = if j < rd {
                        if j % 2 == 0 {
                            j / 2
                        } else {
                            j / 2 + rd / 2
                        }
                    } else {
                        j
                    };
                    let (d, s) = ((h * hd + j) * row_b, (h * hd + sj) * row_b);
                    out[d..d + row_b].copy_from_slice(&src[s..s + row_b]);
                }
            }
            out
        };
        let mut wload = |names: &[&str]| -> AResult<()> {
            let info = |name: &str| {
                g.tensors()
                    .iter()
                    .find(|t| t.name == name)
                    .cloned()
                    .ok_or_else(|| anyhow!("tensor not found: {name}"))
            };
            // Bytes-per-row for the permute: a weight row is `n_embd` elements of the tensor's
            // dtype (block-aligned); a bias "row" is one f32.
            let row_bytes = |name: &str, dt: DType| -> usize {
                if name.ends_with(".bias") {
                    4
                } else {
                    infr_gguf::nbytes(dt, c.n_embd)
                }
            };
            let (bytes, dt, numel) = if let [name] = names {
                let i = info(name)?;
                let tb = g.tensor_bytes_arc(name).map_err(|e| anyhow!("{e}"))?;
                let numel = i.shape.iter().product();
                match qk_perm_heads(name) {
                    Some(heads) => {
                        let rb = row_bytes(name, i.dtype);
                        (WBytes::Owned(permute_rows(&tb, heads, rb)), i.dtype, numel)
                    }
                    None => (WBytes::Mmap(tb), i.dtype, numel),
                }
            } else {
                let mut cat = Vec::new();
                let mut numel = 0usize;
                let dt = info(names[0])?.dtype;
                for name in names {
                    let i = info(name)?;
                    if i.dtype != dt {
                        return Err(anyhow!("wload concat dtype mismatch: {names:?}"));
                    }
                    numel += i.shape.iter().product::<usize>();
                    let tb = g.tensor_bytes_arc(name).map_err(|e| anyhow!("{e}"))?;
                    match qk_perm_heads(name) {
                        Some(heads) => {
                            cat.extend_from_slice(&permute_rows(&tb, heads, row_bytes(name, dt)))
                        }
                        None => cat.extend_from_slice(&tb),
                    }
                }
                (WBytes::Owned(cat), dt, numel)
            };
            // bind_weight returns the EFFECTIVE dtype the buffer holds (the GPU binder may convert float
            // weights to f16), so the graph declares the handle to match what the backend will read.
            let (buf, eff_dt) = bind_weight(names[0], bytes, dt, numel)?;
            wbufs.push(buf);
            wspecs.push((eff_dt, numel));
            Ok(())
        };
        for l in 0..c.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            // qwen35 gated-DeltaNet linear-attention layer (see docs/QWEN35.md): a wholly different
            // mixer, no q/k/v/qk_norm/attn_output/bias at all. `false` for every non-qwen35 model.
            let is_delta = c.qwen35 && !c.is_qwen35_attn_layer(l);
            wload(&[&p("attn_norm.weight")])?;
            if is_delta {
                wload(&[&p("attn_qkv.weight")])?;
                wload(&[&p("attn_gate.weight")])?;
                wload(&[&p("ssm_conv1d.weight")])?;
                wload(&[&p("ssm_alpha.weight")])?;
                wload(&[&p("ssm_beta.weight")])?;
                wload(&[&p("ssm_a")])?;
                wload(&[&p("ssm_dt.bias")])?;
                wload(&[&p("ssm_norm.weight")])?;
                wload(&[&p("ssm_out.weight")])?;
            } else if fuse_qkv {
                wload(&[
                    &p("attn_q.weight"),
                    &p("attn_k.weight"),
                    &p("attn_v.weight"),
                ])?;
            } else {
                wload(&[&p("attn_q.weight")])?;
                wload(&[&p("attn_k.weight")])?;
                if has_wv[l] {
                    wload(&[&p("attn_v.weight")])?;
                }
            }
            // Qwen2/2.5 q/k/v projection biases (small f32 [out_f] vectors). Loaded AFTER the q/k/v
            // weights so the upload order matches the `wpush` order below.
            if c.qkv_bias {
                wload(&[&p("attn_q.bias")])?;
                wload(&[&p("attn_k.bias")])?;
                wload(&[&p("attn_v.bias")])?;
            }
            if qk_norm && !is_delta {
                wload(&[&p("attn_q_norm.weight")])?;
                wload(&[&p("attn_k_norm.weight")])?;
            }
            if !is_delta {
                wload(&[&p("attn_output.weight")])?;
            }
            if gemma {
                wload(&[&p("post_attention_norm.weight")])?;
            }
            // qwen35 names its post-mixer/pre-FFN norm `post_attention_norm.weight` on BOTH layer
            // kinds (not `ffn_norm.weight`) — same role (`lw.ffn_norm`), different tensor name.
            let ffn_norm_name = if c.qwen35 {
                "post_attention_norm.weight"
            } else {
                "ffn_norm.weight"
            };
            wload(&[&p(ffn_norm_name)])?;
            if c.diffusion_gemma {
                // Dual FFN: dense GeGLU (n_ff=2112) ∥ 128-expert MoE (fused gate_up_exps + a
                // per-expert down scale), summed — see docs/DIFFUSIONGEMMA.md's FFN wiring.
                wload(&[&p("ffn_gate.weight")])?;
                wload(&[&p("ffn_up.weight")])?;
                wload(&[&p("ffn_down.weight")])?;
                wload(&[&p("post_ffw_norm_1.weight")])?;
                wload(&[&p("pre_ffw_norm_2.weight")])?;
                wload(&[&p("ffn_gate_inp.weight")])?;
                wload(&[&p("ffn_gate_inp.scale")])?;
                wload(&[&p("ffn_gate_up_exps.weight")])?;
                wload(&[&p("ffn_down_exps.weight")])?;
                wload(&[&p("ffn_down_exps.scale")])?;
                wload(&[&p("post_ffw_norm_2.weight")])?;
            } else if c.moe.is_some() {
                // qwen3moe: router + stacked per-expert gate/up/down banks.
                wload(&[&p("ffn_gate_inp.weight")])?;
                wload(&[&p("ffn_gate_exps.weight")])?;
                wload(&[&p("ffn_up_exps.weight")])?;
                wload(&[&p("ffn_down_exps.weight")])?;
            } else if fuse_gu {
                wload(&[&p("ffn_gate.weight"), &p("ffn_up.weight")])?;
                wload(&[&p("ffn_down.weight")])?;
            } else {
                wload(&[&p("ffn_gate.weight")])?;
                wload(&[&p("ffn_up.weight")])?;
                wload(&[&p("ffn_down.weight")])?;
            }
            if gemma {
                wload(&[&p("post_ffw_norm.weight")])?;
            }
            if e2b {
                // gemma4 E2B per-layer input-embedding application weights.
                wload(&[&p("inp_gate.weight")])?;
                wload(&[&p("proj.weight")])?;
                wload(&[&p("post_norm.weight")])?;
            }
        }
        // Globals: output_norm, lm_head. lm_head = `output.weight`, or (tied) the quantized
        // `token_embd.weight` mapped from the mmap and dequantized per-row by `Op::Linear` — same f32
        // values as the host `token_embd`, but zero-copy.
        wload(&["output_norm.weight"])?;
        if g.tensors().iter().any(|t| t.name == "output.weight") {
            wload(&["output.weight"])?;
        } else {
            wload(&["token_embd.weight"])?;
        }
        // diffusion-gemma: top-level self-conditioning gated MLP. LOADED (occupies a weight-buffer
        // slot like any other tensor) but NOT READ by any Op this phase — Phase 1 is the
        // encoder-only causal prefill, which runs with self-conditioning permanently off (see
        // docs/DIFFUSIONGEMMA.md); the canvas denoise graph (Phase 2+) is the first reader.
        // Loaded BEFORE the e2b block (mutually exclusive with it — no model is both) so the
        // `debug_assert_eq!` below, which indexes `wspecs` directly, isn't straddled by a later
        // `wload` call (the closure's mutable borrow of `wspecs` would conflict with that read).
        if c.diffusion_gemma {
            wload(&["self_cond_pre_norm.weight"])?;
            wload(&["self_cond_gate.weight"])?;
            wload(&["self_cond_up.weight"])?;
            wload(&["self_cond_down.weight"])?;
        }
        // gemma4 E2B: the per-layer input-embedding projection weights, native-uploaded like any
        // other weight (model_proj stays bf16 — the seam's native bf16 GEMV/GEMM reads it directly;
        // proj_norm is f32). The GPU graph prologue (in `build`, below) runs the GEMV + RMSNorm that
        // used to be a host loop. Declared here (in upload order) — `build` pushes the matching
        // handles right after `w_lm`/before `v_ones`.
        if e2b {
            wload(&["per_layer_model_proj.weight"])?;
            wload(&["per_layer_proj_norm.weight"])?;
            // Sanity-check the two uploads landed with the shapes the GPU prologue assumes
            // (model_proj is `[n_layer*npl, n_embd]`, proj_norm is `[npl]`).
            debug_assert_eq!(wspecs[wspecs.len() - 2].1, c.n_layer * npl * ne);
            debug_assert_eq!(wspecs[wspecs.len() - 1].1, npl);
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
        // diffusion-gemma: weightless FULL-WIDTH (ne-wide) RMSNorm for the MoE router's own input
        // (`rmsnorm_noscale(attn_out)`, see the graph-build wiring) — a SEPARATE ones-vector from
        // `v_ones` above (that one's per-HEAD width `max_hd`; this is the whole residual width).
        if c.diffusion_gemma {
            let ones = vec![1.0f32; ne];
            let b = be
                .alloc(ones.len() * 4, BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(b.as_ref(), bytemuck::cast_slice(&ones))
                .map_err(|e| anyhow!("{e}"))?;
            wbufs.push(b);
            wspecs.push((DType::F32, ne));
        }

        // ── persistent KV cache buffers, sized per-layer (gemma4 SWA layers are narrower) and
        //    per-side (K and V pick their dtype independently) ────────────────────────────────
        // qwen35 DeltaNet layers have NO KV cache: `kbufs[l]`/`vbufs[l]` instead hold that layer's
        // conv-history state (`[(d_conv-1), conv_channels]` f32) and DeltaNet recurrent state
        // (`[n_vhead, head_k, head_v]` f32) — fixed-size (NOT `want_ctx`-scaled) and always f32
        // regardless of the session's chosen KV dtype (see `MixerW::DeltaNet` / the `build` closure).
        let mut kbufs: Vec<Box<dyn Buffer>> = Vec::new();
        let mut vbufs: Vec<Box<dyn Buffer>> = Vec::new();
        for l in 0..c.n_layer {
            if c.qwen35 && !c.is_qwen35_attn_layer(l) {
                let conv_elems = (c.ssm_d_conv - 1) * c.q35_conv_channels();
                let s_elems = c.q35_num_v_heads() * c.q35_head_k_dim() * c.q35_head_v_dim();
                kbufs.push(
                    be.alloc(conv_elems * 4, BufferUsage::Activations)
                        .map_err(|e| anyhow!("{e}"))?,
                );
                vbufs.push(
                    be.alloc(s_elems * 4, BufferUsage::Activations)
                        .map_err(|e| anyhow!("{e}"))?,
                );
                continue;
            }
            let kvrow_l = c.layer_n_kv(l) * c.layer_head_dim(l);
            kbufs.push(
                be.alloc(
                    kv_fmt_bytes(k_fmt, want_ctx * kvrow_l),
                    BufferUsage::Activations,
                )
                .map_err(|e| anyhow!("{e}"))?,
            );
            vbufs.push(
                be.alloc(
                    kv_fmt_bytes(v_fmt, want_ctx * kvrow_l),
                    BufferUsage::Activations,
                )
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
        *state = Some(SeamKv {
            weights: std::sync::Arc::new(SeamWeights {
                wbufs,
                wspecs,
                rf_buf,
            }),
            kbufs,
            vbufs,
            k_fmt,
            v_fmt,
            hidden_buf,
            pos_buf,
            ipl_buf,
            logits_buf,
            max_ctx: want_ctx,
            cached: Vec::new(),
            denoise_cache: None,
            self_cond_w: None,
            sc_embt: None,
            sc_ping: None,
            sc_ping_write: 0,
            sc_temp_inv_buf: None,
        });
    }
    let SeamKv {
        weights,
        kbufs,
        vbufs,
        k_fmt: _,
        v_fmt: _,
        hidden_buf,
        pos_buf,
        ipl_buf,
        logits_buf,
        max_ctx,
        cached,
        denoise_cache,
        self_cond_w,
        sc_embt,
        sc_ping,
        sc_ping_write,
        sc_temp_inv_buf,
    } = state.as_mut().expect("seam state just initialized");
    let SeamWeights {
        wbufs,
        wspecs,
        rf_buf,
    } = weights.as_ref();
    let max_ctx = *max_ctx;
    if prompt.len() + max_new + 1 > max_ctx {
        return Err(anyhow!(
            "prompt {} + gen {} exceeds the session KV capacity {max_ctx}",
            prompt.len(),
            max_new
        ));
    }
    // Phase-2 DiffusionGemma denoise: capture the prompt length BEFORE the ordinary prefix-diff
    // logic below runs (a denoise call's `prompt`/`max_new` are empty/0 — see `DenoiseReq`'s
    // caller — so `start`/`cached` are left untouched: the `if denoise_req.is_some()` guard just
    // below makes both a no-op). `P` for the denoise graph is this, NOT `start`.
    let denoise_p = cached.len();
    // ChatSession-style prefix reuse: KV rows 0..start are already materialized for `cached`'s
    // shared prefix — prefill only the suffix. Always leave ≥1 prompt token to process so the
    // first generated token samples from fresh logits.
    //
    // qwen35's gated-DeltaNet recurrent state is an APPEND-ONLY summary, not a per-position cache —
    // it can't rewind to an arbitrary shared prefix the way a real KV cache can. So a turn reuses it
    // ONLY when `prompt` exactly EXTENDS `cached` (mirrors the old seam's `SeamState` rule); anything
    // else (divergent prompt, identical resend, first-ever call) zero-resets every DeltaNet layer's
    // conv/S state and re-prefills from scratch. Dense/attention models keep the generic
    // longest-common-prefix diff.
    let start = if denoise_req.is_some() {
        // No-op: a denoise call never touches `cached` (it isn't part of the prompt/generation
        // token stream) — `cached.truncate(start)` below is then a truncate-to-current-length.
        cached.len()
    } else if c.qwen35 {
        let pfx = common_prefix_len(cached, prompt);
        if pfx == cached.len() && pfx < prompt.len() {
            pfx
        } else {
            if !cached.is_empty() {
                let conv_elems = (c.ssm_d_conv - 1) * c.q35_conv_channels();
                let s_elems = c.q35_num_v_heads() * c.q35_head_k_dim() * c.q35_head_v_dim();
                for l in 0..c.n_layer {
                    if !c.is_qwen35_attn_layer(l) {
                        be.upload(
                            kbufs[l].as_ref(),
                            bytemuck::cast_slice(&vec![0f32; conv_elems]),
                        )
                        .map_err(|e| anyhow!("{e}"))?;
                        be.upload(
                            vbufs[l].as_ref(),
                            bytemuck::cast_slice(&vec![0f32; s_elems]),
                        )
                        .map_err(|e| anyhow!("{e}"))?;
                    }
                }
                cached.clear();
            }
            0
        }
    } else {
        common_prefix_len(cached, prompt).min(prompt.len() - 1)
    };
    cached.truncate(start);

    // Build a forward graph for `batch` tokens starting at absolute position `start_pos`.
    // `batch = 1` is the normal decode path; `batch > 1` is the batched-prefill path.
    // Scratch tensors scale by `batch`; the LM head runs on the last `logits_rows` tokens —
    // 1 everywhere except speculative VERIFY, which needs the distribution after every
    // candidate (logits output = [logits_rows, vocab], logits_rows ∈ {1, batch}).
    // `denoise`: build the DiffusionGemma canvas-denoise variant of this layer stack instead of
    // the ordinary causal forward — see docs/DIFFUSIONGEMMA.md's "Seam extensions". `batch` is the
    // canvas length C, `start_pos` the prompt length P (unchanged meaning: WriteKv still lands at
    // row P, Attention's kv_len is still `start_pos+batch` = P+C, positions are still bound
    // per-row P..P+C-1 by the caller) — ONLY the attention mask and the per-layer output scalar
    // change. Never true for any existing caller (all pass `false`).
    // `gpu_sc`: Phase-B/D perf, DiffusionGemma in-graph self-conditioning (see
    // docs/DIFFUSIONGEMMA.md's Phase-B and the reference's `dg_canvas_embed`) — `None` for every
    // ordinary caller (CPU denoise included: it keeps the Phase-A host `diffusion_self_cond`
    // path, so `hidden` is already the fully-baked residual). `Some(sc_on)` from the Vulkan AND
    // Metal denoise call sites: `hidden` there holds the RAW scaled canvas embedding (no SC add, no
    // norm) and this flag additionally emits the SC subgraph (`sc_on == true`) and/or the
    // weightless canvas-embed post-norm (always, when `Some`) INSIDE the graph. Baked into the
    // cached plan (the `(cc,p,sc_on)` key — see `DenoiseCache`) rather than a runtime gate, so the
    // compiled graph never branches at execute time.
    // `h_tap` (MTP Phase 1, issue #33): also expose the LM-head input as a second graph Output
    // (`DecodeHandles::h_out`) — see that field's doc. `false` for every existing call site;
    // `true` only from the decode loop / speculative-VERIFY call sites below when the caller
    // passed `h_out: Some(_)` to `generate_dense_backend`.
    // `dyn_sc_scale` (Vulkan-only perf, docs/DIFFUSIONGEMMA.md's Phase-B "sc round-trip"
    // elimination): only meaningful when `gpu_sc == Some(true)`. `true` declares an EXTRA 1-element
    // Input (`DecodeHandles::temp_inv`) and wires it as the SC subgraph's `Op::Softmax::scale_buf`
    // instead of baking `scale: 1.0` — the denoise call site uploads a 4-byte scalar there each
    // step instead of premultiplying + reuploading the whole `[batch, vocab]` `sc_logits_in`.
    // `false` for every other call site (Metal keeps the original host-premultiply path).
    let build = |batch: usize,
                 start_pos: usize,
                 logits_rows: usize,
                 denoise: bool,
                 gpu_sc: Option<bool>,
                 dyn_sc_scale: bool,
                 h_tap: bool|
     -> (Graph, DecodeHandles) {
        let mut g = Graph::new();
        let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
        // KV cache dtype: f16 by default (halves memory vs f32, tightens CPU↔GPU parity); Q8_0
        // per-side when the runner enabled it (see `k_fmt`/`v_fmt` at the cache alloc). ONLY the
        // persistent caches take this dtype — the roped q16/k16 staging stays f16
        // (`qk_norm_rope`/`rope_f16` write f16; a Q8_0 decl there would lie to any backend that
        // trusts it, and the Vulkan kv-write peephole fuses on the f16 decl).
        let kd = |n: usize| TensorDesc::new(vec![n], k_fmt);
        let vd = |n: usize| TensorDesc::new(vec![n], v_fmt);
        let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16);
        let hidden = g.input(f32d(batch * ne));
        let positions = g.input(TensorDesc::new(vec![batch], DType::I32));
        let rope_freqs = rf_buf.as_ref().map(|(_, n)| g.input(f32d(*n)));
        // gemma4 E2B per-(token,layer) TOKEN embedding rows `[batch, n_layer*npl]` — host-gathered
        // + dequanted (the big `per_layer_token_embd` table stays off-VRAM, gathered per token).
        // The full `per_layer_inp` consumed by the layer loop is computed from this on the GPU
        // (model_proj GEMV + RMSNorm), further down, once its weights are declared.
        let pl_tok_in = if e2b {
            Some(g.input(f32d(batch * c.n_layer * npl)))
        } else {
            None
        };
        // Phase-B perf: previous-step canvas logits `[batch, vocab]`, premultiplied by temp_inv on
        // the HOST before upload (keeps `Op::Softmax`'s `scale` a constant 1.0 across steps whose
        // temp_inv legitimately changes, so this same plan replays — see the denoise call site).
        let sc_logits_in = if gpu_sc == Some(true) {
            Some(g.input(f32d(batch * c.vocab)))
        } else {
            None
        };
        // Vulkan-only perf: the SC softmax's per-step temperature divisor, read from a 4-byte
        // device buffer instead of baked into the plan (see `dyn_sc_scale`'s doc above and
        // `Op::Softmax::scale_buf`). `None` keeps the SC subgraph's Softmax at the original
        // `scale: 1.0` (host-premultiplied `sc_logits_in` — Metal's path).
        let temp_inv_id = if gpu_sc == Some(true) && dyn_sc_scale {
            Some(g.input(f32d(1)))
        } else {
            None
        };
        // qwen35 DeltaNet layers have no KV cache — `k_cache[l]`/`v_cache[l]` instead declare that
        // layer's conv-state / DeltaNet-S-state Inputs (see the matching alloc in
        // `generate_dense_backend` and `MixerW::DeltaNet`'s use of them below).
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        for l in 0..c.n_layer {
            if c.qwen35 && !c.is_qwen35_attn_layer(l) {
                let conv_elems = (c.ssm_d_conv - 1) * c.q35_conv_channels();
                let s_elems = c.q35_num_v_heads() * c.q35_head_k_dim() * c.q35_head_v_dim();
                k_cache.push(g.input(f32d(conv_elems)));
                v_cache.push(g.input(f32d(s_elems)));
                continue;
            }
            let kvrow_l = c.layer_n_kv(l) * c.layer_head_dim(l);
            k_cache.push(g.input(kd(max_ctx * kvrow_l)));
            v_cache.push(g.input(vd(max_ctx * kvrow_l)));
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
            // qwen35 gated-DeltaNet layer: 9 mixer weights, no q/k/v/qk_norm/bias/wo at all (mirrors
            // the `wload` skip above). `is_delta` is `false` for every non-qwen35 model.
            let is_delta = c.qwen35 && !c.is_qwen35_attn_layer(l);
            let mixer = if is_delta {
                let qkv = wpush(&mut g, &mut weights);
                let gate = wpush(&mut g, &mut weights);
                let conv1d = wpush(&mut g, &mut weights);
                let alpha = wpush(&mut g, &mut weights);
                let beta = wpush(&mut g, &mut weights);
                let ssm_a = wpush(&mut g, &mut weights);
                let dt_bias = wpush(&mut g, &mut weights);
                let ssm_norm = wpush(&mut g, &mut weights);
                let out = wpush(&mut g, &mut weights);
                MixerW::DeltaNet(DeltaW {
                    qkv,
                    gate,
                    conv1d,
                    alpha,
                    beta,
                    ssm_a,
                    dt_bias,
                    ssm_norm,
                    out,
                })
            } else {
                // Fused QKV: ONE concatenated weight handle serves q/k/v (the builder bakes each
                // projection's `w_off` slice); split form declares three.
                let (wq, wk, wv) = if fuse_qkv {
                    let wqkv = wpush(&mut g, &mut weights);
                    (wqkv, wqkv, Some(wqkv))
                } else {
                    let wq = wpush(&mut g, &mut weights);
                    let wk = wpush(&mut g, &mut weights);
                    let wv = if has_wv[l] {
                        Some(wpush(&mut g, &mut weights))
                    } else {
                        None
                    };
                    (wq, wk, wv)
                };
                // Qwen2 q/k/v biases — pushed here to match the `wload` order (after the q/k/v
                // weights, before qk_norm). Always three separate handles (they add to the SPLIT
                // q/k/v buffers, independent of whether the weights were fused).
                let (qb, kb, vb) = if c.qkv_bias {
                    (
                        Some(wpush(&mut g, &mut weights)),
                        Some(wpush(&mut g, &mut weights)),
                        Some(wpush(&mut g, &mut weights)),
                    )
                } else {
                    (None, None, None)
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
                MixerW::Attn(AttnW {
                    wq,
                    wk,
                    wv,
                    qb,
                    kb,
                    vb,
                    q_norm,
                    k_norm,
                    wo,
                })
            };
            let post_attn = if gemma {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let ffn_norm = wpush(&mut g, &mut weights);
            let ffn = if c.diffusion_gemma {
                FfnW::DiffusionMoe {
                    d_gate: wpush(&mut g, &mut weights),
                    d_up: wpush(&mut g, &mut weights),
                    d_down: wpush(&mut g, &mut weights),
                    d_post_norm: wpush(&mut g, &mut weights),
                    m_pre_norm: wpush(&mut g, &mut weights),
                    router: wpush(&mut g, &mut weights),
                    router_scale: wpush(&mut g, &mut weights),
                    gate_up_exps: wpush(&mut g, &mut weights),
                    down_exps: wpush(&mut g, &mut weights),
                    down_scale: wpush(&mut g, &mut weights),
                    m_post_norm: wpush(&mut g, &mut weights),
                }
            } else if c.moe.is_some() {
                FfnW::Moe {
                    router: wpush(&mut g, &mut weights),
                    gate_exps: wpush(&mut g, &mut weights),
                    up_exps: wpush(&mut g, &mut weights),
                    down_exps: wpush(&mut g, &mut weights),
                }
            } else if fuse_gu {
                FfnW::DenseFused {
                    wgu: wpush(&mut g, &mut weights),
                    wdown: wpush(&mut g, &mut weights),
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
                mixer,
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
        // diffusion-gemma: self-conditioning gated-MLP handles — declared to match `wload`'s
        // upload order (right after lm_head, before the e2b projection weights). Read by the
        // in-graph SC subgraph below when `gpu_sc == Some(true)`; harmlessly unread otherwise
        // (every other build, including the CPU denoise path, which computes SC on the
        // host — see `diffusion_self_cond`).
        let (sc_pre_norm_id, sc_gate_id, sc_up_id, sc_down_id) = if c.diffusion_gemma {
            (
                Some(wpush(&mut g, &mut weights)),
                Some(wpush(&mut g, &mut weights)),
                Some(wpush(&mut g, &mut weights)),
                Some(wpush(&mut g, &mut weights)),
            )
        } else {
            (None, None, None, None)
        };
        // Phase-B perf: the SC soft-embedding weight — `token_embd` dequantized + TRANSPOSED to
        // f16 `[n_embd, n_vocab]` (row e holds embedding dim e across every vocab token; the
        // reference's `sc_embT` / `dg_ensure_sc_embT`). NOT a GGUF tensor (`wpush` doesn't cover
        // it) — built once on the host from the already-dequantized `token_embd` and bound
        // separately by the denoise call site (see `SeamKv::sc_embt`).
        let sc_embt_id = if gpu_sc == Some(true) {
            Some(g.weight(TensorDesc::new(vec![c.vocab * ne], DType::F16)))
        } else {
            None
        };
        // gemma4 E2B per-layer input-embedding projection weights — declared here to match the
        // `wload` upload order (right after lm_head/self_cond, before the gemma4 V-norm ones-vector).
        let (mp_w, pn_w) = if e2b {
            (
                Some(wpush(&mut g, &mut weights)),
                Some(wpush(&mut g, &mut weights)),
            )
        } else {
            (None, None)
        };
        let v_ones = if gemma4 {
            Some(wpush(&mut g, &mut weights))
        } else {
            None
        };
        // diffusion-gemma: weightless full-width (ne) RMSNorm ones-vector for the MoE router's own
        // input — see the matching upload in `generate_dense_backend`'s init block.
        let router_ones = if c.diffusion_gemma {
            Some(wpush(&mut g, &mut weights))
        } else {
            None
        };
        let logits = g.output(f32d(c.vocab * logits_rows));

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
        // Fused-QKV prefill staging: the wide GEMM writes [batch, qrow+2·kvrow] here, then three
        // CopyStrided ops split it into q/k/v. Decode (batch==1) skips it (offset GEMVs).
        let qkvbuf = if fuse_qkv && batch > 1 {
            Some(g.internal(f32d(batch * (max_qrow + 2 * max_kvrow))))
        } else {
            None
        };
        // Separate gate/up scratch, or one combined [batch, 2*nff] gu buffer when fused — declare
        // only the shape in use (Internal buffers are allocated by the backend even if never read).
        let (gbuf, ubuf, gubuf) = if fuse_gu {
            let gu = g.internal(f32d(batch * 2 * nff));
            (gu, gu, gu)
        } else {
            let gb = g.internal(f32d(batch * nff));
            let ub = g.internal(f32d(batch * nff));
            (gb, ub, gb)
        };
        let actbuf = g.internal(f32d(batch * nff));
        let sub = g.internal(f32d(batch * ne));
        // E2B per-layer embed scratch: gate `[npl]` and projected `[ne]`.
        let plg = g.internal(f32d(batch * npl.max(1)));
        let plp = g.internal(f32d(batch * ne));

        // diffusion-gemma dual-FFN scratch (see docs/DIFFUSIONGEMMA.md's FFN wiring): the dense
        // branch's own output (`d_out`, before summing with the MoE branch), the router's own
        // input row (`router_tmp` — a DIFFERENT normalization of `attn_out` than either FFN
        // branch reads), the MoE branch's input (`moe_in`) and raw output (`moe_out`). Harmlessly
        // allocated (but unused) on every other arch, like the E2B/qwen35 scratch above.
        let d_out = g.internal(f32d(batch * ne));
        let router_tmp = g.internal(f32d(batch * ne));
        let moe_in = g.internal(f32d(batch * ne));
        let moe_out = g.internal(f32d(batch * ne));

        // qwen35 attention out-gate scratch (the interleaved q+gate trap — see docs/QWEN35.md):
        // `qg` holds the RAW `attn_q` projection (`[batch, nh*2*hd]`, q and gate interleaved per
        // head); `gate_a` holds the split-out gate, packed like `q` (`[batch, nh*hd]`), consumed by
        // the post-attention `GatedAct(Sigmoid)`. Unused (but harmlessly allocated) on every other
        // arch, exactly like the E2B scratch above.
        let qg = g.internal(f32d(batch * max_qrow * 2));
        let gate_a = g.internal(f32d(batch * max_qrow));

        // qwen35 gated-DeltaNet mixer scratch (see docs/QWEN35.md), reused across every DeltaNet
        // layer exactly like `hn`/`sub` above (qwen35's SSM dims are uniform across layers, unlike
        // gemma4's per-layer varying attention dims). `.max(1)`-guarded so a non-qwen35 model (every
        // q35_* dim is 0) still gets a valid, harmlessly-tiny allocation.
        let q35_cc = c.q35_conv_channels();
        let q35_di = c.ssm_d_inner;
        let q35_nk = c.q35_num_k_heads();
        let q35_kd = c.q35_head_k_dim();
        let q35_nv = c.q35_num_v_heads();
        let q35_vd = c.q35_head_v_dim();
        let q35_keydim = q35_nk * q35_kd;
        let dn_qkvbuf = g.internal(f32d(batch * q35_cc.max(1)));
        let dn_zbuf = g.internal(f32d(batch * q35_di.max(1)));
        let dn_convout = g.internal(f32d(batch * q35_cc.max(1)));
        let dn_qbuf = g.internal(f32d(batch * q35_keydim.max(1)));
        let dn_kbuf = g.internal(f32d(batch * q35_keydim.max(1)));
        let dn_vbuf = g.internal(f32d(batch * (q35_nv * q35_vd).max(1)));
        let dn_bbuf = g.internal(f32d(batch * q35_nv.max(1)));
        let dn_abuf = g.internal(f32d(batch * q35_nv.max(1)));
        let dn_out = g.internal(f32d(batch * (q35_nv * q35_vd).max(1)));

        let eps = c.rms_eps;

        // gemma4 E2B prologue: compute the full per-(token,layer) input vector `per_layer_inp`
        // ([batch, n_layer*npl]) that the layer loop below consumes, on the GPU — matches
        // llama.cpp's split (host: gather+dequant the per-layer token embedding row; GPU: the
        // model_proj GEMV + RMSNorm + combine). `hidden` here is already the scaled token
        // embedding (`emb = token_embd[tok] * embed_scale`), so it IS the `emb` the host version
        // used to dot against `model_proj`.
        let per_layer_inp =
            if let (Some(mp_w), Some(pn_w), Some(pl_tok_in)) = (mp_w, pn_w, pl_tok_in) {
                let nlnpl = c.n_layer * npl;
                let acc = g.internal(f32d(batch * nlnpl));
                g.push(Op::Linear {
                    x: hidden,
                    weight: mp_w,
                    dst: acc,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: nlnpl as u32,
                    w_off: 0,
                });
                g.push(Op::Scale {
                    x: acc,
                    dst: acc,
                    s: 1.0 / (ne as f32).sqrt(),
                    n: (batch * nlnpl) as u32,
                });
                let normed = g.internal(f32d(batch * nlnpl));
                g.push(Op::RmsNorm {
                    x: acc,
                    weight: pn_w,
                    dst: normed,
                    rows: (batch * c.n_layer) as u32,
                    dim: npl as u32,
                    eps,
                });
                let ipl = g.internal(f32d(batch * nlnpl));
                g.push(Op::Add {
                    a: normed,
                    b: pl_tok_in,
                    dst: ipl,
                    n: (batch * nlnpl) as u32,
                });
                g.push(Op::Scale {
                    x: ipl,
                    dst: ipl,
                    s: 1.0 / 2f32.sqrt(),
                    n: (batch * nlnpl) as u32,
                });
                Some(ipl)
            } else {
                None
            };

        // Phase-B perf: DiffusionGemma in-graph canvas embedding (ported from the reference's
        // `dg_canvas_embed` in diffusion-gemma.cpp — see docs/DIFFUSIONGEMMA.md's Phase-B).
        // `hidden` at this point holds ONLY the raw scaled canvas embedding
        // (`embed(tok)·√n_embd`, no SC add, no norm — the host caller uploads exactly that
        // instead of the Phase-A fully-baked residual). Runs BEFORE the layer loop below, which
        // reads/mutates `hidden` in place exactly as every other caller's — no change needed
        // there.
        if let Some(sc_on) = gpu_sc {
            if sc_on {
                let sc_logits_in =
                    sc_logits_in.expect("gpu_sc(true) plan always declares sc_logits_in");
                let sc_embt = sc_embt_id.expect("gpu_sc(true) plan always declares sc_embt_id");
                let (sc_pre_norm_id, sc_gate_id, sc_up_id, sc_down_id) = (
                    sc_pre_norm_id.expect("diffusion-gemma always declares sc_pre_norm_id"),
                    sc_gate_id.expect("diffusion-gemma always declares sc_gate_id"),
                    sc_up_id.expect("diffusion-gemma always declares sc_up_id"),
                    sc_down_id.expect("diffusion-gemma always declares sc_down_id"),
                );
                let vocab = c.vocab;
                // probs = softmax(sc_logits * scale). `dyn_sc_scale`: `scale_buf` reads temp_inv
                // from a device buffer per step (Vulkan) — `scale: 1.0` below is then ignored.
                // Otherwise temp_inv was already applied on the host, so `scale: 1.0` is the real
                // value (see `sc_logits_in`'s doc / Metal's path).
                let probs = g.internal(f32d(batch * vocab));
                g.push(Op::Softmax {
                    x: sc_logits_in,
                    dst: probs,
                    rows: batch as u32,
                    dim: vocab as u32,
                    scale: 1.0,
                    scale_buf: temp_inv_id,
                });
                // soft = (probs @ sc_embT) * sqrt(n_embd) — sc_embT is [n_embd, n_vocab]
                // (Op::Linear's `weight: [out_f, in_f]` convention), so this is exactly the
                // reference's `ggml_mul_mat(sc_embT, probs)`.
                let soft = g.internal(f32d(batch * ne));
                g.push(Op::Linear {
                    x: probs,
                    weight: sc_embt,
                    dst: soft,
                    m: batch as u32,
                    in_f: vocab as u32,
                    out_f: ne as u32,
                    w_off: 0,
                });
                g.push(Op::Scale {
                    x: soft,
                    dst: soft,
                    s: (ne as f32).sqrt(),
                    n: (batch * ne) as u32,
                });
                // sc_pre_norm: a NORMAL (weighted) rmsnorm — unlike the canvas embedding's
                // weightless one below.
                let sc_normed = g.internal(f32d(batch * ne));
                g.push(Op::RmsNorm {
                    x: soft,
                    weight: sc_pre_norm_id,
                    dst: sc_normed,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
                // Gated-GELU MLP: down(gelu_tanh(gate·normed) * (up·normed)).
                let sc_g = g.internal(f32d(batch * nff));
                let sc_u = g.internal(f32d(batch * nff));
                g.push(Op::Linear {
                    x: sc_normed,
                    weight: sc_gate_id,
                    dst: sc_g,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: nff as u32,
                    w_off: 0,
                });
                g.push(Op::Linear {
                    x: sc_normed,
                    weight: sc_up_id,
                    dst: sc_u,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: nff as u32,
                    w_off: 0,
                });
                let sc_act = g.internal(f32d(batch * nff));
                g.push(Op::GatedAct {
                    gate: sc_g,
                    up: sc_u,
                    dst: sc_act,
                    rows: batch as u32,
                    nff: nff as u32,
                    act: Activation::Gelu,
                    up_off: 0,
                });
                let sc_sig = g.internal(f32d(batch * ne));
                g.push(Op::Linear {
                    x: sc_act,
                    weight: sc_down_id,
                    dst: sc_sig,
                    m: batch as u32,
                    in_f: nff as u32,
                    out_f: ne as u32,
                    w_off: 0,
                });
                g.push(Op::Add {
                    a: hidden,
                    b: sc_sig,
                    dst: hidden,
                    n: (batch * ne) as u32,
                });
            }
            // weightless canvas-embed post-norm (no scale weight — matches `dg_canvas_embed`
            // exactly). Reuses `router_ones` (the same ne-wide ones vector diffusion-gemma
            // already declares for the MoE router's weightless norm) instead of a new weight.
            let ones =
                router_ones.expect("diffusion-gemma gpu_sc build always declares router_ones");
            g.push(Op::RmsNorm {
                x: hidden,
                weight: ones,
                dst: hidden,
                rows: batch as u32,
                dim: ne as u32,
                eps,
            });
        }

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
            // DiffusionGemma canvas denoise (docs/DIFFUSIONGEMMA.md): every canvas query attends
            // the SAME fixed bidirectional range `[lo, kv_len)` — `lo = 0` on full-attention
            // layers (every prompt + canvas key visible), `lo = max(0, P-(n_swa-1))` on SWA
            // layers (only the last `n_swa-1` prompt positions, but every canvas key — canvas
            // keys live in `[P, kv_len)` ⊆ `[lo, kv_len)` on both layer types since `lo <= P`).
            // `start_pos` IS `P` here (the denoise batch starts right after the cached prompt).
            let mask = if denoise {
                let lo = if swa {
                    start_pos.saturating_sub(c.swa_window.saturating_sub(1))
                } else {
                    0
                };
                AttnMask::Canvas { lo }
            } else if swa {
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
            // gemma4 E2B KV-layer sharing: shared layers compute Q only and attend to an earlier
            // layer's cache. `own_kv`/`kv_src` are `true`/`l` for every layer of a non-sharing model.
            let own_kv = c.has_own_kv(l);
            let kv_src = c.kv_src_layer(l);
            if let MixerW::DeltaNet(dw) = &lw.mixer {
                // gated-DeltaNet linear attention (see docs/QWEN35.md) — no KV cache; the
                // recurrent state lives in `k_cache[l]`/`v_cache[l]` (repurposed as
                // conv_state/s_state, see the matching alloc in `generate_dense_backend`).
                g.push(Op::Linear {
                    x: hn,
                    weight: dw.qkv,
                    dst: dn_qkvbuf,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: q35_cc as u32,
                    w_off: 0,
                });
                g.push(Op::Linear {
                    x: hn,
                    weight: dw.gate,
                    dst: dn_zbuf,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: q35_di as u32,
                    w_off: 0,
                });
                g.push(Op::Conv1dSilu {
                    x: dn_qkvbuf,
                    weight: dw.conv1d,
                    state: k_cache[l],
                    dst: dn_convout,
                    rows: batch as u32,
                    channels: q35_cc as u32,
                    kernel: c.ssm_d_conv as u32,
                });
                // split conv_out [batch, cc=q|k|v] → packed [batch, *] q / k / v (strided/token).
                g.push(Op::CopyStrided {
                    src: dn_convout,
                    src_off: 0,
                    src_stride: q35_cc as u32,
                    dst: dn_qbuf,
                    dst_off: 0,
                    dst_stride: q35_keydim as u32,
                    rows: batch as u32,
                    n: q35_keydim as u32,
                });
                g.push(Op::CopyStrided {
                    src: dn_convout,
                    src_off: q35_keydim as u32,
                    src_stride: q35_cc as u32,
                    dst: dn_kbuf,
                    dst_off: 0,
                    dst_stride: q35_keydim as u32,
                    rows: batch as u32,
                    n: q35_keydim as u32,
                });
                g.push(Op::CopyStrided {
                    src: dn_convout,
                    src_off: (2 * q35_keydim) as u32,
                    src_stride: q35_cc as u32,
                    dst: dn_vbuf,
                    dst_off: 0,
                    dst_stride: (q35_nv * q35_vd) as u32,
                    rows: batch as u32,
                    n: (q35_nv * q35_vd) as u32,
                });
                g.push(Op::Linear {
                    x: hn,
                    weight: dw.beta,
                    dst: dn_bbuf,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: q35_nv as u32,
                    w_off: 0,
                });
                g.push(Op::Linear {
                    x: hn,
                    weight: dw.alpha,
                    dst: dn_abuf,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: q35_nv as u32,
                    w_off: 0,
                });
                g.push(Op::DeltaNet {
                    q: dn_qbuf,
                    k: dn_kbuf,
                    v: dn_vbuf,
                    b: dn_bbuf,
                    a: dn_abuf,
                    a_coef: dw.ssm_a,
                    dt_bias: dw.dt_bias,
                    state: v_cache[l],
                    dst: dn_out,
                    rows: batch as u32,
                    n_vhead: q35_nv as u32,
                    n_khead: q35_nk as u32,
                    head_k: q35_kd as u32,
                    head_v: q35_vd as u32,
                    eps: 1e-6,
                });
                // silu-gated RMSNorm per v-head: rmsnorm(out, ssm_norm) then * silu(z)
                g.push(Op::QkNorm {
                    x: dn_out,
                    weight: dw.ssm_norm,
                    dst: dn_out,
                    rows: batch as u32,
                    n_head: q35_nv as u32,
                    head_dim: q35_vd as u32,
                    eps,
                });
                g.push(Op::GatedAct {
                    gate: dn_zbuf,
                    up: dn_out,
                    dst: dn_out,
                    rows: batch as u32,
                    nff: (q35_nv * q35_vd) as u32,
                    act: Activation::Silu,
                    up_off: 0,
                });
                g.push(Op::Linear {
                    x: dn_out,
                    weight: dw.out,
                    dst: sub,
                    m: batch as u32,
                    in_f: q35_di as u32,
                    out_f: ne as u32,
                    w_off: 0,
                });
                // DeltaNet's residual contribution is already in `sub` — skip the attention-only
                // code below (query/key/value projections, RoPE, Attention, o-proj) entirely.
            } else {
                let MixerW::Attn(aw) = &lw.mixer else {
                    unreachable!("qwen35 DeltaNet handled above")
                };
                if let Some(qkv) = qkvbuf {
                    // Fused QKV (prefill): ONE wide GEMM over the concatenated weight — the separate
                    // q/k/v GEMMs are narrow-n and underfill the GPU — then split rows into q/k/v.
                    let stride = (qrow + 2 * kvrow) as u32;
                    g.push(Op::Linear {
                        x: hn,
                        weight: aw.wq,
                        dst: qkv,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: stride,
                        w_off: 0,
                    });
                    for (dst, off, n) in [
                        (q, 0u32, qrow as u32),
                        (k, qrow as u32, kvrow as u32),
                        (v, (qrow + kvrow) as u32, kvrow as u32),
                    ] {
                        g.push(Op::CopyStrided {
                            src: qkv,
                            src_off: off,
                            src_stride: stride,
                            dst,
                            dst_off: 0,
                            dst_stride: n,
                            rows: batch as u32,
                            n,
                        });
                    }
                } else if fuse_qkv {
                    // Fused QKV (decode): three offset GEMVs into the concatenated weight — the same
                    // dispatch count as the split form, no staging copies.
                    for (dst, off, n) in [
                        (q, 0usize, qrow),
                        (k, qrow * ne, kvrow),
                        (v, (qrow + kvrow) * ne, kvrow),
                    ] {
                        g.push(Op::Linear {
                            x: hn,
                            weight: aw.wq,
                            dst,
                            m: batch as u32,
                            in_f: ne as u32,
                            out_f: n as u32,
                            w_off: off as u32,
                        });
                    }
                } else if c.attn_out_gate {
                    // qwen35 attention layers pack q + a SIGMOID output gate INTERLEAVED per head in
                    // `attn_q` (`[h0 q(hd) | h0 gate(hd) | h1 q | h1 gate | …]`, NOT two contiguous
                    // blocks — see docs/QWEN35.md). Project into `qg` (width 2*qrow) then split each
                    // head's two halves into the packed `q` / `gate_a` scratch.
                    g.push(Op::Linear {
                        x: hn,
                        weight: aw.wq,
                        dst: qg,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: (qrow * 2) as u32,
                        w_off: 0,
                    });
                    for h in 0..nh {
                        g.push(Op::CopyStrided {
                            src: qg,
                            src_off: (h * 2 * hd) as u32,
                            src_stride: (nh * 2 * hd) as u32,
                            dst: q,
                            dst_off: (h * hd) as u32,
                            dst_stride: (nh * hd) as u32,
                            rows: batch as u32,
                            n: hd as u32,
                        });
                        g.push(Op::CopyStrided {
                            src: qg,
                            src_off: (h * 2 * hd + hd) as u32,
                            src_stride: (nh * 2 * hd) as u32,
                            dst: gate_a,
                            dst_off: (h * hd) as u32,
                            dst_stride: (nh * hd) as u32,
                            rows: batch as u32,
                            n: hd as u32,
                        });
                    }
                } else {
                    g.push(Op::Linear {
                        x: hn,
                        weight: aw.wq,
                        dst: q,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: qrow as u32,
                        w_off: 0,
                    });
                }
                // Qwen2 q-bias: `q += qb` after the projection (all three projection paths converge on
                // `q` here), before RoPE. `Wx + b`.
                if let Some(qb) = aw.qb {
                    g.push(Op::AddBias {
                        x: q,
                        bias: qb,
                        dst: q,
                        rows: batch as u32,
                        n: qrow as u32,
                    });
                }
                if own_kv {
                    if !fuse_qkv {
                        g.push(Op::Linear {
                            x: hn,
                            weight: aw.wk,
                            dst: k,
                            m: batch as u32,
                            in_f: ne as u32,
                            out_f: kvrow as u32,
                            w_off: 0,
                        });
                        // V projection, or (gemma4 full layers) V = the raw K projection, copied BEFORE
                        // K is QK-normed + RoPE'd.
                        match aw.wv {
                            Some(wv) => g.push(Op::Linear {
                                x: hn,
                                weight: wv,
                                dst: v,
                                m: batch as u32,
                                in_f: ne as u32,
                                out_f: kvrow as u32,
                                w_off: 0,
                            }),
                            None => g.push(Op::Copy {
                                src: k,
                                src_off: 0,
                                dst: v,
                                dst_off: 0,
                                n: (batch * kvrow) as u32,
                            }),
                        }
                    }
                    // Qwen2 k/v-bias: `k += kb`, `v += vb` after the projections (here q/k/v are all
                    // materialized in every path — fused prefill/decode projected k/v above, the split
                    // form just did), BEFORE the K RoPE and the V-norm/WriteKv. Emitted before the K
                    // QkNormRope so that op stays adjacent to its WriteKv (see below).
                    if let Some(kb) = aw.kb {
                        g.push(Op::AddBias {
                            x: k,
                            bias: kb,
                            dst: k,
                            rows: batch as u32,
                            n: kvrow as u32,
                        });
                    }
                    if let Some(vb) = aw.vb {
                        g.push(Op::AddBias {
                            x: v,
                            bias: vb,
                            dst: v,
                            rows: batch as u32,
                            n: kvrow as u32,
                        });
                    }
                    // gemma4 weightless per-head RMSNorm on V (= x/rms) before caching. Emitted BEFORE
                    // the K QkNormRope so that op stays ADJACENT to its WriteKv — the Vulkan adapter's
                    // kv_write_peephole only fuses an immediately-following pair, and the record-once
                    // decode path REQUIRES the K write fused (a standalone f16 WriteKv has no dyn
                    // kernel). V only depends on the raw K projection, so the order is free.
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
                    // K: fused QkNorm+RoPE (qwen3/gemma) → f16 `k16`, else RoPE alone (llama) in-place f32.
                    let k_write = match aw.k_norm {
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
                            // llama (no k-norm): interleaved RoPE straight to the f16 scratch — the
                            // same fused shape as the qk-norm path, so the Vulkan peephole redirects
                            // the write into the KV cache and the decode replays via rope_f16_dyn.
                            g.push(Op::Rope {
                                x: k,
                                positions,
                                dst: k16,
                                rows: batch as u32,
                                n_head: nkv as u32,
                                head_dim: hd as u32,
                                rope_dim: rope_dim as u32,
                                theta,
                                freq_factors: layer_ff,
                            });
                            k16
                        }
                    };
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
                let q_attn = match aw.q_norm {
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
                        // llama: Q roped to the f16 scratch (the attention kernels read f16 q).
                        g.push(Op::Rope {
                            x: q,
                            positions,
                            dst: q16,
                            rows: batch as u32,
                            n_head: nh as u32,
                            head_dim: hd as u32,
                            rope_dim: rope_dim as u32,
                            theta,
                            freq_factors: layer_ff,
                        });
                        q16
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
                // qwen35: per-head SIGMOID output gate applied to the attention output BEFORE the
                // o-projection (`gate_a` was split out of the interleaved `attn_q` projection above).
                if c.attn_out_gate {
                    g.push(Op::GatedAct {
                        gate: gate_a,
                        up: attn,
                        dst: attn,
                        rows: batch as u32,
                        nff: qrow as u32,
                        act: Activation::Sigmoid,
                        up_off: 0,
                    });
                }
                g.push(Op::Linear {
                    x: attn,
                    weight: aw.wo,
                    dst: sub,
                    m: batch as u32,
                    in_f: qrow as u32,
                    out_f: ne as u32,
                    w_off: 0,
                });
            } // else (MixerW::Attn) — matches the `if let MixerW::DeltaNet` above
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
                FfnW::DenseFused { wgu, wdown } => {
                    g.push(Op::Linear {
                        x: hn,
                        weight: wgu,
                        dst: gubuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: (2 * nff_l) as u32,
                        w_off: 0,
                    });
                    g.push(Op::GatedActFused {
                        gu: gubuf,
                        dst: actbuf,
                        rows: batch as u32,
                        nff: nff_l as u32,
                        act,
                    });
                    g.push(Op::Linear {
                        x: actbuf,
                        weight: wdown,
                        dst: sub,
                        m: batch as u32,
                        in_f: nff_l as u32,
                        out_f: ne as u32,
                        w_off: 0,
                    });
                }
                FfnW::Dense { wgate, wup, wdown } => {
                    g.push(Op::Linear {
                        x: hn,
                        weight: wgate,
                        dst: gbuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                        w_off: 0,
                    });
                    g.push(Op::Linear {
                        x: hn,
                        weight: wup,
                        dst: ubuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                        w_off: 0,
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
                        w_off: 0,
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
                        router_x: hn, // qwen3moe: router reads the SAME normed input as the experts
                        router,
                        gate_exps,
                        up_exps,
                        down_exps,
                        down_scale: None,
                        fused_gate_up: false,
                        dst: sub,
                        ne: ne as u32,
                        n_expert: mc.n_expert as u32,
                        n_used: mc.n_used as u32,
                        n_ff_exp: mc.n_ff_exp as u32,
                        scale: mc.scale,
                        act, // qwen3moe: SwiGLU (act == Silu)
                    });
                }
                FfnW::DiffusionMoe {
                    d_gate,
                    d_up,
                    d_down,
                    d_post_norm,
                    m_pre_norm,
                    router,
                    router_scale,
                    gate_up_exps,
                    down_exps,
                    down_scale,
                    m_post_norm,
                } => {
                    let mc = c.moe.expect("diffusion-gemma layer without MoeConfig");
                    // Dense branch (the "shared expert"): GELU-par gate/up/down on `hn` (already
                    // ffn_norm(attn_out) from above), then its own post-norm. `act` is Gelu here —
                    // gemma implies it (see the `act` computation above).
                    g.push(Op::Linear {
                        x: hn,
                        weight: d_gate,
                        dst: gbuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                        w_off: 0,
                    });
                    g.push(Op::Linear {
                        x: hn,
                        weight: d_up,
                        dst: ubuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                        w_off: 0,
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
                        weight: d_down,
                        dst: d_out,
                        m: batch as u32,
                        in_f: nff_l as u32,
                        out_f: ne as u32,
                        w_off: 0,
                    });
                    g.push(Op::RmsNorm {
                        x: d_out,
                        weight: d_post_norm,
                        dst: d_out,
                        rows: batch as u32,
                        dim: ne as u32,
                        eps,
                    });
                    // Router's OWN input: rmsnorm_noscale(attn_out) · 1/√ne · ffn_gate_inp.scale —
                    // reads the UNNORMED post-attention residual `hidden`, NOT `hn` (neither FFN
                    // branch's normed input). `router_ones` is the weightless full-width RMSNorm
                    // (see its upload next to `v_ones`).
                    let ones = router_ones.expect("diffusion-gemma layer without router_ones");
                    g.push(Op::RmsNorm {
                        x: hidden,
                        weight: ones,
                        dst: router_tmp,
                        rows: batch as u32,
                        dim: ne as u32,
                        eps,
                    });
                    g.push(Op::Scale {
                        x: router_tmp,
                        dst: router_tmp,
                        s: 1.0 / (ne as f32).sqrt(),
                        n: (batch * ne) as u32,
                    });
                    g.push(Op::MulVec {
                        x: router_tmp,
                        vec: router_scale,
                        dst: router_tmp,
                        rows: batch as u32,
                        n: ne as u32,
                    });
                    // MoE branch input: pre_ffw_norm_2(attn_out) — also reads `hidden`, a THIRD
                    // independent normalization of the same residual.
                    g.push(Op::RmsNorm {
                        x: hidden,
                        weight: m_pre_norm,
                        dst: moe_in,
                        rows: batch as u32,
                        dim: ne as u32,
                        eps,
                    });
                    g.push(Op::MoeFfn {
                        x: moe_in,
                        router_x: router_tmp,
                        router,
                        gate_exps: gate_up_exps,
                        up_exps: gate_up_exps, // fused: same handle as gate_exps, never read
                        down_exps,
                        down_scale: Some(down_scale),
                        fused_gate_up: true,
                        dst: moe_out,
                        ne: ne as u32,
                        n_expert: mc.n_expert as u32,
                        n_used: mc.n_used as u32,
                        n_ff_exp: mc.n_ff_exp as u32,
                        scale: mc.scale,
                        act,
                    });
                    g.push(Op::RmsNorm {
                        x: moe_out,
                        weight: m_post_norm,
                        dst: moe_out,
                        rows: batch as u32,
                        dim: ne as u32,
                        eps,
                    });
                    // out = post_ffw_norm(dense + moe) + attn_out — the sum lands in `sub`; the
                    // shared `post_ffw_norm` (`lw.post_ffw`, generic below) and residual add are
                    // the SAME code every gemma layer already runs.
                    g.push(Op::Add {
                        a: d_out,
                        b: moe_out,
                        dst: sub,
                        n: (batch * ne) as u32,
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
                    w_off: 0,
                });
                // gelu(plg) * ipl[r, l*npl .. l*npl+npl] — gather each row's layer-l slice of the
                // [batch, n_layer*npl] input into a contiguous [batch, npl] scratch (one strided-
                // copy dispatch), then the plain gated activation. Keeps GatedAct's semantics
                // unchanged (its up_off has no per-row stride).
                let ipl_l = g.internal(f32d(batch * npl));
                g.push(Op::CopyStrided {
                    src: ipl,
                    src_off: (l * npl) as u32,
                    src_stride: (c.n_layer * npl) as u32,
                    dst: ipl_l,
                    dst_off: 0,
                    dst_stride: npl as u32,
                    rows: batch as u32,
                    n: npl as u32,
                });
                g.push(Op::GatedAct {
                    gate: plg,
                    up: ipl_l,
                    dst: plg,
                    rows: batch as u32,
                    nff: npl as u32,
                    act: Activation::Gelu,
                    up_off: 0,
                });
                g.push(Op::Linear {
                    x: plg,
                    weight: proj_w,
                    dst: plp,
                    m: batch as u32,
                    in_f: npl as u32,
                    out_f: ne as u32,
                    w_off: 0,
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
            // DiffusionGemma denoise reads the DECODER scalar (`layer_output_scale`) instead of
            // the encoder one baked into `out_scale` for every other diffusion-gemma phase (the
            // causal prompt prefill) — see docs/DIFFUSIONGEMMA.md.
            let layer_scale = if denoise {
                dec_out_scale[l]
            } else {
                out_scale[l]
            };
            if let Some(s) = layer_scale {
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
        // For batch > 1 with logits_rows == 1: the LM head runs only on the LAST token's
        // hidden state — extract it via Op::Copy before the projection so the logits output is
        // [vocab]. Speculative verify passes logits_rows == batch and runs the head over every
        // row instead (no Copy).
        let lm_in = if batch > 1 && logits_rows == 1 {
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
        // MTP Phase 1 (issue #33): `lm_in` IS the tap target — exactly the rows `logits` is about
        // to be computed from, one op earlier (the reference's `res->t_h_nextn`, captured right
        // after `output_norm` in `qwen35.cpp`). A plain Copy into a fresh Output, so this never
        // disturbs `lm_in`'s existing consumer (the `Op::Linear` below).
        let h_out = if h_tap {
            let ho = g.output(f32d(ne * logits_rows));
            g.push(Op::Copy {
                src: lm_in,
                src_off: 0,
                dst: ho,
                dst_off: 0,
                n: (ne * logits_rows) as u32,
            });
            Some(ho)
        } else {
            None
        };
        g.push(Op::Linear {
            x: lm_in,
            weight: w_lm,
            dst: logits,
            m: logits_rows as u32,
            in_f: ne as u32,
            out_f: c.vocab as u32,
            w_off: 0,
        });
        if c.final_softcap > 0.0 {
            g.push(Op::Softcap {
                x: logits,
                dst: logits,
                cap: c.final_softcap,
                n: (c.vocab * logits_rows) as u32,
            });
        }
        (
            g,
            DecodeHandles {
                hidden,
                positions,
                rope_freqs,
                pl_tok_in,
                sc_logits: sc_logits_in,
                sc_embt: sc_embt_id,
                temp_inv: temp_inv_id,
                logits,
                h_out,
                k_cache,
                v_cache,
                weights,
            },
        )
    };

    // ── Phase-2 DiffusionGemma canvas denoise (see `DenoiseReq`'s doc) ───────────────────────
    // ONE forward over the C canvas rows, reusing the session's already-prefilled prompt KV
    // (rows 0..P, P = `denoise_p`). Mirrors the VERIFY early-return below (batched multi-row
    // forward, LM head on every row) but with the canvas embedding/mask/decoder-scalar wiring.
    if let Some(req) = denoise_req {
        if !c.diffusion_gemma {
            return Err(anyhow!(
                "canvas denoise forward: diffusion-gemma models only"
            ));
        }
        // Phase-A/B perf: per-step timing, gated on INFR_DIFFUSION_TIME=1 (stderr, one line/step).
        // Phase A found `sc` was ~85% of every step (the host SC matvec); Phase B moved that
        // in-graph on Vulkan, so `sc` now reports only the (cheap) host prep — embed gather, and
        // the temp_inv premultiply on the gpu_sc path — while `exec` absorbs the SC math itself.
        let time_diffusion = std::env::var("INFR_DIFFUSION_TIME").is_ok();
        let canvas = req.canvas_tokens;
        let cc = canvas.len();
        let p = denoise_p;
        if p + cc > max_ctx {
            return Err(anyhow!(
                "denoise: prompt {p} + canvas {cc} exceeds the session KV capacity {max_ctx}"
            ));
        }
        // Phase-B perf: in-graph self-conditioning on Vulkan (see docs/DIFFUSIONGEMMA.md's
        // Phase-B and the reference's `dg_canvas_embed`) — Phase D widened this to Metal too:
        // `Op::Softmax`'s wide kernel handles the [C, vocab] shape unmodified (a plain grid-stride
        // loop, no row/dim limit) and `sc_embT`'s `DType::F16` weight already flows through
        // Metal's ordinary non-quant `Op::Linear` path (`weight_buf` dequant-caches it to f32 —
        // functionally correct, just not the dedicated native-f16 GEMV a quant weight would get;
        // see `weight_buf`'s VRAM-budget guard for the failure mode if it doesn't fit). CPU alone
        // keeps the Phase-A host path (`diffusion_self_cond` + host weightless norm) below.
        let gpu_sc = matches!(be.name(), "vulkan" | "metal");
        let sc_on = req.sc_logits.is_some();
        // The plan shape only varies with SC on the gpu_sc path (CPU's graph never changes;
        // `sc_on` there is purely a host-side input difference) — see `DenoiseCache::sc`'s doc.
        let plan_sc = gpu_sc && sc_on;
        // Perf (Vulkan only — docs/DIFFUSIONGEMMA.md's Phase-B "sc round-trip" elimination): a
        // session-persistent ping-pong pair of GPU buffers (`SeamKv::sc_ping`) stands in for
        // `DenoiseCache`'s per-plan `logits_buf`/`sc_logits_buf`. The previous call's LM-head
        // output is ALREADY resident in one of the pair (it's the very buffer that call
        // downloaded from), so this call's self-conditioning softmax reads it directly instead of
        // the host premultiplying and reuploading the whole `[cc, vocab]` array — see `sc_ping`'s
        // doc and `dyn_sc_scale`'s doc on `build`. Metal keeps the original host-premultiply path
        // (unverified hardware, out of scope here); CPU never reaches `plan_sc` at all.
        let use_ping = be.name() == "vulkan";
        let dyn_sc = plan_sc && use_ping;
        if use_ping && sc_ping.is_none() {
            let bytes = cc * c.vocab * 4;
            *sc_ping = Some([
                be.alloc(bytes, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
                be.alloc(bytes, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
            ]);
            *sc_temp_inv_buf = Some(
                be.alloc(4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
            );
            if plan_sc {
                // Defensive seed: this is the FIRST denoise call ever on this slot and it ALREADY
                // wants self-conditioning, so the ping slot we're about to read from is freshly
                // zero-initialized, not real data. Every actual caller (`diffusion_generate`'s
                // `denoise_block`) resets `sc_logits` to `None` at the start of every block, so
                // production traffic never takes this branch past the very first call ever — seed
                // it once from the host slice the caller gave us so a hypothetical caller that
                // DOESN'T reset stays correct too.
                let read_idx = 1 - *sc_ping_write;
                be.upload(
                    sc_ping.as_ref().expect("just allocated")[read_idx].as_ref(),
                    bytemuck::cast_slice(req.sc_logits.expect("plan_sc implies sc_on")),
                )
                .map_err(|e| anyhow!("{e}"))?;
            }
        }

        let t_sc0 = std::time::Instant::now();
        // 1. Canvas embedding: e = embed(tok)·√n_embd. diffusion-gemma is always gemma-family, so
        // `embed_scale` is always √n_embd — computed locally (the "── drive ──" section below
        // defines its own copy for the ordinary decode loop, unreached by this early return). On
        // the gpu_sc path this is ALL of `hidden_host` — the SC add + weightless norm run IN-GRAPH
        // instead (see `build`'s SC subgraph); on CPU it's completed below exactly as Phase A.
        let embed_scale = if gemma { (ne as f32).sqrt() } else { 1.0 };
        let mut hidden_host: Vec<f32> = Vec::with_capacity(cc * ne);
        for &tok in canvas {
            let base = tok as usize * ne;
            hidden_host.extend(token_embd[base..base + ne].iter().map(|&x| x * embed_scale));
        }
        // Per-step host-premultiplied previous canvas logits for the Metal `gpu_sc` path — `Some`
        // only when `plan_sc && !dyn_sc` (populated below); declared here so it outlives the
        // upload further down. Vulkan's `dyn_sc` path never populates this — see `use_ping`'s doc.
        let mut sc_logits_host: Option<Vec<f32>> = None;
        if let Some(sc_logits) = req.sc_logits {
            if sc_logits.len() != cc * c.vocab {
                return Err(anyhow!(
                    "denoise: sc_logits length {} != {cc}*{} (canvas rows * vocab)",
                    sc_logits.len(),
                    c.vocab
                ));
            }
            if gpu_sc {
                if !dyn_sc {
                    // Premultiply by temp_inv on the HOST (one pass over cc*vocab floats,
                    // threaded) so the in-graph `Op::Softmax`'s `scale` stays a CONSTANT 1.0
                    // across steps whose temp_inv legitimately changes — see `sc_logits_in`'s doc
                    // in `build`. Metal only now: Vulkan's `dyn_sc` path skips this entirely (the
                    // previous step's RAW logits are already GPU-resident in `sc_ping` and the
                    // scale rides `sc_temp_inv_buf` instead — see the bind section below).
                    use rayon::prelude::*;
                    let mut scaled = vec![0f32; sc_logits.len()];
                    scaled
                        .par_iter_mut()
                        .zip(sc_logits.par_iter())
                        .for_each(|(d, &s)| *d = s * req.temp_inv);
                    sc_logits_host = Some(scaled);
                }
            } else {
                // Phase-A host path (CPU only now), unchanged.
                // Phase-A perf: dequantize the self-cond MLP weights ONCE per session, not once
                // per call — `diffusion_self_cond` used to re-run four `load_tensor_dequant`s
                // every step.
                if self_cond_w.is_none() {
                    let (pre_norm, _) = crate::load_tensor_dequant(g, "self_cond_pre_norm.weight")?;
                    let (gate_w, _) = crate::load_tensor_dequant(g, "self_cond_gate.weight")?; // [nff, ne]
                    let (up_w, _) = crate::load_tensor_dequant(g, "self_cond_up.weight")?; // [nff, ne]
                    let (down_w, _) = crate::load_tensor_dequant(g, "self_cond_down.weight")?; // [ne, nff]
                                                                                               // One-time f16 conversion of the embedding table (see `SelfCondWeights::emb16`).
                    let mut emb16 = vec![0u16; token_embd.len()];
                    {
                        use rayon::prelude::*;
                        emb16
                            .par_chunks_mut(1 << 16)
                            .zip(token_embd.par_chunks(1 << 16))
                            .for_each(|(dst, src)| {
                                for (d, &v) in dst.iter_mut().zip(src) {
                                    *d = half::f16::from_f32(v).to_bits();
                                }
                            });
                    }
                    *self_cond_w = Some(std::sync::Arc::new(SelfCondWeights {
                        pre_norm,
                        gate_w,
                        up_w,
                        down_w,
                        emb16,
                    }));
                }
                let scw = self_cond_w.as_ref().expect("just populated above");
                let sc_sig = diffusion_self_cond(scw, c, sc_logits, req.temp_inv, cc)?;
                for (h, s) in hidden_host.iter_mut().zip(sc_sig.iter()) {
                    *h += s;
                }
            }
        }
        if !gpu_sc {
            // Phase-A host weightless canvas-embed norm (CPU only now — the gpu_sc path applies
            // this IN-GRAPH for both the sc-on and no-sc plans, see `build`).
            for row in hidden_host.chunks_mut(ne) {
                let ms: f32 = row.iter().map(|&x| x * x).sum::<f32>() / ne as f32;
                let inv = 1.0 / (ms + c.rms_eps).sqrt();
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
        }
        let sc_secs = t_sc0.elapsed().as_secs_f64();
        let dn_positions: Vec<i32> = (p as i32..(p + cc) as i32).collect();

        // Phase-A/B perf: cache the compiled plan + its staging buffers across denoise() calls,
        // keyed by (cc, p, sc) — see `DenoiseCache`'s doc. A hit skips `build`+`compile`+N `alloc`s
        // entirely; a miss (first call, a block boundary, a resized canvas, or an SC on/off
        // transition on the gpu_sc path) rebuilds once and the NEXT call on this key hits.
        let t_build0 = std::time::Instant::now();
        let stale = match denoise_cache {
            Some(dcache) => dcache.cc != cc || dcache.p != p || dcache.sc != plan_sc,
            None => true,
        };
        if stale {
            // 2/3/4. Per-layer forward: the decoder-scalar / Canvas-mask denoise variant of
            // `build`; 5. logits over ALL C rows (logits_rows = cc).
            let (dg, dh) = build(
                cc,
                p,
                cc,
                true,
                if gpu_sc { Some(plan_sc) } else { None },
                dyn_sc,
                false, // MTP h-tap: diffusion-gemma denoise never taps
            );
            let plan = be.compile(&dg).map_err(|e| anyhow!("{e}"))?;
            let hidden_buf = be
                .alloc(cc * ne * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            let pos_buf = be
                .alloc(cc * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            // Vulkan: the output lives in the session-level `sc_ping` pair instead (see its doc)
            // — no per-plan logits_buf to allocate. Metal/CPU keep the original per-plan buffer.
            let logits_buf = if use_ping {
                None
            } else {
                Some(
                    be.alloc(cc * c.vocab * 4, BufferUsage::Staging)
                        .map_err(|e| anyhow!("{e}"))?,
                )
            };
            let sc_logits_buf = if plan_sc && !dyn_sc {
                Some(
                    be.alloc(cc * c.vocab * 4, BufferUsage::Staging)
                        .map_err(|e| anyhow!("{e}"))?,
                )
            } else {
                None
            };
            *denoise_cache = Some(DenoiseCache {
                cc,
                p,
                sc: plan_sc,
                plan,
                dh,
                hidden_buf,
                pos_buf,
                logits_buf,
                sc_logits_buf,
            });
        }
        let build_secs = t_build0.elapsed().as_secs_f64();

        // Phase-B perf: ensure the one-time SC soft-embedding weight (Vulkan + SC only) — lazy,
        // built ONCE per session (shared across forked slots — see `SeamKv::sc_embt`) from the
        // already-dequantized `token_embd`.
        if plan_sc && sc_embt.is_none() {
            let t_embt0 = std::time::Instant::now();
            *sc_embt = Some(build_sc_embt(be, token_embd, ne, c.vocab)?);
            eprintln!(
                "[diffusion denoise] built the SC soft-embedding weight ({:.0} MB) in {:.2}s",
                (ne * c.vocab * 2) as f64 / 1e6,
                t_embt0.elapsed().as_secs_f64()
            );
        }

        let dcache = denoise_cache.as_ref().expect("just ensured present above");

        be.upload(
            dcache.hidden_buf.as_ref(),
            bytemuck::cast_slice(&hidden_host),
        )
        .map_err(|e| anyhow!("{e}"))?;
        be.upload(dcache.pos_buf.as_ref(), bytemuck::cast_slice(&dn_positions))
            .map_err(|e| anyhow!("{e}"))?;
        let mut db = Bindings::new();
        db.bind(dcache.dh.hidden, dcache.hidden_buf.as_ref());
        db.bind(dcache.dh.positions, dcache.pos_buf.as_ref());
        if let (Some(rid), Some((rb, _))) = (dcache.dh.rope_freqs, &rf_buf) {
            db.bind(rid, rb.as_ref());
        }
        for l in 0..c.n_layer {
            db.bind(dcache.dh.k_cache[l], kbufs[l].as_ref());
            db.bind(dcache.dh.v_cache[l], vbufs[l].as_ref());
        }
        for (i, wid) in dcache.dh.weights.iter().enumerate() {
            db.bind(*wid, wbufs[i].as_ref());
        }
        if plan_sc {
            if dyn_sc {
                // Vulkan perf: the SC input is the OTHER ping slot (this call's write target is
                // the opposite one — see `sc_ping`'s doc) — already GPU-resident, no upload. Only
                // the 4-byte temp_inv scalar moves host->device this step.
                let ping = sc_ping.as_ref().expect("allocated above");
                let read_idx = 1 - *sc_ping_write;
                let temp_inv_buf = sc_temp_inv_buf.as_ref().expect("allocated above");
                be.upload(temp_inv_buf.as_ref(), &req.temp_inv.to_le_bytes())
                    .map_err(|e| anyhow!("{e}"))?;
                db.bind(
                    dcache
                        .dh
                        .sc_logits
                        .expect("plan_sc plan declares sc_logits"),
                    ping[read_idx].as_ref(),
                );
                db.bind(
                    dcache.dh.temp_inv.expect("dyn_sc plan declares temp_inv"),
                    temp_inv_buf.as_ref(),
                );
            } else {
                // Metal: original host-premultiply-and-upload path, unchanged.
                let sc_logits_host = sc_logits_host
                    .as_ref()
                    .expect("plan_sc && !dyn_sc implies sc_logits_host is Some");
                let sc_logits_buf = dcache
                    .sc_logits_buf
                    .as_ref()
                    .expect("plan_sc && !dyn_sc plan always allocates sc_logits_buf");
                be.upload(sc_logits_buf.as_ref(), bytemuck::cast_slice(sc_logits_host))
                    .map_err(|e| anyhow!("{e}"))?;
                db.bind(
                    dcache
                        .dh
                        .sc_logits
                        .expect("plan_sc plan declares sc_logits"),
                    sc_logits_buf.as_ref(),
                );
            }
            db.bind(
                dcache.dh.sc_embt.expect("plan_sc plan declares sc_embt"),
                sc_embt.as_ref().expect("ensured present above").as_ref(),
            );
        }
        // Vulkan: the ping slot THIS call writes into — the opposite of the one just bound above
        // as SC input (when `plan_sc`), or simply the current write slot on a non-SC step. Metal/
        // CPU keep the per-plan `logits_buf`.
        let logits_out_buf: &dyn Buffer = if use_ping {
            sc_ping.as_ref().expect("allocated above")[*sc_ping_write].as_ref()
        } else {
            dcache
                .logits_buf
                .as_ref()
                .expect("non-ping path always allocates logits_buf")
                .as_ref()
        };
        db.bind(dcache.dh.logits, logits_out_buf);
        let t_exec0 = std::time::Instant::now();
        be.execute(dcache.plan.as_ref(), &db)
            .map_err(|e| anyhow!("{e}"))?;
        let exec_secs = t_exec0.elapsed().as_secs_f64();
        req.out_logits.resize(cc * c.vocab, 0.0);
        let t_dl0 = std::time::Instant::now();
        be.download(logits_out_buf, bytemuck::cast_slice_mut(req.out_logits))
            .map_err(|e| anyhow!("{e}"))?;
        let dl_secs = t_dl0.elapsed().as_secs_f64();
        // Vulkan: flip which ping slot is "write" vs "read" for the NEXT call — this call's output
        // (just downloaded above) becomes the next call's self-conditioning input, already
        // GPU-resident (see `sc_ping`'s doc).
        if use_ping {
            *sc_ping_write = 1 - *sc_ping_write;
        }
        if time_diffusion {
            eprintln!(
                "[diffusion denoise] sc={sc_secs:.3}s build={build_secs:.3}s exec={exec_secs:.3}s dl={dl_secs:.3}s total={:.3}s",
                sc_secs + build_secs + exec_secs + dl_secs,
            );
        }
        // `cached`/`start` were left untouched above (the canvas isn't part of the prompt/gen
        // token stream) — the prompt-KV rows 0..P stay exactly as the prior prefill call left
        // them, so the NEXT denoise call (same or different canvas) re-overwrites rows P..P+C
        // again, and a later real prefill still resumes from P.
        return Ok((
            Vec::new(),
            GenStats {
                n_prompt: 0,
                prompt_secs: sc_secs + build_secs + exec_secs + dl_secs,
                n_gen: 0,
                decode_secs: 0.0,
            },
        ));
    }

    // ── speculative VERIFY ──────────────────────────────────────────────────────────
    // One batched forward over the un-cached suffix with the LM head on EVERY row: returns
    // [m, vocab] logits (the distribution after each suffix token) and generates nothing.
    // The suffix-prefill contract doubles as the accept/rollback mechanism: the caller
    // truncates its committed token list and the next call's prefix diff overwrites the
    // stale KV rows. Dense non-E2B models only (mirrors the batched-prefill guard).
    if let Some(out_logits) = verify {
        if c.moe.is_some() || ple.is_some() {
            return Err(anyhow!("speculative verify: dense non-E2B models only"));
        }
        let vf_scale = if gemma { (ne as f32).sqrt() } else { 1.0 };
        let m = prompt.len() - start;
        let mut vf_hidden: Vec<f32> = Vec::with_capacity(m * ne);
        for &tok in &prompt[start..] {
            let base = tok as usize * ne;
            vf_hidden.extend(token_embd[base..base + ne].iter().map(|&x| x * vf_scale));
        }
        let vf_positions: Vec<i32> = (start as i32..(start + m) as i32).collect();
        let vf_hidden_buf = be
            .alloc(m * ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let vf_pos_buf = be
            .alloc(m * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let vf_logits_buf = be
            .alloc(m * c.vocab * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(vf_hidden_buf.as_ref(), bytemuck::cast_slice(&vf_hidden))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(vf_pos_buf.as_ref(), bytemuck::cast_slice(&vf_positions))
            .map_err(|e| anyhow!("{e}"))?;
        // MTP Phase 1 (issue #33): VERIFY already runs the LM head on every one of the `m` rows —
        // exactly the rows the MTP catch-up driver needs `h` for (docs/MTP.md's `process()`).
        // `h_tap` piggybacks on the SAME graph/execute, just an extra Output + download.
        let want_h = h_out.is_some();
        // Phase-4 MTP profiling (issue #33, INFR_MTP_TIME=1): split VERIFY's own wall time into
        // graph-build / plan-compile / execute / download, and report `m` (the rows actually
        // reprocessed) + whether this call is a FULL reprefill (`start == 0` with a nonempty
        // history behind it, i.e. the qwen35 no-rewind fallback fired) vs the cheap incremental
        // suffix-only path. This is the number the MTP perf pass profiles before touching any
        // code — see mtp.rs's `generate_mtp_spec_vulkan_timed` doc on the no-rewind cost.
        let time_verify = std::env::var("INFR_MTP_TIME").is_ok();
        let full_reprefill = start == 0 && m > 1;
        let t_vbuild0 = std::time::Instant::now();
        let (vg, vh) = build(m, start, m, false, None, false, want_h);
        let vbuild_secs = t_vbuild0.elapsed().as_secs_f64();
        let t_vcompile0 = std::time::Instant::now();
        let vplan = be.compile(&vg).map_err(|e| anyhow!("{e}"))?;
        let vcompile_secs = t_vcompile0.elapsed().as_secs_f64();
        let vf_h_buf = if want_h {
            Some(
                be.alloc(m * ne * 4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            None
        };
        let mut vb = Bindings::new();
        vb.bind(vh.hidden, vf_hidden_buf.as_ref());
        vb.bind(vh.positions, vf_pos_buf.as_ref());
        if let (Some(rid), Some((rb, _))) = (vh.rope_freqs, &rf_buf) {
            vb.bind(rid, rb.as_ref());
        }
        for l in 0..c.n_layer {
            vb.bind(vh.k_cache[l], kbufs[l].as_ref());
            vb.bind(vh.v_cache[l], vbufs[l].as_ref());
        }
        for (i, wid) in vh.weights.iter().enumerate() {
            vb.bind(*wid, wbufs[i].as_ref());
        }
        vb.bind(vh.logits, vf_logits_buf.as_ref());
        if let (Some(ho), Some(hb)) = (vh.h_out, &vf_h_buf) {
            vb.bind(ho, hb.as_ref());
        }
        let t0 = std::time::Instant::now();
        be.execute(vplan.as_ref(), &vb)
            .map_err(|e| anyhow!("{e}"))?;
        let vexec_secs = t0.elapsed().as_secs_f64();
        let t_vdl0 = std::time::Instant::now();
        out_logits.resize(m * c.vocab, 0.0);
        be.download(vf_logits_buf.as_ref(), bytemuck::cast_slice_mut(out_logits))
            .map_err(|e| anyhow!("{e}"))?;
        if let (Some(out), Some(hb)) = (h_out.take(), &vf_h_buf) {
            out.resize(m * ne, 0.0);
            be.download(hb.as_ref(), bytemuck::cast_slice_mut(out))
                .map_err(|e| anyhow!("{e}"))?;
        }
        let vdl_secs = t_vdl0.elapsed().as_secs_f64();
        if time_verify {
            eprintln!(
                "[mtp verify] m={m} start={start} full_reprefill={full_reprefill} \
                 build={:.1}ms compile={:.1}ms exec={:.1}ms dl={:.1}ms total={:.1}ms",
                vbuild_secs * 1e3,
                vcompile_secs * 1e3,
                vexec_secs * 1e3,
                vdl_secs * 1e3,
                (vbuild_secs + vcompile_secs + vexec_secs + vdl_secs) * 1e3,
            );
        }
        cached.extend_from_slice(&prompt[start..]);
        return Ok((
            Vec::new(),
            GenStats {
                n_prompt: m,
                prompt_secs: t0.elapsed().as_secs_f64(),
                n_gen: 0,
                decode_secs: 0.0,
            },
        ));
    }

    // ── drive ───────────────────────────────────────────────────────────────────────
    // Sampling: greedy unless INFR_TEMP is set (the CLI sets chat defaults for run/serve; the
    // golden/parity tests pin INFR_TEMP=0 or leave it unset).
    let sampler = crate::sampling::Sampler::from_env();
    let mut rng = crate::sampling::seed_rng();
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
    // Batched MoE prefill needs the adapter's GPU-routed expert path: Q4_K gate/up (split, what
    // qwen3moe ships) or fused Q4_K gate_up (diffusion-gemma's `ffn_gate_up_exps`) +
    // Q4_K/Q6_K/Q8_0/Q5_0 down (Q5_0 is what the shipped diffusiongemma-26B-A4B-it-GGUF actually
    // uses); other stacked formats keep the per-token loop.
    let moe_batched_ok = c.moe.is_some() && {
        let dt = |n: String| g.tensors().iter().find(|t| t.name == n).map(|t| t.dtype);
        if c.diffusion_gemma {
            (0..c.n_layer).all(|l| {
                dt(format!("blk.{l}.ffn_gate_up_exps.weight")) == Some(DType::Q4K)
                    && matches!(
                        dt(format!("blk.{l}.ffn_down_exps.weight")),
                        Some(DType::Q4K) | Some(DType::Q6K) | Some(DType::Q8_0) | Some(DType::Q5_0)
                    )
            })
        } else {
            (0..c.n_layer).all(|l| {
                dt(format!("blk.{l}.ffn_gate_exps.weight")) == Some(DType::Q4K)
                    && dt(format!("blk.{l}.ffn_up_exps.weight")) == Some(DType::Q4K)
                    && matches!(
                        dt(format!("blk.{l}.ffn_down_exps.weight")),
                        Some(DType::Q4K) | Some(DType::Q6K)
                    )
            })
        }
    };
    let decode_start = if prompt.len() - start > 2 && (c.moe.is_none() || moe_batched_ok) {
        // Batch-prefill the un-cached suffix, all but the last prompt token (positions
        // start..plen-1; rows 0..start are reused from the session cache) — in UBATCH CHUNKS.
        // One giant graph would scale the internal activation/attention scratch with the whole
        // prompt (an 8B p8000 prefill built a multi-second single submission whose tail work
        // tripped the amdgpu ring watchdog → device lost) and bakes a multi-second unpreemptible
        // submit; fixed-size chunks bound both, exactly like the bespoke path's ubatches.
        let ubatch: usize = std::env::var("INFR_UBATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1024);
        let pf_end = prompt.len() - 1;
        let mut cstart = start;
        while cstart < pf_end {
            let cend = (cstart + ubatch).min(pf_end);
            let pf_m = cend - cstart;
            let mut pf_hidden: Vec<f32> = Vec::with_capacity(pf_m * ne);
            for &tok in &prompt[cstart..cend] {
                let base = tok as usize * ne;
                pf_hidden.extend(token_embd[base..base + ne].iter().map(|&x| x * embed_scale));
            }
            // Absolute positions [cstart, ..., cend-1].
            let pf_positions: Vec<i32> = (cstart as i32..cend as i32).collect();
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

            // gemma4 E2B: the chunk's per-layer TOKEN embedding rows (gather+dequant only — the
            // model_proj GEMV/RMSNorm/combine run as GPU graph ops in the `build` prologue).
            let pf_ipl_buf = if let Some(ple) = ple {
                let ipl = e2b_ipl_rows(g, ple, &prompt[cstart..cend])?;
                let b = be
                    .alloc(ipl.len() * 4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?;
                be.upload(b.as_ref(), bytemuck::cast_slice(&ipl))
                    .map_err(|e| anyhow!("{e}"))?;
                Some(b)
            } else {
                None
            };
            let pf_t0 = std::time::Instant::now();
            // MTP h-tap gap (Phase 2 TODO, docs/MTP.md): the chunked BATCHED-PREFILL path never
            // taps `h` — it only ever runs `logits_rows == 1` (the last row of each chunk, which
            // this phase never samples/reads), so there's no per-row hidden state worth exposing
            // here yet. The MTP catch-up driver needs `h` for EVERY prefill row (not just chunk
            // tails); wiring that requires this path to also carry `logits_rows == pf_m` on
            // demand, which Phase 2 will add alongside the actual head forward.
            let (pf_g, pf_h) = build(pf_m, cstart, 1, false, None, false, false);
            let t_build = pf_t0.elapsed();
            let pf_plan = be.compile(&pf_g).map_err(|e| anyhow!("{e}"))?;
            let t_compile = pf_t0.elapsed();
            let mut pf_b = Bindings::new();
            pf_b.bind(pf_h.hidden, pf_hidden_buf.as_ref());
            pf_b.bind(pf_h.positions, pf_pos_buf.as_ref());
            // gemma4's proportional-RoPE divisors are a graph input too — bind them (the per-token
            // decode loop below does the same). Without this the batched graph has an unbound
            // `rope_freqs` Input and panics.
            if let (Some(rid), Some((rb, _))) = (pf_h.rope_freqs, &rf_buf) {
                pf_b.bind(rid, rb.as_ref());
            }
            if let (Some(pid), Some(ib)) = (pf_h.pl_tok_in, &pf_ipl_buf) {
                pf_b.bind(pid, ib.as_ref());
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
            // INFR_PROF_PF: split the per-chunk prefill wall time into host graph build, plan
            // compile, and execute (record + submit + GPU) — where a small-batch chunk's fixed
            // cost lives decides whether to attack recording or kernels.
            if std::env::var("INFR_PROF_PF").is_ok() {
                eprintln!(
                    "[pf prof] m={pf_m} build={:.1}ms compile={:.1}ms execute={:.1}ms",
                    t_build.as_secs_f64() * 1e3,
                    (t_compile - t_build).as_secs_f64() * 1e3,
                    (pf_t0.elapsed() - t_compile).as_secs_f64() * 1e3,
                );
            }
            prompt_t += pf_t0.elapsed();
            cstart = cend;
        }

        // KV rows are now filled through position plen-2; the last prompt token is handled by
        // the decode loop below (writes its KV, produces the logits the first sample uses).
        pf_end
    } else {
        start // fall through to per-token loop for MoE / E2B / short suffixes
    };

    // Record-once decode: for an eligible decode on a backend that supports replay (the Vulkan
    // seam), build+compile+bind ONE plan here and reuse it across the whole decode loop. The
    // adapter records the graph once and replays it per token, reading `pos` from the bound
    // positions buffer + a params SSBO — so the baked pos=0 here is irrelevant, and the per-token
    // host cost drops to just the emb/pos (+ E2B ipl) uploads. The gate mirrors the adapter's
    // graph eligibility: every dense arch replays — qk-norm (qwen3), the gemma family (SWA
    // windows + scale via push constants, freq_factors via qk_norm_rope_dyn_ff, V-norm/Softcap/
    // Scale are pos-independent), llama (f16-out interleaved Rope via rope_f16_dyn), MoE. Backends
    // without `decode_replay` (CPU interpreter, which reads the baked `pos`) and every ineligible
    // model keep rebuilding + recompiling per token below.
    // INFR_SEAM_NO_REPLAY=1 forces per-token rebuild (the adapter's static path) — slower, but
    // INFR_PROF2 per-op GPU timestamps work there (the replay path can't report them).
    // This gate MUST stay a strict subset of the adapter's `decode_eligible` — the plan below
    // bakes pos=0/kv_len=1, which is only correct when the adapter replays it (dyn kernels read
    // the live pos/kv_len); an ineligible graph would silently run the static path with the baked
    // values. Hence the per-layer head-dim mirror of the adapter's Attention check.
    // llama (no qk-norm) replays too — its f16-out Rope has a dyn kernel — but only without
    // freq_factors (the standalone Rope kernel has no ff binding; gemma4's ff rides QkNormRope).
    let dyn_replay = be.capabilities().decode_replay
        && std::env::var("INFR_SEAM_NO_REPLAY").is_err()
        && (qk_norm || rope_freqs.is_none())
        // Any quantized KV cache forces the per-execute STATIC decode (see the adapter's
        // `decode_eligible`: Q8's un-fused K-write mis-decodes under record-once replay; the low-bit
        // block quants use a dequant→f16 prepass with a standalone quantizing WriteKv). Must mirror
        // that rejection so this gate stays a strict subset — else the loop bakes pos=0 for a static
        // run. (turbo forces static too but is CPU-only, where decode_replay is off anyway.)
        && !kv_forces_static(k_fmt)
        && !kv_forces_static(v_fmt)
        && (0..c.n_layer)
            .all(|l| c.layer_head_dim(l).is_multiple_of(4) && c.layer_head_dim(l) <= 512)
        // qwen35's gated-DeltaNet layers (`Op::Conv1dSilu`/`Op::DeltaNet`) were never exercised
        // under record-once replay (the old seam always rebuilds per token) and the adapter's
        // replay tape isn't proven to re-read their `state` bindings correctly per replay —
        // measured to silently diverge from the static per-token rebuild after a few decode steps.
        // Force the static path for qwen35 until that's audited; every other arch is unaffected.
        && !c.qwen35
        // MTP h-tap (Phase 1, issue #33): the replay tape binds a FIXED set of tensors once: an
        // h-tap request changes the graph shape (an extra Output + Copy) per-call based on
        // whether THIS position is the one being sampled, which the static replay tape can't
        // express. Force the ordinary per-token rebuild path below when a caller wants the tap —
        // slower, but this is a validation-only hook (see `h_out`'s doc), never a hot path.
        && h_out.is_none();
    let ro = if dyn_replay {
        let (g, h) = build(1, 0, 1, false, None, false, false);
        let plan = be.compile(&g).map_err(|e| anyhow!("{e}"))?;
        let mut b = Bindings::new();
        b.bind(h.hidden, hidden_buf.as_ref());
        b.bind(h.positions, pos_buf.as_ref());
        if let (Some(rid), Some((rb, _))) = (h.rope_freqs, &rf_buf) {
            b.bind(rid, rb.as_ref());
        }
        if let (Some(pid), Some(ib)) = (h.pl_tok_in, &ipl_buf) {
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
        Some((plan, b))
    } else {
        None
    };

    // INFR_IGNORE_EOS=1 (benchmarks): decode the full requested count — a model that emits EOS
    // instantly on a dummy context (gemma at depth) otherwise "finishes" 64 tokens in one step
    // and the reported tok/s is fiction. llama-bench ignores EOS the same way.
    let ignore_eos = std::env::var("INFR_IGNORE_EOS").is_ok();
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

        // gemma4 E2B: this token's per-layer TOKEN embedding row (gather+dequant only — the
        // model_proj GEMV/RMSNorm/combine run as GPU graph ops in the `build` prologue).
        if let (Some(ple), Some(ipl_buf)) = (ple, &ipl_buf) {
            let ipl = e2b_ipl_rows(g, ple, &[tok as u32])?;
            be.upload(ipl_buf.as_ref(), bytemuck::cast_slice(&ipl))
                .map_err(|e| anyhow!("{e}"))?;
        }

        // Only sample once we're past the prompt (decode position = last prompt token onward).
        let is_decode = pos + 1 >= prompt.len();
        // Sample only at the FRONTIER (this position's token is the newest one fed). A constrained
        // step can emit several deterministically-forced tokens at once — they're queued onto
        // `cur` and the following iterations just feed them (no sampling) until the frontier.
        let at_frontier = pos + 1 == cur.len();
        // MTP Phase 1 h-tap (issue #33): only the frontier row is ever sampled/downloaded as
        // logits — same row `h_out` (when requested) captures. `dyn_replay` already excludes
        // `h_out.is_some()` (see its doc), so this is always the rebuild (`else`) branch below.
        let want_h = h_out.is_some() && is_decode && at_frontier;
        let h_tap_buf = if want_h {
            Some(
                be.alloc(ne * 4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            None
        };
        let t_setup = std::time::Instant::now();
        let (setup_el, exec_el);
        if let Some((plan, b)) = &ro {
            // Record-once path: reuse the single compiled plan + bindings (no per-token rebuild).
            setup_el = t_setup.elapsed();
            let t_exec = std::time::Instant::now();
            be.execute(plan.as_ref(), b).map_err(|e| anyhow!("{e}"))?;
            exec_el = t_exec.elapsed();
        } else {
            let (g, h) = build(1, pos, 1, false, None, false, want_h);
            let plan = be.compile(&g).map_err(|e| anyhow!("{e}"))?;
            let mut b = Bindings::new();
            b.bind(h.hidden, hidden_buf.as_ref());
            b.bind(h.positions, pos_buf.as_ref());
            if let (Some(rid), Some((rb, _))) = (h.rope_freqs, &rf_buf) {
                b.bind(rid, rb.as_ref());
            }
            if let (Some(pid), Some(ib)) = (h.pl_tok_in, &ipl_buf) {
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
            if let (Some(ho), Some(hb)) = (h.h_out, &h_tap_buf) {
                b.bind(ho, hb.as_ref());
            }
            setup_el = t_setup.elapsed();
            let t_exec = std::time::Instant::now();
            be.execute(plan.as_ref(), &b).map_err(|e| anyhow!("{e}"))?;
            exec_el = t_exec.elapsed();
        }
        if std::env::var("INFR_PROF_DEC").is_ok() && pos + 1 >= prompt.len() {
            dec_setup += setup_el;
            dec_exec += exec_el;
        }

        if is_decode && at_frontier {
            be.download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
                .map_err(|e| anyhow!("{e}"))?;
            // Phase-1 DiffusionGemma validation hook (see the param doc): this is the FIRST
            // is_decode row — the causal prefill's last-token logits — captured before sampling
            // touches `logits` (grammar-constrained steps overwrite it in place below).
            if let Some(out) = logits_out.take() {
                *out = logits.clone();
            }
            // MTP Phase 1 h-tap (see `want_h` above): same row, one op earlier than `logits`.
            if let (Some(out), Some(hb)) = (h_out.take(), &h_tap_buf) {
                let mut hrow = vec![0f32; ne];
                be.download(hb.as_ref(), bytemuck::cast_slice_mut(&mut hrow))
                    .map_err(|e| anyhow!("{e}"))?;
                *out = hrow;
            }
            if let Some(cst) = constraint.as_deref_mut() {
                // Grammar-forced span (serve's tool_choice "required"/named): the shared
                // llguidance step. Empty step ⇒ the constrained span ended.
                let (step, done) = crate::grammar::constrained_step(cst, &mut logits, &c.eos_ids)
                    .map_err(|e| anyhow!("{e}"))?;
                decode_t += step_t0.elapsed();
                if step.is_empty() {
                    break;
                }
                for &t in &step {
                    out.push(t);
                    on_token(t);
                    cur.push(t);
                    decode_n += 1;
                }
                if done || out.len() >= max_new {
                    break;
                }
            } else {
                let next = crate::sampling::sample_logits(&logits, sampler, &mut rng);
                let is_eos = !ignore_eos && (c.eos_ids.contains(&next) || next == c.eos);
                out.push(next);
                decode_t += step_t0.elapsed();
                decode_n += 1;
                if !is_eos {
                    on_token(next); // stream the token (EOS is not emitted)
                }
                if is_eos || out.len() >= max_new {
                    break;
                }
                cur.push(next);
            }
        } else if is_decode {
            // feeding a queued forced token — its KV write is the whole point of this step
            decode_t += step_t0.elapsed();
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
    // Record what the KV cache now holds for the next turn's prefix diff. `out` includes any
    // sampled EOS (its KV row was written before the loop broke)... it was PUSHED to out before
    // the break, and its KV is written only when fed back — the EOS is never fed, so the cache
    // holds prompt + generated-minus-last-fed. Conservative: cache prompt + all fed tokens; the
    // final sampled token's row is NOT materialized, so exclude it.
    *cached = prompt.to_vec();
    if !out.is_empty() {
        cached.extend_from_slice(&out[..out.len() - 1]);
    }
    let stats = GenStats {
        // The tokens actually PREFILLED this call (the un-cached suffix) — the TTFT-honest count.
        n_prompt: prompt.len() - start,
        prompt_secs: prompt_t.as_secs_f64(),
        n_gen: decode_n,
        decode_secs: decode_t.as_secs_f64(),
    };
    Ok((out, stats))
}
