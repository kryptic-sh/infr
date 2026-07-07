//! MTP (multi-token prediction) head weights + forward for qwen35 (issue #33 — see `docs/MTP.md`).
//!
//! Phase 1 scope: locate + shape-check the tensors the head's 1-layer graph needs (`load_mtp_head`
//! below) — no ops, no forward.
//!
//! Phase 2 scope (this module's second half): the head's 1-layer forward as its own backend-generic
//! `Graph` (`build_mtp_graph`, ported op-for-op from `seam.rs`'s qwen35 full-attention-layer
//! emission — see that function's doc for the exact line citations) plus the two engine-level driver
//! primitives `docs/MTP.md` specifies (`catch_up`/`draft`, porting `common/speculative.cpp`'s
//! `common_speculative_impl_draft_mtp`). NO target-loop integration yet (Phase 3) — `MtpHeadSession`
//! is a standalone session over the head's OWN 1-layer KV, driven by tokens/h-rows the CALLER
//! supplies (from the trunk's `h_out` tap — Phase 1).
//!
//! The head layer sits at GGUF index `blk.{n_layer}` (`Config::n_layer` is already the TRUNK
//! count — see `Config::n_layer_nextn`'s doc), one index past the trunk's last layer, and carries
//! a FULL qwen35 attention-layer tensor set (same names/shapes/interleaved-q+gate layout as a
//! trunk full-attention layer, `docs/QWEN35.md`) plus the `nextn.*` bridging tensors.

use anyhow::{anyhow, bail, Result};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::{TensorInfo, WeightSource};
use infr_gguf::Gguf;

mod backends;
pub use backends::*;

/// One resolved-and-shape-checked tensor: its GGUF metadata (name/shape/dtype/offset), kept around
/// so Phase 2's `wload` can re-derive the bytes from `.name` without this module re-reading the
/// file or holding a second copy of the mmap slice.
pub type MtpTensor = TensorInfo;

/// The qwen35 MTP head's tensors (see the module doc). Every required field here EXISTED in the
/// GGUF and had the expected shape at [`load_mtp_head`] time; the three `Option` fields are the
/// ones the reference allows to fall back to the main model's tensors when absent (`docs/MTP.md`'s
/// confirmed dump: the shipped 4B GGUF omits `embed_tokens`/`shared_head_head` — those two fall
/// back — but DOES ship its own `shared_head_norm`).
pub struct MtpHeadWeights {
    /// The GGUF block index of the head layer (`cfg.n_layer`, i.e. immediately after the trunk).
    pub il: usize,
    // ── standard qwen35 full-attention layer tensors (identical shapes to a trunk full-attn
    //    layer at a `(il+1) % full_attn_interval == 0` index — see `Config::is_qwen35_attn_layer`) ──
    pub attn_norm: MtpTensor,
    /// Interleaved q+gate: `[n_embd, head_dim * n_head * 2]` (see `Config::attn_out_gate`).
    pub attn_q: MtpTensor,
    pub attn_k: MtpTensor,
    pub attn_v: MtpTensor,
    pub attn_q_norm: MtpTensor,
    pub attn_k_norm: MtpTensor,
    pub attn_output: MtpTensor,
    pub post_attention_norm: MtpTensor,
    pub ffn_gate: MtpTensor,
    pub ffn_up: MtpTensor,
    pub ffn_down: MtpTensor,
    // ── NextN bridge (the tensors that make this an MTP head, not just another trunk layer) ──
    /// `[2*n_embd, n_embd]`: projects `concat(rmsnorm(embed(t)), rmsnorm(h_target))` down to
    /// `n_embd` before the layer's own attention (see `docs/MTP.md`'s forward pseudocode).
    pub eh_proj: MtpTensor,
    pub enorm: MtpTensor,
    pub hnorm: MtpTensor,
    /// Falls back to the main model's `token_embd.weight` when absent (the shipped 4B GGUF has no
    /// `nextn.embed_tokens` — see `docs/MTP.md`'s confirmed dump).
    pub embed_tokens: Option<MtpTensor>,
    /// Falls back to the main model's (tied) lm_head when absent.
    pub shared_head_head: Option<MtpTensor>,
    /// Falls back to the main model's `output_norm.weight` when absent.
    pub shared_head_norm: Option<MtpTensor>,
}

/// Cheap arch/head check from a resolved GGUF path — no `Config`/tensor validation, just the
/// metadata flag (mirrors `qwen35::is_qwen35`/`diffusion::is_diffusion_gemma`'s "peek without a
/// full load" convention). `infr compare`'s MTP DECODE section and the `--sweep` `mtp` column
/// (issue #33, phase 4 — perf-bottleneck visibility) use this to decide whether a model needs the
/// extra measurement at all, cheaper than `Config::from_gguf` (which validates every other field
/// too) for a check this narrow.
pub fn has_mtp_head(path: &std::path::Path) -> bool {
    let Ok(g) = Gguf::open(path) else {
        return false;
    };
    let arch = g.metadata().str("general.architecture").unwrap_or("");
    g.metadata()
        .u64(&format!("{arch}.nextn_predict_layers"))
        .unwrap_or(0)
        > 0
}

fn find<'a>(g: &'a Gguf, name: &str) -> Option<&'a TensorInfo> {
    g.tensors().iter().find(|t| t.name == name)
}

fn require(g: &Gguf, name: &str, want: &[usize]) -> Result<MtpTensor> {
    let t = find(g, name).ok_or_else(|| anyhow!("MTP head: missing tensor {name}"))?;
    if t.shape != want {
        bail!(
            "MTP head: {name} has shape {:?}, expected {:?}",
            t.shape,
            want
        );
    }
    Ok(t.clone())
}

/// Like [`require`] but `None` (not an error) when the tensor is simply absent — the reference's
/// fallback tensors (`nextn.embed_tokens`/`shared_head_head`/`shared_head_norm`). A PRESENT tensor
/// with the wrong shape is still an error (a real corruption, not an intentional omission).
fn optional(g: &Gguf, name: &str, want: &[usize]) -> Result<Option<MtpTensor>> {
    match find(g, name) {
        Some(t) if t.shape == want => Ok(Some(t.clone())),
        Some(t) => bail!(
            "MTP head: {name} has shape {:?}, expected {:?}",
            t.shape,
            want
        ),
        None => Ok(None),
    }
}

/// Locate + shape-check the qwen35 MTP head's tensors (see the module doc). Requires
/// `cfg.n_layer_nextn == 1` (Phase 1's only supported case — `Config::from_gguf` already rejects
/// anything else) and `cfg.qwen35`.
pub fn load_mtp_head(g: &Gguf, cfg: &crate::Config) -> Result<MtpHeadWeights> {
    if !cfg.qwen35 || cfg.n_layer_nextn != 1 {
        bail!(
            "load_mtp_head: requires a qwen35 GGUF with nextn_predict_layers==1 (got qwen35={}, \
             n_layer_nextn={})",
            cfg.qwen35,
            cfg.n_layer_nextn,
        );
    }
    let il = cfg.n_layer; // the MTP head sits immediately after the trunk (see Config::n_layer_nextn)
    let p = |s: &str| format!("blk.{il}.{s}");
    let ne = cfg.n_embd;
    let qdim = cfg.head_dim * cfg.n_head * 2; // interleaved q+gate, see Config::attn_out_gate
    let kv_dim = cfg.n_kv * cfg.head_dim;
    Ok(MtpHeadWeights {
        il,
        attn_norm: require(g, &p("attn_norm.weight"), &[ne])?,
        attn_q: require(g, &p("attn_q.weight"), &[ne, qdim])?,
        attn_k: require(g, &p("attn_k.weight"), &[ne, kv_dim])?,
        attn_v: require(g, &p("attn_v.weight"), &[ne, kv_dim])?,
        attn_q_norm: require(g, &p("attn_q_norm.weight"), &[cfg.head_dim])?,
        attn_k_norm: require(g, &p("attn_k_norm.weight"), &[cfg.head_dim])?,
        attn_output: require(
            g,
            &p("attn_output.weight"),
            &[cfg.head_dim * cfg.n_head, ne],
        )?,
        post_attention_norm: require(g, &p("post_attention_norm.weight"), &[ne])?,
        ffn_gate: require(g, &p("ffn_gate.weight"), &[ne, cfg.n_ff])?,
        ffn_up: require(g, &p("ffn_up.weight"), &[ne, cfg.n_ff])?,
        ffn_down: require(g, &p("ffn_down.weight"), &[cfg.n_ff, ne])?,
        eh_proj: require(g, &p("nextn.eh_proj.weight"), &[2 * ne, ne])?,
        enorm: require(g, &p("nextn.enorm.weight"), &[ne])?,
        hnorm: require(g, &p("nextn.hnorm.weight"), &[ne])?,
        embed_tokens: optional(g, &p("nextn.embed_tokens.weight"), &[ne, cfg.vocab])?,
        shared_head_head: optional(g, &p("nextn.shared_head_head.weight"), &[ne, cfg.vocab])?,
        shared_head_norm: optional(g, &p("nextn.shared_head_norm.weight"), &[ne])?,
    })
}

// ─── Phase 2: the head forward + the draft primitives (issue #33 — see `docs/MTP.md`) ──────────

/// A weight binder in the shape `seam.rs`'s (module-private) `BindWeight` uses: turns a
/// native-dtype GGUF tensor into a backend buffer + the effective dtype it now holds. Named here
/// (rather than inlined at each call site) purely to keep clippy's `type_complexity` lint quiet —
/// see [`upload_mtp_head_bufs`] / [`MtpHeadSession::build`].
type BindWeightFn<'a> =
    dyn Fn(&str, crate::seam::WBytes, DType, usize) -> Result<(Box<dyn Buffer>, DType)> + 'a;

/// [`upload_mtp_head_bufs`]'s return shape: the 16 uploaded weight buffers, index-parallel with
/// their (effective dtype, element count) — the pair `build_mtp_graph` needs to declare each
/// handle. Named purely to keep clippy's `type_complexity` lint quiet.
type WBufs = (Vec<Box<dyn Buffer>>, Vec<(DType, usize)>);

/// Resolve a fallback tensor from the MAIN model (the reference's `layer.nextn.X ? layer.nextn.X :
/// model.Y` — `qwen35.cpp:624-638`) when the head's own `nextn.*` tensor is absent.
fn main_output_norm(g: &Gguf, ne: usize) -> Result<MtpTensor> {
    require(g, "output_norm.weight", &[ne])
}

/// The tied/untied lm_head fallback (`qwen35.cpp:636`'s `layer.nextn.shared_head_head ? ... :
/// model.output`), mirroring `seam.rs`'s own tied-weight rule (`wload(&["output.weight"])`
/// vs the `token_embd.weight` fallback right above it in that file).
fn main_lm_head(g: &Gguf, ne: usize, vocab: usize) -> Result<MtpTensor> {
    if find(g, "output.weight").is_some() {
        require(g, "output.weight", &[ne, vocab])
    } else {
        require(g, "token_embd.weight", &[ne, vocab])
    }
}

/// Resolve the head's OWN embedding table (`nextn.embed_tokens`, dequantized) when the GGUF ships
/// one — `None` when it doesn't (the shipped 4B GGUF: `docs/MTP.md`'s confirmed dump), in which
/// case the caller passes the main model's already-dequantized `token_embd` straight to
/// [`MtpHeadSession::new_cpu`]/[`new_vulkan`] instead.
pub fn resolve_own_embed_table(g: &Gguf, head: &MtpHeadWeights) -> Result<Option<Vec<f32>>> {
    match &head.embed_tokens {
        Some(t) => Ok(Some(crate::load_tensor_dequant(g, &t.name)?.0)),
        None => Ok(None),
    }
}

/// Upload the head's 16 graph weights (attention-layer set + NextN bridge, in `wpush` order —
/// MUST match [`build_mtp_graph`]'s `wpush` calls 1:1) through the caller's `bind_weight` — the
/// SAME binder shape `seam.rs`'s `BindWeight` uses (zero-copy mmap on CPU, padded upload on
/// Vulkan — see `MtpHeadSession::new_cpu`/`new_vulkan`), so the head's tensors land in memory
/// exactly like the trunk's do. Falls back to the main model's `output_norm`/tied lm_head for the
/// two optional NextN tensors that are absent (`docs/MTP.md`'s confirmed dump — `shared_head_norm`
/// IS present in the shipped 4B GGUF, so only the lm_head fallback fires there in practice).
fn upload_mtp_head_bufs(
    be: &dyn Backend,
    bind_weight: &BindWeightFn,
    g: &Gguf,
    cfg: &crate::Config,
    head: &MtpHeadWeights,
) -> Result<WBufs> {
    let _ = be; // kept for symmetry with seam's wload closures; binder itself owns `be`
    let mut bufs = Vec::with_capacity(16);
    let mut specs = Vec::with_capacity(16);
    let mut push = |t: &MtpTensor| -> Result<()> {
        let tb = g.tensor_bytes_arc(&t.name).map_err(|e| anyhow!("{e}"))?;
        let numel: usize = t.shape.iter().product();
        let (buf, dt) = bind_weight(&t.name, crate::seam::WBytes::Mmap(tb), t.dtype, numel)?;
        bufs.push(buf);
        specs.push((dt, numel));
        Ok(())
    };
    push(&head.attn_norm)?;
    push(&head.attn_q)?;
    push(&head.attn_k)?;
    push(&head.attn_v)?;
    push(&head.attn_q_norm)?;
    push(&head.attn_k_norm)?;
    push(&head.attn_output)?;
    push(&head.post_attention_norm)?;
    push(&head.ffn_gate)?;
    push(&head.ffn_up)?;
    push(&head.ffn_down)?;
    push(&head.eh_proj)?;
    push(&head.enorm)?;
    push(&head.hnorm)?;
    match &head.shared_head_norm {
        Some(t) => push(t)?,
        None => push(&main_output_norm(g, cfg.n_embd)?)?,
    }
    match &head.shared_head_head {
        Some(t) => push(t)?,
        None => push(&main_lm_head(g, cfg.n_embd, cfg.vocab)?)?,
    }
    Ok((bufs, specs))
}

/// The head graph's tensor handles `MtpHeadSession::forward` binds/reads each call.
struct MtpHandles {
    e_raw: TensorId,
    h_in: TensorId,
    positions: TensorId,
    k_cache: TensorId,
    v_cache: TensorId,
    /// The 16 weight handles, in [`upload_mtp_head_bufs`]'s push order (index-parallel with
    /// `MtpHeadSession::wbufs`).
    weights: Vec<TensorId>,
    logits: TensorId,
    h_mtp: TensorId,
}

/// Build the MTP head's 1-layer forward graph for `rows` `(token, h_target)` pairs starting at
/// absolute KV position `start_pos` (`docs/MTP.md`'s forward pseudocode):
/// ```text
/// e = rmsnorm(embed(t), enorm);  h = rmsnorm(h_target, hnorm)
/// x = eh_proj @ concat([e; h])                    # eh_proj: [2ne, ne] (in=2ne, out=ne)
/// x = qwen35_attention_layer(x, pos)               # own 1-layer KV, causal
/// h_mtp = rmsnorm(x, shared_head_norm)             # fed back on the NEXT draft step
/// logits = lm_head @ h_mtp
/// ```
/// The attention-layer ops (interleaved q+gate split, qk-norm+RoPE, sigmoid out-gate, SwiGLU FFN)
/// are ported op-for-op from `seam.rs`'s qwen35 `c.attn_out_gate` branch (that file's
/// `generate_dense_backend::build` closure, roughly lines 2269-2707: the per-layer attn-input norm,
/// the interleaved qg projection + per-head `CopyStrided` split, k/v projections, `QkNormRope`,
/// `WriteKv`, `Attention`, the post-attention `GatedAct(Sigmoid)` gate, the o-projection, the
/// residual add, the ffn-norm + dense SwiGLU FFN, and the final residual add) — see
/// `qwen35.cpp:556-622` for the reference this ports.
///
/// The `eh_proj` "concat" (`qwen35.cpp:548`'s `ggml_concat(e_norm, h_norm, dim=0)`): with infr's
/// `Op::Linear` weight convention (`[out, in]` row-major — a weight's `in`-dim is the CONTIGUOUS
/// per-row axis), `eh_proj`'s GGUF shape `[2ne, ne]` (in=2ne, out=ne) means row `j`'s `2ne`
/// contiguous elements are `[a_j (ne, dot e) | b_j (ne, dot h)]` — i.e. `ggml_concat(dim=0)` is a
/// COLUMN split of each output row, not a row split. Rather than a strided weight VIEW (the same
/// trick `attn_q`'s interleaved q/gate split already uses, just on the weight side instead of an
/// activation), the least-new-code correct form builds the concatenated ACTIVATION on device with
/// two existing `Op::CopyStrided` (e into `concat[0..ne]`, h into `concat[ne..2ne]`, per row) and
/// then ONE ordinary `Op::Linear` over the full `[2ne,ne]` weight — no new `Op` variant, and it's
/// literally the reference's `ggml_concat` + `mm` in the same order. Proved by
/// `mtp_head_forward_finite`'s finite-logits check plus `h_tap_matches_lm_head`-style consistency
/// implicit in the parity test (a layout bug here would show up as garbage/NaN logits or gross
/// CPU/Vulkan disagreement, not a subtle numeric drift).
fn build_mtp_graph(
    cfg: &crate::Config,
    wspecs: &[(DType, usize)],
    max_ctx: usize,
    rows: usize,
    start_pos: usize,
) -> (Graph, MtpHandles) {
    let ne = cfg.n_embd;
    let nh = cfg.n_head;
    let nkv = cfg.n_kv;
    let hd = cfg.head_dim;
    let qrow = nh * hd;
    let kvrow = nkv * hd;
    let nff = cfg.n_ff;
    let eps = cfg.rms_eps;
    let theta = cfg.rope_theta; // qwen35: uniform (no gemma dual-rope) — Config::layer_rope_theta
    let rope_dim = cfg.rope_dim; // collapses to this for a non-SWA model; see Config's doc.
    let scale = 1.0 / (hd as f32).sqrt(); // qwen35 isn't gemma4 (no QK-norm-only scale of 1.0)

    let mut g = Graph::new();
    let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
    let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16); // KV dtype: fixed f16 (see this
                                                                // fn's doc on the Phase-3 TODO)
    let i32d = |n: usize| TensorDesc::new(vec![n], DType::I32);

    let e_raw = g.input(f32d(rows * ne));
    let h_in = g.input(f32d(rows * ne));
    let positions = g.input(i32d(rows));
    let k_cache = g.input(f16d(max_ctx * kvrow));
    let v_cache = g.input(f16d(max_ctx * kvrow));

    let mut wi = 0usize;
    let mut wpush = |g: &mut Graph| -> TensorId {
        let (dt, n) = wspecs[wi];
        wi += 1;
        g.weight(TensorDesc::new(vec![n], dt))
    };
    // MUST match `upload_mtp_head_bufs`'s push order exactly.
    let w_attn_norm = wpush(&mut g);
    let w_q = wpush(&mut g);
    let w_k = wpush(&mut g);
    let w_v = wpush(&mut g);
    let w_qn = wpush(&mut g);
    let w_kn = wpush(&mut g);
    let w_o = wpush(&mut g);
    let w_ffn_norm = wpush(&mut g);
    let w_gate = wpush(&mut g);
    let w_up = wpush(&mut g);
    let w_down = wpush(&mut g);
    let w_eh_proj = wpush(&mut g);
    let w_enorm = wpush(&mut g);
    let w_hnorm = wpush(&mut g);
    let w_head_norm = wpush(&mut g);
    let w_lm_head = wpush(&mut g);
    let weights = vec![
        w_attn_norm,
        w_q,
        w_k,
        w_v,
        w_qn,
        w_kn,
        w_o,
        w_ffn_norm,
        w_gate,
        w_up,
        w_down,
        w_eh_proj,
        w_enorm,
        w_hnorm,
        w_head_norm,
        w_lm_head,
    ];

    let logits = g.output(f32d(rows * cfg.vocab));
    let h_mtp = g.output(f32d(rows * ne));

    // scratch
    let e_norm = g.internal(f32d(rows * ne));
    let h_norm = g.internal(f32d(rows * ne));
    let concat = g.internal(f32d(rows * 2 * ne));
    let resid = g.internal(f32d(rows * ne)); // eh_proj output; mutated in place across the residuals
    let hn = g.internal(f32d(rows * ne));
    let qg = g.internal(f32d(rows * 2 * qrow));
    let q = g.internal(f32d(rows * qrow));
    let gate_a = g.internal(f32d(rows * qrow));
    let k = g.internal(f32d(rows * kvrow));
    let v = g.internal(f32d(rows * kvrow));
    let q16 = g.internal(f16d(rows * qrow));
    let k16 = g.internal(f16d(rows * kvrow));
    let attn = g.internal(f32d(rows * qrow));
    let sub = g.internal(f32d(rows * ne));
    let gbuf = g.internal(f32d(rows * nff));
    let ubuf = g.internal(f32d(rows * nff));
    let actbuf = g.internal(f32d(rows * nff));
    let hn2 = g.internal(f32d(rows * ne));
    let layer_out = g.internal(f32d(rows * ne));

    // e = rmsnorm(embed(t), enorm); h = rmsnorm(h_target, hnorm) — qwen35.cpp:542-546.
    g.push(Op::RmsNorm {
        x: e_raw,
        weight: w_enorm,
        dst: e_norm,
        rows: rows as u32,
        dim: ne as u32,
        eps,
    });
    g.push(Op::RmsNorm {
        x: h_in,
        weight: w_hnorm,
        dst: h_norm,
        rows: rows as u32,
        dim: ne as u32,
        eps,
    });
    // concat = [e_norm | h_norm] per row (this fn's doc on the eh_proj layout) — qwen35.cpp:548.
    g.push(Op::CopyStrided {
        src: e_norm,
        src_off: 0,
        src_stride: ne as u32,
        dst: concat,
        dst_off: 0,
        dst_stride: (2 * ne) as u32,
        rows: rows as u32,
        n: ne as u32,
    });
    g.push(Op::CopyStrided {
        src: h_norm,
        src_off: 0,
        src_stride: ne as u32,
        dst: concat,
        dst_off: ne as u32,
        dst_stride: (2 * ne) as u32,
        rows: rows as u32,
        n: ne as u32,
    });
    // resid = eh_proj @ concat — qwen35.cpp:551 (`inpSA` in the reference).
    g.push(Op::Linear {
        x: concat,
        weight: w_eh_proj,
        dst: resid,
        m: rows as u32,
        in_f: (2 * ne) as u32,
        out_f: ne as u32,
        w_off: 0,
    });

    // ── one qwen35 full-attention layer on `resid` (own 1-layer KV) — seam.rs's
    //    `c.attn_out_gate` branch, qwen35.cpp:556-619 ──
    g.push(Op::RmsNorm {
        x: resid,
        weight: w_attn_norm,
        dst: hn,
        rows: rows as u32,
        dim: ne as u32,
        eps,
    });
    // interleaved q+gate projection + per-head split (seam.rs:2464-2493, qwen35.cpp:559-576).
    g.push(Op::Linear {
        x: hn,
        weight: w_q,
        dst: qg,
        m: rows as u32,
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
            rows: rows as u32,
            n: hd as u32,
        });
        g.push(Op::CopyStrided {
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
    g.push(Op::Linear {
        x: hn,
        weight: w_k,
        dst: k,
        m: rows as u32,
        in_f: ne as u32,
        out_f: kvrow as u32,
        w_off: 0,
    });
    g.push(Op::Linear {
        x: hn,
        weight: w_v,
        dst: v,
        m: rows as u32,
        in_f: ne as u32,
        out_f: kvrow as u32,
        w_off: 0,
    });
    // K: fused QkNorm+RoPE (own KV write immediately follows, matching the trunk's fused-write
    // adjacency convention — seam.rs:2587-2636).
    g.push(Op::QkNormRope {
        x: k,
        weight: w_kn,
        positions,
        dst: k16,
        rows: rows as u32,
        n_head: nkv as u32,
        head_dim: hd as u32,
        rope_dim: rope_dim as u32,
        theta,
        eps,
        freq_factors: None, // gemma4-only; qwen35 never sets this
    });
    g.push(Op::WriteKv {
        src: k16,
        cache: k_cache,
        rows: rows as u32,
        row_stride: kvrow as u32,
        pos: start_pos as u32,
    });
    g.push(Op::WriteKv {
        src: v,
        cache: v_cache,
        rows: rows as u32,
        row_stride: kvrow as u32,
        pos: start_pos as u32,
    });
    g.push(Op::QkNormRope {
        x: q,
        weight: w_qn,
        positions,
        dst: q16,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rope_dim as u32,
        theta,
        eps,
        freq_factors: None,
    });
    g.push(Op::Attention {
        q: q16,
        k_cache,
        v_cache,
        dst: attn,
        rows: rows as u32,
        kv_len: (start_pos + rows) as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale,
        mask: AttnMask::Causal,
        pos: start_pos as u32,
    });
    // sigmoid out-gate BEFORE the o-projection — seam.rs:2686-2698, qwen35.cpp:602.
    g.push(Op::GatedAct {
        gate: gate_a,
        up: attn,
        dst: attn,
        rows: rows as u32,
        nff: qrow as u32,
        act: Activation::Sigmoid,
        up_off: 0,
    });
    g.push(Op::Linear {
        x: attn,
        weight: w_o,
        dst: sub,
        m: rows as u32,
        in_f: qrow as u32,
        out_f: ne as u32,
        w_off: 0,
    });
    // attn residual — qwen35.cpp:606 (`cur = add(cur, inpSA)`; `inpSA` IS `resid` here).
    g.push(Op::Add {
        a: resid,
        b: sub,
        dst: resid,
        n: (rows * ne) as u32,
    });
    // ffn — qwen35.cpp:610-619 (SwiGLU; qwen35 names the pre-FFN norm `post_attention_norm`, not
    // `ffn_norm` — see `Config::qwen35`'s doc / `w_ffn_norm` bound to `post_attention_norm.weight`
    // in `upload_mtp_head_bufs`).
    g.push(Op::RmsNorm {
        x: resid,
        weight: w_ffn_norm,
        dst: hn2,
        rows: rows as u32,
        dim: ne as u32,
        eps,
    });
    g.push(Op::Linear {
        x: hn2,
        weight: w_gate,
        dst: gbuf,
        m: rows as u32,
        in_f: ne as u32,
        out_f: nff as u32,
        w_off: 0,
    });
    g.push(Op::Linear {
        x: hn2,
        weight: w_up,
        dst: ubuf,
        m: rows as u32,
        in_f: ne as u32,
        out_f: nff as u32,
        w_off: 0,
    });
    g.push(Op::GatedAct {
        gate: gbuf,
        up: ubuf,
        dst: actbuf,
        rows: rows as u32,
        nff: nff as u32,
        act: Activation::Silu,
        up_off: 0,
    });
    g.push(Op::Linear {
        x: actbuf,
        weight: w_down,
        dst: sub,
        m: rows as u32,
        in_f: nff as u32,
        out_f: ne as u32,
        w_off: 0,
    });
    // ffn residual — qwen35.cpp:621.
    g.push(Op::Add {
        a: resid,
        b: sub,
        dst: layer_out,
        n: (rows * ne) as u32,
    });

    // h_mtp = rmsnorm(layer_out, shared_head_norm) — qwen35.cpp:624-631 (`res->t_h_nextn`, ALSO
    // fed back as the next draft step's `h_target` when self-chaining — see `draft`'s doc).
    g.push(Op::RmsNorm {
        x: layer_out,
        weight: w_head_norm,
        dst: h_mtp,
        rows: rows as u32,
        dim: ne as u32,
        eps,
    });
    // logits = lm_head @ h_mtp — qwen35.cpp:633-642.
    g.push(Op::Linear {
        x: h_mtp,
        weight: w_lm_head,
        dst: logits,
        m: rows as u32,
        in_f: ne as u32,
        out_f: cfg.vocab as u32,
        w_off: 0,
    });

    (
        g,
        MtpHandles {
            e_raw,
            h_in,
            positions,
            k_cache,
            v_cache,
            weights,
            logits,
            h_mtp,
        },
    )
}

/// A live MTP head: the head's OWN uploaded weights + its OWN 1-layer KV (independent of the
/// trunk's — `docs/MTP.md`'s driver section), driven by `forward`/`catch_up`/`draft`. Borrows the
/// backend and the host embedding table for its lifetime (mirrors `seam.rs`'s session
/// structs) — construct via [`new_cpu`](Self::new_cpu) / [`new_vulkan`](Self::new_vulkan).
///
/// Not cached like `DenoiseCache` (`seam.rs`): `Op::Attention::kv_len` and `Op::WriteKv::pos`
/// are BAKED into the graph at build time (there's no dynamic-`kv_len` binding in this IR — see
/// `docs/MTP.md`'s IR), and both change on essentially every `catch_up`/`draft` call (`draft`
/// advances `kv_len` by one every step). A shape-keyed cache would almost always miss, so
/// `forward` rebuilds + recompiles a fresh graph every call instead — exactly the trunk's default
/// per-token rebuild path for callers that don't opt into `decode_replay`. Phase 3 can revisit if
/// head-forward dispatch overhead matters on the hot serving path (e.g. a `kv_len`/`pos` INPUT
/// binding instead of an inline constant, mirroring `Capabilities::decode_replay`).
pub struct MtpHeadSession<'a> {
    be: &'a dyn Backend,
    cfg: crate::Config,
    /// The table `forward` gathers embedding rows from (host-side, like every other embed-gather
    /// call site on this seam) — the main model's `token_embd`, or [`resolve_own_embed_table`]'s
    /// result when the GGUF ships its own `nextn.embed_tokens`.
    embed_table: &'a [f32],
    wbufs: Vec<Box<dyn Buffer>>,
    wspecs: Vec<(DType, usize)>,
    k_cache: Box<dyn Buffer>,
    v_cache: Box<dyn Buffer>,
    max_ctx: usize,
    /// Phase 3 perf instrumentation (issue #33, `INFR_MTP_TIME=1` — see
    /// `generate_mtp_spec_vulkan`'s doc on the per-call rebuild cost this is meant to quantify):
    /// cumulative seconds spent building+compiling a fresh [`Graph`] per [`forward`](Self::forward)
    /// call vs executing it, reset by [`take_timing`](Self::take_timing).
    build_secs: f64,
    exec_secs: f64,
}

impl<'a> MtpHeadSession<'a> {
    /// Construct over the CPU reference backend (zero-copy mmap weight upload — mirrors
    /// `seam/model.rs`'s `DiffusionGemmaCpuSession` closures).
    pub fn new_cpu(
        cpu_be: &'a infr_cpu::CpuBackend,
        g: &Gguf,
        cfg: &crate::Config,
        head: &MtpHeadWeights,
        embed_table: &'a [f32],
        max_ctx: usize,
    ) -> Result<Self> {
        Self::build(
            cpu_be,
            &|_name, tb, dt, _n| match tb {
                crate::seam::WBytes::Mmap(tb) => Ok((cpu_be.map_weight(tb), dt)),
                crate::seam::WBytes::Owned(v) => {
                    let buf = cpu_be
                        .alloc(v.len().max(1), BufferUsage::Weights)
                        .map_err(|e| anyhow!("{e}"))?;
                    cpu_be
                        .upload(buf.as_ref(), &v)
                        .map_err(|e| anyhow!("{e}"))?;
                    Ok((buf, dt))
                }
            },
            g,
            cfg,
            head,
            embed_table,
            max_ctx,
        )
    }

    /// Construct over the Vulkan backend (padded upload — mirrors `seam/model.rs`'s
    /// `DiffusionGemmaVulkanSession` closures).
    pub fn new_vulkan(
        vk: &'a infr_vulkan::VulkanBackend,
        g: &Gguf,
        cfg: &crate::Config,
        head: &MtpHeadWeights,
        embed_table: &'a [f32],
        max_ctx: usize,
    ) -> Result<Self> {
        Self::build(
            vk,
            &|_name, tb, dt, _n| {
                let padded = infr_vulkan::linear::pad_to_u32_align(&tb);
                let buf = vk
                    .alloc(padded.len(), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                vk.upload(buf.as_ref(), &padded)
                    .map_err(|e| anyhow!("{e}"))?;
                Ok((buf, dt))
            },
            g,
            cfg,
            head,
            embed_table,
            max_ctx,
        )
    }

    /// Construct over the Metal backend (raw native-dtype upload — mirrors
    /// `seam.rs`'s `generate_dense_metal_session` weight closure).
    #[cfg(target_os = "macos")]
    pub fn new_metal(
        mtl: &'a infr_metal::MetalBackend,
        g: &Gguf,
        cfg: &crate::Config,
        head: &MtpHeadWeights,
        embed_table: &'a [f32],
        max_ctx: usize,
    ) -> Result<Self> {
        Self::build(
            mtl,
            &|_name, tb, dt, _n| {
                let buf = mtl
                    .alloc(tb.len().max(1), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                mtl.upload(buf.as_ref(), &tb).map_err(|e| anyhow!("{e}"))?;
                Ok((buf, dt))
            },
            g,
            cfg,
            head,
            embed_table,
            max_ctx,
        )
    }

    fn build(
        be: &'a dyn Backend,
        bind_weight: &BindWeightFn,
        g: &Gguf,
        cfg: &crate::Config,
        head: &MtpHeadWeights,
        embed_table: &'a [f32],
        max_ctx: usize,
    ) -> Result<Self> {
        let (wbufs, wspecs) = upload_mtp_head_bufs(be, bind_weight, g, cfg, head)?;
        let kvrow = cfg.n_kv * cfg.head_dim;
        let k_cache = be
            .alloc(max_ctx * kvrow * 2, BufferUsage::Activations) // f16 KV — 2 bytes/elem
            .map_err(|e| anyhow!("{e}"))?;
        let v_cache = be
            .alloc(max_ctx * kvrow * 2, BufferUsage::Activations)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(Self {
            be,
            cfg: cfg.clone(),
            embed_table,
            wbufs,
            wspecs,
            k_cache,
            v_cache,
            max_ctx,
            build_secs: 0.0,
            exec_secs: 0.0,
        })
    }

    /// Drain the cumulative build-vs-exec timing since the last call (or construction) —
    /// `generate_mtp_spec_vulkan`'s `INFR_MTP_TIME=1` breakdown reads this once per spec cycle.
    pub fn take_timing(&mut self) -> (f64, f64) {
        (
            std::mem::take(&mut self.build_secs),
            std::mem::take(&mut self.exec_secs),
        )
    }

    /// One head forward over `rows = tokens.len()` `(token, h_target)` pairs at absolute KV
    /// positions `start_pos..start_pos+rows` (`docs/MTP.md`'s forward pseudocode — see
    /// `build_mtp_graph`'s doc for the op-level port). Returns `(logits [rows*vocab], h_mtp
    /// [rows*ne])`. `WriteKv` OVERWRITES any existing rows at these positions (a re-draft's next
    /// `catch_up`/`draft` at the same positions is expected to do exactly this — `docs/MTP.md`'s
    /// driver section, last paragraph).
    pub fn forward(
        &mut self,
        tokens: &[u32],
        h_rows: &[f32],
        start_pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let rows = tokens.len();
        let ne = self.cfg.n_embd;
        anyhow::ensure!(
            h_rows.len() == rows * ne,
            "MTP head forward: h_rows length {} != rows*ne {}",
            h_rows.len(),
            rows * ne
        );
        anyhow::ensure!(
            start_pos + rows <= self.max_ctx,
            "MTP head forward: KV overflow ({start_pos}+{rows} > max_ctx {})",
            self.max_ctx
        );

        let mut e_raw = Vec::with_capacity(rows * ne);
        for &t in tokens {
            let base = t as usize * ne;
            e_raw.extend_from_slice(&self.embed_table[base..base + ne]);
        }
        let positions: Vec<i32> = (start_pos as i32..(start_pos + rows) as i32).collect();

        let t_build = std::time::Instant::now();
        let (graph, h) = build_mtp_graph(&self.cfg, &self.wspecs, self.max_ctx, rows, start_pos);
        let plan = self.be.compile(&graph).map_err(|e| anyhow!("{e}"))?;
        self.build_secs += t_build.elapsed().as_secs_f64();

        let t_exec = std::time::Instant::now();
        let e_buf = self
            .be
            .alloc(rows * ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let h_buf = self
            .be
            .alloc(rows * ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let pos_buf = self
            .be
            .alloc(rows * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let logits_buf = self
            .be
            .alloc(rows * self.cfg.vocab * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        let hmtp_buf = self
            .be
            .alloc(rows * ne * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;

        self.be
            .upload(e_buf.as_ref(), bytemuck::cast_slice(&e_raw))
            .map_err(|e| anyhow!("{e}"))?;
        self.be
            .upload(h_buf.as_ref(), bytemuck::cast_slice(h_rows))
            .map_err(|e| anyhow!("{e}"))?;
        self.be
            .upload(pos_buf.as_ref(), bytemuck::cast_slice(&positions))
            .map_err(|e| anyhow!("{e}"))?;

        let mut b = Bindings::new();
        b.bind(h.e_raw, e_buf.as_ref());
        b.bind(h.h_in, h_buf.as_ref());
        b.bind(h.positions, pos_buf.as_ref());
        b.bind(h.k_cache, self.k_cache.as_ref());
        b.bind(h.v_cache, self.v_cache.as_ref());
        for (i, wid) in h.weights.iter().enumerate() {
            b.bind(*wid, self.wbufs[i].as_ref());
        }
        b.bind(h.logits, logits_buf.as_ref());
        b.bind(h.h_mtp, hmtp_buf.as_ref());

        self.be
            .execute(plan.as_ref(), &b)
            .map_err(|e| anyhow!("{e}"))?;

        let mut logits = vec![0f32; rows * self.cfg.vocab];
        self.be
            .download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
            .map_err(|e| anyhow!("{e}"))?;
        let mut h_mtp = vec![0f32; rows * ne];
        self.be
            .download(hmtp_buf.as_ref(), bytemuck::cast_slice_mut(&mut h_mtp))
            .map_err(|e| anyhow!("{e}"))?;
        self.exec_secs += t_exec.elapsed().as_secs_f64();
        Ok((logits, h_mtp))
    }
}

/// llama.cpp's default `--spec-draft-p-min` (`common/common.h:329`'s
/// `common_params_speculative_draft::p_min = 0.0f`) — i.e. `n_max` alone bounds a draft run unless
/// the caller raises it. Matches `docs/MTP.md`'s captured oracle run (no `--spec-draft-p-min` flag
/// was passed for the 2.0x number).
pub const DEFAULT_P_MIN: f32 = 0.0;

/// The MTP catch-up hook (`docs/MTP.md`'s `process()`, `speculative.cpp:1354-1470`'s single-head
/// branch): re-syncs the head's own KV to the target's committed rows after EVERY target
/// prefill/decode ubatch. `tokens[i]` pairs with `h_rows[i]` — the CALLER does the "shift `h` right
/// by one + splice in the previous call's `pending_h`" (`speculative.cpp:1396-1417`'s `h_tgt`
/// memcpy + `set_h`), so this primitive stays a plain "forward these `R` `(token, h)` pairs at
/// `start_pos`, discard the logits" call with no special-casing of the first row. The head's own KV
/// rows `start_pos..start_pos+tokens.len()` are (over)written — see `forward`'s doc on the
/// overwrite semantics a re-draft relies on.
pub fn catch_up(
    sess: &mut MtpHeadSession,
    tokens: &[u32],
    h_rows: &[f32],
    start_pos: usize,
) -> Result<()> {
    sess.forward(tokens, h_rows, start_pos)?;
    Ok(())
}

/// The MTP draft loop (`docs/MTP.md`'s `draft()`, `speculative.cpp:1472-1621`'s single-head, non-
/// chained, non-mem-shared branch): self-chaining greedy decode over the head's OWN KV, ONE row per
/// step. Starts at `(id_last, pending_h)` at absolute position `n_past` (the reference's
/// `dp.n_past`/`dp.id_last` — the first UNCOMMITTED position and the last committed token); each
/// accepted step re-feeds `(sampled_id, h_mtp)` — the head's OWN hidden output, NOT the target's —
/// at the next position (`speculative.cpp:1542`'s `h_row = llama_get_embeddings_nextn_ith(ctx_dft,
/// ...)`), so the head's KV grows by exactly the rows drafted. Stops at the first row whose top-1
/// probability is `< p_min` (`speculative.cpp:1556`) or once `n_max` tokens are drafted
/// (`speculative.cpp:1570`) — plain greedy argmax either way (temp=0 throughout `docs/MTP.md`'s
/// validated oracle run; the reference's draft sampler is `top_k=10` but only ever reads
/// `data[0]`). Returns `(token, top1_prob)` pairs so a caller can inspect confidence without a
/// second pass; re-drafting from the same `n_past` OVERWRITES the head's KV at those positions
/// (`forward`'s doc) — the caller doesn't need to reset anything first.
pub fn draft(
    sess: &mut MtpHeadSession,
    id_last: u32,
    pending_h: &[f32],
    n_past: usize,
    p_min: f32,
    n_max: usize,
) -> Result<Vec<(u32, f32)>> {
    let mut result = Vec::with_capacity(n_max);
    let mut tok = id_last;
    let mut h = pending_h.to_vec();
    for pos in n_past..n_past + n_max {
        let (logits, h_mtp) = sess.forward(&[tok], &h, pos)?;
        let (id, p) = top1_softmax(&logits);
        if p < p_min {
            break;
        }
        result.push((id, p));
        tok = id;
        h = h_mtp; // self-chain: the head's OWN h_mtp feeds the next draft step
    }
    Ok(result)
}

/// Greedy top-1 token id + its softmax probability over one `[vocab]` logits row
/// (`speculative.cpp`'s `common_sampler_sample`/`cur_p->data[0]`): since only the winning
/// candidate's OWN probability mass is needed (not the runner-up ids a real top-k sampler would
/// keep), argmax + `1/Σexp(logit - max)` is exactly `softmax(logits)[argmax]` without materializing
/// the full distribution.
fn top1_softmax(logits: &[f32]) -> (u32, f32) {
    let (mut amax, mut m) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > m {
            m = v;
            amax = i;
        }
    }
    let z: f32 = logits.iter().map(|&v| (v - m).exp()).sum();
    (amax as u32, 1.0 / z)
}

// ─── Phase 3: the MTP self-speculative generation loop + run/serve wiring (issue #33) ───────────

/// llama.cpp's oracle run used `--spec-draft-n-max 6` (`docs/MTP.md`) — the max candidates drafted
/// per cycle before a batched verify.
pub const DEFAULT_N_MAX: usize = 6;

/// One batched VERIFY forward (`seam.rs`'s VERIFY branch) over `tokens`' un-cached suffix,
/// on a persistent Vulkan session, ALSO capturing `h` for every returned row — the primitive
/// [`generate_mtp_spec_vulkan`] needs every cycle (`crate::seam::model::SeamModel`'s own
/// `verify_dense_*_with_h` helpers all use a FRESH one-shot session; this one threads `state`
/// across calls so the trunk's KV/DeltaNet state persists cycle to cycle — the whole point of
/// self-speculative decoding). Returns `(argmax ids [m], h [m*n_embd])`, `m = tokens.len() -
/// (tokens already cached in `state`)`.
#[allow(clippy::too_many_arguments)]
/// One TARGET-trunk verify forward over `tokens`, returning (per-row greedy argmax ids `[m]`,
/// all-row hidden states `[m*n_embd]`) — the hidden rows are the extra tap the MTP head
/// consumes. Backend-generic: the trunk graph is `generate_dense_backend` (which already takes
/// `be` + a bind closure), so this drives Vulkan, Metal, or CPU by the caller's binder.
///
/// GPU-resident accept (issue #31): the runner's VERIFY branch appends a per-row `Op::Argmax`
/// when the backend supports it (`Capabilities::argmax_rows`; A/B escape via INFR_NO_GPU_ARGMAX
/// or INFR_NO_GPU_MTP_ACCEPT) so only m×4 id bytes cross the bus, not the m×vocab f32 logits.
/// When the device path is off (Metal, env A/B), the runner fills `logits` instead (ids left
/// empty) and THIS function does the exact host argmax per row — either way the caller sees the
/// same `[m]` id vector, bit-identical (both paths argmax the same f32 logits with strict-`>`
/// lowest-index tie-break).
fn run_verify(
    be: &dyn Backend,
    bind: &BindWeightFn,
    g: &Gguf,
    cfg: &crate::Config,
    token_embd: &[f32],
    tokens: &[u32],
    state: &mut Option<crate::seam::SeamKv>,
    max_ctx: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let mut logits = Vec::new();
    let mut ids = Vec::new();
    let mut h = Vec::new();
    crate::seam::generate_dense_backend(
        be,
        bind,
        g,
        cfg,
        token_embd,
        None,
        tokens,
        0,
        |_| {},
        state,
        max_ctx,
        None,
        Some(&mut logits),
        Some(&mut ids),
        None,
        Some(&mut h),
        None,
    )?;
    if ids.is_empty() {
        // Host fallback: the runner downloaded the m×vocab logits instead (see this fn's doc).
        let m = h.len() / cfg.n_embd;
        anyhow::ensure!(
            logits.len() == m * cfg.vocab,
            "run_verify: host-logits fallback expected {}*{} logits, got {}",
            m,
            cfg.vocab,
            logits.len()
        );
        ids = (0..m)
            .map(|j| argmax_row(&logits[j * cfg.vocab..(j + 1) * cfg.vocab]))
            .collect();
    }
    Ok((ids, h))
}

/// Cumulative per-phase wall time + accept-rate counters over one [`generate_mtp_spec_vulkan_timed`]
/// run (issue #33, phase 4 — `infr bench`/`infr compare`'s perf-bottleneck visibility pass: this is
/// the struct return `docs/MTP.md`'s `INFR_MTP_TIME` per-cycle `eprintln!`s are refactored to feed,
/// instead of the caller scraping stderr). `draft_secs`/`verify_secs`/`catchup_secs` sum every
/// cycle's three timed sections (the SAME three the `[mtp cycle N]` debug line prints); the one-time
/// prompt-prime VERIFY forward is deliberately excluded — it's already `GenStats::prompt_secs`, not
/// part of the steady-state decode this breakdown characterizes.
#[derive(Debug, Clone, Copy, Default)]
pub struct MtpTiming {
    pub draft_secs: f64,
    pub verify_secs: f64,
    pub catchup_secs: f64,
    pub total_drafted: usize,
    pub total_accepted: usize,
}

impl MtpTiming {
    /// Fold another run's counters in (bench averages several reps — see `infr-cli`'s
    /// `bench_mtp_tg`).
    pub fn add(&mut self, other: &MtpTiming) {
        self.draft_secs += other.draft_secs;
        self.verify_secs += other.verify_secs;
        self.catchup_secs += other.catchup_secs;
        self.total_drafted += other.total_drafted;
        self.total_accepted += other.total_accepted;
    }

    /// Draft acceptance rate (`accepted / drafted`) — llama.cpp's `alpha`.
    pub fn alpha(&self) -> f64 {
        self.total_accepted as f64 / self.total_drafted.max(1) as f64
    }

    /// `(draft%, verify%, catchup%)` of the three phases' summed wall time — the breakdown that
    /// tells the next perf pass where the net-negative time actually goes.
    pub fn phase_shares(&self) -> (f64, f64, f64) {
        let total = self.draft_secs + self.verify_secs + self.catchup_secs;
        if total <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        (
            100.0 * self.draft_secs / total,
            100.0 * self.verify_secs / total,
            100.0 * self.catchup_secs / total,
        )
    }
}

/// Greedy argmax over one `[vocab]` logits row (unlike [`top1_softmax`], no probability needed —
/// `spec_accept`/the verify-round check only reads the winning id).
fn argmax_row(row: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

/// The backend-generic MTP draft-verify-catchup loop — the shared body of the `_vulkan`/`_metal`
/// drivers. The trunk verify runs through `run_verify(be, bind, …)`; the draft head is the
/// caller-built [`MtpHeadSession`] (bound to the same `be`). Everything between — draft, accept,
/// catch-up, streaming — is backend-agnostic.
#[allow(clippy::too_many_arguments)]
fn generate_mtp_spec_core(
    be: &dyn Backend,
    bind: &BindWeightFn,
    model: &crate::SeamModel,
    head_sess: &mut MtpHeadSession,
    max_ctx: usize,
    prompt: &str,
    max_new: usize,
    mut on_piece: impl FnMut(&str),
) -> Result<(crate::GenStats, MtpTiming)> {
    let cfg = model.config();
    let ne = cfg.n_embd;
    let n_max = DEFAULT_N_MAX;
    let time_mtp = std::env::var("INFR_MTP_TIME").is_ok();
    let ignore_eos = std::env::var("INFR_IGNORE_EOS").is_ok();
    let hit_eos = |t: u32| !ignore_eos && (cfg.eos_ids.contains(&t) || t == cfg.eos);

    let prompt_tokens = model.encode(prompt)?;
    anyhow::ensure!(!prompt_tokens.is_empty(), "generate_mtp_spec: empty prompt");
    let p = prompt_tokens.len();

    let mut trunk_state: Option<crate::seam::SeamKv> = None;

    // ── prime: one VERIFY over the whole prompt, then catch the head up over it ──────────────
    let t_prime = std::time::Instant::now();
    let (ids0, h_rows0) = run_verify(
        be,
        bind,
        model.gguf(),
        cfg,
        model.token_embd(),
        &prompt_tokens,
        &mut trunk_state,
        max_ctx,
    )?;
    let prime_verify_secs = t_prime.elapsed().as_secs_f64();
    anyhow::ensure!(
        h_rows0.len() == p * ne,
        "prime verify: h_rows0 length {} != {p}*{ne}",
        h_rows0.len()
    );

    let mut shifted_h = vec![0f32; p * ne];
    if p > 1 {
        shifted_h[ne..].copy_from_slice(&h_rows0[..(p - 1) * ne]);
    }
    catch_up(head_sess, &prompt_tokens, &shifted_h, 0)?;

    let mut committed = prompt_tokens.clone();
    let mut id_last = prompt_tokens[p - 1];
    let mut pending_h = h_rows0[(p - 1) * ne..].to_vec();
    let mut n_past = p;

    // Cycle 1's virtual leading row (see this fn's doc) — `None` from cycle 2 on, once consumed.
    // Only the row's greedy ARGMAX ID is needed (the accept rule below never reads raw logits),
    // so the GPU-resident verify accept (issue #31) hands back just the ids.
    let mut leading_id: Option<u32> = Some(ids0[p - 1]);
    let mut leading_h: Option<Vec<f32>> = Some(pending_h.clone());

    let mut acc: Vec<u32> = Vec::new();
    let mut printed = 0usize;
    let mut out: Vec<u32> = Vec::new();
    let mut cycle = 0usize;
    let mut total_drafted = 0usize;
    let mut total_accepted = 0usize;
    // Phase 4 (issue #33): the SAME per-cycle sections `INFR_MTP_TIME`'s `eprintln!` below already
    // times, summed across the whole run — this is what `MtpTiming`'s return threads out.
    let mut sum_draft_secs = 0.0f64;
    let mut sum_verify_secs = 0.0f64;
    let mut sum_catchup_secs = 0.0f64;
    let t_decode = std::time::Instant::now();

    'cycles: while out.len() < max_new {
        let n_max_round = n_max.min(max_new - out.len());
        if n_max_round == 0 {
            break;
        }
        cycle += 1;

        let t_draft = std::time::Instant::now();
        let drafted = draft(
            head_sess,
            id_last,
            &pending_h,
            n_past,
            DEFAULT_P_MIN,
            n_max_round,
        )?;
        let draft_secs = t_draft.elapsed().as_secs_f64();
        let cand: Vec<u32> = drafted.iter().map(|&(id, _)| id).collect();

        let mut feed = committed.clone();
        feed.extend_from_slice(&cand);
        let t_verify = std::time::Instant::now();
        let (vids, h_rows) = run_verify(
            be,
            bind,
            model.gguf(),
            cfg,
            model.token_embd(),
            &feed,
            &mut trunk_state,
            max_ctx,
        )?;
        let verify_secs = t_verify.elapsed().as_secs_f64();
        let m = h_rows.len() / ne;

        let has_leading = leading_id.is_some();
        let varg: Vec<u32> = if let (Some(lid), Some(_)) = (leading_id, &leading_h) {
            debug_assert_eq!(m, cand.len(), "cycle 1 verify carries no leading row");
            std::iter::once(lid).chain(vids.iter().copied()).collect()
        } else {
            let base = m - (cand.len() + 1);
            vids[base..].to_vec()
        };

        let (accepted, next_tok) = crate::seam::model::spec_accept(&cand, &varg);
        total_drafted += cand.len();
        total_accepted += accepted;

        // The (accepted+1) rows of `h` to catch the head's KV up to — row 0 is the virtual/real
        // leading row, rows 1.. are this cycle's own verify output (see this fn's doc).
        let catchup_h_all: Vec<f32> = if has_leading {
            let mut v = leading_h.take().expect("has_leading implies Some");
            leading_id = None;
            v.extend_from_slice(&h_rows);
            v
        } else {
            let base = m - (cand.len() + 1);
            h_rows[base * ne..].to_vec()
        };
        let mut catchup_tokens: Vec<u32> = cand[..accepted].to_vec();
        catchup_tokens.push(next_tok);
        let catchup_h = &catchup_h_all[..(accepted + 1) * ne];

        let t_catchup = std::time::Instant::now();
        catch_up(head_sess, &catchup_tokens, catchup_h, n_past + 1)?;
        let catchup_secs = t_catchup.elapsed().as_secs_f64();
        pending_h = catchup_h[accepted * ne..].to_vec();
        sum_draft_secs += draft_secs;
        sum_verify_secs += verify_secs;
        sum_catchup_secs += catchup_secs;

        let (build_secs, exec_secs) = head_sess.take_timing();
        if time_mtp {
            eprintln!(
                "[mtp cycle {cycle}] drafted={} accepted={accepted} \
                 draft={:.1}ms verify={:.1}ms catchup={:.1}ms build={:.1}ms exec={:.1}ms",
                cand.len(),
                draft_secs * 1e3,
                verify_secs * 1e3,
                catchup_secs * 1e3,
                build_secs * 1e3,
                exec_secs * 1e3,
            );
        }

        let mut stop = false;
        for &c in &cand[..accepted] {
            let eos_hit = hit_eos(c);
            out.push(c);
            committed.push(c);
            if !eos_hit {
                crate::stream_token(model.tokenizer(), &mut acc, &mut printed, c, &mut on_piece);
            }
            if eos_hit || out.len() >= max_new {
                stop = true;
                break;
            }
        }
        if !stop {
            let eos_hit = hit_eos(next_tok);
            out.push(next_tok);
            committed.push(next_tok);
            if !eos_hit {
                crate::stream_token(
                    model.tokenizer(),
                    &mut acc,
                    &mut printed,
                    next_tok,
                    &mut on_piece,
                );
            }
            if eos_hit || out.len() >= max_new {
                stop = true;
            }
        }
        if stop {
            break 'cycles;
        }

        id_last = next_tok;
        n_past = committed.len();
    }

    let timing = MtpTiming {
        draft_secs: sum_draft_secs,
        verify_secs: sum_verify_secs,
        catchup_secs: sum_catchup_secs,
        total_drafted,
        total_accepted,
    };
    if time_mtp {
        let (dp, vp, cp) = timing.phase_shares();
        eprintln!(
            "[mtp summary] {cycle} cycles, {total_accepted}/{total_drafted} accepted (alpha={:.3}), {} tokens generated, phase share: draft {dp:.0}% verify {vp:.0}% catchup {cp:.0}%",
            timing.alpha(),
            out.len()
        );
    }

    Ok((
        crate::GenStats {
            n_prompt: p,
            prompt_secs: prime_verify_secs,
            n_gen: out.len(),
            decode_secs: t_decode.elapsed().as_secs_f64(),
        },
        timing,
    ))
}
