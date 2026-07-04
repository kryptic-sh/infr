//! Vulkan adapter for the agnostic `infr_core::Backend` seam: lower a `Graph` of composite ops onto
//! the existing fused Recorder kernels, recorded into one command buffer. Mirrors the CPU
//! interpreter (`infr-llama::cpu_backend`) op-for-op, but executes on the GPU — so the SAME model
//! `Graph` runs on either backend. Built incrementally; ops not yet mapped return an error.

use crate::linear::native_dense_supported;
use crate::recorder::Recorder;
use crate::{be, VulkanBackend};
use infr_core::backend::{Bindings, Buffer, BufferUsage, Plan};
use infr_core::error::Result;
use infr_core::graph::{Activation, AttnMask, Graph, Op, TensorKind};
use infr_core::{Backend, TensorId};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use crate::recorder::RecordedCmd;

/// Vulkan-specific compiled plan. Carries the cloned graph (like the old `GraphPlan`) plus, for an
/// eligible single-token decode graph, a lazily-built record-once replay (see [`DecodeReplay`]). An
/// ineligible graph (prefill batch, gemma/E2B/MoE/qwen35 decode, …) re-records every `execute` — the
/// unchanged static path.
pub(crate) struct VkDecodePlan {
    pub graph: Graph,
    /// True iff the graph is a qwen3-style decode the params-driven `_dyn` kernels can replay (see
    /// [`decode_eligible`]). Computed once at compile time.
    eligible: bool,
    /// Built on the first eligible `execute` and replayed on every one after (interior mutability so
    /// `execute` can stay `&dyn Plan`). `None` until then; always `None` for an ineligible graph.
    replay: Mutex<Option<DecodeReplay>>,
}

impl VkDecodePlan {
    fn boxed(graph: &Graph) -> Box<dyn Plan> {
        Box::new(VkDecodePlan {
            graph: graph.clone(),
            eligible: decode_eligible(graph),
            replay: Mutex::new(None),
        })
    }
}

impl Plan for VkDecodePlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A recorded, replayable decode command buffer plus the persistent buffers its descriptor sets
/// reference. The recording reads `pos`/`kv_len` from `params` (the `_dyn` kernels) and the
/// `attention_kv_dyn` dispatch is fixed regardless of `kv_len`, so ONE recording stays valid across
/// the whole decode — only `params` (and the runner's per-token embedding/position uploads) change.
struct DecodeReplay {
    /// Persistent `Internal` scratch (activations), allocated ONCE. The recorded descriptor sets bind
    /// these, so re-allocating per token would leave the recording pointing at freed buffers.
    scratch: Vec<Option<Box<dyn Buffer>>>,
    /// `[pos, kv_len]` (u32×2) SSBO the `_dyn` kernels read; the only per-token device write the
    /// adapter itself makes.
    params: Box<dyn Buffer>,
    /// The `positions` Input tensor — `execute` downloads element 0 to learn `pos` for `params`.
    positions: TensorId,
    recorded: RecordedCmd,
    /// Transient GEMM/attention scratch the recording references. Empty for an eligible decode (all
    /// ops are GEMV/silu/dyn kernels with no transient), held anyway so a stray one can't dangle.
    _transient: Vec<Box<dyn Buffer>>,
}

pub(crate) fn compile(graph: &Graph) -> Result<Box<dyn Plan>> {
    // Eligibility (record-once replay vs per-execute static recording) is a pure function of the
    // graph shape, so decide it here, once. `execute` builds the replay lazily on first run.
    Ok(VkDecodePlan::boxed(graph))
}

/// Record-once replay applies ONLY to a qwen3-style single-token decode graph the `_dyn` kernels
/// cover: every `Attention` is `rows==1`, `Causal`, scale `1/√head_dim`; RoPE rides on `QkNormRope`
/// (no standalone `Rope`, no `freq_factors`); and there is no `Softcap` / sliding window / `MoeFfn` /
/// `Conv1dSilu` / `DeltaNet` / per-head `QkNorm`. Anything else falls back to the static path — which
/// matters for correctness: `attention_kv_dyn` is gemma-disabled (full causal, hardcoded 1/√hd).
/// Standard-GGUF-block low-bit KV quants that ride the dequant→f16 prepass path (not native reads).
fn is_kv_quant(dt: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(dt, Q4_0 | Q4_1 | Q5_0 | Q5_1 | Iq4Nl)
}

/// Dense non-f16 KV caches (f32/bf16): stored via a cast-store, read via a cast→f16 prepass.
fn is_kv_dense_alt(dt: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(dt, F32 | Bf16)
}

/// TurboQuant KV caches (WHT-rotated): quantizing WriteKv + dequant→f16 prepass.
fn is_turbo(dt: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(dt, Turbo2 | Turbo3 | Turbo4)
}

/// Any KV cache dtype that rides the dequant/cast → f16 prepass (not f16 or native-Q8).
fn is_kv_prepass(dt: infr_core::DType) -> bool {
    is_kv_quant(dt) || is_kv_dense_alt(dt) || is_turbo(dt)
}

fn decode_eligible(graph: &Graph) -> bool {
    // INFR_SEAM_NO_REPLAY forces the static per-execute path (INFR_PROF2 timestamps work there;
    // the replay path can't report them).
    if std::env::var("INFR_SEAM_NO_REPLAY").is_ok() {
        return false;
    }
    let mut has_rope = false;
    let mut has_attn = false;
    for op in &graph.ops {
        match op {
            // Any mask (SWA windows ride push constants + the window-aware prologue) and any
            // scale (gemma4 uses 1.0) — both are baked per-layer into the recorded dispatch.
            // hd%4 ≤ 512 keeps every layer on the self-chunking split path or the scalar
            // fallback, both of which take scale/window.
            Op::Attention {
                rows,
                head_dim,
                k_cache,
                v_cache,
                ..
            } => {
                has_attn = true;
                if *rows != 1 || *head_dim % 4 != 0 || *head_dim > 512 {
                    return false;
                }
                // A Q8_0 KV cache (either side) forces the per-execute STATIC decode: the record-once
                // replay of the un-fused Q8 K-write (store_q8_dyn) mis-decodes despite correct
                // kernels/barriers (the static path with the same kernels is bit-correct). TODO: fix
                // the replay and drop this so Q8 decode gets the record-once speedup too.
                if matches!(graph.desc(*k_cache).dtype, infr_core::DType::Q8_0)
                    || matches!(graph.desc(*v_cache).dtype, infr_core::DType::Q8_0)
                    || is_kv_prepass(graph.desc(*k_cache).dtype)
                    || is_kv_prepass(graph.desc(*v_cache).dtype)
                {
                    return false;
                }
            }
            // freq_factors (gemma4 proportional RoPE) binds via qk_norm_rope_dyn_ff.
            Op::QkNormRope { .. } => has_rope = true,
            // Standalone llama Rope replays via rope_f16_dyn — needs the f16-out builder shape
            // (the f32 in-place form has no dyn kernel) and no freq_factors.
            Op::Rope {
                dst, freq_factors, ..
            } => {
                if freq_factors.is_some()
                    || !matches!(graph.desc(*dst).dtype, infr_core::DType::F16)
                {
                    return false;
                }
                has_rope = true;
            }
            // MoeFfn is REPLAY-SAFE: router GEMV + GPU-side top-k + id-indexed expert GEMVs are
            // all push-constant/pos-independent, and its scratch is plan-held. QkNorm (gemma4
            // V-norm) and Softcap are pos-independent elementwise — replay-safe as recorded.
            // Still rejected: Conv1dSilu/DeltaNet (recurrent state the contract doesn't cover).
            Op::MoeFfn { .. } | Op::QkNorm { .. } | Op::Softcap { .. } => {}
            Op::Conv1dSilu { .. } | Op::DeltaNet { .. } => return false,
            _ => {}
        }
    }
    has_rope && has_attn
}

/// Read `positions[0]` for the record-once decode. The runner writes the position into a host-visible
/// (CpuToGpu) `Staging` buffer via a mapped memcpy each token and no GPU op writes it, so read it
/// straight from the mapping — zero submit/sync. (The generic `download` on a CpuToGpu buffer would
/// allocate a staging buffer + one_shot copy + `queue_wait_idle` EVERY token; on the hot decode loop
/// that per-token full sync is the dominant residual cost.) Falls back to `download` if unmapped.
fn read_pos0(be_: &VulkanBackend, buf: &dyn Buffer) -> Result<u32> {
    if let Some(vb) = buf.as_any().downcast_ref::<crate::VkBuffer>() {
        if let Some(ptr) = vb.mapped_ptr() {
            let mut b = [0u8; 4];
            // Safety: the buffer is ≥4 bytes (positions has ≥1 i32) and persistently mapped.
            unsafe { std::ptr::copy_nonoverlapping(ptr as *const u8, b.as_mut_ptr(), 4) };
            return Ok(i32::from_le_bytes(b) as u32);
        }
    }
    let mut b = [0u8; 4];
    be_.download(buf, &mut b)?;
    Ok(i32::from_le_bytes(b) as u32)
}

/// Resolve a graph tensor to its device buffer: `Internal` from the scratch, everything else
/// (`Input`/`Weight`/`Output`) from the model-provided `Bindings`.
fn resolve<'a>(
    scratch: &'a [Option<Box<dyn Buffer>>],
    bindings: &'a Bindings,
    id: TensorId,
) -> Result<&'a dyn Buffer> {
    match &scratch[id.0 as usize] {
        Some(b) => Ok(b.as_ref()),
        None => bindings
            .get(id)
            .ok_or_else(|| be(format!("vulkan adapter: unbound tensor {}", id.0))),
    }
}

/// Allocate the `Internal` scratch (activations) for `graph`. The leading (row) dim is padded to a
/// multiple of 64 so the prefill GEMM / flash kernels — which write ceil(rows/64)*64 output rows —
/// write DIRECTLY into these buffers (no padded temp + copy). Padding rows are never read (downstream
/// ops touch only the real `rows`, and row-major layout keeps element (r<rows, c) at the same index).
fn alloc_scratch(be_: &VulkanBackend, graph: &Graph) -> Result<Vec<Option<Box<dyn Buffer>>>> {
    let mut scratch: Vec<Option<Box<dyn Buffer>>> =
        (0..graph.tensors.len()).map(|_| None).collect();
    for (i, decl) in graph.tensors.iter().enumerate() {
        if matches!(decl.kind, TensorKind::Internal) {
            let numel = decl.desc.numel();
            let padded = match decl.desc.shape.first() {
                Some(&rows) if rows > 0 => rows.div_ceil(64) * 64 * (numel / rows),
                _ => numel,
            };
            let bytes = decl
                .desc
                .dtype
                .dense_bytes(padded)
                .ok_or_else(|| be("vulkan adapter: internal tensor must be a dense dtype"))?;
            scratch[i] = Some(be_.alloc(bytes.max(4), BufferUsage::Activations)?);
        }
    }
    Ok(scratch)
}

/// Peephole: fuse `QkNormRope(k → k16)` + `WriteKv(k16 → cache, pos)` into a single Qk-norm+RoPE that
/// writes directly into the KV cache with the row offset baked in (matches the production Recorder).
/// This drops the intermediate f16 scratch, the copy dispatch, and one pipeline barrier per layer.
///
/// The intermediate tensor (k16) has ONE TensorId reused across ALL layers' QkNormRope ops, so the
/// map is keyed by OP INDEX (not TensorId): each layer's pair maps to its own KV-cache buffer.
/// Returns (fused: op index of the QkNormRope → (cache, row offset `pos`); skip: absorbed WriteKv ops).
/// Per-execute transient-scratch pool: the SAME (tag, bytes) key across ops returns the SAME
/// buffer. Layers are serialized by dataflow and the recorder's hazard tracking turns each
/// rewrite into an ordinary WAR/WAW barrier (the bespoke path shares its split scratch across
/// layers the same way), so per-tag reuse is safe — and it cuts the held transient VRAM from
/// O(n_layer) to O(1) buffers per tag. Without it, an 8B p8000 prefill held 36 layers × ~1GB of
/// flash-attention partials (≈38 GB) and took the device down.
type ScratchPool = HashMap<(&'static str, usize), Box<dyn Buffer>>;

/// Get-or-alloc the pool buffer for (tag, bytes); returns the map key so callers can hold several
/// pool buffers at once via immutable indexing (`pool[&k].as_ref()`).
fn pooled(
    pool: &mut ScratchPool,
    be_: &VulkanBackend,
    tag: &'static str,
    bytes: usize,
) -> Result<(&'static str, usize)> {
    let key = (tag, bytes.max(4));
    if let std::collections::hash_map::Entry::Vacant(e) = pool.entry(key) {
        // alloc_uninit: every pooled buffer is fully written before read within each op.
        e.insert(be_.alloc_uninit(key.1, BufferUsage::Activations)?);
    }
    Ok(key)
}

/// Per-execute Dynamic-attention shared state: ONE attn_live prologue + ONE pm/pl/pacc split
/// scratch set per distinct (nh, hd, chunk, n_chunks, window) attention shape (see `lower_op`'s
/// `dyn_args`). Uniform models (qwen3) have exactly one; gemma alternates SWA/global layers (and
/// gemma4-12b alternates hd 256/512), so the contexts live in a small Vec looked up by key —
/// same-key layers share one prologue dispatch and one scratch set.
struct DynAttnCtx {
    nh: usize,
    chunk: usize,
    n_chunks: usize,
    hd: usize,
    window: usize,
    args: Box<dyn Buffer>,
    pm: Box<dyn Buffer>,
    pl: Box<dyn Buffer>,
    pacc: Box<dyn Buffer>,
}

/// Fuse `Linear (m==1, native/f16 weight, Internal dst) → Add(residual)` into the fused-residual
/// GEMV (`linear_add_native` / `linear_add`) — the bespoke decode path's `o_or_down` shape (one
/// dispatch + barrier instead of two, and no round-trip of the sublayer output). Keyed by the
/// Linear's op index → (residual, final dst); the absorbed Add lands in the skip set. Only the
/// IMMEDIATELY following Add fuses (the seam builder emits the pair adjacent for non-gemma models;
/// gemma's sandwich norm sits between and correctly blocks the fusion).
fn linear_add_peephole(graph: &Graph) -> (HashMap<usize, (TensorId, TensorId)>, HashSet<usize>) {
    let mut fused: HashMap<usize, (TensorId, TensorId)> = HashMap::new();
    let mut skip: HashSet<usize> = HashSet::new();
    if std::env::var("INFR_NO_FUSE_ADD").is_ok() {
        return (fused, skip);
    }
    for (i, op) in graph.ops.iter().enumerate() {
        let Op::Linear {
            dst,
            m: 1,
            weight,
            out_f,
            ..
        } = op
        else {
            continue;
        };
        if !matches!(graph.tensors[dst.0 as usize].kind, TensorKind::Internal) {
            continue;
        }
        let dt = graph.desc(*weight).dtype;
        if !(native_dense_supported(dt) || matches!(dt, infr_core::DType::F16)) {
            continue;
        }
        if let Some(Op::Add {
            a,
            b,
            dst: add_dst,
            n,
        }) = graph.ops.get(i + 1)
        {
            if *n != *out_f {
                continue;
            }
            let residual = if b == dst && a != dst {
                *a
            } else if a == dst && b != dst {
                *b
            } else {
                continue;
            };
            fused.insert(i, (residual, *add_dst));
            skip.insert(i + 1);
        }
    }
    (fused, skip)
}

fn kv_write_peephole(graph: &Graph) -> (HashMap<usize, (TensorId, usize)>, HashSet<usize>) {
    let mut fused: HashMap<usize, (TensorId, usize)> = HashMap::new();
    let mut skip: HashSet<usize> = HashSet::new();
    for (i, op) in graph.ops.iter().enumerate() {
        // QkNormRope (qwen/gemma) or f16-out Rope (llama) — both write the f16 K row the peephole
        // can redirect straight into the KV cache.
        let kxx = match op {
            Op::QkNormRope { dst, .. } | Op::Rope { dst, .. } => dst,
            _ => continue,
        };
        // Only fuse an Internal (scratch) dst (we redirect the write into the KV cache). The
        // output must be f16 (the shader casts f32→f16); WriteKv of an f16 src is a plain copy.
        if !matches!(graph.tensors[kxx.0 as usize].kind, TensorKind::Internal) {
            continue;
        }
        if !matches!(graph.desc(*kxx).dtype, infr_core::DType::F16) {
            continue;
        }
        if let Some(Op::WriteKv {
            src, cache, pos, ..
        }) = graph.ops.get(i + 1)
        {
            // A Q8_0 cache needs a real quantizing WriteKv (store_q8), so DON'T fuse the f16 rope
            // write into it — leave the standalone WriteKv to run.
            if src == kxx && matches!(graph.desc(*cache).dtype, infr_core::DType::F16) {
                fused.insert(i, (*cache, *pos as usize));
                skip.insert(i + 1);
            }
        }
    }
    (fused, skip)
}

/// How pos-dependent ops (`QkNormRope`, `WriteKv`, `Attention`) are lowered:
/// - `Static`: the classic per-execute recording — the pos is a push constant (read from the
///   `positions` tensor up front into `rope_pos`), the graph's baked `pos`/`kv_len` drive the rest.
/// - `Dynamic`: the record-once path — the `_dyn` kernels read pos/kv_len from the `params` SSBO, so
///   the SAME recording replays across tokens. Only reachable for a [`decode_eligible`] graph.
enum RopeMode<'a> {
    Static(&'a HashMap<u32, usize>),
    Dynamic(&'a dyn Buffer),
}

/// Lower ONE graph op into the recorder. Shared by the static (`execute_static`) and record-once
/// (`record_decode_replay`) paths — only the three pos-dependent ops branch on `mode`.
#[allow(clippy::too_many_arguments)]
fn lower_op(
    be_: &VulkanBackend,
    graph: &Graph,
    op_idx: usize,
    op: &Op,
    rec: &Recorder<'_>,
    scratch: &[Option<Box<dyn Buffer>>],
    bindings: &Bindings,
    fused_kv_write: &HashMap<usize, (TensorId, usize)>,
    fused_add: &HashMap<usize, (TensorId, TensorId)>,
    mode: &RopeMode,
    transient: &mut Vec<Box<dyn Buffer>>,
    // Per-execute Dynamic-attention contexts: the prologue args AND the pm/pl/pacc split scratch,
    // shared by every attention op of the same shape (layers are serialized by dataflow, so one
    // set per shape suffices — the bespoke path shares its split scratch the same way; per-layer
    // sets cost 28x the VRAM and added run-to-run placement variance). A uniform model (qwen3)
    // holds one entry; gemma's SWA/global alternation (and gemma4-12b's hd 256/512) holds a few.
    dyn_args: &mut Vec<DynAttnCtx>,
    pool: &mut ScratchPool,
    dummy: &dyn Buffer,
) -> Result<()> {
    let r = |id: TensorId| resolve(scratch, bindings, id);
    match op {
        Op::RmsNorm {
            x,
            weight,
            dst,
            rows,
            dim,
            eps,
        } => {
            rec.rmsnorm(
                r(*x)?,
                r(*weight)?,
                r(*dst)?,
                *rows as usize,
                *dim as usize,
                *eps,
            );
        }
        // `dst = x · Wᵀ` — dispatch by weight dtype AND row count. Decode (m=1) uses the native
        // GEMV (or f16 GEMV). Prefill (m>1) with a native-quant weight uses the TILED coopmat GEMM
        // `matmul_native` (decode each weight element ONCE, reuse across the 64-row tile) instead
        // of the GEMV (which re-reads every weight row per output row) — the prefill perf win.
        Op::Linear {
            x,
            weight,
            dst,
            m,
            in_f,
            out_f,
            w_off,
        } => {
            let (m, in_f, out_f, w_off) = (
                *m as usize,
                *in_f as usize,
                *out_f as usize,
                *w_off as usize,
            );
            let (w, xb, y) = (r(*weight)?, r(*x)?, r(*dst)?);
            let dt = graph.desc(*weight).dtype;
            // `w_off` (fused-QKV slices) only rides the offset-capable native paths — the runner
            // gates fusion on `native_dense_supported`, so the f16/f32 fallbacks never see it.
            if w_off != 0 && !native_dense_supported(dt) {
                return Err(be("vulkan adapter: Linear w_off on a non-native weight"));
            }
            // Fused Linear+Add (decode residual): one GEMV with the residual added in-kernel —
            // see linear_add_peephole. `y` (the Linear's scratch dst) is never written.
            if let Some((residual, final_dst)) = fused_add.get(&op_idx) {
                if w_off != 0 {
                    return Err(be("vulkan adapter: Linear w_off with fused residual"));
                }
                let (rr, yf) = (r(*residual)?, r(*final_dst)?);
                if native_dense_supported(dt) {
                    rec.linear_add_native(dt, w, xb, rr, yf, m, in_f, out_f);
                } else {
                    rec.linear_add(w, xb, rr, yf, m, in_f, out_f);
                }
                return Ok(());
            }
            // Prefill (m>1): a TILED GEMM writes ceil(m/64)*64 rows DIRECTLY into `y` (Internal
            // buffers are row-padded to 64 up front, so no temp/copy). Needs n%64==0, k%32==0.
            //  • Q4_K → mmq (dp4a int8): quantize activations once, integer matmul on the raw
            //    blocks (no per-GEMM weight dequant) — the u4 prefill default.
            //  • other native quants → coopmat `matmul_native` (in-shader dequant).
            //  • f16 (float weights are uploaded as f16) → f16 coopmat `matmul_proj`.
            // Decode (m=1) and non-tileable shapes fall through to the GEMV.
            // Small multi-row batches (m = 2..8: spec-decode verify rows, a short chat-turn
            // suffix prefill) take the multi-row GEMV first — the single-M-tile coopmat GEMM
            // launches only n/64 workgroups (underfills the GPU: measured 51-182 GB/s effective
            // weight stream vs the GEMV class's 292-651 on a 7900 XTX), and the plain GEMV
            // re-streams the weight per row. 7900 XTX, 8B Q4_K shapes: m=2 is 5.6-8.3x the GEMM
            // route, m=4 1.4-4.1x, m=8 1.7-2.6x — EXCEPT very wide n at m>=5 (gate+up n=24576:
            // 384 tiles fill the GPU and the mrow's per-thread m*32-FMA inner loop turns
            // ALU-bound), so m=5..8 gates on out_f <= 8192 (<=128 tiles = the GEMM-underfill
            // regime). Writes exactly m rows (no padded-dst dance). Formats without an mrow
            // build fall through to the GEMM; INFR_NO_MROW forces the old route (A/B).
            if ((2..=4).contains(&m) || ((5..=8).contains(&m) && out_f <= 8192))
                && in_f % 32 == 0
                && crate::gemm::native_mrow_build_spv(dt).is_some()
                && std::env::var("INFR_NO_MROW").is_err()
            {
                rec.linear_native_mrow(dt, w, w_off, xb, y, m, in_f, out_f);
                return Ok(());
            }
            let gemm_ok = m > 1 && out_f % 64 == 0 && in_f % 32 == 0;
            let is_gemm =
                gemm_ok && (native_dense_supported(dt) || matches!(dt, infr_core::DType::F16));
            if is_gemm {
                // GEMM writes ceil(m/64)*64 rows. Internal `dst` is row-padded → write direct;
                // a non-Internal dst (e.g. the lm_head `logits` Output, unpadded) gets a padded
                // temp + copy of the m real rows.
                let dst_internal =
                    matches!(graph.tensors[dst.0 as usize].kind, TensorKind::Internal);
                let eb = graph.desc(*dst).dtype.dense_bytes(1).unwrap_or(4);
                let tmp = if dst_internal {
                    None
                } else {
                    let mpad = m.div_ceil(64) * 64;
                    // alloc_uninit: the GEMM writes all mpad rows before the copy reads m of them.
                    Some(be_.alloc_uninit((mpad * out_f * eb).max(4), BufferUsage::Activations)?)
                };
                let out: &dyn Buffer = match &tmp {
                    Some(t) => t.as_ref(),
                    None => y,
                };
                // Q4_K: the 8-warp warptile coopmat GEMM (native_gemm_warp, in-shader dequant)
                // beats mmq dp4a at prefill shapes (161 vs 417µs on [512,1024]×[1024,6144] — the
                // wide tile amortizes staging; RADV can't use int8 WMMA). mmq stays for shapes NO
                // warp tile can cover — n%128≠0 (the NARROW_N tile covers n%128; gemma3-1b's
                // ne=1152 projections sat on scalar mmq at ~10 TF under the old n%256 gate).
                // INFR_NO_MMQ also skips mmq for A/B.
                let warp_ok = out_f % 128 == 0
                    && crate::gemm::native_gemm_warp_build_spv(dt).is_some()
                    && std::env::var("INFR_NO_GEMM_WARP").is_err();
                if matches!(dt, infr_core::DType::Q4K)
                    && !warp_ok
                    && std::env::var("INFR_NO_MMQ").is_err()
                {
                    // mmq (dp4a int8): quantize activations once, integer matmul on raw blocks.
                    // Scratch is pooled — every same-shape Linear in the graph reuses one set.
                    let nblk = in_f / 32;
                    let qa = pooled(pool, be_, "mmq_qa", m * in_f)?;
                    let dact = pooled(pool, be_, "mmq_dact", m * nblk * 2)?;
                    let sact = pooled(pool, be_, "mmq_sact", m * nblk * 2)?;
                    rec.quant_q8(
                        xb,
                        pool[&qa].as_ref(),
                        pool[&dact].as_ref(),
                        pool[&sact].as_ref(),
                        m,
                        in_f,
                    );
                    rec.matmul_mmq_q4k(
                        pool[&qa].as_ref(),
                        pool[&dact].as_ref(),
                        pool[&sact].as_ref(),
                        w,
                        w_off,
                        out,
                        m,
                        in_f,
                        out_f,
                    );
                } else if native_dense_supported(dt) {
                    // A_GLOBAL: convert A to f16 ONCE (a cheap cast-copy — the warp kernels
                    // rounded A to f16 in their staging loop anyway, so numerics are identical)
                    // and let the warptiles coopMatLoad it straight from global. Dropping the As
                    // stage shrinks LDS to Bs-only → occupancy 2→3 wgs/WGP → ~1.5x on the 8B
                    // prefill shapes (o proj 29→44 TF). Pool pad rows may hold stale garbage;
                    // GEMM rows are independent, and dst pad rows are never read.
                    let use_ag = out_f % 128 == 0
                        && in_f % 32 == 0
                        && crate::gemm::native_gemm_warp_ag_build_spv(dt).is_some()
                        && std::env::var("INFR_NO_GEMM_WARP").is_err();
                    let a16 = if use_ag {
                        let mpad = m.div_ceil(64) * 64;
                        let key = pooled(pool, be_, "lin_a16", mpad * in_f * 2)?;
                        rec.store_f16(xb, pool[&key].as_ref(), m * in_f, 0);
                        Some(key)
                    } else {
                        None
                    };
                    // SPLIT-K for narrow-n deep-k shapes (o/down projections): the plain tile's
                    // grid underfills the device (n=1024 → 64 wgs on a 96-wg part → 6-11 TFLOPS
                    // vs the kernel's ~36). Split k across enough extra workgroups to fill, with
                    // a fixed-order (deterministic) reduce. Wide/filled shapes keep the direct
                    // path (no partials round-trip).
                    let narrow_grid = m.div_ceil(64) * (out_f / 128).max(1);
                    let splits = if out_f % 128 == 0 && in_f >= 1024 && narrow_grid < 128 {
                        (256 / narrow_grid).next_power_of_two().clamp(1, 8)
                    } else {
                        1
                    };
                    if splits > 1 && crate::gemm::native_gemm_warp_sk_build_spv(dt).is_some() {
                        let mpad = m.div_ceil(64) * 64;
                        let pk = pooled(pool, be_, "splitk_part", splits * mpad * out_f * 4)?;
                        let a: &dyn Buffer = match &a16 {
                            Some(k16) => pool[k16].as_ref(),
                            None => xb,
                        };
                        rec.matmul_native_splitk(
                            dt,
                            a,
                            w,
                            w_off,
                            pool[&pk].as_ref(),
                            out,
                            m,
                            in_f,
                            out_f,
                            splits,
                            a16.is_some(),
                        );
                    } else if let Some(k16) = &a16 {
                        rec.matmul_native_f16a(
                            dt,
                            pool[k16].as_ref(),
                            w,
                            w_off,
                            out,
                            m,
                            in_f,
                            out_f,
                        );
                    } else {
                        rec.matmul_native_off(dt, xb, w, w_off, out, m, in_f, out_f);
                    }
                } else {
                    // f16 coopmat GEMM (dummy scales/mins unused at bits=16).
                    rec.matmul_proj(xb, w, dummy, dummy, out, m, in_f, out_f, 16, 0);
                }
                if let Some(t) = tmp {
                    rec.copy(t.as_ref(), 0, y, 0, m * out_f * eb);
                    transient.push(t);
                }
            } else if native_dense_supported(dt) {
                rec.linear_native_off(dt, w, w_off, xb, y, m, in_f, out_f);
            } else if matches!(dt, infr_core::DType::F32) {
                // Full-precision projection weight (gemma4 E2B per-layer inp_gate/proj): the seam
                // uploads native dtype, and the f16 GEMV would read the f32 bytes as f16 garbage.
                rec.linear_f32(w, xb, y, m, in_f, out_f);
            } else {
                rec.linear(w, xb, y, m, in_f, out_f);
            }
        }
        Op::Add { a, b, dst, n } => rec.add(r(*a)?, r(*b)?, r(*dst)?, *n as usize),
        Op::AddBias {
            x,
            bias,
            dst,
            rows,
            n,
        } => rec.add_bias(r(*x)?, r(*bias)?, r(*dst)?, *rows as usize, *n as usize),
        Op::Scale { x, dst, s, n } => {
            let n = *n as usize;
            // recorder `scale` is in place on its buffer; copy x→dst first if they differ.
            if x != dst {
                let eb = graph.desc(*dst).dtype.dense_bytes(1).unwrap_or(4);
                rec.copy(r(*x)?, 0, r(*dst)?, 0, n * eb);
            }
            rec.scale(r(*dst)?, *s, n);
        }
        Op::Copy {
            src,
            src_off,
            dst,
            dst_off,
            n,
        } => {
            // IR offsets/counts are in ELEMENTS; the recorder copy takes BYTES.
            let eb = graph.desc(*src).dtype.dense_bytes(1).unwrap_or(4);
            rec.copy(
                r(*src)?,
                *src_off as usize * eb,
                r(*dst)?,
                *dst_off as usize * eb,
                *n as usize * eb,
            );
        }
        // Batched strided copy = `rows` buffer-copy regions (cheap vkCmdCopyBuffer). Splits a
        // batched interleaved buffer (conv q|k|v) into packed per-row slices.
        Op::CopyStrided {
            src,
            src_off,
            src_stride,
            dst,
            dst_off,
            dst_stride,
            rows,
            n,
        } => {
            let eb = graph.desc(*src).dtype.dense_bytes(1).unwrap_or(4);
            let (rows_, n_) = (*rows as usize, *n as usize);
            let (so, ss, do_, ds) = (
                *src_off as usize,
                *src_stride as usize,
                *dst_off as usize,
                *dst_stride as usize,
            );
            // One compute dispatch when everything is u32-word aligned (f32 always; f16 when the
            // element counts are even) — the per-row copy loop recorded `rows` vkCmdCopyBuffer +
            // hazard checks per split op (thousands per prefill chunk), dwarfing the bytes moved.
            let word_ok = [so * eb, ss * eb, do_ * eb, ds * eb, n_ * eb]
                .iter()
                .all(|b| b % 4 == 0);
            if word_ok {
                rec.copy_strided(
                    r(*src)?,
                    r(*dst)?,
                    rows_,
                    n_ * eb / 4,
                    so * eb / 4,
                    ss * eb / 4,
                    do_ * eb / 4,
                    ds * eb / 4,
                );
            } else {
                for row in 0..rows_ {
                    rec.copy(
                        r(*src)?,
                        (so + row * ss) * eb,
                        r(*dst)?,
                        (do_ + row * ds) * eb,
                        n_ * eb,
                    );
                }
            }
        }
        // Gated FFN activation: `act(gate) * up[+up_off]`. up_off (E2B per-layer slice) only
        // arises with Gelu (gemma); silu/sigmoid are always up_off==0.
        Op::GatedAct {
            gate,
            up,
            dst,
            rows,
            nff,
            act,
            up_off,
        } => {
            let n = *rows as usize * *nff as usize;
            let (g_, u_, y) = (r(*gate)?, r(*up)?, r(*dst)?);
            match act {
                Activation::Silu => {
                    if *up_off != 0 {
                        return Err(be("vulkan adapter: GatedAct Silu up_off!=0 unsupported"));
                    }
                    rec.silu_mul(g_, u_, y, n);
                }
                Activation::Sigmoid => {
                    if *up_off != 0 {
                        return Err(be("vulkan adapter: GatedAct Sigmoid up_off!=0 unsupported"));
                    }
                    // GatedAct semantics: `act(gate) * up` = sigmoid(gate) * up. mul_sigmoid computes
                    // `a * sigmoid(b)`, so pass (up, gate) — NOT (gate, up), which would sigmoid `up`.
                    rec.mul_sigmoid(u_, g_, y, n);
                }
                Activation::Gelu => {
                    let eb = graph.desc(*up).dtype.dense_bytes(1).unwrap_or(4);
                    rec.gelu_mul_off(g_, u_, *up_off as usize * eb, y, n);
                }
            }
        }
        // Combined [rows, 2*nff] gate|up buffer — the bespoke path's silu/gelu_mul_fused shape
        // (one GEMV/GEMM produced the whole gu, this reads both halves per row).
        Op::GatedActFused {
            gu,
            dst,
            rows,
            nff,
            act,
        } => {
            let (rows, nff) = (*rows as usize, *nff as usize);
            let (gu_, y) = (r(*gu)?, r(*dst)?);
            match act {
                Activation::Silu => rec.silu_mul_fused(gu_, y, rows, nff),
                Activation::Gelu => rec.gelu_mul_fused(gu_, y, rows, nff),
                Activation::Sigmoid => {
                    return Err(be("vulkan adapter: GatedActFused Sigmoid unsupported"))
                }
            }
        }
        // Append a row into the persistent KV cache at row `pos`. store_f16 casts f32→f16 (the
        // common case: V / f32 K); an already-f16 source is a straight copy. In Dynamic mode the
        // write offset (pos*n) comes from `params` instead of the baked `pos`.
        Op::WriteKv {
            src,
            cache,
            rows,
            row_stride,
            pos,
        } => {
            let (rows, rs, pos) = (*rows as usize, *row_stride as usize, *pos as usize);
            let n = rows * rs;
            let (s, c) = (r(*src)?, r(*cache)?);
            // Q8_0 cache: quantize the row(s) into 34 B/32-elem blocks. For a Q8 cache the K-rope
            // peephole is disabled, so the K WriteKv (f16 staging) reaches here alongside the f32 V.
            let cache_dt = graph.desc(*cache).dtype;
            let cache_q8 = matches!(cache_dt, infr_core::DType::Q8_0);
            let src_f16 = matches!(graph.desc(*src).dtype, infr_core::DType::F16);
            // Planar scales region begins at byte `cap` = total cache elements.
            let cap = graph.desc(*cache).numel();
            match mode {
                RopeMode::Static(_) if cache_q8 => rec.store_q8(s, c, n, pos * rs, cap, src_f16),
                RopeMode::Dynamic(params) if cache_q8 => {
                    rec.store_q8_dyn(s, *params, c, n, cap, src_f16)
                }
                // Mainline low-bit KV quants: quantize into standard GGUF blocks (static only — a
                // quantized KV cache forces static decode, so a Dynamic WriteKv never reaches here).
                RopeMode::Static(_) if is_kv_quant(cache_dt) => {
                    rec.quant_kv(cache_dt, s, c, n, pos * rs, src_f16)
                }
                // Dense f32/bf16 cache: cast-store the row (also static-only).
                RopeMode::Static(_) if is_kv_dense_alt(cache_dt) => {
                    rec.store_kv_dense(cache_dt, s, c, n, pos * rs, src_f16)
                }
                // TurboQuant cache: WHT-quantize the row (static-only).
                RopeMode::Static(_) if is_turbo(cache_dt) => {
                    rec.quant_turbo(cache_dt, s, c, n, pos * rs, src_f16)
                }
                RopeMode::Static(_) => match graph.desc(*src).dtype {
                    infr_core::DType::F16 => rec.copy(s, 0, c, pos * rs * 2, n * 2),
                    _ => rec.store_f16(s, c, n, pos * rs),
                },
                RopeMode::Dynamic(params) => match graph.desc(*src).dtype {
                    // The only f16 WriteKv (K) is always fused into the QkNormRope and skipped, so a
                    // standalone f16 WriteKv shouldn't reach the record-once path.
                    infr_core::DType::F16 => {
                        return Err(be("vulkan adapter: dynamic decode f16 WriteKv unexpected"))
                    }
                    _ => rec.store_f16_dyn(s, *params, c, n),
                },
            }
        }
        // Fused per-head RMSNorm + RoPE. Peephole (see `kv_write_peephole`): a QkNormRope whose dst
        // feeds an immediately-following WriteKv is redirected to write the KV cache directly at row
        // `pos` (its WriteKv is skipped). Static uses the pos push constant + `out_base=pos`; Dynamic
        // uses `qk_norm_rope_dyn` (pos from `params`, `out_base_mul` 1 for the cache write, 0 for Q).
        Op::QkNormRope {
            x,
            weight,
            positions,
            dst,
            rows,
            n_head,
            head_dim,
            rope_dim,
            theta,
            eps,
            freq_factors,
        } => {
            let (out_buf, fused) = if let Some(&(cache, pos)) = fused_kv_write.get(&op_idx) {
                (r(cache)?, Some(pos))
            } else {
                (r(*dst)?, None)
            };
            match mode {
                RopeMode::Static(rope_pos) => {
                    let ff = match freq_factors {
                        Some(f) => Some(r(*f)?),
                        None => None,
                    };
                    rec.qk_norm_rope(
                        r(*x)?,
                        r(*weight)?,
                        out_buf,
                        *rows as usize,
                        *n_head as usize,
                        *head_dim as usize,
                        *rope_dim as usize,
                        *theta,
                        rope_pos[&positions.0],
                        fused.unwrap_or(0),
                        *eps,
                        ff,
                    );
                }
                RopeMode::Dynamic(params) => {
                    // gemma4 proportional RoPE (full-attention layers): the ff divisors bind via
                    // the `qk_norm_rope_dyn_ff` variant — pos still comes from `params`.
                    let ff = match freq_factors {
                        Some(f) => Some(r(*f)?),
                        None => None,
                    };
                    // `out_base_mul` is the 0/1 multiplier the shader scales by pos (then internally
                    // by nheads*hd): 1 → write cache row pos, 0 → write row 0 of the Q scratch.
                    let out_base_mul = usize::from(fused.is_some());
                    rec.qk_norm_rope_dyn(
                        r(*x)?,
                        r(*weight)?,
                        *params,
                        ff,
                        out_buf,
                        *rows as usize,
                        *n_head as usize,
                        *head_dim as usize,
                        *rope_dim as usize,
                        *theta,
                        out_base_mul,
                        *eps,
                    );
                }
            }
        }
        // Standalone RoPE (llama: no q/k-norm; INTERLEAVED pairs — llama.cpp's ROPE_TYPE_NORM).
        // An f16 dst takes the same fused shapes as QkNormRope: the kv_write peephole redirects
        // the K write into the cache, and Dynamic mode replays via `rope_f16_dyn` (pos from
        // `params`). The legacy f32 in-place form stays static-only. No freq_factors kernel.
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
            if freq_factors.is_some() {
                return Err(be(
                    "vulkan adapter: standalone Rope with freq_factors unsupported",
                ));
            }
            let f16_out = matches!(graph.desc(*dst).dtype, infr_core::DType::F16);
            let (out_buf, fused) = if let Some(&(cache, pos)) = fused_kv_write.get(&op_idx) {
                (r(cache)?, Some(pos))
            } else {
                (r(*dst)?, None)
            };
            match mode {
                RopeMode::Static(rope_pos) => {
                    if f16_out {
                        rec.rope_f16(
                            r(*x)?,
                            out_buf,
                            *rows as usize,
                            *n_head as usize,
                            *head_dim as usize,
                            *rope_dim as usize,
                            *theta,
                            rope_pos[&positions.0],
                            fused.unwrap_or(0),
                        );
                    } else {
                        rec.rope(
                            r(*x)?,
                            r(*dst)?,
                            *rows as usize,
                            *n_head as usize,
                            *head_dim as usize,
                            *rope_dim as usize,
                            *theta,
                            rope_pos[&positions.0],
                        );
                    }
                }
                RopeMode::Dynamic(params) => {
                    if !f16_out {
                        return Err(be(
                            "vulkan adapter: dynamic decode f32 in-place Rope unsupported",
                        ));
                    }
                    rec.rope_f16_dyn(
                        r(*x)?,
                        *params,
                        out_buf,
                        *rows as usize,
                        *n_head as usize,
                        *head_dim as usize,
                        *rope_dim as usize,
                        *theta,
                        usize::from(fused.is_some()),
                    );
                }
            }
        }
        // GQA scaled-dot-product attention over the f16 KV cache. Dynamic mode (decode, rows==1,
        // causal, 1/√hd) uses `attention_kv_dyn` (pos_offset + kv_len from `params`). Static mode
        // keeps the full lowering: FlashAttention-2 for prefill, `attention_kv` otherwise.
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
            let (rows, kv_len, nh, nkv, hd, pos) = (
                *rows as usize,
                *kv_len as usize,
                *n_head as usize,
                *n_kv as usize,
                *head_dim as usize,
                *pos as usize,
            );
            // Planar Q8_0 KV cache, chosen PER-SIDE (K and V independent). The coalesced f16
            // flash/split kernels can't read Q8, so a Q8 side routes decode through the native-Q8
            // split (attn_partial_{k,v}q8) and prefill through a dequant→f16 prepass. A Q8 cache
            // forces STATIC decode (see decode_eligible), so the Dynamic branch below never sees Q8.
            let k_q8 = matches!(graph.desc(*k_cache).dtype, infr_core::DType::Q8_0);
            let v_q8 = matches!(graph.desc(*v_cache).dtype, infr_core::DType::Q8_0);
            let kv_q8 = k_q8 && v_q8; // coupled (Dynamic-branch kernels; unreachable for Q8)
                                      // Planar Q8 scales region base = total cache elements (K and V caches share numel).
            let cap = graph.desc(*k_cache).numel();
            if let RopeMode::Dynamic(params) = mode {
                // Eligibility guarantees rows==1. Scale rides a push constant (gemma4 uses 1.0;
                // 0.0 → kernel default 1/√hd) and SWA windows ride the window-aware prologue +
                // partial kernel, so gemma-family decode replays too.
                let window = match mask {
                    AttnMask::Causal => 0usize,
                    AttnMask::SlidingWindow(w) => *w,
                };
                let def_scale = (*scale - 1.0 / (hd as f32).sqrt()).abs() < 1e-6;
                let pscale = if def_scale { 0.0 } else { *scale };
                // Flash-decoding split-K: the scalar attention_kv_dyn launches only nh workgroups
                // and rescans the whole cache per token — decode slowed ~4x from kv 100→600 on the
                // seam replay path. `chunk`/`n_chunks` are push constants (dispatch structure), so
                // bake the WORST CASE from the bound cache's capacity: chunks past the live kv_len
                // (read from `params` at replay) produce zero-weight partials (m=-3e38, l=0) the
                // combine ignores. attn_combine's shared array caps n_chunks at 512 → scale chunk
                // with capacity. Scratch is plan-held (`transient`), so replay reuses it.
                let cap = graph.desc(*k_cache).numel() / (nkv * hd);
                // Baked MIN chunk (64, rising only when the capacity would exceed the 1024-chunk
                // scratch/combine cap); the kernel derives the effective adaptive chunk from the
                // live kv_len each token, so one recorded plan serves every depth.
                let chunk = cap.div_ceil(1024).max(64);
                let n_chunks = cap.div_ceil(chunk);
                if hd % 4 == 0 && hd <= 512 && n_chunks > 1 {
                    // ONE prologue + ONE scratch set per (nh, hd, chunk, n_chunks, window) key:
                    // the first Dynamic attention op of a shape records/allocates; every later
                    // same-shape layer reuses (kv_len is per-token, and the layers' attention ops
                    // are serialized by dataflow — pm/pl/pacc hazards are ordinary RAW/WAW
                    // barriers). Gemma alternates SWA/global (and gemma4-12b hd 256/512), so a
                    // handful of keys coexist per execute.
                    let key = |c: &DynAttnCtx| {
                        c.nh == nh
                            && c.chunk == chunk
                            && c.n_chunks == n_chunks
                            && c.hd == hd
                            && c.window == window
                    };
                    if !dyn_args.iter().any(key) {
                        let args = be_.alloc_uninit(16, BufferUsage::Activations)?;
                        rec.attn_live_prologue(*params, args.as_ref(), nh, chunk, window);
                        dyn_args.push(DynAttnCtx {
                            nh,
                            chunk,
                            n_chunks,
                            hd,
                            window,
                            args,
                            pm: be_.alloc_uninit(nh * n_chunks * 4, BufferUsage::Activations)?,
                            pl: be_.alloc_uninit(nh * n_chunks * 4, BufferUsage::Activations)?,
                            pacc: be_
                                .alloc_uninit(nh * n_chunks * hd * 4, BufferUsage::Activations)?,
                        });
                    }
                    let ctx = dyn_args.iter().find(|c| key(c)).unwrap();
                    rec.attention_kv_split_dynac(
                        r(*q)?,
                        r(*k_cache)?,
                        r(*v_cache)?,
                        r(*dst)?,
                        ctx.pm.as_ref(),
                        ctx.pl.as_ref(),
                        ctx.pacc.as_ref(),
                        *params,
                        ctx.args.as_ref(),
                        nh,
                        nkv,
                        hd,
                        chunk,
                        n_chunks,
                        pscale,
                        window,
                        kv_q8,
                        cap,
                    );
                } else {
                    rec.attention_kv_dyn(
                        r(*q)?,
                        r(*k_cache)?,
                        r(*v_cache)?,
                        *params,
                        r(*dst)?,
                        1,
                        nh,
                        nkv,
                        hd,
                        pscale,
                        window,
                        kv_q8,
                        cap,
                    );
                }
            } else {
                // Dequant a quantized KV side → a transient f16 scratch, then run the f16 attention on
                // it. Q8 only dequants on PREFILL (rows>1; decode reads Q8 natively via the coalesced
                // split/scalar kernels — the scalar O(kv²) path TDR-hangs at depth). The mainline
                // low-bit quants (q4_0/…/iq4_nl) have no native read, so they dequant on EVERY pass
                // (prefill + decode). The persistent cache stays quantized (footprint preserved).
                let kdt = graph.desc(*k_cache).dtype;
                let vdt = graph.desc(*v_cache).dtype;
                let deq_k = (k_q8 && rows > 1) || is_kv_prepass(kdt);
                let deq_v = (v_q8 && rows > 1) || is_kv_prepass(vdt);
                let ne = kv_len * nkv * hd;
                // Expand a KV side into the f16 scratch: native Q8 (planar), f32 cast (store_f16),
                // else the shared quant/bf16 dequant (dequant_kv_f16).
                let kc_key = if deq_k {
                    let k = pooled(pool, be_, "kvdeq_k", ne * 2)?;
                    let sc = pool[&k].as_ref();
                    if k_q8 {
                        rec.dequant_q8_f16(r(*k_cache)?, sc, ne, cap);
                    } else if matches!(kdt, infr_core::DType::F32) {
                        rec.store_f16(r(*k_cache)?, sc, ne, 0);
                    } else if is_turbo(kdt) {
                        rec.dequant_turbo_f16(kdt, r(*k_cache)?, sc, ne);
                    } else {
                        rec.dequant_kv_f16(kdt, r(*k_cache)?, sc, ne);
                    }
                    Some(k)
                } else {
                    None
                };
                let vc_key = if deq_v {
                    let k = pooled(pool, be_, "kvdeq_v", ne * 2)?;
                    let sc = pool[&k].as_ref();
                    if v_q8 {
                        rec.dequant_q8_f16(r(*v_cache)?, sc, ne, cap);
                    } else if matches!(vdt, infr_core::DType::F32) {
                        rec.store_f16(r(*v_cache)?, sc, ne, 0);
                    } else if is_turbo(vdt) {
                        rec.dequant_turbo_f16(vdt, r(*v_cache)?, sc, ne);
                    } else {
                        rec.dequant_kv_f16(vdt, r(*v_cache)?, sc, ne);
                    }
                    Some(k)
                } else {
                    None
                };
                // A dequanted side reads the f16 scratch; native Q8 read only when not dequanted.
                let k_q8_eff = k_q8 && !deq_k;
                let v_q8_eff = v_q8 && !deq_v;
                // Prefill (rows≥64), causal, hd==128, standard 1/√hd scale: FlashAttention-2 (split-K
                // online softmax, no materialized [m,kv] scores). The flash kernel hardcodes 1/√hd and
                // writes ceil(rows/64)*64 output rows, so guard the scale and copy the real rows.
                let flash_ok = rows >= 64
                    && hd == 128
                    && matches!(mask, AttnMask::Causal)
                    && (*scale - 1.0 / (hd as f32).sqrt()).abs() < 1e-6
                    && !(k_q8_eff || v_q8_eff);
                // Prefill at hd≠128 (qwen35/gemma hd=256): the non-FA coopmat pipeline
                // (attn_qk → softmax → attn_pv) is hd-general and ~an order faster than the scalar
                // attention_kv. Needs 64-row-padded q/dst (Internal buffers are row-padded), so
                // require both to be Internal.
                let nonfa_ok = !flash_ok
                    && rows >= 64
                    && hd % 64 == 0
                    && hd <= 512
                    && !(k_q8_eff || v_q8_eff)
                    && matches!(graph.tensors[q.0 as usize].kind, TensorKind::Internal)
                    && matches!(graph.tensors[dst.0 as usize].kind, TensorKind::Internal);
                // Decode (rows==1): flash-decoding split-K — each head's KV range splits across
                // ~32 chunks of workgroups instead of the scalar attention_kv's rows*nh (= nh at
                // decode, ~16 workgroups on a 96-CU GPU — the decode bottleneck). attn_partial
                // handles any hd%4==0 ≤ 512 (hd=128 fast path, general path above), any scale
                // (push constant; gemma4 uses 1.0), and SWA windows (chunks below the window
                // clamp to empty → zero-weight partials the combine skips).
                let chunk = (kv_len / 32).clamp(64, 512);
                let split_ok = rows == 1 && kv_len > chunk && hd % 4 == 0 && hd <= 512;
                if flash_ok {
                    let mpad = rows.div_ceil(64) * 64;
                    // Pooled split partials (fully written before the combine reads them) — one
                    // set serves every layer instead of n_layer live copies (~1GB each at 8B p8k).
                    let po = pooled(pool, be_, "flash_po", 8 * mpad * nh * hd * 4)?;
                    let pm = pooled(pool, be_, "flash_pm", 8 * mpad * nh * 4)?;
                    let pl = pooled(pool, be_, "flash_pl", 8 * mpad * nh * 4)?;
                    let kcb = match &kc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*k_cache)?,
                    };
                    let vcb = match &vc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*v_cache)?,
                    };
                    rec.attention_prefill_flash(
                        r(*q)?,
                        kcb,
                        vcb,
                        r(*dst)?,
                        pool[&po].as_ref(),
                        pool[&pm].as_ref(),
                        pool[&pl].as_ref(),
                        rows,
                        kv_len,
                        nh,
                        nkv,
                        hd,
                        pos,
                    );
                } else if nonfa_ok {
                    let window = match mask {
                        AttnMask::Causal => 0,
                        AttnMask::SlidingWindow(w) => *w,
                    };
                    let mpad = rows.div_ceil(64) * 64;
                    let kv_pad = kv_len.div_ceil(256) * 256;
                    // Pooled scores scratch [nh, mpad, kv_pad] f16 + split-K PV partials (≤8
                    // splits) f32 — ~80MB per attention op, fully written before read (attn_qk
                    // fills every [mpad, kv_pad] row; PV partials are written per split before
                    // the reduce), and one set serves every same-shape layer.
                    let s = pooled(pool, be_, "nonfa_s", nh * mpad * kv_pad * 2)?;
                    let pv = pooled(pool, be_, "nonfa_pv", 8 * mpad * nh * hd * 4)?;
                    let kcb = match &kc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*k_cache)?,
                    };
                    let vcb = match &vc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*v_cache)?,
                    };
                    rec.attention_prefill_nonfa(
                        r(*q)?,
                        kcb,
                        vcb,
                        r(*dst)?,
                        pool[&s].as_ref(),
                        pool[&pv].as_ref(),
                        rows,
                        kv_len,
                        nh,
                        nkv,
                        hd,
                        pos,
                        window,
                        *scale,
                    );
                } else if split_ok {
                    let window = match mask {
                        AttnMask::Causal => 0,
                        AttnMask::SlidingWindow(w) => *w,
                    };
                    let n_chunks = kv_len.div_ceil(chunk);
                    let pm = pooled(pool, be_, "split_pm", nh * n_chunks * 4)?;
                    let pl = pooled(pool, be_, "split_pl", nh * n_chunks * 4)?;
                    let pacc = pooled(pool, be_, "split_pacc", nh * n_chunks * hd * 4)?;
                    let kcb = match &kc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*k_cache)?,
                    };
                    let vcb = match &vc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*v_cache)?,
                    };
                    rec.attention_kv_split(
                        r(*q)?,
                        kcb,
                        vcb,
                        r(*dst)?,
                        pool[&pm].as_ref(),
                        pool[&pl].as_ref(),
                        pool[&pacc].as_ref(),
                        kv_len,
                        nh,
                        nkv,
                        hd,
                        chunk,
                        n_chunks,
                        *scale,
                        window,
                        k_q8_eff,
                        v_q8_eff,
                        cap,
                    );
                } else {
                    let window = match mask {
                        AttnMask::Causal => 0,
                        AttnMask::SlidingWindow(w) => *w,
                    };
                    let kcb = match &kc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*k_cache)?,
                    };
                    let vcb = match &vc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*v_cache)?,
                    };
                    rec.attention_kv(
                        r(*q)?,
                        kcb,
                        vcb,
                        r(*dst)?,
                        rows,
                        kv_len,
                        nh,
                        nkv,
                        hd,
                        pos,
                        window,
                        *scale,
                        k_q8_eff,
                        v_q8_eff,
                        cap,
                    );
                }
            }
        }
        // Per-head RMSNorm == rmsnorm over rows*n_head rows of head_dim (gemma4's weightless
        // V-norm passes a ones weight → out = x/rms).
        Op::QkNorm {
            x,
            weight,
            dst,
            rows,
            n_head,
            head_dim,
            eps,
        } => {
            rec.rmsnorm(
                r(*x)?,
                r(*weight)?,
                r(*dst)?,
                (*rows * *n_head) as usize,
                *head_dim as usize,
                *eps,
            );
        }
        // Qwen3-Next SSM: depthwise causal conv + SiLU (rolling conv `state` mutated in place).
        Op::Conv1dSilu {
            x,
            weight,
            state,
            dst,
            rows,
            channels,
            kernel,
        } => {
            // Batch (rows ≥ kconv-1): all rows·cc outputs in parallel + a history rebuild pass,
            // instead of the token-serial history walk. Decode keeps the sequential kernel.
            let cv = if *rows as usize >= (*kernel as usize).saturating_sub(1).max(2) {
                Recorder::conv1d_silu_batch
            } else {
                Recorder::conv1d_silu
            };
            cv(
                rec,
                r(*x)?,
                r(*weight)?,
                r(*state)?,
                r(*dst)?,
                *rows as usize,
                *channels as usize,
                *kernel as usize,
            );
        }
        // Qwen3-Next gated DeltaNet recurrence (persistent `state` S mutated in place).
        Op::DeltaNet {
            q,
            k,
            v,
            b,
            a,
            a_coef,
            dt_bias,
            state,
            dst,
            rows,
            n_vhead,
            n_khead,
            head_k,
            head_v,
            eps,
        } => {
            // Prefill (rows ≥ 32): the chunkwise delta rule processes 32 tokens per state
            // traversal (matmuls + a triangular solve) instead of the token-serial recurrence —
            // the difference between rows and rows/32 sequential state sweeps. The default is the
            // SPLIT form (parallel prep/gates passes + a light state-coupled scan; the monolithic
            // kernel duplicated the prep work per column block). Decode/short rows keep the
            // sequential kernel. INFR_NO_DN_CHUNK forces sequential; INFR_NO_DN_SPLIT keeps the
            // chunked math but in the monolithic kernel (A/B).
            let (rows_, nv_, nk_, kd_, vd_) = (
                *rows as usize,
                *n_vhead as usize,
                *n_khead as usize,
                *head_k as usize,
                *head_v as usize,
            );
            let chunked = rows_ >= 32 && std::env::var("INFR_NO_DN_CHUNK").is_err();
            if chunked && std::env::var("INFR_NO_DN_SPLIT").is_err() {
                let nchunk = rows_.div_ceil(32);
                // alloc_uninit: every slot the scan reads is written by prep/gates first.
                let kn =
                    be_.alloc_uninit((rows_ * nk_ * kd_ * 4).max(4), BufferUsage::Activations)?;
                let qn =
                    be_.alloc_uninit((rows_ * nk_ * kd_ * 4).max(4), BufferUsage::Activations)?;
                let dk =
                    be_.alloc_uninit((nchunk * nk_ * 1024 * 4).max(4), BufferUsage::Activations)?;
                let dq =
                    be_.alloc_uninit((nchunk * nk_ * 1024 * 4).max(4), BufferUsage::Activations)?;
                let bg =
                    be_.alloc_uninit((nchunk * nv_ * 32 * 4).max(4), BufferUsage::Activations)?;
                let gg =
                    be_.alloc_uninit((nchunk * nv_ * 32 * 4).max(4), BufferUsage::Activations)?;
                rec.deltanet_chunked_split(
                    r(*q)?,
                    r(*k)?,
                    r(*v)?,
                    r(*b)?,
                    r(*a)?,
                    r(*a_coef)?,
                    r(*dt_bias)?,
                    r(*state)?,
                    r(*dst)?,
                    kn.as_ref(),
                    qn.as_ref(),
                    dk.as_ref(),
                    dq.as_ref(),
                    bg.as_ref(),
                    gg.as_ref(),
                    rows_,
                    nv_,
                    nk_,
                    kd_,
                    vd_,
                    *eps,
                );
                transient.extend([kn, qn, dk, dq, bg, gg]);
            } else {
                let dn = if chunked {
                    Recorder::deltanet_chunked
                } else {
                    Recorder::deltanet
                };
                dn(
                    rec,
                    r(*q)?,
                    r(*k)?,
                    r(*v)?,
                    r(*b)?,
                    r(*a)?,
                    r(*a_coef)?,
                    r(*dt_bias)?,
                    r(*state)?,
                    r(*dst)?,
                    rows_,
                    nv_,
                    nk_,
                    kd_,
                    vd_,
                    *eps,
                );
            }
        }
        // Elementwise gemma logit softcap `y = cap·tanh(x/cap)` (in-place safe).
        Op::Softcap { x, dst, cap, n } => {
            rec.softcap(r(*x)?, r(*dst)?, *cap, *n as usize);
        }
        // MoE FFN (single token): router GEMV → GPU-resident top-k (softmax-renorm, ×scale) →
        // fused multi-slot expert SwiGLU (gate/up share the row, down reads each slot's act) →
        // weighted accumulate. Mirrors the production GPU-resident decode path (transformer.rs)
        // and the CPU `Op::MoeFfn` interpreter. Expert banks must use an id-native quant format.
        Op::MoeFfn {
            x,
            router,
            gate_exps,
            up_exps,
            down_exps,
            dst,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            scale,
            act,
        } => {
            let (ne, n_expert, n_used, nff) = (
                *ne as usize,
                *n_expert as usize,
                *n_used as usize,
                *n_ff_exp as usize,
            );
            let stride = nff * ne; // elements per expert (identical for gate/up/down banks)
            let (gdt, udt, ddt) = (
                graph.desc(*gate_exps).dtype,
                graph.desc(*up_exps).dtype,
                graph.desc(*down_exps).dtype,
            );
            if !(crate::linear::native_id_kernel_name(gdt).is_some()
                && crate::linear::native_id_kernel_name(udt).is_some()
                && crate::linear::native_id_kernel_name(ddt).is_some())
            {
                return Err(be(
                    "vulkan adapter: MoeFfn expert banks need an id-native quant format",
                ));
            }
            // ── BATCHED MoE FFN (rows > 1, the seam's prefill chunks): GPU-resident expert
            // routing (top-k → bucket count/scan/scatter, all on-GPU) + a prologue that writes
            // per-expert INDIRECT dispatch args from the counts — so the whole expert loop
            // records with NO host readback (the bespoke path downloads counts mid-graph to size
            // its GEMMs; indirect dispatch replaces that). Q4_K gate/up + Q6_K down only (what
            // qwen3moe ships) — the runner routes other formats through the per-token path.
            let rows = graph.desc(*x).numel() / ne;
            if rows > 1 {
                use infr_core::DType::{Q4K, Q6K};
                let down_q6 = matches!(ddt, Q6K);
                if !(matches!(gdt, Q4K)
                    && matches!(udt, Q4K)
                    && (down_q6 || matches!(ddt, Q4K))
                    && matches!(act, Activation::Silu))
                {
                    return Err(be(format!(
                        "vulkan adapter: batched MoeFfn needs Q4_K gate/up + Q4_K/Q6_K down + \
                         SiLU (got gate={gdt:?} up={udt:?} down={ddt:?} act={act:?})"
                    )));
                }
                let al = |n: usize| be_.alloc((n * 4).max(4), BufferUsage::Activations);
                let alu = |n: usize| be_.alloc_uninit(n.max(4), BufferUsage::Activations);
                let n_pairs = rows * n_used;
                let xb = r(*x)?;
                let yb = r(*dst)?;

                // ── SINGLE-DISPATCH-PER-STAGE pipeline over the PACKED bucket layout. The old
                // shape ran every stage per expert (8-way waves of indirect dispatches): ~1050
                // dispatches and ~110 barriers per layer, and at pp512 the launch/serialization
                // overhead — not GEMM math — was ~60% of MoE prefill GPU time (quant 25%,
                // gather+scatter+silu 34%). Instead every stage runs ONCE over all n_pairs
                // packed rows: the expert GEMMs put the expert id on gl_WorkGroupID.y
                // (segment = offsets[e]..+counts[e], worst-case row tiles exit immediately),
                // gather fuses into the activation quant, and a per-token reduce over `inv_pos`
                // (assignment → bucket slot, fixed slot order — deterministic, no atomics)
                // replaces the KWAY per-set dst copies + adds. ~13 dispatches per layer, no
                // indirect args, no host readback.
                //
                // The GEMM As stage reads up to 63 rows past a segment end (garbage, results
                // discarded) — pad the packed row dimension so the LAST expert's overread stays
                // in-bounds.
                let npad = n_pairs.div_ceil(64) * 64 + 64;
                let logits = alu(rows * n_expert * 4)?;
                let ids = alu(n_pairs * 4)?;
                let wts = alu(n_pairs * 4)?;
                let counts = al(n_expert)?; // zeroed below (bucket_count accumulates)
                let offsets = alu(n_expert * 4)?;
                let fill = alu(n_expert * 4)?;
                let bucket_rows = alu(n_pairs * 4)?;
                let bucket_wts = alu(n_pairs * 4)?;
                let inv_pos = alu(n_pairs * 4)?;

                // Packed scratch, POOLED — one set serves every MoE layer in the graph.
                let qa = pooled(pool, be_, "moe_qa", npad * ne)?;
                let qda = pooled(pool, be_, "moe_qda", npad * (ne / 32) * 2)?;
                let qsa = pooled(pool, be_, "moe_qsa", npad * (ne / 32) * 2)?;
                let ge = pooled(pool, be_, "moe_ge", npad * nff * 4)?;
                let ue = pooled(pool, be_, "moe_ue", npad * nff * 4)?;
                let ae = pooled(pool, be_, "moe_ae", npad * nff * 4)?;
                let dqa = pooled(pool, be_, "moe_dqa", npad * nff)?;
                let dda = pooled(pool, be_, "moe_dda", npad * (nff / 32) * 2)?;
                let dsa = pooled(pool, be_, "moe_dsa", npad * (nff / 32) * 2)?;
                let ye = pooled(pool, be_, "moe_ye", npad * ne * 4)?;

                // Router logits for all rows, then GPU routing.
                let rdt = graph.desc(*router).dtype;
                let rw = r(*router)?;
                if native_dense_supported(rdt) {
                    rec.linear_native(rdt, rw, xb, logits.as_ref(), rows, ne, n_expert);
                } else if matches!(rdt, infr_core::DType::F32) {
                    rec.linear_f32(rw, xb, logits.as_ref(), rows, ne, n_expert);
                } else {
                    rec.linear(rw, xb, logits.as_ref(), rows, ne, n_expert);
                }
                rec.moe_topk(
                    logits.as_ref(),
                    ids.as_ref(),
                    wts.as_ref(),
                    rows,
                    n_expert,
                    n_used,
                    *scale,
                );
                rec.zero(counts.as_ref(), n_expert);
                rec.moe_bucket_count(ids.as_ref(), counts.as_ref(), n_pairs);
                rec.moe_bucket_scan(counts.as_ref(), offsets.as_ref(), fill.as_ref(), n_expert);
                rec.moe_bucket_scatter(
                    ids.as_ref(),
                    wts.as_ref(),
                    offsets.as_ref(),
                    fill.as_ref(),
                    bucket_rows.as_ref(),
                    bucket_wts.as_ref(),
                    inv_pos.as_ref(),
                    n_pairs,
                    n_used,
                );

                let (gw, uw, dw) = (r(*gate_exps)?, r(*up_exps)?, r(*down_exps)?);
                // Gather+quant all assignments into the packed layout in one pass.
                rec.quant_q8_gather(
                    xb,
                    bucket_rows.as_ref(),
                    pool[&qa].as_ref(),
                    pool[&qda].as_ref(),
                    pool[&qsa].as_ref(),
                    n_pairs,
                    ne,
                );
                rec.matmul_mmq_experts(
                    gdt,
                    pool[&qa].as_ref(),
                    pool[&qda].as_ref(),
                    Some(pool[&qsa].as_ref()),
                    gw,
                    0,
                    stride,
                    counts.as_ref(),
                    offsets.as_ref(),
                    pool[&ge].as_ref(),
                    n_pairs,
                    ne,
                    nff,
                    n_expert,
                );
                // The up GEMM reads the same quantized activations and writes its own buffer —
                // disjoint from the gate GEMM, no barrier needed between them.
                rec.suppress_sync(true);
                rec.matmul_mmq_experts(
                    udt,
                    pool[&qa].as_ref(),
                    pool[&qda].as_ref(),
                    Some(pool[&qsa].as_ref()),
                    uw,
                    0,
                    stride,
                    counts.as_ref(),
                    offsets.as_ref(),
                    pool[&ue].as_ref(),
                    n_pairs,
                    ne,
                    nff,
                    n_expert,
                );
                rec.suppress_sync(false);
                rec.silu_mul(
                    pool[&ge].as_ref(),
                    pool[&ue].as_ref(),
                    pool[&ae].as_ref(),
                    n_pairs * nff,
                );
                rec.quant_q8(
                    pool[&ae].as_ref(),
                    pool[&dqa].as_ref(),
                    pool[&dda].as_ref(),
                    pool[&dsa].as_ref(),
                    n_pairs,
                    nff,
                );
                rec.matmul_mmq_experts(
                    ddt,
                    pool[&dqa].as_ref(),
                    pool[&dda].as_ref(),
                    (!down_q6).then(|| pool[&dsa].as_ref()),
                    dw,
                    0,
                    stride,
                    counts.as_ref(),
                    offsets.as_ref(),
                    pool[&ye].as_ref(),
                    n_pairs,
                    nff,
                    ne,
                    n_expert,
                );
                rec.moe_scatter_reduce(
                    pool[&ye].as_ref(),
                    bucket_wts.as_ref(),
                    inv_pos.as_ref(),
                    yb,
                    rows,
                    ne,
                    n_used,
                );
                transient.extend([
                    logits,
                    ids,
                    wts,
                    counts,
                    offsets,
                    fill,
                    bucket_rows,
                    bucket_wts,
                    inv_pos,
                ]);
                return Ok(());
            }
            // Per-execute routing scratch. Local boxes used via `.as_ref()`, then moved into
            // `transient` at the end of the arm so they outlive `rec.finish()`.
            let al = |n: usize| be_.alloc((n * 4).max(4), BufferUsage::Activations);
            let logits = al(n_expert)?;
            let ids = al(n_used)?;
            let wts = al(n_used)?;
            let gbuf = al(n_used * nff)?;
            let ubuf = al(n_used * nff)?;
            let abuf = al(n_used * nff)?;
            let ybuf = al(n_used * ne)?;
            let xb = r(*x)?;

            // Router logits over all experts.
            let rdt = graph.desc(*router).dtype;
            let rw = r(*router)?;
            if native_dense_supported(rdt) {
                rec.linear_native(rdt, rw, xb, logits.as_ref(), 1, ne, n_expert);
            } else if matches!(rdt, infr_core::DType::F32) {
                // qwen3moe ships the router (ffn_gate_inp) as F32 — the f16 GEMV would read its
                // bytes as f16 garbage and route to arbitrary experts.
                rec.linear_f32(rw, xb, logits.as_ref(), 1, ne, n_expert);
            } else {
                rec.linear(rw, xb, logits.as_ref(), 1, ne, n_expert);
            }
            // Softmax-renormalized top-`n_used`, weights pre-scaled by `scale`.
            rec.moe_topk(
                logits.as_ref(),
                ids.as_ref(),
                wts.as_ref(),
                1,
                n_expert,
                n_used,
                *scale,
            );
            // Fused per-role expert GEMVs: gate/up read the shared row; down reads each slot's act.
            rec.linear_native_id_multi(
                gdt,
                r(*gate_exps)?,
                ids.as_ref(),
                n_used,
                stride,
                xb,
                false,
                gbuf.as_ref(),
                ne,
                nff,
            );
            rec.linear_native_id_multi(
                udt,
                r(*up_exps)?,
                ids.as_ref(),
                n_used,
                stride,
                xb,
                false,
                ubuf.as_ref(),
                ne,
                nff,
            );
            let n_act = n_used * nff;
            match act {
                Activation::Silu => {
                    rec.silu_mul(gbuf.as_ref(), ubuf.as_ref(), abuf.as_ref(), n_act)
                }
                Activation::Sigmoid => {
                    rec.mul_sigmoid(gbuf.as_ref(), ubuf.as_ref(), abuf.as_ref(), n_act)
                }
                Activation::Gelu => {
                    rec.gelu_mul_off(gbuf.as_ref(), ubuf.as_ref(), 0, abuf.as_ref(), n_act)
                }
            }
            rec.linear_native_id_multi(
                ddt,
                r(*down_exps)?,
                ids.as_ref(),
                n_used,
                stride,
                abuf.as_ref(),
                true,
                ybuf.as_ref(),
                nff,
                ne,
            );
            // `Op::MoeFfn` dst is the pure FFN output (residual Add is a separate op), but
            // moe_accumulate ADDs — zero dst first.
            let dstb = r(*dst)?;
            rec.zero(dstb, ne);
            rec.moe_accumulate(ybuf.as_ref(), wts.as_ref(), dstb, ne, n_used);
            transient.extend([logits, ids, wts, gbuf, ubuf, abuf, ybuf]);
        }
    }
    Ok(())
}

pub(crate) fn execute(be_: &VulkanBackend, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
    let plan = plan
        .as_any()
        .downcast_ref::<VkDecodePlan>()
        .ok_or_else(|| be("vulkan adapter: plan is not a VkDecodePlan"))?;
    if !plan.eligible {
        return execute_static(be_, &plan.graph, bindings);
    }

    // Record-once replay: build the recording on the first execute (cache miss), then only refresh
    // `params` from `positions[0]` and resubmit. The runner reuses the SAME bound buffers across the
    // whole decode loop, so the recorded descriptor sets stay valid.
    let mut guard = plan.replay.lock().unwrap();
    if guard.is_none() {
        *guard = Some(record_decode_replay(be_, &plan.graph, bindings)?);
    }
    let replay = guard.as_ref().unwrap();
    // pos from positions[0] (decode rows=1); kv_len = pos+1.
    let pos = read_pos0(be_, resolve(&replay.scratch, bindings, replay.positions)?)?;
    let kv_len = pos + 1;
    let mut pbytes = [0u8; 8];
    pbytes[0..4].copy_from_slice(&pos.to_le_bytes());
    pbytes[4..8].copy_from_slice(&kv_len.to_le_bytes());
    be_.upload(replay.params.as_ref(), &pbytes)?;
    replay.recorded.replay().map_err(|e| be(e.to_string()))?;
    Ok(())
}

/// Build the record-once decode replay: persistent scratch + params SSBO, one recording via the
/// `_dyn` (params-driven) kernels for the pos-dependent ops. Only called for a [`decode_eligible`]
/// graph. The returned recording is replayed (not submitted here) — the caller updates `params` and
/// calls `replay()`.
fn record_decode_replay(
    be_: &VulkanBackend,
    graph: &Graph,
    bindings: &Bindings,
) -> Result<DecodeReplay> {
    let scratch = alloc_scratch(be_, graph)?;
    // `[pos, kv_len]` — Staging (host-visible mapped) so per-token `upload` is a plain memcpy.
    let params = be_.alloc(8, BufferUsage::Staging)?;
    let positions = graph
        .ops
        .iter()
        .find_map(|op| match op {
            Op::QkNormRope { positions, .. } | Op::Rope { positions, .. } => Some(*positions),
            _ => None,
        })
        .ok_or_else(|| be("vulkan adapter: eligible decode has no positions tensor"))?;
    let (fused_kv_write, mut skip_op) = kv_write_peephole(graph);
    let (fused_add, skip_add) = linear_add_peephole(graph);
    skip_op.extend(skip_add);

    let mut transient: Vec<Box<dyn Buffer>> = Vec::new();
    let mut dyn_args: Vec<DynAttnCtx> = Vec::new();
    let mut pool: ScratchPool = HashMap::new();
    let dummy = be_.alloc(16, BufferUsage::Activations)?;
    let rec = be_.recorder_persistent()?;
    let mode = RopeMode::Dynamic(params.as_ref());
    for (op_idx, op) in graph.ops.iter().enumerate() {
        if skip_op.contains(&op_idx) {
            continue;
        }
        lower_op(
            be_,
            graph,
            op_idx,
            op,
            &rec,
            &scratch,
            bindings,
            &fused_kv_write,
            &fused_add,
            &mode,
            &mut transient,
            &mut dyn_args,
            &mut pool,
            dummy.as_ref(),
        )?;
    }
    for c in dyn_args.drain(..) {
        transient.extend([c.args, c.pm, c.pl, c.pacc]);
    }
    transient.extend(pool.into_values());
    let recorded = rec.finish_record().map_err(|e| be(e.to_string()))?;
    // dummy is unused in an eligible decode (m=1 GEMV path), but hold it (and any transient) so the
    // recording can't reference a freed buffer.
    transient.push(dummy);
    Ok(DecodeReplay {
        scratch,
        params,
        positions,
        recorded,
        _transient: transient,
    })
}

/// Per-execute static recording: allocate `Internal` scratch fresh, record every op via `lower_op`
/// (Static mode — pos as a push constant read from `positions[0]`), submit + wait. Used for prefill
/// batches and every ineligible decode (gemma/E2B/MoE/qwen35).
fn execute_static(be_: &VulkanBackend, graph: &Graph, bindings: &Bindings) -> Result<()> {
    let scratch = alloc_scratch(be_, graph)?;

    // RoPE position: the static `qk_norm_rope`/`rope` kernels take a scalar `rope_pos`, but the IR
    // carries a `positions` i32 tensor. Read `positions[0]` (decode rows=1, or the start of a
    // consecutive-prefill run) up front — `download` syncs, so it must precede the recorder.
    let mut rope_pos: HashMap<u32, usize> = HashMap::new();
    for op in &graph.ops {
        let pid = match op {
            Op::Rope { positions, .. } | Op::QkNormRope { positions, .. } => Some(*positions),
            _ => None,
        };
        if let Some(pid) = pid {
            if let std::collections::hash_map::Entry::Vacant(e) = rope_pos.entry(pid.0) {
                let mut b = [0u8; 4];
                be_.download(resolve(&scratch, bindings, pid)?, &mut b)?;
                e.insert(i32::from_le_bytes(b) as usize);
            }
        }
    }

    let (fused_kv_write, mut skip_op) = kv_write_peephole(graph);
    let (fused_add, skip_add) = linear_add_peephole(graph);
    skip_op.extend(skip_add);

    // Transient buffers allocated inside the op loop (GEMM/attention/MoE scratch) must outlive the
    // recorder — hold them here so they drop only after `rec.finish()` submits.
    let mut transient: Vec<Box<dyn Buffer>> = Vec::new();
    // A tiny unused buffer bound as the (scales, mins) args of the f16 `matmul_proj` GEMM.
    let dummy = be_.alloc(16, BufferUsage::Activations)?;

    let rec = be_.recorder()?;
    let mode = RopeMode::Static(&rope_pos);
    let mut dyn_args: Vec<DynAttnCtx> = Vec::new();
    let mut pool: ScratchPool = HashMap::new();
    for (op_idx, op) in graph.ops.iter().enumerate() {
        if skip_op.contains(&op_idx) {
            continue;
        }
        lower_op(
            be_,
            graph,
            op_idx,
            op,
            &rec,
            &scratch,
            bindings,
            &fused_kv_write,
            &fused_add,
            &mode,
            &mut transient,
            &mut dyn_args,
            &mut pool,
            dummy.as_ref(),
        )?;
    }
    for c in dyn_args.drain(..) {
        transient.extend([c.args, c.pm, c.pl, c.pacc]);
    }
    transient.extend(pool.into_values());
    rec.finish().map_err(|e| be(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::graph::Graph;
    use infr_core::tensor::TensorDesc;
    use infr_core::DType;

    /// Prove the adapter machinery end-to-end (compile → bind → execute → download): a one-op
    /// `RmsNorm` graph run through the Vulkan seam must match a host reference. (Milestone #2: a
    /// small graph runs on Vulkan.)
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn rmsnorm_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (rows, dim, eps) = (2usize, 8usize, 1e-6f32);
        let x: Vec<f32> = (0..rows * dim).map(|i| i as f32 * 0.1 - 0.4).collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + i as f32 * 0.05).collect();
        // host reference rmsnorm
        let mut want = vec![0f32; rows * dim];
        for r in 0..rows {
            let b = r * dim;
            let ss = (0..dim).map(|i| x[b + i] * x[b + i]).sum::<f32>() / dim as f32;
            let s = 1.0 / (ss + eps).sqrt();
            for i in 0..dim {
                want[b + i] = x[b + i] * s * w[i];
            }
        }
        // build the graph
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![dim], DType::F32));
        let yi = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
        g.push(Op::RmsNorm {
            x: xi,
            weight: wi,
            dst: yi,
            rows: rows as u32,
            dim: dim as u32,
            eps,
        });
        // device buffers + bind
        let xb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        let wb = be_.alloc(dim * 4, BufferUsage::Weights).unwrap();
        let yb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        be_.upload(wb.as_ref(), bytemuck::cast_slice(&w)).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(wi, wb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; rows * dim];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..rows * dim {
            assert!(
                (got[i] - want[i]).abs() < 1e-3,
                "rmsnorm mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// A one-op `Linear` graph (f16 weight, 1-row GEMV) through the seam must match a host matvec.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn linear_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (in_f, out_f) = (16usize, 4usize);
        let x: Vec<f32> = (0..in_f).map(|i| i as f32 * 0.1 - 0.8).collect();
        let w: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32 * 0.03).sin()).collect();
        let wf16: Vec<u8> = w
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        // host reference uses the same f16-rounded weight the GPU reads
        let wq: Vec<f32> = wf16
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        let mut want = vec![0f32; out_f];
        for (o, wo) in want.iter_mut().enumerate() {
            *wo = (0..in_f).map(|i| x[i] * wq[o * in_f + i]).sum();
        }
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![1, in_f], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![out_f, in_f], DType::F16));
        let yi = g.output(TensorDesc::new(vec![1, out_f], DType::F32));
        g.push(Op::Linear {
            x: xi,
            weight: wi,
            dst: yi,
            m: 1,
            in_f: in_f as u32,
            out_f: out_f as u32,
            w_off: 0,
        });
        let xb = be_.alloc(in_f * 4, BufferUsage::Activations).unwrap();
        let wb = be_.alloc(wf16.len(), BufferUsage::Weights).unwrap();
        let yb = be_.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        be_.upload(wb.as_ref(), &wf16).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(wi, wb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; out_f];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for o in 0..out_f {
            assert!(
                (got[o] - want[o]).abs() < 1e-2,
                "linear mismatch at {o}: got {} want {}",
                got[o],
                want[o]
            );
        }
    }

    /// A one-op `GatedAct` (SwiGLU: silu(gate)·up) graph through the seam must match a host loop.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn gated_act_silu_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let nff = 8usize;
        let gate: Vec<f32> = (0..nff).map(|i| i as f32 * 0.2 - 0.7).collect();
        let up: Vec<f32> = (0..nff).map(|i| 1.0 - i as f32 * 0.1).collect();
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let want: Vec<f32> = (0..nff).map(|i| silu(gate[i]) * up[i]).collect();
        let mut g = Graph::new();
        let gi = g.input(TensorDesc::new(vec![1, nff], DType::F32));
        let ui = g.input(TensorDesc::new(vec![1, nff], DType::F32));
        let yi = g.output(TensorDesc::new(vec![1, nff], DType::F32));
        g.push(Op::GatedAct {
            gate: gi,
            up: ui,
            dst: yi,
            rows: 1,
            nff: nff as u32,
            act: Activation::Silu,
            up_off: 0,
        });
        let gb = be_.alloc(nff * 4, BufferUsage::Activations).unwrap();
        let ub = be_.alloc(nff * 4, BufferUsage::Activations).unwrap();
        let yb = be_.alloc(nff * 4, BufferUsage::Activations).unwrap();
        be_.upload(gb.as_ref(), bytemuck::cast_slice(&gate))
            .unwrap();
        be_.upload(ub.as_ref(), bytemuck::cast_slice(&up)).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(gi, gb.as_ref());
        bind.bind(ui, ub.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; nff];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..nff {
            assert!(
                (got[i] - want[i]).abs() < 1e-3,
                "gated_act mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// A one-op `QkNormRope` graph (per-head RMSNorm + RoPE, f32 in → f16 out, positions tensor) must
    /// match a host reference (f16 tolerance).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn qk_norm_rope_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (nh, hd, pos) = (2usize, 8usize, 3usize);
        let (eps, theta, rope_dim) = (1e-6f32, 10000.0f32, 8usize);
        let x: Vec<f32> = (0..nh * hd).map(|i| i as f32 * 0.1 - 0.5).collect();
        let w: Vec<f32> = (0..hd).map(|i| 1.0 + i as f32 * 0.05).collect();
        // host reference: per-head rmsnorm (×w) then split-half NEOX rope
        let mut want = vec![0f32; nh * hd];
        let hf = rope_dim / 2;
        for h in 0..nh {
            let b = h * hd;
            let ss = (0..hd).map(|i| x[b + i] * x[b + i]).sum::<f32>() / hd as f32;
            let s = 1.0 / (ss + eps).sqrt();
            let nrm: Vec<f32> = (0..hd).map(|i| x[b + i] * s * w[i]).collect();
            want[b..b + hd].copy_from_slice(&nrm);
            for p in 0..hf {
                let ang = pos as f32 * theta.powf(-2.0 * p as f32 / rope_dim as f32);
                let (sn, c) = (ang.sin(), ang.cos());
                want[b + p] = nrm[p] * c - nrm[p + hf] * sn;
                want[b + p + hf] = nrm[p] * sn + nrm[p + hf] * c;
            }
        }
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![1, nh, hd], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![hd], DType::F32));
        let pi = g.input(TensorDesc::new(vec![1], DType::I32));
        let yi = g.output(TensorDesc::new(vec![1, nh, hd], DType::F16));
        g.push(Op::QkNormRope {
            x: xi,
            weight: wi,
            positions: pi,
            dst: yi,
            rows: 1,
            n_head: nh as u32,
            head_dim: hd as u32,
            rope_dim: rope_dim as u32,
            theta,
            eps,
            freq_factors: None,
        });
        let xb = be_.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
        let wb = be_.alloc(hd * 4, BufferUsage::Weights).unwrap();
        let pb = be_.alloc(4, BufferUsage::Activations).unwrap();
        let yb = be_.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        be_.upload(wb.as_ref(), bytemuck::cast_slice(&w)).unwrap();
        be_.upload(pb.as_ref(), &(pos as i32).to_le_bytes())
            .unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(wi, wb.as_ref());
        bind.bind(pi, pb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut y16 = vec![0u8; nh * hd * 2];
        be_.download(yb.as_ref(), &mut y16).unwrap();
        let got: Vec<f32> = y16
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        for i in 0..nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 2e-2,
                "qk_norm_rope mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// A one-op `Attention` graph (GQA, causal, f16 q + f16 KV) must match a host softmax-attention.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn attention_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (nh, nkv, hd, pos) = (2usize, 1usize, 8usize, 2usize);
        let kv_len = pos + 1; // causal: query at `pos` attends keys 0..=pos
        let scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;
        let to_f16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                .collect()
        };
        let deq = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };
        let q: Vec<f32> = (0..nh * hd).map(|i| (i as f32 * 0.07).sin()).collect();
        let k: Vec<f32> = (0..kv_len * nkv * hd)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..kv_len * nkv * hd)
            .map(|i| i as f32 * 0.01 - 0.1)
            .collect();
        let (qf, kf, vf) = (to_f16(&q), to_f16(&k), to_f16(&v));
        let (qd, kd, vd) = (deq(&qf), deq(&kf), deq(&vf)); // host uses the same f16-rounded values
                                                           // host GQA softmax attention
        let mut want = vec![0f32; nh * hd];
        for h in 0..nh {
            let kvh = h / group;
            let mut sc = vec![0f32; kv_len];
            let mut mx = f32::NEG_INFINITY;
            for (j, scj) in sc.iter_mut().enumerate() {
                let d: f32 = (0..hd)
                    .map(|x| qd[h * hd + x] * kd[(j * nkv + kvh) * hd + x])
                    .sum();
                *scj = d * scale;
                mx = mx.max(*scj);
            }
            let l: f32 = sc.iter().map(|s| (s - mx).exp()).sum();
            for (j, &s) in sc.iter().enumerate() {
                let p = (s - mx).exp() / l;
                for x in 0..hd {
                    want[h * hd + x] += p * vd[(j * nkv + kvh) * hd + x];
                }
            }
        }
        let mut g = Graph::new();
        let qi = g.input(TensorDesc::new(vec![1, nh, hd], DType::F16));
        let ki = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
        let vi = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
        let yi = g.output(TensorDesc::new(vec![1, nh, hd], DType::F32));
        g.push(Op::Attention {
            q: qi,
            k_cache: ki,
            v_cache: vi,
            dst: yi,
            rows: 1,
            kv_len: kv_len as u32,
            n_head: nh as u32,
            n_kv: nkv as u32,
            head_dim: hd as u32,
            scale,
            mask: AttnMask::Causal,
            pos: pos as u32,
        });
        let qb = be_.alloc(qf.len(), BufferUsage::Activations).unwrap();
        let kb = be_.alloc(kf.len(), BufferUsage::Activations).unwrap();
        let vb = be_.alloc(vf.len(), BufferUsage::Activations).unwrap();
        let yb = be_.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
        be_.upload(qb.as_ref(), &qf).unwrap();
        be_.upload(kb.as_ref(), &kf).unwrap();
        be_.upload(vb.as_ref(), &vf).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(qi, qb.as_ref());
        bind.bind(ki, kb.as_ref());
        bind.bind(vi, vb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; nh * hd];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 2e-2,
                "attention mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    // ---- Q8_0 helpers (block=32: f16 scale + 32×int8 = 34 bytes) so a MoE test can bind id-native
    // expert banks and the host reference dequants the SAME rounded values the GPU shader does. ----
    fn q8_0(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 32 * 34);
        for blk in x.chunks(32) {
            let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            for &v in blk {
                out.push(((v * id).round().clamp(-127.0, 127.0) as i8) as u8);
            }
        }
        out
    }
    fn deq_q8(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 34 * 32);
        for blk in bytes.chunks(34) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            for i in 0..32 {
                out.push((blk[2 + i] as i8 as f32) * d);
            }
        }
        out
    }

    /// A one-op `MoeFfn` graph (Q8_0 router + stacked experts) through the seam must match a host
    /// reference that mirrors the CPU `Op::MoeFfn` interpreter on the SAME q8-rounded weights:
    /// router softmax → top-`n_used` → per-expert SwiGLU → weighted (×scale) accumulate.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (32usize, 4usize, 2usize, 32usize);
        let scale = 1.3f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5; // deterministic weight/act filler
        let x: Vec<f32> = (0..ne).map(|i| f(i, 0.11) + 0.05).collect();
        // Distinct per-expert router rows so top-k selection is unambiguous.
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        let gate: Vec<f32> = (0..n_expert * nff * ne).map(|i| f(i, 0.017)).collect();
        let up: Vec<f32> = (0..n_expert * nff * ne).map(|i| f(i, 0.023)).collect();
        let down: Vec<f32> = (0..n_expert * ne * nff).map(|i| f(i, 0.029)).collect();
        let (rq, gq, uq, dq) = (q8_0(&router), q8_0(&gate), q8_0(&up), q8_0(&down));
        // Host reference uses the dequantized (q8-rounded) weights — same values the GPU reads.
        let (rd, gd, ud, dd) = (deq_q8(&rq), deq_q8(&gq), deq_q8(&uq), deq_q8(&dq));
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let silu = |z: f32| z / (1.0 + (-z).exp());
        // router logits → softmax over all experts
        let logits: Vec<f32> = (0..n_expert)
            .map(|e| dot(&rd[e * ne..(e + 1) * ne], &x))
            .collect();
        let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
        let psum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= psum);
        let mut idx: Vec<usize> = (0..n_expert).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        idx.truncate(n_used);
        let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
        let mut want = vec![0f32; ne];
        for &e in &idx {
            let gs = e * nff * ne;
            let ds = e * ne * nff;
            let actv: Vec<f32> = (0..nff)
                .map(|j| {
                    let g = dot(&gd[gs + j * ne..gs + (j + 1) * ne], &x);
                    let u = dot(&ud[gs + j * ne..gs + (j + 1) * ne], &x);
                    silu(g) * u
                })
                .collect();
            let w_e = probs[e] / wsum * scale;
            for (i, wi) in want.iter_mut().enumerate() {
                *wi += w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actv);
            }
        }
        // graph
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![1, ne], DType::F32));
        let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
        let gi = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q8_0));
        let ui = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q8_0));
        let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q8_0));
        let yi = g.output(TensorDesc::new(vec![1, ne], DType::F32));
        g.push(Op::MoeFfn {
            x: xi,
            router: ri,
            gate_exps: gi,
            up_exps: ui,
            down_exps: di,
            dst: yi,
            ne: ne as u32,
            n_expert: n_expert as u32,
            n_used: n_used as u32,
            n_ff_exp: nff as u32,
            scale,
            act: Activation::Silu,
        });
        let mk = |bytes: &[u8], usage| {
            let b = be_.alloc(bytes.len(), usage).unwrap();
            be_.upload(b.as_ref(), bytes).unwrap();
            b
        };
        let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
        let rb = mk(&rq, BufferUsage::Weights);
        let gb = mk(&gq, BufferUsage::Weights);
        let ub = mk(&uq, BufferUsage::Weights);
        let db = mk(&dq, BufferUsage::Weights);
        let yb = be_.alloc(ne * 4, BufferUsage::Activations).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(ri, rb.as_ref());
        bind.bind(gi, gb.as_ref());
        bind.bind(ui, ub.as_ref());
        bind.bind(di, db.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; ne];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..ne {
            assert!(
                (got[i] - want[i]).abs() < 3e-3,
                "moe_ffn mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }
}
