//! Vulkan adapter for the agnostic `infr_core::Backend` seam: lower a `Graph` of composite ops onto
//! the existing fused Recorder kernels, recorded into one command buffer. Mirrors the CPU
//! interpreter (`infr-llama::seam`) op-for-op, but executes on the GPU — so the SAME model
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

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VkDecodePlan {
    fn boxed(graph: &Graph) -> Box<dyn Plan> {
        Box::new(VkDecodePlan {
            graph: graph.clone(),
            eligible: decode_eligible(graph),
            replay: Mutex::new(None),
        })
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
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
    /// `[pos, kv_len]` (u32×2) SSBO the `_dyn` kernels read. When `self_advancing`, a
    /// `params_advance` dispatch recorded FIRST increments it on the device every replay
    /// (initialized to `[pos0-1, pos0]` at record time) — the adapter never touches it again.
    params: Box<dyn Buffer>,
    /// The recording starts with the device-side params increment; `execute` skips the host
    /// `read_pos0` + params upload. INFR_NO_GPU_POS=1 at record time forces the old host path.
    self_advancing: bool,
    /// Chained-decode id ring (64 × u32, host-visible): the recording's trailing `id_log`
    /// dispatch writes `ring[pos & 63] = sampled id` each iteration; `execute_chain` reads the
    /// whole chunk back in one go. `Some` only when the graph ends in Argmax/Sample AND
    /// `self_advancing`.
    ring: Option<Box<dyn Buffer>>,
    /// The `positions` Input tensor — `execute` downloads element 0 to learn `pos` for `params`.
    positions: TensorId,
    recorded: RecordedCmd,
    /// Transient GEMM/attention scratch the recording references. Empty for an eligible decode (all
    /// ops are GEMV/silu/dyn kernels with no transient), held anyway so a stray one can't dangle.
    _transient: Vec<Box<dyn Buffer>>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn compile(graph: &Graph) -> Result<Box<dyn Plan>> {
    // Eligibility (record-once replay vs per-execute static recording) is a pure function of the
    // graph shape, so decide it here, once. `execute` builds the replay lazily on first run.
    Ok(VkDecodePlan::boxed(graph))
}

/// Record-once replay applies to a single-token decode graph the `_dyn` kernels cover: every
/// `Attention` is `rows==1` with a replayable KV dtype (f16, or coupled Q8_0); RoPE rides on
/// `QkNormRope` or an f16-out standalone `Rope` (no ff there). SWA windows/scale/Softcap/MoeFfn/
/// QkNorm and the qwen35 recurrent ops (`Conv1dSilu`/`DeltaNet`) are all replay-safe. Anything
/// else (prefill batches, dequant-prepass KV dtypes, f32 in-place Rope) falls back to the static
/// per-execute path.
/// Standard-GGUF-block low-bit KV quants that ride the dequant→f16 prepass path (not native reads).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn is_kv_quant(dt: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(dt, Q4_0 | Q4_1 | Q5_0 | Q5_1 | Iq4Nl)
}

/// Dense non-f16 KV caches (f32/bf16): stored via a cast-store, read via a cast→f16 prepass.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn is_kv_dense_alt(dt: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(dt, F32 | Bf16)
}

/// TurboQuant KV caches (WHT-rotated): quantizing WriteKv + dequant→f16 prepass.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn is_turbo(dt: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(dt, Turbo2 | Turbo3 | Turbo4)
}

/// Any KV cache dtype that rides the dequant/cast → f16 prepass (not f16 or native-Q8).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn is_kv_prepass(dt: infr_core::DType) -> bool {
    is_kv_quant(dt) || is_kv_dense_alt(dt) || is_turbo(dt)
}

/// `Op::MoeFfn` small-m fast-path threshold: at or below this many rows (tokens in one forward),
/// use the per-active-expert id-indexed GEMV path instead of the batched whole-bank-streaming GEMM
/// (see the two paths' docs at the `Op::MoeFfn` match arm).
///
/// Measured on Qwen3-30B-A3B-Q4_K_M / RDNA3 (7900 XTX), forcing each path via `INFR_MOE_SMALL_M`
/// (`=64` / `=0`) at `-d 4096` so both sides run the SAME row count (small path t/s vs batched
/// path t/s):
///
/// | rows (`-p`) |    small |  batched | small path wins by |
/// |------------:|---------:|---------:|--------------------|
/// |           2 |  114.4   |   72.7   | +57%                |
/// |           4 |  158.1   |  125.0   | +26%                |
/// |           8 |  224.4   |  189.6   | +18%                |
/// |          16 |  275.8   |  286.3   | -4% (batched ahead) |
/// |          32 |  326.0   |  414.0   | -21% (batched ahead)|
/// |          64 |  438.8   |  839.4   | -48% (batched ahead)|
///
/// The crossover sits between rows=8 and rows=16 (batched amortizes its per-active-expert weight
/// read across more of the ~n_used·rows assignments sharing an expert as rows grows; the small
/// path re-reads a shared expert's weight once per occurrence, so it degrades faster). `8` sits
/// right at the edge of the small path's win region with headroom to spare.
///
/// `INFR_MOE_SMALL_M` overrides the threshold for experimentation, clamped to `MOE_SMALL_M_MAX`:
/// the small path's GEMV dispatches one workgroup PER (row, slot, out-column) with no row-tile
/// batching, so at the seam's normal prefill chunk size (`INFR_UBATCH`, 1024 rows by default) an
/// UNCLAMPED override turns every prefill chunk into a multi-second single dispatch — this isn't
/// just slow, it trips the amdgpu ring watchdog and device-losts the whole process (reproduced:
/// `INFR_MOE_SMALL_M=100000` on a 1024-row chunk hangs the GPU; see `runner.rs`'s UBATCH doc for
/// the same failure class with an unchunked prefill).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn moe_small_m_threshold() -> usize {
    const MOE_SMALL_M_MAX: usize = 64;
    std::env::var("INFR_MOE_SMALL_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .min(MOE_SMALL_M_MAX)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn decode_eligible(graph: &Graph) -> bool {
    // INFR_SEAM_NO_REPLAY forces the static per-execute path (INFR_PROF2 timestamps work there;
    // the replay path can't report them).
    if std::env::var("INFR_SEAM_NO_REPLAY").is_ok() {
        return false;
    }
    let mut has_rope = false;
    let mut has_attn = false;
    // Record-once replay drives ALL pos-dependent ops (QkNormRope/WriteKv/Attention) from ONE shared
    // `params` position (a single-token decode: every layer sits at the same absolute position). A
    // graph whose attention ops sit at DIFFERENT positions (e.g. the MTP self-chaining draft, which
    // unrolls n_steps single-token forwards at consecutive positions p, p+1, … into one graph) is
    // NOT a single-token decode — replaying it would collapse every step onto the same KV row + RoPE
    // angle + kv_len. Force those onto the static per-op-position path.
    let mut attn_kv_len: Option<u32> = None;
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
                kv_len,
                ..
            } => {
                has_attn = true;
                if *rows != 1 || *head_dim % 4 != 0 || *head_dim > 512 {
                    return false;
                }
                // Multi-position graph (see the comment above `attn_kv_len`): different kv_len across
                // attention ops ⟹ not a single-token decode ⟹ ineligible for record-once replay.
                match attn_kv_len {
                    Some(prev) if prev != *kv_len => return false,
                    _ => attn_kv_len = Some(*kv_len),
                }
                // A COUPLED Q8_0 KV cache (K==V==Q8) replays: store_q8_dyn writes the row at
                // params' pos and the planar-Q8 dyn attention kernels (attention_kv_dyn_q8 /
                // attn_partial_dynac_q8) read it. (The historic "store_q8_dyn mis-decodes under
                // replay" was actually the Attention lowering passing the ROW capacity where the
                // Q8 kernels expect the ELEMENT capacity `cap` — the planar scales-region base —
                // see the `cap_rows` fix at the Dynamic Attention branch.) A DECOUPLED Q8 side
                // still forces static: the dyn kernels' q8 variant dequants BOTH sides, and the
                // per-side mixed read only exists on the static attn_partial/attention_kv path.
                // The dequant→f16 prepass dtypes (low-bit block quants, f32/bf16, turbo) force
                // static too — their WriteKv/prepass have no dyn kernels.
                let (kdt, vdt) = (graph.desc(*k_cache).dtype, graph.desc(*v_cache).dtype);
                let k_q8 = matches!(kdt, infr_core::DType::Q8_0);
                let v_q8 = matches!(vdt, infr_core::DType::Q8_0);
                if k_q8 != v_q8 || is_kv_prepass(kdt) || is_kv_prepass(vdt) {
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
            // Conv1dSilu/DeltaNet (qwen35 recurrent state) are replay-safe too: decode (rows==1)
            // takes the sequential kernels, which are pure in-place RMW of the PERSISTENT state
            // bindings (kbufs/vbufs, stable across the decode loop) with no pos push constant —
            // the recorded tape re-reads the live state every replay, and the tape's leading
            // global barrier orders iteration i's state write before i+1's read under replay_n.
            // GatedRmsNorm (qwen35 DeltaNet z-gate) is pos-independent elementwise like QkNorm —
            // same replay-safety argument.
            Op::MoeFfn { .. }
            | Op::QkNorm { .. }
            | Op::GatedRmsNorm { .. }
            | Op::Softcap { .. }
            | Op::Conv1dSilu { .. }
            | Op::DeltaNet { .. } => {}
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
#[cfg_attr(infr_profile, infr_prof::instrument)]
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
#[cfg_attr(infr_profile, infr_prof::instrument)]
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
/// [`alloc_scratch`]'s result: the per-tensor `Internal` scratch (indexed by `TensorId`) plus the
/// shared 16-byte `dummy` buffer (bound as the unused scales/mins args of the f16 GEMM).
type ScratchSet = (Vec<Option<Box<dyn Buffer>>>, Box<dyn Buffer>);

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn alloc_scratch(be_: &VulkanBackend, graph: &Graph) -> Result<ScratchSet> {
    let mut scratch: Vec<Option<Box<dyn Buffer>>> =
        (0..graph.tensors.len()).map(|_| None).collect();
    // Batch the whole scratch set into ONE zero-init submit (`alloc_zeroed_batch`): the naive
    // per-tensor `alloc` pays a one-shot submit + queue_wait_idle per Internal (~70 x ~35us =
    // ~2.5ms per execute on a 7900 XTX) — the dominant HOST cost of a small-m prefill step.
    let mut idx: Vec<usize> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
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
            idx.push(i);
            sizes.push(bytes.max(4));
        }
    }
    // The 16-byte `dummy` (bound as the unused scales/mins args of the f16 `matmul_proj` GEMM)
    // rides the same batch — its own `alloc` would pay one more per-execute fill submit.
    sizes.push(16);
    let mut bufs = be_.alloc_zeroed_batch(&sizes, BufferUsage::Activations)?;
    let dummy = bufs.pop().expect("alloc_zeroed_batch returned sizes.len()");
    for (i, buf) in idx.into_iter().zip(bufs) {
        scratch[i] = Some(buf);
    }
    Ok((scratch, dummy))
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

/// mmv (int8 dp4a decode GEMV) size gate for the m≥3 small-m PREFILL mrow path: below this
/// weight-element count the dequant GEMV is already so short (<~10us) that the extra quant_q8
/// dispatch's fixed bubble (~2-3us on a 7900 XTX) eats the kernel saving. Probe data
/// (gemv_vs_mmv, 7900 XTX): 4096x24576 gate+up −38%, 12288x4096 Q6_K down −5.5% (= 48M elements,
/// the smallest clear payer), 4096x4096 o −5% kernel-only but a whole-model LOSS on
/// dispatch-bound decodes (gemma3-1b −2.3%, qwen3moe −6% with no gate).
const MMV_MIN_ELEMS: usize = 48 << 20;

/// The m=1 DECODE int8 mmv is DEFAULT-OFF (opt in with `INFR_MMV_DECODE=1`). The word-parallel
/// K-quant `dqblk` (whole-u32 decode, commit 51a35c8) sped up the SCALAR dequant GEMV enough that
/// it now beats the int8 mmv at m=1 across the board — measured `INFR_NO_MMV=1` (scalar) vs mmv,
/// tg128 r=3 on a 7900 XTX: Qwen3-8B 127.3 vs 123.0 (+3.5%), qwen35-4B 142.5 vs 139.4 (+2.2%),
/// 0.6B 667 vs 662 (+0.8%), 8B tg64@d4096 113.7 vs 110.4 (+3.0%). So decode routes to the scalar
/// GEMV; the mmv kernels stay reachable for A/B + future re-tuning. (The m≥3 mrow PREFILL mmv is
/// UNAFFECTED — it still wins: E2B pp4@d4096 loses 5.3% under `INFR_NO_MMV`.)
fn mmv_decode_enabled() -> bool {
    std::env::var("INFR_MMV_DECODE").is_ok() && std::env::var("INFR_NO_MMV").is_err()
}

/// Multi-warp int8 dp4a decode GEMV route (`native_mmv_mw.comp`, llama's mul_mat_vec_q block:
/// warp-per-row subgroupAdd, WARPS warps/block) for WAVE32-NATIVE GPUs (`subgroup_max == 32`:
/// NVIDIA/Intel). There the AMD-tuned scalar-dequant decode GEMV runs memory-LATENCY-starved
/// (~30 GB/s of 616 on an RTX 2080 Ti) — a per-op dead end for the scalar path, since it is the f32
/// dequant ALU + the `v[32]` register pressure (not the reduction) that cap it (a scalar multi-warp
/// variant measured SLOWER than the tree). Only dp4a (raw int8 blocks, no f32 dequant) breaks the
/// ceiling: Qwen3-1.7B Q4_K_M tg128 17.9 → 61.7 t/s isolated (3.4x), 0.11x → ~0.5x of llama.cpp.
///
/// OPT-IN (`INFR_MMV_MW=1`), NOT default: dp4a's int8-activation quant is coarser than the scalar
/// default and forks from the f32 CPU oracle earlier — the same precision tier as llama.cpp's mmvq,
/// but below infr's stricter token-for-token / deep-KV-Q8 coherence bar (this is exactly why 51a35c8
/// made scalar the default on AMD, where it was ALSO faster). Kept as a validated
/// (`mmv_mw_parity` test: matches `native_mmv` to ≤2e-3) knob so a wave32 deployment that accepts
/// llama-level decode precision can take the 3.4x. AMD (`subgroup_max == 64`) returns None → its
/// decode path is byte-identical (zero regression by construction). `INFR_MMV_MW_WARPS` ∈ {4,8}.
fn mmv_mw_choice(
    caps: &infr_core::backend::Capabilities,
    dt: infr_core::DType,
    in_f: usize,
    out_f: usize,
) -> Option<u32> {
    if std::env::var("INFR_MMV_MW").is_err() || !caps.i8_dot || caps.subgroup_max != 32 {
        return None;
    }
    if !matches!(dt, infr_core::DType::Q4K | infr_core::DType::Q6K) {
        return None;
    }
    // Skip tiny GEMVs where per-dispatch overhead dominates (k/v projections); the projection band
    // and lm_head are all well above 2M elements.
    if !in_f.is_multiple_of(32) || in_f * out_f < (2usize << 20) {
        return None;
    }
    let warps = std::env::var("INFR_MMV_MW_WARPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8u32);
    (matches!(warps, 4 | 8) && crate::gemm::native_mmv_mw_build_spv(dt, false, warps).is_some())
        .then_some(warps)
}

/// Get-or-alloc the pool buffer for (tag, bytes); returns the map key so callers can hold several
/// pool buffers at once via immutable indexing (`pool[&k].as_ref()`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
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

/// Small-m MoE scratch handle: a pooled `(tag, bytes)` key (the default — rides the per-execute
/// `pool`, so ALL MoE layers share one buffer per tag and the alloc happens at most once), or an
/// owned per-layer zero-filled `Backend::alloc` under the `INFR_NO_MOE_SM_POOL=1` escape. Resolve
/// with [`SmB::get`] at each use site (immutable pool indexing, same discipline as the batched
/// arm's `pool[&k]`); the owned variant is drained into `transient` via [`SmB::into_transient`] so
/// it outlives `rec.finish()` (pooled buffers are moved into `transient` at the end of the loop).
enum SmB {
    Pool((&'static str, usize)),
    Own(Box<dyn Buffer>),
}

impl SmB {
    fn get<'a>(&'a self, pool: &'a ScratchPool) -> &'a dyn Buffer {
        match self {
            SmB::Pool(k) => pool[k].as_ref(),
            SmB::Own(b) => b.as_ref(),
        }
    }

    /// Move an owned (escape-path) buffer into `transient` so the recording can't reference a freed
    /// buffer. Pooled buffers are owned by `pool` (drained into `transient` at the end of the loop),
    /// so `Pool` is a no-op here.
    fn into_transient(self, transient: &mut Vec<Box<dyn Buffer>>) {
        if let SmB::Own(b) = self {
            transient.push(b);
        }
    }
}

/// Allocate one small-m MoE scratch buffer of `elems` f32/u32 elements: pooled uninit by default
/// (every buffer is fully written before it is read within the op — see the small-m arm's scratch
/// doc for the per-buffer argument), or a fresh calloc-contract `Backend::alloc` when `no_pool`
/// (the `INFR_NO_MOE_SM_POOL=1` A/B escape — each device-local zero-fill costs a one-shot submit +
/// queue_wait_idle, the exact per-layer overhead pooling kills).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn sm_buf(
    pool: &mut ScratchPool,
    be_: &VulkanBackend,
    no_pool: bool,
    tag: &'static str,
    elems: usize,
) -> Result<SmB> {
    if no_pool {
        Ok(SmB::Own(
            be_.alloc((elems * 4).max(4), BufferUsage::Activations)?,
        ))
    } else {
        Ok(SmB::Pool(pooled(pool, be_, tag, elems * 4)?))
    }
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
#[cfg_attr(infr_profile, infr_prof::instrument)]
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

#[cfg_attr(infr_profile, infr_prof::instrument)]
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
#[cfg_attr(infr_profile, infr_prof::instrument)]
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
    // Multi-row mmv quantized-activation memo: `Some((x, m, in_f))` when the pooled `mmvr_*`
    // scratch already holds `quant_q8(x)` for that shape. Consecutive same-x Linears (fused-QKV
    // offset GEMVs, gate+up) then skip the requant — which not only drops ~40% of the quant
    // dispatches but, more importantly, removes the WAR hazard (quant rewrites qa → barrier
    // against the previous mmv's read) that would otherwise serialize the sibling GEMVs.
    // `take()`n at the top of every call: ONLY the mmv-mrow branch restores it, so any other op
    // in between (which may rewrite `x` — e.g. next layer's RmsNorm into the same scratch id)
    // invalidates the memo by construction.
    mmv_memo: &mut Option<(TensorId, usize, usize)>,
    dummy: &dyn Buffer,
) -> Result<()> {
    let memo_prev = mmv_memo.take();
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
        Op::RmsNormAdd {
            x,
            weight,
            dst,
            rows,
            dim,
            eps,
        } => {
            rec.rmsnorm_add(
                r(*x)?,
                r(*weight)?,
                r(*dst)?,
                *rows as usize,
                *dim as usize,
                *eps,
            );
        }
        // Row-wise softmax over `dim` columns (diffusion-gemma's in-graph self-conditioning — see
        // docs/DIFFUSIONGEMMA.md's Phase-B and the reference's `dg_canvas_embed`). `scale_buf`
        // (Some only on the DiffusionGemma denoise SC path — see its doc in `infr_core::graph`)
        // reads the scale from a device buffer instead of the push-constant `scale` field, so this
        // cached/replayed plan can vary the softmax temperature every step without rebuilding.
        Op::Softmax {
            x,
            dst,
            rows,
            dim,
            scale,
            scale_buf,
        } => match scale_buf {
            Some(sb) => rec.softmax_dyn(r(*x)?, r(*sb)?, r(*dst)?, *rows as usize, *dim as usize),
            None => rec.softmax(r(*x)?, r(*dst)?, *rows as usize, *dim as usize, *scale),
        },
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
            // INFR_PROF2 sub-attribution (no-op otherwise): the vocab-sized GEMMs (lm_head
            // logits, out_f = vocab; DiffusionGemma's SC soft-embedding, in_f = vocab) are 10-100x
            // the FLOPs of a per-layer projection but land in the same "matmul_proj" bucket,
            // which made the bucket un-actionable during the DG slice-7 comparative attribution
            // (llama.cpp's perf logger itemizes per shape; ours didn't). 65536 cleanly separates
            // every real vocab (gemma 262144, qwen 151936) from every hidden/FFN width.
            if out_f >= 65536 {
                rec.label_next("lin_vocab_out");
            } else if in_f >= 65536 {
                rec.label_next("lin_vocab_in");
            }
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
                    // Multi-warp dp4a decode GEMV (wave32-native GPUs) takes precedence — see
                    // `mmv_mw_choice`. AMD returns None here and falls to the scalar/old-mmv path.
                    if let Some(warps) = mmv_mw_choice(be_.caps(), dt, in_f, out_f) {
                        let nblk = in_f / 32;
                        let qa = pooled(pool, be_, "mmv_qa", in_f)?;
                        let dact = pooled(pool, be_, "mmv_dact", nblk * 2)?;
                        let sact = pooled(pool, be_, "mmv_sact", nblk * 2)?;
                        rec.quant_q8(
                            xb,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            1,
                            in_f,
                        );
                        rec.linear_mmv_mw(
                            dt,
                            warps,
                            w,
                            0,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            Some(rr),
                            yf,
                            in_f,
                            out_f,
                        );
                    } else if mmv_decode_enabled()
                        && be_.caps().i8_dot
                        && in_f % 32 == 0
                        && in_f * out_f >= MMV_MIN_ELEMS
                        && crate::gemm::native_mmv_build_spv(dt, true).is_some()
                    {
                        let nblk = in_f / 32;
                        let qa = pooled(pool, be_, "mmv_qa", in_f)?;
                        let dact = pooled(pool, be_, "mmv_dact", nblk * 2)?;
                        let sact = pooled(pool, be_, "mmv_sact", nblk * 2)?;
                        rec.quant_q8(
                            xb,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            1,
                            in_f,
                        );
                        rec.linear_add_mmv(
                            dt,
                            w,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            rr,
                            yf,
                            in_f,
                            out_f,
                        );
                    } else {
                        rec.linear_add_native(dt, w, xb, rr, yf, m, in_f, out_f);
                    }
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
                // Int8 dp4a multi-row GEMV (Q4_K/Q6_K): quantize the m activation rows once
                // (`quant_q8`), then integer-dot the raw weight blocks against ALL rows — the
                // dequant mrow's per-sub-block scalar byte-extract dequant is the m-batch GEMV
                // cost on ALU-bound shapes (E2B pp4@d4096: GEMV class 21.0 → 17.0us/op, pp4
                // 366 → 383 t/s). Gates:
                //  • m >= 3: m=2 is the single-head MTP spec-verify shape, whose accept loop
                //    holds a hard output-identical-to-target-greedy bar (`mtp_spec_matches_
                //    target_only_greedy`) — int8-quantized activations perturb the verify logits
                //    enough to risk argmax flips vs the m=1 decode GEMV, so that shape keeps the
                //    f32-dequant mrow.
                //  • weight >= 8M elements: below that the kernel is dispatch-latency-floor
                //    bound (the int8 math saves nothing) and the extra quant_q8 bubble is a pure
                //    loss — qwen3-0.6b (all projections < 8M) measured pp4@d4096 934 → 854 t/s
                //    ungated, and E2B's own win came from its gu/down (9.4-18.9M elems).
                //  • out_f < 65536: the vocab-sized lm_head GEMV is pure weight-bandwidth (the
                //    dequant ALU hides fully behind the stream) — the int8 form measured a hair
                //    SLOWER there (qwen3-0.6b pp4@d4096 931 vs 916 with lm_head included).
                // Same INFR_NO_MMV escape as the m=1 mmv (A/B).
                if m >= 3
                    && be_.caps().i8_dot
                    && in_f * out_f >= 8 << 20
                    && out_f < 65536
                    && crate::gemm::native_mmv_mrow_build_spv(dt).is_some()
                    && std::env::var("INFR_NO_MMV").is_err()
                {
                    let nblk = in_f / 32;
                    let qa = pooled(pool, be_, "mmvr_qa", m * in_f)?;
                    let dact = pooled(pool, be_, "mmvr_dact", m * nblk * 2)?;
                    let sact = pooled(pool, be_, "mmvr_sact", m * nblk * 2)?;
                    // Same x already quantized by the immediately-preceding Linear(s)? Reuse it
                    // (see `mmv_memo`'s doc — the take()/restore protocol guarantees nothing
                    // rewrote x or the pooled qa in between).
                    if memo_prev != Some((*x, m, in_f)) {
                        rec.quant_q8(
                            xb,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            m,
                            in_f,
                        );
                    }
                    *mmv_memo = Some((*x, m, in_f));
                    rec.linear_mmv_mrow(
                        dt,
                        w,
                        w_off,
                        pool[&qa].as_ref(),
                        pool[&dact].as_ref(),
                        pool[&sact].as_ref(),
                        y,
                        m,
                        in_f,
                        out_f,
                    );
                    return Ok(());
                }
                rec.linear_native_mrow(dt, w, w_off, xb, y, m, in_f, out_f);
                return Ok(());
            }
            let gemm_ok = m > 1 && out_f % 64 == 0 && in_f % 32 == 0;
            // The whole GEMM tier below (matmul_native*/matmul_proj, the coopmat tiled prefill
            // GEMM) dispatches ONLY coopmat SPIR-V — RADV still executes it on hardware without
            // the feature (silent, no fault), so gating is required, not optional (issue: coopmat
            // dispatch on a device that lacks VK_KHR_cooperative_matrix segfaults on some
            // drivers). `!caps.f16_coopmat` routes to the `else` arms below instead
            // (`linear_native_off`/`linear`/`linear_f32` — the SAME scalar dequant-in-shader
            // kernels the m==1 decode GEMV already uses, generalized to `rows=m`; they dispatch
            // `rows*out_f` (or row-tiled) workgroups with no upper bound on `rows`, so they are a
            // drop-in, just without the coopmat tile's weight-reuse-across-64-rows win). This is
            // the coopmat->scalar fallback tier; bit-identical to before when caps are all true.
            let is_gemm = gemm_ok
                && be_.caps().f16_coopmat
                && (native_dense_supported(dt) || matches!(dt, infr_core::DType::F16));
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
                // int8 cooperative-matrix (WMMA) prefill GEMM — MEASUREMENT path, Q8_0 only
                // (crates/infr-vulkan/shaders/native_gemm_i8cm_q8_0.comp). `caps.i8_coopmat` is
                // hardware detection only (see its doc); this dispatch ALSO requires
                // `INFR_I8_COOPMAT=1` (default off) because int8 coopmat hung the GPU on an older
                // Mesa despite enumerating fine there too (commit ad82a77) — the toggle keeps that
                // regression class opt-in until a Mesa-version-gated default is warranted. Default
                // behavior (unset) is completely unaffected: this is a new, additive branch ahead
                // of the existing Q4_K-mmq / warp-coopmat / off-tier arms below, which are
                // untouched.
                // fp8 (E4M3) cooperative-matrix (WMMA) prefill GEMM — Q8_0 only
                // (crates/infr-vulkan/shaders/native_gemm_f8cm_q8_0.comp). `caps.f8_coopmat` is
                // hardware enumeration only (RDNA4/Navi44 confirmed, see lib.rs `has_f8_coopmat`);
                // this dispatch ALSO requires `INFR_F8_COOPMAT=1` (default off) because — unlike
                // `i8cm_ok` below — this kernel hasn't been run/measured on real fp8-coopmat
                // hardware at all yet (this dev box has none), so the opt-in stays until an RDNA4
                // pass validates correctness. Checked AHEAD of `i8cm_ok`: both are Q8_0-only
                // measurement tiers, so when a caller opts into fp8 it takes priority. Default
                // behavior (unset) is completely unaffected — new, additive branch ahead of the
                // i8cm / Q4_K-mmq / warp-coopmat / off-tier arms below, which are untouched.
                //
                // Shape gate mirrors `warp_ok`'s f16 warptile pick: WIDE (BN=256, BK=32) needs
                // out_f%256==0 && in_f%32==0; NARROW_N (BN=128, BK=64) needs out_f%128==0 &&
                // in_f%64==0 for shapes the wide tile can't cover (n%128 not n%256 — see
                // native_gemm_f8cm_q8_0.comp's -DNARROW_N). Neither divides (e.g. out_f%128!=0, or
                // out_f%128==0 with in_f%64!=0 so narrow's BK=64 can't stage either) -> f8cm_ok is
                // false and the shape falls through to i8cm/mmq/native below, unaffected.
                let f8_wide = out_f % 256 == 0 && in_f % 32 == 0;
                let f8_narrow = !f8_wide && out_f % 128 == 0 && in_f % 64 == 0;
                let f8cm_ok = matches!(dt, infr_core::DType::Q8_0)
                    && be_.caps().f8_coopmat
                    && (f8_wide || f8_narrow)
                    && std::env::var("INFR_F8_COOPMAT").is_ok();
                let i8cm_ok = matches!(dt, infr_core::DType::Q8_0)
                    && be_.caps().i8_coopmat
                    && std::env::var("INFR_I8_COOPMAT").is_ok();
                // NATIVE bf16 cooperative-matrix (WMMA) prefill GEMM — the `-DBF16CM` build of the
                // SAME production kernel (crates/infr-vulkan/shaders/native_gemm_warp.comp) that
                // `native_gemm_warp_bf16` (the f16-clamped path below) already uses; only the
                // coopmat A/B operand type differs, so it should match that kernel's speed while
                // keeping bf16's full exponent range. `caps.bf16_coopmat` is hardware enumeration
                // only (RDNA4/Navi44 confirmed, see lib.rs `has_bf16_coopmat`); this dispatch ALSO
                // requires `INFR_BF16_COOPMAT=1` (default off) because this variant hasn't been
                // run/measured on real bf16-coopmat hardware at all yet (this dev box has none), so
                // the opt-in stays until an RDNA4 pass validates correctness. Default behavior
                // (unset) is completely unaffected: Bf16 keeps routing through the existing
                // `native_dense_supported` arm below (`native_gemm_warp_bf16`, the f16-clamped
                // path), byte-identical to before this branch existed. Shape gate mirrors
                // `f8_wide`/`f8_narrow`: WIDE (BN=256, BK=32) needs out_f%256==0 && in_f%32==0;
                // NARROW_N (BN=128, BK=64) needs out_f%128==0 && in_f%64==0.
                let bf16cm_wide = out_f % 256 == 0 && in_f % 32 == 0;
                let bf16cm_narrow = !bf16cm_wide && out_f % 128 == 0 && in_f % 64 == 0;
                let bf16cm_ok = matches!(dt, infr_core::DType::Bf16)
                    && be_.caps().bf16_coopmat
                    && (bf16cm_wide || bf16cm_narrow)
                    && std::env::var("INFR_BF16_COOPMAT").is_ok();
                if f8cm_ok {
                    // E4M3's range is tiny (max normal 448) — unscaled f32 activations overflow to
                    // inf/NaN on cast, which is what produced garbage output on the first RDNA4
                    // run. `quant_f8_row` pre-scales activations into E4M3's range (one amax/scale
                    // per row) before the coopmat GEMM, which descales the output by that same
                    // per-row scale in its epilogue. Q8_0 weights are unscaled (see
                    // native_gemm_f8cm_q8_0.comp doc — their post-dequant magnitude is expected to
                    // already fit E4M3).
                    let qa = pooled(pool, be_, "f8cm_qa", m * in_f)?;
                    let srow = pooled(pool, be_, "f8cm_srow", m * 4)?;
                    rec.quant_f8_row(xb, pool[&qa].as_ref(), pool[&srow].as_ref(), m, in_f);
                    // INFR_F8_PREPACK=1 (requires INFR_F8_COOPMAT=1 too, since it only takes
                    // effect inside this arm): bakes the Q8_0 block scale into an E4M3 weight
                    // buffer ONCE via `repack_q8_to_f8`, then the GEMM's Bs staging reads that
                    // buffer DIRECTLY — no in-shader dqblk. Isolates whether removing the
                    // dequant-ALU bottleneck (the SAME cost f16 pays; fp8-dqblk measured 0.73x
                    // f16 on RDNA4) lets fp8's 2x WMMA rate win. The E4M3 buffer is pooled
                    // per-shape scratch (like every other tier here) and re-repacked on EVERY
                    // forward call — a real deployment would cache the repack at load time, not
                    // redo it per Linear; this is a per-op profiling isolation path, not the
                    // proposed production shape. Unset (default): unchanged dqblk path below,
                    // byte-identical to before this branch existed.
                    if std::env::var("INFR_F8_PREPACK").is_ok() {
                        let f8_w8 = pooled(pool, be_, "f8_w8", out_f * in_f)?;
                        rec.repack_q8_to_f8(w, w_off, pool[&f8_w8].as_ref(), out_f, in_f);
                        rec.matmul_f8cm_q8_0_prepacked(
                            pool[&qa].as_ref(),
                            pool[&srow].as_ref(),
                            pool[&f8_w8].as_ref(),
                            out,
                            m,
                            in_f,
                            out_f,
                        );
                    } else {
                        rec.matmul_f8cm_q8_0(
                            pool[&qa].as_ref(),
                            pool[&srow].as_ref(),
                            w,
                            w_off,
                            out,
                            m,
                            in_f,
                            out_f,
                        );
                    }
                } else if i8cm_ok {
                    // "Idea 2" measurement (INFR_I8_ROW_SCALE=1, requires INFR_I8_COOPMAT=1 too):
                    // whole-row (block-invariant) activation scale instead of quant_q8's
                    // per-32-block scale — see native_gemm_i8cm_q8_0.comp #ifdef ROW_SCALE /
                    // quant_q8_row.comp. Separate pool tag (different buffer size/layout) and a
                    // separate kernel pair, so this can be measured and reverted independently of
                    // the i8cm baseline path above.
                    if std::env::var("INFR_I8_ROW_SCALE").is_ok() {
                        let qa = pooled(pool, be_, "i8cm_qa_row", m * in_f)?;
                        let dact_row = pooled(pool, be_, "i8cm_dact_row", m * 2)?;
                        rec.quant_q8_row(xb, pool[&qa].as_ref(), pool[&dact_row].as_ref(), m, in_f);
                        rec.matmul_i8cm_q8_0_rowscale(
                            pool[&qa].as_ref(),
                            pool[&dact_row].as_ref(),
                            w,
                            w_off,
                            out,
                            m,
                            in_f,
                            out_f,
                        );
                    } else {
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
                        rec.matmul_i8cm_q8_0(
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            w,
                            w_off,
                            out,
                            m,
                            in_f,
                            out_f,
                        );
                    }
                } else if bf16cm_ok {
                    rec.matmul_bf16cm(xb, w, w_off, out, m, in_f, out_f);
                } else if matches!(dt, infr_core::DType::Q4K)
                    && !warp_ok
                    && be_.caps().i8_dot
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
                    // F16 deep-k narrow-n → SPLIT-K warptile (DG slice-7 comparative
                    // attribution): the SC soft-embedding GEMM (m=canvas=256, k=vocab=262144,
                    // n=ne=2816, f16 weight) measured 58ms/step at ~6.5 TFLOPS on the legacy
                    // `gemm_proj` BN=64 tile below (28% of the whole denoise step) vs llama.cpp's
                    // 14.1ms @ 27 TFLOPS for the IDENTICAL shape — the single biggest
                    // infr-vs-fork per-stage delta. Same narrow-grid split-K policy as the
                    // native-quant branch above (out_f%128, deep k, grid < 128 wgs); the F16
                    // decode is exact (see native_decode.glsl FMT_F16), so numerics match the
                    // f16 coopmat route up to accumulation order. Everything that doesn't hit
                    // the split-K window keeps the `matmul_proj` route unchanged.
                    // Same splits policy as the quant branch above. Probed deeper splits at the
                    // SC shape (splits=4/8/16 via a temporary env override): the op shaved
                    // 30.0 -> 25.8-28.4ms but dg-step stayed flat (1100-1107, within noise), so
                    // the shared formula stays — no bespoke tuning knob.
                    let narrow_grid = m.div_ceil(64) * (out_f / 128).max(1);
                    let splits = if out_f % 128 == 0 && in_f >= 1024 && narrow_grid < 128 {
                        (256 / narrow_grid).next_power_of_two().clamp(1, 8)
                    } else {
                        1
                    };
                    if matches!(dt, infr_core::DType::F16)
                        && splits > 1
                        && crate::gemm::native_gemm_warp_sk_build_spv(dt).is_some()
                        && std::env::var("INFR_NO_GEMM_WARP").is_err()
                    {
                        let mpad = m.div_ceil(64) * 64;
                        let pk = pooled(pool, be_, "splitk_part", splits * mpad * out_f * 4)?;
                        rec.matmul_native_splitk(
                            dt,
                            xb,
                            w,
                            w_off,
                            pool[&pk].as_ref(),
                            out,
                            m,
                            in_f,
                            out_f,
                            splits,
                            false,
                        );
                    } else {
                        // f16 coopmat GEMM (dummy scales/mins unused at bits=16).
                        rec.matmul_proj(xb, w, dummy, dummy, out, m, in_f, out_f, 16, 0);
                    }
                }
                if let Some(t) = tmp {
                    rec.copy(t.as_ref(), 0, y, 0, m * out_f * eb);
                    transient.push(t);
                }
            } else if native_dense_supported(dt) {
                // Decode (m=1) on int-dot-capable K-quants → mmv (see the fused-add branch above).
                // Wave32-native GPUs take the multi-warp dp4a route first (`mmv_mw_choice`); AMD
                // falls through to the scalar GEMV (default) or the old mmv (INFR_MMV_DECODE=1).
                let mw = if m == 1 {
                    mmv_mw_choice(be_.caps(), dt, in_f, out_f)
                } else {
                    None
                };
                if let Some(warps) = mw {
                    let nblk = in_f / 32;
                    let qa = pooled(pool, be_, "mmv_qa", in_f)?;
                    let dact = pooled(pool, be_, "mmv_dact", nblk * 2)?;
                    let sact = pooled(pool, be_, "mmv_sact", nblk * 2)?;
                    rec.quant_q8(
                        xb,
                        pool[&qa].as_ref(),
                        pool[&dact].as_ref(),
                        pool[&sact].as_ref(),
                        1,
                        in_f,
                    );
                    rec.linear_mmv_mw(
                        dt,
                        warps,
                        w,
                        w_off,
                        pool[&qa].as_ref(),
                        pool[&dact].as_ref(),
                        pool[&sact].as_ref(),
                        None,
                        y,
                        in_f,
                        out_f,
                    );
                } else if m == 1
                    && mmv_decode_enabled()
                    && be_.caps().i8_dot
                    && in_f % 32 == 0
                    && in_f * out_f >= MMV_MIN_ELEMS
                    && crate::gemm::native_mmv_build_spv(dt, false).is_some()
                {
                    let nblk = in_f / 32;
                    let qa = pooled(pool, be_, "mmv_qa", in_f)?;
                    let dact = pooled(pool, be_, "mmv_dact", nblk * 2)?;
                    let sact = pooled(pool, be_, "mmv_sact", nblk * 2)?;
                    rec.quant_q8(
                        xb,
                        pool[&qa].as_ref(),
                        pool[&dact].as_ref(),
                        pool[&sact].as_ref(),
                        1,
                        in_f,
                    );
                    rec.linear_mmv(
                        dt,
                        w,
                        w_off,
                        pool[&qa].as_ref(),
                        pool[&dact].as_ref(),
                        pool[&sact].as_ref(),
                        y,
                        in_f,
                        out_f,
                    );
                } else {
                    rec.linear_native_off(dt, w, w_off, xb, y, m, in_f, out_f);
                }
            } else if matches!(dt, infr_core::DType::F32) {
                // Full-precision projection weight (gemma4 E2B per-layer inp_gate/proj): the seam
                // uploads native dtype, and the f16 GEMV would read the f32 bytes as f16 garbage.
                rec.linear_f32(w, xb, y, m, in_f, out_f);
            } else if be_.caps().f16 {
                rec.linear(w, xb, y, m, in_f, out_f);
            } else {
                // No shaderFloat16 (implies no coopmat too — f16 is a coopmat prerequisite):
                // linear_f16.comp's `float16_t` SSBO read needs the Float16 SPIR-V capability the
                // device lacks. linear_f16_noext.comp is the same dispatch shape reading the f16
                // weight buffer as packed u32 words and unpacking via the CORE `unpackHalf2x16`
                // builtin instead — correctness-first, not perf-tuned (no row-tiling).
                rec.linear_f16_noext(w, xb, y, m, in_f, out_f);
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
        Op::MulVec {
            x,
            vec,
            dst,
            rows,
            n,
        } => rec.mul_vec(r(*x)?, r(*vec)?, r(*dst)?, *rows as usize, *n as usize),
        Op::MoeSharedExpertAdd {
            moe,
            shexp,
            gate,
            dst,
            rows,
            n,
        } => rec.moe_shared_expert_add(
            r(*moe)?,
            r(*shexp)?,
            r(*gate)?,
            r(*dst)?,
            *rows as usize,
            *n as usize,
        ),
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
        // arises with Gelu (gemma); silu/sigmoid are always up_off==0. up_stride enables per-row
        // strided reads from a wider buffer (eliminates the CopyStrided dispatch per layer).
        Op::GatedAct {
            gate,
            up,
            dst,
            rows,
            nff,
            act,
            up_off,
            up_stride,
        } => {
            let n = *rows as usize * *nff as usize;
            let (g_, u_, y) = (r(*gate)?, r(*up)?, r(*dst)?);
            let eb = graph.desc(*up).dtype.dense_bytes(1).unwrap_or(4);
            match act {
                Activation::Silu => {
                    if *up_off != 0 || *up_stride != 0 {
                        return Err(be(
                            "vulkan adapter: GatedAct Silu up_off/stride!=0 unsupported",
                        ));
                    }
                    rec.silu_mul(g_, u_, y, n);
                }
                Activation::Sigmoid => {
                    if *up_off != 0 || *up_stride != 0 {
                        return Err(be(
                            "vulkan adapter: GatedAct Sigmoid up_off/stride!=0 unsupported",
                        ));
                    }
                    rec.mul_sigmoid(u_, g_, y, n);
                }
                Activation::Gelu => {
                    rec.gelu_mul_off(
                        g_,
                        u_,
                        *up_off as usize * eb,
                        *up_stride as usize * eb,
                        *nff as usize,
                        y,
                        n,
                    );
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
            x_stride,
            ..
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
                    if *x_stride > 0 && ff.is_none() {
                        // Interleaved q+g buffer: read query with stride, skip CopyStrided per head.
                        rec.qk_norm_rope_interleaved(
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
                            *x_stride as usize,
                        );
                    } else {
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
                    } // x_stride > 0
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
            ..
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
            // split (attn_partial_{k,v}q8) and prefill through a dequant→f16 prepass. A DECOUPLED
            // Q8 side forces STATIC decode (see decode_eligible); the Dynamic branch below sees
            // only coupled K==V==Q8 (or no Q8 at all).
            let k_q8 = matches!(graph.desc(*k_cache).dtype, infr_core::DType::Q8_0);
            let v_q8 = matches!(graph.desc(*v_cache).dtype, infr_core::DType::Q8_0);
            let kv_q8 = k_q8 && v_q8; // coupled (the Dynamic-branch kernels' q8 variant)
                                      // Planar Q8 scales region base = total cache elements (K and V caches share numel).
            let cap = graph.desc(*k_cache).numel();
            if let RopeMode::Dynamic(params) = mode {
                // Eligibility guarantees rows==1. Scale rides a push constant (gemma4 uses 1.0;
                // 0.0 → kernel default 1/√hd) and SWA windows ride the window-aware prologue +
                // partial kernel, so gemma-family decode replays too.
                let window = match mask {
                    AttnMask::Causal => 0usize,
                    AttnMask::SlidingWindow(w) => *w,
                    // Record-once decode replay is rows==1 causal-only (DiffusionGemma canvas
                    // denoise is a rows=C static forward, never eligible for this path) — never
                    // actually reached with Canvas; 0 is an arbitrary but harmless placeholder.
                    AttnMask::Canvas { .. } => 0usize,
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
                // NOTE `cap_rows` (row capacity, chunk sizing) vs `cap` (ELEMENT capacity, the
                // planar-Q8 scales-region base push constant): shadowing the latter with the
                // former here is exactly the bug that garbled Q8 KV under record-once replay.
                let cap_rows = graph.desc(*k_cache).numel() / (nkv * hd);
                // Baked MIN chunk (64, rising only when the capacity would exceed the 1024-chunk
                // scratch/combine cap); the kernel derives the effective adaptive chunk from the
                // live kv_len each token, so one recorded plan serves every depth.
                let chunk = cap_rows.div_ceil(1024).max(64);
                let n_chunks = cap_rows.div_ceil(chunk);
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
                // Prefill, causal, hd==128, standard 1/√hd scale: FlashAttention-2 (split-K
                // online softmax, no materialized [m,kv] scores). The flash kernel hardcodes
                // 1/√hd and reads/writes ceil(rows/64)*64 q/dst rows, so guard the scale and
                // require 64-row-padded buffers (Internal). Flash IS the K-tile shape — K/V
                // streamed once per row-tile, padded rows only waste ALU — so at depth it beats
                // the per-row split kernel well below a full 64-row tile. Measured crossover
                // (0.6B, 7900 XTX): flash wins from rows>=24 at kv>=8192 (pp24@8k +13%,
                // pp32@16k +55%) but LOSES to split at kv=4096 up through rows=32 — hence the
                // two-tier floor. INFR_FLASH_MIN_ROWS overrides the deep-kv tier (A/B).
                let flash_min_rows: usize = std::env::var("INFR_FLASH_MIN_ROWS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(24);
                // attn_flash*/attn_qk*/attn_pv* are ALL coopmat SPIR-V (see the module doc / audit
                // for this fallback ladder) — `flash_ok`/`nonfa_ok` additionally require
                // `caps.f16_coopmat` so a device without the feature never gets one
                // dispatched (RADV silently executes coopmat SPIR-V even with the feature bit
                // off, so "it ran" isn't proof; other drivers fault). Gated false, both fall
                // through to `split_ok`/the final `else` — `attention_kv_split`/`attention_kv`,
                // already the scalar decode/short-prefill path, dispatch with no coopmat and no
                // row-count ceiling, so they're a correct (if slower) fallback for ANY `rows`.
                let flash_ok = (rows >= 64 || (rows >= flash_min_rows && kv_len >= 8192))
                    && hd == 128
                    && matches!(mask, AttnMask::Causal)
                    && (*scale - 1.0 / (hd as f32).sqrt()).abs() < 1e-6
                    && !(k_q8_eff || v_q8_eff)
                    && be_.caps().f16_coopmat
                    && matches!(graph.tensors[q.0 as usize].kind, TensorKind::Internal)
                    && matches!(graph.tensors[dst.0 as usize].kind, TensorKind::Internal);
                // Prefill at hd≠128 (qwen35/gemma hd=256): the non-FA coopmat pipeline
                // (attn_qk → softmax → attn_pv) is hd-general and ~an order faster than the scalar
                // attention_kv. Needs 64-row-padded q/dst (Internal buffers are row-padded), so
                // require both to be Internal.
                // DiffusionGemma canvas denoise (docs/DIFFUSIONGEMMA.md): bidirectional, fixed
                // `[lo, kv_len)` reach per row — neither the flash kernel (Causal-only, already
                // excluded above) nor `attention_prefill_nonfa`'s per-row causal-end window
                // understand that shape, so gate BOTH off it and force the split-K path below
                // (which DOES carry a `lo` override) even though rows=C(=256) would otherwise
                // pick one of these tiers. Perf pass deferred (Phase 4).
                let canvas_lo = match mask {
                    AttnMask::Canvas { lo } => Some(*lo),
                    _ => None,
                };
                let nonfa_ok = !flash_ok
                    && canvas_lo.is_none()
                    && rows >= 64
                    && hd % 64 == 0
                    && hd <= 512
                    && !(k_q8_eff || v_q8_eff)
                    && be_.caps().f16_coopmat
                    && matches!(graph.tensors[q.0 as usize].kind, TensorKind::Internal)
                    && matches!(graph.tensors[dst.0 as usize].kind, TensorKind::Internal);
                // Decode (rows==1) AND small-m suffix prefill (rows 2..63, below the flash/nonfa
                // floor): flash-decoding split-K — each (row, head)'s KV range splits across ~32
                // chunks of workgroups instead of the scalar attention_kv's rows*nh (= nh at
                // decode, ~16 workgroups on a 96-CU GPU; the SHORT TURN suffix shape measured
                // ~2.5ms/layer scalar = 97% of the forward at d4096). attn_partial handles any
                // hd%4==0 ≤ 512 (hd=128 fast path, general path above), any scale (push constant;
                // gemma4 uses 1.0), SWA windows, and per-row causal ends (row r attends
                // ≤ pos + r). Native Q8 reads are decode-only (rows>1 already dequanted to the
                // f16 scratch above, so k/v_q8_eff are false there by construction).
                // Rows-BATCHED split tier (rows 12..flash-floor at deep kv): one workgroup per
                // (head, chunk) streams K/V once per 4-row group through attn_partial_mrows_c256
                // (7KB LDS). Measured wins over the per-row grid: pp12/16/20@d16384
                // 741->799/822->924/882->1012 t/s, pp16@8k a wash — below rows=12 or kv=8192
                // the per-row grid's extra workgroups fill the DRAM queue better than the
                // bandwidth saving pays. hd<=128 (one q vec4 per lane per row); q8 never reaches
                // rows>1 (dequanted above). INFR_MROWS_ATTN=1 forces the batched tier on (tests /
                // A/B), INFR_NO_MROWS_ATTN forces it off.
                let batched_attn = rows >= 2
                    && hd <= 128
                    && canvas_lo.is_none()
                    && !(k_q8_eff || v_q8_eff)
                    && std::env::var("INFR_NO_MROWS_ATTN").is_err()
                    && ((rows >= 12 && kv_len >= 8192) || std::env::var("INFR_MROWS_ATTN").is_ok());
                // The batched kernel stages chunk scores in 4KB of LDS → chunk 256; the per-row
                // grid keeps the adaptive ~32-chunks policy.
                //
                // Canvas (DiffusionGemma denoise, slice 7 comparative-attribution against the
                // fork's fused flash-attn oracle): the ordinary `kv_len/32` policy is tuned for
                // DEEP decode contexts (splitting a multi-thousand kv_len across ~32 workgroups
                // keeps each one short); at canvas's shallow kv_len (prompt + canvas, a few
                // hundred) it still floors to 64, forcing 5 chunks at kv_len=283 —
                // `attention_kv_split`'s `pm`/`pl`/`pacc` partials are [rows(=canvas), nh,
                // n_chunks, hd], so every extra chunk is 256(canvas)×nh more partial-softmax
                // writes AND combine-side reads, pure overhead vs. a genuine single-pass flash
                // kernel (which the canvas mask shape can't use — see `canvas_lo` above).
                //
                // Probed the chunk COUNT directly (`INFR_CANVAS_CHUNK_N`, kv_len=283, Vulkan0):
                // n=2 (the `split_ok`-minimum, `kv_len>chunk` strictly) REGRESSED — attn_partial
                // 11.7k->13.8us (occupancy loss: workgroups halve from 5 to 2, so each one's
                // longer serial K/V loop outweighs the partial-buffer saving) even though
                // attn_combine dropped (1.36k->0.84k). n=3 was the sweet spot: attn_partial
                // ~11.2k (flat vs baseline) + attn_combine ~1.0k (-27%) = net -6% on the
                // attn_partial+combine pair vs the old 64-floor's n=5. n=4 landed between the two
                // (small net win, less than n=3). Default 3; overridable for future re-tuning at
                // other canvas lengths.
                // SWA chunk-grid base: a sliding-window layer's rows attend only the union span
                // `[max(0, pos+1-window), kv_len)` (row r's window start only moves UP with r), so
                // chunk THAT span instead of `[0, kv_len)` — at pp4@d4096 a gemma SWA layer
                // (window 512) drops from 33 chunks (28 of them empty: launched, barriered,
                // zero-partial-written, combine-read) to ~9. The split-K partial pass is
                // fixed-cost-per-workgroup bound at small m (E2B attn_partial scaled exactly with
                // nh·n_chunks·rows across depths), so empty chunks cost like full ones.
                // `attn_partial.comp`'s static branch derives the SAME base from its existing
                // pos/window push constants (no layout change); the rows-batched mrows kernel and
                // Canvas don't, so they keep the full-range grid (base 0).
                let swa_window = match mask {
                    AttnMask::SlidingWindow(w) => *w,
                    _ => 0,
                };
                let swa_base = if swa_window > 0 && !batched_attn && canvas_lo.is_none() {
                    (pos + 1).saturating_sub(swa_window)
                } else {
                    0
                };
                let span = kv_len - swa_base;
                let chunk = if canvas_lo.is_some() && kv_len >= 2 {
                    let n = std::env::var("INFR_CANVAS_CHUNK_N")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(3)
                        .max(1);
                    kv_len.div_ceil(n).min(kv_len - 1).max(1)
                } else if batched_attn {
                    256
                } else {
                    (span / 32).clamp(64, 512)
                };
                // Canvas forces the split-K tier regardless of row count (see `canvas_lo` above) —
                // `attn_partial` carries the fixed `lo` override this mask needs; flash/nonfa don't.
                let split_ok = (rows < 64 || canvas_lo.is_some())
                    && span > chunk
                    && hd % 4 == 0
                    && hd <= 512
                    && (rows == 1 || !(k_q8_eff || v_q8_eff));
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
                        AttnMask::Canvas { .. } => unreachable!("nonfa_ok excludes Canvas"),
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
                    // `canvas_lo` (computed above) rides a SEPARATE param into
                    // `attention_kv_split` — `window` stays the ordinary causal/SWA value (0 for
                    // Canvas; unused by the shader when `canvas_lo` is set).
                    let window = match mask {
                        AttnMask::Causal => 0,
                        AttnMask::SlidingWindow(w) => *w,
                        AttnMask::Canvas { .. } => 0,
                    };
                    let n_chunks = span.div_ceil(chunk);
                    // Scratch scales with rows (rows*nh partial planes); rows==1 keeps the old
                    // decode sizes, so the hot decode pool entries are unchanged.
                    let pm = pooled(pool, be_, "split_pm", rows * nh * n_chunks * 4)?;
                    let pl = pooled(pool, be_, "split_pl", rows * nh * n_chunks * 4)?;
                    let pacc = pooled(pool, be_, "split_pacc", rows * nh * n_chunks * hd * 4)?;
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
                        rows,
                        pos,
                        kv_len,
                        nh,
                        nkv,
                        hd,
                        chunk,
                        n_chunks,
                        *scale,
                        window,
                        canvas_lo,
                        k_q8_eff,
                        v_q8_eff,
                        cap,
                        batched_attn,
                    );
                } else {
                    let window = match mask {
                        AttnMask::Causal => 0,
                        AttnMask::SlidingWindow(w) => *w,
                        AttnMask::Canvas { .. } => {
                            return Err(be(
                                "vulkan adapter: AttnMask::Canvas requires the split-K attention \
                                 path (split_ok) — the scalar attention_kv fallback doesn't carry \
                                 a bidirectional `lo` override",
                            ));
                        }
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
            ..
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
        // Fused per-head RMSNorm + SiLU gate multiply (qwen35 DeltaNet z-gate) — one dispatch
        // instead of QkNorm→GatedAct's two, removing the read-after-write barrier between them.
        Op::GatedRmsNorm {
            x,
            weight,
            gate,
            dst,
            rows,
            n_head,
            head_dim,
            eps,
        } => {
            rec.rmsnorm_gate(
                r(*x)?,
                r(*weight)?,
                r(*gate)?,
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
            // Prefill AND multi-row batches (rows ≥ 2): the chunkwise delta rule processes up to 32
            // tokens per state traversal (matmuls + a triangular solve) instead of the token-serial
            // recurrence — the difference between rows and ⌈rows/32⌉ sequential state sweeps. The
            // partial-chunk case (rows < 32, one chunk) is the SAME shader path prefill already uses
            // for its tail chunk (c = min(32, rows-base) is masked throughout deltanet_chunked.comp /
            // deltanet_prep.comp), so this isn't a new code path — just a lower row count triggering
            // it. Threshold was 32 (i.e. only prefill); MTP verify runs a batched trunk forward over
            // only m≈2-19 rows (committed_suffix ++ drafted, prefix-diffed) and was stranded on the
            // sequential kernel below 32, capping the whole spec-decode cycle (verify was 68-69% of
            // it). Measured crossover (gemm_bench.rs microbench, qwen35 dims nv=nk=16 kd=vd=128,
            // isolated dispatch cost): chunked_split beats sequential at EVERY row count from 1..32
            // (2.0x at rows=1 up to 8.2x at rows=32; rows=2..8, the MTP-verify range, is 3.2-4.4x).
            // rows=1 stays on the sequential kernel anyway — this is plain single-token decode, out
            // of scope here (decode has its own hot path; the extra transient-buffer allocs below
            // aren't free at decode's much higher call frequency, unlike prep+scan's per-call cost
            // which the microbench doesn't include either but easily amortizes at rows≥2). The
            // real-workload win: MTP 4B, `deltanet` total dropped 116.3ms→~28ms (prep+gates+scan) at
            // an identical (bit-exact) accept pattern across a 128-token run. The default is the
            // SPLIT form (parallel prep/gates passes + a light state-coupled scan; the monolithic
            // kernel duplicated the prep work per column block). INFR_NO_DN_CHUNK forces sequential
            // (A/B, also the m==1-decode behavior); INFR_NO_DN_SPLIT keeps the chunked math but in
            // the monolithic kernel (A/B).
            let (rows_, nv_, nk_, kd_, vd_) = (
                *rows as usize,
                *n_vhead as usize,
                *n_khead as usize,
                *head_k as usize,
                *head_v as usize,
            );
            let chunked = rows_ >= 2 && std::env::var("INFR_NO_DN_CHUNK").is_err();
            // deltanet_chunked_split's prep pass (deltanet_prep.comp) is the ONLY DeltaNet shader
            // using coopmat (the D/Dq dot matrices); deltanet_chunked (monolithic) and deltanet
            // (sequential) are both scalar. `!caps.f16_coopmat` routes to the still-chunked
            // (still fast, just not split-prep) `deltanet_chunked` kernel below instead of the
            // sequential one — chunked math doesn't require coopmat, only this particular prep
            // kernel's implementation does.
            if chunked && be_.caps().f16_coopmat && std::env::var("INFR_NO_DN_SPLIT").is_err() {
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
        Op::EmbedGather {
            ids,
            table,
            dst,
            rows,
            ne,
            scale,
        } => {
            let dt = graph.desc(*table).dtype;
            rec.embed_gather(
                dt,
                r(*table)?,
                r(*ids)?,
                r(*dst)?,
                *rows as usize,
                *ne as usize,
                *scale,
            );
        }
        Op::Sample {
            x,
            u,
            dst,
            n,
            top_k,
            temp,
            top_p,
        } => {
            let cand = pooled(pool, be_, "sample_cand", 2 * 256 * *top_k as usize * 4)?;
            match mode {
                // Record-once/self-advancing path (single-shot `execute` OR chained
                // `execute_chain` — the same recording serves both): `u` is a 64-slot ring keyed
                // by the self-advancing `params[0]`, matching `id_log`'s ring geometry.
                RopeMode::Dynamic(params) => {
                    rec.sample_topk_chain(
                        r(*x)?,
                        pool[&cand].as_ref(),
                        r(*u)?,
                        *params,
                        r(*dst)?,
                        *n as usize,
                        *top_k as usize,
                        *temp,
                        *top_p,
                    );
                }
                // Classic per-execute recording: `u` is the plain 1-float buffer.
                RopeMode::Static(_) => {
                    rec.sample_topk(
                        r(*x)?,
                        pool[&cand].as_ref(),
                        r(*u)?,
                        r(*dst)?,
                        *n as usize,
                        *top_k as usize,
                        *temp,
                        *top_p,
                    );
                }
            }
        }
        Op::Argmax { x, dst, n, rows } => {
            let part = pooled(pool, be_, "argmax_part", 512 * 4)?;
            rec.argmax(
                r(*x)?,
                pool[&part].as_ref(),
                r(*dst)?,
                *n as usize,
                *rows as usize,
            );
        }
        Op::ArgmaxProb {
            x,
            dst_id,
            dst_prob,
            n,
        } => {
            // 3*256 f32 scratch: (max, idx-bits, sum_exp) triples — see argmax_prob.comp.
            let part = pooled(pool, be_, "argmax_prob_part", 768 * 4)?;
            rec.argmax_prob(
                r(*x)?,
                pool[&part].as_ref(),
                r(*dst_id)?,
                r(*dst_prob)?,
                *n as usize,
            );
        }
        // MoE FFN (single token): router GEMV → GPU-resident top-k (softmax-renorm, ×scale) →
        // fused multi-slot expert SwiGLU (gate/up share the row, down reads each slot's act) →
        // weighted accumulate. Mirrors the production GPU-resident decode path (transformer.rs)
        // and the CPU `Op::MoeFfn` interpreter. Expert banks must use an id-native quant format.
        Op::MoeFfn {
            x,
            router_x,
            router,
            gate_exps,
            up_exps,
            down_exps,
            down_scale,
            fused_gate_up,
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
            // Fused gate_up_exps stores BOTH roles per expert ([ne, 2*nff, n_expert]) — the id-native
            // dtype check below reads `up_exps` too, but the call site binds it to the SAME handle as
            // `gate_exps` when fused (never separately read), so the dtype check still holds.
            // `down_exps` is ALWAYS width-`nff` per expert (unaffected by a fused gate/up) — its own
            // stride, used by the per-token path below (the batched path never sees `fused_gate_up`,
            // guarded above, so reusing `stride` there for gate/up/down alike stays correct).
            let stride = if *fused_gate_up { 2 * nff } else { nff } * ne;
            let down_stride = nff * ne;
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
            let rows = graph.desc(*x).numel() / ne;
            // ── BATCHED MoE FFN (rows > 1: the seam's prefill chunks AND DiffusionGemma's canvas
            // denoise): GPU-resident expert routing (top-k → bucket count/scan/scatter, all
            // on-GPU) + a prologue that writes per-expert INDIRECT dispatch args from the counts —
            // so the whole expert loop records with NO host readback (the bespoke path downloads
            // counts mid-graph to size its GEMMs; indirect dispatch replaces that). Gate/up AND
            // down each independently accept Q4_K/Q5_K/Q6_K/Q8_0/Q5_0 (split OR fused gate/up;
            // Q5_0 is what the shipped diffusiongemma-26B-A4B-it-GGUF's down banks use; Q5_K/Q6_K
            // is what unsloth-dynamic Qwen3.6-MoE quants mix into most layers' gate/up/down
            // banks) — every dtype routes through the SAME dtype-generic `matmul_mmq_experts`
            // dp4a kernel table, so there's no per-role dtype restriction beyond that shared set.
            // Codebook quants (IQ*/Q2_K/Q3_K — no dp4a-mmq kernel) route through the per-token
            // path below regardless of role.
            //
            // Fused gate_up (diffusion-gemma): native block formats can't be split at an
            // arbitrary row offset without block-alignment gymnastics (Q4_K's superblocks straddle
            // 256-element runs), so instead of two GEMMs at different `w_off`s we run ONE GEMM
            // over the whole [ne, 2*nff] expert slice (`gu_width` below) and split gate/up in the
            // activation kernel — `gelu_mul_fused`/`silu_mul_fused` already implement exactly that
            // split (gate half first, up half second per row), reused unchanged from the per-token
            // path. Split gate/up (qwen3moe): unchanged two-GEMM shape.
            //
            // Gated on `rows > moe_small_m_threshold()` (not `rows > 1`): tiny prefill chunks (e.g.
            // `pp4`) take the small-m fast path below instead — see its doc for why.
            // ALSO gated on `caps.i8_dot`: the batched path's expert GEMM (`matmul_mmq_experts`,
            // below) is dp4a int8 ONLY — no coopmat, no scalar sibling. A device lacking packed
            // integer dot product falls through to the small-m path unconditionally instead (its
            // `linear_native_id_multi` id-indexed GEMVs are plain dequant-in-shader, no dp4a) —
            // correct for any `rows`, just without the batched path's cross-token weight-bank
            // reuse (a real perf cost for large-batch MoE prefill on such hardware).
            if rows > moe_small_m_threshold() && be_.caps().i8_dot {
                use infr_core::DType::{Q4K, Q5K, Q5_0, Q6K, Q8_0};
                // `matmul_mmq_experts` is dtype-generic (dp4a mmq kernels exist for all five of
                // these — that's why `down_ok` already covered the wider set) and role-agnostic
                // (gate/up/down all call the SAME function, just with a different weight handle
                // and stride) — so gate/up get the SAME coverage as down, not just Q4_K. Codebook
                // quants (IQ*/Q2_K/Q3_K) have no dp4a-mmq kernel at all and stay out of this set;
                // those experts keep falling through to the per-token path below.
                let mmq_ok = |dt| matches!(dt, Q4K | Q5K | Q6K | Q8_0 | Q5_0);
                let down_ok = mmq_ok(ddt);
                let act_ok = if *fused_gate_up {
                    matches!(act, Activation::Silu | Activation::Gelu)
                } else {
                    // qwen3moe (the only split-gate_up batched caller) ships SiLU only; a non-
                    // fused GELU batched kernel doesn't exist (no caller needs it today).
                    matches!(act, Activation::Silu)
                };
                if !(mmq_ok(gdt) && mmq_ok(udt) && down_ok && act_ok) {
                    return Err(be(format!(
                        "vulkan adapter: batched MoeFfn needs Q4_K/Q5_K/Q6_K/Q8_0/Q5_0 gate/up \
                         + Q4_K/Q5_K/Q6_K/Q8_0/Q5_0 down (+ SiLU, or GELU when fused_gate_up) \
                         (got gate={gdt:?} up={udt:?} \
                         down={ddt:?} act={act:?} fused={fused_gate_up})"
                    )));
                }
                let al = |n: usize| be_.alloc((n * 4).max(4), BufferUsage::Activations);
                let alu = |n: usize| be_.alloc_uninit(n.max(4), BufferUsage::Activations);
                let n_pairs = rows * n_used;
                let xb = r(*x)?;
                let yb = r(*dst)?;
                // diffusion-gemma's router reads a DIFFERENTLY normalized/scaled row than the
                // experts (see the `Op::MoeFfn` doc); qwen3moe binds the same handle as `x`.
                let rxb = r(*router_x)?;

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

                // Packed scratch, POOLED — one set serves every MoE layer in the graph. `gu_width`
                // is the fused gate|up GEMM's output width (2*nff) vs split's (nff); `ue` (the
                // split path's separate up-projection buffer) is unused/unallocated when fused.
                let gu_width = if *fused_gate_up { 2 * nff } else { nff };
                let qa = pooled(pool, be_, "moe_qa", npad * ne)?;
                let qda = pooled(pool, be_, "moe_qda", npad * (ne / 32) * 2)?;
                let qsa = pooled(pool, be_, "moe_qsa", npad * (ne / 32) * 2)?;
                let ge = pooled(pool, be_, "moe_ge", npad * gu_width * 4)?;
                let ue = if *fused_gate_up {
                    None
                } else {
                    Some(pooled(pool, be_, "moe_ue", npad * nff * 4)?)
                };
                let ae = pooled(pool, be_, "moe_ae", npad * nff * 4)?;
                let dqa = pooled(pool, be_, "moe_dqa", npad * nff)?;
                let dda = pooled(pool, be_, "moe_dda", npad * (nff / 32) * 2)?;
                let dsa = pooled(pool, be_, "moe_dsa", npad * (nff / 32) * 2)?;
                let ye = pooled(pool, be_, "moe_ye", npad * ne * 4)?;

                // Router logits for all rows (on `router_x`, NOT `x`), then GPU routing.
                let rdt = graph.desc(*router).dtype;
                let rw = r(*router)?;
                if native_dense_supported(rdt) {
                    rec.linear_native(rdt, rw, rxb, logits.as_ref(), rows, ne, n_expert);
                } else if matches!(rdt, infr_core::DType::F32) {
                    rec.linear_f32(rw, rxb, logits.as_ref(), rows, ne, n_expert);
                } else {
                    rec.linear(rw, rxb, logits.as_ref(), rows, ne, n_expert);
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
                // Per-expert down_scale (diffusion-gemma) is baked into `bucket_wts` HERE (the
                // scatter already has the expert id in hand to index it) rather than as a separate
                // post-GEMM pass — `moe_scatter_reduce` then needs no changes at all, and the
                // scale multiply is exactly equivalent to `moe_accumulate_scaled`'s per-token
                // semantics since it's linear in the down output.
                let dsb: Option<&dyn Buffer> = match down_scale {
                    Some(ds) => Some(r(*ds)?),
                    None => None,
                };
                rec.moe_bucket_scatter(
                    ids.as_ref(),
                    wts.as_ref(),
                    offsets.as_ref(),
                    fill.as_ref(),
                    bucket_rows.as_ref(),
                    bucket_wts.as_ref(),
                    inv_pos.as_ref(),
                    dsb,
                    n_pairs,
                    n_used,
                );

                let (gw, dw) = (r(*gate_exps)?, r(*down_exps)?);
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
                // `sact` (the activation's per-block min-correction sums) is only READ by the
                // Q4_K/Q5_K kernels — Q6_K/Q8_0/Q5_0 are symmetric (no min term), same split
                // `down_needs_sact` already uses below. Mirrored per-role here since gate/up can
                // now each independently be any of the five dtypes (unlike before this change,
                // when gdt/udt were both forced to Q4_K and always needed it).
                let gate_needs_sact = matches!(gdt, Q4K | Q5K);
                rec.matmul_mmq_experts(
                    gdt,
                    "expert_gateup",
                    pool[&qa].as_ref(),
                    pool[&qda].as_ref(),
                    gate_needs_sact.then(|| pool[&qsa].as_ref()),
                    gw,
                    0,
                    stride,
                    counts.as_ref(),
                    offsets.as_ref(),
                    pool[&ge].as_ref(),
                    rows,
                    ne,
                    gu_width,
                    n_expert,
                    n_used,
                );
                if let Some(ue) = ue {
                    // Split (qwen3moe): the up GEMM reads the same quantized activations and
                    // writes its own buffer — disjoint from the gate GEMM, no barrier needed.
                    let up_needs_sact = matches!(udt, Q4K | Q5K);
                    rec.suppress_sync(true);
                    rec.matmul_mmq_experts(
                        udt,
                        "expert_gateup",
                        pool[&qa].as_ref(),
                        pool[&qda].as_ref(),
                        up_needs_sact.then(|| pool[&qsa].as_ref()),
                        r(*up_exps)?,
                        0,
                        stride,
                        counts.as_ref(),
                        offsets.as_ref(),
                        pool[&ue].as_ref(),
                        rows,
                        ne,
                        nff,
                        n_expert,
                        n_used,
                    );
                    rec.suppress_sync(false);
                    rec.silu_mul(
                        pool[&ge].as_ref(),
                        pool[&ue].as_ref(),
                        pool[&ae].as_ref(),
                        n_pairs * nff,
                    );
                } else {
                    // Fused: `ge` already holds [n_pairs, 2*nff] (gate half first, up half second
                    // per row) from the single wide GEMM above.
                    match act {
                        Activation::Silu => {
                            rec.silu_mul_fused(pool[&ge].as_ref(), pool[&ae].as_ref(), n_pairs, nff)
                        }
                        Activation::Gelu => {
                            rec.gelu_mul_fused(pool[&ge].as_ref(), pool[&ae].as_ref(), n_pairs, nff)
                        }
                        Activation::Sigmoid => {
                            return Err(be(
                                "vulkan adapter: fused_gate_up batched MoeFfn Sigmoid unsupported",
                            ))
                        }
                    }
                }
                rec.quant_q8(
                    pool[&ae].as_ref(),
                    pool[&dqa].as_ref(),
                    pool[&dda].as_ref(),
                    pool[&dsa].as_ref(),
                    n_pairs,
                    nff,
                );
                // Only the K-quant min-carrying down formats (Q4_K/Q5_K) bind `sact` — Q6_K and
                // Q8_0 are symmetric (no min), same as the per-token path's dtype-gated `sact` use.
                let down_needs_sact = matches!(ddt, Q4K | Q5K);
                rec.matmul_mmq_experts(
                    ddt,
                    "expert_down",
                    pool[&dqa].as_ref(),
                    pool[&dda].as_ref(),
                    down_needs_sact.then(|| pool[&dsa].as_ref()),
                    dw,
                    0,
                    down_stride,
                    counts.as_ref(),
                    offsets.as_ref(),
                    pool[&ye].as_ref(),
                    rows,
                    nff,
                    ne,
                    n_expert,
                    n_used,
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
            // ── SMALL-m fast path (`rows` ≤ `moe_small_m_threshold()`, covering decode's rows==1
            // AND tiny prefill chunks like `pp4`): the batched path above pays a near-fixed
            // per-ACTIVE-expert cost — its tiled GEMM streams that expert's FULL weight bank no
            // matter how few rows use it (inherent to any approach — the weight has to be read
            // once if the expert is used at all), but ALSO pays quant_q8_gather + bucket
            // count/scan/scatter and a 64-row-tile GEMM that computes 60+ garbage rows when only
            // 1-8 are real. For a handful of tokens that overhead dominates. Instead: one dispatch
            // per stage computes ALL `rows*n_used` (token, expert) pairs' id-indexed GEMVs
            // directly against the native block format (no int8 quant pass, no bucket sort) —
            // `linear_native_id_multi`'s `rows` widening (see its doc) flattens `(row, slot)` into
            // the same indexing the decode path always used for its single token. Trade-off: an
            // expert used by more than one of the `rows` tokens gets its weight bank re-read once
            // per occurrence (no shared BM=64-row-tile reuse like the batched path gets), so this
            // loses once `rows` grows enough for cross-token expert overlap to matter — hence the
            // threshold (measured; see `moe_small_m_threshold`'s doc).
            //
            // Per-layer scratch rides the per-execute `pool` (same (tag, bytes) key across layers
            // → same buffer; the recorder's hazard tracking serializes the reuse, exactly like the
            // batched arm's packed scratch above), resolved at each use site via `SmB::get(pool)`.
            // Every one of these 7 buffers is FULLY WRITTEN before it is read within this op, so
            // `pooled`'s alloc_uninit is safe (no zero-init needed): the router GEMV writes all
            // `rows*n_expert` logits; `moe_topk` writes all `n_slots = rows*n_used` ids/wts
            // (n_slots is EXACT — every top-k slot is a real assignment, no padding/unrouted rows
            // like the batched path's `npad`); the id-GEMVs write every `n_slots` row of gate/up,
            // act, and y; and `dst` (a separate scratch tensor, not one of these) is explicitly
            // `rec.zero`'d before the accumulate. The previous shape alloc'd 7 FRESH zero-filled
            // buffers per MoE layer per execute — `Backend::alloc`'s calloc contract fills each
            // device-local buffer through a one-shot submit + queue_wait_idle (~27us each), and at
            // qwen3moe's 48 MoE layers that was ~340 fence-waited submits of pure host stall
            // (decode never sees this — record-once replay allocs the pool once).
            // `INFR_NO_MOE_SM_POOL=1` restores the per-layer zeroed allocs (A/B correctness oracle).
            let no_pool = std::env::var_os("INFR_NO_MOE_SM_POOL").is_some();
            let n_slots = rows * n_used;
            let logits = sm_buf(pool, be_, no_pool, "moe_sm_logits", rows * n_expert)?;
            let ids = sm_buf(pool, be_, no_pool, "moe_sm_ids", n_slots)?;
            let wts = sm_buf(pool, be_, no_pool, "moe_sm_wts", n_slots)?;
            // Fused: one [n_slots, 2*nff] gate|up buffer (gate half first, up half second per row —
            // `Op::GatedActFused`'s convention); split: separate [n_slots, nff] gate/up buffers.
            let gubuf = if *fused_gate_up {
                Some(sm_buf(pool, be_, no_pool, "moe_sm_gu", n_slots * 2 * nff)?)
            } else {
                None
            };
            let gbuf = if *fused_gate_up {
                None
            } else {
                Some(sm_buf(pool, be_, no_pool, "moe_sm_g", n_slots * nff)?)
            };
            let ubuf = if *fused_gate_up {
                None
            } else {
                Some(sm_buf(pool, be_, no_pool, "moe_sm_u", n_slots * nff)?)
            };
            let abuf = sm_buf(pool, be_, no_pool, "moe_sm_a", n_slots * nff)?;
            let ybuf = sm_buf(pool, be_, no_pool, "moe_sm_y", n_slots * ne)?;
            // All `sm_buf` allocations are done — from here `pool` is only indexed immutably (via
            // `SmB::get`), so multiple pooled buffers can be resolved at once (the batched arm's
            // `pool[&k]` discipline).
            let xb = r(*x)?;
            // diffusion-gemma's router reads a DIFFERENTLY normalized/scaled row than the experts
            // (see the `Op::MoeFfn` doc); qwen3moe binds the same handle as `x`.
            let rxb = r(*router_x)?;

            // Router logits over all experts, all `rows` tokens in one dispatch.
            let rdt = graph.desc(*router).dtype;
            let rw = r(*router)?;
            if native_dense_supported(rdt) {
                rec.linear_native(rdt, rw, rxb, logits.get(pool), rows, ne, n_expert);
            } else if matches!(rdt, infr_core::DType::F32) {
                // qwen3moe ships the router (ffn_gate_inp) as F32 — the f16 GEMV would read its
                // bytes as f16 garbage and route to arbitrary experts.
                rec.linear_f32(rw, rxb, logits.get(pool), rows, ne, n_expert);
            } else {
                rec.linear(rw, rxb, logits.get(pool), rows, ne, n_expert);
            }
            // Softmax-renormalized top-`n_used` per token, weights pre-scaled by `scale`.
            rec.moe_topk(
                logits.get(pool),
                ids.get(pool),
                wts.get(pool),
                rows,
                n_expert,
                n_used,
                *scale,
            );
            let n_act = n_slots * nff;
            if let Some(gubuf) = &gubuf {
                // Fused per-role expert GEMV: ONE dispatch over all rows' [ne, 2*nff] expert slices.
                rec.linear_native_id_multi(
                    gdt,
                    r(*gate_exps)?,
                    ids.get(pool),
                    n_used,
                    stride,
                    xb,
                    false,
                    gubuf.get(pool),
                    ne,
                    2 * nff,
                    rows,
                );
                match act {
                    Activation::Silu => {
                        rec.silu_mul_fused(gubuf.get(pool), abuf.get(pool), n_slots, nff)
                    }
                    Activation::Gelu => {
                        rec.gelu_mul_fused(gubuf.get(pool), abuf.get(pool), n_slots, nff)
                    }
                    Activation::Sigmoid => {
                        return Err(be(
                            "vulkan adapter: fused_gate_up MoeFfn Sigmoid unsupported",
                        ))
                    }
                }
            } else {
                // Split per-role expert GEMVs: gate/up read the shared row; down reads each slot's act.
                let (gbuf, ubuf) = (gbuf.as_ref().unwrap(), ubuf.as_ref().unwrap());
                rec.linear_native_id_multi(
                    gdt,
                    r(*gate_exps)?,
                    ids.get(pool),
                    n_used,
                    stride,
                    xb,
                    false,
                    gbuf.get(pool),
                    ne,
                    nff,
                    rows,
                );
                rec.linear_native_id_multi(
                    udt,
                    r(*up_exps)?,
                    ids.get(pool),
                    n_used,
                    stride,
                    xb,
                    false,
                    ubuf.get(pool),
                    ne,
                    nff,
                    rows,
                );
                match act {
                    Activation::Silu => {
                        rec.silu_mul(gbuf.get(pool), ubuf.get(pool), abuf.get(pool), n_act)
                    }
                    Activation::Sigmoid => {
                        rec.mul_sigmoid(gbuf.get(pool), ubuf.get(pool), abuf.get(pool), n_act)
                    }
                    Activation::Gelu => rec.gelu_mul_off(
                        gbuf.get(pool),
                        ubuf.get(pool),
                        0,
                        0,
                        nff,
                        abuf.get(pool),
                        n_act,
                    ),
                }
            }
            rec.linear_native_id_multi(
                ddt,
                r(*down_exps)?,
                ids.get(pool),
                n_used,
                down_stride,
                abuf.get(pool),
                true,
                ybuf.get(pool),
                nff,
                ne,
                rows,
            );
            // `Op::MoeFfn` dst is the pure FFN output (residual Add is a separate op), but
            // moe_accumulate(_scaled) ADDs — zero dst first.
            let dstb = r(*dst)?;
            rec.zero(dstb, rows * ne);
            match down_scale {
                Some(ds) => rec.moe_accumulate_scaled(
                    ybuf.get(pool),
                    wts.get(pool),
                    ids.get(pool),
                    r(*ds)?,
                    dstb,
                    ne,
                    n_used,
                    rows,
                ),
                None => rec.moe_accumulate(ybuf.get(pool), wts.get(pool), dstb, ne, n_used, rows),
            }
            // Under the `no_pool` escape these are owned per-layer boxes — drain them into
            // `transient` so the recording outlives them; pooled buffers are no-ops here (owned by
            // `pool`, itself drained into `transient` at the end of the op loop).
            for b in [logits, ids, wts, abuf, ybuf] {
                b.into_transient(transient);
            }
            for b in [gubuf, gbuf, ubuf].into_iter().flatten() {
                b.into_transient(transient);
            }
        }
    }
    Ok(())
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
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
    if !replay.self_advancing {
        // Host fallback (INFR_NO_GPU_POS): pos from positions[0] (decode rows=1); kv_len = pos+1.
        let pos = read_pos0(be_, resolve(&replay.scratch, bindings, replay.positions)?)?;
        let kv_len = pos + 1;
        let mut pbytes = [0u8; 8];
        pbytes[0..4].copy_from_slice(&pos.to_le_bytes());
        pbytes[4..8].copy_from_slice(&kv_len.to_le_bytes());
        be_.upload(replay.params.as_ref(), &pbytes)?;
    }
    replay.recorded.replay().map_err(|e| be(e.to_string()))?;
    Ok(())
}

/// Chained decode (`Backend::execute_chain`): replay the record-once recording `n` times in ONE
/// submission — the sampled id flows device-side (the runner bound the sampler output and the
/// embed-gather ids input to one buffer), params self-advance, and the trailing `id_log` writes
/// each iteration's id into the ring. Returns the n ids read from the ring, or `None` when the
/// plan can't chain (ineligible graph, host-pos fallback, no device sampler, or n out of the
/// ring's range) — the caller falls back to per-token `execute`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn execute_chain(
    be_: &VulkanBackend,
    plan: &dyn Plan,
    bindings: &Bindings,
    n: usize,
) -> Result<Option<Vec<u32>>> {
    let Some(plan) = plan.as_any().downcast_ref::<VkDecodePlan>() else {
        return Ok(None);
    };
    if !plan.eligible || n == 0 || n > 64 {
        return Ok(None);
    }
    let mut guard = plan.replay.lock().unwrap();
    if guard.is_none() {
        *guard = Some(record_decode_replay(be_, &plan.graph, bindings)?);
    }
    let replay = guard.as_ref().unwrap();
    let (true, Some(ring)) = (replay.self_advancing, replay.ring.as_ref()) else {
        return Ok(None);
    };
    // The chunk decodes positions p0+1 ..= p0+n (params[0] = the last decoded position).
    let p0 = read_pos0(be_, replay.params.as_ref())?;
    replay.recorded.replay_n(n).map_err(|e| be(e.to_string()))?;
    let mut rbytes = vec![0u8; 64 * 4];
    be_.download(ring.as_ref(), &mut rbytes)?;
    let ring_u32: &[u32] = bytemuck::cast_slice(&rbytes);
    let ids = (1..=n as u32)
        .map(|i| ring_u32[((p0 + i) & 63) as usize])
        .collect();
    Ok(Some(ids))
}

/// Build the record-once decode replay: persistent scratch + params SSBO, one recording via the
/// `_dyn` (params-driven) kernels for the pos-dependent ops. Only called for a [`decode_eligible`]
/// graph. The returned recording is replayed (not submitted here) — the caller updates `params` and
/// calls `replay()`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn record_decode_replay(
    be_: &VulkanBackend,
    graph: &Graph,
    bindings: &Bindings,
) -> Result<DecodeReplay> {
    let (scratch, dummy) = alloc_scratch(be_, graph)?;
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
    let rec = be_.recorder_persistent()?;
    // Device-side position stream: seed params to [pos0-1, pos0] and record a one-thread
    // increment FIRST — every replay self-advances and the host never writes pos/params again
    // (no dyn kernel reads the `positions` buffer; they all read params). INFR_NO_GPU_POS=1
    // keeps the per-replay host read_pos0 + params upload instead (A/B).
    let self_advancing = std::env::var("INFR_NO_GPU_POS").is_err();
    if self_advancing {
        let pos0 = read_pos0(be_, resolve(&scratch, bindings, positions)?)?;
        let mut pbytes = [0u8; 8];
        pbytes[0..4].copy_from_slice(&pos0.wrapping_sub(1).to_le_bytes());
        pbytes[4..8].copy_from_slice(&pos0.to_le_bytes());
        be_.upload(params.as_ref(), &pbytes)?;
        // Cross-iteration fence for the chained decode (replay_n submits this recording n times
        // back-to-back): orders iteration i's tail writes (sampled id, id_log) before iteration
        // i+1's head reads (params_advance, EmbedGather). One extra barrier per token — noise.
        rec.global_barrier();
        rec.params_advance(params.as_ref());
    }
    let mode = RopeMode::Dynamic(params.as_ref());
    let mut mmv_memo: Option<(TensorId, usize, usize)> = None;
    for (op_idx, op) in graph.ops.iter().enumerate() {
        if skip_op.contains(&op_idx) {
            continue;
        }
        // Peephole: fuse Op::Linear (f32, m<=4) + Op::GatedAct (Gelu, strided up) into one
        // e2b_gate dispatch for E2B per-layer inp_gate projections.
        if let Op::Linear {
            x,
            weight,
            dst,
            m,
            in_f,
            out_f,
            ..
        } = op
        {
            if graph.desc(*weight).dtype == infr_core::DType::F32
                && *m <= 4
                && op_idx + 1 < graph.ops.len()
            {
                if let Op::GatedAct {
                    gate: g_gate,
                    up: g_up,
                    dst: g_dst,
                    act: Activation::Gelu,
                    up_off: g_off,
                    up_stride: g_stride,
                    ..
                } = &graph.ops[op_idx + 1]
                {
                    if *g_gate == *dst && *g_dst == *dst && *g_stride > 0 {
                        let (w_, x_, up_, y_) = (
                            resolve(&scratch, bindings, *weight)?,
                            resolve(&scratch, bindings, *x)?,
                            resolve(&scratch, bindings, *g_up)?,
                            resolve(&scratch, bindings, *dst)?,
                        );
                        rec.e2b_gate(
                            w_,
                            x_,
                            up_,
                            *g_off as usize,
                            *g_stride as usize,
                            y_,
                            *m as usize,
                            *in_f as usize,
                            *out_f as usize,
                        );
                        skip_op.insert(op_idx + 1);
                        continue;
                    }
                }
            }
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
            &mut mmv_memo,
            dummy.as_ref(),
        )?;
    }
    for c in dyn_args.drain(..) {
        transient.extend([c.args, c.pm, c.pl, c.pacc]);
    }
    transient.extend(pool.into_values());
    // Chained decode: when the graph ends in a device-side sampler (Argmax/Sample), log each
    // iteration's id into a host-visible ring so `execute_chain` reads the whole chunk in one go.
    let ring = if self_advancing {
        graph
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Argmax { dst, .. } | Op::Sample { dst, .. } => Some(*dst),
                _ => None,
            })
            .map(|dst| -> Result<Box<dyn Buffer>> {
                let rb = be_.alloc(64 * 4, BufferUsage::Readback)?;
                rec.id_log(
                    params.as_ref(),
                    resolve(&scratch, bindings, dst)?,
                    rb.as_ref(),
                );
                Ok(rb)
            })
            .transpose()?
    } else {
        None
    };
    let recorded = rec.finish_record().map_err(|e| be(e.to_string()))?;
    // dummy is unused in an eligible decode (m=1 GEMV path), but hold it (and any transient) so the
    // recording can't reference a freed buffer.
    transient.push(dummy);
    Ok(DecodeReplay {
        scratch,
        params,
        self_advancing,
        ring,
        positions,
        recorded,
        _transient: transient,
    })
}

/// Per-execute static recording: allocate `Internal` scratch fresh, record every op via `lower_op`
/// (Static mode — pos as a push constant read from `positions[0]`), submit + wait. Used for prefill
/// batches and every ineligible decode (gemma/E2B/MoE/qwen35).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn execute_static(be_: &VulkanBackend, graph: &Graph, bindings: &Bindings) -> Result<()> {
    // `dummy`: a tiny unused buffer bound as the (scales, mins) args of the f16 `matmul_proj`
    // GEMM — allocated inside `alloc_scratch`'s single zero-init batch.
    let (scratch, dummy) = alloc_scratch(be_, graph)?;

    // RoPE position: the static `qk_norm_rope`/`rope` kernels take a scalar `rope_pos`, but the IR
    // carries a `positions` i32 tensor. Read `positions[0]` (decode rows=1, or the start of a
    // consecutive-prefill run) up front — `read_pos0` reads host-visible (mapped) positions
    // directly (the seam always binds Staging there) and only falls back to the syncing
    // `download` (a one-shot submit + wait) for a device-local buffer.
    let mut rope_pos: HashMap<u32, usize> = HashMap::new();
    for op in &graph.ops {
        let pid = match op {
            Op::Rope { positions, .. } | Op::QkNormRope { positions, .. } => Some(*positions),
            _ => None,
        };
        if let Some(pid) = pid {
            if let std::collections::hash_map::Entry::Vacant(e) = rope_pos.entry(pid.0) {
                e.insert(read_pos0(be_, resolve(&scratch, bindings, pid)?)? as usize);
            }
        }
    }

    let (fused_kv_write, mut skip_op) = kv_write_peephole(graph);
    let (fused_add, skip_add) = linear_add_peephole(graph);
    skip_op.extend(skip_add);

    // Transient buffers allocated inside the op loop (GEMM/attention/MoE scratch) must outlive the
    // recorder — hold them here so they drop only after `rec.finish()` submits.
    let mut transient: Vec<Box<dyn Buffer>> = Vec::new();

    let rec = be_.recorder()?;
    let mode = RopeMode::Static(&rope_pos);
    let mut dyn_args: Vec<DynAttnCtx> = Vec::new();
    let mut pool: ScratchPool = HashMap::new();
    let mut mmv_memo: Option<(TensorId, usize, usize)> = None;
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
            &mut mmv_memo,
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

    /// A one-op `GatedRmsNorm` graph (fused per-head RMSNorm + SiLU gate, qwen35 DeltaNet z-gate)
    /// through the seam must match `rmsnorm(x,w) * silu(z)` computed in two host passes — i.e.
    /// the SAME thing the split `QkNorm`→`GatedAct` pair produces. Multi-head (n_head=2) shape so
    /// the per-head reduction boundary is exercised, not just a single-row rmsnorm.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn gated_rmsnorm_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (rows, n_head, head_dim, eps) = (2usize, 2usize, 8usize, 1e-6f32);
        let dim = n_head * head_dim;
        let x: Vec<f32> = (0..rows * dim).map(|i| i as f32 * 0.1 - 0.4).collect();
        let w: Vec<f32> = (0..head_dim).map(|i| 1.0 + i as f32 * 0.05).collect();
        let z: Vec<f32> = (0..rows * dim).map(|i| (i as f32 * 0.37).sin()).collect();
        let silu = |v: f32| v / (1.0 + (-v).exp());
        // host reference: per-head rmsnorm, then elementwise silu(z) gate — the split-op semantics.
        let mut want = vec![0f32; rows * dim];
        for r in 0..rows {
            for h in 0..n_head {
                let b = (r * n_head + h) * head_dim;
                let ss = (0..head_dim).map(|i| x[b + i] * x[b + i]).sum::<f32>() / head_dim as f32;
                let s = 1.0 / (ss + eps).sqrt();
                for i in 0..head_dim {
                    want[b + i] = x[b + i] * s * w[i] * silu(z[b + i]);
                }
            }
        }
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![head_dim], DType::F32));
        let zi = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
        let yi = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
        g.push(Op::GatedRmsNorm {
            x: xi,
            weight: wi,
            gate: zi,
            dst: yi,
            rows: rows as u32,
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            eps,
        });
        let xb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        let wb = be_.alloc(head_dim * 4, BufferUsage::Weights).unwrap();
        let zb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        let yb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        be_.upload(wb.as_ref(), bytemuck::cast_slice(&w)).unwrap();
        be_.upload(zb.as_ref(), bytemuck::cast_slice(&z)).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(wi, wb.as_ref());
        bind.bind(zi, zb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; rows * dim];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..rows * dim {
            assert!(
                (got[i] - want[i]).abs() < 1e-3,
                "gated_rmsnorm mismatch at {i}: got {} want {}",
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
            up_stride: 0,
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

    /// A one-op `Softmax` graph (row-wise, with a non-trivial `scale`) must match a host softmax.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn softmax_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (rows, dim, scale) = (3usize, 300usize, 1.3f32);
        let x: Vec<f32> = (0..rows * dim)
            .map(|i| (i as f32 * 0.037).sin() * 5.0)
            .collect();
        let mut want = vec![0f32; rows * dim];
        for r in 0..rows {
            let b = r * dim;
            let row = &x[b..b + dim];
            let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v * scale));
            let mut denom = 0f32;
            for (w, &v) in want[b..b + dim].iter_mut().zip(row) {
                *w = (v * scale - mx).exp();
                denom += *w;
            }
            for w in want[b..b + dim].iter_mut() {
                *w /= denom;
            }
        }
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
        let yi = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
        g.push(Op::Softmax {
            x: xi,
            dst: yi,
            rows: rows as u32,
            dim: dim as u32,
            scale,
            scale_buf: None,
        });
        let xb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        let yb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; rows * dim];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..rows * dim {
            assert!(
                (got[i] - want[i]).abs() < 1e-4,
                "softmax mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// `Op::Softmax` with `scale_buf: Some(_)` (the DiffusionGemma denoise perf path — see its doc)
    /// must match the SAME host softmax as `scale`, with the scale sourced from a bound 1-element
    /// buffer instead of a push constant. Also checks that a plan built ONCE and re-`execute`d with
    /// a DIFFERENT `scale_buf` value each time picks up the new scale — the whole point of the
    /// mechanism (a cached/replayed plan varying its softmax temperature without a rebuild).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn softmax_dyn_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (rows, dim) = (3usize, 300usize);
        let x: Vec<f32> = (0..rows * dim)
            .map(|i| (i as f32 * 0.037).sin() * 5.0)
            .collect();
        let host_softmax = |scale: f32| -> Vec<f32> {
            let mut want = vec![0f32; rows * dim];
            for r in 0..rows {
                let b = r * dim;
                let row = &x[b..b + dim];
                let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v * scale));
                let mut denom = 0f32;
                for (w, &v) in want[b..b + dim].iter_mut().zip(row) {
                    *w = (v * scale - mx).exp();
                    denom += *w;
                }
                for w in want[b..b + dim].iter_mut() {
                    *w /= denom;
                }
            }
            want
        };
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
        let si = g.input(TensorDesc::new(vec![1], DType::F32));
        let yi = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
        g.push(Op::Softmax {
            x: xi,
            dst: yi,
            rows: rows as u32,
            dim: dim as u32,
            scale: 1.0, // ignored — scale_buf takes over
            scale_buf: Some(si),
        });
        let xb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        let sb = be_.alloc(4, BufferUsage::Staging).unwrap();
        let yb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(si, sb.as_ref());
        bind.bind(yi, yb.as_ref());
        // Same compiled plan, replayed with two DIFFERENT scale_buf values — exactly the
        // DiffusionGemma denoise loop's per-step usage (see `crates/infr-llama/src/seam/
        // runner.rs`'s denoise call site).
        for &scale in &[1.3f32, 0.42f32] {
            be_.upload(sb.as_ref(), &scale.to_le_bytes()).unwrap();
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * dim];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            let want = host_softmax(scale);
            for i in 0..rows * dim {
                assert!(
                    (got[i] - want[i]).abs() < 1e-4,
                    "softmax_dyn mismatch at scale={scale} i={i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// DiffusionGemma Phase-B: the FULL in-graph self-conditioning subgraph (softmax → soft-embed
    /// matmul → scale → rmsnorm → gate/up linears → gated-GELU → down linear — the exact op
    /// sequence `seam.rs`'s `build` emits for a SC-on denoise step) at small synthetic dims,
    /// compared against a host reference computed against the SAME f16-rounded weights (isolating
    /// the graph WIRING/layout from f16 rounding noise, like `linear_graph_matches_host`). This is
    /// "one SC denoise step" in miniature — the real model's SC block is this identical op
    /// sequence at production sizes (ne/vocab/nff/canvas_length), just with GGUF-loaded weights.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn diffusion_gemma_sc_subgraph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (cc, vocab, ne, nff) = (4usize, 512usize, 128usize, 256usize);
        let eps = 1e-6f32;
        let scale = 1.0f32; // production always premultiplies temp_inv on the host (see seam.rs)

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

        let sc_logits: Vec<f32> = (0..cc * vocab)
            .map(|i| (i as f32 * 0.0131).sin() * 4.0)
            .collect();
        let embt: Vec<f32> = (0..ne * vocab)
            .map(|i| (i as f32 * 0.0007).cos() * 0.3)
            .collect();
        let gate_w: Vec<f32> = (0..nff * ne)
            .map(|i| (i as f32 * 0.0011).sin() * 0.2)
            .collect();
        let up_w: Vec<f32> = (0..nff * ne)
            .map(|i| (i as f32 * 0.0017).cos() * 0.2)
            .collect();
        let down_w: Vec<f32> = (0..ne * nff)
            .map(|i| (i as f32 * 0.0013).sin() * 0.2)
            .collect();
        let pre_norm: Vec<f32> = (0..ne).map(|i| 1.0 + i as f32 * 0.01).collect();

        // Host uses the SAME f16-rounded weights the GPU reads (isolates wiring, not rounding).
        let (embt16, gate16, up16, down16) = (
            to_f16(&embt),
            to_f16(&gate_w),
            to_f16(&up_w),
            to_f16(&down_w),
        );
        let (embq, gateq, upq, downq) = (deq(&embt16), deq(&gate16), deq(&up16), deq(&down16));

        let mut want = vec![0f32; cc * ne];
        for r in 0..cc {
            let row = &sc_logits[r * vocab..(r + 1) * vocab];
            let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v * scale));
            let mut probs = vec![0f32; vocab];
            let mut denom = 0f32;
            for (p, &v) in probs.iter_mut().zip(row) {
                *p = (v * scale - mx).exp();
                denom += *p;
            }
            for p in probs.iter_mut() {
                *p /= denom;
            }
            // soft = probs @ embT — embq is [ne, vocab] row-major (Op::Linear's weight layout).
            let mut soft = vec![0f32; ne];
            for (e, s) in soft.iter_mut().enumerate() {
                let erow = &embq[e * vocab..(e + 1) * vocab];
                *s = probs.iter().zip(erow).map(|(&p, &w)| p * w).sum::<f32>() * (ne as f32).sqrt();
            }
            let ms: f32 = soft.iter().map(|&x| x * x).sum::<f32>() / ne as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            let normed: Vec<f32> = soft
                .iter()
                .zip(&pre_norm)
                .map(|(&x, &w)| x * inv * w)
                .collect();
            let mut gv = vec![0f32; nff];
            let mut uv = vec![0f32; nff];
            for f in 0..nff {
                let grow = &gateq[f * ne..(f + 1) * ne];
                let urow = &upq[f * ne..(f + 1) * ne];
                gv[f] = grow.iter().zip(&normed).map(|(&w, &x)| w * x).sum();
                uv[f] = urow.iter().zip(&normed).map(|(&w, &x)| w * x).sum();
            }
            let act: Vec<f32> = gv
                .iter()
                .zip(&uv)
                .map(|(&gd, &ud)| {
                    let gelu =
                        0.5 * gd * (1.0 + (0.797_884_6 * (gd + 0.044715 * gd * gd * gd)).tanh());
                    gelu * ud
                })
                .collect();
            for (e, w_row) in want[r * ne..(r + 1) * ne].iter_mut().enumerate() {
                let drow = &downq[e * nff..e * nff + nff];
                *w_row = drow.iter().zip(&act).map(|(&w, &a)| w * a).sum();
            }
        }

        let mut g = Graph::new();
        let logits_i = g.input(TensorDesc::new(vec![cc, vocab], DType::F32));
        let embt_i = g.weight(TensorDesc::new(vec![ne, vocab], DType::F16));
        let pre_norm_i = g.weight(TensorDesc::new(vec![ne], DType::F32));
        let gate_i = g.weight(TensorDesc::new(vec![nff, ne], DType::F16));
        let up_i = g.weight(TensorDesc::new(vec![nff, ne], DType::F16));
        let down_i = g.weight(TensorDesc::new(vec![ne, nff], DType::F16));
        let sig_i = g.output(TensorDesc::new(vec![cc, ne], DType::F32));

        let probs_i = g.internal(TensorDesc::new(vec![cc, vocab], DType::F32));
        g.push(Op::Softmax {
            x: logits_i,
            dst: probs_i,
            rows: cc as u32,
            dim: vocab as u32,
            scale,
            scale_buf: None,
        });
        let soft_i = g.internal(TensorDesc::new(vec![cc, ne], DType::F32));
        g.push(Op::Linear {
            x: probs_i,
            weight: embt_i,
            dst: soft_i,
            m: cc as u32,
            in_f: vocab as u32,
            out_f: ne as u32,
            w_off: 0,
        });
        g.push(Op::Scale {
            x: soft_i,
            dst: soft_i,
            s: (ne as f32).sqrt(),
            n: (cc * ne) as u32,
        });
        let normed_i = g.internal(TensorDesc::new(vec![cc, ne], DType::F32));
        g.push(Op::RmsNorm {
            x: soft_i,
            weight: pre_norm_i,
            dst: normed_i,
            rows: cc as u32,
            dim: ne as u32,
            eps,
        });
        let g_i = g.internal(TensorDesc::new(vec![cc, nff], DType::F32));
        let u_i = g.internal(TensorDesc::new(vec![cc, nff], DType::F32));
        g.push(Op::Linear {
            x: normed_i,
            weight: gate_i,
            dst: g_i,
            m: cc as u32,
            in_f: ne as u32,
            out_f: nff as u32,
            w_off: 0,
        });
        g.push(Op::Linear {
            x: normed_i,
            weight: up_i,
            dst: u_i,
            m: cc as u32,
            in_f: ne as u32,
            out_f: nff as u32,
            w_off: 0,
        });
        let act_i = g.internal(TensorDesc::new(vec![cc, nff], DType::F32));
        g.push(Op::GatedAct {
            gate: g_i,
            up: u_i,
            dst: act_i,
            rows: cc as u32,
            nff: nff as u32,
            act: Activation::Gelu,
            up_off: 0,
            up_stride: 0,
        });
        g.push(Op::Linear {
            x: act_i,
            weight: down_i,
            dst: sig_i,
            m: cc as u32,
            in_f: nff as u32,
            out_f: ne as u32,
            w_off: 0,
        });

        let logits_b = be_.alloc(cc * vocab * 4, BufferUsage::Activations).unwrap();
        let embt_b = be_.alloc(embt16.len(), BufferUsage::Weights).unwrap();
        let pre_norm_b = be_.alloc(ne * 4, BufferUsage::Weights).unwrap();
        let gate_b = be_.alloc(gate16.len(), BufferUsage::Weights).unwrap();
        let up_b = be_.alloc(up16.len(), BufferUsage::Weights).unwrap();
        let down_b = be_.alloc(down16.len(), BufferUsage::Weights).unwrap();
        let sig_b = be_.alloc(cc * ne * 4, BufferUsage::Activations).unwrap();
        be_.upload(logits_b.as_ref(), bytemuck::cast_slice(&sc_logits))
            .unwrap();
        be_.upload(embt_b.as_ref(), &embt16).unwrap();
        be_.upload(pre_norm_b.as_ref(), bytemuck::cast_slice(&pre_norm))
            .unwrap();
        be_.upload(gate_b.as_ref(), &gate16).unwrap();
        be_.upload(up_b.as_ref(), &up16).unwrap();
        be_.upload(down_b.as_ref(), &down16).unwrap();

        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(logits_i, logits_b.as_ref());
        bind.bind(embt_i, embt_b.as_ref());
        bind.bind(pre_norm_i, pre_norm_b.as_ref());
        bind.bind(gate_i, gate_b.as_ref());
        bind.bind(up_i, up_b.as_ref());
        bind.bind(down_i, down_b.as_ref());
        bind.bind(sig_i, sig_b.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; cc * ne];
        be_.download(sig_b.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();

        let cos = |a: &[f32], b: &[f32]| -> f64 {
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (&x, &y) in a.iter().zip(b) {
                dot += x as f64 * y as f64;
                na += x as f64 * x as f64;
                nb += y as f64 * y as f64;
            }
            dot / (na.sqrt() * nb.sqrt())
        };
        for r in 0..cc {
            let (w, g_) = (&want[r * ne..(r + 1) * ne], &got[r * ne..(r + 1) * ne]);
            let c = cos(w, g_);
            println!("diffusion_gemma_sc_subgraph_matches_host: row {r} cosine={c:.5}");
            assert!(c > 0.99, "row {r}: SC subgraph cosine too low: {c}");
        }
    }

    /// Small-m attention through the split-K path (rows 2..63, kv_len > chunk — the SHORT TURN
    /// suffix-prefill shape): every row's output must match a host oracle with the PER-ROW causal
    /// bound (row r attends keys 0..=pos+r), full-causal and sliding-window. rows=17 also covers
    /// the 9..63 band; kv_len=300 > chunk (=64) forces the split route, not the scalar fallback.
    #[test]
    fn attention_small_m_split_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (nh, nkv, hd) = (4usize, 2usize, 64usize);
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
        // kv=300 exercises the per-row split grid; kv=8300 with rows >= 12 exercises the
        // rows-BATCHED tier (attn_partial_mrows_c256) — including a short last row-group (17 =
        // 4+4+4+4+1) and a windowed case. (Flash needs Internal q/dst, so raw-input graphs like
        // these always take the split family.)
        for (rows, window, kv_len) in [
            (2usize, 0usize, 300usize),
            (5, 0, 300),
            (8, 0, 300),
            (17, 0, 300),
            (4, 128, 300),
            (17, 0, 8300),
            (40, 0, 8300),
            (13, 256, 8300),
        ] {
            let pos = kv_len - rows; // the suffix occupies the last `rows` positions
            let q: Vec<f32> = (0..rows * nh * hd)
                .map(|i| (i as f32 * 0.03).sin())
                .collect();
            let k: Vec<f32> = (0..kv_len * nkv * hd)
                .map(|i| (i as f32 * 0.011).cos())
                .collect();
            let v: Vec<f32> = (0..kv_len * nkv * hd)
                .map(|i| ((i % 173) as f32) * 0.011 - 0.9)
                .collect();
            let (qf, kf, vf) = (to_f16(&q), to_f16(&k), to_f16(&v));
            let (qd, kd, vd) = (deq(&qf), deq(&kf), deq(&vf));
            // host oracle: per-row causal end pos+r+1, window lo = max(0, end - window)
            let mut want = vec![0f32; rows * nh * hd];
            for r0 in 0..rows {
                let end = pos + r0 + 1;
                let lo = if window > 0 && end > window {
                    end - window
                } else {
                    0
                };
                for h in 0..nh {
                    let kvh = h / group;
                    let mut sc = vec![0f32; end - lo];
                    let mut mx = f32::NEG_INFINITY;
                    for (jj, scj) in sc.iter_mut().enumerate() {
                        let j = lo + jj;
                        let d: f32 = (0..hd)
                            .map(|x| qd[(r0 * nh + h) * hd + x] * kd[(j * nkv + kvh) * hd + x])
                            .sum();
                        *scj = d * scale;
                        mx = mx.max(*scj);
                    }
                    let l: f32 = sc.iter().map(|s| (s - mx).exp()).sum();
                    for (jj, &s) in sc.iter().enumerate() {
                        let j = lo + jj;
                        let p = (s - mx).exp() / l;
                        for x in 0..hd {
                            want[(r0 * nh + h) * hd + x] += p * vd[(j * nkv + kvh) * hd + x];
                        }
                    }
                }
            }
            let mut g = Graph::new();
            let qi = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F16));
            let ki = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
            let vi = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
            let yi = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
            g.push(Op::Attention {
                q: qi,
                k_cache: ki,
                v_cache: vi,
                dst: yi,
                rows: rows as u32,
                kv_len: kv_len as u32,
                n_head: nh as u32,
                n_kv: nkv as u32,
                head_dim: hd as u32,
                scale,
                mask: if window > 0 {
                    AttnMask::SlidingWindow(window)
                } else {
                    AttnMask::Causal
                },
                pos: pos as u32,
            });
            let qb = be_.alloc(qf.len(), BufferUsage::Activations).unwrap();
            let kb = be_.alloc(kf.len(), BufferUsage::Activations).unwrap();
            let vb = be_.alloc(vf.len(), BufferUsage::Activations).unwrap();
            let yb = be_
                .alloc(rows * nh * hd * 4, BufferUsage::Activations)
                .unwrap();
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
            let mut got = vec![0f32; rows * nh * hd];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for i in 0..rows * nh * hd {
                assert!(
                    (got[i] - want[i]).abs() < 2e-2,
                    "small-m attention mismatch rows={rows} window={window} at {i} \
                     (row {} head {}): got {} want {}",
                    i / (nh * hd),
                    (i / hd) % nh,
                    got[i],
                    want[i]
                );
            }
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

    // ---- Q4_K helpers (block=256: f16 d + f16 dmin + 12-byte packed 6-bit scale/min ×8 sub-blocks
    // + 128 nibbles = 144 bytes) — mirrors `native_gemm_mmq_q4k`'s decode exactly (including the
    // `get_scale_min_k4`-style 6-bit pack/unpack and the (sub_even low nibble, sub_odd high nibble)
    // interleave), so the batched-fused isolation test can synthesize a real Q4_K gate_up bank and
    // have the host reference dequant the SAME rounded values the GPU shader reads. Not a
    // rate-distortion-optimal quantizer (min/max per sub-block, not llama.cpp's search) — only
    // internal round-trip consistency matters for this test.
    fn q4k(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 256 * 144);
        for blk in x.chunks(256) {
            let mut sc = [0u32; 8];
            let mut mn = [0u32; 8];
            let mut sub_lo = [0f32; 8]; // per-sub-block min
            let mut sub_sc = [0f32; 8]; // per-sub-block (max-min)/15
            for (j, sub) in blk.chunks(32).enumerate() {
                let lo = sub.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                sub_lo[j] = lo;
                sub_sc[j] = ((hi - lo) / 15.0).max(1e-8);
            }
            let d = sub_sc.iter().cloned().fold(0f32, f32::max) / 63.0;
            let dmin = sub_lo
                .iter()
                .cloned()
                .fold(0f32, |m, v| m.max(v.abs()))
                .max(1e-8)
                / 63.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let idmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };
            for j in 0..8 {
                sc[j] = ((sub_sc[j] * id).round() as i32).clamp(0, 63) as u32;
                mn[j] = ((sub_lo[j].abs() * idmin).round() as i32).clamp(0, 63) as u32;
            }
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            out.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
            // Pack the 8 (sc,mn) pairs into 12 bytes (get_scale_min_k4 convention).
            let mut scales = [0u8; 12];
            for k in 0..4 {
                scales[k] = (sc[k] & 0x3F) as u8 | (((sc[4 + k] >> 4) & 0x3) as u8) << 6;
                scales[4 + k] = (mn[k] & 0x3F) as u8 | (((mn[4 + k] >> 4) & 0x3) as u8) << 6;
                scales[8 + k] = (sc[4 + k] & 0xF) as u8 | (((mn[4 + k] & 0xF) as u8) << 4);
            }
            out.extend_from_slice(&scales);
            // Quantize each sub-block's 32 elements against ITS recovered (d*sc, dmin*mn) — the
            // SAME values the GPU decode reconstructs — then pack (sub_even low nibble, sub_odd
            // high nibble) per 32-byte pair-region.
            let mut q = [[0u8; 32]; 8];
            for j in 0..8 {
                let scale = d * sc[j] as f32;
                let min = dmin * mn[j] as f32;
                let iscale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                for (l, &v) in blk[j * 32..j * 32 + 32].iter().enumerate() {
                    q[j][l] = (((v - min) * iscale).round() as i32).clamp(0, 15) as u8;
                }
            }
            let mut qs = [0u8; 128];
            for pair in 0..4 {
                let (lo, hi) = (&q[2 * pair], &q[2 * pair + 1]);
                for l in 0..32 {
                    qs[pair * 32 + l] = (lo[l] & 0xF) | (hi[l] << 4);
                }
            }
            out.extend_from_slice(&qs);
        }
        out
    }
    fn deq_q4k(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 144 * 256);
        for blk in bytes.chunks(144) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
            let scales = &blk[4..16];
            let qs = &blk[16..144];
            for j in 0..8u32 {
                let (sc, mn) = if j < 4 {
                    (
                        (scales[j as usize] & 0x3F) as u32,
                        (scales[j as usize + 4] & 0x3F) as u32,
                    )
                } else {
                    let i2 = j as usize - 4;
                    (
                        (scales[j as usize + 4] as u32 & 0xF) | (((scales[i2] as u32) >> 6) << 4),
                        (scales[j as usize + 4] as u32 >> 4)
                            | (((scales[i2 + 4] as u32) >> 6) << 4),
                    )
                };
                let scale = d * sc as f32;
                let min = dmin * mn as f32;
                let pair = (j / 2) as usize;
                let lo = (j % 2) == 0;
                for l in 0..32 {
                    let byte = qs[pair * 32 + l];
                    let nib = if lo { byte & 0xF } else { byte >> 4 };
                    out.push(scale * nib as f32 - min);
                }
            }
        }
        out
    }

    /// A one-op `MoeFfn` graph (Q8_0 router + stacked experts, split gate/up, unscaled — qwen3moe's
    /// exact shape) through the seam must match a host reference that mirrors the CPU `Op::MoeFfn`
    /// interpreter on the SAME q8-rounded weights: router softmax → top-`n_used` → per-expert
    /// SwiGLU → weighted (×scale) accumulate. Runs rows=1 (decode) and rows=4 (a small prefill
    /// chunk, e.g. `pp4`) — both ≤ `moe_small_m_threshold()`, so `gate`/`up` being Q8_0 (not
    /// Q4_K) doesn't matter here: this exercises only the small-m fast path either way, covering
    /// `linear_native_id_multi`'s multi-row widening and (unlike
    /// `moe_ffn_batched_fused_scaled_matches_host`, which only exercises the SCALED accumulate at
    /// multi-row) the plain `moe_accumulate`'s `rows` generalization. Q8_0 gate/up IS also
    /// eligible for the BATCHED path above threshold since the `matmul_mmq_experts` dtype
    /// widening — see `moe_ffn_batched_split_q5k_gate_up_matches_host` for that coverage.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (32usize, 4usize, 2usize, 32usize);
        let scale = 1.3f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5; // deterministic weight/act filler
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

        for &rows in &[1usize, 4usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                // router logits → softmax over all experts
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], xr))
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|p| *p /= psum);
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                for &e in &idx {
                    let gs = e * nff * ne;
                    let ds = e * ne * nff;
                    let actv: Vec<f32> = (0..nff)
                        .map(|j| {
                            let g = dot(&gd[gs + j * ne..gs + (j + 1) * ne], xr);
                            let u = dot(&ud[gs + j * ne..gs + (j + 1) * ne], xr);
                            silu(g) * u
                        })
                        .collect();
                    let w_e = probs[e] / wsum * scale;
                    for i in 0..ne {
                        want[t * ne + i] += w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actv);
                    }
                }
            }
            // graph
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gi = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q8_0));
            let ui = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q8_0));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q8_0));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: xi,
                router: ri,
                gate_exps: gi,
                up_exps: ui,
                down_exps: di,
                down_scale: None,
                fused_gate_up: false,
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
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gi, gb.as_ref());
            bind.bind(ui, ub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 3e-3,
                    "moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// diffusion-gemma's `MoeFfn` shape: a SEPARATE `router_x` (not `x`), a FUSED `gate_up_exps`
    /// tensor (gate rows first, up rows second per expert), and a per-expert `down_scale`. Same
    /// host-reference-vs-seam structure as `moe_ffn_graph_matches_host`, extended for the three
    /// new fields (see `docs/DIFFUSIONGEMMA.md` and the `Op::MoeFfn` doc comment).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_fused_scaled_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (32usize, 4usize, 2usize, 32usize);
        let scale = 1.0f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5; // deterministic weight/act filler
        let x: Vec<f32> = (0..ne).map(|i| f(i, 0.11) + 0.05).collect();
        // A DIFFERENT router input row — if the seam mistakenly routed on `x`, the top-k pick
        // (and hence the whole output) would diverge from this host reference.
        let rx: Vec<f32> = (0..ne).map(|i| f(i, 0.19) - 0.03).collect();
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        // Fused gate|up: per-expert [2*nff, ne], gate rows first, up rows second.
        let gate_up: Vec<f32> = (0..n_expert * 2 * nff * ne).map(|i| f(i, 0.017)).collect();
        let down: Vec<f32> = (0..n_expert * ne * nff).map(|i| f(i, 0.029)).collect();
        let down_scale: Vec<f32> = (0..n_expert).map(|e| 0.7 + 0.1 * e as f32).collect();
        let (rq, guq, dq) = (q8_0(&router), q8_0(&gate_up), q8_0(&down));
        let (rd, gud, dd) = (deq_q8(&rq), deq_q8(&guq), deq_q8(&dq));
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let silu = |z: f32| z / (1.0 + (-z).exp());
        // router logits over rx (NOT x) → softmax over all experts
        let logits: Vec<f32> = (0..n_expert)
            .map(|e| dot(&rd[e * ne..(e + 1) * ne], &rx))
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
            let gus = e * 2 * nff * ne;
            let ds = e * ne * nff;
            let actv: Vec<f32> = (0..nff)
                .map(|j| {
                    let g = dot(&gud[gus + j * ne..gus + (j + 1) * ne], &x);
                    let u = dot(&gud[gus + (nff + j) * ne..gus + (nff + j + 1) * ne], &x);
                    silu(g) * u
                })
                .collect();
            let w_e = probs[e] / wsum * scale * down_scale[e];
            for (i, wi) in want.iter_mut().enumerate() {
                *wi += w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actv);
            }
        }
        // graph
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![1, ne], DType::F32));
        let rxi = g.input(TensorDesc::new(vec![1, ne], DType::F32));
        let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
        let gui = g.weight(TensorDesc::new(vec![n_expert, 2 * nff, ne], DType::Q8_0));
        let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q8_0));
        let dsi = g.weight(TensorDesc::new(vec![n_expert], DType::F32));
        let yi = g.output(TensorDesc::new(vec![1, ne], DType::F32));
        g.push(Op::MoeFfn {
            x: xi,
            router_x: rxi,
            router: ri,
            gate_exps: gui,
            up_exps: gui, // fused: same handle, never read
            down_exps: di,
            down_scale: Some(dsi),
            fused_gate_up: true,
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
        let rxb = mk(bytemuck::cast_slice(&rx), BufferUsage::Activations);
        let rb = mk(&rq, BufferUsage::Weights);
        let gub = mk(&guq, BufferUsage::Weights);
        let db = mk(&dq, BufferUsage::Weights);
        let dsb = mk(bytemuck::cast_slice(&down_scale), BufferUsage::Weights);
        let yb = be_.alloc(ne * 4, BufferUsage::Activations).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(rxi, rxb.as_ref());
        bind.bind(ri, rb.as_ref());
        bind.bind(gui, gub.as_ref());
        bind.bind(di, db.as_ref());
        bind.bind(dsi, dsb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; ne];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..ne {
            assert!(
                (got[i] - want[i]).abs() < 3e-3,
                "moe_ffn (fused+scaled) mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    // ---- Q5_K helpers (block=256: Q4_K's f16 d + f16 dmin + 12-byte packed 6-bit scale/min plus
    // a 32-byte qh high-bit plane + 128 nibbles = 176 bytes) — mirrors `native_gemm_mmq_q5k`'s
    // decode exactly (qh bit `sub` of byte `l` supplies quant bit 4), so the batched split-gate
    // isolation test below can synthesize a real Q5_K down bank and have the host reference
    // dequant the SAME rounded values the GPU shader reads. Same internal-round-trip-only caveat
    // as the Q4_K helpers above.
    fn q5k(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 256 * 176);
        for blk in x.chunks(256) {
            let mut sc = [0u32; 8];
            let mut mn = [0u32; 8];
            let mut sub_lo = [0f32; 8]; // per-sub-block min
            let mut sub_sc = [0f32; 8]; // per-sub-block (max-min)/31
            for (j, sub) in blk.chunks(32).enumerate() {
                let lo = sub.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                sub_lo[j] = lo;
                sub_sc[j] = ((hi - lo) / 31.0).max(1e-8);
            }
            let d = sub_sc.iter().cloned().fold(0f32, f32::max) / 63.0;
            let dmin = sub_lo
                .iter()
                .cloned()
                .fold(0f32, |m, v| m.max(v.abs()))
                .max(1e-8)
                / 63.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let idmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };
            for j in 0..8 {
                sc[j] = ((sub_sc[j] * id).round() as i32).clamp(0, 63) as u32;
                mn[j] = ((sub_lo[j].abs() * idmin).round() as i32).clamp(0, 63) as u32;
            }
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            out.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
            // Pack the 8 (sc,mn) pairs into 12 bytes (get_scale_min_k4 convention).
            let mut scales = [0u8; 12];
            for k in 0..4 {
                scales[k] = (sc[k] & 0x3F) as u8 | (((sc[4 + k] >> 4) & 0x3) as u8) << 6;
                scales[4 + k] = (mn[k] & 0x3F) as u8 | (((mn[4 + k] >> 4) & 0x3) as u8) << 6;
                scales[8 + k] = (sc[4 + k] & 0xF) as u8 | (((mn[4 + k] & 0xF) as u8) << 4);
            }
            out.extend_from_slice(&scales);
            // Quantize to 5 bits against the recovered (d*sc, dmin*mn); low nibble → qs (sub_even
            // low, sub_odd high per 32-byte pair-region), bit 4 → qh bit `sub`.
            let mut q = [[0u8; 32]; 8];
            for j in 0..8 {
                let scale = d * sc[j] as f32;
                let min = dmin * mn[j] as f32;
                let iscale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                for (l, &v) in blk[j * 32..j * 32 + 32].iter().enumerate() {
                    q[j][l] = (((v - min) * iscale).round() as i32).clamp(0, 31) as u8;
                }
            }
            let mut qh = [0u8; 32];
            for (j, qj) in q.iter().enumerate() {
                for (l, &qv) in qj.iter().enumerate() {
                    qh[l] |= ((qv >> 4) & 1) << j;
                }
            }
            out.extend_from_slice(&qh);
            let mut qs = [0u8; 128];
            for pair in 0..4 {
                let (lo, hi) = (&q[2 * pair], &q[2 * pair + 1]);
                for l in 0..32 {
                    qs[pair * 32 + l] = (lo[l] & 0xF) | ((hi[l] & 0xF) << 4);
                }
            }
            out.extend_from_slice(&qs);
        }
        out
    }
    fn deq_q5k(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 176 * 256);
        for blk in bytes.chunks(176) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
            let scales = &blk[4..16];
            let qh = &blk[16..48];
            let qs = &blk[48..176];
            for j in 0..8u32 {
                let (sc, mn) = if j < 4 {
                    (
                        (scales[j as usize] & 0x3F) as u32,
                        (scales[j as usize + 4] & 0x3F) as u32,
                    )
                } else {
                    let i2 = j as usize - 4;
                    (
                        (scales[j as usize + 4] as u32 & 0xF) | (((scales[i2] as u32) >> 6) << 4),
                        (scales[j as usize + 4] as u32 >> 4)
                            | (((scales[i2 + 4] as u32) >> 6) << 4),
                    )
                };
                let scale = d * sc as f32;
                let min = dmin * mn as f32;
                let pair = (j / 2) as usize;
                let lo = (j % 2) == 0;
                for l in 0..32 {
                    let byte = qs[pair * 32 + l];
                    let nib = if lo { byte & 0xF } else { byte >> 4 };
                    let hb = (qh[l] >> j) & 1;
                    out.push(scale * (nib | (hb << 4)) as f32 - min);
                }
            }
        }
        out
    }

    /// The `rows>moe_small_m_threshold()` (batched, GPU-resident routing) twin of
    /// `moe_ffn_fused_scaled_matches_host`: diffusion-gemma's ACTUAL production dtypes (fused Q4_K
    /// gate_up, not Q8_0 — exercises `matmul_mmq_experts`'s Q4_K path with `gu_width = 2*nff`) +
    /// Q8_0 down (exercises the new `native_gemm_mmq_q8_0_xp` kernel) + per-expert `down_scale`
    /// (exercises the `moe_bucket_scatter_scaled` dscale-into-bucket_wts path) + a separate
    /// `router_x` (exercises the batched routing prologue reading `router_x`, not `x`) + GELU
    /// (gemma's activation, not qwen3moe's SiLU). `ne=256` is the minimum Q4_K superblock width
    /// (256-element blocks); the real model's ne=2816=11×256. Runs rows=5 (below the small-m
    /// threshold — exercises the id-indexed GEMV fast path's multi-row widening, `Op::MoeFfn`'s
    /// OTHER new-code branch introduced alongside it), rows=9 (above threshold, a ragged non-
    /// 64-aligned row count — exercises the batched GEMM's row-tile overread/clip path, the
    /// scenario rows=5 used to cover before the small-m threshold existed), and rows=256
    /// (diffusion-gemma's canvas width). The host reference's int8-activation rounding models the
    /// BATCHED path's `quant_q8`/`quant_q8_gather` step exactly (rows=9, rows=256); for rows=5 the
    /// small-m path skips that quantization (reads `x`/`actv` at full f32 precision via
    /// `linear_native_id_multi`), so the reference only bounds the small-m result from above (its
    /// error is quantization noise being ADDED to the reference, not present in `got`) — still
    /// covered by the shared tolerance below, which was sized for the two-quantization-layer case.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_batched_fused_scaled_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (256usize, 4usize, 2usize, 32usize);
        let scale = 1.1f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5;
        let gelu = |z: f32| 0.5 * z * (1.0 + (0.797_884_6 * (z + 0.044715 * z * z * z)).tanh());
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        // Fused gate|up: per-expert [2*nff, ne], gate rows first, up rows second.
        let gate_up: Vec<f32> = (0..n_expert * 2 * nff * ne).map(|i| f(i, 0.017)).collect();
        let down: Vec<f32> = (0..n_expert * ne * nff).map(|i| f(i, 0.029)).collect();
        let down_scale: Vec<f32> = (0..n_expert).map(|e| 0.7 + 0.1 * e as f32).collect();
        let (rq, guq, dq) = (q8_0(&router), q4k(&gate_up), q8_0(&down));
        let (rd, gud, dd) = (deq_q8(&rq), deq_q4k(&guq), deq_q8(&dq));
        let dsb_bytes = bytemuck::cast_slice(&down_scale).to_vec();

        for &rows in &[5usize, 9usize, 256usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            // A DIFFERENT router input row — if the seam mistakenly routed on `x`, the top-k pick
            // (and hence the whole output) would diverge from this host reference.
            let rx: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.19) - 0.03).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                let rxr = &rx[t * ne..(t + 1) * ne];
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], rxr))
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|p| *p /= psum);
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                // The batched path's expert GEMMs read int8-quantized activations (`quant_q8`/
                // `quant_q8_gather`, symmetric per-32-block — SAME scheme as the `q8_0` weight
                // helper above), NOT full-precision `x` — unlike the per-token path's
                // `linear_native_id_multi` GEMV, which dequants weights against f32 x directly.
                // Round-trip `xr` (gate/up input) and `actv` (down input) through it so the host
                // reference matches what the GPU shader actually reads.
                let xrq = deq_q8(&q8_0(xr));
                for &e in &idx {
                    let gus = e * 2 * nff * ne;
                    let ds = e * ne * nff;
                    let actv: Vec<f32> = (0..nff)
                        .map(|j| {
                            let g = dot(&gud[gus + j * ne..gus + (j + 1) * ne], &xrq);
                            let u = dot(&gud[gus + (nff + j) * ne..gus + (nff + j + 1) * ne], &xrq);
                            gelu(g) * u
                        })
                        .collect();
                    let actvq = deq_q8(&q8_0(&actv));
                    let w_e = probs[e] / wsum * scale * down_scale[e];
                    for i in 0..ne {
                        want[t * ne + i] +=
                            w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actvq);
                    }
                }
            }
            // graph
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let rxi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gui = g.weight(TensorDesc::new(vec![n_expert, 2 * nff, ne], DType::Q4K));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q8_0));
            let dsi = g.weight(TensorDesc::new(vec![n_expert], DType::F32));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: rxi,
                router: ri,
                gate_exps: gui,
                up_exps: gui, // fused: same handle, never read
                down_exps: di,
                down_scale: Some(dsi),
                fused_gate_up: true,
                dst: yi,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: n_used as u32,
                n_ff_exp: nff as u32,
                scale,
                act: Activation::Gelu,
            });
            let mk = |bytes: &[u8], usage| {
                let b = be_.alloc(bytes.len(), usage).unwrap();
                be_.upload(b.as_ref(), bytes).unwrap();
                b
            };
            let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
            let rxb = mk(bytemuck::cast_slice(&rx), BufferUsage::Activations);
            let rb = mk(&rq, BufferUsage::Weights);
            let gub = mk(&guq, BufferUsage::Weights);
            let db = mk(&dq, BufferUsage::Weights);
            let dsb = mk(&dsb_bytes, BufferUsage::Weights);
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(rxi, rxb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gui, gub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(dsi, dsb.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            // Tolerance is looser than the single-quantization-layer precedent tests' 3e-3: this
            // path stacks TWO lossy layers the per-token tests never exercise — Q4_K (4-bit, ~10x
            // coarser than Q8_0) gate/up weights AND int8-quantized activations (`quant_q8`/
            // `quant_q8_gather`) on ne=256 (8x the precedent tests' ne=32) — verified in isolation
            // (`matmul_mmq_experts` alone, skewed non-64-aligned expert counts) at ~0.008 max
            // absolute error on comparable data; this loop's ~0.01-0.02 (measured empirically)
            // compounds that through GELU + a second (down) quantization. 2e-2 stays two orders of
            // magnitude below a wrong-dtype/stride/scale/routing bug (which corrupts a whole
            // expert's O(1) contribution, not a couple percent of it).
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 2e-2,
                    "batched fused-scaled moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// Batched split-gate MoeFfn with a Q5_K down bank — qwen35moe's unsloth-dynamic (UD) quant
    /// shape (Q4_K gate/up + Q5_K down on most layers, SiLU, router_x == x, no down_scale).
    /// Exercises the `native_gemm_mmq_q5k_xp` kernel (min-carrying like Q4_K → binds `sact`)
    /// through the full GPU-resident routing pipeline at rows=9 (ragged, non-64-aligned row
    /// tiles) and rows=256. Host reference structure and int8-activation rounding follow
    /// `moe_ffn_batched_fused_scaled_matches_host` (see its tolerance rationale).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_batched_split_q5k_down_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (256usize, 4usize, 2usize, 256usize);
        let scale = 1.0f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5;
        let silu = |z: f32| z / (1.0 + (-z).exp());
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        // 0.3x weight amplitude vs the precedent tests: this shape's dots run over 256 terms at
        // BOTH stages (Q5_K's 256-element superblock forces nff=256, vs the fused test's nff=32),
        // and full-amplitude weights push SwiGLU activations into a dynamic range where the
        // per-32-block int8 activation quant (f16-rounded scale + f16-rounded `sact` block sums —
        // neither modeled by the host reference) costs ~0.05-0.19 absolute; damped, both stages
        // stay well-conditioned and the shared 2e-2 tolerance holds with margin.
        let gate: Vec<f32> = (0..n_expert * nff * ne)
            .map(|i| f(i, 0.017) * 0.3)
            .collect();
        let up: Vec<f32> = (0..n_expert * nff * ne)
            .map(|i| f(i, 0.023) * 0.3)
            .collect();
        let down: Vec<f32> = (0..n_expert * ne * nff)
            .map(|i| f(i, 0.029) * 0.3)
            .collect();
        let (rq, gq, uq, dq) = (q8_0(&router), q4k(&gate), q4k(&up), q5k(&down));
        let (rd, gd, ud, dd) = (deq_q8(&rq), deq_q4k(&gq), deq_q4k(&uq), deq_q5k(&dq));

        for &rows in &[9usize, 256usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], xr))
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|p| *p /= psum);
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                // The batched path reads int8-quantized activations (quant_q8_gather for gate/up,
                // quant_q8 for the down input) — round-trip both so the reference matches.
                let xrq = deq_q8(&q8_0(xr));
                for &e in &idx {
                    let gs = e * nff * ne;
                    let ds = e * ne * nff;
                    let actv: Vec<f32> = (0..nff)
                        .map(|j| {
                            let g = dot(&gd[gs + j * ne..gs + (j + 1) * ne], &xrq);
                            let u = dot(&ud[gs + j * ne..gs + (j + 1) * ne], &xrq);
                            silu(g) * u
                        })
                        .collect();
                    let actvq = deq_q8(&q8_0(&actv));
                    let w_e = probs[e] / wsum * scale;
                    for i in 0..ne {
                        want[t * ne + i] +=
                            w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actvq);
                    }
                }
            }
            // graph
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gi = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
            let ui = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q5K));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: xi,
                router: ri,
                gate_exps: gi,
                up_exps: ui,
                down_exps: di,
                down_scale: None,
                fused_gate_up: false,
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
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gi, gb.as_ref());
            bind.bind(ui, ub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 2e-2,
                    "batched split q5k-down moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// Batched split-gate MoeFfn with NON-Q4_K gate/up — the shape this test targets is exactly
    /// what the batched path rejected before the `matmul_mmq_experts` dtype widening (adapter.rs
    /// `mmq_ok`): gate=Q5_K (min-carrying → binds `sact`, same kernel family as Q4_K),
    /// up=Q8_0 (symmetric → NO `sact`), down=Q4_K. Deliberately gives gate and up DIFFERENT
    /// dtypes with DIFFERENT `sact` requirements (unlike `moe_ffn_batched_split_q5k_down_matches_host`,
    /// where gate/up share one dtype) — this is the case that would silently corrupt output (or
    /// hit a binding-count mismatch) if `gate_needs_sact`/`up_needs_sact` weren't threaded
    /// per-role and independently from `down_needs_sact`. Host reference structure and
    /// int8-activation rounding follow `moe_ffn_batched_fused_scaled_matches_host`.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_batched_split_q5k_gate_up_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (256usize, 4usize, 2usize, 256usize);
        let scale = 1.0f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5;
        let silu = |z: f32| z / (1.0 + (-z).exp());
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        // 0.3x amplitude — same rationale as `moe_ffn_batched_split_q5k_down_matches_host`
        // (256-term dots at both stages need damping to stay in the int8-activation-quant's
        // well-conditioned range).
        let gate: Vec<f32> = (0..n_expert * nff * ne)
            .map(|i| f(i, 0.017) * 0.3)
            .collect();
        let up: Vec<f32> = (0..n_expert * nff * ne)
            .map(|i| f(i, 0.023) * 0.3)
            .collect();
        let down: Vec<f32> = (0..n_expert * ne * nff)
            .map(|i| f(i, 0.029) * 0.3)
            .collect();
        let (rq, gq, uq, dq) = (q8_0(&router), q5k(&gate), q8_0(&up), q4k(&down));
        let (rd, gd, ud, dd) = (deq_q8(&rq), deq_q5k(&gq), deq_q8(&uq), deq_q4k(&dq));

        for &rows in &[9usize, 256usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], xr))
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|p| *p /= psum);
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                let xrq = deq_q8(&q8_0(xr));
                for &e in &idx {
                    let gs = e * nff * ne;
                    let ds = e * ne * nff;
                    let actv: Vec<f32> = (0..nff)
                        .map(|j| {
                            let g = dot(&gd[gs + j * ne..gs + (j + 1) * ne], &xrq);
                            let u = dot(&ud[gs + j * ne..gs + (j + 1) * ne], &xrq);
                            silu(g) * u
                        })
                        .collect();
                    let actvq = deq_q8(&q8_0(&actv));
                    let w_e = probs[e] / wsum * scale;
                    for i in 0..ne {
                        want[t * ne + i] +=
                            w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actvq);
                    }
                }
            }
            // graph
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gi = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q5K));
            let ui = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q8_0));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q4K));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: xi,
                router: ri,
                gate_exps: gi,
                up_exps: ui,
                down_exps: di,
                down_scale: None,
                fused_gate_up: false,
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
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gi, gb.as_ref());
            bind.bind(ui, ub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 2e-2,
                    "batched split q5k/q8_0 gate-up moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }
}
