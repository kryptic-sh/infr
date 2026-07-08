//! Single-command-buffer forward recorder. Records many op dispatches (linear, rmsnorm,
//! rope, attention, silu_mul, add) into ONE command buffer with a conservative global
//! compute→compute barrier between each, then submits once on `finish()`. This collapses
//! the ~per-op submit+wait round-trips (the real bottleneck) to one per forward.
//!
//! Buffers are caller-owned `Box<dyn Buffer>` (from `Backend::alloc`); the recorder only
//! binds them. Reuses the cached kernels in `VulkanShared.kernels`.

use std::cell::RefCell;
use std::collections::HashSet;

use ash::vk;

use infr_core::{backend::Buffer, error::Result};

use super::ops::ComputeKernel;
use super::{as_vk_buf, be, VulkanBackend};

/// Output rows computed per workgroup by the subgroup decode GEMV (`mul_mat_vec_q.comp`); must match
/// the shader's `NUM_ROWS`.
const MMV_NUM_ROWS: u32 = 1;

/// `matmul_mmq_experts`' BM=64 → BM=32 row-tile crossover, in average rows-per-expert
/// (`rows·n_used/n_expert`). Below this, the BM=32 `_xp32` kernel wins (less masked-tile waste);
/// at/above it BM=64 (default) wins. Picked from `moe_expert_grid_bound_bench`'s tile sweep —
/// bumped past qwen3.6-MoE's pp512 avg (~16/expert) but below qwen3-30B-A3B's (~32/expert), so a
/// deeper-context prefill that pushes avg rows/expert up for either model naturally slides back to
/// BM=64 as the segments fill in.
const MOE_EXPERT_SMALL_TILE_AVG_ROWS: usize = 24;

/// Dense A_GLOBAL warp-GEMM `matmul_native_f16a` (n128_ag, non-split-K family) BM=64 → BM=32
/// row-tile crossover, in real batched-prefill rows `m`. MTP verify's draft window runs a batched
/// prefill of the drafted suffix (m≈6-8 steady-state, growing up to ~m30-50 under the no-rewind
/// fallback's carried-forward unaccepted prefix) through every dense projection GEMM on the SAME
/// BM=64 tile prefill uses at m=512+ — at m=8 that's 87.5% masked waste (56 of 64 rows discarded
/// at the store). Below this threshold the BM=32 `_bm32` kernel variant wins (halves the waste,
/// same per-thread math as `matmul_mmq_experts`' `_xp32`); at/above it BM=64 (default, unchanged)
/// wins — picked from `dense_small_m_row_tile_bench`'s crossover sweep (BM=32 wins clearly through
/// m=24, a wash by m=32, a loss by m=64). `INFR_NO_SMALL_BM` forces BM=64 (A/B escape). Same
/// K-accumulation order either way — tile GRANULARITY only, bit-identical to the BM=64 path.
///
/// NOT applied to `matmul_native_splitk`'s sk_ag family (o/down/kv-proj at these same m — the
/// OTHER half of verify's dense GEMM traffic): the same bench shows a net LOSS there across the
/// whole m≈4-64 sweep — split-K's own `splits` dimension already fans the GEMM across enough
/// workgroups to fill the device, so a smaller row tile only adds fixed per-workgroup cost with
/// no occupancy win to pay for it (see `matmul_native_splitk`'s doc).
const DENSE_SMALL_TILE_MAX_M: usize = 24;

/// Elements packed per u32 weight word for a given quant width (8 nibbles for q4, 4 bytes for q8).
/// `None` ⇒ the subgroup GEMV has no specialization for this width; caller uses the WGSL fallback.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn mmv_epw(bits: u32) -> Option<usize> {
    match bits {
        4 => Some(8),
        8 => Some(4),
        _ => None,
    }
}

/// Rows-per-workgroup for the multi-output-row decode GEMV, or `None` to keep the RM=1 path.
/// The RM variant packs RM output rows into one workgroup (RM× the in-flight weight streams per
/// wave) to feed enough MLP at low `out_f`, where the RM=1 grid (out_f workgroups) is too shallow
/// to hide DRAM latency — the big GEMVs (lm_head/gate+up, out_f ≥ ~16k) already saturate on RM=1
/// so they stay there. Only Q4_K/Q6_K have RM builds ([`crate::gemm::native_rm_build_spv`]).
/// Tunable for A/B: `INFR_NO_GEMV_RM` forces the RM=1 path; `INFR_GEMV_RM`=2|4 forces the factor;
/// `INFR_GEMV_RM_MAXOUT` overrides the out_f gate.
fn native_rm_choice(dtype: infr_core::DType, out_f: usize) -> Option<u32> {
    use infr_core::DType::*;
    if !matches!(dtype, Q4K | Q6K) || std::env::var("INFR_NO_GEMV_RM").is_ok() {
        return None;
    }
    let max_out = std::env::var("INFR_GEMV_RM_MAXOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    // Below ~2k outputs the RM=1 grid is already shallow; halving it starves the machine (k/v,
    // out_f=1024, regressed in the cold A/B), so those stay on RM=1.
    let min_out = std::env::var("INFR_GEMV_RM_MINOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048usize);
    if out_f > max_out || out_f < min_out {
        return None;
    }
    let rm = std::env::var("INFR_GEMV_RM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2u32);
    (rm == 2 || rm == 4).then_some(rm)
}

/// NUM_ROWS for the reassociation-tolerant subgroup decode GEMV (`native_gemv_sg.comp`), or `None`
/// to keep the bit-identical tree/RM path. The subgroup form (wave32 + subgroupAdd, NR output rows)
/// keeps 32×NR weight-load requests in flight per wave — the extra MLP that the shallow low-out_f
/// projection grid (q/k/v/o/down) needs to saturate DRAM. It REORDERS the accumulation (not
/// bit-identical), so it is gated to the latency-starved projection BAND ONLY: the big lm_head /
/// gate+up GEMVs (out_f ≥ ~16k) already saturate on the tree kernel and subgroupAdd measured a ~20%
/// occupancy regression there. Q6_K ONLY (see below) — no Q4_K SG build exists.
/// Env: `INFR_NO_GEMV_SG` forces the tree/RM path; `INFR_GEMV_SG_NR`=2|4|8 overrides NR;
/// `INFR_GEMV_SG_MINOUT`/`INFR_GEMV_SG_MAXOUT` override the band.
fn native_sg_choice(dtype: infr_core::DType, in_f: usize, out_f: usize) -> Option<u32> {
    use infr_core::DType::*;
    // Q6_K ONLY: the heavier Q6_K decode (210-byte super-blocks, more unpack ALU) is where wave32 +
    // subgroupAdd's lower register/barrier overhead nets out. On Q4_K the tree/RM kernel already
    // saturates and SG regressed at every measured shape (decode_gemv_bw A/B). Q6_K also loses below
    // out_f≈2048 (attn_v, out_f=1024: grid too shallow) and above the band (lm_head, out_f=151936:
    // already peak on the tree kernel + subgroupAdd occupancy regression) — so the band is the
    // out_f≈4096 Q6_K projections (ffn_down / o). Default NR=2 (best at every winning shape).
    if !matches!(dtype, Q6K) || std::env::var("INFR_NO_GEMV_SG").is_ok() {
        return None;
    }
    // wave32 lanes stride 32-elem sub-blocks; in_f must be a multiple of 32 (all projections are).
    if !in_f.is_multiple_of(32) {
        return None;
    }
    let min_out = std::env::var("INFR_GEMV_SG_MINOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048usize);
    let max_out = std::env::var("INFR_GEMV_SG_MAXOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192usize);
    if out_f < min_out || out_f > max_out {
        return None;
    }
    let nr = std::env::var("INFR_GEMV_SG_NR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2u32);
    matches!(nr, 2 | 4 | 8).then_some(nr)
}

/// NUM_ROWS for the reassociation-tolerant subgroup id-indexed decode GEMV
/// (`native_gemv_id_multi_sg.comp`), or `None` to keep the tree id kernel. Same projection-band
/// discipline as [`native_sg_choice`], but covers BOTH Q6_K (Qwen3-30B expert down, out_f≈2048 — the
/// largest SG win of all decode shapes) AND Q5_K (Qwen3.6-A3B UD-quant stores most expert downs as
/// Q5_K at the same out_f≈2048 shape; A/B-confirmed a win — Q4_K idm already saturates and stays on
/// the tree). Kept independent of `native_sg_choice` so the byte-identical dense SG path is untouched.
/// Env: `INFR_NO_GEMV_ID_SG` forces the tree id path (id-only escape); `INFR_NO_GEMV_SG` disables BOTH
/// the dense and id SG routes; `INFR_GEMV_SG_MINOUT`/`MAXOUT`/`INFR_GEMV_SG_NR` are the shared knobs.
fn native_id_sg_choice(dtype: infr_core::DType, in_f: usize, out_f: usize) -> Option<u32> {
    use infr_core::DType::*;
    if std::env::var("INFR_NO_GEMV_ID_SG").is_ok() || std::env::var("INFR_NO_GEMV_SG").is_ok() {
        return None;
    }
    // Heavy K-quant decodes only (Q6_K/Q5_K): the extra unpack ALU is where wave32 + subgroupAdd's
    // lower register/barrier overhead nets out. Q4_K's tree id kernel already saturates (SG regressed).
    if !matches!(dtype, Q6K | Q5K) {
        return None;
    }
    // wave32 lanes stride 32-elem sub-blocks; in_f must be a multiple of 32 (all projections are).
    if !in_f.is_multiple_of(32) {
        return None;
    }
    let min_out = std::env::var("INFR_GEMV_SG_MINOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048usize);
    let max_out = std::env::var("INFR_GEMV_SG_MAXOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192usize);
    if out_f < min_out || out_f > max_out {
        return None;
    }
    let nr = std::env::var("INFR_GEMV_SG_NR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2u32);
    matches!(nr, 2 | 4 | 8).then_some(nr)
}

pub struct Recorder<'a> {
    be: &'a VulkanBackend,
    cmd: vk::CommandBuffer,
    /// Descriptor pools, GROWN on exhaustion (a batched-MoE prefill chunk records ~50k dispatches
    /// — far beyond any fixed max_sets). The last entry is the active pool; `alloc_set` appends a
    /// fresh one on ERROR_OUT_OF_POOL_MEMORY.
    pools: std::cell::RefCell<Vec<vk::DescriptorPool>>,
    /// Buffers written since the last barrier (for read-after-write / write-after-write detection).
    dirty_writes: RefCell<HashSet<vk::Buffer>>,
    /// Buffers read since the last barrier (for write-after-read detection).
    dirty_reads: RefCell<HashSet<vk::Buffer>>,
    /// Whether any un-barriered write was produced by a transfer (copy) rather than a shader.
    dirty_transfer: std::cell::Cell<bool>,
    barriers: RefCell<usize>,
    /// Set while recording an indirect dispatch so `sync` widens the barrier to cover the
    /// indirect-command read of GPU-written dispatch args.
    indirect_pending: std::cell::Cell<bool>,
    /// Debug knobs, read once (avoid env lookups in the per-dispatch hot path).
    no_barrier: bool,
    full_barrier: bool,
    prof: bool,
    /// Per-op GPU timestamp profiling (INFR_PROF2): a timestamp before EVERY dispatch — labeled
    /// automatically with the dispatched kernel's name at the `dispatch3`/`dispatch_indirect`
    /// chokepoints — plus one at finish. No manual stamp calls; aggregation by label happens at
    /// report time.
    prof2: bool,
    /// INFR_PROF2_SHAPES (on top of INFR_PROF2): GEMV/GEMM dispatches get shape-itemized labels
    /// (`kind:mM:KxN`) instead of the bare kernel name. Read once at construction.
    prof2_shapes: bool,
    query_pool: vk::QueryPool,
    ts_labels: RefCell<Vec<&'static str>>,
    /// Dispatches past the query-pool capacity: counted (and reported) instead of stamped.
    ts_dropped: std::cell::Cell<usize>,
    /// One-shot prof2 label override: the NEXT dispatch's timestamp uses this instead of the
    /// kernel name, then clears it. Lets a caller attribute a generic kernel (e.g. the lm_head
    /// GEMM vs a projection, expert gate_up vs down) to a distinct bucket without per-op API
    /// plumbing.
    next_label: std::cell::Cell<Option<&'static str>>,
    /// Record-once: when set, the command buffer is begun resubmittable (no ONE_TIME_SUBMIT) and
    /// `finish_record` hands back its cmd buffer + pool (a [`RecordedCmd`]) instead of submitting and
    /// freeing — so the caller can replay it across tokens.
    persistent: bool,
    /// Creation time — INFR_PROF prints the host record time vs the submit+GPU wait in `finish`.
    t0: std::time::Instant,
    /// See [`Self::suppress_sync`]: while set, `sync` accumulates hazards without emitting.
    suppress: std::cell::Cell<bool>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl<'a> Recorder<'a> {
    pub(crate) fn new(backend: &'a VulkanBackend) -> Result<Self> {
        Self::new_inner(backend, false)
    }

    /// A recorder whose command buffer is resubmittable (no ONE_TIME_SUBMIT). `finish_record` returns
    /// a [`RecordedCmd`] the caller can replay instead of re-recording. Profiling is disabled on this
    /// path (the recorder is gone after recording, so per-replay timestamps can't be reported).
    pub(crate) fn new_persistent(backend: &'a VulkanBackend) -> Result<Self> {
        Self::new_inner(backend, true)
    }

    fn new_inner(backend: &'a VulkanBackend, persistent: bool) -> Result<Self> {
        let device = &backend.shared.device;
        let cmd_pool = *backend.shared.cmd_pool.lock().unwrap();
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| be(format!("alloc cmd buffer: {e}")))?[0];
        let begin_flags = if persistent {
            // SIMULTANEOUS_USE: the chained decode submits the SAME recorded command buffer n
            // times in one vkQueueSubmit (see RecordedCmd::replay_n) — the buffer must be legal
            // in the pending state more than once.
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE
        } else {
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        };
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(begin_flags),
            )
        }
        .map_err(|e| be(format!("begin cmd buffer: {e}")))?;

        // Big pool: one descriptor set per recorded op.
        let pool = Self::new_desc_pool(device)?;

        // Per-DISPATCH stamps (auto-labels): a batched prefill chunk can record tens of
        // thousands of dispatches; 64k × 8 B = 512 KiB of query pool, profiling runs only.
        // Overflow is counted (`ts_dropped`) and reported, never UB.
        const MAX_TS: u32 = Recorder::<'_>::MAX_TS;
        // No per-op profiling on the persistent (replayed) path — the recorder is dropped after
        // recording, so it can't report timestamps for replays.
        let prof2 = std::env::var("INFR_PROF2").is_ok() && !persistent;
        let prof2_shapes = prof2 && std::env::var("INFR_PROF2_SHAPES").is_ok();
        let query_pool = if prof2 {
            let qp = unsafe {
                device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(MAX_TS),
                    None,
                )
            }
            .map_err(|e| be(format!("create query pool: {e}")))?;
            unsafe { device.cmd_reset_query_pool(cmd, qp, 0, MAX_TS) };
            qp
        } else {
            vk::QueryPool::null()
        };

        Ok(Self {
            be: backend,
            cmd,
            pools: std::cell::RefCell::new(vec![pool]),
            dirty_writes: RefCell::new(HashSet::new()),
            dirty_reads: RefCell::new(HashSet::new()),
            dirty_transfer: std::cell::Cell::new(false),
            barriers: RefCell::new(0),
            indirect_pending: std::cell::Cell::new(false),
            no_barrier: std::env::var("INFR_NOBARRIER").is_ok(),
            full_barrier: std::env::var("INFR_FULLBARRIER").is_ok(),
            prof: std::env::var("INFR_PROF").is_ok(),
            prof2,
            prof2_shapes,
            query_pool,
            ts_labels: RefCell::new(Vec::new()),
            ts_dropped: std::cell::Cell::new(0),
            next_label: std::cell::Cell::new(None),
            persistent,
            t0: std::time::Instant::now(),
            suppress: std::cell::Cell::new(false),
        })
    }

    /// Disjoint-batch barrier suppression: while ON, recorded dispatches accumulate hazard state
    /// but emit NO pipeline barriers. For batches whose members are KNOWN to touch disjoint
    /// regions of the same buffers (the batched-MoE per-expert stage loop — 128 experts' gathers
    /// all write `xe`, each at its routed offset). Leave the batch's FIRST dispatch unsuppressed
    /// so the stage orders after its producers, and turn suppression OFF after the batch (the
    /// next normal dispatch then fences the whole batch with ONE barrier).
    pub fn suppress_sync(&self, on: bool) {
        self.suppress.set(on);
    }

    /// Override the label of the NEXT dispatch's timestamp (INFR_PROF2). Consumed once — a
    /// pre-set override wins over the automatic kernel-name label AND over the shape itemizers
    /// below. No-op without prof2.
    pub fn label_next(&self, label: &'static str) {
        if self.prof2 {
            self.next_label.set(Some(label));
        }
    }

    /// Shape-itemize the next GEMV-class dispatch: with INFR_PROF2_SHAPES set (on top of
    /// INFR_PROF2), its label becomes `kind:mM:in_fxout_f` (e.g. `mmvr:m4:1536x24576`) instead of
    /// the bare kernel name, splitting the bucket per route and per projection shape. This is
    /// what made the E2B pp4 63%-GEMV attribution actionable (llama.cpp's perf logger itemizes
    /// per shape; a flat bucket didn't). A pending `label_next` override wins. The labels leak
    /// one small string per op per record — profiling-only, gated off in production.
    fn label_gemv(&self, kind: &'static str, rows: usize, in_f: usize, out_f: usize) {
        if self.prof2_shapes && self.next_label.get().is_none() {
            let s: &'static str =
                Box::leak(format!("{kind}:m{rows}:{in_f}x{out_f}").into_boxed_str());
            self.next_label.set(Some(s));
        }
    }

    /// Shape-itemize the next GEMM-class dispatch (`kernel:mM:kxn` under INFR_PROF2_SHAPES —
    /// same rationale as [`label_gemv`](Self::label_gemv)). Profiling-only; the leak is gated
    /// off in production.
    fn label_gemm(&self, kernel: &'static str, m: usize, k: usize, n: usize) {
        if self.prof2_shapes && self.next_label.get().is_none() {
            let s: &'static str = Box::leak(format!("{kernel}:m{m}:{k}x{n}").into_boxed_str());
            self.next_label.set(Some(s));
        }
    }

    /// Create one descriptor pool tranche (the chain grows by these on exhaustion).
    fn new_desc_pool(device: &ash::Device) -> Result<vk::DescriptorPool> {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 16384,
        }];
        unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(4096)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|e| be(format!("create recorder pool: {e}")))
    }

    /// Allocate a descriptor set from the active pool, growing the chain when it runs dry.
    fn alloc_set(&self, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let device = &self.be.shared.device;
        let try_alloc = |pool: vk::DescriptorPool| unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(std::slice::from_ref(&layout)),
            )
        };
        let cur = *self.pools.borrow().last().expect("≥1 descriptor pool");
        match try_alloc(cur) {
            Ok(sets) => sets[0],
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                let fresh = Self::new_desc_pool(device).expect("grow descriptor pool");
                self.pools.borrow_mut().push(fresh);
                try_alloc(fresh).expect("alloc descriptor set (fresh pool)")[0]
            }
            Err(e) => panic!("alloc descriptor set: {e}"),
        }
    }

    /// Query-pool capacity (timestamps per recorder). Stamps past `MAX_TS - 1` (one slot is
    /// reserved for the closing timestamp in `finish`) are dropped and counted.
    const MAX_TS: u32 = 65536;

    /// Record a profiling timestamp (BOTTOM_OF_PIPE) tagged with an op label, if INFR_PROF2.
    /// Called automatically at the dispatch chokepoints with the kernel name — op methods no
    /// longer stamp by hand (use [`label_next`](Self::label_next) to override a too-generic
    /// kernel name).
    fn stamp(&self, label: &'static str) {
        if !self.prof2 {
            return;
        }
        let label = self.next_label.take().unwrap_or(label);
        let idx = self.ts_labels.borrow().len() as u32;
        if idx >= Self::MAX_TS - 1 {
            self.ts_dropped.set(self.ts_dropped.get() + 1);
            return;
        }
        unsafe {
            self.be.shared.device.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                idx,
            );
        }
        self.ts_labels.borrow_mut().push(label);
    }

    /// Emit a global compute/transfer barrier only if `reads`/`writes` collide with work recorded
    /// since the last barrier (RAW / WAR / WAW). Independent dispatches then overlap on the GPU.
    /// `dst_transfer` = this op consumes via a transfer (copy) rather than a compute shader.
    /// Returns once any required barrier is recorded; updates the hazard-tracking sets.
    fn sync(&self, reads: &[vk::Buffer], writes: &[vk::Buffer], dst_transfer: bool) {
        if self.no_barrier {
            return;
        }
        // Suppressed (disjoint-batch) dispatch: accumulate the hazard state but emit NO barrier —
        // the caller guarantees this dispatch touches only regions disjoint from the batch's other
        // members (per-expert MoE stage loop). The batch's first dispatch runs unsuppressed, so
        // the stage as a whole still orders after its producers; the NEXT unsuppressed dispatch
        // sees the accumulated dirty state and fences the whole batch at once.
        if self.suppress.get() {
            self.dirty_reads.borrow_mut().extend(reads.iter().copied());
            self.dirty_writes
                .borrow_mut()
                .extend(writes.iter().copied());
            return;
        }
        let dw = self.dirty_writes.borrow();
        let dr = self.dirty_reads.borrow();
        let raw = self.full_barrier || reads.iter().any(|b| dw.contains(b));
        let waw = writes.iter().any(|b| dw.contains(b));
        let war = writes.iter().any(|b| dr.contains(b));
        drop(dw);
        drop(dr);
        if raw || waw || war {
            // Access flags must only name flags supported by a stage present in the paired stage
            // mask (VUID-vkCmdPipelineBarrier-srcAccessMask-02815 / -dstAccessMask-02816) — e.g.
            // TRANSFER_WRITE/TRANSFER_READ are only valid alongside PipelineStageFlags::TRANSFER.
            // The old code set the access masks unconditionally (always including the TRANSFER
            // bits) while only conditionally widening the STAGE masks, so a pure compute→compute
            // barrier (the common case) claimed TRANSFER_WRITE/TRANSFER_READ access against a
            // COMPUTE_SHADER-only stage mask — a real spec violation RADV tolerates but Intel's
            // Mesa ANV validation layer (and, per spec, any conformant driver) rejects. Build both
            // masks from the SAME conditions so they can never drift apart again.
            let mut src = vk::PipelineStageFlags::COMPUTE_SHADER;
            let mut src_access = vk::AccessFlags::SHADER_WRITE;
            if self.dirty_transfer.get() {
                src |= vk::PipelineStageFlags::TRANSFER;
                src_access |= vk::AccessFlags::TRANSFER_WRITE;
            }
            let mut dst = vk::PipelineStageFlags::COMPUTE_SHADER;
            let mut dst_access = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
            if dst_transfer {
                dst |= vk::PipelineStageFlags::TRANSFER;
                dst_access |= vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE;
            }
            if self.indirect_pending.get() {
                // The consumer reads GPU-written dispatch args: cover the indirect-command read.
                dst |= vk::PipelineStageFlags::DRAW_INDIRECT;
                dst_access |= vk::AccessFlags::INDIRECT_COMMAND_READ;
            }
            let mb = vk::MemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(dst_access);
            unsafe {
                self.be.shared.device.cmd_pipeline_barrier(
                    self.cmd,
                    src,
                    dst,
                    vk::DependencyFlags::empty(),
                    &[mb],
                    &[],
                    &[],
                );
            }
            self.dirty_writes.borrow_mut().clear();
            self.dirty_reads.borrow_mut().clear();
            self.dirty_transfer.set(false);
            *self.barriers.borrow_mut() += 1;
        }
        self.dirty_reads.borrow_mut().extend(reads.iter().copied());
        self.dirty_writes
            .borrow_mut()
            .extend(writes.iter().copied());
    }

    fn dispatch(
        &self,
        k: ComputeKernel,
        buffers: &[vk::Buffer],
        n_out: usize,
        push: &[u8],
        groups: u32,
    ) {
        self.dispatch3(k, buffers, n_out, push, groups, 1, 1);
    }

    /// The spec-guaranteed-minimum `VkPhysicalDeviceLimits::maxComputeWorkGroupCount[0]` (every
    /// conformant Vulkan implementation supports at least this many workgroups in dimension 0).
    /// Used as the safe split point below rather than threading the ACTUAL queried device limit
    /// through every GEMV call site — RDNA3/most desktop GPUs report a much larger x-limit, but an
    /// Intel A770 (Mesa ANV) enforces exactly the guaranteed minimum, and a wide GEMV (a
    /// ~150k-vocab lm_head, or `rows * gate_up_intermediate` at multi-row prefill) can dispatch
    /// past it in one dimension — the segfault-adjacent bug this splits around.
    const MAX_GROUP_COUNT_X: u32 = 65_535;

    /// Dispatch a FLAT `n`-workgroup 1-D grid, splitting into a safe 2-D grid
    /// (`MAX_GROUP_COUNT_X` × `ceil(n / MAX_GROUP_COUNT_X)`) whenever `n` would otherwise exceed
    /// the guaranteed limit. Every paired shader recovers the flat index as
    /// `gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x` (identical to `gl_WorkGroupID.x`
    /// when `gy == 1`, i.e. a total no-op for every dispatch that already fit in one dimension —
    /// which is every dispatch on hardware/models exercised so far) and bounds-checks it against
    /// the true element count, since `gx * gy` can overshoot `n` when `n` isn't a multiple of
    /// `gx` — those padding workgroups must return immediately without touching any buffer.
    fn dispatch_wide(
        &self,
        k: ComputeKernel,
        buffers: &[vk::Buffer],
        n_out: usize,
        push: &[u8],
        n: u32,
    ) {
        if n <= Self::MAX_GROUP_COUNT_X {
            self.dispatch3(k, buffers, n_out, push, n, 1, 1);
        } else {
            let gy = n.div_ceil(Self::MAX_GROUP_COUNT_X);
            if std::env::var("INFR_DEBUG_WIDE_DISPATCH").is_ok() {
                eprintln!(
                    "[infr] dispatch_wide SPLIT kernel={:?} n={n} -> gx={} gy={gy}",
                    k.name,
                    Self::MAX_GROUP_COUNT_X
                );
            }
            self.dispatch3(k, buffers, n_out, push, Self::MAX_GROUP_COUNT_X, gy, 1);
        }
    }

    /// Bind `k`'s pipeline and all of `buffers` to descriptor set 0. `VK_KHR_push_descriptor`
    /// (when the device has it) records the bindings straight into the command buffer with one
    /// `cmd_push_descriptor_set` call — no host-side descriptor-pool allocation or
    /// `vkUpdateDescriptorSets` syscall, which the pooled path below pays on EVERY dispatch. That
    /// per-dispatch churn measured as a real chunk of the host-side (non-GPU-timestamped) gap at
    /// small-m shapes (many-op graphs where GPU busy time is small — PERF.md class 4). Falls back
    /// to the pooled alloc_set + update_descriptor_sets + cmd_bind_descriptor_sets sequence when
    /// the extension is unavailable; `k.ds_layout` was built to match (see `ops.rs`).
    fn bind_descriptors(&self, k: ComputeKernel, buffers: &[vk::Buffer]) {
        let device = &self.be.shared.device;
        unsafe { device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline) };
        let infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|&buffer| vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();
        if let Some(pd) = &self.be.shared.push_descriptor {
            let ds_writes: Vec<vk::WriteDescriptorSet> = (0..buffers.len())
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&infos[i..i + 1])
                })
                .collect();
            unsafe {
                pd.cmd_push_descriptor_set(
                    self.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    k.pipeline_layout,
                    0,
                    &ds_writes,
                );
            }
        } else {
            let set = self.alloc_set(k.ds_layout);
            let ds_writes: Vec<vk::WriteDescriptorSet> = (0..buffers.len())
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&infos[i..i + 1])
                })
                .collect();
            unsafe {
                device.update_descriptor_sets(&ds_writes, &[]);
                device.cmd_bind_descriptor_sets(
                    self.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    k.pipeline_layout,
                    0,
                    &[set],
                    &[],
                );
            }
        }
    }

    /// Like [`Self::dispatch`], but the workgroup count comes from `args` (a GPU-written
    /// `[gx,gy,gz]` u32 triple at offset 0 — vkCmdDispatchIndirect). `args` joins the hazard reads
    /// so the barrier after its producer covers the indirect-command read (the barrier's dst stage
    /// widens to DRAW_INDIRECT whenever an indirect consumer follows).
    fn dispatch_indirect(
        &self,
        k: ComputeKernel,
        buffers: &[vk::Buffer],
        n_out: usize,
        push: &[u8],
        args: vk::Buffer,
        args_off: u64,
    ) {
        // Auto-label (INFR_PROF2): every dispatch stamps a timestamp tagged with its kernel name
        // — the chokepoint knows the kernel, so no op method needs a manual stamp call. Placed
        // before `sync` so a barrier's cost lands in the op it fences (same as the old
        // stamp-at-op-start convention). One `prof2` branch when profiling is off.
        self.stamp(k.name);
        let split = buffers.len() - n_out;
        let (reads, writes) = buffers.split_at(split);
        let mut all_reads: Vec<vk::Buffer> = reads.to_vec();
        all_reads.push(args);
        self.indirect_pending.set(true);
        self.sync(&all_reads, writes, false);
        self.indirect_pending.set(false);
        self.bind_descriptors(k, buffers);
        let device = &self.be.shared.device;
        unsafe {
            if k.push_size > 0 {
                device.cmd_push_constants(
                    self.cmd,
                    k.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            device.cmd_dispatch_indirect(self.cmd, args, args_off);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch3(
        &self,
        k: ComputeKernel,
        buffers: &[vk::Buffer],
        n_out: usize,
        push: &[u8],
        gx: u32,
        gy: u32,
        gz: u32,
    ) {
        // Auto-label (INFR_PROF2): see `dispatch_indirect` — kernel-name timestamp per dispatch.
        self.stamp(k.name);
        // The last `n_out` bound buffers are outputs; the rest are inputs. Inputs keep in-place
        // buffers (e.g. rope x==y) so a RAW from a prior op is still seen.
        let split = buffers.len() - n_out;
        let (reads, writes) = buffers.split_at(split);
        self.sync(reads, writes, false);
        self.bind_descriptors(k, buffers);
        let device = &self.be.shared.device;

        unsafe {
            if k.push_size > 0 {
                device.cmd_push_constants(
                    self.cmd,
                    k.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            device.cmd_dispatch(self.cmd, gx, gy, gz);
        }
    }

    fn vkb(b: &dyn Buffer) -> vk::Buffer {
        unsafe { as_vk_buf(b) }.buffer
    }

    /// `y[rows,out] = x[rows,in] · Wᵀ` (W stored `[out,in]`).
    pub fn linear(
        &self,
        w: &dyn Buffer,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.label_gemv("lin_f16", rows, in_f, out_f);
        let k = self
            .be
            .kernel("linear_f16", crate::gemm::linear_f16_spv(), 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[Self::vkb(w), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (rows * out_f) as u32, // one workgroup per output element (coalesced GEMV)
        );
    }

    /// `y[rows,out] = x[rows,in] · Wᵀ` (W stored `[out,in]` f16) — the `!caps.f16` tier: same shape
    /// as [`Self::linear`] but the weight buffer is read as packed u32 + `unpackHalf2x16` instead
    /// of a `float16_t` SSBO element, so it needs no shaderFloat16/Float16 SPIR-V capability. See
    /// `linear_f16_noext.comp`. Correctness-first — not row-tiled like the f32 mrow variants.
    pub fn linear_f16_noext(
        &self,
        w: &dyn Buffer,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.label_gemv("lin_f16_noext", rows, in_f, out_f);
        let k = self.be.kernel(
            "linear_f16_noext",
            crate::gemm::linear_f16_noext_spv(),
            3,
            12,
        );
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[Self::vkb(w), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// f32-weight GEMV `y = x·Wᵀ` — full-precision projection weights (gemma4 E2B's per-layer
    /// inp_gate/proj and qwen3moe's router ship as F32; reading them through the f16 kernel
    /// produced garbage). Reuses the eager path's thread-per-output `linear_f32` kernel
    /// (dispatch = ceil(rows·out_f/64) groups of 64 threads) — these weights are small, so the
    /// simple kernel is fine.
    pub fn linear_f32(
        &self,
        w: &dyn Buffer,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.label_gemv("lin_f32", rows, in_f, out_f);
        // Prefill (rows>1): the ROW-TILED f32 GEMM reads each weight once per 8 rows (grid
        // out_f·ceil(rows/8)) instead of once per row — bit-identical, cuts the F32-projection
        // weight re-reads (E2B inp_gate/proj, qwen3moe router). Decode (rows==1) keeps the 1-row
        // kernel. INFR_NO_F32_MROW forces the 1-row path (A/B).
        let use_mrow = rows > 1 && std::env::var("INFR_NO_F32_MROW").is_err();
        // vec4 K-stream variants (in_f % 4 == 0): the small-m f32 projections launch only
        // out_f workgroups (E2B inp_gate 1536×256 = 256 wgs ≈ 2.6 waves/CU on a 7900 XTX) —
        // latency-bound, so 4x fewer load round trips is the lever. MROW=4 for rows<=4 (half
        // the idle acc lanes), MROW=8 above. vec4-lane accumulation is NOT bit-identical to
        // the scalar kernels; INFR_NO_F32_V4 forces the scalar mrow8 (A/B).
        let use_v4 = use_mrow && in_f.is_multiple_of(4) && std::env::var("INFR_NO_F32_V4").is_err();
        let (name, spv, groups) = if use_v4 && rows <= 4 {
            (
                "linear_f32r_mrow4_v4",
                crate::gemm::linear_f32r_mrow4_v4_spv(),
                (out_f * rows.div_ceil(4)) as u32,
            )
        } else if use_v4 {
            (
                "linear_f32r_mrow8_v4",
                crate::gemm::linear_f32r_mrow8_v4_spv(),
                (out_f * rows.div_ceil(8)) as u32,
            )
        } else if use_mrow {
            (
                "linear_f32r_mrow8",
                crate::gemm::linear_f32r_mrow8_spv(),
                (out_f * rows.div_ceil(8)) as u32,
            )
        } else {
            (
                "linear_f32r",
                crate::gemm::linear_f32r_spv(),
                (rows * out_f) as u32,
            )
        };
        let k = self.be.kernel(name, spv, 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[Self::vkb(w), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            groups,
        );
    }

    /// Prefill projection GEMM: `c[m,n] = a[m,k] · Wᵀ` on the matrix cores (coopmat). `a` is f32;
    /// `wq` is the weight (f16 packed 2/u32 with bits=16, or quant idx with bits=4|8 + scales/mins).
    /// `c` MUST be allocated `ceil(m/64)*64` rows (the kernel writes padded rows as 0). `n%64==0`,
    /// `k%32==0`. For f16 weights pass any small non-empty buffer for scales/mins (unused).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_proj(
        &self,
        a: &dyn Buffer,
        wq: &dyn Buffer,
        scales: &dyn Buffer,
        mins: &dyn Buffer,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
        bits: u32,
        blk_shift: u32,
    ) {
        // Warp tile (BM=64,BN=256, 256 threads / 8 warps — matches llama.cpp's AMD-RADV large
        // warptile; the extra warps hide W-dequant latency). Wins big for M≥768 (low/mid ctx:
        // 4k+21% 8k+19% 16k+5%); at very small M (32k chunk≈500) its wide N tile still loses to the
        // BN=64 tiled kernel, so gate on M. Also needs N%256.
        let warp = m >= 768 && n.is_multiple_of(256);
        let (name, spv, tiles_n) = if warp {
            ("gemm_proj_warp", crate::gemm::gemm_proj_warp_spv(), n / 256)
        } else {
            ("gemm_proj", crate::gemm::gemm_proj_spv(), n / 64)
        };
        self.label_gemm(name, m, k, n);
        let kern = self.be.kernel_sg(name, spv, 5, 20, 32);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&bits.to_ne_bytes());
        push[16..20].copy_from_slice(&blk_shift.to_ne_bytes());
        let groups = (m.div_ceil(64) * tiles_n) as u32; // both kernels use BM=64
        self.dispatch(
            kern,
            &[
                Self::vkb(a),
                Self::vkb(wq),
                Self::vkb(scales),
                Self::vkb(mins),
                Self::vkb(c),
            ],
            1,
            &push,
            groups,
        );
    }

    /// Native-block projection GEMM `c = a · Wᵀ` for prefill: raw GGUF blocks dequantized in-shader
    /// during coopmat tiled staging (decode-once per weight element, reused across the 64-row tile).
    /// `c` is allocated `ceil(m/64)*64` rows. Requires `n%64==0`, `k%32==0`.
    pub fn matmul_native(
        &self,
        dtype: infr_core::DType,
        a: &dyn Buffer,
        w: &dyn Buffer,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.matmul_native_off(dtype, a, w, 0, c, m, k, n);
    }

    /// Native-block tiled coopmat GEMM reading the weight from element offset `w_base` — lets one
    /// stacked MoE expert tensor serve all experts (`w_base = expert_id * out_f * in_f`), so each
    /// expert weight is decoded ONCE and reused across the 64-row tile (vs the per-row GEMV re-read).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_native_off(
        &self,
        dtype: infr_core::DType,
        a: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        // Large-warptile variant (8-warp BM=64×BN=256): 4× the math per staged A-row / decoded
        // W-column vs the 64×64 tile, and the extra warps hide the dequant latency. Needs k%32;
        // only the hot formats are compiled — everything else stays on the 64×64 kernel.
        // Tile pick: the wide BN=256 tile when its grid fills the device, else the NARROW BN=128
        // tile (same per-thread math, 2× the workgroups) — n=1024/2048 GEMMs underfilled a 96-wg
        // part at 5-9 TFLOPS with the wide tile. INFR_NO_GEMM_WARP forces the 64×64 tile (A/B).
        // Empirically the wide tile only wins once its own grid saturates (~2× device capacity).
        const WIDE_GRID_MIN: usize = 128;
        let wide_grid = m.div_ceil(64) * (n / 256).max(1);
        let use_wide = n.is_multiple_of(256) && wide_grid >= WIDE_GRID_MIN;
        let warp = if k.is_multiple_of(32) && std::env::var("INFR_NO_GEMM_WARP").is_err() {
            if use_wide {
                crate::gemm::native_gemm_warp_build_spv(dtype).map(|s| (s, 256))
            } else if n.is_multiple_of(128) {
                crate::gemm::native_gemm_warp_n128_build_spv(dtype).map(|s| (s, 128))
            } else {
                None
            }
        } else {
            None
        };
        let (name, spv) = match (warp, dtype) {
            (Some((spv, 256)), infr_core::DType::Bf16) => ("native_gemm_warp_bf16", spv),
            (Some((spv, 256)), infr_core::DType::Q3K) => ("native_gemm_warp_q3k", spv),
            (Some((spv, 256)), infr_core::DType::Q5_0) => ("native_gemm_warp_q5_0", spv),
            (Some((spv, 256)), infr_core::DType::Q5_1) => ("native_gemm_warp_q5_1", spv),
            (Some((spv, 256)), infr_core::DType::Iq4Xs) => ("native_gemm_warp_iq4xs", spv),
            (Some((spv, 256)), infr_core::DType::Q2K) => ("native_gemm_warp_q2k", spv),
            (Some((spv, 256)), infr_core::DType::Q4_0) => ("native_gemm_warp_q4_0", spv),
            (Some((spv, 256)), infr_core::DType::Q4K) => ("native_gemm_warp_q4k", spv),
            (Some((spv, 256)), infr_core::DType::Q5K) => ("native_gemm_warp_q5k", spv),
            (Some((spv, 256)), infr_core::DType::Q6K) => ("native_gemm_warp_q6k", spv),
            (Some((spv, 256)), infr_core::DType::Q8_0) => ("native_gemm_warp_q8_0", spv),
            (Some((spv, _)), infr_core::DType::Bf16) => ("native_gemm_warp_bf16_n128", spv),
            (Some((spv, _)), infr_core::DType::Q3K) => ("native_gemm_warp_q3k_n128", spv),
            (Some((spv, _)), infr_core::DType::Q5_0) => ("native_gemm_warp_q5_0_n128", spv),
            (Some((spv, _)), infr_core::DType::Q5_1) => ("native_gemm_warp_q5_1_n128", spv),
            (Some((spv, _)), infr_core::DType::Iq4Xs) => ("native_gemm_warp_iq4xs_n128", spv),
            (Some((spv, _)), infr_core::DType::Q2K) => ("native_gemm_warp_q2k_n128", spv),
            (Some((spv, _)), infr_core::DType::Q4_0) => ("native_gemm_warp_q4_0_n128", spv),
            (Some((spv, _)), infr_core::DType::Q4K) => ("native_gemm_warp_q4k_n128", spv),
            (Some((spv, _)), infr_core::DType::Q5K) => ("native_gemm_warp_q5k_n128", spv),
            (Some((spv, _)), infr_core::DType::Q6K) => ("native_gemm_warp_q6k_n128", spv),
            (Some((spv, _)), infr_core::DType::Q8_0) => ("native_gemm_warp_q8_0_n128", spv),
            _ => (
                crate::linear::native_gemm_kernel_name(dtype),
                crate::gemm::native_gemm_build_spv(dtype).expect("native GEMM spv"),
            ),
        };
        self.label_gemm(name, m, k, n);
        let kern = self.be.kernel_sg(name, spv, 3, 16, 32);
        let groups_n = match warp {
            Some((_, bn)) => n / bn,
            None => n / 64,
        };
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let groups = (m.div_ceil(64) * groups_n) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(a), Self::vkb(w), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// [`matmul_native_off`](Self::matmul_native_off) with A already converted to f16 (padded to
    /// ceil(m/64)·64 rows) — the A_GLOBAL warptiles coopMatLoad A straight from global, dropping
    /// the As stage/LDS (occupancy 2→3 wgs/WGP: ~1.5x on the 8B shapes). Same tile pick as the
    /// f32 path; caller guarantees k%32==0, n%128==0 and that the _ag SPIR-V exists for `dtype`
    /// (`native_gemm_warp_ag_build_spv(dtype).is_some()`). Numerics are bit-identical to the f32
    /// path: the staging loop rounded A to f16 anyway, and the MMA order is unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_native_f16a(
        &self,
        dtype: infr_core::DType,
        a16: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        // The BN=128 (n128) ag tile beats the BN=256 (wide) ag tile on EVERY shape this decision
        // can reach, measured on RDNA3 (7900 XTX): the wide tile's WN=64 → 2×4 = 8 accumulator
        // coopmat frags per warp cost enough VGPRs to drop occupancy, and the wide-square n=4096
        // shapes (o-proj, down) land at exactly ceil(m/64)·(n/256) = 128 workgroups — one
        // under-filled wave on the 48-WGP part. Halving the N tile doubles the workgroups AND
        // halves the accumulator footprint.
        //
        // The old gate `use_wide = wide_grid >= 128` had the crossover BACKWARDS: the
        // `wide_n128_crossover_sweep` bench shows n128 wins across the whole wide_grid >= 128
        // regime (the ONLY regime the old gate picked wide) — 8B o 40→70 TF, gate+up 53→77 TF,
        // down 42→73 TF, and the shallow-k small-model shapes too (1152×13824 43→68 TF,
        // 1024×6144 53→69 TF). The wide tile only edges n128 on wide_grid < 128 (small-N) shapes,
        // which the old code ALREADY routed to n128 and which production sends to split-K anyway —
        // so no shape reachable here prefers wide. In-model pp512: 8B +27%, qwen35-4B +10%,
        // gemma-4-E2B +10%; gemma-3-1b/qwen3-0.6b flat (their hot GEMMs are Q5_0 off this path /
        // split-K). Bit-identical to the wide tile (both BK=64, same k-accumulation order).
        // INFR_GEMM_WIDE_TILE restores the old BN=256 tile for A/B.
        let wide_grid = m.div_ceil(64) * (n / 256).max(1);
        let use_wide = n.is_multiple_of(256)
            && wide_grid >= 128
            && std::env::var("INFR_GEMM_WIDE_TILE").is_ok();
        // Small-m batched prefill (MTP verify's draft window, m≈6-24): the default BM=64 row tile
        // is mostly masked waste (see DENSE_SMALL_TILE_MAX_M's doc) — switch to the BM=32 `_bm32`
        // n128_ag variant when it exists for `dtype` (only the K-quant formats verify actually
        // hits: Q4_K/Q5_K/Q6_K/Q8_0) and the wide tile wasn't explicitly requested (wide_grid>=128
        // never fires at these m anyway, short of the vocab-head GEMM under INFR_GEMM_WIDE_TILE).
        let small_bm =
            !use_wide && m <= DENSE_SMALL_TILE_MAX_M && std::env::var("INFR_NO_SMALL_BM").is_err();
        let bm32 = small_bm
            .then(|| crate::gemm::native_gemm_warp_n128_ag_bm32_build_spv(dtype))
            .flatten();
        let (name, spv, bn, bm): (_, _, usize, usize) = if use_wide {
            let (name, spv) = crate::gemm::native_gemm_warp_ag_build_spv(dtype).expect("ag spv");
            (name, spv, 256, 64)
        } else if let Some((name, spv)) = bm32 {
            (name, spv, 128, 32)
        } else {
            let (name, spv) =
                crate::gemm::native_gemm_warp_n128_ag_build_spv(dtype).expect("ag n128 spv");
            (name, spv, 128, 64)
        };
        self.label_gemm(name, m, k, n);
        let kern = self.be.kernel_sg(name, spv, 3, 16, 32);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let groups = (m.div_ceil(bm) * (n / bn)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(a16), Self::vkb(w), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// SPLIT-K narrow-warptile GEMM: `splits` k-partials into `partials` ([splits, mpad, n] f32),
    /// then the deterministic fixed-order reduce into `c`. The occupancy fix for narrow-n GEMMs
    /// with deep k (o/down projections: n = n_embd, k = 2-3·n_embd — 64 workgroups on a 96-wg
    /// part with the plain tile). Requires n%128==0, k%32==0; caller sizes `partials` to
    /// `splits · ceil(m/64)·64 · n · 4` bytes. `a_is_f16`: A is the caller's f16 cast-copy
    /// (padded rows) and the A_GLOBAL tile is used — see [`matmul_native_f16a`](Self::matmul_native_f16a).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_native_splitk(
        &self,
        dtype: infr_core::DType,
        a: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        partials: &dyn Buffer,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
        splits: usize,
        a_is_f16: bool,
    ) {
        // NB: `dense_small_m_row_tile_bench` also probed a BM=32 tile here (mirroring
        // `matmul_native_f16a`'s small-m gate) and found a NET LOSS across the whole m≈4-64
        // sweep on every sk_ag shape tested (down/attn_out/kv/q_proj/o_small) — split-K's own
        // `splits` dimension already fans this GEMM out across enough extra workgroups to fill
        // the device, so a smaller row tile only adds fixed per-workgroup cost (barrier/staging)
        // without an occupancy win to pay for it. BM=64 stays unconditional here; the win lives
        // in `matmul_native_f16a` (the non-split-K n128_ag family), which had no such fill floor.
        let (name, spv) = if a_is_f16 {
            crate::gemm::native_gemm_warp_sk_ag_build_spv(dtype).expect("split-k ag spv")
        } else {
            let spv = crate::gemm::native_gemm_warp_sk_build_spv(dtype).expect("split-k spv");
            let name = match dtype {
                infr_core::DType::F16 => "native_gemm_warp_f16_sk",
                infr_core::DType::Iq4Xs => "native_gemm_warp_iq4xs_sk",
                infr_core::DType::Q2K => "native_gemm_warp_q2k_sk",
                infr_core::DType::Q4_0 => "native_gemm_warp_q4_0_sk",
                infr_core::DType::Q3K => "native_gemm_warp_q3k_sk",
                infr_core::DType::Q4K => "native_gemm_warp_q4k_sk",
                infr_core::DType::Q5K => "native_gemm_warp_q5k_sk",
                infr_core::DType::Q6K => "native_gemm_warp_q6k_sk",
                _ => "native_gemm_warp_q8_0_sk",
            };
            (name, spv)
        };
        self.label_gemm(name, m, k, n);
        let mpad = m.div_ceil(64) * 64;
        let kern = self.be.kernel_sg(name, spv, 3, 24, 32);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(splits as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(mpad as u32).to_ne_bytes());
        let groups = ((mpad / 64) * (n / 128) * splits) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(a), Self::vkb(w), Self::vkb(partials)],
            1,
            &push,
            groups,
        );
        // reduce: out[i] = Σ_s partials[s·plane + i]
        self.label_gemm("splitk_reduce", m, k, n);
        let rk = self
            .be
            .kernel("splitk_reduce", crate::gemm::splitk_reduce_spv(), 2, 12);
        let n_elems = mpad * n;
        let mut rp = [0u8; 12];
        rp[0..4].copy_from_slice(&(n_elems as u32).to_ne_bytes());
        rp[4..8].copy_from_slice(&(splits as u32).to_ne_bytes());
        rp[8..12].copy_from_slice(&(n_elems as u32).to_ne_bytes());
        self.dispatch(
            rk,
            &[Self::vkb(partials), Self::vkb(c)],
            1,
            &rp,
            (n_elems as u32).div_ceil(64),
        );
    }

    /// Tiled Q4_K dp4a (mmq) GEMM for a stacked expert: `c = qa·W[w_base]ᵀ` using hardware int8
    /// dot-product (activations pre-quantized via `quant_q8`). `c` is `ceil(m/64)*64` rows. Faster
    /// than the coopmat-f16 `matmul_native` for u4 weights (the dense path's default for u4).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mmq_q4k(
        &self,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.label_gemm("native_gemm_mmq_q4k", m, k, n);
        let kern = self.be.kernel(
            "native_gemm_mmq_q4k",
            crate::gemm::native_gemm_mmq_q4k_spv(),
            5,
            16,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let groups = (m.div_ceil(64) * (n / 64)) as u32;
        self.dispatch(
            kern,
            &[
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(w),
                Self::vkb(c),
            ],
            1,
            &push,
            groups,
        );
    }

    /// int8 cooperative-matrix (WMMA) prefill GEMM for Q8_0 weights — MEASUREMENT kernel, gated by
    /// the adapter behind `INFR_I8_COOPMAT=1` + `caps.i8_coopmat` (see
    /// `native_gemm_i8cm_q8_0.comp` for the design doc: 16x16 tile per workgroup, per-Q8_0-block
    /// int32 WMMA dot + shared-mem store/scale epilogue). `qa`/`dact` from `quant_q8`. `c` is
    /// `ceil(m/64)*64` rows (same padding convention as `matmul_native_off`). Requires `n%16==0`,
    /// `k%32==0`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_i8cm_q8_0(
        &self,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.label_gemm("native_gemm_i8cm_q8_0", m, k, n);
        let kern = self.be.kernel_sg(
            "native_gemm_i8cm_q8_0",
            crate::gemm::native_gemm_i8cm_q8_0_spv(),
            4,
            16,
            32,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        // BM=128 (see the shader header): one workgroup covers 128 rows x 16 cols.
        let groups = (m.div_ceil(128) * (n / 16)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(qa), Self::vkb(dact), Self::vkb(w), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// "Idea 2" measurement variant of `matmul_i8cm_q8_0`: `dact_row` is ONE f16 scale per row
    /// (whole-K, from `quant_q8_row`) instead of per-32-block — coarser precision, but lets the
    /// shader pull the activation scale out of its per-block sum. Gated by `INFR_I8_ROW_SCALE=1`
    /// (see `native_gemm_i8cm_q8_0.comp` #ifdef ROW_SCALE).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_i8cm_q8_0_rowscale(
        &self,
        qa: &dyn Buffer,
        dact_row: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.label_gemm("native_gemm_i8cm_q8_0_rowscale", m, k, n);
        let kern = self.be.kernel_sg(
            "native_gemm_i8cm_q8_0_rowscale",
            crate::gemm::native_gemm_i8cm_q8_0_rowscale_spv(),
            4,
            16,
            32,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let groups = (m.div_ceil(128) * (n / 16)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(qa), Self::vkb(dact_row), Self::vkb(w), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// Row-wise (whole-K) int8 activation quantize pass — int8-coopmat GEMM "Idea 2" measurement
    /// variant (see `quant_q8_row.comp` / `native_gemm_i8cm_q8_0.comp`): ONE f16 scale per row
    /// (`dact_row[m]`) instead of `quant_q8`'s per-32-block scale, coarser but block-invariant so
    /// the i8cm kernel can pull it out of its per-block sum. Gated by `INFR_I8_ROW_SCALE=1`; used
    /// only by the int8-coopmat GEMM path, `quant_q8` (every other int8 kernel) is untouched.
    pub fn quant_q8_row(&self, a: &dyn Buffer, qa: &dyn Buffer, dact_row: &dyn Buffer, m: usize, k: usize) {
        let kq = self
            .be
            .kernel_sg("quant_q8_row", crate::gemm::quant_q8_row_spv(), 3, 8, 32);
        let mut p = [0u8; 8];
        p[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        p[4..8].copy_from_slice(&(k as u32).to_ne_bytes());
        self.dispatch(
            kq,
            &[Self::vkb(a), Self::vkb(qa), Self::vkb(dact_row)],
            2,
            &p,
            m as u32,
        );
    }

    /// Tiled Q6_K dp4a (mmq) GEMM for a stacked expert (the MoE down projection): `c = qa·W[w_base]ᵀ`
    /// using hardware int8 dot-product (activations pre-quantized via `quant_q8`; the per-block sum is
    /// unused — Q6_K is symmetric, no min). `c` is `ceil(m/64)*64` rows. Faster than the coopmat-f16
    /// `matmul_native` for u6 weights. Requires `n%64`, `k%16`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mmq_q6k(
        &self,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.label_gemm("native_gemm_mmq_q6k", m, k, n);
        let kern = self.be.kernel(
            "native_gemm_mmq_q6k",
            crate::gemm::native_gemm_mmq_q6k_spv(),
            4,
            16,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let groups = (m.div_ceil(64) * (n / 64)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(qa), Self::vkb(dact), Self::vkb(w), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// Integer (dp4a) u4 projection GEMM — the mmq path. Quantizes activations to int8 (Q8 per
    /// 32-block) via `quant_q8`, then runs the dp4a matmul keeping weights quantized (no per-GEMM
    /// dequant). Scratch (caller-allocated): `qa` = m*k bytes (int8), `dact`/`sact` = m*(k/32)*2
    /// bytes (f16). `c` = ceil(m/64)*64 rows f32. Requires u4 weights, `n%64==0`, `k%32==0`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_proj_mmq(
        &self,
        a: &dyn Buffer,
        wq: &dyn Buffer,
        scales: &dyn Buffer,
        mins: &dyn Buffer,
        c: &dyn Buffer,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let nblk = k / 32;
        // pass 1: quantize activations to int8 (Q8, per 32-block) — one subgroup per (row, block)
        let kq = self
            .be
            .kernel_sg("quant_q8", crate::gemm::quant_q8_spv(), 4, 12, 32);
        let mut p1 = [0u8; 12];
        p1[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        p1[4..8].copy_from_slice(&(k as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&32u32.to_ne_bytes());
        self.dispatch(
            kq,
            &[
                Self::vkb(a),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
            ],
            3,
            &p1,
            (m * nblk) as u32,
        );
        // pass 2: integer dp4a matmul
        self.label_gemm("gemm_proj_mmq", m, k, n);
        let km = self
            .be
            .kernel_sg("gemm_proj_mmq", crate::gemm::gemm_proj_mmq_spv(), 7, 12, 32);
        let mut p2 = [0u8; 12];
        p2[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        self.dispatch3(
            km,
            &[
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(wq),
                Self::vkb(scales),
                Self::vkb(mins),
                Self::vkb(c),
            ],
            1,
            &p2,
            (n / 64) as u32,
            m.div_ceil(64) as u32,
            1,
        );
    }

    /// Quantized dequant GEMV `y = x·Wᵀ`. `bits` (4|8) and `blk_shift` (log2 scale-block) select
    /// the packed-weight layout.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_q(
        &self,
        quants: &dyn Buffer,
        scales: &dyn Buffer,
        mins: &dyn Buffer,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
        bits: u32,
        blk_shift: u32,
    ) {
        self.label_gemv("lin_q", rows, in_f, out_f);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&bits.to_ne_bytes());
        push[16..20].copy_from_slice(&blk_shift.to_ne_bytes());
        let bufs = [
            Self::vkb(quants),
            Self::vkb(scales),
            Self::vkb(mins),
            Self::vkb(x),
            Self::vkb(y),
        ];
        if let Some(epw) = mmv_epw(bits) {
            if in_f.is_multiple_of(epw) {
                // subgroup GEMV: NUM_ROWS outputs/workgroup, one wave32 each (no shared reduction)
                let name = if bits == 4 {
                    "mul_mat_vec_q4"
                } else {
                    "mul_mat_vec_q8"
                };
                let k =
                    self.be
                        .kernel_sg(name, crate::gemm::mul_mat_vec_q_spv(bits, false), 5, 20, 32);
                let groups = (out_f as u32).div_ceil(MMV_NUM_ROWS);
                self.dispatch_wide(k, &bufs, 1, &push, rows as u32 * groups);
                return;
            }
        }
        let k = self
            .be
            .kernel("linear_q", crate::gemm::linear_q_spv(), 5, 20);
        self.dispatch_wide(k, &bufs, 1, &push, (rows * out_f) as u32);
    }

    /// Native-block dequant GEMV `y = x·Wᵀ`. Raw GGUF block bytes in `w` (padded
    /// to u32); format identified by `dtype`. Dispatch `rows*out_f` workgroups.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.linear_native_off(dtype, w, 0, x, y, rows, in_f, out_f);
    }

    /// Native-block dequant GEMV reading the weight from element offset `w_base` — lets one stacked
    /// MoE expert tensor serve all experts (`w_base = expert_id * out_f * in_f`).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_off(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        w_base: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        // Reassociation-tolerant subgroup route (m=1 decode, latency-starved projection band) —
        // takes precedence over the bit-identical RM path where it saturates DRAM better.
        if rows == 1 {
            if let Some(nr) = native_sg_choice(dtype, in_f, out_f) {
                if let Some((name, spv)) = crate::gemm::native_sg_build_spv(dtype, false, nr) {
                    self.label_gemv("gemv", rows, in_f, out_f);
                    let k = self.be.kernel_sg(name, spv, 3, 16, 32);
                    let mut push = [0u8; 16];
                    push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
                    push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
                    push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
                    push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
                    self.dispatch_wide(
                        k,
                        &[Self::vkb(w), Self::vkb(x), Self::vkb(y)],
                        1,
                        &push,
                        (out_f as u32).div_ceil(nr),
                    );
                    return;
                }
            }
        }
        // Multi-output-row route (m=1 decode, low out_f → memory-latency-starved single-row grid).
        if rows == 1 {
            if let Some(rm) = native_rm_choice(dtype, out_f) {
                if let Some((name, spv)) = crate::gemm::native_rm_build_spv(dtype, false, rm) {
                    self.label_gemv("gemv", rows, in_f, out_f);
                    let k = self.be.kernel(name, spv, 3, 16);
                    let mut push = [0u8; 16];
                    push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
                    push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
                    push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
                    push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
                    self.dispatch_wide(
                        k,
                        &[Self::vkb(w), Self::vkb(x), Self::vkb(y)],
                        1,
                        &push,
                        (out_f as u32).div_ceil(rm),
                    );
                    return;
                }
            }
        }
        self.label_gemv("gemv", rows, in_f, out_f);
        let name = crate::linear::native_kernel_name(dtype, false);
        let spv = crate::gemm::native_build_spv(dtype, false).expect("native GEMV spv");
        let k = self.be.kernel(name, spv, 3, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[Self::vkb(w), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// Multi-row native GEMV (`2 <= rows <= 8`): the GEMV's out_f-wide cooperative-over-K grid,
    /// each workgroup decoding a weight sub-block ONCE and dotting it against every row — the
    /// spec-decode verify / short-suffix-prefill shape, where the single-M-tile coopmat GEMM
    /// underfills the GPU (measured 51-182 GB/s effective vs the GEMV's 292-651 on a 7900 XTX)
    /// and the plain GEMV re-streams the weight per row. Same push layout as the GEMV; `w_off`
    /// (fused-QKV slices) rides `w_base`. Caller gates on [`crate::gemm::native_mrow_build_spv`].
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_mrow(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        w_base: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        debug_assert!((2..=8).contains(&rows));
        self.label_gemv("mrow", rows, in_f, out_f);
        let name = crate::gemm::native_mrow_kernel_name(dtype);
        let spv = crate::gemm::native_mrow_build_spv(dtype).expect("native mrow spv");
        let k = self.be.kernel(name, spv, 3, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[Self::vkb(w), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            out_f as u32, // one workgroup per OUTPUT — all rows share its weight stream
        );
    }

    /// Multi-row int8 dp4a GEMV (`2 <= rows <= 8`): [`Self::linear_native_mrow`]'s shape with
    /// `native_mmv.comp`'s integer-dot math — the weight sub-block is unpacked once into packed
    /// int8 words and dp4a'd against every row's pre-quantized activation block (`qa`/`dact`/
    /// `sact` from [`Self::quant_q8`] over `rows` rows). NUM_ROWS=2 outputs per workgroup
    /// (grid = ceil(out_f/2)). Caller gates on [`crate::gemm::native_mmv_mrow_build_spv`].
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv_mrow(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        w_base: usize,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        debug_assert!((2..=8).contains(&rows));
        self.label_gemv("mmvr", rows, in_f, out_f);
        // Small-in_f shape (in_f < 2048 → fewer than 64 32-elem sub-blocks): the 2-output wg
        // leaves 64−nsub lanes idle (E2B's in_f=1536: 25% dead, single loop iteration) — the
        // OUTS4 variant splits the wg 4 outputs × 16 K-lanes so every lane stays on a sub-block.
        // INFR_NO_MMV_O4 forces the 2-output layout (A/B).
        let o4 = in_f < 2048 && std::env::var("INFR_NO_MMV_O4").is_err();
        // rows<=4 MR specialization: the MR=8 per-sub-block row loop runs half dead at the pp4
        // serve shape — -DMRV=4 halves it (and the acc registers). INFR_NO_MMV_M4 forces MR=8
        // (A/B). o4+m4 together (with the uvec4 activation loads): E2B pp4@d4096 1536x24576
        // 37.8 → 34.7us/op, pp4 467.6 → 474.9 t/s vs both escapes set — modest; the shape is
        // ~600 GB/s and mostly weight-stream bound, o4 alone measured neutral.
        let m4 = rows <= 4 && std::env::var("INFR_NO_MMV_M4").is_err();
        let name = crate::gemm::native_mmv_mrow_variant_name(dtype, o4, m4);
        let spv =
            crate::gemm::native_mmv_mrow_variant_spv(dtype, o4, m4).expect("native mmv mrow spv");
        let groups = (out_f as u32).div_ceil(if o4 { 4 } else { 2 });
        let k = self.be.kernel(name, spv, 5, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[
                Self::vkb(w),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(y),
            ],
            1,
            &push,
            groups,
        );
    }

    /// Native-block dequant GEMV with fused residual add: `y = residual + x·Wᵀ`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add_native(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        x: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        // push[12..16] = w_base, 0 (residual native GEMV is not used for stacked experts).
        let bufs = [
            Self::vkb(w),
            Self::vkb(x),
            Self::vkb(residual),
            Self::vkb(y),
        ];
        // Reassociation-tolerant subgroup route (see linear_native_off) — precedence over RM.
        if rows == 1 {
            if let Some(nr) = native_sg_choice(dtype, in_f, out_f) {
                if let Some((name, spv)) = crate::gemm::native_sg_build_spv(dtype, true, nr) {
                    let k = self.be.kernel_sg(name, spv, 4, 16, 32);
                    self.dispatch_wide(k, &bufs, 1, &push, (out_f as u32).div_ceil(nr));
                    return;
                }
            }
        }
        // Multi-output-row route (see linear_native_off): o/down decode GEMVs are low-out_f.
        if rows == 1 {
            if let Some(rm) = native_rm_choice(dtype, out_f) {
                if let Some((name, spv)) = crate::gemm::native_rm_build_spv(dtype, true, rm) {
                    let k = self.be.kernel(name, spv, 4, 16);
                    self.dispatch_wide(k, &bufs, 1, &push, (out_f as u32).div_ceil(rm));
                    return;
                }
            }
        }
        let name = crate::linear::native_kernel_name(dtype, true);
        let spv = crate::gemm::native_build_spv(dtype, true).expect("native GEMV spv");
        let k = self.be.kernel(name, spv, 4, 16);
        self.dispatch_wide(k, &bufs, 1, &push, (rows * out_f) as u32);
    }

    /// Gather + dequantize embedding rows (`Op::EmbedGather`): `dst[r,:] = table[ids[r],:] *
    /// scale`. One workgroup per row; per-format decode from native_decode.glsl. Caller gates on
    /// [`crate::gemm::embed_gather_build_spv`].
    #[allow(clippy::too_many_arguments)]
    pub fn embed_gather(
        &self,
        dtype: infr_core::DType,
        table: &dyn Buffer,
        ids: &dyn Buffer,
        dst: &dyn Buffer,
        rows: usize,
        ne: usize,
        scale: f32,
    ) {
        let name = crate::gemm::embed_gather_kernel_name(dtype);
        let spv = crate::gemm::embed_gather_build_spv(dtype).expect("embed_gather spv");
        let k = self.be.kernel(name, spv, 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&scale.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(table), Self::vkb(ids), Self::vkb(dst)],
            1,
            &push,
            rows as u32,
        );
    }

    /// Int8 dp4a decode GEMV (m=1): `y = x·Wᵀ` with `x` pre-quantized via [`Self::quant_q8`]
    /// (qa/dact/sact). NUM_ROWS=2 — one workgroup per 2 consecutive outputs (`ceil(out_f/2)`
    /// grid), the activation block read once for both. `w_base` = element offset (fused-QKV
    /// slices). Caller gates on [`crate::gemm::native_mmv_build_spv`].
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        w_base: usize,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        self.label_gemv("mmv", 1, in_f, out_f);
        let name = crate::gemm::native_mmv_kernel_name(dtype, false);
        let spv = crate::gemm::native_mmv_build_spv(dtype, false).expect("native mmv spv");
        let k = self.be.kernel(name, spv, 5, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&1u32.to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[
                Self::vkb(w),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(y),
            ],
            1,
            &push,
            (out_f as u32).div_ceil(2),
        );
    }

    /// Int8 dp4a decode GEMV with fused residual add: `y = residual + x·Wᵀ` (see
    /// [`Self::linear_mmv`]).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add_mmv(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        let name = crate::gemm::native_mmv_kernel_name(dtype, true);
        let spv = crate::gemm::native_mmv_build_spv(dtype, true).expect("native mmv res spv");
        let k = self.be.kernel(name, spv, 6, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&1u32.to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        // push[12..16] = w_base, 0 (the residual GEMV never reads stacked experts).
        self.dispatch_wide(
            k,
            &[
                Self::vkb(w),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(residual),
                Self::vkb(y),
            ],
            1,
            &push,
            (out_f as u32).div_ceil(2),
        );
    }

    /// Quantized dequant GEMV with fused residual add: `y = residual + x·Wᵀ`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add_q(
        &self,
        quants: &dyn Buffer,
        scales: &dyn Buffer,
        mins: &dyn Buffer,
        x: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
        bits: u32,
        blk_shift: u32,
    ) {
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&bits.to_ne_bytes());
        push[16..20].copy_from_slice(&blk_shift.to_ne_bytes());
        let bufs = [
            Self::vkb(quants),
            Self::vkb(scales),
            Self::vkb(mins),
            Self::vkb(x),
            Self::vkb(residual),
            Self::vkb(y),
        ];
        if let Some(epw) = mmv_epw(bits) {
            if in_f.is_multiple_of(epw) {
                let name = if bits == 4 {
                    "mul_mat_vec_q4_res"
                } else {
                    "mul_mat_vec_q8_res"
                };
                let k =
                    self.be
                        .kernel_sg(name, crate::gemm::mul_mat_vec_q_spv(bits, true), 6, 20, 32);
                let groups = (out_f as u32).div_ceil(MMV_NUM_ROWS);
                self.dispatch_wide(k, &bufs, 1, &push, rows as u32 * groups);
                return;
            }
        }
        let k = self
            .be
            .kernel("linear_res_q", crate::gemm::linear_res_q_spv(), 6, 20);
        self.dispatch_wide(k, &bufs, 1, &push, (rows * out_f) as u32);
    }

    /// `y = residual + x·Wᵀ` (fused residual add). `residual` and `y` may be the same buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add(
        &self,
        w: &dyn Buffer,
        x: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let k = self
            .be
            .kernel("linear_res", crate::gemm::linear_res_spv(), 4, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[
                Self::vkb(w),
                Self::vkb(x),
                Self::vkb(residual),
                Self::vkb(y),
            ],
            1,
            &push,
            (rows * out_f) as u32, // one workgroup per output element (coalesced GEMV)
        );
    }

    pub fn rmsnorm(
        &self,
        x: &dyn Buffer,
        w: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        // 256-thread subgroup kernel (requiredSubgroupSize=32): more load/store parallelism and a
        // single barrier vs the 64-thread WGSL shared-tree. ~2.6× faster as a kernel; end-to-end
        // neutral here (decode is dispatch-latency-bound) but a win on slower/higher-latency GPUs.
        let k = self
            .be
            .kernel_sg("rmsnorm", crate::gemm::rmsnorm_spv(), 3, 12, 32);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&eps.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(w), Self::vkb(y)],
            1,
            &push,
            rows as u32, // one workgroup per row (cooperative reduction)
        );
    }

    /// Fused per-head RMSNorm + SiLU gate multiply: `y[i] = rmsnorm(x)[i] * silu(z[i])`, one
    /// workgroup per (row, head) — `rmsnorm.comp`'s -DGATE build (`Op::GatedRmsNorm`, qwen35
    /// DeltaNet z-gate). Same reduction as `rmsnorm`; `z` must be the SAME shape/layout as `x`
    /// (`rows*dim` flat, indexed identically — see `Op::GatedRmsNorm`'s doc). Removes the
    /// read-after-write barrier a separate `rmsnorm`→`silu_mul` pair would need (GatedAct reading
    /// QkNorm's freshly-written output).
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_gate(
        &self,
        x: &dyn Buffer,
        w: &dyn Buffer,
        z: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        let k = self
            .be
            .kernel_sg("rmsnorm_gate", crate::gemm::rmsnorm_gate_spv(), 4, 12, 32);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&eps.to_ne_bytes());
        // Binding order MUST be x, w, z, y (matches the shader's -DGATE layout) — `y` last so the
        // hazard tracker's `n_out=1` marks only the true output as written (see the shader's doc).
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(w), Self::vkb(z), Self::vkb(y)],
            1,
            &push,
            rows as u32, // one workgroup per row (cooperative reduction)
        );
    }

    /// Row-wise softmax: `y[r,:] = softmax(x[r,:] * scale)` over `dim` columns, one workgroup per
    /// row (diffusion-gemma's in-graph self-conditioning — see docs/DIFFUSIONGEMMA.md's Phase-B
    /// and the reference's `dg_canvas_embed`). Same 256-thread subgroup-reduction shape as
    /// `rmsnorm` — `dim` here is the vocab, so the cooperative reduction matters just as much.
    pub fn softmax(&self, x: &dyn Buffer, y: &dyn Buffer, rows: usize, dim: usize, scale: f32) {
        let k = self
            .be
            .kernel_sg("softmax", crate::gemm::softmax_spv(), 2, 12, 32);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&scale.to_ne_bytes());
        self.dispatch(k, &[Self::vkb(x), Self::vkb(y)], 1, &push, rows as u32);
    }

    /// Like [`Self::softmax`], but the scale comes from `scale_buf[0]` (a 1-element device buffer,
    /// host-updated via a tiny 4-byte `upload` between calls) instead of a push constant — the
    /// DiffusionGemma denoise self-conditioning path uses this to vary the softmax temperature
    /// every step on a plan compiled/cached ONCE (see `Op::Softmax::scale_buf`'s doc and
    /// docs/DIFFUSIONGEMMA.md's Phase-B). `USE_SCALE_BUF`-compiled variant of the same shader.
    pub fn softmax_dyn(
        &self,
        x: &dyn Buffer,
        scale_buf: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        dim: usize,
    ) {
        let k = self
            .be
            .kernel_sg("softmax_dyn", crate::gemm::softmax_dyn_spv(), 3, 8, 32);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(scale_buf), Self::vkb(y)],
            1,
            &push,
            rows as u32,
        );
    }

    /// RoPE in place is allowed (`x` and `y` may be the same buffer). `pos_offset` shifts the
    /// absolute position of the first row (for cached decode).
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &dyn Buffer,
        y: &dyn Buffer,
        t: usize,
        n_heads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        pos_offset: usize,
    ) {
        let k = self.be.kernel("rope", crate::gemm::rope_spv(), 2, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        // [24..28] out_base: 0 (in-place f32 rope has no output shift)
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (t * n_heads) as u32,
        );
    }

    /// Interleaved (llama NORM) RoPE writing f16 — the llama q/k analogue of `qk_norm_rope`'s
    /// f16 output: `out_base` shifts the output row (0 for the Q scratch; `pos` for the fused
    /// K cache write via the kv_write peephole).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_f16(
        &self,
        x: &dyn Buffer,
        y: &dyn Buffer,
        t: usize,
        n_heads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        pos_offset: usize,
        out_base: usize,
    ) {
        let k = self
            .be
            .kernel("rope_f16", crate::gemm::rope_f16_spv(), 2, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(out_base as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (t * n_heads) as u32,
        );
    }

    /// Record-once variant of [`Self::rope_f16`]: pos from `params`; `out_base_mul` is the 0/1
    /// multiplier the shader scales by pos (1 -> write cache row pos, 0 -> row 0 of the Q scratch).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_f16_dyn(
        &self,
        x: &dyn Buffer,
        params: &dyn Buffer,
        y: &dyn Buffer,
        t: usize,
        n_heads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        out_base_mul: usize,
    ) {
        let k = self
            .be
            .kernel("rope_f16_dyn", crate::gemm::rope_f16_dyn_spv(), 3, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        // [20..24] pos_offset: unused (from params)
        push[24..28].copy_from_slice(&(out_base_mul as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(params), Self::vkb(y)],
            1,
            &push,
            (t * n_heads) as u32,
        );
    }

    /// Cached attention: q `[q_len,nh,hd]` (abs positions `pos_offset..`) over the K/V cache
    /// `[kv_len,nkv,hd]`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        // Sliding-window attention: a query at abs pos `p` attends only keys `j > p - window`.
        // `0` = full causal (llama/qwen3 + gemma full-attention layers).
        window: usize,
        // QK scale: `> 0` overrides the default 1/√hd (gemma4 passes 1.0). `0.0` = default.
        scale: f32,
        // Per-side planar Q8_0 KV read (K and V independent). `cap` = total cache elements (planar
        // scales base), unused when both are f16.
        k_q8: bool,
        v_q8: bool,
        cap: usize,
    ) {
        let (name, spv) = match (k_q8, v_q8) {
            (false, false) => ("attention_kv", crate::gemm::attention_kv_spv()),
            (true, false) => ("attention_kv_kq8", crate::gemm::attention_kv_kq8_spv()),
            (false, true) => ("attention_kv_vq8", crate::gemm::attention_kv_vq8_spv()),
            (true, true) => ("attention_kv_q8", crate::gemm::attention_kv_q8_spv()),
        };
        let kern = self.be.kernel(name, spv, 4, 36);
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(q_len as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&scale.to_ne_bytes());
        push[32..36].copy_from_slice(&(cap as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[Self::vkb(q), Self::vkb(kc), Self::vkb(vc), Self::vkb(o)],
            1,
            &push,
            (q_len * nh) as u32,
        );
    }

    /// Non-FA prefill attention: clean coopmat QK → row softmax → coopmat PV (ollama's approach).
    /// `q`=[mpad,nh,hd] f16, `kc`/`vc`=[kv_len,nkv,hd] f16, `attn`=[mpad,nh*hd] f32 out, `s`=
    /// [nh,mpad,kv_pad] f16 scratch (mpad=ceil(n/64)*64, kv_pad=ceil(kv_len/64)*64). `pos_offset` is
    /// the absolute position of query row 0 (for causal masking).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_nonfa(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        attn: &dyn Buffer,
        s: &dyn Buffer,
        pv_part: &dyn Buffer,
        n: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        window: usize,
        scale: f32,
    ) {
        let mpad = (n.div_ceil(64) * 64) as u32;
        // kv padded to 256 (the 8-warp attn_qk's BN); extra cols are masked in softmax. Still %64 so
        // the 4-warp fallback + softmax + attn_pv are unaffected.
        let kv_pad = (kv_len.div_ceil(256) * 256) as u32;
        let hdu = hd as u32;
        // scale override: >0 uses the caller's value (gemma4 = 1.0, QK-norm controls magnitude);
        // 0 keeps the default 1/√hd (qwen3, gemma3).
        let scale = if scale > 0.0 {
            scale
        } else {
            1.0f32 / (hd as f32).sqrt()
        };

        // stage 1: S = scale·Q·Kᵀ. 8-warp/256-thread warptile (BN=256, matches ollama's mul_mm)
        // unless INFR_NO_QK_WARP forces the 4-warp/2×2 attn_qk.
        let qk_warp = std::env::var("INFR_NO_QK_WARP").is_err();
        let (qk_name, qk_spv, qk_bn) = if qk_warp {
            ("attn_qk_warp", crate::gemm::attn_qk_warp_spv(), 256u32)
        } else {
            ("attn_qk", crate::gemm::attn_qk_spv(), 64u32)
        };
        let kqk = self.be.kernel_sg(qk_name, qk_spv, 3, 24, 32);
        let mut p = [0u8; 24];
        p[0..4].copy_from_slice(&mpad.to_ne_bytes());
        p[4..8].copy_from_slice(&kv_pad.to_ne_bytes());
        p[8..12].copy_from_slice(&hdu.to_ne_bytes());
        p[12..16].copy_from_slice(&(nh as u32).to_ne_bytes());
        p[16..20].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p[20..24].copy_from_slice(&scale.to_ne_bytes());
        let qk_tiles = (mpad / 64) * (kv_pad / qk_bn);
        self.dispatch3(
            kqk,
            &[Self::vkb(q), Self::vkb(kc), Self::vkb(s)],
            1,
            &p,
            qk_tiles,
            1,
            nh as u32,
        );

        // stage 2: row softmax (causal + optional sliding window), in place S → P
        let ksm = self
            .be
            .kernel("attn_softmax", crate::gemm::attn_softmax_spv(), 1, 20);
        let mut ps = [0u8; 20];
        ps[0..4].copy_from_slice(&mpad.to_ne_bytes());
        ps[4..8].copy_from_slice(&kv_pad.to_ne_bytes());
        ps[8..12].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        ps[12..16].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        ps[16..20].copy_from_slice(&(window as u32).to_ne_bytes());
        self.dispatch3(ksm, &[Self::vkb(s)], 1, &ps, mpad, 1, nh as u32);

        // stage 3: O = P·V  (one coopmat GEMM per head). Split-K when under-occupied: at high ctx
        // mpad is small so the base workgroup count ((mpad/64)*(hd/64)*nh) is far below the GPU's
        // capacity while each grinds a huge kv reduction → split the kv dim across gl_WorkGroupID.y
        // into partials, then sum them.
        let pv_base_wg = (mpad / 64) * (hdu / 64) * nh as u32;
        let n_splits = match std::env::var("INFR_PV_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            Some(v) => v,
            None => {
                if pv_base_wg >= 1024 || kv_pad < 4096 {
                    1u32
                } else {
                    (2048 / pv_base_wg.max(1))
                        .clamp(2, 8)
                        .min(kv_pad / 2048)
                        .max(1)
                }
            }
        };
        let ksplit = kv_pad.div_ceil(n_splits).div_ceil(32) * 32;
        // 8-warp/256-thread PV warptile (BN=128=hd, matches ollama's mul_mm) when hd%128; else the
        // 4-warp/2×2 attn_pv (also handles hd<128, e.g. hd=64). INFR_NO_PV_WARP forces the 4-warp.
        let pv_warp = hdu.is_multiple_of(128) && std::env::var("INFR_NO_PV_WARP").is_err();
        let (pv_name, pv_spv, pv_bn) = if pv_warp {
            ("attn_pv_warp", crate::gemm::attn_pv_warp_spv(), 128u32)
        } else {
            ("attn_pv", crate::gemm::attn_pv_spv(), 64u32)
        };
        let kpv = self.be.kernel_sg(pv_name, pv_spv, 3, 28, 32);
        let mut pp = [0u8; 28];
        pp[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pp[4..8].copy_from_slice(&kv_pad.to_ne_bytes());
        pp[8..12].copy_from_slice(&hdu.to_ne_bytes());
        pp[12..16].copy_from_slice(&(nh as u32).to_ne_bytes());
        pp[16..20].copy_from_slice(&(nkv as u32).to_ne_bytes());
        pp[20..24].copy_from_slice(&n_splits.to_ne_bytes());
        pp[24..28].copy_from_slice(&ksplit.to_ne_bytes());
        let pv_tiles = (mpad / 64) * (hdu / pv_bn);
        let pv_out = if n_splits == 1 { attn } else { pv_part };
        self.dispatch3(
            kpv,
            &[Self::vkb(s), Self::vkb(vc), Self::vkb(pv_out)],
            1,
            &pp,
            pv_tiles,
            n_splits,
            nh as u32,
        );
        if n_splits > 1 {
            // sum the per-split partials → attn
            let total = mpad * nh as u32 * hdu;
            let kr = self
                .be
                .kernel("attn_pv_reduce", crate::gemm::attn_pv_reduce_spv(), 2, 8);
            let mut pr = [0u8; 8];
            pr[0..4].copy_from_slice(&total.to_ne_bytes());
            pr[4..8].copy_from_slice(&n_splits.to_ne_bytes());
            self.dispatch(
                kr,
                &[Self::vkb(pv_part), Self::vkb(attn)],
                1,
                &pr,
                total.div_ceil(256),
            );
        }
    }

    /// Flash-attention prefill: fused QK→softmax→PV, no materialized S buffer. `q`=[mpad,nh,hd] f16,
    /// `kc`/`vc`=[kv_len,nkv,hd] f16, `attn`=[mpad,nh*hd] f32 out. `pos_offset` = abs position of row
    /// 0 (causal). Split-K over kv for occupancy at high ctx (few q tiles): each (q-tile, head, split)
    /// emits an online-softmax partial into `po`/`pm`/`pl`, merged by attn_flash_combine. Scratch
    /// (caller, sized for ≤8 splits): `po` = 8·mpad·nh·hd f32, `pm`/`pl` = 8·mpad·nh f32. n_splits==1
    /// → single fused kernel writing `attn` directly (no scratch touched).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_flash(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        attn: &dyn Buffer,
        po: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        n: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
    ) {
        let mpad = (n.div_ceil(64) * 64) as u32;
        let base_wg = (mpad / 64) * nh as u32;
        let n_splits = match std::env::var("INFR_FLASH_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            Some(v) => v,
            None => {
                if base_wg >= 1024 || kv_len < 4096 {
                    1u32
                } else {
                    (2048 / base_wg.max(1))
                        .clamp(2, 8)
                        .min((kv_len / 2048).max(1) as u32)
                }
            }
        };
        // hd=128 → register-blocked partial (always via partial+combine). Other hd → the
        // 4-subgroup path: single fused kernel for n_splits==1, else the scalar partial.
        //
        // The warp partial's shared scratch is bm*908 B (Ss+Ps+Os+mrow/lrow/corr, BN=64/HD=128).
        // Pick the largest tile the device's maxComputeSharedMemorySize allows: bm=64 → 58112 B (needs
        // 64 KB, e.g. RADV); bm=32 → 29056 B (fits NVIDIA 48 KB / MoltenVK 32 KB). The transformer
        // skips flash entirely when even bm=32 won't fit, so one of these always fits here.
        let shared_limit = self.be.max_shared_memory_bytes();
        let bm64_shared = 64 * crate::FLASH_SHARED_PER_ROW; // 58112 B
                                                            // INFR_FLASH_BM=32 forces the small (29056 B) tile even on a 64 KB device, so the bm=32
                                                            // shaders get numeric-parity coverage on any GPU (they otherwise only run on sub-64 KB ones).
        let force_bm32 = std::env::var("INFR_FLASH_BM").ok().as_deref() == Some("32");
        // Partial tiles (rows < 64, the small-m deep-kv tier) take BM=32: half the padded-row
        // waste and half the shared scratch (2x resident workgroups). Measured @d16384 on a
        // 7900 XTX: pp24 1122 -> 1312 t/s, pp32 1510 -> 1734 (llama.cpp: 1313/1933). Full
        // tiles (rows >= 64) keep the device-based pick — BM=64 wins there when shared fits.
        let bm: u32 = if !force_bm32 && shared_limit >= bm64_shared && n >= 64 {
            64
        } else {
            32
        };
        // Each workgroup covers `bm` query rows → mpad/bm×nh groups (mpad is 64-aligned → /32 exact).
        let tile_wg = (mpad / bm) * nh as u32;
        // INFR_NO_FLASH_WARP routes to the non-warp partial. Both warp and non-warp paths have a
        // bm=32 build (29056 B) that fits sub-64 KB devices, so the knob is honored everywhere —
        // no longer forced back to warp on NVIDIA / MoltenVK.
        let warp = hd == 128 && std::env::var("INFR_NO_FLASH_WARP").is_err();
        if n_splits == 1 && !warp {
            let (fname, fspv): (&'static str, &[u32]) = if bm == 32 {
                ("attn_flash_bm32", crate::gemm::attn_flash_bm32_spv())
            } else {
                ("attn_flash", crate::gemm::attn_flash_spv())
            };
            let k = self.be.kernel_sg(fname, fspv, 4, 24, 32);
            let mut p = [0u8; 24];
            p[0..4].copy_from_slice(&mpad.to_ne_bytes());
            p[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
            p[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
            p[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
            p[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
            p[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
            self.dispatch(
                k,
                &[Self::vkb(q), Self::vkb(kc), Self::vkb(vc), Self::vkb(attn)],
                1,
                &p,
                tile_wg,
            );
            return;
        }
        // split-K partials
        let ksplit = (kv_len as u32).div_ceil(n_splits).div_ceil(64) * 64;
        // hd=128 → register-blocked warp partial; else the 4-subgroup partial. Each picks its
        // bm-sized shared build (warp/partial share the bm*908 B footprint) and covers tile_wg groups.
        let (pname, pspv): (&'static str, &[u32]) = match (warp, bm) {
            (true, 32) => (
                "attn_flash_warp_bm32",
                crate::gemm::attn_flash_warp_bm32_spv(),
            ),
            (true, _) => ("attn_flash_warp", crate::gemm::attn_flash_warp_spv()),
            (false, 32) => (
                "attn_flash_partial_bm32",
                crate::gemm::attn_flash_partial_bm32_spv(),
            ),
            (false, _) => ("attn_flash_partial", crate::gemm::attn_flash_partial_spv()),
        };
        let kp = self.be.kernel_sg(pname, pspv, 6, 32, 32);
        let mut pp = [0u8; 32];
        pp[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pp[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        pp[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        pp[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        pp[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        pp[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        pp[24..28].copy_from_slice(&n_splits.to_ne_bytes());
        pp[28..32].copy_from_slice(&ksplit.to_ne_bytes());
        self.dispatch3(
            kp,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(po),
                Self::vkb(pm),
                Self::vkb(pl),
            ],
            3,
            &pp,
            tile_wg,
            n_splits,
            1,
        );
        // combine → attn
        let kc2 = self.be.kernel_sg(
            "attn_flash_combine",
            crate::gemm::attn_flash_combine_spv(),
            4,
            16,
            32,
        );
        let mut pc2 = [0u8; 16];
        pc2[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pc2[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        pc2[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        pc2[12..16].copy_from_slice(&n_splits.to_ne_bytes());
        self.dispatch(
            kc2,
            &[Self::vkb(po), Self::vkb(pm), Self::vkb(pl), Self::vkb(attn)],
            1,
            &pc2,
            mpad * nh as u32,
        );
    }

    /// FlashAttention-2 register-O prefill (Br=128, per-thread register accumulator → no [Br][HD]
    /// shared O; 2× the query tile of the shared-Os flash → fewer q-tiles). hd MUST be 128 and the
    /// caller MUST allocate q/attn/po to mpad128 = ceil(n/128)*128 rows. Split-K → partials → combine.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_flash_reg(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        attn: &dyn Buffer,
        po: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        n: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
    ) {
        // mpad is 128-aligned → divisible by both BR tiles.
        let mpad = (n.div_ceil(128) * 128) as u32;
        // Register-O shared = BR*FLASH_REG_SHARED_PER_ROW: BR=128 → 58880 B (needs 64 KB); BR=64 →
        // 29440 B (NVIDIA 48 KB / MoltenVK 32 KB). Largest that fits; transformer skips reg if neither.
        let br: u32 = if self.be.max_shared_memory_bytes() >= 128 * crate::FLASH_REG_SHARED_PER_ROW
        {
            128
        } else {
            64
        };
        let base_wg = (mpad / br) * nh as u32;
        let n_splits = match std::env::var("INFR_FLASH_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            Some(v) => v,
            None => {
                if base_wg >= 1024 || kv_len < 4096 {
                    1u32
                } else {
                    (2048 / base_wg.max(1))
                        .clamp(2, 8)
                        .min((kv_len / 2048).max(1) as u32)
                }
            }
        };
        let ksplit = (kv_len as u32).div_ceil(n_splits).div_ceil(64) * 64;
        let (rname, rspv): (&'static str, &[u32]) = if br == 64 {
            (
                "attn_flash_reg_br64",
                crate::gemm::attn_flash_reg_br64_spv(),
            )
        } else {
            ("attn_flash_reg", crate::gemm::attn_flash_reg_spv())
        };
        let kp = self.be.kernel_sg(rname, rspv, 6, 32, 32);
        let mut pp = [0u8; 32];
        pp[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pp[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        pp[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        pp[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        pp[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        pp[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        pp[24..28].copy_from_slice(&n_splits.to_ne_bytes());
        pp[28..32].copy_from_slice(&ksplit.to_ne_bytes());
        self.dispatch3(
            kp,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(po),
                Self::vkb(pm),
                Self::vkb(pl),
            ],
            3,
            &pp,
            base_wg,
            n_splits,
            1,
        );
        let kc2 = self.be.kernel_sg(
            "attn_flash_combine",
            crate::gemm::attn_flash_combine_spv(),
            4,
            16,
            32,
        );
        let mut pc2 = [0u8; 16];
        pc2[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pc2[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        pc2[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        pc2[12..16].copy_from_slice(&n_splits.to_ne_bytes());
        self.dispatch(
            kc2,
            &[Self::vkb(po), Self::vkb(pm), Self::vkb(pl), Self::vkb(attn)],
            1,
            &pc2,
            mpad * nh as u32,
        );
    }

    /// Cast-copy f32 `src[0..n]` → f16 `dst[off..off+n]` (write f32 activations into the f16 cache).
    pub fn store_f16(&self, src: &dyn Buffer, dst: &dyn Buffer, n: usize, off: usize) {
        let k = self
            .be
            .kernel("store_f16", crate::gemm::store_f16_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(off as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Quantize `src[0..n]` → the Vulkan planar Q8_0 KV cache at element offset `off`. `cap` = total
    /// cache elements (the scales region begins at byte `cap`). `src_f16` selects the f16-source
    /// variant (the un-fused roped K staging); f32 otherwise (the V projection). `n`/`off` are
    /// block-aligned (KV row width is a multiple of 32).
    pub fn store_q8(
        &self,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
        off: usize,
        cap: usize,
        src_f16: bool,
    ) {
        let (name, spv) = if src_f16 {
            ("store_q8_f16", crate::gemm::store_q8_f16_spv())
        } else {
            ("store_q8", crate::gemm::store_q8_spv())
        };
        let k = self.be.kernel(name, spv, 2, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(off as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(cap as u32).to_ne_bytes());
        // One workgroup (32 lanes) per 32-element block.
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            (n as u32) / 32,
        );
    }

    /// Record-once decode variant of [`Recorder::store_q8`]: the write offset is `p_pos*n` (one KV
    /// row at the token's position), `p_pos` read from the `params` SSBO so the buffer replays.
    pub fn store_q8_dyn(
        &self,
        src: &dyn Buffer,
        params: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
        cap: usize,
        src_f16: bool,
    ) {
        let (name, spv) = if src_f16 {
            ("store_q8_f16_dyn", crate::gemm::store_q8_f16_dyn_spv())
        } else {
            ("store_q8_dyn", crate::gemm::store_q8_dyn_spv())
        };
        let k = self.be.kernel(name, spv, 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        // [4..8] off: unused (from params)
        push[8..12].copy_from_slice(&(cap as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(params), Self::vkb(dst)],
            1,
            &push,
            (n as u32) / 32,
        );
    }

    /// Expand `n` elements of the planar Q8_0 cache → an f16 buffer (dequant the KV prefix so the
    /// f16 flash / non-FA prefill kernels can read it; the persistent cache stays Q8). `cap` = total
    /// cache elements (the planar scales region base).
    pub fn dequant_q8_f16(&self, src: &dyn Buffer, dst: &dyn Buffer, n: usize, cap: usize) {
        let k = self
            .be
            .kernel("dequant_q8_f16", crate::gemm::dequant_q8_f16_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(cap as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Quantize `src[0..n]` → the standard GGUF-block KV cache of `dt` at element offset `off` (one
    /// thread per 32-block). `src_f16` = f16 source (roped K); f32 otherwise (V).
    pub fn quant_kv(
        &self,
        dt: infr_core::DType,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
        off: usize,
        src_f16: bool,
    ) {
        let (name, spv) = crate::gemm::quant_kv_kernel(dt, src_f16);
        let k = self.be.kernel(name, spv, 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(off as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            ((n / 32) as u32).div_ceil(64),
        );
    }

    /// Quantize `src[0..n]` → a TurboQuant KV cache of `dt` at element offset `off` (one thread per
    /// 128-block: L2-norm + WHT + centroid). `src_f16` = f16 K source; f32 V otherwise.
    pub fn quant_turbo(
        &self,
        dt: infr_core::DType,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
        off: usize,
        src_f16: bool,
    ) {
        let (name, spv) = crate::gemm::quant_turbo_kernel(dt, src_f16);
        let k = self.be.kernel(name, spv, 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(off as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            ((n / 128) as u32).div_ceil(64),
        );
    }

    /// Expand `n` elements of a TurboQuant KV cache of `dt` → an f16 buffer (unpack + inverse WHT).
    /// One thread per 128-block.
    pub fn dequant_turbo_f16(
        &self,
        dt: infr_core::DType,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
    ) {
        let (name, spv) = crate::gemm::dequant_turbo_kernel(dt);
        let k = self.be.kernel(name, spv, 2, 4);
        let mut push = [0u8; 4];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            ((n / 128) as u32).div_ceil(64),
        );
    }

    /// Cast-store `src[0..n]` → a DENSE KV cache of `dst_dt` (F32/Bf16) at element offset `off`.
    /// `src_f16` = f16 source (roped K); f32 otherwise (V). One thread per element.
    pub fn store_kv_dense(
        &self,
        dst_dt: infr_core::DType,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
        off: usize,
        src_f16: bool,
    ) {
        let (name, spv) = crate::gemm::store_kv_dense_kernel(dst_dt, src_f16);
        let k = self.be.kernel(name, spv, 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(off as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Expand `n` elements of a standard-GGUF-block KV cache of `dt` → an f16 buffer (reuses the
    /// native_decode `dq()`), so the f16 attention can read a quantized KV prefix. One thread/element.
    pub fn dequant_kv_f16(
        &self,
        dt: infr_core::DType,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
    ) {
        let (name, spv) = crate::gemm::dequant_kv_kernel(dt);
        let k = self.be.kernel(name, spv, 2, 4);
        let mut push = [0u8; 4];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Qwen3 QK-norm + RoPE over `x[rows, nheads, hd]` → `y` at rows `out_base..`. `nw` is the
    /// per-head [hd] norm weight. (q: out_base=0; k: out_base=pos so it lands in the cache.)
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope(
        &self,
        x: &dyn Buffer,
        nw: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        rope_pos: usize,
        out_base: usize,
        eps: f32,
        // gemma4 full-attention layers: per-pair RoPE frequency divisors (`Some`); `None` = normal
        // RoPE (qwen3 / gemma3 / gemma4 SWA layers).
        freq_factors: Option<&dyn Buffer>,
    ) {
        let mut push = [0u8; 32];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(rope_pos as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(out_base as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        match freq_factors {
            Some(ff) => {
                let k =
                    self.be
                        .kernel("qk_norm_rope_ff", crate::gemm::qk_norm_rope_ff_spv(), 4, 32);
                self.dispatch(
                    k,
                    &[Self::vkb(x), Self::vkb(nw), Self::vkb(ff), Self::vkb(y)],
                    1,
                    &push,
                    (rows * nheads) as u32,
                );
            }
            None => {
                let k = self
                    .be
                    .kernel("qk_norm_rope", crate::gemm::qk_norm_rope_spv(), 3, 32);
                self.dispatch(
                    k,
                    &[Self::vkb(x), Self::vkb(nw), Self::vkb(y)],
                    1,
                    &push,
                    (rows * nheads) as u32,
                );
            }
        }
    }

    /// Flash-decoding attention (q_len==1): split the KV range into `n_chunks` chunks of `chunk`,
    /// compute per-chunk softmax partials in parallel (`pm`/`pl`/`pacc`), then combine into `o`.
    /// Parallelizes attention across `nh*n_chunks` workgroups so it stays fast at long context.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        // Small-m generalization: `rows` query rows at absolute positions pos..pos+rows-1 (decode
        // = 1 row at kv_len-1). Row r's causal end is pos+r+1; workgroup y picks the row; the
        // partial scratch and `o` are [rows*nh, ...] row-major — the combine treats (rows*nh) as
        // its head count unchanged, and o IS the [rows, nh, hd] dst.
        rows: usize,
        pos: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
        // DiffusionGemma canvas denoise (docs/DIFFUSIONGEMMA.md, `AttnMask::Canvas`): every row
        // attends the SAME fixed bidirectional `[lo, kv_len)`, overriding BOTH the per-row causal
        // end (`qpos1 = pos+row+1` → `kv_len`) and the sliding-window `lo` formula. Rides the
        // shader's otherwise-dead `rows` push-constant slot (see `attn_partial.comp`: the
        // dispatch grid already encodes the row via `gl_WorkGroupID.y`, so the field was never
        // read) as `lo+1` (0 = disabled) — no push-constant layout change, no effect on any
        // existing (non-canvas) caller.
        canvas_lo: Option<usize>,
        // Per-side planar Q8_0 KV read (K and V independent). Planar `cap` = total cache elements
        // (the scales-region base), unused when both are f16.
        k_q8: bool,
        v_q8: bool,
        cap: usize,
        // Rows-batched pass 1 (attn_partial_mrows_c256, K/V once per 4-row group): the caller
        // gates on rows/kv_len/hd (see the adapter's `batched_attn`) and MUST pass chunk <= 256.
        // Mutually exclusive with `canvas_lo` (the adapter's `batched_attn` gate excludes Canvas).
        batched: bool,
    ) {
        // pass 1: per-chunk partials (subgroup-reduction QK; needs requiredSubgroupSize=32)
        let (p1name, p1spv) = match (k_q8, v_q8) {
            (false, false) if crate::gemm::attn_hd_spec_disabled() => {
                ("attn_partial_nohd", crate::gemm::attn_partial_nohd_spv())
            }
            (false, false) => ("attn_partial", crate::gemm::attn_partial_spv()),
            (true, false) => ("attn_partial_kq8", crate::gemm::attn_partial_kq8_spv()),
            (false, true) => ("attn_partial_vq8", crate::gemm::attn_partial_vq8_spv()),
            (true, true) => ("attn_partial_q8", crate::gemm::attn_partial_q8_spv()),
        };
        // Rows-BATCHED pass 1 (INFR_MROWS_ATTN=1 [+ INFR_MROWS_CHUNK=256], OFF by default): one
        // workgroup per (head, chunk) streams K/V ONCE for a 4-row group — one subgroupAdd(vec4)
        // per key, scores staged in LDS. The occupancy sweep on a 7900 XTX (pp4@d16384) was
        // monotone in LDS — 22KB (RB=8) 188 t/s, 11KB (RB=4) 325, 7KB (RB=4 + c256) 456, vs the
        // per-row grid's 548 — but even at 7KB the design beats per-row only in a NARROW band:
        // rows >= ~12 AND deep kv (pp16@d16384 832 -> 925, pp32 963 -> 1123; pp4-8 lose or wash,
        // pp16@d4096 loses). On RDNA3 the per-row grid's rows-x extra workgroups fill the DRAM
        // queue better than the batched form's rows-/ bandwidth saving, so the spec-verify band
        // (m <= 8) keeps per-row. If the m >= 12-at-depth band ever matters, route here on
        // (rows, kv_len) — or better, build the LDS-staged K-TILE kernel (per-thread full dots,
        // no cross-lane reductions), which is how llama.cpp wins that cell (1056 t/s).
        debug_assert!(!batched || (chunk <= 256 && hd <= 128 && !k_q8 && !v_q8 && rows >= 2));
        let (p1name, p1spv) = if batched {
            (
                "attn_partial_mrows_c256",
                crate::gemm::attn_partial_mrows_c256_spv(),
            )
        } else {
            (p1name, p1spv)
        };
        let k1 = self.be.kernel_sg(p1name, p1spv, 6, 44, 32);
        let mut p1 = [0u8; 44];
        p1[0..4].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        p1[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        p1[16..20].copy_from_slice(&(chunk as u32).to_ne_bytes());
        p1[20..24].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p1[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        p1[28..32].copy_from_slice(&scale.to_ne_bytes());
        p1[32..36].copy_from_slice(&(cap as u32).to_ne_bytes());
        p1[36..40].copy_from_slice(&(pos as u32).to_ne_bytes());
        // `rows` is dead in `attn_partial.comp` (this slot carries `canvas_lo+1`, 0 = disabled,
        // instead — see `canvas_lo`'s doc above) but `attn_partial_mrows.comp` (the `batched`
        // kernel) genuinely reads it (`rb = min(RB, pc.rows - row0)`, the short-last-row-group
        // clamp) — canvas is never batched (the adapter's `batched_attn` gate excludes Canvas),
        // so keep sending the real row count there.
        debug_assert!(!batched || canvas_lo.is_none());
        let rows_field = if batched {
            rows as u32
        } else {
            canvas_lo.map(|lo| lo as u32 + 1).unwrap_or(0)
        };
        p1[40..44].copy_from_slice(&rows_field.to_ne_bytes());
        // Batched: workgroup y = 4-row group; per-row: y = row.
        let gy = if batched { rows.div_ceil(4) } else { rows };
        self.dispatch3(
            k1,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(pm),
                Self::vkb(pl),
                Self::vkb(pacc),
            ],
            3,
            &p1,
            (nh * n_chunks) as u32,
            gy as u32,
            1,
        );
        // pass 2: combine — split each (row, head)'s hd outputs across `ntile` workgroups for
        // occupancy. The combine is row-agnostic: rows*nh independent [n_chunks] partial sets.
        let k2 = self
            .be
            .kernel("attn_combine", crate::gemm::attn_combine_spv(), 4, 16);
        let ntile = if hd.is_multiple_of(4) { 4u32 } else { 1u32 };
        let mut p2 = [0u8; 16];
        p2[0..4].copy_from_slice(&((rows * nh) as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(hd as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p2[12..16].copy_from_slice(&ntile.to_ne_bytes());
        self.dispatch(
            k2,
            &[Self::vkb(pm), Self::vkb(pl), Self::vkb(pacc), Self::vkb(o)],
            1,
            &p2,
            (rows * nh) as u32 * ntile,
        );
    }

    // ---- Record-once decode variants (`_dyn`) ----
    // These read the per-token `pos`/`kv_len` from a host-updated `params` SSBO ([pos, kv_len] u32)
    // instead of push constants, so the decode command buffer can be recorded once and replayed every
    // token (only `params` + the embedding change). Used ONLY by the GPU-resident decode path; every
    // other caller keeps the push-constant kernels. `params` is inserted before the output(s) so the
    // recorder's reads|writes split stays output-last.

    /// QK-norm + RoPE, pos from `params`. `out_base_mul` = 0 for Q (write to a temp), 1 for K (write
    /// to the cache at row pos).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_dyn(
        &self,
        x: &dyn Buffer,
        nw: &dyn Buffer,
        params: &dyn Buffer,
        ff: Option<&dyn Buffer>,
        y: &dyn Buffer,
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        out_base_mul: usize,
        eps: f32,
    ) {
        // With freq_factors (gemma4 full-attention layers) `ff` binds at 3 and the output shifts
        // to 4 — same PC layout either way.
        let k = match ff {
            Some(_) => self.be.kernel(
                "qk_norm_rope_dyn_ff",
                crate::gemm::qk_norm_rope_dyn_ff_spv(),
                5,
                32,
            ),
            None => self.be.kernel(
                "qk_norm_rope_dyn",
                crate::gemm::qk_norm_rope_dyn_spv(),
                4,
                32,
            ),
        };
        let mut push = [0u8; 32];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        // [20..24] rope_pos: unused (from params)
        push[24..28].copy_from_slice(&(out_base_mul as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        match ff {
            Some(f) => self.dispatch(
                k,
                &[
                    Self::vkb(x),
                    Self::vkb(nw),
                    Self::vkb(params),
                    Self::vkb(f),
                    Self::vkb(y),
                ],
                1,
                &push,
                (rows * nheads) as u32,
            ),
            None => self.dispatch(
                k,
                &[Self::vkb(x), Self::vkb(nw), Self::vkb(params), Self::vkb(y)],
                1,
                &push,
                (rows * nheads) as u32,
            ),
        }
    }

    /// Cast-copy f32 `src[0..n]` → f16 `dst[pos*n..]` (one KV row at position pos from `params`).
    pub fn store_f16_dyn(&self, src: &dyn Buffer, params: &dyn Buffer, dst: &dyn Buffer, n: usize) {
        let k = self
            .be
            .kernel("store_f16_dyn", crate::gemm::store_f16_dyn_spv(), 3, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        // [4..8] off: unused (computed as pos*n from params)
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(params), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Causal GQA over the KV cache (q_len==1), pos_offset from `params`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_dyn(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        params: &dyn Buffer,
        o: &dyn Buffer,
        q_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        scale: f32,
        window: usize,
        // Q8_0 KV cache (K==V==q8): planar dequant-on-read variant. `false` = f16 cache.
        q8: bool,
        // Planar Q8 scales region base = total cache elements (`cap`). Unused when `q8` is false.
        cap: usize,
    ) {
        let (name, spv) = if q8 {
            (
                "attention_kv_dyn_q8",
                crate::gemm::attention_kv_dyn_q8_spv(),
            )
        } else {
            ("attention_kv_dyn", crate::gemm::attention_kv_dyn_spv())
        };
        let kern = self.be.kernel(name, spv, 5, 36);
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(q_len as u32).to_ne_bytes());
        // [4..8] kv_len: unused
        push[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        // [20..24] pos_offset: unused (from params)
        push[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&scale.to_ne_bytes());
        push[32..36].copy_from_slice(&(cap as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(params),
                Self::vkb(o),
            ],
            1,
            &push,
            (q_len * nh) as u32,
        );
    }

    /// Split-K decode attention, kv_len from `params`. `chunk`/`n_chunks` stay push constants (they
    /// define the dispatch structure; the caller re-records when they change).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split_dyn(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        params: &dyn Buffer,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
    ) {
        self.attention_kv_split_dyn_inner(
            q, kc, vc, o, pm, pl, pacc, params, nh, nkv, hd, chunk, n_chunks, scale, window,
        )
    }

    /// Record the split-K replay prologue ONCE per execute: derives the live chunk count from the
    /// params SSBO's kv_len and writes `args = [nh·live, 1, 1, live]` — the indirect dispatch args
    /// and combine loop bound that every subsequent [`Self::attention_kv_split_dynac`] in the same
    /// execute shares (kv_len is the same for every layer of a token).
    pub fn attn_live_prologue(
        &self,
        params: &dyn Buffer,
        args: &dyn Buffer,
        nh: usize,
        chunk: usize,
        window: usize,
    ) {
        let kl = self
            .be
            .kernel("attn_live", crate::gemm::attn_live_spv(), 2, 12);
        let mut p0 = [0u8; 12];
        p0[0..4].copy_from_slice(&(nh as u32).to_ne_bytes());
        p0[4..8].copy_from_slice(&(chunk as u32).to_ne_bytes());
        p0[8..12].copy_from_slice(&(window as u32).to_ne_bytes());
        self.dispatch(kl, &[Self::vkb(params), Self::vkb(args)], 1, &p0, 1);
    }

    /// SELF-CHUNKING variant for record-once REPLAY over a growing kv_len. A one-thread prologue
    /// (`attn_live`) derives the adaptive chunk (~32 chunks/head, 64..512, floored by the baked
    /// minimum `chunk`) from the LIVE kv_len and writes the partial pass's INDIRECT dispatch args
    /// (`args`, ≥16 bytes: [gx,gy,gz,live]) — so one recorded plan launches exactly nh·live
    /// workgroups at every depth (no dead workgroups shallow, no re-record deep). The combine
    /// loops the prologue's live count. `n_chunks` is only the pm/pl/pacc scratch STRIDE/capacity
    /// (cap.div_ceil(chunk), ≤1024).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split_dynac(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        params: &dyn Buffer,
        args: &dyn Buffer,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
        // Q8_0 KV cache (K==V==q8): coalesced planar Q8 read variant. `false` = f16 cache.
        q8: bool,
        // Planar Q8 scales region base = total cache elements (`cap`). Unused when `q8` is false.
        cap: usize,
    ) {
        // pass 1: self-chunking partials, workgroup count from `args` (the caller records ONE
        // `attn_live_prologue` per (nh, chunk, window) key — kv_len is identical for every layer
        // of a token, so the args buffer is shared across same-key attention ops instead of
        // re-derived per layer).
        let (p1name, p1spv) = if q8 {
            (
                "attn_partial_dynac_q8",
                crate::gemm::attn_partial_dynac_q8_spv(),
            )
        } else if crate::gemm::attn_hd_spec_disabled() {
            (
                "attn_partial_dynac_nohd",
                crate::gemm::attn_partial_dynac_nohd_spv(),
            )
        } else {
            ("attn_partial_dynac", crate::gemm::attn_partial_dynac_spv())
        };
        // attn_partial.comp's `layout(push_constant) uniform PC {...}` block is declared
        // UNCONDITIONALLY (11 x 4-byte members = 44 bytes) — `pos`/`rows` (offsets 36..44) are
        // dead code under `-DUSE_PARAMS` (never read in the compiled SPIR-V for this variant, see
        // the shader's `#ifdef USE_PARAMS` guard), so a range of 36 looked sufficient. It isn't:
        // vkCreateComputePipelines validates the pipeline layout's range against the shader's
        // declared Block SIZE, not per-member static use (VUID-VkComputePipelineCreateInfo-layout-
        // 10069 — confirmed firing on the 7900 XTX under validation layers: "push constant buffer
        // Block with range [0, 44] which is outside the VkPushConstantRange of [0, 36]"). The
        // range must cover the full declared block; the trailing 8 bytes stay zero and unread.
        let k1 = self.be.kernel_sg(p1name, p1spv, 7, 44, 32);
        let mut p1 = [0u8; 44];
        p1[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        p1[16..20].copy_from_slice(&(chunk as u32).to_ne_bytes());
        p1[20..24].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p1[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        p1[28..32].copy_from_slice(&scale.to_ne_bytes());
        p1[32..36].copy_from_slice(&(cap as u32).to_ne_bytes());
        self.dispatch_indirect(
            k1,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(params),
                Self::vkb(pm),
                Self::vkb(pl),
                Self::vkb(pacc),
            ],
            3,
            &p1,
            Self::vkb(args),
            0,
        );

        // pass 2: combine over the live chunks
        let k2 = self.be.kernel(
            "attn_combine_live",
            crate::gemm::attn_combine_live_spv(),
            5,
            16,
        );
        let ntile = if hd.is_multiple_of(4) { 4u32 } else { 1u32 };
        let mut p2 = [0u8; 16];
        p2[0..4].copy_from_slice(&(nh as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(hd as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p2[12..16].copy_from_slice(&ntile.to_ne_bytes());
        self.dispatch(
            k2,
            &[
                Self::vkb(pm),
                Self::vkb(pl),
                Self::vkb(pacc),
                Self::vkb(args),
                Self::vkb(o),
            ],
            1,
            &p2,
            nh as u32 * ntile,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_kv_split_dyn_inner(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        params: &dyn Buffer,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
    ) {
        let (p1name, p1spv) = if crate::gemm::attn_hd_spec_disabled() {
            (
                "attn_partial_dyn_nohd",
                crate::gemm::attn_partial_dyn_nohd_spv(),
            )
        } else {
            ("attn_partial_dyn", crate::gemm::attn_partial_dyn_spv())
        };
        // Same fix as `attention_kv_split_dynac` above: attn_partial.comp's PC block is always 44
        // bytes; a shorter declared range (32, here) fails
        // VUID-VkComputePipelineCreateInfo-layout-10069 even though `cap`/`pos`/`rows` are dead
        // code in this (`-DUSE_PARAMS`, no `-DSELF_CHUNK`) variant.
        let k1 = self.be.kernel_sg(p1name, p1spv, 7, 44, 32);
        let mut p1 = [0u8; 44];
        // [0..4] kv_len: unused (from params)
        p1[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        p1[16..20].copy_from_slice(&(chunk as u32).to_ne_bytes());
        p1[20..24].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p1[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        p1[28..32].copy_from_slice(&scale.to_ne_bytes());
        self.dispatch(
            k1,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(params),
                Self::vkb(pm),
                Self::vkb(pl),
                Self::vkb(pacc),
            ],
            3,
            &p1,
            (nh * n_chunks) as u32,
        );
        // pass 2: combine (structure-only, unchanged from the push-constant path)
        let k2 = self
            .be
            .kernel("attn_combine", crate::gemm::attn_combine_spv(), 4, 16);
        let ntile = if hd.is_multiple_of(4) { 4u32 } else { 1u32 };
        let mut p2 = [0u8; 16];
        p2[0..4].copy_from_slice(&(nh as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(hd as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p2[12..16].copy_from_slice(&ntile.to_ne_bytes());
        self.dispatch(
            k2,
            &[Self::vkb(pm), Self::vkb(pl), Self::vkb(pacc), Self::vkb(o)],
            1,
            &p2,
            nh as u32 * ntile,
        );
    }

    /// Record a buffer→buffer copy of `bytes` from `src[src_offset..]` into `dst[dst_offset..]`.
    pub fn copy(
        &self,
        src: &dyn Buffer,
        src_offset: usize,
        dst: &dyn Buffer,
        dst_offset: usize,
        bytes: usize,
    ) {
        // Transfer chokepoint: previously un-stamped, so copy time silently landed in the
        // preceding label's bucket. Distinct from the `copy_strided` compute kernel.
        self.stamp("copy_buffer");
        let device = &self.be.shared.device;
        self.sync(&[Self::vkb(src)], &[Self::vkb(dst)], true);
        self.dirty_transfer.set(true);
        unsafe {
            device.cmd_copy_buffer(
                self.cmd,
                Self::vkb(src),
                Self::vkb(dst),
                &[vk::BufferCopy {
                    src_offset: src_offset as u64,
                    dst_offset: dst_offset as u64,
                    size: bytes as u64,
                }],
            );
        }
    }

    pub fn attention(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        o: &dyn Buffer,
        t: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) {
        let kern = self
            .be
            .kernel("attention", crate::gemm::attention_spv(), 4, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[Self::vkb(q), Self::vkb(k), Self::vkb(v), Self::vkb(o)],
            1,
            &push,
            (t * nh) as u32,
        );
    }

    /// Fused SwiGLU over a combined `gu` `[rows, 2*nff]` → `y` `[rows, nff]`.
    pub fn silu_mul_fused(&self, gu: &dyn Buffer, y: &dyn Buffer, rows: usize, nff: usize) {
        let k = self
            .be
            .kernel("silu_mul_fused", crate::gemm::silu_mul_fused_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nff as u32).to_ne_bytes());
        // Thread-per-element kernel (local_size 64) — dispatch element-count/64 workgroups.
        self.dispatch(
            k,
            &[Self::vkb(gu), Self::vkb(y)],
            1,
            &push,
            ((rows * nff) as u32).div_ceil(64),
        );
    }

    /// Fused GeGLU (GELU tanh-approx gate) over a combined `gu` `[rows, 2*nff]` → `y` `[rows, nff]`.
    /// Same layout/dispatch as [`silu_mul_fused`]; gemma uses GELU instead of SiLU.
    pub fn gelu_mul_fused(&self, gu: &dyn Buffer, y: &dyn Buffer, rows: usize, nff: usize) {
        let k = self
            .be
            .kernel("gelu_mul_fused", crate::gemm::gelu_mul_fused_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nff as u32).to_ne_bytes());
        // Thread-per-element kernel (local_size 64) — dispatch element-count/64 workgroups.
        self.dispatch(
            k,
            &[Self::vkb(gu), Self::vkb(y)],
            1,
            &push,
            ((rows * nff) as u32).div_ceil(64),
        );
    }

    pub fn silu_mul(&self, gate: &dyn Buffer, up: &dyn Buffer, y: &dyn Buffer, n: usize) {
        let k = self
            .be
            .kernel("silu_mul", crate::gemm::silu_mul_spv(), 3, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(gate), Self::vkb(up), Self::vkb(y)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Gated-DeltaNet recurrence, one token (qwen35 SSM). The persistent `state` buffer
    /// `[nv*kd*vd]` is updated in place; `out` is `[nv*vd]`. One workgroup per value head; the
    /// `nk` q/k heads are tiled up to `nv`. See shaders/deltanet.comp.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        blog: &dyn Buffer,
        alpha: &dyn Buffer,
        acoef: &dyn Buffer,
        dtbias: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
    ) {
        // The shader caches each column block's state [kd, 32] in shared memory (`ss[128*32]`), so kd
        // must be ≤ 128. qwen35 uses kd=128; assert so a larger head_k_dim fails loudly instead of
        // corrupting LDS.
        debug_assert!(
            kd <= 128,
            "deltanet shared-state block assumes kd ≤ 128, got {kd}"
        );
        let kern = self
            .be
            .kernel("deltanet", crate::gemm::deltanet_spv(), 9, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nk as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(kd as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(vd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&eps.to_ne_bytes());
        push[24..28].copy_from_slice(&(1.0f32 / (kd as f32).sqrt()).to_ne_bytes());
        // One workgroup per (value head, block of 32 state columns); local_size_x=32.
        let n_blk = vd.div_ceil(32);
        self.dispatch(
            kern,
            &[
                Self::vkb(q),
                Self::vkb(k),
                Self::vkb(v),
                Self::vkb(blog),
                Self::vkb(alpha),
                Self::vkb(acoef),
                Self::vkb(dtbias),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &push,
            (nv * n_blk) as u32,
        );
    }

    /// CHUNKED gated-DeltaNet prefill (chunkwise delta rule, C=32): per 32-token chunk the
    /// recurrence collapses to dense matmuls + one unit-lower-triangular solve, so the state is
    /// traversed rows/32 times instead of `rows`. Same signature/bindings as `deltanet`; math
    /// validated against the sequential form in tests/chunked_delta_math.rs. Use for rows ≥ 32
    /// (decode keeps the sequential kernel). See shaders/deltanet_chunked.comp.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_chunked(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        blog: &dyn Buffer,
        alpha: &dyn Buffer,
        acoef: &dyn Buffer,
        dtbias: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
    ) {
        debug_assert!(
            kd <= 128,
            "deltanet_chunked LDS chunk tiles assume kd ≤ 128, got {kd}"
        );
        let kern = self.be.kernel(
            "deltanet_chunked",
            crate::gemm::deltanet_chunked_spv(),
            9,
            28,
        );
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nk as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(kd as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(vd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&eps.to_ne_bytes());
        push[24..28].copy_from_slice(&(1.0f32 / (kd as f32).sqrt()).to_ne_bytes());
        // One workgroup per (value head, block of 32 state columns); local_size_x=256.
        // (COLS=16 was tried for occupancy and REGRESSED 2670→3668µs — the per-block duplicated
        // A/Wq dots dominate; that's what the split prep+scan variant hoists out.)
        let n_blk = vd.div_ceil(32);
        self.dispatch(
            kern,
            &[
                Self::vkb(q),
                Self::vkb(k),
                Self::vkb(v),
                Self::vkb(blog),
                Self::vkb(alpha),
                Self::vkb(acoef),
                Self::vkb(dtbias),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &push,
            (nv * n_blk) as u32,
        );
    }

    /// Chunked gated-DeltaNet prefill, SPLIT variant (prep + gates + scan): the chunk-parallel
    /// work (q/k normalization, intra-chunk D=K̂K̂ᵀ / Dq=Q̂K̂ᵀ dot matrices, gates) is hoisted into
    /// two fully-parallel passes, so the sequential scan pass does ONLY state-coupled work — which
    /// parallelizes cleanly over small column blocks (COLS=16 → nv·(vd/16) workgroups). The
    /// monolithic `deltanet_chunked` duplicated that shared work per column block (~37% of it).
    /// Scratch (caller-alloc'd, alloc_uninit-safe — every read slot is written by prep/gates):
    /// kn/qn [rows·nk·kd] f32, dk/dq [nchunk·nk·C·C] f32, betag/gg [nchunk·nv·C] f32, C=32.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_chunked_split(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        blog: &dyn Buffer,
        alpha: &dyn Buffer,
        acoef: &dyn Buffer,
        dtbias: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        kn: &dyn Buffer,
        qn: &dyn Buffer,
        dk: &dyn Buffer,
        dq: &dyn Buffer,
        betag: &dyn Buffer,
        gg: &dyn Buffer,
        rows: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
    ) {
        debug_assert!(kd <= 128, "deltanet split assumes kd ≤ 128, got {kd}");
        debug_assert!(
            kd.is_multiple_of(16),
            "deltanet_prep's coopmat D/Dq tiles assume kd a multiple of 16, got {kd}"
        );
        let nchunk = rows.div_ceil(32);
        // pass 1: prep — (chunk, k-head) grid. kernel_sg: the D/Dq dot matrices are computed via
        // coopmat (RDNA3 needs requiredSubgroupSize=32 pinned for coopmat correctness).
        let kp = self
            .be
            .kernel_sg("deltanet_prep", crate::gemm::deltanet_prep_spv(), 6, 20, 32);
        let mut p1 = [0u8; 20];
        p1[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p1[4..8].copy_from_slice(&(nk as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(kd as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&eps.to_ne_bytes());
        p1[16..20].copy_from_slice(&(1.0f32 / (kd as f32).sqrt()).to_ne_bytes());
        self.dispatch(
            kp,
            &[
                Self::vkb(q),
                Self::vkb(k),
                Self::vkb(kn),
                Self::vkb(qn),
                Self::vkb(dk),
                Self::vkb(dq),
            ],
            4, // kn, qn, dk, dq
            &p1,
            (nchunk * nk) as u32,
        );
        // pass 2: gates — (chunk, value-head) grid
        let kg = self
            .be
            .kernel("deltanet_gates", crate::gemm::deltanet_gates_spv(), 6, 8);
        let mut p2 = [0u8; 8];
        p2[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        self.dispatch(
            kg,
            &[
                Self::vkb(blog),
                Self::vkb(alpha),
                Self::vkb(acoef),
                Self::vkb(dtbias),
                Self::vkb(betag),
                Self::vkb(gg),
            ],
            2, // betag, gg
            &p2,
            (nchunk * nv) as u32,
        );
        // pass 3: scan — (value head, column block) grid, sequential over chunks inside
        let ks = self
            .be
            .kernel_sg("deltanet_scan", crate::gemm::deltanet_scan_spv(), 9, 20, 32);
        let mut p3 = [0u8; 20];
        p3[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p3[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        p3[8..12].copy_from_slice(&(nk as u32).to_ne_bytes());
        p3[12..16].copy_from_slice(&(kd as u32).to_ne_bytes());
        p3[16..20].copy_from_slice(&(vd as u32).to_ne_bytes());
        let n_blk = vd.div_ceil(8); // COLS=8, keep in sync with deltanet_scan.comp
        self.dispatch(
            ks,
            &[
                Self::vkb(v),
                Self::vkb(kn),
                Self::vkb(qn),
                Self::vkb(dk),
                Self::vkb(dq),
                Self::vkb(betag),
                Self::vkb(gg),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &p3,
            (nv * n_blk) as u32,
        );
    }

    /// Causal depthwise conv1d + SiLU, one token (qwen35 SSM input conv). The per-channel history
    /// `state` `[(kconv-1)*cc]` is updated in place; `out` is `[cc]`. See shaders/conv1d_silu.comp.
    pub fn conv1d_silu(
        &self,
        qkv: &dyn Buffer,
        w: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        cc: usize,
        kconv: usize,
    ) {
        let kern = self
            .be
            .kernel("conv1d_silu", crate::gemm::conv1d_silu_spv(), 4, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(cc as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(kconv as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[
                Self::vkb(qkv),
                Self::vkb(w),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &push,
            (cc as u32).div_ceil(256),
        );
    }

    /// BATCH depthwise conv1d + SiLU (rows ≥ kconv-1): pass 1 computes ALL rows·cc outputs in
    /// parallel from the virtual sequence [state ‖ qkv] (the sequential kernel walked the rows
    /// one by one, shuffling the history each token); pass 2 rebuilds the history from the last
    /// kconv-1 input rows. The recorder's hazard tracking orders pass 2 after pass 1 (pass 1
    /// reads the old state pass 2 overwrites). See shaders/conv1d_silu_par.comp / conv1d_shift.comp.
    pub fn conv1d_silu_batch(
        &self,
        qkv: &dyn Buffer,
        w: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        cc: usize,
        kconv: usize,
    ) {
        debug_assert!(rows >= kconv - 1, "conv1d_silu_batch needs rows ≥ kconv-1");
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(cc as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(kconv as u32).to_ne_bytes());
        let k1 = self
            .be
            .kernel("conv1d_silu_par", crate::gemm::conv1d_silu_par_spv(), 4, 12);
        self.dispatch(
            k1,
            &[
                Self::vkb(qkv),
                Self::vkb(w),
                Self::vkb(state),
                Self::vkb(out),
            ],
            1, // out only (state is read-only here)
            &push,
            ((rows * cc) as u32).div_ceil(256),
        );
        let k2 = self
            .be
            .kernel("conv1d_shift", crate::gemm::conv1d_shift_spv(), 2, 12);
        self.dispatch(
            k2,
            &[Self::vkb(qkv), Self::vkb(state)],
            1, // state out
            &push,
            (((kconv - 1) * cc) as u32).div_ceil(256),
        );
    }

    /// Batched strided row copy in ONE dispatch (word granularity): `rows` slices of `nw` u32
    /// words, `dst[dst_off + r*dst_stride + ..nw] = src[src_off + r*src_stride + ..nw]` (all in
    /// words). Replaces the per-row copy-command loop for Op::CopyStrided — at rows=512 that was
    /// 512 vkCmdCopyBuffer + hazard checks per split op, dwarfing the bytes moved.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_strided(
        &self,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        rows: usize,
        nw: usize,
        src_off: usize,
        src_stride: usize,
        dst_off: usize,
        dst_stride: usize,
    ) {
        let kern = self
            .be
            .kernel("copy_strided", crate::gemm::copy_strided_spv(), 2, 24);
        let mut push = [0u8; 24];
        for (i, v) in [rows, nw, src_off, src_stride, dst_off, dst_stride]
            .iter()
            .enumerate()
        {
            push[i * 4..i * 4 + 4].copy_from_slice(&(*v as u32).to_ne_bytes());
        }
        self.dispatch(
            kern,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            ((rows * nw) as u32).div_ceil(256),
        );
    }

    /// Elementwise sigmoid gate: `y[i] = a[i] * sigmoid(b[i])` (Qwen3-Next attention output gate).
    pub fn mul_sigmoid(&self, a: &dyn Buffer, b: &dyn Buffer, y: &dyn Buffer, n: usize) {
        let kern = self
            .be
            .kernel("mul_sigmoid", crate::gemm::mul_sigmoid_spv(), 3, 4);
        let mut push = [0u8; 4];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[Self::vkb(a), Self::vkb(b), Self::vkb(y)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// GeGLU with separate gate/up buffers: `y[i] = gelu(gate[i]) * up[up_off_bytes/4 + i]` (GELU
    /// tanh-approx). `up_off_bytes` lets a layer-major slice of a larger buffer be read in place
    /// (gemma4 per-layer-embd gate: `gelu(inp_gate·hidden) * inp_per_layer[il]`).
    pub fn gelu_mul_off(
        &self,
        gate: &dyn Buffer,
        up: &dyn Buffer,
        up_off_bytes: usize,
        y: &dyn Buffer,
        n: usize,
    ) {
        let k = self
            .be
            .kernel("gelu_mul", crate::gemm::gelu_mul_spv(), 3, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&((up_off_bytes / 4) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(gate), Self::vkb(up), Self::vkb(y)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Whether the id-indexed native GEMV (GPU-resident MoE routing) supports this expert format.
    pub fn native_id_supported(dtype: infr_core::DType) -> bool {
        crate::linear::native_id_kernel_name(dtype).is_some()
    }

    /// GPU MoE router top-k for `n_tokens` tokens (one workgroup per token): softmax-renormalized
    /// top-`n_used` over each token's `logits[n_expert]` → selected expert `ids` + `wts` (per token,
    /// `n_used` each), all in VRAM (no host round-trip). `scale` = routing scale.
    pub fn moe_topk(
        &self,
        logits: &dyn Buffer,
        ids: &dyn Buffer,
        wts: &dyn Buffer,
        n_tokens: usize,
        n_expert: usize,
        n_used: usize,
        scale: f32,
    ) {
        let k = self
            .be
            .kernel("moe_topk", crate::gemm::moe_topk_spv(), 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(n_expert as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_used as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&scale.to_ne_bytes());
        // ids is read-modify-write (exclusion scan); bind it as an output alongside wts.
        self.dispatch(
            k,
            &[Self::vkb(logits), Self::vkb(ids), Self::vkb(wts)],
            2,
            &push,
            n_tokens as u32,
        );
    }

    /// Greedy argmax over `[rows, n]` logits → token ids (u32) in `out_id[0..rows]`. Lets greedy
    /// decode (rows == 1) and the MTP verify accept (rows == m, issue #31) read back 4 bytes/row
    /// instead of the whole vocab logits.
    pub fn argmax(
        &self,
        logits: &dyn Buffer,
        part: &dyn Buffer,
        out_id: &dyn Buffer,
        n: usize,
        rows: usize,
    ) {
        // Two-stage (see argmax.comp): 256 slice partials in parallel across the GPU, then a
        // one-workgroup reduce. `part` = 512 f32 scratch (vals + idx bit-patterns), reused across
        // rows — the hazard tracker orders row r+1's partial pass after row r's reduce (WAR).
        // Rows loop sequentially (m is small, 2-7 for MTP); a single-workgroup whole-vocab scan
        // measured 18% SLOWER than the download it replaced, so each row keeps the full
        // 256-workgroup two-stage shape.
        let k1 = self
            .be
            .kernel("argmax_part", crate::gemm::argmax_part_spv(), 2, 8);
        let k2 = self.be.kernel("argmax", crate::gemm::argmax_spv(), 2, 8);
        for row in 0..rows {
            let mut p1 = [0u8; 8];
            p1[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
            p1[4..8].copy_from_slice(&((row * n) as u32).to_ne_bytes());
            self.dispatch(k1, &[Self::vkb(logits), Self::vkb(part)], 1, &p1, 256);
            let mut p2 = [0u8; 8];
            p2[0..4].copy_from_slice(&256u32.to_ne_bytes());
            p2[4..8].copy_from_slice(&(row as u32).to_ne_bytes());
            self.dispatch(k2, &[Self::vkb(part), Self::vkb(out_id)], 1, &p2, 1);
        }
    }

    /// Fused single-row argmax + softmax top-1 probability (`Op::ArgmaxProb`) — the MTP
    /// draft-loop accept (issue #33 follow-up to [`Recorder::argmax`]'s VERIFY-side fusion,
    /// de35727): `out_id[0]`/`out_prob[0]` read back 8 bytes total instead of the `[n]` logits row
    /// plus the host argmax + `sum exp` scan it replaces (measured the dominant per-step draft
    /// cost, not the download itself — see `Op::ArgmaxProb`'s doc). Two-stage, the SAME
    /// 256-workgroup-partials -> 1-workgroup-merge shape as `argmax`, with an online-softmax
    /// `sum_exp` carried alongside the (max, idx) reduction (see argmax_prob.comp). Single row
    /// only — the draft loop self-chains one token at a time.
    pub fn argmax_prob(
        &self,
        logits: &dyn Buffer,
        part: &dyn Buffer,
        out_id: &dyn Buffer,
        out_prob: &dyn Buffer,
        n: usize,
    ) {
        let k1 = self.be.kernel(
            "argmax_prob_part",
            crate::gemm::argmax_prob_part_spv(),
            2,
            4,
        );
        let k2 = self
            .be
            .kernel("argmax_prob", crate::gemm::argmax_prob_spv(), 3, 4);
        let p1 = (n as u32).to_ne_bytes();
        self.dispatch(k1, &[Self::vkb(logits), Self::vkb(part)], 1, &p1, 256);
        let p2 = 256u32.to_ne_bytes();
        self.dispatch(
            k2,
            &[Self::vkb(part), Self::vkb(out_id), Self::vkb(out_prob)],
            1,
            &p2,
            1,
        );
    }

    /// Emit an UNCONDITIONAL full compute→compute memory barrier. The chained decode replays one
    /// recording back-to-back in a single submission — commands from consecutive replays may
    /// otherwise overlap (the hazard tracker only orders dispatches WITHIN a recording), so a
    /// global barrier at the top of the recording fences iteration i's tail writes (sampled id,
    /// params) from iteration i+1's head reads.
    pub fn global_barrier(&self) {
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        unsafe {
            self.be.shared.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mb],
                &[],
                &[],
            );
        }
        *self.barriers.borrow_mut() += 1;
    }

    /// Log the sampled token id into the chained-decode ring: `ring[params[0] & 63] = id[0]`.
    /// Recorded LAST in the decode replay (after Argmax/Sample wrote `id`) — the host reads the
    /// whole chunk's ids from the ring after ONE n-replay submission.
    pub fn id_log(&self, params: &dyn Buffer, id: &dyn Buffer, ring: &dyn Buffer) {
        let k = self.be.kernel("id_log", crate::gemm::id_log_spv(), 3, 0);
        self.dispatch(
            k,
            &[Self::vkb(params), Self::vkb(id), Self::vkb(ring)],
            1,
            &[],
            1,
        );
    }

    /// Advance the decode-replay `params` SSBO on the device: `[pos, kv_len] -> [pos+1, pos+2]`.
    /// Recorded FIRST in the decode replay so the position stream never leaves the GPU.
    pub fn params_advance(&self, params: &dyn Buffer) {
        let k = self
            .be
            .kernel("params_advance", crate::gemm::params_advance_spv(), 1, 0);
        self.dispatch(k, &[Self::vkb(params)], 1, &[], 1);
    }

    /// Largest `top_k` the GPU stochastic sampler handles; above this the caller samples on the host.
    pub const SAMPLE_KMAX: usize = crate::gemm::SAMPLE_KMAX;

    /// Vocab-scale GPU stochastic sampling (`Op::Sample`): temperature + top-k + top-p over `n`
    /// logits → token id in `out_id[0]`, with the uniform draw read from the 1-float `u` BUFFER
    /// (record-once replayable — the host rewrites 4 bytes per token). Two-stage (see
    /// sample_topk.comp); `cand` = 2*256*top_k f32 scratch. Requires `2 <= top_k <= SAMPLE_KMAX`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_topk(
        &self,
        logits: &dyn Buffer,
        cand: &dyn Buffer,
        u: &dyn Buffer,
        out_id: &dyn Buffer,
        n: usize,
        top_k: usize,
        temp: f32,
        top_p: f32,
    ) {
        debug_assert!((2..=Self::SAMPLE_KMAX).contains(&top_k));
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(top_k as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&temp.to_ne_bytes());
        push[12..16].copy_from_slice(&top_p.to_ne_bytes());
        let k1 = self.be.kernel(
            "sample_topk_part",
            crate::gemm::sample_topk_part_spv(),
            2,
            16,
        );
        self.dispatch(k1, &[Self::vkb(logits), Self::vkb(cand)], 1, &push, 256);
        let mut push2 = [0u8; 16];
        push2[0..4].copy_from_slice(&(256u32 * top_k as u32).to_ne_bytes());
        push2[4..16].copy_from_slice(&push[4..16]);
        let k2 = self
            .be
            .kernel("sample_topk", crate::gemm::sample_topk_spv(), 3, 16);
        self.dispatch(
            k2,
            &[Self::vkb(cand), Self::vkb(u), Self::vkb(out_id)],
            1,
            &push2,
            1,
        );
    }

    /// Vocab-scale GPU stochastic sampling for the CHAINED/record-once decode replay (`Op::Sample`
    /// under `RopeMode::Dynamic`): identical to [`Recorder::sample_topk`], but stage 2 reads its
    /// uniform draw from the 64-slot `u` ring at `u_buf[params[0] & 63]` instead of `u_buf[0]` —
    /// the same ring geometry `id_log` keys its ring on, so ONE recording replays correctly whether
    /// the caller drives it singly (`execute`) or chained (`execute_chain`, N replays/submission),
    /// consuming the host RNG stream in lockstep with the per-token host sampler either way.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_topk_chain(
        &self,
        logits: &dyn Buffer,
        cand: &dyn Buffer,
        u: &dyn Buffer,
        params: &dyn Buffer,
        out_id: &dyn Buffer,
        n: usize,
        top_k: usize,
        temp: f32,
        top_p: f32,
    ) {
        debug_assert!((2..=Self::SAMPLE_KMAX).contains(&top_k));
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(top_k as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&temp.to_ne_bytes());
        push[12..16].copy_from_slice(&top_p.to_ne_bytes());
        let k1 = self.be.kernel(
            "sample_topk_part",
            crate::gemm::sample_topk_part_spv(),
            2,
            16,
        );
        self.dispatch(k1, &[Self::vkb(logits), Self::vkb(cand)], 1, &push, 256);
        let mut push2 = [0u8; 16];
        push2[0..4].copy_from_slice(&(256u32 * top_k as u32).to_ne_bytes());
        push2[4..16].copy_from_slice(&push[4..16]);
        // Binding order matches sample_topk.comp's CHAIN layout: cand(0), u_buf(1), params(2),
        // out_id(3) — `params` is a READ so it sits before the write (`dispatch`'s hazard tracker
        // treats the trailing `n_out` bound buffers as writes).
        let k2 = self.be.kernel(
            "sample_topk_chain",
            crate::gemm::sample_topk_chain_spv(),
            4,
            16,
        );
        self.dispatch(
            k2,
            &[
                Self::vkb(cand),
                Self::vkb(u),
                Self::vkb(params),
                Self::vkb(out_id),
            ],
            1,
            &push2,
            1,
        );
    }

    /// GPU stochastic sampling over `n` logits → token id in `out_id[0]`: temperature + top-k +
    /// top-p (nucleus) via a radix N-ary select, inverse-CDF sampled with the host-drawn uniform `u`.
    /// Requires `2 ≤ top_k ≤ SAMPLE_KMAX`. Only the token reads back — the vocab logits stay in VRAM.
    pub fn sample(
        &self,
        logits: &dyn Buffer,
        out_id: &dyn Buffer,
        n: usize,
        top_k: usize,
        temp: f32,
        top_p: f32,
        u: f32,
    ) {
        let k = self
            .be
            .kernel("moe_sample", crate::gemm::moe_sample_spv(), 2, 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(top_k as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&temp.to_ne_bytes());
        push[12..16].copy_from_slice(&top_p.to_ne_bytes());
        push[16..20].copy_from_slice(&u.to_ne_bytes());
        self.dispatch(k, &[Self::vkb(logits), Self::vkb(out_id)], 1, &push, 1);
    }

    /// DiffusionGemma perf slice 3 (docs/DIFFUSIONGEMMA.md): fused per-canvas-row entropy-bound
    /// sampler reduction — argmax/entropy/CDF-sample over `[rows, dim]` logits, one workgroup per
    /// row. `u` is `rows` host-drawn uniform `[0,1)` floats (the CDF-inversion target draw).
    /// Writes `argmax_out`/`entropy_out`/`sampled_out` (each `[rows]`) — only those tiny arrays
    /// need to leave the GPU, not the `[rows, dim]` logits themselves. See `dg_eb_sample.comp`.
    #[allow(clippy::too_many_arguments)]
    pub fn dg_eb_sample(
        &self,
        logits: &dyn Buffer,
        u: &dyn Buffer,
        argmax_out: &dyn Buffer,
        entropy_out: &dyn Buffer,
        sampled_out: &dyn Buffer,
        rows: usize,
        dim: usize,
        temp_inv: f32,
    ) {
        let k = self
            .be
            .kernel_sg("dg_eb_sample", crate::gemm::dg_eb_sample_spv(), 5, 8, 32);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&temp_inv.to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(logits),
                Self::vkb(u),
                Self::vkb(argmax_out),
                Self::vkb(entropy_out),
                Self::vkb(sampled_out),
            ],
            3,
            &push,
            rows as u32,
        );
    }

    /// Zero a buffer's first `n` 4-byte elements (cmd_fill_buffer) — clears the bucket counters.
    pub fn zero(&self, buf: &dyn Buffer, n: usize) {
        // Transfer chokepoint (see `copy`): previously an un-stamped blind spot.
        self.stamp("zero_fill");
        self.sync(&[], &[Self::vkb(buf)], true);
        unsafe {
            self.be
                .shared
                .device
                .cmd_fill_buffer(self.cmd, Self::vkb(buf), 0, (n * 4) as u64, 0);
        }
    }

    /// MoE bucketing pass 1 (count): tally assignments per expert into `counts` (pre-zeroed).
    pub fn moe_bucket_count(&self, tok_ids: &dyn Buffer, counts: &dyn Buffer, n_pairs: usize) {
        let k = self.be.kernel(
            "moe_bucket_count",
            crate::gemm::moe_bucket_count_spv(),
            2,
            4,
        );
        self.dispatch(
            k,
            &[Self::vkb(tok_ids), Self::vkb(counts)],
            1,
            &(n_pairs as u32).to_ne_bytes(),
            (n_pairs as u32).div_ceil(64),
        );
    }

    /// MoE bucketing pass 2 (scan): exclusive prefix sum `counts → offsets`, and reset `fill` to 0.
    pub fn moe_bucket_scan(
        &self,
        counts: &dyn Buffer,
        offsets: &dyn Buffer,
        fill: &dyn Buffer,
        n_expert: usize,
    ) {
        let k = self
            .be
            .kernel("moe_bucket_scan", crate::gemm::moe_bucket_scan_spv(), 3, 4);
        self.dispatch(
            k,
            &[Self::vkb(counts), Self::vkb(offsets), Self::vkb(fill)],
            2,
            &(n_expert as u32).to_ne_bytes(),
            1,
        );
    }

    /// MoE bucketing pass 3 (scatter): group token rows + weights by expert into `bucket_rows` /
    /// `bucket_wts` (each expert's run starts at `offsets[e]`). `dscale`, when given, is a
    /// per-expert weight (diffusion-gemma's `ffn_down_exps.scale`) baked into `bucket_wts` at
    /// scatter time — the scatter already has the expert id (`e`) in hand to index it, so this
    /// is a free multiply here vs. a separate post-GEMM pass, and `moe_scatter_reduce` needs no
    /// changes at all (it just sums already-scaled weights). Equivalent to the per-token path's
    /// `moe_accumulate_scaled` since the scale is linear in the down output.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_bucket_scatter(
        &self,
        tok_ids: &dyn Buffer,
        tok_wts: &dyn Buffer,
        offsets: &dyn Buffer,
        fill: &dyn Buffer,
        bucket_rows: &dyn Buffer,
        bucket_wts: &dyn Buffer,
        inv_pos: &dyn Buffer,
        dscale: Option<&dyn Buffer>,
        n_pairs: usize,
        n_used: usize,
    ) {
        let (name, spv): (_, _) = match dscale {
            Some(_) => (
                "moe_bucket_scatter_scaled",
                crate::gemm::moe_bucket_scatter_scaled_spv(),
            ),
            None => ("moe_bucket_scatter", crate::gemm::moe_bucket_scatter_spv()),
        };
        let nb = if dscale.is_some() { 8 } else { 7 };
        let k = self.be.kernel(name, spv, nb, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n_pairs as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_used as u32).to_ne_bytes());
        let mut bufs = vec![Self::vkb(tok_ids), Self::vkb(tok_wts), Self::vkb(offsets)];
        if let Some(ds) = dscale {
            bufs.push(Self::vkb(ds));
        }
        bufs.extend_from_slice(&[
            Self::vkb(fill),
            Self::vkb(bucket_rows),
            Self::vkb(bucket_wts),
            Self::vkb(inv_pos),
        ]);
        self.dispatch(k, &bufs, 4, &push, (n_pairs as u32).div_ceil(64));
    }

    /// Fused gather+quant for the batched MoE pipeline: quantize `n_slots` BUCKET rows (each
    /// reading its source token row via `bucket_rows`) straight into the packed expert-grouped
    /// layout — one dispatch replaces the per-expert gather and per-expert quant stages.
    #[allow(clippy::too_many_arguments)]
    pub fn quant_q8_gather(
        &self,
        a: &dyn Buffer,
        bucket_rows: &dyn Buffer,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        n_slots: usize,
        k_dim: usize,
    ) {
        let kq = self.be.kernel_sg(
            "quant_q8_gather",
            crate::gemm::quant_q8_gather_spv(),
            5,
            12,
            32,
        );
        let mut p = [0u8; 12];
        p[0..4].copy_from_slice(&(n_slots as u32).to_ne_bytes());
        p[4..8].copy_from_slice(&(k_dim as u32).to_ne_bytes());
        p[8..12].copy_from_slice(&32u32.to_ne_bytes());
        self.dispatch(
            kq,
            &[
                Self::vkb(a),
                Self::vkb(bucket_rows),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
            ],
            3,
            &p,
            (n_slots * (k_dim / 32)) as u32,
        );
    }

    /// ALL experts' mmq GEMMs in ONE dispatch (`gl_WorkGroupID.y` = expert): activation rows are
    /// packed by expert (bucket layout, segment e = offsets[e]..+counts[e]); the weight bank is
    /// indexed `w_base + e·stride`. Grid x covers the worst-case row tiles (`ceil(rows/BM)` — the
    /// whole chunk landing on one expert); tiles past a segment exit immediately, so the empty
    /// launches cost ~nothing while the dispatch count drops from ~n_expert·stages per layer to
    /// stages. `sact` is Q4_K/Q5_K's min-term row sums (None for Q6_K/Q8_0/Q5_0, which have no
    /// min).
    ///
    /// `rows` is a GRID-SIZING BOUND ONLY (never read by the shader — the kernel derives every
    /// real row range from `counts`/`offsets`): it must be `>=` the largest possible `counts[e]`
    /// over any expert `e`. Pass the CHUNK's TOKEN count (`x.numel()/ne`), not `n_pairs` (=
    /// `tokens·n_used`, the total packed-buffer length) — a top-k router never assigns a token to
    /// the same expert twice, so `counts[e] <= tokens` always (one assignment per token, at most),
    /// an `n_used`×-tighter bound than `n_pairs`. Passing `n_pairs` here is still CORRECT (a valid
    /// superset of row tiles, just with more early-exiting empty ones) but launches `n_used`× the
    /// necessary workgroups — caught during the diffusion-gemma perf campaign (slice 5): DG's
    /// caller was passing `n_pairs` (n_used=8 → 8× the row-tile grid) alongside qwen3moe's caller,
    /// which had the same bug. Fixed at both call sites in `adapter.rs`'s `Op::MoeFfn` lowering.
    ///
    /// `n_used`: NOT read by the shader — used HERE only to pick the row TILE size (BM). The
    /// default kernel tiles M in BM=64 chunks; at small rows-per-expert (Qwen3.6-MoE's 256-expert
    /// pool averages `rows·n_used/n_expert` ≈ 16/expert at pp512) that tile is mostly masked
    /// waste — a full 64-row tile launches, does the full BM×BN×K compute, but only ~16 of the 64
    /// rows are real (the rest are past the expert's segment and their results are discarded at
    /// the store). `rows·n_used/n_expert` (integer division; sum(counts) == rows·n_used == n_pairs
    /// spread evenly is the EXACT mean, not a bound) selects a BM=32 tile
    /// (`native_gemm_mmq_*_xp32`) below `MOE_EXPERT_SMALL_TILE_AVG_ROWS`, halving the average
    /// waste — measured via `moe_expert_grid_bound_bench`'s tile sweep (see its doc): BM=32 wins at
    /// avg~16 (qwen3.6-MoE), BM=64 (unchanged) still wins at avg~32+ (qwen3-30B-A3B, DG) where
    /// tiles are fuller and the smaller tile's extra per-workgroup fixed cost (barrier/staging,
    /// amortized over less work) stops paying for itself. Same K-accumulation order either way —
    /// this is tile GRANULARITY only, so results stay bit-identical to the BM=64 path.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mmq_experts(
        &self,
        dtype: infr_core::DType,
        stage: &'static str,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: Option<&dyn Buffer>,
        w: &dyn Buffer,
        w_base: usize,
        stride: usize,
        counts: &dyn Buffer,
        offsets: &dyn Buffer,
        c: &dyn Buffer,
        rows: usize,
        k: usize,
        n: usize,
        n_expert: usize,
        n_used: usize,
    ) {
        // NB: the profiler label is the CALLER'S stage (gate_up vs down), not the kernel name —
        // the same `native_gemm_mmq_*_xp` kernel serves both roles, so the auto kernel-name
        // label would merge them (and inferring the stage from dtype mislabeled DG's down-proj
        // dispatches as "expert_gateup"). Every caller knows its own role; trust it.
        self.label_next(stage);
        let avg_rows = rows.saturating_mul(n_used) / n_expert.max(1);
        let small_tile = avg_rows <= MOE_EXPERT_SMALL_TILE_AVG_ROWS;
        let bm: usize = if small_tile { 32 } else { 64 };
        let (name, spv, nb): (_, _, usize) = match (dtype, small_tile) {
            (infr_core::DType::Q4K, false) => (
                "native_gemm_mmq_q4k_xp",
                crate::gemm::native_gemm_mmq_q4k_xp_spv(),
                7,
            ),
            (infr_core::DType::Q4K, true) => (
                "native_gemm_mmq_q4k_xp32",
                crate::gemm::native_gemm_mmq_q4k_xp32_spv(),
                7,
            ),
            (infr_core::DType::Q6K, false) => (
                "native_gemm_mmq_q6k_xp",
                crate::gemm::native_gemm_mmq_q6k_xp_spv(),
                6,
            ),
            (infr_core::DType::Q6K, true) => (
                "native_gemm_mmq_q6k_xp32",
                crate::gemm::native_gemm_mmq_q6k_xp32_spv(),
                6,
            ),
            (infr_core::DType::Q8_0, false) => (
                "native_gemm_mmq_q8_0_xp",
                crate::gemm::native_gemm_mmq_q8_0_xp_spv(),
                6,
            ),
            (infr_core::DType::Q8_0, true) => (
                "native_gemm_mmq_q8_0_xp32",
                crate::gemm::native_gemm_mmq_q8_0_xp32_spv(),
                6,
            ),
            (infr_core::DType::Q5_0, false) => (
                "native_gemm_mmq_q5_0_xp",
                crate::gemm::native_gemm_mmq_q5_0_xp_spv(),
                6,
            ),
            (infr_core::DType::Q5_0, true) => (
                "native_gemm_mmq_q5_0_xp32",
                crate::gemm::native_gemm_mmq_q5_0_xp32_spv(),
                6,
            ),
            (infr_core::DType::Q5K, false) => (
                "native_gemm_mmq_q5k_xp",
                crate::gemm::native_gemm_mmq_q5k_xp_spv(),
                7,
            ),
            (infr_core::DType::Q5K, true) => (
                "native_gemm_mmq_q5k_xp32",
                crate::gemm::native_gemm_mmq_q5k_xp32_spv(),
                7,
            ),
            _ => unreachable!("batched MoE expert GEMM: Q4_K/Q5_K/Q6_K/Q8_0/Q5_0 only"),
        };
        let kern = self.be.kernel(name, spv, nb, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(stride as u32).to_ne_bytes()); // pc.m = expert stride
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let gx = (rows.div_ceil(bm) * (n / 64)) as u32;
        let mut bufs = vec![Self::vkb(qa), Self::vkb(dact)];
        if let Some(sa) = sact {
            bufs.push(Self::vkb(sa));
        }
        bufs.extend_from_slice(&[
            Self::vkb(w),
            Self::vkb(counts),
            Self::vkb(offsets),
            Self::vkb(c),
        ]);
        self.dispatch3(kern, &bufs, 1, &push, gx, n_expert as u32, 1);
    }

    /// Batched-MoE epilogue: `dst[t] = Σ_s bucket_wts[p]·y_all[p]` over the token's `n_used`
    /// assignments (p = inv_pos[t·n_used+s]) — fixed slot order, deterministic, no atomics, and
    /// dst is written directly (no zero + per-expert accumulate passes).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_scatter_reduce(
        &self,
        y_all: &dyn Buffer,
        bucket_wts: &dyn Buffer,
        inv_pos: &dyn Buffer,
        dst: &dyn Buffer,
        rows: usize,
        ne: usize,
        n_used: usize,
    ) {
        let k = self.be.kernel(
            "moe_scatter_reduce",
            crate::gemm::moe_scatter_reduce_spv(),
            4,
            12,
        );
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(n_used as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(y_all),
                Self::vkb(bucket_wts),
                Self::vkb(inv_pos),
                Self::vkb(dst),
            ],
            1,
            &push,
            ((rows * ne) as u32).div_ceil(64),
        );
    }

    /// Id-indexed native-block GEMV `y = x · W[ids[slot]]ᵀ` from a stacked expert tensor (element
    /// stride per expert). Lets GPU-resident MoE decode pick the expert from a GPU buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_id(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        ids: &dyn Buffer,
        slot: usize,
        stride: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.label_gemv("gemv_id", rows, in_f, out_f);
        let name = crate::linear::native_id_kernel_name(dtype).expect("native id kernel");
        let spv = crate::gemm::native_id_build_spv(dtype).expect("native id spv");
        let k = self.be.kernel(name, spv, 4, 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(slot as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(stride as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[Self::vkb(w), Self::vkb(x), Self::vkb(ids), Self::vkb(y)],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// Multi-slot id GEMV: all `n_used` experts in ONE dispatch → `y` is [n_used, out_f]. The experts
    /// run concurrently (no inter-expert barrier). `x_per_slot`: false → all slots read the same row
    /// `x` (gate/up); true → slot reads `x[slot*in_f..]` (down). Decode FFN fusion.
    ///
    /// `rows`: widens the SAME dispatch to `rows` independent tokens — `ids`/`wts`/`y` become
    /// `[rows, n_used, ...]` flat (the shader splits its flat `slot_global` index back into
    /// `row`/`slot` using `n_used`, so no push-constant change is needed here). `rows == 1` is the
    /// original decode call. This is the MoE small-m fast path (see `Op::MoeFfn` in adapter.rs):
    /// for a handful of prefill tokens it dispatches per-ACTIVE-expert GEMVs over only the routed
    /// rows, instead of the batched path's whole-expert-bank streaming GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_id_multi(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        ids: &dyn Buffer,
        n_used: usize,
        stride: usize,
        x: &dyn Buffer,
        x_per_slot: bool,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
        rows: usize,
    ) {
        // `rows` (push[20..24]) is used by the shader ONLY to bound-check the flat workgroup index
        // against the padding workgroups a possibly-2-D dispatch grid can add (see
        // native_gemv_id_multi.comp/native_gemv_id_multi_sg.comp's main()) — the addressing math
        // itself never needed the row count, only `n_used` (see the shaders' doc comments).
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(n_used as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(stride as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(x_per_slot as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(rows as u32).to_ne_bytes());
        let bufs = [Self::vkb(w), Self::vkb(x), Self::vkb(ids), Self::vkb(y)];
        // Reassociation-tolerant subgroup route (m=1 decode, Q6_K/Q5_K expert down-projection band) —
        // takes precedence over the tree id kernel where it saturates DRAM better. Decode only
        // (rows == 1); the small-m prefill widening stays on the bit-consistent tree path.
        if rows == 1 {
            if let Some(nr) = native_id_sg_choice(dtype, in_f, out_f) {
                if let Some((name, spv)) = crate::gemm::native_idm_sg_build_spv(dtype, nr) {
                    let k = self.be.kernel_sg(name, spv, 4, 24, 32);
                    // grid = n_used slots × ceil(out_f/NR) output-groups per slot (the shader splits
                    // its flat workgroup id back into slot / o0 using this same per-slot group count).
                    let groups = (n_used * out_f.div_ceil(nr as usize)) as u32;
                    self.dispatch_wide(k, &bufs, 1, &push, groups);
                    return;
                }
            }
        }
        let name = crate::linear::native_idm_kernel_name(dtype).expect("native idm kernel");
        let spv = crate::gemm::native_idm_build_spv(dtype).expect("native idm spv");
        let k = self.be.kernel(name, spv, 4, 24);
        self.dispatch_wide(k, &bufs, 1, &push, (rows * n_used * out_f) as u32);
    }

    /// Quantize f32 activations `a` [m,k] → int8 `qa` [m,k] + per-32-block f16 `dact`/`sact`
    /// ([m, k/32]) for the dp4a mmq matmul. (Pass 1 of mmq, reusable standalone.)
    pub fn quant_q8(
        &self,
        a: &dyn Buffer,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        m: usize,
        k: usize,
    ) {
        let kq = self
            .be
            .kernel_sg("quant_q8", crate::gemm::quant_q8_spv(), 4, 12, 32);
        let mut p = [0u8; 12];
        p[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        p[4..8].copy_from_slice(&(k as u32).to_ne_bytes());
        p[8..12].copy_from_slice(&32u32.to_ne_bytes());
        self.dispatch(
            kq,
            &[
                Self::vkb(a),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
            ],
            3,
            &p,
            (m * (k / 32)) as u32,
        );
    }

    /// Multi-slot id-indexed Q4_K dp4a (mmq) GEMV: like `linear_native_id_multi` but using hardware
    /// int8 dot-product against pre-quantized activations (`qa`/`dact`/`sact` from `quant_q8`, shared
    /// across slots). Q4_K weights only. `y` is [n_used, out_f].
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv_id_multi_q4k(
        &self,
        w: &dyn Buffer,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        ids: &dyn Buffer,
        n_used: usize,
        stride: usize,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        let k = self.be.kernel(
            "native_mmv_id_q4k",
            crate::gemm::native_mmv_id_q4k_spv(),
            6,
            16,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(n_used as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(stride as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[
                Self::vkb(w),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(ids),
                Self::vkb(y),
            ],
            1,
            &push,
            (n_used * out_f) as u32,
        );
    }

    /// Weighted accumulate of all selected experts' down outputs into hidden:
    /// `hidden[row*ne+i] += Σ_slot wts[row*n_used+slot] * down[(row*n_used+slot)*ne + i]`. Folds the
    /// per-expert axpys into one op. `rows` widens this to independent tokens via grid.y (the MoE
    /// small-m fast path — see `Op::MoeFfn` in adapter.rs); `rows == 1` is the original decode call.
    pub fn moe_accumulate(
        &self,
        down: &dyn Buffer,
        wts: &dyn Buffer,
        hidden: &dyn Buffer,
        ne: usize,
        n_used: usize,
        rows: usize,
    ) {
        let k = self
            .be
            .kernel("moe_accumulate", crate::gemm::moe_accumulate_spv(), 3, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_used as u32).to_ne_bytes());
        self.dispatch3(
            k,
            &[Self::vkb(down), Self::vkb(wts), Self::vkb(hidden)],
            1,
            &push,
            (ne as u32).div_ceil(64),
            rows as u32,
            1,
        );
    }

    /// `acc += wts[slot] * x` (indexed axpy) — the scale is read from a GPU buffer (the on-GPU router
    /// weights), so the weighted MoE expert accumulate needs no host scale.
    pub fn add_scaled_id(
        &self,
        x: &dyn Buffer,
        wts: &dyn Buffer,
        slot: usize,
        acc: &dyn Buffer,
        n: usize,
    ) {
        let k = self
            .be
            .kernel("add_scaled_id", crate::gemm::add_scaled_id_spv(), 3, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(slot as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(wts), Self::vkb(acc)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// `acc += scale * x` (axpy), in place into `acc`. Accumulates weighted MoE expert outputs into
    /// the resident hidden state on the GPU (chained across experts via WAW barriers on `acc`).
    /// In-place elementwise scalar multiply: `y[i] *= scale` for `i < n` (gemma4 layer output scale).
    pub fn scale(&self, y: &dyn Buffer, scale: f32, n: usize) {
        let k = self.be.kernel("scale", crate::gemm::scale_spv(), 1, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&scale.to_ne_bytes());
        self.dispatch(k, &[Self::vkb(y)], 1, &push, (n as u32).div_ceil(64));
    }

    /// Elementwise softcap `y[i] = cap·tanh(x[i]/cap)` (gemma final-logit / attn softcap). In-place
    /// safe — bind `x == y`.
    pub fn softcap(&self, x: &dyn Buffer, y: &dyn Buffer, cap: f32, n: usize) {
        let k = self.be.kernel("softcap", crate::gemm::softcap_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&cap.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    pub fn add_scaled(&self, x: &dyn Buffer, acc: &dyn Buffer, scale: f32, n: usize) {
        let k = self
            .be
            .kernel("add_scaled", crate::gemm::add_scaled_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&scale.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(acc)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Elementwise add; in place allowed (`a` may equal `y`).
    pub fn add(&self, a: &dyn Buffer, b: &dyn Buffer, y: &dyn Buffer, n: usize) {
        let k = self.be.kernel("add", crate::gemm::add_spv(), 3, 4);
        self.dispatch(
            k,
            &[Self::vkb(a), Self::vkb(b), Self::vkb(y)],
            1,
            &(n as u32).to_ne_bytes(),
            (n as u32).div_ceil(64),
        );
    }

    /// Broadcast bias add: `y[i] = x[i] + bias[i % n]` over `rows*n` elements (Qwen2 q/k/v `Wx+b`).
    pub fn add_bias(
        &self,
        x: &dyn Buffer,
        bias: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        n: usize,
    ) {
        let total = (rows * n) as u32;
        let k = self
            .be
            .kernel("add_bias", crate::gemm::add_bias_spv(), 3, 8);
        let mut pc = (n as u32).to_ne_bytes().to_vec();
        pc.extend_from_slice(&total.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(bias), Self::vkb(y)],
            1,
            &pc,
            total.div_ceil(64),
        );
    }

    /// Broadcast multiply: `dst[i] = x[i] * vec[i % n]` over `total = rows*n` elements
    /// (diffusion-gemma's router input scale — the multiplicative twin of `add_bias`).
    pub fn mul_vec(&self, x: &dyn Buffer, vec: &dyn Buffer, y: &dyn Buffer, rows: usize, n: usize) {
        let total = (rows * n) as u32;
        let k = self.be.kernel("mul_vec", crate::gemm::mul_vec_spv(), 3, 8);
        let mut pc = (n as u32).to_ne_bytes().to_vec();
        pc.extend_from_slice(&total.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(vec), Self::vkb(y)],
            1,
            &pc,
            total.div_ceil(64),
        );
    }

    /// qwen35moe shared-expert combine: `dst[r,c] = moe[r,c] + sigmoid(gate[r]) * shexp[r,c]` over
    /// `total = rows*n` elements — `gate` is a length-`rows` buffer (one raw pre-sigmoid logit per
    /// row), broadcast across each row's `n` columns (the row-broadcast twin of `mul_vec`'s
    /// column broadcast). See `Op::MoeSharedExpertAdd`'s doc.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_shared_expert_add(
        &self,
        moe: &dyn Buffer,
        shexp: &dyn Buffer,
        gate: &dyn Buffer,
        dst: &dyn Buffer,
        rows: usize,
        n: usize,
    ) {
        let total = (rows * n) as u32;
        let k = self.be.kernel(
            "moe_shared_expert_add",
            crate::gemm::moe_shared_expert_add_spv(),
            4,
            8,
        );
        let mut pc = (n as u32).to_ne_bytes().to_vec();
        pc.extend_from_slice(&total.to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(moe),
                Self::vkb(shexp),
                Self::vkb(gate),
                Self::vkb(dst),
            ],
            1,
            &pc,
            total.div_ceil(64),
        );
    }

    /// Like [`Self::moe_accumulate`], but scales each selected expert's down output by a per-expert
    /// weight BEFORE the weighted sum: `hidden[row*ne+i] += sum_slot wts[row*n_used+slot] *
    /// dscale[ids[row*n_used+slot]] * down[(row*n_used+slot)*ne+i]` (diffusion-gemma
    /// `ffn_down_exps.scale`). `ids` is the same expert-id buffer `moe_topk` filled. `rows` widens
    /// this to independent tokens via grid.y (the MoE small-m fast path); `rows == 1` is the
    /// original per-token call.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_accumulate_scaled(
        &self,
        down: &dyn Buffer,
        wts: &dyn Buffer,
        ids: &dyn Buffer,
        dscale: &dyn Buffer,
        hidden: &dyn Buffer,
        ne: usize,
        n_used: usize,
        rows: usize,
    ) {
        let k = self.be.kernel(
            "moe_accumulate_scaled",
            crate::gemm::moe_accumulate_scaled_spv(),
            5,
            8,
        );
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_used as u32).to_ne_bytes());
        self.dispatch3(
            k,
            &[
                Self::vkb(down),
                Self::vkb(wts),
                Self::vkb(ids),
                Self::vkb(dscale),
                Self::vkb(hidden),
            ],
            1,
            &push,
            (ne as u32).div_ceil(64),
            rows as u32,
            1,
        );
    }

    /// End recording, submit once, wait, and release transient objects.
    pub fn finish(self) -> Result<()> {
        let device = &self.be.shared.device;
        let t_record = self.t0.elapsed();
        if self.prof {
            eprintln!("[prof] barriers emitted = {}", self.barriers.borrow());
        }
        // Final timestamp so the last op has an interval to close.
        if self.prof2 {
            let idx = self.ts_labels.borrow().len() as u32;
            unsafe {
                device.cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    idx,
                );
            }
        }
        unsafe { device.end_command_buffer(self.cmd) }.map_err(|e| be(format!("end cmd: {e}")))?;
        let queue = self.be.shared.queue;
        unsafe {
            device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd))],
                    vk::Fence::null(),
                )
                .map_err(|e| be(format!("queue_submit: {e}")))?;
            device
                .queue_wait_idle(queue)
                .map_err(|e| be(format!("queue_wait_idle: {e}")))?;
            if self.prof {
                eprintln!(
                    "[prof] record={:.1}ms submit+gpu={:.1}ms",
                    t_record.as_secs_f64() * 1e3,
                    (self.t0.elapsed() - t_record).as_secs_f64() * 1e3,
                );
            }
            if self.prof2 {
                self.report_timestamps();
                device.destroy_query_pool(self.query_pool, None);
            }
            let cmd_pool = *self.be.shared.cmd_pool.lock().unwrap();
            device.free_command_buffers(cmd_pool, &[self.cmd]);
            for p in self.pools.borrow().iter() {
                device.destroy_descriptor_pool(*p, None);
            }
        }
        Ok(())
    }

    /// End recording WITHOUT submitting, returning a [`RecordedCmd`] the caller can replay across
    /// tokens (skipping per-token re-recording). Only meaningful for a `new_persistent` recorder; the
    /// descriptor sets bind the (persistent) decode buffers, so replays stay valid as long as those
    /// buffers live.
    pub fn finish_record(self) -> Result<RecordedCmd> {
        if self.prof {
            eprintln!("[prof] barriers emitted = {}", self.barriers.borrow());
        }
        unsafe { self.be.shared.device.end_command_buffer(self.cmd) }
            .map_err(|e| be(format!("end cmd: {e}")))?;
        Ok(RecordedCmd {
            shared: std::sync::Arc::clone(&self.be.shared),
            cmd: self.cmd,
            pools: self.pools.borrow().clone(),
        })
    }

    /// Read back per-dispatch timestamps and print GPU time aggregated by label (the kernel
    /// name, unless overridden via [`label_next`](Self::label_next) or the shape itemizers).
    /// Each label's total also feeds the process-wide aggregate (`infr_prof_rt::gpu_add`),
    /// printed once at exit — and merged with the host-side function table when the binary was
    /// ALSO built with `INFR_PROFILE=1`.
    fn report_timestamps(&self) {
        let labels = self.ts_labels.borrow();
        let n = labels.len();
        if n == 0 {
            return;
        }
        if self.ts_dropped.get() > 0 {
            eprintln!(
                "[prof2] WARNING: {} dispatches past the {}-timestamp query pool went unstamped",
                self.ts_dropped.get(),
                Self::MAX_TS,
            );
        }
        let mut ticks = vec![0u64; n + 1];
        unsafe {
            self.be
                .shared
                .device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .expect("get_query_pool_results");
        }
        let period = unsafe {
            self.be
                .shared
                .instance
                .get_physical_device_properties(self.be.shared.physical_device)
        }
        .limits
        .timestamp_period; // ns per tick
        use std::collections::BTreeMap;
        let mut by: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
        let mut total = 0f64;
        for i in 0..n {
            let us = (ticks[i + 1].wrapping_sub(ticks[i]) as f64) * period as f64 / 1000.0;
            let e = by.entry(labels[i]).or_insert((0.0, 0));
            e.0 += us;
            e.1 += 1;
            total += us;
        }
        eprintln!("[prof2] per-op GPU time (total {total:.0}us across {n} ops):");
        let mut rows: Vec<_> = by.into_iter().collect();
        rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        for (lbl, (us, cnt)) in rows {
            infr_prof_rt::gpu_add(lbl, us, cnt as u64);
            eprintln!(
                "[prof2]   {lbl:>14}  {us:8.0}us  ({cnt:3} ops, {:.1}us/op, {:.0}%)",
                us / cnt as f64,
                100.0 * us / total
            );
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanBackend {
    /// Start recording a single-submit forward.
    pub fn recorder(&self) -> Result<Recorder<'_>> {
        Recorder::new(self)
    }

    /// Start recording a resubmittable forward — `finish_record` returns a [`RecordedCmd`] to replay.
    pub fn recorder_persistent(&self) -> Result<Recorder<'_>> {
        Recorder::new_persistent(self)
    }
}

/// A pre-recorded, resubmittable command buffer (from [`Recorder::finish_record`]). Replaying it skips
/// per-token re-recording in the GPU-resident decode loop. Owns its command buffer + descriptor pool
/// (whose sets bind the persistent decode buffers); both are freed on drop after the GPU drains.
pub struct RecordedCmd {
    shared: std::sync::Arc<crate::VulkanShared>,
    cmd: vk::CommandBuffer,
    pools: Vec<vk::DescriptorPool>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl RecordedCmd {
    /// Resubmit the recorded command buffer `n` times in ONE queue submission (legal via
    /// SIMULTANEOUS_USE) and wait once — the chained decode's n back-to-back iterations. The
    /// recording's leading global barrier orders consecutive iterations.
    pub fn replay_n(&self, n: usize) -> Result<()> {
        let device = &self.shared.device;
        let cmds = vec![self.cmd; n];
        unsafe {
            device
                .queue_submit(
                    self.shared.queue,
                    &[vk::SubmitInfo::default().command_buffers(&cmds)],
                    vk::Fence::null(),
                )
                .map_err(|e| be(format!("replay_n submit: {e}")))?;
            device
                .queue_wait_idle(self.shared.queue)
                .map_err(|e| be(format!("replay_n wait: {e}")))?;
        }
        Ok(())
    }

    /// Resubmit the recorded command buffer and wait for completion.
    pub fn replay(&self) -> Result<()> {
        let device = &self.shared.device;
        unsafe {
            device
                .queue_submit(
                    self.shared.queue,
                    &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd))],
                    vk::Fence::null(),
                )
                .map_err(|e| be(format!("replay submit: {e}")))?;
            device
                .queue_wait_idle(self.shared.queue)
                .map_err(|e| be(format!("replay wait: {e}")))?;
        }
        Ok(())
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Drop for RecordedCmd {
    fn drop(&mut self) {
        let device = &self.shared.device;
        unsafe {
            let _ = device.queue_wait_idle(self.shared.queue);
            let cmd_pool = *self.shared.cmd_pool.lock().unwrap();
            device.free_command_buffers(cmd_pool, &[self.cmd]);
            for p in &self.pools {
                device.destroy_descriptor_pool(*p, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::{backend::BufferUsage, Backend};

    fn rmsnorm_cpu(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let scale = 1.0 / (ss + eps).sqrt();
        (0..n).map(|i| x[i] * scale * w[i]).collect()
    }
    fn dot(w: &[f32], row: usize, ne: usize, x: &[f32]) -> f32 {
        (0..ne).map(|k| w[row * ne + k] * x[k]).sum()
    }
    // ggml NORM-interleaved rope of a head-dim vector in place over the first rope_dim entries.
    fn rope_head(v: &mut [f32], hd: usize, rope_dim: usize, theta: f32, pos: usize) {
        for i in 0..rope_dim / 2 {
            let freq = theta.powf(-2.0 * i as f32 / rope_dim as f32);
            let ang = pos as f32 * freq;
            let (s, co) = (ang.sin(), ang.cos());
            let a = v[2 * i];
            let b = v[2 * i + 1];
            v[2 * i] = a * co - b * s;
            v[2 * i + 1] = a * s + b * co;
        }
        let _ = hd;
    }

    // round f32 → f16 → f32 (matches what the f16 q/k/v buffers store)
    fn r16(v: &[f32]) -> Vec<f32> {
        v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
    }
    // upload f32 values as an f16 buffer
    fn upf16(be: &VulkanBackend, v: &[f32]) -> Box<dyn Buffer> {
        let bits: Vec<u16> = v
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let b = be.alloc(bits.len() * 2, BufferUsage::Staging).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(&bits)).unwrap();
        b
    }

    // Reference attention. `window`>0 = sliding-window lower bound (gemma SWA); `scale_in`>0 overrides
    // the default 1/√hd (gemma4 = 1.0). Matches attention_kv.comp and attn_softmax.comp semantics.
    #[allow(clippy::too_many_arguments)]
    fn attn_kv_cpu(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        window: usize,
        scale_in: f32,
    ) -> Vec<f32> {
        let scale = if scale_in > 0.0 {
            scale_in
        } else {
            1.0 / (hd as f32).sqrt()
        };
        let mut o = vec![0f32; q_len * nh * hd];
        for ti in 0..q_len {
            let abs = pos_offset + ti;
            let lo = if window > 0 && abs + 1 > window {
                abs + 1 - window
            } else {
                0
            };
            for h in 0..nh {
                let kvh = h / (nh / nkv);
                let qb = (ti * nh + h) * hd;
                let mut sc = vec![0f32; abs + 1 - lo]; // valid keys [lo, abs]
                let mut mx = f32::NEG_INFINITY;
                for (jj, scj) in sc.iter_mut().enumerate() {
                    let kb = ((lo + jj) * nkv + kvh) * hd;
                    let d: f32 = (0..hd).map(|x| q[qb + x] * k[kb + x]).sum();
                    *scj = d * scale;
                    mx = mx.max(*scj);
                }
                let mut l = 0f32;
                for s in &sc {
                    l += (s - mx).exp();
                }
                let ob = (ti * nh + h) * hd;
                for (jj, s) in sc.iter().enumerate() {
                    let p = (s - mx).exp() / l;
                    let vb = ((lo + jj) * nkv + kvh) * hd;
                    for x in 0..hd {
                        o[ob + x] += p * v[vb + x];
                    }
                }
            }
        }
        o
    }

    fn run_attn_kv(q_len: usize, kv_len: usize, nh: usize, nkv: usize, hd: usize) {
        let be = VulkanBackend::new().unwrap();
        let pos_offset = kv_len - q_len; // new tokens are the last q_len of the cache
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        // q/k/v are f16 on the GPU; round the reference inputs to f16 too so the test isolates the
        // attention math (not f16 rounding).
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let bq = upf16(&be, &q);
        let bk = upf16(&be, &k);
        let bv = upf16(&be, &v);
        let bo = be
            .alloc(q_len * nh * hd * 4, BufferUsage::Readback)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_kv(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
            0,     // full causal (no sliding window)
            0.0,   // default 1/√hd scale
            false, // k f16
            false, // v f16
            0,     // cap (unused for f16)
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; q_len * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, 0, 0.0);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attention_kv q_len={q_len} kv_len={kv_len} max_err={err:e}");
        assert!(err < 5e-3, "attention_kv mismatch: {err}");
    }

    fn run_attn_kv_split(kv_len: usize, nh: usize, nkv: usize, hd: usize, scale: f32, win: usize) {
        let be = VulkanBackend::new().unwrap();
        let chunk = 512usize;
        let n_chunks = kv_len.div_ceil(chunk);
        let pos_offset = kv_len - 1; // decode: one new token at the end
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let bq = upf16(&be, &q);
        let bk = upf16(&be, &k);
        let bv = upf16(&be, &v);
        let bo = be.alloc(nh * hd * 4, BufferUsage::Readback).unwrap();
        let pm = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pl = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pacc = be
            .alloc(nh * n_chunks * hd * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_kv_split(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            1,          // rows (decode shape)
            kv_len - 1, // pos of the single query row
            kv_len,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            scale,
            win,
            None,  // canvas_lo: causal decode, not DiffusionGemma canvas
            false, // k f16
            false, // v f16
            0,     // cap (unused for f16)
            false, // batched: decode shape stays on the per-row grid
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, 1, nh, nkv, hd, pos_offset, win, scale);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn_kv_split kv_len={kv_len} n_chunks={n_chunks} win={win} max_err={err:e}");
        assert!(err < 5e-3, "split mismatch: {err}");
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attn_kv_split_dynac(
        kv_len: usize,
        cap: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        scale: f32,
        win: usize,
    ) {
        let be = VulkanBackend::new().unwrap();
        let chunk = cap.div_ceil(1024).max(64);
        let n_chunks = cap.div_ceil(chunk);
        let pos_offset = kv_len - 1;
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let bq = upf16(&be, &q);
        let bk = upf16(&be, &k);
        let bv = upf16(&be, &v);
        let bo = be.alloc(nh * hd * 4, BufferUsage::Readback).unwrap();
        let pm = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pl = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pacc = be
            .alloc(nh * n_chunks * hd * 4, BufferUsage::Activations)
            .unwrap();
        let params = be.alloc(8, BufferUsage::Activations).unwrap();
        be.upload(
            params.as_ref(),
            bytemuck::cast_slice(&[pos_offset as u32, kv_len as u32]),
        )
        .unwrap();
        let args = be.alloc(16, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.attn_live_prologue(params.as_ref(), args.as_ref(), nh, chunk, win);
        rec.attention_kv_split_dynac(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            params.as_ref(),
            args.as_ref(),
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            scale,
            win,
            false, // f16 KV cache
            0,     // cap (unused for f16)
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, 1, nh, nkv, hd, pos_offset, win, scale);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "attn_kv_split_dynac kv={kv_len} cap={cap} n_chunks={n_chunks} win={win} max_err={err:e}"
        );
        assert!(err < 5e-3, "dynac split mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_kv_split_dynac_matches_cpu() {
        run_attn_kv_split_dynac(20, 68, 16, 8, 128, 0.0, 0); // shallow: 1 live chunk, tiny cap
        run_attn_kv_split_dynac(600, 8065, 9, 3, 64, 0.0, 0); // wide bake, few live
        run_attn_kv_split_dynac(8000, 8065, 16, 8, 128, 0.0, 0); // deep: 32 live chunks
        run_attn_kv_split_dynac(130, 40960, 4, 2, 128, 0.0, 0); // huge cap (chunk floor rises)
                                                                // gemma-family replay shapes: SWA windows (span-chunked grid) + explicit scale.
        run_attn_kv_split_dynac(2050, 8065, 16, 8, 256, 0.0, 512); // gemma3 SWA deep (window << kv)
        run_attn_kv_split_dynac(300, 8065, 16, 8, 256, 0.0, 512); // SWA shallow (kv < window)
        run_attn_kv_split_dynac(700, 8065, 16, 8, 256, 0.0, 640); // window not chunk-aligned
        run_attn_kv_split_dynac(4000, 8065, 8, 2, 512, 1.0, 1024); // gemma4 hd=512, scale=1.0, SWA
        run_attn_kv_split_dynac(900, 8065, 8, 4, 256, 1.0, 0); // gemma4 full-attn (scale only)
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_kv_split_matches_cpu() {
        run_attn_kv_split(600, 9, 3, 64, 0.0, 0); // 2 chunks
        run_attn_kv_split(2050, 9, 3, 64, 0.0, 0); // 5 chunks, partial last
        run_attn_kv_split(8000, 4, 2, 32, 0.0, 0); // 16 chunks
        run_attn_kv_split(830, 16, 2, 256, 0.0, 0); // qwen35 full-attn decode (hd=256 general path)
        run_attn_kv_split(2050, 16, 8, 256, 0.0, 0); // gemma SWA-shape decode (hd=256, GQA 16:8)
        run_attn_kv_split(2050, 16, 8, 256, 0.0, 512); // SWA window (chunks below lo → empty)
        run_attn_kv_split(4000, 8, 2, 512, 1.0, 1024); // gemma4 hd=512, scale=1.0, SWA
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_kv_decode_matches_cpu() {
        // decode: 1 new token over a cache; 2500 exercises multi-tile flash (>TILE=1024)
        run_attn_kv(1, 200, 9, 3, 64);
        run_attn_kv(1, 13, 4, 2, 32);
        run_attn_kv(1, 1, 4, 2, 32);
        run_attn_kv(1, 2500, 9, 3, 64);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_kv_prefill_matches_cpu() {
        run_attn_kv(17, 17, 9, 3, 64);
        run_attn_kv(40, 1500, 9, 3, 64); // multi-tile prefill (kv_len > TILE)
                                         // gemma4: SWA layers (hd=256, GQA 16:8) and full layers (hd=512, GQA 16:1).
        run_attn_kv(17, 17, 16, 8, 256);
        run_attn_kv(17, 17, 16, 1, 512);
        run_attn_kv(1, 200, 16, 1, 512); // gemma4 full-layer decode
    }

    // upload f32 values as an f32 buffer (qk_norm_rope reads x / nw / freq_factors as f32).
    fn upf32(be: &VulkanBackend, v: &[f32]) -> Box<dyn Buffer> {
        let b = be.alloc(v.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
        b
    }

    /// CPU reference for fused per-head QK-norm (RMSNorm over hd) + NEOX RoPE (optional freq_factors).
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope_cpu(
        x: &[f32],
        nw: &[f32],
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        rope_pos: usize,
        out_base: usize,
        eps: f32,
        ff: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut y = vec![0f32; (out_base + rows) * nheads * hd];
        let hf = rope_dim / 2;
        for r in 0..rows {
            for h in 0..nheads {
                let ib = (r * nheads + h) * hd;
                let ss: f32 = (0..hd).map(|i| x[ib + i] * x[ib + i]).sum::<f32>() / hd as f32;
                let scale = 1.0 / (ss + eps).sqrt();
                let ob = ((out_base + r) * nheads + h) * hd;
                for p in 0..hf {
                    let (i0, i1) = (p, p + hf);
                    let a = x[ib + i0] * scale * nw[i0];
                    let b = x[ib + i1] * scale * nw[i1];
                    let freq = theta.powf(-2.0 * p as f32 / rope_dim as f32);
                    let mut ang = (rope_pos + r) as f32 * freq;
                    if let Some(ff) = ff {
                        ang /= ff[p];
                    }
                    let (s, c) = (ang.sin(), ang.cos());
                    y[ob + i0] = a * c - b * s;
                    y[ob + i1] = a * s + b * c;
                }
            }
        }
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn run_qk_norm_rope(
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        rope_pos: usize,
        out_base: usize,
        with_ff: bool,
    ) {
        let be = VulkanBackend::new().unwrap();
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let x = gen(rows * nheads * hd, 1);
        let nw: Vec<f32> = gen(hd, 2).iter().map(|v| v + 1.05).collect(); // ~gemma norm weights near 1
                                                                          // proportional-rope freq_factors: first quarter rotate (1.0), rest unrotated (1e30) — like gemma4
        let ff: Vec<f32> = (0..rope_dim / 2)
            .map(|p| if p < rope_dim / 4 { 1.0 } else { 1e30 })
            .collect();
        let bx = upf32(&be, &x);
        let bnw = upf32(&be, &nw);
        let bff = upf32(&be, &ff);
        let y_len = (out_base + rows) * nheads * hd;
        let by = be.alloc(y_len * 2, BufferUsage::Readback).unwrap(); // f16 out
        let rec = be.recorder().unwrap();
        rec.qk_norm_rope(
            bx.as_ref(),
            bnw.as_ref(),
            by.as_ref(),
            rows,
            nheads,
            hd,
            rope_dim,
            theta,
            rope_pos,
            out_base,
            1e-6,
            if with_ff { Some(bff.as_ref()) } else { None },
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; y_len * 2];
        be.download(by.as_ref(), &mut bytes).unwrap();
        let got: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let want = qk_norm_rope_cpu(
            &x,
            &nw,
            rows,
            nheads,
            hd,
            rope_dim,
            theta,
            rope_pos,
            out_base,
            1e-6,
            if with_ff { Some(&ff) } else { None },
        );
        // compare only the rows the kernel actually wrote (out_base..out_base+rows)
        let mut err = 0f32;
        for r in 0..rows {
            for h in 0..nheads {
                for i in 0..hd {
                    let idx = ((out_base + r) * nheads + h) * hd + i;
                    err = err.max((got[idx] - want[idx]).abs());
                }
            }
        }
        println!(
            "qk_norm_rope rows={rows} nheads={nheads} hd={hd} rope_dim={rope_dim} ff={with_ff} max_err={err:e}"
        );
        assert!(err < 1e-2, "qk_norm_rope mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn qk_norm_rope_matches_cpu() {
        run_qk_norm_rope(8, 4, 128, 128, 1e6, 0, 0, false); // qwen3 hd=128
        run_qk_norm_rope(8, 4, 256, 256, 1e4, 0, 0, false); // gemma3 hd=256
        run_qk_norm_rope(17, 16, 256, 256, 1e4, 5, 0, false); // gemma4 SWA Q (out_base=0)
        run_qk_norm_rope(17, 8, 256, 256, 1e4, 5, 5, false); // gemma4 SWA K (out_base=pos)
        run_qk_norm_rope(17, 16, 512, 512, 1e6, 5, 0, false); // gemma4 full Q
        run_qk_norm_rope(17, 1, 512, 512, 1e6, 5, 5, false); // gemma4 full K
        run_qk_norm_rope(17, 16, 512, 512, 1e6, 5, 0, true); // gemma4 full Q + freq_factors
        run_qk_norm_rope(17, 1, 512, 512, 1e6, 5, 5, true); // gemma4 full K + freq_factors
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_prefill_nonfa_matches_cpu() {
        // hd=128 (qwen3), full causal, default scale — the pre-existing regression cases.
        run_attn_prefill_nonfa(64, 64, 2, 1, 128, 0, 0.0);
        run_attn_prefill_nonfa(128, 200, 4, 2, 128, 0, 0.0);
        run_attn_prefill_nonfa(70, 70, 2, 2, 128, 0, 0.0);
        run_attn_prefill_nonfa(192, 500, 2, 1, 128, 0, 0.0);
        run_attn_prefill_nonfa(80, 300, 9, 3, 64, 0, 0.0);
        // gemma: hd=256 (SWA, GQA 16:8) and hd=512 (full, GQA 16:1), with the sliding-window mask
        // and the scale override (gemma4 = 1.0). These are the paths the new routing exercises.
        run_attn_prefill_nonfa(128, 300, 16, 8, 256, 100, 1.0); // gemma4 SWA (window)
        run_attn_prefill_nonfa(128, 300, 16, 1, 512, 0, 1.0); // gemma4 full (scale=1)
        run_attn_prefill_nonfa(128, 300, 16, 8, 256, 100, 0.0); // gemma3 SWA (window, 1/√hd)
        run_attn_prefill_nonfa(70, 400, 16, 8, 256, 64, 1.0); // SWA, non-64-aligned q
                                                              // force the split-K PV path (n_splits>1) and verify the partial-sum reduce is correct, incl.
                                                              // with a window (split reduce must respect the softmax mask) and hd=512.
        std::env::set_var("INFR_PV_SPLITS", "4");
        run_attn_prefill_nonfa(70, 300, 4, 2, 128, 0, 0.0);
        run_attn_prefill_nonfa(128, 500, 2, 1, 128, 0, 0.0);
        run_attn_prefill_nonfa(128, 3000, 16, 8, 256, 200, 1.0); // SWA, long kv, split-K
        run_attn_prefill_nonfa(128, 3000, 16, 1, 512, 0, 1.0); // full hd=512, long kv, split-K
        std::env::remove_var("INFR_PV_SPLITS");
    }

    fn run_attn_prefill_nonfa(
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        window: usize,
        scale: f32,
    ) {
        let be = VulkanBackend::new().unwrap();
        let pos_offset = kv_len - q_len;
        let mpad = q_len.div_ceil(64) * 64;
        let kv_pad = kv_len.div_ceil(256) * 256; // recorder pads kv to 256 (8-warp attn_qk BN)
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let mut qp = q.clone();
        qp.resize(mpad * nh * hd, 0.0);
        // K/V must cover the padded kv (the kernel reads padded rows; softmax masks them).
        let mut kp = k.clone();
        kp.resize((kv_pad + 64) * nkv * hd, 0.0);
        let mut vp = v.clone();
        vp.resize((kv_pad + 64) * nkv * hd, 0.0);
        let bq = upf16(&be, &qp);
        let bk = upf16(&be, &kp);
        let bv = upf16(&be, &vp);
        let bo = be.alloc(mpad * nh * hd * 4, BufferUsage::Readback).unwrap();
        let bs = be
            .alloc(nh * mpad * kv_pad * 2, BufferUsage::Activations)
            .unwrap();
        let bpv = be
            .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_prefill_nonfa(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            bs.as_ref(),
            bpv.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
            window,
            scale,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, window, scale);
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let err = got[..q_len * nh * hd]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn_prefill_nonfa q_len={q_len} kv_len={kv_len} max_err={err:e}");
        assert!(err < 5e-3, "attn_prefill_nonfa mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_prefill_flash_matches_cpu() {
        for &(q, kv, nh, nkv, hd) in &[
            (64usize, 64usize, 2usize, 1usize, 128usize),
            (128, 200, 4, 2, 128),
            (70, 70, 2, 2, 128),
            (192, 500, 2, 1, 128),
            (80, 300, 9, 3, 64),
            (448, 2000, 16, 8, 128), // qwen3-shaped, multi-block kv
        ] {
            run_attn_prefill_flash(q, kv, nh, nkv, hd);
        }
        // force the split-K flash path (partial+combine) and verify the merge is correct
        std::env::set_var("INFR_FLASH_SPLITS", "4");
        run_attn_prefill_flash(64, 2000, 16, 8, 128);
        run_attn_prefill_flash(128, 500, 2, 1, 128);
        std::env::remove_var("INFR_FLASH_SPLITS");
        // Force the bm=32 tile (otherwise only selected on sub-64 KB-shared devices like NVIDIA /
        // MoltenVK) so the small shaders get numeric-parity coverage on any GPU: the fused kernel
        // (hd=64), the warp split-K partial+combine (hd=128), and the non-warp partial
        // (INFR_NO_FLASH_WARP). Without this, a 64 KB device only ever exercises the bm=64 build.
        std::env::set_var("INFR_FLASH_BM", "32");
        run_attn_prefill_flash(80, 300, 9, 3, 64); // fused attn_flash_bm32
        run_attn_prefill_flash(128, 200, 4, 2, 128); // warp partial+combine (bm32)
        run_attn_prefill_flash(448, 2000, 16, 8, 128); // warp, multi-block kv
        std::env::set_var("INFR_NO_FLASH_WARP", "1");
        run_attn_prefill_flash(128, 500, 2, 1, 128); // non-warp attn_flash_partial_bm32
        std::env::remove_var("INFR_NO_FLASH_WARP");
        std::env::remove_var("INFR_FLASH_BM");
    }

    fn run_attn_prefill_flash(q_len: usize, kv_len: usize, nh: usize, nkv: usize, hd: usize) {
        let be = VulkanBackend::new().unwrap();
        let pos_offset = kv_len - q_len;
        let mpad = q_len.div_ceil(64) * 64;
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let mut qp = q.clone();
        qp.resize(mpad * nh * hd, 0.0);
        let mut kp = k.clone();
        kp.resize((kv_len + 64) * nkv * hd, 0.0);
        let mut vp = v.clone();
        vp.resize((kv_len + 64) * nkv * hd, 0.0);
        let bq = upf16(&be, &qp);
        let bk = upf16(&be, &kp);
        let bv = upf16(&be, &vp);
        let bo = be.alloc(mpad * nh * hd * 4, BufferUsage::Readback).unwrap();
        let po = be
            .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        let pmb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let plb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_prefill_flash(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            po.as_ref(),
            pmb.as_ref(),
            plb.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, 0, 0.0);
        let err = got[..q_len * nh * hd]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn_prefill_flash q_len={q_len} kv_len={kv_len} max_err={err:e}");
        assert!(err < 5e-3, "attn_prefill_flash mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_prefill_flash_reg_matches_cpu() {
        for &(q, kv, nh, nkv) in &[
            (128usize, 128usize, 4usize, 2usize),
            (128, 300, 2, 1),
            (200, 600, 8, 4),
            (448, 2000, 16, 8), // qwen3-shaped
            (100, 100, 2, 2),   // q<128 → padded tile
        ] {
            run_attn_flash_reg(q, kv, nh, nkv, 128);
        }
        std::env::set_var("INFR_FLASH_SPLITS", "4");
        run_attn_flash_reg(128, 2000, 16, 8, 128);
        run_attn_flash_reg(200, 600, 2, 1, 128);
        std::env::remove_var("INFR_FLASH_SPLITS");
    }

    fn run_attn_flash_reg(q_len: usize, kv_len: usize, nh: usize, nkv: usize, hd: usize) {
        let be = VulkanBackend::new().unwrap();
        let pos_offset = kv_len - q_len;
        let mpad = q_len.div_ceil(128) * 128;
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let mut qp = q.clone();
        qp.resize(mpad * nh * hd, 0.0);
        let mut kp = k.clone();
        kp.resize((kv_len + 128) * nkv * hd, 0.0);
        let mut vp = v.clone();
        vp.resize((kv_len + 128) * nkv * hd, 0.0);
        let bq = upf16(&be, &qp);
        let bk = upf16(&be, &kp);
        let bv = upf16(&be, &vp);
        let bo = be.alloc(mpad * nh * hd * 4, BufferUsage::Readback).unwrap();
        let po = be
            .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        let pmb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let plb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_prefill_flash_reg(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            po.as_ref(),
            pmb.as_ref(),
            plb.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, 0, 0.0);
        let err = got[..q_len * nh * hd]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn_flash_reg q_len={q_len} kv_len={kv_len} nh={nh} max_err={err:e}");
        assert!(err < 5e-3, "attn_flash_reg mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn matmul_proj_mmq_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (m, k, n) = (70usize, 64usize, 128usize); // m not %64 → padding; k=2 blocks of 32; n%64
        let nblk = k / 32;
        let mpad = m.div_ceil(64) * 64;
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();
        // u4 weights: q in 0..15, per-32-block f16 scale/min
        let qv: Vec<u32> = (0..n * k).map(|i| (i * 7 % 16) as u32).collect();
        let scales: Vec<u16> = (0..n * k / 32)
            .map(|b| half::f16::from_f32(0.015 + (b % 5) as f32 * 0.002).to_bits())
            .collect();
        let mins: Vec<u16> = (0..n * k / 32)
            .map(|b| half::f16::from_f32(-0.12 + (b % 3) as f32 * 0.03).to_bits())
            .collect();
        // pack u4: 8 nibbles per u32, weight g = col*k + kk
        let mut packed = vec![0u32; n * k / 8];
        for (g, &q) in qv.iter().enumerate() {
            packed[g / 8] |= q << (4 * (g % 8) as u32);
        }
        let dq = |g: usize| {
            half::f16::from_bits(scales[g / 32]).to_f32() * qv[g] as f32
                + half::f16::from_bits(mins[g / 32]).to_f32()
        };

        let upf = |v: &[f32]| {
            let b = be.alloc(v.len() * 4, BufferUsage::Staging).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let ba = upf(&a);
        let bwq = be
            .upload_weight_bytes(bytemuck::cast_slice(&packed))
            .unwrap();
        let bs = be
            .upload_weight_bytes(bytemuck::cast_slice(&scales))
            .unwrap();
        let bmn = be.upload_weight_bytes(bytemuck::cast_slice(&mins)).unwrap();
        let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        // scratch
        let qa = be.alloc(mpad * k, BufferUsage::Activations).unwrap();
        let dact = be.alloc(mpad * nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(mpad * nblk * 2, BufferUsage::Activations).unwrap();

        let rec = be.recorder().unwrap();
        rec.matmul_proj_mmq(
            ba.as_ref(),
            bwq.as_ref(),
            bs.as_ref(),
            bmn.as_ref(),
            bc.as_ref(),
            qa.as_ref(),
            dact.as_ref(),
            sact.as_ref(),
            m,
            k,
            n,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * n * 4];
        be.download(bc.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let mut e = 0f32;
        for r in 0..m {
            for col in 0..n {
                let want: f32 = (0..k).map(|x| a[r * k + x] * dq(col * k + x)).sum();
                e = e.max((got[r * n + col] - want).abs());
            }
        }
        println!("matmul_proj_mmq max_err={e:e}");
        assert!(e < 2e-2, "matmul_proj_mmq mismatch: {e}"); // int8 activation quant tolerance
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn native_gemm_mmq_q6k_matches_cpu() {
        // Q6_K dp4a GEMM vs a CPU reference: build a synthetic Q6_K weight (q6 ∈ 0..63, i8 sub-scale
        // per 16, f16 super-scale per 256), pack it into the 210-byte block layout, and check the
        // int8-dot GEMM matches the dequantized matmul. k must be a multiple of 256 (a Q6_K superblock)
        // so each column's k packs into whole blocks — exactly the real (down-proj) layout.
        let be = VulkanBackend::new().unwrap();
        let (m, k, n) = (70usize, 256usize, 128usize); // m not %64 → padding; k = 1 superblock; n%64
        let nblk = k / 32;
        let mpad = m.div_ceil(64) * 64;
        let nsb = k / 256; // superblocks per column
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();
        // q6 quants, per-16 i8 scales, per-256 f16 super-scale (weight g = col*k + kk)
        let q6: Vec<u32> = (0..n * k).map(|i| (i * 13 % 64) as u32).collect();
        let sc: Vec<i8> = (0..n * k / 16)
            .map(|b| ((b % 11) as i32 - 5) as i8)
            .collect();
        let d: Vec<half::f16> = (0..n * nsb)
            .map(|b| half::f16::from_f32(0.008 + (b % 5) as f32 * 0.001))
            .collect();
        // pack into 210-byte Q6_K blocks: [ql[128]][qh[64]][i8 scales[16]][f16 d]
        let mut blk = vec![0u8; n * nsb * 210];
        let sc_index = |g: usize| -> usize {
            // global i8-scale index for weight g, mirroring the shader's sc_idx within the superblock
            let p = g % 256;
            let (hf, ph) = (p / 128, p % 128);
            let (og, l) = (ph / 32, ph % 32);
            (g / 256) * 16 + hf * 8 + l / 16 + 2 * og
        };
        for (col, _) in (0..n).map(|c| (c, ())) {
            for sbk in 0..nsb {
                let base = (col * nsb + sbk) * 210;
                for p in 0..256usize {
                    let g = col * k + sbk * 256 + p;
                    let q = q6[g];
                    let (hf, ph) = (p / 128, p % 128);
                    let (og, l) = (ph / 32, ph % 32);
                    let lo = hf * 64;
                    let qh_byte = base + 128 + hf * 32 + l;
                    match og {
                        0 => blk[base + lo + l] |= (q & 0xF) as u8,
                        1 => blk[base + lo + l + 32] |= (q & 0xF) as u8,
                        2 => blk[base + lo + l] |= ((q & 0xF) << 4) as u8,
                        _ => blk[base + lo + l + 32] |= ((q & 0xF) << 4) as u8,
                    }
                    blk[qh_byte] |= (((q >> 4) & 3) << (2 * og)) as u8;
                }
                // i8 scales at +192, f16 d at +208
                for s in 0..16 {
                    blk[base + 192 + s] = sc[(col * nsb + sbk) * 16 + s] as u8;
                }
                let db = d[col * nsb + sbk].to_bits().to_le_bytes();
                blk[base + 208] = db[0];
                blk[base + 209] = db[1];
            }
        }
        let dq = |g: usize| -> f32 {
            d[g / 256].to_f32() * sc[sc_index(g)] as f32 * (q6[g] as f32 - 32.0)
        };

        let upf = |v: &[f32]| {
            let b = be.alloc(v.len() * 4, BufferUsage::Staging).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let ba = upf(&a);
        let bw = be.upload_weight_bytes(&blk).unwrap();
        let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        let qa = be.alloc(mpad * k, BufferUsage::Activations).unwrap();
        let dact = be.alloc(mpad * nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(mpad * nblk * 2, BufferUsage::Activations).unwrap();

        let rec = be.recorder().unwrap();
        rec.quant_q8(ba.as_ref(), qa.as_ref(), dact.as_ref(), sact.as_ref(), m, k);
        rec.matmul_mmq_q6k(
            qa.as_ref(),
            dact.as_ref(),
            bw.as_ref(),
            0,
            bc.as_ref(),
            m,
            k,
            n,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * n * 4];
        be.download(bc.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let mut e = 0f32;
        for r in 0..m {
            for col in 0..n {
                let want: f32 = (0..k).map(|x| a[r * k + x] * dq(col * k + x)).sum();
                e = e.max((got[r * n + col] - want).abs());
            }
        }
        println!("native_gemm_mmq_q6k max_err={e:e}");
        assert!(e < 2e-2, "native_gemm_mmq_q6k mismatch: {e}"); // int8 activation quant tolerance
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn matmul_proj_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (m, k, n) = (800usize, 64usize, 256usize); // m≥768 & not %64 → warp path + padding; n%256
        let mpad = m.div_ceil(64) * 64;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        // weight W[N,K] (row-major [out,in]), f16-rounded
        let w: Vec<f32> = (0..n * k)
            .map(|i| half::f16::from_f32(((i * 13 % 23) as f32 - 11.0) * 0.02).to_f32())
            .collect();
        let upf = |v: &[f32]| {
            let b = be.alloc(v.len() * 4, BufferUsage::Staging).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let ba = upf(&a);
        let dummy = be.alloc(4, BufferUsage::Activations).unwrap();
        let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        let cpu = |label: &str, c: &[f32]| {
            let mut e = 0f32;
            for r in 0..m {
                for col in 0..n {
                    let want: f32 = (0..k).map(|x| a[r * k + x] * w[col * k + x]).sum();
                    e = e.max((c[r * n + col] - want).abs());
                }
            }
            println!("matmul_proj {label} max_err={e:e}");
            assert!(e < 5e-3, "matmul_proj {label} mismatch: {e}");
        };

        // --- f16 weights (bits=16) ---
        let wf16: Vec<u16> = w
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let bw = be.upload_weight_bytes(bytemuck::cast_slice(&wf16)).unwrap();
        let rec = be.recorder().unwrap();
        rec.matmul_proj(
            ba.as_ref(),
            bw.as_ref(),
            dummy.as_ref(),
            dummy.as_ref(),
            bc.as_ref(),
            m,
            k,
            n,
            16,
            0,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * n * 4];
        be.download(bc.as_ref(), &mut bytes).unwrap();
        cpu("f16", bytemuck::cast_slice(&bytes));

        // --- quant weights (bits=8, per-16 scale/min) ---
        let blk = 16usize;
        let mut qu = vec![0u32; n * k / 4];
        let scales: Vec<u16> = (0..n * k / blk)
            .map(|_b| half::f16::from_f32(0.02).to_bits())
            .collect();
        let mins: Vec<u16> = (0..n * k / blk)
            .map(|_b| half::f16::from_f32(-1.5).to_bits())
            .collect();
        // choose u8 so that scale*u8+min == f16-rounded w (approx): u8 = round((w-min)/scale)
        let mut wq_ref = vec![0f32; n * k];
        for g in 0..n * k {
            let s = half::f16::from_bits(scales[g / blk]).to_f32();
            let mn = half::f16::from_bits(mins[g / blk]).to_f32();
            let q = (((w[g] - mn) / s).round().clamp(0.0, 255.0)) as u8;
            qu[g / 4] |= (q as u32) << (8 * (g % 4));
            wq_ref[g] = s * q as f32 + mn;
        }
        let bwq = be.upload_weight_bytes(bytemuck::cast_slice(&qu)).unwrap();
        let bs = be
            .upload_weight_bytes(bytemuck::cast_slice(&scales))
            .unwrap();
        let bm = be.upload_weight_bytes(bytemuck::cast_slice(&mins)).unwrap();
        let bc2 = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.matmul_proj(
            ba.as_ref(),
            bwq.as_ref(),
            bs.as_ref(),
            bm.as_ref(),
            bc2.as_ref(),
            m,
            k,
            n,
            8,
            4,
        );
        rec.finish().unwrap();
        let mut bytes2 = vec![0u8; mpad * n * 4];
        be.download(bc2.as_ref(), &mut bytes2).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes2);
        let mut e = 0f32;
        for r in 0..m {
            for col in 0..n {
                let want: f32 = (0..k).map(|x| a[r * k + x] * wq_ref[col * k + x]).sum();
                e = e.max((got[r * n + col] - want).abs());
            }
        }
        println!("matmul_proj quant max_err={e:e}");
        assert!(e < 5e-3, "matmul_proj quant mismatch: {e}");
    }

    /// The F16-weight SPLIT-K warptile (DG slice-7: the SC soft-embedding GEMM's route — deep k,
    /// narrow n, m below the wide-warp gate) vs a host reference on the SAME f16-rounded weights.
    /// m=70 (not %64) exercises the padded-row store; splits=4 exercises the partial planes +
    /// fixed-order reduce; k=2048 gives each split multiple BK=64 stages.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn matmul_native_splitk_f16_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (m, k, n, splits) = (70usize, 2048usize, 256usize, 4usize);
        let mpad = m.div_ceil(64) * 64;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| half::f16::from_f32(((i * 13 % 23) as f32 - 11.0) * 0.02).to_f32())
            .collect();
        let wf16: Vec<u16> = w
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let ba = be.alloc(a.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(ba.as_ref(), bytemuck::cast_slice(&a)).unwrap();
        let bw = be.upload_weight_bytes(bytemuck::cast_slice(&wf16)).unwrap();
        let pk = be
            .alloc(splits * mpad * n * 4, BufferUsage::Activations)
            .unwrap();
        let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.matmul_native_splitk(
            infr_core::DType::F16,
            ba.as_ref(),
            bw.as_ref(),
            0,
            pk.as_ref(),
            bc.as_ref(),
            m,
            k,
            n,
            splits,
            false,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * n * 4];
        be.download(bc.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let mut e = 0f32;
        for r in 0..m {
            for col in 0..n {
                let want: f32 = (0..k).map(|x| a[r * k + x] * w[col * k + x]).sum();
                e = e.max((got[r * n + col] - want).abs());
            }
        }
        println!("matmul_native_splitk f16 max_err={e:e}");
        assert!(e < 5e-2, "matmul_native_splitk f16 mismatch: {e}"); // f16 A rounding at k=2048
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn linear_q_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (rows, in_f, out_f) = (2usize, 128usize, 5usize); // in_f % 16 == 0
        let numel = in_f * out_f;
        // unified quant: u8 quants + per-16-block f16 scale/min; dequant = scale*q + min
        let qu8: Vec<u8> = (0..numel).map(|i| (i * 7 % 64) as u8).collect();
        let scales: Vec<u16> = (0..numel / 16)
            .map(|b| half::f16::from_f32(0.01 + (b % 5) as f32 * 0.003).to_bits())
            .collect();
        let mins: Vec<u16> = (0..numel / 16)
            .map(|b| half::f16::from_f32(-0.2 + (b % 3) as f32 * 0.05).to_bits())
            .collect();
        let mut quants = vec![0u32; numel / 4];
        for (g, &q) in qu8.iter().enumerate() {
            quants[g / 4] |= (q as u32) << (8 * (g % 4));
        }
        let x: Vec<f32> = (0..rows * in_f)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
            .collect();

        let bq = be
            .upload_weight_bytes(bytemuck::cast_slice(&quants))
            .unwrap();
        let bs = be
            .upload_weight_bytes(bytemuck::cast_slice(&scales))
            .unwrap();
        let bm = be.upload_weight_bytes(bytemuck::cast_slice(&mins)).unwrap();
        let upx = be.alloc(x.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(upx.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let by = be.alloc(rows * out_f * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_q(
            bq.as_ref(),
            bs.as_ref(),
            bm.as_ref(),
            upx.as_ref(),
            by.as_ref(),
            rows,
            in_f,
            out_f,
            8,
            4,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; rows * out_f * 4];
        be.download(by.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);

        let dq = |g: usize| {
            half::f16::from_bits(scales[g / 16]).to_f32() * qu8[g] as f32
                + half::f16::from_bits(mins[g / 16]).to_f32()
        };
        let mut maxe = 0f32;
        for r in 0..rows {
            for o in 0..out_f {
                let want: f32 = (0..in_f).map(|i| dq(o * in_f + i) * x[r * in_f + i]).sum();
                maxe = maxe.max((got[r * out_f + o] - want).abs());
            }
        }
        println!("linear_q max_err = {maxe:e}");
        assert!(maxe < 1e-3, "linear_q mismatch: {maxe}");
    }

    // Covers the subgroup GEMV across BOTH quant widths and the residual variant at realistic
    // projection sizes (the original linear_q test only hit q8 / no-residual / tiny dims).
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn mul_mat_vec_q_all_variants() {
        let be = VulkanBackend::new().unwrap();
        // (4,5)=Q4_0/Q4_1/Q4K  (8,4)=Q6K  (8,5)=Q5_0/Q5_1/Q8_0  (4,4)=Q2K/Q3K
        for &(bits, blk_shift) in &[(4u32, 5u32), (8u32, 4u32), (8u32, 5u32), (4u32, 4u32)] {
            for &res in &[false, true] {
                let (rows, in_f, out_f) = (1usize, 1024usize, 1024usize);
                let numel = in_f * out_f;
                let block = 1usize << blk_shift;
                let nblk = numel / block;
                let qmax = if bits == 4 { 16usize } else { 256usize };
                let qv: Vec<u32> = (0..numel).map(|i| (i * 7 % qmax) as u32).collect();
                let scales: Vec<u16> = (0..nblk)
                    .map(|b| half::f16::from_f32(0.01 + (b % 5) as f32 * 0.003).to_bits())
                    .collect();
                let mins: Vec<u16> = (0..nblk)
                    .map(|b| half::f16::from_f32(-0.2 + (b % 3) as f32 * 0.05).to_bits())
                    .collect();
                // pack quants: q4 = 8 nibbles/u32, q8 = 4 bytes/u32
                let per = if bits == 4 { 8usize } else { 4usize };
                let shift = if bits == 4 { 4u32 } else { 8u32 };
                let mut quants = vec![0u32; numel / per];
                for (g, &q) in qv.iter().enumerate() {
                    quants[g / per] |= q << (shift * (g % per) as u32);
                }
                let x: Vec<f32> = (0..rows * in_f)
                    .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
                    .collect();
                let r: Vec<f32> = (0..rows * out_f).map(|i| (i % 7) as f32 * 0.1).collect();

                let bq = be
                    .upload_weight_bytes(bytemuck::cast_slice(&quants))
                    .unwrap();
                let bs = be
                    .upload_weight_bytes(bytemuck::cast_slice(&scales))
                    .unwrap();
                let bm = be.upload_weight_bytes(bytemuck::cast_slice(&mins)).unwrap();
                let upx = be.alloc(x.len() * 4, BufferUsage::Staging).unwrap();
                be.upload(upx.as_ref(), bytemuck::cast_slice(&x)).unwrap();
                let upr = be.alloc(r.len() * 4, BufferUsage::Staging).unwrap();
                be.upload(upr.as_ref(), bytemuck::cast_slice(&r)).unwrap();
                let by = be.alloc(rows * out_f * 4, BufferUsage::Readback).unwrap();
                let rec = be.recorder().unwrap();
                if res {
                    rec.linear_add_q(
                        bq.as_ref(),
                        bs.as_ref(),
                        bm.as_ref(),
                        upx.as_ref(),
                        upr.as_ref(),
                        by.as_ref(),
                        rows,
                        in_f,
                        out_f,
                        bits,
                        blk_shift,
                    );
                } else {
                    rec.linear_q(
                        bq.as_ref(),
                        bs.as_ref(),
                        bm.as_ref(),
                        upx.as_ref(),
                        by.as_ref(),
                        rows,
                        in_f,
                        out_f,
                        bits,
                        blk_shift,
                    );
                }
                rec.finish().unwrap();
                let mut bytes = vec![0u8; rows * out_f * 4];
                be.download(by.as_ref(), &mut bytes).unwrap();
                let got: &[f32] = bytemuck::cast_slice(&bytes);

                let dq = |g: usize| {
                    half::f16::from_bits(scales[g / block]).to_f32() * qv[g] as f32
                        + half::f16::from_bits(mins[g / block]).to_f32()
                };
                let mut maxe = 0f32;
                for ri in 0..rows {
                    for o in 0..out_f {
                        let mut want: f32 =
                            (0..in_f).map(|i| dq(o * in_f + i) * x[ri * in_f + i]).sum();
                        if res {
                            want += r[ri * out_f + o];
                        }
                        maxe = maxe.max((got[ri * out_f + o] - want).abs());
                    }
                }
                println!("mul_mat_vec_q bits={bits} res={res} max_err={maxe:e}");
                assert!(maxe < 5e-3, "bits={bits} res={res} mismatch: {maxe}");
            }
        }
    }
}
