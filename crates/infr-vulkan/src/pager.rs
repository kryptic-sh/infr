//! GPU-resident paged weight cache: wraps `infr_core::pager::Pager`'s host-side LRU bookkeeping
//! with a fixed-slot VRAM arena, a small host-writable/GPU-readable LUT buffer, and upload
//! machinery through a caller-supplied REUSED pinned staging buffer (validated by
//! `tests/bandwidth_probe.rs` — a fresh staging buffer per call roughly halves throughput; see
//! that test's `fresh` vs `combined` columns. On this box the device-copy phase itself is nearly
//! free — ReBAR puts the staging buffer in device-local host-visible VRAM, so the bottleneck is
//! the host memcpy into it, not the subsequent `vkCmdCopyBuffer`).
//!
//! # Design (block-agnostic core, MoE plugs in today)
//! [`GpuPager`] only knows about uniform `slot_bytes`-sized blocks keyed by an opaque
//! `infr_core::pager::BlockId` — it has no idea a block is "an expert". The MoE integration
//! (`infr-llama`'s seam / this crate's `adapter.rs`) packs a `BlockId` from `(layer, role,
//! expert_id)` and calls [`GpuPager::ensure_resident`] with that block's mmap'd tensor bytes
//! before dispatching the id-indexed GEMV/GEMM through the LUT hop (the `PAGED` branch in
//! `shaders/native_gemv_id.comp` / `native_gemv_id_multi.comp`: `slot = lut[ids[slot]]`, scaled
//! onto the arena's 64-bit device address as `arena_addr + slot * slot_bytes` — see the `lut_host`
//! field's doc and `shaders/native_weight_addr.glsl`). A FUTURE dense layer-streaming policy
//! (NOT implemented here — see the task doc) would reuse this exact struct with `BlockId =
//! layer_idx`, `slot_bytes` = one layer's weight size, and a schedule-driven (not LRU) `touch`
//! order (a dense decode visits layers in a fixed known order, so it can exact-prefetch layer
//! `l+1` while `l` runs) — nothing in the arena/LUT/upload core below assumes MoE or LRU.
//!
//! # LUT
//! The host keeps an `n_blocks`-entry mirror of per-block resident SLOT INDICES
//! (`infr_core::pager::NOT_RESIDENT` for an absent block). The paged EXECUTION path never reads a
//! live device LUT: each (layer, role) batch freezes its `n_expert`-entry window into the
//! session's append-only LUT tape ([`MoePagerSession::lut_window`]) at record time, so staging
//! for later layers can keep mutating the mirror while earlier recorded-but-in-flight segments
//! read a consistent view. The classic per-pager device LUT + [`GpuPager::flush_lut`] remain for
//! the standalone [`GpuPager::ensure_resident`] surface (parity tests / future non-MoE users).
//!
//! # Eviction policy
//! Classic LRU for recency-driven touches (decode's routed-only path) plus the scan-resistant
//! cold-end insertion (`infr_core::pager::Pager::touch_cold`) for the batched prefill's
//! full-set sweeps — see that method's doc for why plain LRU is pathological there. llama.cpp
//! issue #20757's SLRU-with-admission remains the documented upgrade if these thrash on an
//! adversarial pattern.
use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;

use infr_core::backend::{Buffer, BufferUsage};
use infr_core::error::Result;
use infr_core::pager::{BlockId, Pager, PagerStats, Resolution, NOT_RESIDENT};
use infr_core::Backend;

use super::{as_vk_buf, be, VulkanBackend};

/// Fixed-budget evictable VRAM cache of uniform `slot_bytes` blocks. See the module doc.
pub struct GpuPager {
    pager: Pager,
    slot_bytes: usize,
    /// Device-local arena: `n_slots * slot_bytes`, one contiguous `bufferDeviceAddress` buffer.
    /// Both the MoE pools and the dense-streaming pools allocate this and their paged/streamed
    /// kernels read it through a 64-bit pointer (see [`Self::arena_addr`]) — no
    /// `maxStorageBufferRange` cap.
    arena: Box<dyn Buffer>,
    /// This arena's 64-bit `VkDeviceAddress`. The MoE paged kernels compute a slot's byte address
    /// as `arena_addr + lut[...] * slot_bytes`; the dense streamed kernels take the resident
    /// slot's full base address `arena_addr + slot * slot_bytes` (host-computed in
    /// [`DensePagerSession::stage`]) directly as a push constant.
    arena_addr: u64,
    /// Host-visible LUT mirror (mutated in place, re-uploaded on change) + the device buffer it's
    /// pushed to. `n_blocks` entries, each the resident block's SLOT INDEX
    /// (`infr_core::pager::NOT_RESIDENT` for an absent block). The paged MoE kernels read this slot
    /// index (through the session's frozen tape window) and compute the slot's byte address as
    /// `arena_addr + uint64_t(slot) * slot_bytes` in 64-bit — the multiply that used to wrap u32 in
    /// element space (Scout: 41.9M elements/expert overflowed at slot ≥ ~102, the original
    /// coherent-but-wrong bug) is now done on the device address, so no arena size overflows it.
    /// The dense-streaming pool keeps this mirror coherent but never reads it (its dispatch bakes
    /// the slot into a weight element offset instead).
    lut_host: Vec<u32>,
    lut_dev: Box<dyn Buffer>,
    lut_dirty: bool,
}

impl GpuPager {
    /// `n_blocks`: total distinct `BlockId`s that can ever be named (the LUT's fixed size — for
    /// MoE, `n_paged_layers * n_roles * n_experts`). `n_slots`: the VRAM budget in blocks
    /// (`budget_bytes / slot_bytes`, computed by the caller from remaining VRAM — see the
    /// within-batch sizing note on `infr_core::pager::Pager::new`, which applies unchanged here).
    /// `slot_bytes`: one block's PADDED byte size (the largest block the model will ever page —
    /// MoE experts of one model are uniform per role, so this is exact, not a worst-case pad).
    /// Must be u32-aligned (`% 4 == 0`) — the arena is read back a word at a time (see
    /// `shaders/native_weight_addr.glsl`'s `arena_word`).
    ///
    /// The arena always allocates as a `bufferDeviceAddress` buffer, read through a 64-bit
    /// pointer, so it may be as large as VRAM allows — no `maxStorageBufferRange` cap (both the
    /// MoE pools and the dense-streaming pool have taken this path since `36bcbf5`).
    pub fn new(
        vk: &VulkanBackend,
        n_blocks: usize,
        n_slots: usize,
        slot_bytes: usize,
    ) -> Result<Self> {
        assert!(n_slots > 0, "GpuPager needs at least one slot");
        assert!(
            slot_bytes.is_multiple_of(4),
            "GpuPager slot_bytes must be u32-aligned (the arena is read as u32 words)"
        );
        // Pointer-addressed: no per-arena binding cap — a pool spans as much VRAM as the budget
        // allows (the alloc-time VRAM budget guard is the only backstop).
        let (arena, arena_addr) = vk.alloc_arena_bda(n_slots * slot_bytes)?;
        let lut_dev = vk.alloc_uninit(n_blocks.max(1) * 4, BufferUsage::Staging)?;
        let lut_host = vec![NOT_RESIDENT; n_blocks.max(1)];
        // Seed the device LUT with the same all-absent state (arena/LUT start coherent).
        vk.upload(lut_dev.as_ref(), bytemuck::cast_slice(&lut_host))?;
        Ok(Self {
            pager: Pager::new(n_slots),
            slot_bytes,
            arena,
            arena_addr,
            lut_host,
            lut_dev,
            lut_dirty: false,
        })
    }

    /// The arena's 64-bit `VkDeviceAddress`. The paged kernels take this as a push constant and
    /// add `lut_slot * slot_bytes` to reach an expert.
    pub fn arena_addr(&self) -> u64 {
        self.arena_addr
    }

    pub fn n_slots(&self) -> usize {
        self.pager.n_slots()
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    pub fn stats(&self) -> PagerStats {
        self.pager.stats()
    }

    pub fn arena_buffer(&self) -> &dyn Buffer {
        self.arena.as_ref()
    }

    pub fn lut_buffer(&self) -> &dyn Buffer {
        self.lut_dev.as_ref()
    }

    /// Already-resident check with NO mutation (for a caller that wants to decide whether it even
    /// needs `bytes` in hand before calling `ensure_resident` — e.g. skip a host dequant/gather on
    /// a hit).
    pub fn is_resident(&self, id: BlockId) -> bool {
        self.pager.slot_of(id).is_some()
    }

    /// [`Self::ensure_resident`]'s RECORDED twin: on a miss, memcpy `bytes` into the caller's
    /// staging ring at `ring_off` (a host-mapped write) and record the ring→arena slot copy
    /// through `rec` instead of submitting an immediate one-shot — the caller batches many
    /// misses (and whole layers of compute) into one submission. Contract: the ring region
    /// `[ring_off, ring_off + slot_bytes)` must stay untouched until that recording's submit
    /// completes (the adapter's fenced ring-half rotation enforces this). The HOST LUT mirror is
    /// updated exactly like `ensure_resident`; the device-visible copy is the caller's frozen
    /// tape window (see [`MoePagerSession::lut_window`]) — `flush_lut` is NOT required on this
    /// path. Returns the ring bytes consumed (0 on a hit).
    pub fn touch_staged(
        &mut self,
        rec: &crate::recorder::Recorder<'_>,
        ring: &dyn Buffer,
        ring_off: usize,
        id: BlockId,
        bytes: &[u8],
        scan: bool,
    ) -> Result<usize> {
        debug_assert_eq!(
            bytes.len(),
            self.slot_bytes,
            "block byte size must match the arena's slot size"
        );
        // `scan`: full-set sweep (batched prefill's touch-all) → the scan-resistant cold-end
        // policy; otherwise classic LRU (decode's routed-only touches). See
        // `infr_core::pager::Pager::touch_cold`.
        let resolution = if scan {
            self.pager.touch_cold(id)
        } else {
            self.pager.touch(id)
        };
        match resolution {
            Resolution::Hit { .. } => Ok(0),
            Resolution::Miss { slot, evicted } => {
                // Safety: `ring` was allocated by this same backend (session-owned Staging).
                let base = unsafe { as_vk_buf(ring) }
                    .mapped_ptr()
                    .ok_or_else(|| be("pager staging ring is not persistently mapped"))?;
                par_copy_to_mapped(bytes, unsafe { base.add(ring_off) });
                rec.copy(
                    ring,
                    ring_off,
                    self.arena.as_ref(),
                    slot as usize * self.slot_bytes,
                    self.slot_bytes,
                );
                if let Some(e) = evicted {
                    if let Some(v) = self.lut_host.get_mut(e as usize) {
                        *v = NOT_RESIDENT;
                    }
                }
                if let Some(v) = self.lut_host.get_mut(id as usize) {
                    // Slot index — the shader scales it onto the arena's 64-bit base address (see
                    // `lut_host`'s doc).
                    *v = slot;
                }
                self.lut_dirty = true;
                Ok(self.slot_bytes)
            }
        }
    }

    /// `n` host-mirror LUT words starting at block id `base` — the source a frozen tape window
    /// copies from (see [`MoePagerSession::lut_window`]).
    fn lut_words(&self, base: usize, n: usize) -> &[u32] {
        &self.lut_host[base..base + n]
    }

    /// [`Self::touch_staged`]'s DENSE-STREAMING twin: residency via the exact cyclic-sweep policy
    /// (`infr_core::pager::Pager::schedule` — dense layer order is deterministic, so every miss
    /// is known in advance and no LUT/readback machinery is involved) and the block's bytes given
    /// as SEGMENTS (a fused qkv/gate_up block keeps its component tensors' zero-copy mmap slices;
    /// materializing the concat would double the streamed model's host RAM). Returns
    /// `(slot, ring_bytes_consumed)` — 0 consumed on a hit; a miss memcpys the segments
    /// back-to-back into the ring at `ring_off` and records the ring→arena slot copy, exactly
    /// like `touch_staged` (same ring-region-lifetime contract). The segments' total may be up to
    /// `slot_bytes - 3` short of the slot (the stride is padded to the pool's block/word
    /// alignment); the pad tail is never read by a dispatch (every kernel read stays within the
    /// block's `numel`). The caller must have verified the current ring half fits `slot_bytes`
    /// BEFORE calling (a miss here always consumes a full slot stride of ring accounting).
    ///
    /// The host LUT mirror is kept coherent (eviction/insert) so a pager can't be silently
    /// half-adopted by a LUT-reading path later, but dense dispatch never reads it — the slot
    /// index returned here is baked into the dispatch's weight element offset instead.
    pub fn schedule_staged(
        &mut self,
        rec: &crate::recorder::Recorder<'_>,
        ring: &dyn Buffer,
        ring_off: usize,
        id: BlockId,
        segments: &[&[u8]],
    ) -> Result<(u32, usize)> {
        match self.pager.schedule(id) {
            Resolution::Hit { slot } => Ok((slot, 0)),
            Resolution::Miss { slot, evicted } => {
                let total: usize = segments.iter().map(|s| s.len()).sum();
                debug_assert!(
                    total <= self.slot_bytes,
                    "dense block bytes ({total}) exceed the pool's slot stride ({})",
                    self.slot_bytes
                );
                // Safety: `ring` was allocated by this same backend (session-owned Staging).
                let base = unsafe { as_vk_buf(ring) }
                    .mapped_ptr()
                    .ok_or_else(|| be("pager staging ring is not persistently mapped"))?;
                let mut off = ring_off;
                for s in segments {
                    par_copy_to_mapped(s, unsafe { base.add(off) });
                    off += s.len();
                }
                // Word-align the copy length (the ring pad bytes it may carry are never read —
                // see the fn doc); `total <= slot_bytes` and `slot_bytes % 4 == 0` keep it in
                // the slot.
                rec.copy(
                    ring,
                    ring_off,
                    self.arena.as_ref(),
                    slot as usize * self.slot_bytes,
                    total.next_multiple_of(4),
                );
                if let Some(e) = evicted {
                    if let Some(v) = self.lut_host.get_mut(e as usize) {
                        *v = NOT_RESIDENT;
                    }
                }
                if let Some(v) = self.lut_host.get_mut(id as usize) {
                    // Slot index (dense pool keeps this coherent but never reads it).
                    *v = slot;
                }
                self.lut_dirty = true;
                Ok((slot, self.slot_bytes))
            }
        }
    }

    /// Open a touch batch — see `infr_core::pager::Pager::begin_batch`. One batch = one
    /// (layer, role) residency resolution; blocks it touches are eviction-protected until the
    /// next batch opens.
    pub fn begin_batch(&mut self) {
        self.pager.begin_batch();
    }

    /// Ensure `id` is resident, uploading `bytes` (exactly `slot_bytes`) through `staging` if it's
    /// a miss. Updates the HOST lut mirror immediately; the device copy is deferred to
    /// [`flush_lut`](Self::flush_lut) so a caller resolving several ids for one batch (see
    /// `infr_core::pager`'s within-batch note, which applies here unchanged) pays for exactly one
    /// LUT upload per batch, not one per id.
    pub fn ensure_resident(
        &mut self,
        vk: &VulkanBackend,
        staging: &dyn Buffer,
        id: BlockId,
        bytes: &[u8],
    ) -> Result<u32> {
        debug_assert_eq!(
            bytes.len(),
            self.slot_bytes,
            "block byte size must match the arena's slot size"
        );
        match self.pager.touch(id) {
            Resolution::Hit { slot } => Ok(slot),
            Resolution::Miss { slot, evicted } => {
                vk.upload(staging, bytes)?;
                copy_into_slot(vk, staging, self.arena.as_ref(), slot, self.slot_bytes)?;
                if let Some(e) = evicted {
                    if let Some(v) = self.lut_host.get_mut(e as usize) {
                        *v = NOT_RESIDENT;
                    }
                }
                if let Some(v) = self.lut_host.get_mut(id as usize) {
                    // The LUT stores the resident slot INDEX; the shader scales it onto the arena's
                    // 64-bit base address — see `lut_host`'s doc.
                    *v = slot;
                }
                self.lut_dirty = true;
                Ok(slot)
            }
        }
    }

    /// Push the host LUT mirror to the device if anything changed since the last flush. Callers
    /// resolving a whole batch of ids must call this exactly once, AFTER every `ensure_resident`
    /// for that batch and BEFORE recording any dispatch that reads the LUT — the within-batch
    /// eviction-safety argument on `infr_core::pager::Pager` only holds if the LUT a dispatch
    /// reads reflects EVERY id that batch touched, not a partial prefix.
    pub fn flush_lut(&mut self, vk: &VulkanBackend) -> Result<()> {
        if self.lut_dirty {
            vk.upload(self.lut_dev.as_ref(), bytemuck::cast_slice(&self.lut_host))?;
            self.lut_dirty = false;
        }
        Ok(())
    }
}

/// Parallel memcpy of one expert's bytes into the mapped staging ring. The single-thread copy is
/// the staging bottleneck (the bandwidth probe's 22 GB/s is a hot-source best case; streaming
/// distinct experts out of a 37 GB page-cache-backed mmap into write-combined ReBAR runs well
/// below that) — chunked `copy_nonoverlapping` across the rayon pool recovers most of the
/// PCIe/DRAM headroom. 4 MiB chunks: big enough for streaming stores, small enough to spread a
/// 14-18 MB expert across several workers.
fn par_copy_to_mapped(src: &[u8], dst: *mut u8) {
    use rayon::prelude::*;
    const CHUNK: usize = 4 << 20;
    if src.len() <= CHUNK {
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        return;
    }
    let dst_addr = dst as usize; // Send-able; each chunk writes a disjoint range
    src.par_chunks(CHUNK).enumerate().for_each(|(i, c)| unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), (dst_addr + i * CHUNK) as *mut u8, c.len());
    });
}

/// Device-to-device copy of `len` bytes from `src[0..len]` into `dst[slot*len .. (slot+1)*len]` —
/// the pager's slot placement, which the shared `Backend::copy_buffer` can't express (it always
/// copies `[0, bytes)` on both sides). Internal to this crate: raw `ash` calls mirroring
/// `VulkanBackend::upload`'s device-copy branch exactly, just with a nonzero destination offset.
fn copy_into_slot(
    vk: &VulkanBackend,
    src: &dyn Buffer,
    dst: &dyn Buffer,
    slot: u32,
    len: usize,
) -> Result<()> {
    // Safety: every buffer this pager holds was allocated by this same `VulkanBackend`.
    let (s, d) = unsafe { (as_vk_buf(src), as_vk_buf(dst)) };
    let (sb, db) = (s.buffer, d.buffer);
    let dst_offset = slot as u64 * len as u64;
    let shared = Arc::clone(&vk.shared);
    vk.one_shot(move |cmd| unsafe {
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset,
            size: len as u64,
        };
        shared.device.cmd_copy_buffer(cmd, sb, db, &[region]);
    })
}

// ─── MoE expert-bank paging session (slice 2: wiring into the execution path) ─────────────────
//
// The pieces above are the block-agnostic host<->VRAM cache; everything below is the MoE-specific
// glue: one [`GpuPager`] POOL per (expert role, per-expert byte size) pair, a table mapping a
// bound weight BUFFER's identity to where its layer's expert bytes live in the mmap'd GGUF, and
// the one persistent staging buffer every pool's uploads share.
//
// Why (role, slot_bytes) pools and not one pager per role: the arena/LUT design requires every
// block sharing an arena to have the SAME byte size (fixed slot offsets + a word-base LUT), and
// the GEMV/GEMM kernels additionally assume the layer's dtype when decoding a slot's bytes. Two
// shapes break a naive per-role pager:
//   - MIXED-dtype roles: unsloth-dynamic (UD) quants bump a SUBSET of layers' banks to a wider
//     format for quality (gemma-4-MoE: down = Q5_1 on 29 layers + Q8_0 on 1; DiffusionGemma:
//     down = Q5_0/Q8_0 16/14; Qwen3.6-UD: down mixes Q4_K/Q6_K). Slot sizes differ per dtype, so
//     one arena can't hold both — but a pool PER byte-size can: each layer registers into the
//     pool matching its own per-expert byte size, and a dispatch only ever reads ids of ONE
//     layer (whose dtype it knows statically from the graph), so blocks of different dtypes that
//     happen to share a byte size may even share a pool safely.
//   - FUSED gate_up banks (gemma-4 MoE / DiffusionGemma `ffn_gate_up_exps`): a fused expert is
//     just a BIGGER uniform block ([ne, 2*n_ff_exp] instead of [ne, n_ff_exp]) — it pages under
//     `Role::Gate` with its own slot size, and the model simply has no `Role::Up` pool.
// Every pool shares the same GLOBAL block-id space (`layer_index * n_expert + local_id`), so the
// paged kernels' `lut[layer_base + expert]` hop is unchanged — a pool's LUT just holds
// NOT_RESIDENT for the layers that live in other pools (they are never asked for).
//
// Design note (see the task doc): `Op::MoeFfn` carries NO `paged` flag. A paged layer's graph is
// byte-for-byte the same shape as a resident one (same tensor roles, same op) — only the ACTUAL
// buffer bound at `gate_exps`/`up_exps`/`down_exps` differs (a tiny placeholder vs the full
// upload). Threading a per-layer paging flag through `generate_dense_backend` (~20 parameters, 16
// call sites shared by CPU/Vulkan/Metal) to recompute at every graph-build call is a much bigger,
// riskier diff than keying off the buffer ACTUALLY bound at execute time — which the adapter
// already has in hand via `Bindings`. So the placement decision lives entirely on this side: the
// seam registers each paged layer's source bytes once at weight-load time, keyed by the stable
// identity of the (tiny, otherwise-unread) placeholder buffer it bound in place of a real upload;
// `execute_static` looks up that identity when it meets a `MoeFfn` op, and only diverts to the
// segmented paged path on a hit. CPU and Metal never call any of this — zero changes there.
use std::sync::Mutex;

/// One paged expert role. A FUSED gate_up bank registers under `Gate` (see the module-section doc
/// above); a fused model simply has no `Up` sources. Roles with mixed per-expert byte sizes
/// across layers span several pools — the (role, slot_bytes) pair, not the role alone, names a
/// pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    Gate,
    Up,
    Down,
}

impl Role {
    fn name(self) -> &'static str {
        match self {
            Role::Gate => "gate",
            Role::Up => "up",
            Role::Down => "down",
        }
    }
}

/// Stable identity of a bound `&dyn Buffer` — a thin-pointer cast of the trait object's data
/// pointer, which Box/heap allocation guarantees stable for the buffer's whole lifetime (the
/// model's `SeamWeights::wbufs` never reallocates the Boxes themselves once loaded, only the Vec
/// that briefly held them during construction). Used to recognize "the SAME placeholder buffer
/// bound at this TensorId, across however many differently-shaped Graphs reuse it" without
/// depending on `TensorId` staying numerically stable across graphs (it doesn't — see the module
/// doc's design note).
pub fn buffer_identity(b: &dyn Buffer) -> usize {
    std::ptr::from_ref(b) as *const () as usize
}

/// Where one paged layer's whole per-role expert bank lives: a zero-copy view into the GGUF mmap
/// (kept alive via `Arc` — see `infr_gguf::TensorBytes`, which this trait object mirrors without
/// infr-vulkan taking a dependency on infr-gguf), plus the byte stride of ONE expert within it.
/// "expert e is the e-th equal-size contiguous slice" holds for every GGUF MoE bank in this
/// codebase (`Op::MoeFfn`'s doc), so `stride_bytes = bytes.len() / n_expert` locates any expert
/// with no quant-format-specific math.
pub struct ExpertSource {
    pub bytes: Arc<dyn AsRef<[u8]> + Send + Sync>,
    pub stride_bytes: usize,
    /// This layer's offset into the role's shared LUT/arena block-id space
    /// (`layer_index * n_expert`) — turns a per-layer LOCAL expert id (what the router/top-k
    /// produces, `0..n_expert`) into a GLOBAL `BlockId` unique across every paged layer of this
    /// role, so one `Pager`/LUT can hold experts from many layers at once.
    pub layer_base: u32,
}

/// One arena pool: every block in it shares `slot_bytes` (see the section doc above for why the
/// pool key is `(role, slot_bytes)`, not the role alone).
struct Pool {
    role: Role,
    slot_bytes: usize,
    pager: GpuPager,
}

/// One model's whole paged-MoE session: the `(role, slot_bytes)` arena pools + the shared
/// persistent staging buffer their uploads reuse (the bandwidth probe's headline finding — see
/// `pager.rs`'s module doc and `tests/bandwidth_probe.rs`). Lives on the `VulkanBackend` HANDLE
/// (NOT `VulkanShared` — the session's buffers hold `Arc<VulkanShared>` clones, and parking it on
/// the shared state made an Arc cycle that leaked the device's whole VRAM footprint until process
/// exit; see the `moe_pager` field doc in lib.rs) for as long as the backend that loaded the
/// paged model lives (`VulkanBackend::init_moe_pager`); `None` for every non-paged model — zero
/// cost, zero behavior change on the common (fits-in-VRAM) path.
pub struct MoePagerSession {
    pools: Vec<Pool>,
    /// `buffer_identity(placeholder)` -> (role, pool index, this layer's expert source), for
    /// every PAGED `_exps` tensor. A non-paged layer's gate/up/down buffer is never registered
    /// here — the adapter's lookup simply misses and falls through to the ordinary
    /// resident-weight path.
    sources: HashMap<usize, (Role, usize, ExpertSource)>,
    staging: Box<dyn Buffer>,
    /// Pinned staging RING for the recorded-upload path ([`GpuPager::touch_staged`]): two
    /// fence-rotated halves of [`Self::ring_half_bytes`] each, so the CPU stages the next
    /// segment's misses while the GPU executes the previous one (see
    /// `adapter::execute_paged_moe`'s rotation). Sized by [`MoePagerLayout::ring_bytes`].
    ring: Box<dyn Buffer>,
    ring_half_bytes: usize,
    /// LUT tape: an append-only run of frozen per-(layer, role) LUT windows (`n_expert` u32 slot
    /// indices each, written by [`Self::lut_window`]). Dispatches read `tape[window + local_id]`
    /// instead of the live pool LUT, so host-side staging for LATER layers can keep mutating the
    /// mirror while EARLIER layers' recorded-but-in-flight dispatches still read a consistent
    /// view — the in-flight-LUT rule that a single mutable device LUT cannot satisfy once
    /// several layers record into one submission. The cursor is the adapter's (reset only after
    /// a full drain).
    tape: Box<dyn Buffer>,
    tape_words: usize,
    print_stats: bool,
}

/// One pool's spec in [`MoePagerLayout`]: slot counts are INDEPENDENT per pool. Each pool's arena
/// is a `bufferDeviceAddress` buffer (`48ad9c1`) addressed by 64-bit pointer — no per-arena
/// `maxStorageBufferRange` ceiling — but per-pool sizing
/// still matters because of unequal per-expert sizes (Scout: gate/up 13.8 MB, down 18 MB): a
/// shared slot count is dragged down to fit the LARGEST pool's per-slot bytes within the VRAM
/// budget and strands budget the smaller pools could have used as real hit rate (Scout: uniform
/// 238 slots everywhere left ~6 GB of a 19 GB budget unused; per-pool sizing gives gate/up 312
/// each). Each pool has its own LRU/LUT and `touch_role` resolves pools independently, so unequal
/// counts are correctness-neutral — a pool with fewer slots just misses more often. Computed by
/// the caller (budget-driven count, then per-pool split — see `seam::mod`'s placement policy).
pub struct MoePoolSpec {
    pub role: Role,
    pub slot_bytes: usize,
    pub n_slots: usize,
}

/// Fixed layout for [`MoePagerSession::new`] — sizes every arena/LUT UP FRONT, before any tensor
/// is registered. This split (layout now, registration per tensor later) matters for sequencing:
/// the session must exist and answer `is_paged`/`Backend::moe_paged` truthy BEFORE the seam's
/// weight-load closure runs (so a paged tensor's placeholder buffer is recognized the very first
/// time the adapter executes a graph, not just after the whole model is loaded) — see
/// `infr-llama`'s `generate_dense_vulkan_session` for the call order this enables.
pub struct MoePagerLayout {
    /// Total distinct experts nameable per pool's LUT = `n_paged_layers * n_expert` — the GLOBAL
    /// id space every pool shares (a pool only ever resolves ids of the layers registered into
    /// it; other layers' entries stay `NOT_RESIDENT`).
    pub n_blocks: usize,
    pub pools: Vec<MoePoolSpec>,
    /// Total bytes for the pinned upload ring (two fence-rotated halves — see
    /// [`MoePagerSession`]'s `ring` field). `0` picks the default
    /// ([`default_ring_bytes`]); either way each half is floored at the largest pool slot so one
    /// miss always fits. The seam's budget math subtracts this before splitting arena shares.
    pub ring_bytes: usize,
}

/// Upload-ring sizing policy: `INFR_PAGER_RING` (shared size grammar) wins; otherwise an eighth
/// of the pager budget, clamped to [256 MiB, 2 GiB]. Bigger halves = fewer pipeline rotations,
/// and each rotation stalls the CPU on the other half's fence — measured on Scout pp512 (miss-
/// heavy steady state, ~22 GB staged/rep): 256 MiB → 224 t/s, 1 GiB → 324, 2 GiB → 404, flat
/// beyond. The budget fraction keeps small explicit `INFR_CACHE` runs from spending most of
/// their grant on staging instead of arena slots.
pub fn ring_bytes_policy(pager_budget: u64) -> usize {
    const MIB: u64 = 1024 * 1024;
    if let Some(b) = std::env::var("INFR_PAGER_RING")
        .ok()
        .and_then(|v| infr_core::parse_size(&v))
        .map(|s| s.resolve(0) as usize)
        .filter(|&b| b > 0)
    {
        return b;
    }
    (pager_budget / 8).clamp(256 * MIB, 2048 * MIB) as usize
}

impl MoePagerSession {
    pub fn new(vk: &VulkanBackend, layout: MoePagerLayout) -> Result<Self> {
        let mut pools = Vec::with_capacity(layout.pools.len());
        let mut staging_bytes = 4usize;
        for spec in &layout.pools {
            pools.push(Pool {
                role: spec.role,
                slot_bytes: spec.slot_bytes,
                // MoE pools are pointer-addressed (`bufferDeviceAddress`) — no per-arena SSBO cap.
                pager: GpuPager::new(vk, layout.n_blocks, spec.n_slots, spec.slot_bytes)?,
            });
            staging_bytes = staging_bytes.max(spec.slot_bytes);
        }
        let staging = vk.alloc_uninit(staging_bytes, BufferUsage::Staging)?;
        // Each ring half must hold the largest slot, or `touch_staged` could never make progress
        // on that pool (the adapter rotates halves when one fills; a slot bigger than a half
        // would fit in neither).
        let ring_total = if layout.ring_bytes > 0 {
            layout.ring_bytes
        } else {
            ring_bytes_policy(0) // 0 budget → the clamp floor (env override still wins)
        };
        let ring_half_bytes = (ring_total / 2).max(staging_bytes);
        let ring = vk.alloc_uninit(2 * ring_half_bytes, BufferUsage::Staging)?;
        // One graph's windows = paged layers x roles x n_expert words (Scout: 48 x 3 x 16 = 2.3k)
        // — 64k words (256 KiB) leaves an order of magnitude of headroom; `lut_window` hard-errors
        // on overflow rather than wrapping into a region an in-flight segment may still read.
        let tape_words = 64 * 1024;
        let tape = vk.alloc_uninit(tape_words * 4, BufferUsage::Staging)?;
        Ok(Self {
            pools,
            sources: HashMap::new(),
            staging,
            ring,
            ring_half_bytes,
            tape,
            tape_words,
            print_stats: std::env::var("INFR_PAGER_STATS").is_ok(),
        })
    }

    /// Register one paged layer's `role` tensor — called from the seam's weight-load closure
    /// (once per paged `_exps` tensor) instead of uploading it. `buf_id` is the placeholder
    /// buffer's identity (see [`buffer_identity`]); `source` is where its bytes actually live.
    /// The pool is picked by `(role, source.stride_bytes)` — errors if the layout has no matching
    /// pool (a seam sizing bug: the layout enumeration and this registration must derive the slot
    /// size from the same tensor bytes).
    pub fn register(&mut self, role: Role, buf_id: usize, source: ExpertSource) -> Result<()> {
        let pool = self
            .pools
            .iter()
            .position(|p| p.role == role && p.slot_bytes == source.stride_bytes)
            .ok_or_else(|| {
                be(format!(
                    "moe pager: no ({:?}, {} B/expert) pool in the layout for this tensor",
                    role, source.stride_bytes,
                ))
            })?;
        self.sources.insert(buf_id, (role, pool, source));
        Ok(())
    }

    /// Whether `buf_id` (see [`buffer_identity`]) is a registered paged tensor of `role` — the
    /// adapter's per-`MoeFfn` dispatch check.
    pub fn is_paged(&self, role: Role, buf_id: usize) -> bool {
        self.sources.get(&buf_id).is_some_and(|(r, ..)| *r == role)
    }

    /// Resolve residency for every id in `local_ids` (this token's routed experts, LOCAL to the
    /// layer) against `buf_id`'s pool, uploading misses through the shared staging buffer and
    /// flushing the LUT once. Returns the GLOBAL ids (`layer_base + local_id`) the paged GEMV
    /// must read instead of `local_ids` — see [`ExpertSource::layer_base`].
    pub fn touch_role(
        &mut self,
        vk: &VulkanBackend,
        role: Role,
        buf_id: usize,
        local_ids: &[u32],
    ) -> Result<Vec<u32>> {
        let (r, pool, src) = self
            .sources
            .get(&buf_id)
            .ok_or_else(|| be("moe pager: touch on an unregistered buffer"))?;
        debug_assert_eq!(*r, role, "touch_role: role/buffer mismatch");
        let pager = &mut self.pools[*pool].pager;
        let stride = src.stride_bytes;
        // Explicit deref-to-trait-object first: `Arc<T>` itself implements `AsRef<T>`, which
        // would make a bare `src.bytes.as_ref()` resolve to THAT (returning the fat
        // `&(dyn AsRef<[u8]> + Send + Sync)`) instead of the inner `AsRef<[u8]>::as_ref` this
        // needs — force the deref first so only the trait object's own impl is a candidate.
        let inner: &(dyn AsRef<[u8]> + Send + Sync) = &*src.bytes;
        let bytes: &[u8] = inner.as_ref();
        let layer_base = src.layer_base;
        let mut global = Vec::with_capacity(local_ids.len());
        for &lid in local_ids {
            let off = lid as usize * stride;
            let slice = bytes
                .get(off..off + stride)
                .ok_or_else(|| be("moe pager: expert id out of range for this layer's bank"))?;
            pager.ensure_resident(vk, self.staging.as_ref(), layer_base + lid, slice)?;
            global.push(layer_base + lid);
        }
        pager.flush_lut(vk)?;
        Ok(global)
    }

    /// The shared upload ring / its per-half capacity (see the `ring` field's doc). The CURSOR
    /// into it lives with the adapter's per-execute stream state, not here.
    pub fn ring(&self) -> &dyn Buffer {
        self.ring.as_ref()
    }

    pub fn ring_half_bytes(&self) -> usize {
        self.ring_half_bytes
    }

    /// The LUT tape buffer every windowed dispatch binds (see the `tape` field's doc).
    pub fn tape(&self) -> &dyn Buffer {
        self.tape.as_ref()
    }

    /// Whether ALL `n_expert` experts of `buf_id`'s layer are resident in its pool — the
    /// no-readback inline gate for a small-m (decode) layer: when true, any routing the GPU
    /// picks is covered, so the host needs no routing knowledge at all.
    pub fn all_resident(&self, buf_id: usize, n_expert: usize) -> bool {
        let (_, pool, src) = match self.sources.get(&buf_id) {
            Some(s) => s,
            None => return false,
        };
        let pager = &self.pools[*pool].pager;
        (0..n_expert as u32).all(|e| pager.is_resident(src.layer_base + e))
    }

    /// LRU maintenance for an inline-recorded (no-readback) layer: mark all `n_expert` blocks
    /// MRU. Callers gate on [`Self::all_resident`], so every touch is a hit — no uploads, no LUT
    /// mutation (the property that makes inline recording safe while earlier segments are still
    /// in flight).
    pub fn touch_all_hits(&mut self, buf_id: usize, n_expert: usize) {
        let (_, pool, src) = self
            .sources
            .get(&buf_id)
            .expect("moe pager: touch on an unregistered buffer");
        let layer_base = src.layer_base;
        let pager = &mut self.pools[*pool].pager;
        pager.begin_batch();
        for e in 0..n_expert as u32 {
            let r = pager.pager.touch(layer_base + e);
            debug_assert!(
                matches!(r, Resolution::Hit { .. }),
                "touch_all_hits on a non-resident block (all_resident gate violated)"
            );
        }
    }

    /// Open a touch batch on `buf_id`'s pool — call once per (layer, role) residency resolution,
    /// BEFORE the first [`Self::stage_role`] call of that batch (rotations re-call `stage_role`
    /// WITHIN the same batch; the epoch protection must span them).
    pub fn begin_batch(&mut self, buf_id: usize) {
        let (_, pool, _) = self
            .sources
            .get(&buf_id)
            .expect("moe pager: begin_batch on an unregistered buffer");
        self.pools[*pool].pager.begin_batch();
    }

    /// Stage `local_ids`' residency for `buf_id`'s layer through `rec`-recorded ring→arena
    /// copies: hits are marked MRU, misses memcpy into the ring at `half_base + *cursor` and
    /// record the slot copy ([`GpuPager::touch_staged`]). Stops when the current ring half can't
    /// hold the next miss and returns how many ids were FULLY staged — the caller rotates the
    /// ring (submitting the recorder, fencing the half) and re-calls with the remainder; an
    /// expert's bytes are never split across a rotation. Progress is guaranteed: a half holds at
    /// least one slot of every pool (asserted at construction).
    ///
    /// Within-batch eviction safety (`infr_core::pager::Pager`'s invariant) holds across
    /// rotations: a rotation performs no touches, and every id staged earlier in this batch is
    /// MRU-protected from the batch's later touches exactly as in the one-shot path.
    /// `scan` selects the residency policy: `true` = the touch-all full-set sweep (batched
    /// prefill) → scan-resistant cold-end insertion; `false` = classic LRU (decode's routed-only
    /// readback path). See `infr_core::pager::Pager::touch_cold`.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_role(
        &mut self,
        rec: &crate::recorder::Recorder<'_>,
        half_base: usize,
        cursor: &mut usize,
        buf_id: usize,
        local_ids: &[u32],
        scan: bool,
    ) -> Result<usize> {
        let (pool_idx, stride, layer_base, bytes_arc) = {
            let (_, pool, src) = self
                .sources
                .get(&buf_id)
                .ok_or_else(|| be("moe pager: stage on an unregistered buffer"))?;
            (
                *pool,
                src.stride_bytes,
                src.layer_base,
                Arc::clone(&src.bytes),
            )
        };
        // See `touch_role` for why the explicit deref-to-trait-object.
        let inner: &(dyn AsRef<[u8]> + Send + Sync) = &*bytes_arc;
        let bytes: &[u8] = inner.as_ref();
        // Disjoint field borrows (the pool mutably, the ring by ref) — destructure once.
        let Self {
            pools,
            ring,
            ring_half_bytes,
            ..
        } = self;
        let pager = &mut pools[pool_idx].pager;
        let half_bytes = *ring_half_bytes;
        debug_assert!(
            half_bytes >= pager.slot_bytes(),
            "ring half smaller than a slot (construction floor violated)"
        );
        for (i, &lid) in local_ids.iter().enumerate() {
            let id = layer_base + lid;
            if !pager.is_resident(id) && *cursor + pager.slot_bytes() > half_bytes {
                return Ok(i); // half full — caller rotates and continues from here
            }
            let off = lid as usize * stride;
            let slice = bytes
                .get(off..off + stride)
                .ok_or_else(|| be("moe pager: expert id out of range for this layer's bank"))?;
            *cursor +=
                pager.touch_staged(rec, ring.as_ref(), half_base + *cursor, id, slice, scan)?;
        }
        Ok(local_ids.len())
    }

    /// Freeze `buf_id`'s layer LUT window — `n_expert` slot indices starting at its `layer_base`,
    /// copied from the pool's host mirror into the tape at `*tape_cursor` — and return the tape
    /// word offset the layer's dispatches pass as `lut_base` (`lut[base + local_id]`). Must be
    /// called AFTER every `stage_role` call for that (layer, role) batch completed (the
    /// within-batch LUT rule: the window must reflect every id the batch touched). Errors on
    /// tape overflow instead of wrapping — a wrapped window could alias one an in-flight segment
    /// still reads (the cursor only resets after a full drain; see the `tape` field's doc).
    pub fn lut_window(
        &mut self,
        tape_cursor: &mut usize,
        buf_id: usize,
        n_expert: usize,
    ) -> Result<u32> {
        let (_, pool, src) = self
            .sources
            .get(&buf_id)
            .ok_or_else(|| be("moe pager: lut_window on an unregistered buffer"))?;
        if *tape_cursor + n_expert > self.tape_words {
            return Err(be(format!(
                "moe pager: LUT tape overflow ({} + {n_expert} > {} words) — one drain cycle \
                 recorded more layer windows than the tape holds",
                *tape_cursor, self.tape_words,
            )));
        }
        let window = self.pools[*pool]
            .pager
            .lut_words(src.layer_base as usize, n_expert);
        // Safety: the tape is session-owned Staging (persistently mapped) and the region written
        // is fresh this drain cycle — no in-flight reader can see a partial window.
        let base = unsafe { as_vk_buf(self.tape.as_ref()) }
            .mapped_ptr()
            .ok_or_else(|| be("pager LUT tape is not persistently mapped"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                window.as_ptr(),
                base.add(*tape_cursor * 4).cast::<u32>(),
                n_expert,
            );
        }
        let w = *tape_cursor as u32;
        *tape_cursor += n_expert;
        Ok(w)
    }

    fn pool_of(&self, buf_id: usize) -> &Pool {
        let (_, pool, _) = self
            .sources
            .get(&buf_id)
            .expect("moe pager: arena/lut lookup on an unregistered buffer");
        &self.pools[*pool]
    }

    /// The arena buffer `buf_id`'s pool dispatches against (callers gate on [`Self::is_paged`]
    /// first — this panics on an unregistered buffer).
    pub fn arena(&self, buf_id: usize) -> &dyn Buffer {
        self.pool_of(buf_id).pager.arena_buffer()
    }

    /// `buf_id`'s pool arena's 64-bit `VkDeviceAddress` — the base the paged kernels scale the LUT
    /// slot index onto (`arena_addr + slot * slot_bytes`). Passed to the shader as a push constant.
    pub fn arena_addr(&self, buf_id: usize) -> u64 {
        self.pool_of(buf_id).pager.arena_addr()
    }

    /// `buf_id`'s pool per-slot byte stride — the multiplier the paged kernels apply to the LUT
    /// slot index (see [`Self::arena_addr`]).
    pub fn slot_bytes(&self, buf_id: usize) -> usize {
        self.pool_of(buf_id).slot_bytes
    }

    /// [`Self::arena`]'s LUT twin.
    pub fn lut(&self, buf_id: usize) -> &dyn Buffer {
        self.pool_of(buf_id).pager.lut_buffer()
    }

    /// Aggregate stats across every pool of `role` (the pool split is a capacity detail; the
    /// hit/miss story reads per role).
    pub fn stats(&self, role: Role) -> PagerStats {
        let mut agg = PagerStats::default();
        for p in self.pools.iter().filter(|p| p.role == role) {
            let s = p.pager.stats();
            agg.hits += s.hits;
            agg.misses += s.misses;
            agg.evictions += s.evictions;
        }
        agg
    }

    /// `INFR_PAGER_STATS=1`: print each pool's hit/miss/eviction counters. Called after
    /// generation finishes (see the CLI's bench/run/serve exit paths) — cheap enough to always
    /// compute, only printed when asked.
    pub fn print_stats_if_enabled(&self) {
        if !self.print_stats {
            return;
        }
        for p in &self.pools {
            let s = p.pager.stats();
            eprintln!(
                "[moe pager] {}/{:.1}MB: hits={} misses={} evictions={} hit_rate={:.3} slots={}",
                p.role.name(),
                p.slot_bytes as f64 / 1e6,
                s.hits,
                s.misses,
                s.evictions,
                s.hit_rate(),
                p.pager.n_slots(),
            );
        }
    }
}

/// `VulkanBackend::moe_pager`'s field type — a `Mutex` since `touch_role` mutates the LRU/arena and
/// the adapter calls it from `execute_static` (`&VulkanBackend`, not `&mut`).
pub type MoePagerCell = Mutex<Option<MoePagerSession>>;

// ─── Dense layer-streaming session ─────────────────────────────────────────────────────────────
//
// The MoE session above is demand-driven (routing is GPU-decided, residency resolves per touch);
// dense streaming is the SCHEDULE-driven policy `infr_core::pager`'s module doc names: a dense
// forward visits layers in one fixed order every pass, so the host knows every "miss" in advance
// and needs NO readbacks, NO LUT hop and NO paged kernel twins at all. One block = one per-layer
// weight tensor GROUP exactly as the seam uploads it (a fused qkv or gate_up concat is one
// block; split tensors are one block each) — every dense kernel already reads its weight from a
// `w_off` ELEMENT offset (the stacked-MoE-tensor convention), so a streamed dispatch computes the
// resident slot's base BYTE address (`arena_addr + slot * slot_bytes`, 64-bit — see
// `GpuPager::arena_addr`/`DensePagerSession::stage`) and rides the op's own `w_off` on top as a
// within-slot element offset, exactly like the resident path's binding + offset.
// Pools are keyed per (dtype, padded byte stride) tensor class — same reasoning as the MoE
// per-(role, slot_bytes) pools (fixed slot offsets require uniform strides; mixed-precision GGUFs
// bump a subset of layers' tensors to a wider format).
//
// Rejected alternatives (design notes for the seam this replaces):
//   - Descriptor-level (buffer, offset) rebinding: `Recorder::bind_descriptors` binds
//     `(buffer, 0, WHOLE_SIZE)` through ~seventy dispatch helpers — threading a per-binding
//     offset through every signature is a much bigger, riskier diff than reusing the `w_off`
//     element offset the kernels already take, and buys nothing (same descriptor write count).
//   - `-DPAGED` LUT twins of the dense kernels (the MoE route): pointless indirection — the host
//     knows the slot at record time, so the offset can be baked directly; a LUT hop would add a
//     device dependency for information the host already has.
//   - Embeddings / lm_head / norms / biases stay RESIDENT: norms and biases are consumed by ops
//     with no weight-offset support and are tiny (a few KB/layer); token_embd/lm_head are read at
//     every token edge — streaming lm_head would add its full bytes to every token's PCIe bill
//     with zero locality to exploit, a strict loss.

/// Where one streamed dense block's bytes live: one or more consecutive zero-copy views into the
/// GGUF mmap (SEGMENTS, in upload order — a fused qkv/gate_up block lists its component tensors
/// so the concat never materializes in host RAM), plus the block's schedule id within its pool
/// (ascending layer order — the cyclic-sweep key `infr_core::pager::Pager::schedule` expects).
pub struct DenseSource {
    pub segments: Vec<Arc<dyn AsRef<[u8]> + Send + Sync>>,
    pub block_id: u32,
}

/// One dense pool's fixed layout: every block in it shares `slot_bytes` (the PADDED stride —
/// a multiple of 4 (u32 arena) AND of the pool dtype's block byte size, so a slot base is always
/// a whole number of quant blocks). The arena is a `bufferDeviceAddress` buffer read by 64-bit
/// pointer (see [`DensePagerSession`]), so `n_slots` is bounded only by the VRAM budget share (and
/// the seam's floor) — there is NO per-arena `maxStorageBufferRange` cap and NO u32 element-reach
/// cap (a slot's base byte address is computed in 64-bit; the op's `w_off` element offset rides on
/// top within the kernel). Contrast the resident/SSBO path, which those two caps DID bind.
pub struct DensePoolSpec {
    pub slot_bytes: usize,
    pub n_slots: usize,
    pub n_blocks: usize,
}

struct DensePool {
    spec: DensePoolSpec,
    pager: GpuPager,
}

/// Layout for [`DensePagerSession::new`] — like [`MoePagerLayout`], sized up front so the session
/// exists (and `Backend::dense_paged` answers truthy) BEFORE the seam's weight-load closure binds
/// the first placeholder.
pub struct DensePagerLayout {
    pub pools: Vec<DensePoolSpec>,
    /// Pinned upload ring total bytes (two fence-rotated halves); `0` = [`ring_bytes_policy`]'s
    /// floor. Each half is floored at the largest pool slot so one miss always fits.
    pub ring_bytes: usize,
}

/// One model's whole dense layer-streaming session: per-(dtype, stride) arena pools + the shared
/// pinned upload ring. Same ownership story as [`MoePagerSession`] (lives on the `VulkanBackend`
/// handle, `None` for every non-streamed model — zero cost on the resident path). A model is
/// either MoE-paged or dense-streamed, never both (the seam errors on the mixed case).
pub struct DensePagerSession {
    pools: Vec<DensePool>,
    /// `buffer_identity(placeholder)` -> (pool index, source) for every streamed block. A
    /// resident tensor's buffer is never registered here — the adapter's lookup misses and the
    /// op lowers through the ordinary resident path.
    sources: HashMap<usize, (usize, DenseSource)>,
    ring: Box<dyn Buffer>,
    ring_half_bytes: usize,
    print_stats: bool,
}

impl DensePagerSession {
    pub fn new(vk: &VulkanBackend, layout: DensePagerLayout) -> Result<Self> {
        // The streamed kernels read the arena by 64-bit device address (native_weight_addr.glsl), so
        // BDA is required. It is probed and hard-errored globally at init (lib.rs, `caps()
        // .buffer_device_address`); assert here so a future refactor that lands a dense session on a
        // BDA-less device fails loudly rather than allocating an un-addressable arena.
        debug_assert!(
            vk.caps().buffer_device_address,
            "dense streaming needs bufferDeviceAddress (BDA arena)"
        );
        let mut pools = Vec::with_capacity(layout.pools.len());
        let mut max_slot = 4usize;
        for spec in layout.pools {
            max_slot = max_slot.max(spec.slot_bytes);
            pools.push(DensePool {
                // Dense-streaming pool: the arena is a `bufferDeviceAddress` buffer the streamed
                // kernels read through a 64-bit pointer, so it may span as much VRAM as the
                // budget allows — no `maxStorageBufferRange` cap (the pre-BDA ~4 GiB-per-pool
                // ceiling this lifts), no u32 element-reach cap.
                pager: GpuPager::new(vk, spec.n_blocks, spec.n_slots, spec.slot_bytes)?,
                spec,
            });
        }
        let ring_total = if layout.ring_bytes > 0 {
            layout.ring_bytes
        } else {
            ring_bytes_policy(0)
        };
        // Each half must hold the largest slot or `stage` could never make progress on that pool.
        let ring_half_bytes = (ring_total / 2).max(max_slot);
        let ring = vk.alloc_uninit(2 * ring_half_bytes, BufferUsage::Staging)?;
        Ok(Self {
            pools,
            sources: HashMap::new(),
            ring,
            ring_half_bytes,
            print_stats: std::env::var("INFR_PAGER_STATS").is_ok(),
        })
    }

    /// Register one streamed block — called from the seam's weight-load closure (once per
    /// streamed weight group) instead of uploading it. `pool` indexes [`DensePagerLayout::pools`]
    /// (the seam enumerates the layout and the registrations from the same plan, so a mismatch is
    /// a seam bug — validated loudly here).
    pub fn register(&mut self, pool: usize, buf_id: usize, source: DenseSource) -> Result<()> {
        let p = self
            .pools
            .get(pool)
            .ok_or_else(|| be(format!("dense pager: pool index {pool} out of range")))?;
        let total: usize = source
            .segments
            .iter()
            .map(|s| {
                let inner: &(dyn AsRef<[u8]> + Send + Sync) = &**s;
                inner.as_ref().len()
            })
            .sum();
        if total > p.spec.slot_bytes {
            return Err(be(format!(
                "dense pager: block bytes ({total}) exceed pool {pool}'s slot stride ({})",
                p.spec.slot_bytes
            )));
        }
        if source.block_id as usize >= p.spec.n_blocks {
            return Err(be(format!(
                "dense pager: block id {} out of range for pool {pool} ({} blocks)",
                source.block_id, p.spec.n_blocks
            )));
        }
        self.sources.insert(buf_id, (pool, source));
        Ok(())
    }

    /// Whether `buf_id` (see [`buffer_identity`]) is a registered streamed block — the adapter's
    /// per-`Op::Linear` dispatch check.
    pub fn is_streamed(&self, buf_id: usize) -> bool {
        self.sources.contains_key(&buf_id)
    }

    /// Ensure `buf_id`'s block is resident, staging a miss through `rec`-recorded ring→arena
    /// copies at `half_base + *cursor`. Returns the resident slot's arena base BYTE address (the
    /// streamed dispatch sets `w_addr` to it and adds the op's own `w_off` element offset on top —
    /// see native_weight_addr.glsl and [`crate::recorder::Recorder::linear_native_at`]), or
    /// `None` when the current ring half can't hold the miss — the caller rotates the ring
    /// (pipelined submit) and re-calls. The address is computed in 64-bit, so no arena size
    /// overflows it (the u32 element-reach the SSBO path needed is gone). Residency rides the exact
    /// cyclic-sweep policy (`infr_core::pager::Pager::schedule`); one block = one touch batch (the
    /// epoch guard protects it across the caller's rotations).
    pub fn stage(
        &mut self,
        rec: &crate::recorder::Recorder<'_>,
        half_base: usize,
        cursor: &mut usize,
        buf_id: usize,
    ) -> Result<Option<u64>> {
        let Self {
            pools,
            sources,
            ring,
            ring_half_bytes,
            ..
        } = self;
        let (pool_idx, src) = sources
            .get(&buf_id)
            .ok_or_else(|| be("dense pager: stage on an unregistered buffer"))?;
        let pool = &mut pools[*pool_idx];
        let id = src.block_id;
        if !pool.pager.is_resident(id) && *cursor + pool.spec.slot_bytes > *ring_half_bytes {
            return Ok(None); // half full — caller rotates and re-calls
        }
        pool.pager.begin_batch();
        // See `MoePagerSession::touch_role` for why the explicit deref-to-trait-object.
        let seg_refs: Vec<&[u8]> = src
            .segments
            .iter()
            .map(|s| {
                let inner: &(dyn AsRef<[u8]> + Send + Sync) = &**s;
                inner.as_ref()
            })
            .collect();
        let (slot, consumed) =
            pool.pager
                .schedule_staged(rec, ring.as_ref(), half_base + *cursor, id, &seg_refs)?;
        *cursor += consumed;
        // Slot base BYTE address = arena base + slot * slot_bytes, in 64-bit (the BDA arena's
        // `arena_addr()`; the streamed kernel dereferences this pointer). No cap: the multiply and
        // the address are 64-bit, so an arena of any size the VRAM budget allows is addressable.
        let addr = pool.pager.arena_addr() + slot as u64 * pool.spec.slot_bytes as u64;
        Ok(Some(addr))
    }

    pub fn ring_half_bytes(&self) -> usize {
        self.ring_half_bytes
    }

    /// `INFR_PAGER_STATS=1`: per-pool hit/miss/eviction counters (cyclic-sweep hit rate =
    /// `(n_slots-1) / n_blocks` per pass at steady state — the honest expectation to check
    /// against).
    pub fn print_stats_if_enabled(&self) {
        if !self.print_stats {
            return;
        }
        for (i, p) in self.pools.iter().enumerate() {
            let s = p.pager.stats();
            eprintln!(
                "[dense pager] pool{i}/{:.1}MB: hits={} misses={} evictions={} hit_rate={:.3} \
                 slots={}/{}",
                p.spec.slot_bytes as f64 / 1e6,
                s.hits,
                s.misses,
                s.evictions,
                s.hit_rate(),
                p.spec.n_slots,
                p.spec.n_blocks,
            );
        }
    }
}

/// `VulkanBackend::dense_pager`'s field type — same locking story as [`MoePagerCell`].
pub type DensePagerCell = Mutex<Option<DensePagerSession>>;
