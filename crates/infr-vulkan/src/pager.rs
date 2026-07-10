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
//! `shaders/native_gemv_id.comp` / `native_gemv_id_multi.comp`: `wbase = lut[ids[slot]] * stride`
//! instead of `wbase = ids[slot] * stride`). A FUTURE dense layer-streaming policy (NOT
//! implemented here — see the task doc) would reuse this exact struct with `BlockId = layer_idx`,
//! `slot_bytes` = one layer's weight size, and a schedule-driven (not LRU) `touch` order (a dense
//! decode visits layers in a fixed known order, so it can exact-prefetch layer `l+1` while `l`
//! runs) — nothing in the arena/LUT/upload core below assumes MoE or LRU.
//!
//! # LUT
//! One small `Staging` (host-visible, persistently mapped — no GPU submit to update) buffer of
//! `n_blocks` `u32` slot indices (`infr_core::pager::NOT_RESIDENT` for an absent block), mirrored
//! host-side and fully rewritten + re-uploaded whenever residency changes since the last
//! [`GpuPager::flush_lut`] — cheap at the block counts this task's models need (Scout: 48 layers
//! x 16 experts x 3 roles = 2304 entries, 9 KiB).
//!
//! # Eviction upgrade path
//! Plain LRU (see `infr_core::pager`). llama.cpp issue #20757's SLRU-with-admission is the
//! documented upgrade if pure LRU thrashes on an adversarial access pattern — not implemented here.
use std::sync::Arc;

use ash::vk;

use infr_core::backend::{Buffer, BufferUsage};
use infr_core::error::Result;
use infr_core::pager::{BlockId, Pager, PagerStats, Resolution, NOT_RESIDENT};
use infr_core::Backend;

use super::{as_vk_buf, VulkanBackend};

/// Fixed-budget evictable VRAM cache of uniform `slot_bytes` blocks. See the module doc.
pub struct GpuPager {
    pager: Pager,
    slot_bytes: usize,
    /// Device-local arena: `n_slots * slot_bytes`, one contiguous buffer (the id-indexed shaders
    /// address it as `array<u32>` with `lut[id] * stride` offsets — one binding, not N).
    arena: Box<dyn Buffer>,
    /// Host-visible LUT mirror (mutated in place, re-uploaded on change) + the device buffer it's
    /// pushed to. `n_blocks` entries.
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
    pub fn new(
        vk: &VulkanBackend,
        n_blocks: usize,
        n_slots: usize,
        slot_bytes: usize,
    ) -> Result<Self> {
        assert!(n_slots > 0, "GpuPager needs at least one slot");
        let arena = vk.alloc_uninit(n_slots * slot_bytes, BufferUsage::Weights)?;
        let lut_dev = vk.alloc_uninit(n_blocks.max(1) * 4, BufferUsage::Staging)?;
        let lut_host = vec![NOT_RESIDENT; n_blocks.max(1)];
        // Seed the device LUT with the same all-absent state (arena/LUT start coherent).
        vk.upload(lut_dev.as_ref(), bytemuck::cast_slice(&lut_host))?;
        Ok(Self {
            pager: Pager::new(n_slots),
            slot_bytes,
            arena,
            lut_host,
            lut_dev,
            lut_dirty: false,
        })
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
