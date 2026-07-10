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

// ─── MoE expert-bank paging session (slice 2: wiring into the execution path) ─────────────────
//
// The pieces above are the block-agnostic host<->VRAM cache; everything below is the MoE-specific
// glue: one [`GpuPager`] per expert ROLE (gate/up/down — a split, non-fused bank, the only shape
// llama4/Scout needs; a fused gate_up bank would need a 4th role and is a follow-up), a table
// mapping a bound weight BUFFER's identity to where its layer's expert bytes live in the mmap'd
// GGUF, and the one persistent staging buffer every role's uploads share.
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

/// One paged expert role. `gate`/`up`/`down` each get an independent arena+LUT (their per-expert
/// byte sizes need not match, though in practice they're equal — same `ne*n_ff_exp` elements, same
/// dtype).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    Gate,
    Up,
    Down,
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

struct RolePager {
    pager: GpuPager,
    /// `buffer_identity(placeholder)` -> this layer's expert source, for every PAGED layer of this
    /// role. A non-paged layer's gate/up/down buffer is never registered here — the adapter's
    /// lookup simply misses and falls through to the ordinary resident-weight path.
    sources: HashMap<usize, ExpertSource>,
}

impl RolePager {
    fn touch(
        &mut self,
        vk: &VulkanBackend,
        staging: &dyn Buffer,
        buf_id: usize,
        local_ids: &[u32],
    ) -> Result<Vec<u32>> {
        let src = self
            .sources
            .get(&buf_id)
            .ok_or_else(|| be("moe pager: touch on an unregistered buffer"))?;
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
            self.pager
                .ensure_resident(vk, staging, layer_base + lid, slice)?;
            global.push(layer_base + lid);
        }
        self.pager.flush_lut(vk)?;
        Ok(global)
    }
}

/// One model's whole paged-MoE session: gate/up/down role pagers + the shared persistent staging
/// buffer their uploads reuse (the bandwidth probe's headline finding — see `pager.rs`'s module
/// doc and `tests/bandwidth_probe.rs`). Lives on `VulkanShared` for the process's lifetime once a
/// paged model is loaded (`VulkanBackend::init_moe_pager`); `None` for every non-paged model —
/// zero cost, zero behavior change on the common (fits-in-VRAM) path.
pub struct MoePagerSession {
    gate: RolePager,
    up: RolePager,
    down: RolePager,
    staging: Box<dyn Buffer>,
    print_stats: bool,
}

/// Fixed layout for [`MoePagerSession::new`] — sizes the three arenas/LUTs UP FRONT, before any
/// tensor is registered. This split (layout now, [`MoePagerSession::register`] per tensor later)
/// matters for sequencing: the session must exist and answer `is_paged`/`Backend::moe_paged` truthy
/// BEFORE the seam's weight-load closure runs (so a paged tensor's placeholder buffer is
/// recognized the very first time the adapter executes a graph, not just after the whole model is
/// loaded) — see `infr-llama`'s `generate_dense_vulkan_session` for the call order this enables.
pub struct MoePagerLayout {
    /// Total distinct experts nameable per role's LUT = `n_paged_layers * n_expert`.
    pub n_blocks: usize,
    /// VRAM slots to give EACH role (paired 1:1 — one touch always resolves one gate + one up + one
    /// down expert together, so keeping the same slot count per role keeps them from thrashing at
    /// different rates). Computed by the caller from the pager's byte budget / (per-expert bytes
    /// summed across roles) — see `seam::mod`'s placement policy.
    pub n_slots: usize,
    pub gate_slot_bytes: usize,
    pub up_slot_bytes: usize,
    pub down_slot_bytes: usize,
}

impl MoePagerSession {
    pub fn new(vk: &VulkanBackend, layout: MoePagerLayout) -> Result<Self> {
        let gate = RolePager {
            pager: GpuPager::new(vk, layout.n_blocks, layout.n_slots, layout.gate_slot_bytes)?,
            sources: HashMap::new(),
        };
        let up = RolePager {
            pager: GpuPager::new(vk, layout.n_blocks, layout.n_slots, layout.up_slot_bytes)?,
            sources: HashMap::new(),
        };
        let down = RolePager {
            pager: GpuPager::new(vk, layout.n_blocks, layout.n_slots, layout.down_slot_bytes)?,
            sources: HashMap::new(),
        };
        let staging_bytes = layout
            .gate_slot_bytes
            .max(layout.up_slot_bytes)
            .max(layout.down_slot_bytes);
        let staging = vk.alloc_uninit(staging_bytes.max(4), BufferUsage::Staging)?;
        Ok(Self {
            gate,
            up,
            down,
            staging,
            print_stats: std::env::var("INFR_PAGER_STATS").is_ok(),
        })
    }

    /// Register one paged layer's `role` tensor — called from the seam's weight-load closure
    /// (once per paged `_exps` tensor) instead of uploading it. `buf_id` is the placeholder buffer's
    /// identity (see [`buffer_identity`]); `source` is where its bytes actually live.
    pub fn register(&mut self, role: Role, buf_id: usize, source: ExpertSource) {
        let sources = match role {
            Role::Gate => &mut self.gate.sources,
            Role::Up => &mut self.up.sources,
            Role::Down => &mut self.down.sources,
        };
        sources.insert(buf_id, source);
    }

    fn role(&self, role: Role) -> &RolePager {
        match role {
            Role::Gate => &self.gate,
            Role::Up => &self.up,
            Role::Down => &self.down,
        }
    }

    /// Whether `buf_id` (see [`buffer_identity`]) is a registered paged tensor of `role` — the
    /// adapter's per-`MoeFfn` dispatch check.
    pub fn is_paged(&self, role: Role, buf_id: usize) -> bool {
        self.role(role).sources.contains_key(&buf_id)
    }

    /// Resolve residency for every id in `local_ids` (this token's routed experts, LOCAL to the
    /// layer) against `role`'s pager, uploading misses through the shared staging buffer and
    /// flushing the LUT once. Returns the GLOBAL ids (`layer_base + local_id`) the paged GEMV must
    /// read instead of `local_ids` — see [`ExpertSource::layer_base`].
    pub fn touch_role(
        &mut self,
        vk: &VulkanBackend,
        role: Role,
        buf_id: usize,
        local_ids: &[u32],
    ) -> Result<Vec<u32>> {
        // `staging` (a disjoint field) borrowed immutably alongside `&mut self.{gate,up,down}`
        // below — ordinary disjoint-field borrowing, not a whole-`self` borrow, so this needs no
        // helper method gymnastics.
        let staging = self.staging.as_ref();
        match role {
            Role::Gate => self.gate.touch(vk, staging, buf_id, local_ids),
            Role::Up => self.up.touch(vk, staging, buf_id, local_ids),
            Role::Down => self.down.touch(vk, staging, buf_id, local_ids),
        }
    }

    pub fn arena(&self, role: Role) -> &dyn Buffer {
        self.role(role).pager.arena_buffer()
    }

    pub fn lut(&self, role: Role) -> &dyn Buffer {
        self.role(role).pager.lut_buffer()
    }

    pub fn stats(&self, role: Role) -> PagerStats {
        self.role(role).pager.stats()
    }

    /// `INFR_PAGER_STATS=1`: print each role's hit/miss/eviction counters. Called after generation
    /// finishes (see the CLI's bench/run/serve exit paths) — cheap enough to always compute, only
    /// printed when asked.
    pub fn print_stats_if_enabled(&self) {
        if !self.print_stats {
            return;
        }
        for (name, role) in [("gate", Role::Gate), ("up", Role::Up), ("down", Role::Down)] {
            let s = self.stats(role);
            eprintln!(
                "[moe pager] {name}: hits={} misses={} evictions={} hit_rate={:.3} slots={}",
                s.hits,
                s.misses,
                s.evictions,
                s.hit_rate(),
                self.role(role).pager.n_slots(),
            );
        }
    }
}

/// `VulkanShared::moe_pager`'s field type — a `Mutex` since `touch_role` mutates the LRU/arena and
/// the adapter calls it from `execute_static` (`&VulkanBackend`, not `&mut`).
pub type MoePagerCell = Mutex<Option<MoePagerSession>>;
