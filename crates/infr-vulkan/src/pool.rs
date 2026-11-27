//! What a device-local allocation commits, for the VRAM guard (`VulkanBackend::check_vram_commit`).
//!
//! gpu-allocator sub-allocates a buffer below `DEDICATED_MIN` from a shared block of its own, and a
//! device's live memory budget counts each such block whole from the moment it is opened (on a
//! driver whose own figure does not, the backend holds the reading at the budget it had at open,
//! which makes it so: `VulkanShared::budget_ceiling`). A pooled buffer that lands in the free space
//! of an open block therefore commits nothing new, and one that opens a block commits the whole
//! block, whatever its own size. gpu-allocator can say which of the two a request will do only by
//! walking every allocation it holds (`generate_report`), so
//! [`PoolLedger`] keeps its own copy of the shared blocks, updated where the backend allocates and
//! frees, and answers from that.
//!
//! The copy follows gpu-allocator 0.27's rules, read in its source:
//!
//! - `Allocator::new` builds every allocator with `AllocationSizes::default()`, whatever the create
//!   info asks: [`DEVICE_BLOCK`] for a memory type that is not host-visible, [`HOST_BLOCK`] for one
//!   that is.
//! - A request goes to the first memory type its buffer accepts that has the flags its location
//!   prefers, else the first with the flags it requires (`Allocator::allocate`,
//!   `find_memorytype_index`): device-local for `GpuOnly`; host-visible, coherent and device-local,
//!   else host-visible and coherent, for `CpuToGpu`; host-visible, coherent and cached, else
//!   host-visible and coherent, for `GpuToCpu`. (A `CpuToGpu` request that fails in its preferred
//!   type is retried in the required one; only that failure path is not foreseen.)
//! - Within that type a request takes a free range of any open shared block, and a new block is
//!   opened only when none has room (`MemoryType::allocate`). A free range has room when the
//!   request, placed at its alignment, ends inside it (`FreeListAllocator::allocate`); every
//!   allocation this backend makes is linear, so the buffer-image granularity never moves one.
//! - An allocation's chunk starts where the free range it was carved from started, so the padding
//!   its alignment needed is part of it; adjacent free ranges merge when one is freed.
//! - An emptied shared block is destroyed unless it is the last one of its type
//!   (`MemoryType::free`).
//! - A request larger than its type's block gets a block of its own, as a dedicated one does; the
//!   ledger leaves both out.
//!
//! [`Pool`] puts gpu-allocator and the ledger behind one lock and is the only way to reach
//! gpu-allocator, so no allocation or free can miss the ledger, and the guard's check of what an
//! allocation commits and the allocation itself cannot be split by another thread's.
//!
//! `ledger_matches_gpu_allocators_blocks` and `predicted_commits_are_what_gpu_allocator_opens`
//! hold the copy against gpu-allocator's own report on a device, so a gpu-allocator release that
//! changes any of this fails there; `the_mirror_is_of_the_resolved_gpu_allocator` fails without a
//! GPU when the resolved version is not the one these rules were read in.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use infr_core::error::{backend as be, Result};

use crate::{DEDICATED_MIN, VRAM_GUARD_MIN_ALLOC};

/// The gpu-allocator release whose source the module documentation's rules were read in.
pub(crate) const MIRRORED_GPU_ALLOCATOR: &str = "0.27.0";

/// gpu-allocator's shared block for a memory type that is not host-visible
/// (`AllocationSizes::default`'s `device_memblock_size`).
pub(crate) const DEVICE_BLOCK: u64 = 256 << 20;

/// gpu-allocator's shared block for a host-visible memory type (`AllocationSizes::default`'s
/// `host_memblock_size`).
pub(crate) const HOST_BLOCK: u64 = 64 << 20;

/// What admitting a device-local allocation commits on a device whose live budget counts
/// gpu-allocator's blocks whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Commit {
    /// An allocation of its own (dedicated, a block of its own, or the unified-memory spill), or
    /// any allocation on a device without a live budget: the bytes it asks for.
    Own(u64),
    /// A sub-allocation inside a shared block already open: nothing new.
    InOpenBlock,
    /// A sub-allocation no open block has room for: the whole block it opens.
    NewBlock { block: u64, request: u64 },
}

impl Commit {
    /// The bytes this takes from the budget.
    pub(crate) fn bytes(self) -> u64 {
        match self {
            Commit::Own(bytes) => bytes,
            Commit::InOpenBlock => 0,
            Commit::NewBlock { block, .. } => block,
        }
    }

    /// The bytes the guard checks against the budget, or `None` when it lets this through
    /// unasked: below [`VRAM_GUARD_MIN_ALLOC`], so a sub-allocation inside an open block never
    /// costs a driver query, while one that opens a block is checked whatever its own size.
    pub(crate) fn checked(self) -> Option<u64> {
        let bytes = self.bytes();
        (bytes >= VRAM_GUARD_MIN_ALLOC).then_some(bytes)
    }
}

/// The guard's decision: whether an allocation committing `commit` is admitted, `reading` giving
/// the budget's (used, budget) bytes. A commitment the guard lets through unasked
/// ([`Commit::checked`]) is admitted without a reading, so it costs no driver query.
pub(crate) fn admits(commit: Commit, reading: impl FnOnce() -> (u64, u64)) -> bool {
    match commit.checked() {
        None => true,
        Some(want) => {
            let (used, budget) = reading();
            fits_budget(used, want, budget)
        }
    }
}

/// Whether `want` more bytes fit a budget of which `used` is already taken: the comparison the
/// guard and the VRAM-first KV placement share.
pub(crate) fn fits_budget(used: u64, want: u64, budget: u64) -> bool {
    used.saturating_add(want) <= budget
}

/// One open shared block.
struct Block {
    ty: usize,
    size: u64,
    /// Every live chunk, start to end; the gaps between them are the block's free ranges.
    chunks: BTreeMap<u64, u64>,
}

/// A copy of gpu-allocator's shared blocks: their chunks and free ranges, by memory type. See the
/// module documentation for the rules it follows.
pub(crate) struct PoolLedger {
    /// Each memory type's property flags, by index.
    types: Vec<vk::MemoryPropertyFlags>,
    /// Open shared blocks by their `VkDeviceMemory` handle.
    blocks: BTreeMap<u64, Block>,
    /// Every free range of every open block, as (memory type, length, start, block), so the ranges
    /// long enough for a request are one range query away.
    free: BTreeSet<(usize, u64, u64, u64)>,
    /// Open shared blocks per memory type: gpu-allocator's `active_general_blocks`.
    open: Vec<usize>,
    /// Set once an update found the copy disagreeing with what gpu-allocator did. From then on
    /// every pooled request is charged a whole block, which never admits more than the device
    /// commits.
    lost: bool,
}

impl PoolLedger {
    pub(crate) fn new(types: Vec<vk::MemoryPropertyFlags>) -> Self {
        let open = vec![0; types.len()];
        PoolLedger {
            types,
            blocks: BTreeMap::new(),
            free: BTreeSet::new(),
            open,
            lost: false,
        }
    }

    /// The memory type gpu-allocator gives a request at `location` whose buffer accepts
    /// `type_bits`: the first accepted type with the flags the location prefers, else the first
    /// with the flags it requires.
    fn type_for(&self, location: MemoryLocation, type_bits: u32) -> Option<usize> {
        use vk::MemoryPropertyFlags as F;
        let host = F::HOST_VISIBLE | F::HOST_COHERENT;
        let (preferred, required) = match location {
            MemoryLocation::GpuOnly => (F::DEVICE_LOCAL, F::DEVICE_LOCAL),
            MemoryLocation::CpuToGpu => (host | F::DEVICE_LOCAL, host),
            MemoryLocation::GpuToCpu => (host | F::HOST_CACHED, host),
            MemoryLocation::Unknown => (F::empty(), F::empty()),
        };
        let first = |flags: F| {
            (0..self.types.len())
                .find(|&i| type_bits & (1 << i) != 0 && self.types[i].contains(flags))
        };
        first(preferred).or_else(|| first(required))
    }

    /// The memory type an allocation with `props` was placed in: gpu-allocator takes the first
    /// accepted type holding the flags it searched for, so no earlier accepted type has exactly
    /// these.
    fn type_of(&self, props: vk::MemoryPropertyFlags, type_bits: u32) -> Option<usize> {
        (0..self.types.len()).find(|&i| type_bits & (1 << i) != 0 && self.types[i] == props)
    }

    fn block_size(&self, ty: usize) -> u64 {
        if self.types[ty].contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            HOST_BLOCK
        } else {
            DEVICE_BLOCK
        }
    }

    /// What an allocation with `req` at `location` commits. `own` is an allocation that gets
    /// memory of its own (a dedicated scheme, or the unified-memory spill); a request of at least
    /// [`DEDICATED_MIN`] is one too. `live` is a device with a live budget: without one the guard
    /// counts this backend's own allocations by their size, so every request is charged that.
    pub(crate) fn commit(
        &self,
        req: &vk::MemoryRequirements,
        location: MemoryLocation,
        own: bool,
        live: bool,
    ) -> Commit {
        if !live || own || req.size >= DEDICATED_MIN {
            return Commit::Own(req.size);
        }
        self.pooled_commit(req, location)
    }

    /// What a request at `location` sub-allocated by gpu-allocator commits.
    fn pooled_commit(&self, req: &vk::MemoryRequirements, location: MemoryLocation) -> Commit {
        // No type takes the buffer: gpu-allocator refuses it, so nothing is committed and nothing
        // is lost by charging the request.
        let Some(ty) = self.type_for(location, req.memory_type_bits) else {
            return Commit::Own(req.size);
        };
        let block = self.block_size(ty);
        if req.size > block {
            return Commit::Own(req.size);
        }
        if !self.lost && self.has_room(ty, req.size, req.alignment.max(1)) {
            Commit::InOpenBlock
        } else {
            Commit::NewBlock {
                block,
                request: req.size,
            }
        }
    }

    /// Whether an open block of type `ty` has a free range that holds `size` bytes at `align`.
    /// Only ranges at least `size` long are visited, and any at least `size + align - 1` long
    /// holds it, so the walk stops at the first such range.
    fn has_room(&self, ty: usize, size: u64, align: u64) -> bool {
        self.free
            .range((ty, size, 0, 0)..(ty + 1, 0, 0, 0))
            .any(|&(_, len, start, _)| start.next_multiple_of(align) + size <= start + len)
    }

    /// Record a pooled allocation gpu-allocator placed at `offset` in the block `memory`, of type
    /// `props` (its `memory_properties`), for a buffer that accepts `type_bits`.
    pub(crate) fn allocated(
        &mut self,
        memory: u64,
        props: vk::MemoryPropertyFlags,
        type_bits: u32,
        offset: u64,
        size: u64,
    ) {
        let Some(ty) = self.type_of(props, type_bits) else {
            self.lose("an allocation in no memory type its buffer accepts");
            return;
        };
        let block_size = self.block_size(ty);
        if size > block_size {
            return;
        }
        let (free, open) = (&mut self.free, &mut self.open);
        let block = self.blocks.entry(memory).or_insert_with(|| {
            free.insert((ty, block_size, 0, memory));
            open[ty] += 1;
            Block {
                ty,
                size: block_size,
                chunks: BTreeMap::new(),
            }
        });
        if block.ty != ty {
            self.lose("an allocation in a block of another memory type");
            return;
        }
        let before = block.chunks.range(..=offset).next_back();
        if before.is_some_and(|(_, &end)| end > offset) {
            self.lose("an allocation over a live chunk");
            return;
        }
        let start = before.map_or(0, |(_, &end)| end);
        let free_end = block
            .chunks
            .range(offset..)
            .next()
            .map_or(block.size, |(&s, _)| s);
        let end = offset + size;
        if end > free_end || !self.free.remove(&(ty, free_end - start, start, memory)) {
            self.lose("an allocation past its free range");
            return;
        }
        block.chunks.insert(start, end);
        if end < free_end {
            self.free.insert((ty, free_end - end, end, memory));
        }
    }

    /// Record that the pooled allocation at `offset` in the block `memory` was freed. An
    /// allocation the ledger never recorded (one with a block of its own) is ignored.
    pub(crate) fn freed(&mut self, memory: u64, offset: u64) {
        let Some(block) = self.blocks.get_mut(&memory) else {
            return;
        };
        let ty = block.ty;
        let Some((start, end)) = block
            .chunks
            .range(..=offset)
            .next_back()
            .map(|(&s, &e)| (s, e))
            .filter(|&(_, e)| e > offset)
        else {
            self.lose("a free of no live chunk");
            return;
        };
        block.chunks.remove(&start);
        let before = block
            .chunks
            .range(..start)
            .next_back()
            .map_or(0, |(_, &e)| e);
        let after = block
            .chunks
            .range(end..)
            .next()
            .map_or(block.size, |(&s, _)| s);
        let emptied = block.chunks.is_empty();
        let mut merged = true;
        if before < start {
            merged &= self.free.remove(&(ty, start - before, before, memory));
        }
        if end < after {
            merged &= self.free.remove(&(ty, after - end, end, memory));
        }
        if !merged {
            self.lose("a free beside a free range it did not hold");
            return;
        }
        if emptied && self.open[ty] > 1 {
            self.blocks.remove(&memory);
            self.open[ty] -= 1;
        } else {
            self.free.insert((ty, after - before, before, memory));
        }
    }

    fn lose(&mut self, why: &str) {
        if !self.lost {
            tracing::warn!(
                "[infr] VRAM guard: its copy of gpu-allocator's shared blocks no longer matches \
                 them ({why}); every pooled allocation is charged a whole block from now on"
            );
            self.lost = true;
        }
    }

    /// Every open block as (size, its chunks as (offset, size) in offset order), in no particular
    /// order: gpu-allocator's report spells its blocks the same way.
    #[cfg(test)]
    pub(crate) fn layouts(&self) -> Vec<(u64, Vec<(u64, u64)>)> {
        self.blocks
            .values()
            .map(|b| {
                let chunks = b.chunks.iter().map(|(&s, &e)| (s, e - s)).collect();
                (b.size, chunks)
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn is_lost(&self) -> bool {
        self.lost
    }
}

/// gpu-allocator and the ledger of its shared blocks behind one lock: the only way to allocate
/// from gpu-allocator or free to it, so the two cannot go out of step. Must be dropped before the
/// device it allocates from is destroyed.
pub(crate) struct Pool(Mutex<PoolInner>);

struct PoolInner {
    allocator: Allocator,
    ledger: PoolLedger,
}

impl Pool {
    /// `types` is each of the device's memory types' property flags, by index.
    pub(crate) fn new(allocator: Allocator, types: Vec<vk::MemoryPropertyFlags>) -> Self {
        Pool(Mutex::new(PoolInner {
            allocator,
            ledger: PoolLedger::new(types),
        }))
    }

    /// Allocate `desc`, after `admit` has accepted what it commits ([`PoolLedger::commit`]; `live`
    /// as there). `admit` runs under the pool's lock, so no other allocation can take the free
    /// range it was told about before this one lands in it.
    pub(crate) fn allocate(
        &self,
        desc: &AllocationCreateDesc<'_>,
        live: bool,
        admit: impl FnOnce(Commit) -> Result<()>,
    ) -> Result<Allocation> {
        let mut inner = self.0.lock().unwrap();
        let own = desc.allocation_scheme != AllocationScheme::GpuAllocatorManaged;
        admit(
            inner
                .ledger
                .commit(&desc.requirements, desc.location, own, live),
        )?;
        let allocation = inner
            .allocator
            .allocate(desc)
            .map_err(|e| be(format!("gpu_allocator::allocate: {e}")))?;
        if !allocation.is_dedicated() {
            inner.ledger.allocated(
                vk::Handle::as_raw(unsafe { allocation.memory() }),
                allocation.memory_properties(),
                desc.requirements.memory_type_bits,
                allocation.offset(),
                allocation.size(),
            );
        }
        Ok(allocation)
    }

    /// Return `alloc` to gpu-allocator, and its chunk to the ledger once gpu-allocator has taken it
    /// back. A free gpu-allocator refuses leaves the chunk in both.
    pub(crate) fn free(&self, alloc: Allocation) -> std::result::Result<(), String> {
        let (memory, offset) = (
            vk::Handle::as_raw(unsafe { alloc.memory() }),
            alloc.offset(),
        );
        let mut inner = self.0.lock().unwrap();
        inner.allocator.free(alloc).map_err(|e| e.to_string())?;
        inner.ledger.freed(memory, offset);
        Ok(())
    }

    /// `VkDeviceMemory` blocks gpu-allocator holds, shared and dedicated, off its report: a walk of
    /// every live allocation, never for the allocation path.
    pub(crate) fn blocks(&self) -> u64 {
        self.0
            .lock()
            .unwrap()
            .allocator
            .generate_report()
            .blocks
            .len() as u64
    }

    /// `f` over gpu-allocator and the ledger at one instant.
    #[cfg(test)]
    pub(crate) fn inspect<R>(&self, f: impl FnOnce(&Allocator, &PoolLedger) -> R) -> R {
        let inner = self.0.lock().unwrap();
        f(&inner.allocator, &inner.ledger)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DL: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const HOST: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
            | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
    );
    const REBAR: vk::MemoryPropertyFlags =
        vk::MemoryPropertyFlags::from_raw(DL.as_raw() | HOST.as_raw());
    const ALL: u32 = 0b111;

    /// A discrete card's types as RADV lists them: plain VRAM, host memory, host-visible VRAM.
    fn ledger() -> PoolLedger {
        PoolLedger::new(vec![DL, HOST, REBAR])
    }

    fn req(size: u64, alignment: u64, bits: u32) -> vk::MemoryRequirements {
        vk::MemoryRequirements {
            size,
            alignment,
            memory_type_bits: bits,
        }
    }

    /// The charge for a `GpuOnly` request gpu-allocator sub-allocates, at any size: `commit` sends
    /// only requests below `DEDICATED_MIN` here, and the block arithmetic is clearer at block
    /// scale.
    fn pooled(l: &PoolLedger, size: u64, alignment: u64) -> Commit {
        l.pooled_commit(&req(size, alignment, ALL), MemoryLocation::GpuOnly)
    }

    const MIB: u64 = 1 << 20;

    /// A pooled request is charged nothing when an open block of its memory type has a free range
    /// that holds it at its alignment, and the whole block it opens otherwise.
    #[test]
    fn pooled_commit_charges_what_gpu_allocator_will_open() {
        let mut l = ledger();
        let new = |request| Commit::NewBlock {
            block: DEVICE_BLOCK,
            request,
        };
        // Nothing open: the first request opens a block.
        assert_eq!(pooled(&l, 2 * MIB, 256), new(2 * MIB));
        // A 100-byte chunk at the start of a block leaves its tail free from byte 100.
        l.allocated(1, DL, ALL, 0, 100);
        assert_eq!(pooled(&l, 2 * MIB, 256), Commit::InOpenBlock);
        // The tail at alignment 256 starts at 256 and ends at the block's end: a request of exactly
        // what is left fits, and one byte more, although shorter than the tail itself, does not.
        assert_eq!(pooled(&l, DEVICE_BLOCK - 256, 256), Commit::InOpenBlock);
        assert_eq!(pooled(&l, DEVICE_BLOCK - 255, 256), new(DEVICE_BLOCK - 255));
        // At alignment 4 the tail from byte 100 is all usable.
        assert_eq!(pooled(&l, DEVICE_BLOCK - 100, 4), Commit::InOpenBlock);
        assert_eq!(pooled(&l, DEVICE_BLOCK - 99, 4), new(DEVICE_BLOCK - 99));
        // A `GpuOnly` buffer that only the host type takes has no device-local type to go to:
        // gpu-allocator refuses it, so the request is charged its own size. One that only the
        // host-visible VRAM type takes goes there, and that type's block is gpu-allocator's host
        // block.
        assert_eq!(
            l.pooled_commit(&req(MIB, 256, 0b010), MemoryLocation::GpuOnly),
            Commit::Own(MIB),
            "no device-local type takes it: gpu-allocator refuses it"
        );
        assert_eq!(
            l.pooled_commit(&req(MIB, 256, 0b100), MemoryLocation::GpuOnly),
            Commit::NewBlock {
                block: HOST_BLOCK,
                request: MIB
            },
            "only the host-visible VRAM type takes it, and it has no block open"
        );
        // Once a block of the host-visible VRAM type is open, a buffer only that type takes lands
        // in it, although type 0's open block has room too.
        l.allocated(2, REBAR, 0b100, 0, MIB);
        assert_eq!(
            l.pooled_commit(&req(MIB, 256, 0b100), MemoryLocation::GpuOnly),
            Commit::InOpenBlock
        );
        // Host locations: an upload buffer prefers host-visible VRAM (type 2, whose block is now
        // open); a readback buffer finds no cached type and takes the first host-visible,
        // coherent one (type 1), which has no block open, and a host block.
        assert_eq!(
            l.pooled_commit(&req(MIB, 256, ALL), MemoryLocation::CpuToGpu),
            Commit::InOpenBlock
        );
        assert_eq!(
            l.pooled_commit(&req(MIB, 256, ALL), MemoryLocation::GpuToCpu),
            Commit::NewBlock {
                block: HOST_BLOCK,
                request: MIB
            }
        );
    }

    /// The free ranges between chunks are found as well as the tail, frees merge ranges, and an
    /// emptied block stays open (and so has room) while it is its type's last one.
    #[test]
    fn the_ledger_follows_frees_and_keeps_a_types_last_block() {
        let mut l = ledger();
        // Fill one block with four quarter-blocks.
        let q = DEVICE_BLOCK / 4;
        for i in 0..4 {
            l.allocated(1, DL, ALL, i * q, q);
        }
        assert_eq!(
            pooled(&l, MIB, 256),
            Commit::NewBlock {
                block: DEVICE_BLOCK,
                request: MIB
            }
        );
        // Freeing the second quarter opens a hole exactly a quarter long.
        l.freed(1, q);
        assert_eq!(pooled(&l, q, 256), Commit::InOpenBlock);
        assert_eq!(
            pooled(&l, q + 1, 1),
            Commit::NewBlock {
                block: DEVICE_BLOCK,
                request: q + 1
            }
        );
        // Freeing the third merges it with the hole: half a block.
        l.freed(1, 2 * q);
        assert_eq!(pooled(&l, 2 * q, 256), Commit::InOpenBlock);
        // A second block, then both emptied: the first emptied is destroyed, the last one kept.
        l.allocated(2, DL, ALL, 0, DEVICE_BLOCK);
        l.freed(1, 0);
        l.freed(1, 3 * q);
        assert_eq!(l.layouts().len(), 1, "block 1 emptied with block 2 open");
        l.freed(2, 0);
        assert_eq!(
            l.layouts(),
            vec![(DEVICE_BLOCK, vec![])],
            "the type's last block stays open, empty"
        );
        assert_eq!(pooled(&l, DEVICE_BLOCK, 256), Commit::InOpenBlock);
        // A chunk whose alignment padded it: it owns its padding, so freeing it and allocating
        // again sees the same ranges.
        l.allocated(2, DL, ALL, 0, 100);
        l.allocated(2, DL, ALL, 256, 100);
        assert_eq!(
            l.layouts(),
            vec![(DEVICE_BLOCK, vec![(0, 100), (100, 256)])],
            "the second chunk starts where the free range did"
        );
        l.freed(2, 0);
        assert_eq!(l.layouts(), vec![(DEVICE_BLOCK, vec![(100, 256)])]);
        // A free of an allocation never recorded (one with a block of its own) changes nothing.
        l.freed(99, 0);
        assert!(!l.is_lost());
    }

    /// The class boundaries: a request of `DEDICATED_MIN` or more, one that has memory of its own,
    /// and any request on a device without a live budget are charged their size; below
    /// `DEDICATED_MIN` a pooled request is charged by the block it lands in, and the guard checks a
    /// block it opens even when the request itself is below `VRAM_GUARD_MIN_ALLOC`.
    #[test]
    fn the_commit_rule_at_the_class_boundaries() {
        let mut l = ledger();
        l.allocated(1, DL, ALL, 0, 100);
        let at =
            |size, own, live| l.commit(&req(size, 256, ALL), MemoryLocation::GpuOnly, own, live);
        assert_eq!(at(DEDICATED_MIN, false, true), Commit::Own(DEDICATED_MIN));
        assert_eq!(at(DEDICATED_MIN - 1, false, true), Commit::InOpenBlock);
        assert_eq!(at(MIB, true, true), Commit::Own(MIB));
        assert_eq!(at(MIB, false, false), Commit::Own(MIB));
        // The guard's own threshold.
        assert_eq!(Commit::InOpenBlock.checked(), None);
        assert_eq!(Commit::Own(VRAM_GUARD_MIN_ALLOC - 1).checked(), None);
        assert_eq!(
            Commit::Own(VRAM_GUARD_MIN_ALLOC).checked(),
            Some(VRAM_GUARD_MIN_ALLOC)
        );
        let small = VRAM_GUARD_MIN_ALLOC - 1;
        let opens = PoolLedger::new(vec![DL]).commit(
            &req(small, 256, 1),
            MemoryLocation::GpuOnly,
            false,
            true,
        );
        assert_eq!(
            opens,
            Commit::NewBlock {
                block: DEVICE_BLOCK,
                request: small
            }
        );
        assert_eq!(opens.checked(), Some(DEVICE_BLOCK));
    }

    /// The guard's decision (`admits`, which `VulkanBackend::check_vram_commit` makes) sees what
    /// is committed: inside an open block a request is admitted with less room left than its size
    /// (even with none) and without reading the budget; a request that opens a block is admitted
    /// only when the whole block fits, so with room between the request and the block it is
    /// refused. The device tests `a_pooled_buffer_in_an_open_block_is_admitted_below_its_size` and
    /// `a_pooled_buffer_that_opens_a_block_is_refused_by_the_guard` drive the same decision
    /// through `make_buf_ex`.
    #[test]
    fn a_commit_is_admitted_by_what_it_opens() {
        let budget = 8 * DEVICE_BLOCK;
        let request = 2 * MIB;
        let admitted = |c: Commit, room: u64| admits(c, || (budget - room, budget));
        let unread = || -> (u64, u64) { panic!("an unchecked commitment read the budget") };
        assert!(admits(Commit::InOpenBlock, unread));
        assert!(admits(Commit::Own(VRAM_GUARD_MIN_ALLOC - 1), unread));
        assert!(admitted(Commit::InOpenBlock, request - 1));
        assert!(admitted(Commit::InOpenBlock, 0));
        let new = Commit::NewBlock {
            block: DEVICE_BLOCK,
            request,
        };
        assert!(admitted(new, DEVICE_BLOCK));
        assert!(!admitted(new, DEVICE_BLOCK - 1));
        assert!(
            !admitted(new, request),
            "room for the request but not its block"
        );
        assert!(admitted(Commit::Own(DEDICATED_MIN), DEDICATED_MIN));
        assert!(!admitted(Commit::Own(DEDICATED_MIN), DEDICATED_MIN - 1));
        assert!(!fits_budget(u64::MAX, 1, budget), "saturates, never wraps");
    }

    /// A ledger that finds itself out of step with gpu-allocator stops admitting anything into an
    /// open block: every pooled request is charged a block from then on.
    #[test]
    fn a_ledger_that_loses_track_charges_a_block() {
        let mut l = ledger();
        l.allocated(1, DL, ALL, 0, MIB);
        assert_eq!(pooled(&l, MIB, 256), Commit::InOpenBlock);
        // An allocation on top of a live chunk cannot have come from gpu-allocator.
        l.allocated(1, DL, ALL, MIB / 2, MIB);
        assert!(l.is_lost());
        assert_eq!(
            pooled(&l, MIB, 256),
            Commit::NewBlock {
                block: DEVICE_BLOCK,
                request: MIB
            }
        );
    }
}

/// The resolved gpu-allocator is the release the mirrored rules were read in: no GPU needed, so a
/// version bump fails here, in every CI run, before any device test could be skipped.
#[cfg(test)]
mod version_tests {
    /// Every version Cargo.lock resolves for gpu-allocator.
    fn resolved_gpu_allocator() -> Vec<String> {
        let lock = include_str!("../../../Cargo.lock");
        lock.split("[[package]]")
            .filter_map(|pkg| {
                let mut name = None;
                let mut version = None;
                for line in pkg.lines() {
                    if let Some(v) = line.strip_prefix("name = ") {
                        name = Some(v.trim_matches('"'));
                    } else if let Some(v) = line.strip_prefix("version = ") {
                        version = Some(v.trim_matches('"'));
                    }
                }
                (name == Some("gpu-allocator")).then(|| version.map(str::to_owned))?
            })
            .collect()
    }

    #[test]
    fn the_mirror_is_of_the_resolved_gpu_allocator() {
        let versions = resolved_gpu_allocator();
        let [resolved] = &versions[..] else {
            panic!(
                "Cargo.lock resolves gpu-allocator {versions:?}, where `pool`'s ledger mirrors \
                 exactly one, {}: the ledger cannot tell which one allocated a block",
                super::MIRRORED_GPU_ALLOCATOR
            );
        };
        assert_eq!(
            resolved,
            super::MIRRORED_GPU_ALLOCATOR,
            "gpu-allocator resolves to {resolved}, but `pool`'s ledger copies the rules of {}: \
             re-read its block sizes (`AllocationSizes::default` and whether `Allocator::new` \
             honours the create info), its memory type choice, `FreeListAllocator`'s placement and \
             merging and `MemoryType::free`'s rule for an emptied block in the new source, update \
             the ledger and the module documentation to match, run \
             `ledger_matches_gpu_allocators_blocks` and \
             `predicted_commits_are_what_gpu_allocator_opens` on a GPU, then update \
             `MIRRORED_GPU_ALLOCATOR`",
            super::MIRRORED_GPU_ALLOCATOR,
        );
    }
}

/// The ledger and the guard on a device. GPU tests: run serially, one device at a time, since each
/// reads the process's memory budget.
#[cfg(test)]
mod gpu_tests {
    use std::sync::Arc;

    use infr_core::backend::BufferUsage;
    use infr_core::SizeSpec;

    use super::*;
    use crate::{test_config, VulkanBackend, GUARD_HEADROOM};

    /// The location `make_alloc` gives each pooled usage the tests allocate.
    fn location(usage: BufferUsage) -> MemoryLocation {
        match usage {
            BufferUsage::Activations => MemoryLocation::GpuOnly,
            BufferUsage::Staging => MemoryLocation::CpuToGpu,
            BufferUsage::Readback => MemoryLocation::GpuToCpu,
            other => panic!("{other:?} is not a pooled usage these tests allocate"),
        }
    }

    /// What the ledger predicts a `usage` buffer of `size` bytes commits now: its requirements
    /// come from a buffer created as `make_buf_ex` creates one, then destroyed.
    fn commit_of(be: &VulkanBackend, size: usize, usage: BufferUsage) -> Commit {
        let ci = vk::BufferCreateInfo::default()
            .size(crate::fill_span(size))
            .usage(crate::BUFFER_USAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let device = &be.shared.device;
        let buffer = unsafe { device.create_buffer(&ci, None) }.expect("create_buffer");
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        unsafe { device.destroy_buffer(buffer, None) };
        be.shared
            .pool
            .inspect(|_, l| l.commit(&req, location(usage), false, true))
    }

    /// Blocks gpu-allocator's own report lists, shared and dedicated.
    fn reported_blocks(be: &VulkanBackend) -> usize {
        be.shared
            .pool
            .inspect(|a, _| a.generate_report().blocks.len())
    }

    /// A backend whose `kernels.vulkan.vram_limit` leaves its guard `room` bytes above what the
    /// process used before it was built (its own construction takes some of that), or `None`,
    /// saying so, on a device without a live budget, where the guard charges every request its
    /// size.
    fn limited(room: u64) -> Option<VulkanBackend> {
        let probe = VulkanBackend::new_with(Arc::new(test_config())).expect("VulkanBackend");
        let v = probe.vram();
        drop(probe);
        if !v.live {
            eprintln!("skip: no VK_EXT_memory_budget, so no block is charged whole");
            return None;
        }
        let mut cfg = test_config();
        let used = v.total - v.available;
        cfg.kernels.vulkan.vram_limit = Some(SizeSpec::Bytes(used + GUARD_HEADROOM + room));
        Some(VulkanBackend::new_with(Arc::new(cfg)).expect("VulkanBackend under a limit"))
    }

    /// A pooled `Activations` size the guard checks: at least `VRAM_GUARD_MIN_ALLOC`, below
    /// `DEDICATED_MIN`.
    const REQUEST: u64 = DEDICATED_MIN / 4;

    /// Take all but `leave` bytes of `be`'s room with one allocation of its own, and return the
    /// room left as the guard sees it.
    fn fill_to(be: &VulkanBackend, leave: u64) -> (crate::VkBuffer, u64) {
        let room = be.alloc_room();
        let filler = room
            .checked_sub(leave)
            .expect("room above what the test leaves");
        assert!(
            filler >= DEDICATED_MIN,
            "the filler ({filler} B of {room} B of room) must be an allocation of its own"
        );
        let buf = be
            .make_alloc(filler as usize, BufferUsage::Activations)
            .expect("filler");
        (buf, be.alloc_room())
    }

    /// Pooled sizes on and off alignment boundaries, up to the largest whose requirements no
    /// alignment rounds up to the dedicated class.
    const SIZES: [usize; 8] = [
        4,
        300,
        4096,
        100_000,
        1 << 20,
        (3 << 20) + 12,
        REQUEST as usize,
        (DEDICATED_MIN - (64 << 10)) as usize,
    ];

    /// Every pooled usage, round-robin over rounds and sizes.
    fn usage_at(round: usize, i: usize) -> BufferUsage {
        match (round + i) % 4 {
            0 | 1 => BufferUsage::Activations,
            2 => BufferUsage::Staging,
            _ => BufferUsage::Readback,
        }
    }

    /// Drop every third buffer, so later rounds land in the holes and frees merge ranges.
    fn free_every_third(live: &mut Vec<crate::VkBuffer>) {
        let mut i = 0;
        live.retain(|_| {
            i += 1;
            i % 3 != 0
        });
    }

    /// The ledger is gpu-allocator's own layout: after allocations and frees of pooled buffers
    /// in every pooled memory location, at sizes on and off alignment boundaries, every shared
    /// block gpu-allocator reports holds exactly the chunks the ledger has for one of its blocks,
    /// and the other way round, down to a type's last block kept open empty.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn ledger_matches_gpu_allocators_blocks() {
        let be = VulkanBackend::new_with(Arc::new(test_config())).expect("VulkanBackend");
        let compare = |when: &str| {
            be.shared.pool.inspect(|allocator, ledger| {
                let report = allocator.generate_report();
                assert!(!ledger.is_lost(), "{when}: the ledger lost track");
                let mut theirs: Vec<(u64, Vec<(u64, u64)>)> = report
                    .blocks
                    .iter()
                    .filter(|b| b.size == DEVICE_BLOCK || b.size == HOST_BLOCK)
                    .map(|b| {
                        let mut chunks: Vec<(u64, u64)> = report.allocations[b.allocations.clone()]
                            .iter()
                            .map(|a| (a.offset, a.size))
                            .collect();
                        chunks.sort_unstable();
                        (b.size, chunks)
                    })
                    // A block of one allocation filling it is a dedicated one of that size.
                    .filter(|(size, chunks)| chunks[..] != [(0, *size)])
                    .collect();
                let mut ours = ledger.layouts();
                theirs.sort_unstable();
                ours.sort_unstable();
                assert_eq!(
                    ours, theirs,
                    "{when}: the ledger's blocks against gpu-allocator's"
                );
                ours.iter().map(|(_, c)| c.len()).sum::<usize>()
            })
        };
        let mut live = Vec::new();
        for round in 0..4 {
            for (i, &size) in SIZES.iter().enumerate() {
                live.push(
                    be.make_alloc(size, usage_at(round, i))
                        .expect("pooled alloc"),
                );
            }
            free_every_third(&mut live);
        }
        let chunks = compare("with buffers live");
        assert!(
            chunks >= live.len(),
            "compared {chunks} chunks for {} buffers",
            live.len()
        );
        drop(live);
        compare("with every buffer freed");
        assert!(
            be.shared.pool.inspect(|_, l| !l.layouts().is_empty()),
            "a type's last block stays open"
        );
    }

    /// The guard's central prediction: before every pooled allocation, in every pooled memory
    /// location, at sizes on and off alignment boundaries and across frees that leave holes and
    /// merge ranges, gpu-allocator's own report gains a block exactly when the ledger predicted a
    /// new one, and none when it predicted the buffer lands in an open block. Both outcomes must
    /// occur in a device-local type and in a host-visible one.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn predicted_commits_are_what_gpu_allocator_opens() {
        let be = VulkanBackend::new_with(Arc::new(test_config())).expect("VulkanBackend");
        let mut live = Vec::new();
        // (memory location, opened a block) pairs seen.
        let mut seen = std::collections::BTreeSet::new();
        for round in 0..12 {
            for (i, &size) in SIZES.iter().enumerate() {
                let usage = usage_at(round, i);
                let predicted = commit_of(&be, size, usage);
                let before = reported_blocks(&be);
                live.push(be.make_alloc(size, usage).expect("pooled alloc"));
                let opened = reported_blocks(&be) - before;
                let new_block = matches!(predicted, Commit::NewBlock { .. });
                assert!(
                    matches!(predicted, Commit::InOpenBlock | Commit::NewBlock { .. }),
                    "round {round}: a pooled {size} B {usage:?} buffer predicted {predicted:?}"
                );
                assert_eq!(
                    opened,
                    usize::from(new_block),
                    "round {round}: a {size} B {usage:?} buffer predicted {predicted:?}, and \
                     gpu-allocator opened {opened} blocks"
                );
                seen.insert((format!("{:?}", location(usage)), new_block));
            }
            free_every_third(&mut live);
        }
        // A device-local type (GpuOnly) and a host-visible one (CpuToGpu) both open blocks and
        // land in them; readback buffers share a host block construction already opened.
        for (loc, outcomes) in [
            ("GpuOnly", &[false, true][..]),
            ("CpuToGpu", &[false, true][..]),
            ("GpuToCpu", &[false][..]),
        ] {
            for &new_block in outcomes {
                assert!(
                    seen.contains(&(loc.to_string(), new_block)),
                    "no {loc} allocation {} a block: {seen:?}",
                    if new_block { "opened" } else { "landed in" }
                );
            }
        }
    }

    /// The false refusal: a pooled buffer that fits in the free space of an open block, with less
    /// room left under the budget than its own size, commits nothing and is admitted. Charged its
    /// size, as the guard once did, it was refused.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn a_pooled_buffer_in_an_open_block_is_admitted_below_its_size() {
        let Some(be) = limited(4 * DEVICE_BLOCK) else {
            return;
        };
        let first = be
            .make_alloc(REQUEST as usize, BufferUsage::Activations)
            .expect("first pooled buffer");
        assert_eq!(
            commit_of(&be, REQUEST as usize, BufferUsage::Activations),
            Commit::InOpenBlock,
            "the block the first buffer is in has room for a second"
        );
        let (_filler, left) = fill_to(&be, REQUEST / 2);
        eprintln!("room left {left} B for a pooled {REQUEST} B buffer");
        assert!(
            left < REQUEST,
            "the room ({left} B) must be below the request"
        );
        let blocks = reported_blocks(&be);
        let second = be
            .make_alloc(REQUEST as usize, BufferUsage::Activations)
            .expect("a pooled buffer inside an open block commits nothing");
        assert_eq!(
            reported_blocks(&be),
            blocks,
            "gpu-allocator opened a block for the second buffer"
        );
        drop((first, second));
    }

    /// The mirror: a pooled buffer that no open block has room for opens a whole block, so with
    /// room for the buffer but not for the block, the guard refuses it, naming the block and the
    /// limit, before gpu-allocator asks the driver for anything.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn a_pooled_buffer_that_opens_a_block_is_refused_by_the_guard() {
        let Some(be) = limited(4 * DEVICE_BLOCK) else {
            return;
        };
        // Fill the open blocks of the type until the next request would open one.
        let mut held = Vec::new();
        let block = loop {
            match commit_of(&be, REQUEST as usize, BufferUsage::Activations) {
                Commit::NewBlock { block, request } => {
                    assert_eq!(request, REQUEST);
                    break block;
                }
                Commit::InOpenBlock => {
                    assert!(
                        (held.len() as u64) < DEVICE_BLOCK / REQUEST + 1,
                        "more pooled buffers than a block holds, and still room"
                    );
                    held.push(
                        be.make_alloc(REQUEST as usize, BufferUsage::Activations)
                            .expect("pooled buffer into an open block"),
                    );
                }
                other => panic!("a pooled request charged {other:?}"),
            }
        };
        let (_filler, left) = fill_to(&be, (REQUEST + block) / 2);
        eprintln!(
            "room left {left} B for a pooled {REQUEST} B buffer that opens a {block} B block"
        );
        assert!(
            (REQUEST..block).contains(&left),
            "the room ({left} B) must hold the request and not its block"
        );
        let blocks = reported_blocks(&be);
        let refused = be
            .make_alloc(REQUEST as usize, BufferUsage::Activations)
            .err()
            .expect("the guard must refuse a block the budget cannot hold")
            .to_string();
        assert!(
            refused.contains("budget exceeded") && refused.contains("the new shared block"),
            "refused by the guard, naming the block: {refused}"
        );
        assert!(refused.contains("INFR_VRAM_LIMIT"), "{refused}");
        assert_eq!(reported_blocks(&be), blocks, "gpu-allocator opened a block");
    }
}
