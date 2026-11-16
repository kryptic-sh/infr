//! Host-paged weights for the CPU backend: weights read from the model file into a bounded DRAM
//! arena instead of being mapped and left to the OS page cache
//! (`docs/disk-streaming-plan.md` §3.6).
//!
//! # Why pools
//! [`infr_core::hostpager::HostPager`] holds UNIFORM slots, and a model's tensors are anything but
//! — a norm is kilobytes, an lm_head is gigabytes. One arena sized to the largest would spend most
//! of its budget on padding. So this keeps one pager per distinct BLOCK SIZE. That is not an
//! arbitrary bucketing: a transformer repeats the same layer shape, so every layer's `attn_q` is
//! byte-identical in size to every other's, and a dense model lands in a handful of classes with
//! no padding at all.
//!
//! # Why the pools are planned before the load
//! Every pool exists before the first weight is bound, sized from the tensor directory the GGUF
//! already carries. Creating them lazily instead would mean growing the pool list while bound
//! buffers hold references into it, and it would hand the whole budget to whichever size class
//! happened to be bound first — the last layer's classes appear last in a load, so first-come is
//! the wrong order to spend a budget in.
//!
//! # Expert banks are paged per expert
//! A routed-expert bank holds every expert of a layer, and a forward pass reads only the few the
//! router picks. Paged as one block, a bank drags all of its experts in to serve those few: on
//! DeepSeek V4-Flash that was a 0.7 GB read per bank to use 2.75 MB of it per routed expert. So a
//! bank registered with [`PagedWeights::register_experts`] becomes one block per expert, in a pool
//! of experts, and `Op::MoeFfn` pins just the routed ones once routing has picked them.
//!
//! # How the budget is split
//! By what a pass reads, not by what the file holds: every byte of a whole weight is read on every
//! pass, so a slot of it saves a read on every pass, while an expert is read only when routed.
//! Whole weights are seated first and the expert pools take what is left — see [`plan_pools`].
//!
//! # What stays mapped
//! Weights below [`MIN_PAGED_BYTES`], and any class the budget cannot seat. Those keep the
//! zero-copy mmap path, so a budget too small to be useful degrades to today's behaviour instead
//! of failing the load.

use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo};
use infr_core::error::{Error, Result};
use infr_core::graph::WeightReuse;
use infr_core::hostpager::{HostPager, HostPagerStats, Pin};
use infr_core::pager::trace::TraceFile;
use infr_core::pager::Insert;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Smallest weight worth paging. Below this the arena bookkeeping outweighs the bytes: a 1 MiB
/// floor keeps norms, biases and rope tables mapped (a few MB across a whole model) while covering
/// every projection and expert bank, which are what a forward pass streams.
pub const MIN_PAGED_BYTES: usize = 1 << 20;

/// How one forward pass reads a weight, which is what its class's share of the budget follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reads {
    /// Every pass reads all of it: a projection, the output head.
    Whole,
    /// An expert bank of `n_expert` equal slices, of which each row of a pass reads the `n_used`
    /// its router picks. `layer` names the `Op::MoeFfn` that reads it: one op pins the routed
    /// experts of all its banks together, so the banks sharing a `layer` are what one op holds at
    /// once.
    Experts {
        n_expert: usize,
        n_used: usize,
        layer: usize,
    },
}

/// One weight the planner may page: its size and how a pass reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pageable {
    pub nbytes: usize,
    pub reads: Reads,
}

impl Pageable {
    /// The pool this weight's blocks belong to, or `None` below [`MIN_PAGED_BYTES`]. A bank whose
    /// bytes do not split evenly into its experts cannot be cut into equal blocks, so it is paged
    /// whole — the same decision [`PagedWeights::register_experts`] makes when it declines one.
    fn class(&self) -> Option<ClassKey> {
        if self.nbytes < MIN_PAGED_BYTES {
            return None;
        }
        Some(match self.reads {
            Reads::Experts { n_expert, .. } if splits(self.nbytes, n_expert) => ClassKey {
                slot_bytes: self.nbytes / n_expert,
                experts: true,
            },
            _ => ClassKey {
                slot_bytes: self.nbytes,
                experts: false,
            },
        })
    }
}

/// Whether a bank of `nbytes` cuts into `n_expert` equal blocks.
fn splits(nbytes: usize, n_expert: usize) -> bool {
    n_expert > 1 && nbytes.is_multiple_of(n_expert)
}

/// What makes two blocks share a pool: the same size, and the same way of being read. A whole
/// weight and one expert of equal size still get separate pools, because their eviction policies
/// differ (see [`PagedWeights::pin`] and [`PagedWeights::pin_expert`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ClassKey {
    slot_bytes: usize,
    experts: bool,
}

/// Handle to one registered paged weight — what a [`crate::CpuBuffer`] carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedId {
    pool: usize,
    block: infr_core::pager::BlockId,
}

/// Handle to one expert bank registered per expert: expert `e` is block `first + e` of `pool`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedExperts {
    pool: usize,
    first: infr_core::pager::BlockId,
    n_expert: u32,
}

impl PagedExperts {
    pub fn n_expert(&self) -> usize {
        self.n_expert as usize
    }
}

/// Where an op that routes to `routed` of a bank's `n_expert` experts puts the ones it pins.
///
/// - **A decode step, or any op that reads less than half of a bank: [`Insert::Mru`].** Routing is
///   skewed and carries over from one token to the next, so the expert used last is among the
///   likeliest to be used again; a cold-end insert would keep the first experts ever loaded for
///   good and stream every other one through a single slot.
/// - **A prefill chunk that routes to half or more of a bank: [`Insert::Cold`].** Such a chunk
///   reads nearly every expert of every layer in layer order, the cyclic sweep plan 06 describes
///   for dense weights, and recency is the pathological order for it: with a pool smaller than
///   all experts, each layer evicts the experts the next chunk will need first, and the chunk
///   also flushes the decode working set. At the cold end the resident set stays put, hits on it
///   do not move it, and the chunk's misses stream through the coldest slots.
///
/// - **Any op of a graph whose weights are read again next ([`WeightReuse::Again`] or
///   [`WeightReuse::Done`]): [`Insert::Mru`].** A layer-major prefill runs one layer for every
///   chunk of the prompt in turn, so what a chunk brings in is what the next chunk of the same
///   layer needs: at the cold end it would be the next chunk's first victims, read again per
///   chunk. At the hot end the victims are older layers' experts, which that layer's last chunk
///   demoted ([`PagedWeights::demote_experts`]).
///
/// Measured with `a_prefill_sweep_keeps_its_resident_set` (synthetic, counters): see its doc.
pub fn expert_insert(routed: usize, n_expert: usize, reuse: WeightReuse) -> Insert {
    if reuse != WeightReuse::Sweep {
        Insert::Mru
    } else if 2 * routed >= n_expert {
        Insert::Cold
    } else {
        Insert::Mru
    }
}

/// How many experts of each bank a layer's last layer-major chunk keeps out of its demotion
/// ([`done_keep`]). The decode after the prompt routes a few experts per layer (six on V4-Flash).
/// Sized so every layer's kept share of V4-Flash fits the slots a 12 GB arena has beyond one
/// layer's floor, and so survives the prefill (`the_done_share_fits_a_12g_arena` checks it).
pub const DONE_KEEP_EXPERTS: usize = 8;

/// The experts a layer-major layer's last chunk ([`WeightReuse::Done`]) keeps resident for the
/// decode that follows: the ones its LAST row routed to (the decode's first token continues
/// from that row), then the ones most of its rows routed to (`by_count`, expert and row count),
/// up to [`DONE_KEEP_EXPERTS`]. Without it the demotions leave whatever layers happened to be
/// demoted last, and the first decode pass reads most of its experts from disk again.
pub fn done_keep(last_row: &[usize], by_count: &[(usize, usize)]) -> Vec<usize> {
    let mut keep: Vec<usize> = Vec::with_capacity(DONE_KEEP_EXPERTS);
    for &e in last_row {
        if keep.len() < DONE_KEEP_EXPERTS && !keep.contains(&e) {
            keep.push(e);
        }
    }
    let mut ranked = by_count.to_vec();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (e, _) in ranked {
        if keep.len() >= DONE_KEEP_EXPERTS {
            break;
        }
        if !keep.contains(&e) {
            keep.push(e);
        }
    }
    keep
}

/// Where a whole weight a graph pins goes in the recency order: the cold end for an ordinary
/// forward pass (see [`PagedWeights::pin`]), the hot end when the next graphs read it again
/// ([`WeightReuse::Again`], [`WeightReuse::Done`]) — the same reasoning as [`expert_insert`].
pub fn whole_insert(reuse: WeightReuse) -> Insert {
    match reuse {
        WeightReuse::Sweep => Insert::Cold,
        WeightReuse::Again | WeightReuse::Done => Insert::Mru,
    }
}

/// One planned class: `n_slots` resident blocks of `slot_bytes` each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolPlan {
    pub slot_bytes: usize,
    pub n_slots: usize,
    /// How many blocks of this size the model has — the ceiling on useful slots.
    pub n_blocks: usize,
    /// Whether the blocks are single experts of a bank rather than whole weights.
    pub experts: bool,
}

/// One class while it is being planned.
struct Class {
    key: ClassKey,
    n_blocks: usize,
    /// The share of the class's blocks one pass reads, as `(numerator, denominator)`: `1/1` for a
    /// whole weight, `n_used/n_expert` for an expert. A fraction, so that ranking two classes is
    /// exact.
    share: (usize, usize),
    /// For an expert class, how many of its banks each layer has: one op pins all of a layer's
    /// banks together, so the floor is `n_expert` times the most any layer has.
    banks_per_layer: HashMap<usize, usize>,
    /// The slots one op pins at once from this class, below which it cannot serve that op.
    floor: usize,
}

/// Decide the pools for a model's pageable weights within `budget_bytes`.
///
/// One class per block size and read pattern (see [`Pageable`]). The budget follows what a pass
/// reads per byte cached:
///
/// 1. Classes are ranked by the share of their blocks one pass reads — every whole weight (all of
///    it, every pass) ahead of every expert pool (`n_used` of `n_expert` per row) — then by total
///    bytes.
/// 2. In that order each class gets its floor: one block for a whole weight, and for experts every
///    block one `Op::MoeFfn` may pin at once from it — `n_expert` for each of the class's banks
///    that one layer has, since a prefill chunk can route to every expert. A class whose floor no
///    longer fits is dropped and stays mapped.
/// 3. In the same order each seated class is topped up to all of its blocks while budget is left.
///
/// Whole weights fill first because a slot of one saves a read on every pass, while a slot of an
/// expert saves one only on the passes that route to it, and no expert is read twice in a pass.
/// Splitting by file share instead hands most of a MoE model's budget to its expert banks — on
/// DeepSeek V4-Flash 92% — and leaves the dense projections, which every token reads, to be read
/// from disk again on every token.
pub fn plan_pools(budget_bytes: usize, weights: &[Pageable]) -> Vec<PoolPlan> {
    let mut classes: HashMap<ClassKey, Class> = HashMap::new();
    for w in weights {
        let Some(key) = w.class() else { continue };
        let share = match w.reads {
            Reads::Experts {
                n_expert, n_used, ..
            } if key.experts => (n_used.min(n_expert), n_expert),
            _ => (1, 1),
        };
        let c = classes.entry(key).or_insert(Class {
            key,
            n_blocks: 0,
            share,
            banks_per_layer: HashMap::new(),
            floor: 1,
        });
        c.n_blocks += w.nbytes / key.slot_bytes;
        // Banks of one slice size can differ in routing width between layers; the pool has to
        // serve the widest read share and the largest floor among them.
        if share.0 * c.share.1 > c.share.0 * share.1 {
            c.share = share;
        }
        if let Reads::Experts {
            n_expert, layer, ..
        } = w.reads
        {
            if key.experts {
                let banks = c.banks_per_layer.entry(layer).or_insert(0);
                *banks += 1;
                c.floor = c.floor.max(n_expert * *banks);
            }
        }
    }
    let mut order: Vec<Class> = classes.into_values().collect();
    // Largest read share first (cross-multiplied, so exact), then total bytes, then the key, so
    // that the same model and budget give the same plan whatever order the tensors were listed in.
    order.sort_unstable_by(|a, b| {
        (b.share.0 * a.share.1)
            .cmp(&(a.share.0 * b.share.1))
            .then((b.key.slot_bytes * b.n_blocks).cmp(&(a.key.slot_bytes * a.n_blocks)))
            .then(b.key.cmp(&a.key))
    });
    let mut left = budget_bytes;
    let mut slots = vec![0usize; order.len()];
    for (c, n) in order.iter().zip(slots.iter_mut()) {
        let floor = c.floor.clamp(1, c.n_blocks);
        if floor * c.key.slot_bytes <= left {
            *n = floor;
            left -= floor * c.key.slot_bytes;
        }
    }
    for (c, n) in order.iter().zip(slots.iter_mut()) {
        if *n == 0 {
            continue;
        }
        let more = (c.n_blocks - *n).min(left / c.key.slot_bytes);
        *n += more;
        left -= more * c.key.slot_bytes;
    }
    order
        .iter()
        .zip(slots)
        .filter(|&(_, n)| n > 0)
        .map(|(c, n_slots)| PoolPlan {
            slot_bytes: c.key.slot_bytes,
            n_slots,
            n_blocks: c.n_blocks,
            experts: c.key.experts,
        })
        .collect()
}

/// One pool's activity, for the paging report.
#[derive(Clone, Copy, Debug)]
pub struct PoolReport {
    pub slot_bytes: usize,
    pub n_slots: usize,
    /// Whether the pool holds single experts rather than whole weights.
    pub experts: bool,
    pub stats: HostPagerStats,
}

/// The CPU backend's host weight cache: one [`HostPager`] per planned class.
pub struct PagedWeights {
    pools: Vec<HostPager>,
    /// Each pool's class, by pool index. Fixed at construction.
    keys: Vec<ClassKey>,
    /// Per-pool next block id. Registration is load-time only, but it takes `&self` so that bound
    /// buffers can already hold an `Arc` of this store.
    next_block: Mutex<Vec<u32>>,
    /// Wall time callers spent waiting for weights to become resident, in nanoseconds: what the
    /// tier cost the forward passes, which the reads' summed latency overstates when they overlap.
    wait_ns: AtomicU64,
}

impl PagedWeights {
    /// Build the pools named by [`plan_pools`]. Allocates every arena up front, so a budget that
    /// cannot be committed fails here rather than part-way through a generation. With `trace`
    /// (`paging.trace`), every pool records its residency calls there.
    pub fn new(
        plans: &[PoolPlan],
        io: Arc<dyn BlockIo>,
        trace: Option<&Arc<TraceFile>>,
    ) -> Result<Self> {
        let mut pools = Vec::with_capacity(plans.len());
        let mut keys = Vec::with_capacity(plans.len());
        for p in plans {
            let pool = HostPager::new(p.n_slots, p.slot_bytes, Arc::clone(&io))?;
            if let Some(t) = trace {
                let what = if p.experts { "expert" } else { "whole" };
                pool.trace_to(t, format!("cpu {what} {}B", p.slot_bytes), p.n_blocks)?;
            }
            pools.push(pool);
            keys.push(ClassKey {
                slot_bytes: p.slot_bytes,
                experts: p.experts,
            });
        }
        Ok(Self {
            next_block: Mutex::new(vec![0; pools.len()]),
            pools,
            keys,
            wait_ns: AtomicU64::new(0),
        })
    }

    fn pool_of(&self, key: ClassKey) -> Option<usize> {
        self.keys.iter().position(|&k| k == key)
    }

    /// Reserve `n` consecutive block ids in `pool`.
    fn take_ids(&self, pool: usize, n: u32) -> u32 {
        let mut next = self.next_block.lock().unwrap();
        let first = next[pool];
        next[pool] += n;
        first
    }

    /// Register one weight's file ranges, or decline it.
    ///
    /// `Ok(None)` means this weight has no pool (too small, or its class did not fit the budget) —
    /// the caller maps it instead.
    pub fn register(&self, ranges: &[(u64, usize)]) -> Result<Option<PagedId>> {
        let nbytes: usize = ranges.iter().map(|(_, l)| l).sum();
        let Some(pool) = self.pool_of(ClassKey {
            slot_bytes: nbytes,
            experts: false,
        }) else {
            return Ok(None);
        };
        let block = self.take_ids(pool, 1);
        self.pools[pool].register(BlockDesc {
            id: block,
            extents: extents(ranges),
        })?;
        Ok(Some(PagedId { pool, block }))
    }

    /// Register an expert bank of `n_expert` equal slices as one block per expert, or decline it.
    ///
    /// `Ok(None)` means there is no expert pool for this bank — it does not split evenly, or its
    /// class did not fit the budget. The caller then offers it to [`Self::register`] whole, which
    /// finds the pool [`plan_pools`] gave an uneven bank, or declines it too.
    pub fn register_experts(
        &self,
        ranges: &[(u64, usize)],
        n_expert: usize,
    ) -> Result<Option<PagedExperts>> {
        let nbytes: usize = ranges.iter().map(|(_, l)| l).sum();
        if !splits(nbytes, n_expert) {
            return Ok(None);
        }
        let stride = nbytes / n_expert;
        let Some(pool) = self.pool_of(ClassKey {
            slot_bytes: stride,
            experts: true,
        }) else {
            return Ok(None);
        };
        let n = u32::try_from(n_expert)
            .map_err(|_| Error::backend(format!("{n_expert} experts do not fit a block id")))?;
        let first = self.take_ids(pool, n);
        let all = extents(ranges);
        for e in 0..n_expert {
            self.pools[pool].register(BlockDesc {
                id: first + e as u32,
                extents: sub_extents(&all, e * stride, stride),
            })?;
        }
        Ok(Some(PagedExperts {
            pool,
            first,
            n_expert: n,
        }))
    }

    /// Make `id`'s bytes resident and pin them, reading from the model file on a miss.
    ///
    /// [`Insert::Cold`] is the policy: a forward pass visits weights in a fixed order and returns
    /// to the first only after touching every other, so the block just used is the one whose next
    /// use is furthest away. Recency is the pathological choice for that shape — it is what the OS
    /// page cache does, and what `docs/perf/results.md` measured collapsing.
    pub fn pin(&self, id: PagedId) -> Result<Pin<'_>> {
        self.pin_as(id, Insert::Cold)
    }

    /// [`Self::pin`], placing a miss (and promoting a hit) as `insert` says — see
    /// [`whole_insert`] for which.
    pub fn pin_as(&self, id: PagedId, insert: Insert) -> Result<Pin<'_>> {
        self.pools[id.pool].pin(id.block, insert)
    }

    /// Make `id` the next victim of its pool once its pins are gone, if it is resident: its
    /// reader is done with it ([`WeightReuse::Done`]).
    pub fn demote(&self, id: PagedId) {
        self.pools[id.pool].demote(id.block);
    }

    /// [`Self::demote`] every resident expert of `bank` but those in `keep`.
    pub fn demote_experts(&self, bank: PagedExperts, keep: &[usize]) {
        for e in 0..bank.n_expert {
            if !keep.contains(&(e as usize)) {
                self.pools[bank.pool].demote(bank.first + e);
            }
        }
    }

    /// Whether every pool holds a slot for each block registered with it, so nothing is ever
    /// evicted and each block is read at most once (`Backend::host_holds_every_block`).
    pub fn holds_every_block(&self) -> bool {
        let next = self.next_block.lock().unwrap();
        self.pools
            .iter()
            .zip(next.iter())
            .all(|(p, &n)| n as usize <= p.n_slots())
    }

    /// Make expert `e` of `bank` resident and pin it, reading from the model file on a miss, and
    /// place it in the recency order as `insert` says — see [`expert_insert`] for which.
    ///
    /// Safe to call for several experts at once from several threads, and meant to be: each miss
    /// reads outside its pool's lock, so concurrent calls keep several reads in flight.
    pub fn pin_expert(&self, bank: PagedExperts, e: usize, insert: Insert) -> Result<Pin<'_>> {
        if e >= bank.n_expert() {
            return Err(Error::backend(format!(
                "host pager: expert {e} of a {}-expert bank",
                bank.n_expert
            )));
        }
        self.pools[bank.pool].pin(bank.first + e as u32, insert)
    }

    /// Pin `id` only if it is already resident — the read path, which runs after the op's pre-step
    /// has pinned everything the op names.
    pub fn try_pin(&self, id: PagedId) -> Option<Pin<'_>> {
        self.pools[id.pool].try_pin(id.block)
    }

    /// Add `d` to the time spent waiting for residency (see [`Self::wait`]).
    pub fn note_wait(&self, d: std::time::Duration) {
        let ns = u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
        self.wait_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Total wall time callers reported waiting for weights to become resident, through
    /// [`Self::note_wait`].
    pub fn wait(&self) -> std::time::Duration {
        std::time::Duration::from_nanos(self.wait_ns.load(Ordering::Relaxed))
    }

    /// `paging.trace`: hand every pool's buffered trace events to the trace file — every pool,
    /// even after one fails — and return the first failure.
    pub fn flush_trace(&self) -> Result<()> {
        let results: Vec<Result<()>> = self.pools.iter().map(HostPager::flush_trace).collect();
        results.into_iter().collect()
    }

    /// Every pool's shape and activity, for the paging report.
    pub fn pool_stats(&self) -> Vec<PoolReport> {
        self.pools
            .iter()
            .zip(&self.keys)
            .map(|(p, k)| PoolReport {
                slot_bytes: p.slot_bytes(),
                n_slots: p.n_slots(),
                experts: k.experts,
                stats: p.stats(),
            })
            .collect()
    }

    /// Bytes committed to arenas — what the process actually holds.
    pub fn arena_bytes(&self) -> usize {
        self.pools.iter().map(|p| p.arena_bytes()).sum()
    }
}

/// `(offset, len)` file ranges as block extents.
fn extents(ranges: &[(u64, usize)]) -> Vec<BlockExtent> {
    ranges
        .iter()
        .map(|&(offset, len)| BlockExtent { offset, len })
        .collect()
}

/// The extents covering bytes `[at, at + len)` of the concatenation of `all` — one expert's slice
/// of a bank whose bytes may span several file ranges.
fn sub_extents(all: &[BlockExtent], at: usize, len: usize) -> Vec<BlockExtent> {
    let mut out = Vec::new();
    let (mut skip, mut want) = (at, len);
    for e in all {
        if want == 0 {
            break;
        }
        if skip >= e.len {
            skip -= e.len;
            continue;
        }
        let n = (e.len - skip).min(want);
        out.push(BlockExtent {
            offset: e.offset + skip as u64,
            len: n,
        });
        want -= n;
        skip = 0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Serves each byte as a function of its FILE offset, so a block read from the wrong place
    /// shows up as wrong bytes, and counts what it reads.
    #[derive(Default)]
    struct FakeIo {
        reads: AtomicUsize,
        bytes: AtomicUsize,
    }
    impl BlockIo for FakeIo {
        fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.bytes.fetch_add(desc.nbytes(), Ordering::SeqCst);
            let mut at = 0;
            for e in &desc.extents {
                for i in 0..e.len {
                    dst[at + i] = file_byte(e.offset + i as u64);
                }
                at += e.len;
            }
            Ok(())
        }
    }
    impl FakeIo {
        fn reads(&self) -> usize {
            self.reads.load(Ordering::SeqCst)
        }
    }

    /// The byte [`FakeIo`] serves at file offset `at`. A prime modulus, so slices that start at
    /// different multiples of a power of two do not look alike.
    fn file_byte(at: u64) -> u8 {
        (at % 251) as u8
    }

    const MIB: usize = 1 << 20;

    fn whole(spec: &[(usize, usize)]) -> Vec<Pageable> {
        spec.iter()
            .flat_map(|&(nbytes, count)| {
                std::iter::repeat_n(
                    Pageable {
                        nbytes,
                        reads: Reads::Whole,
                    },
                    count,
                )
            })
            .collect()
    }

    /// Banks of a split gate/up/down model: layer `i / PER_LAYER` holds bank `i`.
    fn banks(count: usize, n_expert: usize, expert_bytes: usize, n_used: usize) -> Vec<Pageable> {
        (0..count)
            .map(|i| Pageable {
                nbytes: n_expert * expert_bytes,
                reads: Reads::Experts {
                    n_expert,
                    n_used,
                    layer: i / PER_LAYER,
                },
            })
            .collect()
    }

    /// Expert banks per layer in [`banks`]: gate, up and down.
    const PER_LAYER: usize = 3;

    fn plan_of(plans: &[PoolPlan], slot_bytes: usize, experts: bool) -> Option<PoolPlan> {
        plans
            .iter()
            .copied()
            .find(|p| p.slot_bytes == slot_bytes && p.experts == experts)
    }

    fn committed(plans: &[PoolPlan]) -> usize {
        plans.iter().map(|p| p.n_slots * p.slot_bytes).sum()
    }

    #[test]
    fn tensors_below_the_floor_get_no_pool() {
        let plans = plan_pools(64 * MIB, &whole(&[(MIN_PAGED_BYTES - 1, 8)]));
        assert!(plans.is_empty(), "sub-floor tensors must not be paged");
    }

    /// A bank is cut into one block per expert, and its class is the expert's size.
    #[test]
    fn a_bank_is_planned_as_one_block_per_expert() {
        let plans = plan_pools(1 << 40, &banks(2, 64, 64 << 10, 4));
        assert_eq!(
            plans,
            vec![PoolPlan {
                slot_bytes: 64 << 10,
                n_slots: 128,
                n_blocks: 128,
                experts: true,
            }]
        );
    }

    /// The budget follows what a pass reads: every whole weight, read in full by every pass, is
    /// seated before the expert pool takes anything past its floor — even though the experts hold
    /// far more of the file. A split by file share would leave most of the whole weights out.
    #[test]
    fn whole_weights_are_seated_before_experts() {
        // 8 x 4 MiB whole = 32 MiB, against 4 banks x 64 experts x 1 MiB = 256 MiB of experts.
        let mut w = whole(&[(4 * MIB, 8)]);
        w.extend(banks(4, 64, MIB, 2));
        let floor = PER_LAYER * 64 * MIB;
        let budget = 32 * MIB + floor + 20 * MIB;
        let plans = plan_pools(budget, &w);
        let dense = plan_of(&plans, 4 * MIB, false).expect("whole class");
        let experts = plan_of(&plans, MIB, true).expect("expert class");
        assert_eq!(dense.n_slots, 8, "a whole weight was left to be re-read");
        assert_eq!(experts.n_slots, PER_LAYER * 64 + 20);
        assert_eq!(committed(&plans), budget);
    }

    /// An expert pool that cannot hold one op's pins is not built — pinning would fail mid-op —
    /// and the whole weights still get the budget. Floors are seated before any class is topped
    /// up, so the budget here is one byte short of the whole class's one-block floor plus the
    /// expert floor.
    #[test]
    fn an_expert_pool_below_one_ops_pins_stays_mapped() {
        let mut w = whole(&[(4 * MIB, 8)]);
        w.extend(banks(4, 64, MIB, 2));
        let plans = plan_pools(4 * MIB + PER_LAYER * 64 * MIB - 1, &w);
        assert_eq!(plan_of(&plans, MIB, true), None, "{plans:?}");
        assert_eq!(plan_of(&plans, 4 * MIB, false).map(|p| p.n_slots), Some(8));
    }

    /// A class's floor counts only its own banks per op. A fused gate_up bank and its down bank are
    /// different slice sizes, so one op pins `n_expert` blocks from each class, not three banks'
    /// worth: a budget of exactly one bank per class seats both. Pricing every class at three banks
    /// would drop both to the mapping here.
    #[test]
    fn a_class_floor_counts_its_own_banks_per_op() {
        let (n_expert, layers) = (16usize, 4usize);
        let (gu, down) = (2 * MIB, MIB);
        let w: Vec<Pageable> = (0..layers)
            .flat_map(|layer| {
                [gu, down].map(|eb| Pageable {
                    nbytes: n_expert * eb,
                    reads: Reads::Experts {
                        n_expert,
                        n_used: 2,
                        layer,
                    },
                })
            })
            .collect();
        let plans = plan_pools(n_expert * (gu + down), &w);
        assert_eq!(plan_of(&plans, gu, true).map(|p| p.n_slots), Some(n_expert));
        assert_eq!(
            plan_of(&plans, down, true).map(|p| p.n_slots),
            Some(n_expert)
        );
        // Two banks of one class in a layer (split gate and up) do need two banks' worth.
        let split = banks(PER_LAYER * layers, n_expert, MIB, 2);
        assert_eq!(plan_pools(2 * n_expert * MIB, &split), vec![]);
        assert_eq!(
            plan_pools(PER_LAYER * n_expert * MIB, &split)[0].n_slots,
            PER_LAYER * n_expert
        );
    }

    /// A bank that does not split evenly is planned whole, as `register_experts` would decline it.
    #[test]
    fn an_uneven_bank_is_planned_whole() {
        let w = [Pageable {
            nbytes: 3 * MIB + 1,
            reads: Reads::Experts {
                n_expert: 2,
                n_used: 1,
                layer: 0,
            },
        }];
        let plans = plan_pools(8 * MIB, &w);
        assert_eq!(
            plan_of(&plans, 3 * MIB + 1, false).map(|p| p.n_slots),
            Some(1)
        );
    }

    /// A class the remaining budget cannot seat once is dropped, not squeezed in — and the classes
    /// that matter most are seated first.
    #[test]
    fn a_class_that_cannot_be_seated_is_dropped() {
        // 8 MiB budget: 4 MiB x 8 blocks (32 MiB total) dominates; 6 MiB x 1 is the minor class.
        let plans = plan_pools(8 * MIB, &whole(&[(4 * MIB, 8), (6 * MIB, 1)]));
        assert!(plans.iter().any(|p| p.slot_bytes == 4 * MIB));
        assert!(committed(&plans) <= 8 * MIB, "plan overspent");
    }

    /// The plan does not depend on the order the tensors are listed in, or a `HashMap`'s iteration
    /// order: residency would otherwise differ run to run for the same model and budget.
    #[test]
    fn the_plan_is_stable_across_runs_and_orders() {
        let mut w = whole(&[(4 * MIB, 3), (8 * MIB, 3), (2 * MIB, 3)]);
        w.extend(banks(3, 16, MIB, 2));
        let first = plan_pools(100 * MIB, &w);
        for _ in 0..8 {
            assert_eq!(plan_pools(100 * MIB, &w), first);
        }
        w.reverse();
        assert_eq!(plan_pools(100 * MIB, &w), first);
    }

    #[test]
    fn registration_maps_equal_sizes_to_one_pool() {
        let io = Arc::new(FakeIo::default());
        let plans = plan_pools(64 * MIB, &whole(&[(4 * MIB, 4), (8 * MIB, 2)]));
        let w = PagedWeights::new(&plans, io, None).expect("pools");
        let a = w.register(&[(0, 4 * MIB)]).unwrap().expect("paged");
        let b = w.register(&[(99, 4 * MIB)]).unwrap().expect("paged");
        let c = w.register(&[(0, 8 * MIB)]).unwrap().expect("paged");
        assert_eq!(a.pool, b.pool);
        assert_ne!(a.block, b.block, "each weight needs its own block id");
        assert_ne!(a.pool, c.pool);
        // A size with no pool is declined rather than mis-filed into a neighbouring class.
        assert_eq!(w.register(&[(0, 3 * MIB)]).unwrap(), None);
        // And a whole weight never lands in an expert pool, or a bank in a whole one.
        assert_eq!(w.register_experts(&[(0, 8 * MIB)], 2).unwrap(), None);
    }

    #[test]
    fn pin_reads_once_then_hits() {
        let io = Arc::new(FakeIo::default());
        let plans = plan_pools(64 * MIB, &whole(&[(MIB, 4)]));
        let w = PagedWeights::new(&plans, io.clone(), None).expect("pools");
        let id = w.register(&[(0, MIB)]).unwrap().expect("paged");
        assert_eq!(w.pin(id).unwrap()[..8], [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(io.reads(), 1);
        assert!(w.try_pin(id).is_some());
        assert_eq!(io.reads(), 1, "a hit must not re-read");
    }

    /// Pinning one expert reads that expert's bytes and nothing else: one read of one slice, from
    /// the right place in the file, even when the slice straddles two of the bank's file ranges.
    #[test]
    fn pinning_an_expert_reads_only_that_expert() {
        let (n_expert, eb) = (8usize, 256usize << 10);
        let io = Arc::new(FakeIo::default());
        let plans = plan_pools(1 << 30, &banks(1, n_expert, eb, 2));
        let w = PagedWeights::new(&plans, io.clone(), None).expect("pools");
        // Two file ranges, split mid-way through expert 2.
        let (base, cut) = (7_000u64, 2 * eb + eb / 3);
        let bank = w
            .register_experts(
                &[
                    (base, cut),
                    (base + cut as u64 + 1_000, n_expert * eb - cut),
                ],
                n_expert,
            )
            .unwrap()
            .expect("an expert pool");
        let want = |e: usize| -> Vec<u8> {
            (e * eb..(e + 1) * eb)
                .map(|i| {
                    let gap = if i >= cut { 1_000 } else { 0 };
                    file_byte(base + (i + gap) as u64)
                })
                .collect()
        };
        for e in [5usize, 2] {
            let before = io.bytes.load(Ordering::SeqCst);
            let pin = w.pin_expert(bank, e, Insert::Mru).unwrap();
            assert_eq!(pin.len(), eb);
            assert!(pin[..] == want(e)[..], "expert {e} is not its own slice");
            assert_eq!(
                io.bytes.load(Ordering::SeqCst) - before,
                eb,
                "read past expert {e}"
            );
        }
        assert_eq!(io.reads(), 2, "one read per pinned expert");
        drop(w.pin_expert(bank, 5, Insert::Mru).unwrap());
        assert_eq!(io.reads(), 2, "a resident expert must not be re-read");
        assert!(w.pin_expert(bank, n_expert, Insert::Mru).is_err());
    }

    /// Experts evict by recency: the expert used most recently survives, the least recent goes.
    /// A cold-end insert would keep the first ones loaded and churn the newest instead.
    #[test]
    fn experts_evict_the_least_recently_used() {
        let (n_expert, eb) = (16usize, MIB);
        let io = Arc::new(FakeIo::default());
        // Exactly the floor, so the pool is PER_LAYER * n_expert slots of a 4-bank model.
        let plans = plan_pools(PER_LAYER * n_expert * eb, &banks(4, n_expert, eb, 2));
        assert_eq!(plans[0].n_slots, PER_LAYER * n_expert);
        let w = PagedWeights::new(&plans, io.clone(), None).expect("pools");
        let b: Vec<PagedExperts> = (0..4)
            .map(|i| {
                w.register_experts(&[((i * n_expert * eb) as u64, n_expert * eb)], n_expert)
                    .unwrap()
                    .expect("bank")
            })
            .collect();
        // Fill every slot: banks 0-2 entirely.
        for bank in &b[..3] {
            for e in 0..n_expert {
                drop(w.pin_expert(*bank, e, Insert::Mru).unwrap());
            }
        }
        // Touch bank 0 expert 0 again, so it is the most recent; bank 0 expert 1 is now the least.
        drop(w.pin_expert(b[0], 0, Insert::Mru).unwrap());
        let reads = io.reads();
        drop(w.pin_expert(b[3], 0, Insert::Mru).unwrap()); // a miss: evicts one block
        assert_eq!(io.reads(), reads + 1);
        drop(w.pin_expert(b[0], 0, Insert::Mru).unwrap());
        assert_eq!(
            io.reads(),
            reads + 1,
            "the most recently used expert was evicted"
        );
        drop(w.pin_expert(b[0], 1, Insert::Mru).unwrap());
        assert_eq!(
            io.reads(),
            reads + 2,
            "the least recently used expert survived"
        );
    }

    /// Expert traffic cannot evict a whole weight: they live in separate pools, so a whole weight
    /// pinned once stays resident however many experts stream past it.
    #[test]
    fn experts_never_evict_whole_weights() {
        let (n_expert, eb) = (16usize, MIB);
        let mut spec = whole(&[(eb, 2)]);
        spec.extend(banks(8, n_expert, eb, 2));
        let io = Arc::new(FakeIo::default());
        let plans = plan_pools(2 * eb + PER_LAYER * n_expert * eb, &spec);
        let w = PagedWeights::new(&plans, io.clone(), None).expect("pools");
        let dense: Vec<PagedId> = (0..2)
            .map(|i| w.register(&[(i * eb as u64, eb)]).unwrap().expect("whole"))
            .collect();
        let b: Vec<PagedExperts> = (0..8)
            .map(|i| {
                w.register_experts(
                    &[(((2 + i * n_expert) * eb) as u64, n_expert * eb)],
                    n_expert,
                )
                .unwrap()
                .expect("bank")
            })
            .collect();
        for &d in &dense {
            drop(w.pin(d).unwrap());
        }
        for bank in &b {
            for e in 0..n_expert {
                drop(w.pin_expert(*bank, e, Insert::Mru).unwrap());
            }
        }
        for &d in &dense {
            assert!(
                w.try_pin(d).is_some(),
                "expert traffic evicted a whole weight"
            );
        }
    }

    #[test]
    fn the_done_share_fits_a_12g_arena() {
        // V4-Flash at `INFR_DRAM_CACHE=12g`: 1904 expert slots, a layer's floor of 3 x 256, 43
        // layers of three banks (`docs/deepseek.md` § "Layer-major prefill on the CPU").
        let (slots, floor, layers, banks) = (1904usize, 3 * 256usize, 43usize, 3usize);
        assert!(DONE_KEEP_EXPERTS * banks * layers <= slots - floor);
    }

    /// The last row's experts first, then the most-routed, up to the share, without repeats; ties
    /// by expert id.
    #[test]
    fn done_keep_takes_the_last_row_then_the_most_routed() {
        let by_count = [
            (0, 1),
            (3, 9),
            (5, 2),
            (7, 9),
            (9, 4),
            (11, 1),
            (12, 3),
            (13, 5),
        ];
        assert_eq!(
            done_keep(&[5, 0], &by_count),
            vec![5, 0, 3, 7, 13, 9, 12, 11][..DONE_KEEP_EXPERTS].to_vec()
        );
        assert_eq!(done_keep(&[], &[(4, 1)]), vec![4]);
    }

    #[test]
    fn sub_extents_cut_across_ranges() {
        let all = extents(&[(100, 10), (1_000, 10), (5_000, 10)]);
        assert_eq!(
            sub_extents(&all, 8, 14),
            extents(&[(108, 2), (1_000, 10), (5_000, 2)])
        );
        assert_eq!(sub_extents(&all, 10, 10), extents(&[(1_000, 10)]));
        assert_eq!(sub_extents(&all, 25, 5), extents(&[(5_005, 5)]));
    }
}
