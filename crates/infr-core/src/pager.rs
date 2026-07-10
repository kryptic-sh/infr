//! Generic, backend-agnostic block-paging bookkeeping: a fixed-slot LRU cache mapping an opaque
//! `BlockId` to a slot index, for any backend that wants to keep a working set of uniform-ish
//! byte blocks resident in a budget smaller than the full set.
//!
//! This module holds ONLY the host-side residency/eviction *decision* (no GPU types, no bytes) —
//! [`Pager::touch`] is pure bookkeeping, cheap to unit test without a device. A backend wraps this
//! with the actual VRAM arena + device LUT buffer + upload machinery (see `infr-vulkan`'s
//! `GpuPager`) and drives it by calling `touch` for every block a dispatch is about to read,
//! before recording that dispatch.
//!
//! ## Today: MoE expert paging
//! The Vulkan MoE lowering plugs in `BlockId = (layer, role, expert_id)` packed into a `u32` (see
//! `infr-vulkan`'s `pager` module) with a demand/LRU policy — an expert is paged in on first use
//! by a layer and evicted least-recently-used when the arena is full.
//!
//! ## Planned: dense layer streaming
//! A future policy can reuse the SAME arena/LUT/upload core for `BlockId = layer_idx` in a dense
//! model whose weights don't fit VRAM: since a dense decode visits layers in a fixed, known order
//! every step, that policy would be schedule-driven (exact prefetch of layer `l+1` while layer `l`
//! runs) rather than demand/LRU — `Pager` doesn't bake in LRU-only semantics anywhere a caller
//! couldn't substitute its own `touch`-equivalent, but the prefetch scheduling itself is NOT
//! implemented here (see the task doc); only documented as the intended extension point.
use std::collections::{HashMap, VecDeque};

/// Opaque identifier for one pageable block. The pager never interprets this — callers pack
/// whatever key space they need (an expert id, a `(layer, role, expert)` tuple, a layer index for
/// the planned dense-streaming policy, ...) into it.
pub type BlockId = u32;

/// Sentinel LUT value meaning "not resident" — mirrors what a device-side LUT buffer should hold
/// for any block the pager hasn't (yet) admitted, so an accidental stale read is loud (an
/// out-of-range slot index) rather than silently aliasing slot 0.
pub const NOT_RESIDENT: u32 = u32::MAX;

/// Outcome of [`Pager::touch`]: whether the block was already resident (no upload needed) or had
/// to be paged in (possibly evicting another block first).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    /// Already resident at `slot` — no upload needed, LUT already correct.
    Hit { slot: u32 },
    /// Now resident at `slot` after an upload the caller must perform. `evicted` is the block
    /// that previously occupied `slot`, if any (so the caller can invalidate its LUT entry, or
    /// just leave it — it's never read again unless `evicted` is re-touched, which re-admits it
    /// through the normal miss path).
    Miss { slot: u32, evicted: Option<BlockId> },
}

/// Cumulative pager activity — the `INFR_PAGER_STATS` hit-rate counter the task's validation step
/// asks for rides this.
#[derive(Debug, Clone, Copy, Default)]
pub struct PagerStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl PagerStats {
    /// Hit rate over all `touch` calls so far; `1.0` (vacuously) before any calls.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            1.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Fixed-`n_slots` LRU residency map. `n_slots` slots are handed out first-come (no eviction)
/// until exhausted, then every further miss evicts the least-recently-touched resident block.
///
/// # Within-batch safety
/// A caller MUST resolve every block a single dispatch batch needs (e.g. all `n_used` experts a
/// decode step's top-k picked, or every distinct expert a prefill ubatch's bucket counts named)
/// via `touch` BEFORE recording/dispatching anything that reads the result — and must do so as one
/// uninterrupted sequence of `touch` calls. `touch` marks a block most-recently-used the instant
/// it resolves, and eviction only ever pops the LEAST-recently-used entry, so a block touched
/// earlier in the same batch can never be evicted by a later touch in that SAME batch. This only
/// holds if `n_slots >= ` the number of DISTINCT blocks one batch touches — [`Pager::new`]'s doc
/// repeats this invariant; violating it (a cache budget too small to even hold one batch) is a
/// configuration error the caller should surface, not silently thrash.
pub struct Pager {
    n_slots: usize,
    /// block_id -> slot index for every currently-resident block.
    resident: HashMap<BlockId, u32>,
    /// LRU order, oldest (least-recently-used) at the front. A block appears at most once;
    /// `touch` on an existing entry removes-then-repushes it (O(n_slots), fine at the slot counts
    /// this cache runs at — tens to low hundreds; an intrusive doubly-linked list is the upgrade
    /// path if that ever isn't true, per the module doc's SLRU note).
    lru: VecDeque<BlockId>,
    /// Slot indices never yet handed out (drained before any eviction kicks in).
    free: Vec<u32>,
    stats: PagerStats,
}

impl Pager {
    /// A pager with `n_slots` uniform slots and nothing resident. `n_slots` must be at least the
    /// largest number of distinct blocks any single dispatch batch will `touch` (see the
    /// within-batch safety note) — the pager can't check that itself (it doesn't know about
    /// batches), so a caller sizing `n_slots` from a VRAM budget must clamp it to at least that
    /// floor and error out earlier if the budget can't cover even one batch.
    pub fn new(n_slots: usize) -> Self {
        Self {
            n_slots,
            resident: HashMap::with_capacity(n_slots),
            lru: VecDeque::with_capacity(n_slots),
            free: (0..n_slots as u32).rev().collect(), // pop() hands out slot 0 first
            stats: PagerStats::default(),
        }
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn stats(&self) -> PagerStats {
        self.stats
    }

    /// Number of currently-resident blocks (== `n_slots` once the cache is full).
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    pub fn slot_of(&self, id: BlockId) -> Option<u32> {
        self.resident.get(&id).copied()
    }

    /// Move `id` to the most-recently-used end without changing residency (internal to `touch`,
    /// exposed for the doubly-recorded case: a batch that touches the same id twice).
    fn mark_mru(&mut self, id: BlockId) {
        if let Some(pos) = self.lru.iter().position(|&x| x == id) {
            self.lru.remove(pos);
        }
        self.lru.push_back(id);
    }

    /// Ensure `id` is resident, evicting the LRU block if the cache is full. Returns whether it
    /// was already resident (caller skips the upload) or had to be paged in (caller must upload
    /// the block's bytes into `slot` and write the device LUT entry `id -> slot` before any
    /// dispatch reads it).
    pub fn touch(&mut self, id: BlockId) -> Resolution {
        if let Some(&slot) = self.resident.get(&id) {
            self.stats.hits += 1;
            self.mark_mru(id);
            return Resolution::Hit { slot };
        }
        self.stats.misses += 1;
        let (slot, evicted) = if let Some(s) = self.free.pop() {
            (s, None)
        } else {
            let victim = self
                .lru
                .pop_front()
                .expect("resident.len() == n_slots > 0 with no free slots implies an LRU entry");
            let vslot = self
                .resident
                .remove(&victim)
                .expect("every lru entry has a resident mapping");
            self.stats.evictions += 1;
            (vslot, Some(victim))
        };
        self.resident.insert(id, slot);
        self.lru.push_back(id);
        Resolution::Miss { slot, evicted }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_pager_is_all_misses_until_full() {
        let mut p = Pager::new(3);
        assert_eq!(
            p.touch(10),
            Resolution::Miss {
                slot: 0,
                evicted: None
            }
        );
        assert_eq!(
            p.touch(11),
            Resolution::Miss {
                slot: 1,
                evicted: None
            }
        );
        assert_eq!(
            p.touch(12),
            Resolution::Miss {
                slot: 2,
                evicted: None
            }
        );
        assert_eq!(p.resident_count(), 3);
        assert_eq!(p.stats().misses, 3);
        assert_eq!(p.stats().hits, 0);
    }

    #[test]
    fn repeat_touch_is_a_hit_at_the_same_slot() {
        let mut p = Pager::new(2);
        let Resolution::Miss { slot, .. } = p.touch(5) else {
            panic!("expected miss")
        };
        assert_eq!(p.touch(5), Resolution::Hit { slot });
        assert_eq!(p.touch(5), Resolution::Hit { slot });
        assert_eq!(p.stats().hits, 2);
        assert_eq!(p.stats().misses, 1);
    }

    #[test]
    fn eviction_picks_least_recently_used() {
        let mut p = Pager::new(2);
        p.touch(1); // slot 0
        p.touch(2); // slot 1, full now
        p.touch(1); // hit, 1 is now MRU; 2 is LRU
                    // 3 must evict 2 (LRU), not 1 (just touched).
        let Resolution::Miss { slot, evicted } = p.touch(3) else {
            panic!("expected miss")
        };
        assert_eq!(evicted, Some(2));
        assert_eq!(p.slot_of(1), Some(0)); // 1 kept its original slot
        assert_eq!(p.slot_of(3), Some(slot));
        assert_eq!(p.slot_of(2), None);
        assert_eq!(p.stats().evictions, 1);
    }

    #[test]
    fn slot_reuse_after_eviction_is_exact() {
        // The freed slot from an eviction is the ONLY slot available for the next miss (n_slots
        // fixed) — assert the pager actually reuses it rather than growing.
        let mut p = Pager::new(1);
        let Resolution::Miss { slot: s0, .. } = p.touch(1) else {
            panic!()
        };
        let Resolution::Miss {
            slot: s1,
            evicted: e1,
        } = p.touch(2)
        else {
            panic!()
        };
        assert_eq!(s0, s1); // only one slot exists, must be reused
        assert_eq!(e1, Some(1));
        assert_eq!(p.resident_count(), 1);
    }

    #[test]
    fn within_batch_touch_order_protects_earlier_ids_from_later_ones() {
        // Simulates a decode step's top-k = 3 experts touched in a fixed order against a 3-slot
        // cache with 1 already resident from a PRIOR step — the prior step's expert should be the
        // only eviction victim; none of the 3 in-flight ids may evict each other.
        let mut p = Pager::new(3);
        p.touch(100); // prior step's expert, now LRU-oldest
        for id in [7u32, 8, 9] {
            p.touch(id);
        }
        // Cache is now full (100 evicted by whichever of 7/8/9 needed a slot first); crucially,
        // ALL of 7, 8, 9 must be resident simultaneously — that's the within-batch invariant.
        assert!(p.slot_of(7).is_some());
        assert!(p.slot_of(8).is_some());
        assert!(p.slot_of(9).is_some());
        assert_eq!(p.slot_of(100), None);
    }

    #[test]
    fn lut_coherence_across_a_simulated_token_sequence() {
        // Drives a small cache through a token sequence with a repeating expert-access pattern
        // (like a real MoE decode loop) and checks the (block_id -> slot) mapping stays exactly
        // what a host-mirrored LUT array would need at every step.
        let mut p = Pager::new(4);
        let mut lut = vec![NOT_RESIDENT; 16]; // host mirror of the device LUT buffer
        let apply = |p: &mut Pager, lut: &mut [u32], id: BlockId| match p.touch(id) {
            Resolution::Hit { slot } => assert_eq!(lut[id as usize], slot, "hit must match LUT"),
            Resolution::Miss { slot, evicted } => {
                if let Some(e) = evicted {
                    lut[e as usize] = NOT_RESIDENT;
                }
                lut[id as usize] = slot;
            }
        };
        // A token sequence revisiting a hot set {0,1} plus a cold long tail.
        let seq = [0u32, 1, 2, 0, 1, 3, 4, 0, 1, 5, 0, 1, 6, 7, 0, 1];
        for &id in &seq {
            apply(&mut p, &mut lut, id);
        }
        // The hot set must still be resident (never evicted across the whole run) and every
        // resident block's LUT entry must exactly match the pager's own view.
        for id in [0u32, 1] {
            let slot = p.slot_of(id).expect("hot id stays resident");
            assert_eq!(lut[id as usize], slot);
        }
        for id in 0..16u32 {
            match p.slot_of(id) {
                Some(slot) => assert_eq!(lut[id as usize], slot),
                None => assert_eq!(lut[id as usize], NOT_RESIDENT),
            }
        }
        assert!(p.stats().hits > 0);
        assert!(p.stats().evictions > 0);
    }

    #[test]
    fn hit_rate_reports_sane_values() {
        let mut p = Pager::new(2);
        assert_eq!(p.stats().hit_rate(), 1.0); // vacuous
        p.touch(1);
        p.touch(1);
        assert!((p.stats().hit_rate() - 0.5).abs() < 1e-9);
    }
}
