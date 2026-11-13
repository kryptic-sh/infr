//! The pager as it was before its state moved into slot-indexed arrays and a linked recency list:
//! four `BlockId`-keyed maps, signed recency stamps, a victim scan over every resident block, and
//! group touches as a stamp overlay. Kept, test-only, as the model the [`super::Pager`] is checked
//! against call by call (`tests` below): the refactor changed the cost of each call, and must not
//! change any outcome. Its code is the old one with the trace hooks taken out.

use std::collections::HashMap;

use super::{BlockId, Insert, PagerStats, Resolution};

/// The old `Pager`.
pub struct Pager {
    n_slots: usize,
    /// block_id -> slot index for every currently-resident block.
    resident: HashMap<BlockId, u32>,
    /// block_id -> recency stamp for every resident block: the LRU order is ascending stamp, and
    /// the eviction victim is the lowest stamp among the unprotected blocks. A promotion takes a
    /// stamp above every other ([`Self::hot`]) and a cold-end insert one below every other
    /// ([`Self::cold`]), so a hit or an insert is one map write; the O(n_slots) scan is paid only
    /// when choosing a victim, which already costs a fill.
    recency: HashMap<BlockId, i64>,
    /// The last stamp handed out at the most-recently-used end.
    hot: i64,
    /// The last stamp handed out at the cold end.
    cold: i64,
    /// Slot indices never yet handed out (drained before any eviction kicks in).
    free: Vec<u32>,
    /// block_id -> the [`Self::begin_batch`] epoch of its last touch. Eviction skips blocks
    /// touched in the CURRENT batch — the explicit form of the within-batch safety invariant
    /// (MRU ordering used to carry it implicitly; the scan-resistant [`Self::touch_cold`] path
    /// inserts at the COLD end, so it needs the epoch guard instead).
    epoch: HashMap<BlockId, u64>,
    cur_epoch: u64,
    /// block_id -> outstanding pin count, for blocks a caller is READING right now
    /// ([`Self::resolve_and_pin`]). Only present while non-zero. A pinned block is never an
    /// eviction victim: the epoch guard above cannot express this, because it is scoped to one
    /// dispatch batch while a pin lasts as long as the borrow — a CPU kernel reads a weight for a
    /// whole op, spanning batches.
    pinned: HashMap<BlockId, u32>,
    /// Equal-size block groups ([`Self::group_by`]), or `None` for a pager that never groups.
    /// Only the MoE expert pools group; every other consumer pays one test of this field.
    groups: Option<Groups>,
    stats: PagerStats,
}

impl Pager {
    /// A pager with `n_slots` uniform slots and nothing resident. `n_slots` must be at least the
    /// largest number of distinct blocks any single dispatch batch will `touch` (see the
    /// within-batch safety note) — the pager can't check that itself (it doesn't know about
    /// batches), so a caller sizing `n_slots` from a VRAM budget must clamp it to at least that
    /// floor and error out earlier if the budget can't cover even one batch.
    ///
    /// # Panics
    /// If `n_slots == 0`. A zero-slot pager cannot satisfy even a one-block batch, so there is no
    /// correct behaviour to fall back to: the first `touch` would find `free` empty, find no
    /// resident block to evict, and die inside `Self::take_slot` on an `expect` about that —
    /// a panic that describes the wrong thing entirely, several frames away from the
    /// mistake. Clamping to 1 instead would be worse than either: it manufactures a pager that
    /// silently misses on EVERY touch and evicts the block the caller resolved a moment ago,
    /// violating the within-batch safety invariant above for any batch wider than one block —
    /// precisely the coherent-but-wrong failure this module refuses elsewhere (see `take_slot`'s
    /// batch-overflow assert). The contract quoted above already makes the FLOOR the caller's job,
    /// so a 0 here is a caller bug that must surface at the mistake. The one in-tree caller
    /// (`infr-vulkan`'s `validate_pager_dims`) already rejects a 0-slot budget with a clean `Err`
    /// before constructing anything, so this can only fire for a NEW caller that skipped that step
    /// — never on a shipping path. `new` also runs once per pool at model load, so a check here
    /// costs nothing measurable.
    pub fn new(n_slots: usize) -> Self {
        assert!(
            n_slots >= 1,
            "pager: n_slots must be at least 1 (a zero-slot pager can never satisfy a touch) — \
             size the slot count from the paging budget and reject a budget that can't hold one \
             dispatch batch BEFORE constructing the pager"
        );
        Self {
            n_slots,
            resident: HashMap::with_capacity(n_slots),
            recency: HashMap::with_capacity(n_slots),
            hot: 0,
            cold: 0,
            free: (0..n_slots as u32).rev().collect(), // pop() hands out slot 0 first
            epoch: HashMap::with_capacity(n_slots),
            cur_epoch: 0,
            pinned: HashMap::new(),
            groups: None,
            stats: PagerStats::default(),
        }
    }

    /// Open a new touch batch (one dispatch's worth of `touch`/`touch_cold` calls). Blocks
    /// touched under the current epoch are never eviction victims, which makes the within-batch
    /// safety invariant explicit for BOTH insertion orders (classic MRU and `touch_cold`'s
    /// cold-end scan inserts, where LRU position alone no longer protects batch siblings).
    pub fn begin_batch(&mut self) {
        self.cur_epoch += 1;
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
        self.hot += 1;
        self.recency.insert(id, self.hot);
    }

    /// Place `id` at the cold end — the next victim unless something colder arrives.
    fn mark_cold(&mut self, id: BlockId) {
        self.cold -= 1;
        self.recency.insert(id, self.cold);
    }

    /// The least-recently-used resident block that `eligible` accepts, skipping, when
    /// `batch_guard`, a block the current batch touched through its group ([`Self::touch_group`];
    /// `eligible` sees only the block's own epoch). A block's recency is its own stamp, or the one
    /// its group's last touch gave it if that is later. Each block's group is looked up once, and
    /// only while some group touch is in force: an ungrouped pager ranks exactly as before groups.
    fn coldest(&self, batch_guard: bool, eligible: impl Fn(BlockId) -> bool) -> Option<BlockId> {
        let ranked = self.recency.iter().filter(|&(&b, _)| eligible(b));
        let Some(g) = self.groups.as_ref().filter(|g| g.stamped()) else {
            return ranked.min_by_key(|&(_, &stamp)| stamp).map(|(&b, _)| b);
        };
        ranked
            .filter_map(|(&b, &own)| match g.stamp_of(b) {
                Some((stamp, epoch)) => (!(batch_guard && epoch == self.cur_epoch))
                    .then(|| (b, own.max(stamp + g.offset(b)))),
                None => Some((b, own)),
            })
            .min_by_key(|&(_, stamp)| stamp)
            .map(|(b, _)| b)
    }

    /// `id` becomes resident at `slot` (every miss's admission).
    fn admit(&mut self, id: BlockId, slot: u32) {
        self.group_admit(id);
        self.resident.insert(id, slot);
    }

    /// Ensure `id` is resident, evicting the least-recently-used unprotected block if the cache
    /// is full. Returns whether it was already resident (caller skips the upload) or had to be
    /// paged in (caller must upload the block's bytes into `slot` and write the device LUT entry
    /// `id -> slot` before any dispatch reads it).
    pub fn touch(&mut self, id: BlockId) -> Resolution {
        self.touch_untraced(id)
    }

    fn touch_untraced(&mut self, id: BlockId) -> Resolution {
        self.epoch.insert(id, self.cur_epoch);
        if let Some(&slot) = self.resident.get(&id) {
            self.stats.hits += 1;
            self.mark_mru(id);
            return Resolution::Hit { slot };
        }
        self.stats.misses += 1;
        let (slot, evicted) = self.take_slot();
        self.admit(id, slot);
        self.mark_mru(id);
        Resolution::Miss { slot, evicted }
    }

    /// Ensure `id` is resident AND pin it, so it cannot be evicted until [`Self::unpin`].
    ///
    /// This is the entry point for a tier whose slots are read DIRECTLY (the host DRAM arena: a
    /// CPU kernel dereferences the slot for a whole op, and a staging copy reads it until the copy
    /// is recorded), as opposed to the VRAM pagers, whose callers upload a slot's bytes and then
    /// let residency ride the batch epoch alone.
    ///
    /// `None` means every resident block is protected right now — all pinned, or all touched by
    /// the current batch — so there is no slot to give. Nothing is evicted, nothing is inserted and
    /// no counter moves in that case: the caller has to release pins (or was sized below its own
    /// working set, which is a configuration error it should report). This is a `None` and not a
    /// panic precisely because it IS reachable at runtime, unlike the unpinned callers' floor.
    ///
    /// On a `Miss` the caller must fill the slot before reading it; on a `Hit` it is already valid.
    /// Either way the block comes back pinned exactly once.
    pub fn resolve_and_pin(&mut self, id: BlockId, insert: Insert) -> Option<Resolution> {
        self.resolve_and_pin_untraced(id, insert)
    }

    fn resolve_and_pin_untraced(&mut self, id: BlockId, insert: Insert) -> Option<Resolution> {
        if let Some(&slot) = self.resident.get(&id) {
            self.epoch.insert(id, self.cur_epoch);
            self.stats.hits += 1;
            if insert == Insert::Mru {
                self.mark_mru(id);
            }
            *self.pinned.entry(id).or_insert(0) += 1;
            return Some(Resolution::Hit { slot });
        }
        let (slot, evicted) = self.take_slot_opt()?;
        self.epoch.insert(id, self.cur_epoch);
        self.stats.misses += 1;
        self.admit(id, slot);
        match insert {
            Insert::Mru => self.mark_mru(id),
            Insert::Cold => self.mark_cold(id),
        }
        *self.pinned.entry(id).or_insert(0) += 1;
        Some(Resolution::Miss { slot, evicted })
    }

    /// Take ANOTHER pin on a block that is already resident, without counting anything.
    ///
    /// This is not a residency decision — it is a second borrow of a block the caller has already
    /// resolved (the CPU read path re-borrows what its op's pin pre-step resolved a moment ago).
    /// Counting it would inflate the hit rate by exactly one hit per access, which makes a
    /// thrashing cache report ~50% and a perfect one report 100%: the two become indistinguishable
    /// where it matters. Returns `None` if `id` is not resident.
    pub fn repin(&mut self, id: BlockId) -> Option<u32> {
        self.repin_untraced(id)
    }

    fn repin_untraced(&mut self, id: BlockId) -> Option<u32> {
        let slot = *self.resident.get(&id)?;
        *self.pinned.entry(id).or_insert(0) += 1;
        Some(slot)
    }

    /// Pin `id` only if it is ALREADY resident — never reads, never evicts, never inserts.
    ///
    /// The hit-only probe a tier above this one uses to decide between "copy from here" and "go to
    /// the tier below". A hit counts as a hit; a miss counts as nothing, because no residency
    /// decision was made.
    pub fn pin_if_resident(&mut self, id: BlockId) -> Option<u32> {
        self.pin_if_resident_untraced(id)
    }

    fn pin_if_resident_untraced(&mut self, id: BlockId) -> Option<u32> {
        let slot = *self.resident.get(&id)?;
        self.epoch.insert(id, self.cur_epoch);
        self.stats.hits += 1;
        *self.pinned.entry(id).or_insert(0) += 1;
        Some(slot)
    }

    /// Release one pin taken by [`Self::resolve_and_pin`] / [`Self::pin_if_resident`].
    ///
    /// # Panics
    /// If `id` holds no pin. An unbalanced release is not a state to recover from: it would let a
    /// slot be evicted while a reader still holds bytes from it, which is silent wrong output
    /// rather than a failure. The RAII guard in the host tier is what makes this balanced.
    pub fn unpin(&mut self, id: BlockId) {
        match self.pinned.get_mut(&id) {
            Some(n) if *n > 1 => *n -= 1,
            Some(_) => {
                self.pinned.remove(&id);
            }
            None => panic!("pager: unpin of block {id}, which holds no pin"),
        }
    }

    /// How many distinct blocks are pinned right now — the number a caller sizes `n_slots`
    /// against (see [`Self::resolve_and_pin`]'s `None`).
    pub fn pinned_blocks(&self) -> usize {
        self.pinned.len()
    }

    /// Forget `id` and return its slot to the free list, if it was resident.
    ///
    /// For the caller whose fill FAILED: the slot holds partial bytes, so the block must not stay
    /// resident (the next request would be served that garbage as a hit). This is not a capacity
    /// eviction and is not counted as one — nothing was displaced to make room.
    ///
    /// # Panics
    /// If `id` is pinned. Removing a block someone is reading is the exact failure pins exist to
    /// prevent, so it is a caller bug, not a recoverable state.
    pub fn evict(&mut self, id: BlockId) -> Option<u32> {
        self.evict_untraced(id)
    }

    fn evict_untraced(&mut self, id: BlockId) -> Option<u32> {
        assert!(
            !self.pinned.contains_key(&id),
            "pager: evict of pinned block {id}"
        );
        let slot = self.resident.remove(&id)?;
        self.recency.remove(&id);
        self.epoch.remove(&id);
        self.group_removed(id);
        self.free.push(slot);
        Some(slot)
    }

    /// Schedule-driven residency for a DETERMINISTIC cyclic sweep — the dense layer-streaming
    /// policy the module doc names (`BlockId = layer`, every forward pass visits blocks in the
    /// same fixed order). This is NOT demand/LRU: under a cyclic sweep the block whose next use
    /// is FURTHEST away is precisely the one the sweep just passed, so a miss inserts at the COLD
    /// end (it won't be needed again until every other block has been visited) and a hit keeps
    /// its position (promotion would let the sweep evict the stable set). The cache converges to
    /// a stable resident prefix of `n_slots - 1` blocks re-hit every pass plus one churn slot the
    /// remaining blocks cycle through: `n_slots - 1` hits per pass (see
    /// `schedule_cyclic_sweep_keeps_a_stable_prefix`). That is close to Belady's offline optimum
    /// but not equal to it — MIN rotates which blocks it keeps — and in steady state it misses
    /// about `(n_slots - 1) / (n_blocks - 1)` more blocks per pass than MIN, under one (the
    /// simulator measures it: `sim`'s `schedule_on_a_sweep_is_within_one_miss_per_pass_of_belady`).
    ///
    /// Mechanically this shares [`Self::touch_cold`]'s cold-end insertion; it is a separate entry
    /// point because the CONTRACT differs: `touch_cold` promises scan-resistance for full-set
    /// sweeps MIXED with recency (`touch`) traffic on the same pager, while `schedule` promises
    /// the stable prefix for a pager driven EXCLUSIVELY by one fixed cyclic order (dense layer
    /// streaming never mixes policies on a pool). Within-batch safety still rides the batch
    /// epoch ([`Self::begin_batch`]), not insertion order.
    pub fn schedule(&mut self, id: BlockId) -> Resolution {
        self.touch_cold_untraced(id)
    }

    /// [`Self::touch`]'s SCAN-RESISTANT twin, for full-set sweeps (a batched-prefill layer
    /// pre-ensuring all `n_expert` experts, layer 0..n cyclically): hits keep their LRU position
    /// (no MRU promotion) and misses insert at the COLD end. Under plain LRU a cyclic sweep
    /// larger than the cache is the pathological worst case — every block evicted exactly before
    /// its next use, 0% steady-state reuse (measured on Scout pp512: 768/768 blocks re-uploaded
    /// per rep). Cold insertion makes the sweep churn a single victim region instead and
    /// converges to a stable resident prefix of ~`n_slots` blocks — near the offline-optimal
    /// hit count for a repeated sweep — while recency-driven [`Self::touch`] users (decode's
    /// routed-only path) keep their hot set at the MRU end, out of the sweep's victim zone.
    /// Within-batch safety comes from the batch epoch (see [`Self::begin_batch`]), NOT ordering.
    pub fn touch_cold(&mut self, id: BlockId) -> Resolution {
        self.touch_cold_untraced(id)
    }

    fn touch_cold_untraced(&mut self, id: BlockId) -> Resolution {
        self.epoch.insert(id, self.cur_epoch);
        if let Some(&slot) = self.resident.get(&id) {
            self.stats.hits += 1;
            return Resolution::Hit { slot };
        }
        self.stats.misses += 1;
        let (slot, evicted) = self.take_slot();
        self.admit(id, slot);
        self.mark_cold(id);
        Resolution::Miss { slot, evicted }
    }

    /// A free slot, evicting if none remain, or `None` when every resident block is protected
    /// (pinned, or touched by the current batch). [`Self::take_slot`] is the infallible form the
    /// unpinned callers use.
    fn take_slot_opt(&mut self) -> Option<(u32, Option<BlockId>)> {
        if let Some(s) = self.free.pop() {
            return Some((s, None));
        }
        // A block is protected by its PIN always, and by the batch epoch only once batches are in
        // use: a caller that never opens one leaves `cur_epoch` at 0, which every block's default
        // epoch also is, so an unconditional epoch test would mark the whole cache as in-batch and
        // find no victim at all. (The infallible `take_slot` expresses the same exemption as its
        // `cur_epoch == 0` assert.)
        let batches = self.cur_epoch != 0;
        let victim = self.coldest(batches, |b| {
            !(batches && self.epoch.get(&b) == Some(&self.cur_epoch))
                && !self.pinned.contains_key(&b)
        })?;
        self.recency.remove(&victim);
        let vslot = self
            .resident
            .remove(&victim)
            .expect("every ranked block has a resident mapping");
        // Drop the victim's epoch entry too — otherwise `epoch` is the one pager structure not
        // bounded by `n_slots`, accumulating a stale entry per distinct BlockId ever touched. A
        // stale entry could also mask an id as "current batch" across a `cur_epoch` wraparound.
        self.epoch.remove(&victim);
        self.group_removed(victim);
        self.stats.evictions += 1;
        Some((vslot, Some(victim)))
    }

    /// A free slot, evicting if none remain. The victim is the coldest block NOT touched in the
    /// current batch (epoch guard — see [`Self::begin_batch`]); legacy callers that never open a
    /// batch share epoch 0 everywhere, for which the guard degrades to plain LRU.
    fn take_slot(&mut self) -> (u32, Option<BlockId>) {
        if self.free.is_empty() {
            // Only reachable through the infallible entry points, whose callers hold no pins.
            debug_assert!(
                self.pinned.is_empty(),
                "pins are for the fallible resolve_and_pin path; touch/schedule cannot see them"
            );
        }
        if let Some(s) = self.free.pop() {
            return (s, None);
        }
        let victim = self
            .coldest(true, |b| self.epoch.get(&b) != Some(&self.cur_epoch))
            // Every resident block touched by the CURRENT batch means the batch exceeded
            // n_slots — the sizing floor (`Pager::new`'s doc) was violated upstream. Falling
            // back to plain LRU would silently evict a batch sibling mid-flight (the
            // coherent-but-wrong class); fail loudly instead when batches are in use.
            .unwrap_or_else(|| {
                // Actionable in the voice of the Vulkan VRAM guard (`check_vram_budget`): say what
                // was asked for, say why it cannot be tolerated rather than degraded, and name the
                // knob to turn. The knob is `INFR_CACHE` — the paging budget (`paging.cache`) the
                // seam splits proportionally into per-pool slot counts, so raising it is what makes
                // `n_slots` clear the batch's width. NOT reachable on a default configuration: the
                // MoE seam floors every pool at `min(n_expert, n_blocks)` slots (a batch is one
                // (layer, role) resolution, at most `n_expert` distinct experts) and the dense
                // streaming path opens a fresh batch around each single `schedule` call, so both
                // shipping callers satisfy the floor by construction — reaching this means an
                // explicit `INFR_CACHE` too small for one dispatch, or a new caller that skipped
                // the floor.
                assert!(
                    self.cur_epoch == 0,
                    "pager: a single dispatch batch touched more distinct blocks than the \
                     n_slots={} this cache holds — within-batch eviction safety is unsatisfiable. \
                     Refusing to degrade: evicting a block this same batch already resolved \
                     doesn't fail cleanly, it feeds the dispatch another block's bytes (silent \
                     wrong output, not an error). Raise the paging budget (INFR_CACHE) so every \
                     pool gets at least one slot per block a batch touches, or leave it unset to \
                     take the auto budget, which is sized to that floor.",
                    self.n_slots
                );
                // Epoch never used (legacy caller): plain LRU.
                self.coldest(false, |_| true)
                    .expect("pager: a full cache with no resident block to evict")
            });
        self.recency.remove(&victim);
        let vslot = self
            .resident
            .remove(&victim)
            .expect("every ranked block has a resident mapping");
        // Drop the victim's epoch entry too — otherwise `epoch` is the one pager structure not
        // bounded by `n_slots`, accumulating a stale entry per distinct BlockId ever touched. A
        // stale entry could also mask an id as "current batch" across a `cur_epoch` wraparound.
        self.epoch.remove(&victim);
        self.group_removed(victim);
        self.stats.evictions += 1;
        (vslot, Some(victim))
    }
}

/// A group with no [`Pager::touch_group`] stamp in force.
const NO_STAMP: i64 = i64::MIN;

/// The per-group state. Indexed by group, `id / len`; grown as blocks of higher groups arrive.
#[derive(Debug)]
struct Groups {
    len: u32,
    /// `log2(len)` when `len` is a power of two (an MoE layer's expert count, nearly always), so
    /// the victim scan's per-block group lookup is a shift rather than a division.
    shift: Option<u32>,
    /// Resident members per group.
    resident: Vec<u32>,
    /// The stamp the group's last [`Pager::touch_group`] gave its first member (member `k` has
    /// `stamp + k`), or [`NO_STAMP`].
    stamp: Vec<i64>,
    /// The batch epoch of that touch, meaningful only while `stamp` is set.
    epoch: Vec<u64>,
    /// How many groups have a stamp in force — zero keeps victim selection on its ungrouped path.
    stamped: usize,
}

impl Groups {
    fn new(len: u32) -> Self {
        Self {
            len,
            shift: len.is_power_of_two().then(|| len.trailing_zeros()),
            resident: Vec::new(),
            stamp: Vec::new(),
            epoch: Vec::new(),
            stamped: 0,
        }
    }

    fn index(&self, id: BlockId) -> usize {
        match self.shift {
            Some(s) => (id >> s) as usize,
            None => (id / self.len) as usize,
        }
    }

    /// `id`'s position in its group, as a stamp offset.
    fn offset(&self, id: BlockId) -> i64 {
        i64::from(match self.shift {
            Some(_) => id & (self.len - 1),
            None => id % self.len,
        })
    }

    /// Whether any group's stamp is in force.
    fn stamped(&self) -> bool {
        self.stamped > 0
    }

    /// The stamp in force for `id`'s group, and its epoch.
    fn stamp_of(&self, id: BlockId) -> Option<(i64, u64)> {
        let g = self.index(id);
        match self.stamp.get(g) {
            Some(&s) if s != NO_STAMP => Some((s, self.epoch[g])),
            _ => None,
        }
    }

    fn clear_stamp(&mut self, g: usize) -> Option<(i64, u64)> {
        let s = std::mem::replace(&mut self.stamp[g], NO_STAMP);
        (s != NO_STAMP).then(|| {
            self.stamped -= 1;
            (s, self.epoch[g])
        })
    }

    fn admitted(&mut self, id: BlockId) {
        let g = self.index(id);
        if g >= self.resident.len() {
            self.resident.resize(g + 1, 0);
            self.stamp.resize(g + 1, NO_STAMP);
            self.epoch.resize(g + 1, 0);
        }
        self.resident[g] += 1;
    }

    fn removed(&mut self, id: BlockId) {
        let g = self.index(id);
        self.resident[g] -= 1;
        if self.resident[g] == 0 {
            // No member left for the stamp to apply to; keep `stamped` an exact count.
            self.clear_stamp(g);
        }
    }

    /// The group whose first member is `first`, which must be a group boundary.
    fn group_of_first(&self, first: BlockId) -> usize {
        assert!(
            first.is_multiple_of(self.len),
            "pager: block {first} does not start a group of {}",
            self.len
        );
        self.index(first)
    }
}

impl Pager {
    /// Group this pager's blocks into runs of `len`: block `id` belongs to group `id / len`. Once
    /// grouped, [`Self::group_is_resident`] and [`Self::touch_group`] answer for a whole group in
    /// O(1). May be called at any time (the resident blocks are counted then); calling it again
    /// with the same `len` changes nothing.
    ///
    /// Grouping changes no outcome by itself, so it is not a traced call: a trace records each
    /// group touch with its length, and replay groups the pager from that.
    ///
    /// # Panics
    /// If `len` is zero, or the pager is already grouped by a different length — a block cannot
    /// belong to two groupings, and silently regrouping would leave the old groups' stamps applied
    /// to the wrong blocks.
    pub fn group_by(&mut self, len: u32) {
        assert!(len > 0, "pager: a block group must hold at least one block");
        if let Some(g) = &self.groups {
            assert_eq!(
                g.len, len,
                "pager: blocks are already grouped by {}, not {len}",
                g.len
            );
            return;
        }
        let mut g = Groups::new(len);
        for &id in self.resident.keys() {
            g.admitted(id);
        }
        self.groups = Some(g);
    }

    /// Whether every block of the group starting at `first` is resident. O(1): a count kept on
    /// every admission and removal (a miss, a victim, [`Self::evict`]).
    ///
    /// # Panics
    /// If the pager is not grouped, or `first` is not a group boundary.
    pub fn group_is_resident(&self, first: BlockId) -> bool {
        let g = self
            .groups
            .as_ref()
            .expect("pager: group_is_resident before group_by");
        let i = g.group_of_first(first);
        g.resident.get(i).copied() == Some(g.len)
    }

    /// Touch every block of the group starting at `first` as one call, if every one of them is
    /// resident: exactly what [`Self::touch`] on each member in ascending id order would do (see
    /// the module doc), in O(1) instead of O(group). Returns `false`, and changes and counts
    /// nothing, when a member is absent — a group touch never admits a block, since only the
    /// caller can supply a missing block's bytes.
    ///
    /// # Panics
    /// As [`Self::group_is_resident`].
    pub fn touch_group(&mut self, first: BlockId) -> bool {
        self.touch_group_untraced(first)
    }

    fn touch_group_untraced(&mut self, first: BlockId) -> bool {
        if !self.group_is_resident(first) {
            return false;
        }
        let g = self.groups.as_mut().expect("checked above");
        let i = g.group_of_first(first);
        if g.stamp[i] == NO_STAMP {
            g.stamped += 1;
        }
        // The stamps the member touches would have taken: `hot + 1 ..= hot + len`, in id order.
        g.stamp[i] = self.hot + 1;
        g.epoch[i] = self.cur_epoch;
        self.hot += i64::from(g.len);
        self.stats.hits += u64::from(g.len);
        true
    }

    /// Bookkeeping for `id` becoming resident: count it, and first write a group stamp in force
    /// into the members it covers (see the module doc for why an admission ends it).
    fn group_admit(&mut self, id: BlockId) {
        let Self {
            groups,
            recency,
            epoch,
            ..
        } = self;
        let Some(g) = groups.as_mut() else {
            return;
        };
        let i = g.index(id);
        if i < g.stamp.len() {
            if let Some((stamp, ep)) = g.clear_stamp(i) {
                let first = id - id % g.len;
                for k in 0..g.len {
                    let b = first + k;
                    if let Some(r) = recency.get_mut(&b) {
                        *r = (*r).max(stamp + i64::from(k));
                    }
                    if let Some(e) = epoch.get_mut(&b) {
                        *e = (*e).max(ep);
                    }
                }
            }
        }
        g.admitted(id);
    }

    /// Bookkeeping for `id` leaving the cache.
    fn group_removed(&mut self, id: BlockId) {
        if let Some(g) = self.groups.as_mut() {
            g.removed(id);
        }
    }
}

mod tests {
    use super::super::shapes::{cyclic_sweep, Moe, XorShift64};
    use super::super::trace::{Call, Outcome};
    use super::super::{BlockId, Insert, Pager, Resolution};
    use super::Pager as Reference;

    /// [`Call::apply`] for the reference model.
    fn apply_ref(call: Call, r: &mut Reference) -> Outcome {
        match call {
            Call::BeginBatch => {
                r.begin_batch();
                Outcome::Done
            }
            Call::Touch(id) => Outcome::Resolved(r.touch(id)),
            Call::TouchCold(id) => Outcome::Resolved(r.touch_cold(id)),
            Call::Schedule(id) => Outcome::Resolved(r.schedule(id)),
            Call::ResolveAndPin(id, insert) => r
                .resolve_and_pin(id, insert)
                .map_or(Outcome::Refused, Outcome::Resolved),
            Call::PinIfResident(id) => Outcome::Slot(r.pin_if_resident(id)),
            Call::Repin(id) => Outcome::Slot(r.repin(id)),
            Call::Unpin(id) => {
                r.unpin(id);
                Outcome::Done
            }
            Call::Evict(id) => Outcome::Slot(r.evict(id)),
            Call::TouchGroup(first, len) => {
                r.group_by(len);
                Outcome::Grouped(r.touch_group(first))
            }
        }
    }

    /// Whether a miss through an infallible entry point (`touch`, `touch_cold`, `schedule`) would
    /// panic on `r`: the debug assertion against pins held, or a batch wider than the cache. The
    /// differential run keeps to the contract instead; the panics have tests of their own.
    fn infallible_miss_panics(r: &Reference) -> bool {
        if !r.free.is_empty() {
            return false;
        }
        if !r.pinned.is_empty() {
            return true;
        }
        let cur = r.cur_epoch;
        cur != 0
            && r.resident.keys().all(|b| {
                r.epoch.get(b) == Some(&cur)
                    || r.groups
                        .as_ref()
                        .and_then(|g| g.stamp_of(*b))
                        .is_some_and(|(_, e)| e == cur)
            })
    }

    /// Whether `id`'s group has a group touch in force in `r`.
    fn group_touched(r: &Reference, id: BlockId) -> bool {
        r.groups.as_ref().is_some_and(|g| g.stamp_of(id).is_some())
    }

    /// Both pagers side by side: every call goes to both, and every outcome, counter and piece of
    /// observable state must agree.
    struct Pair {
        a: Pager,
        r: Reference,
        n_ids: u32,
        what: String,
        calls: usize,
    }

    impl Pair {
        fn new(n_slots: usize, n_ids: u32, what: String) -> Self {
            Self {
                a: Pager::new(n_slots),
                r: Reference::new(n_slots),
                n_ids,
                what,
                calls: 0,
            }
        }

        fn group_by(&mut self, len: u32) {
            self.a.group_by(len);
            self.r.group_by(len);
        }

        fn apply(&mut self, call: Call) -> Outcome {
            let got = call.apply(&mut self.a);
            let want = apply_ref(call, &mut self.r);
            self.calls += 1;
            let at = || format!("{}, call {} ({call:?})", self.what, self.calls);
            assert_eq!(got, want, "{}", at());
            let (sa, sr) = (self.a.stats(), self.r.stats());
            assert_eq!(
                (sa.hits, sa.misses, sa.evictions),
                (sr.hits, sr.misses, sr.evictions),
                "{}",
                at()
            );
            assert_eq!(self.a.resident_count(), self.r.resident_count(), "{}", at());
            assert_eq!(self.a.pinned_blocks(), self.r.pinned_blocks(), "{}", at());
            got
        }

        /// `slot_of` for every id, and `group_is_resident` for every group.
        fn check_state(&self) {
            for id in 0..self.n_ids {
                assert_eq!(
                    self.a.slot_of(id),
                    self.r.slot_of(id),
                    "{}, after call {}: slot_of({id})",
                    self.what,
                    self.calls
                );
            }
            if let Some(len) = self.a.group_len() {
                for first in (0..self.n_ids).step_by(len as usize) {
                    assert_eq!(
                        self.a.group_is_resident(first),
                        self.r.group_is_resident(first),
                        "{}, after call {}: group_is_resident({first})",
                        self.what,
                        self.calls
                    );
                }
            }
        }
    }

    /// One configuration of the random differential run.
    #[derive(Debug, Clone, Copy)]
    struct Case {
        n_slots: usize,
        n_ids: u32,
        /// Group length, and the step at which both pagers are grouped (0: from the start).
        group: Option<(u32, usize)>,
        /// One `begin_batch` per this many steps on average, or never (the legacy path).
        batch_every: Option<usize>,
        steps: usize,
    }

    /// How often the run reached the cases the refactor had to get right, so a run that never
    /// reaches one fails instead of passing vacuously.
    #[derive(Debug, Default)]
    struct Reached {
        group_touches: u64,
        refusals: u64,
        evictions_under_group_touch: u64,
        admissions_into_touched_group: u64,
        pins_across_batches: u64,
        cold_hits: u64,
        legacy_evictions: u64,
        readmits_after_group_touch: u64,
    }

    fn run(case: Case, seed: u64, reached: &mut Reached) {
        let mut rng = XorShift64::new(seed);
        let mut p = Pair::new(
            case.n_slots,
            case.n_ids,
            format!("{case:?}, seed {seed:#x}"),
        );
        let n_ids = case.n_ids as usize;
        // A drifting window of ids most calls draw from, so blocks are reused and groups fill.
        let width = (2 * case.n_slots).min(n_ids);
        let mut window = 0;
        let mut held: Vec<BlockId> = Vec::new();
        let full_check_every = if n_ids <= 64 { 1 } else { 64 };
        for step in 0..case.steps {
            if let Some((len, at)) = case.group {
                if step == at {
                    p.group_by(len);
                }
            }
            let len = p.a.group_len();
            if rng.below(48) == 0 {
                window = rng.below(n_ids);
            }
            let id = if rng.below(4) == 0 {
                rng.below(n_ids)
            } else {
                (window + rng.below(width)) % n_ids
            } as BlockId;
            // Keep the pins from starving every admission: past a slot's worth, release first.
            let kind = if held.len() > case.n_slots {
                12
            } else {
                rng.below(26)
            };
            match kind {
                0 => {
                    if let Some(every) = case.batch_every {
                        if rng.below(every) < 22 {
                            if !held.is_empty() {
                                reached.pins_across_batches += 1;
                            }
                            p.apply(Call::BeginBatch);
                        }
                    }
                }
                1..=7 => {
                    let call = match kind {
                        1..=3 => Call::Touch(id),
                        4..=6 => Call::TouchCold(id),
                        _ => Call::Schedule(id),
                    };
                    let resident = p.r.slot_of(id).is_some();
                    if !resident && infallible_miss_panics(&p.r) {
                        continue;
                    }
                    if !resident && group_touched(&p.r, id) {
                        reached.admissions_into_touched_group += 1;
                    }
                    let touched = p.r.groups.as_ref().is_some_and(|g| g.stamped());
                    let legacy = p.r.cur_epoch == 0;
                    match p.apply(call) {
                        Outcome::Resolved(Resolution::Hit { .. })
                            if !matches!(call, Call::Touch(_)) =>
                        {
                            reached.cold_hits += 1
                        }
                        Outcome::Resolved(Resolution::Miss {
                            evicted: Some(_), ..
                        }) => {
                            reached.evictions_under_group_touch += u64::from(touched);
                            reached.legacy_evictions += u64::from(legacy);
                        }
                        _ => {}
                    }
                }
                8 | 9 => {
                    let insert = if kind == 8 { Insert::Mru } else { Insert::Cold };
                    if p.r.slot_of(id).is_none() && group_touched(&p.r, id) {
                        reached.admissions_into_touched_group += 1;
                    }
                    let touched = p.r.groups.as_ref().is_some_and(|g| g.stamped());
                    match p.apply(Call::ResolveAndPin(id, insert)) {
                        Outcome::Refused => reached.refusals += 1,
                        o => {
                            if let Outcome::Resolved(Resolution::Miss {
                                evicted: Some(_), ..
                            }) = o
                            {
                                reached.evictions_under_group_touch += u64::from(touched);
                            }
                            if rng.below(2) == 0 {
                                p.apply(Call::Unpin(id));
                            } else {
                                held.push(id);
                            }
                        }
                    }
                }
                10 | 11 => {
                    let call = if kind == 10 {
                        Call::PinIfResident(id)
                    } else {
                        Call::Repin(id)
                    };
                    if let Outcome::Slot(Some(_)) = p.apply(call) {
                        held.push(id);
                    }
                }
                12 | 13 => {
                    if !held.is_empty() {
                        let id = held.swap_remove(rng.below(held.len()));
                        p.apply(Call::Unpin(id));
                    }
                }
                14 => {
                    if !held.contains(&id) {
                        p.apply(Call::Evict(id));
                    }
                }
                15..=18 => {
                    if let Some(len) = len {
                        let first = id - id % len;
                        if let Outcome::Grouped(true) = p.apply(Call::TouchGroup(first, len)) {
                            reached.group_touches += 1;
                        }
                    }
                }
                22..=24 => {
                    // In one batch: a group touch, one member evicted and re-admitted (which ends
                    // the touch), then another admission, whose victim walk must still see the
                    // batch protection the group touch gave the other members.
                    let Some(len) = len else { continue };
                    if case.batch_every.is_none() {
                        continue;
                    }
                    let first = id - id % len;
                    let m = first + rng.below(len as usize) as BlockId;
                    if m >= case.n_ids || !p.r.group_is_resident(first) || held.contains(&m) {
                        continue;
                    }
                    p.apply(Call::BeginBatch);
                    p.apply(Call::TouchGroup(first, len));
                    p.apply(Call::Evict(m));
                    reached.admissions_into_touched_group += 1;
                    if p.apply(Call::ResolveAndPin(m, Insert::Cold)) == Outcome::Refused {
                        continue;
                    }
                    let other = rng.below(n_ids) as BlockId;
                    if p.r.slot_of(other).is_none() {
                        reached.readmits_after_group_touch += 1;
                        match p.apply(Call::ResolveAndPin(other, Insert::Mru)) {
                            Outcome::Refused => reached.refusals += 1,
                            _ => {
                                p.apply(Call::Unpin(other));
                            }
                        }
                    }
                    p.apply(Call::Unpin(m));
                }
                _ => {
                    // A sweep of one group, member by member (a prefill layer, or a decode layer
                    // touching every expert): what makes whole groups resident.
                    let Some(len) = len else { continue };
                    let first = id - id % len;
                    let cold = rng.below(2) == 0;
                    if case.batch_every.is_some() {
                        p.apply(Call::BeginBatch);
                    }
                    for m in first..(first + len).min(case.n_ids) {
                        if p.r.slot_of(m).is_none() && infallible_miss_panics(&p.r) {
                            break;
                        }
                        p.apply(if cold {
                            Call::TouchCold(m)
                        } else {
                            Call::Touch(m)
                        });
                    }
                }
            }
            if step % full_check_every == 0 {
                p.check_state();
            }
        }
        p.check_state();
        while let Some(id) = held.pop() {
            p.apply(Call::Unpin(id));
        }
        p.check_state();
    }

    /// The refactored pager against the old one on long random call sequences over every public
    /// mutating call: small and large pools, no groups and groups of power-of-two and other
    /// lengths (grouped from the start and part-way through), batches of every width up to wider
    /// than the pool (refused admissions), the legacy path that never opens a batch, and pins held
    /// across batches. Every outcome and counter is compared after every call, and `slot_of` of
    /// every id and `group_is_resident` of every group along the way.
    #[test]
    fn matches_the_reference_on_random_calls() {
        let mut cases = Vec::new();
        for n_slots in [1, 2, 3, 4, 5, 8, 13] {
            for group in [
                None,
                Some((1, 0)),
                Some((2, 0)),
                Some((3, 0)),
                Some((4, 0)),
                Some((4, 150)),
                Some((5, 300)),
            ] {
                for batch_every in [None, Some(22), Some(200)] {
                    cases.push(Case {
                        n_slots,
                        n_ids: 3 * n_slots as u32 + 4,
                        group,
                        batch_every,
                        steps: 800,
                    });
                }
            }
        }
        for (n_slots, group) in [
            (64, None),
            (64, Some((16, 0))),
            (64, Some((6, 0))),
            (200, None),
            (200, Some((32, 0))),
            (200, Some((12, 900))),
            (300, Some((128, 0))),
        ] {
            for batch_every in [None, Some(22), Some(400), Some(5000)] {
                cases.push(Case {
                    n_slots,
                    n_ids: 3 * n_slots as u32,
                    group,
                    batch_every,
                    steps: 5000,
                });
            }
        }
        let mut reached = Reached::default();
        let mut rng = XorShift64::new(0x5107_7a9e);
        for case in cases {
            for _ in 0..3 {
                run(case, rng.next_u64(), &mut reached);
            }
        }
        let Reached {
            group_touches,
            refusals,
            evictions_under_group_touch,
            admissions_into_touched_group,
            pins_across_batches,
            cold_hits,
            legacy_evictions,
            readmits_after_group_touch,
        } = reached;
        for (what, n) in [
            ("group touches", group_touches),
            ("refused admissions", refusals),
            ("evictions under a group touch", evictions_under_group_touch),
            (
                "admissions into a touched group",
                admissions_into_touched_group,
            ),
            ("pins held across a batch", pins_across_batches),
            ("cold-end hits", cold_hits),
            ("evictions without batches", legacy_evictions),
            (
                "admissions after a group touch and a re-admitted member",
                readmits_after_group_touch,
            ),
        ] {
            assert!(n >= 1000, "only {n} {what}: the case was barely exercised");
        }
    }

    /// The access patterns real sessions make (`shapes`), through both pagers: dense sweeps,
    /// MoE prefill, routed decode over constrained pools, and a decode loop that touches a wholly
    /// resident layer as a group and the others through their routed experts — the `paging.cache`
    /// regime where the group touches and the misses meet.
    #[test]
    fn matches_the_reference_on_session_shapes() {
        let moe = Moe {
            n_layers: 6,
            n_expert: 32,
            top_k: 4,
            zipf_s: 1.0,
        };
        let n = moe.n_blocks();
        let mut traces = vec![cyclic_sweep(40, 8, 6), cyclic_sweep(40, 32, 6)];
        for slots in [n / 4, n / 2, 3 * n / 4] {
            traces.push(moe.prefill(slots, 2));
            traces.push(moe.decode_routed(slots, 40, 7));
            traces.push(moe.session(slots, 40, 7));
        }
        traces.push(moe.decode_resident_grouped(4));
        for t in &traces {
            let mut p = Pair::new(t.n_slots, n as u32, t.label.clone());
            for &c in &t.calls {
                p.apply(c);
            }
            p.check_state();
        }

        // Decode on a pool that holds some layers whole: a resident layer is touched as a group,
        // the others route.
        let mut group_touches = 0;
        for slots in [n / 3, n / 2, 3 * n / 4] {
            let mut p = Pair::new(slots, n as u32, format!("mixed decode, {slots} slots"));
            let mut calls = Vec::new();
            moe.prefill_chunk(&mut calls);
            for c in calls {
                p.apply(c);
            }
            p.group_by(moe.n_expert);
            let mut router = moe.router(11);
            for _ in 0..60 {
                let mut token = Vec::new();
                moe.decode_token_routed(&mut router, &mut token);
                let mut layer = 0;
                let mut whole = false;
                for c in token {
                    if c == Call::BeginBatch {
                        let first = layer * moe.n_expert;
                        layer += 1;
                        p.apply(c);
                        whole = p.a.group_is_resident(first);
                        if whole {
                            p.apply(Call::TouchGroup(first, moe.n_expert));
                            group_touches += 1;
                        }
                    } else if !whole {
                        p.apply(c);
                    }
                }
                p.check_state();
            }
        }
        assert!(
            group_touches > 100,
            "only {group_touches} whole-layer touches"
        );
    }
}
