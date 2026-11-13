//! Equal-size block groups: a pool whose blocks come in runs that a caller uses as one unit (an
//! MoE pool's experts of one layer) can ask, in O(1), whether a whole group is resident, and touch
//! the whole group at once.
//!
//! [`Pager::touch_group`] is defined as exactly what touching every member with [`Pager::touch`]
//! in ascending id order does when all of them are resident — the same hits counted, the same
//! batch protection, and the same recency order afterwards, down to the order among the members —
//! but it writes one stamp for the group instead of one per member. A member's recency is then the
//! later of its own stamp and the one the group stamp gives it (`stamp + offset`, the stamp the
//! `touch` of that member would have taken), and its batch epoch the later of its own and the
//! group's. Victim selection reads both ([`Pager::coldest`]).
//!
//! The group stamp stays exact while the group's members only leave or are touched: a member
//! touched later takes a stamp above the group's, one that leaves is out of the ranking. What
//! would break it is a member ADMITTED after the group touch, which must rank by its own insert
//! alone (a cold-end insert especially). So an admission into a stamped group first writes the
//! group stamp into every member still resident and drops it — O(group), at most once per group
//! touch.

use super::trace::{Call, Outcome};
use super::{BlockId, Pager};

/// A group with no [`Pager::touch_group`] stamp in force.
const NO_STAMP: i64 = i64::MIN;

/// The per-group state. Indexed by group, `id / len`; grown as blocks of higher groups arrive.
#[derive(Debug)]
pub(super) struct Groups {
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
    pub(super) fn offset(&self, id: BlockId) -> i64 {
        i64::from(match self.shift {
            Some(_) => id & (self.len - 1),
            None => id % self.len,
        })
    }

    /// Whether any group's stamp is in force.
    pub(super) fn stamped(&self) -> bool {
        self.stamped > 0
    }

    /// The stamp in force for `id`'s group, and its epoch.
    pub(super) fn stamp_of(&self, id: BlockId) -> Option<(i64, u64)> {
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

    /// The group length set by [`Self::group_by`], if any.
    pub fn group_len(&self) -> Option<u32> {
        self.groups.as_ref().map(|g| g.len)
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
        let r = self.touch_group_untraced(first);
        let len = self.groups.as_ref().map_or(0, |g| g.len);
        self.record(Call::TouchGroup(first, len), Outcome::Grouped(r));
        r
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
    pub(super) fn group_admit(&mut self, id: BlockId) {
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
    pub(super) fn group_removed(&mut self, id: BlockId) {
        if let Some(g) = self.groups.as_mut() {
            g.removed(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::shapes::XorShift64;
    use super::super::{Insert, Resolution};
    use super::*;

    const LEN: u32 = 4;

    fn grouped(n_slots: usize) -> Pager {
        let mut p = Pager::new(n_slots);
        p.group_by(LEN);
        p
    }

    fn fill_group(p: &mut Pager, group: u32) {
        for id in group * LEN..(group + 1) * LEN {
            p.touch(id);
        }
    }

    /// The count follows every way a block leaves — `evict` (a failed fill's rollback, a discarded
    /// recording's undo), a victim of `touch`, of `touch_cold` and of `resolve_and_pin` — and every
    /// way one is admitted: a group stops being resident the moment one member leaves, whichever
    /// path took it, and is resident again once the member is back.
    #[test]
    fn group_residency_follows_every_admission_and_removal() {
        type Remove = fn(&mut Pager) -> BlockId;
        let paths: [(&str, Remove); 4] = [
            ("evict", |p| {
                p.evict(1);
                1
            }),
            ("a touch victim", |p| victim(p.touch(LEN))),
            ("a touch_cold victim", |p| {
                p.begin_batch();
                victim(p.touch_cold(LEN))
            }),
            ("a resolve_and_pin victim", |p| {
                let r = p.resolve_and_pin(LEN, Insert::Mru).expect("a victim");
                p.unpin(LEN);
                victim(r)
            }),
        ];
        for (what, remove) in paths {
            let mut p = grouped(LEN as usize);
            assert!(!p.group_is_resident(0), "{what}");
            fill_group(&mut p, 0);
            assert!(p.group_is_resident(0), "{what}: a filled group");
            let gone = remove(&mut p);
            assert!(gone < LEN, "{what}: removed {gone}, outside the group");
            assert!(!p.group_is_resident(0), "{what} left the group resident");
            // Back in (through a free slot, when the removal admitted a block of another group).
            p.evict(LEN);
            p.touch(gone);
            assert!(p.group_is_resident(0), "{what}: re-admission not counted");
        }
    }

    fn victim(r: Resolution) -> BlockId {
        match r {
            Resolution::Miss {
                evicted: Some(v), ..
            } => v,
            other => panic!("expected an eviction, got {other:?}"),
        }
    }

    /// A group touch on a group with an absent member is refused and changes nothing: no counter,
    /// no recency (block 0 stays the LRU victim), no admission.
    #[test]
    fn a_partly_resident_group_touch_is_refused() {
        let mut p = grouped(4);
        for id in [0, 1, 2] {
            p.touch(id);
        }
        p.touch(4);
        let before = p.stats();
        assert!(!p.touch_group(0));
        let after = p.stats();
        assert_eq!(
            (after.hits, after.misses, after.evictions),
            (before.hits, before.misses, before.evictions)
        );
        assert_eq!(p.slot_of(3), None, "a refused group touch admitted a block");
        assert_eq!(
            p.touch(5),
            Resolution::Miss {
                slot: 0,
                evicted: Some(0)
            },
            "a refused group touch moved the recency order"
        );
    }

    /// The batch guard covers a group touch as it covers a member's touch: in the batch that
    /// touched the group, no member is a victim. A group touch takes the hottest stamps, so this
    /// shows only when nothing outside the batch is left to evict — the admission is refused, as it
    /// is when every member was touched one by one, instead of displacing a member the batch's
    /// dispatch is about to read.
    #[test]
    fn a_group_touch_protects_its_members_for_the_batch() {
        let mut p = grouped(5);
        fill_group(&mut p, 0);
        p.touch(4);
        p.begin_batch();
        assert!(p.touch_group(0));
        p.touch(4);
        assert_eq!(
            p.resolve_and_pin(8, Insert::Mru),
            None,
            "a block of the batch's touched group was given up"
        );
        // Next batch: the guard is gone, and the group's first member is the coldest block.
        p.begin_batch();
        assert_eq!(
            p.touch(9),
            Resolution::Miss {
                slot: 0,
                evicted: Some(0)
            }
        );
    }

    /// The policy on an almost-resident pool, as a decode loop drives it: each token touches the
    /// fully resident groups whole and the partly resident one through its routed blocks. Every
    /// whole-group touch is more recent than the routed blocks the previous token left behind, so
    /// a routed miss takes its victim from those, never from a group in use every token — what
    /// touching every member did, at one call per group.
    #[test]
    fn a_routed_miss_does_not_take_a_group_in_use_every_token() {
        // Slots for two whole groups and two blocks of a third.
        let mut p = grouped(2 * LEN as usize + 2);
        fill_group(&mut p, 0);
        fill_group(&mut p, 1);
        let mut evicted = 0;
        for token in 0..8 {
            p.begin_batch();
            assert!(p.touch_group(0), "token {token}");
            p.begin_batch();
            assert!(p.touch_group(LEN), "token {token}");
            // The third group's routed pair alternates, so every token misses twice.
            let pair = if token % 2 == 0 { [8, 9] } else { [10, 11] };
            p.begin_batch();
            for id in pair {
                if let Resolution::Miss {
                    evicted: Some(v), ..
                } = p.touch(id)
                {
                    assert!(
                        v >= 2 * LEN,
                        "token {token}: a routed miss evicted block {v}"
                    );
                    evicted += 1;
                }
            }
        }
        assert!(
            evicted >= 12,
            "only {evicted} evictions — the case never arose"
        );
        assert!(p.group_is_resident(0) && p.group_is_resident(LEN));
    }

    #[test]
    #[should_panic(expected = "already grouped by 4, not 8")]
    fn regrouping_by_another_length_panics() {
        let mut p = grouped(4);
        p.group_by(LEN);
        p.group_by(8);
    }

    #[test]
    #[should_panic(expected = "does not start a group")]
    fn a_group_touch_off_a_boundary_panics() {
        grouped(4).touch_group(1);
    }

    /// Grouping a pager that already holds blocks counts them.
    #[test]
    fn grouping_late_counts_the_resident_blocks() {
        let mut p = Pager::new(8);
        for id in 0..LEN {
            p.touch(id);
        }
        p.group_by(LEN);
        assert!(p.group_is_resident(0));
        assert!(!p.group_is_resident(LEN));
    }

    /// The definition, checked: on random call sequences, a pager driven with `touch_group` and
    /// one driven with a `touch` of every member in id order return the same outcome for every
    /// other call — every slot and every victim — and the same counters, through admissions into
    /// stamped groups, evictions, cold inserts, pins and batches. A batch here never fills the
    /// pool, so the batch guard on a group's members never decides a victim: that is
    /// `a_group_touch_protects_its_members_for_the_batch`. Run with a power-of-two group length
    /// and with one that is not, which the pager indexes differently.
    #[test]
    fn a_group_touch_is_a_touch_of_every_member() {
        for len in [LEN, 3] {
            a_group_touch_is_a_touch_of_every_member_by(len);
        }
    }

    fn a_group_touch_is_a_touch_of_every_member_by(len: u32) {
        let mut rng = XorShift64::new(0x6709);
        let n_groups = 3u32;
        let mut group_touches = 0;
        for case in 0..2000 {
            let n_slots = 2 * (len as usize) + case % 5;
            let (mut a, mut b) = (Pager::new(n_slots), Pager::new(n_slots));
            a.group_by(len);
            let mut batch: Vec<BlockId> = Vec::new();
            for step in 0..60 {
                let id = rng.below((n_groups * len) as usize) as BlockId;
                // 0 opens a batch, 1..=6 is a group touch (often, so groups are often resident when
                // touched), and the rest the other entry points, a plain `touch` the most.
                let kind = rng.below(16);
                let fresh = |batch: &[BlockId], ids: &[BlockId]| {
                    ids.iter().filter(|i| !batch.contains(i)).count()
                };
                let what = format!("case {case} step {step}");
                if kind == 0 {
                    a.begin_batch();
                    b.begin_batch();
                    batch.clear();
                    continue;
                }
                if (1..=6).contains(&kind) {
                    let first = id - id % len;
                    let members: Vec<BlockId> = (first..first + len).collect();
                    let all = members.iter().all(|&m| b.slot_of(m).is_some());
                    if all && batch.len() + fresh(&batch, &members) >= n_slots {
                        continue;
                    }
                    assert_eq!(a.touch_group(first), all, "{what}");
                    if all {
                        group_touches += 1;
                        for &m in &members {
                            assert!(matches!(b.touch(m), Resolution::Hit { .. }));
                            if !batch.contains(&m) {
                                batch.push(m);
                            }
                        }
                    }
                } else {
                    // Keep every batch inside the pager's sizing contract.
                    if batch.len() + fresh(&batch, &[id]) >= n_slots {
                        a.begin_batch();
                        b.begin_batch();
                        batch.clear();
                    }
                    match kind {
                        7 => assert_eq!(a.touch_cold(id), b.touch_cold(id), "{what}"),
                        8 => assert_eq!(a.schedule(id), b.schedule(id), "{what}"),
                        9 => {
                            assert_eq!(a.evict(id), b.evict(id), "{what}");
                            continue;
                        }
                        10 | 11 => {
                            let insert = if kind == 10 {
                                Insert::Mru
                            } else {
                                Insert::Cold
                            };
                            let r = a.resolve_and_pin(id, insert);
                            assert_eq!(r, b.resolve_and_pin(id, insert), "{what}");
                            if r.is_some() {
                                a.unpin(id);
                                b.unpin(id);
                            }
                        }
                        _ => assert_eq!(a.touch(id), b.touch(id), "{what}"),
                    }
                    if !batch.contains(&id) {
                        batch.push(id);
                    }
                }
                let (sa, sb) = (a.stats(), b.stats());
                assert_eq!(
                    (sa.hits, sa.misses, sa.evictions),
                    (sb.hits, sb.misses, sb.evictions),
                    "{what}"
                );
            }
        }
        assert!(
            group_touches > 5000,
            "len {len}: only {group_touches} group touches ran — the property was barely exercised"
        );
    }
}
