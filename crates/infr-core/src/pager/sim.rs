//! Offline eviction-policy simulator: how many misses a trace costs under a policy other than the
//! one [`Pager`](super::Pager) runs, with no model and no device.
//!
//! [`simulate`] owns everything a policy must not change — residency, the slot count, the batch
//! epoch guard, pins, and the hit/miss/eviction counting — exactly as `Pager` defines them. A
//! [`Policy`] only orders the resident blocks and names a victim among those the driver says are
//! evictable. Today's pager is [`Recency`] here, which exists to prove the driver right: on every
//! trace it must reproduce the real pager's counts (the tests check it, and so does the replay
//! benchmark on recorded traces).
//!
//! [`Belady`] is the fewest misses this exact request sequence allows under the same batch guard
//! and pins — but only for traces made of `begin_batch`, the `touch` family, `resolve_and_pin`
//! and `unpin` ([`belady_is_a_bound`]). For those, which blocks are evictable at any point depends
//! only on the requests, never on earlier victim choices, so the classic exchange argument for
//! MIN carries over; that is an argument, not a written proof, and it is checked by brute force
//! against every victim choice on thousands of small random traces (the tests). A trace with
//! `pin_if_resident` probes or `repin`s is NOT covered: those pin a block only if it is resident,
//! so the pins — and with them which blocks may be evicted — depend on the policy, and Belady can
//! lose to another policy there (`belady_is_not_a_bound_once_probes_pin`). `evict` is excluded
//! too, since nothing has checked it. The bound is also only for THIS request sequence: a caller
//! whose requests depend on residency (the MoE decode path picks touch-all or routed touches from
//! `all_resident`) would make different requests under a different policy.
//!
//! A candidate policy (S3-FIFO, ARC, a TinyLFU admission filter, ...) is one more [`Policy`] impl.
//! Admission (declining to cache a block at all) is not expressible yet: the driver admits every
//! miss, as `Pager` must.

use super::trace::Call;
use super::{BlockId, Insert};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// An eviction policy, driven by [`simulate`]. `at` is the index of the current call in the
/// trace, so an offline policy can look ahead.
pub trait Policy {
    fn name(&self) -> String;
    /// Called once before the run with the whole call sequence. Online policies ignore it.
    fn prepare(&mut self, _calls: &[Call]) {}
    /// `call` (at `at`) found `id` resident.
    fn hit(&mut self, at: usize, id: BlockId, call: Call);
    /// `call` (at `at`) admitted `id`, after any eviction it needed.
    fn insert(&mut self, at: usize, id: BlockId, call: Call);
    /// `id` left the cache: as a victim, or through an explicit `evict`.
    fn remove(&mut self, id: BlockId);
    /// The block to evict among the resident ones `evictable` accepts, or `None` if it accepts
    /// none. The driver removes it (and calls [`Self::remove`]).
    fn victim(&mut self, at: usize, evictable: &dyn Fn(BlockId) -> bool) -> Option<BlockId>;
}

/// What a simulated run counted. `hits`, `misses` and `evictions` mean what
/// [`PagerStats`](super::PagerStats) means by them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SimStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Admissions refused because every resident block was protected (`resolve_and_pin`'s
    /// `None`; for the infallible entry points, the state the real pager panics on).
    pub refused: u64,
    /// `unpin`s of a block this run held no pin on. Zero when replaying the recording policy; a
    /// different policy can leave a probe (`pin_if_resident`) unpinned that the recording pinned.
    pub unbalanced_unpins: u64,
}

/// Whether [`Belady`]'s misses are a lower bound for `calls`: true when the trace holds only
/// `begin_batch`, `touch`, `touch_cold`, `schedule`, `resolve_and_pin` and `unpin` (see the module
/// doc for why probes, repins and evicts are out).
pub fn belady_is_a_bound(calls: &[Call]) -> bool {
    calls.iter().all(|c| {
        matches!(
            c,
            Call::BeginBatch
                | Call::Touch(_)
                | Call::TouchCold(_)
                | Call::Schedule(_)
                | Call::ResolveAndPin(..)
                | Call::Unpin(_)
        )
    })
}

/// Run `calls` against a cache of `n_slots` under `policy`.
pub fn simulate(n_slots: usize, calls: &[Call], policy: &mut dyn Policy) -> SimStats {
    let [whole, _] = simulate_split(n_slots, calls, policy, calls.len());
    whole
}

/// [`simulate`], with the counts of calls before `split` and from `split` on reported apart — one
/// run, so the policy's decisions are those of the whole trace (a prefill/decode boundary, say).
pub fn simulate_split(
    n_slots: usize,
    calls: &[Call],
    policy: &mut dyn Policy,
    split: usize,
) -> [SimStats; 2] {
    let mut phases = [SimStats::default(); 2];
    // Resident block -> the batch epoch of its last touch, as `Pager::epoch` keeps it.
    let mut resident: HashMap<BlockId, u64> = HashMap::with_capacity(n_slots);
    let mut pins: HashMap<BlockId, u32> = HashMap::new();
    let mut cur_epoch = 0u64;
    policy.prepare(calls);
    for (at, &call) in calls.iter().enumerate() {
        let s = &mut phases[usize::from(at >= split)];
        let (id, admits, counted_miss) = match call {
            Call::BeginBatch => {
                cur_epoch += 1;
                continue;
            }
            Call::Touch(id) | Call::TouchCold(id) | Call::Schedule(id) => (id, true, true),
            Call::ResolveAndPin(id, _) => (id, true, true),
            Call::PinIfResident(id) => (id, false, false),
            Call::Repin(id) => {
                // Uncounted, and no epoch, as `Pager::repin` — but still a use, which an offline
                // policy has to see to keep its look-ahead current.
                if resident.contains_key(&id) {
                    *pins.entry(id).or_insert(0) += 1;
                    policy.hit(at, id, call);
                }
                continue;
            }
            Call::Unpin(id) => {
                match pins.get_mut(&id) {
                    Some(n) if *n > 1 => *n -= 1,
                    Some(_) => {
                        pins.remove(&id);
                    }
                    None => s.unbalanced_unpins += 1,
                }
                continue;
            }
            Call::Evict(id) => {
                if resident.remove(&id).is_some() {
                    pins.remove(&id);
                    policy.remove(id);
                }
                continue;
            }
        };
        let pins_on_success = matches!(call, Call::ResolveAndPin(..) | Call::PinIfResident(_));
        if resident.contains_key(&id) {
            resident.insert(id, cur_epoch);
            s.hits += 1;
            policy.hit(at, id, call);
        } else if admits {
            if resident.len() >= n_slots {
                // `Pager::take_slot{,_opt}`: the batch guard applies once batches are in use.
                let batches = cur_epoch != 0;
                let evictable = |b: BlockId| {
                    !(batches && resident.get(&b) == Some(&cur_epoch)) && !pins.contains_key(&b)
                };
                let Some(v) = policy.victim(at, &evictable) else {
                    s.refused += 1;
                    continue;
                };
                assert!(
                    evictable(v),
                    "{} chose block {v}, which is not evictable",
                    policy.name()
                );
                resident.remove(&v);
                policy.remove(v);
                s.evictions += 1;
            }
            resident.insert(id, cur_epoch);
            if counted_miss {
                s.misses += 1;
            }
            policy.insert(at, id, call);
        } else {
            continue;
        }
        if pins_on_success {
            *pins.entry(id).or_insert(0) += 1;
        }
    }
    phases
}

/// Today's [`Pager`](super::Pager) policy as a [`Policy`]: recency with the two insertion points.
/// `touch` (and an MRU `resolve_and_pin`) promotes on a hit and inserts at the hot end;
/// `touch_cold`, `schedule` and a cold `resolve_and_pin` leave a hit in place and insert at the
/// cold end; a `pin_if_resident` hit does not promote.
#[derive(Debug, Default)]
pub struct Recency {
    order: BTreeMap<i64, BlockId>,
    stamp: HashMap<BlockId, i64>,
    hot: i64,
    cold: i64,
}

impl Recency {
    fn place(&mut self, id: BlockId, stamp: i64) {
        if let Some(old) = self.stamp.insert(id, stamp) {
            self.order.remove(&old);
        }
        self.order.insert(stamp, id);
    }

    fn mru(&mut self, id: BlockId) {
        self.hot += 1;
        self.place(id, self.hot);
    }
}

impl Policy for Recency {
    fn name(&self) -> String {
        "current (recency)".into()
    }

    fn hit(&mut self, _at: usize, id: BlockId, call: Call) {
        if matches!(call, Call::Touch(_) | Call::ResolveAndPin(_, Insert::Mru)) {
            self.mru(id);
        }
    }

    fn insert(&mut self, _at: usize, id: BlockId, call: Call) {
        match call {
            Call::Touch(_) | Call::ResolveAndPin(_, Insert::Mru) => self.mru(id),
            _ => {
                self.cold -= 1;
                self.place(id, self.cold);
            }
        }
    }

    fn remove(&mut self, id: BlockId) {
        if let Some(s) = self.stamp.remove(&id) {
            self.order.remove(&s);
        }
    }

    fn victim(&mut self, _at: usize, evictable: &dyn Fn(BlockId) -> bool) -> Option<BlockId> {
        self.order.values().copied().find(|&b| evictable(b))
    }
}

/// Belady's MIN: evict the evictable block whose next use is furthest in the future (or that is
/// never used again). Run through [`simulate`], it honours the same batch guard and pins as the
/// real pager.
///
/// **Its misses are a lower bound only where [`belady_is_a_bound`] holds** (no probes, repins or
/// evicts; see the module doc): there the bound holds by an exchange argument and was checked by
/// brute force on small random traces, not proven. On a trace with probes it is a heuristic, and
/// callers must not report it as the optimum. Either way it bounds only the recorded request
/// sequence, not what a caller would have requested under another policy.
///
/// A use is any call that names the block and could hit it (`touch`, `touch_cold`, `schedule`,
/// `resolve_and_pin`, and the out-of-scope `pin_if_resident` and `repin`).
#[derive(Debug, Default)]
pub struct Belady {
    /// For each call index, the index of the next call using the same block (`usize::MAX`: never).
    next_use: Vec<usize>,
    /// Resident blocks by (next use, id).
    by_next: BTreeSet<(usize, BlockId)>,
    key: HashMap<BlockId, usize>,
}

impl Belady {
    fn uses(call: Call) -> Option<BlockId> {
        match call {
            Call::Touch(id)
            | Call::TouchCold(id)
            | Call::Schedule(id)
            | Call::ResolveAndPin(id, _)
            | Call::PinIfResident(id)
            | Call::Repin(id) => Some(id),
            Call::BeginBatch | Call::Unpin(_) | Call::Evict(_) => None,
        }
    }

    fn seen(&mut self, at: usize, id: BlockId) {
        let next = self.next_use[at];
        if let Some(old) = self.key.insert(id, next) {
            self.by_next.remove(&(old, id));
        }
        self.by_next.insert((next, id));
    }
}

impl Policy for Belady {
    fn name(&self) -> String {
        "Belady".into()
    }

    fn prepare(&mut self, calls: &[Call]) {
        self.next_use = vec![usize::MAX; calls.len()];
        let mut later: HashMap<BlockId, usize> = HashMap::new();
        for (i, &c) in calls.iter().enumerate().rev() {
            if let Some(id) = Self::uses(c) {
                if let Some(&n) = later.get(&id) {
                    self.next_use[i] = n;
                }
                later.insert(id, i);
            }
        }
    }

    fn hit(&mut self, at: usize, id: BlockId, _call: Call) {
        self.seen(at, id);
    }

    fn insert(&mut self, at: usize, id: BlockId, _call: Call) {
        self.seen(at, id);
    }

    fn remove(&mut self, id: BlockId) {
        if let Some(k) = self.key.remove(&id) {
            self.by_next.remove(&(k, id));
        }
    }

    fn victim(&mut self, _at: usize, evictable: &dyn Fn(BlockId) -> bool) -> Option<BlockId> {
        self.by_next
            .iter()
            .rev()
            .map(|&(_, b)| b)
            .find(|&b| evictable(b))
    }
}

#[cfg(test)]
mod tests {
    use super::super::shapes::{cyclic_sweep, Moe};
    use super::super::trace::{replay, PoolTrace};
    use super::*;

    fn touches(ids: &[BlockId]) -> Vec<Call> {
        ids.iter().map(|&id| Call::Touch(id)).collect()
    }

    fn misses(n_slots: usize, calls: &[Call], policy: &mut dyn Policy) -> u64 {
        simulate(n_slots, calls, policy).misses
    }

    /// The textbook reference string (Silberschatz et al., "Operating System Concepts", the
    /// page-replacement chapter): with three frames, OPT faults 9 times and LRU 12.
    #[test]
    fn belady_and_lru_match_the_textbook_reference_string() {
        let refs = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1];
        let calls = touches(&refs);
        assert_eq!(misses(3, &calls, &mut Belady::default()), 9);
        assert_eq!(misses(3, &calls, &mut Recency::default()), 12);
    }

    /// The case LRU is pessimal on: a cyclic sweep over one block more than the cache holds.
    /// With 2 slots over 0,1,2 x3, LRU misses all 9; MIN misses at positions 0, 1, 2, 4, 6, 8
    /// (worked by hand: each miss after warm-up evicts the block the sweep reaches last).
    #[test]
    fn belady_beats_lru_on_a_sweep_one_larger_than_the_cache() {
        let calls = touches(&[0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!(misses(2, &calls, &mut Recency::default()), 9);
        assert_eq!(misses(2, &calls, &mut Belady::default()), 6);
    }

    /// The batch guard binds Belady too: block 0, never used again, is the unconstrained victim
    /// at `t2`, but the current batch touched it, so 1 must go and then miss.
    #[test]
    fn belady_honours_the_batch_guard() {
        let calls = [
            Call::BeginBatch,
            Call::Touch(1),
            Call::BeginBatch,
            Call::Touch(0),
            Call::Touch(2),
            Call::BeginBatch,
            Call::Touch(1),
        ];
        assert_eq!(misses(2, &calls, &mut Belady::default()), 4);
        // Without batches the same accesses give MIN its free choice.
        let free: Vec<Call> = calls
            .iter()
            .copied()
            .filter(|c| *c != Call::BeginBatch)
            .collect();
        assert_eq!(misses(2, &free, &mut Belady::default()), 3);
    }

    /// Pins bind Belady too: pinned block 1 is never used again, but 2 must be the victim.
    #[test]
    fn belady_honours_pins() {
        let r = |id| Call::ResolveAndPin(id, Insert::Mru);
        let calls = [r(1), r(2), Call::Unpin(2), r(3), Call::Unpin(3), r(2)];
        let s = simulate(2, &calls, &mut Belady::default());
        assert_eq!((s.misses, s.refused), (4, 0));
        let unpinned = [
            r(1),
            Call::Unpin(1),
            r(2),
            Call::Unpin(2),
            r(3),
            Call::Unpin(3),
            r(2),
        ];
        assert_eq!(misses(2, &unpinned, &mut Belady::default()), 3);
        // Every resident block pinned: refused, as `resolve_and_pin` returns `None`.
        let s = simulate(1, &[r(1), r(2)], &mut Belady::default());
        assert_eq!((s.misses, s.refused), (1, 1));
    }

    /// The reviewer's counterexample: with `pin_if_resident` probes in the trace, Belady is not the
    /// optimum. A probe pins only a resident block, so what is pinned later depends on what MIN
    /// kept, and the brute force finds a schedule with one miss fewer. Documents the limitation;
    /// [`belady_is_a_bound`] is what keeps such a trace from being reported as a bound.
    #[test]
    fn belady_is_not_a_bound_once_probes_pin() {
        let r = |id| Call::ResolveAndPin(id, Insert::Mru);
        let (b, u, q) = (Call::BeginBatch, Call::Unpin, Call::PinIfResident);
        let calls = [
            r(1),
            r(4),
            r(0),
            u(1),
            u(4),
            b,
            b,
            r(3),
            b,
            u(3),
            u(0),
            q(1),
            r(4),
            b,
            u(1),
            q(3),
        ];
        assert!(!belady_is_a_bound(&calls));
        assert_eq!(simulate(3, &calls, &mut Belady::default()).misses, 5);
        assert_eq!(brute_force_optimum(3, &calls), Some(4));
    }

    /// The fewest misses any sequence of victim choices achieves on `calls`, by trying them all,
    /// under exactly [`simulate`]'s rules. Branches that refuse an admission are dropped (a
    /// refusal is not a miss, so counting it as one or as nothing would both mislead); `None` if
    /// every branch refuses. Memoised on the state that decides the future: the call index, and
    /// for each resident block whether the current batch touched it and its pin count.
    fn brute_force_optimum(n_slots: usize, calls: &[Call]) -> Option<u64> {
        type Key = (usize, Vec<(BlockId, bool, u32)>);
        #[derive(Clone)]
        struct St {
            resident: HashMap<BlockId, u64>,
            pins: HashMap<BlockId, u32>,
            epoch: u64,
        }
        fn go(
            at: usize,
            mut st: St,
            n_slots: usize,
            calls: &[Call],
            memo: &mut HashMap<Key, Option<u64>>,
        ) -> Option<u64> {
            let mut key: Vec<_> = st
                .resident
                .iter()
                .map(|(&b, &e)| (b, e == st.epoch, st.pins.get(&b).copied().unwrap_or(0)))
                .collect();
            key.sort_unstable();
            let key = (at, key);
            if let Some(&v) = memo.get(&key) {
                return v;
            }
            let Some(&call) = calls.get(at) else {
                return Some(0);
            };
            let pin = |st: &mut St, id| *st.pins.entry(id).or_insert(0) += 1;
            let out = match call {
                Call::BeginBatch => {
                    st.epoch += 1;
                    go(at + 1, st, n_slots, calls, memo)
                }
                Call::Unpin(id) => {
                    match st.pins.get(&id).copied() {
                        Some(n) if n > 1 => {
                            st.pins.insert(id, n - 1);
                        }
                        Some(_) => {
                            st.pins.remove(&id);
                        }
                        None => {}
                    }
                    go(at + 1, st, n_slots, calls, memo)
                }
                Call::Evict(id) => {
                    if st.resident.remove(&id).is_some() {
                        st.pins.remove(&id);
                    }
                    go(at + 1, st, n_slots, calls, memo)
                }
                Call::Repin(id) => {
                    if st.resident.contains_key(&id) {
                        pin(&mut st, id);
                    }
                    go(at + 1, st, n_slots, calls, memo)
                }
                Call::PinIfResident(id) => {
                    if st.resident.contains_key(&id) {
                        st.resident.insert(id, st.epoch);
                        pin(&mut st, id);
                    }
                    go(at + 1, st, n_slots, calls, memo)
                }
                Call::Touch(id)
                | Call::TouchCold(id)
                | Call::Schedule(id)
                | Call::ResolveAndPin(id, _) => {
                    let pins = matches!(call, Call::ResolveAndPin(..));
                    if st.resident.contains_key(&id) {
                        st.resident.insert(id, st.epoch);
                        if pins {
                            pin(&mut st, id);
                        }
                        go(at + 1, st, n_slots, calls, memo)
                    } else {
                        let admit = |mut st: St, memo: &mut HashMap<Key, Option<u64>>| {
                            st.resident.insert(id, st.epoch);
                            if pins {
                                pin(&mut st, id);
                            }
                            go(at + 1, st, n_slots, calls, memo).map(|m| m + 1)
                        };
                        if st.resident.len() < n_slots {
                            admit(st, memo)
                        } else {
                            let batches = st.epoch != 0;
                            let victims: Vec<BlockId> = st
                                .resident
                                .iter()
                                .filter(|&(b, &e)| {
                                    !(batches && e == st.epoch) && !st.pins.contains_key(b)
                                })
                                .map(|(&b, _)| b)
                                .collect();
                            victims
                                .into_iter()
                                .filter_map(|v| {
                                    let mut next = st.clone();
                                    next.resident.remove(&v);
                                    admit(next, memo)
                                })
                                .min()
                        }
                    }
                }
            };
            memo.insert(key, out);
            out
        }
        let st = St {
            resident: HashMap::new(),
            pins: HashMap::new(),
            epoch: 0,
        };
        go(0, st, n_slots, calls, &mut HashMap::new())
    }

    /// A random trace of the kinds [`belady_is_a_bound`] accepts: touches of every flavour,
    /// batches, and `resolve_and_pin`s each released by a later `unpin`. It stays inside the
    /// pager's own contract — a batch's distinct blocks plus the held pins never reach `n_slots`
    /// (a new batch opens first) — so every admission has a victim under any policy.
    fn random_trace(
        rng: &mut super::super::shapes::XorShift64,
        n_slots: usize,
        n_ids: usize,
        len: usize,
    ) -> Vec<Call> {
        let mut calls = Vec::with_capacity(len);
        let mut held: Vec<BlockId> = Vec::new();
        let mut batch: Vec<BlockId> = Vec::new();
        while calls.len() < len {
            let id = rng.below(n_ids) as BlockId;
            let kind = rng.below(8);
            if kind == 0 {
                calls.push(Call::BeginBatch);
                batch.clear();
                continue;
            }
            if (4..=5).contains(&kind) && !held.is_empty() {
                calls.push(Call::Unpin(held.swap_remove(rng.below(held.len()))));
                continue;
            }
            let pins = kind == 3;
            let grows = usize::from(!batch.contains(&id)) + usize::from(pins);
            if batch.len() + held.len() + grows >= n_slots {
                calls.push(Call::BeginBatch);
                batch.clear();
                if held.len() + 1 + usize::from(pins) >= n_slots {
                    continue; // too many pins held for this one; let an unpin come first
                }
            }
            if !batch.contains(&id) {
                batch.push(id);
            }
            calls.push(match kind {
                1 => Call::TouchCold(id),
                2 => Call::Schedule(id),
                3 => {
                    held.push(id);
                    Call::ResolveAndPin(id, Insert::Mru)
                }
                _ => Call::Touch(id),
            });
        }
        calls
    }

    /// The property the reports rest on: on traces of the supported kinds, Belady's misses equal
    /// the true optimum found by trying every victim choice. Seeded; a trace where Belady refuses
    /// an admission is skipped (and counted, so the test cannot pass by skipping everything).
    #[test]
    fn belady_is_the_optimum_on_small_random_traces() {
        let mut rng = super::super::shapes::XorShift64::new(0x5eed);
        let (mut checked, mut skipped) = (0, 0);
        for i in 0..3000 {
            let n_slots = 2 + i % 3;
            let calls = random_trace(&mut rng, n_slots, n_slots + 3, 14);
            assert!(belady_is_a_bound(&calls));
            let opt = simulate(n_slots, &calls, &mut Belady::default());
            if opt.refused != 0 {
                skipped += 1;
                continue;
            }
            let best = brute_force_optimum(n_slots, &calls);
            assert_eq!(
                Some(opt.misses),
                best,
                "{n_slots} slots: Belady {} misses, optimum {best:?} on {calls:?}",
                opt.misses
            );
            checked += 1;
        }
        assert!(
            checked > 2000,
            "only {checked} traces checked ({skipped} skipped)"
        );
    }

    /// The phase split is the whole run cut in two, not two runs.
    #[test]
    fn a_split_run_adds_up_to_the_whole() {
        let t = Moe {
            n_layers: 4,
            n_expert: 16,
            top_k: 4,
            zipf_s: 1.0,
        }
        .session(24, 20, 9);
        let whole = simulate(t.n_slots, &t.calls, &mut Belady::default());
        let [a, b] = simulate_split(
            t.n_slots,
            &t.calls,
            &mut Belady::default(),
            t.calls.len() / 3,
        );
        assert_eq!(a.misses + b.misses, whole.misses);
        assert_eq!(a.hits + b.hits, whole.hits);
        assert!(a.misses > 0 && b.misses > 0);
    }

    /// The generated shapes at reduced sizes (the full ones run in the replay benchmark).
    fn small_shapes() -> Vec<PoolTrace> {
        let m = Moe {
            n_layers: 6,
            n_expert: 16,
            top_k: 4,
            zipf_s: 1.0,
        };
        let mut out = vec![cyclic_sweep(10, 4, 5), cyclic_sweep(40, 8, 4)];
        for slots in [16, 40, 72] {
            out.push(m.prefill(slots, 3));
            out.push(m.decode_routed(slots, 40, 3));
            out.push(m.session(slots, 40, 5));
        }
        out.push(m.decode_resident(4));
        out
    }

    /// The driver is what Belady's numbers rest on, so it must be right: driving today's policy
    /// through it reproduces the real pager's counts exactly, on every shape.
    #[test]
    fn the_driver_reproduces_the_real_pager() {
        for t in small_shapes() {
            let real = replay(&t).expect("generated traces have no outcomes to diverge from");
            let sim = simulate(t.n_slots, &t.calls, &mut Recency::default());
            assert_eq!(
                (sim.hits, sim.misses, sim.evictions, sim.refused),
                (real.hits, real.misses, real.evictions, 0),
                "{}",
                t.label
            );
        }
    }

    /// MIN is a lower bound: never more misses than today's policy, on any shape.
    #[test]
    fn belady_never_misses_more_than_the_current_policy() {
        for t in small_shapes() {
            let cur = simulate(t.n_slots, &t.calls, &mut Recency::default());
            let opt = simulate(t.n_slots, &t.calls, &mut Belady::default());
            assert!(
                opt.misses <= cur.misses,
                "{}: Belady {} > current {}",
                t.label,
                opt.misses,
                cur.misses
            );
            assert_eq!(opt.refused, 0, "{}", t.label);
        }
    }

    /// `schedule` on a cyclic sweep keeps a fixed resident prefix, which is NOT what MIN does:
    /// MIN rotates which blocks it keeps and wins a little more (on 0,1,2 through 2 slots: 6
    /// misses to the fixed prefix's 7). The pager's own
    /// `schedule_cyclic_sweep_keeps_a_stable_prefix` pins the prefix's count. Pinned here: a gap
    /// exists, and it stays under one miss per pass (in steady state it tends to
    /// `(C - 1) / (N - 1)` per pass; short runs carry an end effect of a few misses).
    #[test]
    fn schedule_on_a_sweep_is_within_one_miss_per_pass_of_belady() {
        for (n_blocks, n_slots, passes) in [(3, 2, 3), (10, 4, 5), (40, 8, 6)] {
            let t = cyclic_sweep(n_blocks, n_slots, passes);
            let cur = simulate(t.n_slots, &t.calls, &mut Recency::default()).misses;
            let opt = simulate(t.n_slots, &t.calls, &mut Belady::default()).misses;
            assert!(opt < cur, "{}: Belady {opt}, schedule {cur}", t.label);
            assert!(
                cur - opt <= passes as u64,
                "{}: Belady {opt}, schedule {cur}",
                t.label
            );
        }
        let t = cyclic_sweep(3, 2, 3);
        assert_eq!(
            simulate(t.n_slots, &t.calls, &mut Belady::default()).misses,
            6
        );
        assert_eq!(
            simulate(t.n_slots, &t.calls, &mut Recency::default()).misses,
            7
        );
    }
}
