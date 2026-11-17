//! `Op::LightningIndexer`'s two halves on the CPU: scoring every eligible key of every query row,
//! split across the pool by KEYS so that a single decode row uses every core, and keeping each
//! row's best `top_k` by a partial selection.
//!
//! Both reproduce what the op computed before they existed bit for bit: each score is the same
//! sequence of floating-point operations (the head sum in ascending head order over `dot`'s
//! value), and the selection keeps the same keys in the same order, ties to the lower index.

use crate::kernels::{dot, dot_mn, dot_n};
use crate::pool::SpinPool;

/// The indexer's key rows, one `head_dim`-wide row per key, read in place when they are f16 and
/// widened per task: the cache is read once, by the task that scores those keys.
#[derive(Clone, Copy)]
pub(crate) enum Keys<'a> {
    F16(&'a [half::f16]),
    F32(&'a [f32]),
}

/// Keys a scoring task scores together through [`dot_n`], so the core has that many independent
/// FMA chains in flight.
const KEYS: usize = 4;

/// Query heads a scoring task dots against the same keys together through [`dot_mn`].
const HEADS: usize = 4;

/// Fewest multiply-adds one scoring task does. A task that does less costs more to hand to a
/// parked worker than to run on the caller: on the synthetic V4 file, whose indexer is a few
/// heads a few dozen wide, splitting by a fixed key count made the op several times slower.
const SCORE_TASK_MACS_MIN: usize = 1 << 17;

/// Scoring tasks per pool thread, so that a slow core does not leave the others idle at the end.
const SCORE_TASKS_PER_THREAD: usize = 4;

/// Keys one scoring task takes when `rows` query rows of `n_head` heads `head_dim` wide score `n`
/// keys on `threads` threads: a share of the keys small enough to balance, never under
/// [`SCORE_TASK_MACS_MIN`], a whole number of [`KEYS`] groups.
fn task_keys(threads: usize, n: usize, rows: usize, n_head: usize, head_dim: usize) -> usize {
    n.div_ceil(threads * SCORE_TASKS_PER_THREAD)
        .max(SCORE_TASK_MACS_MIN.div_ceil((rows * n_head * head_dim).max(1)))
        .next_multiple_of(KEYS)
}

/// `score[j, t] = Σ_h (w[t, h] * scale) * relu(q[t, h] · k[j])` for every query row `t` and every
/// key `j < his[t]` (the row's eligible prefix), the head sum in ascending `h`, as
/// `ggml_sum_rows` sums the contiguous head axis.
///
/// Returned KEY-major, `[max(his), rows]`, so that each task owns a contiguous run of keys for
/// every row; an entry at or past its row's `his[t]` is never computed and reads 0. The result is
/// built in `buf`'s storage, so a caller that hands back the previous op's score buffer (the
/// executor's `Recycler`) does not map a fresh `rows × keys` allocation per op.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score<const CAP: bool>(
    pool: &SpinPool,
    buf: Vec<f32>,
    q: &[f32],
    ws: &[f32],
    keys: Keys<'_>,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    his: &[usize],
) -> Vec<f32> {
    let (rows, nh, hd) = (his.len(), n_head, head_dim);
    let n = his.iter().copied().max().unwrap_or(0);
    let mut sc = buf;
    sc.clear();
    sc.resize(n * rows, 0.0);
    let per = task_keys(pool.threads(), n, rows, nh, hd);
    pool.for_chunks_mut(&mut sc, per * rows, 1, &|c, piece| {
        let (j0, nk) = (c * per, piece.len() / rows);
        // Keys are widened one group at a time, so the group and every query head stay in L1
        // while they are dotted; a whole task's keys widened up front pushed the queries out.
        let mut wide = vec![0f32; KEYS * hd];
        for g0 in (0..nk).step_by(KEYS) {
            let gn = KEYS.min(nk - g0);
            let kr: &[f32] = match keys {
                Keys::F32(k) => &k[(j0 + g0) * hd..(j0 + g0 + gn) * hd],
                Keys::F16(k) => {
                    use half::slice::HalfFloatSliceExt;
                    k[(j0 + g0) * hd..(j0 + g0 + gn) * hd]
                        .convert_to_f32_slice(&mut wide[..gn * hd]);
                    &wide[..gn * hd]
                }
            };
            let key = |i: usize| &kr[i * hd..(i + 1) * hd];
            for (ti, &hi) in his.iter().enumerate() {
                let m = hi.saturating_sub(j0 + g0).min(gn);
                let qt = &q[ti * nh * hd..(ti + 1) * nh * hd];
                let qh = |h: usize| &qt[h * hd..(h + 1) * hd];
                let w = &ws[ti * nh..(ti + 1) * nh];
                let out = |i: usize| (g0 + i) * rows + ti;
                if m == KEYS {
                    let ks: [&[f32]; KEYS] = std::array::from_fn(key);
                    let mut acc = [0f32; KEYS];
                    // Heads in blocks that share each key load, then any left over one at a
                    // time; either way every key's sum takes the heads in ascending order.
                    let blocked = nh - nh % HEADS;
                    for h0 in (0..blocked).step_by(HEADS) {
                        let d: [[f32; KEYS]; HEADS] =
                            dot_mn::<CAP, HEADS, KEYS>(std::array::from_fn(|m| qh(h0 + m)), ks);
                        for (m, d) in d.iter().enumerate() {
                            for (a, &d) in acc.iter_mut().zip(d) {
                                *a += (w[h0 + m] * scale) * d.max(0.0);
                            }
                        }
                    }
                    for h in blocked..nh {
                        let d: [f32; KEYS] = dot_n::<CAP, KEYS>(qh(h), ks);
                        for (a, d) in acc.iter_mut().zip(d) {
                            *a += (w[h] * scale) * d.max(0.0);
                        }
                    }
                    for (i, a) in acc.into_iter().enumerate() {
                        piece[out(i)] = a;
                    }
                } else {
                    for i in 0..m {
                        let mut a = 0f32;
                        for h in 0..nh {
                            a += (w[h] * scale) * dot::<CAP>(qh(h), key(i)).max(0.0);
                        }
                        piece[out(i)] = a;
                    }
                }
            }
        }
    });
    sc
}

/// Keep the best `slots.len()` keys of one query row: its eligible keys `0..sc.len()` by score,
/// descending, a tie to the lower index (the same rule as `Op::Argmax`), then — when fewer are
/// eligible than slots — the ineligible keys `sc.len()..` in index order. Each slot gets its key
/// index as a u32 bit pattern in the f32, the convention `Op::Argmax` uses for a token id.
///
/// A partial selection, `O(n)` plus a sort of the kept keys: with no NaN score, the `k`-th best
/// score `t` splits the keys into those above it, all kept, and those equal to it, kept in index
/// order until `k` are — the lower index winning every tie, as the rank `(score descending, index
/// ascending)` orders them. That rank is total, so the result is the one [`select_top_k_scan`]
/// gives. A NaN has no rank in it, so a row with one is handed to that scan, which defines what
/// a NaN score does.
pub(crate) fn select_top_k(sc: &[f32], slots: &mut [f32]) {
    if sc.iter().any(|s| s.is_nan()) {
        return select_top_k_scan(sc, slots);
    }
    let hi = sc.len();
    let k = slots.len().min(hi);
    let desc = |a: &f32, b: &f32| b.partial_cmp(a).expect("NaN scores take the scan");
    let mut kept: Vec<(f32, u32)> = if k == 0 {
        Vec::new()
    } else if k == hi {
        sc.iter().copied().zip(0..).collect()
    } else {
        let mut v = sc.to_vec();
        let t = *v.select_nth_unstable_by(k - 1, desc).1;
        let above = sc.iter().filter(|&&s| s > t).count();
        let mut ties = k - above;
        let mut kept = Vec::with_capacity(k);
        for (j, &s) in (0..).zip(sc) {
            if s > t {
                kept.push((s, j));
            } else if s == t && ties > 0 {
                kept.push((s, j));
                ties -= 1;
            }
        }
        kept
    };
    kept.sort_unstable_by(|a, b| desc(&a.0, &b.0).then(a.1.cmp(&b.1)));
    let order = kept.iter().map(|&(_, j)| j).chain(hi as u32..);
    for (slot, j) in slots.iter_mut().zip(order) {
        *slot = f32::from_bits(j);
    }
}

/// [`select_top_k`] by repeated linear scan, `O(slots · keys)`: each slot takes the first key, in
/// index order, that no later untaken key beats strictly. Ties therefore go to the lower index,
/// and a NaN score that becomes the running best is never beaten (no comparison with it is true),
/// so it takes the slot.
pub(crate) fn select_top_k_scan(sc: &[f32], slots: &mut [f32]) {
    let hi = sc.len();
    let n = hi + slots.len();
    let mut taken = vec![false; n];
    for slot in slots.iter_mut() {
        let mut bi = usize::MAX;
        for j in 0..n {
            if taken[j] {
                continue;
            }
            let better = match (j < hi, bi < hi) {
                // Nothing chosen yet.
                _ if bi == usize::MAX => true,
                // An eligible key outranks every ineligible one.
                (true, false) => true,
                (false, true) => false,
                // Both eligible: strictly higher score wins (a tie keeps the lower index, which
                // `bi` already holds).
                (true, true) => sc[j] > sc[bi],
                // Both ineligible: index order, so `bi` keeps it.
                (false, false) => false,
            };
            if better {
                bi = j;
            }
        }
        taken[bi] = true;
        *slot = f32::from_bits(bi as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::config::CpuCfg;

    fn ids(slots: &[f32]) -> Vec<u32> {
        slots.iter().map(|s| s.to_bits()).collect()
    }

    /// splitmix64, for reproducible adversarial scores.
    fn mix(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Scores drawn from a few values so most keys tie: exact ties, both zeros, negatives and the
    /// infinities.
    fn tied_scores(n: usize, seed: u64) -> Vec<f32> {
        const POOL: [f32; 8] = [
            0.0,
            -0.0,
            1.5,
            -2.0,
            1.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.25,
        ];
        (0..n as u64)
            .map(|i| POOL[(mix(seed ^ (i << 20)) % POOL.len() as u64) as usize])
            .collect()
    }

    fn both(sc: &[f32], k: usize) -> (Vec<u32>, Vec<u32>) {
        let (mut fast, mut scan) = (vec![0f32; k], vec![0f32; k]);
        select_top_k(sc, &mut fast);
        select_top_k_scan(sc, &mut scan);
        (ids(&fast), ids(&scan))
    }

    /// The partial selection keeps exactly what the scan keeps, in the same order, over many
    /// seeds of adversarial rows: mostly-tied scores, all-equal scores, `top_k` 1, `top_k` equal
    /// to the eligible count, `top_k` past it (the ineligible fill), no eligible key at all, and
    /// rows of distinct random scores of both signs.
    #[test]
    fn select_top_k_matches_the_scan() {
        let mut cases = 0usize;
        for seed in 0..400u64 {
            let n = (mix(seed) % 97) as usize;
            let sc = if seed % 3 == 0 {
                (0..n as u64)
                    .map(|i| (mix(seed << 32 | i) as i64 as f32) * 1e-18)
                    .collect()
            } else if seed % 7 == 0 {
                vec![0.5; n]
            } else {
                tied_scores(n, seed)
            };
            for k in [1, 2, n / 3, n.saturating_sub(1), n, n + 3] {
                if k == 0 {
                    continue;
                }
                let (fast, scan) = both(&sc, k);
                assert_eq!(fast, scan, "seed {seed}, {n} keys, top_k {k}: {sc:?}");
                cases += 1;
            }
        }
        assert!(cases > 1000, "only {cases} cases ran");
    }

    /// Ties go to the lower index: of three equal best scores, `top_k` 2 keeps the first two.
    #[test]
    fn select_top_k_breaks_ties_to_the_lower_index() {
        let sc = [1.0, 3.0, 2.0, 3.0, 3.0, 0.0];
        let mut s = [0f32; 4];
        select_top_k(&sc, &mut s);
        assert_eq!(ids(&s), [1, 3, 4, 2]);
        // -0.0 and 0.0 are one score.
        let sc = [-1.0, 0.0, -0.0, 0.0];
        let mut s = [0f32; 3];
        select_top_k(&sc, &mut s);
        assert_eq!(ids(&s), [1, 2, 3]);
    }

    /// A row with a NaN score selects exactly as the scan does, wherever the NaN sits.
    #[test]
    fn select_top_k_nan_rows_take_the_scan() {
        let rows: [&[f32]; 4] = [
            &[f32::NAN, 5.0, 3.0],
            &[1.0, f32::NAN, 3.0],
            &[5.0, 1.0, f32::NAN, 9.0, 2.0],
            &[f32::NAN, f32::NAN],
        ];
        for sc in rows {
            for k in 1..=sc.len() + 2 {
                let (fast, scan) = both(sc, k);
                assert_eq!(fast, scan, "{sc:?} top_k {k}");
            }
        }
        // The scan's rule, pinned: a NaN that is the running best takes the slot.
        let mut s = [0f32; 3];
        select_top_k(&[1.0, f32::NAN, 3.0], &mut s);
        assert_eq!(ids(&s), [2, 0, 1]);
    }

    /// The scoring definition, one key and one row at a time, in the op's own order.
    fn score_ref(q: &[f32], ws: &[f32], k: &[f32], nh: usize, hd: usize, scale: f32) -> f32 {
        let mut a = 0f32;
        for h in 0..nh {
            a += (ws[h] * scale) * crate::kernels::dot_scalar(&q[h * hd..(h + 1) * hd], k).max(0.0);
        }
        a
    }

    /// Key-split scoring equals the per-key definition bit for bit, for every row's eligible
    /// prefix, whether the keys run as one task or many, with f16 and f32 keys, at key counts that
    /// leave a partial group of [`KEYS`] and a partial last task, and head counts that leave heads
    /// over after the blocks of [`HEADS`].
    #[test]
    fn score_is_the_definition_bit_for_bit() {
        let pool = SpinPool::new(&CpuCfg::default());
        let scale = 0.3f32;
        let gen = |n: usize, s: u64| -> Vec<f32> {
            (0..n as u64)
                .map(|i| (mix(s << 32 | i) >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
                .collect()
        };
        // (rows, keys, heads, head width, whether the keys split over several tasks)
        let cases = [
            (1usize, 3usize, 5usize, 24usize, false),
            (3, 77, 5, 24, false),
            (2, 16, 4, 8, false),
            (1, 1001, 9, 128, true),
            (2, 603, 9, 128, true),
        ];
        for (rows, n_keys, nh, hd, split) in cases {
            assert_eq!(
                n_keys > task_keys(pool.threads(), n_keys, rows, nh, hd),
                split,
                "rows {rows}, keys {n_keys}: the case does not split as it says"
            );
            let his: Vec<usize> = (0..rows).map(|t| n_keys - 2 * (rows - 1 - t)).collect();
            let q = gen(rows * nh * hd, 1);
            let ws = gen(rows * nh, 2);
            let k16: Vec<half::f16> = gen(n_keys * hd, 3)
                .into_iter()
                .map(half::f16::from_f32)
                .collect();
            let k32: Vec<f32> = k16.iter().map(|x| x.to_f32()).collect();
            for keys in [Keys::F16(&k16), Keys::F32(&k32)] {
                let sc = score::<false>(&pool, Vec::new(), &q, &ws, keys, nh, hd, scale, &his);
                let n = *his.iter().max().unwrap();
                assert_eq!(sc.len(), n * rows);
                for (t, &hi) in his.iter().enumerate() {
                    for j in 0..hi {
                        let want = score_ref(
                            &q[t * nh * hd..],
                            &ws[t * nh..],
                            &k32[j * hd..(j + 1) * hd],
                            nh,
                            hd,
                            scale,
                        );
                        assert_eq!(
                            sc[j * rows + t].to_bits(),
                            want.to_bits(),
                            "rows {rows}, keys {n_keys}: row {t} key {j}"
                        );
                    }
                }
            }
        }
    }
}
