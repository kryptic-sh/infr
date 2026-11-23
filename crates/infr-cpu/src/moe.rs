//! MoE expert-bank GEMM dispatch: activation-representation selection (`ActsKind` /
//! `expert_acts_kind`) and the per-o-range GEMM (`expert_gemm_range`) that fans out to the
//! quant-specific batch kernels.
use crate::dequant_row::dequant_dot_rows;
use crate::f32_row;
use crate::kernels::{
    vec_dot_q32_batch8, vec_dot_q4k_batch, vec_dot_q4k_batch8, vec_dot_q5_0_32_batch,
    vec_dot_q5k_batch, vec_dot_q6k_batch, vec_dot_q8_0_32_batch, vec_dot_q8_0_batch, Dots, HiBatch,
};
use crate::pool::SpinPool;
use crate::quant::{quantize_q8, quantize_q8_32};
use crate::quant::{Q8x32, Q8};
use crate::repack::{packed_min_tokens, Q4kPack};
#[cfg(target_arch = "x86_64")]
use crate::repack::{
    q4k_gemm_group_out, q4k_pack_and_gemm_group, q6k_pack_and_gemm_group, GemmOut,
};
use infr_core::graph::MoeGating;
use infr_core::math;
use infr_core::tensor::DType;

/// Activation batch for [`expert_gemm_range`] — the representation the weight dtype dictates
/// (see [`expert_acts_kind`]). The staged `Op::MoeFfn` pipeline builds these ONCE per stage
/// (each distinct hidden row quantized a single time, then cloned per routed pair) and every
/// o-range task borrows a bucket's slice.
pub(crate) enum ExpertActs<'a> {
    /// 256-super-block int8 (`quantize_q8`): Q4K/Q6K/Q8_0/Q5K weights with `in_f % 256 == 0`.
    Super(&'a [Q8]),
    /// 32-block int8 (`quantize_q8_32`): Q8_0/Q5_0 weights at misaligned `in_f % 32 == 0`
    /// (e.g. DiffusionGemma's down `in_f = 704`).
    Blk32(&'a [Q8x32]),
    /// Row-major f32 `[count, in_f]`: f16/bf16/f32 weights, the dequant fallback, and every bank
    /// in reference mode (`int8_ok == false`).
    Raw(&'a [f32]),
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum ActsKind {
    Super,
    Blk32,
    Raw,
}

/// Which activation representation an expert bank of this `(dtype, in_f)` uses — the SAME
/// dispatch order the old `expert_matvec_batch` fast paths had, so every (weights, activations)
/// pairing lands on the identical kernel.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn expert_acts_kind(dt: DType, in_f: usize, int8_ok: bool) -> ActsKind {
    if int8_ok
        && matches!(dt, DType::Q4K | DType::Q6K | DType::Q8_0 | DType::Q5K)
        && in_f.is_multiple_of(256)
    {
        ActsKind::Super
    } else if int8_ok && matches!(dt, DType::Q8_0 | DType::Q5_0) && in_f.is_multiple_of(32) {
        ActsKind::Blk32
    } else {
        ActsKind::Raw
    }
}

/// The router's logits for every row of `rxs` (`[rows, ne]`), row-major `[rows, n_expert]`: ONE
/// batched matvec over every row (the router is tiny, `[n_expert, ne]`), on the pool. Serial it
/// was 12-20ms/layer at 512 rows (134M MAC), long enough that spinning pool workers SMT-throttled
/// it, which is where qwen3moe's phase-3 prefill regression hid (per-op MoeFfn time had IMPROVED).
/// `Op::MoeFfn` and `Op::MoeRoute` both route on these, so the two cannot select differently.
#[allow(clippy::too_many_arguments)]
pub(crate) fn router_logits<const CAP: bool>(
    pool: &SpinPool,
    rbytes: &[u8],
    rdt: DType,
    ne: usize,
    n_expert: usize,
    rxs: &[f32],
    rows: usize,
    int8_ok: bool,
) -> Vec<f32> {
    let r_kind = expert_acts_kind(rdt, ne, int8_ok);
    let rq8: Vec<Q8> = if r_kind == ActsKind::Super {
        pool.collect(rows, &|r| quantize_q8(&rxs[r * ne..r * ne + ne]))
    } else {
        Vec::new()
    };
    let rq832: Vec<Q8x32> = if r_kind == ActsKind::Blk32 {
        pool.collect(rows, &|r| quantize_q8_32(&rxs[r * ne..r * ne + ne]))
    } else {
        Vec::new()
    };
    let racts = match r_kind {
        ActsKind::Super => ExpertActs::Super(&rq8),
        ActsKind::Blk32 => ExpertActs::Blk32(&rq832),
        ActsKind::Raw => ExpertActs::Raw(rxs),
    };
    let mut lt = vec![0f32; n_expert * rows]; // o-major
    pool.for_chunks_mut(&mut lt, 8 * rows, 1, &|c, dst| {
        let o0 = c * 8;
        let o1 = (o0 + 8).min(n_expert);
        expert_gemm_range::<CAP>(rbytes, rdt, ne, n_expert, &racts, None, o0, o1, dst);
    });
    let mut logits = vec![0f32; rows * n_expert];
    pool.for_chunks_mut(&mut logits, n_expert, 4, &|r, lrow| {
        for (o, dst) in lrow.iter_mut().enumerate() {
            *dst = lt[o * rows + r];
        }
    });
    logits
}

/// One row's router probabilities: softmax over all experts (qwen3moe/…) or a per-expert
/// sigmoid (llama4). Both are monotone in the logit, so the top-`n_used` selection is identical
/// either way (llama4 selects by raw logits — same set).
pub(crate) fn router_probs(logits: &[f32], gating: MoeGating) -> Vec<f32> {
    match gating {
        MoeGating::Softmax => {
            let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut p = logits.to_vec();
            math::exp_sub_slice(&mut p, maxl);
            let psum: f32 = p.iter().sum();
            for v in p.iter_mut() {
                *v /= psum;
            }
            p
        }
        MoeGating::Sigmoid => {
            let mut p = logits.to_vec();
            math::sigmoid_slice(&mut p);
            p
        }
        MoeGating::SqrtSoftplus => {
            // sqrt(softplus(x)) = sqrt(ln(1 + exp(x))). Monotone like sigmoid;
            // top-k-by-logit picks the same set.
            logits
                .iter()
                .map(|&v| {
                    let sp = if v > 20.0 {
                        v
                    } else {
                        math::ln(1.0_f32 + math::exp(v))
                    };
                    sp.sqrt()
                })
                .collect()
        }
    }
}

/// The score each expert of one row is ranked by: its probability, plus the router bias, with
/// every expert of a group that lost set to negative infinity.
pub(crate) fn selection_scores(
    probs: &[f32],
    bias: Option<&[f32]>,
    n_expert_groups: u32,
    n_expert_groups_used: u32,
) -> Vec<f32> {
    let n_expert = probs.len();
    // DeepSeek V2+: router bias for SELECTION only — add to a copy used for
    // top-k selection; original probs still used for per-expert weights.
    let mut sel_probs: Vec<f32> = if let Some(epb_w) = bias {
        probs
            .iter()
            .enumerate()
            .map(|(e, &p)| p + epb_w[e])
            .collect()
    } else {
        probs.to_vec()
    };
    // DeepSeek V3+: group-limited routing — per-group top-2, top groups,
    // mask the rest to -inf (matching llama.cpp `build_moe_ffn`).
    if n_expert_groups > 1 && n_expert_groups_used > 0 {
        let n_exp_per_group = n_expert / n_expert_groups as usize;
        let ng = n_expert_groups as usize;
        let ng_used = n_expert_groups_used as usize;
        // Compute per-group score: sum of top-2 within each group.
        let mut gscores = vec![f32::NEG_INFINITY; ng];
        for g in 0..ng {
            let base = g * n_exp_per_group;
            let mut best = [f32::NEG_INFINITY; 2];
            for &s in &sel_probs[base..base + n_exp_per_group] {
                if s > best[0] {
                    best[1] = best[0];
                    best[0] = s;
                } else if s > best[1] {
                    best[1] = s;
                }
            }
            gscores[g] = best[0] + best[1];
        }
        // Top ng_used groups by score.
        let mut gidx: Vec<usize> = (0..ng).collect();
        // `total_cmp`: a NaN score must not panic the backend. Without
        // a NaN it orders as `partial_cmp` did except -0.0 / +0.0; a NaN
        // score ranks first and the routing output is garbage.
        gidx.sort_by(|&a, &b| gscores[b].total_cmp(&gscores[a]));
        gidx.truncate(ng_used);
        // Mask non-chosen groups to -inf.
        let mut chosen = vec![false; ng];
        for &g in &gidx {
            chosen[g] = true;
        }
        for g in 0..ng {
            if !chosen[g] {
                for s in sel_probs[g * n_exp_per_group..(g + 1) * n_exp_per_group].iter_mut() {
                    *s = f32::NEG_INFINITY;
                }
            }
        }
    }
    sel_probs
}

/// The `n_used` experts with the highest [`selection_scores`], best first; the lower id first on
/// an equal score (the sort is stable).
pub(crate) fn top_experts(scores: &[f32], n_used: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx.truncate(n_used);
    idx
}

thread_local! {
    /// Each pool thread's buffer for one GEMM task's o-major block — see [`with_block_scratch`].
    static BLOCK_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` on this thread's scratch buffer resized to `n` floats, whose contents are
/// unspecified (`f` must write what it reads): a task that computes a block only to scatter it
/// elsewhere needs no allocation of its own. Debug builds fill it with NaN first, so a read of
/// an element `f` never wrote shows up in the results.
pub(crate) fn with_block_scratch<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    BLOCK_SCRATCH.with_borrow_mut(|v| {
        v.resize(n, 0.0);
        if cfg!(debug_assertions) {
            v.fill(f32::NAN);
        }
        f(&mut v[..n])
    })
}

/// Compute output rows `[o0, o1)` of ONE expert bank into `out_t` — an o-major slice of exactly
/// `(o1-o0) * count` floats (`out_t[(o - o0) * count + r]`). `o0` MUST be 8-aligned: the
/// Q4_K / 32-block 8-row tiles are anchored at `o = 0`, so 8-aligned chunking reproduces the
/// exact tile boundaries (and therefore bit-identical per-element results) of a whole-bank call,
/// no matter how a task list splits the range. The kernels and their dispatch mirror the old
/// `expert_matvec_batch` fast paths 1/2 + fallback one-for-one.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn expert_gemm_range<const CAP: bool>(
    wbytes: &[u8],
    dt: DType,
    in_f: usize,
    out_f: usize,
    acts: &ExpertActs,
    q4k_pack: Option<&Q4kPack>,
    o0: usize,
    o1: usize,
    out_t: &mut [f32],
) {
    debug_assert!(o0.is_multiple_of(8) && o0 <= o1 && o1 <= out_f);
    let bpr = wbytes.len() / out_f;
    let count = if o1 > o0 { out_t.len() / (o1 - o0) } else { 0 };
    match acts {
        ExpertActs::Super(q8s) => {
            let dot_col = |o: usize, col: &mut [f32]| {
                let row = &wbytes[o * bpr..o * bpr + bpr];
                match dt {
                    DType::Q4K => vec_dot_q4k_batch::<CAP>(row, q8s, in_f, col),
                    DType::Q6K => vec_dot_q6k_batch::<CAP>(row, q8s, in_f, col),
                    DType::Q8_0 => vec_dot_q8_0_batch::<CAP>(row, q8s, in_f, col),
                    DType::Q5K => vec_dot_q5k_batch::<CAP>(row, q8s, in_f, col),
                    _ => unreachable!("Super acts imply a 256-super-block quant dtype"),
                }
            };
            // 8-row tiles: Q4_K always (the cached ilv pack when the caller has one, else the
            // per-call pack or `vec_dot_q4k_batch8`), Q5_K and Q6_K on the packed VNNI GEMM.
            // Q8_0 runs per-row.
            #[cfg(target_arch = "x86_64")]
            let vnni = crate::simd::vnni::<CAP>();
            #[cfg(not(target_arch = "x86_64"))]
            let vnni = false;
            let tiled_end = match dt {
                DType::Q4K => out_f / 8 * 8,
                DType::Q5K | DType::Q6K if vnni && count >= packed_min_tokens(dt) => out_f / 8 * 8,
                _ => 0,
            };
            let mut o = o0;
            while o + 8 <= o1.min(tiled_end) {
                let cols = &mut out_t[(o - o0) * count..(o - o0 + 8) * count];
                let rows: [&[u8]; 8] =
                    std::array::from_fn(|i| &wbytes[(o + i) * bpr..(o + i) * bpr + bpr]);
                #[cfg(target_arch = "x86_64")]
                if vnni {
                    let gout = GemmOut::o_major(cols, count);
                    // SAFETY: VNNI checked above; `cols` is this tile's own block.
                    unsafe {
                        match (dt, q4k_pack) {
                            (DType::Q4K, Some(pack)) => {
                                q4k_gemm_group_out(&pack.groups[o / 8], pack.nb, q8s, gout, 0)
                            }
                            (DType::Q6K, _) => {
                                q6k_pack_and_gemm_group(rows, in_f / 256, q8s, gout, 0)
                            }
                            _ => q4k_pack_and_gemm_group(dt, rows, in_f / 256, q8s, gout, 0),
                        }
                    }
                    o += 8;
                    continue;
                }
                let mut it = cols.chunks_mut(count);
                let outs: [&mut [f32]; 8] = std::array::from_fn(|_| it.next().unwrap());
                vec_dot_q4k_batch8::<CAP>(rows, q8s, in_f, outs);
                o += 8;
            }
            while o < o1 {
                dot_col(o, &mut out_t[(o - o0) * count..(o - o0 + 1) * count]);
                o += 1;
            }
            // The sparse activation parts, as every `Q8` route adds them (`vec_dot_hi`'s bits).
            if let Some(hb) = HiBatch::new(dt, in_f, q8s) {
                hb.add_o_major(wbytes, bpr, o0, o1 - o0, out_t, count);
            }
        }
        ExpertActs::Blk32(q8s) => {
            let q5 = dt == DType::Q5_0;
            let dot_col = |o: usize, col: &mut [f32]| {
                let row = &wbytes[o * bpr..o * bpr + bpr];
                match dt {
                    DType::Q8_0 => vec_dot_q8_0_32_batch::<CAP>(row, q8s, in_f, col),
                    DType::Q5_0 => vec_dot_q5_0_32_batch::<CAP>(row, q8s, in_f, col),
                    _ => unreachable!("Blk32 acts imply Q8_0/Q5_0"),
                }
            };
            // Both Q8_0 and Q5_0 ride the interleaved 8-row tile (old fast path 2).
            let tiled_end = out_f / 8 * 8;
            let mut o = o0;
            while o + 8 <= o1.min(tiled_end) {
                let cols = &mut out_t[(o - o0) * count..(o - o0 + 8) * count];
                let rows: [&[u8]; 8] =
                    std::array::from_fn(|i| &wbytes[(o + i) * bpr..(o + i) * bpr + bpr]);
                let mut it = cols.chunks_mut(count);
                let outs: [&mut [f32]; 8] = std::array::from_fn(|_| it.next().unwrap());
                vec_dot_q32_batch8::<CAP>(rows, q8s, in_f, outs, q5);
                o += 8;
            }
            while o < o1 {
                dot_col(o, &mut out_t[(o - o0) * count..(o - o0 + 1) * count]);
                o += 1;
            }
        }
        ExpertActs::Raw(xin) => {
            let dots = Dots::resolve::<CAP>();
            // f32 dots against raw activation rows: identical math to the old fallback (the
            // weight row is dequantized ONCE, reused across all rows), o-major placement. The
            // quant arm is `dequant_dot_rows`: the same bits without materializing the row.
            let mut accs = Vec::new();
            for o in o0..o1 {
                let row = &wbytes[o * bpr..o * bpr + bpr];
                let col = &mut out_t[(o - o0) * count..(o - o0 + 1) * count];
                match dt {
                    DType::F32 => {
                        // Four rows per `dot_n`: the same per-row result as `dot`, with the
                        // rows' FMA chains overlapped (an f32 router is a whole-batch GEMM).
                        let w32 = f32_row(row);
                        let x_row = |r: usize| &xin[r * in_f..r * in_f + in_f];
                        let (quads, tail) = col.as_chunks_mut::<4>();
                        for (q, dst) in quads.iter_mut().enumerate() {
                            let xs: [&[f32]; 4] = std::array::from_fn(|k| x_row(4 * q + k));
                            *dst = dots.dot_n(&w32, xs);
                        }
                        let done = count / 4 * 4;
                        for (k, dst) in tail.iter_mut().enumerate() {
                            *dst = dots.dot(&w32, x_row(done + k));
                        }
                    }
                    DType::F16 => {
                        for (r, dst) in col.iter_mut().enumerate() {
                            *dst = dots.dot_f16(row, &xin[r * in_f..r * in_f + in_f]);
                        }
                    }
                    DType::Bf16 => {
                        for (r, dst) in col.iter_mut().enumerate() {
                            *dst = dots.dot_bf16(row, &xin[r * in_f..r * in_f + in_f]);
                        }
                    }
                    _ => dequant_dot_rows(dots, row, dt, xin, in_f, col, &mut accs),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{vec_dot_hi, vec_dot_q4k, vec_dot_q5k, vec_dot_q6k};
    use crate::quant::quantize_q8;
    use crate::repack::TILE_TOKENS;
    use crate::test_util::det_quant_rows;

    /// An expert bank's GEMM on `Q8` activations (Q4_K, Q5_K, Q6_K, Q8_0) gives, on every
    /// element, the one-row dense kernel plus the sparse part dotted the way it was before the
    /// fast sparse dot (`vec_dot_hi_reference`), at bucket sizes 1, 2, 5 and 19 (both sides of
    /// the packed thresholds) and an o-range that starts past the first 8-row tile and ends in a
    /// partial one, on rows of every spike pattern (`test_util::spike_patterns`).
    #[test]
    fn expert_gemm_range_adds_the_reference_sparse_dot_bitwise() {
        use crate::kernels::{vec_dot_hi_reference, vec_dot_q8_any};
        use crate::test_util::{spiky_row, synth_weights};
        let (in_f, out_f) = (768usize, 28usize);
        let mut split = 0usize;
        for dt in [DType::Q4K, DType::Q5K, DType::Q6K, DType::Q8_0] {
            let w = synth_weights(dt, out_f, in_f, 0x3c, &[(1.0, 1.0), (-0.5, 2.0)]);
            let bpr = w.len() / out_f;
            for count in [1usize, 2, 5, 19] {
                let q8s: Vec<Q8> = (0..count)
                    .map(|r| quantize_q8(&spiky_row(r, in_f)))
                    .collect();
                let (o0, o1) = (8, out_f);
                let mut out_t = vec![f32::NAN; (o1 - o0) * count];
                expert_gemm_range::<false>(
                    &w,
                    dt,
                    in_f,
                    out_f,
                    &ExpertActs::Super(&q8s),
                    None,
                    o0,
                    o1,
                    &mut out_t,
                );
                for o in o0..o1 {
                    let row = &w[o * bpr..(o + 1) * bpr];
                    for (r, q8) in q8s.iter().enumerate() {
                        let dense = vec_dot_q8_any::<false>(dt, row, q8, in_f);
                        let want = if q8.hi.is_empty() {
                            dense
                        } else {
                            split += 1;
                            dense + vec_dot_hi_reference::<false>(dt, row, q8, in_f)
                        };
                        let got = out_t[(o - o0) * count + r];
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "{dt:?} count={count} o={o} r={r}: got {got}, want {want}"
                        );
                    }
                }
            }
        }
        assert!(split > 0, "no row split");
    }

    /// An expert bank's K-quant GEMM gives the single-token kernel's bits for every bucket size,
    /// on both sides of each dtype's packed threshold (`packed_min_tokens`) — a bucket of one
    /// routed token is common — and for an o-range that starts past the first 8-row tile and
    /// ends in a partial one. Every third row splits (`Q8::hi`): its value is the dense kernel's
    /// plus `vec_dot_hi`.
    #[test]
    fn expert_gemm_range_matches_the_single_token_kernels_bitwise() {
        let (in_f, out_f) = (512usize, 28usize);
        let nb = in_f / 256;
        let mut split = 0usize;
        for (dt, blk, d_offs) in [
            (DType::Q4K, 144usize, &[0usize, 2][..]),
            (DType::Q5K, 176, &[0, 2]),
            (DType::Q6K, 210, &[208]),
        ] {
            let w = det_quant_rows(out_f, nb, blk, d_offs, 7 + blk as u64);
            let bpr = w.len() / out_f;
            let min = packed_min_tokens(dt);
            let mut counts = vec![1, min, 2 * TILE_TOKENS + 1];
            if min > 1 {
                counts.push(min - 1);
            }
            for count in counts {
                let q8s: Vec<Q8> = (0..count)
                    .map(|r| {
                        // Every third row carries first-token spikes, so it splits.
                        let x: Vec<f32> = (0..in_f)
                            .map(|i| match (r % 3, i) {
                                (1, 5) => 900.0,
                                (1, 300) => -350.0,
                                _ => (((i + 31 * r) * 29 % 97) as f32 - 48.0) / 13.0,
                            })
                            .collect();
                        quantize_q8(&x)
                    })
                    .collect();
                let (o0, o1) = (8, out_f);
                let mut out_t = vec![f32::NAN; (o1 - o0) * count];
                expert_gemm_range::<false>(
                    &w,
                    dt,
                    in_f,
                    out_f,
                    &ExpertActs::Super(&q8s),
                    None,
                    o0,
                    o1,
                    &mut out_t,
                );
                for o in o0..o1 {
                    let row = &w[o * bpr..(o + 1) * bpr];
                    for (r, q8) in q8s.iter().enumerate() {
                        let dense = match dt {
                            DType::Q4K => vec_dot_q4k::<false>(row, q8, in_f),
                            DType::Q5K => vec_dot_q5k::<false>(row, q8, in_f),
                            _ => vec_dot_q6k::<false>(row, q8, in_f),
                        };
                        let want = if q8.hi.is_empty() {
                            dense
                        } else {
                            split += 1;
                            dense + vec_dot_hi(dt, row, q8, in_f)
                        };
                        let got = out_t[(o - o0) * count + r];
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "{dt:?} count={count} o={o} r={r}: got {got}, want {want}"
                        );
                    }
                }
            }
        }
        assert!(split > 0, "no row split");
    }
}
