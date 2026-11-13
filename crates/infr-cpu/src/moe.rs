//! MoE expert-bank GEMM dispatch: activation-representation selection (`ActsKind` /
//! `expert_acts_kind`) and the per-o-range GEMM (`expert_gemm_range`) that fans out to the
//! quant-specific batch kernels.
use crate::dequant_row::dequant_dot_rows;
use crate::f32_row;
use crate::kernels::{
    dot, dot_bf16, dot_f16, dot_n, vec_dot_q32_batch8, vec_dot_q4k_batch, vec_dot_q4k_batch8,
    vec_dot_q5_0_32_batch, vec_dot_q5k_batch, vec_dot_q6k_batch, vec_dot_q8_0_32_batch,
    vec_dot_q8_0_batch,
};
use crate::quant::{Q8x32, Q8};
use crate::repack::{packed_min_tokens, Q4kPack};
#[cfg(target_arch = "x86_64")]
use crate::repack::{
    q4k_gemm_group_out, q4k_pack_and_gemm_group, q6k_pack_and_gemm_group, GemmOut,
};
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
    /// Row-major f32 `[count, in_f]`: f16/bf16/f32 weights, the dequant fallback, and ALL
    /// single-row (`int8_ok == false`) calls, which stay byte-for-byte exact (Metal parity).
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
pub(crate) fn expert_gemm_range(
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
                    DType::Q4K => vec_dot_q4k_batch(row, q8s, in_f, col),
                    DType::Q6K => vec_dot_q6k_batch(row, q8s, in_f, col),
                    DType::Q8_0 => vec_dot_q8_0_batch(row, q8s, in_f, col),
                    DType::Q5K => vec_dot_q5k_batch(row, q8s, in_f, col),
                    _ => unreachable!("Super acts imply a 256-super-block quant dtype"),
                }
            };
            // 8-row tiles: Q4_K always (the cached ilv pack when the caller has one, else the
            // per-call pack or `vec_dot_q4k_batch8`), Q5_K and Q6_K on the packed VNNI GEMM.
            // Q8_0 runs per-row.
            #[cfg(target_arch = "x86_64")]
            let vnni =
                is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni");
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
                vec_dot_q4k_batch8(rows, q8s, in_f, outs);
                o += 8;
            }
            while o < o1 {
                dot_col(o, &mut out_t[(o - o0) * count..(o - o0 + 1) * count]);
                o += 1;
            }
        }
        ExpertActs::Blk32(q8s) => {
            let q5 = dt == DType::Q5_0;
            let dot_col = |o: usize, col: &mut [f32]| {
                let row = &wbytes[o * bpr..o * bpr + bpr];
                match dt {
                    DType::Q8_0 => vec_dot_q8_0_32_batch(row, q8s, in_f, col),
                    DType::Q5_0 => vec_dot_q5_0_32_batch(row, q8s, in_f, col),
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
                vec_dot_q32_batch8(rows, q8s, in_f, outs, q5);
                o += 8;
            }
            while o < o1 {
                dot_col(o, &mut out_t[(o - o0) * count..(o - o0 + 1) * count]);
                o += 1;
            }
        }
        ExpertActs::Raw(xin) => {
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
                            *dst = dot_n(&w32, xs);
                        }
                        let done = count / 4 * 4;
                        for (k, dst) in tail.iter_mut().enumerate() {
                            *dst = dot(&w32, x_row(done + k));
                        }
                    }
                    DType::F16 => {
                        for (r, dst) in col.iter_mut().enumerate() {
                            *dst = dot_f16(row, &xin[r * in_f..r * in_f + in_f]);
                        }
                    }
                    DType::Bf16 => {
                        for (r, dst) in col.iter_mut().enumerate() {
                            *dst = dot_bf16(row, &xin[r * in_f..r * in_f + in_f]);
                        }
                    }
                    _ => dequant_dot_rows(row, dt, xin, in_f, col, &mut accs),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{vec_dot_q4k, vec_dot_q5k, vec_dot_q6k};
    use crate::quant::quantize_q8;
    use crate::repack::TILE_TOKENS;
    use crate::test_util::det_quant_rows;

    /// An expert bank's K-quant GEMM gives the single-token kernel's bits for every bucket size,
    /// on both sides of each dtype's packed threshold (`packed_min_tokens`) — a bucket of one
    /// routed token is common — and for an o-range that starts past the first 8-row tile and
    /// ends in a partial one.
    #[test]
    fn expert_gemm_range_matches_the_single_token_kernels_bitwise() {
        let (in_f, out_f) = (512usize, 28usize);
        let nb = in_f / 256;
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
                        let x: Vec<f32> = (0..in_f)
                            .map(|i| (((i + 31 * r) * 29 % 97) as f32 - 48.0) / 13.0)
                            .collect();
                        quantize_q8(&x)
                    })
                    .collect();
                let (o0, o1) = (8, out_f);
                let mut out_t = vec![f32::NAN; (o1 - o0) * count];
                expert_gemm_range(
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
                        let want = match dt {
                            DType::Q4K => vec_dot_q4k(row, q8, in_f),
                            DType::Q5K => vec_dot_q5k(row, q8, in_f),
                            _ => vec_dot_q6k(row, q8, in_f),
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
    }
}
