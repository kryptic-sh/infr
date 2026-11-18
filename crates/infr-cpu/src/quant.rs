//! Activation quantization: Q8 (256-superblock) and Q8x32 (native 32-block) int8 formats,
//! plus small SIMD horizontal-reduction helpers shared by the kernels.

/// Activation quantized to Q8 over 256-element super-blocks: `qs[i] = round(x[i]/d[blk])` (int8),
/// `d[blk] = max|x|/127`. Quantize the activation ONCE per matvec, then integer-dot it against the
/// quantized weight rows (llama.cpp's q8_K path) — no per-row f32 weight expansion.
///
/// A block whose largest one or two magnitudes stand far above the rest ([`HI_KEEP`],
/// [`HI_RATIO`]) is split in two: those elements move to a sparse part ([`Self::hi`]), and the
/// dense codes take the scale of what is left. Without the split such a block's scale is set by
/// its outliers and every other element rounds to 0 or ±1. The first token of a sequence carries
/// such rows into every projection (its massive-activation dimensions, whose keys and values feed
/// every later token); the inputs of the FFN down projections carry them all through a sequence,
/// so there the split works as a general two-scale quantizer (`docs/cpu-activations.md`). A
/// block with nothing to split is quantized as it always was, to the same bits.
#[derive(Clone)]
pub(crate) struct Q8 {
    pub(crate) qs: Vec<i8>,
    pub(crate) d: Vec<f32>,
    /// Sub-block sums: `bsums[b*8+s]` = Σ `qs[b*256 + s*32 .. +32]` as i32.
    /// One entry per 32-element sub-block (8 per 256-element super-block).
    /// Precomputed once at quantization time; reused across all weight rows so the
    /// `sm` accumulation in `vec_dot_q4k` (Σ m·Σq8) avoids O(rows·256) re-summation.
    /// Mirrors llama.cpp's `block_q8_K.bsums`.
    pub(crate) bsums: Vec<i32>,
    /// Per-16-element sub-block sums (`bsums16[b*16+s]` = Σ `qs[b*256 + s*16 .. +16]`) — Q6_K's
    /// `-32` offset correction is per 16-element scale group. Precomputed here for the same
    /// reason as `bsums`: `vec_dot_q6k_batch` used to re-derive these sums with SIMD for EVERY
    /// weight row (262k redundant recomputes per lm_head GEMM — ~22% of a DG denoise step).
    /// Only the x86 SIMD kernel reads it — dead code on aarch64 (macOS CI builds with -D warnings).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) bsums16: Vec<i32>,
    /// The sparse part, in block order: `(b, part)` adds `part` — a one-block `Q8` whose codes are
    /// zero outside the split-off elements — to block `b`. A block can have up to [`HI_LEVELS`]
    /// parts, each holding what the ones before it left of those elements. The activation this
    /// `Q8` stands for is the dense codes plus every part; a dot over it is the dense dot plus each
    /// part's one-block dot against the weight's block `b` ([`crate::kernels::vec_dot_hi`]).
    pub(crate) hi: Vec<(usize, Q8)>,
}

/// The most elements of a 256-element block that can move to the sparse part ([`Q8::hi`]).
pub(crate) const HI_KEEP: usize = 2;

/// An element moves to the sparse part only when its magnitude exceeds this many times the
/// largest magnitude outside the block's top [`HI_KEEP`]. A power of two, so the test multiplies
/// exactly. Measured in `docs/cpu-activations.md`: at this ratio the first token's rows split in
/// every projection, and FFN-down inputs split all through a sequence (a few to a few tens of
/// percent of their rows, by model), which is where the split's prefill cost is (each split
/// block is a second one-block dot per weight row); half of it costs far more, and four times it
/// misses ERNIE-4.5's first token.
pub(crate) const HI_RATIO: f32 = 16.0;

/// The most sparse parts one block can take. Each part takes its largest element to within f32
/// rounding and leaves at most half its scale, `1/254` of its largest, of every other one; parts
/// continue while a remainder still exceeds the rest of the block and the f32 resolution of the
/// block maximum. Two parts settle [`HI_KEEP`] elements; the third is for rounding.
pub(crate) const HI_LEVELS: usize = 3;

/// The largest magnitude in `blk`; NaN is passed over.
#[inline]
fn block_max(blk: &[f32]) -> f32 {
    blk.iter().fold(0f32, |m, &v| m.max(v.abs()))
}

/// Quantize `blk` to codes in `q` on the scale `amax/127` (`amax` its [`block_max`]), returning
/// the scale.
#[inline]
fn quantize_block(blk: &[f32], amax: f32, q: &mut [i8]) -> f32 {
    let dd = amax / 127.0;
    let id = if dd > 0.0 { 1.0 / dd } else { 0.0 };
    for (o, &v) in q.iter_mut().zip(blk) {
        *o = (v * id).round().clamp(-127.0, 127.0) as i8;
    }
    dd
}

/// The elements of a 256-element block that move to the sparse part, in magnitude order (lowest
/// index first on a tie): of the block's [`HI_KEEP`] largest magnitudes, those above [`HI_RATIO`]
/// times the next largest. NaN counts as magnitude zero, as it does for the block maximum `amax`.
fn split_off(blk: &[f32], amax: f32) -> Vec<usize> {
    let mag = |v: f32| if v.is_nan() { 0.0 } else { v.abs() };
    // When HI_KEEP + 1 elements reach amax / HI_RATIO, the (HI_KEEP+1)-th largest does, and
    // nothing can stand HI_RATIO above it: the common case, decided in one pass. (A NaN fails
    // the comparison and is not counted, so the count can only be short, which sends a block to
    // the exact rule below, never past it.)
    let near = blk.iter().filter(|&&v| v.abs() * HI_RATIO >= amax).count();
    if near > HI_KEEP {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..blk.len()).collect();
    order.sort_by(|&a, &b| mag(blk[b]).total_cmp(&mag(blk[a])).then(a.cmp(&b)));
    let next = mag(blk[order[HI_KEEP]]);
    order.truncate(HI_KEEP);
    order.retain(|&i| mag(blk[i]) > HI_RATIO * next);
    order
}

/// A `Q8` over the codes `qs` and per-block scales `d`, with its sub-block sums and no sparse part.
fn from_codes(qs: Vec<i8>, d: Vec<f32>) -> Q8 {
    // Per-16- and per-32-elem sub-block sums (Q6_K offset term / Q4_K min-scale term).
    let bsums16: Vec<i32> = qs
        .chunks(16)
        .map(|c| c.iter().map(|&q| q as i32).sum())
        .collect();
    let bsums = bsums16
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| p[0] + p[1])
        .collect();
    Q8 {
        qs,
        d,
        bsums,
        bsums16,
        hi: Vec::new(),
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quantize_q8(x: &[f32]) -> Q8 {
    let nb = x.len() / 256;
    let mut qs = vec![0i8; nb * 256];
    let mut d = vec![0f32; nb];
    let mut hi = Vec::new();
    for b in 0..nb {
        let blk = &x[b * 256..b * 256 + 256];
        let q = &mut qs[b * 256..b * 256 + 256];
        let amax = block_max(blk);
        let kept = split_off(blk, amax);
        if kept.is_empty() {
            d[b] = quantize_block(blk, amax, q);
            continue;
        }
        // The split-off elements, quantized on their own scale level by level; what the last
        // level leaves of them stays in the dense block with the rest.
        let mut dense = blk.to_vec();
        let rest = (0..256)
            .filter(|i| !kept.contains(i))
            .fold(0f32, |m, i| m.max(blk[i].abs()));
        for _ in 0..HI_LEVELS {
            let a = kept.iter().fold(0f32, |m, &i| m.max(dense[i].abs()));
            if !(a > rest && a > amax * f32::EPSILON && a.is_finite()) {
                break;
            }
            let dh = a / 127.0;
            let id = 1.0 / dh;
            let mut hq = vec![0i8; 256];
            for &i in &kept {
                hq[i] = (dense[i] * id).round().clamp(-127.0, 127.0) as i8;
                dense[i] -= hq[i] as f32 * dh;
            }
            hi.push((b, from_codes(hq, vec![dh])));
        }
        d[b] = quantize_block(&dense, block_max(&dense), q);
    }
    Q8 {
        hi,
        ..from_codes(qs, d)
    }
}

/// The activation a `Q8` stands for: each element's dense code times its block scale, plus its
/// codes in every sparse part times theirs, summed in [`Q8::hi`]'s order.
pub(crate) fn dequantize_q8(q8: &Q8) -> Vec<f32> {
    let mut x: Vec<f32> = q8
        .qs
        .iter()
        .enumerate()
        .map(|(i, &q)| q as f32 * q8.d[i / 256])
        .collect();
    for (b, part) in &q8.hi {
        for (i, &q) in part.qs.iter().enumerate() {
            x[b * 256 + i] += q as f32 * part.d[0];
        }
    }
    x
}

/// Horizontal reduction: sum 8 × i32 in a ymm register to a scalar i32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn hadd_i32_ymm(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let h1 = _mm256_hadd_epi32(v, v);
    let h2 = _mm256_hadd_epi32(h1, h1);
    let lo = _mm256_castsi256_si128(h2);
    let hi = _mm256_extracti128_si256::<1>(h2);
    _mm_cvtsi128_si32(_mm_add_epi32(lo, hi))
}

/// Horizontal reduction: sum 4 × i32 in an xmm register to a scalar i32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn hadd_i32_xmm(v: std::arch::x86_64::__m128i) -> i32 {
    use std::arch::x86_64::*;
    let h = _mm_hadd_epi32(v, v); // [a+b, c+d, a+b, c+d]
    let hh = _mm_hadd_epi32(h, h); // [a+b+c+d, ...]
    _mm_cvtsi128_si32(hh)
}

/// Activation quantized to int8 per NATIVE 32-element block (mirrors [`Q8`] but without the
/// 256-superblock grouping). `bsum` is `Σqs` per block — Q5_0's constant `-16` offset needs `Σx`,
/// which `d[b] * bsum[b]` approximates the same way `Q8::bsums` does for the K-quant min term.
#[derive(Clone)]
pub(crate) struct Q8x32 {
    pub(crate) qs: Vec<i8>,
    pub(crate) d: Vec<f32>,
    pub(crate) bsum: Vec<i32>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quantize_q8_32(x: &[f32]) -> Q8x32 {
    let nb = x.len() / 32;
    let mut qs = vec![0i8; nb * 32];
    let mut d = vec![0f32; nb];
    let mut bsum = vec![0i32; nb];
    for b in 0..nb {
        let blk = &x[b * 32..b * 32 + 32];
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let dd = amax / 127.0;
        let id = if dd > 0.0 { 1.0 / dd } else { 0.0 };
        d[b] = dd;
        let mut s = 0i32;
        for (i, &v) in blk.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            qs[b * 32 + i] = q;
            s += q as i32;
        }
        bsum[b] = s;
    }
    Q8x32 { qs, d, bsum }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The quantizer before the split, kept here for the microbenchmark below to time against.
    fn quantize_q8_unsplit(x: &[f32]) -> Q8 {
        let nb = x.len() / 256;
        let mut qs = vec![0i8; nb * 256];
        let mut d = vec![0f32; nb];
        for b in 0..nb {
            let blk = &x[b * 256..b * 256 + 256];
            d[b] = quantize_block(blk, block_max(blk), &mut qs[b * 256..b * 256 + 256]);
        }
        from_codes(qs, d)
    }

    /// An ordinary row: no block splits, and the quantizer gives the codes it gave before the
    /// split existed.
    #[test]
    fn an_ordinary_row_quantizes_as_before() {
        let x: Vec<f32> = (0..2048)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let (new, old) = (quantize_q8(&x), quantize_q8_unsplit(&x));
        assert!(new.hi.is_empty());
        assert_eq!((new.qs, new.d, new.bsums16), (old.qs, old.d, old.bsums16));
    }

    /// Microbenchmark: ns per activation row for the quantizer before and after the split on
    /// ordinary rows, and for a Q4_K one-row dot over 2048 inputs against an ordinary row and
    /// against a first-token row (two blocks split). Prints; asserts only that it timed.
    ///
    /// `cargo test --release -p infr-cpu --lib quant::tests::microbench -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement: run in release with --nocapture"]
    fn microbench() {
        use crate::kernels::{vec_dot_hi, vec_dot_q8_any};
        use infr_core::tensor::DType;
        let time = |name: &str, reps: usize, f: &mut dyn FnMut()| {
            for _ in 0..reps / 10 {
                f();
            }
            let t = std::time::Instant::now();
            for _ in 0..reps {
                f();
            }
            let ns = t.elapsed().as_nanos() as f64 / reps as f64;
            println!("{name:<44} {ns:>10.1} ns");
            assert!(ns > 0.0);
        };
        for n in [2048usize, 11008] {
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
                .collect();
            time(&format!("quantize unsplit, {n} wide"), 20_000, &mut || {
                std::hint::black_box(quantize_q8_unsplit(std::hint::black_box(&x)));
            });
            time(
                &format!("quantize split, ordinary row, {n} wide"),
                20_000,
                &mut || {
                    std::hint::black_box(quantize_q8(std::hint::black_box(&x)));
                },
            );
        }
        let n = 2048;
        let ordinary: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let mut first = ordinary.clone();
        first[13] = 900.0;
        first[300] = -420.0;
        let w = crate::test_util::det_quant_rows(64, n / 256, 144, &[0, 2], 9);
        let bpr = w.len() / 64;
        for (name, x) in [("ordinary", &ordinary), ("first-token", &first)] {
            let q8 = quantize_q8(x);
            time(
                &format!("quantize split, {name} row, {n} wide"),
                20_000,
                &mut || {
                    std::hint::black_box(quantize_q8(std::hint::black_box(x)));
                },
            );
            time(
                &format!("Q4_K dot x64 rows, {name} ({} parts)", q8.hi.len()),
                20_000,
                &mut || {
                    for o in 0..64 {
                        let row = &w[o * bpr..(o + 1) * bpr];
                        let dense = vec_dot_q8_any::<false>(DType::Q4K, row, &q8, n);
                        let y = if q8.hi.is_empty() {
                            dense
                        } else {
                            dense + vec_dot_hi::<false>(DType::Q4K, row, &q8, n)
                        };
                        std::hint::black_box(y);
                    }
                },
            );
        }
    }
}
