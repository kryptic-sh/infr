//! The NEON kernels of the aarch64 `dotprod` tier ([`crate::simd::dotprod`]): Q4_K, Q6_K and
//! Q8_0, one-row and batch.
//!
//! Each computes its family's scalar kernel's value bit for bit, by one shape: the weight codes
//! become signed bytes (Q4_K nibbles as `0..=15`, Q6_K codes as `q - 32`, Q8_0 bytes as they
//! are), `sdot` (`vdotq_s32`: signed bytes, four products per i32 lane) gives each sub-block's
//! integer dot, and the integer scale arithmetic and the float epilogue are the scalar kernel's
//! own lines, run on those integers in the scalar kernel's order. Integer sums are exact in any
//! order as long as nothing wraps; each helper states its bound. Q6_K dots `q - 32` instead of
//! `q`, which gives `Σ q·x − 32·Σ x`, the integer its scalar kernel forms from two sums,
//! directly.
//!
//! What is not vectorized: the per-block float epilogue (a handful of scalar operations per 256
//! elements, the scalar kernel's own) and the 6-bit scale unpacking (`k4`). The sparse part of a
//! split activation (`Q8::hi`) is not these kernels' business: `vec_dot_hi` adds it outside them
//! at every tier.
//!
//! The batch kernels decode the weight row once and dot it against [`TILE`] activation rows at a
//! time, so each weight load serves that many `sdot`s. The 2- and 8-row Q4_K tiles have no NEON
//! form of their own: below their x86 tiers they run the batch kernel row by row.

use super::*;
use std::arch::aarch64::*;

/// Activation rows a batch kernel dots against each weight load.
const TILE: usize = 4;

/// The 16 bytes of `s` at `off`, every one of them bounds-checked.
#[inline]
#[target_feature(enable = "neon")]
fn ld_s8(s: &[i8], off: usize) -> int8x16_t {
    let c: &[i8; 16] = s[off..off + 16].try_into().unwrap();
    // SAFETY: `c` is 16 readable bytes.
    unsafe { vld1q_s8(c.as_ptr()) }
}

/// The 16 bytes of `s` at `off`, every one of them bounds-checked.
#[inline]
#[target_feature(enable = "neon")]
fn ld_u8(s: &[u8], off: usize) -> uint8x16_t {
    let c: &[u8; 16] = s[off..off + 16].try_into().unwrap();
    // SAFETY: `c` is 16 readable bytes.
    unsafe { vld1q_u8(c.as_ptr()) }
}

/// Store `v` to the 16 bytes of `s` at `off`, every one of them bounds-checked.
#[inline]
#[target_feature(enable = "neon")]
fn st_s8(s: &mut [i8], off: usize, v: int8x16_t) {
    let c: &mut [i8; 16] = (&mut s[off..off + 16]).try_into().unwrap();
    // SAFETY: `c` is 16 writable bytes.
    unsafe { vst1q_s8(c.as_mut_ptr(), v) }
}

/// The lane sums of `v[0]`, `v[1]`, `v[2]` and `v[3]`, in that order.
#[inline]
#[target_feature(enable = "neon")]
fn hsum4(v: [int32x4_t; 4]) -> [i32; 4] {
    // `vpaddq_s32(a, b)` is `[a0+a1, a2+a3, b0+b1, b2+b3]`, so two rounds give each vector's sum.
    let s = vpaddq_s32(vpaddq_s32(v[0], v[1]), vpaddq_s32(v[2], v[3]));
    let mut out = [0i32; 4];
    // SAFETY: `out` is 4 writable i32s; lane 0 is stored to `out[0]`.
    unsafe { vst1q_s32(out.as_mut_ptr(), s) };
    out
}

/// For each of `N` activation super-blocks `a[t]`, the dot of the 256 weight codes `w` with it
/// over each 32-element sub-block: `out[t][s] = Σ_{i<32} w[32s+i]·a[t][32s+i]`. A product is at
/// most 2^14 in magnitude, so a lane (8 products) and a sub-block's sum (32) stay within ±2^19:
/// nothing wraps.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn dots32<const N: usize>(w: &[i8], a: [&[i8]; N]) -> [[i32; 8]; N] {
    let w: &[i8; 256] = w.try_into().unwrap();
    let a: [&[i8; 256]; N] = a.map(|a| a.try_into().unwrap());
    let mut out = [[0i32; 8]; N];
    for q in 0..2 {
        let mut acc = [[vdupq_n_s32(0); 4]; N];
        for j in 0..4 {
            let off = (4 * q + j) * 32;
            let (w0, w1) = (ld_s8(w, off), ld_s8(w, off + 16));
            for t in 0..N {
                let lo = vdotq_s32(vdupq_n_s32(0), w0, ld_s8(a[t], off));
                acc[t][j] = vdotq_s32(lo, w1, ld_s8(a[t], off + 16));
            }
        }
        for t in 0..N {
            out[t][4 * q..4 * q + 4].copy_from_slice(&hsum4(acc[t]));
        }
    }
    out
}

/// [`dots32`] over 16-element sub-blocks: `out[t][s] = Σ_{i<16} w[16s+i]·a[t][16s+i]`, within
/// ±2^18.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn dots16<const N: usize>(w: &[i8], a: [&[i8]; N]) -> [[i32; 16]; N] {
    let w: &[i8; 256] = w.try_into().unwrap();
    let a: [&[i8; 256]; N] = a.map(|a| a.try_into().unwrap());
    let mut out = [[0i32; 16]; N];
    for q in 0..4 {
        let mut acc = [[vdupq_n_s32(0); 4]; N];
        for j in 0..4 {
            let off = (4 * q + j) * 16;
            let wv = ld_s8(w, off);
            for t in 0..N {
                acc[t][j] = vdotq_s32(vdupq_n_s32(0), wv, ld_s8(a[t], off));
            }
        }
        for t in 0..N {
            out[t][4 * q..4 * q + 4].copy_from_slice(&hsum4(acc[t]));
        }
    }
    out
}

/// Activation `q8`'s codes for super-block `b`.
#[inline]
fn act(q8: &Q8, b: usize) -> &[i8] {
    &q8.qs[b * 256..b * 256 + 256]
}

/// `out[r]` for every activation row of `q8s`: `$tile::<TILE>($args.., rows)` over the rows
/// [`TILE`] at a time, then `$tile::<1>` over the rest.
macro_rules! tiled {
    ($q8s:expr, $out:expr, $tile:ident($($arg:expr),*)) => {{
        let (q8s, out): (&[Q8], &mut [f32]) = ($q8s, $out);
        let mut r = 0;
        while r + TILE <= q8s.len() {
            let v = $tile::<TILE>($($arg,)* std::array::from_fn(|t| &q8s[r + t]));
            out[r..r + TILE].copy_from_slice(&v);
            r += TILE;
        }
        while r < q8s.len() {
            out[r] = $tile::<1>($($arg,)* [&q8s[r]])[0];
            r += 1;
        }
    }};
}

// ─── Q4_K ──────────────────────────────────────────────────────────────────────────────────────

/// A Q4_K block's 128 nibble bytes as its 256 codes in element order (`0..=15` each): sub-block
/// `2k` is the low nibbles of bytes `32k..32k+32`, sub-block `2k+1` their high nibbles.
#[inline]
#[target_feature(enable = "neon")]
fn q4k_codes(qs: &[u8], out: &mut [i8]) {
    let mask = vdupq_n_u8(0x0F);
    for k in 0..4 {
        for h in 0..2 {
            let n = ld_u8(qs, 32 * k + 16 * h);
            st_s8(out, 64 * k + 16 * h, vreinterpretq_s8_u8(vandq_u8(n, mask)));
            st_s8(
                out,
                64 * k + 32 + 16 * h,
                vreinterpretq_s8_u8(vshrq_n_u8::<4>(n)),
            );
        }
    }
}

/// A Q4_K super-block's scales, unpacked once per weight row: `d`, `dmin`, and the eight
/// 6-bit sub-block scales and mins (`k4`, `0..=63`).
#[derive(Clone, Copy)]
struct Q4kScales {
    d: f32,
    dmin: f32,
    sc: [i32; 8],
    m: [i32; 8],
}

impl Q4kScales {
    /// The scales of the Q4_K block `blk` (its first 16 bytes are read).
    #[inline]
    fn of(blk: &[u8]) -> Self {
        let scales = &blk[4..16];
        Q4kScales {
            d: rdf16(&blk[0..2]),
            dmin: rdf16(&blk[2..4]),
            sc: std::array::from_fn(|s| k4(s, scales).0 as i32),
            m: std::array::from_fn(|s| k4(s, scales).1 as i32),
        }
    }
}

/// A Q4_K super-block's term, the scalar kernel's lines on its eight sub-block dots `iprod`
/// (each within ±32·15·128 < 2^16, so `sd` and `sm` stay within ±8·63·2^16 < 2^25).
#[inline]
fn q4k_block(k: &Q4kScales, q8: &Q8, b: usize, iprod: &[i32; 8]) -> f32 {
    let (mut sd, mut sm) = (0i32, 0i32);
    for s in 0..8usize {
        sd += k.sc[s] * iprod[s];
        sm += k.m[s] * q8.bsums[b * 8 + s];
    }
    q8.d[b] * (k.d * sd as f32 - k.dmin * sm as f32)
}

/// `vec_dot_q4k` at the `dotprod` tier.
#[target_feature(enable = "neon,dotprod")]
pub(super) fn vec_dot_q4k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let mut sumf = 0f32;
    let mut w = [0i8; 256];
    for b in 0..in_f / 256 {
        let blk = &row[b * 144..b * 144 + 144];
        q4k_codes(&blk[16..144], &mut w);
        let [iprod] = dots32::<1>(&w, [act(q8, b)]);
        sumf += q4k_block(&Q4kScales::of(blk), q8, b, &iprod);
    }
    sumf
}

/// `N` activation rows against a Q4_K row whose codes are decoded in `w` and whose blocks'
/// scales are `ks`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn q4k_tile<const N: usize>(ks: &[Q4kScales], w: &[i8], q: [&Q8; N]) -> [f32; N] {
    let mut sumf = [0f32; N];
    for (b, k) in ks.iter().enumerate() {
        let iprod = dots32::<N>(&w[b * 256..b * 256 + 256], q.map(|q| act(q, b)));
        for t in 0..N {
            sumf[t] += q4k_block(k, q[t], b, &iprod[t]);
        }
    }
    sumf
}

/// `vec_dot_q4k_batch` at the `dotprod` tier.
#[target_feature(enable = "neon,dotprod")]
pub(super) fn vec_dot_q4k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 256;
    let mut w = vec![0i8; nb * 256];
    let mut ks = Vec::with_capacity(nb);
    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        q4k_codes(&blk[16..144], &mut w[b * 256..b * 256 + 256]);
        ks.push(Q4kScales::of(blk));
    }
    tiled!(q8s, out, q4k_tile(&ks, &w));
}

// ─── Q6_K ──────────────────────────────────────────────────────────────────────────────────────

/// A Q6_K block's codes, less 32, in element order (`-32..=31` each). Element `half·128 +
/// 32k + l` (`l < 32`) takes its low four bits from `ql[half·64 + l]` (k = 0, 2) or
/// `ql[half·64 + 32 + l]` (k = 1, 3), the low nibble for k < 2 and the high one above, and its
/// top two bits from bits `2k..2k+2` of `qh[half·32 + l]`: the scalar kernel's `q1..q4`.
#[inline]
#[target_feature(enable = "neon")]
fn q6k_codes(blk: &[u8], out: &mut [i8]) {
    let (ql, qh) = (&blk[0..128], &blk[128..192]);
    let (m0f, m30, k32) = (vdupq_n_u8(0x0F), vdupq_n_u8(0x30), vdupq_n_u8(32));
    for half in 0..2 {
        for c in 0..2 {
            let a = ld_u8(ql, half * 64 + 16 * c);
            let b = ld_u8(ql, half * 64 + 32 + 16 * c);
            let h = ld_u8(qh, half * 32 + 16 * c);
            // `(h & 3) << 4`, `((h >> 2) & 3) << 4`, `((h >> 4) & 3) << 4`, `((h >> 6) & 3) << 4`.
            let q = [
                vorrq_u8(vandq_u8(a, m0f), vandq_u8(vshlq_n_u8::<4>(h), m30)),
                vorrq_u8(vandq_u8(b, m0f), vandq_u8(vshlq_n_u8::<2>(h), m30)),
                vorrq_u8(vshrq_n_u8::<4>(a), vandq_u8(h, m30)),
                vorrq_u8(vshrq_n_u8::<4>(b), vandq_u8(vshrq_n_u8::<2>(h), m30)),
            ];
            for (k, q) in q.into_iter().enumerate() {
                // `q - 32` wraps for q < 32, and the wrapped byte read as signed is `q - 32`.
                let v = vreinterpretq_s8_u8(vsubq_u8(q, k32));
                st_s8(out, half * 128 + 32 * k + 16 * c, v);
            }
        }
    }
}

/// A Q6_K super-block's term, the scalar kernel's lines on its sixteen sub-block dots `p` of
/// `q - 32` (each `sumi − 32·bsum`, within ±16·32·128 = 2^16, so `isum` stays within
/// ±16·128·2^16 = 2^27).
#[inline]
fn q6k_block(blk: &[u8], q8: &Q8, b: usize, p: &[i32; 16]) -> f32 {
    let (scales, d) = (&blk[192..208], rdf16(&blk[208..210]));
    let mut isum = 0i32;
    for sub in 0..16 {
        isum += scales[sub] as i8 as i32 * p[sub];
    }
    d * q8.d[b] * isum as f32
}

/// `vec_dot_q6k` at the `dotprod` tier.
#[target_feature(enable = "neon,dotprod")]
pub(super) fn vec_dot_q6k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let mut sumf = 0f32;
    let mut w = [0i8; 256];
    for b in 0..in_f / 256 {
        let blk = &row[b * 210..b * 210 + 210];
        q6k_codes(blk, &mut w);
        let [p] = dots16::<1>(&w, [act(q8, b)]);
        sumf += q6k_block(blk, q8, b, &p);
    }
    sumf
}

/// `N` activation rows against a Q6_K row whose codes are decoded in `w`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn q6k_tile<const N: usize>(row: &[u8], w: &[i8], q: [&Q8; N]) -> [f32; N] {
    let mut sumf = [0f32; N];
    for b in 0..w.len() / 256 {
        let p = dots16::<N>(&w[b * 256..b * 256 + 256], q.map(|q| act(q, b)));
        for t in 0..N {
            sumf[t] += q6k_block(&row[b * 210..b * 210 + 210], q[t], b, &p[t]);
        }
    }
    sumf
}

/// `vec_dot_q6k_batch` at the `dotprod` tier.
#[target_feature(enable = "neon,dotprod")]
pub(super) fn vec_dot_q6k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 256;
    let mut w = vec![0i8; nb * 256];
    for b in 0..nb {
        q6k_codes(&row[b * 210..b * 210 + 210], &mut w[b * 256..b * 256 + 256]);
    }
    tiled!(q8s, out, q6k_tile(row, &w));
}

// ─── Q8_0 ──────────────────────────────────────────────────────────────────────────────────────

/// Bytes per Q8_0 block: an f16 scale and 32 signed codes.
const Q8_0_BYTES: usize = 34;

/// The Q8_0 blocks of super-block `b` of `row`: eight blocks, the 256 elements of one
/// activation super-block.
#[inline]
fn q8_0_blocks(row: &[u8], b: usize) -> &[u8] {
    &row[b * 8 * Q8_0_BYTES..(b + 1) * 8 * Q8_0_BYTES]
}

/// The weight scales `d_w` of super-block `blocks`' eight Q8_0 blocks, as f32 (exact).
#[inline]
fn q8_0_scales(blocks: &[u8]) -> [f32; 8] {
    std::array::from_fn(|s| rdf16(&blocks[s * Q8_0_BYTES..s * Q8_0_BYTES + 2]))
}

/// A super-block's term of a Q8_0 dot, the scalar kernel's lines: `sumf += d8·d_w·iprod` for
/// each of its eight blocks in order (`iprod[s]` within ±32·2^14 = 2^19).
#[inline]
fn q8_0_block(d_w: &[f32; 8], d8: f32, iprod: &[i32; 8], mut sumf: f32) -> f32 {
    for s in 0..8 {
        sumf += d8 * d_w[s] * iprod[s] as f32;
    }
    sumf
}

/// `vec_dot_q8_0` at the `dotprod` tier: each block's codes are dotted where they lie.
#[target_feature(enable = "neon,dotprod")]
pub(super) fn vec_dot_q8_0(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let mut sumf = 0f32;
    for b in 0..in_f / 256 {
        let blocks = q8_0_blocks(row, b);
        let a = act(q8, b);
        let mut acc = [vdupq_n_s32(0); 8];
        for s in 0..8 {
            let at = s * Q8_0_BYTES + 2;
            let w0 = vreinterpretq_s8_u8(ld_u8(blocks, at));
            let w1 = vreinterpretq_s8_u8(ld_u8(blocks, at + 16));
            let lo = vdotq_s32(vdupq_n_s32(0), w0, ld_s8(a, 32 * s));
            acc[s] = vdotq_s32(lo, w1, ld_s8(a, 32 * s + 16));
        }
        let mut iprod = [0i32; 8];
        iprod[..4].copy_from_slice(&hsum4([acc[0], acc[1], acc[2], acc[3]]));
        iprod[4..].copy_from_slice(&hsum4([acc[4], acc[5], acc[6], acc[7]]));
        sumf = q8_0_block(&q8_0_scales(blocks), q8.d[b], &iprod, sumf);
    }
    sumf
}

/// `N` activation rows against a Q8_0 row whose codes are gathered in `w` and whose
/// super-blocks' weight scales are `dw`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn q8_0_tile<const N: usize>(dw: &[[f32; 8]], w: &[i8], q: [&Q8; N]) -> [f32; N] {
    let mut sumf = [0f32; N];
    for (b, d_w) in dw.iter().enumerate() {
        let iprod = dots32::<N>(&w[b * 256..b * 256 + 256], q.map(|q| act(q, b)));
        for t in 0..N {
            sumf[t] = q8_0_block(d_w, q[t].d[b], &iprod[t], sumf[t]);
        }
    }
    sumf
}

/// `vec_dot_q8_0_batch` at the `dotprod` tier.
#[target_feature(enable = "neon,dotprod")]
pub(super) fn vec_dot_q8_0_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 256;
    let mut w = vec![0i8; nb * 256];
    let blocks = row[..nb * 8 * Q8_0_BYTES].as_chunks::<Q8_0_BYTES>().0;
    for (blk, codes) in blocks.iter().zip(w.as_chunks_mut::<32>().0) {
        for (c, &x) in codes.iter_mut().zip(&blk[2..]) {
            *c = x as i8;
        }
    }
    let dw: Vec<[f32; 8]> = (0..nb).map(|b| q8_0_scales(q8_0_blocks(row, b))).collect();
    tiled!(q8s, out, q8_0_tile(&dw, &w));
}
