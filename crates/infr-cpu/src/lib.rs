//! CPU reference backend — a correctness-first interpreter of the backend-agnostic
//! [`infr_core`] compute [`Graph`]. Projection matmuls and attention use rayon for multi-core
//! parallelism; QK/PV inner loops use an 8-accumulator dot for AVX autovectorization.
//! Weights are read **zero-copy from the GGUF mmap** (no `memcpy`, no owned RAM): the bulk
//! projection weights (`Op::Linear`) are dequantized one row at a time straight from the mapping
//! inside the dot, so 12B / MoE models cost only their on-disk size in page cache. Only the tiny
//! norm weights are dequant-cached; the model writes (KV / conv / recurrent state, per-step IO) use
//! small owned buffers. It exists to (a) run every model without a GPU and (b) serve as the oracle
//! the GPU backends are validated against.
#![allow(clippy::needless_range_loop)]

pub mod kvquant;
pub mod turbo;

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Capabilities, GraphPlan, Plan};
use infr_core::error::Result;
use infr_core::graph::{Activation, AttnMask, Graph, Op, TensorKind};
use infr_core::tensor::{DType, TensorId};
use infr_gguf::dequant::{dequant_block, k4, rdf16};
use infr_gguf::TensorBytes;
use rayon::prelude::*;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Activation quantized to Q8 over 256-element super-blocks: `qs[i] = round(x[i]/d[blk])` (int8),
/// `d[blk] = max|x|/127`. Quantize the activation ONCE per matvec, then integer-dot it against the
/// quantized weight rows (llama.cpp's q8_K path) — no per-row f32 weight expansion.
struct Q8 {
    qs: Vec<i8>,
    d: Vec<f32>,
    /// Sub-block sums: `bsums[b*8+s]` = Σ `qs[b*256 + s*32 .. +32]` as i32.
    /// One entry per 32-element sub-block (8 per 256-element super-block).
    /// Precomputed once at quantization time; reused across all weight rows so the
    /// `sm` accumulation in `vec_dot_q4k` (Σ m·Σq8) avoids O(rows·256) re-summation.
    /// Mirrors llama.cpp's `block_q8_K.bsums`.
    bsums: Vec<i32>,
}

fn quantize_q8(x: &[f32]) -> Q8 {
    let nb = x.len() / 256;
    let mut qs = vec![0i8; nb * 256];
    let mut d = vec![0f32; nb];
    for b in 0..nb {
        let blk = &x[b * 256..b * 256 + 256];
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let dd = amax / 127.0;
        let id = if dd > 0.0 { 1.0 / dd } else { 0.0 };
        d[b] = dd;
        for (i, &v) in blk.iter().enumerate() {
            qs[b * 256 + i] = (v * id).round().clamp(-127.0, 127.0) as i8;
        }
    }
    // Precompute per-32-elem-sub-block sums (used by vec_dot_q4k for the min-scale term).
    let mut bsums = vec![0i32; nb * 8];
    for b in 0..nb {
        for s in 0..8usize {
            bsums[b * 8 + s] = qs[b * 256 + s * 32..b * 256 + s * 32 + 32]
                .iter()
                .map(|&q| q as i32)
                .sum();
        }
    }
    Q8 { qs, d, bsums }
}

/// `Σ weight·x` for one Q4_K row (144 bytes / 256 elems) against the Q8 activation. Weight value is
/// `d·sc_s·q4 − dmin·m_s` over 8 sub-blocks of 32; dispatches to the best SIMD path available at
/// runtime (avx512bw → avx2 → scalar). The `sm` term uses `q8.bsums` (precomputed in `quantize_q8`)
/// instead of re-summing q8 values per row.
fn vec_dot_q4k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512bw detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q4k_avx512bw(row, q8, in_f) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q4k_avx2(row, q8, in_f) };
        }
    }
    vec_dot_q4k_scalar(row, q8, in_f)
}

/// Scalar fallback for `vec_dot_q4k`; also used on non-x86 targets. Uses `q8.bsums` for `isum`.
fn vec_dot_q4k_scalar(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let nb = in_f / 256;
    let mut sumf = 0f32;
    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        let d = rdf16(&blk[0..2]);
        let dmin = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        for s in 0..8usize {
            let (sc, m) = k4(s, scales);
            let (half, hi) = (s / 2, s % 2 == 1);
            let qbyte = &qs[half * 32..half * 32 + 32];
            let q8s = &q8b[s * 32..s * 32 + 32];
            let mut iprod = 0i32;
            for l in 0..32 {
                let q4 = if hi {
                    (qbyte[l] >> 4) as i32
                } else {
                    (qbyte[l] & 0xF) as i32
                };
                iprod += q4 * q8s[l] as i32;
            }
            // isum = Σ q8s — precomputed in Q8::bsums, avoids re-summing per weight row.
            let isum = q8.bsums[b * 8 + s];
            sd += sc as i32 * iprod;
            sm += m as i32 * isum;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// Horizontal reduction: sum 8 × i32 in a ymm register to a scalar i32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hadd_i32_ymm(v: std::arch::x86_64::__m256i) -> i32 {
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
unsafe fn hadd_i32_xmm(v: std::arch::x86_64::__m128i) -> i32 {
    use std::arch::x86_64::*;
    let h = _mm_hadd_epi32(v, v); // [a+b, c+d, a+b, c+d]
    let hh = _mm_hadd_epi32(h, h); // [a+b+c+d, ...]
    _mm_cvtsi128_si32(hh)
}

/// AVX2 kernel for `vec_dot_q4k`: one 32-element sub-block per iteration with 256-bit SIMD.
/// Nibbles are unpacked with `_mm256_maddubs_epi16` (unsigned×signed → i16 pair-sum) then widened
/// to i32 via `_mm256_madd_epi16`. `isum` comes from `q8.bsums`, not the inner loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q4k_avx2(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones = _mm256_set1_epi16(1i16);
    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        let d = rdf16(&blk[0..2]);
        let dmin = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        for s in 0..8usize {
            let (sc, m) = k4(s, scales);
            let hi = s % 2 == 1;
            let half = s / 2;
            // 32-byte nibble chunk shared by sub-blocks `2*half` (lo) and `2*half+1` (hi).
            let nibbles = _mm256_loadu_si256(qs[half * 32..].as_ptr() as *const __m256i);
            // Unpack nibbles: low or high 4 bits of each byte → u8 values 0–15.
            let q4 = if hi {
                _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo)
            } else {
                _mm256_and_si256(nibbles, mask_lo)
            };
            // Load 32 signed Q8 bytes for this sub-block.
            let q8v = _mm256_loadu_si256(q8b[s * 32..].as_ptr() as *const __m256i);
            // maddubs: a=u8 (q4, 0–15), b=i8 (q8) → 16×i16 pair-sums. No i16 overflow:
            // max pair = 15·127 + 15·127 = 3810 < 32767.
            let prod = _mm256_maddubs_epi16(q4, q8v);
            // madd with 1: widen 16×i16 → 8×i32 (pairs summed).
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            let isum = q8.bsums[b * 8 + s];
            sd += sc as i32 * iprod;
            sm += m as i32 * isum;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// AVX-512BW kernel for `vec_dot_q4k`: processes TWO sub-blocks per iteration (64 elements) with
/// 512-bit SIMD. For each pair (s=2k even, s=2k+1 odd), the 32-byte nibble chunk is unpacked into
/// a zmm (low nibbles in lower 256 bits, high nibbles in upper 256 bits) and multiplied against the
/// corresponding 64 contiguous Q8 bytes. The zmm result is split back to two ymm sums, giving both
/// `iprod_even` and `iprod_odd` in one pass — half the memory traffic of the avx2 path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn vec_dot_q4k_avx512bw(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones512 = _mm512_set1_epi16(1i16);
    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        let d = rdf16(&blk[0..2]);
        let dmin = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        // k=0..4 covers sub-block pairs (0,1), (2,3), (4,5), (6,7).
        for k in 0..4usize {
            let (sc_e, m_e) = k4(2 * k, scales);
            let (sc_o, m_o) = k4(2 * k + 1, scales);
            // Load 32 nibble bytes serving both sub-blocks 2k (low) and 2k+1 (high).
            let nibbles = _mm256_loadu_si256(qs[k * 32..].as_ptr() as *const __m256i);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo); // sub-block 2k  (u8, 0–15)
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo); // sub-block 2k+1
                                                                                   // Pack into zmm: lower 256 = lo_nib (for 2k), upper 256 = hi_nib (for 2k+1).
            let q4_zmm = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo_nib), hi_nib);
            // Load 64 Q8 bytes: [2k*32..(2k+1)*32] in lower, [(2k+1)*32..(2k+2)*32] in upper.
            let q8_zmm = _mm512_loadu_si512(q8b[k * 64..].as_ptr() as *const __m512i);
            // maddubs 512-bit: u8(q4)×i8(q8) → 32×i16 pair-sums.
            let prod = _mm512_maddubs_epi16(q4_zmm, q8_zmm);
            // madd with 1: widen 32×i16 → 16×i32.
            let sum32 = _mm512_madd_epi16(prod, ones512);
            // Lower 256 = 8×i32 for sub-block 2k; upper 256 = 8×i32 for sub-block 2k+1.
            let lo_ymm = _mm512_castsi512_si256(sum32);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
            let iprod_e = hadd_i32_ymm(lo_ymm);
            let iprod_o = hadd_i32_ymm(hi_ymm);
            let isum_e = q8.bsums[b * 8 + 2 * k];
            let isum_o = q8.bsums[b * 8 + 2 * k + 1];
            sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
            sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// `Σ weight·x` for one Q6_K row (210 bytes / 256 elems). Dispatches to the best SIMD path
/// available at runtime (avx512bw → avx2 → scalar). Weight value is `d·sc·(q6−32)` over 16
/// sub-blocks of 16 (int8 scales).
fn vec_dot_q6k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512bw detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q6k_avx512bw(row, q8, in_f) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q6k_avx2(row, q8, in_f) };
        }
    }
    vec_dot_q6k_scalar(row, q8, in_f)
}

/// Scalar fallback for `vec_dot_q6k`; also used on non-x86 targets.
/// Accumulates `Σ q6·q8` and `Σ q8` per 16-element sub-block, then applies int8 scales.
fn vec_dot_q6k_scalar(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let nb = in_f / 256;
    let mut sumf = 0f32;
    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        let d = rdf16(&blk[208..210]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut sumi = [0i32; 16];
        let mut bsum = [0i32; 16];
        for half in 0..2 {
            let (qlo, qho, sco, base) = (half * 64, half * 32, half * 8, half * 128);
            for l in 0..32 {
                let is = l / 16;
                let q1 = (ql[qlo + l] & 0xF) | ((qh[qho + l] & 3) << 4);
                let q2 = (ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4);
                let q3 = (ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4);
                let q4 = (ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4);
                for (off, q, sci) in [(0, q1, 0), (32, q2, 2), (64, q3, 4), (96, q4, 6)] {
                    let sub = sco + is + sci;
                    let v = q8b[base + l + off] as i32;
                    sumi[sub] += q as i32 * v;
                    bsum[sub] += v;
                }
            }
        }
        let mut s = 0f32;
        for sub in 0..16 {
            s += scales[sub] as i8 as f32 * (sumi[sub] - 32 * bsum[sub]) as f32;
        }
        sumf += d * q8.d[b] * s;
    }
    sumf
}

/// AVX2 kernel for `vec_dot_q6k`: processes each of the 4 "columns" of 32 q6 values per half
/// using 256-bit SIMD. Each column maps to two consecutive 16-element sub-blocks (lower/upper
/// 128 bits after `madd`). q6 is reconstructed from `ql`/`qh` via byte-wise mask+shift; the
/// dot uses `maddubs(q6_u8, q8_i8) → madd(-, 1)`. The −32·bsum correction is computed from a
/// parallel `maddubs(1_u8, q8_i8)` sum so no per-element loop is needed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q6k_avx2(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;

    let mask_0f = _mm256_set1_epi8(0x0F_u8 as i8); // low nibble
    let mask_30 = _mm256_set1_epi8(0x30_u8 as i8); // bits 4-5
    let mask_03 = _mm256_set1_epi8(0x03_u8 as i8); // low 2 bits
    let ones_u8 = _mm256_set1_epi8(1i8); // for bsum via maddubs(1, q8)
    let ones_i16 = _mm256_set1_epi16(1i16);

    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        let d = rdf16(&blk[208..210]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];

        let mut s = 0f32;

        // Process two halves (each 128 elements = 8 sub-blocks of 16).
        for half in 0..2usize {
            let qlo = half * 64;
            let qho = half * 32;
            let sco = half * 8;
            let base = half * 128;

            // Load qh (32 bytes) and the two ql halves for this block-half.
            let qh_ymm = _mm256_loadu_si256(qh[qho..].as_ptr() as *const __m256i);
            let ql_lo = _mm256_loadu_si256(ql[qlo..].as_ptr() as *const __m256i);
            let ql_hi = _mm256_loadu_si256(ql[qlo + 32..].as_ptr() as *const __m256i);

            // (qh >> 2) & 0x03 per byte — reused in col2 and col6 reconstructions.
            // Trick: _mm256_srli_epi16(v, 2) shifts 16-bit lanes; masking with 0x03 per byte
            // gives the correct byte-wise >>2 result for each byte (cross-byte bleed is masked
            // away: (high_byte << 6) & 0x03 = 0 since high_byte << 6 occupies bits 6-7).
            let qh_sr2 = _mm256_and_si256(_mm256_srli_epi16(qh_ymm, 2), mask_03);

            // Reconstruct 4 × 32 q6 byte columns (values 0–63, stored as u8 in __m256i).
            //
            // col0: (ql_lo & 0x0F) | ((qh & 0x03) << 4)  →  q8b[base..base+32]
            // col2: (ql_hi & 0x0F) | ((qh>>2 & 0x03) << 4) → q8b[base+32..base+64]
            // col4: (ql_lo >> 4) | (qh & 0x30)             → q8b[base+64..base+96]
            // col6: (ql_hi >> 4) | ((qh>>2) & 0x30)        → q8b[base+96..base+128]
            //
            // Left-shift by 4 via _mm256_slli_epi16: for bytes 0–3 in range the low byte
            // result (v << 4) & 0xFF is correct (high byte has no bleed since input ≤ 3).
            let q6_c0 = _mm256_or_si256(
                _mm256_and_si256(ql_lo, mask_0f),
                _mm256_slli_epi16(_mm256_and_si256(qh_ymm, mask_03), 4),
            );
            let q6_c2 = _mm256_or_si256(
                _mm256_and_si256(ql_hi, mask_0f),
                _mm256_slli_epi16(qh_sr2, 4),
            );
            // col4: (qh >> 4 & 3) << 4 = qh & 0x30 (bits 4-5 of qh[i]).
            let q6_c4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_lo, 4), mask_0f),
                _mm256_and_si256(qh_ymm, mask_30),
            );
            // col6: (qh >> 6 & 3) << 4 = (qh >> 2) & 0x30 (bits 6-7 of qh[i] → positions 4-5).
            let q6_c6 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_hi, 4), mask_0f),
                _mm256_and_si256(_mm256_srli_epi16(qh_ymm, 2), mask_30),
            );

            // For each of the 4 columns: dot with the corresponding 32 q8 bytes, then split the
            // 8×i32 ymm result into lower 4×i32 (sub-block `sco+ci*2`) and upper 4×i32
            // (sub-block `sco+ci*2+1`). Reduction order matches scalar sub=0..16, ensuring
            // bit-identical f32 accumulation.
            for (ci, q6_col) in [q6_c0, q6_c2, q6_c4, q6_c6].iter().enumerate() {
                let q8_ymm = _mm256_loadu_si256(q8b[base + ci * 32..].as_ptr() as *const __m256i);

                // maddubs: q6_u8 (0–63) × q8_i8 (±127) → 16×i16 pair-sums (max ±8001 < 32767).
                // madd with 1: widen 16×i16 → 8×i32 (pairs summed).
                let prod = _mm256_maddubs_epi16(*q6_col, q8_ymm);
                let sum32 = _mm256_madd_epi16(prod, ones_i16);

                // bsum: maddubs(1_u8, q8_i8) = q8 pair-sums as i16; madd → 4-group i32 sums.
                let bsum_i16 = _mm256_maddubs_epi16(ones_u8, q8_ymm);
                let bsum_i32 = _mm256_madd_epi16(bsum_i16, ones_i16);

                // Lower 128 bits = elements 0..15 (sub-block `sco+ci*2`).
                // Upper 128 bits = elements 16..31 (sub-block `sco+ci*2+1`).
                let sum_lo = _mm256_castsi256_si128(sum32);
                let sum_hi = _mm256_extracti128_si256::<1>(sum32);
                let bs_lo = _mm256_castsi256_si128(bsum_i32);
                let bs_hi = _mm256_extracti128_si256::<1>(bsum_i32);

                let iprod_0 = hadd_i32_xmm(sum_lo);
                let iprod_1 = hadd_i32_xmm(sum_hi);
                let bs_0 = hadd_i32_xmm(bs_lo);
                let bs_1 = hadd_i32_xmm(bs_hi);

                let sub_0 = sco + ci * 2;
                let sub_1 = sco + ci * 2 + 1;
                s += scales[sub_0] as i8 as f32 * (iprod_0 - 32 * bs_0) as f32;
                s += scales[sub_1] as i8 as f32 * (iprod_1 - 32 * bs_1) as f32;
            }
        }
        sumf += d * q8.d[b] * s;
    }
    sumf
}

/// AVX-512BW kernel for `vec_dot_q6k`: processes BOTH halves of a 256-element block simultaneously
/// using zmm registers (512-bit). The two ql half-lo slices are packed into one zmm via
/// `_mm512_inserti64x4`; qh loads as a single 64-byte zmm (the two halves are contiguous). For
/// each of the 4 q6 columns, both halves' q8 are also packed into one zmm, so a single
/// `maddubs512 → madd512` covers 64 elements at once. Results are split back to two ymm
/// (h0/h1) then two xmm per ymm (per-sub-block) for the scalar scale accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn vec_dot_q6k_avx512bw(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;

    let mask_0f_z = _mm512_set1_epi8(0x0F_u8 as i8);
    let mask_30_z = _mm512_set1_epi8(0x30_u8 as i8);
    let mask_03_z = _mm512_set1_epi8(0x03_u8 as i8);
    let ones_u8_z = _mm512_set1_epi8(1i8);
    let ones_i16_z = _mm512_set1_epi16(1i16);

    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        let d = rdf16(&blk[208..210]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];

        // qh is 64 contiguous bytes: h0 in bits 0-255, h1 in bits 256-511 → one zmm load.
        let qh_z = _mm512_loadu_si512(qh.as_ptr() as *const __m512i);

        // ql_lo_z: lower 256 = ql[0..32] (h0_lo), upper 256 = ql[64..96] (h1_lo).
        // ql_hi_z: lower 256 = ql[32..64] (h0_hi), upper 256 = ql[96..128] (h1_hi).
        // h0 and h1 slices are non-contiguous (separated by 32 bytes), so 2 ymm loads + insert.
        let ql_lo_h0 = _mm256_loadu_si256(ql[0..].as_ptr() as *const __m256i);
        let ql_lo_h1 = _mm256_loadu_si256(ql[64..].as_ptr() as *const __m256i);
        let ql_lo_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ql_lo_h0), ql_lo_h1);

        let ql_hi_h0 = _mm256_loadu_si256(ql[32..].as_ptr() as *const __m256i);
        let ql_hi_h1 = _mm256_loadu_si256(ql[96..].as_ptr() as *const __m256i);
        let ql_hi_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ql_hi_h0), ql_hi_h1);

        // Same byte-wise shift/mask tricks as AVX2 but on 512-bit registers.
        let qh_sr2_z = _mm512_and_si512(_mm512_srli_epi16(qh_z, 2), mask_03_z);

        let q6_c0_z = _mm512_or_si512(
            _mm512_and_si512(ql_lo_z, mask_0f_z),
            _mm512_slli_epi16(_mm512_and_si512(qh_z, mask_03_z), 4),
        );
        let q6_c2_z = _mm512_or_si512(
            _mm512_and_si512(ql_hi_z, mask_0f_z),
            _mm512_slli_epi16(qh_sr2_z, 4),
        );
        let q6_c4_z = _mm512_or_si512(
            _mm512_and_si512(_mm512_srli_epi16(ql_lo_z, 4), mask_0f_z),
            _mm512_and_si512(qh_z, mask_30_z),
        );
        let q6_c6_z = _mm512_or_si512(
            _mm512_and_si512(_mm512_srli_epi16(ql_hi_z, 4), mask_0f_z),
            _mm512_and_si512(_mm512_srli_epi16(qh_z, 2), mask_30_z),
        );

        // Collect per-sub-block i32 values in arrays; accumulate in 0..16 order (scalar-identical).
        let mut simd_sumi = [0i32; 16];
        let mut simd_bsum = [0i32; 16];

        for (ci, q6_col_z) in [q6_c0_z, q6_c2_z, q6_c4_z, q6_c6_z].iter().enumerate() {
            // q8 for h0 column ci: q8b[ci*32..ci*32+32]; h1: q8b[128+ci*32..128+ci*32+32].
            let q8_h0 = _mm256_loadu_si256(q8b[ci * 32..].as_ptr() as *const __m256i);
            let q8_h1 = _mm256_loadu_si256(q8b[128 + ci * 32..].as_ptr() as *const __m256i);
            let q8_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q8_h0), q8_h1);

            // 512-bit dot: maddubs → madd.
            let prod = _mm512_maddubs_epi16(*q6_col_z, q8_z);
            let sum32_z = _mm512_madd_epi16(prod, ones_i16_z);

            let bsum_i16_z = _mm512_maddubs_epi16(ones_u8_z, q8_z);
            let bsum_i32_z = _mm512_madd_epi16(bsum_i16_z, ones_i16_z);

            // Lower ymm = h0 (sub-blocks ci*2, ci*2+1); upper ymm = h1 (sub-blocks 8+ci*2, 8+ci*2+1).
            let sum_h0 = _mm512_castsi512_si256(sum32_z);
            let sum_h1 = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let bsum_h0 = _mm512_castsi512_si256(bsum_i32_z);
            let bsum_h1 = _mm512_extracti64x4_epi64::<1>(bsum_i32_z);

            // Each ymm: lower xmm = first 16 elements (is=0), upper xmm = elements 16..31 (is=1).
            let s_h0_lo = _mm256_castsi256_si128(sum_h0);
            let s_h0_hi = _mm256_extracti128_si256::<1>(sum_h0);
            let s_h1_lo = _mm256_castsi256_si128(sum_h1);
            let s_h1_hi = _mm256_extracti128_si256::<1>(sum_h1);
            let b_h0_lo = _mm256_castsi256_si128(bsum_h0);
            let b_h0_hi = _mm256_extracti128_si256::<1>(bsum_h0);
            let b_h1_lo = _mm256_castsi256_si128(bsum_h1);
            let b_h1_hi = _mm256_extracti128_si256::<1>(bsum_h1);

            simd_sumi[ci * 2] = hadd_i32_xmm(s_h0_lo);
            simd_sumi[ci * 2 + 1] = hadd_i32_xmm(s_h0_hi);
            simd_sumi[8 + ci * 2] = hadd_i32_xmm(s_h1_lo);
            simd_sumi[8 + ci * 2 + 1] = hadd_i32_xmm(s_h1_hi);

            simd_bsum[ci * 2] = hadd_i32_xmm(b_h0_lo);
            simd_bsum[ci * 2 + 1] = hadd_i32_xmm(b_h0_hi);
            simd_bsum[8 + ci * 2] = hadd_i32_xmm(b_h1_lo);
            simd_bsum[8 + ci * 2 + 1] = hadd_i32_xmm(b_h1_hi);
        }

        // Accumulate in sub 0..16 order — identical to scalar's final loop.
        let mut s = 0f32;
        for sub in 0..16 {
            s += scales[sub] as i8 as f32 * (simd_sumi[sub] - 32 * simd_bsum[sub]) as f32;
        }
        sumf += d * q8.d[b] * s;
    }
    sumf
}

// ─── Q8_0 integer dot kernels ─────────────────────────────────────────────────
//
// Q8_0 weight layout: 34 bytes / 32 elements.  Bytes 0..2 = f16 scale `d`; bytes 2..34 = i8 qs.
// Activation comes in as a `Q8` super-block (256 elems), so one super-block covers 8 Q8_0 weight
// blocks.  Since both weight and activation are i8, we use the llama.cpp sign trick:
// `maddubs(abs(qw), sign(qw)·qx)` = `Σ qw[i]·qx[i]` without overflow into i16.

fn vec_dot_q8_0(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q8_0_avx512bw(row, q8, in_f) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q8_0_avx2(row, q8, in_f) };
        }
    }
    vec_dot_q8_0_scalar(row, q8, in_f)
}

fn vec_dot_q8_0_scalar(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let bpr = 34usize; // bytes per Q8_0 weight block (32 elems)
    let nb_super = in_f / 256; // activation super-blocks
    let mut sumf = 0f32;
    for b in 0..nb_super {
        let d8 = q8.d[b];
        for s in 0..8usize {
            let wb = b * 8 + s;
            let blk = &row[wb * bpr..wb * bpr + bpr];
            let d_w = rdf16(&blk[0..2]);
            let qw = &blk[2..34];
            let qx = &q8.qs[b * 256 + s * 32..b * 256 + s * 32 + 32];
            let iprod: i32 = (0..32).map(|i| qw[i] as i8 as i32 * qx[i] as i32).sum();
            sumf += d8 * d_w * iprod as f32;
        }
    }
    sumf
}

/// AVX2 kernel for `vec_dot_q8_0`: one 32-element Q8_0 weight block per iteration.
/// Sign trick: `maddubs(abs(qw), sign(qw)·qx)` handles i8×i8 without overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q8_0_avx2(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let bpr = 34usize;
    let nb_super = in_f / 256;
    let ones = _mm256_set1_epi16(1i16);
    let mut sumf = 0f32;
    for b in 0..nb_super {
        let d8 = q8.d[b];
        for s in 0..8usize {
            let wb = b * 8 + s;
            let blk = &row[wb * bpr..wb * bpr + bpr];
            let d_w = rdf16(&blk[0..2]);
            let qw = _mm256_loadu_si256(blk[2..].as_ptr() as *const __m256i);
            let qx = _mm256_loadu_si256(q8.qs[b * 256 + s * 32..].as_ptr() as *const __m256i);
            // sign trick: qx_signed = sign(qw) * qx;  abs(qw) stays unsigned
            let qw_abs = _mm256_abs_epi8(qw);
            let qx_signed = _mm256_sign_epi8(qx, qw);
            let prod = _mm256_maddubs_epi16(qw_abs, qx_signed);
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d8 * d_w * iprod as f32;
        }
    }
    sumf
}

/// AVX-512BW kernel for `vec_dot_q8_0`: TWO 32-element Q8_0 blocks per iteration (64 elems / zmm).
/// Sign trick is applied at ymm level (no `_mm512_sign_epi8`), then results are packed into zmm
/// for a single `maddubs512 → madd512` pass.  Activation bytes for the pair are contiguous in
/// `q8.qs`, so a single `_mm512_loadu_si512` covers both.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn vec_dot_q8_0_avx512bw(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let bpr = 34usize;
    let nb_super = in_f / 256;
    let ones512 = _mm512_set1_epi16(1i16);
    let mut sumf = 0f32;
    for b in 0..nb_super {
        let d8 = q8.d[b];
        for k in 0..4usize {
            let s0 = 2 * k;
            let s1 = 2 * k + 1;
            let wb0 = b * 8 + s0;
            let wb1 = b * 8 + s1;
            let d_w0 = rdf16(&row[wb0 * bpr..wb0 * bpr + 2]);
            let d_w1 = rdf16(&row[wb1 * bpr..wb1 * bpr + 2]);
            // Load weight i8 bytes (32 each, non-contiguous due to f16 header)
            let qw0 = _mm256_loadu_si256(row[wb0 * bpr + 2..].as_ptr() as *const __m256i);
            let qw1 = _mm256_loadu_si256(row[wb1 * bpr + 2..].as_ptr() as *const __m256i);
            // Load 64 contiguous activation bytes as zmm (s0*32 and s1*32 are adjacent)
            let qx_z = _mm512_loadu_si512(q8.qs[b * 256 + s0 * 32..].as_ptr() as *const __m512i);
            // Sign trick at ymm level (no avx512 sign_epi8)
            let qx0 = _mm512_castsi512_si256(qx_z);
            let qx1 = _mm512_extracti64x4_epi64::<1>(qx_z);
            let qw_abs0 = _mm256_abs_epi8(qw0);
            let qw_abs1 = _mm256_abs_epi8(qw1);
            let qx_s0 = _mm256_sign_epi8(qx0, qw0);
            let qx_s1 = _mm256_sign_epi8(qx1, qw1);
            // Pack into zmm
            let qw_a_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qw_abs0), qw_abs1);
            let qx_s_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qx_s0), qx_s1);
            // 512-bit dot
            let prod = _mm512_maddubs_epi16(qw_a_z, qx_s_z);
            let sum32_z = _mm512_madd_epi16(prod, ones512);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            sumf += d8 * (d_w0 * iprod0 as f32 + d_w1 * iprod1 as f32);
        }
    }
    sumf
}

// ─── Q5_K integer dot kernels ─────────────────────────────────────────────────
//
// Q5_K block layout (176 bytes / 256 elems):
//   [f16 d][f16 dmin][scales[12]][qh[32]][ql[128]]
// q5 = (ql_nibble) | (((qh[l] >> bit) & 1) << 4)  ∈ 0..31  (UNSIGNED → maddubs works directly)
// Dot formula: d·sc·Σ(q5·qx) − dmin·m·Σqx  — identical structure to Q4_K.
// `q8.bsums` provides Σqx per 32-elem sub-block (precomputed in quantize_q8).

fn vec_dot_q5k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q5k_avx512bw(row, q8, in_f) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q5k_avx2(row, q8, in_f) };
        }
    }
    vec_dot_q5k_scalar(row, q8, in_f)
}

fn vec_dot_q5k_scalar(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let nb = in_f / 256;
    let mut sumf = 0f32;
    for b in 0..nb {
        let blk = &row[b * 176..b * 176 + 176];
        let d = rdf16(&blk[0..2]);
        let dmin = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in 0..4usize {
            let (sc_e, m_e) = k4(2 * j, scales);
            let (sc_o, m_o) = k4(2 * j + 1, scales);
            let ql_chunk = &ql[j * 32..j * 32 + 32];
            let q8_e = &q8b[2 * j * 32..(2 * j + 1) * 32];
            let q8_o = &q8b[(2 * j + 1) * 32..(2 * j + 2) * 32];
            let bsum_e = q8.bsums[b * 8 + 2 * j];
            let bsum_o = q8.bsums[b * 8 + 2 * j + 1];
            let mut iprod_e = 0i32;
            let mut iprod_o = 0i32;
            for l in 0..32 {
                let v = ql_chunk[l];
                let q5_e = (v & 0xF) as i32 + if qh[l] & u1 != 0 { 16 } else { 0 };
                let q5_o = (v >> 4) as i32 + if qh[l] & u2 != 0 { 16 } else { 0 };
                iprod_e += q5_e * q8_e[l] as i32;
                iprod_o += q5_o * q8_o[l] as i32;
            }
            sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
            sm += m_e as i32 * bsum_e + m_o as i32 * bsum_o;
            u1 <<= 2;
            u2 <<= 2;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// AVX2 kernel for `vec_dot_q5k`: one nibble-pair per iteration (64 elements = two 32-elem
/// sub-blocks).  High bit per element is extracted from `qh` using per-j bit masks: if the
/// bit is set the value adds 16.  `maddubs` works directly since q5 ∈ 0..31 (unsigned).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q5k_avx2(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let sixteen = _mm256_set1_epi8(0x10_u8 as i8);
    let ones = _mm256_set1_epi16(1i16);
    let zero = _mm256_setzero_si256();
    for b in 0..nb {
        let blk = &row[b * 176..b * 176 + 176];
        let d = rdf16(&blk[0..2]);
        let dmin = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let qh_ymm = _mm256_loadu_si256(qh.as_ptr() as *const __m256i);
        let (mut sd, mut sm) = (0i32, 0i32);
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in 0..4usize {
            let (sc_e, m_e) = k4(2 * j, scales);
            let (sc_o, m_o) = k4(2 * j + 1, scales);
            // Unpack nibbles from ql[j*32..+32]
            let nibbles = _mm256_loadu_si256(ql[j * 32..].as_ptr() as *const __m256i);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo);
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo);
            // High-bit extraction: if (qh[l] & u) != 0 → add 16 to that element.
            let u1v = _mm256_set1_epi8(u1 as i8);
            let u2v = _mm256_set1_epi8(u2 as i8);
            let has_e = _mm256_and_si256(qh_ymm, u1v);
            let has_o = _mm256_and_si256(qh_ymm, u2v);
            // andnot(cmpeq_zero, 0x10) = 0x10 where nonzero, 0 otherwise
            let high_e = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_e, zero), sixteen);
            let high_o = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_o, zero), sixteen);
            let q5_e = _mm256_or_si256(lo_nib, high_e);
            let q5_o = _mm256_or_si256(hi_nib, high_o);
            // Load Q8 activation bytes for both sub-blocks
            let q8_e = _mm256_loadu_si256(q8b[2 * j * 32..].as_ptr() as *const __m256i);
            let q8_o = _mm256_loadu_si256(q8b[(2 * j + 1) * 32..].as_ptr() as *const __m256i);
            // maddubs: q5 is u8 (0..31), q8 is i8 → direct, no sign trick needed
            let prod_e = _mm256_maddubs_epi16(q5_e, q8_e);
            let sum32_e = _mm256_madd_epi16(prod_e, ones);
            let iprod_e = hadd_i32_ymm(sum32_e);
            let prod_o = _mm256_maddubs_epi16(q5_o, q8_o);
            let sum32_o = _mm256_madd_epi16(prod_o, ones);
            let iprod_o = hadd_i32_ymm(sum32_o);
            let isum_e = q8.bsums[b * 8 + 2 * j];
            let isum_o = q8.bsums[b * 8 + 2 * j + 1];
            sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
            sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
            u1 <<= 2;
            u2 <<= 2;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// AVX-512BW kernel for `vec_dot_q5k`: identical structure to the Q4_K AVX-512BW kernel but with
/// the 5th bit ORed in from `qh`.  Each iteration (k=0..4) processes one nibble pair (64 elements)
/// via a zmm.  The high bit is extracted from `qh_ymm` using per-k bit masks at ymm width; results
/// are inserted into zmm for the 512-bit `maddubs → madd` pass.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn vec_dot_q5k_avx512bw(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let sixteen = _mm256_set1_epi8(0x10_u8 as i8);
    let ones512 = _mm512_set1_epi16(1i16);
    let zero256 = _mm256_setzero_si256();
    for b in 0..nb {
        let blk = &row[b * 176..b * 176 + 176];
        let d = rdf16(&blk[0..2]);
        let dmin = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let qh_ymm = _mm256_loadu_si256(qh.as_ptr() as *const __m256i);
        let (mut sd, mut sm) = (0i32, 0i32);
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for k in 0..4usize {
            let (sc_e, m_e) = k4(2 * k, scales);
            let (sc_o, m_o) = k4(2 * k + 1, scales);
            let nibbles = _mm256_loadu_si256(ql[k * 32..].as_ptr() as *const __m256i);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo);
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo);
            // High-bit extraction per bit pair
            let u1v = _mm256_set1_epi8(u1 as i8);
            let u2v = _mm256_set1_epi8(u2 as i8);
            let has_e = _mm256_and_si256(qh_ymm, u1v);
            let has_o = _mm256_and_si256(qh_ymm, u2v);
            let high_e = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_e, zero256), sixteen);
            let high_o = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_o, zero256), sixteen);
            // q5 values (0..31, unsigned)
            let q5_e = _mm256_or_si256(lo_nib, high_e);
            let q5_o = _mm256_or_si256(hi_nib, high_o);
            // Pack into zmm: lower 256 = even sub-block, upper 256 = odd sub-block
            let q5_zmm = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q5_e), q5_o);
            // 64 contiguous Q8 activation bytes for this pair
            let q8_zmm = _mm512_loadu_si512(q8b[k * 64..].as_ptr() as *const __m512i);
            let prod = _mm512_maddubs_epi16(q5_zmm, q8_zmm);
            let sum32_z = _mm512_madd_epi16(prod, ones512);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod_e = hadd_i32_ymm(lo_ymm);
            let iprod_o = hadd_i32_ymm(hi_ymm);
            let isum_e = q8.bsums[b * 8 + 2 * k];
            let isum_o = q8.bsums[b * 8 + 2 * k + 1];
            sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
            sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
            u1 <<= 2;
            u2 <<= 2;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

// ─── Batched dot kernels (prefill: m > 1) ────────────────────────────────────
//
// Each `vec_dot_qXk_batch(row, q8s, in_f, out)` is equivalent to calling
// `vec_dot_qXk(row, &q8s[r], in_f)` for every r, but decodes the weight row
// ONCE and loops over the m token activations with the pre-decoded data.
//
// Bit-identity guarantee: the per-token f32 result equals the single-token
// kernel exactly (integer dots have no rounding, same accumulation grouping).

// ── Q4_K batch ────────────────────────────────────────────────────────────────

fn vec_dot_q4k_batch_scalar(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let nb = in_f / 256;

    // Decode weight row once: per-block d/dmin/sc/m, expanded nibbles (0–15).
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![0u32; nb * 8];
    let mut m_arr = vec![0u32; nb * 8];
    let mut q4_flat = vec![0u8; nb * 256]; // one byte per element, value 0–15

    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        d_arr[b] = rdf16(&blk[0..2]);
        dmin_arr[b] = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales);
            sc_arr[b * 8 + s] = sc;
            m_arr[b * 8 + s] = mv;
            let hi = s % 2 == 1;
            let half = s / 2;
            let qbyte = &qs[half * 32..half * 32 + 32];
            let flat = &mut q4_flat[b * 256 + s * 32..b * 256 + s * 32 + 32];
            for l in 0..32 {
                flat[l] = if hi { qbyte[l] >> 4 } else { qbyte[l] & 0xF };
            }
        }
    }

    // Per-token dot using pre-expanded data (identical order to scalar single-token).
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            for s in 0..8usize {
                let mut iprod = 0i32;
                let fb = &q4_flat[b * 256 + s * 32..b * 256 + s * 32 + 32];
                let q8b = &q8.qs[b * 256 + s * 32..b * 256 + s * 32 + 32];
                for l in 0..32 {
                    iprod += fb[l] as i32 * q8b[l] as i32;
                }
                sd += sc_arr[b * 8 + s] as i32 * iprod;
                sm += m_arr[b * 8 + s] as i32 * q8.bsums[b * 8 + s];
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q4k_batch_avx2(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones = _mm256_set1_epi16(1i16);

    // Pre-expand nibbles into flat[b*256..b*256+256] once.
    // Layout: flat[b*256 + s*32 .. b*256 + s*32 + 32] = expanded q4 for sub-block s.
    // For pair k: flat[b*256 + k*64..b*256 + k*64 + 32] = lo nibbles (sub-block 2k),
    //             flat[b*256 + k*64 + 32..b*256 + k*64 + 64] = hi nibbles (sub-block 2k+1).
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![[0u32; 8]; nb];
    let mut m_arr = vec![[0u32; 8]; nb];
    let mut q4_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        d_arr[b] = rdf16(&blk[0..2]);
        dmin_arr[b] = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales);
            sc_arr[b][s] = sc;
            m_arr[b][s] = mv;
        }
        // Unpack 4 nibble pairs with SIMD, store lo then hi in contiguous slots.
        let flat = &mut q4_flat[b * 256..b * 256 + 256];
        for k in 0..4usize {
            let nibbles = _mm256_loadu_si256(qs[k * 32..].as_ptr() as *const __m256i);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo); // sub-block 2k
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo); // sub-block 2k+1
            _mm256_storeu_si256(flat[k * 64..].as_mut_ptr() as *mut __m256i, lo_nib);
            _mm256_storeu_si256(flat[k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi_nib);
        }
    }

    // Per-token dots: load pre-expanded q4 ymm + q8 ymm, maddubs → madd → hadd.
    // Accumulation order identical to single-token AVX2 kernel → bit-identical result.
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            let flat = &q4_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for s in 0..8usize {
                let q4 = _mm256_loadu_si256(flat[s * 32..].as_ptr() as *const __m256i);
                let q8v = _mm256_loadu_si256(q8b[s * 32..].as_ptr() as *const __m256i);
                let prod = _mm256_maddubs_epi16(q4, q8v);
                let sum32 = _mm256_madd_epi16(prod, ones);
                let iprod = hadd_i32_ymm(sum32);
                let isum = q8.bsums[b * 8 + s];
                sd += sc_arr[b][s] as i32 * iprod;
                sm += m_arr[b][s] as i32 * isum;
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn vec_dot_q4k_batch_avx512bw(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones512 = _mm512_set1_epi16(1i16);

    // Pre-expand nibbles (same layout as AVX2 batch): flat[b*256 + k*64..+64] =
    // [lo_nib_2k (32 B), hi_nib_2k+1 (32 B)], directly loadable as a zmm per pair k.
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![[0u32; 8]; nb];
    let mut m_arr = vec![[0u32; 8]; nb];
    let mut q4_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        d_arr[b] = rdf16(&blk[0..2]);
        dmin_arr[b] = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales);
            sc_arr[b][s] = sc;
            m_arr[b][s] = mv;
        }
        let flat = &mut q4_flat[b * 256..b * 256 + 256];
        for k in 0..4usize {
            let nibbles = _mm256_loadu_si256(qs[k * 32..].as_ptr() as *const __m256i);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo);
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo);
            _mm256_storeu_si256(flat[k * 64..].as_mut_ptr() as *mut __m256i, lo_nib);
            _mm256_storeu_si256(flat[k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi_nib);
        }
    }

    // Per-token: one zmm load per pair (64 pre-expanded bytes) + one zmm q8 load.
    // maddubs512 → madd512 → split ymm → hadd×2. Identical to single-token avx512bw kernel.
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            let flat = &q4_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for k in 0..4usize {
                let (sc_e, m_e) = (sc_arr[b][2 * k], m_arr[b][2 * k]);
                let (sc_o, m_o) = (sc_arr[b][2 * k + 1], m_arr[b][2 * k + 1]);
                // flat[k*64..k*64+64]: lower 256 = sub-block 2k, upper 256 = sub-block 2k+1
                let q4_zmm = _mm512_loadu_si512(flat[k * 64..].as_ptr() as *const __m512i);
                let q8_zmm = _mm512_loadu_si512(q8b[k * 64..].as_ptr() as *const __m512i);
                let prod = _mm512_maddubs_epi16(q4_zmm, q8_zmm);
                let sum32 = _mm512_madd_epi16(prod, ones512);
                let lo_ymm = _mm512_castsi512_si256(sum32);
                let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
                let iprod_e = hadd_i32_ymm(lo_ymm);
                let iprod_o = hadd_i32_ymm(hi_ymm);
                let isum_e = q8.bsums[b * 8 + 2 * k];
                let isum_o = q8.bsums[b * 8 + 2 * k + 1];
                sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
                sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

/// Batch Q4_K dot: `out[r] = vec_dot_q4k(row, &q8s[r], in_f)` for all r, bit-identical to
/// the single-token kernel. The weight row is decoded ONCE; per-token work is the integer dot only.
fn vec_dot_q4k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q4k_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q4k_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q4k_batch_scalar(row, q8s, in_f, out);
}

// ── Q4_K 2-row tiled batch ────────────────────────────────────────────────────
//
// Process TWO output rows simultaneously so the Q8 activation data (loaded from
// L3 cache) is reused for both dots instead of loaded twice. This halves the L3
// bandwidth for Q8 reads which is the dominant bottleneck during large-batch prefill.
//
// `out_a` and `out_b` receive the dots for row `row_a` and `row_b` respectively.
// Bit-identical: each `out_x[r]` equals `vec_dot_q4k(row_x, &q8s[r], in_f)`.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn vec_dot_q4k_batch2_avx512bw(
    row_a: &[u8],
    row_b: &[u8],
    q8s: &[Q8],
    in_f: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones512 = _mm512_set1_epi16(1i16);

    // Pre-expand both weight rows.
    let mut d_a = vec![0f32; nb];
    let mut dmin_a = vec![0f32; nb];
    let mut sc_a = vec![[0u32; 8]; nb];
    let mut m_a = vec![[0u32; 8]; nb];
    let mut flat_a = vec![0u8; nb * 256];

    let mut d_b = vec![0f32; nb];
    let mut dmin_b = vec![0f32; nb];
    let mut sc_b = vec![[0u32; 8]; nb];
    let mut m_b = vec![[0u32; 8]; nb];
    let mut flat_b = vec![0u8; nb * 256];

    for b in 0..nb {
        // Row A
        let blk_a = &row_a[b * 144..b * 144 + 144];
        d_a[b] = rdf16(&blk_a[0..2]);
        dmin_a[b] = rdf16(&blk_a[2..4]);
        let scales_a = &blk_a[4..16];
        let qs_a = &blk_a[16..144];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales_a);
            sc_a[b][s] = sc;
            m_a[b][s] = mv;
        }
        let fa = &mut flat_a[b * 256..b * 256 + 256];
        for k in 0..4usize {
            let nibs = _mm256_loadu_si256(qs_a[k * 32..].as_ptr() as *const __m256i);
            let lo = _mm256_and_si256(nibs, mask_lo);
            let hi = _mm256_and_si256(_mm256_srli_epi16(nibs, 4), mask_lo);
            _mm256_storeu_si256(fa[k * 64..].as_mut_ptr() as *mut __m256i, lo);
            _mm256_storeu_si256(fa[k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi);
        }

        // Row B
        let blk_b = &row_b[b * 144..b * 144 + 144];
        d_b[b] = rdf16(&blk_b[0..2]);
        dmin_b[b] = rdf16(&blk_b[2..4]);
        let scales_b = &blk_b[4..16];
        let qs_b = &blk_b[16..144];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales_b);
            sc_b[b][s] = sc;
            m_b[b][s] = mv;
        }
        let fb = &mut flat_b[b * 256..b * 256 + 256];
        for k in 0..4usize {
            let nibs = _mm256_loadu_si256(qs_b[k * 32..].as_ptr() as *const __m256i);
            let lo = _mm256_and_si256(nibs, mask_lo);
            let hi = _mm256_and_si256(_mm256_srli_epi16(nibs, 4), mask_lo);
            _mm256_storeu_si256(fb[k * 64..].as_mut_ptr() as *mut __m256i, lo);
            _mm256_storeu_si256(fb[k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi);
        }
    }

    // Per-token: load q8 ONCE per block per pair k; compute both row A and row B dots.
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf_a = 0f32;
        let mut sumf_b = 0f32;
        for b in 0..nb {
            let (mut sd_a, mut sm_a) = (0i32, 0i32);
            let (mut sd_b, mut sm_b) = (0i32, 0i32);
            let fa = &flat_a[b * 256..b * 256 + 256];
            let fb = &flat_b[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];

            for k in 0..4usize {
                let (sca_e, ma_e) = (sc_a[b][2 * k], m_a[b][2 * k]);
                let (sca_o, ma_o) = (sc_a[b][2 * k + 1], m_a[b][2 * k + 1]);
                let (scb_e, mb_e) = (sc_b[b][2 * k], m_b[b][2 * k]);
                let (scb_o, mb_o) = (sc_b[b][2 * k + 1], m_b[b][2 * k + 1]);

                // Load q8 ONCE for this pair (shared by both row A and row B).
                let q8_zmm = _mm512_loadu_si512(q8b[k * 64..].as_ptr() as *const __m512i);

                // Row A dot
                let qa_zmm = _mm512_loadu_si512(fa[k * 64..].as_ptr() as *const __m512i);
                let prod_a = _mm512_maddubs_epi16(qa_zmm, q8_zmm);
                let sum32_a = _mm512_madd_epi16(prod_a, ones512);
                let lo_a = _mm512_castsi512_si256(sum32_a);
                let hi_a = _mm512_extracti64x4_epi64::<1>(sum32_a);

                // Row B dot (q8_zmm reused, no reload)
                let qb_zmm = _mm512_loadu_si512(fb[k * 64..].as_ptr() as *const __m512i);
                let prod_b = _mm512_maddubs_epi16(qb_zmm, q8_zmm);
                let sum32_b = _mm512_madd_epi16(prod_b, ones512);
                let lo_b = _mm512_castsi512_si256(sum32_b);
                let hi_b = _mm512_extracti64x4_epi64::<1>(sum32_b);

                let isum_e = q8.bsums[b * 8 + 2 * k];
                let isum_o = q8.bsums[b * 8 + 2 * k + 1];

                sd_a += sca_e as i32 * hadd_i32_ymm(lo_a) + sca_o as i32 * hadd_i32_ymm(hi_a);
                sm_a += ma_e as i32 * isum_e + ma_o as i32 * isum_o;

                sd_b += scb_e as i32 * hadd_i32_ymm(lo_b) + scb_o as i32 * hadd_i32_ymm(hi_b);
                sm_b += mb_e as i32 * isum_e + mb_o as i32 * isum_o;
            }
            sumf_a += q8.d[b] * (d_a[b] * sd_a as f32 - dmin_a[b] * sm_a as f32);
            sumf_b += q8.d[b] * (d_b[b] * sd_b as f32 - dmin_b[b] * sm_b as f32);
        }
        out_a[r] = sumf_a;
        out_b[r] = sumf_b;
    }
}

fn vec_dot_q4k_batch2(
    row_a: &[u8],
    row_b: &[u8],
    q8s: &[Q8],
    in_f: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512bw") {
        return unsafe { vec_dot_q4k_batch2_avx512bw(row_a, row_b, q8s, in_f, out_a, out_b) };
    }
    // Scalar fallback: call single-row batch twice (still unpack once per row).
    vec_dot_q4k_batch(row_a, q8s, in_f, out_a);
    vec_dot_q4k_batch(row_b, q8s, in_f, out_b);
}

// ── Q4_K 8-row tiled batch ────────────────────────────────────────────────────
//
// Process EIGHT output rows simultaneously: the Q8 activation zmm is loaded ONCE
// per (block, nibble-pair) and reused across all 8 row dots. This is 4× less
// activation traffic than the 2-row path and 8× less than the single-row path.
//
// `outs[i][r]` == `vec_dot_q4k(rows[i], &q8s[r], in_f)` — bit-identical to the
// single-token kernel (same per-block accumulation order; tiling only changes which
// rows are computed together, not the per-(row,token) arithmetic).

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn vec_dot_q4k_batch8_avx512bw(
    rows: [&[u8]; 8],
    q8s: &[Q8],
    in_f: usize,
    outs: [&mut [f32]; 8],
) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones512 = _mm512_set1_epi16(1i16);

    // Pre-expand all 8 weight rows once. Layout identical to the single-row AVX-512BW
    // batch kernel: flat[i][b*256 + k*64 .. +64] = [lo_nib 32B, hi_nib 32B] for pair k.
    let mut flats: [Vec<u8>; 8] = std::array::from_fn(|_| vec![0u8; nb * 256]);
    let mut d_arr: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0f32; nb]);
    let mut dmin_arr: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0f32; nb]);
    let mut sc_arr: [Vec<[u32; 8]>; 8] = std::array::from_fn(|_| vec![[0u32; 8]; nb]);
    let mut m_arr: [Vec<[u32; 8]>; 8] = std::array::from_fn(|_| vec![[0u32; 8]; nb]);

    for i in 0..8 {
        for b in 0..nb {
            let blk = &rows[i][b * 144..b * 144 + 144];
            d_arr[i][b] = rdf16(&blk[0..2]);
            dmin_arr[i][b] = rdf16(&blk[2..4]);
            let scales = &blk[4..16];
            let qs = &blk[16..144];
            for s in 0..8usize {
                let (sc, mv) = k4(s, scales);
                sc_arr[i][b][s] = sc;
                m_arr[i][b][s] = mv;
            }
            let f = &mut flats[i][b * 256..b * 256 + 256];
            for k in 0..4usize {
                let nibs = _mm256_loadu_si256(qs[k * 32..].as_ptr() as *const __m256i);
                let lo = _mm256_and_si256(nibs, mask_lo);
                let hi = _mm256_and_si256(_mm256_srli_epi16(nibs, 4), mask_lo);
                _mm256_storeu_si256(f[k * 64..].as_mut_ptr() as *mut __m256i, lo);
                _mm256_storeu_si256(f[k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi);
            }
        }
    }

    // Destructure outs so we can write to 8 independent &mut [f32] without aliasing.
    let [o0, o1, o2, o3, o4, o5, o6, o7] = outs;

    // Per-token dot: for each token r, load the Q8 activation zmm ONCE per (b, k) pair
    // and reuse it across all 8 weight rows — 8× the FMAs per activation load.
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = [0f32; 8];

        for b in 0..nb {
            // ── Deferred-hadd accumulation ──────────────────────────────────────────
            // Instead of hadd_i32_ymm inside the k loop (8 rows × 2 hadd × 4 k = 64
            // hadd calls/block, all on port 5), we accumulate scaled ymm vectors and
            // hadd once per row after all k-pairs are done (16 hadd calls/block).
            //
            // Bit-identical: hadd(Σ_k scale[k]·v[k]) = Σ_k scale[k]·hadd(v[k])
            // because hadd is a linear sum and integer mullo is exact (no overflow for
            // our value ranges: sc≤63, per-element sum≤4×15×127 ≈ 7620 → product ≤ ~480k
            // → fits i32; 8-element accumulation ≤ ~3.8M → fits i32).
            //
            // acc_lo[i] = Σ_k ( sc_e[k] × lo_ymm[k] )   — 8 × i32 lanes
            // acc_hi[i] = Σ_k ( sc_o[k] × hi_ymm[k] )   — 8 × i32 lanes
            // sd_i     = hadd(acc_lo[i]) + hadd(acc_hi[i])
            let mut acc_lo = [_mm256_setzero_si256(); 8];
            let mut acc_hi = [_mm256_setzero_si256(); 8];
            let mut sm = [0i32; 8];
            let q8b = &q8.qs[b * 256..b * 256 + 256];

            for k in 0..4usize {
                // ── ONE activation load for pair k, shared by all 8 weight rows ──
                let q8_zmm = _mm512_loadu_si512(q8b[k * 64..].as_ptr() as *const __m512i);
                let isum_e = q8.bsums[b * 8 + 2 * k];
                let isum_o = q8.bsums[b * 8 + 2 * k + 1];

                // ── 8 weight row dots against the shared q8_zmm ──
                for i in 0..8usize {
                    let (sc_e, ma_e) = (sc_arr[i][b][2 * k], m_arr[i][b][2 * k]);
                    let (sc_o, ma_o) = (sc_arr[i][b][2 * k + 1], m_arr[i][b][2 * k + 1]);
                    let qi_zmm =
                        _mm512_loadu_si512(flats[i][b * 256 + k * 64..].as_ptr() as *const __m512i);
                    let prod = _mm512_maddubs_epi16(qi_zmm, q8_zmm);
                    let sum32 = _mm512_madd_epi16(prod, ones512);
                    let lo_ymm = _mm512_castsi512_si256(sum32);
                    let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
                    // Scale each 8×i32 sub-block by its per-sub-block scale and
                    // accumulate into ymm registers — no hadd in the hot path.
                    acc_lo[i] = _mm256_add_epi32(
                        acc_lo[i],
                        _mm256_mullo_epi32(lo_ymm, _mm256_set1_epi32(sc_e as i32)),
                    );
                    acc_hi[i] = _mm256_add_epi32(
                        acc_hi[i],
                        _mm256_mullo_epi32(hi_ymm, _mm256_set1_epi32(sc_o as i32)),
                    );
                    sm[i] += ma_e as i32 * isum_e + ma_o as i32 * isum_o;
                }
            }
            // ── 2 hadd per row per block (vs 8 in eager version) ──────────────────
            for i in 0..8 {
                let sd_i = hadd_i32_ymm(acc_lo[i]) + hadd_i32_ymm(acc_hi[i]);
                sumf[i] += q8.d[b] * (d_arr[i][b] * sd_i as f32 - dmin_arr[i][b] * sm[i] as f32);
            }
        }
        o0[r] = sumf[0];
        o1[r] = sumf[1];
        o2[r] = sumf[2];
        o3[r] = sumf[3];
        o4[r] = sumf[4];
        o5[r] = sumf[5];
        o6[r] = sumf[6];
        o7[r] = sumf[7];
    }
}

/// Batch Q4_K 8-row tile: `outs[i][r] = vec_dot_q4k(rows[i], &q8s[r], in_f)` for all i,r.
/// Bit-identical to the single-token kernel. On AVX-512BW machines the Q8 activation is
/// loaded once per (block, nibble-pair) and dotted against all 8 weight rows — 8× activation
/// reuse over single-row, 4× over 2-row. Falls back to 8× `vec_dot_q4k_batch` on older CPUs.
fn vec_dot_q4k_batch8(rows: [&[u8]; 8], q8s: &[Q8], in_f: usize, outs: [&mut [f32]; 8]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512bw") {
        return unsafe { vec_dot_q4k_batch8_avx512bw(rows, q8s, in_f, outs) };
    }
    // Fallback: call the per-row batch kernel (avx2/scalar dispatch) for each of the 8 rows.
    let [row0, row1, row2, row3, row4, row5, row6, row7] = rows;
    let [out0, out1, out2, out3, out4, out5, out6, out7] = outs;
    vec_dot_q4k_batch(row0, q8s, in_f, out0);
    vec_dot_q4k_batch(row1, q8s, in_f, out1);
    vec_dot_q4k_batch(row2, q8s, in_f, out2);
    vec_dot_q4k_batch(row3, q8s, in_f, out3);
    vec_dot_q4k_batch(row4, q8s, in_f, out4);
    vec_dot_q4k_batch(row5, q8s, in_f, out5);
    vec_dot_q4k_batch(row6, q8s, in_f, out6);
    vec_dot_q4k_batch(row7, q8s, in_f, out7);
}

// ── Q6_K batch ────────────────────────────────────────────────────────────────

fn vec_dot_q6k_batch_scalar(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let nb = in_f / 256;

    // Pre-expand Q6 nibbles + high bits into flat[b*256..b*256+256] (values 0–63).
    // Layout: flat[b*256 + half*128 + ci*32 .. +32] = q6 column ci for that half
    // (matches the AVX2 single-token column layout; also usable by scalar per sub-block).
    let mut d_arr = vec![0f32; nb];
    let mut scales_arr = vec![0i8; nb * 16];
    let mut q6_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        d_arr[b] = rdf16(&blk[208..210]);
        for i in 0..16 {
            scales_arr[b * 16 + i] = scales[i] as i8;
        }
        let flat = &mut q6_flat[b * 256..b * 256 + 256];
        for half in 0..2usize {
            let (qlo, qho) = (half * 64, half * 32);
            for l in 0..32usize {
                let q1 = (ql[qlo + l] & 0xF) | ((qh[qho + l] & 3) << 4);
                let q2 = (ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4);
                let q3 = (ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4);
                let q4 = (ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4);
                // column-first layout: ci=0→col0(q1), ci=1→col1(q2), ci=2→col2(q3), ci=3→col3(q4)
                flat[half * 128 + l] = q1;
                flat[half * 128 + 32 + l] = q2;
                flat[half * 128 + 64 + l] = q3;
                flat[half * 128 + 96 + l] = q4;
            }
        }
    }

    // Per-token: accumulate sumi/bsum per sub-block, apply int8 scales.
    // sub_off(sub) = (sub/8)*128 + ((sub%8)/2)*32 + ((sub%8)%2)*16
    // This matches both flat q6 and q8.qs layout (same 16-element stride).
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let mut sumi = [0i32; 16];
            let mut bsum = [0i32; 16];
            for sub in 0..16usize {
                let sub_off = (sub / 8) * 128 + ((sub % 8) / 2) * 32 + ((sub % 8) % 2) * 16;
                let q6_ptr = &q6_flat[b * 256 + sub_off..b * 256 + sub_off + 16];
                let q8_ptr = &q8.qs[b * 256 + sub_off..b * 256 + sub_off + 16];
                for i in 0..16 {
                    let q6v = q6_ptr[i] as i32;
                    let v = q8_ptr[i] as i32;
                    sumi[sub] += q6v * v;
                    bsum[sub] += v;
                }
            }
            let mut s = 0f32;
            for sub in 0..16 {
                s += scales_arr[b * 16 + sub] as f32 * (sumi[sub] - 32 * bsum[sub]) as f32;
            }
            sumf += d_arr[b] * q8.d[b] * s;
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q6k_batch_avx2(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;

    let mask_0f = _mm256_set1_epi8(0x0F_u8 as i8);
    let mask_30 = _mm256_set1_epi8(0x30_u8 as i8);
    let mask_03 = _mm256_set1_epi8(0x03_u8 as i8);
    let ones_u8 = _mm256_set1_epi8(1i8);
    let ones_i16 = _mm256_set1_epi16(1i16);

    // Pre-expand all Q6 nibbles using AVX2 shifts (same ops as single-token, done once).
    let mut d_arr = vec![0f32; nb];
    let mut scales_arr = vec![0i8; nb * 16];
    let mut q6_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        d_arr[b] = rdf16(&blk[208..210]);
        for i in 0..16 {
            scales_arr[b * 16 + i] = scales[i] as i8;
        }
        let flat = &mut q6_flat[b * 256..b * 256 + 256];

        for half in 0..2usize {
            let qlo = half * 64;
            let qho = half * 32;

            let qh_ymm = _mm256_loadu_si256(qh[qho..].as_ptr() as *const __m256i);
            let ql_lo = _mm256_loadu_si256(ql[qlo..].as_ptr() as *const __m256i);
            let ql_hi = _mm256_loadu_si256(ql[qlo + 32..].as_ptr() as *const __m256i);

            let qh_sr2 = _mm256_and_si256(_mm256_srli_epi16(qh_ymm, 2), mask_03);

            let q6_c0 = _mm256_or_si256(
                _mm256_and_si256(ql_lo, mask_0f),
                _mm256_slli_epi16(_mm256_and_si256(qh_ymm, mask_03), 4),
            );
            let q6_c2 = _mm256_or_si256(
                _mm256_and_si256(ql_hi, mask_0f),
                _mm256_slli_epi16(qh_sr2, 4),
            );
            let q6_c4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_lo, 4), mask_0f),
                _mm256_and_si256(qh_ymm, mask_30),
            );
            let q6_c6 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_hi, 4), mask_0f),
                _mm256_and_si256(_mm256_srli_epi16(qh_ymm, 2), mask_30),
            );

            // Store columns contiguously: flat[half*128 + ci*32..+32]
            _mm256_storeu_si256(flat[half * 128..].as_mut_ptr() as *mut __m256i, q6_c0);
            _mm256_storeu_si256(flat[half * 128 + 32..].as_mut_ptr() as *mut __m256i, q6_c2);
            _mm256_storeu_si256(flat[half * 128 + 64..].as_mut_ptr() as *mut __m256i, q6_c4);
            _mm256_storeu_si256(flat[half * 128 + 96..].as_mut_ptr() as *mut __m256i, q6_c6);
        }
    }

    // Per-token: dot each column (32 elements) with q8, compute bsum simultaneously.
    // Identical column/scale accumulation order as single-token AVX2 Q6K → bit-identical.
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let d = d_arr[b];
            let flat = &q6_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];

            let mut s = 0f32;
            for half in 0..2usize {
                let sco = half * 8;
                let base = half * 128;

                for ci in 0..4usize {
                    let q6_ymm =
                        _mm256_loadu_si256(flat[half * 128 + ci * 32..].as_ptr() as *const __m256i);
                    let q8_ymm =
                        _mm256_loadu_si256(q8b[base + ci * 32..].as_ptr() as *const __m256i);

                    let prod = _mm256_maddubs_epi16(q6_ymm, q8_ymm);
                    let sum32 = _mm256_madd_epi16(prod, ones_i16);
                    let bsum_i16 = _mm256_maddubs_epi16(ones_u8, q8_ymm);
                    let bsum_i32 = _mm256_madd_epi16(bsum_i16, ones_i16);

                    let sum_lo = _mm256_castsi256_si128(sum32);
                    let sum_hi = _mm256_extracti128_si256::<1>(sum32);
                    let bs_lo = _mm256_castsi256_si128(bsum_i32);
                    let bs_hi = _mm256_extracti128_si256::<1>(bsum_i32);

                    let iprod_0 = hadd_i32_xmm(sum_lo);
                    let iprod_1 = hadd_i32_xmm(sum_hi);
                    let bs_0 = hadd_i32_xmm(bs_lo);
                    let bs_1 = hadd_i32_xmm(bs_hi);

                    let sub_0 = sco + ci * 2;
                    let sub_1 = sco + ci * 2 + 1;
                    s += scales_arr[b * 16 + sub_0] as f32 * (iprod_0 - 32 * bs_0) as f32;
                    s += scales_arr[b * 16 + sub_1] as f32 * (iprod_1 - 32 * bs_1) as f32;
                }
            }
            sumf += d * q8.d[b] * s;
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn vec_dot_q6k_batch_avx512bw(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;

    let mask_0f_z = _mm512_set1_epi8(0x0F_u8 as i8);
    let mask_30_z = _mm512_set1_epi8(0x30_u8 as i8);
    let mask_03_z = _mm512_set1_epi8(0x03_u8 as i8);
    let ones_u8_z = _mm512_set1_epi8(1i8);
    let ones_i16_z = _mm512_set1_epi16(1i16);

    // Pre-expand Q6 both halves simultaneously via zmm, store to q6_flat.
    // flat[b*256 + half*128 + ci*32..+32] = q6 column ci for half.
    let mut d_arr = vec![0f32; nb];
    let mut scales_arr = vec![0i8; nb * 16];
    let mut q6_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        d_arr[b] = rdf16(&blk[208..210]);
        for i in 0..16 {
            scales_arr[b * 16 + i] = scales[i] as i8;
        }
        let flat = &mut q6_flat[b * 256..b * 256 + 256];

        // qh is 64 contiguous bytes for both halves → single zmm load.
        let qh_z = _mm512_loadu_si512(qh.as_ptr() as *const __m512i);

        let ql_lo_h0 = _mm256_loadu_si256(ql[0..].as_ptr() as *const __m256i);
        let ql_lo_h1 = _mm256_loadu_si256(ql[64..].as_ptr() as *const __m256i);
        let ql_lo_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ql_lo_h0), ql_lo_h1);
        let ql_hi_h0 = _mm256_loadu_si256(ql[32..].as_ptr() as *const __m256i);
        let ql_hi_h1 = _mm256_loadu_si256(ql[96..].as_ptr() as *const __m256i);
        let ql_hi_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ql_hi_h0), ql_hi_h1);

        let qh_sr2_z = _mm512_and_si512(_mm512_srli_epi16(qh_z, 2), mask_03_z);

        let q6_c0_z = _mm512_or_si512(
            _mm512_and_si512(ql_lo_z, mask_0f_z),
            _mm512_slli_epi16(_mm512_and_si512(qh_z, mask_03_z), 4),
        );
        let q6_c2_z = _mm512_or_si512(
            _mm512_and_si512(ql_hi_z, mask_0f_z),
            _mm512_slli_epi16(qh_sr2_z, 4),
        );
        let q6_c4_z = _mm512_or_si512(
            _mm512_and_si512(_mm512_srli_epi16(ql_lo_z, 4), mask_0f_z),
            _mm512_and_si512(qh_z, mask_30_z),
        );
        let q6_c6_z = _mm512_or_si512(
            _mm512_and_si512(_mm512_srli_epi16(ql_hi_z, 4), mask_0f_z),
            _mm512_and_si512(_mm512_srli_epi16(qh_z, 2), mask_30_z),
        );

        // Each zmm has h0 in lower 256 and h1 in upper 256.
        // Store as [h0(ci), h1(ci)] per column → deinterleave into two 32-byte stores.
        for (ci, q6_z) in [q6_c0_z, q6_c2_z, q6_c4_z, q6_c6_z].iter().enumerate() {
            let h0 = _mm512_castsi512_si256(*q6_z);
            let h1 = _mm512_extracti64x4_epi64::<1>(*q6_z);
            _mm256_storeu_si256(flat[ci * 32..].as_mut_ptr() as *mut __m256i, h0);
            _mm256_storeu_si256(flat[128 + ci * 32..].as_mut_ptr() as *mut __m256i, h1);
        }
    }

    // Per-token: two halves merged into zmm for 4 columns; split back per scale accumulation.
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let mut simd_sumi = [0i32; 16];
            let mut simd_bsum = [0i32; 16];
            let flat = &q6_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];

            for ci in 0..4usize {
                // Load h0 and h1 columns into zmm.
                let q6_h0 = _mm256_loadu_si256(flat[ci * 32..].as_ptr() as *const __m256i);
                let q6_h1 = _mm256_loadu_si256(flat[128 + ci * 32..].as_ptr() as *const __m256i);
                let q6_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q6_h0), q6_h1);

                let q8_h0 = _mm256_loadu_si256(q8b[ci * 32..].as_ptr() as *const __m256i);
                let q8_h1 = _mm256_loadu_si256(q8b[128 + ci * 32..].as_ptr() as *const __m256i);
                let q8_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q8_h0), q8_h1);

                let prod = _mm512_maddubs_epi16(q6_z, q8_z);
                let sum32_z = _mm512_madd_epi16(prod, ones_i16_z);
                let bsum_i16_z = _mm512_maddubs_epi16(ones_u8_z, q8_z);
                let bsum_i32_z = _mm512_madd_epi16(bsum_i16_z, ones_i16_z);

                let sum_h0 = _mm512_castsi512_si256(sum32_z);
                let sum_h1 = _mm512_extracti64x4_epi64::<1>(sum32_z);
                let bsum_h0 = _mm512_castsi512_si256(bsum_i32_z);
                let bsum_h1 = _mm512_extracti64x4_epi64::<1>(bsum_i32_z);

                let s_h0_lo = _mm256_castsi256_si128(sum_h0);
                let s_h0_hi = _mm256_extracti128_si256::<1>(sum_h0);
                let s_h1_lo = _mm256_castsi256_si128(sum_h1);
                let s_h1_hi = _mm256_extracti128_si256::<1>(sum_h1);
                let b_h0_lo = _mm256_castsi256_si128(bsum_h0);
                let b_h0_hi = _mm256_extracti128_si256::<1>(bsum_h0);
                let b_h1_lo = _mm256_castsi256_si128(bsum_h1);
                let b_h1_hi = _mm256_extracti128_si256::<1>(bsum_h1);

                simd_sumi[ci * 2] = hadd_i32_xmm(s_h0_lo);
                simd_sumi[ci * 2 + 1] = hadd_i32_xmm(s_h0_hi);
                simd_sumi[8 + ci * 2] = hadd_i32_xmm(s_h1_lo);
                simd_sumi[8 + ci * 2 + 1] = hadd_i32_xmm(s_h1_hi);

                simd_bsum[ci * 2] = hadd_i32_xmm(b_h0_lo);
                simd_bsum[ci * 2 + 1] = hadd_i32_xmm(b_h0_hi);
                simd_bsum[8 + ci * 2] = hadd_i32_xmm(b_h1_lo);
                simd_bsum[8 + ci * 2 + 1] = hadd_i32_xmm(b_h1_hi);
            }

            let mut s = 0f32;
            for sub in 0..16 {
                s +=
                    scales_arr[b * 16 + sub] as f32 * (simd_sumi[sub] - 32 * simd_bsum[sub]) as f32;
            }
            sumf += d_arr[b] * q8.d[b] * s;
        }
        out[r] = sumf;
    }
}

fn vec_dot_q6k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q6k_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q6k_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q6k_batch_scalar(row, q8s, in_f, out);
}

// ── Q8_0 batch ────────────────────────────────────────────────────────────────

fn vec_dot_q8_0_batch_scalar(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let nb_super = in_f / 256;
    let bpr = 34usize;

    // Pre-read weight scales and i8 values (no nibble extraction needed for Q8_0).
    let mut dw_arr = vec![0f32; nb_super * 8];
    let mut qw_flat = vec![0i8; nb_super * 256]; // raw i8 weight bytes

    for b in 0..nb_super {
        for s in 0..8usize {
            let wb = b * 8 + s;
            let blk = &row[wb * bpr..wb * bpr + bpr];
            dw_arr[b * 8 + s] = rdf16(&blk[0..2]);
            let src = &blk[2..34];
            let dst = &mut qw_flat[b * 256 + s * 32..b * 256 + s * 32 + 32];
            for i in 0..32 {
                dst[i] = src[i] as i8;
            }
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb_super {
            let d8 = q8.d[b];
            for s in 0..8usize {
                let qw = &qw_flat[b * 256 + s * 32..b * 256 + s * 32 + 32];
                let qx = &q8.qs[b * 256 + s * 32..b * 256 + s * 32 + 32];
                let iprod: i32 = (0..32).map(|i| qw[i] as i32 * qx[i] as i32).sum();
                sumf += d8 * dw_arr[b * 8 + s] * iprod as f32;
            }
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q8_0_batch_avx2(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb_super = in_f / 256;
    let bpr = 34usize;
    let ones = _mm256_set1_epi16(1i16);

    // Pre-read weight scales + abs(qw) (u8) + sign bytes (for _mm256_sign_epi8 per token).
    // Storing qw as raw i8 and applying abs+sign per token avoids two extra arrays.
    let mut dw_arr = vec![0f32; nb_super * 8];
    let mut qw_flat = vec![0u8; nb_super * 256]; // i8 as u8 bytes, to be cast per-kernel call

    for b in 0..nb_super {
        for s in 0..8usize {
            let wb = b * 8 + s;
            let blk = &row[wb * bpr..wb * bpr + bpr];
            dw_arr[b * 8 + s] = rdf16(&blk[0..2]);
            let src = &blk[2..34];
            qw_flat[b * 256 + s * 32..b * 256 + s * 32 + 32].copy_from_slice(src);
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb_super {
            let d8 = q8.d[b];
            for s in 0..8usize {
                let qw = _mm256_loadu_si256(qw_flat[b * 256 + s * 32..].as_ptr() as *const __m256i);
                let qx = _mm256_loadu_si256(q8.qs[b * 256 + s * 32..].as_ptr() as *const __m256i);
                let qw_abs = _mm256_abs_epi8(qw);
                let qx_signed = _mm256_sign_epi8(qx, qw);
                let prod = _mm256_maddubs_epi16(qw_abs, qx_signed);
                let sum32 = _mm256_madd_epi16(prod, ones);
                let iprod = hadd_i32_ymm(sum32);
                sumf += d8 * dw_arr[b * 8 + s] * iprod as f32;
            }
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn vec_dot_q8_0_batch_avx512bw(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb_super = in_f / 256;
    let bpr = 34usize;
    let ones512 = _mm512_set1_epi16(1i16);

    let mut dw_arr = vec![0f32; nb_super * 8];
    let mut qw_flat = vec![0u8; nb_super * 256];

    for b in 0..nb_super {
        for s in 0..8usize {
            let wb = b * 8 + s;
            let blk = &row[wb * bpr..wb * bpr + bpr];
            dw_arr[b * 8 + s] = rdf16(&blk[0..2]);
            qw_flat[b * 256 + s * 32..b * 256 + s * 32 + 32].copy_from_slice(&blk[2..34]);
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb_super {
            let d8 = q8.d[b];
            // Process pairs of sub-blocks with zmm (identical to single-token avx512bw).
            for k in 0..4usize {
                let s0 = 2 * k;
                let s1 = 2 * k + 1;
                let qw0 =
                    _mm256_loadu_si256(qw_flat[b * 256 + s0 * 32..].as_ptr() as *const __m256i);
                let qw1 =
                    _mm256_loadu_si256(qw_flat[b * 256 + s1 * 32..].as_ptr() as *const __m256i);
                // Load 64 contiguous activation bytes
                let qx_z =
                    _mm512_loadu_si512(q8.qs[b * 256 + s0 * 32..].as_ptr() as *const __m512i);
                let qx0 = _mm512_castsi512_si256(qx_z);
                let qx1 = _mm512_extracti64x4_epi64::<1>(qx_z);
                let qw_abs0 = _mm256_abs_epi8(qw0);
                let qw_abs1 = _mm256_abs_epi8(qw1);
                let qx_s0 = _mm256_sign_epi8(qx0, qw0);
                let qx_s1 = _mm256_sign_epi8(qx1, qw1);
                let qw_a_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qw_abs0), qw_abs1);
                let qx_s_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qx_s0), qx_s1);
                let prod = _mm512_maddubs_epi16(qw_a_z, qx_s_z);
                let sum32_z = _mm512_madd_epi16(prod, ones512);
                let lo_ymm = _mm512_castsi512_si256(sum32_z);
                let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
                let iprod0 = hadd_i32_ymm(lo_ymm);
                let iprod1 = hadd_i32_ymm(hi_ymm);
                sumf +=
                    d8 * (dw_arr[b * 8 + s0] * iprod0 as f32 + dw_arr[b * 8 + s1] * iprod1 as f32);
            }
        }
        out[r] = sumf;
    }
}

fn vec_dot_q8_0_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q8_0_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q8_0_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q8_0_batch_scalar(row, q8s, in_f, out);
}

// ── Q5_K batch ────────────────────────────────────────────────────────────────

fn vec_dot_q5k_batch_scalar(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let nb = in_f / 256;

    // Pre-expand Q5 values (0–31) into flat[b*256..b*256+256].
    // Layout: flat[b*256 + s*32 .. +32] = q5 for sub-block s (same as Q4K flat layout).
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![0u32; nb * 8];
    let mut m_arr = vec![0u32; nb * 8];
    let mut q5_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 176..b * 176 + 176];
        d_arr[b] = rdf16(&blk[0..2]);
        dmin_arr[b] = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in 0..4usize {
            let (sc_e, m_e) = k4(2 * j, scales);
            let (sc_o, m_o) = k4(2 * j + 1, scales);
            sc_arr[b * 8 + 2 * j] = sc_e;
            m_arr[b * 8 + 2 * j] = m_e;
            sc_arr[b * 8 + 2 * j + 1] = sc_o;
            m_arr[b * 8 + 2 * j + 1] = m_o;
            let base_e = b * 256 + (2 * j) * 32;
            let base_o = b * 256 + (2 * j + 1) * 32;
            for l in 0..32 {
                let v = ql[j * 32 + l];
                q5_flat[base_e + l] = (v & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 };
                q5_flat[base_o + l] = (v >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            for j in 0..4usize {
                let mut iprod_e = 0i32;
                let mut iprod_o = 0i32;
                let fe = &q5_flat[b * 256 + (2 * j) * 32..b * 256 + (2 * j) * 32 + 32];
                let fo = &q5_flat[b * 256 + (2 * j + 1) * 32..b * 256 + (2 * j + 1) * 32 + 32];
                let q8e = &q8.qs[b * 256 + (2 * j) * 32..b * 256 + (2 * j) * 32 + 32];
                let q8o = &q8.qs[b * 256 + (2 * j + 1) * 32..b * 256 + (2 * j + 1) * 32 + 32];
                for l in 0..32 {
                    iprod_e += fe[l] as i32 * q8e[l] as i32;
                    iprod_o += fo[l] as i32 * q8o[l] as i32;
                }
                sd += sc_arr[b * 8 + 2 * j] as i32 * iprod_e
                    + sc_arr[b * 8 + 2 * j + 1] as i32 * iprod_o;
                sm += m_arr[b * 8 + 2 * j] as i32 * q8.bsums[b * 8 + 2 * j]
                    + m_arr[b * 8 + 2 * j + 1] as i32 * q8.bsums[b * 8 + 2 * j + 1];
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q5k_batch_avx2(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let sixteen = _mm256_set1_epi8(0x10_u8 as i8);
    let ones = _mm256_set1_epi16(1i16);
    let zero = _mm256_setzero_si256();

    // Pre-expand q5 values (0–31) using the same AVX2 logic as the single-token kernel.
    // flat[b*256 + k*64..+32] = even sub-block 2k (lo nibble + high bit),
    // flat[b*256 + k*64+32..+32] = odd sub-block 2k+1 (hi nibble + high bit).
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![[0u32; 8]; nb];
    let mut m_arr = vec![[0u32; 8]; nb];
    let mut q5_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 176..b * 176 + 176];
        d_arr[b] = rdf16(&blk[0..2]);
        dmin_arr[b] = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales);
            sc_arr[b][s] = sc;
            m_arr[b][s] = mv;
        }
        let qh_ymm = _mm256_loadu_si256(qh.as_ptr() as *const __m256i);
        let flat = &mut q5_flat[b * 256..b * 256 + 256];
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for k in 0..4usize {
            let nibbles = _mm256_loadu_si256(ql[k * 32..].as_ptr() as *const __m256i);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo);
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo);
            let u1v = _mm256_set1_epi8(u1 as i8);
            let u2v = _mm256_set1_epi8(u2 as i8);
            let has_e = _mm256_and_si256(qh_ymm, u1v);
            let has_o = _mm256_and_si256(qh_ymm, u2v);
            let high_e = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_e, zero), sixteen);
            let high_o = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_o, zero), sixteen);
            let q5_e = _mm256_or_si256(lo_nib, high_e);
            let q5_o = _mm256_or_si256(hi_nib, high_o);
            _mm256_storeu_si256(flat[k * 64..].as_mut_ptr() as *mut __m256i, q5_e);
            _mm256_storeu_si256(flat[k * 64 + 32..].as_mut_ptr() as *mut __m256i, q5_o);
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            let flat = &q5_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for j in 0..4usize {
                let (sc_e, m_e) = (sc_arr[b][2 * j], m_arr[b][2 * j]);
                let (sc_o, m_o) = (sc_arr[b][2 * j + 1], m_arr[b][2 * j + 1]);
                let q5_e = _mm256_loadu_si256(flat[j * 64..].as_ptr() as *const __m256i);
                let q5_o = _mm256_loadu_si256(flat[j * 64 + 32..].as_ptr() as *const __m256i);
                let q8_e = _mm256_loadu_si256(q8b[2 * j * 32..].as_ptr() as *const __m256i);
                let q8_o = _mm256_loadu_si256(q8b[(2 * j + 1) * 32..].as_ptr() as *const __m256i);
                let prod_e = _mm256_maddubs_epi16(q5_e, q8_e);
                let sum32_e = _mm256_madd_epi16(prod_e, ones);
                let iprod_e = hadd_i32_ymm(sum32_e);
                let prod_o = _mm256_maddubs_epi16(q5_o, q8_o);
                let sum32_o = _mm256_madd_epi16(prod_o, ones);
                let iprod_o = hadd_i32_ymm(sum32_o);
                let isum_e = q8.bsums[b * 8 + 2 * j];
                let isum_o = q8.bsums[b * 8 + 2 * j + 1];
                sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
                sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn vec_dot_q5k_batch_avx512bw(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let sixteen = _mm256_set1_epi8(0x10_u8 as i8);
    let ones512 = _mm512_set1_epi16(1i16);
    let zero256 = _mm256_setzero_si256();

    // Pre-expand q5: same flat layout as AVX2 batch (k*64 = [even 32B, odd 32B]).
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![[0u32; 8]; nb];
    let mut m_arr = vec![[0u32; 8]; nb];
    let mut q5_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 176..b * 176 + 176];
        d_arr[b] = rdf16(&blk[0..2]);
        dmin_arr[b] = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales);
            sc_arr[b][s] = sc;
            m_arr[b][s] = mv;
        }
        let qh_ymm = _mm256_loadu_si256(qh.as_ptr() as *const __m256i);
        let flat = &mut q5_flat[b * 256..b * 256 + 256];
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for k in 0..4usize {
            let nibbles = _mm256_loadu_si256(ql[k * 32..].as_ptr() as *const __m256i);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo);
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo);
            let u1v = _mm256_set1_epi8(u1 as i8);
            let u2v = _mm256_set1_epi8(u2 as i8);
            let has_e = _mm256_and_si256(qh_ymm, u1v);
            let has_o = _mm256_and_si256(qh_ymm, u2v);
            let high_e = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_e, zero256), sixteen);
            let high_o = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_o, zero256), sixteen);
            let q5_e = _mm256_or_si256(lo_nib, high_e);
            let q5_o = _mm256_or_si256(hi_nib, high_o);
            _mm256_storeu_si256(flat[k * 64..].as_mut_ptr() as *mut __m256i, q5_e);
            _mm256_storeu_si256(flat[k * 64 + 32..].as_mut_ptr() as *mut __m256i, q5_o);
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            let flat = &q5_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for k in 0..4usize {
                let (sc_e, m_e) = (sc_arr[b][2 * k], m_arr[b][2 * k]);
                let (sc_o, m_o) = (sc_arr[b][2 * k + 1], m_arr[b][2 * k + 1]);
                // flat[k*64..+64]: lower 32B = even sub-block, upper 32B = odd
                let q5_zmm = _mm512_loadu_si512(flat[k * 64..].as_ptr() as *const __m512i);
                let q8_zmm = _mm512_loadu_si512(q8b[k * 64..].as_ptr() as *const __m512i);
                let prod = _mm512_maddubs_epi16(q5_zmm, q8_zmm);
                let sum32_z = _mm512_madd_epi16(prod, ones512);
                let lo_ymm = _mm512_castsi512_si256(sum32_z);
                let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
                let iprod_e = hadd_i32_ymm(lo_ymm);
                let iprod_o = hadd_i32_ymm(hi_ymm);
                let isum_e = q8.bsums[b * 8 + 2 * k];
                let isum_o = q8.bsums[b * 8 + 2 * k + 1];
                sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
                sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

fn vec_dot_q5k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q5k_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q5k_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q5k_batch_scalar(row, q8s, in_f, out);
}

/// `Σ f16_weight·x` (weight is 2 bytes/elem). `target-cpu=native` lowers the f16→f32 to F16C.
fn dot_f16(w: &[u8], x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n = x.len();
    let chunks = n / 8;
    for c in 0..chunks {
        for (j, ac) in acc.iter_mut().enumerate() {
            let i = c * 8 + j;
            let wv = half::f16::from_le_bytes([w[i * 2], w[i * 2 + 1]]).to_f32();
            *ac += wv * x[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..n {
        s += half::f16::from_le_bytes([w[i * 2], w[i * 2 + 1]]).to_f32() * x[i];
    }
    s
}

/// `Σ bf16_weight·x` (bf16 = top 16 bits of f32).
fn dot_bf16(w: &[u8], x: &[f32]) -> f32 {
    let mut s = 0f32;
    for (i, &xi) in x.iter().enumerate() {
        let wv = f32::from_bits((u16::from_le_bytes([w[i * 2], w[i * 2 + 1]]) as u32) << 16);
        s += wv * xi;
    }
    s
}

/// Dot product with 8 independent accumulators so the reduction isn't latency-bound — lets the
/// autovectorizer (with `target-cpu=native`) keep several AVX FMA lanes in flight. `a`/`b` equal len.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc = [0f32; 8];
    for c in 0..chunks {
        let base = c * 8;
        for (j, ac) in acc.iter_mut().enumerate() {
            *ac += a[base + j] * b[base + j];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// A host buffer. Weights are **mapped** — a zero-copy [`TensorBytes`] view straight into the GGUF
/// mmap (read-only, no `memcpy`, no owned RAM). Everything the model writes (KV / conv / recurrent
/// state, per-step IO) is **owned** — a plain byte vec behind a `Mutex` (so `&dyn Buffer` stays
/// `Send + Sync` and writes are safe). `&dyn Buffer` reads go through [`CpuBuffer::read`].
pub enum CpuBuffer {
    Owned(Mutex<Vec<u8>>),
    Mapped(TensorBytes),
}

/// A uniform read view over either storage (a `MutexGuard` for owned, the slice for mapped); both
/// deref to `[u8]`.
enum CpuRead<'a> {
    Owned(std::sync::MutexGuard<'a, Vec<u8>>),
    Mapped(&'a TensorBytes),
}

impl std::ops::Deref for CpuRead<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            CpuRead::Owned(g) => g,
            CpuRead::Mapped(t) => t,
        }
    }
}

impl CpuBuffer {
    /// Read view of the bytes (zero-copy for mapped weights; mutex guard for owned buffers).
    fn read(&self) -> CpuRead<'_> {
        match self {
            CpuBuffer::Owned(m) => CpuRead::Owned(m.lock().unwrap()),
            CpuBuffer::Mapped(t) => CpuRead::Mapped(t),
        }
    }
    /// Mutable owned storage; panics for mapped (read-only) weights.
    fn owned(&self) -> std::sync::MutexGuard<'_, Vec<u8>> {
        match self {
            CpuBuffer::Owned(m) => m.lock().unwrap(),
            CpuBuffer::Mapped(_) => {
                panic!("cpu backend: write to a mapped (read-only) weight buffer")
            }
        }
    }
}

impl Buffer for CpuBuffer {
    fn len_bytes(&self) -> usize {
        match self {
            CpuBuffer::Owned(m) => m.lock().unwrap().len(),
            CpuBuffer::Mapped(t) => t.len(),
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Default)]
pub struct CpuBackend {
    /// Dequantized-weight cache keyed by the bound buffer's address (weights are bound the same
    /// every step, so dequant once and reuse). Only the small norm weights (`RmsNorm` / `QkNorm`)
    /// land here — the large `Op::Linear` weights are streamed row-by-row instead (see that arm),
    /// so this never holds the whole model in f32.
    weight_cache: Mutex<HashMap<usize, Arc<Vec<f32>>>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Wrap a zero-copy GGUF mmap view as a read-only weight buffer (no allocation, no `memcpy`).
    pub fn map_weight(&self, bytes: TensorBytes) -> Box<dyn Buffer> {
        Box::new(CpuBuffer::Mapped(bytes))
    }
}

/// Reinterpret raw buffer bytes as `f32` values per `dtype` (dequantizing quant/f16/bf16, widening
/// integer position tensors). The universal "read a tensor's value on the host".
fn bytes_to_f32(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        DType::I32 => bytemuck::cast_slice::<u8, i32>(bytes)
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DType::U32 => bytemuck::cast_slice::<u8, u32>(bytes)
            .iter()
            .map(|&v| v as f32)
            .collect(),
        // F16 / Bf16 / all quant + codebook types go through the shared host dequant.
        other => dequant_block(other, bytes).expect("cpu backend: host dequant"),
    }
}

/// Gated-FFN activation applied to the gate value.
fn act_fn(act: Activation, g: f32) -> f32 {
    match act {
        Activation::Silu => g / (1.0 + (-g).exp()),
        // gelu_pytorch_tanh: 0.5 g (1 + tanh(√(2/π)·(g + 0.044715 g³)))
        Activation::Gelu => 0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()),
        Activation::Sigmoid => 1.0 / (1.0 + (-g).exp()),
    }
}

fn cpu_buf(b: &dyn Buffer) -> &CpuBuffer {
    b.as_any()
        .downcast_ref::<CpuBuffer>()
        .expect("cpu backend: buffer is not a CpuBuffer (mixed backends?)")
}

/// Dequantize the first `need` elements of a Q8_0-block buffer (34 B / 32 elems, y = d*q).
pub(crate) fn dequant_prefix_q8_0(bytes: &[u8], need: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(need);
    for b in 0..need.div_ceil(32) {
        let off = b * 34;
        let d = half::f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        for i in 0..32 {
            if out.len() == need {
                break;
            }
            out.push(d * (bytes[off + 2 + i] as i8) as f32);
        }
    }
    out
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            name: "cpu-reference".into(),
            f16: true,
            cooperative_matrix: false,
            max_buffer_bytes: u64::MAX,
            max_shared_memory_bytes: u32::MAX, // scalar interpreter: no shared-memory tiling
            unified_memory: true,
            // The interpreter reads the baked `pos`/`kv_len` from the graph ops, so the decode graph
            // must be rebuilt per token — no record-once replay.
            decode_replay: false,
            combined_gu: false,
        }
    }

    fn alloc(&self, bytes: usize, _usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        Ok(Box::new(CpuBuffer::Owned(Mutex::new(vec![
            0u8;
            bytes.max(4)
        ]))))
    }

    fn alloc_uninit(&self, bytes: usize, _usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        // Debug: poison with 0xFF (= NaN as f32) so a read-before-write surfaces loudly in the CPU
        // tests/oracle instead of silently working. Release: the Vec is zeroed anyway (no CPU perf
        // win to skip it), so stay safe.
        let fill = if cfg!(debug_assertions) { 0xFFu8 } else { 0u8 };
        Ok(Box::new(CpuBuffer::Owned(Mutex::new(vec![
            fill;
            bytes.max(4)
        ]))))
    }

    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        let mut d = cpu_buf(dst).owned();
        d[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        let s = cpu_buf(src).read();
        dst.copy_from_slice(&s[..dst.len()]);
        Ok(())
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        Ok(GraphPlan::boxed(graph))
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        let g = &plan
            .as_any()
            .downcast_ref::<GraphPlan>()
            .expect("cpu backend: plan is not a GraphPlan")
            .graph;

        // f32 working store for every Input/Internal/Output handle (weights are read on demand:
        // norms via the small dequant cache, `Op::Linear` weights streamed row-by-row).
        let mut vals: Vec<Vec<f32>> = vec![Vec::new(); g.tensors.len()];
        // KV-cache tensors (the `cache` of `WriteKv`, the `k_cache`/`v_cache` of `Attention`) are
        // accessed straight from their bound buffers — `WriteKv` writes one row, `Attention` reads
        // `kv_len` rows. They're sized for the WHOLE context (`max_ctx`), so loading them into `vals`
        // (and writing them back) each token would cost O(max_ctx) memory traffic per token instead of
        // O(kv_len) — catastrophic at a large `max_new`. Skip the round-trip for them. Which tensors
        // are written in place is graph semantics, computed once by `Graph::in_place_inputs`.
        let direct = g.in_place_inputs();
        for (i, decl) in g.tensors.iter().enumerate() {
            match decl.kind {
                TensorKind::Internal | TensorKind::Output => {
                    vals[i] = vec![0f32; decl.desc.numel()]
                }
                TensorKind::Input if direct.contains(&TensorId(i as u32)) => {} // read/written in place
                TensorKind::Input => {
                    let buf = bindings
                        .get(TensorId(i as u32))
                        .expect("cpu backend: unbound Input");
                    let bytes = cpu_buf(buf).read();
                    vals[i] = bytes_to_f32(&bytes, decl.desc.dtype);
                }
                TensorKind::Weight => {} // lazily dequantized in `weight()`
            }
        }

        // Fetch a (cached) dequantized weight.
        let weight = |id: TensorId| -> Arc<Vec<f32>> {
            let buf = bindings.get(id).expect("cpu backend: unbound Weight");
            let key = cpu_buf(buf) as *const CpuBuffer as usize;
            if let Some(w) = self.weight_cache.lock().unwrap().get(&key) {
                return w.clone();
            }
            let bytes = cpu_buf(buf).read();
            let w = Arc::new(bytes_to_f32(&bytes, g.desc(id).dtype));
            self.weight_cache.lock().unwrap().insert(key, w.clone());
            w
        };

        let prof_ops = std::env::var("INFR_PROF_OPS").is_ok();
        let mut op_times: HashMap<&'static str, f64> = HashMap::new();
        for op in &g.ops {
            let __t0 = if prof_ops {
                Some(std::time::Instant::now())
            } else {
                None
            };
            match *op {
                Op::RmsNorm {
                    x,
                    weight: w,
                    dst,
                    rows,
                    dim,
                    eps,
                } => {
                    let (rows, dim) = (rows as usize, dim as usize);
                    let xs = &vals[x.0 as usize];
                    let ws = weight(w);
                    let mut out = vec![0f32; rows * dim];
                    for r in 0..rows {
                        let b = r * dim;
                        let ss: f32 =
                            (0..dim).map(|i| xs[b + i] * xs[b + i]).sum::<f32>() / dim as f32;
                        let s = 1.0 / (ss + eps).sqrt();
                        for i in 0..dim {
                            out[b + i] = xs[b + i] * s * ws[i];
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Softmax {
                    x,
                    dst,
                    rows,
                    dim,
                    scale,
                } => {
                    let (rows, dim) = (rows as usize, dim as usize);
                    let xs = &vals[x.0 as usize];
                    let mut out = vec![0f32; rows * dim];
                    out.par_chunks_mut(dim)
                        .zip(xs.par_chunks(dim))
                        .for_each(|(o, row)| {
                            let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v * scale));
                            let mut denom = 0f32;
                            for (dst_v, &v) in o.iter_mut().zip(row) {
                                let e = (v * scale - mx).exp();
                                *dst_v = e;
                                denom += e;
                            }
                            let inv = 1.0 / denom;
                            for dst_v in o.iter_mut() {
                                *dst_v *= inv;
                            }
                        });
                    vals[dst.0 as usize] = out;
                }
                Op::QkNorm {
                    x,
                    weight: w,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    eps,
                } => {
                    let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                    let xs = &vals[x.0 as usize];
                    let ws = weight(w);
                    let mut out = vec![0f32; rows * nh * hd];
                    for r in 0..rows {
                        for h in 0..nh {
                            let b = (r * nh + h) * hd;
                            let ss: f32 =
                                (0..hd).map(|i| xs[b + i] * xs[b + i]).sum::<f32>() / hd as f32;
                            let s = 1.0 / (ss + eps).sqrt();
                            for i in 0..hd {
                                out[b + i] = xs[b + i] * s * ws[i];
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Linear {
                    x,
                    weight: w,
                    dst,
                    m,
                    in_f,
                    out_f,
                    w_off,
                } => {
                    let (m, in_f, out_f) = (m as usize, in_f as usize, out_f as usize);
                    let xs = &vals[x.0 as usize];
                    // Stream the (row-major [out_f, in_f]) weight one row at a time straight from the
                    // mmap, dequantizing inside the dot — no full f32 materialization. GGUF rows are
                    // block-aligned, so each row is an equal `bytes/out_f` slice. Output rows are
                    // independent → fan out over the 32 cores with rayon.
                    let buf = bindings.get(w).expect("cpu backend: unbound Weight");
                    let bytes = cpu_buf(buf).read();
                    let dt = g.desc(w).dtype;
                    // `w_off` (elements, row-aligned) selects a projection's rows inside a
                    // CONCATENATED weight (fused QKV): total rows = declared numel / in_f.
                    let total_rows = g.desc(w).numel() / in_f;
                    let bpr = bytes.len() / total_rows; // bytes per weight row
                    let row0 = w_off as usize / in_f;
                    let wbytes: &[u8] = &bytes[row0 * bpr..(row0 + out_f) * bpr];
                    let mut out = vec![0f32; m * out_f];
                    // One token (decode) is the hot path. Dispatch on the weight dtype to the fastest
                    // per-row kernel: integer Q8×Q4_K/Q6_K dots (quantize the activation once), direct
                    // f16/bf16/f32 dots, else fall back to dequant-to-f32 + dot. All fan out over rows.
                    if m == 1 {
                        let xrow = &xs[..in_f];
                        let q8 = matches!(dt, DType::Q4K | DType::Q6K | DType::Q8_0 | DType::Q5K)
                            .then(|| quantize_q8(xrow));
                        out.par_iter_mut().enumerate().for_each(|(o, dst_o)| {
                            let row = &wbytes[o * bpr..o * bpr + bpr];
                            *dst_o = match dt {
                                DType::Q4K => vec_dot_q4k(row, q8.as_ref().unwrap(), in_f),
                                DType::Q6K => vec_dot_q6k(row, q8.as_ref().unwrap(), in_f),
                                DType::Q8_0 => vec_dot_q8_0(row, q8.as_ref().unwrap(), in_f),
                                DType::Q5K => vec_dot_q5k(row, q8.as_ref().unwrap(), in_f),
                                DType::F32 => dot(bytemuck::cast_slice(row), xrow),
                                DType::F16 => dot_f16(row, xrow),
                                DType::Bf16 => dot_bf16(row, xrow),
                                _ => dot(&bytes_to_f32(row, dt), xrow),
                            };
                        });
                    } else {
                        // PREFILL (m > 1): parallelize over output rows (one weight row per task).
                        // For quant types, use the batched dot kernels: the weight row is decoded
                        // ONCE per output row (inside the batch fn), then the integer dot is
                        // repeated across all m token activations — amortising the expensive
                        // nibble/bit unpacking that the single-token path was redoing m times.
                        //
                        // Layout: out[r * out_f + o].  We accumulate into a transposed buffer
                        // out_t[o * m + r] (contiguous in o-major order) so each parallel chunk
                        // owns a contiguous slice of m floats, then scatter into out at the end.
                        let q8s: Vec<Q8> =
                            if matches!(dt, DType::Q4K | DType::Q6K | DType::Q8_0 | DType::Q5K) {
                                (0..m)
                                    .map(|r| quantize_q8(&xs[r * in_f..r * in_f + in_f]))
                                    .collect()
                            } else {
                                Vec::new()
                            };
                        let mut out_t = vec![0f32; out_f * m];
                        // For Q4_K, use 8-row tiling: each rayon task handles 8 consecutive
                        // output rows and loads the Q8 activation zmm ONCE per (block, nibble-pair),
                        // reusing it across all 8 weight rows. This is 4× less activation traffic
                        // than the 2-row path and 8× less than the single-row path. Remainder rows
                        // (out_f % 8) fall through to the 2-row tile then the 1-row batch.
                        if dt == DType::Q4K && out_f >= 8 {
                            let groups8 = out_f / 8;
                            let rem = out_f % 8;
                            let (g8_t, rest_t) = out_t.split_at_mut(groups8 * 8 * m);
                            // 8-row groups (parallel over rayon).
                            g8_t.par_chunks_mut(8 * m).enumerate().for_each(|(g, dc)| {
                                let o = g * 8;
                                let (r0, rest) = dc.split_at_mut(m);
                                let (r1, rest) = rest.split_at_mut(m);
                                let (r2, rest) = rest.split_at_mut(m);
                                let (r3, rest) = rest.split_at_mut(m);
                                let (r4, rest) = rest.split_at_mut(m);
                                let (r5, rest) = rest.split_at_mut(m);
                                let (r6, r7) = rest.split_at_mut(m);
                                vec_dot_q4k_batch8(
                                    [
                                        &wbytes[o * bpr..o * bpr + bpr],
                                        &wbytes[(o + 1) * bpr..(o + 1) * bpr + bpr],
                                        &wbytes[(o + 2) * bpr..(o + 2) * bpr + bpr],
                                        &wbytes[(o + 3) * bpr..(o + 3) * bpr + bpr],
                                        &wbytes[(o + 4) * bpr..(o + 4) * bpr + bpr],
                                        &wbytes[(o + 5) * bpr..(o + 5) * bpr + bpr],
                                        &wbytes[(o + 6) * bpr..(o + 6) * bpr + bpr],
                                        &wbytes[(o + 7) * bpr..(o + 7) * bpr + bpr],
                                    ],
                                    &q8s,
                                    in_f,
                                    [r0, r1, r2, r3, r4, r5, r6, r7],
                                );
                            });
                            // Remainder: up to 7 rows → 2-row pairs, then at most 1 odd tail.
                            let pairs_rem = rem / 2;
                            let (g2_t, odd_t) = rest_t.split_at_mut(pairs_rem * 2 * m);
                            if pairs_rem > 0 {
                                g2_t.par_chunks_mut(2 * m)
                                    .enumerate()
                                    .for_each(|(pair, dc)| {
                                        let o = groups8 * 8 + pair * 2;
                                        let (chunk_a, chunk_b) = dc.split_at_mut(m);
                                        let row_a = &wbytes[o * bpr..o * bpr + bpr];
                                        let row_b = &wbytes[(o + 1) * bpr..(o + 1) * bpr + bpr];
                                        vec_dot_q4k_batch2(
                                            row_a, row_b, &q8s, in_f, chunk_a, chunk_b,
                                        );
                                    });
                            }
                            if rem % 2 != 0 {
                                let o = out_f - 1;
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_q4k_batch(row, &q8s, in_f, odd_t);
                            }
                        } else if dt == DType::Q4K && out_f >= 2 {
                            // Small out_f < 8: fall back to 2-row tile.
                            let pairs = out_f / 2;
                            let (even_t, odd_t) = out_t.split_at_mut(pairs * 2 * m);
                            even_t
                                .par_chunks_mut(2 * m)
                                .enumerate()
                                .for_each(|(pair, dc)| {
                                    let o = pair * 2;
                                    let (chunk_a, chunk_b) = dc.split_at_mut(m);
                                    let row_a = &wbytes[o * bpr..o * bpr + bpr];
                                    let row_b = &wbytes[(o + 1) * bpr..(o + 1) * bpr + bpr];
                                    vec_dot_q4k_batch2(row_a, row_b, &q8s, in_f, chunk_a, chunk_b);
                                });
                            if out_f % 2 != 0 {
                                let o = out_f - 1;
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_q4k_batch(row, &q8s, in_f, odd_t);
                            }
                        } else {
                            out_t.par_chunks_mut(m).enumerate().for_each(|(o, chunk)| {
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                match dt {
                                    DType::Q4K => vec_dot_q4k_batch(row, &q8s, in_f, chunk),
                                    DType::Q6K => vec_dot_q6k_batch(row, &q8s, in_f, chunk),
                                    DType::Q8_0 => vec_dot_q8_0_batch(row, &q8s, in_f, chunk),
                                    DType::Q5K => vec_dot_q5k_batch(row, &q8s, in_f, chunk),
                                    DType::F32 => {
                                        let w32: &[f32] = bytemuck::cast_slice(row);
                                        for r in 0..m {
                                            chunk[r] = dot(w32, &xs[r * in_f..r * in_f + in_f]);
                                        }
                                    }
                                    DType::F16 => {
                                        for r in 0..m {
                                            chunk[r] = dot_f16(row, &xs[r * in_f..r * in_f + in_f]);
                                        }
                                    }
                                    DType::Bf16 => {
                                        for r in 0..m {
                                            chunk[r] =
                                                dot_bf16(row, &xs[r * in_f..r * in_f + in_f]);
                                        }
                                    }
                                    _ => {
                                        // Dequant the weight row ONCE, reuse across all m tokens.
                                        let wf = bytes_to_f32(row, dt);
                                        for r in 0..m {
                                            chunk[r] = dot(&wf, &xs[r * in_f..r * in_f + in_f]);
                                        }
                                    }
                                }
                            });
                        }
                        // Transpose out_t[o * m + r] → out[r * out_f + o], parallel over the m output
                        // rows (each gathers its out_f values from the o-major temp). The serial
                        // version was ~20% of the matvec at large out_f × m.
                        out.par_chunks_mut(out_f).enumerate().for_each(|(r, orow)| {
                            for (o, dst) in orow.iter_mut().enumerate() {
                                *dst = out_t[o * m + r];
                            }
                        });
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Rope {
                    x,
                    positions,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    rope_dim,
                    theta,
                    freq_factors,
                } => {
                    let (rows, nh, hd, rd) = (
                        rows as usize,
                        n_head as usize,
                        head_dim as usize,
                        rope_dim as usize,
                    );
                    let xs = vals[x.0 as usize].clone();
                    let pos = vals[positions.0 as usize].clone();
                    let ff = freq_factors.map(|f| vals[f.0 as usize].clone());
                    let mut out = xs.clone(); // dims beyond rope_dim pass through unchanged
                                              // Op::Rope is the no-qk-norm (llama-family) rotation: INTERLEAVED pairs
                                              // (2p, 2p+1) — llama.cpp's ROPE_TYPE_NORM, matching the Vulkan `rope` kernel
                                              // and the bespoke fused attn_in. (QkNormRope is the NEOX split-half rotation
                                              // used by qwen/gemma; the two styles are NOT interchangeable.)
                    let hf = rd / 2;
                    for r in 0..rows {
                        let p0 = pos[r];
                        for h in 0..nh {
                            let b = (r * nh + h) * hd;
                            for p in 0..hf {
                                let (i0, i1) = (2 * p, 2 * p + 1);
                                let mut ang = p0 * theta.powf(-2.0 * p as f32 / rd as f32);
                                if let Some(ff) = &ff {
                                    ang /= ff[p];
                                }
                                let (s, c) = (ang.sin(), ang.cos());
                                let a = xs[b + i0];
                                let bb = xs[b + i1];
                                out[b + i0] = a * c - bb * s;
                                out[b + i1] = a * s + bb * c;
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::QkNormRope {
                    x,
                    weight: w,
                    positions,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    rope_dim,
                    theta,
                    eps,
                    freq_factors,
                } => {
                    // Fused QkNorm + Rope: one pass per head — rmsnorm (× weight), then rotate the
                    // first `rope_dim` in place (dims beyond pass through normed). Output-identical to
                    // the separate QkNorm→Rope pair; maps 1:1 to the GPU `qk_norm_rope` kernel.
                    let (rows, nh, hd, rd) = (
                        rows as usize,
                        n_head as usize,
                        head_dim as usize,
                        rope_dim as usize,
                    );
                    let xs = &vals[x.0 as usize];
                    let ws = weight(w);
                    let pos = &vals[positions.0 as usize];
                    let ff = freq_factors.map(|f| vals[f.0 as usize].clone());
                    let mut out = vec![0f32; rows * nh * hd];
                    let hf = rd / 2;
                    // Parallel over the m rows. Within a row the RoPE angles depend only on the
                    // position (not the head), so precompute (cos,sin) per rope index ONCE per row and
                    // reuse across all heads — the powf/sin/cos were the bulk and were redone nh×.
                    out.par_chunks_mut(nh * hd)
                        .enumerate()
                        .for_each(|(r, orow)| {
                            let p0 = pos[r];
                            let cs: Vec<(f32, f32)> = (0..hf)
                                .map(|p| {
                                    let mut ang = p0 * theta.powf(-2.0 * p as f32 / rd as f32);
                                    if let Some(ff) = &ff {
                                        ang /= ff[p];
                                    }
                                    (ang.cos(), ang.sin())
                                })
                                .collect();
                            let xr = &xs[r * nh * hd..r * nh * hd + nh * hd];
                            for h in 0..nh {
                                let b = h * hd;
                                let ss: f32 =
                                    (0..hd).map(|i| xr[b + i] * xr[b + i]).sum::<f32>() / hd as f32;
                                let s = 1.0 / (ss + eps).sqrt();
                                for i in 0..hd {
                                    orow[b + i] = xr[b + i] * s * ws[i];
                                }
                                for p in 0..hf {
                                    let (i0, i1) = (p, p + hf);
                                    let (c, sn) = cs[p];
                                    let a = orow[b + i0];
                                    let bb = orow[b + i1];
                                    orow[b + i0] = a * c - bb * sn;
                                    orow[b + i1] = a * sn + bb * c;
                                }
                            }
                        });
                    vals[dst.0 as usize] = out;
                }
                Op::WriteKv {
                    src,
                    cache,
                    rows,
                    row_stride,
                    pos,
                } => {
                    let (rows, rs, pos) = (rows as usize, row_stride as usize, pos as usize);
                    let s = &vals[src.0 as usize];
                    // Write the new row(s) straight into the persistent KV buffer — only `rows` rows
                    // touched, not the whole `max_ctx`-sized cache. The cache dtype (f16 to match the
                    // GPU and halve memory, or f32) is read from the graph; cast on write.
                    let buf = bindings.get(cache).expect("cpu backend: unbound KV cache");
                    let mut d = cpu_buf(buf).owned();
                    let base = pos * rs;
                    let n = rows * rs;
                    match g.desc(cache).dtype {
                        DType::F16 => {
                            let df: &mut [u16] = bytemuck::cast_slice_mut(&mut d);
                            for i in 0..n {
                                df[base + i] = half::f16::from_f32(s[i]).to_bits();
                            }
                        }
                        DType::Q8_0 => {
                            // Q8_0 blocks (34 B / 32 elems): d = amax/127 (stored f16), q =
                            // round(x/d) — the llama.cpp quantize_row_q8_0 reference formula.
                            // `base`/`n` are element counts and rows are 32-aligned (the runner
                            // gates on it), so blocks never straddle a write.
                            debug_assert!(base % 32 == 0 && n % 32 == 0);
                            for b in 0..n / 32 {
                                let src32 = &s[b * 32..b * 32 + 32];
                                let amax = src32.iter().fold(0f32, |m, &v| m.max(v.abs()));
                                let dq = amax / 127.0;
                                let id = if dq != 0.0 { 1.0 / dq } else { 0.0 };
                                let off = (base / 32 + b) * 34;
                                let dh = half::f16::from_f32(dq).to_bits().to_le_bytes();
                                d[off] = dh[0];
                                d[off + 1] = dh[1];
                                for (i, &v) in src32.iter().enumerate() {
                                    d[off + 2 + i] = (v * id).round_ties_even() as i32 as i8 as u8;
                                }
                            }
                        }
                        DType::Bf16 => {
                            let df: &mut [u16] = bytemuck::cast_slice_mut(&mut d);
                            for i in 0..n {
                                df[base + i] = half::bf16::from_f32(s[i]).to_bits();
                            }
                        }
                        dt @ (DType::Turbo2 | DType::Turbo3 | DType::Turbo4) => {
                            // TurboQuant: each 128-elem group (a head_dim slice) → one block
                            // (L2-norm + WHT + 2/3/4-bit PolarQuant). base/n are 128-aligned (the
                            // runner gates head_dim%128), so blocks never straddle a write.
                            debug_assert!(base % 128 == 0 && n % 128 == 0);
                            let bb = crate::turbo::block_bytes(dt);
                            let blk0 = base / 128;
                            for b in 0..n / 128 {
                                let off = (blk0 + b) * bb;
                                crate::turbo::quantize_block(
                                    dt,
                                    &s[b * 128..b * 128 + 128],
                                    &mut d[off..off + bb],
                                );
                            }
                        }
                        dt if crate::kvquant::supported(dt) => {
                            // Mainline low-bit KV quants (q4_0/q4_1/q5_0/q5_1/iq4_nl): quantize the
                            // f32 activations into 32-elem blocks. base/n are 32-aligned (kv_align_ok).
                            debug_assert!(base % 32 == 0 && n % 32 == 0);
                            let bb = infr_gguf::nbytes(dt, 32);
                            let off = (base / 32) * bb;
                            crate::kvquant::quantize_row(
                                dt,
                                &s[..n],
                                &mut d[off..off + n / 32 * bb],
                            );
                        }
                        _ => {
                            let df: &mut [f32] = bytemuck::cast_slice_mut(&mut d);
                            df[base..base + n].copy_from_slice(&s[..n]);
                        }
                    }
                }
                Op::Attention {
                    q,
                    k_cache,
                    v_cache,
                    dst,
                    rows,
                    kv_len,
                    n_head,
                    n_kv,
                    head_dim,
                    scale,
                    mask,
                    pos,
                } => {
                    let (rows, kv_len, nh, nkv, hd) = (
                        rows as usize,
                        kv_len as usize,
                        n_head as usize,
                        n_kv as usize,
                        head_dim as usize,
                    );
                    let qs = &vals[q.0 as usize];
                    // K/V live in their persistent buffers (f32); borrow them — attention reads only
                    // the first `kv_len` rows, never the whole `max_ctx` cache.
                    let kbuf = bindings.get(k_cache).expect("cpu backend: unbound k_cache");
                    let vbuf = bindings.get(v_cache).expect("cpu backend: unbound v_cache");
                    let kguard = cpu_buf(kbuf).read();
                    let vguard = cpu_buf(vbuf).read();
                    // Materialize the valid KV prefix (`kv_len` rows) as f32. K and V pick their
                    // cache dtype INDEPENDENTLY (f16 matches the GPU's f16 KV; Q8_0 blocks dequant
                    // via y = d*q) — the inner dot then runs in f32 either way.
                    let need = kv_len * nkv * hd;
                    let deq = |b: &[u8], dt: DType| -> Vec<f32> {
                        match dt {
                            DType::F16 => bytemuck::cast_slice::<u8, u16>(b)[..need]
                                .iter()
                                .map(|&x| half::f16::from_bits(x).to_f32())
                                .collect(),
                            DType::Q8_0 => crate::dequant_prefix_q8_0(b, need),
                            // TurboQuant blocks store the WHT-rotated values; dequant + inverse WHT
                            // recovers the original domain so the f32 SDPA below runs unchanged.
                            dt @ (DType::Turbo2 | DType::Turbo3 | DType::Turbo4) => {
                                crate::turbo::dequant_prefix_orig(dt, b, need)
                            }
                            // bf16 + mainline low-bit quants: dequant the block-aligned prefix via the
                            // shared GGUF dequant (only the valid `kv_len` rows, not the whole cache).
                            DType::Bf16
                            | DType::Q4_0
                            | DType::Q4_1
                            | DType::Q5_0
                            | DType::Q5_1
                            | DType::Iq4Nl => {
                                let pb = infr_gguf::nbytes(dt, need);
                                infr_gguf::dequant::dequant_block(dt, &b[..pb])
                                    .expect("cpu backend: KV dequant")
                            }
                            _ => bytemuck::cast_slice::<u8, f32>(b)[..need].to_vec(),
                        }
                    };
                    let ks = deq(&kguard, g.desc(k_cache).dtype);
                    let vs = deq(&vguard, g.desc(v_cache).dtype);
                    let group = nh / nkv;
                    // `Causal`/`SlidingWindow` clip the causal END at `abs+1` (per-row, from
                    // `pos`); `Canvas` (DiffusionGemma denoise — see `AttnMask::Canvas`'s doc)
                    // ignores `pos` entirely and gives every row the SAME fixed bidirectional
                    // range `[lo, kv_len)`.
                    let (window, canvas_lo) = match mask {
                        AttnMask::Causal => (0usize, None),
                        AttnMask::SlidingWindow(w) => (w, None),
                        AttnMask::Canvas { lo } => (0usize, Some(lo)),
                    };
                    let mut out = vec![0f32; rows * nh * hd];
                    // Each (ti, h) pair writes exactly one hd-sized output slice with no
                    // cross-iteration deps → embarrassingly parallel.  Chunk index i = ti*nh+h.
                    out.par_chunks_mut(hd)
                        .enumerate()
                        .for_each(|(i, ob_slice)| {
                            let ti = i / nh;
                            let h = i % nh;
                            let kvh = h / group;
                            let qb = (ti * nh + h) * hd;
                            let abs = pos as usize + ti; // absolute position of this query
                            let (lo, hi) = match canvas_lo {
                                // bidirectional: every row attends the same fixed [lo, kv_len).
                                Some(clo) => (clo, kv_len),
                                // causal (± SWA): [lo, abs] — SWA clips lo to abs-window+1.
                                None => {
                                    let lo = if window > 0 && abs + 1 > window {
                                        abs + 1 - window
                                    } else {
                                        0
                                    };
                                    (lo, abs + 1)
                                }
                            };
                            let n_keys = hi - lo;
                            let mut sc = vec![0f32; n_keys];
                            let mut mx = f32::NEG_INFINITY;
                            for (jj, scj) in sc.iter_mut().enumerate() {
                                let j = lo + jj;
                                let kb = (j * nkv + kvh) * hd;
                                let d: f32 = (0..hd).map(|x| qs[qb + x] * ks[kb + x]).sum();
                                *scj = d * scale;
                                mx = mx.max(*scj);
                            }
                            let mut l = 0f32;
                            for &s in &sc {
                                l += (s - mx).exp();
                            }
                            for (jj, &s) in sc.iter().enumerate() {
                                let j = lo + jj;
                                let p = (s - mx).exp() / l;
                                let vb = (j * nkv + kvh) * hd;
                                for x in 0..hd {
                                    ob_slice[x] += p * vs[vb + x];
                                }
                            }
                        });
                    vals[dst.0 as usize] = out;
                }
                Op::GatedAct {
                    gate,
                    up,
                    dst,
                    rows,
                    nff,
                    act,
                    up_off,
                } => {
                    let (rows, nff, up_off) = (rows as usize, nff as usize, up_off as usize);
                    let gs = &vals[gate.0 as usize];
                    let us = &vals[up.0 as usize];
                    // `up` may be a wider layer-major buffer (E2B); the per-row stride stays `nff`
                    // but the read is shifted by `up_off` (0 for the normal [rows, nff] case).
                    let mut out = vec![0f32; rows * nff];
                    for r in 0..rows {
                        let gb = r * nff;
                        let ub = r * nff + up_off;
                        for i in 0..nff {
                            out[gb + i] = act_fn(act, gs[gb + i]) * us[ub + i];
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::GatedActFused {
                    gu,
                    dst,
                    rows,
                    nff,
                    act,
                } => {
                    // Combined [rows, 2*nff] gate|up buffer: gate half first, up half second.
                    let (rows, nff) = (rows as usize, nff as usize);
                    let gus = &vals[gu.0 as usize];
                    let mut out = vec![0f32; rows * nff];
                    for r in 0..rows {
                        let gb = r * 2 * nff;
                        for i in 0..nff {
                            out[r * nff + i] = act_fn(act, gus[gb + i]) * gus[gb + nff + i];
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Add { a, b, dst, n } => {
                    let n = n as usize;
                    let av = vals[a.0 as usize].clone();
                    let bv = &vals[b.0 as usize];
                    let mut out = vec![0f32; n];
                    for i in 0..n {
                        out[i] = av[i] + bv[i];
                    }
                    vals[dst.0 as usize] = out;
                }
                // Broadcast bias: add the length-`n` `bias` to each of `rows` rows (Qwen2 q/k/v).
                Op::AddBias {
                    x,
                    bias,
                    dst,
                    rows,
                    n,
                } => {
                    let (rows, n) = (rows as usize, n as usize);
                    let xs = vals[x.0 as usize].clone();
                    let bv = weight(bias); // bias is a bound weight, not an activation
                    let mut out = vec![0f32; rows * n];
                    for r in 0..rows {
                        for c in 0..n {
                            out[r * n + c] = xs[r * n + c] + bv[c];
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Scale { x, dst, s, n } => {
                    let n = n as usize;
                    let xs = vals[x.0 as usize].clone();
                    let mut out = vec![0f32; n];
                    for i in 0..n {
                        out[i] = xs[i] * s;
                    }
                    vals[dst.0 as usize] = out;
                }
                // Broadcast multiply: the length-`n` `vec` scales every one of `rows` rows
                // (diffusion-gemma's router input scale — the multiplicative twin of `AddBias`).
                Op::MulVec {
                    x,
                    vec: vecid,
                    dst,
                    rows,
                    n,
                } => {
                    let (rows, n) = (rows as usize, n as usize);
                    let xs = vals[x.0 as usize].clone();
                    let vv = weight(vecid); // vec is a bound weight, not an activation
                    let mut out = vec![0f32; rows * n];
                    for r in 0..rows {
                        for c in 0..n {
                            out[r * n + c] = xs[r * n + c] * vv[c];
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Softcap { x, dst, cap, n } => {
                    let n = n as usize;
                    let xs = vals[x.0 as usize].clone();
                    let mut out = vec![0f32; n];
                    for i in 0..n {
                        out[i] = cap * (xs[i] / cap).tanh();
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Copy {
                    src,
                    src_off,
                    dst,
                    dst_off,
                    n,
                } => {
                    let (so, dof, n) = (src_off as usize, dst_off as usize, n as usize);
                    let s = vals[src.0 as usize].clone();
                    vals[dst.0 as usize][dof..dof + n].copy_from_slice(&s[so..so + n]);
                }
                Op::CopyStrided {
                    src,
                    src_off,
                    src_stride,
                    dst,
                    dst_off,
                    dst_stride,
                    rows,
                    n,
                } => {
                    let (so, ss, dof, ds, n) = (
                        src_off as usize,
                        src_stride as usize,
                        dst_off as usize,
                        dst_stride as usize,
                        n as usize,
                    );
                    let s = vals[src.0 as usize].clone();
                    let d = &mut vals[dst.0 as usize];
                    for r in 0..rows as usize {
                        d[dof + r * ds..dof + r * ds + n]
                            .copy_from_slice(&s[so + r * ss..so + r * ss + n]);
                    }
                }
                Op::MoeFfn {
                    x,
                    router_x,
                    router,
                    gate_exps,
                    up_exps,
                    down_exps,
                    down_scale,
                    fused_gate_up,
                    dst,
                    ne,
                    n_expert,
                    n_used,
                    n_ff_exp,
                    scale,
                    act,
                } => {
                    let (ne, n_expert, n_used, nffx) = (
                        ne as usize,
                        n_expert as usize,
                        n_used as usize,
                        n_ff_exp as usize,
                    );
                    let xs = vals[x.0 as usize].clone();
                    // `router_x` is usually the SAME tensor as `x` (qwen3moe); diffusion-gemma binds
                    // a differently-normalized row (see the `Op::MoeFfn` doc). Clone independently —
                    // it may legitimately be a different handle with its own row layout.
                    let rxs = vals[router_x.0 as usize].clone();
                    // `x` may hold several rows (the seam's batched prefill): route + run the
                    // expert FFN independently per row — the reference semantics for the GPU
                    // adapter's GPU-routed batched form.
                    let rows = xs.len() / ne;
                    // Stream a (row-major [out_f, in_f]) weight slice and matvec it against `v` —
                    // dequant per row, exactly like `Op::Linear`, parallel over rows.
                    let matvec = |bytes: &[u8], dt: DType, v: &[f32], in_f: usize, out_f: usize| {
                        let bpr = bytes.len() / out_f;
                        (0..out_f)
                            .into_par_iter()
                            .map(|r| {
                                let row = bytes_to_f32(&bytes[r * bpr..r * bpr + bpr], dt);
                                dot(&row, &v[..in_f])
                            })
                            .collect::<Vec<f32>>()
                    };
                    let rbuf = bindings.get(router).expect("cpu backend: unbound router");
                    let gbuf = bindings
                        .get(gate_exps)
                        .expect("cpu backend: unbound gate_exps");
                    let dbuf = bindings
                        .get(down_exps)
                        .expect("cpu backend: unbound down_exps");
                    let rbytes = cpu_buf(rbuf).read();
                    let gb = cpu_buf(gbuf).read();
                    let db = cpu_buf(dbuf).read();
                    let gdt = g.desc(gate_exps).dtype;
                    let ddt = g.desc(down_exps).dtype;
                    // Fused: `gate_exps` holds BOTH roles ([ne, 2*n_ff_exp, n_expert], gate rows
                    // first); split gets its own separate up_exps/up buffer.
                    let (ub, udt) = if fused_gate_up {
                        (None, gdt)
                    } else {
                        let ubuf = bindings.get(up_exps).expect("cpu backend: unbound up_exps");
                        (Some(cpu_buf(ubuf).read()), g.desc(up_exps).dtype)
                    };
                    let gst = gb.len() / n_expert;
                    let ust = ub.as_ref().map(|b| b.len() / n_expert);
                    let dst_ = db.len() / n_expert;
                    let dscale = down_scale.map(&weight); // per-expert scale [n_expert], if any
                    let mut out = vec![0f32; rows * ne];
                    for (r, orow) in out.chunks_mut(ne).enumerate() {
                        let xr = &xs[r * ne..r * ne + ne];
                        let xrr = &rxs[r * ne..r * ne + ne];
                        // Router softmax over all experts (reads router_x, NOT the expert input).
                        let logits = matvec(&rbytes, g.desc(router).dtype, xrr, ne, n_expert);
                        let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut probs: Vec<f32> =
                            logits.iter().map(|&v| (v - maxl).exp()).collect();
                        let psum: f32 = probs.iter().sum();
                        for p in probs.iter_mut() {
                            *p /= psum;
                        }
                        // Top-`n_used` experts, renormalized weights.
                        let mut idx: Vec<usize> = (0..n_expert).collect();
                        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                        idx.truncate(n_used);
                        let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                        for &e in &idx {
                            let (gate, up) = if fused_gate_up {
                                // One matvec over the fused [2*nffx, ne] expert slice; first nffx
                                // rows are gate, next nffx are up (Op::GatedActFused's convention).
                                let full =
                                    matvec(&gb[e * gst..(e + 1) * gst], gdt, xr, ne, 2 * nffx);
                                (full[..nffx].to_vec(), full[nffx..].to_vec())
                            } else {
                                let ub = ub.as_ref().expect("split gate/up: up_exps missing");
                                let ust = ust.expect("split gate/up: up stride missing");
                                (
                                    matvec(&gb[e * gst..(e + 1) * gst], gdt, xr, ne, nffx),
                                    matvec(&ub[e * ust..(e + 1) * ust], udt, xr, ne, nffx),
                                )
                            };
                            let actv: Vec<f32> =
                                (0..nffx).map(|i| act_fn(act, gate[i]) * up[i]).collect();
                            let mut y = matvec(&db[e * dst_..(e + 1) * dst_], ddt, &actv, nffx, ne);
                            if let Some(ds) = &dscale {
                                let s = ds[e];
                                for v in y.iter_mut() {
                                    *v *= s;
                                }
                            }
                            let w_e = probs[e] / wsum * scale;
                            for i in 0..ne {
                                orow[i] += w_e * y[i];
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Conv1dSilu {
                    x,
                    weight: w,
                    state,
                    dst,
                    rows,
                    channels,
                    kernel,
                } => {
                    let (rr, cc, kk) = (rows as usize, channels as usize, kernel as usize);
                    let xs = vals[x.0 as usize].clone(); // [rows, channels]
                    let ws = weight(w); // [channels, kernel] row-major (per-channel kernel)
                    let st = &mut vals[state.0 as usize]; // [(kernel-1), channels], oldest row first
                    let mut out = vec![0f32; rr * cc];
                    // Process the rows in sequence, carrying the rolling history across tokens.
                    for t in 0..rr {
                        let xt = &xs[t * cc..t * cc + cc];
                        for ch in 0..cc {
                            // window = [history rows.. , current x]; tap j uses weight[ch*kk + j].
                            let mut acc = 0f32;
                            for j in 0..kk - 1 {
                                acc += st[j * cc + ch] * ws[ch * kk + j];
                            }
                            acc += xt[ch] * ws[ch * kk + (kk - 1)];
                            out[t * cc + ch] = acc / (1.0 + (-acc).exp()); // silu
                        }
                        // shift history (drop oldest, append raw x).
                        for j in 0..kk.saturating_sub(2) {
                            for ch in 0..cc {
                                st[j * cc + ch] = st[(j + 1) * cc + ch];
                            }
                        }
                        if kk >= 2 {
                            for ch in 0..cc {
                                st[(kk - 2) * cc + ch] = xt[ch];
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::DeltaNet {
                    q,
                    k,
                    v,
                    b,
                    a,
                    a_coef,
                    dt_bias,
                    state,
                    dst,
                    rows,
                    n_vhead,
                    n_khead,
                    head_k,
                    head_v,
                    eps,
                } => {
                    let (rr, nv, nk, kd, vd) = (
                        rows as usize,
                        n_vhead as usize,
                        n_khead as usize,
                        head_k as usize,
                        head_v as usize,
                    );
                    let qf = vals[q.0 as usize].clone(); // [rows, nk*kd]
                    let kf = vals[k.0 as usize].clone();
                    let vf = vals[v.0 as usize].clone(); // [rows, nv*vd]
                    let bf = vals[b.0 as usize].clone(); // [rows, nv]
                    let af = vals[a.0 as usize].clone();
                    let acoef = weight(a_coef);
                    let dtb = weight(dt_bias);
                    let st = &mut vals[state.0 as usize]; // [nv, kd, vd]
                    let mut out = vec![0f32; rr * nv * vd];
                    let qscale = 1.0 / (kd as f32).sqrt();
                    let l2 = |slice: &[f32]| -> f32 {
                        (slice.iter().map(|x| x * x).sum::<f32>() + eps).sqrt()
                    };
                    // Sequential scan over the rows, carrying the per-head state S across tokens.
                    for t in 0..rr {
                        let (qb, vb, bb) = (t * nk * kd, t * nv * vd, t * nv);
                        for h in 0..nv {
                            // GQA: q/k heads TILED to nv value heads → v-head h uses q/k head h % nk.
                            let kh_idx = h % nk;
                            let mut qh = qf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
                            let mut kh = kf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
                            let vh = &vf[vb + h * vd..vb + h * vd + vd];
                            let qn = l2(&qh);
                            let kn = l2(&kh);
                            for x in qh.iter_mut() {
                                *x = *x / qn * qscale;
                            }
                            for x in kh.iter_mut() {
                                *x /= kn;
                            }
                            let beta = 1.0 / (1.0 + (-bf[bb + h]).exp());
                            // softplus(a + dt_bias), then g = a_coef * softplus (≤ 0); decay = exp(g).
                            let sp = {
                                let z = af[bb + h] + dtb[h];
                                z.max(0.0) + (-z.abs()).exp().ln_1p()
                            };
                            let decay = (acoef[h] * sp).exp();
                            let sh = &mut st[h * kd * vd..(h + 1) * kd * vd]; // [kd, vd]
                            for x in sh.iter_mut() {
                                *x *= decay;
                            }
                            // kv = kᵀS  [vd]
                            let mut kv = vec![0f32; vd];
                            for kk in 0..kd {
                                let kkv = kh[kk];
                                let row = &sh[kk * vd..kk * vd + vd];
                                for d in 0..vd {
                                    kv[d] += kkv * row[d];
                                }
                            }
                            // delta = (v - kv)*beta ; S += k ⊗ delta
                            let delta: Vec<f32> = (0..vd).map(|d| (vh[d] - kv[d]) * beta).collect();
                            for kk in 0..kd {
                                let kkv = kh[kk];
                                let row = &mut sh[kk * vd..kk * vd + vd];
                                for d in 0..vd {
                                    row[d] += kkv * delta[d];
                                }
                            }
                            // out = qᵀS  [vd]
                            let oh = &mut out[vb + h * vd..vb + h * vd + vd];
                            for kk in 0..kd {
                                let qv = qh[kk];
                                let row = &sh[kk * vd..kk * vd + vd];
                                for d in 0..vd {
                                    oh[d] += qv * row[d];
                                }
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
            }
            if let Some(t0) = __t0 {
                *op_times.entry(op.kind()).or_insert(0.0) += t0.elapsed().as_secs_f64();
            }
        }
        if prof_ops {
            let mut v: Vec<_> = op_times.into_iter().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let tot: f64 = v.iter().map(|(_, t)| t).sum();
            eprintln!("[prof-ops] execute totals ({:.1} ms):", tot * 1000.0);
            for (k, t) in v {
                eprintln!("  {k:12} {:7.2} ms  {:5.1}%", t * 1000.0, t / tot * 100.0);
            }
        }

        // Write back the buffers the model reads after execute: Outputs (logits) and mutated f32
        // Inputs (conv/recurrent state). KV caches (`direct`) were written in place by `WriteKv`, so
        // they're skipped — no full-cache copy. Weights are read-only; positions are I32, unchanged.
        for (i, decl) in g.tensors.iter().enumerate() {
            let write_back = matches!(decl.kind, TensorKind::Output)
                || (decl.kind == TensorKind::Input
                    && decl.desc.dtype == DType::F32
                    && !direct.contains(&TensorId(i as u32)));
            if !write_back {
                continue;
            }
            if let Some(buf) = bindings.get(TensorId(i as u32)) {
                let mut d = cpu_buf(buf).owned();
                d.copy_from_slice(bytemuck::cast_slice(&vals[i]));
            }
        }
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod kernel_tests {
    //! CPU-only, no GPU, no model file: the optimized quant/f16 dot kernels must match the trusted
    //! f32 reference (`dequant_block` → naive `dot`) on the SAME bytes. We dot against the *quantized*
    //! activation (`d8 * q8`) so the only difference is f32 summation order — i.e. this isolates
    //! kernel correctness from the (separate, expected) Q8 activation-quant error.
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed
    }
    fn det_bytes(n: usize, mut seed: u64) -> Vec<u8> {
        (0..n).map(|_| (lcg(&mut seed) >> 33) as u8).collect()
    }
    fn det_x(n: usize, mut seed: u64) -> Vec<f32> {
        (0..n)
            .map(|_| ((lcg(&mut seed) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
            .collect()
    }
    fn put_f16(b: &mut [u8], v: f32) {
        b.copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    /// The reference activation the integer kernels actually see: `d8 * q8` per super-block.
    fn dequant_q8(q8: &Q8) -> Vec<f32> {
        let mut x = vec![0f32; q8.qs.len()];
        for (b, &d) in q8.d.iter().enumerate() {
            for i in 0..256 {
                x[b * 256 + i] = d * q8.qs[b * 256 + i] as f32;
            }
        }
        x
    }
    fn rel_err(got: f32, want: f32) -> f32 {
        (got - want).abs() / want.abs().max(1.0)
    }

    #[test]
    fn q4k_dot_matches_dequant_reference() {
        let in_f = 768; // 3 super-blocks
        let nb = in_f / 256;
        let mut w = det_bytes(nb * 144, 1);
        for k in 0..nb {
            put_f16(&mut w[k * 144..k * 144 + 2], 0.05); // d
            put_f16(&mut w[k * 144 + 2..k * 144 + 4], 0.015); // dmin
        }
        let wref = dequant_block(DType::Q4K, &w).unwrap();
        let q8 = quantize_q8(&det_x(in_f, 2));
        let got = vec_dot_q4k(&w, &q8, in_f);
        let want = dot(&wref, &dequant_q8(&q8));
        assert!(rel_err(got, want) < 1e-3, "q4k: got {got}, want {want}");
    }

    #[test]
    fn q6k_dot_matches_dequant_reference() {
        let in_f = 768;
        let nb = in_f / 256;
        let mut w = det_bytes(nb * 210, 3);
        for k in 0..nb {
            put_f16(&mut w[k * 210 + 208..k * 210 + 210], 0.04); // d
        }
        let wref = dequant_block(DType::Q6K, &w).unwrap();
        let q8 = quantize_q8(&det_x(in_f, 4));
        let got = vec_dot_q6k(&w, &q8, in_f);
        let want = dot(&wref, &dequant_q8(&q8));
        assert!(rel_err(got, want) < 1e-3, "q6k: got {got}, want {want}");
    }

    #[test]
    fn q8_0_dot_matches_dequant_reference() {
        // Q8_0: 34 bytes / 32 elems. in_f must be a multiple of 256 (activation super-block size).
        let in_f = 512; // 2 super-blocks = 16 Q8_0 weight blocks
        let nb_w = in_f / 32;
        let mut w = det_bytes(nb_w * 34, 9);
        for k in 0..nb_w {
            put_f16(&mut w[k * 34..k * 34 + 2], 0.03); // d
        }
        let wref = dequant_block(DType::Q8_0, &w).unwrap();
        let q8 = quantize_q8(&det_x(in_f, 10));
        let got = vec_dot_q8_0(&w, &q8, in_f);
        let want = dot(&wref, &dequant_q8(&q8));
        assert!(rel_err(got, want) < 1e-3, "q8_0: got {got}, want {want}");
    }

    #[test]
    fn q5k_dot_matches_dequant_reference() {
        // Q5_K: 176 bytes / 256 elems. in_f must be a multiple of 256.
        let in_f = 512; // 2 Q5K blocks = 2 super-blocks
        let nb = in_f / 256;
        let mut w = det_bytes(nb * 176, 11);
        for k in 0..nb {
            put_f16(&mut w[k * 176..k * 176 + 2], 0.05); // d
            put_f16(&mut w[k * 176 + 2..k * 176 + 4], 0.01); // dmin
        }
        let wref = dequant_block(DType::Q5K, &w).unwrap();
        let q8 = quantize_q8(&det_x(in_f, 12));
        let got = vec_dot_q5k(&w, &q8, in_f);
        let want = dot(&wref, &dequant_q8(&q8));
        assert!(rel_err(got, want) < 1e-3, "q5k: got {got}, want {want}");
    }

    #[test]
    fn f16_dot_matches_reference() {
        let n = 257; // odd, exercises the tail past the 8-wide chunks
        let x = det_x(n, 5);
        let wf = det_x(n, 6);
        let wbytes: Vec<u8> = wf
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let wref: Vec<f32> = wf
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        assert!(rel_err(dot_f16(&wbytes, &x), dot(&wref, &x)) < 1e-4);
    }

    #[test]
    fn bf16_dot_matches_reference() {
        let n = 130;
        let x = det_x(n, 7);
        let wf = det_x(n, 8);
        let wbytes: Vec<u8> = wf
            .iter()
            .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes()) // bf16 = top 16 bits
            .collect();
        let wref: Vec<f32> = wf
            .iter()
            .map(|&v| f32::from_bits((v.to_bits() >> 16) << 16))
            .collect();
        assert!(rel_err(dot_bf16(&wbytes, &x), dot(&wref, &x)) < 1e-4);
    }

    // ── Batch kernel bit-identity tests ──────────────────────────────────────
    //
    // For each quant type, assert that `vec_dot_qXk_batch` produces the same
    // f32 bits as calling `vec_dot_qXk` per-activation on every token.

    #[test]
    fn q4k_batch_bit_identical_to_single() {
        let in_f = 512; // 2 super-blocks
        let nb = in_f / 256;
        let m = 7usize; // odd count to test non-power-of-2
        let mut w = det_bytes(nb * 144, 20);
        for k in 0..nb {
            put_f16(&mut w[k * 144..k * 144 + 2], 0.05);
            put_f16(&mut w[k * 144 + 2..k * 144 + 4], 0.015);
        }
        let q8s: Vec<Q8> = (0..m)
            .map(|r| quantize_q8(&det_x(in_f, 30 + r as u64)))
            .collect();
        let mut batch_out = vec![0f32; m];
        vec_dot_q4k_batch(&w, &q8s, in_f, &mut batch_out);
        for r in 0..m {
            let want = vec_dot_q4k(&w, &q8s[r], in_f);
            assert_eq!(
                batch_out[r].to_bits(),
                want.to_bits(),
                "q4k batch[{r}]: got {}, want {}",
                batch_out[r],
                want,
            );
        }
    }

    #[test]
    fn q6k_batch_bit_identical_to_single() {
        let in_f = 512;
        let nb = in_f / 256;
        let m = 5usize;
        let mut w = det_bytes(nb * 210, 21);
        for k in 0..nb {
            put_f16(&mut w[k * 210 + 208..k * 210 + 210], 0.04);
        }
        let q8s: Vec<Q8> = (0..m)
            .map(|r| quantize_q8(&det_x(in_f, 40 + r as u64)))
            .collect();
        let mut batch_out = vec![0f32; m];
        vec_dot_q6k_batch(&w, &q8s, in_f, &mut batch_out);
        for r in 0..m {
            let want = vec_dot_q6k(&w, &q8s[r], in_f);
            assert_eq!(
                batch_out[r].to_bits(),
                want.to_bits(),
                "q6k batch[{r}]: got {}, want {}",
                batch_out[r],
                want,
            );
        }
    }

    #[test]
    fn q8_0_batch_bit_identical_to_single() {
        let in_f = 512; // 2 super-blocks = 16 Q8_0 weight blocks
        let nb_w = in_f / 32;
        let m = 6usize;
        let mut w = det_bytes(nb_w * 34, 22);
        for k in 0..nb_w {
            put_f16(&mut w[k * 34..k * 34 + 2], 0.03);
        }
        let q8s: Vec<Q8> = (0..m)
            .map(|r| quantize_q8(&det_x(in_f, 50 + r as u64)))
            .collect();
        let mut batch_out = vec![0f32; m];
        vec_dot_q8_0_batch(&w, &q8s, in_f, &mut batch_out);
        for r in 0..m {
            let want = vec_dot_q8_0(&w, &q8s[r], in_f);
            assert_eq!(
                batch_out[r].to_bits(),
                want.to_bits(),
                "q8_0 batch[{r}]: got {}, want {}",
                batch_out[r],
                want,
            );
        }
    }

    #[test]
    fn q5k_batch_bit_identical_to_single() {
        let in_f = 512;
        let nb = in_f / 256;
        let m = 4usize;
        let mut w = det_bytes(nb * 176, 23);
        for k in 0..nb {
            put_f16(&mut w[k * 176..k * 176 + 2], 0.05);
            put_f16(&mut w[k * 176 + 2..k * 176 + 4], 0.01);
        }
        let q8s: Vec<Q8> = (0..m)
            .map(|r| quantize_q8(&det_x(in_f, 60 + r as u64)))
            .collect();
        let mut batch_out = vec![0f32; m];
        vec_dot_q5k_batch(&w, &q8s, in_f, &mut batch_out);
        for r in 0..m {
            let want = vec_dot_q5k(&w, &q8s[r], in_f);
            assert_eq!(
                batch_out[r].to_bits(),
                want.to_bits(),
                "q5k batch[{r}]: got {}, want {}",
                batch_out[r],
                want,
            );
        }
    }

    /// Assert the 8-row tile produces bit-identical results to `vec_dot_q4k` per (row, token).
    /// Uses 11 rows to exercise the full remainder path: 8-row tile + 2-row pair + 1-row tail.
    #[test]
    fn q4k_batch8_bit_identical_to_single() {
        let n_rows = 11usize; // 8 + 2-row-pair + 1-row-tail
        let in_f = 512; // 2 super-blocks
        let nb = in_f / 256;
        let m = 9usize; // non-power-of-2 token count

        // Build n_rows random weight rows with valid f16 d/dmin.
        let ws: Vec<Vec<u8>> = (0..n_rows)
            .map(|i| {
                let mut w = det_bytes(nb * 144, 100 + i as u64);
                for k in 0..nb {
                    put_f16(&mut w[k * 144..k * 144 + 2], 0.05);
                    put_f16(&mut w[k * 144 + 2..k * 144 + 4], 0.015);
                }
                w
            })
            .collect();

        let q8s: Vec<Q8> = (0..m)
            .map(|r| quantize_q8(&det_x(in_f, 200 + r as u64)))
            .collect();

        // Reference: vec_dot_q4k per (row, token) — the scalar single-token oracle.
        let want: Vec<Vec<f32>> = (0..n_rows)
            .map(|i| (0..m).map(|r| vec_dot_q4k(&ws[i], &q8s[r], in_f)).collect())
            .collect();

        // Test the 8-row tile for rows 0..8.
        let mut got0 = vec![0f32; m];
        let mut got1 = vec![0f32; m];
        let mut got2 = vec![0f32; m];
        let mut got3 = vec![0f32; m];
        let mut got4 = vec![0f32; m];
        let mut got5 = vec![0f32; m];
        let mut got6 = vec![0f32; m];
        let mut got7 = vec![0f32; m];
        vec_dot_q4k_batch8(
            [
                &ws[0], &ws[1], &ws[2], &ws[3], &ws[4], &ws[5], &ws[6], &ws[7],
            ],
            &q8s,
            in_f,
            [
                &mut got0, &mut got1, &mut got2, &mut got3, &mut got4, &mut got5, &mut got6,
                &mut got7,
            ],
        );

        // Test 2-row remainder (rows 8..10) — same path as the dispatch remainder.
        let mut got8 = vec![0f32; m];
        let mut got9 = vec![0f32; m];
        vec_dot_q4k_batch2(&ws[8], &ws[9], &q8s, in_f, &mut got8, &mut got9);

        // Test 1-row tail (row 10).
        let mut got10 = vec![0f32; m];
        vec_dot_q4k_batch(&ws[10], &q8s, in_f, &mut got10);

        let got_all: [&Vec<f32>; 11] = [
            &got0, &got1, &got2, &got3, &got4, &got5, &got6, &got7, &got8, &got9, &got10,
        ];
        for i in 0..n_rows {
            for r in 0..m {
                assert_eq!(
                    got_all[i][r].to_bits(),
                    want[i][r].to_bits(),
                    "q4k_batch8 row {i} token {r}: got {}, want {}",
                    got_all[i][r],
                    want[i][r],
                );
            }
        }
    }
}
