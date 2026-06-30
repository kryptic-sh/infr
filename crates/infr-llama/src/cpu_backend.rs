//! CPU reference backend — a correctness-first interpreter of the backend-agnostic
//! [`infr_core`] compute [`Graph`]. Projection matmuls and attention use rayon for multi-core
//! parallelism; QK/PV inner loops use an 8-accumulator dot for AVX autovectorization.
//! Weights are read **zero-copy from the GGUF mmap** (no `memcpy`, no owned RAM): the bulk
//! projection weights (`Op::Linear`) are dequantized one row at a time straight from the mapping
//! inside the dot, so 12B / MoE models cost only their on-disk size in page cache. Only the tiny
//! norm weights are dequant-cached; the model writes (KV / conv / recurrent state, per-step IO) use
//! small owned buffers. It exists to (a) run every model without a GPU and (b) serve as the oracle
//! the GPU backends are validated against.
//!
//! Lives in `infr-llama` for now (next to [`crate::dequant_block`] + the qwen35 CPU oracle) to
//! avoid a circular crate dep; it implements the agnostic `infr_core::Backend` trait, so it can be
//! extracted to an `infr-cpu` crate later without touching callers.

use crate::{dequant_block, Config, PerLayerEmbd};
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Capabilities, Plan};
use infr_core::error::Result;
use infr_core::graph::{Activation, AttnMask, Graph, Op, TensorKind};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::WeightSource;
use infr_gguf::{Gguf, TensorBytes};
use rayon::prelude::*;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Timing/counts from a CPU generation, for the caller's stats line.
#[derive(Debug, Clone, Copy)]
pub struct CpuStats {
    pub n_prompt: usize,
    pub prompt_secs: f64,
    pub n_gen: usize,
    pub decode_secs: f64,
}

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
        let d = crate::rdf16(&blk[0..2]);
        let dmin = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        for s in 0..8usize {
            let (sc, m) = crate::k4(s, scales);
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
        let d = crate::rdf16(&blk[0..2]);
        let dmin = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        for s in 0..8usize {
            let (sc, m) = crate::k4(s, scales);
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
        let d = crate::rdf16(&blk[0..2]);
        let dmin = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        // k=0..4 covers sub-block pairs (0,1), (2,3), (4,5), (6,7).
        for k in 0..4usize {
            let (sc_e, m_e) = crate::k4(2 * k, scales);
            let (sc_o, m_o) = crate::k4(2 * k + 1, scales);
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
        let d = crate::rdf16(&blk[208..210]);
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
        let d = crate::rdf16(&blk[208..210]);
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
        let d = crate::rdf16(&blk[208..210]);
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
            let d_w = crate::rdf16(&blk[0..2]);
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
            let d_w = crate::rdf16(&blk[0..2]);
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
            let d_w0 = crate::rdf16(&row[wb0 * bpr..wb0 * bpr + 2]);
            let d_w1 = crate::rdf16(&row[wb1 * bpr..wb1 * bpr + 2]);
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
        let d = crate::rdf16(&blk[0..2]);
        let dmin = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in 0..4usize {
            let (sc_e, m_e) = crate::k4(2 * j, scales);
            let (sc_o, m_o) = crate::k4(2 * j + 1, scales);
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
        let d = crate::rdf16(&blk[0..2]);
        let dmin = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let qh_ymm = _mm256_loadu_si256(qh.as_ptr() as *const __m256i);
        let (mut sd, mut sm) = (0i32, 0i32);
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in 0..4usize {
            let (sc_e, m_e) = crate::k4(2 * j, scales);
            let (sc_o, m_o) = crate::k4(2 * j + 1, scales);
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
        let d = crate::rdf16(&blk[0..2]);
        let dmin = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let qh_ymm = _mm256_loadu_si256(qh.as_ptr() as *const __m256i);
        let (mut sd, mut sm) = (0i32, 0i32);
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for k in 0..4usize {
            let (sc_e, m_e) = crate::k4(2 * k, scales);
            let (sc_o, m_o) = crate::k4(2 * k + 1, scales);
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
        d_arr[b] = crate::rdf16(&blk[0..2]);
        dmin_arr[b] = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for s in 0..8usize {
            let (sc, mv) = crate::k4(s, scales);
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
        d_arr[b] = crate::rdf16(&blk[0..2]);
        dmin_arr[b] = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for s in 0..8usize {
            let (sc, mv) = crate::k4(s, scales);
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
        d_arr[b] = crate::rdf16(&blk[0..2]);
        dmin_arr[b] = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for s in 0..8usize {
            let (sc, mv) = crate::k4(s, scales);
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
        d_a[b] = crate::rdf16(&blk_a[0..2]);
        dmin_a[b] = crate::rdf16(&blk_a[2..4]);
        let scales_a = &blk_a[4..16];
        let qs_a = &blk_a[16..144];
        for s in 0..8usize {
            let (sc, mv) = crate::k4(s, scales_a);
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
        d_b[b] = crate::rdf16(&blk_b[0..2]);
        dmin_b[b] = crate::rdf16(&blk_b[2..4]);
        let scales_b = &blk_b[4..16];
        let qs_b = &blk_b[16..144];
        for s in 0..8usize {
            let (sc, mv) = crate::k4(s, scales_b);
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
        d_arr[b] = crate::rdf16(&blk[208..210]);
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
        d_arr[b] = crate::rdf16(&blk[208..210]);
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
        d_arr[b] = crate::rdf16(&blk[208..210]);
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
            dw_arr[b * 8 + s] = crate::rdf16(&blk[0..2]);
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
            dw_arr[b * 8 + s] = crate::rdf16(&blk[0..2]);
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
            dw_arr[b * 8 + s] = crate::rdf16(&blk[0..2]);
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
        d_arr[b] = crate::rdf16(&blk[0..2]);
        dmin_arr[b] = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in 0..4usize {
            let (sc_e, m_e) = crate::k4(2 * j, scales);
            let (sc_o, m_o) = crate::k4(2 * j + 1, scales);
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
        d_arr[b] = crate::rdf16(&blk[0..2]);
        dmin_arr[b] = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        for s in 0..8usize {
            let (sc, mv) = crate::k4(s, scales);
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
        d_arr[b] = crate::rdf16(&blk[0..2]);
        dmin_arr[b] = crate::rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        for s in 0..8usize {
            let (sc, mv) = crate::k4(s, scales);
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

/// A compiled plan = the owned graph (the CPU "compiles" nothing; it interprets at execute time).
pub struct CpuPlan {
    graph: Graph,
}

impl Plan for CpuPlan {
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

/// Op variant name, for INFR_PROF_OPS per-op-type timing.
fn op_kind(op: &Op) -> &'static str {
    match op {
        Op::RmsNorm { .. } => "RmsNorm",
        Op::QkNorm { .. } => "QkNorm",
        Op::Linear { .. } => "Linear",
        Op::Rope { .. } => "Rope",
        Op::QkNormRope { .. } => "QkNormRope",
        Op::WriteKv { .. } => "WriteKv",
        Op::Attention { .. } => "Attention",
        Op::GatedAct { .. } => "GatedAct",
        Op::MoeFfn { .. } => "MoeFfn",
        Op::Conv1dSilu { .. } => "Conv1dSilu",
        Op::DeltaNet { .. } => "DeltaNet",
        Op::Add { .. } => "Add",
        Op::Scale { .. } => "Scale",
        Op::Softcap { .. } => "Softcap",
        Op::Copy { .. } => "Copy",
    }
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
            unified_memory: true,
        }
    }

    fn alloc(&self, bytes: usize, _usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        Ok(Box::new(CpuBuffer::Owned(Mutex::new(vec![
            0u8;
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
        Ok(Box::new(CpuPlan {
            graph: graph.clone(),
        }))
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        let g = &plan
            .as_any()
            .downcast_ref::<CpuPlan>()
            .expect("cpu backend: plan is not a CpuPlan")
            .graph;

        // f32 working store for every Input/Internal/Output handle (weights are read on demand:
        // norms via the small dequant cache, `Op::Linear` weights streamed row-by-row).
        let mut vals: Vec<Vec<f32>> = vec![Vec::new(); g.tensors.len()];
        // KV-cache tensors (the `cache` of `WriteKv`, the `k_cache`/`v_cache` of `Attention`) are
        // accessed straight from their bound buffers — `WriteKv` writes one row, `Attention` reads
        // `kv_len` rows. They're sized for the WHOLE context (`max_ctx`), so loading them into `vals`
        // (and writing them back) each token would cost O(max_ctx) memory traffic per token instead of
        // O(kv_len) — catastrophic at a large `max_new`. Skip the round-trip for them.
        let mut direct: HashSet<u32> = HashSet::new();
        for op in &g.ops {
            match op {
                Op::WriteKv { cache, .. } => {
                    direct.insert(cache.0);
                }
                Op::Attention {
                    k_cache, v_cache, ..
                } => {
                    direct.insert(k_cache.0);
                    direct.insert(v_cache.0);
                }
                _ => {}
            }
        }
        for (i, decl) in g.tensors.iter().enumerate() {
            match decl.kind {
                TensorKind::Internal | TensorKind::Output => {
                    vals[i] = vec![0f32; decl.desc.numel()]
                }
                TensorKind::Input if direct.contains(&(i as u32)) => {} // read/written in place
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
                } => {
                    let (m, in_f, out_f) = (m as usize, in_f as usize, out_f as usize);
                    let xs = &vals[x.0 as usize];
                    // Stream the (row-major [out_f, in_f]) weight one row at a time straight from the
                    // mmap, dequantizing inside the dot — no full f32 materialization. GGUF rows are
                    // block-aligned, so each row is an equal `bytes/out_f` slice. Output rows are
                    // independent → fan out over the 32 cores with rayon.
                    let buf = bindings.get(w).expect("cpu backend: unbound Weight");
                    let bytes = cpu_buf(buf).read();
                    let wbytes: &[u8] = &bytes;
                    let dt = g.desc(w).dtype;
                    let bpr = wbytes.len() / out_f; // bytes per weight row
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
                        // For Q4_K, use 2-row tiling: each rayon task handles two consecutive
                        // output rows and shares the Q8 activation loads between them.
                        // This halves the L3 bandwidth for activation reads (dominant bottleneck).
                        if dt == DType::Q4K && out_f >= 2 {
                            let pairs = out_f / 2;
                            let (even_t, odd_t) = out_t.split_at_mut(pairs * 2 * m);
                            // Process pairs of output rows together.
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
                            // Handle last row if out_f is odd.
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
                    let hf = rd / 2;
                    for r in 0..rows {
                        let p0 = pos[r];
                        for h in 0..nh {
                            let b = (r * nh + h) * hd;
                            for p in 0..hf {
                                let (i0, i1) = (p, p + hf);
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
                    let xs = vals[x.0 as usize].clone();
                    let ws = weight(w);
                    let pos = vals[positions.0 as usize].clone();
                    let ff = freq_factors.map(|f| vals[f.0 as usize].clone());
                    let mut out = vec![0f32; rows * nh * hd];
                    let hf = rd / 2;
                    for r in 0..rows {
                        let p0 = pos[r];
                        for h in 0..nh {
                            let b = (r * nh + h) * hd;
                            let ss: f32 =
                                (0..hd).map(|i| xs[b + i] * xs[b + i]).sum::<f32>() / hd as f32;
                            let s = 1.0 / (ss + eps).sqrt();
                            for i in 0..hd {
                                out[b + i] = xs[b + i] * s * ws[i];
                            }
                            for p in 0..hf {
                                let (i0, i1) = (p, p + hf);
                                let mut ang = p0 * theta.powf(-2.0 * p as f32 / rd as f32);
                                if let Some(ff) = &ff {
                                    ang /= ff[p];
                                }
                                let (sn, c) = (ang.sin(), ang.cos());
                                let a = out[b + i0];
                                let bb = out[b + i1];
                                out[b + i0] = a * c - bb * sn;
                                out[b + i1] = a * sn + bb * c;
                            }
                        }
                    }
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
                    // Materialize the valid KV prefix (`kv_len` rows) as f32, dequantizing an f16
                    // cache (matches the GPU's f16 KV) — the inner dot then runs in f32 either way.
                    let need = kv_len * nkv * hd;
                    let (ks, vs): (Vec<f32>, Vec<f32>) = match g.desc(k_cache).dtype {
                        DType::F16 => {
                            let f = |b: &[u8]| -> Vec<f32> {
                                bytemuck::cast_slice::<u8, u16>(b)[..need]
                                    .iter()
                                    .map(|&x| half::f16::from_bits(x).to_f32())
                                    .collect()
                            };
                            (f(&kguard), f(&vguard))
                        }
                        _ => (
                            bytemuck::cast_slice::<u8, f32>(&kguard)[..need].to_vec(),
                            bytemuck::cast_slice::<u8, f32>(&vguard)[..need].to_vec(),
                        ),
                    };
                    let group = nh / nkv;
                    let window = match mask {
                        AttnMask::Causal => 0usize,
                        AttnMask::SlidingWindow(w) => w,
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
                                                         // visible keys: [lo, abs] (causal); SWA clips lo to abs-window+1.
                            let lo = if window > 0 && abs + 1 > window {
                                abs + 1 - window
                            } else {
                                0
                            };
                            let n_keys = abs + 1 - lo;
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
                Op::Scale { x, dst, s, n } => {
                    let n = n as usize;
                    let xs = vals[x.0 as usize].clone();
                    let mut out = vec![0f32; n];
                    for i in 0..n {
                        out[i] = xs[i] * s;
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
                Op::MoeFfn {
                    x,
                    router,
                    gate_exps,
                    up_exps,
                    down_exps,
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
                    // Router softmax over all experts.
                    let rbuf = bindings.get(router).expect("cpu backend: unbound router");
                    let rbytes = cpu_buf(rbuf).read();
                    let logits = matvec(&rbytes, g.desc(router).dtype, &xs, ne, n_expert);
                    drop(rbytes);
                    let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                    let psum: f32 = probs.iter().sum();
                    for p in probs.iter_mut() {
                        *p /= psum;
                    }
                    // Top-`n_used` experts, renormalized weights.
                    let mut idx: Vec<usize> = (0..n_expert).collect();
                    idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                    idx.truncate(n_used);
                    let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                    // Per-expert stacked-weight byte slices.
                    let gbuf = bindings
                        .get(gate_exps)
                        .expect("cpu backend: unbound gate_exps");
                    let ubuf = bindings.get(up_exps).expect("cpu backend: unbound up_exps");
                    let dbuf = bindings
                        .get(down_exps)
                        .expect("cpu backend: unbound down_exps");
                    let gb = cpu_buf(gbuf).read();
                    let ub = cpu_buf(ubuf).read();
                    let db = cpu_buf(dbuf).read();
                    let (gdt, udt, ddt) = (
                        g.desc(gate_exps).dtype,
                        g.desc(up_exps).dtype,
                        g.desc(down_exps).dtype,
                    );
                    let (gst, ust, dst_) = (
                        gb.len() / n_expert,
                        ub.len() / n_expert,
                        db.len() / n_expert,
                    );
                    let mut out = vec![0f32; ne];
                    for &e in &idx {
                        let gate = matvec(&gb[e * gst..(e + 1) * gst], gdt, &xs, ne, nffx);
                        let up = matvec(&ub[e * ust..(e + 1) * ust], udt, &xs, ne, nffx);
                        let actv: Vec<f32> =
                            (0..nffx).map(|i| act_fn(act, gate[i]) * up[i]).collect();
                        let y = matvec(&db[e * dst_..(e + 1) * dst_], ddt, &actv, nffx, ne);
                        let w_e = probs[e] / wsum * scale;
                        for i in 0..ne {
                            out[i] += w_e * y[i];
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Conv1dSilu {
                    x,
                    weight: w,
                    state,
                    dst,
                    channels,
                    kernel,
                } => {
                    let (cc, kk) = (channels as usize, kernel as usize);
                    let xs = vals[x.0 as usize].clone();
                    let ws = weight(w); // [channels, kernel] row-major (per-channel kernel)
                    let st = &mut vals[state.0 as usize]; // [(kernel-1), channels], oldest row first
                    let mut out = vec![0f32; cc];
                    for ch in 0..cc {
                        // window = [history rows.. , current x]; tap j uses weight[ch*kk + j].
                        let mut acc = 0f32;
                        for j in 0..kk - 1 {
                            acc += st[j * cc + ch] * ws[ch * kk + j];
                        }
                        acc += xs[ch] * ws[ch * kk + (kk - 1)];
                        out[ch] = acc / (1.0 + (-acc).exp()); // silu
                    }
                    // shift history (drop oldest, append raw x).
                    for j in 0..kk.saturating_sub(2) {
                        for ch in 0..cc {
                            st[j * cc + ch] = st[(j + 1) * cc + ch];
                        }
                    }
                    if kk >= 2 {
                        for ch in 0..cc {
                            st[(kk - 2) * cc + ch] = xs[ch];
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
                    n_vhead,
                    n_khead,
                    head_k,
                    head_v,
                    eps,
                } => {
                    let (nv, nk, kd, vd) = (
                        n_vhead as usize,
                        n_khead as usize,
                        head_k as usize,
                        head_v as usize,
                    );
                    let qf = vals[q.0 as usize].clone();
                    let kf = vals[k.0 as usize].clone();
                    let vf = vals[v.0 as usize].clone();
                    let bf = vals[b.0 as usize].clone();
                    let af = vals[a.0 as usize].clone();
                    let acoef = weight(a_coef);
                    let dtb = weight(dt_bias);
                    let st = &mut vals[state.0 as usize]; // [nv, kd, vd]
                    let mut out = vec![0f32; nv * vd];
                    let qscale = 1.0 / (kd as f32).sqrt();
                    let l2 = |slice: &[f32]| -> f32 {
                        (slice.iter().map(|x| x * x).sum::<f32>() + eps).sqrt()
                    };
                    for h in 0..nv {
                        // GQA: q/k heads are TILED to nv value heads → v-head h uses q/k head h % nk.
                        let kh_idx = h % nk;
                        let mut qh = qf[kh_idx * kd..kh_idx * kd + kd].to_vec();
                        let mut kh = kf[kh_idx * kd..kh_idx * kd + kd].to_vec();
                        let vh = &vf[h * vd..h * vd + vd];
                        let qn = l2(&qh);
                        let kn = l2(&kh);
                        for x in qh.iter_mut() {
                            *x = *x / qn * qscale;
                        }
                        for x in kh.iter_mut() {
                            *x /= kn;
                        }
                        let beta = 1.0 / (1.0 + (-bf[h]).exp());
                        // softplus(a + dt_bias), then g = a_coef * softplus (≤ 0); decay = exp(g).
                        let sp = {
                            let z = af[h] + dtb[h];
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
                        let oh = &mut out[h * vd..h * vd + vd];
                        for kk in 0..kd {
                            let qv = qh[kk];
                            let row = &sh[kk * vd..kk * vd + vd];
                            for d in 0..vd {
                                oh[d] += qv * row[d];
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
            }
            if let Some(t0) = __t0 {
                *op_times.entry(op_kind(op)).or_insert(0.0) += t0.elapsed().as_secs_f64();
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
                    && !direct.contains(&(i as u32)));
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

// ─── Qwen3 dense CPU decode runner ───────────────────────────────────────────────
//
// Builds the n=1 decode Graph and drives it through `CpuBackend`, one token at a time, for BOTH
// prompt ingestion (looped) and generation — so no GEMM/flash prefill kernels are needed on CPU.
// The KV cache grows one row per step. Validates the agnostic seam end-to-end against the GPU path.

/// FFN weight handles: a dense gated FFN, or a qwen3moe routed-expert bank (router + stacked
/// per-expert gate/up/down).
enum FfnW {
    Dense {
        wgate: TensorId,
        wup: TensorId,
        wdown: TensorId,
    },
    Moe {
        router: TensorId,
        gate_exps: TensorId,
        up_exps: TensorId,
        down_exps: TensorId,
    },
}

/// Per-layer weight handles captured while building one decode graph (q/k-norm + the gemma
/// sandwich norms are optional; `wv` is absent on gemma4 full-attention layers, which reuse the raw
/// K projection as V). The order they're declared in MUST match the upload order so `weights[i]`
/// binds to `wbufs[i]`.
struct LayerW {
    attn_norm: TensorId,
    wq: TensorId,
    wk: TensorId,
    wv: Option<TensorId>,
    q_norm: Option<TensorId>,
    k_norm: Option<TensorId>,
    wo: TensorId,
    post_attn: Option<TensorId>,
    ffn_norm: TensorId,
    ffn: FfnW,
    post_ffw: Option<TensorId>,
    // gemma4 E2B per-layer input embedding: inp_gate, proj, post_norm.
    pl_inp_gate: Option<TensorId>,
    pl_proj: Option<TensorId>,
    pl_post_norm: Option<TensorId>,
}

/// Handles into one freshly-built decode graph that the driver re-binds each step.
struct DecodeHandles {
    hidden: TensorId,
    positions: TensorId,
    rope_freqs: Option<TensorId>, // gemma4 proportional-RoPE divisors (full-attention layers)
    per_layer_inp: Option<TensorId>, // gemma4 E2B per-(token,layer) input vector `[n_layer*npl]`
    logits: TensorId,
    k_cache: Vec<TensorId>,
    v_cache: Vec<TensorId>,
    weights: Vec<TensorId>, // flat, in declaration == upload order
}

/// Greedy CPU generation for a decoder (Qwen3 / Llama / Gemma 3 / Gemma 4 dense+E2B / qwen3moe). The
/// attention block is shared; the FFN is either a dense gated FFN or a routed-expert MoE bank; gemma4
/// E2B adds per-layer input embeddings + KV-layer sharing. `prompt` is the full token prefix; returns
/// the generated continuation. Stops at EOS or `max_new`.
pub(crate) fn generate_dense_cpu(
    g: &Gguf,
    cfg: &Config,
    token_embd: &[f32],
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, CpuStats)> {
    // Thin CPU wrapper over the backend-generic runner: a CpuBackend + a zero-copy weight binder
    // (maps each tensor straight from the GGUF mmap — no alloc, no memcpy).
    let cpu_be = CpuBackend::new();
    generate_dense_backend(
        &cpu_be,
        &|tb, _dt, _n| Ok(cpu_be.map_weight(tb)),
        g,
        cfg,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
    )
}

/// Backend-generic dense decode runner. Builds the agnostic decode [`Graph`] per token and runs it
/// on `be` (CPU reference or Vulkan). `bind_weight` turns each native-dtype GGUF tensor into a
/// backend buffer: the CPU maps it zero-copy from the mmap; the GPU pads + uploads it to VRAM. This
/// is the single forward both backends share — running it on Vulkan and diffing the CPU oracle is
/// the end-to-end dense parity check.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_dense_backend(
    be: &dyn Backend,
    bind_weight: &dyn Fn(TensorBytes, DType, usize) -> AResult<Box<dyn Buffer>>,
    g: &Gguf,
    cfg: &Config,
    token_embd: &[f32],
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    mut on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, CpuStats)> {
    let c = cfg;
    let (ne, nh) = (c.n_embd, c.n_head);
    // gemma4: per-layer SWA/full dims differ; size shared scratch + KV by the max over layers.
    let max_hd = c.max_head_dim();
    let max_kvrow = c.max_n_kv() * max_hd;
    let max_qrow = nh * max_hd;
    let nff = c.n_ff; // max FFN width
    let gemma = c.gemma;
    let gemma4 = c.gemma4;
    let qk_norm = c.qk_norm;
    let act = if gemma {
        Activation::Gelu
    } else {
        Activation::Silu
    };
    let max_ctx = prompt.len() + max_new + 1;
    // gemma4 E2B (gemma3n): per-layer input embeddings + KV-layer sharing.
    let e2b = c.n_embd_per_layer > 0;
    let npl = c.n_embd_per_layer;

    // Per-layer presence of an explicit V projection. gemma4 full-attention layers omit it (V = the
    // raw K projection); every layer of every other model has one.
    let has_wv: Vec<bool> = (0..c.n_layer)
        .map(|l| {
            g.tensors()
                .iter()
                .any(|t| t.name == format!("blk.{l}.attn_v.weight"))
        })
        .collect();
    // gemma4 per-layer output scale (`layer_output_scale.weight`, a single scalar multiplying the
    // layer output before the next layer). Read host-side; applied as an `Op::Scale`.
    let out_scale: Vec<Option<f32>> = (0..c.n_layer)
        .map(|l| {
            let name = format!("blk.{l}.layer_output_scale.weight");
            if g.tensors().iter().any(|t| t.name == name) {
                crate::load_tensor_dequant(g, &name)
                    .ok()
                    .and_then(|(v, _)| v.first().copied())
            } else {
                None
            }
        })
        .collect();
    // gemma4 proportional-RoPE frequency divisors (`rope_freqs.weight`, `[rope_dim/2]`): applied on
    // full-attention layers only (SWA layers use plain RoPE). Bound as a per-step f32 Input.
    let rope_freqs: Option<Vec<f32>> =
        if gemma4 && g.tensors().iter().any(|t| t.name == "rope_freqs.weight") {
            Some(crate::load_tensor_dequant(g, "rope_freqs.weight").map(|(v, _)| v)?)
        } else {
            None
        };

    // ── upload weights in their NATIVE GGUF dtype (no host pre-dequant — the backend dequants
    //    lazily in `bytes_to_f32`, so a quant weight occupies ~quant size, not 8× f32). `wspecs`
    //    records each (dtype, numel) so `build` can declare the handle with the matching dtype; its
    //    order MUST equal the `g.weight()` order in `build` below. ──────────────────────────────────
    let mut wbufs: Vec<Box<dyn Buffer>> = Vec::new();
    let mut wspecs: Vec<(DType, usize)> = Vec::new();
    // Map a weight tensor zero-copy from the GGUF mmap (no alloc, no memcpy); record its native dtype
    // + element count so `build` declares the handle to match.
    let mut wraw = |name: &str| -> AResult<()> {
        let info = g
            .tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?
            .clone();
        let numel: usize = info.shape.iter().product();
        let tb = g.tensor_bytes_arc(name).map_err(|e| anyhow!("{e}"))?;
        wbufs.push(bind_weight(tb, info.dtype, numel)?);
        wspecs.push((info.dtype, numel));
        Ok(())
    };
    for l in 0..c.n_layer {
        let p = |s: &str| format!("blk.{l}.{s}");
        wraw(&p("attn_norm.weight"))?;
        wraw(&p("attn_q.weight"))?;
        wraw(&p("attn_k.weight"))?;
        if has_wv[l] {
            wraw(&p("attn_v.weight"))?;
        }
        if qk_norm {
            wraw(&p("attn_q_norm.weight"))?;
            wraw(&p("attn_k_norm.weight"))?;
        }
        wraw(&p("attn_output.weight"))?;
        if gemma {
            wraw(&p("post_attention_norm.weight"))?;
        }
        wraw(&p("ffn_norm.weight"))?;
        if c.moe.is_some() {
            // qwen3moe: router + stacked per-expert gate/up/down banks.
            wraw(&p("ffn_gate_inp.weight"))?;
            wraw(&p("ffn_gate_exps.weight"))?;
            wraw(&p("ffn_up_exps.weight"))?;
            wraw(&p("ffn_down_exps.weight"))?;
        } else {
            wraw(&p("ffn_gate.weight"))?;
            wraw(&p("ffn_up.weight"))?;
            wraw(&p("ffn_down.weight"))?;
        }
        if gemma {
            wraw(&p("post_ffw_norm.weight"))?;
        }
        if e2b {
            // gemma4 E2B per-layer input-embedding application weights.
            wraw(&p("inp_gate.weight"))?;
            wraw(&p("proj.weight"))?;
            wraw(&p("post_norm.weight"))?;
        }
    }
    // Globals: output_norm, lm_head. lm_head = `output.weight`, or (tied) the quantized
    // `token_embd.weight` mapped from the mmap and dequantized per-row by `Op::Linear` — same f32
    // values as the host `token_embd`, but zero-copy.
    wraw("output_norm.weight")?;
    if g.tensors().iter().any(|t| t.name == "output.weight") {
        wraw("output.weight")?;
    } else {
        wraw("token_embd.weight")?;
    }
    // gemma4 weightless per-head V-norm = `QkNorm` with a unit weight (out = x/rms). One ones-vector
    // of the max head dim serves every layer (a narrower layer reads its leading prefix).
    if gemma4 {
        let ones = vec![1.0f32; max_hd];
        let b = be
            .alloc(ones.len() * 4, BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(b.as_ref(), bytemuck::cast_slice(&ones))
            .map_err(|e| anyhow!("{e}"))?;
        wbufs.push(b);
        wspecs.push((DType::F32, max_hd));
    }

    // ── persistent KV cache buffers (f32), sized per-layer (gemma4 SWA layers are narrower) ───────
    let mut kbufs: Vec<Box<dyn Buffer>> = Vec::new();
    let mut vbufs: Vec<Box<dyn Buffer>> = Vec::new();
    for l in 0..c.n_layer {
        let kvrow_l = c.layer_n_kv(l) * c.layer_head_dim(l);
        // f16 KV cache (2 bytes/elem) — matches the graph's f16 k_cache/v_cache decls.
        kbufs.push(
            be.alloc(max_ctx * kvrow_l * 2, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?,
        );
        vbufs.push(
            be.alloc(max_ctx * kvrow_l * 2, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?,
        );
    }

    // ── per-step IO buffers ────────────────────────────────────────────────────────
    let hidden_buf = be
        .alloc(ne * 4, BufferUsage::Staging)
        .map_err(|e| anyhow!("{e}"))?;
    let pos_buf = be
        .alloc(4, BufferUsage::Staging)
        .map_err(|e| anyhow!("{e}"))?;
    let rf_buf = match &rope_freqs {
        Some(rf) => {
            let b = be
                .alloc(rf.len() * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(b.as_ref(), bytemuck::cast_slice(rf))
                .map_err(|e| anyhow!("{e}"))?;
            Some((b, rf.len()))
        }
        None => None,
    };
    // gemma4 E2B per-(token,layer) input vector `[n_layer*npl]`, recomputed + re-uploaded each step.
    let ipl_buf = if e2b {
        Some(
            be.alloc(c.n_layer * npl * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?,
        )
    } else {
        None
    };
    let logits_buf = be
        .alloc(c.vocab * 4, BufferUsage::Readback)
        .map_err(|e| anyhow!("{e}"))?;

    // Build a forward graph for `batch` tokens starting at absolute position `start_pos`.
    // `batch = 1` is the normal decode path; `batch > 1` is the batched-prefill path.
    // Scratch tensors scale by `batch`; the LM head always runs on the LAST token only
    // (extracted via Op::Copy for batch > 1) so the logits output is always [vocab].
    let build = |batch: usize, start_pos: usize| -> (Graph, DecodeHandles) {
        let mut g = Graph::new();
        let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
        // KV cache is f16 — matches the GPU's f16 cache (halves memory, tightens CPU↔GPU parity).
        let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16);
        let hidden = g.input(f32d(batch * ne));
        let positions = g.input(TensorDesc::new(vec![batch], DType::I32));
        let rope_freqs = rf_buf.as_ref().map(|(_, n)| g.input(f32d(*n)));
        // gemma4 E2B per-(token,layer) input vector `[n_layer*npl]` (computed host-side each step).
        let per_layer_inp = if e2b {
            Some(g.input(f32d(c.n_layer * npl)))
        } else {
            None
        };
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        for l in 0..c.n_layer {
            let kvrow_l = c.layer_n_kv(l) * c.layer_head_dim(l);
            k_cache.push(g.input(f16d(max_ctx * kvrow_l)));
            v_cache.push(g.input(f16d(max_ctx * kvrow_l)));
        }

        // Weights — declared in the SAME order as the upload loop, pulling (dtype, numel) from
        // `wspecs` so each handle carries its native GGUF dtype (the backend dequants on read).
        // `wpush` records the handle in the flat `weights` list (for binding) and returns it.
        let mut weights: Vec<TensorId> = Vec::new();
        let mut wi = 0usize;
        let mut wpush = |g: &mut Graph, weights: &mut Vec<TensorId>| -> TensorId {
            let (dt, n) = wspecs[wi];
            wi += 1;
            let id = g.weight(TensorDesc::new(vec![n], dt));
            weights.push(id);
            id
        };
        let mut lw: Vec<LayerW> = Vec::new();
        for l in 0..c.n_layer {
            let attn_norm = wpush(&mut g, &mut weights);
            let wq = wpush(&mut g, &mut weights);
            let wk = wpush(&mut g, &mut weights);
            let wv = if has_wv[l] {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let (q_norm, k_norm) = if qk_norm {
                (
                    Some(wpush(&mut g, &mut weights)),
                    Some(wpush(&mut g, &mut weights)),
                )
            } else {
                (None, None)
            };
            let wo = wpush(&mut g, &mut weights);
            let post_attn = if gemma {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let ffn_norm = wpush(&mut g, &mut weights);
            let ffn = if c.moe.is_some() {
                FfnW::Moe {
                    router: wpush(&mut g, &mut weights),
                    gate_exps: wpush(&mut g, &mut weights),
                    up_exps: wpush(&mut g, &mut weights),
                    down_exps: wpush(&mut g, &mut weights),
                }
            } else {
                FfnW::Dense {
                    wgate: wpush(&mut g, &mut weights),
                    wup: wpush(&mut g, &mut weights),
                    wdown: wpush(&mut g, &mut weights),
                }
            };
            let post_ffw = if gemma {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let (pl_inp_gate, pl_proj, pl_post_norm) = if e2b {
                (
                    Some(wpush(&mut g, &mut weights)),
                    Some(wpush(&mut g, &mut weights)),
                    Some(wpush(&mut g, &mut weights)),
                )
            } else {
                (None, None, None)
            };
            lw.push(LayerW {
                attn_norm,
                wq,
                wk,
                wv,
                q_norm,
                k_norm,
                wo,
                post_attn,
                ffn_norm,
                ffn,
                post_ffw,
                pl_inp_gate,
                pl_proj,
                pl_post_norm,
            });
        }
        let w_out_norm = wpush(&mut g, &mut weights);
        let w_lm = wpush(&mut g, &mut weights);
        let v_ones = if gemma4 {
            Some(wpush(&mut g, &mut weights))
        } else {
            None
        };
        let logits = g.output(f32d(c.vocab));

        // scratch (sized to the per-layer max × batch; ops reallocate dst, so these are upper bounds)
        let hn = g.internal(f32d(batch * ne));
        let q = g.internal(f32d(batch * max_qrow));
        let k = g.internal(f32d(batch * max_kvrow));
        let v = g.internal(f32d(batch * max_kvrow));
        // QkNorm+RoPE writes f16 (the GPU `qk_norm_rope` is f32-in→f16-out, can't be in place; the GPU
        // attention reads f16 q). q16/k16 hold the f16 normed+roped q/k for the q/k-norm (qwen3/gemma)
        // path; the llama RoPE-only path stays in f32 q/k. Free on the CPU (its store is f32 regardless).
        let q16 = g.internal(f16d(batch * max_qrow));
        let k16 = g.internal(f16d(batch * max_kvrow));
        let attn = g.internal(f32d(batch * max_qrow));
        let gbuf = g.internal(f32d(batch * nff));
        let ubuf = g.internal(f32d(batch * nff));
        let actbuf = g.internal(f32d(batch * nff));
        let sub = g.internal(f32d(batch * ne));
        // E2B per-layer embed scratch: gate `[npl]` and projected `[ne]`.
        let plg = g.internal(f32d(batch * npl.max(1)));
        let plp = g.internal(f32d(batch * ne));

        let eps = c.rms_eps;
        for (l, lw) in lw.iter().enumerate() {
            // Per-layer dims (gemma4 SWA vs full; uniform for every other model).
            let hd = c.layer_head_dim(l);
            let nkv = c.layer_n_kv(l);
            let kvrow = nkv * hd;
            let qrow = nh * hd;
            let nff_l = c.layer_n_ff(l);
            let theta = c.layer_rope_theta(l); // gemma dual-rope (SWA 1e4 / full 1e6); uniform else
            let rope_dim = c.layer_rope_dim(l);
            let swa = gemma && c.is_swa_layer(l);
            let mask = if swa {
                AttnMask::SlidingWindow(c.swa_window)
            } else {
                AttnMask::Causal
            };
            // gemma4: attn scale 1.0 (QK-norm controls magnitude); everyone else 1/√hd.
            let scale = if gemma4 {
                1.0
            } else {
                1.0 / (hd as f32).sqrt()
            };
            // gemma4 proportional-RoPE applies only on full-attention layers.
            let layer_ff = if gemma4 && !swa { rope_freqs } else { None };
            // attn input norm
            g.push(Op::RmsNorm {
                x: hidden,
                weight: lw.attn_norm,
                dst: hn,
                rows: batch as u32,
                dim: ne as u32,
                eps,
            });
            g.push(Op::Linear {
                x: hn,
                weight: lw.wq,
                dst: q,
                m: batch as u32,
                in_f: ne as u32,
                out_f: qrow as u32,
            });
            // gemma4 E2B KV-layer sharing: shared layers compute Q only and attend to an earlier
            // layer's cache. `own_kv`/`kv_src` are `true`/`l` for every layer of a non-sharing model.
            let own_kv = c.has_own_kv(l);
            let kv_src = c.kv_src_layer(l);
            if own_kv {
                g.push(Op::Linear {
                    x: hn,
                    weight: lw.wk,
                    dst: k,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: kvrow as u32,
                });
                // V projection, or (gemma4 full layers) V = the raw K projection, copied BEFORE K is
                // QK-normed + RoPE'd.
                match lw.wv {
                    Some(wv) => g.push(Op::Linear {
                        x: hn,
                        weight: wv,
                        dst: v,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: kvrow as u32,
                    }),
                    None => g.push(Op::Copy {
                        src: k,
                        src_off: 0,
                        dst: v,
                        dst_off: 0,
                        n: (batch * kvrow) as u32,
                    }),
                }
                // K: fused QkNorm+RoPE (qwen3/gemma) → f16 `k16`, else RoPE alone (llama) in-place f32.
                let k_write = match lw.k_norm {
                    Some(kn) => {
                        g.push(Op::QkNormRope {
                            x: k,
                            weight: kn,
                            positions,
                            dst: k16,
                            rows: batch as u32,
                            n_head: nkv as u32,
                            head_dim: hd as u32,
                            rope_dim: rope_dim as u32,
                            theta,
                            eps,
                            freq_factors: layer_ff,
                        });
                        k16
                    }
                    None => {
                        g.push(Op::Rope {
                            x: k,
                            positions,
                            dst: k,
                            rows: batch as u32,
                            n_head: nkv as u32,
                            head_dim: hd as u32,
                            rope_dim: rope_dim as u32,
                            theta,
                            freq_factors: layer_ff,
                        });
                        k
                    }
                };
                // gemma4 weightless per-head RMSNorm on V (= x/rms) before caching.
                if let Some(ones) = v_ones {
                    g.push(Op::QkNorm {
                        x: v,
                        weight: ones,
                        dst: v,
                        rows: batch as u32,
                        n_head: nkv as u32,
                        head_dim: hd as u32,
                        eps,
                    });
                }
                g.push(Op::WriteKv {
                    src: k_write,
                    cache: k_cache[l],
                    rows: batch as u32,
                    row_stride: kvrow as u32,
                    pos: start_pos as u32,
                });
                g.push(Op::WriteKv {
                    src: v,
                    cache: v_cache[l],
                    rows: batch as u32,
                    row_stride: kvrow as u32,
                    pos: start_pos as u32,
                });
            }
            // Q: fused QkNorm+RoPE (qwen3/gemma) → f16 `q16`, else RoPE alone (llama) in-place f32.
            let q_attn = match lw.q_norm {
                Some(qn) => {
                    g.push(Op::QkNormRope {
                        x: q,
                        weight: qn,
                        positions,
                        dst: q16,
                        rows: batch as u32,
                        n_head: nh as u32,
                        head_dim: hd as u32,
                        rope_dim: rope_dim as u32,
                        theta,
                        eps,
                        freq_factors: layer_ff,
                    });
                    q16
                }
                None => {
                    g.push(Op::Rope {
                        x: q,
                        positions,
                        dst: q,
                        rows: batch as u32,
                        n_head: nh as u32,
                        head_dim: hd as u32,
                        rope_dim: rope_dim as u32,
                        theta,
                        freq_factors: layer_ff,
                    });
                    q
                }
            };
            g.push(Op::Attention {
                q: q_attn,
                k_cache: k_cache[kv_src],
                v_cache: v_cache[kv_src],
                dst: attn,
                rows: batch as u32,
                kv_len: (start_pos + batch) as u32,
                n_head: nh as u32,
                n_kv: nkv as u32,
                head_dim: hd as u32,
                scale,
                mask,
                pos: start_pos as u32,
            });
            g.push(Op::Linear {
                x: attn,
                weight: lw.wo,
                dst: sub,
                m: batch as u32,
                in_f: qrow as u32,
                out_f: ne as u32,
            });
            // gemma sandwich: post-attention norm on the sublayer output BEFORE the residual add.
            if let Some(pa) = lw.post_attn {
                g.push(Op::RmsNorm {
                    x: sub,
                    weight: pa,
                    dst: sub,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
            }
            g.push(Op::Add {
                a: hidden,
                b: sub,
                dst: hidden,
                n: (batch * ne) as u32,
            });
            // ffn
            g.push(Op::RmsNorm {
                x: hidden,
                weight: lw.ffn_norm,
                dst: hn,
                rows: batch as u32,
                dim: ne as u32,
                eps,
            });
            match lw.ffn {
                FfnW::Dense { wgate, wup, wdown } => {
                    g.push(Op::Linear {
                        x: hn,
                        weight: wgate,
                        dst: gbuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                    });
                    g.push(Op::Linear {
                        x: hn,
                        weight: wup,
                        dst: ubuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                    });
                    g.push(Op::GatedAct {
                        gate: gbuf,
                        up: ubuf,
                        dst: actbuf,
                        rows: batch as u32,
                        nff: nff_l as u32,
                        act,
                        up_off: 0,
                    });
                    g.push(Op::Linear {
                        x: actbuf,
                        weight: wdown,
                        dst: sub,
                        m: batch as u32,
                        in_f: nff_l as u32,
                        out_f: ne as u32,
                    });
                }
                FfnW::Moe {
                    router,
                    gate_exps,
                    up_exps,
                    down_exps,
                } => {
                    let mc = c.moe.expect("moe layer without MoeConfig");
                    g.push(Op::MoeFfn {
                        x: hn,
                        router,
                        gate_exps,
                        up_exps,
                        down_exps,
                        dst: sub,
                        ne: ne as u32,
                        n_expert: mc.n_expert as u32,
                        n_used: mc.n_used as u32,
                        n_ff_exp: mc.n_ff_exp as u32,
                        scale: mc.scale,
                        act, // qwen3moe: SwiGLU (act == Silu)
                    });
                }
            }
            if let Some(pf) = lw.post_ffw {
                g.push(Op::RmsNorm {
                    x: sub,
                    weight: pf,
                    dst: sub,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
            }
            g.push(Op::Add {
                a: hidden,
                b: sub,
                dst: hidden,
                n: (batch * ne) as u32,
            });
            // gemma4 E2B per-layer input embedding (gemma3n): mix this layer's input vector into
            // `hidden` after the FFN residual. `g = gelu(inp_gate·hidden) * inp_per_layer[l]`,
            // `p = post_norm(proj·g)`, `hidden += p`.
            if let (Some(gate_w), Some(proj_w), Some(post_norm), Some(ipl)) =
                (lw.pl_inp_gate, lw.pl_proj, lw.pl_post_norm, per_layer_inp)
            {
                g.push(Op::Linear {
                    x: hidden,
                    weight: gate_w,
                    dst: plg,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: npl as u32,
                });
                // gelu(plg) * ipl[l*npl .. l*npl+npl]  (the layer's slice of the input vector).
                g.push(Op::GatedAct {
                    gate: plg,
                    up: ipl,
                    dst: plg,
                    rows: batch as u32,
                    nff: npl as u32,
                    act: Activation::Gelu,
                    up_off: (l * npl) as u32,
                });
                g.push(Op::Linear {
                    x: plg,
                    weight: proj_w,
                    dst: plp,
                    m: batch as u32,
                    in_f: npl as u32,
                    out_f: ne as u32,
                });
                g.push(Op::RmsNorm {
                    x: plp,
                    weight: post_norm,
                    dst: plp,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
                g.push(Op::Add {
                    a: hidden,
                    b: plp,
                    dst: hidden,
                    n: (batch * ne) as u32,
                });
            }
            // gemma4: scale the whole layer output by the per-layer scalar before the next layer.
            if let Some(s) = out_scale[l] {
                g.push(Op::Scale {
                    x: hidden,
                    dst: hidden,
                    s,
                    n: (batch * ne) as u32,
                });
            }
        }
        g.push(Op::RmsNorm {
            x: hidden,
            weight: w_out_norm,
            dst: hn,
            rows: batch as u32,
            dim: ne as u32,
            eps,
        });
        // For batch > 1: the LM head runs only on the LAST token's hidden state — extract it
        // via Op::Copy before the projection so the logits output is always [vocab] regardless
        // of batch size. (For batch = 1, `hn` is already the single token's hidden state.)
        let lm_in = if batch > 1 {
            let hn_last = g.internal(f32d(ne));
            g.push(Op::Copy {
                src: hn,
                src_off: ((batch - 1) * ne) as u32,
                dst: hn_last,
                dst_off: 0,
                n: ne as u32,
            });
            hn_last
        } else {
            hn
        };
        g.push(Op::Linear {
            x: lm_in,
            weight: w_lm,
            dst: logits,
            m: 1,
            in_f: ne as u32,
            out_f: c.vocab as u32,
        });
        if c.final_softcap > 0.0 {
            g.push(Op::Softcap {
                x: logits,
                dst: logits,
                cap: c.final_softcap,
                n: c.vocab as u32,
            });
        }
        (
            g,
            DecodeHandles {
                hidden,
                positions,
                rope_freqs,
                per_layer_inp,
                logits,
                k_cache,
                v_cache,
                weights,
            },
        )
    };

    // ── drive ───────────────────────────────────────────────────────────────────────
    let embed_scale = if gemma { (ne as f32).sqrt() } else { 1.0 };
    let mut out = Vec::new();
    let mut cur = prompt.to_vec();
    let mut logits = vec![0f32; c.vocab];
    // INFR_PROF=1: report prompt-ingest + decode tok/s to stderr (CPU perf iteration).
    let prof = std::env::var("INFR_PROF").is_ok();
    let mut prompt_t = std::time::Duration::ZERO;
    let mut decode_t = std::time::Duration::ZERO;
    let mut decode_n = 0usize;

    // ── batched prefill (dense non-MoE non-E2B models only) ──────────────────────────────────
    // Process all-but-the-last prompt tokens in a single graph execution: each Op::Linear runs
    // m=(N-1) activations against every weight row in parallel (O(out_f) rayon tasks, N-1 dots
    // each), reading each weight row ONCE and reusing it across all tokens. This fills the KV
    // cache for positions 0..N-2. The last prompt token is left for the normal decode loop so
    // that the "decode" stats (tok/s) remain meaningful and the first generated token is sampled
    // in the canonical way.
    //
    // Guard: MoE uses Op::MoeFfn (per-token expert routing, no batched variant yet); E2B/gemma4
    // requires a per-(token,layer) host-side input vector that is computed in the per-step loop.
    // Both fall through to the original token-by-token loop below unchanged.
    let decode_start = if prompt.len() > 2 && c.moe.is_none() && !e2b {
        let pf_m = prompt.len() - 1; // process all but the last prompt token
                                     // Concatenate embeddings for the pf_m tokens: [pf_m × ne] row-major.
        let mut pf_hidden: Vec<f32> = Vec::with_capacity(pf_m * ne);
        for &tok in &prompt[..pf_m] {
            let base = tok as usize * ne;
            pf_hidden.extend(token_embd[base..base + ne].iter().map(|&x| x * embed_scale));
        }
        // Absolute positions [0, 1, ..., pf_m-1].
        let pf_positions: Vec<i32> = (0..pf_m as i32).collect();
        // Allocate staging buffers sized for the prefill batch.
        let pf_hidden_buf = be
            .alloc(pf_m * ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let pf_pos_buf = be
            .alloc(pf_m * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(pf_hidden_buf.as_ref(), bytemuck::cast_slice(&pf_hidden))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(pf_pos_buf.as_ref(), bytemuck::cast_slice(&pf_positions))
            .map_err(|e| anyhow!("{e}"))?;

        let pf_t0 = std::time::Instant::now();
        let (pf_g, pf_h) = build(pf_m, 0);
        let pf_plan = be.compile(&pf_g).map_err(|e| anyhow!("{e}"))?;
        let mut pf_b = Bindings::new();
        pf_b.bind(pf_h.hidden, pf_hidden_buf.as_ref());
        pf_b.bind(pf_h.positions, pf_pos_buf.as_ref());
        // gemma4's proportional-RoPE divisors are a graph input too — bind them (the per-token decode
        // loop below does the same). Without this the batched graph has an unbound `rope_freqs` Input
        // and panics. (E2B's per-layer input is excluded by the `!e2b` guard above.)
        if let (Some(rid), Some((rb, _))) = (pf_h.rope_freqs, &rf_buf) {
            pf_b.bind(rid, rb.as_ref());
        }
        for l in 0..c.n_layer {
            pf_b.bind(pf_h.k_cache[l], kbufs[l].as_ref());
            pf_b.bind(pf_h.v_cache[l], vbufs[l].as_ref());
        }
        for (i, wid) in pf_h.weights.iter().enumerate() {
            pf_b.bind(*wid, wbufs[i].as_ref());
        }
        pf_b.bind(pf_h.logits, logits_buf.as_ref());
        be.execute(pf_plan.as_ref(), &pf_b)
            .map_err(|e| anyhow!("{e}"))?;
        prompt_t += pf_t0.elapsed();

        // KV cache is filled for positions 0..pf_m-1.
        // The last prompt token (position pf_m) is handled by the decode loop below,
        // which will write its KV, get the correct logits, and sample the first generated token.
        pf_m
    } else {
        0 // fall through to per-token loop for MoE / E2B / short prompts
    };

    for pos in decode_start..(prompt.len() + max_new) {
        if out.len() >= max_new {
            break;
        }
        let step_t0 = std::time::Instant::now();
        let tok = cur[pos] as usize;
        // embed (gemma scales by √n_embd; qwen3/llama identity)
        let emb: Vec<f32> = token_embd[tok * ne..tok * ne + ne]
            .iter()
            .map(|&x| x * embed_scale)
            .collect();
        be.upload(hidden_buf.as_ref(), bytemuck::cast_slice(&emb))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(pos_buf.as_ref(), bytemuck::cast_slice(&[pos as i32]))
            .map_err(|e| anyhow!("{e}"))?;

        // gemma4 E2B: build this token's per-layer input vector on the host (mirrors the GPU forward):
        // `ipl[l] = ((model_proj_l·emb)/√n_embd, RMSNorm'd over npl) + (per_layer_tok_embd_row × √npl)) / √2`.
        if let (Some(ple), Some(ipl_buf)) = (ple, &ipl_buf) {
            let (npl, nl, nem) = (ple.npl, ple.n_layer, ple.n_embd);
            let inv_sqrt_ne = 1.0 / (nem as f32).sqrt();
            let sqrt_npl = (npl as f32).sqrt();
            let inv_sqrt2 = 1.0 / 2f32.sqrt();
            let te_bytes = g
                .tensor_bytes("per_layer_token_embd.weight")
                .map_err(|e| anyhow!("{e}"))?;
            let r0 = tok * ple.tok_embd_row_bytes;
            let pl_tok = dequant_block(
                ple.tok_embd_dtype,
                &te_bytes[r0..r0 + ple.tok_embd_row_bytes],
            )
            .map_err(|e| anyhow!("{e}"))?;
            let mut ipl = vec![0f32; nl * npl];
            for layer in 0..nl {
                let mut proj = vec![0f32; npl];
                let mut ss = 0f32;
                for (j, pj) in proj.iter_mut().enumerate() {
                    let wrow =
                        &ple.model_proj[(layer * npl + j) * nem..(layer * npl + j) * nem + nem];
                    let acc: f32 = wrow.iter().zip(&emb).map(|(a, b)| a * b).sum();
                    let v = acc * inv_sqrt_ne;
                    *pj = v;
                    ss += v * v;
                }
                let rms = 1.0 / (ss / npl as f32 + c.rms_eps).sqrt();
                for j in 0..npl {
                    let normed = proj[j] * rms * ple.proj_norm[j];
                    let tokv = pl_tok[layer * npl + j] * sqrt_npl;
                    ipl[layer * npl + j] = (normed + tokv) * inv_sqrt2;
                }
            }
            be.upload(ipl_buf.as_ref(), bytemuck::cast_slice(&ipl))
                .map_err(|e| anyhow!("{e}"))?;
        }

        let (g, h) = build(1, pos);
        let plan = be.compile(&g).map_err(|e| anyhow!("{e}"))?;
        let mut b = Bindings::new();
        b.bind(h.hidden, hidden_buf.as_ref());
        b.bind(h.positions, pos_buf.as_ref());
        if let (Some(rid), Some((rb, _))) = (h.rope_freqs, &rf_buf) {
            b.bind(rid, rb.as_ref());
        }
        if let (Some(pid), Some(ib)) = (h.per_layer_inp, &ipl_buf) {
            b.bind(pid, ib.as_ref());
        }
        for l in 0..c.n_layer {
            b.bind(h.k_cache[l], kbufs[l].as_ref());
            b.bind(h.v_cache[l], vbufs[l].as_ref());
        }
        for (i, wid) in h.weights.iter().enumerate() {
            b.bind(*wid, wbufs[i].as_ref());
        }
        b.bind(h.logits, logits_buf.as_ref());
        be.execute(plan.as_ref(), &b).map_err(|e| anyhow!("{e}"))?;

        // Only sample once we're past the prompt (decode position = last prompt token onward).
        let is_decode = pos + 1 >= prompt.len();
        if is_decode {
            be.download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
                .map_err(|e| anyhow!("{e}"))?;
            let next = argmax(&logits) as u32;
            let is_eos = c.eos_ids.contains(&next) || next == c.eos;
            out.push(next);
            decode_t += step_t0.elapsed();
            decode_n += 1;
            if !is_eos {
                on_token(next); // stream the token (EOS is not emitted)
            }
            if is_eos || out.len() >= max_new {
                break;
            }
            if cur.len() <= pos + 1 {
                cur.push(next);
            }
        } else {
            prompt_t += step_t0.elapsed();
        }
    }
    if prof {
        let ts = |d: std::time::Duration, n: usize| {
            if d.as_secs_f64() > 0.0 {
                n as f64 / d.as_secs_f64()
            } else {
                0.0
            }
        };
        eprintln!(
            "[cpu prof] prompt {} tok in {:.2}s ({:.1} tok/s) | decode {} tok in {:.2}s ({:.2} tok/s)",
            prompt.len(),
            prompt_t.as_secs_f64(),
            ts(prompt_t, prompt.len()),
            decode_n,
            decode_t.as_secs_f64(),
            ts(decode_t, decode_n),
        );
    }
    let stats = CpuStats {
        n_prompt: prompt.len(),
        prompt_secs: prompt_t.as_secs_f64(),
        n_gen: decode_n,
        decode_secs: decode_t.as_secs_f64(),
    };
    Ok((out, stats))
}

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
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
        let wref = crate::dequant_block(DType::Q4K, &w).unwrap();
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
        let wref = crate::dequant_block(DType::Q6K, &w).unwrap();
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
        let wref = crate::dequant_block(DType::Q8_0, &w).unwrap();
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
        let wref = crate::dequant_block(DType::Q5K, &w).unwrap();
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
}
