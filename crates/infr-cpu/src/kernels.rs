//! Per-format vec-dot kernels (scalar/AVX2/AVX-512 + dispatchers): Q4_K, Q6_K, Q8_0, Q5_K
//! single-row and batch families, the Q8_0_32/Q5_0_32 native-block family, and the plain
//! f32/f16/bf16 dots + gated-FFN activation.
#[cfg(target_arch = "x86_64")]
use crate::quant::{hadd_i32_xmm, hadd_i32_ymm};
use crate::quant::{Q8x32, Q8};
use infr_core::graph::Activation;
use infr_gguf::dequant::{k4, rdf16};

/// `Σ weight·x` for one Q4_K row (144 bytes / 256 elems) against the Q8 activation. Weight value is
/// `d·sc_s·q4 − dmin·m_s` over 8 sub-blocks of 32; dispatches to the best SIMD path available at
/// runtime (avx512bw → avx2 → scalar). The `sm` term uses `q8.bsums` (precomputed in `quantize_q8`)
/// instead of re-summing q8 values per row.
pub(crate) fn vec_dot_q4k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
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
pub(crate) fn vec_dot_q6k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
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
        // INTEGER epilogue (llama.cpp's `ggml_vec_dot_q6_K_q8_K` shape): the scale-weighted
        // sub-block dots accumulate exactly in i32 (|Σ| < 2^29, no overflow possible), ONE f32
        // multiply per super-block — fewer roundings than the old per-sub f32 chain, and the
        // i32 sum is order-free, so SIMD variants no longer need an order-pinned f32 tail.
        let mut isum = 0i32;
        for sub in 0..16 {
            isum += scales[sub] as i8 as i32 * (sumi[sub] - 32 * bsum[sub]);
        }
        sumf += d * q8.d[b] * isum as f32;
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

        let mut s = 0i32; // integer epilogue — see `vec_dot_q6k_scalar`

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
                s += scales[sub_0] as i8 as i32 * (iprod_0 - 32 * bs_0);
                s += scales[sub_1] as i8 as i32 * (iprod_1 - 32 * bs_1);
            }
        }
        sumf += d * q8.d[b] * s as f32;
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

        // Integer epilogue — see `vec_dot_q6k_scalar` (i32 sum is order-free and exact).
        let mut s = 0i32;
        for sub in 0..16 {
            s += scales[sub] as i8 as i32 * (simd_sumi[sub] - 32 * simd_bsum[sub]);
        }
        sumf += d * q8.d[b] * s as f32;
    }
    sumf
}

pub(crate) fn vec_dot_q8_0(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
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

pub(crate) fn vec_dot_q5k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
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

/// AVX512-VNNI variant of [`vec_dot_q4k_batch_avx512bw`]: `_mm512_dpbusd_epi32` fuses the
/// maddubs+madd pair into ONE u8×s8→i32 dot-accumulate. Bit-identical — the i32 lanes hold the
/// same per-4-byte-group sums the madd chain produced (q4 nibbles ≤15 × |q8| ≤127 never
/// saturated maddubs' i16 either), and the hadd/scale order is unchanged.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
unsafe fn vec_dot_q4k_batch_vnni(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);

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
                let q4_zmm = _mm512_loadu_si512(flat[k * 64..].as_ptr() as *const __m512i);
                let q8_zmm = _mm512_loadu_si512(q8b[k * 64..].as_ptr() as *const __m512i);
                let sum32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_zmm, q8_zmm);
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
pub(crate) fn vec_dot_q4k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            return unsafe { vec_dot_q4k_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q4k_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q4k_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q4k_batch_scalar(row, q8s, in_f, out);
}

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

/// AVX512-VNNI variant of [`vec_dot_q4k_batch2_avx512bw`] (2-row tile, q8 loaded once per pair,
/// dpbusd replacing maddubs+madd — see [`vec_dot_q4k_batch_vnni`]'s bit-identity note).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
unsafe fn vec_dot_q4k_batch2_vnni(
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

                let q8_zmm = _mm512_loadu_si512(q8b[k * 64..].as_ptr() as *const __m512i);

                let qa_zmm = _mm512_loadu_si512(fa[k * 64..].as_ptr() as *const __m512i);
                let sum32_a = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qa_zmm, q8_zmm);
                let lo_a = _mm512_castsi512_si256(sum32_a);
                let hi_a = _mm512_extracti64x4_epi64::<1>(sum32_a);

                let qb_zmm = _mm512_loadu_si512(fb[k * 64..].as_ptr() as *const __m512i);
                let sum32_b = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qb_zmm, q8_zmm);
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

pub(crate) fn vec_dot_q4k_batch2(
    row_a: &[u8],
    row_b: &[u8],
    q8s: &[Q8],
    in_f: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            return unsafe { vec_dot_q4k_batch2_vnni(row_a, row_b, q8s, in_f, out_a, out_b) };
        }
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q4k_batch2_avx512bw(row_a, row_b, q8s, in_f, out_a, out_b) };
        }
    }
    // Scalar fallback: call single-row batch twice (still unpack once per row).
    vec_dot_q4k_batch(row_a, q8s, in_f, out_a);
    vec_dot_q4k_batch(row_b, q8s, in_f, out_b);
}

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

/// Interleaved-x8 Q4_K GEMM tile (AVX512-VNNI) — the ggml `block_q4_Kx8` idea applied per call:
/// the 8 weight rows' expanded nibbles are INTERLEAVED per (super-block, sub-block, 8-byte group)
/// so one contiguous zmm load carries the SAME 8 activation positions of ALL 8 rows (qword lane i
/// = row i), and the activation qword is broadcast once. Per sub-block that's 4 dpbusd + ONE
/// hadd + ONE mullo for all 8 rows — vs the flat batch8's 8 dpbusd + 16 mullo + per-row hadds —
/// and every weight byte the inner loop touches is sequential.
///
/// `_mm256_hadd_epi32(lo, hi)` yields per-row sums in the fixed permutation [0,1,4,5,2,3,6,7]
/// (its 128-bit-lane semantics); scales are pre-permuted to match and lanes un-permute only at
/// the per-super-block extraction. Bit-identical to the scalar single-row kernel: the integer
/// sums are exact regardless of association, and the per-(row, super-block) f32 expression and
/// its accumulation order are unchanged.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
unsafe fn vec_dot_q4k_batch8_ilv_vnni(
    rows: [&[u8]; 8],
    q8s: &[Q8],
    in_f: usize,
    outs: [&mut [f32]; 8],
) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    // hadd_epi32(lo, hi) sum-position j holds row PERM[j].
    const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];

    // ── repack: expand nibbles per row (as the flat kernels do), then interleave ──
    // ilv[b*2048 + s*256 + g*64 + i*8 .. +8] = row i, sub-block s, byte group g (8 B).
    let mut d_arr: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0f32; nb]);
    let mut dmin_arr: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0f32; nb]);
    let mut m_arr: [Vec<[u32; 8]>; 8] = std::array::from_fn(|_| vec![[0u32; 8]; nb]);
    // per-(b, s) PAIR-DUPLICATED scale vector in dpbusd lane order (lanes 2i, 2i+1 = row i) —
    // consumed by the vertical 512-bit mullo; see q4k_gemm_group's sc16.
    let mut sc16_vec = vec![[_mm512_setzero_si512(); 8]; nb];
    // PERM-ordered per-(b,s) m-scales and per-b f32 d/dmin vectors: lets the sm accumulation and
    // the final `q8.d*(d*sd - dmin*sm)` run 8-rows-wide in lane space (un-permuted only at the
    // very end). No FMA anywhere — each mul/sub/add rounds separately, matching the scalar order.
    let mut m_vec = vec![[_mm256_setzero_si256(); 8]; nb];
    let mut d_vec = vec![_mm256_setzero_ps(); nb];
    let mut dmin_vec = vec![_mm256_setzero_ps(); nb];
    let mut ilv = vec![0u8; nb * 2048];
    {
        let mut sc_rows = [[0u32; 8]; 8]; // [row][s]
        let mut tmp = [[0u8; 256]; 8]; // expanded nibbles, one superblock, all 8 rows
        for b in 0..nb {
            for i in 0..8 {
                let blk = &rows[i][b * 144..b * 144 + 144];
                d_arr[i][b] = rdf16(&blk[0..2]);
                dmin_arr[i][b] = rdf16(&blk[2..4]);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                for s in 0..8usize {
                    let (sc, mv) = k4(s, scales);
                    sc_rows[i][s] = sc;
                    m_arr[i][b][s] = mv;
                }
                for k in 0..4usize {
                    let nibs = _mm256_loadu_si256(qs[k * 32..].as_ptr() as *const __m256i);
                    let lo = _mm256_and_si256(nibs, mask_lo);
                    let hi = _mm256_and_si256(_mm256_srli_epi16(nibs, 4), mask_lo);
                    _mm256_storeu_si256(tmp[i][k * 64..].as_mut_ptr() as *mut __m256i, lo);
                    _mm256_storeu_si256(tmp[i][k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi);
                }
            }
            d_vec[b] = _mm256_setr_ps(
                d_arr[PERM[0]][b],
                d_arr[PERM[1]][b],
                d_arr[PERM[2]][b],
                d_arr[PERM[3]][b],
                d_arr[PERM[4]][b],
                d_arr[PERM[5]][b],
                d_arr[PERM[6]][b],
                d_arr[PERM[7]][b],
            );
            dmin_vec[b] = _mm256_setr_ps(
                dmin_arr[PERM[0]][b],
                dmin_arr[PERM[1]][b],
                dmin_arr[PERM[2]][b],
                dmin_arr[PERM[3]][b],
                dmin_arr[PERM[4]][b],
                dmin_arr[PERM[5]][b],
                dmin_arr[PERM[6]][b],
                dmin_arr[PERM[7]][b],
            );
            for s in 0..8usize {
                sc16_vec[b][s] = _mm512_setr_epi32(
                    sc_rows[0][s] as i32,
                    sc_rows[0][s] as i32,
                    sc_rows[1][s] as i32,
                    sc_rows[1][s] as i32,
                    sc_rows[2][s] as i32,
                    sc_rows[2][s] as i32,
                    sc_rows[3][s] as i32,
                    sc_rows[3][s] as i32,
                    sc_rows[4][s] as i32,
                    sc_rows[4][s] as i32,
                    sc_rows[5][s] as i32,
                    sc_rows[5][s] as i32,
                    sc_rows[6][s] as i32,
                    sc_rows[6][s] as i32,
                    sc_rows[7][s] as i32,
                    sc_rows[7][s] as i32,
                );
                m_vec[b][s] = _mm256_setr_epi32(
                    m_arr[PERM[0]][b][s] as i32,
                    m_arr[PERM[1]][b][s] as i32,
                    m_arr[PERM[2]][b][s] as i32,
                    m_arr[PERM[3]][b][s] as i32,
                    m_arr[PERM[4]][b][s] as i32,
                    m_arr[PERM[5]][b][s] as i32,
                    m_arr[PERM[6]][b][s] as i32,
                    m_arr[PERM[7]][b][s] as i32,
                );
                for g in 0..4usize {
                    let dst =
                        &mut ilv[b * 2048 + s * 256 + g * 64..b * 2048 + s * 256 + g * 64 + 64];
                    for i in 0..8 {
                        dst[i * 8..i * 8 + 8]
                            .copy_from_slice(&tmp[i][s * 32 + g * 8..s * 32 + g * 8 + 8]);
                    }
                }
            }
        }
    }

    let [o0, o1, o2, o3, o4, o5, o6, o7] = outs;
    for r in 0..m {
        let q8 = &q8s[r];
        // Whole-row f32 accumulator, 8 rows wide in PERM lane order — un-permuted only at the
        // final store. Every f32 op below is a separate mul/sub/add (NO fma): identical rounding
        // sequence to the scalar per-row expression.
        let mut sumf_v = _mm256_setzero_ps();
        for b in 0..nb {
            // sd/sm per row: sd vertical in 512-bit pair-lane space, sm in PERM lane order.
            let mut sd_zmm = _mm512_setzero_si512();
            let mut sm_ymm = _mm256_setzero_si256();
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for s in 0..8usize {
                let mut acc = _mm512_setzero_si512();
                for g in 0..4usize {
                    let w = _mm512_loadu_si512(
                        ilv[b * 2048 + s * 256 + g * 64..].as_ptr() as *const __m512i
                    );
                    // q8b is &[i8]; read the 8 activation bytes as one little-endian qword.
                    let a = _mm512_set1_epi64(
                        (q8b[s * 32 + g * 8..].as_ptr() as *const i64).read_unaligned(),
                    );
                    acc = _mm512_dpbusd_epi32(acc, w, a);
                }
                // Vertical: scale pair lanes in 512-bit space; pair-merge deferred to once per
                // super-block (integer-exact — see q4k_gemm_group's note).
                sd_zmm = _mm512_add_epi32(sd_zmm, _mm512_mullo_epi32(acc, sc16_vec[b][s]));
                let isum = _mm256_set1_epi32(q8.bsums[b * 8 + s]);
                sm_ymm = _mm256_add_epi32(sm_ymm, _mm256_mullo_epi32(m_vec[b][s], isum));
            }
            let sd_lo = _mm512_castsi512_si256(sd_zmm);
            let sd_hi = _mm512_extracti64x4_epi64::<1>(sd_zmm);
            let sd_ymm = _mm256_hadd_epi32(sd_lo, sd_hi); // pair-merge -> PERM order
                                                          // q8.d[b] * (d*sd - dmin*sm), 8 rows at once (cvtepi32→f32 is exact for these
                                                          // magnitudes; mul/sub/mul/add sequence matches the scalar expression's rounding).
            let sd_f = _mm256_cvtepi32_ps(sd_ymm);
            let sm_f = _mm256_cvtepi32_ps(sm_ymm);
            let t = _mm256_sub_ps(
                _mm256_mul_ps(d_vec[b], sd_f),
                _mm256_mul_ps(dmin_vec[b], sm_f),
            );
            sumf_v = _mm256_add_ps(sumf_v, _mm256_mul_ps(_mm256_set1_ps(q8.d[b]), t));
        }
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sumf_v);
        o0[r] = lanes[0];
        o1[r] = lanes[1];
        o4[r] = lanes[2];
        o5[r] = lanes[3];
        o2[r] = lanes[4];
        o3[r] = lanes[5];
        o6[r] = lanes[6];
        o7[r] = lanes[7];
    }
}

/// Batch Q4_K 8-row tile: `outs[i][r] = vec_dot_q4k(rows[i], &q8s[r], in_f)` for all i,r.
/// Bit-identical to the single-token kernel. On AVX-512BW machines the Q8 activation is
/// loaded once per (block, nibble-pair) and dotted against all 8 weight rows — 8× activation
/// reuse over single-row, 4× over 2-row. Falls back to 8× `vec_dot_q4k_batch` on older CPUs.
pub(crate) fn vec_dot_q4k_batch8(rows: [&[u8]; 8], q8s: &[Q8], in_f: usize, outs: [&mut [f32]; 8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            return unsafe { vec_dot_q4k_batch8_ilv_vnni(rows, q8s, in_f, outs) };
        }
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q4k_batch8_avx512bw(rows, q8s, in_f, outs) };
        }
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
            // Integer epilogue — see `vec_dot_q6k_scalar`.
            let mut s = 0i32;
            for sub in 0..16 {
                s += scales_arr[b * 16 + sub] as i32 * (sumi[sub] - 32 * bsum[sub]);
            }
            sumf += d_arr[b] * q8.d[b] * s as f32;
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

            let mut s = 0i32; // integer epilogue — see `vec_dot_q6k_scalar`
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
                    s += scales_arr[b * 16 + sub_0] as i32 * (iprod_0 - 32 * bs_0);
                    s += scales_arr[b * 16 + sub_1] as i32 * (iprod_1 - 32 * bs_1);
                }
            }
            sumf += d * q8.d[b] * s as f32;
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
unsafe fn vec_dot_q6k_batch_avx512bw(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;

    let mask_0f_z = _mm512_set1_epi8(0x0F_u8 as i8);
    let mask_30_z = _mm512_set1_epi8(0x30_u8 as i8);
    let mask_03_z = _mm512_set1_epi8(0x03_u8 as i8);

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

    // Per-token: `flat` is plain linear element order (see the store above: sub-block s = elems
    // 16s..16s+16), so each super-block is 4 straight 64-byte chunks. Each chunk's madd yields
    // 16 i32 lanes = 4 sub-blocks × 4 lanes; collapse in-register (2 permute+add steps per chunk,
    // 3 cross-chunk permutes) into ONE zmm holding the 16 per-sub-block dots in order — replacing
    // the 32-extract + 32-hadd storm the old loop paid per (row, super-block). The `-32` offset
    // correction reads `q8.bsums16` (computed once at quantize time) instead of re-deriving the
    // activation sums per weight row — the same values, so bit-identity holds, minus 262k
    // redundant recomputes per lm_head GEMM. The f32 epilogue is UNCHANGED (ascending sub-block
    // chain, then d·d8·s). Rows go in PAIRS sharing the weight loads so the two sequential f32
    // chains overlap in the OoO window instead of exposing their latency back to back.
    let idx_pair = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    let idx_half = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    let idx_q14 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 28, 24, 20, 16, 12, 8, 4, 0);
    let idx_cat = _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0);

    // One row's integer dots for super-block `b` against the four preloaded weight chunks:
    // sumi16[s] = Σ_{i∈sub-block s} q6[i]·q8[i] (q6 still biased +32 — corrected in the epilogue).
    macro_rules! q6k_sumi16 {
        ($q8:expr, $b:expr, $w0:expr, $w1:expr, $w2:expr, $w3:expr) => {{
            let qs = &$q8.qs[$b * 256..$b * 256 + 256];
            let a0 = _mm512_loadu_si512(qs.as_ptr() as *const __m512i);
            let a1 = _mm512_loadu_si512(qs[64..].as_ptr() as *const __m512i);
            let a2 = _mm512_loadu_si512(qs[128..].as_ptr() as *const __m512i);
            let a3 = _mm512_loadu_si512(qs[192..].as_ptr() as *const __m512i);
            // dpbusd: one op per 64-byte chunk instead of the maddubs+madd pair —
            // integer-exact both ways (u8≤63 × i8 groups can't saturate the i16 pairs).
            let z = _mm512_setzero_si512();
            let s0 = _mm512_dpbusd_epi32(z, $w0, a0);
            let s1 = _mm512_dpbusd_epi32(z, $w1, a1);
            let s2 = _mm512_dpbusd_epi32(z, $w2, a2);
            let s3 = _mm512_dpbusd_epi32(z, $w3, a3);
            let c0 = _mm512_add_epi32(s0, _mm512_permutexvar_epi32(idx_pair, s0));
            let c0 = _mm512_add_epi32(c0, _mm512_permutexvar_epi32(idx_half, c0));
            let c1 = _mm512_add_epi32(s1, _mm512_permutexvar_epi32(idx_pair, s1));
            let c1 = _mm512_add_epi32(c1, _mm512_permutexvar_epi32(idx_half, c1));
            let c2 = _mm512_add_epi32(s2, _mm512_permutexvar_epi32(idx_pair, s2));
            let c2 = _mm512_add_epi32(c2, _mm512_permutexvar_epi32(idx_half, c2));
            let c3 = _mm512_add_epi32(s3, _mm512_permutexvar_epi32(idx_pair, s3));
            let c3 = _mm512_add_epi32(c3, _mm512_permutexvar_epi32(idx_half, c3));
            let lo = _mm512_permutex2var_epi32(c0, idx_q14, c1); // subs 0..7 in lanes 0..7
            let hi = _mm512_permutex2var_epi32(c2, idx_q14, c3); // subs 8..15 in lanes 0..7
            _mm512_permutex2var_epi32(lo, idx_cat, hi)
        }};
    }
    // INTEGER epilogue in SIMD (see `vec_dot_q6k_scalar`): `Σ_s sc_s·(dp_s − 32·bsum_s)` fully
    // in 16 i32 lanes (mullo + one reduce), replacing the order-pinned 16-step scalar f32 chain
    // that used to dominate this kernel's critical path.
    macro_rules! q6k_epilogue {
        ($q8:expr, $b:expr, $sumi_z:expr, $scales_z:expr, $sumf:expr) => {{
            let bs = _mm512_loadu_si512($q8.bsums16[$b * 16..].as_ptr() as *const __m512i);
            let corr = _mm512_sub_epi32($sumi_z, _mm512_slli_epi32::<5>(bs));
            let isum = _mm512_reduce_add_epi32(_mm512_mullo_epi32(corr, $scales_z));
            $sumf += d_arr[$b] * $q8.d[$b] * isum as f32;
        }};
    }

    for rp in 0..m / 2 {
        let (ra, rb) = (2 * rp, 2 * rp + 1);
        let (q8a, q8b) = (&q8s[ra], &q8s[rb]);
        let (mut sumf_a, mut sumf_b) = (0f32, 0f32);
        for b in 0..nb {
            let flat = &q6_flat[b * 256..b * 256 + 256];
            let w0 = _mm512_loadu_si512(flat.as_ptr() as *const __m512i);
            let w1 = _mm512_loadu_si512(flat[64..].as_ptr() as *const __m512i);
            let w2 = _mm512_loadu_si512(flat[128..].as_ptr() as *const __m512i);
            let w3 = _mm512_loadu_si512(flat[192..].as_ptr() as *const __m512i);
            let sc_z = _mm512_cvtepi8_epi32(_mm_loadu_si128(
                scales_arr[b * 16..].as_ptr() as *const __m128i
            ));
            let sumi_a = q6k_sumi16!(q8a, b, w0, w1, w2, w3);
            let sumi_b = q6k_sumi16!(q8b, b, w0, w1, w2, w3);
            q6k_epilogue!(q8a, b, sumi_a, sc_z, sumf_a);
            q6k_epilogue!(q8b, b, sumi_b, sc_z, sumf_b);
        }
        out[ra] = sumf_a;
        out[rb] = sumf_b;
    }
    if m % 2 == 1 {
        let r = m - 1;
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let flat = &q6_flat[b * 256..b * 256 + 256];
            let w0 = _mm512_loadu_si512(flat.as_ptr() as *const __m512i);
            let w1 = _mm512_loadu_si512(flat[64..].as_ptr() as *const __m512i);
            let w2 = _mm512_loadu_si512(flat[128..].as_ptr() as *const __m512i);
            let w3 = _mm512_loadu_si512(flat[192..].as_ptr() as *const __m512i);
            let sc_z = _mm512_cvtepi8_epi32(_mm_loadu_si128(
                scales_arr[b * 16..].as_ptr() as *const __m128i
            ));
            let sumi = q6k_sumi16!(q8, b, w0, w1, w2, w3);
            q6k_epilogue!(q8, b, sumi, sc_z, sumf);
        }
        out[r] = sumf;
    }
}

pub(crate) fn vec_dot_q6k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            return unsafe { vec_dot_q6k_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q6k_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q6k_batch_scalar(row, q8s, in_f, out);
}

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

pub(crate) fn vec_dot_q8_0_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
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

pub(crate) fn vec_dot_q5k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
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

/// Batched Q8_0 dot at native 32-block granularity: `y = d_w·qw` exactly (no min term — the i8 sign
/// already encodes it, see `dequant_block`'s Q8_0 case), so `Σy·x = d_w·Σ(qw·x) ≈ d_w·d8·Σ(qw·q8)`.
/// The weight row's per-block `d_w` is read once, then dotted against every one of the `count` token
/// activations — same amortization as the other `_batch` kernels, just without an f32 intermediate.
/// Dispatches to the best SIMD path available at runtime (avx512bw → avx2 → scalar) — this is
/// DiffusionGemma's `down` projection kernel (`n_ff_exp=704` isn't a multiple of 256, so it can't use
/// the K-quant/Q8_0-256 batch path above), previously the single largest scalar hot loop in the MoE
/// arm (measured via `INFR_PROF_OPS=1`'s per-stage MoeFfn breakdown).
/// AVX512-VNNI variant of [`vec_dot_q8_0_32_batch_avx512bw`]: dpbusd fuses the maddubs+madd pair
/// (sign trick unchanged: `dpbusd(|qw|, sign(q8,qw))`; |qw| ≤128 × |q8| ≤127 never saturated the
/// i16 chain either, so this is bit-identical — same per-4-byte i32 group sums, same hadd and the
/// same two SEPARATE f32 adds per pair as the scalar oracle).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni,avx512vl")]
unsafe fn vec_dot_q8_0_32_batch_vnni(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 34usize;
    let pairs = nb / 2;
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for k in 0..pairs {
            let (b0, b1) = (2 * k, 2 * k + 1);
            let qw0 = _mm256_loadu_si256(row[b0 * bpr + 2..].as_ptr() as *const __m256i);
            let qw1 = _mm256_loadu_si256(row[b1 * bpr + 2..].as_ptr() as *const __m256i);
            let qx_z = _mm512_loadu_si512(q8.qs[b0 * 32..].as_ptr() as *const __m512i);
            let qx0 = _mm512_castsi512_si256(qx_z);
            let qx1 = _mm512_extracti64x4_epi64::<1>(qx_z);
            let qw_abs0 = _mm256_abs_epi8(qw0);
            let qw_abs1 = _mm256_abs_epi8(qw1);
            let qx_s0 = _mm256_sign_epi8(qx0, qw0);
            let qx_s1 = _mm256_sign_epi8(qx1, qw1);
            let qw_a_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qw_abs0), qw_abs1);
            let qx_s_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qx_s0), qx_s1);
            let sum32_z = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qw_a_z, qx_s_z);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            sumf += d_arr[b0] * q8.d[b0] * iprod0 as f32;
            sumf += d_arr[b1] * q8.d[b1] * iprod1 as f32;
        }
        if nb % 2 == 1 {
            let b = nb - 1;
            let qw = _mm256_loadu_si256(row[b * bpr + 2..].as_ptr() as *const __m256i);
            let q8v = _mm256_loadu_si256(q8.qs[b * 32..].as_ptr() as *const __m256i);
            let qw_abs = _mm256_abs_epi8(qw);
            let qx_signed = _mm256_sign_epi8(q8v, qw);
            let sum32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qw_abs, qx_signed);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

pub(crate) fn vec_dot_q8_0_32_batch(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: features detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q8_0_32_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512bw detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q8_0_32_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q8_0_32_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q8_0_32_batch_scalar(row, q8s, in_f, out);
}

/// Scalar fallback for `vec_dot_q8_0_32_batch`; also used on non-x86 targets, and the exactness
/// oracle the SIMD kernels below are tested against.
fn vec_dot_q8_0_32_batch_scalar(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 32;
    let bpr = 34usize; // f16 d (2B) + 32 × i8 qs
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let qw = &row[b * bpr + 2..b * bpr + bpr];
            let q8b = &q8.qs[b * 32..b * 32 + 32];
            let mut iprod = 0i32;
            for i in 0..32 {
                iprod += qw[i] as i8 as i32 * q8b[i] as i32;
            }
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// AVX2 kernel for `vec_dot_q8_0_32_batch`: one native 32-element Q8_0 block per iteration. Same
/// sign trick as `vec_dot_q8_0_avx2` (`maddubs(abs(qw), sign(qw)·qx)`), just scaled per-block instead
/// of per-256-superblock (each native block carries its own `d_w`, and the activation's own `d[b]`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q8_0_32_batch_avx2(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 34usize;
    let ones = _mm256_set1_epi16(1i16);
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let qw = _mm256_loadu_si256(row[b * bpr + 2..].as_ptr() as *const __m256i);
            let q8v = _mm256_loadu_si256(q8.qs[b * 32..].as_ptr() as *const __m256i);
            let qw_abs = _mm256_abs_epi8(qw);
            let qx_signed = _mm256_sign_epi8(q8v, qw);
            let prod = _mm256_maddubs_epi16(qw_abs, qx_signed);
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// AVX-512BW kernel for `vec_dot_q8_0_32_batch`: TWO native 32-element blocks per iteration (64
/// elems / zmm), mirroring `vec_dot_q8_0_avx512bw`'s pairing trick — sign trick applied at ymm level,
/// then packed into a zmm for one `maddubs512 → madd512` pass. `nb` is even for every shape this
/// crate has seen (e.g. DiffusionGemma's 22), but an odd tail block is handled scalarly for safety.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn vec_dot_q8_0_32_batch_avx512bw(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 34usize;
    let ones512 = _mm512_set1_epi16(1i16);
    let ones256 = _mm256_set1_epi16(1i16);
    let pairs = nb / 2;
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for k in 0..pairs {
            let (b0, b1) = (2 * k, 2 * k + 1);
            let qw0 = _mm256_loadu_si256(row[b0 * bpr + 2..].as_ptr() as *const __m256i);
            let qw1 = _mm256_loadu_si256(row[b1 * bpr + 2..].as_ptr() as *const __m256i);
            // b0/b1 activation bytes are contiguous (`Q8x32::qs` is laid out `[b*32..b*32+32]`).
            let qx_z = _mm512_loadu_si512(q8.qs[b0 * 32..].as_ptr() as *const __m512i);
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
            // Two SEPARATE f32 adds (not `sumf += a + b`) — matches the scalar oracle's per-block
            // sequential accumulation order exactly (f32 addition isn't associative).
            sumf += d_arr[b0] * q8.d[b0] * iprod0 as f32;
            sumf += d_arr[b1] * q8.d[b1] * iprod1 as f32;
        }
        if nb % 2 == 1 {
            let b = nb - 1;
            let qw = _mm256_loadu_si256(row[b * bpr + 2..].as_ptr() as *const __m256i);
            let q8v = _mm256_loadu_si256(q8.qs[b * 32..].as_ptr() as *const __m256i);
            let qw_abs = _mm256_abs_epi8(qw);
            let qx_signed = _mm256_sign_epi8(q8v, qw);
            let prod = _mm256_maddubs_epi16(qw_abs, qx_signed);
            let sum32 = _mm256_madd_epi16(prod, ones256);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// Batched Q5_0 dot at native 32-block granularity: `y = d_w·(code−16)`, `code ∈ 0..31` (4 nibble
/// bits from `qs` + 1 high bit from `qh`, per `dequant_block`'s Q5_0 case) — so
/// `Σy·x = d_w·(Σcode·x − 16·Σx) ≈ d_w·d8·(Σcode·q8 − 16·bsum)`.
pub(crate) fn vec_dot_q5_0_32_batch(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: features detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q5_0_32_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q5_0_32_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q5_0_32_batch_scalar(row, q8s, in_f, out);
}

/// Interleaved-x8 tile over native 32-element blocks (AVX512-VNNI) for the MoE `down`
/// projections — the [`vec_dot_q4k_batch8_ilv_vnni`] structure at Q8_0/Q5_0's own block size.
/// Weights become UNSIGNED for dpbusd: Q5_0's 5-bit codes already are (0..31; the −16 rides the
/// `−16·bsum` term); Q8_0's signed bytes are biased `+128` at repack and the exact integer
/// correction `−128·bsum[b]` is applied per block — both integer-exact, so results stay
/// bit-identical to the scalar oracles (same per-block f32 expression and order).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
unsafe fn vec_dot_q32_batch8_ilv_vnni(
    rows: [&[u8]; 8],
    q8s: &[Q8x32],
    in_f: usize,
    outs: [&mut [f32]; 8],
    q5: bool, // false = Q8_0 (bias +128), true = Q5_0 (codes, −16·bsum)
) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bias = _mm_set1_epi8(-128i8); // +128 == xor 0x80 on u8 lanes

    // ── repack: per row, unsigned block bytes; then interleave 8-byte groups across rows ──
    // ilv[b*256 + g*64 + i*8 .. +8] = row i, block b, byte group g.
    let mut d_w: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0f32; nb]);
    // PERM-ordered per-block f32 weight-scale vector (see the q4k ilv kernel's d_vec).
    let mut d_vec = vec![_mm256_setzero_ps(); nb];
    let mut ilv = vec![0u8; nb * 256];
    {
        let bpr = if q5 { 22usize } else { 34usize };
        let mut tmp = [[0u8; 32]; 8];
        for b in 0..nb {
            for i in 0..8 {
                let blk = &rows[i][b * bpr..b * bpr + bpr];
                d_w[i][b] = rdf16(&blk[0..2]);
                if q5 {
                    let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
                    let qs = &blk[6..22];
                    for j in 0..16 {
                        let xh0 = ((qh >> j) << 4) & 0x10;
                        let xh1 = (qh >> (j + 12)) & 0x10;
                        tmp[i][j] = (qs[j] as u32 & 0x0F | xh0) as u8;
                        tmp[i][j + 16] = (qs[j] as u32 >> 4 | xh1) as u8;
                    }
                } else {
                    // Q8_0: bias the signed bytes to unsigned (two 16-byte xors).
                    let q = &blk[2..34];
                    let v0 = _mm_loadu_si128(q.as_ptr() as *const __m128i);
                    let v1 = _mm_loadu_si128(q[16..].as_ptr() as *const __m128i);
                    _mm_storeu_si128(tmp[i].as_mut_ptr() as *mut __m128i, _mm_xor_si128(v0, bias));
                    _mm_storeu_si128(
                        tmp[i][16..].as_mut_ptr() as *mut __m128i,
                        _mm_xor_si128(v1, bias),
                    );
                }
            }
            for g in 0..4usize {
                let dst = &mut ilv[b * 256 + g * 64..b * 256 + g * 64 + 64];
                for i in 0..8 {
                    dst[i * 8..i * 8 + 8].copy_from_slice(&tmp[i][g * 8..g * 8 + 8]);
                }
            }
            const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
            d_vec[b] = _mm256_setr_ps(
                d_w[PERM[0]][b],
                d_w[PERM[1]][b],
                d_w[PERM[2]][b],
                d_w[PERM[3]][b],
                d_w[PERM[4]][b],
                d_w[PERM[5]][b],
                d_w[PERM[6]][b],
                d_w[PERM[7]][b],
            );
        }
    }

    let sub = if q5 { 16i32 } else { 128i32 }; // per-block bsum multiplier to subtract
    let [o0, o1, o2, o3, o4, o5, o6, o7] = outs;
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf_v = _mm256_setzero_ps();
        for b in 0..nb {
            let mut acc = _mm512_setzero_si512();
            let q8b = &q8.qs[b * 32..b * 32 + 32];
            for g in 0..4usize {
                let w = _mm512_loadu_si512(ilv[b * 256 + g * 64..].as_ptr() as *const __m512i);
                let a = _mm512_set1_epi64((q8b[g * 8..].as_ptr() as *const i64).read_unaligned());
                acc = _mm512_dpbusd_epi32(acc, w, a);
            }
            let lo = _mm512_castsi512_si256(acc);
            let hi = _mm512_extracti64x4_epi64::<1>(acc);
            let sums_perm = _mm256_hadd_epi32(lo, hi);
            // Same expression as the scalar oracles, 8 rows wide (no FMA — each mul/add rounds
            // separately, matching the scalar `d_w * q8.d * iprod` left-assoc sequence): Q8_0 has
            // no bsum term of its own (the −128·bsum was the bias correction, folded into
            // `corr`); Q5_0's −16·bsum likewise.
            let corr = _mm256_set1_epi32(sub * q8.bsum[b]);
            let iprod_f = _mm256_cvtepi32_ps(_mm256_sub_epi32(sums_perm, corr));
            let scale = _mm256_mul_ps(d_vec[b], _mm256_set1_ps(q8.d[b]));
            sumf_v = _mm256_add_ps(sumf_v, _mm256_mul_ps(scale, iprod_f));
        }
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sumf_v);
        o0[r] = lanes[0];
        o1[r] = lanes[1];
        o4[r] = lanes[2];
        o5[r] = lanes[3];
        o2[r] = lanes[4];
        o3[r] = lanes[5];
        o6[r] = lanes[6];
        o7[r] = lanes[7];
    }
}

/// 8-row tile dispatcher for the native-32-block dots (Q8_0/Q5_0): the interleaved VNNI kernel
/// on capable hardware, else 8x the per-row batch kernel.
pub(crate) fn vec_dot_q32_batch8(
    rows: [&[u8]; 8],
    q8s: &[Q8x32],
    in_f: usize,
    outs: [&mut [f32]; 8],
    q5: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            return unsafe { vec_dot_q32_batch8_ilv_vnni(rows, q8s, in_f, outs, q5) };
        }
    }
    let [r0, r1, r2, r3, r4, r5, r6, r7] = rows;
    let [mut_o0, mut_o1, mut_o2, mut_o3, mut_o4, mut_o5, mut_o6, mut_o7] = outs;
    let per = |row: &[u8], out: &mut [f32]| {
        if q5 {
            vec_dot_q5_0_32_batch(row, q8s, in_f, out);
        } else {
            vec_dot_q8_0_32_batch(row, q8s, in_f, out);
        }
    };
    per(r0, mut_o0);
    per(r1, mut_o1);
    per(r2, mut_o2);
    per(r3, mut_o3);
    per(r4, mut_o4);
    per(r5, mut_o5);
    per(r6, mut_o6);
    per(r7, mut_o7);
}

/// Expand one weight row's Q5_0 codes (5-bit, 0..31, the UNSIGNED pre-`−16` values) into a flat
/// `[nb*32]` u8 buffer ONCE per row — the scalar kernel re-decoded nibble+high-bit per
/// (activation-row, block), which multiplied the decode cost by the batch size. Shared by the
/// SIMD kernels; layout `flat[b*32 + j]` = code j of block b (j 0..15 = lo nibbles, 16..31 = hi).
#[cfg(target_arch = "x86_64")] // only the x86 SIMD kernels call this — dead code on aarch64
#[inline]
fn q5_0_expand_codes(row: &[u8], nb: usize, bpr: usize) -> (Vec<u8>, Vec<f32>) {
    let mut flat = vec![0u8; nb * 32];
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        let blk = &row[b * bpr..b * bpr + bpr];
        d_arr[b] = rdf16(&blk[0..2]);
        let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
        let qs = &blk[6..22];
        let f = &mut flat[b * 32..b * 32 + 32];
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            f[j] = (qs[j] as u32 & 0x0F | xh0) as u8;
            f[j + 16] = (qs[j] as u32 >> 4 | xh1) as u8;
        }
    }
    (flat, d_arr)
}

/// AVX2 kernel for `vec_dot_q5_0_32_batch`: codes pre-expanded once (see [`q5_0_expand_codes`]),
/// then one `maddubs(code_u8, q8_s8)` block dot per (row, block) — codes ≤31 × |q8| ≤127 can't
/// saturate the i16 pair sums. Bit-identical to the scalar oracle (integer dot exact; the
/// per-block f32 accumulation expression and order are unchanged).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q5_0_32_batch_avx2(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 22usize;
    let ones = _mm256_set1_epi16(1i16);
    let (flat, d_arr) = q5_0_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let code = _mm256_loadu_si256(flat[b * 32..].as_ptr() as *const __m256i);
            let q8v = _mm256_loadu_si256(q8.qs[b * 32..].as_ptr() as *const __m256i);
            let prod = _mm256_maddubs_epi16(code, q8v);
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * (iprod as f32 - 16.0 * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// AVX512-VNNI kernel for `vec_dot_q5_0_32_batch`: two blocks per zmm, `dpbusd` in place of the
/// maddubs+madd pair (see the AVX2 variant's bit-identity note; the two per-block f32 adds stay
/// SEPARATE and in scalar order).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni,avx512vl")]
unsafe fn vec_dot_q5_0_32_batch_vnni(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 22usize;
    let pairs = nb / 2;
    let (flat, d_arr) = q5_0_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for k in 0..pairs {
            let (b0, b1) = (2 * k, 2 * k + 1);
            let code_z = _mm512_loadu_si512(flat[b0 * 32..].as_ptr() as *const __m512i);
            let qx_z = _mm512_loadu_si512(q8.qs[b0 * 32..].as_ptr() as *const __m512i);
            let sum32_z = _mm512_dpbusd_epi32(_mm512_setzero_si512(), code_z, qx_z);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            sumf += d_arr[b0] * q8.d[b0] * (iprod0 as f32 - 16.0 * q8.bsum[b0] as f32);
            sumf += d_arr[b1] * q8.d[b1] * (iprod1 as f32 - 16.0 * q8.bsum[b1] as f32);
        }
        if nb % 2 == 1 {
            let b = nb - 1;
            let code = _mm256_loadu_si256(flat[b * 32..].as_ptr() as *const __m256i);
            let q8v = _mm256_loadu_si256(q8.qs[b * 32..].as_ptr() as *const __m256i);
            let sum32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), code, q8v);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * (iprod as f32 - 16.0 * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// Scalar oracle for `vec_dot_q5_0_32_batch` (also the non-x86 path).
fn vec_dot_q5_0_32_batch_scalar(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 32;
    let bpr = 22usize; // f16 d (2B) + u32 qh (4B) + 16 × packed-nibble qs
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let blk = &row[b * bpr..b * bpr + bpr];
            let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
            let qs = &blk[6..22];
            let q8b = &q8.qs[b * 32..b * 32 + 32];
            let mut iprod = 0i32;
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                let code0 = ((qs[j] as u32 & 0x0F) | xh0) as i32;
                let code1 = ((qs[j] as u32 >> 4) | xh1) as i32;
                iprod += code0 * q8b[j] as i32 + code1 * q8b[j + 16] as i32;
            }
            sumf += d_arr[b] * q8.d[b] * (iprod as f32 - 16.0 * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// `Σ f16_weight·x` (weight is 2 bytes/elem). `target-cpu=native` lowers the f16→f32 to F16C.
pub(crate) fn dot_f16(w: &[u8], x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n = x.len();
    let chunks = n / 8;
    for c in 0..chunks {
        for (j, ac) in acc.iter_mut().enumerate() {
            let i = c * 8 + j;
            let wv = half::f16::from_le_bytes([w[i * 2], w[i * 2 + 1]]).to_f32();
            *ac = wv.mul_add(x[i], *ac);
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..n {
        s = half::f16::from_le_bytes([w[i * 2], w[i * 2 + 1]])
            .to_f32()
            .mul_add(x[i], s);
    }
    s
}

/// `Σ bf16_weight·x` (bf16 = top 16 bits of f32).
pub(crate) fn dot_bf16(w: &[u8], x: &[f32]) -> f32 {
    let mut s = 0f32;
    for (i, &xi) in x.iter().enumerate() {
        let wv = f32::from_bits((u16::from_le_bytes([w[i * 2], w[i * 2 + 1]]) as u32) << 16);
        s = wv.mul_add(xi, s);
    }
    s
}

/// Dot product with 8 independent accumulators so the reduction isn't latency-bound — lets the
/// autovectorizer (with `target-cpu=native`) keep several AVX FMA lanes in flight. `mul_add`
/// fuses each lane's multiply+add into one FMA (numerics policy: llama.cpp's f32 dots are FMA
/// too) — this fn is attention's QK score and the F32-weight GEMM fallbacks (e.g. DG's router).
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc = [0f32; 8];
    for c in 0..chunks {
        let base = c * 8;
        for (j, ac) in acc.iter_mut().enumerate() {
            *ac = a[base + j].mul_add(b[base + j], *ac);
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..n {
        s = a[i].mul_add(b[i], s);
    }
    s
}

/// Gated-FFN activation applied to the gate value.
pub(crate) fn act_fn(act: Activation, g: f32) -> f32 {
    match act {
        Activation::Silu => g / (1.0 + (-g).exp()),
        // gelu_pytorch_tanh: 0.5 g (1 + tanh(√(2/π)·(g + 0.044715 g³)))
        Activation::Gelu => 0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()),
        Activation::Sigmoid => 1.0 / (1.0 + (-g).exp()),
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::*;
    use crate::quant::{quantize_q8, quantize_q8_32};
    // These pack/gemm helpers exist only on x86_64 (AVX2/AVX512-VNNI); the tests that use them
    // gate their bodies on the same arch, so the import must be gated too or it fails to resolve
    // on aarch64.
    #[cfg(target_arch = "x86_64")]
    use crate::repack::{q4k_gemm_group, q4k_pack, q6k_gemm_group, q6k_pack};
    use infr_core::tensor::DType;
    use infr_gguf::dequant::dequant_block;

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
    /// The reference activation the `_32` native-block int8 kernels actually see: `d*q8` per 32-block.
    fn dequant_q8_32(q8: &Q8x32) -> Vec<f32> {
        let mut x = vec![0f32; q8.qs.len()];
        for (b, &d) in q8.d.iter().enumerate() {
            for i in 0..32 {
                x[b * 32 + i] = d * q8.qs[b * 32 + i] as f32;
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

    /// MoE `down`'s native-32-block fast path (`in_f` not a multiple of 256, e.g. DiffusionGemma's
    /// 704): the batched Q8_0 kernel must match the trusted `dequant_block` reference, for BOTH a
    /// 256-aligned and a NON-256-aligned `in_f` (704 = 22×32, not a multiple of 256), and across a
    /// multi-row batch (exercises the "decode weight row once, reuse across tokens" path itself).
    #[test]
    fn q8_0_32_batch_matches_dequant_reference() {
        for in_f in [256usize, 704] {
            let nb = in_f / 32;
            let mut w = det_bytes(nb * 34, 20);
            for k in 0..nb {
                put_f16(&mut w[k * 34..k * 34 + 2], 0.02);
            }
            let wref = dequant_block(DType::Q8_0, &w).unwrap();
            let xs: Vec<Vec<f32>> = (0..3).map(|i| det_x(in_f, 21 + i)).collect();
            let q8s: Vec<Q8x32> = xs.iter().map(|x| quantize_q8_32(x)).collect();
            let mut got = vec![0f32; 3];
            vec_dot_q8_0_32_batch(&w, &q8s, in_f, &mut got);
            for (r, q8) in q8s.iter().enumerate() {
                let want = dot(&wref, &dequant_q8_32(q8));
                assert!(
                    rel_err(got[r], want) < 1e-3,
                    "q8_0_32 in_f={in_f} row={r}: got {got_r}, want {want}",
                    got_r = got[r]
                );
            }
        }
    }

    /// The AVX2/AVX-512BW kernels for `vec_dot_q8_0_32_batch` must be BIT-IDENTICAL to the scalar
    /// oracle (`_scalar`) — integer dot products have no rounding, so summation order doesn't matter,
    /// only the final `d_w * d8 * iprod` formula does, which is shared. Runs whichever SIMD tier this
    /// CPU actually has (falls through to the same scalar fn on non-x86 or pre-AVX2 hardware, in which
    /// case the assertion is trivially true).
    #[test]
    fn q6k_pack_gemm_bit_identical_to_batch() {
        #[cfg(target_arch = "x86_64")]
        {
            if !(is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni")) {
                return; // the pack/gemm pair only exists on the VNNI path
            }
            let (in_f, out_f, m) = (512usize, 16usize, 5usize);
            let nb256 = in_f / 256;
            let mut w = det_bytes(out_f * nb256 * 210, 90);
            for o in 0..out_f {
                for b in 0..nb256 {
                    put_f16(
                        &mut w[(o * nb256 + b) * 210 + 208..(o * nb256 + b) * 210 + 210],
                        0.03,
                    );
                }
            }
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 91 + r as u64)))
                .collect();
            let pack = q6k_pack(&w, in_f, out_f);
            let bpr = w.len() / out_f;
            for (g, pg) in pack.groups.iter().enumerate() {
                let mut cols = vec![0f32; 8 * m];
                unsafe { q6k_gemm_group(pg, pack.nb, &q8s, &mut cols) };
                for i in 0..8 {
                    let o = g * 8 + i;
                    let mut want = vec![0f32; m];
                    vec_dot_q6k_batch(&w[o * bpr..o * bpr + bpr], &q8s, in_f, &mut want);
                    for r in 0..m {
                        assert_eq!(
                            cols[i * m + r].to_bits(),
                            want[r].to_bits(),
                            "q6k pack/gemm g={g} i={i} r={r}: got {}, want {}",
                            cols[i * m + r],
                            want[r],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn q4k_pack_gemm_bit_identical_to_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !(is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni")) {
                return; // the pack/gemm pair only exists on the VNNI path
            }
            let (in_f, out_f, m) = (512usize, 16usize, 5usize);
            let nb256 = in_f / 256;
            let mut w = det_bytes(out_f * nb256 * 144, 80);
            for o in 0..out_f {
                for b in 0..nb256 {
                    put_f16(
                        &mut w[(o * nb256 + b) * 144..(o * nb256 + b) * 144 + 2],
                        0.02,
                    );
                    put_f16(
                        &mut w[(o * nb256 + b) * 144 + 2..(o * nb256 + b) * 144 + 4],
                        0.01,
                    );
                }
            }
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 81 + r as u64)))
                .collect();
            let pack = unsafe { q4k_pack(&w, in_f, out_f) };
            let bpr = w.len() / out_f;
            for (g, pg) in pack.groups.iter().enumerate() {
                let mut cols = vec![0f32; 8 * m];
                unsafe { q4k_gemm_group(pg, pack.nb, &q8s, &mut cols) };
                for i in 0..8 {
                    let o = g * 8 + i;
                    let mut want = vec![0f32; m];
                    vec_dot_q4k_batch_scalar(&w[o * bpr..o * bpr + bpr], &q8s, in_f, &mut want);
                    for r in 0..m {
                        assert_eq!(
                            cols[i * m + r].to_bits(),
                            want[r].to_bits(),
                            "pack/gemm g={g} i={i} r={r}: got {}, want {}",
                            cols[i * m + r],
                            want[r]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn q32_batch8_bit_identical_to_scalar() {
        for q5 in [false, true] {
            let bpr = if q5 { 22usize } else { 34 };
            let in_f = 704usize;
            let nb = in_f / 32;
            let m = 5usize;
            let mut ws: Vec<Vec<u8>> = (0..8)
                .map(|i| {
                    let mut w = det_bytes(nb * bpr, 60 + i as u64);
                    for k in 0..nb {
                        put_f16(&mut w[k * bpr..k * bpr + 2], 0.02);
                    }
                    w
                })
                .collect();
            let _ = &mut ws;
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 70 + r as u64)))
                .collect();
            let mut got: Vec<Vec<f32>> = vec![vec![0f32; m]; 8];
            {
                let rows: [&[u8]; 8] = std::array::from_fn(|i| ws[i].as_slice());
                let mut it = got.iter_mut();
                let outs: [&mut [f32]; 8] =
                    std::array::from_fn(|_| it.next().unwrap().as_mut_slice());
                vec_dot_q32_batch8(rows, &q8s, in_f, outs, q5);
            }
            for i in 0..8 {
                let mut want = vec![0f32; m];
                if q5 {
                    vec_dot_q5_0_32_batch_scalar(&ws[i], &q8s, in_f, &mut want);
                } else {
                    vec_dot_q8_0_32_batch_scalar(&ws[i], &q8s, in_f, &mut want);
                }
                for r in 0..m {
                    assert_eq!(
                        got[i][r].to_bits(),
                        want[r].to_bits(),
                        "q32 batch8 q5={q5} row={i} r={r}: got {}, want {}",
                        got[i][r],
                        want[r]
                    );
                }
            }
        }
    }

    #[test]
    fn q5_0_32_batch_simd_bit_identical_to_scalar() {
        for in_f in [256usize, 704] {
            let nb = in_f / 32;
            let m = 5usize;
            let mut w = det_bytes(nb * 22, 41);
            for k in 0..nb {
                put_f16(&mut w[k * 22..k * 22 + 2], 0.02);
            }
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 42 + r as u64)))
                .collect();
            let mut simd_out = vec![0f32; m];
            vec_dot_q5_0_32_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_q5_0_32_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "q5_0_32 in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r]
                );
            }
        }
    }

    #[test]
    fn q8_0_32_batch_simd_bit_identical_to_scalar() {
        for in_f in [256usize, 704] {
            let nb = in_f / 32;
            let m = 5usize;
            let mut w = det_bytes(nb * 34, 25);
            for k in 0..nb {
                put_f16(&mut w[k * 34..k * 34 + 2], 0.02);
            }
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 26 + r as u64)))
                .collect();
            let mut simd_out = vec![0f32; m];
            vec_dot_q8_0_32_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_q8_0_32_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "q8_0_32 in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r],
                );
            }
        }
    }

    /// Same shape of check as `q8_0_32_batch_matches_dequant_reference`, for Q5_0's 5-bit (4 nibble
    /// + 1 high) native block.
    #[test]
    fn q5_0_32_batch_matches_dequant_reference() {
        for in_f in [256usize, 704] {
            let nb = in_f / 32;
            let mut w = det_bytes(nb * 22, 30);
            for k in 0..nb {
                put_f16(&mut w[k * 22..k * 22 + 2], 0.03);
            }
            let wref = dequant_block(DType::Q5_0, &w).unwrap();
            let xs: Vec<Vec<f32>> = (0..3).map(|i| det_x(in_f, 31 + i)).collect();
            let q8s: Vec<Q8x32> = xs.iter().map(|x| quantize_q8_32(x)).collect();
            let mut got = vec![0f32; 3];
            vec_dot_q5_0_32_batch(&w, &q8s, in_f, &mut got);
            for (r, q8) in q8s.iter().enumerate() {
                let want = dot(&wref, &dequant_q8_32(q8));
                assert!(
                    rel_err(got[r], want) < 1e-3,
                    "q5_0_32 in_f={in_f} row={r}: got {got_r}, want {want}",
                    got_r = got[r]
                );
            }
        }
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
