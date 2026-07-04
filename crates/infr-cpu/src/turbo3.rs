//! TurboQuant `turbo3` KV-cache quantization — a from-scratch port of llama.cpp's
//! `GGML_TYPE_TURBO3_0` (`ggml/src/ggml-turbo-quant.c`), bit-compatible block layout.
//!
//! One block = **128 elements → 50 bytes**: `fp16 norm` (2 B) + `qs[32]` (low 2 bits of each 3-bit
//! index, 4/byte) + `signs[16]` (bit 2 of each index, 8/byte). 3.125 bits/value.
//!
//! Pipeline (per 128-element group = one head_dim slice): L2-normalize → forward Walsh–Hadamard
//! Transform (fixed ±1 sign flips) → nearest 3-bit Lloyd-Max centroid → pack. The stored `norm` is
//! `grp_norm / ‖reconstructed centroids‖` so `centroid·norm` reproduces the original magnitude.
//!
//! Dequant returns the values in the **rotated (WHT) domain** — that is deliberate: attention can
//! rotate the query to match (the fused GPU path, later). For the CPU reference we instead apply the
//! inverse WHT ([`dequant_prefix_orig`]) to recover the original domain so the existing f32 SDPA runs
//! unchanged (mirrors [`crate::dequant_prefix_q8_0`]).

use half::f16;

const QK: usize = 128; // elements per block / WHT group
/// Bytes per turbo3 block: 2 (norm) + 32 (qs) + 16 (signs).
pub const BLOCK_BYTES: usize = 50;

/// Lloyd-Max optimal 3-bit centroids for N(0, 1/128) (llama.cpp `CENTROIDS_3BIT`).
const CENTROIDS: [f32; 8] = [
    -0.190207, -0.118786, -0.066822, -0.021663, 0.021663, 0.066822, 0.118786, 0.190207,
];

/// 1/sqrt(128) — the WHT normalization for a 128-element group.
const INV_SQRT: f32 = 0.088_388_35;

// Fixed ±1 diagonal sign vectors for the randomized WHT (llama.cpp `turbo_cpu_s1`/`s2`, extracted
// verbatim — MUST match the reference for bit-compatible caches). Stored as i8; applied as f32.
const S1: [i8; 128] = [
    -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1,
    1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1,
    -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1,
    -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1,
    -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1,
];
const S2: [i8; 128] = [
    1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1,
    1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1,
    -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1,
    -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1,
    1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1,
];

#[inline]
fn nearest_centroid(v: f32) -> usize {
    // Midpoint thresholds between the 8 centroids (llama.cpp `nearest_centroid_3bit`).
    if v < -0.154496 {
        0
    } else if v < -0.092804 {
        1
    } else if v < -0.044243 {
        2
    } else if v < 0.0 {
        3
    } else if v < 0.044243 {
        4
    } else if v < 0.092804 {
        5
    } else if v < 0.154496 {
        6
    } else {
        7
    }
}

/// In-place Hadamard butterfly (unnormalized), shared by the forward/inverse WHT.
#[inline]
fn butterfly(x: &mut [f32; QK]) {
    let mut h = 1;
    while h < QK {
        let mut i = 0;
        while i < QK {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

/// Forward WHT: `y = D(s2)·N·H·D(s1)·x` (N = 1/√128).
fn fwht(x: &mut [f32; QK]) {
    for i in 0..QK {
        x[i] *= S1[i] as f32;
    }
    butterfly(x);
    for i in 0..QK {
        x[i] *= INV_SQRT * S2[i] as f32;
    }
}

/// Inverse WHT: swap the sign diagonals (`H·N` is self-inverse, s1/s2 are ±1 diagonals).
fn fwht_inverse(x: &mut [f32; QK]) {
    for i in 0..QK {
        x[i] *= S2[i] as f32;
    }
    butterfly(x);
    for i in 0..QK {
        x[i] *= INV_SQRT * S1[i] as f32;
    }
}

/// Quantize one 128-element group into a 50-byte turbo3 block (`dst.len() >= 50`).
pub fn quantize_block(src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), QK);
    let mut buf = [0f32; QK];
    let mut norm_sq = 0f32;
    for j in 0..QK {
        buf[j] = src[j];
        norm_sq += buf[j] * buf[j];
    }
    let grp_norm = norm_sq.sqrt();
    let inv = if grp_norm > 1e-10 {
        1.0 / grp_norm
    } else {
        0.0
    };
    for v in buf.iter_mut() {
        *v *= inv;
    }
    fwht(&mut buf);

    let (norm_b, rest) = dst.split_at_mut(2);
    let (qs, signs) = rest.split_at_mut(32);
    qs.fill(0);
    signs.fill(0);
    let mut recon_sq = 0f32;
    for j in 0..QK {
        let idx = nearest_centroid(buf[j]);
        qs[j / 4] |= ((idx & 0x3) as u8) << ((j % 4) * 2);
        if idx & 0x4 != 0 {
            signs[j / 8] |= 1u8 << (j % 8);
        }
        recon_sq += CENTROIDS[idx] * CENTROIDS[idx];
    }
    // Norm correction: centroid·norm reproduces the original magnitude on average.
    let recon_norm = recon_sq.sqrt();
    let corrected = if recon_norm > 1e-10 {
        grp_norm / recon_norm
    } else {
        grp_norm
    };
    norm_b.copy_from_slice(&f16::from_f32(corrected).to_bits().to_le_bytes());
}

/// Quantize `src.len()` elements (a multiple of 128) into `src.len()/128` blocks in `dst`.
pub fn quantize_row(src: &[f32], dst: &mut [u8]) {
    for (b, chunk) in src.chunks_exact(QK).enumerate() {
        quantize_block(chunk, &mut dst[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES]);
    }
}

/// Dequant one block into the ROTATED (WHT) domain: `centroid[idx]·norm`, no inverse WHT.
fn dequant_block_rot(blk: &[u8], out: &mut [f32; QK]) {
    let norm = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
    let qs = &blk[2..34];
    let signs = &blk[34..50];
    for j in 0..QK {
        let low2 = (qs[j / 4] >> ((j % 4) * 2)) & 0x3;
        let hi1 = (signs[j / 8] >> (j % 8)) & 0x1;
        let idx = (low2 | (hi1 << 2)) as usize;
        out[j] = CENTROIDS[idx] * norm;
    }
}

/// Dequant the first `need` elements (a multiple of 128) to the ORIGINAL domain (rotated dequant +
/// inverse WHT), so the CPU reference attention reads normal-domain K/V. Mirrors
/// [`crate::dequant_prefix_q8_0`].
pub fn dequant_prefix_orig(bytes: &[u8], need: usize) -> Vec<f32> {
    let nb = need / QK;
    let mut out = vec![0f32; need];
    for b in 0..nb {
        let blk = &bytes[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let mut v = [0f32; QK];
        dequant_block_rot(blk, &mut v);
        fwht_inverse(&mut v);
        out[b * QK..(b + 1) * QK].copy_from_slice(&v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // WHT then inverse WHT is identity (orthonormal round-trip).
    #[test]
    fn wht_roundtrip_identity() {
        let mut x = [0f32; QK];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i as f32) - 64.0) * 0.013;
        }
        let orig = x;
        fwht(&mut x);
        fwht_inverse(&mut x);
        for i in 0..QK {
            assert!(
                (x[i] - orig[i]).abs() < 1e-4,
                "i={i}: {} vs {}",
                x[i],
                orig[i]
            );
        }
    }

    // Quantize → dequant-to-original recovers a random (roughly Gaussian) vector within 3-bit
    // tolerance. Real K/V vectors rotate to ~N(0,1/128) coefficients, which the centroids target;
    // averaged over many samples the reconstruction error is modest.
    #[test]
    fn quantize_dequant_orig_close() {
        // Box-Muller-ish pseudo-Gaussian from an LCG (deterministic, no rng dep).
        let mut s: u64 = 0x1234_5678;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u32 << 31) as f32) - 1.0 // ~U(-1,1)
        };
        let mut worst = 0f32;
        for _ in 0..32 {
            let mut src = [0f32; QK];
            for v in src.iter_mut() {
                *v = rnd() + rnd() + rnd(); // sum of uniforms ≈ Gaussian
            }
            let mut blk = [0u8; BLOCK_BYTES];
            quantize_block(&src, &mut blk);
            let got = dequant_prefix_orig(&blk, QK);
            let (mut num, mut den) = (0f32, 0f32);
            for i in 0..QK {
                num += (got[i] - src[i]).powi(2);
                den += src[i].powi(2);
            }
            worst = worst.max((num / den).sqrt());
        }
        // 3-bit rotated PolarQuant: per-vector relative L2 error ~10-20%.
        assert!(worst < 0.25, "worst relative L2 error too high: {worst}");
    }

    // Heavy-tailed vector (one dominant component ≫ the rest) — mimics real post-qk-norm K, where a
    // 128-group has max-abs far above the others. The randomized WHT spreads the dominant component
    // across coordinates so 3-bit quant stays bounded (measured ~18-21% here — the RHT works).
    #[test]
    fn quantize_dequant_heavy_tailed() {
        let mut s: u64 = 0xBEEF;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 23) as f32) - 1.0
        };
        let mut worst = 0f32;
        for trial in 0..16 {
            let mut src = [0f32; QK];
            for v in src.iter_mut() {
                *v = rnd() * 3.0;
            }
            src[trial * 7 % QK] = 500.0; // one dominant component
            let mut blk = [0u8; BLOCK_BYTES];
            quantize_block(&src, &mut blk);
            let got = dequant_prefix_orig(&blk, QK);
            let (mut num, mut den) = (0f32, 0f32);
            for i in 0..QK {
                num += (got[i] - src[i]).powi(2);
                den += src[i].powi(2);
            }
            worst = worst.max((num / den).sqrt());
        }
        // The RHT keeps a heavy-tailed vector quantizable; without it a raw 3-bit code would blow up.
        assert!(
            worst < 0.25,
            "RHT failed to bound heavy-tailed error: {worst}"
        );
    }
}
