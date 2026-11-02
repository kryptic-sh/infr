//! Tile-streamed EXL3 projections without expanded weight storage.

use crate::exl3::{decode_3inst, hadamard128, tile_perm, window16_funnel, Codebook};
use infr_core::{Error, Exl3Spec, Result};
use rayon::prelude::*;

pub fn linear(
    trellis: &[u8],
    suh: &[f32],
    svh: &[f32],
    x: &[f32],
    m: usize,
    spec: Exl3Spec,
) -> Result<Vec<f32>> {
    let expected = spec.trellis_bytes()?;
    let k = spec.in_f as usize;
    let n = spec.out_f as usize;
    let input_len = m
        .checked_mul(k)
        .ok_or_else(|| Error::backend("exl3: input size overflow"))?;
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| Error::backend("exl3: output size overflow"))?;
    if trellis.len() != expected || suh.len() != k || svh.len() != n || x.len() != input_len {
        return Err(Error::backend("exl3: projection buffer size mismatch"));
    }
    if suh.iter().chain(svh).any(|v| !v.is_finite()) {
        return Err(Error::backend("exl3: non-finite scale"));
    }
    let mut xh = vec![0.0; x.len()];
    xh.par_chunks_mut(k)
        .zip(x.par_chunks(k))
        .for_each(|(row, input)| {
            for (j, v) in row.iter_mut().enumerate() {
                *v = input[spec.input_index(j)] * suh[j];
            }
            for block in row.as_chunks_mut::<128>().0 {
                hadamard128(block);
            }
        });
    let n_blocks = n / 128;
    let row_groups = m.div_ceil(8);
    let scratch_len = row_groups
        .checked_mul(n_blocks)
        .and_then(|v| v.checked_mul(1024))
        .ok_or_else(|| Error::backend("exl3: scratch size overflow"))?;
    let mut bands = vec![0.0_f32; scratch_len];
    let perm = tile_perm(spec.bits);
    let codebook = Codebook::from(spec.codebook);
    let tile_bytes = 32 * spec.bits as usize;
    let word_count = 8 * spec.bits as usize;
    bands
        .par_chunks_mut(1024)
        .enumerate()
        .for_each(|(job, acc)| {
            let row0 = (job / n_blocks) * 8;
            let out0 = (job % n_blocks) * 128;
            let rows = (m - row0).min(8);
            let mut words = [0_u32; 64];
            let mut tile = [0.0_f32; 256];
            for ki in 0..k / 16 {
                for nj in 0..8 {
                    let offset = (ki * (n / 16) + out0 / 16 + nj) * tile_bytes;
                    for (dst, src) in words[..word_count]
                        .iter_mut()
                        .zip(trellis[offset..offset + tile_bytes].as_chunks::<4>().0)
                    {
                        *dst = u32::from_le_bytes(*src);
                    }
                    for s in 0..256 {
                        let b0 =
                            s * spec.bits as usize + spec.bits as usize + 256 * spec.bits as usize
                                - 16;
                        tile[perm[s] as usize] =
                            decode_3inst(codebook, window16_funnel(&words[..word_count], b0));
                    }
                    for r in 0..rows {
                        let xr = &xh[(row0 + r) * k + ki * 16..];
                        let yr = &mut acc[r * 128 + nj * 16..];
                        for (i, drow) in tile.as_chunks::<16>().0.iter().enumerate() {
                            for j in 0..16 {
                                yr[j] += xr[i] * drow[j];
                            }
                        }
                    }
                }
            }
            for row in acc.as_chunks_mut::<128>().0.iter_mut().take(rows) {
                hadamard128(row);
                for (v, scale) in row.iter_mut().zip(&svh[out0..out0 + 128]) {
                    *v *= scale;
                }
            }
        });
    let mut out = vec![0.0; output_len];
    out.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
        for b in 0..n_blocks {
            let band = &bands[((r / 8) * n_blocks + b) * 1024 + (r % 8) * 128..];
            for (j, &v) in band[..128].iter().enumerate() {
                row[spec.output_index(b * 128 + j)] = v;
            }
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exl3::{apply_scales, decode_matrix};
    use infr_core::{Exl3Codebook, Exl3OutputLayout};

    #[test]
    fn streamed_projection_matches_dense_reference_all_rates_and_tail_batches() {
        for (codebook, bits) in [Exl3Codebook::Cb0, Exl3Codebook::Mcg, Exl3Codebook::Mul1]
            .into_iter()
            .flat_map(|codebook| (1..=8).map(move |bits| (codebook, bits)))
        {
            let spec = Exl3Spec {
                in_f: 256,
                out_f: 384,
                bits,
                codebook,
                input_layout: Exl3OutputLayout::Identity,
                output_layout: Exl3OutputLayout::RopeInterleaved {
                    heads: 6,
                    head_dim: 64,
                    rotary_dim: 48,
                },
            };
            let packed: Vec<u8> = (0..spec.trellis_bytes().unwrap())
                .map(|i| (i.wrapping_mul(79) ^ (i >> 5)) as u8)
                .collect();
            let suh: Vec<f32> = (0..256).map(|i| ((i % 19) as f32 - 9.0) * 0.017).collect();
            let svh: Vec<f32> = (0..384).map(|i| ((i % 23) as f32 - 11.0) * 0.009).collect();
            let words: Vec<u16> = packed
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect();
            let mut d = decode_matrix(Codebook::from(codebook), bits, &words, 256, 384);
            for row in d.as_chunks_mut::<384>().0 {
                for block in row.as_chunks_mut::<128>().0 {
                    hadamard128(block);
                }
            }
            for r0 in [0, 128] {
                for j in 0..384 {
                    let mut col = [0.0; 128];
                    for i in 0..128 {
                        col[i] = d[(r0 + i) * 384 + j];
                    }
                    hadamard128(&mut col);
                    for i in 0..128 {
                        d[(r0 + i) * 384 + j] = col[i];
                    }
                }
            }
            apply_scales(&mut d, &suh, &svh, 256, 384);
            for m in [1, 3, 8, 9] {
                let x: Vec<f32> = (0..m * 256).map(|i| (i as f32 * 0.017).sin()).collect();
                let got = linear(&packed, &suh, &svh, &x, m, spec).unwrap();
                for r in 0..m {
                    for j in 0..384 {
                        let expected: f32 = (0..256).map(|i| x[r * 256 + i] * d[i * 384 + j]).sum();
                        let actual = got[r * 384 + spec.output_index(j)];
                        assert!(
                            (actual - expected).abs() < 3e-5,
                            "codebook={codebook:?} bits={bits} m={m} r={r} j={j}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn grouped_input_and_output_heads_match_independent_dense_reference() {
        for (codebook, bits) in [Exl3Codebook::Cb0, Exl3Codebook::Mcg, Exl3Codebook::Mul1]
            .into_iter()
            .flat_map(|codebook| (1..=8).map(move |bits| (codebook, bits)))
        {
            let spec = Exl3Spec {
                in_f: 384,
                out_f: 512,
                bits,
                codebook,
                input_layout: Exl3OutputLayout::GroupedValueHeads {
                    prefix: 0,
                    key_heads: 3,
                    value_heads: 6,
                    head_dim: 64,
                },
                output_layout: Exl3OutputLayout::GroupedValueHeads {
                    prefix: 128,
                    key_heads: 3,
                    value_heads: 6,
                    head_dim: 64,
                },
            };
            let packed: Vec<u8> = (0..spec.trellis_bytes().unwrap())
                .map(|i| ((i * 79) ^ (i >> 5)) as u8)
                .collect();
            let words: Vec<u16> = packed
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| u16::from_le_bytes(*p))
                .collect();
            let suh: Vec<f32> = (0..384).map(|i| ((i % 19) as f32 - 9.0) * 0.017).collect();
            let svh: Vec<f32> = (0..512).map(|i| ((i % 23) as f32 - 11.0) * 0.009).collect();
            let mut d = decode_matrix(Codebook::from(codebook), bits, &words, 384, 512);
            for row in d.as_chunks_mut::<512>().0 {
                for block in row.as_chunks_mut::<128>().0 {
                    hadamard128(block);
                }
            }
            for r0 in [0, 128, 256] {
                for j in 0..512 {
                    let mut col = [0.0; 128];
                    for i in 0..128 {
                        col[i] = d[(r0 + i) * 512 + j];
                    }
                    hadamard128(&mut col);
                    for i in 0..128 {
                        d[(r0 + i) * 512 + j] = col[i];
                    }
                }
            }
            apply_scales(&mut d, &suh, &svh, 384, 512);
            let tiled_heads = [0, 3, 1, 4, 2, 5];
            for m in [1, 3, 9] {
                let x: Vec<f32> = (0..m * 384).map(|i| (i as f32 * 0.017).sin()).collect();
                let got = linear(&packed, &suh, &svh, &x, m, spec).unwrap();
                for row in 0..m {
                    for j in 0..512 {
                        let expected: f32 = (0..384)
                            .map(|i| {
                                let logical = tiled_heads[i / 64] * 64 + i % 64;
                                x[row * 384 + logical] * d[i * 512 + j]
                            })
                            .sum();
                        let logical = if j < 128 {
                            j
                        } else {
                            128 + tiled_heads[(j - 128) / 64] * 64 + (j - 128) % 64
                        };
                        let actual = got[row * 512 + logical];
                        assert!(
                            (actual - expected).abs() < 3e-5,
                            "{codebook:?} bits={bits} m={m} j={j}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn malformed_buffers_return_errors() {
        let spec = Exl3Spec {
            in_f: 128,
            out_f: 128,
            bits: 4,
            codebook: Exl3Codebook::Cb0,
            input_layout: Exl3OutputLayout::Identity,
            output_layout: Exl3OutputLayout::Identity,
        };
        assert!(linear(&[], &[1.0; 128], &[1.0; 128], &[1.0; 128], 1, spec).is_err());
        assert!(linear(
            &vec![0; 8192],
            &[f32::NAN; 128],
            &[1.0; 128],
            &[1.0; 128],
            1,
            spec
        )
        .is_err());
    }
}
