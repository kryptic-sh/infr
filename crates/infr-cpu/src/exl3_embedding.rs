//! Bounded EXL3 embedding-row reconstruction without expanding the table.

use half::f16;
use infr_core::{Error, Exl3EmbeddingSpec, Result};
use rayon::prelude::*;

use crate::exl3::{decode_3inst, Codebook};

fn fail(message: &str) -> Error {
    Error::Backend(format!("exl3 embedding: {message}"))
}

fn state(stream: &[u8], index: usize, bits: usize) -> u16 {
    let mut word = 0u16;
    for b in 0..16 {
        let position = ((index + 256 - b / bits) % 256) * bits + b % bits;
        word |= u16::from((stream[position / 8] >> (position % 8)) & 1) << b;
    }
    word
}

pub fn lookup(
    trellis: &[u8],
    signs: &[u8],
    ids: &[u32],
    spec: Exl3EmbeddingSpec,
) -> Result<Vec<f32>> {
    let row_bytes = spec.row_bytes()?;
    if trellis.len() != spec.trellis_bytes()? || signs.len() != spec.sign_bytes()? {
        return Err(fail("invalid packed table or sign length"));
    }
    if ids.iter().any(|&id| id >= spec.vocab) {
        return Err(fail("token id is out of range"));
    }
    let width = spec.width as usize;
    let count = ids
        .len()
        .checked_mul(width)
        .ok_or_else(|| fail("row size overflow"))?;
    let mut out = vec![0f32; count];
    let group_bytes = 2 * (16 * spec.bits as usize + 1);
    out.par_chunks_mut(width)
        .zip(ids.par_iter())
        .try_for_each(|(row, &id)| -> Result<()> {
            let packed = &trellis[id as usize * row_bytes..(id as usize + 1) * row_bytes];
            for (group, dst) in row.chunks_mut(256).enumerate() {
                let bytes = &packed[group * group_bytes..(group + 1) * group_bytes];
                let scale = f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
                if !scale.is_finite() {
                    return Err(fail("nonfinite row scale"));
                }
                let mut values = [0f32; 256];
                for (i, v) in values.iter_mut().enumerate() {
                    *v = decode_3inst(Codebook::Cb2, state(&bytes[2..], i, spec.bits as usize))
                        * scale;
                }
                for stride in [1, 2, 4, 8, 16, 32, 64, 128] {
                    for block in values.chunks_exact_mut(2 * stride) {
                        for i in 0..stride {
                            let a = block[i];
                            let b = block[i + stride];
                            block[i] = a + b;
                            block[i + stride] = a - b;
                        }
                    }
                }
                for (i, dst) in dst.iter_mut().enumerate() {
                    let s = 2 * (group * 256 + i);
                    let sign = f16::from_le_bytes([signs[s], signs[s + 1]]).to_f32();
                    let value = f16::from_f32((values[i] / 16.0) * sign).to_f32();
                    if !value.is_finite() {
                        return Err(fail("nonfinite reconstructed row"));
                    }
                    *dst = value;
                }
            }
            Ok(())
        })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(bits: u32, width: u32, vocab: u32) -> (Vec<u8>, Vec<u8>, Exl3EmbeddingSpec) {
        let spec = Exl3EmbeddingSpec { width, vocab, bits };
        let mut packed = vec![0; spec.trellis_bytes().unwrap()];
        let stride = spec.row_bytes().unwrap();
        let group_bytes = 2 * (16 * bits as usize + 1);
        for row in 0..vocab as usize {
            for group in 0..width.div_ceil(256) as usize {
                let at = row * stride + group * group_bytes;
                packed[at..at + 2].copy_from_slice(
                    &f16::from_f32(0.0137 * (row + group + 1) as f32).to_le_bytes(),
                );
                for (i, b) in packed[at + 2..at + group_bytes].iter_mut().enumerate() {
                    *b = (i * 37 + row * 83 + group * 17) as u8;
                }
            }
        }
        let signs = (0..spec.sign_bytes().unwrap() / 2)
            .flat_map(|i| f16::from_f32(if i % 3 == 0 { -1.0 } else { 1.0 }).to_le_bytes())
            .collect();
        (packed, signs, spec)
    }

    #[test]
    fn circular_symbol_order_and_reference_zero_row() {
        let symbols: Vec<u8> = (0..128)
            .map(|i| ((2 * i) % 16) as u8 | ((((2 * i + 1) % 16) as u8) << 4))
            .collect();
        assert_eq!(
            (0..4).map(|i| state(&symbols, i, 4)).collect::<Vec<_>>(),
            [0xdef0, 0xef01, 0xf012, 0x0123]
        );
        let spec = Exl3EmbeddingSpec {
            width: 256,
            vocab: 1,
            bits: 4,
        };
        let mut packed = vec![0; spec.trellis_bytes().unwrap()];
        packed[..2].copy_from_slice(&f16::ONE.to_le_bytes());
        let signs: Vec<_> = (0..256).flat_map(|_| f16::ONE.to_le_bytes()).collect();
        let row = lookup(&packed, &signs, &[0], spec).unwrap();
        assert_eq!(row[0], -55.25);
        assert!(row[1..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rows_match_explicit_walsh_reference_all_rates_and_partial_groups() {
        for bits in 1..=8 {
            let (packed, signs, spec) = fixture(bits, 513, 4);
            let ids = [3, 0, 3, 2];
            let actual = lookup(&packed, &signs, &ids, spec).unwrap();
            let group_bytes = 2 * (16 * bits as usize + 1);
            for (r, id) in ids.into_iter().enumerate() {
                for group in 0..3 {
                    let at = id as usize * spec.row_bytes().unwrap() + group * group_bytes;
                    let bytes = &packed[at..at + group_bytes];
                    let gain = f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
                    let mut decoded = [0f32; 256];
                    for (i, d) in decoded.iter_mut().enumerate() {
                        let mut word = 0u16;
                        for b in 0..16 {
                            let symbol =
                                (i as i32 - b as i32 / bits as i32).rem_euclid(256) as usize;
                            let p = symbol * bits as usize + b % bits as usize;
                            if bytes[2 + p / 8] & (1 << (p % 8)) != 0 {
                                word |= 1 << b;
                            }
                        }
                        *d = decode_3inst(Codebook::Cb2, word) * gain;
                    }
                    for j in 0..(513 - group * 256).min(256) {
                        let mut sum = 0f64;
                        for (i, &d) in decoded.iter().enumerate() {
                            sum += f64::from(d)
                                * if (i & j).count_ones().is_multiple_of(2) {
                                    1.0
                                } else {
                                    -1.0
                                };
                        }
                        let s = 2 * (group * 256 + j);
                        let sign = f16::from_le_bytes([signs[s], signs[s + 1]]).to_f32();
                        let expected = f16::from_f64(sum / 16.0 * f64::from(sign)).to_f32();
                        let got = actual[r * 513 + group * 256 + j];
                        assert!(
                            (got - expected).abs() <= 0.001 * expected.abs().max(0.02),
                            "bits={bits} row={r} group={group} j={j}: {got} != {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pinned_e4b_first_and_last_embedding_groups_match_independent_half_goldens() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/exl3_embedding_rows.json"))
                .unwrap();
        let hex = |value: &serde_json::Value| -> Vec<u8> {
            value
                .as_str()
                .unwrap()
                .as_bytes()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
                .collect()
        };
        for entry in fixture["entries"].as_array().unwrap() {
            let packed = hex(&entry["packed"]);
            let signs = hex(&entry["signs"]);
            let expected = hex(&entry["expected_f16"]);
            let geometry = Exl3EmbeddingSpec {
                width: 512,
                vocab: 3,
                bits: 4,
            };
            let actual = lookup(&packed, &signs, &[2, 0, 2, 1], geometry).unwrap();
            for (row, source) in [2, 0, 2, 1].into_iter().enumerate() {
                for i in 0..512 {
                    let p = 2 * (source * 512 + i);
                    let value = f16::from_le_bytes([expected[p], expected[p + 1]]).to_f32();
                    assert_eq!(
                        actual[row * 512 + i],
                        value,
                        "{} row={source} index={i}",
                        entry["name"]
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_bad_geometry_lengths_ids_and_nonfinite_parameters() {
        let (packed, signs, spec) = fixture(4, 512, 2);
        assert!(lookup(&packed, &signs, &[], spec).unwrap().is_empty());
        assert!(lookup(&packed[..packed.len() - 1], &signs, &[0], spec).is_err());
        assert!(lookup(&packed, &signs[..signs.len() - 1], &[0], spec).is_err());
        assert!(lookup(&packed, &signs, &[2], spec).is_err());
        for bad in [
            Exl3EmbeddingSpec { width: 0, ..spec },
            Exl3EmbeddingSpec { vocab: 0, ..spec },
            Exl3EmbeddingSpec { bits: 0, ..spec },
            Exl3EmbeddingSpec { bits: 9, ..spec },
        ] {
            assert!(lookup(&packed, &signs, &[0], bad).is_err());
        }
        let mut bad = packed.clone();
        bad[..2].copy_from_slice(&f16::NAN.to_le_bytes());
        assert!(lookup(&bad, &signs, &[0], spec).is_err());
        let mut bad_signs = signs.clone();
        bad_signs[..2].copy_from_slice(&f16::INFINITY.to_le_bytes());
        assert!(lookup(&packed, &bad_signs, &[0], spec).is_err());
    }
}
