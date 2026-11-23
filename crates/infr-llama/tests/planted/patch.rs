//! Weight-level bugs, written into a copy of a model file: the bytes of the whole file are edited
//! in memory at the tensors' own offsets (read off the original with `infr_gguf`), so the header
//! and every other tensor stay as they were and the backend under test loads the copy with its
//! production code.

use infr_core::{DType, WeightSource};
use infr_gguf::Gguf;

/// Elements per block of the K-quant formats, and the unit a "block" means for a float tensor.
const BLOCK: usize = 256;
/// Bytes per `Q4_K` block: `d` and `dmin` (f16 each), the packed 6-bit scales, then the nibbles.
const Q4K_BYTES: usize = 144;
/// Where a `Q4_K` block's nibbles start.
const Q4K_QS: usize = 16;
/// Bytes per `Q6_K` block; its super-block scale `d` (f16) is the last two.
const Q6K_BYTES: usize = 210;

/// A model file's bytes and its parsed header, edited tensor by tensor.
pub struct Patch {
    pub gguf: Gguf,
    pub bytes: Vec<u8>,
}

impl Patch {
    pub fn open(path: &std::path::Path) -> Self {
        Self {
            gguf: Gguf::open(path).expect("parse the model file"),
            bytes: std::fs::read(path).expect("read the model file"),
        }
    }

    /// A metadata integer of the model's architecture (`{arch}.{key}`).
    pub fn arch_u64(&self, key: &str) -> u64 {
        let m = self.gguf.metadata();
        let arch = m
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .expect("general.architecture");
        m.get(&format!("{arch}.{key}"))
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("{arch}.{key} missing"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.gguf.tensors().iter().any(|t| t.name == name)
    }

    /// The number of [`BLOCK`]-value blocks tensor `name` holds.
    pub fn blocks(&self, name: &str) -> usize {
        let (_, shape, _) = self.tensor(name);
        shape.iter().product::<usize>() / BLOCK
    }

    /// The number of rows (output features) of the matrix `name`.
    pub fn rows(&self, name: &str) -> usize {
        self.tensor(name).1[1]
    }

    /// `(dtype, shape, byte range in the file)` of tensor `name`.
    fn tensor(&self, name: &str) -> (DType, Vec<usize>, std::ops::Range<usize>) {
        let t = self
            .gguf
            .tensors()
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("no tensor {name}"));
        let (off, len) = self.gguf.tensor_file_range(name).expect("tensor range");
        (t.dtype, t.shape.clone(), off as usize..off as usize + len)
    }

    /// Every value of tensor `name`, or only those of its block `block`, times `factor`. A K-quant
    /// block is scaled through its super-block scales (`Q4_K`: `d` and `dmin`, so the block's
    /// every value scales; `Q6_K`: `d`), each rounded to f16, so the factor lands within an f16
    /// rounding of `factor`; a float tensor's values are scaled one by one.
    pub fn scale(&mut self, name: &str, block: Option<usize>, factor: f32) {
        let (dt, shape, range) = self.tensor(name);
        let numel: usize = shape.iter().product();
        let blocks = match block {
            Some(b) => b..b + 1,
            None => 0..numel / BLOCK,
        };
        assert!(blocks.end * BLOCK <= numel, "{name} has no block {block:?}");
        let data = &mut self.bytes[range];
        let scale_f16 = |b: &mut [u8]| {
            let v = half::f16::from_le_bytes([b[0], b[1]]).to_f32() * factor;
            b.copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
        };
        for bi in blocks {
            match dt {
                DType::Q4K => {
                    let blk = &mut data[bi * Q4K_BYTES..(bi + 1) * Q4K_BYTES];
                    scale_f16(&mut blk[0..2]);
                    scale_f16(&mut blk[2..4]);
                }
                DType::Q6K => {
                    let blk = &mut data[bi * Q6K_BYTES..(bi + 1) * Q6K_BYTES];
                    scale_f16(&mut blk[Q6K_BYTES - 2..]);
                }
                DType::F16 => {
                    for e in data[bi * BLOCK * 2..(bi + 1) * BLOCK * 2]
                        .as_chunks_mut::<2>()
                        .0
                    {
                        scale_f16(e);
                    }
                }
                DType::F32 => {
                    for e in data[bi * BLOCK * 4..(bi + 1) * BLOCK * 4]
                        .as_chunks_mut::<4>()
                        .0
                    {
                        *e = (f32::from_le_bytes(*e) * factor).to_le_bytes();
                    }
                }
                _ => panic!("{name}: no block scaling for {dt:?}"),
            }
        }
    }

    /// The low and high nibble of every packed byte of the `Q4_K` tensor `name` swapped: what a
    /// dequantizer reading the nibbles in the wrong order computes. Each value then decodes the
    /// code of the value 32 elements away, under its own sub-block's scale and minimum.
    pub fn swap_nibbles(&mut self, name: &str) {
        let (dt, _, range) = self.tensor(name);
        assert_eq!(dt, DType::Q4K, "{name}: the nibble map is a Q4_K bug");
        for blk in self.bytes[range].as_chunks_mut::<Q4K_BYTES>().0 {
            for b in &mut blk[Q4K_QS..] {
                *b = b.rotate_left(4);
            }
        }
    }

    /// Tensors `a` and `b` (same dtype and shape) trade places.
    pub fn swap(&mut self, a: &str, b: &str) {
        let (da, sa, ra) = self.tensor(a);
        let (db, sb, rb) = self.tensor(b);
        assert_eq!((da, &sa), (db, &sb), "{a} and {b} differ in kind");
        let copy = self.bytes[ra.clone()].to_vec();
        self.bytes.copy_within(rb.clone(), ra.start);
        self.bytes[rb].copy_from_slice(&copy);
    }

    /// Every value of tensor `name` zeroed (an f32 tensor).
    pub fn zero(&mut self, name: &str) {
        let (dt, _, range) = self.tensor(name);
        assert_eq!(dt, DType::F32, "{name}");
        self.bytes[range].fill(0);
    }

    /// Within each `head_dim`-row head of the projection `name` (output rows, each a whole number
    /// of blocks), and across the `head_dim` values of the per-head norm weight `norm` when the
    /// model has one, the rows reordered so that the rope rotates the other pairing: on a NEOX
    /// model (`neox`, pairs `(p, p + head_dim/2)`) the pairs become `(2p, 2p + 1)`, on a NORM
    /// model the reverse. Rope is the only thing that tells rows of one head apart before the
    /// attention scores, which are sums over the head, so this is exactly a rope kernel pairing
    /// the wrong elements.
    pub fn misorder_rope_pairs(
        &mut self,
        name: &str,
        norm: Option<&str>,
        head_dim: usize,
        neox: bool,
    ) {
        let half = head_dim / 2;
        // `from[j]`: the original row that lands at row `j` of the head.
        let from: Vec<usize> = (0..head_dim)
            .map(|j| match (neox, j < half) {
                (true, true) => 2 * j,
                (true, false) => 2 * (j - half) + 1,
                (false, _) if j % 2 == 0 => j / 2,
                (false, _) => j / 2 + half,
            })
            .collect();
        let (dt, shape, range) = self.tensor(name);
        let rows = shape[1];
        assert_eq!(
            rows % head_dim,
            0,
            "{name}: {rows} rows are not whole heads"
        );
        let row_bytes = infr_gguf::nbytes(dt, shape[0]);
        let old = self.bytes[range.clone()].to_vec();
        for h in 0..rows / head_dim {
            for (j, &f) in from.iter().enumerate() {
                let dst = range.start + (h * head_dim + j) * row_bytes;
                let src = (h * head_dim + f) * row_bytes;
                self.bytes[dst..dst + row_bytes].copy_from_slice(&old[src..src + row_bytes]);
            }
        }
        if let Some(norm) = norm {
            let (dt, shape, range) = self.tensor(norm);
            assert_eq!((dt, shape), (DType::F32, vec![head_dim]), "{norm}");
            let old = self.bytes[range.clone()].to_vec();
            for (j, &f) in from.iter().enumerate() {
                self.bytes[range.start + 4 * j..range.start + 4 * j + 4]
                    .copy_from_slice(&old[4 * f..4 * f + 4]);
            }
        }
    }
}
