//! A test tap on the activation side of every dense `Op::Linear`, and a test-only switch that
//! simulates an activation quantizer inside the reference mode. Both are off unless a test turns
//! them on, on the thread that executes the graphs; untapped, the interpreter pays one
//! thread-local read per `Op::Linear`.
//!
//! **The tap** ([`start`] / [`take`]) records, for each `Op::Linear` it sees and each activation
//! row a test asked for, the row entering the op, the op's output for that row, and the same
//! row's output recomputed exactly (the weight decoded to f32 and dotted in f64). In production
//! mode the gap between the last two is that op's OWN int8-activation error on that row, free of
//! whatever upstream ops passed on to it; in reference mode it is the f32 dot's rounding.
//!
//! **The simulation** ([`simulate`]) makes the reference mode's `Op::Linear` dot each activation
//! row after a round trip through a quantizer ([`ActSim`]), so the quality of an activation
//! scheme can be scored end to end before any kernel computes it: an int8 kernel's result is the
//! exact dot of the decoded weight with the decoded activation, up to float summation order.

use infr_core::DType;
use std::cell::RefCell;

/// One `Op::Linear` the tap saw, in execution order.
#[derive(Clone, Debug)]
pub struct TappedLinear {
    /// The op's index in its graph.
    pub op_idx: usize,
    /// The weight's storage type.
    pub dtype: DType,
    /// Rows in the op's batch.
    pub m: usize,
    pub in_f: usize,
    pub out_f: usize,
    /// The requested rows that exist in this op's batch.
    pub rows: Vec<TappedRow>,
}

/// One activation row of a [`TappedLinear`].
#[derive(Clone, Debug)]
pub struct TappedRow {
    /// The row's index in the op's batch.
    pub row: usize,
    /// The activation row entering the op (`in_f` values).
    pub x: Vec<f32>,
    /// The op's output for this row (`out_f` values).
    pub out: Vec<f32>,
    /// The row's output with the weight decoded to f32 and each dot taken in f64.
    pub exact: Vec<f64>,
    /// [`Self::exact`] with the row first sent through the production 256-block quantizer
    /// (`ActSim { block: 256, keep: 0, qmax: 127 }`): what an int8 kernel on that activation
    /// computes, up to float summation order.
    pub exact_q256: Vec<f64>,
    /// [`Self::exact_q256`] for the native 32-block quantizer (`block: 32`).
    pub exact_q32: Vec<f64>,
}

/// An activation quantizer for [`simulate`]: each row is cut into blocks of `block` elements;
/// in each block up to `keep` of the largest magnitudes pass through unquantized, and the rest
/// are rounded to the nearest multiple of `amax / qmax` (with `amax` the largest magnitude left)
/// and clamped to `±qmax`. A candidate passes only when its magnitude exceeds `ratio` times the
/// largest magnitude outside the block's top `keep` (`ratio: 0` passes all `keep`).
/// With `split`, the passed elements are not exact but quantized too, on a scale of their own
/// (`max / qmax` over them), and what that leaves of each is quantized with the rest: the block is
/// the sum of a sparse part and a dense part, each `qmax`-bounded integers times one scale.
/// With `sub > 0` (and nothing passed) each `sub`-element piece of a block takes the block's
/// scale divided by a power of two, `2^e` with `e <= emax` the largest that keeps the piece within
/// `qmax`: per-piece scales an integer kernel can apply as a shift.
/// `ActSim { block: 256, keep: 0, ratio: 0, qmax: 127, split: false, sub: 0, emax: 0 }` is the
/// production `Q8` activation, `block: 32` the native-block `Q8x32` one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActSim {
    pub block: usize,
    pub keep: usize,
    pub ratio: u32,
    pub qmax: u32,
    pub split: bool,
    pub sub: usize,
    pub emax: u32,
}

impl ActSim {
    /// The indices of `blk` this quantizer passes through unquantized.
    pub fn kept(&self, blk: &[f32]) -> Vec<usize> {
        if self.keep == 0 {
            return Vec::new();
        }
        // Magnitude order, lowest index first on a tie.
        let mut order: Vec<usize> = (0..blk.len()).collect();
        order.sort_by(|&a, &b| blk[b].abs().total_cmp(&blk[a].abs()).then(a.cmp(&b)));
        let k = self.keep.min(blk.len());
        let next = order.get(k).map_or(0.0, |&i| blk[i].abs());
        let floor = self.ratio as f32 * next;
        order[..k]
            .iter()
            .copied()
            .filter(|&i| blk[i].abs() > floor)
            .collect()
    }

    /// `x` after a round trip through this quantizer. A tail shorter than `block` is a block of
    /// its own.
    pub fn round_trip(&self, x: &[f32]) -> Vec<f32> {
        let qmax = self.qmax as f32;
        let mut out = x.to_vec();
        // The production quantizer's arithmetic (`quant::quantize_q8`): `(v * (1/d)).round()`.
        let quant = |v: f32, d: f32| {
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            (v * id).round().clamp(-qmax, qmax) * d
        };
        for blk in out.chunks_mut(self.block) {
            if self.sub > 0 {
                let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / qmax;
                for piece in blk.chunks_mut(self.sub) {
                    let a = piece.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let mut e = 0;
                    while e < self.emax && a * (2u32 << e) as f32 <= amax {
                        e += 1;
                    }
                    let dp = d / (1u32 << e) as f32;
                    for v in piece.iter_mut() {
                        *v = quant(*v, dp);
                    }
                }
                continue;
            }
            let kept = self.kept(blk);
            // The sparse part, when split: each passed element on the passed elements' own scale.
            // Split: the passed elements quantized on their own scale level by level (the
            // production quantizer's rule, `quant::quantize_q8`), what is left of them staying
            // with the rest. Unsplit: the passed elements exact, the rest on their own.
            let mut dense: Vec<f32> = blk.to_vec();
            let mut levels: Vec<(f32, Vec<usize>, Vec<f32>)> = Vec::new();
            if self.split && !kept.is_empty() {
                let rest = (0..blk.len())
                    .filter(|i| !kept.contains(i))
                    .fold(0f32, |m, i| m.max(blk[i].abs()));
                let amax = kept.iter().fold(0f32, |m, &i| m.max(blk[i].abs()));
                for _ in 0..crate::quant::HI_LEVELS {
                    let a = kept.iter().fold(0f32, |m, &i| m.max(dense[i].abs()));
                    if !(a > rest && a > amax * f32::EPSILON && a.is_finite()) {
                        break;
                    }
                    let dh = a / qmax;
                    let codes: Vec<f32> = kept
                        .iter()
                        .map(|&i| {
                            let c = (dense[i] * (1.0 / dh)).round().clamp(-qmax, qmax);
                            dense[i] -= c * dh;
                            c
                        })
                        .collect();
                    levels.push((dh, kept.clone(), codes));
                }
            } else {
                for &i in &kept {
                    dense[i] = 0.0;
                }
            }
            let amax = dense.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / qmax;
            let mut vals: Vec<f32> = dense.iter().map(|&v| quant(v, d)).collect();
            for (dh, idx, codes) in &levels {
                for (&i, &c) in idx.iter().zip(codes) {
                    vals[i] += c * dh;
                }
            }
            if !self.split {
                for &i in &kept {
                    vals[i] = blk[i];
                }
            }
            blk.copy_from_slice(&vals);
        }
        out
    }
}

thread_local! {
    static TAP: RefCell<Option<(Vec<usize>, Vec<TappedLinear>)>> = const { RefCell::new(None) };
    static SIM: RefCell<Option<(ActSim, ActSim)>> = const { RefCell::new(None) };
}

/// Start recording the activation rows `rows` of every `Op::Linear` on this thread, dropping
/// anything recorded before.
pub fn start(rows: &[usize]) {
    TAP.with(|t| *t.borrow_mut() = Some((rows.to_vec(), Vec::new())));
}

/// Stop recording and return every op recorded since [`start`], in execution order.
pub fn take() -> Vec<TappedLinear> {
    TAP.with(|t| t.borrow_mut().take().map(|(_, v)| v).unwrap_or_default())
}

/// Make the REFERENCE mode's `Op::Linear` on this thread dot every activation row after a round
/// trip through a simulated quantizer: `sim` for the weights production feeds the 256-element
/// `Q8` activation, `sim32` for those it feeds the 32-element `Q8x32` one ([`uses_q8_32`]).
/// `None` restores the plain f32 activation. Production mode ignores it.
pub fn simulate(sim: Option<(ActSim, ActSim)>) {
    SIM.with(|s| *s.borrow_mut() = sim);
}

/// The simulated quantizer for a weight of `dt` over `in_f` inputs, when one is set.
pub(crate) fn simulated(dt: DType, in_f: usize) -> Option<ActSim> {
    SIM.with(|s| {
        s.borrow()
            .map(|(a, b)| if uses_q8_32(dt, in_f) { b } else { a })
    })
}

/// Whether production quantizes this weight's activations to 32-element blocks (`Q8x32`) rather
/// than 256-element ones: mirrored from the CPU's `Op::Linear` dispatch.
pub fn uses_q8_32(dt: DType, in_f: usize) -> bool {
    matches!(
        dt,
        DType::Q4_0
            | DType::Q4_1
            | DType::Q5_0
            | DType::Q5_1
            | DType::Iq4Nl
            | DType::Mxfp4
            | DType::Nvfp4
            | DType::Q2_0
    ) || (dt == DType::Q8_0 && !in_f.is_multiple_of(256))
}

/// The rows the tap wants from an op of `m` rows, or `None` when the tap is off.
pub(crate) fn wanted(m: usize) -> Option<Vec<usize>> {
    TAP.with(|t| {
        t.borrow()
            .as_ref()
            .map(|(rows, _)| rows.iter().copied().filter(|&r| r < m).collect())
    })
}

pub(crate) fn record(op: TappedLinear) {
    TAP.with(|t| {
        if let Some((_, v)) = t.borrow_mut().as_mut() {
            v.push(op);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The simulation at `block: 256, keep: 0, qmax: 127` is the production quantizer, value for
    /// value, on a row whose spikes stand out but not enough to split (four times the rest).
    #[test]
    fn round_trip_matches_the_production_quantizer() {
        let x: Vec<f32> = (0..512)
            .map(|i| {
                let v = ((i * 37 % 101) as f32 - 50.0) / 25.0;
                if i % 200 == 7 {
                    v * 4.0
                } else {
                    v
                }
            })
            .collect();
        let q = crate::quant::quantize_q8(&x);
        assert!(q.hi.is_empty());
        let want: Vec<f32> = (0..x.len())
            .map(|i| q.qs[i] as f32 * q.d[i / 256])
            .collect();
        let sim = ActSim {
            block: 256,
            keep: 0,
            ratio: 0,
            qmax: 127,
            split: false,
            sub: 0,
            emax: 0,
        };
        assert_eq!(sim.round_trip(&x), want);
        // Keeping the outliers changes the result, so the comparison above is not vacuous.
        assert_ne!(ActSim { keep: 2, ..sim }.round_trip(&x), want);
    }

    /// The simulation's split mode at the production constants is the production quantizer,
    /// sparse part and all: on a row with nothing to split, and on rows with one spike in one
    /// block and two in another, where the second spike takes a second part.
    #[test]
    fn split_round_trip_matches_the_production_quantizer() {
        use crate::quant::{dequantize_q8, quantize_q8, HI_KEEP, HI_RATIO};
        let sim = ActSim {
            block: 256,
            keep: HI_KEEP,
            ratio: HI_RATIO as u32,
            qmax: 127,
            split: true,
            sub: 0,
            emax: 0,
        };
        let mut levels_seen = Vec::new();
        for spike in [0.0f32, 50.0, 3.0e3, 2.5e5, 7.0e8] {
            let x: Vec<f32> = (0..768)
                .map(|i| {
                    let v = ((i * 37 % 101) as f32 - 50.0) / 25.0;
                    match i {
                        5 => spike,
                        9 => spike * 0.3137,
                        300 => -spike * 0.75,
                        _ => v,
                    }
                })
                .collect();
            let q = quantize_q8(&x);
            levels_seen.push(q.hi.len());
            assert_eq!(sim.round_trip(&x), dequantize_q8(&q), "spike {spike}");
        }
        // Nothing split, then one part in each of two blocks, then more levels in the block
        // whose second spike the first part leaves a remainder of.
        assert_eq!(levels_seen[0], 0);
        assert_eq!(levels_seen[1], 2);
        assert_eq!(levels_seen[2..], [3, 3, 3]);
    }
}
