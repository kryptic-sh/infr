//! The sparse part of a split activation ([`Q8::hi`]) dotted with a weight row.
//!
//! A part is a one-block activation that is zero except at one or two indices. Its dot is defined
//! as the one-row kernel's value on that block (`vec_dot_q8_any` over 256 inputs, the scalar
//! kernel's bits at every tier), and that is what was computed until this module: a whole
//! 256-element kernel call per part and weight row. Each function here evaluates the same
//! expression the scalar kernel evaluates for one block, reading only the weight elements at the
//! part's indices, so it gives the same bits:
//!
//! - **Integer sums.** Every kernel forms integer sums of `weight · code` (and of codes, for a
//!   min or offset term) per sub-block, which are exact, so summing only the nonzero codes gives
//!   the same integers in any order.
//! - **One float epilogue per block** (Q4_K, Q5_K, Q2_K, Q3_K, Q6_K, IQ4_XS, TQ2_0, TQ1_0): the
//!   kernel's expression is evaluated as written, from the same integers and the same decoded
//!   scales.
//! - **A float term per sub-block** (Q8_0's 32-element blocks, the grid formats' 8-element
//!   groups): the kernel adds the terms in sub-block order to a sum that starts at `+0.0`. A
//!   sub-block with no code has an integer sum of zero, so its term is `scale · 0`, which is
//!   `±0.0` when the scale is finite. Such a sum is never `-0.0` (`+0.0 + -0.0` and an exact
//!   cancellation both round to `+0.0`), and `x + ±0.0 == x` for every other `x`, so the term
//!   changes nothing and is skipped; a sub-block whose scale is not finite (or could overflow) is
//!   added as the kernel adds it, so its `NaN` reaches the result as it would there.
//!
//! The kernel's outer loop over blocks starts its sum at `+0.0` too and runs once here, so each
//! function returns `0.0 + <the block's term>` (which turns a `-0.0` term into `+0.0`, as the
//! kernel does). The reference these are tested against, the one-row kernels on the part made
//! whole ([`HiPart::to_q8`]), is kept as [`vec_dot_hi_reference`].

use super::{
    iq1m_expand_row, iq1s_expand_row, iq2s_expand_row, iq2xs_expand_row, iq2xxs_expand_row,
    iq3s_expand_row, iq3xxs_expand_row, iq4xs_ls_minus_32, q3k_scales, TQ1_POW3,
};
use crate::quant::{HiPart, Q8};
use infr_core::tensor::DType;
use infr_gguf::dequant::{k4, rdf16, KVALUES_IQ4NL};

/// The sparse part of `q8` ([`Q8::hi`]) dotted with a weight row of `dt` over `in_f` inputs:
/// each part's one-block dot against the weight's matching 256-element block, summed from zero in
/// the parts' order. Every route that dots a `Q8` adds this to the dense result
/// (`dense + vec_dot_hi(..)`) for a row whose `hi` is not empty, so the sum is one expression
/// however the dense part was computed; a row with no sparse part adds nothing. Each part's dot is
/// the one-row kernel's value on the part ([`vec_dot_hi_reference`]), computed from the weight
/// elements at its codes alone (the module documentation says why the bits are the same).
pub(crate) fn vec_dot_hi(dt: DType, row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let bpb = row.len() / (in_f / 256);
    q8.hi.iter().fold(0f32, |s, p| {
        s + part_dot(dt, &row[p.b * bpb..p.b * bpb + bpb], p)
    })
}

/// [`vec_dot_hi`] as it was defined: each part made whole and dotted by the one-row kernel of
/// `dt` ([`super::vec_dot_q8_any`], at the tier `CAP` selects), summed the same way. The tests
/// hold the fast path to it bit for bit.
#[cfg(test)]
pub(crate) fn vec_dot_hi_reference<const CAP: bool>(
    dt: DType,
    row: &[u8],
    q8: &Q8,
    in_f: usize,
) -> f32 {
    let bpb = row.len() / (in_f / 256);
    q8.hi.iter().fold(0f32, |s, p| {
        s + super::vec_dot_q8_any::<CAP>(dt, &row[p.b * bpb..p.b * bpb + bpb], &p.to_q8(), 256)
    })
}

/// Weight rows one [`HiBatch`] table chunk covers: the lanes its epilogue runs across, and the
/// packed GEMM's group of weight rows, so a GEMM task adds its own group's parts
/// (`repack::GemmOut::add8` takes the eight lanes).
const HI_LANES: usize = 8;

/// The sparse parts of a batch of activation rows, made ready to dot against many weight rows at
/// once (a prefill `Op::Linear`, an MoE expert bank). Each value is [`vec_dot_hi`]'s for its
/// (weight row, activation row), bit for bit; what changes is the work per value.
///
/// For a format with one float epilogue per block ([`Finish`]) the weight elements the parts
/// need are decoded once per weight row into a table chunk of [`HI_LANES`] weight rows
/// ([`Self::fill`]): the integer factors of each distinct element position (outlier channels
/// repeat across rows, so there are few) and the scales of each distinct block. A part's value
/// is then two integer multiply-adds and the epilogue, evaluated for the chunk's weight rows at
/// once ([`Self::eval`]), each lane the scalar expression [`part_dot`] evaluates. The other
/// formats (Q8_0, the grid formats) take [`vec_dot_hi`] per value. Callers run a chunk right
/// after the dense dot of the same weight rows, while their bytes are in cache.
pub(crate) struct HiBatch<'a> {
    dt: DType,
    in_f: usize,
    q8s: &'a [Q8],
    fin: Option<Finish>,
    /// The activation rows with a sparse part, in row order.
    hot: Vec<usize>,
    /// The distinct element positions (`b·256 + i`) the parts' codes sit at.
    elems: Vec<usize>,
    /// The distinct blocks the parts add to.
    blocks: Vec<usize>,
    /// Each hot row's parts, in [`Q8::hi`]'s order.
    parts: Vec<Vec<TablePart>>,
}

/// A part as [`HiBatch::eval`] reads it: its block's slot, its scale, and its codes as
/// `(element slot, code)`, a missing second code as `(first slot, 0)`.
struct TablePart {
    block: usize,
    dh: f32,
    codes: [(usize, i32); 2],
}

impl<'a> HiBatch<'a> {
    /// The sparse parts of `q8s` against weights of `dt` over `in_f` inputs, or `None` when no
    /// row has one.
    pub(crate) fn new(dt: DType, in_f: usize, q8s: &'a [Q8]) -> Option<Self> {
        let hot: Vec<usize> = (0..q8s.len()).filter(|&r| !q8s[r].hi.is_empty()).collect();
        if hot.is_empty() {
            return None;
        }
        let fin = Finish::of(dt);
        let all = || hot.iter().flat_map(|&r| &q8s[r].hi);
        let mut elems: Vec<usize> = Vec::new();
        let mut blocks: Vec<usize> = Vec::new();
        if fin.is_some() {
            elems = all()
                .flat_map(|p| p.codes().iter().map(|c| p.b * 256 + c.0 as usize))
                .collect();
            elems.sort_unstable();
            elems.dedup();
            blocks = all().map(|p| p.b).collect();
            blocks.sort_unstable();
            blocks.dedup();
        }
        let slot = |v: &[usize], x: usize| v.binary_search(&x).expect("collected above");
        let parts = hot
            .iter()
            .map(|&r| {
                if fin.is_none() {
                    return Vec::new();
                }
                q8s[r]
                    .hi
                    .iter()
                    .map(|p| {
                        let c = p.codes();
                        let at = |k: usize| (slot(&elems, p.b * 256 + c[k].0 as usize), c[k].1);
                        let first = at(0);
                        let second = if c.len() > 1 { at(1) } else { (first.0, 0) };
                        TablePart {
                            block: slot(&blocks, p.b),
                            dh: p.d,
                            codes: [(first.0, first.1 as i32), (second.0, second.1 as i32)],
                        }
                    })
                    .collect()
            })
            .collect();
        Some(Self {
            dt,
            in_f,
            q8s,
            fin,
            hot,
            elems,
            blocks,
            parts,
        })
    }

    /// The `i32`s one table chunk ([`HI_LANES`] weight rows) takes: per element two factor lanes,
    /// per block two scale lanes (f32 bits). Zero for the formats that take [`vec_dot_hi`].
    pub(crate) fn chunk_len(&self) -> usize {
        2 * HI_LANES * (self.elems.len() + self.blocks.len())
    }

    /// Fill the table chunk `tab` ([`Self::chunk_len`] long) for the `n` (at most [`HI_LANES`])
    /// weight rows from `o0` of `wbytes` (`bpr` bytes each): lane `l` holds weight row `o0 + l`.
    fn fill(&self, wbytes: &[u8], bpr: usize, o0: usize, n: usize, tab: &mut [i32]) {
        if self.fin.is_none() {
            return;
        }
        let bpb = bpr / (self.in_f / 256);
        let e2 = 2 * self.elems.len();
        for l in 0..n {
            let row = &wbytes[(o0 + l) * bpr..(o0 + l + 1) * bpr];
            for (k, &b) in self.blocks.iter().enumerate() {
                let (d, dmin) = block_scales(self.dt, &row[b * bpb..b * bpb + bpb]);
                tab[(e2 + 2 * k) * HI_LANES + l] = d.to_bits() as i32;
                tab[(e2 + 2 * k + 1) * HI_LANES + l] = dmin.to_bits() as i32;
            }
            for (k, &pos) in self.elems.iter().enumerate() {
                let b = pos / 256;
                let (f1, f2) = elem(self.dt, &row[b * bpb..b * bpb + bpb], pos % 256);
                tab[2 * k * HI_LANES + l] = f1;
                tab[(2 * k + 1) * HI_LANES + l] = f2;
            }
        }
    }

    /// `acc[l]` becomes [`vec_dot_hi`] of hot row `k` (an index into [`Self::hot`]) against
    /// weight row `o0 + l`, for `l < n`, from the chunk `tab` [`Self::fill`] filled for those rows.
    #[allow(clippy::too_many_arguments)]
    fn eval(
        &self,
        k: usize,
        wbytes: &[u8],
        bpr: usize,
        o0: usize,
        n: usize,
        tab: &[i32],
        acc: &mut [f32; HI_LANES],
    ) {
        let Some(fin) = self.fin else {
            let q8 = &self.q8s[self.hot[k]];
            for (l, a) in acc.iter_mut().enumerate().take(n) {
                let row = &wbytes[(o0 + l) * bpr..(o0 + l + 1) * bpr];
                *a = vec_dot_hi(self.dt, row, q8, self.in_f);
            }
            return;
        };
        let lanes = |slot: usize| -> &[i32; HI_LANES] {
            tab[slot * HI_LANES..(slot + 1) * HI_LANES]
                .try_into()
                .expect("a whole chunk")
        };
        let e2 = 2 * self.elems.len();
        // Every lane is computed, so the loop below runs whole vectors; lanes `n..` read whatever
        // an earlier fill left in the chunk (or zeros), and their values are never used.
        acc.fill(0.0);
        for p in &self.parts[k] {
            let [(u1, c1), (u2, c2)] = p.codes;
            let (f1a, f2a, f1b, f2b) = (
                lanes(2 * u1),
                lanes(2 * u1 + 1),
                lanes(2 * u2),
                lanes(2 * u2 + 1),
            );
            let (d, dmin) = (lanes(e2 + 2 * p.block), lanes(e2 + 2 * p.block + 1));
            for l in 0..HI_LANES {
                let s1 = f1a[l] * c1 + f1b[l] * c2;
                let s2 = f2a[l] * c1 + f2b[l] * c2;
                let v = fin.value(
                    p.dh,
                    f32::from_bits(d[l] as u32),
                    f32::from_bits(dmin[l] as u32),
                    s1,
                    s2,
                );
                acc[l] += v;
            }
        }
    }

    /// For the `n` (at most [`HI_LANES`]) weight rows from `o0`: fill the table chunk `tab`
    /// ([`Self::chunk_len`] long) and call `add(r, acc)` for each hot row `r` in row order, where
    /// `acc[l]` is [`vec_dot_hi`] of row `r` against weight row `o0 + l` (`l < n`).
    pub(crate) fn for_group(
        &self,
        wbytes: &[u8],
        bpr: usize,
        o0: usize,
        n: usize,
        tab: &mut [i32],
        mut add: impl FnMut(usize, &[f32; HI_LANES]),
    ) {
        self.fill(wbytes, bpr, o0, n, tab);
        let mut acc = [0f32; HI_LANES];
        for (k, &r) in self.hot.iter().enumerate() {
            self.eval(k, wbytes, bpr, o0, n, tab, &mut acc);
            add(r, &acc);
        }
    }

    /// Add every hot row's sparse dot into the o-major block `out` (`out[(o - o0) * m + r]`) of
    /// the `rows` weight rows from `o0`: each value becomes `dense + vec_dot_hi(..)`.
    pub(crate) fn add_o_major(
        &self,
        wbytes: &[u8],
        bpr: usize,
        o0: usize,
        rows: usize,
        out: &mut [f32],
        m: usize,
    ) {
        let mut tab = vec![0i32; self.chunk_len()];
        for c0 in (0..rows).step_by(HI_LANES) {
            let n = HI_LANES.min(rows - c0);
            self.for_group(wbytes, bpr, o0 + c0, n, &mut tab, |r, acc| {
                for (l, &a) in acc.iter().enumerate().take(n) {
                    out[(c0 + l) * m + r] += a;
                }
            });
        }
    }
}

/// One part's dot with the weight block `blk` (the 256 elements of block `p.b`): the scalar
/// one-row kernel's value for a one-block activation holding `p`'s codes.
fn part_dot(dt: DType, blk: &[u8], p: &HiPart) -> f32 {
    let (dh, c) = (p.d, p.codes());
    if let Some(fin) = Finish::of(dt) {
        let (d, dmin) = block_scales(dt, blk);
        let (mut s1, mut s2) = (0i32, 0i32);
        for &(i, q) in c {
            let (f1, f2) = elem(dt, blk, i as usize);
            s1 += f1 * q as i32;
            s2 += f2 * q as i32;
        }
        return fin.value(dh, d, dmin, s1, s2);
    }
    match dt {
        DType::Q8_0 => q8_0(blk, dh, c),
        DType::Iq2S => grid(dh, c, iq2s_expand_row(blk, 256), None),
        DType::Iq2Xs => grid(dh, c, iq2xs_expand_row(blk, 256), None),
        DType::Iq2Xxs => grid(dh, c, iq2xxs_expand_row(blk, 256), None),
        DType::Iq3S => grid(dh, c, iq3s_expand_row(blk, 256), None),
        DType::Iq3Xxs => grid(dh, c, iq3xxs_expand_row(blk, 256), None),
        DType::Iq1S => {
            let (w, dls, deltas) = iq1s_expand_row(blk, 256);
            grid(dh, c, (w, dls), Some(&deltas))
        }
        DType::Iq1M => {
            let (w, dls, deltas) = iq1m_expand_row(blk, 256);
            grid(dh, c, (w, dls), Some(&deltas))
        }
        other => unreachable!("{other:?} does not take Q8 activations"),
    }
}

/// How the one-block kernel of a format with one float epilogue per block finishes, from the
/// block's decoded scales `d` and `dmin` and the two integer sums `s1 = Σ f1(i)·code` and
/// `s2 = Σ f2(i)·code` over the codes ([`elem`] gives `f1`, `f2`).
#[derive(Clone, Copy)]
enum Finish {
    /// `0 + dh·(d·s1 − dmin·s2)`: Q4_K, Q5_K, Q2_K (`sumf += q8.d[b] * (d * sd as f32 - dmin *
    /// sm as f32)` with `sd = s1`, `sm = s2`).
    Min,
    /// `0 + d·dh·s1`: Q6_K, Q3_K, IQ4_XS, TQ2_0, TQ1_0 (`sumf += d * q8.d[b] * isum as f32`).
    One,
}

impl Finish {
    /// The epilogue of `dt`'s one-row kernel, for the formats that have one per block.
    fn of(dt: DType) -> Option<Self> {
        match dt {
            DType::Q4K | DType::Q5K | DType::Q2K => Some(Self::Min),
            DType::Q6K | DType::Q3K | DType::Iq4Xs | DType::Tq2_0 | DType::Tq1_0 => Some(Self::One),
            _ => None,
        }
    }

    /// The kernel's value for one block: `0.0 +` its term, as the kernel's sum from `+0.0`.
    #[inline(always)]
    fn value(self, dh: f32, d: f32, dmin: f32, s1: i32, s2: i32) -> f32 {
        match self {
            Self::Min => 0f32 + dh * (d * s1 as f32 - dmin * s2 as f32),
            Self::One => 0f32 + d * dh * s1 as f32,
        }
    }
}

/// The block scales [`Finish::value`] takes, `(d, dmin)`, decoded from the f16s the scalar
/// kernel reads (`dmin` is 0 for [`Finish::One`] formats, which have none).
fn block_scales(dt: DType, blk: &[u8]) -> (f32, f32) {
    match dt {
        DType::Q4K | DType::Q5K => (rdf16(&blk[0..2]), rdf16(&blk[2..4])),
        DType::Q2K => (rdf16(&blk[80..82]), rdf16(&blk[82..84])),
        DType::Q6K => (rdf16(&blk[208..210]), 0.0),
        DType::Q3K => (rdf16(&blk[108..110]), 0.0),
        DType::Iq4Xs => (rdf16(&blk[0..2]), 0.0),
        DType::Tq2_0 => (rdf16(&blk[64..66]), 0.0),
        DType::Tq1_0 => (rdf16(&blk[52..54]), 0.0),
        other => unreachable!("{other:?} has no one-epilogue block"),
    }
}

/// Element `i` of the weight block `blk` as the integer factors `(f1, f2)` of the scalar kernel's
/// sums (`s1 += f1·code`, `s2 += f2·code`):
///
/// - Q4_K / Q5_K: `(sc_s·q, m_s)`, `s = i / 32`, `q` the 4-bit code (Q5_K: plus 16 for its
///   `qh` bit `s`).
/// - Q2_K: `(sc_is & 15)·q2, sc_is >> 4)`, `is = i / 16`, in the kernel's traversal (sub-block
///   `is = n·8 + j·2 + half` reads `qs[n·32 + half·16..]` shifted right by `2j`).
/// - Q6_K: `(sc_sub·(q6 − 32), 0)`, `sub = i / 16`.
/// - Q3_K: `(sc_is·(q3 − 4), 0)`, `q3` the 2-bit field plus 4 for a set `hmask` bit of plane
///   `n·4 + j` (Q2_K's traversal).
/// - IQ4_XS: `((ls_ib − 32)·kvalue, 0)`, `ib = i / 32`.
/// - TQ2_0 / TQ1_0: `(digit − 1, 0)`, decoded as `tq2_0_expand_row` / `tq1_0_expand_row` do.
fn elem(dt: DType, blk: &[u8], i: usize) -> (i32, i32) {
    match dt {
        DType::Q4K => {
            let s = i / 32;
            let (sc, m) = k4(s, &blk[4..16]);
            let byte = blk[16 + s / 2 * 32 + i % 32];
            let q4 = if s % 2 == 1 { byte >> 4 } else { byte & 0xF } as i32;
            (sc as i32 * q4, m as i32)
        }
        DType::Q5K => {
            let (s, l) = (i / 32, i % 32);
            let (sc, m) = k4(s, &blk[4..16]);
            let byte = blk[48 + s / 2 * 32 + l];
            let lo = if s % 2 == 1 { byte >> 4 } else { byte & 0xF } as i32;
            let q5 = lo + if blk[16 + l] & (1 << s) != 0 { 16 } else { 0 };
            (sc as i32 * q5, m as i32)
        }
        DType::Q2K => {
            let is = i / 16;
            let (n, j, half) = (is / 8, is % 8 / 2, is % 2);
            let q2 = ((blk[16 + n * 32 + half * 16 + i % 16] >> (2 * j)) & 3) as i32;
            let sc = blk[is];
            ((sc & 0xF) as i32 * q2, (sc >> 4) as i32)
        }
        DType::Q6K => {
            // Element `i` is column `col` (0..4) of half `i / 128`, at `l` within the column.
            let (half, col, l) = (i / 128, i % 128 / 32, i % 32);
            let lb = blk[half * 64 + l + 32 * (col % 2)];
            let lo = if col < 2 { lb & 0xF } else { lb >> 4 };
            let hb = (blk[128 + half * 32 + l] >> (2 * col)) & 3;
            let q6 = (lo | (hb << 4)) as i32;
            (blk[192 + i / 16] as i8 as i32 * (q6 - 32), 0)
        }
        DType::Q3K => {
            let (is, l) = (i / 16, i % 16);
            let (n, j, half) = (is / 8, is % 8 / 2, is % 2);
            let low2 = ((blk[32 + n * 32 + half * 16 + l] >> (2 * j)) & 3) as i32;
            let high = if blk[half * 16 + l] & (1 << (n * 4 + j)) != 0 {
                4
            } else {
                0
            };
            (
                q3k_scales(&blk[96..108])[is] as i32 * ((low2 | high) - 4),
                0,
            )
        }
        DType::Iq4Xs => {
            let (ib, j) = (i / 32, i % 32);
            // Low nibbles hold elements 0..16 of the sub-block, high nibbles 16..32.
            let byte = blk[8 + ib * 16 + j % 16];
            let nib = if j < 16 { byte & 0xF } else { byte >> 4 };
            let w = KVALUES_IQ4NL[nib as usize] as i32;
            let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
            (iq4xs_ls_minus_32(scales_h, &blk[4..8], ib) * w, 0)
        }
        DType::Tq2_0 => {
            // Element `chunk·128 + l·32 + m` is bits `2l` of `qs[chunk·32 + m]`.
            let q = (blk[i / 128 * 32 + i % 32] >> (2 * (i % 128 / 32))) & 3;
            (q as i32 - 1, 0)
        }
        DType::Tq1_0 => {
            // Three segments: `qs[0..32]` (elements `n·32 + m`), `qs[32..48]` (`160 + n·16 + m`),
            // `qh[0..4]` (`240 + n·4 + j`), digit `n` of each byte.
            let (byte, n) = match i {
                0..160 => (blk[i % 32], i / 32),
                160..240 => (blk[32 + (i - 160) % 16], (i - 160) / 16),
                _ => (blk[48 + (i - 240) % 4], (i - 240) / 4),
            };
            let digit = ((byte.wrapping_mul(TQ1_POW3[n]) as u16 * 3) >> 8) as i32;
            (digit - 1, 0)
        }
        other => unreachable!("{other:?} has no one-epilogue block"),
    }
}

/// The largest finite f16 value: a finite f16 scale times a scale `dh` with `dh · F16_MAX` finite
/// is finite.
const F16_MAX: f32 = 65504.0;

/// Whether the f16 at `b[0..2]` is finite (its exponent is not all ones).
fn f16_finite(b: &[u8]) -> bool {
    u16::from_le_bytes([b[0], b[1]]) & 0x7C00 != 0x7C00
}

/// Q8_0 over a 256-element span (`vec_dot_q8_0_scalar`): eight 32-element blocks, each adding
/// `dh·d_s·iprod_s` to a sum from `+0.0` in block order. A block with no code is skipped when its
/// term is `±0.0` (module documentation): its `d_s` finite and `dh·F16_MAX` finite.
fn q8_0(blk: &[u8], dh: f32, c: &[(u8, i8)]) -> f32 {
    let skippable = (dh * F16_MAX).is_finite();
    let mut sumf = 0f32;
    let mut k = 0;
    for s in 0..8usize {
        let wb = &blk[s * 34..s * 34 + 34];
        let mut iprod = 0i32;
        let mut any = false;
        while k < c.len() && c[k].0 as usize / 32 == s {
            iprod += wb[2 + c[k].0 as usize % 32] as i8 as i32 * c[k].1 as i32;
            any = true;
            k += 1;
        }
        if !any && skippable && f16_finite(&wb[0..2]) {
            continue;
        }
        sumf += dh * rdf16(&wb[0..2]) * iprod as f32;
    }
    sumf
}

/// The grid formats over one expanded block (`weights` and per-8 scales `dls`, and IQ1_S/IQ1_M's
/// per-8 `deltas`): `0 + dh·block_sum`, `block_sum` adding `dls_g·iprod_g` (IQ1:
/// `dls_g·(iprod_g + delta_g·asum_g)`) over the 32 groups of 8 in order from `+0.0`. A group with
/// no code is skipped when its `dls_g` is finite (module documentation).
fn grid(dh: f32, c: &[(u8, i8)], (w, dls): (Vec<i8>, Vec<f32>), deltas: Option<&[f32]>) -> f32 {
    let mut block_sum = 0f32;
    let mut k = 0;
    for g in 0..32usize {
        let (mut iprod, mut asum, mut any) = (0i32, 0i32, false);
        while k < c.len() && c[k].0 as usize / 8 == g {
            iprod += w[c[k].0 as usize] as i32 * c[k].1 as i32;
            asum += c[k].1 as i32;
            any = true;
            k += 1;
        }
        if !any && dls[g].is_finite() {
            continue;
        }
        block_sum += match deltas {
            Some(dt) => dls[g] * (iprod as f32 + dt[g] * asum as f32),
            None => dls[g] * iprod as f32,
        };
    }
    0f32 + dh * block_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::quantize_q8;
    use crate::test_util::{synth_weights, Q8_DTYPES};

    /// Microbenchmark: ns per weight row of a [`HiBatch`] table chunk's fill and of one hot row's
    /// evaluation from it, single-threaded, Q6_K weights 6144 wide, one activation row with two
    /// split blocks of one code each. Prints three repeats; asserts only that it timed.
    ///
    /// `cargo test --release -p infr-cpu --lib kernels::hi::tests::batch_microbench --
    /// --ignored --nocapture`
    #[test]
    #[ignore = "measurement: run in release with --nocapture"]
    fn batch_microbench() {
        let (in_f, out_f) = (6144usize, 2048usize);
        let dt = DType::Q6K;
        let w = synth_weights(dt, out_f, in_f, 5, &[(1.0, 1.0), (-0.5, 2.0)]);
        let bpr = w.len() / out_f;
        let mut x: Vec<f32> = (0..in_f)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 70.0)
            .collect();
        x[700] = 90.0;
        x[3000] = -90.0;
        let q8s = vec![quantize_q8(&x)];
        let hb = HiBatch::new(dt, in_f, &q8s).unwrap();
        let tl = hb.chunk_len();
        let nch = out_f / HI_LANES;
        let mut tab = vec![0i32; tl * nch];
        for _ in 0..3 {
            let t = std::time::Instant::now();
            for c in 0..nch {
                hb.fill(
                    &w,
                    bpr,
                    c * HI_LANES,
                    HI_LANES,
                    &mut tab[c * tl..(c + 1) * tl],
                );
            }
            let tf = t.elapsed().as_nanos() as f64 / out_f as f64;
            let t = std::time::Instant::now();
            let mut acc = [0f32; HI_LANES];
            let mut s = 0f32;
            for c in 0..nch {
                hb.eval(
                    0,
                    &w,
                    bpr,
                    c * HI_LANES,
                    HI_LANES,
                    &tab[c * tl..(c + 1) * tl],
                    &mut acc,
                );
                s += acc[3];
            }
            std::hint::black_box(s);
            let te = t.elapsed().as_nanos() as f64 / out_f as f64;
            println!(
                "E={} B={} fill {tf:.1} ns/row eval {te:.1} ns/row",
                hb.elems.len(),
                hb.blocks.len()
            );
            assert!(tf > 0.0 && te > 0.0);
        }
    }

    /// Microbenchmark: ns per weight row for the sparse part of one activation row, the fast dot
    /// against the one-row kernels it replaced (`vec_dot_hi_reference`, uncapped), per format, on
    /// a 2048-wide row with one split block holding one code and on one with nine codes in five
    /// split blocks. Prints; asserts only that both agree and that it timed.
    ///
    /// `cargo test --release -p infr-cpu --lib kernels::hi::tests::microbench -- --ignored
    /// --nocapture`
    #[test]
    #[ignore = "measurement: run in release with --nocapture"]
    fn microbench() {
        let (in_f, rows) = (2048usize, 256usize);
        let base = |i: usize| ((i * 37 % 101) as f32 - 50.0) / 70.0;
        let mut one: Vec<f32> = (0..in_f).map(base).collect();
        one[700] = 90.0;
        let mut four = one.clone();
        for (i, v) in [
            (13, 80.0),
            (14, -60.0),
            (300, 70.0),
            (301, 75.0),
            (1100, -50.0),
            (1101, 40.0),
            (2000, 33.0),
            (2047, -45.0),
        ] {
            four[i] = v;
        }
        for dt in Q8_DTYPES {
            let w = synth_weights(dt, rows, in_f, 5, &[(1.0, 1.0), (-0.5, 2.0)]);
            let bpr = w.len() / rows;
            for (name, x) in [("one code", &one), ("nine codes", &four)] {
                let q8 = quantize_q8(x);
                let time = |f: &dyn Fn(&[u8]) -> f32| {
                    let reps = 200usize;
                    let mut acc = 0f32;
                    let t = std::time::Instant::now();
                    for _ in 0..reps {
                        for o in 0..rows {
                            acc += f(std::hint::black_box(&w[o * bpr..(o + 1) * bpr]));
                        }
                    }
                    std::hint::black_box(acc);
                    t.elapsed().as_nanos() as f64 / (reps * rows) as f64
                };
                let fast = time(&|r| vec_dot_hi(dt, r, &q8, in_f));
                let old = time(&|r| vec_dot_hi_reference::<false>(dt, r, &q8, in_f));
                for o in 0..rows {
                    let r = &w[o * bpr..(o + 1) * bpr];
                    assert_eq!(
                        vec_dot_hi(dt, r, &q8, in_f).to_bits(),
                        vec_dot_hi_reference::<false>(dt, r, &q8, in_f).to_bits()
                    );
                }
                println!(
                    "{:<8} {name:<11} ({} parts): fast {fast:>7.1} ns, one-row kernels {old:>7.1} \
                     ns per weight row",
                    format!("{dt:?}"),
                    q8.hi.len()
                );
                assert!(fast > 0.0 && old > 0.0);
            }
        }
    }
}
