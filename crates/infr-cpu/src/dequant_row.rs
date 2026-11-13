//! f32 dot of a quantized weight row against f32 activations, bit-identical to
//! `dot(&bytes_to_f32(row, dt), x)` but without materializing the dequantized row.
//!
//! The f32-activation expert GEMM (`ExpertActs::Raw`, which every single-row MoE call takes)
//! used to dequantize each weight row through `bytes_to_f32`: a decode into the factored form
//! (three allocations) and a second expansion pass into a fourth, per row, per routed expert, per
//! token — the dominant cost of a CPU MoE decode. Every affine quant (the formats with a
//! factored form) now decodes here a span at a time on the stack to the same bits (see
//! [`decode_span`] for which formats share `bytes_to_f32`'s own decode and which have
//! hand-written copies), and the dot keeps [`dot`]'s lane order.
use crate::bytes_to_f32;
use crate::kernels::dot;
use infr_core::tensor::DType;
use infr_gguf::dequant::{expand_factored, factor_block, has_factored_form, k4, rdf16};

/// Elements decoded per step: one K-quant super-block, or eight 32-element blocks.
const SPAN: usize = 256;

/// `col[r] = dot(dequant(row), x_r)` for every activation row `x_r` of `xin` (`[col.len(),
/// in_f]` row-major) — the same bits as `dot(&bytes_to_f32(row, dt), x_r)`, without materializing
/// the dequantized row. An affine quant decodes one [`SPAN`] at a time into a stack buffer and
/// feeds it straight into each row's accumulators; any other dtype (the codebook quants, the fp4
/// formats) takes exactly that reference expression. `accs` is caller-owned scratch, reused
/// across calls so a range of output rows allocates it once.
///
/// Why it is the same sum: [`dot`] keeps 8 lanes, lane `i % 8` accumulating element `i` by FMA in
/// increasing `i`, then folds the lanes with `iter().sum()` and finishes an `n % 8` tail serially.
/// Every span here is a whole number of quant blocks (each a multiple of 16 elements), so walking
/// the spans in order hands each lane the same FMAs in the same order, the lanes are folded by the
/// same expression, and there is no tail.
pub(crate) fn dequant_dot_rows(
    row: &[u8],
    dt: DType,
    xin: &[f32],
    in_f: usize,
    col: &mut [f32],
    accs: &mut Vec<[f32; 8]>,
) {
    if col.is_empty() {
        return;
    }
    if !has_factored_form(dt) {
        let wf = bytes_to_f32(row, dt);
        for (r, dst) in col.iter_mut().enumerate() {
            *dst = dot(&wf, &xin[r * in_f..r * in_f + in_f]);
        }
        return;
    }
    let (qpb, bpb) = infr_core::decode_spec::block_layout(dt);
    assert!(
        row.len().is_multiple_of(bpb) && row.len() / bpb * qpb == in_f,
        "dequant_dot_rows: {} bytes of {dt:?} do not hold a row of {in_f} elements",
        row.len(),
    );
    assert!(
        xin.len() >= col.len() * in_f,
        "dequant_dot_rows: {} activations for {} rows of {in_f}",
        xin.len(),
        col.len()
    );
    accs.clear();
    accs.resize(col.len(), [0f32; 8]);
    let mut w = [0f32; SPAN];
    for (si, span) in row.chunks(SPAN / qpb * bpb).enumerate() {
        let n = span.len() / bpb * qpb;
        decode_span(span, dt, &mut w[..n]);
        let e0 = si * SPAN;
        for (r, acc) in accs.iter_mut().enumerate() {
            let x = &xin[r * in_f + e0..r * in_f + e0 + n];
            // `n` is a multiple of 8 (see the doc), so the 8-chunks cover the span exactly. Fixed
            // `[f32; 8]` chunks index without bounds checks, which keeps this one FMA vector.
            let (wc, xc) = (w[..n].as_chunks::<8>().0, x.as_chunks::<8>().0);
            // On a local copy: `acc` lives in caller-owned heap scratch, and accumulating through
            // the reference left the lanes in memory, a measurably slower decode.
            let mut a = *acc;
            for (wv, xv) in wc.iter().zip(xc) {
                for j in 0..8 {
                    a[j] = wv[j].mul_add(xv[j], a[j]);
                }
            }
            *acc = a;
        }
    }
    for (dst, acc) in col.iter_mut().zip(accs.iter()) {
        *dst = acc.iter().sum();
    }
}

/// Dequantize whole blocks of the affine `dt` (`span` holds `out.len()` elements' worth) into
/// `out`.
///
/// Q4_K, Q5_K, Q6_K and Q8_0 — the formats MoE expert banks actually ship in — decode through the
/// hand-written per-format functions below, which write f32 straight from the block bytes. The
/// other affine formats go through `infr_gguf::dequant::{factor_block, expand_factored}`, the
/// decode `bytes_to_f32` itself uses, so they match by construction. The four hand-written
/// decoders DUPLICATE `factor_block`'s bit layouts: routing them through it too was measured
/// at about half the decode speed on ERNIE-4.5-21B-A3B even with the
/// shared decode inlined and monomorphised per format (see `docs/backlog.md`). The tests hold
/// each against `bytes_to_f32` bit for bit.
fn decode_span(span: &[u8], dt: DType, out: &mut [f32]) {
    macro_rules! hand {
        ($f:ident, $d:ident) => {{
            let (qpb, bpb) = const { infr_core::decode_spec::block_layout(DType::$d) };
            for (blk, o) in span.chunks_exact(bpb).zip(out.chunks_exact_mut(qpb)) {
                $f(blk, o);
            }
        }};
    }
    // The geometry as a compile-time constant: `block_layout` is not inlined across the crate
    // boundary, and a runtime block size leaves every code store bounds-checked.
    macro_rules! arm {
        ($d:ident) => {
            decode_span_as(
                DType::$d,
                const { infr_core::decode_spec::block_layout(DType::$d) },
                span,
                out,
            )
        };
    }
    match dt {
        DType::Q4_0 => arm!(Q4_0),
        DType::Q4_1 => arm!(Q4_1),
        DType::Q5_0 => arm!(Q5_0),
        DType::Q5_1 => arm!(Q5_1),
        DType::Q8_0 => hand!(q8_0_block, Q8_0),
        DType::Q2K => arm!(Q2K),
        DType::Q3K => arm!(Q3K),
        DType::Q4K => hand!(q4k_block, Q4K),
        DType::Q5K => hand!(q5k_block, Q5K),
        DType::Q6K => hand!(q6k_block, Q6K),
        other => unreachable!("decode_span: {other:?} has no factored form"),
    }
}

#[inline(always)]
fn decode_span_as(dt: DType, (qpb, bpb): (usize, usize), span: &[u8], out: &mut [f32]) {
    let mut codes = [0u8; SPAN];
    let mut scm = [0i16; SPAN / 16 * 2];
    let (codes, scm) = (&mut codes[..qpb], &mut scm[..qpb / 16 * 2]);
    for (blk, o) in span.chunks_exact(bpb).zip(out.chunks_exact_mut(qpb)) {
        let [d, dmin] = factor_block(dt, blk, codes, scm);
        expand_factored(d.to_f32(), dmin.to_f32(), codes, scm, o);
    }
}

/// `out[i] = sc · codes[i] + mn` over one run sharing a `(sc, mn)` pair — the expansion
/// `dequant_block` applies per 16-element block (a plain multiply then add: no FMA contraction).
#[inline(always)]
fn expand(sc: f32, mn: f32, codes: impl Iterator<Item = u8>, out: &mut [f32]) {
    for (o, c) in out.iter_mut().zip(codes) {
        *o = sc * c as f32 + mn;
    }
}

/// Q4_K (144 bytes, 256 elements): 8 sub-blocks of 32, 6-bit `(sc, m)` each from `k4`; the
/// factored form stores `(sc, -m)` against `(d, dmin)`.
fn q4k_block(blk: &[u8], out: &mut [f32]) {
    let d = rdf16(&blk[0..2]);
    let dmin = rdf16(&blk[2..4]);
    let scales = &blk[4..16];
    let qs = &blk[16..144];
    for j in 0..4 {
        let q = &qs[j * 32..j * 32 + 32];
        for (s, hi) in [(2 * j, false), (2 * j + 1, true)] {
            let (sc, m) = k4(s, scales);
            let scf = d * (sc as i16) as f32;
            let mnf = dmin * (-(m as i16)) as f32;
            let o = &mut out[s * 32..s * 32 + 32];
            if hi {
                expand(scf, mnf, q.iter().map(|&v| v >> 4), o);
            } else {
                expand(scf, mnf, q.iter().map(|&v| v & 0xF), o);
            }
        }
    }
}

/// Q5_K (176 bytes, 256 elements): Q4_K's scales with a fifth code bit per element from `qh`.
fn q5k_block(blk: &[u8], out: &mut [f32]) {
    let d = rdf16(&blk[0..2]);
    let dmin = rdf16(&blk[2..4]);
    let scales = &blk[4..16];
    let qh = &blk[16..48];
    let qs = &blk[48..176];
    for j in 0..4 {
        let q = &qs[j * 32..j * 32 + 32];
        let (u1, u2) = (1u8 << (2 * j), 2u8 << (2 * j));
        for (s, hi) in [(2 * j, false), (2 * j + 1, true)] {
            let (sc, m) = k4(s, scales);
            let scf = d * (sc as i16) as f32;
            let mnf = dmin * (-(m as i16)) as f32;
            let o = &mut out[s * 32..s * 32 + 32];
            if hi {
                let codes = q
                    .iter()
                    .zip(qh)
                    .map(|(&v, &h)| (v >> 4) + if h & u2 != 0 { 16 } else { 0 });
                expand(scf, mnf, codes, o);
            } else {
                let codes = q
                    .iter()
                    .zip(qh)
                    .map(|(&v, &h)| (v & 0xF) + if h & u1 != 0 { 16 } else { 0 });
                expand(scf, mnf, codes, o);
            }
        }
    }
}

/// Q6_K (210 bytes, 256 elements): an i8 scale `s` per 16 elements, factored as `(s, -32·s)`
/// against `(d, d)`; 6-bit codes split over `ql` (low nibble) and `qh` (top two bits).
fn q6k_block(blk: &[u8], out: &mut [f32]) {
    let ql = &blk[0..128];
    let qh = &blk[128..192];
    let scales = &blk[192..208];
    let d = rdf16(&blk[208..210]);
    for half in 0..2 {
        let (ql, qh) = (
            &ql[half * 64..half * 64 + 64],
            &qh[half * 32..half * 32 + 32],
        );
        // The four 32-element quarters of this half: (ql offset, ql nibble shift, qh shift).
        for (k, (lo, nib, hs)) in [(0, 0, 0), (32, 0, 2), (0, 4, 4), (32, 4, 6)]
            .into_iter()
            .enumerate()
        {
            let base = half * 128 + k * 32;
            for g in 0..2 {
                let t = base / 16 + g; // the 16-element block, and its scale's index
                let s = scales[t] as i8 as i16;
                let scf = d * s as f32;
                let mnf = d * (-32 * s) as f32;
                let r = g * 16..g * 16 + 16;
                let codes = ql[lo + r.start..lo + r.end]
                    .iter()
                    .zip(&qh[r.clone()])
                    .map(|(&l, &h)| ((l >> nib) & 0xF) | (((h >> hs) & 3) << 4));
                expand(scf, mnf, codes, &mut out[base + r.start..base + r.end]);
            }
        }
    }
}

/// Q8_0 (34 bytes, 32 elements): the factored form biases each i8 by +128 and folds the bias
/// back as `min = d·(-128)`, so the value is `d·(q + 128) + d·(-128)`. Kept rather than
/// simplified to `d·q`: for a finite `d` the two are equal, but they differ in the sign of zero
/// at `q == 0` and for an infinite `d`, and the reference computes this form.
fn q8_0_block(blk: &[u8], out: &mut [f32]) {
    let d = rdf16(&blk[0..2]);
    let scf = d * 1.0;
    let mnf = d * (-128i16) as f32;
    let codes = blk[2..34].iter().map(|&q| (q as i8 as i16 + 128) as u8);
    expand(scf, mnf, codes, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every format `dequant_dot_rows` decodes itself — `has_factored_form`'s set, spelled out so
    /// a format leaving or joining it shows up here.
    const AFFINE: [DType; 10] = [
        DType::Q4_0,
        DType::Q4_1,
        DType::Q5_0,
        DType::Q5_1,
        DType::Q8_0,
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
    ];

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed
    }

    /// Byte offsets of each block's f16 scale fields.
    fn f16_fields(dt: DType) -> &'static [usize] {
        match dt {
            DType::Q4_0 | DType::Q5_0 | DType::Q8_0 => &[0],
            DType::Q4_1 | DType::Q5_1 | DType::Q4K | DType::Q5K => &[0, 2],
            DType::Q2K => &[80, 82],
            DType::Q3K => &[108],
            DType::Q6K => &[208],
            other => unreachable!("{other:?} is not an affine quant"),
        }
    }

    /// Random bytes for `nblk` blocks of `dt` with every f16 scale field set to a finite,
    /// realistically sized value — or, with `specials`, one block in four given an awkward one
    /// (zero, negative zero, an f16 subnormal, the f16 max, inf, NaN) instead.
    fn blocks(dt: DType, nblk: usize, seed: u64, specials: bool) -> Vec<u8> {
        let (_, bpb) = infr_core::decode_spec::block_layout(dt);
        let mut s = seed;
        let mut b: Vec<u8> = (0..nblk * bpb).map(|_| (lcg(&mut s) >> 33) as u8).collect();
        let odd = [0.0f32, -0.0, 1.5e-6, 65504.0, f32::INFINITY, f32::NAN];
        for blk in b.chunks_exact_mut(bpb) {
            for &off in f16_fields(dt) {
                let v = if specials && lcg(&mut s).is_multiple_of(4) {
                    odd[(lcg(&mut s) >> 33) as usize % odd.len()]
                } else {
                    ((lcg(&mut s) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.05
                };
                blk[off..off + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        b
    }

    fn xs(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| ((lcg(&mut s) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
            .collect()
    }

    /// Run the fused dot on `row` for `count` activation rows and compare each output to
    /// `dot(&bytes_to_f32(row), x_r)`, bit for bit; returns how many outputs were finite.
    fn check_dot(
        dt: DType,
        row: &[u8],
        in_f: usize,
        count: usize,
        seed: u64,
        label: &str,
    ) -> usize {
        let wf = bytes_to_f32(row, dt);
        let x = xs(count * in_f, seed);
        let mut got = vec![0f32; count];
        let mut accs = Vec::new();
        dequant_dot_rows(row, dt, &x, in_f, &mut got, &mut accs);
        let mut finite = 0;
        for (r, g) in got.iter().enumerate() {
            let want = dot(&wf, &x[r * in_f..r * in_f + in_f]);
            assert!(
                g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan()),
                "{dt:?} {label} in_f {in_f} row {r}/{count}: {g} vs reference {want}"
            );
            finite += g.is_finite() as usize;
        }
        finite
    }

    /// The decoded span is the reference's bit for bit (NaN payloads included) for every affine
    /// format, on finite scales and on the awkward ones.
    #[test]
    fn decode_span_matches_bytes_to_f32_bitwise() {
        for dt in AFFINE {
            let (qpb, _) = infr_core::decode_spec::block_layout(dt);
            for seed in 0..64u64 {
                let nblk = SPAN / qpb;
                let row = blocks(dt, nblk, seed * 7919 + 1, seed % 2 == 1);
                let want = bytes_to_f32(&row, dt);
                let mut got = vec![f32::from_bits(0x7fc0_dead); SPAN];
                decode_span(&row, dt, &mut got);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "{dt:?} seed {seed} element {i}: {g} vs reference {w}"
                    );
                }
            }
        }
    }

    /// The fused dot is `dot(&bytes_to_f32(row), x)` to the bit for every affine format, for one
    /// activation row (decode) and several, over row lengths that end mid-span for the 32-element
    /// formats and span several K-quant super-blocks. Finite scales, so a decoder slip shows as a
    /// numeric mismatch rather than hiding in a NaN — the test asserts the outputs it compared
    /// were finite.
    #[test]
    fn dequant_dot_rows_matches_reference_dot_bitwise() {
        for dt in AFFINE {
            let (qpb, _) = infr_core::decode_spec::block_layout(dt);
            let (mut compared, mut finite) = (0, 0);
            for seed in 0..24u64 {
                let nblk = 1 + (seed as usize * 5 % 23);
                let row = blocks(dt, nblk, seed * 104729 + 11, false);
                for count in [1usize, 3] {
                    let seed = seed * 31 + count as u64;
                    finite += check_dot(dt, &row, nblk * qpb, count, seed, "finite");
                    compared += count;
                }
            }
            assert_eq!(
                finite, compared,
                "{dt:?}: a finite-scale dot came out non-finite"
            );
        }
    }

    /// The same comparison on rows whose blocks carry non-finite and degenerate scales (NaN
    /// matches NaN here), plus the dtypes without a factored form, which take the reference
    /// expression itself.
    #[test]
    fn dequant_dot_rows_matches_reference_on_odd_scales_and_fallback_dtypes() {
        for dt in AFFINE {
            let (qpb, _) = infr_core::decode_spec::block_layout(dt);
            for seed in 0..16u64 {
                let nblk = 1 + (seed as usize * 3 % 17);
                let row = blocks(dt, nblk, seed * 6007 + 5, true);
                check_dot(dt, &row, nblk * qpb, 2, seed, "odd scales");
            }
        }
        for dt in [DType::Iq4Nl, DType::Mxfp4] {
            let (qpb, bpb) = infr_core::decode_spec::block_layout(dt);
            let mut s = 9u64;
            let row: Vec<u8> = (0..4 * bpb).map(|_| (lcg(&mut s) >> 33) as u8).collect();
            check_dot(dt, &row, 4 * qpb, 2, 17, "fallback");
        }
    }

    /// No activation rows: nothing to compute, and nothing is decoded or written.
    #[test]
    fn dequant_dot_rows_with_no_rows_is_a_no_op() {
        let mut accs = vec![[1f32; 8]; 3];
        dequant_dot_rows(&[0u8; 3], DType::Q4K, &[], 256, &mut [], &mut accs);
        assert_eq!(accs.len(), 3, "scratch untouched");
    }
}
