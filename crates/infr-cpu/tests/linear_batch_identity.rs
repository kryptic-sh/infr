//! Prefill `Op::Linear` against decode `Op::Linear`, bit for bit, on every weight dtype.
//!
//! The CPU backend runs a one-row `Op::Linear` (decode) on per-row `vec_dot_*` kernels and a
//! several-row one (prefill) on batched kernels: per-row `*_batch` forms, the Q4_K 2- and 8-row
//! tiles, and on AVX-512 VNNI hosts the packed K-quant GEMM. A chunked prefill is only the same
//! computation as one-position-per-graph prefill (and as the decode steps after it) if every one
//! of those routes gives the one-row kernel's bits. This file pins that, through the public op:
//! one `execute` over `R` rows against `R` executes of one row each, on the same weight bytes.
//! The dtypes in [`ONE_ROW_F32`] differ by design today; their gap is pinned instead.
//!
//! Each case also scores both against a from-definition f64 reference (the host decode of the
//! same bytes, then `Σ w·x` in f64) under an error bound derived from the arithmetic the kernels
//! do, so a mistake both routes share cannot pass as "identical".
//!
//! What the routes cannot be forced to here: the scalar and AVX2-only kernel tiers (chosen by
//! runtime CPU feature detection, with no config switch) and aarch64 (no SIMD kernels there; the
//! scalar ones run). This file runs on whatever tier the host has; `kernels::batch_tier_tests`
//! checks each x86_64 one-row/batch tier pairing directly.
//!
//! Run: `cargo nextest run -p infr-cpu --test linear_batch_identity`

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Plan};
use infr_core::config::{Config, CpuCfg, KernelCfg};
use infr_core::decode_spec::{block_spec, WEIGHT_QUANTS};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_cpu::CpuBackend;
use std::sync::Arc;

/// Prefill row counts: 2 to 5 (both sides of the packed K-quant GEMM's token tile and of Q6_K's
/// packed threshold), and each side of 8, 32, 64 and 128, so every way of tiling the rows meets
/// both a whole tile and a tail.
const ROWS: &[usize] = &[1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 64, 127, 129];

/// Output counts: one row, an odd count under the 8-row tile (2-row pair + odd tail), one 8-row
/// group plus a 7-row tail, and whole 8-row groups.
const OUT_F: &[usize] = &[1, 3, 15, 16];

/// Weight dtypes that are NOT bit-identical between one row and several: their one-row
/// `Op::Linear` dequantizes the weight and takes an f32 dot, while several rows quantize the
/// activations to int8 and run the batch kernel (`vec_dot_q5_0_32_batch`). Only Q5_0, and only
/// because its one-row arm was never moved when its batched one was.
///
/// The alternatives are int8 activations at one row too (every other quantized dtype's choice:
/// consistent, and 2.2-8.8x faster per decode step, but measured further from llama.cpp's logits
/// on gemma-3-1b Q4_K_M: mean KL 8.64e-3 against 7.04e-3, top-1 agreement 45/50 against 48/50)
/// or f32 activations for batches too (closest to the exact sum, at the cost of Q5_0 prefill
/// speed). Which one is the owner's call: docs/backlog.md, B201. Until then this file pins the
/// gap ([`Q5_0_GAP_REL`]) instead of asserting identity.
const ONE_ROW_F32: &[DType] = &[DType::Q5_0];

/// The largest Q5_0 `|prefill − one row| / Σ|w·x|` this file's shapes may show, per
/// [`act_row`] pattern: about 1.5x what was measured (2026-11-17, AVX-512 VNNI host), to allow
/// another host's kernel tier. Ordinary rows sit near 2.6e-3; the spike row (2) loses every
/// other value of its block to the int8 scale, and the subnormal-only row (5) is the quantizer's
/// overflow case (B200), so those two are large on purpose. The zero row (1) must not differ.
const Q5_0_GAP_REL: [f64; 8] = [4e-3, 0.0, 0.65, 5e-3, 6.5e-3, 1.8, 6e-3, 2.5e-2];

/// The largest Q5_0 `|prefill − one row|` over every row: measured 33.2, on the wide-range row.
const Q5_0_GAP_ABS: f64 = 50.0;

/// The float weight dtypes, which take f32 activations at every row count.
const FLOATS: &[DType] = &[DType::F32, DType::F16, DType::Bf16];

/// How a dtype's `Op::Linear` quantizes its activations — mirrored from the dispatch in
/// `infr_cpu`'s `Op::Linear` arm. It sets the error bound, and where one row and several differ
/// ([`ONE_ROW_F32`]) the sweep measures the gap instead of comparing bits.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Acts {
    /// f32 activations: the only error is the f32 dot's rounding.
    F32,
    /// int8 activations, one scale per `n`-element block (`quantize_q8` at 256, `quantize_q8_32`
    /// at 32).
    Int8(usize),
}

fn acts(dt: DType, in_f: usize, rows: usize, reference: bool) -> Acts {
    if reference || FLOATS.contains(&dt) || (rows == 1 && ONE_ROW_F32.contains(&dt)) {
        return Acts::F32;
    }
    match dt {
        DType::Q8_0 if !in_f.is_multiple_of(256) => Acts::Int8(32),
        DType::Q4_0
        | DType::Q4_1
        | DType::Q5_0
        | DType::Q5_1
        | DType::Iq4Nl
        | DType::Mxfp4
        | DType::Nvfp4
        | DType::Q2_0 => Acts::Int8(32),
        _ => Acts::Int8(256),
    }
}

/// Weight widths for a dtype: a single block, an odd block count that is not a multiple of the
/// 256-element activation super-block (or, for the floats, of any SIMD width), and for the
/// 32-block formats a multiple of 256 too (Q8_0 switches activation form on it).
fn widths(dt: DType) -> &'static [usize] {
    if FLOATS.contains(&dt) {
        return &[7, 33, 273];
    }
    match block_spec(dt).block_elems {
        32 => &[32, 288, 512],
        64 => &[64, 320],
        256 => &[256, 768],
        n => panic!("{dt:?}: no widths for a {n}-element block"),
    }
}

/// A deterministic value in `[-1, 1)`.
fn uniform(seed: u64, i: usize) -> f32 {
    let mut z = seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 23) as f32 - 1.0
}

/// Activation row `r`: a cycle of ordinary and adversarial rows. Every pattern is finite: NaN and
/// infinity have no defined result on the int8 path (`quantize_q8` drops a NaN to code 0, and an
/// infinite block maximum zeroes the block's codes under an infinite scale, so the block gives
/// NaN), so they are not compared here.
fn act_row(r: usize, in_f: usize) -> Vec<f32> {
    let s = 0xac7 + r as u64;
    (0..in_f)
        .map(|i| match r % 8 {
            0 => uniform(s, i),
            1 => 0.0,
            // One large value: every other element of its block quantizes to zero.
            2 if i == (r * 7) % in_f => 1.0e4,
            2 => uniform(s, i) * 1e-2,
            3 => -(uniform(s, i).abs() + 1e-2),
            // Exactly halfway between two int8 codes. Each 32-element block holds `127·STEP`, so
            // its scale is `STEP` (a power of two: the division and its inverse are exact) at
            // both block sizes, and every other value lands on `k + 0.5` codes.
            4 if i % 32 == 0 => 127.0 * TIE_STEP,
            4 => (((i * 37 + r) % 253) as f32 - 126.5) * TIE_STEP,
            // Subnormals only: the block maximum's inverse overflows in the quantizer.
            5 => uniform(s, i) * 1e-39,
            // Ordinary values with subnormals and exact zeros mixed in.
            6 if i % 5 == 0 => uniform(s, i) * 1e-40,
            6 if i % 7 == 0 => 0.0,
            6 => uniform(s, i),
            // A wide dynamic range inside each block.
            _ => uniform(s, i) * 10f32.powi((i % 7) as i32 - 3),
        })
        .collect()
}

/// The tie rows' code step; see [`act_row`].
const TIE_STEP: f32 = 0.125;

/// Weight bytes for `out_f` rows of `in_f`.
fn weight(dt: DType, in_f: usize, out_f: usize, seed: u32) -> Vec<u8> {
    let n = in_f * out_f;
    let vals = || (0..n).map(|i| uniform(seed as u64, i) * 0.5);
    match dt {
        DType::F32 => vals().flat_map(f32::to_le_bytes).collect(),
        DType::F16 => vals()
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect(),
        DType::Bf16 => vals()
            .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
            .collect(),
        _ => infr_testkit::synth_weight(dt, n, seed),
    }
}

/// One `Op::Linear` over `m` rows, compiled once and run on as many activations as asked.
struct LinearRun<'a> {
    be: &'a CpuBackend,
    plan: Box<dyn Plan>,
    x: Box<dyn Buffer>,
    out: Box<dyn Buffer>,
    binds: (TensorId, TensorId),
    w: TensorId,
    m: usize,
    out_f: usize,
}

impl<'a> LinearRun<'a> {
    fn new(be: &'a CpuBackend, dt: DType, m: usize, in_f: usize, out_f: usize) -> Self {
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
        let w = g.weight(TensorDesc::new(vec![out_f, in_f], dt));
        let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
        g.push(Op::Linear {
            x,
            weight: w,
            dst,
            m: m as u32,
            in_f: in_f as u32,
            out_f: out_f as u32,
            w_off: 0,
        });
        Self {
            be,
            plan: be.compile(&g).expect("compile"),
            x: be
                .alloc(m * in_f * 4, BufferUsage::Activations)
                .expect("alloc x"),
            out: be
                .alloc(m * out_f * 4, BufferUsage::Readback)
                .expect("alloc out"),
            binds: (x, dst),
            w,
            m,
            out_f,
        }
    }

    fn run(&self, w: &dyn Buffer, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len() % self.m, 0);
        self.be
            .upload(self.x.as_ref(), bytemuck::cast_slice(x))
            .expect("upload x");
        let mut b = Bindings::new();
        b.bind(self.binds.0, self.x.as_ref());
        b.bind(self.w, w);
        b.bind(self.binds.1, self.out.as_ref());
        self.be.execute(self.plan.as_ref(), &b).expect("execute");
        let mut o = vec![0f32; self.m * self.out_f];
        self.be
            .download(self.out.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .expect("download");
        o
    }
}

/// The f64 reference for one output and the error bound the kernels' arithmetic allows there.
///
/// int8 activations: `quantize_q8*` sets a block's scale `d = max|x|/127` and its codes
/// `round(x/d)`, so each activation moves by at most `d/2`, and the dot by at most
/// `Σ|w|·d/2` over the block. When `1/d` overflows (a subnormal block maximum) every nonzero
/// value takes the full code and moves by up to `max|x|`; the bound takes that. On top, the
/// kernels' f32 arithmetic: per-block products and partial sums (scale × integer dot, the K-quant
/// min term) rounded `O(blocks)` times, bounded by `in_f·ε` times the sum of the magnitudes those
/// terms carry. f32 activations: `in_f·ε·Σ|w·x|` (the textbook bound on an f32 dot), plus the
/// absolute loss of subnormal products.
fn reference(w: &[f32], x: &[f32], acts: Acts) -> (f64, f64) {
    let n = x.len();
    let eps = f32::EPSILON as f64;
    let exact: f64 = w.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum();
    let mag: f64 = w
        .iter()
        .zip(x)
        .map(|(&a, &b)| (a as f64 * b as f64).abs())
        .sum();
    let tiny = n as f64 * f32::from_bits(1) as f64;
    let bound = match acts {
        Acts::F32 => n as f64 * eps * mag + tiny,
        Acts::Int8(blk) => {
            let wmax = w.iter().fold(0f64, |m, &v| m.max(v.abs() as f64));
            let mut quant = 0f64;
            let mut carried = 0f64;
            for (wb, xb) in w.chunks(blk).zip(x.chunks(blk)) {
                let amax = xb.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let d = amax / 127.0;
                let step = if d > 0.0 && (1.0 / d).is_finite() {
                    d as f64 / 2.0
                } else {
                    amax as f64
                };
                let wsum: f64 = wb.iter().map(|&v| v.abs() as f64).sum();
                quant += wsum * step;
                // The dequantized activations the integer dot stands for, against the weight
                // magnitudes and (for the K-quant mins) the row's largest weight.
                let xsum: f64 = xb.iter().map(|&v| v.abs() as f64 + step).sum();
                carried += (wsum + wmax * blk as f64) * (amax as f64 + step) + wmax * xsum;
            }
            // The quantizer's `1/d` and `x·(1/d)` are f32: a code can land a rounding past the
            // half step, which the small relative margin on `quant` covers.
            quant * (1.0 + 1e-3) + n as f64 * eps * carried + tiny
        }
    };
    (exact, bound)
}

/// [`reference`] plus `Σ|w·x|`, the scale a gap between two routes is measured against.
fn reference_mag(w: &[f32], x: &[f32], acts: Acts) -> (f64, f64, f64) {
    let (exact, bound) = reference(w, x, acts);
    let mag = w
        .iter()
        .zip(x)
        .map(|(&a, &b)| (a as f64 * b as f64).abs())
        .sum();
    (exact, bound, mag)
}

/// What one dtype's sweep found: how many elements were compared, and the largest
/// `|got − reference| / bound` over them (≤ 1 passes).
struct Sweep {
    elems: usize,
    worst: f64,
    /// For a dtype in [`ONE_ROW_F32`]: how far its prefill values sit from its one-row ones.
    gap: Gap,
}

/// The gap between prefill and one-row values of a dtype that is not bit-identical, by
/// activation pattern ([`act_row`]'s `r % 8`).
#[derive(Default)]
struct Gap {
    compared: usize,
    differing: usize,
    /// Largest `|prefill − one row|` per pattern.
    abs: [f64; 8],
    /// Largest `|prefill − one row| / Σ|w·x|` per pattern: the gap against the output's own
    /// magnitude scale. A zero row has none and stays 0.
    rel: [f64; 8],
}

/// Check one dtype at every width, output count and row count on `be`: each prefill row bit for
/// bit equal to the same row run alone, and every value inside the reference bound. Panics with
/// the first offender.
fn sweep(be: &CpuBackend, dt: DType, reference_mode: bool) -> Sweep {
    let max_m = *ROWS.iter().max().unwrap();
    let mut s = Sweep {
        elems: 0,
        worst: 0.0,
        gap: Gap::default(),
    };
    for (wi, &in_f) in widths(dt).iter().enumerate() {
        let a = acts(dt, in_f, 1, reference_mode);
        let a_batch = acts(dt, in_f, 2, reference_mode);
        let xs: Vec<Vec<f32>> = (0..max_m).map(|r| act_row(r, in_f)).collect();
        for &out_f in OUT_F {
            let wbytes = weight(dt, in_f, out_f, 0x11 + (wi * 31 + out_f) as u32);
            let wvals = infr_testkit::dequant_oracle(dt, &wbytes);
            let wbuf = be
                .alloc(wbytes.len(), BufferUsage::Weights)
                .expect("alloc w");
            be.upload(wbuf.as_ref(), &wbytes).expect("upload w");
            let one = LinearRun::new(be, dt, 1, in_f, out_f);
            let single: Vec<Vec<f32>> = xs.iter().map(|x| one.run(wbuf.as_ref(), x)).collect();
            // Score the one-row results against the reference once; the prefill ones then only
            // have to match them bit for bit.
            for (r, (x, got)) in xs.iter().zip(&single).enumerate() {
                for (o, &g) in got.iter().enumerate() {
                    let (want, bound) = reference(&wvals[o * in_f..(o + 1) * in_f], x, a);
                    let err = (g as f64 - want).abs();
                    assert!(
                        err <= bound,
                        "{dt:?} in_f={in_f} out_f={out_f} row {r} (pattern {}) o={o}: \
                         one-row {g:e} vs reference {want:e}: error {err:e} over bound {bound:e}",
                        r % 8
                    );
                    s.worst = s.worst.max(if bound > 0.0 { err / bound } else { 0.0 });
                    s.elems += 1;
                }
            }
            for &m in ROWS.iter().filter(|&&m| m > 1) {
                let x: Vec<f32> = xs[..m].concat();
                let got = LinearRun::new(be, dt, m, in_f, out_f).run(wbuf.as_ref(), &x);
                for (r, want) in single[..m].iter().enumerate() {
                    for (o, w) in want.iter().enumerate() {
                        let g = got[r * out_f + o];
                        if a_batch != a {
                            // A dtype whose one row and batches quantize differently: score the
                            // batch against the reference on its own, and measure the gap.
                            let (exact, bound, mag) =
                                reference_mag(&wvals[o * in_f..(o + 1) * in_f], &xs[r], a_batch);
                            let err = (g as f64 - exact).abs();
                            assert!(
                                err <= bound,
                                "{dt:?} in_f={in_f} out_f={out_f} m={m} row {r} o={o}: prefill \
                                 {g:e} vs reference {exact:e}: error {err:e} over bound {bound:e}"
                            );
                            let (d, p) = ((g as f64 - *w as f64).abs(), r % 8);
                            s.gap.compared += 1;
                            s.gap.differing += (g.to_bits() != w.to_bits()) as usize;
                            s.gap.abs[p] = s.gap.abs[p].max(d);
                            if mag > 0.0 {
                                s.gap.rel[p] = s.gap.rel[p].max(d / mag);
                            }
                            continue;
                        }
                        assert_eq!(
                            g.to_bits(),
                            w.to_bits(),
                            "{dt:?} in_f={in_f} out_f={out_f} m={m} row {r} (pattern {}) o={o}: \
                             prefill {g:e} != one-row {w:e}",
                            r % 8
                        );
                    }
                }
                s.elems += m * out_f;
            }
        }
    }
    s
}

fn every_dtype() -> Vec<DType> {
    FLOATS.iter().chain(WEIGHT_QUANTS).copied().collect()
}

/// The production kernels: every weight dtype's prefill route gives its decode route's bits,
/// except the [`ONE_ROW_F32`] ones, whose gap is pinned instead (and must be there: if it closes,
/// move the dtype back to the bitwise check).
#[test]
fn prefill_linear_matches_one_row_linear_bitwise_for_every_dtype() {
    let be = CpuBackend::new_with(Arc::new(Config::default()));
    for dt in every_dtype() {
        let s = sweep(&be, dt, false);
        if ONE_ROW_F32.contains(&dt) {
            let g = &s.gap;
            assert!(
                g.compared > 0 && g.rel[0] > 0.0,
                "{dt:?}: prefill now equals the one-row path on ordinary rows; make it bitwise"
            );
            for (p, (&got, &max)) in g.rel.iter().zip(&Q5_0_GAP_REL).enumerate() {
                assert!(
                    got <= max,
                    "{dt:?} pattern {p}: prefill is {got:e} of the output's scale from the \
                     one-row path, over the pinned {max:e}"
                );
            }
            let abs = g.abs.iter().fold(0f64, |m, &v| m.max(v));
            assert!(
                abs <= Q5_0_GAP_ABS,
                "{dt:?}: prefill is {abs:e} from one row"
            );
        } else {
            assert_eq!(s.gap.compared, 0, "{dt:?} skipped the bitwise check");
        }
        eprintln!(
            "{dt:?}: {} values, worst error {:.3} of the bound; {} of {} prefill values off \
             the one-row ones, by pattern: abs {:?}, relative {:?}",
            s.elems,
            s.worst,
            s.gap.differing,
            s.gap.compared,
            sci(&s.gap.abs),
            sci(&s.gap.rel)
        );
    }
}

/// `v` in scientific notation, for the sweep's report.
fn sci(v: &[f64]) -> Vec<String> {
    v.iter().map(|x| format!("{x:.2e}")).collect()
}

/// Reference mode (`kernels.cpu.reference`): the f32-dequant dot at every row count.
#[test]
fn reference_prefill_linear_matches_one_row_linear_bitwise_for_every_dtype() {
    let cfg = Config {
        kernels: KernelCfg {
            cpu: CpuCfg {
                reference: true,
                ..CpuCfg::default()
            },
            ..KernelCfg::default()
        },
        ..Config::default()
    };
    let be = CpuBackend::new_with(Arc::new(cfg));
    for dt in every_dtype() {
        let s = sweep(&be, dt, true);
        eprintln!(
            "reference {dt:?}: {} values, worst error {:.3} of the bound",
            s.elems, s.worst
        );
    }
}
