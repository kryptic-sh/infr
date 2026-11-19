//! Every kernel family gives its scalar kernel's bits at every SIMD tier, for every input.
//!
//! The CPU's output is defined to be the same at every tier (`crate::simd`'s rule): integer
//! accumulation is exact in any order, and the floating-point expression a kernel finishes with
//! is evaluated in one order everywhere. This holds each family to it through the dispatcher
//! production calls, run on this thread at each tier the host has ([`with_tier`]), against the
//! same dispatcher at [`SimdTier::Scalar`]. Each run also asserts the tier the dispatcher
//! actually took (the widest tier a check allowed), so a ceiling that did nothing, or a family
//! that gained or lost a tier without this table noticing, fails here instead of passing as
//! "identical".
//!
//! The inputs are integer-generated: LCG weight payloads with every code pattern, per-block
//! scales of both signs, zero, negative zero, f16 subnormals and large ones, and activation rows
//! that are zero, single-spike, all at the int8 extremes (the most `maddubs`-style pair sums can
//! reach), on exact rounding ties, subnormal, negative zero. Lengths cover one block, a few, and
//! odd block counts against the kernels' 2- and 4-block steps; activation counts cover the batch
//! kernels' row-pair and row-quad tails.
//!
//! Families with no SIMD tier (the grid-codebook and ternary formats) are listed too: they must
//! report no tier at all. The f32 dots (`dot`, `dot_n`, `dot_mn`, `dot_f16`, `dot_bf16`, `axpy`,
//! `dequant_dot_rows`) have their own tests at the end: on a build without FMA they dispatch to
//! FMA code at the AVX2 tier, which must give the bits of the C runtime's `fmaf`, and every one
//! of them must be fused at every tier.

// The module is test-only (`kernels.rs` declares it under `cfg(test)`); the attribute says so
// to `infr-core/tests/no_bare_print.rs` too, which scans a file down to its first one.
#[cfg(test)]
use super::*;
use crate::quant::{quantize_q8, quantize_q8_32};
use crate::simd::{with_tier, SimdTier};
use infr_core::decode_spec::block_spec;
use infr_core::tensor::DType;
use SimdTier::{Avx2, Avx512, Avx512Vnni as Vnni};

/// A dispatcher's two compilations, `[uncapped, capped]` (`crate::simd`), indexed by `CAP`; a
/// family with no SIMD tier has one function in both places.
type Pair2<F> = [F; 2];
/// A one-row kernel.
type OneFn = fn(&[u8], &Q8, usize) -> f32;
/// A batch kernel against `Q8` activations.
type BatchFn = fn(&[u8], &[Q8], usize, &mut [f32]);
/// A batch kernel against `Q8x32` activations.
type Batch32Fn = fn(&[u8], &[Q8x32], usize, &mut [f32]);

/// Both compilations of a tiered dispatcher.
macro_rules! tiered_fns {
    ($f:ident) => {
        [$f::<false>, $f::<true>]
    };
}

/// How a family is run: every form computes `out[row][act]` for the 8 weight rows.
#[derive(Clone, Copy)]
enum Form {
    /// One weight row against one `Q8` activation.
    One(Pair2<OneFn>),
    /// One weight row against every `Q8` activation.
    Batch(Pair2<BatchFn>),
    /// Q4_K's 2-row tile.
    Pair,
    /// Q4_K's 8-row tile.
    Eight,
    /// One weight row against every `Q8x32` activation.
    Batch32(Pair2<Batch32Fn>),
    /// The 8-row tile of the native-32-block Q8_0 (`false`) or Q5_0 (`true`).
    Eight32(bool),
}

/// A kernel family: its name, its weight format, the tiers its dispatcher has besides scalar,
/// and how to run it.
type Family = (&'static str, DType, &'static [SimdTier], Form);

/// Weight rows per case: one full 8-row tile.
const ROWS: usize = 8;

fn families() -> Vec<Family> {
    use Form::*;
    let k: &[SimdTier] = &[Avx2, Avx512];
    let kv: &[SimdTier] = &[Avx2, Avx512, Vnni];
    let v: &[SimdTier] = &[Avx2, Vnni];
    let none: &[SimdTier] = &[];
    vec![
        ("q4k", DType::Q4K, k, One(tiered_fns!(vec_dot_q4k))),
        ("q2k", DType::Q2K, k, One(tiered_fns!(vec_dot_q2k))),
        ("q3k", DType::Q3K, k, One(tiered_fns!(vec_dot_q3k))),
        ("q5k", DType::Q5K, k, One(tiered_fns!(vec_dot_q5k))),
        ("q6k", DType::Q6K, k, One(tiered_fns!(vec_dot_q6k))),
        ("q8_0", DType::Q8_0, k, One(tiered_fns!(vec_dot_q8_0))),
        ("iq4xs", DType::Iq4Xs, k, One(tiered_fns!(vec_dot_iq4xs))),
        ("iq2s", DType::Iq2S, none, One([vec_dot_iq2s, vec_dot_iq2s])),
        (
            "iq2xxs",
            DType::Iq2Xxs,
            none,
            One([vec_dot_iq2xxs, vec_dot_iq2xxs]),
        ),
        (
            "iq2xs",
            DType::Iq2Xs,
            none,
            One([vec_dot_iq2xs, vec_dot_iq2xs]),
        ),
        ("iq3s", DType::Iq3S, none, One([vec_dot_iq3s, vec_dot_iq3s])),
        (
            "iq3xxs",
            DType::Iq3Xxs,
            none,
            One([vec_dot_iq3xxs, vec_dot_iq3xxs]),
        ),
        ("iq1s", DType::Iq1S, none, One([vec_dot_iq1s, vec_dot_iq1s])),
        ("iq1m", DType::Iq1M, none, One([vec_dot_iq1m, vec_dot_iq1m])),
        (
            "tq1_0",
            DType::Tq1_0,
            none,
            One([vec_dot_tq1_0, vec_dot_tq1_0]),
        ),
        (
            "tq2_0",
            DType::Tq2_0,
            none,
            One([vec_dot_tq2_0, vec_dot_tq2_0]),
        ),
        (
            "q4k_batch",
            DType::Q4K,
            kv,
            Batch(tiered_fns!(vec_dot_q4k_batch)),
        ),
        ("q4k_batch2", DType::Q4K, kv, Pair),
        // Below AVX-512 the tile is the per-row batch kernel, AVX2 included.
        ("q4k_batch8", DType::Q4K, k, Eight),
        (
            "q2k_batch",
            DType::Q2K,
            kv,
            Batch(tiered_fns!(vec_dot_q2k_batch)),
        ),
        (
            "q3k_batch",
            DType::Q3K,
            kv,
            Batch(tiered_fns!(vec_dot_q3k_batch)),
        ),
        (
            "q5k_batch",
            DType::Q5K,
            k,
            Batch(tiered_fns!(vec_dot_q5k_batch)),
        ),
        (
            "q6k_batch",
            DType::Q6K,
            v,
            Batch(tiered_fns!(vec_dot_q6k_batch)),
        ),
        (
            "q8_0_batch",
            DType::Q8_0,
            k,
            Batch(tiered_fns!(vec_dot_q8_0_batch)),
        ),
        (
            "iq4xs_batch",
            DType::Iq4Xs,
            &[Avx2],
            Batch(tiered_fns!(vec_dot_iq4xs_batch)),
        ),
        (
            "iq2s_batch",
            DType::Iq2S,
            none,
            Batch([vec_dot_iq2s_batch, vec_dot_iq2s_batch]),
        ),
        (
            "iq2xxs_batch",
            DType::Iq2Xxs,
            none,
            Batch([vec_dot_iq2xxs_batch, vec_dot_iq2xxs_batch]),
        ),
        (
            "iq2xs_batch",
            DType::Iq2Xs,
            none,
            Batch([vec_dot_iq2xs_batch, vec_dot_iq2xs_batch]),
        ),
        (
            "iq3s_batch",
            DType::Iq3S,
            none,
            Batch([vec_dot_iq3s_batch, vec_dot_iq3s_batch]),
        ),
        (
            "iq3xxs_batch",
            DType::Iq3Xxs,
            none,
            Batch([vec_dot_iq3xxs_batch, vec_dot_iq3xxs_batch]),
        ),
        (
            "iq1s_batch",
            DType::Iq1S,
            none,
            Batch([vec_dot_iq1s_batch, vec_dot_iq1s_batch]),
        ),
        (
            "iq1m_batch",
            DType::Iq1M,
            none,
            Batch([vec_dot_iq1m_batch, vec_dot_iq1m_batch]),
        ),
        (
            "tq1_0_batch",
            DType::Tq1_0,
            none,
            Batch([vec_dot_tq1_0_batch, vec_dot_tq1_0_batch]),
        ),
        (
            "tq2_0_batch",
            DType::Tq2_0,
            none,
            Batch([vec_dot_tq2_0_batch, vec_dot_tq2_0_batch]),
        ),
        (
            "q8_0_32",
            DType::Q8_0,
            kv,
            Batch32(tiered_fns!(vec_dot_q8_0_32_batch)),
        ),
        (
            "q5_0_32",
            DType::Q5_0,
            v,
            Batch32(tiered_fns!(vec_dot_q5_0_32_batch)),
        ),
        (
            "q4_0_32",
            DType::Q4_0,
            v,
            Batch32(tiered_fns!(vec_dot_q4_0_32_batch)),
        ),
        (
            "q4_1_32",
            DType::Q4_1,
            v,
            Batch32(tiered_fns!(vec_dot_q4_1_32_batch)),
        ),
        (
            "q5_1_32",
            DType::Q5_1,
            v,
            Batch32(tiered_fns!(vec_dot_q5_1_32_batch)),
        ),
        (
            "q2_0",
            DType::Q2_0,
            v,
            Batch32(tiered_fns!(vec_dot_q2_0_batch)),
        ),
        (
            "iq4nl_32",
            DType::Iq4Nl,
            v,
            Batch32(tiered_fns!(vec_dot_iq4nl_32_batch)),
        ),
        (
            "mxfp4_32",
            DType::Mxfp4,
            v,
            Batch32(tiered_fns!(vec_dot_mxfp4_32_batch)),
        ),
        (
            "nvfp4",
            DType::Nvfp4,
            v,
            Batch32(tiered_fns!(vec_dot_nvfp4_batch)),
        ),
        // Below VNNI the tile is the per-row kernel, whose own tiers then show.
        ("q8_0_32_batch8", DType::Q8_0, kv, Eight32(false)),
        ("q5_0_32_batch8", DType::Q5_0, v, Eight32(true)),
    ]
}

/// The tier a family's dispatcher takes under a ceiling: its widest tier not above it.
fn expected(tiers: &[SimdTier], ceiling: SimdTier) -> Option<SimdTier> {
    tiers.iter().copied().filter(|&t| t <= ceiling).max()
}

/// `(d, min)` multipliers of the format's own magnitudes, one per block in turn: both signs,
/// zero and negative zero, an f16 subnormal, a large scale.
const SCALES: [(f32, f32); 7] = [
    (1.0, 1.0),
    (-1.0, 0.5),
    (0.0, -1.0),
    (-0.0, -0.0),
    (1.0 / 4096.0, 2.0),
    (64.0, -64.0),
    (3.0, 0.0),
];

/// `ROWS` weight rows of `in_f`: LCG payloads (every code pattern), the scales cycling through
/// [`SCALES`] so neighbouring blocks differ.
fn weights(dt: DType, in_f: usize, seed: u32) -> Vec<u8> {
    crate::test_util::synth_weights(dt, ROWS, in_f, seed, &SCALES)
}

/// Activation row `r` of `in_f`, from integers: ordinary values, zero, one spike, every element
/// at the int8 extreme of one sign or with the sign alternating, exact rounding ties,
/// subnormals, negative zeros among ordinary values.
fn act_row(r: usize, in_f: usize) -> Vec<f32> {
    let base = |i: usize| ((i * 29 + r * 7) % 97) as f32 / 13.0 - 3.6875;
    (0..in_f)
        .map(|i| match r % 9 {
            0 => base(i),
            1 => 0.0,
            2 if i == (r * 11) % in_f => 1.0e4,
            2 => base(i) * 1e-2,
            3 => 5.0,
            4 => -5.0,
            5 => {
                if (i / 3) % 2 == 0 {
                    7.5
                } else {
                    -7.5
                }
            }
            // A block maximum of 127/8 makes the scale 1/8 exactly, so the rest sit on `k + ½`.
            6 if i % 32 == 0 => 127.0 / 8.0,
            6 => (((i * 37 + r) % 253) as f32 - 126.5) / 8.0,
            7 => base(i) * 1e-39,
            _ if i % 5 == 0 => -0.0,
            _ => base(i),
        })
        .collect()
}

/// `out[row * m + act]` for the 8 weight rows `w` (`bpr` bytes each) and `m` activations, through
/// the dispatchers' `CAP` compilation.
fn run<const CAP: bool>(form: Form, w: &[u8], bpr: usize, x: &[Vec<f32>], in_f: usize) -> Vec<f32> {
    let m = x.len();
    let mut out = vec![f32::NAN; ROWS * m];
    let rows: Vec<&[u8]> = w.chunks(bpr).collect();
    let q8s: Vec<Q8> = match form {
        Form::Batch32(_) | Form::Eight32(_) => Vec::new(),
        _ => x.iter().map(|r| quantize_q8(r)).collect(),
    };
    let q32s: Vec<Q8x32> = match form {
        Form::Batch32(_) | Form::Eight32(_) => x.iter().map(|r| quantize_q8_32(r)).collect(),
        _ => Vec::new(),
    };
    let eight: [&[u8]; ROWS] = std::array::from_fn(|i| rows[i]);
    match form {
        Form::One(f) => {
            for (o, row) in rows.iter().enumerate() {
                for (r, q8) in q8s.iter().enumerate() {
                    out[o * m + r] = f[CAP as usize](row, q8, in_f);
                }
            }
        }
        Form::Batch(f) => {
            for (row, col) in rows.iter().zip(out.chunks_mut(m)) {
                f[CAP as usize](row, &q8s, in_f, col);
            }
        }
        Form::Pair => {
            for (p, cols) in out.chunks_mut(2 * m).enumerate() {
                let (a, b) = cols.split_at_mut(m);
                vec_dot_q4k_batch2::<CAP>(rows[2 * p], rows[2 * p + 1], &q8s, in_f, a, b);
            }
        }
        Form::Eight => {
            let mut it = out.chunks_mut(m);
            vec_dot_q4k_batch8::<CAP>(
                eight,
                &q8s,
                in_f,
                std::array::from_fn(|_| it.next().unwrap()),
            );
        }
        Form::Batch32(f) => {
            for (row, col) in rows.iter().zip(out.chunks_mut(m)) {
                f[CAP as usize](row, &q32s, in_f, col);
            }
        }
        Form::Eight32(q5) => {
            let mut it = out.chunks_mut(m);
            vec_dot_q32_batch8::<CAP>(
                eight,
                &q32s,
                in_f,
                std::array::from_fn(|_| it.next().unwrap()),
                q5,
            );
        }
    }
    out
}

/// The lengths a family runs at: whole blocks of its format and of its activation format.
fn lengths(dt: DType, form: Form) -> Vec<usize> {
    let block = block_spec(dt).block_elems;
    let all: &[usize] = match form {
        Form::Batch32(_) | Form::Eight32(_) => &[32, 64, 96, 192, 288, 576, 1056],
        _ => &[256, 512, 768, 1280],
    };
    all.iter().copied().filter(|n| n % block == 0).collect()
}

/// Activation counts: one, the batch kernels' pair and quad tails, a row-pair tile plus one.
const ACTS: [usize; 4] = [1, 3, 9, 17];

/// Each family, at each tier this CPU has, gives the scalar tier's bits on every input, and its
/// dispatcher took the tier the table says; and so does each dispatcher's uncapped compilation. Turns red when a tier's kernel computes a different
/// value (a different float association, a saturating intermediate, a different rounding) or a
/// dispatcher ignores the ceiling.
#[test]
fn every_kernel_family_gives_the_scalar_bits_at_every_tier() {
    let host: Vec<SimdTier> = SimdTier::ALL
        .into_iter()
        .filter(|t| t.supported())
        .collect();
    let lacking: Vec<&str> = SimdTier::ALL
        .into_iter()
        .filter(|t| !t.supported())
        .map(|t| t.name())
        .collect();
    if !lacking.is_empty() {
        eprintln!("tier_tests: this CPU lacks {lacking:?}; those tiers are not checked here");
    }
    let mut ran = std::collections::BTreeMap::<SimdTier, usize>::new();
    let mut checked = 0usize;
    let mut uncapped = 0usize;
    let mut failures = Vec::new();
    for (name, dt, tiers, form) in families() {
        for in_f in lengths(dt, form) {
            let w = weights(dt, in_f, 0x7e5 ^ in_f as u32);
            let bpr = w.len() / ROWS;
            for m in ACTS {
                let x: Vec<Vec<f32>> = (0..m).map(|r| act_row(r + m, in_f)).collect();
                let (want, took) =
                    with_tier(SimdTier::Scalar, || run::<true>(form, &w, bpr, &x, in_f));
                assert_eq!(took, None, "{name}: a check passed at the scalar tier");
                for &t in &host {
                    let (got, took) = with_tier(t, || run::<true>(form, &w, bpr, &x, in_f));
                    assert_eq!(
                        took,
                        expected(tiers, t),
                        "{name} in_f {in_f} m {m} at {}: dispatched to a different tier",
                        t.name()
                    );
                    *ran.entry(t).or_default() += 1;
                    let diff = got
                        .iter()
                        .zip(&want)
                        .position(|(g, w)| g.to_bits() != w.to_bits());
                    if let Some(i) = diff {
                        failures.push(format!(
                            "{name} at {}: in_f {in_f}, m {m}, weight row {}, act {}: \
                             {:e} ({:#010x}) vs scalar {:e} ({:#010x})",
                            t.name(),
                            i / m,
                            i % m,
                            got[i],
                            got[i].to_bits(),
                            want[i],
                            want[i].to_bits()
                        ));
                    }
                    checked += 1;
                }
                // The uncapped compilation, the one production runs with no ceiling: the CPU's
                // widest kernels, by the feature checks alone. It asks no ceiling, so it records
                // no tier.
                let (got, took) =
                    with_tier(SimdTier::Scalar, || run::<false>(form, &w, bpr, &x, in_f));
                assert_eq!(
                    took, None,
                    "{name}: the uncapped compilation asked the ceiling"
                );
                if got
                    .iter()
                    .zip(&want)
                    .any(|(g, w)| g.to_bits() != w.to_bits())
                {
                    failures.push(format!("{name} uncapped: in_f {in_f}, m {m} differs"));
                }
                uncapped += 1;
            }
        }
    }
    eprintln!("tier_tests: {checked} capped runs, by tier {ran:?}; {uncapped} uncapped");
    assert!(
        failures.is_empty(),
        "{} runs differ from scalar:\n{}",
        failures.len(),
        failures.join("\n")
    );
    // Every host tier ran every family at every length and count: nothing matched nothing.
    let per_tier: usize = families()
        .iter()
        .map(|&(_, dt, _, form)| lengths(dt, form).len() * ACTS.len())
        .sum();
    assert!(per_tier > 0);
    for t in host {
        assert_eq!(ran.get(&t), Some(&per_tier), "{t:?}");
    }
    assert_eq!(uncapped, per_tier);
}

/// Every weight format the `Op::Linear` interpreter dispatches to an int8 kernel has a family
/// above (dense formats included in `infr_core::decode_spec::WEIGHT_QUANTS`).
#[test]
fn every_quantized_format_has_a_family() {
    let fams = families();
    for &dt in infr_core::decode_spec::WEIGHT_QUANTS {
        let covered = fams.iter().any(|f| f.1 == dt);
        let turbo = matches!(dt, DType::Turbo2 | DType::Turbo3 | DType::Turbo4);
        assert!(covered || turbo, "{dt:?}: no kernel family in tier_tests");
    }
}

/// The sparse part of a split activation (`Q8::hi`, dotted by `vec_dot_hi`) gives the scalar
/// tier's bits at every tier, and so does the dense dot beside it, for every weight dtype that
/// takes `Q8` activations, on rows that split: two spikes in the first block, and at the longer
/// length one more in the last.
#[test]
fn the_sparse_activation_part_gives_the_scalar_bits_at_every_tier() {
    let dts = [
        DType::Q4K,
        DType::Q6K,
        DType::Q8_0,
        DType::Q5K,
        DType::Q2K,
        DType::Q3K,
        DType::Iq4Xs,
        DType::Iq2S,
        DType::Iq2Xs,
        DType::Iq2Xxs,
        DType::Iq1S,
        DType::Iq1M,
        DType::Iq3S,
        DType::Iq3Xxs,
        DType::Tq2_0,
        DType::Tq1_0,
    ];
    let host: Vec<SimdTier> = SimdTier::ALL
        .into_iter()
        .filter(|t| t.supported())
        .collect();
    let mut checked = 0usize;
    for dt in dts {
        for in_f in [256usize, 768] {
            let w = weights(dt, in_f, 0x51 ^ in_f as u32);
            let bpr = w.len() / ROWS;
            let x: Vec<f32> = (0..in_f)
                .map(|i| match i {
                    3 => 412.5,
                    7 => -129.3,
                    i if i == in_f - 9 => -3.0e4,
                    _ => ((i * 29 % 97) as f32 - 48.0) / 31.0,
                })
                .collect();
            let q8 = quantize_q8(&x);
            assert!(
                !q8.hi.is_empty(),
                "{dt:?} in_f {in_f}: the row did not split"
            );
            let both = |cap: bool| -> Vec<(f32, f32)> {
                (0..ROWS)
                    .map(|o| {
                        let row = &w[o * bpr..(o + 1) * bpr];
                        if cap {
                            (
                                vec_dot_q8_any::<true>(dt, row, &q8, in_f),
                                vec_dot_hi(dt, row, &q8, in_f),
                            )
                        } else {
                            (
                                vec_dot_q8_any::<false>(dt, row, &q8, in_f),
                                vec_dot_hi(dt, row, &q8, in_f),
                            )
                        }
                    })
                    .collect()
            };
            let (want, _) = with_tier(SimdTier::Scalar, || both(true));
            assert!(
                want.iter().any(|&(_, h)| h != 0.0),
                "{dt:?} in_f {in_f}: the sparse part dotted to zero on every row"
            );
            let bits = |v: &[(f32, f32)]| -> Vec<(u32, u32)> {
                v.iter().map(|&(a, b)| (a.to_bits(), b.to_bits())).collect()
            };
            for &t in &host {
                let (got, _) = with_tier(t, || both(true));
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{dt:?} in_f {in_f} at {}",
                    t.name()
                );
                checked += 1;
            }
            let (got, _) = with_tier(SimdTier::Scalar, || both(false));
            assert_eq!(bits(&got), bits(&want), "{dt:?} in_f {in_f} uncapped");
        }
    }
    assert_eq!(checked, dts.len() * 2 * host.len());
}

/// Scales for [`the_fast_sparse_dot_gives_the_one_row_kernels_bits`]: [`SCALES`]' finite set,
/// then one where some blocks' scales are infinite or NaN, so a sparse dot that skipped such a
/// block's term (whose `NaN` the one-row kernel keeps) would differ.
const HI_SCALES: [&[(f32, f32)]; 2] = [
    &SCALES,
    &[
        (1.0, 1.0),
        (f32::INFINITY, 1.0),
        (1.0, f32::NEG_INFINITY),
        (f32::NAN, 0.5),
        (-1.0, 0.5),
    ],
];

/// The sparse part's fast dot (`vec_dot_hi`, which reads only the weight elements at the part's
/// codes) gives the bits of the one-row kernels it replaced (`vec_dot_hi_reference`: each part
/// made whole and dotted by `vec_dot_q8_any`) at every tier the host has, for every format that
/// takes `Q8` activations: on every pattern of `test_util::spike_patterns` (one spike, two,
/// several blocks, the first and last index of a block and of each sub-block size, the last
/// block, equal and opposite spikes, a second part, a scale that overflows against an f16
/// maximum), on weights with finite scales and with infinite and NaN ones.
#[test]
fn the_fast_sparse_dot_gives_the_one_row_kernels_bits() {
    use crate::test_util::{spike_patterns, spiky_row, Q8_DTYPES};
    let host: Vec<SimdTier> = SimdTier::ALL
        .into_iter()
        .filter(|t| t.supported())
        .collect();
    let in_f = 768usize;
    let rows: Vec<Q8> = (0..spike_patterns(in_f).len())
        .map(|r| quantize_q8(&spiky_row(r, in_f)))
        .collect();
    // The patterns reach what they are meant to: a part with two codes, a part in the last
    // block, a block with two parts, a code at the last index of a block.
    let parts = || rows.iter().flat_map(|q| &q.hi);
    assert!(
        rows.iter().all(|q| !q.hi.is_empty()),
        "a pattern did not split"
    );
    assert!(parts().any(|p| p.codes().len() == 2));
    assert!(parts().any(|p| p.b == in_f / 256 - 1));
    assert!(parts().any(|p| p.codes().iter().any(|c| c.0 == 255)));
    assert!(rows
        .iter()
        .any(|q| q.hi.windows(2).any(|w| w[0].b == w[1].b)));
    let mut checked = 0usize;
    for dt in Q8_DTYPES {
        for (k, scales) in HI_SCALES.iter().enumerate() {
            let w = crate::test_util::synth_weights(dt, ROWS, in_f, 0x61 + k as u32, scales);
            let bpr = w.len() / ROWS;
            for (r, q8) in rows.iter().enumerate() {
                for o in 0..ROWS {
                    let row = &w[o * bpr..(o + 1) * bpr];
                    let got = vec_dot_hi(dt, row, q8, in_f).to_bits();
                    for &t in &host {
                        let (want, _) =
                            with_tier(t, || vec_dot_hi_reference::<true>(dt, row, q8, in_f));
                        assert_eq!(
                            got,
                            want.to_bits(),
                            "{dt:?} scales {k} pattern {r} weight row {o} at {}: fast {} \
                             reference {want}",
                            t.name(),
                            f32::from_bits(got)
                        );
                    }
                    let want = vec_dot_hi_reference::<false>(dt, row, q8, in_f);
                    assert_eq!(got, want.to_bits(), "{dt:?} {k} {r} {o} uncapped");
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, Q8_DTYPES.len() * 2 * rows.len() * ROWS);
}

// ─── the f32 dots ──────────────────────────────────────────────────────────────────────────────

/// The tier the f32 dots' dispatchers take under a ceiling: their FMA code (an AVX2-tier path)
/// exists only on a build whose target lacks FMA, and runs only where the CPU has FMA and F16C.
fn float_expected(ceiling: SimdTier) -> Option<SimdTier> {
    #[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
    let fma_tier = is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c");
    #[cfg(not(all(target_arch = "x86_64", not(target_feature = "fma"))))]
    let fma_tier = false;
    (fma_tier && ceiling >= Avx2).then_some(Avx2)
}

/// `a·b + c` fused and unfused differ on this pair: `W·X = 1 + 2^-6 + 2^-18 + 2^-24` is an exact
/// halfway case at binary32, which rounds to even (dropping the `2^-24`) when the product is
/// rounded on its own. Added to `-1` the fused result keeps it: [`FUSED`] against [`UNFUSED`].
/// `W` is exact in f16 and bf16 (and as `65 · 2^-6` in Q8_0), so every dot can be fed it.
const W: f32 = 1.0 + 1.0 / 64.0;
const X: f32 = 1.0 + 1.0 / 262_144.0;
const FUSED: f32 = 1.0 / 64.0 + 1.0 / 262_144.0 + 1.0 / 16_777_216.0;
const UNFUSED: f32 = 1.0 / 64.0 + 1.0 / 262_144.0;

/// A float input stream: ordinary values, both zeros, subnormals, huge values that cancel
/// against their neighbours, and the values of the fused-or-not pairs ([`W`], [`X`], `-1`).
fn float_vec(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let k = i * 31 + seed * 7;
            let sign = if (k / 11).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            match k % 11 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::from_bits(1 + (k as u32 % 4096)) * sign,
                3 => 3.0e30 * sign,
                4 => W,
                5 => X,
                6 => -1.0,
                _ => ((k * 29 % 97) as f32 - 48.0) / 13.0,
            }
        })
        .collect()
}

/// The values compared bit for bit, every NaN folded to one (the C runtime's `fmaf` and the
/// instruction need only agree on numbers; no input here makes a NaN).
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter()
        .map(|x| if x.is_nan() { 0x7fc0_0000 } else { x.to_bits() })
        .collect()
}

/// Every f32 dot's results on one input set, through a `Dots` resolved in the `CAP` compilation.
fn float_dots<const CAP: bool>(n: usize, seed: usize) -> Vec<f32> {
    let a = float_vec(seed, n);
    let bs: [Vec<f32>; 4] = std::array::from_fn(|k| float_vec(seed + 1 + k, n));
    let br: [&[f32]; 4] = std::array::from_fn(|k| &bs[k][..]);
    let a2 = float_vec(seed + 9, n);
    let f16: Vec<u8> = (bs[0].iter())
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect();
    let bf16: Vec<u8> = (bs[1].iter())
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
    let mut out = vec![Dots::resolve::<CAP>().dot(&a, &bs[0])];
    out.extend(Dots::resolve::<CAP>().dot_n(&a, br));
    out.extend(
        Dots::resolve::<CAP>()
            .dot_mn([&a, &a2], br)
            .into_iter()
            .flatten(),
    );
    out.push(Dots::resolve::<CAP>().dot_f16(&f16, &a));
    out.push(Dots::resolve::<CAP>().dot_bf16(&bf16, &a));
    let mut o = a2.clone();
    Dots::resolve::<CAP>().axpy(&mut o, &bs[2], X);
    out.extend(o);
    let mut o = a2.clone();
    Dots::resolve::<CAP>().mul_add_into(&mut o, &bs[3], &a);
    out.extend(o);
    // The f16 KV write's narrowing, its bits carried as small f32 values.
    let mut h = vec![0u16; n];
    Dots::resolve::<CAP>().f16_from_f32(&mut h, &bs[3]);
    out.extend(h.iter().map(|&b| b as f32));
    out
}

/// Every weight row of a few formats through `dequant_dot_rows` (the f32-activation expert
/// GEMM's arm): affine formats it decodes itself, and one (IQ4_NL) that takes the `bytes_to_f32`
/// fallback.
fn dequant_dots<const CAP: bool>(in_f: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for dt in [
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Q8_0,
        DType::Q2K,
        DType::Q3K,
        DType::Q4_0,
        DType::Q5_1,
        DType::Iq4Nl,
    ] {
        if !in_f.is_multiple_of(block_spec(dt).block_elems) {
            continue;
        }
        let w = weights(dt, in_f, 0x5eed ^ in_f as u32);
        let x: Vec<f32> = (0..3).flat_map(|r| act_row(r, in_f)).collect();
        for row in w.chunks(w.len() / ROWS) {
            let mut col = [0f32; 3];
            let mut accs = Vec::new();
            crate::dequant_row::dequant_dot_rows(
                Dots::resolve::<CAP>(),
                row,
                dt,
                &x,
                in_f,
                &mut col,
                &mut accs,
            );
            out.extend(col);
        }
    }
    out
}

/// The f32 dots (`dot`, `dot_n`, `dot_mn`, `dot_f16`, `dot_bf16`, `axpy`, `dequant_dot_rows`)
/// give the scalar tier's bits at every tier this CPU has, through both compilations, on lengths
/// that leave every kind of remainder of 8 and on inputs with both zeros, subnormals,
/// cancellation and the fused-or-not values; and each dispatcher took the tier
/// [`float_expected`] says. On a baseline x86-64 build that holds the FMA code to the C runtime's
/// `fmaf`; on a build with FMA there is one code path, and the check is that no dispatcher grew a
/// tier.
#[test]
fn float_dots_give_the_scalar_bits_at_every_tier() {
    let host: Vec<SimdTier> = SimdTier::ALL
        .into_iter()
        .filter(|t| t.supported())
        .collect();
    eprintln!(
        "tier_tests: the f32 dots' FMA tier {}",
        if float_expected(SimdTier::best()).is_some() {
            "runs on this build"
        } else {
            "is not compiled or not supported here"
        }
    );
    let mut runs = 0;
    let lens = [0usize, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 128, 257];
    for &n in &lens {
        for seed in 0..3 {
            let (want, took) = with_tier(SimdTier::Scalar, || float_dots::<true>(n, seed));
            assert_eq!(took, None, "n {n}: a check passed at the scalar tier");
            for &t in &host {
                let (got, took) = with_tier(t, || float_dots::<true>(n, seed));
                assert_eq!(took, float_expected(t), "n {n} at {}", t.name());
                assert_eq!(bits(&got), bits(&want), "n {n} seed {seed} at {}", t.name());
                runs += 1;
            }
            let (got, took) = with_tier(SimdTier::Scalar, || float_dots::<false>(n, seed));
            assert_eq!(took, None, "the uncapped compilation asked the ceiling");
            assert_eq!(bits(&got), bits(&want), "n {n} seed {seed} uncapped");
        }
    }
    let widths = [32usize, 256, 512, 1056];
    for in_f in widths {
        let (want, _) = with_tier(SimdTier::Scalar, || dequant_dots::<true>(in_f));
        assert!(!want.is_empty(), "in_f {in_f}: no format ran");
        for &t in &host {
            let (got, took) = with_tier(t, || dequant_dots::<true>(in_f));
            assert_eq!(
                took,
                float_expected(t),
                "dequant in_f {in_f} at {}",
                t.name()
            );
            assert_eq!(
                bits(&got),
                bits(&want),
                "dequant in_f {in_f} at {}",
                t.name()
            );
            runs += 1;
        }
        let (got, _) = with_tier(SimdTier::Scalar, || dequant_dots::<false>(in_f));
        assert_eq!(bits(&got), bits(&want), "dequant in_f {in_f} uncapped");
    }
    assert_eq!(runs, host.len() * (lens.len() * 3 + widths.len()));
}

/// Every f32 dot is FUSED at every tier and in both compilations: on inputs where `a·b + c`
/// rounded twice gives [`UNFUSED`], each returns [`FUSED`]: inside a lane (element 8 after a `-1`
/// at element 0, n = 16), in the serial tail (n = 9: the lanes sum to `-1`, then element 8), in
/// `axpy`, and through `dequant_dot_rows` on Q8_0 codes `-64` and `65` at `d = 2^-6`. Turns red if
/// a tier, or the scalar code, rounds the product and the sum separately.
#[test]
fn float_dots_are_fused_at_every_tier() {
    assert_ne!(FUSED, UNFUSED);
    assert_eq!(
        W * X - 1.0,
        UNFUSED,
        "the pair is not a double-rounding case"
    );
    assert_eq!(W.mul_add(X, -1.0), FUSED);
    let host: Vec<SimdTier> = SimdTier::ALL
        .into_iter()
        .filter(|t| t.supported())
        .collect();
    let mut checked = 0;
    let mut check = |what: &str, f: &dyn Fn(bool) -> Vec<f32>| {
        for &t in &host {
            for cap in [true, false] {
                for got in with_tier(t, || f(cap)).0 {
                    assert_eq!(
                        got.to_bits(),
                        FUSED.to_bits(),
                        "{what} at {} (capped {cap}): {got:e}, want the fused {FUSED:e}",
                        t.name()
                    );
                    checked += 1;
                }
            }
        }
    };
    for n in [9usize, 16] {
        let (mut a, mut b) = (vec![0f32; n], vec![0f32; n]);
        (a[0], b[0], a[8], b[8]) = (-1.0, 1.0, W, X);
        let wf16: Vec<u8> = (a.iter())
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let wbf16: Vec<u8> = (a.iter())
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        check(&format!("dot n {n}"), &|cap| {
            vec![if cap {
                Dots::resolve::<true>().dot(&a, &b)
            } else {
                Dots::resolve::<false>().dot(&a, &b)
            }]
        });
        check(&format!("dot_n n {n}"), &|cap| {
            if cap {
                Dots::resolve::<true>().dot_n(&a, [&b, &b]).to_vec()
            } else {
                Dots::resolve::<false>().dot_n(&a, [&b, &b]).to_vec()
            }
        });
        check(&format!("dot_mn n {n}"), &|cap| {
            let r = if cap {
                Dots::resolve::<true>().dot_mn([&a, &a], [&b, &b, &b])
            } else {
                Dots::resolve::<false>().dot_mn([&a, &a], [&b, &b, &b])
            };
            r.into_iter().flatten().collect()
        });
        check(&format!("dot_f16 n {n}"), &|cap| {
            vec![if cap {
                Dots::resolve::<true>().dot_f16(&wf16, &b)
            } else {
                Dots::resolve::<false>().dot_f16(&wf16, &b)
            }]
        });
        check(&format!("dot_bf16 n {n}"), &|cap| {
            vec![if cap {
                Dots::resolve::<true>().dot_bf16(&wbf16, &b)
            } else {
                Dots::resolve::<false>().dot_bf16(&wbf16, &b)
            }]
        });
    }
    check("mul_add_into", &|cap| {
        let mut o = vec![-1f32; 19];
        let (v, e) = (vec![W; 19], vec![X; 19]);
        if cap {
            Dots::resolve::<true>().mul_add_into(&mut o, &v, &e);
        } else {
            Dots::resolve::<false>().mul_add_into(&mut o, &v, &e);
        }
        o
    });
    check("axpy", &|cap| {
        let mut o = vec![-1f32; 19];
        let v = vec![W; 19];
        if cap {
            Dots::resolve::<true>().axpy(&mut o, &v, X);
        } else {
            Dots::resolve::<false>().axpy(&mut o, &v, X);
        }
        o
    });
    let mut blk = half::f16::from_f32(1.0 / 64.0).to_le_bytes().to_vec();
    let mut codes = [0i8; 32];
    (codes[0], codes[8]) = (-64, 65);
    blk.extend(codes.iter().map(|&c| c as u8));
    let mut x = vec![0f32; 32];
    (x[0], x[8]) = (1.0, X);
    check("dequant_dot_rows Q8_0", &|cap| {
        let (mut col, mut accs) = ([0f32; 1], Vec::new());
        if cap {
            crate::dequant_row::dequant_dot_rows(
                Dots::resolve::<true>(),
                &blk,
                DType::Q8_0,
                &x,
                32,
                &mut col,
                &mut accs,
            );
        } else {
            crate::dequant_row::dequant_dot_rows(
                Dots::resolve::<false>(),
                &blk,
                DType::Q8_0,
                &x,
                32,
                &mut col,
                &mut accs,
            );
        }
        col.to_vec()
    });
    // 5 dots at 2 lengths (dot_n gives 2 values, dot_mn 6), mul_add_into's and axpy's 19, one
    // dequant: per tier and compilation.
    assert_eq!(
        checked,
        host.len() * 2 * (2 * (1 + 2 + 6 + 1 + 1) + 19 + 19 + 1)
    );
}

/// Median ns per element of each f32 dot, single thread, interleaved rounds: its scalar definition
/// (`*_scalar`, which on a build without FMA calls the C runtime's `fmaf` per element) against
/// its dispatcher (uncapped: the FMA tier where the build lacks FMA and the CPU has it).
/// Information for docs/math.md; run on a baseline build to see the FMA tier's gain, and on a
/// native one, where both columns are the same code.
#[test]
#[ignore = "benchmark: run in release on a quiet machine"]
fn bench_float_dots() {
    use std::hint::black_box;
    use std::time::Instant;
    const ROUNDS: usize = 15;
    const REPS: usize = 2000;
    let a = float_vec(1, 4096);
    let b: Vec<f32> = (0..4096)
        .map(|i| ((i * 29 % 97) as f32 - 48.0) / 13.0)
        .collect();
    let bs: [&[f32]; 4] = std::array::from_fn(|k| &b[k * 128..k * 128 + 128]);
    let wf16: Vec<u8> = (b.iter())
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect();
    let wbf16: Vec<u8> = (b.iter())
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
    let q4k = weights(DType::Q4K, 4096, 3);
    let q4k_row = &q4k[..q4k.len() / ROWS];
    type Case<'a> = (&'a str, usize, Box<dyn Fn(bool) -> f32 + 'a>);
    // Resolved once, as the interpreter does per graph.
    let dots = Dots::resolve::<false>();
    let src = &b[..];
    let cases: Vec<Case> = vec![
        (
            "f16_from_f32, 4096",
            4096,
            Box::new(|fast| {
                let mut h = [0u16; 4096];
                let x = black_box(src);
                if fast {
                    dots.f16_from_f32(&mut h, x)
                } else {
                    for (o, &v) in h.iter_mut().zip(x) {
                        *o = half::f16::from_f32(v).to_bits();
                    }
                }
                black_box(&h)[7] as f32
            }),
        ),
        (
            "dot, 128",
            128,
            Box::new(|d| {
                let (x, y) = (black_box(&a[..128]), black_box(&b[..128]));
                if d {
                    dots.dot(x, y)
                } else {
                    dot_scalar(x, y)
                }
            }),
        ),
        (
            "dot_n<4>, 128",
            512,
            Box::new(|d| {
                let x = black_box(&a[..128]);
                let r = if d {
                    dots.dot_n(x, bs)
                } else {
                    dot_n_scalar(x, bs)
                };
                r[0] + r[3]
            }),
        ),
        (
            "dot_f16, 4096",
            4096,
            Box::new(|d| {
                let (w, x) = (black_box(&wf16[..]), black_box(&a[..]));
                if d {
                    dots.dot_f16(w, x)
                } else {
                    dot_f16_scalar(w, x)
                }
            }),
        ),
        (
            "dot_bf16, 4096",
            4096,
            Box::new(|d| {
                let (w, x) = (black_box(&wbf16[..]), black_box(&a[..]));
                if d {
                    dots.dot_bf16(w, x)
                } else {
                    dot_bf16_scalar(w, x)
                }
            }),
        ),
        (
            "axpy, 128",
            128,
            Box::new(|d| {
                let mut o = [0f32; 128];
                let v = black_box(&b[..128]);
                if d {
                    dots.axpy(&mut o, v, 0.5)
                } else {
                    axpy_scalar(&mut o, v, 0.5)
                }
                black_box(&o)[7]
            }),
        ),
        (
            "dequant_dot_rows Q4_K, 4096",
            4096,
            Box::new(|d| {
                let (mut col, mut accs) = ([0f32; 1], Vec::new());
                let x = black_box(&a[..]);
                if d {
                    crate::dequant_row::dequant_dot_rows(
                        Dots::resolve::<false>(),
                        q4k_row,
                        DType::Q4K,
                        x,
                        4096,
                        &mut col,
                        &mut accs,
                    )
                } else {
                    crate::dequant_row::with_scalar_dot(
                        q4k_row,
                        DType::Q4K,
                        x,
                        4096,
                        &mut col,
                        &mut accs,
                    )
                }
                col[0]
            }),
        ),
    ];
    println!("ns per element, median of {ROUNDS} interleaved rounds: scalar | dispatch");
    for (name, elems, f) in &cases {
        let mut t = [Vec::new(), Vec::new()];
        for _ in 0..ROUNDS {
            for (k, d) in [false, true].into_iter().enumerate() {
                let s = Instant::now();
                let mut acc = 0u32;
                for _ in 0..REPS {
                    acc ^= black_box(f(d)).to_bits();
                }
                black_box(acc);
                t[k].push(s.elapsed().as_nanos() as f64 / (REPS * elems) as f64);
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        println!("{name:30} {:8.4} | {:8.4}", med(&mut t[0]), med(&mut t[1]));
    }
}

// ─── the f16 KV write ──────────────────────────────────────────────────────────────────────────

/// `Dots::f16_from_f32` against `half`'s software conversion (`from_f32_const`, which never uses
/// F16C) on the inputs `xs`, at every tier and uncapped, in slices of awkward lengths. Returns
/// `(compared, NaN inputs whose bits differ)` and panics on any non-NaN difference.
fn f16_store_check(xs: &[f32]) -> (usize, usize) {
    let want: Vec<u16> = xs
        .iter()
        .map(|&x| half::f16::from_f32_const(x).to_bits())
        .collect();
    let mut nan_diff = 0;
    let host: Vec<SimdTier> = SimdTier::ALL
        .into_iter()
        .filter(|t| t.supported())
        .collect();
    // Each tier this CPU has, capped, then `None`: the uncapped compilation.
    let mut runs: Vec<Option<SimdTier>> = host.into_iter().map(Some).collect();
    runs.push(None);
    let run = |t: Option<SimdTier>, d: &mut [u16], x: &[f32]| match t {
        Some(t) => {
            with_tier(t, || Dots::resolve::<true>().f16_from_f32(d, x));
        }
        None => Dots::resolve::<false>().f16_from_f32(d, x),
    };
    let mut got = vec![0u16; xs.len()];
    for &t in &runs {
        // Lengths 13 and 8 leave a tail every way the 8-wide loop can.
        let (mut at, mut k) = (0, 0);
        while at < xs.len() {
            let hi = (at + if k % 2 == 0 { 13 } else { 8 }).min(xs.len());
            run(t, &mut got[at..hi], &xs[at..hi]);
            (at, k) = (hi, k + 1);
        }
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            if g != w {
                assert!(
                    xs[i].is_nan(),
                    "f16 of {:#010x}: {g:#06x}, software {w:#06x}",
                    xs[i].to_bits()
                );
                nan_diff += 1;
            }
        }
    }
    (xs.len() * runs.len(), nan_diff)
}

/// The f16 KV write gives `half`'s software bits on every awkward input: both zeros, the f16
/// subnormal range and its rounding boundary, overflow to infinity and the largest finite value,
/// exact halfway cases, infinities, and NaNs of several payloads. Turns red if the wide path
/// rounds differently (another rounding mode, flushing subnormals) from the scalar one.
#[test]
fn f16_store_matches_half_on_awkward_values() {
    let mut xs: Vec<f32> = vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        65504.0,
        65519.99,
        65520.0,
        -65520.0,
    ];
    for e in -26i32..=16 {
        let b = 2f32.powi(e);
        // The value, and the two halfway points around it at f16's 11-bit precision.
        for m in [1.0f32, 1.0 + 1.0 / 2048.0, 1.0 + 3.0 / 2048.0, 1.5, 1.999] {
            xs.push(b * m);
            xs.push(-b * m);
        }
    }
    for bits in [
        0x7fc0_0000u32,
        0x7f80_0001,
        0x7fbf_ffff,
        0xffc0_1234,
        0x7fff_ffff,
        0x7f80_2000,
    ] {
        xs.push(f32::from_bits(bits));
    }
    let (n, nan_diff) = f16_store_check(&xs);
    eprintln!("f16 store: {n} conversions, {nan_diff} NaN results with other bits");
    assert_eq!(n, xs.len() * f16_store_runs());
    assert_eq!(
        nan_diff, 0,
        "NaN inputs narrowed to other bits than half's software path"
    );
}

/// How many paths [`f16_store_check`] runs: each tier this CPU has, and the uncapped one.
fn f16_store_runs() -> usize {
    SimdTier::ALL.into_iter().filter(|t| t.supported()).count() + 1
}

/// [`f16_store_matches_half_on_awkward_values`] over every one of the 2^32 f32 inputs, NaNs
/// included: every result must have the software conversion's bits, and every input must have
/// been converted on every path.
#[test]
#[ignore = "exhaustive: 2^32 inputs at every tier, minutes"]
fn f16_store_matches_half_on_every_f32() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (total, nans) = (AtomicUsize::new(0), AtomicUsize::new(0));
    const SPAN: u64 = 1 << 24;
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed) as u64;
                if k * SPAN >= 1 << 32 {
                    break;
                }
                let xs: Vec<f32> = (k * SPAN..(k + 1) * SPAN)
                    .map(|b| f32::from_bits(b as u32))
                    .collect();
                let (n, d) = f16_store_check(&xs);
                total.fetch_add(n, Ordering::Relaxed);
                nans.fetch_add(d, Ordering::Relaxed);
            });
        }
    });
    let (total, nans) = (total.into_inner(), nans.into_inner());
    eprintln!(
        "f16 store: {total} conversions over all 2^32 inputs, {nans} NaN results with other bits"
    );
    assert_eq!(total, (1usize << 32) * f16_store_runs());
    assert_eq!(
        nans, 0,
        "NaN inputs narrowed to other bits than half's software path"
    );
}
