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
//! report no tier at all. The f32/f16/bf16 dots (`dot`, `dot_n`, `dot_mn`, `dot_f16`,
//! `dot_bf16`) have no run-time dispatch; their one source is compiled for the build's target
//! and fixes the order of every `mul_add` and of the lane sum.

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
    let spec = block_spec(dt);
    let (d, min) = infr_testkit::synth_scales(dt);
    let nblk = ROWS * in_f / spec.block_elems;
    let mut w = Vec::with_capacity(nblk * spec.block_bytes);
    for b in 0..nblk {
        let mut blk =
            infr_testkit::lcg_bytes(seed ^ (b as u32).wrapping_mul(0x9e37), spec.block_bytes);
        let (fd, fm) = SCALES[(b * 5 + b / 7) % SCALES.len()];
        spec.write_scales(&mut blk, d * fd, min * fm);
        w.extend_from_slice(&blk);
    }
    w
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
