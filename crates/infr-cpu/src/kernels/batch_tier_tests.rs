//! The batched K-quant and Q8_0 kernels give the one-row kernel's bits on every CPU tier pairing
//! a host can dispatch, not only the one the test host's own dispatchers pick.
//!
//! `Op::Linear` runs a one-row graph on `vec_dot_*` and a several-row one on `vec_dot_*_batch`
//! (and Q4_K's 2- and 8-row tiles), each dispatcher picking its own tier from the CPU's
//! features: scalar, AVX2, AVX-512BW, AVX-512 VNNI. A chunked prefill equals one row per graph
//! only if, on the HOST'S pairing, the two give the same bits. The op-level test
//! (`tests/linear_batch_identity.rs`) checks the pairing of whatever host runs it; this checks
//! each pairing an x86_64 host can land on by calling the tier kernels directly, so an AVX-512
//! box also covers what an AVX2-only or AVX-512-without-VNNI runner runs. aarch64 runs the
//! scalar pair, checked here as the `Scalar` host.
//!
//! The Q8x32-activation kernels (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4, NVFP4, Q2_0, sub-256
//! Q8_0) are absent: `Op::Linear` calls the same `*_batch` dispatcher at one row and at several,
//! so both land on one tier on any host. So are the grid-codebook and ternary formats, whose
//! one-row and batch kernels have no tiers.

use super::*;
use crate::quant::quantize_q8;
use infr_core::tensor::DType;

/// A one-row tier kernel.
type Dot = fn(&[u8], &Q8, usize) -> f32;
/// A batch tier kernel over weight rows of `bpr` bytes, writing `out[o * m + r]` (the layout
/// `Op::Linear` hands its batch kernels).
type Batch = fn(&[u8], usize, &[Q8], usize, &mut [f32]);

/// Which host feature set a pairing is the dispatch of.
#[derive(Clone, Copy, Debug)]
enum Host {
    Scalar,
    Avx2,
    /// AVX-512BW without VNNI.
    Avx512bw,
    Vnni,
}

impl Host {
    /// Whether this CPU can run the pairing's kernels.
    fn runs_here(self) -> bool {
        match self {
            Host::Scalar => true,
            Host::Avx2 => is_x86_feature_detected!("avx2"),
            Host::Avx512bw => is_x86_feature_detected!("avx512bw"),
            Host::Vnni => {
                is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni")
            }
        }
    }
}

/// A per-row batch kernel run over each weight row. SAFETY (the `unsafe` tier kernels): only
/// called when [`Host::runs_here`] says this CPU has the tier's features.
macro_rules! per_row {
    (safe $f:path) => {
        |w: &[u8], bpr: usize, q8s: &[Q8], in_f: usize, out: &mut [f32]| {
            let m = q8s.len();
            for (o, row) in w.chunks(bpr).enumerate() {
                $f(row, q8s, in_f, &mut out[o * m..(o + 1) * m])
            }
        }
    };
    ($f:path) => {
        |w: &[u8], bpr: usize, q8s: &[Q8], in_f: usize, out: &mut [f32]| {
            let m = q8s.len();
            for (o, row) in w.chunks(bpr).enumerate() {
                unsafe { $f(row, q8s, in_f, &mut out[o * m..(o + 1) * m]) }
            }
        }
    };
}

/// A one-row tier kernel; SAFETY as for [`per_row`].
macro_rules! dot {
    ($f:path) => {
        |row: &[u8], q8: &Q8, in_f: usize| unsafe { $f(row, q8, in_f) }
    };
}

/// One of Q4_K's 2-row tile kernels.
type Pair2 = unsafe fn(&[u8], &[u8], &[Q8], usize, &mut [f32], &mut [f32]);

/// Q4_K's 2-row tile over pairs of weight rows, as `Op::Linear` runs it when `out_f < 8` and on
/// the rows past the 8-row groups. The odd row goes to the AVX-512BW per-row kernel here (both
/// hosts with a 2-row tile have AVX-512BW); each host's own per-row kernel is a pairing of its own.
fn q4k_pairs(pair: Pair2, w: &[u8], bpr: usize, q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let rows: Vec<&[u8]> = w.chunks(bpr).collect();
    for (p, outs) in out.chunks_mut(2 * m).enumerate() {
        if outs.len() == m {
            // SAFETY: the caller runs this pairing only on an AVX-512BW host.
            unsafe { vec_dot_q4k_batch_avx512bw(rows[2 * p], q8s, in_f, outs) };
            continue;
        }
        let (a, b) = outs.split_at_mut(m);
        // SAFETY: as above; `pair` is an AVX-512BW or VNNI tile the caller checked.
        unsafe { pair(rows[2 * p], rows[2 * p + 1], q8s, in_f, a, b) };
    }
}

/// Q4_K's 8-row tile over whole groups of weight rows. It has no VNNI form, so both AVX-512 hosts
/// run it.
fn q4k_groups(w: &[u8], bpr: usize, q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let rows: Vec<&[u8]> = w.chunks(bpr).collect();
    for (g, outs) in out.chunks_exact_mut(8 * m).enumerate() {
        let mut it = outs.chunks_mut(m);
        let outs: [&mut [f32]; 8] = std::array::from_fn(|_| it.next().unwrap());
        let rows: [&[u8]; 8] = std::array::from_fn(|i| rows[8 * g + i]);
        // SAFETY: the caller runs this pairing only on an AVX-512BW host.
        unsafe { vec_dot_q4k_batch8_avx512bw(rows, q8s, in_f, outs) };
    }
}

/// One pairing: a dtype, the host whose dispatch it is, the batch form, and the two kernels.
type Pairing = (DType, Host, &'static str, Dot, Batch);

/// Every one-row/batch pairing a host dispatches, by dtype: what `vec_dot_X` and
/// `vec_dot_X_batch` (and Q4_K's tiles) select on each feature set.
fn pairings() -> Vec<Pairing> {
    use Host::*;
    vec![
        (
            DType::Q4K,
            Scalar,
            "batch",
            vec_dot_q4k_scalar,
            per_row!(safe vec_dot_q4k_batch_scalar),
        ),
        (
            DType::Q4K,
            Avx2,
            "batch",
            dot!(vec_dot_q4k_avx2),
            per_row!(vec_dot_q4k_batch_avx2),
        ),
        (
            DType::Q4K,
            Avx512bw,
            "batch",
            dot!(vec_dot_q4k_avx512bw),
            per_row!(vec_dot_q4k_batch_avx512bw),
        ),
        (
            DType::Q4K,
            Vnni,
            "batch",
            dot!(vec_dot_q4k_avx512bw),
            per_row!(vec_dot_q4k_batch_vnni),
        ),
        (
            DType::Q4K,
            Avx512bw,
            "batch2",
            dot!(vec_dot_q4k_avx512bw),
            |w, bpr, q8s, in_f, out| q4k_pairs(vec_dot_q4k_batch2_avx512bw, w, bpr, q8s, in_f, out),
        ),
        (
            DType::Q4K,
            Vnni,
            "batch2",
            dot!(vec_dot_q4k_avx512bw),
            |w, bpr, q8s, in_f, out| q4k_pairs(vec_dot_q4k_batch2_vnni, w, bpr, q8s, in_f, out),
        ),
        (
            DType::Q4K,
            Avx512bw,
            "batch8",
            dot!(vec_dot_q4k_avx512bw),
            q4k_groups,
        ),
        (
            DType::Q5K,
            Scalar,
            "batch",
            vec_dot_q5k_scalar,
            per_row!(safe vec_dot_q5k_batch_scalar),
        ),
        (
            DType::Q5K,
            Avx2,
            "batch",
            dot!(vec_dot_q5k_avx2),
            per_row!(vec_dot_q5k_batch_avx2),
        ),
        // No VNNI form: both AVX-512 hosts run this one.
        (
            DType::Q5K,
            Avx512bw,
            "batch",
            dot!(vec_dot_q5k_avx512bw),
            per_row!(vec_dot_q5k_batch_avx512bw),
        ),
        (
            DType::Q6K,
            Scalar,
            "batch",
            vec_dot_q6k_scalar,
            per_row!(safe vec_dot_q6k_batch_scalar),
        ),
        (
            DType::Q6K,
            Avx2,
            "batch",
            dot!(vec_dot_q6k_avx2),
            per_row!(vec_dot_q6k_batch_avx2),
        ),
        // No AVX-512BW batch form: without VNNI the batch drops to AVX2.
        (
            DType::Q6K,
            Avx512bw,
            "batch",
            dot!(vec_dot_q6k_avx512bw),
            per_row!(vec_dot_q6k_batch_avx2),
        ),
        (
            DType::Q6K,
            Vnni,
            "batch",
            dot!(vec_dot_q6k_avx512bw),
            per_row!(vec_dot_q6k_batch_vnni),
        ),
        (
            DType::Q2K,
            Scalar,
            "batch",
            vec_dot_q2k_scalar,
            per_row!(safe vec_dot_q2k_batch_scalar),
        ),
        (
            DType::Q2K,
            Avx2,
            "batch",
            dot!(vec_dot_q2k_avx2),
            per_row!(vec_dot_q2k_batch_avx2),
        ),
        (
            DType::Q2K,
            Avx512bw,
            "batch",
            dot!(vec_dot_q2k_avx512bw),
            per_row!(vec_dot_q2k_batch_avx512bw),
        ),
        (
            DType::Q2K,
            Vnni,
            "batch",
            dot!(vec_dot_q2k_avx512bw),
            per_row!(vec_dot_q2k_batch_vnni),
        ),
        (
            DType::Q3K,
            Scalar,
            "batch",
            vec_dot_q3k_scalar,
            per_row!(safe vec_dot_q3k_batch_scalar),
        ),
        (
            DType::Q3K,
            Avx2,
            "batch",
            dot!(vec_dot_q3k_avx2),
            per_row!(vec_dot_q3k_batch_avx2),
        ),
        (
            DType::Q3K,
            Avx512bw,
            "batch",
            dot!(vec_dot_q3k_avx512bw),
            per_row!(vec_dot_q3k_batch_avx512bw),
        ),
        (
            DType::Q3K,
            Vnni,
            "batch",
            dot!(vec_dot_q3k_avx512bw),
            per_row!(vec_dot_q3k_batch_vnni),
        ),
        (
            DType::Q8_0,
            Scalar,
            "batch",
            vec_dot_q8_0_scalar,
            per_row!(safe vec_dot_q8_0_batch_scalar),
        ),
        (
            DType::Q8_0,
            Avx2,
            "batch",
            dot!(vec_dot_q8_0_avx2),
            per_row!(vec_dot_q8_0_batch_avx2),
        ),
        // No VNNI form: both AVX-512 hosts run this one.
        (
            DType::Q8_0,
            Avx512bw,
            "batch",
            dot!(vec_dot_q8_0_avx512bw),
            per_row!(vec_dot_q8_0_batch_avx512bw),
        ),
        (
            DType::Iq4Xs,
            Scalar,
            "batch",
            vec_dot_iq4xs_scalar,
            per_row!(safe vec_dot_iq4xs_batch_scalar),
        ),
        (
            DType::Iq4Xs,
            Avx2,
            "batch",
            dot!(vec_dot_iq4xs_avx2),
            per_row!(vec_dot_iq4xs_batch_avx2),
        ),
        // The batch kernel tops out at AVX2: every AVX-512 host pairs it with this one.
        (
            DType::Iq4Xs,
            Avx512bw,
            "batch",
            dot!(vec_dot_iq4xs_avx512bw),
            per_row!(vec_dot_iq4xs_batch_avx2),
        ),
    ]
}

/// Activation row `r` of `in_f`: ordinary values, a zero row, one large value, all negative,
/// exact int8 rounding ties, subnormals.
fn act_row(r: usize, in_f: usize, x: &[f32]) -> Vec<f32> {
    (0..in_f)
        .map(|i| match r % 6 {
            0 => x[i],
            1 => 0.0,
            2 if i == (r * 11) % in_f => 1.0e4,
            2 => x[i] * 1e-2,
            3 => -x[i].abs() - 1e-2,
            // A block maximum of 127/8 makes the scale 1/8 exactly, so the rest sit on `k + ½`.
            4 if i % 32 == 0 => 127.0 / 8.0,
            4 => (((i * 37 + r) % 253) as f32 - 126.5) / 8.0,
            _ => x[i] * 1e-39,
        })
        .collect()
}

/// Each pairing's batch kernel over 9 weight rows (one 8-row tile and a tail) and 13 activation
/// rows equals the one-row kernel on every (weight row, activation row), bit for bit, at two and
/// three super-blocks. Pairings this CPU cannot run are skipped; the scalar ones run everywhere.
#[test]
fn batch_kernels_give_the_one_row_bits_on_every_host_pairing() {
    let (rows, m) = (9usize, 13usize);
    let mut ran = 0;
    for (dt, host, form, one, batch) in pairings() {
        if !host.runs_here() {
            continue;
        }
        for in_f in [512usize, 768] {
            let w = infr_testkit::synth_weight(dt, rows * in_f, 0x7e5 + in_f as u32);
            let bpr = w.len() / rows;
            let base: Vec<f32> = (0..in_f)
                .map(|i| ((i * 29 % 97) as f32 - 48.0) / 13.0)
                .collect();
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&act_row(r, in_f, &base)))
                .collect();
            let mut out = vec![f32::NAN; rows * m];
            // The 8-row tile covers whole groups only; `Op::Linear` hands the tail elsewhere.
            let covered = if form == "batch8" { 8 } else { rows };
            batch(
                &w[..covered * bpr],
                bpr,
                &q8s,
                in_f,
                &mut out[..covered * m],
            );
            for o in 0..covered {
                for (r, q8) in q8s.iter().enumerate() {
                    let want = one(&w[o * bpr..(o + 1) * bpr], q8, in_f);
                    let got = out[o * m + r];
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{dt:?} {form} on {host:?}, in_f {in_f}, weight row {o}, activation \
                         row {r}: batch {got:e}, one row {want:e}"
                    );
                }
            }
            ran += matches!(host, Host::Scalar) as usize;
        }
    }
    // Every scalar pairing ran at both widths: the loop cannot have matched nothing.
    let scalar = pairings()
        .iter()
        .filter(|p| matches!(p.1, Host::Scalar))
        .count();
    assert_eq!(ran, 2 * scalar, "a scalar pairing did not run");
}
