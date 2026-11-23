//! The int8 kernels' speed at each SIMD tier this CPU has, side by side in one process: an
//! `#[ignore]`d test that prints a table, run by CI on the macOS (Apple Silicon) leg to measure
//! the NEON kernels against the scalar ones, and by hand anywhere:
//!
//! ```text
//! cargo test --release -p infr-cpu --lib kernels::bench -- --ignored --nocapture
//! ```
//!
//! Every tier runs the same dispatcher under a per-thread ceiling ([`with_tier`]), the rounds
//! interleave the tiers so drift lands on all of them, and the table reports each one's median.
//! On a shared runner the absolute figures move from run to run; the ratio to scalar, taken on
//! one machine within one minute, is the number to read. The test asserts nothing about speed:
//! the bits are `tier_tests`' business.
//!
//! Every output line starts with [`PREFIX`], so the table can be pulled from a CI log with
//! `grep`. A batch kernel's `ns/row` is per weight row and activation row pair.

// Test-only (`kernels.rs` declares it under `cfg(test)`); the attribute also tells
// `infr-core/tests/no_bare_print.rs`, which scans a file down to its first one.
#[cfg(test)]
use super::*;
use crate::quant::quantize_q8;
use crate::simd::{with_tier, SimdTier};
use infr_core::tensor::DType;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// The start of every line this bench prints.
const PREFIX: &str = "kbench:";

/// Row width: a small model's hidden size.
const IN_F: usize = 2048;

/// Weight rows per pass.
const ROWS: usize = 256;

/// Activation rows a batch kernel takes per pass: a prefill chunk.
const BATCH: usize = 16;

/// Interleaved rounds; the table reports the median.
const ROUNDS: usize = 9;

/// The least time one sample takes: passes repeat until a scalar pass reaches it.
const SAMPLE: Duration = Duration::from_millis(10);

/// One timed kernel: a name, its weight format, and whether it is a batch kernel.
type Case = (&'static str, DType, bool);

const CASES: [Case; 6] = [
    ("q4k", DType::Q4K, false),
    ("q6k", DType::Q6K, false),
    ("q8_0", DType::Q8_0, false),
    ("q4k_batch", DType::Q4K, true),
    ("q6k_batch", DType::Q6K, true),
    ("q8_0_batch", DType::Q8_0, true),
];

/// One pass of `case` over every weight row, through the dispatcher's capped compilation.
fn pass(case: Case, w: &[u8], bpr: usize, q8s: &[Q8], out: &mut [f32]) {
    let (_, dt, batch) = case;
    for (o, row) in w.chunks(bpr).enumerate() {
        if batch {
            let col = &mut out[o * BATCH..o * BATCH + BATCH];
            match dt {
                DType::Q4K => vec_dot_q4k_batch::<true>(row, q8s, IN_F, col),
                DType::Q6K => vec_dot_q6k_batch::<true>(row, q8s, IN_F, col),
                _ => vec_dot_q8_0_batch::<true>(row, q8s, IN_F, col),
            }
        } else {
            out[o] = vec_dot_q8_any::<true>(dt, row, &q8s[0], IN_F);
        }
    }
    black_box(out);
}

#[test]
#[ignore = "a benchmark: run in release, by hand or in CI's macOS timing step"]
fn kernel_speed_by_tier() {
    let tiers: Vec<SimdTier> = SimdTier::ALL
        .into_iter()
        .filter(|t| t.supported())
        .collect();
    // libtest prints `test <name> ... ` without a newline when output is not captured: end
    // that line first, so the header starts one of its own and `grep '^kbench:'` finds it.
    println!();
    println!(
        "{PREFIX} in_f {IN_F}, {ROWS} weight rows, batch {BATCH} activations, median of \
         {ROUNDS} interleaved rounds; tiers {:?}",
        tiers.iter().map(|t| t.name()).collect::<Vec<_>>()
    );
    println!(
        "{PREFIX} {:<11} {:<10} {:>12} {:>12} {:>9}",
        "kernel", "tier", "ns/row", "ns/elem", "x scalar"
    );
    let x: Vec<Vec<f32>> = (0..BATCH)
        .map(|r| {
            (0..IN_F)
                .map(|i| ((i * 29 + r * 7) % 97) as f32 / 13.0 - 3.6875)
                .collect()
        })
        .collect();
    let q8s: Vec<Q8> = x.iter().map(|r| quantize_q8(r)).collect();
    for case in CASES {
        let (name, dt, batch) = case;
        let w = crate::test_util::synth_weights(dt, ROWS, IN_F, 0xbe7c, &[(1.0, 1.0)]);
        let bpr = w.len() / ROWS;
        let acts = if batch { BATCH } else { 1 };
        let mut out = vec![0f32; ROWS * acts];
        // Repeat passes until the scalar tier's sample takes `SAMPLE`; the faster tiers then
        // take less, at the same count.
        let reps = {
            let t0 = Instant::now();
            let mut n = 0u32;
            with_tier(SimdTier::Scalar, || {
                while t0.elapsed() < SAMPLE {
                    pass(case, &w, bpr, &q8s, &mut out);
                    n += 1;
                }
            });
            n
        };
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ROUNDS); tiers.len()];
        for _ in 0..ROUNDS {
            for (k, &t) in tiers.iter().enumerate() {
                let t0 = Instant::now();
                with_tier(t, || {
                    for _ in 0..reps {
                        pass(case, &w, bpr, &q8s, &mut out);
                    }
                });
                samples[k].push(t0.elapsed().as_secs_f64() * 1e9 / reps as f64);
            }
        }
        let medians: Vec<f64> = samples
            .iter_mut()
            .map(|s| {
                s.sort_by(f64::total_cmp);
                s[s.len() / 2]
            })
            .collect();
        for (k, &t) in tiers.iter().enumerate() {
            let per_row = medians[k] / (ROWS * acts) as f64;
            println!(
                "{PREFIX} {name:<11} {:<10} {per_row:>12.1} {:>12.4} {:>9.2}",
                t.name(),
                per_row / IN_F as f64,
                medians[0] / medians[k]
            );
        }
    }
}
