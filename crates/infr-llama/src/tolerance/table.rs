//! How far a GPU may be from the CPU reference, per path class.

use super::metrics::RunScore;

/// The kind of arithmetic a GPU configuration runs, which decides how far from the CPU reference
/// it can legitimately land.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathClass {
    /// Float activations end to end: f32 or f16 weights, or quantized weights with every
    /// int8-activation kernel off (`kernels.vulkan` `i8_dot`, `i8cm_kq`, `dp4a_kq`, `mmq`, `mrow`
    /// off and `mmv_mw = Some(false)`), and the f16 KV cache. What is left is the GPU's own float
    /// rounding: f16 activations into the coopmat GEMMs, f16 attention, reduction order.
    Float,
    /// Quantized weights through the int8-activation kernels (the production default): the int8
    /// cooperative-matrix and dp4a prefill GEMMs, the `mmq` GEMMs, the int8 multi-row and decode
    /// GEMVs. Each quantizes its activations per 32-element block, which the CPU reference does
    /// not, so the distance is dominated by that quantization.
    Int8,
    /// [`PathClass::Float`] with an 8-bit KV cache (`q8_0` K and V), scored against the CPU
    /// reference running the same cache format.
    KvQ8,
}

/// The largest distance a configuration of one [`PathClass`] may show against the CPU reference,
/// per metric, over one run's positions of one phase (prefill or decode).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    /// Largest KL(reference ‖ test) at any position.
    pub kl_max: f64,
    /// Largest mean KL over the run's positions.
    pub kl_mean: f64,
    /// Smallest top-k probability cosine at any position.
    pub topk_cos_min: f64,
    /// Largest centred logit deviation, relative to the reference row's range.
    pub logit_dev_max: f64,
    /// The reference top-2 margin (nats) above which top-1 must agree.
    pub near_tie: f64,
    /// Largest number of tokens the test may rank above the reference's top-1.
    pub rank_max: usize,
}

/// Fewest positions of a whole case (every prompt, both phases) whose reference margin clears
/// [`Tolerance::near_tie`]: a case where (almost) every position is a near tie would leave the
/// top-1 check vacuous.
pub const MIN_TOP1_SCORED: usize = 16;

/// One row of the tolerance table: a model (its GGUF file name, or a synthetic model's name in
/// the GPU tolerance tests) and a path class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub model: &'static str,
    pub class: PathClass,
    pub tol: Tolerance,
}

/// One [`TABLE`] row.
#[allow(clippy::too_many_arguments)]
const fn row(
    model: &'static str,
    class: PathClass,
    kl_max: f64,
    kl_mean: f64,
    topk_cos_min: f64,
    logit_dev_max: f64,
    near_tie: f64,
    rank_max: usize,
) -> Entry {
    Entry {
        model,
        class,
        tol: Tolerance {
            kl_max,
            kl_mean,
            topk_cos_min,
            logit_dev_max,
            near_tie,
            rank_max,
        },
    }
}

use PathClass::{Float, Int8, KvQ8};

/// The tolerance table. Rows are per model as well as per class because the same arithmetic
/// lands at very different distances on different models (gemma-3-1b is about a hundred times more
/// sensitive than Qwen3-0.6B in both classes), so one value per class would have to admit the most
/// sensitive model's noise everywhere.
///
/// **Provenance.** Measured 2026-11-18 by `measure_gpu_tolerance` (`tests/gpu_tolerance.rs`) on
/// branch `feat/gpu-tolerance` from `main` at `8c58ad4e`, Windows 11, AMD proprietary driver
/// (`driver_version=0x800161` in `infr tune show`) on both devices: `Vulkan0` RX 7900 XTX (RDNA3,
/// 96 CUs) and `Vulkan1` the Ryzen 9950X3D's integrated GPU (RDNA2, 2 CUs), each native and with
/// `INFR_FORCE_WAVE64`, tuning rows on and off (on `Vulkan1` a subset: `docs/gpu-tolerance.md`).
/// Three prompts of 12 to 185 tokens, 32 forced decode steps, prefill at the default chunk and at
/// 32 rows.
///
/// **How each value was set**, from the worst legitimate value over every device, configuration,
/// prompt and phase: KL max and mean times 2; the top-k cosine's distance from 1 times 2 (floored
/// at 0.01, where the class has no usable cosine at all); the logit deviation times 1.5; the rank
/// at twice the worst plus one; the near-tie margin at 1.5 times the widest legitimate top-1 miss,
/// and at least 0.25 nats (float) or 1 nat (int8, q8 KV). Each margin is about the spread between
/// the two devices for that row, and small enough that the planted bugs the doc lists still fail;
/// the doc gives both sides of each gap, and the rows where there is none.
pub const TABLE: &[Entry] = &[
    row(
        "ERNIE-4.5-0.3B-PT-F16.gguf",
        Float,
        1.9e-4,
        7.6e-6,
        0.9996,
        3.7e-3,
        0.25,
        1,
    ),
    row(
        "ERNIE-4.5-0.3B-PT-Q4_K_M.gguf",
        Float,
        1.2e-4,
        1.5e-5,
        0.9998,
        3.7e-3,
        0.25,
        1,
    ),
    row(
        "ERNIE-4.5-0.3B-PT-Q4_K_M.gguf",
        Int8,
        4.9e-2,
        6.8e-3,
        0.9399,
        5.7e-2,
        1.00,
        5,
    ),
    row(
        "gemma-3-1b-it-Q4_K_M.gguf",
        Float,
        6.4e-3,
        4.9e-4,
        0.9927,
        2.6e-2,
        0.25,
        3,
    ),
    row(
        "gemma-3-1b-it-Q4_K_M.gguf",
        Int8,
        1.8e0,
        3.2e-2,
        0.0100,
        1.9e-1,
        2.00,
        9,
    ),
    row(
        "LFM3-1.7B-F16.gguf",
        Float,
        4.2e-4,
        3.5e-5,
        0.9997,
        5.1e-3,
        0.25,
        1,
    ),
    row(
        "LFM3-1.7B-Q4_K_M.gguf",
        Float,
        6.9e-4,
        3.1e-5,
        0.9996,
        7.2e-3,
        0.25,
        1,
    ),
    row(
        "LFM3-1.7B-Q4_K_M.gguf",
        Int8,
        5.8e-2,
        8.9e-3,
        0.9518,
        1.4e-1,
        1.00,
        3,
    ),
    row(
        "Qwen3-0.6B-Q4_K_M.gguf",
        Float,
        8.5e-5,
        7.3e-6,
        0.9999,
        1.9e-3,
        0.25,
        1,
    ),
    row(
        "Qwen3-0.6B-Q4_K_M.gguf",
        Int8,
        7.0e-2,
        9.2e-3,
        0.9314,
        5.2e-2,
        1.00,
        5,
    ),
    row(
        "Qwen3-0.6B-Q4_K_M.gguf",
        KvQ8,
        3.9e-2,
        2.5e-3,
        0.9464,
        3.1e-2,
        1.00,
        3,
    ),
    row(
        "Qwen3.5-0.8B-Q4_K_M.gguf",
        Float,
        2.2e-5,
        4.2e-6,
        0.9999,
        2.1e-3,
        0.25,
        1,
    ),
    row(
        "Qwen3.5-0.8B-Q4_K_M.gguf",
        Int8,
        2.0e-2,
        2.0e-3,
        0.9831,
        5.4e-2,
        1.00,
        3,
    ),
    row(
        "synthetic-f16",
        Float,
        3.5e-3,
        2.1e-4,
        0.9963,
        1.6e-2,
        0.25,
        3,
    ),
    row(
        "synthetic-f32",
        Float,
        6.7e-4,
        5.9e-5,
        0.9992,
        3.9e-3,
        0.25,
        1,
    ),
];

/// The [`TABLE`] entry for `model` and `class`.
pub fn tolerance(model: &str, class: PathClass) -> &'static Tolerance {
    TABLE
        .iter()
        .find(|e| e.model == model && e.class == class)
        .map(|e| &e.tol)
        .unwrap_or_else(|| panic!("no tolerance for {model} {class:?}"))
}

impl Tolerance {
    /// Every metric of `s` that is outside this tolerance, described; empty when the run passes.
    /// `s` must have been scored with this tolerance's near-tie threshold.
    pub fn violations(&self, s: &RunScore) -> Vec<String> {
        assert_eq!(
            s.near_tie, self.near_tie,
            "scored at another near-tie threshold"
        );
        let mut v = Vec::new();
        if s.kl.worst.value > self.kl_max {
            v.push(format!(
                "KL {:.4e} at position {} over {:.4e}",
                s.kl.worst.value, s.kl.worst.pos, self.kl_max
            ));
        }
        if s.kl.mean() > self.kl_mean {
            v.push(format!(
                "mean KL {:.4e} over {:.4e}",
                s.kl.mean(),
                self.kl_mean
            ));
        }
        if s.topk_cos.worst.value < self.topk_cos_min {
            v.push(format!(
                "top-k cosine {:.6} at position {} under {:.6}",
                s.topk_cos.worst.value, s.topk_cos.worst.pos, self.topk_cos_min
            ));
        }
        if s.logit_dev.worst.value > self.logit_dev_max {
            v.push(format!(
                "logit deviation {:.4e} at position {} over {:.4e}",
                s.logit_dev.worst.value, s.logit_dev.worst.pos, self.logit_dev_max
            ));
        }
        if !s.top1_misses.is_empty() {
            v.push(format!(
                "top-1 differs at positions {:?}, each with a reference margin over {} nats",
                s.top1_misses, self.near_tie
            ));
        }
        if s.ref_top1_rank.worst.value > self.rank_max as f64 {
            v.push(format!(
                "the reference's top-1 ranked {} at position {} (at most {} above it allowed)",
                s.ref_top1_rank.worst.value, s.ref_top1_rank.worst.pos, self.rank_max
            ));
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: Tolerance = Tolerance {
        kl_max: 1e-3,
        kl_mean: 1e-4,
        topk_cos_min: 0.999,
        logit_dev_max: 1e-2,
        near_tie: 0.5,
        rank_max: 1,
    };

    /// A run of `n` positions of a 4-token row, the test row `t` against the reference `r`.
    fn run(r: [f32; 4], t: [f32; 4], n: usize) -> RunScore {
        let mut s = RunScore::new(TOL.near_tie);
        for pos in 0..n {
            s.add_rows(&r, &t, 4, &[pos]).unwrap();
        }
        s
    }

    const R: [f32; 4] = [4.0, 1.0, 0.0, -1.0];

    #[test]
    fn a_matching_run_passes() {
        assert!(TOL.violations(&run(R, R, 3)).is_empty());
    }

    /// Each limit is checked on its own: with every other limit open, a run past that one fails,
    /// naming it, and nothing else.
    #[test]
    fn each_limit_fails_alone() {
        let open = Tolerance {
            kl_max: 1e9,
            kl_mean: 1e9,
            topk_cos_min: 0.0,
            logit_dev_max: 1e9,
            near_tie: TOL.near_tie,
            rank_max: 9,
        };
        // A small distance that keeps the top-1, and a flip of it by more than the near tie.
        let near = [4.0, 1.1, 0.0, -1.0];
        let flip = [4.0, 4.8, 0.0, -1.0];
        let cases: [(&str, Tolerance, [f32; 4]); 6] = [
            (
                "KL",
                Tolerance {
                    kl_max: 0.0,
                    ..open
                },
                near,
            ),
            (
                "mean KL",
                Tolerance {
                    kl_mean: 0.0,
                    ..open
                },
                near,
            ),
            (
                "top-k cosine",
                Tolerance {
                    topk_cos_min: 1.0,
                    ..open
                },
                near,
            ),
            (
                "logit deviation",
                Tolerance {
                    logit_dev_max: 0.0,
                    ..open
                },
                near,
            ),
            (
                "ranked",
                Tolerance {
                    rank_max: 0,
                    near_tie: 1e9,
                    ..open
                },
                flip,
            ),
            ("top-1 differs", open, flip),
        ];
        assert!(open.violations(&run(R, near, 2)).is_empty());
        for (name, tol, t) in cases {
            let mut s = RunScore::new(tol.near_tie);
            for pos in 0..2 {
                s.add_rows(&R, &t, 4, &[pos]).unwrap();
            }
            let v = tol.violations(&s);
            assert!(v.len() == 1 && v[0].contains(name), "{name}: {v:?}");
        }
    }

    /// The table is well formed: one row per (model, class), every limit in its range.
    #[test]
    fn the_table_is_well_formed() {
        assert!(!TABLE.is_empty());
        for (i, e) in TABLE.iter().enumerate() {
            let t = &e.tol;
            assert!(
                t.kl_mean > 0.0 && t.kl_mean <= t.kl_max,
                "{} {:?}: mean above max",
                e.model,
                e.class
            );
            assert!(
                t.topk_cos_min > 0.0 && t.topk_cos_min < 1.0,
                "{} {:?}",
                e.model,
                e.class
            );
            assert!(
                t.logit_dev_max > 0.0 && t.near_tie > 0.0,
                "{} {:?}",
                e.model,
                e.class
            );
            assert!(
                TABLE[..i]
                    .iter()
                    .all(|o| (o.model, o.class) != (e.model, e.class)),
                "{} {:?} listed twice",
                e.model,
                e.class
            );
            assert_eq!(tolerance(e.model, e.class), t);
        }
    }
}
