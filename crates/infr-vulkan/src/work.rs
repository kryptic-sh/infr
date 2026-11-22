//! The work bound on one dispatch and one submit (plan 12, S2).
//!
//! A driver watchdog drops a GPU's work when one submit runs too long: on AMD's Windows driver
//! an engine reset (`LiveKernelEvent` 0x141), on Linux amdgpu's job timeout. A bound on
//! dispatches per submit (`VulkanShared::submit_dispatch_cap`) cannot see a dispatch that is
//! heavy on its own, which a slow device meets at a large batch: a prefill GEMV over 512 rows on
//! the 2-CU integrated GPU here is one 1.5 s dispatch. So every heavy prefill GEMM and GEMV, and
//! the non-coopmat flash prefill attention, says what it costs before it is recorded
//! (`Recorder::work_next`), and the recorder
//!
//! - splits a dispatch whose estimate passes the budget into several dispatches over ranges of
//!   its workgroup grid (`vkCmdDispatchBase`), each in a submit of its own and each running the
//!   same shader on the same inputs as in the whole grid, so the output is bit-identical;
//! - closes the submit once the work recorded into it reaches the budget
//!   (`crate::SegmentCut::Work`);
//! - refuses, with an error before anything is submitted, a dispatch it cannot split whose
//!   MEASURED cost is far past the budget on a device prone to resets.
//!
//! **The estimate** is the op's FLOPs times a time per FLOP: the one measured for that kernel in
//! this process when there is one, else a prior, `submit.prior_gflops` (a tuning parameter, scaled
//! by compute units) times a speed-up for the kernel's [`Family`]. A dispatch whose estimate is a
//! noticeable part of the budget is bracketed by two timestamps, and its measured time per FLOP
//! replaces the prior for that kernel (the first sample) or raises it (any later, slower one): a
//! kernel's estimate can go down once, on evidence, and only up after that. Dispatches with no
//! estimate (every other kernel) count as free, as do single-row ones: a decode step is never
//! split, timed or cut by this, and the dispatch cap is what bounds a submit of those.

use std::collections::HashMap;
use std::sync::Mutex;

/// `submit.budget_ms`'s default for a device a watchdog is known to reset under long work
/// (`tuning::DeviceFacts::reset_prone`), or any other.
///
/// The watchdogs measured so far reset a submit after about 2 s (Windows' default `TdrDelay`;
/// amdgpu's default `lockup_timeout` on the RX 580 of issue 94), and Vulkan reports neither limit,
/// so the defaults assume it. A reset-prone device gets the submit splitter's own target
/// ([`infr_core::SUBMIT_BUDGET_NS`]), an eighth of that: its estimate is a prior until a kernel
/// has been timed, and another program on the GPU stretches a submit. Any other device gets the
/// time at which the splitter already starts cutting its forwards
/// ([`infr_core::SUBMIT_DANGER_NS`]), half the limit: its estimates stay the pessimistic prior
/// for all but its heaviest dispatches, and a tighter budget would cut forwards that measure
/// far under it.
pub(crate) fn default_budget_ms(reset_prone: bool) -> u32 {
    let ns = if reset_prone {
        infr_core::SUBMIT_BUDGET_NS
    } else {
        infr_core::SUBMIT_DANGER_NS
    };
    (ns / 1_000_000) as u32
}

/// `submit.prior_gflops` per compute unit. Measured on the 2-CU integrated GPU (RDNA2), the slowest
/// family, the per-row GEMV, ran at 19 to 30 GFLOP/s depending on the weight's shape (9 to 15 per
/// unit); 8 keeps the prior under the slowest of those.
const PRIOR_GFLOPS_PER_CU: u32 = 8;

/// The compute units assumed when the device reports none: the weakest part measured.
const PRIOR_UNKNOWN_CU: u32 = 2;

/// `submit.prior_gflops`'s derived default for a device of `compute_units` (0 = unknown).
pub(crate) fn prior_gflops(compute_units: u32) -> u32 {
    let cu = if compute_units == 0 {
        PRIOR_UNKNOWN_CU
    } else {
        compute_units
    };
    PRIOR_GFLOPS_PER_CU.saturating_mul(cu)
}

/// A dispatch whose estimate is at least the budget divided by this is timed (two timestamps)
/// so its kernel's estimate follows measurement.
const WATCH_FRACTION: u64 = 32;

/// A dispatch that cannot be split is refused only when its measured estimate passes the budget
/// by this factor: with the default budget, 2 s, the watchdog's own limit.
const REFUSE_FACTOR: u64 = 8;

/// How a kernel family's throughput compares with the prior, the slowest family's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Family {
    /// A GEMV run over several rows: it streams the weight once per row (the prefill fallback
    /// without int8 or matrix kernels).
    Gemv,
    /// A tiled float GEMM without matrix units (`native_gemm_fma`).
    FloatTile,
    /// An int8-dot GEMM (`native_gemm_dp4a_*`, `native_gemm_mmq_*`).
    Int8,
    /// A cooperative-matrix GEMM (f16 or int8).
    Matrix,
}

impl Family {
    /// The family's prior throughput as a multiple of `submit.prior_gflops`. Measured on the
    /// integrated GPU: the dp4a GEMM ran about 23 times as fast as the per-row GEMV (690 against
    /// 30 GFLOP/s); 8 keeps that pessimistic. An int8 dot the driver only emulates gets no
    /// speed-up. Matrix units get 16, a fraction of what the RX 7900 XTX's run.
    fn speedup(self, dot_accelerated: bool) -> f64 {
        match self {
            Family::Gemv | Family::FloatTile => 1.0,
            Family::Int8 if dot_accelerated => 8.0,
            Family::Int8 => 1.0,
            Family::Matrix => 16.0,
        }
    }
}

/// How a dispatch's grid may be cut into ranges, stated by the op that records it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Split {
    /// Not at all: the workgroups depend on each other or on the grid's size.
    No,
    /// The shader reads only `gl_WorkGroupID`, so any contiguous range of the highest non-unit
    /// dimension runs exactly as in the whole grid. Checked against the SPIR-V
    /// (`ComputeKernel::reads_num_workgroups`) before a split.
    Flat,
    /// `Recorder::dispatch_wide`'s convention: the shader recovers its flat index as
    /// `gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x`, so a range of whole `y` rows
    /// (every `x`, `gl_NumWorkGroups.x` unchanged) or, on a one-row grid, of `x` runs exactly as
    /// in the whole grid.
    WideGrid,
}

/// What an op says about its next dispatch (`Recorder::work_next`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WorkTag {
    pub family: Family,
    pub flops: u64,
    pub split: Split,
}

/// The FLOPs of a GEMM or GEMV over `rows` rows of a `in_f` x `out_f` weight.
pub(crate) fn gemm_flops(rows: usize, in_f: usize, out_f: usize) -> u64 {
    2 * rows as u64 * in_f as u64 * out_f as u64
}

/// One dispatch's estimated GPU time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Estimate {
    pub ns: u64,
    /// From a measurement of this kernel in this process, not the prior.
    pub measured: bool,
}

/// The per-backend estimator: the budget, the priors, and each kernel's measured time per FLOP.
pub(crate) struct WorkModel {
    /// `0` = no work bound.
    budget_ns: u64,
    /// The slowest family's prior time per FLOP.
    prior_ns_per_flop: f64,
    dot_accelerated: bool,
    /// A device a watchdog is known to reset under long work (integrated, or in the slow class):
    /// the only kind on which a dispatch is ever refused.
    reset_prone: bool,
    /// Measured nanoseconds per FLOP, by kernel name.
    measured: Mutex<HashMap<&'static str, f64>>,
}

impl WorkModel {
    pub(crate) fn new(
        budget_ms: u32,
        prior_gflops: u32,
        dot_accelerated: bool,
        reset_prone: bool,
    ) -> Self {
        WorkModel {
            budget_ns: u64::from(budget_ms) * 1_000_000,
            prior_ns_per_flop: 1.0 / f64::from(prior_gflops.max(1)),
            dot_accelerated,
            reset_prone,
            measured: Mutex::new(HashMap::new()),
        }
    }

    /// Whether the work bound is on (`submit.budget_ms` > 0).
    pub(crate) fn enabled(&self) -> bool {
        self.budget_ns > 0
    }

    pub(crate) fn budget_ns(&self) -> u64 {
        self.budget_ns
    }

    /// The estimated GPU time of `flops` of `kernel`, a `family` kernel.
    pub(crate) fn estimate(&self, kernel: &'static str, family: Family, flops: u64) -> Estimate {
        let measured = self
            .measured
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(kernel)
            .copied();
        let (rate, measured) = match measured {
            Some(r) => (r, true),
            None => (
                self.prior_ns_per_flop / family.speedup(self.dot_accelerated),
                false,
            ),
        };
        Estimate {
            ns: (flops as f64 * rate).ceil() as u64,
            measured,
        }
    }

    /// Feed one timed dispatch back: `flops` of `kernel` took `ns`. The first sample of a kernel
    /// replaces its prior; a later one only ever raises it.
    pub(crate) fn record(&self, kernel: &'static str, flops: u64, ns: u64) {
        if flops == 0 {
            return;
        }
        let rate = ns as f64 / flops as f64;
        let mut m = self.measured.lock().unwrap_or_else(|e| e.into_inner());
        let e = m.entry(kernel).or_insert(rate);
        if rate > *e {
            *e = rate;
        }
    }

    /// Every kernel measured so far with its throughput in GFLOP/s, sorted by name.
    pub(crate) fn measured_gflops(&self) -> Vec<(&'static str, f64)> {
        let m = self.measured.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<_> = m
            .iter()
            .map(|(&k, &ns_per_flop)| (k, 1.0 / ns_per_flop))
            .collect();
        v.sort_unstable_by_key(|&(k, _)| k);
        v
    }

    /// Into how many ranges a dispatch estimated at `est_ns` is cut so each stays within the
    /// budget, given `extent` workgroups along the dimension that is cut. 1 = whole.
    pub(crate) fn pieces(&self, est_ns: u64, extent: u32) -> u32 {
        if self.budget_ns == 0 || est_ns <= self.budget_ns {
            return 1;
        }
        let want = est_ns.div_ceil(self.budget_ns);
        want.min(u64::from(extent.max(1))) as u32
    }

    /// Whether a dispatch estimated at `est_ns` is timed so its kernel's estimate follows
    /// measurement.
    pub(crate) fn watch(&self, est_ns: u64) -> bool {
        self.budget_ns > 0 && est_ns >= self.budget_ns / WATCH_FRACTION
    }

    /// Whether a dispatch that cannot be split is refused before submit: only on a reset-prone
    /// device, only on a measured estimate, and only far past the budget. Anywhere else it is
    /// submitted as it always was.
    pub(crate) fn refuse(&self, est: Estimate) -> bool {
        self.budget_ns > 0
            && self.reset_prone
            && est.measured
            && est.ns > self.budget_ns.saturating_mul(REFUSE_FACTOR)
    }
}

/// The ranges of `extent` workgroups a dispatch cut into `pieces` runs: `(base, count)` pairs,
/// contiguous, covering every workgroup once, sizes differing by at most one.
pub(crate) fn ranges(extent: u32, pieces: u32) -> impl Iterator<Item = (u32, u32)> {
    let pieces = pieces.clamp(1, extent.max(1));
    let (q, r) = (extent / pieces, extent % pieces);
    (0..pieces).scan(0u32, move |base, i| {
        let n = q + u32::from(i < r);
        let out = (*base, n);
        *base += n;
        Some(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;

    /// The integrated GPU here (2 CUs): its per-row GEMV over the SmolLM3 gate/up weight at 512
    /// rows measured about 1.57 s, the reproduction's heaviest dispatch.
    fn igpu() -> WorkModel {
        WorkModel::new(default_budget_ms(true), prior_gflops(2), true, true)
    }

    #[test]
    fn the_prior_follows_compute_units_and_assumes_the_weakest_part_when_unknown() {
        assert_eq!(prior_gflops(2), 16);
        assert_eq!(prior_gflops(96), 768);
        assert_eq!(prior_gflops(0), prior_gflops(PRIOR_UNKNOWN_CU));
        assert_eq!(default_budget_ms(true), 250);
        assert_eq!(default_budget_ms(false), 1000);
    }

    /// The first forward on the integrated GPU, before anything is measured: the prior must not
    /// be optimistic about the dispatch that was reset, so it is split into ranges of at most
    /// the budget by the prior's own reckoning, which is slower than the real kernel.
    #[test]
    fn the_first_forward_prior_is_conservative_on_the_slow_device() {
        let m = igpu();
        let flops = gemm_flops(512, 2048, 22016);
        let est = m.estimate("gemv", Family::Gemv, flops);
        assert!(!est.measured);
        let measured_ns = 1_570 * MS;
        assert!(
            est.ns >= measured_ns,
            "prior {} ms under the measured {} ms",
            est.ns / MS,
            measured_ns / MS
        );
        let pieces = m.pieces(est.ns, 172);
        assert!(
            measured_ns / u64::from(pieces) <= m.budget_ns(),
            "{pieces} ranges leave a real range over the budget"
        );
        // The old rule, a bound on dispatches only, never cut a single dispatch.
        assert!(pieces > 1);
    }

    /// The RX 7900 XTX's device facts.
    fn fast() -> WorkModel {
        WorkModel::new(default_budget_ms(false), prior_gflops(96), true, false)
    }

    /// On the RX 7900 XTX, by the prior alone, the default-batch GEMMs of the models run there
    /// stay whole and unwatched: nothing changes on a fast device.
    #[test]
    fn a_fast_device_at_default_batch_is_left_alone() {
        let m = fast();
        // Qwen3-1.7B gate/up at 512 rows on the coopmat GEMM, and Qwen3-14B's at 1024 rows.
        for flops in [gemm_flops(512, 2048, 12288), gemm_flops(1024, 5120, 34816)] {
            let est = m.estimate("native_gemm_warp_q4k", Family::Matrix, flops);
            assert_eq!(m.pieces(est.ns, 1 << 16), 1);
            assert!(est.ns < m.budget_ns() / 4, "{} ms", est.ns / MS);
            assert!(!m.watch(est.ns));
        }
    }

    /// A whole prefill forward of a 3B model at the default 1024-row batch (SmolLM3-3B's shapes:
    /// 36 layers, width 2048, feed-forward 11008, 4 of 16 heads' worth of K and V) stays under
    /// the fast device's budget by the prior alone, so no submit is closed by work. Under the
    /// reset-prone budget the same estimate would cut it.
    #[test]
    fn a_fast_device_forward_at_default_batch_takes_no_work_cut() {
        let m = fast();
        let rows = 1024;
        let layer: u64 = [
            (2048, 2048),      // q
            (2048, 512),       // k
            (2048, 512),       // v
            (2048, 2048),      // o
            (2048, 2 * 11008), // gate and up, fused
            (11008, 2048),     // down
        ]
        .iter()
        .map(|&(i, o)| m.estimate("g", Family::Matrix, gemm_flops(rows, i, o)).ns)
        .sum();
        let forward = 36 * layer;
        assert!(
            crate::segment_cut(1, 0, forward, m.budget_ns(), 0).is_none(),
            "{} ms of estimated work",
            forward / MS
        );
        let prone = u64::from(default_budget_ms(true)) * MS;
        assert_eq!(
            crate::segment_cut(1, 0, forward, prone, 0),
            Some(crate::SegmentCut::Work)
        );
    }

    #[test]
    fn a_measurement_replaces_the_prior_then_only_raises_it() {
        let m = igpu();
        let flops = gemm_flops(512, 2048, 22016);
        let prior = m.estimate("dp4a", Family::Int8, flops).ns;
        // Measured 66 ms: the estimate drops to it.
        m.record("dp4a", flops, 66 * MS);
        let e = m.estimate("dp4a", Family::Int8, flops);
        assert!(e.measured);
        assert!(e.ns < prior);
        assert!(e.ns.abs_diff(66 * MS) < MS);
        // A faster sample does not lower it again; a slower one raises it.
        m.record("dp4a", flops, 30 * MS);
        assert!(m.estimate("dp4a", Family::Int8, flops).ns.abs_diff(66 * MS) < MS);
        m.record("dp4a", flops, 132 * MS);
        assert!(
            m.estimate("dp4a", Family::Int8, flops)
                .ns
                .abs_diff(132 * MS)
                < MS
        );
        // Per kernel: another kernel keeps its prior.
        assert!(!m.estimate("mmq", Family::Int8, flops).measured);
    }

    #[test]
    fn a_dispatch_is_cut_only_past_the_budget() {
        let m = igpu();
        assert_eq!(m.pieces(m.budget_ns(), 100), 1);
        assert_eq!(m.pieces(m.budget_ns() + 1, 100), 2);
        assert_eq!(m.pieces(10 * m.budget_ns(), 100), 10);
        // Never more ranges than workgroups along the cut dimension.
        assert_eq!(m.pieces(10 * m.budget_ns(), 3), 3);
        let off = WorkModel::new(0, 16, true, true);
        assert!(!off.enabled());
        assert_eq!(off.pieces(u64::MAX, 100), 1);
        assert!(!off.watch(u64::MAX));
    }

    #[test]
    fn ranges_cover_the_grid_once() {
        for (extent, pieces) in [(172u32, 7u32), (10, 10), (5, 9), (1, 4), (65535, 2)] {
            let r: Vec<_> = ranges(extent, pieces).collect();
            assert_eq!(r.len() as u32, pieces.min(extent));
            let mut next = 0;
            for &(base, n) in &r {
                assert_eq!(base, next);
                assert!(n >= 1);
                next += n;
            }
            assert_eq!(next, extent);
            let (lo, hi) = r
                .iter()
                .fold((u32::MAX, 0), |(lo, hi), &(_, n)| (lo.min(n), hi.max(n)));
            assert!(hi - lo <= 1);
        }
    }

    /// Refusal: only a reset-prone device, only a measured estimate, only past the factor. A
    /// fast discrete device with no slow-class facts submits as before, however heavy.
    #[test]
    fn refusal_needs_a_reset_prone_device_and_a_measured_estimate() {
        let far = Estimate {
            ns: 9 * 250 * MS,
            measured: true,
        };
        let prone = igpu();
        assert!(prone.refuse(far));
        assert!(!prone.refuse(Estimate {
            measured: false,
            ..far
        }));
        assert!(!prone.refuse(Estimate {
            ns: 8 * 250 * MS,
            ..far
        }));
        assert!(!fast().refuse(far));
        assert!(!WorkModel::new(0, 16, true, true).refuse(far));
    }

    /// A user's budget wins: a larger one keeps a dispatch whole that the default would cut.
    #[test]
    fn a_user_budget_wins() {
        let flops = gemm_flops(512, 2048, 22016);
        let dflt = igpu();
        let est = dflt.estimate("g", Family::Gemv, flops).ns;
        assert!(dflt.pieces(est, 172) > 1);
        let big = WorkModel::new(600_000 / 10, prior_gflops(2), true, true);
        assert_eq!(big.pieces(est, 172), 1);
    }
}
