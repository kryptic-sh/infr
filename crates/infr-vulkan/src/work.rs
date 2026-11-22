//! The work bound on one dispatch and one submit (plan 12, S2).
//!
//! A driver watchdog drops a GPU's work when one submit runs too long: on AMD's Windows driver
//! an engine reset (`LiveKernelEvent` 0x141), on Linux amdgpu's job timeout. A bound on
//! dispatches per submit (`VulkanShared::submit_dispatch_cap`) cannot see a dispatch that is
//! heavy on its own, which a slow device meets at a large batch: a prefill GEMV over 512 rows on
//! the 2-CU integrated GPU here is one 1.5 s dispatch. So the heavy prefill GEMMs and GEMVs, and
//! the non-coopmat flash prefill attention, say what they cost before they are recorded
//! (`Recorder::work_next`), and the recorder
//!
//! - splits a dispatch whose estimate passes the budget into several dispatches over ranges of
//!   its workgroup grid (`vkCmdDispatchBase`), each in a submit of its own and each running the
//!   same shader on the same inputs as in the whole grid, so the output is bit-identical;
//! - closes the submit at the next op boundary once the work recorded into it reaches the budget
//!   (`crate::SegmentCut::Work`).
//!
//! A dispatch whose kernel cannot be cut that way, or that is recorded where no submit can be
//! opened, stays whole, as before the bound existed.
//!
//! **The estimate** is the op's FLOPs times a time per FLOP: the one learned for that kernel in
//! this process when there is one, else a prior, `submit.prior_gflops` (a tuning parameter, scaled
//! by compute units) times a speed-up for the kernel's [`Family`]. The prior is meant to be
//! pessimistic, and is replaced only by what the GPU measured, from two sources:
//!
//! - a dispatch whose estimate is a noticeable part of the budget is bracketed by two timestamps
//!   ([`WorkModel::watch`]), and its time per FLOP replaces the prior for that kernel (the first
//!   sample) or raises it (any later, slower one);
//! - a whole segment's GPU time, which every forward segment already measures, corrects the
//!   kernels in it that have no timed dispatch ([`WorkModel::calibrate`]). Without it a device
//!   whose dispatches all stay under the timing threshold would keep its prior, and a forward
//!   whose prior passes the budget would be cut on every pass however fast it ran.
//!
//! So a kernel's estimate goes down only on what the GPU measured, never by more than
//! [`MAX_PRIOR_GAIN`] from its prior, and once one of its dispatches has been timed, only up.
//! Until a segment has completed, a device's first forward runs on the prior alone. Dispatches
//! with no estimate (every other kernel) count as free, as do single-row ones: a decode step is
//! never split, timed or cut by this, and the dispatch cap is what bounds a submit of those.

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
/// ([`infr_core::SUBMIT_DANGER_NS`]), half the limit: a tighter budget would cut forwards that
/// the dispatch-count splitter leaves whole.
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
/// unit); this keeps the prior under the slowest of those.
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

/// The prior throughput of an int8-dot GEMM the device accelerates, as a multiple of
/// `submit.prior_gflops`. Measured on the integrated GPU: the dp4a GEMM ran about 23 times as
/// fast as the per-row GEMV (690 against 30 GFLOP/s); this keeps the prior pessimistic.
const INT8_SPEEDUP: f64 = 8.0;

/// The prior throughput of a cooperative-matrix GEMM as a multiple of `submit.prior_gflops`: a
/// fraction of what the RX 7900 XTX's matrix units run.
const MATRIX_SPEEDUP: f64 = 16.0;

/// The most a measurement may raise a kernel's throughput over its prior: the ratio of the
/// watchdog's limit (twice [`infr_core::SUBMIT_DANGER_NS`], see [`default_budget_ms`]) to the
/// reset-prone budget ([`infr_core::SUBMIT_BUDGET_NS`]).
///
/// A timed dispatch is bracketed by two timestamps with no barrier before the first, so it can
/// read short when it overlaps earlier work, and a segment's time is shared out among its
/// kernels by their priors, which can favour one of them. Without a bound, one such reading
/// would take a kernel's estimate to nearly nothing for the rest of the process, out of both
/// the split and the submit cut. With it, the worst a bad reading does to a kernel whose prior
/// was exact is a budget of estimated work that really takes this many budgets, which on a
/// reset-prone device is the watchdog's limit and no more.
const MAX_PRIOR_GAIN: f64 = (2 * infr_core::SUBMIT_DANGER_NS / infr_core::SUBMIT_BUDGET_NS) as f64;

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
    /// The family's prior throughput as a multiple of `submit.prior_gflops`: [`INT8_SPEEDUP`]
    /// for an int8 dot the device accelerates (one the driver only emulates gets none) and
    /// [`MATRIX_SPEEDUP`] for matrix units.
    fn speedup(self, dot_accelerated: bool) -> f64 {
        match self {
            Family::Gemv | Family::FloatTile => 1.0,
            Family::Int8 if dot_accelerated => INT8_SPEEDUP,
            Family::Int8 => 1.0,
            Family::Matrix => MATRIX_SPEEDUP,
        }
    }
}

/// How a dispatch's grid may be cut into ranges, stated by the op that records it. A kernel
/// whose workgroups depend on each other, or on the grid's size in any other way, is not tagged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Split {
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

/// The estimated work recorded into one segment (one command buffer), by kernel: what
/// [`WorkModel::calibrate`] sets against the GPU time the segment measured.
#[derive(Default)]
pub(crate) struct SegmentWork {
    /// Each kernel's family and the FLOPs of its tagged dispatches in the segment.
    kernels: Vec<(&'static str, Family, u64)>,
}

impl SegmentWork {
    /// Count `flops` of `kernel`, a `family` kernel, into the segment.
    pub(crate) fn add(&mut self, kernel: &'static str, family: Family, flops: u64) {
        match self.kernels.iter_mut().find(|(k, _, _)| *k == kernel) {
            Some((_, _, f)) => *f = f.saturating_add(flops),
            None => self.kernels.push((kernel, family, flops)),
        }
    }
}

/// A kernel's learned time per FLOP, and where it came from.
#[derive(Clone, Copy)]
struct Learned {
    ns_per_flop: f64,
    /// From a timed dispatch of the kernel ([`WorkModel::record`]), which a segment's time no
    /// longer corrects; otherwise from segments' GPU times ([`WorkModel::calibrate`]).
    timed: bool,
}

/// The per-backend estimator: the budget, the priors, and each kernel's learned time per FLOP.
pub(crate) struct WorkModel {
    /// `0` = no work bound.
    budget_ns: u64,
    /// The slowest family's prior time per FLOP.
    prior_ns_per_flop: f64,
    dot_accelerated: bool,
    /// What has been learned so far, by kernel name.
    measured: Mutex<HashMap<&'static str, Learned>>,
}

impl WorkModel {
    pub(crate) fn new(budget_ms: u32, prior_gflops: u32, dot_accelerated: bool) -> Self {
        WorkModel {
            budget_ns: u64::from(budget_ms) * 1_000_000,
            prior_ns_per_flop: 1.0 / f64::from(prior_gflops.max(1)),
            dot_accelerated,
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

    /// A `family` kernel's prior time per FLOP.
    fn prior_rate(&self, family: Family) -> f64 {
        self.prior_ns_per_flop / family.speedup(self.dot_accelerated)
    }

    /// The estimated GPU time, in ns, of `flops` of `kernel`, a `family` kernel.
    pub(crate) fn estimate(&self, kernel: &'static str, family: Family, flops: u64) -> u64 {
        let learned = self
            .measured
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(kernel)
            .map(|l| l.ns_per_flop);
        let rate = learned.unwrap_or_else(|| self.prior_rate(family));
        (flops as f64 * rate).ceil() as u64
    }

    /// Feed one timed dispatch back: `flops` of `kernel`, a `family` kernel, took `ns`. The first
    /// sample of a kernel replaces its prior, or what segments had made of it; a later one only
    /// ever raises it. A sample of no time or no work says nothing and is dropped, and no sample
    /// takes the kernel's throughput past [`MAX_PRIOR_GAIN`] times its prior.
    pub(crate) fn record(&self, kernel: &'static str, family: Family, flops: u64, ns: u64) {
        if flops == 0 || ns == 0 {
            return;
        }
        let rate = (ns as f64 / flops as f64).max(self.prior_rate(family) / MAX_PRIOR_GAIN);
        let mut m = self.measured.lock().unwrap_or_else(|e| e.into_inner());
        let sample = Learned {
            ns_per_flop: rate,
            timed: true,
        };
        let e = m.entry(kernel).or_insert(sample);
        if !e.timed || rate > e.ns_per_flop {
            *e = sample;
        }
    }

    /// Set a completed segment's estimated work against the GPU time it measured, `gpu_ns`. When
    /// the segment ran faster than its estimate, every kernel in it without a timed dispatch has
    /// its time per FLOP scaled by measured over estimated, never past [`MAX_PRIOR_GAIN`] from
    /// its prior. Kernels with a timed dispatch keep their rate, so the segment's estimate
    /// afterwards is never under the time it measured, and equals it when no kernel in it had
    /// been timed.
    ///
    /// The segment's time also holds whatever ran in it without an estimate, so this errs slow.
    /// A later segment that runs faster still lowers the rate again: a small first forward, whose
    /// time is mostly per-dispatch overhead, must not fix the estimate of the large one after it.
    /// A segment that ran slower than its estimate changes nothing: its untagged work could be
    /// all of the difference, and only a timed dispatch ([`Self::record`]) raises an estimate.
    pub(crate) fn calibrate(&self, work: &SegmentWork, gpu_ns: u64) {
        if self.budget_ns == 0 || gpu_ns == 0 {
            return;
        }
        let mut m = self.measured.lock().unwrap_or_else(|e| e.into_inner());
        let rate = |m: &HashMap<&'static str, Learned>, kernel, family| {
            m.get(kernel)
                .map_or_else(|| self.prior_rate(family), |l| l.ns_per_flop)
        };
        let estimated: f64 = work
            .kernels
            .iter()
            .map(|&(kernel, family, flops)| flops as f64 * rate(&m, kernel, family))
            .sum();
        let measured = gpu_ns as f64;
        if measured >= estimated {
            return;
        }
        let scale = measured / estimated;
        for &(kernel, family, _) in &work.kernels {
            if m.get(kernel).is_some_and(|l| l.timed) {
                continue;
            }
            let floor = self.prior_rate(family) / MAX_PRIOR_GAIN;
            let learned = Learned {
                ns_per_flop: (rate(&m, kernel, family) * scale).max(floor),
                timed: false,
            };
            m.insert(kernel, learned);
        }
    }

    /// Every kernel with a learned rate so far and its throughput in GFLOP/s, sorted by name.
    pub(crate) fn measured_gflops(&self) -> Vec<(&'static str, f64)> {
        let m = self.measured.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<_> = m.iter().map(|(&k, l)| (k, 1.0 / l.ns_per_flop)).collect();
        v.sort_unstable_by_key(|&(k, _)| k);
        v
    }

    /// Into how many ranges a dispatch estimated at `est_ns` is cut so each stays within the
    /// budget, given `extent` workgroups along the dimension that is cut. 1 = whole. A grid has
    /// no more ranges than workgroups along that dimension, so a range can still pass the budget.
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
        WorkModel::new(default_budget_ms(true), prior_gflops(2), true)
    }

    /// Whether `kernel` has a learned rate in `m`, from a timed dispatch or a segment.
    fn learned(m: &WorkModel, kernel: &str) -> bool {
        m.measured_gflops().iter().any(|&(k, _)| k == kernel)
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
        assert!(!learned(&m, "gemv"));
        let measured_ns = 1_570 * MS;
        assert!(
            est >= measured_ns,
            "prior {} ms under the measured {} ms",
            est / MS,
            measured_ns / MS
        );
        let pieces = m.pieces(est, 172);
        assert!(
            measured_ns / u64::from(pieces) <= m.budget_ns(),
            "{pieces} ranges leave a real range over the budget"
        );
        // The old rule, a bound on dispatches only, never cut a single dispatch.
        assert!(pieces > 1);
    }

    /// The RX 7900 XTX's device facts.
    fn fast() -> WorkModel {
        WorkModel::new(default_budget_ms(false), prior_gflops(96), true)
    }

    /// On the RX 7900 XTX, by the prior alone, the default-batch GEMMs of the models run there
    /// stay whole and untimed.
    #[test]
    fn a_fast_device_at_default_batch_keeps_its_dispatches_whole_and_untimed() {
        let m = fast();
        // Qwen3-1.7B gate/up at 512 rows on the coopmat GEMM, and Qwen3-14B's at 1024 rows.
        for flops in [gemm_flops(512, 2048, 12288), gemm_flops(1024, 5120, 34816)] {
            let est = m.estimate("native_gemm_warp_q4k", Family::Matrix, flops);
            assert_eq!(m.pieces(est, 1 << 16), 1);
            assert!(est < m.budget_ns() / 4, "{} ms", est / MS);
            assert!(!m.watch(est));
        }
    }

    /// The tagged GEMMs of one SmolLM3-3B layer (width 2048, feed-forward 11008, 4 of 16 heads'
    /// worth of K and V) at `rows` rows, as `(kernel, FLOPs)`: the K-quant file keeps some
    /// weights in a second format, so a forward runs two kernels.
    fn smollm3_layer(rows: usize) -> [(&'static str, u64); 6] {
        [
            ("q4k", gemm_flops(rows, 2048, 2048)),      // q
            ("q4k", gemm_flops(rows, 2048, 512)),       // k
            ("q6k", gemm_flops(rows, 2048, 512)),       // v
            ("q4k", gemm_flops(rows, 2048, 2048)),      // o
            ("q4k", gemm_flops(rows, 2048, 2 * 11008)), // gate and up, fused
            ("q6k", gemm_flops(rows, 11008, 2048)),     // down
        ]
    }

    const SMOLLM3_LAYERS: u64 = 36;

    /// The estimated work of a whole SmolLM3-3B prefill forward at `rows` rows, and the same as
    /// one segment's work.
    fn smollm3_forward(m: &WorkModel, rows: usize) -> (u64, SegmentWork) {
        let mut work = SegmentWork::default();
        let mut ns = 0;
        for (kernel, flops) in smollm3_layer(rows) {
            ns += SMOLLM3_LAYERS * m.estimate(kernel, Family::Matrix, flops);
            work.add(kernel, Family::Matrix, SMOLLM3_LAYERS * flops);
        }
        (ns, work)
    }

    /// A whole prefill forward of a 3B model at the default 1024-row batch stays under the fast
    /// device's budget by the prior alone, so no submit is closed by work. Under the reset-prone
    /// budget the same estimate would cut it.
    #[test]
    fn a_fast_device_forward_at_default_batch_takes_no_work_cut() {
        let m = fast();
        let (forward, _) = smollm3_forward(&m, 1024);
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

    /// The case the segment calibration exists for, with the figures measured on the RX 7900 XTX
    /// (SmolLM3-3B, 4096 rows, 2026-11-22: a forward of about 850 ms of GPU time): no GEMM of
    /// it is heavy enough to be timed, and by the prior the forward passes the budget, so it is
    /// cut on every pass. Once a segment's GPU time has been set against its estimate, the same
    /// forward is estimated at what it measured and is left whole.
    #[test]
    fn a_segment_faster_than_its_prior_stops_the_cuts_it_caused() {
        let m = fast();
        let rows = 4096;
        for (kernel, flops) in smollm3_layer(rows) {
            assert!(!m.watch(m.estimate(kernel, Family::Matrix, flops)));
        }
        let (by_prior, work) = smollm3_forward(&m, rows);
        assert_eq!(
            crate::segment_cut(1, 0, by_prior, m.budget_ns(), 0),
            Some(crate::SegmentCut::Work),
            "{} ms by the prior",
            by_prior / MS
        );
        // Before any segment completes nothing has moved: a first forward runs on the prior.
        assert_eq!(smollm3_forward(&m, rows).0, by_prior);
        assert!(m.measured_gflops().is_empty());

        let measured = 850 * MS;
        m.calibrate(&work, measured);
        let (after, _) = smollm3_forward(&m, rows);
        // Every kernel was on the prior, so the estimate is now the measured time (each
        // dispatch's estimate rounds up a nanosecond), never under it.
        assert!(
            after >= measured,
            "{after} ns under the measured {measured}"
        );
        assert!(after - measured < MS, "{} ms", after / MS);
        assert_eq!(crate::segment_cut(1, 0, after, m.budget_ns(), 0), None);
    }

    /// A segment's time can only make an estimate less pessimistic: a segment that ran slower
    /// than estimated changes nothing (its untagged work may be the whole difference), a faster
    /// one lowers it again, and once a dispatch of the kernel has been timed, segments leave it
    /// alone.
    #[test]
    fn a_segment_only_lowers_an_estimate_and_a_timed_dispatch_settles_it() {
        let m = igpu();
        let flops = gemm_flops(64, 2048, 2048);
        let mut work = SegmentWork::default();
        work.add("dp4a", Family::Int8, flops);
        let prior = m.estimate("dp4a", Family::Int8, flops);
        let est = |m: &WorkModel| m.estimate("dp4a", Family::Int8, flops);

        m.calibrate(&work, 3 * prior);
        assert!(!learned(&m, "dp4a"));
        assert_eq!(est(&m), prior);

        m.calibrate(&work, prior / 2);
        assert!(est(&m).abs_diff(prior / 2) <= 1, "{}", est(&m));
        m.calibrate(&work, prior / 4);
        assert!(est(&m).abs_diff(prior / 4) <= 1, "{}", est(&m));
        // Slower than the estimate now is: nothing moves.
        m.calibrate(&work, prior / 3);
        assert!(est(&m).abs_diff(prior / 4) <= 1, "{}", est(&m));

        // A timed dispatch replaces what the segments made of it, faster or slower, and then
        // only a slower timed dispatch moves it.
        m.record("dp4a", Family::Int8, flops, prior / 5);
        assert!(est(&m).abs_diff(prior / 5) <= 1, "{}", est(&m));
        m.calibrate(&work, prior / 6);
        assert!(est(&m).abs_diff(prior / 5) <= 1, "{}", est(&m));
        m.record("dp4a", Family::Int8, flops, prior);
        assert!(est(&m).abs_diff(prior) <= 1, "{}", est(&m));
        m.record("dp4a", Family::Int8, flops, prior / 2);
        assert!(est(&m).abs_diff(prior) <= 1, "{}", est(&m));
    }

    /// A small forward first (a short prompt: its GPU time is mostly per-dispatch overhead, so
    /// it reads close to the prior) must not fix the estimate the large forward after it gets:
    /// the large one is cut once, by the estimate the small one left, and not again.
    #[test]
    fn a_small_first_forward_does_not_keep_a_large_one_cut() {
        let m = fast();
        let (small_est, small) = smollm3_forward(&m, 64);
        m.calibrate(&small, small_est - small_est / 25);
        let (first, work) = smollm3_forward(&m, 4096);
        assert_eq!(
            crate::segment_cut(1, 0, first, m.budget_ns(), 0),
            Some(crate::SegmentCut::Work)
        );
        let measured = 850 * MS;
        m.calibrate(&work, measured);
        let (after, _) = smollm3_forward(&m, 4096);
        assert!(after >= measured && after - measured < MS, "{after} ns");
        assert_eq!(crate::segment_cut(1, 0, after, m.budget_ns(), 0), None);
    }

    /// In a segment that mixes a timed kernel with one still on the prior, only the second
    /// moves, and the segment's estimate stays at or above what it measured: the correction
    /// cannot make a submit look lighter than it was.
    #[test]
    fn a_calibrated_segment_is_never_estimated_under_its_measured_time() {
        let m = igpu();
        let (heavy, light) = (gemm_flops(512, 2048, 22016), gemm_flops(512, 2048, 512));
        m.record("timed", Family::Int8, heavy, 66 * MS);
        let timed = m.estimate("timed", Family::Int8, heavy);
        let mut work = SegmentWork::default();
        work.add("timed", Family::Int8, heavy);
        work.add("fresh", Family::Int8, light);
        let before = timed + m.estimate("fresh", Family::Int8, light);
        for measured in [before - MS, before / 2, 67 * MS, MS] {
            let m = igpu();
            m.record("timed", Family::Int8, heavy, 66 * MS);
            m.calibrate(&work, measured);
            assert_eq!(m.estimate("timed", Family::Int8, heavy), timed);
            let after = timed + m.estimate("fresh", Family::Int8, light);
            assert!(after < before, "measured {measured}: nothing was corrected");
            assert!(after >= measured, "measured {measured}: estimated {after}");
        }
    }

    /// No reading takes a kernel more than `MAX_PRIOR_GAIN` past its prior, and one of no time
    /// is not a reading: a timed dispatch that read zero or far too short (it overlapped earlier
    /// work), or a segment that read almost nothing, would otherwise put the kernel's estimate
    /// at nothing for the rest of the process, outside both the split and the submit cut.
    #[test]
    fn a_bad_reading_cannot_take_a_kernel_out_of_the_bound() {
        let flops = gemm_flops(512, 2048, 22016);
        let prior = igpu().estimate("gemv", Family::Gemv, flops);
        let floor = (prior as f64 / MAX_PRIOR_GAIN) as u64;

        let zero = igpu();
        zero.record("gemv", Family::Gemv, flops, 0);
        assert!(!learned(&zero, "gemv"), "a zero-time sample was kept");
        assert_eq!(zero.estimate("gemv", Family::Gemv, flops), prior);
        let mut work = SegmentWork::default();
        work.add("gemv", Family::Gemv, flops);
        zero.calibrate(&work, 0);
        assert_eq!(zero.estimate("gemv", Family::Gemv, flops), prior);

        let short = igpu();
        short.record("gemv", Family::Gemv, flops, 1);
        let est = short.estimate("gemv", Family::Gemv, flops);
        assert!(
            est.abs_diff(floor) <= 1,
            "{est} ns against a floor of {floor}"
        );
        // Still a dispatch the bound cuts and times.
        assert!(short.pieces(est, 172) > 1);
        assert!(short.watch(est));
        // A real, slower sample then raises it as usual.
        short.record("gemv", Family::Gemv, flops, prior / 2);
        assert!(
            short
                .estimate("gemv", Family::Gemv, flops)
                .abs_diff(prior / 2)
                <= 1
        );

        let segment = igpu();
        segment.calibrate(&work, 1);
        let est = segment.estimate("gemv", Family::Gemv, flops);
        assert!(
            est.abs_diff(floor) <= 1,
            "{est} ns against a floor of {floor}"
        );
    }

    /// The gain bound is the reset-prone budget's margin under the watchdog's limit: a budget of
    /// estimated work that a bad reading understated by the whole bound still fits the limit.
    #[test]
    fn the_gain_bound_is_the_reset_prone_budgets_margin() {
        let budget = u64::from(default_budget_ms(true)) * MS;
        let limit = 2 * u64::from(default_budget_ms(false)) * MS;
        assert_eq!((budget as f64 * MAX_PRIOR_GAIN) as u64, limit);
    }

    #[test]
    fn a_measurement_replaces_the_prior_then_only_raises_it() {
        let m = igpu();
        let flops = gemm_flops(512, 2048, 22016);
        let prior = m.estimate("dp4a", Family::Int8, flops);
        // Measured 66 ms: the estimate drops to it.
        m.record("dp4a", Family::Int8, flops, 66 * MS);
        let e = m.estimate("dp4a", Family::Int8, flops);
        assert!(learned(&m, "dp4a"));
        assert!(e < prior);
        assert!(e.abs_diff(66 * MS) < MS);
        // A faster sample does not lower it again; a slower one raises it.
        m.record("dp4a", Family::Int8, flops, 60 * MS);
        assert!(m.estimate("dp4a", Family::Int8, flops).abs_diff(66 * MS) < MS);
        m.record("dp4a", Family::Int8, flops, 132 * MS);
        assert!(m.estimate("dp4a", Family::Int8, flops).abs_diff(132 * MS) < MS);
        // Per kernel: another kernel keeps its prior.
        assert!(!learned(&m, "mmq"));
    }

    #[test]
    fn a_dispatch_is_cut_only_past_the_budget() {
        let m = igpu();
        assert_eq!(m.pieces(m.budget_ns(), 100), 1);
        assert_eq!(m.pieces(m.budget_ns() + 1, 100), 2);
        assert_eq!(m.pieces(10 * m.budget_ns(), 100), 10);
        // Never more ranges than workgroups along the cut dimension.
        assert_eq!(m.pieces(10 * m.budget_ns(), 3), 3);
        let off = WorkModel::new(0, 16, true);
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

    /// A user's budget wins: a larger one keeps a dispatch whole that the default would cut.
    #[test]
    fn a_user_budget_wins() {
        let flops = gemm_flops(512, 2048, 22016);
        let dflt = igpu();
        let est = dflt.estimate("g", Family::Gemv, flops);
        assert!(dflt.pieces(est, 172) > 1);
        let big = WorkModel::new(600_000 / 10, prior_gflops(2), true);
        assert_eq!(big.pieces(est, 172), 1);
    }
}
