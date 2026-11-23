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
//!   ([`WorkModel::watch`]), and the kernel's time per FLOP becomes the slowest of its last
//!   [`TIMED_WINDOW`] such samples, the prior standing in for the samples it does not have
//!   yet;
//! - a whole segment's GPU time, which every forward segment already measures, corrects the
//!   kernels in it that have no timed dispatch ([`WorkModel::calibrate`]), as if they had taken
//!   all of it. Without it a device whose dispatches all stay under the timing threshold would
//!   keep its prior, and a forward whose prior passes the budget would be cut on every pass
//!   however fast it ran.
//!
//! So a kernel's estimate goes down only on what the GPU measured, and never by more than the
//! model's gain bound ([`max_prior_gain`]) from its prior: a submit of [`SUBMIT_BUDGETS`]
//! budgets whose every estimate a bad reading took to that floor still runs under
//! [`floor_submit_ceiling_ns`] if the priors were right. That holds for what the bound sees:
//! work without an estimate, a tagged dispatch that stays whole, and a range held to one
//! workgroup can each put more into a submit than [`SUBMIT_BUDGETS`] budgets. Until a segment
//! has completed, a device's first forward runs on the prior alone.
//! Dispatches with no estimate (every other kernel) count as free, as do single-row ones: a
//! decode step is never split, timed or cut by this, and the dispatch cap is what bounds a
//! submit of those.

use std::collections::HashMap;
use std::sync::Mutex;

/// `submit.budget_ms`'s default for a device a watchdog is known to reset under long work
/// (`tuning::DeviceFacts::reset_prone`), or any other.
///
/// The watchdogs measured so far reset a submit after about 2 s (Windows' default `TdrDelay`;
/// amdgpu's default `lockup_timeout` on the RX 580 of issue 94), and Vulkan reports neither limit,
/// so the defaults assume it ([`WATCHDOG_NS`]). A reset-prone device gets the submit splitter's
/// own target ([`infr_core::SUBMIT_BUDGET_NS`]), a small part of that: its estimate is a prior
/// until a kernel has been timed, and another program on the GPU stretches a submit. Any other
/// device gets
/// [`infr_core::SUBMIT_DANGER_NS`] over [`SUBMIT_BUDGETS`]: Windows' `TdrDelay` is not per
/// adapter, so a fast discrete card is under the same limit, and once its estimates have been
/// corrected to what its submits measure, a submit of [`SUBMIT_BUDGETS`] budgets is that much
/// real time. This keeps such a submit at the time from which the dispatch-count splitter
/// treats a forward as dangerous on any device.
pub(crate) fn default_budget_ms(reset_prone: bool) -> u32 {
    let ns = if reset_prone {
        infr_core::SUBMIT_BUDGET_NS
    } else {
        infr_core::SUBMIT_DANGER_NS / SUBMIT_BUDGETS
    };
    (ns / 1_000_000) as u32
}

/// How many budgets of estimated work one submit can hold: it is closed at the next op boundary
/// after its work reaches the budget (`crate::segment_cut`), and the op that takes it there
/// records at most one tagged dispatch, itself at most a budget (a heavier one is cut and opens
/// submits of its own).
///
/// That counts estimated work only, and only where the cut can act. A submit also holds
/// whatever was recorded without an estimate; a tagged dispatch that cannot be cut, or is
/// recorded where no submit can be opened, goes in whole however heavy; and a dispatch has no
/// more ranges than workgroups along the dimension that is cut, so one range can pass a budget.
const SUBMIT_BUDGETS: u64 = 2;

/// The watchdog's limit the defaults assume (see [`default_budget_ms`]).
const WATCHDOG_NS: u64 = 2 * infr_core::SUBMIT_DANGER_NS;

/// What [`floor_submit_ceiling_ns`] keeps free under [`WATCHDOG_NS`] on a reset-prone device.
const RESET_PRONE_MARGIN_NS: u64 = WATCHDOG_NS / 4;

/// The real GPU time a submit may take when a bad reading has put every estimate in it at the
/// gain floor and the priors were exact. On a reset-prone device, where a reset is known to
/// follow, [`WATCHDOG_NS`] less [`RESET_PRONE_MARGIN_NS`]; on any other, [`WATCHDOG_NS`] itself,
/// with no margin. [`max_prior_gain`] is derived from it.
pub(crate) fn floor_submit_ceiling_ns(reset_prone: bool) -> u64 {
    if reset_prone {
        WATCHDOG_NS - RESET_PRONE_MARGIN_NS
    } else {
        WATCHDOG_NS
    }
}

/// The most a measurement may raise a kernel's throughput over its prior under a budget of
/// `budget_ns`: what keeps [`SUBMIT_BUDGETS`] budgets of work estimated at that floor within
/// [`floor_submit_ceiling_ns`] of real time when the prior was exact. Never under 1: under a
/// budget too large for the ceiling no measurement lowers an estimate below the prior. A slow
/// timed sample still raises one.
///
/// It bounds what [`SUBMIT_BUDGETS`] describes and no more (see there for what a submit can
/// hold beyond it).
///
/// A timed dispatch is bracketed by two timestamps with no barrier before the first, so it can
/// read short when it overlaps earlier work, and a segment's time is shared out among its
/// untimed kernels by their priors, which can favour one of them. Without a bound, one such
/// reading would take a kernel's estimate to nearly nothing for the rest of the process, out of
/// both the split and the submit cut.
pub(crate) fn max_prior_gain(budget_ns: u64, reset_prone: bool) -> f64 {
    let full_submit = SUBMIT_BUDGETS.saturating_mul(budget_ns).max(1);
    (floor_submit_ceiling_ns(reset_prone) as f64 / full_submit as f64).max(1.0)
}

/// The most a timed sample may lower a kernel's throughput under its prior. The priors are
/// meant to be pessimistic, so a sample this far on the slow side is taken for a bad reading
/// (a timestamp pair across a discontinuity, a dispatch that waited on another program) and
/// held here: unbounded, it would cut every dispatch of the kernel into one range per workgroup
/// until [`TIMED_WINDOW`] later samples had replaced it.
const MAX_PRIOR_SLOWDOWN: f64 = 64.0;

/// How many of a kernel's latest timed samples its rate is taken from (the slowest of them).
/// One slow outlier is forgotten after this many more samples. Until a kernel has this many, the
/// prior stands in for the ones it lacks, so lowering the rate takes this many fast readings in
/// a row, each within the gain bound, from the first sample on.
pub(crate) const TIMED_WINDOW: usize = 4;

/// The ticks from timestamp `start` to timestamp `end` on a queue whose timestamps have
/// `valid_bits` valid bits (`timestampValidBits`): the counter wraps at two to that power, so
/// the difference is taken modulo it. Without the mask a pair that straddles a wrap of a
/// counter narrower than 64 bits reads as nearly 2^64 ticks.
pub(crate) fn ticks_between(start: u64, end: u64, valid_bits: u32) -> u64 {
    let mask = if valid_bits >= u64::BITS {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    };
    end.wrapping_sub(start) & mask
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
    /// The kernel's latest timed samples ([`WorkModel::record`]), newest last, of which
    /// `ns_per_flop` is the slowest. Empty while the rate comes from segments' GPU times
    /// ([`WorkModel::calibrate`]), which no longer correct a kernel once it has a sample.
    timed: [Option<f64>; TIMED_WINDOW],
}

impl Learned {
    fn is_timed(&self) -> bool {
        self.timed.iter().any(Option::is_some)
    }
}

/// The per-backend estimator: the budget, the priors, and each kernel's learned time per FLOP.
pub(crate) struct WorkModel {
    /// `0` = no work bound.
    budget_ns: u64,
    /// [`max_prior_gain`] of the budget and the device class.
    max_gain: f64,
    /// The slowest family's prior time per FLOP.
    prior_ns_per_flop: f64,
    dot_accelerated: bool,
    /// What has been learned so far, by kernel name.
    measured: Mutex<HashMap<&'static str, Learned>>,
}

impl WorkModel {
    /// `reset_prone`: a device a watchdog is known to reset under long work
    /// (`tuning::DeviceFacts::reset_prone`), which keeps more margin under the limit.
    pub(crate) fn new(
        budget_ms: u32,
        prior_gflops: u32,
        dot_accelerated: bool,
        reset_prone: bool,
    ) -> Self {
        WorkModel {
            budget_ns: u64::from(budget_ms) * 1_000_000,
            max_gain: max_prior_gain(u64::from(budget_ms) * 1_000_000, reset_prone),
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

    /// Feed one timed dispatch back: `flops` of `kernel`, a `family` kernel, took `ns`. The
    /// kernel's rate becomes the slowest of its last [`TIMED_WINDOW`] samples, this one included,
    /// with the prior counted for every sample it does not have yet: one slow reading is
    /// forgotten once that many others have followed, and the rate only comes down when that
    /// many in a row read faster. So a kernel's first sample cannot lower its estimate on its
    /// own (and take it under the [`WorkModel::watch`] threshold, never to be timed again); it
    /// can raise it. It also ends what segments had made of the kernel: until the window is
    /// full the estimate is back at the prior or above. A sample of no time or no work says
    /// nothing and is dropped; any other is held between the gain bound ([`max_prior_gain`]) on
    /// the fast side of the kernel's prior and [`MAX_PRIOR_SLOWDOWN`] on the slow side.
    pub(crate) fn record(&self, kernel: &'static str, family: Family, flops: u64, ns: u64) {
        if flops == 0 || ns == 0 {
            return;
        }
        let prior = self.prior_rate(family);
        let rate =
            (ns as f64 / flops as f64).clamp(prior / self.max_gain, prior * MAX_PRIOR_SLOWDOWN);
        let mut m = self.measured.lock().unwrap_or_else(|e| e.into_inner());
        let e = m.entry(kernel).or_insert(Learned {
            ns_per_flop: rate,
            timed: [None; TIMED_WINDOW],
        });
        e.timed.rotate_left(1);
        e.timed[TIMED_WINDOW - 1] = Some(rate);
        e.ns_per_flop = e
            .timed
            .iter()
            .map(|sample| sample.unwrap_or(prior))
            .fold(rate, f64::max);
    }

    /// Set a completed segment's estimated work against the GPU time it measured, `gpu_ns`. The
    /// kernels in it without a timed dispatch are treated as if they had taken ALL of that time:
    /// when even so they ran faster than their estimate, each has its time per FLOP scaled by
    /// measured over their estimate, never past the gain bound ([`max_prior_gain`]) from its
    /// prior. Their estimate for this segment is then the time the whole segment measured, or
    /// more where the bound held a kernel back.
    ///
    /// So for the untimed kernels of this segment TAKEN TOGETHER, in the mix the segment ran,
    /// this errs slow and only slow: together they cannot have taken more than the segment did,
    /// whatever the timed kernels and the work without an estimate took of it. It says nothing
    /// of each one. They share one factor, so a kernel slower against its prior than its
    /// neighbours comes out too fast, by at most the gain bound, and only a timed dispatch
    /// raises it again.
    ///
    /// The timed kernels' share is deliberately not subtracted. Their rate is the slowest of a few
    /// samples, so it overstates them, and taking that out of the measured time would hand the
    /// overstatement to the untimed kernels as speed nothing measured, again with every segment.
    /// The price is that a segment mostly made of timed work corrects nothing.
    ///
    /// A later segment that runs faster still lowers the rate again: a small first forward, whose
    /// time is mostly per-dispatch overhead, must not fix the estimate of the large one after it.
    /// A segment that ran slower changes nothing: only a timed dispatch ([`Self::record`]) raises
    /// an estimate.
    pub(crate) fn calibrate(&self, work: &SegmentWork, gpu_ns: u64) {
        if self.budget_ns == 0 || gpu_ns == 0 {
            return;
        }
        let mut m = self.measured.lock().unwrap_or_else(|e| e.into_inner());
        let untimed = |m: &HashMap<&'static str, Learned>, kernel| {
            !m.get(kernel).is_some_and(Learned::is_timed)
        };
        let rate = |m: &HashMap<&'static str, Learned>, kernel, family| {
            m.get(kernel)
                .map_or_else(|| self.prior_rate(family), |l| l.ns_per_flop)
        };
        let estimated: f64 = work
            .kernels
            .iter()
            .filter(|&&(kernel, _, _)| untimed(&m, kernel))
            .map(|&(kernel, family, flops)| flops as f64 * rate(&m, kernel, family))
            .sum();
        let measured = gpu_ns as f64;
        if measured >= estimated {
            return;
        }
        let scale = measured / estimated;
        for &(kernel, family, _) in &work.kernels {
            if !untimed(&m, kernel) {
                continue;
            }
            let floor = self.prior_rate(family) / self.max_gain;
            let current = rate(&m, kernel, family);
            let corrected = (current * scale).max(floor);
            // Held at the floor already, or under a gain bound of 1: nothing to learn.
            if corrected < current {
                let learned = Learned {
                    ns_per_flop: corrected,
                    timed: [None; TIMED_WINDOW],
                };
                m.insert(kernel, learned);
            }
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

    /// The limit the watchdogs were measured at, about 2 s (plan 12 sections 0d and 0e), written
    /// out here and not taken from the module's constants: these tests are what says the
    /// constants respect it.
    const MEASURED_LIMIT_NS: u64 = 2_000 * MS;

    /// The integrated GPU here (2 CUs): its per-row GEMV over the SmolLM3 gate/up weight at 512
    /// rows measured about 1.57 s, the reproduction's heaviest dispatch.
    fn igpu() -> WorkModel {
        WorkModel::new(default_budget_ms(true), prior_gflops(2), true, true)
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
        assert_eq!(default_budget_ms(false), 500);
    }

    /// With estimates corrected to what the GPU measures, a submit's estimate is its real time,
    /// and a submit holds up to `SUBMIT_BUDGETS` budgets: on every device class that is at most
    /// half the watchdog's limit by default.
    #[test]
    fn a_full_submit_of_true_estimates_keeps_half_the_limit_free() {
        for reset_prone in [true, false] {
            let budget = u64::from(default_budget_ms(reset_prone)) * MS;
            assert!(
                SUBMIT_BUDGETS * budget <= MEASURED_LIMIT_NS / 2,
                "reset_prone {reset_prone}: a full submit is {} ms",
                SUBMIT_BUDGETS * budget / MS
            );
        }
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
        WorkModel::new(default_budget_ms(false), prior_gflops(96), true, false)
    }

    /// On the RX 7900 XTX, by the prior alone, the default-batch GEMMs of the models run there
    /// stay whole, and Qwen3-1.7B's are not even timed.
    #[test]
    fn a_fast_device_at_default_batch_keeps_its_dispatches_whole() {
        let m = fast();
        // Qwen3-1.7B gate/up at 512 rows on the coopmat GEMM, and Qwen3-14B's at 1024 rows.
        let small = m.estimate("g", Family::Matrix, gemm_flops(512, 2048, 12288));
        let large = m.estimate("g", Family::Matrix, gemm_flops(1024, 5120, 34816));
        for est in [small, large] {
            assert_eq!(m.pieces(est, 1 << 16), 1);
            assert!(est < m.budget_ns() / 4, "{} ms", est / MS);
        }
        assert!(!m.watch(small));
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

    /// The rows and the GPU time of the case the segment correction exists for: a SmolLM3-3B
    /// forward on the RX 7900 XTX whose GEMMs are all too light to be timed and whose prior
    /// passes the budget. The time is half of what the 4096-row forward measured there
    /// (2026-11-22, about 850 ms), not a measurement of its own.
    const UNTIMED_ROWS: usize = 2048;
    const UNTIMED_FORWARD_NS: u64 = 425 * MS;

    /// By the prior that forward passes the budget, so it is cut on every pass, and nothing in
    /// it is timed. Once a segment's GPU time has been set against its estimate, the same
    /// forward is estimated under the budget, never under what it measured, and is left whole.
    #[test]
    fn a_segment_faster_than_its_prior_stops_the_cuts_it_caused() {
        let m = fast();
        for (kernel, flops) in smollm3_layer(UNTIMED_ROWS) {
            assert!(!m.watch(m.estimate(kernel, Family::Matrix, flops)));
        }
        let (by_prior, work) = smollm3_forward(&m, UNTIMED_ROWS);
        assert_eq!(
            crate::segment_cut(1, 0, by_prior, m.budget_ns(), 0),
            Some(crate::SegmentCut::Work),
            "{} ms by the prior",
            by_prior / MS
        );
        // Before any segment completes nothing has moved: a first forward runs on the prior.
        assert_eq!(smollm3_forward(&m, UNTIMED_ROWS).0, by_prior);
        assert!(m.measured_gflops().is_empty());

        m.calibrate(&work, UNTIMED_FORWARD_NS);
        let (after, _) = smollm3_forward(&m, UNTIMED_ROWS);
        assert!(after >= UNTIMED_FORWARD_NS, "{after} ns under the measured");
        assert!(after < by_prior);
        assert_eq!(crate::segment_cut(1, 0, after, m.budget_ns(), 0), None);
    }

    /// When the gain bound does not hold a kernel back, a segment of untimed kernels is estimated
    /// afterwards at exactly what it measured (each dispatch's estimate rounds up a nanosecond).
    #[test]
    fn a_corrected_segment_is_estimated_at_what_it_measured() {
        let m = fast();
        let (by_prior, work) = smollm3_forward(&m, UNTIMED_ROWS);
        let measured = by_prior - by_prior / 4;
        m.calibrate(&work, measured);
        let (after, _) = smollm3_forward(&m, UNTIMED_ROWS);
        assert!(
            after >= measured,
            "{after} ns under the measured {measured}"
        );
        assert!(after - measured < MS, "{} ms", after / MS);
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
        let tenths = |n: u64| prior * n / 10;

        m.calibrate(&work, 3 * prior);
        assert!(!learned(&m, "dp4a"));
        assert_eq!(est(&m), prior);

        m.calibrate(&work, tenths(8));
        assert!(est(&m).abs_diff(tenths(8)) <= 1, "{}", est(&m));
        m.calibrate(&work, tenths(6));
        assert!(est(&m).abs_diff(tenths(6)) <= 1, "{}", est(&m));
        // Slower than the estimate now is: nothing moves.
        m.calibrate(&work, tenths(7));
        assert!(est(&m).abs_diff(tenths(6)) <= 1, "{}", est(&m));

        // A timed dispatch ends what the segments made of the kernel: with one sample the
        // estimate is back at the prior, and a faster segment no longer moves it.
        m.record("dp4a", Family::Int8, flops, tenths(5));
        assert_eq!(est(&m), prior);
        m.calibrate(&work, tenths(4));
        assert_eq!(est(&m), prior);
        // A full window of timed samples settles it, and segments still leave it alone.
        for _ in 1..TIMED_WINDOW {
            m.record("dp4a", Family::Int8, flops, tenths(5));
        }
        assert!(est(&m).abs_diff(tenths(5)) <= 1, "{}", est(&m));
        m.calibrate(&work, tenths(4));
        assert!(est(&m).abs_diff(tenths(5)) <= 1, "{}", est(&m));
    }

    /// A small forward first (a short prompt: its GPU time is mostly per-dispatch overhead, so
    /// it reads close to the prior) must not fix the estimate the large forward after it gets:
    /// the large one is cut once, by the estimate the small one left, and not again.
    #[test]
    fn a_small_first_forward_does_not_keep_a_large_one_cut() {
        let m = fast();
        let (small_est, small) = smollm3_forward(&m, 64);
        m.calibrate(&small, small_est - small_est / 25);
        let (first, work) = smollm3_forward(&m, UNTIMED_ROWS);
        assert_eq!(
            crate::segment_cut(1, 0, first, m.budget_ns(), 0),
            Some(crate::SegmentCut::Work)
        );
        m.calibrate(&work, UNTIMED_FORWARD_NS);
        let (after, _) = smollm3_forward(&m, UNTIMED_ROWS);
        assert!(after >= UNTIMED_FORWARD_NS, "{after} ns");
        assert_eq!(crate::segment_cut(1, 0, after, m.budget_ns(), 0), None);
    }

    /// In a segment that mixes a timed kernel with an untimed one, the untimed one is corrected
    /// as if it had taken the whole segment: it moves only when the segment ran faster than the
    /// untimed kernel's own estimate, the timed kernel never moves, and the segment's estimate
    /// stays at or above what it measured.
    #[test]
    fn an_untimed_kernel_is_corrected_as_if_it_took_the_whole_segment() {
        let (heavy, light) = (gemm_flops(512, 2048, 22016), gemm_flops(512, 2048, 512));
        let mut work = SegmentWork::default();
        work.add("timed", Family::Int8, heavy);
        work.add("fresh", Family::Int8, light);
        let model = || {
            let m = igpu();
            m.record("timed", Family::Int8, heavy, 150 * MS);
            m
        };
        let timed = model().estimate("timed", Family::Int8, heavy);
        let fresh = model().estimate("fresh", Family::Int8, light);
        // The segment ran faster than its whole estimate, but not faster than the untimed
        // kernel's share of it: no evidence about that kernel, nothing moves.
        for measured in [timed + fresh - MS, timed, fresh + MS] {
            let m = model();
            m.calibrate(&work, measured);
            assert!(!learned(&m, "fresh"), "measured {measured}");
            assert_eq!(m.estimate("fresh", Family::Int8, light), fresh);
        }
        // Faster than the untimed kernel's own estimate: it cannot have taken longer.
        for measured in [fresh - MS, fresh / 2, MS] {
            let m = model();
            m.calibrate(&work, measured);
            assert_eq!(m.estimate("timed", Family::Int8, heavy), timed);
            let after = m.estimate("fresh", Family::Int8, light);
            assert!(after < fresh, "measured {measured}: nothing was corrected");
            assert!(after >= measured, "measured {measured}: estimated {after}");
        }
    }

    /// The sequence that must not move an untimed kernel: a timed kernel whose rate latched on
    /// one slow sample, so every later segment holding both reads faster than its estimate. The
    /// untimed kernel's prior is exact here, and segment after segment says nothing about it;
    /// subtracting the timed kernel's overstated share from the measured time would take the
    /// untimed one toward the gain floor a step at a time.
    #[test]
    fn a_slow_timed_kernel_does_not_push_an_untimed_one_toward_the_floor() {
        let m = igpu();
        let (t_flops, u_flops) = (gemm_flops(512, 2048, 22016), gemm_flops(512, 2048, 4096));
        let u_prior = m.estimate("u", Family::Int8, u_flops);
        // The timed kernel really takes 100 ms; its one sample read 250.
        let t_real = 100 * MS;
        m.record("t", Family::Int8, t_flops, 250 * MS);
        assert!(m.estimate("t", Family::Int8, t_flops) > 2 * t_real);
        let mut work = SegmentWork::default();
        work.add("t", Family::Int8, t_flops);
        work.add("u", Family::Int8, u_flops);
        for _ in 0..32 {
            m.calibrate(&work, t_real + u_prior);
        }
        assert!(!learned(&m, "u"), "the untimed kernel was corrected");
        assert_eq!(m.estimate("u", Family::Int8, u_flops), u_prior);
    }

    /// No reading takes a kernel past the gain bound from its prior, and one of no time is not a
    /// reading: a timed dispatch that read zero or far too short (it overlapped earlier work),
    /// or a segment that read almost nothing, would otherwise put the kernel's estimate at
    /// nothing for the rest of the process, outside both the split and the submit cut.
    #[test]
    fn a_bad_reading_cannot_take_a_kernel_out_of_the_bound() {
        let flops = gemm_flops(512, 2048, 22016);
        let prior = igpu().estimate("gemv", Family::Gemv, flops);
        let floor = (prior as f64 / max_prior_gain(igpu().budget_ns(), true)) as u64;
        assert!(floor < prior);

        let zero = igpu();
        zero.record("gemv", Family::Gemv, flops, 0);
        assert!(!learned(&zero, "gemv"), "a zero-time sample was kept");
        assert_eq!(zero.estimate("gemv", Family::Gemv, flops), prior);
        let mut work = SegmentWork::default();
        work.add("gemv", Family::Gemv, flops);
        zero.calibrate(&work, 0);
        assert_eq!(zero.estimate("gemv", Family::Gemv, flops), prior);

        let short = igpu();
        for _ in 0..TIMED_WINDOW {
            short.record("gemv", Family::Gemv, flops, 1);
        }
        let est = short.estimate("gemv", Family::Gemv, flops);
        assert!(
            est.abs_diff(floor) <= 1,
            "{est} ns against a floor of {floor}"
        );
        // Still a dispatch the bound cuts and times.
        assert!(short.pieces(est, 172) > 1);
        assert!(short.watch(est));
        // A real, slower sample then raises it.
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

    /// What the gain bound is for. Take a kernel whose prior is exact, let bad readings (a timed
    /// dispatch and a segment that both read almost nothing) put its estimate at the floor, and
    /// fill a submit with as much of it as a submit can hold by estimate, `SUBMIT_BUDGETS`
    /// budgets. Its real time, at the prior's rate, must stay under the watchdog's limit, with a
    /// quarter of the limit to spare on a reset-prone device: for the default budgets and for
    /// budgets a user sets.
    #[test]
    fn a_submit_estimated_at_the_gain_floor_stays_under_the_watchdog() {
        let cases = [
            (default_budget_ms(true), true),
            (default_budget_ms(false), false),
            (40, true),
            (100, false),
            (600, true),
            (900, false),
        ];
        for (budget_ms, reset_prone) in cases {
            for timed in [true, false] {
                let m = WorkModel::new(budget_ms, prior_gflops(2), true, reset_prone);
                let unit = gemm_flops(64, 2048, 2048);
                let real_unit = m.estimate("k", Family::Gemv, unit);
                if timed {
                    for _ in 0..TIMED_WINDOW {
                        m.record("k", Family::Gemv, unit, 1);
                    }
                } else {
                    let mut work = SegmentWork::default();
                    work.add("k", Family::Gemv, unit);
                    m.calibrate(&work, 1);
                }
                let floor_unit = m.estimate("k", Family::Gemv, unit);
                // As many dispatches as a submit holds by estimate, and their real time.
                let held = (SUBMIT_BUDGETS * m.budget_ns()).div_ceil(floor_unit);
                let real = held * real_unit;
                let ceiling = if reset_prone {
                    MEASURED_LIMIT_NS * 3 / 4
                } else {
                    MEASURED_LIMIT_NS
                };
                // One dispatch of rounding: `held` rounds up to a whole dispatch.
                assert!(
                    real <= ceiling + real_unit,
                    "budget {budget_ms} ms, reset_prone {reset_prone}, timed {timed}: a full \
                     submit at the floor really runs {} ms",
                    real / MS
                );
            }
        }
        // Under a budget too large for any gain to be safe, nothing lowers an estimate: a fast
        // timed window leaves it at the prior, and a segment does not even list the kernel as
        // learned. A slow sample still raises it.
        let m = WorkModel::new(1500, prior_gflops(2), true, true);
        let unit = gemm_flops(64, 2048, 2048);
        let prior = m.estimate("k", Family::Gemv, unit);
        let mut work = SegmentWork::default();
        work.add("k", Family::Gemv, unit);
        m.calibrate(&work, 1);
        assert!(
            !learned(&m, "k"),
            "a kernel that did not move is listed as learned"
        );
        for _ in 0..TIMED_WINDOW {
            m.record("k", Family::Gemv, unit, 1);
        }
        assert_eq!(m.estimate("k", Family::Gemv, unit), prior);
        m.record("k", Family::Gemv, unit, 2 * prior);
        assert!(m.estimate("k", Family::Gemv, unit) > prior);
    }

    /// Lowering a kernel's rate takes `TIMED_WINDOW` fast readings in a row from its very first
    /// sample: the prior stands in for the samples it does not have yet. One fast reading alone
    /// would otherwise take the estimate to the gain floor, and where that is under the timing
    /// threshold the kernel would never be timed again. A first reading slower than the prior
    /// raises the estimate at once.
    #[test]
    fn a_first_fast_sample_does_not_lower_an_estimate_on_its_own() {
        let flops = gemm_flops(512, 2048, 22016);
        let est = |m: &WorkModel| m.estimate("dp4a", Family::Int8, flops);
        let m = igpu();
        let prior = est(&m);
        for sample in 1..TIMED_WINDOW {
            m.record("dp4a", Family::Int8, flops, 1);
            assert_eq!(est(&m), prior, "lowered after {sample} sample(s)");
            assert!(m.watch(est(&m)), "no longer timed after {sample} sample(s)");
        }
        m.record("dp4a", Family::Int8, flops, 1);
        assert!(
            est(&m) < prior,
            "a full window of fast readings changed nothing"
        );

        let slow = igpu();
        slow.record("dp4a", Family::Int8, flops, 2 * prior);
        assert!(est(&slow).abs_diff(2 * prior) <= 2, "{}", est(&slow));
    }

    /// A timed kernel's rate is the slowest of its last `TIMED_WINDOW` samples: a slower one
    /// raises the rate at once, and a faster one lowers it only when the slower ones have left
    /// the window.
    #[test]
    fn a_timed_rate_is_the_slowest_of_the_last_few_samples() {
        let m = igpu();
        let flops = gemm_flops(512, 2048, 22016);
        let prior = m.estimate("dp4a", Family::Int8, flops);
        let est = |m: &WorkModel| m.estimate("dp4a", Family::Int8, flops);
        for _ in 0..TIMED_WINDOW {
            m.record("dp4a", Family::Int8, flops, 150 * MS);
        }
        assert!(learned(&m, "dp4a"));
        assert!(est(&m) < prior);
        assert!(est(&m).abs_diff(150 * MS) < MS);
        // A faster sample does not lower it while the slower one is in the window; a slower one
        // raises it.
        m.record("dp4a", Family::Int8, flops, 130 * MS);
        assert!(est(&m).abs_diff(150 * MS) < MS);
        m.record("dp4a", Family::Int8, flops, 200 * MS);
        assert!(est(&m).abs_diff(200 * MS) < MS);
        // Per kernel: another kernel keeps its prior.
        assert!(!learned(&m, "mmq"));
    }

    /// One slow outlier (another program held the GPU during that dispatch) does not stay: the
    /// estimate is back at the kernel's usual time once `TIMED_WINDOW` later samples have pushed
    /// it out, and not before. The other direction cannot be reached by one reading: a single
    /// fast sample among usual ones changes nothing.
    #[test]
    fn a_slow_outlier_is_forgotten_after_a_window_of_samples() {
        let m = igpu();
        let flops = gemm_flops(512, 2048, 22016);
        let est = |m: &WorkModel| m.estimate("dp4a", Family::Int8, flops);
        let usual = 150 * MS;
        m.record("dp4a", Family::Int8, flops, usual);
        m.record("dp4a", Family::Int8, flops, 20 * usual);
        let high = est(&m);
        assert!(high > 2 * usual, "{} ms", high / MS);
        for _ in 1..TIMED_WINDOW {
            m.record("dp4a", Family::Int8, flops, usual);
            assert_eq!(est(&m), high, "the outlier left the window early");
        }
        m.record("dp4a", Family::Int8, flops, usual);
        assert!(est(&m).abs_diff(usual) < MS, "{} ms", est(&m) / MS);

        m.record("dp4a", Family::Int8, flops, usual * 2 / 3);
        assert!(est(&m).abs_diff(usual) < MS, "one fast sample lowered it");
    }

    /// A sample far on the slow side of the prior is held at `MAX_PRIOR_SLOWDOWN` times the
    /// prior, so a garbage reading cannot cut the kernel into one range per workgroup.
    #[test]
    fn a_sample_is_bounded_on_the_slow_side_too() {
        let m = igpu();
        let flops = gemm_flops(512, 2048, 22016);
        let prior = m.estimate("gemv", Family::Gemv, flops);
        m.record("gemv", Family::Gemv, flops, u64::MAX / 2);
        let est = m.estimate("gemv", Family::Gemv, flops);
        let bound = (prior as f64 * MAX_PRIOR_SLOWDOWN) as u64;
        assert!(est.abs_diff(bound) <= MAX_PRIOR_SLOWDOWN as u64, "{est} ns");
    }

    /// A timestamp counter narrower than 64 bits wraps, and an interval across the wrap is the
    /// difference under the counter's mask: taken over 64 bits it reads as nearly 2^64 ticks,
    /// which the model would take for the slowest sample a kernel ever had.
    #[test]
    fn an_interval_across_a_timestamp_wrap_is_read_under_the_valid_bits() {
        for bits in [32u32, 36, 48] {
            let top = 1u64 << bits;
            assert_eq!(ticks_between(top - 100, 400, bits), 500, "{bits} bits");
            assert_eq!(ticks_between(1000, 1500, bits), 500, "{bits} bits");
            // What the unmasked difference reads.
            assert!(400u64.wrapping_sub(top - 100) > u64::MAX / 2);
        }
        assert_eq!(ticks_between(u64::MAX - 99, 400, 64), 500);
        assert_eq!(ticks_between(7, 7, 36), 0);
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

    /// A user's budget wins: a larger one keeps a dispatch whole that the default would cut.
    #[test]
    fn a_user_budget_wins() {
        let flops = gemm_flops(512, 2048, 22016);
        let dflt = igpu();
        let est = dflt.estimate("g", Family::Gemv, flops);
        assert!(dflt.pieces(est, 172) > 1);
        let big = WorkModel::new(600_000 / 10, prior_gflops(2), true, true);
        assert_eq!(big.pieces(est, 172), 1);
    }
}
