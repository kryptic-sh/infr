//! How much host memory a weight arena may take, and how much there is to take
//! (`docs/disk-streaming-plan.md` §7 question 3).
//!
//! The DRAM tier's arena is ANONYMOUS, non-evictable memory — the expensive kind backlog B30
//! measured — so its size cannot be a guess. Two separate concerns live here:
//!
//! - [`available_bytes`], a PLATFORM probe of what could be committed right now. It is deliberately
//!   allowed to answer "I do not know" rather than estimate, because an over-estimate here is an
//!   out-of-memory kill or a swap storm mid-generation, not a slow run.
//! - [`auto_arena_bytes`], the PURE arithmetic that turns that answer into a budget. Separate so
//!   the policy can be tested without a machine that happens to have the right amount of RAM free.

/// Host memory that could be committed right now, or `None` where this platform has no probe.
///
/// `None` is a real answer and callers must treat it as one: it means "do not auto-size", not
/// "assume zero" and not "assume plenty". The tier then stays off unless the user names a budget,
/// which is the conservative failure — a model that would have streamed simply does not, and says
/// so, instead of the process being killed part-way through a generation.
///
/// The probe itself, and what each platform's figure does and does not account for, lives in
/// [`infr_plat::mem`]. This drops the provenance that [`infr_plat::mem::available`] carries,
/// because everything below is PURE arithmetic that only needs the number; a caller reporting on
/// the figure should read the provenance from the seam instead.
pub fn available_bytes() -> Option<u64> {
    infr_plat::mem::available().map(|a| a.bytes)
}

/// Never take the last of the machine's memory: the larger of this and [`HEADROOM_FRACTION`] of
/// what is available is left alone, on top of what the caller says the run will hold besides the
/// arena (`committed`). A fixed floor matters on small hosts where a fraction rounds to nothing.
const HEADROOM_MIN: u64 = 1 << 30;

/// The share of available memory an automatic arena leaves to everything else on the machine.
/// Reciprocal — `available / HEADROOM_FRACTION`.
///
/// Half, because the arena is anonymous memory the OS cannot reclaim, and what it buys flattens
/// well before it takes everything: a dense model's arena only has to outlast the mapped sweep it
/// replaces, and a MoE model's expert hit rate grows far slower than the slots added to it. What
/// the rest of the machine loses does not flatten. The rule this replaced kept an eighth: on the
/// Windows box, with 43.97 GB available, it sized a 38.47 GB arena for DeepSeek V4-Flash, and
/// available memory fell from 42109 MB to 10957 MB within a minute, where a watchdog stopped the
/// run. A fixed margin would not scale instead: the test guard's
/// `UNIFIED_HOST_MARGIN` (`infr-llama/tests/cpu_backend.rs`) is what a shared development box owes
/// its other sessions, and as a floor it would turn the tier off on any host with less than that
/// available. On that box half is the larger of the two.
const HEADROOM_FRACTION: u64 = 2;

/// Below this an arena is not worth building: the tier costs a copy per streamed block, and a
/// budget this small holds so little of a model that the hit rate cannot pay for it.
const MIN_USEFUL: u64 = 256 << 20;

/// The arena budget to take, given what is available and what the run has already spoken for.
///
/// - `available` — from [`available_bytes`].
/// - `committed` — host bytes this run will already hold that `available` does not know about. On a
///   UNIFIED-memory device (iGPU, APU, Metal) the "VRAM" budget is carved out of this same physical
///   RAM, so passing it here is what stops the two tiers from spending the same bytes twice. Zero
///   on a discrete GPU, whose VRAM is a separate pool.
/// - `pageable` — total bytes of the weights that could be paged. Budgeting past this buys nothing:
///   every block would already be resident.
///
/// Returns `0` when nothing worth having is left, which callers treat as "stay on the mmap path".
pub fn auto_arena_bytes(available: u64, committed: u64, pageable: u64) -> u64 {
    let usable = spare_bytes(available, committed).min(pageable);
    if usable < MIN_USEFUL {
        return 0;
    }
    usable
}

/// The anonymous, non-reclaimable host memory a run may take on top of `committed` (what it already
/// holds that `available`, from [`available_bytes`], does not see): `available` less `committed`
/// less the headroom left to the rest of the machine, the larger of `HEADROOM_MIN` and
/// `available / HEADROOM_FRACTION`. The rule [`auto_arena_bytes`] sizes an arena by, without the
/// arena's own floor (`MIN_USEFUL` is about whether an arena is worth building, not about what the
/// host can spare), so a caller placing a buffer it needs takes the same headroom.
pub fn spare_bytes(available: u64, committed: u64) -> u64 {
    let headroom = HEADROOM_MIN.max(available / HEADROOM_FRACTION);
    available.saturating_sub(committed).saturating_sub(headroom)
}

/// What a caller should do about a host weight arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaPlan {
    /// Build an arena of this many bytes.
    Take(u64),
    /// Build a tier that CACHES NOTHING — see [`crate::hostpager::HostPager::stream_only`]. The
    /// blocks still come from explicit positioned reads rather than the GGUF mapping, which is the
    /// whole point on a unified-memory device: the arena above is already GPU-accessible RAM, so
    /// the only thing missing beneath it is a reader that does not go through a page cache
    /// evicting by recency.
    StreamOnly,
    /// Keep the zero-copy mmap path, for this reason. Every reason is something the caller should
    /// SAY — a run that quietly did not page when it needed to is the confusing case.
    Skip(Skip),
}

/// Why a host arena was not built. Distinguished rather than collapsed to `None` because the
/// caller's message differs per case, and "we cannot tell" must never read as "it fits".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
    /// The weights fit the memory available — mmap is zero-copy and strictly better.
    Fits,
    /// No host-memory probe on this platform, so nothing can be sized. See [`available_bytes`].
    NoProbe,
    /// Streaming is needed but too little memory is free to seat a useful arena.
    TooLittle,
    /// The user turned it off by name (`paging.dram = 0`).
    Disabled,
}

/// What the user's `paging.dram` asked for. A budget of ZERO is not "no budget" — it is the
/// explicit OFF switch, and it has to be distinguishable from unset now that unset means "size it
/// yourself". Without it there is no way to A/B the tier against the mmap path it replaces, and no
/// way for a user to refuse an arena on a machine where the automatic answer is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requested {
    /// Nothing set — size from what the host can spare.
    Auto,
    /// `paging.dram = 0` — keep the mmap path whatever the fit test says.
    Off,
    /// A budget named by the user, which wins over every automatic rung.
    Fixed(u64),
    /// `paging.dram_bypass` — no host cache at all: blocks are read from disk straight into the
    /// arena above. A size cannot express this, which is why it is its own state.
    Bypass,
}

impl Requested {
    /// Read a resolved `paging.dram` / `paging.dram_bypass` pair: bypass wins (it says something a
    /// size cannot), then `None` unset, `Some(0)` off, `Some(n)` fixed.
    pub fn from_config(v: Option<u64>, bypass: bool) -> Self {
        if bypass {
            return Self::Bypass;
        }
        match v {
            None => Self::Auto,
            Some(0) => Self::Off,
            Some(n) => Self::Fixed(n),
        }
    }
}

/// The arena plan for a run that has ALREADY been decided to stream — the Vulkan dense and MoE
/// tiers, whose callers only reach them once residency was rejected.
///
/// `explicit` (the user's `paging.dram`) always wins, including on unified memory: forcing this
/// path on hardware that would not choose it is how it gets tested at all.
///
/// The automatic size is the same half-of-available rule as the CPU's, with nothing committed
/// (the KV cache lives in VRAM). Deliberately the same: what the rule protects against is the
/// arena itself, anonymous memory the OS cannot reclaim, and a GPU run's arena is that just as
/// much as a CPU run's. What it costs was measured on 2026-11-16 on Qwen3-30B-A3B Q4_K_M, RX
/// 7900 XTX (`paging.cache` 2 GiB, 41 decoded tokens): the arenas the old eighth rule and this one
/// would size on a machine with 9 GB available, forced by `INFR_DRAM_CACHE`, decoded at 7.96 and
/// 7.53 tokens/s with host-tier hit rates of 53-55% and 37-39%.
pub fn streaming_arena_plan(
    explicit: Requested,
    available: Option<u64>,
    unified: bool,
    pageable: u64,
) -> ArenaPlan {
    match explicit {
        // `Bypass` outranks a size: it is the one that says "no host cache at all", which a
        // number cannot express. It exists so the unified-memory shape can be exercised on a
        // discrete GPU, which is the only hardware this is developed on.
        Requested::Bypass => return ArenaPlan::StreamOnly,
        Requested::Fixed(b) => return ArenaPlan::Take(b),
        Requested::Off => return ArenaPlan::Skip(Skip::Disabled),
        Requested::Auto => {}
    }
    if unified {
        return ArenaPlan::StreamOnly;
    }
    let Some(available) = available else {
        return ArenaPlan::Skip(Skip::NoProbe);
    };
    match auto_arena_bytes(available, 0, pageable) {
        0 => ArenaPlan::Skip(Skip::TooLittle),
        n => ArenaPlan::Take(n),
    }
}

/// The arena plan for a backend with no VRAM ladder to decide for it — the CPU one, which must ask
/// whether the weights fit host memory itself.
///
/// The extra rung over [`streaming_arena_plan`] is [`Skip::Fits`]: when the weights fit, the mmap
/// path is zero-copy and an arena could only add copies, so paging a model that fits would be a
/// regression. `explicit` still wins over that test in both directions.
///
/// `committed` is the host memory the run will hold besides the arena — its KV cache, the weights
/// that stay mapped, its activations. Both automatic decisions count it the same way:
///
/// - **Fits** when `pageable + committed <= available`: the page cache can then hold every weight
///   beside the run's own memory. The mapping needs no headroom on top, because page-cache pages
///   are reclaimable — the OS takes them back from the run under pressure, which it cannot do with
///   an arena.
/// - **Otherwise an arena** of [`auto_arena_bytes`]' size, which leaves half of what is available
///   untouched, since an arena is anonymous memory nobody can reclaim.
///
/// That makes the step at the boundary deliberate: a model a little too big for memory goes from
/// the whole page cache to an arena of about half of what is available. The page cache does not
/// degrade gently there — on a cyclic sweep a recency-evicting cache that is short by a little
/// misses on nearly every block (`docs/perf/results.md` measured decode 23-33x slower) — while an
/// arena with a cold-end insert keeps its share of the weights resident whatever its size, so the
/// smaller arena should still be the faster side (inferred from those two behaviours, not measured
/// at the boundary); and sizing it up to most of the machine near the
/// boundary is the starvation the half rule exists to prevent. An explicit budget is not reduced
/// by `committed`.
pub fn cpu_arena_plan(
    explicit: Requested,
    available: Option<u64>,
    committed: u64,
    pageable: u64,
) -> ArenaPlan {
    match explicit {
        Requested::Fixed(b) => return ArenaPlan::Take(b),
        Requested::Off => return ArenaPlan::Skip(Skip::Disabled),
        // Bypassing the host cache means "read straight into the tier above", and for the CPU
        // backend there IS no tier above — this arena is the only one. Reading through to nothing
        // would be a pure regression on the mapping, so the flag simply keeps the mmap path.
        Requested::Bypass => return ArenaPlan::Skip(Skip::Disabled),
        Requested::Auto => {}
    }
    let Some(available) = available else {
        return ArenaPlan::Skip(Skip::NoProbe);
    };
    if pageable.saturating_add(committed) <= available {
        return ArenaPlan::Skip(Skip::Fits);
    }
    match auto_arena_bytes(available, committed, pageable) {
        0 => ArenaPlan::Skip(Skip::TooLittle),
        n => ArenaPlan::Take(n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// What the host can spare a run, at known values: half of what is free above 2 GiB, 1 GiB
    /// kept below it, `committed` off the top, saturating at zero, and no worth-building floor (a
    /// few MiB spare is a few MiB, where an arena would be refused).
    #[test]
    fn spare_bytes_takes_the_headroom_and_nothing_else() {
        const MIB: u64 = 1 << 20;
        assert_eq!(spare_bytes(64 * GIB, 0), 32 * GIB);
        assert_eq!(spare_bytes(4 * GIB, 0), 2 * GIB);
        assert_eq!(spare_bytes(4 * GIB, 1800 * MIB), 248 * MIB);
        assert_eq!(spare_bytes(GIB + 8 * MIB, 0), 8 * MIB);
        assert_eq!(auto_arena_bytes(GIB + 8 * MIB, 0, u64::MAX), 0);
        assert_eq!(spare_bytes(GIB, 0), 0);
        assert_eq!(spare_bytes(GIB / 2, 0), 0);
        assert_eq!(spare_bytes(4 * GIB, 8 * GIB), 0);
    }

    /// Headroom is the point: the budget never equals what is available, however much there is.
    #[test]
    fn headroom_is_always_left() {
        for &avail in &[2 * GIB, 8 * GIB, 64 * GIB, 512 * GIB] {
            let got = auto_arena_bytes(avail, 0, u64::MAX);
            assert!(got < avail, "took all {avail} bytes");
            assert!(
                avail - got >= HEADROOM_MIN,
                "left less than the floor at {avail}: took {got}"
            );
        }
    }

    /// On a large host the fraction is what binds, not the floor: half of what is available is
    /// left alone.
    #[test]
    fn a_large_host_leaves_the_fraction() {
        let avail = 64 * GIB;
        assert_eq!(auto_arena_bytes(avail, 0, u64::MAX), 32 * GIB);
    }

    /// The rule by hand, on the figures that motivated it and at both ends of the scale.
    ///
    /// - The Windows box running DeepSeek V4-Flash: 43.97 GB available and 98.54 GB pageable. Half
    ///   is left (21.985 GB), and a 1 GiB `committed` comes off the rest: 43.97e9 - 21.985e9 -
    ///   1073741824 = 20911258176 bytes. The rule this replaced answered 38.47 GB here.
    /// - 4 GiB available: half (2 GiB) beats the 1 GiB floor, so 2 GiB is the arena.
    /// - 1.5 GiB available: half is 0.75 GiB, below the floor, so 1 GiB is left and 0.5 GiB taken.
    /// - 1.2 GiB available: the floor leaves 0.2 GiB, under `MIN_USEFUL`, so no arena.
    #[test]
    fn the_budget_by_hand() {
        assert_eq!(
            auto_arena_bytes(43_970_000_000, GIB, 98_540_000_000),
            20_911_258_176
        );
        assert_eq!(auto_arena_bytes(4 * GIB, 0, u64::MAX), 2 * GIB);
        assert_eq!(auto_arena_bytes(3 * GIB / 2, 0, u64::MAX), GIB / 2);
        assert_eq!(auto_arena_bytes(6 * GIB / 5, 0, u64::MAX), 0);
        // The CPU plan passes its committed bytes through; an explicit budget ignores them.
        assert_eq!(
            cpu_arena_plan(Requested::Auto, Some(43_970_000_000), GIB, 98_540_000_000),
            ArenaPlan::Take(20_911_258_176)
        );
        assert_eq!(
            cpu_arena_plan(
                Requested::Fixed(12 * GIB),
                Some(43_970_000_000),
                40 * GIB,
                u64::MAX
            ),
            ArenaPlan::Take(12 * GIB)
        );
    }

    /// Unified memory: the VRAM budget comes out of the same RAM, so it must reduce the arena
    /// one-for-one. Without this the two tiers each plan to use the same bytes.
    #[test]
    fn committed_bytes_reduce_the_budget_one_for_one() {
        let avail = 32 * GIB;
        let free = auto_arena_bytes(avail, 0, u64::MAX);
        let with = auto_arena_bytes(avail, 4 * GIB, u64::MAX);
        assert_eq!(
            free - with,
            4 * GIB,
            "committed bytes must come straight off"
        );
    }

    /// Never budget past what could actually be paged.
    #[test]
    fn the_pageable_total_is_a_ceiling() {
        assert_eq!(auto_arena_bytes(64 * GIB, 0, 3 * GIB), 3 * GIB);
    }

    /// **The flags must keep working.** A machine big enough to hold the model resident is exactly
    /// the machine streaming has to be tested on, so an explicit budget wins over every automatic
    /// rung — including the fits-in-RAM test, the no-probe case and unified memory.
    #[test]
    fn an_explicit_budget_always_wins() {
        let forced = Requested::Fixed(3 * GIB);
        // CPU: a model that comfortably fits is still paged when asked.
        assert_eq!(
            cpu_arena_plan(forced, Some(64 * GIB), 0, GIB),
            ArenaPlan::Take(3 * GIB)
        );
        // ...and with no probe at all.
        assert_eq!(
            cpu_arena_plan(forced, None, 0, GIB),
            ArenaPlan::Take(3 * GIB)
        );
        // Streaming tiers: honoured on unified memory, which auto-sizing declines.
        assert_eq!(
            streaming_arena_plan(forced, Some(64 * GIB), true, 40 * GIB),
            ArenaPlan::Take(3 * GIB)
        );
        assert_eq!(
            streaming_arena_plan(forced, None, false, 40 * GIB),
            ArenaPlan::Take(3 * GIB)
        );
    }

    /// A budget of ZERO turns the tier off by name, on both paths and whatever the automatic rungs
    /// would have decided. Without this there is no way to A/B the tier against the mmap path it
    /// replaces once auto-sizing turns it on by itself, and `0` would otherwise read as "unset".
    #[test]
    fn a_zero_budget_is_the_off_switch() {
        assert_eq!(Requested::from_config(Some(0), false), Requested::Off);
        assert_eq!(
            cpu_arena_plan(Requested::Off, Some(GIB), 0, 200 * GIB),
            ArenaPlan::Skip(Skip::Disabled)
        );
        assert_eq!(
            streaming_arena_plan(Requested::Off, Some(64 * GIB), false, 200 * GIB),
            ArenaPlan::Skip(Skip::Disabled)
        );
        // ...and it is distinct from unset, which on that same host DOES build one.
        assert!(matches!(
            streaming_arena_plan(Requested::Auto, Some(64 * GIB), false, 200 * GIB),
            ArenaPlan::Take(_)
        ));
    }

    #[test]
    fn from_config_maps_unset_and_sizes() {
        assert_eq!(Requested::from_config(None, false), Requested::Auto);
        assert_eq!(
            Requested::from_config(Some(42), false),
            Requested::Fixed(42)
        );
    }

    /// Requirement one: if it fits, everything stays resident. Paging a model that fits would add
    /// a copy per block over the zero-copy mapping and buy nothing.
    #[test]
    fn a_model_that_fits_is_not_paged() {
        assert_eq!(
            cpu_arena_plan(Requested::Auto, Some(64 * GIB), 0, 8 * GIB),
            ArenaPlan::Skip(Skip::Fits)
        );
        // Exactly fitting still counts as fitting.
        assert_eq!(
            cpu_arena_plan(Requested::Auto, Some(8 * GIB), 0, 8 * GIB),
            ArenaPlan::Skip(Skip::Fits)
        );
    }

    /// The fit test counts `committed` as the budget does: at the boundary and one byte either side
    /// of it, and the size of the step from the mapping to an arena there. 10 GiB available with
    /// 1 GiB committed: 9 GiB of weights fit; one byte more does not, and gets the half rule's
    /// arena, 10 - 5 - 1 = 4 GiB.
    #[test]
    fn the_fit_boundary_counts_committed_memory() {
        let (avail, committed) = (10 * GIB, GIB);
        let plan = |pageable| cpu_arena_plan(Requested::Auto, Some(avail), committed, pageable);
        assert_eq!(plan(9 * GIB - 1), ArenaPlan::Skip(Skip::Fits));
        assert_eq!(plan(9 * GIB), ArenaPlan::Skip(Skip::Fits));
        assert_eq!(plan(9 * GIB + 1), ArenaPlan::Take(4 * GIB));
        // Weights that fit only if the run's own memory is ignored do not fit.
        assert_eq!(plan(10 * GIB), ArenaPlan::Take(4 * GIB));
        // With nothing committed the boundary is the available memory itself.
        assert_eq!(
            cpu_arena_plan(Requested::Auto, Some(avail), 0, avail),
            ArenaPlan::Skip(Skip::Fits)
        );
    }

    /// Requirement two: a model that does NOT fit streams, without being asked to.
    #[test]
    fn a_model_that_does_not_fit_streams() {
        match cpu_arena_plan(Requested::Auto, Some(32 * GIB), 0, 200 * GIB) {
            ArenaPlan::Take(n) => assert!(n > 0 && n < 32 * GIB, "implausible budget {n}"),
            other => panic!("an over-sized model must stream, got {other:?}"),
        }
    }

    /// Unified memory reads DISK → GPU-accessible RAM with no host cache: the arena above is
    /// already in the one pool of RAM, so caching beneath it would hold a second copy the device
    /// cannot read in place — but the reads themselves must still be block-granular rather than
    /// left to the mapping, which is what `StreamOnly` expresses.
    #[test]
    fn unified_memory_streams_without_caching() {
        assert_eq!(
            streaming_arena_plan(Requested::Auto, Some(64 * GIB), true, 40 * GIB),
            ArenaPlan::StreamOnly
        );
        // It must NOT collapse to "keep the mmap path" — that is the thing it replaces.
        assert_ne!(
            streaming_arena_plan(Requested::Auto, Some(64 * GIB), true, 40 * GIB),
            ArenaPlan::Skip(Skip::Fits)
        );
        // A host with no memory to spare still streams on unified memory, because the decision
        // does not depend on having any to give.
        assert_eq!(
            streaming_arena_plan(Requested::Auto, Some(GIB), true, 40 * GIB),
            ArenaPlan::StreamOnly
        );
        // The same host WITHOUT unified memory caches instead — otherwise this test would pass
        // for a version that simply never auto-sizes.
        assert!(matches!(
            streaming_arena_plan(Requested::Auto, Some(64 * GIB), false, 40 * GIB),
            ArenaPlan::Take(_)
        ));
    }

    /// "Cannot tell" must never be reported as "fits": the two lead to opposite advice.
    #[test]
    fn no_probe_is_distinct_from_fitting() {
        assert_eq!(
            cpu_arena_plan(Requested::Auto, None, 0, 200 * GIB),
            ArenaPlan::Skip(Skip::NoProbe)
        );
        assert_eq!(
            streaming_arena_plan(Requested::Auto, None, false, 200 * GIB),
            ArenaPlan::Skip(Skip::NoProbe)
        );
    }

    /// A host with nothing to spare declines rather than building a useless arena.
    #[test]
    fn a_squeezed_host_declines() {
        assert_eq!(auto_arena_bytes(GIB, 0, u64::MAX), 0);
        assert_eq!(auto_arena_bytes(64 * GIB, 63 * GIB, u64::MAX), 0);
        // Just under the useful floor, with headroom accounted for.
        assert_eq!(auto_arena_bytes(2 * GIB, 0, MIN_USEFUL - 1), 0);
    }
}
