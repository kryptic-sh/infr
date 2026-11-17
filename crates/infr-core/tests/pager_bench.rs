//! Pager replay benchmark and eviction-policy report (`docs/plan/06-pager-eviction.md`, step 1).
//!
//! The tests are `#[ignore]`d because they report timings rather than pass or fail on them, and
//! timings are only worth reading from a release build run alone on an otherwise quiet machine:
//!
//! ```text
//! cargo test --release -p infr-core --test pager_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `generated_traces` needs nothing but the CPU: its traces are built by
//! [`infr_core::pager::shapes`]. `recorded_traces` and `recorded_trace_counters` read the files
//! named in `INFR_TEST_PAGER_TRACE` (separated by the platform's path-list separator), as written
//! by a run with `paging.trace` set.
//!
//! Before a trace's row is printed, its replay is checked against its recording (when it has
//! one), the simulator's model of today's policy against the real pager, and Belady against both.
//! Per trace, one row: the pool's blocks (the id space its header names), the distinct ids the
//! trace touched, its slots, the calls replayed, the hit rate, nanoseconds per call / hit / miss
//! under today's [`Pager`], misses under today's policy and under Belady's optimum (the gap is the
//! most any eviction policy could save), and the share of pager time a miss costs beyond a hit —
//! the victim scan, which the data-structure refactor would remove.

use infr_core::pager::shapes::{cyclic_sweep, Moe};
use infr_core::pager::sim::{
    belady_is_a_bound, expand_groups, simulate, simulate_split, Belady, Recency,
};
use infr_core::pager::trace::{self, Call, Outcome, PoolTrace};
use infr_core::pager::Resolution;
use infr_core::Pager;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Replays per trace: at least this many, and more until this much time has been spent.
const MIN_REPS: usize = 5;
const MIN_TIME: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Hit,
    Miss,
    Other,
}

fn class(o: Outcome, call: Call) -> Class {
    match o {
        Outcome::Resolved(Resolution::Hit { .. }) => Class::Hit,
        Outcome::Resolved(Resolution::Miss { .. }) => Class::Miss,
        // A probe that finds its block counts as a hit, as `Pager::pin_if_resident` counts it.
        Outcome::Slot(Some(_)) if matches!(call, Call::PinIfResident(_)) => Class::Hit,
        _ => Class::Other,
    }
}

fn replay_once(t: &PoolTrace) -> Duration {
    let mut p = Pager::new(t.n_slots);
    let start = Instant::now();
    for &c in &t.calls {
        black_box(c.apply(black_box(&mut p)));
    }
    start.elapsed()
}

/// Median nanoseconds to replay `t.calls[from..]` into a pager that has served `t.calls[..from]`
/// untimed.
fn time_after(t: &PoolTrace, from: usize) -> f64 {
    let mut runs = Vec::new();
    let begun = Instant::now();
    while runs.len() < MIN_REPS || begun.elapsed() < MIN_TIME {
        let mut p = Pager::new(t.n_slots);
        for &c in &t.calls[..from] {
            c.apply(&mut p);
        }
        let start = Instant::now();
        for &c in &t.calls[from..] {
            black_box(c.apply(black_box(&mut p)));
        }
        runs.push(start.elapsed().as_nanos() as f64);
    }
    runs.sort_by(f64::total_cmp);
    runs[runs.len() / 2]
}

/// What one `Instant::now()` costs here, so a segment's clock read can be taken back out.
fn clock_overhead() -> f64 {
    const N: u32 = 1_000_000;
    let start = Instant::now();
    for _ in 0..N {
        black_box(Instant::now());
    }
    start.elapsed().as_nanos() as f64 / N as f64
}

struct Timing {
    ns_call: f64,
    ns_hit: f64,
    ns_miss: f64,
    /// One replay's total by the segmented clock — the denominator the split is a share of.
    ns_split_total: f64,
}

/// Whole-trace time per call (median over repetitions), and its split into hits and misses.
///
/// The split reads the clock only where the outcome class changes between consecutive calls
/// (known in advance from an untimed replay, since replay is deterministic), so long runs of hits
/// or misses are timed as one segment; each segment's clock read is subtracted. On a platform
/// whose clock ticks coarsely (100 ns on Windows) a single short segment is quantised, but the
/// error is unbiased and averages out over the thousands of segments a trace has.
fn time_trace(t: &PoolTrace, classes: &[Class], overhead: f64) -> Timing {
    let mut whole = Vec::new();
    let begun = Instant::now();
    while whole.len() < MIN_REPS || begun.elapsed() < MIN_TIME {
        whole.push(replay_once(t).as_nanos() as f64);
    }
    whole.sort_by(f64::total_cmp);
    let reps = whole.len();
    let ns_call = whole[reps / 2] / t.calls.len() as f64;

    // Segment boundaries: the index where each run of one class ends.
    let mut ends = Vec::new();
    for i in 1..classes.len() {
        if classes[i] != classes[i - 1] {
            ends.push(i);
        }
    }
    ends.push(classes.len());
    let mut acc = [0f64; 3];
    for _ in 0..reps {
        let mut p = Pager::new(t.n_slots);
        let mut from = 0;
        let mut last = Instant::now();
        for &end in &ends {
            for &c in &t.calls[from..end] {
                black_box(c.apply(black_box(&mut p)));
            }
            let now = Instant::now();
            let k = classes[from] as usize;
            acc[k] += (now - last).as_nanos() as f64 - overhead;
            last = now;
            from = end;
        }
    }
    let count = |k: Class| classes.iter().filter(|&&c| c == k).count() as f64 * reps as f64;
    let per = |k: Class| {
        let n = count(k);
        if n == 0.0 {
            f64::NAN
        } else {
            acc[k as usize].max(0.0) / n
        }
    };
    Timing {
        ns_call,
        ns_hit: per(Class::Hit),
        ns_miss: per(Class::Miss),
        ns_split_total: acc.iter().map(|a| a.max(0.0)).sum::<f64>() / reps as f64,
    }
}

fn header() {
    println!(
        "| trace | blocks | ids seen | slots | calls | hit rate | ns/call | ns/hit | ns/miss | misses: current \
         | misses: Belady | gap | miss-excess share |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
}

/// One table row for `t`. A recorded trace is first checked against its recording, and the
/// simulator's model of today's policy against the real pager, before anything is reported.
fn report(t: &PoolTrace, overhead: f64) {
    let stats = trace::replay(t).unwrap_or_else(|d| panic!("{}: replay diverged: {d}", t.label));
    let cur = simulate(t.n_slots, &t.calls, &mut Recency::default());
    assert_eq!(
        (cur.hits, cur.misses, cur.evictions),
        (stats.hits, stats.misses, stats.evictions),
        "{}: the simulator's model of today's policy disagrees with the pager",
        t.label
    );
    assert_eq!(
        cur.refused, 0,
        "{}: today's policy refused an admission",
        t.label
    );
    let opt = belady(t);

    let mut p = Pager::new(t.n_slots);
    let classes: Vec<Class> = t.calls.iter().map(|&c| class(c.apply(&mut p), c)).collect();
    let tm = time_trace(t, &classes, overhead);
    // With no hits there is no hit cost to measure a miss's excess against.
    let excess = match (stats.misses, tm.ns_hit.is_nan()) {
        (0, _) => Some(0.0),
        (_, true) => None,
        (m, false) => Some(m as f64 * (tm.ns_miss - tm.ns_hit).max(0.0) / tm.ns_split_total),
    };
    let ns = |v: f64, digits: usize| {
        if v.is_nan() {
            "—".to_string()
        } else {
            format!("{v:.digits$}")
        }
    };
    let (opt_misses, gap) = match opt {
        Some(o) => (o.to_string(), format!("{:.1}%", gap(cur.misses, o) * 100.0)),
        None => ("—".to_string(), "—".to_string()),
    };
    let ids: std::collections::HashSet<_> = t.calls.iter().filter_map(|c| c.block()).collect();
    println!(
        "| {} | {} | {} | {} | {} | {:.1}% | {:.1} | {} | {} | {} | {} | {} | {} |",
        t.label,
        t.n_blocks,
        ids.len(),
        t.n_slots,
        t.calls.len(),
        stats.hit_rate() * 100.0,
        tm.ns_call,
        ns(tm.ns_hit, 1),
        ns(tm.ns_miss, 0),
        cur.misses,
        opt_misses,
        gap,
        excess.map_or("—".to_string(), |e| format!("{:.1}%", e * 100.0)),
    );
}

/// Belady's misses on `t`, or `None` (said loudly) where they would not be a lower bound: a trace
/// with probes or repins (`sim::belady_is_a_bound`), or a run in which Belady had to refuse an
/// admission — a refusal is not counted as a miss, so it would read as a saved one.
fn belady(t: &PoolTrace) -> Option<u64> {
    if !belady_is_a_bound(&t.calls) {
        println!(
            "NOTE {}: probes/repins/evicts in the trace, so Belady is not a bound here; no figure",
            t.label
        );
        return None;
    }
    let cur = simulate(t.n_slots, &t.calls, &mut Recency::default());
    let opt = simulate(t.n_slots, &t.calls, &mut Belady::default());
    if opt.refused != 0 {
        println!(
            "WARNING {}: Belady refused {} admissions; no figure",
            t.label, opt.refused
        );
        return None;
    }
    assert!(
        opt.misses <= cur.misses,
        "{}: Belady above current",
        t.label
    );
    Some(opt.misses)
}

fn gap(current: u64, optimum: u64) -> f64 {
    if current == 0 {
        0.0
    } else {
        (current - optimum) as f64 / current as f64
    }
}

/// The generated shapes at the sizes real models give their pools.
#[test]
#[ignore = "timing: run with --ignored --nocapture in a quiet window"]
fn generated_traces() {
    let overhead = clock_overhead();
    println!("clock read: {overhead:.1} ns");
    header();
    // Dense streaming: a pool of about forty streamed groups.
    report(&cyclic_sweep(40, 8, 32), overhead);
    report(&cyclic_sweep(40, 32, 32), overhead);

    let qwen = Moe::QWEN3_30B_A3B;
    // DeepSeek-scale: about eleven thousand blocks per role.
    let deep = Moe {
        n_layers: 43,
        n_expert: 256,
        top_k: 8,
        zipf_s: 1.0,
    };
    let mut resident_per_token = Vec::new();
    for m in [qwen, deep] {
        let n = m.n_blocks();
        for slots in [n / 4, n / 2, 3 * n / 4] {
            report(&m.prefill(slots, 2), overhead);
        }
        // Resident decode, as each token's per-expert touches made it and as its group touches do
        // now. Both traces start with the same token touching every expert, which admits them
        // all; only the tokens after it are timed.
        let tokens = 8;
        let (per_expert, grouped) = (
            m.decode_resident(1 + tokens),
            m.decode_resident_grouped(tokens),
        );
        report(&per_expert, overhead);
        report(&grouped, overhead);
        let admitting = m.decode_resident(1).calls.len();
        let per_token = |t: &PoolTrace| time_after(t, admitting) / tokens as f64;
        resident_per_token.push((m, per_token(&per_expert), per_token(&grouped)));
        for slots in [n / 4, n / 2, 3 * n / 4] {
            report(&m.decode_routed(slots, 64, 1), overhead);
        }
        report(&m.session(n / 2, 64, 1), overhead);
    }
    println!();
    println!("| resident decode, one pool | pager µs/token: touch every expert | touch each layer as a group |");
    println!("|---|---:|---:|");
    for (m, per_expert, grouped) in resident_per_token {
        println!(
            "| {}x{} | {:.2} | {:.3} |",
            m.n_layers,
            m.n_expert,
            per_expert / 1e3,
            grouped / 1e3
        );
    }
}

/// Every pool that made a call, from the files named in `INFR_TEST_PAGER_TRACE`.
fn recorded() -> Vec<PoolTrace> {
    let paths = std::env::var_os("INFR_TEST_PAGER_TRACE")
        .expect("set INFR_TEST_PAGER_TRACE to one or more recorded trace files");
    let mut out = Vec::new();
    for path in std::env::split_paths(&paths) {
        let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let parsed = trace::parse(std::io::BufReader::new(file))
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        if let Some(t) = &parsed.truncated {
            println!("WARNING {}: {t}", path.display());
        }
        let parsed = parsed.pools;
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        for mut t in parsed.into_iter().filter(|t| !t.calls.is_empty()) {
            t.label = format!("{} [{name}]", t.label);
            out.push(t);
        }
    }
    assert!(!out.is_empty(), "no pool in the named traces made a call");
    out
}

/// Recorded traces (`paging.trace`), named in `INFR_TEST_PAGER_TRACE`.
#[test]
#[ignore = "timing: run with --ignored --nocapture in a quiet window"]
fn recorded_traces() {
    let pools = recorded();
    let overhead = clock_overhead();
    println!("clock read: {overhead:.1} ns");
    header();
    for t in &pools {
        report(t, overhead);
    }
}

/// The replayed counters of every recorded pool, in the shape `paging.stats` prints them, to set
/// beside the run's own lines: equal counts are the proof the trace saw every call.
#[test]
#[ignore = "needs recorded traces: set INFR_TEST_PAGER_TRACE"]
fn recorded_trace_counters() {
    for t in recorded() {
        let s = trace::replay(&t).unwrap_or_else(|d| panic!("{}: replay diverged: {d}", t.label));
        println!(
            "{} ({}): hits={} misses={} evictions={} hit_rate={:.3} slots={}",
            t.label,
            match t.tier {
                trace::Tier::Vram => "vram",
                trace::Tier::Dram => "dram",
            },
            s.hits,
            s.misses,
            s.evictions,
            s.hit_rate(),
            t.n_slots,
        );
    }
}

/// For each recorded pool with a batched prefill (`touch_cold` calls): today's and Belady's misses
/// before and after the last prefill call, from one run each, and how the decode-phase batches
/// split between touch-all ones (every expert of a layer MRU-touched: `touch_all_hits`) and
/// routed ones. Counts only; no timing.
#[test]
#[ignore = "needs recorded traces: set INFR_TEST_PAGER_TRACE"]
fn recorded_phase_split() {
    // Group touches are counted as the touches of every member they stand for, so a touch-all
    // batch reads the same whether it was recorded before or after group touches landed.
    println!(
        "| pool | prefill misses: current / Belady (gap) | decode misses: current / Belady (gap)          | decode batches: touch-all / routed |"
    );
    println!("|---|---|---|---|");
    for mut t in recorded() {
        t.calls = expand_groups(&t.calls);
        t.outcomes.clear();
        let Some(last_cold) = t
            .calls
            .iter()
            .rposition(|c| matches!(c, Call::TouchCold(_)))
        else {
            continue;
        };
        let split = last_cold + 1;
        let [cp, cd] = simulate_split(t.n_slots, &t.calls, &mut Recency::default(), split);
        // Signed: Belady can spend more misses in one phase to save them in the other.
        let phase = |c: u64, o: Option<u64>| match o {
            Some(o) if c > 0 => format!(
                "{c} / {o} ({:+.1}%)",
                (c as f64 - o as f64) / c as f64 * 100.0
            ),
            Some(o) => format!("{c} / {o}"),
            None => format!("{c} / —"),
        };
        let (bp, bd) = if belady(&t).is_some() {
            let [p, d] = simulate_split(t.n_slots, &t.calls, &mut Belady::default(), split);
            (Some(p.misses), Some(d.misses))
        } else {
            (None, None)
        };
        // A layer's batch size tells the two decode paths apart: touch-all touches every expert
        // (as many as a prefill batch cold-touches), the routed path only the router's picks.
        let batch_sizes = |calls: &[Call]| {
            let mut sizes: Vec<usize> = Vec::new();
            for c in calls {
                match c {
                    Call::BeginBatch => sizes.push(0),
                    Call::Touch(_) | Call::TouchCold(_) => {
                        if let Some(n) = sizes.last_mut() {
                            *n += 1;
                        }
                    }
                    _ => {}
                }
            }
            sizes.retain(|&n| n > 0);
            sizes
        };
        let n_expert = batch_sizes(&t.calls[..split])
            .into_iter()
            .max()
            .unwrap_or(0);
        let decode = batch_sizes(&t.calls[split..]);
        let all = decode.iter().filter(|&&n| n == n_expert).count();
        println!(
            "| {} | {} | {} | {all} (of {n_expert}) / {} |",
            t.label,
            phase(cp.misses, bp),
            phase(cd.misses, bd),
            decode.len() - all,
        );
    }
}
