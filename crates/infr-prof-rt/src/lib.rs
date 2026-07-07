//! Runtime for the `INFR_PROFILE=1` build-time instrumentation (see `infr-prof` and
//! docs/PERF.md § "Build-time auto-instrumentation").
//!
//! `#[infr_prof::instrument]` rewrites every `fn` it covers to open a [`Site`]-keyed span at
//! entry via [`enter`]; the returned RAII [`Guard`] closes the span on any exit path (`?`,
//! `return`, panic unwind). Accounting is strictly thread-local on the hot path — a
//! `thread_local!` span stack plus a per-thread fixed-size accumulator table — so there is no
//! shared-state contention. Accumulator slots are `AtomicU64`s written with plain
//! load-then-store `Relaxed` ops (single writer: the owning thread); the atomics exist only so
//! the exit-time reporter may read tables of threads that are still alive (rayon workers never
//! unwind their TLS). The merged report prints to stderr at process exit (`atexit`), sorted by
//! self time, and is also written as JSON to `$INFR_PROFILE_OUT` if set.
//!
//! Metrics per site: call count, inclusive total (recursion-aware: only the outermost frame of
//! a site on a given thread adds to total), self time (inclusive minus instrumented children),
//! and average self per call.
//!
//! This crate compiles to dead code in default builds — nothing calls it unless a crate was
//! built with the `infr_profile` cfg (emitted by build.rs when `INFR_PROFILE=1`).

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Hard cap on distinct instrumented call sites. Sites past the cap are counted but not timed
/// (reported as `[dropped]`). 8192 * 24 B = 192 KiB per thread — cheap for a profiling build.
const MAX_SITES: usize = 8192;

/// One static per instrumented `fn`, created by the `#[infr_prof::instrument]` expansion.
/// The id is assigned lazily on first call (relaxed fast path, mutex slow path).
pub struct Site {
    module: &'static str,
    name: &'static str,
    /// 0 = unassigned; otherwise site index + 1.
    id: AtomicU32,
}

impl Site {
    pub const fn new(module: &'static str, name: &'static str) -> Self {
        Site {
            module,
            name,
            id: AtomicU32::new(0),
        }
    }

    #[inline]
    fn id(&'static self) -> u32 {
        let id = self.id.load(Relaxed);
        if id != 0 {
            return id - 1;
        }
        self.register()
    }

    #[cold]
    fn register(&'static self) -> u32 {
        let g = global();
        let mut names = g.names.lock().unwrap();
        // Double-check under the lock (another thread may have registered this site).
        let id = self.id.load(Relaxed);
        if id != 0 {
            return id - 1;
        }
        let idx = names.len() as u32;
        names.push((self.module, self.name));
        self.id.store(idx + 1, Relaxed);
        idx
    }
}

struct Slot {
    count: AtomicU64,
    total_ns: AtomicU64,
    self_ns: AtomicU64,
}

/// Per-thread accumulator table, shared with the global registry so the exit reporter can read
/// it while worker threads (rayon pools, spin pools) are still parked.
struct AccumTable {
    slots: Box<[Slot]>,
}

impl AccumTable {
    fn new() -> Self {
        let mut v = Vec::with_capacity(MAX_SITES);
        for _ in 0..MAX_SITES {
            v.push(Slot {
                count: AtomicU64::new(0),
                total_ns: AtomicU64::new(0),
                self_ns: AtomicU64::new(0),
            });
        }
        AccumTable {
            slots: v.into_boxed_slice(),
        }
    }
}

/// Single-writer add: plain load+store (a `mov`, not `lock xadd`) — the owning thread is the
/// only writer; the reporter only reads.
#[inline]
fn add(a: &AtomicU64, v: u64) {
    a.store(a.load(Relaxed).wrapping_add(v), Relaxed);
}

struct Frame {
    id: u32,
    start: Instant,
    /// Nanoseconds spent in instrumented callees of this frame (for self-time).
    child_ns: u64,
}

struct ThreadProf {
    table: Arc<AccumTable>,
    stack: Vec<Frame>,
    /// Per-site recursion depth on this thread (inclusive total only counts the outermost).
    depth: Vec<u32>,
}

impl ThreadProf {
    fn new() -> Self {
        let table = Arc::new(AccumTable::new());
        global().tables.lock().unwrap().push(table.clone());
        ThreadProf {
            table,
            stack: Vec::with_capacity(64),
            depth: Vec::new(),
        }
    }
}

thread_local! {
    static TP: RefCell<ThreadProf> = RefCell::new(ThreadProf::new());
}

struct Global {
    names: Mutex<Vec<(&'static str, &'static str)>>,
    tables: Mutex<Vec<Arc<AccumTable>>>,
    start: Instant,
}

static GLOBAL: OnceLock<Global> = OnceLock::new();

fn global() -> &'static Global {
    GLOBAL.get_or_init(|| {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        extern "C" fn report_at_exit() {
            report();
        }
        unsafe {
            atexit(report_at_exit);
        }
        Global {
            names: Mutex::new(Vec::new()),
            tables: Mutex::new(Vec::new()),
            start: Instant::now(),
        }
    })
}

/// Open a span for `site` on the current thread. Returns an RAII guard; span closes when the
/// guard drops (any exit path). Must be strictly LIFO per thread — guaranteed because the guard
/// is a local of the instrumented fn.
#[inline]
pub fn enter(site: &'static Site) -> Guard {
    let id = site.id();
    // try_with: during thread teardown the TLS is gone — record nothing rather than panic.
    let active = TP
        .try_with(|tp| {
            let mut tp = tp.borrow_mut();
            let idx = id as usize;
            if tp.depth.len() <= idx {
                tp.depth.resize(idx + 1, 0);
            }
            tp.depth[idx] += 1;
            tp.stack.push(Frame {
                id,
                start: Instant::now(),
                child_ns: 0,
            });
        })
        .is_ok();
    Guard { active }
}

pub struct Guard {
    active: bool,
}

impl Drop for Guard {
    #[inline]
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let end = Instant::now();
        let _ = TP.try_with(|tp| {
            let mut tp = tp.borrow_mut();
            let Some(frame) = tp.stack.pop() else { return };
            let elapsed = end.duration_since(frame.start).as_nanos() as u64;
            let self_ns = elapsed.saturating_sub(frame.child_ns);
            let idx = frame.id as usize;
            tp.depth[idx] -= 1;
            let outermost = tp.depth[idx] == 0;
            if let Some(parent) = tp.stack.last_mut() {
                parent.child_ns += elapsed;
            }
            if idx < MAX_SITES {
                let slot = &tp.table.slots[idx];
                add(&slot.count, 1);
                add(&slot.self_ns, self_ns);
                if outermost {
                    add(&slot.total_ns, elapsed);
                }
            }
        });
    }
}

struct Row {
    name: String,
    count: u64,
    total_ns: u64,
    self_ns: u64,
}

fn collect() -> (Vec<Row>, usize, u64) {
    let g = global();
    let names = g.names.lock().unwrap().clone();
    let tables = g.tables.lock().unwrap();
    let n_threads = tables.len();
    let wall_ns = g.start.elapsed().as_nanos() as u64;
    let mut rows: Vec<Row> = names
        .iter()
        .map(|(m, n)| Row {
            name: format!("{m}::{n}"),
            count: 0,
            total_ns: 0,
            self_ns: 0,
        })
        .collect();
    for t in tables.iter() {
        for (i, row) in rows.iter_mut().enumerate() {
            if i >= MAX_SITES {
                break;
            }
            let s = &t.slots[i];
            row.count += s.count.load(Relaxed);
            row.total_ns += s.total_ns.load(Relaxed);
            row.self_ns += s.self_ns.load(Relaxed);
        }
    }
    rows.retain(|r| r.count > 0);
    rows.sort_by_key(|r| std::cmp::Reverse(r.self_ns));
    (rows, n_threads, wall_ns)
}

fn fmt_ns(ns: u64) -> String {
    if ns >= 10_000_000_000 {
        format!("{:.1}s", ns as f64 / 1e9)
    } else if ns >= 1_000_000_000 {
        format!("{:.2}s", ns as f64 / 1e9)
    } else if ns >= 1_000_000 {
        format!("{:.1}ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.1}us", ns as f64 / 1e3)
    } else {
        format!("{ns}ns")
    }
}

/// Print the merged profile to stderr (top entries by self time) and, if `INFR_PROFILE_OUT` is
/// set, write the full table as JSON to that path. Runs automatically at process exit; may also
/// be called on demand.
pub fn report() {
    // atexit may fire in odd states; never run twice.
    static DONE: AtomicBool = AtomicBool::new(false);
    if DONE.swap(true, Relaxed) {
        return;
    }
    if GLOBAL.get().is_none() {
        return;
    }
    let (rows, n_threads, wall_ns) = collect();
    if rows.is_empty() {
        return;
    }
    let accounted: u64 = rows.iter().map(|r| r.self_ns).sum();
    eprintln!();
    eprintln!(
        "== INFR_PROFILE report: {} sites, {} threads, wall {} (since first instrumented call), accounted self {} ==",
        rows.len(),
        n_threads,
        fmt_ns(wall_ns),
        fmt_ns(accounted),
    );
    eprintln!(
        "{:>12} {:>7} {:>12} {:>12} {:>10}  function",
        "self", "self%", "total", "calls", "avg(self)"
    );
    const TOP: usize = 50;
    for r in rows.iter().take(TOP) {
        eprintln!(
            "{:>12} {:>6.2}% {:>12} {:>12} {:>10}  {}",
            fmt_ns(r.self_ns),
            100.0 * r.self_ns as f64 / wall_ns.max(1) as f64,
            fmt_ns(r.total_ns),
            r.count,
            fmt_ns(r.self_ns / r.count.max(1)),
            r.name
        );
    }
    if rows.len() > TOP {
        eprintln!(
            "  ... {} more sites (set INFR_PROFILE_OUT=path for the full table as JSON)",
            rows.len() - TOP
        );
    }
    if let Ok(path) = std::env::var("INFR_PROFILE_OUT") {
        if !path.is_empty() {
            match write_json(&path, &rows, n_threads, wall_ns) {
                Ok(()) => eprintln!("profile JSON written to {path}"),
                Err(e) => eprintln!("failed to write INFR_PROFILE_OUT={path}: {e}"),
            }
        }
    }
}

fn write_json(path: &str, rows: &[Row], n_threads: usize, wall_ns: u64) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "{{")?;
    writeln!(f, "  \"wall_ns\": {wall_ns},")?;
    writeln!(f, "  \"threads\": {n_threads},")?;
    writeln!(f, "  \"sites\": [")?;
    for (i, r) in rows.iter().enumerate() {
        let name = r.name.replace('\\', "\\\\").replace('"', "\\\"");
        let comma = if i + 1 < rows.len() { "," } else { "" };
        writeln!(
            f,
            "    {{\"name\": \"{name}\", \"calls\": {}, \"total_ns\": {}, \"self_ns\": {}}}{comma}",
            r.count, r.total_ns, r.self_ns
        )?;
    }
    writeln!(f, "  ]")?;
    writeln!(f, "}}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nesting_and_recursion() {
        static OUTER: Site = Site::new("m", "outer");
        static INNER: Site = Site::new("m", "inner");
        static REC: Site = Site::new("m", "rec");

        fn rec(n: u32) {
            let _g = enter(&REC);
            std::thread::sleep(std::time::Duration::from_millis(2));
            if n > 0 {
                rec(n - 1);
            }
        }
        {
            let _o = enter(&OUTER);
            {
                let _i = enter(&INNER);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        rec(2);

        let (rows, _, _) = collect();
        let get = |n: &str| rows.iter().find(|r| r.name == format!("m::{n}")).unwrap();
        let outer = get("outer");
        let inner = get("inner");
        let rec = get("rec");
        assert_eq!(outer.count, 1);
        assert_eq!(inner.count, 1);
        assert_eq!(rec.count, 3);
        // outer's self excludes inner's time
        assert!(outer.self_ns < outer.total_ns);
        assert!(inner.total_ns >= 5_000_000);
        assert!(outer.total_ns >= inner.total_ns);
        // recursion: inclusive total counted once at the outermost frame (~6ms, not ~12ms
        // double-counted); self sums each level's own ~2ms.
        assert!(rec.total_ns >= 6_000_000 && rec.total_ns < 11_000_000);
        assert!(rec.self_ns >= 6_000_000 && rec.self_ns <= rec.total_ns);
    }
}
