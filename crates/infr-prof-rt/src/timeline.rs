//! Per-span timeline recording for `INFR_PROF_TRACE` (Chrome Trace Event Format, `"ph":"X"`
//! slices). The exit report aggregates; this keeps the SEQUENCE, which is what makes gaps and
//! per-token shapes visible (open the file in ui.perfetto.dev or `chrome://tracing`).
//!
//! Hot path: `enter`/`Guard::drop` in [`crate`] push a 24-byte [`Event`] into the calling
//! thread's buffer. A thread hands its buffer to the shared vector when it reaches
//! [`FLUSH_EVENTS`] or when its outermost span closes, so memory stays bounded by one span's
//! worth of events per thread and the tail is written even though other threads' TLS cannot be
//! reached at exit.
//!
//! Cost when `INFR_PROF_TRACE` is unset: one relaxed load per span drop. Nothing is allocated.

use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

/// One completed span. `ts_ns` is relative to the process-wide profiling epoch, so the file is
/// small and no subtraction is needed at write time.
pub(crate) struct Event {
    pub tid: u32,
    pub site: u32,
    pub ts_ns: u64,
    pub dur_ns: u64,
}

/// Worst-case retained events (24 B each + vector slack). Past it the OLDEST half is dropped: a
/// long run keeps what you were looking at, not the warmup.
const MAX_EVENTS: usize = 1_500_000;

/// Per-thread buffer size that hands off to the shared vector mid-span.
pub(crate) const FLUSH_EVENTS: usize = 8192;

/// Spans shorter than this are not recorded. Instrumentation covers every function, so the
/// sub-microsecond majority would be most of the events (and most of the file) while telling
/// you nothing the aggregate report does not: a 72ns `Recorder::dispatch` row is bookkeeping,
/// not a gap you can act on. 5µs is above dispatch bookkeeping and below a real kernel.
pub(crate) const MIN_NS: u64 = 5_000;

static ENABLED: AtomicBool = AtomicBool::new(false);
static EVENTS: Mutex<Vec<Event>> = Mutex::new(Vec::new());
static TRACE_OUT: Mutex<Option<String>> = Mutex::new(None);
/// Events discarded by the [`MAX_EVENTS`] ring (oldest first).
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Install the destination path. Empty means "no trace", matching the `INFR_PROF_OUT=` grammar.
pub(crate) fn set_out(path: Option<&str>) {
    *TRACE_OUT.lock().unwrap() = path.map(str::to_owned);
    ENABLED.store(matches!(path, Some(p) if !p.is_empty()), Relaxed);
}

#[inline]
pub(crate) fn enabled() -> bool {
    ENABLED.load(Relaxed)
}

/// Append one thread's completed events, trimming the ring if it overflowed.
pub(crate) fn flush(buf: Vec<Event>) {
    if buf.is_empty() {
        return;
    }
    let mut g = EVENTS.lock().unwrap();
    g.extend(buf);
    let dropped = trim(&mut g);
    if dropped > 0 {
        DROPPED.fetch_add(dropped as u64, Relaxed);
    }
}

/// Drop the oldest half once the buffer is over [`MAX_EVENTS`]; returns how many went.
fn trim(g: &mut Vec<Event>) -> usize {
    if g.len() <= MAX_EVENTS {
        return 0;
    }
    let drop = g.len() - MAX_EVENTS / 2;
    g.drain(..drop);
    drop
}

pub(crate) fn has_events() -> bool {
    !EVENTS.lock().unwrap().is_empty()
}

/// Write the collected timeline, if `INFR_PROF_TRACE` named a path. `names` is the site
/// registry's `(module, fn name)` table, indexed by [`Event::site`].
pub(crate) fn write(names: &[(&'static str, &'static str)]) {
    let Some(path) = TRACE_OUT.lock().unwrap().clone().filter(|p| !p.is_empty()) else {
        return;
    };
    let mut events = std::mem::take(&mut *EVENTS.lock().unwrap());
    let dropped = DROPPED.swap(0, Relaxed);
    if events.is_empty() {
        // print-ok: the user asked for a trace with INFR_PROF_TRACE; silence would look like a bug.
        eprintln!(
            "INFR_PROF_TRACE={path}: no span events recorded (host spans need an INFR_PROFILE=1 build)"
        );
        return;
    }
    // Group by thread, then by time: Perfetto sorts tracks itself, but a deterministic file is
    // reviewable in a diff and the write is cache-friendly.
    events.sort_unstable_by_key(|e| (e.tid, e.ts_ns));
    match write_file(&path, &events, names) {
        // print-ok: profiling report requested by the user via INFR_PROF_TRACE.
        Ok(n) => eprintln!(
            "trace JSON written to {path} ({n} events, {dropped} oldest dropped) — open it in ui.perfetto.dev"
        ),
        // print-ok: the requested trace failed; the user must see why.
        Err(e) => eprintln!("failed to write INFR_PROF_TRACE={path}: {e}"),
    }
}

fn write_file(
    path: &str,
    events: &[Event],
    names: &[(&'static str, &'static str)],
) -> std::io::Result<usize> {
    // The same site repeats for every call; format and escape each name once.
    let full: Vec<String> = names
        .iter()
        .map(|(m, n)| esc(&format!("{m}::{n}")))
        .collect();
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    // `name` carries the module, so no `cat`: Chrome groups by `cat` only for filters, and
    // repeating it on every slice is a quarter of the file. `args` is omitted for the same reason.
    writeln!(f, "{{\"displayTimeUnit\":\"ms\",\"traceEvents\":[")?;
    writeln!(
        f,
        "{{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,\"args\":{{\"name\":\"infr\"}}}},"
    )?;
    let mut last_tid = u32::MAX;
    for (i, e) in events.iter().enumerate() {
        if e.tid != last_tid {
            last_tid = e.tid;
            writeln!(
                f,
                "{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":{last_tid},\"args\":{{\"name\":\"thr-{last_tid}\"}}}},"
            )?;
        }
        let name = full.get(e.site as usize).map_or("?", String::as_str);
        let comma = if i + 1 < events.len() { "," } else { "" };
        writeln!(
            f,
            "{{\"name\":\"{name}\",\"ph\":\"X\",\"ts\":{:.3},\"dur\":{:.3},\"pid\":1,\"tid\":{}}}{comma}",
            e.ts_ns as f64 / 1000.0,
            e.dur_ns as f64 / 1000.0,
            e.tid,
        )?;
    }
    writeln!(f, "]}}")?;
    f.flush()?;
    Ok(events.len())
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_keeps_the_tail() {
        let mut v: Vec<Event> = (0..MAX_EVENTS + 10)
            .map(|i| Event {
                tid: 0,
                site: 0,
                ts_ns: i as u64,
                dur_ns: 1,
            })
            .collect();
        let dropped = trim(&mut v);
        assert_eq!(dropped, MAX_EVENTS / 2 + 10, "trims back to half the cap");
        assert_eq!(v.len(), MAX_EVENTS / 2);
        assert_eq!(v.last().unwrap().ts_ns, (MAX_EVENTS + 9) as u64);
        assert_eq!(trim(&mut v), 0, "already under the cap");
    }

    #[test]
    fn write_file_emits_chrome_slices() {
        let dir = std::env::temp_dir().join(format!("infr-trace-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trace.json");
        let events = [
            Event {
                tid: 3,
                site: 0,
                ts_ns: 1_500,
                dur_ns: 2_000,
            },
            Event {
                tid: 3,
                site: 1,
                ts_ns: 9_000,
                dur_ns: 250,
            },
        ];
        let names = [("infr_x::m", "outer"), ("infr_x::m", "inner")];
        let n = write_file(path.to_str().unwrap(), &events, &names).unwrap();
        assert_eq!(n, 2);
        let s = std::fs::read_to_string(&path).unwrap();
        assert!(s.starts_with("{\"displayTimeUnit\":\"ms\",\"traceEvents\":["));
        assert!(s.contains("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":3"));
        assert!(s.contains(
            "{\"name\":\"infr_x::m::outer\",\"ph\":\"X\",\"ts\":1.500,\"dur\":2.000,\"pid\":1,\"tid\":3"
        ));
        assert!(s.trim_end().ends_with("]}"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
