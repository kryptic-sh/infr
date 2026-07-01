//! Opt-in execution profiler (`INFR_METAL_PROFILE=1`). Aggregates, across the whole run, the
//! per-op wall time and the commit+wait ("dispatch") wall vs actual GPU-active time, then prints a
//! summary on drop. This is the evidence for *where* the reference backend spends its time — the
//! per-op command-buffer barrier, not the arithmetic.

use std::collections::HashMap;
use std::time::Duration;

#[derive(Default)]
pub(crate) struct Profile {
    /// op name → (call count, total wall time spent in `run_op` for that op)
    per_op: HashMap<&'static str, (u64, Duration)>,
    /// total wall time inside `dispatch()` (commit + GPU schedule + wait), summed over all ops
    dispatch_wall: Duration,
    dispatch_count: u64,
    forwards: u64,
}

impl Profile {
    pub fn add_op(&mut self, name: &'static str, d: Duration) {
        let e = self.per_op.entry(name).or_default();
        e.0 += 1;
        e.1 += d;
    }

    pub fn add_dispatch(&mut self, wall: Duration) {
        self.dispatch_wall += wall;
        self.dispatch_count += 1;
    }

    pub fn add_forward(&mut self) {
        self.forwards += 1;
    }

    pub fn print_summary(&self) {
        if self.forwards == 0 {
            return;
        }
        let total: Duration = self.per_op.values().map(|(_, d)| *d).sum();
        let total_s = total.as_secs_f64().max(1e-9);
        let mut rows: Vec<_> = self.per_op.iter().collect();
        rows.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));

        eprintln!("\n── infr-metal profile ({} forwards) ──", self.forwards);
        eprintln!("{:<12} {:>8} {:>11} {:>7}", "op", "calls", "wall(ms)", "%");
        for (name, (calls, d)) in rows {
            let ms = d.as_secs_f64() * 1e3;
            let pct = 100.0 * d.as_secs_f64() / total_s;
            eprintln!("{name:<12} {calls:>8} {ms:>11.1} {pct:>6.1}%");
        }

        // The per-op wall above is CPU-side *encode* time (each op appends to the batch). The GPU
        // actually runs at flush (commit + wait), which the batch defers — so report the two
        // separately rather than as fractions of each other.
        let dwall = self.dispatch_wall.as_secs_f64();
        let f = self.forwards as f64;
        eprintln!(
            "── CPU encode: {:.1} ms total ({:.2} ms/forward)",
            total_s * 1e3,
            total_s * 1e3 / f
        );
        eprintln!(
            "── GPU (commit+wait): {:.1} ms total ({:.2} ms/forward) over {} command buffers ({:.2}/forward)",
            dwall * 1e3,
            dwall * 1e3 / f,
            self.dispatch_count,
            self.dispatch_count as f64 / f
        );
    }
}
