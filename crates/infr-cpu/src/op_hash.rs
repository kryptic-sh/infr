//! The per-op hash tap (`debug.cpu_op_hash`, `INFR_CPU_OP_HASH=<path>`): a bisecting aid that
//! appends, for every op the interpreter runs, one line per tensor the op READS (before it runs)
//! and one per tensor it WRITES (after it ran), each with an FNV-1a hash of the tensor's bits.
//! Two runs that should agree — the same model on two operating systems, two SIMD tiers, two
//! thread counts — are then diffed, and the first differing line names the op.
//!
//! Line grammar, one tensor per line:
//!
//! ```text
//! b<backend> g<graph> op<op> <R|W> t<tensor> n=<elements> <fnv1a-64 hex> <op label>
//! ```
//!
//! `backend` numbers the process's tapped CPU backends in the order they first record (a
//! session, its draft model, a reference oracle each get their own), `graph` counts that
//! backend's tapped `execute` calls from 0, `op` is the op's index in the graph, `R`/`W` says
//! whether the tensor was read or written, and the label is the profiler's
//! ([`infr_core::prof::op_label`]), so the lines line up with an `INFR_PROF_OPS` table. Reads skip
//! weights: they come from the model file, identical on both sides of any comparison this is
//! for, and hashing every weight per op would dominate the run. An in-place tensor (a KV cache, a
//! `+=` destination, recurrent state) appears as a read AND a write, so an op that is itself the
//! SOURCE of a difference shows as the first `W` line that differs after `R` lines that all
//! agree; an op that is only downstream of one shows a differing `R` line first.
//!
//! The hash covers the bytes the interpreter holds: the f32 working store for graph-internal
//! tensors, and the bound buffer (at its own dtype) for tensors written in place there.
//!
//! **Several writers.** One `execute`'s lines are collected in memory and appended with a
//! single write while a process-wide lock is held, so executes running concurrently in one
//! process (several backends, or one backend from several threads) never splice into each
//! other's lines, and the `b` field tells their lines apart. Separate PROCESSES must not share
//! one file: nothing orders their appends.

use infr_core::error::{Error, Result};
use infr_core::graph::{Graph, Op, TensorKind};
use infr_core::tensor::TensorId;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Serializes the appends of every tap in the process.
static WRITE_LOCK: Mutex<()> = Mutex::new(());

/// The next backend ordinal to hand out.
static NEXT_BACKEND: AtomicU64 = AtomicU64::new(0);

/// A backend's tap identity: its ordinal (taken on first use) and its graph counter.
#[derive(Default)]
pub(crate) struct TapCounter {
    backend: std::sync::OnceLock<u64>,
    graphs: AtomicU64,
}

impl TapCounter {
    /// `(backend ordinal, next graph number)` for one `execute`.
    pub(crate) fn next(&self) -> (u64, u64) {
        let b = *self
            .backend
            .get_or_init(|| NEXT_BACKEND.fetch_add(1, Ordering::Relaxed));
        (b, self.graphs.fetch_add(1, Ordering::Relaxed))
    }
}

/// One `execute`'s tap. Built only when the knob is set, so an untapped run never formats a
/// label.
pub(crate) struct OpHashTap {
    path: PathBuf,
    lines: String,
    backend: u64,
    graph: u64,
}

/// `(elements, FNV-1a of the bits)` of an f32 slice — the working-store half of a tap hash.
pub(crate) fn hash_f32(v: &[f32]) -> (usize, u64) {
    (
        v.len(),
        infr_core::kernel_cache::fnv1a(bytemuck::cast_slice(v)),
    )
}

impl OpHashTap {
    /// A tap for one `execute`, numbered from `counter`, appending to `path` when it finishes.
    pub(crate) fn new(path: &Path, counter: &TapCounter) -> Self {
        let (backend, graph) = counter.next();
        Self {
            path: path.to_path_buf(),
            lines: String::new(),
            backend,
            graph,
        }
    }

    /// Record the tensors `op` (the graph's op number `op_idx`) reads (`write == false`) or
    /// writes (`write == true`), each hashed by `hash` as `(elements, fnv1a)`. Reads skip weights
    /// (see the module doc).
    pub(crate) fn record(
        &mut self,
        g: &Graph,
        op_idx: usize,
        op: &Op,
        write: bool,
        hash: impl Fn(TensorId) -> (usize, u64),
    ) {
        let (reads, writes) = op.io();
        let ids = if write { writes } else { reads };
        let label = infr_core::prof::op_label(op, g);
        let tag = if write { 'W' } else { 'R' };
        for id in ids {
            if !write && matches!(g.tensors[id.0 as usize].kind, TensorKind::Weight) {
                continue;
            }
            let (n, h) = hash(id);
            self.lines.push_str(&format!(
                "b{} g{} op{op_idx} {tag} t{} n={n} {h:016x} {label}\n",
                self.backend, self.graph, id.0
            ));
        }
    }

    /// Append this `execute`'s lines to the file in one write, under the process-wide lock.
    pub(crate) fn finish(self) -> Result<()> {
        let err = |what: &str, e: std::io::Error| {
            Error::Backend(format!(
                "cpu backend: INFR_CPU_OP_HASH: {what} {}: {e}",
                self.path.display()
            ))
        };
        let _guard = WRITE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| err("cannot open", e))?;
        f.write_all(self.lines.as_bytes())
            .map_err(|e| err("cannot write", e))
    }
}
