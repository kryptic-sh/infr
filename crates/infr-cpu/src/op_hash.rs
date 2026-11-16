//! The per-op hash tap (`debug.cpu_op_hash`, `INFR_CPU_OP_HASH=<path>`): a bisecting aid that
//! appends, for every op the interpreter runs, one line per tensor the op READS (before it runs)
//! and one per tensor it WRITES (after it ran), each with an FNV-1a hash of the tensor's bits.
//! Two runs that should agree — the same model on two operating systems, two SIMD tiers, two
//! thread counts — are then diffed, and the first differing line names the op.
//!
//! Line grammar, one tensor per line:
//!
//! ```text
//! g<graph> op<op> <R|W> t<tensor> n=<elements> <fnv1a-64 hex> <op label>
//! ```
//!
//! `graph` counts the backend's tapped `execute` calls from 0, `op` is the op's index in the
//! graph, `R`/`W` says whether the tensor was read or written, and the label is the profiler's
//! ([`infr_core::prof::op_label`]), so the lines line up with an `INFR_PROF_OPS` table. Reads skip
//! weights: they come from the model file, identical on both sides of any comparison this is
//! for, and hashing every weight per op would dominate the run. An in-place tensor (a KV cache, a
//! `+=` destination, recurrent state) appears as a read AND a write, so an op that is itself the
//! SOURCE of a difference shows as the first `W` line that differs after `R` lines that all
//! agree; an op that is only downstream of one shows a differing `R` line first.
//!
//! The hash covers the bytes the interpreter holds: the f32 working store for graph-internal
//! tensors, and the bound buffer (at its own dtype) for tensors written in place there.

use infr_core::error::{Error, Result};
use infr_core::graph::{Graph, Op, TensorKind};
use infr_core::tensor::TensorId;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// One `execute`'s open tap. Built only when the knob is set, so an untapped run never opens a
/// file or formats a label.
pub(crate) struct OpHashTap {
    out: BufWriter<File>,
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
    /// Open `path` for appending, as graph number `graph`.
    pub(crate) fn open(path: &Path, graph: u64) -> Result<Self> {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| {
                Error::Backend(format!(
                    "cpu backend: INFR_CPU_OP_HASH: cannot open {}: {e}",
                    path.display()
                ))
            })?;
        Ok(Self {
            out: BufWriter::new(f),
            graph,
        })
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
    ) -> Result<()> {
        let (reads, writes) = op.io();
        let ids = if write { writes } else { reads };
        let label = infr_core::prof::op_label(op, g);
        let tag = if write { 'W' } else { 'R' };
        for id in ids {
            if !write && matches!(g.tensors[id.0 as usize].kind, TensorKind::Weight) {
                continue;
            }
            let (n, h) = hash(id);
            writeln!(
                self.out,
                "g{} op{op_idx} {tag} t{} n={n} {h:016x} {label}",
                self.graph, id.0
            )
            .map_err(|e| Error::Backend(format!("cpu backend: INFR_CPU_OP_HASH: write: {e}")))?;
        }
        Ok(())
    }

    /// Flush the lines of this `execute`.
    pub(crate) fn finish(mut self) -> Result<()> {
        self.out
            .flush()
            .map_err(|e| Error::Backend(format!("cpu backend: INFR_CPU_OP_HASH: flush: {e}")))
    }
}
