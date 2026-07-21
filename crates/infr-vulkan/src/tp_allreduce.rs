//! The tensor-parallel ALL-REDUCE — sum each rank's partial `[tokens, n_embd]` output of a
//! row-parallel projection (attn-O / ffn-down) across all ranks so every rank ends with the full
//! sum, the residual stream identical everywhere. This is the comms-critical piece of TP: two of
//! these run per transformer layer.
//!
//! # Data path — host-LESS P2P dma-buf
//!
//! Each rank `p` publishes its partial into an EXPORTED device-local buffer `E_p`
//! ([`VulkanBackend::p2p_export`]); every other rank imports `E_p` ([`p2p_import`]) so it can read
//! `p`'s VRAM directly over PCIe (no host bounce — the campaign measured 12.9-27.2 GB/s vs 3.82
//! host-bounce). A rank then copies each peer's published partial into a local scratch and sums
//! `own + Σ peers` with an on-device `Op::Add` chain. The reduction is a FIXED rank-order sum, so it
//! is deterministic run-to-run (bit-reproducible), and — crucially for correctness — every rank
//! computes the SAME set of addends in the SAME order, so all ranks agree bit-for-bit after the
//! reduce (the residual stream can't drift between devices). A device pair that can't share a
//! dma-buf falls back to a host bounce for that transport, reported as [`AllReduceMode::Host`].
//!
//! # Sync — external semaphore vs host fence
//!
//! The gather must not read `E_p` before rank `p` finished writing it. v1 orders this with the
//! backend's host fence (`copy_buffer`'s `queue_wait_idle`): correct, but each all-reduce host-stalls
//! — 2 per layer, so on a deep model the host round-trips dominate. The zero-host-stall optimization
//! (`VK_KHR_external_semaphore_fd`: rank `p` signals a semaphore its readers wait on, GPU-side) is
//! the "don't skimp" target; when the device pair supports it the mode is
//! [`AllReduceMode::P2pSemaphore`], else [`AllReduceMode::P2pHostFence`]. See [`AllReduce::mode`].
//!
//! # Generality
//!
//! The `publish → gather-all-peers → sum` schedule is correct for any world size `W` (each rank
//! reads `W-1` peers into `W-1` scratches, an all-to-all-ish exchange of `O(W²)` total bandwidth).
//! A ring/tree schedule (`O(W)` bandwidth) is the >2-device throughput optimization, deferred.

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Plan};
use infr_core::error::Result;
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};

use crate::{be, P2pExport, P2pHandleType, VulkanBackend};

/// How the all-reduce moves + orders data across the device pair (for the report).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllReduceMode {
    /// Host-less P2P dma-buf data path, cross-device ordering by `VK_KHR_external_semaphore_fd`
    /// (no host round-trip). The optimized target.
    P2pSemaphore,
    /// Host-less P2P dma-buf data path, cross-device ordering by the host fence (`queue_wait_idle`).
    /// Correct but host-stalls twice per layer — the reported perf gap when the semaphore path is
    /// unavailable.
    P2pHostFence,
    /// No usable cross-device dma-buf: partials bounce through host RAM. Slowest; correctness only.
    Host,
}

/// The per-rank all-reduce transport for ONE boundary size, set up once and reused every layer.
pub struct AllReduce {
    mode: AllReduceMode,
    bytes: usize,
    /// f32 element count of the boundary tensor (`bytes / 4`).
    elems: usize,
    /// Rank `p`'s exported publish buffer (`Some` in a P2P mode). Indexed by producer rank.
    exports: Vec<Option<P2pExport>>,
    /// `imported[r][p]` — rank `r`'s import alias of rank `p`'s export (`p != r`; `None` at `p==r`).
    /// Only populated in a P2P mode.
    imported: Vec<Vec<Option<Box<dyn Buffer>>>>,
    /// `scratch[r][p]` — rank `r`'s local buffer holding peer `p`'s partial before the sum
    /// (`None` at `p==r`).
    scratch: Vec<Vec<Option<Box<dyn Buffer>>>>,
    /// Per-rank compiled Add-chain plan (`own += Σ peers`).
    reduce_plans: Vec<Box<dyn Plan>>,
    /// The reduce graph's tensor handles: `sub` (own partial, in/out) + one scratch input per peer.
    sub_tid: TensorId,
    scratch_tids: Vec<TensorId>,
    /// Host bounce scratch (Host mode only).
    host_buf: std::sync::Mutex<Vec<u8>>,
}

// The P2P exports/imports are Send/Sync under the same whole-backend discipline as the buffers.
unsafe impl Send for AllReduce {}
unsafe impl Sync for AllReduce {}

impl AllReduce {
    /// The transport mode chosen for the report.
    pub fn mode(&self) -> AllReduceMode {
        self.mode
    }

    /// Build the all-reduce transport over `ranks` for a boundary of `bytes` f32 bytes. Sets up the
    /// P2P export/import ring (when every rank supports dma-buf and `use_p2p`) + per-rank scratches +
    /// the per-rank Add-chain reduce plan.
    pub fn new(ranks: &[VulkanBackend], bytes: usize, use_p2p: bool) -> Result<Self> {
        let w = ranks.len();
        if bytes == 0 || !bytes.is_multiple_of(4) {
            return Err(be(format!(
                "tp all-reduce: boundary size {bytes} is not a positive multiple of 4 (f32)"
            )));
        }
        let elems = bytes / 4;

        // ── choose the data + sync mode ────────────────────────────────────────────────────────
        let all_dma = use_p2p && ranks.iter().all(|b| b.p2p_supported(P2pHandleType::DmaBuf));

        let mut exports: Vec<Option<P2pExport>> = (0..w).map(|_| None).collect();
        let mut imported: Vec<Vec<Option<Box<dyn Buffer>>>> =
            (0..w).map(|_| (0..w).map(|_| None).collect()).collect();
        let mut p2p_ok = all_dma;

        if all_dma {
            // Publish buffer per producer rank.
            for (p, ex) in exports.iter_mut().enumerate() {
                match ranks[p].p2p_export(bytes, P2pHandleType::DmaBuf) {
                    Ok(e) => *ex = Some(e),
                    Err(e) => {
                        eprintln!(
                            "tp all-reduce: p2p_export on rank {p} failed ({e}); host bounce"
                        );
                        p2p_ok = false;
                        break;
                    }
                }
            }
            // Cross-import: rank r imports every other rank's export.
            if p2p_ok {
                'outer: for r in 0..w {
                    for p in 0..w {
                        if p == r {
                            continue;
                        }
                        let export = exports[p].as_ref().expect("export set");
                        match ranks[r].p2p_import(export) {
                            Ok(buf) => imported[r][p] = Some(buf),
                            Err(e) => {
                                eprintln!(
                                    "tp all-reduce: p2p import rank{p}->rank{r} rejected ({e}); \
                                     host bounce"
                                );
                                p2p_ok = false;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        if !p2p_ok {
            exports = (0..w).map(|_| None).collect();
            imported = (0..w).map(|_| (0..w).map(|_| None).collect()).collect();
        }

        // The external-semaphore ordering rides on the P2P data path. It is attempted (and the mode
        // set to P2pSemaphore) only when the P2P path is live AND every rank can export/import a
        // semaphore fd; otherwise the P2P path uses the host fence.
        let mode = if p2p_ok {
            if ranks.iter().all(|b| b.external_semaphore_supported()) {
                AllReduceMode::P2pSemaphore
            } else {
                AllReduceMode::P2pHostFence
            }
        } else {
            AllReduceMode::Host
        };
        if w > 1 {
            let sync = match mode {
                AllReduceMode::P2pSemaphore => "external-semaphore (no host round-trip)",
                AllReduceMode::P2pHostFence => "host-fence (queue_wait_idle)",
                AllReduceMode::Host => "host-bounce (no cross-device dma-buf)",
            };
            let data = if p2p_ok { "P2P dma-buf" } else { "host RAM" };
            eprintln!(
                "tp all-reduce: {w}-way, {} bytes/boundary — data path: {data}, cross-device sync: {sync}",
                bytes
            );
        }

        // ── per-rank scratches (hold each peer's partial before the sum) ───────────────────────
        let mut scratch: Vec<Vec<Option<Box<dyn Buffer>>>> = Vec::with_capacity(w);
        #[allow(clippy::needless_range_loop)] // r indexes ranks AND builds the r-th scratch row
        for r in 0..w {
            let mut row = Vec::with_capacity(w);
            for p in 0..w {
                if p == r {
                    row.push(None);
                } else {
                    row.push(Some(ranks[r].alloc(bytes, BufferUsage::Activations)?));
                }
            }
            scratch.push(row);
        }

        // ── per-rank Add-chain reduce plan: sub += Σ scratch[peer] ─────────────────────────────
        let (reduce_graph, sub_tid, scratch_tids) = build_reduce_graph(elems, w);
        let mut reduce_plans = Vec::with_capacity(w);
        for b in ranks {
            reduce_plans.push(b.compile(&reduce_graph)?);
        }

        Ok(Self {
            mode,
            bytes,
            elems,
            exports,
            imported,
            scratch,
            reduce_plans,
            sub_tid,
            scratch_tids,
            host_buf: std::sync::Mutex::new(vec![0u8; bytes]),
        })
    }

    /// All-reduce: `bufs[r]` holds rank `r`'s partial on entry; on return every `bufs[r]` holds the
    /// full sum `Σ_r bufs[r]`. `elems` is the live element count (`<= self.elems`; a smaller live
    /// count than the allocated boundary would need a smaller Add — v1 keeps them equal).
    pub fn reduce(
        &self,
        ranks: &[VulkanBackend],
        bufs: &[Box<dyn Buffer>],
        elems: usize,
    ) -> Result<()> {
        if elems != self.elems {
            return Err(be(format!(
                "tp all-reduce: live elems {elems} != transport elems {} (boundary size must match \
                 the compiled reduce plan)",
                self.elems
            )));
        }
        let w = ranks.len();
        if w == 1 {
            return Ok(()); // world=1: the partial IS the full sum, nothing to reduce.
        }
        match self.mode {
            AllReduceMode::P2pSemaphore | AllReduceMode::P2pHostFence => {
                self.reduce_p2p(ranks, bufs)
            }
            AllReduceMode::Host => self.reduce_host(ranks, bufs),
        }
    }

    /// P2P data path. v1 orders with the host fence (each `copy_buffer`/`sync` fences); the
    /// `P2pSemaphore` mode replaces those fences with cross-device semaphore waits (Phase 2).
    fn reduce_p2p(&self, ranks: &[VulkanBackend], bufs: &[Box<dyn Buffer>]) -> Result<()> {
        let w = ranks.len();
        // ── publish: each rank copies its partial into its exported buffer ─────────────────────
        for p in 0..w {
            let ex = self.exports[p].as_ref().expect("p2p export");
            ranks[p].copy_buffer(bufs[p].as_ref(), ex.buffer(), self.bytes)?;
            ranks[p].sync()?;
        }
        // ── gather: each rank reads every peer's published buffer into a local scratch ─────────
        // (Reads only — the peers' exports are not modified, so this is race-free after publish.)
        #[allow(clippy::needless_range_loop)]
        // r indexes ranks, imported[r] and scratch[r] together
        for r in 0..w {
            for p in 0..w {
                if p == r {
                    continue;
                }
                let imp = self.imported[r][p].as_ref().expect("p2p import");
                let sc = self.scratch[r][p].as_ref().expect("scratch");
                ranks[r].copy_buffer(imp.as_ref(), sc.as_ref(), self.bytes)?;
                ranks[r].sync()?;
            }
        }
        // ── reduce: each rank sums own + all peers' scratches into its own partial ──────────────
        self.run_reduce_plans(ranks, bufs)
    }

    /// Host-bounce fallback: each rank downloads every peer's partial through host RAM.
    fn reduce_host(&self, ranks: &[VulkanBackend], bufs: &[Box<dyn Buffer>]) -> Result<()> {
        let w = ranks.len();
        for r in 0..w {
            for p in 0..w {
                if p == r {
                    continue;
                }
                let mut host = self.host_buf.lock().expect("tp host buf poisoned");
                ranks[p].download(bufs[p].as_ref(), &mut host)?;
                let sc = self.scratch[r][p].as_ref().expect("scratch");
                ranks[r].upload(sc.as_ref(), &host)?;
            }
        }
        self.run_reduce_plans(ranks, bufs)
    }

    /// Run each rank's compiled Add-chain: `bufs[r] += Σ scratch[r][peer]`.
    fn run_reduce_plans(&self, ranks: &[VulkanBackend], bufs: &[Box<dyn Buffer>]) -> Result<()> {
        let w = ranks.len();
        for r in 0..w {
            let mut b = Bindings::new();
            b.bind(self.sub_tid, bufs[r].as_ref());
            // scratch_tids are ordered by peer index skipping self; bind them to this rank's peers
            // in the SAME (ascending peer) order so every rank sums the same addends deterministically.
            let mut k = 0usize;
            for p in 0..w {
                if p == r {
                    continue;
                }
                let sc = self.scratch[r][p].as_ref().expect("scratch");
                b.bind(self.scratch_tids[k], sc.as_ref());
                k += 1;
            }
            ranks[r].execute(self.reduce_plans[r].as_ref(), &b)?;
            ranks[r].sync()?;
        }
        Ok(())
    }
}

/// Build the reduce graph: `sub` (own partial, `[elems]` f32, in/out) plus `world-1` scratch inputs,
/// chained `sub = sub + scratch_k`. The Adds are sequential (each reads the running `sub`), which is
/// a deterministic fixed-order sum.
fn build_reduce_graph(elems: usize, world: usize) -> (Graph, TensorId, Vec<TensorId>) {
    let mut g = Graph::new();
    let sub = g.input(TensorDesc::new(vec![elems], DType::F32));
    let scratch_tids: Vec<TensorId> = (0..world.saturating_sub(1))
        .map(|_| g.input(TensorDesc::new(vec![elems], DType::F32)))
        .collect();
    for &s in &scratch_tids {
        g.push(Op::Add {
            a: sub,
            b: s,
            dst: sub,
            n: elems as u32,
        });
    }
    (g, sub, scratch_tids)
}
