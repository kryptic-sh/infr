//! CPU model runner — builds and drives the agnostic decode [`infr_core::graph::Graph`] through [`CpuBackend`].
//! The backend itself lives in `infr-cpu`; this module is the model-specific "glue" that
//! assembles the layer graph, uploads weights, and steps the KV cache.
//!
//! Split into submodules (pure move, zero behavior change): `weights` holds the per-layer
//! weight-handle structs and the persistent seam session state; `sc` holds the DiffusionGemma
//! self-conditioning pieces; `runner` holds the giant backend-generic `generate_dense_backend`
//! and its `DecodeHandles`. This file keeps the thin per-backend entry wrappers, the `verify_*`
//! family, and the small shared helpers every submodule reaches into.
#![allow(clippy::too_many_arguments)]

use crate::{dequant_block, Config, EngineConfig, GenStats, PerLayerEmbd};
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Buffer, BufferUsage, KvAllocCost};
use infr_core::tensor::DType;
use infr_core::WeightSource;
use infr_cpu::CpuBackend;
use infr_gguf::{Gguf, TensorBytes};

mod dsv4_emit;
mod dsv4_plan;
pub mod model;
mod routing;
mod runner;
mod sc;
mod weights;
mod wshape;

pub use routing::Routing;
pub(crate) use runner::generate_dense_backend;
pub(crate) use sc::DenoiseReq;
pub use sc::{DenoiseOutcome, EbReduced};
pub(crate) use weights::SeamKv;

/// A LAZILY-dequantized host f32 token-embedding table, threaded through the seam runners in place
/// of a `&[f32]`.
///
/// `token_embd.weight` blown up to f32 is enormous — Qwen3-14B's 151936×5120 Q4_K table becomes
/// 3.1 GiB of host RAM and costs ~4s of dequant, which used to be paid EAGERLY by every
/// `SeamModel::load`, i.e. by every model load on every backend. But the Vulkan and Metal dense
/// paths upload `token_embd.weight` to the device in its NATIVE dtype and gather embeddings ON
/// DEVICE (`Op::EmbedGather` / the tied-lm_head `Op::Linear`), so they never look at the host
/// table. Only the host-gather consumers touch it: the CPU runner's embed, the DiffusionGemma SC
/// soft-embed, and the MTP heads.
///
/// Passing this handle instead of a materialized slice keeps the dequant OFF the GPU load path
/// while leaving every host consumer byte-for-byte identical — they call [`get`](Self::get), which
/// dequantizes once into the owning [`model::SeamModel`]'s cache and returns the cached table on
/// every later call.
#[derive(Clone, Copy)]
pub(crate) struct TokenEmbd<'a> {
    cell: &'a std::sync::OnceLock<Vec<f32>>,
    gguf: &'a Gguf,
}

impl<'a> TokenEmbd<'a> {
    pub(crate) fn new(cell: &'a std::sync::OnceLock<Vec<f32>>, gguf: &'a Gguf) -> Self {
        Self { cell, gguf }
    }

    /// The dequantized `[vocab, n_embd]` row-major table — dequantized on first call, cached after.
    /// `Config::from_gguf` already validated the tensor exists at load, but a truncated/corrupt GGUF
    /// can still fail the dequant here (the first host gather, lazily). FALLIBLE so that a real
    /// non-programmer input surfaces a clear error at the call site instead of aborting the process.
    pub(crate) fn get(&self) -> AResult<&'a [f32]> {
        if let Some(v) = self.cell.get() {
            return Ok(v);
        }
        let (v, _) =
            crate::quant::load_tensor_dequant(self.gguf, "token_embd.weight").map_err(|e| {
                anyhow!("token_embd.weight: dequant failed (corrupt/truncated GGUF?): {e}")
            })?;
        // Race-safe cache fill: another thread may have won the init; `set` drops ours if so, then
        // `get` returns whichever value stuck (get_or_init semantics without a fallible init).
        let _ = self.cell.set(v);
        Ok(self
            .cell
            .get()
            .expect("cell was just set or already initialized"))
    }
}

/// The CPU seam weight binder, hoisted so the CPU runner and every CPU `verify_*` entry share one
/// copy: map an mmap slice zero-copy; alloc+upload owned bytes (the combined gate+up concat — never
/// produced for CPU since `combined_gu` is false there, but stays correct if it ever is).
/// Build the CPU backend's host weight cache, or `None` to keep the zero-copy mmap path.
///
/// # When this turns itself on
/// **Only when the weights do not fit.** The CPU backend has no VRAM ladder to tell it that, the
/// way the Vulkan paths do (see [`vulkan_host_tier`], whose callers are already past that
/// decision), so it asks directly: do the pageable weights fit the host memory this machine can
/// actually spare? If they do, the mmap path is right and stays — it is zero-copy, the page cache
/// holds the model, and an arena would only add copies. If they do not, the page cache is about to
/// thrash on the cyclic sweep a forward pass performs (`docs/perf/results.md` measured the
/// collapse: decode 23-33x slower once the weights stop fitting), and the arena's own policy is
/// what fixes it.
///
/// An explicit `paging.dram` always wins in BOTH directions — it forces the arena on a model that
/// would have fit, and pins the size on one that would not. That is deliberate and must stay: a
/// machine with enough RAM to hold the model is exactly the machine you want to exercise the
/// streaming path on, so `INFR_DRAM_CACHE=<size>` is how a CPU run is put on it regardless of what
/// the fit test would have decided.
///
/// Returns `None` — not an error — when nothing seats, so every degraded case falls back to
/// today's behaviour rather than failing the load.
///
/// # What the automatic budget leaves
/// The arena is taken from what is available only after [`cpu_committed_bytes`] — the run's KV
/// cache, the weights that stay mapped, its activations — and after the headroom
/// [`infr_core::hostmem::auto_arena_bytes`] leaves to the rest of the machine, so the budget is
/// what the process adds on top of everything else it holds.
fn cpu_paged_store(
    ec: &EngineConfig,
    g: &Gguf,
    cfg: &Config,
) -> AResult<Option<std::sync::Arc<infr_cpu::paged::PagedWeights>>> {
    // `0` is the explicit OFF switch, not "unset" — so it must reach `Requested` unfiltered.
    let explicit = infr_core::hostmem::Requested::from_config(
        ec.paging.dram.map(|s| s.resolve(0)),
        ec.paging.dram_bypass,
    );
    let weights = cpu_pageable(g, cfg);
    // Only the weights this backend would actually page count toward "does it fit": the sub-floor
    // tensors stay mapped either way, so counting them could page a model that fits.
    let pageable: u64 = weights
        .iter()
        .filter(|w| w.nbytes >= infr_cpu::paged::MIN_PAGED_BYTES)
        .map(|w| w.nbytes as u64)
        .sum();
    let available = infr_core::hostmem::available_bytes();
    let committed = cpu_committed_bytes(ec, g, cfg);
    let budget = match infr_core::hostmem::cpu_arena_plan(explicit, available, committed, pageable)
    {
        // Only the GPU tiers can want a reader with no cache under them (their arena IS the cache
        // on unified memory). For the CPU backend this arena is the only tier there is, so a
        // read-through with nothing behind it would be a pure regression on the mapping.
        infr_core::hostmem::ArenaPlan::StreamOnly => {
            debug_assert!(false, "cpu_arena_plan never answers StreamOnly");
            return Ok(None);
        }
        infr_core::hostmem::ArenaPlan::Take(n) => {
            // An automatic arena too small for the weights may serve a layer-major prefill, which
            // holds its carried rows beside it: that comes out of the arena, and only then. An
            // explicit budget is the user's, as before.
            let n = if explicit == infr_core::hostmem::Requested::Auto && n < pageable {
                n.saturating_sub(layer_major_carry_bytes(ec, cfg))
            } else {
                n
            };
            if explicit == infr_core::hostmem::Requested::Auto {
                tracing::info!(
                    "host paging: {:.2} GB of weights exceed the {:.2} GB of host memory \
                     available, so they stream from disk through a {:.2} GB arena instead of the \
                     OS page cache, leaving half of what is available beside the run's other \
                     {:.2} GB (set INFR_DRAM_CACHE to override)",
                    pageable as f64 / 1e9,
                    available.unwrap_or(0) as f64 / 1e9,
                    n as f64 / 1e9,
                    committed as f64 / 1e9,
                );
            }
            n as usize
        }
        infr_core::hostmem::ArenaPlan::Skip(why) => {
            use infr_core::hostmem::Skip;
            // `Fits` and `NoProbe` are the ordinary paths — the weights are mapped, exactly as
            // before this tier existed — so neither says anything. `TooLittle` is the one case
            // where the run is about to be slow for a reason the user can act on.
            if why == Skip::TooLittle {
                tracing::warn!(
                    "host paging: {:.2} GB of weights do not fit the {:.2} GB of host memory \
                     available, but too little is free to seat a useful arena — falling back to \
                     the OS page cache, which thrashes on a forward pass's cyclic sweep. Free \
                     memory, or set INFR_DRAM_CACHE explicitly",
                    pageable as f64 / 1e9,
                    available.unwrap_or(0) as f64 / 1e9,
                );
            }
            return Ok(None);
        }
    };
    let plans = infr_cpu::paged::plan_pools(budget, &weights);
    if plans.is_empty() {
        tracing::warn!(
            "host paging: a {:.2} GB budget seats no weight class of this model — keeping the \
             mmap path (raise INFR_DRAM_CACHE past one tensor's size to page)",
            budget as f64 / 1e9,
        );
        return Ok(None);
    }
    let io = std::sync::Arc::new(
        infr_core::blockio::FileBlockIo::from_handles(&g.shard_handles())
            .map_err(|e| anyhow!("{e}"))?,
    );
    let trace = infr_core::pager::trace::TraceFile::for_config(ec.paging.trace.as_deref())
        .map_err(|e| anyhow!("{e}"))?;
    let store = infr_cpu::paged::PagedWeights::new(&plans, io, trace.as_ref())
        .map_err(|e| anyhow!("{e}"))?;
    let experts: usize = plans
        .iter()
        .filter(|p| p.experts)
        .map(|p| p.n_slots * p.slot_bytes)
        .sum();
    tracing::info!(
        "host paging: {} weight class(es), {:.2} GB arena of a {:.2} GB budget ({:.2} GB of it \
         single experts)",
        plans.len(),
        store.arena_bytes() as f64 / 1e9,
        budget as f64 / 1e9,
        experts as f64 / 1e9,
    );
    Ok(Some(std::sync::Arc::new(store)))
}

/// Every weight of `g` as the CPU host tier sees it: an expert bank (see [`is_expert_bank`]) of a
/// MoE model is read `n_used` experts at a time, everything else whole. A bank whose name carries
/// no layer index is counted as one more bank of a shared layer, which can only raise its class's
/// floor.
fn cpu_pageable(g: &Gguf, cfg: &Config) -> Vec<infr_cpu::paged::Pageable> {
    use infr_cpu::paged::{Pageable, Reads};
    g.tensors()
        .iter()
        .map(|t| Pageable {
            nbytes: t.nbytes,
            reads: match &cfg.moe {
                Some(m) if is_expert_bank(&t.name) => Reads::Experts {
                    n_expert: m.n_expert,
                    n_used: m.n_used,
                    layer: exps_layer(&t.name).unwrap_or(usize::MAX),
                },
                _ => Reads::Whole,
            },
        })
        .collect()
}

/// The layer index of a `blk.{l}.…_exps…` tensor name.
fn exps_layer(name: &str) -> Option<usize> {
    if !name.contains("_exps") {
        return None;
    }
    name.strip_prefix("blk.")
        .and_then(|r| r.split('.').next())
        .and_then(|l| l.parse::<usize>().ok())
}

/// Host memory a CPU run holds besides its weight arena, which the automatic arena budget must
/// leave room for: the KV cache at the context the run will use, the weights too small to page
/// (mapped, and in the working set once a pass reads them), [`CPU_ACTIVATION_RESERVE`], and the
/// Q4_K repack cache where the run can fill it ([`cpu_repack_reserve`]).
///
/// The context is `device.ctx` when set, else the trained context capped at
/// [`CPU_BUDGET_CTX`] — a CPU generation sizes its cache to its own prompt and output, which is
/// unknown when the arena is sized once for every generation the model will serve. The cache is
/// priced at f16, the CPU default, without the sliding-window ring, which only shrinks it.
fn cpu_committed_bytes(ec: &EngineConfig, g: &Gguf, cfg: &Config) -> u64 {
    let ctx = crate::chat::cfg_ctx(ec, cfg.n_ctx_train)
        .unwrap_or_else(|| cfg.n_ctx_train.min(CPU_BUDGET_CTX));
    // No backend exists when the arena is budgeted. The CPU backend rounds its cache neither way,
    // which `cpu_kv_alloc_is_the_estimate_without_the_snapshot` checks on `CpuBackend` itself.
    let kv = kv_bytes_estimate(cfg, ctx, false, 1, KvAlign::NONE, false);
    let small: u64 = g
        .tensors()
        .iter()
        .filter(|t| t.nbytes < infr_cpu::paged::MIN_PAGED_BYTES)
        .map(|t| t.nbytes as u64)
        .sum();
    kv + small
        + CPU_ACTIVATION_RESERVE
        + cpu_repack_reserve(ec, g.tensors(), infr_cpu::q4k_expert_repack_available())
}

/// What a layer-major prefill over the CPU's host arena holds between layers at most: one
/// segment of carried rows (`layer_major_segment_rows`) at the run's context, on a family that
/// takes the order there ([`host_layer_major`]); 0 for every other family.
fn layer_major_carry_bytes(ec: &EngineConfig, cfg: &Config) -> u64 {
    if !host_layer_major(cfg, usize::MAX, false) {
        return 0;
    }
    let ctx = crate::chat::cfg_ctx(ec, cfg.n_ctx_train)
        .unwrap_or_else(|| cfg.n_ctx_train.min(CPU_BUDGET_CTX));
    let rows = ctx.min(layer_major_segment_rows(ec, cfg)) as u64;
    rows * layer_major_row_bytes(cfg)
}

/// The CPU repack cache's budget (`kernels.cpu.repack_mb`) when this run can fill it, else 0.
///
/// The fused Q4_K expert path keeps a pack of every routed expert it multiplies, up to that budget
/// (`CpuBackend::q4k_pack_for`), on CPUs where `has_path` ([`infr_cpu::q4k_expert_repack_available`])
/// and for models with a fused Q4_K gate_up bank (qwen35moe, gemma4's MoE). Those packs are
/// anonymous memory beside the arena, so they are priced at the whole budget: the cache keeps
/// what it built until the budget is spent.
fn cpu_repack_reserve(
    ec: &EngineConfig,
    tensors: &[infr_core::loader::TensorInfo],
    has_path: bool,
) -> u64 {
    let fused_q4k = tensors
        .iter()
        .any(|t| t.name.ends_with("ffn_gate_up_exps.weight") && t.dtype == DType::Q4K);
    if has_path && fused_q4k {
        (ec.kernels.cpu.repack_mb as u64) << 20
    } else {
        0
    }
}

/// The context a CPU run's KV cache is priced at when `device.ctx` does not name one — the same
/// default the Metal chat opens its sessions at.
const CPU_BUDGET_CTX: usize = 8192;

/// Host memory set aside for what a CPU forward allocates per pass — activations, the MoE pair
/// buffers of a prefill chunk, the interpreter's recycled scratch — which nothing prices per model.
const CPU_ACTIVATION_RESERVE: u64 = 1 << 30;

/// Build the host DRAM tier under a set of Vulkan arena pools — one
/// [`infr_core::hostpager::HostPager`] per pool, since a pool is already exactly a block-size class,
/// which is the uniform-slot shape the host tier needs.
///
/// `classes` is `(slot stride, block count)` per pool, in pool order; the result matches that
/// order, `None` where the budget seated nothing. `what` names the caller in the log line — the
/// dense streaming pools and the MoE expert pools both come through here, and a run reporting one
/// arena should say which. A budget too small to seat a pool leaves that pool on the mmap path
/// rather than failing the load.
///
/// # Where the budget comes from
/// **Both call sites are already past the point where the model did not fit VRAM** — dense
/// streaming and the paged MoE cache are each only reached when residency was rejected — so
/// reaching here IS the signal that this run has to stream. An explicit `paging.dram` still wins;
/// otherwise the budget is sized from what the host can actually spare
/// ([`infr_core::hostmem`]), because a tier that is off by default helps nobody on the one run
/// that needs it, and the alternative is a user guessing a number that is worth 1.6x
/// (`docs/perf/results.md`).
///
/// # Forcing this path on hardware that would not take it
/// Auto-sizing never REPLACES the two knobs, because a machine big enough to hold the model
/// resident is exactly the machine you want to test streaming on. `INFR_CACHE` (`paging.cache`)
/// caps the VRAM paging budget, which is what forces residency to be rejected and this function to
/// be reached at all; `INFR_DRAM_CACHE` (`paging.dram`) then pins this arena's size instead of
/// letting it be measured. Both together are how every figure in `docs/perf/results.md` was taken
/// on a card that would otherwise have kept the model resident — e.g. `--cache 2g --dram 3g`.
///
/// # Unified memory has only ONE host tier, and it is the arena above this one
/// `unified` is `DeviceCaps::unified_memory` — an iGPU or APU whose "VRAM" IS system RAM. There the
/// arena above already sits in host memory, so a host tier beneath it would hold extra blocks in
/// the same RAM that the GPU CANNOT read directly: every hit would still be copied through the
/// staging ring, while the identical bytes spent on the arena above are read in place. It is not
/// merely double-counted, it is strictly worse than making that arena bigger. So auto-sizing
/// declines on unified memory and says why; an explicit `paging.dram` is still honoured, because a
/// user asking for it by name may be working around something this does not model.
fn vulkan_host_tier(
    ec: &EngineConfig,
    g: &Gguf,
    what: &str,
    classes: &[(usize, usize)],
    unified: bool,
) -> AResult<Vec<Option<std::sync::Arc<infr_core::hostpager::HostPager>>>> {
    let unpaged = || vec![None; classes.len()];
    // `0` is the explicit OFF switch, not "unset" — so it must reach `Requested` unfiltered.
    let explicit = infr_core::hostmem::Requested::from_config(
        ec.paging.dram.map(|s| s.resolve(0)),
        ec.paging.dram_bypass,
    );
    let available = infr_core::hostmem::available_bytes();
    let pageable: u64 = classes.iter().map(|&(s, n)| (s * n) as u64).sum();
    let budget =
        match infr_core::hostmem::streaming_arena_plan(explicit, available, unified, pageable) {
            // Unified memory: the arena above is already GPU-accessible RAM, so there is nothing
            // to cache down here — but its misses still come from BLOCK reads instead of the
            // mapping, which is what lets a big model run on these machines at all.
            infr_core::hostmem::ArenaPlan::StreamOnly => {
                let io = std::sync::Arc::new(
                    infr_core::blockio::FileBlockIo::from_handles(&g.shard_handles())
                        .map_err(|e| anyhow!("{e}"))?,
                );
                tracing::info!(
                    "{what} host tier: streaming DISK -> GPU-accessible RAM with no host cache — \
                     this device's memory IS host memory, so the {} pool(s) above are already the \
                     only useful cache and a second one would hold bytes the GPU cannot read in \
                     place. Raise INFR_CACHE to cache more; INFR_DRAM_CACHE forces a host arena",
                    classes.len(),
                );
                let mut out = Vec::with_capacity(classes.len());
                for &(slot_bytes, _) in classes {
                    let p = infr_core::hostpager::HostPager::stream_only(slot_bytes, io.clone())
                        .map_err(|e| anyhow!("{e}"))?;
                    out.push(Some(std::sync::Arc::new(p)));
                }
                return Ok(out);
            }
            infr_core::hostmem::ArenaPlan::Take(n) => {
                if explicit == infr_core::hostmem::Requested::Auto {
                    tracing::info!(
                    "{what} host tier: sized automatically to {:.2} GB of {:.2} GB available host \
                     memory (set INFR_DRAM_CACHE to override)",
                    n as f64 / 1e9,
                    available.unwrap_or(0) as f64 / 1e9,
                );
                }
                n as usize
            }
            infr_core::hostmem::ArenaPlan::Skip(why) => {
                use infr_core::hostmem::Skip;
                match why {
                    Skip::NoProbe => tracing::warn!(
                    "{what} host tier: this model must stream, but there is no host-memory probe \
                     on this platform, so the arena cannot be sized automatically — set \
                     INFR_DRAM_CACHE to page weights through DRAM instead of leaving them to the \
                     OS page cache"
                ),
                    Skip::TooLittle => tracing::warn!(
                    "{what} host tier: this model must stream, but only {:.2} GB of host memory \
                     is available — too little to seat a useful arena, so the weights stay on the \
                     OS page cache. Free memory, or set INFR_DRAM_CACHE explicitly",
                    available.unwrap_or(0) as f64 / 1e9,
                ),
                    // Off by name: say so once, because a user who set it and then wonders why
                    // streaming is slow should find the reason in the log.
                    Skip::Disabled => tracing::info!(
                        "{what} host tier: disabled by paging.dram=0 — weights stream from the OS \
                     page cache instead of an arena"
                    ),
                    // Unreachable here by construction: `streaming_arena_plan` is for callers already
                    // past the fit decision, so it never answers `Fits`.
                    Skip::Fits => tracing::warn!("{what} host tier: not built (weights fit)"),
                }
                return Ok(unpaged());
            }
        };
    let slots = infr_core::hostpager::plan_slots(budget, classes);
    if slots.iter().all(|&n| n == 0) {
        tracing::warn!(
            "{what} host tier: a {:.2} GB budget seats no block class of this model — keeping \
             the mmap path (raise INFR_DRAM_CACHE past one block's size to page)",
            budget as f64 / 1e9,
        );
        return Ok(unpaged());
    }
    let io = std::sync::Arc::new(
        infr_core::blockio::FileBlockIo::from_handles(&g.shard_handles())
            .map_err(|e| anyhow!("{e}"))?,
    );
    let mut arena = 0usize;
    let mut out = Vec::with_capacity(classes.len());
    for (&n_slots, &(slot_bytes, _)) in slots.iter().zip(classes) {
        if n_slots == 0 {
            out.push(None);
            continue;
        }
        let p = infr_core::hostpager::HostPager::new(n_slots, slot_bytes, io.clone())
            .map_err(|e| anyhow!("{e}"))?;
        arena += p.arena_bytes();
        out.push(Some(std::sync::Arc::new(p)));
    }
    tracing::info!(
        "{what} host tier: {} of {} pool(s) paged, {:.2} GB arena of a {:.2} GB budget",
        slots.iter().filter(|&&n| n > 0).count(),
        classes.len(),
        arena as f64 / 1e9,
        budget as f64 / 1e9,
    );
    Ok(out)
}

fn cpu_upload_bind(be: &CpuBackend) -> Box<BindWeight<'_>> {
    cpu_bind_with(be, None, None)
}

/// The CPU binder. With a `store`, a weight whose bytes come straight off disk (no load-time
/// rewrite) and whose size class has a pool is registered and read on demand; everything else takes
/// the same paths as before.
fn cpu_bind_with<'a>(
    be: &'a CpuBackend,
    store: Option<std::sync::Arc<infr_cpu::paged::PagedWeights>>,
    n_expert: Option<usize>,
) -> Box<BindWeight<'a>> {
    Box::new(move |name, tb, dt, _n| {
        if let Some(store) = &store {
            // `file_ranges` is `None` exactly for bytes the loader REWROTE (the qwen2 q/k permute,
            // the BitNet dequant), which correspond to nothing on disk and so cannot be re-read.
            if let Some(ranges) = tb.file_ranges() {
                // An expert bank goes to the per-expert pool when there is one, so that a pass
                // reads only its routed experts; a bank that pool declines is offered whole below.
                if let Some(n) = n_expert.filter(|_| is_expert_bank(name)) {
                    if let Some(bank) = store
                        .register_experts(&ranges, n)
                        .map_err(|e| anyhow!("{e}"))?
                    {
                        let nbytes = ranges.iter().map(|(_, l)| l).sum();
                        return Ok((be.paged_experts_weight(store.clone(), bank, nbytes), dt));
                    }
                }
                if let Some(id) = store.register(&ranges).map_err(|e| anyhow!("{e}"))? {
                    let nbytes = ranges.iter().map(|(_, l)| l).sum();
                    return Ok((be.paged_weight(store.clone(), id, nbytes), dt));
                }
            }
        }
        cpu_bind_resident(be, tb, dt)
    })
}

/// The non-paged half of the CPU binder: map an mmap slice zero-copy, or materialize owned/fused
/// bytes into one host buffer.
fn cpu_bind_resident(be: &CpuBackend, tb: WBytes, dt: DType) -> AResult<(Box<dyn Buffer>, DType)> {
    match tb {
        WBytes::Mmap(tb) => Ok((be.map_weight(tb), dt)),
        // Owned bytes (a load-time rewrite) and a fused group both become one host buffer here.
        // The concat is materialized only on this arm — a binder that pages the group never
        // reaches it.
        other => {
            let v = other.materialize();
            let buf = be
                .alloc(v.len().max(1), BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(buf.as_ref(), &v).map_err(|e| anyhow!("{e}"))?;
            Ok((buf, dt))
        }
    }
}

/// The Metal seam weight binder (raw native-dtype upload; the backend dequantizes lazily), hoisted
/// so the Metal session runner and `verify_dense_metal2` share one copy.
#[cfg(target_os = "macos")]
fn metal_upload_bind(be: &infr_metal::MetalBackend) -> Box<BindWeight<'_>> {
    Box::new(move |_name, tb, dt, _n| {
        let buf = be
            .alloc(tb.len().max(1), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(buf.as_ref(), &tb.materialize())
            .map_err(|e| anyhow!("{e}"))?;
        Ok((buf, dt))
    })
}

// ─── Qwen3 dense CPU decode runner ───────────────────────────────────────────────
//
// Builds the n=1 decode Graph and drives it through `CpuBackend`, one token at a time, for BOTH
// prompt ingestion (looped) and generation — so no GEMM/flash prefill kernels are needed on CPU.
// The KV cache grows one row per step. Validates the agnostic seam end-to-end against the GPU path.

/// A CPU backend kept with what it costs to rebuild: its paged-weight store (`paging.dram`) and the
/// weights bound for it. A [`crate::SeamModel`] holds one between generations, so the bound
/// weights, the backend's derived-weight caches (keyed by those buffers' uids, which stay the
/// same) and the host arena's resident blocks all carry over to the next request instead of being
/// rebuilt by it. The spin pool does not: each generation releases it
/// ([`CpuBackend::release_pool`]) and the next builds its own.
///
/// One generation at a time: a caller takes the resident out of its model for the whole call and
/// puts it back afterwards (`SeamModel::with_cpu`), so nothing in it is ever used by two
/// generations at once. A generation that finds none there opens its own.
pub(crate) struct CpuResident {
    be: CpuBackend,
    store: Option<std::sync::Arc<infr_cpu::paged::PagedWeights>>,
    /// Set by the first generation whose cold init got as far as binding every weight.
    bound: Option<weights::BoundWeights>,
}

impl CpuResident {
    /// A resident on `be` with nothing bound yet. The paged store is sized here, once, from the
    /// host memory available now.
    pub(crate) fn open(be: CpuBackend, ec: &EngineConfig, g: &Gguf, cfg: &Config) -> AResult<Self> {
        Ok(Self {
            store: cpu_paged_store(ec, g, cfg)?,
            be,
            bound: None,
        })
    }

    /// Whether a generation bound the weights. Only then is the resident worth keeping: a cold init
    /// that failed part of the way may have registered some weights with the paged store, and the
    /// next attempt would register them again.
    pub(crate) fn is_bound(&self) -> bool {
        self.bound.is_some()
    }

    /// The bound weights' identity, for a test to see that a later generation reused them.
    #[cfg(test)]
    pub(crate) fn weights_ptr(&self) -> Option<*const weights::SeamWeights> {
        self.bound.as_ref().map(|b| b.weights_ptr())
    }

    /// The paged store's `(hits, bytes read from the model file)` since it opened; `None` without
    /// one.
    #[cfg(test)]
    pub(crate) fn paged_counts(&self) -> Option<(u64, u64)> {
        self.store.as_ref().map(|st| {
            st.pool_stats().iter().fold((0, 0), |(h, b), p| {
                (h + p.stats.pager.hits, b + p.stats.bytes_read)
            })
        })
    }

    /// The paged store's identity; `None` without one.
    #[cfg(test)]
    pub(crate) fn store_ptr(&self) -> Option<*const infr_cpu::paged::PagedWeights> {
        self.store.as_ref().map(std::sync::Arc::as_ptr)
    }
}

/// Greedy CPU generation for a decoder (Qwen3 / Llama / Gemma 3 / Gemma 4 dense+E2B / qwen3moe). The
/// attention block is shared; the FFN is either a dense gated FFN or a routed-expert MoE bank; gemma4
/// E2B adds per-layer input embeddings + KV-layer sharing. `prompt` is the full token prefix; returns
/// the generated continuation. Stops at EOS or `max_new`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_dense_cpu(
    res: &mut CpuResident,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    req: Option<&crate::sampling::RequestCtx>,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_cpu_mode(
        res, g, cfg, ec, token_embd, ple, prompt, max_new, None, // constraint
        req, on_token,
    )
}

/// [`generate_dense_cpu`] with an optional grammar constraint, on a caller-supplied
/// [`CpuResident`], so a comparison can pick one on the REFERENCE backend
/// ([`CpuBackend::reference`]) instead of the production int8-activation kernels. The distinction
/// matters at low bit-widths: the int8 activation quant carries ~4e-3 relative error on every quant
/// dtype, which flips greedy tokens at Q2_K and can still flip a near tie at Q4_K — so a
/// backend-vs-backend token comparison must be scored against the reference mode.
///
/// Each call allocates its own KV cache, sized to `prompt + max_new + 1`, and starts it empty: the
/// resident carries the backend and its weights between calls, never a conversation.
#[cfg_attr(infr_profile, infr_prof::instrument)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_dense_cpu_mode(
    res: &mut CpuResident,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    constraint: Option<&mut crate::grammar::Constraint>,
    req: Option<&crate::sampling::RequestCtx>,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    // Thin CPU wrapper over the backend-generic runner: a CpuBackend + a weight binder that maps
    // each tensor straight from the GGUF mmap (no alloc, no memcpy) — or, when `paging.dram` asks
    // for a host weight cache, registers the big ones to be read from the file on demand. Once
    // the resident holds bound weights the binder is never called.
    let mut state = None;
    let out = {
        let bind = cpu_bind_with(
            &res.be,
            res.store.clone(),
            cfg.moe.as_ref().map(|m| m.n_expert),
        );
        runner::generate_dense_backend_bound(
            res.bound.as_ref(),
            &res.be,
            &bind,
            g,
            cfg,
            ec,
            token_embd,
            ple,
            prompt,
            max_new,
            on_token,
            &mut state,
            prompt.len() + max_new + 1,
            constraint,
            None, // verify
            None, // verify_ids
            None, // logits_out
            None, // h_out
            None, // denoise_req
            req,
            None, // prefill_logits
            None, // route
        )
    };
    // A cold init that got as far as building `state` left the bound weights in it, whole, even
    // when the generation failed after that; they are kept. One that failed earlier (a weight that
    // would not bind, a refused KV allocation) left `state` empty, so nothing is kept, and
    // `SeamModel::with_cpu` drops a resident that never got bound weights.
    match (&res.bound, &state) {
        (None, _) => res.bound = state.as_ref().map(SeamKv::bound),
        (Some(b), Some(kv)) => debug_assert!(
            kv.shares(b),
            "the generation bound its own weights beside the resident's"
        ),
        (Some(_), None) => {}
    }
    // Kept: the caches. Not kept: the spin pool, which measured slower across requests than
    // built per request (`CpuBackend::release_pool`).
    res.be.release_pool();
    if let Some(store) = &res.store {
        // Flushed where the stats are taken, so the trace on disk covers what they report.
        if let Err(e) = store.flush_trace() {
            tracing::warn!("[host pager] {e}");
        }
        if ec.paging.stats {
            report_host_paging(store);
        }
    }
    out
}

/// `paging.stats` (`INFR_PAGER_STATS=1`): what the host tier actually did, per size class, since
/// the store opened — on a model's [`CpuResident`], since its first CPU generation, so each
/// report includes every earlier request's counts.
///
/// The hit rate alone cannot distinguish a tier that is working from one that was never asked, so
/// the read count and the bytes moved are reported beside it: those are what a run's throughput is
/// bounded by, and what the mmap baseline in `docs/perf/results.md` is compared against.
fn report_host_paging(store: &infr_cpu::paged::PagedWeights) {
    for p in store.pool_stats() {
        let s = p.stats;
        let what = if p.experts { "expert" } else { "whole" };
        tracing::info!(
            "[host pager] {what} {:.1}MB x {} slots: {} hits, {} misses ({:.1}% hit), {} \
             evictions, {} reads, {:.2} GB from disk in {:.1}s of reads ({} re-reads in {:.1}s)",
            p.slot_bytes as f64 / 1e6,
            p.n_slots,
            s.pager.hits,
            s.pager.misses,
            s.pager.hit_rate() * 100.0,
            s.pager.evictions,
            s.reads,
            s.bytes_read as f64 / 1e9,
            s.read_ns as f64 / 1e9,
            s.rereads,
            s.reread_ns as f64 / 1e9,
        );
    }
    tracing::info!(
        "[host pager] {:.1}s waiting for weights to become resident",
        store.wait().as_secs_f64()
    );
}

/// GPU seam runner: the SAME dense forward as [`generate_dense_cpu`], but on the Vulkan backend
/// through the agnostic [`Graph`] adapter (weights padded + uploaded to VRAM instead of mmap-mapped).
/// This is the end-to-end GPU parity/perf path — running it and diffing the CPU oracle proves the
/// adapter, and its decode tok/s (still recompiling the graph per token) is the baseline
/// record-once replay must close. Prefill's batched attention is decode-only on the seam, so the
/// caller may pass short prompts to force the per-token path.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_vulkan(
    vk: &infr_vulkan::VulkanBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    let _scope = enter_model_scope(vk, cfg);
    generate_dense_vulkan_session(
        vk,
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
        &mut None,
        prompt.len() + max_new + 1,
        None, // constraint
        None, // req: the one-shot runner is a sole sequence — config sampling, no gate
    )
}

/// The binder of a WARM call: every weight is already bound, so asking it for one is a broken
/// invariant, reported by name.
fn warm_binder<'a>() -> Box<BindWeight<'a>> {
    Box::new(|name: &str, _tb, _dt, _n| Err(anyhow!("warm session must not re-bind {name}")))
}

/// [`generate_dense_vulkan`] with a caller-held [`SeamKv`]: hold `state` (+ a `want_ctx` capacity)
/// across calls and each turn prefills only the suffix that differs from the cached tokens —
/// ChatSession-style KV reuse on the agnostic seam.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_vulkan_session(
    vk: &infr_vulkan::VulkanBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
    state: &mut Option<SeamKv>,
    want_ctx: usize,
    constraint: Option<&mut crate::grammar::Constraint>,
    req: Option<&crate::sampling::RequestCtx>,
) -> AResult<(Vec<u32>, GenStats)> {
    // A caller-held `state` was sized against the pins of the scope it was made in, so the caller
    // must hold that scope across calls: a fresh one here would forget a chunk placement pinned.
    debug_assert_scope_for(vk);
    // The model's chunk cap, for a caller whose scope was made without it. Before placement,
    // which prices the chunk.
    pin_model_chunk_cap(cfg);
    // Placement can allocate + upload (the pager arenas, a weight re-bind), i.e. it RECORDS on the
    // Vulkan command pool — so it takes a turn on the baton like any other GPU region. Scoped: the
    // baton is released before the runner starts stepping. See `StepGate`.
    //
    // A WARM call (`state.is_some()`) has every weight already resident, and the runner never calls
    // `bind_weight` again (see the `state.is_none()` init block in `generate_dense_backend`), so
    // building the full `vulkan_moe_binder` — which re-resolves the placement tiers and allocs a
    // Box — is pure per-turn waste. Skip it: a no-op binder that errors loudly if the invariant
    // ever breaks.
    let bind = if state.is_some() {
        warm_binder()
    } else {
        let _gp = req.and_then(|r| r.gate_pass());
        vulkan_moe_binder(vk, g, cfg, ec, true, want_ctx)?
    };
    let out = generate_dense_backend(
        vk, &*bind, g, cfg, ec, token_embd, ple, prompt, max_new, on_token, state, want_ctx,
        constraint, None, None, None, None, None, req,
    )?;
    // INFR_PAGER_STATS=1: cumulative hit/miss/eviction counters since this pager was installed
    // (persists across calls on the same session — see `MoePagerSession`). A no-op when no paged
    // model is loaded. Printed every call rather than gated to "last call only" since neither the
    // CLI's run/serve loop nor this function know which call is the process's last one.
    vk.print_moe_pager_stats();
    vk.print_dense_pager_stats();
    Ok(out)
}

/// Honest activation/scratch reservation for a DENSE model's placement decision: the transient
/// VRAM a resident session needs BEYOND weights + KV, at the largest shape it will ever run — a
/// full prefill chunk of `rows = min(ubatch, want_ctx)` rows (the runner chunks batched prefill at
/// INFR_UBATCH, default 1024; decode's single row is dwarfed by this).
///
/// Every term is per ROW of the prefill chunk, and the whole estimate is now checked against the
/// backend's own high-water mark of live activation bytes at the end of every generation (see
/// `Backend::activation_peak` and the runner's `activation reserve too low` warning), so the
/// numbers below are a fit to measurements rather than an argument:
/// - Internal graph tensors (`alloc_scratch`): fused gate_up out `[rows, 2*n_ff]` f32 + activated
///   intermediate `[rows, n_ff]` f32 + fused qkv staging + ~a dozen `[rows, n_embd]`-class f32/f16
///   temps. Modeled as `12*n_ff + 96*n_embd` per row (the n_embd umbrella also absorbs the
///   lin_a16/mmq activation-quant pools, which are n_embd/n_ff-wide f16/i8).
/// - `nonfa_pv`/`flash_po`: `8*rows*n_head*head_dim*4` per DISTINCT head shape — gemma4 alternates
///   SWA(256)/full(512) head dims, so BOTH pools live at once →
///   `32*n_head*(head_dim + head_dim_swa-if-distinct)` per row.
/// - `nonfa_s` (score tiles, non-flash tier only — any model with SWA layers or head_dim != 128):
///   `n_head*rows*kv_pad*2`, kv_pad = kv_len rounded up to 256, i.e. `2*n_head*ctx_pad` per row at
///   the final context. ONE live tile, not two: the pool key includes the byte size, so a deep
///   prefill does allocate a fresh (larger) tile per chunk, but each chunk's `execute` drops its
///   pool before the next one builds — and every layer of a chunk shares one kv_len, so one size
///   is live at a time. Uniform-hd-128 no-SWA models (llama/qwen3) ride the single-pass flash
///   tier: no score tiles, only the (negligible) flash_pm/pl partials — term skipped.
///
/// Times [`ACT_RESERVE_PAD`]. What is deliberately NOT here any more: a fixed 256 MiB that stood
/// in for gpu-allocator block granularity, retained upload staging and weight-buffer padding.
/// Those are not activations at all — they are exactly what the runner's post-load re-clamp
/// ([`reclamp_ctx_to_live_room`]) prices by ASKING the device, so carrying an estimate of them
/// here as well is double-counting, and the estimate was 3.5x wrong at a 128-row chunk.
///
/// Always taken at an EXPLICIT chunk height: every caller (the try-resident sweep, the streaming
/// budget, the context-fit math) walks [`ubatch_candidates`] rather than assuming the default
/// 1024-row chunk — assuming it is what let the KV-format decision and the placement decision
/// disagree.
pub(crate) fn dense_act_reserve_at(cfg: &Config, want_ctx: usize, ubatch: usize) -> u64 {
    // Prefill GEMM outputs pad rows to 64 (see the Vulkan adapter's `alloc_scratch`).
    let rows = ubatch.min(want_ctx).max(1).next_multiple_of(64) as u64;
    // Attention pv accumulators: one pool per distinct (n_head, head_dim) shape.
    let hd_shapes = if cfg.swa_window > 0 && cfg.head_dim_swa != cfg.head_dim {
        cfg.head_dim + cfg.head_dim_swa
    } else {
        cfg.head_dim
    };
    let attn_pv = 32 * cfg.n_head * hd_shapes;
    // Non-flash score tiles (see the doc above): skipped for uniform-hd-128 no-SWA models.
    // When ONLY the SWA layers miss the flash tier (hd == 128, e.g. gemma3-12b: full layers are
    // Causal+hd128 = flash), the widest score tile is the SWA ring's `window + chunk` rows, not
    // the full context.
    let attn_s = if cfg.swa_window == 0 && cfg.max_head_dim() == 128 {
        0
    } else {
        let kv_span = if cfg.max_head_dim() == 128 {
            want_ctx.min(cfg.swa_window + ubatch)
        } else {
            want_ctx
        };
        2 * cfg.n_head * kv_span.next_multiple_of(256)
    };
    // MoE expert scratch (`moe_*` / `moe_pgb_*` in the Vulkan adapter). Its pools are sized by
    // (row, expert) PAIRS, not rows — the batched path buckets every row's `n_used` picks and runs
    // them through the expert FFN together — so the per-row cost carries an `n_used` multiplier.
    // Per pair: the gate+up output `2*n_ff_exp` f32, the activated intermediate `n_ff_exp` f32,
    // the down-projection's f32 result `n_embd`, and the int8 activation-quant pools (one byte per
    // element plus two f16 scales per 32-element block, on both the `n_embd` and `n_ff_exp` sides).
    let moe = cfg.moe.as_ref().map_or(0, |m| {
        let per_pair = 3 * m.n_ff_exp * 4 + cfg.n_embd * 4 + m.n_ff_exp + cfg.n_embd;
        m.n_used * per_pair
    });
    // qwen35's gated-DeltaNet mixer scratch, which the `n_embd` umbrella above does not cover: its
    // buffers are keyed on the SSM dims, not on n_embd, and a hybrid model held 1.53x the umbrella
    // (Qwen3.5-9B, `activation reserve too low` at a 1024-row chunk). Named one for one after the
    // `dn_*` internals the runner's graph declares, all f32 and all `batch`-wide, so the two lists
    // can be read side by side: qkv + conv out (conv channels each), z (d_inner), q + k (key dim
    // each), v + out (value dim each), beta + alpha (one per v-head).
    // Plus its attention out-gate pair (`qg` + `gate_a`), which every arch declares but only this
    // one makes big: qwen35 interleaves q and gate in one projection, so `qg` is DOUBLE the q
    // width and the umbrella's n_embd term no longer covers the three of them.
    let deltanet = if cfg.qwen35 {
        4 * (2 * cfg.q35_conv_channels()
            + cfg.ssm_d_inner
            + 2 * cfg.q35_num_k_heads() * cfg.q35_head_k_dim()
            + 2 * cfg.q35_num_v_heads() * cfg.q35_head_v_dim()
            + 2 * cfg.q35_num_v_heads())
            + 12 * cfg.n_head * cfg.max_head_dim()
    } else {
        0
    };
    // LFM's scratch (the runner's `lfm_scratch`): the B | C | X
    // in-projection (three n_embd), the conv output (n_embd) and the per-head gate logits.
    let shortconv = if cfg.lfm3 {
        4 * (4 * cfg.n_embd + cfg.n_head)
    } else {
        0
    };
    let per_row =
        (12 * cfg.n_ff + 96 * cfg.n_embd + attn_pv + attn_s + moe + deltanet + shortconv) as u64;
    rows * per_row * ACT_RESERVE_PAD.0 / ACT_RESERVE_PAD.1
}

/// Activation bytes LAYER-MAJOR prefill holds ON TOP of [`dense_act_reserve_at`]: the residual
/// stream for the WHOLE prompt at once — one `[ubatch, n_embd]` f32 buffer per chunk, all live
/// across the layer boundary — where chunk-major keeps only the chunk it is running.
///
/// Priced at the full context because that is the longest prompt the session can be handed, and
/// these buffers are allocated mid-prefill out of whatever the arenas left: an under-reserve
/// surfaces as a VRAM-guard refusal on a long prompt, not as a smaller cache. A short prompt
/// really does allocate less, so on a session that never fills its window this is held back and
/// unused — the price of sizing an arena before the prompts arrive.
///
/// Zero when the session prefills chunk-major ([`layer_major_prefill`]).
pub(crate) fn layer_major_act_bytes(cfg: &Config, want_ctx: usize, ubatch: usize) -> u64 {
    let ctx = want_ctx.max(1);
    let ub = ubatch.clamp(1, ctx);
    ctx.div_ceil(ub) as u64 * ub as u64 * cfg.n_embd as u64 * 4
}

/// May this session prefill LAYER-MAJOR — every chunk through layer L before any chunk reaches
/// L+1 — instead of running the whole model per chunk? Whether a given prompt then does is
/// [`layer_major_this_prompt`].
///
/// The two orders compute the same thing; they differ in what they re-read. Chunk-major sweeps the
/// entire weight set once PER CHUNK, which is free when the weights are resident and is the whole
/// prefill cost when they stream: measured on Qwen3-14B Q8_0 at ctx 4096, the 1024-row default read
/// 25.26 GB against a single sweep's 6.32 GB, and prefilled 3.08x slower for it. Layer-major reads
/// one sweep regardless of chunk count, at the cost of holding every chunk's residual stream at once
/// ([`layer_major_act_bytes`]) — a trade that only pays when there is I/O to save, hence the default
/// of "on exactly when the weights stream" (and, per prompt, only when it can save a read).
///
/// `paging.layer_major` overrides in both directions (A/B, and the only way to put a RESIDENT model
/// on this path). Either way it needs a backend that carries a bound `Input` from one execute to the
/// next, which is what threads the residual stream between two layers' dispatches, and an
/// architecture whose layer stack can be entered past layer 0 at all (`spannable`).
pub(crate) fn layer_major_prefill(
    ec: &EngineConfig,
    caps: &infr_core::backend::Capabilities,
    streaming: bool,
    spannable: bool,
) -> bool {
    let want = ec.paging.layer_major.unwrap_or(streaming);
    if want && !spannable {
        // gemma4-E2B: its layer stack reads `per_layer_inp`, which the graph PROLOGUE builds, so a
        // span starting past layer 0 would read an unbound tensor. This is the gate — the matching
        // `assert!` in `build` is an internal invariant, and a streamed E2B model reached it as a
        // PANIC on an ordinary `bench`/`run` until this arm existed. (DeepSeek V4's wider residual
        // is carried by the spans themselves: `DecodeHandles::hc_resid`.)
        tracing::warn!(
            "layer-major prefill cannot split this architecture's layer stack (gemma4-E2B's \
             per-layer inputs are built by the graph prologue) — prefilling chunk-major"
        );
        return false;
    }
    if want && !caps.graph_input_inplace {
        // Only reachable through the explicit override on a wrapper backend (TP/EP/pipeline) —
        // the auto rule needs a dense pager, which those do not host. Say so rather than silently
        // running the other order.
        tracing::warn!(
            backend = caps.name,
            "layer-major prefill needs a backend that carries a bound graph Input across \
             executes; this one does not — prefilling chunk-major"
        );
        return false;
    }
    want
}

/// Does THIS prompt prefill layer-major, given whether the session would
/// ([`layer_major_prefill`])?
///
/// The auto rule runs it only where it can save a read: a prompt of more than one chunk, on a
/// cache that cannot hold every block. One chunk is one sweep in either order, and a cache with a
/// slot per block reads each block once per session whatever the order, so layer-major there only
/// adds a span per layer, each paying its own graph build, compile, scratch and blocking submit.
/// `paging.layer_major = true` still forces it (the A/B arm); `false` never arrives as wanted.
///
/// The activation reserve does not follow this per-prompt answer: it is priced for the session
/// ([`layer_major_act_bytes`]) because the arenas are sized before any prompt arrives, so a prompt
/// that runs chunk-major leaves it unused rather than short.
pub(crate) fn layer_major_this_prompt(
    ec: &EngineConfig,
    wanted: bool,
    n_chunks: usize,
    holds_every_block: bool,
) -> bool {
    wanted && (ec.paging.layer_major == Some(true) || (n_chunks > 1 && !holds_every_block))
}

/// Whether a prompt whose first prefill chunk is `first_chunk_rows` tall prefills layer-major
/// when `cfg`'s weights stream through the CPU's host arena (`Backend::host_streams`);
/// [`layer_major_this_prompt`] still asks for more than one chunk. `paging.layer_major`
/// overrides it either way.
///
/// **DeepSeek V4 only, deliberately.** It is the family this was built and measured for, and no
/// other family has been checked bit for bit against its chunk-major prefill on the CPU.
///
/// **A sole request only** (`shared` is a request that takes turns with others,
/// `RequestCtx::shares_gpu`, as `infr serve --parallel` makes). Only sole requests were
/// measured; a shared one prefills in shorter chunks (`device.ubatch_parallel`), and the
/// layer-major residency hints (hot-end inserts, the demotion of each finished layer) assume one
/// prefill owns the pool's recency order, where chunk-major's cold-end inserts were chosen to
/// leave other sessions' decode working sets alone.
///
/// **A first chunk of at least [`HOST_LAYER_MAJOR_MIN_ROWS`] only.** The gate reads the FIRST
/// chunk, whose height `dsv4_emit::prefill_chunk_rows` derives from the session's chunk height
/// and the prefill's START position: on a file with a CSA layer it shrinks the chunk as the
/// context grows. So a prompt prefilled from position 0 at the default height qualifies at any
/// length (its later chunks may shrink below the minimum and still run layer-major), while a turn
/// that extends a session from deep in its context may get a first chunk below the minimum and
/// prefill chunk-major (`layer_major_gate_follows_the_first_chunk` pins where). Measured on V4-Flash with a 12 GB
/// arena (`docs/deepseek.md` § "Layer-major prefill on the CPU"): at the default 1024-row chunk a
/// 1443-token prompt prefilled in 206.8 s layer-major, where chunk-major had spent 209.5 s on its
/// first chunk alone; in 32-row chunks a 69-token prompt was SLOWER layer-major (98.96 and 114.89
/// s against 91.59 s). A small chunk routes to a small part of each bank, so chunk-major re-reads
/// little per chunk, and what it re-reads it had read seconds before, which the OS file cache
/// largely still holds; layer-major's reads are all first reads.
pub(crate) fn host_layer_major(cfg: &Config, first_chunk_rows: usize, shared: bool) -> bool {
    cfg.deepseek4 && !shared && first_chunk_rows >= HOST_LAYER_MAJOR_MIN_ROWS
}

/// The shortest first prefill chunk [`host_layer_major`] runs layer-major. The two measured
/// heights are 32 rows (slower) and 1024 (faster); nothing between them was run, and this sits
/// between them so that a sole request keeps the order while V4's long-context shrink
/// (`dsv4_emit::prefill_chunk_rows`) leaves its first chunk at least this tall.
pub(crate) const HOST_LAYER_MAJOR_MIN_ROWS: usize = 256;

/// What a layer-major prefill may hold between layers for its whole segment of the prompt: the
/// residual stream (DeepSeek V4's `hc_mult` streams wide) and the chunk's embedded input rows.
/// Taken out of an automatic CPU arena too small for the weights ([`layer_major_carry_bytes`]);
/// `layer_major_segments_carry_one_gib_at_v4_flash_geometry` pins the rows it gives V4-Flash.
pub(crate) const LAYER_MAJOR_CARRY_BYTES: u64 = 1 << 30;

/// The bytes one prompt row costs a layer-major prefill between layers: the f32 residual stream
/// (`hc_mult` streams of `n_embd` on DeepSeek V4, one elsewhere), the embedded input row the first
/// span reads, and its token id and position.
pub(crate) fn layer_major_row_bytes(cfg: &Config) -> u64 {
    let streams = if cfg.deepseek4 { cfg.hc_mult as u64 } else { 1 };
    (streams + 1) * cfg.n_embd as u64 * 4 + 8
}

/// The most prompt rows one layer-major SEGMENT carries ([`layer_major_segments`]):
/// `paging.layer_major_rows` when set, else, on DeepSeek V4, as many as
/// [`LAYER_MAJOR_CARRY_BYTES`] holds at [`layer_major_row_bytes`]. Every other family carries the
/// whole prompt, as it always has (its residual is priced for the whole context by
/// [`layer_major_act_bytes`]).
pub(crate) fn layer_major_segment_rows(ec: &EngineConfig, cfg: &Config) -> usize {
    match ec.paging.layer_major_rows {
        Some(n) => n.max(1),
        None if cfg.deepseek4 => {
            usize::try_from(LAYER_MAJOR_CARRY_BYTES / layer_major_row_bytes(cfg))
                .unwrap_or(usize::MAX)
                .max(1)
        }
        None => usize::MAX,
    }
}

/// Group consecutive prefill `chunks` (`(start, end)` positions) into segments of at most
/// `max_rows` rows, as ranges of chunk indices; a chunk taller than `max_rows` is a segment alone.
/// A layer-major prefill runs each segment through every layer before the next segment starts, so
/// it holds one segment's residual streams at a time and sweeps the weights once per segment.
pub(crate) fn layer_major_segments(
    chunks: &[(usize, usize)],
    max_rows: usize,
) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let (mut first, mut rows) = (0, 0usize);
    for (i, &(s, e)) in chunks.iter().enumerate() {
        let r = e - s;
        if i > first && rows + r > max_rows {
            out.push(first..i);
            (first, rows) = (i, 0);
        }
        rows += r;
    }
    if first < chunks.len() {
        out.push(first..chunks.len());
    }
    out
}

/// Safety pad on [`dense_act_reserve_at`]'s per-row terms, as `(numerator, denominator)`.
///
/// **What it covers.** The terms above name the pools a forward allocates, but WHICH pools it
/// allocates is a tier decision the Vulkan adapter takes per layer, per op, on row count / head
/// dim / mask / KV dtype / coopmat capability — and the per-arch algebra is fit to the graphs that
/// have been measured, not to the ones that have not. This pad is that distance.
///
/// **Sized by measurement, against the backend's own high-water mark** (`Backend::activation_peak`
/// — every row below is reproducible with `RUST_LOG=infr_llama=debug` on the run named, and the
/// runner warns when a peak lands above the reserve). Unpadded model against measured peak, all on
/// a 24 GiB 7900 XTX:
///
/// | model                | chunk | ctx     | modeled MiB | measured MiB | measured / modeled |
/// | -------------------- | ----- | ------- | ----------- | ------------ | ------------------ |
/// | Qwen3.5-4B-MTP Q4_K_M| 1024  |   2 064 |         724 |        1 027 |              1.42x |
/// | Qwen3.5-9B Q4_K_M    | 1024  |   2 064 |         904 |        1 112 |              1.23x |
/// | Llama-3.2-1B Q4_K_M  | 1024  |   2 064 |         406 |          429 |              1.06x |
/// | Qwen3-30B-A3B Q4_K_M | 1024  |     528 |         309 |          291 |              0.94x |
/// | gemma-3-12b Q4_K_M   | 1024  | 131 072 |       3 811 |        4 735 |              1.24x |
/// | gemma-4-31B UD-Q5_K_XL| 128  |  15 440 |         261 |          334 |              1.28x |
///
/// 1.5 is the first rung above the worst of them (the MTP 4B at 1.42x). The hybrid archs are what
/// set it: their DeltaNet mixer and double-width q projection are named terms now, and the residue
/// is still the largest — which is the argument for deriving these bytes from the graph the runner
/// already builds rather than re-deriving them here (backlog B8).
const ACT_RESERVE_PAD: (u64, u64) = (3, 2);

/// Batched-prefill micro-batch: rows per prefill chunk (`device.ubatch` / `INFR_UBATCH`, else what
/// placement pinned, else the default for the backend class the current [`PlacementPins`] were
/// made for — [`default_ubatch_rows`], smaller on an INTEGRATED GPU). ONE reader funnel — the
/// prefill loop, the activation reserve, and the SWA ring sizing below all derive from this,
/// because the ring's correctness bound is "window + one whole prefill chunk".
///
/// **`ubatch_specified` IS needed.**
/// `INFR_UBATCH` is a two-consumer knob — the VALUE here, and the PRESENCE the placement
/// sweeps read ([`user_pinned_ubatch`]) — and the two DISAGREE about an unusable value, exactly as
/// the KV dtypes do. The old value site was
/// `.parse().ok().filter(|&v| v > 0).unwrap_or_else(pin/default)` and the old presence site was a
/// bare `is_err()` on the raw variable, so `INFR_UBATCH=0` (and `=abc`) yielded NO height
/// while still disabling the sweep. That is not a corner case: `infr … -u 0` is the DOCUMENTED
/// "stay adaptive" spelling and the CLI publishes it verbatim. Collapsing both readers onto
/// `device.ubatch.is_some()` would silently re-enable the residency sweep for it, so `DeviceCfg`
/// carries the presence flag separately and every input — valid value, `0`, garbage, unset — keeps
/// today's behaviour bit-for-bit (R1).
///
/// Whatever the height, the scope's model cap ([`moe_small_m_chunk_cap`], pinned by
/// [`enter_model_scope`] / [`PlacementPins::for_model`]) bounds it — an explicit `INFR_UBATCH`
/// included, since the cap keeps a chunk inside the range the small-m expert path was measured at
/// rather than tuning throughput.
pub(crate) fn ubatch_rows(ec: &EngineConfig) -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    with_placement_pins(|p| {
        let rows =
            ec.device
                .ubatch
                .filter(|&v| v > 0)
                .unwrap_or_else(|| match p.ubatch.load(Relaxed) {
                    0 => p.default_ubatch, // nothing pinned
                    rows => rows,
                });
        match p.moe_chunk_cap.load(Relaxed) {
            0 => rows, // no cap
            cap => rows.min(cap),
        }
    })
}

/// The prefill chunk cap a model needs on Vulkan, or `None`: an MoE model whose expert widths fail
/// [`infr_core::tensor::moe_expert_gemm_widths_ok`] runs every `Op::MoeFfn` through the small-m
/// expert path, at any chunk height, so its chunks are held to
/// [`infr_core::tensor::MOE_SMALL_M_CHUNK_CAP`], the rows that path was measured at. Gate/up are
/// one GEMM of twice `n_ff_exp` when fused ([`Config::dual_moe`], as the runner emits them).
pub(crate) fn moe_small_m_chunk_cap(cfg: &Config) -> Option<usize> {
    let mc = cfg.moe.as_ref()?;
    let gate_up_width = if cfg.dual_moe() {
        2 * mc.n_ff_exp
    } else {
        mc.n_ff_exp
    };
    (!infr_core::tensor::moe_expert_gemm_widths_ok(gate_up_width, cfg.n_embd))
        .then_some(infr_core::tensor::MOE_SMALL_M_CHUNK_CAP)
}

/// Pin [`moe_small_m_chunk_cap`] of `cfg` on the current scope's pins: for a Vulkan run of `cfg`
/// whose scope was made without it. A scope serves one model; the cap only ever lowers. Outside
/// any scope it does nothing: the process-wide [`FALLBACK_PINS`] serve the CPU and Metal, which
/// this cap is not for, and every Vulkan entry point runs in a scope of its own.
pub(crate) fn pin_model_chunk_cap(cfg: &Config) {
    CURRENT_PINS.with(|c| {
        if let Some(p) = c.borrow().as_ref() {
            pin_moe_chunk_cap(p, cfg);
        }
    });
}

/// Pin [`moe_small_m_chunk_cap`] of `cfg` on `pins` (a no-op for a model that needs none).
fn pin_moe_chunk_cap(pins: &PlacementPins, cfg: &Config) {
    if let Some(cap) = moe_small_m_chunk_cap(cfg) {
        lower_moe_chunk_cap(pins, cap);
    }
}

/// Cap `pins`' prefill chunk at `cap` rows. The cap only ever lowers: pins that already hold one
/// keep the smaller.
fn lower_moe_chunk_cap(pins: &PlacementPins, cap: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    // A compare-exchange loop rather than the std helper for it, whose name differs between the
    // toolchains this builds on.
    let mut cur = pins.moe_chunk_cap.load(Relaxed);
    loop {
        let new = if cur == 0 { cap } else { cur.min(cap) };
        match pins
            .moe_chunk_cap
            .compare_exchange_weak(cur, new, Relaxed, Relaxed)
        {
            Ok(_) => break,
            Err(seen) => cur = seen,
        }
    }
}

/// Did the user PIN a prefill chunk height? The PRESENCE half of `INFR_UBATCH` — the dense
/// placement sweeps skip themselves when it is true, because the user's height is authoritative.
/// True even for a value this reader cannot use (`0`, garbage): see [`ubatch_rows`].
pub(crate) fn user_pinned_ubatch(ec: &EngineConfig) -> bool {
    ec.device.ubatch_specified
}

/// The SHRINK ladder the dense placement sweeps walk when the default prefill chunk's activation
/// reserve is what tips a model out of residency: 512 → 256 → 128 rows. A shorter chunk shrinks
/// both the activation reserve (whole-chunk scratch scales with rows) and the SWA ring rows
/// (`window + chunk`), and resident-at-512 decodes ~10x faster than streaming at the PCIe ceiling
/// — so trading prefill chunk height for residency is strictly the right call.
///
/// 128 is the floor: below it the per-dispatch launch overhead dominates prefill entirely.
pub(crate) const DENSE_UBATCH_LADDER: [usize; 3] = [512, 256, 128];

/// Every prefill chunk height a dense placement decision is allowed to settle on, TALLEST FIRST:
/// the current/default height ([`ubatch_rows`]) followed by the [`DENSE_UBATCH_LADDER`] rungs
/// BELOW it. A user-pinned `INFR_UBATCH` is authoritative — the sweeps skip themselves — so the
/// list is then just that one height.
///
/// **One ladder, two readers.** `vulkan_moe_binder`'s residency / auto-q8 / streaming sweeps walk
/// this to pick a height, and `SeamModel::kv_fit_ctx_fmt` walks the SAME list to decide how much
/// context fits: a context is accepted when it fits at ANY height placement could settle on.
/// Before this was shared, the fit math priced the DEFAULT 1024-row chunk's reserve while
/// placement went on to pick 512 — the KV format was chosen against an assumption the very next
/// step abandoned (gemma-3-12b: an unnecessary auto-q8 at ctx 131072, while f16 at a 512-row
/// chunk fits with room to spare). `dense_ubatch_ladder_is_the_only_one` guards the drift.
///
/// Filtering to heights `< ubatch_rows(ec)` matters on an INTEGRATED GPU, whose default chunk is
/// already below 512 ([`default_ubatch_rows`]): an unfiltered ladder would let a "shrink" sweep
/// RAISE the chunk past the watchdog-safe default and trip `VK_ERROR_DEVICE_LOST`.
pub(crate) fn ubatch_candidates(ec: &EngineConfig) -> Vec<usize> {
    let now = ubatch_rows(ec);
    let mut cands = vec![now];
    if !user_pinned_ubatch(ec) {
        cands.extend(DENSE_UBATCH_LADDER.into_iter().filter(|&c| c < now));
    }
    cands
}

/// The prefill chunk a backend of `caps`'s class takes when neither INFR_UBATCH nor the placement
/// sweep pinned one: 1024 rows, EXCEPT on an integrated GPU, where a chunk that big is a single
/// multi-second command buffer and trips the ~10 s GPU watchdog (`ring gfx_0.0.0 timeout` ->
/// `VK_ERROR_DEVICE_LOST`). See [`infr_core::integrated_ubatch_rows`] for the measurements behind
/// the smaller default.
///
/// A DISCRETE device, and the CPU and Metal backends (neither reports `integrated`), take the 1024
/// branch, so no tuned dGPU shape moves. [`PlacementPins::for_caps`] reads it once per session.
fn default_ubatch_rows(caps: &infr_core::backend::Capabilities) -> usize {
    if caps.integrated {
        infr_core::integrated_ubatch_rows(caps.compute_units)
    } else {
        1024
    }
}

/// Enter a placement scope whose default prefill chunk is the one for `caps`, the backend this
/// one-shot actually runs on. The multi-device wrappers need it: a world that mixes a discrete
/// card with an integrated GPU must size every chunk for the integrated one (each wrapper reports
/// its ranks' most conservative class). An explicit `device.ubatch` still wins (see
/// [`ubatch_rows`]); a chunk or auto-q8 KV the enclosing scope already pinned carries over, the
/// chunk lowered to the backend's default when that is smaller. The model `cfg`'s chunk cap
/// ([`moe_small_m_chunk_cap`]) applies, as does one the enclosing scope held.
fn backend_chunk_scope(caps: &infr_core::backend::Capabilities, cfg: &Config) -> PlacementScope {
    use std::sync::atomic::Ordering::Relaxed;
    let pins = PlacementPins::for_model(caps, cfg);
    let rows = pins.default_ubatch;
    with_placement_pins(|cur| {
        let pinned = cur.ubatch.load(Relaxed);
        pins.ubatch
            .store(if pinned == 0 { rows } else { pinned.min(rows) }, Relaxed);
        if cur.kv_q8.get().is_some() {
            let _ = pins.kv_q8.set(());
        }
        let cap = cur.moe_chunk_cap.load(Relaxed);
        if cap != 0 {
            lower_moe_chunk_cap(&pins, cap);
        }
    });
    PlacementScope::enter(std::sync::Arc::new(pins))
}

/// Prefill chunk (rows) for a sequence SHARING the GPU with other in-flight sequences
/// (`infr serve --parallel N`, i.e. the runner's `req` carries a `StepGate`).
///
/// A prefill chunk is unpreemptible GPU: the whole chunk holds the baton, so it is exactly how long
/// a newly-admitted request's prefill stalls every in-flight decode. The solo default (1024 rows,
/// [`ubatch_rows`]) is ~100ms+ on a 14B — a visible hitch across 3 other streams. 256 rows bounds
/// that to ~25-30ms (about the cost of ~4 decode steps) at a small prefill-throughput cost, which
/// is the right trade when N clients are streaming. Never applies to a sole request: `infr run`,
/// `bench`, the goldens, and a `-np 1` server all keep the full [`ubatch_rows`] chunk, so prefill
/// throughput there is UNCHANGED. INFR_UBATCH_PARALLEL overrides; it only ever SHRINKS the chunk
/// (the runner takes the `min` with [`ubatch_rows`]).
pub(crate) fn ubatch_rows_parallel(ec: &EngineConfig) -> usize {
    ec.device.ubatch_parallel
}

/// The two placement decisions the VRAM ladder pins for a session and then keeps STABLE for its
/// whole lifetime (set before the first KV allocation, never changed after — warm calls and rebuilt
/// graphs must agree with the buffers they were sized with):
///   - `ubatch`: the pinned prefill chunk (rows) when the dense try-resident sweep found a smaller
///     chunk is what makes a big model fully resident (residency at a 512-row chunk decodes ~10x
///     faster than streaming at the PCIe ceiling — see `vulkan_moe_binder`'s dense tier). Read by
///     [`ubatch_rows`] when INFR_UBATCH is unset, so the prefill loop, the activation reserve, and
///     the SWA ring sizing all agree on the same height.
///   - `kv_q8`: auto-q8 KV, chosen to keep a session RESIDENT (dense try-resident tier) or to avoid
///     shrinking a DEFAULT context (`SeamModel::clamp_default_ctx`) — the "q8 KV" rung of the VRAM
///     ladder. Read by the runner's per-side KV-format selection (and every KV-footprint estimate)
///     ONLY when the user set none of INFR_KV_TYPE_K / INFR_KV_TYPE_V / INFR_KV_Q8 — an explicit
///     setting always wins, both directions. Policy: BOTH sides go q8_0 (coupled Q8 keeps
///     record-once decode replay and satisfies K>=V symmetrically); never below q8_0.
///
/// Beside the two pins, the session's DEFAULT prefill chunk (`default_ubatch`): what
/// [`ubatch_rows`] answers when nothing above pinned one, fixed when the pins are made
/// ([`PlacementPins::for_caps`]) from the class of the backend the session runs on. It used to be
/// a process-global published by the first Vulkan device the process opened, so a second backend
/// of another class (a CPU run beside an iGPU, a test binary that opened the iGPU first) took the
/// first one's chunk.
///
/// These were process-global `OnceLock`s, which LEAKED across models in a multi-model process
/// (`infr multi` hosts N models, each its own `DenseVulkanSession`, and the server runs their
/// generations CONCURRENTLY): a second model's `.set()` was a silent no-op, so it inherited model
/// A's pinned chunk height / q8 decision — which may not fit B's VRAM. They now live PER SESSION
/// (owned by `DenseVulkanSession`, entered via [`PlacementScope`] around each placement + generate),
/// so each model's ladder decision is isolated.
/// The chunk height has ONE exception to "set once, then stable": the runner's post-load
/// re-clamp ([`reclamp_ctx_to_live_room`]) may LOWER it, because the sweep that pinned it decided
/// against an estimate of the weight bytes and the re-clamp knows the real ones. It still runs
/// before the first KV allocation, so everything that reads the height — the prefill loop, the
/// activation reserve, the SWA ring sizing — still sees one value for the session's whole life.
/// `0` = unpinned (see [`repin_ubatch`]).
///
/// No `Default` outside tests: every set of pins names the backend it is for, so a Vulkan entry
/// point cannot silently take the discrete chunk on an integrated GPU.
pub(crate) struct PlacementPins {
    ubatch: std::sync::atomic::AtomicUsize,
    /// The model's prefill chunk cap ([`moe_small_m_chunk_cap`]), `0` for none: [`ubatch_rows`]
    /// answers at most this, whoever set the height, so the prefill loop, the activation reserve,
    /// the SWA ring and every fit estimate see the capped chunk. Set when the scope is made for a
    /// model ([`PlacementPins::for_model`], [`enter_model_scope`]) and only ever lowered.
    moe_chunk_cap: std::sync::atomic::AtomicUsize,
    /// The prefill chunk when nothing pinned one ([`default_ubatch_rows`] of the backend these
    /// pins were made for). Never changes after construction.
    default_ubatch: usize,
    kv_q8: std::sync::OnceLock<()>,
    /// Has this session already reported an activation peak above what it reserved (the runner's
    /// `activation reserve too low` warning)? The condition persists for the session's whole life —
    /// the peak is a high-water mark and the reserve is fixed — so without a latch a server would
    /// repeat the same line on every request for as long as it runs.
    act_over_reserve_reported: std::sync::atomic::AtomicBool,
    /// Host memory the session's weight-streaming host tier was sized to take (its arenas'
    /// bytes, `0` with none), set by placement each time it runs for a cold init. The tier's
    /// arenas are zero-filled on first touch, so the live host probe does not see them yet when
    /// the post-load re-clamp prices a spilled cache's system-RAM part against what the host has
    /// free ([`refuse_unhostable_kv`]); this is what it counts for them.
    ///
    /// Counted once there, not twice: a session cold-inits only its first slot (later ones are
    /// forks, `SlotPool::pick` and `ParallelSeam::init_slots`), and a cold init that failed and
    /// is retried runs placement again, which replaces the tier (`init_dense_pager`, dropping the
    /// old arenas) and re-pins this to the new one before any weight is read through it — the
    /// weights a streamed session uploads are resident ones, so nothing touches the tier between
    /// placement and the re-clamp. The fork check after a first generation
    /// ([`refuse_unhostable_forks`]) is different: by then the generation has touched part of the
    /// arenas, which the probe sees and this still counts, so it over-counts the tier there — the
    /// conservative side (traced in code 2026-11-25, not run).
    host_tier_bytes: std::sync::atomic::AtomicU64,
}

impl PlacementPins {
    /// Fresh pins (nothing pinned) for a session on a backend reporting `caps`: its default prefill
    /// chunk is that backend class's ([`default_ubatch_rows`]).
    pub(crate) fn for_caps(caps: &infr_core::backend::Capabilities) -> Self {
        Self {
            ubatch: Default::default(),
            moe_chunk_cap: Default::default(),
            default_ubatch: default_ubatch_rows(caps),
            kv_q8: Default::default(),
            act_over_reserve_reported: Default::default(),
            host_tier_bytes: Default::default(),
        }
    }

    /// [`for_caps`](Self::for_caps) for a Vulkan session of the model `cfg`: its prefill chunk is
    /// capped as that model needs ([`moe_small_m_chunk_cap`]).
    pub(crate) fn for_model(caps: &infr_core::backend::Capabilities, cfg: &Config) -> Self {
        let pins = Self::for_caps(caps);
        pin_moe_chunk_cap(&pins, cfg);
        pins
    }
}

/// Tests' pins: those of a backend reporting default capabilities, which is the class the CPU
/// backend reports (discrete, so the 1024-row chunk). A test that runs a GPU backend makes its
/// pins with [`PlacementPins::for_caps`] or [`enter_backend_scope`] instead; the runner's check
/// ([`debug_assert_pins_for`]) fails one that does not on an integrated GPU.
#[cfg(test)]
impl Default for PlacementPins {
    fn default() -> Self {
        Self::for_caps(&infr_core::backend::Capabilities::default())
    }
}

thread_local! {
    /// The session whose placement pins the current thread's `ubatch_rows`/`kv_auto_q8` reads and
    /// writes resolve against — set by [`PlacementScope`] for the duration of a placement + runner
    /// call. `None` outside any scope (the one-shot `generate_dense_*`, CPU/Metal, tests), which
    /// falls back to [`FALLBACK_PINS`].
    static CURRENT_PINS: std::cell::RefCell<Option<std::sync::Arc<PlacementPins>>> =
        const { std::cell::RefCell::new(None) };
}

/// Process-wide fallback pins for callers with no active [`PlacementScope`] (the CPU and Metal
/// paths, tests). Made for default capabilities, the class the CPU and Metal backends report, so
/// their default chunk is what it always was. A Vulkan backend does not run on these: each Vulkan
/// entry point enters a scope made for its backend ([`enter_backend_scope`], the sessions' own
/// pins), which the placement and runner checks ([`debug_assert_scope_for`],
/// [`debug_assert_pins_for`]) hold it to.
static FALLBACK_PINS: std::sync::OnceLock<std::sync::Arc<PlacementPins>> =
    std::sync::OnceLock::new();

/// Run `f` against the current thread's session pins (or the process fallback when no session scope
/// is active). The single reader/writer funnel for both placement pins.
fn with_placement_pins<R>(f: impl FnOnce(&PlacementPins) -> R) -> R {
    CURRENT_PINS.with(|c| match c.borrow().as_ref() {
        Some(p) => f(p),
        None => f(FALLBACK_PINS.get_or_init(|| {
            std::sync::Arc::new(PlacementPins::for_caps(
                &infr_core::backend::Capabilities::default(),
            ))
        })),
    })
}

/// Whether this thread is inside a [`PlacementScope`].
fn in_placement_scope() -> bool {
    CURRENT_PINS.with(|c| c.borrow().is_some())
}

/// Debug check that the pins the current thread reads its default prefill chunk from were made for
/// a backend of `be`'s class: the scope's, or the fallback's when `be` is of the CPU's class. A
/// Vulkan entry point that forgot its scope runs an integrated GPU at the discrete chunk, which can
/// trip the GPU watchdog; this turns that into a failure in debug builds. Asks `be` for its
/// capabilities only in debug builds, so a release build pays nothing.
pub(crate) fn debug_assert_pins_for(be: &dyn Backend) {
    if cfg!(debug_assertions) {
        let (have, want) = (
            with_placement_pins(|p| p.default_ubatch),
            default_ubatch_rows(&be.capabilities()),
        );
        assert_eq!(
            have,
            want,
            "the placement pins in force were made for another backend class than `{}`'s: enter \
             a scope made for it (`seam::enter_backend_scope`, or a session's own pins)",
            be.name()
        );
    }
}

/// [`debug_assert_pins_for`], and that a [`PlacementScope`] is active at all: the check for the
/// Vulkan-only paths (placement, a session's generation), which never run on the fallback pins.
pub(crate) fn debug_assert_scope_for(be: &dyn Backend) {
    debug_assert!(
        in_placement_scope(),
        "a `{}` placement or session call ran outside any placement scope",
        be.name()
    );
    debug_assert_pins_for(be);
}

/// The placement scope a one-shot entry point on `be` runs in: the caller's when one is active (a
/// session's, or a test that pinned a chunk), checked to be for `be`'s class, else a fresh one
/// made for `be` ([`PlacementPins::for_caps`]), held for the whole call so every runner step and
/// the placement before it read the same pins.
pub(crate) fn enter_backend_scope(be: &dyn Backend) -> Option<PlacementScope> {
    if in_placement_scope() {
        debug_assert_pins_for(be);
        None
    } else {
        Some(PlacementScope::enter(std::sync::Arc::new(
            PlacementPins::for_caps(&be.capabilities()),
        )))
    }
}

/// [`enter_backend_scope`] for a Vulkan run of the model `cfg`: the scope in force (the caller's,
/// or a fresh one for `be`) gets that model's prefill chunk cap ([`moe_small_m_chunk_cap`]) before
/// anything reads a chunk height. A scope serves one model; the cap only ever lowers it.
pub(crate) fn enter_model_scope(be: &dyn Backend, cfg: &Config) -> Option<PlacementScope> {
    let scope = enter_backend_scope(be);
    pin_model_chunk_cap(cfg);
    scope
}

/// RAII guard binding `pins` as the current thread's placement scope. Every seam entry that runs a
/// placement decision or a runner step for a session (`DenseVulkanSession::generate*`, the
/// default-ctx clamp) enters one around its work so the free-function readers
/// ([`ubatch_rows`]/[`kv_auto_q8`]) resolve against THAT session's pins. Restores the previous
/// scope on drop (scopes may nest — the clamp runs inside session construction).
pub(crate) struct PlacementScope {
    prev: Option<std::sync::Arc<PlacementPins>>,
}

impl PlacementScope {
    pub(crate) fn enter(pins: std::sync::Arc<PlacementPins>) -> Self {
        let prev = CURRENT_PINS.with(|c| c.borrow_mut().replace(pins));
        Self { prev }
    }
}

impl Drop for PlacementScope {
    fn drop(&mut self) {
        CURRENT_PINS.with(|c| *c.borrow_mut() = self.prev.take());
    }
}

/// Pin the placement prefill chunk (rows) for the current session scope, if nothing pinned one
/// yet — the placement sweeps' spelling, which keeps the first decision (a racing second sweep
/// must not move a height an earlier one already priced buffers against).
fn pin_ubatch(rows: usize) {
    with_placement_pins(|p| {
        let _ = p.ubatch.compare_exchange(
            0,
            rows,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        );
    });
}

/// LOWER the pinned prefill chunk, overriding whatever the placement sweeps pinned — the runner's
/// post-load re-clamp only. A shorter chunk shrinks both the activation reserve and the SWA ring,
/// so it buys context; the sweeps chose their height against an ESTIMATE of the weight bytes,
/// and by here the real ones are known. Refuses to RAISE a height (that could outgrow buffers a
/// pre-load decision already sized, and on an integrated GPU it is the watchdog bound).
fn repin_ubatch_lower(rows: usize) {
    with_placement_pins(|p| {
        let cur = p.ubatch.load(std::sync::atomic::Ordering::Relaxed);
        if cur == 0 || rows < cur {
            p.ubatch.store(rows, std::sync::atomic::Ordering::Relaxed);
        }
    });
}

/// Claim the one-shot "activation reserve too low" report for the current session scope: `true`
/// exactly once, for the first caller that finds the peak above the reserve.
pub(crate) fn claim_act_over_reserve_report() -> bool {
    with_placement_pins(|p| {
        !p.act_over_reserve_reported
            .swap(true, std::sync::atomic::Ordering::Relaxed)
    })
}

/// Record the host memory the current session's weight-streaming host tier takes (see
/// [`PlacementPins`]'s `host_tier_bytes`).
fn pin_host_tier_bytes(bytes: u64) {
    with_placement_pins(|p| {
        p.host_tier_bytes
            .store(bytes, std::sync::atomic::Ordering::Relaxed)
    });
}

/// The host memory the current session's weight-streaming host tier takes, as placement recorded
/// it (`0` before any placement, or with no tier).
pub(crate) fn host_tier_bytes() -> u64 {
    with_placement_pins(|p| p.host_tier_bytes.load(std::sync::atomic::Ordering::Relaxed))
}

/// Whether the placement ladder pinned auto-q8 KV for the current session scope (see
/// [`PlacementPins`]).
pub(crate) fn kv_auto_q8() -> bool {
    with_placement_pins(|p| p.kv_q8.get().is_some())
}

/// Pin auto-q8 KV for the current session scope (the default-ctx clamp path in `model.rs`, and the
/// binder's own dense rung). Idempotent (OnceLock).
pub(crate) fn pin_kv_auto_q8() {
    with_placement_pins(|p| {
        let _ = p.kv_q8.set(());
    });
}

/// True when the user expressed NO explicit KV-format choice — the only state auto-q8 may fill.
///
/// Use the `*_specified` flags, NOT `type_k.is_some()`. An unrecognized format name
/// (`INFR_KV_TYPE_K=nonsense`) parses to no dtype, so the runner falls through to f16 — but it was
/// still SUPPLIED, and today's `is_err()` reads it as "the user chose", suppressing auto-q8. Both
/// halves of that asymmetry are preserved. A model whose V follows K ([`kv_v_follows_k`]) has no V
/// setting to supply.
pub(crate) fn kv_unset(cfg: &Config, ec: &EngineConfig) -> bool {
    !ec.kv.type_k_specified && (kv_v_follows_k(cfg) || !ec.kv.type_v_specified) && !ec.kv.force_q8
}

/// Does this model's V cache format follow its K format, whatever `INFR_KV_TYPE_V` says?
///
/// True for deepseek4: it has no V cache at all (its attention reads the K cache as V, see
/// [`kv_row_elems`]), so a V format would describe nothing. Every place a V format is chosen or
/// gates something — the runner's format pick, the SWA ring gate, Metal's mixed-format clamp, the
/// fit estimates and auto-q8 — reads it through [`kv_v_fmt`] or this predicate, so the setting is
/// ignored the same way everywhere rather than half-honoured.
pub(crate) fn kv_v_follows_k(cfg: &Config) -> bool {
    cfg.deepseek4
}

/// The format the V side takes for `cfg`: `k_fmt` when [`kv_v_follows_k`], else `v_fmt`.
pub(crate) fn kv_v_fmt(cfg: &Config, k_fmt: DType, v_fmt: DType) -> DType {
    if kv_v_follows_k(cfg) {
        k_fmt
    } else {
        v_fmt
    }
}

/// Per-token (K, V) ELEMENT counts for layer `l` — the one answer to "how wide is a row of this
/// layer's KV cache", shared by the runner's allocation, the graph declaration, the fork/seed
/// copies and every footprint estimate. Multiply by a row count for a whole cache side.
///
/// DeepSeek2 MLA caches ONE compressed row per token (`kv_lora_rank + qk_rope_dim` — 576 on
/// V2-Lite) and has **no V cache at all**: V is an aliased prefix view of that same row, so the V
/// side is 0 elements and its buffer/tensor is a placeholder ([`kv_side_elems`]). Every other arch
/// stores `n_kv * head_dim` per side. Reading it off `layer_n_kv * layer_head_dim` for an MLA model
/// yields `1 * 192` — a third of what the kernels index — which is exactly what `SeamKv::fork`,
/// `SeamKv::seed_from` and the VRAM estimate each did while the allocation carried its own copy of
/// the branch (docs/backlog.md B41).
///
/// **deepseek32 (V3.2) puts its lightning indexer's SECOND cache on that free V side**: one
/// `indexer_head_size`-wide row per token per layer, on top of the compressed MLA row. It is a
/// genuinely independent per-token cache written by its own `Op::WriteKv` and read by
/// `Op::LightningIndexer`, and it is carried here — rather than as a third per-layer buffer — so
/// that the SIX sites this helper feeds (allocation, graph declaration, `SeamKv::fork`,
/// `SeamKv::seed_from`, and both VRAM estimates) size and copy it without any of them growing a
/// private branch. Under-reserving it is exactly the failure B41 records. The precedent is
/// `MixerW::DeltaNet`, which puts a qwen35 layer's recurrent `S` state in the same `v_cache[l]`
/// slot; the difference is that DeltaNet's state is fixed-size and this one is per-token, which is
/// what makes it fit the K side's own geometry.
///
/// The indexer cache **must never ring**: `Op::LightningIndexer` masks causally only, so position 0
/// stays eligible for every query row and every backend refuses a cache holding fewer rows than
/// `kv_len`. That is why it is sized off [`kv_rows_at`] like any full-context side — V3.2 has no
/// sliding-window layer, so `kv_rows_at` returns the whole context — and why the allocation asserts it
/// (see `generate_dense_backend`'s KV loop).
///
/// Recurrent-state layers (qwen35 DeltaNet, LFM short convolution) cache nothing per token, so
/// both counts are 0: their two buffers hold a fixed-size state instead
/// ([`Config::recurrent_state_elems`]), which the allocation and the graph declaration size
/// before they would read a row width.
///
/// **deepseek4 (V4) caches ONE `head_dim`-wide MQA row per token and has no V cache**: its raw
/// attention is `build_attn_mha(q, k_all, k_all, …)`, so K and V are the same rows. The emit writes
/// the row once and points `Op::Attention`'s `k_cache` and `v_cache` at the same tensor, which every
/// backend accepts (the CPU arm reads an aliased pair through one guard and dequantizes it once).
/// The V side has no rows: the graph declares no V tensor for a V4 layer (its `v_cache` handle IS
/// the K cache's) and the session's V buffer is an unbound placeholder. Its format follows K
/// ([`kv_v_follows_k`]).
///
/// The compressed caches (CSA/HCA/LID) and the compressor state rings a ratio-4 / ratio-128 layer
/// owns are NOT modelled here: they are per-layer (a ratio-0 layer has none), one row per BLOCK
/// rather than per token, and the rings are fixed-size. [`dsv4_comp_bufs`] owns their geometry.
pub(crate) fn kv_row_elems(cfg: &Config, l: usize) -> (usize, usize) {
    if cfg.deepseek2 {
        return (cfg.kv_lora_rank + cfg.qk_rope_dim, cfg.indexer_head_size);
    }
    if cfg.is_recurrent_layer(l) {
        return (0, 0);
    }
    if cfg.deepseek4 {
        // Read straight off `head_dim` rather than through `layer_head_dim`/`layer_n_kv`: those
        // route via `is_swa_layer`, which is TRUE for every V4 layer, and so would answer with
        // gemma4's `head_dim_swa`/`n_kv_swa` fields. They happen to equal `head_dim`/`n_kv` for
        // every non-gemma4 model, which is an accident this arch should not depend on.
        return (cfg.head_dim, 0);
    }
    let row = cfg.layer_n_kv(l) * cfg.layer_head_dim(l);
    (row, row)
}

/// One persistent buffer a DeepSeek V4 compressed layer owns beside its raw cache: a compressed
/// cache (`kv_hca`, `kv_csa`, `kv_lid` in `llama_kv_cache_dsv4`) or one half of a compressor's
/// state ring (`llama_dsv4_comp_state`'s kv or score tensor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dsv4CompBuf {
    pub(crate) dtype: DType,
    pub(crate) elems: usize,
    /// A state ring (f32, fixed size, reset with the session); `false` for a compressed cache.
    pub(crate) ring: bool,
}

impl Dsv4CompBuf {
    /// Bytes the session allocates for it.
    pub(crate) fn bytes(&self) -> usize {
        kv_side_bytes(self.dtype, self.elems)
    }
}

/// Buffers per compressor tier in [`dsv4_comp_bufs`]: the compressed cache, then the ring's kv
/// half, then its score half.
pub(crate) const DSV4_TIER_BUFS: usize = 3;

/// Layer `l`'s compressed-tier buffers for a `want_ctx`-token session whose raw cache is `k_fmt`,
/// in the order the graph declares and the session allocates them; empty for a ratio-0 layer and
/// for every other arch. The ONE answer to their geometry, shared by the allocation, the graph
/// declaration, `SeamKv::fork`, the reset and every footprint estimate.
///
/// Each compressor tier contributes [`DSV4_TIER_BUFS`] buffers:
/// - its compressed cache, one `width`-wide row per block of `ratio` positions,
///   `ceil(want_ctx / ratio)` rows (`dsv4_comp_size`), in `k_fmt`: the CPU attention's compressed
///   segment shares the raw cache's dtype (`check_attention`), as `llama_kv_cache_dsv4` builds
///   every tier with the same `type_k`. infr skips the reference's 256-row padding, which only
///   widens a masked view;
/// - the ring's kv and score halves, f32 `[state_rows, state_width]` each: `ratio` rows of `width`
///   for the non-overlapping HCA compressor, `2*ratio` rows of `2*width` for the overlapping
///   CSA and LID ones (`llama_kv_cache_dsv4`'s `csa_state` / `hca_state` / `lid_state`).
///
/// An HCA layer (ratio 128) has one tier at `head_dim`. A CSA layer (ratio 4) has two: its own at
/// `head_dim`, then the lightning indexer's at `indexer_head_size`.
pub(crate) fn dsv4_comp_bufs(
    cfg: &Config,
    l: usize,
    want_ctx: usize,
    k_fmt: DType,
) -> Vec<Dsv4CompBuf> {
    if !cfg.deepseek4 {
        return Vec::new();
    }
    let ratio = cfg.layer_compress_ratio(l);
    let tier = |ratio: usize, width: usize, overlap: bool| {
        let (state_rows, state_width) = if overlap {
            (2 * ratio, 2 * width)
        } else {
            (ratio, width)
        };
        let ring = Dsv4CompBuf {
            dtype: DType::F32,
            elems: state_rows * state_width,
            ring: true,
        };
        [
            Dsv4CompBuf {
                dtype: k_fmt,
                elems: want_ctx.div_ceil(ratio).max(1) * width,
                ring: false,
            },
            ring,
            ring,
        ]
    };
    match ratio as u32 {
        0 => Vec::new(),
        dsv4_plan::DSV4_HCA_RATIO => tier(ratio, cfg.head_dim, false).to_vec(),
        dsv4_plan::DSV4_CSA_RATIO => [
            tier(ratio, cfg.head_dim, true),
            tier(ratio, cfg.indexer_head_size, true),
        ]
        .concat(),
        _ => unreachable!("Config::from_gguf accepts compress ratios 0, 4 and 128 only"),
    }
}

/// Elements a KV side that the arch does NOT cache still allocates: MLA's V and deepseek4's V,
/// whose [`kv_row_elems`] count is 0. No backend ever indexes it (the MLA kernels read V as a
/// prefix of the K row; deepseek4's attention reads its K cache as V). MLA still declares and binds
/// a graph Input for the side; deepseek4 declares none (its layer's `v_cache` handle is the K
/// cache's) and leaves the buffer unbound. A zero-size allocation is not portable, so the
/// placeholder is a handful of elements, sized identically at the allocation and any declaration.
pub(crate) const KV_PLACEHOLDER_ELEMS: usize = 4;

/// A KV side's element count as DECLARED in the graph: `elems`, or the placeholder when the arch
/// does not cache that side at all (see [`KV_PLACEHOLDER_ELEMS`]).
pub(crate) fn kv_side_elems(elems: usize) -> usize {
    if elems == 0 {
        KV_PLACEHOLDER_ELEMS
    } else {
        elems
    }
}

/// Smallest KV-side allocation any backend will accept.
///
/// The floor has to be in BYTES rather than elements, because a block-quant format prices
/// [`KV_PLACEHOLDER_ELEMS`] at zero bytes (`kv_fmt_bytes(Q8_0, 4)` is `4 / 32 * 34`) and a
/// zero-size allocation is refused outright by the Vulkan allocator.
const KV_MIN_SIDE_BYTES: usize = 4;

/// Bytes to ALLOCATE for one KV side holding [`kv_side_elems`]`(elems)` values of `fmt`, floored at
/// [`KV_MIN_SIDE_BYTES`]. A side the arch really caches is far past that floor at any usable
/// context; only the placeholder ever meets it.
pub(crate) fn kv_side_bytes(fmt: DType, elems: usize) -> usize {
    kv_fmt_bytes(fmt, kv_side_elems(elems)).max(KV_MIN_SIDE_BYTES)
}

/// Layout half of the Q8_0 / block-quant KV gate — every layer's KV row must be whole 32-elem
/// blocks. Pure geometry: it says the format *fits* the rows, not that this model may use it (see
/// [`kv_q8_layout_ok`], which adds the MLA exclusion).
pub(crate) fn kv_row_align_ok(cfg: &Config) -> bool {
    (0..cfg.n_layer).all(|l| {
        let (k, v) = kv_row_elems(cfg, l);
        // A V4 compressed cache takes the raw cache's format, so its rows are held to it too; at
        // a one-token context each compressed cache is exactly one row.
        let comp_ok = dsv4_comp_bufs(cfg, l, 1, DType::F32)
            .iter()
            .filter(|b| !b.ring)
            .all(|b| b.elems.is_multiple_of(32));
        k.is_multiple_of(32) && v.is_multiple_of(32) && comp_ok
    })
}

/// Layout half of the TurboQuant KV gate: every layer's cache row is whole 128-element groups
/// (a WHT group is a 128-element slice of the row). A V4 compressed cache takes the raw cache's
/// format, so its rows are held to it too, the LID cache's indexer-wide row among them.
pub(crate) fn kv_turbo_align_ok(cfg: &Config) -> bool {
    (0..cfg.n_layer).all(|l| {
        let comp_ok = dsv4_comp_bufs(cfg, l, 1, DType::F32)
            .iter()
            .filter(|b| !b.ring)
            .all(|b| b.elems.is_multiple_of(128));
        cfg.layer_head_dim(l).is_multiple_of(128) && comp_ok
    })
}

/// Whether a Q8_0 KV cache may be CHOSEN for this model: the rows are 32-block aligned
/// ([`kv_row_align_ok`]) **and** the model is not MLA.
///
/// The MLA exclusion is not about alignment (576 is 18 whole blocks) — it is that the Vulkan and
/// Metal MLA kernels read the cache as f16 unconditionally, so `generate_dense_backend` forces f16
/// there ([`mla_kv_fmt`]). This gate is what the Vulkan auto-q8 placement PIN consults before
/// pricing the VRAM estimate at q8 (`model.rs`'s `clamp_default_ctx`), so excluding MLA here is
/// what keeps the estimate and the allocation in agreement: a format the runner will refuse to
/// build can never be pinned and priced in the first place.
pub(crate) fn kv_q8_layout_ok(cfg: &Config) -> bool {
    !cfg.deepseek2 && kv_row_align_ok(cfg)
}

/// The KV formats an MLA (deepseek2) session may actually run with on `backend`, given the pair
/// the dtype ladder resolved.
///
/// The GPU MLA kernels type the cache f16 UNCONDITIONALLY — `mla.comp`'s `kread` is
/// `unpackHalf2x16`, Metal's `mla_f16kv_one` takes `device const half*` — so any other dtype is
/// REINTERPRETED rather than converted: silent wrong output, no error. (The CPU `Op::Mla` arm
/// dequantizes through its own closure and is dtype-correct, which is why a CPU-vs-GPU parity run
/// would show a correct oracle beside a wrong GPU and read as a kernel bug — docs/backlog.md B42.)
///
/// So on every backend but `cpu`, a deepseek2 session is f16 on both sides. A format the USER named
/// is REFUSED rather than downgraded behind their back; a format nothing asked for (the ladder's
/// own default, or a placement pin — which [`kv_q8_layout_ok`] already prevents for MLA) is forced.
pub(crate) fn mla_kv_fmt(
    cfg: &Config,
    backend: &str,
    ec: &EngineConfig,
    k_fmt: DType,
    v_fmt: DType,
) -> AResult<(DType, DType)> {
    if !cfg.deepseek2 || backend == "cpu" {
        return Ok((k_fmt, v_fmt));
    }
    let named = |dt: Option<DType>| matches!(dt, Some(dt) if dt != DType::F16);
    if named(ec.kv.type_k) || named(ec.kv.type_v) || ec.kv.force_q8 {
        return Err(anyhow!(
            "deepseek2 (MLA) KV cache is f16-only on the {backend} backend: its attention kernel \
             reads the compressed KV row as f16 and would reinterpret any other dtype. Requested \
             k={:?} (INFR_KV_TYPE_K), v={:?} (INFR_KV_TYPE_V), force_q8={} (INFR_KV_Q8) — unset \
             them, ask for f16, or run this model on the CPU backend, which dequantizes every KV \
             dtype.",
            ec.kv.type_k,
            ec.kv.type_v,
            ec.kv.force_q8
        ));
    }
    Ok((DType::F16, DType::F16))
}

/// Row capacity of layer `l`'s K/V cache at context `want_ctx`, for prefill chunk height
/// `ubatch`. With `ring` (SWA ring sizing on for this session — see [`kv_ring_wanted`] + the
/// backend's `Capabilities::kv_swa_ring`), a sliding-window layer allocates only
/// `min(want_ctx, round64(window + ubatch))` rows and the backends write/read position `p` at row
/// `p % rows` (WriteKv/Attention ring semantics).
///
/// Correctness bound: during one forward of `B <= ubatch` rows starting at position `p0`, the
/// oldest position any query's window reaches is `p0 + 1 - window` and the newest written is
/// `p0 + B - 1` — at most `window + B - 1 <= window + ubatch` distinct live positions, so a ring
/// of `window + ubatch` rows never recycles a row the sliding-window mask hasn't ALREADY excluded
/// (that mask discards everything older than `pos - window`); attention output is therefore
/// identical to the full-context cache. Global (non-SWA) layers keep full `want_ctx` rows.
///
/// A live slot passes the height its rings were sized with (`SeamKv::ubatch`); placement passes
/// [`ubatch_rows`] or the candidate height it is pricing.
///
/// `row_align` is the backend's `Backend::kv_row_align`: the capacity is rounded up to it, so a
/// backend whose attention reads whole KV blocks finds every row it reads allocated. The extra
/// rows are capacity only (the session's context stays `want_ctx`).
pub(crate) fn kv_rows_at(
    cfg: &Config,
    l: usize,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    row_align: usize,
) -> usize {
    let rows = if ring && cfg.is_swa_layer(l) {
        want_ctx.min((cfg.swa_window + ubatch).next_multiple_of(64))
    } else {
        want_ctx
    };
    rows.next_multiple_of(row_align)
}

/// The rows the shortest sliding-window ring holds in a slot of `max_ctx` tokens whose prefill
/// chunk is `ubatch` rows, on a backend whose cache rows align to `row_align`
/// (`Backend::kv_row_align`), or `None` for a model with no window layer. A slot that has written
/// past that many positions has wrapped the ring and recycled its earliest rows.
pub fn window_ring_rows(
    cfg: &Config,
    max_ctx: usize,
    ubatch: usize,
    row_align: usize,
) -> Option<usize> {
    (0..cfg.n_layer)
        .filter(|&l| cfg.is_swa_layer(l))
        .map(|l| kv_rows_at(cfg, l, max_ctx, true, ubatch, row_align))
        .min()
}

/// How the backend a session runs on sizes its KV cache: every layer's row capacity rounded up to
/// `rows` (`Backend::kv_row_align`, see [`kv_rows_at`]), and what the device commits for each
/// buffer, `commit` (`Backend::kv_alloc_cost`). The footprint estimates take both; the allocation
/// itself asks for the padded rows and leaves the rest to the allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KvAlign {
    pub(crate) rows: usize,
    pub(crate) commit: KvAllocCost,
}

impl KvAlign {
    /// No rounding at all: the CPU backend's, and what a sizing decision without a backend prices.
    pub(crate) const NONE: KvAlign = KvAlign {
        rows: 1,
        commit: KvAllocCost::IDENTITY,
    };

    /// How `be` sizes a KV cache.
    pub(crate) fn of(be: &(impl Backend + ?Sized)) -> KvAlign {
        KvAlign {
            rows: be.kv_row_align(),
            commit: be.kv_alloc_cost(),
        }
    }
}

/// [`kv_bytes_estimate_fmt`] with one format for both sides: `q8` prices them Q8_0, else f16. The
/// pricing helper the MoE expert-placement budget and the CPU arena budget read, so both reserve
/// the cache the runner will allocate, a pinned auto-q8 cache included.
pub(crate) fn kv_bytes_estimate(
    cfg: &Config,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    align: KvAlign,
    q8: bool,
) -> u64 {
    let side = if q8 { DType::Q8_0 } else { DType::F16 };
    kv_bytes_estimate_fmt(cfg, want_ctx, ring, ubatch, align, side, side)
}

/// Bytes of device memory one session slot's KV cache takes: every buffer of [`kv_slot_allocs`],
/// each priced at what the device commits for it (`align.commit`), so the figure every fit and
/// placement decision compares against a budget is the allocation itself and not a model of it.
/// `align` is the [`KvAlign`] of the backend the session will run on.
///
/// The recurrent-state snapshot is in it although a slot allocates that on first use: a context
/// sized to fill the device has to leave room for it. That is where this differs from the refuse
/// rung's floor ([`kv_floor_alloc_bytes`]), which lists only what the runner's init allocates.
///
/// On Vulkan every KV buffer has its own allocation, and `align.commit` is the driver's rounding
/// and per-allocation cost as the backend measured them off its own heap usage. Measured on the
/// integrated GPU only, where they match each buffer exactly; the device's free memory still drops
/// by a few KB a buffer more than that, unexplained and unpriced (docs/backlog.md B13), and a
/// driver the probe cannot measure is priced by a guess.
pub(crate) fn kv_bytes_estimate_fmt(
    cfg: &Config,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    align: KvAlign,
    k_fmt: DType,
    v_fmt: DType,
) -> u64 {
    let committed = |bytes: usize| align.commit.committed(bytes as u64);
    kv_slot_allocs(cfg, want_ctx, ring, ubatch, align.rows, k_fmt, v_fmt)
        .map(|(a, b)| committed(a) + committed(b))
        .sum()
}

/// The `BufferUsage::KvCache` allocations the runner's cold init makes for one slot, in bytes, in
/// the `(bytes, bytes)` form the KV guard reads ([`refuse_unallocatable_kv`]), in the order the
/// cold init makes them:
///
/// - first `cfg.n_layer` entries, layer by layer: its two cache buffers
///   ([`kv_layer_alloc_bytes`]), which `SeamKv::fork` also allocates through;
/// - then a DeepSeek V4 compressed layer's block caches and rings ([`dsv4_comp_allocs`]), layer by
///   layer.
///
/// The cold init allocates its K/V pairs from the leading `cfg.n_layer` entries of this list.
pub(crate) fn kv_init_allocs(
    cfg: &Config,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    row_align: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> impl Iterator<Item = (usize, usize)> + '_ {
    (0..cfg.n_layer)
        .map(move |l| kv_layer_alloc_bytes(cfg, l, want_ctx, ring, ubatch, row_align, k_fmt, v_fmt))
        .chain((0..cfg.n_layer).flat_map(move |l| dsv4_comp_allocs(cfg, l, want_ctx, k_fmt)))
}

/// Every `BufferUsage::KvCache` allocation one session slot can make, in the form
/// [`kv_init_allocs`] gives: those, then the snapshot of each recurrent layer's state
/// ([`kv_recurrent_alloc_bytes`]), which `snapshot_delta` allocates on first use.
pub(crate) fn kv_slot_allocs(
    cfg: &Config,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    row_align: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> impl Iterator<Item = (usize, usize)> + '_ {
    kv_init_allocs(cfg, want_ctx, ring, ubatch, row_align, k_fmt, v_fmt)
        .chain((0..cfg.n_layer).filter_map(|l| kv_recurrent_alloc_bytes(cfg, l)))
}

/// Config/env-level gate for SWA ring KV sizing, shared by the runner's allocation and the
/// KV-footprint ESTIMATES (ctx clamp, dense/MoE placement) so they price the same allocation the
/// runner will make. The runner additionally requires the backend capability
/// (`Capabilities::kv_swa_ring`) and the FINAL per-side KV formats; this checks the env-requested
/// formats (a format the runner gates back to f16 stays ring-capable, so the estimate is only
/// ever conservative). Gated OFF for:
///   - non-SWA models (no window — nothing to ring);
///   - DiffusionGemma (its canvas denoise attends a fixed bidirectional `[lo, kv_len)` range that
///     is NOT a per-query sliding window, so the ring's mask-already-excludes-it argument doesn't
///     hold there);
///   - non-f16/q8 KV formats (the low-bit block quants / bf16 / f32 / turbo read the cache
///     through a dequant-the-prefix prepass sized in positions, and their static-only writes
///     never learned the ring split — they keep full-context caches, documented scope gate);
///   - `kv.ring = false` (`INFR_NO_KV_RING=1`, A/B and escape hatch).
pub(crate) fn kv_ring_wanted(cfg: &Config, ec: &EngineConfig) -> bool {
    // Not-supplied = the f16 default = ring-capable; otherwise the requested format must PARSE to
    // f16 or q8 (a name the runner would not recognize either is not ring-capable — the
    // `specified && dtype.is_none()` case). The dtype comes from the ONE shared
    // spelling table (`budget::parse_kv_dtype`, now applied in the config's env layer), so adding
    // an alias cannot make this gate and the runner disagree.
    let fmt_ok = |specified: bool, dt: Option<DType>| {
        !specified || matches!(dt, Some(DType::F16 | DType::Q8_0))
    };
    cfg.swa_window > 0
        && !cfg.diffusion_gemma
        && ec.kv.ring
        && fmt_ok(ec.kv.type_k_specified, ec.kv.type_k)
        && (kv_v_follows_k(cfg) || fmt_ok(ec.kv.type_v_specified, ec.kv.type_v))
}

/// The smallest context a session is worth opening with. A window below this is useless for any
/// real prompt, so it is the line between "clamp the default context" and "this model cannot be
/// served on this device at all" (`SeamModel::clamp_default_ctx`'s refuse rung), and the floor
/// every derived per-slot / fractional window is held to.
pub(crate) const MIN_SESSION_CTX: usize = 1024;

// ── one budget, two families of callers ───────────────────────────────────────────────────────
//
// The context-fit math ([`kv_fit_ctx_for`], via `SeamModel::kv_fit_ctx_fmt`) and the placement
// sweeps (`vulkan_moe_binder`'s residency / auto-q8 / streaming / MoE-expert budgets) are two
// readers of ONE question: what still fits this device? Every helper below takes the raw
// [`infr_vulkan::VramInfo`] snapshot and derives its ceiling from [`VramInfo::alloc_room`] —
// the allocator's own limit — so neither family can plan bytes `check_vram_budget` will refuse.
// `budgets_agree_with_the_allocator_ceiling` and `fit_math_and_placement_pick_the_same_rung`
// guard the drift.

/// Will the KV cache a placement/fit decision is pricing actually RING (SWA rows capped at
/// `window + chunk`)? The config/env gate AND f16/q8 on BOTH sides — the same pair of conditions
/// the runner applies. A low-bit side keeps full-context caches, so pricing it as a ring would
/// hand out a context the allocation cannot honor.
pub(crate) fn placement_ring(cfg: &Config, ec: &EngineConfig, k_fmt: DType, v_fmt: DType) -> bool {
    kv_ring_wanted(cfg, ec)
        && matches!(k_fmt, DType::F16 | DType::Q8_0)
        && matches!(kv_v_fmt(cfg, k_fmt, v_fmt), DType::F16 | DType::Q8_0)
}

/// Bytes a FULLY-RESIDENT dense session needs at one EXPLICIT prefill chunk height and KV format
/// pair: weights + the exact KV allocation ([`kv_bytes_estimate_fmt`]) + the activation reserve
/// ([`dense_act_reserve_at`]). The arithmetic both the fit math and the placement sweep compare
/// against the ceiling, in one place so the two cannot price a session differently.
pub(crate) fn dense_resident_need(
    cfg: &Config,
    weights: u64,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    align: KvAlign,
    k_fmt: DType,
    v_fmt: DType,
) -> u64 {
    weights
        .saturating_add(kv_bytes_estimate_fmt(
            cfg, want_ctx, ring, ubatch, align, k_fmt, v_fmt,
        ))
        .saturating_add(dense_act_reserve_at(cfg, want_ctx, ubatch))
}

/// Does a fully-resident dense session at this chunk height fit the ALLOCATOR's ceiling?
///
/// `vram.alloc_room()`, NOT `vram.available`: the VRAM guard reserves a fixed headroom below the
/// free figure and refuses anything that reaches into it, so a residency decision taken against
/// the raw figure can declare a model resident while planning 256 MiB the allocator will never
/// hand out — and then fail on an activation alloc mid-prefill.
pub(crate) fn dense_placement_fits(
    cfg: &Config,
    ec: &EngineConfig,
    weights: u64,
    vram: &infr_vulkan::VramInfo,
    want_ctx: usize,
    ubatch: usize,
    align: KvAlign,
    k_fmt: DType,
    v_fmt: DType,
) -> bool {
    let ring = placement_ring(cfg, ec, k_fmt, v_fmt);
    // A RESIDENT session prefills chunk-major unless the user forced the other order, so the
    // whole-prompt residual stream is priced only for the forced case. `Some(true)` is the sole
    // value that reaches here as layer-major: unset means "on iff streaming", and a session that
    // fits does not stream.
    let lm = if ec.paging.layer_major == Some(true) {
        layer_major_act_bytes(cfg, want_ctx, ubatch)
    } else {
        0
    };
    dense_resident_need(cfg, weights, want_ctx, ring, ubatch, align, k_fmt, v_fmt)
        .saturating_add(lm)
        <= vram.alloc_room()
}

/// The TALLEST [`ubatch_candidates`] rung at which this dense session fits resident, or `None` when
/// none of them does (the caller streams). The rung the placement sweep settles on — and the same
/// walk [`kv_fit_ctx_for`] makes when it decides a context fits.
pub(crate) fn dense_resident_rung(
    cfg: &Config,
    ec: &EngineConfig,
    weights: u64,
    vram: &infr_vulkan::VramInfo,
    want_ctx: usize,
    align: KvAlign,
    k_fmt: DType,
    v_fmt: DType,
) -> Option<usize> {
    ubatch_candidates(ec)
        .into_iter()
        .find(|&ub| dense_placement_fits(cfg, ec, weights, vram, want_ctx, ub, align, k_fmt, v_fmt))
}

/// Streaming budget: what is left of the allocator's ceiling for the dense weight-streaming arenas
/// once the always-resident weights, the KV cache and the activation reserve are paid for. Same
/// ceiling as [`dense_placement_fits`] — every byte this over-states is a slot the arena allocates
/// and the guard then refuses.
pub(crate) fn dense_stream_budget_at(
    cfg: &Config,
    ec: &EngineConfig,
    resident_weights: u64,
    vram: &infr_vulkan::VramInfo,
    want_ctx: usize,
    ubatch: usize,
    align: KvAlign,
    k_fmt: DType,
    v_fmt: DType,
) -> u64 {
    let ring = placement_ring(cfg, ec, k_fmt, v_fmt);
    let lm = stream_layer_major_bytes(cfg, ec, want_ctx, ubatch);
    vram.alloc_room()
        .saturating_sub(dense_resident_need(
            cfg,
            resident_weights,
            want_ctx,
            ring,
            ubatch,
            align,
            k_fmt,
            v_fmt,
        ))
        .saturating_sub(lm)
}

/// What a streamed session's layer-major prefill holds beside the arenas. A streamed session
/// prefills LAYER-MAJOR by default, which holds every chunk's residual stream at once — those
/// bytes come out of the same VRAM the arenas want, and they are allocated later, so the arena has
/// to leave room for them (see `layer_major_act_bytes`).
fn stream_layer_major_bytes(
    cfg: &Config,
    ec: &EngineConfig,
    want_ctx: usize,
    ubatch: usize,
) -> u64 {
    if ec.paging.layer_major == Some(false) {
        0
    } else {
        layer_major_act_bytes(cfg, want_ctx, ubatch)
    }
}

/// [`dense_stream_budget_at`] for a session whose KV cache spills whole (`kv.overflow`): the
/// cache takes none of the room, which holds the weight upload's staging (`staging`, live when
/// the cache is allocated) and the streamed session's spill reserve ([`kv_spill_reserve`])
/// instead, so what the arenas leave is the reserve and the cache's buffers land in system RAM
/// ([`kv_spill_split`] of a room at the reserve).
pub(crate) fn dense_stream_spill_budget_at(
    cfg: &Config,
    ec: &EngineConfig,
    resident_weights: u64,
    staging: u64,
    vram: &infr_vulkan::VramInfo,
    want_ctx: usize,
    ubatch: usize,
) -> u64 {
    vram.alloc_room()
        .saturating_sub(resident_weights)
        .saturating_sub(staging)
        .saturating_sub(kv_spill_reserve(cfg, ec, want_ctx, ubatch, true))
}

/// Headroom the MoE expert budget holds back on top of the dense weights and the KV cache: covers
/// activation scratch (pooled, but per-tag sizes scale with n_embd/n_ff and the `fp`/`kv_bytes`
/// terms are estimates, not the exact bytes gpu-allocator's block granularity commits) plus the
/// pager's own arena+staging+LUT allocations, which aren't counted in `fp` at all. Sized
/// empirically on Scout's 48-layer, 37 GB Q2_K pager placement — 512 MiB undershot by a few
/// hundred MiB and the guard rightly refused to over-commit.
const MOE_ACT_HEADROOM: u64 = 2 * 1024 * 1024 * 1024;

/// Bytes left for the MoE expert banks after this model's dense weights, its KV cache and
/// [`MOE_ACT_HEADROOM`] — against the allocator's ceiling, like every other budget here. `None`
/// when the dense half alone does not fit, which is not a paging decision but a hard error (dense
/// layer streaming does not cover an MoE model's dense part).
pub(crate) fn moe_expert_budget(vram: &infr_vulkan::VramInfo, dense: u64, kv: u64) -> Option<u64> {
    let room = vram.alloc_room();
    let base = dense.saturating_add(kv);
    (base <= room).then(|| room.saturating_sub(base.saturating_add(MOE_ACT_HEADROOM)))
}

/// The pager budget an explicit `paging.cache` (`INFR_CACHE`) sets: an absolute size as given, and
/// a percentage of the allocator's ceiling ([`VramInfo::alloc_room`](infr_vulkan::VramInfo::alloc_room)),
/// the room every other budget here is held to, so `100%` is a budget the guard can admit (plan 13
/// S3).
pub(crate) fn paging_cache_budget(spec: infr_core::SizeSpec, vram: &infr_vulkan::VramInfo) -> u64 {
    spec.resolve(vram.alloc_room())
}

/// Hard ceiling on [`kv_fit_ctx_for`]'s search. Reached only by a model whose KV bytes AND
/// activation reserve both PLATEAU with context — every attention layer sliding-window (so the
/// ring caps its rows) and head_dim 128 with no score tile. There the fit is bounded by nothing
/// physical, and a number is still needed; 4 Mi tokens is past any trained window in existence.
const CTX_FIT_SEARCH_CAP: usize = 1 << 22;

/// The EXACT largest context whose KV cache + activation reserve fit `available` bytes alongside
/// `weights` — the arithmetic half of [`SeamModel::kv_fit_ctx_fmt`], split out so it is testable
/// without a GPU (`kv_fit_*` in `seam_helper_tests`). `weights` is everything the device holds
/// beside the cache and the reserve: `SeamModel::preload_held` passes the weights as the arena
/// commits them, the upload staging and [`POST_KV_DEVICE_RESERVE`].
///
/// Two things make it exact where the old estimator was not:
///
///  - **KV bytes are the real allocation**, not a bytes-per-token rate divided into a budget:
///    [`kv_bytes_estimate_fmt`] prices each layer's K and V buffers through the same
///    `kv_fmt_bytes` sizer the runner hands `Backend::alloc`, including the SWA ring's row cap
///    and each side's own dtype. A `0.95` fudge factor used to stand in for the block-quant
///    rounding and the ring split that this now computes directly.
///  - **The reserve is priced at the chunk height placement will ACTUALLY settle on.** A context
///    is accepted when it fits at ANY height in [`ubatch_candidates`], because the dense
///    placement sweep walks that same ladder and will shrink the prefill chunk to keep the
///    session resident. Pricing only the default 1024-row chunk here made the KV-format decision
///    against an assumption placement then abandoned — which is exactly how gemma-3-12b talked
///    itself into an unnecessary q8 cache at ctx 131072.
///
/// Returns the RAW fit, which may be below [`MIN_SESSION_CTX`] (or `0`) — that is the signal the
/// refuse rung reads. `None` for a pure recurrent-state arch: no per-token KV to size.
///
/// MoE models keep the plain `total/12` heuristic: their placement budgets pager arenas
/// separately from this and never walks the dense chunk ladder.
pub(crate) fn kv_fit_ctx_for(
    cfg: &Config,
    ec: &EngineConfig,
    weights: u64,
    vram: &infr_vulkan::VramInfo,
    align: KvAlign,
    k_fmt: DType,
    v_fmt: DType,
) -> Option<usize> {
    // The ALLOCATOR's ceiling, derived by the same function the placement sweeps use, so the two
    // decide against one budget (see the `budgets_agree_with_the_allocator_ceiling` drift test).
    // MoE keeps the plain `total/12` heuristic: expert banks and pager arenas are budgeted by
    // `moe_expert_budget`, not by the dense activation reserve.
    let moe_reserve = cfg
        .moe
        .is_some()
        .then(|| (vram.total / 12).max(1024 * 1024 * 1024));
    kv_fit_ctx_in_budget(
        cfg,
        ec,
        vram.alloc_room().saturating_sub(weights),
        moe_reserve,
        &ubatch_candidates(ec),
        align,
        k_fmt,
        v_fmt,
    )
}

/// The search half of [`kv_fit_ctx_for`], against a budget that is ALREADY net of the weights:
/// the largest context whose KV cache plus its activation reserve fit `budget` bytes.
///
/// Split out because the two callers know the weight bytes with very different confidence. Before
/// the load, [`kv_fit_ctx_for`] can only subtract an ESTIMATE of them (for a dense model
/// `SeamModel::preload_held`: the arena's commitment at the least ([`weight_arena_floor`]), the
/// upload staging and [`POST_KV_DEVICE_RESERVE`], but not the tails first-fit strands past that
/// floor nor the driver's own memory; for an MoE model the bare `weight_footprint`). After the
/// load, the runner asks the device what is left (`Backend::device_alloc_room`) and passes that
/// here — a budget with the tails, the retained staging and the driver's own memory already
/// netted out, because they have been allocated. Same arithmetic either way; only the confidence
/// in the input differs.
///
/// `moe_reserve` is the MoE arm's flat headroom in place of the dense activation reserve (`None`
/// for a dense model).
pub(crate) fn kv_fit_ctx_in_budget(
    cfg: &Config,
    ec: &EngineConfig,
    budget: u64,
    moe_reserve: Option<u64>,
    cands: &[usize],
    align: KvAlign,
    k_fmt: DType,
    v_fmt: DType,
) -> Option<usize> {
    if (0..cfg.n_layer).all(|l| kv_row_elems(cfg, l) == (0, 0)) {
        return None;
    }
    // Same gate the runner applies (`generate_dense_backend`'s `kv_ring`) — see `placement_ring`.
    let ring = placement_ring(cfg, ec, k_fmt, v_fmt);
    let fits = |ctx: usize| -> bool {
        cands.iter().any(|&ubatch| {
            let kv = kv_bytes_estimate_fmt(cfg, ctx, ring, ubatch, align, k_fmt, v_fmt);
            let reserve = moe_reserve.unwrap_or_else(|| dense_act_reserve_at(cfg, ctx, ubatch));
            kv.saturating_add(reserve) <= budget
        })
    };
    // Monotone in ctx (both terms grow with it), so double-then-bisect finds the exact boundary.
    if !fits(0) {
        return Some(0);
    }
    let (mut lo, mut hi) = (0usize, 1usize);
    while hi < CTX_FIT_SEARCH_CAP && fits(hi) {
        lo = hi;
        hi = (hi * 2).min(CTX_FIT_SEARCH_CAP);
    }
    if fits(hi) {
        return Some(hi); // plateaued: the cap IS the answer.
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

/// The units a fully resident weight load uploads, in bytes: each tensor at the bytes the runner
/// uploads it as ([`crate::weights::tensor_resident_bytes`]), except that a layer's q/k/v
/// projections, and its gate/up projections, are one unit each wherever the runner uploads them
/// concatenated ([`runner::fuse_qkv_decision`], [`runner::fuse_gu_decision`]). Not covered: the
/// tensors the runner widens to f32 by name (B193) count at their file size, and the small
/// ones-vectors some architectures upload count not at all.
pub(crate) fn weight_upload_units(
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    combined_gu: bool,
) -> Vec<u64> {
    weight_upload_units_of(g, cfg, ec, combined_gu, |_| true)
}

/// [`weight_upload_units`] of the tensors whose names `keep` accepts: the units a load that
/// uploads only those makes, such as the weights a streamed session keeps resident.
pub(crate) fn weight_upload_units_of(
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    combined_gu: bool,
    keep: impl Fn(&str) -> bool,
) -> Vec<u64> {
    let fuse_qkv = runner::fuse_qkv_decision(combined_gu, g, cfg, ec);
    let fuse_gu = runner::fuse_gu_decision(combined_gu, g, cfg);
    // The fused unit a tensor belongs to, keyed by its layer and which concatenation it joins.
    let fused_unit = |name: &str| -> Option<(usize, bool)> {
        let (layer, tensor) = name.strip_prefix("blk.")?.split_once('.')?;
        let layer = layer.parse().ok()?;
        match tensor {
            "attn_q.weight" | "attn_k.weight" | "attn_v.weight" if fuse_qkv => Some((layer, false)),
            "ffn_gate.weight" | "ffn_up.weight" if fuse_gu => Some((layer, true)),
            _ => None,
        }
    };
    let mut units = Vec::new();
    let mut fused: std::collections::BTreeMap<(usize, bool), u64> = Default::default();
    for t in g.tensors().iter().filter(|t| keep(&t.name)) {
        let numel: usize = t.shape.iter().product();
        let bytes = crate::weights::tensor_resident_bytes(t.dtype, numel, t.nbytes);
        match fused_unit(&t.name) {
            Some(key) => *fused.entry(key).or_default() += bytes,
            None => units.push(bytes),
        }
    }
    units.extend(fused.into_values());
    units
}

/// The least device memory the resident-BDA arena can commit for upload units of `units` bytes,
/// by the placement rule [`infr_vulkan::WeightLoadCost`] documents: every unit rounded up to the
/// arena's alignment, a unit of at least a block taking a block of exactly its size, and the rest
/// sharing blocks, of which there are never fewer than their rounded total fills. A LOWER bound
/// on what the arena commits for the units it is given, whatever order they arrive in: first-fit
/// can strand a block's tail and open more blocks, never fewer. The units themselves
/// ([`weight_upload_units`]) can include tensors a run never uploads — `token_embd` when the GPU
/// embedding gather is off on an untied model, gemma-4 E2B's `per_layer_token_embd` when its
/// on-device gather is off — which over-counts those, the safe direction for a fit.
pub(crate) fn weight_arena_floor(units: &[u64], cost: infr_vulkan::WeightLoadCost) -> u64 {
    let (mut whole, mut shared) = (0u64, 0u64);
    for &unit in units {
        let rounded = unit.max(1).next_multiple_of(cost.align);
        if rounded >= cost.block_min {
            whole += rounded;
        } else {
            shared += rounded;
        }
    }
    whole + shared.div_ceil(cost.block_min) * cost.block_min
}

/// What a dense model's resident weight load takes from the device's room besides the KV cache and
/// what is allocated after it, priced before the load: `(weights, staging)`, the weights as the
/// resident arena commits them at the least ([`weight_arena_floor`]) and the upload staging still
/// live when the post-load re-clamp reads the room. The weights half of what the fit before the
/// load (`SeamModel::preload_held`) and the spill rung of dense placement hold back.
pub(crate) fn resident_load_held(
    vk: &infr_vulkan::VulkanBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
) -> (u64, u64) {
    let cost = vk.weight_load_cost();
    let units = weight_upload_units(g, cfg, ec, vk.capabilities().combined_gu);
    (weight_arena_floor(&units, cost), cost.staging)
}

/// The per-side KV dtypes a fit/footprint estimate must price: the user's explicit
/// `INFR_KV_TYPE_K`/`_V` when set, else the default ladder (legacy `INFR_KV_Q8`, then a
/// placement-pinned or candidate auto-q8, then f16) — the same ladder `parse_kv_fmt` walks in
/// the runner. ESTIMATE ONLY in one direction: the runner additionally gates each format on
/// backend/alignment and falls back to f16, so a gated-out low-bit request under-estimates
/// here and the alloc-time VRAM guard backstops it.
pub(crate) fn kv_fit_fmts(cfg: &Config, ec: &EngineConfig, auto_q8: bool) -> (DType, DType) {
    let side = |want: Option<DType>| match want {
        Some(dt) => dt,
        None if ec.kv.force_q8 || auto_q8 => DType::Q8_0,
        None => DType::F16,
    };
    let k = side(ec.kv.type_k);
    (k, kv_v_fmt(cfg, k, side(ec.kv.type_v)))
}

/// Device memory a live session commits AFTER its KV cache is allocated, which therefore has to be
/// held back from [`reclamp_ctx_to_live_room`]'s budget: the compute pipelines, descriptor pools
/// and command buffers the driver builds while recording the first forwards. It is the driver's
/// own memory, so no `Backend` allocation accounts for it and only the live budget query sees it —
/// which is why it is a term here rather than a footprint somewhere.
///
/// Sized by measurement (7900 XTX, RADV): the gap between the driver's reported used bytes and
/// this backend's own tally grows from 187 MiB right after the weight load to 368 MiB at the peak
/// of a deep gemma-4-31B prefill — 181 MiB of pipeline/descriptor memory built during the run,
/// with the idle desktop's 43 MiB present in both figures and cancelling out. Rounded up to 256
/// MiB: the term protects the LAST thing to allocate, and under-reserving it lands as the
/// mid-prefill failure this whole path exists to prevent.
///
/// Deliberately NOT the same money as `infr_vulkan`'s `GUARD_HEADROOM`. That headroom is the
/// allocator's own cushion below the free figure (alignment, block rounding); spending it on
/// pipelines is what left the guard nothing to cushion with and turned a 2 MiB overshoot into a
/// failed request.
pub(crate) const POST_KV_DEVICE_RESERVE: u64 = 256 * 1024 * 1024;

/// The device memory a spilled cache leaves free for what the run allocates after it, term by
/// term ([`kv_spill_reserve_terms`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KvSpillReserve {
    /// The activation reserve the fit prices ([`dense_act_reserve_at`]).
    pub(crate) activations: u64,
    /// The layer-major prefill's residual stream, where placement prices one
    /// ([`layer_major_act_bytes`]).
    pub(crate) layer_major: u64,
    /// [`POST_KV_DEVICE_RESERVE`].
    pub(crate) driver: u64,
    /// The user's extra margin (`kv.overflow_reserve_mb`, [`infr_core::budget::reserve_bytes`]).
    pub(crate) margin: u64,
}

impl KvSpillReserve {
    pub(crate) fn total(&self) -> u64 {
        self.activations
            .saturating_add(self.layer_major)
            .saturating_add(self.driver)
            .saturating_add(self.margin)
    }
}

/// The terms of [`kv_spill_reserve`] at window `ctx` and prefill chunk `ubatch`, for a session
/// whose weights stream (`streamed`) or not. The layer-major term is the one placement prices: on
/// a streamed session unless `paging.layer_major` is off, on a resident one only when it is forced
/// on.
pub(crate) fn kv_spill_reserve_terms(
    cfg: &Config,
    ec: &EngineConfig,
    ctx: usize,
    ubatch: usize,
    streamed: bool,
) -> KvSpillReserve {
    let layer_major = if streamed {
        stream_layer_major_bytes(cfg, ec, ctx, ubatch)
    } else if ec.paging.layer_major == Some(true) {
        layer_major_act_bytes(cfg, ctx, ubatch)
    } else {
        0
    };
    KvSpillReserve {
        activations: dense_act_reserve_at(cfg, ctx, ubatch),
        layer_major,
        driver: POST_KV_DEVICE_RESERVE,
        margin: infr_core::budget::reserve_bytes(ec.kv.overflow_reserve_mb),
    }
}

/// The device memory a spilled cache leaves free for what the run allocates after it
/// ([`kv_spill_reserve_terms`], summed). The one figure placement's spill rung, the re-clamp
/// after the load and the backend's KV placement (`Backend::set_kv_spill_reserve`) hold back for
/// a spill.
pub(crate) fn kv_spill_reserve(
    cfg: &Config,
    ec: &EngineConfig,
    ctx: usize,
    ubatch: usize,
    streamed: bool,
) -> u64 {
    kv_spill_reserve_terms(cfg, ec, ctx, ubatch, streamed).total()
}

/// The prefill chunk a spilled session runs at: the one the user pinned (`INFR_UBATCH`), else the
/// shortest [`ubatch_candidates`] allows. A spilled cache already gave up device memory, and a
/// shorter chunk shrinks the activations it must leave room for, so it takes the cheapest height
/// prefill can run at rather than the one placement would pick for a resident cache.
pub(crate) fn spill_ubatch(ec: &EngineConfig) -> usize {
    if user_pinned_ubatch(ec) {
        ubatch_rows(ec)
    } else {
        ubatch_candidates(ec)
            .into_iter()
            .min()
            .expect("ubatch_candidates is never empty")
    }
}

/// How a spilled KV cache divides between device memory and system RAM, in bytes, each buffer at
/// what the device commits for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KvSplit {
    pub(crate) vram: u64,
    pub(crate) host: u64,
}

/// How `n_slots` caches of `ctx` tokens divide under `kv.overflow` when `room` bytes of device
/// memory are free before the first buffer, `reserve` ([`kv_spill_reserve`]) must stay free and
/// `cap` ([`kv_spill_cap`]) bounds what the cache places in device memory:
/// every buffer of [`kv_slot_allocs`], slot after slot, in the order the runner allocates them,
/// through the backend's own placement rule ([`infr_vulkan::kv_vram_bytes`]). The one answer to
/// "how much of this cache goes to VRAM" for the fit before the load, placement and the re-clamp
/// after it, each passing the room as it knows it; the allocation applies the same rule to the
/// live budget, buffer by buffer.
///
/// The recurrent-state snapshots come last, as in [`kv_slot_allocs`], priced against the room the
/// init buffers leave. A slot allocates them on first use, during its first generation rather than
/// at its cold init, when the run's activations may hold part of that room: a snapshot this prices
/// in device memory can land in system RAM, never the other way (the reserve it must leave is the
/// same either way).
#[allow(clippy::too_many_arguments)]
pub(crate) fn kv_spill_split(
    cfg: &Config,
    ctx: usize,
    ring: bool,
    ubatch: usize,
    align: KvAlign,
    k_fmt: DType,
    v_fmt: DType,
    n_slots: usize,
    room: u64,
    reserve: u64,
    cap: Option<u64>,
) -> KvSplit {
    let buffers = || {
        (0..n_slots.max(1)).flat_map(move |_| {
            kv_slot_allocs(cfg, ctx, ring, ubatch, align.rows, k_fmt, v_fmt)
                .flat_map(|(k, v)| [k, v])
                .map(move |b| (b as u64, align.commit.committed(b as u64)))
        })
    };
    let vram = infr_vulkan::kv_vram_bytes(buffers(), room, reserve, cap);
    KvSplit {
        vram,
        host: buffers().map(|(_, c)| c).sum::<u64>() - vram,
    }
}

/// The diagnostic cap `kv.overflow_vram_mb` puts on a spilled cache's device memory
/// ([`infr_vulkan::kv_overflow_vram_cap`]), which the allocation applies: every split passes it.
pub(crate) fn kv_spill_cap(ec: &EngineConfig) -> Option<u64> {
    infr_vulkan::kv_overflow_vram_cap(ec)
}

/// Refuse a spilled cache whose system-RAM part does not fit the host: `Err`, naming the sizes,
/// when `split.host` exceeds what the host can spare a run of `host_free`, the memory it reports
/// it could commit ([`infr_core::hostmem::available_bytes`]; `None` where it cannot say, which
/// refuses nothing). The spilled buffers are anonymous memory the OS cannot reclaim, so they take
/// the headroom the repo's host arenas leave ([`infr_core::hostmem::spare_bytes`]), after
/// `committed`: what the run already holds there that the probe does not see, the weight-streaming
/// host tier ([`host_tier_bytes`]).
pub(crate) fn refuse_unhostable_kv(
    ctx: usize,
    split: KvSplit,
    host_free: Option<u64>,
    committed: u64,
    k_fmt: DType,
    v_fmt: DType,
) -> AResult<()> {
    let Some(free) = host_free else {
        return Ok(());
    };
    let spare = infr_core::hostmem::spare_bytes(free, committed);
    if split.host <= spare {
        return Ok(());
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    Err(anyhow!(
        "context {ctx} cannot be served even with its KV cache spilled to system RAM \
         (INFR_KV_OVERFLOW): {:.2} GiB of its {k_fmt:?}/{v_fmt:?} cache does not fit the device \
         ({:.2} GiB does), and of the {:.2} GiB the host reports it could commit, {:.2} GiB is \
         left for it after the {:.2} GiB the weights' host tier takes and the {:.2} GiB kept for \
         the rest of the machine. Ask for a smaller window (--ctx N / INFR_CTX=N), quantize the \
         cache (INFR_KV_TYPE_K=q8_0 INFR_KV_TYPE_V=q8_0), or free host memory.",
        gib(split.host),
        gib(split.vram),
        gib(free),
        gib(spare),
        gib(committed),
        gib(free.saturating_sub(committed).saturating_sub(spare)),
    ))
}

/// The host-memory check for the `n_forks` slots an engine forks off `slot` once its cold init has
/// placed it (`ParallelSeam::init_slots`), taken before the first fork is allocated: under an
/// armed spill, every fork goes through the backend's placement past the reserve the cold init
/// left set, so the part of the `n_forks` caches that would spill
/// ([`kv_spill_split`] against the room the device reports now) must fit what the host can spare
/// ([`refuse_unhostable_kv`]). Nothing to check on a backend that does not spill or cannot report
/// a room. The diagnostic cap is counted from the first fork, where the allocation counts what
/// `slot` already placed against it too, so under the cap this prices the forks' host part low.
pub(crate) fn refuse_unhostable_forks(
    be: &dyn Backend,
    cfg: &Config,
    ec: &EngineConfig,
    slot: &SeamKv,
    n_forks: usize,
) -> AResult<()> {
    if n_forks == 0 || !be.kv_spill_armed() {
        return Ok(());
    }
    let Some(room) = be.device_alloc_room() else {
        return Ok(());
    };
    let reserve = kv_spill_reserve(cfg, ec, slot.max_ctx, slot.ubatch, be.dense_paged());
    let split = kv_spill_split(
        cfg,
        slot.max_ctx,
        slot.kv_ring,
        slot.ubatch,
        KvAlign::of(be),
        slot.k_fmt,
        slot.v_fmt,
        n_forks,
        room,
        reserve,
        kv_spill_cap(ec),
    );
    refuse_unhostable_kv(
        slot.max_ctx,
        split,
        infr_core::hostmem::available_bytes(),
        host_tier_bytes(),
        slot.k_fmt,
        slot.v_fmt,
    )
}

/// What [`reclamp_ctx_to_live_room`] settled on: the window and formats to allocate, and whether
/// the cache is spilled (the ladder's `Spill` rung), so the runner holds the spill reserve back
/// from this cache's own buffers only when it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LiveCtx {
    pub(crate) ctx: usize,
    pub(crate) k_fmt: DType,
    pub(crate) v_fmt: DType,
    pub(crate) spilled: bool,
}

/// Re-decide the session's context against what the device says is LEFT, now that the weights are
/// resident — the runner's cold init calls this between the weight upload and the KV allocation.
///
/// **Why this exists at all.** Every pre-load estimate of "will this fit?" has to model the weight
/// bytes. The fit before the load prices the arena's block rounding as a lower bound
/// ([`weight_arena_floor`]) and the upload staging, but not what first-fit strands beyond that
/// bound nor the driver's own memory. The arena's tails were measured at **+2.20%** of the
/// footprint on gemma-4-31B UD-Q5_K_XL (481 MiB), +2.43% on gemma-3-12b and +1.16% on Qwen3-14B
/// Q4_K_M; how much of that the bound covers was not measured. At THIS point none of that has to
/// be modelled: the weights, the tails, the staging and the driver's memory so far are all
/// allocated, so the live budget query prices them exactly. What remains predicted is the
/// activation reserve ([`dense_act_reserve_at`]) and [`POST_KV_DEVICE_RESERVE`].
///
/// **Only ever shrinks**, only a context the SESSION chose, and never below the minimum
/// ([`post_load_ctx`] has the rule): a window that would fall below it is served in q8_0 when the
/// auto-q8 policy allows it, which re-pins the session's KV format before anything is allocated
/// and is why the formats are returned, else spilled at the minimum under `kv.overflow`, else
/// refused with the numbers. A user-pinned `--ctx`/`INFR_CTX` is documented as taken verbatim
/// (never clamped), so it is warned about and honored — [`refuse_unallocatable_kv`] refuses one
/// whose KV cache alone cannot land, and the alloc-time VRAM guard stays the backstop for the
/// rest. Returns the window and formats unchanged on a backend with no budget to report
/// (`Backend::device_alloc_room` → `None`: CPU, Metal today).
///
/// May also LOWER the pinned prefill chunk ([`repin_ubatch_lower`]) when a shorter one serves more
/// context — the same trade the pre-load sweep makes, retaken with the weight bytes known.
pub(crate) fn reclamp_ctx_to_live_room(
    be: &dyn Backend,
    cfg: &Config,
    ec: &EngineConfig,
    want_ctx: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> AResult<LiveCtx> {
    let host_free = infr_core::hostmem::available_bytes();
    reclamp_ctx_with_host(be, cfg, ec, want_ctx, k_fmt, v_fmt, host_free)
}

/// [`reclamp_ctx_to_live_room`] against a stated `host_free` (what
/// [`infr_core::hostmem::available_bytes`] reports) rather than the machine's own reading, so a
/// test of the spill's host check does not depend on the memory of the machine it runs on.
fn reclamp_ctx_with_host(
    be: &dyn Backend,
    cfg: &Config,
    ec: &EngineConfig,
    want_ctx: usize,
    k_fmt: DType,
    v_fmt: DType,
    host_free: Option<u64>,
) -> AResult<LiveCtx> {
    let served = |ctx: usize, k_fmt: DType, v_fmt: DType| LiveCtx {
        ctx,
        k_fmt,
        v_fmt,
        spilled: false,
    };
    let Some(room) = be.device_alloc_room() else {
        return Ok(served(want_ctx, k_fmt, v_fmt));
    };
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let budget = room.saturating_sub(POST_KV_DEVICE_RESERVE);
    let align = KvAlign::of(be);
    // Walk the chunk ladder HERE too, and price each rung on its own: a shorter chunk shrinks both
    // the activation reserve and the SWA ring, so it buys context, and the rung the pre-load sweep
    // pinned was chosen against an estimate of the weight bytes. Pricing only the pinned rung
    // leaves that context on the floor (measured on gemma-4-31B: 10 440 tokens at the pinned
    // 256-row chunk against 15 440 at 128). Tallest-first, so a rung is only lowered when the
    // taller one genuinely cannot serve the window.
    let cands = ubatch_candidates(ec);
    // MoE: the pager's arenas are already allocated by the binder at this point, so the live room
    // has them netted out and the flat `total/12` stand-in would double-count. The dense reserve
    // is what an MoE forward's activations actually need beside them.
    //
    // The fit prices a session the same whether or not its cache can spill (`kv.overflow` only
    // decides what happens to a window that does not fit), so arming the spill changes no window
    // served whole; a spilled one is held to the spill reserve below.
    let spill = be.kv_spill_armed();
    let streamed = be.dense_paged();
    let at = |ub: usize, k: DType, v: DType| {
        kv_fit_ctx_in_budget(cfg, ec, budget, None, &[ub], align, k, v)
    };
    // The tallest rung that serves the whole window, else the rung that serves the most of it:
    // `(chunk, fit)` for one format pair, or `None` for a pure recurrent-state arch, which has no
    // per-token KV to size.
    let walk = |k: DType, v: DType| -> Option<(usize, usize)> {
        at(cands[0], k, v)?;
        let fits = || cands.iter().map(|&ub| (ub, at(ub, k, v).unwrap_or(0)));
        Some(fits().find(|&(_, fit)| fit >= want_ctx).unwrap_or_else(|| {
            fits()
                .max_by_key(|&(ub, fit)| (fit, ub))
                .expect("ubatch_candidates is never empty")
        }))
    };
    let Some((chunk, fit)) = walk(k_fmt, v_fmt) else {
        return Ok(served(want_ctx, k_fmt, v_fmt));
    };
    let explicit = ec.device.ctx.is_some();
    // q8_0 is tried only where the pre-load ladder may pick it on its own (`may_auto_q8`) and the
    // runner reads a q8_0 pin (`parse_kv_fmt`'s placement arm is Vulkan-only).
    let q8_allowed =
        be.name() == "vulkan" && (k_fmt, v_fmt) == (DType::F16, DType::F16) && may_auto_q8(cfg, ec);
    let mut q8_walk = None;
    let decision = post_load_ctx(want_ctx, explicit, spill, fit, || {
        q8_walk = q8_allowed.then(|| walk(DType::Q8_0, DType::Q8_0)).flatten();
        q8_walk.map(|(_, fit)| fit)
    });
    // The chunk the chosen cache was priced at. Lowered BEFORE the KV buffers are sized: the ring
    // is `window + chunk` rows, and the reserve the fit was priced with is that height's.
    let chunk = match (decision, q8_walk) {
        (CtxRung::ServeQ8(_), Some((q8_chunk, _))) => q8_chunk,
        // Nothing fits at any height, so the walk's pick says nothing: a spilled session runs at
        // its own chunk, and its reserve is priced there.
        (CtxRung::Spill(_), _) => spill_ubatch(ec),
        _ => chunk,
    };
    if chunk < ubatch_rows(ec) {
        repin_ubatch_lower(chunk);
    }
    let ubatch = ubatch_rows(ec);
    let ring = placement_ring(cfg, ec, k_fmt, v_fmt);
    tracing::debug!(
        room_bytes = room,
        post_kv_reserve_bytes = POST_KV_DEVICE_RESERVE,
        act_reserve_bytes = dense_act_reserve_at(cfg, fit, ubatch),
        kv_bytes = kv_bytes_estimate_fmt(cfg, fit, ring, ubatch, align, k_fmt, v_fmt),
        ubatch,
        want_ctx,
        fit,
        ?decision,
        "ctx fit against the live device room"
    );
    match decision {
        CtxRung::Serve(ctx) if ctx == want_ctx => {
            if fit < want_ctx {
                tracing::warn!(
                    requested_ctx = want_ctx,
                    fits_ctx = fit,
                    "ctx: only {fit} tokens fit the {:.2} GiB the device reports free after the \
                     weights, but the context was set explicitly — honoring it. The allocation \
                     guard will fail this session if it really does not fit; lower INFR_CTX/--ctx \
                     to pick a window that does.",
                    gib(room),
                );
            }
            Ok(served(want_ctx, k_fmt, v_fmt))
        }
        CtxRung::Serve(ctx) => {
            tracing::warn!(
                requested_ctx = want_ctx,
                fits_ctx = ctx,
                "ctx clamp (measured): {want_ctx} -> {ctx} tokens against the {:.2} GiB the device \
                 reports free after the weights are resident (KV {:.2} GiB + activation reserve \
                 {:.2} GiB at a {ubatch}-row chunk, minus {:.2} GiB held for the driver's own \
                 later allocations) — the pre-load estimate prices the weights' arena and staging \
                 but not the driver's own memory; set INFR_CTX to override",
                gib(room),
                gib(kv_bytes_estimate_fmt(cfg, ctx, ring, ubatch, align, k_fmt, v_fmt)),
                gib(dense_act_reserve_at(cfg, ctx, ubatch)),
                gib(POST_KV_DEVICE_RESERVE),
            );
            Ok(served(ctx, k_fmt, v_fmt))
        }
        CtxRung::ServeQ8(ctx) => {
            // Under the session's placement scope, so every later call of this session reads q8_0
            // too (`parse_kv_fmt`); nothing is allocated in either format yet.
            pin_kv_auto_q8();
            tracing::warn!(
                requested_ctx = want_ctx,
                f16_fit_ctx = fit,
                q8_fit_ctx = ctx,
                kv_dtype = "q8_0",
                "kv auto-quant (measured): q8_0 KV cache — after the weights an f16 cache leaves \
                 only {fit} tokens of context, below the usable minimum; serving {ctx} tokens in \
                 q8_0 (decode is ~10-16% slower on q8). Set INFR_KV_TYPE_K/V=f16 to force f16."
            );
            Ok(served(ctx, DType::Q8_0, DType::Q8_0))
        }
        CtxRung::Spill(ctx) => {
            let terms = kv_spill_reserve_terms(cfg, ec, ctx, ubatch, streamed);
            let reserve = terms.total();
            let split = kv_spill_split(
                cfg,
                ctx,
                ring,
                ubatch,
                align,
                k_fmt,
                v_fmt,
                1,
                room,
                reserve,
                kv_spill_cap(ec),
            );
            // The room measured, not estimated: a spilled cache can give the run nothing, so the
            // weights (and on a streamed session the arenas) must have left the reserve itself.
            // Placement admitted the session against a lower bound on the weights' arena, so it
            // can land short; refuse it here, before a KV buffer is allocated, rather than have
            // an activation refused mid-forward.
            if room < reserve {
                return Err(anyhow!(
                    "context {ctx} cannot be served on this device even with its KV cache in \
                     system RAM (INFR_KV_OVERFLOW): after its weights are resident the device \
                     reports {:.2} GiB allocatable, and the run needs {:.2} GiB of it after the \
                     cache ({:.2} GiB of activations at a {ubatch}-row chunk, {:.2} GiB of \
                     layer-major prefill, {:.2} GiB held for the driver's own later allocations, \
                     {:.2} GiB of kv.overflow_reserve_mb), {:.2} GiB more than is left. Free \
                     device memory held by other applications, or use a smaller quant of the \
                     model.",
                    gib(room),
                    gib(reserve),
                    gib(terms.activations),
                    gib(terms.layer_major),
                    gib(terms.driver),
                    gib(terms.margin),
                    gib(reserve - room),
                ));
            }
            refuse_unhostable_kv(ctx, split, host_free, host_tier_bytes(), k_fmt, v_fmt)?;
            tracing::warn!(
                requested_ctx = want_ctx,
                fits_ctx = fit,
                kv_vram_bytes = split.vram,
                kv_host_bytes = split.host,
                kv_snapshot_bytes = (0..cfg.n_layer)
                    .filter_map(|l| kv_recurrent_alloc_bytes(cfg, l))
                    .map(|(k, v)| (k + v) as u64)
                    .sum::<u64>(),
                "ctx overflow (measured): only {fit} tokens fit the {:.2} GiB the device reports \
                 free after the weights; keeping a {ctx}-token context with {:.2} GiB of its KV \
                 cache in VRAM and {:.2} GiB in system RAM (INFR_KV_OVERFLOW), read over PCIe, \
                 leaving {:.2} GiB free for the activations and the driver's own later \
                 allocations (a recurrent-state snapshot is priced last and placed when first \
                 taken, which can move it to system RAM)",
                gib(room),
                gib(split.vram),
                gib(split.host),
                gib(reserve),
            );
            Ok(LiveCtx {
                ctx,
                k_fmt,
                v_fmt,
                spilled: true,
            })
        }
        CtxRung::Refuse => {
            let floor = want_ctx.min(MIN_SESSION_CTX);
            let kv = kv_bytes_estimate_fmt(cfg, floor, ring, ubatch, align, k_fmt, v_fmt);
            let act = dense_act_reserve_at(cfg, floor, ubatch);
            let need = kv + act + POST_KV_DEVICE_RESERVE;
            Err(anyhow!(
                "this model does not fit the device's free memory: after its weights are resident \
                 the device reports {:.2} GiB allocatable, and the smallest usable context, \
                 {floor} tokens, needs {:.2} GiB ({:.2} GiB of {k_fmt:?}/{v_fmt:?} KV cache, \
                 {:.2} GiB of activations at a {ubatch}-row chunk, and {:.2} GiB held for the \
                 driver's own later allocations), {:.2} GiB more than is left. What fits: {fit} \
                 tokens in {k_fmt:?}/{v_fmt:?}{}. Free device memory held by other applications, \
                 ask for a smaller window (--ctx N / INFR_CTX=N), {}use INFR_KV_OVERFLOW=1 to \
                 place the cache in system RAM, cap the expert cache (INFR_CACHE / \
                 paging.cache), or use a smaller quant of the model.",
                gib(room),
                gib(need),
                gib(kv),
                gib(act),
                gib(POST_KV_DEVICE_RESERVE),
                gib(need.saturating_sub(room)),
                match q8_walk {
                    Some((_, q8)) => format!(", {q8} in q8_0"),
                    None => String::new(),
                },
                if q8_walk.is_none() && (k_fmt, v_fmt) != (DType::Q8_0, DType::Q8_0) {
                    "try a q8_0 cache (INFR_KV_TYPE_K=q8_0 INFR_KV_TYPE_V=q8_0), "
                } else {
                    ""
                },
            ))
        }
    }
}

/// What a session's window becomes against the room it is fitted to, from the fits alone: the
/// answer of the fit before the load ([`preload_ctx`]) and of the post-load re-clamp
/// ([`post_load_ctx`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CtxRung {
    /// Serve this many tokens in the session's formats: the window asked for when it fits or was
    /// set explicitly, else the largest that fits.
    Serve(usize),
    /// Serve this many tokens with a q8_0 cache, where the session's own formats cannot hold the
    /// window and q8_0 can, or leave less than the minimum and q8_0 does not.
    ServeQ8(usize),
    /// Keep this many tokens with the part of the cache that does not fit placed in system RAM
    /// (`kv.overflow`): the window a user set, or the minimum of one the session chose.
    Spill(usize),
    /// Nothing at or above the minimum fits.
    Refuse,
}

/// The rule the fit before the load applies to a session's window, given `fit`, the largest window
/// the session's formats hold per cache, and `q8_fit`, asked only when `fit` falls short: the q8_0
/// cache's fit where the auto-q8 policy allows q8_0 for this session, else `None`; `overflow` is
/// whether the cache can spill (`Backend::kv_spill_armed`). It is the ladder plan 13 decided for
/// B62, in order:
///
/// 1. A window that fits is served as asked.
/// 2. One a q8_0 cache holds whole is served as asked, in q8_0.
/// 3. One the user set (`explicit`) is never cut: under `kv.overflow` it is kept with the cache
///    spilled, else served as asked, where `refuse_unallocatable_kv` refuses one whose cache cannot
///    land at all.
/// 4. One the session chose is cut to the fit, never below the minimum; below it a q8_0 cache that
///    holds the minimum is served at what it holds, then `kv.overflow` keeps the minimum with the
///    cache spilled, then it is refused ([`post_load_ctx`]'s own rule, shared with it).
///
/// So turning `kv.overflow` on spills a session-chosen window only when nothing usable fits VRAM:
/// the clamp comes first. Whatever this serves, [`post_load_ctx`] keeps for the same room.
pub(crate) fn preload_ctx(
    want: usize,
    explicit: bool,
    overflow: bool,
    fit: usize,
    q8_fit: impl FnOnce() -> Option<usize>,
) -> CtxRung {
    if fit >= want {
        return CtxRung::Serve(want);
    }
    let q8 = q8_fit();
    if q8.is_some_and(|q8| q8 >= want) {
        return CtxRung::ServeQ8(want);
    }
    match (explicit, overflow) {
        (true, true) => CtxRung::Spill(want),
        (true, false) => CtxRung::Serve(want),
        (false, _) => clamp_rung(want, overflow, fit, || q8),
    }
}

/// The rule [`reclamp_ctx_to_live_room`] applies once the weights are resident, given `fit`, the
/// largest window the session's formats hold now, and `q8_fit`, asked only when needed: the q8_0
/// cache's fit where the auto-q8 policy allows q8_0 for this session, else `None`. `overflow` is
/// whether the cache can spill (`Backend::kv_spill_armed`).
///
/// A window that fits is served as asked; one that the user set (`explicit`, taken verbatim) and
/// does not fit is kept with its cache spilled under `overflow`, else served as asked. Else it is
/// cut to the fit, but never below the minimum: `MIN_SESSION_CTX`, or the window itself when it
/// asked for less. Below that, a q8_0 cache that holds the minimum is served at what it holds, up
/// to the window, as the pre-load clamp's rung does; then `overflow` keeps the minimum with the
/// cache spilled (B62); then it is refused.
pub(crate) fn post_load_ctx(
    want: usize,
    explicit: bool,
    overflow: bool,
    fit: usize,
    q8_fit: impl FnOnce() -> Option<usize>,
) -> CtxRung {
    if fit >= want {
        return CtxRung::Serve(want);
    }
    if explicit {
        return if overflow {
            CtxRung::Spill(want)
        } else {
            CtxRung::Serve(want)
        };
    }
    clamp_rung(want, overflow, fit, q8_fit)
}

/// The clamp both sides apply to a session-chosen window that `fit` does not hold whole: cut to the
/// fit, never below the minimum (`MIN_SESSION_CTX`, or the window when it asked for less); below
/// it the q8_0 cache that holds the minimum, then the spill at the minimum, then the refusal.
fn clamp_rung(
    want: usize,
    overflow: bool,
    fit: usize,
    q8_fit: impl FnOnce() -> Option<usize>,
) -> CtxRung {
    let floor = want.min(MIN_SESSION_CTX);
    if fit >= floor {
        return CtxRung::Serve(fit);
    }
    if let Some(q8) = q8_fit().filter(|&q8| q8 >= floor) {
        return CtxRung::ServeQ8(q8.min(want));
    }
    if overflow {
        return CtxRung::Spill(floor);
    }
    CtxRung::Refuse
}

/// Whether a fit may switch a session's KV cache to q8_0 on its own: the user set no KV format and
/// no rung already pinned one, and the model is dense with a q8_0-compatible layout — the case the
/// auto-q8 rungs were validated on (MoE placement budgets its pager arenas apart from the fit).
pub(crate) fn may_auto_q8(cfg: &Config, ec: &EngineConfig) -> bool {
    cfg.moe.is_none() && !kv_auto_q8() && kv_unset(cfg, ec) && kv_q8_layout_ok(cfg)
}

/// Bytes a recurrent layer's two state buffers allocate (`Config::recurrent_state_elems`, f32
/// whatever the session's KV formats), or `None` for a layer that caches K/V per token. The slot's
/// live state ([`kv_layer_alloc_bytes`]) and the one snapshot of it a slot can hold
/// (`SeamKv::mtp_delta_ckpt`, shared by MTP rollback and the decode loop's prefill-boundary
/// snapshot) are each this size.
pub(crate) fn kv_recurrent_alloc_bytes(cfg: &Config, l: usize) -> Option<(usize, usize)> {
    let f32_bytes = |elems: usize| elems * std::mem::size_of::<f32>();
    cfg.recurrent_state_elems(l)
        .map(|(k_elems, v_elems)| (f32_bytes(k_elems), f32_bytes(v_elems)))
}

/// Bytes layer `l`'s two cache buffers ALLOCATE at `want_ctx` and prefill chunk `ubatch` —
/// `(K side, V side)`, exactly what the runner's cold init and `SeamKv::fork` hand `Backend::alloc`
/// with `BufferUsage::KvCache`, and the per-layer term of every estimate ([`kv_slot_allocs`]).
/// A recurrent layer (qwen35 DeltaNet, LFM short convolution) holds its fixed-size f32 state there
/// instead ([`Config::recurrent_state_elems`], not `want_ctx`-scaled); an attention layer holds
/// `kv_rows_at` rows of its own per-side width.
pub(crate) fn kv_layer_alloc_bytes(
    cfg: &Config,
    l: usize,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    row_align: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> (usize, usize) {
    if let Some(state) = kv_recurrent_alloc_bytes(cfg, l) {
        return state;
    }
    // Per-token row widths, MLA included: DeepSeek2 caches ONE compressed row
    // (kv_lora_rank + qk_rope_dim) per token and has no V side, so `v_row` is 0 and `kv_side_bytes`
    // allocates the placeholder buffer the backends still bind. deepseek32 puts its
    // lightning-indexer cache on that otherwise-unused V side, so `v_row` there is
    // `indexer_head_size` and is reserved like any other side.
    let (k_row, v_row) = kv_row_elems(cfg, l);
    // SWA ring: window layers hold only `window + ubatch` rows (the ring recycles rows the window
    // mask already excludes — `kv_rows_at` has the argument).
    let rows = kv_rows_at(cfg, l, want_ctx, ring, ubatch, row_align);
    // The indexer cache MUST hold the whole context: `Op::LightningIndexer` masks causally only, so
    // position 0 stays eligible for every query row and a ring that had wrapped would have
    // overwritten it — every backend refuses `cap_rows < kv_len` rather than score a row holding
    // some other position. V3.2 declares no sliding window, so `kv_rows_at` already returns
    // `want_ctx` here; this is what keeps that true if the ring gate ever widens, checked where the
    // buffer is actually sized.
    assert!(
        !cfg.deepseek32 || rows >= want_ctx,
        "deepseek32 layer {l}: the lightning indexer's KV cache was sized at {rows} rows for a \
         {want_ctx}-token context — it must not ring, because the indexer's causal-only mask keeps \
         position 0 eligible for every query"
    );
    (
        kv_side_bytes(k_fmt, rows * k_row),
        kv_side_bytes(v_fmt, rows * v_row),
    )
}

/// The smallest KV cache `n_slots` sessions of `want_ctx` tokens can still end up allocating,
/// priced BEFORE the runner settles the details: [`kv_init_allocs`] with each side at the cheapest
/// of the format the fit math expects, f16 and q8_0 (the runner falls back to one of those when it
/// gates a format out), the SWA ring wherever ringing is possible at all, and the shortest prefill
/// chunk placement may lower to. Each of those choices can only shrink the figure, so
/// [`refuse_unallocatable_kv`] reading it stays provable before a single weight is loaded.
///
/// It lists what the runner's init must allocate and nothing more, which is where it differs from
/// the estimate a context is fitted with ([`kv_bytes_estimate_fmt`]): the recurrent-state snapshot
/// is left out. A slot allocates that on first use, a decode loop that is refused it carries on
/// without it, and a slot that never snapshots never allocates it, so counting it here could
/// refuse a run that would have worked. Nor is the allocator's byte rounding here: the guard
/// checks a request's memory requirements, which are never below the request.
pub(crate) fn kv_floor_alloc_bytes(
    cfg: &Config,
    ec: &EngineConfig,
    want_ctx: usize,
    n_slots: usize,
    row_align: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> Vec<(usize, usize)> {
    let cheapest = |dt: DType| {
        [dt, DType::F16, DType::Q8_0]
            .into_iter()
            .min_by_key(|&d| kv_side_bytes(d, 1 << 20))
            .expect("the candidate list is not empty")
    };
    let (k_fmt, v_fmt) = (cheapest(k_fmt), cheapest(v_fmt));
    let ring = kv_ring_wanted(cfg, ec);
    let ubatch = ubatch_candidates(ec)
        .into_iter()
        .min()
        .expect("ubatch_candidates is never empty");
    let one: Vec<(usize, usize)> =
        kv_init_allocs(cfg, want_ctx, ring, ubatch, row_align, k_fmt, v_fmt).collect();
    std::iter::repeat_n(one, n_slots.max(1)).flatten().collect()
}

/// Layer `l`'s [`dsv4_comp_bufs`] as allocation entries in the `(bytes, bytes)` form the KV guard
/// reads ([`refuse_unallocatable_kv`]): one `(bytes, 0)` per buffer, since each is its own
/// allocation.
pub(crate) fn dsv4_comp_allocs(
    cfg: &Config,
    l: usize,
    want_ctx: usize,
    k_fmt: DType,
) -> impl Iterator<Item = (usize, usize)> {
    dsv4_comp_bufs(cfg, l, want_ctx, k_fmt)
        .into_iter()
        .map(|b| (b.bytes(), 0))
}

/// The KV allocations of `sizes` the Vulkan VRAM guard will check, summed — `None` when nothing can
/// be proven: no budget to compare against, or a configuration where the guard is not the last word
/// (`spill`, `Backend::kv_spill_armed`: `INFR_KV_OVERFLOW` places what does not fit in system RAM;
/// `INFR_NO_VRAM_GUARD` turns the guard off).
///
/// A LOWER bound on what the guard will see, never an over-estimate: a sub-MiB request is skipped
/// even though its aligned size may reach the guard, and the guard checks `requirements.size`,
/// which is never below the request.
fn kv_guarded_bytes(
    sizes: &[(usize, usize)],
    room: Option<u64>,
    ec: &EngineConfig,
    spill: bool,
) -> Option<u64> {
    if room.is_none() || spill || ec.kernels.vulkan.no_vram_guard {
        return None;
    }
    let total = sizes
        .iter()
        .flat_map(|&(k, v)| [k as u64, v as u64])
        .filter(|&b| b >= infr_vulkan::VRAM_GUARD_MIN_ALLOC)
        .sum();
    Some(total)
}

/// Refuse rung for a context no rung can save: `Err` when the KV cache buffers `sizes` ALONE
/// cannot be allocated in `room`, the device's allocation ceiling. Two callers: the explicit-`--ctx`
/// path before any weight loads (pricing [`kv_floor_alloc_bytes`] against the whole device), and the
/// runner between the weight upload and the KV allocation (pricing the exact buffers against what
/// the device reports is left). `fit` supplies the window to suggest instead, when one is known.
/// `spill` is whether the cache can spill (`Backend::kv_spill_armed`), which refuses nothing.
///
/// Provable, so it can never refuse a run that would have worked: it counts only the buffers the
/// guard checks ([`kv_guarded_bytes`]), against the guard's own ceiling, with no weights and no
/// activation reserve. If they sum past `room`, the first buffer that crosses it is refused, and
/// the room only shrinks as the earlier ones land. Without this an oversized explicit window failed
/// on a misleading streaming-floor error or on one layer's buffer with the guard's generic message.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refuse_unallocatable_kv(
    ec: &EngineConfig,
    spill: bool,
    sizes: &[(usize, usize)],
    room: Option<u64>,
    want_ctx: usize,
    k_fmt: DType,
    v_fmt: DType,
    room_is: &str,
    fit: impl FnOnce() -> Option<usize>,
) -> AResult<()> {
    let Some(need) = kv_guarded_bytes(sizes, room, ec, spill) else {
        return Ok(());
    };
    let room = room.expect("kv_guarded_bytes is None without a room");
    if need <= room {
        return Ok(());
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    Err(anyhow!(
        "context {want_ctx} cannot be served on this device: its KV cache alone needs at least \
         {:.2} GiB (requested {k_fmt:?}/{v_fmt:?}), and only {:.2} GiB can be allocated {room_is}.\n  \
         Try a smaller window (--ctx N / INFR_CTX=N{}), a more quantized cache (INFR_KV_TYPE_K / \
         INFR_KV_TYPE_V), INFR_KV_OVERFLOW=1 to place the cache in system RAM, or a smaller quant \
         of the model.",
        gib(need),
        gib(room),
        match fit() {
            Some(f) if f > 0 => format!("; about {f} tokens fit here"),
            _ => String::new(),
        },
    ))
}

/// One resident-BDA / streamed-arena addressing unit's ELEMENT-count cap (the invariant's element
/// half; see `infr_vulkan`'s `BdaWeightArena` doc and `BDA_ADDRESSING_UNIT_MAX`). In-kernel element
/// indices are u32, so an addressing unit — one dense tensor, or one per-expert slice of a stacked
/// bank — must stay under 2^32 ELEMENTS. This is the binding limit for sub-byte / low-bpw quants
/// (Q2_K etc.), where 4 Gi elements is only ~1.3 GiB of bytes and so trips well before the 4 GiB
/// BYTE cap `bda_weight_alloc` enforces. Load-time guard: reject LOUDLY here (shape+dtype are known)
/// instead of letting a u32 index wrap into coherent-but-wrong reads in-shader. Enforced today by
/// model reality (no single tensor / expert slice is anywhere near 4 Gi elements); a model that
/// crossed it would need a wider in-kernel addressing scheme, not just a bigger allocation.
const BDA_ELEMENT_UNIT_MAX: usize = 1 << 32;

fn check_bda_element_cap(name: &str, unit: &str, elems: usize) -> AResult<()> {
    if elems as u64 >= BDA_ELEMENT_UNIT_MAX as u64 {
        return Err(anyhow!(
            "weight tensor {name}: {unit} has {elems} elements, at/above the u32 addressing unit \
             cap (2^32) — in-kernel element indices are u32 and would wrap; this needs a wider \
             addressing scheme, not a bigger allocation"
        ));
    }
    Ok(())
}

/// The dense weights whose ONLY GPU consumers are the chunk-covered dispatches (issue #77): the
/// output projection's decode GEMV and `Op::EmbedGather`, both of which split a `>= 2^32`-element
/// tensor into output-row chunks at dispatch time (`infr_vulkan`'s `dispatch_gemv_chunked` /
/// `embed_gather`, u64 per-row base). These may exceed the u32 ELEMENT cap; every OTHER dense
/// tensor keeps the loud whole-tensor `check_bda_element_cap`, and the 4 GiB BYTE cap in
/// `bda_weight_alloc` still bounds the single contiguous allocation for ALL of them (so an
/// over-cap table must be low-bpw enough to fit — a quantized frontier 256k-vocab lm_head).
/// `output.weight` = the untied lm_head; `token_embd.weight` = the input embedding (and the TIED
/// lm_head, read by both `Op::EmbedGather` and the lm_head `Op::Linear`);
/// `per_layer_token_embd.weight` = gemma4-E2B's per-layer gather table.
fn chunk_covered_dense_tensor(name: &str) -> bool {
    matches!(
        name,
        "output.weight" | "token_embd.weight" | "per_layer_token_embd.weight"
    )
}

/// The host memory the arenas of `tiers` take ([`infr_core::hostpager::HostPager::arena_bytes`]).
fn host_arena_bytes<'a>(
    tiers: impl IntoIterator<Item = &'a Option<std::sync::Arc<infr_core::hostpager::HostPager>>>,
) -> u64 {
    tiers
        .into_iter()
        .flatten()
        .map(|t| t.arena_bytes() as u64)
        .sum()
}

/// Decide this model's MoE expert placement, install the pager session when the decision pages
/// (FIRST load only), and return the Vulkan weight binder that implements it. Shared by every
/// Vulkan weight-uploading session — [`generate_dense_vulkan_session`] and the DiffusionGemma
/// session (`model.rs`), which drives `generate_dense_backend` directly and would otherwise
/// silently skip placement (observed: `INFR_CACHE` was a no-op on DG).
///
/// For a non-MoE model (or a warm call — `first_load == false`) this degrades to the plain
/// pad-and-upload resident binder: placement is decided ONCE per weight upload, both because
/// only the first load ever calls the binder, and because the tier-3 budget math is only
/// consistent BEFORE the upload (see the double-count note inside).
pub(crate) fn vulkan_moe_binder<'a>(
    vk: &'a infr_vulkan::VulkanBackend,
    g: &'a Gguf,
    cfg: &'a Config,
    ec: &EngineConfig,
    first_load: bool,
    want_ctx: usize,
) -> AResult<Box<BindWeight<'a>>> {
    // Every height this placement prices and pins is read through the current scope's pins.
    debug_assert_scope_for(vk);
    // ── MoE expert placement ─────────────────────────────────────────────────────────────────
    // The pager (`infr_vulkan::pager`) is the ONLY MoE offload mechanism — the legacy
    // host-visible (HostWeights/GTT) split and its INFR_NCMOE knob are gone. Tiers, in
    // precedence order:
    //   1. `INFR_CACHE=<size>` EXPLICIT override — force EVERY expert layer through the pager
    //      with that byte budget, regardless of whether the banks would fit resident. Lets a
    //      caller (or a test) force the paged path deterministically instead of depending on
    //      this box's free VRAM — see the `gpu_seam_paged_moe_matches_*` tests. The value is the
    //      shared size grammar (`infr_core::parse_size`): plain bytes, `k/m/g/t` 1024-suffixes
    //      (`INFR_CACHE=19g`), or a percentage of the allocatable room at first load
    //      (`INFR_CACHE=80%`; [`paging_cache_budget`]: free VRAM less the guard's headroom).
    //   2. Auto (unset): fully resident (the fast path, zero change) when the banks fit VRAM;
    //      otherwise the pager with budget = remaining VRAM after dense+KV+headroom.
    // Paging rides the adapter's paged executor split (`infr_vulkan::adapter::execute_static`'s
    // paged branch). FUSED gate_up banks (gemma-4 MoE / DiffusionGemma) page under `Role::Gate`
    // with a double-width slot, and MIXED-dtype roles (unsloth-dynamic quants bumping a subset
    // of layers' banks to a wider format) split into per-(role, slot_bytes) arena pools — see
    // `infr_vulkan::pager`'s MoE-session doc.
    let cache_override = ec.paging.cache;
    // Every estimate below prices the cache this backend will allocate, rounding included.
    let align = KvAlign::of(vk);

    let mut n_paged = 0usize; // paged layer-count (0 = fully resident, or all = cfg.n_layer)
    let mut pager_budget_bytes = 0u64;
    // Placement is decided ONCE, on the session's FIRST load — the only call where `bind_weight`
    // runs (see the `state.is_none()` init block in `generate_dense_backend`) and the only moment
    // the tier-3 budget math is consistent: `vram.available` is LIVE (heapBudget − heapUsage), so
    // once this model's weights are resident a recompute would subtract `fp.dense` from an
    // `available` that ALREADY excludes it — double-counting the model against itself and
    // collapsing the budget (observed: a fully-resident 16.4 GB model "re-placed" as 5/30
    // resident on the warm second call of a bench). Warm calls leave `n_paged` at 0; nothing
    // consumes it (no binding, and the pager init below is first_load-gated anyway). A first
    // load racing ANOTHER resident model (swap mid-drain) still reads reduced `available` —
    // that's real pressure, deliberately not compensated; the alloc-time VRAM budget guard is
    // the backstop against over-commit.
    if first_load && cfg.moe.is_some() {
        // NB: the load-time expert-bank dtype gate that used to live here (field report:
        // MXFP4_MOE expert banks `expect`-panicked mid-inference before it existed) is GONE —
        // the id-indexed GEMV floor now covers every dtype a GGUF expert bank can hold (the
        // full dense native set plus F16/F32 for float banks), so
        // `infr_vulkan::linear::moe_expert_dtype_ok` is true for all of them; the invariant is
        // pinned by `moe_expert_floor_covers_dense_set` in infr-vulkan's linear.rs tests.
        match cache_override {
            Some(spec) => {
                n_paged = cfg.n_layer;
                // A percentage resolves against the allocatable room at this (first-load,
                // pre-upload) moment — the same consistent snapshot the auto tier uses below.
                pager_budget_bytes = paging_cache_budget(spec, &vk.vram());
            }
            None => {
                let fp = crate::weights::weight_footprint(g);
                let vram = vk.vram();
                // Per-layer rows: SWA layers ring at window+ubatch rows (see `kv_rows_at`), so a
                // mostly-SWA model's KV prices far below n_layer * ctx.
                let ring = kv_ring_wanted(cfg, ec);
                // Route through the shared q8-aware helper (the dense placement path already does),
                // so a pinned auto-q8 KV cache is priced at ~half the bytes instead of the old
                // hard-coded f16 `*2*2` — which over-reserved ~2× and forced avoidable expert paging.
                let kv_bytes: u64 =
                    kv_bytes_estimate(cfg, want_ctx, ring, ubatch_rows(ec), align, kv_auto_q8());
                // Budgeted against the ALLOCATOR's ceiling (`VramInfo::alloc_room`), like the
                // dense sweeps below and the context-fit math: the last 256 MiB of "free" VRAM is
                // the guard's headroom and can never be handed out, so an expert budget sized
                // against the raw figure hands the pager 256 MiB of arena the allocator refuses.
                // `moe_expert_budget` also holds back `MOE_ACT_HEADROOM` for the pager's own
                // arenas/staging and the activation scratch neither `fp` nor `kv_bytes` counts.
                //
                // Mixed oversize (an MoE model whose DENSE part alone doesn't fit) is out of
                // scope for dense layer streaming — `None`, and we fail with a clear message
                // instead of letting a degenerate expert budget stumble into the alloc-time VRAM
                // guard's generic over-commit error.
                let Some(budget) = moe_expert_budget(&vram, fp.dense, kv_bytes) else {
                    return Err(anyhow!(
                        "this MoE model's dense weights ({:.2} GB) + KV cache ({:.2} GB) exceed \
                         the allocatable VRAM ({:.2} GB, free minus the guard headroom) — dense \
                         layer streaming does not cover MoE models' dense parts; reduce ctx or \
                         run on the CPU backend (INFR_DEV=cpu)",
                        fp.dense as f64 / 1e9,
                        kv_bytes as f64 / 1e9,
                        vram.alloc_room() as f64 / 1e9,
                    ));
                };
                if budget < fp.expert {
                    // Page EVERY expert layer with the WHOLE budget (tier-2 semantics), NOT
                    // "keep the first gpu_layers banks resident and page the overflow with
                    // the leftover": the leftover degenerates to a few slots and the paged
                    // layers thrash at a 0% hit rate (measured, Scout Q2_K on 24GB:
                    // resident-26-layers + 3-slot pager decoded at 3.3 t/s; all-paged with
                    // the same total VRAM = 307 slots/role, 66% hits, 6.2 t/s). A shared LRU
                    // arena keeps the hot experts of EVERY layer resident — strictly more
                    // flexible than pinning whole layer banks.
                    n_paged = cfg.n_layer;
                    pager_budget_bytes = budget;
                }
            }
        }
    }
    // A FUSED gate_up bank pages under `Role::Gate` (one double-width slot per expert; the model
    // then has no `Role::Up` sources at all) — see `infr_vulkan::pager`'s MoE-session doc.
    let moe_role_of = |name: &str| -> Option<infr_vulkan::pager::Role> {
        use infr_vulkan::pager::Role;
        if name.ends_with("ffn_gate_exps.weight") || name.ends_with("ffn_gate_up_exps.weight") {
            Some(Role::Gate)
        } else if name.ends_with("ffn_up_exps.weight") {
            Some(Role::Up)
        } else if name.ends_with("ffn_down_exps.weight") {
            Some(Role::Down)
        } else {
            None
        }
    };
    // The session must exist (and answer `Backend::moe_paged` truthy) BEFORE `generate_dense_backend`
    // below uploads a single weight: the FIRST paged tensor the `bind_weight` closure sees still
    // has to bind a placeholder the adapter recognizes as paged the very first time a graph
    // executes — sizing (and installing) it AFTER the call, once every weight was already bound
    // to a 4-byte placeholder nobody registered, would leave `execute_static` reading that
    // placeholder as if it were the full bank (see `pager::MoePagerLayout`'s doc). Only on the
    // FIRST load of a session (`bind_weight` isn't called again once `state` already holds
    // uploaded weights, so a second `init_moe_pager` would wipe an already-warm cache for nothing;
    // `n_paged > 0` already implies `first_load` — the placement calc above is first_load-gated —
    // but keep the guard explicit).
    // The MoE twin, keyed by the `(role, per-expert bytes)` pair that identifies an expert pool —
    // the binder receives a tensor, not a pool index, and re-derives that key the same way
    // `MoePagerSession::register` does.
    let mut moe_host: Vec<(
        infr_vulkan::pager::Role,
        usize,
        Option<std::sync::Arc<infr_core::hostpager::HostPager>>,
    )> = Vec::new();
    if first_load && n_paged > 0 {
        use infr_vulkan::pager::Role;
        let moe = cfg.moe.as_ref().expect("n_paged > 0 implies MoE");
        let n_expert = moe.n_expert.max(1);
        // Enumerate every paged `_exps` weight bank's (role, per-expert bytes): one arena POOL
        // per distinct pair. A uniform split-bank model (Scout) yields the classic three pools;
        // a fused-bank model (gemma-4 MoE / DiffusionGemma) yields a double-width Gate pool and
        // no Up pool; a mixed-dtype role (UD quants) yields one pool per byte size, each holding
        // exactly the layers whose banks match it (`MoePagerSession::register` re-derives the
        // same key from the tensor bytes). `(slot_bytes, blocks-in-pool)` per pool.
        let mut pool_blocks: Vec<(Role, usize, usize)> = Vec::new();
        for t in g.tensors() {
            if exps_layer(&t.name).is_none_or(|l| l >= n_paged) {
                continue; // not a paged layer's `_exps` tensor
            }
            let Some(role) = moe_role_of(&t.name) else {
                continue; // `_exps` but not a weight bank (e.g. the per-expert `.scale` vector)
            };
            let sb = (t.nbytes / n_expert).max(4);
            match pool_blocks
                .iter_mut()
                .find(|(r, s, _)| *r == role && *s == sb)
            {
                Some((_, _, n)) => *n += n_expert,
                None => pool_blocks.push((role, sb, n_expert)),
            }
        }
        if pool_blocks.is_empty() {
            // Defensive: an MoE config with NO pageable `_exps` weight banks (no arch this crate
            // loads ships that). Nothing to page — stay fully resident and let the alloc-time
            // VRAM budget guard produce its clear error if that overflows. `n_paged = 0` also
            // turns the binder's paged divert below into a no-op (it re-checks `n_paged`).
            tracing::warn!(
                "MoE pager: no pageable `_exps` weight banks found — keeping every expert \
                 resident (the VRAM budget guard is the backstop)"
            );
            n_paged = 0;
        }
        if n_paged > 0 {
            let n_blocks = n_paged * n_expert;
            // The session's pinned upload ring (two fence-rotated halves — see
            // `MoePagerSession`'s `ring` doc) lives in the same VRAM the arenas do: subtract it
            // from the budget BEFORE splitting arena shares so the paged footprint stays within
            // what the caller granted (INFR_CACHE) / what the auto tier measured as free.
            let ring_bytes = infr_core::pager::ring_bytes(pager_budget_bytes, vk.cfg().paging.ring);
            pager_budget_bytes = pager_budget_bytes.saturating_sub(ring_bytes as u64);
            let total_bytes: u64 = pool_blocks
                .iter()
                .map(|&(_, sb, nb)| (sb * nb) as u64)
                .sum::<u64>()
                .max(1);
            // No per-pool arena ceiling: each MoE pool is a `bufferDeviceAddress` buffer read by
            // pointer, so it is NOT capped by one SSBO binding's maxStorageBufferRange (~4 GiB on
            // RADV) the way it was when the arena was a bound SSBO. A pool now holds as many experts
            // as its budget share allows — the whole point of the u64 addressing lift. The only
            // backstop is the alloc-time VRAM budget guard (`GpuPager::new` -> `alloc_arena_bda`);
            // the proportional split below never over-subscribes VRAM because it partitions
            // `pager_budget_bytes`, which the caller derived from the remaining VRAM.
            // The tier below VRAM, one host pager per expert pool, sized before the pools so a
            // `Host` bank can name it. Pool order is preserved, so pool `i`'s tier is `moe_host[i]`.
            let classes: Vec<(usize, usize)> =
                pool_blocks.iter().map(|&(_, sb, nb)| (sb, nb)).collect();
            let host_tier =
                vulkan_host_tier(ec, g, "MoE", &classes, vk.capabilities().unified_memory)?;
            moe_host = pool_blocks
                .iter()
                .zip(&host_tier)
                .map(|(&(role, sb, _), h)| (role, sb, h.clone()))
                .collect();
            let pools: Vec<infr_vulkan::pager::MoePoolSpec> = pool_blocks
                .iter()
                .zip(&host_tier)
                .map(|(&(role, sb, nb), host)| {
                    // Budget split PROPORTIONALLY to each pool's total bank bytes — the byte share is
                    // also the access share under uniform routing (every (layer, expert) read touches
                    // gate+up+down alike), so proportional slots equalize expected hit rates across
                    // pools; any fancier split would need routing statistics that don't exist at
                    // load time.
                    let share = (pager_budget_bytes as u128 * (sb * nb) as u128
                        / total_bytes as u128) as u64;
                    // Floor at `min(n_expert, nb)`: a chunked batched-prefill `Op::MoeFfn` (rows>1)
                    // runs ALL of a layer's routed buckets in ONE dispatch
                    // (`matmul_mmq_experts_paged`), touching up to `n_expert` DISTINCT experts of
                    // that layer that must be simultaneously resident (the within-batch safety
                    // invariant — see `infr_core::pager::Pager::new`'s doc). Decode's rows=1 needs
                    // only `n_used`, but the batched bound subsumes it and `n_expert` slots is tiny
                    // next to any real budget (Scout: 16 x ~18 MB per role). Capped at `nb` (no
                    // point holding more slots than the pool has distinct experts).
                    let floor = n_expert.min(nb).max(1);
                    let budget_slots = ((share / sb as u64) as usize).clamp(floor, nb);
                    infr_vulkan::pager::MoePoolSpec {
                        role,
                        slot_bytes: sb,
                        n_slots: budget_slots,
                        host: host.clone(),
                    }
                })
                .collect();
            let cached: usize = pools.iter().map(|p| p.n_slots).sum();
            let pool_desc: Vec<String> = pool_blocks
                .iter()
                .zip(&pools)
                .map(|(&(role, sb, nb), p)| {
                    format!("{role:?}[{:.1}MB] {}/{}", sb as f64 / 1e6, p.n_slots, nb)
                })
                .collect();
            tracing::info!(
                "MoE pager: {n_paged}/{} expert layers PAGED ({cached} expert blocks cached — {}; \
             {:.2} GB budget; ctx={want_ctx})",
                cfg.n_layer,
                pool_desc.join(", "),
                pager_budget_bytes as f64 / 1e9,
            );
            vk.init_moe_pager(infr_vulkan::pager::MoePagerLayout {
                n_blocks,
                pools,
                ring_bytes,
            })
            .map_err(|e| anyhow!("{e}"))?;
        }
    }

    // ── Dense layer streaming placement ──────────────────────────────────────────────────────
    // The DENSE twin of the MoE tiers above (`infr_vulkan::pager::DensePagerSession`): when a
    // dense model's per-layer weights (minus what must stay resident) exceed the budget, stream
    // them through per-(dtype, stride) arena pools driven by the exact cyclic-sweep policy
    // (`infr_core::pager::Pager::schedule`). One block = one weight GROUP exactly as `wload`
    // uploads it (fused qkv / gate_up concats are one block — the shared `fuse_*_decision`
    // helpers keep this enumeration and the runner's upload order from drifting). Embeddings,
    // lm_head, norms and biases stay resident: norms/biases are consumed by ops without weight
    // offsets and are tiny; token_embd/lm_head are read at every token edge, so streaming them
    // adds their full bytes to every token's PCIe bill with zero locality to exploit.
    //
    //   1. `INFR_CACHE=<size>` on a DENSE model — force EVERY streamable block through the
    //      streamer with that byte budget (deterministic test hook, same grammar as the MoE tier).
    //   2. Auto (unset): TRY RESIDENT FIRST — fully resident (the fast path, zero change) when
    //      weights + KV + the honest dense activation reserve (`dense_act_reserve_at`) fit live
    //      VRAM; otherwise stream with budget = remaining VRAM after resident-weights+KV+reserve.
    //      An explicit oversized INFR_CTX whose KV can't sit beside resident weights falls back
    //      to streaming the same way (never clamped here — the ctx the caller asked for is kept).
    // Streamable = the per-layer Linear projection groups whose dtype the streamed dispatch reads
    // from an arena slot (`dense_stream_supported`: every native format, plus F16/F32, which run
    // the resident float kernels at the slot's address) and whose bytes upload unmodified from the
    // mmap (the qwen2 NEOX q/k row permute rewrites bytes at load, so those tensors stay resident).
    let mut dense_plan: std::collections::HashMap<String, (usize, u32, Vec<String>)> =
        std::collections::HashMap::new();
    // The tier BELOW VRAM, one entry per dense pool (empty = none: every miss reads the mmap).
    let mut dense_host: Vec<Option<std::sync::Arc<infr_core::hostpager::HostPager>>> = Vec::new();
    if first_load && cfg.moe.is_none() {
        let fuse_gu = runner::fuse_gu_decision(vk.capabilities().combined_gu, g, cfg);
        let fuse_qkv = runner::fuse_qkv_decision(vk.capabilities().combined_gu, g, cfg, ec);
        // Candidate groups in LAYER ORDER — the cyclic-sweep schedule key. Key = names[0] (what
        // `bind_weight` receives for the group).
        let mut groups: Vec<Vec<String>> = Vec::new();
        for l in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            if !cfg.permute_qk_neox {
                if fuse_qkv {
                    groups.push(vec![
                        p("attn_q.weight"),
                        p("attn_k.weight"),
                        p("attn_v.weight"),
                    ]);
                } else {
                    groups.push(vec![p("attn_q.weight")]);
                    groups.push(vec![p("attn_k.weight")]);
                    groups.push(vec![p("attn_v.weight")]);
                }
            } else if !fuse_qkv {
                // Permuted q/k stay resident (their upload bytes are load-time rewrites of the
                // mmap); v uploads raw and can still stream.
                groups.push(vec![p("attn_v.weight")]);
            }
            groups.push(vec![p("attn_output.weight")]);
            if fuse_gu {
                groups.push(vec![p("ffn_gate.weight"), p("ffn_up.weight")]);
            } else {
                groups.push(vec![p("ffn_gate.weight")]);
                groups.push(vec![p("ffn_up.weight")]);
            }
            groups.push(vec![p("ffn_down.weight")]);
        }
        let tinfo = |n: &str| g.tensors().iter().find(|t| t.name == n);
        // Eligible groups with their (dtype, raw bytes, numel) — a group whose tensors are
        // missing (DeltaNet layers, gemma4 V-less layers) or whose dtype has no arena-addressed
        // dispatch simply stays resident.
        let eligible: Vec<(Vec<String>, infr_core::DType, usize, usize)> = groups
            .into_iter()
            .filter_map(|comps| {
                let infos: Vec<_> = comps.iter().map(|n| tinfo(n)).collect::<Option<_>>()?;
                let dt = infos[0].dtype;
                if !infos.iter().all(|t| t.dtype == dt)
                    || !infr_vulkan::linear::dense_stream_supported(dt)
                {
                    return None;
                }
                let raw: usize = infos.iter().map(|t| t.nbytes).sum();
                let numel: usize = infos
                    .iter()
                    .map(|t| t.shape.iter().product::<usize>())
                    .sum();
                Some((comps, dt, raw, numel))
            })
            .collect();
        let streamable_resident: u64 = eligible
            .iter()
            .map(|(_, dt, raw, numel)| crate::weights::tensor_resident_bytes(*dt, *numel, *raw))
            .sum();
        let fp = crate::weights::weight_footprint(g);
        let vram = vk.vram();
        // The per-side KV formats a chunk/format candidate prices: `q8` = BOTH sides Q8_0 (34
        // bytes / 32 elems), false = f16 (2 B/elem). Per-layer rows ring at window+ubatch rows
        // (see `kv_rows_at`) — what lets a mostly-SWA model (gemma-4-31B: 50/60 layers SWA) price its
        // KV small enough to take the try-resident tier at real contexts instead of streaming.
        let kv_fmts = |q8: bool| {
            let side = if q8 { DType::Q8_0 } else { DType::F16 };
            (side, side)
        };
        // Does weights + KV + the honest activation reserve fit at this (chunk, fmt)? Through the
        // SHARED predicate, so this decision and `kv_fit_ctx_for`'s price the same session against
        // the same ceiling — the ALLOCATOR's (`VramInfo::alloc_room`), not the raw free figure.
        // Against `vram.available` this could declare a model resident while planning 256 MiB the
        // VRAM guard will refuse, and the two ladders picked different rungs (gemma-3-12b @131072:
        // the fit math validated the context at 256 rows while this went resident at 512).
        let fits = |ubatch: usize, q8: bool| {
            let (k, v) = kv_fmts(q8);
            dense_placement_fits(cfg, ec, fp.total(), &vram, want_ctx, ubatch, align, k, v)
        };
        // Try-resident-first: a dense model goes FULLY RESIDENT (the exact pre-streaming fast
        // path) whenever weights + this session's KV + an HONEST dense activation estimate fit
        // the allocatable VRAM; only a genuine miss streams. The MoE tier's 2 GiB ACT_HEADROOM is
        // sized for pager arenas/staging that a dense-resident session doesn't have — reusing it
        // here streamed gemma-4-31B (21.9 GB weights on a 24 GB card, decode 33 t/s resident vs
        // ~3 t/s streamed at the PCIe ceiling). If residency is chosen but a later activation
        // alloc still misses (fragmentation, another process grabbing VRAM), the alloc-time VRAM
        // guard fails that request cleanly — INFR_CACHE=<size> is the escape hatch that forces
        // streaming. `kv_auto_q8()` may already be pinned by the default-ctx clamp path (see
        // `SeamModel::clamp_default_ctx`) — then every check here prices the q8 cache the runner
        // will actually allocate.
        //
        // Residency sweep: when the FULL-chunk reserve is what tips a big model into streaming,
        // try smaller prefill chunks — a smaller chunk shrinks BOTH the activation reserve
        // (whole-chunk logits/gate_up scratch scale with rows) and the SWA ring rows
        // (window + chunk). Resident-with-a-512-row-chunk decodes ~10x faster than streaming at
        // the PCIe ceiling (gemma-4-31B @ d4096: 27.6 vs 2.9 t/s), so trading prefill chunk
        // height for residency is strictly the right call. Pinned per-session (`PlacementPins`)
        // so the prefill loop and the runner's ring sizing use exactly the priced height; an
        // explicit INFR_UBATCH disables the sweep (the user's height is authoritative). Runs
        // BEFORE the auto-q8 rung below: a shorter prefill chunk costs only some prefill
        // throughput, while q8 KV costs ~10-16% GQA decode — prefer the cheaper concession.
        let mut resident = fits(ubatch_rows(ec), kv_auto_q8());
        if !resident && !user_pinned_ubatch(ec) && cache_override.is_none() {
            // `dense_resident_rung` walks the shared ladder — the SAME walk `kv_fit_ctx_for` makes
            // to decide a context fits, so the rung it lands on here is the rung that math priced.
            // Its first entry is the height `resident` above already priced and rejected.
            let (k, v) = kv_fmts(kv_auto_q8());
            if let Some(cand) =
                dense_resident_rung(cfg, ec, fp.total(), &vram, want_ctx, align, k, v)
            {
                pin_ubatch(cand);
                // Re-read through the pin (a racing earlier set wins — use whatever stuck).
                if fits(ubatch_rows(ec), kv_auto_q8()) {
                    tracing::warn!(
                        "dense placement: resident with a {}-row prefill chunk (the default \
                         1024-row chunk's activation reserve wouldn't fit); set INFR_UBATCH \
                         to override",
                        ubatch_rows(ec).min(want_ctx),
                    );
                    resident = true;
                }
            }
        }
        // ── auto-q8 KV rung (the placement-ladder step between the SWA ring and streaming):
        // f16 KV missed residency at every chunk height, but a Q8_0 cache (roughly HALF the KV
        // bytes) might fit — placing RESIDENT with q8 KV beats the remaining rungs by an order
        // of magnitude (streaming decodes at the PCIe ceiling; the explicit-ctx path never
        // clamps). Gates: the user set NO KV format (see `PlacementPins`'s policy doc — explicit
        // settings always win, both sides go q8_0, never below q8), no INFR_CACHE override
        // (that's the deterministic force-streaming hook), and the runner's own q8 layout gate
        // (32-elem block alignment; this binder is the Vulkan path, a native q8-KV backend —
        // decode reads Q8 natively and coupled K==V==Q8 keeps record-once replay; batched
        // prefill reads it through the dequant prepass, which the SWA ring kept for q8).
        // Tries the pinned/current chunk height first, then the same smaller-chunk ladder as
        // the f16 sweep (floor 128).
        if !resident
            && cache_override.is_none()
            && !kv_auto_q8()
            && kv_unset(cfg, ec)
            && kv_q8_layout_ok(cfg)
        {
            let (k, v) = kv_fmts(true);
            if let Some(cand) =
                dense_resident_rung(cfg, ec, fp.total(), &vram, want_ctx, align, k, v)
            {
                pin_kv_auto_q8();
                if cand != ubatch_rows(ec) {
                    pin_ubatch(cand);
                }
                // Re-read through the pins (racing earlier sets win — use whatever stuck).
                if kv_auto_q8() && fits(ubatch_rows(ec), true) {
                    tracing::warn!(
                        requested_ctx = want_ctx,
                        prefill_chunk = ubatch_rows(ec),
                        kv_dtype = "q8_0",
                        "kv auto-quant: q8_0 KV cache — an f16 cache would not fit resident \
                         at ctx={want_ctx} at any prefill chunk height; set \
                         INFR_KV_TYPE_K/V=f16 to force f16 (decode is ~10-16% slower on q8)"
                    );
                    resident = true;
                }
            }
        }
        // ── spill rung (`kv.overflow`, B62): no cache format fits beside the weights at any chunk
        // height, but the weights fit without it, with the spill reserve (`kv_spill_reserve`: the
        // activation reserve, the driver's later allocations, the user's margin) left free at the
        // spilled session's chunk (`spill_ubatch`). Keep them resident and let the cache take what
        // device memory is left, the rest in system RAM (`kv_spill_split`): reading the spilled
        // rows costs PCIe in proportion to the depth reached, where streaming re-reads every
        // streamed weight each token. The weights are priced at the least the arena commits for
        // them and the upload staging (`resident_load_held`), as the fit before the load prices
        // them; what first-fit strands past that bound and the driver's own memory are not, so a
        // session this keeps resident can still be refused by the re-clamp's measured check.
        let spill = vk.kv_spill_armed() && cache_override.is_none();
        if !resident && spill {
            let (held, staging) = resident_load_held(vk, g, cfg, ec);
            let room = vram
                .alloc_room()
                .saturating_sub(held.saturating_add(staging));
            let ubatch = spill_ubatch(ec);
            let reserve = kv_spill_reserve(cfg, ec, want_ctx, ubatch, false);
            if room >= reserve {
                repin_ubatch_lower(ubatch);
                let (k, v) = kv_fit_fmts(cfg, ec, kv_auto_q8());
                let ring = placement_ring(cfg, ec, k, v);
                let split = kv_spill_split(
                    cfg,
                    want_ctx,
                    ring,
                    ubatch,
                    align,
                    k,
                    v,
                    1,
                    room,
                    reserve,
                    kv_spill_cap(ec),
                );
                let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                tracing::warn!(
                    requested_ctx = want_ctx,
                    kv_vram_bytes = split.vram,
                    kv_host_bytes = split.host,
                    "dense placement: weights resident, KV cache spilled — no {k:?}/{v:?} cache of \
                     {want_ctx} tokens fits beside the weights, so {:.2} GiB of it is placed in \
                     VRAM and {:.2} GiB in system RAM (INFR_KV_OVERFLOW), at a {ubatch}-row chunk",
                    gib(split.vram),
                    gib(split.host),
                );
                resident = true;
            }
        }
        // Slot stride (`dense_stream_slot_stride`): the group's raw bytes padded to a whole
        // number of quant blocks AND to the dtype's slot alignment (u32 words for the native
        // kernels, the resident arena's placement for the float ones), so every slot base is
        // block-aligned (the kernels' element-offset weight addressing needs `slot_byte_base =
        // whole blocks`) and the arena reads as `array<u32>`. Hoisted above the budget decision
        // so the streaming-chunk sweep below can price the full-arena need with the exact
        // strides the pools use.
        let stride_of = infr_vulkan::linear::dense_stream_slot_stride;
        // Pools keyed by (dtype, stride); blocks assigned ids in layer order per pool.
        let mut pools: Vec<(infr_core::DType, usize, usize)> = Vec::new(); // (dt, stride, n_blocks)
        let mut planned: Vec<(Vec<String>, usize, u32)> = Vec::new(); // (comps, pool, block_id)
        for (comps, dt, raw, _numel) in &eligible {
            let stride = stride_of(*dt, *raw);
            let pool = match pools.iter().position(|&(d, s, _)| d == *dt && s == stride) {
                Some(i) => i,
                None => {
                    pools.push((*dt, stride, 0));
                    pools.len() - 1
                }
            };
            let block_id = pools[pool].2 as u32;
            pools[pool].2 += 1;
            planned.push((comps.clone(), pool, block_id));
        }
        // The pools' slot floor (`DensePoolSpec`s below): the least arena streaming can run in.
        let floor_slots = |nb: usize| 2.min(nb).max(1);
        let stream_floor: u64 = pools
            .iter()
            .map(|&(_, s, nb)| (floor_slots(nb) * s) as u64)
            .sum();
        // Whether a streaming budget holds the floor once its own upload ring is out of it.
        let holds_floor = |b: u64| {
            b.saturating_sub(infr_core::pager::ring_bytes(b, vk.cfg().paging.ring) as u64)
                >= stream_floor
        };
        // The weights that stay resident beside the streamed ones, as the arena commits them at
        // the least (`weight_arena_floor` over their `weight_upload_units_of`, as the fit before
        // the load prices the whole set), for the budget a wholly spilled cache leaves: that cache
        // gives the run nothing back, so the room the spill reserve needs must really be there
        // after the load, and the bare footprint left it short by the arena's block rounding
        // (Qwen3-1.7B, RX 7900 XTX: about 52 MB). Priced once, and only where the spill is armed.
        let resident_held = spill.then(|| {
            let streamed: std::collections::HashSet<&str> = eligible
                .iter()
                .flat_map(|(comps, ..)| comps.iter().map(String::as_str))
                .collect();
            let units = weight_upload_units_of(g, cfg, ec, vk.capabilities().combined_gu, |n| {
                !streamed.contains(n)
            });
            weight_arena_floor(&units, vk.weight_load_cost()).max(fp.total() - streamable_resident)
        });
        // Set when the streaming budget below is the one a wholly spilled cache leaves.
        let mut stream_spills = false;
        let budget = match cache_override {
            Some(spec) => Some(paging_cache_budget(spec, &vram)),
            None if resident => None,
            None => {
                // Streaming is inevitable. Edge-aware chunk sweep — the STREAMING twin of the
                // residency sweep above: a smaller prefill chunk shrinks the activation reserve
                // and the SWA ring rows, and every byte freed is a byte of streaming budget →
                // more resident slots, fewer PCIe refetches per weight sweep. But a taller
                // chunk prefills faster (fewer whole-model weight sweeps per prompt), so don't
                // shrink past the point of gain: pick the TALLEST chunk whose budget already
                // holds EVERY streamable block resident (extra budget past that buys nothing);
                // if no chunk reaches that, take the floor — 128 rows, the maximum-budget
                // choice. An explicit INFR_UBATCH is authoritative and skips the sweep; the
                // INFR_CACHE tier above is untouched (its budget is the caller's, not derived
                // from the reserve). Pinned via `PlacementPins` like the residency sweep, so the
                // prefill loop, the runner's ring sizing, and this budget all agree.
                let q8 = kv_auto_q8();
                let base = fp.total() - streamable_resident;
                // Same ceiling as `fits` above (`VramInfo::alloc_room`): every byte this
                // over-states is an arena slot the allocator then refuses.
                let budget_at = |ub: usize| {
                    let (k, v) = kv_fmts(q8);
                    dense_stream_budget_at(cfg, ec, base, &vram, want_ctx, ub, align, k, v)
                };
                if !user_pinned_ubatch(ec) && !eligible.is_empty() {
                    let need: u64 = eligible
                        .iter()
                        .map(|(_, dt, raw, _)| stride_of(*dt, *raw) as u64)
                        .sum();
                    // "Covers": the budget minus its own upload-ring share holds every block.
                    let covers = |b: u64| {
                        b.saturating_sub(
                            infr_core::pager::ring_bytes(b, vk.cfg().paging.ring) as u64
                        ) >= need
                    };
                    let ub_now = ubatch_rows(ec);
                    let cands = ubatch_candidates(ec);
                    let pick = cands
                        .iter()
                        .copied()
                        .find(|&c| covers(budget_at(c)))
                        .unwrap_or(*cands.last().expect("cands is never empty"));
                    if pick != ub_now {
                        pin_ubatch(pick);
                    }
                }
                let budget = budget_at(ubatch_rows(ec));
                // `kv.overflow`: when the cache leaves the arenas less than their floor, stream
                // with the whole cache in system RAM instead — the spill comes after the stream.
                let spilled = resident_held.map(|held| {
                    dense_stream_spill_budget_at(
                        cfg,
                        ec,
                        held,
                        vk.weight_load_cost().staging,
                        &vram,
                        want_ctx,
                        spill_ubatch(ec),
                    )
                });
                if let Some(spilled) = spilled.filter(|&b| !holds_floor(budget) && holds_floor(b)) {
                    repin_ubatch_lower(spill_ubatch(ec));
                    tracing::warn!(
                        requested_ctx = want_ctx,
                        "dense placement: weights streamed, KV cache spilled — the weights do not \
                         fit resident and the cache leaves the streaming arenas less than their \
                         floor, so the whole cache is placed in system RAM (INFR_KV_OVERFLOW)"
                    );
                    stream_spills = true;
                    Some(spilled)
                } else {
                    Some(budget)
                }
            }
        };
        if let (Some(mut budget), false) = (budget, eligible.is_empty()) {
            // The pinned upload ring lives in the same VRAM the arenas do — subtract it first.
            let ring_bytes = infr_core::pager::ring_bytes(budget, vk.cfg().paging.ring);
            budget = budget.saturating_sub(ring_bytes as u64);
            let total_bytes: u64 = pools
                .iter()
                .map(|&(_, s, nb)| (s * nb) as u64)
                .sum::<u64>()
                .max(1);
            // The tier below VRAM, sized before the pools so a `Host` source can name it. Pool
            // order is preserved, so pool `i`'s host tier is `dense_host[i]`.
            let classes: Vec<(usize, usize)> = pools.iter().map(|&(_, s, nb)| (s, nb)).collect();
            dense_host =
                vulkan_host_tier(ec, g, "dense", &classes, vk.capabilities().unified_memory)?;
            if stream_spills {
                // The whole cache goes to system RAM beside the host tier just sized from the
                // same memory: refused here, before a weight is uploaded, when it does not fit.
                let (k, v) = kv_fit_fmts(cfg, ec, kv_auto_q8());
                let ub = ubatch_rows(ec);
                let ring = placement_ring(cfg, ec, k, v);
                let split = kv_spill_split(
                    cfg,
                    want_ctx,
                    ring,
                    ub,
                    align,
                    k,
                    v,
                    1,
                    0,
                    0,
                    kv_spill_cap(ec),
                );
                refuse_unhostable_kv(
                    want_ctx,
                    split,
                    infr_core::hostmem::available_bytes(),
                    host_arena_bytes(&dense_host),
                    k,
                    v,
                )?;
            }
            let specs: Vec<infr_vulkan::pager::DensePoolSpec> = pools
                .iter()
                .enumerate()
                .map(|(i, &(dtype, stride, nb))| {
                    // Proportional budget split (byte share == access share: every block is read
                    // exactly once per sweep). Floor 2 slots so the next block's upload can
                    // overlap the previous block's dispatch instead of serializing on one slot.
                    // `n_slots` is bounded ONLY by the VRAM budget share and this floor: the pool
                    // arena is a `bufferDeviceAddress` buffer read purely by 64-bit pointer (see
                    // `DensePagerSession`), NEVER bound as a descriptor, so BOTH pre-BDA caps are
                    // gone — no `maxStorageBufferRange` binding ceiling and no u32 element-reach
                    // limit. A single pool may span well past 4 GiB (matching the paged-MoE arena,
                    // e.g. Scout's 6.57 GB role pools).
                    let share =
                        (budget as u128 * (stride * nb) as u128 / total_bytes as u128) as u64;
                    let budget_slots =
                        ((share / stride as u64) as usize).clamp(floor_slots(nb), nb);
                    infr_vulkan::pager::DensePoolSpec {
                        dtype,
                        slot_bytes: stride,
                        n_slots: budget_slots,
                        n_blocks: nb,
                        host: dense_host[i].clone(),
                    }
                })
                .collect();
            let alloc: u64 = specs
                .iter()
                .map(|s| (s.n_slots * s.slot_bytes) as u64)
                .sum();
            if alloc > budget.max(1) && cache_override.is_none() {
                // Auto tier only: the floors overran what's actually free — streaming can't help.
                // Name the KV cache: it is what a too-big context fills the device with, and a
                // message about the weights alone sends the user after the wrong thing.
                let (k, v) = kv_fmts(kv_auto_q8());
                let ub = ubatch_rows(ec);
                let kv = kv_bytes_estimate_fmt(
                    cfg,
                    want_ctx,
                    placement_ring(cfg, ec, k, v),
                    ub,
                    align,
                    k,
                    v,
                );
                if let Some(held) = resident_held {
                    let spilled = dense_stream_spill_budget_at(
                        cfg,
                        ec,
                        held,
                        vk.weight_load_cost().staging,
                        &vram,
                        want_ctx,
                        spill_ubatch(ec),
                    );
                    let spilled = spilled
                        .saturating_sub(
                            infr_core::pager::ring_bytes(spilled, vk.cfg().paging.ring) as u64,
                        );
                    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                    return Err(anyhow!(
                        "context {want_ctx} cannot be served on this device: even with its whole \
                         KV cache ({:.2} GiB in {k:?}/{v:?}) in system RAM (INFR_KV_OVERFLOW), \
                         the activation reserve and the driver's own later allocations leave \
                         {:.2} GiB for streaming the {:.2} GiB of dense weights, below the \
                         streaming floor ({:.2} GiB) — use a smaller quant of the model, free \
                         device memory, or run on the CPU backend (INFR_DEV=cpu)",
                        gib(kv),
                        gib(spilled),
                        gib(fp.total()),
                        gib(stream_floor),
                    ));
                }
                return Err(anyhow!(
                    "context {want_ctx} does not fit this device: after its KV cache ({:.2} GB in \
                     {k:?}/{v:?}) and the activation reserve, {:.2} GB is left for streaming the \
                     {:.2} GB of dense weights, below the streaming floor ({:.2} GB) — reduce ctx \
                     (--ctx N / INFR_CTX=N), quantize the cache (INFR_KV_TYPE_K/V=q8_0), or run \
                     on the CPU backend (INFR_DEV=cpu)",
                    kv as f64 / 1e9,
                    budget as f64 / 1e9,
                    fp.total() as f64 / 1e9,
                    alloc as f64 / 1e9,
                ));
            }
            let cached: usize = specs.iter().map(|s| s.n_slots).sum();
            let n_blocks: usize = specs.iter().map(|s| s.n_blocks).sum();
            tracing::info!(
                "dense streaming: {n_blocks} weight blocks across {} pools, {cached} slots \
                 cached ({:.2} GB arena + {:.2} GB ring; budget {:.2} GB; ctx={want_ctx}; \
                 chunk={})",
                specs.len(),
                alloc as f64 / 1e9,
                ring_bytes as f64 / 1e9,
                (budget + ring_bytes as u64) as f64 / 1e9,
                ubatch_rows(ec).min(want_ctx),
            );
            vk.init_dense_pager(infr_vulkan::pager::DensePagerLayout {
                pools: specs,
                ring_bytes,
            })
            .map_err(|e| anyhow!("{e}"))?;
            for (comps, pool, block_id) in planned {
                dense_plan.insert(comps[0].clone(), (pool, block_id, comps));
            }
        }
    }

    // What the session's host tiers take of host memory, for the post-load re-clamp's spill check
    // (their arenas fill on first touch, so the live host probe does not see them yet).
    if first_load {
        pin_host_tier_bytes(
            host_arena_bytes(&dense_host) + host_arena_bytes(moe_host.iter().map(|(_, _, h)| h)),
        );
    }

    // A resident weight's mapped pages are released once its bytes are in VRAM. Not on a
    // unified-memory device: there the "VRAM" is host RAM, and nothing has been measured to say
    // releasing the source pages is worth a refault there.
    let releaser =
        (ec.paging.release_uploaded && !vk.capabilities().unified_memory).then(PageReleaser::new);

    Ok(Box::new(move |name, tb, dt, numel| {
        // Raw upload for EVERY dtype — the file's bytes go straight to VRAM (u32-padded) and the
        // kernel reads/dequants the native dtype in-shader. F16 → f16 coopmat GEMM / f16 GEMV;
        // F32 stays native (rmsnorm/qk_norm_rope read f32); bf16 → in-shader expand (bf16 is the
        // top 16 bits of an f32, EXACT; the warp GEMM narrows to f16 for the matrix cores like
        // every other format); quant weights → raw blocks. No host dtype conversion on any path.
        //
        // Paged: register this layer's mmap bytes with the pager and bind a tiny
        // placeholder instead of uploading the full bank — the Vulkan adapter recognizes the
        // placeholder's identity (see `infr_vulkan::pager`'s module doc) and diverts to the
        // paged executor split at execute time. `down_scale`/router/every other tensor of a
        // paged layer is unaffected — only the `_exps` weight banks divert here.
        // Dense layer streaming: bind a tiny placeholder and register the group's ZERO-COPY mmap
        // segments with the dense session instead of uploading (the adapter recognizes the
        // placeholder's identity at execute time and dispatches against the pool arena — see
        // `infr_vulkan::pager`'s dense-session doc). The `tb` byte-length check is the drift
        // guard between this plan's group enumeration and the runner's actual upload grouping
        // (`fuse_*_decision` keeps them aligned; a mismatch here is a bug, caught loudly).
        if let Some((pool, block_id, comps)) = dense_plan.get(name) {
            // The same u32 element cap the resident upload below applies: a streamed group runs
            // the same kernels, and the float ones index their weight in u32 elements.
            check_bda_element_cap(name, "tensor", numel)?;
            // With a host tier under this pool, the group's bytes are read from the model file on
            // demand instead of faulted in through the mmap. `file_ranges` is `None` exactly for
            // bytes the loader REWROTE, which correspond to nothing on disk — those keep the mmap
            // source even when a tier exists (the eligibility filter already excludes them, so
            // this is a fallback, not a normal path).
            let host = dense_host.get(*pool).and_then(|h| h.as_ref());
            let (bytes, plan_bytes) = match (host, tb.file_ranges()) {
                (Some(h), Some(ranges)) => {
                    let total: usize = ranges.iter().map(|(_, l)| l).sum();
                    h.register(infr_core::blockio::BlockDesc {
                        id: *block_id,
                        extents: ranges
                            .iter()
                            .map(|&(offset, len)| infr_core::blockio::BlockExtent { offset, len })
                            .collect(),
                    })
                    .map_err(|e| anyhow!("{e}"))?;
                    (infr_vulkan::pager::DenseBytes::Host, total)
                }
                _ => {
                    let segments: Vec<std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>> = comps
                        .iter()
                        .map(|c| {
                            Ok(std::sync::Arc::new(
                                g.tensor_bytes_arc(c).map_err(|e| anyhow!("{e}"))?,
                            )
                                as std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>)
                        })
                        .collect::<AResult<_>>()?;
                    let total = segments.iter().map(|s| s.as_ref().as_ref().len()).sum();
                    (infr_vulkan::pager::DenseBytes::Mmap(segments), total)
                }
            };
            if plan_bytes != tb.len() {
                return Err(anyhow!(
                    "dense streaming plan out of sync with the upload order for {name}: plan \
                     bytes {plan_bytes} != uploaded bytes {}",
                    tb.len()
                ));
            }
            let placeholder = vk
                .alloc_uninit(4, BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            let buf_id = infr_vulkan::pager::buffer_identity(placeholder.as_ref());
            vk.register_dense_stream(
                *pool,
                buf_id,
                infr_vulkan::pager::DenseSource {
                    bytes,
                    block_id: *block_id,
                },
            )
            .map_err(|e| anyhow!("{e}"))?;
            return Ok((placeholder, dt));
        }
        if let Some(l) = exps_layer(name).filter(|&l| l < n_paged) {
            if let (WBytes::Mmap(bytes), Some(role)) = (&tb, moe_role_of(name)) {
                let n_expert = cfg
                    .moe
                    .as_ref()
                    .expect("a paged tensor implies an MoE config")
                    .n_expert
                    .max(1);
                // The ADDRESSING UNIT of a stacked bank is ONE per-expert slice, not the whole
                // bank (the arena kernels index `arena + expert * stride`), so the whole-bank
                // element count may legitimately exceed 2^32 while each slice must not.
                check_bda_element_cap(name, "per-expert slice", numel / n_expert)?;
                let stride_bytes = bytes.len() / n_expert;
                let placeholder = vk
                    .alloc_uninit(4, BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                let buf_id = infr_vulkan::pager::buffer_identity(placeholder.as_ref());
                let layer_base = (l * n_expert) as u32;
                // With a host tier under this pool, register the bank's experts INDIVIDUALLY — one
                // block per expert, at its own file offset within the bank, under the same global
                // id the arena uses. That is what lets a routed miss read one expert instead of
                // faulting in the whole bank through the mapping.
                let host = moe_host
                    .iter()
                    .find(|&&(r, sb, _)| r == role && sb == stride_bytes)
                    .and_then(|(_, _, h)| h.as_ref());
                let source_bytes = match host {
                    Some(h) => {
                        let (base, len) = bytes.file_range();
                        if len != stride_bytes * n_expert {
                            return Err(anyhow!(
                                "MoE host tier: {name}'s file range is {len} bytes, but the pool \
                                 expects {n_expert} x {stride_bytes}"
                            ));
                        }
                        for e in 0..n_expert {
                            h.register(infr_core::blockio::BlockDesc {
                                id: layer_base + e as u32,
                                extents: vec![infr_core::blockio::BlockExtent {
                                    offset: base + (e * stride_bytes) as u64,
                                    len: stride_bytes,
                                }],
                            })
                            .map_err(|e| anyhow!("{e}"))?;
                        }
                        infr_vulkan::pager::ExpertBytes::Host
                    }
                    None => {
                        infr_vulkan::pager::ExpertBytes::Mmap(std::sync::Arc::new(bytes.clone())
                            as std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>)
                    }
                };
                let source = infr_vulkan::pager::ExpertSource {
                    bytes: source_bytes,
                    stride_bytes,
                    layer_base,
                };
                vk.register_paged_expert(role, buf_id, source, n_expert)
                    .map_err(|e| anyhow!("{e}"))?;
                return Ok((placeholder, dt));
            }
        }
        // Ordinary (non-paged, non-streamed) weight. A RESIDENT stacked expert bank addresses ONE
        // per-expert slice at a time (the arena kernels index `arena + expert * stride`), exactly
        // like the paged path above — so its whole-bank element count may legitimately exceed 2^32
        // while each per-expert slice must not. This used to take the conservative WHOLE-tensor cap
        // because the flag-off resident id kernels did whole-bank u32 element math; those kernels
        // are gone (weights are u64 BDA-addressed only), so the slice cap is now correct. Every
        // non-expert weight is still one whole-tensor addressing unit.
        if moe_role_of(name).is_some() {
            let n_expert = cfg
                .moe
                .as_ref()
                .expect("an expert-bank tensor implies an MoE config")
                .n_expert
                .max(1);
            check_bda_element_cap(name, "per-expert slice", numel / n_expert)?;
        } else if chunk_covered_dense_tensor(name) {
            // lm_head / embedding tables (issue #77): read ONLY by chunk-covered dispatches (the
            // output projection's decode GEMV + Op::EmbedGather), which split a >= 2^32-element
            // tensor into output-row chunks at DISPATCH, so the whole-tensor element cap no longer
            // applies. `bda_weight_alloc`'s 4 GiB BYTE cap still bounds the single contiguous
            // allocation (this over-cap table must fit — a quantized 256k-vocab lm_head), and a
            // multi-row lm_head GEMM (MTP verify / all-position logits) is still caught loudly at
            // dispatch (the adapter's tiled-GEMM breach guard). Every OTHER dense tensor below
            // keeps the loud whole-tensor element cap.
        } else {
            check_bda_element_cap(name, "tensor", numel)?;
        }
        let bytes = tb.materialize();
        let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
        // alloc_uninit: the `upload` right below writes the buffer's FULL extent (it is sized to
        // exactly `padded.len()`), so the calloc contract's zero-fill is dead work — and an
        // expensive kind: on the device-local path it costs a `vkCmdFillBuffer` over the whole
        // model plus a submit + `queue_wait_idle` PER TENSOR, doubling the load's stall count.
        let buf = vk
            .alloc_uninit(padded.len(), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        vk.upload(buf.as_ref(), &padded)
            .map_err(|e| anyhow!("{e}"))?;
        drop(padded);
        drop(bytes);
        if let Some(r) = &releaser {
            r.release(name, tb);
        }
        Ok((buf, dt))
    }))
}

/// Hands the mapped pages of weights that now live in VRAM back to the OS file cache
/// (`paging.release_uploaded`), on a thread of its own so the trim stays off the load's critical
/// path. Measured on Windows (2026-11-16): about 0.3 s for the 3.46 GB LFM3-1.7B F16 file
/// inline, against an 8 s load. Dropping it closes the queue and joins the thread, which by then
/// (the binder lives for the whole first generation) has long finished.
///
/// Only the resident upload path sends weights here: a paged or streamed weight is read through
/// the mapping again on every miss, so its pages are the pager's to keep. `Owned` bytes have
/// nothing to release: they are a load-time rewrite with no mapped form, and the source pages that
/// rewrite read are not tracked. A host reader that touches a released weight later (a CPU leg on
/// the same model, the host embedding gather) faults it back in once and keeps it until a later
/// binder for the same model uploads, and so releases, that weight again.
struct PageReleaser {
    tx: Option<std::sync::mpsc::Sender<(String, Vec<TensorBytes>)>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl PageReleaser {
    /// Start the release thread; if the OS refuses one, releases happen inline on the caller.
    fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<(String, Vec<TensorBytes>)>();
        match std::thread::Builder::new()
            .name("infr-release-mapped".into())
            .spawn(move || {
                for (name, views) in rx {
                    release_mapped_pages(&name, &views);
                }
            }) {
            Ok(thread) => Self {
                tx: Some(tx),
                thread: Some(thread),
            },
            Err(e) => {
                tracing::debug!("no thread for releasing mapped pages, releasing inline: {e}");
                Self {
                    tx: None,
                    thread: None,
                }
            }
        }
    }

    /// Queue `tb`'s mapped views for release (or release them now, without a thread).
    fn release(&self, name: &str, tb: WBytes) {
        let views = match tb {
            WBytes::Mmap(t) => vec![t],
            WBytes::Concat(parts) => parts,
            WBytes::Owned(_) => return,
        };
        match &self.tx {
            // A send fails only once the thread is gone, which takes a panic in it; the views come
            // back in the error and are released here instead.
            Some(tx) => {
                if let Err(std::sync::mpsc::SendError((name, views))) =
                    tx.send((name.to_owned(), views))
                {
                    release_mapped_pages(&name, &views);
                }
            }
            None => release_mapped_pages(name, &views),
        }
    }
}

impl Drop for PageReleaser {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(t) = self.thread.take() {
            if t.join().is_err() {
                tracing::warn!("the mapped-page release thread panicked");
            }
        }
    }
}

/// Release one weight's mapped views (see [`PageReleaser`]). A release that fails costs host
/// memory, never correctness, so it is logged rather than failing the load; an unsupported
/// platform is logged once per weight at debug level.
fn release_mapped_pages(name: &str, views: &[TensorBytes]) {
    for v in views {
        match v.release_resident() {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Unsupported => {
                tracing::debug!("{name}: mapped pages kept resident: {e}");
                return;
            }
            Err(e) => {
                tracing::warn!("{name}: releasing its mapped pages after the upload failed: {e}");
                return;
            }
        }
    }
}

/// Open one Vulkan backend per physical device index (the shared front of every multi-GPU
/// `generate_*` wrapper). Errors name the failing `VulkanN`.
///
/// Every backend gets the SAME `Arc<Config>`: per-device configs are explicitly out of scope for
/// the config campaign, and today's multi-GPU paths all read one process environment, so one
/// shared handle reproduces that exactly.
fn open_vulkan_devices(
    devices: &[usize],
    ec: &EngineConfig,
) -> AResult<Vec<infr_vulkan::VulkanBackend>> {
    let cfg = std::sync::Arc::new(ec.clone());
    devices
        .iter()
        .map(|&idx| {
            infr_vulkan::VulkanBackend::new_on_with(idx, cfg.clone())
                .map_err(|e| anyhow!("vulkan init (Vulkan{idx}): {e}"))
        })
        .collect()
}

/// The arch guards the DENSE multi-GPU paths (pipeline, tensor-parallel) share: they cover
/// dense-attention Qwen3/Llama/Gemma only, rejecting MoE / qwen35 DeltaNet / gemma-E2B /
/// diffusion-gemma with a clear per-flag message (`label` = the env var). Expert parallelism has
/// its own guard (it REQUIRES MoE).
fn dense_multi_gpu_guard(cfg: &Config, label: &str) -> AResult<()> {
    if cfg.moe.is_some() {
        return Err(anyhow!(
            "{label} supports dense models only; this is an MoE model — use INFR_EXPERT_PARALLEL"
        ));
    }
    if cfg.has_recurrent_state() {
        return Err(anyhow!(
            "{label} does not support recurrent-state layers (qwen35 DeltaNet, LFM short \
             convolution); their placement is a separate slice"
        ));
    }
    if cfg.n_embd_per_layer > 0 {
        return Err(anyhow!(
            "{label} does not support gemma E2B per-layer embeddings"
        ));
    }
    if cfg.diffusion_gemma {
        return Err(anyhow!("{label} does not support diffusion-gemma"));
    }
    Ok(())
}

/// Drive `generate_dense_backend` as a ONE-SHOT (single conversation, no slot pool): fresh `None`
/// state, `want_ctx = prompt + max_new + 1`, and the eight trailing hooks (constraint / verify /
/// verify_ids / logits_out / h_out / denoise_req / req) all unused. The shared tail of the
/// multi-GPU `generate_*` wrappers, run inside [`backend_chunk_scope`] so the prefill chunk is
/// sized for `be`'s devices.
fn run_dense_oneshot(
    be: &dyn Backend,
    bind: &BindWeight<'_>,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    let _chunk = backend_chunk_scope(&be.capabilities(), cfg);
    generate_dense_backend(
        be,
        bind,
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
        &mut None,
        prompt.len() + max_new + 1,
        None, // constraint
        None, // verify
        None, // verify_ids
        None, // logits_out
        None, // h_out
        None, // denoise_req
        None, // req
    )
}

/// The `multi.pipeline` device list (`INFR_PIPELINE=Vulkan0,Vulkan1,…`), or `None` for
/// single-device. Needs >=2 devices for a layer split; garbage or too-few errors LOUDLY — but now
/// at `Config::load`, not here, since the grammar and the minimum both moved into
/// `infr_core::config::parse_device_spec`. The `parse_device_spec` / `parse_device_list` pair this
/// crate used to own is DELETED — it was a second copy of that grammar.
pub fn pipeline_devices(ec: &EngineConfig) -> Option<&[usize]> {
    ec.multi.pipeline.as_deref()
}

/// A device-aware [`BindWeight`] for the multi-GPU pipeline: each weight is placed on the device
/// [`PipelineBackend::device_for_weight`] chooses (by tensor name — `blk.{l}.*` → layer `l`'s
/// device, `output*`/`token_embd` → last device) and wrapped in a single-device
/// [`infr_vulkan::PipelineBuffer`] so the executor can pin its readers to that device. Dense
/// resident weights only (pipeline v1 is dense-attention; the MoE-paged / dense-streamed placement
/// tiers of [`vulkan_moe_binder`] are out of scope).
fn pipeline_binder<'a>(pb: &'a infr_vulkan::PipelineBackend) -> Box<BindWeight<'a>> {
    Box::new(move |name: &str, tb: WBytes, dt: DType, numel: usize| {
        // lm_head / embedding tables (issue #77) are read only by dispatch-chunked ops, so they may
        // legitimately exceed the u32 element cap — mirror the resident/EP binders and exempt them
        // (a large/quantized-vocab lm_head must not hard-reject a model that runs single-device/EP).
        if !chunk_covered_dense_tensor(name) {
            check_bda_element_cap(name, "tensor", numel)?;
        }
        let d = pb.device_for_weight(name);
        let bytes = tb.materialize();
        let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
        let buf = pb
            .backend(d)
            .alloc_uninit(padded.len(), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        pb.backend(d)
            .upload(buf.as_ref(), &padded)
            .map_err(|e| anyhow!("{e}"))?;
        Ok((infr_vulkan::PipelineBuffer::single(d, buf), dt))
    })
}

/// Multi-GPU PIPELINE (layer-split) dense generation — the layers of ONE model are split across
/// the `devices` (a physical index list, e.g. `[0, 1]`), each layer's weights + KV resident on its
/// device, the residual hidden state handed across at the split (P2P dma-buf when available, else
/// host-bounce). The forward is BIT-IDENTICAL to the same model run single-device (identical ops +
/// per-device kernels; only the boundary residual crosses via a value-preserving copy).
///
/// Dense attention models only (Qwen3/Llama/Gemma-dense); MoE / qwen35 DeltaNet / gemma E2B / the
/// paged + streamed weight tiers are rejected with a clear message (they need per-layer placement
/// work beyond this slice). Runs one-shot (a single conversation, no slot pool).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_vulkan_pipeline(
    devices: &[usize],
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    dense_multi_gpu_guard(cfg, "INFR_PIPELINE")?;
    let backends = open_vulkan_devices(devices, ec)?;
    let layer_map = infr_vulkan::PipelineBackend::balanced_layer_map(cfg.n_layer, backends.len());
    // Placement report: how many layers landed on each physical device.
    let names = backends
        .iter()
        .map(|b| {
            use infr_core::backend::Backend;
            b.capabilities().name
        })
        .collect::<Vec<_>>();
    // `multi.pipeline_p2p` (`INFR_PIPELINE_HOST` inverted): its ABSENCE selects P2P.
    let use_p2p = ec.multi.pipeline_p2p;
    let pb = infr_vulkan::PipelineBackend::new(backends, layer_map.clone(), use_p2p)
        .map_err(|e| anyhow!("{e}"))?;
    tracing::info!(
        "pipeline: {}-way layer split of {} layers:",
        devices.len(),
        cfg.n_layer
    );
    for (di, &idx) in devices.iter().enumerate() {
        let lo = layer_map.iter().position(|&d| d == di);
        let hi = layer_map.iter().rposition(|&d| d == di);
        let count = layer_map.iter().filter(|&&d| d == di).count();
        match (lo, hi) {
            (Some(lo), Some(hi)) => tracing::info!(
                "  Vulkan{idx} ({}): layers [{lo}..={hi}] ({count} layers){}",
                names[di],
                if di + 1 < devices.len() {
                    ""
                } else {
                    " + final norm + lm_head"
                }
            ),
            _ => tracing::info!("  Vulkan{idx} ({}): (no layers)", names[di]),
        }
    }
    let bind = pipeline_binder(&pb);
    run_dense_oneshot(
        &pb, &*bind, g, cfg, ec, token_embd, ple, prompt, max_new, on_token,
    )
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Tensor parallelism (dense) — Megatron-style intra-op weight sharding. See `infr_vulkan::tp`.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The `multi.tensor_parallel` device list (`INFR_TENSOR_PARALLEL=Vulkan0,Vulkan1,…`), or `None`
/// when unset. Sibling of [`pipeline_devices`]; needs >=2 devices for a real split.
pub fn tensor_parallel_devices(ec: &EngineConfig) -> Option<&[usize]> {
    ec.multi.tensor_parallel.as_deref()
}

/// The tensor-parallel device role of a weight (by GGUF tensor name), plus its INNER dim `in_f` (for
/// the row-parallel byte stride). `None` = replicated (norms, biases, embeddings, the LM head).
///
/// Column-parallel (sliced by output rows): q/k/v/gate/up. Row-parallel (sliced by input columns):
/// attn_output / ffn_down.
fn tp_weight_role(name: &str, cfg: &Config) -> Option<(infr_vulkan::TpRole, usize)> {
    let layer = name
        .strip_prefix("blk.")
        .and_then(|r| r.split('.').next())
        .and_then(|s| s.parse::<usize>().ok());
    let l = layer?;
    // Only a projection MATRIX is sharded. Its bias (qwen2's `attn_q.bias`) shares the projection
    // name, but it is a 1-D vector the column slicer cannot cut by rows; biases are replicated.
    // e.g. "attn_q" from "blk.3.attn_q.weight"
    let after = name.strip_suffix(".weight")?.rsplit('.').next()?;
    let ne = cfg.n_embd;
    let qrow = cfg.n_head * cfg.layer_head_dim(l);
    let nff = cfg.layer_n_ff(l);
    match after {
        "attn_q" | "attn_k" | "attn_v" => Some((infr_vulkan::TpRole::Column, ne)),
        "ffn_gate" | "ffn_up" => Some((infr_vulkan::TpRole::Column, ne)),
        "attn_output" => Some((infr_vulkan::TpRole::Row, qrow)),
        "ffn_down" => Some((infr_vulkan::TpRole::Row, nff)),
        _ => None, // attn_norm/ffn_norm/q_norm/k_norm/biases → replicated
    }
}

/// Refuse a model with q/k/v projection biases under tensor parallelism. Each rank computes only
/// its band of the column-sharded projection, and the TP lowering has no sharded `Op::AddBias`
/// to add the matching band of a replicated bias (it refuses an elementwise op on a sharded
/// tensor) — so say which model feature is missing rather than fail in the lowering.
fn tp_bias_guard(cfg: &Config) -> AResult<()> {
    if cfg.qkv_bias {
        return Err(anyhow!(
            "INFR_TENSOR_PARALLEL does not support attention q/k/v projection biases \
             (attn_q/attn_k/attn_v.bias, e.g. qwen2): sharding their projections needs a sharded \
             bias add — run this model on one device or with INFR_PIPELINE"
        ));
    }
    Ok(())
}

/// Column slice: rank `r` of `world` takes the contiguous output-row band
/// `[r·out_f/W, (r+1)·out_f/W)` of a row-major `[out_f, in_f]` tensor. Quant-block-safe: blocks tile
/// along `in_f` (within a row), so a whole-row band never cuts a block.
fn tp_slice_column(
    bytes: &[u8],
    dt: DType,
    in_f: usize,
    r: usize,
    world: usize,
) -> AResult<Vec<u8>> {
    let (be, bb) = infr_gguf::block_layout(dt);
    if !in_f.is_multiple_of(be) {
        return Err(anyhow!(
            "tp column slice: in_f={in_f} not a multiple of block {be}"
        ));
    }
    let row_bytes = (in_f / be) * bb;
    if !bytes.len().is_multiple_of(row_bytes) {
        return Err(anyhow!(
            "tp column slice: {} bytes not a multiple of row {row_bytes}",
            bytes.len()
        ));
    }
    let out_f = bytes.len() / row_bytes;
    if !out_f.is_multiple_of(world) {
        return Err(anyhow!(
            "tp column slice: out_f={out_f} not divisible by world {world}"
        ));
    }
    let rows = out_f / world;
    let start = r * rows * row_bytes;
    Ok(bytes[start..start + rows * row_bytes].to_vec())
}

/// Row slice: rank `r` takes the input-column band `[r·in_f/W, (r+1)·in_f/W)` of every one of the
/// `out_f` rows and re-packs them contiguously into a `[out_f, in_f/W]` tensor. Needs `in_f/W`
/// block-aligned so each per-row band is a whole number of quant blocks.
fn tp_slice_row(bytes: &[u8], dt: DType, in_f: usize, r: usize, world: usize) -> AResult<Vec<u8>> {
    let (be, bb) = infr_gguf::block_layout(dt);
    if !in_f.is_multiple_of(be * world) {
        return Err(anyhow!(
            "tp row slice: in_f={in_f} not divisible by world·block ({world}·{be}) — the input-column \
             split must land on quant-block boundaries"
        ));
    }
    let row_bytes = (in_f / be) * bb;
    if !bytes.len().is_multiple_of(row_bytes) {
        return Err(anyhow!(
            "tp row slice: {} bytes not a multiple of row {row_bytes}",
            bytes.len()
        ));
    }
    let out_f = bytes.len() / row_bytes;
    let band_bytes = ((in_f / world) / be) * bb;
    let col_off = r * band_bytes;
    let mut out = Vec::with_capacity(out_f * band_bytes);
    for row in 0..out_f {
        let s = row * row_bytes + col_off;
        out.extend_from_slice(&bytes[s..s + band_bytes]);
    }
    Ok(out)
}

/// A device-aware [`BindWeight`] for tensor parallelism: q/k/v/gate/up are COLUMN-sliced (output
/// rows), attn_output/ffn_down are ROW-sliced (input columns) and each rank uploads only its slice;
/// norms/biases/embeddings/lm_head are REPLICATED to every rank. Each slice is padded to u32
/// alignment and uploaded to its rank's device, returned as an `infr_vulkan::TpBuffer` carrying the
/// device role the TP lowering reads.
fn tensor_parallel_binder<'a>(
    tp: &'a infr_vulkan::TensorParallelBackend,
    cfg: &'a Config,
) -> Box<BindWeight<'a>> {
    let world = tp.world();
    Box::new(move |name: &str, tb: WBytes, dt: DType, numel: usize| {
        // lm_head / embedding tables (issue #77) are read only by dispatch-chunked ops, so they may
        // legitimately exceed the u32 element cap — mirror the resident/EP binders and exempt them
        // (they are replicated below, never sharded, so the whole-tensor cap would wrongly reject a
        // large/quantized-vocab lm_head that runs fine single-device/EP).
        if !chunk_covered_dense_tensor(name) {
            check_bda_element_cap(name, "tensor", numel)?;
        }
        match tp_weight_role(name, cfg) {
            Some((role, in_f)) => {
                // Every rank's slice is cut from the whole tensor, so this binder needs the bytes
                // themselves — materialize once for all `world` cuts, not per rank.
                let tb = tb.materialize();
                let mut bufs = Vec::with_capacity(world);
                for r in 0..world {
                    let slice = match role {
                        infr_vulkan::TpRole::Column => tp_slice_column(&tb, dt, in_f, r, world)?,
                        infr_vulkan::TpRole::Row => tp_slice_row(&tb, dt, in_f, r, world)?,
                        infr_vulkan::TpRole::Replicated => tb.to_vec(),
                    };
                    let padded = infr_vulkan::linear::pad_to_u32_align(&slice);
                    let buf = tp
                        .rank(r)
                        .alloc_uninit(padded.len(), BufferUsage::Weights)
                        .map_err(|e| anyhow!("{e}"))?;
                    tp.rank(r)
                        .upload(buf.as_ref(), &padded)
                        .map_err(|e| anyhow!("{e}"))?;
                    bufs.push(buf);
                }
                Ok((infr_vulkan::TpBuffer::weight(role, bufs), dt))
            }
            None => {
                // Replicated: the full padded tensor on every rank.
                let bytes = tb.materialize();
                let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
                let mut bufs = Vec::with_capacity(world);
                for r in 0..world {
                    let buf = tp
                        .rank(r)
                        .alloc_uninit(padded.len(), BufferUsage::Weights)
                        .map_err(|e| anyhow!("{e}"))?;
                    tp.rank(r)
                        .upload(buf.as_ref(), &padded)
                        .map_err(|e| anyhow!("{e}"))?;
                    bufs.push(buf);
                }
                Ok((infr_vulkan::TpBuffer::replica(bufs), dt))
            }
        }
    })
}

/// Multi-GPU TENSOR-PARALLEL (dense) generation — each transformer layer's weight matrices are
/// SHARDED across the `devices` (column-parallel q/k/v/gate/up, row-parallel o/down), each device
/// computes its shard and the partials are all-reduced (P2P dma-buf) per attention + per FFN. The
/// output equals the single-device forward to reduction-order tolerance. Dense attention models only.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_vulkan_tp(
    devices: &[usize],
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    dense_multi_gpu_guard(cfg, "INFR_TENSOR_PARALLEL")?;
    tp_bias_guard(cfg)?;
    let backends = open_vulkan_devices(devices, ec)?;
    let names = backends
        .iter()
        .map(|b| {
            use infr_core::backend::Backend;
            b.capabilities().name
        })
        .collect::<Vec<_>>();
    // `multi.tp_p2p` (`INFR_TP_HOST` inverted): its ABSENCE selects P2P.
    let use_p2p = ec.multi.tp_p2p;
    let tp =
        infr_vulkan::TensorParallelBackend::new(backends, cfg.n_head, cfg.n_kv, cfg.n_ff, use_p2p)
            .map_err(|e| anyhow!("{e}"))?;
    tracing::info!(
        "tensor-parallel: {}-way weight split (n_head={}, n_kv={}, n_ff={}):",
        devices.len(),
        cfg.n_head,
        cfg.n_kv,
        cfg.n_ff
    );
    for (di, &idx) in devices.iter().enumerate() {
        tracing::info!(
            "  Vulkan{idx} ({}): rank {di} — {}/{} heads, {}/{} kv-heads, {}/{} ffn per matrix",
            names[di],
            cfg.n_head / devices.len(),
            cfg.n_head,
            cfg.n_kv / devices.len(),
            cfg.n_kv,
            cfg.n_ff / devices.len(),
            cfg.n_ff,
        );
    }
    let bind = tensor_parallel_binder(&tp, cfg);
    run_dense_oneshot(
        &tp, &*bind, g, cfg, ec, token_embd, ple, prompt, max_new, on_token,
    )
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Expert parallelism (MoE) — shard the experts across devices. See `infr_vulkan::ep`.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The `multi.expert_parallel` device list (`INFR_EXPERT_PARALLEL=Vulkan0,Vulkan1,…`), or `None`
/// when unset. Sibling of [`tensor_parallel_devices`], but its minimum is **1**, not 2 (a single
/// device is the identity, used only as the correctness reference) — the three minimums differ and
/// are preserved by the env layer.
pub fn expert_parallel_devices(ec: &EngineConfig) -> Option<&[usize]> {
    ec.multi.expert_parallel.as_deref()
}

/// Whether `name` is a stacked expert-bank weight (`ffn_{gate,up,down}_exps` or fused
/// `ffn_gate_up_exps`) — the ONLY tensors Expert Parallelism shards; everything else is replicated.
fn is_expert_bank(name: &str) -> bool {
    name.ends_with("ffn_gate_exps.weight")
        || name.ends_with("ffn_up_exps.weight")
        || name.ends_with("ffn_down_exps.weight")
        || name.ends_with("ffn_gate_up_exps.weight")
}

/// A device-aware [`BindWeight`] for Expert Parallelism: the stacked expert banks
/// (`ffn_{gate,up,down}_exps`) are split by EXPERT — rank `r` of `world` gets the contiguous band
/// `[r·E/W, (r+1)·E/W)` (each per-expert slice is `nbytes/n_expert`, block-aligned, so a whole-band
/// cut never splits a quant block) and uploads ONLY that band; every other weight (router,
/// attention, norms, embeddings, the LM head) is REPLICATED to every rank. Each slice/replica is
/// padded to u32 alignment and uploaded to its rank's device, returned as an
/// `infr_vulkan::EpBuffer`.
fn expert_parallel_binder<'a>(
    ep: &'a infr_vulkan::ExpertParallelBackend,
    cfg: &'a Config,
) -> Box<BindWeight<'a>> {
    let world = ep.world();
    let n_expert = cfg.moe.as_ref().map(|m| m.n_expert).unwrap_or(0).max(1);
    Box::new(move |name: &str, tb: WBytes, dt: DType, numel: usize| {
        if is_expert_bank(name) {
            // Split the bank by expert. The stacked bank is `n_expert` contiguous per-expert slices
            // (the arena kernels index `arena + expert·stride`), so band `[r·nl, (r+1)·nl)` is a
            // contiguous byte range and its per-expert element cap is `numel/n_expert`.
            let total = tb.len();
            if !total.is_multiple_of(n_expert) {
                return Err(anyhow!(
                    "ep: expert bank '{name}' {total} bytes not divisible by n_expert={n_expert}"
                ));
            }
            let stride_bytes = total / n_expert;
            if !n_expert.is_multiple_of(world) {
                return Err(anyhow!(
                    "ep: n_expert={n_expert} not divisible by world {world}"
                ));
            }
            let nl = n_expert / world;
            check_bda_element_cap(name, "per-expert slice", numel / n_expert)?;
            // Each rank uploads its own band of the bank, so the bytes are needed here.
            let tb = tb.materialize();
            let mut bufs = Vec::with_capacity(world);
            for r in 0..world {
                let start = r * nl * stride_bytes;
                let slice = &tb[start..start + nl * stride_bytes];
                let padded = infr_vulkan::linear::pad_to_u32_align(slice);
                let buf = ep
                    .rank(r)
                    .alloc_uninit(padded.len(), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                ep.rank(r)
                    .upload(buf.as_ref(), &padded)
                    .map_err(|e| anyhow!("{e}"))?;
                bufs.push(buf);
            }
            Ok((infr_vulkan::EpBuffer::shard(bufs), dt))
        } else {
            // Replicated: the full padded tensor on every rank. Mirror the resident binder's
            // per-expert-slice / chunk-covered / whole-tensor element-cap policy for non-bank
            // weights (a replicated router is a whole tensor; lm_head/token_embd are chunk-covered).
            if chunk_covered_dense_tensor(name) {
                // no whole-tensor cap (dispatch-chunked reads)
            } else {
                check_bda_element_cap(name, "tensor", numel)?;
            }
            let bytes = tb.materialize();
            let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
            let mut bufs = Vec::with_capacity(world);
            for r in 0..world {
                let buf = ep
                    .rank(r)
                    .alloc_uninit(padded.len(), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                ep.rank(r)
                    .upload(buf.as_ref(), &padded)
                    .map_err(|e| anyhow!("{e}"))?;
                bufs.push(buf);
            }
            Ok((infr_vulkan::EpBuffer::wrap(bufs), dt))
        }
    })
}

/// The architectures expert parallelism refuses, checked before any device is opened. Returns the
/// model's routed-expert config, which EP requires.
fn expert_parallel_guard(cfg: &Config) -> AResult<&crate::MoeConfig> {
    let Some(moe) = cfg.moe.as_ref() else {
        return Err(anyhow!(
            "INFR_EXPERT_PARALLEL needs a routed-expert (MoE) model — this is a dense model (use \
             INFR_TENSOR_PARALLEL or INFR_PIPELINE)"
        ));
    };
    if cfg.qwen35 {
        return Err(anyhow!(
            "INFR_EXPERT_PARALLEL does not yet support qwen35 (DeltaNet recurrent-state replication \
             is a separate slice)"
        ));
    }
    if cfg.shexp_ff > 0 {
        return Err(anyhow!(
            "INFR_EXPERT_PARALLEL v1 does not yet place a SHARED expert (qwen35moe / llama4 \
             MoeSharedExpertAdd) — only routed-only MoE (qwen3moe). Shared-expert placement is a \
             separate slice"
        ));
    }
    if cfg.n_embd_per_layer > 0 {
        return Err(anyhow!(
            "INFR_EXPERT_PARALLEL does not support gemma E2B per-layer embeddings"
        ));
    }
    if cfg.diffusion_gemma {
        return Err(anyhow!(
            "INFR_EXPERT_PARALLEL does not support diffusion-gemma"
        ));
    }
    // The dual dense ∥ MoE FFN (gemma4 26B-A4B): its dense branch is computed before the MoE op
    // and summed after it, across the cut EP makes there, and its per-expert down scale is
    // indexed by the band-local expert id. Neither is carried by the EP lowering.
    if cfg.dual_moe() {
        return Err(anyhow!(
            "INFR_EXPERT_PARALLEL does not support gemma4 MoE (the dual dense + MoE FFN with a \
             per-expert down scale) — run it on one device"
        ));
    }
    Ok(moe)
}

/// Multi-GPU EXPERT-PARALLEL (MoE) generation — the model's experts are split across the `devices`
/// (rank `r` owns experts `[r·E/W, (r+1)·E/W)`), the router + attention + norms run replicated on
/// every rank, each rank computes only its band's experts, and one P2P all-reduce per MoE layer
/// combines the partial expert outputs. The output equals the single-device MoE to reduction-order
/// tolerance (token-identical greedy). qwen3moe (split gate/up, softmax, no shared expert) in v1.
#[cfg_attr(infr_profile, infr_prof::instrument)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_moe_vulkan_ep(
    devices: &[usize],
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    let moe = expert_parallel_guard(cfg)?;
    let backends = open_vulkan_devices(devices, ec)?;
    let names = backends
        .iter()
        .map(|b| {
            use infr_core::backend::Backend;
            b.capabilities().name
        })
        .collect::<Vec<_>>();
    // `multi.ep_p2p` (`INFR_EP_HOST` inverted): its ABSENCE selects P2P.
    let use_p2p = ec.multi.ep_p2p;
    let ep = infr_vulkan::ExpertParallelBackend::new(backends, moe.n_expert, use_p2p)
        .map_err(|e| anyhow!("{e}"))?;
    let nl = ep.experts_per_device();
    tracing::info!(
        "expert-parallel: {}-way expert split ({} experts, {} used/token → {nl} experts/device):",
        devices.len(),
        moe.n_expert,
        moe.n_used,
    );
    for (di, &idx) in devices.iter().enumerate() {
        tracing::info!(
            "  Vulkan{idx} ({}): rank {di} — experts [{}..{}) ({nl} of {})",
            names[di],
            di * nl,
            (di + 1) * nl,
            moe.n_expert,
        );
    }
    let bind = expert_parallel_binder(&ep, cfg);
    run_dense_oneshot(
        &ep, &*bind, g, cfg, ec, token_embd, ple, prompt, max_new, on_token,
    )
}

/// Metal seam runner: the SAME dense forward as [`generate_dense_cpu`], on the reference Metal
/// backend through the agnostic [`Graph`]. Weights are uploaded to Metal buffers in their NATIVE
/// GGUF dtype (the backend dequantizes lazily in its own `bytes_to_f32`, exactly like the CPU
/// interpreter — so a quant weight occupies ~quant size, not 8× f32).
#[cfg(target_os = "macos")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_metal(
    mtl: &infr_metal::MetalBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    req: Option<&crate::sampling::RequestCtx>,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_metal_session(
        mtl,
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
        &mut None,
        prompt.len() + max_new + 1,
        None,
        req,
    )
}

/// Persistent-session Metal seam runner — the Metal twin of [`generate_dense_vulkan_session`]:
/// weights upload once into `state`, the KV cache is sized to `want_ctx`, and each call prefills
/// only the suffix that differs from the tokens already materialized in the cache.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_metal_session(
    mtl: &infr_metal::MetalBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
    state: &mut Option<SeamKv>,
    want_ctx: usize,
    constraint: Option<&mut crate::grammar::Constraint>,
    req: Option<&crate::sampling::RequestCtx>,
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_backend(
        mtl,
        &metal_upload_bind(mtl),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
        state,
        want_ctx,
        constraint,
        None,
        None,
        None,
        None,
        None,
        req,
    )
}

/// Speculative VERIFY on the Metal seam: one batched forward of `tokens`' un-cached suffix with
/// the LM head on every suffix row. Returns the [m, vocab] logits plus the graph-execute
/// seconds, and leaves the session's KV + `cached` covering all of `tokens` — the caller
/// commits the accepted prefix and the next call's prefix diff overwrites whatever was
/// speculatively written past it.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_dense_metal2(
    mtl: &infr_metal::MetalBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
    state: &mut Option<SeamKv>,
    want_ctx: usize,
) -> AResult<(Vec<f32>, f64)> {
    let mut logits = Vec::new();
    let (_, stats) = generate_dense_backend(
        mtl,
        &metal_upload_bind(mtl),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        tokens,
        0,
        |_| {},
        state,
        want_ctx,
        None,
        Some(&mut logits),
        None,
        None,
        None,
        None,
        None,
    )?;
    Ok((logits, stats.prompt_secs))
}

/// DiffusionGemma Phase-1 validation: a causal prefill of `tokens` (a fresh one-shot forward, no
/// session) through the CPU reference backend, returning the LAST token's raw (pre-softmax, post-
/// softcap) logits. Rides the ordinary per-token decode loop (`max_new = 1`, the one generated
/// token discarded) — MoE-compatible, unlike the batched `verify` path.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_dense_cpu(
    g: &Gguf,
    cfg: &Config,
    ec: &std::sync::Arc<EngineConfig>,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<Vec<f32>> {
    let cpu_be = CpuBackend::new_with(ec.clone());
    let mut logits = Vec::new();
    let mut state = None;
    generate_dense_backend(
        &cpu_be,
        &cpu_upload_bind(&cpu_be),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        tokens,
        1,
        |_| {},
        &mut state,
        tokens.len() + 2,
        None,
        None,
        None,
        Some(&mut logits),
        None,
        None,
        None,
    )?;
    Ok(logits)
}

/// [`verify_dense_cpu`]'s MTP Phase 1 twin (issue #33, docs/mtp.md): ALSO captures the LM-head
/// input rows (`h_out` — `DecodeHandles::h_out`'s doc) alongside the logits, for the
/// `lm_head(h_row) == logits_row` consistency check `docs/mtp.md`'s Phase 1 validation calls for.
/// Returns `(logits, h)`, both `[vocab]`/`[n_embd]` for the last prompt token.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_dense_cpu_with_h(
    g: &Gguf,
    cfg: &Config,
    ec: &std::sync::Arc<EngineConfig>,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<(Vec<f32>, Vec<f32>)> {
    let cpu_be = CpuBackend::new_with(ec.clone());
    let mut logits = Vec::new();
    let mut h = Vec::new();
    let mut state = None;
    generate_dense_backend(
        &cpu_be,
        &cpu_upload_bind(&cpu_be),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        tokens,
        1,
        |_| {},
        &mut state,
        tokens.len() + 2,
        None,
        None,
        None,
        Some(&mut logits),
        Some(&mut h),
        None,
        None,
    )?;
    Ok((logits, h))
}

/// [`verify_dense_cpu_with_h`]'s ALL-ROWS twin (MTP Phase 2, issue #33): rides the speculative-
/// VERIFY batched forward (the `verify` param, not `logits_out`) so `h`/`logits` cover EVERY one of
/// `tokens`, not just the last — the shape `crate::mtp::catch_up` needs to prime the head's KV over
/// a whole prompt in one call (`docs/mtp.md`'s `process()` runs after every target ubatch, not just
/// the sampled row). Dense non-MoE models only (mirrors the VERIFY branch's own guard). Returns
/// `(logits [tokens.len()*vocab], h [tokens.len()*n_embd])`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_rows_cpu_with_h(
    g: &Gguf,
    cfg: &Config,
    ec: &std::sync::Arc<EngineConfig>,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<(Vec<f32>, Vec<f32>)> {
    let cpu_be = CpuBackend::new_with(ec.clone());
    let mut logits = Vec::new();
    let mut h = Vec::new();
    let mut state = None;
    generate_dense_backend(
        &cpu_be,
        &cpu_upload_bind(&cpu_be),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        tokens,
        0,
        |_| {},
        &mut state,
        tokens.len() + 2,
        None,
        Some(&mut logits),
        None,
        None,
        Some(&mut h),
        None,
        None,
    )?;
    Ok((logits, h))
}

/// The Vulkan binder the one-shot verify forwards share: pad each weight to a 4-byte multiple and
/// upload it.
fn vulkan_upload_bind(vk: &infr_vulkan::VulkanBackend) -> Box<BindWeight<'_>> {
    Box::new(move |_name, tb, dt, _n| {
        let bytes = tb.materialize();
        let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
        let buf = vk
            .alloc(padded.len(), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        vk.upload(buf.as_ref(), &padded)
            .map_err(|e| anyhow!("{e}"))?;
        Ok((buf, dt))
    })
}

/// [`verify_dense_cpu`]'s Vulkan twin — the same one-shot causal prefill through the production
/// Vulkan seam, for the CPU/Vulkan cross-backend parity check.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_dense_vulkan(
    vk: &infr_vulkan::VulkanBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<Vec<f32>> {
    let _scope = enter_model_scope(vk, cfg);
    let mut logits = Vec::new();
    let mut state = None;
    generate_dense_backend(
        vk,
        &vulkan_upload_bind(vk),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        tokens,
        1,
        |_| {},
        &mut state,
        tokens.len() + 2,
        None,
        None,
        None,
        Some(&mut logits),
        None,
        None,
        None,
    )?;
    Ok(logits)
}

/// [`verify_rows_cpu_with_h`]'s Vulkan twin without the `h` rows: one batched forward of every
/// token through the production Vulkan seam, returning `[tokens.len() * vocab]` logits — the
/// all-positions shape a next-token accuracy comparison against the CPU reference needs. Dense
/// non-MoE models only, as the VERIFY branch it rides.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_rows_vulkan(
    vk: &infr_vulkan::VulkanBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<Vec<f32>> {
    let _scope = enter_model_scope(vk, cfg);
    let mut logits = Vec::new();
    let mut state = None;
    generate_dense_backend(
        vk,
        &vulkan_upload_bind(vk),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        tokens,
        0,
        |_| {},
        &mut state,
        tokens.len() + 2,
        None,
        Some(&mut logits),
        None,
        None,
        None,
        None,
        None,
    )?;
    Ok(logits)
}

/// The backend a teacher-forced run drives ([`crate::SeamModel::forced_logits`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForcedOn {
    /// The CPU backend in reference mode (`kernels.cpu.reference`: quantized weights decoded to
    /// f32 and dotted in f32, no int8 activations).
    CpuReference,
    /// The production CPU backend.
    Cpu,
    /// The Vulkan backend on the engine configuration's device.
    Vulkan,
}

/// Which prefill rows a teacher-forced run scores ([`crate::SeamModel::forced_logits`]). Each
/// scored row costs a vocabulary-wide head the production prefill never runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillHead {
    /// Every prompt row before the frontier.
    All,
    /// The last row of each prefill chunk.
    LastOfChunk,
    /// No prefill row: the batched prefill runs exactly as in production, and only the decode
    /// rows are scored.
    Off,
}

/// What a teacher-forced run produced: the logits at every position it scored, in position
/// order. See [`crate::SeamModel::forced_logits`].
#[derive(Clone, Debug)]
pub struct ForcedLogits {
    /// The scored positions; row `r` of `logits` is the next-token distribution after
    /// `seq[..=positions[r]]`.
    pub positions: Vec<usize>,
    /// `[positions.len() * vocab]` raw logits (post-softcap, pre-softmax).
    pub logits: Vec<f32>,
    pub vocab: usize,
    /// How many of the leading rows came from the batched prefill; the rest came from one decode
    /// step each.
    pub prefill_rows: usize,
    /// The whole sequence the run fed: the prompt, then each forced token.
    pub seq: Vec<u32>,
    /// The compute kernels the backend built for the run (Vulkan; empty on the CPU), so a test
    /// can assert which kernel families a configuration actually reached.
    pub kernels: Vec<&'static str>,
    /// What the backend's own routers selected at every position the run fed, whether or not
    /// the run was forced along another run's selection; `None` for a model with no
    /// router-selected expert layer.
    pub routing: Option<Routing>,
}

/// Drive `prompt` and then `n_steps` forced tokens through `be`, capturing the logits at every
/// position, on the production paths: one ORDINARY batched prefill of `prompt[..len-1]` (chunked
/// at the engine's `device.ubatch`; the LM head is added over its rows through
/// `generate_dense_backend_bound`'s `prefill_logits`, nothing else changes), then one per-token
/// decode step per position from `len-1` on, each a warm call on the same session that feeds
/// exactly the one new token, so it reads KV written by the prefill and by every earlier decode
/// step. `next(step, row)` picks the token fed after each decode row: the reference's greedy
/// argmax, or the reference's recorded token when the backend under test is forced along it.
///
/// The decode step is the per-token path: where the backend replays, each step is a warm call that
/// builds, records and executes a fresh replay plan once, so the record-once-replay-many mechanism
/// (parameters advancing across replays of one recording) and the chained decode (several steps
/// per submit, the production greedy path) are NOT exercised; the static per-token path runs when
/// the configuration sets `kernels.vulkan.no_replay`. Forcing needs every step's logits, which the
/// chain does not read back; [`greedy_tail_vulkan`] is how those two are compared with this path.
///
/// A model whose prompt takes no batched prefill (MoE without the batched kernels, gemma-4 E2B)
/// returns no prefill rows; its prompt positions are not scored.
///
/// On a model with router-selected expert layers, every graph also reads back what each such
/// router selected ([`ForcedLogits::routing`], with the selection scores when `route_scores`),
/// and with `route` every such layer runs on that selection instead of its own, at every position
/// (`Op::MoeFfn::expert_ids`): the run is then forced along a reference's routing as well as its
/// tokens. The selection it reads back is its own either way.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forced_run(
    be: &dyn Backend,
    bind_weight: &BindWeight,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    n_steps: usize,
    head: PrefillHead,
    route: Option<&Routing>,
    route_scores: bool,
    mut next: impl FnMut(usize, &[f32]) -> u32,
) -> AResult<ForcedLogits> {
    let routed = !routing::router_selected_layers(cfg).is_empty();
    if route.is_some() && !routed {
        return Err(anyhow!(
            "a routing was supplied for a model with no router-selected expert layer"
        ));
    }
    let mut route_tap = routed.then_some(routing::RouteTap {
        forced: route,
        scores: route_scores,
        taken: Vec::new(),
    });
    if prompt.len() < 4 {
        return Err(anyhow!(
            "a forced run needs a prompt of at least 4 tokens for a batched prefill, got {}",
            prompt.len()
        ));
    }
    let want_ctx = prompt.len() + n_steps + 2;
    let mut state = None;
    let mut tap = runner::PrefillTap {
        last_only: head == PrefillHead::LastOfChunk,
        ..Default::default()
    };
    runner::generate_dense_backend_bound(
        None,
        be,
        bind_weight,
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        0,
        |_| {},
        &mut state,
        want_ctx,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        (head != PrefillHead::Off).then_some(&mut tap),
        route_tap.as_mut(),
    )?;
    let vocab = cfg.vocab;
    let prefill_rows = tap.positions.len();
    if tap.logits.len() != prefill_rows * vocab {
        return Err(anyhow!(
            "prefill logits are not one row of {vocab} per position"
        ));
    }
    // Every row the head ran must be a prompt position before the frontier, in order, and an
    // every-row head must have covered all of them.
    let in_order = tap.positions.windows(2).all(|w| w[0] < w[1])
        && tap.positions.last().is_none_or(|&p| p + 2 == prompt.len());
    if !in_order
        || (head == PrefillHead::All && prefill_rows != 0 && prefill_rows != prompt.len() - 1)
    {
        return Err(anyhow!(
            "the batched prefill returned rows at {:?} for a {}-token prompt",
            tap.positions,
            prompt.len()
        ));
    }
    let mut positions = tap.positions;
    let mut logits = tap.logits;
    let mut seq = prompt.to_vec();
    for step in 0..n_steps {
        let mut row = Vec::new();
        let (_, stats) = runner::generate_dense_backend_bound(
            None,
            be,
            bind_weight,
            g,
            cfg,
            ec,
            token_embd,
            ple,
            &seq,
            1,
            |_| {},
            &mut state,
            want_ctx,
            None,
            None,
            None,
            Some(&mut row),
            None,
            None,
            None,
            None,
            route_tap.as_mut(),
        )?;
        if row.len() != vocab {
            return Err(anyhow!(
                "decode step {step} returned {} logits, not {vocab}",
                row.len()
            ));
        }
        // A step that re-prefilled the sequence would score the prefill again, not the decode
        // path reading the session's KV.
        if (stats.n_cached, stats.n_prompt) != (seq.len() - 1, 1) {
            return Err(anyhow!(
                "decode step {step} reused {} of {} tokens and fed {}, not exactly the new one",
                stats.n_cached,
                seq.len(),
                stats.n_prompt
            ));
        }
        positions.push(seq.len() - 1);
        seq.push(next(step, &row));
        logits.extend_from_slice(&row);
    }
    // Every token but the last was fed: the last forced token is only a row's choice.
    let routing = match &route_tap {
        Some(t) => Some(routing::assemble_routing(
            routing::router_selected_layers(cfg),
            cfg.moe.as_ref().expect("a routed model has a MoeConfig"),
            &t.taken,
            seq.len() - 1,
            route_scores,
        )?),
        None => None,
    };
    Ok(ForcedLogits {
        positions,
        logits,
        vocab,
        prefill_rows,
        seq,
        kernels: Vec::new(),
        routing,
    })
}

/// [`forced_run`] on a fresh CPU backend (`reference` picks `kernels.cpu.reference`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn forced_run_cpu(
    reference: bool,
    g: &Gguf,
    cfg: &Config,
    ec: &std::sync::Arc<EngineConfig>,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    n_steps: usize,
    head: PrefillHead,
    route: Option<&Routing>,
    next: impl FnMut(usize, &[f32]) -> u32,
) -> AResult<ForcedLogits> {
    let be = if reference {
        CpuBackend::reference_with(ec.clone())
    } else {
        CpuBackend::new_with(ec.clone())
    };
    let bind = cpu_upload_bind(&be);
    forced_run(
        &be, &bind, g, cfg, ec, token_embd, ple, prompt, n_steps, head, route, true, next,
    )
}

/// [`forced_run`] on a fresh Vulkan backend for `ec`'s device.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forced_run_vulkan(
    g: &Gguf,
    cfg: &Config,
    ec: &std::sync::Arc<EngineConfig>,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    n_steps: usize,
    head: PrefillHead,
    route: Option<&Routing>,
    next: impl FnMut(usize, &[f32]) -> u32,
) -> AResult<ForcedLogits> {
    let vk = infr_vulkan::VulkanBackend::new_with(ec.clone())
        .map_err(|e| anyhow!("vulkan init: {e}"))?;
    let _scope = enter_model_scope(&vk, cfg);
    let bind = vulkan_upload_bind(&vk);
    // `moe_topk` writes a selection, not the scores it ranked by (`Op::MoeRoute::dst_scores`).
    let mut out = forced_run(
        &vk, &bind, g, cfg, ec, token_embd, ple, prompt, n_steps, head, route, false, next,
    )?;
    out.kernels = vk.built_kernel_names();
    Ok(out)
}

/// What a production greedy generation on Vulkan produced, and what it left in its session
/// ([`crate::SeamModel::greedy_tail_vulkan`]).
#[derive(Clone, Debug)]
pub struct GreedyTail {
    /// The tokens the generation sampled, in order.
    pub ids: Vec<u32>,
    /// How many of them came back from chained submits ([`GenStats::n_chained`]).
    pub chained: usize,
    /// The record-once decode recordings the generation made: one where the whole decode replayed
    /// a single recording, none where the backend does not replay the model.
    pub recordings: u64,
    /// The generation ended itself on an end token ([`GenStats::natural_stop`]), which is then
    /// the last of `ids`; otherwise it ran the length it was asked for.
    pub natural_stop: bool,
    /// The logits after `prompt ++ ids`: one more per-token step on the same session, which feeds
    /// the last sampled token and attends over every KV row the generation wrote.
    pub row: Vec<f32>,
}

/// Generate `n` greedy tokens after `prompt` on a fresh Vulkan backend for `ec`'s device through
/// [`generate_dense_vulkan_session`], the function [`generate_dense_vulkan`] is (the production
/// placement, prefill and decode: chained where `spec.decode_chain` and the device allow, else
/// one replay of a single recording per token, else the static per-token path), then run one more
/// per-token step on the same session and return its logits. That row is how a caller sees what
/// the generation left in the KV cache, which the chain itself never reads back.
///
/// One thing differs from [`generate_dense_vulkan`]: the session is opened with room for that
/// extra step, a context one row longer than the one-shot call asks for.
pub(crate) fn greedy_tail_vulkan(
    g: &Gguf,
    cfg: &Config,
    ec: &std::sync::Arc<EngineConfig>,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    n: usize,
) -> AResult<GreedyTail> {
    let vk = infr_vulkan::VulkanBackend::new_with(ec.clone())
        .map_err(|e| anyhow!("vulkan init: {e}"))?;
    // One scope over the generation and the step after it, which reuses its session.
    let _scope = enter_model_scope(&vk, cfg);
    let want_ctx = prompt.len() + n + 2;
    let mut state = None;
    let (ids, stats) = generate_dense_vulkan_session(
        &vk,
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        n,
        |_| {},
        &mut state,
        want_ctx,
        None,
        None,
    )?;
    let recordings = vk.decode_recordings();
    let seq: Vec<u32> = prompt.iter().chain(&ids).copied().collect();
    let mut row = Vec::new();
    let (_, tail) = runner::generate_dense_backend(
        &vk,
        &*warm_binder(),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        &seq,
        1,
        |_| {},
        &mut state,
        want_ctx,
        None,
        None,
        None,
        Some(&mut row),
        None,
        None,
        None,
    )?;
    // A step that re-prefilled would score a prefill, not the KV the generation wrote.
    if (tail.n_cached, tail.n_prompt) != (seq.len() - 1, 1) {
        return Err(anyhow!(
            "the step after the generation reused {} of {} tokens and fed {}, not exactly the last \
             sampled one",
            tail.n_cached,
            seq.len(),
            tail.n_prompt
        ));
    }
    if row.len() != cfg.vocab {
        return Err(anyhow!(
            "the step after the generation returned {} logits, not {}",
            row.len(),
            cfg.vocab
        ));
    }
    Ok(GreedyTail {
        ids,
        chained: stats.n_chained,
        recordings,
        natural_stop: stats.natural_stop,
        row,
    })
}

/// Backend-generic dense decode runner. Builds the agnostic decode [`Graph`] per token and runs it
/// on `be` (CPU reference or Vulkan). `bind_weight` turns each native-dtype GGUF tensor into a
/// backend buffer: the CPU maps it zero-copy from the mmap; the GPU pads + uploads it to VRAM. This
/// is the single forward both backends share — running it on Vulkan and diffing the CPU oracle is
/// the end-to-end dense parity check.
/// Weight bytes handed to a binder: a zero-copy mmap slice (the normal case), or an owned
/// concatenation (the combined gate+up upload — only produced when `Capabilities::combined_gu`).
pub(crate) enum WBytes {
    Mmap(TensorBytes),
    Owned(Vec<u8>),
    /// Several mapped tensors to be laid down back to back — the fused qkv / gate+up groups.
    ///
    /// Kept as its COMPONENTS rather than a concatenated buffer, because a binder that pages or
    /// streams the group never wants the bytes at all: it registers the components' file ranges and
    /// has them read straight into a slot later. Materializing first meant building a multi-MB
    /// concat per fused group at load and immediately dropping it (and touching every one of those
    /// pages, which for a model that does not fit memory is the cost this whole tier exists to
    /// avoid).
    Concat(Vec<TensorBytes>),
}

impl WBytes {
    /// Byte length, without materializing anything.
    pub(crate) fn len(&self) -> usize {
        match self {
            WBytes::Mmap(tb) => tb.len(),
            WBytes::Owned(v) => v.len(),
            WBytes::Concat(parts) => parts.iter().map(|p| p.len()).sum(),
        }
    }

    /// The components' `(offset, len)` ranges in the model file, in layout order — what a binder
    /// that reads weights itself registers instead of taking the bytes.
    ///
    /// `None` for `Owned` bytes: those exist only because the loader REWROTE them (the qwen2 q/k
    /// row permute, the BitNet dequant), so they correspond to nothing on disk and re-reading the
    /// file would produce the pre-rewrite bytes.
    pub(crate) fn file_ranges(&self) -> Option<Vec<(u64, usize)>> {
        match self {
            WBytes::Mmap(tb) => Some(vec![tb.file_range()]),
            WBytes::Concat(parts) => Some(parts.iter().map(|p| p.file_range()).collect()),
            WBytes::Owned(_) => None,
        }
    }

    /// The bytes themselves. Borrowed for a single mapped tensor or an owned buffer; a `Concat` is
    /// joined HERE, so the cost lands on the binders that genuinely need the bytes and on no one
    /// else.
    pub(crate) fn materialize(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            WBytes::Mmap(tb) => std::borrow::Cow::Borrowed(tb),
            WBytes::Owned(v) => std::borrow::Cow::Borrowed(v),
            WBytes::Concat(parts) => {
                let mut out = Vec::with_capacity(self.len());
                for p in parts {
                    out.extend_from_slice(p);
                }
                std::borrow::Cow::Owned(out)
            }
        }
    }
}

/// Turns a native-dtype GGUF tensor into a backend buffer + the EFFECTIVE dtype it now holds (the
/// GPU binder may convert float weights to f16), so the graph declares the handle to match. The
/// tensor NAME lets a binder place specific tensors differently (the Vulkan binder puts
/// auto-fit-offloaded MoE expert banks in host-visible memory instead of VRAM).
type BindWeight<'a> = dyn Fn(&str, WBytes, DType, usize) -> AResult<(Box<dyn Buffer>, DType)> + 'a;

/// Persistent per-session seam state: the uploaded weights, the KV cache (sized to `max_ctx`
/// once), the per-step IO buffers, and the token ids currently MATERIALIZED in the cache. A caller
/// holding one across `generate_dense_backend` calls gets ChatSession-style KV reuse — each turn
/// prefills only the token suffix that differs from `cached` (the common-prefix diff), so a
/// growing conversation stops re-prefilling its whole history. Pass a fresh `None` for the old
/// one-shot behavior.
/// Byte size of `elems` KV-cache elements stored as `dt`. Q8_0 = 34 bytes / 32-elem block
/// (a 2-byte f16 scale + 32 int8), F16 = 2 bytes, else raw f32. K and V pick their dtype
/// independently, so this is called per-side. Q8_0 is rounded up to a u32 multiple so the Vulkan
/// backend can bind the buffer as a `uint` array (its planar Q8 layout reads codes/scales as words).
/// A quantized KV cache dtype that forces per-execute static decode on the GPU (record-once replay
/// is disabled for it). Must match the adapter's `decode_eligible` rejection — with one pair-wise
/// exception the caller handles: COUPLED Q8_0 (K==V==Q8) replays (store_q8_dyn + the planar-Q8 dyn
/// attention read), so `runner`'s gate checks the pair before consulting this per-side predicate.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn kv_forces_static(dt: DType) -> bool {
    matches!(
        dt,
        DType::Q8_0
            | DType::Q4_0
            | DType::Q4_1
            | DType::Q5_0
            | DType::Q5_1
            | DType::Iq4Nl
            | DType::Turbo2
            | DType::Turbo3
            | DType::Turbo4
            // Dense f32/bf16 caches also un-fuse the K write on the GPU → force static decode.
            | DType::F32
            | DType::Bf16
    )
}

/// Exact KV buffer size for a format — pure format arithmetic, so it lives in the shared seam
/// ([`infr_core::budget::kv_fmt_bytes`], which owns the doc and the pinning tests) next to the
/// per-element rate the placement estimates use. Re-exported under the old crate-private name
/// because every call site here is a `Backend::alloc` argument.
pub(crate) use infr_core::budget::kv_fmt_bytes;

/// gemma4 E2B: gather + dequant this chunk's per-layer TOKEN embedding rows on the host — the ONLY
/// part llama.cpp keeps host-side ("very little benefit to offloading the input layer"); the
/// model_proj GEMV + RMSNorm + combine now run as GPU graph ops (see the E2B prologue in `build`).
/// Returns `pl_tok_scaled[r][l*npl+j] = per_layer_tok_embd[tok_r][l*npl+j] * √npl`, `[rows,
/// n_layer*npl]` row-major — uploaded to `ipl_buf` and bound to the graph Input `pl_tok_in`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn e2b_ipl_rows(g: &Gguf, ple: &PerLayerEmbd, tokens: &[u32]) -> AResult<Vec<f32>> {
    use rayon::prelude::*;
    let (npl, nl) = (ple.npl, ple.n_layer);
    let sqrt_npl = (npl as f32).sqrt();
    let te_bytes = g
        .tensor_bytes("per_layer_token_embd.weight")
        .map_err(|e| anyhow!("{e}"))?;
    let mut out = vec![0f32; tokens.len() * nl * npl];
    out.par_chunks_mut(nl * npl)
        .zip(tokens.par_iter())
        .try_for_each(|(dst, &tok)| -> AResult<()> {
            let tok = tok as usize;
            let r0 = tok * ple.tok_embd_row_bytes;
            let pl_tok = dequant_block(
                ple.tok_embd_dtype,
                &te_bytes[r0..r0 + ple.tok_embd_row_bytes],
            )
            .map_err(|e| anyhow!("{e}"))?;
            for (d, s) in dst.iter_mut().zip(pl_tok.iter()) {
                *d = s * sqrt_npl;
            }
            Ok(())
        })?;
    Ok(out)
}

/// Longest shared prefix of the cached tokens and the new prompt (the KV rows that stay valid).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod bda_cap_tests {
    use super::{
        check_bda_element_cap, chunk_covered_dense_tensor, expert_parallel_guard, tp_bias_guard,
        tp_weight_role, Config, BDA_ELEMENT_UNIT_MAX,
    };

    #[test]
    fn chunk_covered_names_are_exactly_lm_head_and_embed() {
        // Only the output projection / embedding tables (chunk-covered, issue #77) may exceed the
        // element cap — the binder skips check_bda_element_cap for exactly these.
        for n in [
            "output.weight",
            "token_embd.weight",
            "per_layer_token_embd.weight",
        ] {
            assert!(chunk_covered_dense_tensor(n), "{n} must be chunk-covered");
        }
        // A per-layer projection / norm / bias is NOT — it keeps the loud whole-tensor element cap
        // (and never breaches anyway: its element count is orders of magnitude under 2^32).
        for n in [
            "blk.0.attn_q.weight",
            "blk.10.ffn_down.weight",
            "output_norm.weight",
            "blk.0.ffn_gate_exps.weight",
        ] {
            assert!(!chunk_covered_dense_tensor(n), "{n} must stay capped");
        }
    }

    #[test]
    fn element_cap_accepts_realistic_and_boundary_below() {
        // A big-but-real per-expert slice (2112 * 5120 ≈ 10.8M elems) and the last legal count
        // both pass — model reality stays well under the cap.
        check_bda_element_cap(
            "blk.0.ffn_gate_exps.weight",
            "per-expert slice",
            2112 * 5120,
        )
        .expect("realistic expert slice must pass");
        check_bda_element_cap("t", "tensor", BDA_ELEMENT_UNIT_MAX - 1)
            .expect("2^32 - 1 elements is the last legal count");
    }

    #[test]
    fn element_cap_rejects_at_and_above_2p32() {
        // At the cap and above it fail LOUDLY (u32 index would wrap) rather than corrupt output.
        assert!(check_bda_element_cap("t", "tensor", BDA_ELEMENT_UNIT_MAX).is_err());
        assert!(check_bda_element_cap("t", "per-expert slice", BDA_ELEMENT_UNIT_MAX + 7).is_err());
    }

    #[test]
    fn element_cap_resident_expert_bank_uses_per_slice() {
        // A RESIDENT stacked expert bank whose WHOLE-bank element count clears 2^32 but whose
        // per-expert slice stays legal: the binder's `moe_role_of(name)` branch checks the
        // per-expert slice (arena kernels index `arena + expert * stride`), NOT the whole bank —
        // the same relaxation the paged path already made. This is the invariant that branch relies
        // on, and the flag-off resident id kernels that once forced the conservative whole-bank
        // count are gone (weights are u64 BDA-addressed only).
        let n_expert = 512usize;
        let slice = 9_000_000usize; // < 2^32
        let whole = n_expert * slice; // 4.608e9 > 2^32
        assert!(whole > BDA_ELEMENT_UNIT_MAX && slice < BDA_ELEMENT_UNIT_MAX);
        // The old WHOLE-bank cap would (wrongly) reject this legal resident bank...
        assert!(check_bda_element_cap("blk.0.ffn_up_exps.weight", "tensor", whole).is_err());
        // ...but the per-expert-slice cap the resident-bank path now applies accepts it.
        check_bda_element_cap(
            "blk.0.ffn_up_exps.weight",
            "per-expert slice",
            whole / n_expert,
        )
        .expect("resident expert bank per-slice must pass");
    }

    /// Tensor parallelism shards projection MATRICES only. qwen2's q/k/v biases share their
    /// projection's name but are 1-D, so classifying them as column-parallel sent them to the
    /// row slicer and failed with a slice-geometry error; they are replicated, and a model that
    /// has them is refused up front by name.
    #[test]
    fn tp_shards_projection_weights_and_refuses_qkv_biases() {
        let cfg = Config {
            n_layer: 1,
            n_head: 2,
            head_dim: 4,
            n_embd: 8,
            n_ff: 16,
            qkv_bias: true,
            ..Default::default()
        };
        assert!(matches!(
            tp_weight_role("blk.0.attn_q.weight", &cfg),
            Some((infr_vulkan::TpRole::Column, 8))
        ));
        assert!(matches!(
            tp_weight_role("blk.0.attn_output.weight", &cfg),
            Some((infr_vulkan::TpRole::Row, 8))
        ));
        for bias in [
            "blk.0.attn_q.bias",
            "blk.0.attn_k.bias",
            "blk.0.attn_v.bias",
        ] {
            assert!(
                tp_weight_role(bias, &cfg).is_none(),
                "{bias} must be replicated"
            );
        }
        let err = tp_bias_guard(&cfg).expect_err("q/k/v biases must be refused under TP");
        assert!(
            err.to_string().starts_with(
                "INFR_TENSOR_PARALLEL does not support attention q/k/v projection biases"
            ),
            "{err}"
        );
        tp_bias_guard(&Config {
            qkv_bias: false,
            ..cfg
        })
        .expect("a bias-free model is not refused");
    }

    /// gemma4 MoE's dense branch is computed before the MoE op and summed after it, across the cut
    /// expert parallelism makes there, and its per-expert down scale is indexed by the band-local
    /// expert id: both wrong under EP, so it is refused by name before any device is opened.
    #[test]
    fn expert_parallel_refuses_the_dual_moe_ffn() {
        let moe = crate::MoeConfig {
            n_expert: 4,
            n_used: 1,
            n_ff_exp: 16,
            scale: 1.0,
            gating: infr_core::graph::MoeGating::Softmax,
            norm_w: true,
            weight_before: false,
            n_expert_groups: 0,
            n_expert_groups_used: 0,
        };
        let qwen3moe = Config {
            n_layer: 1,
            moe: Some(moe),
            ..Default::default()
        };
        expert_parallel_guard(&qwen3moe).expect("routed-only MoE is EP's own case");
        let gemma4_moe = Config {
            gemma4_moe: true,
            ..qwen3moe
        };
        let err = expert_parallel_guard(&gemma4_moe).expect_err("gemma4 MoE must be refused");
        assert!(
            err.to_string()
                .starts_with("INFR_EXPERT_PARALLEL does not support gemma4 MoE"),
            "{err}"
        );
    }

    #[test]
    fn tp_pipeline_cap_exempts_chunk_covered_enforces_others() {
        // The TP and pipeline binders now gate the whole-tensor cap on `!chunk_covered_dense_tensor`
        // (mirroring EP/resident) — model that exact decision: a >= 2^32-element lm_head / embedding
        // table is EXEMPT (dispatch-chunked reads), a normal huge non-chunk-covered tensor is caught.
        let over_cap = BDA_ELEMENT_UNIT_MAX + 1;
        for chunked in [
            "output.weight",
            "token_embd.weight",
            "per_layer_token_embd.weight",
        ] {
            assert!(chunk_covered_dense_tensor(chunked));
            // The binder's gate: `if !chunk_covered_dense_tensor(name) { check_bda_element_cap(..) }`
            // — chunk-covered means the cap is SKIPPED, so an over-cap table binds fine.
            if !chunk_covered_dense_tensor(chunked) {
                check_bda_element_cap(chunked, "tensor", over_cap).expect("unreachable");
            }
        }
        // A normal (non-chunk-covered) tensor over the cap is still rejected loudly under both.
        assert!(!chunk_covered_dense_tensor("blk.0.attn_q.weight"));
        assert!(check_bda_element_cap("blk.0.attn_q.weight", "tensor", over_cap).is_err());
    }
}

#[cfg(test)]
mod seam_helper_tests {
    use super::{cpu_repack_reserve, Config, DType, EngineConfig, PlacementPins, PlacementScope};

    // NB: `parse_device_spec`'s own cases moved to `infr_core::config::tests` with the function
    // (this crate's duplicate of that grammar was deleted).

    /// `device.ubatch` is TWO readers, and an unusable value must split them (see the
    /// `ubatch_specified` decision recorded on [`super::ubatch_rows`]): `-u 0` / a typo yields no
    /// chunk height yet still counts as "the user pinned one", which is what disables the dense
    /// placement sweeps. Collapsing them onto `Option::is_some` would silently re-enable the sweep.
    #[test]
    fn ubatch_value_and_presence_are_separate_readers() {
        // Own scope: the fallback pins are process-global, and this test asserts the UNPINNED
        // default. (That this is even possible per-scope is the point of `PlacementPins`.)
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let unset = EngineConfig::default();
        assert!(!super::user_pinned_ubatch(&unset));
        assert_eq!(
            super::ubatch_rows(&unset),
            1024,
            "no pin, no iGPU: the 1024 default"
        );

        let pinned = EngineConfig {
            device: infr_core::config::DeviceCfg {
                ubatch: Some(512),
                ubatch_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(super::user_pinned_ubatch(&pinned));
        assert_eq!(super::ubatch_rows(&pinned), 512);

        // `-u 0` / `INFR_UBATCH=abc`: specified, but no usable height.
        let adaptive = EngineConfig {
            device: infr_core::config::DeviceCfg {
                ubatch: Some(0),
                ubatch_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            super::user_pinned_ubatch(&adaptive),
            "an unusable value is still a pin — the sweep must stay off"
        );
        assert_eq!(
            super::ubatch_rows(&adaptive),
            1024,
            "…and the height falls back"
        );
    }

    /// An MoE model whose expert widths the batched expert GEMMs cannot cover runs its prefill on
    /// the small-m expert path, so its chunk is capped at
    /// [`infr_core::tensor::MOE_SMALL_M_CHUNK_CAP`]: the default, a placement pin and an explicit
    /// `INFR_UBATCH` alike, and through every candidate height the fit sweeps price. An aligned
    /// model, fused or split, keeps the default.
    #[test]
    fn a_width_failing_moe_model_gets_the_capped_chunk() {
        use infr_core::tensor::{MOE_EXPERT_GEMM_TILE_N, MOE_SMALL_M_CHUNK_CAP};
        let moe = |n_ff_exp| crate::MoeConfig {
            n_expert: 16,
            n_used: 2,
            n_ff_exp,
            scale: 1.0,
            gating: infr_core::graph::MoeGating::Softmax,
            norm_w: true,
            weight_before: false,
            n_expert_groups: 0,
            n_expert_groups_used: 0,
        };
        let tile = MOE_EXPERT_GEMM_TILE_N;
        let model = |n_embd, n_ff_exp, fused| Config {
            n_layer: 1,
            n_embd,
            moe: Some(moe(n_ff_exp)),
            gemma4_moe: fused,
            ..Default::default()
        };
        let aligned = [
            model(tile, tile, false),
            model(2 * tile, 3 * tile, false),
            // Fused gate|up is one GEMM of twice `n_ff_exp`: half a tile each fills it.
            model(tile, tile / 2, true),
        ];
        let failing = [
            model(tile, tile / 2, false),
            model(tile, 3 * tile / 2, false),
            model(3 * tile / 2, tile, false),
            model(3 * tile / 2, tile / 2, true),
        ];
        let dense = Config {
            n_layer: 1,
            n_embd: tile / 2,
            ..Default::default()
        };
        let unset = EngineConfig::default();
        let explicit = EngineConfig {
            device: infr_core::config::DeviceCfg {
                ubatch: Some(512),
                ubatch_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let default_rows = {
            let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
            super::ubatch_rows(&unset)
        };
        assert!(
            default_rows > MOE_SMALL_M_CHUNK_CAP,
            "the default chunk is already under the cap, so this checks nothing"
        );
        for cfg in aligned.iter().chain([&dense]) {
            assert_eq!(super::moe_small_m_chunk_cap(cfg), None, "{:?}", cfg.moe);
            let pins = PlacementPins::for_model(&Default::default(), cfg);
            let _scope = PlacementScope::enter(std::sync::Arc::new(pins));
            assert_eq!(super::ubatch_rows(&unset), default_rows, "{:?}", cfg.moe);
            assert_eq!(super::ubatch_rows(&explicit), 512, "{:?}", cfg.moe);
        }
        for cfg in &failing {
            assert_eq!(
                super::moe_small_m_chunk_cap(cfg),
                Some(MOE_SMALL_M_CHUNK_CAP),
                "{:?}",
                cfg.moe
            );
            let pins = PlacementPins::for_model(&Default::default(), cfg);
            let _scope = PlacementScope::enter(std::sync::Arc::new(pins));
            assert_eq!(super::ubatch_rows(&unset), MOE_SMALL_M_CHUNK_CAP);
            assert_eq!(super::ubatch_rows(&explicit), MOE_SMALL_M_CHUNK_CAP);
            super::pin_ubatch(256);
            assert_eq!(super::ubatch_rows(&unset), MOE_SMALL_M_CHUNK_CAP);
            assert!(
                super::ubatch_candidates(&unset)
                    .iter()
                    .all(|&c| c <= MOE_SMALL_M_CHUNK_CAP),
                "a fit sweep would price a chunk past the cap"
            );
        }
        // The scope a one-shot Vulkan entry point enters for a model carries the cap too, over a
        // scope the caller made without it.
        let _outer = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cpu = infr_cpu::CpuBackend::new();
        let inner = super::enter_model_scope(&cpu, &failing[0]);
        assert!(inner.is_none(), "an active scope is reused");
        assert_eq!(super::ubatch_rows(&unset), MOE_SMALL_M_CHUNK_CAP);
    }

    /// Two backends of different classes in one process each get their own default prefill chunk:
    /// the class comes from the pins made for each backend, not from whichever device the process
    /// opened first. Interleaved, nested and on two threads at once, as a test
    /// binary runs an integrated-GPU test beside a CPU one.
    #[test]
    fn each_backend_class_gets_its_own_default_chunk() {
        use infr_core::backend::{Backend, Capabilities};
        let integrated = Capabilities {
            integrated: true,
            compute_units: 2,
            ..Default::default()
        };
        let igpu = infr_core::integrated_ubatch_rows(integrated.compute_units);
        let cpu = infr_cpu::CpuBackend::new();
        // What a CPU run took when no Vulkan backend had published a device class.
        assert_eq!(super::default_ubatch_rows(&cpu.capabilities()), 1024);
        assert_ne!(
            igpu, 1024,
            "the two classes share a chunk, so this checks nothing"
        );
        let unset = EngineConfig::default();
        let chunk_for = |caps: &Capabilities| {
            let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::for_caps(caps)));
            super::ubatch_rows(&unset)
        };
        assert_eq!(chunk_for(&integrated), igpu);
        {
            let _cpu = super::enter_backend_scope(&cpu).expect("no scope was active");
            assert_eq!(
                super::ubatch_rows(&unset),
                1024,
                "the CPU took the integrated GPU's chunk"
            );
            assert_eq!(
                chunk_for(&integrated),
                igpu,
                "nested inside the CPU's scope"
            );
            assert_eq!(super::ubatch_rows(&unset), 1024, "after the nested scope");
        }
        let (ig, dc) = std::thread::scope(|s| {
            let ig = s.spawn(|| chunk_for(&integrated));
            let dc = s.spawn(|| chunk_for(&Capabilities::default()));
            (
                ig.join().expect("igpu thread"),
                dc.join().expect("cpu thread"),
            )
        });
        assert_eq!((ig, dc), (igpu, 1024), "on two threads at once");
    }

    /// The `*_specified` rule: an UNRECOGNIZED KV format name still suppresses
    /// auto-q8 (it was supplied) while yielding no dtype, and it is not ring-capable either.
    #[test]
    fn kv_specified_beats_a_parsed_dtype() {
        let unset = EngineConfig::default();
        assert!(
            super::kv_unset(&Config::default(), &unset),
            "nothing supplied ⇒ auto-q8 may fill"
        );

        // `INFR_KV_TYPE_K=nonsense`: specified, no dtype.
        let nonsense = EngineConfig {
            kv: infr_core::config::KvCfg {
                type_k: None,
                type_k_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            !super::kv_unset(&Config::default(), &nonsense),
            "an unparseable name still counts as an explicit choice"
        );

        // The legacy both-sides alias counts too.
        let q8 = EngineConfig {
            kv: infr_core::config::KvCfg {
                force_q8: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(!super::kv_unset(&Config::default(), &q8));
    }

    // ── MLA (deepseek2) KV geometry + dtype gate — docs/backlog.md B41/B42 ───────────────────

    /// The KV-geometry fields of DeepSeek-V2-Lite-Chat, as `cpu_deepseek2_config` prints them off
    /// the real GGUF (`n_layer=27 n_head=16 n_embd=2048 kv_lora_rank=512 qk_rope_dim=64
    /// head_k_mla=128 v_head_dim=128`, and `n_kv == 1` which that test asserts). `head_dim` is
    /// MLA's `key_length_mla` = `head_k_mla + qk_rope_dim`. Every other field stays at its default:
    /// nothing below reads them.
    ///
    /// `n_kv * head_dim` is 192 here and the real cached row is 576, which is exactly why the
    /// open-coded product looked plausible at five call sites while being a third of the truth.
    fn deepseek_v2_lite_kv() -> Config {
        Config {
            deepseek2: true,
            n_layer: 27,
            n_head: 16,
            n_kv: 1,
            head_dim: 192,
            n_embd: 2048,
            kv_lora_rank: 512,
            qk_rope_dim: 64,
            head_k_mla: 128,
            v_head_dim: 128,
            ..Default::default()
        }
    }

    /// The B41 defect in one assertion: an MLA layer's cached row is `kv_lora_rank + qk_rope_dim`
    /// with NO V side, not `n_kv * head_dim` on both sides.
    #[test]
    fn kv_row_elems_mla_is_the_compressed_row_with_no_v_side() {
        let cfg = deepseek_v2_lite_kv();
        let (k, v) = super::kv_row_elems(&cfg, 0);
        assert_eq!(k, cfg.kv_lora_rank + cfg.qk_rope_dim, "MLA K row");
        assert_eq!(k, 576);
        assert_eq!(v, 0, "MLA has no V cache — V is a prefix view of the K row");
        assert_ne!(
            k,
            cfg.layer_n_kv(0) * cfg.layer_head_dim(0),
            "the open-coded product (1 x 192) is what three of the five sites used"
        );
        // Every non-MLA arch keeps the plain per-side product on both sides.
        let dense = qwen3_14b();
        assert_eq!(
            super::kv_row_elems(&dense, 0),
            (
                dense.layer_n_kv(0) * dense.layer_head_dim(0),
                dense.layer_n_kv(0) * dense.layer_head_dim(0)
            )
        );
        // The side an arch does not cache still gets a bindable placeholder, not a zero-size
        // allocation — and a real side is never rewritten.
        assert_eq!(super::kv_side_elems(v), super::KV_PLACEHOLDER_ELEMS);
        assert_eq!(super::kv_side_elems(k), k);
    }

    /// The VRAM estimate must price the row the runner ALLOCATES. Before B41 it priced
    /// `2 x (n_kv x head_dim)` = 384 elements/token/layer against a reality of 576 K-only —
    /// under-reserving by exactly 1.5x, with the context clamp and the resident-fit sweep computing
    /// off that.
    #[test]
    fn the_cpu_repack_cache_is_priced_only_where_it_can_fill() {
        let t = |name: &str, dtype| infr_core::loader::TensorInfo {
            name: name.to_string(),
            shape: vec![1],
            dtype,
            offset: 0,
            nbytes: 1,
        };
        let ec = EngineConfig::default();
        let budget = (ec.kernels.cpu.repack_mb as u64) << 20;
        let fused = [t("blk.0.ffn_gate_up_exps.weight", DType::Q4K)];
        assert_eq!(cpu_repack_reserve(&ec, &fused, true), budget);
        assert_eq!(cpu_repack_reserve(&ec, &fused, false), 0, "no VNNI path");
        let split = [
            t("blk.0.ffn_gate_exps.weight", DType::Q4K),
            t("blk.0.ffn_up_exps.weight", DType::Q4K),
        ];
        assert_eq!(cpu_repack_reserve(&ec, &split, true), 0, "split banks");
        let q6 = [t("blk.0.ffn_gate_up_exps.weight", DType::Q6K)];
        assert_eq!(cpu_repack_reserve(&ec, &q6, true), 0, "not Q4_K");
        let mut small = EngineConfig::default();
        small.kernels.cpu.repack_mb = 7;
        assert_eq!(cpu_repack_reserve(&small, &fused, true), 7 << 20);
    }

    #[test]
    fn kv_bytes_estimate_prices_the_mla_row_it_allocates() {
        let cfg = deepseek_v2_lite_kv();
        let ec = EngineConfig::default();
        let (ctx, ubatch) = (4096usize, super::ubatch_rows(&ec));
        let est = super::kv_bytes_estimate_fmt(
            &cfg,
            ctx,
            false,
            ubatch,
            super::KvAlign::NONE,
            DType::F16,
            DType::F16,
        );
        // Per layer: the context's rows of the compressed K row in f16, and the V placeholder.
        let k_side = ctx * (cfg.kv_lora_rank + cfg.qk_rope_dim) * 2;
        let want = (cfg.n_layer * (k_side + super::kv_side_bytes(DType::F16, 0))) as u64;
        assert_eq!(est, want);
        // The pre-fix pricing, spelled out: K+V at the head-dim product.
        let old = ctx * 2 * (cfg.n_kv * cfg.head_dim) * 2;
        assert_eq!(k_side, old * 3 / 2, "the estimate was 1.5x short");
    }

    /// deepseek4 stores its K=V row once (`build_attn_mha(q, k_all, k_all, …)`), so the footprint
    /// is the K side alone and the V side is the bindable placeholder. Pinned at the V4-Flash
    /// geometry (`deepseek-ai/DeepSeek-V4-Flash` `config.json`: 43 layers, head dim 512, MQA,
    /// window 128) from hand-computed byte counts; a V side priced like K doubles every figure.
    #[test]
    fn kv_footprint_of_deepseek4_is_the_k_side_alone() {
        let cfg = Config {
            n_layer: 43,
            n_embd: 4096,
            n_head: 64,
            n_kv: 1,
            head_dim: 512,
            swa_window: 128,
            deepseek4: true,
            ..Default::default()
        };
        assert_eq!(super::kv_row_elems(&cfg, 0), (512, 0));
        // Full context 4096: 4096 rows x 512 elements x 2 bytes (f16) of K and the 8-byte V
        // placeholder, x 43 layers.
        let full = super::kv_bytes_estimate_fmt(
            &cfg,
            4096,
            false,
            512,
            super::KvAlign::NONE,
            DType::F16,
            DType::F16,
        );
        assert_eq!(full, 180_355_416);
        // Ring: min(4096, (128 + 512) rounded up to 64) = 640 rows per layer.
        let ring = super::kv_bytes_estimate_fmt(
            &cfg,
            4096,
            true,
            512,
            super::KvAlign::NONE,
            DType::F16,
            DType::F16,
        );
        assert_eq!(ring, 28_180_824);
        // The allocation: 4096 x 512 x 2 bytes of K, and the placeholder V (4 f16 elements).
        let alloc =
            super::kv_layer_alloc_bytes(&cfg, 0, 4096, false, 512, 1, DType::F16, DType::F16);
        assert_eq!(alloc, (4_194_304, 8));
    }

    /// The V4-Flash footprint with its compressed tiers, from hand-computed byte counts at the
    /// real file's geometry (`compress_ratios` `[0, 0, 4, 128, 4, 128, …, 4]` over 43 layers: 21
    /// CSA, 20 HCA, 2 ratio-0; head dim 512, indexer key width 128, window 128) at a 4096-token
    /// f16 context with full-context raw caches. A compressed buffer left out of the estimate,
    /// sized at another row count or ring height, or priced in another dtype changes the total.
    #[test]
    fn kv_footprint_of_deepseek4_flash_counts_its_compressed_tiers() {
        let ratios: Vec<usize> = (0..43)
            .map(|l| match l {
                0 | 1 => 0,
                l if l % 2 == 0 => 4,
                _ => 128,
            })
            .collect();
        let cfg = Config {
            n_layer: 43,
            n_embd: 4096,
            n_head: 64,
            n_kv: 1,
            head_dim: 512,
            swa_window: 128,
            indexer_head_size: 128,
            compress_ratios: ratios,
            deepseek4: true,
            ..Default::default()
        };
        // An HCA layer: 4096/128 = 32 rows x 512 x 2 bytes of cache, and a ring of 128 rows x 512
        // f32, kv and score.
        let hca: Vec<usize> = super::dsv4_comp_bufs(&cfg, 3, 4096, DType::F16)
            .iter()
            .map(super::Dsv4CompBuf::bytes)
            .collect();
        assert_eq!(hca, [32_768, 262_144, 262_144]);
        // A CSA layer: 4096/4 = 1024 rows x 512 x 2 bytes, a ring of 8 rows x 1024 f32 (kv,
        // score); then the indexer's 1024 rows x 128 x 2 bytes and its ring of 8 rows x 256 f32.
        let csa: Vec<usize> = super::dsv4_comp_bufs(&cfg, 2, 4096, DType::F16)
            .iter()
            .map(super::Dsv4CompBuf::bytes)
            .collect();
        assert_eq!(csa, [1_048_576, 32_768, 32_768, 262_144, 8_192, 8_192]);
        assert!(super::dsv4_comp_bufs(&cfg, 1, 4096, DType::F16).is_empty());
        // The raw caches as before (`kv_footprint_of_deepseek4_is_the_k_side_alone`: 180_355_416),
        // plus 20 HCA layers x 557_056 = 11_141_120 and 21 CSA layers x 1_392_640 = 29_245_440.
        let full = super::kv_bytes_estimate_fmt(
            &cfg,
            4096,
            false,
            512,
            super::KvAlign::NONE,
            DType::F16,
            DType::F16,
        );
        assert_eq!(full, 220_741_976);
        // The pre-load floor lists each compressed buffer as an allocation of its own beside the
        // layer's K/V pair: 43 pairs, 20 x 3 HCA buffers and 21 x 6 CSA ones.
        let floor = super::kv_floor_alloc_bytes(
            &cfg,
            &EngineConfig::default(),
            4096,
            1,
            1,
            DType::F16,
            DType::F16,
        );
        assert_eq!(floor.len(), 43 + 20 * 3 + 21 * 6);
    }

    /// TurboQuant's 128-element groups must tile every V4 cache row, the compressed ones too: a
    /// V4-Flash geometry passes, and the same geometry with a 64-wide indexer (an LID row of half a
    /// group) does not, though its raw head would.
    #[test]
    fn kv_turbo_layout_holds_deepseek4_compressed_rows() {
        let cfg = |indexer_head_size| Config {
            n_layer: 3,
            head_dim: 512,
            indexer_head_size,
            compress_ratios: vec![0, 4, 128],
            deepseek4: true,
            ..Default::default()
        };
        assert!(super::kv_turbo_align_ok(&cfg(128)));
        assert!(!super::kv_turbo_align_ok(&cfg(64)));
        let ratio0 = Config {
            compress_ratios: vec![0, 0, 0],
            ..cfg(64)
        };
        assert!(super::kv_turbo_align_ok(&ratio0));
    }

    /// deepseek4 has no V cache, so a V format setting must not gate anything for it: not the SWA
    /// ring (`kv_ring_wanted`, `placement_ring`), not auto-q8 (`kv_unset`). The same settings on a
    /// model with a real V cache keep gating, which is what shows the predicate is doing the work.
    #[test]
    fn v_kv_type_gates_nothing_for_deepseek4() {
        let dsv4 = Config {
            n_layer: 2,
            n_kv: 1,
            head_dim: 128,
            swa_window: 128,
            deepseek4: true,
            ..Default::default()
        };
        let swa = Config {
            deepseek4: false,
            ..dsv4.clone()
        };
        let v_only = |dt| EngineConfig {
            kv: infr_core::config::KvCfg {
                type_v: Some(dt),
                type_v_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        for dt in [DType::Turbo3, DType::Q4_0, DType::F32] {
            let ec = v_only(dt);
            assert!(super::kv_ring_wanted(&dsv4, &ec), "{dt:?}: ring gated");
            assert!(
                super::placement_ring(&dsv4, &ec, DType::F16, dt),
                "{dt:?}: placement ring gated"
            );
            assert_eq!(super::kv_v_fmt(&dsv4, DType::F16, dt), DType::F16);
            assert!(super::kv_unset(&dsv4, &ec), "{dt:?}: auto-q8 suppressed");
            // The control: a V cache that exists is gated by its format.
            assert!(!super::kv_ring_wanted(&swa, &ec), "{dt:?}: control ring");
            assert!(!super::placement_ring(&swa, &ec, DType::F16, dt));
            assert_eq!(super::kv_v_fmt(&swa, DType::F16, dt), dt);
            assert!(!super::kv_unset(&swa, &ec));
        }
    }

    /// A recurrent layer's state does not grow with the context, so the estimate prices it at the
    /// fixed size the runner allocates (`Config::recurrent_state_elems`), while an attention layer
    /// beside it still scales with the rows. The state is priced twice: the live copy and the one
    /// snapshot a slot can hold beside it (`SeamKv::mtp_delta_ckpt`, shared by MTP rollback and the
    /// prefill-boundary snapshot), whose allocation would otherwise reach past a context fitted
    /// to fill VRAM.
    #[test]
    fn kv_bytes_estimate_prices_recurrent_state_at_its_fixed_size() {
        let cfg = Config {
            n_layer: 2,
            n_embd: 64,
            n_head: 4,
            n_kv: 2,
            head_dim: 16,
            lfm3: true,
            shortconv_layers: vec![true, false],
            shortconv_l_cache: 4,
            ..Default::default()
        };
        let est = |ctx| {
            super::kv_bytes_estimate_fmt(
                &cfg,
                ctx,
                false,
                512,
                super::KvAlign::NONE,
                DType::F16,
                DType::F16,
            )
        };
        let state = ((cfg.shortconv_l_cache - 1) * cfg.n_embd + 1) * 4;
        for ctx in [1024usize, 4096] {
            let attn = ctx * 2 * (cfg.n_kv * cfg.head_dim) * 2;
            assert_eq!(est(ctx), (2 * state + attn) as u64, "ctx {ctx}");
        }
    }

    /// qwen35's MTP rollback snapshots the DeltaNet state of every recurrent layer (conv history
    /// and `S`) and fails the request when that allocation is refused, so the estimate must hold
    /// room for the copy: two of each recurrent layer's state, one attention layer's rows.
    #[test]
    fn kv_bytes_estimate_prices_the_qwen35_snapshot_copy() {
        let cfg = qwen35_small();
        let recurrent: Vec<usize> = (0..cfg.n_layer)
            .filter(|&l| cfg.is_recurrent_layer(l))
            .collect();
        assert_eq!(
            recurrent,
            [0, 1, 2],
            "three DeltaNet layers, then attention"
        );
        let (k, v) = cfg.recurrent_state_elems(0).expect("a DeltaNet layer");
        let ctx = 2048usize;
        let attn = ctx * 2 * (cfg.n_kv * cfg.head_dim) * 2;
        let want = recurrent.len() * 2 * (k + v) * 4 + attn;
        let est = super::kv_bytes_estimate_fmt(
            &cfg,
            ctx,
            false,
            512,
            super::KvAlign::NONE,
            DType::F16,
            DType::F16,
        );
        assert_eq!(est, want as u64);
    }

    /// `kv_q8_layout_ok` is the "may this model use a q8 cache" gate the Vulkan auto-q8 placement
    /// PIN consults, so it must reject MLA — the pin is priced into the VRAM estimate, and
    /// `generate_dense_backend` will build an f16 cache for MLA no matter what was pinned. The
    /// LAYOUT question underneath it is separate and answers yes (576 is 18 whole 32-blocks),
    /// which is what keeps the CPU backend — whose `Op::Mla` dequantizes every KV dtype — able to
    /// honor an explicit q8 request.
    #[test]
    fn q8_gate_rejects_mla_even_though_its_rows_are_block_aligned() {
        let mla = deepseek_v2_lite_kv();
        assert!(super::kv_row_align_ok(&mla), "576 % 32 == 0");
        assert!(!super::kv_q8_layout_ok(&mla), "MLA may not be pinned to q8");
        let dense = qwen3_14b();
        assert!(super::kv_row_align_ok(&dense) && super::kv_q8_layout_ok(&dense));
    }

    /// B42: the GPU MLA kernels read the cache as f16 unconditionally, so a deepseek2 session on a
    /// GPU backend is f16 on both sides — a NAMED non-f16 format is refused rather than silently
    /// downgraded, and the CPU backend (dtype-correct `Op::Mla`) is untouched.
    #[test]
    fn mla_kv_fmt_forces_f16_on_gpu_and_refuses_a_named_format() {
        let mla = deepseek_v2_lite_kv();
        let unset = EngineConfig::default();
        let q8 = |k: Option<DType>, v: Option<DType>, force: bool| EngineConfig {
            kv: infr_core::config::KvCfg {
                type_k: k,
                type_k_specified: k.is_some(),
                type_v: v,
                type_v_specified: v.is_some(),
                force_q8: force,
                ..Default::default()
            },
            ..Default::default()
        };

        // Nothing named: whatever the ladder resolved is forced to f16 (this is the shape an
        // auto-q8 placement pin would arrive in, which `kv_q8_layout_ok` also prevents upstream).
        for be in ["vulkan", "metal"] {
            assert_eq!(
                super::mla_kv_fmt(&mla, be, &unset, DType::Q8_0, DType::Q8_0).expect("forced"),
                (DType::F16, DType::F16)
            );
        }
        // Named non-f16 — per side, and through the legacy both-sides alias — is refused.
        for ec in [
            q8(Some(DType::Q8_0), None, false),
            q8(None, Some(DType::Turbo3), false),
            q8(None, None, true),
        ] {
            let err = super::mla_kv_fmt(&mla, "vulkan", &ec, DType::F16, DType::F16)
                .expect_err("must refuse");
            assert!(
                err.to_string().contains("f16-only"),
                "unhelpful refusal: {err}"
            );
        }
        // f16 asked for explicitly is not a refusal.
        assert_eq!(
            super::mla_kv_fmt(
                &mla,
                "vulkan",
                &q8(Some(DType::F16), Some(DType::F16), false),
                DType::F16,
                DType::F16
            )
            .expect("f16 is what MLA runs"),
            (DType::F16, DType::F16)
        );
        // CPU keeps its dtype freedom (its MLA arm dequantizes), and a non-MLA model is untouched
        // on every backend.
        assert_eq!(
            super::mla_kv_fmt(
                &mla,
                "cpu",
                &q8(Some(DType::Q8_0), Some(DType::Q8_0), false),
                DType::Q8_0,
                DType::Q8_0
            )
            .expect("cpu passthrough"),
            (DType::Q8_0, DType::Q8_0)
        );
        assert_eq!(
            super::mla_kv_fmt(&qwen3_14b(), "vulkan", &unset, DType::Q8_0, DType::Q8_0)
                .expect("non-MLA passthrough"),
            (DType::Q8_0, DType::Q8_0)
        );
    }

    // ── context-fit math (`kv_fit_ctx_for`) ──────────────────────────────────────────────────
    //
    // GPU-free: the function takes the weight footprint and the device's byte figures as plain
    // arguments, so every case below is exact arithmetic. Model geometry and weight footprints are
    // read off the real GGUFs (`Config::from_gguf` + `weights::weight_footprint`); the byte figures
    // are this box's 7900 XTX (`/sys/class/drm/card1/device/mem_info_vram_total` = 25 753 026 560,
    // live free ~23.94 GiB on an idle desktop).

    /// RX 7900 XTX, 24 GB.
    const XTX_TOTAL: u64 = 25_753_026_560;
    /// Live FREE bytes on an idle XTX (~23.94 GiB) — the raw `VramInfo::available` figure, which is
    /// NOT the budget: see [`XTX_ROOM`].
    const XTX_FREE: u64 = 25_701_257_216;
    /// What `VramInfo::alloc_room()` yields for that snapshot: live free minus the allocator
    /// guard's own 256 MiB headroom. Anything a fit or placement decision plans past this the VRAM
    /// guard will refuse, so this — not the raw free figure — is the budget.
    const XTX_ROOM: u64 = XTX_FREE - 256 * 1024 * 1024;

    /// That box's VRAM snapshot at an arbitrary free figure, as the backend would report it.
    fn xtx(available: u64) -> infr_vulkan::VramInfo {
        infr_vulkan::VramInfo {
            total: XTX_TOTAL,
            available,
            live: true,
            uma: false,
            limited_from: None,
        }
    }

    /// gemma-3-12b-it-Q4_K_M — the reported case. Geometry from the GGUF metadata.
    fn gemma3_12b() -> Config {
        Config {
            n_layer: 48,
            n_head: 16,
            n_kv: 8,
            n_kv_swa: 8,
            head_dim: 256,
            head_dim_swa: 256,
            n_embd: 3840,
            n_ff: 15360,
            swa_window: 1024,
            swa_pattern: 6,
            n_ctx_train: 131072,
            ..Default::default()
        }
    }
    /// `weights::weight_footprint` of that GGUF.
    const GEMMA3_12B_WEIGHTS: u64 = 7_292_694_912;

    /// A plain dense model with NO sliding window and head_dim 128 — the flash tier, so
    /// `dense_act_reserve_at`'s score-tile term is zero and the KV grows strictly linearly.
    /// Qwen3-14B's geometry.
    fn qwen3_14b() -> Config {
        Config {
            n_layer: 40,
            n_head: 40,
            n_kv: 8,
            head_dim: 128,
            n_embd: 5120,
            n_ff: 17408,
            n_ctx_train: 40960,
            ..Default::default()
        }
    }

    /// Re-derive the accept predicate from the shared primitives and assert `fit` is EXACTLY its
    /// boundary: it fits, and one more token does not. Returns nothing — it panics with detail.
    fn assert_exact_boundary(
        cfg: &Config,
        ec: &EngineConfig,
        weights: u64,
        room: u64,
        k: DType,
        v: DType,
        fit: usize,
    ) {
        let ring = super::kv_ring_wanted(cfg, ec)
            && matches!(k, DType::F16 | DType::Q8_0)
            && matches!(v, DType::F16 | DType::Q8_0);
        let need = |ctx: usize, ub: usize| {
            weights
                + super::kv_bytes_estimate_fmt(cfg, ctx, ring, ub, super::KvAlign::NONE, k, v)
                + super::dense_act_reserve_at(cfg, ctx, ub)
        };
        let cands = super::ubatch_candidates(ec);
        let best = |ctx: usize| cands.iter().map(|&ub| need(ctx, ub)).min().expect("ladder");
        assert!(
            best(fit) <= room,
            "fit {fit} must FIT: cheapest need {} > room {room}",
            best(fit)
        );
        assert!(
            best(fit + 1) > room,
            "fit {fit} must be the LARGEST: {} tokens also fits (need {} <= room {room})",
            fit + 1,
            best(fit + 1)
        );
    }

    /// The linear (no-ring) branch, both element sizes: the returned context is the exact largest
    /// that fits, not an approximation of it. A `0.95` haircut or a `saturating_sub(64)` on the
    /// RESULT — what this replaced — fails the second half of the boundary check.
    #[test]
    fn kv_fit_linear_is_the_exact_largest_context() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = qwen3_14b();
        let ec = EngineConfig::default();
        let weights = 9 * (1u64 << 30);
        for (k, v) in [(DType::F16, DType::F16), (DType::Q8_0, DType::Q8_0)] {
            let fit = super::kv_fit_ctx_for(
                &cfg,
                &ec,
                weights,
                &xtx(XTX_FREE),
                super::KvAlign::NONE,
                k,
                v,
            )
            .expect("has KV");
            assert_exact_boundary(&cfg, &ec, weights, XTX_ROOM, k, v, fit);
        }
        // q8 is ~half the bytes per token, so it must buy materially more context than f16.
        let f16 = super::kv_fit_ctx_for(
            &cfg,
            &ec,
            weights,
            &xtx(XTX_FREE),
            super::KvAlign::NONE,
            DType::F16,
            DType::F16,
        )
        .expect("has KV");
        let q8 = super::kv_fit_ctx_for(
            &cfg,
            &ec,
            weights,
            &xtx(XTX_FREE),
            super::KvAlign::NONE,
            DType::Q8_0,
            DType::Q8_0,
        )
        .expect("has KV");
        assert!(q8 > f16, "q8 {q8} must beat f16 {f16}");
        assert!(q8 < 2 * f16, "q8 {q8} is ~2x f16 {f16}, not more");
    }

    /// The SWA-ring branch, both element sizes. A window layer's KV stops growing with context
    /// once the ring caps its rows, so the fit is enormously larger than the same model priced
    /// without the ring — and it is still the EXACT boundary.
    #[test]
    fn kv_fit_swa_ring_is_exact_and_much_larger_than_linear() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = gemma3_12b();
        let no_ring = EngineConfig {
            kv: infr_core::config::KvCfg {
                ring: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let ring = EngineConfig::default();
        assert!(super::kv_ring_wanted(&cfg, &ring) && !super::kv_ring_wanted(&cfg, &no_ring));
        for (k, v) in [(DType::F16, DType::F16), (DType::Q8_0, DType::Q8_0)] {
            let a = super::kv_fit_ctx_for(
                &cfg,
                &ring,
                GEMMA3_12B_WEIGHTS,
                &xtx(XTX_FREE),
                super::KvAlign::NONE,
                k,
                v,
            )
            .expect("has KV");
            let b = super::kv_fit_ctx_for(
                &cfg,
                &no_ring,
                GEMMA3_12B_WEIGHTS,
                &xtx(XTX_FREE),
                super::KvAlign::NONE,
                k,
                v,
            )
            .expect("has KV");
            assert_exact_boundary(&cfg, &ring, GEMMA3_12B_WEIGHTS, XTX_ROOM, k, v, a);
            assert_exact_boundary(&cfg, &no_ring, GEMMA3_12B_WEIGHTS, XTX_ROOM, k, v, b);
            assert!(
                a > b * 4,
                "40 of 48 layers ring: {a} should dwarf the full-cache fit {b}"
            );
        }
    }

    /// **Regression, the reported case.** gemma-3-12b at its trained 131072 window on a 24 GiB
    /// XTX fits at f16 — it never needed the q8 cache the clamp used to pin. What used to make it
    /// miss is priced here explicitly: the DEFAULT 1024-row prefill chunk's activation reserve
    /// does not fit, a SHORTER rung of the same ladder the dense placement sweep walks does, and
    /// the sweep would have shrunk to it anyway. Pricing only the default chunk decided the KV
    /// format against an assumption the very next step abandoned.
    ///
    /// Deliberately does not name the winning rung. Which one it is moves with the reserve's own
    /// coefficients and that is not what this is guarding — the invariant is "the default chunk
    /// misses, a lower rung on the SHARED ladder saves it, and f16 therefore reaches the trained
    /// window".
    ///
    /// (Whole-run confirmation is `infr bench -d 120000` on the real model, which peaks at
    /// 17.5 GiB of 24.0 GiB — GPU + 8 GiB of weights, so not something a unit test can host.)
    #[test]
    fn kv_fit_walks_the_placement_chunk_ladder_gemma3_12b() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = gemma3_12b();
        let ec = EngineConfig::default();
        let want = cfg.n_ctx_train;
        let need = |ub: usize| {
            GEMMA3_12B_WEIGHTS
                + super::kv_bytes_estimate_fmt(
                    &cfg,
                    want,
                    true,
                    ub,
                    super::KvAlign::NONE,
                    DType::F16,
                    DType::F16,
                )
                + super::dense_act_reserve_at(&cfg, want, ub)
        };
        let cands = super::ubatch_candidates(&ec);
        assert_eq!(cands[0], 1024, "the default chunk leads the ladder");
        // With the measured reserve this model now fits at the DEFAULT chunk — it no longer needs
        // a shorter rung to reach its trained window, which is a strictly better outcome than the
        // one this test was written for (and matches the device: `infr bench -p 131056` runs at
        // ubatch 1024, 780 t/s, peaking 4735 MiB of activations against a 7146 MiB reserve).
        // What still has to hold is that SOME rung of the shared ladder serves it.
        assert!(
            cands.iter().any(|&ub| need(ub) <= XTX_ROOM),
            "a rung of the shared ladder must serve the trained window: {:?}",
            cands.iter().map(|&ub| (ub, need(ub))).collect::<Vec<_>>()
        );

        let fit = super::kv_fit_ctx_for(
            &cfg,
            &ec,
            GEMMA3_12B_WEIGHTS,
            &xtx(XTX_FREE),
            super::KvAlign::NONE,
            DType::F16,
            DType::F16,
        )
        .expect("has KV");
        assert!(
            fit >= want,
            "f16 must reach the trained window ({want}); got {fit} — the auto-q8 rung would fire"
        );
    }

    /// The reserve is the model documented on `dense_act_reserve_at`, term by term, at
    /// gemma-4-31B's shape and a 128-row chunk — so changing a coefficient, or dropping the pad,
    /// fails HERE rather than as a `VRAM budget exceeded` on someone's deep prefill.
    ///
    /// The two numbers this pins that a reader is most likely to "simplify": the score tile is ONE
    /// live pool (`2 * n_head * ctx_pad`), not two, and there is no fixed byte term — the
    /// non-activation slop a 256 MiB constant used to stand in for is measured by
    /// `reclamp_ctx_to_live_room` instead of estimated twice.
    #[test]
    fn act_reserve_is_the_measured_model() {
        let cfg = Config {
            n_head: 32,
            head_dim: 512,
            head_dim_swa: 256,
            swa_window: 1024,
            swa_pattern: 6,
            n_embd: 5376,
            n_ff: 21504,
            ..Default::default()
        };
        let (ctx, ubatch) = (16384usize, 128usize);
        let rows: u64 = 128; // already a multiple of 64
        let attn_pv: u64 = 32 * 32 * 768; // n_head x (head_dim + head_dim_swa), both shapes live
        let attn_s: u64 = 2 * 32 * 16384; // ONE non-flash score tile: hd 512 misses the flash tier
        let per_row = 12 * 21504 + 96 * 5376 + attn_pv + attn_s;
        let unpadded = rows * per_row;
        let got = super::dense_act_reserve_at(&cfg, ctx, ubatch);
        assert_eq!(
            got,
            unpadded * super::ACT_RESERVE_PAD.0 / super::ACT_RESERVE_PAD.1,
            "reserve must be the per-row model times the pad"
        );
        assert_eq!(got, 500_957_184);
        // No fixed term: halving the rows must halve the reserve, which a constant would floor.
        let half = super::dense_act_reserve_at(&cfg, ctx, 64);
        assert_eq!(half * 2, got, "the reserve is purely per-row");
    }

    /// The ladder is ONE list. Both readers — `vulkan_moe_binder`'s residency / auto-q8 /
    /// streaming sweeps and `kv_fit_ctx_for` — call [`super::ubatch_candidates`], so a rung added
    /// or removed cannot reach one and miss the other. This pins its shape, including the two
    /// rules that are easy to lose in a refactor: the current height leads, and a user-pinned
    /// `INFR_UBATCH` collapses the list to that one height.
    #[test]
    fn dense_ubatch_ladder_is_the_only_one() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        assert_eq!(super::DENSE_UBATCH_LADDER, [512, 256, 128]);
        let unset = EngineConfig::default();
        assert_eq!(super::ubatch_rows(&unset), 1024);
        assert_eq!(super::ubatch_candidates(&unset), vec![1024, 512, 256, 128]);

        // Rungs at or above the current height are filtered out — a SHRINK ladder must never
        // raise an integrated GPU past its watchdog-safe default.
        let pinned = |rows: usize| EngineConfig {
            device: infr_core::config::DeviceCfg {
                ubatch: Some(rows),
                ubatch_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(super::ubatch_candidates(&pinned(256)), vec![256]);
        assert_eq!(super::ubatch_candidates(&pinned(2048)), vec![2048]);
    }

    /// **Drift guard.** Every VRAM budget in this file — the residency predicate, the
    /// streaming budget, the MoE expert budget — must be taken against the ALLOCATOR's ceiling
    /// (`VramInfo::alloc_room` = free minus the guard's 256 MiB headroom), never the raw free
    /// figure. The placement sweeps used to compare against `vram.available`, so they could declare
    /// a model resident, or hand a pager an arena, 256 MiB past anything `check_vram_budget` will
    /// ever allocate — which surfaces as a failed activation alloc mid-prefill.
    ///
    /// Each assertion below is placed ONE BYTE either side of the ceiling, so restoring any
    /// `vram.available` comparison flips it: the raw figure accepts every "must not" case here.
    #[test]
    fn budgets_agree_with_the_allocator_ceiling() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let vram = xtx(XTX_FREE);
        const GUARD: u64 = 256 * 1024 * 1024;
        assert_eq!(vram.alloc_room(), XTX_ROOM);
        assert_eq!(
            vram.available - vram.alloc_room(),
            GUARD,
            "the ceiling is the free figure minus the guard headroom"
        );

        let cfg = gemma3_12b();
        let ec = EngineConfig::default();
        let (ctx, ub) = (32768usize, 256usize);
        let f16 = (DType::F16, DType::F16);
        // Weights that make a resident session land EXACTLY on the ceiling (KV + reserve priced
        // from the shared primitives, weights = whatever is left of the ceiling).
        let kv_and_act =
            super::dense_resident_need(&cfg, 0, ctx, true, ub, super::KvAlign::NONE, f16.0, f16.1);
        let exact = XTX_ROOM - kv_and_act;
        let fits = |w: u64| {
            super::dense_placement_fits(
                &cfg,
                &ec,
                w,
                &vram,
                ctx,
                ub,
                super::KvAlign::NONE,
                f16.0,
                f16.1,
            )
        };
        assert!(
            fits(exact),
            "a session that exactly fills the ceiling is resident"
        );
        assert!(
            !fits(exact + 1),
            "one byte PAST the guard's ceiling must not be placed resident"
        );
        assert!(
            exact + 1 + kv_and_act <= vram.available,
            "…and that byte is one the raw free figure would have accepted, which is the bug"
        );

        // Streaming budget: exhausted at the ceiling, and it never offers the guard's headroom.
        // A streamed session also prefills layer-major, whose whole-prompt residual stream is
        // allocated AFTER the arenas out of the same VRAM — so the budget holds that back too.
        let lm = super::layer_major_act_bytes(&cfg, ctx, ub);
        assert!(
            lm > 0,
            "the layer-major term must be a real subtraction here"
        );
        let budget = |w: u64| {
            super::dense_stream_budget_at(
                &cfg,
                &ec,
                w,
                &vram,
                ctx,
                ub,
                super::KvAlign::NONE,
                f16.0,
                f16.1,
            )
        };
        assert_eq!(
            budget(exact),
            0,
            "nothing left to stream into at the ceiling"
        );
        assert_eq!(
            budget(exact - (1 << 30)),
            (1 << 30) - lm,
            "and it is exactly the room below the ceiling less the residual stream, not the free \
             figure"
        );
        // Forcing chunk-major back on gives those bytes to the arena, which is the knob's point.
        let chunk_major = EngineConfig {
            paging: infr_core::config::PagingCfg {
                layer_major: Some(false),
                ..ec.paging.clone()
            },
            ..ec.clone()
        };
        assert_eq!(
            super::dense_stream_budget_at(
                &cfg,
                &chunk_major,
                exact - (1 << 30),
                &vram,
                ctx,
                ub,
                super::KvAlign::NONE,
                f16.0,
                f16.1
            ),
            1 << 30,
            "paging.layer_major = false takes the residual-stream reserve back off the budget"
        );

        // MoE expert placement: same ceiling, minus the pager's own headroom.
        assert_eq!(
            super::moe_expert_budget(&vram, 0, 0),
            Some(XTX_ROOM - super::MOE_ACT_HEADROOM)
        );
        assert_eq!(super::moe_expert_budget(&vram, XTX_ROOM, 0), Some(0));
        assert_eq!(
            super::moe_expert_budget(&vram, XTX_ROOM + 1, 0),
            None,
            "a dense half past the ceiling is a hard error, not a 256 MiB overdraft"
        );
        assert!(
            XTX_ROOM < vram.available,
            "…again a case the raw free figure would have waved through"
        );
    }

    /// Segments are runs of whole consecutive chunks of at most `max_rows` rows, a chunk taller
    /// than that alone, every chunk in exactly one, in order.
    #[test]
    fn layer_major_segments_take_whole_chunks_up_to_the_cap() {
        let chunks = [(0, 37), (37, 74), (74, 111), (111, 148), (148, 150)];
        assert_eq!(super::layer_major_segments(&chunks, usize::MAX), vec![0..5]);
        assert_eq!(
            super::layer_major_segments(&chunks, 80),
            vec![0..2, 2..5],
            "two 37-row chunks fit 80 rows, a third does not; 37 + 37 + 2 do"
        );
        assert_eq!(
            super::layer_major_segments(&chunks, 74),
            vec![0..2, 2..4, 4..5],
            "a segment may fill the cap exactly"
        );
        assert_eq!(
            super::layer_major_segments(&chunks, 10),
            vec![0..1, 1..2, 2..3, 3..4, 4..5],
            "a chunk taller than the cap is a segment alone"
        );
        assert!(super::layer_major_segments(&[], 10).is_empty());
    }

    /// The automatic segment holds what [`super::LAYER_MAJOR_CARRY_BYTES`] carries: at DeepSeek
    /// V4-Flash's geometry (`n_embd` 4096, four hyper-connection streams) a row costs the four
    /// streams and the embedded input row in f32 plus its id and position, 81 928 bytes, so a
    /// GiB holds 13 105 rows. Every other family carries the whole prompt unless
    /// `paging.layer_major_rows` says otherwise, and that setting wins on V4 too.
    #[test]
    fn layer_major_segments_carry_one_gib_at_v4_flash_geometry() {
        let v4 = Config {
            deepseek4: true,
            hc_mult: 4,
            n_embd: 4096,
            ..Config::default()
        };
        assert_eq!(super::layer_major_row_bytes(&v4), 5 * 4096 * 4 + 8);
        let ec = EngineConfig::default();
        assert_eq!(super::layer_major_segment_rows(&ec, &v4), 13_105);
        let dense = Config {
            n_embd: 4096,
            ..Config::default()
        };
        assert_eq!(super::layer_major_segment_rows(&ec, &dense), usize::MAX);
        let mut capped = EngineConfig::default();
        capped.paging.layer_major_rows = Some(512);
        assert_eq!(super::layer_major_segment_rows(&capped, &v4), 512);
        assert_eq!(super::layer_major_segment_rows(&capped, &dense), 512);
        let min = super::HOST_LAYER_MAJOR_MIN_ROWS;
        assert!(super::host_layer_major(&v4, min, false));
        assert!(super::host_layer_major(&v4, 1024, false));
        assert!(!super::host_layer_major(&v4, min - 1, false));
        assert!(!super::host_layer_major(&v4, 32, false));
        assert!(!super::host_layer_major(&dense, 1024, false));
        // A request sharing the backend (`infr serve --parallel`) keeps chunk-major at any height,
        // its own parallel chunk height included.
        let parallel = super::ubatch_rows_parallel(&ec);
        assert!(!super::host_layer_major(&v4, parallel, true));
        assert!(!super::host_layer_major(&v4, 1024, true));
    }

    /// The per-prompt half of the layer-major gate: auto runs it only where it can save a read
    /// (several chunks over a cache that cannot hold every block), and `true` forces it anyway.
    #[test]
    fn layer_major_runs_only_where_it_saves_a_read() {
        let with = |lm: Option<bool>| EngineConfig {
            paging: infr_core::config::PagingCfg {
                layer_major: lm,
                ..Default::default()
            },
            ..Default::default()
        };
        let auto = with(None);
        // (wanted, chunks, holds every block) -> layer-major
        let cases = [
            ((true, 1, false), false),
            ((true, 1, true), false),
            ((true, 3, true), false),
            ((true, 3, false), true),
            ((false, 3, false), false),
        ];
        for ((wanted, n, holds), want) in cases {
            assert_eq!(
                super::layer_major_this_prompt(&auto, wanted, n, holds),
                want,
                "auto: wanted={wanted} chunks={n} holds_every_block={holds}"
            );
        }
        let forced = with(Some(true));
        for (n, holds) in [(1, true), (1, false), (3, true)] {
            assert!(
                super::layer_major_this_prompt(&forced, true, n, holds),
                "paging.layer_major = true forces it (chunks={n}, holds={holds})"
            );
        }
        // `false` reaches here as not wanted (`layer_major_prefill` refuses it first).
        assert!(!super::layer_major_this_prompt(
            &with(Some(false)),
            false,
            3,
            false
        ));
    }

    /// **Drift guard, the rung half.** The shared `ubatch_candidates` ladder exists so
    /// the context-fit math and the placement sweep settle on the SAME prefill chunk. They only do
    /// while both budget against the same ceiling: with the sweep on `vram.available` and the fit
    /// math on `alloc_room()`, gemma-3-12b @131072 was validated at one rung and placed at a taller
    /// one (the reported case). Re-derives the expected rung from the primitives — weights + KV +
    /// reserve against `XTX_ROOM` — rather than from the function under test.
    #[test]
    fn fit_math_and_placement_pick_the_same_rung() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = gemma3_12b();
        let ec = EngineConfig::default();
        let vram = xtx(XTX_FREE);
        let (k, v) = (DType::F16, DType::F16);
        let cands = super::ubatch_candidates(&ec);
        let need = |ctx: usize, ub: usize| {
            GEMMA3_12B_WEIGHTS
                + super::kv_bytes_estimate_fmt(&cfg, ctx, true, ub, super::KvAlign::NONE, k, v)
                + super::dense_act_reserve_at(&cfg, ctx, ub)
        };
        let expect_rung = |ctx: usize| cands.iter().copied().find(|&ub| need(ctx, ub) <= XTX_ROOM);

        // A shape where the ladder is genuinely WALKED — otherwise "they agree" would be satisfied
        // by both picking the default rung and the guard would prove nothing. Derived, not
        // hardcoded: the heaviest weights that still fit at the 512-row rung, which by construction
        // cannot fit at 1024. (The reported gemma-3-12b @131072 case no longer needs a shorter rung
        // at all — the measured reserve is small enough that the default chunk holds it — so the
        // agreement is checked here and the trained-window outcome in
        // `kv_fit_walks_the_placement_chunk_ladder_gemma3_12b`.)
        let want = cfg.n_ctx_train;
        let heavy = XTX_ROOM
            - super::kv_bytes_estimate_fmt(&cfg, want, true, 512, super::KvAlign::NONE, k, v)
            - super::dense_act_reserve_at(&cfg, want, 512);
        let need_heavy = |ub: usize| {
            heavy
                + super::kv_bytes_estimate_fmt(&cfg, want, true, ub, super::KvAlign::NONE, k, v)
                + super::dense_act_reserve_at(&cfg, want, ub)
        };
        assert!(need_heavy(1024) > XTX_ROOM, "the default chunk must miss");
        assert!(
            need_heavy(512) <= XTX_ROOM,
            "…and the 512-row rung must save it"
        );
        assert_eq!(
            super::dense_resident_rung(&cfg, &ec, heavy, &vram, want, super::KvAlign::NONE, k, v),
            Some(512),
            "placement must settle on the rung the fit math priced"
        );

        // And at the exact boundary the fit math hands out: placement takes it resident, and one
        // token past it neither of them accepts.
        let fit = super::kv_fit_ctx_for(
            &cfg,
            &ec,
            GEMMA3_12B_WEIGHTS,
            &vram,
            super::KvAlign::NONE,
            k,
            v,
        )
        .expect("has KV");
        assert_eq!(
            super::dense_resident_rung(
                &cfg,
                &ec,
                GEMMA3_12B_WEIGHTS,
                &vram,
                fit,
                super::KvAlign::NONE,
                k,
                v
            ),
            expect_rung(fit),
        );
        assert!(
            super::dense_resident_rung(
                &cfg,
                &ec,
                GEMMA3_12B_WEIGHTS,
                &vram,
                fit,
                super::KvAlign::NONE,
                k,
                v
            )
            .is_some(),
            "the advertised context must be one placement can hold resident"
        );
        assert_eq!(
            super::dense_resident_rung(
                &cfg,
                &ec,
                GEMMA3_12B_WEIGHTS,
                &vram,
                fit + 1,
                super::KvAlign::NONE,
                k,
                v
            ),
            None,
            "one token past the advertised context must miss at EVERY rung — a placement that \
             still says yes here is budgeting against a wider ceiling than the fit math"
        );
    }

    /// The refuse rung's input: when the weights leave no usable room, the fit reports the honest
    /// small number (possibly `0`) rather than a floored 1024 that reads as "a session fits".
    /// `clamp_default_ctx` turns that into an error naming the numbers; that half needs a live
    /// backend, so it is only exercised by the GPU tests.
    #[test]
    fn kv_fit_reports_below_the_floor_instead_of_pretending() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = gemma3_12b();
        let ec = EngineConfig::default();
        // Weights fill the card: room left is under the 256 MiB fixed activation reserve alone.
        let fit = super::kv_fit_ctx_for(
            &cfg,
            &ec,
            XTX_ROOM - 64 * 1024 * 1024,
            &xtx(XTX_FREE),
            super::KvAlign::NONE,
            DType::F16,
            DType::F16,
        )
        .expect("has KV");
        assert!(
            fit < super::MIN_SESSION_CTX,
            "must report the real (unusable) fit, got {fit}"
        );
    }

    /// A pure recurrent-state arch has no per-token KV to size — `None`, not `0`, so callers keep
    /// the trained window instead of clamping to nothing.
    #[test]
    fn kv_fit_is_none_without_a_per_token_cache() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = Config {
            n_layer: 24,
            n_ctx_train: 32768,
            ..Default::default() // n_kv == head_dim == 0
        };
        assert!(super::kv_fit_ctx_for(
            &cfg,
            &EngineConfig::default(),
            1 << 30,
            &xtx(XTX_FREE),
            super::KvAlign::NONE,
            DType::F16,
            DType::F16,
        )
        .is_none());
    }

    /// A host with memory to spare for any cache these tests spill, handed to the re-clamp in
    /// place of the machine's own reading: what is under test is the device-side rule, and a
    /// runner short of memory would otherwise be refused by [`super::refuse_unhostable_kv`].
    const ROOMY_HOST: Option<u64> = Some(1 << 40);

    /// A `Backend` that reports nothing but a name and a live allocation budget — enough to drive
    /// [`super::reclamp_ctx_to_live_room`], which touches no other method. `room: None` stands in
    /// for the CPU/Metal backends, which have no budget to report.
    struct RoomOnly {
        room: Option<u64>,
        name: &'static str,
        /// Whether it reports its weights streamed (`Backend::dense_paged`).
        streamed: bool,
        /// Whether it reports the KV spill armed (`Backend::kv_spill_armed`).
        armed: bool,
    }

    impl RoomOnly {
        /// A backend no KV-format pin applies to.
        fn new(room: Option<u64>) -> Self {
            Self {
                room,
                name: "room-only",
                streamed: false,
                armed: false,
            }
        }

        /// A backend named as Vulkan, whose runner reads a q8_0 pin (`parse_kv_fmt`).
        fn vulkan(room: u64) -> Self {
            Self {
                room: Some(room),
                name: "vulkan",
                streamed: false,
                armed: false,
            }
        }

        /// This backend with the spill armed or not.
        fn armed(self, armed: bool) -> Self {
            Self { armed, ..self }
        }

        /// [`RoomOnly::vulkan`] reporting its weights streamed.
        fn streaming(room: u64) -> Self {
            Self {
                streamed: true,
                ..Self::vulkan(room)
            }
        }
    }

    impl infr_core::backend::Backend for RoomOnly {
        fn name(&self) -> &str {
            self.name
        }
        fn capabilities(&self) -> infr_core::backend::Capabilities {
            infr_core::backend::Capabilities::default()
        }
        fn alloc(
            &self,
            _bytes: usize,
            _usage: infr_core::backend::BufferUsage,
        ) -> infr_core::Result<Box<dyn infr_core::backend::Buffer>> {
            unreachable!("the re-clamp allocates nothing")
        }
        fn upload(
            &self,
            _dst: &dyn infr_core::backend::Buffer,
            _src: &[u8],
        ) -> infr_core::Result<()> {
            unreachable!("the re-clamp uploads nothing")
        }
        fn download(
            &self,
            _src: &dyn infr_core::backend::Buffer,
            _dst: &mut [u8],
        ) -> infr_core::Result<()> {
            unreachable!("the re-clamp downloads nothing")
        }
        fn compile(
            &self,
            _graph: &infr_core::graph::Graph,
        ) -> infr_core::Result<Box<dyn infr_core::backend::Plan>> {
            unreachable!("the re-clamp compiles nothing")
        }
        fn execute(
            &self,
            _plan: &dyn infr_core::backend::Plan,
            _bindings: &infr_core::backend::Bindings,
        ) -> infr_core::Result<()> {
            unreachable!("the re-clamp executes nothing")
        }
        fn sync(&self) -> infr_core::Result<()> {
            Ok(())
        }
        fn device_alloc_room(&self) -> Option<u64> {
            self.room
        }
        fn dense_paged(&self) -> bool {
            self.streamed
        }
        fn kv_spill_armed(&self) -> bool {
            self.armed
        }
    }

    /// The post-load re-clamp only ever SHRINKS, and only a context the session chose. Each arm is
    /// a decision someone could quietly invert: a backend that cannot report a budget must leave
    /// the window alone (CPU/Metal), a roomy device must not touch it, a tight one must cut it to
    /// something that fits its OWN measured budget, and a user-pinned `--ctx` is documented as
    /// verbatim — the alloc-time guard is its backstop, not this.
    #[test]
    fn reclamp_only_shrinks_and_never_a_pinned_ctx() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = gemma3_12b();
        let ec = EngineConfig::default();
        let (k, v) = (DType::F16, DType::F16);
        let want = 32768;
        let call = |be: &dyn infr_core::backend::Backend, ec: &EngineConfig| {
            let live = super::reclamp_ctx_to_live_room(be, &cfg, ec, want, k, v).expect("reclamp");
            assert_eq!(
                (live.k_fmt, live.v_fmt),
                (k, v),
                "no backend here takes a q8_0 pin"
            );
            assert!(!live.spilled, "nothing spills without kv.overflow");
            live.ctx
        };

        // No budget to report: the caller's window survives untouched.
        assert_eq!(call(&RoomOnly::new(None), &ec), want);

        // Roomy: the fit is past `want`, so the window is kept rather than raised.
        let roomy =
            super::kv_bytes_estimate_fmt(&cfg, want, true, 1024, super::KvAlign::NONE, k, v)
                + super::dense_act_reserve_at(&cfg, want, 1024)
                + super::POST_KV_DEVICE_RESERVE
                + (1 << 30);
        assert_eq!(call(&RoomOnly::new(Some(roomy)), &ec), want);

        // Tight: cut, and cut to a window that really fits the budget it was given.
        let tight = roomy / 3;
        let got = call(&RoomOnly::new(Some(tight)), &ec);
        assert!(got < want, "a tight device must shrink the window: {got}");
        let ub = super::ubatch_rows(&ec);
        let need = super::kv_bytes_estimate_fmt(&cfg, got, true, ub, super::KvAlign::NONE, k, v)
            + super::dense_act_reserve_at(&cfg, got, ub);
        assert!(
            need <= tight - super::POST_KV_DEVICE_RESERVE,
            "the clamped window must fit the measured budget: {need} > {tight}"
        );

        // Pinned by the user: honored verbatim, however tight the device is.
        let pinned = EngineConfig {
            device: infr_core::config::DeviceCfg {
                ctx: Some(infr_core::SizeSpec::Bytes(want as u64)),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(call(&RoomOnly::new(Some(tight)), &pinned), want);

        // Not one token fits: refused with the measured room, not handed on as a context of 0
        // for the request to overflow later ("exceeds the session KV capacity 0").
        let err = super::reclamp_ctx_to_live_room(
            &RoomOnly::new(Some(super::POST_KV_DEVICE_RESERVE)),
            &cfg,
            &ec,
            want,
            k,
            v,
        )
        .expect_err("a room holding only the driver reserve must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("does not fit") && msg.contains("0.25 GiB allocatable"),
            "{msg}"
        );
    }

    /// `post_load_ctx` as it stood before B62's spill reached the explicit window (HEAD 21c005f6),
    /// transcribed, for the equivalence check below.
    fn post_load_ctx_before_b62(
        want: usize,
        explicit: bool,
        overflow: bool,
        fit: usize,
        q8_fit: Option<usize>,
    ) -> super::CtxRung {
        use super::CtxRung::{Refuse, Serve, ServeQ8, Spill};
        if fit >= want || explicit {
            return Serve(want);
        }
        let floor = want.min(super::MIN_SESSION_CTX);
        if fit >= floor {
            return Serve(fit);
        }
        if let Some(q8) = q8_fit.filter(|&q8| q8 >= floor) {
            return ServeQ8(q8.min(want));
        }
        if overflow {
            return Spill(floor);
        }
        Refuse
    }

    /// The re-clamp's rule is the one it was wherever the spill is not armed, over every window,
    /// fit and q8_0 fit at and around each boundary (zero, the minimum, the window), both window
    /// kinds; with the spill armed only an explicit window that does not fit changes, from
    /// "served as asked" to "spilled as asked", the same window either way.
    #[test]
    fn post_load_ctx_changes_only_the_armed_explicit_spill() {
        use super::CtxRung::{Serve, Spill};
        let min = super::MIN_SESSION_CTX;
        let wants = [1, min / 2, min - 1, min, min + 1, 4 * min, 40 * min];
        let mut points: Vec<usize> = vec![0, 1, 2];
        for &w in &wants {
            points.extend([w.saturating_sub(1), w, w + 1]);
        }
        points.sort_unstable();
        points.dedup();
        let mut checked = 0usize;
        for &want in &wants {
            for &fit in &points {
                let q8s = std::iter::once(None).chain(points.iter().map(|&q| Some(q)));
                for q8 in q8s {
                    for explicit in [false, true] {
                        for overflow in [false, true] {
                            let now = super::post_load_ctx(want, explicit, overflow, fit, || q8);
                            let was = post_load_ctx_before_b62(want, explicit, overflow, fit, q8);
                            let case = format!(
                                "want {want}, fit {fit}, q8_0 {q8:?}, explicit {explicit}, \
                                 overflow {overflow}"
                            );
                            if overflow && explicit && fit < want {
                                assert_eq!((was, now), (Serve(want), Spill(want)), "{case}");
                            } else {
                                assert_eq!(now, was, "{case}");
                            }
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 5000, "only {checked} cases checked");
    }

    /// A spilled session runs at the shortest chunk the ladder allows when `INFR_UBATCH` is not
    /// pinned, and its reserve is priced there: where nothing fits at any height, a room holding
    /// the reserve at that chunk opens spilled and pins the chunk, and a byte less is refused.
    /// With the chunk pinned, the pinned height is the one priced.
    #[test]
    fn a_spilled_session_takes_the_shortest_chunk() {
        let cfg = gemma3_12b();
        let (k, v) = (DType::F16, DType::F16);
        let want = 8192;
        for pinned in [None, Some(512)] {
            let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
            let mut ec = EngineConfig::default();
            ec.kv.overflow = true;
            ec.device.ctx = Some(infr_core::SizeSpec::Bytes(want as u64));
            ec.device.ubatch = pinned;
            ec.device.ubatch_specified = pinned.is_some();
            let chunk = pinned.unwrap_or_else(|| {
                super::ubatch_candidates(&ec)
                    .into_iter()
                    .min()
                    .expect("a ladder")
            });
            assert!(
                pinned.is_some() || chunk < super::ubatch_rows(&ec),
                "a ladder to walk"
            );
            let reserve = super::kv_spill_reserve(&cfg, &ec, want, chunk, false);
            let reclamp = |room: u64| {
                let be = RoomOnly::vulkan(room).armed(true);
                super::reclamp_ctx_with_host(&be, &cfg, &ec, want, k, v, ROOMY_HOST)
            };
            reclamp(reserve - 1).expect_err("a byte under the reserve at its chunk is refused");
            let live = reclamp(reserve).expect("the reserve at the spilled chunk opens");
            assert!(live.spilled, "pinned {pinned:?}: {live:?}");
            assert_eq!(super::ubatch_rows(&ec), chunk, "pinned {pinned:?}");
        }
    }

    /// After the load, a spilled session whose measured room is under the spill reserve it holds
    /// is refused, naming the sizes, before a KV buffer is allocated; at the reserve it opens
    /// spilled. Driven through the re-clamp with a backend that reports a chosen room
    /// (`RoomOnly`), the test seam for "the weights landed just short": a real device cannot be
    /// steered to a room within a byte of the reserve. Both kinds of window: an explicit one,
    /// kept spilled, and an auto-sized one, spilled at the minimum.
    #[test]
    fn a_spilled_session_short_of_its_reserve_is_refused_after_the_load() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = gemma3_12b();
        let (k, v) = (DType::F16, DType::F16);
        let ub = 512;
        let want = 8192;
        for explicit in [true, false] {
            let mut ec = EngineConfig::default();
            ec.kv.overflow = true;
            ec.device.ubatch = Some(ub);
            ec.device.ubatch_specified = true;
            if explicit {
                ec.device.ctx = Some(infr_core::SizeSpec::Bytes(want as u64));
            }
            let ctx = if explicit {
                want
            } else {
                super::MIN_SESSION_CTX
            };
            let reserve = super::kv_spill_reserve(&cfg, &ec, ctx, ub, false);
            let reclamp = |room: u64| {
                let be = RoomOnly::vulkan(room).armed(true);
                super::reclamp_ctx_with_host(&be, &cfg, &ec, want, k, v, ROOMY_HOST)
            };
            let live = reclamp(reserve).expect("a room at the reserve opens spilled");
            assert_eq!(
                live,
                super::LiveCtx {
                    ctx,
                    k_fmt: k,
                    v_fmt: v,
                    spilled: true
                },
                "explicit {explicit}"
            );
            let err = reclamp(reserve - 1).expect_err("one byte under the reserve is refused");
            let msg = err.to_string();
            assert!(
                msg.contains(&format!("context {ctx} cannot be served")) && msg.contains("GiB"),
                "explicit {explicit}: the refusal names the window and the sizes: {msg}"
            );
        }
    }

    /// Arming the spill changes no window the re-clamp serves: for an auto-sized session, every
    /// room where the re-clamp without `kv.overflow` serves a window (in the session's formats or
    /// in q8_0) gives the same window and formats with it, on a resident and on a streamed session,
    /// over every `paging.layer_major` setting, with and without `kv.overflow_reserve_mb`; where
    /// it refuses, the armed one spills the minimum or refuses. Rooms from below the minimum's
    /// need to past the window's.
    #[test]
    fn arming_the_spill_serves_the_windows_it_served() {
        let cfg = gemma3_12b();
        let (k, v) = (DType::F16, DType::F16);
        let ub = 512;
        let want = 16384;
        let mut checked = 0usize;
        for streamed in [false, true] {
            for layer_major in [None, Some(true), Some(false)] {
                for margin in [None, Some(64)] {
                    let config = |overflow: bool| {
                        let mut ec = EngineConfig::default();
                        ec.kv.overflow = overflow;
                        ec.kv.overflow_reserve_mb = margin;
                        ec.device.ubatch = Some(ub);
                        ec.device.ubatch_specified = true;
                        ec.paging.layer_major = layer_major;
                        ec
                    };
                    let (off, on) = (config(false), config(true));
                    let ring = super::placement_ring(&cfg, &off, k, v);
                    let need = |ctx: usize| {
                        super::kv_bytes_estimate_fmt(
                            &cfg,
                            ctx,
                            ring,
                            ub,
                            super::KvAlign::NONE,
                            k,
                            v,
                        ) + super::dense_act_reserve_at(&cfg, ctx, ub)
                            + super::POST_KV_DEVICE_RESERVE
                    };
                    let (lo, hi) = (need(super::MIN_SESSION_CTX / 2), need(want) + (1 << 30));
                    for i in 0..=48u64 {
                        let room = lo + (hi - lo) * i / 48;
                        let be = |armed: bool| {
                            let be = if streamed {
                                RoomOnly::streaming(room)
                            } else {
                                RoomOnly::vulkan(room)
                            };
                            be.armed(armed)
                        };
                        let case = format!(
                            "streamed {streamed}, layer_major {layer_major:?}, margin {margin:?}, \
                             room {room}"
                        );
                        // Each in its own placement scope: a q8_0 pin one makes is the session's.
                        let reclamp = |armed: bool, ec: &EngineConfig| {
                            let _scope = PlacementScope::enter(Default::default());
                            super::reclamp_ctx_with_host(
                                &be(armed),
                                &cfg,
                                ec,
                                want,
                                k,
                                v,
                                ROOMY_HOST,
                            )
                        };
                        let (was, now) = (reclamp(false, &off), reclamp(true, &on));
                        match was {
                            Ok(was) => assert_eq!(now.ok(), Some(was), "{case}"),
                            Err(_) => assert!(
                                now.as_ref()
                                    .map_or(true, |l| l.spilled && l.ctx == super::MIN_SESSION_CTX),
                                "{case}: refused without the flag, {now:?} with it"
                            ),
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 500, "only {checked} rooms checked");
    }

    /// The spill is armed exactly when the flag is on, the device is not unified memory and its
    /// driver reports a live budget: one of the eight cases.
    #[test]
    fn kv_spill_is_armed_only_with_a_live_budget_off_unified_memory() {
        for overflow in [false, true] {
            for unified in [false, true] {
                for live in [false, true] {
                    assert_eq!(
                        infr_vulkan::kv_spill_armed(overflow, unified, live),
                        (overflow, unified, live) == (true, false, true),
                        "overflow {overflow}, unified {unified}, live budget {live}"
                    );
                }
            }
        }
    }

    /// The KV refuse rung reads the spill flag it is handed where it read `kv.overflow` before:
    /// with the flag off nothing changes, over buffers past, at and under the room, with and
    /// without the guard, on either kind of device; with it on, only a unified-memory device (where
    /// nothing spills) now refuses, and exactly what the rung refuses without the flag.
    #[test]
    fn kv_refuse_rung_reads_the_armed_spill() {
        let (k, v) = (DType::F16, DType::F16);
        let mib = 1usize << 20;
        let sizes = vec![(64 * mib, 64 * mib); 4];
        let need = 512 * (mib as u64);
        // `refuse_unallocatable_kv` as it stood at HEAD 21c005f6: whether it refused.
        let before = |room: Option<u64>, ec: &EngineConfig| {
            room.is_some_and(|r| !ec.kv.overflow && !ec.kernels.vulkan.no_vram_guard && need > r)
        };
        for room in [None, Some(0), Some(need - 1), Some(need), Some(need + 1)] {
            for overflow in [false, true] {
                for no_guard in [false, true] {
                    for unified in [false, true] {
                        let mut ec = EngineConfig::default();
                        ec.kv.overflow = overflow;
                        ec.kernels.vulkan.no_vram_guard = no_guard;
                        let spill = infr_vulkan::kv_spill_armed(overflow, unified, true);
                        let refused = super::refuse_unallocatable_kv(
                            &ec,
                            spill,
                            &sizes,
                            room,
                            32768,
                            k,
                            v,
                            "here",
                            || None,
                        )
                        .is_err();
                        let case = format!(
                            "room {room:?}, overflow {overflow}, no guard {no_guard}, \
                             unified {unified}"
                        );
                        if overflow && unified {
                            let mut off = ec.clone();
                            off.kv.overflow = false;
                            assert_eq!(refused, before(room, &off), "{case}");
                        } else {
                            assert_eq!(refused, before(room, &ec), "{case}");
                        }
                    }
                }
            }
        }
    }

    /// The backend's placement rule, buffer by buffer: a buffer stays in device memory exactly when
    /// it and the reserve fit what is left, so at the boundary it stays and one byte past it does
    /// not; one that does not fit goes to system RAM and a smaller one after it can still stay; a
    /// reserve past the room keeps nothing. Exhaustive over small rooms, reserves and buffer lists,
    /// against the rule written out longhand.
    #[test]
    fn kv_vram_bytes_keeps_the_reserve_free() {
        let none = infr_core::budget::SpillTally::default();
        let stays = |used, budget, reserve, c| {
            infr_vulkan::kv_stays_in_vram(used, budget, reserve, c, &none, None, c)
        };
        let vram = |list: &[u64], room, reserve| {
            infr_vulkan::kv_vram_bytes(list.iter().map(|&b| (b, b)), room, reserve, None)
        };
        assert!(stays(10, 100, 30, 60));
        assert!(!stays(10, 100, 30, 61));
        assert!(!stays(0, 100, u64::MAX, 1));
        assert!(!stays(u64::MAX, 100, 0, 1));
        assert_eq!(vram(&[60, 50, 10], 100, 30), 70);
        assert_eq!(vram(&[10, 20], 5, 10), 0);
        let sizes = [0u64, 1, 2, 3, 5];
        let mut lists: Vec<Vec<u64>> = vec![vec![]];
        for _ in 0..4 {
            let longer: Vec<Vec<u64>> = lists
                .iter()
                .filter(|l| l.len() == lists.last().map_or(0, |l| l.len()))
                .flat_map(|l| {
                    sizes.iter().map(move |&b| {
                        let mut l = l.clone();
                        l.push(b);
                        l
                    })
                })
                .collect();
            lists.extend(longer);
        }
        let mut checked = 0usize;
        for list in &lists {
            for room in 0..=12u64 {
                for reserve in 0..=6u64 {
                    let mut left = room;
                    let mut kept = 0;
                    for &b in list {
                        if b + reserve <= left {
                            left -= b;
                            kept += b;
                        }
                    }
                    let got = vram(list, room, reserve);
                    assert_eq!(got, kept, "{list:?}, room {room}, reserve {reserve}");
                    assert!(
                        got == 0 || room - got >= reserve,
                        "{list:?}, room {room}, reserve {reserve}: {got} kept"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 50_000, "only {checked} cases checked");
    }

    /// A spilled cache's split: device memory and system RAM add up to the cache as the device
    /// commits it, a room with the reserve and the whole cache spare spills nothing, a room at or
    /// below the reserve keeps nothing in device memory, every room in between keeps the reserve
    /// free, and n slots split as n caches one after another.
    #[test]
    fn kv_spill_split_divides_the_cache_it_prices() {
        let cfg = gemma3_12b();
        let (k, v) = (DType::F16, DType::F16);
        let align = super::KvAlign {
            rows: 1,
            commit: infr_core::backend::KvAllocCost::new(4096, 2 << 20, 1 << 20, 4096),
        };
        let ctx = 8192;
        let ub = 512;
        let total =
            |slots: u64| slots * super::kv_bytes_estimate_fmt(&cfg, ctx, true, ub, align, k, v);
        let reserve = 300 << 20;
        let split = |slots: usize, room: u64| {
            super::kv_spill_split(&cfg, ctx, true, ub, align, k, v, slots, room, reserve, None)
        };
        for slots in [1usize, 2, 3] {
            let whole = total(slots as u64);
            let all = split(slots, whole + reserve);
            assert_eq!(
                all,
                super::KvSplit {
                    vram: whole,
                    host: 0
                },
                "{slots} slots"
            );
            assert!(split(slots, whole + reserve - 1).host > 0, "{slots} slots");
            for room in [0, 1, reserve - 1, reserve] {
                assert_eq!(
                    split(slots, room),
                    super::KvSplit {
                        vram: 0,
                        host: whole
                    },
                    "{slots} slots, room {room}"
                );
            }
            for i in 0..=64u64 {
                let room = reserve + whole * i / 64;
                let s = split(slots, room);
                assert_eq!(s.vram + s.host, whole, "{slots} slots, room {room}");
                assert!(
                    room - s.vram >= reserve,
                    "{slots} slots, room {room}: {s:?}"
                );
            }
        }
        // Two slots in twice the room of one keep at least what one keeps, and at most twice it.
        let one = split(1, reserve + total(1) / 2);
        let two = split(2, reserve + total(1));
        assert!(
            two.vram >= one.vram && two.vram <= 2 * one.vram + total(1),
            "{one:?} {two:?}"
        );
    }

    /// The diagnostic cap (`kv.overflow_vram_mb`) is part of the split as it is of the allocation:
    /// on a device with room for the whole cache, the split keeps in device memory exactly the
    /// buffers whose requested bytes, added in allocation order, stay within the cap (one that
    /// would pass it goes to system RAM and a smaller one after it can still fit), priced at what
    /// they commit; a cap of zero keeps nothing, none keeps everything. Every cap from zero past
    /// the whole cache in steps, against the rule written out longhand.
    #[test]
    fn kv_spill_split_keeps_to_the_cap() {
        let cfg = gemma3_12b();
        let (k, v) = (DType::F16, DType::F16);
        let align = super::KvAlign {
            rows: 1,
            commit: infr_core::backend::KvAllocCost::new(4096, 2 << 20, 1 << 20, 4096),
        };
        let (ctx, ub, reserve) = (8192, 512, 300u64 << 20);
        let buffers: Vec<u64> = super::kv_slot_allocs(&cfg, ctx, true, ub, 1, k, v)
            .flat_map(|(a, b)| [a as u64, b as u64])
            .collect();
        let raw: u64 = buffers.iter().sum();
        let whole: u64 = buffers.iter().map(|&b| align.commit.committed(b)).sum();
        let room = whole + reserve;
        let split =
            |cap| super::kv_spill_split(&cfg, ctx, true, ub, align, k, v, 1, room, reserve, cap);
        assert_eq!(split(None).vram, whole);
        assert_eq!(split(Some(0)).vram, 0);
        for i in 0..=40u64 {
            let cap = raw * i / 32;
            let (mut placed, mut kept) = (0u64, 0u64);
            for &b in &buffers {
                if placed + b <= cap {
                    placed += b;
                    kept += align.commit.committed(b);
                }
            }
            let got = split(Some(cap));
            assert_eq!(got.vram, kept, "cap {cap}");
            assert_eq!(got.vram + got.host, whole, "cap {cap}");
        }
    }

    /// The spill reserve's terms at known values, on gemma-3-12b at 8192 tokens and a 512-row
    /// chunk: the layer-major stream is 16 chunks of 512 rows of 3840 f32s (125 829 120 bytes)
    /// where placement prices one — streamed unless `paging.layer_major` is off, resident only when
    /// it is forced on — the driver's reserve 256 MiB, the margin `kv.overflow_reserve_mb` in
    /// bytes, and the total their sum with the activation reserve: dropping any term from the
    /// total, or pricing one where placement does not, fails.
    #[test]
    fn kv_spill_reserve_is_its_terms() {
        const LAYER_MAJOR: u64 = 125_829_120;
        const DRIVER: u64 = 268_435_456;
        let cfg = gemma3_12b();
        let (ctx, ub) = (8192, 512);
        let act = super::dense_act_reserve_at(&cfg, ctx, ub);
        assert!(act > 0);
        for (layer_major, margin, streamed, want_lm, want_margin) in [
            (None, None, false, 0, 0),
            (None, None, true, LAYER_MAJOR, 0),
            (Some(true), Some(64), false, LAYER_MAJOR, 67_108_864),
            (Some(true), Some(64), true, LAYER_MAJOR, 67_108_864),
            (Some(false), Some(0), false, 0, 0),
            (Some(false), Some(1), true, 0, 1_048_576),
        ] {
            let mut ec = EngineConfig::default();
            ec.paging.layer_major = layer_major;
            ec.kv.overflow_reserve_mb = margin;
            let case =
                format!("layer_major {layer_major:?}, margin {margin:?}, streamed {streamed}");
            let terms = super::kv_spill_reserve_terms(&cfg, &ec, ctx, ub, streamed);
            assert_eq!(
                (terms.layer_major, terms.driver, terms.margin),
                (want_lm, DRIVER, want_margin),
                "{case}"
            );
            assert_eq!(
                super::kv_spill_reserve(&cfg, &ec, ctx, ub, streamed),
                act + want_lm + DRIVER + want_margin,
                "{case}"
            );
        }
    }

    /// A spilled cache's system-RAM part is refused exactly when it exceeds what the host can spare
    /// a run once `committed` (the weights' host tier) is off, by the headroom rule the repo's host
    /// arenas follow (`infr_core::hostmem::spare_bytes`), and never by an arena's
    /// worth-building floor: a small spilled part fits a small spare. Never when the host cannot
    /// say. Expected values are written out for hosts with 4 and 64 GiB free, where the headroom is
    /// half of what is free; then every boundary of the rule, from free memory at and around the
    /// headroom's minimum, with and without a host tier; the refusal names the sizes.
    #[test]
    fn unhostable_kv_is_refused_past_what_a_run_may_take() {
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        let (k, v) = (DType::F16, DType::F16);
        let split = |host: u64| super::KvSplit {
            vram: 7 * MIB,
            host,
        };
        let refused = |host: u64, free: u64, committed: u64| {
            super::refuse_unhostable_kv(4096, split(host), Some(free), committed, k, v).is_err()
        };
        // 4 GiB free leaves 2 GiB; with a 1800 MiB tier, 248 MiB: an 86 MB spill fits it.
        assert!(!refused(86_000_000, 4 * GIB, 1800 * MIB));
        assert!(!refused(248 * MIB, 4 * GIB, 1800 * MIB));
        assert!(refused(248 * MIB + 1, 4 * GIB, 1800 * MIB));
        // 64 GiB free leaves 32 GiB.
        assert!(!refused(32 * GIB, 64 * GIB, 0));
        assert!(refused(32 * GIB + 1, 64 * GIB, 0));
        let mut checked = 0usize;
        for free in [
            0,
            1,
            GIB,
            2 * GIB - 1,
            2 * GIB,
            2 * GIB + 1,
            3 * GIB,
            64 * GIB,
        ] {
            for committed in [0, MIB, GIB, 3 * GIB] {
                let spare = infr_core::hostmem::spare_bytes(free, committed);
                for host in [0, 1, spare.saturating_sub(1), spare, spare + 1, 4 * GIB] {
                    assert!(
                        super::refuse_unhostable_kv(4096, split(host), None, committed, k, v)
                            .is_ok()
                    );
                    let got =
                        super::refuse_unhostable_kv(4096, split(host), Some(free), committed, k, v);
                    assert_eq!(
                        got.is_err(),
                        host > spare,
                        "host part {host}, free {free}, committed {committed}, spare {spare}"
                    );
                    if let Err(e) = got {
                        let msg = e.to_string();
                        assert!(
                            msg.contains("context 4096") && msg.contains("GiB"),
                            "the refusal names the window and the sizes: {msg}"
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 150, "only {checked} cases checked");
    }

    /// The post-load rule, case by case: a window is never cut below the minimum —
    /// `MIN_SESSION_CTX`, or the window itself when it asked for less — and below it the q8_0 cache
    /// is tried (only then), then the spill, then the refusal. An explicit window is kept as asked
    /// whatever fits: served without the spill armed, spilled with it when it does not fit.
    #[test]
    fn post_load_ctx_never_serves_below_the_minimum() {
        use super::CtxRung::{Refuse, Serve, ServeQ8, Spill};
        let min = super::MIN_SESSION_CTX;
        let unasked = || -> Option<usize> { panic!("q8_0 is tried only below the minimum") };
        let post = |want, explicit, overflow, fit, q8: Option<usize>| {
            super::post_load_ctx(want, explicit, overflow, fit, || q8)
        };
        // Fits, or set explicitly: as asked.
        assert_eq!(
            super::post_load_ctx(8192, false, false, 8192, unasked),
            Serve(8192)
        );
        assert_eq!(
            super::post_load_ctx(8192, true, false, 0, unasked),
            Serve(8192)
        );
        // Cut, down to the minimum and no further.
        assert_eq!(
            super::post_load_ctx(8192, false, false, 5000, unasked),
            Serve(5000)
        );
        assert_eq!(
            super::post_load_ctx(8192, false, false, min, unasked),
            Serve(min)
        );
        assert_eq!(post(8192, false, false, min - 1, None), Refuse);
        assert_eq!(post(8192, false, false, 64, None), Refuse);
        // A window below the minimum is its own minimum.
        assert_eq!(
            super::post_load_ctx(500, false, false, 500, unasked),
            Serve(500)
        );
        assert_eq!(post(500, false, false, 499, None), Refuse);
        // q8_0 rescues the minimum: served at what it holds, never past the window.
        assert_eq!(post(8192, false, false, 64, Some(min)), ServeQ8(min));
        assert_eq!(post(8192, false, false, 64, Some(3000)), ServeQ8(3000));
        assert_eq!(post(500, false, false, 64, Some(3000)), ServeQ8(500));
        assert_eq!(post(8192, false, false, 64, Some(min - 1)), Refuse);
        // The spill keeps the minimum, after q8_0 and before the refusal.
        assert_eq!(post(8192, false, true, 64, None), Spill(min));
        // An explicit window that does not fit is spilled as asked; one that fits is served.
        assert_eq!(post(8192, true, true, 64, None), Spill(8192));
        assert_eq!(
            super::post_load_ctx(8192, true, true, 8192, unasked),
            Serve(8192)
        );
        assert_eq!(post(8192, false, true, 64, Some(3000)), ServeQ8(3000));
        assert_eq!(post(500, false, true, 64, None), Spill(500));
    }

    /// The pre-load ladder's order (B62): q8_0 first, then the clamp, then the spill, then the
    /// refusal, for a window the session chose and one the user set, with `kv.overflow` off and
    /// on, in every room shape: the window fits, fits only in q8_0, fits only cut, fits only below
    /// the minimum where q8_0 still holds the minimum, and fits nowhere.
    #[test]
    fn preload_ctx_clamps_before_it_spills() {
        use super::CtxRung::{Refuse, Serve, ServeQ8, Spill};
        let min = super::MIN_SESSION_CTX;
        let want = 8 * min;
        let pre = |explicit, overflow, fit, q8: Option<usize>| {
            super::preload_ctx(want, explicit, overflow, fit, || q8)
        };
        // (room, f16 fit, q8_0 fit, [auto off, auto overflow, explicit off, explicit overflow]).
        let rooms = [
            ("fits", want, None, [Serve(want); 4]),
            (
                "fits only in q8_0",
                5 * min,
                Some(9 * min),
                [ServeQ8(want); 4],
            ),
            (
                "fits only cut",
                5 * min,
                Some(7 * min),
                [Serve(5 * min), Serve(5 * min), Serve(want), Spill(want)],
            ),
            (
                "fits only cut, no q8_0",
                5 * min,
                None,
                [Serve(5 * min), Serve(5 * min), Serve(want), Spill(want)],
            ),
            (
                "f16 below the minimum, q8_0 holds it",
                min / 2,
                Some(3 * min),
                [ServeQ8(3 * min), ServeQ8(3 * min), Serve(want), Spill(want)],
            ),
            (
                "below the minimum in both",
                min / 2,
                Some(min - 1),
                [Refuse, Spill(min), Serve(want), Spill(want)],
            ),
            (
                "nothing fits",
                0,
                None,
                [Refuse, Spill(min), Serve(want), Spill(want)],
            ),
        ];
        for (room, fit, q8, expect) in rooms {
            let got = [
                pre(false, false, fit, q8),
                pre(false, true, fit, q8),
                pre(true, false, fit, q8),
                pre(true, true, fit, q8),
            ];
            assert_eq!(
                got, expect,
                "{room}: [auto, auto + overflow, explicit, explicit + overflow]"
            );
        }
        // The q8_0 fit is priced only when f16 falls short.
        let unasked = || -> Option<usize> { panic!("q8_0 is priced only when f16 falls short") };
        for (explicit, overflow) in [(false, false), (false, true), (true, false), (true, true)] {
            assert_eq!(
                super::preload_ctx(want, explicit, overflow, want, unasked),
                Serve(want)
            );
        }
    }

    /// The fit before the load and the re-clamp after it give the same answer for the same room:
    /// whatever the pre-load ladder opens a session with, the re-clamp keeps — the same window,
    /// with no switch to q8_0 — and a room the pre-load side refuses the re-clamp refuses too. For
    /// an auto-sized window the re-clamp also spills exactly when the pre-load side did; an
    /// explicit one it serves as asked by design (`post_load_ctx`), so only its window is
    /// compared. Over a grid of windows, fits and q8_0 fits at and around every boundary the rules
    /// compare against, for both kinds of window and both `kv.overflow` settings, with the same
    /// kind of window on both sides (`ec.device.ctx` against the entry point is backlog B62's).
    /// The re-clamp is fed what the session carries into it: the window it opened with, the fit
    /// of the format it pinned, and no q8_0 retry once q8_0 is pinned (the re-clamp tries q8_0
    /// only on an f16 cache). Agreement cannot see a pre-load side that cuts too much, since the
    /// re-clamp keeps any smaller window: `preload_ctx_clamps_before_it_spills` pins that.
    #[test]
    fn preload_and_post_load_agree_for_the_same_room() {
        use super::CtxRung::{Refuse, Serve, ServeQ8, Spill};
        let min = super::MIN_SESSION_CTX;
        let wants = [min / 2, min - 1, min, min + 1, 4 * min, 40 * min];
        let mut points: Vec<usize> = vec![0, 1];
        for &w in &wants {
            points.extend([w.saturating_sub(1), w, w + 1]);
        }
        points.extend([min / 3, 3 * min, 20 * min, 50 * min]);
        points.sort_unstable();
        points.dedup();
        let mut checked = 0usize;
        for &want in &wants {
            for &fit in &points {
                let q8s = std::iter::once(None)
                    .chain(points.iter().filter(|&&q8| q8 >= fit).map(|&q8| Some(q8)));
                for q8 in q8s {
                    for explicit in [false, true] {
                        for overflow in [false, true] {
                            let case = format!(
                                "want {want}, f16 fit {fit}, q8_0 fit {q8:?}, explicit {explicit}, \
                                 overflow {overflow}"
                            );
                            let pre = super::preload_ctx(want, explicit, overflow, fit, || q8);
                            let (ctx, q8_pinned) = match pre {
                                Serve(ctx) | Spill(ctx) => (ctx, false),
                                ServeQ8(ctx) => (ctx, true),
                                Refuse => {
                                    assert_eq!(
                                        super::post_load_ctx(want, explicit, overflow, fit, || q8),
                                        Refuse,
                                        "{case}: refused before the load only"
                                    );
                                    checked += 1;
                                    continue;
                                }
                            };
                            let (post_fit, post_q8) = if q8_pinned {
                                (q8.expect("q8_0 was served"), None)
                            } else {
                                (fit, q8)
                            };
                            let post =
                                super::post_load_ctx(ctx, explicit, overflow, post_fit, || post_q8);
                            let kept = match post {
                                Serve(c) | Spill(c) => c,
                                ServeQ8(_) | Refuse => {
                                    panic!("{case}: opened {pre:?}, the re-clamp gave {post:?}")
                                }
                            };
                            assert_eq!(kept, ctx, "{case}: opened {pre:?}, re-clamped {post:?}");
                            // Both sides spill the same windows, an explicit one included: the
                            // re-clamp prices a spilled cache as the fit before the load did.
                            assert_eq!(
                                matches!(pre, Spill(_)),
                                matches!(post, Spill(_)),
                                "{case}: opened {pre:?}, re-clamped {post:?}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        // The grid ran: every window against every fit pair, both window kinds and settings.
        assert!(checked > 1000, "only {checked} rooms checked");
    }

    /// An `INFR_CACHE` percentage is of the allocatable room (plan 13 S3), not of the free figure the
    /// guard keeps its headroom inside, nor of the device's total: on the reference box `100%` is
    /// exactly the bytes the allocator can still hand out. An absolute size is taken as given.
    #[test]
    fn paging_cache_percentage_is_of_the_allocatable_room() {
        use infr_core::SizeSpec::{Bytes, Percent};
        let vram = xtx(XTX_FREE);
        assert!(vram.alloc_room() < vram.available && vram.available < vram.total);
        assert_eq!(super::paging_cache_budget(Percent(1.0), &vram), XTX_ROOM);
        assert_eq!(
            super::paging_cache_budget(Percent(0.5), &vram),
            (XTX_ROOM as f64 * 0.5) as u64
        );
        assert_eq!(super::paging_cache_budget(Bytes(19 << 30), &vram), 19 << 30);
    }

    /// The q8_0 retry inside the re-clamp itself, on a room where an f16 cache leaves less than the
    /// minimum and a q8_0 one does not: a Vulkan session whose user set no format is switched to
    /// q8_0 (and pinned, so its later calls read it), one whose user set f16 is refused with q8_0
    /// named as the thing to try, and a backend the pin does not reach is refused.
    #[test]
    fn reclamp_retries_q8_below_the_minimum_where_the_policy_allows() {
        let cfg = gemma3_12b();
        let (f16, q8) = ((DType::F16, DType::F16), (DType::Q8_0, DType::Q8_0));
        let min = super::MIN_SESSION_CTX;
        let ec = EngineConfig::default();
        let cands = super::ubatch_candidates(&ec);
        let need = |(k, v): (DType, DType), ctx: usize| {
            cands
                .iter()
                .map(|&ub| {
                    super::kv_bytes_estimate_fmt(&cfg, ctx, true, ub, super::KvAlign::NONE, k, v)
                        + super::dense_act_reserve_at(&cfg, ctx, ub)
                })
                .min()
                .expect("the ladder is not empty")
        };
        // Exactly the minimum in q8_0, which f16 cannot hold.
        let room = super::POST_KV_DEVICE_RESERVE + need(q8, min);
        assert!(
            need(f16, min) > need(q8, min),
            "f16 must miss where q8_0 holds"
        );
        let want = 32768;

        let scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let super::LiveCtx {
            ctx,
            k_fmt: k,
            v_fmt: v,
            ..
        } = super::reclamp_ctx_to_live_room(&RoomOnly::vulkan(room), &cfg, &ec, want, f16.0, f16.1)
            .expect("q8_0 holds the minimum");
        assert_eq!((k, v), q8);
        assert!(
            (min..want).contains(&ctx),
            "q8_0 holds at least the minimum here and not the window: {ctx}"
        );
        assert!(
            super::kv_auto_q8(),
            "the session's later calls must read q8_0 too"
        );
        drop(scope);

        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let mut user_f16 = EngineConfig::default();
        user_f16.kv.type_k = Some(DType::F16);
        user_f16.kv.type_k_specified = true;
        user_f16.kv.type_v = Some(DType::F16);
        user_f16.kv.type_v_specified = true;
        let refused = |be: &RoomOnly, ec: &EngineConfig| {
            super::reclamp_ctx_to_live_room(be, &cfg, ec, want, f16.0, f16.1)
                .expect_err("f16 leaves less than the minimum")
                .to_string()
        };
        let msg = refused(&RoomOnly::vulkan(room), &user_f16);
        assert!(msg.contains("INFR_KV_TYPE_K=q8_0"), "{msg}");
        assert!(
            !super::kv_auto_q8(),
            "a format the user set is not overridden"
        );
        let msg = refused(&RoomOnly::new(Some(room)), &ec);
        assert!(msg.contains("INFR_KV_TYPE_K=q8_0"), "{msg}");
    }

    /// The arena floor's rounding goes the safe way: never above what the arena's first-fit
    /// placement commits for the same units in any order (a fit priced with it never counts memory
    /// the arena will not commit for them), never below the units' own rounded bytes, and equal to
    /// the placement where the shared blocks pack whole — with the arena's blocks at their usual
    /// floor and on a device whose `maxMemoryAllocationSize` caps them below it.
    ///
    /// `first_fit` is a hand copy of `VulkanBackend::bda_weight_alloc`'s placement, cap included,
    /// so a change there is not seen here; the GPU test `arena_commits_its_floor_for_a_model`
    /// compares the floor with the real arena's blocks.
    #[test]
    fn weight_arena_floor_is_a_lower_bound_on_first_fit() {
        let mib = 1u64 << 20;
        // The arena's block floor (`BDA_BLOCK_MIN`), and the device caps tried: none that binds,
        // and one below the floor.
        let block_floor = 64 * mib;
        for max_alloc in [4096 * mib, 48 * mib] {
            let cost = infr_vulkan::WeightLoadCost {
                align: 256,
                block_min: block_floor.min(max_alloc),
                staging: 0,
            };
            let round = |u: u64| u.max(1).next_multiple_of(cost.align);
            // First-fit over every open block; when none has room, a new block of
            // `max(unit, block floor)` bytes capped at `max_alloc` (a unit past the cap is an
            // error there, and none is generated here).
            let first_fit = |units: &[u64]| -> u64 {
                let mut blocks: Vec<(u64, u64)> = Vec::new(); // (size, cursor)
                for &u in units {
                    let want = round(u);
                    let size = want.max(block_floor).min(max_alloc);
                    assert!(want <= size, "a unit past the device's cap");
                    match blocks
                        .iter_mut()
                        .find(|(size, cursor)| *cursor + want <= *size)
                    {
                        Some((_, cursor)) => *cursor += want,
                        None => blocks.push((size, want)),
                    }
                }
                blocks.iter().map(|&(size, _)| size).sum()
            };
            let packed = [16 * mib, 16 * mib, 100 * mib, 16 * mib, 16 * mib];
            if max_alloc >= 100 * mib {
                // Packs whole: four 16 MiB units share one block, a 100 MiB unit has its own.
                assert_eq!(super::weight_arena_floor(&packed, cost), first_fit(&packed));
                assert_eq!(super::weight_arena_floor(&packed, cost), 164 * mib);
                // A unit of exactly one block is whole, one alignment step less is shared, and a
                // unit is rounded up before it is classed.
                assert_eq!(super::weight_arena_floor(&[64 * mib], cost), 64 * mib);
                assert_eq!(
                    super::weight_arena_floor(&[64 * mib - 256, 1], cost),
                    64 * mib
                );
                assert_eq!(
                    super::weight_arena_floor(&[64 * mib - 1, 1], cost),
                    128 * mib
                );
                assert_eq!(super::weight_arena_floor(&[3], cost), 64 * mib);
            } else {
                // Capped: three 16 MiB units fill one 48 MiB block, the fourth opens another.
                let capped = [16 * mib; 4];
                assert_eq!(super::weight_arena_floor(&capped, cost), first_fit(&capped));
                assert_eq!(super::weight_arena_floor(&capped, cost), 96 * mib);
                assert_eq!(super::weight_arena_floor(&[3], cost), 48 * mib);
            }
            let big = (max_alloc - block_floor.min(max_alloc)).min(200 * mib) + 1;
            for seed in 0..64u64 {
                let units: Vec<u64> = (0..40)
                    .map(|i| {
                        let r = infr_testkit::gguf::splitmix64(seed * 1000 + i);
                        match r % 4 {
                            0 => cost.block_min + r % big,
                            _ => 1 + r % (cost.block_min * 5 / 8),
                        }
                    })
                    .collect();
                let floor = super::weight_arena_floor(&units, cost);
                assert!(
                    floor <= first_fit(&units),
                    "cap {max_alloc}, seed {seed}: floor {floor} above first-fit {}",
                    first_fit(&units)
                );
                let rounded: u64 = units.iter().map(|&u| round(u)).sum();
                assert!(
                    floor >= rounded,
                    "cap {max_alloc}, seed {seed}: floor {floor} below the units' {rounded}"
                );
            }
        }
    }

    /// The KV refuse rung fires only when the buffers the guard checks provably cannot land: past
    /// the room it refuses with the numbers, at the room it lets the session through, and it stays
    /// out of the way wherever the guard is not the last word.
    #[test]
    fn kv_refuse_rung_fires_only_when_the_guard_must_refuse() {
        let (k, v) = (DType::F16, DType::F16);
        let mib = 1usize << 20;
        let sizes = vec![(64 * mib, 64 * mib); 4]; // 512 MiB in guarded buffers
        let need = 512 * mib as u64;
        let call = |sizes: &[(usize, usize)], room: Option<u64>, ec: &EngineConfig| {
            let spill = infr_vulkan::kv_spill_armed(ec.kv.overflow, false, true);
            super::refuse_unallocatable_kv(ec, spill, sizes, room, 32768, k, v, "here", || {
                Some(1024)
            })
        };
        let ec = EngineConfig::default();

        let err = call(&sizes, Some(need - 1), &ec).expect_err("one byte short must refuse");
        assert!(
            err.to_string().contains("cannot be served on this device"),
            "{err}"
        );
        call(&sizes, Some(need), &ec).expect("an exact fit must be allowed");
        call(&sizes, None, &ec).expect("no budget to report: nothing to prove");

        // Sub-MiB buffers are never checked by the guard, so they cannot prove a refusal however
        // many there are.
        let tiny = vec![(mib - 1, mib - 1); 1024];
        call(&tiny, Some(1), &ec).expect("unguarded buffers must not count");

        // KV overflow spills what does not fit; a disabled guard refuses nothing.
        let mut overflow = EngineConfig::default();
        overflow.kv.overflow = true;
        call(&sizes, Some(1), &overflow).expect("INFR_KV_OVERFLOW must not refuse");
        // ...except on a unified-memory device, where the spill takes from the same budget.
        let unified = infr_vulkan::kv_spill_armed(overflow.kv.overflow, true, true);
        assert!(!unified, "a unified-memory device cannot spill");
        super::refuse_unallocatable_kv(
            &overflow,
            unified,
            &sizes,
            Some(1),
            32768,
            k,
            v,
            "",
            || None,
        )
        .expect_err("a unified-memory spill must refuse what the guard would");
        let mut unguarded = EngineConfig::default();
        unguarded.kernels.vulkan.no_vram_guard = true;
        call(&sizes, Some(1), &unguarded).expect("INFR_NO_VRAM_GUARD must not refuse");
    }

    /// A backend's KV row alignment pads every layer's capacity up to it — the global layers and
    /// the SWA rings alike — and an alignment of 1 (CPU, Vulkan) leaves every size as it was.
    #[test]
    fn kv_row_align_pads_capacity_and_one_changes_nothing() {
        let cfg = gemma3_12b();
        for want in [1usize, 17, 100, 4097] {
            for ring in [false, true] {
                for l in 0..cfg.n_layer {
                    let plain = super::kv_rows_at(&cfg, l, want, ring, 512, 1);
                    let padded = super::kv_rows_at(&cfg, l, want, ring, 512, 16);
                    assert_eq!(padded, plain.next_multiple_of(16), "layer {l}, ctx {want}");
                    assert_eq!(
                        super::kv_layer_alloc_bytes(
                            &cfg,
                            l,
                            want,
                            ring,
                            512,
                            1,
                            DType::F16,
                            DType::F16
                        ),
                        (
                            super::kv_side_bytes(
                                DType::F16,
                                plain * super::kv_row_elems(&cfg, l).0
                            ),
                            super::kv_side_bytes(
                                DType::F16,
                                plain * super::kv_row_elems(&cfg, l).1
                            ),
                        ),
                        "align 1 must allocate exactly the unpadded rows"
                    );
                }
            }
        }
        // A one-shot context of prompt + reply + 1 that ends inside a block gets the block's rows.
        assert_eq!(super::kv_rows_at(&cfg, 0, 101, false, 512, 16), 112);
    }

    /// A qwen35 model in miniature: DeltaNet layers with a full-attention layer after them.
    fn qwen35_small() -> Config {
        Config {
            n_layer: 4,
            n_embd: 64,
            n_head: 4,
            n_kv: 2,
            head_dim: 16,
            qwen35: true,
            full_attn_interval: 4,
            ssm_d_conv: 4,
            ssm_d_inner: 64,
            ssm_d_state: 16,
            ssm_n_group: 2,
            ssm_dt_rank: 4,
            ..Default::default()
        }
    }

    /// Every KV geometry family the sizing tests cover, by name: recurrent state beside attention
    /// (qwen35; LFM3's short convolutions around a sliding-window layer), sliding-window rings
    /// (gemma3), MLA with and without the lightning indexer's cache, DeepSeek V4's shared K/V row
    /// and compressed tiers, and plain attention.
    fn kv_families() -> Vec<(&'static str, Config)> {
        let lfm3 = Config {
            n_layer: 3,
            n_embd: 64,
            n_head: 4,
            n_kv: 2,
            head_dim: 16,
            lfm3: true,
            shortconv_layers: vec![true, false, true],
            shortconv_l_cache: 4,
            swa_window: 4,
            ..Default::default()
        };
        let deepseek32 = Config {
            deepseek32: true,
            indexer_head_size: 128,
            ..deepseek_v2_lite_kv()
        };
        let deepseek4 = Config {
            n_layer: 5,
            n_embd: 4096,
            n_head: 64,
            n_kv: 1,
            head_dim: 512,
            swa_window: 128,
            indexer_head_size: 128,
            compress_ratios: vec![0, 0, 4, 128, 4],
            deepseek4: true,
            ..Default::default()
        };
        vec![
            ("qwen35", qwen35_small()),
            ("lfm3", lfm3),
            ("gemma3 ring", gemma3_12b()),
            ("mla", deepseek_v2_lite_kv()),
            ("mla + indexer", deepseek32),
            ("deepseek4", deepseek4),
            ("plain", qwen3_14b()),
        ]
    }

    /// Bytes the runner's cold init allocates for one slot, summed from the allocation sites' own
    /// sizers: each layer's two buffers and a V4 layer's compressed buffers.
    fn cold_init_bytes(
        cfg: &Config,
        ctx: usize,
        ring: bool,
        ubatch: usize,
        row_align: usize,
        k_fmt: DType,
        v_fmt: DType,
    ) -> usize {
        (0..cfg.n_layer)
            .map(|l| {
                let (k, v) =
                    super::kv_layer_alloc_bytes(cfg, l, ctx, ring, ubatch, row_align, k_fmt, v_fmt);
                let comp: usize = super::dsv4_comp_bufs(cfg, l, ctx, k_fmt)
                    .iter()
                    .map(super::Dsv4CompBuf::bytes)
                    .sum();
                k + v + comp
            })
            .sum()
    }

    /// Bytes of a slot's recurrent-state snapshot, from the config alone: a second f32 copy of
    /// every recurrent layer's two state buffers.
    fn snapshot_bytes(cfg: &Config) -> usize {
        (0..cfg.n_layer)
            .filter_map(|l| cfg.recurrent_state_elems(l))
            .map(|(k, v)| (k + v) * 4)
            .sum()
    }

    /// The estimate is the sum of what a slot allocates, for any geometry: each layer's two cache
    /// buffers as the runner and `SeamKv::fork` size them (`kv_layer_alloc_bytes`), a V4
    /// compressed layer's buffers as `alloc_dsv4_comp_bufs` sizes them, and a second copy of each
    /// recurrent layer's state for its snapshot, every buffer rounded up to the allocator's byte
    /// alignment. Seeded draws over [`kv_families`], contexts, chunk heights, row and byte
    /// alignments and per-side formats. An estimate that pads a row, drops the snapshot or rounds
    /// a sum instead of each buffer differs from the sum.
    #[test]
    fn kv_estimate_is_the_sum_of_the_slot_allocations() {
        let formats = [
            DType::F16,
            DType::Q8_0,
            DType::F32,
            DType::Q4_0,
            DType::Turbo3,
        ];
        // No rounding; a word; the integrated GPU's 4 KiB pages; 2 MiB pages from 2 MiB with 4
        // KiB below and a cost per allocation.
        let costs = [
            super::KvAllocCost::IDENTITY,
            super::KvAllocCost::new(16, 16, u64::MAX, 0),
            super::KvAllocCost::new(4096, 4096, 2 << 20, 0),
            super::KvAllocCost::new(4096, 2 << 20, 2 << 20, 512),
        ];
        let mut rng = infr_core::pager::shapes::XorShift64::new(0x5eed_4b56);
        let (mut recurrent, mut ringed, mut compressed, mut mixed, mut rounded) = (0, 0, 0, 0, 0);
        for (name, cfg) in &kv_families() {
            for _ in 0..200 {
                let ctx = match rng.below(4) {
                    0 => rng.below(3),
                    1 => 1 + rng.below(200),
                    _ => 1 + rng.below(1 << 18),
                };
                let ring = rng.below(2) == 1;
                let ubatch = [8, 128, 512, 1024][rng.below(4)];
                let align = super::KvAlign {
                    rows: [1, 16, 64][rng.below(3)],
                    commit: costs[rng.below(costs.len())],
                };
                let k_fmt = formats[rng.below(formats.len())];
                let v_fmt = formats[rng.below(formats.len())];
                let committed = |b: usize| align.commit.committed(b as u64) as usize;
                let mut want = 0usize;
                let mut pairs = Vec::new();
                for l in 0..cfg.n_layer {
                    let (k, v) = super::kv_layer_alloc_bytes(
                        cfg, l, ctx, ring, ubatch, align.rows, k_fmt, v_fmt,
                    );
                    pairs.push((k, v));
                    want += committed(k) + committed(v);
                    rounded += usize::from(committed(k) != k || committed(v) != v);
                    if cfg.is_recurrent_layer(l) {
                        want += committed(k) + committed(v);
                        recurrent += 1;
                    }
                    let comp = super::dsv4_comp_bufs(cfg, l, ctx, k_fmt);
                    compressed += comp.len();
                    want += comp.iter().map(|b| committed(b.bytes())).sum::<usize>();
                    ringed += usize::from(
                        super::kv_rows_at(cfg, l, ctx, ring, ubatch, align.rows)
                            < ctx.next_multiple_of(align.rows),
                    );
                }
                mixed += usize::from(k_fmt != v_fmt);
                // The runner's cold init allocates its K/V pairs from the head of this list.
                let init: Vec<(usize, usize)> =
                    super::kv_init_allocs(cfg, ctx, ring, ubatch, align.rows, k_fmt, v_fmt)
                        .collect();
                assert_eq!(
                    &init[..cfg.n_layer],
                    &pairs[..],
                    "{name}: the pairs lead the list"
                );
                assert_eq!(
                    super::kv_bytes_estimate_fmt(cfg, ctx, ring, ubatch, align, k_fmt, v_fmt),
                    want as u64,
                    "{name}: ctx {ctx}, ring {ring}, chunk {ubatch}, {align:?}, {k_fmt:?}/{v_fmt:?}"
                );
            }
        }
        assert!(
            recurrent > 0 && ringed > 0 && compressed > 0 && mixed > 0 && rounded > 0,
            "the draws missed a family: {recurrent} recurrent layers, {ringed} ringed, \
             {compressed} compressed buffers, {mixed} mixed-format draws, {rounded} rounded layers"
        );
    }

    /// Each side is sized in its own format, by hand. `INFR_KV_TYPE_K=q8_0` with V at f16 on a
    /// plain layer of 2 KV heads x 64: at 1000 rows, K is 128 000 elements in 34-byte blocks of
    /// 32 and V is 128 000 f16. At a row alignment of 16 the rows are 1008; on an MLA model with
    /// the indexer cache the two sides also differ in width (576 and 128 elements), so a swap of
    /// the formats changes the estimate too. A byte alignment rounds each buffer, not their sum.
    #[test]
    fn kv_sizes_price_each_side_in_its_own_format() {
        let plain = Config {
            n_layer: 1,
            n_embd: 256,
            n_head: 4,
            n_kv: 2,
            head_dim: 64,
            ..Default::default()
        };
        let (k8, v16) = (DType::Q8_0, DType::F16);
        assert_eq!(
            super::kv_layer_alloc_bytes(&plain, 0, 1000, false, 512, 1, k8, v16),
            (136_000, 256_000)
        );
        assert_eq!(
            super::kv_bytes_estimate_fmt(&plain, 1000, false, 512, super::KvAlign::NONE, k8, v16),
            392_000
        );
        let page = super::KvAlign {
            rows: 1,
            commit: super::KvAllocCost::new(4096, 4096, u64::MAX, 0),
        };
        assert_eq!(
            super::kv_bytes_estimate_fmt(&plain, 1000, false, 512, page, k8, v16),
            139_264 + 258_048
        );
        // Classes and a cost per allocation: K (136 000 B) is below a 200 000-byte boundary and
        // rounds to 4 KiB, V (256 000 B) is above it and rounds to 64 KiB (262 144), and each
        // buffer adds 100 bytes.
        let classes = super::KvAlign {
            rows: 1,
            commit: super::KvAllocCost::new(4096, 65_536, 200_000, 100),
        };
        assert_eq!(
            super::kv_bytes_estimate_fmt(&plain, 1000, false, 512, classes, k8, v16),
            139_264 + 100 + 262_144 + 100
        );
        assert_eq!(
            super::kv_layer_alloc_bytes(&plain, 0, 1000, false, 512, 16, k8, v16),
            (137_088, 258_048)
        );
        let mla = Config {
            deepseek32: true,
            indexer_head_size: 128,
            ..deepseek_v2_lite_kv()
        };
        let rows16 = super::KvAlign {
            rows: 16,
            commit: super::KvAllocCost::IDENTITY,
        };
        // Per layer: 1008 x 576 q8_0 = 616 896 bytes of K, 1008 x 128 f16 = 258 048 of V.
        assert_eq!(
            super::kv_bytes_estimate_fmt(&mla, 1000, false, 512, rows16, k8, v16),
            27 * (616_896 + 258_048)
        );
    }

    /// The refuse rung's floor is what the runner's cold init allocates and the fit's estimate
    /// adds the recurrent-state snapshot to it. At the floor's own choices (the cheapest format,
    /// the ring wherever one is possible, the shortest chunk), for every family: the floor is the
    /// init's bytes exactly, and the estimate is those plus the snapshot, which only the recurrent
    /// families have.
    #[test]
    fn kv_floor_is_the_cold_init_and_the_estimate_adds_the_snapshot() {
        let ec = EngineConfig::default();
        let want = 2048;
        let ub = super::ubatch_candidates(&ec)
            .into_iter()
            .min()
            .expect("a chunk height");
        for (name, cfg) in &kv_families() {
            let ring = super::kv_ring_wanted(cfg, &ec);
            // Asked for f16, the floor prices q8_0, the cheaper format the runner can fall to.
            let floor: usize =
                super::kv_floor_alloc_bytes(cfg, &ec, want, 1, 1, DType::F16, DType::F16)
                    .iter()
                    .map(|&(k, v)| k + v)
                    .sum();
            let init = cold_init_bytes(cfg, want, ring, ub, 1, DType::Q8_0, DType::Q8_0);
            assert_eq!(floor, init, "{name}: the floor is the cold init");
            let snapshot = snapshot_bytes(cfg);
            assert_eq!(
                snapshot > 0,
                matches!(*name, "qwen35" | "lfm3"),
                "{name}: a snapshot"
            );
            let estimate = super::kv_bytes_estimate_fmt(
                cfg,
                want,
                ring,
                ub,
                super::KvAlign::NONE,
                DType::Q8_0,
                DType::Q8_0,
            );
            assert_eq!(
                estimate,
                (init + snapshot) as u64,
                "{name}: the estimate adds the snapshot"
            );
        }
    }

    /// The pre-load floor must never price more than any cache the runner's cold init can still
    /// allocate — that is what lets the refuse rung read it before a weight is loaded without ever
    /// refusing a run that would have worked. Every family, both formats the runner falls back to,
    /// every chunk height placement may settle on, ring on and off, and two row alignments.
    #[test]
    fn kv_floor_never_exceeds_a_cache_the_runner_could_allocate() {
        let ec = EngineConfig::default();
        let want = 32768;
        let total =
            |sizes: Vec<(usize, usize)>| -> usize { sizes.iter().map(|&(k, v)| k + v).sum() };
        for (name, cfg) in &kv_families() {
            for row_align in [1, 16] {
                let floor = |n: usize, dt: DType| {
                    total(super::kv_floor_alloc_bytes(
                        cfg, &ec, want, n, row_align, dt, dt,
                    ))
                };
                let f16 = floor(1, DType::F16);
                for ub in super::ubatch_candidates(&ec) {
                    for ring in [false, true] {
                        for dt in [DType::F16, DType::Q8_0] {
                            let exact = cold_init_bytes(cfg, want, ring, ub, row_align, dt, dt);
                            assert!(
                                f16 <= exact,
                                "{name}: floor {f16} > the {dt:?} cache at chunk {ub}, ring \
                                 {ring}, row align {row_align} ({exact})"
                            );
                        }
                    }
                }
                assert_eq!(
                    floor(2, DType::F16),
                    2 * f16,
                    "{name}: each slot owns a cache"
                );
            }
        }
        let gemma3 = gemma3_12b();
        let floor = |dt: DType| {
            total(super::kv_floor_alloc_bytes(
                &gemma3, &ec, want, 1, 1, dt, dt,
            ))
        };
        assert!(
            floor(DType::Turbo2) < floor(DType::F16),
            "a requested format cheaper than q8_0 is the floor"
        );
    }

    #[test]
    fn placement_pins_are_per_scope_not_process_global() {
        // The multi-model fix: two independent sessions' pins are isolated. Session A pins a chunk
        // and q8; inside A's scope the readers see them, and OUTSIDE any scope (or in a fresh
        // session B's scope) they are unset — the old process-global `OnceLock` leaked A's decision
        // into B (a silent no-op `.set()`).
        let a = std::sync::Arc::new(PlacementPins::default());
        let b = std::sync::Arc::new(PlacementPins::default());
        {
            let _sa = PlacementScope::enter(a.clone());
            super::pin_ubatch(512);
            super::pin_kv_auto_q8();
            assert!(super::kv_auto_q8(), "A's scope sees its own q8 pin");
            assert_eq!(a.ubatch.load(std::sync::atomic::Ordering::Relaxed), 512);
        }
        {
            let _sb = PlacementScope::enter(b.clone());
            assert!(!super::kv_auto_q8(), "B must NOT inherit A's q8 pin");
            assert_eq!(
                b.ubatch.load(std::sync::atomic::Ordering::Relaxed),
                0,
                "B must NOT inherit A's chunk"
            );
        }
    }
}
