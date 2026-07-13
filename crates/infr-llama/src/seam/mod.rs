//! CPU model runner — builds and drives the agnostic decode [`Graph`] through [`CpuBackend`].
//! The backend itself lives in `infr-cpu`; this module is the model-specific "glue" that
//! assembles the layer graph, uploads weights, and steps the KV cache.
//!
//! Split into submodules (pure move, zero behavior change): [`weights`] holds the per-layer
//! weight-handle structs and the persistent seam session state; [`sc`] holds the DiffusionGemma
//! self-conditioning pieces; [`runner`] holds the giant backend-generic `generate_dense_backend`
//! and its `DecodeHandles`. This file keeps the thin per-backend entry wrappers, the `verify_*`
//! family, and the small shared helpers every submodule reaches into.
#![allow(clippy::too_many_arguments)]

use crate::{dequant_block, Config, GenStats, PerLayerEmbd};
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::tensor::DType;
use infr_core::WeightSource;
use infr_cpu::CpuBackend;
use infr_gguf::{Gguf, TensorBytes};

pub mod model;
mod runner;
mod sc;
mod weights;

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
    /// `Config::from_gguf` already validated the tensor exists at load, so this can only fail on a
    /// corrupt/truncated GGUF.
    pub(crate) fn get(&self) -> &'a [f32] {
        self.cell.get_or_init(|| {
            crate::quant::load_tensor_dequant(self.gguf, "token_embd.weight")
                .expect("token_embd.weight: validated at load by Config::from_gguf")
                .0
        })
    }
}

// ─── Qwen3 dense CPU decode runner ───────────────────────────────────────────────
//
// Builds the n=1 decode Graph and drives it through `CpuBackend`, one token at a time, for BOTH
// prompt ingestion (looped) and generation — so no GEMM/flash prefill kernels are needed on CPU.
// The KV cache grows one row per step. Validates the agnostic seam end-to-end against the GPU path.

/// Greedy CPU generation for a decoder (Qwen3 / Llama / Gemma 3 / Gemma 4 dense+E2B / qwen3moe). The
/// attention block is shared; the FFN is either a dense gated FFN or a routed-expert MoE bank; gemma4
/// E2B adds per-layer input embeddings + KV-layer sharing. `prompt` is the full token prefix; returns
/// the generated continuation. Stops at EOS or `max_new`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_dense_cpu(
    g: &Gguf,
    cfg: &Config,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    req: Option<&crate::sampling::RequestCtx>,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    // Thin CPU wrapper over the backend-generic runner: a CpuBackend + a zero-copy weight binder
    // (maps each tensor straight from the GGUF mmap — no alloc, no memcpy).
    let cpu_be = CpuBackend::new();
    generate_dense_backend(
        &cpu_be,
        &|_name, tb, dt, _n| match tb {
            WBytes::Mmap(tb) => Ok((cpu_be.map_weight(tb), dt)),
            // Owned bytes (combined gate+up) never reach the CPU binder — combined_gu is false —
            // but stay correct if they ever do.
            WBytes::Owned(v) => {
                let buf = cpu_be
                    .alloc(v.len().max(1), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                cpu_be
                    .upload(buf.as_ref(), &v)
                    .map_err(|e| anyhow!("{e}"))?;
                Ok((buf, dt))
            }
        },
        g,
        cfg,
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
        req,
    )
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
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_vulkan_session(
        vk,
        g,
        cfg,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
        &mut None,
        prompt.len() + max_new + 1,
        None, // constraint
        None, // req: the one-shot runner is a sole sequence — env sampling, no gate
    )
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
    // Placement can allocate + upload (the pager arenas, a weight re-bind), i.e. it RECORDS on the
    // Vulkan command pool — so it takes a turn on the baton like any other GPU region. Scoped: the
    // baton is released before the runner starts stepping. See `StepGate`.
    let bind = {
        let _gp = req.and_then(|r| r.gate_pass());
        vulkan_moe_binder(vk, g, cfg, state.is_none(), want_ctx)?
    };
    let out = generate_dense_backend(
        vk, &*bind, g, cfg, token_embd, ple, prompt, max_new, on_token, state, want_ctx,
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
/// Derivation — measured on gemma-4-31B UD-Q5_K_XL (n_embd 5376, n_ff 32768, n_head 32,
/// head_dim 512/256 full/SWA) on a 24 GiB 7900 XTX with a tagged allocation trace
/// (INFR_ALLOC_TRACE-style eprintln on every ≥16 MiB activation alloc) at a 1024-row prefill
/// chunk, ctx 2064:
/// - Internal graph tensors (`alloc_scratch`, ~850 MiB): fused gate_up out `[rows, 2*n_ff]` f32
///   (256 MiB) + activated intermediate `[rows, n_ff]` f32 (128 MiB) + fused qkv staging
///   (168 MiB = rows*8*n_embd*4 here) + ~a dozen `[rows, n_embd]`-class f32/f16 temps.
///   Modeled as `12*n_ff + 96*n_embd` per row (the n_embd umbrella also absorbs the
///   lin_a16/mmq activation-quant pools, which are n_embd/n_ff-wide f16/i8).
/// - Attention pools — the term a previous calibration MISATTRIBUTED as "rows*vocab*2
///   whole-chunk f16 logits" (batched prefill has run a last-row-only m=1 LM head since long
///   before that trace, and is fully headless now — no logits allocation scales with rows;
///   2*vocab merely coincided with the real per-row attention bytes on this model, where
///   2*262144 == 8*n_head*head_dim*4 at head_dim 512):
/// - `nonfa_pv`/`flash_po`: `8*rows*n_head*head_dim*4` per DISTINCT head shape — gemma4
///   alternates SWA(256)/full(512) head dims, so BOTH pools live at once (512 + 256 MiB
///   measured) → `32*n_head*(head_dim + head_dim_swa-if-distinct)` per row.
/// - `nonfa_s` (score tiles, non-flash tier only — any model with SWA layers or
///   head_dim != 128): `n_head*rows*kv_pad*2`, kv_pad = kv_len rounded up to 256. The pool
///   key includes the byte size, so as kv grows across chunks stale sizes are retained —
///   modeled as 2 live pools at the final ctx: `4*n_head*ctx_pad` per row. Uniform-hd-128
///   no-SWA models (llama/qwen3) ride the single-pass flash tier: no score tiles, only the
///   (negligible) flash_pm/pl partials — term skipped.
///
/// All times a 1.25 margin for unmeasured tails (split-path partials, per-shape pool
/// duplicates), plus a fixed 256 MiB for what shapes don't scale: gpu-allocator's block
/// granularity, retained upload staging (device-local under ReBAR), and the weight-buffer
/// u32/dedicated-alloc padding not in `weight_footprint`. Deliberately a slight over-reserve:
/// under-reserving makes the alloc-time VRAM guard error a live request mid-prefill (exactly
/// what the old formula did on this 31B at pp2048 — the second chunk's 512 MiB `nonfa_pv`
/// tripped the guard mid-run), over-reserving only streams/clamps a borderline model.
pub(crate) fn dense_act_reserve(cfg: &Config, want_ctx: usize) -> u64 {
    dense_act_reserve_at(cfg, want_ctx, ubatch_rows())
}

/// [`dense_act_reserve`] at an EXPLICIT chunk height (the try-resident sweep).
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
        4 * cfg.n_head * kv_span.next_multiple_of(256)
    };
    let per_row = (12 * cfg.n_ff + 96 * cfg.n_embd + attn_pv + attn_s) as u64;
    const FIXED: u64 = 256 * 1024 * 1024;
    FIXED + rows * per_row * 5 / 4
}

/// Batched-prefill micro-batch: rows per prefill chunk (INFR_UBATCH, default 1024). ONE reader
/// funnel — the prefill loop, the activation reserve, and the SWA ring sizing below all derive
/// from this, because the ring's correctness bound is "window + one whole prefill chunk".
pub(crate) fn ubatch_rows() -> usize {
    std::env::var("INFR_UBATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(|| PINNED_UBATCH.get().copied().unwrap_or(1024))
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
pub(crate) fn ubatch_rows_parallel() -> usize {
    std::env::var("INFR_UBATCH_PARALLEL")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(256)
}

/// Placement-pinned prefill chunk (rows): set ONCE by the dense try-resident sweep when a smaller
/// chunk is what makes a big model fully resident (see `vulkan_moe_binder`'s dense tier —
/// residency at a 512-row chunk decodes ~10x faster than streaming at the PCIe ceiling). Read by
/// [`ubatch_rows`] when INFR_UBATCH is unset, so the prefill loop, the activation reserve, and
/// the SWA ring sizing all agree on the same height. Set BEFORE the runner's first KV allocation
/// (placement runs first in the same call), never changed after.
static PINNED_UBATCH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Placement-pinned auto-q8 KV: set ONCE when the placement ladder chooses a Q8_0 KV cache to
/// keep a session RESIDENT (dense try-resident tier in [`vulkan_moe_binder`]) or to avoid
/// shrinking a DEFAULT context (`SeamModel::clamp_default_ctx`) — the "q8 KV" rung between the
/// SWA ring and ctx-clamp/weight-streaming rungs of the VRAM placement ladder. Read by the
/// runner's per-side KV-format selection (and by every KV-footprint estimate) ONLY when the
/// user set none of INFR_KV_TYPE_K / INFR_KV_TYPE_V / INFR_KV_Q8 — an explicit setting always
/// wins, in both directions (an explicit `f16` forces f16 and the placement falls through to
/// the next rung). Policy, deliberately conservative:
///   - BOTH sides go q8_0 (the existing coherence-tested replayable config — coupled Q8 keeps
///     record-once decode replay; llama.cpp guidance says keep K >= V precision, and q8/q8
///     satisfies it symmetrically);
///   - never auto-picks anything BELOW q8_0 (the low-bit/turbo formats trade real quality and
///     stay explicit opt-ins).
///
/// Like [`PINNED_UBATCH`], set before the session's first KV allocation and never changed —
/// warm calls and rebuilt graphs must agree with the buffers they were sized with.
static PINNED_KV_Q8: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// Whether the placement ladder pinned auto-q8 KV for this process (see [`PINNED_KV_Q8`]).
pub(crate) fn kv_auto_q8() -> bool {
    PINNED_KV_Q8.get().is_some()
}

/// Pin auto-q8 KV from outside this module (the default-ctx clamp path in `model.rs`); the
/// binder's own rung sets [`PINNED_KV_Q8`] directly. Idempotent (OnceLock).
pub(crate) fn pin_kv_auto_q8() {
    let _ = PINNED_KV_Q8.set(());
}

/// True when the user expressed NO explicit KV-format choice — the only state auto-q8 may fill.
pub(crate) fn kv_env_unset() -> bool {
    std::env::var("INFR_KV_TYPE_K").is_err()
        && std::env::var("INFR_KV_TYPE_V").is_err()
        && std::env::var("INFR_KV_Q8").is_err()
}

/// Layout gate for a Q8_0 KV cache — every layer's KV row must be whole 32-elem blocks (the
/// same alignment the runner's own `parse_kv_fmt` gate checks; keeping them identical means a
/// pinned auto-q8 is never silently gated back to f16 with an under-sized placement estimate).
pub(crate) fn kv_q8_layout_ok(cfg: &Config) -> bool {
    (0..cfg.n_layer).all(|l| (cfg.layer_n_kv(l) * cfg.layer_head_dim(l)).is_multiple_of(32))
}

/// [`kv_rows`] at an EXPLICIT chunk height (the try-resident sweep prices candidate heights
/// before pinning one; everyone else goes through `kv_rows`/`ubatch_rows`).
pub(crate) fn kv_rows_at(
    cfg: &Config,
    l: usize,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
) -> usize {
    if ring && cfg.is_swa_layer(l) {
        want_ctx.min((cfg.swa_window + ubatch).next_multiple_of(64))
    } else {
        want_ctx
    }
}

/// Row capacity of layer `l`'s K/V cache at context `want_ctx`. With `ring` (SWA ring sizing on
/// for this session — see [`kv_ring_wanted`] + the backend's `Capabilities::kv_swa_ring`), a
/// sliding-window layer allocates only `min(want_ctx, round64(window + ubatch))` rows and the
/// backends write/read position `p` at row `p % rows` (WriteKv/Attention ring semantics).
///
/// Correctness bound: during one forward of `B <= ubatch` rows starting at position `p0`, the
/// oldest position any query's window reaches is `p0 + 1 - window` and the newest written is
/// `p0 + B - 1` — at most `window + B - 1 <= window + ubatch` distinct live positions, so a ring
/// of `window + ubatch` rows never recycles a row the sliding-window mask hasn't ALREADY excluded
/// (that mask discards everything older than `pos - window`); attention output is therefore
/// identical to the full-context cache. Global (non-SWA) layers keep full `want_ctx` rows.
pub(crate) fn kv_rows(cfg: &Config, l: usize, want_ctx: usize, ring: bool) -> usize {
    kv_rows_at(cfg, l, want_ctx, ring, ubatch_rows())
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
///   - INFR_NO_KV_RING=1 (A/B and escape hatch).
pub(crate) fn kv_ring_wanted(cfg: &Config) -> bool {
    let fmt_ok = |var: &str| {
        matches!(
            std::env::var(var).ok().as_deref(),
            None | Some("f16") | Some("F16") | Some("q8_0") | Some("q8") | Some("Q8_0")
        )
    };
    cfg.swa_window > 0
        && !cfg.diffusion_gemma
        && std::env::var("INFR_NO_KV_RING").is_err()
        && fmt_ok("INFR_KV_TYPE_K")
        && fmt_ok("INFR_KV_TYPE_V")
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
    first_load: bool,
    want_ctx: usize,
) -> AResult<Box<BindWeight<'a>>> {
    // ── MoE expert placement ─────────────────────────────────────────────────────────────────
    // The pager (`infr_vulkan::pager`) is the ONLY MoE offload mechanism — the legacy
    // host-visible (HostWeights/GTT) split and its INFR_NCMOE knob are gone. Tiers, in
    // precedence order:
    //   1. `INFR_CACHE=<size>` EXPLICIT override — force EVERY expert layer through the pager
    //      with that byte budget, regardless of whether the banks would fit resident. Lets a
    //      caller (or a test) force the paged path deterministically instead of depending on
    //      this box's free VRAM — see the `gpu_seam_paged_moe_matches_*` tests. The value is the
    //      shared size grammar (`infr_core::parse_size`): plain bytes, `k/m/g/t` 1024-suffixes
    //      (`INFR_CACHE=19g`), or a percentage of the device's AVAILABLE VRAM at first load
    //      (`INFR_CACHE=80%` — device-appropriate base: the cache lives in VRAM).
    //   2. Auto (unset): fully resident (the fast path, zero change) when the banks fit VRAM;
    //      otherwise the pager with budget = remaining VRAM after dense+KV+headroom.
    // Paging rides the adapter's paged executor split (`infr_vulkan::adapter::execute_static`'s
    // paged branch). FUSED gate_up banks (gemma-4 MoE / DiffusionGemma) page under `Role::Gate`
    // with a double-width slot, and MIXED-dtype roles (unsloth-dynamic quants bumping a subset
    // of layers' banks to a wider format) split into per-(role, slot_bytes) arena pools — see
    // `infr_vulkan::pager`'s MoE-session doc.
    let cache_override = std::env::var("INFR_CACHE")
        .ok()
        .and_then(|v| infr_core::parse_size(&v));

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
                // Percent resolves against AVAILABLE VRAM at this (first-load, pre-upload)
                // moment — the same consistent snapshot the auto tier uses below.
                pager_budget_bytes = spec.resolve(vk.vram().available);
            }
            None => {
                let fp = crate::weights::weight_footprint(g);
                let vram = vk.vram();
                // Per-layer rows: SWA layers ring at window+ubatch rows (see `kv_rows`), so a
                // mostly-SWA model's KV prices far below n_layer * ctx. +64 rows/layer slop.
                let ring = kv_ring_wanted(cfg);
                let kv_bytes: u64 = (0..cfg.n_layer)
                    .map(|l| {
                        (cfg.layer_n_kv(l) * cfg.layer_head_dim(l) * 2 * 2) as u64
                            * (kv_rows(cfg, l, want_ctx, ring) as u64 + 64)
                    })
                    .sum::<u64>();
                // 2 GiB: covers activation scratch (pooled, but per-tag sizes scale with n_embd/
                // n_ff and this budget calc's `fp`/`kv_bytes` are estimates, not the exact bytes
                // `alloc`/gpu-allocator's 256 MiB block granularity ends up committing) plus the
                // pager's own arena+staging+LUT allocations, which aren't counted in `fp` at all
                // (found empirically sizing Scout's 48-layer, 37 GB Q2_K pager placement — 512 MiB
                // undershot by a few hundred MiB and the guard rightly refused to over-commit).
                const ACT_HEADROOM: u64 = 2 * 1024 * 1024 * 1024;
                // Mixed oversize (an MoE model whose DENSE part alone doesn't fit) is out of
                // scope for dense layer streaming — fail with a clear message instead of letting
                // a degenerate expert budget stumble into the alloc-time VRAM guard's generic
                // over-commit error.
                if vram.available < fp.dense + kv_bytes {
                    return Err(anyhow!(
                        "this MoE model's dense weights ({:.2} GB) + KV cache ({:.2} GB) exceed \
                         available VRAM ({:.2} GB) — dense layer streaming does not cover MoE \
                         models' dense parts; reduce ctx or run on the CPU backend (INFR_CPU=1)",
                        fp.dense as f64 / 1e9,
                        kv_bytes as f64 / 1e9,
                        vram.available as f64 / 1e9,
                    ));
                }
                let budget = vram
                    .available
                    .saturating_sub(fp.dense + kv_bytes + ACT_HEADROOM);
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
    // The layer index of a `blk.{l}.…_exps…` tensor name.
    let exps_layer = |name: &str| -> Option<usize> {
        if !name.contains("_exps") {
            return None;
        }
        name.strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|l| l.parse::<usize>().ok())
    };
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
            eprintln!(
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
            let ring_bytes = infr_vulkan::pager::ring_bytes_policy(pager_budget_bytes);
            pager_budget_bytes = pager_budget_bytes.saturating_sub(ring_bytes as u64);
            let total_bytes: u64 = pool_blocks
                .iter()
                .map(|&(_, sb, nb)| (sb * nb) as u64)
                .sum::<u64>()
                .max(1);
            // Per-pool ceiling: each pool's arena is ONE SSBO binding, capped by the smaller of the
            // paged kernels' u32 word reach (16 GiB) and the device's maxStorageBufferRange (4 GiB
            // on RADV — found empirically: Scout's auto budget wanted a 7.6 GiB down arena, and
            // reads past the binding range came back as garbage → NaN logits → sentinel router ids).
            // `GpuPager::new` hard-errors past this; clamping here keeps an oversized budget usable
            // instead of fatal. Applied PER POOL (their expert sizes differ — Scout: gate/up 13.8 MB
            // vs down 18 MB — so a shared count dragged to the largest pool's cap strands budget the
            // smaller pools could hold as real hit rate; see `MoePoolSpec`'s doc). Splitting a pool
            // across several arena buffers (or u64 shader addressing where the device allows bigger
            // buffers) is the lift that raises the cap.
            let arena_cap = infr_vulkan::pager::GpuPager::max_arena_bytes(vk);
            let pools: Vec<infr_vulkan::pager::MoePoolSpec> = pool_blocks
                .iter()
                .map(|&(role, sb, nb)| {
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
                    // next to any real budget (Scout: 16 x ~18 MB per role).
                    let floor = n_expert.min(nb).max(1);
                    let cap = ((arena_cap / sb as u64) as usize).max(1);
                    let budget_slots = ((share / sb as u64) as usize).clamp(floor, nb);
                    if budget_slots > cap {
                        eprintln!(
                            "MoE pager: clamping a pool's {budget_slots} -> {cap} slots (per-pool \
                         arena capped by the device's storage-buffer range / u32 word addressing)"
                        );
                    }
                    infr_vulkan::pager::MoePoolSpec {
                        role,
                        slot_bytes: sb,
                        n_slots: budget_slots.min(cap),
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
            eprintln!(
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
    //      weights + KV + the honest dense activation reserve (`dense_act_reserve`) fit live
    //      VRAM; otherwise stream with budget = remaining VRAM after resident-weights+KV+reserve.
    //      An explicit oversized INFR_CTX whose KV can't sit beside resident weights falls back
    //      to streaming the same way (never clamped here — the ctx the caller asked for is kept).
    // Streamable = the per-layer Linear projection groups whose dtype has offset-capable native
    // kernels (`native_dense_supported`, F16/F32 excluded — `matmul_proj`/`linear_f32` take no
    // weight offset) and whose bytes upload unmodified from the mmap (the qwen2 NEOX q/k row
    // permute rewrites bytes at load, so those tensors stay resident).
    let mut dense_plan: std::collections::HashMap<String, (usize, u32, Vec<String>)> =
        std::collections::HashMap::new();
    if first_load && cfg.moe.is_none() {
        let fuse_gu = runner::fuse_gu_decision(vk.capabilities().combined_gu, g, cfg);
        let fuse_qkv = runner::fuse_qkv_decision(vk.capabilities().combined_gu, g, cfg);
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
        // missing (DeltaNet layers, gemma4 V-less layers) or whose dtype lacks offset-capable
        // kernels simply stays resident.
        let eligible: Vec<(Vec<String>, infr_core::DType, usize, usize)> = groups
            .into_iter()
            .filter_map(|comps| {
                let infos: Vec<_> = comps.iter().map(|n| tinfo(n)).collect::<Option<_>>()?;
                let dt = infos[0].dtype;
                if !infos.iter().all(|t| t.dtype == dt)
                    || !infr_vulkan::linear::native_dense_supported(dt)
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
        // Per-layer rows: SWA layers ring at window+ubatch rows (see `kv_rows`) — this is what
        // lets a mostly-SWA model (gemma-4-31B: 50/60 layers SWA) price its KV small enough to
        // take the try-resident tier at real contexts instead of streaming. +64 rows/layer slop.
        let ring = kv_ring_wanted(cfg);
        // KV bytes at an EXPLICIT chunk height and side format — the ONE pricing helper every
        // decision below shares (try-resident, the chunk sweeps, the auto-q8 rung, the streaming
        // budget), so they all price exactly the allocation the runner will make. `q8` prices
        // BOTH sides Q8_0 (34 bytes / 32 elems, mirroring `kv_fmt_bytes`); false = f16 (2 B/elem).
        let kv_total_at = |ubatch: usize, q8: bool| -> u64 {
            (0..cfg.n_layer)
                .map(|l| {
                    let elems = (cfg.layer_n_kv(l) * cfg.layer_head_dim(l)) as u64
                        * (kv_rows_at(cfg, l, want_ctx, ring, ubatch) as u64 + 64);
                    if q8 {
                        2 * (elems / 32 * 34).next_multiple_of(4)
                    } else {
                        2 * 2 * elems
                    }
                })
                .sum()
        };
        // Does weights + KV + the honest activation reserve fit live VRAM at this (chunk, fmt)?
        let fits = |ubatch: usize, q8: bool| {
            fp.total() + kv_total_at(ubatch, q8) + dense_act_reserve_at(cfg, want_ctx, ubatch)
                <= vram.available
        };
        // Try-resident-first: a dense model goes FULLY RESIDENT (the exact pre-streaming fast
        // path) whenever weights + this session's KV + an HONEST dense activation estimate fit
        // the live free VRAM; only a genuine miss streams. The MoE tier's 2 GiB ACT_HEADROOM is
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
        // height for residency is strictly the right call. Pinned process-wide (PINNED_UBATCH)
        // so the prefill loop and the runner's ring sizing use exactly the priced height; an
        // explicit INFR_UBATCH disables the sweep (the user's height is authoritative). Runs
        // BEFORE the auto-q8 rung below: a shorter prefill chunk costs only some prefill
        // throughput, while q8 KV costs ~10-16% GQA decode — prefer the cheaper concession.
        let mut resident = fits(ubatch_rows(), kv_auto_q8());
        if !resident && std::env::var("INFR_UBATCH").is_err() && cache_override.is_none() {
            for cand in [512usize, 256, 128] {
                if fits(cand, kv_auto_q8()) {
                    let _ = PINNED_UBATCH.set(cand);
                    // Re-read through the pin (a racing earlier set wins — use whatever stuck).
                    if fits(ubatch_rows(), kv_auto_q8()) {
                        eprintln!(
                            "dense placement: resident with a {}-row prefill chunk (the default \
                             1024-row chunk's activation reserve wouldn't fit); set INFR_UBATCH \
                             to override",
                            ubatch_rows().min(want_ctx),
                        );
                        resident = true;
                    }
                    break;
                }
            }
        }
        // ── auto-q8 KV rung (the placement-ladder step between the SWA ring and streaming):
        // f16 KV missed residency at every chunk height, but a Q8_0 cache (roughly HALF the KV
        // bytes) might fit — placing RESIDENT with q8 KV beats the remaining rungs by an order
        // of magnitude (streaming decodes at the PCIe ceiling; the explicit-ctx path never
        // clamps). Gates: the user set NO KV format (see `PINNED_KV_Q8`'s policy doc — explicit
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
            && kv_env_unset()
            && kv_q8_layout_ok(cfg)
        {
            let ub_now = ubatch_rows();
            let mut cands = vec![ub_now];
            if std::env::var("INFR_UBATCH").is_err() {
                cands.extend([512usize, 256, 128].into_iter().filter(|&c| c < ub_now));
            }
            for cand in cands {
                if fits(cand, true) {
                    let _ = PINNED_KV_Q8.set(());
                    if cand != ubatch_rows() {
                        let _ = PINNED_UBATCH.set(cand);
                    }
                    // Re-read through the pins (racing earlier sets win — use whatever stuck).
                    if kv_auto_q8() && fits(ubatch_rows(), true) {
                        eprintln!(
                            "kv auto-quant: q8_0 (f16 KV wouldn't fit resident at ctx={want_ctx}; \
                             INFR_KV_TYPE_K/V=f16 to force f16)"
                        );
                        resident = true;
                    }
                    break;
                }
            }
        }
        // Slot stride: the group's raw bytes padded to a whole number of quant blocks AND
        // u32 words, so every slot base is block-aligned (the kernels' element-offset weight
        // addressing needs `slot_byte_base = whole blocks`) and the arena binds as
        // `array<u32>`. Hoisted above the budget decision so the streaming-chunk sweep below
        // can price the full-arena need with the exact strides the pools use.
        let lcm = |a: usize, b: usize| {
            let gcd = {
                let (mut x, mut y) = (a, b);
                while y != 0 {
                    (x, y) = (y, x % y);
                }
                x
            };
            a / gcd * b
        };
        let stride_of = |dt: infr_core::DType, raw: usize| {
            raw.next_multiple_of(lcm(infr_gguf::block_layout(dt).1, 4))
        };
        let budget = match cache_override {
            Some(spec) => Some(spec.resolve(vram.available)),
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
                // from the reserve). Pinned via PINNED_UBATCH like the residency sweep, so the
                // prefill loop, the runner's ring sizing, and this budget all agree.
                let q8 = kv_auto_q8();
                let base = fp.total() - streamable_resident;
                let budget_at = |ub: usize| {
                    vram.available.saturating_sub(
                        base + kv_total_at(ub, q8) + dense_act_reserve_at(cfg, want_ctx, ub),
                    )
                };
                if std::env::var("INFR_UBATCH").is_err() && !eligible.is_empty() {
                    let need: u64 = eligible
                        .iter()
                        .map(|(_, dt, raw, _)| stride_of(*dt, *raw) as u64)
                        .sum();
                    // "Covers": the budget minus its own upload-ring share holds every block.
                    let covers = |b: u64| {
                        b.saturating_sub(infr_vulkan::pager::ring_bytes_policy(b) as u64) >= need
                    };
                    let ub_now = ubatch_rows();
                    let mut cands = vec![ub_now];
                    cands.extend([512usize, 256, 128].into_iter().filter(|&c| c < ub_now));
                    let pick = cands
                        .iter()
                        .copied()
                        .find(|&c| covers(budget_at(c)))
                        .unwrap_or(*cands.last().expect("cands is never empty"));
                    if pick != ub_now {
                        let _ = PINNED_UBATCH.set(pick);
                    }
                }
                Some(budget_at(ubatch_rows()))
            }
        };
        if let (Some(mut budget), false) = (budget, eligible.is_empty()) {
            // Pools keyed by (dtype, stride); blocks assigned ids in layer order per pool.
            let mut pools: Vec<(infr_core::DType, usize, u64, usize)> = Vec::new(); // (dt, stride, elems_per_slot, n_blocks)
            let mut planned: Vec<(Vec<String>, usize, u32)> = Vec::new(); // (comps, pool, block_id)
            for (comps, dt, raw, _numel) in &eligible {
                let (blk_e, blk_b) = infr_gguf::block_layout(*dt);
                let stride = stride_of(*dt, *raw);
                let eps = (stride / blk_b * blk_e) as u64;
                let pool = match pools.iter().position(|&(d, s, ..)| d == *dt && s == stride) {
                    Some(i) => i,
                    None => {
                        pools.push((*dt, stride, eps, 0));
                        pools.len() - 1
                    }
                };
                let block_id = pools[pool].3 as u32;
                pools[pool].3 += 1;
                planned.push((comps.clone(), pool, block_id));
            }
            // The pinned upload ring lives in the same VRAM the arenas do — subtract it first.
            let ring_bytes = infr_vulkan::pager::ring_bytes_policy(budget);
            budget = budget.saturating_sub(ring_bytes as u64);
            let total_bytes: u64 = pools
                .iter()
                .map(|&(_, s, _, nb)| (s * nb) as u64)
                .sum::<u64>()
                .max(1);
            let arena_cap = infr_vulkan::pager::GpuPager::max_arena_bytes(vk);
            let specs: Vec<infr_vulkan::pager::DensePoolSpec> = pools
                .iter()
                .map(|&(_, stride, eps, nb)| {
                    // Proportional budget split (byte share == access share: every block is read
                    // exactly once per sweep). Floor 2 slots so the next block's upload can
                    // overlap the previous block's dispatch instead of serializing on one slot.
                    let share =
                        (budget as u128 * (stride * nb) as u128 / total_bytes as u128) as u64;
                    let floor = 2.min(nb).max(1);
                    // Caps: one SSBO binding per arena (device range), AND the kernels' u32
                    // ELEMENT reach — `n_slots * elems_per_slot + one block's numel` must fit
                    // u32 (a Q4_K pool's elements outnumber its bytes 1.78:1, so this cap can
                    // bind before the byte cap does).
                    let cap_bytes = ((arena_cap / stride as u64) as usize).max(1);
                    let cap_elems = ((u32::MAX as u64 / eps).saturating_sub(1) as usize).max(1);
                    let budget_slots = ((share / stride as u64) as usize).clamp(floor, nb);
                    infr_vulkan::pager::DensePoolSpec {
                        slot_bytes: stride,
                        n_slots: budget_slots.min(cap_bytes).min(cap_elems),
                        n_blocks: nb,
                        elems_per_slot: eps,
                    }
                })
                .collect();
            let alloc: u64 = specs
                .iter()
                .map(|s| (s.n_slots * s.slot_bytes) as u64)
                .sum();
            if alloc > budget.max(1) && cache_override.is_none() {
                // Auto tier only: the floors overran what's actually free — streaming can't help.
                return Err(anyhow!(
                    "dense weights exceed VRAM and the leftover budget ({:.2} GB) can't hold \
                     even the streaming floor ({:.2} GB) — reduce ctx or run on the CPU backend \
                     (INFR_CPU=1)",
                    budget as f64 / 1e9,
                    alloc as f64 / 1e9,
                ));
            }
            let cached: usize = specs.iter().map(|s| s.n_slots).sum();
            let n_blocks: usize = specs.iter().map(|s| s.n_blocks).sum();
            eprintln!(
                "dense streaming: {n_blocks} weight blocks across {} pools, {cached} slots \
                 cached ({:.2} GB arena + {:.2} GB ring; budget {:.2} GB; ctx={want_ctx}; \
                 chunk={})",
                specs.len(),
                alloc as f64 / 1e9,
                ring_bytes as f64 / 1e9,
                (budget + ring_bytes as u64) as f64 / 1e9,
                ubatch_rows().min(want_ctx),
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

    Ok(Box::new(move |name, tb, dt, _n| {
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
            let segments: Vec<std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>> = comps
                .iter()
                .map(|c| {
                    Ok(
                        std::sync::Arc::new(g.tensor_bytes_arc(c).map_err(|e| anyhow!("{e}"))?)
                            as std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>,
                    )
                })
                .collect::<AResult<_>>()?;
            let seg_total: usize = segments.iter().map(|s| s.as_ref().as_ref().len()).sum();
            if seg_total != tb.len() {
                return Err(anyhow!(
                    "dense streaming plan out of sync with the upload order for {name}: plan \
                     bytes {seg_total} != uploaded bytes {}",
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
                    segments,
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
                let stride_bytes = bytes.len() / n_expert;
                let placeholder = vk
                    .alloc_uninit(4, BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                let buf_id = infr_vulkan::pager::buffer_identity(placeholder.as_ref());
                let source = infr_vulkan::pager::ExpertSource {
                    bytes: std::sync::Arc::new(bytes.clone())
                        as std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>,
                    stride_bytes,
                    layer_base: (l * n_expert) as u32,
                };
                vk.register_paged_expert(role, buf_id, source)
                    .map_err(|e| anyhow!("{e}"))?;
                return Ok((placeholder, dt));
            }
        }
        let padded = infr_vulkan::linear::pad_to_u32_align(&tb);
        // alloc_uninit: the `upload` right below writes the buffer's FULL extent (it is sized to
        // exactly `padded.len()`), so the calloc contract's zero-fill is dead work — and an
        // expensive kind: on the device-local path it costs a `vkCmdFillBuffer` over the whole
        // model plus a submit + `queue_wait_idle` PER TENSOR, doubling the load's stall count.
        let buf = vk
            .alloc_uninit(padded.len(), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        vk.upload(buf.as_ref(), &padded)
            .map_err(|e| anyhow!("{e}"))?;
        Ok((buf, dt))
    }))
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
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_metal_session(
        mtl,
        g,
        cfg,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
        &mut None,
        prompt.len() + max_new + 1,
        None,
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
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
    state: &mut Option<SeamKv>,
    want_ctx: usize,
    constraint: Option<&mut crate::grammar::Constraint>,
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_backend(
        mtl,
        &|_name, tb, dt, _n| {
            let buf = mtl
                .alloc(tb.len().max(1), BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            mtl.upload(buf.as_ref(), &tb).map_err(|e| anyhow!("{e}"))?;
            Ok((buf, dt))
        },
        g,
        cfg,
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
        None,
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
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
    state: &mut Option<SeamKv>,
    want_ctx: usize,
) -> AResult<(Vec<f32>, f64)> {
    let mut logits = Vec::new();
    let (_, stats) = generate_dense_backend(
        mtl,
        &|_name, tb, dt, _n| {
            let buf = mtl
                .alloc(tb.len().max(1), BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            mtl.upload(buf.as_ref(), &tb).map_err(|e| anyhow!("{e}"))?;
            Ok((buf, dt))
        },
        g,
        cfg,
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
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<Vec<f32>> {
    let cpu_be = CpuBackend::new();
    let mut logits = Vec::new();
    let mut state = None;
    generate_dense_backend(
        &cpu_be,
        &|_name, tb, dt, _n| match tb {
            WBytes::Mmap(tb) => Ok((cpu_be.map_weight(tb), dt)),
            WBytes::Owned(v) => {
                let buf = cpu_be
                    .alloc(v.len().max(1), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                cpu_be
                    .upload(buf.as_ref(), &v)
                    .map_err(|e| anyhow!("{e}"))?;
                Ok((buf, dt))
            }
        },
        g,
        cfg,
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

/// [`verify_dense_cpu`]'s MTP Phase 1 twin (issue #33, docs/MTP.md): ALSO captures the LM-head
/// input rows (`h_out` — `DecodeHandles::h_out`'s doc) alongside the logits, for the
/// `lm_head(h_row) == logits_row` consistency check `docs/MTP.md`'s Phase 1 validation calls for.
/// Returns `(logits, h)`, both `[vocab]`/`[n_embd]` for the last prompt token.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_dense_cpu_with_h(
    g: &Gguf,
    cfg: &Config,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<(Vec<f32>, Vec<f32>)> {
    let cpu_be = CpuBackend::new();
    let mut logits = Vec::new();
    let mut h = Vec::new();
    let mut state = None;
    generate_dense_backend(
        &cpu_be,
        &|_name, tb, dt, _n| match tb {
            WBytes::Mmap(tb) => Ok((cpu_be.map_weight(tb), dt)),
            WBytes::Owned(v) => {
                let buf = cpu_be
                    .alloc(v.len().max(1), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                cpu_be
                    .upload(buf.as_ref(), &v)
                    .map_err(|e| anyhow!("{e}"))?;
                Ok((buf, dt))
            }
        },
        g,
        cfg,
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
/// a whole prompt in one call (`docs/MTP.md`'s `process()` runs after every target ubatch, not just
/// the sampled row). Dense non-MoE models only (mirrors the VERIFY branch's own guard). Returns
/// `(logits [tokens.len()*vocab], h [tokens.len()*n_embd])`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_rows_cpu_with_h(
    g: &Gguf,
    cfg: &Config,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<(Vec<f32>, Vec<f32>)> {
    let cpu_be = CpuBackend::new();
    let mut logits = Vec::new();
    let mut h = Vec::new();
    let mut state = None;
    generate_dense_backend(
        &cpu_be,
        &|_name, tb, dt, _n| match tb {
            WBytes::Mmap(tb) => Ok((cpu_be.map_weight(tb), dt)),
            WBytes::Owned(v) => {
                let buf = cpu_be
                    .alloc(v.len().max(1), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                cpu_be
                    .upload(buf.as_ref(), &v)
                    .map_err(|e| anyhow!("{e}"))?;
                Ok((buf, dt))
            }
        },
        g,
        cfg,
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

/// [`verify_dense_cpu`]'s Vulkan twin — the same one-shot causal prefill through the production
/// Vulkan seam, for the CPU/Vulkan cross-backend parity check.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_dense_vulkan(
    vk: &infr_vulkan::VulkanBackend,
    g: &Gguf,
    cfg: &Config,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<Vec<f32>> {
    let mut logits = Vec::new();
    let mut state = None;
    generate_dense_backend(
        vk,
        &|_name, tb, dt, _n| {
            let padded = infr_vulkan::linear::pad_to_u32_align(&tb);
            let buf = vk
                .alloc(padded.len(), BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            vk.upload(buf.as_ref(), &padded)
                .map_err(|e| anyhow!("{e}"))?;
            Ok((buf, dt))
        },
        g,
        cfg,
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
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl std::ops::Deref for WBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            WBytes::Mmap(tb) => tb,
            WBytes::Owned(v) => v,
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

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn kv_fmt_bytes(dt: DType, elems: usize) -> usize {
    match dt {
        DType::Q8_0 => (elems / 32 * 34).next_multiple_of(4),
        // TurboQuant 128-elem blocks: turbo2 = 34 B, turbo3 = 50 B, turbo4 = 66 B.
        DType::Turbo2 => elems / 128 * 34,
        DType::Turbo3 => elems / 128 * 50,
        DType::Turbo4 => elems / 128 * 66,
        // Mainline low-bit KV quants (32-elem blocks) + bf16.
        DType::Q4_0 | DType::Iq4Nl => elems / 32 * 18,
        DType::Q4_1 => elems / 32 * 20,
        DType::Q5_0 => elems / 32 * 22,
        DType::Q5_1 => elems / 32 * 24,
        DType::F16 | DType::Bf16 => elems * 2,
        _ => elems * 4, // F32
    }
}

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
