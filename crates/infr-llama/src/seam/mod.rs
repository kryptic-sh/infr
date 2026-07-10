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
pub(crate) fn generate_dense_cpu(
    g: &Gguf,
    cfg: &Config,
    token_embd: &[f32],
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
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
        None,
        None,
        None,
        None,
        None,
        None,
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
    token_embd: &[f32],
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
        None,
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
    token_embd: &[f32],
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
    state: &mut Option<SeamKv>,
    want_ctx: usize,
    constraint: Option<&mut crate::grammar::Constraint>,
) -> AResult<(Vec<u32>, GenStats)> {
    let bind = vulkan_moe_binder(vk, g, cfg, state.is_none(), want_ctx)?;
    let out = generate_dense_backend(
        vk, &*bind, g, cfg, token_embd, ple, prompt, max_new, on_token, state, want_ctx,
        constraint, None, None, None, None, None,
    )?;
    // INFR_PAGER_STATS=1: cumulative hit/miss/eviction counters since this pager was installed
    // (persists across calls on the same session — see `MoePagerSession`). A no-op when no paged
    // model is loaded. Printed every call rather than gated to "last call only" since neither the
    // CLI's run/serve loop nor this function know which call is the process's last one.
    vk.print_moe_pager_stats();
    Ok(out)
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
                let kv_bytes: u64 = (0..cfg.n_layer)
                    .map(|l| (cfg.layer_n_kv(l) * cfg.layer_head_dim(l) * 2 * 2) as u64)
                    .sum::<u64>()
                    * (want_ctx as u64 + 64);
                // 2 GiB: covers activation scratch (pooled, but per-tag sizes scale with n_embd/
                // n_ff and this budget calc's `fp`/`kv_bytes` are estimates, not the exact bytes
                // `alloc`/gpu-allocator's 256 MiB block granularity ends up committing) plus the
                // pager's own arena+staging+LUT allocations, which aren't counted in `fp` at all
                // (found empirically sizing Scout's 48-layer, 37 GB Q2_K pager placement — 512 MiB
                // undershot by a few hundred MiB and the guard rightly refused to over-commit).
                const ACT_HEADROOM: u64 = 2 * 1024 * 1024 * 1024;
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
            vk.init_moe_pager(infr_vulkan::pager::MoePagerLayout { n_blocks, pools })
                .map_err(|e| anyhow!("{e}"))?;
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
        let buf = vk
            .alloc(padded.len(), BufferUsage::Weights)
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
    token_embd: &[f32],
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
    token_embd: &[f32],
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
    token_embd: &[f32],
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
    token_embd: &[f32],
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
    token_embd: &[f32],
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
    token_embd: &[f32],
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
    token_embd: &[f32],
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
