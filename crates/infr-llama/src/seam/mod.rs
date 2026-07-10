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
    // ── MoE expert placement ─────────────────────────────────────────────────────────────────
    // Three tiers, in precedence order:
    //   1. `INFR_NCMOE=N` EXPLICIT override — the legacy host-visible split: the FIRST N layers'
    //      banks land in HOST-VISIBLE memory (HostWeights = system-RAM GTT), the GPU reading them
    //      over the bus every dispatch. Kept verbatim for models/hardware where that already
    //      works today (some overflow, no readback stalls) — explicit opt-in, unconditional (forces
    //      N layers off VRAM even if everything would otherwise fit).
    //   2. `INFR_MOE_CACHE_GB=X` EXPLICIT override (NCMOE unset) — force EVERY expert layer through
    //      the pager (`infr_vulkan::pager`) with an X GB byte budget, regardless of whether the
    //      banks would fit resident. Lets a caller (or a test) force the paged path deterministically
    //      instead of depending on this box's free VRAM — see the `gpu_seam_paged_moe_matches_*`
    //      tests. Needs a SPLIT (non-fused) gate/up bank (`fused_bank` below); a fused-bank model
    //      falls back to fully resident here (paging can't help it, and forcing host-visible on a
    //      model that already fits would be a pure regression).
    //   3. Auto (both unset): fully resident (today's fast path, zero change) when the banks fit
    //      VRAM; otherwise the pager with budget = remaining VRAM after dense+KV+headroom — same
    //      split-bank-only caveat as tier 2 (a fused bank falls back to the legacy host-visible
    //      split here, unchanged from before this task).
    // Tier 2/3 paging needs the adapter's paged executor split
    // (`infr_vulkan::adapter::execute_static`'s paged branch) — llama4/qwen3moe/qwen35moe-shaped
    // (split gate/up) banks only.
    let explicit_ncmoe = std::env::var("INFR_NCMOE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let moe_cache_gb_override = std::env::var("INFR_MOE_CACHE_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok());
    let fused_bank = g
        .tensors()
        .iter()
        .any(|t| t.name.contains("ffn_gate_up_exps"));
    // The pager's arena addresses slot `k` at a FIXED byte offset `k * slot_bytes` (uniform per
    // role), while the GEMV shader computes the SAME slot's element offset as `k * stride`
    // (`stride` = `n_ff_exp * n_embd`, a pure matrix-shape constant — identical regardless of
    // quant format). Those only agree if EVERY expert ever placed in that role's arena shares the
    // SAME dtype (so "stride elements" converts to the SAME byte count everywhere): a role whose
    // dtype varies by layer — some "unsloth-dynamic" (UD) quants bump `ffn_down_exps` to a wider
    // K-quant on a subset of layers for quality, e.g. Qwen3-30B-A3B-Q4_K_M's down bank mixes
    // Q4_K/Q6_K — would have two different experts claim the SAME slot's byte range, silently
    // corrupting whichever doesn't match the dtype the arena was sized from (found empirically:
    // layers past the first few produced wildly out-of-range GEMV output on a mixed-dtype model;
    // see the task's write-up). Detect this UP FRONT and refuse to page — falling back to the
    // legacy host-visible split, exactly like `fused_bank` — rather than risk it. llama4/Scout
    // ships ONE dtype per role across every layer (verified), so this never trips there.
    let role_dtype_uniform = |suffix: &str| -> bool {
        let mut seen: Option<(infr_core::DType, usize)> = None;
        for l in 0..cfg.n_layer {
            let name = format!("blk.{l}.{suffix}");
            if let Some(t) = g.tensors().iter().find(|t| t.name == name) {
                let key = (t.dtype, t.nbytes);
                match seen {
                    None => seen = Some(key),
                    Some(prev) if prev != key => return false,
                    _ => {}
                }
            }
        }
        true
    };
    let cannot_page = fused_bank
        || !role_dtype_uniform("ffn_gate_exps.weight")
        || !role_dtype_uniform("ffn_up_exps.weight")
        || !role_dtype_uniform("ffn_down_exps.weight");

    let mut n_host_moe = 0usize; // tier 1: first N layers host-visible
    let mut n_paged = 0usize; // tier 2/3: first N layers paged
    let mut pager_budget_bytes = 0u64;
    // Placement is decided ONCE, on the session's FIRST load — the only call where `bind_weight`
    // runs (see the `state.is_none()` init block in `generate_dense_backend`) and the only moment
    // the tier-3 budget math is consistent: `vram.available` is LIVE (heapBudget − heapUsage), so
    // once this model's weights are resident a recompute would subtract `fp.dense` from an
    // `available` that ALREADY excludes it — double-counting the model against itself and
    // collapsing the budget (observed: a fully-resident 16.4 GB model "re-placed" as 5/30
    // resident on the warm second call of a bench). Warm calls leave n_host_moe/n_paged at 0;
    // nothing consumes them (no binding, and the pager init below is first_load-gated anyway).
    // A first load racing ANOTHER resident model (swap mid-drain) still reads reduced
    // `available` — that's real pressure, deliberately not compensated; the alloc-time VRAM
    // budget guard is the backstop against over-commit.
    let first_load = state.is_none();
    if first_load && cfg.moe.is_some() {
        match (explicit_ncmoe, moe_cache_gb_override) {
            (Some(n), _) => n_host_moe = n.min(cfg.n_layer),
            (None, Some(gb)) if !cannot_page => {
                n_paged = cfg.n_layer;
                pager_budget_bytes = (gb * 1e9) as u64;
            }
            (None, _) => {
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
                let per_layer = (fp.expert / cfg.n_layer.max(1) as u64).max(1);
                let gpu_layers = (budget / per_layer).min(cfg.n_layer as u64) as usize;
                let overflow = cfg.n_layer - gpu_layers;
                if overflow > 0 {
                    if cannot_page {
                        n_host_moe = overflow; // tier 2/3 can't handle a fused/mixed-dtype bank
                    } else {
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
    }
    if n_host_moe > 0 {
        eprintln!(
            "MoE auto-fit: {}/{} expert layers on GPU, {n_host_moe} host-visible (GPU reads over \
             the bus; ctx={want_ctx})",
            cfg.n_layer - n_host_moe,
            cfg.n_layer,
        );
    }
    // A layer-l stacked expert bank ("blk.l.ffn_{gate,up,down}_exps.weight") of an offloaded layer.
    let host_bank = move |name: &str| -> bool {
        if n_host_moe == 0 || !name.contains("_exps") {
            return false;
        }
        name.strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|l| l.parse::<usize>().ok())
            .is_some_and(|l| l < n_host_moe)
    };
    // A layer-l `_exps` tensor of a PAGED layer — split gate/up/down only (see `fused_bank` above).
    let paged_layer = move |name: &str| -> Option<usize> {
        if n_paged == 0 || !name.contains("_exps") {
            return None;
        }
        name.strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|l| l.parse::<usize>().ok())
            .filter(|&l| l < n_paged)
    };
    let moe_role_of = |name: &str| -> Option<infr_vulkan::pager::Role> {
        use infr_vulkan::pager::Role;
        if name.ends_with("ffn_gate_exps.weight") {
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
        let moe = cfg.moe.as_ref().expect("n_paged > 0 implies MoE");
        let tensor_bytes = |suffix: &str| -> Option<usize> {
            (0..n_paged).find_map(|l| {
                let name = format!("blk.{l}.{suffix}");
                g.tensors()
                    .iter()
                    .find(|t| t.name == name)
                    .map(|t| t.nbytes)
            })
        };
        let (gb, ub, db) = (
            tensor_bytes("ffn_gate_exps.weight").unwrap_or(4) / moe.n_expert.max(1),
            tensor_bytes("ffn_up_exps.weight").unwrap_or(4) / moe.n_expert.max(1),
            tensor_bytes("ffn_down_exps.weight").unwrap_or(4) / moe.n_expert.max(1),
        );
        let n_blocks = n_paged * moe.n_expert;
        let per_slot = (gb + ub + db).max(1) as u64;
        // Floor at `n_expert`: a chunked batched-prefill `Op::MoeFfn` (rows>1) runs ALL of a
        // layer's routed buckets in ONE dispatch (`matmul_mmq_experts_paged` — Scout's Q2_K/Q3_K
        // banks ARE batched-mmq-eligible now), touching up to `n_expert` DISTINCT experts that
        // must be simultaneously resident (the within-batch safety invariant — see
        // `infr_core::pager::Pager::new`'s doc). Decode's rows=1 needs only `n_used`, but the
        // batched bound subsumes it and `n_expert` slots is tiny next to any real budget
        // (Scout: 16 x ~18 MB per role).
        let budget_slots = ((pager_budget_bytes / per_slot) as usize)
            .clamp(moe.n_expert.max(moe.n_used).max(1), n_blocks);
        // Per-role ceiling: each role's arena is ONE SSBO binding, capped by the smaller of the
        // paged kernels' u32 word reach (16 GiB) and the device's maxStorageBufferRange (4 GiB
        // on RADV — found empirically: Scout's auto budget wanted a 7.6 GiB down arena, and
        // reads past the binding range came back as garbage → NaN logits → sentinel router ids).
        // `GpuPager::new` hard-errors past this; clamping here keeps an oversized budget usable
        // instead of fatal. Applied PER ROLE (their expert sizes differ — Scout: gate/up 13.8 MB
        // vs down 18 MB — so a shared count dragged to the largest role's cap strands budget the
        // smaller roles could hold as real hit rate; see `MoePagerLayout`'s doc). Every role's
        // count starts from the same budget-driven figure, so the total can only come in UNDER
        // budget when a cap bites. Splitting a role across several arena buffers (or u64 shader
        // addressing where the device allows bigger buffers) is the lift that raises the cap.
        let arena_cap = infr_vulkan::pager::GpuPager::max_arena_bytes(vk);
        let role_slots = |slot_bytes: usize| -> usize {
            let cap = (arena_cap / slot_bytes.max(4) as u64) as usize;
            if budget_slots > cap {
                eprintln!(
                    "MoE pager: clamping a role's {budget_slots} -> {cap} slots (per-role arena \
                     capped by the device's storage-buffer range / u32 word addressing)"
                );
            }
            budget_slots.min(cap).min(n_blocks)
        };
        let (gate_n_slots, up_n_slots, down_n_slots) =
            (role_slots(gb), role_slots(ub), role_slots(db));
        eprintln!(
            "MoE pager: {n_paged}/{} expert layers PAGED (gate/up/down {gate_n_slots}/\
             {up_n_slots}/{down_n_slots} of {n_blocks} experts cached, {:.2} GB budget; \
             ctx={want_ctx})",
            cfg.n_layer,
            pager_budget_bytes as f64 / 1e9,
        );
        vk.init_moe_pager(infr_vulkan::pager::MoePagerLayout {
            n_blocks,
            gate_n_slots,
            up_n_slots,
            down_n_slots,
            gate_slot_bytes: gb,
            up_slot_bytes: ub,
            down_slot_bytes: db,
        })
        .map_err(|e| anyhow!("{e}"))?;
    }

    let out = generate_dense_backend(
        vk,
        &|name, tb, dt, _n| {
            // Raw upload for EVERY dtype — the file's bytes go straight to VRAM (u32-padded) and the
            // kernel reads/dequants the native dtype in-shader. F16 → f16 coopmat GEMM / f16 GEMV;
            // F32 stays native (rmsnorm/qk_norm_rope read f32); bf16 → in-shader expand (bf16 is the
            // top 16 bits of an f32, EXACT; the warp GEMM narrows to f16 for the matrix cores like
            // every other format); quant weights → raw blocks. No host dtype conversion on any path.
            //
            // Tier 2/3 (paged): register this layer's mmap bytes with the pager and bind a tiny
            // placeholder instead of uploading the full bank — the Vulkan adapter recognizes the
            // placeholder's identity (see `infr_vulkan::pager`'s module doc) and diverts to the
            // paged executor split at execute time. `down_scale`/router/every other tensor of a
            // paged layer is unaffected — only the three `_exps` banks divert here.
            if let Some(l) = paged_layer(name) {
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
                    vk.register_paged_expert(role, buf_id, source);
                    return Ok((placeholder, dt));
                }
            }
            let padded = infr_vulkan::linear::pad_to_u32_align(&tb);
            // Auto-fit-offloaded expert banks land in HOST memory (HostWeights = system-RAM GTT the
            // GPU reads over the bus; it binds as an SSBO like any weight). alloc_uninit: the upload
            // covers the whole extent.
            let usage = if host_bank(name) {
                BufferUsage::HostWeights
            } else {
                BufferUsage::Weights
            };
            let buf = if matches!(usage, BufferUsage::HostWeights) {
                vk.alloc_uninit(padded.len(), usage)
            } else {
                vk.alloc(padded.len(), usage)
            }
            .map_err(|e| anyhow!("{e}"))?;
            vk.upload(buf.as_ref(), &padded)
                .map_err(|e| anyhow!("{e}"))?;
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
    )?;
    // INFR_PAGER_STATS=1: cumulative hit/miss/eviction counters since this pager was installed
    // (persists across calls on the same session — see `MoePagerSession`). A no-op when no paged
    // model is loaded. Printed every call rather than gated to "last call only" since neither the
    // CLI's run/serve loop nor this function know which call is the process's last one.
    vk.print_moe_pager_stats();
    Ok(out)
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
