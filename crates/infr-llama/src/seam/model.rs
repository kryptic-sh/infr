//! The GPU-free `SeamModel` for the CPU reference backend (no Vulkan/VRAM; weights streamed
//! from the GGUF mmap at forward time). Split out of `lib.rs` (no logic change).
use crate::*;
use anyhow::{anyhow, Result};
use infr_chat::{render_chat_jinja, render_chat_user};
use infr_core::backend::{Backend, BufferUsage};
use infr_cpu::CpuBackend;
use infr_gguf::Gguf;
use std::path::Path;
use tokenizers::Tokenizer;

/// A **GPU-free** model for the CPU reference backend. Holds only what the agnostic CPU compute
/// graph needs — the parsed [`Config`], the host f32 token embeddings (for the gather + tied lm
/// head), the tokenizer, and the gemma4 E2B per-layer-embd tensors. No `VulkanBackend`, no VRAM,
/// no weight upload: the projection weights are streamed straight from the kept-open GGUF mmap at
/// forward time. Dense Qwen3/Llama, Gemma 3, Gemma 4 (dense + E2B), qwen3moe, and qwen35
/// (`MixerW::DeltaNet`) all drive this same struct.
pub struct SeamModel {
    gguf: Gguf,
    cfg: Config,
    /// Host f32 token-embedding table — materialized LAZILY (see [`SeamModel::token_embd`]).
    /// Dequantizing it eagerly at load cost ~4s and ~3.1 GiB of RSS on Qwen3-14B (a 151936×5120
    /// Q4_K table blown up to f32) for every load, while the GPU/Metal dense path never reads it:
    /// those upload `token_embd.weight` to the device in its NATIVE dtype and gather on-device.
    /// Only the host-gather consumers (MTP heads, the DiffusionGemma SC soft-embed, the CPU
    /// runner) touch it, so they now pay for it — and only on first use.
    token_embd: std::sync::OnceLock<Vec<f32>>,
    per_layer_embd: Option<PerLayerEmbd>,
    tokenizer: Tokenizer,
}

/// The conversation SLOTS a persistent GPU seam session owns: up to `INFR_KV_SLOTS` (default 4)
/// [`crate::seam::SeamKv`]s — each a KV cache + the token ids materialized in it, all
/// sharing one weight upload (`Arc<SeamWeights>`). Per request the best-prefix slot is picked: a
/// prompt that EXTENDS a slot's cache continues it (the classic next-turn suffix prefill); a
/// prompt that diverges early (a different conversation) forks a fresh slot and SEEDS it with the
/// longest shared prefix (e.g. a common system prompt) via a device-side KV copy instead of
/// re-prefilling it; when all slots are taken the LRU one is recycled. Single-conversation
/// callers (run/bench/spec drivers) stay on one slot. Backend-agnostic: fork and seed go through
/// `&dyn Backend` (`copy_buffer` is the seeding primitive), so the Vulkan and Metal sessions
/// share this policy verbatim.
struct SlotPool {
    slots: Vec<Option<crate::seam::SeamKv>>,
    last_used: Vec<u64>,
    tick: u64,
}

/// A persistent Vulkan seam session (see [`SeamModel::vulkan_session`]): owns the backend and the
/// conversation [`SlotPool`].
pub struct DenseVulkanSession {
    vk: infr_vulkan::VulkanBackend,
    pool: SlotPool,
    max_ctx: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DenseVulkanSession {
    /// Forget every slot's materialized tokens (buffers and the weight upload stay) — discards a
    /// warmup generation so the first real prompt starts from clean slots.
    pub fn reset_cache(&mut self) {
        self.pool.reset_cache();
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl SlotPool {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            last_used: Vec::new(),
            tick: 0,
        }
    }

    /// Forget every slot's materialized tokens (buffers and the weight upload stay) — discards a
    /// warmup generation so the first real prompt starts from clean slots.
    fn reset_cache(&mut self) {
        for s in self.slots.iter_mut().flatten() {
            s.reset();
        }
    }

    /// The single-conversation slot (the bench/spec drivers, which manage one token stream and
    /// never contend): slot 0, created empty on first use.
    #[cfg(target_os = "macos")]
    fn single(&mut self) -> &mut Option<crate::seam::SeamKv> {
        if self.slots.is_empty() {
            self.slots.push(None);
            self.last_used.push(self.tick);
        }
        &mut self.slots[0]
    }

    /// Pick (and prepare) the slot for `prompt`; returns its index. See the struct doc for the
    /// policy. A freshly created slot is `None` — the runner's first call uploads the weights.
    ///
    /// This best-prefix choice is prefix-OPTIMAL for a real per-position KV cache (dense/
    /// attention arches): the picked slot always has the longest reusable prefix. For qwen35
    /// (gated-DeltaNet: an append-only recurrent summary, not a per-position cache — see the
    /// no-rewind rule in `seam::generate_dense_backend`) a `prefix_score` match is only
    /// scored on shared TOKENS, not on whether the state can actually rewind to it — so the pick
    /// here is merely CORRECT, not necessarily optimal: the runner independently re-checks EXACT
    /// extension (`prompt` extends `cached` verbatim) before reusing the slot's state, and
    /// zero-resets it otherwise. A suboptimal pick for qwen35 just costs an extra full
    /// re-prefill; it can never reuse a wrong recurrent state.
    fn pick(
        &mut self,
        be: &dyn infr_core::backend::Backend,
        cfg: &crate::Config,
        prompt: &[u32],
    ) -> Result<usize> {
        // Seeding shorter prefixes than this isn't worth the copy submit.
        const MIN_SEED: usize = 16;
        let max_slots: usize = std::env::var("INFR_KV_SLOTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| v > 0)
            .unwrap_or(4);
        self.tick += 1;
        if self.slots.is_empty() {
            self.slots.push(None);
            self.last_used.push(self.tick);
            return Ok(0);
        }
        let score =
            |st: &Option<crate::seam::SeamKv>| st.as_ref().map_or(0, |s| s.prefix_score(prompt));
        // A slot whose cache the prompt EXTENDS (or equals) is this conversation continuing.
        if let Some(i) = (0..self.slots.len()).find(|&i| {
            self.slots[i].as_ref().is_some_and(|s| {
                let p = s.prefix_score(prompt);
                p > 0 && (p == s.cached_len() || p == prompt.len())
            })
        }) {
            self.last_used[i] = self.tick;
            return Ok(i);
        }
        // Different conversation: the best shared prefix (if any) seeds the slot we hand out.
        let (best_i, best_s) = (0..self.slots.len())
            .map(|i| (i, score(&self.slots[i])))
            .max_by_key(|&(_, s)| s)
            .unwrap();
        // An EMPTY slot (freshly created, or reset — e.g. by the warmup discard) holds no cached
        // prefix worth keeping, so reusing it can never lose another conversation's KV. Prefer it
        // (LRU first) over forking: forking here would permanently strand a whole full-ctx KV
        // allocation per process (every `infr run` / first serve request pays one otherwise).
        let empty = (0..self.slots.len())
            .filter(|&i| self.slots[i].as_ref().is_none_or(|s| s.cached_len() == 0))
            .min_by_key(|&i| self.last_used[i]);
        let target = if let Some(i) = empty {
            i
        } else if self.slots.len() < max_slots {
            // Fork a fresh slot off any initialized one (shared weights, own KV). A fork that
            // fails the VRAM budget (each slot is a whole max_ctx KV cache — on a big model one
            // slot can own most of free VRAM) degrades to recycling the LRU slot instead of
            // failing the request: correctness is identical (the slot re-prefills from scratch),
            // only cross-conversation KV reuse is lost.
            let src = self
                .slots
                .iter()
                .flatten()
                .next()
                .expect("pick_slot: no initialized slot to fork from");
            match src.fork(be, cfg) {
                Ok(fresh) => {
                    self.slots.push(Some(fresh));
                    self.last_used.push(self.tick);
                    self.slots.len() - 1
                }
                Err(e) => {
                    eprintln!(
                        "kv slots: fork of a {}th slot failed ({e}); recycling the LRU slot \
                         instead (fewer concurrent conversations keep their KV)",
                        self.slots.len() + 1,
                    );
                    (0..self.slots.len())
                        .min_by_key(|&i| self.last_used[i])
                        .unwrap()
                }
            }
        } else {
            // Recycle the least-recently-used slot.
            (0..self.slots.len())
                .min_by_key(|&i| self.last_used[i])
                .unwrap()
        };
        if best_s >= MIN_SEED && best_i != target {
            // Give the slot the shared prefix (system prompt etc.) via device-side KV copy —
            // only when it beats whatever prefix the slot already shares with the prompt.
            if best_s > score(&self.slots[target]) {
                let src = self.slots[best_i].take().expect("scored slot is Some");
                if let Some(dst) = self.slots[target].as_mut() {
                    dst.seed_from(be, cfg, &src, best_s)?;
                }
                self.slots[best_i] = Some(src);
            }
        }
        self.last_used[target] = self.tick;
        Ok(target)
    }
}

/// A persistent Metal seam session — the Apple-GPU twin of [`DenseVulkanSession`]: owns the
/// backend and the conversation [`SlotPool`], so every later
/// [`SeamModel::generate_metal_session`] call prefills only the suffix that differs from its
/// slot's previous turn, and concurrent conversations (serve) each keep their own KV slot off
/// the one shared weight upload. Slot switches re-record the decode replay tape (its fingerprint
/// covers the bound KV/IO buffer addresses) — one graph-walk token per switch, never a stale
/// replay.
#[cfg(target_os = "macos")]
pub struct DenseMetalSession {
    mtl: infr_metal::MetalBackend,
    pool: SlotPool,
    max_ctx: usize,
}

#[cfg(target_os = "macos")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DenseMetalSession {
    /// Forget every slot's materialized tokens (buffers and the weight upload stay) — discards a
    /// warmup generation so the first real prompt starts from clean slots.
    pub fn reset_cache(&mut self) {
        self.pool.reset_cache();
    }
}

/// Estimated KV-cache bytes per element for one side (K or V), from the same env override the
/// runner honors (`INFR_KV_TYPE_K/V`, legacy `INFR_KV_Q8`). ESTIMATE ONLY — the runner
/// additionally gates each format on backend/alignment and falls back to f16, so a gated-out
/// low-bit request can under-estimate here; the alloc-time VRAM budget guard backstops that.
/// Unknown/unset → `auto_q8` picks between f16 (2 bytes, the GPU default) and Q8_0 — pass the
/// current [`crate::seam::kv_auto_q8`] pin (or `true` to PRICE a candidate auto-q8 placement
/// before pinning it, as `clamp_default_ctx` does).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn kv_bytes_per_elem(var: &str, auto_q8: bool) -> f64 {
    let side = std::env::var(var).ok();
    match side.as_deref() {
        Some("f32") => 4.0,
        Some("bf16") | Some("f16") | Some("F16") => 2.0,
        Some("q8_0") | Some("q8") | Some("Q8_0") => 34.0 / 32.0,
        Some("q4_0") | Some("iq4_nl") => 18.0 / 32.0,
        Some("q4_1") => 20.0 / 32.0,
        Some("q5_0") => 22.0 / 32.0,
        Some("q5_1") => 24.0 / 32.0,
        Some("turbo2") => 34.0 / 128.0,
        Some("turbo3") => 50.0 / 128.0,
        Some("turbo4") => 66.0 / 128.0,
        _ if std::env::var("INFR_KV_Q8").is_ok() => 34.0 / 32.0,
        _ if auto_q8 => 34.0 / 32.0,
        _ => 2.0,
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl SeamModel {
    /// Load a model for CPU inference without touching the GPU. `tokenizer_path` overrides the
    /// GGUF's embedded vocab when given.
    pub fn load(gguf_path: &Path, tokenizer_path: Option<&Path>) -> Result<Self> {
        let g = Gguf::open(gguf_path).map_err(|e| anyhow!("open gguf: {e}"))?;
        let mut cfg = Config::from_gguf(&g)?;
        let tokenizer = match tokenizer_path {
            Some(p) => Tokenizer::from_file(p).map_err(|e| anyhow!("load tokenizer: {e}"))?,
            None => build_tokenizer(&g)?,
        };
        add_chat_eos(&mut cfg, &tokenizer);
        // `token_embd.weight` is NOT dequantized here — see the field's doc. `Config::from_gguf`
        // above already read its shape, so a model missing the tensor still fails at load, not on
        // the lazy path below.
        let per_layer_embd = build_per_layer_embd(&g, &cfg)?;
        Ok(Self {
            gguf: g,
            cfg,
            token_embd: std::sync::OnceLock::new(),
            per_layer_embd,
            tokenizer,
        })
    }

    /// Open a persistent Vulkan seam session: weights uploaded ONCE, the KV cache sized to
    /// `max_ctx`, and the materialized-token cache that makes every later
    /// [`generate_vulkan_session`](Self::generate_vulkan_session) call prefill only the suffix
    /// that differs from the previous turn (ChatSession-style KV reuse on the agnostic seam).
    pub fn vulkan_session(&self, max_ctx: usize) -> Result<DenseVulkanSession> {
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        Ok(DenseVulkanSession {
            vk,
            pool: SlotPool::new(),
            max_ctx,
        })
    }

    /// [`vulkan_session`](Self::vulkan_session) with the DEFAULT context window: the model's
    /// trained context (`n_ctx_train`) clamped so weights + KV cache + activation headroom fit
    /// the VRAM budget — a long-context model's trained window (128k+) would otherwise allocate
    /// a KV cache that blows VRAM at startup or on the first request. Explicit contexts
    /// (INFR_CTX → `vulkan_session(ctx)`) are NEVER clamped; the Vulkan allocation budget
    /// guard still fails a truly-oversized one cleanly at alloc time.
    pub fn vulkan_session_default(&self) -> Result<DenseVulkanSession> {
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        let max_ctx = self.clamp_default_ctx(&vk, self.cfg.n_ctx_train);
        Ok(DenseVulkanSession {
            vk,
            pool: SlotPool::new(),
            max_ctx,
        })
    }

    /// [`vulkan_session`](Self::vulkan_session) sized as a FRACTION of the device's free-VRAM KV
    /// capacity (`INFR_CTX=50%` → half the tokens that would fit after weights + headroom —
    /// device-appropriate %-base: this KV cache lives in VRAM). Uses the same fit math as
    /// [`vulkan_session_default`]'s clamp, scaled by `frac`; unlike an explicit token count the
    /// result is inherently within budget, and the alloc-time guard stays the backstop.
    pub fn vulkan_session_frac(&self, frac: f64) -> Result<DenseVulkanSession> {
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        // A pure recurrent-state arch has no per-token KV to size a fraction of — fall back to
        // the trained context (same shape as the default path's `kv_per_tok == 0` escape).
        let fit = self.kv_fit_ctx(&vk).unwrap_or(self.cfg.n_ctx_train);
        let max_ctx = ((fit as f64 * frac) as usize).max(1024);
        Ok(DenseVulkanSession {
            vk,
            pool: SlotPool::new(),
            max_ctx,
        })
    }

    /// Clamp a DEFAULT context length so the full weight footprint + one KV cache fit the VRAM
    /// budget (live free bytes when VK_EXT_memory_budget is present — the backend is created
    /// before the weights upload, so `available` still includes their space). KV bytes/token
    /// follows the per-layer KV geometry and the runner's KV-dtype env overrides
    /// (INFR_KV_TYPE_K/V, INFR_KV_Q8). Only ever shrinks, and logs when it does. Extra KV slots
    /// (INFR_KV_SLOTS forks) and MoE expert host-offload aren't modeled — the alloc-time budget
    /// guard remains the backstop for those.
    fn clamp_default_ctx(&self, vk: &infr_vulkan::VulkanBackend, want: usize) -> usize {
        let Some(fit) = self.kv_fit_ctx(vk) else {
            return want; // pure recurrent-state arch: no per-token KV to size.
        };
        if fit < want {
            // Auto-q8 KV rung, clamp flavor (see `crate::seam::PINNED_KV_Q8` for the policy):
            // before shrinking the DEFAULT context below the trained window, try a Q8_0 KV
            // cache — roughly half the bytes per token. Only a FULL rescue pins q8 (fit at q8
            // reaches `want`); a partial rescue keeps the predictable f16 cache and clamps as
            // before — auto-q8 exists to avoid losing capability (ctx / residency), not to
            // trade decode speed for a somewhat-larger-but-still-clamped default window.
            // Dense non-MoE models only, matching the placement rung this was validated on
            // (MoE placement budgets pager arenas separately from this fit math).
            if self.cfg.moe.is_none()
                && !crate::seam::kv_auto_q8()
                && crate::seam::kv_env_unset()
                && crate::seam::kv_q8_layout_ok(&self.cfg)
                && self.kv_fit_ctx_fmt(vk, true).is_some_and(|f| f >= want)
            {
                crate::seam::pin_kv_auto_q8();
                eprintln!(
                    "kv auto-quant: q8_0 (f16 KV would clamp the default ctx {want} -> {fit}; \
                     INFR_KV_TYPE_K/V=f16 to force f16)"
                );
                return want;
            }
            let vram = vk.vram();
            let fp = crate::weights::weight_footprint(&self.gguf);
            eprintln!(
                "ctx clamp: default context {want} -> {fit} to fit VRAM (weights {:.2} GiB vs \
                 {:.2} GiB available{}); set INFR_CTX to override",
                (fp.dense + fp.expert) as f64 / (1u64 << 30) as f64,
                vram.available as f64 / (1u64 << 30) as f64,
                if vram.live { ", live" } else { ", total heap" },
            );
            return fit;
        }
        want
    }

    /// The VRAM-fit KV capacity in tokens: how much context fits in the device's AVAILABLE
    /// memory after the full weight footprint + activation headroom (live free bytes when
    /// VK_EXT_memory_budget is present — call before the weights upload so `available` still
    /// includes their space). KV bytes/token follows the per-layer KV geometry and the runner's
    /// KV-dtype env overrides (INFR_KV_TYPE_K/V, INFR_KV_Q8). `None` for a pure recurrent-state
    /// arch (no per-token KV). Extra KV slots (INFR_KV_SLOTS forks) aren't modeled — the
    /// alloc-time budget guard remains the backstop.
    fn kv_fit_ctx(&self, vk: &infr_vulkan::VulkanBackend) -> Option<usize> {
        // Price whatever the runner will actually allocate: the auto-q8 pin (if the placement
        // ladder set it earlier in this process) or the plain env-driven formats.
        self.kv_fit_ctx_fmt(vk, crate::seam::kv_auto_q8())
    }

    /// [`kv_fit_ctx`](Self::kv_fit_ctx) at an EXPLICIT assumed-when-env-unset KV format:
    /// `auto_q8 = true` prices both sides Q8_0 wherever the user set nothing — how the ctx
    /// clamp's auto-q8 rung asks "would the trained window fit if we quantized the cache?"
    /// BEFORE pinning that choice process-wide.
    fn kv_fit_ctx_fmt(&self, vk: &infr_vulkan::VulkanBackend, auto_q8: bool) -> Option<usize> {
        /// Take only this fraction of the KV bytes that nominally fit — absorbs allocation slop
        /// (alignment, dedicated-buffer rounding) and estimate error, same spirit as the alloc
        /// guard's fixed headroom.
        const FIT_FRACTION: f64 = 0.95;
        /// Below this a session is useless anyway — let the alloc guard produce its clear error.
        const MIN_CTX: usize = 1024;
        // Reserve beyond weights+KV: activations/scratch PLUS the measured non-modeled residents.
        // Empirics (gemma-3-12b Q4_K_M, 7900 XTX): live usage ran ~1.5 GiB past weights+KV —
        // upload-staging pools land in the device-local host-visible heap under ReBAR and
        // gpu-allocator retains freed blocks, dedicated buffers round up, and the warmup graph's
        // activations stay resident. A flush clamp just moves the failure to the first real
        // request's activation alloc (observed as a 500), so reserve generously: max(1 GiB,
        // total/12) — ~2 GiB on a 24 GiB card, 1 GiB floor on small ones. Over-clamping is safe;
        // under-clamping errors requests.
        // 2026-07 re-audit (after dedicated weight-upload staging landed in infr-vulkan): /sys
        // VRAM watermarks on the 14B (Q4_K_M) and gemma-4-31B (UD-Q5_K_XL) loads were unchanged
        // to within ~0.2 MiB — the residual this reserve absorbs is warmup activation pools,
        // gpu-allocator block granularity, and driver internals, NOT reclaimable staging, so
        // the total/12 headroom stays.
        let vram = vk.vram();
        let mut act_headroom: u64 = (vram.total / 12).max(1024 * 1024 * 1024);
        // Keep the clamp CONSISTENT with the dense placement decision (`vulkan_moe_binder`'s
        // try-resident tier): reserve at least what placement will demand as its activation
        // estimate at this ctx, so a DEFAULT ctx this fit math hands out always lands RESIDENT
        // instead of clamping to a window the placement then streams anyway. MoE models keep the
        // plain heuristic (their placement reserves pager arenas separately from this).
        if self.cfg.moe.is_none() {
            act_headroom = act_headroom.max(crate::seam::dense_act_reserve(
                &self.cfg,
                self.cfg.n_ctx_train,
            ));
        }
        // Bytes per token across all layers, K side + V side (bytes-per-element from the same
        // env the runner honors; formats it would gate back to f16 are an estimate only — the
        // alloc guard catches a resulting overflow).
        let (kb, vb) = (
            kv_bytes_per_elem("INFR_KV_TYPE_K", auto_q8),
            kv_bytes_per_elem("INFR_KV_TYPE_V", auto_q8),
        );
        let kv_per_tok: u64 = (0..self.cfg.n_layer)
            .map(|l| {
                let elems = (self.cfg.layer_n_kv(l) * self.cfg.layer_head_dim(l)) as f64;
                (elems * (kb + vb)).ceil() as u64
            })
            .sum();
        if kv_per_tok == 0 {
            return None;
        }
        let fp = crate::weights::weight_footprint(&self.gguf);
        let free = vram
            .available
            .saturating_sub(fp.dense + fp.expert + act_headroom);
        // SeamKv pads its buffers past max_ctx by ~64 rows; mirror that.
        let fit_linear = ((free as f64 * FIT_FRACTION / kv_per_tok as f64) as usize)
            .saturating_sub(64)
            .max(MIN_CTX);
        // SWA ring sizing (see `crate::seam::kv_rows`): past `ring_rows` a window layer's KV
        // stops growing with ctx, so bytes(ctx) = full_per_tok*ctx + swa_fixed and the fit is
        // (free - swa_fixed) / full_per_tok — a mostly-SWA model's default ctx clamp relaxes
        // enormously. The linear fit stays authoritative while it lands BELOW ring_rows (no
        // layer would actually ring there).
        if crate::seam::kv_ring_wanted(&self.cfg) && fit_linear >= 1024 {
            let ring_rows = (self.cfg.swa_window + crate::seam::ubatch_rows()).next_multiple_of(64);
            if fit_linear >= ring_rows {
                let (mut full_per_tok, mut swa_fixed) = (0f64, 0f64);
                for l in 0..self.cfg.n_layer {
                    let bytes =
                        (self.cfg.layer_n_kv(l) * self.cfg.layer_head_dim(l)) as f64 * (kb + vb);
                    if self.cfg.is_swa_layer(l) {
                        swa_fixed += bytes * ring_rows as f64;
                    } else {
                        full_per_tok += bytes;
                    }
                }
                if full_per_tok > 0.0 {
                    let fit = (((free as f64 * FIT_FRACTION - swa_fixed) / full_per_tok) as usize)
                        .saturating_sub(64)
                        .max(MIN_CTX);
                    return Some(fit.max(fit_linear));
                }
            }
        }
        Some(fit_linear)
    }

    /// Greedy generation on the Vulkan seam through a persistent session (see
    /// [`vulkan_session`](Self::vulkan_session)). `stats.n_prompt` reports the tokens actually
    /// PREFILLED (the un-cached suffix) — the TTFT-honest count.
    pub fn generate_vulkan_session(
        &self,
        session: &mut DenseVulkanSession,
        prompt: &str,
        max_new: usize,
        on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        self.generate_vulkan_session_constrained(session, prompt, max_new, None, on_piece)
    }

    /// [`generate_vulkan_session`](Self::generate_vulkan_session) with an optional llguidance
    /// grammar constraint (serve's forced tool_choice) applied to the decode.
    pub fn generate_vulkan_session_constrained(
        &self,
        session: &mut DenseVulkanSession,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let slot = session.pool.pick(&session.vk, &self.cfg, &prompt_tokens)?;
        let max_ctx = session.max_ctx;
        // Cap the reply to the context that's actually left ("a turn also caps to remaining
        // context" — the CLI's generation ceiling is a default, not a demand): a VRAM-clamped
        // default session (e.g. a 21.9 GB model on a 24 GB card clamps 262k -> ~1.7k) would
        // otherwise hard-error on `infr run`'s default max_new=2048 before generating a single
        // token. EOS ends almost every reply long before this cap; an over-long PROMPT still
        // errors cleanly in the runner (its `prompt + gen + 1 > max_ctx` guard stays).
        let max_new = max_new.min(max_ctx.saturating_sub(prompt_tokens.len() + 1));
        let (_generated, stats) = crate::seam::generate_dense_vulkan_session(
            &session.vk,
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
            &mut session.pool.slots[slot],
            max_ctx,
            constraint,
        )?;
        Ok(stats)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The open GGUF handle — MTP Phase 2 (issue #33) needs it to resolve/upload the head's own
    /// tensors (`crate::mtp::load_mtp_head` / `MtpHeadSession::new_cpu`/`new_vulkan`).
    pub fn gguf(&self) -> &Gguf {
        &self.gguf
    }

    /// The host f32 token-embedding table (`token_embd.weight`), dequantized ONCE on first call
    /// and cached — MTP Phase 2 (issue #33) gathers embedding rows from this on the host,
    /// mirroring every other embed-gather call site on this seam (see `crate::mtp::MtpHeadSession`).
    ///
    /// Lazy on purpose: the Vulkan/Metal dense path uploads `token_embd.weight` to the device in
    /// its native dtype and never calls this, so it must not pay the dequant (~4s / ~3.1 GiB on
    /// Qwen3-14B). `Config::from_gguf` validated the tensor exists at load, so the dequant here
    /// can only fail on a corrupt/truncated file.
    pub fn token_embd(&self) -> &[f32] {
        self.embd().get()
    }

    /// The LAZY handle to the host token-embedding table, as threaded into the seam runners.
    /// Prefer this over [`token_embd`](Self::token_embd) at any call site that only PASSES the
    /// table on: the GPU/Metal dense runners never read it, so handing them the handle (rather
    /// than a materialized slice) keeps the ~4s / ~3.1 GiB dequant off the GPU load path.
    pub(crate) fn embd(&self) -> crate::seam::TokenEmbd<'_> {
        crate::seam::TokenEmbd::new(&self.token_embd, &self.gguf)
    }

    /// Tokenize raw text with the model's own tokenizer (no chat template) — for callers that
    /// need token ids directly (e.g. a raw-forward validation harness), as opposed to
    /// [`render_chat`](Self::render_chat) + the generation loop.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Raw tokenizer accessor for callers outside this module that need incremental detok (the
    /// diffusion decode loop, Phase 3) via the shared [`crate::stream_token`] helper.
    pub(crate) fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Detokenize ids back to text (`encode`'s twin, `skip_special_tokens=true` — matches
    /// [`crate::stream_token`]'s convention so a thinking model's `<|channel>thought`/`<channel|>`
    /// markers, which aren't in the tokenizer's added-specials set, still come through as text) —
    /// for callers that drive a decode loop directly on token ids (the diffusion decode loop's own
    /// tests) instead of through one of the `generate_*` string-in/string-out helpers.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// DiffusionGemma Phase-1 validation: a causal prefill of `tokens` on the CPU reference
    /// backend, returning the LAST token's raw logits (`[vocab]`, pre-softmax, post-softcap). Not
    /// specific to diffusion-gemma — works for any arch on this seam — but this is its only
    /// caller today (see `docs/DIFFUSIONGEMMA.md`).
    pub fn prefill_logits_cpu(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        crate::seam::verify_dense_cpu(
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            tokens,
        )
    }

    /// [`prefill_logits_cpu`](Self::prefill_logits_cpu)'s MTP Phase 1 twin (issue #33,
    /// docs/MTP.md): ALSO returns the LM-head INPUT row (post-`output_norm`, pre-lm_head) for the
    /// same last-prompt-token row the logits came from — the `h_p` primitive Phase 2's MTP driver
    /// needs, validated here via `lm_head(h) == logits`. Returns `(logits, h)`.
    pub fn prefill_logits_and_h_cpu(&self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
        crate::seam::verify_dense_cpu_with_h(
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            tokens,
        )
    }

    /// [`prefill_logits_and_h_cpu`](Self::prefill_logits_and_h_cpu)'s ALL-ROWS twin (MTP Phase 2,
    /// issue #33): returns the LM-head input row for EVERY one of `tokens`, not just the last — the
    /// shape `crate::mtp::catch_up` needs to prime the head's KV over a whole prompt in one call
    /// (`docs/MTP.md`'s `process()` hook runs after every target ubatch, not just the sampled row).
    /// Dense non-MoE models only. Returns `(logits [tokens.len()*vocab], h [tokens.len()*n_embd])`.
    pub fn verify_logits_and_h_cpu(&self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
        crate::seam::verify_rows_cpu_with_h(
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            tokens,
        )
    }

    /// [`prefill_logits_cpu`](Self::prefill_logits_cpu)'s Vulkan twin, for the CPU/Vulkan
    /// cross-backend parity check.
    pub fn prefill_logits_vulkan(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        crate::seam::verify_dense_vulkan(
            &vk,
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            tokens,
        )
    }

    /// Open a Phase-2 DiffusionGemma denoise session on the CPU reference backend (see
    /// `docs/DIFFUSIONGEMMA.md`): [`prefill`](DiffusionGemmaCpuSession::prefill) causally
    /// prefills the prompt ONCE (encoder scalars, KV rows `0..P`), then repeated
    /// [`denoise`](DiffusionGemmaCpuSession::denoise) calls forward the C-row canvas (decoder
    /// scalars, the bidirectional `Canvas` mask) against the SAME session — each call OVERWRITES
    /// KV rows `P..P+C`, so a caller can denoise the same block with a different partially-
    /// unmasked canvas over and over (the loop Phase 3 drives). `max_ctx` sizes the session's KV
    /// cache (must fit the whole prompt + canvas_length + any headroom for later blocks).
    pub fn diffusion_gemma_cpu_session(&self, max_ctx: usize) -> DiffusionGemmaCpuSession {
        DiffusionGemmaCpuSession {
            be: CpuBackend::new(),
            state: None,
            max_ctx,
        }
    }

    /// [`diffusion_gemma_cpu_session`](Self::diffusion_gemma_cpu_session)'s Vulkan twin, for the
    /// CPU/Vulkan cross-backend parity check.
    pub fn diffusion_gemma_vulkan_session(
        &self,
        max_ctx: usize,
    ) -> Result<DiffusionGemmaVulkanSession> {
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        Ok(DiffusionGemmaVulkanSession {
            be: vk,
            state: None,
            max_ctx,
        })
    }

    /// [`diffusion_gemma_vulkan_session`](Self::diffusion_gemma_vulkan_session)'s Metal twin
    /// (Phase D — see docs/DIFFUSIONGEMMA.md's Metal note). The denoise path
    /// (`generate_dense_backend`'s `denoise_req` branch) is backend-generic, so this is exactly
    /// the Vulkan constructor with Metal's own weight-upload closure.
    #[cfg(target_os = "macos")]
    pub fn diffusion_gemma_metal_session(
        &self,
        max_ctx: usize,
    ) -> Result<DiffusionGemmaMetalSession> {
        let mtl = infr_metal::MetalBackend::new().map_err(|e| anyhow!("metal init: {e}"))?;
        Ok(DiffusionGemmaMetalSession {
            be: mtl,
            state: None,
            max_ctx,
        })
    }

    /// Token-level bench on the CPU reference backend (no GPU): prefill `n_prompt` dummy tokens, then
    /// decode `n_gen`, returning the timing ([`crate::GenStats`] has `prompt_secs`/`decode_secs`). Lets
    /// `infr bench -ngl 0` measure prefill (pp = n_prompt/prompt_secs) and decode (tg = n_gen/decode_secs)
    /// directly comparable to `llama-bench -ngl 0`. Dummy tokens — timing is data-independent.
    pub fn bench(&self, n_prompt: usize, n_gen: usize) -> Result<crate::GenStats> {
        let prompt: Vec<u32> = (0..n_prompt.max(1)).map(|i| (i % 100) as u32).collect();
        let (_, stats) = crate::seam::generate_dense_cpu(
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt,
            n_gen,
            |_| {},
        )?;
        Ok(stats)
    }

    /// Run the dense decode through the agnostic compute seam on the **Vulkan** backend — the GPU
    /// twin of [`generate_cpu`](Self::generate_cpu). Each native-dtype GGUF weight is padded + uploaded
    /// to VRAM (the CPU path maps it zero-copy instead); the per-token [`infr_core::graph::Graph`] is
    /// compiled + executed by `VulkanBackend`; greedy tokens are detokenized. Same graph, two
    /// backends — this is the end-to-end dense CPU↔GPU parity path.
    pub fn generate_dense_vulkan(&self, prompt: &str, max_new: usize) -> Result<String> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        let (generated, _stats) = crate::seam::generate_dense_vulkan(
            &vk,
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |_| {},
        )?;
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Token-level Vulkan greedy generation: prefill the given prompt token ids and stream each
    /// GENERATED token id through `on_id` (BOS/template handling is entirely the caller's — nothing
    /// is prepended). Returns the generated ids. The id-exact, Vulkan-backed counterpart to
    /// [`Self::generate_cpu_ids`] — used for token-identity checks (CPU oracle, paged-vs-resident
    /// MoE) that need raw ids rather than detokenized text. A fresh [`infr_vulkan::VulkanBackend`]
    /// per call (like [`Self::generate_dense_vulkan`]) — every weight re-uploads, so the MoE
    /// placement decision (resident/host-visible/paged — see `generate_dense_vulkan_session`) is
    /// re-made fresh each call, which is exactly what a paged-vs-resident A/B needs.
    pub fn generate_vulkan_ids(
        &self,
        prompt_tokens: &[u32],
        max_new: usize,
        on_id: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        let (generated, _stats) = crate::seam::generate_dense_vulkan(
            &vk,
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            prompt_tokens,
            max_new,
            on_id,
        )?;
        Ok(generated)
    }

    /// Token-level bench on the Vulkan seam, llama-bench-comparable: ONE weight upload (a
    /// persistent session) + an untimed pipeline warmup, then per rep — reset the KV, warm it to
    /// `depth` (untimed), and time ONE metric: `pg` = a whole (P prefill + G decode) turn,
    /// `n_gen > 0` = decode at depth, else prefill of `n_prompt` at depth. Returns the per-rep
    /// tokens/sec samples.
    pub fn bench_vulkan(
        &self,
        n_prompt: usize,
        n_gen: usize,
        depth: usize,
        pg: Option<(usize, usize)>,
        reps: usize,
    ) -> Result<Vec<f64>> {
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        let (p_eff, g_eff) = pg.unwrap_or((n_prompt, n_gen));
        // +16: the untimed warmup runs an 8-prompt + 2-gen turn through the SAME session, so the
        // KV must fit it even when the measured shape is tiny (pp2 with +8 sized the cache to 10
        // and the warmup itself overflowed it).
        let want = depth + p_eff.max(1) + g_eff + 16;
        let dummy = |n: usize| -> Vec<u32> { (0..n.max(1)).map(|i| (i % 100) as u32).collect() };
        let mut state: Option<crate::seam::SeamKv> = None;
        let run = |prompt_len: usize,
                   gen: usize,
                   state: &mut Option<crate::seam::SeamKv>|
         -> Result<crate::GenStats> {
            let (_, stats) = crate::seam::generate_dense_vulkan_session(
                &vk,
                &self.gguf,
                &self.cfg,
                self.embd(),
                self.per_layer_embd.as_ref(),
                &dummy(prompt_len),
                gen,
                |_| {},
                state,
                want,
                None,
            )?;
            Ok(stats)
        };
        // Untimed work must stay out of the INFR_PROF2 profile: the warmup turn's m7 batched
        // rows and the depth warm's huge prefill would otherwise dominate/pollute the per-shape
        // aggregate for the tiny timed shape (recorders read the env at construction, so
        // suppressing it around a run() disables their timestamps entirely). Same pattern as
        // DenseSeamChat::warmup. `gpu_reset` additionally drops anything profiled before the
        // timed reps (e.g. session-init submits) from the exit aggregate.
        let unprofiled = |prompt_len: usize,
                          gen: usize,
                          state: &mut Option<crate::seam::SeamKv>|
         -> Result<crate::GenStats> {
            let prof2 = std::env::var_os("INFR_PROF2");
            if prof2.is_some() {
                std::env::remove_var("INFR_PROF2");
            }
            let r = run(prompt_len, gen, state);
            if let Some(v) = prof2 {
                std::env::set_var("INFR_PROF2", v);
            }
            r
        };
        // Untimed warmup: uploads the weights and compiles every pipeline the timed reps hit.
        unprofiled(8, 2, &mut state)?;
        infr_prof_rt::gpu_reset();
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps.max(1) {
            if let Some(st) = state.as_mut() {
                st.reset();
            }
            if depth > 0 {
                unprofiled(depth, 0, &mut state)?; // warm the cache to `depth` (untimed)
            }
            if let Some((p, g)) = pg {
                // coding-agent turn: prompt ingest + reply generation timed together.
                let s = run(depth + p, g, &mut state)?;
                samples.push((p + g) as f64 / (s.prompt_secs + s.decode_secs).max(1e-9));
            } else if n_gen > 0 {
                // decode at depth: 1-token suffix feeds the loop, the timed part is the decode.
                let s = run(depth + 1, n_gen, &mut state)?;
                samples.push(n_gen as f64 / s.decode_secs.max(1e-9));
            } else {
                // +1: the suffix's LAST token is the decode feed and is never processed at
                // gen=0, and a suffix of <= 2 skips batched prefill entirely — so `depth + N`
                // measured N-1 batched rows (and pp2 measured nothing, reporting the 1e-9
                // floor). With +1, exactly N rows batch-prefill (positions depth..depth+N) and
                // prompt_secs covers precisely them — llama-bench's -p N semantics.
                let s = run(depth + n_prompt + 1, 0, &mut state)?;
                samples.push(n_prompt as f64 / s.prompt_secs.max(1e-9));
            }
        }
        Ok(samples)
    }

    /// Open a persistent Metal seam session (the Apple-GPU twin of
    /// [`vulkan_session`](Self::vulkan_session)): weights uploaded ONCE, KV sized to `max_ctx`,
    /// later calls prefill only the un-cached suffix.
    #[cfg(target_os = "macos")]
    pub fn metal_session(&self, max_ctx: usize) -> Result<DenseMetalSession> {
        let mtl = infr_metal::MetalBackend::new().map_err(|e| anyhow!("metal init: {e}"))?;
        Ok(DenseMetalSession {
            mtl,
            pool: SlotPool::new(),
            max_ctx,
        })
    }

    /// Greedy generation on the Metal seam through a persistent session (see
    /// [`metal_session`](Self::metal_session)). `stats.n_prompt` reports the tokens actually
    /// PREFILLED (the un-cached suffix) — the TTFT-honest count.
    #[cfg(target_os = "macos")]
    pub fn generate_metal_session(
        &self,
        session: &mut DenseMetalSession,
        prompt: &str,
        max_new: usize,
        on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        self.generate_metal_session_constrained(session, prompt, max_new, None, on_piece)
    }

    /// [`generate_metal_session`](Self::generate_metal_session) with an optional llguidance
    /// grammar constraint (serve's forced tool_choice) applied to the decode.
    #[cfg(target_os = "macos")]
    pub fn generate_metal_session_constrained(
        &self,
        session: &mut DenseMetalSession,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let slot = session.pool.pick(&session.mtl, &self.cfg, &prompt_tokens)?;
        let (_generated, stats) = crate::seam::generate_dense_metal_session(
            &session.mtl,
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
            &mut session.pool.slots[slot],
            session.max_ctx,
            constraint,
        )?;
        Ok(stats)
    }

    /// Greedy generation on the reference **Metal** backend through the agnostic seam (the
    /// Apple-GPU twin of [`generate_cpu`](Self::generate_cpu)): weights are uploaded to Metal
    /// buffers, the per-token [`infr_core::graph::Graph`] is compiled + executed by `MetalBackend`,
    /// and generated tokens stream through `on_piece`.
    #[cfg(target_os = "macos")]
    pub fn generate_metal(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        let mtl = infr_metal::MetalBackend::new().map_err(|e| anyhow!("metal init: {e}"))?;
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let (_generated, stats) = crate::seam::generate_dense_metal(
            &mtl,
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
        )?;
        Ok(stats)
    }

    /// Speculative decoding on the Metal seam (`self` = TARGET, `draft` = the small
    /// same-tokenizer model): the draft session proposes `k` greedy tokens, ONE batched verify
    /// forward of the target checks all of them (LM head on every candidate row), the matching
    /// prefix commits plus the target's own next token as a bonus. Greedy-only (INFR_TEMP=0) —
    /// every committed token is checked against (or produced by) a verify-forward argmax, so
    /// the committed stream is the target's greedy stream over the VERIFY forward. That equals
    /// target-only greedy decode exactly unless a near-tie logit splits between the batched
    /// f16 verify kernels and decode's exact-f32 GEMV; end-to-end equality is pinned by
    /// `metal_spec_decode_matches_target_only_greedy`. Rollback is the session prefix-diff:
    /// rejected rows just get overwritten by the next round's suffix prefill.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_metal_spec(
        &self,
        session: &mut DenseMetalSession,
        draft: &SeamModel,
        draft_session: &mut DenseMetalSession,
        prompt: &str,
        max_new: usize,
        k: usize,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let sampler = crate::sampling::Sampler::from_env();
        if sampler.temp > 0.0 {
            return Err(anyhow!(
                "speculative decoding is greedy-only — set INFR_TEMP=0"
            ));
        }
        let vocab = self.cfg.vocab;
        let argmax = |row: &[f32]| -> u32 {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            bi as u32
        };
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let mut committed: Vec<u32> = enc.get_ids().to_vec();
        let n_prompt = committed.len();

        // Conversation slots, like the plain session chat: pick the best-prefix slot in BOTH
        // sessions from the prompt, once per call (the indices stay stable — LRU recycling only
        // happens inside pick). A returning conversation suffix-prefills its own slots; a
        // different conversation forks/seeds instead of clobbering — multi-user spec serve
        // stops paying a full re-prefill of BOTH models on every conversation switch.
        let t_slot = session.pool.pick(&session.mtl, &self.cfg, &committed)?;
        let d_slot = draft_session
            .pool
            .pick(&draft_session.mtl, &draft.cfg, &committed)?;

        // Initial fill: the target's normal (chunked-prefill) path produces the first token —
        // verify forwards are only for the small k+1 suffixes.
        let t0 = std::time::Instant::now();
        let mut acc_buf: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let (first, _stats) = crate::seam::generate_dense_metal_session(
            &session.mtl,
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &committed,
            1,
            |_| {},
            &mut session.pool.slots[t_slot],
            session.max_ctx,
            None,
        )?;
        let prompt_secs = t0.elapsed().as_secs_f64();
        let mut t_next = *first.first().ok_or_else(|| anyhow!("empty first token"))?;
        let mut out: Vec<u32> = Vec::new();
        let ignore_eos = std::env::var("INFR_IGNORE_EOS").is_ok();
        let t1 = std::time::Instant::now();
        // Adaptive draft length: k tracks recent acceptance (EMA) — code-shaped output
        // accepts ~3/round and deserves the full k; high-entropy text accepts ~1-2 and a
        // long draft just adds verify rows that get thrown away (measured net-NEGATIVE at
        // fixed k=4 on prose). Bounds [1, k].
        let mut ema = 2.0f64;
        'outer: while out.len() < max_new {
            // Commit the token the target already chose (initial sample or verify bonus).
            let is_eos =
                !ignore_eos && (self.cfg.eos_ids.contains(&t_next) || t_next == self.cfg.eos);
            out.push(t_next);
            if !is_eos {
                stream_token(
                    &self.tokenizer,
                    &mut acc_buf,
                    &mut printed,
                    t_next,
                    &mut on_piece,
                );
            }
            committed.push(t_next);
            if is_eos || out.len() >= max_new {
                break;
            }
            // Draft proposes up to k greedy continuations of the committed stream (its session
            // suffix-prefills the divergence from last round automatically).
            let k_now = (ema.round() as usize).clamp(1, k);
            let budget = k_now.min(max_new - out.len());
            let td = std::time::Instant::now();
            let (cand, _) = crate::seam::generate_dense_metal_session(
                &draft_session.mtl,
                &draft.gguf,
                &draft.cfg,
                draft.embd(),
                draft.per_layer_embd.as_ref(),
                &committed,
                budget,
                |_| {},
                &mut draft_session.pool.slots[d_slot],
                draft_session.max_ctx,
                None,
            )?;
            // Verify: one batched target forward over [t_next, cand..]; row i's argmax is the
            // target's choice after consuming everything up to and including suffix row i.
            let mut feed = committed.clone();
            feed.extend_from_slice(&cand);
            if feed.len() + 1 > session.max_ctx {
                break; // context full: the committed stream is still exact
            }
            let td = td.elapsed();
            let tv = std::time::Instant::now();
            let (logits, vstats) = crate::seam::verify_dense_metal2(
                &session.mtl,
                &self.gguf,
                &self.cfg,
                self.embd(),
                self.per_layer_embd.as_ref(),
                &feed,
                &mut session.pool.slots[t_slot],
                session.max_ctx,
            )?;
            if std::env::var("INFR_SPEC_DEBUG").is_ok() {
                eprintln!(
                    "[spec] draft {:.1}ms verify {:.1}ms (exec {:.1}ms) cand={}",
                    td.as_secs_f64() * 1e3,
                    tv.elapsed().as_secs_f64() * 1e3,
                    vstats * 1e3,
                    cand.len()
                );
            }
            let m = logits.len() / vocab;
            // Rows cover feed's un-cached suffix; the candidate checks use the LAST cand.len()+1
            // rows (the row before cand[0] is t_next's — its argmax checks cand[0]).
            let base = m - (cand.len() + 1);
            // Target's greedy choice at each candidate row plus the bonus row, then the pure
            // accept decision (unit-tested in `spec_accept_tests`, incl. the rejection branch the
            // self-spec e2e test can't reach): the longest prefix the target ratifies and the
            // next committed token — the target's correction at the first mismatch, or the bonus
            // token when all candidates accept.
            let varg: Vec<u32> = (0..=cand.len())
                .map(|j| argmax(&logits[(base + j) * vocab..(base + j + 1) * vocab]))
                .collect();
            let (accepted, next_tok) = spec_accept(&cand, &varg);
            for &c in &cand[..accepted] {
                let is_eos = !ignore_eos && (self.cfg.eos_ids.contains(&c) || c == self.cfg.eos);
                out.push(c);
                if !is_eos {
                    stream_token(
                        &self.tokenizer,
                        &mut acc_buf,
                        &mut printed,
                        c,
                        &mut on_piece,
                    );
                }
                committed.push(c);
                if is_eos || out.len() >= max_new {
                    break 'outer;
                }
            }
            ema = 0.7 * ema + 0.3 * (accepted as f64 + 1.0);
            // Roll the committed view back past the rejected tail: the session's `cached` holds
            // all of `feed`, and the next prefix diff overwrites the stale rows.
            t_next = next_tok;
        }
        Ok(crate::GenStats {
            n_prompt,
            prompt_secs,
            n_gen: out.len(),
            decode_secs: t1.elapsed().as_secs_f64(),
        })
    }

    /// Token-level bench on the **Metal** backend through the agnostic seam (the Apple-GPU twin of
    /// [`bench`](Self::bench)): prefill `n_prompt` dummy tokens, decode `n_gen`, return the timing.
    /// Lets `infr bench` (with `INFR_METAL=1`) measure pp/tg directly comparable to `llama-bench`
    /// on the Metal build.
    /// Runs through a persistent [`DenseMetalSession`] so backend, uploaded weights, compiled
    /// pipelines, and the dequant/repack weight caches all survive across reps — a fresh backend
    /// per rep re-paid every one-time cost inside the measurement (a factored-format checkpoint
    /// re-repacked hundreds of MB per rep). The materialized tokens reset each call, so every
    /// rep still measures a FULL prefill (llama-bench keeps one context across reps the same way).
    #[cfg(target_os = "macos")]
    pub fn bench_metal(
        &self,
        session: &mut DenseMetalSession,
        n_prompt: usize,
        n_gen: usize,
    ) -> Result<crate::GenStats> {
        let prompt: Vec<u32> = (0..n_prompt.max(1)).map(|i| (i % 100) as u32).collect();
        if let Some(s) = session.pool.single().as_mut() {
            s.reset_tokens();
        }
        let (_, stats) = crate::seam::generate_dense_metal_session(
            &session.mtl,
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt,
            n_gen,
            |_| {},
            session.pool.single(),
            session.max_ctx,
            None,
        )?;
        Ok(stats)
    }

    /// Render a user turn with the model's OWN embedded chat template (so an instruct model — Gemma,
    /// Qwen, … — answers coherently). Errors if the GGUF has no `tokenizer.chat_template` or it fails
    /// to render — infr only supports models that ship one (no fabricated-ChatML fallback).
    pub fn render_chat(&self, user: &str) -> Result<String> {
        render_chat_user(&self.gguf, &self.tokenizer, self.cfg.eos, user)
            .ok_or_else(no_template_err)
    }

    /// Render a multi-turn conversation `(role, content)` through the model's OWN embedded chat
    /// template — the [`crate::chat::ChatModel::render`] primitive for the CPU dense/MoE path, so the
    /// shared [`crate::chat::Chat`] can drive a history-based REPL. Same template + error contract as
    /// [`render_chat`](Self::render_chat), generalized past a single user turn.
    pub fn render_chat_messages(&self, messages: &[(&str, &str)]) -> Result<String> {
        render_chat_jinja(&self.gguf, &self.tokenizer, self.cfg.eos, messages, true)
            .ok_or_else(no_template_err)
    }

    /// Greedy generation on the CPU reference backend (no GPU). Returns the decoded text plus
    /// timing/counts ([`crate::GenStats`]) for the caller's stats line.
    /// The generated text is delivered through `on_piece` as it streams; only timing/counts are
    /// returned.
    pub fn generate_cpu(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        // Stream each generated token: incrementally detokenize and emit the new suffix.
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let (_generated, stats) = crate::seam::generate_dense_cpu(
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
        )?;
        Ok(stats)
    }

    /// Token-level CPU greedy generation: prefill the given prompt token ids and stream each
    /// GENERATED token id through `on_id` (BOS/template handling is entirely the caller's — nothing
    /// is prepended). Returns the generated ids. The id-exact counterpart to [`Self::generate_cpu`],
    /// used for token-identity checks against a reference implementation.
    pub fn generate_cpu_ids(
        &self,
        prompt_tokens: &[u32],
        max_new: usize,
        on_id: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let (generated, _stats) = crate::seam::generate_dense_cpu(
            &self.gguf,
            &self.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            prompt_tokens,
            max_new,
            on_id,
        )?;
        Ok(generated)
    }
}

/// A persistent CPU-reference session for DiffusionGemma's two-pass forward (Phase 2 — see
/// docs/DIFFUSIONGEMMA.md and [`SeamModel::diffusion_gemma_cpu_session`]). Model-independent (like
/// [`DenseVulkanSession`]/[`DenseMetalSession`]) — `prefill`/`denoise` take the `&SeamModel` per
/// call instead of borrowing it at construction, so a [`crate::chat::ChatModel`] can hold both an
/// owned `SeamModel` and a persistent session side by side (Phase 3 — no self-referential borrow).
pub struct DiffusionGemmaCpuSession {
    be: CpuBackend,
    state: Option<crate::seam::SeamKv>,
    max_ctx: usize,
}

/// [`DiffusionGemmaCpuSession`]'s Vulkan twin (see [`SeamModel::diffusion_gemma_vulkan_session`]).
pub struct DiffusionGemmaVulkanSession {
    be: infr_vulkan::VulkanBackend,
    state: Option<crate::seam::SeamKv>,
    max_ctx: usize,
}

/// [`DiffusionGemmaVulkanSession`]'s Metal twin (Phase D — see
/// [`SeamModel::diffusion_gemma_metal_session`]).
#[cfg(target_os = "macos")]
pub struct DiffusionGemmaMetalSession {
    be: infr_metal::MetalBackend,
    state: Option<crate::seam::SeamKv>,
    max_ctx: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DiffusionGemmaCpuSession {
    /// Causal prefill of `tokens` (encoder scalars, chunked/per-token like every other dense
    /// prefill on this seam) — writes KV rows `0..tokens.len()`. Call once per block before
    /// [`denoise`](Self::denoise); a second call with a prompt that EXTENDS the previous one
    /// continues the session (ChatSession-style prefix reuse), matching every other session on
    /// this seam.
    pub fn prefill(&mut self, model: &SeamModel, tokens: &[u32]) -> Result<()> {
        crate::seam::generate_dense_backend(
            &self.be,
            &|_name, tb, dt, _n| match tb {
                crate::seam::WBytes::Mmap(tb) => Ok((self.be.map_weight(tb), dt)),
                crate::seam::WBytes::Owned(v) => {
                    let buf = self
                        .be
                        .alloc(v.len().max(1), BufferUsage::Weights)
                        .map_err(|e| anyhow!("{e}"))?;
                    self.be
                        .upload(buf.as_ref(), &v)
                        .map_err(|e| anyhow!("{e}"))?;
                    Ok((buf, dt))
                }
            },
            &model.gguf,
            &model.cfg,
            model.embd(),
            model.per_layer_embd.as_ref(),
            tokens,
            1, // rides the ordinary per-token causal-prefill loop (see `verify_dense_cpu`'s doc)
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        Ok(())
    }

    /// ONE canvas denoise forward over the session's already-prefilled prompt (see
    /// `DenoiseReq`'s doc): `canvas_tokens.len()` MUST match every call on this session (the
    /// model's `canvas_length`). Returns `[canvas_tokens.len() * vocab]` raw logits. `sc_logits`
    /// is the PREVIOUS step's raw canvas logits for self-conditioning (`None` = off, matching the
    /// reference's step-0 gate); `temp_inv` is the self-conditioning softmax temperature divisor
    /// (unused when `sc_logits` is `None`). Panics-free even before [`prefill`](Self::prefill) is
    /// ever called — errors instead (an empty prompt, P=0).
    pub fn denoise(
        &mut self,
        model: &SeamModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
    ) -> Result<Vec<f32>> {
        let mut out_logits = Vec::new();
        let mut reduced = None; // CPU never requests the GPU reducer (`u: None` below) — stays `None`
        crate::seam::generate_dense_backend(
            &self.be,
            &|_name, tb, dt, _n| match tb {
                crate::seam::WBytes::Mmap(tb) => Ok((self.be.map_weight(tb), dt)),
                crate::seam::WBytes::Owned(v) => {
                    let buf = self
                        .be
                        .alloc(v.len().max(1), BufferUsage::Weights)
                        .map_err(|e| anyhow!("{e}"))?;
                    self.be
                        .upload(buf.as_ref(), &v)
                        .map_err(|e| anyhow!("{e}"))?;
                    Ok((buf, dt))
                }
            },
            &model.gguf,
            &model.cfg,
            model.embd(),
            model.per_layer_embd.as_ref(),
            &[], // denoise never touches the prompt/generation token stream — see `DenoiseReq`
            0,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
            None,
            Some(crate::seam::DenoiseReq {
                canvas_tokens,
                sc_logits,
                temp_inv,
                out_logits: &mut out_logits,
                u: None,
                sample_temp_inv: 0.0,
                reduced: &mut reduced,
            }),
        )?;
        Ok(out_logits)
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DiffusionGemmaVulkanSession {
    /// [`DiffusionGemmaCpuSession::prefill`]'s Vulkan twin. Weights bind through the SHARED
    /// [`crate::seam::vulkan_moe_binder`] so DG gets the same MoE expert placement tiers
    /// (resident / `INFR_CACHE` / auto-paged) as every other MoE model — DG's fused
    /// `ffn_gate_up_exps` bank pages under `Role::Gate` and its mixed Q5_0/Q8_0 down banks split
    /// into per-byte-size pools (see `infr_vulkan::pager`'s MoE-session doc).
    pub fn prefill(&mut self, model: &SeamModel, tokens: &[u32]) -> Result<()> {
        let bind = crate::seam::vulkan_moe_binder(
            &self.be,
            &model.gguf,
            &model.cfg,
            self.state.is_none(),
            self.max_ctx,
        )?;
        crate::seam::generate_dense_backend(
            &self.be,
            &*bind,
            &model.gguf,
            &model.cfg,
            model.embd(),
            model.per_layer_embd.as_ref(),
            tokens,
            1,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        // Once per prefill (a denoise step would print per step — far too noisy).
        self.be.print_moe_pager_stats();
        Ok(())
    }

    /// [`DiffusionGemmaCpuSession::denoise`]'s Vulkan twin. Perf slice 3
    /// (docs/DIFFUSIONGEMMA.md): when `u` is `Some`, this asks `generate_dense_backend` to try the
    /// GPU entropy-bound sampler reducer on this step's logits (see
    /// [`crate::seam::DenoiseOutcome`]) — `sample_temp_inv` is THIS step's sampler temperature
    /// divisor (`denoise_block`'s local `temp_inv`, distinct from `temp_inv` above, which is the
    /// PREVIOUS step's self-conditioning divisor) and `u` is `canvas_tokens.len()` host-drawn
    /// uniform `[0,1)` floats (the seeded CDF-inversion draw). `u: None` always takes the ordinary
    /// full-logits path (`gpu_seam_matches_cpu_diffusion_gemma_denoise`'s direct row-by-row
    /// comparison needs the full array); the backend may ALSO decline on its own (falls back to
    /// `DenoiseOutcome::Logits`) even when `u` is `Some` — the caller (`diffusion.rs::denoise_block`,
    /// or a direct test) handles both outcomes.
    pub fn denoise(
        &mut self,
        model: &SeamModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
        sample_temp_inv: f32,
        u: Option<&[f32]>,
    ) -> Result<crate::seam::DenoiseOutcome> {
        let mut out_logits = Vec::new();
        let mut reduced = None;
        // The shared placement-aware binder (see `prefill`): only ever CALLED when this denoise
        // is the session's first load (no prior `prefill` — a direct-denoise test), where it must
        // make the same placement decision prefill would have.
        let bind = crate::seam::vulkan_moe_binder(
            &self.be,
            &model.gguf,
            &model.cfg,
            self.state.is_none(),
            self.max_ctx,
        )?;
        crate::seam::generate_dense_backend(
            &self.be,
            &*bind,
            &model.gguf,
            &model.cfg,
            model.embd(),
            model.per_layer_embd.as_ref(),
            &[],
            0,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
            None,
            Some(crate::seam::DenoiseReq {
                canvas_tokens,
                sc_logits,
                temp_inv,
                out_logits: &mut out_logits,
                u,
                sample_temp_inv,
                reduced: &mut reduced,
            }),
        )?;
        Ok(match reduced {
            Some(r) => crate::seam::DenoiseOutcome::Reduced(r),
            None => crate::seam::DenoiseOutcome::Logits(out_logits),
        })
    }
}

/// Phase D: Metal twin of [`DiffusionGemmaVulkanSession`] (see
/// [`SeamModel::diffusion_gemma_metal_session`]). `generate_dense_backend`'s `denoise_req` branch
/// is backend-generic (verified — nothing in it besides the Phase-B `gpu_sc` gate distinguishes
/// Vulkan, and that gate now includes Metal too, see `seam.rs`), so this only differs from
/// the Vulkan twin in its weight-upload closure: Metal uploads weights in their NATIVE GGUF dtype
/// with no padding (matching `generate_dense_metal_session`'s closure), unlike Vulkan's
/// `pad_to_u32_align`.
#[cfg(target_os = "macos")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DiffusionGemmaMetalSession {
    /// [`DiffusionGemmaCpuSession::prefill`]'s Metal twin.
    pub fn prefill(&mut self, model: &SeamModel, tokens: &[u32]) -> Result<()> {
        crate::seam::generate_dense_backend(
            &self.be,
            &|_name, tb, dt, _n| {
                let buf = self
                    .be
                    .alloc(tb.len().max(1), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                self.be
                    .upload(buf.as_ref(), &tb)
                    .map_err(|e| anyhow!("{e}"))?;
                Ok((buf, dt))
            },
            &model.gguf,
            &model.cfg,
            model.embd(),
            model.per_layer_embd.as_ref(),
            tokens,
            1,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        Ok(())
    }

    /// [`DiffusionGemmaCpuSession::denoise`]'s Metal twin.
    pub fn denoise(
        &mut self,
        model: &SeamModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
    ) -> Result<Vec<f32>> {
        let mut out_logits = Vec::new();
        let mut reduced = None; // Metal never requests the GPU reducer (`u: None` below, Phase D — Vulkan only for this slice)
        crate::seam::generate_dense_backend(
            &self.be,
            &|_name, tb, dt, _n| {
                let buf = self
                    .be
                    .alloc(tb.len().max(1), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                self.be
                    .upload(buf.as_ref(), &tb)
                    .map_err(|e| anyhow!("{e}"))?;
                Ok((buf, dt))
            },
            &model.gguf,
            &model.cfg,
            model.embd(),
            model.per_layer_embd.as_ref(),
            &[],
            0,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
            None,
            Some(crate::seam::DenoiseReq {
                canvas_tokens,
                sc_logits,
                temp_inv,
                out_logits: &mut out_logits,
                u: None,
                sample_temp_inv: 0.0,
                reduced: &mut reduced,
            }),
        )?;
        Ok(out_logits)
    }
}

/// The speculative-decoding accept decision for one verify round (pure — no backend). Given the
/// draft `cand` and `verify_argmax` — the target verify-forward's greedy choice at each candidate
/// row plus the bonus row (`cand.len() + 1` entries; index `j` is the target's token after
/// consuming `cand[..j]`) — return the number of leading candidates the target ratifies (the
/// longest prefix where its argmax equals the draft) and the next committed token: the target's
/// correction at the first mismatch, or the bonus token when every candidate is accepted.
///
/// Correctness property (see `spec_accept_tests`): the committed stream `cand[..accepted] ++
/// [next]` is always exactly `verify_argmax[..=accepted]` — the target's own greedy stream — no
/// matter what the draft proposed. That is what makes speculative decoding output-identical to
/// target-only greedy; a wrong draft only shortens the accepted prefix, never commits a wrong
/// token. Backend-agnostic (no `cfg` gate): the macOS spec driver above and `crate::mtp`'s Vulkan
/// MTP spec driver (issue #33) both call this — one pure acceptance rule for every spec flavor.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn spec_accept(cand: &[u32], verify_argmax: &[u32]) -> (usize, u32) {
    debug_assert_eq!(verify_argmax.len(), cand.len() + 1);
    let accepted = cand
        .iter()
        .zip(verify_argmax)
        .take_while(|(c, v)| c == v)
        .count();
    (accepted, verify_argmax[accepted])
}

/// Temperature-aware speculative accept rule — the stochastic sibling of [`spec_accept`], used
/// when `INFR_TEMP > 0` (the greedy rule above requires the trunk's argmax as its one-and-only
/// target, which is why pure greedy makes thinking models degenerate — see `crate::sampling`'s
/// `Sampler` doc — and why MTP needs this at all). Implements the standard speculative-sampling
/// acceptance rule (Leviathan & Kalman 2023 / Chen et al. 2023):
///
/// Draft head proposes `cand[i]` from its own truncated proposal distribution `q_dists[i]`
/// (top-k/top-p-truncated at the SAME `Sampler` config the target uses — `sampling::truncated_dist`
/// applied to the head's own logits). The target trunk's verify forward gives the truncated
/// distributions `p_dists[0..=cand.len()]` at those same positions (row `j` = the target's
/// distribution conditioned on `cand[..j]`; `p_dists[cand.len()]` is the bonus row, conditioned on
/// every candidate being accepted). For `i` in `0..cand.len()`: draw `u ~ Uniform(0,1)`; accept
/// `cand[i]` iff `u < min(1, p_i(cand[i]) / q_i(cand[i]))`. On the first rejection, sample the next
/// token from the normalized residual `max(p_i - q_i, 0)` and STOP this cycle (no bonus). If every
/// candidate is accepted, sample the bonus token from `p_dists[cand.len()]`.
///
/// This provably preserves the TARGET's sampling distribution — unlike `spec_accept`, the
/// committed stream is not required to equal any single model's argmax/sample stream token-for-
/// token across a re-run; it's a sample from the target's distribution either way (see
/// `spec_accept_stochastic_tests` for the acceptance-rule properties this is pinned to: identical
/// p/q always accepts, an off-support draft always rejects, the residual always renormalizes, and
/// a fixed seed is deterministic).
fn dist_prob(dist: &[(u32, f32)], x: u32) -> f32 {
    dist.iter()
        .find(|&&(id, _)| id == x)
        .map(|&(_, p)| p)
        .unwrap_or(0.0)
}

/// The rejection-sampling residual `max(p - q, 0)`, renormalized to sum to 1 — the distribution
/// [`spec_accept_stochastic`] resamples from on a coin-flip rejection. Only entries in `p`'s own
/// support can contribute a positive residual (an entry outside `p` has `p == 0`, which clamps to
/// 0 regardless of `q`), so the returned ids are always a subset of `p`'s ids. Empty when `p` and
/// `q` agree pointwise everywhere on `p`'s support — the degenerate case `spec_accept_stochastic`
/// falls back to `p`'s own top choice for (see `spec_accept_stochastic_tests` for both properties).
fn residual_dist(p: &[(u32, f32)], q: &[(u32, f32)]) -> Vec<(u32, f32)> {
    let mut residual: Vec<(u32, f32)> = Vec::with_capacity(p.len());
    let mut sum = 0.0f32;
    for &(id, pv) in p {
        let r = (pv - dist_prob(q, id)).max(0.0);
        if r > 0.0 {
            residual.push((id, r));
            sum += r;
        }
    }
    if sum > 0.0 {
        for e in residual.iter_mut() {
            e.1 /= sum;
        }
    }
    residual
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn spec_accept_stochastic(
    cand: &[u32],
    q_dists: &[Vec<(u32, f32)>],
    p_dists: &[Vec<(u32, f32)>],
    rng: &mut u64,
) -> (usize, u32) {
    debug_assert_eq!(q_dists.len(), cand.len());
    debug_assert_eq!(p_dists.len(), cand.len() + 1);
    for i in 0..cand.len() {
        let x = cand[i];
        let q_x = dist_prob(&q_dists[i], x);
        let p_x = dist_prob(&p_dists[i], x);
        // q_x > 0 always holds in practice (x was drawn from q_dists[i]'s own support), but guard
        // the division for a caller that hands in an inconsistent (cand, q_dists) pair.
        let ratio = if q_x > 0.0 { (p_x / q_x).min(1.0) } else { 0.0 };
        let u = crate::sampling::next_uniform(rng);
        if u < ratio {
            continue; // accept — draw the next position's coin
        }
        // Reject: sample the next committed token from the normalized residual max(p_i - q_i, 0).
        let residual = residual_dist(&p_dists[i], &q_dists[i]);
        let next = if !residual.is_empty() {
            crate::sampling::sample_from_dist(&residual, rng)
        } else {
            // Degenerate: q_i == p_i pointwise on p_i's support, so the residual is empty, yet the
            // coin still rejected (only possible if x itself sat outside p_i's support, p_x == 0).
            // Fall back to p_i's own top choice so the cycle always makes forward progress.
            p_dists[i]
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|&(id, _)| id)
                .unwrap_or(0)
        };
        return (i, next);
    }
    // Every candidate accepted: sample the bonus token from the target's own distribution.
    let bonus = crate::sampling::sample_from_dist(&p_dists[cand.len()], rng);
    (cand.len(), bonus)
}

#[cfg(test)]
mod spec_accept_tests {
    use super::spec_accept;

    #[test]
    fn all_accepted_returns_bonus() {
        // Draft fully matches the target → accept all three, next = the bonus-row token.
        assert_eq!(spec_accept(&[10, 11, 12], &[10, 11, 12, 99]), (3, 99));
    }

    #[test]
    fn reject_at_zero_returns_correction() {
        // Draft wrong at position 0 → accept none, next = the target's correction.
        assert_eq!(spec_accept(&[10, 11, 12], &[7, 11, 12, 99]), (0, 7));
    }

    #[test]
    fn reject_in_middle_commits_prefix() {
        // Accept [10], reject at 1 → next = the correction 8; the rest is discarded.
        assert_eq!(spec_accept(&[10, 11, 12], &[10, 8, 12, 99]), (1, 8));
    }

    #[test]
    fn empty_draft_returns_sole_token() {
        // No candidates (adaptive k floored mid-round) → next = the sole bonus token.
        assert_eq!(spec_accept(&[], &[42]), (0, 42));
    }

    #[test]
    fn committed_stream_is_target_greedy_for_any_draft() {
        // The rejection-sampling invariant the self-spec e2e test cannot reach (its draft always
        // agrees, so acceptance is always full): whatever the draft proposes, the committed
        // tokens equal a prefix of the target's own argmax stream — never a token the target
        // wouldn't have produced. This is the branch the original review flagged as untested.
        let varg = [5u32, 6, 7, 8]; // the target's greedy choice at each of the four rows
        for draft in [
            vec![5, 6, 7], // perfect draft → accept 3
            vec![9, 6, 7], // wrong at 0    → accept 0
            vec![5, 9, 7], // wrong at 1    → accept 1
            vec![5, 6, 9], // wrong at 2    → accept 2
            vec![0, 0, 0], // all wrong     → accept 0
        ] {
            let (accepted, next) = spec_accept(&draft, &varg);
            let mut committed: Vec<u32> = draft[..accepted].to_vec();
            committed.push(next);
            assert_eq!(
                committed,
                varg[..=accepted].to_vec(),
                "draft {draft:?} committed a non-greedy stream"
            );
        }
    }
}

#[cfg(test)]
mod spec_accept_stochastic_tests {
    use super::{residual_dist, spec_accept_stochastic};

    #[test]
    fn identical_p_q_always_accepts() {
        // q == p everywhere ⇒ the ratio is exactly 1 at every position, so `u < 1` always holds
        // (next_uniform draws are in [0,1)) ⇒ every candidate is accepted, every seed.
        let dist = vec![(10u32, 0.5f32), (11, 0.3), (12, 0.2)];
        let cand = [10u32, 11, 12];
        let q_dists = vec![dist.clone(), dist.clone(), dist.clone()];
        let p_dists = vec![dist.clone(), dist.clone(), dist.clone(), dist.clone()];
        for seed in 1..50u64 {
            let mut rng = seed | 1;
            let (accepted, next) = spec_accept_stochastic(&cand, &q_dists, &p_dists, &mut rng);
            assert_eq!(accepted, cand.len(), "seed {seed}: expected full accept");
            // The bonus token must come from the (only nonzero-mass) support of p_dists[3].
            assert!(dist.iter().any(|&(id, _)| id == next));
        }
    }

    #[test]
    fn zero_target_prob_always_rejects() {
        // The drafted token (99) has zero probability under the target's truncated distribution
        // (it isn't in p_dist's support at all) ⇒ ratio == 0 ⇒ always rejected regardless of the
        // coin draw, and the residual must be resampled (never token 99 itself).
        let q_dist = vec![(99u32, 1.0f32)]; // draft head is CERTAIN of a token the target rejects
        let p_dist = vec![(10u32, 0.6f32), (11, 0.4)]; // target never puts mass on 99
        let cand = [99u32];
        let q_dists = vec![q_dist];
        let p_dists = vec![p_dist.clone(), p_dist.clone()];
        for seed in 1..50u64 {
            let mut rng = seed | 1;
            let (accepted, next) = spec_accept_stochastic(&cand, &q_dists, &p_dists, &mut rng);
            assert_eq!(accepted, 0, "seed {seed}: p(x)=0 must always reject");
            assert_ne!(
                next, 99,
                "seed {seed}: rejected token must not be re-emitted"
            );
            assert!(p_dist.iter().any(|&(id, _)| id == next));
        }
    }

    #[test]
    fn residual_normalizes_and_is_nonnegative() {
        // p has mass q doesn't (10, 12); q has mass p doesn't (99, ignored); both have 11 with
        // q's share larger than p's (clamps to 0 there).
        let p = vec![(10u32, 0.5f32), (11, 0.2), (12, 0.3)];
        let q = vec![(11u32, 0.9f32), (99, 0.1)];
        let r = residual_dist(&p, &q);
        // Only p's support can appear; 11's residual (0.2 - 0.9 clamped to 0) must be dropped.
        assert!(r.iter().all(|&(id, _)| id == 10 || id == 12));
        assert!(!r.iter().any(|&(id, _)| id == 11));
        assert!(r.iter().all(|&(_, p)| p >= 0.0));
        let sum: f32 = r.iter().map(|&(_, p)| p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "residual must renormalize to 1, got {sum}"
        );
    }

    #[test]
    fn residual_empty_when_q_dominates_p_everywhere() {
        // q >= p pointwise on p's whole support ⇒ every residual clamps to 0 ⇒ empty.
        let p = vec![(10u32, 0.3f32), (11, 0.3)];
        let q = vec![(10u32, 0.5f32), (11, 0.5)];
        assert!(residual_dist(&p, &q).is_empty());
    }

    #[test]
    fn deterministic_given_fixed_seed() {
        let q_dist = vec![(5u32, 1.0f32)];
        let p_dist = vec![(5u32, 0.4f32), (6, 0.6)];
        let cand = [5u32];
        let q_dists = vec![q_dist];
        let p_dists = vec![p_dist.clone(), p_dist];
        // A fixed, non-env-dependent seed (not `sampling::seed_rng()`, which falls back to a
        // wall-clock seed and would make this test flaky under parallel `cargo test`). Must be odd
        // (the xorshift64 state must be nonzero and `seed_rng()` always ORs in `1`).
        let mut rng_a = 1u64;
        let mut rng_b = 1u64;
        let a = spec_accept_stochastic(&cand, &q_dists, &p_dists, &mut rng_a);
        let b = spec_accept_stochastic(&cand, &q_dists, &p_dists, &mut rng_b);
        assert_eq!(a, b, "same seed must reproduce the same accept decision");
        assert_eq!(rng_a, rng_b, "same seed must reproduce the same rng stream");
    }
}
