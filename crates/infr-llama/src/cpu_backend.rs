//! CPU model runner — builds and drives the agnostic decode [`Graph`] through [`CpuBackend`].
//! The backend itself lives in `infr-cpu`; this module is the model-specific "glue" that
//! assembles the layer graph, uploads weights, and steps the KV cache.
#![allow(clippy::too_many_arguments)]

use crate::{dequant_block, Config, GenStats, PerLayerEmbd};
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::WeightSource;
use infr_cpu::CpuBackend;
use infr_gguf::{Gguf, TensorBytes};

// ─── Qwen3 dense CPU decode runner ───────────────────────────────────────────────
//
// Builds the n=1 decode Graph and drives it through `CpuBackend`, one token at a time, for BOTH
// prompt ingestion (looped) and generation — so no GEMM/flash prefill kernels are needed on CPU.
// The KV cache grows one row per step. Validates the agnostic seam end-to-end against the GPU path.

/// FFN weight handles: a dense gated FFN, or a qwen3moe routed-expert bank (router + stacked
/// per-expert gate/up/down).
enum FfnW {
    Dense {
        wgate: TensorId,
        wup: TensorId,
        wdown: TensorId,
    },
    /// Combined gate+up weight `[2*nff, ne]` (one GEMV/GEMM + `GatedActFused`); see `fuse_gu`.
    DenseFused { wgu: TensorId, wdown: TensorId },
    Moe {
        router: TensorId,
        gate_exps: TensorId,
        up_exps: TensorId,
        down_exps: TensorId,
    },
}

/// Per-layer weight handles captured while building one decode graph (q/k-norm + the gemma
/// sandwich norms are optional; `wv` is absent on gemma4 full-attention layers, which reuse the raw
/// K projection as V). The order they're declared in MUST match the upload order so `weights[i]`
/// binds to `wbufs[i]`.
struct LayerW {
    attn_norm: TensorId,
    wq: TensorId,
    wk: TensorId,
    wv: Option<TensorId>,
    q_norm: Option<TensorId>,
    k_norm: Option<TensorId>,
    wo: TensorId,
    post_attn: Option<TensorId>,
    ffn_norm: TensorId,
    ffn: FfnW,
    post_ffw: Option<TensorId>,
    // gemma4 E2B per-layer input embedding: inp_gate, proj, post_norm.
    pl_inp_gate: Option<TensorId>,
    pl_proj: Option<TensorId>,
    pl_post_norm: Option<TensorId>,
}

/// Handles into one freshly-built decode graph that the driver re-binds each step.
struct DecodeHandles {
    hidden: TensorId,
    positions: TensorId,
    rope_freqs: Option<TensorId>, // gemma4 proportional-RoPE divisors (full-attention layers)
    // gemma4 E2B host-gathered per-layer TOKEN embedding rows `[n_layer*npl]` — the graph Input
    // the driver binds `ipl_buf` to; the GPU prologue turns this into the layer loop's actual
    // per-layer input vector (see `per_layer_inp` inside `build`).
    pl_tok_in: Option<TensorId>,
    logits: TensorId,
    k_cache: Vec<TensorId>,
    v_cache: Vec<TensorId>,
    weights: Vec<TensorId>, // flat, in declaration == upload order
}

/// Greedy CPU generation for a decoder (Qwen3 / Llama / Gemma 3 / Gemma 4 dense+E2B / qwen3moe). The
/// attention block is shared; the FFN is either a dense gated FFN or a routed-expert MoE bank; gemma4
/// E2B adds per-layer input embeddings + KV-layer sharing. `prompt` is the full token prefix; returns
/// the generated continuation. Stops at EOS or `max_new`.
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
    )
}

/// GPU seam runner: the SAME dense forward as [`generate_dense_cpu`], but on the Vulkan backend
/// through the agnostic [`Graph`] adapter (weights padded + uploaded to VRAM instead of mmap-mapped).
/// This is the end-to-end GPU parity/perf path — running it and diffing the CPU oracle proves the
/// adapter, and its decode tok/s (still recompiling the graph per token) is the baseline
/// record-once replay must close. Prefill's batched attention is decode-only on the seam, so the
/// caller may pass short prompts to force the per-token path.
pub(crate) fn generate_dense_gpu(
    vk: &infr_vulkan::VulkanBackend,
    g: &Gguf,
    cfg: &Config,
    token_embd: &[f32],
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_gpu_session(
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

/// [`generate_dense_gpu`] with a caller-held [`SeamKv`]: hold `state` (+ a `want_ctx` capacity)
/// across calls and each turn prefills only the suffix that differs from the cached tokens —
/// ChatSession-style KV reuse on the agnostic seam.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_dense_gpu_session(
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
    // ── MoE expert auto-fit ──────────────────────────────────────────────────────────────────
    // When the full weight set (+ the want_ctx KV cache + activation headroom) exceeds VRAM,
    // keep the FIRST n_host_moe layers' stacked expert banks in HOST-VISIBLE memory instead of
    // VRAM. The graph and lowering are untouched — the banks bind like any other weight and the
    // GPU reads them over the bus (the seam's zero-readback GPU routing can't know active experts
    // host-side, so per-expert streaming à la bespoke doesn't apply; resident-or-host per layer
    // does). INFR_NCMOE overrides the automatic count.
    let n_host_moe: usize = if cfg.moe.is_some() {
        let explicit = std::env::var("INFR_NCMOE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        match explicit {
            Some(n) => n.min(cfg.n_layer),
            None => {
                let fp = crate::weights::weight_footprint(g);
                let vram = vk.vram();
                let kv_bytes: u64 = (0..cfg.n_layer)
                    .map(|l| (cfg.layer_n_kv(l) * cfg.layer_head_dim(l) * 2 * 2) as u64)
                    .sum::<u64>()
                    * (want_ctx as u64 + 64);
                const ACT_HEADROOM: u64 = 512 * 1024 * 1024;
                let budget = vram
                    .available
                    .saturating_sub(fp.dense + kv_bytes + ACT_HEADROOM);
                let per_layer = (fp.expert / cfg.n_layer.max(1) as u64).max(1);
                let gpu_layers = (budget / per_layer).min(cfg.n_layer as u64) as usize;
                cfg.n_layer - gpu_layers
            }
        }
    } else {
        0
    };
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
    generate_dense_backend(
        vk,
        &|name, tb, dt, _n| {
            // Raw upload for EVERY dtype — the file's bytes go straight to VRAM (u32-padded) and the
            // kernel reads/dequants the native dtype in-shader. F16 → f16 coopmat GEMM / f16 GEMV;
            // F32 stays native (rmsnorm/qk_norm_rope read f32); bf16 → in-shader expand (bf16 is the
            // top 16 bits of an f32, EXACT; the warp GEMM narrows to f16 for the matrix cores like
            // every other format); quant weights → raw blocks. No host dtype conversion on any path.
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
    )
}

/// Metal seam runner: the SAME dense forward as [`generate_dense_cpu`], on the reference Metal
/// backend through the agnostic [`Graph`]. Weights are uploaded to Metal buffers in their NATIVE
/// GGUF dtype (the backend dequantizes lazily in its own `bytes_to_f32`, exactly like the CPU
/// interpreter — so a quant weight occupies ~quant size, not 8× f32).
#[cfg(target_os = "macos")]
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

/// Persistent-session Metal seam runner — the Metal twin of [`generate_dense_gpu_session`]:
/// weights upload once into `state`, the KV cache is sized to `want_ctx`, and each call prefills
/// only the suffix that differs from the tokens already materialized in the cache.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
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
    )
}

/// Speculative VERIFY on the Metal seam: one batched forward of `tokens`' un-cached suffix with
/// the LM head on every suffix row. Returns the [m, vocab] logits plus the graph-execute
/// seconds, and leaves the session's KV + `cached` covering all of `tokens` — the caller
/// commits the accepted prefix and the next call's prefix diff overwrites whatever was
/// speculatively written past it.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
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
    )?;
    Ok((logits, stats.prompt_secs))
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
/// is disabled for it). Must match the adapter's `decode_eligible` rejection.
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

pub(crate) struct SeamKv {
    /// The uploaded weights, SHARED across slots (Arc): forking a new conversation slot costs
    /// only its KV + IO buffers, never a re-upload.
    weights: std::sync::Arc<SeamWeights>,
    kbufs: Vec<Box<dyn Buffer>>,
    vbufs: Vec<Box<dyn Buffer>>,
    /// KV cache element dtypes, chosen per-side (K and V independent). Fork/seed reuse them so a
    /// forked slot sizes + copies its buffers to match this slot's layout.
    k_fmt: DType,
    v_fmt: DType,
    hidden_buf: Box<dyn Buffer>,
    pos_buf: Box<dyn Buffer>,
    ipl_buf: Option<Box<dyn Buffer>>,
    logits_buf: Box<dyn Buffer>,
    max_ctx: usize,
    /// Token ids whose KV rows are materialized (prompt + generated of the last turn).
    cached: Vec<u32>,
}

/// The upload-once half of a [`SeamKv`]: weight buffers + their declared (dtype, numel) specs and
/// the rope_freqs constant. Shared across conversation slots via `Arc`.
pub(crate) struct SeamWeights {
    wbufs: Vec<Box<dyn Buffer>>,
    wspecs: Vec<(DType, usize)>,
    rf_buf: Option<(Box<dyn Buffer>, usize)>,
}

impl SeamKv {
    /// Longest common prefix of this slot's materialized tokens and `prompt` — the slot-selection
    /// score for multi-conversation serve.
    pub(crate) fn prefix_score(&self, prompt: &[u32]) -> usize {
        common_prefix_len(&self.cached, prompt)
    }

    /// Forget the materialized tokens WITHOUT dropping weights or buffers: the next call
    /// re-prefills from position 0 into the same session. Bench reps use this so each rep
    /// measures a full prefill while weights/pipelines/repack caches stay warm.
    /// (cfg-gated with its only caller, the Metal bench session — dead code on other targets.)
    #[cfg(target_os = "macos")]
    pub(crate) fn reset_tokens(&mut self) {
        self.cached.clear();
    }

    /// Number of token ids materialized in this slot's KV cache.
    pub(crate) fn cached_len(&self) -> usize {
        self.cached.len()
    }

    /// Forget the materialized tokens (the KV rows become dead; the next prompt prefills from
    /// row 0). Used to discard a warmup generation without dropping the slot's buffers.
    pub(crate) fn reset(&mut self) {
        self.cached.clear();
    }

    /// Fork a fresh conversation slot: same (Arc-shared) weights, its own zero KV + IO buffers.
    pub(crate) fn fork(&self, be: &dyn Backend, cfg: &Config) -> AResult<SeamKv> {
        let e2b = self.ipl_buf.is_some();
        let npl = cfg.n_embd_per_layer.max(1);
        let mut kbufs: Vec<Box<dyn Buffer>> = Vec::new();
        let mut vbufs: Vec<Box<dyn Buffer>> = Vec::new();
        for l in 0..cfg.n_layer {
            let kvrow_l = cfg.layer_n_kv(l) * cfg.layer_head_dim(l);
            kbufs.push(
                be.alloc(
                    kv_fmt_bytes(self.k_fmt, self.max_ctx * kvrow_l),
                    BufferUsage::Activations,
                )
                .map_err(|e| anyhow!("{e}"))?,
            );
            vbufs.push(
                be.alloc(
                    kv_fmt_bytes(self.v_fmt, self.max_ctx * kvrow_l),
                    BufferUsage::Activations,
                )
                .map_err(|e| anyhow!("{e}"))?,
            );
        }
        Ok(SeamKv {
            weights: std::sync::Arc::clone(&self.weights),
            kbufs,
            vbufs,
            k_fmt: self.k_fmt,
            v_fmt: self.v_fmt,
            hidden_buf: be
                .alloc(cfg.n_embd * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?,
            pos_buf: be
                .alloc(4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?,
            ipl_buf: if e2b {
                Some(
                    be.alloc(cfg.n_layer * npl * 4, BufferUsage::Staging)
                        .map_err(|e| anyhow!("{e}"))?,
                )
            } else {
                None
            },
            logits_buf: be
                .alloc(cfg.vocab * 4, BufferUsage::Readback)
                .map_err(|e| anyhow!("{e}"))?,
            max_ctx: self.max_ctx,
            cached: Vec::new(),
        })
    }

    /// Seed this slot's KV cache with the first `p` rows of `src`'s (the shared conversation
    /// prefix — e.g. the system prompt) via a device-side buffer copy, so the new conversation
    /// skips re-prefilling it. `p` must be ≤ src's materialized length.
    pub(crate) fn seed_from(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        src: &SeamKv,
        p: usize,
    ) -> AResult<()> {
        let p = p.min(src.cached.len()).min(self.max_ctx);
        if p == 0 {
            return Ok(());
        }
        for l in 0..cfg.n_layer {
            let elems = p * cfg.layer_n_kv(l) * cfg.layer_head_dim(l);
            be.copy_buffer(
                src.kbufs[l].as_ref(),
                self.kbufs[l].as_ref(),
                kv_fmt_bytes(self.k_fmt, elems),
            )
            .map_err(|e| anyhow!("{e}"))?;
            be.copy_buffer(
                src.vbufs[l].as_ref(),
                self.vbufs[l].as_ref(),
                kv_fmt_bytes(self.v_fmt, elems),
            )
            .map_err(|e| anyhow!("{e}"))?;
        }
        self.cached = src.cached[..p].to_vec();
        Ok(())
    }
}

/// gemma4 E2B: gather + dequant this chunk's per-layer TOKEN embedding rows on the host — the ONLY
/// part llama.cpp keeps host-side ("very little benefit to offloading the input layer"); the
/// model_proj GEMV + RMSNorm + combine now run as GPU graph ops (see the E2B prologue in `build`).
/// Returns `pl_tok_scaled[r][l*npl+j] = per_layer_tok_embd[tok_r][l*npl+j] * √npl`, `[rows,
/// n_layer*npl]` row-major — uploaded to `ipl_buf` and bound to the graph Input `pl_tok_in`.
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
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_dense_backend(
    be: &dyn Backend,
    bind_weight: &BindWeight,
    g: &Gguf,
    cfg: &Config,
    token_embd: &[f32],
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    mut on_token: impl FnMut(u32),
    state: &mut Option<SeamKv>,
    want_ctx: usize,
    mut constraint: Option<&mut crate::grammar::Constraint>,
    verify: Option<&mut Vec<f32>>,
) -> AResult<(Vec<u32>, GenStats)> {
    let c = cfg;
    let (ne, nh) = (c.n_embd, c.n_head);
    // gemma4: per-layer SWA/full dims differ; size shared scratch + KV by the max over layers.
    let max_hd = c.max_head_dim();
    let max_kvrow = c.max_n_kv() * max_hd;
    let max_qrow = nh * max_hd;
    let nff = c.n_ff; // max FFN width
    let gemma = c.gemma;
    let gemma4 = c.gemma4;
    let qk_norm = c.qk_norm;
    let act = if gemma {
        Activation::Gelu
    } else {
        Activation::Silu
    };
    // gemma4 E2B (gemma3n): per-layer input embeddings + KV-layer sharing.
    let e2b = c.n_embd_per_layer > 0;
    let npl = c.n_embd_per_layer;

    // Per-layer presence of an explicit V projection. gemma4 full-attention layers omit it (V = the
    // raw K projection); every layer of every other model has one.
    let has_wv: Vec<bool> = (0..c.n_layer)
        .map(|l| {
            g.tensors()
                .iter()
                .any(|t| t.name == format!("blk.{l}.attn_v.weight"))
        })
        .collect();
    // gemma4 per-layer output scale (`layer_output_scale.weight`, a single scalar multiplying the
    // layer output before the next layer). Read host-side; applied as an `Op::Scale`.
    let out_scale: Vec<Option<f32>> = (0..c.n_layer)
        .map(|l| {
            let name = format!("blk.{l}.layer_output_scale.weight");
            if g.tensors().iter().any(|t| t.name == name) {
                crate::load_tensor_dequant(g, &name)
                    .ok()
                    .and_then(|(v, _)| v.first().copied())
            } else {
                None
            }
        })
        .collect();
    // gemma4 proportional-RoPE frequency divisors (`rope_freqs.weight`, `[rope_dim/2]`): applied on
    // full-attention layers only (SWA layers use plain RoPE). Bound as a per-step f32 Input.
    let rope_freqs: Option<Vec<f32>> =
        if gemma4 && g.tensors().iter().any(|t| t.name == "rope_freqs.weight") {
            Some(crate::load_tensor_dequant(g, "rope_freqs.weight").map(|(v, _)| v)?)
        } else {
            None
        };
    // Combined gate+up FFN weights (one GEMV/GEMM + GatedActFused instead of two Linears +
    // GatedAct — the bespoke path's fused-gu shape, ~1 dispatch/layer off the decode hot loop).
    // Requires the backend to opt in (Vulkan; the CPU keeps zero-copy separate tensors) AND every
    // dense layer's gate/up to share a dtype (the concat is one [2*nff, ne] tensor). The decision
    // is global so the upload order and `build`'s handle declarations always agree.
    let fuse_gu = be.capabilities().combined_gu
        && c.moe.is_none()
        && (0..c.n_layer).all(|l| {
            let dt = |s: &str| {
                let name = format!("blk.{l}.{s}");
                g.tensors().iter().find(|t| t.name == name).map(|t| t.dtype)
            };
            dt("ffn_gate.weight").is_some() && dt("ffn_gate.weight") == dt("ffn_up.weight")
        });
    // Combined QKV: one [qrow+2·kvrow, ne] weight → prefill runs ONE wide GEMM (the separate
    // q/k/v GEMMs are narrow-n and underfill a big GPU — the pp512 sweep's dominant cost), and
    // decode keeps three offset GEMVs into the same buffer (`Op::Linear.w_off`), so its dispatch
    // count is unchanged. Needs every layer to own all three projections in ONE native-supported
    // dtype (gemma4's V-less full layers keep the split form), uniform dims (the offsets are
    // baked once), and a backend that opted into combined weights.
    let fuse_qkv = be.capabilities().combined_gu
        && (0..c.n_layer).all(|l| {
            let dt = |s: &str| {
                let name = format!("blk.{l}.{s}");
                g.tensors().iter().find(|t| t.name == name).map(|t| t.dtype)
            };
            let q = dt("attn_q.weight");
            q.is_some()
                && q == dt("attn_k.weight")
                && q == dt("attn_v.weight")
                && q.is_some_and(|d| {
                    infr_vulkan::linear::native_dense_supported(d) && d != DType::F16
                })
                && c.layer_head_dim(l) == c.head_dim
                && c.layer_n_kv(l) == c.n_kv
                && c.has_own_kv(l)
        });

    // KV cache dtype, chosen PER-SIDE (K and V independent, like llama's --cache-type-k /
    // --cache-type-v). Q8_0 stores 34 bytes / 32 elems — half the f16 footprint and bandwidth.
    //   INFR_KV_TYPE_K / INFR_KV_TYPE_V ∈ {f16, q8_0}  (per-side override)
    //   INFR_KV_Q8=1                                    legacy alias: any side not otherwise set → q8_0
    // Per-side KV dtype, chosen from INFR_KV_TYPE_K/V (llama's --cache-type-k/-v). The graph decl
    // carries the dtype and the env is stable for the process, so a warm session and its rebuilt
    // graphs always agree. Gates: Q8_0 needs each layer's KV row (n_kv*head_dim) 32-block-aligned and
    // a backend with the Q8 read/write (cpu/vulkan/metal). TurboQuant (turbo2/3/4) is WHT-rotated,
    // 128-elem blocks = head_dim slices — CPU-only, needs head_dim%128. The mainline low-bit quants
    // (q4_0/q4_1/q5_0/q5_1/iq4_nl) + f32/bf16 are CPU-only too (no GPU KV kernel yet); the block
    // quants need 32-alignment. All of these are footprint knobs (quantized KV is slower on CPU).
    let kv_align_ok =
        (0..c.n_layer).all(|l| (c.layer_n_kv(l) * c.layer_head_dim(l)).is_multiple_of(32));
    let kv_q8_backend = matches!(be.name(), "metal" | "cpu" | "vulkan");
    // TurboQuant (turbo2/3/4): CPU + Vulkan + Metal (both GPUs use a dequant→f16 prepass); needs
    // head_dim % 128 (a WHT group is a 128-elem head_dim slice).
    let kv_turbo_ok = matches!(be.name(), "cpu" | "vulkan" | "metal")
        && (0..c.n_layer).all(|l| c.layer_head_dim(l).is_multiple_of(128));
    // Mainline low-bit block quants (q4_0/…/iq4_nl): CPU + Vulkan + Metal (both GPUs dequant→f16
    // prepass); need 32-block alignment.
    let blk_ok = matches!(be.name(), "cpu" | "vulkan" | "metal") && kv_align_ok;
    // Dense f32/bf16 KV: CPU + Vulkan + Metal. Vulkan/Metal store dense; f32 reads natively (its
    // own f32 attention), bf16 reads via a cast→f16 prepass.
    let dense_ok = matches!(be.name(), "cpu" | "vulkan" | "metal");
    let parse_kv_fmt = |var: &str| -> DType {
        let side = std::env::var(var).ok();
        match side.as_deref() {
            Some("turbo2") if kv_turbo_ok => DType::Turbo2,
            Some("turbo3") if kv_turbo_ok => DType::Turbo3,
            Some("turbo4") if kv_turbo_ok => DType::Turbo4,
            Some("q8_0") | Some("q8") | Some("Q8_0") if kv_align_ok && kv_q8_backend => DType::Q8_0,
            Some("q4_0") if blk_ok => DType::Q4_0,
            Some("q4_1") if blk_ok => DType::Q4_1,
            Some("q5_0") if blk_ok => DType::Q5_0,
            Some("q5_1") if blk_ok => DType::Q5_1,
            Some("iq4_nl") if blk_ok => DType::Iq4Nl,
            Some("bf16") if dense_ok => DType::Bf16,
            Some("f32") if dense_ok => DType::F32,
            Some("f16") | Some("F16") => DType::F16,
            // unset/unknown → legacy INFR_KV_Q8 alias (both sides q8) or f16.
            _ if std::env::var("INFR_KV_Q8").is_ok() && kv_align_ok && kv_q8_backend => DType::Q8_0,
            _ => DType::F16,
        }
    };
    let mut k_fmt = parse_kv_fmt("INFR_KV_TYPE_K");
    let mut v_fmt = parse_kv_fmt("INFR_KV_TYPE_V");
    // Metal's Q8 and F32 KV use native-read attention that reads BOTH sides as one dtype, so a
    // mixed request with q8/f32 on one side would misread the other — clamp those to coupled f16.
    // The prepass formats (block quants / bf16 / turbo) expand each side to its own f16 scratch, so
    // they compose freely with each other and with a native-f16 side (per-side, like Vulkan/CPU).
    if be.name() == "metal" && k_fmt != v_fmt {
        let native_read = |dt| matches!(dt, DType::Q8_0 | DType::F32);
        if native_read(k_fmt) || native_read(v_fmt) {
            k_fmt = DType::F16;
            v_fmt = DType::F16;
        }
    }

    // ── one-time session init: weights, KV cache, per-step IO (skipped when `state` is warm) ──
    if state.is_none() {
        // Weight-load progress: opened HERE — the single weight-upload funnel every runner path
        // (CPU/Vulkan/Metal × one-shot/session/bench/serve) goes through — so no entry point can
        // load without it. The ticking lives in each backend's `alloc` (Weights/HostWeights);
        // backends without a display return a no-op scope. Guard drops when the init block ends.
        let fp = crate::weights::weight_footprint(g);
        let _weight_pb = be.weight_progress(Some(fp.dense + fp.expert));
        // ── upload weights in their NATIVE GGUF dtype (no host pre-dequant — the backend dequants
        //    lazily in `bytes_to_f32`, so a quant weight occupies ~quant size, not 8× f32). `wspecs`
        //    records each (dtype, numel) so `build` can declare the handle with the matching dtype; its
        //    order MUST equal the `g.weight()` order in `build` below. ──────────────────────────────────
        let mut wbufs: Vec<Box<dyn Buffer>> = Vec::new();
        let mut wspecs: Vec<(DType, usize)> = Vec::new();
        // Load one weight (zero-copy mmap slice — no alloc, no memcpy) or CONCATENATE several into
        // one owned buffer (the combined gate+up upload; same dtype, row-major concat of [nff, ne]
        // tensors = a valid [k*nff, ne] tensor). Records the native dtype + element count so
        // `build` declares the handle to match.
        let mut wload = |names: &[&str]| -> AResult<()> {
            let info = |name: &str| {
                g.tensors()
                    .iter()
                    .find(|t| t.name == name)
                    .cloned()
                    .ok_or_else(|| anyhow!("tensor not found: {name}"))
            };
            let (bytes, dt, numel) = if let [name] = names {
                let i = info(name)?;
                let tb = g.tensor_bytes_arc(name).map_err(|e| anyhow!("{e}"))?;
                (WBytes::Mmap(tb), i.dtype, i.shape.iter().product())
            } else {
                let mut cat = Vec::new();
                let mut numel = 0usize;
                let dt = info(names[0])?.dtype;
                for name in names {
                    let i = info(name)?;
                    if i.dtype != dt {
                        return Err(anyhow!("wload concat dtype mismatch: {names:?}"));
                    }
                    numel += i.shape.iter().product::<usize>();
                    cat.extend_from_slice(&g.tensor_bytes_arc(name).map_err(|e| anyhow!("{e}"))?);
                }
                (WBytes::Owned(cat), dt, numel)
            };
            // bind_weight returns the EFFECTIVE dtype the buffer holds (the GPU binder may convert float
            // weights to f16), so the graph declares the handle to match what the backend will read.
            let (buf, eff_dt) = bind_weight(names[0], bytes, dt, numel)?;
            wbufs.push(buf);
            wspecs.push((eff_dt, numel));
            Ok(())
        };
        for l in 0..c.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            wload(&[&p("attn_norm.weight")])?;
            if fuse_qkv {
                wload(&[
                    &p("attn_q.weight"),
                    &p("attn_k.weight"),
                    &p("attn_v.weight"),
                ])?;
            } else {
                wload(&[&p("attn_q.weight")])?;
                wload(&[&p("attn_k.weight")])?;
                if has_wv[l] {
                    wload(&[&p("attn_v.weight")])?;
                }
            }
            if qk_norm {
                wload(&[&p("attn_q_norm.weight")])?;
                wload(&[&p("attn_k_norm.weight")])?;
            }
            wload(&[&p("attn_output.weight")])?;
            if gemma {
                wload(&[&p("post_attention_norm.weight")])?;
            }
            wload(&[&p("ffn_norm.weight")])?;
            if c.moe.is_some() {
                // qwen3moe: router + stacked per-expert gate/up/down banks.
                wload(&[&p("ffn_gate_inp.weight")])?;
                wload(&[&p("ffn_gate_exps.weight")])?;
                wload(&[&p("ffn_up_exps.weight")])?;
                wload(&[&p("ffn_down_exps.weight")])?;
            } else if fuse_gu {
                wload(&[&p("ffn_gate.weight"), &p("ffn_up.weight")])?;
                wload(&[&p("ffn_down.weight")])?;
            } else {
                wload(&[&p("ffn_gate.weight")])?;
                wload(&[&p("ffn_up.weight")])?;
                wload(&[&p("ffn_down.weight")])?;
            }
            if gemma {
                wload(&[&p("post_ffw_norm.weight")])?;
            }
            if e2b {
                // gemma4 E2B per-layer input-embedding application weights.
                wload(&[&p("inp_gate.weight")])?;
                wload(&[&p("proj.weight")])?;
                wload(&[&p("post_norm.weight")])?;
            }
        }
        // Globals: output_norm, lm_head. lm_head = `output.weight`, or (tied) the quantized
        // `token_embd.weight` mapped from the mmap and dequantized per-row by `Op::Linear` — same f32
        // values as the host `token_embd`, but zero-copy.
        wload(&["output_norm.weight"])?;
        if g.tensors().iter().any(|t| t.name == "output.weight") {
            wload(&["output.weight"])?;
        } else {
            wload(&["token_embd.weight"])?;
        }
        // gemma4 E2B: the per-layer input-embedding projection weights, native-uploaded like any
        // other weight (model_proj stays bf16 — the seam's native bf16 GEMV/GEMM reads it directly;
        // proj_norm is f32). The GPU graph prologue (in `build`, below) runs the GEMV + RMSNorm that
        // used to be a host loop. Declared here (in upload order) — `build` pushes the matching
        // handles right after `w_lm`/before `v_ones`.
        if e2b {
            wload(&["per_layer_model_proj.weight"])?;
            wload(&["per_layer_proj_norm.weight"])?;
            // Sanity-check the two uploads landed with the shapes the GPU prologue assumes
            // (model_proj is `[n_layer*npl, n_embd]`, proj_norm is `[npl]`).
            debug_assert_eq!(wspecs[wspecs.len() - 2].1, c.n_layer * npl * ne);
            debug_assert_eq!(wspecs[wspecs.len() - 1].1, npl);
        }
        // gemma4 weightless per-head V-norm = `QkNorm` with a unit weight (out = x/rms). One ones-vector
        // of the max head dim serves every layer (a narrower layer reads its leading prefix).
        if gemma4 {
            let ones = vec![1.0f32; max_hd];
            let b = be
                .alloc(ones.len() * 4, BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(b.as_ref(), bytemuck::cast_slice(&ones))
                .map_err(|e| anyhow!("{e}"))?;
            wbufs.push(b);
            wspecs.push((DType::F32, max_hd));
        }

        // ── persistent KV cache buffers, sized per-layer (gemma4 SWA layers are narrower) and
        //    per-side (K and V pick their dtype independently) ────────────────────────────────
        let mut kbufs: Vec<Box<dyn Buffer>> = Vec::new();
        let mut vbufs: Vec<Box<dyn Buffer>> = Vec::new();
        for l in 0..c.n_layer {
            let kvrow_l = c.layer_n_kv(l) * c.layer_head_dim(l);
            kbufs.push(
                be.alloc(
                    kv_fmt_bytes(k_fmt, want_ctx * kvrow_l),
                    BufferUsage::Activations,
                )
                .map_err(|e| anyhow!("{e}"))?,
            );
            vbufs.push(
                be.alloc(
                    kv_fmt_bytes(v_fmt, want_ctx * kvrow_l),
                    BufferUsage::Activations,
                )
                .map_err(|e| anyhow!("{e}"))?,
            );
        }

        // ── per-step IO buffers ────────────────────────────────────────────────────────
        let hidden_buf = be
            .alloc(ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let pos_buf = be
            .alloc(4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let rf_buf = match &rope_freqs {
            Some(rf) => {
                let b = be
                    .alloc(rf.len() * 4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?;
                be.upload(b.as_ref(), bytemuck::cast_slice(rf))
                    .map_err(|e| anyhow!("{e}"))?;
                Some((b, rf.len()))
            }
            None => None,
        };
        // gemma4 E2B per-(token,layer) input vector `[n_layer*npl]`, recomputed + re-uploaded each step.
        let ipl_buf = if e2b {
            Some(
                be.alloc(c.n_layer * npl * 4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            None
        };
        let logits_buf = be
            .alloc(c.vocab * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        *state = Some(SeamKv {
            weights: std::sync::Arc::new(SeamWeights {
                wbufs,
                wspecs,
                rf_buf,
            }),
            kbufs,
            vbufs,
            k_fmt,
            v_fmt,
            hidden_buf,
            pos_buf,
            ipl_buf,
            logits_buf,
            max_ctx: want_ctx,
            cached: Vec::new(),
        });
    }
    let SeamKv {
        weights,
        kbufs,
        vbufs,
        k_fmt: _,
        v_fmt: _,
        hidden_buf,
        pos_buf,
        ipl_buf,
        logits_buf,
        max_ctx,
        cached,
    } = state.as_mut().expect("seam state just initialized");
    let SeamWeights {
        wbufs,
        wspecs,
        rf_buf,
    } = weights.as_ref();
    let max_ctx = *max_ctx;
    if prompt.len() + max_new + 1 > max_ctx {
        return Err(anyhow!(
            "prompt {} + gen {} exceeds the session KV capacity {max_ctx}",
            prompt.len(),
            max_new
        ));
    }
    // ChatSession-style prefix reuse: KV rows 0..start are already materialized for `cached`'s
    // shared prefix — prefill only the suffix. Always leave ≥1 prompt token to process so the
    // first generated token samples from fresh logits.
    let start = common_prefix_len(cached, prompt).min(prompt.len() - 1);
    cached.truncate(start);

    // Build a forward graph for `batch` tokens starting at absolute position `start_pos`.
    // `batch = 1` is the normal decode path; `batch > 1` is the batched-prefill path.
    // Scratch tensors scale by `batch`; the LM head runs on the last `logits_rows` tokens —
    // 1 everywhere except speculative VERIFY, which needs the distribution after every
    // candidate (logits output = [logits_rows, vocab], logits_rows ∈ {1, batch}).
    let build = |batch: usize, start_pos: usize, logits_rows: usize| -> (Graph, DecodeHandles) {
        let mut g = Graph::new();
        let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
        // KV cache dtype: f16 by default (halves memory vs f32, tightens CPU↔GPU parity); Q8_0
        // per-side when the runner enabled it (see `k_fmt`/`v_fmt` at the cache alloc). ONLY the
        // persistent caches take this dtype — the roped q16/k16 staging stays f16
        // (`qk_norm_rope`/`rope_f16` write f16; a Q8_0 decl there would lie to any backend that
        // trusts it, and the Vulkan kv-write peephole fuses on the f16 decl).
        let kd = |n: usize| TensorDesc::new(vec![n], k_fmt);
        let vd = |n: usize| TensorDesc::new(vec![n], v_fmt);
        let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16);
        let hidden = g.input(f32d(batch * ne));
        let positions = g.input(TensorDesc::new(vec![batch], DType::I32));
        let rope_freqs = rf_buf.as_ref().map(|(_, n)| g.input(f32d(*n)));
        // gemma4 E2B per-(token,layer) TOKEN embedding rows `[batch, n_layer*npl]` — host-gathered
        // + dequanted (the big `per_layer_token_embd` table stays off-VRAM, gathered per token).
        // The full `per_layer_inp` consumed by the layer loop is computed from this on the GPU
        // (model_proj GEMV + RMSNorm), further down, once its weights are declared.
        let pl_tok_in = if e2b {
            Some(g.input(f32d(batch * c.n_layer * npl)))
        } else {
            None
        };
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        for l in 0..c.n_layer {
            let kvrow_l = c.layer_n_kv(l) * c.layer_head_dim(l);
            k_cache.push(g.input(kd(max_ctx * kvrow_l)));
            v_cache.push(g.input(vd(max_ctx * kvrow_l)));
        }

        // Weights — declared in the SAME order as the upload loop, pulling (dtype, numel) from
        // `wspecs` so each handle carries its native GGUF dtype (the backend dequants on read).
        // `wpush` records the handle in the flat `weights` list (for binding) and returns it.
        let mut weights: Vec<TensorId> = Vec::new();
        let mut wi = 0usize;
        let mut wpush = |g: &mut Graph, weights: &mut Vec<TensorId>| -> TensorId {
            let (dt, n) = wspecs[wi];
            wi += 1;
            let id = g.weight(TensorDesc::new(vec![n], dt));
            weights.push(id);
            id
        };
        let mut lw: Vec<LayerW> = Vec::new();
        for l in 0..c.n_layer {
            let attn_norm = wpush(&mut g, &mut weights);
            // Fused QKV: ONE concatenated weight handle serves q/k/v (the builder bakes each
            // projection's `w_off` slice); split form declares three.
            let (wq, wk, wv) = if fuse_qkv {
                let wqkv = wpush(&mut g, &mut weights);
                (wqkv, wqkv, Some(wqkv))
            } else {
                let wq = wpush(&mut g, &mut weights);
                let wk = wpush(&mut g, &mut weights);
                let wv = if has_wv[l] {
                    Some(wpush(&mut g, &mut weights))
                } else {
                    None
                };
                (wq, wk, wv)
            };
            let (q_norm, k_norm) = if qk_norm {
                (
                    Some(wpush(&mut g, &mut weights)),
                    Some(wpush(&mut g, &mut weights)),
                )
            } else {
                (None, None)
            };
            let wo = wpush(&mut g, &mut weights);
            let post_attn = if gemma {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let ffn_norm = wpush(&mut g, &mut weights);
            let ffn = if c.moe.is_some() {
                FfnW::Moe {
                    router: wpush(&mut g, &mut weights),
                    gate_exps: wpush(&mut g, &mut weights),
                    up_exps: wpush(&mut g, &mut weights),
                    down_exps: wpush(&mut g, &mut weights),
                }
            } else if fuse_gu {
                FfnW::DenseFused {
                    wgu: wpush(&mut g, &mut weights),
                    wdown: wpush(&mut g, &mut weights),
                }
            } else {
                FfnW::Dense {
                    wgate: wpush(&mut g, &mut weights),
                    wup: wpush(&mut g, &mut weights),
                    wdown: wpush(&mut g, &mut weights),
                }
            };
            let post_ffw = if gemma {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let (pl_inp_gate, pl_proj, pl_post_norm) = if e2b {
                (
                    Some(wpush(&mut g, &mut weights)),
                    Some(wpush(&mut g, &mut weights)),
                    Some(wpush(&mut g, &mut weights)),
                )
            } else {
                (None, None, None)
            };
            lw.push(LayerW {
                attn_norm,
                wq,
                wk,
                wv,
                q_norm,
                k_norm,
                wo,
                post_attn,
                ffn_norm,
                ffn,
                post_ffw,
                pl_inp_gate,
                pl_proj,
                pl_post_norm,
            });
        }
        let w_out_norm = wpush(&mut g, &mut weights);
        let w_lm = wpush(&mut g, &mut weights);
        // gemma4 E2B per-layer input-embedding projection weights — declared here to match the
        // `wload` upload order (right after lm_head, before the gemma4 V-norm ones-vector).
        let (mp_w, pn_w) = if e2b {
            (
                Some(wpush(&mut g, &mut weights)),
                Some(wpush(&mut g, &mut weights)),
            )
        } else {
            (None, None)
        };
        let v_ones = if gemma4 {
            Some(wpush(&mut g, &mut weights))
        } else {
            None
        };
        let logits = g.output(f32d(c.vocab * logits_rows));

        // scratch (sized to the per-layer max × batch; ops reallocate dst, so these are upper bounds)
        let hn = g.internal(f32d(batch * ne));
        let q = g.internal(f32d(batch * max_qrow));
        let k = g.internal(f32d(batch * max_kvrow));
        let v = g.internal(f32d(batch * max_kvrow));
        // QkNorm+RoPE writes f16 (the GPU `qk_norm_rope` is f32-in→f16-out, can't be in place; the GPU
        // attention reads f16 q). q16/k16 hold the f16 normed+roped q/k for the q/k-norm (qwen3/gemma)
        // path; the llama RoPE-only path stays in f32 q/k. Free on the CPU (its store is f32 regardless).
        let q16 = g.internal(f16d(batch * max_qrow));
        let k16 = g.internal(f16d(batch * max_kvrow));
        let attn = g.internal(f32d(batch * max_qrow));
        // Fused-QKV prefill staging: the wide GEMM writes [batch, qrow+2·kvrow] here, then three
        // CopyStrided ops split it into q/k/v. Decode (batch==1) skips it (offset GEMVs).
        let qkvbuf = if fuse_qkv && batch > 1 {
            Some(g.internal(f32d(batch * (max_qrow + 2 * max_kvrow))))
        } else {
            None
        };
        // Separate gate/up scratch, or one combined [batch, 2*nff] gu buffer when fused — declare
        // only the shape in use (Internal buffers are allocated by the backend even if never read).
        let (gbuf, ubuf, gubuf) = if fuse_gu {
            let gu = g.internal(f32d(batch * 2 * nff));
            (gu, gu, gu)
        } else {
            let gb = g.internal(f32d(batch * nff));
            let ub = g.internal(f32d(batch * nff));
            (gb, ub, gb)
        };
        let actbuf = g.internal(f32d(batch * nff));
        let sub = g.internal(f32d(batch * ne));
        // E2B per-layer embed scratch: gate `[npl]` and projected `[ne]`.
        let plg = g.internal(f32d(batch * npl.max(1)));
        let plp = g.internal(f32d(batch * ne));

        let eps = c.rms_eps;

        // gemma4 E2B prologue: compute the full per-(token,layer) input vector `per_layer_inp`
        // ([batch, n_layer*npl]) that the layer loop below consumes, on the GPU — matches
        // llama.cpp's split (host: gather+dequant the per-layer token embedding row; GPU: the
        // model_proj GEMV + RMSNorm + combine). `hidden` here is already the scaled token
        // embedding (`emb = token_embd[tok] * embed_scale`), so it IS the `emb` the host version
        // used to dot against `model_proj`.
        let per_layer_inp =
            if let (Some(mp_w), Some(pn_w), Some(pl_tok_in)) = (mp_w, pn_w, pl_tok_in) {
                let nlnpl = c.n_layer * npl;
                let acc = g.internal(f32d(batch * nlnpl));
                g.push(Op::Linear {
                    x: hidden,
                    weight: mp_w,
                    dst: acc,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: nlnpl as u32,
                    w_off: 0,
                });
                g.push(Op::Scale {
                    x: acc,
                    dst: acc,
                    s: 1.0 / (ne as f32).sqrt(),
                    n: (batch * nlnpl) as u32,
                });
                let normed = g.internal(f32d(batch * nlnpl));
                g.push(Op::RmsNorm {
                    x: acc,
                    weight: pn_w,
                    dst: normed,
                    rows: (batch * c.n_layer) as u32,
                    dim: npl as u32,
                    eps,
                });
                let ipl = g.internal(f32d(batch * nlnpl));
                g.push(Op::Add {
                    a: normed,
                    b: pl_tok_in,
                    dst: ipl,
                    n: (batch * nlnpl) as u32,
                });
                g.push(Op::Scale {
                    x: ipl,
                    dst: ipl,
                    s: 1.0 / 2f32.sqrt(),
                    n: (batch * nlnpl) as u32,
                });
                Some(ipl)
            } else {
                None
            };

        for (l, lw) in lw.iter().enumerate() {
            // Per-layer dims (gemma4 SWA vs full; uniform for every other model).
            let hd = c.layer_head_dim(l);
            let nkv = c.layer_n_kv(l);
            let kvrow = nkv * hd;
            let qrow = nh * hd;
            let nff_l = c.layer_n_ff(l);
            let theta = c.layer_rope_theta(l); // gemma dual-rope (SWA 1e4 / full 1e6); uniform else
            let rope_dim = c.layer_rope_dim(l);
            let swa = gemma && c.is_swa_layer(l);
            let mask = if swa {
                AttnMask::SlidingWindow(c.swa_window)
            } else {
                AttnMask::Causal
            };
            // gemma4: attn scale 1.0 (QK-norm controls magnitude); everyone else 1/√hd.
            let scale = if gemma4 {
                1.0
            } else {
                1.0 / (hd as f32).sqrt()
            };
            // gemma4 proportional-RoPE applies only on full-attention layers.
            let layer_ff = if gemma4 && !swa { rope_freqs } else { None };
            // attn input norm
            g.push(Op::RmsNorm {
                x: hidden,
                weight: lw.attn_norm,
                dst: hn,
                rows: batch as u32,
                dim: ne as u32,
                eps,
            });
            // gemma4 E2B KV-layer sharing: shared layers compute Q only and attend to an earlier
            // layer's cache. `own_kv`/`kv_src` are `true`/`l` for every layer of a non-sharing model.
            let own_kv = c.has_own_kv(l);
            let kv_src = c.kv_src_layer(l);
            if let Some(qkv) = qkvbuf {
                // Fused QKV (prefill): ONE wide GEMM over the concatenated weight — the separate
                // q/k/v GEMMs are narrow-n and underfill the GPU — then split rows into q/k/v.
                let stride = (qrow + 2 * kvrow) as u32;
                g.push(Op::Linear {
                    x: hn,
                    weight: lw.wq,
                    dst: qkv,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: stride,
                    w_off: 0,
                });
                for (dst, off, n) in [
                    (q, 0u32, qrow as u32),
                    (k, qrow as u32, kvrow as u32),
                    (v, (qrow + kvrow) as u32, kvrow as u32),
                ] {
                    g.push(Op::CopyStrided {
                        src: qkv,
                        src_off: off,
                        src_stride: stride,
                        dst,
                        dst_off: 0,
                        dst_stride: n,
                        rows: batch as u32,
                        n,
                    });
                }
            } else if fuse_qkv {
                // Fused QKV (decode): three offset GEMVs into the concatenated weight — the same
                // dispatch count as the split form, no staging copies.
                for (dst, off, n) in [
                    (q, 0usize, qrow),
                    (k, qrow * ne, kvrow),
                    (v, (qrow + kvrow) * ne, kvrow),
                ] {
                    g.push(Op::Linear {
                        x: hn,
                        weight: lw.wq,
                        dst,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: n as u32,
                        w_off: off as u32,
                    });
                }
            } else {
                g.push(Op::Linear {
                    x: hn,
                    weight: lw.wq,
                    dst: q,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: qrow as u32,
                    w_off: 0,
                });
            }
            if own_kv {
                if !fuse_qkv {
                    g.push(Op::Linear {
                        x: hn,
                        weight: lw.wk,
                        dst: k,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: kvrow as u32,
                        w_off: 0,
                    });
                    // V projection, or (gemma4 full layers) V = the raw K projection, copied BEFORE
                    // K is QK-normed + RoPE'd.
                    match lw.wv {
                        Some(wv) => g.push(Op::Linear {
                            x: hn,
                            weight: wv,
                            dst: v,
                            m: batch as u32,
                            in_f: ne as u32,
                            out_f: kvrow as u32,
                            w_off: 0,
                        }),
                        None => g.push(Op::Copy {
                            src: k,
                            src_off: 0,
                            dst: v,
                            dst_off: 0,
                            n: (batch * kvrow) as u32,
                        }),
                    }
                }
                // gemma4 weightless per-head RMSNorm on V (= x/rms) before caching. Emitted BEFORE
                // the K QkNormRope so that op stays ADJACENT to its WriteKv — the Vulkan adapter's
                // kv_write_peephole only fuses an immediately-following pair, and the record-once
                // decode path REQUIRES the K write fused (a standalone f16 WriteKv has no dyn
                // kernel). V only depends on the raw K projection, so the order is free.
                if let Some(ones) = v_ones {
                    g.push(Op::QkNorm {
                        x: v,
                        weight: ones,
                        dst: v,
                        rows: batch as u32,
                        n_head: nkv as u32,
                        head_dim: hd as u32,
                        eps,
                    });
                }
                // K: fused QkNorm+RoPE (qwen3/gemma) → f16 `k16`, else RoPE alone (llama) in-place f32.
                let k_write = match lw.k_norm {
                    Some(kn) => {
                        g.push(Op::QkNormRope {
                            x: k,
                            weight: kn,
                            positions,
                            dst: k16,
                            rows: batch as u32,
                            n_head: nkv as u32,
                            head_dim: hd as u32,
                            rope_dim: rope_dim as u32,
                            theta,
                            eps,
                            freq_factors: layer_ff,
                        });
                        k16
                    }
                    None => {
                        // llama (no k-norm): interleaved RoPE straight to the f16 scratch — the
                        // same fused shape as the qk-norm path, so the Vulkan peephole redirects
                        // the write into the KV cache and the decode replays via rope_f16_dyn.
                        g.push(Op::Rope {
                            x: k,
                            positions,
                            dst: k16,
                            rows: batch as u32,
                            n_head: nkv as u32,
                            head_dim: hd as u32,
                            rope_dim: rope_dim as u32,
                            theta,
                            freq_factors: layer_ff,
                        });
                        k16
                    }
                };
                g.push(Op::WriteKv {
                    src: k_write,
                    cache: k_cache[l],
                    rows: batch as u32,
                    row_stride: kvrow as u32,
                    pos: start_pos as u32,
                });
                g.push(Op::WriteKv {
                    src: v,
                    cache: v_cache[l],
                    rows: batch as u32,
                    row_stride: kvrow as u32,
                    pos: start_pos as u32,
                });
            }
            // Q: fused QkNorm+RoPE (qwen3/gemma) → f16 `q16`, else RoPE alone (llama) in-place f32.
            let q_attn = match lw.q_norm {
                Some(qn) => {
                    g.push(Op::QkNormRope {
                        x: q,
                        weight: qn,
                        positions,
                        dst: q16,
                        rows: batch as u32,
                        n_head: nh as u32,
                        head_dim: hd as u32,
                        rope_dim: rope_dim as u32,
                        theta,
                        eps,
                        freq_factors: layer_ff,
                    });
                    q16
                }
                None => {
                    // llama: Q roped to the f16 scratch (the attention kernels read f16 q).
                    g.push(Op::Rope {
                        x: q,
                        positions,
                        dst: q16,
                        rows: batch as u32,
                        n_head: nh as u32,
                        head_dim: hd as u32,
                        rope_dim: rope_dim as u32,
                        theta,
                        freq_factors: layer_ff,
                    });
                    q16
                }
            };
            g.push(Op::Attention {
                q: q_attn,
                k_cache: k_cache[kv_src],
                v_cache: v_cache[kv_src],
                dst: attn,
                rows: batch as u32,
                kv_len: (start_pos + batch) as u32,
                n_head: nh as u32,
                n_kv: nkv as u32,
                head_dim: hd as u32,
                scale,
                mask,
                pos: start_pos as u32,
            });
            g.push(Op::Linear {
                x: attn,
                weight: lw.wo,
                dst: sub,
                m: batch as u32,
                in_f: qrow as u32,
                out_f: ne as u32,
                w_off: 0,
            });
            // gemma sandwich: post-attention norm on the sublayer output BEFORE the residual add.
            if let Some(pa) = lw.post_attn {
                g.push(Op::RmsNorm {
                    x: sub,
                    weight: pa,
                    dst: sub,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
            }
            g.push(Op::Add {
                a: hidden,
                b: sub,
                dst: hidden,
                n: (batch * ne) as u32,
            });
            // ffn
            g.push(Op::RmsNorm {
                x: hidden,
                weight: lw.ffn_norm,
                dst: hn,
                rows: batch as u32,
                dim: ne as u32,
                eps,
            });
            match lw.ffn {
                FfnW::DenseFused { wgu, wdown } => {
                    g.push(Op::Linear {
                        x: hn,
                        weight: wgu,
                        dst: gubuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: (2 * nff_l) as u32,
                        w_off: 0,
                    });
                    g.push(Op::GatedActFused {
                        gu: gubuf,
                        dst: actbuf,
                        rows: batch as u32,
                        nff: nff_l as u32,
                        act,
                    });
                    g.push(Op::Linear {
                        x: actbuf,
                        weight: wdown,
                        dst: sub,
                        m: batch as u32,
                        in_f: nff_l as u32,
                        out_f: ne as u32,
                        w_off: 0,
                    });
                }
                FfnW::Dense { wgate, wup, wdown } => {
                    g.push(Op::Linear {
                        x: hn,
                        weight: wgate,
                        dst: gbuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                        w_off: 0,
                    });
                    g.push(Op::Linear {
                        x: hn,
                        weight: wup,
                        dst: ubuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                        w_off: 0,
                    });
                    g.push(Op::GatedAct {
                        gate: gbuf,
                        up: ubuf,
                        dst: actbuf,
                        rows: batch as u32,
                        nff: nff_l as u32,
                        act,
                        up_off: 0,
                    });
                    g.push(Op::Linear {
                        x: actbuf,
                        weight: wdown,
                        dst: sub,
                        m: batch as u32,
                        in_f: nff_l as u32,
                        out_f: ne as u32,
                        w_off: 0,
                    });
                }
                FfnW::Moe {
                    router,
                    gate_exps,
                    up_exps,
                    down_exps,
                } => {
                    let mc = c.moe.expect("moe layer without MoeConfig");
                    g.push(Op::MoeFfn {
                        x: hn,
                        router,
                        gate_exps,
                        up_exps,
                        down_exps,
                        dst: sub,
                        ne: ne as u32,
                        n_expert: mc.n_expert as u32,
                        n_used: mc.n_used as u32,
                        n_ff_exp: mc.n_ff_exp as u32,
                        scale: mc.scale,
                        act, // qwen3moe: SwiGLU (act == Silu)
                    });
                }
            }
            if let Some(pf) = lw.post_ffw {
                g.push(Op::RmsNorm {
                    x: sub,
                    weight: pf,
                    dst: sub,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
            }
            g.push(Op::Add {
                a: hidden,
                b: sub,
                dst: hidden,
                n: (batch * ne) as u32,
            });
            // gemma4 E2B per-layer input embedding (gemma3n): mix this layer's input vector into
            // `hidden` after the FFN residual. `g = gelu(inp_gate·hidden) * inp_per_layer[l]`,
            // `p = post_norm(proj·g)`, `hidden += p`.
            if let (Some(gate_w), Some(proj_w), Some(post_norm), Some(ipl)) =
                (lw.pl_inp_gate, lw.pl_proj, lw.pl_post_norm, per_layer_inp)
            {
                g.push(Op::Linear {
                    x: hidden,
                    weight: gate_w,
                    dst: plg,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: npl as u32,
                    w_off: 0,
                });
                // gelu(plg) * ipl[r, l*npl .. l*npl+npl] — gather each row's layer-l slice of the
                // [batch, n_layer*npl] input into a contiguous [batch, npl] scratch (one strided-
                // copy dispatch), then the plain gated activation. Keeps GatedAct's semantics
                // unchanged (its up_off has no per-row stride).
                let ipl_l = g.internal(f32d(batch * npl));
                g.push(Op::CopyStrided {
                    src: ipl,
                    src_off: (l * npl) as u32,
                    src_stride: (c.n_layer * npl) as u32,
                    dst: ipl_l,
                    dst_off: 0,
                    dst_stride: npl as u32,
                    rows: batch as u32,
                    n: npl as u32,
                });
                g.push(Op::GatedAct {
                    gate: plg,
                    up: ipl_l,
                    dst: plg,
                    rows: batch as u32,
                    nff: npl as u32,
                    act: Activation::Gelu,
                    up_off: 0,
                });
                g.push(Op::Linear {
                    x: plg,
                    weight: proj_w,
                    dst: plp,
                    m: batch as u32,
                    in_f: npl as u32,
                    out_f: ne as u32,
                    w_off: 0,
                });
                g.push(Op::RmsNorm {
                    x: plp,
                    weight: post_norm,
                    dst: plp,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
                g.push(Op::Add {
                    a: hidden,
                    b: plp,
                    dst: hidden,
                    n: (batch * ne) as u32,
                });
            }
            // gemma4: scale the whole layer output by the per-layer scalar before the next layer.
            if let Some(s) = out_scale[l] {
                g.push(Op::Scale {
                    x: hidden,
                    dst: hidden,
                    s,
                    n: (batch * ne) as u32,
                });
            }
        }
        g.push(Op::RmsNorm {
            x: hidden,
            weight: w_out_norm,
            dst: hn,
            rows: batch as u32,
            dim: ne as u32,
            eps,
        });
        // For batch > 1 with logits_rows == 1: the LM head runs only on the LAST token's
        // hidden state — extract it via Op::Copy before the projection so the logits output is
        // [vocab]. Speculative verify passes logits_rows == batch and runs the head over every
        // row instead (no Copy).
        let lm_in = if batch > 1 && logits_rows == 1 {
            let hn_last = g.internal(f32d(ne));
            g.push(Op::Copy {
                src: hn,
                src_off: ((batch - 1) * ne) as u32,
                dst: hn_last,
                dst_off: 0,
                n: ne as u32,
            });
            hn_last
        } else {
            hn
        };
        g.push(Op::Linear {
            x: lm_in,
            weight: w_lm,
            dst: logits,
            m: logits_rows as u32,
            in_f: ne as u32,
            out_f: c.vocab as u32,
            w_off: 0,
        });
        if c.final_softcap > 0.0 {
            g.push(Op::Softcap {
                x: logits,
                dst: logits,
                cap: c.final_softcap,
                n: (c.vocab * logits_rows) as u32,
            });
        }
        (
            g,
            DecodeHandles {
                hidden,
                positions,
                rope_freqs,
                pl_tok_in,
                logits,
                k_cache,
                v_cache,
                weights,
            },
        )
    };

    // ── speculative VERIFY ──────────────────────────────────────────────────────────
    // One batched forward over the un-cached suffix with the LM head on EVERY row: returns
    // [m, vocab] logits (the distribution after each suffix token) and generates nothing.
    // The suffix-prefill contract doubles as the accept/rollback mechanism: the caller
    // truncates its committed token list and the next call's prefix diff overwrites the
    // stale KV rows. Dense non-E2B models only (mirrors the batched-prefill guard).
    if let Some(out_logits) = verify {
        if c.moe.is_some() || ple.is_some() {
            return Err(anyhow!("speculative verify: dense non-E2B models only"));
        }
        let vf_scale = if gemma { (ne as f32).sqrt() } else { 1.0 };
        let m = prompt.len() - start;
        let mut vf_hidden: Vec<f32> = Vec::with_capacity(m * ne);
        for &tok in &prompt[start..] {
            let base = tok as usize * ne;
            vf_hidden.extend(token_embd[base..base + ne].iter().map(|&x| x * vf_scale));
        }
        let vf_positions: Vec<i32> = (start as i32..(start + m) as i32).collect();
        let vf_hidden_buf = be
            .alloc(m * ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let vf_pos_buf = be
            .alloc(m * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let vf_logits_buf = be
            .alloc(m * c.vocab * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(vf_hidden_buf.as_ref(), bytemuck::cast_slice(&vf_hidden))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(vf_pos_buf.as_ref(), bytemuck::cast_slice(&vf_positions))
            .map_err(|e| anyhow!("{e}"))?;
        let (vg, vh) = build(m, start, m);
        let vplan = be.compile(&vg).map_err(|e| anyhow!("{e}"))?;
        let mut vb = Bindings::new();
        vb.bind(vh.hidden, vf_hidden_buf.as_ref());
        vb.bind(vh.positions, vf_pos_buf.as_ref());
        if let (Some(rid), Some((rb, _))) = (vh.rope_freqs, &rf_buf) {
            vb.bind(rid, rb.as_ref());
        }
        for l in 0..c.n_layer {
            vb.bind(vh.k_cache[l], kbufs[l].as_ref());
            vb.bind(vh.v_cache[l], vbufs[l].as_ref());
        }
        for (i, wid) in vh.weights.iter().enumerate() {
            vb.bind(*wid, wbufs[i].as_ref());
        }
        vb.bind(vh.logits, vf_logits_buf.as_ref());
        let t0 = std::time::Instant::now();
        be.execute(vplan.as_ref(), &vb)
            .map_err(|e| anyhow!("{e}"))?;
        out_logits.resize(m * c.vocab, 0.0);
        be.download(vf_logits_buf.as_ref(), bytemuck::cast_slice_mut(out_logits))
            .map_err(|e| anyhow!("{e}"))?;
        cached.extend_from_slice(&prompt[start..]);
        return Ok((
            Vec::new(),
            GenStats {
                n_prompt: m,
                prompt_secs: t0.elapsed().as_secs_f64(),
                n_gen: 0,
                decode_secs: 0.0,
            },
        ));
    }

    // ── drive ───────────────────────────────────────────────────────────────────────
    // Sampling: greedy unless INFR_TEMP is set (the CLI sets chat defaults for run/serve; the
    // golden/parity tests pin INFR_TEMP=0 or leave it unset).
    let sampler = crate::sampling::Sampler::from_env();
    let mut rng = crate::sampling::seed_rng();
    let embed_scale = if gemma { (ne as f32).sqrt() } else { 1.0 };
    let mut out = Vec::new();
    let mut cur = prompt.to_vec();
    let mut logits = vec![0f32; c.vocab];
    // INFR_PROF=1: report prompt-ingest + decode tok/s to stderr (CPU perf iteration).
    let prof = std::env::var("INFR_PROF").is_ok();
    let mut prompt_t = std::time::Duration::ZERO;
    let mut decode_t = std::time::Duration::ZERO;
    let mut decode_n = 0usize;
    // INFR_PROF_DEC: split decode per-token wall time into host setup (build graph + compile + bind)
    // vs execute (record + submit + GPU + wait) to guide the record-once-replay decision.
    let mut dec_setup = std::time::Duration::ZERO;
    let mut dec_exec = std::time::Duration::ZERO;

    // ── batched prefill (dense non-MoE non-E2B models only) ──────────────────────────────────
    // Process all-but-the-last prompt tokens in a single graph execution: each Op::Linear runs
    // m=(N-1) activations against every weight row in parallel (O(out_f) rayon tasks, N-1 dots
    // each), reading each weight row ONCE and reusing it across all tokens. This fills the KV
    // cache for positions 0..N-2. The last prompt token is left for the normal decode loop so
    // that the "decode" stats (tok/s) remain meaningful and the first generated token is sampled
    // in the canonical way.
    //
    // Guard: MoE uses Op::MoeFfn (per-token expert routing, no batched variant yet); E2B/gemma4
    // requires a per-(token,layer) host-side input vector that is computed in the per-step loop.
    // Both fall through to the original token-by-token loop below unchanged.
    // Batched MoE prefill needs the adapter's GPU-routed expert path (Q4_K gate/up + Q6_K down —
    // what qwen3moe ships); other stacked formats keep the per-token loop.
    let moe_batched_ok = c.moe.is_some() && {
        let dt = |n: String| g.tensors().iter().find(|t| t.name == n).map(|t| t.dtype);
        (0..c.n_layer).all(|l| {
            dt(format!("blk.{l}.ffn_gate_exps.weight")) == Some(DType::Q4K)
                && dt(format!("blk.{l}.ffn_up_exps.weight")) == Some(DType::Q4K)
                && matches!(
                    dt(format!("blk.{l}.ffn_down_exps.weight")),
                    Some(DType::Q4K) | Some(DType::Q6K)
                )
        })
    };
    let decode_start = if prompt.len() - start > 2 && (c.moe.is_none() || moe_batched_ok) {
        // Batch-prefill the un-cached suffix, all but the last prompt token (positions
        // start..plen-1; rows 0..start are reused from the session cache) — in UBATCH CHUNKS.
        // One giant graph would scale the internal activation/attention scratch with the whole
        // prompt (an 8B p8000 prefill built a multi-second single submission whose tail work
        // tripped the amdgpu ring watchdog → device lost) and bakes a multi-second unpreemptible
        // submit; fixed-size chunks bound both, exactly like the bespoke path's ubatches.
        let ubatch: usize = std::env::var("INFR_UBATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1024);
        let pf_end = prompt.len() - 1;
        let mut cstart = start;
        while cstart < pf_end {
            let cend = (cstart + ubatch).min(pf_end);
            let pf_m = cend - cstart;
            let mut pf_hidden: Vec<f32> = Vec::with_capacity(pf_m * ne);
            for &tok in &prompt[cstart..cend] {
                let base = tok as usize * ne;
                pf_hidden.extend(token_embd[base..base + ne].iter().map(|&x| x * embed_scale));
            }
            // Absolute positions [cstart, ..., cend-1].
            let pf_positions: Vec<i32> = (cstart as i32..cend as i32).collect();
            let pf_hidden_buf = be
                .alloc(pf_m * ne * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            let pf_pos_buf = be
                .alloc(pf_m * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(pf_hidden_buf.as_ref(), bytemuck::cast_slice(&pf_hidden))
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(pf_pos_buf.as_ref(), bytemuck::cast_slice(&pf_positions))
                .map_err(|e| anyhow!("{e}"))?;

            // gemma4 E2B: the chunk's per-layer TOKEN embedding rows (gather+dequant only — the
            // model_proj GEMV/RMSNorm/combine run as GPU graph ops in the `build` prologue).
            let pf_ipl_buf = if let Some(ple) = ple {
                let ipl = e2b_ipl_rows(g, ple, &prompt[cstart..cend])?;
                let b = be
                    .alloc(ipl.len() * 4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?;
                be.upload(b.as_ref(), bytemuck::cast_slice(&ipl))
                    .map_err(|e| anyhow!("{e}"))?;
                Some(b)
            } else {
                None
            };
            let pf_t0 = std::time::Instant::now();
            let (pf_g, pf_h) = build(pf_m, cstart, 1);
            let t_build = pf_t0.elapsed();
            let pf_plan = be.compile(&pf_g).map_err(|e| anyhow!("{e}"))?;
            let t_compile = pf_t0.elapsed();
            let mut pf_b = Bindings::new();
            pf_b.bind(pf_h.hidden, pf_hidden_buf.as_ref());
            pf_b.bind(pf_h.positions, pf_pos_buf.as_ref());
            // gemma4's proportional-RoPE divisors are a graph input too — bind them (the per-token
            // decode loop below does the same). Without this the batched graph has an unbound
            // `rope_freqs` Input and panics.
            if let (Some(rid), Some((rb, _))) = (pf_h.rope_freqs, &rf_buf) {
                pf_b.bind(rid, rb.as_ref());
            }
            if let (Some(pid), Some(ib)) = (pf_h.pl_tok_in, &pf_ipl_buf) {
                pf_b.bind(pid, ib.as_ref());
            }
            for l in 0..c.n_layer {
                pf_b.bind(pf_h.k_cache[l], kbufs[l].as_ref());
                pf_b.bind(pf_h.v_cache[l], vbufs[l].as_ref());
            }
            for (i, wid) in pf_h.weights.iter().enumerate() {
                pf_b.bind(*wid, wbufs[i].as_ref());
            }
            pf_b.bind(pf_h.logits, logits_buf.as_ref());
            be.execute(pf_plan.as_ref(), &pf_b)
                .map_err(|e| anyhow!("{e}"))?;
            // INFR_PROF_PF: split the per-chunk prefill wall time into host graph build, plan
            // compile, and execute (record + submit + GPU) — where a small-batch chunk's fixed
            // cost lives decides whether to attack recording or kernels.
            if std::env::var("INFR_PROF_PF").is_ok() {
                eprintln!(
                    "[pf prof] m={pf_m} build={:.1}ms compile={:.1}ms execute={:.1}ms",
                    t_build.as_secs_f64() * 1e3,
                    (t_compile - t_build).as_secs_f64() * 1e3,
                    (pf_t0.elapsed() - t_compile).as_secs_f64() * 1e3,
                );
            }
            prompt_t += pf_t0.elapsed();
            cstart = cend;
        }

        // KV rows are now filled through position plen-2; the last prompt token is handled by
        // the decode loop below (writes its KV, produces the logits the first sample uses).
        pf_end
    } else {
        start // fall through to per-token loop for MoE / E2B / short suffixes
    };

    // Record-once decode: for an eligible decode on a backend that supports replay (the Vulkan
    // seam), build+compile+bind ONE plan here and reuse it across the whole decode loop. The
    // adapter records the graph once and replays it per token, reading `pos` from the bound
    // positions buffer + a params SSBO — so the baked pos=0 here is irrelevant, and the per-token
    // host cost drops to just the emb/pos (+ E2B ipl) uploads. The gate mirrors the adapter's
    // graph eligibility: every dense arch replays — qk-norm (qwen3), the gemma family (SWA
    // windows + scale via push constants, freq_factors via qk_norm_rope_dyn_ff, V-norm/Softcap/
    // Scale are pos-independent), llama (f16-out interleaved Rope via rope_f16_dyn), MoE. Backends
    // without `decode_replay` (CPU interpreter, which reads the baked `pos`) and every ineligible
    // model keep rebuilding + recompiling per token below.
    // INFR_SEAM_NO_REPLAY=1 forces per-token rebuild (the adapter's static path) — slower, but
    // INFR_PROF2 per-op GPU timestamps work there (the replay path can't report them).
    // This gate MUST stay a strict subset of the adapter's `decode_eligible` — the plan below
    // bakes pos=0/kv_len=1, which is only correct when the adapter replays it (dyn kernels read
    // the live pos/kv_len); an ineligible graph would silently run the static path with the baked
    // values. Hence the per-layer head-dim mirror of the adapter's Attention check.
    // llama (no qk-norm) replays too — its f16-out Rope has a dyn kernel — but only without
    // freq_factors (the standalone Rope kernel has no ff binding; gemma4's ff rides QkNormRope).
    let dyn_replay = be.capabilities().decode_replay
        && std::env::var("INFR_SEAM_NO_REPLAY").is_err()
        && (qk_norm || rope_freqs.is_none())
        // Any quantized KV cache forces the per-execute STATIC decode (see the adapter's
        // `decode_eligible`: Q8's un-fused K-write mis-decodes under record-once replay; the low-bit
        // block quants use a dequant→f16 prepass with a standalone quantizing WriteKv). Must mirror
        // that rejection so this gate stays a strict subset — else the loop bakes pos=0 for a static
        // run. (turbo forces static too but is CPU-only, where decode_replay is off anyway.)
        && !kv_forces_static(k_fmt)
        && !kv_forces_static(v_fmt)
        && (0..c.n_layer)
            .all(|l| c.layer_head_dim(l).is_multiple_of(4) && c.layer_head_dim(l) <= 512);
    let ro = if dyn_replay {
        let (g, h) = build(1, 0, 1);
        let plan = be.compile(&g).map_err(|e| anyhow!("{e}"))?;
        let mut b = Bindings::new();
        b.bind(h.hidden, hidden_buf.as_ref());
        b.bind(h.positions, pos_buf.as_ref());
        if let (Some(rid), Some((rb, _))) = (h.rope_freqs, &rf_buf) {
            b.bind(rid, rb.as_ref());
        }
        if let (Some(pid), Some(ib)) = (h.pl_tok_in, &ipl_buf) {
            b.bind(pid, ib.as_ref());
        }
        for l in 0..c.n_layer {
            b.bind(h.k_cache[l], kbufs[l].as_ref());
            b.bind(h.v_cache[l], vbufs[l].as_ref());
        }
        for (i, wid) in h.weights.iter().enumerate() {
            b.bind(*wid, wbufs[i].as_ref());
        }
        b.bind(h.logits, logits_buf.as_ref());
        Some((plan, b))
    } else {
        None
    };

    // INFR_IGNORE_EOS=1 (benchmarks): decode the full requested count — a model that emits EOS
    // instantly on a dummy context (gemma at depth) otherwise "finishes" 64 tokens in one step
    // and the reported tok/s is fiction. llama-bench ignores EOS the same way.
    let ignore_eos = std::env::var("INFR_IGNORE_EOS").is_ok();
    for pos in decode_start..(prompt.len() + max_new) {
        if out.len() >= max_new {
            break;
        }
        let step_t0 = std::time::Instant::now();
        let tok = cur[pos] as usize;
        // embed (gemma scales by √n_embd; qwen3/llama identity)
        let emb: Vec<f32> = token_embd[tok * ne..tok * ne + ne]
            .iter()
            .map(|&x| x * embed_scale)
            .collect();
        be.upload(hidden_buf.as_ref(), bytemuck::cast_slice(&emb))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(pos_buf.as_ref(), bytemuck::cast_slice(&[pos as i32]))
            .map_err(|e| anyhow!("{e}"))?;

        // gemma4 E2B: this token's per-layer TOKEN embedding row (gather+dequant only — the
        // model_proj GEMV/RMSNorm/combine run as GPU graph ops in the `build` prologue).
        if let (Some(ple), Some(ipl_buf)) = (ple, &ipl_buf) {
            let ipl = e2b_ipl_rows(g, ple, &[tok as u32])?;
            be.upload(ipl_buf.as_ref(), bytemuck::cast_slice(&ipl))
                .map_err(|e| anyhow!("{e}"))?;
        }

        let t_setup = std::time::Instant::now();
        let (setup_el, exec_el);
        if let Some((plan, b)) = &ro {
            // Record-once path: reuse the single compiled plan + bindings (no per-token rebuild).
            setup_el = t_setup.elapsed();
            let t_exec = std::time::Instant::now();
            be.execute(plan.as_ref(), b).map_err(|e| anyhow!("{e}"))?;
            exec_el = t_exec.elapsed();
        } else {
            let (g, h) = build(1, pos, 1);
            let plan = be.compile(&g).map_err(|e| anyhow!("{e}"))?;
            let mut b = Bindings::new();
            b.bind(h.hidden, hidden_buf.as_ref());
            b.bind(h.positions, pos_buf.as_ref());
            if let (Some(rid), Some((rb, _))) = (h.rope_freqs, &rf_buf) {
                b.bind(rid, rb.as_ref());
            }
            if let (Some(pid), Some(ib)) = (h.pl_tok_in, &ipl_buf) {
                b.bind(pid, ib.as_ref());
            }
            for l in 0..c.n_layer {
                b.bind(h.k_cache[l], kbufs[l].as_ref());
                b.bind(h.v_cache[l], vbufs[l].as_ref());
            }
            for (i, wid) in h.weights.iter().enumerate() {
                b.bind(*wid, wbufs[i].as_ref());
            }
            b.bind(h.logits, logits_buf.as_ref());
            setup_el = t_setup.elapsed();
            let t_exec = std::time::Instant::now();
            be.execute(plan.as_ref(), &b).map_err(|e| anyhow!("{e}"))?;
            exec_el = t_exec.elapsed();
        }
        if std::env::var("INFR_PROF_DEC").is_ok() && pos + 1 >= prompt.len() {
            dec_setup += setup_el;
            dec_exec += exec_el;
        }

        // Only sample once we're past the prompt (decode position = last prompt token onward).
        let is_decode = pos + 1 >= prompt.len();
        // Sample only at the FRONTIER (this position's token is the newest one fed). A constrained
        // step can emit several deterministically-forced tokens at once — they're queued onto
        // `cur` and the following iterations just feed them (no sampling) until the frontier.
        let at_frontier = pos + 1 == cur.len();
        if is_decode && at_frontier {
            be.download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
                .map_err(|e| anyhow!("{e}"))?;
            if let Some(cst) = constraint.as_deref_mut() {
                // Grammar-forced span (serve's tool_choice "required"/named): the shared
                // llguidance step. Empty step ⇒ the constrained span ended.
                let (step, done) = crate::grammar::constrained_step(cst, &mut logits, &c.eos_ids)
                    .map_err(|e| anyhow!("{e}"))?;
                decode_t += step_t0.elapsed();
                if step.is_empty() {
                    break;
                }
                for &t in &step {
                    out.push(t);
                    on_token(t);
                    cur.push(t);
                    decode_n += 1;
                }
                if done || out.len() >= max_new {
                    break;
                }
            } else {
                let next = crate::sampling::sample_logits(&logits, sampler, &mut rng);
                let is_eos = !ignore_eos && (c.eos_ids.contains(&next) || next == c.eos);
                out.push(next);
                decode_t += step_t0.elapsed();
                decode_n += 1;
                if !is_eos {
                    on_token(next); // stream the token (EOS is not emitted)
                }
                if is_eos || out.len() >= max_new {
                    break;
                }
                cur.push(next);
            }
        } else if is_decode {
            // feeding a queued forced token — its KV write is the whole point of this step
            decode_t += step_t0.elapsed();
        } else {
            prompt_t += step_t0.elapsed();
        }
    }
    if prof {
        let ts = |d: std::time::Duration, n: usize| {
            if d.as_secs_f64() > 0.0 {
                n as f64 / d.as_secs_f64()
            } else {
                0.0
            }
        };
        eprintln!(
            "[cpu prof] prompt {} tok in {:.2}s ({:.1} tok/s) | decode {} tok in {:.2}s ({:.2} tok/s)",
            prompt.len(),
            prompt_t.as_secs_f64(),
            ts(prompt_t, prompt.len()),
            decode_n,
            decode_t.as_secs_f64(),
            ts(decode_t, decode_n),
        );
    }
    if std::env::var("INFR_PROF_DEC").is_ok() && decode_n > 0 {
        eprintln!(
            "[dec prof] {} decode tok | setup(build+compile+bind) {:.3}ms/tok | exec(record+submit+gpu) {:.3}ms/tok",
            decode_n,
            dec_setup.as_secs_f64() * 1e3 / decode_n as f64,
            dec_exec.as_secs_f64() * 1e3 / decode_n as f64,
        );
    }
    // Record what the KV cache now holds for the next turn's prefix diff. `out` includes any
    // sampled EOS (its KV row was written before the loop broke)... it was PUSHED to out before
    // the break, and its KV is written only when fed back — the EOS is never fed, so the cache
    // holds prompt + generated-minus-last-fed. Conservative: cache prompt + all fed tokens; the
    // final sampled token's row is NOT materialized, so exclude it.
    *cached = prompt.to_vec();
    if !out.is_empty() {
        cached.extend_from_slice(&out[..out.len() - 1]);
    }
    let stats = GenStats {
        // The tokens actually PREFILLED this call (the un-cached suffix) — the TTFT-honest count.
        n_prompt: prompt.len() - start,
        prompt_secs: prompt_t.as_secs_f64(),
        n_gen: decode_n,
        decode_secs: decode_t.as_secs_f64(),
    };
    Ok((out, stats))
}
