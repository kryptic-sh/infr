//! CPU reference backend — a correctness-first interpreter of the backend-agnostic
//! [`infr_core`] compute [`Graph`]. Projection matmuls and attention use rayon for multi-core
//! parallelism; QK/PV inner loops use an 8-accumulator dot for AVX autovectorization.
//! Weights are read **zero-copy from the GGUF mmap** (no `memcpy`, no owned RAM): the bulk
//! projection weights (`Op::Linear`) are dequantized one row at a time straight from the mapping
//! inside the dot, so 12B / MoE models cost only their on-disk size in page cache. Only the tiny
//! norm weights are dequant-cached; the model writes (KV / conv / recurrent state, per-step IO) use
//! small owned buffers. It exists to (a) run every model without a GPU and (b) serve as the oracle
//! the GPU backends are validated against.
#![allow(clippy::needless_range_loop)]

mod attn_kv;
mod indexer;
#[doc(hidden)]
pub mod indexer_tap;
pub mod kvquant;
pub mod linear_tap;
mod op_hash;
pub mod op_tap;
pub mod paged;
#[cfg(test)]
mod paged_moe_tests;
mod pool;
pub mod turbo;

mod dequant_row;
mod kernels;
mod moe;
mod quant;
mod recycle;
mod repack;
mod simd;

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Capabilities, GraphPlan, Plan};
use infr_core::config::Config;
use infr_core::error::Result;
use infr_core::exec::Provision;
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::math;
use infr_core::tensor::{DType, TensorId};
use infr_gguf::dequant::dequant_block;
use infr_gguf::TensorBytes;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use kernels::{
    gated_act_row, vec_dot_hi, vec_dot_iq1m_batch, vec_dot_iq1s_batch, vec_dot_iq2s_batch,
    vec_dot_iq2xs_batch, vec_dot_iq2xxs_batch, vec_dot_iq3s_batch, vec_dot_iq3xxs_batch,
    vec_dot_iq4nl_32_batch, vec_dot_iq4xs_batch, vec_dot_mxfp4_32_batch, vec_dot_nvfp4_batch,
    vec_dot_q2_0_batch, vec_dot_q2k_batch, vec_dot_q3k_batch, vec_dot_q4_0_32_batch,
    vec_dot_q4_1_32_batch, vec_dot_q4k_batch, vec_dot_q4k_batch2, vec_dot_q4k_batch8,
    vec_dot_q5_0_32_batch, vec_dot_q5_1_32_batch, vec_dot_q5k_batch, vec_dot_q6k_batch,
    vec_dot_q8_0_32_batch, vec_dot_q8_0_batch, vec_dot_q8_any, vec_dot_tq1_0_batch,
    vec_dot_tq2_0_batch, Dots, HiBatch,
};
use moe::{expert_acts_kind, expert_gemm_range, ActsKind, ExpertActs};
use quant::{quantize_q8, quantize_q8_32, Q8x32, Q8};
#[cfg(target_arch = "x86_64")]
use repack::q4k_pack;
use repack::{Q4kPack, RepackCacheState};

/// Dequantized-weight cache: `(buffer address, byte-len, dtype) -> Arc<f32>`. See `weight_cache`.
type WeightCache = HashMap<(u64, DType), Arc<Vec<f32>>>;

/// RoPE frequency tables ([`math::rope_freqs`]) keyed by `(theta.to_bits(), rope_dim)`, the only
/// inputs a table depends on (a per-pair divisor is applied per angle, not baked in).
type RopeFreqCache = HashMap<(u32, usize), Arc<[f32]>>;

// ─── Q8_0 integer dot kernels ─────────────────────────────────────────────────
//
// Q8_0 weight layout: 34 bytes / 32 elements.  Bytes 0..2 = f16 scale `d`; bytes 2..34 = i8 qs.
// Activation comes in as a `Q8` super-block (256 elems), so one super-block covers 8 Q8_0 weight
// blocks.  Since both weight and activation are i8, we use the llama.cpp sign trick:
// `maddubs(abs(qw), sign(qw)·qx)` = `Σ qw[i]·qx[i]` without overflow into i16.

// ─── Q5_K integer dot kernels ─────────────────────────────────────────────────
//
// Q5_K block layout (176 bytes / 256 elems):
//   [f16 d][f16 dmin][scales[12]][qh[32]][ql[128]]
// q5 = (ql_nibble) | (((qh[l] >> bit) & 1) << 4)  ∈ 0..31  (UNSIGNED → maddubs works directly)
// Dot formula: d·sc·Σ(q5·qx) − dmin·m·Σqx  — identical structure to Q4_K.
// `q8.bsums` provides Σqx per 32-elem sub-block (precomputed in quantize_q8).

// ─── Batched dot kernels (prefill: m > 1) ────────────────────────────────────
//
// Each `vec_dot_qXk_batch(row, q8s, in_f, out)` is equivalent to calling
// `vec_dot_qXk(row, &q8s[r], in_f)` for every r, but decodes the weight row
// ONCE and loops over the m token activations with the pre-decoded data.
//
// Bit-identity guarantee: the per-token f32 result equals the single-token
// kernel exactly (integer dots have no rounding, same accumulation grouping).

// ── Q4_K batch ────────────────────────────────────────────────────────────────

// ── Q4_K 2-row tiled batch ────────────────────────────────────────────────────
//
// Process TWO output rows simultaneously so the Q8 activation data (loaded from
// L3 cache) is reused for both dots instead of loaded twice. This halves the L3
// bandwidth for Q8 reads which is the dominant bottleneck during large-batch prefill.
//
// `out_a` and `out_b` receive the dots for row `row_a` and `row_b` respectively.
// Bit-identical: each `out_x[r]` equals `vec_dot_q4k(row_x, &q8s[r], in_f)`.

// ── Q4_K 8-row tiled batch ────────────────────────────────────────────────────
//
// Process EIGHT output rows simultaneously: the Q8 activation zmm is loaded ONCE
// per (block, nibble-pair) and reused across all 8 row dots. This is 4× less
// activation traffic than the 2-row path and 8× less than the single-row path.
//
// `outs[i][r]` == `vec_dot_q4k(rows[i], &q8s[r], in_f)` — bit-identical to the
// single-token kernel (same per-block accumulation order; tiling only changes which
// rows are computed together, not the per-(row,token) arithmetic).

// ── Q6_K batch ────────────────────────────────────────────────────────────────

// ── Q8_0 batch ────────────────────────────────────────────────────────────────

// ── Q5_K batch ────────────────────────────────────────────────────────────────

// ─── Native 32-block int8 dot kernels (Q8_0 / Q5_0 at their OWN block size) ──────────────────────
//
// The K-quant/Q8_0 batch kernels above all group activations into 256-element super-blocks (`Q8`),
// which requires `in_f % 256 == 0` — true for most projections but NOT MoE's `down` projection,
// whose `in_f = n_ff_exp` can be any multiple of 32 (e.g. DiffusionGemma's 704 = 22×32). Q8_0/Q5_0's
// own native block is 32 elements, so this activation quantizes at THAT granularity instead —
// scalar only (no SIMD; this path is memory/allocation-bound, not compute-bound, so a plain scalar
// loop over int8 bytes already beats dequantizing the whole row to f32 first).

/// Hands out a process-unique [`CpuBuffer::uid`]. Monotonic and never reused, which is the whole
/// point: an address IS reused (a freed buffer's allocation, or a fresh mmap landing where the
/// previous model's was), so a cache keyed on one can hand back another weight's data.
fn next_buffer_uid() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// A host buffer. Weights are **mapped** — a zero-copy [`TensorBytes`] view straight into the GGUF
/// mmap (read-only, no `memcpy`, no owned RAM). Everything the model writes (KV / conv / recurrent
/// state, per-step IO) is **owned** — a plain byte vec behind a `Mutex` (so `&dyn Buffer` stays
/// `Send + Sync` and writes are safe). `&dyn Buffer` reads go through `CpuBuffer::read`.
pub struct CpuBuffer {
    /// This buffer's identity for the backend's derived-data caches (`weight_cache`, the repack
    /// caches). It is a `uid` and not an address because those caches outlive a model: a reused
    /// `CpuBackend` (serve model reload) frees every buffer and maps a new file, and a stale entry
    /// keyed by an address the allocator or the kernel handed out again would be returned for a
    /// DIFFERENT weight — plausible numbers, no error. A uid is never handed out twice.
    uid: u64,
    store: CpuStore,
}

enum CpuStore {
    /// Locked through [`lock_bytes`], never `lock().unwrap()`.
    Owned(Mutex<Vec<u8>>),
    Mapped(TensorBytes),
    /// Read from the model file into the host arena on demand (`paging.dram`). The bytes are only
    /// valid while pinned, which is why every op pins what it reads before running — see
    /// [`CpuBackend::pin_op_weights`].
    Paged {
        store: Arc<paged::PagedWeights>,
        id: paged::PagedId,
        nbytes: usize,
    },
    /// An expert bank paged per expert: only `Op::MoeFfn` reads it, and only the experts its
    /// routing picked, which it pins itself once routing has run — see [`CpuBackend::moe_banks`].
    PagedExperts {
        store: Arc<paged::PagedWeights>,
        bank: paged::PagedExperts,
        nbytes: usize,
    },
}

/// A uniform read view over any storage (a `MutexGuard` for owned, the slice for mapped, a pin for
/// paged); all deref to `[u8]`.
enum CpuRead<'a> {
    Owned(std::sync::MutexGuard<'a, Vec<u8>>),
    Mapped(&'a TensorBytes),
    /// Holds the block resident for as long as the view lives.
    Pinned(infr_core::hostpager::Pin<'a>),
}

/// One expert bank as `Op::MoeFfn` reads it: expert `e`'s bytes through [`Bank::expert`].
enum Bank<'a> {
    /// Held as one view — mapped, owned, or paged whole — of `n_expert` slices `stride` bytes
    /// apart.
    Whole { bytes: CpuRead<'a>, stride: usize },
    /// Paged per expert: the op's routed experts, each pinned, and `None` for every other.
    Experts(Vec<Option<infr_core::hostpager::Pin<'a>>>),
}

impl Bank<'_> {
    fn expert(&self, e: usize) -> &[u8] {
        match self {
            Bank::Whole { bytes, stride } => &bytes[e * stride..(e + 1) * stride],
            Bank::Experts(pins) => pins[e].as_deref().unwrap_or_else(|| {
                panic!("cpu Op::MoeFfn read expert {e}, which its routing did not pin")
            }),
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl std::ops::Deref for CpuRead<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            CpuRead::Owned(g) => g,
            CpuRead::Mapped(t) => t,
            CpuRead::Pinned(p) => p,
        }
    }
}

/// An owned buffer's bytes, poisoned or not.
///
/// A panic in an op poisons every owned buffer the op held (its outputs, a whole expert bank, a KV
/// cache), and `unwrap` would then fail every later graph that binds one. No in-tree caller reuses
/// a backend after a panic (each drops it and its buffers), but a library caller that catches the
/// panic and keeps the backend would be stuck with it, so the poison is cleared here.
///
/// Only the locks are usable again: after a panic the buffers' CONTENTS are unspecified (a KV row
/// an earlier `WriteKv` of the failed graph wrote is not marked as such). A caller that catches the
/// panic must discard or rewind its session state, as every in-tree caller does by dropping it.
fn lock_bytes(m: &Mutex<Vec<u8>>) -> std::sync::MutexGuard<'_, Vec<u8>> {
    m.lock().unwrap_or_else(|poisoned| {
        m.clear_poison();
        poisoned.into_inner()
    })
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl CpuBuffer {
    /// A writable buffer over `bytes` (KV, recurrent state, per-step IO).
    fn owned_buf(bytes: Vec<u8>) -> Self {
        Self {
            uid: next_buffer_uid(),
            store: CpuStore::Owned(Mutex::new(bytes)),
        }
    }
    /// A read-only zero-copy view of a GGUF mmap region (weights).
    fn mapped_buf(bytes: TensorBytes) -> Self {
        Self {
            uid: next_buffer_uid(),
            store: CpuStore::Mapped(bytes),
        }
    }
    /// Read view of the bytes (zero-copy for mapped weights; mutex guard for owned buffers; a pin
    /// for paged ones).
    ///
    /// # Panics
    /// If a PAGED weight is not resident. This path cannot read from disk — it has no way to
    /// report an I/O failure — so residency is established first, by the op's pin pre-step
    /// ([`CpuBackend::pin_op_weights`]), which runs before any op body and does return errors. A
    /// panic here means an op read a weight its `Op::io()` did not list.
    fn read(&self) -> CpuRead<'_> {
        match &self.store {
            CpuStore::Owned(m) => CpuRead::Owned(lock_bytes(m)),
            CpuStore::Mapped(t) => CpuRead::Mapped(t),
            CpuStore::Paged { store, id, .. } => CpuRead::Pinned(store.try_pin(*id).expect(
                "cpu backend: a paged weight was read without being pinned first — the op's \
                 Op::io() reads must name every weight its body reads",
            )),
            CpuStore::PagedExperts { .. } => panic!(
                "cpu backend: an expert bank paged per expert was read whole — only \
                 Op::MoeFfn reads one, through its routed experts' pins"
            ),
        }
    }
    /// Mutable owned storage; panics for mapped (read-only) weights.
    fn owned(&self) -> std::sync::MutexGuard<'_, Vec<u8>> {
        match &self.store {
            CpuStore::Owned(m) => lock_bytes(m),
            CpuStore::Mapped(_) | CpuStore::Paged { .. } | CpuStore::PagedExperts { .. } => {
                panic!("cpu backend: write to a read-only weight buffer")
            }
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Buffer for CpuBuffer {
    fn len_bytes(&self) -> usize {
        match &self.store {
            CpuStore::Owned(m) => lock_bytes(m).len(),
            CpuStore::Mapped(t) => t.len(),
            CpuStore::Paged { nbytes, .. } | CpuStore::PagedExperts { nbytes, .. } => *nbytes,
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct CpuBackend {
    /// Dequantized-weight cache keyed by the bound buffer's `(uid, dtype)` (weights are bound the
    /// same every step, so dequant once and reuse). Only the small norm weights (`RmsNorm` /
    /// `QkNorm`) land here — the large `Op::Linear` weights are streamed row-by-row instead (see
    /// that arm), so this never holds the whole model in f32. The uid is what makes a reused
    /// `CpuBackend` (serve model reload) safe: a freed weight's ADDRESS can be handed out again for
    /// a different weight, and an address key would then return the previous model's f32 (see
    /// [`CpuBuffer::uid`]).
    weight_cache: Mutex<WeightCache>,
    /// RoPE frequency tables, built once per distinct `(theta, rope_dim)` for the backend's life:
    /// each entry costs one correctly rounded `powf` per pair, which a one-row decode op would
    /// otherwise pay on every call. A model has a table per distinct base (gemma-3: the local
    /// and the global one), so this stays a handful of entries.
    rope_freq_cache: Mutex<RopeFreqCache>,
    /// (layer, expert)-granular repack cache for the interleaved-x8 Q4_K GEMM ([`Q4kPack`]):
    /// keyed by the expert weight slice's [`repack::PackKey`] (same identity argument as
    /// `weight_cache`, extended to a slice WITHIN a buffer). ggml pays
    /// its `block_q4_Kx8` repack once at LOAD; this pays it once per (expert, session) instead
    /// of once per CALL. Byte-budgeted (`kernels.cpu.repack_mb`, default 4096 MiB): over budget,
    /// packs are built transient and not inserted. The `usize` is the current cached-bytes total.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    repack_cache: Mutex<RepackCacheState>,
    /// Persistent spin-pool for the op interpreter's parallel loops (see `pool.rs`, threadpool
    /// restructure phase 2). Built on first use so backends constructed for tests/tiny work
    /// never spawn threads; `execute` builds it up front only for a graph with an `Op::Linear`,
    /// to place the dispatching thread for the walk.
    pool: std::sync::OnceLock<pool::SpinPool>,
    /// The engine configuration this backend reads its knobs from — one value, held for the
    /// backend's whole life, borrowed (never cloned) at every read site. The knobs it reads are
    /// the `kernels.cpu` ones ([`infr_core::config::CpuCfg`]), `prof.prof_ops` and
    /// `debug.moe_counts{,_dump}` and `debug.cpu_op_hash`.
    ///
    /// Every construction site inside this repo goes through [`CpuBackend::new_with`] (the seam
    /// threads its `Arc<Config>` into them). [`CpuBackend::new`] survives as the ENV-SOURCED twin
    /// for a library caller that has no config of its own: ONE
    /// [`infr_core::config::Config::load_from_env`] per backend, in place of the scattered
    /// per-read `std::env::var` calls it replaced. It is not a bridge to be deleted — it is the
    /// documented no-argument constructor.
    cfg: Arc<Config>,
    /// Query rows one `Op::Attention` task runs for one head: [`ATTN_QUERY_BLOCK`]. A field so a
    /// test can set 1, the one-pair-per-task decomposition the blocking replaced.
    attn_query_block: usize,
    /// This backend's identity for the per-op hash tap (`debug.cpu_op_hash`, [`op_hash`]): its
    /// backend ordinal and the number of the next graph it records.
    op_hash: op_hash::TapCounter,
    /// A test's deliberate fault: `Op::MoeFfn` reads a bank held whole at twice its expert
    /// stride, so its upper experts slice past the bank's end inside the pool's tasks.
    #[cfg(test)]
    double_whole_bank_stride: std::sync::atomic::AtomicBool,
}

/// `default()` is `new()` — the env-sourced bridge, NOT `Config::default()`. Keeping them the same
/// value is what makes this behaviour-preserving: `CpuBackend` used to derive `Default`
/// and `new()` was `default()`, and every knob was read from the environment further down.
impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl CpuBackend {
    /// A backend on the shipped configuration, with the environment folded in (see the `cfg`
    /// field). Prefer [`Self::new_with`] when you have a `Config`; nothing in this repo calls this.
    pub fn new() -> Self {
        Self::new_with(Arc::new(Config::load_from_env()))
    }

    /// A backend that reads its knobs from `cfg`. This is the constructor the campaign is heading
    /// for: no environment, and two backends with different configurations can coexist in one
    /// test, with ONE exception: `kernels.cpu.max_tier` is a process-wide debug switch. Every
    /// construction applies it ([`infr_core::simd::set_max`]; unset = no ceiling, the widest
    /// tier the CPU has; a ceiling above the CPU's widest runs the widest, and so does another
    /// architecture's tier, with a warning), so the backend built last decides the SIMD tier of
    /// every kernel and math slice for every backend in the process. Only speed is at stake:
    /// every tier gives the same bits.
    pub fn new_with(cfg: Arc<Config>) -> Self {
        if let Some(t) = cfg.kernels.cpu.max_tier.filter(|t| !t.native()) {
            tracing::warn!(
                "kernels.cpu.max_tier = {:?} is not a tier of this CPU's architecture: no ceiling",
                t.name()
            );
        }
        infr_core::simd::set_max(cfg.kernels.cpu.max_tier);
        Self {
            weight_cache: Mutex::default(),
            rope_freq_cache: Mutex::default(),
            repack_cache: Mutex::default(),
            pool: std::sync::OnceLock::new(),
            cfg,
            attn_query_block: ATTN_QUERY_BLOCK,
            op_hash: op_hash::TapCounter::default(),
            #[cfg(test)]
            double_whole_bank_stride: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// A backend in REFERENCE mode: every quantized `Op::Linear` (dense and MoE-expert) decodes
    /// its weight to f32 and takes a plain f32 dot, so the only error left is the weight's own
    /// quantization. The production path instead quantizes the ACTIVATIONS to int8 (the
    /// llama.cpp `vec_dot_*_q8_K` regime) — same arithmetic ggml uses, and ~2-3 orders of
    /// magnitude cheaper, but it carries ~4e-3 relative error on every dtype. That error flips
    /// greedy tokens at Q2_K, and at Q4_K it can still flip a near tie, which is why a GPU-vs-CPU
    /// token comparison must use THIS mode as its oracle rather than the production kernels (see
    /// `gpu_seam_matches_cpu_qwen3_q2k`): the GPU's f32 dequant GEMV is ~1e-7 against the host
    /// decode, so the int8 CPU leg was the drifting one.
    ///
    /// Reference mode is for oracles and triage only — it is far slower and it is NOT what the
    /// CPU goldens pin (`cpu_golden_qwen3` = the production int8 path).
    ///
    /// The mode is the `kernels.cpu.reference` config field and this is the shorthand for
    /// setting it — `new()`'s configuration with that one field flipped, so an oracle run still
    /// honours the same `kernels.cpu.*` knobs the production backend does. It has no env key and
    /// never had one: the caller chooses the mode, which is the shape the rest of the campaign
    /// converges on.
    pub fn reference() -> Self {
        Self::reference_with(Arc::new(Config::load_from_env()))
    }

    /// [`Self::reference`] on a CALLER-supplied configuration — the same `kernels.cpu.reference`
    /// flip, applied to the config the caller already holds instead of one built from the
    /// environment. The seam uses this (it owns an `Arc<Config>` and must not re-read the
    /// environment); `reference()` stays as the shorthand for callers that have none.
    pub fn reference_with(cfg: Arc<Config>) -> Self {
        let mut cfg = (*cfg).clone();
        cfg.kernels.cpu.reference = true;
        Self::new_with(Arc::new(cfg))
    }

    /// Drop the spin pool and its threads; the next [`Backend::execute`] that needs one builds a
    /// new one. For a caller that keeps this backend between generations for its caches but not
    /// its threads: measured on 2026-11-15, a pool kept across ERNIE-4.5-21B-A3B requests
    /// prefilled about 10% slower than one built per request (docs/backlog.md, B179). An idle
    /// backend then holds no threads either.
    pub fn release_pool(&mut self) {
        self.pool = std::sync::OnceLock::new();
    }

    /// The number of threads a job on this backend's pool runs on (the caller and the workers),
    /// once an [`Backend::execute`] has built it: the rayon pool size of the thread that built
    /// it. `None` before. Lets a test confirm the thread count it asked for is the one it got.
    pub fn pool_threads(&self) -> Option<usize> {
        self.pool.get().map(pool::SpinPool::threads)
    }

    fn pool(&self) -> &pool::SpinPool {
        self.pool
            .get_or_init(|| pool::SpinPool::new(&self.cfg.kernels.cpu))
    }

    /// Add each activation row's sparse part ([`Q8::hi`], [`vec_dot_hi`]) into a prefill
    /// `Op::Linear`'s o-major output `out` (`out[o * m + r]`) over the weight rows `wbytes`
    /// (`bpr` bytes each). Each value becomes `dense + vec_dot_hi(..)`, the one-row route's
    /// expression ([`HiBatch`] computes the same bits for many weight rows at once); rows with
    /// no sparse part are not touched, and when no row has one nothing runs. (The packed GEMM's
    /// row-major output takes its parts in [`Self::linear_packed`].)
    fn add_hi(
        &self,
        dt: DType,
        wbytes: &[u8],
        bpr: usize,
        in_f: usize,
        q8s: &[Q8],
        out: &mut [f32],
    ) {
        let Some(hb) = HiBatch::new(dt, in_f, q8s) else {
            return;
        };
        // Each task owns a run of whole weight rows' columns.
        let m = q8s.len();
        self.pool()
            .for_chunks_mut(out, HI_ROWS * m, 1, &|c, block| {
                hb.add_o_major(wbytes, bpr, c * HI_ROWS, block.len() / m, block, m);
            });
    }

    /// Record one `Op::Linear` for the test tap (`linear_tap`): the `rows` of its activation
    /// `xs` (`m × in_f`), its output `out` (`m × out_f`), and each row recomputed exactly from the
    /// weight rows `wbytes` (`bpr` bytes each, stored as `dt`).
    #[allow(clippy::too_many_arguments)]
    fn tap_linear(
        &self,
        op_idx: usize,
        dt: DType,
        wbytes: &[u8],
        bpr: usize,
        xs: &[f32],
        out: &[f32],
        m: usize,
        in_f: usize,
        out_f: usize,
        rows: &[usize],
    ) {
        // Each row three ways: as it is, and after a round trip through each production
        // activation quantizer.
        let sim = |block| linear_tap::ActSim {
            block,
            keep: 0,
            ratio: 0,
            qmax: 127,
            split: false,
            sub: 0,
            emax: 0,
        };
        let variants: Vec<[Vec<f32>; 3]> = rows
            .iter()
            .map(|&r| {
                let x = &xs[r * in_f..r * in_f + in_f];
                [
                    x.to_vec(),
                    quant::dequantize_q8(&quantize_q8(x)),
                    sim(32).round_trip(x),
                ]
            })
            .collect();
        let exact: Vec<Vec<[f64; 3]>> = self.pool().collect(out_f, &|o| {
            let w = bytes_to_f32(&wbytes[o * bpr..o * bpr + bpr], dt);
            let dot = |x: &[f32]| w.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum();
            variants
                .iter()
                .map(|v| [dot(&v[0]), dot(&v[1]), dot(&v[2])])
                .collect()
        });
        let rows = rows
            .iter()
            .enumerate()
            .map(|(k, &r)| linear_tap::TappedRow {
                row: r,
                x: xs[r * in_f..r * in_f + in_f].to_vec(),
                out: out[r * out_f..r * out_f + out_f].to_vec(),
                exact: exact.iter().map(|e| e[k][0]).collect(),
                exact_q256: exact.iter().map(|e| e[k][1]).collect(),
                exact_q32: exact.iter().map(|e| e[k][2]).collect(),
            })
            .collect();
        linear_tap::record(linear_tap::TappedLinear {
            op_idx,
            dtype: dt,
            m,
            in_f,
            out_f,
            rows,
        });
    }

    /// The repack cache's byte budget (`kernels.cpu.repack_mb`, MiB → bytes), read by
    /// [`Self::q4k_pack_for`].
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) fn repack_budget_bytes(&self) -> usize {
        self.cfg.kernels.cpu.repack_mb * 1024 * 1024
    }

    /// Fetch-or-build the [`Q4kPack`] for one weight bank slice (see `repack_cache`'s doc).
    /// `key` identifies the slice by its buffer's uid + byte range — see [`repack::PackKey`].
    /// SAFETY-adjacent note: callers only reach this from the VNNI ilv path, which implies AVX2
    /// for the pack expansion.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn q4k_pack_for(
        &self,
        key: repack::PackKey,
        w: &[u8],
        in_f: usize,
        out_f: usize,
    ) -> Arc<Q4kPack> {
        if let Some(p) = self.repack_cache.lock().unwrap().0.get(&key) {
            return p.clone();
        }
        let pack = Arc::new(unsafe { q4k_pack(w, in_f, out_f) });
        let bytes = pack.bytes();
        // Re-check under the lock: another thread may have inserted `key` between our miss and here.
        let mut guard = self.repack_cache.lock().unwrap();
        repack::cache_insert_if_absent(&mut guard, key, pack, bytes, self.repack_budget_bytes())
    }

    /// Prefill `Op::Linear` on the packed VNNI GEMM (see [`linear_packed_ok`]): `out` is the
    /// row-major `[m, out_f]` result, written in full. Each task packs one 8-row group into its
    /// thread's scratch group and runs the GEMM over every token: the pack costs one pass over
    /// the group's weights, which the `m` tokens share, and nothing is held past the call. The
    /// `out_f % 8` tail rows take the per-row batch kernels. Every kernel here is bit-identical
    /// to the single-token one, so the result does not depend on the route. Rows with a sparse
    /// activation part ([`Q8::hi`]) get it added as every route adds it (`dense +
    /// vec_dot_hi(..)`, through [`HiBatch`]).
    #[cfg(target_arch = "x86_64")]
    fn linear_packed<const CAP: bool>(
        &self,
        dt: DType,
        wbytes: &[u8],
        in_f: usize,
        out_f: usize,
        q8s: &[Q8],
        out: &mut [f32],
    ) {
        let m = q8s.len();
        assert_eq!(out.len(), m * out_f);
        let (nb, bpr, groups) = (in_f / 256, wbytes.len() / out_f, out_f / 8);
        let gout = repack::GemmOut::row_major(out, out_f);
        // SAFETY: `linear_packed_ok` checked AVX-512BW + VNNI; each task writes only its own
        // group's rows `8g..8g + 8` of `out`, which outlives the `run`.
        let gemm = |g: usize| unsafe {
            let rows = std::array::from_fn(|i| &wbytes[(8 * g + i) * bpr..(8 * g + i + 1) * bpr]);
            match dt {
                DType::Q4K | DType::Q5K => {
                    repack::q4k_pack_and_gemm_group(dt, rows, nb, q8s, gout, 8 * g)
                }
                DType::Q6K => repack::q6k_pack_and_gemm_group(rows, nb, q8s, gout, 8 * g),
                _ => unreachable!("linear_packed_ok admits Q4_K, Q5_K and Q6_K only"),
            }
        };
        // The sparse activation parts ([`HiBatch`]): each GEMM task adds them to its own group's
        // results while it holds those weight rows, so they read no weight memory again.
        let hb = HiBatch::new(dt, in_f, q8s);
        match &hb {
            Some(hb) => self.pool().run(groups, &|g| {
                gemm(g);
                // One allocation per group task (`chunk_len` i32s: 16 per distinct element and block);
                // the thread-local scratch (`moe::with_block_scratch`) holds f32s, so is not reused.
                let mut tab = vec![0i32; hb.chunk_len()];
                hb.for_group(wbytes, bpr, 8 * g, 8, &mut tab, |r, acc| {
                    // SAFETY: as the GEMM's stores: this task's group, rows `8g..8g + 8`.
                    unsafe { gout.add8(8 * g, r, acc) }
                });
            }),
            None => self.pool().run(groups, &|g| gemm(g)),
        }
        let tail = out_f - 8 * groups;
        if tail > 0 {
            let mut cols = vec![0f32; tail * m];
            self.pool().for_chunks_mut(&mut cols, m, 1, &|i, col| {
                let o = 8 * groups + i;
                let row = &wbytes[o * bpr..(o + 1) * bpr];
                match dt {
                    DType::Q4K => vec_dot_q4k_batch::<CAP>(row, q8s, in_f, col),
                    DType::Q5K => vec_dot_q5k_batch::<CAP>(row, q8s, in_f, col),
                    _ => vec_dot_q6k_batch::<CAP>(row, q8s, in_f, col),
                }
            });
            if let Some(hb) = &hb {
                hb.add_o_major(wbytes, bpr, 8 * groups, tail, &mut cols, m);
            }
            for (i, col) in cols.chunks(m).enumerate() {
                for (r, &v) in col.iter().enumerate() {
                    out[r * out_f + 8 * groups + i] = v;
                }
            }
        }
    }

    /// Wrap a zero-copy GGUF mmap view as a read-only weight buffer (no allocation, no `memcpy`).
    pub fn map_weight(&self, bytes: TensorBytes) -> Box<dyn Buffer> {
        Box::new(CpuBuffer::mapped_buf(bytes))
    }

    /// Wrap a registered paged weight (see [`paged::PagedWeights`]) as a read-only weight buffer.
    /// Nothing is read here: the bytes arrive on the first pin.
    pub fn paged_weight(
        &self,
        store: Arc<paged::PagedWeights>,
        id: paged::PagedId,
        nbytes: usize,
    ) -> Box<dyn Buffer> {
        Box::new(CpuBuffer {
            uid: next_buffer_uid(),
            store: CpuStore::Paged { store, id, nbytes },
        })
    }

    /// Wrap an expert bank registered per expert (see [`paged::PagedWeights::register_experts`])
    /// as a read-only weight buffer. Nothing is read here: `Op::MoeFfn` pins the experts it routes
    /// to.
    pub fn paged_experts_weight(
        &self,
        store: Arc<paged::PagedWeights>,
        bank: paged::PagedExperts,
        nbytes: usize,
    ) -> Box<dyn Buffer> {
        Box::new(CpuBuffer {
            uid: next_buffer_uid(),
            store: CpuStore::PagedExperts {
                store,
                bank,
                nbytes,
            },
        })
    }

    /// Pin every paged weight `op` READS, returning guards that keep them resident until dropped.
    ///
    /// Driven by [`Op::io`], the same exhaustive enumeration the multi-device pipeline executor
    /// uses — so a new `Op` variant cannot forget to declare its reads here without also failing
    /// that. A weight an op reads without listing panics at `CpuBuffer::read` rather than silently
    /// reading an evicted slot.
    ///
    /// An expert bank paged per expert is skipped for `Op::MoeFfn`, which pins its routed experts
    /// itself ([`Self::moe_banks`]) — which ones is not known until its router has run. Any other op
    /// naming one is refused: nothing else reads a bank, and reading all of it would defeat the
    /// point of paging it by expert.
    fn pin_op_weights<'b>(
        &self,
        op: &Op,
        g: &Graph,
        bindings: &'b Bindings<'b>,
    ) -> Result<Vec<infr_core::hostpager::Pin<'b>>> {
        let mut pins = Vec::new();
        let t0 = std::time::Instant::now();
        let mut waited_on: Option<&paged::PagedWeights> = None;
        for id in op.io().0 {
            let Some(buf) = bindings.get(id) else {
                continue;
            };
            // Not every bound operand is a `CpuBuffer` in every test harness; a non-CPU buffer
            // cannot be paged, so it is simply not ours to pin.
            let Some(cb) = buf.as_any().downcast_ref::<CpuBuffer>() else {
                continue;
            };
            match &cb.store {
                CpuStore::Paged { store, id, .. } => {
                    pins.push(store.pin_as(*id, paged::whole_insert(g.weight_reuse))?);
                    waited_on = Some(store);
                }
                CpuStore::PagedExperts { .. } if matches!(op, Op::MoeFfn { .. }) => {}
                CpuStore::PagedExperts { .. } => {
                    return Err(infr_core::error::Error::backend(format!(
                        "cpu backend: {} reads an expert bank paged per expert, which only \
                         Op::MoeFfn can read",
                        infr_core::prof::op_label(op, g)
                    )))
                }
                CpuStore::Owned(_) | CpuStore::Mapped(_) => {}
            }
        }
        if let Some(store) = waited_on {
            store.note_wait(t0.elapsed());
        }
        Ok(pins)
    }

    /// The expert banks of one `Op::MoeFfn`, ready to read for the experts in `routed`.
    ///
    /// A bank paged per expert gets exactly those experts pinned, and nothing else read; any other
    /// bank is read whole, as before. The pins of every bank go out in one wave on
    /// [`infr_core::blockio::IO_FANOUT`] threads of their own, so a decode step's misses — a few
    /// experts in each of three banks — are read concurrently rather than one after another. Not
    /// on the compute pool: a run held to one compute thread (`-t 1`) would otherwise read one
    /// expert at a time. More lanes than that would only queue on the file's handles.
    ///
    /// Which end of the recency order they go to is [`paged::expert_insert`]'s call: the most
    /// recent for a decode step, the cold end for a chunk that sweeps most of every bank.
    fn moe_banks<'b>(
        &self,
        bufs: &[&'b CpuBuffer],
        n_expert: usize,
        routed: &[usize],
        reuse: infr_core::graph::WeightReuse,
    ) -> Result<Vec<Bank<'b>>> {
        let mut jobs: Vec<(usize, usize)> = Vec::new();
        let mut store: Option<&'b paged::PagedWeights> = None;
        for (i, b) in bufs.iter().enumerate() {
            if let CpuStore::PagedExperts { store: s, bank, .. } = &b.store {
                if bank.n_expert() != n_expert {
                    return Err(infr_core::error::Error::backend(format!(
                        "cpu Op::MoeFfn: a bank registered with {} experts is routed over {n_expert}",
                        bank.n_expert()
                    )));
                }
                jobs.extend(routed.iter().map(|&e| (i, e)));
                store = Some(s);
            }
        }
        let mut pinned: Vec<Vec<Option<infr_core::hostpager::Pin<'b>>>> = bufs
            .iter()
            .map(|b| match b.store {
                CpuStore::PagedExperts { .. } => (0..n_expert).map(|_| None).collect(),
                _ => Vec::new(),
            })
            .collect();
        if let Some(store) = store {
            let t0 = std::time::Instant::now();
            let insert = paged::expert_insert(routed.len(), n_expert, reuse);
            let lanes = infr_core::blockio::IO_FANOUT.min(jobs.len());
            let jobs = &jobs;
            let pin_lane = &|lane: usize| {
                jobs.iter()
                    .skip(lane)
                    .step_by(lanes)
                    .map(|&(i, e)| {
                        let CpuStore::PagedExperts { store, bank, .. } = &bufs[i].store else {
                            unreachable!("only per-expert banks were queued");
                        };
                        (i, e, store.pin_expert(*bank, e, insert))
                    })
                    .collect::<Vec<_>>()
            };
            // The calling thread takes lane 0, so a single-lane wave spawns nothing.
            let got = std::thread::scope(|s| {
                let others: Vec<_> = (1..lanes)
                    .map(|lane| s.spawn(move || pin_lane(lane)))
                    .collect();
                let mut got = pin_lane(0);
                for h in others {
                    // The lane's own panic, not a generic one: it names what failed.
                    got.extend(h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)));
                }
                got
            });
            store.note_wait(t0.elapsed());
            for (i, e, pin) in got {
                pinned[i][e] = Some(pin?);
            }
        }
        Ok(bufs
            .iter()
            .zip(pinned)
            .map(|(b, pins)| match b.store {
                CpuStore::PagedExperts { .. } => Bank::Experts(pins),
                _ => {
                    let bytes = b.read();
                    let stride = bytes.len() / n_expert;
                    #[cfg(test)]
                    let stride = if self
                        .double_whole_bank_stride
                        .load(std::sync::atomic::Ordering::Relaxed)
                    {
                        2 * stride
                    } else {
                        stride
                    };
                    Bank::Whole { bytes, stride }
                }
            })
            .collect())
    }

    /// Fetch-or-build a dequantized weight from `weight_cache` under its `(buffer uid, dtype)` key
    /// (see `weight_cache`'s doc). `bytes` is dequantized per `key.1` on a miss.
    pub(crate) fn cached_weight(&self, key: (u64, DType), bytes: &[u8]) -> Arc<Vec<f32>> {
        if let Some(w) = self.weight_cache.lock().unwrap().get(&key) {
            return w.clone();
        }
        let w = Arc::new(bytes_to_f32(bytes, key.1));
        self.weight_cache.lock().unwrap().insert(key, w.clone());
        w
    }

    /// [`math::rope_freqs`]`(theta, rope_dim)` from `rope_freq_cache`, built on the first use.
    fn rope_freqs(&self, theta: f32, rope_dim: usize) -> Arc<[f32]> {
        self.rope_freq_cache
            .lock()
            .unwrap()
            .entry((theta.to_bits(), rope_dim))
            .or_insert_with(|| math::rope_freqs(theta, rope_dim).into())
            .clone()
    }
}

/// Reinterpret raw buffer bytes as `f32` values per `dtype` (dequantizing quant/f16/bf16, widening
/// integer position tensors). The universal "read a tensor's value on the host".
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn bytes_to_f32(bytes: &[u8], dtype: DType) -> Vec<f32> {
    // Decoded per element, never cast: a weight slice of the GGUF mapping is only as aligned as the
    // file's `general.alignment`, which may be below 4.
    match dtype {
        DType::I32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c) as f32)
            .collect(),
        DType::U32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c) as f32)
            .collect(),
        // F32 / F16 / Bf16 / all quant + codebook types go through the shared host dequant.
        other => dequant_block(other, bytes).expect("cpu backend: host dequant"),
    }
}

/// Weight rows one task of [`CpuBackend::add_hi`] takes on an o-major output.
const HI_ROWS: usize = 64;

/// Fewest weight rows a single-token `Op::Linear` task claims: a row's dot is well under a
/// microsecond, so fewer would make the shared task cursor the bottleneck.
const DECODE_ROWS_MIN: usize = 16;
/// Most weight rows a single-token `Op::Linear` task claims. Measured on Qwen3-1.7B Q4_K_M
/// decode at 16 threads (9950X3D): a fixed 16 rows per task lost to a fixed 128, a fixed 256 or
/// 512 lost to 128 again (too few tasks to balance), and per-shape sizing capped here matched
/// or beat the fixed 128 on both Qwen3 and LFM3 (CHANGELOG).
const DECODE_ROWS_MAX: usize = 128;

/// Query rows one `Op::Attention` task runs for one head; see the dispatch there.
const ATTN_QUERY_BLOCK: usize = 16;

/// Whether a prefill `Op::Linear` of `m` tokens over this weight runs on
/// [`CpuBackend::linear_packed`]: Q4_K, Q5_K or Q6_K with at least one full 8-row group, at
/// least [`repack::packed_min_tokens`] tokens, at the AVX-512 VNNI tier.
#[cfg(target_arch = "x86_64")]
fn linear_packed_ok<const CAP: bool>(dt: DType, in_f: usize, out_f: usize, m: usize) -> bool {
    matches!(dt, DType::Q4K | DType::Q5K | DType::Q6K)
        && out_f >= 8
        && in_f.is_multiple_of(256)
        && m >= repack::packed_min_tokens(dt)
        && simd::vnni::<CAP>()
}

/// The packed GEMM is x86_64-only, so off it no Linear takes that route.
#[cfg(not(target_arch = "x86_64"))]
const fn linear_packed_ok<const CAP: bool>(
    _dt: DType,
    _in_f: usize,
    _out_f: usize,
    _m: usize,
) -> bool {
    false
}

/// An F32 weight row as `f32`s: borrowed in place when it is 4-aligned (the hot path — every
/// real GGUF's alignment is a multiple of 4), copied through [`bytes_to_f32`] when the file's
/// `general.alignment` left it misaligned, where a cast would panic.
pub(crate) fn f32_row(row: &[u8]) -> std::borrow::Cow<'_, [f32]> {
    match bytemuck::try_cast_slice(row) {
        Ok(w) => std::borrow::Cow::Borrowed(w),
        Err(_) => std::borrow::Cow::Owned(bytes_to_f32(row, DType::F32)),
    }
}

/// Borrow two *distinct* elements of a slice as `(&a, &mut b)` — used by `Op::Copy`/`CopyStrided`
/// to read the source in place instead of cloning the whole tensor. Panics if `a == b` (callers
/// handle the aliasing case separately with a temp).
fn borrow_two_mut<T>(v: &mut [T], a: usize, b: usize) -> (&T, &mut T) {
    assert_ne!(a, b, "borrow_two_mut requires distinct indices");
    if a < b {
        let (lo, hi) = v.split_at_mut(b);
        (&lo[a], &mut hi[0])
    } else {
        let (lo, hi) = v.split_at_mut(a);
        (&hi[0], &mut lo[b])
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
/// Whether this CPU takes the fused Q4_K expert path whose packs fill the repack cache
/// (`kernels.cpu.repack_mb`): the VNNI GEMM, which needs the AVX-512 VNNI tier (under the
/// `kernels.cpu.max_tier` ceiling the backends built so far set). A model with a fused Q4_K
/// gate_up bank can then hold up to that budget in packs on top of its weights, which the seam
/// prices when it sizes a host weight arena.
pub fn q4k_expert_repack_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        simd::vnni::<true>()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn cpu_buf(b: &dyn Buffer) -> &CpuBuffer {
    b.as_any()
        .downcast_ref::<CpuBuffer>()
        .expect("cpu backend: buffer is not a CpuBuffer (mixed backends?)")
}

/// Refuse a graph that binds one OWNED buffer to `held` and to any of `others`, for an op that
/// keeps `held` locked while it locks those. The second lock would deadlock on the first, with no
/// message: an owned buffer is a `std` mutex, which is not reentrant. Mapped and paged buffers
/// take no lock, so they are not compared.
///
/// # Panics
/// Naming the op and both operands, when two of them are the same owned buffer.
fn refuse_relock(op: &str, held: (&str, &dyn Buffer), others: &[(&str, Option<&dyn Buffer>)]) {
    let owned_uid = |b: &dyn Buffer| {
        let b = cpu_buf(b);
        matches!(b.store, CpuStore::Owned(_)).then_some(b.uid)
    };
    let Some(uid) = owned_uid(held.1) else {
        return;
    };
    for (name, other) in others {
        if other.is_some_and(|o| owned_uid(o) == Some(uid)) {
            panic!(
                "cpu {op}: {} and {name} are bound to the same owned buffer, which the op would \
                 lock twice at once (a deadlock); bind each to its own buffer",
                held.0
            );
        }
    }
}

/// Dequantize the first `need` elements of a one-row-per-token KV cache at its declared `dtype`.
///
/// The DeepSeek ops (`Op::Mla`, `Op::LightningIndexer`) read their cache straight out of the bound
/// buffer rather than through the interpreter's f32 store, and both want only the `kv_len` rows in
/// use — never the whole `max_ctx`-row allocation, which at DeepSeek's 576-wide latent is most of
/// the cost of a decode step. Every dtype `WriteKv` can produce is covered; anything else is read
/// as plain f32.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn dequant_cache_prefix(b: &[u8], dtype: DType, need: usize) -> Vec<f32> {
    match dtype {
        DType::F16 => bytemuck::cast_slice::<u8, u16>(b)[..need]
            .iter()
            .map(|&x| half::f16::from_bits(x).to_f32())
            .collect(),
        DType::Q8_0 => crate::dequant_prefix_q8_0(b, need),
        dt @ (DType::Turbo2 | DType::Turbo3 | DType::Turbo4) => {
            crate::turbo::dequant_prefix_orig(dt, b, need)
        }
        DType::Bf16 | DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Iq4Nl => {
            let pb = infr_gguf::nbytes(dtype, need);
            infr_gguf::dequant::dequant_block(dtype, &b[..pb]).expect("cpu backend: KV dequant")
        }
        _ => bytemuck::cast_slice::<u8, f32>(b)[..need].to_vec(),
    }
}

/// Dequantize the first `need` elements of a Q8_0-block buffer (34 B / 32 elems, y = d*q).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn dequant_prefix_q8_0(bytes: &[u8], need: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(need);
    for b in 0..need.div_ceil(32) {
        let off = b * 34;
        let d = half::f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        for i in 0..32 {
            if out.len() == need {
                break;
            }
            out.push(d * (bytes[off + 2 + i] as i8) as f32);
        }
    }
    out
}

/// Refuse a stream count the fixed-size per-token scratch every backend uses cannot hold. See
/// `infr_core::graph::HYPER_CONNECT_MAX_MULT`; `what` names the op for the panic message.
fn hyper_check_hc(hc: usize, what: &str) {
    let max = infr_core::graph::HYPER_CONNECT_MAX_MULT as usize;
    assert!(
        hc >= 1 && hc <= max,
        "{what}: hc_mult {hc} outside the supported 1..={max} (every shipped DeepSeek V4 config \
         uses 4; the GPU kernels hold a token's whole hc*hc mixing matrix in a fixed-size array)"
    );
}

/// One token's Sinkhorn normalisation for `Op::HyperConnectMix`: in place over the `hc*hc` logit
/// matrix `m`, laid out `dst + hc*src` so `dst` is the fast axis (ggml's `[dst_hc, src_hc]`).
///
/// Transcribed from `deepseek4.cpp`'s `build_hc_sinkhorn`: softmax over `dst`, `+ eps` on every
/// element, then ONE normalisation over `src` followed by `n_iter - 1` (over-`dst`, over-`src`)
/// rounds. The asymmetry is the reference's (its loop starts at `i = 1`), and the `+ eps` on each
/// sum before it divides is the second and third of the three eps sites — see
/// `Op::HyperConnectMix`'s doc. Both reductions run their axis ASCENDING, `ggml_sum_rows`' order.
///
/// Named for the axis each normalisation REDUCES over, which is the opposite of what llama.cpp's
/// `norm_cols`/`norm_rows` lambdas are called there.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn hyper_sinkhorn(m: &mut [f32], hc: usize, eps: f32, n_iter: u32) {
    // Softmax over dst — a contiguous run of `hc` per src column — then the post-softmax eps.
    for src in 0..hc {
        let col = &mut m[src * hc..][..hc];
        let mx = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        math::exp_sub_slice(col, mx);
        for v in col.iter() {
            sum += *v;
        }
        for v in col.iter_mut() {
            *v = *v / sum + eps;
        }
    }
    // Divide each dst row by its sum over src.
    let norm_src = |m: &mut [f32]| {
        for dst in 0..hc {
            let mut s = 0f32;
            for src in 0..hc {
                s += m[dst + hc * src];
            }
            let d = s + eps;
            for src in 0..hc {
                m[dst + hc * src] /= d;
            }
        }
    };
    // Divide each src column by its sum over dst.
    let norm_dst = |m: &mut [f32]| {
        for src in 0..hc {
            let col = &mut m[src * hc..][..hc];
            let d = col.iter().sum::<f32>() + eps;
            for v in col.iter_mut() {
                *v /= d;
            }
        }
    };
    norm_src(m);
    for _ in 1..n_iter {
        norm_dst(m);
        norm_src(m);
    }
}

/// Device-side stochastic sampling reference (see `Op::Sample`) — IDENTICAL order of operations to
/// the host `sample_logits` (top-k select desc, softmax(temp), nucleus, CDF walk) with the uniform
/// draw `uu` factored out. `top_k == 0` is the "disable top-k" sentinel → NO truncation (`k == len`);
/// any `top_k >= 1` truncates to `min(top_k, len)`. Returns the sampled token index.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn sample_token(logits: &[f32], uu: f32, top_k: usize, temp: f32, top_p: f32) -> u32 {
    // top_k == 0 disables truncation; otherwise cap at the vocab length. Guarding here keeps the
    // `select_nth_unstable_by(k - 1)` below from underflowing when top_k == 0 (k would be 0).
    let k = if top_k == 0 {
        logits.len()
    } else {
        top_k.min(logits.len())
    };
    // `total_cmp`, not `partial_cmp` with an `Equal` fallback: that is not a total order once a
    // NaN is present, and `select_nth_unstable_by` / `sort_unstable_by` panic on one. On NaN-free
    // logits the two agree on every pair but `-0.0` / `+0.0`, which `total_cmp` orders. A NaN
    // logit no longer panics but ranks first, so the probabilities come out NaN as before.
    // Ties go to the lower index: both sorts are unstable, so without it two equal logits could
    // come back in either order, and which one a given `u` picks would depend on the standard
    // library's sort rather than on the input.
    let cmp = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then_with(|| a.cmp(b));
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    if k < logits.len() {
        idx.select_nth_unstable_by(k - 1, cmp);
        idx.truncate(k);
    }
    idx.sort_unstable_by(cmp);
    let maxl = logits[idx[0]];
    let mut probs: Vec<f32> = idx.iter().map(|&i| (logits[i] - maxl) / temp).collect();
    math::exp_slice(&mut probs);
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let mut cum = 0.0;
    let mut cutoff = probs.len();
    for (j, &p) in probs.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cutoff = j + 1;
            break;
        }
    }
    let total: f32 = probs[..cutoff].iter().sum();
    let r = uu * total;
    let mut tok = idx[cutoff - 1] as u32;
    let mut acc = 0.0;
    for j in 0..cutoff {
        acc += probs[j];
        if r <= acc {
            tok = idx[j] as u32;
            break;
        }
    }
    tok
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    /// The `Op::DeltaNet` arm slices q/k/v out of the shared row when `src_stride > 0`.
    fn deltanet_strided(&self) -> bool {
        true
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            name: "cpu-reference".into(),
            // The scalar interpreter has no GPU-kernel tier choice; these per-type feature flags
            // don't gate any CPU op handler (they select Vulkan/Metal shaders). f16/i8 math is
            // available in the interpreter; the matrix/dot-primitive shapes are N-A → None/off.
            f16: true,
            coopmat_f16: None,
            f8: false,
            coopmat_f8: None,
            i8: true,
            i8_dot: false,
            coopmat_i8: None,
            bf16: false,
            coopmat_bf16: None,
            subgroup_min: 0,
            subgroup_max: 0,
            sg_pref: 0, // no subgroup pinning on the scalar interpreter
            // Device-class fields: there is no GPU here, so no watchdog and nothing to bound.
            // (`integrated` means "an iGPU whose submits must stay under a TDR" — NOT "shares
            // system memory", which is what `unified_memory` below already says.)
            integrated: false,
            compute_units: 0,
            buffer_device_address: false, // CPU interpreter has no device-address concept
            max_shared_memory_bytes: u32::MAX, // scalar interpreter: no shared-memory tiling
            unified_memory: true,
            // The interpreter reads the baked `pos`/`kv_len` from the graph ops, so the decode graph
            // must be rebuilt per token — no record-once replay.
            decode_replay: false,
            combined_gu: false,
            embed_gather: true,
            gpu_sample: true,
            argmax_rows: true,
            argmax_prob: true,
            // The interpreter implements `Op::GatedRmsNorm` (below) for direct-op parity tests,
            // but leaves the capability false so the runner keeps emitting the split
            // `QkNorm`→`GatedAct` pair here — no GPU barrier to save on a scalar interpreter, and
            // it keeps the CPU oracle's op sequence matching the pre-fusion baseline.
            gated_rmsnorm: false,
            // The interpreter's WriteKv/Attention honor ring KV caches (row = pos % cap_rows), so
            // the runner may window-size SWA layers' caches here.
            kv_swa_ring: true,
            // `execute`'s write-back loop copies every mutated f32 `Input` back into its bound
            // buffer (`infr_core::exec::writes_back`), so a value the ops advanced survives.
            graph_input_inplace: true,
        }
    }

    fn alloc(&self, bytes: usize, _usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        Ok(Box::new(CpuBuffer::owned_buf(vec![0u8; bytes.max(4)])))
    }

    fn alloc_uninit(&self, bytes: usize, _usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        // Debug: poison with 0xFF (= NaN as f32) so a read-before-write surfaces loudly in the CPU
        // tests/oracle instead of silently working. Release: the Vec is zeroed anyway (no CPU perf
        // win to skip it), so stay safe.
        let fill = if cfg!(debug_assertions) { 0xFFu8 } else { 0u8 };
        Ok(Box::new(CpuBuffer::owned_buf(vec![fill; bytes.max(4)])))
    }

    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        let mut d = cpu_buf(dst).owned();
        d[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        let s = cpu_buf(src).read();
        dst.copy_from_slice(&s[..dst.len()]);
        Ok(())
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        Ok(GraphPlan::boxed(graph))
    }

    /// Runs the graph through `CpuBackend::execute_with`'s uncapped compilation unless a
    /// `kernels.cpu.max_tier` ceiling is in force (`crate::simd`): the one place the ceiling is
    /// consulted for a whole graph.
    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        if simd::capped() {
            self.execute_with::<true>(plan, bindings)
        } else {
            self.execute_with::<false>(plan, bindings)
        }
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }

    /// The arena the first paged buffer among `weights` reads from: a model's paged weights all
    /// share one ([`paged::PagedWeights::streams`] says whether it streams).
    fn host_streams(&self, weights: &mut dyn Iterator<Item = &dyn Buffer>) -> bool {
        for b in weights {
            let Some(cb) = b.as_any().downcast_ref::<CpuBuffer>() else {
                continue;
            };
            match &cb.store {
                CpuStore::Paged { store, .. } | CpuStore::PagedExperts { store, .. } => {
                    return store.streams();
                }
                CpuStore::Owned(_) | CpuStore::Mapped(_) => {}
            }
        }
        false
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl CpuBackend {
    /// The graph interpreter, compiled once per `CAP` (`crate::simd`): every SIMD dispatch in it
    /// takes `::<CAP>`, so the uncapped compilation decides the kernel tier from the CPU's
    /// features alone (a constant on a build that has them) and the capped one asks the
    /// `kernels.cpu.max_tier` ceiling at every dispatch, on every thread.
    fn execute_with<const CAP: bool>(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        let g = &plan
            .as_any()
            .downcast_ref::<GraphPlan>()
            .expect("cpu backend: plan is not a GraphPlan")
            .graph;
        // The interpreter never replays, and a graph built for replay is wrong when walked: every
        // execute would decode at the baked position (see `Graph::replay_only`).
        if g.replay_only {
            return Err(infr_core::error::Error::Backend(
                "cpu backend: refusing a graph built for record-once decode replay \
                 (Graph::replay_only): this backend walks it statically, which would decode every \
                 token at its baked position"
                    .into(),
            ));
        }
        // The f32 dots' compilation, chosen once for the whole graph and copied into every op's
        // loops (`kernels::Dots`).
        let dots = Dots::resolve::<CAP>();

        // The op loop runs on this thread, which takes a share of every pool job: give it the
        // pool's placement for the walk. A graph with a Linear is about to use the pool, so it is
        // built here if it does not exist yet; one without leaves an unbuilt pool unbuilt (the
        // scan is once per walk, not per op).
        let pool_in_use =
            self.pool.get().is_some() || g.ops.iter().any(|op| matches!(op, Op::Linear { .. }));
        let _placed = pool_in_use.then(|| self.pool().place_caller()).flatten();
        #[cfg(test)]
        tests::EXEC_PLACEMENT.set(infr_plat::cpu::current_placement());

        // f32 working store for every Input/Internal/Output handle (weights are read on demand:
        // norms via the small dequant cache, `Op::Linear` weights streamed row-by-row).
        let mut vals: Vec<Vec<f32>> = vec![Vec::new(); g.tensors.len()];
        // Large op outputs come out of (and replaced tensors go back into) this pool, so a
        // layer reuses the buffers the previous layer freed instead of faulting in fresh pages.
        let mut rc = recycle::Recycler::default();
        // KV-cache tensors (the `cache` of `WriteKv`, the `k_cache`/`v_cache` of `Attention`) are
        // accessed straight from their bound buffers — `WriteKv` writes one row, `Attention` reads
        // `kv_len` rows. They're sized for the WHOLE context (`max_ctx`), so loading them into `vals`
        // (and writing them back) each token would cost O(max_ctx) memory traffic per token instead of
        // O(kv_len) — catastrophic at a large `max_new`. Skip the round-trip for them. Which tensors
        // are written in place is graph semantics, computed once by `Graph::in_place_inputs`; the
        // per-kind classification is the shared `infr_core::exec::Provision` (metal's interpreter
        // makes the identical calls).
        for (id, decl, p) in infr_core::exec::provisions(g) {
            let i = id.0 as usize;
            match p {
                Provision::Zero => vals[i] = rc.zeroed(decl.desc.numel()),
                Provision::Load => {
                    let buf = bindings.get(id).expect("cpu backend: unbound Input");
                    let bytes = cpu_buf(buf).read();
                    vals[i] = bytes_to_f32(&bytes, decl.desc.dtype);
                }
                // A Weight (lazily dequantized in `weight()`) or an in-place KV Input.
                Provision::Skip => {}
            }
        }

        // Fetch a (cached) dequantized weight.
        let weight = |id: TensorId| -> Arc<Vec<f32>> {
            let buf = bindings.get(id).expect("cpu backend: unbound Weight");
            let key = (cpu_buf(buf).uid, g.desc(id).dtype);
            if let Some(w) = self.weight_cache.lock().unwrap().get(&key) {
                return w.clone();
            }
            let bytes = cpu_buf(buf).read();
            self.cached_weight(key, &bytes)
        };

        // The knobs this interpreter loop consults, read ONCE off the backend's config: the loop
        // below runs them per op (`reference`) or per MoeFfn op (`moe_counts`), so a resolve per
        // iteration would be a cost the env reads never had either.
        // The shared predicate (`INFR_PROF_OPS` or `INFR_PROF_OPS`, ANDed with warmup suppression) —
        // this backend used to read `prof.prof_ops` alone, so a bench that suppressed profiling
        // around its untimed warmup still got that forward in the table.
        let prof_ops = infr_core::prof::enabled(&self.cfg.prof);
        let reference = self.cfg.kernels.cpu.reference;
        let moe_counts = self.cfg.debug.moe_counts;
        let moe_counts_dump = self.cfg.debug.moe_counts_dump;
        // Labeled with the shared grammar (`infr_core::prof::op_label`), so a cpu-vs-GPU
        // comparison lines up row for row instead of comparing a bare `Linear` against
        // `Linear m=512 1536x1536 Q4K`.
        let mut op_times: HashMap<String, (u64, f64)> = HashMap::new();
        // `WeightReuse::Done`: the experts each paged bank keeps out of the demotion, by the bank
        // buffer's uid ([`paged::done_keep`]).
        let mut kept_experts: HashMap<u64, Vec<usize>> = HashMap::new();
        // The per-op hash tap: `None` unless `debug.cpu_op_hash` is set. An op's writes are
        // recorded at the top of the NEXT iteration (and after the loop for the last op), so an
        // arm that `continue`s out of the match is still covered.
        let mut hash_tap = self
            .cfg
            .debug
            .cpu_op_hash
            .as_deref()
            .map(|path| op_hash::OpHashTap::new(path, &self.op_hash));
        let tap_hash = |vals: &[Vec<f32>], id: TensorId| -> (usize, u64) {
            if g.in_place_inputs().contains(&id) {
                match bindings.get(id) {
                    Some(b) => (
                        g.desc(id).numel(),
                        infr_core::kernel_cache::fnv1a(&cpu_buf(b).read()),
                    ),
                    None => (0, infr_core::kernel_cache::fnv1a(&[])),
                }
            } else {
                op_hash::hash_f32(&vals[id.0 as usize])
            }
        };
        for (op_idx, op) in g.ops.iter().enumerate() {
            if let Some(tap) = hash_tap.as_mut() {
                if let Some(prev) = op_idx.checked_sub(1) {
                    tap.record(g, prev, &g.ops[prev], true, |id| tap_hash(&vals, id));
                }
                tap.record(g, op_idx, op, false, |id| tap_hash(&vals, id));
            }
            // Make every PAGED weight this op reads resident, and hold it resident for the op's
            // whole body. This is the one fallible point in the interpreter, which is exactly why
            // it is here and not inside `CpuBuffer::read`: that returns a view, not a `Result`, so
            // an I/O failure there could only panic. The guards drop at the end of the iteration.
            // Empty (and free) for a model with no paged weights, which is every model until
            // `paging.dram` is set.
            let _pins = self.pin_op_weights(op, g, bindings)?;
            let __t0 = if prof_ops {
                Some(std::time::Instant::now())
            } else {
                None
            };
            match *op {
                Op::RmsNorm {
                    x,
                    weight: w,
                    dst,
                    rows,
                    dim,
                    eps,
                } => {
                    let (rows, dim) = (rows as usize, dim as usize);
                    let xs = &vals[x.0 as usize];
                    let ws = weight(w);
                    // Rows are independent (no cross-row reduction) — spin-pool over rows, same
                    // per-row math and order as before (bit-identical, just distributed).
                    let mut out = rc.for_overwrite(rows * dim);
                    self.pool().for_chunks_mut(&mut out, dim, 4, &|r, orow| {
                        let xrow = &xs[r * dim..r * dim + dim];
                        let ss: f32 = (0..dim).map(|i| xrow[i] * xrow[i]).sum::<f32>() / dim as f32;
                        let s = 1.0 / (ss + eps).sqrt();
                        for i in 0..dim {
                            orow[i] = xrow[i] * s * ws[i];
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::RmsNormAdd {
                    x,
                    weight: w,
                    dst,
                    rows: _rows,
                    dim,
                    eps,
                } => {
                    let dim = dim as usize;
                    let xs = &vals[x.0 as usize];
                    let ws = weight(w);
                    let mut dst_vals = rc.copy_of(&vals[dst.0 as usize]);
                    self.pool()
                        .for_chunks_mut(&mut dst_vals, dim, 4, &|r, drow| {
                            let xrow = &xs[r * dim..r * dim + dim];
                            let ss: f32 =
                                (0..dim).map(|i| xrow[i] * xrow[i]).sum::<f32>() / dim as f32;
                            let s = 1.0 / (ss + eps).sqrt();
                            for i in 0..dim {
                                drow[i] += xrow[i] * s * ws[i];
                            }
                        });
                    rc.set(&mut vals, dst.0 as usize, dst_vals);
                }
                Op::LayerNorm {
                    x,
                    weight: w,
                    bias,
                    dst,
                    rows,
                    dim,
                    eps,
                } => {
                    let (rows, dim) = (rows as usize, dim as usize);
                    let xs = &vals[x.0 as usize];
                    let ws = weight(w);
                    let bs = weight(bias);
                    // Mean-centred norm (ggml_norm): biased variance, eps inside the sqrt, then
                    // `* weight + bias`. Rows are independent — spin-pool over rows like RmsNorm.
                    let mut out = rc.for_overwrite(rows * dim);
                    self.pool().for_chunks_mut(&mut out, dim, 4, &|r, orow| {
                        let xrow = &xs[r * dim..r * dim + dim];
                        let mean = xrow.iter().sum::<f32>() / dim as f32;
                        let var =
                            xrow.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
                        let s = 1.0 / (var + eps).sqrt();
                        for i in 0..dim {
                            orow[i] = (xrow[i] - mean) * s * ws[i] + bs[i];
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::Softmax {
                    x,
                    dst,
                    rows,
                    dim,
                    scale,
                    scale_buf,
                } => {
                    let (rows, dim) = (rows as usize, dim as usize);
                    // Perf (DiffusionGemma denoise, Vulkan) — see `Op::Softmax::scale_buf`'s doc.
                    // CPU never builds a graph with this set (the CPU denoise path keeps its host
                    // self-conditioning), but implements it generically rather than panicking.
                    let scale = match scale_buf {
                        Some(sb) => vals[sb.0 as usize][0],
                        None => scale,
                    };
                    let xs = &vals[x.0 as usize];
                    let mut out = rc.zeroed(rows * dim);
                    self.pool().for_chunks_mut(&mut out, dim, 1, &|r, o| {
                        let row = &xs[r * dim..r * dim + dim];
                        let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v * scale));
                        for (dst_v, &v) in o.iter_mut().zip(row) {
                            *dst_v = v * scale - mx;
                        }
                        math::exp_slice(o);
                        let mut denom = 0f32;
                        for &e in o.iter() {
                            denom += e;
                        }
                        let inv = 1.0 / denom;
                        for dst_v in o.iter_mut() {
                            *dst_v *= inv;
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::QkNorm {
                    x,
                    weight: w,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    eps,
                    x_stride,
                } => {
                    // Strided reads (qwen35's q+g interleave) are wired only into QkNormRope, and
                    // no emitter strides this op; refuse rather than read the rows as packed.
                    assert_eq!(x_stride, 0, "QkNorm: strided x is not supported");
                    let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                    let xs = &vals[x.0 as usize];
                    // `None` = V4's weightless Q norm (bare `ggml_rms_norm` per head): the scale is
                    // the only factor. The reduction is identical either way.
                    let ws = w.map(weight);
                    let mut out = rc.zeroed(rows * nh * hd);
                    for r in 0..rows {
                        for h in 0..nh {
                            let b = (r * nh + h) * hd;
                            let ss: f32 =
                                (0..hd).map(|i| xs[b + i] * xs[b + i]).sum::<f32>() / hd as f32;
                            let s = 1.0 / (ss + eps).sqrt();
                            match &ws {
                                Some(ws) => {
                                    for i in 0..hd {
                                        out[b + i] = xs[b + i] * s * ws[i];
                                    }
                                }
                                None => {
                                    for i in 0..hd {
                                        out[b + i] = xs[b + i] * s;
                                    }
                                }
                            }
                        }
                    }
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                // Fused per-head RMSNorm + SiLU gate multiply (qwen35 DeltaNet z-gate). Reduction
                // structure is IDENTICAL to the `QkNorm` arm above (bit-stable normalization); the
                // gate multiply is the extra elementwise step `GatedAct(Silu)` would otherwise do
                // as a second op.
                Op::GatedRmsNorm {
                    x,
                    weight: w,
                    gate,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    eps,
                } => {
                    let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                    let xs = &vals[x.0 as usize];
                    let zs = &vals[gate.0 as usize];
                    let ws = weight(w);
                    let mut out = rc.zeroed(rows * nh * hd);
                    let mut silu = vec![0f32; hd];
                    for r in 0..rows {
                        for h in 0..nh {
                            let b = (r * nh + h) * hd;
                            let ss: f32 =
                                (0..hd).map(|i| xs[b + i] * xs[b + i]).sum::<f32>() / hd as f32;
                            let s = 1.0 / (ss + eps).sqrt();
                            silu.copy_from_slice(&zs[b..b + hd]);
                            math::silu_slice(&mut silu);
                            for i in 0..hd {
                                let normed = xs[b + i] * s * ws[i];
                                out[b + i] = normed * silu[i];
                            }
                        }
                    }
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::Linear {
                    x,
                    weight: w,
                    dst,
                    m,
                    in_f,
                    out_f,
                    w_off,
                } => {
                    let (m, in_f, out_f) = (m as usize, in_f as usize, out_f as usize);
                    let xs = &vals[x.0 as usize];
                    // Stream the (row-major [out_f, in_f]) weight one row at a time straight from the
                    // mmap, dequantizing inside the dot — no full f32 materialization. GGUF rows are
                    // block-aligned, so each row is an equal `bytes/out_f` slice. Output rows are
                    // independent → fan out over the 32 cores with rayon.
                    let buf = bindings.get(w).expect("cpu backend: unbound Weight");
                    let bytes = cpu_buf(buf).read();
                    let dt = g.desc(w).dtype;
                    // `w_off` (elements, row-aligned) selects a projection's rows inside a
                    // CONCATENATED weight (fused QKV): total rows = declared numel / in_f.
                    let total_rows = g.desc(w).numel() / in_f;
                    let bpr = bytes.len() / total_rows; // bytes per weight row
                    let row0 = w_off as usize / in_f;
                    let wbytes: &[u8] = &bytes[row0 * bpr..(row0 + out_f) * bpr];
                    let mut out = rc.for_overwrite(m * out_f);
                    // The test tap (`linear_tap`): the rows it wants, `None` when it is off.
                    let tap_rows = linear_tap::wanted(m);
                    let tap = |out: &[f32]| {
                        if let Some(rows) = &tap_rows {
                            self.tap_linear(op_idx, dt, wbytes, bpr, xs, out, m, in_f, out_f, rows);
                        }
                    };
                    // REFERENCE mode (see `CpuBackend::reference`): decode the weight row to f32 and
                    // take a plain f32 dot, so the ONLY error is the weight's own quantization —
                    // the int8-activation kernels below carry ~4e-3 relative error, which flips
                    // greedy tokens at Q2_K. The float dtypes already take an exact dot, so they
                    // stay on their (identical, faster) production arms.
                    if reference && !matches!(dt, DType::F32 | DType::F16 | DType::Bf16) {
                        // A test's simulated activation quantizer (`linear_tap::simulate`).
                        let sim_xs: Option<Vec<f32>> = linear_tap::simulated(dt, in_f)
                            .map(|sim| xs.chunks(in_f).flat_map(|r| sim.round_trip(r)).collect());
                        let xd: &[f32] = sim_xs.as_deref().unwrap_or(xs);
                        let mut out_t = rc.zeroed(out_f * m);
                        self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                            let row = &wbytes[o * bpr..o * bpr + bpr];
                            let wf = bytes_to_f32(row, dt);
                            for r in 0..m {
                                chunk[r] = dots.dot(&wf, &xd[r * in_f..r * in_f + in_f]);
                            }
                        });
                        self.pool().for_chunks_mut(&mut out, out_f, 1, &|r, orow| {
                            for (o, dst) in orow.iter_mut().enumerate() {
                                *dst = out_t[o * m + r];
                            }
                        });
                        rc.give(out_t);
                        tap(&out);
                        rc.set(&mut vals, dst.0 as usize, out);
                        continue;
                    }
                    // One token (decode) is the hot path. Dispatch on the weight dtype to the fastest
                    // per-row kernel: integer Q8×Q4_K/Q6_K dots (quantize the activation once), direct
                    // f16/bf16/f32 dots, else fall back to dequant-to-f32 + dot. All fan out over rows.
                    if m == 1 {
                        let xrow = &xs[..in_f];
                        // Q8_0 is a NATIVE 32-block weight, so its `in_f` can be a multiple of 32
                        // but not 256 (e.g. gemma3-1b's 1152-dim projections). The 256-superblock
                        // Q8 activation path silently drops the sub-256 tail, so route those to the
                        // native 32-block Q8x32 kernel instead. Every other quant routed through Q8
                        // has a 256-element weight block, so its `in_f` is always 256-aligned.
                        let q8_0_blk32 = dt == DType::Q8_0 && !in_f.is_multiple_of(256);
                        let q8 = (matches!(
                            dt,
                            DType::Q4K
                                | DType::Q6K
                                | DType::Q8_0
                                | DType::Q5K
                                | DType::Iq4Xs
                                | DType::Iq2S
                                | DType::Iq2Xs
                                | DType::Iq2Xxs
                                | DType::Iq1S
                                | DType::Iq1M
                                | DType::Iq3S
                                | DType::Iq3Xxs
                                | DType::Q2K
                                | DType::Q3K
                                | DType::Tq2_0
                                | DType::Tq1_0
                        ) && !q8_0_blk32)
                            .then(|| quantize_q8(xrow));
                        // Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL (and sub-256 Q8_0) use the native-32-block
                        // int8 activation (Q8x32), not the 256-superblock Q8; quantize the single
                        // activation row once. Each of them runs the SAME batch kernel prefill
                        // runs, on a one-row batch, so a row's result does not depend on how
                        // many rows share its `Op::Linear`.
                        let q8_32: Option<[Q8x32; 1]> = (matches!(
                            dt,
                            DType::Q4_0
                                | DType::Q4_1
                                | DType::Q5_0
                                | DType::Q5_1
                                | DType::Iq4Nl
                                | DType::Mxfp4
                                | DType::Nvfp4
                                | DType::Q2_0
                        ) || q8_0_blk32)
                            .then(|| [quantize_q8_32(xrow)]);
                        // Rows per claimed task: enough tasks for every thread to take two,
                        // but no smaller than `DECODE_ROWS_MIN` (per-row claims would be all
                        // cursor contention) nor larger than `DECODE_ROWS_MAX`.
                        let grain = (out_f / (2 * self.pool().threads()))
                            .clamp(DECODE_ROWS_MIN, DECODE_ROWS_MAX);
                        self.pool().for_chunks_mut(&mut out, 1, grain, &|o, dst_o| {
                            let row = &wbytes[o * bpr..o * bpr + bpr];
                            // The int8 kernels on the 256-element activation: the dense dot,
                            // plus the sparse part's for a row that has one (`vec_dot_hi`).
                            if let Some(q) = q8.as_ref() {
                                let dense = vec_dot_q8_any::<CAP>(dt, row, q, in_f);
                                dst_o[0] = if q.hi.is_empty() {
                                    dense
                                } else {
                                    dense + vec_dot_hi(dt, row, q, in_f)
                                };
                                return;
                            }
                            dst_o[0] = match dt {
                                DType::Q8_0 => {
                                    vec_dot_q8_0_32_batch::<CAP>(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Q4_0 => {
                                    vec_dot_q4_0_32_batch::<CAP>(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Q4_1 => {
                                    vec_dot_q4_1_32_batch::<CAP>(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Q5_0 => {
                                    vec_dot_q5_0_32_batch::<CAP>(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Q5_1 => {
                                    vec_dot_q5_1_32_batch::<CAP>(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Iq4Nl => {
                                    vec_dot_iq4nl_32_batch::<CAP>(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Mxfp4 => {
                                    vec_dot_mxfp4_32_batch::<CAP>(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Q2_0 => {
                                    vec_dot_q2_0_batch::<CAP>(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Nvfp4 => {
                                    vec_dot_nvfp4_batch::<CAP>(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::F32 => dots.dot(&f32_row(row), xrow),
                                DType::F16 => dots.dot_f16(row, xrow),
                                DType::Bf16 => dots.dot_bf16(row, xrow),
                                _ => dots.dot(&bytes_to_f32(row, dt), xrow),
                            };
                        });
                    } else {
                        // PREFILL (m > 1): parallelize over output rows (one weight row per task).
                        // For quant types, use the batched dot kernels: the weight row is decoded
                        // ONCE per output row (inside the batch fn), then the integer dot is
                        // repeated across all m token activations — amortising the expensive
                        // nibble/bit unpacking that the single-token path was redoing m times.
                        //
                        // Layout: out[r * out_f + o].  We accumulate into a transposed buffer
                        // out_t[o * m + r] (contiguous in o-major order) so each parallel chunk
                        // owns a contiguous slice of m floats, then scatter into out at the end.
                        // Parallel: at m=256 canvas rows this collect was ~0.7 ms of SERIAL work
                        // per Linear (31 threads idle) — rows are independent, order preserved by
                        // the indexed collect, bit-identical.
                        // Q8_0 is a native 32-block weight: a sub-256 `in_f` (e.g. gemma3-1b's
                        // 1152-dim projections) must take the 32-block Q8x32 path below, not the
                        // 256-superblock Q8 batch kernel (which drops the sub-256 tail). Skip
                        // building the wasteful 256-block activation in that case.
                        let q8_0_blk32 = dt == DType::Q8_0 && !in_f.is_multiple_of(256);
                        let q8s: Vec<Q8> = if !q8_0_blk32
                            && matches!(
                                dt,
                                DType::Q4K
                                    | DType::Q6K
                                    | DType::Q8_0
                                    | DType::Q5K
                                    | DType::Iq4Xs
                                    | DType::Iq2S
                                    | DType::Iq2Xs
                                    | DType::Iq2Xxs
                                    | DType::Iq1S
                                    | DType::Iq1M
                                    | DType::Iq3S
                                    | DType::Iq3Xxs
                                    | DType::Q2K
                                    | DType::Q3K
                                    | DType::Tq2_0
                                    | DType::Tq1_0
                            ) {
                            self.pool()
                                .collect(m, &|r| quantize_q8(&xs[r * in_f..r * in_f + in_f]))
                        } else {
                            Vec::new()
                        };
                        // AVX-512 VNNI: Q4_K, Q5_K and Q6_K run the packed 8-row GEMM straight
                        // into the row-major output, with no o-major temp to transpose after.
                        if linear_packed_ok::<CAP>(dt, in_f, out_f, m) {
                            #[cfg(target_arch = "x86_64")]
                            self.linear_packed::<CAP>(dt, wbytes, in_f, out_f, &q8s, &mut out);
                            #[cfg(not(target_arch = "x86_64"))]
                            unreachable!("linear_packed_ok is false off x86_64");
                        } else {
                            let mut out_t = rc.zeroed(out_f * m);
                            // For Q4_K, use 8-row tiling: each rayon task handles 8 consecutive
                            // output rows and loads the Q8 activation zmm ONCE per (block, nibble-pair),
                            // reusing it across all 8 weight rows. This is 4× less activation traffic
                            // than the 2-row path and 8× less than the single-row path. Remainder rows
                            // (out_f % 8) fall through to the 2-row tile then the 1-row batch.
                            if dt == DType::Q4K && out_f >= 8 {
                                let groups8 = out_f / 8;
                                let rem = out_f % 8;
                                let (g8_t, rest_t) = out_t.split_at_mut(groups8 * 8 * m);
                                // 8-row groups across the spin-pool.
                                self.pool().for_chunks_mut(g8_t, 8 * m, 1, &|g, dc| {
                                    let o = g * 8;
                                    let (r0, rest) = dc.split_at_mut(m);
                                    let (r1, rest) = rest.split_at_mut(m);
                                    let (r2, rest) = rest.split_at_mut(m);
                                    let (r3, rest) = rest.split_at_mut(m);
                                    let (r4, rest) = rest.split_at_mut(m);
                                    let (r5, rest) = rest.split_at_mut(m);
                                    let (r6, r7) = rest.split_at_mut(m);
                                    vec_dot_q4k_batch8::<CAP>(
                                        [
                                            &wbytes[o * bpr..o * bpr + bpr],
                                            &wbytes[(o + 1) * bpr..(o + 1) * bpr + bpr],
                                            &wbytes[(o + 2) * bpr..(o + 2) * bpr + bpr],
                                            &wbytes[(o + 3) * bpr..(o + 3) * bpr + bpr],
                                            &wbytes[(o + 4) * bpr..(o + 4) * bpr + bpr],
                                            &wbytes[(o + 5) * bpr..(o + 5) * bpr + bpr],
                                            &wbytes[(o + 6) * bpr..(o + 6) * bpr + bpr],
                                            &wbytes[(o + 7) * bpr..(o + 7) * bpr + bpr],
                                        ],
                                        &q8s,
                                        in_f,
                                        [r0, r1, r2, r3, r4, r5, r6, r7],
                                    );
                                });
                                // Remainder: up to 7 rows → 2-row pairs, then at most 1 odd tail.
                                let pairs_rem = rem / 2;
                                let (g2_t, odd_t) = rest_t.split_at_mut(pairs_rem * 2 * m);
                                if pairs_rem > 0 {
                                    self.pool().for_chunks_mut(g2_t, 2 * m, 1, &|pair, dc| {
                                        let o = groups8 * 8 + pair * 2;
                                        let (chunk_a, chunk_b) = dc.split_at_mut(m);
                                        let row_a = &wbytes[o * bpr..o * bpr + bpr];
                                        let row_b = &wbytes[(o + 1) * bpr..(o + 1) * bpr + bpr];
                                        vec_dot_q4k_batch2::<CAP>(
                                            row_a, row_b, &q8s, in_f, chunk_a, chunk_b,
                                        );
                                    });
                                }
                                if rem % 2 != 0 {
                                    let o = out_f - 1;
                                    let row = &wbytes[o * bpr..o * bpr + bpr];
                                    vec_dot_q4k_batch::<CAP>(row, &q8s, in_f, odd_t);
                                }
                            } else if dt == DType::Q4K && out_f >= 2 {
                                // Small out_f < 8: fall back to 2-row tile.
                                let pairs = out_f / 2;
                                let (even_t, odd_t) = out_t.split_at_mut(pairs * 2 * m);
                                self.pool().for_chunks_mut(even_t, 2 * m, 1, &|pair, dc| {
                                    let o = pair * 2;
                                    let (chunk_a, chunk_b) = dc.split_at_mut(m);
                                    let row_a = &wbytes[o * bpr..o * bpr + bpr];
                                    let row_b = &wbytes[(o + 1) * bpr..(o + 1) * bpr + bpr];
                                    vec_dot_q4k_batch2::<CAP>(
                                        row_a, row_b, &q8s, in_f, chunk_a, chunk_b,
                                    );
                                });
                                if out_f % 2 != 0 {
                                    let o = out_f - 1;
                                    let row = &wbytes[o * bpr..o * bpr + bpr];
                                    vec_dot_q4k_batch::<CAP>(row, &q8s, in_f, odd_t);
                                }
                            } else if q8_0_blk32 {
                                // Q8_0 weight with a sub-256 `in_f` (multiple of 32 but not 256): the
                                // 256-superblock Q8 batch kernel would truncate the tail, so use the
                                // native 32-block Q8x32-activation batch kernel instead. `in_f % 32 == 0`
                                // always holds for a Q8_0 (32-block) weight.
                                let q8s32: Vec<Q8x32> = self.pool().collect(m, &|r| {
                                    quantize_q8_32(&xs[r * in_f..r * in_f + in_f])
                                });
                                self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                    let row = &wbytes[o * bpr..o * bpr + bpr];
                                    vec_dot_q8_0_32_batch::<CAP>(row, &q8s32, in_f, chunk);
                                });
                            } else if dt == DType::Q5_0 && in_f.is_multiple_of(32) {
                                // Dense-layer Q5_0 (DG stores 16 of its 30 dense ffn_down weights as
                                // Q5_0): reuse the MoE path's Q8x32-activation batch kernel. This
                                // dtype previously fell through to the dequant+f32-dot fallback below
                                // (~11% of every DG denoise step in the samply profile) — the switch
                                // puts it on the same int8-quantized-activation regime every other
                                // quantized dtype in this dispatch already uses.
                                let q8s32: Vec<Q8x32> = self.pool().collect(m, &|r| {
                                    quantize_q8_32(&xs[r * in_f..r * in_f + in_f])
                                });
                                self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                    let row = &wbytes[o * bpr..o * bpr + bpr];
                                    vec_dot_q5_0_32_batch::<CAP>(row, &q8s32, in_f, chunk);
                                });
                            } else if dt == DType::Q5_1 && in_f.is_multiple_of(32) {
                                // Q5_1 (affine sibling of Q5_0: `y = d·q5 + m`, 5-bit code + f16 `m`
                                // min, block stride 24): previously fell through to the dequant+f32
                                // fallback below. Reuse the Q8x32-activation batch kernel with the
                                // affine per-block form.
                                let q8s32: Vec<Q8x32> = self.pool().collect(m, &|r| {
                                    quantize_q8_32(&xs[r * in_f..r * in_f + in_f])
                                });
                                self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                    let row = &wbytes[o * bpr..o * bpr + bpr];
                                    vec_dot_q5_1_32_batch::<CAP>(row, &q8s32, in_f, chunk);
                                });
                            } else if dt == DType::Q4_0 && in_f.is_multiple_of(32) {
                                // Q4_0 (same 32-block int8-activation regime as Q5_0, minus the 5th
                                // bit and with offset 8): previously fell through to the dequant+f32
                                // fallback below. Reuse the Q8x32-activation batch kernel.
                                let q8s32: Vec<Q8x32> = self.pool().collect(m, &|r| {
                                    quantize_q8_32(&xs[r * in_f..r * in_f + in_f])
                                });
                                self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                    let row = &wbytes[o * bpr..o * bpr + bpr];
                                    vec_dot_q4_0_32_batch::<CAP>(row, &q8s32, in_f, chunk);
                                });
                            } else if dt == DType::Q4_1 && in_f.is_multiple_of(32) {
                                // Q4_1 (affine sibling of Q4_0: `y = d·q4 + m`, block stride 20):
                                // previously fell through to the dequant+f32 fallback below. Reuse the
                                // Q8x32-activation batch kernel with the affine per-block form.
                                let q8s32: Vec<Q8x32> = self.pool().collect(m, &|r| {
                                    quantize_q8_32(&xs[r * in_f..r * in_f + in_f])
                                });
                                self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                    let row = &wbytes[o * bpr..o * bpr + bpr];
                                    vec_dot_q4_1_32_batch::<CAP>(row, &q8s32, in_f, chunk);
                                });
                            } else if dt == DType::Iq4Nl && in_f.is_multiple_of(32) {
                                // IQ4_NL (flat 32-block codebook cousin of IQ4_XS: ONE f16 `d` per
                                // 32-weight block, signed KVALUES codebook, no super-block scales):
                                // previously fell through to the dequant+f32 fallback below. Reuse the
                                // Q8x32-activation batch kernel with the codebook signed-dot.
                                let q8s32: Vec<Q8x32> = self.pool().collect(m, &|r| {
                                    quantize_q8_32(&xs[r * in_f..r * in_f + in_f])
                                });
                                self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                    let row = &wbytes[o * bpr..o * bpr + bpr];
                                    vec_dot_iq4nl_32_batch::<CAP>(row, &q8s32, in_f, chunk);
                                });
                            } else if dt == DType::Mxfp4 && in_f.is_multiple_of(32) {
                                // MXFP4 (flat 32-block codebook, structurally identical to IQ4_NL but
                                // with a 1-byte E8M0 exponent scale `d = 2^(e−128)` per block and the
                                // KVALUES_MXFP4 codebook): previously fell through to the dequant+f32
                                // fallback below. Reuse the Q8x32-activation batch kernel.
                                let q8s32: Vec<Q8x32> = self.pool().collect(m, &|r| {
                                    quantize_q8_32(&xs[r * in_f..r * in_f + in_f])
                                });
                                self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                    let row = &wbytes[o * bpr..o * bpr + bpr];
                                    vec_dot_mxfp4_32_batch::<CAP>(row, &q8s32, in_f, chunk);
                                });
                            } else if dt == DType::Q2_0 && in_f.is_multiple_of(64) {
                                // Q2_0 ("Bonsai ternary", 2-bit linear-offset quant on a 64-weight
                                // block): previously fell through to the dequant+f32 fallback below.
                                // Reuse the Q8x32-activation batch kernel — each 64-weight block maps to
                                // two consecutive 32-element activation sub-blocks. Guarded on
                                // `is_multiple_of(64)` (the block size) so the sub-block count is even.
                                let q8s32: Vec<Q8x32> = self.pool().collect(m, &|r| {
                                    quantize_q8_32(&xs[r * in_f..r * in_f + in_f])
                                });
                                self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                    let row = &wbytes[o * bpr..o * bpr + bpr];
                                    vec_dot_q2_0_batch::<CAP>(row, &q8s32, in_f, chunk);
                                });
                            } else if dt == DType::Nvfp4 && in_f.is_multiple_of(64) {
                                // NVFP4 (E2M1 codebook + four per-16-element UE4M3 sub-block scales on a
                                // 64-weight block): previously fell through to the dequant+f32 fallback
                                // below. Reuse the Q8x32-activation batch kernel — each 64-weight block
                                // maps to two consecutive 32-element activation sub-blocks, and each of
                                // those spans two NVFP4 sub-blocks with distinct weight scales. Guarded on
                                // `is_multiple_of(64)` (the block size) so the sub-block count is even.
                                let q8s32: Vec<Q8x32> = self.pool().collect(m, &|r| {
                                    quantize_q8_32(&xs[r * in_f..r * in_f + in_f])
                                });
                                self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                    let row = &wbytes[o * bpr..o * bpr + bpr];
                                    vec_dot_nvfp4_batch::<CAP>(row, &q8s32, in_f, chunk);
                                });
                            } else {
                                // Grain 8: at lm_head shape this loop is 262k one-row chunks; per-row
                                // cursor claims (or rayon's per-item splitting) are real overhead.
                                self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                    let row = &wbytes[o * bpr..o * bpr + bpr];
                                    match dt {
                                        DType::Q4K => {
                                            vec_dot_q4k_batch::<CAP>(row, &q8s, in_f, chunk)
                                        }
                                        DType::Q6K => {
                                            vec_dot_q6k_batch::<CAP>(row, &q8s, in_f, chunk)
                                        }
                                        DType::Q8_0 => {
                                            vec_dot_q8_0_batch::<CAP>(row, &q8s, in_f, chunk)
                                        }
                                        DType::Q5K => {
                                            vec_dot_q5k_batch::<CAP>(row, &q8s, in_f, chunk)
                                        }
                                        DType::Q2K => {
                                            vec_dot_q2k_batch::<CAP>(row, &q8s, in_f, chunk)
                                        }
                                        DType::Q3K => {
                                            vec_dot_q3k_batch::<CAP>(row, &q8s, in_f, chunk)
                                        }
                                        DType::Iq4Xs => {
                                            vec_dot_iq4xs_batch::<CAP>(row, &q8s, in_f, chunk)
                                        }
                                        DType::Iq2S => vec_dot_iq2s_batch(row, &q8s, in_f, chunk),
                                        DType::Iq2Xs => vec_dot_iq2xs_batch(row, &q8s, in_f, chunk),
                                        DType::Iq2Xxs => {
                                            vec_dot_iq2xxs_batch(row, &q8s, in_f, chunk)
                                        }
                                        DType::Iq1S => vec_dot_iq1s_batch(row, &q8s, in_f, chunk),
                                        DType::Iq1M => vec_dot_iq1m_batch(row, &q8s, in_f, chunk),
                                        DType::Iq3S => vec_dot_iq3s_batch(row, &q8s, in_f, chunk),
                                        DType::Iq3Xxs => {
                                            vec_dot_iq3xxs_batch(row, &q8s, in_f, chunk)
                                        }
                                        DType::Tq2_0 => vec_dot_tq2_0_batch(row, &q8s, in_f, chunk),
                                        DType::Tq1_0 => vec_dot_tq1_0_batch(row, &q8s, in_f, chunk),
                                        DType::F32 => {
                                            let w32 = f32_row(row);
                                            for r in 0..m {
                                                chunk[r] =
                                                    dots.dot(&w32, &xs[r * in_f..r * in_f + in_f]);
                                            }
                                        }
                                        DType::F16 => {
                                            for r in 0..m {
                                                chunk[r] = dots
                                                    .dot_f16(row, &xs[r * in_f..r * in_f + in_f]);
                                            }
                                        }
                                        DType::Bf16 => {
                                            for r in 0..m {
                                                chunk[r] = dots
                                                    .dot_bf16(row, &xs[r * in_f..r * in_f + in_f]);
                                            }
                                        }
                                        _ => {
                                            // Dequant the weight row ONCE, reuse across all m tokens.
                                            let wf = bytes_to_f32(row, dt);
                                            for r in 0..m {
                                                chunk[r] =
                                                    dots.dot(&wf, &xs[r * in_f..r * in_f + in_f]);
                                            }
                                        }
                                    }
                                });
                            }
                            // The sparse activation parts (empty `q8s` on the 32-block and float
                            // routes, so nothing runs there).
                            self.add_hi(dt, wbytes, bpr, in_f, &q8s, &mut out_t);
                            // Transpose out_t[o * m + r] → out[r * out_f + o], parallel over the m output
                            // rows (each gathers its out_f values from the o-major temp). The serial
                            // version was ~20% of the matvec at large out_f × m.
                            self.pool().for_chunks_mut(&mut out, out_f, 1, &|r, orow| {
                                for (o, dst) in orow.iter_mut().enumerate() {
                                    *dst = out_t[o * m + r];
                                }
                            });
                            rc.give(out_t);
                        }
                    }
                    tap(&out);
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::Rope {
                    x,
                    positions,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    rope_dim,
                    theta,
                    freq_factors,
                    neox,
                    backward,
                    x_stride,
                } => {
                    // As for QkNorm: only QkNormRope reads strided rows.
                    assert_eq!(x_stride, 0, "Rope: strided x is not supported");
                    let (rows, nh, hd, rd) = (
                        rows as usize,
                        n_head as usize,
                        head_dim as usize,
                        rope_dim as usize,
                    );
                    // Borrow the inputs (no clone) — `out` is the single owned copy we mutate and
                    // then move into `vals[dst]`; the borrows end before that write (NLL). If
                    // dst == x this still holds: `out` is an independent Vec.
                    let xs = &vals[x.0 as usize];
                    let pos = &vals[positions.0 as usize];
                    let ff = freq_factors.map(|f| &vals[f.0 as usize]);
                    let mut out = rc.copy_of(xs); // dims beyond rope_dim pass through unchanged
                                                  // Pair p's two elements: INTERLEAVED (2p, 2p+1) for llama.cpp's ROPE_TYPE_NORM
                                                  // (the llama family, DeepSeek's q_pe/k_pe), SPLIT-HALF (p, p + rope_dim/2) for
                                                  // NEOX (deepseek32's lightning indexer). Angles and pass-through tail are the
                                                  // same either way — see `Op::Rope::neox`, and QkNormRope for the same pairing
                                                  // applied unconditionally.
                    let hf = rd / 2;
                    // `ggml_rope_ext_back` is this rotation with `sin` negated and `cos`
                    // untouched (`sin_sign` in ggml_compute_forward_rope_flt) — the transpose,
                    // which is the inverse because nothing scales the magnitude here. See
                    // `Op::Rope::backward`.
                    let sin_sign = if backward { -1.0 } else { 1.0 };
                    // Parallel over rows. A pair's frequency depends only on the op's shape, so
                    // it is computed once per op; its angle depends on the row's position and the
                    // pair index, never the head, so each row computes its (sin, cos) table once
                    // and every head reuses it — the same values the per-head form computed.
                    let freqs = self.rope_freqs(theta, rd);
                    self.pool().for_chunks_mut(
                        &mut out[..rows * nh * hd],
                        nh * hd,
                        1,
                        &|r, orow| {
                            let p0 = pos[r];
                            let sc: Vec<(f32, f32)> = (0..hf)
                                .map(|p| {
                                    let (s, c) =
                                        math::rope_sin_cos(p0, freqs[p], ff.map(|ff| ff[p]));
                                    (s * sin_sign, c)
                                })
                                .collect();
                            let xr = &xs[r * nh * hd..(r + 1) * nh * hd];
                            for h in 0..nh {
                                let b = h * hd;
                                for (p, &(s, c)) in sc.iter().enumerate() {
                                    let (i0, i1) = if neox {
                                        (p, p + hf)
                                    } else {
                                        (2 * p, 2 * p + 1)
                                    };
                                    let a = xr[b + i0];
                                    let bb = xr[b + i1];
                                    orow[b + i0] = a * c - bb * s;
                                    orow[b + i1] = a * s + bb * c;
                                }
                            }
                        },
                    );
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::QkNormRope {
                    x,
                    weight: w,
                    positions,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    rope_dim,
                    theta,
                    eps,
                    freq_factors,
                    x_stride,
                    ..
                } => {
                    let (rows, nh, hd, rd) = (
                        rows as usize,
                        n_head as usize,
                        head_dim as usize,
                        rope_dim as usize,
                    );
                    let raw = &vals[x.0 as usize];
                    let x_stride = x_stride as usize;
                    // If input is interleaved (stride > 0), extract packed query data per row.
                    let xs: std::borrow::Cow<[f32]> = if x_stride > 0 {
                        let mut packed = vec![0f32; rows * nh * hd];
                        for r in 0..rows {
                            let row_base = r * x_stride;
                            for h in 0..nh {
                                let src = row_base + h * 2 * hd; // head h query starts here
                                let dst = (r * nh + h) * hd;
                                packed[dst..dst + hd].copy_from_slice(&raw[src..src + hd]);
                            }
                        }
                        packed.into()
                    } else {
                        raw.as_slice().into()
                    };
                    let xs = &xs;
                    let ws = weight(w);
                    let pos = &vals[positions.0 as usize];
                    let ff = freq_factors.map(|f| &vals[f.0 as usize]);
                    let mut out = rc.for_overwrite(rows * nh * hd);
                    let hf = rd / 2;
                    // Parallel over the m rows. The frequencies depend only on the op's shape
                    // (computed once per op); within a row the RoPE angles depend only on the
                    // position (not the head), so precompute (cos,sin) per rope index ONCE per row
                    // and reuse across all heads — the sin/cos were the bulk and were redone nh×.
                    let freqs = self.rope_freqs(theta, rd);
                    self.pool()
                        .for_chunks_mut(&mut out, nh * hd, 1, &|r, orow| {
                            let p0 = pos[r];
                            let cs: Vec<(f32, f32)> = (0..hf)
                                .map(|p| {
                                    let (s, c) =
                                        math::rope_sin_cos(p0, freqs[p], ff.map(|ff| ff[p]));
                                    (c, s)
                                })
                                .collect();
                            let xr = &xs[r * nh * hd..r * nh * hd + nh * hd];
                            for h in 0..nh {
                                let b = h * hd;
                                let ss: f32 =
                                    (0..hd).map(|i| xr[b + i] * xr[b + i]).sum::<f32>() / hd as f32;
                                let s = 1.0 / (ss + eps).sqrt();
                                for i in 0..hd {
                                    orow[b + i] = xr[b + i] * s * ws[i];
                                }
                                for p in 0..hf {
                                    let (i0, i1) = (p, p + hf);
                                    let (c, sn) = cs[p];
                                    let a = orow[b + i0];
                                    let bb = orow[b + i1];
                                    orow[b + i0] = a * c - bb * sn;
                                    orow[b + i1] = a * sn + bb * c;
                                }
                            }
                        });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::WriteKv {
                    src,
                    cache,
                    rows,
                    row_stride,
                    pos,
                } => {
                    let (rows, rs, pos) = (rows as usize, row_stride as usize, pos as usize);
                    let s = &vals[src.0 as usize];
                    // Write the new row(s) straight into the persistent KV buffer — only `rows` rows
                    // touched, not the whole `max_ctx`-sized cache. The cache dtype (f16 to match the
                    // GPU and halve memory, or f32) is read from the graph; cast on write.
                    let buf = bindings.get(cache).expect("cpu backend: unbound KV cache");
                    let mut d = cpu_buf(buf).owned();
                    // SWA ring cache: position p lands at row p % cap_rows — the ring only ever
                    // recycles rows whose positions the sliding-window mask already excludes (see
                    // the seam runner's ring sizing), so this is exact. A write crossing the wrap
                    // splits into two contiguous segments `(src_row, dst_row, n_rows)`; on a
                    // full-context cache pos < cap_rows and segment 0 is the old single write.
                    let cap_rows = g.desc(cache).numel() / rs.max(1);
                    let pos_r = if cap_rows > 0 { pos % cap_rows } else { pos };
                    let segs: [(usize, usize, usize); 2] =
                        if cap_rows > 0 && pos_r + rows > cap_rows {
                            let r1 = cap_rows - pos_r;
                            [(0, pos_r, r1), (r1, 0, rows - r1)]
                        } else {
                            [(0, pos_r, rows), (0, 0, 0)]
                        };
                    for &(sr, dr, nr) in segs.iter().filter(|&&(_, _, nr)| nr > 0) {
                        let s = &s[sr * rs..sr * rs + nr * rs];
                        let base = dr * rs;
                        let n = nr * rs;
                        match g.desc(cache).dtype {
                            DType::F16 => {
                                let df: &mut [u16] = bytemuck::cast_slice_mut(&mut d);
                                dots.f16_from_f32(&mut df[base..base + n], &s[..n]);
                            }
                            DType::Q8_0 => {
                                // Q8_0 blocks (34 B / 32 elems): d = amax/127 (stored f16), q =
                                // round(x/d) via `round_ties_even`. NOTE: llama.cpp's
                                // quantize_row_q8_0 uses `roundf` (half-away-from-zero) and the GPU
                                // `store_q8.comp` uses GLSL `round()` (halfway is impl-defined), so
                                // this backend deliberately differs on exact-halfway activations.
                                // That is intentional and safe: no path pins the CPU KV Q8 bytes
                                // bit-for-bit — the CPU/GPU KV Q8 tests are all coherence/tolerance
                                // checks (kv_q8_roundtrip err<0.02, gpu_seam_kv_q8_tracks_cpu),
                                // never exact-byte goldens — and halfway ties are vanishingly rare,
                                // so the <0.5 LSB difference never changes a decoded token.
                                // `base`/`n` are element counts and rows are 32-aligned (the runner
                                // gates on it), so blocks never straddle a write.
                                debug_assert!(base % 32 == 0 && n % 32 == 0);
                                for b in 0..n / 32 {
                                    let src32 = &s[b * 32..b * 32 + 32];
                                    let amax = src32.iter().fold(0f32, |m, &v| m.max(v.abs()));
                                    let dq = amax / 127.0;
                                    let id = if dq != 0.0 { 1.0 / dq } else { 0.0 };
                                    let off = (base / 32 + b) * 34;
                                    let dh = half::f16::from_f32(dq).to_bits().to_le_bytes();
                                    d[off] = dh[0];
                                    d[off + 1] = dh[1];
                                    for (i, &v) in src32.iter().enumerate() {
                                        d[off + 2 + i] =
                                            (v * id).round_ties_even() as i32 as i8 as u8;
                                    }
                                }
                            }
                            DType::Bf16 => {
                                let df: &mut [u16] = bytemuck::cast_slice_mut(&mut d);
                                for i in 0..n {
                                    df[base + i] = half::bf16::from_f32(s[i]).to_bits();
                                }
                            }
                            dt @ (DType::Turbo2 | DType::Turbo3 | DType::Turbo4) => {
                                // TurboQuant: each 128-elem group (a head_dim slice) → one block
                                // (L2-norm + WHT + 2/3/4-bit PolarQuant). base/n are 128-aligned (the
                                // runner gates head_dim%128), so blocks never straddle a write.
                                debug_assert!(base % 128 == 0 && n % 128 == 0);
                                let bb = crate::turbo::block_bytes(dt);
                                let blk0 = base / 128;
                                for b in 0..n / 128 {
                                    let off = (blk0 + b) * bb;
                                    crate::turbo::quantize_block(
                                        dt,
                                        &s[b * 128..b * 128 + 128],
                                        &mut d[off..off + bb],
                                    );
                                }
                            }
                            dt if crate::kvquant::supported(dt) => {
                                // Mainline low-bit KV quants (q4_0/q4_1/q5_0/q5_1/iq4_nl): quantize the
                                // f32 activations into 32-elem blocks. base/n are 32-aligned (kv_align_ok).
                                debug_assert!(base % 32 == 0 && n % 32 == 0);
                                let bb = infr_gguf::nbytes(dt, 32);
                                let off = (base / 32) * bb;
                                crate::kvquant::quantize_row(
                                    dt,
                                    &s[..n],
                                    &mut d[off..off + n / 32 * bb],
                                );
                            }
                            _ => {
                                let df: &mut [f32] = bytemuck::cast_slice_mut(&mut d);
                                df[base..base + n].copy_from_slice(&s[..n]);
                            }
                        }
                    }
                }
                Op::Attention {
                    q,
                    k_cache,
                    v_cache,
                    dst,
                    rows,
                    kv_len,
                    n_head,
                    n_kv,
                    head_dim,
                    scale,
                    mask,
                    pos,
                    sinks,
                    key_bias,
                    comp,
                } => {
                    if let Err(e) = infr_core::graph::check_attention(g, op) {
                        panic!("cpu backend: {e}");
                    }
                    op_tap::record(op_tap::TappedOp::Attention { rows });
                    let (rows, kv_len, nh, nkv, hd) = (
                        rows as usize,
                        kv_len as usize,
                        n_head as usize,
                        n_kv as usize,
                        head_dim as usize,
                    );
                    let qs = &vals[q.0 as usize];
                    // K/V live in their persistent buffers (f32); borrow them — attention reads only
                    // the first `kv_len` rows, never the whole `max_ctx` cache.
                    let kbuf = bindings.get(k_cache).expect("cpu backend: unbound k_cache");
                    let vbuf = bindings.get(v_cache).expect("cpu backend: unbound v_cache");
                    // Materialize the valid KV rows as f32. K and V pick their cache dtype
                    // INDEPENDENTLY (f16 matches the GPU's f16 KV; Q8_0 blocks dequant via
                    // y = d*q) — the inner dot then runs in f32 either way.
                    //
                    // SWA ring cache: the cache holds only `cap_rows` rows (window + ubatch — see
                    // the seam runner's ring sizing) and position j lives at row j % cap_rows.
                    // The ring only ever recycles rows whose positions the sliding-window mask
                    // (`lo`) already excludes, so the mapping is exact; on a full-context cache
                    // kv_len <= cap_rows and everything below is the old identity indexing.
                    let cap_rows = g.desc(k_cache).numel() / (nkv * hd).max(1);
                    let need = kv_len.min(cap_rows) * nkv * hd;
                    // One guard at a time: an owned buffer's guard is a non-reentrant mutex, and a
                    // cache that serves as both K and V (deepseek4, whose attention reads its K
                    // rows as V) binds one buffer to both sides. That pair is dequantized once.
                    let ks = attn_kv::kv_prefix_f32(
                        self.pool(),
                        &mut rc,
                        &cpu_buf(kbuf).read(),
                        g.desc(k_cache).dtype,
                        need,
                    );
                    let vs_own = (v_cache != k_cache).then(|| {
                        attn_kv::kv_prefix_f32(
                            self.pool(),
                            &mut rc,
                            &cpu_buf(vbuf).read(),
                            g.desc(v_cache).dtype,
                            need,
                        )
                    });
                    let vs: &[f32] = vs_own.as_deref().unwrap_or(&ks);
                    // DeepSeek V4's compressed segment (`None` everywhere else): its first `len`
                    // rows, which serve as both its keys and its values, and its optional
                    // `[rows, len]` score bias. See `Op::Attention::comp`.
                    let cs = comp.map(|c| {
                        let cbuf = bindings
                            .get(c.cache)
                            .expect("cpu backend: unbound compressed cache");
                        let bias = c.bias.map(|b| &vals[b.0 as usize][..]);
                        let seg = attn_kv::comp_rows(
                            self.pool(),
                            &mut rc,
                            &cpu_buf(cbuf).read(),
                            g.desc(c.cache).dtype,
                            nkv * hd,
                            bias,
                            rows,
                            pos,
                            c,
                        );
                        (seg, bias, c)
                    });
                    // Per-head attention sinks (V4's `attn_sinks`, `None` everywhere else): one
                    // extra logit per head in the max AND the denominator, never in the numerator.
                    // See `Op::Attention::sinks`.
                    let sk = sinks.map(weight);
                    // DeepSeek V4 CSA's additive per-(row, key) top-k score mask (`None` everywhere
                    // else) — indexed by KEY POSITION, not the ring row. See `Op::Attention::key_bias`.
                    let kbias = key_bias.map(|b| vals[b.0 as usize].clone());
                    let group = nh / nkv;
                    // `Causal`/`SlidingWindow` clip the causal END at `abs+1` (per-row, from
                    // `pos`); `Canvas` (DiffusionGemma denoise — see `AttnMask::Canvas`'s doc)
                    // ignores `pos` entirely and gives every row the SAME fixed bidirectional
                    // range `[lo, kv_len)`.
                    let (window, canvas_lo) = match mask {
                        AttnMask::Causal => (0usize, None),
                        AttnMask::SlidingWindow(w) => (w, None),
                        AttnMask::Canvas { lo } => (0usize, Some(lo)),
                    };
                    let mut out = rc.for_overwrite(rows * nh * hd);
                    // Each (ti, h) pair writes exactly one hd-sized output slice with no
                    // cross-iteration deps → embarrassingly parallel.
                    let attend = |ti: usize, h: usize, ob_slice: &mut [f32]| {
                        // The weighted V rows accumulate into this slice from zero.
                        ob_slice.fill(0.0);
                        let kvh = h / group;
                        let qb = (ti * nh + h) * hd;
                        let abs = pos as usize + ti; // absolute position of this query
                        let (lo, hi) = match canvas_lo {
                            // bidirectional: every row attends the same fixed [lo, kv_len).
                            Some(clo) => (clo, kv_len),
                            // causal (± SWA): [lo, abs] — SWA clips lo to abs-window+1.
                            None => {
                                let lo = if window > 0 && abs + 1 > window {
                                    abs + 1 - window
                                } else {
                                    0
                                };
                                (lo, abs + 1)
                            }
                        };
                        let n_keys = hi - lo;
                        // The compressed keys that can carry weight, scored after the raw ones in
                        // the same buffer so one max, one denominator and one weighted sum cover
                        // both (the reference's concatenated `[raw | compressed]` attention). A
                        // key the bias sets to `-inf` is not scored at all: its exponential is
                        // exactly 0, so leaving it out of the max, the denominator and the sum
                        // changes none of them — llama.cpp's flash attention skips it the same
                        // way.
                        let kept_j: &[u32] = cs.as_ref().map_or(&[], |(seg, _, _)| &seg.keys[ti]);
                        let n_comp = kept_j.len();
                        let mut sc = vec![0f32; n_keys + n_comp];
                        let (sc_raw, sc_comp) = sc.split_at_mut(n_keys);
                        let mut mx = f32::NEG_INFINITY;
                        let qrow = &qs[qb..qb + hd];
                        // Key j's cache row, j % cap_rows on a ring, walked without a division
                        // per key.
                        let ring_rows = |lo: usize| {
                            let mut jr = if cap_rows > 0 { lo % cap_rows } else { lo };
                            std::iter::from_fn(move || {
                                let r = jr;
                                jr += 1;
                                if jr == cap_rows {
                                    jr = 0;
                                }
                                Some(r)
                            })
                        };
                        // Raw q·k dots, several keys at a time (`dot_n` overlaps their FMA
                        // chains; each value is bit-identical to a lone `dot`).
                        const KEYS: usize = 4;
                        let krow = |jr: usize| {
                            let kb = (jr * nkv + kvh) * hd;
                            &ks[kb..kb + hd]
                        };
                        let mut key_rows = ring_rows(lo);
                        let (quads, tail) = sc_raw.as_chunks_mut::<KEYS>();
                        for quad in quads {
                            let jr: [usize; KEYS] =
                                std::array::from_fn(|_| key_rows.next().unwrap());
                            *quad = dots.dot_n(qrow, jr.map(krow));
                        }
                        for (scj, jr) in tail.iter_mut().zip(key_rows) {
                            *scj = dots.dot(qrow, krow(jr));
                        }
                        for (jj, scj) in sc_raw.iter_mut().enumerate() {
                            let j = lo + jj;
                            *scj *= scale;
                            // The top-k mask joins the score BEFORE the max, at key POSITION `j`
                            // (not the ring row) — same rule as `Op::Mla::key_bias`.
                            if let Some(kbias) = &kbias {
                                *scj += kbias[ti * kv_len + j];
                            }
                            mx = mx.max(*scj);
                        }
                        // Compressed block `j` is the cache's row `j`: no ring, the cache holds
                        // every block. Only the rows some query keeps were converted, at their
                        // `slot`.
                        let crow = |j: u32| {
                            let seg = &cs.as_ref().expect("n_comp > 0 implies comp").0;
                            seg.row(j, kvh, hd, nkv)
                        };
                        let (cquads, ctail) = sc_comp.as_chunks_mut::<KEYS>();
                        let (kquads, ktail) = kept_j.as_chunks::<KEYS>();
                        for (quad, js) in cquads.iter_mut().zip(kquads) {
                            *quad = dots.dot_n(qrow, js.map(crow));
                        }
                        for (scj, &j) in ctail.iter_mut().zip(ktail) {
                            *scj = dots.dot(qrow, crow(j));
                        }
                        let cbias = cs.as_ref().and_then(|(_, b, c)| b.map(|b| (b, c.len)));
                        for (scj, &j) in sc_comp.iter_mut().zip(kept_j) {
                            *scj *= scale;
                            if let Some((b, len)) = cbias {
                                *scj += b[ti * len as usize + j as usize];
                            }
                            mx = mx.max(*scj);
                        }
                        // The sink joins the max first (so a dominating sink can't overflow the
                        // exps), then the denominator — `ggml_compute_forward_soft_max_f32`'s two
                        // `if (sk)` lines, in that order.
                        let sink = sk.as_ref().map(|w| w[h]);
                        if let Some(sv) = sink {
                            mx = mx.max(sv);
                        }
                        // Each score becomes its exponential in place: the denominator sums
                        // them in key order and the weights below divide the same values, so
                        // every exp is taken once.
                        math::exp_sub_slice(&mut sc, mx);
                        let mut l = 0f32;
                        for &s in sc.iter() {
                            l += s;
                        }
                        if let Some(sv) = sink {
                            l += math::exp(sv - mx);
                        }
                        let (e_raw, e_comp) = sc.split_at(n_keys);
                        for (&e, jr) in e_raw.iter().zip(ring_rows(lo)) {
                            let p = e / l;
                            let vb = (jr * nkv + kvh) * hd;
                            // Independent lanes — vectorizes to FMA.
                            dots.axpy(ob_slice, &vs[vb..vb + hd], p);
                        }
                        // A compressed key's value row is its key row.
                        for (&e, &j) in e_comp.iter().zip(kept_j) {
                            let p = e / l;
                            dots.axpy(ob_slice, crow(j), p);
                        }
                    };
                    // One task is one head over a block of consecutive query rows, so the block
                    // re-reads that head's K/V rows from its core's own cache; with one
                    // (row, head) pair per task every query fetched the whole prefix again
                    // from the shared L3. Blocks run latest first: under a causal mask a later
                    // row attends more keys, and the big tasks are best not left for the tail.
                    // Each pair's arithmetic is unchanged, so the output is too.
                    let qblk = self.attn_query_block;
                    let n_blocks = rows.div_ceil(qblk);
                    assert_eq!(out.len(), rows * nh * hd);
                    let outp = pool::SendPtr::new(out.as_mut_ptr());
                    self.pool().run(n_blocks * nh, &|task| {
                        let blk = n_blocks - 1 - task / nh;
                        let h = task % nh;
                        let t1 = ((blk + 1) * qblk).min(rows);
                        for ti in blk * qblk..t1 {
                            assert!(ti < rows && h < nh, "attention task outside the output");
                            // SAFETY: `(ti, h)` with `ti < rows` and `h < nh` (both asserted) is
                            // in bounds of `out` (asserted length), and each pair belongs to exactly one task,
                            // so no two tasks touch the same slice; `out` outlives the `run`.
                            let ob_slice = unsafe {
                                std::slice::from_raw_parts_mut(
                                    outp.get().add((ti * nh + h) * hd),
                                    hd,
                                )
                            };
                            attend(ti, h, ob_slice);
                        }
                    });
                    rc.give(ks);
                    if let Some(vs) = vs_own {
                        rc.give(vs);
                    }
                    if let Some((seg, _, _)) = cs {
                        rc.give(seg.rows);
                    }
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::GatedAct {
                    gate,
                    up,
                    dst,
                    rows,
                    nff,
                    act,
                    up_off,
                    up_stride,
                    gate_stride,
                    gate_block_width,
                    swiglu_clamp,
                    ..
                } => {
                    assert!(
                        swiglu_clamp.is_none() || !matches!(act, Activation::Sigmoid),
                        "cpu Op::GatedAct: swiglu_clamp is defined for the gated SiLU/GELU forms \
                         only (llama.cpp clamps in its LLM_FFN_SILU arm); got {act:?}"
                    );
                    let (rows, nff, up_off, up_stride, gate_stride, gate_block_width) = (
                        rows as usize,
                        nff as usize,
                        up_off as usize,
                        up_stride as usize,
                        gate_stride as usize,
                        gate_block_width as usize,
                    );
                    let gs = &vals[gate.0 as usize];
                    let us = &vals[up.0 as usize];
                    let mut out = rc.for_overwrite(rows * nff);
                    self.pool().for_chunks_mut(&mut out, nff, 1, &|r, orow| {
                        let gb = if gate_block_width > 0 || gate_stride > 0 {
                            r * gate_stride
                        } else {
                            r * nff
                        };
                        let ub = if up_stride == 0 {
                            r * nff + up_off
                        } else {
                            r * up_stride + up_off
                        };
                        let gate_at = |i: usize| {
                            let gi = if gate_block_width > 0 {
                                let headw = gate_block_width / 2;
                                let head = i / headw;
                                let off = i % headw;
                                gb + head * gate_block_width + headw + off
                            } else {
                                gb + i
                            };
                            gs[gi]
                        };
                        gated_act_row(act, swiglu_clamp, &mut orow[..nff], gate_at, |i| us[ub + i]);
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::GatedActFused {
                    gu,
                    dst,
                    rows,
                    nff,
                    act,
                    swiglu_clamp,
                } => {
                    assert!(
                        swiglu_clamp.is_none() || !matches!(act, Activation::Sigmoid),
                        "cpu Op::GatedActFused: swiglu_clamp is defined for the gated SiLU/GELU \
                         forms only (llama.cpp clamps in its LLM_FFN_SILU arm); got {act:?}"
                    );
                    // Combined [rows, 2*nff] gate|up buffer: gate half first, up half second. Rows
                    // are independent — spin-pool over rows, bit-identical to the serial version.
                    let (rows, nff) = (rows as usize, nff as usize);
                    let gus = &vals[gu.0 as usize];
                    let mut out = rc.for_overwrite(rows * nff);
                    self.pool().for_chunks_mut(&mut out, nff, 1, &|r, orow| {
                        let gb = r * 2 * nff;
                        gated_act_row(
                            act,
                            swiglu_clamp,
                            &mut orow[..nff],
                            |i| gus[gb + i],
                            |i| gus[gb + nff + i],
                        );
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::Add { a, b, dst, n } => {
                    let n = n as usize;
                    // Elementwise with no cross-element dependency — chunked spin-pool, bit-
                    // identical. (The oldest form CLONED the whole `a` vector, then added serially:
                    // a ~0.7 MB memcpy + a one-thread loop per Add while 31 threads slept.)
                    let av = &vals[a.0 as usize];
                    let bv = &vals[b.0 as usize];
                    let mut out = rc.for_overwrite(n);
                    self.pool().for_chunks_mut(&mut out, 4096, 4, &|c, oc| {
                        let base = c * 4096;
                        for (i, o) in oc.iter_mut().enumerate() {
                            *o = av[base + i] + bv[base + i];
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                // Broadcast bias: add the length-`n` `bias` to each of `rows` rows (Qwen2 q/k/v).
                Op::AddBias {
                    x,
                    bias,
                    dst,
                    rows,
                    n,
                } => {
                    let (rows, n) = (rows as usize, n as usize);
                    let xs = &vals[x.0 as usize];
                    let bv = weight(bias); // bias is a bound weight, not an activation
                                           // Rows independent — spin-pool over rows, bit-identical (no whole-input clone).
                    let mut out = rc.zeroed(rows * n);
                    self.pool().for_chunks_mut(&mut out, n, 1, &|r, orow| {
                        for (c, o) in orow.iter_mut().enumerate() {
                            *o = xs[r * n + c] + bv[c];
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::Scale { x, dst, s, n } => {
                    let n = n as usize;
                    let xs = &vals[x.0 as usize];
                    // Elementwise — chunked spin-pool, bit-identical (no whole-input clone).
                    let mut out = rc.zeroed(n);
                    self.pool().for_chunks_mut(&mut out, 4096, 4, &|c, oc| {
                        let base = c * 4096;
                        for (i, o) in oc.iter_mut().enumerate() {
                            *o = xs[base + i] * s;
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                // Broadcast multiply: the length-`n` `vec` scales every one of `rows` rows
                // (diffusion-gemma's router input scale — the multiplicative twin of `AddBias`).
                Op::MulVec {
                    x,
                    vec: vecid,
                    dst,
                    rows,
                    n,
                } => {
                    op_tap::record(op_tap::TappedOp::MulVec { rows, n });
                    let (rows, n) = (rows as usize, n as usize);
                    let xs = &vals[x.0 as usize];
                    let vv = weight(vecid); // vec is a bound weight, not an activation
                                            // Rows independent — spin-pool over rows, bit-identical (no whole-input clone).
                    let mut out = rc.zeroed(rows * n);
                    self.pool().for_chunks_mut(&mut out, n, 1, &|r, orow| {
                        for (c, o) in orow.iter_mut().enumerate() {
                            *o = xs[r * n + c] * vv[c];
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                // Position-dependent attention temperature — see the `Op::AttnTemp` doc.
                // `llm_graph_input_attn_temp::set_input` takes the logarithm in f64 and rounds the
                // factor to f32 once; the logarithm here is the correctly rounded f32 one (the
                // platform's is off limits, `infr_core::math`), and the rest is f64 like there.
                Op::AttnTemp {
                    x,
                    dst,
                    rows,
                    n,
                    pos,
                    floor_scale,
                    scale,
                    offset,
                } => {
                    if floor_scale == 0 {
                        return Err(infr_core::error::Error::backend(
                            "cpu backend: Op::AttnTemp with floor_scale 0".to_string(),
                        ));
                    }
                    let (rows, n) = (rows as usize, n as usize);
                    let xs = &vals[x.0 as usize];
                    let mut out = rc.zeroed(rows * n);
                    self.pool().for_chunks_mut(&mut out, n, 1, &|r, orow| {
                        let p = u64::from(pos) + r as u64 + u64::from(offset);
                        let k = (p / u64::from(floor_scale)) as f32;
                        let t = (f64::from(math::ln(k + 1.0)) * f64::from(scale) + 1.0) as f32;
                        for (c, o) in orow.iter_mut().enumerate() {
                            *o = xs[r * n + c] * t;
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                // qwen35moe shared-expert combine: `moe` (routed-MoE output) + sigmoid(`gate[r]`) *
                // `shexp` (the shared expert's own dense FFN output), row-broadcast — see the
                // `Op::MoeSharedExpertAdd` doc. Rows independent — spin-pool over rows, mirrors
                // `Op::MulVec` above (bit-identical, no whole-input clone).
                Op::MoeSharedExpertAdd {
                    moe,
                    shexp,
                    gate,
                    dst,
                    rows,
                    n,
                } => {
                    let (rows, n) = (rows as usize, n as usize);
                    let mv = &vals[moe.0 as usize];
                    let sv = &vals[shexp.0 as usize];
                    let gv = &vals[gate.0 as usize]; // [rows] raw pre-sigmoid logits
                    let mut out = rc.zeroed(rows * n);
                    self.pool().for_chunks_mut(&mut out, n, 1, &|r, orow| {
                        let g = math::sigmoid(gv[r]);
                        for (c, o) in orow.iter_mut().enumerate() {
                            *o = mv[r * n + c] + g * sv[r * n + c];
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                // LFM3 attention output gate: one sigmoid per (row, head), broadcast over the
                // head's `head_dim` lanes — see the `Op::HeadGate` doc.
                Op::HeadGate {
                    x,
                    gate,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                } => {
                    let (nh, hd) = (n_head as usize, head_dim as usize);
                    let xs = &vals[x.0 as usize];
                    let gv = &vals[gate.0 as usize]; // [rows, n_head] raw logits
                    let mut out = rc.zeroed(rows as usize * nh * hd);
                    self.pool().for_chunks_mut(&mut out, hd, 16, &|rh, orow| {
                        let g = math::sigmoid(gv[rh]);
                        for (i, o) in orow.iter_mut().enumerate() {
                            *o = xs[rh * hd + i] * g;
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::Softcap { x, dst, cap, n } => {
                    let n = n as usize;
                    let xs = &vals[x.0 as usize];
                    // Pure elementwise map (no cross-element dependency) — safe to fan out over
                    // rayon with ZERO numeric change. At the lm_head's shape (256 rows × 262144
                    // vocab = ~67M `tanh` calls) this was the single most expensive scalar loop in
                    // the interpreter outside `Op::Linear`/`Op::MoeFfn`.
                    // Chunked (not per-element zip — 67M one-element items is pure plumbing).
                    let mut out = rc.zeroed(n);
                    self.pool().for_chunks_mut(&mut out, 4096, 4, &|c, oc| {
                        let base = c * 4096;
                        for (i, o) in oc.iter_mut().enumerate() {
                            *o = xs[base + i] / cap;
                        }
                        math::tanh_slice(oc);
                        for o in oc.iter_mut() {
                            *o *= cap;
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::EmbedGather {
                    ids,
                    table,
                    dst,
                    rows,
                    ne,
                    scale,
                } => {
                    // Dequantize the selected rows straight from the raw (mmap'd) table bytes —
                    // identical math to the load-time dequant that used to build the host f32
                    // table, so outputs are bit-equal to the old host-embed path.
                    op_tap::record(op_tap::TappedOp::EmbedGather {
                        rows,
                        table: g.desc(table).dtype,
                    });
                    let (rows, ne) = (rows as usize, ne as usize);
                    let buf = bindings.get(table).expect("cpu backend: unbound Weight");
                    let bytes = cpu_buf(buf).read();
                    let dt = g.desc(table).dtype;
                    let total_rows = g.desc(table).numel() / ne;
                    let bpr = bytes.len() / total_rows;
                    let ids_v = vals[ids.0 as usize].clone();
                    let mut out = rc.zeroed(rows * ne);
                    for r in 0..rows {
                        let tok = ids_v[r] as usize;
                        let row = bytes_to_f32(&bytes[tok * bpr..(tok + 1) * bpr], dt);
                        for (o, v) in out[r * ne..(r + 1) * ne].iter_mut().zip(&row) {
                            *o = v * scale;
                        }
                    }
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::GatherI32 {
                    ids,
                    table,
                    dst,
                    rows,
                    ne,
                } => {
                    // `ggml_get_rows` on an I32 table: values are copied, never dequantized. Both
                    // `ids` and `dst` are I32 handles, which this interpreter carries as f32-widened
                    // plain integers (`bytes_to_f32`'s I32 arm) — the same representation
                    // `Op::MoeFfn::expert_ids` reads back.
                    let (rows, ne) = (rows as usize, ne as usize);
                    let buf = bindings.get(table).expect("cpu backend: unbound Weight");
                    let dt = g.desc(table).dtype;
                    assert_eq!(
                        dt,
                        DType::I32,
                        "cpu Op::GatherI32: the table must be an I32 tensor"
                    );
                    // Widen only the ROWS asked for, straight out of the raw (mmap'd) table bytes
                    // — same shape as `Op::EmbedGather` above, and for the same reason: the whole
                    // table is `n_expert_used * n_vocab` entries and converting all of it per op
                    // would allocate megabytes per token to read a handful of integers.
                    let bytes = cpu_buf(buf).read();
                    let bpr = ne * 4; // I32, one row of `ne`
                    let ids_v = vals[ids.0 as usize].clone();
                    let mut out = rc.zeroed(rows * ne);
                    for r in 0..rows {
                        let tok = ids_v[r] as usize;
                        assert!(
                            (tok + 1) * bpr <= bytes.len(),
                            "cpu Op::GatherI32: id {tok} is past the {} rows of the table",
                            bytes.len() / bpr
                        );
                        let row = bytes_to_f32(&bytes[tok * bpr..(tok + 1) * bpr], dt);
                        out[r * ne..(r + 1) * ne].copy_from_slice(&row);
                    }
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::GatherRows {
                    src,
                    idx,
                    dst,
                    rows,
                    idx_off,
                    src_stride,
                    src_col,
                    n,
                    group,
                    dst_group_stride,
                    dst_row_off,
                    fill,
                } => {
                    let n_src = infr_core::graph::check_gather_rows(g, op)
                        .unwrap_or_else(|e| panic!("cpu {e}"));
                    let (rows, idx_off, ss, sc, n) = (
                        rows as usize,
                        idx_off as usize,
                        src_stride as usize,
                        src_col as usize,
                        n as usize,
                    );
                    let (group, dgs, dro) = (
                        group as usize,
                        dst_group_stride as usize,
                        dst_row_off as usize,
                    );
                    // `idx` is an I32 handle, carried here as f32-widened integers. The dtype
                    // check makes it a third tensor, so it is set aside while `src` and `dst` are
                    // borrowed and put back after.
                    let ids = std::mem::take(&mut vals[idx.0 as usize]);
                    let (s, d) = borrow_two_mut(&mut vals, src.0 as usize, dst.0 as usize);
                    for r in 0..rows {
                        let i = ids[idx_off + r];
                        let row = (r / group) * dgs + dro + r % group;
                        let out = &mut d[row * n..(row + 1) * n];
                        if i < 0.0 {
                            out.fill(fill);
                        } else {
                            let i = i as usize;
                            assert!(
                                i < n_src,
                                "cpu Op::GatherRows: index {i} at entry {} is past the {n_src} \
                                 source rows",
                                idx_off + r
                            );
                            out.copy_from_slice(&s[i * ss + sc..i * ss + sc + n]);
                        }
                    }
                    vals[idx.0 as usize] = ids;
                }
                Op::Sample {
                    x,
                    u,
                    dst,
                    n,
                    top_k,
                    temp,
                    top_p,
                } => {
                    // Device-side stochastic sampling — see `sample_token` (top_k == 0 = no
                    // truncation; the uniform draw is factored out into the 1-float `u` input).
                    let logits = &vals[x.0 as usize][..n as usize];
                    let uu = vals[u.0 as usize][0];
                    let tok = sample_token(logits, uu, top_k as usize, temp, top_p);
                    rc.set(&mut vals, dst.0 as usize, vec![f32::from_bits(tok)]);
                }
                Op::Argmax { x, dst, n, rows } => {
                    // Greedy device-side sampling: strict `>` keeps the lowest index on ties —
                    // identical to the host sampler's argmax (per row). Each id is stored as a
                    // u32 bit-pattern in the f32 slot (the graph's only tensor dtype). rows == 1
                    // is the decode loop; rows > 1 is the MTP verify accept (issue #31).
                    let n = n as usize;
                    let mut out = Vec::with_capacity(rows as usize);
                    for r in 0..rows as usize {
                        let xs = &vals[x.0 as usize][r * n..(r + 1) * n];
                        let mut best = f32::NEG_INFINITY;
                        let mut bi = 0u32;
                        for (i, &v) in xs.iter().enumerate() {
                            if v > best {
                                best = v;
                                bi = i as u32;
                            }
                        }
                        out.push(f32::from_bits(bi));
                    }
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::ArgmaxProb {
                    x,
                    dst_id,
                    dst_prob,
                    n,
                } => {
                    // Exact reference: this MUST match `infr_llama::mtp::top1_softmax` bit-for-bit
                    // (same strict-`>` argmax scan, then the SAME left-to-right `sum_j
                    // exp(x[j]-max)` accumulation order) — the CPU backend is the golden oracle the
                    // GPU two-stage reduction is checked against (`mtp_head_cpu_vulkan_parity`).
                    let n = n as usize;
                    let xs = &vals[x.0 as usize][0..n];
                    let mut best = f32::NEG_INFINITY;
                    let mut bi = 0u32;
                    for (i, &v) in xs.iter().enumerate() {
                        if v > best {
                            best = v;
                            bi = i as u32;
                        }
                    }
                    let mut e = xs.to_vec();
                    math::exp_sub_slice(&mut e, best);
                    let z: f32 = e.iter().sum();
                    rc.set(&mut vals, dst_id.0 as usize, vec![f32::from_bits(bi)]);
                    rc.set(&mut vals, dst_prob.0 as usize, vec![1.0 / z]);
                }
                Op::Copy {
                    src,
                    src_off,
                    dst,
                    dst_off,
                    n,
                } => {
                    let (so, dof, n) = (src_off as usize, dst_off as usize, n as usize);
                    if src.0 == dst.0 {
                        // Aliasing copy within one tensor: a temp is required (can't borrow the
                        // same Vec mutably and immutably). Clone only the read window.
                        let s = vals[src.0 as usize][so..so + n].to_vec();
                        vals[dst.0 as usize][dof..dof + n].copy_from_slice(&s);
                    } else {
                        let (s, d) = borrow_two_mut(&mut vals, src.0 as usize, dst.0 as usize);
                        d[dof..dof + n].copy_from_slice(&s[so..so + n]);
                    }
                }
                Op::CopyStrided {
                    src,
                    src_off,
                    src_stride,
                    dst,
                    dst_off,
                    dst_stride,
                    rows,
                    n,
                } => {
                    let (so, ss, dof, ds, n) = (
                        src_off as usize,
                        src_stride as usize,
                        dst_off as usize,
                        dst_stride as usize,
                        n as usize,
                    );
                    if src.0 == dst.0 {
                        // Aliasing strided copy: clone the source rows (can't hold &s and &mut d on
                        // one Vec). Rare — kept correct rather than fast.
                        let s = vals[src.0 as usize].clone();
                        let d = &mut vals[dst.0 as usize];
                        for r in 0..rows as usize {
                            d[dof + r * ds..dof + r * ds + n]
                                .copy_from_slice(&s[so + r * ss..so + r * ss + n]);
                        }
                    } else {
                        let (s, d) = borrow_two_mut(&mut vals, src.0 as usize, dst.0 as usize);
                        for r in 0..rows as usize {
                            d[dof + r * ds..dof + r * ds + n]
                                .copy_from_slice(&s[so + r * ss..so + r * ss + n]);
                        }
                    }
                }
                Op::MoeFfn {
                    x,
                    router_x,
                    router,
                    gate_exps,
                    up_exps,
                    down_exps,
                    down_scale,
                    fused_gate_up,
                    dst,
                    ne,
                    n_expert,
                    n_used,
                    n_ff_exp,
                    scale,
                    act,
                    gating,
                    norm_w,
                    weight_before,
                    ep_band: _, // EP is a Vulkan-only path; the CPU reference always runs full-expert
                    exp_probs_b,
                    n_expert_groups,
                    n_expert_groups_used,
                    expert_ids,
                    swiglu_clamp,
                } => {
                    // `weight_before` folds the routing weight into gate/up BEFORE the activation,
                    // so a clamp alongside it would clamp the weighted values — a combination no
                    // reference defines (llama4 has no clamp, V4 has no weight_before).
                    assert!(
                        swiglu_clamp.is_none()
                            || (!matches!(act, Activation::Sigmoid) && !weight_before),
                        "cpu Op::MoeFfn: swiglu_clamp is defined for the gated SiLU/GELU forms \
                         with output-side weighting only (llama.cpp clamps in its LLM_FFN_SILU \
                         arm); got act={act:?} weight_before={weight_before}"
                    );
                    // Hash routing supplies the selection outright, so the two SELECTION-only
                    // inputs have nothing left to act on — llama.cpp nulls `exp_probs_b` on a hash
                    // layer for exactly this reason. Refuse rather than ignore them silently.
                    assert!(
                        expert_ids.is_none() || (exp_probs_b.is_none() && n_expert_groups <= 1),
                        "cpu Op::MoeFfn: expert_ids (hash routing) supplies the selection, so \
                         exp_probs_b (bias: {}) and group routing (groups: {n_expert_groups}) \
                         have nothing to select — they must be absent",
                        exp_probs_b.is_some(),
                    );
                    let (ne, n_expert, n_used, nffx) = (
                        ne as usize,
                        n_expert as usize,
                        n_used as usize,
                        n_ff_exp as usize,
                    );
                    let xs = rc.copy_of(&vals[x.0 as usize]);
                    // `router_x` is usually the SAME tensor as `x` (qwen3moe); diffusion-gemma binds
                    // a differently-normalized row (see the `Op::MoeFfn` doc). Clone independently —
                    // it may legitimately be a different handle with its own row layout.
                    let rxs = rc.copy_of(&vals[router_x.0 as usize]);
                    // `x` may hold several rows (the seam's batched prefill): route + run the
                    // expert FFN independently per row — the reference semantics for the GPU
                    // adapter's GPU-routed batched form.
                    let rows = xs.len() / ne;
                    let rbuf = bindings.get(router).expect("cpu backend: unbound router");
                    let gbuf = bindings
                        .get(gate_exps)
                        .expect("cpu backend: unbound gate_exps");
                    let dbuf = bindings
                        .get(down_exps)
                        .expect("cpu backend: unbound down_exps");
                    // The router stays locked to the end of the op, the banks from `moe_banks` on;
                    // `down_scale` and `exp_probs_b` are read under the router's lock.
                    let ubuf_any = (!fused_gate_up)
                        .then(|| bindings.get(up_exps).expect("cpu backend: unbound up_exps"));
                    let opt = |id: Option<TensorId>| id.and_then(|id| bindings.get(id));
                    refuse_relock(
                        "Op::MoeFfn",
                        ("router", rbuf),
                        &[
                            ("gate_exps", Some(gbuf)),
                            ("up_exps", ubuf_any),
                            ("down_exps", Some(dbuf)),
                            ("down_scale", opt(down_scale)),
                            ("exp_probs_b", opt(exp_probs_b)),
                        ],
                    );
                    refuse_relock(
                        "Op::MoeFfn",
                        ("gate_exps", gbuf),
                        &[("up_exps", ubuf_any), ("down_exps", Some(dbuf))],
                    );
                    if let Some(u) = ubuf_any {
                        refuse_relock("Op::MoeFfn", ("up_exps", u), &[("down_exps", Some(dbuf))]);
                    }
                    let rbytes = cpu_buf(rbuf).read();
                    let rdt = g.desc(router).dtype;
                    let gdt = g.desc(gate_exps).dtype;
                    let ddt = g.desc(down_exps).dtype;
                    // Fused: `gate_exps` holds BOTH roles ([ne, 2*n_ff_exp, n_expert], gate rows
                    // first); split gets its own separate up_exps/up buffer. The banks themselves
                    // are bound only once routing has picked the experts: a bank paged per expert
                    // has nothing to read before then.
                    let (ubuf, udt) = match ubuf_any {
                        None => (None, gdt),
                        Some(u) => (Some(cpu_buf(u)), g.desc(up_exps).dtype),
                    };
                    // One expert's byte band in the gate bank: the repack cache's key.
                    #[cfg(target_arch = "x86_64")]
                    let gst = cpu_buf(gbuf).len_bytes() / n_expert;
                    let dscale = down_scale.map(weight); // per-expert scale [n_expert], if any

                    // ── Router: ONE batched matvec over every row (router_x is tiny — [n_expert, ne]
                    // — so this alone replaces what used to be `rows` separate re-dequants of it),
                    // then a per-row softmax + top-`n_used` selection (independent per row, so
                    // parallel over rows is safe).
                    // Quantized banks take int8 activations at EVERY row count, one row (decode)
                    // included: each row is quantized on its own and the expert kernels give a
                    // row the same bits in a bucket of any size, so a row's result does not
                    // depend on how many rows the op has (`tests/moe_batch_identity.rs`).
                    // REFERENCE mode (`CpuBackend::reference`) forces every bank onto
                    // `ActsKind::Raw`, so an oracle run has no int8-activation error anywhere.
                    let int8_ok = !reference;
                    let pool = self.pool();
                    let logits_all: Vec<f32> = moe::router_logits::<CAP>(
                        pool, &rbytes, rdt, ne, n_expert, &rxs, rows, int8_ok,
                    );
                    struct RouteRow {
                        /// Selected experts, sorted by DESCENDING router prob — the same order the
                        /// original per-row loop accumulated in. On a HASH-routed layer these are
                        /// the supplied ids in their table order instead (nothing sorts them).
                        idx: Vec<usize>,
                        /// Final per-expert weight (renormalized top-k prob × `scale`), aligned to `idx`.
                        w: Vec<f32>,
                    }
                    // DeepSeek V4 hash routing: the selection arrives as an already-gathered
                    // `[rows, n_used]` I32 tensor, widened to f32 by `bytes_to_f32` like every
                    // other integer input, so every selection step below is SKIPPED, not computed
                    // and overwritten. The router GEMM above still ran: llama.cpp's `build_moe_ffn`
                    // takes the routing WEIGHTS from `ggml_get_rows(probs, selected_experts)` even
                    // when the ids are supplied, so the probs are still needed.
                    let hash_ids: Option<Vec<f32>> = expert_ids.map(|id| {
                        let v = vals[id.0 as usize].clone();
                        assert!(
                            v.len() >= rows * n_used,
                            "cpu Op::MoeFfn: expert_ids holds {} slots but the op routes {rows} \
                             rows × {n_used} experts",
                            v.len(),
                        );
                        v
                    });
                    let routes: Vec<RouteRow> = pool.collect(rows, &|r| {
                        let logits = &logits_all[r * n_expert..r * n_expert + n_expert];
                        let probs = moe::router_probs(logits, gating);
                        let idx: Vec<usize> = if let Some(hids) = &hash_ids {
                            // Hash-routed (V4): the token's table row IS the selection. Everything
                            // the `else` branch computes — the biased copy, the group scores, the
                            // sort — exists only to RANK experts, so on this path none of it runs
                            // at all rather than running and being overwritten.
                            (0..n_used)
                                .map(|k| {
                                    let e = hids[r * n_used + k] as usize;
                                    assert!(
                                        e < n_expert,
                                        "cpu Op::MoeFfn: hash-routed expert id {e} at (row {r}, \
                                         slot {k}) is past n_expert {n_expert} — the ids did not \
                                         come from a matching ffn_gate_tid2eid table"
                                    );
                                    e
                                })
                                .collect()
                        } else {
                            let bias = exp_probs_b.map(weight);
                            let scores = moe::selection_scores(
                                &probs,
                                bias.as_ref().map(|b| &b[..]),
                                n_expert_groups,
                                n_expert_groups_used,
                            );
                            moe::top_experts(&scores, n_used)
                        };
                        // `norm_w`: renormalize the selected weights to sum to 1 before scaling
                        // (softmax MoE); llama4 uses the raw sigmoid prob × scale (no renorm).
                        // Hash routing changes nothing here — llama.cpp reads the weights out of
                        // the same unbiased `probs` at the selected ids on either path.
                        let w: Vec<f32> = if norm_w {
                            let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                            idx.iter().map(|&e| probs[e] / wsum * scale).collect()
                        } else {
                            idx.iter().map(|&e| probs[e] * scale).collect()
                        };
                        RouteRow { idx, w }
                    });

                    // ── Phase 3 (threadpool restructure): the per-expert rayon fan-out became a
                    // STAGED pipeline of flat task lists on the spin-pool — the same design the
                    // Vulkan backend's single-dispatch-per-stage batched MoE uses. Work is
                    // reordered, never re-derived: every kernel sees the same bytes in the same
                    // per-output order as the old per-expert calls, so outputs stay bit-identical
                    // (the (row, rank) bookkeeping below replays the original per-row summation
                    // order exactly like the old `row_slots` scatter did).
                    //
                    // pair = one (row → expert) routing, grouped contiguously by expert:
                    let mut buckets: Vec<(usize, usize, usize)> = Vec::new(); // (expert, p0, count)
                    let mut pair_row: Vec<usize> = Vec::new();
                    let mut pair_bucket: Vec<u32> = Vec::new();
                    let mut pair_local: Vec<u32> = Vec::new();
                    // llama4 `weight_before_ffn`: the routing weight per (row→expert) pair, folded
                    // into that pair's gate/up activations (stage C) instead of the output (stage E).
                    let mut pair_w: Vec<f32> = Vec::new();
                    // slot_pair[r][rank] = flat pair index — stage E's replay map.
                    let mut slot_pair: Vec<Vec<usize>> = routes
                        .iter()
                        .map(|rr| vec![usize::MAX; rr.idx.len()])
                        .collect();
                    {
                        let mut expert_tasks: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_expert];
                        for (r, rr) in routes.iter().enumerate() {
                            for (rank, &e) in rr.idx.iter().enumerate() {
                                expert_tasks[e].push((r, rank));
                            }
                        }
                        for (e, tasks) in expert_tasks.iter().enumerate() {
                            if tasks.is_empty() {
                                continue;
                            }
                            let p0 = pair_row.len();
                            for (i, &(r, rank)) in tasks.iter().enumerate() {
                                slot_pair[r][rank] = pair_row.len();
                                pair_row.push(r);
                                pair_bucket.push(buckets.len() as u32);
                                pair_local.push(i as u32);
                                pair_w.push(routes[r].w[rank]);
                            }
                            buckets.push((e, p0, pair_row.len() - p0));
                        }
                    }
                    let n_pairs = pair_row.len();
                    // The banks, for exactly the experts routing picked. Gate first, down second,
                    // up (split banks only) last.
                    let routed: Vec<usize> = buckets.iter().map(|&(e, _, _)| e).collect();
                    let mut bank_bufs = vec![cpu_buf(gbuf), cpu_buf(dbuf)];
                    bank_bufs.extend(ubuf);
                    if g.weight_reuse == infr_core::graph::WeightReuse::Done {
                        let by_count: Vec<(usize, usize)> =
                            buckets.iter().map(|&(e, _, n)| (e, n)).collect();
                        let last = routes.last().map_or(&[][..], |r| &r.idx[..]);
                        for b in &bank_bufs {
                            if let CpuStore::PagedExperts { store, bank, .. } = &b.store {
                                let share = store.done_keep_share(*bank, bank_bufs.len());
                                kept_experts
                                    .insert(b.uid, paged::done_keep(last, &by_count, share));
                            }
                        }
                    }
                    let mut banks = self
                        .moe_banks(&bank_bufs, n_expert, &routed, g.weight_reuse)?
                        .into_iter();
                    let (gb, db) = (
                        banks.next().expect("gate bank"),
                        banks.next().expect("down bank"),
                    );
                    let ub = banks.next();
                    // `debug.moe_counts`: print the REAL per-expert row-count distribution
                    // for this chunk/layer (this CPU interpreter's top-k routing is bit-identical
                    // to the GPU's, so this is a host-side stand-in for reading the GPU's own
                    // counts/offsets buffers without a mid-graph readback). Used to check whether
                    // `matmul_mmq_experts`' BM=32/64 row-tile heuristic (recorder.rs,
                    // MOE_EXPERT_SMALL_TILE_AVG_ROWS) — picked against a MEAN-BALANCED synthetic
                    // distribution — still holds up against real (possibly skewed) routing.
                    // `debug.moe_counts_dump` additionally dumps the raw
                    // per-expert counts array (for feeding into an isolated GPU tile-choice bench,
                    // e.g. `moe_expert_row_tile_bench_real_skew` in infr-vulkan's gemm_bench.rs).
                    if moe_counts {
                        let mut counts = vec![0usize; n_expert];
                        for &(e, _, c) in buckets.iter() {
                            counts[e] = c;
                        }
                        let mut sorted = counts.clone();
                        sorted.sort_unstable();
                        let sum: usize = counts.iter().sum();
                        let max = *sorted.last().unwrap_or(&0);
                        let min = *sorted.first().unwrap_or(&0);
                        let mean = sum as f64 / n_expert as f64;
                        let p50 = sorted[n_expert / 2];
                        let p90 = sorted[(n_expert * 9) / 10];
                        let nz = counts.iter().filter(|&&c| c > 0).count();
                        let buckets_hist = [0usize, 4, 8, 16, 24, 32, 48, 64, usize::MAX];
                        let mut hist = vec![0usize; buckets_hist.len() - 1];
                        for &c in counts.iter() {
                            for w in 0..buckets_hist.len() - 1 {
                                if c >= buckets_hist[w] && c < buckets_hist[w + 1] {
                                    hist[w] += 1;
                                    break;
                                }
                            }
                        }
                        let waste = |bm: usize| -> usize {
                            counts
                                .iter()
                                .map(|&c| c.div_ceil(bm).max(1) * bm - c)
                                .sum::<usize>()
                        };
                        let real_tiles = |bm: usize| -> usize {
                            counts.iter().map(|&c| c.div_ceil(bm)).sum::<usize>()
                        };
                        // per-expert OPTIMAL tile from {16,32,64} (min waste, tie -> larger BM
                        // for fewer real tiles / less fixed per-workgroup overhead):
                        let opt_waste: usize = counts
                            .iter()
                            .map(|&c| {
                                [16usize, 32, 64]
                                    .iter()
                                    .map(|&bm| c.div_ceil(bm).max(1) * bm - c)
                                    .min()
                                    .unwrap()
                            })
                            .sum();
                        tracing::info!(
                            "MOE_COUNTS rows={rows} n_expert={n_expert} n_used={n_used} \
                             n_pairs={n_pairs} nonzero={nz} min={min} max={max} mean={mean:.2} \
                             p50={p50} p90={p90} hist(0-3,4-7,8-15,16-23,24-31,32-47,48-63,64+)={hist:?} \
                             waste16={} waste32={} waste64={} waste_opt={opt_waste} \
                             tiles16={} tiles32={} tiles64={}",
                            waste(16),
                            waste(32),
                            waste(64),
                            real_tiles(16),
                            real_tiles(32),
                            real_tiles(64),
                        );
                        if moe_counts_dump {
                            tracing::info!("MOE_COUNTS_RAW {counts:?}");
                        }
                    }
                    let out_gu = if fused_gate_up { 2 * nffx } else { nffx };
                    // o-major output offsets per bucket for the gate_up GEMM stage.
                    let mut gu_off = vec![0usize; buckets.len() + 1];
                    for (b, &(_, _, count)) in buckets.iter().enumerate() {
                        gu_off[b + 1] = gu_off[b] + out_gu * count;
                    }
                    // 8-aligned o-chunks (64 rows) per bucket — the flat GEMM task list. Dynamic
                    // claiming spreads a straggler expert's chunks over every idle thread, which
                    // replaces both the old per-expert fan-out AND its nested straggler split.
                    let gemm_tasks = |out_f: usize| -> Vec<(u32, u32, u32)> {
                        let mut t = Vec::new();
                        for b in 0..buckets.len() as u32 {
                            let mut o = 0u32;
                            while (o as usize) < out_f {
                                let o1 = (o + 64).min(out_f as u32);
                                t.push((b, o, o1));
                                o = o1;
                            }
                        }
                        t
                    };

                    // ── Stage A: gate/up activations, quantized ONCE per distinct row (the old
                    // per-expert gather re-quantized a row for every expert it routed to), then
                    // cloned per pair so each bucket owns a contiguous slice.
                    let t_a = std::time::Instant::now();
                    let g_kind = expert_acts_kind(gdt, ne, int8_ok);
                    let u_kind = if fused_gate_up {
                        g_kind
                    } else {
                        expert_acts_kind(udt, ne, int8_ok)
                    };
                    let need = |k: ActsKind| g_kind == k || u_kind == k;
                    let q8_pairs: Vec<Q8> = if need(ActsKind::Super) {
                        let q8_rows: Vec<Q8> =
                            pool.collect(rows, &|r| quantize_q8(&xs[r * ne..r * ne + ne]));
                        pool.collect(n_pairs, &|p| q8_rows[pair_row[p]].clone())
                    } else {
                        Vec::new()
                    };
                    let q832_pairs: Vec<Q8x32> = if need(ActsKind::Blk32) {
                        let q_rows: Vec<Q8x32> =
                            pool.collect(rows, &|r| quantize_q8_32(&xs[r * ne..r * ne + ne]));
                        pool.collect(n_pairs, &|p| q_rows[pair_row[p]].clone())
                    } else {
                        Vec::new()
                    };
                    let xin_pairs: Vec<f32> = if need(ActsKind::Raw) {
                        let mut v = vec![0f32; n_pairs * ne];
                        pool.for_chunks_mut(&mut v, ne, 4, &|p, dstrow| {
                            dstrow.copy_from_slice(&xs[pair_row[p] * ne..pair_row[p] * ne + ne]);
                        });
                        v
                    } else {
                        Vec::new()
                    };
                    let acts_for = |k: ActsKind, b: usize| -> ExpertActs {
                        let (_, p0, count) = buckets[b];
                        match k {
                            ActsKind::Super => ExpertActs::Super(&q8_pairs[p0..p0 + count]),
                            ActsKind::Blk32 => ExpertActs::Blk32(&q832_pairs[p0..p0 + count]),
                            ActsKind::Raw => {
                                ExpertActs::Raw(&xin_pairs[p0 * ne..(p0 + count) * ne])
                            }
                        }
                    };
                    let t_gather = t_a.elapsed();

                    // ── Stage B: gate_up GEMMs, o-major per bucket. Cached ilv packs (fused Q4_K
                    // + VNNI only, exactly the old gate) are fetched per bucket up front.
                    let t_b = std::time::Instant::now();
                    #[cfg(target_arch = "x86_64")]
                    let gu_packs: Vec<Option<std::sync::Arc<Q4kPack>>> = {
                        let packable = int8_ok
                            && fused_gate_up
                            && gdt == DType::Q4K
                            && ne.is_multiple_of(256)
                            && (2 * nffx).is_multiple_of(8)
                            && q4k_expert_repack_available();
                        if packable {
                            pool.collect(buckets.len(), &|b| {
                                let e = buckets[b].0;
                                // Keyed by the bank buffer's uid + this expert's byte band inside
                                // it, never the slice address — see `repack::PackKey`.
                                Some(self.q4k_pack_for(
                                    (cpu_buf(gbuf).uid, e * gst, gst),
                                    gb.expert(e),
                                    ne,
                                    2 * nffx,
                                ))
                            })
                        } else {
                            vec![None; buckets.len()]
                        }
                    };
                    #[cfg(not(target_arch = "x86_64"))]
                    let gu_packs: Vec<Option<std::sync::Arc<Q4kPack>>> = vec![None; buckets.len()];
                    let mut gu_all = rc.for_overwrite(gu_off[buckets.len()]);
                    let mut up_all: Vec<f32> = if fused_gate_up {
                        Vec::new()
                    } else {
                        rc.for_overwrite(gu_off[buckets.len()])
                    };
                    {
                        let tasks = gemm_tasks(out_gu);
                        let gu_ptr = pool::SendPtr::new(gu_all.as_mut_ptr());
                        pool.run(tasks.len(), &|t| {
                            let (b, o0, o1) = tasks[t];
                            let (b, o0, o1) = (b as usize, o0 as usize, o1 as usize);
                            let (e, _, count) = buckets[b];
                            // SAFETY: (bucket, o-range) slices of gu_all are disjoint per task.
                            let dst = unsafe {
                                std::slice::from_raw_parts_mut(
                                    gu_ptr.get().add(gu_off[b] + o0 * count),
                                    (o1 - o0) * count,
                                )
                            };
                            expert_gemm_range::<CAP>(
                                gb.expert(e),
                                gdt,
                                ne,
                                out_gu,
                                &acts_for(g_kind, b),
                                gu_packs[b].as_deref(),
                                o0,
                                o1,
                                dst,
                            );
                        });
                        if !fused_gate_up {
                            let ub = ub.as_ref().expect("split gate/up: up_exps missing");
                            let up_ptr = pool::SendPtr::new(up_all.as_mut_ptr());
                            pool.run(tasks.len(), &|t| {
                                let (b, o0, o1) = tasks[t];
                                let (b, o0, o1) = (b as usize, o0 as usize, o1 as usize);
                                let (e, _, count) = buckets[b];
                                // SAFETY: disjoint (bucket, o-range) slices, as above.
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        up_ptr.get().add(gu_off[b] + o0 * count),
                                        (o1 - o0) * count,
                                    )
                                };
                                expert_gemm_range::<CAP>(
                                    ub.expert(e),
                                    udt,
                                    ne,
                                    nffx,
                                    &acts_for(u_kind, b),
                                    None,
                                    o0,
                                    o1,
                                    dst,
                                );
                            });
                        }
                    }
                    let t_gate_up = t_b.elapsed();

                    // ── Stage C: gated activation per pair (strided o-major reads — the same
                    // values the old un-transpose+split produced), quantized straight into the
                    // representation the DOWN bank's dtype wants.
                    let t_c = std::time::Instant::now();
                    let d_kind = expert_acts_kind(ddt, nffx, int8_ok);
                    let act_row_of = |p: usize| -> Vec<f32> {
                        let b = pair_bucket[p] as usize;
                        let i = pair_local[p] as usize;
                        let count = buckets[b].2;
                        // llama4 weight-before-FFN: scale gate & up by the pair's routing weight
                        // BEFORE the activation. Exact (`gate`/`up` are linear, so `gate(w·x) =
                        // w·gate(x)`): `silu(w·gate)·(w·up)` == applying `w` to the FFN input.
                        let wp = if weight_before { pair_w[p] } else { 1.0 };
                        let mut row = vec![0f32; nffx];
                        if fused_gate_up {
                            let gu = &gu_all[gu_off[b]..gu_off[b] + 2 * nffx * count];
                            gated_act_row(
                                act,
                                swiglu_clamp,
                                &mut row,
                                |f| wp * gu[f * count + i],
                                |f| wp * gu[(nffx + f) * count + i],
                            );
                        } else {
                            let gt = &gu_all[gu_off[b]..gu_off[b] + nffx * count];
                            let ut = &up_all[gu_off[b]..gu_off[b] + nffx * count];
                            gated_act_row(
                                act,
                                swiglu_clamp,
                                &mut row,
                                |f| wp * gt[f * count + i],
                                |f| wp * ut[f * count + i],
                            );
                        }
                        row
                    };
                    let (dq8_pairs, dq832_pairs, dact_pairs): (Vec<Q8>, Vec<Q8x32>, Vec<f32>) =
                        match d_kind {
                            ActsKind::Super => (
                                pool.collect(n_pairs, &|p| quantize_q8(&act_row_of(p))),
                                Vec::new(),
                                Vec::new(),
                            ),
                            ActsKind::Blk32 => (
                                Vec::new(),
                                pool.collect(n_pairs, &|p| quantize_q8_32(&act_row_of(p))),
                                Vec::new(),
                            ),
                            ActsKind::Raw => {
                                let mut v = vec![0f32; n_pairs * nffx];
                                pool.for_chunks_mut(&mut v, nffx, 1, &|p, dstrow| {
                                    dstrow.copy_from_slice(&act_row_of(p));
                                });
                                (Vec::new(), Vec::new(), v)
                            }
                        };
                    let dacts_for = |b: usize| -> ExpertActs {
                        let (_, p0, count) = buckets[b];
                        match d_kind {
                            ActsKind::Super => ExpertActs::Super(&dq8_pairs[p0..p0 + count]),
                            ActsKind::Blk32 => ExpertActs::Blk32(&dq832_pairs[p0..p0 + count]),
                            ActsKind::Raw => {
                                ExpertActs::Raw(&dact_pairs[p0 * nffx..(p0 + count) * nffx])
                            }
                        }
                    };
                    let t_act = t_c.elapsed();

                    // ── Stage D: down GEMMs, each task's o-major block un-transposed straight
                    // into its pairs' rows of `y_pairs` with the per-expert `down_scale` folded
                    // in (`y = s * raw`, the same scale-then-weight order the old per-expert
                    // loop applied). The block is still in cache when it is scattered; a separate
                    // un-transpose pass over a whole-layer o-major buffer read a fresh line per
                    // element (stride `count`).
                    let t_d = std::time::Instant::now();
                    let mut y_pairs = rc.for_overwrite(n_pairs * ne);
                    {
                        let tasks = gemm_tasks(ne);
                        let y_ptr = pool::SendPtr::new(y_pairs.as_mut_ptr());
                        pool.run(tasks.len(), &|t| {
                            let (b, o0, o1) = tasks[t];
                            let (b, o0, o1) = (b as usize, o0 as usize, o1 as usize);
                            let (e, p0, count) = buckets[b];
                            let s_e = dscale.as_ref().map(|ds| ds[e]);
                            moe::with_block_scratch((o1 - o0) * count, |blk| {
                                expert_gemm_range::<CAP>(
                                    db.expert(e),
                                    ddt,
                                    nffx,
                                    ne,
                                    &dacts_for(b),
                                    None,
                                    o0,
                                    o1,
                                    blk,
                                );
                                for i in 0..count {
                                    // SAFETY: pair rows `p0..p0 + count` belong to bucket `b`
                                    // alone and columns `o0..o1` to this task, so tasks write
                                    // disjoint ranges of `y_pairs`, which outlives the `run`.
                                    let yrow = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            y_ptr.get().add((p0 + i) * ne + o0),
                                            o1 - o0,
                                        )
                                    };
                                    for (d, y) in yrow.iter_mut().enumerate() {
                                        *y = match s_e {
                                            Some(s_e) => blk[d * count + i] * s_e,
                                            None => blk[d * count + i],
                                        };
                                    }
                                }
                            });
                        });
                    }
                    let t_down = t_d.elapsed();
                    if prof_ops {
                        let (mut bmin, mut bmax, mut single) = (usize::MAX, 0usize, 0usize);
                        for &(_, _, c) in &buckets {
                            bmin = bmin.min(c);
                            bmax = bmax.max(c);
                            single += (c == 1) as usize;
                        }
                        tracing::info!(
                            "[prof-moe] gather {:.2}ms gate_up {:.2}ms act {:.2}ms down {:.2}ms  bucket[min={},max={}] active={} single={}",
                            t_gather.as_secs_f64() * 1e3,
                            t_gate_up.as_secs_f64() * 1e3,
                            t_act.as_secs_f64() * 1e3,
                            t_down.as_secs_f64() * 1e3,
                            bmin,
                            bmax,
                            buckets.len(),
                            single,
                        );
                    }

                    // ── Stage E: accumulate each row's expert contributions in the SAME order
                    // the original per-row loop used (rank order = idx sorted by descending
                    // router prob) — bit-identical to the old row_slots scatter+accumulate.
                    let mut out = rc.zeroed(rows * ne);
                    pool.for_chunks_mut(&mut out, ne, 1, &|r, orow| {
                        let rr = &routes[r];
                        for (rank, &p) in slot_pair[r].iter().enumerate() {
                            debug_assert_ne!(p, usize::MAX);
                            // weight_before already folded the routing weight into stage C.
                            let w_e = if weight_before { 1.0 } else { rr.w[rank] };
                            let y = &y_pairs[p * ne..p * ne + ne];
                            for (o, &yv) in orow.iter_mut().zip(y) {
                                *o += w_e * yv;
                            }
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                    for stage in [gu_all, up_all, y_pairs, xs, rxs] {
                        rc.give(stage);
                    }
                }
                Op::MoeRoute {
                    router_x,
                    router,
                    exp_probs_b,
                    dst_ids,
                    dst_scores,
                    ne,
                    n_expert,
                    n_used,
                    gating,
                    n_expert_groups,
                    n_expert_groups_used,
                } => {
                    // `Op::MoeFfn`'s selection and nothing after it, through the helpers that op
                    // routes with, so the two select the same experts from the same operands.
                    let (ne, n_expert, n_used) = (ne as usize, n_expert as usize, n_used as usize);
                    let rxs = rc.copy_of(&vals[router_x.0 as usize]);
                    let rows = rxs.len() / ne;
                    let rbuf = bindings.get(router).expect("cpu backend: unbound router");
                    refuse_relock(
                        "Op::MoeRoute",
                        ("router", rbuf),
                        &[("exp_probs_b", exp_probs_b.and_then(|id| bindings.get(id)))],
                    );
                    let logits_all = {
                        let rbytes = cpu_buf(rbuf).read();
                        moe::router_logits::<CAP>(
                            self.pool(),
                            &rbytes,
                            g.desc(router).dtype,
                            ne,
                            n_expert,
                            &rxs,
                            rows,
                            !reference,
                        )
                    };
                    let bias = exp_probs_b.map(weight);
                    let mut ids = rc.zeroed(rows * n_used);
                    let mut scores_all = dst_scores.map(|_| Vec::with_capacity(rows * n_expert));
                    for r in 0..rows {
                        let probs = moe::router_probs(
                            &logits_all[r * n_expert..(r + 1) * n_expert],
                            gating,
                        );
                        let scores = moe::selection_scores(
                            &probs,
                            bias.as_ref().map(|b| &b[..]),
                            n_expert_groups,
                            n_expert_groups_used,
                        );
                        for (slot, e) in ids[r * n_used..(r + 1) * n_used]
                            .iter_mut()
                            .zip(moe::top_experts(&scores, n_used))
                        {
                            *slot = f32::from_bits(e as u32);
                        }
                        if let Some(all) = &mut scores_all {
                            all.extend_from_slice(&scores);
                        }
                    }
                    rc.set(&mut vals, dst_ids.0 as usize, ids);
                    if let (Some(id), Some(all)) = (dst_scores, scores_all) {
                        rc.set(&mut vals, id.0 as usize, all);
                    }
                }
                Op::Conv1dSilu {
                    x,
                    weight: w,
                    state,
                    dst,
                    rows,
                    channels,
                    kernel,
                } => {
                    let (rr, cc, kk) = (rows as usize, channels as usize, kernel as usize);
                    let ws = weight(w); // [channels, kernel] row-major (per-channel kernel)
                    debug_assert_ne!(x, state, "Conv1dSilu reads its own history as input");
                    // Taken out for the call so the history can be written while `x` is read.
                    let mut st = std::mem::take(&mut vals[state.0 as usize]);
                    let mut out = rc.for_overwrite(rr * cc);
                    let xs = &vals[x.0 as usize]; // [rows, channels]
                    conv1d_silu(self.pool(), xs, &ws, &mut st, &mut out, rr, cc, kk);
                    vals[state.0 as usize] = st;
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::ShortConv {
                    bcx,
                    weight: w,
                    state,
                    dst,
                    rows,
                    channels,
                    kernel,
                } => {
                    let (rr, cc, kk) = (rows as usize, channels as usize, kernel as usize);
                    let ws = weight(w); // [channels, kernel]
                    debug_assert_ne!(bcx, state, "ShortConv reads its own history as input");
                    // Taken out for the call so the history can be written while `bcx` is read.
                    let mut st = std::mem::take(&mut vals[state.0 as usize]);
                    let mut out = rc.for_overwrite(rr * cc);
                    short_conv(
                        self.pool(),
                        &vals[bcx.0 as usize],
                        &ws,
                        &mut st,
                        &mut out,
                        rr,
                        cc,
                        kk,
                    );
                    vals[state.0 as usize] = st;
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::DeltaNet {
                    q,
                    k,
                    v,
                    b,
                    a,
                    a_coef,
                    dt_bias,
                    state,
                    dst,
                    rows,
                    n_vhead,
                    n_khead,
                    head_k,
                    head_v,
                    eps,
                    src_stride,
                    ..
                } => {
                    let (rr, nv, nk, kd, vd) = (
                        rows as usize,
                        n_vhead as usize,
                        n_khead as usize,
                        head_k as usize,
                        head_v as usize,
                    );
                    let qf_raw = vals[q.0 as usize].clone();
                    let src_stride = src_stride as usize;
                    // For strided DeltaNet, q/k/v share one buffer; extract packed arrays. Only
                    // `qf_raw` is read on that path — defer the k/v clones to the non-strided branch
                    // that actually consumes them.
                    let (qf, kf, vf) = if src_stride > 0 {
                        let mut qv = vec![0f32; rr * nk * kd];
                        let mut kv = vec![0f32; rr * nk * kd];
                        let mut vv = vec![0f32; rr * nv * vd];
                        for t in 0..rr {
                            let row = t * src_stride;
                            qv[t * nk * kd..(t + 1) * nk * kd]
                                .copy_from_slice(&qf_raw[row..row + nk * kd]);
                            kv[t * nk * kd..(t + 1) * nk * kd]
                                .copy_from_slice(&qf_raw[row + nk * kd..row + 2 * nk * kd]);
                            vv[t * nv * vd..(t + 1) * nv * vd].copy_from_slice(
                                &qf_raw[row + 2 * nk * kd..row + 2 * nk * kd + nv * vd],
                            );
                        }
                        (qv, kv, vv)
                    } else {
                        let kf_raw = vals[k.0 as usize].clone();
                        let vf_raw = vals[v.0 as usize].clone();
                        (qf_raw, kf_raw, vf_raw)
                    };
                    // [rows, nk*kd], [rows, nk*kd], [rows, nv*vd]
                    let bf = vals[b.0 as usize].clone(); // [rows, nv]
                    let af = vals[a.0 as usize].clone();
                    let acoef = weight(a_coef);
                    let dtb = weight(dt_bias);
                    let st = &mut vals[state.0 as usize]; // [nv, kd, vd]
                                                          // The token scan is inherently sequential (state S carries across tokens), but
                                                          // value heads are fully independent — parallelize the per-head t-scan (see
                                                          // `deltanet_scan`). Bit-identical to the old serial nest: same float ops, order.
                    let out = deltanet_scan(
                        self.pool(),
                        &qf,
                        &kf,
                        &vf,
                        &bf,
                        &af,
                        &acoef,
                        &dtb,
                        st,
                        rr,
                        nv,
                        nk,
                        kd,
                        vd,
                        eps,
                    );
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::Mla {
                    q,
                    k_cache,
                    wk_b,
                    wv_b,
                    dst,
                    rows,
                    kv_len,
                    n_head,
                    q_head_dim,
                    kv_lora_rank,
                    qk_nope_dim,
                    qk_rope_dim,
                    v_head_dim,
                    scale,
                    mask,
                    pos,
                    theta,
                    freq_factors,
                    key_bias,
                } => {
                    op_tap::record(op_tap::TappedOp::Mla { rows });
                    if let Some(b) = key_bias {
                        if let Err(e) = infr_core::graph::check_score_bias(
                            g,
                            b,
                            rows,
                            kv_len,
                            "Op::Mla key_bias",
                        ) {
                            panic!("cpu backend: {e}");
                        }
                    }
                    let (rows, kv_len, nh, qhd, kv_lora, np, qkr, vhd) = (
                        rows as usize,
                        kv_len as usize,
                        n_head as usize,
                        q_head_dim as usize,
                        kv_lora_rank as usize,
                        qk_nope_dim as usize,
                        qk_rope_dim as usize,
                        v_head_dim as usize,
                    );
                    let key_len = kv_lora + qkr;
                    let qs = &vals[q.0 as usize];
                    // K cache: one key_len-wide row per token. V = first kv_lora columns of same row.
                    let kbuf = bindings.get(k_cache).expect("cpu backend: unbound k_cache");
                    // `k_cache` stays locked while `weight` may read `wk_b` and `wv_b`.
                    refuse_relock(
                        "Op::Mla",
                        ("k_cache", kbuf),
                        &[("wk_b", bindings.get(wk_b)), ("wv_b", bindings.get(wv_b))],
                    );
                    let kguard = cpu_buf(kbuf).read();
                    let cap_rows = g.desc(k_cache).numel() / key_len.max(1);
                    let need = kv_len.min(cap_rows) * key_len;
                    let ks = dequant_cache_prefix(&kguard, g.desc(k_cache).dtype, need);
                    // Weights: wk_b [n_head, kv_lora, qk_nope], wv_b [n_head, kv_lora, v_head].
                    let wk = weight(wk_b);
                    let wv = weight(wv_b);
                    // Masking
                    let (window, canvas_lo) = match mask {
                        AttnMask::Causal => (0usize, None),
                        AttnMask::SlidingWindow(w) => (w, None),
                        AttnMask::Canvas { lo } => (0usize, Some(lo)),
                    };
                    let hf = qkr / 2; // rope pair count
                    let mla_freqs = self.rope_freqs(theta, qkr);
                    let ff = freq_factors.map(|f| vals[f.0 as usize].clone());
                    // deepseek32's top-k score mask `[rows, kv_len]` (0 on the selected keys,
                    // -inf elsewhere) — indexed by KEY POSITION, not by the ring row. `None` on
                    // deepseek2, where the arithmetic below is byte-for-byte what it was.
                    let kb = key_bias.map(|b| vals[b.0 as usize].clone());
                    let mut out = rc.zeroed(rows * nh * vhd);
                    self.pool()
                        .for_chunks_mut(&mut out, vhd, 2, &|i, ob_slice| {
                            let ti = i / nh;
                            let h = i % nh;
                            let abs = pos as usize + ti;
                            let (lo, hi) = match canvas_lo {
                                Some(clo) => (clo, kv_len),
                                None => {
                                    let lo = if window > 0 && abs + 1 > window {
                                        abs + 1 - window
                                    } else {
                                        0
                                    };
                                    (lo, abs + 1)
                                }
                            };
                            let n_keys = hi - lo;
                            // Build q_full: absorb q_nope, rope q_pe.
                            let qhead_off = (ti * nh + h) * qhd;
                            let q_nope = &qs[qhead_off..qhead_off + np];
                            let q_pe_raw = &qs[qhead_off + np..qhead_off + qhd];
                            // q_absorbed = wk_b[h]^T @ q_nope  (shape [kv_lora])
                            let wk_off = h * kv_lora * np;
                            let mut q_full = vec![0f32; key_len];
                            for j in 0..kv_lora {
                                let mut s = 0f32;
                                for i in 0..np {
                                    // wk_b[h][i][j]: element [i][j] of the per-head
                                    // [qk_nope_dim, kv_lora_rank] matrix as stored in the GGUF —
                                    // i is the FAST (row) dim: flat = i + j*qk_nope_dim.
                                    s += wk[wk_off + i + j * np] * q_nope[i];
                                }
                                q_full[j] = s;
                            }
                            // Rope q_pe into q_full[kv_lora..]
                            for p in 0..hf {
                                let (i0, i1) = (2 * p, 2 * p + 1);
                                let (s, c) = math::rope_sin_cos(
                                    abs as f32,
                                    mla_freqs[p],
                                    ff.as_ref().map(|ff| ff[p]),
                                );
                                let a = q_pe_raw[i0];
                                let b = q_pe_raw[i1];
                                q_full[kv_lora + i0] = a * c - b * s;
                                q_full[kv_lora + i1] = a * s + b * c;
                            }
                            // SDPA: dot q_full against each cached K row, softmax, accumulate V.
                            let mut sc = vec![0f32; n_keys];
                            let mut mx = f32::NEG_INFINITY;
                            for (jj, scj) in sc.iter_mut().enumerate() {
                                let j = lo + jj;
                                let jr = if cap_rows > 0 { j % cap_rows } else { j };
                                let krow = jr * key_len;
                                // dot(q_full, K row j), plus the optional top-k mask at key j
                                *scj = dots.dot(&q_full, &ks[krow..krow + key_len]) * scale;
                                if let Some(kb) = &kb {
                                    *scj += kb[ti * kv_len + j];
                                }
                                mx = mx.max(*scj);
                            }
                            // Each score becomes its exponential once; the sum and the
                            // weights read the same values.
                            math::exp_sub_slice(&mut sc, mx);
                            let mut l = 0f32;
                            for &e in &sc {
                                l += e;
                            }
                            // Accumulate: V is first kv_lora columns of K. Each key's projected
                            // value row is gathered first, then added in by one `axpy`
                            // (`p.mul_add(v, o)` and `v.mul_add(p, o)` are the same fused value).
                            let mut vrow = vec![0f32; vhd];
                            for (jj, &e) in sc.iter().enumerate() {
                                let j = lo + jj;
                                let jr = if cap_rows > 0 { j % cap_rows } else { j };
                                let p = e / l;
                                let krow = jr * key_len;
                                // attn_out[jj] += p * V[jj] where V[jj] = ks[krow..krow+kv_lora]
                                // Apply wv_b[h] to the accumulated output directly:
                                // ob_slice += p * wv_b[h] @ V[jj]
                                let wv_off = h * kv_lora * vhd;
                                for (o_idx, v) in vrow.iter_mut().enumerate() {
                                    let mut vs = 0f32;
                                    for a_idx in 0..kv_lora {
                                        // wv_b[h][a_idx][o_idx]: element [a][o] with a the FAST
                                        // (row) dim — flat = a + o*kv_lora (GGUF layout).
                                        vs +=
                                            wv[wv_off + a_idx + o_idx * kv_lora] * ks[krow + a_idx];
                                    }
                                    *v = vs;
                                }
                                dots.axpy(&mut ob_slice[..vhd], &vrow, p);
                            }
                        });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::LightningIndexer {
                    q,
                    k_cache,
                    weights,
                    dst,
                    rows,
                    kv_len,
                    n_head,
                    head_dim,
                    top_k,
                    scale,
                    pos,
                    block_ratio,
                } => {
                    assert!(
                        block_ratio >= 1,
                        "cpu Op::LightningIndexer: block_ratio 0 — a key covers at least one \
                         position"
                    );
                    let (rows, kv_len, nh, hd, top_k, pos, block_ratio) = (
                        rows as usize,
                        kv_len as usize,
                        n_head as usize,
                        head_dim as usize,
                        top_k as usize,
                        pos as usize,
                        block_ratio as usize,
                    );
                    assert!(
                        kv_len > 0 && top_k > 0,
                        "cpu Op::LightningIndexer: kv_len {kv_len} / top_k {top_k} — an indexer \
                         over no keys selects nothing; the caller skips it (and its TopkMask)"
                    );
                    assert!(
                        top_k <= kv_len,
                        "cpu Op::LightningIndexer: top_k {top_k} exceeds kv_len {kv_len} — there \
                         are not that many keys to name (llama.cpp clamps with \
                         `min(n_kv, n_indexer_top_k)`)"
                    );
                    let qs = &vals[q.0 as usize];
                    let ws = &vals[weights.0 as usize];
                    // Indexer K cache: ONE head_dim-wide row per token (MQA), key j at row j — no
                    // ring fold (see `Op::LightningIndexer`'s doc: a causal-only mask makes every
                    // position back to 0 eligible, so a wrapped cache could not serve this op).
                    let kbuf = bindings
                        .get(k_cache)
                        .expect("cpu backend: unbound indexer k_cache");
                    let kguard = cpu_buf(kbuf).read();
                    let cap_rows = g.desc(k_cache).numel() / hd.max(1);
                    assert!(
                        cap_rows >= kv_len,
                        "cpu Op::LightningIndexer: indexer cache holds {cap_rows} rows but kv_len \
                         is {kv_len} — a wrapped indexer cache has already lost keys this op must \
                         score"
                    );
                    // Causal: a key whose block the query has not completed is not eligible
                    // (with one position per key, any key past the query's own position), and
                    // the cache holds only kv_len keys. Only the eligible prefix is scored or
                    // read — nothing reads the rest.
                    let his: Vec<usize> = (0..rows)
                        .map(|ti| ((pos + ti + 1) / block_ratio).min(kv_len))
                        .collect();
                    let n_read = his.iter().copied().max().unwrap_or(0) * hd;
                    let kdt = g.desc(k_cache).dtype;
                    let wide_keys;
                    let keys = if kdt == DType::F16 {
                        use half::slice::HalfBitsSliceExt;
                        indexer::Keys::F16(
                            bytemuck::cast_slice::<u8, u16>(&kguard)[..n_read].reinterpret_cast(),
                        )
                    } else {
                        wide_keys = dequant_cache_prefix(&kguard, kdt, n_read);
                        indexer::Keys::F32(&wide_keys)
                    };
                    // Split across the pool by keys, so that one decode row scores on every core.
                    // `rows × keys` f32, in a buffer an earlier op of this execute freed when one
                    // fits (the previous CSA layer's scores, given back below).
                    let buf = rc.for_overwrite(his.iter().copied().max().unwrap_or(0) * rows);
                    let sc =
                        indexer::score::<CAP>(self.pool(), buf, qs, ws, keys, nh, hd, scale, &his);
                    let mut out = vec![0f32; rows * top_k];
                    let tap = indexer_tap::active().then(|| std::sync::Mutex::new(Vec::new()));
                    self.pool()
                        .for_chunks_mut(&mut out, top_k, 1, &|ti, slots| {
                            let row: Vec<f32> = (0..his[ti]).map(|j| sc[j * rows + ti]).collect();
                            indexer::select_top_k(&row, slots);
                            if let Some(tap) = &tap {
                                tap.lock().expect("indexer tap lock").push((ti, row));
                            }
                        });
                    if let Some(tap) = tap {
                        let mut scored = tap.into_inner().expect("indexer tap lock");
                        scored.sort_by_key(|&(ti, _)| ti);
                        let out = &out;
                        indexer_tap::record(|op| {
                            scored
                                .into_iter()
                                .map(move |(ti, scores)| indexer_tap::IndexerRow {
                                    op,
                                    pos: pos + ti,
                                    kept: out[ti * top_k..(ti + 1) * top_k]
                                        .iter()
                                        .map(|v| v.to_bits())
                                        .collect(),
                                    scores,
                                })
                        });
                    }
                    rc.give(sc);
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::TopkMask {
                    idx,
                    dst,
                    rows,
                    kv_len,
                    top_k,
                } => {
                    assert!(
                        kv_len > 0 && top_k > 0,
                        "cpu Op::TopkMask: kv_len {kv_len} / top_k {top_k} — a mask over no keys \
                         selects nothing; the caller skips it"
                    );
                    let (rows, kv_len, top_k) = (rows as usize, kv_len as usize, top_k as usize);
                    let ids = &vals[idx.0 as usize];
                    // `rows × kv_len`: the previous CSA layer's mask buffer when one is free.
                    let mut out = rc.for_overwrite(rows * kv_len);
                    out.fill(f32::NEG_INFINITY);
                    for (t, row) in out.chunks_exact_mut(kv_len).enumerate() {
                        for s in 0..top_k {
                            // i32 index carried as a u32 bit pattern in the f32 slot — the
                            // convention `Op::LightningIndexer` writes and `Op::Argmax` uses.
                            let j = ids[t * top_k + s].to_bits() as usize;
                            assert!(
                                j < kv_len,
                                "cpu Op::TopkMask: index {j} at (row {t}, slot {s}) is past \
                                 kv_len {kv_len} — the indices did not come from a matching \
                                 Op::LightningIndexer"
                            );
                            row[j] = 0.0;
                        }
                    }
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::HyperConnectMix {
                    mixes,
                    scale,
                    base,
                    pre,
                    gates,
                    rows,
                    hc,
                    eps,
                    n_iter,
                } => {
                    let (rows, hc) = (rows as usize, hc as usize);
                    hyper_check_hc(hc, "cpu Op::HyperConnectMix");
                    assert!(
                        n_iter >= 1,
                        "cpu Op::HyperConnectMix: n_iter {n_iter} — Sinkhorn runs at least one \
                         normalisation over src"
                    );
                    let mx = &vals[mixes.0 as usize];
                    let sc = weight(scale);
                    let bs = weight(base);
                    // The head form's `mixes` is the `pre` chunk alone (its `output_hc_fn` is
                    // `{hc_dim, hc}`) — see `Op::HyperConnectMix`'s doc.
                    let (mix_dim, n_scale) = if gates.is_some() {
                        ((2 + hc) * hc, 3)
                    } else {
                        (hc, 1)
                    };
                    assert!(
                        mx.len() >= rows * mix_dim && sc.len() >= n_scale && bs.len() >= mix_dim,
                        "cpu Op::HyperConnectMix: mixes {} / scale {} / base {} too small for rows \
                         {rows}, hc {hc} (needs {}, {n_scale}, {mix_dim})",
                        mx.len(),
                        sc.len(),
                        bs.len(),
                        rows * mix_dim,
                    );
                    let mut pre_out = vec![0f32; rows * hc];
                    for (t, p) in pre_out.chunks_exact_mut(hc).enumerate() {
                        for (h, o) in p.iter_mut().enumerate() {
                            let z = mx[t * mix_dim + h] * sc[0] + bs[h];
                            // sigmoid, then + eps — the FOURTH eps site (`ggml_scale_bias(pre,
                            // 1.0f, eps)`), distinct from the three inside Sinkhorn.
                            *o = math::sigmoid(z) + eps;
                        }
                    }
                    // Every output is built before any is stored: `mx`/`sc`/`bs` borrow `vals`.
                    let gate_out = gates.map(|infr_core::graph::HyperGates { post, comb }| {
                        let mut post_out = vec![0f32; rows * hc];
                        for (t, p) in post_out.chunks_exact_mut(hc).enumerate() {
                            for (h, o) in p.iter_mut().enumerate() {
                                let z = mx[t * mix_dim + hc + h] * sc[1] + bs[hc + h];
                                // sigmoid, then × 2 — NOT + eps. Adjacent chunk, different tail.
                                *o = 2.0 / (1.0 + math::exp(-z));
                            }
                        }
                        let mut comb_out = vec![0f32; rows * hc * hc];
                        // Rows are independent; each is a `hc*hc` Sinkhorn, so the grain is a whole
                        // token's matrix.
                        self.pool()
                            .for_chunks_mut(&mut comb_out, hc * hc, 8, &|t, m| {
                                // comb's chunk starts at 2*hc and its flat index is `dst + hc*src`
                                // — the same layout Sinkhorn and `Op::HyperConnectPost` read.
                                for (i, o) in m.iter_mut().enumerate() {
                                    *o = mx[t * mix_dim + 2 * hc + i] * sc[2] + bs[2 * hc + i];
                                }
                                hyper_sinkhorn(m, hc, eps, n_iter);
                            });
                        (post, post_out, comb, comb_out)
                    });
                    rc.set(&mut vals, pre.0 as usize, pre_out);
                    if let Some((post, post_out, comb, comb_out)) = gate_out {
                        rc.set(&mut vals, post.0 as usize, post_out);
                        rc.set(&mut vals, comb.0 as usize, comb_out);
                    }
                }
                Op::HyperConnectPre {
                    x,
                    weights,
                    dst,
                    rows,
                    hc,
                    n_embd,
                } => {
                    let (rows, hc, ne) = (rows as usize, hc as usize, n_embd as usize);
                    hyper_check_hc(hc, "cpu Op::HyperConnectPre");
                    let xs = &vals[x.0 as usize];
                    let ws = &vals[weights.0 as usize];
                    let mut out = rc.zeroed(rows * ne);
                    self.pool().for_chunks_mut(&mut out, ne, 1, &|t, o| {
                        // dst[t, i] = Σ_h x[t, h, i] * w[t, h], h ASCENDING — the accumulation
                        // order llama.cpp's chain of `ggml_add`s builds up.
                        for h in 0..hc {
                            let w = ws[t * hc + h];
                            let xr = &xs[(t * hc + h) * ne..][..ne];
                            dots.axpy(o, xr, w);
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::HyperConnectPost {
                    x,
                    residual,
                    post,
                    comb,
                    dst,
                    rows,
                    hc,
                    n_embd,
                } => {
                    let (rows, hc, ne) = (rows as usize, hc as usize, n_embd as usize);
                    hyper_check_hc(hc, "cpu Op::HyperConnectPost");
                    let xs = &vals[x.0 as usize];
                    let rs = &vals[residual.0 as usize];
                    let ps = &vals[post.0 as usize];
                    let cs = &vals[comb.0 as usize];
                    let mut out = rc.zeroed(rows * hc * ne);
                    // One chunk per (token, dst stream).
                    self.pool().for_chunks_mut(&mut out, ne, 4, &|c, o| {
                        let (t, d) = (c / hc, c % hc);
                        let xr = &xs[t * ne..][..ne];
                        let g = ps[t * hc + d];
                        for (o, &xv) in o.iter_mut().zip(xr) {
                            *o = xv * g;
                        }
                        for src in 0..hc {
                            // comb[t, dst, src] at `dst + hc*src` — transposing this is the
                            // failure `Op::HyperConnectMix`'s doc warns about.
                            let w = cs[t * hc * hc + d + hc * src];
                            let rr = &rs[(t * hc + src) * ne..][..ne];
                            dots.axpy(o, rr, w);
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
                Op::CompressPool {
                    values,
                    scores,
                    dst,
                    blocks,
                    window,
                    n_embd,
                } => {
                    let (blocks, window, ne) = (blocks as usize, window as usize, n_embd as usize);
                    assert!(
                        window >= 1 && ne >= 1,
                        "cpu Op::CompressPool: window {window} / n_embd {ne} — a block pools at \
                         least one row of at least one channel"
                    );
                    let vs = &vals[values.0 as usize];
                    let ss = &vals[scores.0 as usize];
                    let need = blocks * window * ne;
                    assert!(
                        vs.len() >= need && ss.len() >= need,
                        "cpu Op::CompressPool: values {} / scores {} too small for blocks \
                         {blocks}, window {window}, n_embd {ne} (needs {need})",
                        vs.len(),
                        ss.len(),
                    );
                    let mut out = rc.zeroed(blocks * ne);
                    // One chunk per block. Within it every pass walks the window rows in order and
                    // the channels contiguously, so the softmax reduces over the STRIDED window
                    // axis without ever transposing the slab — the two `ggml_cont`s the reference
                    // pays for its permutes.
                    self.pool().for_chunks_mut(&mut out, ne, 1, &|b, o| {
                        let vb = &vs[b * window * ne..][..window * ne];
                        let sb = &ss[b * window * ne..][..window * ne];
                        let mut m = vec![f32::NEG_INFINITY; ne];
                        for row in sb.chunks_exact(ne) {
                            for (m, &s) in m.iter_mut().zip(row) {
                                *m = m.max(s);
                            }
                        }
                        let mut den = vec![0f32; ne];
                        let mut es = vec![0f32; ne];
                        for (vr, sr) in vb.chunks_exact(ne).zip(sb.chunks_exact(ne)) {
                            for ((e, &m), &s) in es.iter_mut().zip(m.iter()).zip(sr) {
                                *e = s - m;
                            }
                            math::exp_slice(&mut es);
                            for (d, &e) in den.iter_mut().zip(es.iter()) {
                                *d += e;
                            }
                            dots.mul_add_into(o, vr, &es);
                        }
                        for ((o, &d), &m) in o.iter_mut().zip(den.iter()).zip(m.iter()) {
                            // An all-`-inf` window is `0/0` (`s - m` is `NaN` for every lane): the
                            // op defines it as 0.0 on every backend — see `Op::CompressPool`.
                            *o = if m == f32::NEG_INFINITY { 0.0 } else { *o / d };
                        }
                    });
                    rc.set(&mut vals, dst.0 as usize, out);
                }
            }
            if let Some(t0) = __t0 {
                let e = op_times
                    .entry(infr_core::prof::op_label(op, g))
                    .or_insert((0, 0.0));
                e.0 += 1;
                e.1 += t0.elapsed().as_secs_f64();
            }
        }
        if let Some(mut tap) = hash_tap {
            if let Some(last) = g.ops.len().checked_sub(1) {
                tap.record(g, last, &g.ops[last], true, |id| tap_hash(&vals, id));
            }
            tap.finish()?;
        }
        if prof_ops {
            // `Unit::Device`: on a host backend the wall time IS the execution time, so these rows
            // are directly comparable to a GPU backend's and belong in the same exit aggregate.
            let mut p = infr_core::prof::OpProf::new("cpu", infr_core::prof::Unit::Device);
            for (label, (calls, secs)) in op_times {
                p.add_n(label, secs * 1e6, calls);
            }
            p.flush();
        }

        if g.weight_reuse == infr_core::graph::WeightReuse::Done {
            demote_paged_reads(g, bindings, &kept_experts);
        }

        // Write back the buffers the model reads after execute: Outputs (logits) and mutated f32
        // Inputs (conv/recurrent state). KV caches were written in place by `WriteKv`, so they're
        // skipped — no full-cache copy. Weights are read-only; positions are I32, unchanged. The
        // predicate is the shared `infr_core::exec::writes_back` (metal selects the same set).
        for id in infr_core::exec::write_back_targets(g) {
            if let Some(buf) = bindings.get(id) {
                let mut d = cpu_buf(buf).owned();
                d.copy_from_slice(bytemuck::cast_slice(&vals[id.0 as usize]));
            }
        }
        Ok(())
    }
}

/// [`infr_core::graph::WeightReuse::Done`]: make every paged weight some op of `g` read the next
/// victim of its pool — a whole weight, and every expert of a bank paged per expert but those
/// `keep` names for that bank ([`paged::done_keep`]) — so the reads that follow evict what this
/// graph was the last to need before anything still live.
fn demote_paged_reads(g: &Graph, bindings: &Bindings, keep: &HashMap<u64, Vec<usize>>) {
    let mut seen = std::collections::HashSet::new();
    for op in &g.ops {
        for id in op.io().0 {
            let Some(cb) = bindings
                .get(id)
                .and_then(|b| b.as_any().downcast_ref::<CpuBuffer>())
            else {
                continue;
            };
            if !seen.insert(cb.uid) {
                continue;
            }
            match &cb.store {
                CpuStore::Paged { store, id, .. } => store.demote(*id),
                CpuStore::PagedExperts { store, bank, .. } => {
                    store.demote_experts(*bank, keep.get(&cb.uid).map_or(&[][..], |k| &k[..]))
                }
                CpuStore::Owned(_) | CpuStore::Mapped(_) => {}
            }
        }
    }
}

/// qwen35's `Op::Conv1dSilu`: [`causal_conv`] over the `[rows, channels]` input `xs`, then SiLU.
/// Returns the output and leaves `st` holding the trailing `kk-1` rows of `xs`.
#[allow(clippy::too_many_arguments)]
fn conv1d_silu(
    pool: &pool::SpinPool,
    xs: &[f32],
    ws: &[f32],
    st: &mut [f32],
    out: &mut [f32],
    rr: usize,
    cc: usize,
    kk: usize,
) {
    causal_conv(
        pool,
        &PlainRows { xs, cc },
        ws,
        st,
        out,
        rr,
        cc,
        kk,
        &|_, orow| math::silu_slice(orow),
    );
}

/// LFM's gated short convolution (`Op::ShortConv`): for each row `t` of `bcx = [B | C | X]`
/// (`[rows, 3·channels]`), `out[t] = C[t] ⊙ conv(B ⊙ X)[t]`, the conv being [`causal_conv`] with
/// no activation. `st` holds the trailing `kk-1` rows of `B ⊙ X` and is advanced in place.
///
/// `B ⊙ X` is formed as each tap reads it rather than stored: a multiply per tap is cheaper than
/// a separate pass writing a `[rows, channels]` intermediate, and the products are the same f32
/// values either way, so the result is bit-identical to building `B ⊙ X` first.
#[allow(clippy::too_many_arguments)]
fn short_conv(
    pool: &pool::SpinPool,
    bcx: &[f32],
    ws: &[f32],
    st: &mut [f32],
    out: &mut [f32],
    rr: usize,
    cc: usize,
    kk: usize,
) {
    causal_conv(
        pool,
        &GatedRows { bcx, cc },
        ws,
        st,
        out,
        rr,
        cc,
        kk,
        &|t, orow| {
            let c = &bcx[t * 3 * cc + cc..t * 3 * cc + 2 * cc];
            for (o, &cv) in orow.iter_mut().zip(c) {
                *o *= cv;
            }
        },
    );
}

/// The input sequence of a [`causal_conv`], read a row at a time.
trait ConvRows: Sync {
    /// `acc[ch] += row_r[ch] * w[ch]` for every channel.
    fn madd(&self, r: usize, w: &[f32], acc: &mut [f32]);
    /// `dst[ch] = row_r[ch]` for every channel.
    fn copy_row(&self, r: usize, dst: &mut [f32]);
}

/// A `[rows, channels]` input read as is.
struct PlainRows<'a> {
    xs: &'a [f32],
    cc: usize,
}

impl ConvRows for PlainRows<'_> {
    fn madd(&self, r: usize, w: &[f32], acc: &mut [f32]) {
        let x = &self.xs[r * self.cc..(r + 1) * self.cc];
        for ((a, &xv), &wv) in acc.iter_mut().zip(x).zip(w) {
            *a += xv * wv;
        }
    }

    fn copy_row(&self, r: usize, dst: &mut [f32]) {
        dst.copy_from_slice(&self.xs[r * self.cc..(r + 1) * self.cc]);
    }
}

/// `B ⊙ X` of `[B | C | X]` rows, multiplied as it is read.
struct GatedRows<'a> {
    bcx: &'a [f32],
    cc: usize,
}

impl GatedRows<'_> {
    fn bx(&self, r: usize) -> (&[f32], &[f32]) {
        let row = &self.bcx[r * 3 * self.cc..(r + 1) * 3 * self.cc];
        (&row[..self.cc], &row[2 * self.cc..])
    }
}

impl ConvRows for GatedRows<'_> {
    fn madd(&self, r: usize, w: &[f32], acc: &mut [f32]) {
        let (b, x) = self.bx(r);
        for (((a, &bv), &xv), &wv) in acc.iter_mut().zip(b).zip(x).zip(w) {
            *a += (bv * xv) * wv;
        }
    }

    fn copy_row(&self, r: usize, dst: &mut [f32]) {
        let (b, x) = self.bx(r);
        for ((d, &bv), &xv) in dst.iter_mut().zip(b).zip(x) {
            *d = bv * xv;
        }
    }
}

/// Depthwise causal conv with a fixed per-channel kernel `ws` (`[channels, kk]`) over the virtual
/// sequence `[st ‖ input]`, where `st` is the rolling history of the last `kk-1` input rows carried
/// across token batches: `out[t][ch] = Σ_{j<kk} virt[t+j][ch] · ws[ch·kk + j]`, then `post(t, row)`.
/// Leaves `st` holding the trailing `kk-1` rows of the virtual sequence.
///
/// Every output row reads only the virtual sequence, so rows are independent and run across the
/// pool. Each channel sums its taps from zero in ascending `j`, the order of a serial loop that
/// shifts the history per token, so the two are bit-identical (the tests hold this against such a
/// loop). The taps run as whole rows against a transposed kernel so the channel loop vectorizes.
#[allow(clippy::too_many_arguments)]
fn causal_conv<R: ConvRows>(
    pool: &pool::SpinPool,
    input: &R,
    ws: &[f32],
    st: &mut [f32],
    out: &mut [f32],
    rr: usize,
    cc: usize,
    kk: usize,
    post: &(dyn Fn(usize, &mut [f32]) + Sync),
) {
    let km1 = kk - 1;
    // `wt[j*cc + ch] = ws[ch*kk + j]`: tap `j` of every channel as one contiguous row.
    let mut wt = vec![0f32; kk * cc];
    for ch in 0..cc {
        for j in 0..kk {
            wt[j * cc + ch] = ws[ch * kk + j];
        }
    }
    {
        let hist: &[f32] = st;
        // Chunk `out` by row (chunk == cc, so the chunk index is the row); grain of 4 matches the
        // sibling row-parallel ops.
        pool.for_chunks_mut(out, cc, 4, &|t, orow| {
            orow.fill(0.0);
            for (j, wj) in wt.chunks_exact(cc).enumerate() {
                let i = t + j;
                if i < km1 {
                    let h = &hist[i * cc..(i + 1) * cc];
                    for ((o, &hv), &w) in orow.iter_mut().zip(h).zip(wj) {
                        *o += hv * w;
                    }
                } else {
                    input.madd(i - km1, wj, orow);
                }
            }
            post(t, orow);
        });
    }
    // New history = the trailing `km1` virtual rows. Ascending `j` reads old rows at `j + rr > j`,
    // which are not yet overwritten, so the shift runs in place.
    for j in 0..km1 {
        let i = rr + j;
        if i < km1 {
            st.copy_within(i * cc..(i + 1) * cc, j * cc);
        } else {
            input.copy_row(i - km1, &mut st[j * cc..(j + 1) * cc]);
        }
    }
}

/// Parallel DeltaNet scan: ONE pool task per value head. The outer token scan is inherently
/// sequential (state S carries across tokens), but heads are fully independent — head `h` touches
/// only its own state slice `st[h*kd*vd..]`, writes only its own output columns, and otherwise
/// READS shared inputs (`kh_idx = h % nk` is a read index) — so the whole per-head t-scan runs as
/// one task. Returns `out` [rows, nv*vd]; mutates `st` [nv, kd, vd] in place. Bit-identical to the
/// old serial nest: per head the float ops (l2 norm, normalize, beta, softplus/decay, S*=decay,
/// kv=kᵀS, delta, S+=k⊗delta, out=qᵀS) run in the SAME order — heads never interacted, so
/// cross-head order never mattered, and `pool.for_chunks_mut` is pure scheduling.
///
/// Each task scans its head's state slice where it lies. (It used to scan a copy and a serial
/// gather copied every head back: two copies of the whole state per call, which dominated a
/// one-row decode call.)
#[allow(clippy::too_many_arguments)]
fn deltanet_scan(
    pool: &pool::SpinPool,
    qf: &[f32], // [rows, nk*kd]
    kf: &[f32], // [rows, nk*kd]
    vf: &[f32], // [rows, nv*vd]
    bf: &[f32], // [rows, nv]
    af: &[f32], // [rows, nv]
    acoef: &[f32],
    dtb: &[f32],
    st: &mut [f32], // [nv, kd, vd]
    rr: usize,
    nv: usize,
    nk: usize,
    kd: usize,
    vd: usize,
    eps: f32,
) -> Vec<f32> {
    let qscale = 1.0 / (kd as f32).sqrt();
    let l2 = |slice: &[f32]| -> f32 { (slice.iter().map(|x| x * x).sum::<f32>() + eps).sqrt() };
    // Per-head outputs, [nv, rows, vd]: each head's rows contiguous, so a task owns one run.
    let mut out_h = vec![0f32; nv * rr * vd];
    // One work item per value head: its state slice [kd, vd] and its output run [rows, vd],
    // disjoint from every other head's, so the pool hands each task exactly one.
    let mut work: Vec<(&mut [f32], &mut [f32])> = st[..nv * kd * vd]
        .chunks_mut(kd * vd)
        .zip(out_h.chunks_mut(rr * vd))
        .collect();
    pool.for_chunks_mut(&mut work, 1, 1, &|h, item| {
        let (sh, oh_all) = &mut item[0];
        // GQA: q/k heads TILED to nv value heads → v-head h uses q/k head h % nk.
        let kh_idx = h % nk;
        // Sequential scan over the rows, carrying this head's state S across tokens.
        for t in 0..rr {
            let (qb, vb, bb) = (t * nk * kd, t * nv * vd, t * nv);
            let mut qh = qf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
            let mut kh = kf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
            let vh = &vf[vb + h * vd..vb + h * vd + vd];
            let qn = l2(&qh);
            let kn = l2(&kh);
            for x in qh.iter_mut() {
                *x = *x / qn * qscale;
            }
            for x in kh.iter_mut() {
                *x /= kn;
            }
            let beta = math::sigmoid(bf[bb + h]);
            // softplus(a + dt_bias), then g = a_coef * softplus (≤ 0); decay = exp(g).
            let sp = math::softplus(af[bb + h] + dtb[h]);
            let decay = math::exp(acoef[h] * sp);
            for x in sh.iter_mut() {
                *x *= decay;
            }
            // kv = kᵀS  [vd]
            let mut kv = vec![0f32; vd];
            for kk in 0..kd {
                let kkv = kh[kk];
                let row = &sh[kk * vd..kk * vd + vd];
                for d in 0..vd {
                    kv[d] += kkv * row[d];
                }
            }
            // delta = (v - kv)*beta ; S += k ⊗ delta
            let delta: Vec<f32> = (0..vd).map(|d| (vh[d] - kv[d]) * beta).collect();
            for kk in 0..kd {
                let kkv = kh[kk];
                let row = &mut sh[kk * vd..kk * vd + vd];
                for d in 0..vd {
                    row[d] += kkv * delta[d];
                }
            }
            // out = qᵀS  [vd]
            let oh = &mut oh_all[t * vd..t * vd + vd];
            for kk in 0..kd {
                let qv = qh[kk];
                let row = &sh[kk * vd..kk * vd + vd];
                for d in 0..vd {
                    oh[d] += qv * row[d];
                }
            }
        }
    });
    drop(work);
    if rr == 1 {
        // [nv, 1, vd] is already [1, nv*vd].
        return out_h;
    }
    // Scatter each head's rows into the [rows, nv*vd] layout.
    let mut out = vec![0f32; rr * nv * vd];
    for h in 0..nv {
        for t in 0..rr {
            let dst = t * nv * vd + h * vd;
            let src = (h * rr + t) * vd;
            out[dst..dst + vd].copy_from_slice(&out_h[src..src + vd]);
        }
    }
    out
}

/// Fixtures shared by the crate's unit tests.
#[cfg(test)]
pub(crate) mod test_util {
    use infr_core::tensor::DType;

    /// Every weight format whose activations take the 256-element quantizer (`quantize_q8`), and
    /// so can meet a sparse part (`Q8::hi`).
    pub(crate) const Q8_DTYPES: [DType; 16] = [
        DType::Q4K,
        DType::Q6K,
        DType::Q8_0,
        DType::Q5K,
        DType::Q2K,
        DType::Q3K,
        DType::Iq4Xs,
        DType::Iq2S,
        DType::Iq2Xs,
        DType::Iq2Xxs,
        DType::Iq1S,
        DType::Iq1M,
        DType::Iq3S,
        DType::Iq3Xxs,
        DType::Tq2_0,
        DType::Tq1_0,
    ];

    /// `rows` weight rows of `in_f` in format `dt`: LCG payloads (every code pattern), each
    /// block's scales the format's own magnitudes times `scales[k]`, `k` cycling through the
    /// list block by block (`(b * 5 + b / 7) % len`) so neighbouring blocks differ.
    pub(crate) fn synth_weights(
        dt: DType,
        rows: usize,
        in_f: usize,
        seed: u32,
        scales: &[(f32, f32)],
    ) -> Vec<u8> {
        let spec = infr_core::decode_spec::block_spec(dt);
        let (d, min) = infr_testkit::synth_scales(dt);
        let nblk = rows * in_f / spec.block_elems;
        let mut w = Vec::with_capacity(nblk * spec.block_bytes);
        for b in 0..nblk {
            let mut blk =
                infr_testkit::lcg_bytes(seed ^ (b as u32).wrapping_mul(0x9e37), spec.block_bytes);
            let (fd, fm) = scales[(b * 5 + b / 7) % scales.len()];
            spec.write_scales(&mut blk, d * fd, min * fm);
            w.extend_from_slice(&blk);
        }
        w
    }

    /// The spike patterns the sparse part's tests feed (`(index, value)` over a rest of
    /// magnitude under 1.6, so a spike of 30 or more splits its block): one spike; two in a
    /// block; spikes in several blocks; at the first and last index of a block; at the first and
    /// last index of 8-, 16- and 32-element sub-blocks; in the last block; two equal; two equal
    /// and opposite in one 8-element group; two far apart, whose second needs a second part; one
    /// so large that its scale times an f16 maximum overflows. `in_f` is a multiple of 256 of
    /// at least 512.
    pub(crate) fn spike_patterns(in_f: usize) -> Vec<Vec<(usize, f32)>> {
        let last = in_f - 256;
        vec![
            vec![(5, 400.0)],
            vec![(3, 412.5), (7, -129.3)],
            vec![(10, 300.0), (300, -250.0), (last + 77, 500.0)],
            vec![(0, 350.0), (255, -280.0)],
            vec![(256, 200.0), (511, -330.0)],
            vec![(8, 300.0), (15, -200.0)],
            vec![(16, 310.0), (31, -90.0)],
            vec![(32, 250.0), (63, 260.0)],
            vec![(127, -300.0), (128, 300.0)],
            vec![(in_f - 1, -3.0e4)],
            vec![(last, 500.0), (last + 129, 400.0)],
            vec![(40, 300.0), (41, 300.0)],
            vec![(70, 300.0), (71, -300.0)],
            vec![(40, 300.0), (200, -300.0)],
            vec![(3, 3000.0), (20, 941.0625)],
            vec![(9, 1.0e37)],
        ]
    }

    /// Activation row `r` of `in_f`: ordinary values of magnitude under 1.6 with
    /// [`spike_patterns`]' pattern `r % len` placed over them.
    pub(crate) fn spiky_row(r: usize, in_f: usize) -> Vec<f32> {
        let pats = spike_patterns(in_f);
        let mut x: Vec<f32> = (0..in_f)
            .map(|i| (((i * 29 + r * 13) % 97) as f32 - 48.0) / 31.0)
            .collect();
        for &(i, v) in &pats[r % pats.len()] {
            x[i] = v;
        }
        x
    }
    /// Deterministic quant weight bytes with every block's f16 scale(s) at `d_offs` set to small
    /// finite values (random bits there would be NaN/inf half the time).
    pub(crate) fn det_quant_rows(
        rows: usize,
        nb: usize,
        blk: usize,
        d_offs: &[usize],
        seed: u64,
    ) -> Vec<u8> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut w: Vec<u8> = (0..rows * nb * blk)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u8
            })
            .collect();
        for (k, b) in w.chunks_mut(blk).enumerate() {
            for (j, &off) in d_offs.iter().enumerate() {
                let d = half::f16::from_f32(0.01 + 0.001 * ((k + j) % 7) as f32);
                b[off..off + 2].copy_from_slice(&d.to_le_bytes());
            }
        }
        w
    }

    /// Run `body` on a thread of its own and return what it returns, failing if it takes longer
    /// than a generous bound: a test of a hang otherwise waits forever instead of failing. A panic
    /// in `body` fails the test with that panic's message.
    pub(crate) fn within<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
        let limit = std::time::Duration::from_secs(if cfg!(miri) { 900 } else { 60 });
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
            tx.send(got).expect("the test is waiting for this")
        });
        match rx.recv_timeout(limit) {
            Ok(Ok(v)) => v,
            Ok(Err(p)) => panic!("the test body panicked: {}", payload_text(p)),
            Err(e) => panic!("no result after {limit:?} ({e}): hung"),
        }
    }

    /// Run `f` under `catch_unwind`: the message of its panic, or `None` when it returned.
    pub(crate) fn panic_of(f: impl FnOnce()) -> Option<String> {
        let p = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).err()?;
        Some(payload_text(p))
    }

    /// A panic payload's message: the `String` or `&str` a `panic!` carries.
    fn payload_text(p: Box<dyn std::any::Any + Send>) -> String {
        match p.downcast::<String>() {
            Ok(s) => *s,
            Err(p) => p
                .downcast::<&'static str>()
                .map(|s| (*s).to_owned())
                .unwrap_or_else(|_| "<a payload that is not a string>".to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    thread_local! {
        /// The dispatching thread's placement as `execute` saw it once it had placed itself:
        /// what the placement tests read to see the walk ran placed.
        pub(crate) static EXEC_PLACEMENT: std::cell::Cell<Option<infr_plat::cpu::Placement>> =
            const { std::cell::Cell::new(None) };
    }
    use super::test_util::det_quant_rows;
    use super::*;
    use crate::kernels::{vec_dot_q4k, vec_dot_q5k, vec_dot_q6k};
    use infr_core::graph::MoeGating;
    use infr_core::tensor::TensorDesc;

    /// One owned buffer bound to `Op::Mla`'s `k_cache` and to a weight it reads is refused by
    /// name: the op holds the cache's lock while it reads `wk_b` and `wv_b`, so the second lock on
    /// the same mutex would hang with no message.
    #[test]
    fn mla_refuses_one_owned_buffer_bound_to_the_cache_and_a_weight() {
        super::test_util::within(|| {
            let mut g = Graph::new();
            let q = g.input(TensorDesc::new(vec![1, 8], DType::F32));
            let k_cache = g.input(TensorDesc::new(vec![2, 8], DType::F32));
            let wk_b = g.weight(TensorDesc::new(vec![1, 4, 4], DType::F32));
            let wv_b = g.weight(TensorDesc::new(vec![1, 4, 4], DType::F32));
            let dst = g.output(TensorDesc::new(vec![1, 4], DType::F32));
            g.push(Op::Mla {
                q,
                k_cache,
                wk_b,
                wv_b,
                dst,
                rows: 1,
                kv_len: 2,
                n_head: 1,
                q_head_dim: 8,
                kv_lora_rank: 4,
                qk_nope_dim: 4,
                qk_rope_dim: 4,
                v_head_dim: 4,
                scale: 1.0,
                mask: AttnMask::Causal,
                pos: 1,
                theta: 1e4,
                freq_factors: None,
                key_bias: None,
            });
            let be = CpuBackend::new();
            let alloc = |n: usize| be.alloc(n * 4, BufferUsage::Activations).unwrap();
            let (qb, shared, wvb, yb) = (alloc(8), alloc(16), alloc(16), alloc(4));
            let mut bind = Bindings::new();
            bind.bind(q, qb.as_ref());
            bind.bind(k_cache, shared.as_ref());
            bind.bind(wk_b, shared.as_ref());
            bind.bind(wv_b, wvb.as_ref());
            bind.bind(dst, yb.as_ref());
            let plan = be.compile(&g).unwrap();
            let msg = super::test_util::panic_of(|| {
                let _ = be.execute(plan.as_ref(), &bind);
            })
            .expect("the doubly bound cache was not refused");
            assert!(
                msg.contains("Op::Mla: k_cache and wk_b are bound to the same owned buffer"),
                "{msg}"
            );
        })
    }

    /// B133: with `general.alignment` below 4 an F32 or I32 tensor can start at an address that is
    /// not 4-aligned; reading it must decode, not panic in a cast.
    #[test]
    fn misaligned_tensor_bytes_decode() {
        let want = [1.5f32, -2.0, 3.25];
        let mut buf = vec![0u8; size_of_val(&want) + 4];
        let start = (1..4)
            .find(|&o| !(buf.as_ptr() as usize + o).is_multiple_of(4))
            .unwrap();
        let end = start + size_of_val(&want);
        for (i, v) in want.iter().enumerate() {
            buf[start + 4 * i..start + 4 * (i + 1)].copy_from_slice(&v.to_le_bytes());
        }
        assert_eq!(bytes_to_f32(&buf[start..end], DType::F32), want);
        assert_eq!(&*f32_row(&buf[start..end]), &want[..]);
        for (i, v) in [7i32, -1, 3].iter().enumerate() {
            buf[start + 4 * i..start + 4 * (i + 1)].copy_from_slice(&v.to_le_bytes());
        }
        assert_eq!(bytes_to_f32(&buf[start..end], DType::I32), [7.0, -1.0, 3.0]);
    }

    /// `Op::AttnTemp` against an f64 reference of `llm_graph_input_attn_temp`'s factor, out of
    /// place and in place, on every shared case. Within a few f32 roundings: the factor and the
    /// product are each rounded once, and the logarithm is the correctly rounded f32 one.
    #[test]
    fn attn_temp_matches_the_reference_factor() {
        let be = CpuBackend::new();
        for case in infr_testkit::attn_temp_cases() {
            infr_testkit::check_attn_temp(&be, case, f64::ln, 5e-7);
        }
    }

    // Naive serial reference: a byte-for-byte copy of the ORIGINAL shift-register conv loop. The
    // parallel `conv1d_silu` must reproduce both `out` and the rebuilt `state` exactly.
    fn conv1d_silu_ref(
        xs: &[f32],
        ws: &[f32],
        st: &mut [f32],
        rr: usize,
        cc: usize,
        kk: usize,
        silu: bool,
    ) -> Vec<f32> {
        let mut out = vec![0f32; rr * cc];
        for t in 0..rr {
            let xt = &xs[t * cc..t * cc + cc];
            for ch in 0..cc {
                let mut acc = 0f32;
                for j in 0..kk - 1 {
                    acc += st[j * cc + ch] * ws[ch * kk + j];
                }
                acc += xt[ch] * ws[ch * kk + (kk - 1)];
                out[t * cc + ch] = if silu {
                    acc / (1.0 + math::exp(-acc))
                } else {
                    acc
                };
            }
            for j in 0..kk.saturating_sub(2) {
                for ch in 0..cc {
                    st[j * cc + ch] = st[(j + 1) * cc + ch];
                }
            }
            if kk >= 2 {
                for ch in 0..cc {
                    st[(kk - 2) * cc + ch] = xt[ch];
                }
            }
        }
        out
    }

    /// Deterministic pseudo-random pattern (no RNG) for the conv tests: a hashed ramp in [-1, 1).
    fn conv_pat(seed: usize, i: usize) -> f32 {
        let h = (i
            .wrapping_mul(2654435761)
            .wrapping_add(seed.wrapping_mul(40503)))
            & 0xffff;
        (h as f32 / 32768.0) - 1.0
    }

    // `conv1d_silu` (the parallel `causal_conv`) must be BIT-IDENTICAL (exact f32 equality) to the
    // naive serial reference for BOTH the output and the rebuilt state, across a matrix of
    // (rows, kernel, channels) that spans: real-prefill sizes, decode (rr==1), rr == kk-1, a case
    // with rr < kk-1 (rebuild reads leftover old state), kk==1 (no history), kk==2, small channel
    // counts, and row counts that split across the pool's tasks.
    #[test]
    fn conv1d_silu_bit_identical_to_serial() {
        let pat = conv_pat;
        let be = CpuBackend::new();
        let pool = be.pool();
        for &(rr, kk) in &[
            (8usize, 4usize),
            (1, 4),
            (3, 4),
            (16, 4),
            (5, 1),
            (2, 2),
            (2, 4),
        ] {
            for cc in [40usize, 512] {
                let km1 = kk - 1;
                let xs: Vec<f32> = (0..rr * cc).map(|i| pat(1, i)).collect();
                let ws: Vec<f32> = (0..cc * kk).map(|i| pat(2, i)).collect();
                let st0: Vec<f32> = (0..km1 * cc).map(|i| pat(3, i)).collect();

                let mut st_ref = st0.clone();
                let out_ref = conv1d_silu_ref(&xs, &ws, &mut st_ref, rr, cc, kk, true);

                let mut st_par = st0.clone();
                let mut out_par = vec![f32::NAN; rr * cc];
                conv1d_silu(pool, &xs, &ws, &mut st_par, &mut out_par, rr, cc, kk);

                assert_eq!(
                    out_par, out_ref,
                    "output mismatch at rr={rr} kk={kk} cc={cc}"
                );
                assert_eq!(
                    st_par, st_ref,
                    "rebuilt state mismatch at rr={rr} kk={kk} cc={cc}"
                );
            }
        }
    }

    // `short_conv` must be BIT-IDENTICAL to the composition it fuses — `B ⊙ X`, the serial
    // activation-free conv, then `C ⊙` — for the output and the advanced history, over LFM3's
    // kernel width and the edge shapes of the test above: decode (rr==1), rr < kk-1 (the history
    // shift reads old rows), kk==1 and kk==2, and a row count that splits across the pool's tasks.
    #[test]
    fn short_conv_bit_identical_to_composition() {
        let be = CpuBackend::new();
        let pool = be.pool();
        for &(rr, kk) in &[
            (1usize, 7usize),
            (3, 7),
            (6, 7),
            (64, 7),
            (8, 4),
            (5, 1),
            (2, 2),
        ] {
            for cc in [40usize, 512] {
                let km1 = kk - 1;
                let bcx: Vec<f32> = (0..rr * 3 * cc).map(|i| conv_pat(1, i)).collect();
                let ws: Vec<f32> = (0..cc * kk).map(|i| conv_pat(2, i)).collect();
                let st0: Vec<f32> = (0..km1 * cc).map(|i| conv_pat(3, i)).collect();

                let bx: Vec<f32> = (0..rr * cc)
                    .map(|i| {
                        bcx[(i / cc) * 3 * cc + i % cc] * bcx[(i / cc) * 3 * cc + 2 * cc + i % cc]
                    })
                    .collect();
                let mut st_ref = st0.clone();
                let mut out_ref = conv1d_silu_ref(&bx, &ws, &mut st_ref, rr, cc, kk, false);
                for (i, o) in out_ref.iter_mut().enumerate() {
                    *o *= bcx[(i / cc) * 3 * cc + cc + i % cc];
                }

                let mut st = st0.clone();
                let mut out = vec![f32::NAN; rr * cc];
                short_conv(pool, &bcx, &ws, &mut st, &mut out, rr, cc, kk);

                assert_eq!(out, out_ref, "output mismatch at rr={rr} kk={kk} cc={cc}");
                assert_eq!(st, st_ref, "history mismatch at rr={rr} kk={kk} cc={cc}");
            }
        }
    }

    // Naive serial reference: a byte-for-byte copy of the ORIGINAL DeltaNet scan nest (loop order
    // `for t { for h }`). The parallel `deltanet_scan` (loop order `for h (parallel) { for t }`)
    // must reproduce both `out` and the mutated state EXACTLY.
    #[allow(clippy::too_many_arguments)]
    fn deltanet_scan_ref(
        qf: &[f32],
        kf: &[f32],
        vf: &[f32],
        bf: &[f32],
        af: &[f32],
        acoef: &[f32],
        dtb: &[f32],
        st: &mut [f32],
        rr: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0f32; rr * nv * vd];
        let qscale = 1.0 / (kd as f32).sqrt();
        let l2 = |slice: &[f32]| -> f32 { (slice.iter().map(|x| x * x).sum::<f32>() + eps).sqrt() };
        for t in 0..rr {
            let (qb, vb, bb) = (t * nk * kd, t * nv * vd, t * nv);
            for h in 0..nv {
                let kh_idx = h % nk;
                let mut qh = qf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
                let mut kh = kf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
                let vh = &vf[vb + h * vd..vb + h * vd + vd];
                let qn = l2(&qh);
                let kn = l2(&kh);
                for x in qh.iter_mut() {
                    *x = *x / qn * qscale;
                }
                for x in kh.iter_mut() {
                    *x /= kn;
                }
                let beta = 1.0 / (1.0 + math::exp(-bf[bb + h]));
                let sp = {
                    let z = af[bb + h] + dtb[h];
                    z.max(0.0) + math::ln_1p(math::exp(-z.abs()))
                };
                let decay = math::exp(acoef[h] * sp);
                let sh = &mut st[h * kd * vd..(h + 1) * kd * vd];
                for x in sh.iter_mut() {
                    *x *= decay;
                }
                let mut kv = vec![0f32; vd];
                for kk in 0..kd {
                    let kkv = kh[kk];
                    let row = &sh[kk * vd..kk * vd + vd];
                    for d in 0..vd {
                        kv[d] += kkv * row[d];
                    }
                }
                let delta: Vec<f32> = (0..vd).map(|d| (vh[d] - kv[d]) * beta).collect();
                for kk in 0..kd {
                    let kkv = kh[kk];
                    let row = &mut sh[kk * vd..kk * vd + vd];
                    for d in 0..vd {
                        row[d] += kkv * delta[d];
                    }
                }
                let oh = &mut out[vb + h * vd..vb + h * vd + vd];
                for kk in 0..kd {
                    let qv = qh[kk];
                    let row = &sh[kk * vd..kk * vd + vd];
                    for d in 0..vd {
                        oh[d] += qv * row[d];
                    }
                }
            }
        }
        out
    }

    // The head-parallel `deltanet_scan` must be BIT-IDENTICAL (exact f32 equality) to the naive
    // serial reference for BOTH the output and the mutated state, across a matrix of
    // (rows, n_vhead, n_khead, head_k, head_v) that spans: decode (rr==1) and prefill (rr>1),
    // GQA (nk < nv) and nk==nv, small dims, and a large multi-head case that forces the pool's
    // multi-task path (nv=16, kd=vd=128).
    #[test]
    fn deltanet_scan_bit_identical_to_serial() {
        // Deterministic pseudo-random pattern (no RNG): a cheap hashed ramp in [-1, 1).
        let pat = |seed: usize, i: usize| -> f32 {
            let h = (i
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(40503)))
                & 0xffff;
            (h as f32 / 32768.0) - 1.0
        };
        let be = CpuBackend::new();
        let pool = be.pool();
        let eps = 1e-6f32;
        for &(rr, nv, nk, kd, vd) in &[
            (1usize, 8usize, 2usize, 16usize, 16usize), // decode, GQA
            (4, 8, 2, 16, 16),                          // prefill, GQA
            (7, 16, 4, 32, 32),                         // prefill, GQA
            (3, 4, 4, 16, 16),                          // nk == nv
            (1, 4, 4, 32, 32),                          // decode, nk == nv
            (5, 16, 16, 128, 128),                      // large, multi-task, nk == nv
            (6, 16, 4, 128, 128),                       // large, multi-task, GQA
        ] {
            let qf: Vec<f32> = (0..rr * nk * kd).map(|i| pat(1, i)).collect();
            let kf: Vec<f32> = (0..rr * nk * kd).map(|i| pat(2, i)).collect();
            let vf: Vec<f32> = (0..rr * nv * vd).map(|i| pat(3, i)).collect();
            let bf: Vec<f32> = (0..rr * nv).map(|i| pat(4, i)).collect();
            let af: Vec<f32> = (0..rr * nv).map(|i| pat(5, i)).collect();
            let acoef: Vec<f32> = (0..nv).map(|i| pat(6, i)).collect();
            let dtb: Vec<f32> = (0..nv).map(|i| pat(7, i)).collect();
            let st0: Vec<f32> = (0..nv * kd * vd).map(|i| pat(8, i)).collect();

            let mut st_ref = st0.clone();
            let out_ref = deltanet_scan_ref(
                &qf,
                &kf,
                &vf,
                &bf,
                &af,
                &acoef,
                &dtb,
                &mut st_ref,
                rr,
                nv,
                nk,
                kd,
                vd,
                eps,
            );

            let mut st_par = st0.clone();
            let out_par = deltanet_scan(
                pool,
                &qf,
                &kf,
                &vf,
                &bf,
                &af,
                &acoef,
                &dtb,
                &mut st_par,
                rr,
                nv,
                nk,
                kd,
                vd,
                eps,
            );

            assert_eq!(
                out_par, out_ref,
                "output mismatch at rr={rr} nv={nv} nk={nk} kd={kd} vd={vd}"
            );
            assert_eq!(
                st_par, st_ref,
                "state mismatch at rr={rr} nv={nv} nk={nk} kd={kd} vd={vd}"
            );
        }
    }

    // #1: `sample_token` must not panic when top_k == 0 (the "disable top-k" sentinel) and must
    // return a valid in-range token. Regression for the `select_nth_unstable_by(k - 1)` underflow.
    #[test]
    fn sample_top_k_zero_no_panic() {
        let logits = [0.1f32, 3.0, 0.2, 2.0, 0.05];
        // top_k == 0 => no truncation. Greedy-ish draw (u≈0) should land on the argmax (index 1).
        let tok = sample_token(&logits, 0.0, 0, 1.0, 1.0);
        assert!((tok as usize) < logits.len(), "token {tok} out of range");
        assert_eq!(tok, 1, "u≈0 with full softmax picks the max-logit index");
        // u≈1 walks the CDF to the last (sorted) entry — still a valid index.
        let tok_hi = sample_token(&logits, 0.999_999, 0, 1.0, 1.0);
        assert!((tok_hi as usize) < logits.len());
    }

    /// A NaN router score must not panic the group-limited (DeepSeek) routing: both of its sorts
    /// compared with `partial_cmp(..).unwrap()`, which panics on the first NaN pair. A NaN router
    /// score reaches the expert sort; a group score skips NaN members and turns NaN only as
    /// `+inf + -inf`, which the second case builds from the selection bias.
    #[test]
    fn moe_group_routing_survives_a_nan_router_score() {
        use infr_core::tensor::TensorDesc;
        let (ne, n_expert, n_used, nff) = (32usize, 8usize, 2usize, 16usize);
        let be = CpuBackend::new();
        let upload = |v: &[f32], usage| {
            let b = be.alloc(v.len() * 4, usage).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let fill = |n: usize, s: f32| (0..n).map(|i| (i as f32 * s).sin()).collect::<Vec<f32>>();
        let mut nan_router = fill(n_expert * ne, 0.37);
        // Expert 1's score is NaN.
        nan_router[ne] = f32::NAN;
        // Group 0 is experts 0 and 1: best `+inf`, and the NaN never fills the second slot.
        let mut bias = vec![0f32; n_expert];
        bias[0] = f32::INFINITY;
        bias[1] = f32::NAN;
        for (router, bias) in [(nan_router, None), (fill(n_expert * ne, 0.37), Some(bias))] {
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![1, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::F32));
            let gui = g.weight(TensorDesc::new(vec![n_expert, 2 * nff, ne], DType::F32));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::F32));
            let bi = g.weight(TensorDesc::new(vec![n_expert], DType::F32));
            let yi = g.output(TensorDesc::new(vec![1, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: xi,
                router: ri,
                gate_exps: gui,
                up_exps: gui,
                down_exps: di,
                down_scale: None,
                fused_gate_up: true,
                dst: yi,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: n_used as u32,
                n_ff_exp: nff as u32,
                scale: 1.0,
                act: Activation::Silu,
                gating: MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                ep_band: None,
                exp_probs_b: bias.is_some().then_some(bi),
                n_expert_groups: 4,
                n_expert_groups_used: 2,
                swiglu_clamp: None,
                expert_ids: None,
            });
            let xb = upload(&fill(ne, 0.11), BufferUsage::Activations);
            let rb = upload(&router, BufferUsage::Weights);
            let gub = upload(&fill(n_expert * 2 * nff * ne, 0.017), BufferUsage::Weights);
            let db = upload(&fill(n_expert * ne * nff, 0.029), BufferUsage::Weights);
            let bb = upload(bias.as_deref().unwrap_or(&[0.0; 8]), BufferUsage::Weights);
            let yb = be.alloc(ne * 4, BufferUsage::Activations).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gui, gub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(bi, bb.as_ref());
            bind.bind(yi, yb.as_ref());
            let plan = be.compile(&g).unwrap();
            be.execute(plan.as_ref(), &bind).unwrap();
        }
    }

    /// Equal logits are taken in index order: a uniform draw walking the CDF of a flat row picks
    /// the token whose index it lands on, with or without a top-k cut. Turns red if the order of
    /// ties is left to the unstable sorts.
    #[test]
    fn sample_token_breaks_ties_by_index() {
        let logits = vec![0.5f32; 300];
        for (top_k, n) in [(0usize, 300usize), (40, 40)] {
            for j in 0..n {
                let u = (j as f32 + 0.5) / n as f32;
                assert_eq!(
                    sample_token(&logits, u, top_k, 1.0, 1.0),
                    j as u32,
                    "top_k {top_k}"
                );
            }
        }
    }

    /// A NaN logit must not panic the sampler: its comparator once called NaN "equal" to
    /// everything, which is not a total order, and the selection then panicked.
    #[test]
    fn sample_token_survives_nan_logits() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for n in [600usize, 1590, 3000] {
            for every in [2usize, 3, 5] {
                let logits: Vec<f32> = (0..n)
                    .map(|i| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        if i % every == 0 {
                            f32::NAN
                        } else {
                            (state >> 40) as f32 / (1u64 << 24) as f32 * 10.0
                        }
                    })
                    .collect();
                for k in [0usize, 1, 24, 40, 64] {
                    assert!((sample_token(&logits, 0.5, k, 0.8, 0.95) as usize) < n);
                }
            }
        }
    }

    // #1: top_k == 0 (no truncation) and top_k == len must agree — both consider every logit.
    #[test]
    fn sample_top_k_zero_equals_full_k() {
        let logits = [1.0f32, 5.0, 2.0, 4.0, 3.0, 0.5];
        for &u in &[0.0f32, 0.25, 0.5, 0.75, 0.999] {
            let a = sample_token(&logits, u, 0, 0.8, 0.95);
            let b = sample_token(&logits, u, logits.len(), 0.8, 0.95);
            assert_eq!(a, b, "top_k==0 must match top_k==len at u={u}");
        }
    }

    // #1: a normal top_k >= 1 still truncates and returns a valid token (path unchanged).
    #[test]
    fn sample_top_k_truncates() {
        let logits = [0.0f32, 9.0, 1.0, 8.0, 2.0];
        let tok = sample_token(&logits, 0.0, 2, 1.0, 1.0);
        // Only indices {1,3} survive top_k=2; u≈0 picks the larger (index 1).
        assert_eq!(tok, 1);
    }

    // #2: `cached_weight` keys on (buffer uid, dtype). Same key returns the CACHED value (even if
    // the bytes changed — proves it dedups); a key differing in uid or dtype MISSES and dequants
    // fresh.
    #[test]
    fn cached_weight_keys_on_uid_and_dtype() {
        let be = CpuBackend::new();
        let a = [1.0f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        let key = (7u64, DType::F32);
        let w1 = be.cached_weight(key, &a);
        assert_eq!(&**w1, &[1.0, 2.0]);

        // Same key, DIFFERENT bytes => returns the cached f32 (not re-dequantized).
        let b = [9.0f32, 9.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        let w2 = be.cached_weight(key, &b);
        assert_eq!(&**w2, &[1.0, 2.0], "same key must hit the cache");

        // Same uid but a DIFFERENT dtype => distinct key => fresh dequant (no collision).
        let c: Vec<u8> = [3i32, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
        let w3 = be.cached_weight((7, DType::I32), &c);
        assert_eq!(
            &**w3,
            &[3.0, 4.0],
            "different dtype must miss and dequant fresh"
        );

        // A different buffer (uid) with the same dtype is a distinct weight => fresh dequant.
        let d = [7.0f32, 8.0, 9.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        let w4 = be.cached_weight((8, DType::F32), &d);
        assert_eq!(
            &**w4,
            &[7.0, 8.0, 9.0],
            "different uid must miss and dequant fresh"
        );
    }

    /// The property every derived-data cache in this backend now keys on: a buffer's `uid` is never
    /// handed out twice, INCLUDING to a buffer that reuses a freed one's address. Freeing an
    /// allocation and immediately re-allocating the same size is the shape a serve model reload
    /// takes, and an allocator frequently returns the very same address for it — which is exactly
    /// how an address-keyed `weight_cache`/`repack_cache` entry could be returned for a different
    /// weight. The address may repeat here (that is not asserted, it is allocator-dependent); the
    /// uid may not.
    #[test]
    fn buffer_uids_are_never_reused() {
        let be = CpuBackend::new();
        let first = be.alloc(4096, BufferUsage::Weights).expect("alloc");
        let first_uid = cpu_buf(first.as_ref()).uid;
        drop(first);
        let second = be.alloc(4096, BufferUsage::Weights).expect("alloc");
        assert_ne!(
            cpu_buf(second.as_ref()).uid,
            first_uid,
            "a uid was reused after the first buffer was freed"
        );
        // Two live buffers never collide either.
        let third = be.alloc(4096, BufferUsage::Weights).expect("alloc");
        assert_ne!(cpu_buf(third.as_ref()).uid, cpu_buf(second.as_ref()).uid);
    }

    /// `kernels.cpu.repack_mb` is the Q4_K/Q6_K repack caches' byte budget, in MiB, and BOTH
    /// caches take it off the same field of the backend's own `Config` — no environment, so two
    /// backends in one process can carry different budgets. The shipped default is still 4096 MiB;
    /// what the number then DOES (insert vs. build-transient) is `repack::cache_insert_if_absent`'s
    /// own test.
    #[test]
    fn repack_budget_comes_off_the_config() {
        assert_eq!(Config::default().kernels.cpu.repack_mb, 4096);
        let dflt = CpuBackend::new_with(Arc::new(Config::default()));
        assert_eq!(dflt.repack_budget_bytes(), 4096 * 1024 * 1024);

        let mut cfg = Config::default();
        cfg.kernels.cpu.repack_mb = 3;
        let small = CpuBackend::new_with(Arc::new(cfg));
        assert_eq!(small.repack_budget_bytes(), 3 * 1024 * 1024);
        // Independent values, live at the same time — the thing a process-wide env read could not do.
        assert_eq!(dflt.repack_budget_bytes(), 4096 * 1024 * 1024);
    }

    /// One RoPE frequency table per `(theta, rope_dim)` for the backend's life, holding exactly
    /// [`math::rope_freqs`]' values. Turns red if `rope_freqs` builds a fresh table per call (the
    /// per-op `powf` cost a one-row decode never amortizes).
    #[test]
    fn rope_freqs_are_built_once_per_shape() {
        let be = CpuBackend::new_with(Arc::new(Config::default()));
        let a = be.rope_freqs(10_000.0, 128);
        assert_eq!(&a[..], &math::rope_freqs(10_000.0, 128)[..]);
        assert!(Arc::ptr_eq(&a, &be.rope_freqs(10_000.0, 128)));
        let other_theta = be.rope_freqs(1_000_000.0, 128);
        assert_eq!(&other_theta[..], &math::rope_freqs(1_000_000.0, 128)[..]);
        assert!(!Arc::ptr_eq(&a, &other_theta));
        assert_eq!(be.rope_freqs(10_000.0, 64).len(), 32);
    }

    /// The packed prefill Linear writes the row-major output directly, with an `out_f % 8` tail
    /// on the per-row kernels: every element must be the single-token kernel's bits, for every
    /// dtype it takes, a token count that is not a whole number of GEMM tiles, and a scratch
    /// group reused across calls of different widths (it must not leak the previous call's rows).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn linear_packed_matches_the_single_token_kernels_bitwise() {
        let (out_f, m) = (20usize, 7usize);
        assert!(out_f % 8 != 0 && m > repack::TILE_TOKENS && m % repack::TILE_TOKENS != 0);
        let be = CpuBackend::new();
        for (dt, in_f) in [
            (DType::Q4K, 512usize),
            (DType::Q4K, 256),
            (DType::Q4K, 768),
            (DType::Q5K, 512),
            (DType::Q5K, 256),
            (DType::Q6K, 512),
        ] {
            if !linear_packed_ok::<false>(dt, in_f, out_f, m) {
                // Only AVX-512 VNNI hosts take this route; others run the per-row kernels,
                // which `linear_prefill_matches_the_single_token_kernels_bitwise` covers.
                eprintln!("SKIP linear_packed: this CPU lacks AVX-512BW + AVX-512 VNNI");
                return;
            }
            let nb = in_f / 256;
            let w = match dt {
                DType::Q4K => det_quant_rows(out_f, nb, 144, &[0, 2], in_f as u64),
                DType::Q5K => det_quant_rows(out_f, nb, 176, &[0, 2], in_f as u64),
                _ => det_quant_rows(out_f, nb, 210, &[208], in_f as u64),
            };
            let x: Vec<f32> = (0..m * in_f)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
                .collect();
            let q8s: Vec<Q8> = x.chunks(in_f).map(quantize_q8).collect();
            let mut out = vec![f32::NAN; m * out_f];
            be.linear_packed::<false>(dt, &w, in_f, out_f, &q8s, &mut out);
            let bpr = w.len() / out_f;
            for o in 0..out_f {
                let row = &w[o * bpr..(o + 1) * bpr];
                for (r, q8) in q8s.iter().enumerate() {
                    let want = match dt {
                        DType::Q4K => vec_dot_q4k::<false>(row, q8, in_f),
                        DType::Q5K => vec_dot_q5k::<false>(row, q8, in_f),
                        _ => vec_dot_q6k::<false>(row, q8, in_f),
                    };
                    let got = out[r * out_f + o];
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{dt:?} in_f={in_f} o={o} r={r}: got {got}, want {want}"
                    );
                }
            }
        }
    }

    /// A causal prefill `Op::Attention` of `rows` query rows at position `pos` over an f16 K/V
    /// cache, through `execute` on `be` inside a rayon pool of `threads` (which sizes the spin
    /// pool), returning its `[rows, n_head * head_dim]` output.
    fn run_attention(be: CpuBackend, threads: usize, rows: usize, pos: usize) -> Vec<f32> {
        let (nh, nkv, hd) = (4usize, 2usize, 64usize);
        let kv_len = pos + rows;
        let q: Vec<f32> = (0..rows * nh * hd)
            .map(|i| ((i * 31 % 97) as f32 - 48.0) / 40.0)
            .collect();
        let kv = |seed: usize| -> Vec<u8> {
            (0..kv_len * nkv * hd)
                .flat_map(|i| {
                    let v = ((i * 17 + seed) % 89) as f32 / 30.0 - 1.4;
                    half::f16::from_f32(v).to_le_bytes()
                })
                .collect()
        };
        let mut g = Graph::new();
        let qi = g.input(TensorDesc::new(vec![rows, nh * hd], DType::F32));
        let ki = g.input(TensorDesc::new(vec![kv_len * nkv * hd], DType::F16));
        let vi = g.input(TensorDesc::new(vec![kv_len * nkv * hd], DType::F16));
        let dst = g.output(TensorDesc::new(vec![rows, nh * hd], DType::F32));
        g.push(Op::Attention {
            q: qi,
            k_cache: ki,
            v_cache: vi,
            dst,
            rows: rows as u32,
            kv_len: kv_len as u32,
            n_head: nh as u32,
            n_kv: nkv as u32,
            head_dim: hd as u32,
            scale: 1.0 / (hd as f32).sqrt(),
            mask: AttnMask::Causal,
            pos: pos as u32,
            sinks: None,
            key_bias: None,
            comp: None,
        });
        let up = |bytes: &[u8]| {
            let b = be.alloc(bytes.len(), BufferUsage::Activations).unwrap();
            be.upload(b.as_ref(), bytes).unwrap();
            b
        };
        let (qb, kb, vb) = (up(&f32_bytes(&q)), up(&kv(3)), up(&kv(11)));
        let ob = be
            .alloc(rows * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        let mut b = Bindings::new();
        b.bind(qi, qb.as_ref());
        b.bind(ki, kb.as_ref());
        b.bind(vi, vb.as_ref());
        b.bind(dst, ob.as_ref());
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| be.execute(be.compile(&g).unwrap().as_ref(), &b).unwrap());
        let mut bytes = vec![0u8; rows * nh * hd * 4];
        be.download(ob.as_ref(), &mut bytes).unwrap();
        bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Blocked prefill attention gives the bits of one (row, head) pair per task, the
    /// decomposition it replaced, on one thread and on several: a whole number of blocks plus a
    /// partial last one, from position 0 and from a later position.
    #[test]
    fn attention_blocks_match_one_pair_per_task_bitwise() {
        let rows = 2 * ATTN_QUERY_BLOCK + 5;
        for pos in [0usize, 9] {
            let mut per_pair = CpuBackend::new();
            per_pair.attn_query_block = 1;
            let want = run_attention(per_pair, 1, rows, pos);
            assert!(want.iter().all(|v| v.is_finite()), "non-finite output");
            let nonzero = want.iter().filter(|v| **v != 0.0).count();
            assert!(
                nonzero > want.len() / 2,
                "vacuous output: {nonzero} non-zero"
            );
            for threads in [1usize, 8] {
                let got = run_attention(CpuBackend::new(), threads, rows, pos);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "pos={pos} threads={threads} element {i}: {g} vs {w}"
                    );
                }
            }
        }
    }

    /// Run `body` on a fresh thread inside a rayon pool of four threads (which sizes the spin
    /// pool), with a backend whose pool is already built; skip, saying why, when this machine
    /// gives a four-thread pool no placement.
    #[cfg(windows)]
    fn with_placed_pool(name: &str, body: impl FnOnce(&CpuBackend) + Send) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                let be = CpuBackend::new();
                if be.pool().placement.is_none() {
                    eprintln!(
                        "SKIP {name}: no placement for four threads here (fewer than four \
                         allowed cores, nothing to narrow, or default CPU sets)"
                    );
                    return;
                }
                body(&be);
            });
    }

    /// A one-Linear graph and its bindings over `be`'s buffers; `bind_weight = false` leaves the
    /// weight unbound so the walk panics after placing itself.
    #[cfg(windows)]
    fn small_linear(be: &CpuBackend, bind_weight: bool) -> std::thread::Result<()> {
        let (m, in_f, out_f) = (2usize, 256usize, 8usize);
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![out_f, in_f], DType::F32));
        let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
        g.push(Op::Linear {
            x: xi,
            weight: wi,
            dst,
            m: m as u32,
            in_f: in_f as u32,
            out_f: out_f as u32,
            w_off: 0,
        });
        let up = |bytes: &[u8]| {
            let b = be.alloc(bytes.len(), BufferUsage::Activations).unwrap();
            be.upload(b.as_ref(), bytes).unwrap();
            b
        };
        let xb = up(&f32_bytes(&vec![0.5; m * in_f]));
        let wb = up(&f32_bytes(&vec![0.25; out_f * in_f]));
        let ob = be.alloc(m * out_f * 4, BufferUsage::Activations).unwrap();
        let mut b = Bindings::new();
        b.bind(xi, xb.as_ref());
        if bind_weight {
            b.bind(wi, wb.as_ref());
        }
        b.bind(dst, ob.as_ref());
        let plan = be.compile(&g).unwrap();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            be.execute(plan.as_ref(), &b).unwrap();
        }))
    }

    /// Every worker runs placed: a job with a task per worker reads the selection back on each
    /// thread other than the dispatching one.
    #[cfg(windows)]
    #[test]
    fn pool_workers_run_placed() {
        with_placed_pool("pool_workers_run_placed", |be| {
            let pool = be.pool();
            let me = std::thread::current().id();
            let seen = std::sync::Mutex::new(Vec::new());
            // Enough tasks, each slow enough, that every worker claims some.
            pool.run(64, &|_| {
                std::thread::sleep(std::time::Duration::from_millis(2));
                if std::thread::current().id() != me {
                    seen.lock()
                        .unwrap()
                        .push(infr_plat::cpu::current_placement());
                }
            });
            let seen = seen.into_inner().unwrap();
            assert!(!seen.is_empty(), "no task ran on a worker");
            for p in &seen {
                assert_eq!(
                    p, &pool.placement,
                    "a worker ran without the pool's placement"
                );
            }
        });
    }

    /// The walk runs placed and leaves the dispatching thread unplaced afterwards, whether it
    /// returns or panics (the "unbound Weight" panic message in the test output is that case).
    #[cfg(windows)]
    #[test]
    fn execute_places_the_caller_only_for_the_walk() {
        with_placed_pool("execute_places_the_caller_only_for_the_walk", |be| {
            let want = be.pool().placement.clone();
            EXEC_PLACEMENT.set(None);
            small_linear(be, true).expect("the walk");
            assert_eq!(EXEC_PLACEMENT.take(), want, "the walk ran unplaced");
            assert_eq!(
                infr_plat::cpu::current_placement(),
                None,
                "left placed after return"
            );
            assert!(small_linear(be, false).is_err(), "an unbound weight panics");
            assert_eq!(
                EXEC_PLACEMENT.take(),
                want,
                "the panicking walk ran unplaced"
            );
            assert_eq!(
                infr_plat::cpu::current_placement(),
                None,
                "left placed after a panic"
            );
        });
    }

    /// A walk inside an outer placement leaves the outer one in force until the outer guard
    /// drops, and a thread with a selection of its own keeps it through a walk.
    #[cfg(windows)]
    #[test]
    fn execute_leaves_an_existing_placement_alone() {
        with_placed_pool("execute_leaves_an_existing_placement_alone", |be| {
            let want = be.pool().placement.clone();
            let outer = be.pool().place_caller().expect("outer placement");
            small_linear(be, true).expect("the inner walk");
            assert_eq!(
                infr_plat::cpu::current_placement(),
                want,
                "the inner walk cleared it"
            );
            drop(outer);
            assert_eq!(infr_plat::cpu::current_placement(), None);

            let own = infr_plat::cpu::spread_placement(1).expect("a one-thread placement");
            assert!(infr_plat::cpu::apply_placement(&own).expect("apply our own"));
            small_linear(be, true).expect("the walk");
            assert_eq!(
                infr_plat::cpu::current_placement().as_ref(),
                Some(&own),
                "the walk replaced or cleared the thread's own selection"
            );
            infr_plat::cpu::clear_placement().expect("clear our own");
        });
    }

    /// A prefill `Op::Linear` of `m` tokens through `execute`, returning its `[m, out_f]` output.
    fn run_linear(dt: DType, w: &[u8], x: &[f32], m: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![out_f, in_f], dt));
        let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
        g.push(Op::Linear {
            x: xi,
            weight: wi,
            dst,
            m: m as u32,
            in_f: in_f as u32,
            out_f: out_f as u32,
            w_off: 0,
        });
        let be = CpuBackend::new();
        let up = |bytes: &[u8]| {
            let b = be.alloc(bytes.len(), BufferUsage::Activations).unwrap();
            be.upload(b.as_ref(), bytes).unwrap();
            b
        };
        let (xb, wb) = (up(&f32_bytes(x)), up(w));
        let ob = be.alloc(m * out_f * 4, BufferUsage::Activations).unwrap();
        let mut b = Bindings::new();
        b.bind(xi, xb.as_ref());
        b.bind(wi, wb.as_ref());
        b.bind(dst, ob.as_ref());
        be.execute(be.compile(&g).unwrap().as_ref(), &b).unwrap();
        let mut bytes = vec![0u8; m * out_f * 4];
        be.download(ob.as_ref(), &mut bytes).unwrap();
        bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Every `Op::Linear` route that adds a sparse activation part (one row; the batch kernels,
    /// tiles and packed GEMM with their row-major and o-major adds) gives, on every element, the
    /// one-row dense kernel plus the sparse part dotted the way it was before the fast sparse dot
    /// (`vec_dot_hi_reference`), for every format that takes `Q8` activations, at 1, 2, 5 and 19
    /// rows, on rows of every spike pattern (`test_util::spike_patterns`) and some with none.
    #[test]
    fn linear_routes_add_the_reference_sparse_dot_bitwise() {
        use crate::kernels::{vec_dot_hi_reference, vec_dot_q8_any};
        use crate::test_util::{spiky_row, synth_weights, Q8_DTYPES};
        let (in_f, out_f) = (768usize, 19usize);
        let row = |r: usize| -> Vec<f32> {
            if r % 4 == 3 {
                (0..in_f)
                    .map(|i| ((i * 7 + r) % 23) as f32 / 9.0 - 1.2)
                    .collect()
            } else {
                spiky_row(r, in_f)
            }
        };
        let mut split = 0usize;
        for dt in Q8_DTYPES {
            let w = synth_weights(
                dt,
                out_f,
                in_f,
                0x7a,
                &[(1.0, 1.0), (-0.5, 2.0), (3.0, 0.0)],
            );
            let bpr = w.len() / out_f;
            for m in [1usize, 2, 5, 19] {
                let xs: Vec<Vec<f32>> = (0..m).map(row).collect();
                let got = run_linear(dt, &w, &xs.concat(), m, in_f, out_f);
                for (r, x) in xs.iter().enumerate() {
                    let q8 = quantize_q8(x);
                    for o in 0..out_f {
                        let wr = &w[o * bpr..(o + 1) * bpr];
                        let dense = vec_dot_q8_any::<false>(dt, wr, &q8, in_f);
                        let want = if q8.hi.is_empty() {
                            dense
                        } else {
                            split += 1;
                            dense + vec_dot_hi_reference::<false>(dt, wr, &q8, in_f)
                        };
                        let g = got[r * out_f + o];
                        assert_eq!(
                            g.to_bits(),
                            want.to_bits(),
                            "{dt:?} m={m} row {r} o={o}: got {g}, want {want}"
                        );
                    }
                }
            }
        }
        assert!(split > 0, "no row split");
    }

    /// Prefill `Op::Linear` on K-quant weights gives the single-token kernel's bits on every
    /// element, whichever route this host takes: the packed GEMM (AVX-512 VNNI) or the per-row
    /// batch kernels (any other CPU), and on both sides of each dtype's packed threshold
    /// (`repack::packed_min_tokens`). On a VNNI host it also checks that each count takes the
    /// route its threshold says. The thresholds' values are measurements, recorded where they
    /// are defined; this test follows whatever they are set to.
    #[test]
    fn linear_prefill_matches_the_single_token_kernels_bitwise() {
        let (in_f, out_f) = (512usize, 20usize);
        let nb = in_f / 256;
        for (dt, blk, d_offs) in [
            (DType::Q4K, 144usize, &[0usize, 2][..]),
            (DType::Q5K, 176, &[0, 2]),
            (DType::Q6K, 210, &[208]),
        ] {
            let min = repack::packed_min_tokens(dt);
            let w = det_quant_rows(out_f, nb, blk, d_offs, blk as u64);
            // Below the threshold (when there is a below: prefill starts at two tokens), at it,
            // and past a whole number of GEMM tiles.
            let mut counts = vec![min.max(2), 2 * repack::TILE_TOKENS + 1];
            if min > 2 {
                counts.push(min - 1);
            }
            for m in counts {
                let packed = linear_packed_ok::<false>(dt, in_f, out_f, m);
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
                    assert_eq!(packed, m >= min, "{dt:?} m={m}: wrong route");
                } else {
                    eprintln!("{dt:?} m={m}: no AVX-512 VNNI, per-row kernels only");
                }
                let x: Vec<f32> = (0..m * in_f)
                    .map(|i| ((i * 29 % 97) as f32 - 48.0) / 13.0)
                    .collect();
                let got = run_linear(dt, &w, &x, m, in_f, out_f);
                let bpr = w.len() / out_f;
                for (r, xr) in x.chunks(in_f).enumerate() {
                    let q8 = quantize_q8(xr);
                    for o in 0..out_f {
                        let row = &w[o * bpr..(o + 1) * bpr];
                        let want = match dt {
                            DType::Q4K => vec_dot_q4k::<false>(row, &q8, in_f),
                            DType::Q5K => vec_dot_q5k::<false>(row, &q8, in_f),
                            _ => vec_dot_q6k::<false>(row, &q8, in_f),
                        };
                        let g = got[r * out_f + o];
                        assert_eq!(
                            g.to_bits(),
                            want.to_bits(),
                            "{dt:?} m={m} packed={packed} o={o} r={r}: got {g}, want {want}"
                        );
                    }
                }
            }
        }
    }

    /// `kernels.cpu.reference` is a plain field on the config the backend was handed, and
    /// [`CpuBackend::reference`] is the shorthand that sets it. (The numeric consequence — an f32
    /// decode dot instead of the int8-activation kernels — is
    /// `decode_parity::cpu_reference_gemv_matches_the_shared_decoder_for_every_weight_quant`.)
    #[test]
    fn reference_mode_is_a_config_field() {
        assert!(!Config::default().kernels.cpu.reference);
        assert!(
            !CpuBackend::new_with(Arc::new(Config::default()))
                .cfg
                .kernels
                .cpu
                .reference
        );
        assert!(CpuBackend::reference().cfg.kernels.cpu.reference);
    }

    /// Bind `bound` (raw bytes per tensor) and a buffer for `out`, and run `g` once.
    fn run_graph(g: &Graph, bound: &[(TensorId, Vec<u8>)], out: TensorId) {
        let be = CpuBackend::new();
        let mut bufs = Vec::new();
        for (id, bytes) in bound {
            let buf = be.alloc(bytes.len(), BufferUsage::Activations).unwrap();
            be.upload(buf.as_ref(), bytes).unwrap();
            bufs.push((*id, buf));
        }
        let ob = be
            .alloc(g.desc(out).numel() * 4, BufferUsage::Activations)
            .unwrap();
        let mut b = Bindings::new();
        for (id, buf) in &bufs {
            b.bind(*id, buf.as_ref());
        }
        b.bind(out, ob.as_ref());
        be.execute(be.compile(g).unwrap().as_ref(), &b).unwrap();
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// A graph with one strided `QkNorm` or `Rope`; `rope` picks which.
    fn strided_head_op(rope: bool) -> (Graph, Vec<(TensorId, Vec<u8>)>, TensorId) {
        let (nh, hd) = (2usize, 4usize);
        let stride = nh * 2 * hd;
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![stride], DType::F32));
        let dst = g.output(TensorDesc::new(vec![nh * hd], DType::F32));
        let mut bound = vec![(x, f32_bytes(&vec![1.0; stride]))];
        if rope {
            let positions = g.input(TensorDesc::new(vec![1], DType::F32));
            bound.push((positions, f32_bytes(&[3.0])));
            g.push(Op::Rope {
                x,
                positions,
                dst,
                rows: 1,
                n_head: nh as u32,
                head_dim: hd as u32,
                rope_dim: hd as u32,
                theta: 1e4,
                freq_factors: None,
                x_stride: stride as u32,
                neox: false,
                backward: false,
            });
        } else {
            g.push(Op::QkNorm {
                x,
                weight: None,
                dst,
                rows: 1,
                n_head: nh as u32,
                head_dim: hd as u32,
                eps: 1e-6,
                x_stride: stride as u32,
            });
        }
        (g, bound, dst)
    }

    /// B128: `QkNorm` and `Rope` read packed rows only; a strided `x` used to be read as packed
    /// without a word. Metal asserts the same.
    #[test]
    #[should_panic(expected = "strided x")]
    fn strided_qk_norm_is_refused() {
        let (g, bound, dst) = strided_head_op(false);
        run_graph(&g, &bound, dst);
    }

    #[test]
    #[should_panic(expected = "strided x")]
    fn strided_rope_is_refused() {
        let (g, bound, dst) = strided_head_op(true);
        run_graph(&g, &bound, dst);
    }

    /// Run RmsNorm → Linear → WriteKv (into a bound f32 cache) twice on a new backend under the
    /// per-op hash tap writing to `path`, with the Linear weight's first element set to `w0`.
    fn op_hash_run(path: &std::path::Path, w0: f32) {
        let (dim, out_f, ctx) = (8usize, 4usize, 3usize);
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![dim], DType::F32));
        let wn = g.weight(TensorDesc::new(vec![dim], DType::F32));
        let wl = g.weight(TensorDesc::new(vec![out_f, dim], DType::F32));
        let cache = g.input(TensorDesc::new(vec![ctx, out_f], DType::F32));
        let h = g.internal(TensorDesc::new(vec![dim], DType::F32));
        let y = g.output(TensorDesc::new(vec![out_f], DType::F32));
        g.push(Op::RmsNorm {
            x,
            weight: wn,
            dst: h,
            rows: 1,
            dim: dim as u32,
            eps: 1e-6,
        });
        g.push(Op::Linear {
            x: h,
            weight: wl,
            dst: y,
            m: 1,
            in_f: dim as u32,
            out_f: out_f as u32,
            w_off: 0,
        });
        g.push(Op::WriteKv {
            src: y,
            cache,
            rows: 1,
            row_stride: out_f as u32,
            pos: 1,
        });
        let mut cfg = Config::default();
        cfg.debug.cpu_op_hash = Some(path.to_path_buf());
        let be = CpuBackend::new_with(Arc::new(cfg));
        let up = |bytes: &[u8]| {
            let b = be.alloc(bytes.len(), BufferUsage::Activations).unwrap();
            be.upload(b.as_ref(), bytes).unwrap();
            b
        };
        let xs: Vec<f32> = (0..dim).map(|i| i as f32 * 0.25 - 1.0).collect();
        let mut wls: Vec<f32> = (0..out_f * dim)
            .map(|i| (i % 7) as f32 * 0.5 - 1.5)
            .collect();
        wls[0] = w0;
        let (xb, wnb, wlb) = (
            up(&f32_bytes(&xs)),
            up(&f32_bytes(&vec![1.5; dim])),
            up(&f32_bytes(&wls)),
        );
        let cb = up(&vec![0u8; ctx * out_f * 4]);
        let yb = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        let mut b = Bindings::new();
        for (id, buf) in [(x, &xb), (wn, &wnb), (wl, &wlb), (cache, &cb), (y, &yb)] {
            b.bind(id, buf.as_ref());
        }
        let plan = be.compile(&g).unwrap();
        // Two executes: the tap numbers the graphs, and the second one reads the cache row the
        // first one wrote.
        be.execute(plan.as_ref(), &b).unwrap();
        be.execute(plan.as_ref(), &b).unwrap();
    }

    /// [`op_hash_run`]'s lines with the backend field (`b<n>`, which differs between backends)
    /// dropped after checking it is the same on every line; the file is removed.
    fn op_hash_lines(path: &std::path::Path, w0: f32) -> Vec<String> {
        op_hash_run(path, w0);
        let text = std::fs::read_to_string(path).unwrap();
        std::fs::remove_file(path).unwrap();
        let backends: std::collections::BTreeSet<&str> =
            text.lines().map(|l| l.split(' ').next().unwrap()).collect();
        assert_eq!(backends.len(), 1, "{backends:?}");
        text.lines()
            .map(|l| l.split_once(' ').unwrap().1.to_owned())
            .collect()
    }

    /// Two backends tapping into one file get their own `b<n>` ordinals, and each execute's
    /// lines arrive whole and together.
    #[test]
    fn op_hash_tap_tells_backends_apart() {
        let path =
            std::env::temp_dir().join(format!("infr-op-hash-{}-two.txt", std::process::id()));
        if path.exists() {
            std::fs::remove_file(&path).unwrap();
        }
        std::thread::scope(|s| {
            s.spawn(|| op_hash_run(&path, -1.5));
            s.spawn(|| op_hash_run(&path, -1.25));
        });
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        // Two backends, two graphs each, seven lines per graph.
        assert_eq!(lines.len(), 28, "{text}");
        let mut blocks: Vec<(&str, &str)> = Vec::new();
        for l in &lines {
            let f: Vec<&str> = l.split(' ').collect();
            assert!(
                f.len() >= 8 && f[0].starts_with('b') && f[1].starts_with('g'),
                "{l}"
            );
            let key = (f[0], f[1]);
            if blocks.last() != Some(&key) {
                assert!(!blocks.contains(&key), "execute {key:?} was split:\n{text}");
                blocks.push(key);
            }
        }
        let backends: std::collections::BTreeSet<&str> = blocks.iter().map(|k| k.0).collect();
        assert_eq!((backends.len(), blocks.len()), (2, 4), "{blocks:?}");
    }

    /// The per-op hash tap (`debug.cpu_op_hash`) is reproducible run to run, records every op's
    /// non-weight reads and its writes (an in-place cache as both), and a perturbed weight moves
    /// the hashes from the first op that reads it on while leaving the ops before it alone.
    #[test]
    fn op_hash_tap_pins_the_first_differing_op() {
        let dir = std::env::temp_dir();
        let path = |tag: &str| dir.join(format!("infr-op-hash-{}-{tag}.txt", std::process::id()));
        let a = op_hash_lines(&path("a"), -1.5);
        let b = op_hash_lines(&path("b"), -1.5);
        assert_eq!(a, b, "two runs of one graph must hash identically");
        // Per graph: RmsNorm reads x and writes h; Linear reads h and writes y; WriteKv reads y
        // and the cache and writes the cache.
        let shape: Vec<(String, String)> = a
            .iter()
            .map(|l| {
                let f: Vec<&str> = l.split(' ').collect();
                (format!("{} {} {}", f[0], f[1], f[2]), f[6].to_owned())
            })
            .collect();
        let want = |g: u32| {
            [
                ("op0 R", "RmsNorm"),
                ("op0 W", "RmsNorm"),
                ("op1 R", "Linear"),
                ("op1 W", "Linear"),
                ("op2 R", "WriteKv"),
                ("op2 R", "WriteKv"),
                ("op2 W", "WriteKv"),
            ]
            .map(|(o, k)| (format!("g{g} {o}"), k.to_owned()))
        };
        let expect: Vec<_> = want(0).into_iter().chain(want(1)).collect();
        assert_eq!(shape, expect, "{a:#?}");
        // The first graph reads the zeroed cache, the second the row the first wrote.
        assert_ne!(a[5].split(' ').nth(5), a[12].split(' ').nth(5), "{a:#?}");

        let c = op_hash_lines(&path("c"), -1.25);
        let first_diff = a.iter().zip(&c).position(|(x, y)| x != y);
        // RmsNorm and Linear's read agree; Linear's write is the first line to differ.
        assert_eq!(first_diff, Some(3), "{a:#?}\n{c:#?}");
        assert!(c[3].contains(" op1 W "), "{c:#?}");
        // Everything downstream of it moves: WriteKv's read of `y` and the cache it wrote.
        for i in [4, 6] {
            assert_ne!(a[i], c[i], "line {i}");
        }
    }
}
