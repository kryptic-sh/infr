//! The compute backend seam — the ONLY device-aware trait. Everything above is generic over it.
//!
//! Object-safe on purpose so the engine can hold `Arc<dyn Backend>` and stay blind to whether
//! Vulkan / CPU / CUDA / ROCm / Metal / MLX is underneath. See PLAN.md "backend abstraction".
//!
//! ## Execution model
//!
//! 1. The model builds a [`Graph`] (op-list) once per forward *shape* and `compile`s it to a
//!    [`Plan`] (pipelines, buffer-size planning, recorded command buffer for Vulkan).
//! 2. The model owns long-lived buffers (weights, KV cache) and per-step input/output buffers,
//!    and binds them to the graph's `Input`/`Weight`/`Output` handles via [`Bindings`].
//! 3. `execute(plan, bindings)` allocates the graph's `Internal` scratch, runs the ops, and
//!    writes results into the bound `Output` buffers. `Internal`/`Output` scratch is transient.

use crate::error::Result;
use crate::graph::Graph;
use crate::tensor::TensorId;
use std::collections::HashMap;

/// Device capabilities the graph compiler queries to pick fast vs fallback kernels.
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub name: String,
    pub f16: bool,
    pub cooperative_matrix: bool,
    pub max_buffer_bytes: u64,
    /// `maxComputeSharedMemorySize` — the per-workgroup shared-memory budget. Vulkan only guarantees
    /// 16 KB; RADV gives 64 KB, NVIDIA 48 KB, MoltenVK/mobile often 32 KB. The flash-attention tile
    /// height is picked to fit this (and flash is skipped entirely if even the smallest tile won't).
    pub max_shared_memory_bytes: u32,
    pub unified_memory: bool,
    /// The backend records a single-token decode graph ONCE and replays it per token, reading the
    /// position from a device-side params buffer (Vulkan seam) instead of the graph's baked `pos`.
    /// When set, the runner may compile the decode graph once (pos=0) and reuse it across the whole
    /// decode loop. Backends that read the baked `pos`/`kv_len` (CPU interpreter) leave this false.
    pub decode_replay: bool,
    /// The backend prefers the dense FFN's gate+up weights CONCATENATED into one `[2*nff, ne]`
    /// tensor (one GEMV/GEMM + `Op::GatedActFused` instead of two Linears + `Op::GatedAct`).
    /// Costs an owned copy of both weights at load, so zero-copy mmap backends (CPU) leave this
    /// false and keep the separate-tensor form.
    pub combined_gu: bool,
    /// The backend executes [`crate::Op::EmbedGather`] (dequantize embedding-table rows selected
    /// by a device-side id buffer). When set, the runner uploads TOKEN IDS (4 bytes each) instead
    /// of host-dequantized f32 embedding rows — decode feeds 4B/token, prefill 4B/token instead
    /// of `4*n_embd`. Backends without a table-row dequant kernel leave this false and keep the
    /// host embed path.
    pub embed_gather: bool,
    /// The backend executes [`crate::Op::Sample`] (device-side temperature + top-k + top-p
    /// sampling; only the 4-byte token id reads back). False = the runner downloads the logits
    /// and samples on the host.
    pub gpu_sample: bool,
    /// The backend executes [`crate::Op::Argmax`] with `rows > 1` (per-row greedy argmax over
    /// `[rows, n]` logits — the MTP speculative-verify accept reads back m 4-byte ids instead of
    /// the m×vocab logits, issue #31). Every backend handles `rows == 1` (the decode loop);
    /// backends whose kernel is single-row-only (Metal) leave this false and the runner keeps
    /// the host-logits accept path there.
    pub argmax_rows: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferUsage {
    Weights,
    Activations,
    /// Host→device staging (host-visible, mapped): `upload` is a direct memcpy.
    Staging,
    /// Device→host readback (host-visible, mapped): `download` is a direct memcpy.
    Readback,
    /// Weights pinned in HOST memory but device-readable (GTT) — the MoE auto-fit's offloaded
    /// expert banks. On ReBAR systems `Staging` (CpuToGpu) lands in device-local host-visible
    /// VRAM, which defeats offloading; this class guarantees system RAM. The GPU reads it over
    /// the bus. Backends without the distinction (CPU, unified-memory Metal) treat it as Weights.
    HostWeights,
}

/// Opaque device-memory handle owned by a backend.
/// RAII scope for a weight-load progress display (see [`Backend::weight_progress`]). Purely a
/// lifetime marker: constructing it starts the display, dropping it finishes/clears it.
pub trait ProgressScope: Send {}

pub trait Buffer: Send + Sync {
    fn len_bytes(&self) -> usize;
    /// Downcast hook so a backend can recover its concrete buffer type from a `&dyn Buffer`
    /// bound by the model (every buffer a backend sees was allocated by itself).
    fn as_any(&self) -> &dyn std::any::Any;
}

/// A compiled, ready-to-run graph (pipelines + command buffers for Vulkan, an op schedule for CPU).
pub trait Plan: Send + Sync {
    /// Downcast hook so a backend can recover its concrete plan type from a `&dyn Plan`.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// The trivial [`Plan`] both current backends share: it just carries a clone of the [`Graph`] — the
/// CPU interpreter and the Vulkan adapter each re-walk the ops every `execute`, so "compiling" is
/// only storing the graph. A backend's `compile` returns [`GraphPlan::boxed`]; its `execute` recovers
/// the graph via `plan.as_any().downcast_ref::<GraphPlan>()`. (Was duplicated as `CpuPlan`/`VkPlan`.)
pub struct GraphPlan {
    pub graph: crate::graph::Graph,
}

impl GraphPlan {
    pub fn boxed(graph: &crate::graph::Graph) -> Box<dyn Plan> {
        Box::new(GraphPlan {
            graph: graph.clone(),
        })
    }
}

impl Plan for GraphPlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Binds a [`Graph`]'s `Input`/`Weight`/`Output` handles to concrete backend buffers for one
/// `execute`. The model holds the buffers; this only borrows them, so re-binding per step is cheap
/// and the graph/plan is reused across steps without recompilation.
#[derive(Default)]
pub struct Bindings<'a> {
    map: HashMap<TensorId, &'a dyn Buffer>,
}

impl<'a> Bindings<'a> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Bind a graph handle to a buffer the model owns.
    pub fn bind(&mut self, id: TensorId, buf: &'a dyn Buffer) -> &mut Self {
        self.map.insert(id, buf);
        self
    }

    /// Look up a bound buffer (backend uses this while executing).
    pub fn get(&self, id: TensorId) -> Option<&'a dyn Buffer> {
        self.map.get(&id).copied()
    }
}

/// A compute device. Implementations: `infr-vulkan`, `infr-cpu`, later CUDA / ROCm / Metal / MLX.
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Capabilities;

    // ---- memory ----
    /// Allocate a buffer of `bytes`, **guaranteed zero-initialized** (calloc semantics). This is the
    /// safe default: code that reads a buffer before fully writing it (accumulators, recurrent state,
    /// KV caches, padding rows) behaves identically on every backend. GPU backends return recycled,
    /// uninitialized VRAM otherwise, so relying on implicit zeroing is a silent CPU-works/GPU-garbage
    /// trap — always `alloc` unless you can prove every element is written before it's read.
    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>>;
    /// Allocate WITHOUT zero-initialization — an explicit opt-out for hot buffers whose full extent is
    /// provably written before any read (e.g. weights, which are immediately uploaded). Faster (skips
    /// the zero-fill) but UNSAFE if misused. In debug builds the returned memory is POISONED (filled
    /// with a non-zero pattern) so a read-before-write surfaces loudly in tests instead of silently
    /// working on CPU. The default forwards to [`alloc`](Self::alloc) (safe); backends override for
    /// the perf win.
    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        self.alloc(bytes, usage)
    }
    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()>;
    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()>;
    /// Open a weight-load progress scope: while the returned guard lives, this backend's weight
    /// allocations (`BufferUsage::Weights`/`HostWeights`) advance a visible progress display;
    /// dropping the guard finishes and clears it. The ticking lives in each backend's `alloc`, so
    /// no loader can forget a tensor — a loader only opens the scope around its upload loop.
    /// Backends without a display (the CPU interpreter's zero-copy mmap load is instant) keep
    /// this default no-op scope.
    fn weight_progress(&self, _total_bytes: Option<u64>) -> Box<dyn ProgressScope> {
        struct NoProgress;
        impl ProgressScope for NoProgress {}
        Box::new(NoProgress)
    }
    /// Copy the first `bytes` of `src` into the start of `dst` (both buffers this backend's).
    /// The KV prefix-sharing primitive: a new chat slot seeds its cache from another slot's
    /// common prefix instead of re-prefilling it. Default is a host bounce (download the whole
    /// src, upload the prefix) — correct everywhere; backends override with a device-side copy.
    fn copy_buffer(&self, src: &dyn Buffer, dst: &dyn Buffer, bytes: usize) -> Result<()> {
        let mut tmp = vec![0u8; src.len_bytes()];
        self.download(src, &mut tmp)?;
        self.upload(dst, &tmp[..bytes])
    }

    // ---- execution (compile once per shape, execute per token/step) ----
    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>>;
    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()>;
    /// Chained decode: run the record-once decode plan `n` times in ONE submission, the sampled
    /// token id flowing device-side from each iteration's `Op::Argmax`/`Op::Sample` into the next
    /// iteration's `Op::EmbedGather` — the caller MUST have bound the sampler output and the
    /// embed-gather ids input to the SAME buffer, seeded with the first token to feed. Returns
    /// the `n` sampled ids, or `None` when this backend/plan can't chain (the caller falls back
    /// to per-token `execute`). Positions/params self-advance (a chaining backend implies the
    /// device-side pos increment).
    fn execute_chain(
        &self,
        _plan: &dyn Plan,
        _bindings: &Bindings,
        _n: usize,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }
    fn sync(&self) -> Result<()>;

    /// Perf (DiffusionGemma denoise, perf slice 3 — docs/DIFFUSIONGEMMA.md): fused per-canvas-row
    /// entropy-bound sampler reduction over raw `[rows, dim]` logits, avoiding a full `[rows,
    /// dim]` host download. `u` is `rows` host-drawn uniform `[0,1)` floats (the seeded
    /// CDF-inversion target draw — reproducibility rides the SAME host RNG as the CPU sampler,
    /// only the reduction itself moves to the GPU). On success, writes:
    ///   - `argmax_out[r]` (u32): argmax token id over `raw[r,:] * temp_inv`.
    ///   - `entropy_out[r]` (f32): entropy of `softmax(raw[r,:] * temp_inv)`.
    ///   - `sampled_out[r]` (u32): one multinomial draw from that softmax via forward (vocab-
    ///     order) CDF inversion against `u[r]` — matches the host sampler's algorithm exactly
    ///     (same order, same target), NOT bit-identical (GPU float reduction order differs).
    ///
    /// Returns `Ok(false)` (default: every backend but Vulkan) when unsupported — the caller falls
    /// back to a full `download` + host reduction. Never partially writes the outputs on `Ok(false)`.
    #[allow(clippy::too_many_arguments)]
    fn eb_sample_reduce(
        &self,
        logits: &dyn Buffer,
        u: &dyn Buffer,
        rows: usize,
        dim: usize,
        temp_inv: f32,
        argmax_out: &dyn Buffer,
        entropy_out: &dyn Buffer,
        sampled_out: &dyn Buffer,
    ) -> Result<bool> {
        let _ = (
            logits,
            u,
            rows,
            dim,
            temp_inv,
            argmax_out,
            entropy_out,
            sampled_out,
        );
        Ok(false)
    }
}
