//! The compute backend seam — the ONLY device-aware trait. Everything above is generic over it.
//!
//! Object-safe on purpose so the engine can hold `Arc<dyn Backend>` and stay blind to whether
//! Vulkan / CPU / CUDA / ROCm / Metal / MLX is underneath. See docs/PLAN.md.
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

/// The 16x16x16 (M,N,K) cooperative-matrix tile every production coopmat shader is built for
/// (RADV/RDNA3+, NVIDIA — every `coopmat<...,16,16,...>` declaration across
/// gemm_warp/native_gemm*/attn_*/deltanet_prep).
pub const COOPMAT_TILE_16: (u32, u32, u32) = (16, 16, 16);
/// The 8x8x16 (M,N,K) tile Intel Arc (Mesa ANV, XMX) enumerates for f16 — the ONLY non-16x16x16
/// shape any kernel here is built for, and only `native_gemm_warp`'s `_cm8` variants at that
/// (opt-in via `INFR_CM_8X8=1`; see the Vulkan backend's shape selection).
pub const COOPMAT_TILE_8: (u32, u32, u32) = (8, 8, 16);

/// Device capabilities the graph compiler queries to pick fast vs fallback kernels.
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub name: String,
    // ── per-type compute capabilities ───────────────────────────────────────────────────────────
    // Naming: `f16`/`f8`/`i8` are the infr spelling for the numeric types (== fp16/fp8/int8). Two
    // ORTHOGONAL axes, expressed as flat `<type>` + `<type>_<primitive>` bools (a device can have
    // one without the other — e.g. coopmat with only f16 components, or f16 ALU without coopmat):
    //   • `<type>`          — the device supports this type for scalar/vector storage & math
    //                         (f16 = shaderFloat16, f8 = the float8 storage/convert ext,
    //                         i8 = shaderInt8). Drives the scalar / non-coopmat fallback kernels.
    //   • `<type>_<prim>`   — the type is usable by a matrix/dot PRIMITIVE: `_coopmat` = accepted as
    //                         a cooperative-matrix component (enumerated from the device's coopmat
    //                         config list); `i8_dot` = packed dp4a integer dot.
    // infr's coopmat GEMM dequants any on-disk weight dtype to f16 IN-SHADER before the multiply,
    // so today's GEMM keys off `f16_coopmat` specifically; `f8_coopmat` is the (pending) fp8 tier.
    // (No `f32` capability flag: f32 ALU is universal — every device has it — so a flag would gate
    // nothing. f32 weights that DO ship in some GGUFs, e.g. qwen3moe's router, run on the
    // unconditional `linear_f32` GEMV. And no target GPU enumerates an f32-operand coopmat config.)
    /// f16 (== fp16) scalar/vector ALU (`shaderFloat16`) — the non-coopmat f16 warp/GEMV fallback.
    pub f16: bool,
    /// The (M,N,K) tile the cooperative-matrix unit accepts for f16 components (f16×f16→f16/f32),
    /// picked from the device's enumerated config list by preference order: [`COOPMAT_TILE_16`]
    /// first (the production shape every coopmat shader is built for), then [`COOPMAT_TILE_8`]
    /// (Intel Arc XMX — only honored when the `INFR_CM_8X8=1` opt-in is set, since only the
    /// `native_gemm_warp` `_cm8` builds exist at that shape). `None` = no usable f16 coopmat
    /// (route to the non-coopmat tiers). Call sites that mean "the full 16x16x16 kernel set is
    /// live" use the [`f16_coopmat`](Self::f16_coopmat) accessor, which is EXACTLY the pre-shape-
    /// table boolean.
    pub coopmat_f16: Option<(u32, u32, u32)>,
    /// f8 (== fp8, E4M3/E5M2) storage/convert support (`VK_EXT_shader_float8`-class). infr has no
    /// scalar f8 math path today; this exists for symmetry / future use. False on RDNA3.
    pub f8: bool,
    /// The (M,N,K) tile the cooperative-matrix unit accepts for f8 (E4M3/E5M2) A/B components —
    /// the fp8 coopmat GEMM tier. Detected by enumerating the device's coopmat configs; NOT a
    /// subset of `coopmat_f16` (a unit can do f16 components but not f8). Only
    /// [`COOPMAT_TILE_16`] is ever selected (no f8 8x8x16 kernels exist). `None` on all
    /// pre-RDNA4/pre-Ada HW.
    pub coopmat_f8: Option<(u32, u32, u32)>,
    /// i8 (== int8) storage & math in shaders (`shaderInt8`) — the scalar integer path.
    pub i8: bool,
    /// The device advertises packed i8 dot-product (`VK_KHR_shader_integer_dot_product` /
    /// `dotPacked4x8AccSat`, core in Vulkan 1.3) — the decode i8 `mmv` path's dp4a accumulate. A
    /// SEPARATE primitive from coopmat (hence not `coopmat_i8`). False = route to the scalar
    /// dequant GEMV (needs no extension). Independent of `f16`/`coopmat_f16`.
    pub i8_dot: bool,
    /// The (M,N,K) tile the cooperative-matrix unit accepts for SINT8×SINT8→SINT32 components
    /// (enumerated from the device's coopmat config list, same discipline as `coopmat_f16`; only
    /// [`COOPMAT_TILE_16`] is ever selected — the shape every int8 coopmat shader is hardcoded
    /// to). DETECTION ONLY — Some on this RX 7900 XTX (Mesa 26.1.4), but int8 coopmat previously
    /// HUNG the GPU on an older Mesa (commit ad82a77; the standalone `coopmat_int8_test` harness
    /// confirmed the fix). The kernel is therefore an ADDITIONAL opt-in gate on top of this:
    /// the adapter only dispatches it when `INFR_I8_COOPMAT=1` is also set (default off) — see
    /// adapter.rs's `Op::Linear` GEMM branch.
    pub coopmat_i8: Option<(u32, u32, u32)>,
    /// bf16 (bfloat16) scalar storage/convert support (`VK_KHR_shader_bfloat16`-class). Distinct
    /// from `f16` (IEEE half): same 16 bits but 8 exponent / 7 mantissa. False on RDNA3.
    pub bf16: bool,
    /// The (M,N,K) tile the cooperative-matrix unit accepts for bf16 components
    /// (bf16×bf16→bf16/f32). Enumerated from the device's coopmat config list (raw
    /// `VK_COMPONENT_TYPE_BFLOAT16_KHR`, confirmed on RDNA4/Navi44). NOT a subset of
    /// `coopmat_f16`. Only [`COOPMAT_TILE_16`] is ever selected. `None` on all pre-RDNA4 HW.
    pub coopmat_bf16: Option<(u32, u32, u32)>,
    /// Supported subgroup-size range (`VkPhysicalDeviceSubgroupSizeControlProperties`). The coopmat
    /// GEMM pins `requiredSubgroupSize = 32` (RDNA3 wave32); a device whose range excludes 32 can't
    /// run the pinned kernel and must fall back to a non-pinned variant. `(0, 0)` =
    /// subgroup-size-control unsupported (can't pin at all — use the driver's default subgroup).
    pub subgroup_min: u32,
    pub subgroup_max: u32,
    /// PREFERRED pinned subgroup size for the bandwidth-critical decode GEMV/reduction kernel
    /// family (the `_sg16` SPIR-V twins: `native_gemv_sg` / `native_gemv_id_multi_sg` /
    /// `native_mmv_mw` / `mul_mat_vec_q` / `quant_q8_row`). 16 on Intel (`vendor_intel` and the
    /// device can pin 16): compiling those kernels SIMD32 on SIMD8-EU hardware strangles per-lane
    /// registers — llama.cpp pins 16 for mul_mat_vec there (ggml-vulkan.cpp:4839). 32 everywhere
    /// else (RADV wave32 — 16 is not even pinnable there, `subgroup_min == 32`). Every OTHER
    /// pinned-32 kernel (rmsnorm/softmax/coopmat GEMM/attention…) is unaffected by this field.
    /// `INFR_SG=16|32` overrides for A/B; a request the device can't pin falls back to 32.
    /// 0 on backends without subgroup pinning (CPU/Metal — the field gates Vulkan shader picks).
    pub sg_pref: u32,
    /// `vendorID == 0x8086` (Intel). Drives Intel-measured kernel-policy defaults (decode mmv
    /// dp4a tier default-on, `sg_pref = 16`) — NOT detected from subgroup sizes (some Xe2 SKUs
    /// report `subgroup_min` 8, others 16, so size-sniffing would misclassify).
    pub vendor_intel: bool,
    /// `VkPhysicalDeviceProperties.deviceType == INTEGRATED_GPU` — an iGPU/APU sharing the CPU's
    /// memory controller and carrying one to two ORDERS OF MAGNITUDE less compute than the
    /// discrete cards every dispatch shape here is tuned for (a Ryzen 9950X3D's RDNA2 iGPU is
    /// ~2 CU against a 7900 XTX's 96).
    ///
    /// The load-bearing consequence is the GPU WATCHDOG, not throughput: amdgpu resets a `gfx`-ring
    /// job that runs past ~10 s (`ring gfx_0.0.0 timeout` -> `VK_ERROR_DEVICE_LOST`), and infr
    /// submits ONE command buffer per forward pass — so a whole prefill chunk is a SINGLE watchdog
    /// job. A default 1024-row chunk measures 0.78-1.12 s of GPU on a 7900 XTX in the NON-COOPMAT
    /// tier an iGPU is forced onto (RDNA2 has no cooperative matrix), which at the ~32x slowdown
    /// calibrated from a real iGPU run lands at 25-36 s — a guaranteed TDR. Devices that set this
    /// get a smaller default prefill chunk (see the seam's `ubatch_rows`), the one knob that
    /// bounds per-submit GPU time. Discrete devices leave it false and every shape stays as tuned.
    pub integrated: bool,
    /// Shader/compute-unit count when the device advertises one (`VK_AMD_shader_core_properties`),
    /// else 0 = UNKNOWN. ADVISORY only — it scales the `integrated` chunk budget where present.
    /// Never gate correctness on it: most drivers report nothing.
    pub compute_units: u32,
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
    /// The backend executes [`crate::Op::ArgmaxProb`] (fused single-row argmax + softmax top-1
    /// probability — the MTP draft-loop accept, issue #33 follow-up: only 8 bytes read back
    /// instead of the `[vocab]` logits + host argmax/softmax scan). Backends without the kernel
    /// leave this false and the MTP driver keeps the host-logits `top1_softmax` path.
    pub argmax_prob: bool,
    /// The backend executes [`crate::Op::GatedRmsNorm`] (fused per-head RMSNorm + SiLU gate
    /// multiply — qwen35's DeltaNet z-gate, one dispatch instead of `QkNorm`→`GatedAct`'s two).
    /// False = the runner keeps emitting the split pair (identical math, one extra
    /// read-after-write barrier on backends that have one).
    pub gated_rmsnorm: bool,
    /// The backend treats every KV cache buffer as a RING over its allocated row count: `WriteKv`
    /// lands row `pos % cap_rows` and `Attention` reads key/value position `j` at row
    /// `j % cap_rows`, where `cap_rows = declared cache elements / row width`. A full-context
    /// cache (`cap_rows >= every kv_len`) makes the modulo an identity, so setting this only
    /// matters when the runner allocates a sliding-window layer's cache at window size instead of
    /// full context (see the seam's SWA ring sizing). Backends whose kernels index rows directly
    /// by position (Metal) leave this false and get full-context allocations for every layer.
    pub kv_swa_ring: bool,
}

/// The DEFAULT prefill chunk (rows) for an INTEGRATED GPU with `cu` compute units (0 = unknown).
///
/// WHY a separate default at all: the GPU watchdog is per-SUBMIT, and infr submits one command
/// buffer per forward pass — so a prefill chunk is a single indivisible watchdog job. The chunk
/// height is therefore the ONLY knob that bounds it.
///
/// Where the numbers come from (measured, 7900 XTX under `INFR_NO_COOPMAT=1` — the exact kernel
/// tier an iGPU is forced onto, since RDNA2 enumerates no cooperative matrix):
///   * Qwen3-8B  Q4_K_M, one 512-row prefill chunk: 363 ms of GPU.
///   * gemma-3-12b Q4_K_M, same: 505 ms.
///   * The iGPU:dGPU slowdown on that tier CALIBRATES to ~32x from a real iGPU datapoint
///     (gemma-3-12b decode: 345 ms/tok observed on the iGPU vs 10.7 ms/tok here).
///
/// So a 512-row chunk is 12-16 s on the iGPU and a default 1024-row chunk is 25-36 s — both past
/// the ~10 s `gfx`-ring timeout. 128 rows puts the same chunk at ~3-4 s, a >2x margin, and is the
/// reference point below (the surveyed iGPU is a ~2-CU RDNA2 Raphael part).
///
/// Scaling: linear in CU count where the driver reports one, so a beefier iGPU (a 12-CU 780M) is
/// not needlessly throttled, clamped to the discrete default so this can only ever SHRINK the
/// chunk. An unknown CU count takes the conservative floor.
pub fn integrated_ubatch_rows(cu: u32) -> usize {
    /// The surveyed iGPU (RDNA2 Raphael/Mendocino) — the part the 128-row figure is calibrated on.
    const BASE_CU: u32 = 2;
    /// Watchdog-safe chunk at BASE_CU (see above).
    const BASE_ROWS: usize = 128;
    /// Never exceed the discrete default — this is a hardening cap, not a tuning knob.
    const MAX_ROWS: usize = 1024;
    if cu == 0 {
        return BASE_ROWS; // unknown: assume the weakest part we have measured
    }
    (BASE_ROWS * (cu as usize).div_ceil(BASE_CU as usize).max(1)).clamp(BASE_ROWS, MAX_ROWS)
}

impl Capabilities {
    /// The full 16x16x16 f16 coopmat kernel set is live (`coopmat_f16 == COOPMAT_TILE_16`) —
    /// byte-for-byte the boolean every pre-shape-table gate keyed on. All existing coopmat
    /// shaders (GEMM warptiles, flash attention, deltanet_prep) are built ONLY at this shape,
    /// so they must stay dark when the device's tile is 8x8x16.
    pub fn f16_coopmat(&self) -> bool {
        self.coopmat_f16 == Some(COOPMAT_TILE_16)
    }
    /// The 8x8x16 f16 tile was selected (Intel Arc XMX under `INFR_CM_8X8=1`) — gates ONLY the
    /// `native_gemm_warp` `_cm8` builds; every other coopmat family has no kernel at this shape.
    pub fn f16_coopmat_8x8(&self) -> bool {
        self.coopmat_f16 == Some(COOPMAT_TILE_8)
    }
    /// 16x16x16 f8 coopmat GEMM tier available (see `coopmat_f8`).
    pub fn f8_coopmat(&self) -> bool {
        self.coopmat_f8 == Some(COOPMAT_TILE_16)
    }
    /// 16x16x16 i8 coopmat available — DETECTION ONLY, the adapter additionally requires
    /// `INFR_I8_COOPMAT=1` (see `coopmat_i8`).
    pub fn i8_coopmat(&self) -> bool {
        self.coopmat_i8 == Some(COOPMAT_TILE_16)
    }
    /// 16x16x16 bf16 coopmat GEMM tier available (see `coopmat_bf16`).
    pub fn bf16_coopmat(&self) -> bool {
        self.coopmat_bf16 == Some(COOPMAT_TILE_16)
    }
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

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl GraphPlan {
    pub fn boxed(graph: &crate::graph::Graph) -> Box<dyn Plan> {
        Box::new(GraphPlan {
            graph: graph.clone(),
        })
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
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

#[cfg_attr(infr_profile, infr_prof::instrument)]
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

    /// Whether the currently loaded model is running any MoE expert layer through a paged VRAM
    /// cache (an expert bank too big to keep fully resident — see `infr_vulkan::pager::GpuPager`).
    /// Paged execution needs a host readback of the router's chosen expert ids BETWEEN a layer's
    /// GEMV stages (to resolve/upload cache misses before the id-indexed GEMV reads them), which a
    /// cached record-once decode replay can't express (the whole point of replay is recording
    /// every position-dependent op ONCE with no host round-trip in between). The seam's decode-loop
    /// gate and the Vulkan adapter's own `execute`/`execute_chain` both check this and force the
    /// per-execute static path instead — see `infr_vulkan::adapter::execute`'s doc.
    /// `false` for every backend but Vulkan, and for Vulkan whenever no paged model is loaded (the
    /// overwhelming common case: MoE experts that fit VRAM are fully resident with zero change).
    fn moe_paged(&self) -> bool {
        false
    }

    /// Whether the currently loaded DENSE model streams per-layer weight blocks through a paged
    /// VRAM cache (dense layer streaming — the `crate::pager::Pager::schedule` policy; see
    /// `infr_vulkan::pager`'s dense session). Unlike [`Backend::moe_paged`] no host readback is
    /// ever needed (dense layer order is deterministic, every "miss" is known in advance), but
    /// the record-once decode replay still can't express the per-token ring staging +
    /// arena-offset rebinding, so the same gates that force a paged-MoE model onto the
    /// per-execute static path check this too. `false` for every backend but Vulkan, and for
    /// Vulkan whenever the loaded model's dense weights fit VRAM (the overwhelming common case —
    /// zero change there).
    fn dense_paged(&self) -> bool {
        false
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    /// The integrated chunk policy. The load-bearing property is the CEILING: this may only ever
    /// SHRINK the discrete default (1024), never grow it — it is a watchdog guard, not a tuning
    /// knob — and it must never return 0 (a zero-row prefill chunk would not terminate).
    #[test]
    fn integrated_ubatch_rows_is_a_bounded_shrink() {
        // Unknown CU count (the common case: only AMD reports one) takes the conservative floor —
        // the ~2-CU RDNA2 iGPU the 128-row figure is calibrated on.
        assert_eq!(integrated_ubatch_rows(0), 128);
        assert_eq!(integrated_ubatch_rows(2), 128); // the calibrated part
        assert_eq!(integrated_ubatch_rows(1), 128); // never below the floor
                                                    // Linear in CU count, so a beefier iGPU is not needlessly throttled.
        assert_eq!(integrated_ubatch_rows(4), 256);
        assert_eq!(integrated_ubatch_rows(12), 768); // Radeon 780M class
                                                     // ...but clamped at the DISCRETE default: this can only shrink the chunk, never grow it.
        assert_eq!(integrated_ubatch_rows(16), 1024);
        assert_eq!(integrated_ubatch_rows(64), 1024);
        assert_eq!(integrated_ubatch_rows(u32::MAX), 1024);
        for cu in [0, 1, 2, 3, 7, 8, 33, 96, 1024, u32::MAX] {
            let r = integrated_ubatch_rows(cu);
            assert!((128..=1024).contains(&r), "cu={cu} -> {r} out of bounds");
        }
    }

    /// A DISCRETE GPU must be untouched by any of this: `integrated` defaults false, so the seam's
    /// `default_ubatch_rows` takes its 1024 branch and no tuned dGPU shape moves.
    #[test]
    fn discrete_is_the_default() {
        let caps = Capabilities::default();
        assert!(!caps.integrated);
        assert_eq!(caps.compute_units, 0);
    }
}
