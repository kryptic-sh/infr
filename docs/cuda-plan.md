# cuda-plan.md — a native CUDA backend for infr

Plan for adding `infr-cuda`, a fifth compute backend (alongside CPU, Vulkan,
Metal, and the planned ROCm — see `docs/rocm-plan.md`) targeting NVIDIA GPUs
through the CUDA stack. Like the ROCm plan, this is ordered **correctness first,
performance second**: it reaches a fully correct backend — every model and every
quant format infr supports, generating token-for-token agreement with the CPU
reference — before a single perf lever is pulled, then climbs from naive kernels
to a fast kernel for every supported model × quant combination.

> Status: **PLAN ONLY — nothing built yet.** This doc is the roadmap and the
> design record. Update the phase checkboxes and the commit refs as slices land,
> the same way `docs/igpu.md` and `docs/cpu-perf.md` track their campaigns.

CUDA and ROCm/HIP are near-mirror images (HIP was designed as a CUDA clone), so
this plan is a deliberate sibling of `docs/rocm-plan.md` — same backend seam,
same phase arc, same correctness discipline. The two differ only in toolchain
(NVRTC/cuBLASLt/CUTLASS/Tensor Cores vs hiprtc/rocBLAS/WMMA/MFMA) and, most
consequentially, in **where it can be validated** (see the hardware note below).

## Why a CUDA backend

The Vulkan backend already runs on NVIDIA GPUs (coopmat GEMM, dp4a `mmq`, flash
attention) and is portable. A native CUDA backend is still worth building:

- **The highest perf ceiling infr can target.** CUDA exposes **Tensor Cores**
  directly (WMMA / `mma.sync`), the most mature GEMM libraries in existence
  (**cuBLASLt**, **CUTLASS**), native **`__dp4a`** int8 (`sm_61`+), **fp8**
  Tensor Cores (Ada `sm_89` / Hopper `sm_90`), and — the one genuine
  low-bit-float compute win called out in `docs/perf.md` — **Blackwell's
  block-scaled fp4 MMA** (`mma…kind::mxf4nvf4.block_scale…e2m1…ue4m3`,
  `sm_100`+), which no other backend can reach. For MXFP4/NVFP4 models this is a
  decisive advantage over every dequant-to-f16 path.
- **The definitive reference target.** `llama.cpp` built with `-DGGML_CUDA=ON`
  is the most mature, most optimized ggml backend; matching it on the same
  NVIDIA device is the highest bar infr can set, and the cleanest ratio.
- **Cross-platform reach.** CUDA runs on Linux **and** Windows, widening infr's
  first-class-GPU footprint beyond the Vulkan/Metal split.
- **Tooling.** Nsight Compute / Nsight Systems and `cuda-gdb` give
  hardware-counter and warp-level attribution beyond what the Vulkan timestamp
  path offers.

Non-goal: replacing Vulkan. CUDA is additive; Vulkan stays the portable default.

## Hardware caveat — CUDA cannot be validated on the dev box

Unlike the ROCm plan (the dev box's RX 7900 XTX is an official ROCm card, so
that backend is A/B-testable against Vulkan on the _same_ silicon), **there is
no NVIDIA GPU on the dev box** (`GPU0` = RX 7900 XTX, `GPU1` = AMD iGPU — see
`docs/igpu.md`). CUDA validation therefore requires NVIDIA hardware that is not
local:

- **Primary path — a cloud / rented NVIDIA instance** (e.g. a single L4 / A10 /
  RTX 4090 / L40S box). The workflow mirrors the **blind Metal** campaign
  (`docs/kernels.md`): author kernels against the CPU oracle locally,
  byte-verify the decode math, push, and run the ignored parity suite on the
  remote GPU box.
- **Secondary — a GPU CI runner.** A self-hosted or cloud NVIDIA runner running
  the ignored parity/goldens suite (the twin of the `test-macos` job) is the
  eventual gate; unlike macOS, GitHub offers GPU runners, so this is more
  attainable than the ROCm case.

This caveat shapes the whole plan: every kernel must be **byte-verifiable
against the CPU reference before it ever runs on a GPU** (the discipline that
made the blind Metal quant kernels correct), because the edit→validate loop is
remote, not local.

## What a backend actually has to do (the seam is already generic)

Identical to the ROCm plan — everything above the device is backend-agnostic:
the `Graph`/`Op` IR, the seam runner (`generate_dense_backend`), the chat REPL,
KV-cache and MoE routing policy. A backend is exactly:

1. Implement **`infr_core::backend::Backend`**
   (`crates/infr-core/src/backend.rs:464`) plus `Buffer` (`:374`), `Plan`
   (`:397`), `ProgressScope` (`:372`).
2. Fill in **`Capabilities`** (`backend.rs:42`) honestly — the runner degrades
   to split-op fallbacks for anything reported unsupported.
3. Execute every **`Op`** variant (`crates/infr-core/src/graph.rs`, flat list in
   `Op::kind` `:554`).

`infr-metal` remains the from-scratch template to mirror (assemble kernel
source, compile once at init, cache pipelines by name, walk the graph in a
`run_op` match). `infr-rocm` will be a closer sibling still; wherever this plan
says "same as ROCm", it means the identical seam surface with the CUDA toolchain
swapped in.

### The `Backend` trait surface (required unless noted)

| method                                   | ref              | correctness-phase behavior                                    |
| ---------------------------------------- | ---------------- | ------------------------------------------------------------- |
| `name(&self) -> &str`                    | `backend.rs:465` | `"cuda"`                                                      |
| `capabilities(&self) -> Capabilities`    | `:466`           | conservative set (below)                                      |
| `alloc(bytes, usage)`                    | `:474`           | `cuMemAlloc` + **zero-init** (`cuMemsetD8`) — calloc contract |
| `upload(dst, &[u8])`                     | `:484`           | `cuMemcpyHtoD`                                                |
| `download(src, &mut [u8])`               | `:485`           | `cuMemcpyDtoH`                                                |
| `compile(&Graph)`                        | `:517`           | `GraphPlan::boxed(graph)` (clone; work happens in execute)    |
| `execute(plan, bindings)`                | `:518`           | walk graph → launch kernels on the stream → one sync          |
| `sync()`                                 | `:552`           | `cuStreamSynchronize`                                         |
| `alloc_uninit` (default → `alloc`)       | `:481`           | skip zero-init once proven; debug-poison                      |
| `weight_progress` (default no-op)        | `:498`           | drive the `indicatif` bar from inside `alloc` (Metal does)    |
| `copy_buffer` (default host-bounce)      | `:508`           | override with `cuMemcpyDtoD` (KV prefix-share primitive)      |
| `execute_chain` (default `Ok(None)`)     | `:526`           | perf phase: n back-to-back decode steps, one launch batch     |
| `max_decode_chain` (default `MAX`)       | `:548`           | perf phase clamp                                              |
| `moe_paged` / `dense_paged` (`false`)    | `:564/:577`      | perf phase (VRAM streaming)                                   |
| `eb_sample_reduce` (default `Ok(false)`) | `:595`           | perf phase (fused diffusion sampler)                          |

No `warmup` on the trait — it is `ChatModel::warmup` (`chat/mod.rs`), inherited
by the CUDA chat wrapper for free.

### `Capabilities` — the correctness dial

Same rule as ROCm: **advertise the minimum, let the runner split everything**
(`backend.rs:42`). Start with:

- `name = "cuda"`, `f16 = true`, `i8 = false`, `i8_dot = false`, all
  `coopmat_* = None`.
- `integrated = false` (any real NVIDIA target is discrete; skip the
  submit-splitter math, `backend.rs:223/271`). Jetson/integrated parts are out
  of scope for the first pass.
- `compute_units` = SM count (cosmetic until perf).
- Every fused-op opt-in **false**: `decode_replay`, `combined_gu`,
  `embed_gather`, `gpu_sample`, `argmax_rows`, `argmax_prob`, `gated_rmsnorm`,
  `kv_swa_ring`.
- `buffer_device_address` — CUDA device pointers are raw addresses; can be
  `true` early if a kernel wants pointer-fed args, else leave `false`.

Each field flips to `true` only once the matching native kernel exists and
passes parity. Flipping a cap before the kernel exists is the one way to get
silent garbage.

## Kernel authoring strategy

Same three routes as ROCm, CUDA-flavored; **(A) for correctness, (B/C) for
perf**:

- **(A) NVRTC runtime compilation** — mirror Metal exactly: keep CUDA C++
  kernels as `.cu` source `include_str!`'d into the crate, assemble one
  translation unit at `CudaBackend::new()`, compile once with **NVRTC**
  (`nvrtcCompileProgram` → PTX), load with `cuModuleLoadData`, and lazily
  fetch+cache `CUfunction` by name (`cuModuleGetFunction`). Quant codebook LUTs
  are emitted from `infr_core::iquant_grids` (same statics every other backend
  reads — bit-exact by construction). Fastest path to "all kernels present";
  matches `infr-metal/src/shaders.rs`.
- **(B) Offline `nvcc` in `build.rs`** — compile `.cu` → PTX/cubin **fatbins**
  across the target compute capabilities (`-gencode arch=…`), `include_bytes!`
  the blobs, load with `cuModuleLoadFatBinary`/`cuModuleLoadData`. Better
  startup latency + full optimizer; add once the kernel set stabilizes
  (analogous to Vulkan's `build.rs` glslc pipeline).
- **(C) Vendor libraries** — **cuBLASLt** for the f16/bf16/int8 GEMM tiles once
  weights are dequantized (its int8 IMMA and fp8 paths are best-in-class), and
  **CUTLASS** for fused/custom epilogues where a hand kernel would otherwise be
  needed. Used in the perf phases for the wide prefill GEMM; custom kernels stay
  for quant-decode GEMV, attention, MoE routing, DeltaNet.

Reuse `infr_core::dequant` and `infr_core::iquant_grids` verbatim — the shared
correctness anchors.

## Crate + workspace wiring

Mirror `infr-metal`'s platform gating (and the planned `infr-rocm`) so the
workspace still builds everywhere:

- New crate `crates/infr-cuda`, added to `Cargo.toml` `members`.
- Gate the real implementation behind a cargo feature (**`cuda`**) — CUDA runs
  on Linux and Windows, so the gate is the feature (toolkit present), not the
  OS. Without the feature the crate is a no-op empty lib with zero CUDA deps,
  exactly as `infr-metal` is off macOS (`crates/infr-metal/src/lib.rs:4`).
- Deps: `infr-core`, `infr-gguf`, `half`, `bytemuck`, `indicatif`; the CUDA
  Driver API + NVRTC FFI behind the feature (hand-written `extern "C"` to
  `libcuda`/`libnvrtc`, or a vetted `cudarc`/`cust`-style crate — evaluate in
  Phase 0; `cudarc` is a strong candidate: pure-FFI, driver-API, no build-time
  CUDA required for the bindings themselves). Dev-dep `infr-cpu` as the parity
  oracle.

### Selection / registration (the `INFR_DEV` grammar)

Add a `cuda` token to the one grammar shared by `--dev` and `INFR_DEV`
(`crates/infr-cli/src/main.rs`):

1. `enum Backend { Vulkan(Option<String>), Metal, Cpu, Cuda }` (`main.rs:62`).
2. `parse_dev_spec` (`:99`) — accept `cuda` (optionally `cuda:N` to pin a
   device, matching `VulkanN`).
3. `DeviceOpts::resolve` / `selected_backend` (`:187/:134`) — publish/read it.
4. `build_chat_model` (`:841`) and the diffusion-gemma 3-arm match (`:861`) —
   add a `Backend::Cuda` arm building a `CudaSeamChat` (twin of
   `metal_chat_model`, `:1392`). Without the `cuda` feature, fall back with a
   clear error (mirror Metal's non-macOS fallback).

### Chat + session plumbing

- `crates/infr-llama/src/chat/cuda.rs` — `CudaSeamChat: ChatModel`
  (`chat/mod.rs:38`), twin of `chat/metal.rs`. Re-export in `chat/mod.rs`.
- `DenseCudaSession` in `crates/infr-llama/src/seam/model.rs` (twin of
  `DenseMetalSession`, `:272`) owning a `CudaBackend`; a
  `SeamModel::cuda_session` calling `CudaBackend::new()` (mirror
  `metal_session`, `:1194`).
- `cuda_upload_bind` in `crates/infr-llama/src/seam/mod.rs` (twin of
  `metal_upload_bind`, `:97`): alloc device buffer + upload the **raw native
  dtype** bytes, lazy-dequant on first use. Then drive the existing
  `generate_dense_backend` runner unchanged.

### The `be.name()` capability gates to thread `"cuda"` into

Same set as ROCm — the seam decisions that branch on the backend name string, in
`crates/infr-llama/src/seam/runner.rs`: `kv_q8_backend` (`:432`), `kv_turbo_ok`
(`:435`), `blk_ok` (`:439`), `dense_ok` (`:442`), the Metal-only
`k_fmt != v_fmt` guard (`:477`), and the `gpu_sc` gate (`:3030`). Add the
`"cuda"` arm as each feature lands — never before its kernel exists.

---

# PART A — CORRECTNESS (all models, all quants, token-for-token)

Goal: `INFR_DEV=cuda infr run <any supported model>:<any supported quant>`
loads, generates coherent text, and matches the CPU reference token-for-token —
with **naive kernels only**. No perf work. Validated on a cloud/remote NVIDIA
box.

## Phase 0 — Scaffolding: an empty backend that builds and is selectable

- Create `crates/infr-cuda` (feature-gated empty lib), wire it into the
  workspace and the `INFR_DEV=cuda` selection chain (all sites above).
- Evaluate the CUDA FFI surface: `cuInit`, `cuDeviceGet`, `cuCtxCreate`,
  `cuMemAlloc/Free`, `cuMemcpyHtoD/DtoH/DtoD`, `cuMemsetD8`,
  `cuStreamCreate/ Synchronize`, `nvrtcCompileProgram`,
  `cuModuleLoadData/GetFunction`, `cuLaunchKernel`, `cuDeviceGetAttribute` (SM
  count, compute capability). Decide `cudarc` vs hand-rolled `extern "C"`.
- Provision the **remote NVIDIA validation box** (cloud instance or CI runner)
  and a push→run loop, since nothing here can run locally.
- `CudaBackend::new()` enumerates the device, creates a context + stream,
  reports `capabilities()` — no kernels yet.
- **Gate:** `cargo build -p infr-cuda --features cuda` (on the remote box or any
  machine with the toolkit); workspace `cargo build` green everywhere without
  the feature. `infr` recognizes `--dev cuda` and errors cleanly if the feature
  is absent. First parity test: `upload`→`download` byte-identity on the remote
  GPU.
- **Exit:** the backend constructs, allocs/uploads/downloads/`sync`s on real
  NVIDIA hardware.

## Phase 1 — Correctness baseline: dequant→f16 everything

The same day-one-coverage trick as ROCm: never decode a quant natively yet. In
`cuda_upload_bind`, keep raw bytes; in `Linear`/`EmbedGather`/expert paths,
dequant the block to `f16` on first touch via `infr_core::dequant`, cache it,
and run a **single naive f16 GEMV/GEMM kernel**. All 24 weight quant formats +
floats reduce to the same f16 matmul → full coverage immediately
(`docs/kernels.md`).

Kernels this phase (all naive, one variant each, no tiling): `linear_f16`
GEMV/GEMM; the norm/elementwise/rope set (`RmsNorm`, `RmsNormAdd`, `Softmax`,
`QkNorm`, `Rope`, `QkNormRope`, `GatedRmsNorm`, `Add`, `AddBias`, `Scale`,
`MulVec`, `Softcap`, `GatedAct`, `Copy`, `CopyStrided`, `EmbedGather`, `Argmax`,
`Sample`); `WriteKv` + scalar `Attention` (Causal + SlidingWindow, f16/f32 KV
only); `MoeFfn` unfused split path + `MoeSharedExpertAdd`; `Conv1dSilu` +
sequential `DeltaNet` (persistent `S` state, mutated in place, **surviving
across `execute` calls**).

Advertise nothing fused, so the runner emits the split ops (keeps the kernel
count minimal).

- **Gate:** per-op parity via `crates/infr-llama/tests/seam_op_parity.rs`
  (one-op agnostic `Graph`, CPU vs CUDA, incl.
  `state_persists_across_executes`), run `--include-ignored` on the remote box.
- **Exit:** a small dense model (Qwen3-0.6B Q4_K_M, gemma-3-1b) generates
  coherent text on `--dev cuda`.

## Phase 2 — Op completeness, all architectures, blessed goldens

Extend until **every** model family runs and the goldens lock:

- Cover every arch the CPU/Vulkan backends do: llama, qwen2/2.5, qwen3
  dense+MoE, gemma3, gemma4 dense/E2B/26B-A4B MoE, qwen3.5/3.6 (`qwen35`
  DeltaNet) + `qwen35moe`, diffusion-gemma, llama4 Scout, plus the fine-tunes
  (Ornith, Ternary-Bonsai).
- KV cache: f16 + f32 correct.
- Add the **CUDA `gpu_seam` golden**: a backend-specific FNV-1a hash blessed
  with `INFR_BLESS=1` (each backend locks its own — `cpu_backend.rs:48`), plus a
  token-for-token `seam_cuda_matches_cpu` sweep over the quant families (twin of
  `seam_vulkan_matches_cpu`, `cpu_backend.rs:380+`). Wire the `linear.rs` drift
  guards.
- Thread `"cuda"` into the `blk_ok`/`dense_ok` KV gates as formats gain correct
  read/write.

- **Gate:** full `gpu_seam` suite green on the remote box; token-for-token with
  the CPU oracle on every arch × a representative quant.
- **Exit of PART A:** every supported model × every supported quant produces
  correct, coherent output on CUDA, matching the reference. **Zero perf
  claims.**

## CI / validation strategy (applies across the plan)

Because there is no local NVIDIA GPU, this is the load-bearing section:

- **Byte-verify every kernel against the CPU reference before it runs on a GPU**
  (the blind-Metal discipline) — the remote loop is too slow for
  trial-and-error.
- A **`cuda-check`** CI job that `cargo check`s the workspace **without** the
  `cuda` feature (empty-lib path) to catch `Op`-signature / match-arm drift —
  the analogue of the existing `metal-check` guard (`ci.yml`).
- A **`test-cuda`** job on a GPU runner (self-hosted or cloud NVIDIA) running
  the ignored parity + `gpu_seam` suite — the twin of `test-macos`. GPU CI
  runners are attainable (unlike macOS-for-Metal there is no OS lock-in), so
  this is the real gate rather than a single dev box.
- Every kernel lands with a parity test vs a CPU reference **before** its
  `Capabilities` flag flips.

---

# PART B — PERFORMANCE (fast kernel per model × quant)

Goal: climb from the naive baseline to a fast kernel for every supported model ×
quant, benchmarked against `llama.cpp` (CUDA build) and, where a comparison box
exists, the Vulkan backend on the same NVIDIA device — targeting **≥1.0×** on
prefill and decode for every model×quant, without moving a golden. Follow the
`docs/perf.md` loop: sweep → profile → one lever → validate → bench serially,
biggest-gap-first. `llama.cpp` CUDA is the most optimized reference infr faces,
so expect this to be the hardest perf climb of any backend.

The NVIDIA levers, mapped to the Vulkan tiering (`adapter.rs:1340-1560`):

| Vulkan tier                   | CUDA analogue                                                     |
| ----------------------------- | ----------------------------------------------------------------- |
| f16 coopmat GEMM (prefill)    | **Tensor Cores** via WMMA / `mma.sync` / CUTLASS; or **cuBLASLt** |
| dp4a `mmq` int8 (decode)      | native **`__dp4a`** (`sm_61`+); IMMA Tensor Core int8 (`sm_75`+)  |
| scalar dequant-in-shader GEMV | plain CUDA GEMV (the Phase-1 baseline)                            |
| non-coopmat warptile          | shared-memory CUDA warptile (pre-Turing / no-TC shapes)           |
| — (no Vulkan analogue)        | **fp8** TC (Ada/Hopper), **block-scaled fp4 MMA** (Blackwell)     |

## Phase 3 — Native quant decode (drop the host dequant)

Replace the dequant→f16 cache with **in-kernel block decode** GEMV per format,
so decode streams the compact quant bytes (the dominant decode lever). One
kernel family, per-`DType` template/`#define` variants (mirror Vulkan's
`-DFMT_<QUANT>` build variants), driven off `ALL_DTYPES` (`linear.rs:373`) and
the IQ grid LUTs. Each format lands with a GEMV parity test vs the CPU
`vec_dot` + `dequant_block` (the Metal DEC16 pattern). Covers all 24 weight
formats.

- **Exit:** decode reads native quant bytes; per-format GEMV parity green.

## Phase 4 — Integer-dot decode (`__dp4a` / IMMA)

Add the int8-activation path: quantize the activation row to int8 once, integer
dot against native weight codes with **`__dp4a`** (the instruction Vulkan's dp4a
`mmq` and the CPU VNNI dots are modeled on — so this maps _directly_, and is the
most native of all backends here). Flip `i8`/`i8_dot`; derive coverage from the
shared `MOE_MMQ_DTYPES` SSOT (`tensor.rs`). Add the multi-row (`m=2..16`) GEMV
variant. Evaluate IMMA Tensor Core int8 (`sm_75`+) for the batched case.

- **Exit:** int8 decode parity-clean for the mmq dtype set; measured decode win.

## Phase 5 — Tensor Core GEMM (prefill) + the tiering decision tree

The big prefill lever, porting `adapter.rs`'s decision tree:

- **Tensor Core WMMA / `mma.sync` GEMM** with in-shader dequant (`matmul_native`
  analogue) + an f16/bf16 variant. Populate `coopmat_f16`/`coopmat_bf16` in
  `Capabilities` with the real TC tile (`16x16x16`).
- **IMMA int8 `mmq`** batched prefill for the mmq dtypes.
- Evaluate **cuBLASLt / CUTLASS** for the wide f16/bf16/int8 tiles vs a custom
  warptile — pick per-shape on a micro-bench (`gemm_bench` twin). cuBLASLt is
  likely to win the wide dense tails outright; the custom path stays for
  dequant-fused and narrow shapes.
- **fp8** (Ada/Hopper) and **Blackwell block-scaled fp4 MMA** for MXFP4/NVFP4 —
  the native low-bit-float compute path `docs/perf.md` identifies as the only
  real operand-level win; gate on compute capability.
- Tune the shared-memory / occupancy budget per arch (the occupancy-not-math
  lesson from the Vulkan A_GLOBAL slice applies).

- **Exit:** prefill on Tensor Cores; per-shape TFLOPS near the cuBLASLt ceiling;
  goldens unmoved (bit-identical construction where possible).

## Phase 6 — Fast attention + KV quantization

- **Flash attention** (fused, `hd==128`, Causal, Tensor Core) — the
  `attention_prefill_flash` analogue, geometry guards porting directly; consider
  leaning on a vetted flash-attention CUDA kernel where license permits, else a
  CUTLASS-based one. **Split-KV / flash-decoding** for decode.
- **KV quant:** Q8_0 planar + block formats via dequant→f16 prepass first, then
  dequant-in-attention for the cheap formats (Vulkan's `INFR_FLASH_DEQUANT`
  set). TurboQuant via the shared dequant→f16 prepass. Thread `"cuda"` into
  `kv_q8_backend`/`kv_turbo_ok`/`blk_ok` as each lands; enable `kv_swa_ring`
  once the sliding-window ring is correct.

- **Exit:** attention off the scalar fallback; KV-quant coverage matches the
  other GPU backends; decode-at-depth competitive with llama.cpp CUDA.

## Phase 7 — MoE (batched, id-GEMV, paged) + DeltaNet fast paths

- **MoE:** GPU-side top-k routing (router GEMV → bucket count/scan/scatter, no
  host readback — the class-5 host-in-the-loop fix); resident small-`m`
  **id-indexed GEMVs** and batched **`mmq`/IMMA experts** for larger `m` (gated
  on `moe_mmq_ok`). Enable `combined_gu`.
- **Paged experts** for models exceeding VRAM (llama4 Scout, big MoE): a CUDA
  `GpuPager` analogue (LUT hop to an arena base address; raw device pointers
  make this natural). Override `moe_paged()`/`dense_paged()`.
- **DeltaNet:** the chunked Tensor Core prep-pass (`deltanet_prep` analogue,
  needs TC + `kd==128`) over the sequential fallback; `Conv1dSilu` batched.

- **Exit:** MoE + qwen35 engines on the fast paths; paged models load and run.

## Phase 8 — Fused ops, chained decode, device-side sampling

Turn on the remaining `Capabilities` opt-ins, each behind its now-existing
kernel:

- **`decode_replay`** — a record-once decode tape (device-side pos/kv params),
  via **CUDA Graphs** (`cuGraph*`) which capture a launch sequence and replay it
  with near-zero host overhead — a natural fit for the replay tape and the
  strongest version of this lever on any backend. Then **`execute_chain`** +
  `max_decode_chain` for n decode steps per captured graph.
- **Device sampling:** `gpu_sample`, `argmax_rows` (MTP verify), `argmax_prob`
  (MTP draft), `eb_sample_reduce` (diffusion-gemma fused sampler) — enables MTP
  speculative decode and the DG in-graph sampler on CUDA.
- Fused `gated_rmsnorm`, `RmsNormAdd`, `QkNormRope`, `GatedActFused`,
  `embed_gather`.

- **Exit:** CUDA reaches feature parity with the Vulkan fused-op set; MTP + DG
  fast paths live; CUDA Graphs cut decode host overhead.

## Phase 9 — Perf endgame: per model × quant tuning to ≥1.0×

The closing campaign, run like `docs/perf.md`:

- Sweep the full **model × quant** matrix (`infr compare --sweep`) against
  `llama.cpp` CUDA on the target NVIDIA device(s); rank worst-gap-first. Repeat
  per compute-capability class that matters (Ampere / Ada / Hopper / Blackwell —
  the fp8/fp4 paths only exist on the newer ones).
- Profile with **Nsight Compute** (hardware counters, warp stalls) — classify
  each bottleneck by the `docs/perf.md` taxonomy and fix one lever at a time.
- Per-shape GEMM tuning on a `gemm_bench` twin; verify prefill and decode stay
  bit-identical (goldens never move).
- Record the campaign in this doc (landed slices + before/after ratios).

- **Exit (end state):** a fast kernel for **every supported model × quant
  combination**, ≥1.0× vs llama.cpp CUDA where the hardware allows (with the
  fp8/fp4 paths giving MXFP4/NVFP4 an edge on Ada/Hopper/Blackwell), correctness
  goldens from Part A still green. Update `docs/kernels.md` to add a CUDA
  column, and `docs/perf.md`/`README.md` to list CUDA as a first-class backend.

---

## Milestone checklist

- [ ] **P0** crate scaffold, CUDA FFI, `--dev cuda` selectable, remote GPU box,
      buffer round-trip
- [ ] **P1** dequant→f16 baseline: naive kernels for the full Op set, one dense
      model coherent
- [ ] **P2** all archs + blessed CUDA goldens + token-for-token vs CPU → **PART
      A (full correctness) complete**
- [ ] **P3** native per-DType quant-decode GEMV (all 24 formats)
- [ ] **P4** `__dp4a`/IMMA int8 decode + multi-row GEMV
- [ ] **P5** Tensor Core (WMMA/cuBLASLt/CUTLASS) prefill GEMM + tiering tree (+
      fp8/fp4 on new archs)
- [ ] **P6** flash + split-KV attention + KV quant
- [ ] **P7** MoE (batched/id/paged) + DeltaNet fast paths
- [ ] **P8** fused ops + CUDA-Graph decode replay + device sampling (MTP, DG)
- [ ] **P9** per model×quant perf to ≥1.0× vs llama.cpp CUDA → **PART B
      complete**

## Risks & open questions

- **No local NVIDIA hardware.** The defining constraint — the whole
  edit→validate loop is remote (cloud instance / GPU CI runner). Byte-verify
  every kernel against the CPU oracle before it runs. Budget for remote-box time
  and a push→run harness from Phase 0.
- **CUDA FFI choice.** `cudarc` (pure driver-API FFI, no build-time CUDA needed,
  active) vs hand-rolled `extern "C"` vs `cust` (heavier). Decide in P0;
  `cudarc` is the likely default for the driver + NVRTC surface.
- **Toolkit at build vs runtime.** NVRTC needs the CUDA runtime present where
  the binary runs; offline `nvcc` fatbins (P5+) shift the toolchain to build
  time but must cover every target compute capability (`-gencode` fan-out;
  PTX-JIT fallback for unknown future archs).
- **Compute-capability fan-out.** `sm_75`/`80`/`86`/`89`/`90`/`100`+ differ in
  TC shapes, fp8 (`sm_89`/`90`), and fp4 (`sm_100`). The `Capabilities` tile
  const + kernel selection must key off the queried compute capability, not
  assume one arch.
- **Matching llama.cpp CUDA is the hardest bar.** It is the most mature ggml
  backend; ≥1.0× on decode especially will be a real fight. Lean on cuBLASLt /
  CUTLASS rather than out-engineering NVIDIA's GEMM by hand.
- **Numerical parity.** Match rounding/accumulation order to keep goldens
  bit-identical where possible; where the f16/TC path legitimately differs (e.g.
  TF32 or fp8 accumulation), bless a CUDA-specific golden only after verifying
  coherence — never blind-accept a diff.
