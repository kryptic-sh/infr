# sycl-plan.md — a native Intel oneAPI/SYCL backend for infr

Plan for adding `infr-sycl`, a compute backend targeting Intel GPUs through the
**oneAPI** stack (SYCL/DPC++ programming model, **Level Zero** runtime, oneMKL/
oneDNN libraries, XMX matrix engines). It is the Intel sibling of the NVIDIA
(`docs/cuda-plan.md`) and AMD (`docs/rocm-plan.md`) backend plans, ordered the
same way — **correctness first, performance second**: reach a fully correct
backend (every model and every quant format infr supports, token-for-token with
the CPU reference) before any perf lever, then climb from naive kernels to a
fast kernel for every supported model × quant combination.

> Status: **PLAN ONLY — nothing built yet.** This doc is the roadmap and the
> design record. Update the phase checkboxes and commit refs as slices land, the
> way `docs/igpu.md` and `docs/cpu-perf.md` track their campaigns.

The three vendor stacks map cleanly onto each other — CUDA ↔ ROCm/HIP ↔
oneAPI-SYCL, over runtimes CUDA-driver ↔ HIP ↔ **Level Zero**, with GEMM libs
cuBLASLt ↔ rocBLAS ↔ **oneMKL/oneDNN**, and matrix engines Tensor Cores ↔
WMMA/MFMA ↔ **XMX**. This plan is a deliberate sibling of the other two;
wherever it says "same as the CUDA/ROCm plan", it means the identical backend
seam with the oneAPI toolchain swapped in.

## Why an Intel backend (infr already runs on Arc via Vulkan)

The Vulkan backend already runs on Intel Arc — including the **XMX `8x8x16`
cooperative-matrix path** (opt-in `INFR_CM_8X8`, `caps.f16_coopmat_8x8()`), the
non-coopmat `nc_tier` (`nc_fma`/`nc_mmq`) for Arc/ANV, and validated A770 decode
parity (`docs/perf.md`; the memory note _Intel Arc kernel plan_ — decode parity
on A770, prefill the remaining gap). So infr already carries hard-won Intel-GPU
knowledge. A native oneAPI backend is still worth building:

- **XMX / oneDNN directly, past the Vulkan coopmat ceiling.** Intel's matrix
  engines (**XMX** on Arc Alchemist/Battlemage and Data Center GPU Max / Ponte
  Vecchio) are reachable natively through SYCL `joint_matrix`, the `SPV_INTEL`
  subgroup-matrix SPIR-V extension, and the tuned **oneMKL/oneDNN** GEMM/DPAS
  kernels — a second, independent route to close the Vulkan-on-Arc **prefill**
  gap that is the known weak spot.
- **The Intel reference target.** `llama.cpp` built with `-DGGML_SYCL=ON` (and
  Intel's IPEX-LLM) is the Intel-GPU baseline; matching it on the same device is
  the clean ratio. infr's own Vulkan-on-Arc numbers are a second oracle.
- **oneAPI tooling.** VTune / Advisor and `unitrace` give XMX-occupancy and
  memory attribution beyond the Vulkan timestamp path.
- **Cross-platform.** oneAPI runs on Linux and Windows.

Non-goal: replacing Vulkan. oneAPI is additive; Vulkan stays the portable
default and the Intel iGPU/older-driver fallback.

## Hardware caveat — no Intel GPU on the dev box

Like the CUDA plan (and unlike ROCm, which the dev box's 7900 XTX validates
directly), **there is no Intel GPU on the dev box** (`GPU0` = RX 7900 XTX,
`GPU1` = AMD iGPU — `docs/igpu.md`; the CPU is a Ryzen, so there is no Intel
iGPU either). Validation therefore needs Intel hardware that is not local:

- **Primary — a remote / cloud Intel GPU** (Arc A770 Alchemist, Arc B-series
  Battlemage, or a Data Center GPU Max via the Intel Tiber Developer Cloud). The
  workflow mirrors the **blind Metal** campaign (`docs/kernels.md`): author
  kernels against the CPU oracle locally, byte-verify the decode math, push, run
  the ignored parity suite remotely.
- **Secondary — a GPU CI runner** with an Intel GPU running the ignored parity/
  goldens suite (the twin of `test-macos`).
- **Partial local proxy.** infr's **Vulkan `nc_tier` + `INFR_CM_8X8`** paths
  emulate the Arc kernel tier's _structure_ (no coopmat / XMX-8x8) and can be
  exercised on the dev box's AMD GPU under `INFR_NO_COOPMAT=1` — useful for
  shaking out the split-op fallback logic before remote time, but NOT a
  substitute for real XMX validation.

As with CUDA: byte-verify every kernel against the CPU reference before it runs
on a GPU, because the edit→validate loop is remote.

## What a backend actually has to do (the seam is already generic)

Identical to the other backend plans — everything above the device is
backend-agnostic: the `Graph`/`Op` IR, the seam runner
(`generate_dense_backend`), the chat REPL, KV-cache and MoE routing policy. A
backend is exactly:

1. Implement **`infr_core::backend::Backend`**
   (`crates/infr-core/src/backend.rs:464`) plus `Buffer` (`:374`), `Plan`
   (`:397`), `ProgressScope` (`:372`).
2. Fill in **`Capabilities`** (`backend.rs:42`) honestly — the runner degrades
   to split-op fallbacks for anything reported unsupported.
3. Execute every **`Op`** variant (`crates/infr-core/src/graph.rs`, flat list in
   `Op::kind` `:554`).

`infr-metal`/`infr-vulkan` are the from-scratch templates to mirror; the SYCL
backend leans especially on **Vulkan**, because both consume **SPIR-V** (see the
kernel strategy below).

### The `Backend` trait surface (required unless noted)

| method                                   | ref              | correctness-phase behavior                                     |
| ---------------------------------------- | ---------------- | -------------------------------------------------------------- |
| `name(&self) -> &str`                    | `backend.rs:465` | `"sycl"`                                                       |
| `capabilities(&self) -> Capabilities`    | `:466`           | conservative set (below)                                       |
| `alloc(bytes, usage)`                    | `:474`           | `zeMemAllocDevice` + **zero-init** (memset) — calloc contract  |
| `upload(dst, &[u8])`                     | `:484`           | `zeCommandListAppendMemoryCopy` H2D (+ sync)                   |
| `download(src, &mut [u8])`               | `:485`           | memory-copy D2H                                                |
| `compile(&Graph)`                        | `:517`           | `GraphPlan::boxed(graph)` (clone; work happens in execute)     |
| `execute(plan, bindings)`                | `:518`           | walk graph → append kernel launches to a command list → sync   |
| `sync()`                                 | `:552`           | `zeCommandQueueSynchronize`                                    |
| `alloc_uninit` (default → `alloc`)       | `:481`           | skip zero-init once proven; debug-poison                       |
| `weight_progress` (default no-op)        | `:498`           | drive the `indicatif` bar from inside `alloc` (Metal does)     |
| `copy_buffer` (default host-bounce)      | `:508`           | override with a device-to-device memory copy (KV prefix-share) |
| `execute_chain` (default `Ok(None)`)     | `:526`           | perf phase: n decode steps in one immediate command list       |
| `max_decode_chain` (default `MAX`)       | `:548`           | perf phase clamp                                               |
| `moe_paged` / `dense_paged` (`false`)    | `:564/:577`      | perf phase (VRAM streaming)                                    |
| `eb_sample_reduce` (default `Ok(false)`) | `:595`           | perf phase (fused diffusion sampler)                           |

No `warmup` on the trait — it is `ChatModel::warmup` (`chat/mod.rs`), inherited
by the SYCL chat wrapper for free.

### `Capabilities` — the correctness dial

Same rule: **advertise the minimum, let the runner split everything**
(`backend.rs:42`). Start with:

- `name = "sycl"`, `f16 = true`, `i8 = false`, `i8_dot = false`, all
  `coopmat_* = None`.
- `vendor_intel = true` — infr's compiler/runner already special-cases Intel in
  a few shader picks; set it so the same routing applies.
- `integrated = false` for a discrete Arc / GPU Max. (Intel iGPUs exist and are
  a watchdog-TDR class like the AMD iGPU, `docs/igpu.md`; treat them as a later
  best-effort target and set `integrated = true` there so the submit-splitter
  math at `backend.rs:223/271` engages.)
- `compute_units` = Xe-core count (cosmetic until perf).
- Every fused-op opt-in **false**: `decode_replay`, `combined_gu`,
  `embed_gather`, `gpu_sample`, `argmax_rows`, `argmax_prob`, `gated_rmsnorm`,
  `kv_swa_ring`.
- `buffer_device_address` — Level Zero device pointers are raw addresses; enable
  when a kernel wants pointer-fed args.

Each field flips to `true` only once the matching native kernel exists and
passes parity. In the perf phase, `coopmat_f16`/`coopmat_f16_8x8` get the real
**XMX** tile (Intel's subgroup-matrix is `8x8x16` / `16x16x16` depending on the
part) — the same `f16_coopmat_8x8()` accessor the Vulkan Arc path already uses
(`backend.rs:317-343`).

## Kernel authoring strategy (the key structural difference)

CUDA and ROCm compile kernel **strings at runtime** (NVRTC / hiprtc). SYCL is
single-source C++ normally compiled **ahead-of-time** by the DPC++ compiler
(`icpx -fsycl`), which is awkward to drive from a Rust crate. The cleaner fit
for infr — and the reason this backend is closer to Vulkan than to CUDA — is the
substrate SYCL itself sits on:

- **(A, correctness — recommended) Level Zero + Kernel-flavor SPIR-V, launched
  from Rust.** Level Zero loads a **SPIR-V** module (`zeModuleCreate`,
  `ZE_MODULE_FORMAT_IL_SPIRV`), fetches a kernel by name (`zeKernelCreate`), and
  dispatches it (`zeCommandListAppendLaunchKernel`) — **exactly the load-module
  / launch-by-name pattern the Vulkan and Metal backends already use**,
  driveable purely from Rust FFI with **no C++ single-source translation unit
  and no `icpx` at build time**. Kernels are authored in OpenCL C (or SYCL
  free-function kernels) and compiled to Kernel-flavor SPIR-V by `clang`/`icx`
  in a `build.rs` (analogous to Vulkan's `glslc` `build.rs`). Quant codebook
  LUTs are emitted from `infr_core::iquant_grids` (the shared statics —
  bit-exact by construction). This carries the entire correctness phase with a
  Rust-native runtime.

  Caveat vs Vulkan: Vulkan SPIR-V and Level Zero/OpenCL SPIR-V are **different
  execution environments** (Vulkan vs Kernel capability set), so the existing
  `.comp`→SPIR-V blobs are **not** drop-in reusable — but the kernel _logic_
  (quant decode, norms, attention math) ports directly, and the `build.rs`
  SPIR-V-emission machinery transfers.

- **(B, perf — the oneAPI libraries) a SYCL/oneDNN C-ABI shim.** For the heavy
  GEMM and XMX paths, add a small C++ translation unit compiled by `icpx -fsycl`
  in `build.rs`, exposing `extern "C"` launch functions that call **oneMKL /
  oneDNN** (tuned GEMM, DPAS int8) or hand-written SYCL `joint_matrix` kernels,
  FFI'd from Rust. Introduced only in the perf phases, so the correctness build
  needs no C++ toolchain.

- **(C, XMX inside SPIR-V) the `SPV_INTEL` subgroup-matrix extension.** XMX ops
  can be emitted **directly in the Level Zero SPIR-V kernels** via
  `cl_intel_subgroup_matrix_multiply_accumulate` / `SPV_INTEL_joint_matrix` —
  the Kernel-flavor analogue of the Vulkan cooperative-matrix path infr already
  runs on Arc. This lets the matrix GEMM stay on route (A) without the oneDNN
  shim, the same way the Vulkan backend does XMX through coopmat.

Recommendation: **(A)** all through Part A, **(C)** for the XMX GEMM in Part B,
with **(B)/oneDNN** kept as a measured alternative for the wide dense tails (the
`docs/perf.md` "measure the library vs the hand kernel per shape" discipline).

## Crate + workspace wiring

Mirror the Metal/ROCm/CUDA gating so the workspace still builds everywhere:

- New crate `crates/infr-sycl`, added to `Cargo.toml` `members`.
- Gate behind a cargo feature (**`sycl`**) — oneAPI runs on Linux and Windows,
  so the gate is the feature (Level Zero loader present), not the OS. Without it
  the crate is a no-op empty lib with zero oneAPI deps (as `infr-metal` is off
  macOS, `crates/infr-metal/src/lib.rs:4`).
- Deps: `infr-core`, `infr-gguf`, `half`, `bytemuck`, `indicatif`; the Level
  Zero FFI behind the feature (hand-written `extern "C"` to `libze_loader`, or a
  vetted Level Zero bindings crate). The optional oneDNN shim (Part B) adds an
  `icpx` `build.rs` step behind a further sub-feature. Dev-dep `infr-cpu` as the
  parity oracle.

### Selection / registration (the `INFR_DEV` grammar)

Add a `sycl` token to the one grammar shared by `--dev` and `INFR_DEV`
(`crates/infr-cli/src/main.rs`):

1. `enum Backend { Vulkan(Option<String>), Metal, Cpu, Sycl }` (`main.rs:62`).
2. `parse_dev_spec` (`:99`) — accept `sycl` (optionally `sycl:N` to pin a
   device).
3. `DeviceOpts::resolve` / `selected_backend` (`:187/:134`) — publish/read it.
4. `build_chat_model` (`:841`) and the diffusion-gemma 3-arm match (`:861`) —
   add a `Backend::Sycl` arm building a `SyclSeamChat` (twin of
   `metal_chat_model`, `:1392`). Without the `sycl` feature, fall back with a
   clear error.

### Chat + session plumbing

- `crates/infr-llama/src/chat/sycl.rs` — `SyclSeamChat: ChatModel`
  (`chat/mod.rs:38`), twin of `chat/metal.rs`. Re-export in `chat/mod.rs`.
- `DenseSyclSession` in `crates/infr-llama/src/seam/model.rs` (twin of
  `DenseMetalSession`, `:272`) owning a `SyclBackend`; a
  `SeamModel::sycl_session` calling `SyclBackend::new()` (mirror
  `metal_session`, `:1194`).
- `sycl_upload_bind` in `crates/infr-llama/src/seam/mod.rs` (twin of
  `metal_upload_bind`, `:97`): alloc device buffer + upload the **raw native
  dtype** bytes, lazy-dequant on first use. Then drive the existing
  `generate_dense_backend` runner unchanged.

### The `be.name()` capability gates to thread `"sycl"` into

Same set as the other plans — the seam decisions that branch on the backend name
string, in `crates/infr-llama/src/seam/runner.rs`: `kv_q8_backend` (`:432`),
`kv_turbo_ok` (`:435`), `blk_ok` (`:439`), `dense_ok` (`:442`), the Metal-only
`k_fmt != v_fmt` guard (`:477`), and the `gpu_sc` gate (`:3030`). Add `"sycl"`
as each feature lands — never before its kernel exists.

---

# PART A — CORRECTNESS (all models, all quants, token-for-token)

Goal: `INFR_DEV=sycl infr run <any supported model>:<any supported quant>`
loads, generates coherent text, and matches the CPU reference token-for-token —
with **naive kernels only**. No perf work. Validated on a remote Intel GPU.

## Phase 0 — Scaffolding: an empty backend that builds and is selectable

- Create `crates/infr-sycl` (feature-gated empty lib), wire it into the
  workspace and the `INFR_DEV=sycl` selection chain (all sites above).
- Evaluate the Level Zero FFI surface: `zeInit`, `zeDriverGet`, `zeDeviceGet`,
  `zeContextCreate`, `zeMemAllocDevice/Free`, `zeCommandListCreateImmediate`,
  `zeCommandListAppendMemoryCopy/LaunchKernel`, `zeCommandQueueSynchronize`,
  `zeModuleCreate` (SPIR-V), `zeKernelCreate`, `zeDeviceGetProperties`
  (Xe-core/subgroup info). Decide bindings crate vs hand-rolled.
- Stand up the `build.rs` OpenCL-C/SYCL → Kernel-flavor SPIR-V compile step
  (route A) and confirm a trivial kernel loads + launches on the remote Arc.
- Provision the **remote Intel GPU box** and a push→run loop.
- `SyclBackend::new()` enumerates the device, builds a context + immediate
  command list, reports `capabilities()`.
- **Gate:** `cargo build -p infr-sycl --features sycl`; workspace `cargo build`
  green everywhere without the feature. `--dev sycl` recognized, clean error
  when absent. First parity test: `upload`→`download` byte-identity on the
  remote GPU.
- **Exit:** the backend constructs, allocs/uploads/downloads/`sync`s, and
  launches one SPIR-V kernel on real Intel hardware.

## Phase 1 — Correctness baseline: dequant→f16 everything

Same day-one-coverage trick as the sibling plans: never decode a quant natively
yet. In `sycl_upload_bind`, keep raw bytes; in `Linear`/`EmbedGather`/expert
paths, dequant the block to `f16` on first touch via `infr_core::dequant`, cache
it, and run a **single naive f16 GEMV/GEMM SPIR-V kernel**. All 24 weight quant
formats + floats reduce to that one matmul → full coverage immediately
(`docs/kernels.md`).

Kernels this phase (all naive, one variant each): `linear_f16` GEMV/GEMM; the
norm/elementwise/rope set (`RmsNorm`, `RmsNormAdd`, `Softmax`, `QkNorm`, `Rope`,
`QkNormRope`, `GatedRmsNorm`, `Add`, `AddBias`, `Scale`, `MulVec`, `Softcap`,
`GatedAct`, `Copy`, `CopyStrided`, `EmbedGather`, `Argmax`, `Sample`); `WriteKv`

- scalar `Attention` (Causal + SlidingWindow, f16/f32 KV only); `MoeFfn` unfused
  split path + `MoeSharedExpertAdd`; `Conv1dSilu` + sequential `DeltaNet`
  (persistent `S` state, in place, **surviving across `execute` calls**).

Advertise nothing fused, so the runner emits the split ops.

- **Gate:** per-op parity via `crates/infr-llama/tests/seam_op_parity.rs`
  (one-op agnostic `Graph`, CPU vs SYCL, incl.
  `state_persists_across_executes`), run `--include-ignored` on the remote box.
- **Exit:** a small dense model (Qwen3-0.6B Q4_K_M, gemma-3-1b) generates
  coherent text on `--dev sycl`.

## Phase 2 — Op completeness, all architectures, blessed goldens

Extend until **every** model family runs and goldens lock:

- Cover every arch the CPU/Vulkan backends do: llama, qwen2/2.5, qwen3
  dense+MoE, gemma3, gemma4 dense/E2B/26B-A4B MoE, qwen3.5/3.6 (`qwen35`
  DeltaNet) + `qwen35moe`, diffusion-gemma, llama4 Scout, plus the fine-tunes
  (Ornith, Ternary-Bonsai).
- KV cache: f16 + f32 correct.
- Add the **SYCL `gpu_seam` golden**: a backend-specific FNV-1a hash blessed
  with `INFR_BLESS=1` (each backend locks its own — `cpu_backend.rs:48`), plus a
  token-for-token `seam_sycl_matches_cpu` sweep over the quant families (twin of
  `seam_vulkan_matches_cpu`, `cpu_backend.rs:380+`). Wire the `linear.rs` drift
  guards.
- Thread `"sycl"` into the `blk_ok`/`dense_ok` KV gates as formats gain correct
  read/write.

- **Gate:** full `gpu_seam` suite green on the remote box; token-for-token with
  the CPU oracle on every arch × a representative quant.
- **Exit of PART A:** every supported model × every supported quant produces
  correct, coherent output on the SYCL backend, matching the reference. **Zero
  perf claims.**

## CI / validation strategy (applies across the plan)

Same shape as the CUDA plan (remote hardware), with an Intel-specific local
proxy:

- **Byte-verify every kernel against the CPU reference before it runs on a
  GPU.**
- A **`sycl-check`** CI job that `cargo check`s the workspace **without** the
  `sycl` feature (empty-lib path) to catch `Op`-signature / match-arm drift —
  the analogue of the existing `metal-check` guard (`ci.yml`).
- A **`test-sycl`** job on an Intel-GPU runner running the ignored
  parity/goldens suite (twin of `test-macos`) — the eventual gate.
- **Local proxy:** run the split-op fallback logic under the Vulkan `nc_tier`
  (`INFR_NO_COOPMAT=1`) on the dev box's AMD GPU to shake out routing before
  remote time — structural only, not XMX numerical validation.

---

# PART B — PERFORMANCE (fast kernel per model × quant)

Goal: climb from the naive baseline to a fast kernel for every supported model ×
quant, benchmarked against `llama.cpp` SYCL and infr's own Vulkan-on-Arc numbers
on the same Intel device — targeting **≥1.0×** on prefill and decode for every
model×quant, without moving a golden. Follow the `docs/perf.md` loop. The known
Intel weak spot is **prefill** (decode already reaches parity on A770 via
Vulkan), so prefill/XMX is the first lever after native decode.

The Intel levers, mapped to the Vulkan tiering (`adapter.rs:1340-1560`):

| Vulkan tier                   | Intel oneAPI analogue                                                   |
| ----------------------------- | ----------------------------------------------------------------------- |
| f16 coopmat GEMM (prefill)    | **XMX** via `joint_matrix` / `SPV_INTEL` subgroup-matrix; or **oneDNN** |
| f16 XMX-8x8 coopmat (Arc)     | Intel subgroup-matrix `8x8x16` — the same `INFR_CM_8X8` shape, native   |
| dp4a `mmq` int8 (decode)      | Intel **DPAS** int8 / `dp4a` builtin                                    |
| scalar dequant-in-shader GEMV | plain SPIR-V GEMV (the Phase-1 baseline)                                |
| non-coopmat SLM warptile      | `nc_fma`-equivalent SLM (shared-local-memory) warptile                  |

## Phase 3 — Native quant decode (drop the host dequant)

In-kernel block decode GEMV per format, so decode streams the compact quant
bytes (the dominant decode lever). One kernel family, per-`DType`
`#define`/spec-const variants (mirror Vulkan's `-DFMT_<QUANT>` build variants),
driven off `ALL_DTYPES` (`linear.rs:373`) and the IQ grid LUTs. Each format
lands with a GEMV parity test vs the CPU `vec_dot` + `dequant_block`. Covers all
24 weight formats.

- **Exit:** decode reads native quant bytes; per-format GEMV parity green.

## Phase 4 — Integer-dot decode (DPAS / dp4a)

Int8-activation path: quantize the activation row to int8 once, integer dot
against native weight codes with Intel **`dp4a`** / **DPAS** int8. Flip
`i8`/`i8_dot`; derive coverage from the shared `MOE_MMQ_DTYPES` SSOT
(`tensor.rs`). Add the multi-row (`m=2..16`) GEMV variant.

- **Exit:** int8 decode parity-clean for the mmq dtype set; measured decode win.

## Phase 5 — XMX GEMM (prefill) + the tiering decision tree — the priority lever

The big Intel lever (prefill is the gap). Port `adapter.rs`'s decision tree:

- **XMX `joint_matrix` / `SPV_INTEL` subgroup-matrix GEMM** with in-shader
  dequant (`matmul_native` analogue) + f16/bf16 variants. Populate
  `coopmat_f16`/`coopmat_f16_8x8` with the real XMX tile (`8x8x16` on Arc). This
  is the direct upgrade over the Vulkan coopmat path infr already runs on Arc.
- **DPAS int8 `mmq`** batched prefill for the mmq dtypes.
- Evaluate **oneMKL/oneDNN** (via the icpx C-ABI shim, route B) for the wide
  f16/bf16/int8 tiles vs the hand XMX kernel — pick per-shape on a `gemm_bench`
  twin (the measure-the-library discipline from `docs/perf.md`; the Vulkan
  campaign found occupancy, not math, was the ceiling — expect the SLM-budget
  class of win here too).

- **Exit:** prefill on XMX; per-shape throughput near the oneDNN ceiling; the
  Vulkan-on-Arc prefill gap closed; goldens unmoved.

## Phase 6 — Fast attention + KV quantization

- **Flash attention** (fused, `hd==128`, Causal, XMX) — the
  `attention_prefill_flash` analogue; **split-KV / flash-decoding** for decode.
- **KV quant:** Q8_0 planar + block formats via dequant→f16 prepass first, then
  dequant-in-attention for the cheap formats (Vulkan's `INFR_FLASH_DEQUANT`
  set); TurboQuant via the shared prepass. Thread `"sycl"` into
  `kv_q8_backend`/`kv_turbo_ok`/`blk_ok`; enable `kv_swa_ring` once correct.

- **Exit:** attention off the scalar fallback; KV-quant coverage matches the
  other GPU backends; decode-at-depth competitive.

## Phase 7 — MoE (batched, id-GEMV, paged) + DeltaNet fast paths

- **MoE:** GPU-side top-k routing (router GEMV → bucket count/scan/scatter, no
  host readback — the class-5 host-in-the-loop fix); resident small-`m`
  **id-indexed GEMVs** and batched **DPAS `mmq` experts** for larger `m` (gated
  on `moe_mmq_ok`). Enable `combined_gu`.
- **Paged experts** for models exceeding VRAM (llama4 Scout, big MoE): a Level
  Zero `GpuPager` analogue (LUT hop to an arena base address). Override
  `moe_paged()`/`dense_paged()`.
- **DeltaNet:** the chunked XMX prep-pass (`deltanet_prep` analogue, needs
  subgroup-matrix + `kd==128`) over the sequential fallback; `Conv1dSilu`
  batched.

- **Exit:** MoE + qwen35 engines on the fast paths; paged models load and run.

## Phase 8 — Fused ops, chained decode, device-side sampling

Turn on the remaining `Capabilities` opt-ins, each behind its now-existing
kernel:

- **`decode_replay`** — a record-once decode tape (device-side pos/kv params).
  Use a **Level Zero immediate command list** (or `zeCommandListImmediateAppend`
  replay) to cut per-token host overhead. Then **`execute_chain`** +
  `max_decode_chain` for n decode steps per submit.
- **Device sampling:** `gpu_sample`, `argmax_rows` (MTP verify), `argmax_prob`
  (MTP draft), `eb_sample_reduce` (diffusion-gemma fused sampler) — enables MTP
  speculative decode and the DG in-graph sampler on SYCL.
- Fused `gated_rmsnorm`, `RmsNormAdd`, `QkNormRope`, `GatedActFused`,
  `embed_gather`.

- **Exit:** the SYCL backend reaches feature parity with the Vulkan fused-op
  set; MTP + DG fast paths live.

## Phase 9 — Perf endgame: per model × quant tuning to ≥1.0×

The closing campaign, run like `docs/perf.md`:

- Sweep the full **model × quant** matrix (`infr compare --sweep`) against
  `llama.cpp` SYCL and infr's Vulkan-on-Arc numbers on the target Intel device;
  rank worst-gap-first (prefill first, per the known gap).
- Profile with **VTune / Advisor / unitrace** (XMX occupancy, memory bandwidth)
  — classify each bottleneck by the `docs/perf.md` taxonomy, fix one lever at a
  time.
- Per-shape GEMM tuning on a `gemm_bench` twin; verify prefill and decode stay
  bit-identical (goldens never move).
- Record the campaign in this doc (landed slices + before/after ratios).

- **Exit (end state):** a fast kernel for **every supported model × quant
  combination**, ≥1.0× vs llama.cpp SYCL where the hardware allows, correctness
  goldens from Part A still green. Update `docs/kernels.md` to add a SYCL
  column, and `docs/perf.md`/`README.md` to list Intel oneAPI as a first-class
  backend.

---

## Milestone checklist

- [ ] **P0** crate scaffold, Level Zero FFI, SPIR-V `build.rs`, `--dev sycl`
      selectable, remote Intel box, buffer round-trip + one kernel launch
- [ ] **P1** dequant→f16 baseline: naive kernels for the full Op set, one dense
      model coherent
- [ ] **P2** all archs + blessed SYCL goldens + token-for-token vs CPU → **PART
      A (full correctness) complete**
- [ ] **P3** native per-DType quant-decode GEMV (all 24 formats)
- [ ] **P4** DPAS/dp4a int8 decode + multi-row GEMV
- [ ] **P5** XMX (`joint_matrix`/oneDNN) prefill GEMM + tiering tree — closes
      the prefill gap
- [ ] **P6** flash + split-KV attention + KV quant
- [ ] **P7** MoE (batched/id/paged) + DeltaNet fast paths
- [ ] **P8** fused ops + command-list decode replay + device sampling (MTP, DG)
- [ ] **P9** per model×quant perf to ≥1.0× vs llama.cpp SYCL → **PART B
      complete**

## Risks & open questions

- **No local Intel GPU.** The whole edit→validate loop is remote (cloud Intel
  GPU / GPU CI runner); the Vulkan `nc_tier`/`INFR_CM_8X8` proxy on the AMD box
  only validates _structure_, not XMX numerics. Budget remote time from Phase 0.
- **SPIR-V flavor mismatch.** Vulkan `.comp` SPIR-V is not drop-in on Level Zero
  (Vulkan vs Kernel execution environment); the kernel _logic_ ports but the
  blobs must be re-emitted from OpenCL-C/SYCL. Confirm the `build.rs` clang/icx
  → Kernel-SPIR-V path early (Phase 0).
- **Level Zero FFI choice.** Hand-rolled `extern "C"` to `libze_loader` (thin,
  zero-dep) vs a bindings crate. Decide in P0; hand-rolled is viable given the
  small surface.
- **When to bring in the `icpx`/oneDNN shim.** Route A (Level Zero + SPIR-V)
  keeps the correctness build C++-free; the oneMKL/oneDNN shim (route B) adds an
  `icpx -fsycl` `build.rs` step and a heavier toolchain requirement — introduce
  it only if hand XMX kernels (route C) lose to the library on the `gemm_bench`.
- **XMX shape / part variance.** Alchemist, Battlemage, and GPU Max differ in
  subgroup-matrix shapes and int8/bf16 support; the `Capabilities` tile const +
  kernel selection must key off the queried device, not assume one part.
- **Numerical parity.** Match rounding/accumulation order to keep goldens
  bit-identical where possible; where the f16/XMX path legitimately differs,
  bless a SYCL-specific golden only after verifying coherence — never
  blind-accept a diff.
