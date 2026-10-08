# Developing infr on Windows

infr builds, tests and runs natively on Windows (x86_64, MSVC toolchain), on the
same Vulkan backend as Linux. This page gets a fresh machine from nothing to a
green `cargo test` and a model generating on the GPU, and lists the places where
Windows genuinely behaves differently.

Verified on Windows 11 with an AMD RX 7900 XTX and the Ryzen iGPU beside it, on
the AMD proprietary (Adrenalin) Vulkan driver. CI builds, lints and unit-tests
on `windows-2025` and runs one end-to-end CPU generation there
([ci-matrix.md](ci-matrix.md)); CI has no GPU, so the Vulkan path on Windows is
verified only on developer machines.

## Prerequisites

| Tool                            | Why                                                        | Install                                                      |
| ------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| Visual Studio Build Tools (C++) | the MSVC linker and C runtime `rustc` targets              | `winget install Microsoft.VisualStudio.2022.BuildTools`\*    |
| Rust (stable, MSVC host)        | the toolchain                                              | `winget install Rustlang.Rustup` (or `scoop install rustup`) |
| Vulkan SDK                      | `glslc`, which `infr-vulkan/build.rs` runs on every shader | `winget install KhronosGroup.VulkanSDK`                      |
| A GPU driver with Vulkan        | the runtime (`vulkan-1.dll` ships with the driver)         | your vendor's current driver                                 |
| Git                             |                                                            | `winget install Git.Git`                                     |
| Node.js (optional)              | `npx prettier` for the markdown docs and changelog         | `winget install OpenJS.NodeJS.LTS`                           |
| `cargo-nextest` (optional)      | the runner CI uses                                         | `cargo install cargo-nextest --locked`                       |

\* In the installer, select the **Desktop development with C++** workload
(`rustup-init` offers to install it for you if it is missing).

The Vulkan SDK is a build-time requirement only: shaders are compiled to SPIR-V
at build time and there is no prebuilt fallback. At runtime infr needs nothing
beyond the GPU driver. The SDK installer sets `VULKAN_SDK` and adds its `Bin` to
the system `PATH`, but only shells started **after** the install see that; the
build script falls back to `%VULKAN_SDK%\Bin\glslc.exe` when `glslc` is not on
`PATH`, so an old shell still builds. Check the install with:

```powershell
glslc --version          # shaderc — any 2025+ release (needs GL_EXT_integer_dot_product)
vulkaninfo --summary     # lists the GPUs the driver exposes to Vulkan
```

## Build and run

```powershell
git clone https://github.com/kryptic-sh/infr
cd infr
cargo build --release -p infr-cli
.\target\release\infr.exe devices
.\target\release\infr.exe run unsloth/Qwen3-0.6B-GGUF:Q4_K_M "What is the capital of France?"
```

The first build compiles every shader variant — several hundred `glslc` runs,
spread across all cores. Later builds recompile shaders only when something
under `crates/infr-vulkan/shaders/` changes.

Models are cached where `huggingface_hub` and llama.cpp put them:
`%USERPROFILE%\.cache\huggingface\hub`, unless `HF_HUB_CACHE`, `HF_HOME` or
`XDG_CACHE_HOME` says otherwise. It is not `%LOCALAPPDATA%` — see the
`Store::discover` doc comment for why.

## Picking a GPU

`infr devices` lists every Vulkan device with the index `--dev` takes. With no
`--dev` infr binds the first **discrete** GPU; on a desktop with an APU that is
the add-in card, not the integrated graphics. To use the iGPU:

```powershell
infr devices
#   Vulkan0: AMD Radeon RX 7900 XTX [discrete, 24.0 GiB device-local]  <- default
#   Vulkan1: AMD Radeon(TM) Graphics [integrated, 31.8 GiB device-local]
infr run --dev Vulkan1 unsloth/Qwen3-0.6B-GGUF:Q4_K_M "hello"
$env:INFR_DEV = "Vulkan1"   # or for the whole shell, tests included
```

`--dev cpu` runs the CPU reference backend and needs no GPU at all.

On an integrated GPU infr detects unified memory and budgets against system RAM,
and splits each forward into several submits to stay under the driver's hang
watchdog — on Windows that is TDR, which resets the GPU after about two seconds
of work in one submit. The startup banner says both. [igpu.md](igpu.md) has the
details.

**AMD proprietary driver:** it advertises cooperative-matrix support on every
AMD GPU, but only RDNA3 and newer have the hardware, so infr refuses it on older
parts (an RDNA2 iGPU, for instance) and logs a warning saying so. The
non-coopmat kernels run instead; the warning is expected. The same driver
truncates f32→f16 conversions on RDNA2 unless a shader asks otherwise. infr
checks this with a known-answer dispatch when it opens a device and, where the
driver truncates, declares round-to-nearest-even on every kernel that uses f16
(`infr_vulkan::spirv`); the log says so. The same driver reports 32 KB of
compute shared memory where Mesa RADV reports 64 KB on the same hardware.

## Tests

```powershell
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cargo test --workspace            # or: cargo nextest run --workspace
```

The model-gated tests find their GGUFs through the same cache lookup as `infr`,
so they run wherever the model is already downloaded and print `skip:` where it
is not. The GPU tests are `#[ignore]`d (CI has no GPU) and run on the default
device; `INFR_DEV` points `infr-vulkan`'s tests and `infr-llama`'s `cpu_backend`
suite at another:

```powershell
$env:INFR_DEV = "Vulkan1"
cargo test --release -p infr-vulkan -p infr-llama -- --include-ignored --test-threads 1
```

`cargo test` builds at `-O0` apart from the compute crates, so the model tests
are much faster with `--release`. `--test-threads 1` keeps the GPU tests from
contending for one device's memory.

On a small iGPU, do **not** enable the whole ignored suite. `bandwidth_probe`
retains large allocations, `decode_gemv_bw` retains cache-busting workloads and
unbounded repetition submissions, and `interconnect_probe` uses POSIX-only
cross-process sharing (backlog B69). Name the binaries to run instead.

### Bounded iGPU benchmarks (2026-10-09)

The following synthetic benchmarks were run headlessly and serially on native
Windows with `Vulkan1` (AMD Radeon(TM) Graphics, integrated RDNA2). Confirm the
integrated index with `infr devices` on your machine before setting `INFR_DEV`.
No model download is needed. Run from a headless terminal; process launchers
should use no-window process creation and drain both output streams.

```powershell
$env:INFR_DEV = "Vulkan1"
cargo test --release --locked --offline -p infr-vulkan --test small_m_bench --test attn_dsplit_probe --test attn_ktile_probe --test gemm_bench --test moe_id_gemv_real_dims -- --include-ignored --nocapture --test-threads=1
```

These binaries select small integrated shapes before allocating test buffers.
Their shared `bench_support` helper reports the selected device, capabilities,
built kernels, logical-operation and actual-dispatch counts, and
record/submit/wait wall time. Integrated submissions contain one logical
operation; the dispatch cap is checked before submission. The completed-batch
budget stops further batches, **not a dispatch already running**. Cold timings
are separate, and stopped or insufficient samples produce no throughput result.
One cold expert-GEMM sample stopped in the combined validation run; this is not
complete timing coverage of every case. TDR settings and production submission
policies are unchanged.

Attention retains finite/nonzero reference checks and its numerical tolerance.
MoE checks all grid formats, both projection orientations and every bank after
execution; integrated coverage is scaled parity, not real-dimension performance.
The original discrete timing budgets still fail on incomplete required samples.
`gemm_bench` uses zero inputs for performance only and does not establish
parity.

Unsupported cooperative-matrix and shared-memory-heavy variants are reported
before recording, not replaced with another kernel under the same label. On this
device that excludes direct cooperative-matrix GEMMs, non-FA attention, DeltaNet
variants and the `w128` K-tile. Those paths, discrete-device execution and other
vendors remain unverified by this Windows run.

For memory-constrained local workspace checks, use
`cargo nextest run --workspace --locked --offline --test-threads=1` and
`cargo test --workspace --locked --offline -- --test-threads=1`.
Default-parallel nextest exhausted Windows commit capacity during validation;
serial runs passed the whole workspace without changing assertions or the page
file.

## Where Windows differs

- **Ctrl-C** goes through a console control handler instead of `SIGINT`, with
  the same effect: the first press stops at the next submit boundary, drains the
  GPU and exits 130; a second press exits at once. Closing the console window
  counts as `SIGTERM` (exit 143), and Windows gives the process only a few
  seconds before killing it regardless. An idle `infr run` prompt notices Ctrl-C
  immediately only when input is an interactive console — piped input cannot be
  interrupted mid-read.
- **Symlinks.** The HF cache links each snapshot file to its blob. Creating a
  symlink needs Developer Mode or an elevated shell; without either infr falls
  back to a hard link, which works the same for reading. `infr_plat::link` has
  the details.
- **Multi-GPU sharing across processes** (`p2p.rs`, `tp_sem.rs`) exports POSIX
  file descriptors and is unavailable on Windows — see backlog B69.
- **Built for this CPU.** `.cargo/config.toml` sets `target-cpu=native` for
  x86_64 Windows as it does for Linux, so the CPU backend uses this machine's
  AVX2/AVX-512 — on a Ryzen 9 9950X3D that took Qwen3-0.6B on `--dev cpu` from
  ~380 to ~580 tok/s prefill and ~48 to ~61 tok/s decode. The binary will not
  run on an older CPU; build on the machine that runs it.
- **CPU threads.** With no `-t`, infr runs one thread per physical core rather
  than per logical processor: the CPU backend's spin pool loses badly when two
  workers share a core's hyperthreads (about half the decode speed on the
  9950X3D). `-t N` or `RAYON_NUM_THREADS` overrides it. Other platforms still
  default to every logical processor (backlog B76).
- **CPU golden hashes.** The long `cpu_golden_*` cases produce different (still
  coherent) text on Windows than on Linux on the same CPU, deterministically, so
  those cases carry a Windows hash (`per_os` in
  `infr-llama/tests/cpu_backend.rs`). Re-bless with `INFR_BLESS=1` on both
  platforms when a golden legitimately moves.

## Small-model iGPU baseline (2026-10-08)

Measured headlessly on `Vulkan1`, the Ryzen RDNA2 iGPU with AMD's proprietary
Windows driver, using the cached `unsloth/Qwen3-0.6B-GGUF:Q4_K_M` model. The
infr baseline was `af2355a`, built with Rust 1.98.1; the comparison used
llama.cpp's `b11491` Windows Vulkan release. These are baseline measurements,
not an optimization result.

Both tools used `-r 3`, depth zero and f16 KV. infr used `--ctx 2048` and its
default `ubatch=128`, `submit_cap=128`; llama-bench used `-ub 128 -fa on`. Each
leg ran infr prefill/decode, then llama-bench prefill/decode, serially on the
same iGPU. Prefill used `-p 128 -n 0`; decode used `-p 0 -n 64`.

| Workload | Tool        | Leg | Individual repetitions (tok/s) |
| -------- | ----------- | --- | ------------------------------ |
| Prefill  | infr        | 1   | 356.57, 356.41, 356.23         |
| Prefill  | llama-bench | 1   | 434.433, 433.707, 435.839      |
| Prefill  | infr        | 2   | 355.83, 355.44, 355.47         |
| Prefill  | llama-bench | 2   | 434.356, 434.082, 433.67       |
| Decode   | infr        | 1   | 53.86, 53.80, 53.91            |
| Decode   | llama-bench | 1   | 56.057, 55.7353, 56.135        |
| Decode   | infr        | 2   | 53.82, 53.77, 53.78            |
| Decode   | llama-bench | 2   | 55.7126, 56.6712, 56.5547      |

A separate `INFR_PROF_OPS=1` prefill run attributed 61.16% of device time to
`native_gemm_mmq_q4k_streamed`, 14.48% to `native_gemm_mmq_q6k_streamed`, and
17.67% to `attn_nc_fa_hd128`. The packed-Q4_K follow-up below targets the
largest measured cost; backlog B77 records remaining coverage gaps.

After matching CI's Rust 1.99 toolchain, workspace formatting, Clippy, tests
with `INFR_DEV=Vulkan1 --test-threads 1`, the release CLI build, and the
Apple-target cross-lint passed. The explicitly enabled GPU suites
`nc_gemm_parity`, `pager_mmq_parity`, and `mmq_wide_bn_determinism` also passed
on the iGPU. Cached Qwen3-0.6B, Gemma-3-1B, and Qwen3.5-0.8B Q4_K_M models
answered the capital-of-France smoke prompt correctly. This does not cover the
entire ignored GPU suite or other GPU vendors.

## Packed Q4_K follow-up (2026-10-08)

The `native_gemm_mmq_q4k.comp` packed-word unpack change in `67c31e9` was
compared with baseline `7c79a95`, both built with Rust 1.99.0 and the same
shader compiler. Runs were headless and serial on `Vulkan1`, the AMD RDNA2
integrated GPU, using the cached Qwen3-0.6B Q4_K_M GGUF pathname. No model was
downloaded. Profiling was disabled.

Each leg ran prefill then decode, alternating baseline and candidate. Both used
`--ctx 2048 -u 128 -d 0 -r 3`, f16 KV and the unchanged submit cap. Prefill used
`-p 128 -n 0`; decode used `-p 0 -n 64`. Values below are the benchmark's
printed throughput and repetition range, in tok/s.

| Leg         | Prefill | Prefill range | Decode | Decode range |
| ----------- | ------- | ------------- | ------ | ------------ |
| Baseline 1  | 354.3   | 353.6–354.8   | 53.7   | 53.6–53.8    |
| Candidate 1 | 362.3   | 361.9–362.9   | 54.0   | 53.7–54.4    |
| Baseline 2  | 356.9   | 355.1–357.9   | 54.0   | 53.6–54.2    |
| Candidate 2 | 362.7   | 361.9–363.6   | 53.4   | 53.3–53.5    |

Prefill improved in both comparisons; decode results were mixed. This is not a
cross-model or cross-driver performance claim. The optimization changes only
packed quant reads, not scales, accumulation order, tile sizes or watchdog
budgets.

`nc_gemm_parity` now compares bound SSBO, offset resident-BDA and paged Q4_K
weights after actual eviction/reload against every output bit of a host-checked
dense result. The selected `nc_gemm_parity`, `pager_mmq_parity`,
`weight_addr_parity` and `mmq_wide_bn_determinism` suites passed serially on the
iGPU, as did the workspace gate, workspace tests/doctests and warning-denied
intra-doc-link build. The full ignored GPU suite and other vendors were not run;
see backlog B77.

The Windows Job Object headroom tests also passed under native AddressSanitizer
using the installed nightly toolchain and MSVC ASan runtime. Application
Verifier could not run without elevation; ASan is not equivalent coverage of
Win32 API contracts.

## Troubleshooting

- **`failed to run glslc`** — the Vulkan SDK is missing, or neither `PATH` nor
  `VULKAN_SDK` reaches it. Open a new shell after installing the SDK.
- **`LNK1104: cannot open file 'target\release\infr.exe'`** or
  `Access is denied. (os error 5)` during a build — an `infr.exe` from that
  target directory is still running (a `serve` left in another window). Stop it
  and build again.
- **`linker stdout: LINK : warning LNK4098: defaultlib 'LIBCMT' conflicts`** — a
  C/C++ dependency is built against the static C runtime. Harmless, and not from
  infr's own code; see backlog.
- **No Vulkan devices** — `vulkaninfo --summary` shows what the driver exposes;
  if it lists nothing, the driver install is the problem, not infr.
