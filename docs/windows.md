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

On a small iGPU, leave out `infr-vulkan`'s benchmark and probe binaries
(`attn_dsplit_probe`, `attn_ktile_probe`, `bandwidth_probe`, `decode_gemv_bw`,
`gemm_bench`, `small_m_bench`, `interconnect_probe`, `moe_id_gemv_real_dims`):
they are sized for a discrete card, some run a single submit past the ~2 s TDR
limit and lose the device, and a lost iGPU drops out of enumeration for a while,
failing every test after it. Name the correctness binaries with `--lib` and
`--test <name>` instead. `interconnect_probe` fails everywhere on Windows: the
cross-process sharing it probes is POSIX-only (backlog B69).

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
