# CI platform matrix

**Landed.** What CI does now, and the two things that deliberately stayed
single-platform.

## The matrix

`clippy` and `test` each run on `ubuntu-26.04`, `macos-15` and `windows-2025`
with `fail-fast: false`, so one platform's failure never hides another's.

Every runner needs `glslc`: `infr-vulkan/build.rs` shells out to it once per
compute shader and has no prebuilt-SPIR-V fallback, so it is a hard build
requirement rather than a Vulkan-runtime concern. The per-platform recipe (apt,
Homebrew, and the Vulkan SDK on Windows, whose installer leaves its `Bin`
directory off `PATH`) lives in `.github/actions/install-glslc`, which ends by
running `glslc --version` — an install step that silently no-ops would otherwise
surface minutes later as a `build.rs` panic on whichever job reached
`infr-vulkan` first.

## Single-platform jobs, and why

- **`cpu-goldens`** asserts exact-token FNV hashes against real models. Commit
  `273f8d4` records that `cpu_golden_qwen35`'s hash was not reproducible between
  the GitHub x86 runner and the dev box — same OS, same ISA family, different
  microarchitecture, coherent output, different tokens. Fanning that across
  three platforms multiplies a known-flaky check. The fix is
  [synthetic-models.md](synthetic-models.md): tolerance-scored logit goldens on
  synthetic fixtures are portable by construction. **That plan is the enabler
  for model-level coverage on all three platforms.**
- **`test-macos`** runs the Metal parity suite, which needs a real Metal device
  and `--include-ignored`. `tests/pcache.rs`'s `--nocapture` output is the only
  place the cold-vs-warm pipeline-cache measurement exists, so it is
  load-bearing rather than debug noise.
- **`windows-smoke`** pulls a model and runs one CPU generation on the Windows
  runner. The matrix already compiles and unit-tests there, so this exists for
  the part unit tests cannot reach: the cache location, a real ranged download,
  the blob link, and a GGUF open, on an actual Windows filesystem. It is the
  only place `link_blob`'s hard-link fallback runs — a GitHub runner's account
  has no `SeCreateSymbolicLinkPrivilege`, so `symlink_file` is refused there.
  Windows-only because the Linux and macOS legs already reach this path through
  `cpu-goldens` and `test-macos`.
- **`metal-check`** cross-lints for `aarch64-apple-darwin` from the Linux
  runner. It covers the nine crates that cross-build; `ring` (via rustls →
  reqwest → infr-hub) and `esaxx-rs` (via tokenizers → infr-chat/infr-llama) are
  C/C++ and need an Apple toolchain, which rules out the other six. Those are
  covered by the native leg. Keeping this job is about latency, not coverage: an
  arch-conditional problem fails here in about a minute instead of three.

## Things worth knowing before editing it

- **`target-cpu=native` is keyed on the Linux host triple** in
  `.cargo/config.toml`, not set under `[build]`. A `[build] rustflags` reaches
  cross-target builds, where it names the host's CPU and the cross compiler
  rejects it — `cargo check --target aarch64-apple-darwin` then buries real
  diagnostics under `'znver5' is not a recognized processor for this target`. Do
  not add an arm for a triple that is also a cross target, for the same reason:
  a `[target.<triple>]` table applies whenever that triple is the build target,
  whatever the host.
- **The Windows leg builds `msvc`,** while a local cross-lint from Linux must
  use `x86_64-pc-windows-gnu` (`msvc` wants `lib.exe`). Different ABI, so a
  local cross-lint is a fast pre-check and not a substitute for the runner.
- **Runner cost.** Windows and macOS bill at higher multipliers than Linux. If
  the matrix ever needs trimming, `clippy` is the cheaper leg to keep and `test`
  the expensive one.
