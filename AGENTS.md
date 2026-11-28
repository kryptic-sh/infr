# Repository instructions

## Shell and local tests

- Use Git Bash on Windows when available. Do not use PowerShell.
- Keep Windows builds and tests native to Windows; do not substitute WSL for
  Windows verification.
- Run all local tests and their subprocesses headlessly, without visible console
  windows. Use hidden or no-window process creation when a runner needs it.
- Both GPUs on the Windows box may be used: the integrated GPU and the RX 7900
  XTX. Pick the device with `INFR_DEV` after checking `infr devices`, say which
  one a result came from, and run GPU workloads serially.
- Prefer cached small models. Downloading a model to test with is allowed while
  disk space lasts; check the free space and the file size first.

## Branch and delivery

- Keep all work on `main`.
- Pull the latest changes before continuing each slice of work, preserving local
  changes when integrating upstream work.
- Commit and push each completed slice to `main`.
- Run what CI runs before pushing: `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --locked -- -D warnings`,
  `cargo nextest run --workspace --locked`, and
  `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked`. The
  docs job fails on a public item's doc linking to a private one.
- Watch CI for the pushed commit, inspect its jobs, and resolve failures before
  considering the slice complete. A successful push alone is not verification. A
  push cancels the run before it, so let a run finish before pushing the next
  slice when its verdict is still owed.

## How work is run

- The main agent orchestrates. Each slice goes through a coder agent, then a
  separate reviewer agent that reads the diff without being told what to
  conclude; the orchestrator re-checks both reports against the tree before
  committing.
- While sub-agents are working, the orchestrator plans what comes next rather
  than waiting: plans under `docs/plan/`, findings in `docs/backlog.md`.
- When the backlog's actionable items run out, have a reviewer read a part of
  the codebase no earlier pass covered and record what it finds in the backlog.

## What to work on

- Accuracy comes first. A performance change is accepted only when it has no
  accuracy drawback: its output is unchanged, or at least as close to the CPU
  reference as before, and the change shows that with a bit-identity check or
  the tolerance harness. Fast and wrong is not an option.
- Performance comes next, ahead of everything else in improvement work. Where
  accuracy costs speed, take the accuracy, then look for a way to win the speed
  back without giving it up. An accuracy gap that exists today is a defect to
  investigate and close, not a cost that was accepted.
- Do not shy away from a large refactor when it is worth it for correctness, for
  performance, or for both. Say what it buys, and verify it: a test that can
  fail for correctness, a measurement for performance.
- New model families and planned features are in scope. Support recent models
  people will actually run; an older or less useful family gets support only
  when it is genuinely little work. The reference for a port is the llama.cpp
  checkout at `ref/llama.cpp` (gitignored: clone
  https://github.com/ggml-org/llama.cpp there).
- Streaming is a standing goal: disk to DRAM to VRAM, so that any model size
  runs on any hardware. That includes the cache eviction logic in every tier,
  which should be as fast as it can be made.

## Before opening a pull request

Outside contributions come as pull requests against `main`. Each of these was
missed in a pull request we then had to correct by hand:

**Shape**

- One change per pull request, based on current `main`. A stacked pull request
  names the one it is stacked on.
- Tick every box in the pull request template's checklist. An item that does not
  apply is ticked too, with "N/A" and the reason after it, so the list reads
  complete and an unticked box always means work still owed.
- Describe behaviour as it is on `main`, not on a fork: check every claim about
  which kernel or path a device takes against this tree.
- A new dependency is agreed in an issue first and lands as an optional cargo
  feature. `Cargo.lock` changes only by what the manifest change requires.
- New model work goes through the seam, so it runs on CPU, Vulkan and Metal and
  has a CPU reference. A separate forward pass beside the backends is not taken.

**Commits**

- Conventional Commits: `type(scope): subject`, at most 72 characters, then a
  blank line and a body wrapped at 72 columns that says what changed and why.

**Changelog and docs, in the same pull request**

- A `CHANGELOG.md` entry under `[Unreleased]`, in the matching section, naming
  the key, flag or behaviour that changed.
- A changed public API is a `Breaking` entry: a new enum variant, a new public
  struct field, a changed signature.
- A new config key updates `docs/config.md` and `infr.example.toml`; a new
  server route or request field updates the README's server section; a new
  tuning parameter updates the tuning docs.
- Markdown goes through `npx prettier --write`.

**Tests**

- A performance change shows that its output did not change, or did not move
  away from the CPU reference: the bit-identity check or the tolerance rows it
  was held to. See "What to work on".
- A fix comes with a test that fails without it, and the test exercises the line
  that changed. Say in the pull request that you saw it fail.
- GPU tests are `#[ignore]`d and CI has no GPU, so a green run says nothing
  about a shader or kernel change. Say which ignored tests you ran and on which
  device.
- GPU tests use tolerance testing, not hash goldens. A new test of what a GPU
  kernel or a GPU run of a model produces compares it with a reference within a
  stated bound: the tolerance harness
  (`crates/infr-llama/tests/gpu_tolerance.rs`, scored against the CPU reference)
  for a model, a host reference for a single op. Do not add a GPU test that pins
  a stored hash, digest or byte-exact golden value, and do not add or re-bless a
  golden file: GPUs and drivers legitimately differ in the last bits, so such a
  test fails on hardware that is right. Comparing two kernels bit for bit inside
  one test, on the same device and input, is fine, and is how "this change keeps
  the output identical" is shown.
- The tolerance gate keeps its CPU reference in an on-disk cache
  (`INFR_TOL_REFCACHE`: `on` by default, `off`, `verify`, `refresh`, `clear`;
  off when `CI` is set). Say which mode a reported run used; `verify` recomputes
  every reference and fails on a stored entry that differs. See
  `docs/gpu-tolerance.md`, "The reference cache".
- Model math is compared with a reference: llama.cpp's embeddings or logits on
  the same input. Output that reads well is not evidence.
- A server feature has a test against a real small model, not only the stub
  generator.

**Code**

- Fix every site with the same defect: grep for the pattern before opening the
  pull request.
- Reuse the shared path instead of copying it. A copied handler loses the error
  mapping and cancellation the original has.
- Comments say why. Measurements and history go in the pull request's
  description and the changelog, never in code or shader comments.

**Performance claims**

- State the device, the driver, the exact command and the number of runs, with
  the two builds alternated. Say which kernels and device classes the change
  reaches, and which you could not measure.
