# infr docs

Design docs, backend architecture, performance playbooks, and campaign logs for
the `infr` inference engine. The top-level project overview lives in the root
[`README.md`](../README.md); everything here is deeper reference.

## Using infr

- [config.md](config.md) — the configuration reference: the four layers
  (defaults < config file < `INFR_*` env < CLI flags) and their precedence, the
  TOML file format and lookup order, `--set`, and a per-section walkthrough of
  what is tunable. Start here before reaching for an `INFR_*` variable.

## Performance

Everything performance-related lives under **[perf/](perf/README.md)** — start
at that index. It holds:

- [perf/results.md](perf/results.md) — the numbers: every validated model ×
  quant against llama.cpp on an RX 7900 XTX, per-row footnotes for each kernel
  slice that moved a column, and where infr still loses. Moved out of the root
  README, which now carries only the headline.
- [perf/benchmarking.md](perf/benchmarking.md) — how to produce them:
  `infr bench` / `infr compare --sweep` against `llama-bench`, per-op GPU
  profiling, shape-itemised buckets, CPU `samply`.
- [perf/playbook.md](perf/playbook.md) — the optimization method and the
  recorded dead ends. Read before starting a perf slice.
- [perf/kernels.md](perf/kernels.md) — cross-backend fast-kernel coverage (24/24
  quant formats on CPU / Vulkan / Metal) and each backend's decode strategy.
- [perf/cpu.md](perf/cpu.md) — the CPU backend's own roadmap.
- [perf/vulkan-review.md](perf/vulkan-review.md) — multi-vendor review: what is
  RDNA3-tuned versus portable, and the per-vendor gaps.

## Backends

- [metal.md](metal.md) — Apple GPU backend (`infr-metal`) architecture: the
  `DEC16` decode kernels, decode-parity campaign, multi-slot serve, native-read
  KV, MTP, and the replay-tape correctness fix.
- [igpu.md](igpu.md) — integrated-GPU correctness campaign (AMD APU / Intel iGPU
  / Strix Halo class): the UMA heap-table insight, the per-submit watchdog
  root-cause + submit-splitter fix, and the model survey. Phase 1 complete.

## Models & architectures

- [qwen35.md](qwen35.md) — Qwen3.5 / Qwen3.6 (`qwen35`): the gated-DeltaNet
  linear-attention + full-attention hybrid, and the interleaved q+gate trap.
- [qwen38.md](qwen38.md) — Qwen3.8 support plan. One release name, three archs:
  the dense 27B is `qwen35` and may already run, the 2.4T is `qwen35moe`, and
  Flash-Next is a net-new `qwen4exp` (hyper-connections, n-gram PLE,
  block-sparse QSA) that reuses DeepSeek V4's compressed-KV machinery.
- [diffusion-gemma.md](diffusion-gemma.md) — DiffusionGemma design for the
  unified seam: block text-diffusion, the canvas denoise graph, and
  self-conditioning.
- [mtp.md](mtp.md) — multi-token prediction (MTP) speculative decoding for
  qwen35's single NextN head (issue #33).
- [deepseek.md](deepseek.md) — the DeepSeek family port plan (V1 → V2/V3 → V3.2
  → V4), staged around the fact that only the first two stages have a model
  small enough to develop against. Nothing implemented yet.

## Roadmaps & history

- [plan.md](plan.md) — the whole-system shape in one place: what shipped against
  the original MVP, the crate layout and backend seam, the step-by-step recipe
  for **adding a model architecture**, the ranked candidate families, and the
  original milestones as history.
- [train.md](train.md) — LLM training support plan (not yet built).

## Testing & CI

- [synthetic-models.md](synthetic-models.md) — plan to give every architecture a
  fake in-repo GGUF fixture, so real multi-GB models can be deleted without
  losing coverage. Why token-hash goldens are not portable, why a golden must be
  blessed against an external oracle, and which archs are dark today.
- [infr-plat.md](infr-plat.md) — the platform seam, as shipped: `infr-plat` owns
  every direct use of an operating system (file locks, positional IO, memory
  query, signals, process liveness, config/cache dirs, interruptible stdin), and
  a test forbids any other crate from depending on `libc` or `windows`. Read it
  before adding platform-specific code.
- [ci-matrix.md](ci-matrix.md) — the three-platform CI matrix, as shipped: what
  runs natively on Linux, macOS and Windows, which jobs stay single-platform and
  why, and the `target-cpu=native` and `msvc`-vs-`gnu` traps to know before
  editing the workflow.

## Audit

- [audit.md](audit.md) — module-by-module codebase audit for bugs, correctness,
  perf, DRY, and YAGNI.
- [backlog.md](backlog.md) — triaged work that is deliberately not done, with
  why (blocked on hardware, scoped out, or declined), plus withdrawn findings
  recorded so they are not rediscovered. The whole-tree correctness reviews that
  used to live in `code-review.md` were folded into it on 2026-08-03 and that
  file deleted: the re-verified findings are B19–B26, and the reviews' cleared /
  hardening / coverage lists are B27–B29.
