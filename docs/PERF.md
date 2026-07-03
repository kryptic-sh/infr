# PERF.md — the performance optimization playbook

How to make infr faster, written for an agent (or human) picking up the perf
campaign. The reference target is llama.cpp on the same GPU: every number we
care about is a **ratio** (`infr t/s ÷ llama.cpp t/s`) on matched flags. The
campaign goal is ≥1.0x on every supported model × quant, prefill and decode,
without ever sacrificing correctness.

## The loop

Every perf slice follows the same shape. Do not skip steps, and do not do two
slices at once.

1. **Sweep** — measure the full model×quant matrix against llama.cpp and rank by
   gap (`infr compare --sweep`, below). Old numbers go stale fast: any landed
   slice can shift every ratio, so re-measure before choosing a target.
2. **Pick the biggest win** — not the most interesting one. Weigh
   `gap × how much anyone runs that config`. A model class at 0.5x beats a model
   at 0.9x. A whole quant family on a slow path beats one shape.
3. **Profile before designing** — `INFR_PROF2=1` per-op GPU timestamps. Form a
   hypothesis about the bottleneck _class_ (taxonomy below) and verify it with a
   micro-benchmark before writing the fix. Two campaign lessons: a "kernel math"
   hypothesis turned out to be occupancy (LDS budget), and a "GEMM" hypothesis
   turned out to be dispatch-launch overhead. Measurement beat intuition both
   times.
4. **Fix one lever** — the smallest change that addresses the measured
   bottleneck.
5. **Validate correctness FIRST** — parity suites + goldens (below). Only then
   benchmark.
6. **Benchmark serially** — one GPU job at a time, the changed config _and_ the
   configs that share the touched code path (a gate change for one model can
   silently reroute every other model).
7. **Commit with the numbers** — before/after t/s and ratio in the commit
   message. Update the perf log in project memory/docs.

## Benchmarking & comparing

The README "Benchmarking & profiling" section has the full command reference.
The short version:

```bash
# infr and llama-bench take the same -p/-n/-d/-r flags:
infr bench "$M" -p 512 -n 0 -r 3          # pp512 (prefill throughput)
infr bench "$M" -p 0 -n 128 -r 3          # tg128 (decode throughput)
llama-bench -m model.gguf -p 512 -n 0 -fa 1 -r 3

# The whole matrix, ranked worst-gap-first:
infr compare --sweep <models...> --sweep-depth 4096
```

Rules that exist because we got burned:

- **One benchmark at a time.** Concurrent GPU work skews both sides.
- **r=2 is noisy.** A "regression" that appears at r=2 may be variance —
  re-measure with r=3+ and compare against the historical range before reacting.
  Never auto-revert; diagnose first.
- **Fixed-count semantics.** `infr bench` sets `INFR_IGNORE_EOS` so decode
  benches generate exactly n tokens like llama-bench does. If a number looks
  impossibly good, check whether generation stopped early.
- **Rebuild freshness.** If a build tool reports "Finished in 0.06s" after a
  real edit, the output may be cached — force a real build before benching.
- Prefill and decode are separate machines. A prefill change must leave tg
  bit-for-bit alone (verify tg128 before/after); a decode change must leave pp
  alone.

## Archiving sweeps

`scripts/perf-sweep.sh <models...>` (with `INFR_METAL=1` on macOS) runs the
sweep and archives the matrix under `target/perf/<utc>-<sha>.txt` — keep the
model list fixed so every commit's ratios are a diff away, and paste the
matrix into the PR when a slice lands.

## Profiling

```bash
INFR_PROF2=1 infr bench "$M" -p 512 -n 0 -r 1   # per-op GPU timestamps
INFR_PROF_PF=1 ...                              # per-chunk prefill wall time
```

Metal uses `INFR_METAL_PROFILE` instead: `=1` encode/GPU wall split, `=3`
per-op GPU time from stage-boundary counter samples — **the only honest per-op
mode**: the decode replay tape re-executes tokens without walking ops, so the
flush mode (`=2`) silently attributes a sample of one token. `=3` disables the
tape for the run. Sweep with `--dev MTL0` (llama-bench rejects `Vulkan0` on a
Metal box). Micro-probes live in `crates/infr-metal/tests/`
(`gemv_bw`, `dispatch_overhead`) — chained-dispatch numbers there are
thermally unstable across back-to-back runs; trust e2e benches for accept /
revert calls.

- `INFR_PROF2` prints one block per submit. **Trust the percentages and per-op
  relative times; the absolute µs totals can overflow.** The op label breakdown
  is the primary signal.
- Sum GEMM-ish labels vs elementwise labels vs attention. Then look at **op
  counts**: 50k dispatches per chunk is a finding in itself, regardless of what
  any one op costs.
- For a suspect GEMM shape, add it to the per-shape micro-bench
  (`crates/infr-vulkan/tests/gemm_bench.rs`) and print µs + TFLOPS. Compare
  against the kernel's known ceiling — a shape far below ceiling is a shape
  problem (grid/occupancy), the ceiling itself being too low is a kernel
  micro-arch problem.

## Bottleneck taxonomy — what the profile means

Classify before you fix. Each class has a different lever, and the campaign has
a worked example of each:

**1. Wrong kernel entirely (coverage/gate bugs).** A fast path exists but a
routing gate excludes the shape or dtype, so it falls to a slow fallback (scalar
GEMV, dp4a mmq, the 64×64 tile). Cheapest wins in the codebase — one-line gate
fixes. Example: gemma3-1b's ne=1152 projections sat on scalar mmq at ~10 TF
because the warp-GEMM gate still required `n % 256 == 0` after a `n % 128` tile
had been added. Audit gates whenever a new kernel variant lands: _every_
eligibility check upstream of it is now potentially stale.

**2. Grid underfill.** The kernel is fine but the dispatch doesn't fill the GPU
(narrow n, small m → fewer workgroups than compute units). Levers: narrower
tiles (more workgroups for the same work), split-K (parallelize the reduction
dimension — keep the reduce fixed-order for determinism), fusing sibling GEMMs
into one wide dispatch (Q/K/V; gate+up).

**3. Occupancy ceiling.** The grid is full but each SM/WGP runs too few
workgroups concurrently to hide latency — usually LDS or register budget.
Diagnose by experiment: if bigger tiles _lose_, you're occupancy-bound, not
math-bound. Lever: shrink per-workgroup resources. Example: dropping the A-tile
from shared memory (pre-cast A to f16 once, `coopMatLoad` straight from global)
cut LDS 25→20 KB, lifted 2→3 workgroups/WGP, and took the 8B GEMMs from ~28 to
~44 TF.

**4. Dispatch/launch overhead.** Total GPU time is dominated by many tiny
dispatches and the barriers between them, not by any op's math. Tell-tales:
elementwise ops costing as much per-op as GEMMs; op counts in the tens of
thousands; small-m chunks taking absurd wall time (fixed cost doesn't scale
down). Levers: **one dispatch per stage over all items** (put the item index on
a grid dimension, early-exit surplus workgroups — empty workgroups are nearly
free), fuse adjacent elementwise stages into one kernel, drop indirect-args
machinery when a fixed worst-case grid + early-exit is simpler. Example: the
batched MoE FFN went from ~1050 dispatches + ~110 barriers per layer (per-expert
waves) to ~13 single dispatches — pp512 0.59x → 0.91x with zero GEMM changes.

**5. CPU-bound / host-in-the-loop.** The GPU is _idle_ while the host does work
that could be a shader: routing decisions, readbacks mid-graph, per-token graph
rebuilds, scheduling that forces a submit-wait-resubmit pattern. Tell-tale: wall
time ≫ summed GPU op time, or `INFR_PROF` showing record/host time rivaling GPU
time. **The standing rule: if an op runs on the CPU and can be expressed as a
shader in the pipeline, move it into the pipeline.** Every host readback is a
full pipeline stall (submit + wait + resubmit); every host-side decision that a
kernel could make keeps the GPU starved. Worked examples already in-tree:
GPU-resident MoE routing (top-k, bucket count/scan/ scatter all on-GPU — the old
path downloaded counts mid-graph to size its GEMMs), and record-once decode
replay (the `_dyn` kernels read pos/kv_len from a device-side params buffer so
the host doesn't re-record per token). When you find a host loop feeding the GPU
small submits, the fix is almost always "make the GPU compute its own control
data".

**6. Kernel micro-architecture.** Only after 1–5 are excluded: the kernel runs
at its own ceiling on a full grid and the ceiling is below the competition's.
This is the most expensive class to fix (tile shapes, pipelining, accumulator
precision) and experiments frequently lose — measure every variant on the
per-shape micro-bench before believing it.

Work the classes in that order: they're roughly sorted by effort-per-percent.

## Correctness is non-negotiable

Perf work is only real if the output is unchanged.

- **Parity first, bench second.** The CPU backend is the oracle:
  `cargo test -p infr-llama --release --test cpu_backend` (goldens for every
  model family) plus the infr-vulkan op-parity suites (`-- --ignored` for the
  GPU-gated ones).
- **Never re-bless a golden without proof.** If a hash changes, first capture a
  real generation (`infr run …`) before and after and verify the text; only then
  re-bless, and say so in the commit. Prefer designs that are bit-identical by
  construction: fixed-order reductions instead of atomics, same rounding points,
  same accumulation order. Almost every slice in this campaign (QKV fusion,
  split-K, A_GLOBAL, batched MoE) was engineered to be bit-identical — the
  goldens never moved.
- **Determinism.** Any parallel reduction you introduce must have a fixed
  summation order. Atomics that affect _values_ (not just placement) are out.
- **Zero-init discipline.** `Backend::alloc` is calloc-style; `alloc_uninit`
  only when every element is provably written before read. Padding rows that
  hold garbage are fine only when the garbage provably never feeds a real output
  (GEMM rows are independent; document the argument in a comment).
- Watch for **env leaks in tests** (parallel in-process tests + env-driven
  backend switches) — a "flaky golden" has turned out to be another test's
  leftover env var. Use serial locks and `remove_var`.

## Codebase habits

- Full-workspace clippy
  (`cargo clippy --workspace --all-targets --locked -- -D warnings`) and
  `cargo fmt --all` before every commit — per-crate clippy has missed CI
  failures.
- infr-metal is macOS-gated: Linux builds pass silently when you change a shared
  type (`Op` fields, trait signatures). Patch its sources _and tests_ blind, and
  let macOS CI verify.
- Deleting a shader: delete its build.rs entry, its `_spv` accessor, and its
  recorder method together, then make sure a _clean_ build passes (stale `.spv`
  files in the target dir can mask a missing entry locally).
- New kernel variants via compile-time defines on one `.comp` source beat
  copy-pasted shaders. Recorder binding arrays are hazard-ordered: **inputs
  first, outputs last** — a variant that adds a readonly input must renumber
  bindings to keep outputs at the end.
- Scratch buffers: pool them (`pooled(pool, be_, tag, bytes)`) so every
  same-shape op in a graph shares one allocation.

## Where the time went (campaign log, 2026-07)

For calibration — the kind of yield each class of fix produced on pp512 ratios
vs llama.cpp:

| slice                             | class             | result                                                           |
| --------------------------------- | ----------------- | ---------------------------------------------------------------- |
| fused QKV + narrow tile + split-K | grid underfill    | 0.6B 0.56 → 0.74x                                                |
| A_GLOBAL (LDS → occupancy)        | occupancy         | 0.6B → 0.92x, 8B 0.72 → 0.83x                                    |
| batched MoE (dispatch collapse)   | dispatch overhead | MoE 0.59 → 0.91x                                                 |
| warp-gate n%256 → n%128           | coverage gate     | gemma3-1b 0.60 → 0.67x                                           |
| warp-GEMM Q5_K/Q4_0/Q2_K          | coverage gate     | Q4_0 0.6B 0.43 → 0.85x, Q5_K 0.42 → 0.61x, Q2_K 14B 0.47 → 0.68x |
| IQ4_XS dqblk + warp-GEMM          | coverage gate     | 0.6B pp 0.25 → 0.67x (2.7x), tg 0.39 → 0.68x                     |

Known open items (re-sweep before trusting): the quant cliff — **largely
closed** (Q5_K/Q4_0/Q2_K/IQ4_XS now on the warptile; for the K-quants the dqblk
decoders already existed so it was pure wiring, and IQ4_XS just needed a
`dqblk_iq4xs` written — a 32-elem sub-block is exactly one IQ4_XS sub-block, so
the amortized decode also fixed its slow `native_gemv` decode). Remaining,
biggest-gap-first: **gemma-4-E2B decode is now the worst Q4_K_M combo (tg 0.64x
@d4096, 0.70x tg128)**; qwen35 engine (DeltaNet occupancy, narrow-n split-K);
gemma3-1b's narrow-shape GEMM efficiency (pp 0.66x); the warptile's ~36 TF
ceiling vs llama's ~45-50 on wide shapes (class-6, dense pp tails); and Metal
parity for everything above (most fast paths are Vulkan-only; Metal states its
own capabilities, so it degrades gracefully but slowly).
