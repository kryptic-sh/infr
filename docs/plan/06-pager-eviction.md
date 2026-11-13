# Phase 06 — pager eviction: cost and policy

**Goal:** the eviction logic in every tier (VRAM arena, DRAM arena, the CPU
paged pools) costs as little as it can per touch, and picks victims as well as
the access pattern allows. A large refactor is in scope where the measurements
justify it.

Read for this plan on 2026-11-13: `infr_core::pager::Pager` in full. **Not
read:** `hostpager.rs`, `infr-vulkan/src/pager.rs`, `infr-cpu/src/paged.rs`, and
how the Vulkan sessions batch their touches. Step 1 below reads them before any
design is fixed.

## What the code does today

`Pager` keeps four `HashMap`s keyed by `BlockId` (`resident`, `recency`,
`epoch`, `pinned`), with recency as a signed stamp per block.

- A hit is a few hash writes (`epoch.insert`, `resident.get`, `recency.insert`).
- **A miss on a full cache scans every resident block:** `coldest` iterates the
  whole `recency` map and, for each entry, looks the block up in `epoch` (and in
  `pinned` on the fallible path) before taking the minimum stamp. The doc on
  `recency` accepts this because "it is paid only when choosing a victim, which
  already costs a fill".
- A batch that misses on k blocks runs that scan k times.
- Policy is recency with two insertion points: the hot end (`touch`) and the
  cold end (`touch_cold`, `schedule`). `schedule` on a cyclic sweep is
  Belady-optimal by test (`schedule_cyclic_sweep_is_belady_optimal`). Nothing
  uses frequency.

Whether the scan matters is a question of pool size and miss rate: it is
negligible beside a multi-megabyte upload for a pool of a few hundred blocks,
and not for per-expert pools of a large MoE (a block per expert per role per
layer) during a prefill sweep that misses on most of them. That has not been
measured, which is step 1.

## Steps

### 1. Measure before designing

- Read the three consumers and write down, per session type, the pool sizes real
  models produce and how many touches and misses one decode token and one
  prefill chunk make.
- A pager micro-benchmark in `infr-core` (a `#[ignore]`d test or a bench binary,
  following how the crate already does timing tests): replay a touch trace
  against `Pager` and report nanoseconds per touch and per miss. Traces: a
  cyclic sweep, a Zipf-skewed routed pattern with a prefill sweep mixed in, and
  at least one trace **recorded from a real MoE run**. Recording needs a trace
  hook on the sessions behind `INFR_PAGER_STATS` (or a sibling debug key); the
  hook is part of this step.
- From the same traces, an offline policy simulator: misses under today's
  policy, under each candidate in step 3, and under Belady (the lower bound a
  trace allows). This is the instrument that says whether policy work is worth
  anything for a given model.
- Take the timing numbers in a quiet window, per the repo's measurement rules.

### 2. Data-structure refactor (cost per touch)

Decided by step 1's numbers; the expected shape, to be confirmed against the
consumers:

- **Slot-indexed state instead of id-keyed maps.** Residency is bounded by
  `n_slots`, so recency links, the epoch and the pin count can live in arrays
  indexed by slot. The `id → slot` map is the one lookup that needs a key, and
  where block ids are dense (the sessions assign them from a `layer_base`) it
  can be a direct-index table rather than a hash.
- **An intrusive doubly-linked LRU list** over slots: a hit is two link
  rewrites, and the coldest unprotected block is a walk from the cold end that
  stops at the first eligible slot, instead of a scan of everything. Both
  insertion points map onto it directly (push at head, push at tail).
- **Batch-aware victim selection:** hand the pager the whole batch (or its
  count) so k victims come out of one pass.
- The public API (`touch`, `touch_cold`, `schedule`, `resolve_and_pin`, the pin
  calls) keeps its semantics; the existing test module is the contract and must
  pass unchanged. If the consumers need a different entry point (a batched
  resolve), add it and move them in the same change.

This is the large refactor if it is warranted: `Pager` is shared by every tier,
so one change pays everywhere, and its tests run without hardware.

### 3. Policy (misses per trace)

Only where the simulator shows a gap to Belady worth closing:

- **MoE routing is skewed**, and recency alone lets a prefill sweep or a burst
  of rare experts displace the hot set (the cold-end insert and the admission
  doorkeeper in `HostPager` are partial answers). Candidates to simulate: a
  frequency-aware admission filter (TinyLFU-style), S3-FIFO, ARC, and LFU with
  aging. Pick by miss count on the recorded traces, then by bookkeeping cost.
- **Tier-coordinated placement for MoE:** today each tier caches by the same
  policy, so the hottest experts are held twice (VRAM and DRAM). Exclusive
  placement would make DRAM hold the next-hottest set instead (B36 lists it as
  unbuilt). Simulate the two-tier miss counts before building.
- **Dense stays as it is:** the cyclic sweep is already optimal under
  `schedule`, and the planned partition (`TierPlan`) is a placement question,
  not an eviction one.

### 4. The costs around eviction

Read in step 1, measured before changing:

- What an eviction costs on the GPU side beyond the upload: the device LUT
  write, a pipeline rotation or fence wait when the staging ring fills
  (`ring_bytes` documents the stall), and whether misses within a batch are
  uploaded together.
- Lock scope: `DensePagerSession::stage` copies under the session mutex
  (`disk-streaming-plan.md` §1.1). A pager that is cheap per touch still
  serialises `serve --parallel` behind that lock.

## Verification

- The existing `Pager` tests pass unchanged through the refactor; new tests for
  any new entry point, each run red first.
- The micro-benchmark and the simulator are kept in the tree so a later policy
  change is compared the same way.
- End to end: `INFR_PAGER_STATS` hit, miss and eviction counts are unchanged by
  the data-structure refactor on the same run (it must not change policy), and
  decode and prefill throughput on a capped MoE run and a capped dense run are
  taken before and after, interleaved, in a quiet window.
- Content-checked parity (`dense_tier_parity`, the `*_matches_resident` tests)
  on both GPUs after each step: a wrong victim is silent wrong output.
