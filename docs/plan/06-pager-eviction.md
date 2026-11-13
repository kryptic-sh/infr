# Phase 06 — pager eviction: cost and policy

**Goal:** the eviction logic in every tier (VRAM arena, DRAM arena, the CPU
paged pools) costs as little as it can per touch, and picks victims as well as
the access pattern allows. A large refactor is in scope where the measurements
justify it.

Read for this plan on 2026-11-13: `infr_core::pager::Pager` in full by the
planner, then `hostpager.rs`, `infr-vulkan/src/pager.rs`,
`infr-cpu/src/paged.rs` and the call sites that drive them by a read-only survey
(step 1's reading half, recorded under "What the survey found"). Nothing was
measured yet.

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

## What the survey found (2026-11-13)

A read-only pass, nothing built or run. The first item was re-read at its site
by the planner; the rest is the survey's account; its line references are not
repeated here and each claim should be re-read before code is written against
it.

**How much the pager is called.** Dense streaming is light: one `schedule` per
streamed weight group, four to seven per layer, in pools of about forty blocks,
so the scan is negligible there. MoE is where the volume is. A block is one
expert of one role in one layer, and a pool holds `layers × n_expert` of them:
768 per role for Llama-4-Scout, 6144 for Qwen3-30B-A3B (48 layers × 128 experts,
derived), about eleven thousand for DeepSeek V4-Flash. On 30B-A3B:

- a prefill chunk that touches every expert makes 48 × 3 × 128 = 18,432
  `touch_cold` calls, each miss scanning its pool;
- a decode token with every layer resident makes 18,432 MRU touches through
  `touch_all_hits` plus as many `slot_of` lookups through `all_resident`, to
  record nothing useful: promoting every expert of a layer leaves recency with
  no information about which were routed;
- a decode token on the split path makes 48 × 3 × 8 = 1,152 touches, and any
  absent expert in a layer costs a blocking submit, a full drain and an id
  readback before its touches. That, not the scan, is expected to dominate a
  decode miss (inferred, not measured).

**A bug in the tiering, confirmed at the site.** `MoePagerSession::stage_role`
calls `HostPager::pin` for every id before `GpuPager::touch_staged`, which
ignores the bytes on a hit. So an expert that is resident in VRAM and absent
from the DRAM arena is read from disk, may evict a DRAM block, and the bytes are
discarded; and every VRAM hit pays two host locks. The dense path consults the
tier below only on a miss, and `dense_tier_parity` asserts it; the MoE path has
no such assertion. It also makes the DRAM tier's recency mirror VRAM's, which is
part of why the hottest experts are held twice.

**Other costs per touch:** `HostPager::fill` clones the block's extent list on
every call, hits included; `lut_window` copies `n_expert` words per layer and
role whether or not anything missed; misses in one batch are separate copy
commands; ids on the small-m split path are not deduplicated.

**Lock scope is not the serve bottleneck it was taken for:** the per-step GPU
baton (`StepGate`) already serialises whole forward steps under
`serve --parallel`, so the session mutex held across a copy or a disk read
serialises nothing further.

**Dead code:** `MoePagerSession::touch_role`, `GpuPager::ensure_resident` and
`flush_lut` have no production caller (tests only), and the streaming design doc
still describes `touch_role` as the decode path.

**No instrument exists:** nothing records a touch trace, nothing times `Pager`,
and `infr-core` has no ignored tests or benches to follow as a convention.

**What a refactor must keep** (each pinned by an existing test): a miss returns
the slot's previous occupant; a resident block keeps its slot; a block touched
in the current batch is never a victim, with plain LRU when batches are not in
use; cold-end hits do not promote and hot-end hits do; pins are counted and
exhaustion changes nothing; `evict` is an uncounted removal that returns its
slot to a LIFO free list handed out from slot 0; the free list drains before any
eviction; and the hit, miss and eviction counters keep their exact semantics.

## Order of work, revised by the survey

1. **Stop consulting the host tier on a VRAM hit** in the MoE path, with a test
   in the shape of `dense_tier_parity`'s "consulted == vram.misses". A bug fix,
   independent of everything below.
2. **The trace hook, the micro-benchmark and the simulator** (step 1's measuring
   half, below).
3. **Fully resident decode:** replace the per-token touch of every expert with
   something that costs O(1) per layer and keeps recency meaningful. Needs a
   design: the routed ids are on the GPU, which is why every expert is touched
   today.
4. **The data-structure refactor** (step 2), sized by the benchmark on the MoE
   pool sizes above.
5. **Policy** (step 3), and the decode-miss stall as its own question: whether
   the readback and drain can be avoided or batched.
6. Small ones alongside: the per-call clone in `HostPager::fill`, deduplicating
   split-path ids, deleting the dead entry points and correcting the design doc.

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
