# Phase 06 — pager eviction: cost and policy

**Goal:** the eviction logic in every tier (VRAM arena, DRAM arena, the CPU
paged pools) costs as little as it can per touch, and picks victims as well as
the access pattern allows. A large refactor is in scope where the measurements
justify it.

Read for this plan on 2026-11-13: `infr_core::pager::Pager` in full by the
planner, then `hostpager.rs`, `infr-vulkan/src/pager.rs`,
`infr-cpu/src/paged.rs` and the call sites that drive them by a read-only survey
(step 1's reading half, recorded under "What the survey found"). Step 1's
measuring half landed the same day; its numbers are under "Measurements".

## What the code does today

Since step 2, `Pager` keeps its state by slot: the block in each slot, its batch
epoch and its pin count, one `BlockId → slot` map (FxHash-keyed), and recency as
an intrusive doubly linked list over the slots (`pager::lru`), coldest first.

- A hit is a map lookup and an epoch write, and an MRU hit also moves the slot
  to the hot end: a few index writes.
- **A miss on a full cache walks from the cold end** to the first block that is
  neither pinned nor touched by the current batch, so it costs the protected
  blocks it passes, not the cache size. A batch that cold-inserts k misses walks
  past its own earlier inserts, k²/2 steps over the batch.
- A group touch (`Pager::touch_group`) moves one node, the group's, to the hot
  end; the walk reaching it takes the group's first unprotected member in id
  order (`pager::groups`' module doc).
- Policy is recency with two insertion points: the hot end (`touch`) and the
  cold end (`touch_cold`, `schedule`). `schedule` on a cyclic sweep keeps a
  stable resident prefix: `n_slots − 1` hits per pass, close to Belady but not
  equal to it (MIN rotates what it keeps; in steady state it misses about
  `(n_slots − 1)/(n_blocks − 1)` fewer blocks per pass). Nothing uses frequency.

Before step 2 the same policy ran on four `HashMap`s keyed by `BlockId`
(`resident`, `recency`, `epoch`, `pinned`) with recency as a signed stamp per
block, and **a miss on a full cache scanned every resident block** (`coldest`
took the minimum stamp over the whole `recency` map, looking each block up in
`epoch` and `pinned`), k times for a batch of k misses. That implementation is
kept as the test-only reference model the new one is checked against
(`pager::reference`).

Whether the scan mattered was a question of pool size and miss rate: it was
negligible beside a multi-megabyte upload for a pool of a few hundred blocks,
and not for per-expert pools of a large MoE (a block per expert per role per
layer) during a prefill sweep that misses on most of them. "Measurements" has
the numbers.

## What the survey found (2026-11-13)

A read-only pass, nothing built or run. The survey's account; its line
references are not repeated here and each claim should be re-read before code is
written against it. The tiering bug it found (the MoE session consulting the
host tier on a VRAM hit) is fixed, and its remaining findings outside the order
of work below are B156.

**How much the pager is called.** Dense streaming is light: one `schedule` per
streamed weight group, four to seven per layer, in pools of about forty blocks,
so the scan is negligible there. MoE is where the volume is. A block is one
expert of one role in one layer, and a pool holds `layers × n_expert` of them:
768 per role for Llama-4-Scout, 6144 for Qwen3-30B-A3B (48 layers × 128 experts,
derived), about eleven thousand for DeepSeek V4-Flash. On 30B-A3B:

- a prefill chunk that touches every expert makes 48 × 3 × 128 = 18,432
  `touch_cold` calls, each miss scanning its pool;
- a decode token with every layer resident made 18,432 MRU touches through
  `touch_all_hits` plus as many `slot_of` lookups through `all_resident`. Both
  are now one call per layer and role (a block group; see "Fully resident
  decode" under "Measurements");
- a decode token on the split path makes 48 × 3 × 8 = 1,152 touches, and any
  absent expert in a layer costs a blocking submit, a full drain and an id
  readback before its touches. That, not the scan, is expected to dominate a
  decode miss (inferred, not measured).

**Other costs per touch:** `HostPager::fill` clones the block's extent list on
every call, hits included; `lut_window` copies `n_expert` words per layer and
role whether or not anything missed; misses in one batch are separate copy
commands; ids on the small-m split path are not deduplicated.

**Lock scope is not the serve bottleneck it was taken for:** the per-step GPU
baton (`StepGate`) already serialises whole forward steps under
`serve --parallel`, so the session mutex held across a copy or a disk read
serialises nothing further.

**Dead code:** `MoePagerSession::touch_role` is deleted.
`GpuPager::ensure_resident`, `flush_lut` and the device LUT behind them have no
production caller, but are how the paged-kernel parity tests fill an arena; see
B156.

**No instrument exists:** nothing records a touch trace, nothing times `Pager`,
and `infr-core` has no ignored tests or benches to follow as a convention.

**What a refactor must keep** (each pinned by an existing test): a miss returns
the slot's previous occupant; a resident block keeps its slot; a block touched
in the current batch is never a victim, with plain LRU when batches are not in
use; cold-end hits do not promote and hot-end hits do; pins are counted and
exhaustion changes nothing; `evict` is an uncounted removal that returns its
slot to a LIFO free list handed out from slot 0; the free list drains before any
eviction; the hit, miss and eviction counters keep their exact semantics; a
group touch (`Pager::touch_group`) leaves the pager exactly as a `touch` of each
member in id order would, and a group is resident exactly while every member is
(`pager::groups`' tests, among them a randomized equivalence check).

## Order of work, revised by the measurements

What the numbers below support, largest measured end-to-end share first:

1. **The trace hook, the micro-benchmark and the simulator** (step 1). Done.
2. **The data-structure refactor** (step 2). Done: pager time per call falls by
   2.9× (DeepSeek-scale routed decode whose blocks all fit) to 437× (the same
   with slots for a quarter of the blocks) on the recorded and generated traces,
   and decode at `paging.cache=4g` + `paging.dram=12g` ran 9% faster
   ("Slot-indexed pager" under "Measurements"). The scan it removed was about 1%
   of end-to-end time in the disk-bound run and up to 6% of decode where misses
   are served fast.
3. **MoE policy** (step 3), decode only: two-tier placement first (a DRAM tier
   smaller than VRAM works only as a victim cache), then eviction candidates
   compared in `sim` on the recorded `moe` trace. The decode-miss stall
   (readback and drain) is its own question beside it.
4. Small ones alongside: the per-call clone in `HostPager::fill` and
   deduplicating split-path ids (both in B156).

## Measurements (2026-11-13)

Taken with the step-1 tools on the Windows box (Ryzen 9 9950X3D). **Busy
machine:** other sessions shared it; total CPU was 1–6% while the benchmark ran
and no quiet window was arranged, so the nanosecond columns are indicative. The
hit rates and miss counts are exact.

Columns: `blocks` is the id space a pool's header names (the MoE pools of one
role share the model's whole `48 × 128` space, so a `down` pool, whose layers
are split between two quant sizes, reaches only half of it: `ids`); `ns/hit` and
`ns/miss` are split from the per-call total by timing runs of same-outcome calls
(the method is in the benchmark's `time_trace`); `gap` is the share of today's
misses Belady avoids on that request sequence; `scan share` is the share of
pager time misses spend beyond what a hit costs — the victim scan, essentially.

**What a Belady figure means here.** It is the fewest misses _this recorded
request sequence_ allows with the same slots, batch guard and pins, and it is a
lower bound only for traces of `begin_batch`, the `touch` family,
`resolve_and_pin` and `unpin`: there it holds by an exchange argument and was
checked against a brute-force optimum on 3000 small random traces, not proven.
With `pin_if_resident` probes or `repin`s it is not a bound (a probe pins only a
resident block, so pins depend on the policy; `sim` has a counterexample), and
no figure is given: the `cpu` pools below. It also does not say what another
policy would save end to end where the requests themselves depend on residency,
as MoE decode's do (below).

### Recorded traces

Five runs, each checked first: **every pool's replayed hit, miss and eviction
counts equal the `paging.stats` line the run printed** (35 pools, all equal), so
the traces are complete.

- `moe`: Qwen3-30B-A3B Q4_K_M on the RX 7900 XTX, `paging.cache=4g`,
  `paging.dram=1g`, a 441-token prompt and 48 decode tokens (prefill 441 tokens
  in 28,366 ms, decode 11.7 tok/s). About a fifth of the experts fit VRAM.
- `moe-bigdram`: the same with `paging.dram=12g` (decode 19.8 tok/s, prefill
  28,720 ms). Its VRAM pools' counts are identical to `moe`'s.
- `moe-resident`: `paging.cache=18g`, which seats 98% of the experts (decode
  132.9 tok/s, mostly `touch_all_hits`).
- `dense`: Qwen3-1.7B Q4_K_M on the integrated GPU, `paging.cache=400m` (unified
  memory: no DRAM arena, so only VRAM pools).
- `cpu`: Qwen3-1.7B Q4_K_M on the CPU backend, `paging.dram=300m`, 16 tokens.

| trace / pool                         | blocks |  ids | slots |  calls | hit rate | ns/call | ns/hit | ns/miss | misses: current | misses: Belady |   gap | scan share |
| ------------------------------------ | -----: | ---: | ----: | -----: | -------: | ------: | -----: | ------: | --------------: | -------------: | ----: | ---------: |
| moe, down 1.3 MB                     |   6144 | 3072 |   657 |  19704 |    64.1% |  1717.2 |   25.2 |    5087 |            6616 |           4773 | 27.9% |      98.6% |
| moe, down 1.3 MB host                |   6144 | 3072 |   187 |  13232 |     0.0% |   502.3 |      — |     989 |            6616 |           5520 | 16.6% |          — |
| moe, gate 0.9 MB                     |   6144 | 6144 |  1315 |  33600 |    61.1% |  3587.1 |   27.0 |    9981 |           12112 |           8865 | 26.8% |      99.3% |
| moe, gate host                       |   6144 | 6144 |   375 |  24224 |     0.0% |   916.8 |      — |    1824 |           12112 |          10261 | 15.3% |          — |
| moe, up 0.9 MB                       |   6144 | 6144 |  1315 |  33600 |    61.1% |  3612.5 |   27.2 |   10057 |           12112 |           8865 | 26.8% |      99.3% |
| moe, up host                         |   6144 | 6144 |   375 |  24224 |     0.0% |   916.9 |      — |    1828 |           12112 |          10261 | 15.3% |          — |
| moe, down 0.9 MB                     |   6144 | 3072 |   657 |  13896 |    56.1% |  2030.5 |   28.1 |    5102 |            5558 |           4247 | 23.6% |      98.7% |
| moe, down 0.9 MB host                |   6144 | 3072 |   187 |  11116 |     0.0% |   497.7 |      — |     990 |            5558 |           4860 | 12.6% |          — |
| moe-bigdram, down 1.3 MB host        |   6144 | 3072 |  2255 |  13232 |    45.9% |  1151.9 |   43.2 |    4204 |            3580 |           3072 | 14.2% |      97.0% |
| moe-bigdram, gate host (up alike)    |   6144 | 6144 |  4510 |  24224 |    40.8% |  2480.7 |   43.8 |    8112 |            7166 |           6144 | 14.3% |      98.6% |
| moe-bigdram, down 0.9 MB host        |   6144 | 3072 |  2255 |  11116 |    35.4% |  1327.4 |   41.1 |    4079 |            3588 |           3072 | 14.4% |      97.5% |
| moe-resident, down 1.3 MB            |   6144 | 3072 |  3006 | 145152 |    97.5% |   108.5 |   22.1 |    3606 |            3504 |           3336 |  4.8% |      79.8% |
| moe-resident, gate 0.9 MB (up alike) |   6144 | 6144 |  6013 | 296112 |    97.6% |   180.0 |   23.4 |    6768 |            6950 |           6538 |  5.9% |      87.3% |
| moe-resident, down 0.9 MB            |   6144 | 3072 |  3006 | 150960 |    97.6% |   107.0 |   22.1 |    3684 |            3516 |           3336 |  5.1% |      79.6% |
| moe-resident, gate host              |   6144 | 6144 |  6144 |  13900 |    11.6% |    28.0 |   38.1 |      49 |            6144 |           6144 |  0.0% |      13.6% |
| dense, pool0 2.4 MB                  |     56 |   56 |     9 |   6048 |    14.0% |    58.1 |   25.9 |     135 |            2600 |           2449 |  5.8% |      72.6% |
| dense, pool1 1.2 MB                  |     42 |   42 |     7 |   4536 |    15.6% |    52.2 |   25.2 |     123 |            1914 |           1831 |  4.3% |      70.5% |
| dense, pool3 14.2 MB                 |     28 |   28 |     4 |   3024 |    15.5% |    41.7 |   25.9 |     100 |            1278 |           1272 |  0.5% |      63.9% |
| dense, pool2/4/5 (each)              |     14 |   14 |     2 |   1512 |    12.2% |    36.6 |   30.0 |      83 |             664 |            660 |  0.6% |      54.4% |
| cpu, 2.4 MB                          |     56 |   56 |    16 |   3808 |    25.2% |    38.9 |   33.3 |     156 |             712 |              — |     — |      52.9% |
| cpu, 7.1 MB                          |     70 |   70 |    20 |   4760 |    25.5% |    42.7 |   33.1 |     177 |             886 |              — |     — |      56.9% |

`moe-bigdram`'s VRAM rows equal `moe`'s, and `moe-resident`'s other host pools
read like its gate host; the other `cpu` classes read like the two shown.

### Prefill and decode apart

The `moe` gaps above mix the two phases. Split at each pool's last prefill call
(`touch_cold`), from one simulated run each (`recorded_phase_split`):

| pool                          | prefill: current / Belady | decode: current / Belady | decode batches: touch-all / routed |
| ----------------------------- | ------------------------- | ------------------------ | ---------------------------------- |
| moe, down 1.3 MB              | 3072 / 3072 (0.0%)        | 3544 / 1701 (52.0%)      | 48 / 1104                          |
| moe, gate (up alike)          | 6144 / 6144 (0.0%)        | 5968 / 2721 (54.4%)      | 48 / 2256                          |
| moe, down 0.9 MB              | 3072 / 3072 (0.0%)        | 2486 / 1175 (52.7%)      | 0 / 1152                           |
| moe-resident, down 1.3 MB     | 3468 / 3270 (5.7%)        | 36 / 66 (−83.3%)         | 1008 / 144                         |
| moe-resident, gate (up alike) | 6876 / 6537 (4.9%)        | 74 / 1 (98.6%)           | 2064 / 240                         |
| moe-resident, down 0.9 MB     | 3468 / 3270 (5.7%)        | 48 / 66 (−37.5%)         | 1056 / 96                          |

A touch-all batch is one layer's `touch_all_hits`: all 128 experts MRU-touched;
a routed batch touches the 8 experts the router picked. The decode path picks
between them per layer by `all_resident`, which the trace does not record, so a
different eviction policy would have produced a different request stream.

- **`moe`:** the prefill is a first sweep through a cache a fifth of its size,
  and every policy misses each block once. **Decode is where the gap is: about
  half of the decode misses (52–54%)** on a stream that is 96–100% routed
  batches, whose requests are the router's demand. It is still a bound on this
  sequence only.
- **`moe-resident`:** decode is 88–92% touch-all batches, whose "next uses" are
  bookkeeping touches of every expert, not demand; Belady's decode figures there
  (including the negative ones, where it spends decode misses to save prefill
  ones) say nothing about routed demand. Its 5–6% is the first fill of a cache
  2% smaller than the expert set. It does not show room for policy.

### The DRAM tier

In `moe` every host pool hit 0.0%, and all 36,398 host reads went to disk. That
follows from the configuration, not from the policy: the host tier sees only
VRAM's misses, and a block VRAM misses twice has had many other VRAM misses —
each a host access to a distinct block — in between. Checked on the trace: for
blocks the host was asked for twice, the distinct host blocks accessed between
the two requests were a median of 910 / 1965 / 1965 / 1103 (down 1.3 MB, gate,
up, down 0.9 MB) against 187 / 375 / 375 / 187 host slots; 3, 10, 10 and 6 of
the 3544, 5968, 5968 and 2486 re-requests had fewer in between than the slot
count, and none of them hit. **A DRAM tier smaller than VRAM under the same
policy can only work as a victim cache** — admitting what VRAM evicts, not what
it misses (exclusive placement, step 3).

With `paging.dram=12g` (`moe-bigdram`: 2255 / 4510 / 4510 / 2255 host slots,
3.4× VRAM's) the host tier hits 35–46%, its disk reads fall from 36,398 to
21,500 and decode runs at 19.8 tok/s instead of 11.7. That is the regime the
auto-sized DRAM tier is meant for; this one run is all that was measured of it.
Belady's gap on it is 14%.

### The cost of the hook itself

Unset, `paging.trace` is one `Option` test per call. Measured by building
`infr-core` at `HEAD` and with the hook in two otherwise identical workspaces
and running the same driver against each, alternately, three times (busy
machine): a 6144-id cold sweep through 3072 slots 12.92–12.99 µs per call
before, 12.80–13.05 after; MRU hits 22.39–23.45 ns before, 22.65–22.76 after. No
difference outside the noise. (The same comparison found an older anomaly in how
`infr-core` builds the pager: B164.)

### What the numbers say

Arithmetic from the tables (ns/call × calls; wall time from the runs' own
lines):

- **(i) The victim scan is nearly all of the pager's own time on MoE pools:**
  98.6–99.3% of it on the `moe` VRAM pools, 80–88% even on `moe-resident`.
- **(ii) As a share of end-to-end time it is about 1% in the recorded disk-bound
  run.** `moe`'s pager time over all eight pools: VRAM 19704 × 1717.2 + 33600 ×
  3587.1 + 33600 × 3612.5 + 13896 × 2030.5 ns = 304.0 ms, host 13232 × 502.3 +
  24224 × 916.8 + 24224 × 916.9 + 11116 × 497.7 ns = 56.6 ms, together 0.36 s.
  The run took 28.37 s of prefill plus 48 / 11.7 = 4.10 s of decode, 32.47 s:
  **1.1%**. Each of its misses is a ~1 MB expert read from disk: the host tier
  read 34.9 GB (`paging.stats`) in that time.
- **(iii) The scan matters end to end only where misses are served fast, or at
  larger pools.** At an assumed 25 GB/s host-to-VRAM (not measured here), a 0.88
  MB expert upload takes 35 µs and a 1.29 MB one 52 µs; the scan beside it is 10
  µs per miss in the 1315-slot pools and 5 µs in the 657-slot ones, so 10–28% of
  a miss's cost when the bytes are already in DRAM. In decode, the VRAM pools'
  misses cost (5968 × 9981 + 5968 × 10057 + 3544 × 5087 + 2486 × 5102) ns plus
  their hits, about 152 ms over 48 tokens: 3.2 ms per token. That is 3.7% of
  `moe`'s 85.5 ms tokens and 6.3% of `moe-bigdram`'s 50.5 ms ones. On the
  generated DeepSeek-scale prefill a miss scans for 20–31 µs, comparable to its
  upload, and a chunk costs 0.20–0.26 s of pager time per pool.
- **(iv) Fully resident decode is the largest end-to-end share measured, and it
  is spent on hits.** `moe-resident`'s decode made 130,140 + 266,038 + 266,038 +
  135,888 = 798,104 hits over 48 tokens (16,627 per token); at 22.1–23.4 ns each
  that is 18.2 ms, 0.38 ms per token, against 1 / 132.9 = 7.52 ms per token:
  **5.0%** (the `all_resident` lookups beside them are not pager calls and are
  not counted). A slot-indexed structure makes each of those hits cheaper; a
  design that stops touching every expert removes them — which is what landed
  next (below).
- **Policy:** prefill sweeps are within 1.2–1.8% of Belady (generated) and 0%
  (recorded first sweep), dense streaming within 0.5–5.8%. **Constrained MoE
  decode is the only place with room: Belady halves the decode misses of the
  `moe` request sequence.** Each saved miss is an expert upload and, there, a
  disk read — the bigger lever than the scan in that regime, with the caveats
  above about what the bound means.

### Fully resident decode (2026-11-13)

**What the touches were for.** A wholly resident layer took the inline path:
`all_resident` (one `slot_of` per expert and role), then `touch_all_hits` (a
`begin_batch` and an MRU `touch` of every expert). The batch epoch those touches
set protected nothing: the next call on the pool is `stage_and_window`'s own
`begin_batch`, before any touch, and the slots a frozen window names are ordered
against later copies by the adapter's arena barriers. What the touches did do is
keep the layer recent, so that a miss elsewhere evicts the paged layers' stale
experts rather than a layer used every token — which would send that layer down
the split path (a blocking readback) on its next token.

**What replaced them.** `Pager` groups blocks into equal runs (`group_by`; an
MoE pool's layer is a group, registered by `MoePagerSession::register`). It
counts each group's resident members on every admission and removal, so
`all_resident` is one lookup, and `touch_group` stands for an MRU touch of every
member in id order: one stamp for the group (member `k` ranks at `stamp + k`),
its batch epoch, and the hits counted. Victim selection ranks a block by the
later of its own stamp and its group's; an admission into a stamped group first
writes the group stamp into the members still resident and drops it. **The
policy is unchanged on purpose**: a group touch leaves the pager exactly as the
per-member touches did (`pager::groups`' randomized equivalence test, for
power-of-two and other group lengths), so the cost goes and the eviction
decisions stay. What that policy means is in B156.

**Exact counts**, Qwen3-30B-A3B Q4_K_M on the RX 7900 XTX, a 365-token prompt
and 48 greedy tokens, traces recorded with the `HEAD` binary and the changed
one:

| configuration                 | VRAM pool calls per decode token: before / after | output, `paging.stats` | call sequence (group touches expanded) |
| ----------------------------- | ------------------------------------------------ | ---------------------- | -------------------------------------- |
| `paging.cache=18g`            | 16,905 / 522                                     | identical              | identical, all 8 pools                 |
| `paging.cache=4g`, `dram=12g` | 1,659 / 1,278                                    | identical              | identical, all 8 pools                 |

Per wholly resident layer and role that is 130 calls (`b`, 128 `t`, `b`) before
and 3 (`b`, `g`, `b`) after. "Decode" is everything after the pool's last cold
touch. At 4 GB one layer stays wholly resident in three of the four pools, so
the fast path runs there too, once per token.

**Pager time (replay benchmark, busy machine).** Generated resident decode, one
pool, per token: 146.9 → 1.85 µs (48 × 128) and 271.9 → 2.96 µs (43 × 256).
Replaying the recorded traces, the four VRAM pools' pager time over the whole
run (prefill included), two repetitions each: 18 GB 110.4 / 111.7 ms before,
92.3 / 92.5 ms after; 4 GB 301.7 / 299.1 ms before, 312.5 / 295.0 ms after — the
same within the spread. (A first version looked up a block's group twice per
victim-scan step and by division; at 4 GB its per-miss scan was 3–7% slower, so
the lookup is now one, by shift for a power-of-two group.)

**Decode throughput (`infr bench -p 0 -n 48 -d 365 -r 3`, RX 7900 XTX, busy
machine, binaries alternated):**

| configuration                 | before (t/s)        | after (t/s)         |
| ----------------------------- | ------------------- | ------------------- |
| `paging.cache=18g`            | 145.0, 146.3, 145.6 | 145.5, 145.6, 145.5 |
| `paging.cache=4g`, `dram=12g` | 74.9, 76.0, 75.8    | 74.9, 76.1, 75.5    |

No difference outside the run-to-run spread, at 18 GB either, although the
replay puts the pager work removed at about 0.38 ms per token ((110.4 − 92.3) ms
over 48 tokens, matching the step-1 estimate). In these runs the submit splitter
is armed (the paging prefill's first forward passes its danger threshold), so
the host records while the GPU runs earlier segments; the removed work is
evidently off the critical path there. That is an inference from the numbers,
not a measurement of the overlap.

### Slot-indexed pager (2026-11-13)

Step 2, measured on the Windows box (Ryzen 9 9950X3D, RX 7900 XTX). **Busy
machine:** other sessions shared it and no quiet window was arranged. The `HEAD`
and changed builds of the replay benchmark were alternated, three runs each;
each cell is the range over the three (one value where they agreed). Recorded
traces are step 1's `moe`, `moe-bigdram`, `dense` and `cpu`, and two recorded
with group touches (`after-18g`: `paging.cache=18g`; `after-4g`:
`paging.cache=4g`, `paging.dram=12g`). Every recorded pool replayed to its
recorded outcomes under both builds. A selection of the rows:

| trace / pool                                          | slots | ns/call before |       after | ns/hit before |     after | ns/miss before |   after |
| ----------------------------------------------------- | ----: | -------------: | ----------: | ------------: | --------: | -------------: | ------: |
| generated: cyclic sweep 40 blocks x32                 |     8 |      55.4–55.7 |   10.6–11.2 |     26.8–27.0 | 13.0–14.4 |        133–135 |      27 |
| generated: cyclic sweep 40 blocks x32                 |    32 |      42.6–42.8 |         5.3 |     24.8–24.9 | 11.6–12.9 |        294–295 |      23 |
| generated: moe prefill x2, 48x128 blocks              |  3072 |    12932–13033 |   49.8–50.1 |     15.6–15.8 |       4.8 |    17200–17303 |   63–64 |
| generated: moe decode resident x9, 48x128 blocks      |  6144 |      27.1–27.6 |     6.3–7.2 |     23.5–23.8 |   6.1–6.2 |          59–62 |      10 |
| generated: moe decode top-8 zipf 1 x64, 48x128 blocks |  1536 |      4437–4460 |   13.7–13.9 |     31.2–32.1 | 12.4–12.5 |    10993–11080 |      24 |
| generated: moe prefill + decode x64, 48x128 blocks    |  3072 |      6784–6800 |   17.6–17.7 |     29.8–30.4 | 10.3–10.5 |    18951–19094 |      37 |
| generated: moe prefill x2, 43x256 blocks              |  2752 |    17748–17788 | 111.3–111.6 |     15.2–15.4 |   4.7–4.8 |    20055–20219 | 126–127 |
| generated: moe prefill x2, 43x256 blocks              |  8256 |    18050–18225 |   43.2–44.4 |     17.9–19.2 |   4.7–4.8 |    28417–28584 |   66–68 |
| generated: moe decode top-8 zipf 1 x64, 43x256 blocks |  2752 |      5958–6027 |   13.8–14.1 |     33.1–33.6 | 12.1–12.6 |    16037–16103 |   26–27 |
| moe down/1290240B host [moe.trace]                    |   187 |    483.6–487.3 |   12.8–13.1 |             — |         — |        962–967 |      29 |
| moe gate/884736B [moe.trace]                          |  1315 |      3549–3574 |   21.0–21.5 |     27.1–27.2 |   9.3–9.5 |      9787–9936 |   47–48 |
| moe gate/884736B host [moe-bigdram.trace]             |  4510 |      2396–2407 |     8.8–8.9 |     42.7–44.5 | 17.9–18.6 |      8050–8283 |   22–23 |
| dense pool0/2359296B [dense.trace]                    |     9 |      60.3–60.7 |   11.3–11.7 |     26.5–26.8 | 12.9–14.0 |        141–143 |   27–28 |
| dense pool2/1720320B [dense.trace]                    |     2 |      38.8–39.1 |   11.2–11.4 |     32.0–32.2 | 14.5–16.0 |          90–91 |      25 |
| cpu 2359296B [cpu.trace]                              |    16 |      39.5–40.8 |     9.0–9.4 |     35.1–35.5 | 14.0–14.5 |        161–162 |   28–29 |
| moe gate/884736B host [after-18g.trace]               |  6144 |      28.7–29.3 |     8.3–8.4 |     39.9–40.3 | 14.5–15.3 |             53 |      18 |
| moe gate/884736B [after-18g.trace]                    |  6013 |      1287–1298 |     8.2–8.3 |          18.1 |   5.7–5.8 |      5265–5293 |      16 |
| moe gate/884736B host [after-4g.trace]                |  4510 |      2508–2524 |     8.9–9.0 |     42.9–43.5 | 17.9–18.3 |      8045–8088 |      22 |
| moe gate/884736B [after-4g.trace]                     |  1315 |      4214–4235 |   25.0–25.1 |     28.7–29.5 | 10.8–11.2 |    10071–10113 |   50–51 |

Over all recorded pools:

- **Misses no longer scan.** MoE VRAM pools: 2.8–10.1 µs per miss becomes 15–51
  ns; host pools 52 ns–8.3 µs becomes 17–32 ns; dense 90–143 ns becomes 25–28
  ns; CPU 102–182 ns becomes 28–30 ns. The generated DeepSeek-scale prefill,
  where a batch walks past its own cold-end inserts, is the most expensive at
  66–127 ns (20–31 µs before).
- **Hits cost half or less on every pool, ungrouped ones included**: dense
  26.5–32.5 → 12.5–16.0 ns, CPU 35.0–38.5 → 14.0–16.0 ns, host 38.3–44.5 →
  14.2–18.6 ns, MoE VRAM 17.7–29.5 → 5.6–11.2 ns. A hit is one FxHash lookup, an
  epoch write and a list move, where it was several SipHash map operations.
- **The group touch moved with the build, not with its own code.** Resident
  decode with group touches, one pool, per token (a token is 48 `begin_batch`
  and 48 `touch_group` calls on 48 × 128, 43 of each on 43 × 256):

  | build                                   | 48 × 128, µs | 43 × 256, µs |
  | --------------------------------------- | ------------ | ------------ |
  | `HEAD`, alternated with the first build | 2.200–2.212  | 3.625–3.638  |
  | first build                             | 2.413–2.425  | 3.688–3.737  |
  | `HEAD`, alternated with the final build | 2.200–2.237  | 3.550–3.625  |
  | final build                             | 2.038–2.050  | 3.175–3.188  |

  The first build was 0.20–0.23 µs per token slower on 48 × 128, 4.2–4.7 ns per
  group touch, and 1.2–2.6 ns per touch on 43 × 256; trimming its duplicate
  group lookup changed nothing (2.400–2.413 µs). The final build differs from it
  only in the hash's finishing rotate and the pin assertion in `take_slot`,
  neither on the group-touch path, and is 0.15–0.19 µs per token faster than
  `HEAD`. So a few nanoseconds per group touch is within what code placement
  moves here; the node move the touch now makes is not a measurable cost.
  Per-expert resident touches went from 144–148 µs to 36 µs per token.

- **The `id → slot` map:** a direct-index table in its place, alternated with
  the FxHash map twice on the generated shapes, was 1–4 ns per call faster
  (dense sweep at 8 slots 6.8–6.9 against 11.1–11.2 ns per call; routed decode
  at 1536 slots 10.2–10.3 against 13.5–13.8). Not taken; see step 2.
- **The pin assertion and the finishing rotate** (added after review) cost
  nothing visible: the final build against the first, alternated three times,
  ns/miss: dense sweep at 8 slots 24–26 against 24–27, MoE prefill at 3072 slots
  60–63 against 62–64, routed decode at 1536 slots 24–25 against 22–24,
  DeepSeek-scale prefill at 8256 slots 65–66 against 65–66; ns/hit on the dense
  sweep 12.6–13.3 against 11.3–14.2. The table above is from the first build.
- **B164 is gone.** Its harness (a 6144-id cold sweep through 3072 slots, and
  MRU hits on a full 6144-slot pool, driving `Pager` from `infr-core` and the
  same source built as a crate of its own, in one binary) reproduces the gap at
  `HEAD` (sweep 12.58–12.61 µs against 8.40–8.46 µs, hit 22.65–22.80 ns against
  16.15–16.18 ns) and shows none with the change (sweep 11.4–11.7 ns against
  11.6–11.7 ns, hit 5.29–5.32 ns against 5.33–5.66 ns). Its cause was never
  identified.

**End to end**, Qwen3-30B-A3B Q4_K_M on the RX 7900 XTX,
`infr bench -p 0 -n 48 -d 365 -r 3 --set paging.cache=4g --set paging.dram=12g`,
binaries alternated (busy machine):

| build   | tg48 @ d365, t/s: three runs of three reps |
| ------- | ------------------------------------------ |
| `HEAD`  | 75.9, 75.1, 76.1                           |
| changed | 82.6, 82.5, 82.7                           |

About 9% faster decode, in line with step 1's estimate that the pager was 6.3%
of `moe-bigdram`'s decode tokens. One `infr run` of the same model and
configuration (the 365-token prompt, 48 greedy tokens, `paging.stats`) gave
byte-identical output and identical hit, miss and eviction counts on all eight
pools under both binaries. The 18 GB configuration was not rerun.

## Steps

### 1. Measure before designing (done)

The reading half is "What the survey found". The measuring half is three tools,
kept in the tree so every later step is compared the same way:

- **The trace:** `paging.trace` (`INFR_PAGER_TRACE`) names a file; every pager
  pool of the run (VRAM expert and streaming arenas, the DRAM tier under them,
  the CPU backend's pools) writes its calls there. The recorder lives inside
  `Pager`, given to a pool at construction (`Pager::trace_to`); the format is in
  `infr_core::pager::trace`'s module doc. A trace replays to the exact outcome
  of every call (`trace::replay`, tested), so its counts equal the run's
  `paging.stats` lines.
- **The simulator:** `infr_core::pager::sim`. `simulate` owns residency, the
  batch guard, pins and counting; a `Policy` only orders blocks and names a
  victim. `Recency` is today's pager (checked against the real one on every
  trace it is run on), `Belady` the lower bound under the same constraints for
  traces without probes or repins (`belady_is_a_bound`; see "What a Belady
  figure means here"). A candidate for step 3 is one more `Policy` impl.
- **The benchmark:** `crates/infr-core/tests/pager_bench.rs`, `#[ignore]`d.
  `generated_traces` replays the shapes in `infr_core::pager::shapes` (dense
  sweep, MoE prefill, resident and routed decode, prefill plus decode, at
  Qwen3-30B-A3B and DeepSeek-scale pool sizes); `recorded_traces` replays the
  files in `INFR_TEST_PAGER_TRACE`; `recorded_trace_counters` prints a recorded
  trace's replayed counters beside which to set the run's `paging.stats`.

To reproduce the measurements below:

```text
infr run <model> "<prompt>" --dev Vulkan0 --temp 0 --max-new 48 \
  --set paging.cache=4g --set paging.dram=1g --set paging.stats=true \
  --set paging.trace=moe.trace
cargo test --release -p infr-core --test pager_bench -- --ignored --nocapture \
  --test-threads=1 generated_traces
INFR_TEST_PAGER_TRACE=moe.trace cargo test --release -p infr-core \
  --test pager_bench -- --ignored --nocapture --test-threads=1 recorded_traces
```

Timing numbers still belong in a quiet window, per the repo's measurement rules;
the miss counts are exact on any machine.

### 2. Data-structure refactor (cost per touch) (done)

What landed (the data structure is under "What the code does today"):

- **Slot-indexed state**: the block, its epoch and its pin count live in an
  array indexed by slot, and recency in an intrusive doubly linked list over the
  slots (`pager::lru`). An MRU hit is a few link rewrites; a victim is a walk
  from the cold end to the first unprotected slot.
- **Group touches in the list**: a group's touch moves one node for the group,
  which ranks its members in id order until they leave the run (an MRU touch, or
  leaving the cache) or until an admission into the group places them
  individually where the node was (`pager::groups`' module doc).
- **The `id → slot` map is a `HashMap` keyed by FxHash** (rustc-hash 1.x's
  multiply-rotate step, finished with rustc-hash 2.x's rotate left by 26 so that
  ids differing only in high bits, such as `layer << 16 | expert`, do not share
  a bucket; in-tree ids are dense from 0), not a direct-index table. The table
  was measured (below) and is 1–4 ns per call faster, but sizes itself to the
  largest id a caller names, and `BlockId` is opaque to the pager; the hash
  keeps memory bounded by `n_slots` at that cost.
- **Not built: batch-aware victim selection.** With the walk, a batch of k
  cold-end misses costs k²/2 steps past its own inserts; on the generated
  DeepSeek-scale prefill that is 66–127 ns per miss in all, against 20–31 µs
  before, so a batched entry point would save nanoseconds beside an upload.
- **Equivalence, mechanically**: the old implementation is kept, test-only, as
  `pager::reference`, and `matches_the_reference_on_random_calls` and
  `matches_the_reference_on_session_shapes` drive both with the same calls
  (every public mutating call; groups of power-of-two and other lengths, grouped
  from the start and part-way; batches up to wider than the pool; the legacy
  no-batch path; pins held across batches) and compare every outcome, every
  counter, `slot_of` of every id, `resident_count`, `pinned_blocks` and
  `group_is_resident`. Twelve deliberate breakages of the new structure, each
  restored after, and the tests that went red (`random` and `shapes` are the two
  differential tests):
  - cold-end insert at the hot end: `random`, `shapes`, `sim`'s driver check,
    and four tests of cold sweeps and `schedule`'s prefix;
  - a cold hit that promotes: `random` only;
  - `take_slot` ignoring the batch guard: `random`, `shapes`, `sim`'s driver
    check, `a_reused_slot_does_not_keep_the_victims_epoch`,
    `batch_larger_than_cache_fails_loudly` and the two batch-sibling tests;
  - a group's run walked in reverse id order: `random` and two `groups` tests;
  - the group's batch guard ignored: `random` and
    `a_group_touch_protects_its_members_for_the_batch`;
  - an MRU touch not leaving the group's run, an admission not moving the run's
    members where the node was, and the group cursor skipping protected members:
    each `random` and `a_group_touch_is_a_touch_of_every_member`;
  - an admission into a touched group dropping the group's epoch: `random` and
    `an_admission_into_a_touched_group_keeps_the_groups_batch_protection`;
  - a new block keeping its slot's previous epoch: `random`, `shapes`, `sim`'s
    driver check, `a_reused_slot_does_not_keep_the_victims_epoch`,
    `a_group_touch_is_a_touch_of_every_member` and three batch tests;
  - no pin check in `take_slot`:
    `an_infallible_miss_never_evicts_a_pinned_block` (the differential tests
    keep to the contract there);
  - no finishing rotate in the hash: the two hash tests.

  The existing tests pass, with one rewritten:
  `eviction_drops_the_victim_epoch_entry` checked a per-id epoch map that no
  longer exists (an epoch lives in its slot), so it could not fail;
  `a_reused_slot_does_not_keep_the_victims_epoch` checks what can now go wrong
  instead. Every recorded trace replays to its recorded outcomes.

- **A pinned victim on the infallible path panics.** `touch`, `touch_cold` and
  `schedule` do not look at pins; the old pager only `debug_assert`ed that no
  pin was held when they evicted, so a release build evicted a pinned block
  under its reader. `take_slot` now asserts, in every build, that its victim
  holds no pin. No in-tree caller mixes pins with those entry points
  (`HostPager` pins through `resolve_and_pin` and its probes only).

### 3. Policy (misses per trace)

Only where the simulator shows a gap to Belady worth closing:

- **MoE routing is skewed**, and recency alone lets a prefill sweep or a burst
  of rare experts displace the hot set (the cold-end insert and the admission
  doorkeeper in `HostPager` are partial answers). Candidates to simulate: a
  frequency-aware admission filter (TinyLFU-style), S3-FIFO, ARC, and LFU with
  aging. Pick by miss count on the recorded traces, then by bookkeeping cost.
- **Tier-coordinated placement for MoE:** each tier caches by the same policy,
  and the DRAM tier sees only VRAM's misses, so it holds the experts most
  recently missed rather than a mirror of VRAM's hot set
  (`disk-streaming-plan.md` §3.6 says what that costs). Exclusive placement
  would make DRAM hold the next-hottest set instead (B36 lists it as unbuilt).
  Simulate the two-tier miss counts before building.
- **Dense stays as it is:** `schedule`'s stable prefix is within
  `(n_slots − 1)/(n_blocks − 1)` misses per pass of Belady in steady state (0.5–
  5.8% of misses on the recorded dense pools), and the planned partition
  (`TierPlan`) is a placement question, not an eviction one.

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
