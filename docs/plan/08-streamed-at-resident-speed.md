# Phase 08 — a streamed model that fits runs at resident speed

**Goal:** on Vulkan, a dense model that streams (disk → host tier → VRAM arena)
but whose blocks all have a slot runs, after its first pass, at nearly the speed
of the resident path, with nothing given up when blocks do not fit.

**Why:** the streamed path engages whenever a budget is capped or placement
decides a model is borderline, and it cost throughput even when nothing was ever
evicted. The recorded case is in `backlog.md` B193 (ERNIE-4.5-0.3B F16, RX 7900
XTX, every block resident: prefill and decode both well below the resident
figures, same kernels).

**Status:** steps 0–2 done (2026-11-15). The findings below were read from the
code first; step 0 measured them, and the order of the remaining steps follows
that measurement.

## What step 0 measured

`infr bench <gguf> --dev Vulkan0 -r 3 --json`, `-p 512` for prefill and
`-p 0 -n 128` for decode, on the RX 7900 XTX, two interleaved rounds, host CPU
load 0–6% at every start and at least 37 GB of host memory free. "Holds all" is
a `paging.cache` at which every pool has a slot per block (`INFR_PAGER_STATS=1`
showed `slots=n/n` and no evictions): 1G for ERNIE-4.5-0.3B F16, 2G for
Qwen3-1.7B Q4_K_M. The evicting control is 512M and 800M. "nobar" is a temporary
build that skipped both arena barriers on a block already resident (all of them,
after the first pass, at a holds-all budget); it paid an environment lookup per
streamed Linear that the shipped change does not. Figures are t/s, round 1 /
round 2.

| Case                                    | ERNIE pp512   | ERNIE tg128   | Qwen3 pp512   | Qwen3 tg128   |
| --------------------------------------- | ------------- | ------------- | ------------- | ------------- |
| resident                                | 22607 / 21626 | 449.4 / 450.1 | 12110 / 11896 | 478.8 / 478.8 |
| resident, `INFR_SEAM_NO_REPLAY=1`       |               | 380.1 / 380.9 |               | 331.0 / 331.7 |
| resident, `INFR_DECODE_CHAIN=0`         |               | 448.3 / 448.4 |               | 477.0 / 476.6 |
| resident, no replay, `INFR_NO_FUSE_ADD` |               | 375.2 / 376.7 |               | 329.9 / 327.4 |
| streamed, holds all                     | 8227 / 8454   | 357.7 / 357.9 | 4149 / 4077   | 293.0 / 292.4 |
| + `paging.layer_major=false`            | 22443 / 22261 |               | 7655 / 7604   |               |
| + `paging.early_submit=0`               | 7842 / 7883   | 344.2 / 343.8 | 3923 / 3870   | 271.8 / 268.7 |
| + layer-major off, early submit off     | 21929 / 22334 |               | 7571 / 7523   |               |
| + nobar (layer-major off for pp)        | 22529 / 22624 | 385.0 / 384.7 | 7717 / 7629   | 307.9 / 307.7 |
| + nobar, early submit off               | 22752 / 22200 | 371.0 / 371.4 | 7582 / 7544   | 291.0 / 291.4 |
| streamed, evicting                      | 6719 / 6972   | 72.2 / 72.0   | 3803 / 3756   | 58.3 / 57.8   |

What that splits the gap into:

- **Prefill, ERNIE:** layer-major on a one-chunk prompt is the whole gap (8.3k →
  22.4k against 22.1k resident). The barriers and early submit are within run
  noise.
- **Prefill, Qwen3 Q4_K_M:** layer-major is 3.5k of the 7.9k gap. The rest is
  the int8 cooperative-matrix K-quant GEMM the streamed path does not route to
  (finding 3): the resident run with `INFR_NO_I8CM_KQ=1` prefilled 7536 / 7562
  against 7582 / 7603 streamed with layer-major off, interleaved. Barriers are
  about 1%.
- **Decode, ERNIE:** the record-once replay is worth 69 t/s (449.8 → 380.5) and
  the chain within it 1.4; the barriers are worth 27 (357.8 → 384.9). Streamed
  without the barriers then sits at the resident no-replay figure, so replay and
  barriers are the whole decode gap. Linear+Add fusion is about 1% (380.5 →
  376.0).
- **Decode, Qwen3:** replay is worth 147 t/s (478.8 → 331.4), the barriers 15
  (292.7 → 307.8). That leaves 24 t/s between streamed without barriers and
  resident without replay, which fusion (about 3) does not explain. Not
  diagnosed: the streamed GEMV route for a K-quant (`linear_native_at`) against
  the resident decode GEMV choices, and the per-op staging bookkeeping, are the
  two unmeasured candidates.
- **Early submit** is a gain on every streamed decode measured here (ERNIE 357.8
  against 344.0 with it off, Qwen3 292.7 against 270.2) and no loss on prefill,
  so step 7 has nothing to take back yet.

The measurement changes the order of what is left: replay (step 3) is the
largest decode lever on both models, and the K-quant prefill routing (now
step 4) is larger than fusion (now step 5), which is about 1%.

## Findings

Extra work the streamed path did per forward once every block was in VRAM, in
the order step 0 found them to cost:

1. **Layer-major prefill ran even when it saves no reads.** Fixed by step 1.
2. **Decode never replays and never chains.** `adapter::replays` is false while
   `dense_paged()`, and the runner's `mirror_admits` repeats the condition, so
   every token rebuilds, recompiles, re-records and waits. The largest remaining
   decode term on both models.
3. **Quantised streamed prefill skips resident kernel tiers.**
   `streamed_gemm_route` does not consider the int8 K-quant coopmat GEMM
   (`matmul_i8cm_kq` takes a weight address) nor the multi-row kernels for small
   row counts. Measured as the whole remaining Q4_K_M prefill gap on the RX 7900
   XTX.
4. **Two arena barriers per streamed Linear, hit or miss.** Fixed by step 2.
5. **Linear+Add is unfused for streamed weights** (`execute_static`), although
   `Recorder::linear_add_at`, `linear_add_native_at` and `linear_add_mmv_at`
   take an address. About 1% of resident decode with replay off.
6. **Early-submit cadence** is active only on paged paths, and a net gain there.
7. **The paged MoE path still records its arena barriers on a hit.**
   `adapter::execute_paged_moe` records one `arena_stream_barrier` before and
   one after each paged `MoeFfn`'s staging whether or not anything was copied,
   so a wholly resident expert layer pays two full barriers. Not measured, not
   changed here; the same copies-only rule is a possible later step for the MoE
   path.

Checked and found the same on both paths: fused QKV and gate/up decisions, KV
handling, per-op pager bookkeeping on a hit (tens of nanoseconds, plan 06), and
statistics, which are opt-in.

Already declined with evidence in B185 and plan 06, not to be re-proposed here:
busy-polling the fence at drains, splitting the staging copy across threads,
carrying the cadence across a ring rotation, a fixed-count cadence.

## Steps

0. ~~Split the gap with existing switches.~~ Done; above.
1. ~~Layer-major only when it can save reads.~~ Done:
   `seam::layer_major_this_prompt` runs it for a prompt of more than one chunk
   on a cache that cannot hold every block (`Backend::dense_holds_every_block`,
   answered by `DensePagerSession::holds_every_block`); `paging.layer_major`
   still forces it either way. The activation reserve stays priced for the
   session, since the arenas are sized before any prompt. Observable:
   `VulkanBackend::static_executes`, asserted per prefill in
   `synthetic_streamed_prefill_is_layer_major_only_where_it_saves_reads`.
2. ~~Barriers only around a copy that was recorded.~~ Done:
   `DensePagerSession::stage` records the two arena barriers around a miss's
   copy and none on a hit; its doc carries the ordering argument. Observable:
   `VulkanBackend::arena_barriers`, asserted in
   `synthetic_streamed_arena_barriers_only_around_copies`.
3. **Replay and chain decode while placements are stable.** Admit a dense-paged
   session to `replays()` when it holds every block. Add a placement epoch to
   the GPU pager, bumped on every placement change; record it in the replay and
   re-record when it moves. Before recording, stage any streamed weight that is
   not yet resident in its own submit. While recording, resolve each streamed
   weight to its slot address without mutating the pager, and fail rather than
   fall back if one is missing. A session that cannot hold every block keeps the
   current path. Re-check `DensePagerSession::stage`'s ordering argument, which
   assumes recordings are submitted in the order they were recorded. Observable:
   `decode_recordings()` reads one after a run, and two after a forced epoch
   change, with identical tokens; a decode that starts from an empty cache
   exercises the staging pass.
4. **Resident kernel routing for quantised streamed prefill:** the int8 K-quant
   GEMM and the multi-row kernels, under the resident gates. Observable: the
   existing int8 kernel counters move on a streamed prefill.
5. **Keep Linear+Add fused for streamed weights**, through the address-taking
   variants, in the static path and in the replay. Observable: a streamed decode
   forward has the same dispatch count as a resident one.
6. **Find the Qwen3 decode residue** (24 t/s on the RX 7900 XTX, and a third of
   resident on the iGPU, below) once replay lands, if it survives it.
7. **Only if a gap remains:** early submit off for static forwards of a session
   that holds every block, decided by measurement.

## Results of steps 1 and 2

Same harness and cases as step 0, the three builds interleaved in a rotating
order over two rounds: the tree before this phase, step 1 alone, and steps 1
and 2. RX 7900 XTX, host CPU load 0–7% at every start. t/s, round 1 / round 2.

| Case          | before        | step 1        | steps 1+2     |
| ------------- | ------------- | ------------- | ------------- |
| ERNIE pp512 R | 22209 / 21932 | 22697 / 21986 | 22763 / 22134 |
| ERNIE pp512 S | 8505 / 8236   | 22412 / 22588 | 22614 / 22707 |
| ERNIE pp512 E | 6533 / 6676   | 22153 / 21864 | 21733 / 21903 |
| ERNIE tg128 R | 449.8 / 450.0 | 449.7 / 449.8 | 449.8 / 449.6 |
| ERNIE tg128 S | 357.5 / 358.3 | 357.7 / 358.3 | 385.3 / 383.4 |
| ERNIE tg128 E | 72.4 / 72.3   | 72.2 / 72.4   | 72.5 / 72.5   |
| Qwen3 pp512 R | 11917 / 11916 | 12083 / 12003 | 12159 / 11970 |
| Qwen3 pp512 S | 4148 / 4226   | 7696 / 7606   | 7711 / 7666   |
| Qwen3 pp512 E | 3832 / 3726   | 7547 / 7521   | 7586 / 7546   |
| Qwen3 tg128 R | 479.0 / 478.7 | 478.8 / 478.5 | 478.8 / 478.7 |
| Qwen3 tg128 S | 292.3 / 291.9 | 293.3 / 292.7 | 307.7 / 307.8 |
| Qwen3 tg128 E | 58.4 / 58.4   | 58.4 / 58.4   | 58.6 / 58.5   |

R is resident, S holds every block, E the evicting control. A 512-token prompt
is one chunk at the default 1024-row ubatch, so the evicting prefill gains too:
it reads one sweep either way and no longer pays a span per layer. Nothing
regressed. What is left: ERNIE prefill is at resident speed; ERNIE decode is the
replay (step 3); Qwen3 prefill is the int8 K-quant GEMM (step 4); Qwen3 decode
is the replay plus the 24 t/s residue (step 6).

Spot check on the integrated GPU (`Vulkan1`, Raphael iGPU, unified memory), one
round, same three builds, host CPU load 0–5%. The default chunk there is 128
rows, so pp512 is four chunks; at 1G / 2G every pool still holds every block.

| Case          | before | step 1 | steps 1+2 |
| ------------- | ------ | ------ | --------- |
| ERNIE pp512 R | 294.1  | 293.7  | 295.1     |
| ERNIE pp512 S | 294.2  | 292.2  | 292.7     |
| ERNIE pp512 E | 278.7  | 279.5  | 279.8     |
| ERNIE tg128 R | 37.59  | 37.60  | 37.59     |
| ERNIE tg128 S | 36.33  | 36.35  | 36.43     |
| ERNIE tg128 E | 15.00  | 14.85  | 15.06     |
| Qwen3 pp512 R | 231.9  | 231.9  | 231.9     |
| Qwen3 pp512 S | 223.7  | 231.9  | 231.9     |
| Qwen3 pp512 E | 215.8  | 215.2  | 215.4     |
| Qwen3 tg128 R | 27.44  | 27.43  | 27.44     |
| Qwen3 tg128 S | 9.26   | 9.26   | 9.28      |
| Qwen3 tg128 E | 7.42   | 7.37   | 7.40      |

The iGPU is GPU-bound, so the host work these steps remove barely shows, except
Qwen3's streamed prefill, now at the resident rate. Qwen3's streamed decode
there is a third of resident with nothing evicted (9.3 against 27.4 t/s); ERNIE
F16 shows no such gap. Not diagnosed; the streamed K-quant GEMV route is the
first suspect, as for the smaller Qwen3 residue on the discrete GPU.

## Verification

- Token-identical output against the resident path at every step, on a budget
  that holds every block and on one that evicts (`*_matches_resident`,
  `dense_tier_parity`, the float-stream tests).
- Each step's observable asserted in a test that fails without the step.
- Throughput before and after each step, streamed holding every block against
  resident, and an evicting budget that must not regress (B36's Qwen3-14B
  shape), on both GPUs, interleaved, with host load recorded.

## Not known

- What a barrier that names the transfer stage costs on the AMD driver, apart
  from the per-forward totals above.
- Whether address-taking variants exist for the multi-weight GEMV route used on
  other vendors' drivers.
- B36's Qwen3-14B evicting shape was not re-measured (the session was held to
  models under 4 GB); a multi-chunk prompt on an evicting cache still runs
  layer-major, so step 1 does not touch it, and step 2 removes barriers only on
  hits.
