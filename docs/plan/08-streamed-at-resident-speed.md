# Phase 08 — a streamed model that fits runs at resident speed

**Goal:** on Vulkan, a dense model that streams (disk → host tier → VRAM arena)
but whose blocks all have a slot runs, after its first pass, at nearly the speed
of the resident path, with nothing given up when blocks do not fit.

**Why:** the streamed path engages whenever a budget is capped or placement
decides a model is borderline, and it cost throughput even when nothing was ever
evicted. The recorded case is in `backlog.md` B193 (ERNIE-4.5-0.3B F16, RX 7900
XTX, every block resident: prefill and decode both well below the resident
figures, same kernels).

**Status:** steps 0–4 done (2026-11-15). The findings below were read from the
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
2. **Decode never replayed and never chained.** `adapter::replays` was false
   while `dense_paged()`, and the runner's `mirror_admits` repeated the
   condition, so every token rebuilt, recompiled, re-recorded and waited. Fixed
   by step 3 for a cache that holds every block.
3. **Quantised streamed Linears skipped resident kernel tiers.** Fixed by step
   4: the streamed route had its own decision tree, which left out the int8
   K-quant coopmat GEMM, the multi-row GEMVs and every decode GEMV tier but the
   plain one. It was the whole remaining Q4_K_M prefill gap on the RX 7900 XTX,
   and most of the streamed Q4_K_M decode gap on both GPUs.
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
3. ~~Replay and chain decode while placements are stable.~~ Done: a dense
   streaming session whose cache holds every block replays and chains its decode
   like a resident one; see below.
4. ~~Resident kernel routing for quantised streamed Linears.~~ Done; see below.
5. **Keep Linear+Add fused for streamed weights**, through the address-taking
   variants, in the static path and in the replay. Observable: a streamed decode
   forward has the same dispatch count as a resident one.
6. ~~Find the Qwen3 decode residue~~ (24 t/s on the RX 7900 XTX, and a third of
   resident on the iGPU). It was the streamed GEMV route, removed by step 4:
   streamed Qwen3 decode holding every block went from 308 to 387 t/s on the RX
   7900 XTX (resident with replay: 479), and from 9.3 to 26.9 on the iGPU
   against 27.5 resident. Whether anything is left once replay lands is step 3's
   to measure.
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

## Results of step 4

What the two routes did before, by device and row count, for a native quantised
weight (F16/F32 already shared `float_linear`):

| Rows | Resident, coopmat device                                                                                                                                       | Streamed before, coopmat device               | Resident, no coopmat                   | Streamed before, no coopmat |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | -------------------------------------- | --------------------------- |
| 1    | multi-warp int8 GEMV where wave32-native, else the int8 mmv (opt-in), else the SG → RM-v2 → RM → plain dequant GEMV                                            | plain dequant GEMV                            | same as coopmat                        | plain dequant GEMV          |
| 2–16 | the int8 multi-row GEMV (m ≥ 3, weight ≥ 8M elements, int8 dot) or the dequant one (m ≤ 8), per `MROW_BANDS`                                                   | plain GEMV per row                            | same                                   | plain GEMV per row          |
| > 16 | opt-in `_cm8`/fp8/int8 Q8_0/bf16 coopmat tiers, then the int8 K-quant coopmat GEMM (Q4_K/Q5_K/Q6_K, AMD proprietary), then narrow Q4_K mmq, then the warp tile | narrow Q4_K mmq (or dp4a), else the warp tile | dp4a K-quant GEMM (m > 64) or mmq, fma | same as resident            |

Now both run one function, `adapter::quant_linear`, with the weight given as an
address plus the resident buffer when there is one. A streamed slot skips only
the opt-in tiers whose kernel binds the weight buffer (`_cm8`, fp8, int8 Q8_0
and bf16 coopmat), reads the narrow Q4_K mmq through its arena-addressed build,
and keeps the chunked per-row GEMV for a weight past the u32 addressing cap or a
non-coopmat grid past the device's x limit (`tiled_gemm_fits`). A resident
weight takes that GEMV too at up to the multi-row band's 16 rows, where the
multi-row GEMVs, which are not chunked, used to take a breaching weight; past 16
rows the resident Linear still fails loudly, as before. With `INFR_CM_8X8`,
`INFR_F8_COOPMAT`, `INFR_I8_COOPMAT` or `INFR_BF16_COOPMAT` set, a streamed
Linear takes the next tier down from its resident twin, so the bit-identity
below holds on the default configuration only. The quantised-activation memo is
keyed by the activation only and its pooled scratch lives for the whole forward,
so staging copies, early-submit cuts and ring rotations between two Linears that
share `x` do not invalidate it; `streamed_quant_takes_the_resident_kernels`
counts one quant pass per execution over three Linears on an evicting pool.

Observable: `VulkanBackend::i8cm_kq_counts` and the new `linear_tier_counts`
(multi-row int8 and dequant GEMVs, dp4a K-quant GEMMs), asserted equal between a
streamed and a resident run in `streamed_quant_takes_the_resident_kernels`
(`infr-vulkan/tests/streamed_nc_gemm.rs`), with bit-identical outputs. On the
tree before the step it fails on both GPUs: on `Vulkan0` the m=200 Q4_K case
counts no int8 coopmat tile streamed against six resident, and the m=1 case
differs in the first output element; on `Vulkan1` the m=4 case counts no
multi-row GEMV streamed against six resident.

Measured with the step 0 harness, the tree before the step and after it
interleaved in alternating order over two rounds on the RX 7900 XTX, host CPU
load 0–9% and at least 36.8 GB of host memory free at every start. "Holds all"
is `INFR_CACHE=2G` for both models (`INFR_PAGER_STATS=1` showed every pool at
`slots=n/n`, no evictions); the evicting control is 800M for Qwen3-1.7B and 1G
for SmolLM3-3B. t/s, round 1 / round 2:

| Case          | before        | after         |
| ------------- | ------------- | ------------- |
| Qwen3 pp512 R | 12031 / 11818 | 11841 / 11798 |
| Qwen3 pp512 S | 7659 / 7576   | 12122 / 12037 |
| Qwen3 pp512 E | 7508 / 7480   | 11893 / 11889 |
| Qwen3 tg128 R | 479.1 / 478.5 | 479.1 / 478.2 |
| Qwen3 tg128 S | 308.3 / 308.0 | 387.4 / 387.5 |
| Qwen3 tg128 E | 58.4 / 57.9   | 59.3 / 58.7   |
| Smol pp512 R  | 6289 / 6245   | 6299 / 6260   |
| Smol pp512 S  | 4033 / 4042   | 6344 / 6320   |
| Smol pp512 E  | 3986 / 3968   | 6111 / 6092   |
| Smol tg128 R  | 228.0 / 228.2 | 228.0 / 228.3 |
| Smol tg128 S  | 172.2 / 172.1 | 251.6 / 250.7 |
| Smol tg128 E  | 21.90 / 21.72 | 21.78 / 21.57 |

The integrated GPU (`Vulkan1`), one round, same budgets, host CPU load 0–3%:

| Case          | before | after |
| ------------- | ------ | ----- |
| Qwen3 pp512 R | 231.5  | 231.8 |
| Qwen3 pp512 S | 231.7  | 231.7 |
| Qwen3 pp512 E | 214.4  | 214.2 |
| Qwen3 tg128 R | 27.47  | 27.45 |
| Qwen3 tg128 S | 9.28   | 26.85 |
| Qwen3 tg128 E | 7.23   | 13.64 |
| Smol pp512 R  | 120.3  | 119.8 |
| Smol pp512 S  | 120.4  | 119.6 |
| Smol pp512 E  | 110.6  | 110.8 |
| Smol tg128 R  | 12.90  | 12.66 |
| Smol tg128 S  | 4.64   | 12.68 |
| Smol tg128 E  | 3.45   | 5.48  |

Streamed prefill is now at the resident rate on the discrete GPU, holding every
block or evicting (one 512-token chunk reads one sweep either way). On the iGPU
prefill already took the same non-coopmat GEMMs, and nothing moved. Streamed
decode gains on both GPUs; holding every block it now sits 2.2% (Qwen3) and
0.2–1.7% (SmolLM3) below resident on the iGPU. On the RX 7900 XTX the replay
(step 3) is the expected rest of the gap; this window did not split it. The
evicting decode on the RX 7900 XTX moved within run noise (SmolLM3 0.5–0.7%
lower, Qwen3 1.4% higher). The resident rows did not change in code; the 1.9%
SmolLM3 decode difference on the iGPU is one round of noise, not retaken.

## Results of step 3

What was built:

- **Admission.** `adapter::replays` admits a dense streaming session when
  `DensePagerSession::holds_every_block`, and the runner's `mirror_admits` asks
  `Backend::dense_holds_every_block`. A cache that cannot hold every block keeps
  the static path, unchanged; so does a model the replay cannot express
  (SmolLM3's NoPE layers, B157).
- **Placement epoch.** `GpuPager::placements` counts every placement change (a
  miss placing its block and its eviction, a rolled-back miss, a placement
  `forget_unsubmitted` takes back); `DensePagerSession::placement_epoch` sums it
  over the pools. A decode recording stores the epoch its slot addresses were
  read at (`DecodeReplay::epoch`).
- **Before every replay or chained submit** (`adapter::current_replay`, called
  from `execute` and `execute_chain`), the epoch is read under the dense
  session's lock; a moved epoch drops the recording and records again, as a
  rebound buffer does, and the lock is held until the replay has completed.
- **Staging pass.** Before recording, any streamed weight of the graph that is
  not resident yet is staged in a forward of its own (`stage_for_replay`,
  through `DensePagerSession::stage`, blocking submit). The recording then reads
  each slot address through `DensePagerSession::resident_addr`, which changes
  nothing, and fails with an error if a streamed weight is not in that set.
- **Fusion.** The recording drops the Linear+Add fusion of a streamed weight
  through the same helper the static path uses (`unfuse_streamed_adds`); keeping
  the pairs fused is still step 5.
- **Statistics.** A replayed step calls no `stage`, so it counts no hit and
  moves no recency; with a slot per block there is no victim to choose, so
  recency decides nothing. `paging.stats` and `dense_pager_totals` say so.

The ordering argument is on `DensePagerSession::stage`: a replay reads only
slots whose copies, with their trailing arena barrier, were submitted and waited
before the recording was made, and that barrier's second scope covers the
replay's pointer reads in either `INFR_NO_GPU_POS` mode; a slot a recording
reads can be written only after its block leaves it, which moves the epoch. It
assumes executes on one backend are serialised, which `serve --parallel` does
with its step baton; the execute sites were traced and are listed in
`backlog.md` B193, and
`synthetic_streamed_parallel_sequences_rebuild_on_an_epoch_move` runs two
sequences on one engine across an epoch move. Every placement change is shown to
move the epoch by `every_placement_change_moves_the_dense_epoch`.

Observable: `decode_recordings()` reads one after a streamed decode that holds
every block and none on an evicting one
(`synthetic_streamed_decode_replays_while_every_block_is_held`, F16 and F32,
chain on and off, and `gpu_seam_dense_stream_holding_every_block_replays` on
Qwen3-1.7B Q4_K_M and ERNIE-4.5-0.3B F16), two after a forced epoch move
(`synthetic_streamed_replay_records_again_when_placements_move`,
`VulkanBackend::force_dense_epoch_move_after`), and a one-token prompt on an
empty cache stages every block once, runs no static forward and replays from one
recording (`synthetic_streamed_replay_stages_an_empty_cache_first`). Tokens are
the resident run's and the streamed run's with `INFR_SEAM_NO_REPLAY` in every
case.

Measured with the step 0 harness, the tree before the step and after it
interleaved in alternating order over two rounds on the RX 7900 XTX, host CPU
load 0–4% at every start but one (26%, the first round's evicting ERNIE decode
on the tree before), and at least 35.8 GB of host memory free. "Holds all" is
`INFR_CACHE=2G` for Qwen3-1.7B and SmolLM3-3B and 1G for ERNIE-4.5-0.3B; the
evicting control is 800M, 1G and 512M. t/s, round 1 / round 2:

| Case          | before        | after         |
| ------------- | ------------- | ------------- |
| Qwen3 tg128 R | 479.3 / 478.5 | 479.0 / 478.8 |
| Qwen3 tg128 S | 386.4 / 385.5 | 469.4 / 469.1 |
| Qwen3 tg128 E | 59.1 / 58.7   | 58.8 / 58.6   |
| Qwen3 pp512 R | 11953 / 11904 | 12020 / 11857 |
| Qwen3 pp512 S | 12212 / 12079 | 12232 / 12094 |
| Qwen3 pp512 E | 11910 / 11871 | 11999 / 11885 |
| ERNIE tg128 R | 449.9 / 449.5 | 449.7 / 449.9 |
| ERNIE tg128 S | 384.7 / 384.8 | 444.3 / 444.2 |
| ERNIE tg128 E | 71.2 / 71.5   | 72.6 / 71.5   |
| ERNIE pp512 R | 22141 / 22261 | 22330 / 22046 |
| ERNIE pp512 S | 22287 / 22409 | 22805 / 22767 |
| ERNIE pp512 E | 22050 / 21768 | 21804 / 21871 |
| Smol tg128 R  | 228.6 / 227.1 | 228.2 / 228.1 |
| Smol tg128 S  | 251.8 / 251.0 | 251.2 / 251.5 |
| Smol tg128 E  | 21.7 / 21.6   | 21.6 / 21.6   |
| Smol pp512 R  | 6269 / 6275   | 6274 / 6258   |
| Smol pp512 S  | 6333 / 6319   | 6330 / 6303   |
| Smol pp512 E  | 6080 / 6071   | 6094 / 6075   |

The integrated GPU (`Vulkan1`), Qwen3-1.7B tg128, one round, same budgets, host
CPU load 0–6%:

| Case          | before | after |
| ------------- | ------ | ----- |
| Qwen3 tg128 R | 27.39  | 27.42 |
| Qwen3 tg128 S | 26.81  | 27.54 |
| Qwen3 tg128 E | 12.95  | 12.92 |

Streamed decode holding every block is now within 2.0% of resident for Qwen3
(469 against 479) and 1.2% for ERNIE (444 against 450). Fusion (step 5) is the
known difference left: step 0 measured it at about 1% of ERNIE's resident decode
without replay and about 3 t/s of Qwen3's. SmolLM3 does not replay (NoPE) and is
unchanged, as are prefill and the evicting budget, within run noise.

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
