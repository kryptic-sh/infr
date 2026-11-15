# Phase 08 — a streamed model that fits runs at resident speed

**Goal:** on Vulkan, a dense model that streams (disk → host tier → VRAM arena)
but whose blocks all have a slot runs, after its first pass, at nearly the speed
of the resident path, with nothing given up when blocks do not fit.

**Why:** the streamed path engages whenever a budget is capped or placement
decides a model is borderline, and today it costs throughput even when nothing
is ever evicted. The recorded case is in `backlog.md` B193 (ERNIE-4.5-0.3B F16,
RX 7900 XTX, every block resident: prefill and decode both well below the
resident figures, same kernels).

**Status of the analysis:** every finding below was read from the code on
2026-11-15 and none was measured. Step 0 measures them; reorder the later steps
by what it shows.

## Findings

Extra work the streamed path does per forward once every block is in VRAM, in
suspected order of cost.

1. **Layer-major prefill runs even when it saves no reads.** The runner uses it
   whenever `be.dense_paged()` unless `paging.layer_major` says otherwise
   (`generate_dense_backend`, `seam::layer_major_prefill`). With one chunk it
   cannot save I/O, yet it runs one span per layer, and each span repeats the
   host work the resident path does once: graph build, compile, bindings,
   `execute_static`, `alloc_scratch` with its zero-fill submit, recording and a
   blocking `submit_wait`. Leading suspect for prefill.
2. **Decode never replays and never chains.** `adapter::replays` is false while
   `dense_paged()`, and the runner's `mirror_admits` repeats the condition, so
   every token rebuilds, recompiles, re-records and waits, where the resident
   path records once and replays several tokens per submit. Leading suspect for
   decode.
3. **Two arena barriers per streamed Linear, hit or miss.**
   `adapter::stage_dense_linear` records `Recorder::arena_stream_barrier` before
   and after `stage` even when nothing is copied. Its comment calls that noise
   because streaming is transfer-bound, which is not true once every block is
   resident. It also serialises GEMVs the hazard tracker would let overlap.
4. **Linear+Add is unfused for streamed weights** (`execute_static`), although
   `Recorder::linear_add_at`, `linear_add_native_at` and `linear_add_mmv_at`
   take an address. One more dispatch per pair on decode.
5. **Quantised streamed prefill skips resident kernel tiers.**
   `streamed_gemm_route` does not consider the int8 K-quant coopmat GEMM
   (`matmul_i8cm_kq` takes a weight address) nor the multi-row kernels for small
   row counts. Not part of the f16 case; expected to matter for Q4_K/Q6_K
   prefill on the AMD proprietary driver.
6. **Early-submit cadence** is active only on paged paths. Measured as a net
   gain for a path that re-records every token (plan 06); expected to stop
   mattering for decode once item 2 is fixed.

Checked and found the same on both paths: fused QKV and gate/up decisions, KV
handling, per-op pager bookkeeping on a hit (tens of nanoseconds, plan 06), and
statistics, which are opt-in.

Already declined with evidence in B185 and plan 06, not to be re-proposed here:
busy-polling the fence at drains, splitting the staging copy across threads,
carrying the cadence across a ring rotation, a fixed-count cadence.

## Steps

0. **Split the gap with existing switches, in a quiet window, interleaved.**
   Streamed: `paging.layer_major=false`, `paging.early_submit=0`,
   `INFR_PROF_STAGES=1`. Resident: `INFR_SEAM_NO_REPLAY=1`,
   `INFR_DECODE_CHAIN=0`, `INFR_NO_FUSE_ADD`. Only the barriers need a temporary
   code change to measure. On an f16 and a Q4_K_M model, both GPUs.
1. **Layer-major only when it can save reads.** Use it when the session wants
   it, there is more than one chunk and the cache cannot hold every block; an
   explicit `paging.layer_major=true` still forces it. Needs a backend query for
   "every pool has a slot per block". Correct because both orders compute the
   same result and a cache that holds every block never evicts (confirm in
   `infr_core::pager` that a free slot is always taken before an eviction).
   Observable: a count of `execute_static` calls per prefill.
2. **Barriers only around a copy that was recorded.** Skip the pre-barrier on a
   hit and the post-barrier when `stage` copied nothing. Correct because with no
   copy there is no write to order against, and a slot's last write was already
   followed by its own barrier. Observable: an arena-barrier counter that reads
   zero on a second forward of a session that holds every block, and twice the
   misses on an evicting one. A missing barrier shows as wrong output, possibly
   intermittently, so the parity suites run several times.
3. **Replay and chain decode while placements are stable.** Admit a dense-paged
   session to `replays()` when it holds every block. Add a placement epoch to
   the GPU pager, bumped on every placement change; record it in the replay and
   re-record when it moves. Before recording, stage any streamed weight that is
   not yet resident in its own submit. While recording, resolve each streamed
   weight to its slot address without mutating the pager, and fail rather than
   fall back if one is missing. A session that cannot hold every block keeps the
   current path. Observable: `decode_recordings()` reads one after a run, and
   two after a forced epoch change, with identical tokens; a decode that starts
   from an empty cache exercises the staging pass.
4. **Keep Linear+Add fused for streamed weights**, through the address-taking
   variants, in the static path and in the replay. Observable: a streamed decode
   forward has the same dispatch count as a resident one.
5. **Resident kernel routing for quantised streamed prefill:** the int8 K-quant
   GEMM and the multi-row kernels, under the resident gates. Observable: the
   existing int8 kernel counters move on a streamed prefill.
6. **Only if a gap remains:** early submit off for static forwards of a session
   that holds every block, decided by measurement.

## Verification

- Token-identical output against the resident path at every step, on a budget
  that holds every block and on one that evicts (`*_matches_resident`,
  `dense_tier_parity`, the float-stream tests).
- Each step's observable asserted in a test that fails without the step.
- Throughput before and after each step, streamed holding every block against
  resident, and an evicting budget that must not regress (B36's Qwen3-14B
  shape), on both GPUs, interleaved, with host load recorded.

## Not known

- How the prefill gap splits between host work per span and the GPU idling
  between spans.
- What a barrier that names the transfer stage costs on the AMD driver.
- Whether address-taking variants exist for the multi-weight GEMV route used on
  other vendors' drivers.
