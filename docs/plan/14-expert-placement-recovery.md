# 14 — Expert placement that recovers

**Why.** A MoE run pages every expert layer for the whole process when the one
VRAM read at first load comes out short (`vulkan_moe_binder`,
`moe_expert_budget` in `infr-llama` `seam/mod.rs`). A run started within about
150 ms of another process exiting reads that process's memory as in use and runs
five times slower for good (backlog B-PP-SLOW-RUNS, with the evidence). The
owner decided 2026-11-25: wait briefly at start-up, promote experts back into
VRAM when room appears later, and label the placement in the bench output.

Nothing here budgets past the live `VK_EXT_memory_budget` figure; budgeting
against the heap size stays declined.

## How placement works today

- `vram_info` (`infr-vulkan` `lib.rs`) reports free memory per counted heap as
  budget minus usage; `VramInfo` does not say how much of the used part other
  processes hold.
- The automatic tier reads it once, and if the budget is below the expert bytes
  pages every layer. Explicit `paging.cache` (`INFR_CACHE`) pins paging and
  never reaches that arm.
- A paged `_exps` bank is a small placeholder buffer registered with
  `MoePagerSession` under its `buffer_identity`; `execute_static` asks the pager
  whether a layer's gate bank is paged and takes `execute_paged_moe` (`_xpg`
  builds, no decode replay) or the resident arm (`_xp` builds). `Bindings` are
  rebuilt on every call and the replay plan is decided per generate call.
- Each pager pool is one contiguous arena, so it cannot grow in place.
- Pipeline and tensor parallel reject MoE and expert parallel binds resident
  only, so multi-GPU never pages.

## Slices

Each ships alone, through coder then reviewer, with a test that can fail.

1. **L: placement label.** `BenchPlacement` (`seam/model.rs`) gains the paged
   and total MoE layer counts; `placement_note` and `bench_json_line`
   (`infr-cli`) print `experts paged N/M` and the JSON fields `experts_paged`,
   `expert_layers`, for MoE models only. The pager log line counts MoE layers,
   not `n_layer`. Test: the JSON line, and a GPU run with paging forced (N = M)
   against a resident one (0).
2. **W: bounded wait.** `VramInfo` gains what other processes hold (heap size
   minus heap budget over the counted heaps; absent without the extension). A
   pure helper beside `moe_expert_budget` takes the first reading, a reader and
   a sleeper; it arms only when the budget is short and would not be short if
   the others released, polls on a named interval up to a named bound, stops at
   the first reading that fits, and stops early when the others no longer
   explain the shortfall. Logged at info level on arming and on exit. Never arms
   without a live budget or under explicit `paging.cache`. Tests on scripted
   readings: fits at once, no live budget, shortfall larger than the others,
   frees on poll k, never frees, others grow mid-wait, capped snapshot. Measure:
   the twelve back-to-back `-p 512 -n 0` runs (none paged), and the start-up
   delay under real pressure.
3. **M0: measure, no code.** Qwen3-30B-A3B Q4_K_M, RX 7900 XTX, quiet window,
   interleaved: resident against `INFR_CACHE` holding every expert, pp512 and
   tg128; and the time to upload one layer's banks. This decides whether
   per-layer promotion is worth more than a bigger arena.
4. **A: weight alias map.** A backend map from a placeholder's identity to a
   resident buffer, applied to `Bindings` at `execute`, `execute_chain` and
   `decode_replay_ready`, behind a flag so a run with no alias pays nothing.
   Test on the `gpu_seam_paged_moe_matches_*` harness: alias one layer to an
   uploaded bank and compare with the resident run. Measure: resident pp512 and
   tg128 unchanged.
5. **P: promote one layer.** `MoePagerSession::unregister_layer` and a
   `promote_layer` that uploads every role of the layer through the pager's
   ring, aliases them and unregisters, all roles together (the paged check reads
   only the gate role). Test: tokens equal the resident run and the pager stats
   show the layer untouched afterwards.
6. **Q: policy at safe points.** Promote the next layer when the live room
   covers its banks, the observed activation peak and a slack; two qualifying
   reads a named interval apart; back off after a refused allocation; never
   demote. Checked at the end of a generate call under the serve step gate, and
   once between the bench warm-up and its timed reps. Disarmed on unified
   memory, without a live budget, and under explicit `paging.cache`. Tests: the
   policy on scripted rooms and a scripted clock; end to end with a test-owned
   ballast buffer that forces paging, is dropped, and the next call promotes.
7. **D: drop the pager.** When the rest fits, idle the queue, stop the fill
   reader, drop the session and promote the rest, so decode replay returns on
   the next call; on a failed allocation rebuild a session for the remaining
   layers. Test: replay is ready after full promotion; an injected failure
   leaves a working paged session.
8. **U (only if M0 or serve traces call for it): upkeep inside a long call.**

## To measure or check before the slice that needs it

- What "held by others" reads on an idle RX 7900 XTX and on the integrated GPU;
  a driver that holds back a constant share makes the wait arm on every paged
  start there (W).
- Whether the context clamp before the load (`kv_fit_ctx`, `vulkan_slot_ctx`) is
  also hit by the stale read, and whether the dense-streaming automatic tier,
  which has the same single read, should reuse the helper (W).
- Whether a weights buffer allocated outside a load scope is addressable the way
  the resident kernels need, and the upload rate (P).
- Whether every static execute waits before returning, and how the MTP trunk's
  backend is scoped (U, Q).
- Whether the host tier should be dropped with the pager under an explicit
  `INFR_DRAM_CACHE`: the owner's call (D).

## Considered and declined

- **Whole re-placement** (drop KV and weights, load again): seconds of upload,
  serve loses its KV and prefix cache, and a failure midway leaves no model.
- **Only a bigger arena**: cheap, but a fully held paged model still runs the
  paged kernels without decode replay (plan 06 measured 177 against 186 t/s);
  kept as a fallback if M0 shows the gap is small.
