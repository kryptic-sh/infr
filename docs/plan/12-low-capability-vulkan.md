# 12 — Vulkan devices without matrix units or fast int8 dot (GCN class)

A device with no cooperative matrix and an int8 dot product the driver only
emulates (AMD GCN: Polaris, Vega under RADV) must run every supported model,
with kernels chosen for that class. Today upstream `main` loses the device on
the first K-quant prefill on an RX 580 (issue 94). The community fork
KessokuNet/infr carries the kernels and the submit rule that fix it; this phase
brings them over as cherry-picks or PRs with their author's attribution kept as
written.

Tags: **[R]** read in code or history, **[I]** inferred, **[U]** unknown until
measured. Nothing here has been built or run: the fork was read and its commits
trial-applied as text onto `bc9077b7` on 2026-11-18.

## 0. Where things stand

- K-quant prefill on that class takes `native_gemm_dp4a_q4k` / `_q6k` (the nc
  tier, `dp4a_kq_ok`), which GCN emulates in ALU **[R]**.
- A discrete GPU's first forward is one submit: `initial_submit_dispatch_cap` is
  0 for discrete parts and the splitter arms only after a forward completes past
  `SUBMIT_DANGER_NS` **[R]**. The fork measured this failure on an RX 580 and
  fixed it in its commit `a35dac2f`, which its 2026-11-17 re-land dropped
  **[R]**. The fork survives today because its `kq_fma` kernel makes the forward
  short; it has no submit-time protection either **[R]**.
- The likely cause of the device loss is that long first submit outlasting the
  amdgpu job timeout **[I]**. Less likely: one dispatch alone outlasts it (then
  a dispatch cap cannot help), or a compiler problem specific to that shader. S0
  tells them apart.
- Nothing upstream distinguishes "int8 dot present" from "int8 dot accelerated";
  `DeviceProbe::dot4x8_signed_accelerated` is probed and read only for Intel sg8
  **[R]**. Every fork tier for this class keys on that bit.
- GCN has never run here (B-WAVE64-GCN-UNVERIFIED). Every slice is verified on
  the RX 7900 XTX and the integrated GPU through forced masks
  (`INFR_FORCE_WAVE64`, `INFR_NO_COOPMAT`, `INFR_NO_F16`, `INFR_NO_I8DOT`,
  `INFR_NO_DP4A_KQ`, `INFR_SUBMIT_DISPATCHES`), which reproduce the kernel
  selection but not the timeout, and confirmed on an RX 580 by the issue
  reporter.
- The fork's switches are config keys whose default comes from a capability, not
  tuning-table rows **[R]**. In plan 10's terms each becomes a
  capability-derived default once the dot fact is in the tuning `DeviceFacts`.
  The `rx580-gcn-radv` fixture exists, matches no rows and carries no dot fact.

## 0b. Reproduced locally (2026-11-18)

A first forward that stalls or loses the device reproduces on this box's
integrated GPU (`Vulkan1`, RDNA2, 2 CUs, AMD proprietary driver, initial cap
128), not through GCN's emulated dot but through a submit containing the GPU
tolerance harness's prefill head (a vocabulary-wide float GEMM over every
prefill-chunk row) with every int8 kernel off **[ran]**. Qwen3.5-0.8B Q4_K_M,
its 180-token prompt (`tests/support/forced.rs` `PROMPTS[2]`), configuration
`gpu-float`:

| Variant (one change each)             | Result (runs)                                                 |
| ------------------------------------- | ------------------------------------------------------------- |
| as is (cap 128)                       | stalled, one CPU core busy, killed (3/3)                      |
| `device.submit_dispatches` 16         | stalled, killed (1/1)                                         |
| `device.submit_dispatches` 8          | device lost (1/1)                                             |
| `device.submit_dispatches` 4          | completed, 16 s (4/7); stalled (3/7, the last three in a row) |
| prefill head on each chunk's last row | completed, 9 s (1/1)                                          |
| no prefill head (production graph)    | completed, 9 s (1/1)                                          |
| chunk height 32 or 64 rows            | completed, 16-17 s (2/2, 1/1)                                 |

Qwen3-0.6B completed in every variant this time, though one earlier gate run
lost the device on the same kind of run. To reproduce:

```text
INFR_DEV=Vulkan1 INFR_TOL_MODELS=qwen35-0.8b INFR_TOL_ONLY=gpu-float INFR_TOL_PROMPTS=2 \
  cargo test -p infr-llama --release --test gpu_tolerance measure_gpu_tolerance \
  -- --ignored --nocapture
```

(`INFR_TOL_SUBMIT`, `INFR_TOL_HEAD` and `INFR_TOL_UBATCH` are the knobs above.)
In the full matrix at cap 4, gemma-3-1b (and Qwen3.5 on another prompt) still
stalled; gemma completed with the head on each chunk's last row.

What is observed, and no more: on the integrated GPU a submit containing the
vocabulary-wide head stalled or lost the device at the default cap; reducing
dispatches per submit to 4 (four runs of seven), or dropping the head, made
Qwen3.5 complete; gemma still stalled at cap 4. Neither the limit nor the
mechanism is established. Qwen3.5 completing at cap 4 WITH the full head says
its head dispatch alone outlasts nothing, which fits a limit on a submit's total
work; gemma stalling at cap 4 fits a single dispatch, which neither a smaller
count nor a work budget per submit (S2) can split. A stall of minutes with one
CPU core busy is not what a driver watchdog reset looks like (that is a lost
device), and cap 8 losing the device while 16 and 128 stalled is not monotone,
so a host-side wait is not ruled out.

A second look, sampling the process's GPU-engine utilisation (`typeperf`, one
sample a second) and its CPU time **[ran]**, with the measure binary of the runs
above:

| Run (prompt 2, `gpu-float`)                    | Result                                                                                                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3.5, full head, cap 4 (3 runs)             | stalled every time: the 3D engine busy at 97-99% for 3-4 s, then idle (one blip) for the rest; CPU time +17 s in 20 s (one core spinning)         |
| Qwen3.5, full head, cap 128 (2 runs)           | the 3D engine busy for 3-4 s, then idle; once stalled to the 240 s kill, once the next `queue_submit` reported a lost device 55 s after the start |
| gemma-3-1b, full head, cap 4                   | stalled (killed at 240 s)                                                                                                                         |
| gemma-3-1b, head on last row, cap 128 (2 runs) | completed, GPU part 15.7 and 14.8 s; the 3D engine busy throughout, 12 s without a gap                                                            |
| gemma-3-1b, head on last row, cap 4            | completed, GPU part 70.6 s                                                                                                                        |

So the stall is not the host waiting correctly on a long-running dispatch: the
GPU stops working after 3-4 s, and the host then waits on a fence nothing
signals (`PendingSegment::drain` waits with no timeout), spinning a core, or
meets the lost device at its next submit. The stall and the lost device look
like two outcomes of one event, the GPU's work being cut off. A watchdog reset
would do that, but it is not confirmed **[I]**: the System event log shows no
display-driver recovery event (ID 4101) in its recent history, and a completing
run kept the engine busy for 12 s without a gap, so it is not simply the engine
being busy too long; a fault in the head's own dispatch (its output is
vocabulary times chunk rows) fits as well. Two of the cap-4 Qwen3.5 runs, one
cap-128 Qwen3.5 run and one gemma head-last run were in a quiet window; the
others overlapped another session's GPU tests, which may have run on this GPU
too. Both the stall and the loss track the full head: gemma's head on each
chunk's last row completed at the default cap, and cap 4 with the full head
stalled Qwen3.5 once in five runs. What is established: a submit containing the
full head can stop the integrated GPU after a few seconds; the head on each
chunk's last row has not done so in any run. Not established: whether one
dispatch or a submit's total is what crosses the limit. And whatever the limit,
a host wait without a timeout turns the loss into a hang that only a watchdog
outside the process ends; a bounded fence wait reporting a lost device would be
a fix of its own. _(Superseded by section 0c: the "no 4101 event" reading looked
in the wrong log, an engine reset (0x141) is recorded for these runs, and the
spinning core was a timestamp read after the reset had signalled the fence.)_

For this phase: S1's initial cap alone would not have prevented this case, which
already had one, and whether S2 would is open until the limit is known. The
tolerance tests work around it on an integrated GPU, for configurations with
every int8 knob off, with both the head on each chunk's last row and a cap of 4
(B207); with that the whole `gpu_seam_tolerance_*` set passed on `Vulkan1` in
637 s. Production never runs that head. _(Superseded by section 0c: the reset
and the hang are now explained, and a production path without the harness hits
the same reset with a large `-u`.)_

## 0c. What cuts the work off on the integrated GPU (2026-11-18, later)

With bounded waits in place (`device.gpu_wait_timeout_ms`, see `gpu_wait.rs`)
the reproduction of section 0b ends in an error instead of a hang, which made
more of it measurable. All runs on `Vulkan1`; the harness runs are Qwen3.5-0.8B,
prompt 2, `gpu-float`; each is listed with its run count.

**Established [ran]:**

- **An engine reset happens.** The Application log has `LiveKernelEvent` Windows
  Error Reporting entries within seconds of each failing run whose time was
  compared (the first, at 21:42, and every one from 21:53 to 22:22 local time)
  and none near the five that completed; the seven of those 21:53-22:22 entries
  that carry a bucket so far are all `LKD_0x141_Tdr:6_IMAGE_amdkmdag.sys`
  (0x141: a GPU engine timed out and was reset; the rest had not been bucketed
  when read). The same entries run through the afternoon of section 0b's
  experiments: 0b looked for event 4101 in the System log, which an engine reset
  does not write. No `TdrDelay`, `TdrDdiDelay` or `TdrLevel` is set under
  `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` (read only), so
  Windows' defaults apply: a 2 s `TdrDelay`.
- **After the reset the fence reports signalled and the work is dropped.** With
  a trace of every wait (2 runs at the default cap): the wait on the forward
  holding the head returned success after 6.2 and 5.8 s, and the read of that
  segment's timestamps with `WAIT` then never returned: that read is the core
  spinning forever. A third traced run completed. With the timestamps read
  without waiting, every harness run that met the reset since (3 of 3 at the
  default cap, 2 of 2 at one dispatch per submit) ended in under 30 s with
  `the GPU device was lost: submitting a one-shot command buffer failed; VK_EXT_device_fault: "No fault detected"`,
  the process exited, and `infr devices` listed both GPUs afterwards.
- **It happens only with a long dispatch in flight.** The harness head is one
  `native_q6k` dispatch (a scalar-dequant float GEMM, vocabulary-wide) of about
  25 ms per chunk row. One dispatch per submit (`INFR_TOL_SUBMIT=1`), each
  dispatch timed by its own segment timestamps:

  | Chunk rows (`INFR_TOL_UBATCH`) | Runs | Head dispatch | Result                         |
  | ------------------------------ | ---- | ------------- | ------------------------------ |
  | 32                             | 1    | 0.82-0.83 s   | completed                      |
  | 64                             | 1    | 1.63 s        | completed                      |
  | 96                             | 1    | 2.44 s        | device lost at the next submit |
  | 128 (default)                  | 1    | 3.26 s        | device lost at the next submit |
  | 128, head on the last row      | 1    | under 0.064 s | completed                      |

  The head on each chunk's last row has never been reset in any run (sections 0b
  and 0c together).

- **A production path without the harness hits it.** `infr bench` of SmolLM3-3B
  Q4_K_M on `Vulkan1`, one run each: `-u 512` with the default int8 kernels
  completed (99.1 t/s, segments of about 0.97 s); `-u 512` with
  `INFR_NO_I8DOT=1`, and `-u 2048` with and without it, met the reset (each with
  `LiveKernelEvent` entries at its time). The `-u 2048` int8 run had submitted
  128-dispatch segments that ran 4.9 and 5.2 s. All three now end with the
  missing-timestamp error, for example
  `the GPU signalled a forward segment of 129 dispatches (first rope_f16_bda, last native_gemm_dp4a_q4k) complete but never wrote its closing timestamp`:
  the check fired on real hardware there, three times. In the harness runs it
  never did; there the next submit reported the lost device first. The default
  integrated chunk (`integrated_ubatch_rows`, 128 rows) with the int8 kernels on
  has not met it. Re-run on 2026-11-19 with the branch build and no other
  session on the machine, three runs each: `-u 512` with `INFR_NO_I8DOT=1` met
  the reset every time and ended in 10 to 15 s with the same error (segments of
  107 and 108 dispatches, first `embed_gather_q6k`), the process exiting and
  `infr devices` listing both GPUs after each; `-u 2048 -p 2048` with the
  default kernels COMPLETED all three times (71.8, 75.3 and 73.3 t/s), where it
  had been reset on 2026-11-18 while another session was using the machine.

**Not established [I]: the rule.** Two models fit parts of the data:

- **A fixed limit on how long one dispatch may run (about 2 s).** Explains the
  sweep above (completed at 1.63 s, reset at 2.44 and 3.26 s) and the production
  runs. Does not explain the completions: in section 0b the same 3.26 s head
  completed at cap 4 in four runs of seven, a third traced run here completed
  with it at the default cap, a gemma-3 run kept the engine busy 12 s without a
  gap, and cap 8 lost the device while caps 16 and 128 stalled.
- **A preemption timeout.** Windows asks the engine to preempt when another
  context needs it (on this box the desktop shares the integrated GPU), and
  resets it when the running work does not yield within `TdrDelay`; if this
  engine preempts only between dispatches, a long dispatch is reset only when a
  preemption request lands during it. Explains the nondeterminism and the
  completions, and fits the `-u 2048` runs completing on a quiet machine and
  being reset on a busy one (one busy run, three quiet). Not tested: nothing
  here controlled or observed the preemption requests, and the
  dispatch-versus-command-buffer granularity is not known
  (`infr_core::SUBMIT_BUDGET_NS`'s comment records a 2-CU submit reset at 2.06
  s; in the GPU gate a single 126-dispatch LFM3 forward waited 2.75 s and
  completed).

Either way, keeping every dispatch well under 2 s on this GPU avoids it, and no
submit boundary shortens a dispatch. The RX 580's limit under Linux is amdgpu's
per-job `lockup_timeout`, a different watchdog: what it counts is the reporter's
to measure (S0).

## 1. Slices

| #   | Slice                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Verified here                                                                                                                                                                                                                                       | Needs the RX 580                             |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| S0  | Diagnosis on `main`, no code: the reporter runs the commands asked for on issue 94 (submit cap 64 and 16, `INFR_NO_DP4A_KQ`, a per-op profile, `lockup_timeout`, dmesg). A stall or lost device reproduced locally on the integrated GPU with a heavy harness dispatch, limit and mechanism not yet established: section 0b                                                                                                                                            | n/a                                                                                                                                                                                                                                                 | yes; decides whether S1 alone fixes the hang |
| S1  | Protect the first forward on slow discrete devices: a device fact for accelerated int8 dot; a capability-derived initial submit cap for discrete devices without coopmat and without accelerated dot, and for wave64-only devices; reported in `infr tune show`. Port the fork's `a35dac2f` (author kept), re-expressed against `initial_submit_dispatch_cap`. Section 0b's local stall, which already had a cap, says S1 alone does not close every case of the class | the cap function's unit test goes red on the old rule; only the `rx580-gcn-radv` snapshot changes; `INFR_FORCE_WAVE64=1` on both GPUs shows cuts on the first forward with goldens unchanged; default runs show no cuts                             | pp512 completes on Q4_K_M                    |
| S2  | Work-bounded submits: also cut on an estimated work budget per submit scaled by compute-unit count (llama.cpp's rule, which a code comment already names as worth adopting)                                                                                                                                                                                                                                                                                            | unit tests on the budget; per-submit GPU time stays under `SUBMIT_BUDGET_NS` with a forced large ubatch on the integrated GPU                                                                                                                       | a 3B or larger pp2048 completes              |
| S3  | The dot-accelerated capability and the fork's fp32-dequant K-quant decode GEMV and verify GEMV (`e600b702 23cda86d d90b1c97 688fdbca 26d639d6 ad2256df a6929cd1 67582783`: apply as text, unbuilt) plus the Vulkan hunks of `244cd37c` and `6c0293a1`; not `ca67e1ee`. The default becomes a tuning parameter derived from the dot fact                                                                                                                                | the fork's `kq_fma_gemv_parity`, `kq_fma_mrow_parity`, `kq_fma_mtp_verify_tight` on both GPUs; `subgroup_kernels_are_wave64_safe`; forced on, with and without forced wave64, the whole-model GPU-vs-CPU tests; default kernel-name lists unchanged | tg128 against `main`, profile                |
| S4  | The fork's fp32 K-quant prefill GEMM (`29f9c317`, with its split-K twin): manual resolution or a PR from the fork; selected ahead of the int8 GEMM when dot is not accelerated, as a derived tuning parameter                                                                                                                                                                                                                                                          | the fork's `nc_kq_fma_gemm_parity_and_determinism` on both GPUs; forced on, the GPU gate; forced pp512 against the int8 kernel on the RX 7900 XTX recorded (no default change)                                                                      | pp512 against llama.cpp, profile             |
| S5  | Decode attention with a 16-key split floor (`6b63767b`) and the `attn_combine` load overlap (`a8e0977d`), both apply as text; the floor becomes a tuning parameter (ahead of plan 10's S11). Upstream already has split-K decode attention; the difference is the floor                                                                                                                                                                                                | the fork's `attn_decode_parity` bitwise; tg128 at depth 0, 128 and 4096 on both GPUs under plan 10's measurement protocol; kept only if neither regresses                                                                                           | tg128                                        |
| S6  | Q4_K packed metadata in the row-tiled decode MMV (`ea3008ca`, `7a6147b3`), re-measured against current `main`                                                                                                                                                                                                                                                                                                                                                          | the fork's exact-match test; tg128 on both GPUs                                                                                                                                                                                                     | optional                                     |
| S7  | Decision, not code: submitting on a compute-only queue (`a9c50bce`); it changes which amdgpu ring and timeout apply                                                                                                                                                                                                                                                                                                                                                    | A/B desktop responsiveness and throughput on both GPUs                                                                                                                                                                                              | yes                                          |

**Status (branch `fix/vulkan-submit-bounds`, 2026-11-19).** Done beside the
slices: every host wait is bounded and a failure poisons the backend with an
error naming the submit (`device.gpu_wait_timeout_ms`). A fence signalled for
dropped work is caught ONLY on a forward segment from a non-persistent recorder
on a queue that writes timestamps, by its missing closing timestamp (section
0c); decode replays, chained decode, one-shot copies, staging copies and any
queue without timestamps are not checked, so a reset during a decode replay can
still hand back wrong logits until the next submit reports the lost device
(B222). **S1 done:** the fork's `a35dac2f` is cherry-picked with its message as
written; on top of it the tuning `DeviceFacts` carry `matrix_units` and
`dot_accelerated`, `DeviceFacts::slow_class` is the fork's rule, and the initial
cap is the tuning parameter `submit.initial_dispatches` (`infr tune show`: 64 on
the `rx580-gcn-radv` and `unknown` fixtures, 128 integrated, 0 elsewhere). The
brief expected only the GCN snapshot to change; adding a parameter changes all
seven (each gains that one line and a new hash, so the banner hash changes on
every device), and no other value moves in any of them; `infr tune show` on both
real GPUs against a `main` build differs only by that line (0 on the RX 7900
XTX, 128 on the integrated GPU) and the hash. Verified:
`initial_submit_cap_follows_the_device_facts` failed with the slow class removed
from the default (`left: 0, right: 64` for the GCN fixture, 2026-11-18) and
passes with it; `first_forward_split` (re-run 2026-11-19 on the final branch) on
the RX 7900 XTX cut the first prefill forward twice with `INFR_FORCE_WAVE64`
(cap 64), bit-identical to the unsplit forced run, left a single-token forward
whole, and cut nothing natively; on the integrated GPU, whose own cap of 128
applies either way, it cut once forced and once native. The GPU gate on the
branch (2026-11-19, the filter of `docs/gpu-tolerance.md` plus
`mtp_head_cpu_vulkan_parity`): 652 passed on the RX 7900 XTX and 591 on the
integrated GPU (`main` at `086ebff0`: 643 and 582); with `INFR_FORCE_WAVE64=1`
on the XTX 590 passed and 1 failed, the native-device half of
`first_forward_split`, which forcing makes slow-class and which now skips under
it (re-run: both pass). **S2 not done:** section 0c shows a per-submit budget is
not enough on its own, because a long dispatch is what is reset, and no submit
boundary shortens it. What it needs, from the measurements: a per-op cost
estimate (FLOPs from the op's shape) over a throughput that is per device AND
per kernel family (on the integrated GPU the dp4a GEMMs and the scalar-dequant
ones differ by about an order of magnitude, estimated from segment times, not
measured per op [I]), refined from measured segment times; cutting submits on
the estimate; and SPLITTING an op whose own estimate passes the budget into row
ranges, or refusing it with a message (B221). Until then a large `-u` on a slow
device fails with an error instead of a hang. The RX 580 still needs the
reporter: S1 is the fork's measured fix for that card, untested here on GCN.

Declined: the fork's opt-in SSBO weight binding, which its own config doc marks
as producing wrong output in a model on the RX 580, and its GQA decode pass,
which its own commit measured as a 5–7% loss on its target. Native 64-lane
subgroup decode stays with plan 10's S4; the reporter measured no gain from it
on Q4_K_M (123.3 against 123.1 t/s), because K-quant decode never selects the
kernel family it switches.

Not analysed in the fork: KV paging, EXL3, vision and MTP additions, server
features, the extra CLI commands.

## 2. Attribution

The fork's 203 non-merge commits are by its author, and many carry that author's
own trailers. They are taken as cherry-picks of the re-land hashes (the hashes
above), or as PRs the author opens, with every line of their messages kept as
written.

## 3. Must not change

Default kernel selection and output bits on the RX 7900 XTX and the integrated
GPU. A fixture snapshot other than the GCN one changing is a failure.
