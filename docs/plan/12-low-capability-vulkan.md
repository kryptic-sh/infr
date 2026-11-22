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
to measure (S0). _(Both measured since: sections 0d and 0e. It is a submit that
is counted, at about 2 s, on both.)_

## 0d. Which work is reset: a submit, at about 2 s (2026-11-19 and 2026-11-22)

Section 0c's open question, one dispatch or one submit, measured on `Vulkan1`.
The probe is `crates/infr-vulkan/tests/dispatch_budget_probe.rs`: one Q4_K
`Op::Linear` over SmolLM3-3B's fused gate/up shape (2048 x 22016) with the int8
kernels off, which lowers to the per-row GEMV (`native_q4k`), about 3.0 ms per
row, so the row count sets the duration; each iteration is one execute, ten per
point unless a failure ends it (the backend is then poisoned). The registry
still sets no `TdrDelay`, `TdrDdiDelay` or `TdrLevel` under
`HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` (read 2026-11-19), so
Windows' defaults apply (`TdrDelay` 2 s).

**Established [ran]:**

| One submit holding                        | Quiet machine                                                                      | Second GPU client (a Qwen3-0.6B decode bench on `Vulkan1`) |
| ----------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 1 dispatch, 256 rows (0.79 s)             | not run                                                                            | 10 of 10                                                   |
| 1 dispatch, 512 rows (1.55 s)             | 10 of 10                                                                           | 10 of 10 (2.57 s wall)                                     |
| 1 dispatch, 640 rows (1.93 s)             | 10 of 10                                                                           | not run                                                    |
| 1 dispatch, 768 rows (2.3 to 2.4 s)       | lost the device after its 1st iteration, and after the 4th in a repeat (see below) | 10 of 10 (3.33 s wall)                                     |
| 1 dispatch, 896 rows (2.7 s)              | reset in its 3rd iteration                                                         | not run                                                    |
| 1 dispatch, 1024 rows (3.1 s)             | reset in its 1st iteration                                                         | not run                                                    |
| 8 independent dispatches of 128 rows      | reset in its 1st iteration                                                         | reset in its 2nd iteration                                 |
| 8 dispatches of 128 rows, barrier between | 6 completed, reset in its 7th iteration                                            | not run                                                    |
| 16 dispatches of 64 rows, barrier between | 10 of 10 (4.1 s wall)                                                              | not run                                                    |
| the same 8 x 128 rows, one submit EACH    | 10 of 10 (3.1 s)                                                                   | not run                                                    |

The quiet 512, 640 and 768 points are from 2026-11-22 (the first sweep's log of
them was not kept, and its note for 768 rows, ten of ten, did not repeat); the
rest are from 2026-11-19.

- **The watchdog counts a submit, not a dispatch.** Eight dispatches of 0.39 s
  in one submit were reset like one 3.1 s dispatch; the same eight in eight
  submits completed every time. Cutting a heavy dispatch into ranges therefore
  only helps when each range gets its own submit.
- **The limit for one submit here lies between 1.93 s (ten of ten) and 2.3 s
  (lost).** That is the default `TdrDelay` of 2 s plus whatever the driver takes
  to act on it.
- **A submit just past the limit can complete and still lose the device.** At
  768 rows the iteration that crossed the limit returned normally, with its
  closing timestamp written, after 2.3 to 2.4 s; the NEXT submit failed with a
  lost device (`LiveKernelEvent` 0x141 entries followed). So the error names the
  submit after the guilty one, which is the shape issue 94's reporter saw on the
  RX 580 (section 0e); a failure now also lists the forward segments submitted
  last with their GPU times (`gpu_wait::RecentSegment`), and here it read
  `1 dispatch (native_q4k) submitted 2.4 s earlier, ran 2.3 s on the GPU`.
- **A second GPU client did not make resets easier.** At 512 and 768 rows the
  probe completed ten times of ten with a decode bench running on the same GPU;
  the client took its share (wall time up by about a second) and the long
  dispatch was not reset, 768 rows included, which lost the device on a quiet
  machine. The preemption model of section 0c predicted the opposite.

**Not established [I]:** why barriers between dispatches inside one submit
helped (16 chained ranges completed ten times, 8 chained ones were reset once in
seven), which fits a driver that can preempt at a barrier and not inside a
dispatch, but nothing here observed preemption itself; why the second client's
presence let a 2.3 s submit through; what the 0xA1000001 entries (live dumps by
AMD's `amdfendr`, logged in runs that completed) are; and two 0x141 entries
logged during runs all of whose segments wrote their closing timestamps (the
eight-submit run, and a ten-of-ten run of 1024 rows as ranges in one submit), so
a 0x141 entry alone does not mean work was dropped.

**Heaviest dispatches by configuration** (`INFR_PROF_OPS=1` with
`INFR_PROF_OP_SHAPES=1`, one profiled forward, `infr bench -p N -u N -r 1`;
per-dispatch figures only from runs with `INFR_FULLBARRIER=1`, since without
barriers a dispatch's time lands on the next label):

| Device, model, `-u`, int8       | Heaviest single dispatches                                                                     | Largest submits        |
| ------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------- |
| `Vulkan1`, SmolLM3-3B, 128, off | per-row GEMV gate/up 2048x22016 391 ms, down 11008x2048 223 ms, 2048x2048 36 ms                | 880 ms (16 dispatches) |
| `Vulkan1`, SmolLM3-3B, 512, on  | `native_gemm_dp4a_q4k` gate/up 65 ms, down (q4_k / q6_k) 30 / 29 ms, `attn_nc_fa_hd128` 9.3 ms | 264 ms (38 dispatches) |
| `Vulkan1`, Qwen3-1.7B, 512, on  | `native_gemm_dp4a_q4k` gate/up 35 ms, down 19 / 17 ms, `attn_nc_fa_hd128` 9.2 ms (no barrier)  | 257 ms (55 dispatches) |

The per-row GEMV is the family behind the production reproduction: about 3.0 ms
per row for the SmolLM3 gate/up weight, so 1.55 s at `-u 512` and about 6.2 s at
`-u 2048` [I: linear in the rows, as the probe's points are]. In GFLOP/s, as the
work bound measures them on this GPU: the per-row GEMV 19 to 30 depending on the
weight's shape, the dp4a GEMMs about 670 to 780, the flash prefill attention
about 170.

## 0e. The RX 580 on `main`, reported (2026-11-19)

From the issue reporter's comment on issue 94 (korewaChino, RX 580, RADV on
Linux, `main` at `9151fbd2`, empty config, Qwen3-1.7B Q4_K_M). Their numbers,
not run here:

- `submit.initial_dispatches` reads 64, `matrix_units` and `dot_accelerated`
  both false. pp512 COMPLETES with the split: 238.8 t/s (`submit split/62`);
  tg128 113.2 t/s, unchanged from 112.7 on `bc9077b7`; pp2048 with `-u 2048`
  213.7 t/s (`submit split/16`).
- With `INFR_SUBMIT_DISPATCHES=0` pp512 dies every time after about 2.4 s:
  `radv/amdgpu: The CS has been cancelled because the context is lost. This context is guilty of a soft recovery`,
  and `dmesg` shows `ring gfx timeout, but soft recovered`. `modinfo -p amdgpu`
  gives `lockup_timeout` a default of 2000 ms. So on Linux the limit is amdgpu's
  job timeout, about 2 s per submit: a time limit, like the one measured in
  section 0d.
- `INFR_SUBMIT_DISPATCHES=128` and `=256` still end on `submit split/62` with
  the same throughput: the measured loop chooses the cap, and the setting only
  moves where it starts. The docs said "cap"; they now say what it does.
- The lost device surfaced on a one-shot copy after the forward and did not name
  the forward. Fixed here: see section 0d's third point.
- Their fork's prefill is 1136.1 t/s against 238.8 on the same card, from its
  `kq_fma` prefill GEMM (S4).

Since then `main` took the fork's compute-ring change
(`kernels.vulkan.gfx_queue`, PR 106): a discrete card submits on a compute-only
queue family. amdgpu documents a longer job timeout for compute rings than for
the graphics ring [R: the kernel's `lockup_timeout` parameter], so the 2 s above
may no longer be the limit that applies to that card by default; it is not
measured, and the work bound keeps assuming 2 s, the tighter figure, which also
covers `gfx_queue = true` and Windows.

## 1. Slices

| #   | Slice                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Verified here                                                                                                                                                                                                                                                                                                              | Needs the RX 580                                                                  |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| S0  | Diagnosis on `main`, no code: the reporter runs the commands asked for on issue 94 (submit cap 64 and 16, `INFR_NO_DP4A_KQ`, a per-op profile, `lockup_timeout`, dmesg). A stall or lost device reproduced locally on the integrated GPU with a heavy harness dispatch, limit and mechanism not yet established: section 0b                                                                                                                                            | n/a                                                                                                                                                                                                                                                                                                                        | yes; decides whether S1 alone fixes the hang                                      |
| S1  | Protect the first forward on slow discrete devices: a device fact for accelerated int8 dot; a capability-derived initial submit cap for discrete devices without coopmat and without accelerated dot, and for wave64-only devices; reported in `infr tune show`. Port the fork's `a35dac2f` (author kept), re-expressed against `initial_submit_dispatch_cap`. Section 0b's local stall, which already had a cap, says S1 alone does not close every case of the class | the cap function's unit test goes red on the old rule; only the `rx580-gcn-radv` snapshot changes; `INFR_FORCE_WAVE64=1` on both GPUs shows cuts on the first forward with goldens unchanged; default runs show no cuts                                                                                                    | pp512 completes on Q4_K_M                                                         |
| S2  | Work-bounded submits: an estimated GPU time per prefill dispatch (FLOPs over a measured or prior throughput), a submit closed once its estimated work reaches a time budget, and a dispatch past the budget cut into ranges of its grid, each in its own submit. **Done**, see the status below                                                                                                                                                                        | unit tests on the estimate and the cut rule that fail on the old rule; bit-identity of every cut kernel family under a forced 1 ms budget on both GPUs, native and forced wave64; the integrated GPU's production reproduction completing at default settings; no measurable change on the RX 7900 XTX at default settings | pp512 with the dispatch cap off and only the work bound on; a 3B or larger pp2048 |
| S3  | The dot-accelerated capability and the fork's fp32-dequant K-quant decode GEMV and verify GEMV (`e600b702 23cda86d d90b1c97 688fdbca 26d639d6 ad2256df a6929cd1 67582783`: apply as text, unbuilt) plus the Vulkan hunks of `244cd37c` and `6c0293a1`; not `ca67e1ee`. The default becomes a tuning parameter derived from the dot fact                                                                                                                                | the fork's `kq_fma_gemv_parity`, `kq_fma_mrow_parity`, `kq_fma_mtp_verify_tight` on both GPUs; `subgroup_kernels_are_wave64_safe`; forced on, with and without forced wave64, the whole-model GPU-vs-CPU tests; default kernel-name lists unchanged                                                                        | tg128 against `main`, profile                                                     |
| S4  | The fork's fp32 K-quant prefill GEMM (`29f9c317`, with its split-K twin): manual resolution or a PR from the fork; selected ahead of the int8 GEMM when dot is not accelerated, as a derived tuning parameter                                                                                                                                                                                                                                                          | the fork's `nc_kq_fma_gemm_parity_and_determinism` on both GPUs; forced on, the GPU gate; forced pp512 against the int8 kernel on the RX 7900 XTX recorded (no default change)                                                                                                                                             | pp512 against llama.cpp, profile                                                  |
| S5  | Decode attention with a 16-key split floor (`6b63767b`) and the `attn_combine` load overlap (`a8e0977d`), both apply as text; the floor becomes a tuning parameter (ahead of plan 10's S11). Upstream already has split-K decode attention; the difference is the floor                                                                                                                                                                                                | the fork's `attn_decode_parity` bitwise; tg128 at depth 0, 128 and 4096 on both GPUs under plan 10's measurement protocol; kept only if neither regresses                                                                                                                                                                  | tg128                                                                             |
| S6  | Q4_K packed metadata in the row-tiled decode MMV (`ea3008ca`, `7a6147b3`), re-measured against current `main`                                                                                                                                                                                                                                                                                                                                                          | the fork's exact-match test; tg128 on both GPUs                                                                                                                                                                                                                                                                            | optional                                                                          |
| S7  | Decision, not code: submitting on a compute-only queue (`a9c50bce`); it changes which amdgpu ring and timeout apply                                                                                                                                                                                                                                                                                                                                                    | A/B desktop responsiveness and throughput on both GPUs                                                                                                                                                                                                                                                                     | yes                                                                               |

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
it (re-run: both pass). The RX 580's reporter has since confirmed S1 on the card
(section 0e).

**S2 done (branch `fix/vulkan-work-budget`, 2026-11-22).** `crate::work`: every
prefill GEMM and multi-row GEMV, and the non-coopmat flash prefill attention,
estimates its GPU time before it is recorded (FLOPs over the throughput measured
for that kernel in the process, else the prior `submit.prior_gflops`, 8 GFLOP/s
per compute unit, times 8 for an accelerated int8 dot and 16 for matrix units);
a dispatch estimated past `submit.budget_ms` is cut into ranges of its workgroup
grid through a `DISPATCH_BASE` twin of its pipeline, each range in a submit of
its own; and `segment_cut` closes a submit at the next op boundary once its
estimated work reaches the budget, after the dispatch cap and before the paged
cadence. The budget is a time because the limit is (sections 0d and 0e): 250 ms,
an eighth of the 2 s, on an integrated or slow-class device, and 1000 ms
elsewhere. Decisions are made when a forward is recorded; a decode replay tape
and a single-row forward are never cut or timed.

What ran, all on 2026-11-22 with the final binary unless said:

- **The production reproduction completes** on `Vulkan1` at default settings,
  `infr bench` SmolLM3-3B Q4_K_M, one run per process: `-u 512` with
  `INFR_NO_I8DOT=1` five of five (5.1 t/s each; 144 dispatches cut into ranges
  and 869 to 1041 work cuts in a run), `-u 2048 -p 2048` five of five (91.3 to
  91.9 t/s; 143 dispatches cut, 364 or 365 work cuts), and `-u 2048` with
  `INFR_NO_I8DOT=1`, whose gate/up GEMV is about 6 s whole, once (5.1 t/s, 811
  s; 361 dispatches cut, 3675 work cuts). No `LiveKernelEvent` entry of any code
  was logged during the three batches, and `infr devices` listed both GPUs after
  every run. An earlier build of the branch, before the last changes, had done
  the first two batches five of five as well. On `main` the first was reset in
  every run (section 0c).
- **Bit-identity under a forced budget of 1 ms** (`tests/work_split_parity.rs`
  and `attention_prefill_nc_fa_cut_is_bit_identical`), each case against the
  same graph with the bound off, on both GPUs, natively and with
  `INFR_FORCE_WAVE64=1`: the per-row GEMV on a two-dimensional and on a one-row
  grid, the dp4a K-quant and the mmq int8 GEMMs, each device's default tier
  (`native_gemm_i8cm_q4k_m`, `native_gemm_warp_q8_0_sk_ag` and
  `native_gemm_warp_f16_sk` on the RX 7900 XTX; `native_gemm_dp4a_q4k`,
  `native_gemm_mmq_q8_0` and `native_gemm_fma_f16` on the integrated GPU and
  under forced wave64) and the flash prefill attention. Running the second range
  over the first one's workgroups failed five of the six parity tests on the RX
  7900 XTX, and shortening the first range failed four.
- **Issue 94's class, simulated**
  (`..._slow_class_forward_under_the_default_budget`: wave64, no matrix kernels,
  no int8 dot, the RX 580's own prior of 288 GFLOP/s, the default budget): a
  1024-row forward over SmolLM3's feed-forward shapes on the RX 7900 XTX had two
  dispatches cut into ranges and three submits opened by the work bound in its
  first forward, bit-identical to the same forward in one submit. The GEMV there
  measured about 410 GFLOP/s, so that prior is pessimistic for the XTX; what it
  is on GCN is the reporter's to measure.
- **Defaults on the fast path.** `infr tune show` against `main` differs on both
  GPUs only by the two new lines and the hash (`submit.budget_ms` 1000 and
  `submit.prior_gflops` 768 on the RX 7900 XTX, 250 and 16 on the integrated
  GPU). Qwen3-1.7B Q4_K_M at default settings on the RX 7900 XTX: no cut and no
  timed dispatch in any forward. On the integrated GPU: the dispatch cap's cuts
  as before, plus three work cuts in the first prefill forward and none after.
- **No measurable change at default settings.** `infr bench` Qwen3-1.7B Q4_K_M
  against `main` (`86f90c95`), interleaved, two file copies of each binary, the
  first copy rotating each round, nothing else running. RX 7900 XTX, twelve
  rounds: pp512 11538.68 against 11542.07 t/s (-0.03%; the two copies of one
  binary differ by 0.12% and 0.19%), tg128 469.18 against 469.28 (-0.02%; copies
  0.01% and 0.00%). A first set of six rounds read -0.21% and -0.06% with copies
  up to 0.27% and 0.10% apart. Integrated GPU, six rounds: pp512 231.03 against
  231.06 (-0.01%; copies 0.02% and 0.04%), tg128 27.02 against 27.02 (copies
  0.06% and 0.19%).
- **The unit tests fail on the old rule**: with the work arm removed from
  `segment_cut`, `the_work_cut_closes_a_heavy_segment_of_few_dispatches` fails;
  with a dispatch never cut, three of `work::tests` fail; with a prior ten times
  faster, three; with a measurement allowed to lower an estimate twice, one.
- **Gates.** The GPU gate (the filter of `docs/gpu-tolerance.md` plus
  `mtp_head_cpu_vulkan_parity` and `work_split_parity`): 616 passed on the
  integrated GPU, the tolerance tests among them with the harness workaround
  removed and no table row changed; on the RX 7900 XTX 676 passed and 1 failed,
  `gpu_seam_golden_qwen3`. It fails with the same hash on `main` at `86f90c95`
  (the three fork PRs merged), which the gate of 2026-11-19, before them, did
  not report, so one of those PRs moved that golden on native lanes (the
  decode-attention change is the likely one [I]), not this slice; left for the
  slice that reviews those PRs. With `INFR_FORCE_WAVE64=1` on the RX 7900 XTX
  616 passed.

What it leaves out is in the backlog (B226): kernels without an estimate, the
thin margin of the prior for one GEMV shape, the cuts a large model takes on a
fast card by estimate, and the RX 580's limit on the compute ring.

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
