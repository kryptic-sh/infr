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

## 1. Slices

| #   | Slice                                                                                                                                                                                                                                                                                                                                                         | Verified here                                                                                                                                                                                                                                       | Needs the RX 580                             |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| S0  | Diagnosis on `main`, no code: the reporter runs the commands asked for on issue 94 (submit cap 64 and 16, `INFR_NO_DP4A_KQ`, a per-op profile, `lockup_timeout`, dmesg)                                                                                                                                                                                       | n/a                                                                                                                                                                                                                                                 | yes; decides whether S1 alone fixes the hang |
| S1  | Protect the first forward on slow discrete devices: a device fact for accelerated int8 dot; a capability-derived initial submit cap for discrete devices without coopmat and without accelerated dot, and for wave64-only devices; reported in `infr tune show`. Port the fork's `a35dac2f` (author kept), re-expressed against `initial_submit_dispatch_cap` | the cap function's unit test goes red on the old rule; only the `rx580-gcn-radv` snapshot changes; `INFR_FORCE_WAVE64=1` on both GPUs shows cuts on the first forward with goldens unchanged; default runs show no cuts                             | pp512 completes on Q4_K_M                    |
| S2  | Work-bounded submits: also cut on an estimated work budget per submit scaled by compute-unit count (llama.cpp's rule, which a code comment already names as worth adopting)                                                                                                                                                                                   | unit tests on the budget; per-submit GPU time stays under `SUBMIT_BUDGET_NS` with a forced large ubatch on the integrated GPU                                                                                                                       | a 3B or larger pp2048 completes              |
| S3  | The dot-accelerated capability and the fork's fp32-dequant K-quant decode GEMV and verify GEMV (`e600b702 23cda86d d90b1c97 688fdbca 26d639d6 ad2256df a6929cd1 67582783`: apply as text, unbuilt) plus the Vulkan hunks of `244cd37c` and `6c0293a1`; not `ca67e1ee`. The default becomes a tuning parameter derived from the dot fact                       | the fork's `kq_fma_gemv_parity`, `kq_fma_mrow_parity`, `kq_fma_mtp_verify_tight` on both GPUs; `subgroup_kernels_are_wave64_safe`; forced on, with and without forced wave64, the whole-model GPU-vs-CPU tests; default kernel-name lists unchanged | tg128 against `main`, profile                |
| S4  | The fork's fp32 K-quant prefill GEMM (`29f9c317`, with its split-K twin): manual resolution or a PR from the fork; selected ahead of the int8 GEMM when dot is not accelerated, as a derived tuning parameter                                                                                                                                                 | the fork's `nc_kq_fma_gemm_parity_and_determinism` on both GPUs; forced on, the GPU gate; forced pp512 against the int8 kernel on the RX 7900 XTX recorded (no default change)                                                                      | pp512 against llama.cpp, profile             |
| S5  | Decode attention with a 16-key split floor (`6b63767b`) and the `attn_combine` load overlap (`a8e0977d`), both apply as text; the floor becomes a tuning parameter (ahead of plan 10's S11). Upstream already has split-K decode attention; the difference is the floor                                                                                       | the fork's `attn_decode_parity` bitwise; tg128 at depth 0, 128 and 4096 on both GPUs under plan 10's measurement protocol; kept only if neither regresses                                                                                           | tg128                                        |
| S6  | Q4_K packed metadata in the row-tiled decode MMV (`ea3008ca`, `7a6147b3`), re-measured against current `main`                                                                                                                                                                                                                                                 | the fork's exact-match test; tg128 on both GPUs                                                                                                                                                                                                     | optional                                     |
| S7  | Decision, not code: submitting on a compute-only queue (`a9c50bce`); it changes which amdgpu ring and timeout apply                                                                                                                                                                                                                                           | A/B desktop responsiveness and throughput on both GPUs                                                                                                                                                                                              | yes                                          |

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
