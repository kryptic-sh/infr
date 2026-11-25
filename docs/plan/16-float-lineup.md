# 16 — A full float kernel lineup for GPUs without native int8

**Why.** On a GPU whose packed int8 dot product is missing or emulated, the int8
kernels buy no speed, and they cost accuracy. Two f32-activation kernels exist
for that hardware (`native_gemv_kq_fma` for decode, `native_gemm_kq_fma` for
prefill), but each covers only some weight formats, and a model that runs some
Linears on f32 and the rest on int8 measured **further** from the CPU reference
than all-int8 does (gemma-3, worst KL 0.741 against 0.310, bound 0.52; all-float
runs sit around 3e-6: `docs/gpu-tolerance.md`). The owner decided 2026-11-26 to
build the full lineup: on that hardware every Linear of a model runs on f32
activations, prefill, decode and MoE experts, so the run belongs to the Float
tolerance class at the best speed the device has. The contributor of the two
kernels has no further float kernels in their fork (checked 2026-11-26, told on
PR #112), so this is ours to build; they own an RX 580 and can measure it.

Nothing here is built. Statements marked **guess** are guesses. The format
table's "what people run" column comes from reading the tensor types of the
cached GGUF files, not from a survey.

## The rule

**A model runs all-float or all-int8 on a device, never a mix.** The lineup is
decided once per model and every lowering site follows it. A model with any
Linear that has no f32 kernel fast enough on that device runs all-int8, and says
why.

## Device classes

Decided by two facts: `caps.i8_dot` (the extension and feature present, after
the `kernels.vulkan.i8_dot` mask) and `DeviceFacts.dot_accelerated` (the
driver's `integerDotProduct4x8BitPackedSignedAccelerated`; false when the
extension is absent).

| Class | Facts                                         | Today                                                                                                                                                                                 |
| ----- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a)   | no int8 dot                                   | all-float already: every activation quantize sits behind `i8_dot`. Slow: quantized prefill takes the per-row GEMV unless `gemm.kq_fma` is set, and MoE prefill takes the small-m path |
| (b)   | dot present, not accelerated (RX 580 fixture) | a **partial** lineup by default: decode runs Q4_K/Q6_K on f32 (`gemv.kq_fma`) and Q5_0, Q4_0, Q2_K, Q5_1, IQ4_NL on int8; prefill is int8                                             |
| (c)   | dot accelerated (both GPUs here)              | int8 where gated, coopmat where present; out of scope                                                                                                                                 |

Other facts (wave64, integrated, shared memory) change submit budgets and
whether a tile fits, not the routing. A device with f16 matrix units and an
unaccelerated dot is not in any fixture (**guess:** rare); it is in scope for
the rule, not for new kernels.

## What exists, by format

T is a tiled f32 GEMM, G a real f32 GEMV, R a per-row GEMV that re-reads the
weight for every row (slow), and a dash means int8 only.

| Formats                          | Prefill                      | Decode   | MoE batched experts | Found in                                           |
| -------------------------------- | ---------------------------- | -------- | ------------------- | -------------------------------------------------- |
| F32, F16, BF16                   | T                            | G        | never batched       | every file                                         |
| Q4_K, Q6_K                       | T (`in_f` a multiple of 256) | G, tuned | —                   | every Q4_K_M file                                  |
| Q4_0                             | T                            | G        | —                   |                                                    |
| Q5_0, Q8_0, Q4_1, Q5_1, IQ4_NL   | R                            | G        | —                   | gemma-3 and Qwen2.5 Q4_K_M files; Qwen3.5 UD files |
| Q5_K, Q3_K, Q2_K, IQ4_XS         | R                            | G        | —                   | ERNIE-21B, Qwen3.5 UD, gemma-4 E2B                 |
| the other IQ, TQ and fp4 formats | R                            | G        | — or small-m        | rare                                               |

Every format has some f32 kernel, so a float lineup is reachable with no new
kernel. What is missing is speed: the R cells and the MoE batched column.

## Design

- **One decision per model.** A lineup value on the graph (`Int8` or `Float`),
  computed by the seam once per model from every Linear's format and shape and
  the device, through a backend hook, and stamped on every graph the seam builds
  (layer spans and multi-GPU segments see subsets of the Linears, the problem
  `Graph::part_of_routed` exists for). The backend answers from a new tuning
  parameter, `linear.lineup` = `int8`, `float` or `auto`.
- **One gate for every int8 site.** Each bare read of `caps.i8_dot` in the
  Linear, verify and MoE lowering becomes "the device has the dot and this
  graph's lineup is int8"; a test scans the source so no other read appears.
  Under Float the f32 kernels are chosen unconditionally for their formats, and
  a Float graph that reaches an int8 arm is an error by name, never a fallback.
  Under Int8 the f32 decode kernel is off for that model too, which ends class
  (b)'s decode mix.
- **`gemv.kq_fma` and `gemm.kq_fma`** stop choosing lineups. They are retired or
  kept as speed choices inside Float (the owner's call below).
  `mmv_mw.moe_float` is subsumed.
- **Kernels by format, not one shader each.** `native_gemm_kq_fma.comp` and
  `native_gemv_kq_fma.comp` already select their staging with `-DFMT_*` and
  share `native_decode.glsl`, as the mmq shaders do. New formats are arms:
  specialised ones for the common formats, and one generic block-decode arm for
  the rest (**guess:** far faster than R, slower than a specialised arm).
  Thirty-two-element formats need only `in_f` a multiple of 32, which is what
  gemma-3 (1152) and Qwen2.5 (896) need.
- **No int8 fallback when a tile does not fit.** Today a grid wider than the
  device allows falls through to the int8 arm. The tile gets a two-dimensional
  grid as `native_gemm_fma.comp` has, so only shared memory can decline it, and
  then the fallback is the float per-row kernel.
- **MoE experts.** An expert-grid build of the f32 tile, resident then paged,
  reading f32 activation rows directly (no quantize, no packed copy), with a
  short tile for the few rows an expert sees.
- **Names** of new kernels avoid the substrings the tolerance harness treats as
  int8 (`INT8_KERNEL_MARKERS`).
- **Reporting.** One log line per model load (lineup, cause, how many Linears
  are tiled, vector or per-row), the parameter in `infr tune show`, and the
  decision with its reasons in `infr tune check`.

## Verification

Under the accuracy rule, each slice shows its output is in the right class.

- **Per kernel and build, none left untested:** a host-reference tolerance test
  (the `nc_gemm_parity` pattern), plus same-run comparisons that must be bit for
  bit: twice in a row, streamed against resident, split grid against unsplit,
  paged against resident, and each decode kernel against its multi-row twin row
  for row. No hash goldens.
- **The forced gate:** `INFR_NO_COOPMAT=1` with the lineup forced to float, with
  and without forced wave64, on both GPUs here. Every gated model's shipped runs
  must sit inside its Float row and build no int8 kernel; the harness scores a
  run against the class the backend reports, not a fixed one.
- **What forcing cannot show:** a GCN driver's code generation and rounding, its
  limits and watchdog, and speed. Those need the RX 580.
- **Our GPUs do not change:** with `auto`, an accelerated dot keeps int8, shown
  by the tuning snapshots (one new line) and the kernel lists in the tolerance
  tests.

## Slices

Each ships alone through coder then reviewer.

| #   | Slice                                                                                                                                                                                                                                                                                                          | Unlocks                                                                                      |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| S1  | The lineup plumbing, no new kernel: the graph stamp, the backend hook, `linear.lineup` (default `int8`), the single gate at every int8 site, f32 kernels unconditional under Float, the f32 decode kernel for streamed weights, the two-dimensional tile grid, and the seam's MoE chunk cap asking the backend | opt-in Float for every model; Q4_K_M dense files fully tiled                                 |
| S2  | Default `auto`: Float where the dot is not accelerated and the model is fully tiled; class (a) always Float                                                                                                                                                                                                    | Q4_K_M dense models float by default on class (b), after the RX 580 measurements             |
| S3  | Tile arms for the 32-element formats (Q5_0, Q8_0, Q4_1, Q5_1, IQ4_NL) and the multiple-of-32 gate                                                                                                                                                                                                              | gemma-3 and Qwen2.5 Q4_K_M                                                                   |
| S4  | The f32 expert GEMM, resident then paged                                                                                                                                                                                                                                                                       | Qwen3-30B-A3B; ERNIE-21B with S5                                                             |
| S5  | K-quant arms (Q5_K, Q3_K, Q2_K, IQ4_XS), then the generic arm for the rest                                                                                                                                                                                                                                     | Qwen3.5 UD files, low-bit files                                                              |
| S6  | Tuned f32 decode GEMVs for Q5_K, Q8_0, Q5_0 with their multi-row twins                                                                                                                                                                                                                                         | only if the RX 580 shows the generic GEMV well behind                                        |
| S7  | GPUs with a native int8 dot: per kernel family on both GPUs here, float against int8 for accuracy (tolerance rows) and speed (alternated, quiet window); float becomes the default where it is not slower                                                                                                      | better accuracy at no cost on the hardware most users have; the rest reported for a decision |

## Found on the way, to fix whatever this plan does

- **A no-int8-dot device sends full MoE prefill chunks through the small-m
  path.** The seam caps the chunk only when the expert widths fail
  (`moe_small_m_chunk_cap`), and `moe_batched_ok` never asks about `i8_dot`, so
  such a device records the per-row expert path at full chunk height, the shape
  `moe_small_m_threshold`'s doc warns can hang a GPU. Read from code, not run.
  S1 fixes it by having the seam ask the backend.
- **Class (b) streamed weights skip the f32 decode kernel** (`quant_linear` is
  called without a buffer on that path) and fall to int8: a second mix.

## Owner's decisions (2026-11-26)

1. **`auto` goes Float even when some Linears land on the slow per-row kernel.**
   Accuracy first; prefill on the uncovered formats is slow until their tiled
   arms land. It ships only after a watchdog check of the per-row path at full
   chunk height on the forced tier.
2. **Multi-GPU: Float if any device needs it.** When one device of a set lacks a
   native int8 dot, the whole model runs Float on every device. It logs a
   warning naming the device that forced it and what it costs the others, and it
   is documented in `docs/config.md` and the multi-GPU docs.
3. **`gemv.kq_fma` and `gemm.kq_fma` are retired** when `linear.lineup` lands,
   with the config alias `kernels.vulkan.kq_fma_gemv`; a removed key is a load
   error naming `linear.lineup`. A `Breaking` changelog entry.
4. **Class (b)'s decode mix today is decided from its measurement** (taken with
   the PR #112 follow-up): failing a tolerance row means default off at once;
   passing everywhere means it stays until the lineup replaces it; inside its
   bound but further from the reference than int8 goes to the owner with the
   figures.
5. **GPUs with a native int8 dot: measure, then prefer float where it is free.**
   Its own slice (S7), per kernel family on both GPUs here: float becomes the
   default where it is not slower; where int8 is clearly faster it stays, with
   the accuracy and the speed gap recorded for the owner to decide case by case.

## To measure first

- On the RX 580: prefill speed of the generic arm and of the unaligned-block
  arms against int8; the expert tile's size; the f32 expert path's memory
  traffic (four bytes an activation against one).
- Whether a cross-phase mix (int8 prefill, f32 decode) is also harmful. Only the
  within-prefill mix has been measured, and why a mix is worse than all-int8 is
  not established.
