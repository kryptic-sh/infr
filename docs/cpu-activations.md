# CPU int8 activations

The production CPU backend quantizes each activation row to int8 before an
`Op::Linear` (or an MoE expert bank) dots it against a quantized weight, so the
weight never expands to f32. Weights with a 256-element super-block (Q4_K, Q5_K,
Q6_K, Q2_K, Q3_K, the IQ and TQ formats, and Q8_0 when the row width is a
multiple of 256) take `quant::quantize_q8`: one scale per 256 activations,
`max|x|/127`. The formats with 32-element blocks take `quantize_q8_32`, one
scale per 32. Reference mode (`kernels.cpu.reference`) dots the decoded weight
against the f32 activation instead; it is the oracle every number below is
scored against.

## The first-token defect

The first token of a sequence carries a few enormous activations (the
"massive-activation" dimensions transformer LMs put there). In a 256-element
block holding one, `max|x|/127` is set by it, and every other element of the
block rounds to 0 or ±1. Those rows are the first token's queries, keys and
values, so the error does not stay at position 0: every later token attends to
that key and value.

### Where it comes from (measured 2026-11-18)

`crates/infr-llama/tests/cpu_act_quant.rs` scores the production CPU against the
reference position by position, on the reference's own greedy tokens (three
chat-template prompts and one raw prompt, eight decoded tokens each). The tables
below were taken with each whole sequence in one prefill and an f64 KL of the
test's own; the teacher-forced driver (`SeamModel::forced_logits`: the prompt
prefilled, then one decode step per token) with `infr_llama::tolerance`'s KL,
which the test uses now, gives the same figures at every position, as the CPU's
row-count identity says it must (checked after the fix on all five models). Its
`localise_the_position_0_error` test taps every `Op::Linear`
(`infr_cpu::linear_tap`) and, for chosen rows, recomputes the op's output
exactly from the same input row: the op's own error, free of what upstream ops
passed on. Its `per_position_against_the_reference` test can also make the
reference dot a round trip of each activation row through a simulated quantizer
(`linear_tap::simulate`), which scores a scheme end to end before any kernel
computes it.

What it found, on SmolLM3-3B Q4_K_M first:

- Reference and production differ only in `Op::Linear` (and the MoE banks), so
  the first diverging op is a `Linear`. The kernels compute exactly the dot of
  the dequantized activation: every op's output matches the f64 dot of the
  weight with the round-tripped activation to within 3.3e-7 relative, on every
  row. No accumulator saturates or wraps; the error is the quantizer's.
- Row 0's local error is larger than the other rows' in most projections, and
  largest where the output is small against its inputs: the V projection of
  three late layers runs 0.13 to 0.19 relative error at row 0 against 0.015 to
  0.028 at rows 1 and 134. Position 0's attention output is its own value row,
  so that error goes straight into the residual stream; the final hidden state
  of row 0 ends 0.205 away from the reference's (rows 1 and 134: 0.05 to 0.09).
- The rows entering the attention and FFN projections have, at row 0, a largest
  block crest (block maximum over RMS) of 8.8 to 16.0 — 16 is the most a
  256-element block can have, one element holding all its energy — and 3 to 47%
  of their elements quantize to zero; at rows 1 and 134, crests of 3.7 to 13.0
  and 0.6 to 5.9% zeros.
- The simulated 256-block quantizer reproduces production's position-0 KL (5.582
  against 5.581), so the quantizer is the whole cause.

Per-position KL of the production CPU against the reference before the fix
(position 0 / position 1 / median and worst of positions 2 on, per prompt):

| Model                   | position 0      | position 1      | positions 2.. median | worst |
| ----------------------- | --------------- | --------------- | -------------------- | ----- |
| SmolLM3-3B (chat / raw) | 5.58 / 1.92     | 1e-5 / 5.7e-4   | 0 to 1.1e-3          | 0.22  |
| Qwen3-0.6B              | 0.086 / 0.075   | 7.2e-3 / 2.4e-3 | 1.8e-3 to 5.6e-3     | 0.11  |
| Qwen3-1.7B              | 0.020 / 0.010   | 4.1e-4 / 9.0e-3 | 2.6e-4 to 4.8e-3     | 0.073 |
| ERNIE-4.5-0.3B          | 0.051 / 0.60    | 0.19 / 0.025    | 5.6e-3 to 1.4e-2     | 0.10  |
| gemma-3-1b              | 8.7e-4 / 7.0e-4 | 4.7e-3 / 7.9e-3 | 2.6e-3 to 7.2e-3     | 0.67  |

ERNIE's position 1 is the same defect one step on: with two keys, its attention
output is half position 0's value row.

### llama.cpp has it too

llama.cpp's CPU quantizes the activation of every K-quant matmul to `Q8_K`
(`quantize_row_q8_K_ref`, which the x86 build calls directly): `QK_K = 256`
elements per block, `iscale = -127/max` with `max` the block's largest
magnitude, so one scale per 256 elements — the scheme infr used. Its repacked
AVX2/AVX-512 path (`ggml_quantize_mat_q8_K_4x8`) blocks the same way. Measured
with `llama-debug --save-logits` (CPU build, one-token prompts, so the dumped
row is position 0) through `llama_cpp_last_row_against_the_reference`:

| Model, row                       | llama.cpp vs reference | infr before | infr after | infr before vs llama.cpp | infr after vs llama.cpp |
| -------------------------------- | ---------------------- | ----------- | ---------- | ------------------------ | ----------------------- |
| SmolLM3-3B, `The`, position 0    | 1.85                   | 1.92        | 3.8e-5     | 2.0e-3                   | 9.53                    |
| Qwen3-0.6B, `The`, position 0    | 0.076                  | 0.075       | 2.0e-3     | 1.9e-4                   | 0.075                   |
| ERNIE-4.5-0.3B, BOS, position 0  | 0.128                  | 0.129       | 0.012      | 2.1e-4                   | 0.094                   |
| SmolLM3-3B, 12-token prompt, end | 4.9e-5                 | 1.9e-5      | 2.0e-5     | 5.3e-5                   | 8.7e-5                  |

So the defect is the `Q8_K` scheme's, and infr matched llama.cpp's position 0 to
KL 2e-4 to 2e-3 before the fix. After it, infr's position 0 is close to the
reference and therefore far from llama.cpp's; at a later position the two
engines agree as before.

## The split

`quantize_q8` now splits a block whose largest one or two magnitudes stand far
above the rest. Of a block's `HI_KEEP` (2) largest magnitudes, those above
`HI_RATIO` (16) times the next largest move to a sparse part (`Q8::hi`): a
one-block `Q8` on their own scale, `max/127`, whose codes are zero elsewhere.
What that leaves of them stays in the dense block, which takes the scale of what
is left, so every element of the block is represented to within half a step of
the block's non-outlier scale. A remainder that still exceeds the rest of the
block takes a second part (`HI_LEVELS` caps it at three). A block with nothing
to split quantizes exactly as before, to the same bits.

A dot over the activation is the dense dot plus each part's one-block dot
against the weight's matching 256-element block (`kernels::vec_dot_hi`), summed
from zero in the parts' order and added once: `dense + vec_dot_hi(..)`. Every
route that dots a `Q8` adds it — the one-row `Op::Linear`, the prefill batch
kernels and tiles, the packed VNNI GEMM, the MoE expert banks — so the value
does not depend on the route. The parts use the existing one-row kernels, so no
kernel changed and every format and tier is covered as it was.

### Identity

- **Tiers.** The dense dot and each part's dot are the existing one-row kernels,
  which give the scalar bits at every tier; the sum is one expression.
  `the_sparse_activation_part_gives_the_scalar_bits_at_every_tier` checks every
  format.
- **Row count and chunk height.** Each row is quantized on its own and every
  route adds the same sum, so a row's result does not depend on the batch.
  `linear_batch_identity`'s sweep includes a split row (its spike pattern), and
  `first_token_spikes_keep_the_rest_of_their_block` checks row 0 of a prefill
  against the same row alone.
- **The rule is per block and platform-independent**: integer comparisons of f32
  magnitudes times a power of two, no fast-math, no data-dependent order.

### Options measured

Each simulated in the reference (`linear_tap::simulate`; the 32-block weights
keep production's 32-block quantizer), position-0 KL on a chat prompt and on the
raw prompt (one figure where only the chat prompt was run):

| Scheme                                     | SmolLM3-3B    | Qwen3-0.6B     | Qwen3-1.7B     | ERNIE-0.3B     |
| ------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| 256 blocks (before)                        | 5.58, 1.90    | 0.086, 0.075   | 0.020, 0.010   | 0.051, 0.60    |
| 32 blocks                                  | 0.032         | 0.025, 0.034   | 2.3e-3         | 1.4e-3, 3.8e-3 |
| 16 blocks                                  | 2.7e-3        | 0.013, 0.019   | 3.6e-4, 7.1e-4 | 6.4e-4, 1.9e-3 |
| 256 with power-of-two 32-element subscales | 0.060, 1.2e-4 | 0.021, 0.036   | 4.0e-3, 2.0e-3 | 1.7e-3, 5.6e-3 |
| 256 with power-of-two 16-element subscales | 4.9e-3, 4e-5  | 0.013, 0.019   | 5.4e-4, 1.6e-3 | 7.7e-4, 2.9e-3 |
| 256, two outliers exact (no gate)          | 1e-5          | 6.8e-4, 5.7e-4 | 3.4e-4, 6.6e-4 | 1.2e-3, 4.0e-3 |
| 256, split at ratio 4                      | 8e-5, 0       | 1.7e-3, 1.4e-3 | 1.3e-3, 1.4e-3 | 5.4e-3, 5.3e-3 |
| 256, split at ratio 8                      | 2e-5, 2e-5    | 2.0e-3, 1.9e-3 | 2.8e-3, 1.6e-3 | 7.4e-3, 1.0e-2 |
| **256, split at ratio 16 (committed)**     | 2.5e-4, 5e-5  | 4.1e-3, 1.8e-3 | 3.0e-3, 6.5e-3 | 7.5e-3, 1.5e-2 |
| 256, split at ratio 32                     | 1.2e-4, 2e-5  | 4.3e-3, 4.4e-3 | 3.3e-3, 3.4e-3 | 7.7e-3, 1.6e-2 |
| 256, split at ratio 64                     | 0.30, 7.6e-4  | 5.8e-3, 4.4e-3 | 4.4e-3, 3.2e-3 | 0.049, 0.61    |
| int16, 256 blocks                          | 0             | 4e-5, 1e-5     | 1e-5           | 1e-5, 7e-5     |

- **32-element scales everywhere** (what the GPU does) leave SmolLM3's position
  0 at 0.032, and need every K-quant kernel's epilogue to take a float scale per
  sub-block: a rewrite of every family, tier and tile.
- **Power-of-two subscales** fold into the integer weight scales as shifts, so
  they would cost little per block, but they need the same rewrite, and at 32
  elements leave SmolLM3 at 0.060.
- **Exact outliers in a float side path** need a per-element weight decoder for
  every format; ungated, every block pays for two.
- **int16 activations** are exact for practical purposes but halve the integer
  throughput of every kernel.
- **The split** needs no kernel change and costs nothing on a row that does not
  split; a split block costs a second one-block kernel call per weight row,
  several times a block of the packed GEMM. The ratio is the trade: on
  `infr bench`'s prompt at 8 it splits 0.4 to 5.5% of the blocks entering the
  projections at positions past 0 (one special token aside; 16: 0 to 1.1%, 32: 0
  to 0.3%), which measured 14 to 17% slower Qwen3-1.7B pp512 in three pairs with
  the other agent compiling, so 8 was dropped. At 64 ERNIE's and SmolLM3's first
  tokens are not fixed; 32 is the last ratio that fixes them both, with no
  margin. 16 is committed (backlog B210). What the ratio does not reach at 16 is
  ERNIE's position 1 on its chat template (below).

### After the fix

| Model                   | position 0      | position 1      | positions 2.. median | worst |
| ----------------------- | --------------- | --------------- | -------------------- | ----- |
| SmolLM3-3B (chat / raw) | 1.4e-4 / 4e-5   | 1e-5 / 1.4e-3   | 0 to 1.3e-3          | 0.33  |
| Qwen3-0.6B              | 4.0e-3 / 2.0e-3 | 6.6e-4 / 2.6e-3 | 1.8e-3 to 5.3e-3     | 0.40  |
| Qwen3-1.7B              | 3.0e-3 / 6.5e-3 | 2.8e-4 / 9.9e-3 | 2.6e-4 to 3.2e-3     | 0.050 |
| ERNIE-4.5-0.3B          | 7.7e-3 / 0.013  | 0.13 / 0.020    | 5.0e-3 to 9.9e-3     | 0.055 |
| gemma-3-1b              | 3.0e-4 / 8.5e-4 | 3.8e-3 / 0.016  | 2.9e-3 to 7.0e-3     | 0.26  |

Every position 0 is now at its model's steady-state order. The medians are
level: of the 17 per-prompt medians that are not zero, 8 are lower and 9 higher,
the ratios after/before running from 0.57 to 2.4 (gemma-3-1b's second prompt,
3.0e-3 to 7.0e-3), the spread two runs of either quantizer show. The worst
positions are single rows near a tie in the reference, and they move both ways
(gemma 0.67 to 0.21, Qwen3-0.6B's 0.11 to 0.40 at the same position 18).

ERNIE-4.5-0.3B's position 1 on its chat template went from 0.19 to 0.13, not to
its steady state: that row's own outliers stand 8 to 16 times above the rest of
their blocks (at ratio 8 it reaches 0.029). It is the place the ratio trade
shows (B210).

### Speed

`infr bench --dev cpu`, Windows, native builds, three reps per run, eight rounds
with `origin/main` (`old`), the split at 16 (`16`, as committed) and at 32
(`32`) interleaved in each, tok/s. Another agent's test suite ran through all
eight rounds (`cargo-nextest` in the process list at every round's start), which
this box's noise floor shows: single runs drop by up to two thirds.

| Run, model        | old                                                            | 16                                                            | 32                                                             |
| ----------------- | -------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------- |
| pp512, Qwen3-1.7B | 1038.5, 1099.6, 1039.1, 1080.3, 1015.0, 1129.8, 1106.3, 1128.4 | 1030.4, 1078.4, 377.5, 1045.7, 1073.3, 1066.3, 1045.2, 1031.4 | 1113.6, 1091.2, 1133.9, 1082.6, 1099.6, 1119.1, 1130.2, 1070.8 |
| pp512, gemma-3-1b | 569.1, 560.9, 557.5, 536.9, 579.8, 568.7, 560.6, 544.4         | 539.8, 538.2, 564.8, 540.8, 546.2, 566.1, 540.9, 493.3        | 545.1, 555.9, 566.1, 558.8, 538.6, 541.9, 526.4, 501.9         |
| tg64, Qwen3-1.7B  | 51.6, 53.1, 50.2, 16.5, 51.6, 54.5, 54.5, 47.5                 | 55.7, 40.0, 52.8, 49.7, 53.5, 55.8, 54.8, 53.3                | 56.3, 53.6, 55.2, 54.6, 54.8, 55.7, 52.6, 51.0                 |
| tg64, gemma-3-1b  | 44.2, 45.0, 45.8, 42.5, 41.3, 45.2, 45.2, 41.2                 | 43.3, 27.7, 43.6, 45.6, 44.1, 45.3, 40.4, 41.3                | 45.0, 43.7, 42.2, 43.5, 45.1, 43.2, 45.5, 19.7                 |

Paired, `16` against `old`: Qwen3-1.7B pp512 -0.8, -1.9, (round 3 disturbed),
-3.2, +5.7, -5.6, -5.5, -8.6% (median of the seven -3.2%); gemma-3-1b pp512
-5.1, -4.0, +1.3, +0.7, -5.8, -0.4, -3.5, -9.4% (median -3.8%). `32` against
`old` on Qwen3-1.7B pp512: +7.2, -0.8, +9.1, +0.2, +8.3, -0.9, +2.2, -5.1%
(median +1.2%); on gemma-3-1b pp512 -4.2, -0.9, +1.5, +4.1, -7.1, -4.7, -6.1,
-7.8% (median -4.5%). Decode is level within the noise on both models at both
ratios. So on Qwen3-1.7B the split at 16 costs prefill about 3%, which 32, with
far fewer blocks split past position 0, does not; gemma-3-1b is about 4% slower
at both ratios, which the split blocks do not explain (at 32 they are under
0.3%) and which these disturbed rounds cannot tell from noise. Not resolved: a
quiet window would settle both (backlog B210). The quantizer itself: 1.41
against 1.35 µs per 2048-wide ordinary row (`quant::tests::microbench`); a
first-token row with two parts, 7.4 µs, and its 64 Q4_K rows 2.73 against 1.92
µs.

A baseline build (`-C target-cpu=x86-64`) was measured only against an earlier
form of the split whose quantizer made four passes over every block where it now
makes three: Qwen3-1.7B pp512 +0.7, -1.9, -2.4, -2.6% in the four rounds with no
other build or test running, and -5.1 to -36.7% in four that had one; decode
level. Not re-measured since.

### What it does not fix

gemma-3-1b's large mid-sequence KLs (0.40 to 0.67 at its prompts' last
positions) are not outliers: those rows split the fewest blocks of any, their
own per-op errors are ordinary, and the distance builds up through the layers.
In simulation, int16 activations on the 256-block weights alone take the worst
one from 0.52 to 0.09, on the 32-block weights (gemma's Q5_0 and Q8_0
projections at width 1152) alone to 0.32, and on both to 1e-5: it is the int8
activations' ordinary noise on a model that amplifies it, on both paths (backlog
B209).
