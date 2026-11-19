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
does not depend on the route.

### How a part is dotted

A part's dot is defined as the one-row kernel's value on a one-block activation
holding the part's codes (`vec_dot_q8_any` over 256 inputs, which gives the
scalar kernel's bits at every tier). The split's first form computed exactly
that: a whole 256-element kernel call per part and weight row, for one or two
nonzero codes. `kernels/hi.rs` now evaluates the scalar kernel's own expression
from the weight elements at the codes alone, so every value keeps its bits:

- **Integer sums.** Every kernel sums `weight · code` (and the codes, for a min
  or offset term) in integers per sub-block. Those sums are exact, so the
  nonzero codes alone give the same integers in any order.
- **One float epilogue per block** (Q4_K, Q5_K, Q2_K: `0 + dh·(d·sd − dmin·sm)`;
  Q6_K, Q3_K, IQ4_XS, TQ2_0, TQ1_0: `0 + d·dh·isum`): evaluated as written, from
  the same integers and the same decoded scales.
- **A float term per sub-block** (Q8_0's 32-element blocks, the grid formats'
  8-element groups), added in order to a sum from `+0.0`. A sub-block with no
  code adds `scale · 0`, which is `±0.0` when the scale is finite; such a sum is
  never `-0.0` and `x + ±0.0 == x` for any other `x`, so that term is skipped. A
  sub-block whose scale is not finite (or whose product with the part's scale
  could overflow) is added as the kernel adds it, so its `NaN` reaches the
  result. The grid formats still expand their one block (`*_expand_row`) per
  part and weight row, and so stay slower than the rest (backlog B223).

In a prefill (`kernels::HiBatch`) the element factors every part needs are
decoded once per weight row into a small table: outlier channels repeat across
rows, so an FFN-down op of Qwen3-1.7B's bench prompt needs about 30 distinct
elements in about 18 blocks for over 500 parts. Each part is then two integer
multiply-adds and the epilogue, run across eight weight rows at once. The packed
VNNI GEMM adds the parts inside each 8-row group's task, right after the group's
results are stored and while its weight rows are in cache. The o-major routes
(the other formats, and every format without VNNI) add them in a pass of their
own, 64 weight rows a task; an MoE bank's task adds them over its own weight
rows after its dense dot. The first form (each part made whole and dotted by
`vec_dot_q8_any`) stays in the tests as `vec_dot_hi_reference`, and
`the_fast_sparse_dot_gives_the_one_row_kernels_bits`,
`linear_routes_add_the_reference_sparse_dot_bitwise` and
`expert_gemm_range_adds_the_reference_sparse_dot_bitwise` hold the new path to
it bit for bit (every format, tier, spike pattern, row count and route).

### Where it fires

Measured by the review on 2026-11-18 with an instrumented build (not reproduced
here): on the inputs of the attention and FFN-in (gate/up) projections the split
is mostly a row-0 event, but on the inputs of the FFN down projections it fires
all through the sequence. Rows of the down-projection input with at least one
split block, Q4_K / Q6_K down weights:

| Prompt               | Qwen3-1.7B   | gemma-3-1b | SmolLM3-3B  |
| -------------------- | ------------ | ---------- | ----------- |
| `infr bench`'s pp512 | 5.4% / 16.1% | 32% / 38%  | not counted |
| a real chat prompt   | 6.6% / 13.5% | 16% / 23%  | 1.6% / 3.9% |

That is about 1,930 split blocks per forward on Qwen3-1.7B and about 7,100 on
gemma-3-1b (bench prompt). No block anywhere had three spikes, and about six
times as many unsplit blocks sit at 8 to 16 times their next magnitude as split.
So on FFN-down inputs the split acts as a general two-scale quantizer, and that
is where all of its cost is (below).

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
- **Across systems.** `op_hashes`' spiky cases (`Linear spiky …` for every
  256-block format at 1, 5 and 19 rows, two 32-block formats, three
  `MoeFfn spiky …`) feed rows whose blocks hold no spike, one, two of opposite
  signs, three (the two-element cap), a spike exactly 16 times the next
  magnitude and one just over, two equal spikes, a spike over an all-zero rest,
  and two far apart that need a second part. Their hashes are checked in from
  CI's Linux, Windows and macOS runs.
- **The rule is per block and platform-independent**: float comparisons of f32
  magnitudes, one side multiplied by a power of two (exact), no fast-math, no
  data-dependent order.
- **The exact sparse dot against its reference.** Planted one at a time
  (2026-11-19), each of these turned the new tests red: a Q4_K sub-block's scale
  taken from the next sub-block, a Q6_K element index off by one, a Q6_K scale
  from the next 16-element sub-block, Q2_K's traversal with `j` and `half`
  swapped, the one-epilogue product reassociated (`dh·(d·s)`), the parts summed
  in reverse (one row, and in the batch table), Q8_0's sub-blocks summed in
  reverse, the code-less sub-block or group skipped whatever its scale (Q8_0,
  the grid formats: only the NaN-scale weights catch it), a part's second code
  dropped in the batch table, and the packed GEMM's parts (or its tail rows'
  parts) not added. One plant changed nothing: dropping the leading `0 +` of the
  Q4_K-family epilogue, whose `-0.0` the sum over parts, which also starts at
  `+0.0`, turns to `+0.0` anyway.
- **The quantizer's no-split bound** (`quant::cannot_split`, below) clears a
  block from the max pass only when no element can split; its test drives blocks
  a few ulps either side of the bound, and with the bound's rounding margin
  removed (254 in place of 254.25) it clears a block that splits.

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
- **The split** needs no kernel change. A row with no split block costs one
  extra pass of the quantizer and nothing in the dots. As first landed, each
  split block cost a second one-block kernel call per weight row, about 70 to 90
  ns, roughly 15 to 20 times a block of the packed GEMM (the review's per-op
  profile), and because the FFN-down inputs split all through a sequence
  (above), that was a steady prefill cost in those layers; the exact sparse dot
  ("How a part is dotted") brought it within the noise (Speed). The ratio is the
  trade: at 8 the split fires on several times as many blocks and measured 14 to
  17% slower Qwen3-1.7B pp512 in three pairs with the other agent compiling, so
  8 was dropped. At 64 ERNIE's and SmolLM3's first tokens are not fixed; 32 is
  the last ratio that fixes them both, with no margin. 16 is committed (backlog
  B210). What the ratio does not reach at 16 is ERNIE's position 1 on its chat
  template (below).

### After the fix

| Model                   | position 0      | position 1      | positions 2.. median | worst |
| ----------------------- | --------------- | --------------- | -------------------- | ----- |
| SmolLM3-3B (chat / raw) | 1.4e-4 / 4e-5   | 1e-5 / 1.4e-3   | 0 to 1.3e-3          | 0.33  |
| Qwen3-0.6B              | 4.0e-3 / 2.0e-3 | 6.6e-4 / 2.6e-3 | 1.8e-3 to 5.3e-3     | 0.40  |
| Qwen3-1.7B              | 3.0e-3 / 6.5e-3 | 2.8e-4 / 9.9e-3 | 2.6e-4 to 3.2e-3     | 0.050 |
| ERNIE-4.5-0.3B          | 7.7e-3 / 0.013  | 0.13 / 0.020    | 5.0e-3 to 9.9e-3     | 0.055 |
| gemma-3-1b              | 3.0e-4 / 8.5e-4 | 3.8e-3 / 0.016  | 2.9e-3 to 7.0e-3     | 0.26  |

Every position 0 is now at its model's steady-state order. Away from position 0,
as the review measured it (geometric mean over positions 1 on of the KL ratio
after/before; the same prompts as the tables):

| Model          | KL after/before, positions 1 on |
| -------------- | ------------------------------- |
| SmolLM3-3B     | 0.74                            |
| Qwen3-1.7B     | 0.85                            |
| gemma-3-1b     | 0.98                            |
| Qwen3-0.6B     | 0.87                            |
| ERNIE-4.5-0.3B | 0.70                            |

So level to slightly better overall, with single positions moving both ways,
mostly rows near a tie in the reference: worse at Qwen3-0.6B position 18 (0.109
to 0.402) and 23 (0.033 to 0.203), SmolLM3's chat positions 229 and 240 and
gemma-3-1b position 22 (0.049 to 0.245); better at gemma-3-1b position 24 (0.667
to 0.148).

ERNIE-4.5-0.3B's position 1 on its chat template went from 0.19 to 0.13, not to
its steady state: that row's own outliers stand 8 to 16 times above the rest of
their blocks (at ratio 8 it reaches 0.029). It is the place the ratio trade
shows (B210).

### MoE models, one row

One-row `Op::MoeFfn` takes the int8 expert kernels since this change's companion
commit. With the prefill chunk height at 1 (`INFR_AQ_UBATCH=1`), so every
position, position 0 included, is a one-row graph, production against reference
on the first chat prompt and the raw prompt:

| Model             | old quantizer: position 0 | position 1     | split: position 0 | position 1     | positions 2.. median, split |
| ----------------- | ------------------------- | -------------- | ----------------- | -------------- | --------------------------- |
| Qwen3-30B-A3B     | 0.16 / 1.21               | 0.025 / 0.018  | 9.1e-4 / 9.1e-3   | 0.020 / 7.7e-3 | 2.0e-3 / 2.3e-3             |
| ERNIE-4.5-21B-A3B | 9.8e-3 / 0.016            | 0.041 / 6.6e-3 | 1.6e-3 / 3.3e-3   | 0.032 / 5.2e-3 | 6.5e-3 / 3.0e-4             |

The first is the defect the earlier review found when one-row MoE first moved to
int8 (1.21 on the raw prompt); with the split, position 0 is ordinary on both.

**Speed.** `infr bench --dev cpu`, native release builds of `1b16e66b` (before:
the f32 one-row path and no split) and `afbb3b14` (after: one-row int8 and the
split), ten interleaved rounds per model on the quiet machine of the Speed
section below, tok/s:

| Model, run              | before                                                               | after                                                                |
| ----------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Qwen3-30B-A3B tg64      | 27.6, 27.5, 27.6, 28.0, 27.2, 27.6, 27.7, 27.7, 27.6, 27.5           | 28.2, 27.8, 28.1, 28.4, 28.4, 28.0, 27.6, 28.4, 28.4, 28.5           |
| Qwen3-30B-A3B pp512     | 419.9, 417.7, 417.9, 422.3, 411.4, 416.2, 417.6, 415.5, 415.7, 418.7 | 413.5, 414.3, 413.7, 407.6, 415.9, 413.2, 415.6, 414.1, 417.4, 416.1 |
| ERNIE-4.5-21B-A3B tg64  | 26.2, 26.8, 26.5, 26.5, 26.5, 26.7, 26.4, 26.3, 26.7, 26.6           | 27.4, 27.2, 27.0, 27.2, 27.5, 27.0, 26.6, 27.4, 27.1, 27.1           |
| ERNIE-4.5-21B-A3B pp512 | 469.9, 470.2, 468.0, 471.8, 471.8, 473.7, 474.5, 469.3, 471.4, 472.2 | 455.9, 460.4, 460.1, 456.2, 457.6, 455.5, 459.5, 460.3, 458.4, 459.7 |

Decode is faster: median paired +2.0% on Qwen3-30B-A3B (faster in 9 of 10
rounds) and +1.9% on ERNIE-4.5-21B-A3B (10 of 10). The two builds differ in both
the one-row int8 expert path and the split, so this is the two together.
Prefill, which the one-row change does not touch, pays the split: -2.9% on ERNIE
(10 of 10), and -0.7% on Qwen3-30B-A3B (8 of 10, inside the noise).

### Speed

The split as it first landed cost 3.5% of Qwen3-1.7B's prefill and 1.9% of
gemma-3-1b's on a native build (1.65% and 1.5% on a baseline build), all of it
in the FFN-down layers, and 2.9% of ERNIE-4.5-21B-A3B's MoE prefill (above).
Dotting the parts exactly and cheaply ("How a part is dotted") removes that cost
to within the protocol's noise, with no output bit changed: see "Speed with the
exact sparse dot" below. The rounds that follow here measured the first form.

Measured on a quiet machine on 2026-11-19 (no other build, test or model run;
the process list read before every round showed only an idle, hung `cargo test`
of another project): `infr bench --dev cpu` (three reps a run, tok/s), Windows,
release builds of `1b16e66b` (`before`, the parent of the split's commit
`18d5f2bf`) and `afbb3b14` (`split`, `main` with the split and the later
commits, none of which touches a dense model's CPU path), native and baseline
(`-C target-cpu=x86-64`). Twelve rounds, the binaries' order reversed every
other round; the same rounds also measured the FMA tier (`docs/math.md`):

| Model, run          | native `before`                                                                                | native `split`                                                                                 | baseline `before`                                                                              | baseline `split`                                                                             |
| ------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Qwen3-1.7B pp512    | 1125.5, 1158.2, 1149.8, 1147.5, 1151.7, 1140.4, 1146.9, 1145.5, 1145.9, 1144.0, 1114.2, 1121.7 | 1113.7, 1129.3, 1124.4, 1106.7, 1105.8, 1092.0, 1101.1, 1127.3, 1097.3, 1104.5, 1047.9, 1106.8 | 1028.5, 1025.8, 1042.2, 1023.9, 1030.8, 1027.3, 1032.0, 1041.0, 1044.8, 1021.7, 1032.5, 1025.2 | 990.9, 1014.8, 1014.6, 1014.6, 1007.8, 1016.5, 1025.4, 1019.8, 993.9, 1010.7, 1019.5, 1002.5 |
| gemma-3-1b-it pp512 | 595.6, 596.6, 596.6, 590.9, 590.0, 596.6, 588.7, 597.5, 594.8, 590.9, 601.0, 601.7             | 578.0, 584.5, 588.0, 585.1, 588.1, 585.6, 592.3, 580.3, 582.9, 582.8, 581.7, 583.5             | 580.7, 590.1, 592.5, 592.7, 592.4, 578.8, 586.3, 581.6, 589.3, 588.4, 592.2, 587.8             | 582.1, 580.8, 583.5, 571.1, 586.5, 573.5, 578.5, 573.0, 580.0, 576.9, 582.4, 581.6           |
| Qwen3-1.7B tg64     | 57.8, 57.4, 57.2, 57.9, 57.0, 58.0, 57.4, 57.4, 57.8, 58.3, 57.6, 57.7                         | 56.7, 57.7, 57.8, 57.3, 57.9, 57.8, 58.1, 57.1, 57.7, 57.6, 57.9, 57.9                         | 53.7, 52.5, 52.6, 53.2, 53.6, 53.2, 52.7, 52.0, 52.2, 52.9, 52.6, 52.6                         | 52.4, 50.6, 51.0, 52.0, 50.9, 52.3, 52.2, 52.3, 51.7, 52.9, 52.9, 52.4                       |
| gemma-3-1b-it tg64  | 47.0, 47.4, 46.7, 46.8, 46.9, 47.0, 46.5, 47.2, 47.4, 45.3, 46.9, 47.0                         | 46.7, 47.7, 46.9, 46.0, 47.4, 45.5, 46.8, 46.8, 47.1, 46.6, 47.0, 46.8                         | 42.4, 42.3, 42.3, 42.4, 42.4, 42.4, 42.1, 42.9, 41.9, 42.9, 41.5, 41.9                         | 41.9, 42.2, 41.0, 42.2, 42.2, 42.9, 42.3, 42.2, 42.9, 42.0, 42.4, 41.0                       |

Paired, `split` against `before`, per round, computed from the values as the
table shows them:

- Qwen3-1.7B pp512, native: -1.0, -2.5, -2.2, -3.6, -4.0, -4.2, -4.0, -1.6,
  -4.2, -3.5, -6.0, -1.3% (median -3.5%, lower in 12 of 12); baseline: -3.7,
  -1.1, -2.6, -0.9, -2.2, -1.1, -0.6, -2.0, -4.9, -1.1, -1.3, -2.2% (median
  -1.65%, 12 of 12).
- gemma-3-1b pp512, native: -3.0, -2.0, -1.4, -1.0, -0.3, -1.8, +0.6, -2.9,
  -2.0, -1.4, -3.2, -3.0% (median -1.9%, 11 of 12); baseline: +0.2, -1.6, -1.5,
  -3.6, -1.0, -0.9, -1.3, -1.5, -1.6, -2.0, -1.7, -1.1% (median -1.5%, 11 of
  12).
- Decode, medians: Qwen3-1.7B native +0.1% (lower in 6 of 12), baseline -1.3% (9
  of 12); gemma-3-1b native -0.1% (6 of 12), baseline -0.5% (8 of 12).

The noise, from the same protocol with one native binary against a copy of
itself (five rounds): single-round differences of -1.8 to +3.2%, medians within
±1.1% (`docs/math.md` lists them). So the prefill costs are real (every or all
but one of the rounds, medians above the noise) and decode is level; the
baseline Qwen3-1.7B decode median of -1.3% is at the edge of the noise.

**Per op** (`INFR_PROF_OPS=1`, the last of two reps of pp512, native builds,
four alternating rounds, ms per forward over every call of the op): Qwen3-1.7B's
FFN-down `Linear m=511 6144x2048` went from 41.5-45.2 to 49.3-55.8 ms (Q6_K, 14
calls) and from 35.5-39.3 to 37.9-42.9 ms (Q4_K, 14 calls), the whole forward
from 432.7-445.0 to 445.0-464.0 ms; gemma-3-1b's `Linear m=511 6912x1152` from
30.3-31.5 to 39.9-42.0 ms (Q6_K, 13 calls) and from 25.4-26.2 to 34.2-35.0 ms
(Q4_K, 13 calls), the forward from 863.0-869.4 to 871.1-892.6 ms. Every other op
moved by under 2 ms per forward (midpoints of its ranges). The quantizer itself:
1.41 against 1.35 µs per 2048-wide ordinary row (`quant::tests::microbench`); a
first-token row with two parts, 7.4 µs, and its 64 Q4_K rows 2.73 against 1.92
µs.

Before this measurement the figure rested on rounds taken with another agent's
tests running (medians of -1.7% and -3.2% on Qwen3-1.7B, -1.2% and -3.8% on
gemma-3-1b, and "about 2%"); the quiet rounds put Qwen3-1.7B's native cost at
the high end of that and settle the baseline build's, which had been measured
only against an earlier form of the split.

### Speed with the exact sparse dot

Measured on a quiet machine on 2026-11-19 (no other build, test or model run;
the process list read before every round showed only an idle `cargo` of another
session, started the day before, and the idle WSL service),
`infr bench --dev cpu` (three reps a run, tok/s), Windows, release builds of
`1b16e66b` (`before`: no split), `origin/main` at `579fec0a` (`main`: the split
with the one-block kernel calls) and this change (`this`), native and baseline
(`-C target-cpu=x86-64`), and a second copy of the native `this` binary for the
noise. Twelve rounds for the dense models, eight for the MoE models, the
binaries' order reversed every other round. Paired medians are computed from the
values as the tables show them.

| Model, run          | native `before`                                                                                | native `main`                                                                                  | native `this`                                                                                  | native `this` (copy)                                                                           |
| ------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Qwen3-1.7B pp512    | 1135.1, 1142.4, 1134.6, 1145.8, 1158.7, 1153.6, 1146.7, 1120.8, 1145.7, 1129.8, 1155.4, 1143.9 | 1123.7, 1133.2, 1110.7, 1112.7, 1108.4, 1096.9, 1105.3, 1111.0, 1124.4, 1131.0, 1100.1, 1129.3 | 1146.9, 1141.7, 1137.2, 1135.3, 1124.0, 1134.6, 1131.2, 1130.6, 1140.5, 1140.1, 1138.0, 1133.1 | 1156.0, 1149.5, 1153.7, 1138.1, 1144.6, 1148.0, 1145.5, 1157.3, 1146.3, 1150.7, 1154.9, 1149.0 |
| gemma-3-1b-it pp512 | 595.5, 591.9, 589.3, 594.4, 583.3, 591.4, 587.9, 576.7, 591.7, 576.9, 586.7, 586.7             | 588.5, 584.1, 584.1, 573.0, 580.3, 579.8, 579.8, 581.9, 575.2, 581.2, 579.5, 572.7             | 598.5, 588.5, 581.3, 587.2, 587.3, 584.8, 587.1, 580.6, 591.1, 596.8, 589.1, 587.0             | 593.4, 585.8, 584.0, 567.5, 582.8, 576.2, 586.0, 581.5, 575.7, 575.4, 591.4, 586.2             |
| Qwen3-1.7B tg64     | 57.7, 58.3, 58.5, 58.5, 57.4, 57.9, 57.1, 57.4, 58.1, 57.6, 57.5, 58.5                         | 55.9, 57.5, 57.1, 57.5, 56.7, 57.3, 56.5, 57.6, 56.6, 54.4, 58.0, 57.1                         | 58.0, 56.9, 58.0, 57.0, 57.9, 56.5, 57.2, 57.8, 57.9, 57.0, 56.5, 57.6                         | 57.1, 57.4, 56.6, 57.9, 57.5, 57.2, 57.7, 56.2, 58.1, 57.4, 56.9, 57.8                         |
| gemma-3-1b-it tg64  | 47.2, 46.8, 47.3, 47.1, 46.5, 46.3, 47.6, 47.0, 45.2, 46.1, 46.1, 45.3                         | 46.3, 46.5, 46.7, 47.3, 46.2, 46.5, 46.2, 46.8, 46.3, 47.0, 44.2, 47.4                         | 46.5, 46.6, 46.6, 46.2, 46.1, 46.7, 46.5, 45.5, 46.0, 46.5, 47.3, 44.3                         | 46.5, 46.9, 46.7, 45.7, 46.3, 46.2, 46.7, 46.4, 47.4, 46.2, 46.2, 46.4                         |

| Model, run          | baseline `before`                                                                              | baseline `main`                                                                                | baseline `this`                                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Qwen3-1.7B pp512    | 1042.2, 1038.4, 1034.2, 1057.2, 1053.6, 1028.2, 1017.5, 1035.4, 1035.6, 1047.2, 1036.3, 1029.8 | 1024.9, 1026.5, 1009.7, 1010.5, 1000.5, 1021.8, 1007.1, 1022.7, 1010.5, 1009.3, 1016.8, 1021.8 | 1065.3, 1029.0, 1055.3, 1033.9, 1051.0, 1057.2, 1039.6, 1026.3, 1054.2, 1029.4, 1047.4, 1051.9 |
| gemma-3-1b-it pp512 | 581.1, 593.9, 582.1, 589.8, 588.3, 580.0, 576.0, 571.9, 579.8, 578.6, 582.5, 584.4             | 574.0, 572.1, 565.1, 575.3, 570.4, 573.8, 567.6, 569.6, 579.3, 575.0, 576.4, 577.3             | 589.9, 578.8, 582.0, 570.1, 582.1, 579.4, 575.9, 580.2, 570.7, 580.9, 582.7, 582.5             |
| Qwen3-1.7B tg64     | 52.6, 51.6, 51.7, 51.9, 51.5, 51.3, 50.7, 51.8, 51.6, 51.2, 50.2, 51.2                         | 52.9, 50.4, 52.3, 51.7, 52.2, 50.7, 51.4, 51.7, 52.6, 52.0, 50.2, 52.9                         | 51.2, 52.5, 52.3, 52.1, 51.2, 49.7, 52.7, 52.7, 52.2, 51.9, 52.7, 52.7                         |
| gemma-3-1b-it tg64  | 42.5, 43.1, 41.8, 40.3, 42.6, 41.4, 41.3, 41.9, 42.0, 41.7, 42.5, 41.7                         | 42.3, 41.4, 42.0, 42.4, 42.5, 40.2, 41.4, 40.6, 41.0, 41.8, 41.5, 41.5                         | 42.4, 41.8, 41.8, 42.0, 42.0, 40.5, 41.5, 41.8, 42.4, 40.4, 40.4, 41.2                         |

| Model, run              | native `before`                                        | native `main`                                          | native `this`                                          | baseline `before`                                      | baseline `main`                                        | baseline `this`                                        |
| ----------------------- | ------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------ |
| ERNIE-4.5-21B-A3B pp512 | 471.3, 444.6, 472.1, 470.2, 441.7, 457.6, 453.9, 462.0 | 458.9, 458.0, 460.2, 457.8, 454.3, 460.9, 459.1, 458.9 | 467.2, 467.1, 466.6, 468.0, 465.3, 467.4, 464.8, 464.1 | 438.0, 443.8, 446.6, 440.1, 433.8, 419.4, 441.8, 432.7 | 433.4, 431.4, 433.3, 432.7, 431.4, 433.1, 429.7, 433.2 | 442.4, 440.3, 439.4, 441.6, 437.9, 441.1, 437.8, 441.1 |
| ERNIE-4.5-21B-A3B tg64  | 25.9, 26.1, 25.9, 26.4, 26.5, 26.0, 26.2, 26.1         | 26.7, 27.1, 27.4, 27.1, 27.0, 26.9, 26.6, 26.7         | 26.6, 26.9, 26.9, 27.0, 27.0, 27.2, 26.7, 27.5         | 20.0, 20.4, 20.9, 20.1, 20.0, 20.8, 19.9, 20.2         | 25.9, 25.9, 25.7, 25.7, 25.7, 25.4, 26.5, 25.7         | 25.8, 26.2, 25.4, 26.1, 26.6, 26.6, 25.9, 25.8         |
| Qwen3-30B-A3B pp512     | 418.5, 417.3, 420.4, 417.3, 418.6, 418.1, 419.2, 418.5 | 418.4, 416.4, 416.7, 416.4, 415.1, 415.6, 398.4, 416.6 | 418.0, 416.4, 412.0, 417.7, 415.7, 416.5, 414.4, 417.6 | 389.9, 389.7, 391.6, 386.7, 380.1, 388.4, 382.3, 391.8 | 391.2, 385.2, 386.8, 384.9, 379.4, 390.4, 386.1, 386.0 | 389.7, 388.5, 392.2, 390.8, 387.0, 392.2, 391.0, 391.0 |
| Qwen3-30B-A3B tg64      | 27.1, 27.1, 27.4, 27.3, 27.1, 27.3, 27.1, 27.4         | 28.1, 28.4, 28.7, 28.2, 28.0, 28.4, 28.3, 28.1         | 28.3, 27.9, 28.0, 28.3, 28.4, 28.6, 28.4, 28.5         | 21.8, 21.4, 21.8, 21.7, 21.4, 21.6, 21.6, 21.6         | 26.7, 26.7, 27.3, 26.9, 26.5, 27.1, 27.2, 27.0         | 27.2, 27.2, 27.0, 26.9, 27.4, 26.9, 26.9, 27.1         |

Paired medians, with the rounds in which the first build was the lower:

| Run                     | build    | `this` vs `before` | `this` vs `main` | `main` vs `before` |
| ----------------------- | -------- | ------------------ | ---------------- | ------------------ |
| Qwen3-1.7B pp512        | native   | -0.69% (8 of 12)   | +1.90% (0 of 12) | -1.98% (11 of 12)  |
| Qwen3-1.7B pp512        | baseline | +1.43% (5 of 12)   | +3.12% (0 of 12) | -1.77% (12 of 12)  |
| gemma-3-1b pp512        | native   | -0.03% (6 of 12)   | +1.46% (2 of 12) | -1.27% (10 of 12)  |
| gemma-3-1b pp512        | baseline | -0.06% (8 of 12)   | +1.13% (2 of 12) | -1.22% (12 of 12)  |
| ERNIE-4.5-21B-A3B pp512 | native   | +1.30% (3 of 8)    | +1.61% (0 of 8)  | +0.03% (4 of 8)    |
| ERNIE-4.5-21B-A3B pp512 | baseline | +0.64% (3 of 8)    | +1.87% (0 of 8)  | -1.37% (6 of 8)    |
| Qwen3-30B-A3B pp512     | native   | -0.30% (7 of 8)    | +0.18% (2 of 8)  | -0.53% (8 of 8)    |
| Qwen3-30B-A3B pp512     | baseline | +0.57% (3 of 8)    | +1.28% (1 of 8)  | -0.32% (5 of 8)    |
| Qwen3-1.7B tg64         | native   | -0.95% (8 of 12)   | +1.06% (4 of 12) | -1.54% (10 of 12)  |
| Qwen3-1.7B tg64         | baseline | +1.26% (3 of 12)   | -0.10% (6 of 12) | +0.87% (4 of 12)   |
| gemma-3-1b tg64         | native   | -1.17% (8 of 12)   | -0.22% (7 of 12) | -0.53% (7 of 12)   |
| gemma-3-1b tg64         | baseline | -0.72% (8 of 12)   | -0.12% (6 of 12) | -0.48% (8 of 12)   |
| ERNIE-4.5-21B-A3B tg64  | native   | +2.88% (0 of 8)    | -0.18% (4 of 8)  | +2.87% (0 of 8)    |
| ERNIE-4.5-21B-A3B tg64  | baseline | +28.72% (0 of 8)   | +0.77% (3 of 8)  | +27.54% (0 of 8)   |
| Qwen3-30B-A3B tg64      | native   | +4.22% (0 of 8)    | +0.53% (2 of 8)  | +3.86% (0 of 8)    |
| Qwen3-30B-A3B tg64      | baseline | +24.65% (0 of 8)   | +0.19% (3 of 8)  | +24.88% (0 of 8)   |

The noise, from the native `this` binary against its copy in the same rounds:
+1.22% (Qwen3-1.7B pp512, the copy lower in 0 of 12), -0.61% (gemma-3-1b pp512,
9 of 12), +0.52% and +0.32% on the decodes; single rounds -3.6 to +4.7%. So a
median under about 1.2% is not resolved, and a fixed offset of that size between
two copies of one binary is possible (the copy's lead held in all twelve
Qwen3-1.7B prefill rounds).

So the split's prefill cost is gone to within the noise on every workload,
native and baseline: `this` is within ±1.43% of `before` on every prefill
(medians), where `main` was 1.2 to 2.0% below it on the dense models in 10 to 12
of 12 rounds, and `this` is above `main` in 44 of the 48 dense prefill rounds
(native and baseline) and in all 16 ERNIE ones. Decode does not move against
`main` (medians -0.22 to +1.06%); the MoE decode gains against `before` are the
one-row int8 expert path's (above). `before`'s ERNIE prefill was noisy in this
window (two rounds near 442 and 445), so its `main` against `before` reads level
here where the ten earlier rounds read -2.9%; against `main`, `this` is faster
in 8 of 8 on both builds.

**Per op** (`INFR_PROF_OPS=1`, the last of two reps of pp512, native builds,
four alternating rounds, ms per forward over every call of the op), `before` /
`main` / `this`: Qwen3-1.7B's FFN-down `Linear m=511 6144x2048` Q6_K 44.2-46.9 /
50.5-53.8 / 44.0-44.4 ms and Q4_K 36.7-40.0 / 39.9-42.2 / 35.9-36.6 ms, the
whole forward 441.2-449.5 / 449.1-458.7 / 435.3-437.5 ms; gemma-3-1b's
`Linear m=511 6912x1152` Q6_K 30.0-30.9 / 39.0-40.0 / 33.1-34.0 ms and Q4_K
25.3-26.0 / 33.8-34.1 / 27.3-28.5 ms, the forward 842.0-860.6 / 854.0-863.5 /
847.7-854.1 ms. What is left is gemma-3-1b's FFN-down, about 5 ms per forward
(0.6% of it; it has about 7,100 split blocks a forward against Qwen3-1.7B's
1,930), below what the bench rounds resolve.

**Per weight row** (`kernels::hi::tests::microbench`, single-threaded, a
2048-wide row, ns per weight row for one code, then for nine codes in five split
blocks), the new dot against the one-block kernel calls it replaced: Q4_K 5.8 /
115.4 and 22.7 / 581.9, Q6_K 6.0 / 116.5 and 23.5 / 580.6, Q5_K 6.2 / 122.0,
Q2_K 5.6 / 119.4, Q3_K 7.9 / 121.8, IQ4_XS 5.9 / 119.6, Q8_0 8.4 / 121.2, TQ2_0
4.9 / 217.5, TQ1_0 5.2 / 233.1; the grid formats 97.1 to 181.3 against 251.8 to
292.1 (they still expand the block, backlog B223). A prefill's table
(`batch_microbench`, Q6_K 6144 wide, two parts' elements): 8.3 ns per weight row
to fill, 0.5 ns per weight row to evaluate one row's parts from it.

**The quantizer** (`quant::tests::microbench`, three runs each): an ordinary
2048-wide row took 1343 to 1369 ns with the split's count pass against 1296 to
1301 ns for the unsplit quantizer; with the bound in the max pass
(`quant::cannot_split`), 1327 to 1345 ns against 1318 to 1330 (11008 wide: 6779
to 6874 against 6694 to 6758, where it was 6841 to 6874 against 6589 to 6605). A
row whose blocks split pays `split_off`'s sort as before (6.9 µs for a 2048-wide
row with two split blocks).

**Quality** does not move: every value is the bits it was (the op hashes, the
CPU goldens and the identity suites pass unchanged). `cpu_act_quant`'s
`per_position_against_the_reference` on SmolLM3-3B, run on this change and on
`origin/main` the same day, printed the same 44 score lines (production and
every simulated scheme, four prompts): position 0 at 1.4e-4 on the chat prompts
and 4e-5 on the raw one, position 1 at 1e-5 and 1.42e-3, the raw prompt's median
1.3e-3, as "After the fix" lists. The chat prompts' worst position (240) scored
0.691 on both builds, where that table lists 0.33 from the split's own
measurement; what moved it in between is not traced here.

### What it does not fix

gemma-3-1b's large mid-sequence KLs (0.40 to 0.67 at its prompts' last
positions) are not outliers: those rows split the fewest blocks of any, their
own per-op errors are ordinary, and the distance builds up through the layers.
In simulation, int16 activations on the 256-block weights alone take the worst
one from 0.52 to 0.09, on the 32-block weights (gemma's Q5_0 and Q8_0
projections at width 1152) alone to 0.32, and on both to 1e-5: it is the int8
activations' ordinary noise on a model that amplifies it, on both paths (backlog
B209).
