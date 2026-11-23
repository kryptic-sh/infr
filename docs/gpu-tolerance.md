# GPU logits within tolerance of the CPU reference

The CPU backend gives the same bits on every system, tier, thread count and
prefill chunk height ([plan 11](plan/11-cpu-bit-identical.md)), so it is the one
reference every GPU is held to. GPU kernels do not reproduce its bits, and
become tunable per device ([plan 10](plan/10-tunable-kernels.md)), so the check
is a tolerance. This page says how the comparison is made, what it can and
cannot see, and where the tolerances came from. The values themselves live in
one place, `infr_llama::tolerance::table::TABLE`
(`crates/infr-llama/src/tolerance/table.rs`); this page does not repeat them.

## The comparison

**Teacher forcing.** The CPU reference backend (`kernels.cpu.reference`:
quantized weights decoded to f32 and dotted in f32, no int8 activations) runs a
prompt greedily for `N_DECODE` tokens. The backend under test is then fed the
very same token sequence, so the two never fork at a near tie, and both
backends' full-vocabulary logits are captured at every position
(`SeamModel::forced_logits`, `crates/infr-llama/src/seam/mod.rs` `forced_run`):

- **Prefill rows**: one ordinary chunked batched prefill of the prompt, at the
  engine's `device.ubatch` (the tests run the default and `SMALL_UBATCH`, so the
  long prompt spans several chunks and the small-`m` GEMM tiers run). The LM
  head is added over every chunk row (`PrefillHead::All`) or each chunk's last
  row (`PrefillHead::LastOfChunk`) through `generate_dense_backend_bound`'s
  `prefill_logits` hook; it is `None` (no head, no buffer, no download) on every
  production call. That head is a vocabulary-wide GEMM production never runs, so
  prefill-row scores include its error; that it changes nothing else is a test
  (`gpu_seam_tolerance_prefill_head_leaves_decode_unchanged`: the decode rows,
  which read the KV the prefill wrote, are bit-identical with and without it).
- **Decode rows**: one per-token decode step per position, each a warm call on
  the same session that feeds exactly the one new token (checked: every cached
  token reused, one fed), so it reads KV written by the prefill and by every
  earlier decode step.

**What the decode side does not exercise.** Where the backend replays, each
forced step builds, records and executes a fresh replay plan once: the
record-once-replay-many mechanism (one recording replayed with advancing
parameters) and the chained decode (`decode_chain`, several steps per submit,
the production greedy path) are never run, because forcing needs every step's
logits and the chain does not read them back. The exact GPU goldens and the
exact-token GPU-against-CPU tests are what cover those paths today, and G4 must
not drop them without a replacement. The static per-token path
(`kernels.vulkan.no_replay`, which production takes for some model families and
quantized-KV configurations) is gated: every model runs a `gpu-float-static`
configuration.

The driver is held to the reference's own row identity in
`forced_run_reproduces_the_cpu_reference_on_every_path` (normal test job, no GPU
or model): another chunk height, and decode rows against prefill rows at the
same positions, bit for bit. Writing the prefill chunks' logits in the wrong
order fails it.

**Metrics per position** (`crates/infr-llama/src/tolerance/metrics.rs`), in f64
from the two f32 rows, on `infr_core::math` (the workspace takes no math from
the platform library):

| Metric         | Definition                                                                                                                                                         | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KL             | KL(reference ‖ test) over the full vocabulary                                                                                                                      | The primary metric: the expected extra log-loss, shift-invariant. Computed as `ln(1 + E_p[expm1(e - c)]) - E_p[e - c]` with `e = t - r`, so two rows a rounding apart score their real distance; a reference token far down the row (its `e - c` too large to exponentiate, or its f32 probability underflowed) enters the normalizer as `e^(t - lse_r - c)`, so promoting one is charged what it costs; only a term itself beyond f32's range scores infinite |
| top-k cosine   | cosine of the two rows' probability vectors over the union of both rows' top sets (0.99 of the mass or 64 tokens each), plus one component for the mass outside it | Secondary, never the sole detector of a class of bug: sees mass moved within or out of the top tokens even when the reference's top set is one token, but only means something when the union holds most of the mass (on a high-entropy row the rest component dominates: two rows flat over different 1000-token sets of 32000 score 0.99993 at KL 30). A whole-vocabulary logit cosine cannot separate a match from a mismatch (B60)                         |
| logit dev      | the largest difference after centring each row on its mean, over the reference row's range (max − min)                                                             | Shift-invariant; the mean does not move with the top logits, so one wrong top logit counts once                                                                                                                                                                                                                                                                                                                                                                |
| top-1          | the argmax agrees, counted only where the reference's top-2 margin exceeds the row's near-tie threshold                                                            | A greedy flip that is not a tie. Near ties left out are reported, and each of the prefill and decode phases must score at least `MIN_TOP1_SCORED` positions over a case                                                                                                                                                                                                                                                                                        |
| ref top-1 rank | how many tokens the test row ranks above the reference's choice                                                                                                    | Separates "second by a hair" from "lost"                                                                                                                                                                                                                                                                                                                                                                                                                       |

Each is aggregated per run (one prompt, one configuration, one phase) as the
worst position, with where it was, and the mean. A NaN or infinite logit is an
immediate failure naming the position.

**The first rows are their own phase.** The first `FIRST_ROWS` positions carry a
model's largest activations, and every quantized path misses them by the most
(Qwen3-0.6B's, gemma-3's and LFM3's worst int8 KL is at position 0 or 1). Scored
together with the rest, that row would set every position's bound. Each table
row therefore has a `first` bound for them and a `steady` bound for every later
position, prefill and decode alike.

Probes (unit tests): the old top-k cosine was exactly 1 when the reference's top
token held 0.99 of the mass, whatever the test row did (SmolLM3's production-CPU
position-0 row: KL 4.88, cosine 1.000000); the new one gives 0.016 on that row
and below 0.5 on the constructed probe. A tail that takes half the mass from a
one-token top set now gives a cosine under 0.9 (was 1). A test row promoting a
reference token from far down the row (`[0, -d, -5, -6]` against
`[0, 5, -5, -6]`) scored KL 0 for `d` from 104 on and infinity for `d` from 80
to 103, against a true 4.9976 (mpmath); it now scores that value to six digits
across the range, and a token at -100 moved to -15 scores its true 3.03e-7
instead of infinity.

## The three-way comparison: which CPU mode is the reference

Measured 2026-11-18 on the RX 7900 XTX (`Vulkan0`, native lanes, tuning rows
on), the three `PROMPTS` (12 to 185 tokens) and `N_DECODE` forced decode steps,
by `measure_gpu_tolerance`. Each cell is the worst KL at any position (that
position) / the worst KL from position 2 on / the mean KL from position 2 on.
"GPU int8" is the shipped default; "GPU float" turns every int8-activation
kernel off (`vulkan_float` in `tests/support/forced.rs`).

| Model                 | CPU prod vs ref                | GPU int8 vs ref                | GPU float vs ref               | GPU int8 vs CPU prod     |
| --------------------- | ------------------------------ | ------------------------------ | ------------------------------ | ------------------------ |
| Qwen3-0.6B Q4_K_M     | 7.7e-2 (0) / 6.1e-2 / 8.2e-3   | 3.5e-2 (0) / 1.5e-2 / 1.7e-3   | 3.6e-5 (155) / 3.6e-5 / 2.6e-6 | 6.9e-2 / 6.9e-2 / 9.3e-3 |
| Qwen3-1.7B Q4_K_M     | 1.4e-1 (203) / 1.4e-1 / 7.3e-3 | 3.3e-2 (101) / 3.3e-2 / 1.5e-3 | not run                        | 1.7e-1 / 1.7e-1 / 9.0e-3 |
| gemma-3-1b Q4_K_M     | 1.88 (0) / 1.35 / 3.5e-2       | 0.33 (0) / 2.6e-2 / 2.3e-3     | 3.2e-3 (40) / 3.2e-3 / 4.4e-5  | 1.97 / 1.97 / 3.8e-2     |
| Qwen3.5-0.8B Q4_K_M   | 6.8e-2 (0) / 3.4e-2 / 2.8e-3   | 7.1e-3 (7) / 7.1e-3 / 6.5e-4   | 9.9e-6 (0) / 5.3e-6 / 7.7e-7   | 6.3e-2 / 5.6e-2 / 3.3e-3 |
| LFM3-1.7B Q4_K_M      | 0.10 (0) / 4.5e-2 / 4.1e-3     | 2.9e-2 (0) / 2.4e-2 / 1.3e-3   | 1.5e-4 (0) / 2.8e-5 / 2.1e-6   | 0.12 / 0.12 / 4.8e-3     |
| ERNIE-4.5-0.3B Q4_K_M | 0.84 (0) / 0.22 / 1.2e-2       | 2.4e-2 (50) / 2.4e-2 / 2.2e-3  | 5.9e-5 (0) / 1.8e-5 / 3.3e-6   | 0.70 / 0.17 / 1.4e-2     |
| Llama-3.2-1B Q4_K_M   | 7.2e-2 (8) / 7.2e-2 / 3.1e-3   | 1.3e-2 (14) / 1.3e-2 / 5.4e-4  | not run                        | 0.13 / 0.13 / 3.6e-3     |
| Qwen2.5-0.5B Q4_K_M   | 0.10 (0) / 1.6e-2 / 1.9e-3     | 1.8e-3 (0) / 1.3e-3 / 1.9e-4   | not run                        | 0.12 / 1.8e-2 / 2.1e-3   |
| SmolLM3-3B Q4_K_M     | 4.88 (0) / 2.0e-2 / 2.7e-3     | 8.2e-3 (0) / 7.0e-3 / 6.7e-4   | not run                        | 7.93 / 2.9e-2 / 3.2e-3   |

(The F16 and synthetic f32/f16 models have no quantized `Linear`, so the two CPU
modes are the same bits there and the GPU has no int8 path to take.)

What it says:

- **The CPU reference is the right reference.** From position 2 on, the shipped
  int8 GPU is 2 to 50 times closer to the reference (worst KL) than the
  production CPU is, and its distance to the production CPU is about the
  production CPU's own distance to the reference: the two int8 paths quantize
  different values at different points, and their errors add. Scoring a GPU
  against the production CPU would charge it with the CPU's error.
- **The production CPU's first-token row is wrong on SmolLM3-3B** (KL 1.9 to 4.9
  at position 0 on every prompt, at most 2e-2 after it), and the production CPU
  is far off away from position 0 on gemma-3 (1.35 at position 13) and ERNIE
  (0.22). The int8 GPU is within 8e-3 on SmolLM3's same rows. Recorded in B207
  with the per-position breakdown; it is a CPU slice of its own.
- **The GPU's int8 distance is its own quantization, not a bug.** Turning the
  int8 kernels off brings the quantized models to 5e-6 to 4e-5 worst KL from
  position 2 on (gemma-3 3e-3): the same kernels' float arithmetic.
- **Sensitivity is per model**: gemma-3-1b is about a hundred times more
  sensitive than the Qwen models; the table has rows per model.

## Path classes

`tolerance::table::PathClass`, each with its own rows:

- **Float**: f32/f16 weights, or quantized weights with every int8-activation
  kernel off (`vulkan_float`: `i8_dot` gates every packed-int8 dot kernel,
  including the non-coopmat `mmq` tier that wave64 and devices without
  cooperative matrices take; `i8cm_kq`, `dp4a_kq`, `mmq`, `mrow` and
  `mmv_mw = Some(false)` name the rest), f16 KV cache. That the class is
  int8-free is a test (`gpu_seam_tolerance_float_runs_build_no_int8_kernel`: the
  Float run builds no int8-activation kernel, the shipped run on the same model
  does).
- **FloatNoCoopmat**: the Float configuration with the whole cooperative-matrix
  tier off as well (`vulkan_float_nocoopmat`: `kernels.vulkan.coopmat`, what
  `INFR_NO_COOPMAT` sets), so the batched prefill's GEMMs take f32 operands on
  every device. Only gemma-3 has a row here, because only gemma-3's Float row is
  dominated by the f16 operands of the cooperative-matrix prefill GEMMs
  (`native_gemm_warp_*`); this class is its fine check of what still runs. With
  the tier off the kernels that serve those GEMMs are not built either
  (`splitk_reduce` and `store_f16`, on the RX 7900 XTX under RADV), so on a
  device with the tier they, like the GEMMs, are scored only under the Float
  row. That the class runs none of them is a test
  (`gpu_seam_tolerance_nocoopmat_runs_build_no_coopmat_gemm`: such a run builds
  no `gemm_warp` kernel, and the Float run on the same model does wherever the
  device has the tier). On a device without the tier, and with wave64 forced,
  the Float configurations build the same kernels as these, and are held to the
  Float row all the same.
- **Int8**: quantized weights through the int8-activation kernels, the shipped
  default. Dominated by the activation quantization.
- **KvQ8**: the Float configuration with a `q8_0` K and V cache, scored against
  the CPU reference running the same cache. The GPU quantizes the f16 RoPE
  output and the CPU its f32 value, so the two caches round to different codes:
  two orders above Float on Qwen3-0.6B.

Lane width and the tuning rows: forced wave64 drops the cooperative-matrix
tiers, so its prefill runs other kernels and its own distances; the tuning rows
change the RX 7900 XTX's native prefill and change nothing measured under wave64
(the wave64 rows-off combination was therefore not re-measured after the
first-row split).

## Mutations

The gate is shown to fail by planting bugs and scoring them, in committed tests
built only with the `mutations` feature of `infr-llama` (a default build has no
trace of them: the hooks are `#[cfg(feature = "mutations")]`, and the default
release rlib carries no `infr_llama::mutation` symbol).

**How a bug is planted.** Each test loads the model twice: the clean copy runs
the CPU reference, the other carries the bug and runs on the backend under test,
in the same process, with nothing global (`crates/infr-llama/tests/planted/`).

- **Graph bugs** live in the loaded model's `Config`
  (`SeamModel::plant(infr_llama::mutation::Planted)`), so every graph that model
  builds carries them and no other model does: (a) the query rope reads a
  positions input of its own holding position + 1 (a per-step input, so the
  graph cannot replay); (b) `Op::Attention`'s `pos` and `kv_len` are one short
  in every graph that does not start at position 0, so each row loses its own
  key (decode and every prefill chunk after the first; the first chunk starts at
  0 and is untouched); (c) the model's `rms_eps` is multiplied by 10 (Qwen3:
  1e-5 for 1e-6); (f) one layer's attention residual add is not emitted. (a) and
  (b) depend on the position a graph was built at, so their graphs opt out of
  the replayed decode (`Graph::no_decode_replay`) and every decode step runs the
  static path at its own position; the others reach the replayed decode too.
- **Weight bugs** are written into a copy of the model file (the tensor's bytes
  edited at its own offset, everything else as it was), so the backend runs its
  production code on wrong weights: (d) every block of one `ffn_down` times 1.01
  (Q4_K: `d` and `dmin` scaled, so the whole block scales, within an f16
  rounding of 1.01; F16: each value) and (d1) one block of it. On gemma-3 (d)
  scales the post-FFN norm's gain (`post_ffw_norm`) instead: its sandwich norms
  renormalize each sublayer's output before the residual add, so a scale of the
  whole `ffn_down` never reaches the residual stream (planted there, the run on
  the f32-operand path measured 9.4e-4 against the clean run's 7.1e-4), and the
  norm's gain is the weight whose scale does. (d1) stays in `ffn_down`: one
  block's scale is not uniform, so the norm does not remove it; (e) every
  layer's `ffn_gate` and `ffn_up` swapped; (g) a RoPE pairing error: within each
  head of `attn_q` and `attn_k` (and the per-head q/k norm weights) the rows
  reordered so the rope rotates `(2p, 2p+1)` where `(p, p+d/2)` is meant, or the
  reverse on a NORM-rope model, which is exactly a rope kernel pairing the wrong
  elements; (h) a wrong quant nibble map: the two nibbles of every packed byte
  of one Q4_K `ffn_down` swapped; (i) a router that ignores its bias: every
  `exp_probs_b.bias` zeroed (B174's case).

**What each test asserts.** A clean control runs first, twice: it must pass the
gate, and its two runs must agree bit for bit. Every planted run must then
differ from the clean run in the decode rows of every prompt and configuration
(the bug reached decode). Where the gate must catch the bug, the run must fail
it with a failure that names a metric or a non-finite logit at a prompt. Where a
miss is on record ((c), (d) and (d1) in Int8 and in gemma-3's Float smoke row,
(d1) in every other row), the ratio is printed and nothing more is asserted, so
a tighter table later cannot break the test. Each bug and run prints a `PLANTED`
line with its verdict and its worst ratio over the steady phases (from position
2 on).

CI runs the CPU demonstration and clippy with the feature on its Linux legs
(`.github/workflows/ci.yml`, the `test` and `clippy` jobs); the GPU ones need a
GPU and run by hand.

```text
# no GPU or model: every bug on the synthetic llama and the synthetic MoE, CPU reference
cargo nextest run -p infr-llama --release --features mutations --test gpu_tolerance \
  --no-capture -E 'test(planted_bugs_fail_the_gate_on_the_cpu)'
# the GPU demonstrations, one GPU process at a time
INFR_DEV=Vulkan0 cargo nextest run -p infr-llama --release --features mutations \
  --test gpu_tolerance --run-ignored all --test-threads 1 --no-capture -E 'test(gpu_planted)'
```

`gpu_planted_qwen3` runs Qwen3-0.6B Q4_K_M in the Int8 (`gpu`) and Float
(`gpu-float`) classes, every bug but (i); `gpu_planted_gemma3` runs gemma-3-1b
Q4_K_M in each class its rows are scored in (`gpu` Int8, `gpu-float` Float,
`gpu-float-nocoopmat` FloatNoCoopmat), the same bugs, and
`planted_gemma3_bugs_fail_the_fine_row_on_the_cpu` (ignored: it needs the model,
not a GPU) runs them on the CPU reference against the clean model's reference,
which gives each bug's own size on gemma with no GPU's noise in it;
`gpu_planted_ernie_f16` runs ERNIE-4.5-0.3B F16 (a float-weight model: no
nibbles, so no (h)) in Float; `gpu_planted_router_bias` runs (i) on a synthetic
ERNIE-4.5 MoE (eight f32 experts, two used, a softmax router with a selection
bias), since ERNIE-4.5-21B was not cached on the machine when these ran (it is
now; (i) has not been planted in it yet).

**Measured 2026-11-23**, Linux 7.3.5-arch1-1, Mesa RADV 26.3.1, headless (no
compositor running), on the final test code: `Vulkan0` RX 7900 XTX (RADV
NAVI31), `Vulkan1` the Ryzen 9 9950X3D's integrated GPU (RADV
RAPHAEL_MENDOCINO), native lanes, tuning rows on. Each cell is the worst KL from
position 2 on over the row's steady KL bound, `Vulkan0` / `Vulkan1`; misses also
give the mean-KL ratio.

| Bug                                                | Qwen3-0.6B Int8                                  | Qwen3-0.6B Float                                | ERNIE-4.5-0.3B F16 Float                                       |
| -------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------- | -------------------------------------------------------------- |
| clean control (passes)                             | 0.10 / 0.29                                      | 0.31 / 0.13                                     | 0.19 / 0.17                                                    |
| (a) query rope at position + 1                     | caught: 421 / 424                                | caught: 2.2e5 / 2.2e5                           | caught: 1.4e5 / 1.4e5                                          |
| (b) mask drops the last visible key                | caught in decode: 391 / 389                      | caught in decode: 2.1e5 / 2.1e5                 | caught in decode: 3.6e4 / 3.6e4                                |
| (c) RMSNorm epsilon x10                            | **missed**: 0.34 (mean 0.36) / 0.17 (mean 0.39)  | caught: 3.8 (mean 5.2) / 3.0 (mean 4.3)         | caught: 2.4e3 / 2.4e3                                          |
| (d) every block of `blk.10.ffn_down` x1.01         | **missed**: 0.087 (mean 0.24) / 0.34 (mean 0.42) | caught: 10.5 (mean 5.3) / 9.2 (mean 4.7)        | caught on mean KL (`blk.6`): 0.83 (mean 3.0) / 0.96 (mean 3.0) |
| (d1) one block of it x1.01                         | **missed**: 0.18 (mean 0.23) / 0.32 (mean 0.40)  | **missed**: 0.35 (mean 0.52) / 0.12 (mean 0.19) | **missed**: 0.094 (mean 0.40) / 0.18 (mean 0.28)               |
| (e) gate and up swapped                            | caught: NaN logit at position 0 / 620            | caught: NaN logit at position 0 / 3.3e5         | caught: 2.1e5 / 2.1e5                                          |
| (f) one layer's attention residual dropped (14; 9) | caught: 20.5 / 21.7                              | caught: 1.1e4 / 1.1e4                           | caught: 1.2e4 / 1.2e4                                          |
| (g) rope pairs the wrong elements                  | caught: 200 / 203                                | caught: 1.1e5 / 1.1e5                           | caught: 8.1e4 / 8.1e4                                          |
| (h) Q4_K nibbles swapped in `blk.10.ffn_down`      | caught: 91 / 92                                  | caught: 4.8e4 / 4.8e4                           | not applicable (F16 weights)                                   |

**gemma-3-1b**, measured the same day on the same machine and drivers, headless,
against its rows as re-set that day (`gpu_planted_gemma3`; "The table" below has
the rows). Cells as above, `Vulkan0` / `Vulkan1`. Its three classes are the
shipped kernels (Int8, a smoke row), the int8-free ones (Float, a smoke row of
the cooperative-matrix f16 prefill path; the integrated GPU has no such tier and
runs the f32-operand kernels here too) and the int8-free ones with the tier off
(FloatNoCoopmat, the fine row). The last column is the bug's own size: the
planted copy on the CPU reference against the clean copy's CPU reference
(`planted_gemma3_bugs_fail_the_fine_row_on_the_cpu`), over the fine row's bound.

| Bug                                             | Int8 (smoke)                                      | Float (smoke)                                                             | FloatNoCoopmat (fine)                           | CPU reference, fine row  |
| ----------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------ |
| clean control (passes)                          | 0.044 / 0.60                                      | 0.49 / 0.034                                                              | 0.40 / 0.40                                     | 0                        |
| (a) query rope at position + 1                  | caught: 35 / 35                                   | caught: 870 / 869                                                         | caught: 1.0e4 / 1.0e4                           | caught: 1.0e4            |
| (b) mask drops the last visible key             | caught in decode: 3.6 / 4.6                       | caught in decode: 90 / 105                                                | caught in decode: 1.1e3 / 1.2e3                 | caught: 1.1e3            |
| (c) RMSNorm epsilon x10                         | **missed**: 0.039 (mean 0.13) / caught: 1.4       | caught on mean KL and the logit deviation: 0.90 (mean 1.15) / caught: 1.2 | caught: 14 (mean 21) / 14 (mean 21)             | caught: 13 (mean 21)     |
| (d) every value of `blk.10.post_ffw_norm` x1.01 | **missed**: 0.030 (mean 0.11) / 0.47 (mean 0.53)  | **missed**: 0.53 (mean 0.32) / 0.11 (mean 0.047)                          | caught: 1.26 (mean 0.66) / 1.26 (mean 0.66)     | caught: 1.45 (mean 0.60) |
| (d1) one block of `blk.10.ffn_down` x1.01       | **missed**: 0.024 (mean 0.097) / 0.56 (mean 0.58) | **missed**: 0.50 (mean 0.30) / 0.033 (mean 0.020)                         | **missed**: 0.39 (mean 0.28) / 0.39 (mean 0.28) | **missed**: 0.001        |
| (e) gate and up swapped                         | caught: 83 / 82                                   | caught: 2.1e3 / 2.1e3                                                     | caught: 2.4e4 / 2.4e4                           | caught: 2.4e4            |
| (f) one layer's attention residual dropped (13) | caught: 3.0 / 2.8                                 | caught: 75 / 75                                                           | caught: 878 / 878                               | caught: 883              |
| (g) rope pairs the wrong elements               | caught: 21 / 21                                   | caught: 518 / 518                                                         | caught: 6.0e3 / 6.0e3                           | caught: 6.0e3            |
| (h) Q4_K nibbles swapped in `blk.10.ffn_down`   | caught: 13 / 13                                   | caught: 310 / 309                                                         | caught: 3.6e3 / 3.6e3                           | caught: 3.6e3            |

What gemma adds:

- **The fine row catches what Qwen3's Float row catches**: every bug but (d1),
  on both GPUs, with the same values on both to the digits shown except (b).
- **(d) is the fine row's thinnest catch.** Its steady worst KL is 2.3e-3 on
  both GPUs (2.6e-3 on the CPU reference), 1.26 times the KL bound, where the
  clean run is at 0.40 of it. It also fails the first rows' mean KL, 3.0e-4
  against a bound of 1.2e-4 where the clean run measures 1.6e-6, and the steady
  cosine, 0.99686 against 0.9979.
- **The Float row is a smoke row.** On the RX 7900 XTX it misses (d) and catches
  (c) only on mean KL (1.15 times its bound) and the logit deviation (6.7e-2
  against 4.2e-2), with the worst KL at 0.90; it catches nothing the fine row
  does not. What it alone covers is a gross bug inside a cooperative-matrix
  GEMM, which no planted bug is.
- **(c) moves gemma's int8 distance on the integrated GPU** (worst KL 0.70
  against the clean run's 0.31, over the Int8 bound) and not on the RX 7900 XTX,
  whose Int8 configuration prefills on the f16 GEMMs under RADV. It stays a
  recorded miss: nothing is asserted either way.
- **(e) gives finite logits on both GPUs here**, where on Qwen3 the RX 7900 XTX
  gives NaN.

(i), the router that ignores its bias, on the synthetic MoE: the CPU
demonstration catches it at 5274x (clean control 0); on both GPUs the planted
run measures 5239x, but the clean control measures 4.29x (mean 1.78x) itself,
outside the `synthetic-f32` Float row the MoE borrows (the table has no MoE
row), at the first decode row of the long prompt. So on the GPU this is a
measurement, not a gate verdict: `gpu_planted_router_bias` asserts only that the
bug reaches decode. The synthetic MoE also takes no batched prefill, so only its
decode rows are scored. The clean control's distance is one row: at most 1.03e-4
KL on prompts 0 and 1, and on prompt 2 2.87e-3 at position 179, the first decode
row after the 180-token prompt, the rest of that prompt averaging about 1.6e-5
(from its mean). Both GPUs give the same values to seven digits.

**That row is a router near tie, not a defect** (2026-11-23, `Vulkan0`). The
outlier does not follow the end of the prompt: with the same sequence and the
prompt ended at 100, 150, 170, 178 or 179 tokens, position 179 is a decode row
well past the boundary and still measures 2.87e-3, on the replayed and the
static decode alike, while the row at each new boundary is ordinary (at most
1.3e-5). Position 179 was only the first row the test scores, since this model
takes no batched prefill. Scoring the earlier rows shows where it comes from:
position 107 measures 0.371, and positions 110, 137, 138, 143 and 173 are raised
with it. At position 107 the CPU reference's layer-0 router separates its second
and third expert by 2.14e-4 of selection probability, the smallest margin of the
422 router calls of the sequence (median 6.3e-2). Flipping that one pick in the
CPU reference reproduces the GPU's profile (107: 0.371, 110: 4.65e-3, 138:
2.31e-2, 143: 5.62e-2, 179: 2.79e-3), and against the flipped reference no GPU
row is above 2.4e-4; that the GPU picked the other expert there is inferred from
that match, its expert ids were not read back. Position 107's layer-1 keys and
values then differ, and the later rows that attend to it carry the difference.
So a clean MoE run can legitimately pick another expert at a router near tie,
and every later row may inherit it: an MoE row in the table needs a stated rule
for that (B207), which is why none is set here.

The CPU demonstration (synthetic llama, CPU reference, against the clean model's
CPU reference, where the clean distance is exactly 0) catches every dense bug
but (d1): (a) 2.7e4, (b) 6.3e3, (c) on the logit deviation (KL 1.19x), (d) on
mean KL (1.43x), (e) 2.4e4, (f) 1.7e4, (g) 3.2e4; (d1) 0.003.

What it says, adding to the first measurement (Qwen3 alone, RX 7900 XTX on
Windows, scratch build) and agreeing with it everywhere it overlaps:

- **The Int8 rows catch gross bugs only.** (c), (d) and (d1) leave the int8
  run's distance where the clean run's is, on both GPUs; the Float runs catch
  (c) and (d). The new bugs, (g) and (h), are gross and caught in both classes.
  A bug living only inside an int8 kernel is the Float run's blind spot, covered
  by the op-level bounds (B207).
- **A sub-percent error in one weight block (d1) is invisible** to every row and
  on every path, the CPU reference's own included (0.003x on the synthetic
  model). Only a weight-level check (the dequant parity tests) covers it.
- **(d) on ERNIE F16 is caught on mean KL only** (worst KL 0.83x and 0.96x): the
  margin on that model is thin.
- **(e) gives NaN logits on the RX 7900 XTX and finite ones on the integrated
  GPU**, both classes (the first measurement also saw NaN): either way the gate
  fails, but the two devices handle the same overflow differently; not looked
  into.

**Committed red-path tests.** `tolerance_gate_goes_red_on_a_scaled_weight_cpu`
(normal test job, every system) runs the gate's evaluation on a synthetic llama
with `blk.1.ffn_down` scaled by 1.02, driven on the CPU reference against the
unmodified model's CPU reference, and asserts it fails naming KL (it trips KL,
mean KL, the deviation and the cosine, KL at 2.9x and mean KL at 4.7x its
bound), with the unmodified model against itself as the passing control.
`gpu_seam_tolerance_goes_red_on_a_scaled_weight` does the same on the Vulkan
seam. With (c) and with (f) planted, `gpu_seam_tolerance_qwen3` itself also
failed on the production decode path (run by hand).

## The table

`TABLE` has one row per (model, class) the tests run, each with a `steady` and a
`first` bound; its doc comment carries the provenance and the margins: KL max
and mean times 2, the cosine's distance from 1 times 2, the logit deviation
times 1.5, the rank at twice the worst plus one, the near-tie margin at 1.5
times the widest legitimate top-1 miss (at least 0.25 nats in Float, 1 nat in
Int8 and KvQ8), each from the worst legitimate value over every device and
configuration measured. The worst legitimate KL of a row (from position 2 on)
moves a long way between the paths measured — the RX 7900 XTX native, its wave64
path, its generic tuning (rows off) and the integrated GPU: within about 4x for
most rows, but 9x to 22x for gemma-3's Float and Int8 rows and LFM3's Float and
Int8 rows, and 83x for Qwen3.5's Float row (whose integrated-GPU figure scores
one prefill row per chunk). A row's bound is its worst path's, so it is loose
for the paths far below that. The margin of 2 on top is headroom for a device or
driver not measured here, kept small because the planted bugs sit only 3 to 4.5
times above the Float tolerance (6.4x on KL and 9.2x on mean KL above the
legitimate worst).

**Both sides of each gap, on Qwen3-0.6B** (from the first measurement, on
Windows; the committed demonstrations above agree with it), from position 2 on,
worst legitimate value / tolerance / smallest planted bug that crossed it:

| Class | KL max                       | mean KL                      | Verdict                                                                        |
| ----- | ---------------------------- | ---------------------------- | ------------------------------------------------------------------------------ |
| Float | 4.2e-5 / 8.5e-5 / 2.7e-4 (c) | 3.6e-6 / 7.3e-6 / 3.3e-5 (c) | A gap of 3x (KL) to 4.5x (mean) above the tolerance; (c) and (d) caught        |
| Int8  | 2.2e-2 / 4.5e-2 / 0.89 (f)   | 2.0e-3 / 4.1e-3 / 0.13 (f)   | Gross bugs only: (c) and (d) do not move the int8 distance off the clean run's |
| KvQ8  | 1.9e-2 / 3.9e-2 / not run    | 1.2e-3 / 2.5e-3 / not run    | No bug was planted with the q8 cache; by its distances it sees what Int8 sees  |

**gemma-3-1b's rows were re-set 2026-11-23** from `measure_gpu_tolerance` on
Linux 7.3.5-arch1-1, Mesa RADV 26.3.1, headless (no compositor running), on both
GPUs. At the prompts' own ends: every gemma configuration below, native lanes
and wave64 forced, tuning rows on and off (no tuning row matches RADV, B231, and
the two runs print the same values). With the prompt end moved by
`INFR_TOL_SPLIT` (prompt 0 ended after 4, 8 and 20 tokens; prompt 1 after 4, 5,
6, 8, 10, 12, 14, 16, 20, 24, 28, 40 and 50; prompt 2 after 4, 16, 64, 150, 170,
183 and 190): native lanes only, and `gpu`, `gpu-float`, `gpu-float-static`,
`gpu-float-nocoopmat` and `gpu-float-nocoopmat-static` only, so no `-ub32`
configuration. A moved prompt end is not a gate configuration. It counts as
legitimate for the two Float rows, because it is the same sequence on the same
kernels and gemma's distance at one position moves several-fold with it (B207).
It does not count for the Int8 row, whose other limits those runs break (below).

- **FloatNoCoopmat, the fine row** (`gpu-float-nocoopmat`, `-ub32`, `-static`).
  Both GPUs measure the same worst values to the digits given: KL 8.81e-4
  (prompt 1 ended after 12 tokens, position 13; 7.14e-4 at the prompt's own
  end), mean KL 5.7e-5, cosine 0.998998, logit deviation 7.4e-3, no top-1 miss
  and nothing ranked above the reference's choice. The integrated GPU's Float
  configurations and the RX 7900 XTX's with wave64 forced build the same kernels
  and measure the same KL. Every limit is the rule applied to those values, so
  the row is RADV-only: no Windows run of these configurations exists. The one
  older Windows figure that may be of this path, the integrated GPU's Float
  worst of 1.6e-3 as B207 quotes it (no printed run behind it, the code of
  2026-11-18, its path inferred), set nothing and would sit inside the KL bound.
  A Windows run outside the row is a finding to investigate, not a reason to
  widen it. The first rows measure far less (KL 2.1e-6) and take the steady
  bound.
- **Float, now a smoke row of the cooperative-matrix f16 prefill path**
  (`gpu-float`, `-ub32`, `-static`), labelled so in the code. On the RX 7900 XTX
  under RADV those configurations measure a worst KL of 1.03e-2 (prompt 1 ended
  after 6 tokens, position 13; 1.02e-2 at the prompt's own end), mean KL 8.1e-4
  (4.9e-4), cosine 0.98952, logit deviation 2.8e-2, at most one token ranked
  above the reference's choice and no top-1 miss at a reference margin over
  0.004 nats: the accuracy cost of those GEMMs' f16 operands (B207), above the
  Windows figure for the same path (3.2e-3), which set the row before. The row
  is the rule applied to the RADV values.
- **Int8, a smoke row as before, with a wider steady near tie.** At the prompts'
  own ends the widest top-1 miss from position 2 on is at a 1.185-nat reference
  margin (the integrated GPU, and the RX 7900 XTX with wave64 forced: prompt 2,
  position 179), and the margin is 1.5 times that. The first rows' margin is as
  it was (their widest miss here is at 1.08 nats, inside it). At the new margin
  the top-1 check counts 65 of the 216 prefill positions and 57 of the 96 decode
  positions of a `gpu` run, above `MIN_TOP1_SCORED`. No other limit of the row
  was changed, and two findings stand beside it (B207). Its margins at the
  prompts' own ends are thin on the integrated GPU: worst KL 0.310 (prompt 1,
  position 13) against the 0.26 the row was set from on Windows, mean KL 1.65e-2
  and logit deviation 0.115, which are 1.7, 1.63 and 1.30 times under their
  bounds where the rule asks for 2, 2 and 1.5. And the row does not hold at
  moved prompt ends: with prompt 1 ended after 10 or 12 tokens the RX 7900 XTX's
  int8 decode rows measure a worst KL of 0.62 and 0.70, a mean of 9.0e-2, a
  deviation of 0.165 and 0.192 and a top-1 miss at 1.349 nats, and the
  integrated GPU's prefill mean reaches 3.46e-2 and 4.0e-2 with prompt 2 ended
  after 16 and 4 tokens. The older account still holds: the same runs on Windows
  gave 0.26 on the integrated GPU (0.87 in the first rows, cosine 0.86) and
  2.6e-2 and 0.33 on the RX 7900 XTX natively, 0.11 and 0.57 with wave64 forced,
  1.2e-2 with the generic tuning, ten times apart between the two GPUs,
  unexplained.

**Both sides of each gap, on gemma-3-1b** (RADV, both GPUs, the committed
demonstrations above), worst legitimate value / tolerance / smallest planted bug
that crossed it, from position 2 on unless it says otherwise:

| Row                   | KL max                                                    | mean KL                      | Other limits a fine bug crossed                                                          | Verdict                                                                                                                 |
| --------------------- | --------------------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| FloatNoCoopmat (fine) | 8.8e-4 / 1.8e-3 / 2.3e-3 (d)                              | 5.7e-5 / 1.2e-4 / 2.6e-3 (c) | first rows' mean KL 1.6e-6 / 1.2e-4 / 3.0e-4 (d); cosine 0.998998 / 0.9979 / 0.99686 (d) | (d) 1.26x over the KL bound and 2.5x over the first rows' mean; (c) 14x over the KL bound; (d1) missed                  |
| Float (smoke)         | 1.03e-2 / 2.1e-2 / 1.56 (f); (c) at 1.9e-2 does not cross | 8.1e-4 / 1.7e-3 / 1.9e-3 (c) | logit deviation 2.8e-2 / 4.2e-2 / 6.7e-2 (c)                                             | Gross bugs, and (c) thinly (1.15x on mean KL, 1.6x on the deviation); (d) missed. Catches nothing the fine row does not |
| Int8 (smoke)          | 0.31 / 0.52 / 1.46 (f)                                    | 1.7e-2 / 2.7e-2 / 0.30 (f)   | none                                                                                     | Gross bugs only, (f) the smallest at 2.8x; (c) crosses on the integrated GPU alone (0.70), a recorded miss              |

The Float figures are the RX 7900 XTX's, the Int8 ones the integrated GPU's (the
RX 7900 XTX's Int8 configuration prefills on the f16 GEMMs under RADV and
measures 2.3e-2 clean, with (f) at 1.56). The bug figures are from the prompts'
own ends, and so are the Int8 row's legitimate ones; the two Float rows'
legitimate figures include the moved ends.

For every row but Qwen3-0.6B's and gemma-3's, the gap is unmeasured: the
tolerances are set from legitimate runs only.

**No ratchet.** A row may be loosened only by its owner, with the measurement
that needs it (device, driver, configuration, the run's printed values) and the
planted bugs above re-run against the new value, keeping a stated gap; never to
make a failing run pass without that evidence. A device that lands outside a row
is a finding first. The same rule is in `TABLE`'s doc comment.

## Running it

The gate's tests are `gpu_seam_tolerance_*` in
`crates/infr-llama/tests/gpu_tolerance.rs`, ignored by default like every GPU
test:

```text
INFR_DEV=Vulkan0 cargo nextest run -p infr-llama --release --test gpu_tolerance \
  --run-ignored all --test-threads 1 --no-capture -E 'test(gpu_seam_tolerance)'
```

Every test ends as a pass, a fail or a printed `skip:` line, never a hang:

- it skips without its model (and fails instead under
  `INFR_TEST_REQUIRE_MODELS`) or without a Vulkan device, and on an integrated
  GPU when free host memory is under twice the model plus a 12 GB margin;
- a watchdog (`GATE_TIMEOUT`, 20 minutes: about 4.5 times the slowest test
  measured, LFM3's 266 s on the integrated GPU) fails a case that runs too long,
  saying it is a hang. The bound is per `gate()` call, and some tests call it
  twice, so the worst case for the whole set is several hours, not 20 minutes;
  the watchdog ends the test process, so under plain `cargo test` (one process)
  the tests after it do not run;
- an integrated GPU scores the full prefill head like every other device. With
  every int8 kernel off that head is one vocabulary-wide dispatch of about 25 ms
  per chunk row there (3.26 s at the default 128-row chunk), and one submit of
  more than about 2 s is what Windows resets
  ([plan 12](plan/12-low-capability-vulkan.md) sections 0c and 0d). The backend
  now cuts such a dispatch into ranges of its grid, each in a submit of its own
  and bit-identical to the whole (`submit.budget_ms`, plan 12 S2), so the head
  runs as it is. Until 2026-11-22 those runs scored the prefill on each chunk's
  last row only, under a submit cap of 4: a harness workaround, removed with no
  table row changed.

These tests are one part of the **GPU gate**, the ignored GPU tests run on each
device before a slice lands. Its filter was kept in the scripts of whoever ran
it; it is written here so that a new GPU test is either selected by it or added
to it. The work bound's tests are an example: `binary(work_split_parity)` is
named, and the two ignored lib tests
(`attention_prefill_nc_fa_cut_is_bit_identical` and
`a_failed_one_shot_names_the_forward_segments_before_it`) come in through
`kind(lib)` of `infr-vulkan`, which no `test(gpu_seam)`-style name filter
selects.

```text
INFR_DEV=Vulkan0 cargo nextest run --release --run-ignored all --test-threads 1 \
  --no-fail-fast -E '(binary(seam_op_parity)
  | (package(infr-vulkan) & (kind(lib) | binary(dense_tier_parity)
    | binary(shared_decode_parity) | binary(pager_gemv_parity)
    | binary(pager_mmq_parity) | binary(first_forward_split)
    | binary(work_split_parity) | binary(dense_float_stream_parity)
    | binary(attn_decode_parity) | binary(attn_split_host)
    | binary(kq_fma_gemv) | binary(logits_tail) | test(linear_f32)))
  | test(gpu_seam) | test(gpu_synthetic) | test(gpu_prefill) | test(lfm3)
  | test(qwen35) | test(gemma) | test(paged) | test(deepseek) | test(dsv4)
  | test(mtp_head_cpu_vulkan_parity))
  & not test(qwen3_14b) & not binary(gemm_bench)
  & not (test(/_bw$/) | test(logits_tail_time))'
```

The second line of `infr-vulkan` binaries came with the kernels of PRs 105 to
107 (2026-11-22): the float decode GEMVs' streamed parity, decode split-K
attention against its general build and against a host reference, the fp32
K-quant GEMVs with the tuning parameter that selects them, the logits tail, and
`small_m_bench`'s two f32 GEMV parity tests by name. The last line keeps the
micro-benches that share those binaries out. A test that opens its backend
through `infr_testkit::gpu_or_skip` skips, printing why, when no Vulkan device
opens; with `INFR_DEV` set, as in every gate run, a device that does not open is
a failure instead, so a gate run cannot pass such a test without running it.

On the integrated GPU the gate of 2026-11-22 ran the same filter with
`& not (test(moe) | test(ernie))` appended, and the ERNIE tolerance case on its
own. `infr-vulkan`'s bench and probe binaries stay out on purpose: some of them
have lost the device on the integrated GPU.

`INFR_FORCE_WAVE64=1` runs the wave64 path, `INFR_TUNE_NO_ROWS=1` the generic
tuning defaults, and `INFR_TUNE=family.param=value` one tuning parameter forced
(all three reach these tests through `infr_testkit::honour_env`). The last is
how a device here stands in for one it is not:
`INFR_TUNE=gemv.kq_fma=true,attn_split.decode_min_chunk=16` scores every model
on the decode kernels and the attention chunking an RX 580 gets by default. The
capability masks (`INFR_NO_I8DOT`, `INFR_NO_COOPMAT`) do NOT reach the tests of
`infr-llama`, which build their config from the shipped defaults: the tolerance
tests run their own int8-free configurations (`GPU_CONFIGS`), and, gemma-3's
`gpu-float-nocoopmat*` runs apart, no whole-model test runs with the matrix
units masked off. `measure_gpu_tolerance` (also ignored) prints the full
comparison for the models named in `INFR_TOL_MODELS` (keys of `MODELS` in
`tests/support/forced.rs`), with `INFR_TOL_ONLY` to pick configurations and
`INFR_TOL_PROMPTS`, `INFR_TOL_SUBMIT`, `INFR_TOL_HEAD`, `INFR_TOL_UBATCH` and
`INFR_TOL_NO_REPLAY` as diagnosis knobs; it is what the table was set from. Two
more knobs answer "is this row wrong, or only sensitive": `INFR_TOL_ROWS=1`
prints every scored position's KL (and each top-1 that differs, with the
reference's margin) and the kernels the run built, and `INFR_TOL_SPLIT=<tokens>`
keeps the reference's sequence but ends the prompt after that many tokens, so
the same positions are scored with the prefill-to-decode boundary elsewhere (no
shorter than the prompt a forced run accepts: `seam::forced_run` refuses one too
short for a batched prefill). One configuration is for diagnosis only
(`DIAGNOSIS_CONFIGS`): no gate scores it, and `measure_gpu_tolerance` runs it
only when `INFR_TOL_ONLY` names it. `gpu-no-dp4a-kq` is the shipped
configuration without the dp4a K-quant prefill GEMM. The three
`gpu-float-nocoopmat*` configurations (the Float ones with the whole
cooperative-matrix tier off) were diagnosis too until 2026-11-23; they are gated
now, for gemma-3, and a default measurement runs them for every model.

**A tuning value left in the environment changes a gate verdict.** `INFR_TUNE`,
`INFR_TUNE_NO_ROWS` and `INFR_FORCE_WAVE64` reach these tests on purpose, so a
stray one is not ignored: with `INFR_TUNE=split_k.fixed_grid=0` set, gemma-3's
Float worst KL on the RX 7900 XTX under RADV went from 1.02e-2 to 4.18e-3,
across the row's bound of the time (below). Check the environment before reading
a gate run as the default path's.

**Results, 2026-11-18** (the ten `gpu_seam_tolerance_*` tests: seven model
cases, the red path, the prefill-head check and the int8-free check). On the RX
7900 XTX all ten passed natively and with wave64 forced, tuning rows on and off,
about 6.5 minutes a run (natively: Qwen3 61 s, Qwen3.5 73 s, gemma-3 77 s, LFM3
(both files) 112 s, ERNIE (both files) 54 s, the rest under 5 s each). On the
integrated GPU, native lanes, all ten passed with the workaround above in 637 s
(Qwen3 80 s, Qwen3.5 82 s, gemma-3 133 s, LFM3 266 s, ERNIE 67 s); wave64 and
rows off were not gate-run there. These GPU results predate the rebase onto the
CPU's FMA-tier commits on `main`.

**Results, 2026-11-22, integrated GPU, native lanes, workaround removed** (the
full head scored on every prefill row of every run, the table untouched): all
ten `gpu_seam_tolerance_*` tests passed, nine in the gate run (Qwen3 89 s,
Qwen3.5 99 s, gemma-3 140 s, LFM3 239 s, the rest under 5 s each) and the ERNIE
case, which the integrated GPU's gate filter leaves out, on its own in 69 s.

**Results, 2026-11-23, Linux 7.3.5-arch1-1, Mesa 26.3.1 (`mesa` and
`vulkan-radeon` 26.3.1-2), RADV, headless (no compositor running)**: the table
on a third driver, every model the tests name cached. On both the RX 7900 XTX
(`Vulkan0`, 283 s: ERNIE both files 38 s, gemma-3 63 s, LFM3 both files 73 s,
Qwen3 50 s, Qwen3.5 57 s, the rest under 2 s each) and the integrated GPU
(`Vulkan1`, 562 s: ERNIE 57 s, gemma-3 116 s, LFM3 219 s, Qwen3 82 s, Qwen3.5 84
s) nine of the ten tests passed and `gpu_seam_tolerance_gemma3` failed, with the
same values in every run, against the table as it then was. Neither failure is a
driver defect, each is an accuracy cost of a prefill GEMM family; gemma's rows
were re-set from these runs the same day ("The table" has the rows, the
paragraph after the two failures the gate's result on them, B207 what is left
open).

- **`Vulkan0`, gemma-3's Float row, prompt 1**: prefill worst KL 1.02e-2 at
  position 13 against the 6.4e-3 the row then allowed, decode 7.87e-3 at
  position 40. Not a driver defect: an accuracy cost of the cooperative-matrix
  f16 prefill GEMM path (`native_gemm_warp_*`). With the cooperative-matrix tier
  off (`gpu-float-nocoopmat`) the two rows measure 7.14e-4 and 4.22e-4, the
  integrated GPU's values, which has no cooperative matrices; decode builds the
  same kernels either way. Most of the distance arrives through the KV of the
  first few positions, not through row 13's own prefill: with the prompt ended
  after 4 tokens (`INFR_TOL_SPLIT=4`), so that only positions 0 to 2 go through
  those GEMMs and position 13 is a decode row, the two rows still measure
  8.73e-3 and 6.55e-3. f16 operand rounding is the inferred mechanism: the CPU
  reference with its batched-prefill operands rounded to f16 (a scratch hook) is
  3.4e-3 and 2.0e-3 from the unmodified reference on those rows, and the GPU is
  1.8e-3 and 1.9e-3 from it. The one kernel build probed
  (`native_gemm_warp_q4k_n128_ag`) agrees with a host model of f16 nearest-even
  weights and exact products to 1.7e-6 relative. Turning split-K off
  (`INFR_TUNE=split_k.fixed_grid=0`) moves the two rows to 4.18e-3 and 1.98e-3,
  inside the bound: one reordering, which shows the value moves with the
  summation and does not show which path is the more accurate. No tuning row
  matches RADV (B231), so the card's Int8 configurations prefill on the same f16
  GEMMs (the identical 1.022885e-2).
- **`Vulkan1`, gemma-3's Int8 smoke row, prompt 2**: position 179, the first
  decode row, flips its top-1 at a 1.185-nat margin against the 1-nat near tie
  of the time (KL 0.18, inside the bound). The row is that far off wherever the
  prompt ends (`INFR_TOL_SPLIT`): 0.200 with a flip as a decode row 30 past the
  boundary, 0.170 without one 10 past it, 0.166 without one as a batched-prefill
  row. The distance is in the KV the dp4a K-quant prefill GEMM wrote: without
  that kernel (`gpu-no-dp4a-kq`, still int8 activations in prefill and decode)
  the row measures 2.17e-2 and does not flip.

To reproduce the first:

```text
INFR_DEV=Vulkan0 INFR_TOL_MODELS=gemma3-1b INFR_TOL_PROMPTS=1 INFR_TOL_ROWS=1 \
  INFR_TOL_ONLY=gpu-float,gpu-float-nocoopmat cargo nextest run -p infr-llama --release \
  --test gpu_tolerance --run-ignored all --no-capture -E 'test(measure_gpu_tolerance)'
```

**Results after gemma's rows were re-set, 2026-11-23, same machine and drivers,
headless**, no environment knob but `INFR_DEV`. `gpu_seam_tolerance_gemma3`,
which now runs eight configurations (the five of every quantized model and the
three `gpu-float-nocoopmat*`), passes on both GPUs against the rows as they
stand, in 97 s on the RX 7900 XTX and 202 s on the integrated GPU, and so does
the check that the coopmat-free class builds no cooperative-matrix GEMM. The
other nine `gpu_seam_tolerance_*` tests were run on both GPUs with the gemma
rows of an earlier draft of that day (a looser FloatNoCoopmat KL bound and Int8
near tie; no other model's row or configuration differed) and all passed; they
were not run again after gemma's rows were tightened to what is in the table.
With wave64 forced on `Vulkan0`, gemma-3's test and the kernel check passed
under that draft too.

**The GPU gate on RADV, 2026-11-23** (the filter above, same machine, headless,
before gemma's rows were re-set; the full filter was not run again after). On
`Vulkan0`: 713 tests, 712 passed, the one failure `gpu_seam_tolerance_gemma3`,
549 s. On `Vulkan1`, with the integrated GPU's exclusion: 652 tests, 650 passed,
830 s; the failures are `gpu_seam_tolerance_gemma3` and `gpu_seam_golden_qwen3`,
whose first prompt hashes to `0xe91c03c96815974c` where `0xfd63781ea3bfa785` is
blessed (the text it printed reads as a normal answer, and
`gpu_seam_tolerance_qwen3` passes there). The hash is of the text, and the same
text is on record for this GPU from another change (B207), so it is the other
branch of a near tie in that prompt: an exact hash of one device's path, which
G4 retires. Which op takes the other branch under RADV was not found. Tests that
need a model not cached here printed `skip:` and count as passed: DeepSeek
V2-Lite, V3.2, LLM-7B and V4-Flash, DiffusionGemma, Qwen3.6-35B-A3B,
Llama-4-Scout, BitNet, the Q8_0 file of Llama-3.2-1B, three other quantizations
of Qwen3-0.6B and a BF16 model; on `Vulkan1` also the tests that need the f16
cooperative-matrix tier or a decode chain longer than 1.
`mtp_head_cpu_vulkan_parity` (in the filter) passed on both.
`mtp_spec_matches_target_only_greedy`, which no term of the filter selects and
which is ignored as "MTP parked", was run by name: it passed on `Vulkan0` and
failed on `Vulkan1`, where the speculative stream differs from plain greedy from
its first words, the failure its own doc comment describes.

The red-path tests scale `blk.1.ffn_down.weight` of the synthetic model by 1.02.
On the CPU a 1.01 scale already trips the gate and 1.008 and below pass
(measured in review), so 1.02 is about twice the threshold; the green control is
the reference against itself, at a distance of exactly 0. The int8-free check
reads the kernels a run BUILT, which are built lazily when first recorded: a
superset of what ran, so its negative check is the conservative one; its marker
list is typed by hand (B207).

## The existing exact GPU tests across the matrix

For G4: `cpu_backend`'s exact-hash GPU goldens and exact-token GPU-against-CPU
tests (`gpu_seam_golden_*`, `gpu_seam_matches_cpu_*`, 24 tests) all passed on
2026-11-18 on the RX 7900 XTX natively and with wave64 forced, tuning rows on
and off, and on the integrated GPU natively (20 tests, the MoE ones left out by
the memory rule). Most self-skip for want of a cached model: 12 or 13 actually
ran on the 7900 XTX (the Qwen3 and Qwen3-MoE goldens, ERNIE 0.3B and 21B,
gemma-3, LFM3, Qwen2, SmolLM3, unified Qwen3.5) and 9 on the integrated GPU. So
today's exact tests do not yet distinguish the configurations; they will once
tuned kernels change bits per device. They are also what covers the chained and
replayed decode the tolerance tests do not exercise. Not measured with the new
metrics: B60's cases (Qwen3.5-MoE, DiffusionGemma: not cached here) and B174's
router-bias case on ERNIE-4.5-21B (the Mutations section has it on a synthetic
MoE).

## Not measured

- **Metal.** Nothing here ran on Apple hardware; the harness drives the CPU and
  Vulkan seams only. A Mac owner would add a `ForcedOn::Metal` arm to
  `SeamModel::forced_logits` (the Metal seam runs the same
  `generate_dense_backend`), run `measure_gpu_tolerance` on the gated models,
  and add Metal rows to the table from those numbers.
- **MoE models**, quantized KV caches below 8 bits, the chained and replayed
  decode: see B207 in [`backlog.md`](backlog.md).
