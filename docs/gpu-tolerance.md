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
  head is added over every chunk row through `generate_dense_backend_bound`'s
  `prefill_logits` hook; nothing else in the graph changes, and the hook is
  `None` (no head, no buffer, no download) on every production call.
- **Decode rows**: one per-token decode step per position, each a warm call on
  the same session that feeds exactly the one new token (checked: every cached
  token reused, one fed), so it reads KV written by the prefill and by every
  earlier decode step. This is the per-token path, the record-once replay on
  Vulkan; a chained decode runs the same recorded graph several steps per submit
  and is not exercised, since forcing needs every step's logits.

The driver is held to the reference's own row identity in
`forced_run_reproduces_the_cpu_reference_on_every_path` (normal test job, no GPU
or model): another chunk height, and decode rows against prefill rows at the
same positions, bit for bit. Writing the prefill chunks' logits in the wrong
order fails it.

**Metrics per position** (`crates/infr-llama/src/tolerance/metrics.rs`), in f64
from the two f32 rows, on `infr_core::math` (the workspace takes no math from
the platform library):

| Metric         | Definition                                                                                                         | Why                                                                                                                                                                                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KL             | KL(reference ‖ test) over the full vocabulary                                                                      | The expected extra log-loss; shift-invariant. Computed as `ln(1 + E_p[expm1(e - c)]) - E_p[e - c]` with `e = t - r`, so two rows a rounding apart score their real distance (a direct sum on f32 exponentials is off by about 1e-7, above a float path's mean) |
| top-k cosine   | cosine of the two rows' probabilities over the reference's most probable tokens, until they hold 0.99 or 64 tokens | Where the distribution actually lives. A whole-vocabulary logit cosine is dominated by the bias every row shares and cannot separate a match from a mismatch (B60)                                                                                             |
| logit dev      | the largest difference after centring each row on its mean, over the reference row's range (max − min)             | Shift-invariant; the mean does not move with the top logits, so one wrong top logit counts once rather than shifting every log-probability                                                                                                                     |
| top-1          | the argmax agrees, counted only where the reference's top-2 margin exceeds the class's near-tie threshold          | A greedy flip that is not a tie. The number of positions left out is reported, and a case must score at least `MIN_TOP1_SCORED`, so the check cannot go vacuous                                                                                                |
| ref top-1 rank | how many tokens the test row ranks above the reference's choice                                                    | Separates "second by a hair" from "lost"                                                                                                                                                                                                                       |

Each is aggregated per run (one prompt, one configuration, one phase) as the
worst position, with where it was, and the mean. A NaN or infinite logit is an
immediate failure naming the position.

## The three-way comparison: which CPU mode is the reference

Measured 2026-11-18 on the RX 7900 XTX (`Vulkan0`, native lanes, tuning rows
on), three prompts of 12 to 185 tokens and 32 forced decode steps each, every
row prefill and decode together. Each cell is the worst KL at any position / the
mean KL over all positions (`measure_gpu_tolerance`). "GPU int8" is the shipped
default; "GPU float" turns every int8-activation kernel off (`vulkan_float` in
`tests/support/forced.rs`).

| Model                 | CPU prod vs ref    | GPU int8 vs ref    | GPU int8 vs CPU prod | GPU float vs ref   | GPU float vs CPU prod |
| --------------------- | ------------------ | ------------------ | -------------------- | ------------------ | --------------------- |
| Qwen3-0.6B Q4_K_M     | 7.69e-02 / 8.7e-03 | 3.50e-02 / 1.9e-03 | 6.87e-02 / 9.3e-03   | 3.60e-05 / 2.6e-06 | 7.47e-02 / 8.7e-03    |
| Qwen3-1.7B Q4_K_M     | 1.42e-01 / 7.3e-03 | 3.33e-02 / 1.5e-03 | 1.65e-01 / 9.0e-03   | 2.13e-05 / 1.9e-06 | 1.68e-01 / 7.6e-03    |
| gemma-3-1b Q4_K_M     | 1.88e+00 / 4.6e-02 | 3.30e-01 / 3.4e-03 | 1.97e+00 / 4.5e-02   | 3.16e-03 / 4.8e-05 | 1.99e+00 / 4.4e-02    |
| Qwen3.5-0.8B Q4_K_M   | 6.78e-02 / 3.1e-03 | 7.07e-03 / 6.9e-04 | 6.26e-02 / 3.6e-03   | 9.88e-06 / 8.2e-07 | 7.24e-02 / 3.1e-03    |
| LFM3-1.7B Q4_K_M      | 1.03e-01 / 5.0e-03 | 2.86e-02 / 1.5e-03 | 1.23e-01 / 5.4e-03   | 1.49e-04 / 2.7e-06 | 1.02e-01 / 4.8e-03    |
| ERNIE-4.5-0.3B Q4_K_M | 8.38e-01 / 1.8e-02 | 2.44e-02 / 2.2e-03 | 7.00e-01 / 1.8e-02   | 5.92e-05 / 3.5e-06 | 7.14e-01 / 1.7e-02    |
| Llama-3.2-1B Q4_K_M   | 7.17e-02 / 3.1e-03 | 1.33e-02 / 5.4e-04 | 1.27e-01 / 3.7e-03   | 4.36e-05 / 1.8e-06 | 6.30e-02 / 3.1e-03    |
| Qwen2.5-0.5B Q4_K_M   | 1.03e-01 / 2.5e-03 | 1.82e-03 / 2.0e-04 | 1.20e-01 / 2.7e-03   | 8.09e-04 / 8.1e-06 | 1.28e-01 / 2.6e-03    |
| SmolLM3-3B Q4_K_M     | 4.88e+00 / 3.4e-02 | 8.19e-03 / 6.9e-04 | 7.93e+00 / 6.6e-02   | 1.32e-05 / 1.3e-06 | 1.07e+01 / 8.7e-02    |

(The F16 and synthetic f32/f16 models have no quantized `Linear`, so the two CPU
modes are the same bits there and the GPU has no int8 path to take.)

What it says:

- **The CPU reference is the right reference, and production is not.** The
  shipped int8 GPU is 2 to 600 times closer to the reference (worst KL) than the
  production CPU is, and the GPU's distance to the production CPU is about the
  production CPU's own distance to the reference: the two int8 paths quantize
  different values at different points, and their errors add rather than cancel.
  Scoring a GPU against the production CPU would charge it with the CPU's error.
- **The GPU's int8 distance is its own quantization, not a bug.** Turning the
  int8 kernels off brings every quantized model to 1e-5 to 1e-4 worst KL
  (gemma-3 and Qwen2.5 aside, below), two to three orders of magnitude closer:
  the same kernels' float arithmetic is that close to the reference.
- **The production CPU is the furthest thing from the reference here.** On
  SmolLM3-3B its prefill reached KL 4.88 with a top-1 flip at a 8.8-nat
  reference margin, and on ERNIE-4.5-0.3B the reference's top-1 ranked 245th at
  one position — far beyond the GPU int8 path on the same rows. Not a GPU
  question, and not looked into (B207).
- **Sensitivity is per model.** gemma-3-1b is about a hundred times more
  sensitive than the Qwen models in both float and int8; that is why the table
  has rows per model, not one value per class.

## Path classes

`tolerance::table::PathClass`, each with its own rows:

- **Float**: f32/f16 weights, or quantized weights with every int8-activation
  kernel off (`vulkan_float`: `i8_dot` gates every packed-int8 dot kernel,
  including the non-coopmat `mmq` tier that wave64 and devices without
  cooperative matrices take; `i8cm_kq`, `dp4a_kq`, `mmq`, `mrow` and
  `mmv_mw = Some(false)` name the rest), f16 KV cache. What is left is the GPU's
  float rounding (f16 activations into the coopmat GEMMs, f16 attention,
  reduction order). Worst KL 1e-7 to 1e-4 on the real models, except gemma-3
  (3e-3 on the 7900 XTX at native lanes).
- **Int8**: quantized weights through the int8-activation kernels, the shipped
  default (the int8 coopmat and dp4a prefill GEMMs, the `mmq` GEMMs, the int8
  multi-row and decode GEMVs). Dominated by the activation quantization: worst
  KL about 1e-2, gemma-3 up to 0.87.
- **KvQ8**: the Float configuration with a `q8_0` K and V cache, scored against
  the CPU reference running the same cache. The GPU quantizes the f16 RoPE
  output and the CPU its f32 value, so the two caches round to different codes:
  worst KL about 2e-2 on Qwen3-0.6B, two orders above Float.

Every quantized model is gated in both Float and Int8: the Int8 rows catch only
gross bugs (below), so the fine check comes from the Float runs, which exercise
every non-int8 kernel (attention, norms, RoPE, the float GEMMs and GEMVs, the
recurrent scans) at the precision where small bugs show. A bug that lives only
inside an int8 kernel is the Float run's blind spot, covered only by the
op-level bounds (B207).

Lane width and the tuning rows: forced wave64 drops the cooperative-matrix
tiers, so its prefill runs other kernels and its own distances (lower on most
models in Float); the tuning rows change the RX 7900 XTX's native prefill (rows
off is the f16 coopmat GEMM, so its Int8 distances are smaller) and change
nothing measured under wave64 or on the integrated GPU.

## Mutations

Six bugs planted by hand in a scratch build (never committed) on Qwen3-0.6B, RX
7900 XTX native lanes, with the per-token decode on the static path
(`kernels.vulkan.no_replay`, so a graph-level edit reaches decode), each scored
against the Qwen3-0.6B rows (ratios are the measured value over the tolerance;
cosine as distance from 1):

| Bug                                                                             | Int8 run                                                                                              | Float run                                                                       |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| (a) RoPE: the query rotated at position + 1 (keys correct)                      | caught: KL 19 (270x), every metric                                                                    | caught: KL 19, every metric                                                     |
| (b) attention mask: the last visible key dropped (decode and every later chunk) | caught in decode: KL 17, every metric; the prompt's single prefill chunk starts at 0 and is untouched | caught in decode, as Int8                                                       |
| (c) RMSNorm epsilon ten times too large (1e-5 for 1e-6)                         | **missed**: KL 4.0e-2 (0.57x), mean 0.51x, deviation 0.74x                                            | caught: KL 3.1x, mean KL 4.5x, cosine 1.2x; deviation 1.4x                      |
| (d1) one Q4_K block of `blk.10.ffn_down` with its scale times 1.01              | **missed**: indistinguishable from the clean run                                                      | **missed**: KL 0.35x, mean 0.52x, deviation 0.5x                                |
| (d) every block of `blk.10.ffn_down` with its scale times 1.01                  | **missed**: KL 0.51x, mean 0.5x                                                                       | caught: KL 5.7x, mean 11x, cosine 5.2x, deviation 4.3x, a rank and top-1 change |
| (e) gate and up swapped in the FFN (split gate/up path)                         | caught at once: NaN logits, the position named                                                        | (the run stopped at the NaN)                                                    |
| (f) layer 14's attention residual dropped                                       | caught: KL 0.89 (13x), cosine 9x, deviation 4x, rank 16 against 5, top-1 flips at 1.5-nat margins     | caught: KL 1.1e4x, every metric                                                 |

What the gate cannot see: a sub-percent error confined to one weight block (d1)
in either class, and anything smaller than the int8 quantization (c, d) when the
int8 kernels are on. Those are why every quantized model also runs in Float, and
why op-level and dequant checks stay. Not tried: a router that ignores its bias
(B174's case, MoE), a wrong quant nibble map, a RoPE pairing error (B207).

## The table

`TABLE` has one row per (model, class) the tests run; its doc comment carries
the provenance (date, devices, driver, base commit) and the margins: KL max and
mean times 2, the cosine's distance from 1 times 2, the logit deviation times
1.5, the rank at twice the worst plus one, the near-tie margin at 1.5 times the
widest legitimate top-1 miss (at least 0.25 nats in Float, 1 nat in Int8 and
KvQ8). The worst legitimate KL of a row moves by up to about 4x between the RX
7900 XTX native, its wave64 path and the integrated GPU (Qwen3-0.6B Float:
4.2e-5, 9.8e-6, 1.5e-5), and the tolerance starts from the worst of all of them;
the margin of 2 on top is headroom for a device or driver not measured here,
kept small because the planted bugs below sit only 3 to 5 times above the
legitimate worst in Float.

**Both sides of each gap, on Qwen3-0.6B** (the only model the bugs were planted
in), worst legitimate value / tolerance / smallest planted bug that crossed it:

| Class | KL max                       | mean KL                      | Verdict                                                                                                                     |
| ----- | ---------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Float | 4.2e-5 / 8.5e-5 / 2.7e-4 (c) | 3.6e-6 / 7.3e-6 / 3.3e-5 (c) | A gap of 3x (KL) to 4.5x (mean) above the tolerance; a tenfold norm epsilon and a 1% tensor-wide scale error are caught     |
| Int8  | 3.5e-2 / 7.0e-2 / 0.89 (f)   | 4.6e-3 / 9.2e-3 / 0.13 (f)   | Only gross bugs: (c) and (d) sit at half the tolerance, inside the legitimate range — the int8 class cannot see them at all |
| KvQ8  | 1.9e-2 / 3.9e-2 / not run    | 1.2e-3 / 2.5e-3 / not run    | No bug was planted with the q8 cache; by its distances it sees what Int8 sees                                               |

**Rows with no usable gap.** gemma-3-1b in Int8: its legitimate worst KL is 0.87
(integrated GPU) and its top-k cosine falls to 0.41, so its KL-max and cosine
limits (1.8, and a floor of 0.01) admit a bug the size of (f) on Qwen3-0.6B;
only its mean KL (0.032 against (f)'s 0.13 on Qwen3) would still see one, and
nothing was planted in gemma to confirm it. Its Float row is the real check for
gemma. For every row but Qwen3-0.6B's, the gap is unmeasured: the tolerances are
set from legitimate runs only.

**The gate fails on a planted bug end to end**, not only in the measurement:
with (c) and with (f) planted, `gpu_seam_tolerance_qwen3` failed on the RX 7900
XTX on the production decode path (the record-once replay), (c) through the
Float runs (KL 2.3e-4 against 8.5e-5, mean KL twice its limit, cosine 0.99984
against 0.9999), (f) through every run.

## Running it

The gate's tests are `gpu_seam_tolerance_*` in
`crates/infr-llama/tests/gpu_tolerance.rs`, ignored by default like every GPU
test. Each skips without its model (and fails instead under
`INFR_TEST_REQUIRE_MODELS`) or without a Vulkan device, and prints every
measured value on success, so a new device's numbers can be read from the log:

```text
INFR_DEV=Vulkan0 cargo nextest run -p infr-llama --release --test gpu_tolerance \
  --run-ignored all --test-threads 1 --no-capture -E 'test(gpu_seam_tolerance)'
```

**Results, 2026-11-18.** On the RX 7900 XTX all seven tests passed natively and
with wave64 forced, tuning rows on and off (each run about 5 minutes; per test
natively: Qwen3 48 s, Qwen3.5 54 s, gemma-3 66 s, LFM3 (both files) 94 s, ERNIE
(both files) 47 s, the synthetic models about 1 s each). On the integrated GPU
(native lanes) the synthetic and ERNIE tests passed (4 s and 65 s);
`gpu_seam_tolerance_qwen3` failed with a lost device on its `q8_0` KV + Float
run of the 180-token prompt, and `gpu_seam_tolerance_lfm3` stopped making
progress in its Float run of that prompt and was killed after 18 minutes;
Qwen3.5 and gemma-3 were not run there, having crashed the same way in the
measurement. These are the B207 Float-on-the-integrated-GPU failures; the gate
reached no tolerance verdict for those two models on that device. The wave64 and
rows-off combinations were not gate-run on the integrated GPU.

`INFR_FORCE_WAVE64=1` runs the wave64 path and `INFR_TUNE_NO_ROWS=1` the generic
tuning defaults (both reach these tests through `infr_testkit::honour_env`).
`measure_gpu_tolerance` (also ignored) prints the full comparison for the models
named in `INFR_TOL_MODELS` (keys of `MODELS` in `tests/support/forced.rs`), with
`INFR_TOL_ONLY` to pick configurations; it is what the table was set from.

## The existing exact GPU tests across the matrix

For G4: `cpu_backend`'s exact-hash GPU goldens and exact-token GPU-against-CPU
tests (`gpu_seam_golden_*`, `gpu_seam_matches_cpu_*`, 24 tests) all passed on
2026-11-18 on the RX 7900 XTX natively and with wave64 forced, tuning rows on
and off, and on the integrated GPU natively (20 tests, the MoE ones left out by
the memory rule). Most self-skip for want of a cached model: 12 or 13 actually
ran on the 7900 XTX (the Qwen3 and Qwen3-MoE goldens, ERNIE 0.3B and 21B,
gemma-3, LFM3, Qwen2, SmolLM3, unified Qwen3.5) and 9 on the integrated GPU. So
today's exact tests do not yet distinguish the configurations; they will once
tuned kernels change bits per device. Not measured with the new metrics: B60's
cases (Qwen3.5-MoE, DiffusionGemma: not cached here) and B174's router-bias case
(needs a `moe_topk.comp` edit and ERNIE-4.5-21B). B141's KV-format check is
superseded in measurement by the KvQ8 row and the `q8_0`, `q4_0` and `turbo3`
figures in B207.

## Not measured

- **Metal.** Nothing here ran on Apple hardware; the harness drives the CPU and
  Vulkan seams only. A Mac owner would add a `ForcedOn::Metal` arm to
  `SeamModel::forced_logits` (the Metal seam already runs the same
  `generate_dense_backend` with `logits_out`), run `measure_gpu_tolerance` on
  the gated models, and add Metal rows to the table from those numbers.
- **MoE models**, quantized KV caches below 8 bits, and chained decode: see B207
  in [`backlog.md`](backlog.md).
