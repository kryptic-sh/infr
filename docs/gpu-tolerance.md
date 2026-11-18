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

Six bugs planted by hand in a scratch build (never committed) on Qwen3-0.6B, RX
7900 XTX native lanes, with the per-token decode on the static path
(`kernels.vulkan.no_replay`, so a graph-level edit reaches decode), scored
against the Qwen3-0.6B rows from position 2 on (ratios are the measured value
over the tolerance; cosine as distance from 1):

| Bug                                                                             | Int8 run                                                                                              | Float run                                                                |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| (a) RoPE: the query rotated at position + 1 (keys correct)                      | caught: KL 19 (420x), every metric                                                                    | caught: every metric                                                     |
| (b) attention mask: the last visible key dropped (decode and every later chunk) | caught in decode: KL 17 (380x); the first chunk starts at 0 and is untouched                          | caught in decode                                                         |
| (c) RMSNorm epsilon ten times too large (1e-5 for 1e-6)                         | **missed**: KL 1.35e-2 (0.30x), mean 0.44x — the clean run measured 1.47e-2 and 0.47x                 | caught: KL 3.1x, mean KL 4.5x, cosine 1.2x, deviation 1.4x               |
| (d1) one Q4_K block of `blk.10.ffn_down` with its scale times 1.01              | **missed** (0.33x)                                                                                    | **missed**: KL 0.35x, mean 0.52x                                         |
| (d) every block of `blk.10.ffn_down` with its scale times 1.01                  | **missed**: KL 0.32x, mean 0.54x                                                                      | caught: KL 5.7x, mean 11x, cosine 2.8x, deviation 4.3x, rank 2 against 1 |
| (e) gate and up swapped in the FFN (split gate/up path)                         | caught at once: NaN logits, the position named                                                        | (the run stopped at the NaN)                                             |
| (f) layer 14's attention residual dropped                                       | caught: KL 0.89 (20x), mean 32x, cosine 20x, deviation 4x, rank 16 against 5, top-1 flips at 1.5 nats | caught: KL 1.1e4x, every metric                                          |

Splitting the first rows out tightened Qwen3's steady Int8 bound by a third, and
it does not change the Int8 conclusion: (c) and (d) leave the int8 run's
distance where the clean run's is, so no bound on that run can see them. The
Int8 rows catch gross bugs only; the fine check is the Float runs, which
exercise every non-int8 kernel (attention, norms, RoPE, the float GEMMs and
GEMVs, the recurrent scans) at the precision where small bugs show. A bug living
only inside an int8 kernel is the Float run's blind spot, covered by the
op-level bounds (B207). A sub-percent error confined to one weight block (d1) is
invisible to both. Not tried: a router that ignores its bias (B174's case, MoE),
a wrong quant nibble map, a RoPE pairing error.

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

**Both sides of each gap, on Qwen3-0.6B** (the only model the bugs were planted
in), from position 2 on, worst legitimate value / tolerance / smallest planted
bug that crossed it:

| Class | KL max                       | mean KL                      | Verdict                                                                        |
| ----- | ---------------------------- | ---------------------------- | ------------------------------------------------------------------------------ |
| Float | 4.2e-5 / 8.5e-5 / 2.7e-4 (c) | 3.6e-6 / 7.3e-6 / 3.3e-5 (c) | A gap of 3x (KL) to 4.5x (mean) above the tolerance; (c) and (d) caught        |
| Int8  | 2.2e-2 / 4.5e-2 / 0.89 (f)   | 2.0e-3 / 4.1e-3 / 0.13 (f)   | Gross bugs only: (c) and (d) do not move the int8 distance off the clean run's |
| KvQ8  | 1.9e-2 / 3.9e-2 / not run    | 1.2e-3 / 2.5e-3 / not run    | No bug was planted with the q8 cache; by its distances it sees what Int8 sees  |

**No usable gap: gemma-3-1b in Int8**, labelled a smoke check in the code. Its
legitimate worst KL is 0.26 from position 2 on (0.87 in the first rows) and its
cosine falls to 0.86, all on the integrated GPU (native lanes, configuration
`gpu`, prompt 2, positions 115 and 1). The same runs on the RX 7900 XTX give
2.6e-2 and 0.33 natively, 0.11 and 0.57 with wave64 forced, and 1.2e-2 with the
generic tuning: ten times apart between the two GPUs, unexplained (B207), so its
bounds admit a bug the size of (f) on Qwen3; its Float row is the real check for
gemma. For every row but Qwen3-0.6B's, the gap is unmeasured: the tolerances are
set from legitimate runs only.

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
- on an integrated GPU the runs whose configuration turns every int8 kernel off
  score the prefill on each chunk's last row only and run with a submit cap of 4
  (`INTEGRATED_FLOAT_SUBMIT_CAP`), said in the log. This is a **harness
  workaround for a production weakness**, not a fix: with every int8 kernel off,
  the harness's prefill head is one vocabulary-wide dispatch of about 25 ms per
  chunk row there, and at the default 128-row chunk (3.26 s) Windows' 2 s
  display-driver watchdog resets the engine even with that dispatch alone in its
  submit ([plan 12](plan/12-low-capability-vulkan.md) sections 0b and 0c). It
  was kept when the cause was established (2026-11-18): no submit cap splits one
  dispatch, so it stays until the backend splits heavy ops (backlog B221). Such
  a run now ends in a backend error instead of a hang. The prefill phase of
  those runs scores one row per chunk, is not held to `MIN_TOP1_SCORED`, and
  does not score positions 0 and 1 at all unless one ends a chunk. It is keyed
  on the device being integrated: a slow discrete card (issue 94's class) gets
  no workaround and would meet the same error; the float-weight models' shipped
  `gpu` runs and the int8-free check keep the full head there too, and
  completed.

`INFR_FORCE_WAVE64=1` runs the wave64 path and `INFR_TUNE_NO_ROWS=1` the generic
tuning defaults (both reach these tests through `infr_testkit::honour_env`).
`measure_gpu_tolerance` (also ignored) prints the full comparison for the models
named in `INFR_TOL_MODELS` (keys of `MODELS` in `tests/support/forced.rs`), with
`INFR_TOL_ONLY` to pick configurations and `INFR_TOL_PROMPTS`,
`INFR_TOL_SUBMIT`, `INFR_TOL_HEAD`, `INFR_TOL_UBATCH` and `INFR_TOL_NO_REPLAY`
as diagnosis knobs; it is what the table was set from.

**Results, 2026-11-18** (the ten `gpu_seam_tolerance_*` tests: seven model
cases, the red path, the prefill-head check and the int8-free check). On the RX
7900 XTX all ten passed natively and with wave64 forced, tuning rows on and off,
about 6.5 minutes a run (natively: Qwen3 61 s, Qwen3.5 73 s, gemma-3 77 s, LFM3
(both files) 112 s, ERNIE (both files) 54 s, the rest under 5 s each). On the
integrated GPU, native lanes, all ten passed with the workaround above in 637 s
(Qwen3 80 s, Qwen3.5 82 s, gemma-3 133 s, LFM3 266 s, ERNIE 67 s); wave64 and
rows off were not gate-run there. These GPU results predate the rebase onto the
CPU's FMA-tier commits on `main`.

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
router-bias case (needs a `moe_topk.comp` edit and ERNIE-4.5-21B).

## Not measured

- **Metal.** Nothing here ran on Apple hardware; the harness drives the CPU and
  Vulkan seams only. A Mac owner would add a `ForcedOn::Metal` arm to
  `SeamModel::forced_logits` (the Metal seam runs the same
  `generate_dense_backend`), run `measure_gpu_tolerance` on the gated models,
  and add Metal rows to the table from those numbers.
- **MoE models**, quantized KV caches below 8 bits, the chained and replayed
  decode: see B207 in [`backlog.md`](backlog.md).
