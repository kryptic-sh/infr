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

**Routed-expert (MoE) models are forced along the reference's routing too.** A
router picks each token's top experts, and at a near tie two backends
legitimately pick different ones; from there they compute different functions,
and every later row that attends to that position inherits the difference (the
synthetic MoE's case under "Mutations"). So a forced run takes, at every
position it feeds and in every router-selected layer, the experts the CPU
reference selected there, through `Op::MoeFfn`'s `expert_ids` (the path DeepSeek
V4's hash-routed layers take in production: the router still runs and its
probabilities still weight the experts, only the selection is supplied). The
logit rows then compare the two backends' arithmetic under one routing. The
reference run is left to its own routers, and `ForcedLogits::routing` carries
what they selected, with the score every expert was ranked by.

**What the routers select is compared on its own.** Each forced graph also holds
an `Op::MoeRoute` per such layer, which reads back the selection the backend's
own router makes on the hidden state it has: the same router dispatch and the
same `moe_topk` selection its `Op::MoeFfn` lowering runs, with the selection
bias and group-limited routing, nothing run on the result. That is what the run
would have routed to had it been left alone, given the reference's routing up to
that layer. `tolerance::metrics::RouteScore` compares it with the reference's
selection as a set, at every position fed (prompt positions included, whether or
not their logits are scored) and every layer. Where the two differ, the margin
(`RouteFlip::margin`) is how far the reference's scores (selection probability
plus bias) would have to move for its router to select the run's set: over the
experts in exactly one of the two sets, its highest score among those only it
selected, less its lowest among those only the run selected. For one expert
swapped for another that is the gap between the two; a run that drops the
reference's clear first choice is not excused by a near tie further down the
ranking, and one that selects an expert of a group the reference masked out is
at an infinite margin. Whether two selections differ is read off the ids alone.
A readback that is not a selection (an expert named twice, as a top-k that did
not mask its pick or a buffer nothing wrote would give, or an id past the
experts) is an error of the run, not a result. A difference at a margin over the
row's `route_near_tie` fails the gate, the routing counterpart of the top-1 near
tie, and over a case at least `MIN_TOP1_SCORED` of the reference's selections
must clear that margin or the check is vacuous. Hash-routed layers have no
router selection and are left as they are.

Two checks hold the mechanism itself. On the CPU, in the normal test job
(`forced_run_reads_back_and_takes_a_routing_on_the_cpu`): a run forced along its
own selections is the same run bit for bit, and with one selection replaced
every earlier row keeps its bits, that position's row changes, and the router
there still reports the selection it would have made. On a GPU
(`gpu_seam_tolerance_own_routing_is_the_selection_it_runs`): a run left on its
own routers and the same run forced along the selections it read back give the
same logits bit for bit, on the replayed and the static decode, so the readback
is the selection `Op::MoeFfn` makes; with one selection replaced the bits
change. Nothing of this exists outside a forced run: `Op::MoeRoute` is emitted
only when `generate_dense_backend_bound` is handed a `RouteTap`, which only
`seam::forced_run` does. The CPU's router code is now shared between the two
ops, and neither the op-hash suite's existing `MoeFfn` hashes nor the CPU MoE
goldens (`cpu_golden_ernie4_5_moe`, `cpu_golden_qwen3moe`) moved.

**What the decode side does not exercise.** Where the backend replays, each
forced step builds, records and executes a fresh replay plan once: the
record-once-replay-many mechanism (one recording replayed with advancing
parameters) and the chained decode (`decode_chain`, several steps per submit,
the production greedy path) are never run by a forced run, because forcing needs
every step's logits and the chain does not read them back. They are held to the
per-token path by a comparison of their own, which every gate case runs as well
("The greedy check" below). The static per-token path
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

| Metric         | Definition                                                                                                                                                                                                                 | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KL             | KL(reference ‖ test) over the full vocabulary                                                                                                                                                                              | The primary metric: the expected extra log-loss, shift-invariant. Computed as `ln(1 + E_p[expm1(e - c)]) - E_p[e - c]` with `e = t - r`, so two rows a rounding apart score their real distance; a reference token far down the row (its `e - c` too large to exponentiate, or its f32 probability underflowed) enters the normalizer as `e^(t - lse_r - c)`, so promoting one is charged what it costs; only a term itself beyond f32's range scores infinite |
| top-k cosine   | cosine of the two rows' probability vectors over the union of both rows' top sets (0.99 of the mass or 64 tokens each), plus one component for the mass outside it                                                         | Secondary, never the sole detector of a class of bug: sees mass moved within or out of the top tokens even when the reference's top set is one token, but only means something when the union holds most of the mass (on a high-entropy row the rest component dominates: two rows flat over different 1000-token sets of 32000 score 0.99993 at KL 30). A whole-vocabulary logit cosine cannot separate a match from a mismatch (B60)                         |
| logit dev      | the largest difference after centring each row on its mean, over the reference row's range (max − min)                                                                                                                     | Shift-invariant; the mean does not move with the top logits, so one wrong top logit counts once                                                                                                                                                                                                                                                                                                                                                                |
| top-1          | the argmax agrees, counted only where the reference's top-2 margin exceeds the row's near-tie threshold                                                                                                                    | A greedy flip that is not a tie. Near ties left out are reported, and each of the prefill and decode phases must score at least `MIN_TOP1_SCORED` positions over a case                                                                                                                                                                                                                                                                                        |
| ref top-1 rank | how many tokens the test row ranks above the reference's choice                                                                                                                                                            | Separates "second by a hair" from "lost"                                                                                                                                                                                                                                                                                                                                                                                                                       |
| routing        | MoE only, per (position, layer) and not per row: the run's own router selects the reference's set of experts, counted only where the reference's margin between the experts in question exceeds the row's routing near tie | The logit rows are forced along the reference's routing and cannot see a router that selects wrongly; this is what does                                                                                                                                                                                                                                                                                                                                        |

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
kernel off (`vulkan_float` in `infr_llama::tolerance::gate`).

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

## The greedy check

A forced run never replays one recording at advancing positions and never chains
several decode steps into one submit ("The comparison"). Until 2026-11-23 those
two paths, the production greedy decode, were held by exact hashes of generated
text and exact-token comparisons with the CPU, which a legitimate change of a
decode kernel's arithmetic breaks: `gpu_seam_golden_qwen3` failed for
`attn_split.decode_min_chunk` at 16 and for `gemv.kq_fma`, both inside the
tolerance (B227), and under RADV it failed on the integrated GPU as blessed.
Every `gate()` case now runs this instead (`infr_llama::tolerance::greedy`,
`crates/infr-llama/src/tolerance/greedy.rs`), in each of its configurations that
holds a set of decode kernels of its own (`GREEDY_CONFIGS`: the shipped kernels,
the int8-free ones, the q8 KV cache's).

**The prompts** are the case's three plain-text ones and, for a real model, one
more rendered through the model's own chat template (`greedy::chat_prompt`), so
that the template's special tokens go through the device's embedding gather,
which the forced runs' prompts never feed. That prompt is in the greedy check
only: the forced rows are not scored on it, because the table's rows were not
measured with it.

**Three generations of each prompt, on one device and configuration**, each
asked for `GREEDY_STEPS` greedy tokens. That count is one more than a whole
number of chained submits at the shipped `spec.decode_chain`, so on a device
that chains the call ends with a single replayed step after its last chain: the
hand-off from a chain to one replay inside a call. It is a constant of its own;
the forced runs keep `N_DECODE`, which the rows were measured at.

- **chained**: `SeamModel::greedy_tail_vulkan` on the configuration as it is,
  which calls the function `generate_dense_vulkan` is
  (`generate_dense_vulkan_session`: the production placement, prefill and
  decode) and chains `spec.decode_chain` steps per submit. Run only on a device
  that chains (`Backend::max_decode_chain`); an integrated GPU runs one step per
  submit, and its production decode is the next one.
- **replayed**: the same with `spec.decode_chain = 1`: one recording, replayed
  once per token with its position advancing.
- **stepped**: the teacher-forced driver left to its own argmax
  (`forced_logits`, no prefill head), which records a fresh plan for every step:
  the path every forced run scores against the table.

After each of the first two, one more per-token step on the same session returns
the logits after the last sampled token (`GreedyTail::row`). That step attends
over every KV row the generation wrote, so it is how the test reads what a chain
left behind, which the chain itself never hands back.

**What is asserted**, per prompt and configuration:

- **Each generation stops where the per-token path does.** The stepped driver
  does not stop itself, so its tokens say where a generation must: at its first
  end token, or at the length asked for. Each generation must have sampled
  exactly that many tokens and report a natural stop
  (`GreedyTail::natural_stop`) exactly when the last of them is an end token.
  One cut short and one that runs past its end token both fail.
- **Each generation ran the path it stands for.** Each made exactly one decode
  recording; the replayed one reports no token from a chained submit
  (`GenStats::n_chained`); the chained one reports exactly the count its length
  and the chain imply, and a full-length one must hold both chained tokens and a
  token that is not, or the hand-off was not exercised. Whether a model chains
  is a property the test states of the case (`Case::chains`), as
  `Case::batched_prefill` is: the synthetic models are marked as not chaining
  (their token embedding is gathered on the host, so every step has host work
  before the next) and must then chain nothing. A case none of whose runs is in
  `GREEDY_CONFIGS` fails, instead of passing with nothing compared.
- **The three choose the same tokens, and each tail row is the stepped run's row
  at that position bit for bit.** One device compared with itself: no tolerance,
  no table row, and a tuning change that moves the arithmetic moves all three
  alike. This is the part that replaces what the hashes covered and the forced
  runs do not.
- **The production tokens are the reference's outside near ties.** The CPU
  reference is forced along the tokens the GPU chose, with the configuration's
  KV format, so both see the same context at every step. Wherever the
  reference's top-2 margin is above the row's steady near tie, the GPU's token
  must be the reference's top token (`tolerance::TokenScore`: the table's top-1
  rule, for a path that returns tokens and no logits), and over a configuration
  at least `MIN_TOP1_SCORED` steps must clear the margin. A routed-expert
  model's tokens are not held to the reference: left to its own routers a run
  legitimately parts from it at a router near tie ("Mutations", the synthetic
  MoE), which this rule cannot tell from a wrong token. Its three generations
  are still compared with each other.

**What it does not see:**

- **A chained step's logits.** The chain is seen through its tokens and through
  the one row after it. An error in the head or the argmax of a chained step
  that flips no token is invisible.
- **Fine bugs, through the tokens.** The token rule sees far less than the logit
  rows (the table below); the forced runs are what catch those.
- **Not exercised:** the chained sampler (`Op::Sample`, temperature above 0), a
  chain length other than the shipped `spec.decode_chain`, a chain or a replay
  on a KV format other than f16 and coupled q8_0, on a session continued from an
  earlier turn, and on any model without a gate case.
- **Windows.** Bit-for-bit agreement of a chain, a replay and a fresh recording
  was measured under RADV only.

**Measured 2026-11-23**, Linux 7.3.5-arch1-1, Mesa RADV 26.3.1, headless, no
environment knob but `INFR_DEV` (the `gpu_seam_tolerance_*` tests, with their
output). On every model, configuration and prompt below the generations stopped
at the same step, chose the same tokens and left bit-identical tail rows: on
`Vulkan0` (RX 7900 XTX) chained, replayed and stepped, every full-length
generation of a real model taking 32 of its 33 tokens from chained submits and
the last from one replay; on `Vulkan1` (the integrated GPU, which does not
chain) replayed and stepped. Each cell is the steps whose reference margin
clears the near tie, all of which chose the reference's token, then the widest
reference margin at which a token differed, in nats (0: the GPU's tokens are the
reference's own). A configuration generates over four prompts, 132 steps unless
a generation ends itself.

| Model, configuration (class)                | `Vulkan0`  | `Vulkan1` |
| ------------------------------------------- | ---------- | --------- |
| Qwen3-0.6B Q4_K_M, `gpu` (Int8)             | 93; 0.148  | 95; 0.106 |
| Qwen3-0.6B Q4_K_M, `gpu-float` (Float)      | 123; 0.009 | 125; 0    |
| Qwen3-0.6B Q4_K_M, `gpu-kv-q8-float` (KvQ8) | 95; 0.146  | 90; 0.030 |
| Qwen3.5-0.8B Q4_K_M, `gpu` (Int8)           | 96; 0      | 98; 0.026 |
| Qwen3.5-0.8B Q4_K_M, `gpu-float` (Float)    | 123; 0     | 123; 0    |
| gemma-3-1b Q4_K_M, `gpu` (Int8)             | 66; 0      | 69; 1.185 |
| gemma-3-1b Q4_K_M, `gpu-float` (Float)      | 105; 0.004 | 105; 0    |
| LFM3-1.7B Q4_K_M, `gpu` (Int8)              | 93; 0      | 93; 0     |
| LFM3-1.7B Q4_K_M, `gpu-float` (Float)       | 119; 0     | 119; 0    |
| LFM3-1.7B F16, `gpu` (Float)                | 116; 0     | 116; 0    |
| ERNIE-4.5-0.3B Q4_K_M, `gpu` (Int8)         | 75; 0.120  | 76; 0.120 |
| ERNIE-4.5-0.3B Q4_K_M, `gpu-float` (Float)  | 95; 0      | 95; 0     |
| ERNIE-4.5-0.3B F16, `gpu` (Float)           | 91; 0      | 91; 0     |

No model's row failed the token rule on the chat prompt. gemma-3's Int8 miss on
the integrated GPU is the near tie the row's margin was set from (prompt 2,
position 179, "The table"). Generations that end themselves, the same on both
GPUs: the chat prompt's on ERNIE (9 tokens on the Q4_K_M file, 8 on the F16 one)
and on gemma-3 (10), all from chained submits on `Vulkan0`; the others run the
full length. The synthetic models end on their end token on their shorter
prompts (57 steps counted on both GPUs for the f32 model, 57 and 50 for the f16
one, whose long prompt ends after 26 tokens on `Vulkan1` and runs the full
length on `Vulkan0`). The check takes 0.3 to 25.2 s of a model's case on
`Vulkan0` and up to 108.3 s on `Vulkan1` (LFM3 Q4_K_M).

**The motivating case, run the same day on `Vulkan0`** with the check as first
written (32-token generations, no chat prompt, no stop rule; not run again after
those were added): with
`INFR_TUNE=gemv.kq_fma=true,attn_split.decode_min_chunk=16`, the two values
`gpu_seam_golden_qwen3` failed for, every model's case passed, forced rows and
greedy check alike (the shipped runs then build the `native_gemv_kq_fma_*`
kernels). One test of the set fails under it,
`gpu_seam_tolerance_float_runs_build_no_int8_kernel`, at its own guard: with the
fp32 decode GEMVs forced, the shipped configuration builds no int8 kernel under
RADV, so the check reports that it cannot fail (B207).

**Shown to fail** (`--features mutations`, `tests/planted/greedy.rs`;
2026-11-23, Linux 7.3.5-arch1-1, Mesa RADV 26.3.1, headless, both GPUs;
Qwen3-0.6B Q4_K_M, `gpu` and `gpu-float`):

- **Bugs no forced run can carry.** With (j) or (k) planted ("Mutations") every
  forced run is the clean model's bit for bit, which each test asserts by
  digest, and the gate's rows pass. The greedy check passes on the clean model
  in the same test and must then fail on every prompt (the chat prompt included)
  and configuration, for the replayed generation and, on a device that chains,
  for the chained one, each failure naming its generation:
  - **The token half** (`gpu_planted_replay_feeds_stale_token`, (j)): on the RX
    7900 XTX the chained generation parts from the per-token path at step 8, the
    first token of its second submit, and the replayed one at step 1; on the
    integrated GPU the replayed one at step 1. No failure of it may be the KV
    comparison's.
  - **The KV half** (`gpu_planted_replay_rewrites_last_row`, (k)): every
    generation samples the per-token path's tokens, so the test requires that no
    failure names a token, and the row one step later differs from the per-token
    path's at every token of the vocabulary or all but one. This is the
    comparison that reads what a chain left behind, failing on its own.
- **The stop and the chained count**, by throwaway edits on ERNIE-4.5-0.3B
  (`gpu_seam_tolerance_ernie`, `Vulkan0`; restored after). With the runner
  returning after its first chained submit, the chained generation fails as "8
  tokens and did not stop itself, where the per-token path is still generating
  after 33", and its chained count as 8 where the chain implies 32 of 33. With
  the chained generation told to ignore end tokens, it fails on the chat prompt
  as "33 tokens and did not stop itself, where the per-token path reaches its
  end token after 9".
- **What the token rule sees of a bug in the model**
  (`gpu_planted_greedy_tokens`): the planted copy generates on the GPU and its
  tokens are held to the clean copy's CPU reference. The verdicts are the same
  on both GPUs:

  | Bug                                     | Int8 (`gpu`)           | Float (`gpu-float`)    |
  | --------------------------------------- | ---------------------- | ---------------------- |
  | (c) RMSNorm epsilon x10                 | **missed**             | **missed**             |
  | (d) every block of one `ffn_down` x1.01 | **missed**             | **missed**             |
  | (e) gate and up swapped                 | caught, several tokens | caught, several tokens |
  | (f) one attention residual dropped      | caught, one token      | caught, one token      |
  | (g) rope pairs the wrong elements       | caught, several tokens | caught, several tokens |
  | (h) Q4_K nibbles swapped                | caught, one token      | caught, one token      |

  A cell counts the tokens past the near tie on the first prompt that fails,
  which is all the test prints; (f) in Int8 shows on the chat prompt alone. The
  test requires (e) and (g) in both classes and prints the rest. It also
  requires that no bug of the model makes a generation part from the per-token
  path: such a bug is in all three generations alike, so the bit-for-bit half
  says nothing about the model and everything about how its decode is driven.
  (a) and (b) are left out: their graphs opt out of the replayed decode, and the
  check then stops at the missing recording.

- **The rule itself** has a unit test in the normal job
  (`token_score_counts_other_choices_above_the_near_tie`).

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
  `exp_probs_b.bias` zeroed (B174's case). On the synthetic MoE, (d) scales one
  layer's routed down bank (`ffn_down_exps`) and (e) swaps every layer's routed
  gate and up banks.
- **Bugs in how a decode recording is driven**, neither in a graph nor in a
  weight, for the greedy check ("The greedy check"); a forced run executes each
  recording once and cannot carry them. (j) Every execution of a recording after
  its first within a generation is fed the token before the one it should be
  (`Planted::ReplayFeedsStaleToken`: a replayed step's token, and the first
  token of every chained submit after the first). (k) A generation that executed
  its recording more than once leaves the first layer's V row of the last
  position it wrote with every sign flipped (`Planted::ReplayRewritesLastRow`):
  no sampled token changes, only what is left in the KV cache.

**What each test asserts.** A clean control runs first, twice: it must pass the
gate, and its two runs must agree bit for bit. Every planted run must then
differ from the clean run in the decode part of every prompt and configuration:
its decode rows, and on a routed-expert model the experts its own routers
selected at those positions (the bug reached decode). Where the gate must catch
the bug, the run must fail it with a failure that names a metric, a non-finite
logit or the routing at a prompt. Where a miss is on record ((c), (d) and (d1)
in Int8 and in gemma-3's Float smoke row, (d1) in every other row, (d) in the
synthetic MoE's), the ratio is printed and nothing more is asserted, so a
tighter table later cannot break the test. Each bug and run prints a `PLANTED`
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
nibbles, so no (h)) in Float; `gpu_planted_router_bias` runs (i), and (d) and
(e) in the routed expert banks, on a synthetic ERNIE-4.5 MoE (eight f32 experts,
two used, a softmax router with a selection bias) against its own row. No bug
has been planted in a real MoE model.

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

**The synthetic MoE**, against its own row (`synthetic-moe`, set the same day
from forced-routing runs; "The table" has it), measured the same day on the same
machine and drivers (`planted_bugs_fail_the_gate_on_the_cpu` and
`gpu_planted_router_bias`, configuration `gpu`). The model takes no batched
prefill, so only its decode rows are scored on logits; its routing is compared
at every position. Cells are the worst steady KL over its bound (mean KL in
brackets), CPU reference / `Vulkan0` / `Vulkan1`:

| Bug                                                       | Synthetic MoE Float                                                                                                             |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| clean control (passes)                                    | 0 / 0.22 (0.49) / 0.22 (0.49); on both GPUs one selection of the long prompt's 422 differs, at a 2.14e-4 margin                 |
| (i) router ignores its bias                               | caught on the routing alone: 42 of 86, 70 of 130 and 216 of 422 selections differ, at margins up to 0.37; KL as the clean run's |
| (d) every value of `blk.1.ffn_down_exps` x1.01            | **missed** on the CPU: 0.35 (mean 0.77); over the mean bound on both GPUs: 0.36 (mean 1.36)                                     |
| (e) `ffn_gate_exps` and `ffn_up_exps` swapped, each layer | caught: 1.35e4 / 1.35e4 / 1.35e4                                                                                                |

- **(i) is caught by the routing comparison and by nothing else.** Forced along
  the reference's selections, the model with its bias zeroed computes the clean
  model's function: on the CPU its logits are the clean run's bit for bit (KL
  0), on the GPUs they are the clean GPU run's. What differs is what its routers
  would select, at half the selections and at reference margins a thousand times
  the near tie, on the CPU and on both GPUs. The test's "the bug reached decode"
  check therefore reads a run's own selections along with its logit rows. This
  is B174's case as a planted weight bug; a backend whose selection kernel
  ignored the bias would show the same way, since the readback runs that kernel.
- **(d) in the routed experts is a recorded miss.** A 1% scale of one layer's
  whole down bank leaves the shared expert beside it unscaled; on the CPU, where
  the clean distance is 0, it measures 0.35 of the KL bound and 0.77 of the mean
  bound. On the GPUs the device's own distance adds to it and the mean goes over
  (1.36); nothing is asserted either way.
- **(e) also moves the routing**: with layer 0's experts wrong, layer 1's router
  sees another hidden state and selects otherwise at 117 of the long prompt's
  422 selections on the CPU, every one over the near tie, and at 118 on each
  GPU, 117 of them over it (margins up to 1.14). A routing failure beside a
  logit failure is a consequence; a routing failure alone, as in (i), is a
  router.

**Why the routing is forced: a router near tie** (2026-11-23, both GPUs).
Before, the same clean run was only forced along the reference's tokens, and
landed outside the synthetic f32 llama's row it then borrowed: worst KL at most
1.03e-4 on prompts 0 and 1, and on prompt 2 2.87e-3 at position 179, the first
decode row after the 180-token prompt. The outlier does not follow the end of
the prompt (ended at 100, 150, 170, 178 or 179 tokens, position 179 still
measures 2.87e-3, on the replayed and the static decode alike), and scoring the
earlier rows shows where it comes from: position 107 measures 0.371, and
positions 110, 137, 138, 143 and 173 are raised with it. At position 107 the CPU
reference's layer-0 router separates its second and third expert by 2.14e-4 of
selection probability, the smallest margin of the 422 router calls of the
sequence (median 6.3e-2), and flipping that one pick in the CPU reference
reproduces the GPU's profile. The readback now shows it directly
(`INFR_TOL_OWN_ROUTING=1`, which leaves the GPU on its own routers as before):
both GPUs select another expert at position 107, layer 0, at that 2.14e-4
margin, and then at position 107, layer 1, at a margin of 9.8e-2, because layer
1's router now sees a different hidden state. That second difference is what an
unforced comparison cannot tell from a bug. Forced along the reference's routing
the same runs measure a worst KL of 6.3e-5 on prompt 2 and 1.03e-4 over all
three, and the one difference left is the near tie itself.

**The same on a real model, ERNIE-4.5-21B-A3B Q4_K_M** (`measure_gpu_tolerance`,
key `ernie-21b-moe`, the three prompts, native lanes, the same day; a
measurement: the model has no row and no gate test). Unlike the synthetic model
it has quantized expert banks and takes the batched prefill, so its prefill rows
are scored and, in the shipped configurations, the supplied selections go
through the batched expert GEMMs. Each cell is the worst KL from position 2 on /
in the first two positions, then how many of the run's 8802 own selections
differ from the reference's and the widest margin among them (in selection
score; a uniform share among the model's 64 experts is 1.6e-2):

| Device, configurations                                       | Routing forced                              | Routing not forced (`INFR_TOL_OWN_ROUTING`) |
| ------------------------------------------------------------ | ------------------------------------------- | ------------------------------------------- |
| `Vulkan0`, `gpu-float`, `gpu-float-ub32`, `gpu-float-static` | 6.1e-6 / 1.1e-6; 12 to 18 differ (2.7e-5)   | 6.8e-3 / 1.1e-6; 56 to 61 differ (4.6e-3)   |
| `Vulkan0`, `gpu`, `gpu-ub32`                                 | 3.6e-3 / 1.2e-3; 242 to 247 differ (1.7e-3) | 1.4e-2 / 5.3e-3; 417 to 429 differ (6.5e-3) |
| `Vulkan1`, `gpu-float`, `gpu-float-ub32`, `gpu-float-static` | 3.8e-6 / 1.5e-7; 8 to 13 differ (1.1e-5)    | not run                                     |
| `Vulkan1`, `gpu`, `gpu-ub32`                                 | 1.3e-2 / 1.7e-3; 425 to 434 differ (3.2e-2) | not run                                     |

Forcing the routing takes the int8-free runs from 6.8e-3 to 6.1e-6, three orders
of magnitude: unforced, prompt 1's decode row at position 54 measures 6.83e-3
with four of that position's routers selecting otherwise (layers 14, 22, 23 and
24, margins up to 2.6e-4), and forced the same row's phase measures 2.4e-6 with
one selection of the prompt differing, at a margin of 3.8e-6. The int8 runs keep
their own activation quantization, and it reaches the routers: on the integrated
GPU a selection differs at a margin of 3.2e-2, twice a uniform share, so a
routing near tie set from the int8 runs of this model would hold almost nothing.
What its rows should be is open (B207). The whole measurement took 64 s on
`Vulkan0` and 343 s on `Vulkan1`.

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
configuration measured. A routed-expert model's row is set from runs forced
along the reference's routing, and carries a routing near tie
(`Entry::route_near_tie`) at 1.5 times the widest reference margin at which a
legitimate run's own router selected other experts. The worst legitimate KL of a
row (from position 2 on) moves a long way between the paths measured — the RX
7900 XTX native, its wave64 path, its generic tuning (rows off) and the
integrated GPU: within about 4x for most rows, but 9x to 22x for gemma-3's Float
and Int8 rows and LFM3's Float and Int8 rows, and 83x for Qwen3.5's Float row
(whose integrated-GPU figure scores one prefill row per chunk). A row's bound is
its worst path's, so it is loose for the paths far below that. The margin of 2
on top is headroom for a device or driver not measured here, kept small because
the planted bugs sit only 3 to 4.5 times above the Float tolerance (6.4x on KL
and 9.2x on mean KL above the legitimate worst).

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

**The synthetic MoE's row was set 2026-11-23**, the table's first routed-expert
row, from `measure_gpu_tolerance` on Linux 7.3.5-arch1-1, Mesa RADV 26.3.1,
headless, on both GPUs, every run forced along the reference's routing:
configurations `gpu` (replayed decode) and `gpu-float-static`, native lanes,
wave64 forced and tuning rows off at the prompts' own ends, and native lanes (on
`Vulkan0` also wave64 forced) with every prompt ended after 4 tokens
(`INFR_TOL_SPLIT=4`). The model takes no batched prefill, so a moved prompt end
changes no kernel and only scores more of the same decode rows (620 against
192); those count as legitimate. With native lanes both GPUs give the same
values to the digits given.

| Limit            | Worst legitimate value                                    | Tolerance | Smallest planted bug over it                                                      |
| ---------------- | --------------------------------------------------------- | --------- | --------------------------------------------------------------------------------- |
| KL max           | 2.31e-4 (prompt 2 ended after 4 tokens, position 90)      | 4.7e-4    | (e) at 1.35e4 times the bound; (d) does not cross (0.35 on the CPU)               |
| mean KL          | 1.52e-5 (prompt 2, its own end, wave64 forced, `Vulkan0`) | 3.1e-5    | (d) on the GPUs only, 1.36 times the bound, a recorded miss (0.77 on the CPU)     |
| top-k cosine     | 0.999750 (prompt 2 ended after 4 tokens)                  | 0.9995    | not looked at                                                                     |
| logit deviation  | 2.89e-3 (prompt 1)                                        | 4.4e-3    | not looked at                                                                     |
| top-1, rank      | no miss, nothing ranked above the reference's choice      | 0.25, 1   | not looked at                                                                     |
| routing near tie | one selection, at a margin of 2.14e-4 (prompt 2, 107/0)   | 3.3e-4    | (i): 42 of prompt 0's 86 selections, the narrowest of them at a margin of 1.06e-3 |

The cosine bound is the rule applied to the value as the log prints it, to six
places (twice its distance from 1 gives 0.99950), cut to four places as
gemma-3's rows are (0.997996 to 0.9979, 0.97904 to 0.9790).

That the model takes no batched prefill is a property the test states of the
case (`Case::batched_prefill` in `infr_llama::tolerance::gate`), not one read
off the run: a case so marked must produce no prefill row, and only then is the
vacuous count of scored prefill rows dropped; any other case that produces none
fails the gate by name (`Reference::check_prefill`).

Under the earlier comparison (tokens forced, routing not) the same clean runs
measured a worst KL of 2.87e-3 at the prompts' own ends and 0.371 with the
earlier rows scored, so no row could have been set that a wrong expert would not
also pass. The row is from RADV alone: Windows has run none of this. It is a toy
model with f32 experts: it runs the per-token expert kernels (`native_idm_f32`)
and never the batched expert GEMMs or an int8 path, which only a real MoE model
reaches (below, and B207).

For every row but Qwen3-0.6B's, gemma-3's and the synthetic MoE's, the gap is
unmeasured: the tolerances are set from legitimate runs only.

**No ratchet.** A row may be loosened only by its owner, with the measurement
that needs it (device, driver, configuration, the run's printed values) and the
planted bugs above re-run against the new value, keeping a stated gap; never to
make a failing run pass without that evidence. A device that lands outside a row
is a finding first. The same rule is in `TABLE`'s doc comment.

## Running it

The evaluation is library code, `infr_llama::tolerance::gate` and `greedy`
(`crates/infr-llama/src/tolerance/`): the prompts, the configurations
(`GPU_CONFIGS`), which of them a model runs and against which class's row
(`runs_for`, read off the rows the table has for the model), the scoring, and
`gate::check`, which runs the forced comparison and then the greedy check. Two
things call `gate::check`: the tests below, and the command `infr tune check`
("The command" further down). The test support code (`tests/support/forced.rs`)
keeps what only tests need: opening a case by key or from a synthetic or planted
file, the measurement, the watchdog and the skips.

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
short for a batched prefill). On a routed-expert model each comparison against
the reference also prints a `ROUTE` line: how many of the run's own selections
differ from the reference's, the widest margin among them, and each one as
`position/layer:margin`, which is what a row's routing near tie is set from.
`INFR_TOL_OWN_ROUTING=1` leaves the Vulkan runs on their own routers'
selections, the comparison as it was before the routing was forced: a distance
that goes away without it was a routing difference, one that stays is
arithmetic. One configuration is for diagnosis only (`DIAGNOSIS_CONFIGS`): no
gate scores it, and `measure_gpu_tolerance` runs it only when `INFR_TOL_ONLY`
names it. `gpu-no-dp4a-kq` is the shipped configuration without the dp4a K-quant
prefill GEMM. The three `gpu-float-nocoopmat*` configurations (the Float ones
with the whole cooperative-matrix tier off) were diagnosis too until 2026-11-23;
they are gated now, for gemma-3, and a default measurement runs them for every
model.

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

**Results with the routing forced, 2026-11-23, same machine and drivers,
headless**, no environment knob but `INFR_DEV`: the two new tests,
`gpu_seam_tolerance_synthetic_moe` and
`gpu_seam_tolerance_own_routing_is_the_selection_it_runs`, pass on both GPUs in
under a second each, and with them the whole set: all 13 `gpu_seam_tolerance_*`
tests on `Vulkan0` in 319 s on the final code, and on `Vulkan1` in 655 s on the
code before a review round (which changed the routing comparison, the prefill
check of `gate()` and the new row's cosine bound, and moved code; the two new
tests and `gpu_planted_router_bias` were run again on both GPUs after it, and
pass), against a table in which no row but the new one changed. The production
MoE tests of `cpu_backend` were run on `Vulkan0` as the check that nothing
outside a forced run moved, and pass: the exact hash of
`gpu_seam_golden_qwen3moe`, `gpu_seam_matches_cpu_ernie4_5_moe` and its
`_tokens` form, `gpu_seam_ernie4_5_moe_decode_replays_from_one_recording`, and
the two paged comparisons (`gpu_seam_paged_ernie4_5_moe_matches_resident`,
`gpu_seam_paged_moe_matches_resident_and_cpu`). Not run on `Vulkan1`. The four
`gpu_planted_*` tests passed on both GPUs before that review round (610 s and
1029 s), with the dense models' verdicts and ratios as the tables above have
them; the dense three were not run again after it.

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

**Results with the greedy check in every case, 2026-11-23, same machine and
drivers, headless**, on the check as it stands (the chat prompt, the stop rule,
`GREEDY_STEPS`). The GPU gate on `Vulkan0` (the filter above; no knob but
`INFR_DEV`, a model that is not cached printing `skip:`): 709 tests, 709 passed,
659 s. The 13 `gpu_seam_tolerance_*` tests alone: 430 s on `Vulkan0` (ERNIE both
files 51 s, gemma-3 115 s, LFM3 both files 110 s, Qwen3 74 s, Qwen3.5 70 s) and
1070 s on `Vulkan1` (94 s, 281 s, 385 s, 157 s, 139 s). The seven
`gpu_planted_*` tests, the three of the greedy check among them: 838 s on
`Vulkan0`, 1602 s on `Vulkan1`, all passing (the misses on record are printed,
not asserted, and were not compared with the tables under "Mutations" again).
The full gate filter was not run again on `Vulkan1` after the check changed; it
ran there with the check as first written (32-token generations, no chat prompt,
no stop rule): 649 tests, 649 passed, 1157 s, with the integrated GPU's
exclusion. `gpu_seam_int8_prefill_tracks_cpu` printed its skip line on `Vulkan0`
(no int8 cooperative-matrix prefill GEMM under RADV).

The red-path tests scale `blk.1.ffn_down.weight` of the synthetic model by 1.02.
On the CPU a 1.01 scale already trips the gate and 1.008 and below pass
(measured in review), so 1.02 is about twice the threshold; the green control is
the reference against itself, at a distance of exactly 0. The int8-free check
reads the kernels a run BUILT, which are built lazily when first recorded: a
superset of what ran, so its negative check is the conservative one; its marker
list is typed by hand (B207).

## The command: `infr tune check`

`infr tune check <model>` runs the gate on one model from the shipped binary,
with no test suite: for a tuning parameter someone is changing, a new driver, or
a device no table row was written for.

```text
infr tune check unsloth/Qwen3-0.6B-GGUF:Q4_K_M --dev Vulkan0
INFR_TUNE=gemv.kq_fma=true infr tune check unsloth/Qwen3-0.6B-GGUF:Q4_K_M
infr tune check model.gguf --set kernels.vulkan.tune.attn_split.decode_min_chunk=16 --json
```

**What it runs** is `gate::check`, the function the `gpu_seam_tolerance_*` tests
call, on the runs `gate::runs_for` reads off the model's rows: for a quantized
model the shipped kernels (held to its Int8 row) and the same with every
int8-activation kernel off (its Float row, the check that can see a small
error), at both prefill chunk heights and on the static decode, plus whatever
other rows the model has; then the greedy check on the configurations it covers.
The prompts, the number of decode steps and the limits are the tests'.

**What is under test** is the device under the configuration the invocation
resolves: the tuning table's rows for the device, the tuning file, `INFR_TUNE`,
`--set` and a config file all apply, in every run, since each gate configuration
is an edit on top of that configuration. The CPU reference is not: it is loaded
under the shipped defaults, so a setting that would change the reference's own
arithmetic (a KV format) cannot move it along with the device. Such a setting
then shows as a distance, which is the honest answer: the rows are of the
shipped KV format. The prompts are tokenized with the GGUF's own vocabulary, as
in the tests, never a `tokenizer.json` beside the file.

**A configuration's edit writes over the settings it is about**: the int8 kernel
switches in the `gpu-float*` runs, the prefill chunk height in the `*-ub32`
runs, the KV format in `gpu-kv-q8-float`, the replay switch in the `*-static`
runs. Where that replaces something the invocation set, the run is not of what
was set, and the report says so under that configuration: one line per setting,
with the value set and the value the run ran with (`overridden` in the JSON).
The list is read by walking every config path before and after the edit
(`gate::overridden`), so it follows the edits; a setting given at its default
value is not listed, since nothing distinguishes it from one not given.

**The model** is required, given as `infr run` takes one. Only a model with rows
can be checked: any other is an error that names the file and the models that do
have rows, exit status 1, and scores nothing. There is no fallback to another
model's row and no mode that prints distances without a verdict (B220). The
refusal comes before anything is downloaded wherever the reference itself gives
the file name: a path, a model already in the cache, an explicit `.gguf`
selector. A quant selector (`org/repo:Q4_K_M`) of a repository not in the cache
names no file until the hub is asked, so that one is fetched first and refused
after (B234).

**Rows are found by file name and by nothing else.** A renamed file, or another
quantizer's file under the same name, is scored against rows that were measured
on a different file. The report names the row it matched and how (`row`), and
what the file says of itself (`file`: its path, `general.architecture`,
`general.name`, whether its weights are quantized, its tensor count and their
bytes), on stdout, in the JSON and in a log line, so a reader can see a
mismatch. One mismatch is refused, the only one the rows themselves can show: a
model with an Int8 row was measured on a quantized file and one without on a
float file, and a file of the other kind exits 1 unscored (`gate::check_file`).
Nothing ties a row to its file's contents beyond that (B234).

**`--against cpu-reference`** is the default and the only value; another is a
usage error. The op-level families that plan 11 first named for this command are
not in it: they are test binaries of `infr-vulkan`, which a shipped binary does
not carry.

**Output.** stdout has the row and the file, then per configuration and phase
each limit with the worst value measured over the prompts, its bound and `ok` or
`FAIL`; then the greedy check's token counts; then, when anything failed, the
evaluation's own description of each failure (the position, the prompt); then
what the verdict does not cover on this device and model; then the verdict. The
verdict is the evaluation's failure list, the one the tests assert empty, so a
failure with no line of its own (a row that could not be scored, a chained
decode that left other KV behind) still fails the command and is listed. **What
was not exercised** is part of the verdict block, not a log line: on a device
that runs one decode step per submit (an integrated GPU) no chained decode
exists to compare, and a PASS there says so right above itself; a model that
takes no batched prefill, and a routed-expert model's greedy tokens, are named
the same way (`not_exercised` in the JSON). The log goes to stderr;
`RUST_LOG=debug` adds every prompt's values.

```text
row:      Qwen3-0.6B-Q4_K_M.gguf (matched by file name only)
file:     ~/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf
          architecture qwen3, general.name "Qwen3-0.6B", quantized weights, 310 tensors, 390753280 bytes of them
device:   Vulkan0 AMD Radeon RX 7900 XTX (RADV NAVI31) (tune:af2eda63)
against:  cpu-reference

gpu (Int8)
  phase    metric             measured  bound           verdict
  first    kl_max             3.168e-6  <= 0.070000     ok
  ...
gpu-float (Float)
  phase    metric             measured  bound           verdict
  ...
  decode   kl_max             1.602e-5  <= 8.500e-5     ok
  decode   kl_mean            2.788e-6  <= 7.300e-6     ok
  decode   topk_cos_min       0.999986  >= 0.999900     ok
  decode   logit_dev_max      6.985e-4  <= 0.001900     ok
  decode   top1_misses               0  <= 0            ok
  decode   rank_max                  1  <= 1            ok
  decode   top1_scored              90  >= 16           ok
...
greedy decode, gpu-float (Float)
  phase    metric             measured  bound           verdict
  tokens   token_misses              0  <= 0            ok
  tokens   token_scored            123  >= 16           ok

verdict:  PASS
```

`--json` prints one document instead: `against`, `model`, `row` (`model`,
`matched_by`), `file` (`path`, `architecture`, `name`, `quantized`, `tensors`,
`tensor_bytes`), `device`, `tune_hash`, `pass`, `runs` and `greedy` (each an
array of `config`, `class`, `overridden` (`path`, `set`, `ran`) and `checks`,
and each check `phase`, `metric`, `measured`, `limit` (`at_most` or `at_least`),
`bound` and `pass`), `failures` and `not_exercised`. The metric names are
`kl_max`, `kl_mean`, `topk_cos_min`, `logit_dev_max`, `top1_misses`, `rank_max`,
`top1_scored`, `route_misses`, `route_scored`, `token_misses` and
`token_scored`.

**Exit status:** 0 within tolerance, 3 outside it, 1 when the check could not
run (no row, a file of the wrong kind for its rows, no Vulkan device, a failed
load or run), 2 on a usage error. 101 is a panic: a bug in infr, not a verdict.
The mapping is one function (`exit_status` in `crates/infr-cli/src/main.rs`),
which a test holds to these values.

**Recorded runs, 2026-11-23** (Linux 7.3.5-arch1-1, RADV, headless; the release
binary; Qwen3-0.6B Q4_K_M, whose six configurations and three greedy
configurations make 126 lines). "Closest" is the largest share of its bound any
KL, cosine or logit-deviation line reached (the cosine by its distance from 1).

| Device    | Forced                                                     | Verdict, exit | Lines `ok` / `FAIL` | Closest to a bound                            |
| --------- | ---------------------------------------------------------- | ------------- | ------------------- | --------------------------------------------- |
| `Vulkan0` | nothing                                                    | PASS, 0       | 126 / 0             | 0.49: `gpu-float` prefill mean KL             |
| `Vulkan1` | nothing                                                    | PASS, 0       | 126 / 0             | 0.66: `gpu` (Int8) first-rows logit deviation |
| `Vulkan0` | `--set kernels.vulkan.tune.attn_split.decode_min_chunk=16` | PASS, 0       | 126 / 0             | 0.53: `gpu-float-ub32` decode logit deviation |
| `Vulkan0` | `INFR_TUNE=gemv.kq_fma=true`                               | PASS, 0       | 126 / 0             | 0.57: `gpu-float-ub32` decode logit deviation |
| `Vulkan0` | `--set device.ubatch=64`                                   | PASS, 0       | 126 / 0             | 0.49: `gpu-float` prefill mean KL             |
| `Vulkan0` | `--set kv.type_v=q4_0`                                     | FAIL, 3       | 60 / 66             | over: an int8-free run's worst KL is 0.134    |

The two tuning rows are the case this command was wanted for (B227): each
parameter alone moves the values and stays inside the rows; they were taken
before the report gained its row, file, override and not-exercised lines, the
other four after.

The `Vulkan1` pass ends with what it does not cover, above the verdict:

```text
not exercised (the verdict says nothing of these):
  Qwen3-0.6B-Q4_K_M.gguf: this device runs one decode step per submit; its greedy decode is one replay per token, and no chained run is compared
verdict:  PASS
```

With `--set device.ubatch=64` the two runs whose configuration sets the chunk
height say that they did not run at 64, and the others ran at it:

```text
gpu-ub32 (Int8)
  ran with device.ubatch = 32, not the 64 that was set: this configuration sets it
```

The last row is how the command was shown to fail: a 4-bit V cache on the device
against the reference's shipped f16 one puts the int8-free runs' worst KL at
0.134, against a bound of 8.5e-5, and the Int8 smoke rows over too; the greedy
check fails as well, with no line of its own (with that cache each generation of
the `gpu` and `gpu-float` configurations made no decode recording, and the
failure list says so for every prompt). `gpu-kv-q8-float` still passes there,
and says why: `ran with kv.type_v = Q8_0, not the Q4_0 that was set`. A model
with no row exits 1 with the error naming it, whether it is in the cache
(`unsloth/Qwen3-1.7B-GGUF:Q4_K_M`) or named by an explicit file of a repository
that is not (`…:Some-Model-Q4_K_M.gguf`, refused with nothing fetched);
`--against gpu` exits 2, `--dev cpu` exits 1. On `Vulkan0` a run takes about 75
s (the forced runs 49.4 s, the greedy check 25.1 s), on `Vulkan1` about 160 s
(81.4 s and 76.6 s).

**What it does not do.** It has no watchdog and no host-memory check (the tests
have both: a hang is the user's to interrupt, and on an integrated GPU the model
must fit beside the reference in host memory). It checks one device per run. It
runs on Vulkan only. What it leaves otherwise is what the gate leaves (B207).

## The exact and cosine GPU tests: what G4 did with each

Before plan 11's G4 the GPU side of `crates/infr-llama/tests/cpu_backend.rs` was
held by exact hashes of generated text, exact tokens against the CPU reference
and whole-vocabulary logit cosines against the CPU. This is every one of them,
the path it exercises, whether the tolerance gate exercises that path, and what
was done (2026-11-23). "Production greedy" is `generate_dense_vulkan`: the
batched prefill where the model has one, then the decode the model takes
(chained on a device that chains, else one replay of a single recording per
token, else the static per-token path). A test is retired only where the gate
runs the same path on the same model file; everything else is in place,
unchanged, with what it would take.

**Retired**, each now covered by the model's `gpu_seam_tolerance_*` test: its
forced runs score the batched prefill's rows and the per-token decode's against
the table, and its greedy check holds the chained and the replayed decode to
that per-token path bit for bit and the greedy tokens to the CPU reference
outside near ties.

| Test                                   | Model file            | Asserted                                  | Path                                                                               | Now covered by                                                                                                                                                                                           |
| -------------------------------------- | --------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gpu_seam_golden_qwen3`                | Qwen3-0.6B Q4_K_M     | hash of the greedy text, two chat prompts | production greedy on the shipped int8 kernels                                      | `gpu_seam_tolerance_qwen3`: `gpu`, `gpu-ub32` (Int8), `gpu-float*` (Float), `gpu-kv-q8-float` (KvQ8); greedy check on `gpu`, `gpu-float` and `gpu-kv-q8-float`                                           |
| `gpu_seam_flash_matches_cpu`           | Qwen3-0.6B Q4_K_M     | a common prefix with the CPU reference    | flash-attention batched prefill of a prompt over 64 tokens, then the greedy decode | the same test: its long prompt prefills in one default chunk through `attn_flash_warp` and `attn_flash_combine` (the kernel list of a `gpu` run, `INFR_TOL_ROWS=1`), and every one of its rows is scored |
| `gpu_seam_matches_cpu_ernie4_5`        | ERNIE-4.5-0.3B F16    | exact tokens against the CPU reference    | production greedy, float weights                                                   | `gpu_seam_tolerance_ernie`, the F16 case (Float), with its greedy check                                                                                                                                  |
| `gpu_seam_matches_cpu_ernie4_5_q4_k_m` | ERNIE-4.5-0.3B Q4_K_M | exact tokens against the CPU reference    | production greedy, int8 kernels                                                    | `gpu_seam_tolerance_ernie`, the Q4_K_M case (Int8 and Float), with its greedy check                                                                                                                      |
| `unified_qwen35_gpu_seam_matches_cpu`  | Qwen3.5-0.8B Q4_K_M   | exact tokens against the CPU reference    | production greedy through the DeltaNet layers and the gated attention              | `gpu_seam_tolerance_qwen35` (Int8 and Float), with its greedy check                                                                                                                                      |
| `gpu_seam_matches_cpu_lfm3`            | LFM3-1.7B F16         | exact tokens against the CPU reference    | production greedy; the short convolution's history through prefill and decode      | `gpu_seam_tolerance_lfm3`, the F16 case (Float), with its greedy check                                                                                                                                   |
| `gpu_seam_matches_cpu_lfm3_q4_k_m`     | LFM3-1.7B Q4_K_M      | exact tokens against the CPU reference    | the same on the quantized kernels                                                  | `gpu_seam_tolerance_lfm3`, the Q4_K_M case (Int8 and Float), with its greedy check                                                                                                                       |

What was lost with them, stated plainly:

- **The exact prompts and lengths.** The golden generated 32 and 48 tokens after
  two chat prompts; the greedy check generates 33 after three plain prompts and
  one chat prompt. No generation is 48 tokens long any more.
- **The text hash end to end.** The golden hashed detokenized text, so it also
  held the tokenizer's decode of the GPU's tokens; the greedy check compares
  token ids. The CPU goldens still hash detokenized text of the same models.
- **Exactness against the CPU on a chosen prompt.** The exact-token tests failed
  on any difference from the reference; the token rule lets a near tie pass,
  which is the point, and catches far less than a hash through tokens alone
  ("The greedy check" has what it misses).
- **Red paths shown on the old tests** (a Vulkan rope base cut 1000×, a halved
  attention scale, LFM3's history rebuild) have not been shown again on the gate
  cases.

What is kept that a first look might count as lost: special tokens through the
device's embedding gather (the chat prompt of the greedy check), a generation
that ends on its end token, and the hand-off from a chained submit to a single
replay (the 9-token ERNIE test drove it; `GREEDY_STEPS` does now). Every row
these replacements rest on was set on Windows (`TABLE`'s provenance), and the
greedy check's bit-for-bit half needs no row; the greedy check itself has run
under RADV only.

In the tables below, "here" is the Linux machine's `Vulkan0` gate of 2026-11-23,
run with every test's output: **ran** means the test printed no `skip:` line and
passed, **skipped** that it printed one (the model file is not in the cache
unless another reason is given) and so counted as passed without running.

**Kept, an exact hash:**

| Test                       | Model file           | Path                                                                                                        | Here                                                                                                                                      | Why it stays                                                                                                                                                                              |
| -------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gpu_seam_golden_qwen3moe` | Qwen3-30B-A3B Q4_K_M | production greedy of a real MoE model: batched GPU-routed prefill on the int8 expert GEMMs, then the decode | ran: 3.2 s for its 8 tokens. Both of its early returns print `skip:` and neither did; past them it asserts the hash of the generated text | No real MoE model has a row, and what its Int8 routing check should be is undecided (B207). It is the one text hash left, and a tuning change that moves its tokens still has to bless it |

**Kept, exact tokens against the CPU** (`seam_vulkan_matches_cpu` unless noted),
each on a prompt picked for clear margins. None but gemma-3 has a row, so
retiring one is a measurement on both GPUs with `measure_gpu_tolerance`, a row
by the table's rule, the planted bugs re-run, and a `gpu_seam_tolerance_*` case;
the greedy check then comes with the case.

| Test                                               | Model file                   | Path beyond production greedy                                                    | Here    | Note                                                                                                                                                          |
| -------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gpu_seam_matches_cpu_gemma3`                      | gemma-3-1b Q4_K_M            | sliding-window layers, dual rope, sandwich norms                                 | ran     | The one kept although its model has rows and `gpu_seam_tolerance_gemma3` runs its path: those rows were re-set under RADV and Windows has not run them (B207) |
| `gpu_seam_matches_cpu_gemma3_q2k_iq4nl`            | gemma-3-1b Q2_K              | the IQ4_NL GEMM family and its decode GEMV                                       | ran     | A common prefix, not equality                                                                                                                                 |
| `gpu_seam_matches_cpu_qwen2`                       | Qwen2.5-0.5B Q4_K_M          | q/k/v biases                                                                     | ran     | Measured on Windows, no row: its float prefill is twenty times further than the other Qwens' (B207)                                                           |
| `gpu_seam_matches_cpu_qwen3_iq4xs`                 | Qwen3-0.6B IQ4_XS            | the IQ4_XS kernels                                                               | skipped |                                                                                                                                                               |
| `gpu_seam_matches_cpu_qwen3_q2k`                   | Qwen3-0.6B Q2_K              | the Q2_K and Q3_K kernels                                                        | skipped |                                                                                                                                                               |
| `gpu_seam_matches_cpu_qwen3_q8_0_i8coopmat`        | Qwen3-0.6B Q8_0              | the int8 cooperative-matrix Q8_0 prefill GEMM (`kernels.vulkan.i8_coopmat`)      | skipped | An opt-in kernel                                                                                                                                              |
| `gpu_seam_bf16_matches_cpu`                        | Qwen3-0.6B BF16              | bf16 projections converted to f16                                                | skipped | Against the production CPU, not the reference                                                                                                                 |
| `gpu_seam_matches_cpu_llama`                       | Llama-3.2-1B Q8_0            | interleaved rope, its replayed f16 rope                                          | skipped | The cache here holds the Q4_K_M file only, which this test does not load; the Q4_K_M file was measured on Windows (the three-way table) and has no row        |
| `gpu_seam_matches_cpu_smollm3`                     | SmolLM3-3B Q4_K_M            | NoPE layers; the static decode, which records nothing                            | ran     | Measured on Windows, no row                                                                                                                                   |
| `gpu_seam_matches_cpu_gemma4`                      | gemma-4-12b Q4_K_M           | heterogeneous head widths                                                        | ran     |                                                                                                                                                               |
| `gpu_seam_matches_cpu_gemma4_e2b`                  | gemma-4-E2B Q4_K_M           | per-layer token embeddings (no chain)                                            | ran     | Its prompt takes no batched prefill, so a forced run would score its decode rows only                                                                         |
| `gpu_seam_matches_cpu_ernie4_5_moe_tokens`         | ERNIE-4.5-21B-A3B Q4_K_M     | a real MoE model on its own routers                                              | ran     | Measured forced along the reference's routing, no row (B207)                                                                                                  |
| `gpu_seam_paged_moe_matches_resident_and_cpu`      | Qwen3-30B-A3B Q4_K_M         | paged expert cache, one-row prefill chunks, against the resident run and the CPU | ran     | A paged MoE model cannot be forced (B207)                                                                                                                     |
| `gpu_seam_paged_moe_matches_scout_oracle`          | Llama-4-Scout                | paged expert cache with real eviction, against a recorded prefix                 | skipped |                                                                                                                                                               |
| `gpu_seam_dense_stream_matches_resident_and_cpu`   | Qwen3-1.7B Q4_K_M            | streamed dense weights, against the resident run and the CPU                     | ran     | The GPU-against-GPU half needs no row                                                                                                                         |
| `gpu_seam_dense_stream_matches_resident_qwen3_14b` | Qwen3-14B Q8_0               | the same on the fused q/k/v form                                                 | not run | The gate filter leaves it out (`not test(qwen3_14b)`), and the file is not cached here                                                                        |
| `mtp_head_cpu_vulkan_parity`                       | Qwen3.5-4B (UD-Q4_K_XL, MTP) | the MTP head's draft ids, CPU against Vulkan                                     | ran     | No forced run drives the head                                                                                                                                 |

**Kept, a whole-vocabulary logit cosine against the CPU** (B60: such a cosine
cannot separate a match from a mismatch; each also asserts something else):

| Test                                           | Model file               | Path                                                                       | Here    | Why it stays                                                                                                                                                                       |
| ---------------------------------------------- | ------------------------ | -------------------------------------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gpu_seam_kv_q8_tracks_cpu`                    | Qwen3-0.6B Q4_K_M        | the q8_0 KV cache, coupled and on one side only                            | ran     | Only the coupled cache has a row (KvQ8, which the gate and the greedy check run); the one-sided ones have none                                                                     |
| `gpu_seam_kv_formats_track_cpu`                | Qwen3-0.6B Q4_K_M        | every other KV cache format                                                | ran     | No row below 8 bits, and turbo3's distance is unexplained (B207)                                                                                                                   |
| `gpu_seam_int8_prefill_tracks_cpu`             | Qwen3-0.6B Q4_K_M        | the int8 cooperative-matrix K-quant prefill GEMM, with a check that it ran | skipped | Skips because that GEMM does not run under RADV (B231), not for a missing file. Qwen3's Int8 row scores the same prefill wherever the kernel runs, which could not be checked here |
| `gpu_seam_matches_cpu_ernie4_5_moe`            | ERNIE-4.5-21B-A3B Q4_K_M | a real MoE model's prefill, last row                                       | ran     | No real MoE row (B207)                                                                                                                                                             |
| `gpu_seam_matches_cpu_qwen35moe`               | Qwen3.6-35B-A3B          | the same on the DeltaNet skeleton                                          | skipped | B60                                                                                                                                                                                |
| `gpu_seam_matches_cpu_diffusion_gemma`         | DiffusionGemma-26B-A4B   | causal prefill, last row                                                   | skipped |                                                                                                                                                                                    |
| `gpu_seam_matches_cpu_diffusion_gemma_denoise` | DiffusionGemma-26B-A4B   | a canvas denoise step                                                      | skipped | A forced run does not drive a denoise step                                                                                                                                         |
| `gpu_seam_matches_cpu_bitnet`                  | bitnet_b1_58-large       | TQ2_0 prefill, last row                                                    | skipped |                                                                                                                                                                                    |
| `gpu_seam_matches_cpu_deepseek2`               | DeepSeek-V2-Lite         | MLA attention and MoE prefill, last row                                    | skipped |                                                                                                                                                                                    |
| `gpu_prefill_matches_llama_debug_dump`         | a llama.cpp dump's model | prefill against llama.cpp's logits                                         | skipped | Skips without `INFR_LLAMA_DUMP`. Not a comparison with the CPU: llama.cpp comparisons stay tolerance-based by plan 11 section 4                                                    |

**Not in scope, unchanged:** the GPU-against-GPU comparisons (session reuse and
prefix sharing against a fresh prefill, paged and streamed against resident,
replay against static for DiffusionGemma, the multi-GPU splits against one
device, the MTP draft chain against its per-step form), the tests that count
decode recordings (`*_decode_replays_from_one_recording`), and the synthetic
models' Vulkan-against-CPU tests (`gpu_synthetic_*`, `synthetic_lfm3_vulkan_*`),
which bound a logit difference on a fixture rather than hash, token or cosine.
`mtp_spec_matches_target_only_greedy` is parked and fails on `Vulkan1` as its
doc comment says; it was not touched.

## Not measured

- **Metal.** Nothing here ran on Apple hardware; the harness drives the CPU and
  Vulkan seams only. A Mac owner would add a `ForcedOn::Metal` arm to
  `SeamModel::forced_logits` (the Metal seam runs the same
  `generate_dense_backend`), run `measure_gpu_tolerance` on the gated models,
  and add Metal rows to the table from those numbers. For a routed-expert model
  that arm also needs `Op::MoeRoute` on Metal, which refuses it by name today.
- **Windows, for everything routed-expert**: the synthetic MoE's row, its gate
  test, the own-routing check and its planted bugs ran under RADV only.
- **Real MoE models** in the gate (one is measured above, none has a row), paged
  MoE models (a forced run of one is refused), quantized KV caches below 8 bits,
  and what the greedy check leaves of the chained and the replayed decode (the
  chained sampler, other chain lengths, Windows): see B207 in
  [`backlog.md`](backlog.md).
