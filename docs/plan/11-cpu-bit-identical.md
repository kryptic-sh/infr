# 11 — CPU output bit-identical on every platform, GPUs within tolerance of it

The CPU backend is to produce the same bits on every operating system, every CPU
tier and every architecture, with tests that exploit that: one checked-in value
per case, exact comparisons, the same assertions on every CI system. Every GPU
is then checked against that one CPU reference within a stated tolerance. This
phase comes before the rest of the kernel-tuning phase (10), which needs the
fixed reference to judge tuned values.

Audited and scoped on 2026-11-17. Tags: **[read]** verified by reading,
**[ran]** verified by running, **[I]** inferred, **[U]** unknown. S0 to S4 (with
S5 folded into S2) and S6 are built (see section 5), and the GPU slices G1 and
G2 (section 6); G3 and G4 are not.

## 0. The cause, measured

A standalone probe evaluated each math function the CPU path takes from the C
runtime on a few hundred thousand f32 inputs, natively on Windows (rustc
1.100.0, MSVC, UCRT) and under WSL Ubuntu 26.04 (rustc 1.99.0, glibc 2.43) on
the same Ryzen, at baseline and at `target-cpu=native` **[ran]**. Baseline and
native agree with each other on each system. Windows against Linux:

| Function                                   | Inputs differing | Max ULP |
| ------------------------------------------ | ---------------- | ------- |
| `f32::exp`                                 | 267 of 400k      | 1       |
| `f32::ln`                                  | 75 of 300k       | 1       |
| `f32::ln_1p`                               | 69,165 of 300k   | 1       |
| `f32::tanh`                                | 5,131 of 300k    | 1       |
| `f32::powf` (RoPE bases)                   | 153 of 203k      | 1       |
| `sin`/`cos` at Qwen3's RoPE angles         | 4,970 of 524k    | 1       |
| `f32::mul_add`, `f64::mul_add` (any build) | 0                | 0       |

So the recorded guess was right: the difference is the C runtime's math
functions, and it enters at layer 0 of every forward (RoPE `sin`/`cos`/`powf`),
then every attention softmax (`exp`), every SiLU or GELU, the MoE router and
DeltaNet. Greedy text forks at the first near-tie, which is why short goldens
agreed and long ones carried `per_os` hashes until S2.

Neither runtime is correctly rounded across the board; glibc 2.43's `tanhf` and
`log1pf` are **[ran]**, which means Linux output also depends on the glibc
VERSION (the released binary links the user's glibc), a plausible cause of the
runner-against-dev-box divergence `docs/synthetic-models.md` records **[I]**.

Nothing else was found that differs between operating systems **[read]**: thread
count never changes an arithmetic order (the pool only partitions outputs);
there is no fast-math, contraction, FTZ or DAZ; allocator state never reaches
results; the RNG is integer xorshift. `mul_add` is exact everywhere (hardware
FMA, or the C runtime's software `fma`, which the probe found identical) but is
a slow library call on the baseline x86 builds CI and the release ship.

## 1. Where platform math enters

Rust lowers `exp`, `ln`, `ln_1p`, `tanh`, `powf`, `sin`, `cos` to calls into the
C runtime; `sqrt`, `round`, `floor`, `ceil`, `trunc` are exact everywhere
**[read, ran]**. Production call sites:

- `infr-cpu/src/lib.rs`: `Op::Softmax`, `Op::GatedRmsNorm`, `Op::Rope`,
  `Op::QkNormRope`, `Op::Attention`'s softmax, `Op::HeadGate`, `Op::Softcap`,
  `Op::ArgmaxProb`, the MoE router's softmax, sigmoid and sqrt-softplus,
  `Op::Mla`'s rope and softmax, the hyper-connection sigmoids, `CompressPool`,
  `hyper_sinkhorn`, `sample_token`, `conv1d_silu`, `deltanet_scan`.
- `infr-cpu/src/kernels.rs`: `act_fn` (SiLU, GELU through `tanh`, sigmoid).
- `infr-llama`: `sampling.rs` (temperature softmax, min-p), `config.rs`
  (`mla_rope_mscale`, `yarn_freq_divisors`), `seam/runner.rs` (the MLA YaRN
  scale), `seam/sc.rs` (Diffusion-Gemma self-conditioning), `mtp/mod.rs`,
  `diffusion.rs`.
- `infr-core/src/pager/shapes.rs` uses an f64 `powf` for a pager workload shape,
  not inference.
- `infr-core/src/decode_spec.rs`: `e8m0_from_f32` takes an E8M0 exponent from
  `f32::log2` (MXFP4 quantization); it can come from the float's bits instead.
- `infr-metal/src/exec.rs`: ten host-side calls in the Metal backend's CPU paths
  (`act_fn`, the short convolution's SiLU, softmaxes, the MoE router's softmax,
  the DeltaNet gate's sigmoid, softplus and decay).

All of them moved to `infr_core::math` in S2; the lint
`crates/infr-core/tests/no_platform_libm.rs` now fails on any platform math call
in production code, with no allowlist.

`infr-gguf` has none, and tokenization has no float arithmetic that reaches
output.

**SIMD tiers [read]:** dispatch is by `is_x86_feature_detected!`; aarch64 runs
the scalar kernels (there are no NEON kernels). Every quantized family
accumulates in i32 and finishes with one f32 expression written without FMA in
every tier, and the f32 dots use one fixed `mul_add` order. So tier identity
holds by design, but it is only partly TESTED: `batch_tier_tests` compares one
row against a batch within a tier, for the K-quants and Q8_0; tier against
scalar for tiers below the host's, and the 32-element-block, IQ and ternary
families, are asserted by comments.

**Row count [U]:** `Op::Linear` is identical across row counts (B200).
`Op::MoeFfn` differs by design between one row and several (int8 activations),
attention's row blocking was not reviewed for it, and the prefill chunk height
can depend on a machine's memory.

**Host math that feeds the GPUs [read]:** the YaRN divisors, the MLA YaRN scale,
host sampling and Diffusion-Gemma's self-conditioning.

## 2. The fix: project-owned, correctly rounded f32 math

`infr_core::math`: `exp`, `ln`, `ln_1p`, `tanh`, `sin_cos`, `powf` for f32, and
thin `sigmoid`, `silu`, `gelu_tanh`, `softplus` built on them. Chosen over a
pure-Rust libm dependency (a new dependency, not correctly rounded, with
arch-specific paths to audit **[I]**) and over simple polynomials (accurate to a
ULP or two, so test expectations would encode the implementation).

Correctly rounded means each result is the mathematically nearest f32, which
makes it a fact that any other correctly rounded library agrees with, and lets
test expectations come from a high-precision reference rather than from the code
under test. The design is CORE-MATH's: evaluate in f64, with a rare hard-case
fallback.

**The determinism rule inside the module:** only `+ - * /`, `sqrt` and integer
operations; no `mul_add`; nothing that depends on target features. Then
baseline, native and aarch64 builds, and lane-wise SIMD versions, give the same
bits, and one code path is fast everywhere. Whether 53 bits of working precision
is enough without FMA is to be confirmed per function against CORE-MATH's bounds
**[I]**.

**Speed [I]:** a scalar correctly rounded `exp` of this design should cost about
what glibc's does. The backlog's measurements (B173) put the scalar `exp` in the
activation at a few percent of a prefill, and a vectorized `exp` as a win that
was blocked on a three-platform re-bless; a lane-wise version with identical
operations becomes possible here, provably identical to the scalar one over
every input. Measure per-op times for the activation, attention, RoPE and
softmax before and after on both systems.

## 3. The guarantee, as milestones

- **L1 — identical across operating systems.** The math module, every call site
  migrated, a lint against the platform library. One re-bless of the CPU
  goldens.
- **L2 — identical across SIMD tiers, CPUs and thread counts.** Holds by design;
  needs proof: a forced-tier switch (`kernels.cpu.max_tier`), op-level hashes at
  every tier the host supports and at several thread counts, a
  tier-against-scalar kernel test for every family, and row-count identity for
  the MoE layer and attention. Cheap unless a family fails, in which case its
  kernel is made to match the scalar expression.
- **L3 — identical across architectures.** aarch64 runs the scalar kernels and
  `mul_add` is exact on both, so this should follow from L1 **[I]**; it needs
  the same hashes asserted on macOS arm64, an audit of `f32::max` at ±0, and a
  rule that any future NEON kernel matches scalar (a possible speed cost only
  then).

No part is expected to cost measurable speed; the one to measure is correctly
rounded `sin`/`cos` for RoPE at long context.

## 4. Tests

1. **Known answers per function:** a checked-in table of inputs, hard cases and
   range edges included, with the correctly rounded result, generated once by a
   development-time script from a multi-precision library (never run in CI); an
   ignored exhaustive test over every f32 input per unary function; a dense-grid
   hash asserted on every CI system.
2. **A lint:** `crates/infr-core/tests/no_platform_libm.rs`, modelled on
   `no_bare_print.rs`, forbids direct `.exp(`, `.ln(`, `.tanh(`, `.sin(`,
   `.powf(` and their relatives in production code outside the math module, with
   an explicit allowlist.
3. **Op-level hashes, no model needed:** synthetic graphs per op family with
   integer-generated weights, each output's bits hashed against ONE checked-in
   value, in the normal test job on every system, at every forced tier and at
   several thread counts.
4. **Whole-model goldens:** short single-hash cases asserted on Windows too
   (today the Windows halves are never asserted in CI); `per_os` deleted. The
   goldens job is CI's critical path today; the hash subset becomes its own
   short step.
5. **Simpler afterwards:** the per-OS hashes and the notes that explain them,
   the exclusion of Qwen3.5 from CI as not portable, the caveats about goldens
   not reproducing across CPUs.
6. **Stays tolerance-based:** comparisons with llama.cpp dumps, GPU against CPU,
   and comparisons of two CPU paths that differ by design (production against
   reference, quantized KV against f32).

## 5. Slices

| #   | Slice                                                                                                                                                                                                                                                                                                                                         | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0  | **Done.** An env-gated per-op output hash in the CPU executor (`INFR_CPU_OP_HASH`), kept as a debug aid; the cross-OS bisect on the cached Qwen3-0.6B case whose `per_os` hashes differ                                                                                                                                                       | **[ran]** First difference: graph 0, op 5, layer 0's `QkNormRope`, its inputs identical on both systems; 470 of graph 0's 477 ops differ after it. Ops that are a SOURCE of difference (inputs equal, output not): `QkNormRope` (96), `GatedAct`/SiLU (37), `Attention`'s softmax (7); never Linear, RmsNorm, Add, WriteKv, EmbedGather or Argmax. First differing token in graph 41 (the second prompt)                                                                                                                                                     |
| S1  | **Done.** `infr_core::math`, the known-answer table and its generator, the exhaustive test, the grid hash on every system, the lint with today's sites allowlisted; slice forms of `exp`, `tanh`, `sigmoid`, `silu`, `gelu_tanh`                                                                                                              | **[ran]** Every unary function correctly rounded on all 2^32 inputs; `powf` correctly rounded on every input tested (not proven); grid hashes identical on Windows (native, baseline) and WSL; slices identical to the scalars on all 2^32 inputs. Details and speed: [`docs/math.md`](../math.md)                                                                                                                                                                                                                                                           |
| S2  | **Done.** Every call site migrated (slice forms on the hot paths, one shared RoPE helper), the allowlist emptied and the lint made exceptionless, the CPU goldens re-blessed once, `per_os` deleted; with it S5: the slices dispatched at run time to baseline/AVX2/AVX-512 (`kernels.cpu.math_tier` forces one), and the two-lane path tuned | **[ran]** One hash per case, identical natively on Windows and under WSL, for every cached CPU golden (five of the 17 cached cases changed text on each system, against the checked-in hashes); the goldens identical with the math forced to each tier on a baseline build; every slice identical to its scalar function at every tier on all 2^32 inputs; llama.cpp distances unchanged (synthetic oracles, Qwen3-0.6B dump). Speed and the table of changes: [`docs/math.md`](../math.md)                                                                 |
| S3  | **Done.** The op-level hash suite (`crates/infr-cpu/tests/op_hashes.rs`) in the normal test job on all three systems; two goldens (Qwen3-0.6B, Qwen3.5-0.8B) after the `test` matrix suite on Windows and macOS, with `INFR_TEST_REQUIRE_MODELS` so a missing model fails; Qwen3.5-0.8B back in the Linux goldens job                         | **[ran]** 158 cases covering all 39 `Op` variants, every KV cache format and each pool size threshold, one hash each at every tier (scalar, AVX2, AVX-512, AVX-512 VNNI) and at 1, 2, 3, 7, 33 and 32 threads, the same on Windows and under WSL. **[U]** CI steps not run before the push: Windows and macOS goldens, macOS op hashes (the first aarch64 run; the macOS golden step is informational until it passes, B205)                                                                                                                                 |
| S4  | **Done.** `kernels.cpu.max_tier` (one ceiling for the kernels and the math slices, replacing `math_tier`); every kernel family against its scalar kernel at every tier; thread-count and row-count identity; the float `min`/`max` audit                                                                                                      | **[ran]** One family failed: AVX-512 Q8_0 (super-block activations) associated its block sums differently; fixed, which re-blessed Qwen3.5-0.8B (the new knight hash is the one a CI runner printed in July; the new France hash is not: unsettled until its first CI run). Every other family bit-identical at every tier. Six dense models bit-identical at chunk heights 512, 7, 3 and 1, Windows and WSL; MoE excluded by design (B203)                                                                                                                  |
| S5  | **Done, in S2.** A vectorized `exp` (and `tanh`) identical to the scalar one                                                                                                                                                                                                                                                                  | see S2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| S6  | **Done.** The f32 dots (`kernels::Dots`: `dot`, `dot_n`, `dot_mn`, `dot_f16`, `dot_bf16`, `axpy`, `mul_add_into`, `dequant_dot_rows`) run, on a build without FMA, an AVX2+FMA+F16C compilation of the same source, chosen once per graph under the `avx2` ceiling; the f16 dot and the f16 KV write convert eight values at a time           | **[ran]** Bits unchanged: `tier_tests` (each dot against scalar at every tier, and fused, with planted failures; the f16 narrowing against `half`'s software conversion on all 2^32 inputs), op hashes and goldens intact on Windows native and baseline and WSL. Baseline build: Qwen3-1.7B pp512 314-325 to 986-1027 tok/s (native 1036-1155), reference-mode decode 14.0-14.3 to 26.6-27.3 (one low run per build left out); native build level over eight paired rounds (one core busy with another agent's test). Details: [`docs/math.md`](../math.md) |

**What is left for L2 and L3.** L2 holds on the one AVX-512 machine every x86
tier was run on **[ran]**: every kernel family and the op suite at each tier,
thread counts 1 to 33, prefill chunk heights for six cached dense families. Its
listed exception is `Op::MoeFfn` across row counts (B203, a decision for the
owner). L3 **[ran, CI 2026-11-17]**: on `macos-15` (aarch64) the op suite, the
math grid hashes and `mul_add_exact` passed in the normal test job, and the
goldens subset (Qwen3-0.6B, Qwen3.5-0.8B) reproduced the same hashes as Linux
and Windows; that step is now required. Not run there: the other goldens, the
exhaustive math tests and the row-identity tests on real models. The
`f32::min`/`max` audit (`docs/math.md`, "Identity across tiers, …") found one
zero-sign site reaching stored bits (fixed). `powf` is correctly rounded on
every input tested, not proven. S6 (FMA for the f32 dots on baseline builds) is
done.

## 6. GPUs against the one CPU reference

**Today [read]** (`crates/infr-llama/tests/cpu_backend.rs`): GPU tests are a mix
of exact tokens against the CPU reference path, whole-vocabulary logit cosine
against the CPU (which B60 showed cannot separate a match from a mismatch; B174
and B141 record specific weak ones), exact-hash GPU goldens, and GPU-against-GPU
self-consistency. Op-level parity uses the production CPU with absolute error
thresholds.

**The reference is the CPU's reference mode** (`kernels.cpu.reference`: f32
dots, no int8 activations). Its error is closest to exact, and B192 already
scores every GPU variant against it. The production CPU path has its own int8
activation error in places that differ from a GPU's, and keeps its own exact
goldens.

**Target.** Teacher-force the CPU reference's tokens through the GPU so the two
never fork, and score several positions with: KL divergence over the full
vocabulary; cosine over the reference's top-k probability mass; the largest
logit difference relative to the reference row's logit range; and top-1
agreement where the reference's margin is not a near-tie. Tolerances are per
class (f32/f16 paths, int8-activation paths, quantized KV per bit width, MoE),
each set from the worst value measured on the RX 7900 XTX and the integrated
GPU, native and with 64 lanes forced, times a stated margin, in one table with
provenance. Each must be shown to fail on a set of deliberate bugs: a dropped
bias, a mis-scaled block, an attention mask off by one, a RoPE pairing or
position error, a router that ignores its bias, a wrong quant nibble map,
swapped gate and up, a wrong norm epsilon.

**The exact-token GPU goldens** are retired from gating in favour of the
tolerance test, since tunable kernels let each device choose parameters that
change bits and contributors cannot bless hashes for their own devices. One
command for hardware nobody has seen: `infr tune check --against cpu-reference`,
which runs the op-level families and the teacher-forced comparison on a small
model and prints the distances with pass or fail.

| #   | Slice                                                                 | Status                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| G1  | The teacher-forced harness and metrics                                | **Done.** `infr_llama::tolerance::metrics`; `SeamModel::forced_logits` (ordinary chunked prefill rows through a `prefill_logits` hook, then per-token decode steps on one session); its row-identity self-check in the normal test job. See [`gpu-tolerance.md`](../gpu-tolerance.md)                                                                                                                                                  |
| G2  | Measure both GPUs at both lane widths; set the tolerance table        | **Done** for nine models (two synthetic, seven real, Float/Int8/KvQ8) on both GPUs, native and forced wave64, tuning rows on and off: `tolerance::table::TABLE`, tests `gpu_seam_tolerance_*`. **[ran]** The CPU reference, not production, is the right reference; the int8 class sees only gross bugs, so every quantized model is also gated with its int8 kernels off. MoE and KV formats below 8 bits are not in the table (B207) |
| G3  | The mutation demonstrations                                           | Part done: six planted bugs measured by hand on Qwen3-0.6B (the doc's table); still to do: commit them as feature-gated tests, run them on the integrated GPU and a float model, add the router-bias and nibble-map bugs                                                                                                                                                                                                               |
| G4  | Convert the cosine tests and the golden tests; wire `infr tune check` | Open. The CLI calls `SeamModel::forced_logits` and the table; the existing exact goldens' state across the matrix is in the doc                                                                                                                                                                                                                                                                                                        |

G1 to G4 follow L1. The weak assertions B60, B174 and B141 name can be tightened
before that.

## 7. What changes for users

Every CPU output changes once, on every platform, when the call sites move to
the new math; goldens are re-blessed once and each case then has one hash.
GPU-against-CPU token tests may flip where margins are thin and are re-checked.
Distances to llama.cpp should move negligibly, since its own math already
differs from both runtimes by a ULP or two; measure.
