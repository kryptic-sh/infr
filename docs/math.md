# `infr_core::math`: the same bits on every platform

`f32::exp`, `ln`, `ln_1p`, `tanh`, `powf`, `sin` and `cos` lower to the C
runtime, and the C runtimes disagree in the last bit: UCRT on Windows against
glibc on Linux, one glibc version against another, and so on. Every CPU output
computed through one of them inherits the disagreement, which is why some CPU
goldens used to carry one hash per operating system. `infr_core::math` replaces
them with functions that return the **correctly rounded** binary32 result — the
one nearest the exact mathematical value, ties to even — so there is exactly one
right answer, the same everywhere, and tests can take it from a multi-precision
library instead of from the code under test. The plan this serves is
[`plan/11-cpu-bit-identical.md`](plan/11-cpu-bit-identical.md).

Every call site in the workspace uses it (slice S2, which changed every CPU
output once), and the lint `crates/infr-core/tests/no_platform_libm.rs` fails on
any platform math call in production code outside the module: there is no
allowlist.

## What it guarantees

| Function                                   | Correct rounding                     | Verified by                                |
| ------------------------------------------ | ------------------------------------ | ------------------------------------------ |
| `exp`, `ln`, `ln_1p`, `tanh`               | every binary32 input                 | exhaustive test, all 2^32 inputs           |
| `sin`, `cos`, `sin_cos`                    | every binary32 input                 | exhaustive test, all 2^32 inputs           |
| `powf`                                     | every input tested, not proven       | known answers, grid hash, review (below)   |
| `sigmoid`, `silu`, `gelu_tanh`, `softplus` | not correctly rounded: deterministic | the call sites' own expressions, see below |
| the slice functions                        | equal to the scalar function         | exhaustive identity test, all 2^32 inputs  |

**Platforms.** The bits depend on nothing but IEEE 754 arithmetic, so they
should be the same everywhere. What has been run: Windows (MSVC) and Linux
(glibc, under WSL) on one AVX-512 Ryzen, each with `target-cpu=native` and with
a baseline `x86-64` build, all giving identical grid hashes. aarch64 has only
been compiled (`cargo clippy --target aarch64-apple-darwin`); CI's macOS job is
the first run there, and its grid hashes will say.

**`powf`.** The fast path's error bound and the exact-result handling are argued
in `crates/infr-core/src/math/pow.rs`: `x^y` can be exactly a rounding boundary
(`11^7`, `66049^1.5`, `(2^-16)^9.375`), and those are detected with integer
arithmetic and rounded to even. What is NOT proven is that every inexact `x^y`
lies further than the accurate path's error (about `2^-90` relative) from a
boundary: there are 2^64 input pairs, so no exhaustive test, and the module
relies on no published worst-case bound. It is tested on the known-answer table
(mpmath, including RoPE's `theta^(-2p/d)` for the bases models use) and on the
grid, and review checked it against mpmath on the whole RoPE domain and on three
sets of 2^32 random pairs with no mismatch. Its production uses are RoPE's
frequencies (f32), which depend only on an op's shape so the caller can compute
them once rather than per element, and a pager workload shape, which is f64 and
has no counterpart here yet.

**The helpers** are the expressions the CPU backend's call sites write,
operation for operation, with the platform function replaced:
`sigmoid(x) = 1/(1 + exp(-x))`, `silu(x) = x/(1 + exp(-x))`, `gelu_tanh` as
`act_fn` writes it, `softplus(x) = max(x, 0) + ln_1p(exp(-|x|))` as the DeltaNet
scan writes it. Moving a call site onto one changes its bits only where the
platform function was wrong.

## Special values and NaN

**One NaN.** Every function, scalar and slice, returns exactly `math::NAN`
(`0x7fc00000`, the positive quiet NaN with no payload) whenever its result is
NaN: for any NaN input — quiet or signalling, any payload, either sign — and for
an invalid operation. The code selects it explicitly instead of leaving it to
the hardware, because x86 and AArch64 differ: x86 generates `0xffc00000` for an
invalid operation and propagates input payloads, AArch64's default NaN is
`0x7fc00000`. The grid hashes hash the real result bits, NaN inputs of each kind
included, so a NaN that slipped through would change them.

| Function    | Special results                                                                                                                                            |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `exp`       | `exp(±0) = 1`; `+inf` from about 88.72 up and at `+inf`; `+0` from about -103.97 down and at `-inf`                                                        |
| `ln`        | `-inf` at `±0`, NaN for `x < 0` (`-inf` included), `+inf` at `+inf`, `+0` at 1                                                                             |
| `ln_1p`     | `±0` at `±0`, `-inf` at -1, NaN below -1, `+inf` at `+inf`                                                                                                 |
| `tanh`      | `±0` at `±0`, `±1` beyond about 9.01 and at `±inf`                                                                                                         |
| `sin`       | `±0` at `±0`, NaN at `±inf`                                                                                                                                |
| `cos`       | 1 at `±0`, NaN at `±inf`                                                                                                                                   |
| `powf`      | C99 F.9.4.4: `x^±0 = 1` and `1^y = 1` for any `x`/`y` (NaN too); `±0`, `±inf` and `(-1)^±inf` per C99; a negative finite `x` with a non-integer `y` is NaN |
| `sigmoid`   | 1 at `+inf`, 0 at `-inf`                                                                                                                                   |
| `silu`      | `+inf` at `+inf`, `-0` at `-0`, **NaN at `-inf`**                                                                                                          |
| `gelu_tanh` | `+inf` at `+inf`, `-0` at `-0` and for large negative `x`, **NaN at `-inf`**                                                                               |
| `softplus`  | `+inf` at `+inf`, `+0` at `-inf`                                                                                                                           |

`silu(-inf)` and `gelu_tanh(-inf)` are NaN because the call sites' expressions
are: `x / (1 + e^-x)` is `-inf/inf` and `0.5 x (1 + tanh(...))` is `-inf * 0`
there (`helper_special_values` checks the helpers against those expressions on
the platform functions). The mathematical limits are 0 and `-0`. The helpers
keep today's NaN so the migration changes nothing but the platform's last bit;
whether an activation should map `-inf` to 0 is a separate decision (an `-inf`
activation input is already a broken forward).

## Slices

`exp_slice` (in place), `exp_slice_to` (source to destination),
`exp_sub_slice(xs, m)` (`exp(x - m)` with the subtraction rounded to binary32
first, as a softmax over a row writes `(x - max).exp()`), `tanh_slice`,
`sigmoid_slice`, `silu_slice` and `gelu_tanh_slice`. Each runs, per element, the
same branch-free function the scalar version IS: only `+ - * /`, conversions,
comparisons used as selects and table lookups by integer index, no intrinsics
and no `std::simd`. So a slice equals its scalar function element for element on
every target, whatever the compiler vectorizes it into; the identity is tested
on all 2^32 inputs at every SIMD tier (below), and every CI system hashes the
slices' output on the grid. A call site whose expression is not one of these
forms (`(v * scale - max).exp()`, say) computes the argument into a buffer
first: the binary32 operations are the same, so the bits are too.

What makes them vectorize: no data-dependent control flow. Out-of-range inputs
are clamped before the arithmetic and the clamped ends produce the special
results by themselves (`e^89` rounds to `inf`, `e^-104` to 0, `tanh(9.1)` to 1);
only NaN is selected afterwards. An earlier version that selected `inf` and 0
after the fact compiled to a branch around the arithmetic and did not vectorize
at baseline at all. `ln`, `ln_1p`, `sin`/`cos` and `powf` have no slice form:
none is on a per-element hot path. RoPE's frequencies depend only on an op's
shape: `rope_freqs` computes them once per op and `rope_sin_cos` turns each
angle into its sine and cosine with one reduction, for every RoPE variant on the
CPU.

### Tiers, chosen at run time

The slice loops are compiled once per SIMD tier (`math::Tier`): `baseline` (the
build's own target: SSE2 for a baseline x86-64 build, NEON on aarch64), and on
x86-64 `avx2` and `avx512` (F, DQ, VL, BW), each the same loop inside a
`#[target_feature]` function (`src/math/slices.rs`). The first call picks the
widest tier the CPU has (`is_x86_feature_detected!`), so a binary built for
baseline x86-64 — the released binaries and CI — still runs four or eight
binary64 lanes where the CPU has them. FMA instructions may be available to the
compiler in the wide tiers (enabling `avx512f` implies the `fma` target feature,
and a `target-cpu=native` build has FMA in every tier). The bits do not depend
on that: Rust never contracts `a*b + c` into a fused operation without an
explicit `mul_add`, which this module never calls, and never reassociates
floating-point operations. The tiers are the same source performing the same
operations per element in the same order, so they give the same bits by
construction; the exhaustive tier-identity tests below would fail if a tier
contracted or reordered anything.

**Capping the tier.** The slices run at `infr_core::simd::active`'s tier, the
one every SIMD dispatch in the CPU backend asks too: the widest the CPU has,
under the ceiling `kernels.cpu.max_tier` (`INFR_CPU_MAX_TIER`: `scalar`, `avx2`,
`avx512`, `avx512vnni`). The slices have no VNNI form, so `avx512vnni` runs
their AVX-512 loops, and `scalar` their baseline ones. It is a process-wide
debug switch: every CPU backend applies its value when it is built
(`simd::set_max`; unset means no ceiling), so the backend built last decides for
every backend in the process, and a backend built without the setting lifts an
earlier one's ceiling rather than inheriting it. A ceiling above what the CPU
has runs the CPU's widest tier. A misspelt name is a configuration error at
load, from the file, `--set` or the environment alike. It changes only speed; it
exists so the narrow paths every aarch64 machine and every pre-AVX2 x86 CPU runs
can be measured and tested on a machine with wider ones. In a
`target-cpu=native` build `baseline` is the native code, so a baseline
measurement needs a baseline build (`RUSTFLAGS="-C target-cpu=x86-64"`).

**The two-lane path.** With two binary64 lanes and no gather, the loop the
compiler vectorized was slower than the C runtime's scalar `expf` (1.77 against
1.09 ns per element). Three things closed most of it, each checked to leave
every result bit unchanged (correct rounding makes that the natural check: any
correctly rounded variant has the same bits) by the exhaustive tests:

- `exp`'s binary64 value takes a 256-entry table and a degree-4 polynomial by
  Estrin's scheme instead of 64 entries and degree 6 by Horner (a shorter
  dependency chain), and builds `2^(k>>8) * T` by one integer addition to the
  table entry's bits (`exp.rs` has the error bound). The tables stay literals
  from the generator; two derived arrays are computed at compile time by exact
  bit arithmetic and one rounded division per entry.
- The baseline tier writes its loop in blocks of four elements, which the SLP
  vectorizer packs into two-lane vectors where the loop vectorizer judged the
  `exp` loop unprofitable and left it scalar (`NARROW` in `slices.rs`; two and
  eight were slower).
- `sigmoid`, `silu` and `gelu_tanh` take two passes per 64 elements, at every
  tier: the `exp`/`tanh` lanes into a buffer, then the binary32 division and
  products in a plain loop at twice the lanes the binary64 part has. The
  operations are the helpers' own (`sigmoid_finish` and the like), so the bits
  are the same; it was faster at every tier than one fused loop.
- Scalar `ln_1p` on a baseline build took 16.5 ns per call against 6.7 natively,
  on inputs that never reach its `x >= 2^24` branch; with that branch out of
  line (`ln_1p_big`) it takes 6.4. Presumably the compiler had been computing
  the branch's division speculatively.

The division in `sigmoid`/`silu` stays: the result must be the correctly rounded
binary32 quotient of `x` and `1 + e` (the call sites' own expression). A
reciprocal times `x` rounds twice and differs from it on some inputs; the
binary64 quotient rounded to binary32 is identical (double rounding is innocuous
for a binary32 division done in binary64) but a binary64 division has half the
lanes of a binary32 one (not measured here). No cheaper form with the same bits
was found; the two-pass loop instead runs the binary32 division four lanes wide.

## Identity across tiers, threads, row counts and architectures

Plan 11's S3 and S4 extend the guarantee from this module to the whole CPU
backend. What is proven, and how:

- **Every SIMD tier.** `kernels.cpu.max_tier` (`scalar`, `avx2`, `avx512`,
  `avx512vnni`; `infr_core::simd`) caps every run-time dispatch in the CPU
  backend and the slices above. `crates/infr-cpu/src/kernels/tier_tests.rs` runs
  every quantized kernel family (one-row and batch, Q2_K to Q8_0, the IQ and
  ternary formats, the native-32-block family, MXFP4, NVFP4, Q4_K's 2- and 8-row
  tiles, the 8-row Q8_0/Q5_0 tile) through its dispatcher at each tier the host
  has and holds it to the scalar tier's bits, on inputs that reach the int8
  extremes, both zero signs, subnormals and every code pattern, and checks which
  tier each dispatcher took. One family differed (AVX-512 Q8_0, fixed in S4; see
  the golden table below). The f32/f16/bf16 dots have no run-time dispatch: one
  source fixes the order of every `mul_add` and of the lane sum.
- **Every op, tier and thread count.** `crates/infr-cpu/tests/op_hashes.rs`:
  every `Op` variant, one checked-in hash per case at every tier and at 1, 2, 3,
  7, 33 and the machine's threads. The packed VNNI GEMMs and the MoE expert
  tiles are covered there (they have no scalar form of their own to compare
  with).
- **Every prefill chunk height** for the dense families
  (`crates/infr-llama/tests/support/row_identity.rs`): the same ids and the same
  bits in everything the last position and the decode steps write, at heights
  512, 7, 3 and 1, on Qwen3, gemma-3, Qwen3.5, LFM3, ERNIE-4.5 and SmolLM3 and
  on the synthetic LFM3 file in CI. **Exception, by design:** `Op::MoeFfn`
  quantizes expert activations to int8 for two rows or more and not for one
  (docs/backlog.md, B203).
- **Windows and Linux** reproduce all of the above with the same values. aarch64
  runs the scalar kernels and is the first thing CI's macOS jobs check.

**What "L2 proven" covers, exactly.** Run on one AVX-512 VNNI machine (Ryzen 9
9950X3D), Windows and Linux under WSL:

- every quantized kernel family's dispatcher at every x86 tier, both of its
  compilations, against the scalar kernel bit for bit (`tier_tests`);
- every `Op` variant at every tier and at 1, 2, 3, 7, 33 and 32 threads
  (`op_hashes`), including every KV cache format the CPU backend implements
  (F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, Turbo2/3/4) written and
  read back, and one shape past each pool size threshold (a decode `Op::Linear`
  whose rows per task range over `DECODE_ROWS_MIN..=MAX`, a decode attention
  whose f16 prefix widens in several `KV_WIDEN_CHUNK` pieces, a lightning
  indexer with several score tasks);
- prefill chunk heights 512, 7, 3 and 1 on six cached dense models, and in CI on
  the synthetic LFM3 with f32 weights and with Q4_K, Q5_0 and Q8_0 weights;
- `f32::mul_add` correctly rounded on native and baseline builds
  (`tests/mul_add_exact.rs`), which the f32 dots' identity across builds rests
  on.

**By reading only:** that no parallel split reduces across tasks for shapes
larger than the suite's (the threshold cases cross each threshold once, not
every chunk count); that `Op::MoeFfn`'s expert tiles are identical across thread
counts at real model sizes (the suite's MoE cases are small); the packed VNNI
GEMMs against the per-row kernels on shapes other than the suite's. **Not
covered:** other x86 microarchitectures (the tiers are the same code on any of
them; a CPU whose `fmaf` or hardware FMA were incorrect would not be), and
aarch64, whose first runs are CI's macOS jobs.

**The architecture audit (L3).** Float `min`/`max` return either zero when given
`+0` and `-0`, and x86 and AArch64 lower them differently; NaN handling is
specified (the other operand). Every production use in the CPU backend and
`infr_core` was read:

| Site                                                                    | Can a zero sign reach an output bit?                                   |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| q4_1/q5_1 KV quantizers' block minimum (`kvquant.rs`)                   | **yes**: the stored `min`'s sign bit. Fixed: `min_max` keeps the first |
| softmax, attention, MLA, router, sinkhorn and `CompressPool` row maxima | no: the max only offsets `exp(x - max)`, and `exp(±0)` is 1            |
| lightning indexer `d.max(0.0)`                                          | no: the term is added to a sum that starts at `+0`                     |
| SwiGLU clamp `gate.min(limit)`, `up.clamp(-l, l)`                       | no: `limit` is never zero; `clamp` is comparisons                      |
| int8 quantizers' `max(abs(x))`                                          | no: `abs` makes every zero `+0`                                        |
| `softplus`, ReLU-like `if x > 0.0 { x } else { 0.0 }`                   | no: written as comparisons already                                     |

Also checked: float-to-int casts saturate (defined), `f64 as f32` rounds to
nearest, `sqrt`/`round`/`floor`/`ceil`/`trunc` are exact (where `round` is a C
runtime call, the result is still the exact one), integer accumulators wrap the
same everywhere, and the hashes never depend on `usize` width or byte order
(FNV-1a over little-endian bytes). Two more sites gave results that depended on
the standard library's unstable sort rather than on the input: `Op::Sample` and
the host sampler's top-k path ordered equal logits arbitrarily. Both now break
ties by token id.

**Speed: what the ceiling costs when it is not set.** The first version of the
switch asked the ceiling at every dispatch (`simd::allows`, one load and a
compare). On a `target-cpu=native` build that was a real cost: `origin/main`'s
`is_x86_feature_detected!` is a compile-time constant there, so its dispatchers
were bare calls to the widest kernel. gemma-3-1b's pp512 fell 3-8%
(`Linear m=511 1152x6912 Q5_0`: medians of 16 per-call times 11.81 ms on main,
12.38 with the check, 11.71 with the check put back to the constant). Hoisting
the check to once per op, or keeping the narrow tiers out of line, recovered
part of it, not all. What did: two compilations of the whole interpreter
(`CpuBackend::execute_with::<CAP>`), chosen once per graph from
`infr_core::simd::capped()`. With no ceiling the interpreter is `main`'s code
with every dispatch the constant; under a ceiling every dispatch asks it, on
every thread (`crate::simd` in `infr-cpu`).

Measured 2026-11-17 on Windows, `infr bench --dev cpu`, three binaries
interleaved per round (`main` = `origin/main`, `head` = the branch with the
per-dispatch check, `after5` = the branch as committed), eight rounds of three
reps each, per-round medians in t/s, native builds; no other build, test or GPU
job running (`tasklist`), desktop apps idle:

| Model, run           | `main`                                                         | `head`                                                         | `after5`                                                       |
| -------------------- | -------------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| gemma-3-1b-it, pp512 | 545.6, 579.3, 568.2, 600.5, 575.3, 563.7, 564.5, 559.7         | 560.8, 559.0, 557.2, 583.2, 556.9, 538.5, 560.7, 544.0         | 558.0, 572.1, 593.5, 594.1, 549.5, 561.4, 550.8, 590.4         |
| gemma-3-1b-it, tg64  | 41.0, 41.1, 41.2, 39.2, 41.6, 40.4, 41.2, 41.4                 | 44.9, 45.1, 44.1, 46.2, 43.9, 42.6, 44.9, 46.1                 | 46.4, 46.5, 45.4, 46.3, 46.2, 45.3, 46.0, 46.6                 |
| Qwen3-1.7B, pp512    | 1112.7, 1106.5, 1102.6, 1098.2, 1110.1, 1114.1, 1114.2, 1110.6 | 1100.5, 1108.6, 1139.5, 1097.2, 1129.3, 1126.4, 1079.6, 1076.5 | 1142.1, 1142.2, 1099.1, 1052.8, 1131.6, 1126.9, 1115.1, 1112.8 |
| Qwen3-1.7B, tg64     | 55.8, 55.9, 57.4, 53.6, 56.6, 53.7, 54.3, 55.2                 | 54.3, 56.4, 56.4, 56.9, 56.0, 55.6, 50.5, 55.5                 | 55.0, 57.2, 57.3, 54.9, 57.8, 55.0, 54.2, 54.5                 |

Read the rounds in pairs (each round ran the three builds back to back), because
a criterion on middle values and ranges does not separate these builds: `head`'s
gemma-3-1b pp512 middle value sits inside `main`'s range too. Paired against
`main` on gemma-3-1b pp512, `head` is below in 7 of 8 rounds, a mean difference
of -12.1 t/s (-2.1%); `after5` is below in 5 of 8, a mean difference of +1.6 t/s
(+0.3%), with middle values (the mean of the 4th and 5th of 8) 566.8 against
566.4. On Qwen3-1.7B pp512 `after5` is below in 2 of 8, mean +6.7 t/s (+0.6%).
So the fix shows no measurable cost at this protocol's resolution, which is
about ±2%; a smaller cost would not be seen. Decode is the same on Qwen3 and
faster on gemma in every round. That gemma decode gain is not traced: it shows
only on the native builds (the baseline pair below is level), which points at
code generation or layout under `target-cpu=native` and not at a change in
sampling or the KV quantizer, which would show on both. The baseline builds
(`-C target-cpu=x86-64`, where `main`'s check was a run-time load already), main
against after, same protocol:

| Model, run           | `mainbase`                                             | `afterbase`                                            |
| -------------------- | ------------------------------------------------------ | ------------------------------------------------------ |
| gemma-3-1b-it, pp512 | 387.2, 375.6, 374.5, 369.9, 379.0, 382.4, 370.6, 382.8 | 380.1, 374.4, 376.3, 367.7, 379.4, 371.3, 371.2, 369.7 |
| gemma-3-1b-it, tg64  | 40.4, 40.6, 42.1, 38.7, 40.9, 39.6, 41.6, 41.0         | 40.5, 42.0, 40.2, 40.8, 39.9, 41.3, 41.3, 39.5         |
| Qwen3-1.7B, pp512    | 330.3, 335.0, 335.8, 333.1, 329.6, 328.5, 317.7, 322.1 | 329.8, 327.0, 334.3, 331.1, 330.3, 327.9, 318.5, 321.4 |
| Qwen3-1.7B, tg64     | 47.3, 49.6, 50.9, 51.0, 49.7, 51.1, 49.2, 50.0         | 50.4, 46.7, 47.2, 50.4, 50.1, 51.0, 50.1, 46.7         |

Paired the same way: gemma-3-1b pp512 is below main in 5 of 8 rounds, a mean
difference of -4.0 t/s (-1.1%), middle values 372.9 against 377.3; Qwen3-1.7B
pp512 is below in 6 of 8, mean -1.5 t/s (-0.4%), middle values 328.9 against
329.9; decode level. Both lean toward `main` by less than the protocol resolves,
so a cost of about 1% on baseline builds is neither shown nor ruled out.

## The determinism rule

Inside the module only binary64/binary32 `+ - * /`, `sqrt`, comparisons,
conversions, and integer and bit operations. No `mul_add`, no standard-library
or C-runtime math, no table computed at run time (two are derived at compile
time from generated ones, by the same operations), and no arithmetic that
depends on target features (the tiers above change how many elements run at
once, not what each one computes). IEEE 754 specifies each of those operations
exactly and Rust never contracts or reassociates them, so baseline x86-64,
AVX-512 and aarch64 builds give identical bits.

## How it works

CORE-MATH's design for binary32: evaluate in binary64 with a proven error bound;
when every real number within that bound rounds to the same binary32, that is
the answer; otherwise (rarely) evaluate in double-double, about 100 bits, and
round that. Double-double products use Dekker's splitting, not FMA.

For `exp` and `tanh` the check is not needed and is gone: the ignored test
`fast_path_alone_against_the_correctly_rounded_result` (in
`src/math/fast_path_tests.rs`) walks all 2^32 inputs and finds the binary64
value, rounded straight to binary32, equal to the double-double result on every
one, and it asserts that, so a change to the arithmetic that breaks it fails
there. For the others the check stays; over all 2^32 inputs:

| Function | Check fires on | Binary64 value alone wrong on |
| -------- | -------------: | ----------------------------: |
| `exp`    |              8 |                             0 |
| `tanh`   |             20 |                             0 |
| `ln`     |            100 |                             5 |
| `ln_1p`  |             74 |                            10 |
| `sin`    |            130 |                             2 |
| `cos`    |            140 |                             4 |

Each source file states its algorithm, its sources and its error bounds:

| File        | Functions             | Reduction                                                   |
| ----------- | --------------------- | ----------------------------------------------------------- |
| `exp.rs`    | `exp`, `tanh`         | `x = k ln2/N + r`, table of `2^(j/N)`, `N` 256 (`exp`), 64  |
| `log.rs`    | `ln`, `ln_1p`         | 128-entry reciprocal table on the significand               |
| `trig.rs`   | `sin`, `cos`          | Cody–Waite below `2^22`, Payne–Hanek (bits of `2/pi`) above |
| `pow.rs`    | `powf`                | `exp(y ln x)` in double-double, exact cases by integers     |
| `dd.rs`     | double-double         | error-free transformations, rounding to binary32            |
| `tables.rs` | constants (generated) |                                                             |

## Tests

All in `crates/infr-core/tests/math.rs`:

- `known_answers`: `tests/data/math_kat.txt`, generated by
  `scripts/gen_math_kat.py` with mpmath — special values, subnormals, range
  edges, a spread of inputs, RoPE angles for two model geometries, `powf` exact
  and midpoint cases, and per unary function the hardest inputs found (those
  whose exact result lies nearest a rounding boundary).
- `grid_hashes`: each function over a fixed grid (a spread over all bit patterns
  plus a dense range of the values the backend feeds it), the result bits hashed
  against one checked-in value. CI runs it on Linux, Windows and macOS; one
  value for all three is the cross-platform guarantee.
- `exhaustive_*` (ignored): every binary32 input against an independent
  reference in `tests/math_support/oracle.rs`, which has its own double-double
  arithmetic, its own constants (`ln 2` from a series, `pi` from its published
  hex digits) and different algorithms (no tables; the trigonometric reduction
  by big-integer division). It reports any result it cannot decide rather than
  guessing.
- `every_nan_result_is_the_one_nan`, `helper_special_values`: the NaN rule and
  the helpers' special values against the call sites' own expressions.
- `slices_equal_scalars` (a dense grid at awkward lengths and unaligned starts)
  and `exhaustive_slices_equal_scalars` (ignored, every binary32 input), each at
  every tier the CPU supports (and the active one): they find the supported
  tiers independently of the module, assert they ran each one, and print the
  tiers they skipped. `tier_switch` checks the names and that under each
  `kernels.cpu.max_tier` ceiling the slices run the narrower of it and the CPU's
  widest tier.
- `platform_libm_differences`, `bench_against_std` (ignored): information only.

Run them:

```sh
cargo test -p infr-core --release --test math                         # KAT, grid
cargo test -p infr-core --release --test math -- --ignored exhaustive  # minutes each
cargo test -p infr-core --release --test math -- --ignored --nocapture platform_libm bench
```

`bench_against_std` prints, per slice function, ns per element for the
platform's scalar function written as the call sites wrote it and for every tier
the CPU has. On a Mac (or any aarch64 machine), where nothing has been measured
yet, run it as is: the build has no `target-cpu` flags there, so `baseline` is
the NEON path.

## Regenerating the tables

```sh
python -m venv .venv && .venv/bin/pip install mpmath   # never the system Python
.venv/bin/python -I scripts/gen_math_kat.py tables
MATH_HARD_OUT=hard.txt cargo test -p infr-core --release --test math -- --ignored exhaustive
.venv/bin/python -I scripts/gen_math_kat.py kat --hard hard.txt
```

`tables` must reproduce `tables.rs` byte for byte; a difference means the script
or mpmath changed and has to be explained before it is committed. `kat` takes
candidate hard cases from the exhaustive run (`MATH_HARD_OUT`) and from its own
random scan, and mpmath decides which are hardest. Each expected value is
decided robustly: an exact rational result (only `powf` has them, such as
`11^7`) is rounded with integer arithmetic, and any other value is re-evaluated
at rising precision until it lies clearly on one side of every rounding boundary
(the script refuses rather than guess). Making it robust left the checked-in
table unchanged.

## What the migration changed in the CPU goldens

S2 re-blessed every cached CPU golden in
`crates/infr-llama/tests/cpu_backend.rs` once. Each case now has one hash, the
same natively on Windows and under WSL. Before, a case either had one hash or
one per operating system ("Win / Linux" below). Five of the 17 cases changed
text on each system, every one at a near tie, and each new text is still
coherent.

| Test, case                                            | Before (Win / Linux)                      | After                |
| ----------------------------------------------------- | ----------------------------------------- | -------------------- |
| `cpu_golden_qwen3` (Qwen3-0.6B), France               | `0xfd63781ea3bfa785`                      | unchanged            |
| `cpu_golden_qwen3`, explain a computer                | `0xec2b171ab4126f09 / 0xcf56ba8c4bb5c455` | `0xcf56ba8c4bb5c455` |
| `cpu_golden_qwen3`, ocean                             | `0xfb31f388412112f1 / 0x29f45fb169b84b9a` | `0x92ef4b86905d27dc` |
| `cpu_golden_qwen3_quants`, Q4_K_M (the cached quant)  | `0xfd63781ea3bfa785`                      | unchanged            |
| `cpu_golden_gemma3` (gemma-3-1b), France              | `0xe5a37ab078db3a2c`                      | unchanged            |
| `cpu_golden_gemma3`, knight                           | `0x5147de9a0ddfae50`                      | `0x4f1efdec82f477e5` |
| `cpu_golden_smollm3` (SmolLM3-3B), France             | `0x8bfd779284f134df`                      | unchanged            |
| `cpu_golden_smollm3`, ocean                           | `0x44563169e114b778`                      | unchanged            |
| `cpu_golden_ernie4_5` (ERNIE-4.5-0.3B), France        | `0xfd644a0cebde4e73`                      | unchanged            |
| `cpu_golden_ernie4_5`, ocean                          | `0x1a643a859ae6ff58`                      | `0x055f6196af273214` |
| `cpu_golden_ernie4_5_moe` (ERNIE-4.5-21B-A3B), France | `0x2bdcaa6640a6c155`                      | unchanged            |
| `cpu_golden_ernie4_5_moe`, count to twenty            | `0x181422562a27a9dd`                      | unchanged            |
| `cpu_golden_qwen35` (Qwen3.5-0.8B), France            | `0x542a9dd055c58884`                      | unchanged            |
| `cpu_golden_qwen35`, knight                           | `0xbe06c5580ea33d78`                      | unchanged            |
| `cpu_golden_lfm3` (LFM3-1.7B), France                 | `0x8d3a3dd9844d239f`                      | `0xcaeef40f04920074` |
| `cpu_golden_lfm3`, lighthouse keeper                  | `0x6e5098d3a5bec173`                      | unchanged            |
| `cpu_golden_qwen3moe` (Qwen3-30B-A3B), France         | `0xe1a4ed6cecc5eb14 / 0xbc4f22b22d3e3c1d` | `0xe1a4ed6cecc5eb14` |

Plan 11's S4 then changed one case once more: the AVX-512 Q8_0 kernels
associated their block sums differently from the scalar and AVX2 ones (see
`docs/plan/11-cpu-bit-identical.md`), and making them compute the scalar
expression changed Qwen3.5-0.8B's output on AVX-512 machines, the same natively
on Windows and under WSL. Both texts stay coherent; each forks at its first near
tie:

| Test, case                  | Before               | After                |
| --------------------------- | -------------------- | -------------------- |
| `cpu_golden_qwen35`, France | `0x542a9dd055c58884` | `0x01829861302f0b8a` |
| `cpu_golden_qwen35`, knight | `0xbe06c5580ea33d78` | `0x0a0d2a6554ca9f21` |

France: "**Analyze the Request:** The user is asking for the capital of France."
became "**Analyze the Request:**\n \* Question: …"; knight: "**Topic:** A short
story about a brave knight." became "**Topic:** A brave knight.". The new knight
hash is the value a CI runner printed when the golden was dropped from CI as not
portable (`docs/synthetic-models.md`). The France hash does not match that run:
France passed there with the old value, so a July runner and this code disagree
on it (much else has changed since, the math migration and RoPE among it), and
the Q8_0 association is not established as the July cause. The golden's first
Linux CI run after this change is the real test.

How the changed texts fork, for example: Qwen3-0.6B's ocean paragraph went from
"brainstorming the key elements … Maybe use words like" to "brainstorming key
elements … Maybe compare it to"; gemma's knight story keeps its opening line;
LFM3's answer opens "We need to answer the question:" where it opened "The user
asks:".

## Measured (2026-11-17, Ryzen 9 9950X3D, AVX-512)

**Exhaustive**, Windows, release build, 32 threads: for each function every one
of the 4,294,967,296 inputs correctly rounded, none wrong, none undecided, every
zero result with the reference's sign.

| Function | Time  |
| -------- | ----- |
| `exp`    | 76 s  |
| `ln`     | 72 s  |
| `ln_1p`  | 71 s  |
| `tanh`   | 78 s  |
| `sin`    | 559 s |
| `cos`    | 543 s |

(`sin`/`cos` are slow because the reference reduces by big-integer long
division.) After S2's changes to `exp`'s binary64 value and `ln_1p`'s branch,
`exp` and `ln_1p` were re-run: all 2^32 correctly rounded again (108.5 and 106.9
s, at a higher load), and the fast-path test still finds `exp`'s binary64 value
alone right on every input. The slice identity test found no element differing
from the scalar function on any of the 2^32 inputs, for every slice function at
every tier (baseline, AVX2, AVX-512), on Windows native and baseline builds and
under WSL (native and baseline builds, every tier).

**The grid hash**, now over the real result bits with NaN inputs of every kind
in the grid, and over the slices too, is the same on Windows (MSVC,
`target-cpu=native`, debug and release, and a baseline `x86-64` release build)
and on Linux under WSL (glibc 2.43, `target-cpu=native` and baseline, release).

**How far the platform libraries are from it** on the grid
(`platform_libm_differences`, measured before the grid gained its special
inputs); every difference is 1 ulp. These are the outputs that move when the
call sites migrate:

| Function | Windows (UCRT) | Linux (glibc 2.43) |
| -------- | -------------: | -----------------: |
| `exp`    |             10 |                 67 |
| `ln`     |            253 |                607 |
| `ln_1p`  |         13,212 |                  0 |
| `tanh`   |          2,938 |                  0 |
| `sin`    |            261 |              2,627 |
| `cos`    |            252 |              2,581 |
| `powf`   |             13 |                 11 |

out of 300,000 inputs per unary function and 200,000 `powf` pairs.

**Speed** (`bench_against_std`, after S2): one thread, release, median ns per
call (or per element, for slices) over interleaved rounds, against the C
runtime's `f32` function (for slices: a plain loop applying it in the call
sites' expression). "Native" is a `target-cpu=native` build, "base" a baseline
`x86-64` build; the slice columns are the tiers, and a base build at `avx512` is
what a released binary runs on this CPU. Windows CPU load 1-3% around the runs;
WSL load average 0.00 at the start.

Per call:

| Function (inputs)       | Win native | UCRT | Win base | UCRT | WSL native | glibc | WSL base | glibc |
| ----------------------- | ---------: | ---: | -------: | ---: | ---------: | ----: | -------: | ----: |
| `exp` (softmax, -20..0) |       2.34 | 1.40 |     2.62 | 1.38 |       2.39 |  1.13 |     2.77 |  1.15 |
| `ln` (1e-6..4)          |       6.14 | 1.78 |     5.80 | 1.79 |       6.24 |  1.39 |     6.15 |  1.32 |
| `ln_1p` (-0.9..4)       |       6.81 | 2.73 |     6.41 | 2.75 |       6.91 |  2.42 |     6.53 |  2.43 |
| `tanh` (-6..6)          |       4.53 | 4.86 |     4.93 | 4.88 |       4.52 |  2.79 |     5.06 |  2.79 |
| `sin` (RoPE angles)     |       7.67 | 3.49 |     8.03 | 3.47 |       7.66 |  2.12 |     7.66 |  2.19 |
| `cos` (RoPE angles)     |       7.69 | 3.46 |     8.29 | 3.33 |       7.59 |  2.26 |     7.96 |  2.37 |
| `silu` (-6..6)          |       3.30 | 2.22 |     4.37 | 2.23 |       3.33 |  1.71 |     4.38 |  1.92 |
| `powf` (RoPE `theta^e`) |      21.78 | 4.08 |    22.87 | 4.14 |      22.34 |  2.88 |    22.98 |  2.80 |

Per slice element, by tier:

| Slice             | Win base: UCRT | baseline | avx2 | avx512 | WSL base: glibc | baseline | avx2 | avx512 |
| ----------------- | -------------: | -------: | ---: | -----: | --------------: | -------: | ---: | -----: |
| `exp_slice`       |           1.10 |     1.13 | 0.77 |   0.62 |            1.05 |     1.17 | 0.80 |   0.64 |
| `exp_sub_slice`   |           1.10 |     1.20 | 0.83 |   0.65 |            1.06 |     1.26 | 0.87 |   0.67 |
| `tanh_slice`      |           4.25 |     2.38 | 1.28 |   0.87 |            2.32 |     2.48 | 1.33 |   0.90 |
| `sigmoid_slice`   |           1.15 |     1.43 | 0.91 |   0.70 |            1.17 |     1.50 | 0.94 |   0.73 |
| `silu_slice`      |           1.15 |     1.43 | 0.91 |   0.70 |            1.19 |     1.49 | 0.95 |   0.73 |
| `gelu_tanh_slice` |           4.74 |     2.63 | 1.41 |   0.97 |            3.50 |     2.76 | 1.47 |   1.13 |

In a native build every tier is the native code (the build's own features are
already wide; the columns differ only in loop shape): Windows `exp_slice`
0.63-0.65, `silu_slice` 0.70-0.73, `gelu_tanh_slice` 1.04-1.16 per element; WSL
0.65-0.67, 0.73-0.77, 1.03-1.11.

**The two-lane path against the C runtime.** How the tuning moved the baseline
tier (Windows, baseline build, ns per element; the C runtime's loop 1.09-1.11
for `exp`, 1.15-1.18 for `sigmoid`/`silu`; each row's bits identical to the row
above, by the exhaustive tests where the row was kept):

| Variant                                                                | `exp_slice` | `sigmoid_slice` | `gelu_tanh_slice` |
| ---------------------------------------------------------------------- | ----------: | --------------: | ----------------: |
| S1: 64 entries, degree 6 by Horner, `2^k` by a multiply                |        1.76 |            2.55 |              3.60 |
| 64 entries, degree 6 by Estrin, `2^k` built in the table's bits        |        1.48 |            2.17 |              3.65 |
| 256 entries, degree 4 by Estrin (the loop vectorizer leaves it scalar) |        2.20 |            1.94 |              3.58 |
| the same, blocks of 2 for the SLP vectorizer                           |        1.27 |            1.91 |              3.49 |
| the same, blocks of 4 (kept)                                           |        1.14 |            1.97 |              3.54 |
| blocks of 8                                                            |        2.22 |            1.98 |              3.62 |
| blocks of 4, helpers in two passes (kept)                              |        1.15 |            1.46 |              2.69 |

(Single runs at 1-5% load; the kept numbers are re-measured in the table above.)
So at baseline `exp` is at the C runtime's speed (1.13-1.17 against 1.05-1.10,
3-11% slower), and `sigmoid`/`silu` are 24-28% slower than the C runtime's loop,
which needs half the accuracy; `gelu_tanh` is faster than both C runtimes,
`tanh` faster than UCRT and 7% slower than glibc 2.43. The remaining gap is the
two-lane binary64 arithmetic that correct rounding needs (about twice the
polynomial work of `expf`'s) and the table lookups, which stay scalar without a
gather. On x86-64 this tier only runs on a CPU without AVX2; on aarch64 it is
the NEON path, not measured.

**What it costs a forward** (per-op profile, `INFR_PROF_OPS=1`, one pp512
forward of each warm rep, two per build, 16 threads, Windows at 1-3% load; the
same binaries interleaved; "forced" is the baseline build with the math slices
forced to their baseline tier, the no-wide-path case, by the switch
`kernels.cpu.max_tier` has since replaced), in ms:

| Model, build                | `GatedAct` before |       after | `QkNormRope` before |     after | forward before |         after |
| --------------------------- | ----------------: | ----------: | ------------------: | --------: | -------------: | ------------: |
| Qwen3-1.7B Q4_K_M, native   |       17.37-18.14 | 16.28-16.57 |           9.14-9.35 | 7.89-8.62 |    450.1-455.3 |   443.1-448.1 |
| Qwen3-1.7B, baseline build  |       17.90-18.73 | 15.23-15.94 |           8.96-9.43 | 8.53-8.88 |  1515.2-1558.2 | 1490.2-1569.5 |
| Qwen3-1.7B, forced baseline |       17.90-18.73 | 17.91-18.64 |           8.96-9.43 | 8.56-8.84 |  1515.2-1558.2 | 1498.6-1550.6 |
| gemma-3-1b Q4_K_M, native   |       46.52-47.77 | 18.53-18.99 |           7.05-7.17 | 3.95-4.15 |    844.8-866.6 |   831.5-846.8 |
| gemma-3-1b, baseline build  |       47.87-48.44 | 18.36-18.83 |           5.73-6.30 | 4.33-4.85 |  1352.7-1392.8 | 1311.5-1374.5 |
| gemma-3-1b, forced baseline |       47.87-48.44 | 25.25-25.35 |           5.73-6.30 | 4.29-4.59 |  1352.7-1392.8 | 1338.7-1366.0 |

`Attention` did not move measurably (Qwen3 80.8-85.9 ms native before and after;
its `exp` was about 5% of it). The baseline build is three times slower than
native for reasons outside this module (attention's dots and the other
autovectorized loops). Whole-prefill throughput on Qwen3-1.7B did not change
measurably: `infr bench` pp512, six interleaved rounds of three reps, gave
976-1143 t/s before and 1124-1159 after, and the S2 reviewer's nine interleaved
rounds gave 1075-1152 before and 960-1138 after. The reviewer measured
gemma-3-1b decode at 39.4-40.8 t/s before and 43.5-44.7 after, with its
`GatedAct` going from about 1.8 to 0.73 ms per call in prefill and from 52 to
11.7 µs in decode. (An earlier six-round tg128 run on gemma saw 37.9-39.4 before
and 37.2-39.4 after; it was a different window, so its figures do not compare
with the reviewer's.)

**Decode RoPE, and the RoPE frequency cache.** S2 first computed the RoPE
frequencies (one correctly rounded `powf` per pair, about 22 ns against the C
runtime's 4) once per op, which a one-row decode op never amortizes: per-call
`QkNormRope` time went up. The CPU backend now keeps one table per distinct
`(theta, rope_dim)` for its whole life (`CpuBackend::rope_freqs`), so decode
pays no `powf` at all. Per-call decode time (`INFR_PROF_OPS=1`,
`infr bench -p 0 -n 64 -r 2`, native builds, four interleaved rounds, no other
build or test process running; total CPU 33% on average including the bench's
own threads), in µs:

| Model, op                  | `origin/main` | S2 before the cache | with the cache |
| -------------------------- | ------------: | ------------------: | -------------: |
| gemma-3-1b, `QkNormRope`   |       4.0-4.1 |             5.6-5.7 |        2.3-2.4 |
| Qwen3-0.6B, `QkNormRope`   |           3.3 |             4.0-4.1 |            2.5 |
| Qwen3.5-0.8B, `QkNormRope` |       3.3-3.5 |             3.6-3.9 |        2.7-2.9 |
| Qwen3.5-0.8B, `DeltaNet`   |   112.9-117.5 |         121.3-124.8 |      40.3-41.4 |

`DeltaNet` was 5-10% slower per decode call on the S2 branch than on
`origin/main` (reproduced over 6 and 5 interleaved rounds at tg128). Moving the
three gate functions out of the per-head task changed nothing (128.7-133.7 µs
against 129.6-133.1 without the move, same rounds), so the gate math was not the
cause, and no codegen cause was found. The call was dominated by copying the
recurrent state, not by arithmetic: each head scanned a copy of its state and a
serial gather copied every head back, two copies of the whole state per call.
`deltanet_scan` now scans each head's state slice where it lies, the same
operations in the same order (its bit-identity test against the serial reference
passes), and a decode call takes about a third of the time. Qwen3.5-0.8B tg64
went from 56.2-56.9 t/s (`origin/main`) and 57.2-58.2 (S2 before) to 64.7-65.3.
gemma-3-1b tg64 in the same rounds: 35.5-36.6, 39.2-39.7 and 37.4-38.0; the last
build's `Linear m=1 1152x6912 Q5_0` took 280-282 µs per call against 249-259 in
the other two, a kernel no change here touches (see `docs/backlog.md`).
