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
binary64 lanes where the CPU has them. FMA is not among the enabled features, so
no tier could contract `a*b + c` even if Rust did (it does not: it never fuses
or reassociates floating-point operations, with or without a target feature).
The tiers are the same source performing the same operations per element in the
same order, so they give the same bits by construction; the tests below prove it
on every input.

**Forcing a tier.** `kernels.cpu.math_tier` (`INFR_CPU_MATH_TIER`: `baseline`,
`avx2`, `avx512`) makes every slice run at that tier, process-wide, from the
moment a CPU backend is built. A tier the CPU lacks is refused (the backend
panics with the reason; a misspelt `INFR_CPU_MATH_TIER` is a configuration
error). It changes only speed; it exists so the two-lane path every aarch64
machine and every pre-AVX2 x86 CPU runs can be measured and tested on a machine
with wider ones. In a `target-cpu=native` build `baseline` is the native code,
so a baseline measurement needs a baseline build
(`RUSTFLAGS="-C target-cpu=x86-64"`).

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
  tiers they skipped. `tier_switch` checks the names and that an unsupported
  tier is refused.
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
same binaries interleaved; "forced" is the baseline build with
`INFR_CPU_MATH_TIER=baseline`, the no-wide-path case), in ms:

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
autovectorized loops). `infr bench` pp512 on Qwen3-1.7B native, six interleaved
rounds of three reps with no other build or test running: 976-1143 t/s before,
1124-1159 after. Decode (tg128) moved within its noise on both models:
gemma-3-1b native 37.9-39.4 before, 37.2-39.4 after, six interleaved rounds.
