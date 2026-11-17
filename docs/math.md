# `infr_core::math`: the same bits on every platform

`f32::exp`, `ln`, `ln_1p`, `tanh`, `powf`, `sin` and `cos` lower to the C
runtime, and the C runtimes disagree in the last bit: UCRT on Windows against
glibc on Linux, one glibc version against another, and so on. Every CPU output
computed through one of them inherits the disagreement, which is why some CPU
goldens carry one hash per operating system. `infr_core::math` replaces them
with functions that return the **correctly rounded** binary32 result — the one
nearest the exact mathematical value, ties to even — so there is exactly one
right answer, the same everywhere, and tests can take it from a multi-precision
library instead of from the code under test. The plan this serves is
[`plan/11-cpu-bit-identical.md`](plan/11-cpu-bit-identical.md).

As of this writing the module exists and is tested; **no call site uses it yet**
(that is slice S2, which changes every CPU output once). The lint
`crates/infr-core/tests/no_platform_libm.rs` counts the call sites still on the
platform functions, per file.

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
on all 2^32 inputs, at `target-cpu=native` and baseline `x86-64` on both
systems, and every CI system hashes the slices' output on the grid. A call site
whose expression is not one of these forms (`(v * scale - max).exp()`, say)
computes the argument into a buffer first: the binary32 operations are the same,
so the bits are too.

What makes them vectorize: no data-dependent control flow. Out-of-range inputs
are clamped before the arithmetic and the clamped ends produce the special
results by themselves (`e^89` rounds to `inf`, `e^-104` to 0, `tanh(9.1)` to 1);
only NaN is selected afterwards. An earlier version that selected `inf` and 0
after the fact compiled to a branch around the arithmetic and did not vectorize
at baseline at all. `ln`, `ln_1p`, `sin`/`cos` and `powf` have no slice form:
none is on a per-element hot path.

## The determinism rule

Inside the module only binary64/binary32 `+ - * /`, `sqrt`, comparisons,
conversions, and integer and bit operations. No `mul_add`, no standard-library
or C-runtime math, nothing that depends on target features, no table computed at
run time. IEEE 754 specifies each of those operations exactly and Rust never
contracts or reassociates them, so baseline x86-64, AVX-512 and aarch64 builds
give identical bits.

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
| `exp.rs`    | `exp`, `tanh`         | `x = k ln2/64 + r`, table of `2^(j/64)`                     |
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
  and `exhaustive_slices_equal_scalars` (ignored, every binary32 input).
- `platform_libm_differences`, `bench_against_std` (ignored): information only.

Run them:

```sh
cargo test -p infr-core --release --test math                         # KAT, grid
cargo test -p infr-core --release --test math -- --ignored exhaustive  # minutes each
cargo test -p infr-core --release --test math -- --ignored --nocapture platform_libm bench
```

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
division.) The slice identity test found no element differing from the scalar
function on any of the 2^32 inputs, for every slice function, on Windows and
WSL, native and baseline builds.

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

**Speed** (`bench_against_std`): one thread, release, median ns per call (or per
element, for slices) over interleaved rounds, against the C runtime's `f32`
function (for slices: a plain loop applying it in the call sites' expression).
Windows CPU load 0-5% around the runs; the WSL runs started at a load average of
10.5 inside the VM, just after their compile.

Per call:

| Function (inputs)       | Win native | UCRT | Win base | UCRT | WSL native | glibc | WSL base | glibc |
| ----------------------- | ---------: | ---: | -------: | ---: | ---------: | ----: | -------: | ----: |
| `exp` (softmax, -20..0) |       3.37 | 1.60 |     3.65 | 1.37 |       3.47 |  1.67 |     3.79 |  1.66 |
| `ln` (1e-6..4)          |       6.11 | 1.77 |     5.79 | 1.76 |       6.36 |  1.70 |     6.00 |  1.67 |
| `ln_1p` (-0.9..4)       |       6.74 | 2.76 |    16.62 | 2.69 |       6.91 |  2.46 |    17.40 |  2.40 |
| `tanh` (-6..6)          |       4.51 | 4.86 |     4.84 | 4.77 |       4.60 |  2.80 |     5.04 |  2.76 |
| `sin` (RoPE angles)     |       7.62 | 3.60 |     7.88 | 3.46 |       7.89 |  2.38 |     7.64 |  2.34 |
| `cos` (RoPE angles)     |       7.71 | 3.55 |     8.12 | 3.35 |       7.77 |  2.59 |     7.96 |  2.53 |
| `silu` (-6..6)          |       4.42 | 2.39 |     5.44 | 1.92 |       4.62 |  1.94 |     5.61 |  2.00 |
| `powf` (RoPE `theta^e`) |      22.15 | 4.11 |    22.73 | 4.08 |      22.91 |  2.86 |    23.54 |  2.80 |

Per slice element:

| Slice             | Win native | UCRT | Win base | UCRT | WSL native | glibc | WSL base | glibc |
| ----------------- | ---------: | ---: | -------: | ---: | ---------: | ----: | -------: | ----: |
| `exp_slice`       |       0.75 | 1.10 |     1.77 | 1.09 |       0.76 |  1.03 |     1.80 |  1.03 |
| `exp_sub_slice`   |       0.77 | 1.11 |     1.87 | 1.10 |       0.81 |  1.14 |     1.93 |  1.06 |
| `tanh_slice`      |       0.86 | 4.30 |     2.34 | 4.28 |       0.91 |  2.37 |     2.43 |  2.32 |
| `sigmoid_slice`   |       0.89 | 1.17 |     2.56 | 1.20 |       0.94 |  1.23 |     2.66 |  1.17 |
| `silu_slice`      |       0.88 | 1.16 |     2.55 | 1.21 |       0.95 |  1.24 |     2.65 |  1.19 |
| `gelu_tanh_slice` |       1.21 | 4.72 |     3.62 | 4.72 |       1.28 |  3.58 |     3.77 |  3.49 |

Vectorization is confirmed by these numbers: a slice element costs a fifth to a
quarter of a scalar call natively (AVX-512: 8 binary64 lanes) and about half at
baseline (SSE2: 2 binary64 lanes, and no gather, so the table lookups stay
scalar). Natively every slice beats the C runtime; at baseline only `tanh` and
`gelu_tanh` do, and `exp`/`sigmoid`/`silu` are 1.6-2.2 times the C runtime's
`expf`. The scalar functions are 2-4 times the C runtimes (8 for `powf`, called
once per RoPE frequency, not per element). `ln_1p`'s baseline scalar time (16.6
and 17.4 ns, against 6.7-6.9 natively, on both systems) is unexplained and not
investigated; `ln_1p` is not on a hot path.

**Expected effect of the migration (an estimate, not a measurement).** From
B173's profile of one Qwen3-1.7B pp512 forward at `target-cpu=native` on this
machine (435.4 ms, 16 threads): `GatedAct` 18.17 ms, and `exp` about 5% of
`Attention`'s 77.63 ms (taking `exp` out of a single-threaded layer saved 1.87
of 34.49 ms), about 4 ms. If both are exp-bound and scale with the per-element
times above:

- Scalar functions at the call sites: `GatedAct` about +85% (4.42 against 2.39
  ns per SiLU), attention's `exp` about +110%: up to about +20 ms, +4.6% per
  forward.
- Slices where the shape allows: `GatedAct` about -24% (0.88 against 1.16),
  attention's `exp` about -30% (0.77 against 1.11): roughly -6 ms, -1.4%.
- At baseline (the build CI and a portable release use), slices: `GatedAct`
  about +110% and attention's `exp` about +70%: roughly +23 ms, +5%.

So the migration should use the slice forms for SiLU/GELU and the softmaxes; a
baseline build will still be slower than today until a wider vector path or the
runtime-dispatched native build (plan slice S6) covers it. Attention's exp is
not a plain loop today (it runs inside the blocked q·k loop), so whether it can
take `exp_sub_slice` is for S2 to see.
