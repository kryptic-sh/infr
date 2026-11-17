//! `exp`, `tanh`, and the `e^x` core they share with `powf`.
//!
//! **Method** (the table-driven reduction of Tang, "Table-driven implementation of the
//! exponential function in IEEE floating-point arithmetic", ACM TOMS 15, 1989): write
//! `x = k*ln(2)/64 + r` with `k = round(64x/ln 2)` and `|r| <= ln(2)/128 + 2^-60`, so that
//! `e^x = 2^(k>>6) * 2^((k&63)/64) * e^r`. The 64 values `2^(j/64)` are a double-double table;
//! `ln(2)/64` is split into three parts whose first two have 39 bits, so `k*part` is exact for
//! the `|k| < 2^14` every caller produces. `e^r - 1` is its Taylor polynomial.
//!
//! **The binary64 value's error.** `x - k*C1` is exact (`k*C1` is exact and within a factor 2
//! of `x` when `k != 0`); subtracting `k*C2` rounds once (relative `2^-53` of `|r|`); `k*C3`,
//! omitted, is below `2^-70`. The degree-6 Taylor polynomial of `e^r - 1` leaves
//! `|r|^7/7! < 2^-65`, and evaluating it costs at most `2.1 * 2^-53` relative of `|q| < 0.0055`.
//! `T + (T*q + T_lo)` then rounds once more (`2^-53` relative), and `2^(k>>6)` is exact. The
//! total relative error is below `1.1 * 2^-53`.
//!
//! **Why `exp` and `tanh` round that value directly.** The CORE-MATH structure checks whether
//! every real number within the error bound rounds to the same binary32 and otherwise takes a
//! double-double path. For these two functions the check is not needed: the exhaustive test
//! `fast_path_alone_against_the_correctly_rounded_result` (in `fast_path_tests.rs`, ignored by
//! default) walked all `2^32` inputs and found the binary64 value rounded straight to binary32
//! equal to the double-double path's correctly rounded result on every one (the check would
//! have fired on 8 inputs of `exp` and 20 of `tanh`, each of which the fast value got right
//! anyway). The double-double paths ([`exp_accurate`], [`tanh_accurate`]) stay, compiled for
//! tests only, as the reference that test compares against; `tests/math.rs` checks the
//! functions themselves against an independent reference. Rounding is then a plain conversion,
//! which is what lets the slice versions vectorize. Any change to the arithmetic here must
//! re-run that exhaustive test.
//!
//! **Accurate path.** The same reduction carried in double-double (`k*C3` included) and the
//! Taylor polynomial to degree 11 (remainder below `2^-119`): relative error below `2^-100`,
//! rounded to binary32 by [`dd_to_f32`]. `powf` uses it.
//!
//! **Branch-free lanes.** [`exp_lane`] and [`tanh_lane`] are written without data-dependent
//! control flow: out-of-range and NaN inputs are clamped to a harmless value before the
//! arithmetic, and the special results are chosen afterwards by comparison. The scalar
//! functions ARE the lanes, and the slice functions call the same lanes in a loop the compiler
//! can vectorize, so slice and scalar agree by construction.

#[cfg(test)]
use super::dd::{dd_to_f32, exp2i};
use super::dd::{horner, horner_dd, pow2, two_sum, Dd};
use super::tables::{EXP2_J64, INV_FACT, INV_LN2_64, LN2_64_1, LN2_64_2, LN2_64_3};
use super::NAN;

/// `1.5 * 2^52`: adding and subtracting it rounds a binary64 of magnitude below `2^51` to the
/// nearest integer (ties to even).
pub(super) const SHIFT: f64 = 6_755_399_441_055_744.0;

/// The error allowance a rounding check would use for `exp` (see the module doc): kept for the
/// exhaustive test that shows the check is never needed.
#[cfg(test)]
pub(super) const EXP_EPS: f64 = exp2i(-50);
/// Likewise for `tanh`: the polynomial's error plus one addition and one division.
#[cfg(test)]
pub(super) const TANH_EPS: f64 = exp2i(-48);

/// `x > EXP_OVERFLOW` gives `+inf`: `e^89 > 2^128`.
const EXP_OVERFLOW: f32 = 89.0;
/// `x < EXP_UNDERFLOW` gives `+0`: `e^-104 < 2^-150`, half the least subnormal.
const EXP_UNDERFLOW: f32 = -104.0;
/// `|x| > TANH_ONE` gives `±1`: `1 - tanh(9.1) = 2/(e^18.2 + 1) < 2^-25`, under half the spacing
/// of binary32 below 1.
pub(super) const TANH_ONE: f32 = 9.1;

/// `k = round(64x/ln 2)` and `r = x - k*ln(2)/64` to binary64 accuracy (`k*C3` omitted).
#[inline(always)]
fn reduce(x: f64) -> (i32, f64) {
    let t = x * INV_LN2_64 + SHIFT;
    let kf = t - SHIFT;
    let r = (x - kf * LN2_64_1) - kf * LN2_64_2;
    // `t`'s significand holds `2^51 + k` exactly; its low 32 bits are `k` in two's complement.
    (t.to_bits() as i32, r)
}

/// `e^r - 1` for `|r| <= ln(2)/128`, degree 6.
#[inline(always)]
fn expm1_poly(r: f64) -> f64 {
    r * horner(r, &INV_FACT[1..7])
}

/// `e^x` for binary32 `x`, correctly rounded (ties to even); `+inf` above the overflow
/// threshold, `+0` below the underflow one, [`NAN`](super::NAN) for any NaN. See the module
/// doc.
#[inline]
pub fn exp(x: f32) -> f32 {
    exp_lane(x)
}

/// [`exp`], branch-free (see the module doc).
#[inline(always)]
pub(super) fn exp_lane(x: f32) -> f32 {
    // Clamp into the range the arithmetic handles (a NaN fails the first comparison and becomes
    // the lower end). The clamped ends produce the special results by themselves: e^89 rounds
    // to +inf and e^-104 to +0 in binary32, so beyond them nothing needs choosing. Only NaN is
    // selected afterwards; with no other select, the compiler keeps the loop free of branches
    // and vectorizes it even for baseline SSE2.
    let xc = if x >= EXP_UNDERFLOW { x } else { EXP_UNDERFLOW };
    let xc = if xc > EXP_OVERFLOW { EXP_OVERFLOW } else { xc };
    let y = exp_fast(xc as f64) as f32;
    if x.is_nan() {
        NAN
    } else {
        y
    }
}

/// `exp`'s binary64 value for `EXP_UNDERFLOW <= x <= EXP_OVERFLOW`.
#[inline(always)]
pub(super) fn exp_fast(x: f64) -> f64 {
    let (k, r) = reduce(x);
    let (th, tl) = EXP2_J64[(k & 63) as usize];
    (th + (th * expm1_poly(r) + tl)) * pow2(k >> 6)
}

/// `exp`'s double-double path: the correctly rounded result the exhaustive test compares the
/// binary64 value against.
#[cfg(test)]
pub(super) fn exp_accurate(x: f64) -> f32 {
    let (k, q) = exp_parts_dd(Dd::new(x, 0.0));
    dd_to_f32(exp_from_parts(k, q))
}

/// `e^x` for a double-double argument in binary64 precision, for `powf`'s fast path. The
/// argument's low part enters `r`; its error budget is in `pow.rs`.
#[inline]
pub(super) fn exp_dd_fast(x: Dd) -> f64 {
    let kf = (x.hi * INV_LN2_64 + SHIFT) - SHIFT;
    let r = ((x.hi - kf * LN2_64_1) - kf * LN2_64_2) + x.lo;
    let k = kf as i32;
    let (th, tl) = EXP2_J64[(k & 63) as usize];
    (th + (th * expm1_poly(r) + tl)) * pow2(k >> 6)
}

/// `k` and `q = e^r - 1` in double-double, where `x = k*ln(2)/64 + r`.
pub(super) fn exp_parts_dd(x: Dd) -> (i32, Dd) {
    let kf = (x.hi * INV_LN2_64 + SHIFT) - SHIFT;
    // Exact: k*C1 and k*C2 are exact products and x.hi - k*C1 is exact (Sterbenz).
    let r = two_sum(x.hi - kf * LN2_64_1, -(kf * LN2_64_2))
        .add_f(x.lo)
        .add_f(-(kf * LN2_64_3));
    let q = r.mul(horner_dd(r, &INV_FACT[1..12]));
    (kf as i32, q)
}

/// `2^(k>>6) * 2^((k&63)/64) * (1 + q)`.
pub(super) fn exp_from_parts(k: i32, q: Dd) -> Dd {
    let t = Dd::from_pair(EXP2_J64[(k & 63) as usize]);
    t.add(t.mul(q)).scale(k >> 6)
}

/// `e^t - 1` for `0 <= t <= 2*TANH_ONE`, relative error below `2^-52`. `2^e*T - 1` is exact or
/// rounds once (Sterbenz covers `2^e*T` in `[1/2, 2]`, where cancellation could otherwise hurt)
/// and for `k != 0` the result is at least `e^(ln(2)/128) - 1`, so the absolute errors stay
/// relative. At `k = 0` (`T = 1`, `T_lo = 0`, `2^e = 1`) the expression is exactly `q`: no branch
/// is needed for the small arguments.
#[inline(always)]
fn expm1_fast(t: f64) -> f64 {
    let (k, r) = reduce(t);
    let q = expm1_poly(r);
    let s = pow2(k >> 6);
    let (th, tl) = EXP2_J64[(k & 63) as usize];
    (s * th - 1.0) + s * (th * q + tl)
}

/// `e^t - 1` in double-double for `0 <= t <= 2*TANH_ONE`: relative error below `2^-97` (for
/// `k != 0` the absolute error `2^-104` of `T*(1+q)` meets a result of at least `2^-7.5`).
#[cfg(test)]
fn expm1_dd(t: f64) -> Dd {
    let (k, q) = exp_parts_dd(Dd::new(t, 0.0));
    if k == 0 {
        return q;
    }
    exp_from_parts(k, q).add_f(-1.0)
}

/// `tanh(x)` for binary32 `x`, correctly rounded (ties to even): `tanh(|x|) = m/(m + 2)` with
/// `m = e^(2|x|) - 1`, odd in `x`; `±1` beyond `TANH_ONE`; `±0` at `±0`;
/// [`NAN`](super::NAN) for any NaN.
///
/// The binary64 value's error: `expm1_fast`'s `2^-52`, scaled by `2/(m+2) <= 1` through the
/// quotient, plus one rounding each for `m + 2` and the division: below `2^-51` relative. It is
/// rounded directly; the module doc says why that is exact. `tanh(x)` is irrational for
/// binary32 `x != 0`, so no input is a rounding boundary.
#[inline]
pub fn tanh(x: f32) -> f32 {
    tanh_lane(x)
}

/// [`tanh`], branch-free.
#[inline(always)]
pub(super) fn tanh_lane(x: f32) -> f32 {
    let sign = x.to_bits() & 0x8000_0000;
    let a = f32::from_bits(x.to_bits() & 0x7fff_ffff);
    // Clamped at TANH_ONE, where the value already rounds to 1, so larger inputs (infinity
    // included) need no select; a NaN fails `a <= TANH_ONE`, is computed as TANH_ONE, and is
    // replaced below.
    let ac = if a <= TANH_ONE { a } else { TANH_ONE };
    let v = tanh_fast(2.0 * ac as f64) as f32;
    // tanh is odd and tanh(+0) = +0, so the input's sign bit is the result's (±0 included).
    let v = f32::from_bits(v.to_bits() | sign);
    if x.is_nan() {
        NAN
    } else {
        v
    }
}

/// `tanh`'s binary64 value of `tanh(t/2)` for `0 <= t <= 2*TANH_ONE`.
#[inline(always)]
pub(super) fn tanh_fast(t: f64) -> f64 {
    let m = expm1_fast(t);
    m / (m + 2.0)
}

/// `tanh`'s double-double path for `t = 2|x|`: the reference the exhaustive test compares the
/// binary64 value against.
#[cfg(test)]
pub(super) fn tanh_accurate(t: f64) -> f32 {
    let m = expm1_dd(t);
    dd_to_f32(m.div(m.add_f(2.0)))
}
