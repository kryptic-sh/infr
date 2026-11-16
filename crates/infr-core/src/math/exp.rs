//! `exp`, `tanh`, and the `e^x` core they share with `powf`.
//!
//! **Method** (the table-driven reduction of Tang, "Table-driven implementation of the
//! exponential function in IEEE floating-point arithmetic", ACM TOMS 15, 1989, with CORE-MATH's
//! fast-path-then-accurate-path structure for binary32): write
//! `x = k*ln(2)/64 + r` with `k = round(64x/ln 2)` and `|r| <= ln(2)/128 + 2^-60`, so that
//! `e^x = 2^(k>>6) * 2^((k&63)/64) * e^r`. The 64 values `2^(j/64)` are a double-double table;
//! `ln(2)/64` is split into three parts whose first two have 39 bits, so `k*part` is exact for
//! the `|k| < 2^14` every caller produces. `e^r - 1` is its Taylor polynomial.
//!
//! **Fast path error bound.** `x - k*C1` is exact (`k*C1` is exact and within a factor 2 of
//! `x` when `k != 0`); subtracting `k*C2` rounds once (relative `2^-53` of `|r|`); `k*C3`,
//! omitted, is below `2^-70`. The degree-6 Taylor polynomial of `e^r - 1` leaves
//! `|r|^7/7! < 2^-65`, and evaluating it costs at most `2.1 * 2^-53` relative of `|q| < 0.0055`.
//! `T + (T*q + T_lo)` then rounds once more (`2^-53` relative), and `2^(k>>6)` is exact. The
//! total relative error is below `1.1 * 2^-53`; the fast path accepts a result only when every
//! real number within [`EXP_EPS`] `= 2^-50` relative of it rounds to the same binary32.
//!
//! **Accurate path.** The same reduction carried in double-double (`k*C3` included) and the
//! Taylor polynomial to degree 11 (remainder below `2^-119`): relative error below `2^-100`,
//! rounded to binary32 by [`dd_to_f32`]. `e^x` is irrational for every binary32 `x` but 0
//! (Lindemann), so no input is an exact rounding boundary; the exhaustive test confirms that
//! `2^-100` separates every binary32 input from its nearest boundary.

use super::dd::{dd_to_f32, exp2i, horner, horner_dd, pow2, round_if_clear, two_sum, Dd};
use super::tables::{EXP2_J64, INV_FACT, INV_LN2_64, LN2_64_1, LN2_64_2, LN2_64_3};

/// `1.5 * 2^52`: adding and subtracting it rounds a binary64 of magnitude below `2^51` to the
/// nearest integer (ties to even).
pub(super) const SHIFT: f64 = 6_755_399_441_055_744.0;

/// The fast paths' relative error allowance (see the module doc for the bound it covers).
const EXP_EPS: f64 = exp2i(-50);
/// `tanh`'s fast-path allowance: [`expm1_fast`]'s error plus one addition and one division.
const TANH_EPS: f64 = exp2i(-48);

/// `x > EXP_OVERFLOW` gives `+inf`: `e^89 > 2^128`.
const EXP_OVERFLOW: f32 = 89.0;
/// `x < EXP_UNDERFLOW` gives `+0`: `e^-104 < 2^-150`, half the least subnormal.
const EXP_UNDERFLOW: f32 = -104.0;
/// `|x| > TANH_ONE` gives `±1`: `1 - tanh(9.1) = 2/(e^18.2 + 1) < 2^-25`, under half the spacing
/// of binary32 below 1.
const TANH_ONE: f32 = 9.1;

/// `k = round(64x/ln 2)` and `r = x - k*ln(2)/64` to binary64 accuracy (`k*C3` omitted).
#[inline]
fn reduce(x: f64) -> (i32, f64) {
    let t = x * INV_LN2_64 + SHIFT;
    let kf = t - SHIFT;
    let r = (x - kf * LN2_64_1) - kf * LN2_64_2;
    // `t`'s significand holds `2^51 + k` exactly; its low 32 bits are `k` in two's complement.
    (t.to_bits() as i32, r)
}

/// `e^r - 1` for `|r| <= ln(2)/128`, degree 6.
#[inline]
fn expm1_poly(r: f64) -> f64 {
    r * horner(r, &INV_FACT[1..7])
}

/// `e^x` for binary32 `x`, correctly rounded (ties to even); `+inf` above the overflow
/// threshold, `+0` below the underflow one, NaN for NaN. See the module doc.
#[inline]
pub fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > EXP_OVERFLOW {
        return f32::INFINITY;
    }
    if x < EXP_UNDERFLOW {
        return 0.0;
    }
    let xd = x as f64;
    let (k, r) = reduce(xd);
    let (th, tl) = EXP2_J64[(k & 63) as usize];
    let y = (th + (th * expm1_poly(r) + tl)) * pow2(k >> 6);
    match round_if_clear(y, y * EXP_EPS) {
        Some(v) => v,
        None => exp_accurate(xd),
    }
}

/// [`exp`]'s accurate path, kept out of line: it runs for a handful of inputs.
#[cold]
#[inline(never)]
fn exp_accurate(x: f64) -> f32 {
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

/// `e^t - 1` for `0 <= t <= 2*TANH_ONE`, relative error below `2^-52`. For `k = 0` it is the
/// polynomial alone; otherwise `2^e*T - 1` is exact or rounds once (Sterbenz covers
/// `2^e*T` in `[1/2, 2]`, where cancellation could otherwise hurt) and the result is at least
/// `e^(ln(2)/128) - 1`, so the absolute errors stay relative.
#[inline]
fn expm1_fast(t: f64) -> f64 {
    let (k, r) = reduce(t);
    let q = expm1_poly(r);
    if k == 0 {
        return q;
    }
    let s = pow2(k >> 6);
    let (th, tl) = EXP2_J64[(k & 63) as usize];
    (s * th - 1.0) + s * (th * q + tl)
}

/// `e^t - 1` in double-double for `0 <= t <= 2*TANH_ONE`: relative error below `2^-97` (for
/// `k != 0` the absolute error `2^-104` of `T*(1+q)` meets a result of at least `2^-7.5`).
fn expm1_dd(t: f64) -> Dd {
    let (k, q) = exp_parts_dd(Dd::new(t, 0.0));
    if k == 0 {
        return q;
    }
    exp_from_parts(k, q).add_f(-1.0)
}

/// `tanh(x)` for binary32 `x`, correctly rounded (ties to even): `tanh(|x|) = m/(m + 2)` with
/// `m = e^(2|x|) - 1`, odd in `x`; `±1` beyond `TANH_ONE`; `±0` at `±0`.
///
/// **Fast path:** `expm1_fast`'s `2^-52`, scaled by `2/(m+2) <= 1` through the quotient, plus
/// one rounding each for `m + 2` and the division: below `2^-51` relative, against an allowance
/// of `TANH_EPS` `= 2^-48`. **Accurate path:** `expm1_dd` and a double-double division,
/// relative error below `2^-95`. `tanh(x)` is irrational for binary32 `x != 0`, so no input is
/// a rounding boundary.
#[inline]
pub fn tanh(x: f32) -> f32 {
    if x.is_nan() || x == 0.0 {
        return x;
    }
    let neg = x < 0.0;
    let a = if neg { -x } else { x };
    let v = if a > TANH_ONE {
        1.0
    } else {
        let t = 2.0 * a as f64;
        let m = expm1_fast(t);
        let y = m / (m + 2.0);
        match round_if_clear(y, y * TANH_EPS) {
            Some(v) => v,
            None => tanh_accurate(t),
        }
    };
    if neg {
        -v
    } else {
        v
    }
}

/// [`tanh`]'s accurate path for `t = 2|x|`, kept out of line.
#[cold]
#[inline(never)]
fn tanh_accurate(t: f64) -> f32 {
    let m = expm1_dd(t);
    dd_to_f32(m.div(m.add_f(2.0)))
}
