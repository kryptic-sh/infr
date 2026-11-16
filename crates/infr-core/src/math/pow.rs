//! `powf`.
//!
//! **Method:** `|x|^y = e^(y * ln|x|)`, with the C99/IEEE 754 `pow` special cases first and the
//! sign restored for a negative `x` raised to an odd integer.
//!
//! **Fast path error bound.** [`ln_pow_fast`] gives `ln|x|` as a double-double to about
//! `2^-66` relative (only its degree-3-and-up tail is rounded); the product with `y` is exact
//! up to that and `2^-104`, so `t = y*ln|x|` carries an absolute error below `2^-59` over the
//! range `|t| <= 104` that does not overflow or underflow. [`exp_dd_fast`] adds `t`'s low part
//! into its reduced argument (two more roundings of a quantity below `2^-7.5`) and otherwise has
//! `exp`'s `1.1 * 2^-53`. The total stays below `1.2 * 2^-53` relative; the allowance
//! [`POW_EPS`] is `exp`'s.
//!
//! **Accurate path.** `ln` in double-double ([`ln_dd`]), the product, and `e^t` in
//! double-double: relative error below `2^-90`. Unlike the unary functions, `x^y` CAN be a
//! binary32 rounding boundary exactly (`11^7` is a 25-bit odd integer, `66049^1.5 = 257^3`,
//! `(2^-16)^9.375 = 2^-150`), and no finite precision separates those from their boundary. So a
//! result within [`MIDPOINT_TOL`] of a boundary is tested for exactness with integer
//! arithmetic ([`is_exact`]); an exact boundary rounds to even.
//!
//! **What is proven and what is tested.** The special cases, the exactness test, and the fast
//! path's bound are argued here. That every inexact `x^y` lies further than `2^-90` relative
//! from a rounding boundary is NOT proven: there are `2^64` input pairs and no published bound
//! for binary32 `pow` that this code relies on. It is tested: the known-answer table (hard and
//! exact cases from mpmath, and RoPE's `theta^(-2p/d)` for the bases models use) and a
//! dense-grid hash. docs/math.md records the gap.

use super::dd::{dd_to_f32, exp2i, next_down, next_up, round_if_clear};
use super::exp::{exp_dd_fast, exp_from_parts, exp_parts_dd};
use super::log::{abs, ln_dd, ln_pow_fast};

/// The fast path's relative error allowance.
const POW_EPS: f64 = exp2i(-50);
/// An accurate result this close (relative) to a rounding boundary is checked for being that
/// boundary exactly.
const MIDPOINT_TOL: f64 = exp2i(-85);
/// `y*ln|x| > POW_OVERFLOW` overflows (`e^89 > 2^128`) and `< POW_UNDERFLOW` underflows to 0.
const POW_OVERFLOW: f64 = 89.0;
const POW_UNDERFLOW: f64 = -104.0;

/// `x^y` for binary32 `x` and `y`, correctly rounded (ties to even) with C99 `powf`'s special
/// cases: `x^±0 = 1` and `1^y = 1` for every `x`/`y` (NaN included); otherwise NaN in, NaN
/// out; `±0`, `±inf` and `-1^±inf` per C99 F.9.4.4; a negative finite `x` with a non-integer
/// finite `y` is NaN. See the module doc for what is proven and what is tested.
pub fn powf(x: f32, y: f32) -> f32 {
    if y == 0.0 || x == 1.0 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }
    let y_odd = is_odd_integer(y);
    let x_neg = x.is_sign_negative();
    if x == 0.0 {
        return match (y < 0.0, x_neg && y_odd) {
            (true, true) => f32::NEG_INFINITY,
            (true, false) => f32::INFINITY,
            (false, true) => -0.0,
            (false, false) => 0.0,
        };
    }
    if y.is_infinite() {
        if x == -1.0 {
            return 1.0;
        }
        let ax_lt_1 = -1.0 < x && x < 1.0;
        return if ax_lt_1 == (y < 0.0) {
            f32::INFINITY
        } else {
            0.0
        };
    }
    if x.is_infinite() {
        let big = if y < 0.0 { 0.0 } else { f32::INFINITY };
        return if x_neg && y_odd { -big } else { big };
    }
    if x_neg && !is_integer(y) {
        return f32::NAN;
    }
    let ax = if x_neg { -x } else { x };
    let v = pow_pos(ax, y);
    if x_neg && y_odd {
        -v
    } else {
        v
    }
}

/// `ax^y` for finite positive `ax != 1` and finite nonzero `y`.
fn pow_pos(ax: f32, y: f32) -> f32 {
    let u = ax as f64;
    let yd = y as f64;
    let t = ln_pow_fast(u).mul_f(yd);
    if t.hi > POW_OVERFLOW {
        return f32::INFINITY;
    }
    if t.hi < POW_UNDERFLOW {
        return 0.0;
    }
    let v = exp_dd_fast(t);
    if let Some(r) = round_if_clear(v, v * POW_EPS) {
        return r;
    }
    let t = ln_dd(u).mul_f(yd);
    let (k, q) = exp_parts_dd(t);
    let v = exp_from_parts(k, q);
    let r = dd_to_f32(v);
    if !r.is_finite() {
        return r;
    }
    // The boundaries on either side of r; an exact boundary result rounds to even.
    let rf = r as f64;
    for nb in [next_down(r), next_up(r)] {
        if !nb.is_finite() || nb < 0.0 {
            continue;
        }
        let m = (rf + nb as f64) * 0.5;
        if abs(v.add_f(-m).hi) <= m * MIDPOINT_TOL && is_exact(ax, y, m) {
            return m as f32;
        }
    }
    r
}

/// Is `ax^y` exactly `m`, for positive finite binary32 `ax`, finite nonzero binary32 `y`, and a
/// positive binary64 `m`? Integer arithmetic only.
///
/// Write `ax = mx * 2^ex`, `m = mm * 2^em` and `y = n / 2^k` with `mx`, `mm` and (for `k > 0`)
/// `n` odd. Then `ax^y = m` iff `mm^(2^k) = mx^n` and `em * 2^k = ex * n`. For `n > 0`, since
/// `n` is odd when `k > 0`, `mm^(2^k) = mx^n` iff `mx = g^(2^k)` and `mm = g^n` for an integer
/// `g` (unique factorization). For `n < 0` the odd parts must both be 1.
fn is_exact(ax: f32, y: f32, m: f64) -> bool {
    let (mx, ex) = odd_parts_f32(ax);
    let (mm, em) = odd_parts_f64(m);
    let (ny, ey) = odd_parts_f32(y);
    let neg_y = y < 0.0;
    let (n, k) = if ey >= 0 {
        if ey > 16 {
            // |n| >= 2^17: x^n is exactly a 25-bit midpoint only if mx = 1, and then
            // |ex * n| >= 2^17 lies far outside binary32's exponent range.
            return false;
        }
        ((ny as i128) << ey, 0u32)
    } else {
        (ny as i128, (-ey) as u32)
    };
    let n = if neg_y { -n } else { n };
    if k > 64 {
        // em * 2^k = ex * n with |ex * n| < 2^32 forces ex * n = 0, so mx = 1 and ax = 1,
        // which the caller excludes.
        return false;
    }
    if (em as i128) << k != ex as i128 * n {
        return false;
    }
    if n < 0 {
        return mx == 1 && mm == 1;
    }
    let Some(g) = root_pow2(mx, k) else {
        return false;
    };
    if g == 1 {
        return mm == 1;
    }
    // g >= 2, so the loop leaves through `v <= mm` within 53 steps.
    let mut acc: u64 = 1;
    for _ in 0..n {
        match acc.checked_mul(g) {
            Some(v) if v <= mm => acc = v,
            _ => return false,
        }
    }
    acc == mm
}

/// `g` with `g^(2^k) = v`, if one exists.
fn root_pow2(mut v: u64, k: u32) -> Option<u64> {
    for _ in 0..k {
        if v == 1 {
            return Some(1);
        }
        let s = isqrt(v);
        if s * s != v {
            return None;
        }
        v = s;
    }
    Some(v)
}

/// `floor(sqrt(v))` for `v < 2^53` (binary64 `sqrt` is correctly rounded; the loops fix the
/// last unit).
fn isqrt(v: u64) -> u64 {
    let mut s = (v as f64).sqrt() as u64;
    while s * s > v {
        s -= 1;
    }
    while (s + 1) * (s + 1) <= v {
        s += 1;
    }
    s
}

/// `(odd, e)` with `|v| = odd * 2^e`, for finite nonzero binary32 `v`.
fn odd_parts_f32(v: f32) -> (u64, i32) {
    let b = v.to_bits() & 0x7fff_ffff;
    let be = (b >> 23) as i32;
    let (m, e) = if be == 0 {
        (b as u64, -149)
    } else {
        (((b & 0x7f_ffff) | 0x80_0000) as u64, be - 150)
    };
    let tz = m.trailing_zeros();
    (m >> tz, e + tz as i32)
}

/// `(odd, e)` with `v = odd * 2^e`, for positive finite binary64 `v`.
fn odd_parts_f64(v: f64) -> (u64, i32) {
    let b = v.to_bits();
    let be = (b >> 52) as i32;
    let (m, e) = if be == 0 {
        (b, -1074)
    } else {
        ((b & 0x000f_ffff_ffff_ffff) | (1 << 52), be - 1075)
    };
    let tz = m.trailing_zeros();
    (m >> tz, e + tz as i32)
}

/// Is finite `y` an integer? From its bits: no fraction bits below the binary point.
fn is_integer(y: f32) -> bool {
    let b = y.to_bits() & 0x7fff_ffff;
    let e = (b >> 23) as i32 - 127;
    if e < 0 {
        return b == 0;
    }
    e >= 23 || b & ((1 << (23 - e)) - 1) == 0
}

/// Is finite `y` an odd integer?
fn is_odd_integer(y: f32) -> bool {
    let b = y.to_bits() & 0x7fff_ffff;
    let e = (b >> 23) as i32 - 127;
    if !(0..=23).contains(&e) || !is_integer(y) {
        return false;
    }
    let m = (b & 0x7f_ffff) | 0x80_0000;
    (m >> (23 - e)) & 1 == 1
}
