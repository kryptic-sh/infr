//! `ln`, `ln_1p`, and the natural-log core they share with `powf`.
//!
//! **Method** (Tang's table-driven logarithm, ACM TOMS 16, 1990, with CORE-MATH's
//! fast-then-accurate structure): for a positive binary64 `u` with at most 33 significant bits,
//! `u = 2^e * m` with `m` in `[1, 2)`. The top 7 bits of `m`'s fraction pick interval `i`; for
//! `i >= 64` (`m >= 1.5`) `m` is halved and `e` raised, so `m` lies in `[0.75, 1.5)` and values
//! near 1 from either side keep `e = 0`. With `R_i` a 20-bit approximation of `1/m`, `m*R_i` is
//! exact (33 + 20 bits) and so is `z = m*R_i - 1` (Sterbenz), `|z| < 1/126`. Then
//! `ln u = e*ln 2 - ln R_i + ln(1 + z)`. The two intervals touching 1 have `R_i = 1`, so for `u`
//! near 1 the result is the polynomial alone and keeps full relative accuracy.
//!
//! **Fast path error bound.** `e*LN2_1 + L_i` is formed exactly (TwoSum); the remaining parts
//! (`e*LN2_2`, `L_i`'s low half, the polynomial) are added into its low half, and one final
//! addition rounds. `ln(1 + z)`'s Taylor polynomial to degree 9 leaves `|z|^10/10`, under
//! `2^-62` relative to `|z|`, and costs about `2^-52` relative to evaluate. Outside the two
//! intervals touching 1 the result is at least `ln(1 + 1/256)` in magnitude and the parts it is
//! summed from are at most twice the result, so the relative error stays below `2^-51`; the
//! allowance [`LN_EPS`] has a margin of 4 over it.
//!
//! **Accurate path.** The same decomposition in double-double, `ln(1 + z)` to degree 17
//! (remainder below `2^-118` relative): relative error below `2^-100`. `ln x` is irrational for
//! binary32 `x != 1` (Lindemann), so no input is an exact rounding boundary.

use super::dd::{dd_to_f32, exp2i, horner, horner_dd, round_if_clear, two_prod, two_sum, Dd};
use super::tables::{LN2_1, LN2_2, LN2_3, LOG1P_C, LOG_L, LOG_R};
use super::NAN;

/// The fast paths' relative error allowance.
pub(super) const LN_EPS: f64 = exp2i(-49);

/// Below this magnitude `ln_1p` evaluates `ln(1 + x)` as the polynomial in `x` directly; at or
/// above it `1 + x` is exact in binary64 with at most 33 significant bits (up to
/// [`LN1P_BIG`]).
const LN1P_SMALL: f32 = exp2i(-8) as f32;
/// From here on `ln_1p(x) = ln(x) + ln(1 + 1/x)`: `1 + x` would need more significant bits than
/// the table reduction keeps exact.
const LN1P_BIG: f32 = exp2i(24) as f32;

/// `(e, i, z)`: `u = 2^e * m`, interval `i`, and the exact reduced argument `z = m*R_i - 1`.
#[inline]
fn decompose(u: f64) -> (i32, usize, f64) {
    let bits = u.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let i = ((bits >> 45) & 127) as usize;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if i >= 64 {
        m *= 0.5;
        e += 1;
    }
    (e, i, m * LOG_R[i] - 1.0)
}

/// `ln(1 + z)` for `|z| < 1/126`, degree 9, binary64.
#[inline]
fn log1p_poly(z: f64) -> f64 {
    z * horner(z, &LOG1P_C[1..10])
}

/// `ln(u) + c` in binary64 for a positive normal binary64 `u` with at most 33 significant bits
/// and a tiny correction `c` (`|c| < 2^-23`).
#[inline]
pub(super) fn ln_fast(u: f64, c: f64) -> f64 {
    let (e, i, z) = decompose(u);
    let ef = e as f64;
    let (lh, ll) = LOG_L[i];
    let s = two_sum(ef * LN2_1, lh);
    s.hi + (s.lo + (ef * LN2_2 + ll) + log1p_poly(z) + c)
}

/// `ln(u)` in double-double, `u` as for [`ln_fast`].
pub(super) fn ln_dd(u: f64) -> Dd {
    let (e, i, z) = decompose(u);
    let ef = e as f64;
    let zd = Dd::new(z, 0.0);
    let p = zd.mul(horner_dd(zd, &LOG1P_C[1..18]));
    two_sum(ef * LN2_1, ef * LN2_2)
        .add_f(ef * LN2_3)
        .add(Dd::from_pair(LOG_L[i]))
        .add(p)
}

/// `ln(u)` as a double-double to about `2^-66` relative, cheaply, for `powf`'s fast path:
/// `z - z^2/2` is formed exactly and only the tail `z^3 * (...)` (to degree 12) is rounded.
pub(super) fn ln_pow_fast(u: f64) -> Dd {
    let (e, i, z) = decompose(u);
    let ef = e as f64;
    let h = two_prod(z, z);
    let lead = two_sum(z, -0.5 * h.hi).add_f(-0.5 * h.lo);
    let tail = z * z * z * horner(z, &LOG1P_C[3..13]);
    two_sum(ef * LN2_1, ef * LN2_2)
        .add_f(ef * LN2_3)
        .add(Dd::from_pair(LOG_L[i]))
        .add(lead)
        .add_f(tail)
}

/// `ln(x)` for binary32 `x`, correctly rounded (ties to even): NaN for NaN and negative `x`,
/// `-inf` at `±0`, `+inf` at `+inf`, `+0` at 1. See the module doc.
#[inline]
pub fn ln(x: f32) -> f32 {
    if x.is_nan() {
        return NAN;
    }
    if x < 0.0 {
        return NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x == f32::INFINITY {
        return x;
    }
    let u = x as f64;
    let y = ln_fast(u, 0.0);
    match round_if_clear(y, abs(y) * LN_EPS) {
        Some(v) => v,
        None => ln_accurate(u),
    }
}

/// `ln(1 + x)` for binary32 `x`, correctly rounded (ties to even): NaN for NaN and `x < -1`,
/// `-inf` at -1, `+inf` at `+inf`, `±0` at `±0`.
///
/// Three regimes. `|x| <` `LN1P_SMALL`: the polynomial in `x` itself (the interval of 1 in
/// [`ln`]'s table, with `z = x`). Up to `LN1P_BIG`: `1 + x` is exact in binary64 with at most
/// 33 significant bits, so [`ln`]'s reduction applies to it unchanged. Beyond: `ln(x) + ln(1 +
/// 1/x)`, the second term (below `2^-24`) as `c - c^2/2` in the fast path and to degree 3 in
/// double-double in the accurate one. Error bounds and the allowance are [`ln`]'s;
/// `ln(1 + x)` is irrational for binary32 `x != 0`.
#[inline]
pub fn ln_1p(x: f32) -> f32 {
    if x.is_nan() {
        return NAN;
    }
    if x == 0.0 {
        return x;
    }
    if x < -1.0 {
        return NAN;
    }
    if x == -1.0 {
        return f32::NEG_INFINITY;
    }
    if x == f32::INFINITY {
        return x;
    }
    let y = ln_1p_fast(x);
    match round_if_clear(y, abs(y) * LN_EPS) {
        Some(v) => v,
        None => ln_1p_accurate(x),
    }
}

/// [`ln_1p`]'s binary64 value for finite `x > -1`, `x != 0`.
#[inline]
pub(super) fn ln_1p_fast(x: f32) -> f64 {
    let xd = x as f64;
    if abs(xd) < LN1P_SMALL as f64 {
        log1p_poly(xd)
    } else if x < LN1P_BIG {
        ln_fast(1.0 + xd, 0.0)
    } else {
        ln_1p_big(xd)
    }
}

/// [`ln_1p_fast`] from `LN1P_BIG` up, kept out of line. Inlined, a baseline x86-64 build of
/// `ln_1p` took about 2.5 times as long per call as a native one on inputs that never reach this
/// branch, and out of line the two are level (docs/math.md); the compiler presumably computed
/// the branch's division speculatively.
#[cold]
#[inline(never)]
fn ln_1p_big(xd: f64) -> f64 {
    let c = 1.0 / xd;
    ln_fast(xd, c - 0.5 * c * c)
}

/// [`ln`]'s accurate path, kept out of line.
#[cold]
#[inline(never)]
fn ln_accurate(u: f64) -> f32 {
    dd_to_f32(ln_dd(u))
}

/// [`ln_1p`]'s accurate path for a finite `x > -1`, `x != 0`, kept out of line.
#[cold]
#[inline(never)]
fn ln_1p_accurate(x: f32) -> f32 {
    let xd = x as f64;
    let d = if abs(xd) < LN1P_SMALL as f64 {
        let zd = Dd::new(xd, 0.0);
        zd.mul(horner_dd(zd, &LOG1P_C[1..18]))
    } else if x < LN1P_BIG {
        ln_dd(1.0 + xd)
    } else {
        let c = Dd::new(1.0, 0.0).div(Dd::new(xd, 0.0));
        ln_dd(xd).add(c.mul(horner_dd(c, &LOG1P_C[1..4])))
    };
    dd_to_f32(d)
}

/// `|v|` by clearing the sign bit.
#[inline]
pub(super) fn abs(v: f64) -> f64 {
    f64::from_bits(v.to_bits() & !(1u64 << 63))
}
