//! Double-double arithmetic, and the two roundings to binary32 the functions end in.
//!
//! A [`Dd`] is an unevaluated sum `hi + lo` of two binary64 values with `|lo| <= ulp(hi)/2`,
//! carrying about 106 bits. Every operation here uses only binary64 `+ - * /` and comparisons —
//! no fused multiply-add — so it gives the same bits on every IEEE 754 target: the exact product
//! comes from Dekker's splitting ([`two_prod`]) rather than from an FMA. The error-free
//! transformations are the classical ones (Knuth's TwoSum, Dekker's FastTwoSum and Mul12);
//! the compound operations follow the "accurate" double-double algorithms of Joldes, Muller and
//! Popescu, "Tight and rigorous error bounds for basic building blocks of double-word
//! arithmetic" (ACM TOMS 44, 2017): relative error below 2^-103 for an addition and 2^-101 for
//! a product on operands of these magnitudes.

/// `hi + lo`, with `|lo| <= ulp(hi)/2` after normalization.
#[derive(Clone, Copy, Debug)]
pub(super) struct Dd {
    pub hi: f64,
    pub lo: f64,
}

/// Dekker's splitting constant `2^27 + 1`.
const SPLITTER: f64 = 134_217_729.0;

impl Dd {
    pub const fn new(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    pub const fn from_pair(p: (f64, f64)) -> Self {
        Self { hi: p.0, lo: p.1 }
    }

    pub fn neg(self) -> Self {
        Self::new(-self.hi, -self.lo)
    }

    /// `self + b`.
    pub fn add(self, b: Dd) -> Dd {
        let s = two_sum(self.hi, b.hi);
        let t = two_sum(self.lo, b.lo);
        let s = fast_two_sum(s.hi, s.lo + t.hi);
        fast_two_sum(s.hi, s.lo + t.lo)
    }

    /// `self + b` for a binary64 `b`.
    pub fn add_f(self, b: f64) -> Dd {
        let s = two_sum(self.hi, b);
        fast_two_sum(s.hi, s.lo + self.lo)
    }

    /// `self * b`.
    pub fn mul(self, b: Dd) -> Dd {
        let p = two_prod(self.hi, b.hi);
        fast_two_sum(p.hi, p.lo + (self.hi * b.lo + self.lo * b.hi))
    }

    /// `self * b` for a binary64 `b`.
    pub fn mul_f(self, b: f64) -> Dd {
        let p = two_prod(self.hi, b);
        fast_two_sum(p.hi, p.lo + self.lo * b)
    }

    /// `self / b`: three binary64 quotient digits, each correcting the remainder of the last.
    pub fn div(self, b: Dd) -> Dd {
        let q1 = self.hi / b.hi;
        let r = self.add(b.mul_f(q1).neg());
        let q2 = r.hi / b.hi;
        let r = r.add(b.mul_f(q2).neg());
        let q3 = r.hi / b.hi;
        fast_two_sum(q1, q2).add_f(q3)
    }

    /// `self * 2^e`, exact while the result's parts stay normal binary64 numbers.
    pub fn scale(self, e: i32) -> Dd {
        let s = pow2(e);
        Dd::new(self.hi * s, self.lo * s)
    }
}

/// `2^e` for `-1022 <= e <= 1023`, built from its bits.
#[inline]
pub(super) fn pow2(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e), "{e}");
    exp2i(e)
}

/// `2^e` as a constant expression, for `-1022 <= e <= 1023`.
pub(super) const fn exp2i(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// Knuth's TwoSum: `a + b` exactly, as the rounded sum and its error.
#[inline]
pub(super) fn two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    Dd::new(s, e)
}

/// Dekker's FastTwoSum: `a + b` exactly, for `|a| >= |b|` (or `a == 0`).
#[inline]
pub(super) fn fast_two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let e = b - (s - a);
    Dd::new(s, e)
}

/// Veltkamp's split of `a` into two halves of at most 26 significant bits each.
#[inline]
fn split(a: f64) -> (f64, f64) {
    let c = SPLITTER * a;
    let hi = c - (c - a);
    (hi, a - hi)
}

/// Dekker's Mul12: `a * b` exactly, as the rounded product and its error, without an FMA. Exact
/// while neither the product nor the split overflows and the error is not subnormal, which
/// holds for every operand the math module passes (all within `2^±400`).
#[inline]
pub(super) fn two_prod(a: f64, b: f64) -> Dd {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    Dd::new(p, e)
}

/// The binary32 nearest to every real number within `err` of `y`, when they all round to the
/// same binary32; `None` when a rounding boundary may lie in between (the caller then takes its
/// accurate path). Rounding is monotonic, so checking the two ends of the interval decides it.
/// The two ends are themselves rounded binary64 values; callers pass an `err` with enough margin
/// that the rounded ends still enclose the true interval.
#[inline]
pub(super) fn round_if_clear(y: f64, err: f64) -> Option<f32> {
    let a = (y - err) as f32;
    let b = (y + err) as f32;
    (a.to_bits() == b.to_bits()).then_some(a)
}

/// `2^128 - 2^103`: halfway between `f32::MAX` and `2^128`; a binary64 at or beyond it rounds to
/// infinity as a binary32.
const OVERFLOW_EDGE: f64 = 33_554_431.0 * exp2i(103);

/// `|v|` by clearing the sign bit.
fn abs_f64(v: f64) -> f64 {
    f64::from_bits(v.to_bits() & !(1u64 << 63))
}

/// The binary32 nearest to `d.hi + d.lo` (ties to even), for a normalized `d`.
///
/// Converting `d.hi` alone is right except when `d.hi` is exactly a binary32 rounding boundary
/// (a midpoint between two neighbours): a boundary strictly between `d.hi` and `d.hi + d.lo`
/// cannot exist, since it would be a binary64 number nearer to `d.hi + d.lo` than `d.hi` is. At
/// a midpoint the sign of `d.lo` says which neighbour the sum is nearer to.
pub(super) fn dd_to_f32(d: Dd) -> f32 {
    let r = d.hi as f32;
    if r.is_infinite() && d.hi.is_finite() {
        // `d.hi` is at or beyond the overflow boundary, halfway between `f32::MAX` and `2^128`
        // (a midpoint like any other, with infinity on its far side). Exactly on it, a low part
        // pointing back toward zero puts the sum below it.
        let toward_zero = (d.lo < 0.0) == (d.hi > 0.0) && d.lo != 0.0;
        if abs_f64(d.hi) == OVERFLOW_EDGE && toward_zero {
            return if d.hi > 0.0 { f32::MAX } else { -f32::MAX };
        }
        return r;
    }
    let rf = r as f64;
    if d.lo == 0.0 || rf == d.hi || !r.is_finite() {
        return r;
    }
    let other = if d.hi > rf { next_up(r) } else { next_down(r) } as f64;
    if d.hi != (rf + other) * 0.5 {
        return r;
    }
    let nudged = if d.lo > 0.0 {
        f64::from_bits(if d.hi > 0.0 {
            d.hi.to_bits() + 1
        } else {
            d.hi.to_bits() - 1
        })
    } else {
        f64::from_bits(if d.hi > 0.0 {
            d.hi.to_bits() - 1
        } else {
            d.hi.to_bits() + 1
        })
    };
    nudged as f32
}

/// The next binary32 toward `+inf` (finite, non-NaN input).
pub(super) fn next_up(x: f32) -> f32 {
    let b = x.to_bits();
    if x == 0.0 {
        f32::from_bits(1)
    } else if b >> 31 == 0 {
        f32::from_bits(b + 1)
    } else {
        f32::from_bits(b - 1)
    }
}

/// The next binary32 toward `-inf` (finite, non-NaN input).
pub(super) fn next_down(x: f32) -> f32 {
    -next_up(-x)
}

/// Horner's rule in binary64 over `c` (constant term first): `c[0] + x*(c[1] + ...)`, for a
/// non-empty `c` and finite `x`. It starts from the last coefficient itself, which is the value
/// `0*x + c_last` the first step would give for finite `x`, one multiply and add sooner.
#[inline(always)]
pub(super) fn horner(x: f64, c: &[(f64, f64)]) -> f64 {
    let (&(last, _), rest) = c.split_last().expect("horner: no coefficients");
    let mut acc = last;
    for &(hi, _) in rest.iter().rev() {
        acc = acc * x + hi;
    }
    acc
}

/// Horner's rule in double-double over `c` (constant term first).
pub(super) fn horner_dd(x: Dd, c: &[(f64, f64)]) -> Dd {
    let mut acc = Dd::new(0.0, 0.0);
    for &p in c.iter().rev() {
        acc = acc.mul(x).add(Dd::from_pair(p));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The error-free transformations are exact: checked against integer arithmetic on values
    /// whose exact sums and products fit in an i128.
    #[test]
    fn two_sum_and_two_prod_are_exact() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..10_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let a = (s >> 11) as i64 - (1 << 52); // |a| < 2^53: exact in binary64
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let b = (s >> 11) as i64 - (1 << 52);
            let (af, bf) = (a as f64, b as f64);
            let t = two_sum(af, bf);
            assert_eq!(t.hi as i128 + t.lo as i128, a as i128 + b as i128);
            let p = two_prod(af, bf);
            assert_eq!(p.hi as i128 + p.lo as i128, a as i128 * b as i128);
        }
    }

    /// A value exactly on a binary32 midpoint follows the sign of its low part.
    #[test]
    fn dd_to_f32_breaks_midpoints_by_the_low_part() {
        let mid = 1.0 + f64::powi(2.0, -24); // halfway between 1 and next_up(1)
        assert_eq!(dd_to_f32(Dd::new(mid, 1e-30)), next_up(1.0));
        assert_eq!(dd_to_f32(Dd::new(mid, -1e-30)), 1.0);
        assert_eq!(dd_to_f32(Dd::new(-mid, -1e-30)), -next_up(1.0));
        assert_eq!(dd_to_f32(Dd::new(-mid, 1e-30)), -1.0);
        // Below a power of two the spacing halves.
        let mid = 1.0 - f64::powi(2.0, -25);
        assert_eq!(dd_to_f32(Dd::new(mid, 1e-30)), 1.0);
        assert_eq!(dd_to_f32(Dd::new(mid, -1e-30)), next_down(1.0));
        // The overflow boundary: below it is f32::MAX, at or above it infinity.
        let edge = OVERFLOW_EDGE;
        assert_eq!(dd_to_f32(Dd::new(edge, -1e10)), f32::MAX);
        assert_eq!(dd_to_f32(Dd::new(-edge, 1e10)), -f32::MAX);
        assert_eq!(dd_to_f32(Dd::new(edge, 1e10)), f32::INFINITY);
        assert_eq!(dd_to_f32(Dd::new(edge, 0.0)), f32::INFINITY);
        assert_eq!(dd_to_f32(Dd::new(edge * 1.5, -1e10)), f32::INFINITY);
        // Subnormal range: halfway between 0 and the least subnormal.
        let mid = f64::powi(2.0, -150);
        assert_eq!(dd_to_f32(Dd::new(mid, 1e-60)), f32::from_bits(1));
        assert_eq!(dd_to_f32(Dd::new(mid, -1e-60)), 0.0);
    }
}
