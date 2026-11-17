//! An independent high-precision reference for `infr_core::math`, for the exhaustive tests.
//!
//! Independent of the code under test by construction: its own double-double arithmetic, its
//! constants derived here (`ln 2` from a series at first use, `pi` from its published hex
//! digits rather than the module's `2/pi` table), and different algorithms — `e^r` by a long
//! Taylor series on `|r| <= ln(2)/2` with no table, `ln` by the `atanh` series on a
//! significand in `[0.7, 1.42)`, and the trigonometric reduction by exact big-integer
//! DIVISION by `pi/2` instead of multiplication by `2/pi`. Every result carries about 100
//! correct bits, which [`verdict`] uses to decide whether a binary32 result is the correct
//! rounding — and to say "undecided" rather than guess when the reference is too close to a
//! rounding boundary to tell.
//!
//! No standard-library transcendental function is used: their bits are what differs between
//! platforms.

#![allow(dead_code)]

use std::sync::OnceLock;

/// `hi + lo`.
#[derive(Clone, Copy, Debug)]
pub struct D {
    pub hi: f64,
    pub lo: f64,
}

const SPLIT: f64 = 134_217_729.0;

fn ts(a: f64, b: f64) -> D {
    let s = a + b;
    let v = s - a;
    D {
        hi: s,
        lo: (a - (s - v)) + (b - v),
    }
}

fn fts(a: f64, b: f64) -> D {
    let s = a + b;
    D {
        hi: s,
        lo: b - (s - a),
    }
}

fn tp(a: f64, b: f64) -> D {
    let p = a * b;
    let c = SPLIT * a;
    let ah = c - (c - a);
    let al = a - ah;
    let c = SPLIT * b;
    let bh = c - (c - b);
    let bl = b - bh;
    D {
        hi: p,
        lo: ((ah * bh - p) + ah * bl + al * bh) + al * bl,
    }
}

impl D {
    pub fn f(x: f64) -> D {
        D { hi: x, lo: 0.0 }
    }
    pub fn add(self, b: D) -> D {
        let s = ts(self.hi, b.hi);
        let t = ts(self.lo, b.lo);
        let s = fts(s.hi, s.lo + t.hi);
        fts(s.hi, s.lo + t.lo)
    }
    pub fn sub(self, b: D) -> D {
        self.add(D {
            hi: -b.hi,
            lo: -b.lo,
        })
    }
    pub fn mul(self, b: D) -> D {
        let p = tp(self.hi, b.hi);
        fts(p.hi, p.lo + (self.hi * b.lo + self.lo * b.hi))
    }
    pub fn div(self, b: D) -> D {
        let q1 = self.hi / b.hi;
        let r = self.sub(b.mul(D::f(q1)));
        let q2 = r.hi / b.hi;
        let r = r.sub(b.mul(D::f(q2)));
        let q3 = r.hi / b.hi;
        fts(q1, q2).add(D::f(q3))
    }
    /// Times `2^e` (exact while the parts stay normal).
    pub fn ldexp(self, e: i32) -> D {
        let s = p2(e);
        D {
            hi: self.hi * s,
            lo: self.lo * s,
        }
    }
    pub fn neg(self) -> D {
        D {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
    /// Sign of `self - b` for a binary64 `b`: -1, 0 or 1, exactly.
    pub fn cmp_f(self, b: f64) -> i32 {
        let d = ts(self.hi, -b);
        let s = d.hi + (d.lo + self.lo);
        if s > 0.0 {
            1
        } else if s < 0.0 {
            -1
        } else {
            0
        }
    }
}

fn p2(e: i32) -> f64 {
    if e < -1000 {
        return p2(e + 500) * p2(-500);
    }
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// `n` as a double-double (exact for |n| < 2^106).
fn int(n: i64) -> D {
    D::f(n as f64)
}

/// `ln 2 = 2 * atanh(1/3) = 2 * sum (1/3)^(2k+1)/(2k+1)`, computed once in double-double.
pub fn ln2() -> D {
    static V: OnceLock<(f64, f64)> = OnceLock::new();
    let &(hi, lo) = V.get_or_init(|| {
        let third = D::f(1.0).div(D::f(3.0));
        let ninth = third.mul(third);
        let mut pow = third;
        let mut sum = D::f(0.0);
        for k in 0..80 {
            sum = sum.add(pow.div(int(2 * k + 1)));
            pow = pow.mul(ninth);
        }
        let v = sum.add(sum);
        (v.hi, v.lo)
    });
    D { hi, lo }
}

/// `e^x` for `|x| < 200`: `x = k*ln2 + r`, `|r| <= ln(2)/2 + tiny`, Taylor series to 2^-110.
pub fn exp(x: D) -> D {
    // Any k near x/ln2 works: the series converges for the resulting |r| either way.
    let k = (x.hi / ln2().hi).round_ties_even();
    let r = x.sub(ln2().mul(D::f(k)));
    let mut term = D::f(1.0);
    let mut sum = D::f(1.0);
    for n in 1..40 {
        term = term.mul(r).div(int(n));
        sum = sum.add(term);
        if term.hi.abs() < 1e-40 {
            break;
        }
    }
    sum.ldexp(k as i32)
}

/// `e^x - 1`: the series itself near 0, where `e^x - 1` would cancel.
pub fn expm1(x: D) -> D {
    if x.hi.abs() < 0.5 {
        let mut term = x;
        let mut sum = x;
        for n in 2..60 {
            term = term.mul(x).div(int(n));
            sum = sum.add(term);
            if term.hi.abs() < sum.hi.abs() * 1e-36 {
                break;
            }
        }
        return sum;
    }
    exp(x).sub(D::f(1.0))
}

/// `ln a` for positive normal `a`: `a = 2^e * m`, `m` in `[0.7, 1.42)`, `ln m = 2 atanh(u)` with
/// `u = (m - 1)/(m + 1)`, `|u| < 0.18`.
pub fn ln(a: D) -> D {
    let bits = a.hi.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mut m = a.ldexp(-e);
    if m.hi > 1.42 {
        m = m.ldexp(-1);
        e += 1;
    }
    let u = m.sub(D::f(1.0)).div(m.add(D::f(1.0)));
    let u2 = u.mul(u);
    let mut pow = u;
    let mut sum = D::f(0.0);
    for k in 0..60 {
        let t = pow.div(int(2 * k + 1));
        sum = sum.add(t);
        if t.hi.abs() < 1e-36 * sum.hi.abs().max(1e-300) {
            break;
        }
        pow = pow.mul(u2);
    }
    sum.add(sum).add(ln2().mul(int(e as i64)))
}

/// `tanh x = m/(m + 2)`, `m = e^(2x) - 1`.
pub fn tanh(x: f64) -> D {
    let m = expm1(D::f(2.0 * x));
    m.div(m.add(D::f(2.0)))
}

// ── trigonometric reduction ──────────────────────────────────────────────────────────────────

/// The first 576 fraction bits of `pi` in hexadecimal (`pi = 3.243F6A88...`), the published
/// digits (they also seed Blowfish's P-array).
const PI_FRAC_HEX: &str = "243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89\
452821E638D01377BE5466CF34E90C6CC0AC29B7C97C50DD3F84D5B5B54709179216D5D98979FB1B";

/// Fraction bits kept in the fixed-point reduction.
const P: u32 = 400;

/// Little-endian base-2^32 big integer.
type Big = Vec<u32>;

fn big_shl1(a: &mut Big) {
    let mut carry = 0u32;
    for w in a.iter_mut() {
        let n = *w >> 31;
        *w = (*w << 1) | carry;
        carry = n;
    }
    if carry != 0 {
        a.push(carry);
    }
}

fn big_cmp(a: &Big, b: &Big) -> std::cmp::Ordering {
    let n = a.len().max(b.len());
    for i in (0..n).rev() {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        if x != y {
            return x.cmp(&y);
        }
    }
    std::cmp::Ordering::Equal
}

fn big_sub(a: &mut Big, b: &Big) {
    let mut borrow = 0i64;
    for (i, w) in a.iter_mut().enumerate() {
        let v = *w as i64 - b.get(i).copied().unwrap_or(0) as i64 - borrow;
        borrow = (v < 0) as i64;
        *w = v.rem_euclid(1 << 32) as u32;
    }
    assert_eq!(borrow, 0);
}

fn big_bits(a: &Big) -> u32 {
    for i in (0..a.len()).rev() {
        if a[i] != 0 {
            return i as u32 * 32 + 32 - a[i].leading_zeros();
        }
    }
    0
}

fn big_bit(a: &Big, n: u32) -> bool {
    a.get((n / 32) as usize)
        .is_some_and(|w| (w >> (n % 32)) & 1 == 1)
}

/// `floor(pi/2 * 2^P)` as a big integer.
fn half_pi_fixed() -> &'static Big {
    static V: OnceLock<Big> = OnceLock::new();
    V.get_or_init(|| {
        // pi * 2^(P-1) = 3 * 2^(P-1) + frac * 2^(P-1); frac's first P-1 bits suffice.
        let mut bits: Vec<bool> = Vec::new();
        for c in PI_FRAC_HEX.chars() {
            let d = c.to_digit(16).unwrap();
            for k in (0..4).rev() {
                bits.push((d >> k) & 1 == 1);
            }
        }
        let nbits = P - 1;
        let mut v: Big = vec![0; (P as usize + 64) / 32];
        // integer part 3 at bit position P-1 and P.
        let set = |v: &mut Big, n: u32| v[(n / 32) as usize] |= 1 << (n % 32);
        set(&mut v, nbits);
        set(&mut v, nbits + 1);
        for (i, &b) in bits.iter().take(nbits as usize).enumerate() {
            if b {
                set(&mut v, nbits - 1 - i as u32);
            }
        }
        v
    })
}

/// `(n mod 4, r)` with `|x| = n*pi/2 + r`, `|r| <= pi/4` (to the reference's precision), for a
/// finite binary32 `|x| >= 0.7`, by long division of the exact fixed-point `|x|` by `pi/2`.
fn reduce_half_pi(x: f32) -> (u32, D) {
    let b = x.to_bits() & 0x7fff_ffff;
    let be = (b >> 23) as i32;
    let (mx, ex) = ((b & 0x7f_ffff) | 0x80_0000, be - 150);
    // X = mx * 2^(ex + P), an integer since ex >= -24 here.
    let sh = (ex + P as i32) as u32;
    let mut xbig: Big = vec![0; (sh as usize + 24) / 32 + 2];
    for k in 0..24 {
        if (mx >> k) & 1 == 1 {
            let n = sh + k;
            xbig[(n / 32) as usize] |= 1 << (n % 32);
        }
    }
    let h = half_pi_fixed();
    // Long division, most significant quotient bit first.
    let mut rem: Big = vec![0];
    let mut q: u128 = 0;
    for n in (0..big_bits(&xbig)).rev() {
        big_shl1(&mut rem);
        if big_bit(&xbig, n) {
            rem[0] |= 1;
        }
        q <<= 1;
        if big_cmp(&rem, h) != std::cmp::Ordering::Less {
            big_sub(&mut rem, h);
            q |= 1;
        }
    }
    // rem in [0, H); round the quotient to nearest: if 2*rem > H, use rem - H.
    let mut twice = rem.clone();
    big_shl1(&mut twice);
    let (n, mag, neg) = if big_cmp(&twice, h) == std::cmp::Ordering::Greater {
        let mut m = h.clone();
        big_sub(&mut m, &rem);
        (q + 1, m, true)
    } else {
        (q, rem, false)
    };
    let r = big_to_d(&mag).ldexp(-(P as i32));
    ((n % 4) as u32, if neg { r.neg() } else { r })
}

/// A big integer as a double-double, from its leading 106 bits.
fn big_to_d(a: &Big) -> D {
    let len = big_bits(a) as i32;
    if len == 0 {
        return D::f(0.0);
    }
    let take = |from: i32| -> f64 {
        let mut v = 0u64;
        for k in 0..53 {
            let n = from + k;
            if n >= 0 && big_bit(a, n as u32) {
                v |= 1 << k;
            }
        }
        v as f64 * p2(from)
    };
    fts(take(len - 53), take(len - 106))
}

/// `(sin r, cos r)` for `|r| <= pi/4` by Taylor series.
fn sin_cos_small(r: D) -> (D, D) {
    let r2 = r.mul(r);
    let mut ts_ = r;
    let mut s = r;
    let mut tc = D::f(1.0);
    let mut c = D::f(1.0);
    for k in 1..40i64 {
        ts_ = ts_.mul(r2).div(int((2 * k) * (2 * k + 1))).neg();
        tc = tc.mul(r2).div(int((2 * k - 1) * (2 * k))).neg();
        s = s.add(ts_);
        c = c.add(tc);
        if tc.hi.abs() < 1e-40 && ts_.hi.abs() < 1e-40 * s.hi.abs().max(1e-300) {
            break;
        }
    }
    (s, c)
}

/// `(sin x, cos x)` for finite binary32 `x`.
pub fn sin_cos(x: f32) -> (D, D) {
    let neg = x < 0.0;
    let a = x.abs();
    let (s, c) = if a < 0.7 {
        sin_cos_small(D::f(a as f64))
    } else {
        let (q, r) = reduce_half_pi(a);
        let (s, c) = sin_cos_small(r);
        match q {
            0 => (s, c),
            1 => (c, s.neg()),
            2 => (s.neg(), c.neg()),
            _ => (c.neg(), s),
        }
    };
    (if neg { s.neg() } else { s }, c)
}

// ── deciding a result ────────────────────────────────────────────────────────────────────────

/// The reference's relative accuracy that [`verdict`] relies on.
pub const REL: f64 = 1e-28; // about 2^-93

/// What the reference says about a binary32 result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// The correct rounding, with the reference this far (relative) from the nearer boundary.
    Ok(f64),
    /// Not the correct rounding.
    Wrong,
    /// The reference lies within its own error of a boundary of `got`.
    Undecided,
}

fn next_up(x: f32) -> f32 {
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let b = x.to_bits();
    f32::from_bits(if x > 0.0 { b + 1 } else { b - 1 })
}

fn next_down(x: f32) -> f32 {
    -next_up(-x)
}

/// `2^128 - 2^103`, halfway between `f32::MAX` and the next power of two: values at or beyond
/// it round to infinity.
fn overflow_edge() -> f64 {
    33_554_431.0 * p2(103)
}

/// Is `got` the binary32 nearest to the reference value `v`?
pub fn verdict(got: f32, v: D) -> Verdict {
    if got.is_nan() {
        return Verdict::Wrong;
    }
    let mag = v.hi.abs();
    let tol = mag * REL;
    if got.is_infinite() {
        let edge = overflow_edge() * got.signum() as f64;
        let d = v.cmp_f(edge) * got.signum() as i32;
        let dist = (v.hi - edge).abs();
        return if d > 0 && dist > tol {
            Verdict::Ok(dist / mag)
        } else if d < 0 && dist > tol {
            Verdict::Wrong
        } else {
            Verdict::Undecided
        };
    }
    let lo = (got as f64 + next_down(got) as f64) * 0.5;
    let hi = if got == f32::MAX {
        overflow_edge()
    } else {
        (got as f64 + next_up(got) as f64) * 0.5
    };
    let above_lo = v.sub(D::f(lo));
    let below_hi = D::f(hi).sub(v);
    let (a, b) = (above_lo.hi, below_hi.hi);
    if a > tol && b > tol {
        Verdict::Ok(a.min(b) / mag.max(f64::MIN_POSITIVE))
    } else if a < -tol || b < -tol {
        Verdict::Wrong
    } else {
        Verdict::Undecided
    }
}
