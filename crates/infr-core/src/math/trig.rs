//! `sin`, `cos` and `sin_cos`.
//!
//! **Method** (the structure of CORE-MATH's binary32 `sinf`/`cosf`): reduce `|x|` modulo
//! `pi/32`, `|x| = k*pi/32 + r` with `|r| <= pi/64`, and use
//! `sin(a + r) = sin a + sin a*(cos r - 1) + cos a*sin r` (likewise `cos`) with `a = (k mod 64)*pi/32`
//! from a double-double table and Taylor polynomials for `sin r` and `cos r - 1`.
//!
//! **Reduction.** Below [`CW_LIMIT`] (which covers every RoPE angle in practice) the fast path
//! reduces Cody–Waite style: `pi/32` in three parts whose first two have 27 bits, so `k*part` is
//! exact for `k < 2^26`; the result `r` has relative error at most `2^-52` plus an absolute error
//! below `2^-83`, the latter carried as [`CW_ABS`] in the rounding test so that an `x` very
//! close to a multiple of `pi/32` is never trusted to the fast path. At and above it, and in the
//! accurate path always, the reduction is Payne–Hanek: `|x| * 32/pi mod 64` from the bits of
//! `2/pi` ([`TWO_OVER_PI`]) by exact 256-bit integer arithmetic on a 192-bit window of them,
//! correct for the whole binary32 range with an absolute error below `2^-160` (in units of
//! `pi/32`), then converted to double-double.
//!
//! **Fast path error bound.** Degree-9 `sin r` and degree-8 `cos r - 1` leave remainders below
//! `2^-68` and `2^-65` relative to the terms they feed; one final addition rounds; the inner sum
//! is at most `0.05 * 2^-52` absolute against a result of at least `sin(pi/64)` unless the table
//! term is 0, where the result is `±sin r` itself with relative error at most `2^-51`. The
//! allowance is [`TRIG_EPS`] relative plus the reduction's absolute error.
//!
//! **Accurate path.** Payne–Hanek, then double-double polynomials to degree 15 and 14:
//! relative error below `2^-100` unless the result is within `2^-60` of zero, which no nonzero
//! binary32 input reaches (`sin x = 0` only at `x = 0`; `cos x != 0` for every binary32).

use super::dd::{dd_to_f32, exp2i, fast_two_sum, horner, horner_dd, round_if_clear, Dd};
use super::exp::SHIFT;
use super::log::abs;
use super::tables::{
    COS_C, COS_J32, INV_PI_32, PI_32, PI_32_1, PI_32_2, PI_32_3, SIN_C, SIN_J32, TWO_OVER_PI,
};
use super::NAN;

/// The fast reduction's domain: `k < 2^26` for every `|x|` below it.
const CW_LIMIT: f32 = exp2i(22) as f32;
/// The fast reduction's absolute error allowance, above its `2^-83.6` bound.
const CW_ABS: f64 = exp2i(-78);
/// The fast paths' relative error allowance.
pub(super) const TRIG_EPS: f64 = exp2i(-49);

/// `sin(x)` for binary32 `x`, correctly rounded (ties to even); NaN for NaN and `±inf`, `±0`
/// at `±0`. See the module doc.
#[inline]
pub fn sin(x: f32) -> f32 {
    sin_cos(x).0
}

/// `cos(x)` for binary32 `x`, correctly rounded (ties to even); NaN for NaN and `±inf`. See
/// the module doc.
#[inline]
pub fn cos(x: f32) -> f32 {
    sin_cos(x).1
}

/// `(sin(x), cos(x))`, each correctly rounded; one reduction serves both.
#[inline]
pub fn sin_cos(x: f32) -> (f32, f32) {
    if !x.is_finite() {
        return (NAN, NAN);
    }
    if x == 0.0 {
        return (x, 1.0);
    }
    let neg = x < 0.0;
    let a = abs(x as f64);
    let (sv, cv, abs_err) = sin_cos_fast(a);
    let fast = (
        round_if_clear(sv, abs(sv) * TRIG_EPS + abs_err),
        round_if_clear(cv, abs(cv) * TRIG_EPS + abs_err),
    );
    let (s, c) = match fast {
        (Some(s), Some(c)) => (s, c),
        _ => sin_cos_accurate(a as f32),
    };
    (if neg { -s } else { s }, c)
}

/// The binary64 values of `sin a` and `cos a` for finite positive `a` (a binary32), and the
/// absolute error allowance of the reduction they came from.
#[inline]
pub(super) fn sin_cos_fast(a: f64) -> (f64, f64, f64) {
    let (j, r, abs_err) = if (a as f32) < CW_LIMIT {
        let kf = (a * INV_PI_32 + SHIFT) - SHIFT;
        let r = ((a - kf * PI_32_1) - kf * PI_32_2) - kf * PI_32_3;
        // At k = 0 the reduction is `r = a` exactly: no absolute error to allow for, which
        // matters for tiny `a`, where `CW_ABS` would dwarf `sin a` itself.
        let abs_err = if kf == 0.0 { 0.0 } else { CW_ABS };
        ((kf as i64 & 63) as usize, r, abs_err)
    } else {
        let (j, r) = payne_hanek_cold(a as f32);
        (j, r.hi, 0.0)
    };
    let r2 = r * r;
    let s = r + r * r2 * horner(r2, &SIN_C[1..5]);
    let c = r2 * horner(r2, &COS_C[1..5]);
    let (sh, sl) = SIN_J32[j];
    let (ch, cl) = COS_J32[j];
    let sv = sh + (((sl + sh * c) + ch * s) + cl * s);
    let cv = ch + (((cl + ch * c) - sh * s) - sl * s);
    (sv, cv, abs_err)
}

/// `(sin a, cos a)` for finite positive binary32 `a`, through the Payne–Hanek reduction and
/// double-double polynomials. Kept out of line: it runs for a handful of inputs.
#[cold]
#[inline(never)]
fn sin_cos_accurate(a: f32) -> (f32, f32) {
    let (j, r) = payne_hanek(a);
    let r2 = r.mul(r);
    let s = r.add(r.mul(r2).mul(horner_dd(r2, &SIN_C[1..8])));
    let c = r2.mul(horner_dd(r2, &COS_C[1..8]));
    let sj = Dd::from_pair(SIN_J32[j]);
    let cj = Dd::from_pair(COS_J32[j]);
    let sv = sj.add(sj.mul(c)).add(cj.mul(s));
    let cv = cj.add(cj.mul(c)).add(sj.mul(s).neg());
    (dd_to_f32(sv), dd_to_f32(cv))
}

/// `(k mod 64, r)` with `a*32/pi = k + f`, `|f| <= 1/2`, and `r = f*pi/32` in double-double, for
/// finite positive binary32 `a`, by exact integer arithmetic on the bits of `2/pi`.
///
/// `a = mx * 2^ex` with integer `mx < 2^24`, so `a*32/pi = mx * 2^(ex+4) * sum(b_i * 2^-i)`
/// over the bits `b_i` of `2/pi`. Bits with `ex + 4 - i >= 6` contribute multiples of 64 and
/// are dropped; the 192 bits from the first kept one are multiplied by `mx` exactly (216 bits),
/// which leaves 6 integer bits mod 64 and at least 186 fraction bits, the truncated tail
/// weighing less than `2^-162`.
fn payne_hanek(a: f32) -> (usize, Dd) {
    let bits = a.to_bits();
    let be = (bits >> 23) as i32;
    let (mx, ex) = if be == 0 {
        (bits as u64, -149)
    } else {
        (((bits & 0x7f_ffff) | 0x80_0000) as u64, be - 150)
    };
    // The first kept bit (1-based) and the binary point: the product holds `a*32/pi mod 64`
    // as an integer scaled by `2^frac_bits`.
    let first = (ex - 1).max(1) as usize;
    let frac_bits = (first as i32 + 187 - ex) as u32;
    let w = [
        bits64(first - 1),
        bits64(first - 1 + 64),
        bits64(first - 1 + 128),
    ];
    // p = mx * w, little-endian 64-bit limbs.
    let mut p = [0u64; 4];
    let mut carry = 0u128;
    for (k, &wk) in w.iter().rev().enumerate() {
        let t = mx as u128 * wk as u128 + carry;
        p[k] = t as u64;
        carry = t >> 64;
    }
    p[3] = carry as u64;
    let int = (shr(&p, frac_bits) as u64 & 63) as usize;
    let mut frac = p;
    mask_low(&mut frac, frac_bits);
    let round_up = bit(&p, frac_bits - 1);
    let (j, mag) = if round_up {
        // 2^frac_bits - frac, the distance up to the next integer.
        ((int + 1) & 63, neg_mod(&frac, frac_bits))
    } else {
        (int, frac)
    };
    let f = to_dd(&mag, frac_bits);
    let r = f.mul(Dd::from_pair(PI_32));
    (j, if round_up { r.neg() } else { r })
}

/// [`payne_hanek`] out of line, for the fast path's large arguments (beyond any RoPE angle).
#[inline(never)]
fn payne_hanek_cold(a: f32) -> (usize, Dd) {
    payne_hanek(a)
}

/// Bits `p..p + 64` (0-based, most significant first) of `2/pi`'s expansion.
fn bits64(p: usize) -> u64 {
    let (w, o) = (p / 64, p % 64);
    if o == 0 {
        TWO_OVER_PI[w]
    } else {
        (TWO_OVER_PI[w] << o) | (TWO_OVER_PI[w + 1] >> (64 - o))
    }
}

/// Bit `n` of a little-endian 256-bit integer.
fn bit(p: &[u64; 4], n: u32) -> bool {
    n < 256 && (p[(n / 64) as usize] >> (n % 64)) & 1 == 1
}

/// The low 128 bits of `p >> n`.
fn shr(p: &[u64; 4], n: u32) -> u128 {
    let mut out = 0u128;
    for k in 0..128u32 {
        if bit(p, n + k) {
            out |= 1 << k;
        }
    }
    out
}

/// Clear every bit of `p` at position `n` and above.
fn mask_low(p: &mut [u64; 4], n: u32) {
    for (k, limb) in p.iter_mut().enumerate() {
        let lo = k as u32 * 64;
        if n <= lo {
            *limb = 0;
        } else if n < lo + 64 {
            *limb &= (1u64 << (n - lo)) - 1;
        }
    }
}

/// `2^n - p` for `0 < p < 2^n`, `n < 256`.
fn neg_mod(p: &[u64; 4], n: u32) -> [u64; 4] {
    // Two's complement over 256 bits, then keep the low n bits.
    let mut out = [0u64; 4];
    let mut borrow = 1u64; // ~p + 1
    for k in 0..4 {
        let (v, c) = (!p[k]).overflowing_add(borrow);
        out[k] = v;
        borrow = c as u64;
    }
    mask_low(&mut out, n);
    out
}

/// `m * 2^-scale` as a double-double, truncating `m` to its leading 106 bits (relative error
/// below `2^-105`).
fn to_dd(m: &[u64; 4], scale: u32) -> Dd {
    let Some(top) = (0..4).rev().find(|&k| m[k] != 0) else {
        return Dd::new(0.0, 0.0);
    };
    let len = top as i64 * 64 + 64 - m[top].leading_zeros() as i64;
    // 53 bits of m from bit `from` up; bits below position 0 read as zero.
    let take = |from: i64| -> u64 {
        (0..53i64)
            .filter(|&k| from + k >= 0 && bit(m, (from + k) as u32))
            .fold(0, |v, k| v | 1 << k)
    };
    let (hi_from, lo_from) = (len - 53, len - 106);
    let hi = take(hi_from) as f64 * exp2i((hi_from - scale as i64) as i32);
    let lo = take(lo_from) as f64 * exp2i((lo_from - scale as i64) as i32);
    fast_two_sum(hi, lo)
}
