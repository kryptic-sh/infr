//! `f32::mul_add` is a correctly rounded fused multiply-add on this platform and build.
//!
//! The CPU backend's f32 dots (`dot`, `dot_n`, `dot_mn`, `dot_f16`, `dot_bf16`, attention's value
//! accumulation) rest on `mul_add` giving the same bits everywhere. On a build with the `fma`
//! target feature it is the hardware instruction; on a baseline x86-64 build (CI, the released
//! binaries) it is a call into the C runtime's `fmaf` (UCRT on Windows, glibc on Linux), and on
//! aarch64 the hardware instruction again. IEEE 754 defines fused multiply-add as exactly one
//! rounding, so every correct implementation gives the same bits; this checks that the one in use
//! is correct, against a reference computed here:
//!
//! `a * b` is exact in binary64 (two 24-bit significands make at most 48 bits). Its sum with `c`
//! is computed exactly as a binary64 pair by TwoSum, then rounded ONCE to binary32 through
//! round-to-odd: the binary64 sum with its last bit forced odd when the pair is inexact, then
//! converted (Boldo and Melquiond, "When double rounding is odd", 2005: rounding to odd at 53
//! bits and then to nearest at 24 is the correct rounding, 53 >= 24 + 2). Plain `(a*b) as f64 +
//! c` then `as f32` rounds twice and is wrong on the hard cases below, which the reference is
//! checked against by known answers first.
//!
//! The inputs: values that cancel, sums landing just off a binary32 halfway point (the double
//! rounding traps), subnormal results, overflow, signed zeros, and a few million LCG bit
//! patterns. The first runs on Windows baseline, on macOS (aarch64) and on CI's Linux baseline
//! are CI's; this machine checked Windows and Linux, native and baseline.

/// Correctly rounded `a * b + c` in binary32, for finite inputs (see the module doc).
fn fma_ref(a: f32, b: f32, c: f32) -> f32 {
    let p = a as f64 * b as f64;
    let c = c as f64;
    let s = p + c;
    // TwoSum: `s + e == p + c` exactly.
    let bb = s - p;
    let e = (p - (s - bb)) + (c - bb);
    if e == 0.0 || !s.is_finite() {
        return s as f32;
    }
    let bits = s.to_bits();
    let odd = if bits & 1 == 1 {
        s
    } else if (e > 0.0) == (s > 0.0) {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    };
    odd as f32
}

/// The two results agree bit for bit (NaN with NaN, whatever its payload).
fn same(x: f32, y: f32) -> bool {
    x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
}

/// Hard cases with known answers: the sum lies a tiny amount off a binary32 halfway point, so a
/// binary64 sum rounds onto the halfway point and a second rounding goes the wrong way.
fn known_answers() -> Vec<(f32, f32, f32, f32)> {
    let mut v = Vec::new();
    // c's spacing is 2^(E-23); a*b = 2^(E-24) (1 - 2^-46): just below half a spacing above c.
    // With c's last significand bit odd, nearest-even of the halfway point would round up,
    // but the exact sum is below it: the answer is c.
    let a = 1.0 + f32::EPSILON; // 1 + 2^-23
    let b = 1.0 - f32::EPSILON; // 1 - 2^-23, so a*b = 1 - 2^-46
    for e in [-100i32, -20, 0, 1, 30, 100] {
        let c = (2f32.powi(e)) * (1.0 + f32::EPSILON); // odd last bit
        let half = 2f32.powi(e - 24);
        // a*b scaled to half a spacing: (1 - 2^-46) * 2^(e-24), split over the two factors.
        let (sa, sb) = (2f32.powi((e - 24) / 2), 2f32.powi((e - 24) - (e - 24) / 2));
        assert_eq!(sa * sb, half);
        v.push((a * sa, b * sb, c, c));
        // The mirror: just above the halfway point below an even neighbour... from the other
        // side: c - (half - tiny) rounds back to c.
        v.push((-(a * sa), b * sb, c, c));
        // Exactly halfway (a*b = half exactly): nearest-even, the even neighbour above.
        v.push((sa, sb, c, c + 2.0 * half));
    }
    // Cancellation to an exact small result: 1/3 in binary32 is 11184811 * 2^-25, so
    // 3 * (1/3) - 1 is exactly 2^-25 (where `3.0 * (1.0 / 3.0) - 1.0` rounds the product to 1
    // and gives 0); and to zero of each sign.
    v.push((3.0, 1.0 / 3.0, -1.0, 2f32.powi(-25)));
    v.push((2.0, 0.5, -1.0, 0.0));
    v.push((-0.0, 1.0, -0.0, -0.0));
    v.push((-0.0, 1.0, 0.0, 0.0));
    v
}

#[test]
fn reference_matches_the_known_answers() {
    for (i, &(a, b, c, want)) in known_answers().iter().enumerate() {
        let got = fma_ref(a, b, c);
        assert!(
            same(got, want),
            "case {i}: fma_ref({a:e}, {b:e}, {c:e}) = {got:e}, want {want:e}"
        );
    }
    // The cases are hard: plain double rounding gets some of them wrong.
    let naive_wrong = known_answers()
        .iter()
        .filter(|&&(a, b, c, want)| !same(((a as f64 * b as f64) + c as f64) as f32, want))
        .count();
    assert!(
        naive_wrong > 0,
        "no known answer separates double rounding from one rounding"
    );
}

#[test]
fn mul_add_is_correctly_rounded() {
    let mut cases: Vec<(f32, f32, f32)> = known_answers()
        .iter()
        .map(|&(a, b, c, _)| (a, b, c))
        .collect();
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 32) as u32
    };
    let finite = |bits: u32| {
        let f = f32::from_bits(bits);
        if f.is_finite() {
            f
        } else {
            f32::from_bits(bits & 0xBF7F_FFFF)
        }
    };
    for i in 0..3_000_000u32 {
        let (a, b) = (finite(next()), finite(next()));
        let c = match i % 4 {
            // Cancellation: c near -a*b.
            0 => -(a * b) * (1.0 + (next() % 7) as f32 * f32::EPSILON),
            // Small results: operands near the subnormal range.
            1 => finite(next() & 0x80FF_FFFF),
            2 => finite(next()),
            _ => -(a * b),
        };
        let (a, b) = if i % 4 == 1 {
            (finite(next() & 0x9FFF_FFFF), finite(next() & 0x9FFF_FFFF))
        } else {
            (a, b)
        };
        if c.is_finite() {
            cases.push((a, b, c));
        }
    }
    // Overflow and the largest finite values.
    cases.push((f32::MAX, 2.0, -f32::MAX));
    cases.push((f32::MAX, 1.5, 0.0));
    cases.push((-f32::MAX, f32::MAX, f32::MAX));
    let mut wrong = Vec::new();
    for &(a, b, c) in &cases {
        let (got, want) = (a.mul_add(b, c), fma_ref(a, b, c));
        if !same(got, want) {
            wrong.push(format!(
                "mul_add({:#010x}, {:#010x}, {:#010x}) = {:#010x}, correctly rounded {:#010x}",
                a.to_bits(),
                b.to_bits(),
                c.to_bits(),
                got.to_bits(),
                want.to_bits()
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} cases not correctly rounded:\n{}",
        wrong.len(),
        cases.len(),
        wrong[..wrong.len().min(20)].join("\n")
    );
}
