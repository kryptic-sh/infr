//! How often each function's rounding check fires, and whether its binary64 fast path ALONE
//! (rounded straight to binary32) already gives the correctly rounded result, over every
//! binary32 input. The answer decides where the check can be dropped; the test pins it.

use super::dd::round_if_clear;
use super::exp::{exp_accurate, exp_fast, tanh_accurate, tanh_fast, EXP_EPS, TANH_EPS, TANH_ONE};
use super::log::{abs, ln_1p_fast, ln_fast, LN_EPS};
use super::trig::{sin_cos_fast, TRIG_EPS};

/// `(inputs where the check fires, inputs where the fast value rounded alone differs from the
/// function)` over all 2^32 inputs, for one function, on every core.
fn walk(case: fn(f32) -> Option<(bool, bool)>) -> (u64, u64) {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()) as u64;
    let chunk = (1u64 << 32).div_ceil(threads);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let (mut fires, mut differs) = (0u64, 0u64);
                    for b in t * chunk..((t + 1) * chunk).min(1 << 32) {
                        if let Some((f, d)) = case(f32::from_bits(b as u32)) {
                            fires += f as u64;
                            differs += d as u64;
                        }
                    }
                    (fires, differs)
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().unwrap())
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    })
}

/// Print one function's counts (to stdout explicitly: this file is test-only, which the
/// repo's print lint cannot see, as it is declared from `mod.rs`) and return the second.
fn report(name: &str, (fires, differs): (u64, u64)) -> u64 {
    use std::io::Write;
    writeln!(
        std::io::stdout(),
        "{name}: rounding check fires on {fires} inputs; fast path alone wrong on {differs}"
    )
    .expect("stdout");
    differs
}

#[test]
#[ignore = "exhaustive: all 2^32 inputs per function; run in release"]
fn fast_path_alone_against_the_correctly_rounded_result() {
    let exp = walk(|x| {
        if !(-104.0..=89.0).contains(&x) {
            return None;
        }
        let y = exp_fast(x as f64);
        Some((
            round_if_clear(y, y * EXP_EPS).is_none(),
            (y as f32).to_bits() != exp_accurate(x as f64).to_bits(),
        ))
    });
    let tanh = walk(|x| {
        let a = f32::from_bits(x.to_bits() & 0x7fff_ffff);
        if !(a > 0.0 && a <= TANH_ONE) {
            return None;
        }
        let y = tanh_fast(2.0 * a as f64);
        Some((
            round_if_clear(y, y * TANH_EPS).is_none(),
            (y as f32).to_bits() != tanh_accurate(2.0 * a as f64).to_bits(),
        ))
    });
    let ln = walk(|x| {
        if !(x > 0.0 && x.is_finite()) {
            return None;
        }
        let y = ln_fast(x as f64, 0.0);
        Some((
            round_if_clear(y, abs(y) * LN_EPS).is_none(),
            (y as f32).to_bits() != super::ln(x).to_bits(),
        ))
    });
    let ln_1p = walk(|x| {
        if !(x > -1.0 && x.is_finite() && x != 0.0) {
            return None;
        }
        let y = ln_1p_fast(x);
        Some((
            round_if_clear(y, abs(y) * LN_EPS).is_none(),
            (y as f32).to_bits() != super::ln_1p(x).to_bits(),
        ))
    });
    let sin = walk(|x| trig(x, true));
    let cos = walk(|x| trig(x, false));
    // `exp` and `tanh` round their binary64 value directly on the strength of this.
    assert_eq!(
        report("exp", exp),
        0,
        "exp: the binary64 value alone is no longer exact"
    );
    assert_eq!(
        report("tanh", tanh),
        0,
        "tanh: the binary64 value alone is no longer exact"
    );
    report("ln", ln);
    report("ln_1p", ln_1p);
    report("sin", sin);
    report("cos", cos);
}

fn trig(x: f32, want_sin: bool) -> Option<(bool, bool)> {
    if !(x.is_finite() && x != 0.0) {
        return None;
    }
    let (s, c, e) = sin_cos_fast(abs(x as f64));
    let y = if want_sin { s } else { c };
    let mut v = y as f32;
    if want_sin && x < 0.0 {
        v = -v;
    }
    let want = if want_sin {
        super::sin(x)
    } else {
        super::cos(x)
    };
    Some((
        round_if_clear(y, abs(y) * TRIG_EPS + e).is_none(),
        v.to_bits() != want.to_bits(),
    ))
}
