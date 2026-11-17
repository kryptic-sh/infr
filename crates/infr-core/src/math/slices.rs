//! The slice functions, compiled once per SIMD tier and chosen at run time.
//!
//! Each slice function is a plain loop applying one branch-free lane function (the same one the
//! scalar function IS) to every element. That loop body is compiled several times: once for the
//! build's own target ([`Tier::Baseline`]) and, on x86-64, again inside functions carrying
//! `#[target_feature(enable = ...)]` for AVX2 and AVX-512, so a binary built for baseline x86-64
//! (the released binaries, CI) still runs four or eight binary64 lanes on a CPU that has them.
//! The first call picks the widest tier the CPU supports; [`set_tier`] chooses one instead (the
//! `kernels.cpu.math_tier` setting, which every CPU backend applies when it is built). The
//! active tier is process-wide: one setting for every thread and every backend.
//!
//! **Why every tier gives the same bits.** The tiers are the SAME Rust source, so they perform
//! the same IEEE 754 operations in the same order on each element; vectorizing only changes how
//! many elements run at once. Rust never contracts `a*b + c` into a fused multiply-add or
//! reassociates floating-point operations, with or without a target feature, so a wider tier
//! cannot round differently. FMA instructions may well be available to the compiler in the wide
//! tiers (enabling `avx512f` implies the `fma` target feature, and a `target-cpu=native` build
//! has it in every tier); the bits rest on the language never contracting without an explicit
//! `mul_add`, which `infr_core::math` never calls. `tests/math.rs` checks every tier the CPU
//! supports against the scalar functions on a dense grid in every test run, and on all `2^32`
//! inputs in an ignored test; a contraction would make them fail.

use std::sync::atomic::{AtomicU8, Ordering};

use super::exp::{exp_lane, tanh_lane};
use super::{gelu_tanh_arg, gelu_tanh_finish, sigmoid_finish, silu_finish};

/// A SIMD tier the slice functions are compiled for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    /// The build's own target features: SSE2 on a baseline x86-64 build, NEON on aarch64, and
    /// whatever `-C target-cpu` enables otherwise.
    Baseline,
    /// AVX2 (x86-64 only). FMA may be available to the compiler here; see the module doc for
    /// why it cannot change a result.
    Avx2,
    /// AVX-512 F, DQ, VL and BW (x86-64 only). These imply the `fma` target feature; see the
    /// module doc for why it cannot change a result.
    Avx512,
}

impl Tier {
    /// Every tier, narrowest first.
    pub const ALL: [Tier; 3] = [Tier::Baseline, Tier::Avx2, Tier::Avx512];

    /// The tier's name as `kernels.cpu.math_tier` spells it.
    pub fn name(self) -> &'static str {
        match self {
            Tier::Baseline => "baseline",
            Tier::Avx2 => "avx2",
            Tier::Avx512 => "avx512",
        }
    }

    /// The tier named `s` (as [`Tier::name`] spells it, ASCII case ignored).
    pub fn parse(s: &str) -> Option<Tier> {
        Tier::ALL
            .into_iter()
            .find(|t| t.name().eq_ignore_ascii_case(s.trim()))
    }

    /// Whether this CPU can run the tier.
    pub fn supported(self) -> bool {
        match self {
            Tier::Baseline => true,
            #[cfg(target_arch = "x86_64")]
            Tier::Avx2 => is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Tier::Avx512 => {
                is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512dq")
                    && is_x86_feature_detected!("avx512vl")
                    && is_x86_feature_detected!("avx512bw")
            }
            #[cfg(not(target_arch = "x86_64"))]
            Tier::Avx2 | Tier::Avx512 => false,
        }
    }

    /// The widest tier this CPU supports.
    pub fn best() -> Tier {
        Tier::ALL
            .into_iter()
            .rev()
            .find(|t| t.supported())
            .unwrap_or(Tier::Baseline)
    }
}

/// One tier's slice functions. Every entry equals the scalar function of the same name element
/// for element (see the module doc); the slice functions of [`super`] call the active tier's.
#[derive(Debug)]
pub struct SliceFns {
    /// The tier these were compiled for.
    pub tier: Tier,
    /// [`exp_slice`].
    pub exp: fn(&mut [f32]),
    /// [`exp_slice_to`] (the lengths are checked by the caller).
    pub exp_to: fn(&[f32], &mut [f32]),
    /// [`exp_sub_slice`].
    pub exp_sub: fn(&mut [f32], f32),
    /// [`tanh_slice`].
    pub tanh: fn(&mut [f32]),
    /// [`sigmoid_slice`].
    pub sigmoid: fn(&mut [f32]),
    /// [`silu_slice`].
    pub silu: fn(&mut [f32]),
    /// [`gelu_tanh_slice`].
    pub gelu_tanh: fn(&mut [f32]),
}

// The loop bodies, inlined into each tier's functions so each is compiled with that tier's
// features. `B` is the loop's block: 1 is a plain loop, which the loop vectorizer widens to the
// tier's vectors; a larger block writes `B` independent lanes out per iteration, which the SLP
// vectorizer packs into two-lane vectors where the loop vectorizer would not (it judges the
// two-lane `exp` loop unprofitable at baseline x86-64) — see `NARROW`. Each element still goes
// through the same lane function either way.

/// `xs[i] = f(xs[i])`, in blocks of `B` (a plain loop for 1).
#[inline(always)]
fn map<const B: usize>(xs: &mut [f32], f: impl Fn(f32) -> f32) {
    if B == 1 {
        for x in xs {
            *x = f(*x);
        }
        return;
    }
    let (blocks, tail) = xs.as_chunks_mut::<B>();
    for c in blocks {
        let mut v = [0f32; B];
        for (y, &x) in v.iter_mut().zip(c.iter()) {
            *y = f(x);
        }
        *c = v;
    }
    for x in tail {
        *x = f(*x);
    }
}

/// `dst[i] = f(src[i])`, in blocks of `B` (a plain loop for 1), for equal lengths.
#[inline(always)]
fn map_to<const B: usize>(src: &[f32], dst: &mut [f32], f: impl Fn(f32) -> f32) {
    if B == 1 {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = f(s);
        }
        return;
    }
    let (sb, st) = src.as_chunks::<B>();
    let (db, dt) = dst.as_chunks_mut::<B>();
    for (d, s) in db.iter_mut().zip(sb) {
        let mut v = [0f32; B];
        for (y, &x) in v.iter_mut().zip(s.iter()) {
            *y = f(x);
        }
        *d = v;
    }
    for (d, &s) in dt.iter_mut().zip(st) {
        *d = f(s);
    }
}

#[inline(always)]
fn exp_body<const B: usize>(xs: &mut [f32]) {
    map::<B>(xs, exp_lane)
}

#[inline(always)]
fn exp_to_body<const B: usize>(src: &[f32], dst: &mut [f32]) {
    map_to::<B>(src, dst, exp_lane)
}

#[inline(always)]
fn exp_sub_body<const B: usize>(xs: &mut [f32], m: f32) {
    map::<B>(xs, |x| exp_lane(x - m))
}

#[inline(always)]
fn tanh_body<const B: usize>(xs: &mut [f32]) {
    map::<B>(xs, tanh_lane)
}

// The helpers in two passes per `PASS` elements, at every tier: the `exp`/`tanh` lanes in
// blocks of `B` into a buffer, then the helper's binary32 arithmetic (the same `*_finish` its
// lane function ends with) in a plain loop, which runs at twice the lanes the binary64 part
// can, the division included. Measured faster at every tier than one fused loop (docs/math.md).

/// Elements per pass of the two-pass helpers.
const PASS: usize = 64;

#[inline(always)]
fn sigmoid_body<const B: usize>(xs: &mut [f32]) {
    for c in xs.chunks_mut(PASS) {
        let mut e = [0f32; PASS];
        let e = &mut e[..c.len()];
        map_to::<B>(c, e, |x| exp_lane(-x));
        for (y, &e) in c.iter_mut().zip(e.iter()) {
            *y = sigmoid_finish(e);
        }
    }
}

#[inline(always)]
fn silu_body<const B: usize>(xs: &mut [f32]) {
    for c in xs.chunks_mut(PASS) {
        let mut e = [0f32; PASS];
        let e = &mut e[..c.len()];
        map_to::<B>(c, e, |x| exp_lane(-x));
        for (y, &e) in c.iter_mut().zip(e.iter()) {
            *y = silu_finish(*y, e);
        }
    }
}

#[inline(always)]
fn gelu_tanh_body<const B: usize>(xs: &mut [f32]) {
    for c in xs.chunks_mut(PASS) {
        let mut t = [0f32; PASS];
        let t = &mut t[..c.len()];
        for (t, &x) in t.iter_mut().zip(c.iter()) {
            *t = gelu_tanh_arg(x);
        }
        map::<B>(t, tanh_lane);
        for (y, &t) in c.iter_mut().zip(t.iter()) {
            *y = gelu_tanh_finish(*y, t);
        }
    }
}

/// The block of the baseline tier, the two-lane path (SSE2, NEON). Chosen by measurement on
/// x86-64; see docs/math.md.
const NARROW: usize = 4;

static BASELINE: SliceFns = SliceFns {
    tier: Tier::Baseline,
    exp: exp_body::<NARROW>,
    exp_to: exp_to_body::<NARROW>,
    exp_sub: exp_sub_body::<NARROW>,
    tanh: tanh_body::<NARROW>,
    sigmoid: sigmoid_body::<NARROW>,
    silu: silu_body::<NARROW>,
    gelu_tanh: gelu_tanh_body::<NARROW>,
};

/// One x86-64 tier: the loop bodies compiled with `$features`, and their table. The table is
/// only reachable through [`slice_fns`], which checks [`Tier::supported`] first; that check is
/// what makes calling the `target_feature` functions sound.
#[cfg(target_arch = "x86_64")]
macro_rules! x86_tier {
    ($module:ident, $tier:expr, $features:literal) => {
        mod $module {
            use super::*;

            #[target_feature(enable = $features)]
            unsafe fn exp(xs: &mut [f32]) {
                exp_body::<1>(xs)
            }
            #[target_feature(enable = $features)]
            unsafe fn exp_to(src: &[f32], dst: &mut [f32]) {
                exp_to_body::<1>(src, dst)
            }
            #[target_feature(enable = $features)]
            unsafe fn exp_sub(xs: &mut [f32], m: f32) {
                exp_sub_body::<1>(xs, m)
            }
            #[target_feature(enable = $features)]
            unsafe fn tanh(xs: &mut [f32]) {
                tanh_body::<1>(xs)
            }
            #[target_feature(enable = $features)]
            unsafe fn sigmoid(xs: &mut [f32]) {
                sigmoid_body::<1>(xs)
            }
            #[target_feature(enable = $features)]
            unsafe fn silu(xs: &mut [f32]) {
                silu_body::<1>(xs)
            }
            #[target_feature(enable = $features)]
            unsafe fn gelu_tanh(xs: &mut [f32]) {
                gelu_tanh_body::<1>(xs)
            }

            // SAFETY (every entry): `FNS` is handed out only by `slice_fns`, after
            // `Tier::supported` confirmed the CPU has `$features`.
            pub(super) static FNS: SliceFns = SliceFns {
                tier: $tier,
                exp: |xs| unsafe { exp(xs) },
                exp_to: |s, d| unsafe { exp_to(s, d) },
                exp_sub: |xs, m| unsafe { exp_sub(xs, m) },
                tanh: |xs| unsafe { tanh(xs) },
                sigmoid: |xs| unsafe { sigmoid(xs) },
                silu: |xs| unsafe { silu(xs) },
                gelu_tanh: |xs| unsafe { gelu_tanh(xs) },
            };
        }
    };
}

#[cfg(target_arch = "x86_64")]
x86_tier!(avx2, Tier::Avx2, "avx2");
#[cfg(target_arch = "x86_64")]
x86_tier!(avx512, Tier::Avx512, "avx512f,avx512dq,avx512vl,avx512bw");

/// The slice functions compiled for `tier`, or `None` if this CPU cannot run them.
pub fn slice_fns(tier: Tier) -> Option<&'static SliceFns> {
    if !tier.supported() {
        return None;
    }
    match tier {
        Tier::Baseline => Some(&BASELINE),
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 => Some(&avx2::FNS),
        #[cfg(target_arch = "x86_64")]
        Tier::Avx512 => Some(&avx512::FNS),
        #[cfg(not(target_arch = "x86_64"))]
        Tier::Avx2 | Tier::Avx512 => None,
    }
}

/// The active tier, as `index + 1` into [`Tier::ALL`]; 0 until the first call resolves it.
static ACTIVE: AtomicU8 = AtomicU8::new(0);

fn encode(t: Tier) -> u8 {
    Tier::ALL.iter().position(|&a| a == t).expect("a tier") as u8 + 1
}

fn decode(n: u8) -> Tier {
    Tier::ALL[n as usize - 1]
}

/// The tier the slice functions run: the one [`set_tier`] last chose, else [`Tier::best`],
/// installed on first use.
pub fn tier() -> Tier {
    match ACTIVE.load(Ordering::Relaxed) {
        0 => install_first(Tier::best()),
        n => decode(n),
    }
}

/// Install `auto` as the active tier only if nothing is active yet, and return the active tier:
/// a first use racing a [`set_tier`] must not overwrite the tier that call chose.
fn install_first(auto: Tier) -> Tier {
    match ACTIVE.compare_exchange(0, encode(auto), Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => auto,
        Err(n) => decode(n),
    }
}

/// Run every slice function at `tier` from now on, or at [`Tier::best`] for `None`. This is
/// the `kernels.cpu.math_tier` debug setting, and it is process-wide: the last call wins for
/// every thread and every backend. Only speed is at stake; the result bits are the same at
/// every tier.
///
/// # Errors
/// If this CPU cannot run `tier`; the active tier is then unchanged.
pub fn set_tier(tier: Option<Tier>) -> Result<(), String> {
    let tier = match tier {
        None => Tier::best(),
        Some(t) if t.supported() => t,
        Some(t) => {
            return Err(format!(
                "math tier `{}` is not supported by this CPU (best: `{}`)",
                t.name(),
                Tier::best().name()
            ))
        }
    };
    ACTIVE.store(encode(tier), Ordering::Relaxed);
    Ok(())
}

#[inline]
fn active() -> &'static SliceFns {
    slice_fns(tier()).unwrap_or(&BASELINE)
}

/// `xs[i] = exp(xs[i])` for every element.
pub fn exp_slice(xs: &mut [f32]) {
    (active().exp)(xs)
}

/// `dst[i] = exp(src[i])`.
///
/// # Panics
/// If the two slices differ in length.
pub fn exp_slice_to(src: &[f32], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "exp_slice_to: length mismatch");
    (active().exp_to)(src, dst)
}

/// `xs[i] = exp(xs[i] - m)`, the subtraction rounded to binary32 first, as a softmax over a
/// row writes `(x - max).exp()`.
pub fn exp_sub_slice(xs: &mut [f32], m: f32) {
    (active().exp_sub)(xs, m)
}

/// `xs[i] = tanh(xs[i])`.
pub fn tanh_slice(xs: &mut [f32]) {
    (active().tanh)(xs)
}

/// `xs[i] = sigmoid(xs[i])`.
pub fn sigmoid_slice(xs: &mut [f32]) {
    (active().sigmoid)(xs)
}

/// `xs[i] = silu(xs[i])`.
pub fn silu_slice(xs: &mut [f32]) {
    (active().silu)(xs)
}

/// `xs[i] = gelu_tanh(xs[i])`.
pub fn gelu_tanh_slice(xs: &mut [f32]) {
    (active().gelu_tanh)(xs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A first use that lost the race to [`set_tier`] keeps the tier `set_tier` chose. Turns red
    /// if `install_first` stores unconditionally (the race the compare-exchange closes).
    #[test]
    fn first_use_never_overwrites_a_chosen_tier() {
        set_tier(Some(Tier::Baseline)).unwrap();
        // A first use that read 0 before `set_tier` ran, now installing the auto tier.
        let auto = Tier::ALL[Tier::ALL.len() - 1];
        assert_eq!(install_first(auto), Tier::Baseline);
        assert_eq!(tier(), Tier::Baseline);
        set_tier(None).unwrap();
        assert_eq!(tier(), Tier::best());
    }
}
