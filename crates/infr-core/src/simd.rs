//! The SIMD tiers the CPU backend's code is compiled for, and the process-wide ceiling on them:
//! `kernels.cpu.max_tier` (`INFR_CPU_MAX_TIER`).
//!
//! The CPU backend's quantized dot kernels and [`crate::math`]'s slice functions each exist in
//! several versions, one per instruction set, and pick one at run time from the CPU's features.
//! [`active`] is the tier they all ask: the widest one the CPU has, or a narrower one when
//! [`set_max`] lowered the ceiling. One ceiling covers both, so a capped run executes the code an
//! older CPU (or one without the instructions, at [`SimdTier::Scalar`]) would.
//!
//! Each architecture has its own tiers above the shared [`SimdTier::Scalar`]: AVX2, AVX-512 and
//! AVX-512 VNNI on x86-64, the dot-product extension on aarch64 ([`SimdTier::ALL`] lists this
//! build's). Every tier's name parses on every architecture, so one configuration file serves
//! both; another architecture's tier ranks above every tier of this one, so as a ceiling it is a
//! tier this CPU lacks and is clamped like one: it runs the CPU's widest.
//!
//! Every tier computes the same value for every input: integer accumulations are exact in any
//! order, and each floating-point expression is written once and evaluated in the same order at
//! every tier (`infr-cpu`'s `kernels/tier_tests.rs` and `tests/math.rs` hold every kernel family
//! and every slice function to that). The ceiling therefore changes only speed. It exists to run
//! and test the narrow paths on a machine that has wide ones.
//!
//! The tiers are cumulative: each one's [`SimdTier::supported`] requires every narrower tier's
//! features too, so a code path gated on "the active tier is at least X" may use the features of
//! X and of every tier below it.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::math;

/// Where the x86-64 tiers' discriminants start: right after [`SimdTier::Scalar`] on x86-64, and
/// above every native tier elsewhere, so there a foreign tier compares above the CPU's widest.
const X86_BASE: u8 = if cfg!(target_arch = "x86_64") { 1 } else { 16 };
/// Where the aarch64 tiers' discriminants start: [`X86_BASE`]'s rule for aarch64, and clear of
/// the x86-64 range on a third architecture, where both are foreign.
const ARM_BASE: u8 = if cfg!(target_arch = "aarch64") { 1 } else { 32 };

/// A SIMD tier. The order is the ceiling's order: this architecture's tiers run narrowest first
/// from [`SimdTier::Scalar`], and every other architecture's tiers rank above all of them (the
/// discriminants are chosen per target for that), so `min` against the CPU's widest tier clamps
/// a foreign ceiling to it and [`allows`] never allows a foreign tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SimdTier {
    /// The portable code: the kernels' `*_scalar` functions and the math slices' baseline
    /// loops, compiled for the build's own target (SSE2 on a baseline x86-64 build, NEON on
    /// aarch64, whatever `-C target-cpu` enables otherwise). The only tier on any other
    /// architecture.
    Scalar = 0,
    /// AVX2 (x86-64).
    Avx2 = X86_BASE,
    /// AVX-512 F, DQ, VL and BW (x86-64), with AVX2.
    Avx512 = X86_BASE + 1,
    /// AVX-512 VNNI on top of [`SimdTier::Avx512`] (x86-64): the int8 GEMM and batch kernels
    /// that have a VNNI form. The math slices have none and run their AVX-512 loops here.
    Avx512Vnni = X86_BASE + 2,
    /// NEON with the dot-product extension (aarch64 `dotprod`, Armv8.2's `sdot`): the int8
    /// kernels that have a NEON form. Every Apple M-series core has it, as do recent Arm server
    /// cores. The math slices have no form of their own and run their baseline loops here.
    Dotprod = ARM_BASE,
}

impl SimdTier {
    /// This architecture's tiers, narrowest first: the tiers a CPU running this build can have,
    /// which every "each tier the host has" loop walks.
    #[cfg(target_arch = "x86_64")]
    pub const ALL: [SimdTier; 4] = [
        SimdTier::Scalar,
        SimdTier::Avx2,
        SimdTier::Avx512,
        SimdTier::Avx512Vnni,
    ];
    /// This architecture's tiers, narrowest first: the tiers a CPU running this build can have,
    /// which every "each tier the host has" loop walks.
    #[cfg(target_arch = "aarch64")]
    pub const ALL: [SimdTier; 2] = [SimdTier::Scalar, SimdTier::Dotprod];
    /// This architecture's tiers, narrowest first: the tiers a CPU running this build can have,
    /// which every "each tier the host has" loop walks.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub const ALL: [SimdTier; 1] = [SimdTier::Scalar];

    /// Every tier of every architecture: the names `kernels.cpu.max_tier` accepts.
    pub const NAMED: [SimdTier; 5] = [
        SimdTier::Scalar,
        SimdTier::Avx2,
        SimdTier::Avx512,
        SimdTier::Avx512Vnni,
        SimdTier::Dotprod,
    ];

    /// The tier's name as `kernels.cpu.max_tier` spells it.
    pub fn name(self) -> &'static str {
        match self {
            SimdTier::Scalar => "scalar",
            SimdTier::Avx2 => "avx2",
            SimdTier::Avx512 => "avx512",
            SimdTier::Avx512Vnni => "avx512vnni",
            SimdTier::Dotprod => "dotprod",
        }
    }

    /// The tier named `s` (as [`SimdTier::name`] spells it, ASCII case ignored), of any
    /// architecture.
    pub fn parse(s: &str) -> Option<SimdTier> {
        SimdTier::NAMED
            .into_iter()
            .find(|t| t.name().eq_ignore_ascii_case(s.trim()))
    }

    /// Whether the tier is one of this architecture's ([`SimdTier::ALL`]).
    pub fn native(self) -> bool {
        SimdTier::ALL.contains(&self)
    }

    /// Whether this CPU can run the tier, and with it every narrower one.
    pub fn supported(self) -> bool {
        match self {
            SimdTier::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx2 => is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx512 => {
                SimdTier::Avx2.supported()
                    && is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512dq")
                    && is_x86_feature_detected!("avx512vl")
                    && is_x86_feature_detected!("avx512bw")
            }
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx512Vnni => {
                SimdTier::Avx512.supported() && is_x86_feature_detected!("avx512vnni")
            }
            #[cfg(not(target_arch = "x86_64"))]
            SimdTier::Avx2 | SimdTier::Avx512 | SimdTier::Avx512Vnni => false,
            // NEON is part of every aarch64 target; `dotprod` is part of the Apple ones (a
            // constant there) and a run-time feature on the others.
            #[cfg(target_arch = "aarch64")]
            SimdTier::Dotprod => {
                cfg!(target_feature = "dotprod")
                    || std::arch::is_aarch64_feature_detected!("dotprod")
            }
            #[cfg(not(target_arch = "aarch64"))]
            SimdTier::Dotprod => false,
        }
    }

    /// The widest tier this CPU supports.
    pub fn best() -> SimdTier {
        SimdTier::ALL
            .into_iter()
            .rev()
            .find(|t| t.supported())
            .unwrap_or(SimdTier::Scalar)
    }

    /// The math slices' tier this one runs ([`math::Tier`]: they have no VNNI form, and no
    /// aarch64 form beyond the baseline loops).
    pub fn math(self) -> math::Tier {
        match self {
            SimdTier::Scalar | SimdTier::Dotprod => math::Tier::Baseline,
            SimdTier::Avx2 => math::Tier::Avx2,
            SimdTier::Avx512 | SimdTier::Avx512Vnni => math::Tier::Avx512,
        }
    }
}

/// The active tier, as `index + 1` into [`SimdTier::ALL`]; 0 until the first call resolves it.
/// Only this architecture's tiers are ever stored, and each one's discriminant is its index in
/// `ALL` (`names_and_order` holds that), so every other architecture's tier compares above every
/// stored value.
static ACTIVE: AtomicU8 = AtomicU8::new(0);

fn encode(t: SimdTier) -> u8 {
    debug_assert!(t.native(), "{t:?} is not one of this architecture's tiers");
    t as u8 + 1
}

fn decode(n: u8) -> SimdTier {
    SimdTier::ALL[n as usize - 1]
}

/// The tier code runs at: the one [`set_max`] last chose, else [`SimdTier::best`], installed on
/// first use.
#[inline]
pub fn active() -> SimdTier {
    match ACTIVE.load(Ordering::Relaxed) {
        0 => install_first(SimdTier::best()),
        n => decode(n),
    }
}

/// Whether code of tier `t` may run: `active() >= t`, as one relaxed load and one integer
/// comparison. Every SIMD dispatch in the CPU backend asks this per call (per weight row, for the
/// batch kernels), so it is kept to that. False for every other architecture's tier.
#[inline(always)]
pub fn allows(t: SimdTier) -> bool {
    match ACTIVE.load(Ordering::Relaxed) {
        0 => allows_first(t),
        // `encode(a) = a + 1`, so `a >= t` is `encode(a) > t`.
        n => n > t as u8,
    }
}

#[cold]
#[inline(never)]
fn allows_first(t: SimdTier) -> bool {
    active() >= t
}

/// Install `auto` as the active tier only if nothing is active yet, and return the active tier:
/// a first use racing a [`set_max`] must not overwrite the tier that call chose.
#[cold]
fn install_first(auto: SimdTier) -> SimdTier {
    match ACTIVE.compare_exchange(0, encode(auto), Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => auto,
        Err(n) => decode(n),
    }
}

/// Whether a ceiling below [`SimdTier::best`] is in force: false until [`set_max`] lowers it.
static CAPPED: AtomicBool = AtomicBool::new(false);

/// Whether [`set_max`] put a ceiling below [`SimdTier::best`] in force. The CPU backend reads it
/// once per graph to choose between its two compilations of the interpreter: with no ceiling, the
/// dispatch it compiles from the CPU feature checks alone, which a build that already has the
/// features (`target-cpu=native`, or `dotprod` on an Apple target) folds to a constant; under
/// one, the dispatch that asks [`allows`] at every call.
#[inline]
pub fn capped() -> bool {
    CAPPED.load(Ordering::Relaxed)
}

/// Run at most `max` from now on (`None`: no ceiling, the widest tier the CPU has), and return
/// the tier that is now active. A ceiling above what the CPU supports is clamped to
/// [`SimdTier::best`]: it is a ceiling, not a demand. Another architecture's tier ranks above
/// every tier of this one, so it is clamped the same way. This is the `kernels.cpu.max_tier`
/// debug setting, and it is process-wide: the last call wins for every thread and every backend.
pub fn set_max(max: Option<SimdTier>) -> SimdTier {
    let best = SimdTier::best();
    let tier = max.map_or(best, |m| m.min(best));
    ACTIVE.store(encode(tier), Ordering::Relaxed);
    CAPPED.store(tier < best, Ordering::Relaxed);
    tier
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Names round-trip for every architecture's tiers, `ALL` is in the order the ceiling
    /// compares (the `Ord` derive and `encode` both rest on it), and every other architecture's
    /// tier ranks above this one's and is never supported.
    #[test]
    fn names_and_order() {
        for (i, t) in SimdTier::ALL.into_iter().enumerate() {
            assert!(t.native());
            assert_eq!(decode(encode(t)), t);
            assert_eq!(t as usize, i);
        }
        for t in SimdTier::NAMED {
            assert_eq!(SimdTier::parse(t.name()), Some(t));
            assert_eq!(SimdTier::parse(&t.name().to_uppercase()), Some(t));
        }
        assert!(SimdTier::ALL.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(SimdTier::parse("baseline"), None);
        assert_eq!(SimdTier::parse("neon"), None);
        let widest = SimdTier::ALL[SimdTier::ALL.len() - 1];
        let foreign: Vec<SimdTier> = SimdTier::NAMED
            .into_iter()
            .filter(|t| !t.native())
            .collect();
        assert_eq!(foreign.len() + SimdTier::ALL.len(), SimdTier::NAMED.len());
        for t in foreign {
            assert!(t > widest && !t.supported(), "{t:?}");
        }
        // Supported tiers are a prefix of `ALL`, the tiers being cumulative.
        let n = SimdTier::ALL.iter().filter(|t| t.supported()).count();
        assert!(SimdTier::ALL[..n].iter().all(|t| t.supported()));
        assert_eq!(SimdTier::best(), SimdTier::ALL[n - 1]);
    }

    /// The ceiling clamps to what the CPU has (another architecture's tier included), `None`
    /// restores the widest, and a first use that lost the race to [`set_max`] keeps the tier
    /// `set_max` chose. Turns red if `set_max` stores a tier the CPU lacks or `install_first`
    /// stores unconditionally.
    #[test]
    fn ceiling_clamps_and_first_use_never_overwrites() {
        for t in SimdTier::NAMED.into_iter().filter(|t| !t.supported()) {
            assert_eq!(set_max(Some(t)), SimdTier::best(), "{t:?}");
            assert_eq!(active(), SimdTier::best());
            assert!(!capped());
            assert!(!allows(t), "{t:?}");
        }
        assert_eq!(set_max(Some(SimdTier::Scalar)), SimdTier::Scalar);
        for t in SimdTier::NAMED {
            assert_eq!(allows(t), t == SimdTier::Scalar, "{t:?}");
        }
        assert_eq!(capped(), SimdTier::best() > SimdTier::Scalar);
        assert_eq!(install_first(SimdTier::best()), SimdTier::Scalar);
        assert_eq!(active(), SimdTier::Scalar);
        assert_eq!(math::tier(), math::Tier::Baseline);
        assert_eq!(set_max(None), SimdTier::best());
        assert_eq!(active(), SimdTier::best());
        assert!(!capped());
        for t in SimdTier::NAMED {
            assert_eq!(allows(t), t.supported(), "{t:?}");
        }
        assert_eq!(math::tier(), SimdTier::best().math());
    }
}
