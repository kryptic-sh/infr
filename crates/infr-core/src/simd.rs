//! The SIMD tiers the CPU backend's code is compiled for, and the process-wide ceiling on them:
//! `kernels.cpu.max_tier` (`INFR_CPU_MAX_TIER`).
//!
//! The CPU backend's quantized dot kernels and [`crate::math`]'s slice functions each exist in
//! several versions, one per instruction set, and pick one at run time from the CPU's features.
//! [`active`] is the tier they all ask: the widest one the CPU has, or a narrower one when
//! [`set_max`] lowered the ceiling. One ceiling covers both, so a capped run executes the code an
//! older CPU (or an aarch64 machine, at [`SimdTier::Scalar`]) would.
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

use std::sync::atomic::{AtomicU8, Ordering};

use crate::math;

/// A SIMD tier, narrowest first: the order is the ceiling's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdTier {
    /// The portable code: the kernels' `*_scalar` functions and the math slices' baseline
    /// loops, compiled for the build's own target (SSE2 on a baseline x86-64 build, NEON on
    /// aarch64, whatever `-C target-cpu` enables otherwise). The only tier off x86-64.
    Scalar,
    /// AVX2 (x86-64).
    Avx2,
    /// AVX-512 F, DQ, VL and BW (x86-64), with AVX2.
    Avx512,
    /// AVX-512 VNNI on top of [`SimdTier::Avx512`] (x86-64): the int8 GEMM and batch kernels
    /// that have a VNNI form. The math slices have none and run their AVX-512 loops here.
    Avx512Vnni,
}

impl SimdTier {
    /// Every tier, narrowest first.
    pub const ALL: [SimdTier; 4] = [
        SimdTier::Scalar,
        SimdTier::Avx2,
        SimdTier::Avx512,
        SimdTier::Avx512Vnni,
    ];

    /// The tier's name as `kernels.cpu.max_tier` spells it.
    pub fn name(self) -> &'static str {
        match self {
            SimdTier::Scalar => "scalar",
            SimdTier::Avx2 => "avx2",
            SimdTier::Avx512 => "avx512",
            SimdTier::Avx512Vnni => "avx512vnni",
        }
    }

    /// The tier named `s` (as [`SimdTier::name`] spells it, ASCII case ignored).
    pub fn parse(s: &str) -> Option<SimdTier> {
        SimdTier::ALL
            .into_iter()
            .find(|t| t.name().eq_ignore_ascii_case(s.trim()))
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

    /// The math slices' tier this one runs ([`math::Tier`]: they have no VNNI form).
    pub fn math(self) -> math::Tier {
        match self {
            SimdTier::Scalar => math::Tier::Baseline,
            SimdTier::Avx2 => math::Tier::Avx2,
            SimdTier::Avx512 | SimdTier::Avx512Vnni => math::Tier::Avx512,
        }
    }
}

/// The active tier, as `index + 1` into [`SimdTier::ALL`]; 0 until the first call resolves it.
static ACTIVE: AtomicU8 = AtomicU8::new(0);

fn encode(t: SimdTier) -> u8 {
    t as u8 + 1
}

fn decode(n: u8) -> SimdTier {
    SimdTier::ALL[n as usize - 1]
}

/// The tier code runs at: the one [`set_max`] last chose, else [`SimdTier::best`], installed on
/// first use. One relaxed load once resolved, the cost of the `is_x86_feature_detected!` it
/// replaces.
#[inline]
pub fn active() -> SimdTier {
    match ACTIVE.load(Ordering::Relaxed) {
        0 => install_first(SimdTier::best()),
        n => decode(n),
    }
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

/// Run at most `max` from now on (`None`: no ceiling, the widest tier the CPU has), and return
/// the tier that is now active. A ceiling above what the CPU supports is clamped to
/// [`SimdTier::best`]: it is a ceiling, not a demand. This is the `kernels.cpu.max_tier` debug
/// setting, and it is process-wide: the last call wins for every thread and every backend.
pub fn set_max(max: Option<SimdTier>) -> SimdTier {
    let best = SimdTier::best();
    let tier = max.map_or(best, |m| m.min(best));
    ACTIVE.store(encode(tier), Ordering::Relaxed);
    tier
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Names round-trip, and `ALL` is in the order the ceiling compares (the `Ord` derive and
    /// `encode` both rest on it).
    #[test]
    fn names_and_order() {
        for (i, t) in SimdTier::ALL.into_iter().enumerate() {
            assert_eq!(SimdTier::parse(t.name()), Some(t));
            assert_eq!(SimdTier::parse(&t.name().to_uppercase()), Some(t));
            assert_eq!(decode(encode(t)), t);
            assert_eq!(t as usize, i);
        }
        assert!(SimdTier::ALL.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(SimdTier::parse("baseline"), None);
        // Supported tiers are a prefix of `ALL`, the tiers being cumulative.
        let n = SimdTier::ALL.iter().filter(|t| t.supported()).count();
        assert!(SimdTier::ALL[..n].iter().all(|t| t.supported()));
        assert_eq!(SimdTier::best(), SimdTier::ALL[n - 1]);
    }

    /// The ceiling clamps to what the CPU has, `None` restores the widest, and a first use that
    /// lost the race to [`set_max`] keeps the tier `set_max` chose. Turns red if `set_max` stores
    /// a tier the CPU lacks or `install_first` stores unconditionally.
    #[test]
    fn ceiling_clamps_and_first_use_never_overwrites() {
        assert_eq!(set_max(Some(SimdTier::Avx512Vnni)), SimdTier::best());
        assert_eq!(active(), SimdTier::best());
        assert_eq!(set_max(Some(SimdTier::Scalar)), SimdTier::Scalar);
        assert_eq!(install_first(SimdTier::best()), SimdTier::Scalar);
        assert_eq!(active(), SimdTier::Scalar);
        assert_eq!(math::tier(), math::Tier::Baseline);
        assert_eq!(set_max(None), SimdTier::best());
        assert_eq!(active(), SimdTier::best());
        assert_eq!(math::tier(), SimdTier::best().math());
    }
}
