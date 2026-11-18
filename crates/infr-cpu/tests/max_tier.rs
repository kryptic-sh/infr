//! `kernels.cpu.max_tier` is a process-wide ceiling that every CPU backend applies when it is
//! built: a backend with the setting unset lifts it, so an earlier backend's debug setting cannot
//! silently slow every later backend in the process, and a ceiling above the CPU's widest tier
//! runs the widest. (That the kernels obey the active tier is `infr-cpu`'s
//! `kernels::tier_tests`, which records the tier each dispatcher ran.)
//!
//! Its own test binary, so no other test builds a backend while this one reads the active tier.
//! (Only speed is at stake: every tier gives the same bits.)

use std::sync::Arc;

use infr_core::config::Config;
use infr_core::math;
use infr_core::simd::{self, SimdTier};
use infr_cpu::CpuBackend;

fn backend(tier: Option<SimdTier>) -> CpuBackend {
    let mut cfg = Config::default();
    cfg.kernels.cpu.max_tier = tier;
    CpuBackend::new_with(Arc::new(cfg))
}

/// Turns red if a backend built without the setting leaves the process tier alone, if a ceiling
/// is not applied, or if one above the CPU's widest is not clamped. On a CPU whose widest tier IS
/// the scalar one (aarch64, a pre-AVX2 x86) the tiers coincide and the reset cannot be observed;
/// the test says so rather than passing silently.
#[test]
fn every_backend_sets_the_ceiling_and_the_last_one_wins() {
    let best = SimdTier::best();
    for t in SimdTier::ALL {
        let _b = backend(Some(t));
        let want = t.min(best);
        assert_eq!(simd::active(), want, "ceiling {t:?}");
        assert_eq!(math::tier(), want.math(), "ceiling {t:?}");
        assert_eq!(
            infr_cpu::q4k_expert_repack_available(),
            want == SimdTier::Avx512Vnni,
            "ceiling {t:?}"
        );
    }
    let _forced = backend(Some(SimdTier::Scalar));
    assert_eq!(simd::active(), SimdTier::Scalar);
    let _auto = backend(None);
    assert_eq!(simd::active(), best);
    if best == SimdTier::Scalar {
        eprintln!(
            "max_tier: this CPU's widest tier is the scalar one; the reset is not observable"
        );
    }
    let _forced_again = backend(Some(SimdTier::Scalar));
    assert_eq!(simd::active(), SimdTier::Scalar);
    assert_eq!(math::tier(), math::Tier::Baseline);
}
