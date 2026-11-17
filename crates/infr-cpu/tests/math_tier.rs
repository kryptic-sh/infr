//! `kernels.cpu.math_tier` is a process-wide switch that every CPU backend applies when it is
//! built: a backend with the setting unset puts the math slices back on the widest tier, so an
//! earlier backend's debug setting cannot silently slow every later backend in the process.
//!
//! Its own test binary, so no other test builds a backend while this one reads the active tier.
//! (Only speed is at stake: every tier gives the same bits.)

use std::sync::Arc;

use infr_core::config::Config;
use infr_core::math::{self, Tier};
use infr_cpu::CpuBackend;

fn backend(tier: Option<Tier>) -> CpuBackend {
    let mut cfg = Config::default();
    cfg.kernels.cpu.math_tier = tier;
    CpuBackend::new_with(Arc::new(cfg))
}

/// Turns red if a backend built without the setting leaves the process tier alone. On a CPU
/// whose widest tier IS the baseline (aarch64, a pre-AVX2 x86) the two tiers coincide and the
/// test cannot tell the difference; it says so rather than passing silently.
#[test]
fn every_backend_sets_the_tier_and_the_last_one_wins() {
    let _forced = backend(Some(Tier::Baseline));
    assert_eq!(math::tier(), Tier::Baseline);
    let _auto = backend(None);
    assert_eq!(math::tier(), Tier::best());
    if Tier::best() == Tier::Baseline {
        eprintln!("math_tier: this CPU's widest tier is the baseline; the reset is not observable");
    }
    let _forced_again = backend(Some(Tier::Baseline));
    assert_eq!(math::tier(), Tier::Baseline);
}
