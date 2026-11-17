//! The tier checks every SIMD dispatch in this crate makes: "may this call run tier X's code?".
//!
//! Each asks [`infr_core::simd::active`], the CPU's widest tier under the `kernels.cpu.max_tier`
//! ceiling, never `is_x86_feature_detected!` directly, so the ceiling reaches every kernel. The
//! tiers are cumulative ([`SimdTier::supported`] requires every narrower tier's features), so a
//! `true` from [`avx512`] also licenses the AVX2 code under it, and [`vnni`] licenses AVX-512 F,
//! DQ, VL and BW besides VNNI. That is what makes calling a `#[target_feature]` function behind
//! one of these checks sound.
//!
//! **The rule for new kernels**, on every architecture: a tier's kernel computes the same value
//! as the scalar kernel of its family, bit for bit, for every input. Integer accumulation may be
//! reordered freely (it is exact, as long as no intermediate saturates or wraps where the scalar
//! one does not); a floating-point expression may not: it is written once, and every tier
//! evaluates the same operations in the same order, using `mul_add` exactly where the scalar
//! kernel does. A future NEON (or SVE, or RISC-V V) kernel is one more tier behind one more check
//! here and is held to the same rule by `kernels/tier_tests.rs`, which runs every family at every
//! tier the host has against the scalar one.
//!
//! In test builds a thread can run at a tier of its own ([`with_tier`]), without touching the
//! process-wide ceiling other tests run under, and each check records the widest tier it allowed
//! on that thread, so a test can tell which tier's code a dispatcher actually ran.

#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) use infr_core::simd::SimdTier;

/// Whether the active tier is at least `t`. (Off x86-64 every tier above scalar is absent, so
/// nothing asks.)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn at_least(t: SimdTier) -> bool {
    #[cfg(not(test))]
    {
        infr_core::simd::allows(t)
    }
    #[cfg(test)]
    {
        let ok = testing::current() >= t;
        if ok {
            testing::record(t);
        }
        ok
    }
}

/// May this call run AVX2 code?
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn avx2() -> bool {
    at_least(SimdTier::Avx2)
}

/// May this call run AVX-512 (F, DQ, VL, BW) code?
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn avx512() -> bool {
    at_least(SimdTier::Avx512)
}

/// May this call run AVX-512 VNNI code (with AVX-512 F, DQ, VL, BW)?
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn vnni() -> bool {
    at_least(SimdTier::Avx512Vnni)
}

#[cfg(test)]
pub(crate) use testing::with_tier;

#[cfg(test)]
mod testing {
    use super::SimdTier;
    use std::cell::Cell;

    thread_local! {
        /// This thread's own tier, overriding the process-wide one.
        static OVERRIDE: Cell<Option<SimdTier>> = const { Cell::new(None) };
        /// The widest tier a check allowed on this thread since [`with_tier`] began.
        static WIDEST: Cell<Option<SimdTier>> = const { Cell::new(None) };
    }

    #[cfg(target_arch = "x86_64")]
    pub(super) fn current() -> SimdTier {
        OVERRIDE.get().unwrap_or_else(infr_core::simd::active)
    }

    #[cfg(target_arch = "x86_64")]
    pub(super) fn record(t: SimdTier) {
        WIDEST.set(WIDEST.get().max(Some(t)));
    }

    /// Run `f` on this thread at `tier` (which the CPU must support), and return its result with
    /// the widest tier any check allowed meanwhile: `None` when every dispatcher took its scalar
    /// path. Other threads, the pool's included, keep the process-wide tier.
    pub(crate) fn with_tier<R>(tier: SimdTier, f: impl FnOnce() -> R) -> (R, Option<SimdTier>) {
        assert!(tier.supported(), "{tier:?}: this CPU cannot run it");
        let outer = (OVERRIDE.replace(Some(tier)), WIDEST.replace(None));
        let r = f();
        let widest = WIDEST.get();
        OVERRIDE.set(outer.0);
        WIDEST.set(outer.1);
        (r, widest)
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    /// At each tier the host has, the checks allow exactly that tier and the ones below it, and
    /// report it as the widest they allowed. Turns red if a check ignores the tier it runs under
    /// or reports another. (A ceiling above the host's widest is clamped by
    /// `infr_core::simd::set_max`; `tests/max_tier.rs` checks that.)
    #[test]
    fn checks_obey_the_tier_they_run_under() {
        for t in SimdTier::ALL.into_iter().filter(|t| t.supported()) {
            let (allowed, took) = with_tier(t, || [avx2(), avx512(), vnni()]);
            let want = [
                t >= SimdTier::Avx2,
                t >= SimdTier::Avx512,
                t >= SimdTier::Avx512Vnni,
            ];
            assert_eq!(allowed, want, "{t:?}");
            assert_eq!(took, (t > SimdTier::Scalar).then_some(t), "{t:?}");
        }
    }
}
