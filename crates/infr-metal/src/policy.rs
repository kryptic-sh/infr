//! Pure selection logic the executor calls, free of Metal types so it also builds and is tested
//! off macOS: `tests/policy.rs` includes this file by path, which is how a Linux run checks it.

/// The key entries per lane (`KPL`) the DeltaNet kernels are instantiated for, in the order of
/// the executor's per-variant kernel name lists. The 32 lanes of a simdgroup split a key head, so
/// a head runs only on the instantiation whose `KPL * 32` is exactly its width.
pub(crate) const DELTANET_KPL: [usize; 4] = [1, 2, 4, 8];

/// Index into [`DELTANET_KPL`] of the DeltaNet kernel instantiation for key-head width `kd`, or
/// `None` when no instantiation fits it and the op must take the host path. A wider instantiation
/// is never a fallback: its lanes would walk `lane * KPL + j` past `kd`, into the next head's
/// state.
pub(crate) fn deltanet_kpl_index(kd: usize) -> Option<usize> {
    DELTANET_KPL.iter().position(|&kpl| kpl * 32 == kd)
}

/// The capacity, in elements, to allocate for a pooled scratch buffer that holds `have` elements
/// (`None`: nothing pooled yet) when an op needs `need`, on a device whose largest buffer holds
/// `max` elements; `None` means the pooled buffer already fits and is reused. Growth at least
/// doubles, so a request that grows by a little per call (the KV dequant prepass grows by one row
/// per decode token) reallocates a logarithmic number of times over a generation rather than on
/// every token; a first request, or a jump past double, gets exactly what it asked for. Doubling
/// never takes a request that fits the device past `max`; one that does not fit is asked for as
/// is, so the allocation error names its size.
pub(crate) fn scratch_capacity(have: Option<usize>, need: usize, max: usize) -> Option<usize> {
    match have {
        Some(cap) if cap >= need => None,
        Some(cap) => {
            let grown = need.max(cap.saturating_mul(2));
            Some(if need <= max { grown.min(max) } else { grown })
        }
        None => Some(need),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deltanet_picks_only_an_exact_lane_width() {
        for (i, kpl) in DELTANET_KPL.iter().enumerate() {
            assert_eq!(deltanet_kpl_index(kpl * 32), Some(i));
        }
        // Multiples of 32 between the instantiations: each would have run the 8-per-lane kernel.
        for kd in [96, 160, 192, 224] {
            assert_eq!(deltanet_kpl_index(kd), None, "head_k {kd}");
        }
        for kd in [0, 16, 100, 288, 512] {
            assert_eq!(deltanet_kpl_index(kd), None, "head_k {kd}");
        }
    }

    /// A device limit no test request comes near.
    const ROOMY: usize = usize::MAX;

    #[test]
    fn scratch_reuses_a_buffer_that_fits() {
        assert_eq!(scratch_capacity(None, 10, ROOMY), Some(10));
        assert_eq!(scratch_capacity(Some(10), 10, ROOMY), None);
        assert_eq!(scratch_capacity(Some(10), 3, ROOMY), None);
    }

    #[test]
    fn scratch_growth_doubles_or_takes_the_jump() {
        assert_eq!(scratch_capacity(Some(10), 11, ROOMY), Some(20));
        assert_eq!(scratch_capacity(Some(10), 50, ROOMY), Some(50));
        assert_eq!(
            scratch_capacity(Some(usize::MAX - 1), usize::MAX, ROOMY),
            Some(usize::MAX)
        );
    }

    /// A tag holding 55% of the device maximum that is asked for 60%: doubling would ask for 110%
    /// and fail where the request itself fits, so the growth stops at the maximum.
    #[test]
    fn scratch_growth_stops_at_the_device_maximum() {
        let max = 1000;
        assert_eq!(scratch_capacity(Some(550), 600, max), Some(max));
        assert_eq!(scratch_capacity(Some(550), max, max), Some(max));
        // A request past the maximum is passed through, for the allocation to refuse by size.
        assert_eq!(scratch_capacity(Some(550), 1200, max), Some(1200));
        assert_eq!(scratch_capacity(None, 1200, max), Some(1200));
    }

    /// The B126 shape: one tag asked for a size that grows by one row per decode token. The pool
    /// must reallocate a logarithmic number of times, and end within double the largest request.
    #[test]
    fn a_per_token_growing_request_reallocates_logarithmically() {
        let row = 8 * 128;
        let mut cap = None;
        let mut allocs = 0;
        for kv_len in 1..=4096 {
            if let Some(c) = scratch_capacity(cap, kv_len * row, ROOMY) {
                cap = Some(c);
                allocs += 1;
            }
        }
        // One per doubling of kv_len from 1, plus the first allocation.
        assert_eq!(allocs, 4096usize.ilog2() as usize + 1);
        let cap = cap.unwrap();
        assert!(cap >= 4096 * row && cap < 2 * 4096 * row);
    }
}
