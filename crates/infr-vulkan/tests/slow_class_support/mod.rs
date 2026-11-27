//! How a slow-device test puts its device in the slow class (`tuning::DeviceFacts::slow_class`),
//! for the tests of what that class does to a first forward: the submit splitter's seed
//! (`first_forward_split`) and the work bound's default budget (`work_split_parity`).

use infr_vulkan::tuning::DeviceFacts;

/// How the slow class is reached on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlowBy {
    /// Force the wave64 path (`INFR_FORCE_WAVE64`), which puts any device in the slow class.
    Wave64,
    /// Run the device as it is, which is slow already: it is in the slow class, or it is
    /// integrated. Every default the slow class sets is one an integrated device gets anyway
    /// (`reset_prone`, `infr_core::initial_submit_dispatch_cap`, the decode chunk and the
    /// prefill-only seed, which are for discrete devices only), so there the class is no
    /// different from the device's own.
    AsItIs,
}

/// How to reach the slow class on the device whose `facts` a backend opened WITHOUT the wave64
/// path reports: force wave64 where the device can pin 64 (`subgroup_max` is its native range
/// there), else run the device as it is where that is slow already. `None`, with a skip line,
/// on a fast discrete device that cannot pin 64 (one with an accelerated int8 dot or matrix
/// units, and a range up to 32): nothing a test can set puts it in the class, since the int8 dot
/// counts by the driver's report, which no mask changes.
pub fn slow_by(facts: &DeviceFacts) -> Option<SlowBy> {
    if facts.subgroup_max >= 64 {
        Some(SlowBy::Wave64)
    } else if facts.slow_class() || facts.integrated {
        Some(SlowBy::AsItIs)
    } else {
        eprintln!(
            "skip: {} cannot pin a subgroup size of 64 (range [{}, {}]), so wave64 cannot be \
             forced, and it is a discrete device outside the slow class (matrix units: {}, \
             accelerated int8 dot: {})",
            facts.name,
            facts.subgroup_min,
            facts.subgroup_max,
            facts.matrix_units,
            facts.dot_accelerated
        );
        None
    }
}
