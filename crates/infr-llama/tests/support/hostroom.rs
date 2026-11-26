//! Whether a model test may load its model on the device it picked without starving the host: the
//! residency rule the tolerance gate (`forced::device_for`) and `cpu_backend`'s resident-room tests
//! check before loading.
//!
//! Shared by `cpu_backend.rs` and `support/forced.rs` (so `gpu_tolerance.rs`) through `#[path]`.

/// Host bytes a resident load takes on a unified-memory device per byte of GGUF: the device copy,
/// which is system RAM there, plus the mapped file pages the upload read through, which stay in the
/// process's working set while the model is up. Measured on Windows on 2026-11-15 with the
/// integrated GPU, each test run alone: Qwen3-30B-A3B Q4_K_M's 18.56 GB file took available memory
/// from 38058 MiB to 3596 MiB (1.95 bytes per file byte), and ERNIE-4.5-21B-A3B Q4_K_M's 13.33 GB
/// file from 35444 MiB to 11135 MiB (1.91), rounded up here.
pub const UNIFIED_RESIDENT_FACTOR: u64 = 2;

/// The cap on the host memory a model resident on a unified-memory device must still leave free.
/// The gate runs on a box other sessions share, and a gate run on the integrated GPU that took the
/// host down to its last few GB was killed for low memory and took other sessions' runs with it, so
/// the margin is the headroom those sessions are owed, not what this run needs to finish. Also the
/// margin taken when the host's total memory is unknown, where the conservative answer is the cap.
pub const UNIFIED_HOST_MARGIN: u64 = 12 << 30;

/// The share of the host's total memory the margin is held to when that is below
/// [`UNIFIED_HOST_MARGIN`]. Reciprocal — `total / UNIFIED_HOST_FRACTION`.
///
/// The margin is headroom owed to the rest of a shared machine, and what the rest of a machine runs
/// scales with the machine: on a 16 GB host, the resident load plus a fixed [`UNIFIED_HOST_MARGIN`]
/// is more than the host has free once its desktop is running, so every model test on its
/// integrated GPU would skip, however small the model.
pub const UNIFIED_HOST_FRACTION: u64 = 4;

/// The host memory a unified-memory load must leave free on a host with `total` bytes of physical
/// memory (`None` where the platform cannot say): the smaller of [`UNIFIED_HOST_MARGIN`] and
/// `total / UNIFIED_HOST_FRACTION`.
pub fn unified_host_margin(total: Option<u64>) -> u64 {
    total.map_or(UNIFIED_HOST_MARGIN, |t| {
        (t / UNIFIED_HOST_FRACTION).min(UNIFIED_HOST_MARGIN)
    })
}

/// Whether a test may load a `model_bytes` model, given whether the device's memory is the host's
/// (`unified`), the host memory `available` now and the host's `total` memory. `Err` carries why
/// not, naming the memory needed, the margin in it and what is available.
///
/// A discrete device always runs: its VRAM is its own pool and the host only maps the file. A
/// unified device runs when the load, priced at [`UNIFIED_RESIDENT_FACTOR`] per file byte, leaves
/// [`unified_host_margin`] free, and skips when it would not — or when the platform cannot say how
/// much is free, since that is no evidence it fits.
pub fn resident_verdict(
    unified: bool,
    available: Option<u64>,
    total: Option<u64>,
    model_bytes: u64,
) -> Result<(), String> {
    if !unified {
        return Ok(());
    }
    let gb = |b: u64| b as f64 / 1e9;
    let margin = unified_host_margin(total);
    let need = model_bytes
        .saturating_mul(UNIFIED_RESIDENT_FACTOR)
        .saturating_add(margin);
    match available {
        None => Err(format!(
            "({:.1} GB) on a unified-memory device: no figure for free host memory (no probe on \
             this platform, or it failed), so nothing shows it fits",
            gb(model_bytes)
        )),
        Some(avail) if avail < need => Err(format!(
            "({:.1} GB) on a unified-memory device needs {:.1} GB of host memory \
             ({UNIFIED_RESIDENT_FACTOR} x the model + a {:.1} GB margin); {:.1} GB available",
            gb(model_bytes),
            gb(need),
            gb(margin),
            gb(avail)
        )),
        Some(_) => Ok(()),
    }
}

/// [`resident_verdict`] on this host's memory, probed now.
pub fn host_room(unified: bool, model_bytes: u64) -> Result<(), String> {
    resident_verdict(
        unified,
        infr_core::hostmem::available_bytes(),
        infr_plat::mem::total(),
        model_bytes,
    )
}
