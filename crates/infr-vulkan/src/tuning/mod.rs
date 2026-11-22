//! Per-device tuning parameters for the Vulkan kernels: ONE place that says which hardware-shaped
//! number each kernel family runs with on this device, and why.
//!
//! - `families` declares every parameter: its family, kind, type and range, the qualifier axis
//!   it is keyed on, its generic default as a function of the device's [`DeviceFacts`], and the
//!   older config knobs that are now aliases of it.
//! - `table.toml` (compiled in) holds tuned VALUES as rows that match device classes or specific
//!   devices, each with its provenance. A user's tuning file uses the same schema
//!   (`kernels.vulkan.tuning.file`).
//! - `resolve` layers them, per parameter: generic default, class rows, device rows, the user's
//!   tuning file, config, environment, command line — the last level that sets a value wins.
//! - [`report`] prints the result (`infr tune show` / `infr tune list`) and its short hash, which
//!   the device banner carries.
//!
//! The set is resolved ONCE, when the backend is created, into the typed `Tuning` struct the
//! kernels read by field; nothing on a dispatch path looks a parameter up by name.

pub(crate) mod families;
pub mod report;
pub(crate) mod resolve;
pub(crate) mod table;

#[cfg(test)]
mod classes;
#[cfg(test)]
pub(crate) mod tests;

pub use report::{ParamReport, RowReport, TuningReport};

use crate::caps::DeviceArch;
use ash::vk;

/// What the tuning table may know about a device: everything its rows can match on and every
/// fact a generic default is derived from. Gathered once, at backend creation, from what the
/// backend already probed — no query of its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceFacts {
    /// `VkPhysicalDeviceProperties::deviceName`.
    pub name: String,
    /// PCI vendor id.
    pub vendor_id: u32,
    /// PCI device id.
    pub device_id: u32,
    /// The driver id's Vulkan name (`AMD_PROPRIETARY`, `MESA_RADV`, …), `None` when the driver
    /// reports none (pre-1.2).
    pub driver: Option<String>,
    /// `VkPhysicalDeviceProperties::driverVersion`, raw (its encoding is the vendor's).
    pub driver_version: u32,
    /// The architecture bucket's name (`AmdRdna3`, `IntelXe2`, …, `Other` when unidentified).
    pub arch: &'static str,
    /// An integrated GPU.
    pub integrated: bool,
    /// Shader cores (AMD CUs, NVIDIA SMs, Intel Xe-cores); 0 = unknown.
    pub compute_units: u32,
    /// The pinnable subgroup range as the kernels see it (`[64, 64]` on the wave64 path).
    pub subgroup_min: u32,
    pub subgroup_max: u32,
    /// The device runs the wave64 emulation (`shaders/wave64.glsl`).
    pub wave64: bool,
    /// `maxComputeSharedMemorySize`, bytes.
    pub max_shared_memory: u32,
    /// The 16x16x16 f16 cooperative-matrix kernels are live (`Capabilities::f16_coopmat`, after
    /// any forced mask such as `INFR_NO_COOPMAT` and the wave64 path's coopmat-off rule).
    pub matrix_units: bool,
    /// The driver reports packed signed 4x8-bit dot products as accelerated
    /// (`integerDotProduct4x8BitPackedSignedAccelerated`): `false` where the int8 dot the
    /// kernels use is emulated in ALU, as on GCN under RADV.
    pub dot_accelerated: bool,
}

impl DeviceFacts {
    /// A device whose work per dispatch is slow enough that a whole first forward in one submit
    /// can outlast a driver's job timeout (issue 94's RX 580): a wave64-only part, or one with
    /// neither matrix units nor an accelerated int8 dot.
    pub fn slow_class(&self) -> bool {
        self.wave64 || (!self.matrix_units && !self.dot_accelerated)
    }

    /// A device a driver's watchdog is known to reset under long work: an integrated GPU, or one
    /// in the slow class. It gets the tight default of `submit.budget_ms` (`crate::work`).
    pub fn reset_prone(&self) -> bool {
        self.integrated || self.slow_class()
    }

    /// The vendor's short name, as a row's `match.vendor` spells it.
    pub fn vendor(&self) -> &'static str {
        vendor_name(self.vendor_id)
    }
}

/// `amd` / `intel` / `nvidia`, or `other` for any vendor id the table has no name for.
pub(crate) fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        crate::caps::VENDOR_AMD => "amd",
        crate::caps::VENDOR_INTEL => "intel",
        crate::caps::VENDOR_NVIDIA => "nvidia",
        _ => "other",
    }
}

/// A driver id's Vulkan enumerant name (`AMD_PROPRIETARY`), as rows and fixtures spell it.
pub(crate) fn driver_name(id: vk::DriverId) -> String {
    format!("{id:?}")
}

/// The architecture bucket's name, as rows and fixtures spell it.
pub(crate) fn arch_name(a: DeviceArch) -> &'static str {
    match a {
        DeviceArch::Other => "Other",
        DeviceArch::AmdGcn => "AmdGcn",
        DeviceArch::AmdRdna1 => "AmdRdna1",
        DeviceArch::AmdRdna2 => "AmdRdna2",
        DeviceArch::AmdRdna3 => "AmdRdna3",
        DeviceArch::IntelXe1 => "IntelXe1",
        DeviceArch::IntelXe2 => "IntelXe2",
        DeviceArch::NvidiaPreTuring => "NvidiaPreTuring",
        DeviceArch::NvidiaTuring => "NvidiaTuring",
    }
}

/// [`arch_name`]'s inverse.
pub(crate) fn arch_from_name(s: &str) -> Option<DeviceArch> {
    [
        DeviceArch::Other,
        DeviceArch::AmdGcn,
        DeviceArch::AmdRdna1,
        DeviceArch::AmdRdna2,
        DeviceArch::AmdRdna3,
        DeviceArch::IntelXe1,
        DeviceArch::IntelXe2,
        DeviceArch::NvidiaPreTuring,
        DeviceArch::NvidiaTuring,
    ]
    .into_iter()
    .find(|&a| arch_name(a) == s)
}

/// The quant formats the `mmv_mw.warps` parameter is qualified by, in its storage order, with the
/// qualifier spelling a key uses (`mmv_mw.warps@q4_k`): every format with an int8 decode arm.
pub(crate) const MMV_MW_DTYPES: &[(&str, infr_core::DType)] = {
    use infr_core::DType::*;
    &[
        ("q4_k", Q4K),
        ("q6_k", Q6K),
        ("q2_k", Q2K),
        ("q3_k", Q3K),
        ("q5_k", Q5K),
        ("q8_0", Q8_0),
        ("q4_0", Q4_0),
        ("q5_0", Q5_0),
        ("q4_1", Q4_1),
        ("q5_1", Q5_1),
        ("iq4_nl", Iq4Nl),
    ]
};

/// The most split-K partials the coopmat attention prefill can take: the adapter sizes the flash
/// (`flash_po`/`pm`/`pl`) and PV (`nonfa_pv`) partials scratch for this many, and the
/// `attn_flash.*` / `attn_pv.*` split counts are bounded by it.
pub(crate) const ATTN_PREFILL_MAX_SPLITS: u32 = 8;

/// The split-count policy one attention prefill family runs ([`Tuning::flash`] / [`Tuning::pv`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AttnSplit {
    /// A fixed split count, or 0 for the policy below.
    pub splits: u32,
    /// No split once the plain grid has at least this many workgroups.
    pub full_wg: u32,
    /// Aim the split grid at about this many workgroups.
    pub target_wg: u32,
    /// The most splits the policy picks (the scratch is sized for this many).
    pub max_splits: u32,
    /// No split below this many keys.
    pub min_kv: u32,
    /// At least this many keys per split.
    pub kv_per_split: u32,
}

impl AttnSplit {
    /// The split count for a plain grid of `base_wg` workgroups over `kv` keys: the fixed count
    /// when one is set, else the policy (one split for a full grid or a short span, otherwise
    /// enough splits to reach about `target_wg` workgroups, between 2 and `max_splits`, with at
    /// least `kv_per_split` keys each).
    pub fn count(&self, base_wg: u32, kv: u32) -> u32 {
        let n = self.policy(base_wg, kv);
        // The partials scratch holds this many; a larger count writes past its end. The ranges of
        // `splits` and `max_splits` already keep every resolved set inside it.
        assert!(
            (1..=ATTN_PREFILL_MAX_SPLITS).contains(&n),
            "attention split count {n} outside 1..={ATTN_PREFILL_MAX_SPLITS}"
        );
        n
    }

    fn policy(&self, base_wg: u32, kv: u32) -> u32 {
        if self.splits > 0 {
            return self.splits;
        }
        if base_wg >= self.full_wg || kv < self.min_kv {
            1
        } else {
            (self.target_wg / base_wg.max(1))
                .clamp(2, self.max_splits)
                .min((kv / self.kv_per_split).max(1))
        }
    }
}

/// The resolved parameter set, typed, one field per parameter. Built once per backend by
/// [`resolve::resolve`]; every kernel choice reads its field here.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Tuning {
    /// `device.sg_pref`: the subgroup width the decode GEMV/reduction family pins.
    pub sg_pref: u32,
    /// `gemv.variant`: the multi-row decode GEMV build (`None` = the plain RM path).
    pub gemv_variant: Option<String>,
    /// `gemv_rm.*`: the multi-output-row decode GEMV.
    pub gemv_rm_enabled: bool,
    pub gemv_rm_rows: u32,
    pub gemv_rm_min_out: usize,
    pub gemv_rm_max_out: usize,
    /// `gemv_sg.*` / `gemv_id_sg.enabled`: the subgroup decode GEMVs.
    pub gemv_sg_enabled: bool,
    pub gemv_id_sg_enabled: bool,
    pub gemv_sg_rows: u32,
    pub gemv_sg_min_out: usize,
    pub gemv_sg_max_out: usize,
    /// `mmv_mw.warps@<format>`, in [`MMV_MW_DTYPES`] order.
    pub mmv_mw_warps: [u32; MMV_MW_DTYPES.len()],
    /// `mmv_mrow.*`: when the int8 multi-row GEMV takes its four-outputs-per-workgroup build.
    pub mmv_o4_max_in_f: usize,
    pub mmv_o4_all_widths: bool,
    /// `attn_flash.*`: the coopmat flash-attention prefill.
    pub flash: AttnSplit,
    pub flash_min_rows: usize,
    pub flash_force_bm32: bool,
    /// `attn_pv.*`: the non-flash coopmat attention's P·V split.
    pub pv: AttnSplit,
    /// `attn_split.decode_min_chunk`: the chunk floor of decode split-K attention
    /// (`crate::attn_chunk`).
    pub attn_decode_min_chunk: usize,
    /// `attn_canvas.chunks`: the canvas-mask split-K chunk divisor.
    pub canvas_chunks: usize,
    /// `moe.small_m`: the small-batch MoE path's row ceiling.
    pub moe_small_m: usize,
    /// `split_k.*`: the coopmat prefill GEMM's narrow-grid split-K window.
    pub split_k_fixed_grid: usize,
    pub split_k_fill_per_cu: usize,
    pub split_k_scale_by_cu: bool,
    /// `gemm_i8cm_kq.*`: the int8 cooperative-matrix K-quant prefill GEMM.
    pub i8cm_kq_enabled: bool,
    pub i8cm_kq_l_fill_per_cu: usize,
    /// `rmsnorm.*`: the 1024-wide decode RMSNorm build.
    pub rmsnorm_wide: bool,
    pub rmsnorm_wide_min_dim: usize,
    /// `queue.compute_only`: submit on a COMPUTE-only queue family when the device has a usable
    /// one (`VulkanBackend::pick_queue_family`).
    pub queue_compute_only: bool,
    /// `submit.initial_dispatches`: the submit splitter's cap before any forward has been
    /// measured (`0` = never split).
    pub submit_initial_dispatches: usize,
    /// `submit.budget_ms`: the estimated GPU time past which a dispatch is cut into ranges and
    /// a submit is closed at the next op boundary (`0` = no work bound; see `crate::work`).
    pub submit_budget_ms: u32,
    /// `submit.prior_gflops`: the throughput the work estimate assumes for a kernel this process
    /// has not measured yet, in GFLOP/s for the whole device.
    pub submit_prior_gflops: u32,
}

impl Tuning {
    /// `mmv_mw.warps` for `dt`, or `None` for a format the parameter has no qualifier for.
    pub fn mmv_mw_warps(&self, dt: infr_core::DType) -> Option<u32> {
        MMV_MW_DTYPES
            .iter()
            .position(|&(_, d)| d == dt)
            .map(|i| self.mmv_mw_warps[i])
    }
}
