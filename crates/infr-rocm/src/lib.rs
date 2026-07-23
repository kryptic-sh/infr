//! Reference ROCm/HIP compute backend — a correctness-first implementation of the
//! [`Backend`] seam on AMD GPUs through the ROCm/HIP stack.
//!
//! The crate compiles to an empty stub unless BOTH `cfg(target_os = "linux")` AND the
//! `rocm` cargo feature are active — without them the crate has no HIP deps and the
//! workspace builds everywhere. When the feature is on a `RocmBackend` is available.
//!
//! This is the fourth compute backend (alongside CPU, Vulkan, Metal). See
//! `docs/rocm-plan.md` for the full roadmap.

/// The real backend, defined only when the `rocm` feature is active on Linux.
#[cfg(all(target_os = "linux", feature = "rocm"))]
mod backend;

#[cfg(all(target_os = "linux", feature = "rocm"))]
pub use backend::{RocmBackend, RocmBuffer};
