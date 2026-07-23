//! The ROCm/HIP backend — mirrors `infr-metal`'s structure: backend struct, buffer,
//! and the `Backend` trait impl. For Phase 0 (scaffolding) this is a skeleton that
//! constructs, allocs, uploads/downloads, and syncs. Real kernels come in Phase 1.
//!
//! Compiled only when `cfg(all(target_os = "linux", feature = "rocm"))`.

use infr_core::backend::{Backend, Bindings, BufferUsage, Capabilities, GraphPlan, Plan};
use infr_core::error::{Error, Result};
use infr_core::graph::Graph;

fn be(msg: impl std::fmt::Display) -> Error {
    Error::backend(msg)
}

/// A device buffer allocated with `hipMalloc`.
pub struct RocmBuffer {
    // TODO: fill in with a HIP device pointer + byte length once the FFI is wired.
    _len: usize,
}

impl infr_core::backend::Buffer for RocmBuffer {
    fn len_bytes(&self) -> usize {
        self._len
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// HIP device pointers are Send/Sync (they identify a VRAM region, not a CPU address).
unsafe impl Send for RocmBuffer {}
unsafe impl Sync for RocmBuffer {}

/// The ROCm/HIP compute backend.
pub struct RocmBackend {
    // TODO: device handle, stream, module (once HIP FFI is wired).
}

// The backend owns streams and device handles which are Send/Sync.
unsafe impl Send for RocmBackend {}
unsafe impl Sync for RocmBackend {}

impl RocmBackend {
    /// Create a new ROCm backend, enumerating the first available HIP device.
    pub fn new() -> Result<Self> {
        // TODO: HIP device enumeration, stream creation (Phase 0 HIP FFI eval).
        Err(be(
            "ROCm backend not yet implemented — HIP FFI is pending (Phase 0)",
        ))
    }
}

impl Backend for RocmBackend {
    fn name(&self) -> &str {
        "rocm"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            name: "AMD ROCm/HIP".into(),
            f16: true,
            coopmat_f16: None,
            f8: false,
            coopmat_f8: None,
            i8: false,
            i8_dot: false,
            coopmat_i8: None,
            bf16: false,
            coopmat_bf16: None,
            subgroup_min: 0,
            subgroup_max: 0,
            sg_pref: 0,
            vendor_intel: false,
            integrated: false,
            compute_units: 0,
            buffer_device_address: false,
            max_shared_memory_bytes: 65536,
            unified_memory: false,
            // ── correctness-dial: start with NOTHING fused ──
            decode_replay: false,
            combined_gu: false,
            embed_gather: false,
            gpu_sample: false,
            argmax_rows: false,
            argmax_prob: false,
            gated_rmsnorm: false,
            kv_swa_ring: false,
        }
    }

    fn alloc(
        &self,
        bytes: usize,
        _usage: BufferUsage,
    ) -> Result<Box<dyn infr_core::backend::Buffer>> {
        // TODO: hipMalloc + hipMemset (calloc contract).
        let _ = bytes;
        Err(be("ROCm alloc not yet implemented"))
    }

    fn upload(&self, _dst: &dyn infr_core::backend::Buffer, _src: &[u8]) -> Result<()> {
        // TODO: hipMemcpy H2D.
        Err(be("ROCm upload not yet implemented"))
    }

    fn download(&self, _src: &dyn infr_core::backend::Buffer, _dst: &mut [u8]) -> Result<()> {
        // TODO: hipMemcpy D2H.
        Err(be("ROCm download not yet implemented"))
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        Ok(GraphPlan::boxed(graph))
    }

    fn execute(&self, _plan: &dyn Plan, _bindings: &Bindings) -> Result<()> {
        // TODO: walk graph, dispatch kernels, stream sync (Phase 1+).
        Err(be("ROCm execute not yet implemented"))
    }

    fn sync(&self) -> Result<()> {
        // TODO: hipStreamSynchronize / hipDeviceSynchronize.
        Err(be("ROCm sync not yet implemented"))
    }
}
