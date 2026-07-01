// Apple-only: the `metal`/`objc` crates link the Objective-C runtime and don't build off macOS.
// The whole crate compiles to nothing elsewhere (its consumers gate their use to macOS too), so the
// Linux-only CI still builds the workspace.
#![cfg(target_os = "macos")]
//! Reference Metal compute backend — a correctness-first implementation of the [`Backend`] seam
//! (the same one `infr-cpu` and `infr-vulkan` implement) on Apple's Metal API.
//!
//! Priorities are *correctness and clarity* first, then not being needlessly slow. Each op's
//! arithmetic runs in a small Metal compute kernel over f32 `MTLBuffer`s, following `infr-cpu`'s
//! dataflow closely so it stays in numeric parity (see `tests/parity.rs`).
//!
//! To avoid a CPU↔GPU round-trip per op, graph tensors stay *resident on the device*: a per-forward
//! executor (`exec::Resident`) tracks whether each tensor's current value is on the host or the
//! device, encodes consecutive GPU ops into a single command buffer (Metal hazard-tracks the
//! barriers), and only syncs to the host at a host-side op or the final write-back. Quantized
//! `Op::Linear` weights are kept in a compact unified form (u8 codes + one `(scale, min)` per 16
//! elems) and decoded inline by `linear_qui`, so the kernels stay format-agnostic and never blow a
//! quant weight up to f32. `INFR_METAL_PROFILE=1` prints a per-op / GPU-wall breakdown on drop.
//!
//! Not bit-for-bit identical to the CPU everywhere: quantized `Op::Linear` reconstructs the exact
//! `dequant` value but dots in f32, whereas the CPU path quantizes the *activation* to Q8 and uses
//! integer dots — so this backend is actually the slightly more accurate of the two. Faster matvec
//! kernels (GEMV occupancy / fusion) are future work.

use infr_core::backend::{Backend, Bindings, BufferUsage, Capabilities, GraphPlan, Plan};
use infr_core::error::{Error, Result};
use infr_core::graph::Graph;
use metal::{Buffer as MtlBuffer, CommandQueue, Device};

mod exec;
mod profile;
mod shaders;

fn be(msg: impl std::fmt::Display) -> Error {
    Error::backend(msg)
}

/// A device buffer. On Apple Silicon `StorageModeShared` memory is CPU-visible, so `upload`/
/// `download` are plain `memcpy`s against [`MtlBuffer::contents`].
pub struct MetalBuffer {
    raw: MtlBuffer,
    len: usize,
}

// MTLBuffer is documented thread-safe for the create/read/write use here; the raw pointer in the
// metal-rs wrapper makes it neither Send nor Sync by default.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl infr_core::backend::Buffer for MetalBuffer {
    fn len_bytes(&self) -> usize {
        self.len
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct MetalBackend {
    device: Device,
    queue: CommandQueue,
    pipelines: shaders::Pipelines,
    /// Dequantized-weight cache: bound-buffer address → device f32 buffer. Weights are bound the
    /// same every step, so a quantized weight is dequantized once and reused.
    ///
    /// Keyed by address only, and never invalidated — safe because this has a *single-generation
    /// lifetime*: each `generate_*` builds a fresh `MetalBackend`, so the cache lives for exactly one
    /// generation and distinct weights have distinct addresses. Reusing an instance across models
    /// (with buffer free/realloc) could return a stale f32 for a recycled address; don't.
    weight_cache: std::sync::Mutex<std::collections::HashMap<usize, std::sync::Arc<MtlBuffer>>>,
    /// Native-quant weight cache (same single-generation lifetime as `weight_cache`): a quantized
    /// weight kept in its compact unified form — u8 codes + one (scale, min) per 16 elems — that the
    /// `linear_qui` kernel decodes inline. ~12 bpw vs f32's 32, and reconstructs the exact same value.
    qui_cache: std::sync::Mutex<std::collections::HashMap<usize, std::sync::Arc<exec::QuiWeight>>>,
    /// Opt-in execution profiler; active only when `INFR_METAL_PROFILE` is set. `profiling` is
    /// cached so the hot path avoids an env lookup and skips the `Instant` calls when off.
    pub(crate) profiling: bool,
    pub(crate) prof: std::sync::Mutex<profile::Profile>,
}

// MTLDevice / MTLCommandQueue are documented thread-safe; the pipeline states are immutable after
// creation. The metal-rs wrappers are !Send/!Sync only because they hold raw obj-c pointers.
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| be("no Metal device found"))?;
        let queue = device.new_command_queue();
        let pipelines = shaders::Pipelines::build(&device)?;
        Ok(Self {
            device,
            queue,
            pipelines,
            weight_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            qui_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            profiling: std::env::var("INFR_METAL_PROFILE").is_ok(),
            prof: std::sync::Mutex::new(profile::Profile::default()),
        })
    }
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        if self.profiling {
            self.prof.lock().unwrap().print_summary();
        }
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            name: self.device.name().to_string(),
            f16: true,
            cooperative_matrix: false,
            max_buffer_bytes: self.device.max_buffer_length(),
            // Metal's per-threadgroup memory limit (MTLDevice.maxThreadgroupMemoryLength) — the
            // analogue of Vulkan's maxComputeSharedMemorySize (typically 32 KB, 64 KB on Apple GPUs).
            max_shared_memory_bytes: self.device.max_threadgroup_memory_length() as u32,
            unified_memory: self.device.has_unified_memory(),
            // Like the CPU interpreter, this backend reads the baked `pos`/`kv_len` from the graph
            // ops each execute, so the decode graph is rebuilt per token (no record-once replay).
            decode_replay: false,
        }
    }

    fn alloc(
        &self,
        bytes: usize,
        _usage: BufferUsage,
    ) -> Result<Box<dyn infr_core::backend::Buffer>> {
        let len = bytes.max(4);
        let raw = self
            .device
            .new_buffer(len as u64, metal::MTLResourceOptions::StorageModeShared);
        Ok(Box::new(MetalBuffer { raw, len }))
    }

    fn upload(&self, dst: &dyn infr_core::backend::Buffer, src: &[u8]) -> Result<()> {
        let b = metal_buf(dst);
        if src.len() > b.len {
            return Err(be(format!("upload: src {} > buffer {}", src.len(), b.len)));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), b.raw.contents() as *mut u8, src.len());
        }
        Ok(())
    }

    fn download(&self, src: &dyn infr_core::backend::Buffer, dst: &mut [u8]) -> Result<()> {
        let b = metal_buf(src);
        if dst.len() > b.len {
            return Err(be(format!(
                "download: dst {} > buffer {}",
                dst.len(),
                b.len
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                b.raw.contents() as *const u8,
                dst.as_mut_ptr(),
                dst.len(),
            );
        }
        Ok(())
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        Ok(GraphPlan::boxed(graph))
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        self.execute_graph(plan, bindings)
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}

fn metal_buf(b: &dyn infr_core::backend::Buffer) -> &MetalBuffer {
    b.as_any()
        .downcast_ref::<MetalBuffer>()
        .expect("metal backend: buffer is not a MetalBuffer (mixed backends?)")
}
