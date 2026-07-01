// Apple-only: the `metal`/`objc` crates link the Objective-C runtime and don't build off macOS.
// The whole crate compiles to nothing elsewhere (its consumers gate their use to macOS too), so the
// Linux-only CI still builds the workspace.
#![cfg(target_os = "macos")]
//! Reference Metal compute backend — a correctness-first implementation of the [`Backend`] seam
//! (the same one `infr-cpu` and `infr-vulkan` implement) on Apple's Metal API.
//!
//! Priorities are *correctness and clarity*, not speed: the backend keeps each graph tensor's
//! working values in host f32 vectors (exactly like the CPU interpreter), and delegates each op's
//! arithmetic to a small Metal compute kernel operating on f32 `MTLBuffer`s. Quantized `Op::Linear`
//! weights are dequantized to f32 on the host (reusing `infr_gguf::dequant`) and cached as device
//! buffers, so the MSL kernels never need to understand a single GGUF quant format.
//!
//! This follows `infr-cpu`'s execution model closely, which is what makes it easy to keep in
//! numeric parity (see `tests/parity.rs`). It is not bit-for-bit identical everywhere: quantized
//! `Op::Linear` runs a full-f32 dequant dot here, whereas the CPU path quantizes the *activation*
//! to Q8 and uses integer dots — so this backend is actually the slightly more accurate of the two.
//! A resident-on-device, kernel-fused fast path is future work.

use infr_core::backend::{Backend, Bindings, BufferUsage, Capabilities, GraphPlan, Plan};
use infr_core::error::{Error, Result};
use infr_core::graph::Graph;
use metal::{Buffer as MtlBuffer, CommandQueue, Device};

mod exec;
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
        })
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
