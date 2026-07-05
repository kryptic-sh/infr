//! MSL compute kernels and lazy pipeline-state cache.
//!
//! All kernels operate on `float` (f32) buffers — quantized weights are dequantized to f32 on the
//! host before they reach a kernel, so the shaders stay format-agnostic and simple. The full MSL
//! source is compiled once at backend init; individual `MTLComputePipelineState`s are created on
//! first use and cached by function name.

use crate::be;
use infr_core::error::Result;
use metal::{ComputePipelineState, Device, Library};
use std::collections::HashMap;
use std::sync::Mutex;

pub struct Pipelines {
    device: Device,
    library: Library,
    cache: Mutex<HashMap<&'static str, ComputePipelineState>>,
}

unsafe impl Send for Pipelines {}
unsafe impl Sync for Pipelines {}

/// The complete assembled MSL source — the ONE string the backend compiles. Public so the
/// kernel-name tripwire test resolves names against exactly what the runtime compiles (a
/// separately-maintained file list in the test would drift the same way a duplicated source
/// copy once did).
pub fn msl_source() -> String {
    MSL_PARTS.concat()
}

impl Pipelines {
    pub fn build(device: &Device) -> Result<Self> {
        let opts = metal::CompileOptions::new();
        // Reference backend: prefer accurate transcendentals (sin/cos/tanh) over fast intrinsics so
        // results stay in tight numeric parity with the CPU interpreter.
        opts.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(&msl_source(), &opts)
            .map_err(|e| be(format!("compile MSL library: {e}")))?;
        Ok(Self {
            device: device.clone(),
            library,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Get (creating + caching on first use) the compute pipeline for an MSL kernel function.
    pub fn get(&self, name: &'static str) -> Result<ComputePipelineState> {
        if let Some(p) = self.cache.lock().unwrap().get(name) {
            return Ok(p.clone());
        }
        let func = self
            .library
            .get_function(name, None)
            .map_err(|e| be(format!("get MSL function {name}: {e}")))?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| be(format!("pipeline for {name}: {e}")))?;
        self.cache.lock().unwrap().insert(name, pso.clone());
        Ok(pso)
    }
}

/// Metal Shading Language source for every kernel, split into domain files under `shaders/`
/// (see each file's header). Concatenated IN ORDER into one string so it compiles as a single
/// library — MSL requires define-before-use, so the file order here is load-bearing (helpers and
/// constant tables in earlier files are referenced by later ones). `include_str!` makes cargo
/// track the files for rebuilds automatically.
///
/// There is deliberately NO other copy of the shader source: an embedded-string duplicate once
/// drifted (the string was restored in a rebase while new kernels landed only in the files),
/// which silently disabled every kernel that existed only in the non-live copy — the pipeline
/// cap-checks treat a missing function as "capability absent" and fall back, so nothing errors.
const MSL_PARTS: [&str; 8] = [
    include_str!("../shaders/common.metal"),
    include_str!("../shaders/elementwise_norms.metal"),
    include_str!("../shaders/linear.metal"),
    include_str!("../shaders/moe.metal"),
    include_str!("../shaders/rope_ffn.metal"),
    include_str!("../shaders/attention.metal"),
    include_str!("../shaders/deltanet.metal"),
    include_str!("../shaders/kv_cache.metal"),
];
