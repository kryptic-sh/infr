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
    // The IQ codebook grids (IQ2/IQ3) are generated from `infr_core::iquant_grids` — the SAME
    // tables the CPU dequant reads, so the native kernels stay bit-exact by construction rather
    // than by hand-transcribing 256..1024-entry tables into MSL. They must land before
    // `linear.metal` (whose DEC16_IQ2XXS etc reference them): common + norms, then grids, then
    // the rest in order.
    let mut s = String::with_capacity(256 * 1024);
    s.push_str(MSL_PARTS[0]);
    s.push_str(MSL_PARTS[1]);
    s.push_str(&iquant_grids_msl());
    for part in &MSL_PARTS[2..] {
        s.push_str(part);
    }
    s
}

/// Emit an MSL `constant <ty> NAME[N] = { ... };` from a Rust static, `sfx` the integer-literal
/// suffix that pins the element type (`ul` for ulong/u64, `u` for uint/u32, empty for uchar/u8).
fn emit_grid<T: std::fmt::Display>(s: &mut String, ty: &str, name: &str, arr: &[T], sfx: &str) {
    use std::fmt::Write;
    write!(s, "constant {ty} {name}[{}] = {{", arr.len()).unwrap();
    for (i, v) in arr.iter().enumerate() {
        if i % 8 == 0 {
            s.push('\n');
        }
        write!(s, "{v}{sfx},").unwrap();
    }
    s.push_str("\n};\n");
}

/// Emit the IQ codebook grid + sign tables as MSL `constant` arrays, formatted from the Rust
/// statics in `infr_core::iquant_grids` so there is exactly one copy of each table.
fn iquant_grids_msl() -> String {
    use infr_core::iquant_grids as ig;
    let mut s =
        String::from("// Auto-generated from infr_core::iquant_grids (single source of truth).\n");
    emit_grid(&mut s, "uchar", "KSIGNS_IQ2XS", &ig::KSIGNS_IQ2XS, "");
    emit_grid(&mut s, "ulong", "IQ2XXS_GRID", &ig::IQ2XXS_GRID, "ul");
    emit_grid(&mut s, "ulong", "IQ2XS_GRID", &ig::IQ2XS_GRID, "ul");
    emit_grid(&mut s, "ulong", "IQ2S_GRID", &ig::IQ2S_GRID, "ul");
    emit_grid(&mut s, "uint", "IQ3XXS_GRID", &ig::IQ3XXS_GRID, "u");
    emit_grid(&mut s, "uint", "IQ3S_GRID", &ig::IQ3S_GRID, "u");
    s
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
