//! GPU logits within tolerance of the CPU reference (`docs/gpu-tolerance.md`).
//!
//! Each case runs the CPU reference backend (`kernels.cpu.reference`) greedily on a prompt, then
//! teacher-forces the very same tokens through the Vulkan seam and compares the two backends'
//! full-vocabulary logits at every position — the batched prefill's rows (at two chunk heights)
//! and one decode step per generated token — with `infr_llama::tolerance`'s metrics, against the
//! tolerance table for the model's path class.
//!
//! ```text
//! INFR_DEV=Vulkan0 cargo nextest run -p infr-llama --release --test gpu_tolerance \
//!   --run-ignored all --test-threads 1 --no-capture
//! ```

#[path = "support/forced.rs"]
mod forced;

#[test]
fn forced_run_reproduces_the_cpu_reference_on_every_path() {
    forced::self_check_on_a_synthetic_llama();
}

/// A measurement, not a gate: CPU production and every Vulkan configuration against the CPU
/// reference on the models named in `INFR_TOL_MODELS` (keys of [`forced::MODELS`]), printed as one
/// line per pair and phase (see [`forced::measure`]).
#[test]
#[ignore = "measurement: needs cached models and a Vulkan GPU"]
fn measure_gpu_tolerance() {
    let Ok(keys) = std::env::var("INFR_TOL_MODELS") else {
        eprintln!("skip: INFR_TOL_MODELS unset");
        return;
    };
    for key in keys.split(',') {
        forced::measure(key);
    }
}

use infr_llama::tolerance::PathClass::{self, Float, Int8, KvQ8};

/// A quantized model: the production kernels at both prefill chunk heights, and the same with
/// every int8-activation kernel off.
const QUANT_RUNS: &[(&str, PathClass)] = &[
    ("gpu", Int8),
    ("gpu-ub32", Int8),
    ("gpu-float", Float),
    ("gpu-float-ub32", Float),
];

/// A float-weight model: the production kernels at both prefill chunk heights.
const FLOAT_RUNS: &[(&str, PathClass)] = &[("gpu", Float), ("gpu-ub32", Float)];

#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_tolerance_synthetic_f32() {
    forced::gate("synthetic-f32", FLOAT_RUNS);
}

#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_tolerance_synthetic_f16() {
    forced::gate("synthetic-f16", FLOAT_RUNS);
}

#[test]
#[ignore = "needs Qwen3-0.6B Q4_K_M and a Vulkan GPU"]
fn gpu_seam_tolerance_qwen3() {
    let mut runs = QUANT_RUNS.to_vec();
    runs.push(("gpu-kv-q8-float", KvQ8));
    forced::gate("qwen3-0.6b", &runs);
}

#[test]
#[ignore = "needs gemma-3-1b-it Q4_K_M and a Vulkan GPU"]
fn gpu_seam_tolerance_gemma3() {
    forced::gate("gemma3-1b", QUANT_RUNS);
}

#[test]
#[ignore = "needs Qwen3.5-0.8B Q4_K_M and a Vulkan GPU"]
fn gpu_seam_tolerance_qwen35() {
    forced::gate("qwen35-0.8b", QUANT_RUNS);
}

#[test]
#[ignore = "needs LFM3-1.7B Q4_K_M and F16 and a Vulkan GPU"]
fn gpu_seam_tolerance_lfm3() {
    forced::gate("lfm3-q4", QUANT_RUNS);
    forced::gate("lfm3-f16", FLOAT_RUNS);
}

#[test]
#[ignore = "needs ERNIE-4.5-0.3B Q4_K_M and F16 and a Vulkan GPU"]
fn gpu_seam_tolerance_ernie() {
    forced::gate("ernie-0.3b-q4", QUANT_RUNS);
    forced::gate("ernie-0.3b-f16", FLOAT_RUNS);
}
