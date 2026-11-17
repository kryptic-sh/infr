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
