//! CPU backend validation: the backend-agnostic compute Graph, run on the CPU reference backend,
//! must produce the same greedy generation as the GPU path for a dense Qwen3 model.
//!
//! Run (needs a Vulkan GPU for the reference side + the Qwen3 GGUF):
//!   INFR_TEMP=0 cargo test --release -p infr-llama --test cpu_backend -- --ignored --nocapture

use std::path::PathBuf;

fn qwen3_06b() -> PathBuf {
    if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
        return PathBuf::from(p);
    }
    let hub = std::env::var("HOME").unwrap() + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--unsloth--Qwen3-0.6B-GGUF/snapshots");
    for e in std::fs::read_dir(&base).expect("snapshots dir") {
        let f = e.unwrap().path().join("Qwen3-0.6B-Q4_K_M.gguf");
        if f.exists() {
            return f;
        }
    }
    panic!("Qwen3-0.6B gguf not found; set INFR_TEST_MODEL");
}

/// CPU greedy generation must match the GPU greedy generation token-for-token (both argmax).
/// Set INFR_TEMP=0 so the GPU side is greedy too.
#[test]
#[ignore = "needs a Vulkan GPU + the Qwen3-0.6B GGUF; run with INFR_TEMP=0"]
fn cpu_matches_gpu_greedy() {
    std::env::set_var("INFR_TEMP", "0");
    let m = qwen3_06b();
    let llama = infr_llama::Llama::load_opt(&m, None).expect("load");

    let prompt = "The capital of France is";
    let n = 24;

    let gpu = llama.generate(prompt, n, |_| {}).expect("gpu generate");
    let cpu = llama.generate_cpu(prompt, n).expect("cpu generate");

    println!("GPU: {gpu:?}");
    println!("CPU: {cpu:?}");
    assert_eq!(
        cpu, gpu,
        "CPU reference output must match GPU greedy output"
    );
}

fn gemma3_1b() -> PathBuf {
    let hub = std::env::var("HOME").unwrap() + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--unsloth--gemma-3-1b-it-GGUF/snapshots");
    for e in std::fs::read_dir(&base).expect("snapshots dir") {
        let f = e.unwrap().path().join("gemma-3-1b-it-Q4_K_M.gguf");
        if f.exists() {
            return f;
        }
    }
    panic!("gemma-3-1b gguf not found");
}

/// Gemma 3 (sandwich norms, GeGLU, dual-RoPE, SWA, √n_embd embed scale) on the CPU backend must
/// match the GPU greedy path token-for-token.
#[test]
#[ignore = "needs a Vulkan GPU + the gemma-3-1b GGUF; run with INFR_TEMP=0"]
fn cpu_matches_gpu_gemma3() {
    std::env::set_var("INFR_TEMP", "0");
    let llama = infr_llama::Llama::load_opt(&gemma3_1b(), None).expect("load");
    let prompt = "The capital of France is";
    let n = 24;
    let gpu = llama.generate(prompt, n, |_| {}).expect("gpu generate");
    let cpu = llama.generate_cpu(prompt, n).expect("cpu generate");
    println!("GPU: {gpu:?}");
    println!("CPU: {cpu:?}");
    assert_eq!(cpu, gpu, "gemma3 CPU must match GPU greedy output");
}
