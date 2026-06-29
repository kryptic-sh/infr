//! CPU backend validation: the backend-agnostic compute Graph, run on the CPU reference backend,
//! must produce the same greedy generation as the GPU path for a dense Qwen3 model.
//!
//! Run (needs a Vulkan GPU for the reference side + the Qwen3 GGUF):
//!   INFR_TEMP=0 cargo test --release -p infr-llama --test cpu_backend -- --ignored --nocapture

use std::path::PathBuf;

// ─── CPU-only correctness (no GPU) ───────────────────────────────────────────────
//
// These don't compare against the GPU: the CPU does the math in f32 while the GPU uses f16 + quant
// kernels, so greedy decode would split on near-ties (precision, not a bug). Instead the CPU path is
// validated by a **golden hash**: plain-text prompts are rendered through the model's jinja chat
// template, generated greedily (deterministic), and a stable FNV-1a of the output is locked — each
// golden was captured with `INFR_BLESS=1` and the text read to confirm it's coherent + correct, so
// any op regression flips the hash. Kernel-level math (the Q4_K/Q6_K dot vs the f32 reference) is
// unit-tested in `src/cpu_backend.rs`. The `cpu_matches_gpu_*` tests below remain as one-shot GPU
// *integration* checks from bring-up.

/// Stable FNV-1a-64 over a string. (`std::hash::DefaultHasher` is NOT stable across toolchains, so we
/// roll our own for golden values.)
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Greedy CPU generation with NO GPU: load via [`infr_llama::CpuModel`] (Vulkan-free), render the
/// prompt with the model's own chat template (so an instruct model answers coherently), collect the
/// streamed text. This is exactly the production `INFR_CPU=1` path.
fn cpu_gen(model: &infr_llama::CpuModel, prompt: &str, n: usize) -> String {
    // Inputs are plain text; `render_chat` (the GGUF's jinja template) turns them into the exact
    // token stream the instruct model expects.
    let mut out = String::new();
    model
        .generate_cpu(&model.render_chat(prompt), n, |p| out.push_str(p))
        .expect("cpu generate");
    out
}

/// Golden hashes of the deterministic CPU output `(prompt, n, fnv1a)`. Capture/refresh with
/// `INFR_BLESS=1` (prints the tuples); paste them here. A buggy op flips the hash.
// Captured + verified coherent (chat-templated, Qwen3 thinks then answers): "…France's capital is
// Paris", a simple-terms computer explanation, an ocean paragraph.
const QWEN3_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0xfd63781ea3bfa785),
    (
        "Explain how a computer works in simple terms.",
        48,
        0xcf56ba8c4bb5c455,
    ),
    (
        "Write a short paragraph about the ocean.",
        48,
        0xe78aa4678afa273b,
    ),
];
// Captured + verified coherent: "The capital of France is Paris. 😊", a brave-knight short story.
const GEMMA3_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0xe5a37ab078db3a2c),
    (
        "Tell me a short story about a brave knight.",
        48,
        0x5147de9a0ddfae50,
    ),
];

/// Assert (or, with `INFR_BLESS=1`, print) the golden hash for each case.
fn check_golden(model: &infr_llama::CpuModel, cases: &[(&str, usize, u64)]) {
    let bless = std::env::var("INFR_BLESS").is_ok();
    for (prompt, n, want) in cases {
        let out = cpu_gen(model, prompt, *n);
        let h = fnv1a(&out);
        if bless {
            // Print the text too so a human can verify it's coherent before locking the hash.
            println!("    ({prompt:?}, {n}, 0x{h:016x}),  // {out:?}");
        } else {
            assert_eq!(
                h, *want,
                "golden hash changed for {prompt:?} (n={n})\n  out: {out:?}\n  got 0x{h:016x} want 0x{want:016x}"
            );
        }
    }
}

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

/// CPU-only (no GPU): the deterministic Qwen3 output (short + long) must match its golden hash. Any
/// op regression flips the hash. Refresh with `INFR_BLESS=1`.
#[test]
#[ignore = "needs the Qwen3-0.6B GGUF (no GPU)"]
fn cpu_golden_qwen3() {
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&qwen3_06b(), None).expect("cpu load");
    check_golden(&model, QWEN3_GOLDEN);
}

/// CPU-only (no GPU): Gemma 3 (sandwich norms, GeGLU, dual-RoPE, SWA) golden-hash lock.
#[test]
#[ignore = "needs the gemma-3-1b GGUF (no GPU)"]
fn cpu_golden_gemma3() {
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&gemma3_1b(), None).expect("cpu load");
    check_golden(&model, GEMMA3_GOLDEN);
}

fn qwen35_08b() -> PathBuf {
    let hub = std::env::var("HOME").unwrap() + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots");
    for e in std::fs::read_dir(&base).expect("snapshots dir") {
        let f = e.unwrap().path().join("Qwen3.5-0.8B-Q4_K_M.gguf");
        if f.exists() {
            return f;
        }
    }
    panic!("Qwen3.5-0.8B gguf not found");
}

// Captured + verified coherent (qwen35 / Qwen3-Next: gated-DeltaNet + gated full-attention): "The
// capital of France is **Paris**. It is the largest city …", a knight story ("Elara … Aethelgard").
const QWEN35_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0x41a2c8d41bca554d),
    (
        "Tell me a short story about a brave knight.",
        48,
        0x0001ef9a6385fe30,
    ),
];

/// CPU-only (no GPU): qwen35 / Qwen3-Next golden-hash lock (the gated-DeltaNet recurrence + conv +
/// gated full-attention path). Uses the dedicated `qwen35::generate_cpu` runner.
#[test]
#[ignore = "needs the Qwen3.5-0.8B GGUF (no GPU)"]
fn cpu_golden_qwen35() {
    std::env::set_var("INFR_TEMP", "0");
    let path = qwen35_08b();
    let bless = std::env::var("INFR_BLESS").is_ok();
    for (prompt, n, want) in QWEN35_GOLDEN {
        let rendered = infr_llama::qwen35::render_chat(&path, prompt).expect("render");
        let mut out = String::new();
        infr_llama::qwen35::generate_cpu(&path, &rendered, *n, |p| out.push_str(p)).expect("gen");
        let h = fnv1a(&out);
        if bless {
            println!("    ({prompt:?}, {n}, 0x{h:016x}),  // {out:?}");
        } else {
            assert_eq!(
                h, *want,
                "qwen35 golden changed for {prompt:?}\n  out: {out:?}"
            );
        }
    }
}

fn qwen3moe_30b() -> PathBuf {
    let hub = std::env::var("HOME").unwrap() + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--unsloth--Qwen3-30B-A3B-GGUF/snapshots");
    for e in std::fs::read_dir(&base).expect("snapshots dir") {
        let f = e.unwrap().path().join("Qwen3-30B-A3B-Q4_K_M.gguf");
        if f.exists() {
            return f;
        }
    }
    panic!("Qwen3-30B-A3B gguf not found");
}

/// qwen3moe (routed-expert FFN: softmax router → top-k → renormalized weighted SwiGLU sum) on the
/// CPU backend must match the reference greedy path token-for-token. Only `n_used` experts run per
/// token, so the active params are ~3B (faster than a 12B dense despite the 30B total).
///
/// `INFR_NCMOE=999` forces the reference (GPU) side to run the experts on the host in **f32** — the
/// same precision as the CPU seam. Without it, the GPU's f16/quant expert kernels compute slightly
/// different router logits and flip the top-k expert *selection* (a near-tie at the 8th/9th of 128
/// experts), which cascades into a different greedy continuation. That's an inherent precision
/// difference, not a correctness gap: against the f32 reference the two match exactly.
#[test]
#[ignore = "needs a Vulkan GPU + the Qwen3-30B-A3B GGUF; run with INFR_TEMP=0"]
fn cpu_matches_gpu_qwen3moe() {
    std::env::set_var("INFR_TEMP", "0");
    std::env::set_var("INFR_NCMOE", "999"); // experts on host f32 (clamped to n_layer)
    let llama = infr_llama::Llama::load_opt(&qwen3moe_30b(), None).expect("load");
    let prompt = "The capital of France is";
    let n = 16;
    // MoE uses the dedicated GPU path (routed-expert FFN); INFR_TEMP=0 makes it greedy.
    let gpu = llama.generate_moe(prompt, n, |_| {}).expect("gpu generate");
    let cpu = llama.generate_cpu(prompt, n).expect("cpu generate");
    println!("GPU: {gpu:?}");
    println!("CPU: {cpu:?}");
    assert_eq!(
        cpu, gpu,
        "qwen3moe CPU must match host-f32 reference greedy output"
    );
}

fn gemma4_e2b() -> PathBuf {
    let hub = std::env::var("HOME").unwrap() + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--unsloth--gemma-4-E2B-it-GGUF/snapshots");
    for e in std::fs::read_dir(&base).expect("snapshots dir") {
        let f = e.unwrap().path().join("gemma-4-E2B-it-Q4_K_M.gguf");
        if f.exists() {
            return f;
        }
    }
    panic!("gemma-4-E2B gguf not found");
}

/// Gemma 4 E2B (gemma3n): per-layer input embeddings + KV-layer sharing, on top of the gemma4 dense
/// path. CPU backend must match the GPU greedy path token-for-token.
#[test]
#[ignore = "needs a Vulkan GPU + the gemma-4-E2B GGUF; run with INFR_TEMP=0"]
fn cpu_matches_gpu_gemma4_e2b() {
    std::env::set_var("INFR_TEMP", "0");
    let llama = infr_llama::Llama::load_opt(&gemma4_e2b(), None).expect("load");
    let prompt = "The capital of France is";
    let n = 16;
    let gpu = llama.generate(prompt, n, |_| {}).expect("gpu generate");
    let cpu = llama.generate_cpu(prompt, n).expect("cpu generate");
    println!("GPU: {gpu:?}");
    println!("CPU: {cpu:?}");
    assert_eq!(cpu, gpu, "gemma4 E2B CPU must match GPU greedy output");
}

// Captured + verified coherent (gemma4 E2B: per-layer input embeds + KV sharing): "The capital of
// France is **Paris**.", a brave-knight story ("Sir Kaelan … kingdom of Eldoria …").
const GEMMA4_E2B_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0xfd644a0cebde4e73),
    (
        "Tell me a short story about a brave knight.",
        48,
        0x2588804a8fb4c88f,
    ),
];

/// CPU-only (no GPU): Gemma 4 E2B golden-hash lock.
#[test]
#[ignore = "needs the gemma-4-E2B GGUF (no GPU)"]
fn cpu_golden_gemma4_e2b() {
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&gemma4_e2b(), None).expect("cpu load");
    check_golden(&model, GEMMA4_E2B_GOLDEN);
}

fn gemma4_12b() -> PathBuf {
    let hub = std::env::var("HOME").unwrap() + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--unsloth--gemma-4-12b-it-GGUF/snapshots");
    for e in std::fs::read_dir(&base).expect("snapshots dir") {
        let f = e.unwrap().path().join("gemma-4-12b-it-Q4_K_M.gguf");
        if f.exists() {
            return f;
        }
    }
    panic!("gemma-4-12b gguf not found");
}

/// Gemma 4 dense (per-layer SWA/full head dims, weightless V-norm, V=K reuse on full layers,
/// proportional-RoPE freq_factors, attn scale 1.0, per-layer output scale, final softcap) on the CPU
/// backend must match the GPU greedy path token-for-token. Small `n` — 12B re-dequants per step.
#[test]
#[ignore = "needs a Vulkan GPU + the gemma-4-12b GGUF; run with INFR_TEMP=0 (slow: 12B on CPU)"]
fn cpu_matches_gpu_gemma4() {
    std::env::set_var("INFR_TEMP", "0");
    let llama = infr_llama::Llama::load_opt(&gemma4_12b(), None).expect("load");
    let prompt = "The capital of France is";
    let n = 8;
    let gpu = llama.generate(prompt, n, |_| {}).expect("gpu generate");
    let cpu = llama.generate_cpu(prompt, n).expect("cpu generate");
    println!("GPU: {gpu:?}");
    println!("CPU: {cpu:?}");
    assert_eq!(cpu, gpu, "gemma4 CPU must match GPU greedy output");
}
