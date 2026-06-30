//! Generation goldens for both backends: a plain-text prompt is rendered through the model's jinja
//! chat template, generated greedily, and a stable FNV-1a of the output is locked. The CPU goldens
//! run the backend-agnostic compute Graph on the CPU reference backend (no GPU); the GPU goldens run
//! the production Vulkan path. Both are captured with `INFR_BLESS=1` and read for coherence.
//!
//! These are NOT `#[ignore]`d — each self-skips at runtime when its GGUF isn't in the HF cache (and
//! the GPU goldens additionally skip when no Vulkan device is present), so they RUN automatically
//! wherever the models + hardware exist, and quietly no-op elsewhere:
//!   INFR_TEMP=0 cargo test --release -p infr-llama --test cpu_backend -- --nocapture

use std::path::PathBuf;

/// Locate a cached GGUF `<file>` under `~/.cache/huggingface/hub/models--<repo>/snapshots/*/`, or
/// `None` if it isn't downloaded (the test self-skips). `repo` is the HF id with `/` → `--`.
fn find_gguf(repo: &str, file: &str) -> Option<PathBuf> {
    let hub = std::env::var("HOME").ok()? + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--{repo}/snapshots");
    std::fs::read_dir(&base).ok()?.find_map(|e| {
        let f = e.ok()?.path().join(file);
        f.exists().then_some(f)
    })
}

/// Resolve a model path or self-skip the test (it runs only when the GGUF is present).
macro_rules! need_model {
    ($opt:expr, $what:expr) => {
        match $opt {
            Some(p) => p,
            None => {
                eprintln!("skip: {} not in the HF cache", $what);
                return;
            }
        }
    };
}

/// Self-skip the test when there's no Vulkan device (the GPU goldens run only with a GPU present).
macro_rules! need_gpu {
    () => {
        if !infr_llama::gpu_available() {
            eprintln!("skip: no Vulkan GPU");
            return;
        }
    };
}

// ─── CPU-only correctness (no GPU) ───────────────────────────────────────────────
//
// The CPU and GPU goldens use SEPARATE hashes: the CPU does the math in f32 while the GPU uses f16 +
// native-quant kernels, so greedy decode can split on near-ties (precision, not a bug) — comparing
// the two token-for-token is brittle. Instead each backend locks its own FNV-1a golden, captured
// with `INFR_BLESS=1` and read to confirm it's coherent + correct, so any op regression flips the
// hash. Kernel-level math (the Q4_K/Q6_K dot vs the f32 reference) is unit-tested in
// `src/cpu_backend.rs`.

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
        .generate_cpu(&model.render_chat(prompt).expect("render chat"), n, |p| {
            out.push_str(p)
        })
        .expect("cpu generate");
    out
}

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

/// Greedy GPU generation: render the plain-text prompt with the model's chat template, generate on
/// the GPU dense path, return the text. The production GPU path; mirrors [`cpu_gen`].
fn gpu_gen(llama: &infr_llama::Llama, prompt: &str, n: usize) -> String {
    llama
        .generate(&llama.render_chat(prompt).expect("render chat"), n, |_| {})
        .expect("gpu generate")
}

/// As [`gpu_gen`] but via the routed-expert MoE forward ([`Llama::generate_moe`]).
fn gpu_gen_moe(llama: &infr_llama::Llama, prompt: &str, n: usize) -> String {
    llama
        .generate_moe(&llama.render_chat(prompt).expect("render chat"), n, |_| {})
        .expect("gpu moe generate")
}

/// Assert (or, with `INFR_BLESS=1`, print) the GPU golden hash for each `(prompt, n, fnv1a)` case.
fn check_gpu_golden(gen: impl Fn(&str, usize) -> String, cases: &[(&str, usize, u64)]) {
    let bless = std::env::var("INFR_BLESS").is_ok();
    for (prompt, n, want) in cases {
        let out = gen(prompt, *n);
        let h = fnv1a(&out);
        if bless {
            println!("    ({prompt:?}, {n}, 0x{h:016x}),  // {out:?}");
        } else {
            assert_eq!(
                h, *want,
                "GPU golden changed for {prompt:?} (n={n})\n  out: {out:?}\n  got 0x{h:016x} want 0x{want:016x}"
            );
        }
    }
}

// ─── Qwen3-0.6B (dense) ───────────────────────────────────────────────────────────

fn qwen3_06b() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
        return Some(PathBuf::from(p));
    }
    find_gguf("unsloth--Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf")
}

/// Path to a specific Qwen3-0.6B quantization in the HF cache (for the quant-coverage sweep).
fn qwen3_quant(quant: &str) -> Option<PathBuf> {
    find_gguf(
        "unsloth--Qwen3-0.6B-GGUF",
        &format!("Qwen3-0.6B-{quant}.gguf"),
    )
}

// Captured + verified coherent (chat-templated, Qwen3 thinks then answers): "…France's capital is
// Paris", a simple-terms computer explanation, an ocean paragraph.
const QWEN3_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0xfd63781ea3bfa785),
    (
        "Explain how a computer works in simple terms.",
        48,
        0xcb0381bae31a7d8f,
    ),
    (
        "Write a short paragraph about the ocean.",
        48,
        0xabca2bf79a3cdda2,
    ),
];

/// CPU-only: the deterministic Qwen3 output (short + long) must match its golden hash.
#[test]
fn cpu_golden_qwen3() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    check_golden(&model, QWEN3_GOLDEN);
}

// Captured + verified coherent on the GPU (chat-templated Qwen3-0.6B Q4_K_M).
const QWEN3_GPU_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0xfd63781ea3bfa785),
    (
        "Explain how a computer works in simple terms.",
        48,
        0xcf56ba8c4bb5c455,
    ),
];

/// GPU dense Qwen3-0.6B golden-hash lock.
#[test]
fn gpu_golden_qwen3() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    need_gpu!();
    std::env::set_var("INFR_TEMP", "0");
    let llama = infr_llama::Llama::load_opt(&path, None).expect("load");
    check_gpu_golden(|p, n| gpu_gen(&llama, p, n), QWEN3_GPU_GOLDEN);
}

// CPU quant coverage: the SAME prompt through every available quantization of Qwen3-0.6B — legacy
// round (Q4_0), k-quants (Q2_K/Q4_K/Q5_K/Q6_K), high-bit (Q8_0), i-quant codebook (IQ4_XS), and float
// (BF16). Each exercises a different dequant/dot path; the per-quant golden hash is locked (each
// verified coherent at capture). Missing quants are skipped. Refresh with `INFR_BLESS=1`.
// All verified coherent at capture — every quant recalls "France's capital is Paris" (Q2_K is a
// touch repetitive, as expected for 2-bit; the float-ish quants still converge: Q5_K==Q8_0==BF16).
const QWEN3_QUANT_GOLDEN: &[(&str, usize, u64)] = &[
    ("IQ4_XS", 32, 0xd028ff03b524cb28),
    ("Q2_K", 32, 0x6442c2818c12ca56),
    ("Q4_0", 32, 0x88221dcfca820246),
    ("Q4_K_M", 32, 0xfd63781ea3bfa785),
    ("Q5_K_M", 32, 0xb68f96c3aa8d22fe),
    ("Q6_K", 32, 0x925b523a6f67356b),
    ("Q8_0", 32, 0xb68f96c3aa8d22fe),
    ("BF16", 32, 0xb68f96c3aa8d22fe),
];

#[test]
fn cpu_golden_qwen3_quants() {
    std::env::set_var("INFR_TEMP", "0");
    let bless = std::env::var("INFR_BLESS").is_ok();
    let prompt = "The capital of France is";
    for (quant, n, want) in QWEN3_QUANT_GOLDEN {
        let Some(path) = qwen3_quant(quant) else {
            eprintln!("skip {quant}: not downloaded");
            continue;
        };
        let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
        let out = cpu_gen(&model, prompt, *n);
        let h = fnv1a(&out);
        if bless {
            println!("    ({quant:?}, {n}, 0x{h:016x}),  // {out:?}");
        } else {
            assert_eq!(h, *want, "quant {quant} golden changed\n  out: {out:?}");
        }
    }
}

// GPU quant coverage: the SAME prompt through every downloaded Qwen3-0.6B quant, all via the raw
// native-block upload — affine (Q4_0, Q2_K/Q4_K/Q5_K/Q6_K, Q8_0) AND the IQ4_XS codebook i-quant,
// which now decodes natively in-shader (no host→f16). BF16 is float → the plain f16 GEMV. Hashes are
// GPU-specific; captured INFR_BLESS=1 and read coherent ("…Paris"). Missing quants are skipped.
const QWEN3_QUANT_GPU_GOLDEN: &[(&str, usize, u64)] = &[
    ("IQ4_XS", 32, 0xd028ff03b524cb28),
    ("Q2_K", 32, 0x6442c2818c12ca56),
    ("Q4_0", 32, 0x88221dcfca820246),
    ("Q4_K_M", 32, 0xfd63781ea3bfa785),
    ("Q5_K_M", 32, 0x4e510646d603bc03),
    ("Q6_K", 32, 0xb68f96c3aa8d22fe),
    ("Q8_0", 32, 0xb68f96c3aa8d22fe),
    ("BF16", 32, 0xb68f96c3aa8d22fe),
];

/// GPU native-upload coverage across quant formats — proves the codebook IQ4_XS path runs natively
/// alongside the affine k-quants. Refresh with `INFR_BLESS=1`.
#[test]
fn gpu_golden_qwen3_quants() {
    need_gpu!();
    std::env::set_var("INFR_TEMP", "0");
    let bless = std::env::var("INFR_BLESS").is_ok();
    let prompt = "The capital of France is";
    for (quant, n, want) in QWEN3_QUANT_GPU_GOLDEN {
        let Some(path) = qwen3_quant(quant) else {
            eprintln!("skip {quant}: not downloaded");
            continue;
        };
        let llama = infr_llama::Llama::load_opt(&path, None).expect("load");
        let out = gpu_gen(&llama, prompt, *n);
        let h = fnv1a(&out);
        if bless {
            println!("    ({quant:?}, {n}, 0x{h:016x}),  // {out:?}");
        } else {
            assert_eq!(h, *want, "GPU quant {quant} golden changed\n  out: {out:?}");
        }
    }
}

// ─── Gemma 3 (dense) ────────────────────────────────────────────────────────────

fn gemma3_1b() -> Option<PathBuf> {
    find_gguf("unsloth--gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf")
}

// Captured + verified coherent: "The capital of France is Paris. 😊", a brave-knight short story.
const GEMMA3_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0xe5a37ab078db3a2c),
    (
        "Tell me a short story about a brave knight.",
        48,
        0x3fd9d19a20538db3,
    ),
];

/// CPU-only: Gemma 3 (sandwich norms, GeGLU, dual-RoPE, SWA) golden-hash lock.
#[test]
fn cpu_golden_gemma3() {
    let path = need_model!(gemma3_1b(), "gemma-3-1b");
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    check_golden(&model, GEMMA3_GOLDEN);
}

// Captured + verified coherent on the GPU (gemma-3-1b Q4_K_M: sandwich norms, GeGLU, dual-RoPE, SWA).
const GEMMA3_GPU_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0xe5a37ab078db3a2c),
    (
        "Tell me a short story about a brave knight.",
        48,
        0x5147de9a0ddfae50,
    ),
];

/// GPU dense Gemma 3 golden-hash lock (sandwich norms, GeGLU, dual-RoPE, SWA, √n_embd embed scale).
#[test]
fn gpu_golden_gemma3() {
    let path = need_model!(gemma3_1b(), "gemma-3-1b");
    need_gpu!();
    std::env::set_var("INFR_TEMP", "0");
    let llama = infr_llama::Llama::load_opt(&path, None).expect("load");
    check_gpu_golden(|p, n| gpu_gen(&llama, p, n), GEMMA3_GPU_GOLDEN);
}

// ─── Qwen3.5 / Qwen3-Next (gated DeltaNet) ──────────────────────────────────────

fn qwen35_08b() -> Option<PathBuf> {
    find_gguf("unsloth--Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf")
}

// Captured + verified coherent (qwen35 / Qwen3-Next: gated-DeltaNet + gated full-attention): "The
// capital of France is **Paris**. It is the largest city …", a knight story ("Elara … Aethelgard").
// Renders at the template's default (non-thinking for Qwen3-Next; no INFR_THINK override).
const QWEN35_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0x41a2c8d41bca554d),
    (
        "Tell me a short story about a brave knight.",
        48,
        0x0001ef9a6385fe30,
    ),
];

/// CPU-only: qwen35 / Qwen3-Next golden-hash lock (gated-DeltaNet recurrence + conv + gated full
/// attention). Uses the dedicated `qwen35::generate_cpu` runner.
#[test]
fn cpu_golden_qwen35() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    std::env::set_var("INFR_TEMP", "0");
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

// ─── Qwen3-MoE (routed experts) ─────────────────────────────────────────────────

fn qwen3moe_30b() -> Option<PathBuf> {
    find_gguf("unsloth--Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q4_K_M.gguf")
}

// Captured + verified coherent (qwen3moe: routed-expert FFN, ~3B active of 30B).
const QWEN3MOE_GOLDEN: &[(&str, usize, u64)] =
    &[("The capital of France is", 24, 0xdac3e0eea1da12ed)];

/// CPU-only: qwen3moe golden-hash lock (the Op::MoeFfn routed-expert path). 30B but only `n_used`
/// experts run per token; still slow on CPU, so a single short case.
#[test]
fn cpu_golden_qwen3moe() {
    let path = need_model!(qwen3moe_30b(), "Qwen3-30B-A3B");
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    check_golden(&model, QWEN3MOE_GOLDEN);
}

// Captured + verified coherent on the GPU (Qwen3-30B-A3B Q4_K_M: routed-expert FFN, ~3B active).
const QWEN3MOE_GPU_GOLDEN: &[(&str, usize, u64)] =
    &[("The capital of France is", 24, 0x193c084bdd8c8c48)];

/// GPU qwen3moe golden-hash lock (routed-expert FFN: softmax router → top-k → renormalized weighted
/// SwiGLU sum). Only `n_used` of 128 experts run per token; uses the dedicated MoE GPU forward.
#[test]
fn gpu_golden_qwen3moe() {
    let path = need_model!(qwen3moe_30b(), "Qwen3-30B-A3B");
    need_gpu!();
    std::env::set_var("INFR_TEMP", "0");
    let llama = infr_llama::Llama::load_opt(&path, None).expect("load");
    check_gpu_golden(|p, n| gpu_gen_moe(&llama, p, n), QWEN3MOE_GPU_GOLDEN);
}

// ─── Gemma 4 E2B (gemma3n: per-layer embeds + KV sharing) ───────────────────────

fn gemma4_e2b() -> Option<PathBuf> {
    find_gguf("unsloth--gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_K_M.gguf")
}

// Captured + verified coherent (gemma4 E2B: per-layer input embeds + KV sharing): "The capital of
// France is **Paris**.", a brave-knight story ("Sir Kaelan … kingdom of Eldoria …").
const GEMMA4_E2B_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0xfd644a0cebde4e73),
    (
        "Tell me a short story about a brave knight.",
        48,
        0xd1281a5e24ad58b9,
    ),
];

/// CPU-only: Gemma 4 E2B golden-hash lock.
#[test]
fn cpu_golden_gemma4_e2b() {
    let path = need_model!(gemma4_e2b(), "gemma-4-E2B");
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    check_golden(&model, GEMMA4_E2B_GOLDEN);
}

// Captured + verified coherent on the GPU (gemma-4-E2B Q4_K_M: per-layer input embeds + KV sharing).
const GEMMA4_E2B_GPU_GOLDEN: &[(&str, usize, u64)] =
    &[("The capital of France is", 32, 0xfd644a0cebde4e73)];

/// GPU Gemma 4 E2B (gemma3n) golden-hash lock: per-layer input embeddings + KV-layer sharing on top
/// of the gemma4 dense path.
#[test]
fn gpu_golden_gemma4_e2b() {
    let path = need_model!(gemma4_e2b(), "gemma-4-E2B");
    need_gpu!();
    std::env::set_var("INFR_TEMP", "0");
    let llama = infr_llama::Llama::load_opt(&path, None).expect("load");
    check_gpu_golden(|p, n| gpu_gen(&llama, p, n), GEMMA4_E2B_GPU_GOLDEN);
}

// ─── Gemma 4 12b (dense) ────────────────────────────────────────────────────────

fn gemma4_12b() -> Option<PathBuf> {
    find_gguf("unsloth--gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf")
}

// Captured + verified coherent on the GPU (gemma-4-12b Q4_K_M: per-layer SWA/full head dims,
// weightless V-norm, V=K reuse, freq_factors, attn scale 1.0, per-layer output scale, final softcap).
const GEMMA4_12B_GPU_GOLDEN: &[(&str, usize, u64)] =
    &[("The capital of France is", 32, 0xfd644a0cebde4e73)];

/// GPU dense Gemma 4 (12b) golden-hash lock: per-layer SWA/full head dims, weightless V-norm, V=K
/// reuse on full layers, proportional-RoPE freq_factors, attn scale 1.0, per-layer output scale,
/// final softcap.
#[test]
fn gpu_golden_gemma4() {
    let path = need_model!(gemma4_12b(), "gemma-4-12b");
    need_gpu!();
    std::env::set_var("INFR_TEMP", "0");
    let llama = infr_llama::Llama::load_opt(&path, None).expect("load");
    check_gpu_golden(|p, n| gpu_gen(&llama, p, n), GEMMA4_12B_GPU_GOLDEN);
}
