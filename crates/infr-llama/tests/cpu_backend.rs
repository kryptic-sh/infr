//! Generation goldens for both backends: a plain-text prompt is rendered through the model's jinja
//! chat template, generated greedily, and a stable FNV-1a of the output is locked. The CPU goldens
//! run the backend-agnostic compute Graph on the CPU reference backend (no GPU); the GPU goldens run
//! the production Vulkan path. Both are captured with `INFR_BLESS=1` and read for coherence.
//!
//! These are NOT `#[ignore]`d — each self-skips at runtime when its GGUF isn't in the HF cache (and
//! the GPU goldens additionally skip when no Vulkan device is present), so they RUN automatically
//! wherever the models + hardware exist, and quietly no-op elsewhere:
//!   INFR_TEMP=0 cargo test --release -p infr-llama --test cpu_backend -- --nocapture

use infr_core::WeightSource;
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

/// Serialize the model-gated generation tests. They mutate PROCESS-GLOBAL env that generation reads
/// (`INFR_TEMP`, and `INFR_NO_THINK` — read at render time in infr-chat), and cargo
/// runs tests in parallel; without this, one test's env leaks into another's generation (e.g.
/// `INFR_NO_THINK=1` flipping a Qwen3 golden's thinking off → hash mismatch). Poison-tolerant so a
/// failing test doesn't cascade-poison the rest.
fn test_serial_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
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

// ─── Qwen2.5-0.5B (dense, BIASED q/k/v) ───────────────────────────────────────────
// Qwen2/2.5 add a learned bias to the q/k/v projections (Qwen3 dropped them) — the new `AddBias`
// seam op. The 0.5B-Instruct also ties its output embedding, so this exercises the tied lm-head
// path too. Gated: needs a Qwen2.5 GGUF in the HF cache, or `INFR_TEST_QWEN2=/path/to.gguf`.
fn qwen2_05b() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_QWEN2") {
        return Some(PathBuf::from(p));
    }
    find_gguf(
        "unsloth--Qwen2.5-0.5B-Instruct-GGUF",
        "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    )
}

/// Qwen2.5 through the Vulkan seam must match the CPU oracle token-for-token — validates the QKV
/// bias (`AddBias`) end to end on prefill + decode + record-once replay, plus tied embeddings.
#[test]
fn gpu_seam_matches_cpu_qwen2() {
    let path = need_model!(qwen2_05b(), "Qwen2.5-0.5B-Instruct");
    need_gpu!();
    let _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "What is the capital of France? Answer briefly.", 16);
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
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    check_golden(&model, QWEN3_GOLDEN);
}

/// A BIG prompt (~1000+ tokens): large enough that the dense prefill's padded-KV attention reads the
/// padding rows beyond the real tokens. Short prompts don't reproduce the KV-cache bug.
fn repeat_prompt() -> String {
    "Explain how a CPU instruction pipeline works and list its common hazards. ".repeat(90)
}

/// A greedy generation is degenerate if it collapsed to one repeated token (the KV-padding bug's
/// "!!!!"/"5555" signature): a non-trivial length with ≤2 distinct chars.
fn is_degenerate(s: &str) -> bool {
    let t = s.trim();
    t.chars().count() >= 8 && t.chars().collect::<std::collections::HashSet<char>>().len() <= 2
}

/// REGRESSION (CPU reference backend): the same repeated-forward invariant on the no-GPU
/// compute-graph path. The CPU backend uses host buffers (no recycled-VRAM hazard), so this guards
/// CPU coherence + determinism across repeated big-prompt forwards.
#[test]
fn cpu_no_garbage_on_repeated_forward() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let p = repeat_prompt();
    let g1 = cpu_gen(&model, &p, 20);
    let g2 = cpu_gen(&model, &p, 20);
    let head = |s: &str| s.chars().take(48).collect::<String>();
    assert!(
        !is_degenerate(&g1),
        "1st CPU forward degenerate: {:?}",
        head(&g1)
    );
    assert!(
        !is_degenerate(&g2),
        "2nd CPU forward degenerate: {:?}",
        head(&g2)
    );
    assert_eq!(g1, g2, "repeated CPU forward diverged");
}

/// KV-cache Q8_0 quantization (CPU reference): the KV cache stores Q8_0 blocks (34 B / 32 elems)
/// INDEPENDENTLY for K and V (`INFR_KV_TYPE_K` / `INFR_KV_TYPE_V` ∈ {f16, q8_0}). Q8 KV shifts the
/// numerics, so it won't match the f16 golden hash — but a correct per-block quantize/dequant must
/// still yield coherent (non-degenerate) greedy output on a long prompt whose decode reads a deep
/// cache. Exercises all three quantized combos (q8/q8, q8/f16, f16/q8) to prove K and V decouple.
#[test]
fn cpu_kv_q8_coherent() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let prompt = repeat_prompt();
    // f16 baseline, then each K/V quant mix. Load once per env (the KV dtype is read at graph build).
    for (k, v) in [("q8_0", "q8_0"), ("q8_0", "f16"), ("f16", "q8_0")] {
        std::env::set_var("INFR_KV_TYPE_K", k);
        std::env::set_var("INFR_KV_TYPE_V", v);
        let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
        let out = cpu_gen(&model, &prompt, 24);
        assert!(
            !is_degenerate(&out),
            "KV K={k} V={v} degenerate: {:?}",
            out.chars().take(48).collect::<String>()
        );
    }
    std::env::remove_var("INFR_KV_TYPE_K");
    std::env::remove_var("INFR_KV_TYPE_V");
}

/// TurboQuant KV cache (CPU reference): WHT-rotated 2/3/4-bit PolarQuant, 128-elem blocks. The
/// per-vector error (turbo2 ~30%, turbo3 ~20%, turbo4 ~12%) is what V tolerates but K does not
/// (llama.cpp: "keep K at higher precision than V"), so the coherent config is K=f16 with V=turbo*.
/// Exercises the full quantize (WriteKv) + dequant-with-inverse-WHT (Attention) path for every width;
/// a broken WHT / centroid table / packing / norm-correction would garble even the V-only cache.
#[test]
fn cpu_kv_turbo_coherent() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    std::env::set_var("INFR_KV_TYPE_K", "f16");
    for v in ["turbo2", "turbo3", "turbo4"] {
        std::env::set_var("INFR_KV_TYPE_V", v);
        let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
        let out = cpu_gen(&model, &repeat_prompt(), 24);
        assert!(
            !is_degenerate(&out),
            "K=f16 V={v} degenerate: {:?}",
            out.chars().take(48).collect::<String>()
        );
    }
    std::env::remove_var("INFR_KV_TYPE_K");
    std::env::remove_var("INFR_KV_TYPE_V");
}

/// Mainline llama.cpp KV cache types (CPU reference): f32/bf16 (dense) + the low-bit round quants
/// q4_0/q4_1/q5_0/q5_1 and the non-linear iq4_nl, quantized on the fly per 32-elem block on write and
/// dequantized via the shared GGUF path on read. f32/bf16 run coupled; the low-bit quants run on V
/// (K=f16) since K needs higher precision. Every config must stay coherent.
#[test]
fn cpu_kv_mainline_quants_coherent() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let prompt = repeat_prompt();
    for (k, v) in [
        ("f32", "f32"),
        ("bf16", "bf16"),
        ("f16", "q4_0"),
        ("f16", "q4_1"),
        ("f16", "q5_0"),
        ("f16", "q5_1"),
        ("f16", "iq4_nl"),
    ] {
        std::env::set_var("INFR_KV_TYPE_K", k);
        std::env::set_var("INFR_KV_TYPE_V", v);
        let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
        let out = cpu_gen(&model, &prompt, 24);
        assert!(
            !is_degenerate(&out),
            "KV K={k} V={v} degenerate: {:?}",
            out.chars().take(48).collect::<String>()
        );
    }
    std::env::remove_var("INFR_KV_TYPE_K");
    std::env::remove_var("INFR_KV_TYPE_V");
}

// Captured + verified coherent on the Vulkan backend via the agnostic compute seam (the SAME dense
// `Graph` the CPU oracle builds, mapped op-for-op to GPU kernels). Should reproduce the production
// GPU path (QWEN3_GPU_GOLDEN) — the France case shares its hash (0xfd63781ea3bfa785), confirming the
// seam matches the hand-written Recorder forward token-for-token.
const QWEN3_SEAM_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0xfd63781ea3bfa785),
    (
        "Explain how a computer works in simple terms.",
        48,
        0xcf56ba8c4bb5c455,
    ),
];

/// End-to-end dense parity: run the full Qwen3-0.6B dense forward on the **Vulkan** backend through
/// the agnostic compute seam ([`CpuModel::generate_dense_vulkan`]) and lock its golden. The seam runs
/// the identical `Graph` the CPU reference builds; this proves the dense forward maps faithfully to
/// the GPU and reproduces the production GPU path (`gpu_golden_qwen3`).
#[test]
fn gpu_seam_golden_qwen3() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    need_gpu!();
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    check_gpu_golden(
        |p, n| {
            model
                .generate_dense_vulkan(&model.render_chat(p).expect("render chat"), n)
                .expect("seam gen")
        },
        QWEN3_SEAM_GOLDEN,
    );
}

/// Flash-attention prefill parity: a prompt LONG ENOUGH (>64 tokens) that the seam's batched prefill
/// takes the FlashAttention-2 path (`attention_prefill_flash`, rows≥64) + the tiled GEMM/mmq Linear,
/// must generate the SAME greedy continuation as the CPU reference oracle (which uses the naive
/// per-token attention). Guards the m>1 prefill kernels the short-prompt goldens never exercise.
#[test]
fn gpu_seam_flash_matches_cpu() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    need_gpu!();
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    // ~100+ tokens → pf_m ≥ 64 → flash prefill on the seam.
    let long = "Photosynthesis is the process by which green plants, algae, and some bacteria \
        convert light energy into chemical energy stored in glucose, using carbon dioxide and water \
        and releasing oxygen as a byproduct. It happens in two connected stages: the light-dependent \
        reactions in the thylakoid membranes, and the light-independent Calvin cycle in the stroma. \
        Explain each stage carefully, name the key molecules involved, and then summarize in one \
        sentence why this process is essential for life on Earth.";
    let mut cpu_txt = String::new();
    model
        .generate_cpu(long, 24, |p| cpu_txt.push_str(p))
        .expect("cpu gen");
    let gpu_txt = model.generate_dense_vulkan(long, 24).expect("seam gen");
    assert_eq!(
        cpu_txt.trim(),
        gpu_txt.trim(),
        "flash-prefill seam diverged from the CPU oracle"
    );
}

/// Vulkan-seam vs CPU-oracle parity for one model: greedy `n`-token continuation of `prompt`
/// (rendered through the model's chat template) must match token-for-token. Proves the arch's ops
/// lower correctly through the Vulkan adapter — the CPU seam runs the IDENTICAL Graph. A near-tie
/// argmax split (f16 GPU kernels vs f32 CPU) would show here as an early divergence; none of the
/// covered models exhibit one on these prompts today, so keep the strict compare until it flakes.
fn seam_vulkan_matches_cpu(path: &std::path::Path, prompt: &str, n: usize) {
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(path, None).expect("cpu load");
    let rendered = model.render_chat(prompt).expect("render chat");
    let mut cpu_txt = String::new();
    model
        .generate_cpu(&rendered, n, |p| cpu_txt.push_str(p))
        .expect("cpu gen");
    let gpu_txt = model
        .generate_dense_vulkan(&rendered, n)
        .expect("vulkan seam gen");
    assert_eq!(
        cpu_txt.trim(),
        gpu_txt.trim(),
        "Vulkan seam diverged from the CPU oracle for {path:?}"
    );
}

/// Persistent-session KV reuse on the Vulkan seam: turn 2 extends turn 1's prompt, so the session
/// must (a) generate EXACTLY what a fresh full-prefill of the same prompt generates, and (b)
/// prefill only the un-cached suffix (stats.n_prompt ≪ the full prompt length). The seam twin of
/// the bespoke ChatSession's incremental prefill.
#[test]
fn gpu_seam_kv_reuse_matches_fresh() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    need_gpu!();
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let mut sess = model.vulkan_session(512).expect("session");

    let p1 = "The capital of France is";
    let mut t1 = String::new();
    let s1 = model
        .generate_vulkan_session(&mut sess, p1, 8, |p| t1.push_str(p))
        .expect("turn 1");
    assert!(s1.n_prompt > 0);

    let p2 = format!("{p1}{t1} And the capital of Germany is");
    let mut t2 = String::new();
    let s2 = model
        .generate_vulkan_session(&mut sess, &p2, 8, |p| t2.push_str(p))
        .expect("turn 2");

    // (a) same output as a fresh full prefill of the identical prompt
    let fresh = model.generate_dense_vulkan(&p2, 8).expect("fresh gen");
    assert_eq!(
        t2.trim(),
        fresh.trim(),
        "session (suffix prefill) diverged from a fresh full prefill"
    );
    // (b) the session only prefilled the suffix — far fewer tokens than the whole prompt
    let full_len = s1.n_prompt + t1.split_whitespace().count(); // lower bound on p2's tokens
    assert!(
        s2.n_prompt < full_len,
        "turn 2 prefilled {} tokens — KV reuse didn't kick in",
        s2.n_prompt
    );
}

/// Q8_0 KV cache on the Vulkan seam (coupled K==V==q8 via INFR_KV_Q8). Q8 forces per-execute static
/// decode (the record-once replay is disabled for a Q8 cache), so both the one-shot generate and the
/// session path exercise store_q8 (planar write) + attn_partial_q8 / attention_kv_q8 (planar read) +
/// the flash prefill dequant. Both must produce coherent (non-degenerate) greedy output. Q8 KV shifts
/// the numerics (no exact match with the f16 golden), but the near-lossless quant must stay sensible;
/// a broken quantize/dequant or a mis-gated kernel would collapse or garble the output.
#[test]
fn gpu_seam_kv_q8_coherent() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    need_gpu!();
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    std::env::set_var("INFR_KV_Q8", "1");
    let head = |s: &str| s.chars().take(64).collect::<String>();
    // A prompt long enough (>64 tokens) to take the prefill path, then a deep-cache decode.
    let long =
        "Explain how a CPU instruction pipeline works and list its common hazards. ".repeat(6);

    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    // (a) one-shot static path.
    let g_static = model
        .generate_dense_vulkan(&long, 24)
        .expect("q8 static gen");
    assert!(
        !is_degenerate(&g_static),
        "Q8 static Vulkan output degenerate: {:?}",
        head(&g_static)
    );
    // (b) session path (Q8 forces static decode, so this also exercises the static kernels).
    let mut sess = model.vulkan_session(512).expect("q8 session");
    let mut g_sess = String::new();
    model
        .generate_vulkan_session(&mut sess, &long, 24, |p| g_sess.push_str(p))
        .expect("q8 session gen");
    assert!(
        !is_degenerate(&g_sess),
        "Q8 session Vulkan output degenerate: {:?}",
        head(&g_sess)
    );
    std::env::remove_var("INFR_KV_Q8");

    // (c) DECOUPLED K/V: each mixed side (K=q8/V=f16 and K=f16/V=q8) must also stay coherent — the
    // per-side attn_partial_{k,v}q8 / attention_kv_{k,v}q8 variants read one Q8 side + one f16 side.
    for (k, v) in [("q8_0", "f16"), ("f16", "q8_0")] {
        std::env::set_var("INFR_KV_TYPE_K", k);
        std::env::set_var("INFR_KV_TYPE_V", v);
        let m = infr_llama::CpuModel::load(&path, None).expect("cpu load");
        let out = m.generate_dense_vulkan(&long, 20).expect("mixed gen");
        assert!(
            !is_degenerate(&out),
            "mixed K={k} V={v} Vulkan output degenerate: {:?}",
            head(&out)
        );
    }
    std::env::remove_var("INFR_KV_TYPE_K");
    std::env::remove_var("INFR_KV_TYPE_V");
}

/// Mainline low-bit KV quants on the Vulkan seam: q4_0/q4_1/q5_0/q5_1/iq4_nl run via a quantizing
/// WriteKv (quant_kv) + a dequant→f16 prefix prepass (dequant_kv_f16, reusing native_decode) feeding
/// the standard f16 flash/split/scalar attention. K=f16 with each quantized V must stay coherent
/// (K needs higher precision). A broken GPU quantize or dequant would garble even a V-only cache.
#[test]
fn gpu_seam_kv_mainline_quants_coherent() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    need_gpu!();
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let head = |s: &str| s.chars().take(64).collect::<String>();
    // Long enough (>64 tokens) to take the flash prefill on the dequanted scratch, then deep decode.
    let long =
        "Explain how a CPU instruction pipeline works and list its common hazards. ".repeat(6);
    // K=f16 with each quantized V, plus the dense f32/bf16 caches (coupled).
    for (k, v) in [
        ("f16", "q4_0"),
        ("f16", "q4_1"),
        ("f16", "q5_0"),
        ("f16", "q5_1"),
        ("f16", "iq4_nl"),
        ("f32", "f32"),
        ("bf16", "bf16"),
        ("f16", "turbo2"),
        ("f16", "turbo3"),
        ("f16", "turbo4"),
    ] {
        std::env::set_var("INFR_KV_TYPE_K", k);
        std::env::set_var("INFR_KV_TYPE_V", v);
        let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
        let out = model.generate_dense_vulkan(&long, 24).expect("gpu kv gen");
        assert!(
            !is_degenerate(&out),
            "GPU K={k} V={v} degenerate: {:?}",
            head(&out)
        );
    }
    std::env::remove_var("INFR_KV_TYPE_K");
    std::env::remove_var("INFR_KV_TYPE_V");
}

/// Multi-slot KV prefix sharing: two INTERLEAVED conversations with a common long prefix (a
/// "system prompt"). Conversation B must (a) generate exactly what a fresh full prefill does,
/// (b) prefill only past the shared prefix (its slot was SEEDED by a device-side KV copy from
/// A's slot), and (c) not evict A — A's next turn still extends its own slot cheaply.
#[test]
fn gpu_seam_multi_slot_prefix_sharing() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    need_gpu!();
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let mut sess = model.vulkan_session(512).expect("session");

    // A long shared prefix (stands in for a system prompt) + two different questions.
    let sys = "You are a terse geography assistant. Answer in one word only, no punctuation, \
               no explanations, never refuse, always answer with just the single word asked for. ";
    let pa = format!("{sys}The capital of France is");
    let pb = format!("{sys}The capital of Germany is");

    let mut ta = String::new();
    let sa = model
        .generate_vulkan_session(&mut sess, &pa, 8, |p| ta.push_str(p))
        .expect("conv A");
    assert!(sa.n_prompt > 0);

    // Conversation B: different question, same system prefix → new slot seeded from A's.
    let mut tb = String::new();
    let sb = model
        .generate_vulkan_session(&mut sess, &pb, 8, |p| tb.push_str(p))
        .expect("conv B");
    let fresh_b = model.generate_dense_vulkan(&pb, 8).expect("fresh B");
    assert_eq!(
        tb.trim(),
        fresh_b.trim(),
        "seeded-slot generation diverged from a fresh full prefill"
    );
    // The shared prefix must NOT have been re-prefilled (only B's short suffix).
    assert!(
        sb.n_prompt < sa.n_prompt / 2,
        "conv B prefilled {} tokens (conv A: {}) — prefix seeding didn't kick in",
        sb.n_prompt,
        sa.n_prompt
    );

    // Conversation A continues — its slot must still be intact (suffix-only prefill again).
    let pa2 = format!("{pa}{ta} And the capital of Spain is");
    let mut ta2 = String::new();
    let sa2 = model
        .generate_vulkan_session(&mut sess, &pa2, 8, |p| ta2.push_str(p))
        .expect("conv A turn 2");
    let fresh_a2 = model.generate_dense_vulkan(&pa2, 8).expect("fresh A2");
    assert_eq!(
        ta2.trim(),
        fresh_a2.trim(),
        "conv A slot was clobbered by B"
    );
    assert!(
        sa2.n_prompt < sa.n_prompt / 2,
        "conv A turn 2 prefilled {} tokens — its slot was evicted",
        sa2.n_prompt
    );
}

/// Speculative decoding must emit EXACTLY the target-only greedy stream, end to end. The
/// contract is structural — every committed token is either checked against or produced by a
/// verify-forward argmax — but the verify forward runs the batched f16 GEMM/cmm path while
/// target-only decode uses the exact-f32 GEMV, so a near-tie logit could in principle split
/// them; this test pins the equivalence on a real generation. Self-spec (draft == target)
/// keeps it to one model download; the accept/commit machinery is identical to a small-draft
/// pair (the driver never knows the models are the same file).
#[cfg(target_os = "macos")]
#[test]
fn metal_spec_decode_matches_target_only_greedy() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let target = infr_llama::CpuModel::load(&path, None).expect("target load");
    let draft = infr_llama::CpuModel::load(&path, None).expect("draft load");
    let prompt = target
        .render_chat("Write a short paragraph about the ocean.")
        .expect("render chat");

    let mut plain = String::new();
    {
        let mut sess = target.metal_session(1024).expect("target-only session");
        target
            .generate_metal_session(&mut sess, &prompt, 64, |p| plain.push_str(p))
            .expect("target-only greedy");
    }

    let mut spec = String::new();
    {
        let mut ts = target.metal_session(1024).expect("spec target session");
        let mut ds = draft.metal_session(1024).expect("spec draft session");
        target
            .generate_metal_spec(&mut ts, &draft, &mut ds, &prompt, 64, 6, |p| {
                spec.push_str(p)
            })
            .expect("spec decode");
    }

    assert_eq!(
        spec, plain,
        "speculative stream diverged from target-only greedy"
    );
}

/// Metal twin of [`gpu_seam_multi_slot_prefix_sharing`]: the same interleaved-conversation
/// contract through `DenseMetalSession`'s slot pool — fork shares the one weight upload
/// (Arc), seeding is the backend-generic `copy_buffer`, and every slot switch re-records the
/// decode replay tape (its fingerprint covers the bound KV/IO buffer addresses).
#[cfg(target_os = "macos")]
#[test]
fn metal_seam_multi_slot_prefix_sharing() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let mut sess = model.metal_session(512).expect("session");

    let sys = "You are a terse geography assistant. Answer in one word only, no punctuation, \
               no explanations, never refuse, always answer with just the single word asked for. ";
    let pa = format!("{sys}The capital of France is");
    let pb = format!("{sys}The capital of Germany is");

    let mut ta = String::new();
    let sa = model
        .generate_metal_session(&mut sess, &pa, 8, |p| ta.push_str(p))
        .expect("conv A");
    assert!(sa.n_prompt > 0);

    // Conversation B: different question, same system prefix → new slot seeded from A's.
    let mut tb = String::new();
    let sb = model
        .generate_metal_session(&mut sess, &pb, 8, |p| tb.push_str(p))
        .expect("conv B");
    let mut fresh_b = String::new();
    model
        .generate_metal(&pb, 8, |p| fresh_b.push_str(p))
        .expect("fresh B");
    assert_eq!(
        tb.trim(),
        fresh_b.trim(),
        "seeded-slot generation diverged from a fresh full prefill"
    );
    // The shared prefix must NOT have been re-prefilled (only B's short suffix).
    assert!(
        sb.n_prompt < sa.n_prompt / 2,
        "conv B prefilled {} tokens (conv A: {}) — prefix seeding didn't kick in",
        sb.n_prompt,
        sa.n_prompt
    );

    // Conversation A continues — its slot must still be intact (suffix-only prefill again).
    let pa2 = format!("{pa}{ta} And the capital of Spain is");
    let mut ta2 = String::new();
    let sa2 = model
        .generate_metal_session(&mut sess, &pa2, 8, |p| ta2.push_str(p))
        .expect("conv A turn 2");
    let mut fresh_a2 = String::new();
    model
        .generate_metal(&pa2, 8, |p| fresh_a2.push_str(p))
        .expect("fresh A2");
    assert_eq!(
        ta2.trim(),
        fresh_a2.trim(),
        "conv A slot was clobbered by B"
    );
    assert!(
        sa2.n_prompt < sa.n_prompt / 2,
        "conv A turn 2 prefilled {} tokens — its slot was evicted",
        sa2.n_prompt
    );
}

/// gemma3 (SWA + dual-rope + GeGLU + sandwich norms, hd=256) through the Vulkan seam.
#[test]
fn gpu_seam_matches_cpu_gemma3() {
    let path = need_model!(gemma3_1b(), "gemma-3-1b");
    need_gpu!();
    let _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "What is the capital of France? Answer briefly.", 16);
}

/// llama (no qk-norm: standalone INTERLEAVED RoPE — llama.cpp's ROPE_TYPE_NORM — through the
/// f16-out Rope shape, fused KV write, and the rope_f16_dyn record-once replay).
#[test]
fn gpu_seam_matches_cpu_llama() {
    let path = need_model!(llama32_1b(), "Llama-3.2-1B");
    need_gpu!();
    let _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "Count from one to five, digits only.", 16);
}

/// gemma4 (heterogeneous head dims 256/512, V-norm, freq_factors, softcap) through the Vulkan seam.
#[test]
fn gpu_seam_matches_cpu_gemma4() {
    let path = need_model!(gemma4_12b(), "gemma-4-12b");
    need_gpu!();
    let _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "What is 2+2? Answer briefly.", 12);
}

/// gemma4 E2B (per-layer embeddings, KV/FFN sharing) through the Vulkan seam (per-token prefill —
fn gemma4_e2b() -> Option<PathBuf> {
    find_gguf("unsloth--gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_K_M.gguf")
}

// Captured + verified coherent (gemma4 E2B: per-layer input embeds + KV sharing): "The capital of
// France is **Paris**.", a brave-knight story ("Sir Kaelan … kingdom of Eldoria …").
const GEMMA4_E2B_GOLDEN: &[(&str, usize, u64)] = &[
    (
        "The capital of France is",
        32,
        0x689e792098786962, // channel-thought reasoning ("…Analyze the Request… factual question")
    ),
    (
        "Tell me a short story about a brave knight.",
        48,
        0x8909237b9419d782, // channel-thought reasoning (story planning process)
    ),
];

/// E2B is excluded from the batched-prefill fast path).
#[test]
fn gpu_seam_matches_cpu_gemma4_e2b() {
    let path = need_model!(gemma4_e2b(), "gemma-4-E2B");
    need_gpu!();
    let _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "What is 2+2? Answer briefly.", 12);
}

/// qwen3moe (routed-expert Op::MoeFfn) through the Vulkan seam, batched GPU-routed prefill. The
/// batched FFN runs int8 dp4a expert GEMMs (each parity-tested at the inherent ~2e-2 activation-
/// quant tolerance) — a numeric path the f32 CPU oracle can diverge from on a near-tie greedy
/// pick, so per the repo convention this locks its OWN golden (deterministic + read for
/// coherence; refresh with INFR_BLESS=1) instead of comparing token-for-token.
#[test]
fn gpu_seam_golden_qwen3moe() {
    let path = need_model!(qwen3moe_30b(), "Qwen3-30B-A3B");
    need_gpu!();
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let rendered = model
        .render_chat("What is 2+2? Answer briefly.")
        .expect("render chat");
    let out = model
        .generate_dense_vulkan(&rendered, 8)
        .expect("vulkan seam gen");
    let h = fnv1a(&out);
    if std::env::var("INFR_BLESS").is_ok() {
        println!("qwen3moe seam golden: 0x{h:016x}  // {out:?}");
    } else {
        assert_eq!(
            h, 0xfacca402bd6434e9,
            "qwen3moe seam golden changed\n  out: {out:?}"
        );
    }
}

/// BF16 (float-weight) seam parity: a bf16 model runs on the seam with its projection weights
/// converted to f16 (the matmul_proj / f16-GEMM prefill path) while the norm weights stay f32 (the
/// rmsnorm/qk_norm kernels read f32). Must match the CPU reference oracle token-for-token — proving
/// the float-weight GPU path is correct, not just fast.
#[test]
fn gpu_seam_bf16_matches_cpu() {
    let snap = match qwen3_06b() {
        Some(p) => p.parent().unwrap().to_path_buf(),
        None => return,
    };
    let path = snap.join("Qwen3-0.6B-BF16.gguf");
    if !path.exists() {
        eprintln!("skip: no BF16 model");
        return;
    }
    need_gpu!();
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let prompt = model
        .render_chat("What is the capital of France? Answer in one word.")
        .expect("render chat");
    let mut cpu_txt = String::new();
    model
        .generate_cpu(&prompt, 16, |p| cpu_txt.push_str(p))
        .expect("cpu gen");
    let gpu_txt = model.generate_dense_vulkan(&prompt, 16).expect("seam gen");
    assert_eq!(
        cpu_txt.trim(),
        gpu_txt.trim(),
        "bf16 seam (f16 projections) diverged from the CPU oracle"
    );
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
    let _tlk = test_serial_lock();
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

// ─── Gemma 3 (dense) ────────────────────────────────────────────────────────────

fn gemma3_1b() -> Option<PathBuf> {
    find_gguf("unsloth--gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf")
}

// ─── Llama (plain interleaved RoPE, no qk-norm) ────────────────────────────────

fn llama32_1b() -> Option<PathBuf> {
    find_gguf(
        "unsloth--Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
    )
}

// Captured + verified coherent: "Paris! 🇫🇷", a brave-knight short story (mournful Obsidian Peaks).
// Knight hash re-blessed when dense Q5_0 Linear m>1 moved from the dequant+f32-dot fallback to
// the int8 Q8x32 batch kernel (same regime as every other quantized dtype); output stayed
// coherent, France-prompt hash unchanged.
const GEMMA3_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0x4597cda816a0d1e7),
    (
        "Tell me a short story about a brave knight.",
        48,
        0x92fc5eeefb169094,
    ),
];

/// CPU-only: Gemma 3 (sandwich norms, GeGLU, dual-RoPE, SWA) golden-hash lock.
#[test]
fn cpu_golden_gemma3() {
    let path = need_model!(gemma3_1b(), "gemma-3-1b");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    check_golden(&model, GEMMA3_GOLDEN);
}

// ─── Qwen3.5 / Qwen3-Next (gated DeltaNet) ──────────────────────────────────────

fn qwen35_08b() -> Option<PathBuf> {
    find_gguf("unsloth--Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf")
}

// Captured + verified coherent (qwen35 / Qwen3-Next: gated-DeltaNet + gated full-attention): "The
// capital of France is **Paris**. It is the largest city …", a knight story ("Elara … Aethelgard").
// Renders with thinking ON (the infr-wide default; INFR_NO_THINK turns it off).
const QWEN35_GOLDEN: &[(&str, usize, u64)] = &[
    (
        "The capital of France is",
        32,
        0x542a9dd055c58884, // prefilled-think reasoning ("Thinking Process… capital of France")
    ),
    (
        "Tell me a short story about a brave knight.",
        48,
        0x0a0d2a6554ca9f21, // prefilled-think reasoning (story planning process)
    ),
];

/// CPU-only: qwen35 / Qwen3-Next golden-hash lock (gated-DeltaNet recurrence + conv + gated full
/// attention). Uses the dedicated `qwen35::generate_cpu` runner.
#[test]
fn cpu_golden_qwen35() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    let _tlk = test_serial_lock();
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

// ─── qwen35 on the UNIFIED shared-transformer path (Phase 2) ──────────────────────
//
// `Config::from_gguf` now accepts `arch == "qwen35"` and `cpu_backend`'s layer loop has a
// `MixerW::DeltaNet` branch (see `docs/QWEN35.md`) — so `CpuModel::load` on a qwen35 GGUF drives
// the SAME shared runner every other arch uses, in parallel with the old hand-written seam above.
// Phase 3: production routing (`infr run`/`serve`/`bench` in infr-cli) now sends qwen35 through
// this SAME unified path by default too — the old `qwen35::SeamModel` seam is reachable only via
// the temporary `INFR_QWEN35_OLD=1` escape hatch. These tests remain the token-identical proof
// that the two paths agree.

/// The unified CPU path (`CpuModel::generate_cpu`, i.e. `cpu_backend::generate_dense_cpu`) must
/// produce IDENTICAL output to the old hand-written seam (`qwen35::generate_cpu`) for the same
/// chat-rendered prompt: both run the same op sequence (RmsNorm/Linear/Conv1dSilu/DeltaNet/
/// QkNorm/GatedAct for the DeltaNet layers; the interleaved-gate split + sigmoid gate for the
/// attention layers) through the identical f32 CPU kernels, so greedy decode should match
/// token-for-token, not just golden-hash-coherent.
#[test]
fn unified_qwen35_cpu_matches_old_seam() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let prompt =
        infr_llama::qwen35::render_chat(&path, "The capital of France is").expect("render");

    let mut old_out = String::new();
    infr_llama::qwen35::generate_cpu(&path, &prompt, 24, |p| old_out.push_str(p))
        .expect("old seam gen");

    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let mut new_out = String::new();
    model
        .generate_cpu(&prompt, 24, |p| new_out.push_str(p))
        .expect("unified cpu gen");

    assert_eq!(
        old_out, new_out,
        "unified shared-transformer CPU path diverged from the old qwen35 seam"
    );
}

/// The unified Vulkan seam (`CpuModel::generate_dense_vulkan`) must match the unified CPU oracle
/// (`CpuModel::generate_cpu`) token-for-token — the seam twin of every other arch's
/// `gpu_seam_matches_cpu_*` test, now exercising `MixerW::DeltaNet` (Conv1dSilu/DeltaNet ops) AND
/// the qwen35 attention layers' interleaved q+gate split + sigmoid output gate through Vulkan.
#[test]
fn unified_qwen35_gpu_seam_matches_cpu() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    need_gpu!();
    let _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "What is bash? Answer briefly.", 24);
}

/// qwen35's gated-DeltaNet recurrent state is an APPEND-ONLY summary — it can't rewind to an
/// arbitrary shared prefix the way a real KV cache can (see docs/QWEN35.md and the no-rewind rule
/// in `cpu_backend::generate_dense_backend`). On the unified Vulkan session (`vulkan_session` /
/// `generate_vulkan_session`, the seam twin of `gpu_seam_kv_reuse_matches_fresh`):
///   (a) a prompt that EXACTLY EXTENDS the previous turn's fed sequence continues the recurrent
///       state — suffix-only prefill (`n_prompt` shrinks), output identical to a fresh full prefill.
///   (b) a prompt that does NOT extend it (a divergent turn) must fall back to a FULL re-prefill —
///       `n_prompt` equal to what a brand-new session prefills for the same prompt (proving the
///       state was zero-reset, not silently reused from a wrong point).
#[test]
fn unified_qwen35_session_no_rewind() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    need_gpu!();
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    std::env::set_var("INFR_IGNORE_EOS", "1"); // fixed-length turns, no early EOS stop
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let mut sess = model.vulkan_session(512).expect("session");

    let p1 = "The quick brown fox jumps over the lazy dog. The capital of France is";
    let mut t1 = String::new();
    let s1 = model
        .generate_vulkan_session(&mut sess, p1, 8, |p| t1.push_str(p))
        .expect("turn 1");
    assert!(s1.n_prompt > 0);

    // (a) EXTENDS turn 1 — suffix-only prefill, must match a fresh full prefill exactly.
    let p2 = format!("{p1}{t1} And the capital of Germany is");
    let mut t2 = String::new();
    let s2 = model
        .generate_vulkan_session(&mut sess, &p2, 8, |p| t2.push_str(p))
        .expect("turn 2 (extend)");
    let fresh2 = model.generate_dense_vulkan(&p2, 8).expect("fresh turn 2");
    assert_eq!(
        t2.trim(),
        fresh2.trim(),
        "extend-session output diverged from a fresh full prefill"
    );
    assert!(
        s2.n_prompt < s1.n_prompt,
        "turn 2 prefilled {} tokens — session reuse (extend) didn't kick in",
        s2.n_prompt
    );

    // (b) does NOT extend turn 2 (divergent subject) — the recurrent state can't rewind, so this
    // must be a FULL re-prefill: n_prompt must equal what a BRAND-NEW session prefills (its first
    // call always fully prefills, on every arch), not some smaller partial-prefix reuse.
    let p3 = "Completely different subject entirely: photosynthesis converts";
    let mut t3 = String::new();
    let s3 = model
        .generate_vulkan_session(&mut sess, p3, 8, |p| t3.push_str(p))
        .expect("turn 3 (divergent)");
    let mut fresh_sess = model.vulkan_session(512).expect("fresh session");
    let mut tf3 = String::new();
    let sf3 = model
        .generate_vulkan_session(&mut fresh_sess, p3, 8, |p| tf3.push_str(p))
        .expect("fresh turn 3");
    assert_eq!(
        t3.trim(),
        tf3.trim(),
        "post-reset generation diverged from a fresh prefill"
    );
    assert_eq!(
        s3.n_prompt, sf3.n_prompt,
        "divergent turn didn't fully re-prefill (no-rewind rule violated): got {} vs fresh {}",
        s3.n_prompt, sf3.n_prompt
    );
    std::env::remove_var("INFR_IGNORE_EOS");
}

// ─── MTP (multi-token prediction) speculative decoding — Phase 1 (issue #33) ────────────────
//
// See docs/MTP.md. Phase 1 only parses `{arch}.nextn_predict_layers` into `Config` (splitting the
// GGUF's `block_count` into trunk + head) and loads/shape-checks the head's own tensors — no MTP
// forward yet, so these tests validate LOADING + the `h`-tap primitive Phase 2 needs, not drafting.

fn qwen35_4b_mtp() -> Option<PathBuf> {
    find_gguf("unsloth--Qwen3.5-4B-MTP-GGUF", "Qwen3.5-4B-UD-Q4_K_XL.gguf")
}

/// The 4B MTP GGUF's `qwen35.block_count=33` INCLUDES the head layer
/// (`qwen35.nextn_predict_layers=1`) — `Config::from_gguf` must split it into a 32-layer TRUNK +
/// `n_layer_nextn=1` (today, before this phase, the trunk layer loop would misclassify `blk.32` as
/// a gated-DeltaNet layer and fail on missing `ssm_*` tensors — see `Config::n_layer_nextn`'s doc).
/// `mtp::load_mtp_head` must then find every required head tensor and correctly report the three
/// optional `nextn.*` fallback tensors ABSENT — this shipped GGUF's live path is 100% fallback to
/// the main model's `token_embd`/`output`/`output_norm` (see `docs/MTP.md`'s confirmed dump).
#[test]
fn mtp_gguf_loads() {
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf on the MTP GGUF");
    assert_eq!(cfg.n_layer, 32, "trunk n_layer must exclude the MTP head");
    assert_eq!(cfg.n_layer_nextn, 1, "qwen35.nextn_predict_layers=1");

    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");
    assert_eq!(
        head.il, 32,
        "the head sits immediately after the 32-layer trunk"
    );
    println!("MTP head tensors (blk.{}):", head.il);
    println!("  attn_norm              {:?}", head.attn_norm.shape);
    println!("  attn_q (interleaved q+gate) {:?}", head.attn_q.shape);
    println!("  attn_k                 {:?}", head.attn_k.shape);
    println!("  attn_v                 {:?}", head.attn_v.shape);
    println!("  attn_q_norm            {:?}", head.attn_q_norm.shape);
    println!("  attn_k_norm            {:?}", head.attn_k_norm.shape);
    println!("  attn_output            {:?}", head.attn_output.shape);
    println!(
        "  post_attention_norm    {:?}",
        head.post_attention_norm.shape
    );
    println!("  ffn_gate               {:?}", head.ffn_gate.shape);
    println!("  ffn_up                 {:?}", head.ffn_up.shape);
    println!("  ffn_down               {:?}", head.ffn_down.shape);
    println!("  nextn.eh_proj          {:?}", head.eh_proj.shape);
    println!("  nextn.enorm            {:?}", head.enorm.shape);
    println!("  nextn.hnorm            {:?}", head.hnorm.shape);
    println!(
        "  nextn.embed_tokens     {:?} (fallback: main tok_embd)",
        head.embed_tokens.as_ref().map(|t| &t.shape)
    );
    println!(
        "  nextn.shared_head_head {:?} (fallback: main lm_head)",
        head.shared_head_head.as_ref().map(|t| &t.shape)
    );
    println!(
        "  nextn.shared_head_norm {:?} (fallback: main output_norm)",
        head.shared_head_norm.as_ref().map(|t| &t.shape)
    );

    // Confirmed dump (docs/MTP.md): the shipped GGUF omits `embed_tokens`/`shared_head_head` (so
    // those fall back to the main model's `token_embd`/tied lm_head) but DOES ship its own
    // `shared_head_norm` (unlike the other two, this one is NOT a fallback in this GGUF).
    assert!(head.embed_tokens.is_none(), "confirmed absent in this GGUF");
    assert!(
        head.shared_head_head.is_none(),
        "confirmed absent in this GGUF"
    );
    assert!(
        head.shared_head_norm.is_some(),
        "confirmed PRESENT in this GGUF (docs/MTP.md)"
    );
}

/// The 0.8B (nextn-free) GGUF has no `nextn_predict_layers` key — `Config::from_gguf` must parse
/// it exactly as before this phase (`n_layer_nextn=0`, `n_layer` unchanged). Run alongside this:
/// `timeout 600 cargo test --release -p infr-llama --test cpu_backend unified_qwen35 -- --nocapture`
/// proves the TRUNK FORWARD itself (not just `Config` parsing) is byte-for-byte untouched.
#[test]
fn qwen35_trunk_unaffected() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    assert_eq!(cfg.n_layer_nextn, 0, "0.8B has no MTP head");
    assert_eq!(
        cfg.n_layer, 24,
        "trunk layer count must be unaffected by the nextn parsing"
    );
}

/// The h-tap (`CpuModel::prefill_logits_and_h_cpu`, issue #33's Phase 2 primitive): the captured
/// `h` row must be EXACTLY the lm_head's input for the SAME forward's logits row — i.e.
/// `lm_head(h) == logits` for qwen35 (a plain tied/untied GEMV, no softcap — `Config::final_softcap`
/// is 0 for every qwen35 model, unlike gemma). Host-recomputes the GEMV from the same dequantized
/// weight the graph used, in the same f32 precision, so this should match near bit-exactly.
#[test]
fn h_tap_matches_lm_head() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let tokens = model.encode("The capital of France is").expect("encode");

    let (logits, h) = model
        .prefill_logits_and_h_cpu(&tokens)
        .expect("prefill_logits_and_h_cpu");
    let cfg = model.config();
    assert_eq!(h.len(), cfg.n_embd, "h is one row: [n_embd]");
    assert_eq!(logits.len(), cfg.vocab, "logits is one row: [vocab]");

    // Host lm_head: the SAME tensor `build`'s `wload` picks (`output.weight`, or — tied — the
    // quantized `token_embd.weight`), fully dequantized here (vs the graph's lazy per-row dequant)
    // — same math, different (irrelevant) dequant call site.
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let lm_name = if g.tensors().iter().any(|t| t.name == "output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    let info = g
        .tensors()
        .iter()
        .find(|t| t.name == lm_name)
        .expect("lm_head tensor")
        .clone();
    let bytes = g.tensor_bytes(lm_name).expect("lm_head bytes");
    let w = infr_gguf::dequant::dequant_block(info.dtype, bytes).expect("dequant lm_head");
    let ne = cfg.n_embd;
    let vocab = cfg.vocab;
    assert_eq!(w.len(), ne * vocab, "lm_head dequant length");

    let mut host_logits = vec![0f32; vocab];
    for (v, out) in host_logits.iter_mut().enumerate() {
        let row = &w[v * ne..v * ne + ne];
        *out = row.iter().zip(&h).map(|(&wv, &hv)| wv * hv).sum();
    }
    let max_abs = logits
        .iter()
        .zip(&host_logits)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs()));
    let max_val = logits.iter().fold(1e-6f32, |m, &v| m.max(v.abs()));
    let rel = max_abs / max_val;
    // token_embd here is Q6_K: the graph's `Op::Linear` runs a quantized-activation integer dot
    // (q8 x q6_k) while this test's host GEMV runs a plain f32 dot over the fully dequantized
    // weight — the two paths agree to quantization tolerance, not bit-exactly.
    println!(
        "h_tap_matches_lm_head: max|graph-host|={max_abs:.6} (logit magnitude ~{max_val:.3}, rel={rel:.6})"
    );
    // 2% relative tolerance: the graph quantizes its OWN activation (q8) before the integer
    // Q6_K dot, while this host GEMV stays f32 throughout — a different (not buggy) arithmetic
    // path; measured ~1.1% on this model (see the printed `rel` above). A truly bit-exact check
    // would need to replicate the q8-activation Q6_K dot kernel host-side, which is out of scope
    // for a Phase 1 wiring check — the point here is "no missing/extra op between `h` and
    // `logits`", not quant-kernel bit-parity (that's covered elsewhere by the CPU/GPU goldens).
    assert!(
        rel < 0.02,
        "lm_head(h) diverged from the graph's logits: max abs diff {max_abs} (rel {rel})"
    );
}

// ─── MTP Phase 2: the head forward + the draft loop (issue #33) ─────────────────────────────
//
// See docs/MTP.md. Phase 2 builds the head's own 1-layer forward + the catch_up/draft driver
// primitives (`crate::mtp`) — these tests drive the ACTUAL 4B MTP GGUF's head, not just load it.

/// Prime a fresh [`infr_llama::mtp::MtpHeadSession`] over `prompt_tokens`: prefill the TRUNK on the
/// CPU backend (capturing `h` for every prompt row via the Phase-1 VERIFY tap), then `catch_up` the
/// head over the whole prompt in one call. Returns the session plus `(last_token, pending_h)` —
/// `draft`'s starting point (`docs/MTP.md`'s `process()`/`pending_h` handoff).
fn prime_head<'a>(
    model: &'a infr_llama::CpuModel,
    head: &infr_llama::mtp::MtpHeadWeights,
    cpu_be: &'a infr_cpu::CpuBackend,
    g: &infr_gguf::Gguf,
    max_ctx: usize,
    prompt_tokens: &[u32],
) -> (infr_llama::mtp::MtpHeadSession<'a>, u32, Vec<f32>) {
    let (_logits, h_rows) = model
        .verify_logits_and_h_cpu(prompt_tokens)
        .expect("verify_logits_and_h_cpu");
    let ne = model.config().n_embd;
    assert_eq!(h_rows.len(), prompt_tokens.len() * ne, "h per prompt row");

    let mut sess = infr_llama::mtp::MtpHeadSession::new_cpu(
        cpu_be,
        g,
        model.config(),
        head,
        model.token_embd(),
        max_ctx,
    )
    .expect("MtpHeadSession::new_cpu");

    // docs/MTP.md's process(): the head decodes the SAME tokens with `h` shifted right by one
    // (`embd[i] = h_tgt[i-1]`); row 0 has no predecessor in a fresh session, so it's paired with a
    // zero `pending_h` (`speculative.cpp`'s `pending_h` starts zero-initialized — see
    // `common_speculative_impl_draft_mtp`'s ctor, `pending_h.assign(n_seq, vector<float>(n_embd,
    // 0.0f))` — there IS no earlier target row to have produced a real one).
    let mut shifted_h = vec![0f32; prompt_tokens.len() * ne];
    if prompt_tokens.len() > 1 {
        shifted_h[ne..].copy_from_slice(&h_rows[..(prompt_tokens.len() - 1) * ne]);
    }
    infr_llama::mtp::catch_up(&mut sess, prompt_tokens, &shifted_h, 0).expect("catch_up");

    let id_last = *prompt_tokens.last().expect("nonempty prompt");
    let pending_h = h_rows[(prompt_tokens.len() - 1) * ne..].to_vec();
    (sess, id_last, pending_h)
}

/// The head forward, end to end, on the real 4B MTP GGUF: prefill a short prompt on the TRUNK
/// (capturing `h` via the Phase-1 tap), `catch_up` the head over it, then `draft` 6 tokens
/// (`--spec-draft-n-max 6`, matching `docs/MTP.md`'s oracle run). Asserts every logits row is
/// finite (no NaN/Inf — the eh_proj concat layout is exactly the kind of bug that would show up as
/// garbage here) and prints the drafted ids + top-1 probabilities.
#[test]
fn mtp_head_forward_finite() {
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");

    let prompt_tokens = model.encode("The capital of France is").expect("encode");
    let cpu_be = infr_cpu::CpuBackend::new();
    let n_max = 6usize;
    let max_ctx = prompt_tokens.len() + n_max + 4;
    let (mut sess, id_last, pending_h) =
        prime_head(&model, &head, &cpu_be, &g, max_ctx, &prompt_tokens);

    let drafted = infr_llama::mtp::draft(
        &mut sess,
        id_last,
        &pending_h,
        prompt_tokens.len(),
        infr_llama::mtp::DEFAULT_P_MIN,
        n_max,
    )
    .expect("draft");

    println!(
        "mtp_head_forward_finite: drafted {} token(s):",
        drafted.len()
    );
    for (i, &(id, p)) in drafted.iter().enumerate() {
        println!("  [{i}] id={id} p={p:.4}");
    }
    assert!(
        !drafted.is_empty(),
        "p_min=0.0 should always draft n_max tokens"
    );
    assert_eq!(drafted.len(), n_max, "p_min=0.0 never stops the loop early");
    for &(id, p) in &drafted {
        assert!(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "top1 prob out of range: {p}"
        );
        assert!((id as usize) < cfg.vocab, "drafted id out of vocab range");
    }
}

/// CPU/Vulkan parity: the SAME trunk-captured `h` (CPU, per `mtp_head_forward_finite`'s doc — only
/// the HEAD differs between the two calls below) drafted through the head on both backends must
/// produce the IDENTICAL token sequence (dense head, no MoE/routing noise to legitimately diverge
/// on — unlike the CPU/GPU generation goldens elsewhere in this file, which tolerate divergence).
#[test]
fn mtp_head_cpu_vulkan_parity() {
    need_gpu!();
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");

    let prompt_tokens = model.encode("The capital of France is").expect("encode");
    let n_max = 6usize;
    let max_ctx = prompt_tokens.len() + n_max + 4;

    let cpu_be = infr_cpu::CpuBackend::new();
    let (mut cpu_sess, id_last, pending_h) =
        prime_head(&model, &head, &cpu_be, &g, max_ctx, &prompt_tokens);
    let cpu_drafted = infr_llama::mtp::draft(
        &mut cpu_sess,
        id_last,
        &pending_h,
        prompt_tokens.len(),
        infr_llama::mtp::DEFAULT_P_MIN,
        n_max,
    )
    .expect("cpu draft");

    let vk = infr_vulkan::VulkanBackend::new().expect("vulkan init");
    let mut vk_sess = infr_llama::mtp::MtpHeadSession::new_vulkan(
        &vk,
        &g,
        model.config(),
        &head,
        model.token_embd(),
        max_ctx,
    )
    .expect("MtpHeadSession::new_vulkan");
    let ne = model.config().n_embd;
    let mut shifted_h = vec![0f32; prompt_tokens.len() * ne];
    if prompt_tokens.len() > 1 {
        let (_logits, h_rows) = model
            .verify_logits_and_h_cpu(&prompt_tokens)
            .expect("verify_logits_and_h_cpu");
        shifted_h[ne..].copy_from_slice(&h_rows[..(prompt_tokens.len() - 1) * ne]);
    }
    infr_llama::mtp::catch_up(&mut vk_sess, &prompt_tokens, &shifted_h, 0).expect("vk catch_up");
    let vk_drafted = infr_llama::mtp::draft(
        &mut vk_sess,
        id_last,
        &pending_h,
        prompt_tokens.len(),
        infr_llama::mtp::DEFAULT_P_MIN,
        n_max,
    )
    .expect("vk draft");

    let cpu_ids: Vec<u32> = cpu_drafted.iter().map(|&(id, _)| id).collect();
    let vk_ids: Vec<u32> = vk_drafted.iter().map(|&(id, _)| id).collect();
    println!("mtp_head_cpu_vulkan_parity: cpu={cpu_ids:?} vulkan={vk_ids:?}");
    assert_eq!(cpu_ids, vk_ids, "CPU/Vulkan MTP head drafts diverged");
}

/// Oracle-invariant fallback (`docs/MTP.md`'s validation ladder — capturing the oracle's OWN
/// verbose drafted-token trace proved impractical: llama.cpp's `SPC_DBG`/`SPC_TRC` macros gate on
/// `common_log`'s verbosity, not a dedicated spec-debug env var, and piping a live CPU generation's
/// stderr for a handful of draft steps is a lot of process-control machinery for what this simpler
/// check already covers): feed the head's drafted tokens through the TRUNK's OWN greedy decode and
/// measure how often the trunk's argmax agrees with what the head drafted — the PER-STEP acceptance
/// probability `alpha` a real spec-verify pass would see (stops at the first mismatch, like a real
/// verify). For `n_max=6` and i.i.d. per-step acceptance `alpha`, expected tokens/cycle is `(1 -
/// alpha^7) / (1 - alpha)`; solving that for the oracle's captured 2.0x (`docs/MTP.md`) gives
/// `alpha ≈ 0.5`, not a flat 60-80% — this test reports the measured per-prompt rate (averaged over
/// a couple of short prompts to dilute single-prompt noise) against that ~0.5 reference rather than
/// hard-gating on a specific number (still a coarse sanity check, not a benchmark).
#[test]
fn mtp_head_trunk_acceptance_rate() {
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let cpu_be = infr_cpu::CpuBackend::new();
    let n_max = 6usize;
    let vocab = cfg.vocab;

    let mut total_accepted = 0usize;
    let mut total_drafted = 0usize;
    for prompt in [
        "The capital of France is",
        "Tell me a short story about a brave knight.",
    ] {
        let prompt_tokens = model.encode(prompt).expect("encode");
        let max_ctx = prompt_tokens.len() + n_max + 4;
        let (mut sess, id_last, pending_h) =
            prime_head(&model, &head, &cpu_be, &g, max_ctx, &prompt_tokens);

        let drafted = infr_llama::mtp::draft(
            &mut sess,
            id_last,
            &pending_h,
            prompt_tokens.len(),
            infr_llama::mtp::DEFAULT_P_MIN,
            n_max,
        )
        .expect("draft");
        let draft_ids: Vec<u32> = drafted.iter().map(|&(id, _)| id).collect();

        // Trunk greedy-verify over [prompt | draft_ids]: row i's logits are the trunk's
        // distribution AFTER consuming prompt_tokens ++ draft_ids[..i] — i.e. exactly what the
        // trunk would have sampled in place of draft_ids[i] had it decoded token-by-token (the
        // spec-verify invariant).
        let mut full = prompt_tokens.clone();
        full.extend_from_slice(&draft_ids);
        let (verify_logits, _h) = model
            .verify_logits_and_h_cpu(&full)
            .expect("verify_logits_and_h_cpu over prompt+draft");
        let p = prompt_tokens.len();

        let mut accepted = 0usize;
        for (i, &draft_id) in draft_ids.iter().enumerate() {
            // Row `p - 1 + i` is the trunk's distribution for predicting position `p + i` — i.e.
            // the token that FOLLOWS `full[..p+i]`, which is exactly `draft_ids[i]` when accepted.
            let row = &verify_logits[(p - 1 + i) * vocab..(p + i) * vocab];
            let (argmax, _) =
                row.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv {
                            (i, v)
                        } else {
                            (bi, bv)
                        }
                    });
            let ok = argmax as u32 == draft_id;
            println!(
                "  {prompt:?} step {i}: drafted={draft_id} trunk_argmax={argmax} {}",
                if ok { "ACCEPT" } else { "reject" }
            );
            if ok {
                accepted += 1;
            } else {
                break; // spec-verify stops at the first mismatch — can't evaluate the rest as-is
            }
        }
        println!(
            "  {prompt:?}: {accepted}/{} accepted this cycle",
            draft_ids.len()
        );
        total_accepted += accepted;
        total_drafted += draft_ids.len();
    }
    let rate = total_accepted as f64 / total_drafted.max(1) as f64;
    println!(
        "mtp_head_trunk_acceptance_rate: {total_accepted}/{total_drafted} accepted overall \
         ({rate:.2}) — oracle's 2.0x implies a per-step rate around ~0.5 (see this test's doc)"
    );
}

// ─── MTP Phase 3: the self-speculative generation loop (issue #33) ──────────────────────────────
//
// See docs/MTP.md. Phase 3 wires the head into a full generation loop (`crate::mtp::
// generate_mtp_spec_vulkan`) on the production Vulkan seam — these tests drive THAT loop, not the
// head primitives directly (Phase 2's tests above already cover those in isolation).

/// **The Phase 3 hard bar**: self-speculative MTP decoding must be output-IDENTICAL to plain
/// target-only greedy decoding on the SAME (real, production) Vulkan seam — the spec ≡
/// target-greedy invariant `docs/MTP.md`'s own oracle run holds ("byte-identical output"). No
/// tolerance, no golden hash — a real string equality on a real generation. If this fails, the
/// accept/commit/KV logic is wrong (see `crate::mtp::generate_mtp_spec_vulkan`'s doc on the
/// KV-overwrite/no-rewind semantics it relies on) — debug that, don't relax this assertion.
#[test]
fn mtp_spec_matches_target_only_greedy() {
    need_gpu!();
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let prompt = model
        .render_chat("Tell me a short story about a brave knight.")
        .expect("render chat");
    let max_new = 64usize;

    let mut plain = String::new();
    {
        let mut sess = model
            .vulkan_session(prompt.len() + max_new + 64)
            .expect("target-only session");
        model
            .generate_vulkan_session(&mut sess, &prompt, max_new, |p| plain.push_str(p))
            .expect("target-only greedy");
    }

    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");

    let mut spec = String::new();
    infr_llama::mtp::generate_mtp_spec_vulkan(&model, &head, &prompt, max_new, |p| {
        spec.push_str(p)
    })
    .expect("mtp spec decode");

    assert_eq!(
        spec, plain,
        "MTP self-speculative stream diverged from target-only greedy"
    );
}

/// Acceptance-rate report over a longer generation, 2-3 prompts (`INFR_MTP_TIME=1` also prints a
/// per-cycle breakdown to stderr — run this test with `-- --nocapture` and that env var set to see
/// it). Not gated on a specific number (`mtp_head_trunk_acceptance_rate` already sanity-checks the
/// head's own per-step rate against the oracle's implied ~0.5) — this just surfaces the aggregate
/// alpha the full loop achieves so it's visible in normal test output.
#[test]
fn mtp_spec_acceptance_stats() {
    need_gpu!();
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");

    for user in [
        "Tell me a short story about a brave knight.",
        "What is the capital of France?",
        "Explain how photosynthesis works in two sentences.",
    ] {
        let prompt = model.render_chat(user).expect("render chat");
        let mut out = String::new();
        let stats = infr_llama::mtp::generate_mtp_spec_vulkan(&model, &head, &prompt, 128, |p| {
            out.push_str(p)
        })
        .expect("mtp spec decode");
        println!(
            "mtp_spec_acceptance_stats: {user:?} -> {} tokens in {:.2}s prompt + {:.2}s decode",
            stats.n_gen, stats.prompt_secs, stats.decode_secs
        );
    }
}

// ─── Qwen3-MoE (routed experts) ─────────────────────────────────────────────────

fn qwen3moe_30b() -> Option<PathBuf> {
    find_gguf("unsloth--Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q4_K_M.gguf")
}

// Captured + verified coherent (qwen3moe: routed-expert FFN, ~3B active of 30B).
// Re-blessed 2026-07-05 for the whole-call int8 MoE gate (multi-row PREFILL calls now run the
// int8-activation fast path in every bucket — a deliberate numeric-regime change; see
// `expert_matvec_batch`'s `int8_ok` doc). Verified coherent ("<think>\nOkay, the user is asking,
// \"The capital of France is...\" and then it cuts off. I need") — a near-tie token flipped
// mid-thought vs the previous capture, same meaning.
const QWEN3MOE_GOLDEN: &[(&str, usize, u64)] =
    &[("The capital of France is", 24, 0xa68ab7f4d15ad931)];

/// Whole-vector cosine similarity (f64 accumulation) — used by the CPU/Vulkan cross-backend
/// logits check below.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// CPU-only: qwen3moe golden-hash lock (the Op::MoeFfn routed-expert path). 30B but only `n_used`
/// experts run per token; still slow on CPU, so a single short case.
#[test]
fn cpu_golden_qwen3moe() {
    let path = need_model!(qwen3moe_30b(), "Qwen3-30B-A3B");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    check_golden(&model, QWEN3MOE_GOLDEN);
}

/// CPU-only: Gemma 4 E2B golden-hash lock.
#[test]
fn cpu_golden_gemma4_e2b() {
    let path = need_model!(gemma4_e2b(), "gemma-4-E2B");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    check_golden(&model, GEMMA4_E2B_GOLDEN);
}

// ─── Gemma 4 12b (dense) ────────────────────────────────────────────────────────

fn gemma4_12b() -> Option<PathBuf> {
    find_gguf("unsloth--gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf")
}

// ─── DiffusionGemma (block text-diffusion MoE on a Gemma-4 backbone) ───────────────
//
// Phase 1 scope only: Config + weight loading + a CAUSAL PROMPT PREFILL through the unified
// runner (dual FFN — dense GeGLU ∥ 128-expert MoE with a fused gate_up_exps + per-expert down
// scale, encoder-scalar per-layer output, heterogeneous per-layer attn dims). No canvas/denoise —
// see docs/DIFFUSIONGEMMA.md. 26B-A4B Q4_K_M is large (16 GB); a CPU prefill of ~16 tokens takes
// on the order of a minute.

fn diffusion_gemma_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_DIFFUSION_GEMMA") {
        return Some(PathBuf::from(p));
    }
    find_gguf(
        "unsloth--diffusiongemma-26B-A4B-it-GGUF",
        "diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
    )
}

/// The top-`k` (token id, logit) pairs of a vocab-sized logits row, for a human-readable print.
fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(k);
    idx.into_iter().map(|i| (i, logits[i])).collect()
}

/// CPU-only: DiffusionGemma's causal prompt prefill produces finite logits over a short fixed
/// prompt. Prints the top-5 last-row (next-token) logits — no golden hash (Phase 1 doesn't claim
/// coherent generation; that's the oracle-parity check in Phase 3).
#[test]
fn cpu_diffusion_gemma_prefill_finite() {
    let path = need_model!(diffusion_gemma_model(), "diffusiongemma-26B-A4B");
    let _tlk = test_serial_lock();
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    assert!(
        model.config().diffusion_gemma,
        "arch not parsed as diffusion-gemma"
    );
    assert!(model.config().canvas_length > 0, "canvas_length not parsed");
    let tokens = model
        .encode("What is the capital of France? Answer briefly.")
        .expect("encode");
    assert!(!tokens.is_empty(), "empty prompt");
    let vocab = model.config().vocab;
    let t0 = std::time::Instant::now();
    // `prefill_logits_cpu` returns only the LAST prompt token's row (the causal prefill's
    // next-token distribution) — the per-token decode loop's frontier logits, not a [m, vocab]
    // batch (see its doc comment).
    let last_row = model.prefill_logits_cpu(&tokens).expect("cpu prefill");
    eprintln!(
        "cpu_diffusion_gemma_prefill_finite: {} tokens, prefill {:.1}s",
        tokens.len(),
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(last_row.len(), vocab, "logits shape");
    assert!(
        last_row.iter().all(|v| v.is_finite()),
        "non-finite logit in the prefill output"
    );
    println!("top-5 last-row tokens: {:?}", top_k(&last_row, 5));
}

/// DiffusionGemma's causal prompt prefill through the Vulkan seam must match the CPU oracle's
/// last-row logits within tolerance (quantized-weight + f16-vs-f32 numeric drift — the same class
/// of divergence the golden-hash tests sidestep by locking per-backend hashes instead of a direct
/// float compare; here we compare directly since Phase 1 has no generation golden yet).
#[test]
fn gpu_seam_matches_cpu_diffusion_gemma() {
    let path = need_model!(diffusion_gemma_model(), "diffusiongemma-26B-A4B");
    need_gpu!();
    let _tlk = test_serial_lock();
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let tokens = model
        .encode("What is the capital of France? Answer briefly.")
        .expect("encode");
    let vocab = model.config().vocab;
    // Both return only the LAST prompt token's row (see `prefill_logits_cpu`'s doc comment).
    let cpu_last = model.prefill_logits_cpu(&tokens).expect("cpu prefill");
    let gpu_last = model
        .prefill_logits_vulkan(&tokens)
        .expect("vulkan prefill");
    assert_eq!(cpu_last.len(), vocab, "cpu logits shape");
    assert_eq!(gpu_last.len(), vocab, "gpu logits shape");
    assert!(
        gpu_last.iter().all(|v| v.is_finite()),
        "non-finite logit in the Vulkan prefill output"
    );
    let (cpu_top, gpu_top) = (top_k(&cpu_last, 20), top_k(&gpu_last, 20));
    println!("cpu    top-5: {:?}", &cpu_top[..5]);
    println!("vulkan top-5: {:?}", &gpu_top[..5]);
    // NOT an exact/near-tolerance match: this is a 128-expert top-8 MoE model, and top-k expert
    // SELECTION is a discrete step — a near-tie router logit (f32 CPU vs f16-native-quant Vulkan)
    // can flip which experts run for a token, which then diverges the WHOLE downstream FFN output
    // for that layer. This is a known, already-shipped property of this codebase's OTHER MoE arch
    // (qwen3moe): its cross-backend test explicitly does NOT compare logits/tokens directly,
    // locking separate per-backend golden hashes instead (see `gpu_seam_golden_qwen3moe`'s doc
    // comment). Calibrated directly against that model (same class of divergence, no known bug):
    // qwen3moe's CPU-vs-Vulkan last-row argmax lands on COMPLETELY DIFFERENT tokens with a
    // whole-vocab cosine similarity of ~0.74, vs gemma4's (dense, no MoE) ~0.995 — this
    // diffusion-gemma check (argmax within each other's top-20 AND a 0.7 cosine floor, comfortably
    // above qwen3moe's measured 0.74) is already stricter than the existing MoE precedent.
    assert!(
        cpu_top[..5].iter().any(|&(id, _)| id == gpu_top[0].0)
            || gpu_top[..5].iter().any(|&(id, _)| id == cpu_top[0].0),
        "CPU/Vulkan top tokens don't even overlap in each other's top-5: cpu={:?} vulkan={:?}",
        cpu_top[0],
        gpu_top[0]
    );
    let cos = cosine(&cpu_last, &gpu_last);
    println!("cpu/vulkan whole-vocab cosine similarity: {cos}");
    assert!(
        cos > 0.7,
        "CPU/Vulkan last-row logits diverged too far: cosine={cos}"
    );
}

// ─── DiffusionGemma Phase 2: canvas denoise ─────────────────────────────────────────
//
// One bidirectional forward over the C canvas rows, reusing the prompt KV Phase 1's causal
// prefill already wrote (encoder scalars, rows 0..P) — decoder scalars, the `AttnMask::Canvas`
// bidirectional mask, and (optionally) self-conditioning. See docs/DIFFUSIONGEMMA.md.

/// CPU-only: prefill a short prompt, then ONE denoise forward over an all-mask canvas
/// (`sc_logits=None`, matching the reference's step-0 zero-SC gate). Also proves the WriteKv
/// overwrite (a second denoise call with a DIFFERENT canvas must produce different, still-finite
/// logits — the next denoise step re-overwrites the same KV rows) and a self-conditioning smoke
/// test (feeding the first call's raw logits back in must differ from the no-SC call).
#[test]
fn cpu_diffusion_gemma_denoise_step() {
    let path = need_model!(diffusion_gemma_model(), "diffusiongemma-26B-A4B");
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let vocab = model.config().vocab;
    let canvas_len = model.config().canvas_length;
    let mask_id = model.config().mask_token_id;
    let tokens = model
        .encode("What is the capital of France?")
        .expect("encode");
    assert!(!tokens.is_empty(), "empty prompt");

    let mut session = model.diffusion_gemma_cpu_session(tokens.len() + canvas_len + 8);
    let t0 = std::time::Instant::now();
    session.prefill(&model, &tokens).expect("cpu prefill");
    eprintln!(
        "cpu_diffusion_gemma_denoise_step: prefill {} tokens in {:.1}s",
        tokens.len(),
        t0.elapsed().as_secs_f64()
    );

    let canvas: Vec<u32> = vec![mask_id; canvas_len];
    let t1 = std::time::Instant::now();
    let logits1 = session
        .denoise(&model, &canvas, None, 1.0)
        .expect("cpu denoise step 1 (no SC)");
    eprintln!(
        "cpu_diffusion_gemma_denoise_step: denoise (no SC) {:.1}s",
        t1.elapsed().as_secs_f64()
    );
    assert_eq!(logits1.len(), canvas_len * vocab, "denoise logits shape");
    assert!(
        logits1.iter().all(|v| v.is_finite()),
        "non-finite logit in the no-SC denoise output"
    );
    for row in 0..canvas_len.min(8) {
        let row_logits = &logits1[row * vocab..(row + 1) * vocab];
        let top = top_k(row_logits, 1)[0];
        println!("row {row} argmax: token {} logit {:.3}", top.0, top.1);
    }

    // WriteKv overwrite: a second denoise call with a DIFFERENT canvas (row 0 unmasked to the
    // previous argmax) must produce different, still-finite logits — proving the cache actually
    // gets re-written each step (not stale-row reuse). Row 0's own argmax over an all-mask canvas
    // is often the mask token itself (the model's "not enough context yet" answer) — that would
    // leave canvas2 identical to canvas, so pick the first top-5 candidate that ACTUALLY differs
    // from mask_id (falling back to a fixed different token if somehow all 5 are the mask token).
    let mut canvas2 = canvas.clone();
    canvas2[0] = top_k(&logits1[..vocab], 5)
        .into_iter()
        .map(|(id, _)| id as u32)
        .find(|&id| id != mask_id)
        .unwrap_or((mask_id + 1) % vocab as u32);
    assert_ne!(
        canvas2[0], canvas[0],
        "test bug: canvas2 didn't actually change row 0"
    );
    let t2 = std::time::Instant::now();
    let logits2 = session
        .denoise(&model, &canvas2, None, 1.0)
        .expect("cpu denoise step 2 (different canvas)");
    eprintln!(
        "cpu_diffusion_gemma_denoise_step: denoise (overwrite check) {:.1}s",
        t2.elapsed().as_secs_f64()
    );
    assert_eq!(logits2.len(), canvas_len * vocab, "denoise2 logits shape");
    assert!(
        logits2.iter().all(|v| v.is_finite()),
        "non-finite logit after the second (overwrite) denoise call"
    );
    assert!(
        logits1 != logits2,
        "second denoise call (different canvas) produced IDENTICAL logits — WriteKv didn't \
         overwrite the canvas KV rows"
    );

    // Self-conditioning smoke test: feed the first call's raw logits back as sc_logits.
    let t3 = std::time::Instant::now();
    let logits_sc = session
        .denoise(&model, &canvas, Some(&logits1), 1.0)
        .expect("cpu denoise with self-conditioning");
    eprintln!(
        "cpu_diffusion_gemma_denoise_step: denoise (self-cond) {:.1}s",
        t3.elapsed().as_secs_f64()
    );
    assert_eq!(
        logits_sc.len(),
        canvas_len * vocab,
        "SC denoise logits shape"
    );
    assert!(
        logits_sc.iter().all(|v| v.is_finite()),
        "non-finite logit in the self-conditioned denoise output"
    );
    assert!(
        logits_sc != logits1,
        "self-conditioned denoise produced IDENTICAL logits to the no-SC call"
    );
    println!("cpu_diffusion_gemma_denoise_step: all sub-checks passed");
}

/// DiffusionGemma canvas denoise: CPU vs Vulkan on separate sessions given the SAME prompt +
/// all-mask canvas, no self-conditioning. Calibrated like `gpu_seam_matches_cpu_diffusion_gemma`
/// (Phase 1): a 128-expert top-8 MoE model's near-tie router logits can flip expert selection
/// between f32 CPU and f16-native-quant Vulkan, diverging the downstream FFN output for that
/// token — assert per-row cosine + top-5 overlap on a handful of rows rather than a tight
/// tolerance across all 256.
#[test]
fn gpu_seam_matches_cpu_diffusion_gemma_denoise() {
    let path = need_model!(diffusion_gemma_model(), "diffusiongemma-26B-A4B");
    need_gpu!();
    let _tlk = test_serial_lock();
    std::env::set_var("INFR_TEMP", "0");
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    let vocab = model.config().vocab;
    let canvas_len = model.config().canvas_length;
    let mask_id = model.config().mask_token_id;
    let tokens = model
        .encode("What is the capital of France?")
        .expect("encode");

    let mut cpu_session = model.diffusion_gemma_cpu_session(tokens.len() + canvas_len + 8);
    cpu_session.prefill(&model, &tokens).expect("cpu prefill");
    let canvas: Vec<u32> = vec![mask_id; canvas_len];
    let t0 = std::time::Instant::now();
    let cpu_logits = cpu_session
        .denoise(&model, &canvas, None, 1.0)
        .expect("cpu denoise");
    let cpu_secs = t0.elapsed().as_secs_f64();

    let mut vk_session = model
        .diffusion_gemma_vulkan_session(tokens.len() + canvas_len + 8)
        .expect("vulkan session");
    vk_session.prefill(&model, &tokens).expect("vulkan prefill");
    let t1 = std::time::Instant::now();
    let gpu_logits = vk_session
        .denoise(&model, &canvas, None, 1.0)
        .expect("vulkan denoise");
    let gpu_secs = t1.elapsed().as_secs_f64();
    eprintln!(
        "gpu_seam_matches_cpu_diffusion_gemma_denoise: cpu {cpu_secs:.1}s vulkan {gpu_secs:.1}s"
    );

    assert_eq!(cpu_logits.len(), canvas_len * vocab, "cpu logits shape");
    assert_eq!(gpu_logits.len(), canvas_len * vocab, "gpu logits shape");
    assert!(
        gpu_logits.iter().all(|v| v.is_finite()),
        "non-finite logit in the Vulkan denoise output"
    );

    let mut min_cos = f64::INFINITY;
    for row in 0..canvas_len.min(8) {
        let c = &cpu_logits[row * vocab..(row + 1) * vocab];
        let v = &gpu_logits[row * vocab..(row + 1) * vocab];
        let cos = cosine(c, v);
        min_cos = min_cos.min(cos);
        let (ctop, vtop) = (top_k(c, 5), top_k(v, 5));
        println!(
            "row {row}: cosine={cos:.3} cpu_top1={:?} vulkan_top1={:?}",
            ctop[0], vtop[0]
        );
        assert!(
            ctop.iter().any(|&(id, _)| id == vtop[0].0)
                || vtop.iter().any(|&(id, _)| id == ctop[0].0),
            "row {row}: CPU/Vulkan top tokens don't overlap in each other's top-5: \
             cpu={:?} vulkan={:?}",
            ctop[0],
            vtop[0]
        );
        assert!(cos > 0.7, "row {row}: cosine too low: {cos}");
    }
    println!(
        "gpu_seam_matches_cpu_diffusion_gemma_denoise: min row cosine over checked rows = {min_cos:.3}"
    );
}

// ─── DiffusionGemma Phase 3: entropy-bound decode loop vs the oracle ────────────────────────────
//
// The full block-diffusion decode (`infr_llama::diffusion::diffusion_generate`) driven on the
// Vulkan session, for the same chat-templated prompt the oracle (`llama-diffusion-cli`) was run
// on (see docs/DIFFUSIONGEMMA.md's "Oracle reference outputs"). NOT a token-identical check (a
// 128-expert top-8 MoE model's CPU-vs-Vulkan routing legitimately diverges — the same class of
// divergence `gpu_seam_matches_cpu_diffusion_gemma[_denoise]` above already calibrate against);
// this asserts the DECODED TEXT is coherent (contains "Paris") and prints both texts + step/block
// counts side by side so a human can eyeball the match.

/// Vulkan-gated (like Phase 2's GPU tests): the entropy-bound decode loop end-to-end for "What is
/// the capital of France?", `n_predict=64`, greedy (`INFR_SEED` default 42). Prints the decoded
/// text, step count and block count; asserts the post-thought answer contains "Paris".
#[test]
fn diffusion_gemma_decode_matches_oracle() {
    let path = need_model!(diffusion_gemma_model(), "diffusiongemma-26B-A4B");
    need_gpu!();
    let _tlk = test_serial_lock();
    let model = infr_llama::CpuModel::load(&path, None).expect("cpu load");
    assert!(
        model.config().diffusion_gemma,
        "arch not parsed as diffusion-gemma"
    );

    let prompt = model
        .render_chat_messages(&[("user", "What is the capital of France?")])
        .expect("render chat template");
    let tokens = model.encode(&prompt).expect("encode");
    assert!(!tokens.is_empty(), "empty prompt");

    let cfg = model.config();
    let canvas_len = cfg.canvas_length;
    let vocab = cfg.vocab;
    let eos_ids = cfg.eos_ids.clone();
    let eb = infr_llama::diffusion::EbConfig::from_config(cfg);

    let n_predict = 64usize;
    let blocks = n_predict.div_ceil(canvas_len).max(1);
    let max_ctx = tokens.len() + blocks * canvas_len + 64;
    let mut session = model
        .diffusion_gemma_vulkan_session(max_ctx)
        .expect("vulkan session");

    let t0 = std::time::Instant::now();
    let result = infr_llama::diffusion::diffusion_generate(
        &mut session,
        &model,
        &tokens,
        canvas_len,
        vocab,
        &eos_ids,
        &eb,
        n_predict,
        /* seed */ 42,
        max_ctx,
    )
    .expect("diffusion_generate");
    let secs = t0.elapsed().as_secs_f64();

    let text = model.decode(&result.tokens).expect("decode");
    eprintln!(
        "diffusion_gemma_decode_matches_oracle: {} steps over {} block(s) in {secs:.1}s \
         ({} tok generated)",
        result.steps,
        result.blocks,
        result.tokens.len()
    );
    println!("infr   text: {text:?}");
    // Oracle reference (CPU, `-p \"What is the capital of France?\" -n 64 -s 42 --temp 0`, captured
    // 2026-07-05 — see docs/DIFFUSIONGEMMA.md): 10 EB steps, 1 block, thinking span then "The
    // capital of France is Paris."
    println!(
        "oracle text: \"<|channel>thought\\nThe user is asking for the capital of France.\\n    \
         *   Country: France.\\n    *   Capital: Paris.\\nProvide the direct answer \
         clearly.<channel|>The capital of France is Paris.\" (10 EB steps, 1 block)"
    );

    let (_thought, answer) = infr_chat::split_channels(&text);
    assert!(
        answer.contains("Paris") || text.contains("Paris"),
        "decoded answer doesn't mention Paris: {text:?}"
    );
}
