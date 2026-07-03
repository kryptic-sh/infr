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
const GEMMA3_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0x4597cda816a0d1e7),
    (
        "Tell me a short story about a brave knight.",
        48,
        0xe7c90b188b42cee0,
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
