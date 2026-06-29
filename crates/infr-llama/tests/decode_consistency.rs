//! GPU integration test (#[ignore], needs a Vulkan device + the gemma GGUF): does the incremental
//! DECODE path build the same KV/logits as a one-shot PREFILL of the identical tokens? Teacher-forced
//! (we feed a fixed token sequence), so prefill-all and prefill-1-then-decode-rest MUST agree if the
//! decode path is numerically equivalent. A growing divergence is the source of long-generation drift.
//!
//! Run: `INFR_TEST_MODEL=<gguf> cargo test --release -p infr-llama --test decode_consistency -- --ignored --nocapture`

use std::path::PathBuf;

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
        return PathBuf::from(p);
    }
    // default: the shared HF hub snapshot for unsloth/gemma-3-1b-it-GGUF Q4_K_M
    let hub = std::env::var("HOME").unwrap() + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--unsloth--gemma-3-1b-it-GGUF/snapshots");
    for e in std::fs::read_dir(&base).expect("snapshots dir") {
        let d = e.unwrap().path();
        let f = d.join("gemma-3-1b-it-Q4_K_M.gguf");
        if f.exists() {
            return f;
        }
    }
    panic!("gemma gguf not found; set INFR_TEST_MODEL");
}

fn argmax(v: &[f32]) -> (usize, f32) {
    let mut bi = 0;
    for i in 1..v.len() {
        if v[i] > v[bi] {
            bi = i;
        }
    }
    (bi, v[bi])
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

#[test]
#[ignore = "needs a Vulkan GPU + the gemma GGUF"]
fn decode_matches_prefill() {
    let m = model_path();
    let llama = infr_llama::Llama::load_opt(&m, None).expect("load");
    let vocab = llama.config().vocab;
    // A fixed, deterministic token sequence (teacher-forcing); content is irrelevant — we only test
    // that decode and prefill produce the same logits for the SAME tokens. Stay clear of specials.
    let n = 160usize;
    let toks: Vec<u32> = (0..n).map(|i| ((i * 977 + 1234) % (vocab - 300) + 200) as u32).collect();

    // Method A: one-shot prefill of all tokens.
    let mut kv_a = llama.new_kv(512).expect("kv a");
    let logits_a = llama.forward_resident_kv(&toks, &mut kv_a).expect("prefill all");

    // Method B: prefill the first token, then decode the rest one at a time (teacher-forced).
    let mut kv_b = llama.new_kv(512).expect("kv b");
    let _ = llama.forward_resident_kv(&toks[..1], &mut kv_b).expect("prefill 1");
    let mut logits_b = Vec::new();
    for &t in &toks[1..] {
        logits_b = llama.forward_resident_kv(&[t], &mut kv_b).expect("decode step");
    }

    let (ai, av) = argmax(&logits_a);
    let (bi, bv) = argmax(&logits_b);
    let diff = l2(&logits_a, &logits_b);
    let max_abs = logits_a
        .iter()
        .zip(&logits_b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!(
        "[consistency] n={n} prefill_argmax=({ai},{av:.3}) decode_argmax=({bi},{bv:.3}) \
         L2={diff:.4} max|Δ|={max_abs:.4}"
    );
    // Prefill and decode should be numerically near-identical (only kernel-order f16 noise).
    assert_eq!(ai, bi, "decode argmax diverged from prefill — decode-path bug");
    assert!(max_abs < 0.5, "decode logits drifted from prefill by {max_abs} (>0.5)");
}
