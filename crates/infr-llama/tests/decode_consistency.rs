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
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Teacher-force infr through llama.cpp's coherent greedy trajectory (ids in INFR_TEST_CONT) and at
/// each step report (a) whether infr's argmax == llama's actual next token, (b) infr's argmax-logit
/// magnitude. High disagreement / flattening on a KNOWN-COHERENT path = a real per-step logit bug.
#[test]
#[ignore = "needs a Vulkan GPU + the gemma GGUF + INFR_TEST_CONT"]
fn agreement_with_llama() {
    let m = model_path();
    let llama = infr_llama::Llama::load_opt(&m, None).expect("load");
    let ids = |s: String| -> Vec<u32> {
        s.trim()
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect()
    };
    let (Ok(pids), Ok(cpath)) = (
        std::env::var("INFR_TEST_PROMPT"),
        std::env::var("INFR_TEST_CONT"),
    ) else {
        eprintln!("skip: set INFR_TEST_PROMPT (comma ids) + INFR_TEST_CONT (ids file)");
        return;
    };
    let prompt: Vec<u32> = ids(pids);
    let cont: Vec<u32> = ids(std::fs::read_to_string(cpath).unwrap());
    let mut kv = llama.new_kv(512).expect("kv");
    let mut logits = llama
        .forward_resident_kv(&prompt, &mut kv)
        .expect("prefill");
    let mut agree = 0usize;
    for (i, &t) in cont.iter().enumerate() {
        let (ai, av) = argmax(&logits);
        let ll = logits[t as usize];
        let ok = ai as u32 == t;
        if ok {
            agree += 1;
        }
        if i % 10 == 0 || !ok {
            println!(
                "[agree] step={i} infr_argmax=({ai},{av:.2}) llama_tok={t} (infr_logit={ll:.2}) {}",
                if ok { "ok" } else { "DISAGREE" }
            );
        }
        logits = llama.forward_resident_kv(&[t], &mut kv).expect("decode");
    }
    println!(
        "[agree] agreement {}/{} = {:.1}%",
        agree,
        cont.len(),
        100.0 * agree as f32 / cont.len() as f32
    );
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
    let toks: Vec<u32> = (0..n)
        .map(|i| ((i * 977 + 1234) % (vocab - 300) + 200) as u32)
        .collect();

    // Method A: one-shot prefill of all tokens.
    let mut kv_a = llama.new_kv(512).expect("kv a");
    let logits_a = llama
        .forward_resident_kv(&toks, &mut kv_a)
        .expect("prefill all");

    // Method B: prefill the first token, then decode the rest one at a time (teacher-forced).
    let mut kv_b = llama.new_kv(512).expect("kv b");
    let _ = llama
        .forward_resident_kv(&toks[..1], &mut kv_b)
        .expect("prefill 1");
    let mut logits_b = Vec::new();
    for &t in &toks[1..] {
        logits_b = llama
            .forward_resident_kv(&[t], &mut kv_b)
            .expect("decode step");
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
    println!(
        "[consistency] cross: prefill[{bi}]={:.4} decode[{ai}]={:.4}",
        logits_a[bi], logits_b[ai]
    );
    // Prefill and decode run DIFFERENT kernel stacks (coopmat GEMM + flash/non-FA attention vs
    // GEMV + split-K decode attention), so the logits legitimately carry kernel-order f16 noise
    // (~0.3 across the 262k vocab as those stacks diverged). A hard argmax equality is then brittle
    // exactly when the top-2 candidates are a statistical tie on this random-token (gibberish)
    // input — accept a flip only when BOTH runs agree the two candidates are within the tie
    // tolerance. A real wiring bug (wrong rope base / KV row / mask) shifts logits by O(1) and
    // still fails both assertions.
    let tie =
        (logits_a[ai] - logits_a[bi]).abs() < 0.05 && (logits_b[ai] - logits_b[bi]).abs() < 0.05;
    assert!(
        ai == bi || tie,
        "decode argmax diverged from prefill beyond a top-2 tie — decode-path bug\n  \
         prefill: [{ai}]={:.4} [{bi}]={:.4}\n  decode:  [{ai}]={:.4} [{bi}]={:.4}",
        logits_a[ai],
        logits_a[bi],
        logits_b[ai],
        logits_b[bi]
    );
    assert!(
        max_abs < 0.5,
        "decode logits drifted from prefill by {max_abs} (>0.5)"
    );
}
