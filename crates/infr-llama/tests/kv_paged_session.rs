//! PagedAttention on a real model: two conversations through one Vulkan session, run once with a
//! contiguous KV cache and once with the page pool. The second conversation FORKS the first — the
//! only thing that builds a pool — so this drives the fork, the shared pages and the copy-on-write
//! rather than just the descriptor plumbing.
//!
//! Runs only when a suitable GGUF is in the HF cache and a Vulkan device exists; otherwise it
//! self-skips. Greedy sampling on a fixed prompt set, so the two runs must agree byte for byte.
//!
//! Run: `cargo test -p infr-llama --test kv_paged_session -- --ignored --nocapture`

use std::io::Write;
use std::sync::{Arc, Mutex};

/// One shared "system prompt" long enough to span more than two 512-row pages.
fn shared_prefix() -> String {
    let sentence = "The maintenance log of the Honshu line is kept in triplicate: one copy rides \
                    the train, one sits in the depot office, and one is sealed in the yardmaster's \
                    safe. ";
    let mut s = String::from("You are a terse assistant. Use this reference material:\n");
    for _ in 0..40 {
        s.push_str(sentence);
    }
    s
}

/// A model that caches every layer's KV (no sliding-window ring), so the page pool can hold all of
/// them. Honours `INFR_TEST_MODEL` so a local file can be used instead of the hub cache.
fn model_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
        return Some(std::path::PathBuf::from(p));
    }
    let store = infr_hub::Store::discover().ok()?;
    [
        (
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        ),
        ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
        ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
    ]
    .into_iter()
    .find_map(|(repo, file)| store.snapshot_file(repo, file))
}

fn engine(paged: bool) -> infr_llama::EngineConfig {
    engine_kv(paged, infr_core::DType::F16)
}

fn engine_kv(paged: bool, fmt: infr_core::DType) -> infr_llama::EngineConfig {
    let mut c = infr_llama::EngineConfig::default();
    c.kv.paged = Some(paged);
    c.kv.slots = 2;
    // No auto-q8: the format under test is the one asked for, and Vulkan's Q8_0 cache is planar
    // (codes first, then the scales), which is what the row copies have to handle.
    c.kv.type_k = Some(fmt);
    c.kv.type_v = Some(fmt);
    // A tiny host store, so a recycled slot restores into its block (and copies a shared page
    // before it writes) — the two prefix mechanisms have to compose.
    c.kv.tiers.push(infr_core::config::KvTier {
        backend: "redb".to_string(),
        size: Some(infr_core::SizeSpec::Bytes(64 << 20)),
        ..Default::default()
    });
    c
}

/// Log capture, so the test can assert the pool actually engaged instead of trusting a
/// byte-identical answer that a silently-linear slot would also produce.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Captured {
    type Writer = Captured;
    fn make_writer(&'a self) -> Captured {
        self.clone()
    }
}

fn conversations(model: &infr_llama::SeamModel, ctx: usize) -> (Vec<String>, Vec<usize>) {
    let mut sess = model.vulkan_session(ctx).expect("vulkan session");
    let sys = shared_prefix();
    let pa = format!("{sys}Name the three places the log is kept.");
    let pb = format!("{sys}Which copy rides the train?");
    let mut texts = Vec::new();
    let mut prefilled = Vec::new();
    let mut turn = |sess: &mut _, prompt: &str| {
        let mut t = String::new();
        let stats = model
            .generate_vulkan_session(sess, prompt, 8, None, |p| t.push_str(p))
            .expect("generation");
        texts.push(t.clone());
        prefilled.push(stats.n_prompt);
        t
    };
    let a = turn(&mut sess, &pa);
    turn(&mut sess, &pb);
    // Back to the first conversation: its slot must still hold A's prefix after B forked it.
    turn(&mut sess, &format!("{pa}{a} And the fourth place?"));
    (texts, prefilled)
}

#[test]
#[ignore = "requires a Vulkan GPU and a cached GGUF"]
fn paged_slots_answer_the_same_bytes_as_contiguous_ones() {
    let Some(path) = model_path() else {
        eprintln!("skip: no GGUF in the HF cache");
        return;
    };
    let ctx = 4096;
    let linear = infr_llama::SeamModel::load_with(&path, None, Arc::new(engine(false)))
        .expect("load contiguous");
    let mixed = {
        let c = linear.config();
        (0..c.n_layer).any(|l| c.is_swa_layer(l))
    };
    if mixed {
        eprintln!(
            "skip: {} keeps sliding-window layers linear, so no prefix is shared",
            path.display()
        );
        return;
    }
    let (want, want_prefill) = conversations(&linear, ctx);
    drop(linear);

    let log = Captured::default();
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(log.clone())
        .try_init();
    let paged =
        infr_llama::SeamModel::load_with(&path, None, Arc::new(engine(true))).expect("load paged");
    let (got, got_prefill) = conversations(&paged, ctx);
    drop(paged);

    let log = String::from_utf8_lossy(&log.0.lock().unwrap()).to_string();
    assert_eq!(
        got, want,
        "a paged slot diverged from a contiguous one (prefill tokens: paged {got_prefill:?}, \
         contiguous {want_prefill:?}):\n{log}"
    );

    assert!(
        log.contains("kv.paged:"),
        "the pool never engaged, so the comparison proved nothing:\n{log}"
    );
    let shared: usize = log
        .split("forked slot shares ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);
    assert!(
        shared >= 2,
        "the forked slot shared {shared} pages, so the shared-page path was not exercised:\n{log}"
    );
    // The shared prefix is seeded in whole pages, so the paged slot re-prefills at most one page
    // more than the contiguous one — and far less than a full re-prefill.
    assert!(
        want_prefill[1] < 64,
        "the contiguous slot prefilled {} tokens for a prompt sharing its prefix — no seed happened",
        want_prefill[1]
    );
    assert!(
        got_prefill[1] <= want_prefill[1] + 512 + 8,
        "the paged slot re-prefilled {} tokens against {} contiguous",
        got_prefill[1],
        want_prefill[1]
    );
}

/// Vulkan keeps a Q8_0 KV side planar: every int8 code, then every block's f16 scale. The store,
/// a fork's tail copy and a page's copy-on-write all move rows of that cache, so this runs the same
/// prompt twice through a paged engine: the second answer comes from rows restored out of the store
/// into a page pool, and has to match the first one byte for byte.
///
/// Run: `cargo test -p infr-llama --test kv_paged_session -- --ignored --nocapture`
#[test]
#[ignore = "requires a Vulkan GPU and a cached GGUF"]
fn q8_kv_keeps_its_rows_through_pages_and_the_store() {
    let Some(path) = model_path() else {
        eprintln!("skip: no GGUF in the HF cache");
        return;
    };
    let ctx = 2048;
    let model = infr_llama::SeamModel::load_with(
        &path,
        None,
        Arc::new(engine_kv(true, infr_core::DType::Q8_0)),
    )
    .expect("load q8 paged");
    let seam = infr_llama::parallel::ParallelSeam::new(
        model,
        2,
        Some(infr_core::SizeSpec::Bytes(ctx as u64)),
    )
    .expect("engine");
    assert!(
        seam.kv_pages().is_some(),
        "the pool was not built for a planar Q8_0 cache"
    );

    let greedy = infr_llama::sampling::RequestSampling {
        temp: Some(0.0),
        ..Default::default()
    };
    let run = |prompt: &str| {
        let req = seam.request_ctx(greedy.clone());
        let mut text = String::new();
        let stats = seam
            .generate(prompt, 8, None, &req, |p| text.push_str(p))
            .expect("generation");
        (text, stats)
    };

    let sys = shared_prefix();
    let prompt = format!("{sys}Name the three places the log is kept.");
    let (first, _) = run(&prompt);
    let (again, warm) = run(&prompt);
    assert!(warm.n_cached > 0, "the second run restored nothing");
    let (restores, restored_tokens, captures) = seam.kv_store_stats();
    assert!(
        captures > 0 && restores > 0 && restored_tokens > 0,
        "the prefix store skipped a planar Q8_0 cache: {captures} captures, {restores} restores"
    );
    assert_eq!(
        again, first,
        "a restored planar Q8_0 cache changed the answer"
    );

    // A different tail off the same prefix writes into the pages the fork shares.
    let (_, other) = run(&format!("{sys}Which copy rides the train?"));
    assert!(
        other.n_cached > 0,
        "the second conversation reused none of the shared prefix"
    );
}

/// Two conversations in a pool that holds one: the second parks the first, and each resumes by
/// adopting the pages the parked one left in the pool, with no upload from the store, and answers
/// as a cold run does.
///
/// Run: `cargo test -p infr-llama --test kv_paged_session -- --ignored --nocapture`
#[test]
#[ignore = "requires a Vulkan GPU and a cached GGUF"]
fn parked_slots_resume_two_conversations_in_a_one_conversation_pool() {
    let Some(path) = model_path() else {
        eprintln!("skip: no GGUF in the HF cache");
        return;
    };
    let ctx: usize = 2048;
    let cfg0 = infr_llama::SeamModel::load_with(&path, None, Arc::new(engine(true)))
        .expect("load")
        .config()
        .clone();
    if (0..cfg0.n_layer).any(|l| cfg0.is_swa_layer(l)) {
        eprintln!("skip: sliding-window layers keep a private cache");
        return;
    }

    // One conversation's pages, plus the zero page: a second one cannot fit beside it.
    let row = infr_core::budget::kv_fmt_bytes(
        infr_core::DType::F16,
        infr_vulkan::paged::PAGE_ROWS * cfg0.n_kv * cfg0.head_dim,
    );
    let page_bytes = (row * 2 * cfg0.n_layer) as u64;
    let pps = ctx.div_ceil(infr_vulkan::paged::PAGE_ROWS) as u64;
    let mut c = engine(true);
    c.kv.pool_mb = Some((page_bytes * (pps + 1)).div_ceil(1 << 20));
    c.kv.tiers[0].size = Some(infr_core::SizeSpec::Bytes(256 << 20));
    let model = infr_llama::SeamModel::load_with(&path, None, Arc::new(c)).expect("load paged");
    let seam = infr_llama::parallel::ParallelSeam::new(
        model,
        2,
        Some(infr_core::SizeSpec::Bytes(ctx as u64)),
    )
    .expect("engine");
    let pages = seam.kv_pages().expect("the pool was built");
    assert!(
        pages.0 + pages.1 <= pps as usize + 1,
        "the pool holds {} pages, so both conversations fit and nothing would park",
        pages.0 + pages.1
    );

    let greedy = infr_llama::sampling::RequestSampling {
        temp: Some(0.0),
        ..Default::default()
    };
    let sys = shared_prefix();
    let run = |prompt: &str| {
        let req = seam.request_ctx(greedy.clone());
        let mut text = String::new();
        let stats = seam
            .generate(prompt, 8, None, &req, |p| text.push_str(p))
            .expect("parked generation");
        (text, stats)
    };

    let (a, _) = run(&format!("{sys}Name the three places the log is kept."));
    let (b, _) = run(&format!("{sys}Which copy rides the train?"));
    // Both conversations return. Each one parked the other, whose whole pages stayed in the pool
    // as a resident, so the shared prefix is adopted in place rather than read from the store.
    let resume_a = format!("{sys}Name the three places the log is kept.{a} And the fourth?");
    let resume_b = format!("{sys}Which copy rides the train?{b} Who reads it?");
    let (store0, (hits0, ..)) = (seam.kv_store_stats(), seam.kv_resident_stats());
    let (text_a, again_a) = run(&resume_a);
    let (store1, (hits1, ..)) = (seam.kv_store_stats(), seam.kv_resident_stats());
    let (text_b, again_b) = run(&resume_b);
    let (store2, (hits2, ..)) = (seam.kv_store_stats(), seam.kv_resident_stats());
    assert!(
        again_a.n_cached > 0 && again_b.n_cached > 0,
        "a parked conversation re-prefilled instead of resuming (cached {} and {} tokens)",
        again_a.n_cached,
        again_b.n_cached
    );
    for (name, before, after, hits) in [
        ("first", store0, store1, hits1 - hits0),
        ("second", store1, store2, hits2 - hits1),
    ] {
        assert!(hits >= 1, "the {name} resume did not adopt resident pages");
        assert_eq!(
            (before.0, before.1),
            (after.0, after.1),
            "the {name} resume uploaded rows from the store"
        );
    }
    drop(seam);

    // A fresh engine runs each resumed prompt cold: the adopted pages must not change the answer.
    let model =
        infr_llama::SeamModel::load_with(&path, None, Arc::new(engine(true))).expect("load fresh");
    let fresh = infr_llama::parallel::ParallelSeam::new(
        model,
        2,
        Some(infr_core::SizeSpec::Bytes(ctx as u64)),
    )
    .expect("fresh engine");
    let cold = |prompt: &str| {
        let req = fresh.request_ctx(greedy.clone());
        let mut text = String::new();
        fresh
            .generate(prompt, 8, None, &req, |p| text.push_str(p))
            .expect("fresh generation");
        text
    };
    assert_eq!(
        text_a,
        cold(&resume_a),
        "the first resume changed the answer"
    );
    assert_eq!(
        text_b,
        cold(&resume_b),
        "the second resume changed the answer"
    );
}
