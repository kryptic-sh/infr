//! PagedAttention on a real model: two conversations through one Vulkan session, run once with a
//! contiguous KV cache and once with the page pool. The second conversation FORKS the first — the
//! only thing that builds a pool — so this drives the fork, the shared pages and the copy-on-write
//! rather than just the descriptor plumbing.
//!
//! Runs only when a suitable GGUF is in the HF cache and a Vulkan device exists; otherwise it
//! self-skips. Greedy sampling on a fixed prompt set, so the two runs must agree byte for byte.

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
/// them — plus the Gemma 4 file this machine serves, which exercises the mixed case.
fn model_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
        return Some(std::path::PathBuf::from(p));
    }
    let store = infr_hub::Store::discover().ok()?;
    [
        ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
        (
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        ),
        (
            "unsloth/gemma-4-E4B-it-qat-GGUF",
            "Gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
        ),
    ]
    .into_iter()
    .find_map(|(repo, file)| store.snapshot_file(repo, file))
}

fn engine(paged: bool) -> infr_llama::EngineConfig {
    let mut c = infr_llama::EngineConfig::default();
    c.kv.paged = Some(paged);
    c.kv.slots = 2;
    // A tiny host store, so a recycled slot restores into its block (and copies a shared page
    // before it writes) — the two prefix mechanisms have to compose.
    c.kv.store_mb = Some(64);
    c
}

/// Logink capture, so the test can assert the pool actually engaged instead of trusting a
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

    assert_eq!(
        got, want,
        "a paged slot diverged from a contiguous one (prefill tokens: paged {got_prefill:?}, \
         contiguous {want_prefill:?})"
    );

    let log = String::from_utf8_lossy(&log.0.lock().unwrap()).to_string();
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
