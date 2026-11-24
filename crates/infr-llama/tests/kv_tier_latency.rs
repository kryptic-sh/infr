//! What a restored prefix costs, per tier, on a real device: the first token of a prompt against
//! the first token of the same prompt once its rows are in the store, for a host-RAM store, a redb
//! store and a Valkey store, and the same comparison at 4k and 32k.
//!
//! Runs only when a suitable GGUF is in the HF cache and a Vulkan device exists. The Valkey row
//! needs `INFR_TEST_REDIS_URL` pointing at an EMPTY database: the store is keyed by model
//! geometry, so rows left by an earlier run are restored and the "cold" column is not cold.
//! Prints a table instead of asserting a threshold, because the point is the ratio, which moves
//! with the model and the card.
//!
//! Run: `cargo test -p infr-llama --release --test kv_tier_latency -- --ignored --nocapture`

use std::sync::Arc;
use std::time::Instant;

/// A prompt of about `tokens` tokens: a repeated sentence, then a question that differs per run.
fn prompt(tokens: usize, tag: &str) -> String {
    let sentence = "The maintenance log of the Honshu line is kept in triplicate: one copy rides \
                    the train, one sits in the depot office, and one is sealed in the yardmaster's \
                    safe. ";
    let mut s = String::from("Use this reference material:\n");
    while s.len() < tokens * 4 {
        s.push_str(sentence);
    }
    s.push_str(&format!("Report the {tag} entry."));
    s
}

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
        ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
    ]
    .into_iter()
    .find_map(|(repo, file)| store.snapshot_file(repo, file))
}

fn engine(
    store_mb: u64,
    disk: Option<&std::path::Path>,
    url: Option<&str>,
) -> infr_llama::EngineConfig {
    let mut c = infr_llama::EngineConfig::default();
    c.kv.slots = 1;
    let mut redb = infr_core::config::KvTier {
        backend: "redb".to_string(),
        size: Some(infr_core::SizeSpec::Bytes(store_mb << 20)),
        ..Default::default()
    };
    redb.path = disk.map(std::path::Path::to_path_buf);
    c.kv.tiers.push(redb);
    if let Some(url) = url {
        c.kv.tiers.push(infr_core::config::KvTier {
            backend: "redis".to_string(),
            url: Some(url.to_string()),
            ..Default::default()
        });
    }
    c
}

struct Timing {
    tokens: usize,
    cold: f64,
    warm: f64,
    cached: usize,
}

/// One prompt twice, after a shorter prompt built the pipelines this shape uses: the first run
/// fills the store, the second restores it. The clock covers the whole call at `max_new = 1`, so it
/// is time-to-first-token plus one decode step, restore included.
fn twice(
    path: &std::path::Path,
    cfg: infr_llama::EngineConfig,
    ctx: usize,
    tokens: usize,
    tag: &str,
) -> Timing {
    let model = infr_llama::SeamModel::load_with(path, None, Arc::new(cfg)).expect("load");
    let seam = infr_llama::parallel::ParallelSeam::new(
        model,
        1,
        Some(infr_core::SizeSpec::Bytes(ctx as u64)),
    )
    .expect("engine");
    let greedy = infr_llama::sampling::RequestSampling {
        temp: Some(0.0),
        ..Default::default()
    };
    let run = |p: &str| {
        let req = seam.request_ctx(greedy.clone());
        let t0 = Instant::now();
        let stats = seam.generate(p, 1, None, &req, |_| {}).expect("generation");
        (t0.elapsed().as_secs_f64(), stats)
    };
    run(&prompt(512, "warmup"));
    let p = prompt(tokens, tag);
    let (cold, cold_stats) = run(&p);
    let (warm, warm_stats) = run(&p);
    Timing {
        tokens: cold_stats.n_prompt + cold_stats.n_cached,
        cold,
        warm,
        cached: warm_stats.n_cached,
    }
}

fn row(label: &str, t: &Timing) -> String {
    format!(
        "{label:8} {:6} {:9.0} ms {:12.0} ms {:7.1}x  ({} restored)",
        t.tokens,
        t.cold * 1e3,
        t.warm * 1e3,
        t.cold / t.warm.max(1e-6),
        t.cached
    )
}

#[test]
#[ignore = "requires a Vulkan GPU and a cached GGUF"]
fn restore_cost_per_tier() {
    let Some(path) = model_path() else {
        eprintln!("skip: no GGUF in the HF cache");
        return;
    };
    let dir = std::env::temp_dir().join(format!("infr-kv-latency-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let tokens = 2048;
    let ctx = 4096;
    println!("\ntier        tokens       first  same prompt again  speedup");

    // One in-memory database: the store lives in RSS and dies with the process.
    println!(
        "{}",
        row(
            "redb-mem",
            &twice(&path, engine(1024, None, None), ctx, tokens, "mem")
        )
    );

    // One database file: the same code path, with the rows on disk and the page cache in front.
    let disk_path = dir.join("kv.redb");
    let disk = twice(
        &path,
        engine(1024, Some(&disk_path), None),
        ctx,
        tokens,
        "redb",
    );
    println!("{}", row("redb-file", &disk));

    match std::env::var("INFR_TEST_REDIS_URL") {
        Ok(url) => {
            let redis = twice(&path, engine(1024, None, Some(&url)), ctx, tokens, "valkey");
            println!("{}", row("valkey", &redis));
        }
        Err(_) => println!("valkey      skipped (INFR_TEST_REDIS_URL is unset)"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "requires a Vulkan GPU and a cached GGUF; prefills 32k three times"]
fn restore_cost_at_long_context() {
    let Some(path) = model_path() else {
        eprintln!("skip: no GGUF in the HF cache");
        return;
    };
    println!("\ntokens         first  same prompt again  speedup  (in-memory store)");
    for (tokens, ctx) in [(4096, 8192), (32768, 36864)] {
        let t = twice(&path, engine(1024, None, None), ctx, tokens, "long");
        println!("{}", row(&tokens.to_string(), &t));
    }
}
