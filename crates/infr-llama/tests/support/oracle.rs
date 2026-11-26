//! The reference cache (`infr_llama::tolerance::oracle`) on synthetic models, in temp directories,
//! with an injected code fingerprint: each input of the key turns a lookup into a miss, the same
//! file at another path is a hit, a cached reference is the computed one bit for bit, and verify
//! and the canary each catch an entry tampered with behind the checksum.

use super::forced::{self, Case};
use infr_llama::seam::{ForcedLogits, PrefillHead};
use infr_llama::tolerance::gate::{Reference, N_DECODE};
use infr_llama::tolerance::oracle::{Limits, Mode, Oracle, Query, RefCache};
use infr_llama::{EngineConfig, SeamModel};
use infr_testkit::gguf::TempGguf;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const FP: &str = "test-fingerprint";

/// A private cache root for one test, removed when dropped.
struct Root(PathBuf);

impl Root {
    fn new(tag: &str) -> Self {
        let d = std::env::temp_dir().join(format!("infr-refcache-it-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        Root(d)
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A cache at `root` and the lines it logs.
fn open(root: &Root, fp: &str, mode: Mode) -> (RefCache, Arc<Mutex<Vec<String>>>) {
    let lines = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&lines);
    let cache = RefCache::open_with(&root.0, fp, mode, Limits::DEFAULT, move |l| {
        sink.lock().unwrap().push(l.to_string())
    })
    .expect("open the cache");
    (cache, lines)
}

/// What the last lookup was: `hit`, `stored` (a miss, computed and written) or `verified`.
fn last(lines: &Mutex<Vec<String>>) -> String {
    let lines = lines.lock().unwrap();
    // A hit's canary logs a line of its own after it.
    let l = lines
        .iter()
        .rev()
        .find(|l| !l.starts_with("refcache canary"))
        .expect("a logged lookup");
    l.split_whitespace().nth(1).expect("an outcome").to_string()
}

fn bits(r: &ForcedLogits) -> Vec<u32> {
    r.logits.iter().map(|x| x.to_bits()).collect()
}

fn path_of(model: &SeamModel) -> PathBuf {
    model.gguf().shard_handles()[0].path.to_path_buf()
}

fn load(path: &Path, cfg: EngineConfig) -> SeamModel {
    SeamModel::load_with(path, None, Arc::new(cfg)).expect("model load")
}

fn synthetic() -> Case {
    forced::open_case("synthetic-f32").expect("synthetic model")
}

/// Every input of the key misses when it alone changes: one byte of the weights, one prompt
/// token, the step count, the KV format pair, the code fingerprint, one config path. A lookup
/// repeated with nothing changed hits, so each miss is the change's.
#[test]
fn reference_cache_misses_on_each_input_of_the_key() {
    let root = Root::new("inputs");
    let case = synthetic();
    let model = &case.model;
    let (cache, lines) = open(&root, FP, Mode::On);
    let oracle = Oracle::new(model, Some(&cache)).unwrap();
    let prompt = case.prompts[0].clone();
    let lookup = |o: &Oracle, q: &Query| {
        o.run(q).expect("a reference run");
        last(&lines)
    };
    let base = Query::greedy(&prompt, N_DECODE);
    assert_eq!(lookup(&oracle, &base), "stored");
    assert_eq!(lookup(&oracle, &base), "hit");

    let mut other = prompt.clone();
    other[3] = (other[3] + 1) % 64;
    assert_eq!(
        lookup(&oracle, &Query::greedy(&other, N_DECODE)),
        "stored",
        "a prompt token"
    );
    assert_eq!(
        lookup(&oracle, &Query::greedy(&prompt, N_DECODE - 1)),
        "stored",
        "the steps"
    );

    let r = forced::forced_reference(model, &prompt, N_DECODE);
    let kv = |pair| Query::along(&r, PrefillHead::All, None, Some(pair));
    assert_eq!(lookup(&oracle, &kv(("q8_0", "q8_0"))), "stored");
    assert_eq!(lookup(&oracle, &kv(("q8_0", "q8_0"))), "hit");
    assert_eq!(
        lookup(&oracle, &kv(("f16", "q4_0"))),
        "stored",
        "the KV pair"
    );
    assert_eq!(
        lookup(&oracle, &kv(("q8_0", "q4_0"))),
        "stored",
        "half the KV pair"
    );

    // One byte of one weight: the middle of a projection's data, its lowest bit.
    let (off, len) = model
        .gguf()
        .tensor_file_range("blk.1.ffn_down.weight")
        .expect("the tensor");
    let mut bytes = std::fs::read(path_of(model)).unwrap();
    bytes[(off + len as u64 / 2) as usize] ^= 1;
    let edited = TempGguf::write_copy(&std::env::temp_dir(), "refcache-byte", &bytes);
    let flipped = load(edited.path(), forced::env_cfg());
    let o = Oracle::new(&flipped, Some(&cache)).unwrap();
    assert_eq!(lookup(&o, &base), "stored", "one weight byte");

    // One config path, here one that leaves the CPU reference's bits alone: a key that missed
    // it would serve the stored run, and only the key can tell.
    let mut cfg = forced::env_cfg();
    cfg.device.ubatch = Some(5);
    let rechunked = load(&path_of(model), cfg);
    let o = Oracle::new(&rechunked, Some(&cache)).unwrap();
    assert_eq!(lookup(&o, &base), "stored", "one config path");

    let (other_build, other_lines) = open(&root, "another-fingerprint", Mode::On);
    let o = Oracle::new(model, Some(&other_build)).unwrap();
    o.run(&base).unwrap();
    assert_eq!(last(&other_lines), "stored", "the code fingerprint");
}

/// On a routed-expert model, forced along a routing: one expert id changed is a miss.
#[test]
fn reference_cache_misses_on_one_routing_id() {
    let root = Root::new("routing");
    let file = TempGguf::write("refcache-moe", &forced::synthetic_moe());
    let case = forced::open_file(file, true, false);
    let (cache, lines) = open(&root, FP, Mode::On);
    let oracle = Oracle::new(&case.model, Some(&cache)).unwrap();
    let r = forced::forced_reference(&case.model, &case.prompts[1], N_DECODE);
    let routing = r.routing.clone().expect("a routed model's routing");
    let along = |route| Query::along(&r, PrefillHead::All, Some(route), None);
    oracle.run(&along(&routing)).unwrap();
    assert_eq!(last(&lines), "stored");
    oracle.run(&along(&routing)).unwrap();
    assert_eq!(last(&lines), "hit");
    let mut moved = routing.clone();
    let at = moved.ids.len() / 2;
    moved.ids[at] = (moved.ids[at] + 1) % moved.n_expert as u32;
    oracle.run(&along(&moved)).unwrap();
    assert_eq!(last(&lines), "stored", "one routing id");
}

/// The same file at another path is the same weights: a hit.
#[test]
fn reference_cache_hits_on_the_same_file_at_another_path() {
    let root = Root::new("moved");
    let case = synthetic();
    let (cache, lines) = open(&root, FP, Mode::On);
    let q = Query::greedy(&case.prompts[1], N_DECODE);
    Oracle::new(&case.model, Some(&cache))
        .unwrap()
        .run(&q)
        .unwrap();
    assert_eq!(last(&lines), "stored");
    let bytes = std::fs::read(path_of(&case.model)).unwrap();
    let copy = TempGguf::write_copy(&std::env::temp_dir(), "refcache-moved", &bytes);
    assert_ne!(copy.path(), path_of(&case.model));
    let moved = load(copy.path(), forced::env_cfg());
    Oracle::new(&moved, Some(&cache)).unwrap().run(&q).unwrap();
    assert_eq!(last(&lines), "hit");
}

/// A reference served from the cache is the computed one, bit for bit, cold and warm, on every
/// prompt and on a KV-format run along it.
#[test]
fn cached_reference_equals_the_computed_one() {
    let root = Root::new("equal");
    let case = synthetic();
    let computed = Reference::new(&case).unwrap();
    let (cache, lines) = open(&root, FP, Mode::On);
    for pass in ["cold", "warm"] {
        let cached = Reference::cached(&case, &cache).unwrap();
        for (a, b) in computed.runs.iter().zip(&cached.runs) {
            assert!(bits(a) == bits(b), "{pass}: the logits differ");
            assert_eq!((&a.seq, &a.positions), (&b.seq, &b.positions), "{pass}");
            assert_eq!(a.prefill_rows, b.prefill_rows, "{pass}");
        }
        let kv = ("q8_0", "q8_0");
        let want = computed.along_kv(&computed.runs[2], kv).unwrap();
        let got = cached.along_kv(&cached.runs[2], kv).unwrap();
        assert!(bits(&want) == bits(&got), "{pass}: the KV run differs");
    }
    let lines = lines.lock().unwrap();
    assert!(
        lines.iter().any(|l| l.starts_with("refcache hit")),
        "{lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.starts_with("refcache canary")),
        "{lines:?}"
    );
}

/// Rewrite the one entry under `cache` with the last logit's top byte flipped, sealed again so
/// the checksum holds: a stale or wrong answer the envelope cannot see. Returns its file name.
fn tamper_the_only_entry(cache: &RefCache) -> String {
    let entries = cache.entries();
    assert_eq!(entries.len(), 1, "{entries:?}");
    let path = &entries[0];
    let data = std::fs::read(path).unwrap();
    let (magic, key, payload) =
        infr_core::kernel_cache::sealed_parts(&data).expect("a sealed entry");
    let mut payload = payload.to_vec();
    // The payload ends with the routing tag; the byte before it is the last logit's top byte.
    let at = payload.len() - 2;
    payload[at] ^= 0x40;
    let sealed = infr_core::kernel_cache::seal(magic, key, &payload).unwrap();
    std::fs::write(path, sealed).unwrap();
    path.file_name().unwrap().to_string_lossy().into_owned()
}

/// Store only the shortest prompt's greedy run, then tamper with it.
fn stored_and_tampered(case: &Case, cache: &RefCache) -> String {
    let shortest = case.prompts.iter().min_by_key(|p| p.len()).unwrap();
    Oracle::new(&case.model, Some(cache))
        .unwrap()
        .run(&Query::greedy(shortest, N_DECODE))
        .unwrap();
    tamper_the_only_entry(cache)
}

/// Verify recomputes and fails on a stored entry that differs, naming it.
#[test]
fn verify_catches_a_tampered_entry() {
    let root = Root::new("verify");
    let case = synthetic();
    let (cache, _) = open(&root, FP, Mode::On);
    let name = stored_and_tampered(&case, &cache);
    let (verify, _) = open(&root, FP, Mode::Verify);
    let e = match Reference::cached(&case, &verify) {
        Ok(_) => panic!("verify served a tampered entry"),
        Err(e) => format!("{e:#}"),
    };
    assert!(e.contains("verify") && e.contains(&name), "{e}");
}

/// The canary recomputes the shortest prompt's served run and fails on a difference, naming the
/// entry.
#[test]
fn the_canary_catches_a_tampered_entry() {
    let root = Root::new("canary");
    let case = synthetic();
    let (cache, _) = open(&root, FP, Mode::On);
    let name = stored_and_tampered(&case, &cache);
    let e = match Reference::cached(&case, &cache) {
        Ok(_) => panic!("the canary let a tampered entry through"),
        Err(e) => format!("{e:#}"),
    };
    assert!(e.contains("canary") && e.contains(&name), "{e}");
}

/// `CI` turns the cache off whatever is asked; unset, it is `on` by default and not asked for;
/// each named mode is asked for; an unknown value is an error naming it.
#[test]
fn the_cache_mode_is_read_from_ci_and_the_variable() {
    use forced::{cache_choice, CacheChoice};
    for setting in [
        None,
        Some("on"),
        Some("verify"),
        Some("clear"),
        Some("bogus"),
    ] {
        assert_eq!(
            cache_choice(true, setting),
            Ok(CacheChoice::Off("CI is set")),
            "{setting:?}"
        );
    }
    let open = |mode, asked| Ok(CacheChoice::Open { mode, asked });
    assert_eq!(cache_choice(false, None), open(Mode::On, false));
    assert_eq!(cache_choice(false, Some("on")), open(Mode::On, true));
    assert_eq!(
        cache_choice(false, Some("verify")),
        open(Mode::Verify, true)
    );
    assert_eq!(
        cache_choice(false, Some("refresh")),
        open(Mode::Refresh, true)
    );
    assert_eq!(
        cache_choice(false, Some("off")),
        Ok(CacheChoice::Off("INFR_TOL_REFCACHE=off"))
    );
    assert_eq!(cache_choice(false, Some("clear")), Ok(CacheChoice::Clear));
    let e = cache_choice(false, Some("bogus")).unwrap_err();
    assert!(e.contains("bogus"), "{e}");
}

/// A cache that did not open fails the run only for a build without a fingerprint whose mode was
/// asked for by name; every other failure runs the gate without the cache.
#[test]
fn only_an_asked_for_cache_without_a_fingerprint_refuses_to_run() {
    use forced::refuse_without_cache;
    assert!(refuse_without_cache(true, false));
    assert!(!refuse_without_cache(false, false));
    assert!(!refuse_without_cache(true, true));
    assert!(!refuse_without_cache(false, true));
}
