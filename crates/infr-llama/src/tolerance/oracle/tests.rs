//! The cache's own parts on files alone: the encoding, the envelope's refusals, concurrent
//! writers and every cleanup rule. What needs a model (a key that misses on each input, a hit at
//! another path, the canary, verify) is in `tests/support/oracle.rs`.

use super::*;

const FP: &str = "fingerprint-a";
const HOUR: Duration = Duration::from_secs(3600);
const DAY: Duration = Duration::from_secs(24 * 3600);

/// A private directory for one test, removed at its end.
fn tmp_root(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("infr-refcache-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn open(root: &Path, fp: &str) -> RefCache {
    RefCache::open_with(root, fp, Mode::On, Limits::DEFAULT, |_| {}).expect("open")
}

/// A small run with a routing, so every field of the encoding is exercised.
fn run(seed: u32) -> ForcedLogits {
    ForcedLogits {
        positions: vec![0, 1, 2],
        logits: (0..6).map(|i| (i + seed) as f32 * 0.5 - 1.0).collect(),
        vocab: 2,
        prefill_rows: 2,
        seq: vec![seed, 3, 4, 5],
        kernels: Vec::new(),
        routing: Some(Routing {
            layers: vec![1, 3],
            n_used: 1,
            n_expert: 4,
            ids: (0..8).map(|i| (i + seed) % 4).collect(),
            scores: vec![0.25; 32],
        }),
    }
}

fn set_age(path: &Path, now: SystemTime, age: Duration) {
    File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(now - age)
        .unwrap();
}

/// Every file under `root`, relative, sorted.
fn tree(root: &Path) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            if std::fs::symlink_metadata(&p).unwrap().is_dir() {
                walk(root, &p, out);
            } else {
                out.push(
                    p.strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

#[test]
fn a_run_round_trips_through_its_encoding() {
    for r in [
        run(1),
        ForcedLogits {
            routing: None,
            ..run(2)
        },
    ] {
        let back = decode(&encode(&r).unwrap()).unwrap();
        assert!(same_bits(&r, &back).unwrap());
        assert_eq!(back.positions, r.positions);
        assert_eq!(back.seq, r.seq);
        assert_eq!(back.routing, r.routing);
    }
    let bytes = encode(&run(1)).unwrap();
    assert!(decode(&bytes[..bytes.len() - 1]).is_err(), "a short tail");
    let mut long = bytes.clone();
    long.push(0);
    assert!(decode(&long).is_err(), "a byte past the end");
    let mut gpu = run(1);
    gpu.kernels.push("native_gemv");
    assert!(encode(&gpu).is_err(), "a run that built kernels");
    // Each value is in the bytes: a logit one ulp away encodes differently.
    let mut ulp = run(1);
    ulp.logits[5] = f32::from_bits(ulp.logits[5].to_bits() + 1);
    assert!(!same_bits(&run(1), &ulp).unwrap());
}

/// An entry with one flipped payload byte is refused and deleted; one written under another key
/// is a miss and stays.
#[test]
fn a_flipped_byte_is_rejected_and_deleted() {
    let root = tmp_root("flip");
    let c = open(&root, FP);
    c.write(b"key", &run(1)).unwrap();
    assert!(c.read(b"key").unwrap().is_some());
    let path = c.entry_path(b"key");
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x10;
    std::fs::write(&path, &bytes).unwrap();
    assert!(
        c.read(b"key").unwrap().is_none(),
        "a damaged entry was served"
    );
    assert!(!path.exists(), "a damaged entry is deleted");
    // The same file under another key's name: well formed, the wrong key.
    c.write(b"key", &run(1)).unwrap();
    std::fs::copy(&path, c.entry_path(b"other")).unwrap();
    assert!(c.read(b"other").unwrap().is_none());
    assert!(
        c.entry_path(b"other").exists(),
        "a stale entry is not damaged"
    );
    std::fs::remove_dir_all(&root).ok();
}

/// Two writers of one key, racing, leave an entry that reads whole.
#[test]
fn two_writers_leave_a_readable_entry() {
    let root = tmp_root("writers");
    let c = open(&root, FP);
    let want = run(3);
    std::thread::scope(|s| {
        for _ in 0..2 {
            s.spawn(|| {
                for _ in 0..50 {
                    c.write(b"same", &want).unwrap();
                }
            });
        }
        s.spawn(|| {
            for _ in 0..50 {
                if let Some(got) = c.read(b"same").unwrap() {
                    assert!(same_bits(&got, &want).unwrap(), "a torn entry was read");
                }
            }
        });
    });
    let got = c.read(b"same").unwrap().expect("an entry");
    assert!(same_bits(&got, &want).unwrap());
    assert_eq!(c.entries().len(), 1, "{:?}", tree(&root));
    std::fs::remove_dir_all(&root).ok();
}

/// Rule 1: another generation unused past the age limit goes, files and directory; one used
/// recently stays, and so does the current one, however old its entries' stamps.
#[test]
fn cleanup_removes_an_old_generation_only() {
    let root = tmp_root("rule1");
    let now = SystemTime::now();
    let old = open(&root, "old");
    old.write(b"k", &run(1)).unwrap();
    let recent = open(&root, "recent");
    recent.write(b"k", &run(1)).unwrap();
    let current = open(&root, FP);
    current.write(b"k", &run(1)).unwrap();
    let stale = now - 4 * DAY;
    write_sealed(
        &old.generation.join(STAMP_NAME),
        &seal(STAMP_MAGIC, b"", &unix_secs(stale).to_le_bytes()).unwrap(),
    )
    .unwrap();
    for f in tree(&old.generation) {
        set_age(&old.generation.join(f), now, 4 * DAY);
    }
    for f in tree(&recent.generation) {
        set_age(&recent.generation.join(f), now, 2 * DAY);
    }
    let removed = cleanup(&root, FP, Limits::DEFAULT, now);
    assert_eq!(removed, 3, "{:?}", tree(&root));
    assert!(!old.generation.exists(), "{:?}", tree(&root));
    assert_eq!(recent.entries().len(), 1);
    assert_eq!(current.entries().len(), 1);
    std::fs::remove_dir_all(&root).ok();
}

/// Rule 2: an entry of the current generation unused past the age limit goes; a used one stays.
#[test]
fn cleanup_removes_an_unused_entry_of_the_current_generation() {
    let root = tmp_root("rule2");
    let now = SystemTime::now();
    let c = open(&root, FP);
    c.write(b"unused", &run(1)).unwrap();
    c.write(b"used", &run(2)).unwrap();
    set_age(&c.entry_path(b"unused"), now, 4 * DAY);
    set_age(&c.entry_path(b"used"), now, 2 * DAY);
    assert_eq!(cleanup(&root, FP, Limits::DEFAULT, now), 1);
    assert!(!c.entry_path(b"unused").exists());
    assert!(c.entry_path(b"used").exists());
    std::fs::remove_dir_all(&root).ok();
}

/// Rule 3: past the size cap the least recently used entries go, across generations, until the
/// rest fit; a hit makes an entry recently used.
#[test]
fn cleanup_evicts_the_least_recently_used_past_the_cap() {
    let root = tmp_root("rule3");
    let now = SystemTime::now();
    let other = open(&root, "other");
    other.write(b"a", &run(1)).unwrap();
    let c = open(&root, FP);
    c.write(b"b", &run(2)).unwrap();
    c.write(b"c", &run(3)).unwrap();
    set_age(&other.entry_path(b"a"), now, HOUR * 3);
    set_age(&c.entry_path(b"b"), now, HOUR * 4);
    set_age(&c.entry_path(b"c"), now, HOUR * 5);
    // A hit on `c` makes it the most recently used.
    assert!(c.read(b"c").unwrap().is_some());
    let one = std::fs::metadata(c.entry_path(b"b")).unwrap().len();
    let limits = Limits {
        size_cap: 2 * one,
        ..Limits::DEFAULT
    };
    // The hit's stamp is `SystemTime::now()`, past this test's `now`: evaluate a little later.
    let later = SystemTime::now() + HOUR * 2;
    assert_eq!(cleanup(&root, FP, limits, later), 1, "{:?}", tree(&root));
    assert!(!c.entry_path(b"b").exists(), "the least recently used goes");
    assert!(other.entry_path(b"a").exists() && c.entry_path(b"c").exists());
    std::fs::remove_dir_all(&root).ok();
}

/// Rule 4: a file of the cache's name and family in an older magic or envelope version goes.
#[test]
fn cleanup_removes_an_older_layout() {
    let root = tmp_root("rule4");
    let now = SystemTime::now();
    let c = open(&root, FP);
    c.write(b"new", &run(1)).unwrap();
    let old_magic = c.entry_path(b"old-magic");
    write_sealed(&old_magic, &seal(*b"INFRTRE0", b"old-magic", b"x").unwrap()).unwrap();
    let old_envelope = c.entry_path(b"old-envelope");
    let mut bytes = seal(ENTRY_MAGIC, b"old-envelope", b"x").unwrap();
    bytes[8..10].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
    std::fs::write(&old_envelope, bytes).unwrap();
    for p in [&old_magic, &old_envelope, &c.entry_path(b"new")] {
        set_age(p, now, HOUR * 2);
    }
    assert_eq!(cleanup(&root, FP, Limits::DEFAULT, now), 2);
    assert!(!old_magic.exists() && !old_envelope.exists());
    assert!(c.entry_path(b"new").exists());
    std::fs::remove_dir_all(&root).ok();
}

/// Make `link` a symlink to the file `target`, or `None` where this system will not let the test
/// make one (Windows without the privilege).
fn symlink_file(target: &Path, link: &Path) -> Option<()> {
    #[cfg(unix)]
    let made = std::os::unix::fs::symlink(target, link);
    #[cfg(windows)]
    let made = std::os::windows::fs::symlink_file(target, link);
    #[cfg(not(any(unix, windows)))]
    let made: std::io::Result<()> = Err(std::io::ErrorKind::Unsupported.into());
    made.ok()
}

/// Nothing the cache cannot prove its own is deleted by any rule or by a clear: a file of
/// another name, one of the entries' name with a foreign header, a symlink named as an entry
/// (and the file it points at, outside the root), a generation named as the cache names them
/// with no marker of its own, and any file newer than the grace period.
#[test]
fn cleanup_and_clear_leave_what_they_cannot_prove_their_own() {
    let root = tmp_root("foreign");
    let outside = tmp_root("foreign-outside");
    let now = SystemTime::now();
    let old = open(&root, "old");
    old.write(b"k", &run(1)).unwrap();
    let foreign_name = old.generation.join("notes.txt");
    std::fs::write(&foreign_name, b"keep me").unwrap();
    let foreign_header = old.entry_path(b"foreign");
    std::fs::write(&foreign_header, [7u8; 64]).unwrap();
    let target = outside.join("victim.ref");
    std::fs::copy(old.entry_path(b"k"), &target).unwrap();
    // Old enough for every rule, so only the symlink check keeps it.
    set_age(&target, now, 10 * DAY);
    let link = old.entry_path(b"link");
    let linked = symlink_file(&target, &link);
    // Only Windows may refuse one (it takes a privilege there); then the symlink half is not
    // exercised on that machine. Everywhere else a refusal fails the test.
    assert!(
        linked.is_some() || cfg!(windows),
        "no symlink could be made"
    );
    let unmarked = root.join(format!("{GEN_PREFIX}{}", "0".repeat(GEN_HEX)));
    std::fs::create_dir_all(&unmarked).unwrap();
    std::fs::copy(
        old.entry_path(b"k"),
        unmarked.join(format!("{ENTRY_PREFIX}{}.ref", "1".repeat(ENTRY_HEX))),
    )
    .unwrap();
    let fresh = open(&root, "fresh");
    fresh.write(b"k", &run(1)).unwrap();
    for f in tree(&root) {
        let p = root.join(&f);
        if !f.starts_with(&generation_name("fresh"))
            && std::fs::symlink_metadata(&p).unwrap().is_file()
        {
            set_age(&p, now, 10 * DAY);
        }
    }
    write_sealed(
        &old.generation.join(STAMP_NAME),
        &seal(STAMP_MAGIC, b"", &unix_secs(now - 10 * DAY).to_le_bytes()).unwrap(),
    )
    .unwrap();
    set_age(&old.generation.join(STAMP_NAME), now, 10 * DAY);
    write_sealed(
        &fresh.generation.join(STAMP_NAME),
        &seal(STAMP_MAGIC, b"", &unix_secs(now - 10 * DAY).to_le_bytes()).unwrap(),
    )
    .unwrap();
    let limits = Limits {
        size_cap: 0,
        ..Limits::DEFAULT
    };
    cleanup(&root, FP, limits, now);
    let left = tree(&root);
    assert!(foreign_name.exists() && foreign_header.exists(), "{left:?}");
    assert!(
        old.generation.exists(),
        "a directory with a foreign file in it stays"
    );
    assert!(
        !old.entry_path(b"k").exists(),
        "the old generation's own entry goes: {left:?}"
    );
    if linked.is_some() {
        assert!(
            std::fs::symlink_metadata(&link).is_ok(),
            "the symlink was deleted"
        );
    }
    assert!(target.exists(), "the file outside the root was deleted");
    assert_eq!(
        tree(&unmarked).len(),
        1,
        "a generation without its marker was touched"
    );
    // The fresh generation's files are inside the grace period: even the cap of zero keeps them.
    assert_eq!(fresh.entries().len(), 1, "{left:?}");

    let removed = RefCache::clear(&root);
    assert!(removed > 0);
    let left = tree(&root);
    assert!(!fresh.generation.exists(), "clear left {left:?}");
    assert!(
        foreign_name.exists() && foreign_header.exists() && target.exists(),
        "{left:?}"
    );
    if linked.is_some() {
        assert!(
            std::fs::symlink_metadata(&link).is_ok(),
            "clear deleted the symlink"
        );
    }
    assert_eq!(tree(&unmarked).len(), 1);
    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_dir_all(&outside).ok();
}

#[test]
fn an_empty_fingerprint_opens_nothing() {
    let root = tmp_root("nofp");
    assert!(RefCache::open_with(&root, "", Mode::On, Limits::DEFAULT, |_| {}).is_err());
    std::fs::remove_dir_all(&root).ok();
}

/// This build has a fingerprint, so the tests' own build can open the cache.
#[test]
fn this_build_has_a_fingerprint() {
    let fp = fingerprint().expect("build.rs wrote no fingerprint");
    assert_eq!(fp.len(), 64, "{fp}");
}

/// Rule 2 deletes nothing within the grace period, even under an age limit shorter than it.
#[test]
fn rule_two_keeps_what_is_within_the_grace_period() {
    let root = tmp_root("rule2-grace");
    let now = SystemTime::now();
    let c = open(&root, FP);
    c.write(b"young", &run(1)).unwrap();
    set_age(&c.entry_path(b"young"), now, HOUR / 2);
    let limits = Limits {
        max_age: Duration::from_secs(60),
        ..Limits::DEFAULT
    };
    assert_eq!(cleanup(&root, FP, limits, now), 0);
    assert!(c.entry_path(b"young").exists());
    std::fs::remove_dir_all(&root).ok();
}

/// A temp file a writer left before its header was written is the cache's by its name alone:
/// removed once older than the grace period and the age limit, kept while younger.
#[test]
fn a_headerless_temp_file_goes_only_once_stale() {
    let root = tmp_root("headerless");
    let now = SystemTime::now();
    let c = open(&root, FP);
    let entry = c.entry_path(b"k");
    let stem = entry.file_stem().unwrap().to_string_lossy().into_owned();
    let old = c.generation.join(format!("{stem}.tmp.1.1"));
    let young = c.generation.join(format!("{stem}.tmp.1.2"));
    for p in [&old, &young] {
        std::fs::write(p, b"short").unwrap();
    }
    set_age(&old, now, 4 * DAY);
    set_age(&young, now, 2 * DAY);
    assert_eq!(
        cleanup(&root, FP, Limits::DEFAULT, now),
        1,
        "{:?}",
        tree(&root)
    );
    assert!(!old.exists() && young.exists());
    // In another generation unused for long, every file of its own goes, but a headerless temp
    // file younger than the age limit may still be a write in progress: it stays.
    let other = open(&root, "other");
    let pending = other.generation.join(format!("{stem}.tmp.2.1"));
    std::fs::write(&pending, b"short").unwrap();
    for f in tree(&other.generation) {
        set_age(&other.generation.join(f), now, 2 * DAY);
    }
    write_sealed(
        &other.generation.join(STAMP_NAME),
        &seal(STAMP_MAGIC, b"", &unix_secs(now - 4 * DAY).to_le_bytes()).unwrap(),
    )
    .unwrap();
    set_age(&other.generation.join(STAMP_NAME), now, 4 * DAY);
    cleanup(&root, FP, Limits::DEFAULT, now);
    assert!(pending.exists(), "{:?}", tree(&root));
    // A file of an entry's own name with no header of the cache's is not proven by its name.
    std::fs::write(&entry, b"short").unwrap();
    set_age(&entry, now, 10 * DAY);
    cleanup(&root, FP, Limits::DEFAULT, now);
    assert!(entry.exists());
    std::fs::remove_dir_all(&root).ok();
}

/// A generation directory whose marker names another fingerprint (two fingerprints whose hashes
/// share the directory's name) is refused, not shared.
#[test]
fn a_marker_of_another_fingerprint_is_refused() {
    let root = tmp_root("collision");
    let dir = root.join(generation_name(FP));
    std::fs::create_dir_all(&dir).unwrap();
    write_sealed(
        &dir.join(MARKER_NAME),
        &seal(MARKER_MAGIC, b"", b"another fingerprint").unwrap(),
    )
    .unwrap();
    let e = match RefCache::open_with(&root, FP, Mode::On, Limits::DEFAULT, |_| {}) {
        Ok(_) => panic!("a generation of another fingerprint was opened"),
        Err(e) => format!("{e:#}"),
    };
    assert!(e.contains(REFUSED), "{e}");
    std::fs::remove_dir_all(&root).ok();
}

/// The crates the fingerprint hashes (`build.rs`'s `REFERENCE_CRATES`) are every workspace crate
/// infr-llama's normal and build dependencies reach, except those [`NOT_RUN`] names: a new
/// workspace dependency fails here until someone says which it is.
#[test]
fn the_fingerprint_covers_every_crate_the_reference_can_run() {
    /// Workspace crates infr-llama depends on whose code the CPU reference does not run: the GPU
    /// backends, reached only through `ForcedOn::Vulkan` and the Metal session.
    const NOT_RUN: &[&str] = &["infr-vulkan", "infr-metal"];
    let hashed: Vec<&str> = env!("REFCACHE_CODE_CRATES").split(',').collect();
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut reached = vec!["infr-llama".to_string()];
    let mut next = 0;
    while next < reached.len() {
        let manifest = crates.join(&reached[next]).join("Cargo.toml");
        next += 1;
        let text = std::fs::read_to_string(&manifest)
            .unwrap_or_else(|e| panic!("read {}: {e}", manifest.display()));
        for dep in workspace_deps(&text) {
            if !reached.contains(&dep) {
                reached.push(dep);
            }
        }
    }
    // The walk saw the crates the reference is known to run, so it is not vacuous.
    for known in ["infr-core", "infr-cpu", "infr-plat"] {
        assert!(reached.iter().any(|c| c == known), "{reached:?}");
    }
    let unclassified: Vec<&String> = reached
        .iter()
        .filter(|c| !hashed.contains(&c.as_str()) && !NOT_RUN.contains(&c.as_str()))
        .collect();
    assert!(
        unclassified.is_empty(),
        "infr-llama reaches {unclassified:?}: add each to REFERENCE_CRATES in build.rs, or to \
         NOT_RUN here with the reason the CPU reference does not run it"
    );
    let stale: Vec<&&str> = hashed
        .iter()
        .filter(|c| !reached.iter().any(|r| r == *c))
        .collect();
    assert!(
        stale.is_empty(),
        "REFERENCE_CRATES names {stale:?}, which infr-llama does not reach"
    );
}

/// The `infr-*` crates `text` (a manifest) declares in a normal or build dependency table, any
/// target's; dev dependencies are not the library's. Line-based, like the platform-seam test's
/// reading of the same manifests: `name.workspace = true` and `name = { ... }` are both edges.
fn workspace_deps(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_deps = false;
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            in_deps = line.ends_with("dependencies]") && !line.ends_with("dev-dependencies]");
            continue;
        }
        if !in_deps || !line.starts_with("infr-") {
            continue;
        }
        let name: String = line
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        out.push(name);
    }
    out
}
