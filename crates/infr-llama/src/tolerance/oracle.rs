//! An on-disk cache of the CPU reference the tolerance gate scores against
//! (`docs/plan/17-reference-cache.md`, `docs/gpu-tolerance.md`).
//!
//! The CPU reference is bit-identical across systems, tiers and thread counts, so one commit's
//! reference is the same for every GPU, lane width and tuning mode that is held to it. This
//! module stores each reference run once and serves it to every later gate run that asks the same
//! question of the same code and weights.
//!
//! # The danger
//!
//! A stale entry serving as the oracle: the reference's code or inputs change, the key does not,
//! and a test passes or fails against an old answer. Everything here is built against that:
//!
//! - **The key is everything the run depends on.** The code fingerprint `build.rs` hashes over
//!   the sources, manifests and lockfile of every crate the reference runs and the compiler,
//!   target, profile, `CARGO_CFG_*` values, features and flags names the generation directory;
//!   its marker holds the fingerprint whole and an open with any other is refused. Within it,
//!   each entry's key is stored verbatim and compared on read
//!   ([`infr_core::kernel_cache::unseal`]): a content hash of every model shard's bytes; the
//!   prompt, the step count, the prefill head, how each next token is picked and a digest of any
//!   forced routing; every path of the engine configuration as text; the architecture and OS. A
//!   build with no fingerprint (no sibling sources to hash) cannot open a cache at all.
//! - **The library reads no environment.** A caller hands an opened [`RefCache`] to the cached
//!   constructors ([`super::gate::Reference::cached`], [`super::gate::check_cached`]); the plain
//!   reference functions never read one.
//! - **A canary** under [`Mode::On`]: the first reference of an [`Oracle`] served from the cache
//!   is recomputed (the shortest prompt's, for a gate's greedy runs) and compared bit for bit; a
//!   mismatch fails the run and names the entry. It catches a gap in the key or the fingerprint.
//! - **[`Mode::Verify`]** recomputes every lookup and fails on any stored entry that differs.
//!
//! # On disk
//!
//! Under [`RefCache::default_root`], one directory per code fingerprint (a generation), each with
//! a marker naming its fingerprint, a last-used stamp and one file per entry, all in
//! `infr_core::kernel_cache`'s envelope (magic, version, verbatim key, checksum, atomic durable
//! write). A checksum failure deletes the entry; any other mismatch is a miss. Cleanup
//! (`cleanup`) runs once per process when a cache opens, only by the process holding its lock,
//! and deletes nothing it cannot prove its own by name and header, nothing newer than
//! [`Limits::grace`], no symlink and nothing outside its root.

use super::gate::{argmax, kv_types};
use crate::seam::{ForcedLogits, ForcedOn, PrefillHead, Routing};
use crate::SeamModel;
use anyhow::{anyhow, bail, ensure, Context, Result};
use infr_core::kernel_cache::{
    seal, sealed_header, unseal, write_sealed, Rejected, FORMAT_VERSION, HEADER_LEN,
};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

/// The code fingerprint `build.rs` computed for this build, or `None` when it had no sources to
/// hash.
pub fn fingerprint() -> Option<&'static str> {
    option_env!("REFCACHE_CODE_FINGERPRINT").filter(|f| !f.is_empty())
}

/// What a cache does on a lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Serve a stored entry (with the canary), compute and store a missing one.
    On,
    /// Always compute, fail on a stored entry that differs, store a missing one.
    Verify,
    /// Always compute and store, never read.
    Refresh,
}

/// How old an entry or generation may get, and how large the cache, before the cleanup removes
/// it.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// A generation unused this long, and an entry of the current one, is removed.
    pub max_age: Duration,
    /// Past this many bytes of entries, the least recently used go.
    pub size_cap: u64,
    /// Nothing modified more recently than this is deleted: a concurrent run may be using it.
    pub grace: Duration,
}

impl Limits {
    /// What every cache outside the tests runs with; each field's value is its own.
    pub const DEFAULT: Limits = Limits {
        max_age: Duration::from_secs(3 * 24 * 3600),
        size_cap: 6_000_000_000,
        grace: Duration::from_secs(3600),
    };
}

/// The directory under the user cache directory the cache lives in.
const ROOT_NAME: &str = "tolerance-reference";

/// The first six bytes of every magic here; the seventh names the kind, the eighth its version.
const FAMILY: &[u8; 6] = b"INFRTR";
const ENTRY_MAGIC: [u8; 8] = *b"INFRTRE1";
const MARKER_MAGIC: [u8; 8] = *b"INFRTRG1";
const STAMP_MAGIC: [u8; 8] = *b"INFRTRS1";

const MARKER_NAME: &str = "gen.mark";
const STAMP_NAME: &str = "last-used.stamp";
const LOCK_NAME: &str = "cleanup.lock";
const ENTRY_PREFIX: &str = "e-";
const ENTRY_EXT: &str = "ref";
const GEN_PREFIX: &str = "g-";
/// Hex digits of a generation's directory name: the first half of the SHA-256 of the fingerprint,
/// while the marker inside holds the whole fingerprint, compared on open.
const GEN_HEX: usize = 32;
/// Hex digits of an entry's file name: the whole SHA-256 of its key.
const ENTRY_HEX: usize = 64;

/// An open reference cache: a root directory, this build's generation in it, and what a lookup
/// does. See the module doc.
pub struct RefCache {
    generation: PathBuf,
    mode: Mode,
    log: Box<dyn Fn(&str) + Send + Sync>,
}

impl RefCache {
    /// `<user cache dir>/tolerance-reference`, or `None` when there is no home directory.
    pub fn default_root() -> Option<PathBuf> {
        infr_plat::paths::cache_dir().map(|d| d.join(ROOT_NAME))
    }

    /// Open the cache at `root` for this build's [`fingerprint`]; an error on a build that has
    /// none. `log` gets one line per lookup and one per cleanup that removed anything.
    pub fn open(
        root: &Path,
        mode: Mode,
        log: impl Fn(&str) + Send + Sync + 'static,
    ) -> Result<Self> {
        let fp = fingerprint().ok_or_else(|| {
            anyhow!(
                "this build has no code fingerprint (built without the sibling crates' sources), \
                 so the reference cache cannot tell its entries apart from another build's and \
                 refuses to run"
            )
        })?;
        Self::open_with(root, fp, mode, Limits::DEFAULT, log)
    }

    /// [`open`](Self::open) with the fingerprint and limits given. It exists for the tests, which
    /// inject a fingerprint and limits of their own; anything else must use [`open`](Self::open),
    /// because a fingerprint that is not this build's lets one build's entries serve another's.
    /// The first open of `root` in a process runs the cleanup when it wins the cleanup lock.
    #[doc(hidden)]
    pub fn open_with(
        root: &Path,
        fingerprint: &str,
        mode: Mode,
        limits: Limits,
        log: impl Fn(&str) + Send + Sync + 'static,
    ) -> Result<Self> {
        ensure!(!fingerprint.is_empty(), "an empty code fingerprint");
        std::fs::create_dir_all(root)
            .with_context(|| format!("create the reference cache at {}", root.display()))?;
        if first_open_in_process(root) {
            match infr_plat::lock::FileLock::try_acquire(&root.join(LOCK_NAME)) {
                Ok(Some(_held)) => {
                    let removed = cleanup(root, fingerprint, limits, SystemTime::now());
                    if removed > 0 {
                        log(&format!(
                            "refcache: cleanup removed {removed} files under {}",
                            root.display()
                        ));
                    }
                }
                Ok(None) => {} // another process is cleaning it
                Err(e) => log(&format!(
                    "refcache: no cleanup, the lock at {} could not be taken: {e}",
                    root.display()
                )),
            }
        }
        let generation = root.join(generation_name(fingerprint));
        // Another process's cleanup may remove a generation unused for long between the two
        // steps; the second try makes it again.
        let opened = open_generation(&generation, fingerprint);
        let opened = match opened {
            Err(e) if !is_refusal(&e) => open_generation(&generation, fingerprint),
            other => other,
        };
        opened.with_context(|| format!("open the generation {}", generation.display()))?;
        Ok(Self {
            generation,
            mode,
            log: Box::new(log),
        })
    }

    /// Every entry file of this build's generation, sorted by name.
    pub fn entries(&self) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = std::fs::read_dir(&self.generation)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| matches!(name_kind(p), Some(Kind::Entry)))
            .collect();
        out.sort();
        out
    }

    /// Remove every file and directory under `root` that the cache can prove its own, whatever
    /// its age; the cleanup lock file stays. Returns how many files were removed.
    pub fn clear(root: &Path) -> usize {
        let mut removed = 0;
        let now = SystemTime::now();
        for gen in generations(root) {
            // A temp file proven only by its name may be a write in progress: it goes only once
            // old enough for the cleanup to take it.
            removed += remove_owned(&gen.files, |f| {
                !f.by_name_only || stale_by_name(f, now, Limits::DEFAULT)
            });
            removed += remove_generation_if_bare(&gen, None);
        }
        removed
    }

    fn entry_path(&self, key: &[u8]) -> PathBuf {
        self.generation.join(format!(
            "{ENTRY_PREFIX}{}.{ENTRY_EXT}",
            hex(&Sha256::digest(key))
        ))
    }

    /// The entry stored under `key`, if one is there and whole. A damaged one is deleted.
    fn read(&self, key: &[u8]) -> Result<Option<ForcedLogits>> {
        let path = self.entry_path(key);
        let Ok(data) = std::fs::read(&path) else {
            return Ok(None);
        };
        match unseal(&data, ENTRY_MAGIC, key) {
            Ok(payload) => {
                // The entry's modification time is its last use, which the size cap evicts by.
                // Setting it is a hint for that order only, so a refusal changes nothing else.
                if let Ok(f) = File::options().write(true).open(&path) {
                    let _ = f.set_modified(SystemTime::now());
                }
                decode(payload)
                    .map(Some)
                    .with_context(|| format!("decode the reference cache entry {}", path.display()))
            }
            Err(Rejected::Checksum) => {
                (self.log)(&format!(
                    "refcache: {} failed its checksum and was deleted",
                    path.display()
                ));
                match std::fs::remove_file(&path) {
                    // Another process found the same damaged entry and deleted it first.
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    other => other.with_context(|| {
                        format!("delete the damaged cache entry {}", path.display())
                    })?,
                }
                Ok(None)
            }
            Err(Rejected::Malformed | Rejected::Key) => Ok(None),
        }
    }

    fn write(&self, key: &[u8], run: &ForcedLogits) -> Result<()> {
        let path = self.entry_path(key);
        write_sealed(&path, &seal(ENTRY_MAGIC, key, &encode(run)?)?)
            .with_context(|| format!("write the reference cache entry {}", path.display()))
    }
}

/// The message of an open refused because the generation's marker names another fingerprint.
const REFUSED: &str = "names another fingerprint";

fn is_refusal(e: &anyhow::Error) -> bool {
    e.to_string().contains(REFUSED)
}

/// Make the generation directory, check or write its marker, and stamp it as used now. A marker
/// holding another fingerprint is a refusal: two fingerprints whose hashes share the directory's
/// name, which must not share entries. A marker of an older layout, or a damaged one, is
/// rewritten.
fn open_generation(generation: &Path, fingerprint: &str) -> Result<()> {
    std::fs::create_dir_all(generation)?;
    let marker = generation.join(MARKER_NAME);
    match std::fs::read(&marker) {
        Ok(d) => match unseal(&d, MARKER_MAGIC, b"") {
            Ok(fp) if fp == fingerprint.as_bytes() => {}
            Ok(_) => bail!("the marker {} {REFUSED}", marker.display()),
            Err(_) => write_sealed(&marker, &seal(MARKER_MAGIC, b"", fingerprint.as_bytes())?)?,
        },
        Err(_) => write_sealed(&marker, &seal(MARKER_MAGIC, b"", fingerprint.as_bytes())?)?,
    }
    let now = unix_secs(SystemTime::now());
    write_sealed(
        &generation.join(STAMP_NAME),
        &seal(STAMP_MAGIC, b"", &now.to_le_bytes())?,
    )?;
    Ok(())
}

/// Whether this is the first open of `root` in this process.
fn first_open_in_process(root: &Path) -> bool {
    static OPENED: std::sync::Mutex<Vec<PathBuf>> = std::sync::Mutex::new(Vec::new());
    let mut opened = OPENED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if opened.iter().any(|p| p == root) {
        return false;
    }
    opened.push(root.to_path_buf());
    true
}

fn generation_name(fingerprint: &str) -> String {
    let h = hex(&Sha256::digest(fingerprint.as_bytes()));
    format!("{GEN_PREFIX}{}", &h[..GEN_HEX])
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unix_secs(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// ─── what a reference run is, as bytes ───────────────────────────────────────────

/// A [`ForcedLogits`] of the CPU reference in a defined encoding, field by field, little-endian:
/// `vocab`, `prefill_rows`, the positions, the sequence, the logits' bits, the routing (a tag,
/// then its layers, `n_used`, `n_expert`, ids and scores' bits). An error for a run that built
/// kernels: the CPU builds none, and a kernel name cannot be read back.
fn encode(f: &ForcedLogits) -> Result<Vec<u8>> {
    ensure!(
        f.kernels.is_empty(),
        "a run that built kernels is not a CPU reference run"
    );
    let mut out = Vec::with_capacity(f.logits.len() * 4 + f.seq.len() * 4 + 64);
    let u = |out: &mut Vec<u8>, v: u64| out.extend_from_slice(&v.to_le_bytes());
    u(&mut out, f.vocab as u64);
    u(&mut out, f.prefill_rows as u64);
    u(&mut out, f.positions.len() as u64);
    for &p in &f.positions {
        u(&mut out, p as u64);
    }
    u(&mut out, f.seq.len() as u64);
    out.extend(f.seq.iter().flat_map(|t| t.to_le_bytes()));
    u(&mut out, f.logits.len() as u64);
    out.extend(f.logits.iter().flat_map(|x| x.to_bits().to_le_bytes()));
    match &f.routing {
        None => out.push(0),
        Some(r) => {
            out.push(1);
            encode_routing(&mut out, r);
        }
    }
    Ok(out)
}

fn encode_routing(out: &mut Vec<u8>, r: &Routing) {
    let u = |out: &mut Vec<u8>, v: u64| out.extend_from_slice(&v.to_le_bytes());
    u(out, r.layers.len() as u64);
    for &l in &r.layers {
        u(out, l as u64);
    }
    u(out, r.n_used as u64);
    u(out, r.n_expert as u64);
    u(out, r.ids.len() as u64);
    out.extend(r.ids.iter().flat_map(|t| t.to_le_bytes()));
    u(out, r.scores.len() as u64);
    out.extend(r.scores.iter().flat_map(|x| x.to_bits().to_le_bytes()));
}

/// [`encode`]'s inverse. An error for bytes it did not write: a short field, a length past the
/// end, or anything left over.
fn decode(b: &[u8]) -> Result<ForcedLogits> {
    let mut r = Reader(b);
    let vocab = r.usize()?;
    let prefill_rows = r.usize()?;
    let n = r.usize()?;
    let positions = (0..n).map(|_| r.usize()).collect::<Result<_>>()?;
    let seq = r.u32s()?;
    let logits = r.f32s()?;
    let routing = match r.take(1)?[0] {
        0 => None,
        1 => {
            let n = r.usize()?;
            let layers = (0..n).map(|_| r.usize()).collect::<Result<_>>()?;
            Some(Routing {
                layers,
                n_used: r.usize()?,
                n_expert: r.usize()?,
                ids: r.u32s()?,
                scores: r.f32s()?,
            })
        }
        t => bail!("routing tag {t}"),
    };
    ensure!(r.0.is_empty(), "{} bytes past the end", r.0.len());
    Ok(ForcedLogits {
        positions,
        logits,
        vocab,
        prefill_rows,
        seq,
        kernels: Vec::new(),
        routing,
    })
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(n <= self.0.len(), "a field runs past the end");
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn usize(&mut self) -> Result<usize> {
        let v = u64::from_le_bytes(self.take(8)?.try_into().unwrap());
        usize::try_from(v).map_err(|_| anyhow!("{v} does not fit a usize"))
    }

    fn words(&mut self) -> Result<impl Iterator<Item = [u8; 4]> + 'a> {
        let n = self.usize()?;
        let bytes = self.take(
            n.checked_mul(4)
                .ok_or_else(|| anyhow!("a length overflows"))?,
        )?;
        Ok(bytes.as_chunks::<4>().0.iter().copied())
    }

    fn u32s(&mut self) -> Result<Vec<u32>> {
        Ok(self.words()?.map(u32::from_le_bytes).collect())
    }

    fn f32s(&mut self) -> Result<Vec<f32>> {
        Ok(self
            .words()?
            .map(|w| f32::from_bits(u32::from_le_bytes(w)))
            .collect())
    }
}

// ─── the oracle: a model's reference runs, through the cache ─────────────────────

/// How a reference run picks the token it feeds after each decode row.
#[derive(Clone, Copy, Debug)]
pub enum Pick<'a> {
    /// Its own argmax: the reference's greedy run.
    Greedy,
    /// The given tokens, one per step: the reference forced along another run's tokens.
    Along(&'a [u32]),
}

/// One CPU reference run, as [`SeamModel::forced_logits`] takes it on
/// [`ForcedOn::CpuReference`]: the engine configuration the model was loaded with, with the KV
/// format pair `kv` set when given.
#[derive(Clone, Copy, Debug)]
pub struct Query<'a> {
    pub prompt: &'a [u32],
    pub steps: usize,
    pub head: PrefillHead,
    pub pick: Pick<'a>,
    pub route: Option<&'a Routing>,
    pub kv: Option<(&'a str, &'a str)>,
}

impl<'a> Query<'a> {
    /// The greedy run of `steps` tokens after `prompt`, every prefill row scored.
    pub fn greedy(prompt: &'a [u32], steps: usize) -> Self {
        Query {
            prompt,
            steps,
            head: PrefillHead::All,
            pick: Pick::Greedy,
            route: None,
            kv: None,
        }
    }

    /// Forced along `run`'s tokens (its prompt, then the tokens it fed), on `route`'s expert
    /// selections when given.
    pub fn along(
        run: &'a ForcedLogits,
        head: PrefillHead,
        route: Option<&'a Routing>,
        kv: Option<(&'a str, &'a str)>,
    ) -> Self {
        let steps = run.positions.len() - run.prefill_rows;
        let p = run.seq.len() - steps;
        Query {
            prompt: &run.seq[..p],
            steps,
            head,
            pick: Pick::Along(&run.seq[p..]),
            route,
            kv,
        }
    }

    /// The run itself, never through a cache.
    pub fn compute(&self, model: &SeamModel) -> Result<ForcedLogits> {
        if let Pick::Along(t) = self.pick {
            ensure!(
                t.len() >= self.steps,
                "{} tokens to force along for {} steps",
                t.len(),
                self.steps
            );
        }
        let pick = self.pick;
        model.forced_logits(
            ForcedOn::CpuReference,
            |c| {
                if let Some((k, v)) = self.kv {
                    kv_types(c, k, v);
                }
            },
            self.prompt,
            self.steps,
            self.head,
            self.route,
            move |s, row| match pick {
                Pick::Greedy => argmax(row),
                Pick::Along(t) => t[s],
            },
        )
    }

    /// The query in a few words, for the log.
    fn summary(&self) -> String {
        format!(
            "{} prompt tokens, {} steps, head {:?}, {}, kv {}, {} routing",
            self.prompt.len(),
            self.steps,
            self.head,
            match self.pick {
                Pick::Greedy => "greedy",
                Pick::Along(_) => "forced along given tokens",
            },
            self.kv
                .map_or("default".to_string(), |(k, v)| format!("{k}/{v}")),
            if self.route.is_some() {
                "forced"
            } else {
                "its own"
            },
        )
    }
}

/// A model's CPU reference runs, served from a [`RefCache`] when it has one and computed
/// otherwise, with the seconds spent on each.
pub struct Oracle<'a> {
    model: &'a SeamModel,
    /// The cache and the lines of the key that name this model's weights.
    cache: Option<(&'a RefCache, String)>,
    /// Whether a reference served from the cache has been recomputed and compared yet.
    canaried: Cell<bool>,
    /// The time spent hashing the weights, which [`Oracle::new`] does once.
    hashed: Duration,
    computed: Cell<Duration>,
    read: Cell<Duration>,
}

/// Where an [`Oracle`]'s time went, in seconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct Seconds {
    /// Hashing the model's weights for the cache key.
    pub hashed: f64,
    /// Computing reference runs: misses, verifies and the canary.
    pub computed: f64,
    /// Reading entries the cache served.
    pub read: f64,
}

impl<'a> Oracle<'a> {
    /// `model`'s oracle; with a cache, its weights are hashed here, once. An error for a model
    /// carrying a planted bug, whose reference is not its file's.
    pub fn new(model: &'a SeamModel, cache: Option<&'a RefCache>) -> Result<Self> {
        let t0 = Instant::now();
        let cache = match cache {
            None => None,
            Some(c) => {
                #[cfg(feature = "mutations")]
                ensure!(
                    model.config().planted.is_none(),
                    "a model with a planted bug is never served from the reference cache"
                );
                Some((c, weights_lines(model).context("hash the model's weights")?))
            }
        };
        Ok(Self {
            model,
            cache,
            canaried: Cell::new(false),
            hashed: t0.elapsed(),
            computed: Cell::new(Duration::ZERO),
            read: Cell::new(Duration::ZERO),
        })
    }

    /// The run `q` asks for. Served from the cache under [`Mode::On`] when stored, the first
    /// such run recomputed and compared bit for bit (the canary).
    pub fn run(&self, q: &Query) -> Result<ForcedLogits> {
        let (run, served) = self.lookup(q)?;
        if served && !self.canaried.get() {
            self.canary(q, &run)?;
        }
        Ok(run)
    }

    /// [`run`](Self::run) without the canary, and whether the run was served from the cache:
    /// for a caller that canaries one of several lookups itself.
    pub(crate) fn lookup(&self, q: &Query) -> Result<(ForcedLogits, bool)> {
        let Some((cache, weights)) = &self.cache else {
            return Ok((self.compute(q)?, false));
        };
        let key = self.key(weights, q);
        let name = cache.entry_path(&key);
        let name = name
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into());
        let log = |what: &str, secs: f64| {
            (cache.log)(&format!(
                "refcache {what} {name} ({}) {secs:.2}s",
                q.summary()
            ))
        };
        if cache.mode == Mode::On {
            let t0 = Instant::now();
            if let Some(run) = cache.read(&key)? {
                let dt = t0.elapsed();
                self.read.set(self.read.get() + dt);
                log("hit", dt.as_secs_f64());
                return Ok((run, true));
            }
        }
        let t0 = Instant::now();
        let run = self.compute(q)?;
        let dt = t0.elapsed().as_secs_f64();
        if cache.mode == Mode::Verify {
            if let Some(stored) = cache.read(&key)? {
                ensure!(
                    same_bits(&stored, &run)?,
                    "reference cache verify: the entry {} ({}) is not what the CPU reference \
                     computes now: a gap in the cache key or the code fingerprint. Clear the \
                     cache (INFR_TOL_REFCACHE=clear) and report it",
                    cache.entry_path(&key).display(),
                    q.summary()
                );
                log("verified", dt);
                return Ok((run, false));
            }
        }
        match cache.write(&key, &run) {
            Ok(()) => log("stored", dt),
            Err(e) => log(&format!("miss, not stored ({e:#}):"), dt),
        }
        Ok((run, false))
    }

    /// Recompute `q` and fail unless the result is `served` bit for bit.
    pub(crate) fn canary(&self, q: &Query, served: &ForcedLogits) -> Result<()> {
        self.canaried.set(true);
        let fresh = self.compute(q)?;
        let Some((cache, weights)) = &self.cache else {
            return Ok(());
        };
        let entry = cache.entry_path(&self.key(weights, q));
        ensure!(
            same_bits(served, &fresh)?,
            "reference cache canary: the entry {} ({}) is not what the CPU reference computes \
             now: a gap in the cache key or the code fingerprint. Clear the cache \
             (INFR_TOL_REFCACHE=clear) and report it",
            entry.display(),
            q.summary()
        );
        (cache.log)(&format!(
            "refcache canary {} ({}): recomputed, bit-identical",
            entry
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into()),
            q.summary()
        ));
        Ok(())
    }

    /// Where this oracle's time went so far.
    pub fn seconds(&self) -> Seconds {
        Seconds {
            hashed: self.hashed.as_secs_f64(),
            computed: self.computed.get().as_secs_f64(),
            read: self.read.get().as_secs_f64(),
        }
    }

    fn compute(&self, q: &Query) -> Result<ForcedLogits> {
        let t0 = Instant::now();
        let run = q.compute(self.model).context("cpu reference run");
        self.computed.set(self.computed.get() + t0.elapsed());
        run
    }

    /// Everything the run `q` depends on, as text: see the module doc.
    ///
    /// The code fingerprint is not a line of it: it names the generation directory the entry is
    /// in, and that directory's marker holds it whole and refuses any other.
    fn key(&self, weights: &str, q: &Query) -> Vec<u8> {
        use std::fmt::Write;
        let ids = |t: &[u32]| t.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let mut k = String::new();
        let _ = writeln!(k, "infr tolerance reference");
        let _ = writeln!(
            k,
            "machine {} {}",
            std::env::consts::ARCH,
            std::env::consts::OS
        );
        k.push_str(weights);
        let _ = writeln!(k, "backend cpu-reference");
        let _ = writeln!(k, "prompt {}", ids(q.prompt));
        let _ = writeln!(k, "steps {}", q.steps);
        let _ = writeln!(k, "head {:?}", q.head);
        match q.pick {
            Pick::Greedy => k.push_str("pick greedy\n"),
            Pick::Along(t) => {
                let _ = writeln!(k, "pick along {}", ids(&t[..q.steps.min(t.len())]));
            }
        }
        match q.route {
            None => k.push_str("route none\n"),
            Some(r) => {
                let mut bytes = Vec::new();
                encode_routing(&mut bytes, r);
                let _ = writeln!(k, "route {}", hex(&Sha256::digest(&bytes)));
            }
        }
        let mut ec = (**self.model.engine_config()).clone();
        if let Some((kk, v)) = q.kv {
            kv_types(&mut ec, kk, v);
        }
        k.push_str("config\n");
        for path in crate::EngineConfig::all_paths() {
            match ec.get_path(&path) {
                Some(v) => {
                    let _ = writeln!(k, "{path}={v:?}");
                }
                None => {
                    let _ = writeln!(k, "{path}!");
                }
            }
        }
        // Every field, including any a path does not reach: a difference here can only cost a
        // miss, never serve another configuration's run.
        let _ = writeln!(k, "config-debug {ec:?}");
        k.into_bytes()
    }
}

/// Whether two runs are the same, bit for bit, in every field the cache stores.
fn same_bits(a: &ForcedLogits, b: &ForcedLogits) -> Result<bool> {
    Ok(encode(a)? == encode(b)?)
}

/// One key line per shard of `model`'s file: its index, length and a SHA-256 over its bytes.
fn weights_lines(model: &SeamModel) -> Result<String> {
    let mut out = String::new();
    for (i, shard) in model.gguf().shard_handles().iter().enumerate() {
        let len = shard.stamp.len;
        let digest =
            hash_file(shard.file, len).with_context(|| format!("read {}", shard.path.display()))?;
        out.push_str(&format!("shard {i} {len} {}\n", hex(&digest)));
    }
    Ok(out)
}

/// The bytes a chunk of [`hash_file`] covers.
const HASH_CHUNK: u64 = 16 << 20;

/// SHA-256 over the length and the SHA-256 of each [`HASH_CHUNK`] of `file`'s first `len`
/// bytes, the chunks hashed in parallel.
fn hash_file(file: &File, len: u64) -> std::io::Result<[u8; 32]> {
    use rayon::prelude::*;
    let chunks = len.div_ceil(HASH_CHUNK);
    let digests: Vec<[u8; 32]> = (0..chunks)
        .into_par_iter()
        .map_init(Vec::new, |buf, i| {
            let off = i * HASH_CHUNK;
            buf.resize((len - off).min(HASH_CHUNK) as usize, 0);
            infr_plat::fileio::read_exact_at(file, buf, off)?;
            Ok(Sha256::digest(&buf[..]).into())
        })
        .collect::<std::io::Result<_>>()?;
    let mut h = Sha256::new();
    h.update(len.to_le_bytes());
    for d in digests {
        h.update(d);
    }
    Ok(h.finalize().into())
}

// ─── cleanup ─────────────────────────────────────────────────────────────────────

/// What a file of the cache is, by its name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Entry,
    Marker,
    Stamp,
    /// A write in progress, or one a dead writer left: `<stem>.tmp.<pid>.<seq>`, the temp name
    /// [`write_sealed`] gives, of a file of the kind inside.
    Temp(&'static Kind),
}

impl Kind {
    fn magic(self) -> [u8; 8] {
        match self {
            Kind::Entry => ENTRY_MAGIC,
            Kind::Marker => MARKER_MAGIC,
            Kind::Stamp => STAMP_MAGIC,
            Kind::Temp(k) => k.magic(),
        }
    }
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The kind of cache file `path` names, by its name alone.
fn name_kind(path: &Path) -> Option<Kind> {
    let name = path.file_name()?.to_str()?;
    let entry_stem = |s: &str| {
        s.strip_prefix(ENTRY_PREFIX)
            .is_some_and(|h| is_hex(h, ENTRY_HEX))
    };
    if let Some((stem, rest)) = name.split_once(".tmp.") {
        let (pid, seq) = rest.split_once('.')?;
        let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
        if !digits(pid) || !digits(seq) {
            return None;
        }
        let stem_of = |file: &str| file.rsplit_once('.').map_or(file, |(s, _)| s).to_string();
        return if entry_stem(stem) {
            Some(Kind::Temp(&Kind::Entry))
        } else if stem == stem_of(MARKER_NAME) {
            Some(Kind::Temp(&Kind::Marker))
        } else if stem == stem_of(STAMP_NAME) {
            Some(Kind::Temp(&Kind::Stamp))
        } else {
            None
        };
    }
    match name {
        MARKER_NAME => Some(Kind::Marker),
        STAMP_NAME => Some(Kind::Stamp),
        _ => name
            .strip_suffix(&format!(".{ENTRY_EXT}"))
            .filter(|s| entry_stem(s))
            .map(|_| Kind::Entry),
    }
}

/// A file the cache can prove its own: its name is one the cache gives, it is a regular file and
/// not a symlink, and its header carries a magic of the cache's family for that kind.
#[derive(Debug)]
struct Owned {
    path: PathBuf,
    kind: Kind,
    bytes: u64,
    modified: SystemTime,
    /// The header is the current magic and envelope version, not an older layout's.
    current: bool,
    /// A temp file whose header is not the cache's (a writer that died before writing it): its
    /// name is the only proof, which [`stale_by_name`] accepts only once it is old.
    by_name_only: bool,
}

/// Whether a file proven only by its name is old enough to delete: past the grace period and the
/// age limit, long after any writer that could still be filling it.
fn stale_by_name(f: &Owned, now: SystemTime, limits: Limits) -> bool {
    older_than(f, now, limits.grace) && older_than(f, now, limits.max_age)
}

fn own_file(path: &Path) -> Option<Owned> {
    let kind = name_kind(path)?;
    let md = std::fs::symlink_metadata(path).ok()?;
    if !md.file_type().is_file() {
        return None;
    }
    let mut head = Vec::with_capacity(HEADER_LEN);
    File::open(path)
        .ok()?
        .take(HEADER_LEN as u64)
        .read_to_end(&mut head)
        .ok()?;
    let want = kind.magic();
    let header = sealed_header(&head).filter(|(magic, _)| {
        magic[..FAMILY.len()] == FAMILY[..] && magic[FAMILY.len()] == want[FAMILY.len()]
    });
    let (current, by_name_only) = match header {
        Some((magic, version)) => (magic == want && version == FORMAT_VERSION, false),
        None if matches!(kind, Kind::Temp(_)) => (true, true),
        None => return None,
    };
    Some(Owned {
        path: path.to_path_buf(),
        kind,
        bytes: md.len(),
        modified: md.modified().ok()?,
        current,
        by_name_only,
    })
}

/// A generation directory the cache can prove its own: a directory (not a symlink) named as the
/// cache names them, holding a marker of its own; with every file of its own in it.
struct Generation {
    dir: PathBuf,
    name: String,
    files: Vec<Owned>,
    /// When it was last opened: its stamp's time, or else its marker's modification time.
    last_used: SystemTime,
}

fn generations(root: &Path) -> Vec<Generation> {
    let Ok(dirs) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in dirs.flatten() {
        let dir = e.path();
        let Some(name) = e.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if !name
            .strip_prefix(GEN_PREFIX)
            .is_some_and(|h| is_hex(h, GEN_HEX))
        {
            continue;
        }
        if !std::fs::symlink_metadata(&dir).is_ok_and(|m| m.file_type().is_dir()) {
            continue;
        }
        let Some(marker) = own_file(&dir.join(MARKER_NAME)) else {
            continue;
        };
        let files: Vec<Owned> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|f| own_file(&f.path()))
            .collect();
        let stamped = std::fs::read(dir.join(STAMP_NAME)).ok().and_then(|d| {
            let secs = unseal(&d, STAMP_MAGIC, b"").ok()?.try_into().ok()?;
            Some(SystemTime::UNIX_EPOCH + Duration::from_secs(u64::from_le_bytes(secs)))
        });
        out.push(Generation {
            dir,
            name,
            last_used: stamped.unwrap_or(marker.modified),
            files,
        });
    }
    out
}

/// Remove each of `files` that `pick` accepts, one by one; the count removed.
fn remove_owned<'a>(
    files: impl IntoIterator<Item = &'a Owned>,
    pick: impl Fn(&Owned) -> bool,
) -> usize {
    files
        .into_iter()
        .filter(|f| f.kind != Kind::Marker && pick(f))
        .filter(|f| std::fs::remove_file(&f.path).is_ok())
        .count()
}

/// Remove `gen`'s marker and directory once nothing else is left in it, the marker only when
/// older than `grace` (`None`: whatever its age); the count of files removed.
fn remove_generation_if_bare(gen: &Generation, grace: Option<(SystemTime, Duration)>) -> usize {
    let left: Vec<PathBuf> = std::fs::read_dir(&gen.dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .collect();
    let marker = gen.dir.join(MARKER_NAME);
    if left.iter().any(|p| *p != marker) {
        return 0;
    }
    let Some(m) = own_file(&marker) else {
        return 0;
    };
    if grace.is_some_and(|(now, g)| !older_than(&m, now, g)) {
        return 0;
    }
    let removed = usize::from(std::fs::remove_file(&marker).is_ok());
    // Only an empty directory goes: `remove_dir` refuses any other.
    let _ = std::fs::remove_dir(&gen.dir);
    removed
}

fn older_than(f: &Owned, now: SystemTime, age: Duration) -> bool {
    now.duration_since(f.modified).is_ok_and(|d| d > age)
}

/// The cache's cleanup at `root` for the generation of `fingerprint`, as of `now`, each step
/// deleting only files [`own_file`] proves the cache's and that are older than `limits.grace`:
///
/// 1. A generation other than the current one, unused for longer than `limits.max_age`, is
///    removed: its files one by one, then its marker and directory when nothing else is left.
/// 2. Entries of the current generation (and temp files a dead writer left) unused that long are
///    removed.
/// 3. While the entries of every generation take more than `limits.size_cap` bytes, the least
///    recently used is removed.
/// 4. Files of an older magic or envelope version are removed.
///
/// Returns how many files were removed.
pub(crate) fn cleanup(root: &Path, fingerprint: &str, limits: Limits, now: SystemTime) -> usize {
    let current = generation_name(fingerprint);
    let fresh = |f: &Owned| !older_than(f, now, limits.grace);
    // Whether a step may delete `f` at all: never within the grace period, and a file proven
    // only by its name only once it is stale.
    let deletable = |f: &Owned| !fresh(f) && (!f.by_name_only || stale_by_name(f, now, limits));
    let mut removed = 0;
    // Rule 4 first, so an old layout never counts against the cap.
    for gen in generations(root) {
        removed += remove_owned(&gen.files, |f| !f.current && deletable(f));
    }
    for gen in generations(root) {
        let unused = now
            .duration_since(gen.last_used)
            .is_ok_and(|d| d > limits.max_age);
        if gen.name != current && unused {
            removed += remove_owned(&gen.files, deletable);
            removed += remove_generation_if_bare(&gen, Some((now, limits.grace)));
        } else if gen.name == current {
            removed += remove_owned(&gen.files, |f| {
                matches!(f.kind, Kind::Entry | Kind::Temp(_))
                    && older_than(f, now, limits.max_age)
                    && deletable(f)
            });
        }
    }
    let gens = generations(root);
    let mut pool: Vec<&Owned> = gens
        .iter()
        .flat_map(|g| &g.files)
        .filter(|f| matches!(f.kind, Kind::Entry | Kind::Temp(_)))
        .collect();
    pool.sort_by_key(|f| f.modified);
    let mut total: u64 = pool.iter().map(|f| f.bytes).sum();
    for f in pool {
        if total <= limits.size_cap || fresh(f) {
            break;
        }
        if !deletable(f) {
            continue;
        }
        if std::fs::remove_file(&f.path).is_ok() {
            total -= f.bytes;
            removed += 1;
        }
    }
    removed
}

#[cfg(test)]
mod tests;
