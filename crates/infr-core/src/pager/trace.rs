//! Pager touch traces (`paging.trace`): record every call a [`Pager`] serves, per pool, and
//! replay it to the exact outcome.
//!
//! The recorder lives INSIDE [`Pager`] because that is the one place every tier's calls pass
//! through — the VRAM arenas, the DRAM arenas and the CPU backend's pools all drive one. A pool is
//! given a recorder once, before its first call ([`Pager::trace_to`]); an untraced pager pays one
//! `Option` test per call and nothing else (the recording path is out of line and `#[cold]`).
//!
//! # Format (`v1`)
//! Line-oriented ASCII, one file per run holding every pool:
//!
//! ```text
//! # infr pager trace v1
//! pool <p> <tier> <n_slots> <n_blocks> <label...>
//! <p> <op> [<args>]
//! ```
//!
//! `<p>` is a pool number unique within the process; `<tier>` is `vram` or `dram`; `<label>` is
//! the rest of the line. A pool's header precedes its events, and its events are in the order its
//! pager served them. Pools buffer separately, so different pools' events interleave in chunks:
//! the cross-pool order is NOT recorded. The ops, each followed by its block id where it has one:
//!
//! | op | call | outcome |
//! |----|------|---------|
//! | `b` | [`Pager::begin_batch`] | — |
//! | `t <id>` | [`Pager::touch`] | `h <slot>` or `m <slot> <evicted or ->` |
//! | `c <id>` | [`Pager::touch_cold`] | as `t` |
//! | `s <id>` | [`Pager::schedule`] | as `t` |
//! | `r <id> <m or c>` | [`Pager::resolve_and_pin`] with [`Insert::Mru`] / [`Insert::Cold`] | as `t`, or `-` when refused |
//! | `q <id>` | [`Pager::pin_if_resident`] | `<slot>` or `-` |
//! | `p <id>` | [`Pager::repin`] | `<slot>` or `-` |
//! | `u <id>` | [`Pager::unpin`] | — |
//! | `e <id>` | [`Pager::evict`] | `<slot>` or `-` |
//!
//! A call that returns something may carry `?` instead of its outcome: "not recorded", which is
//! how a generated trace ([`PoolTrace::write`] of a pool with no outcomes) is written. A pool's
//! events are either all recorded or all `?`.
//!
//! A pager's behaviour is a function of its slot count and this call sequence alone, so
//! [`replay`] into a fresh pager of the header's `n_slots` reproduces every outcome, and with
//! them the hit, miss and eviction counts `paging.stats` printed for the pool.
//!
//! Every line the recorder writes ends in a newline, and after one failed write the file takes no
//! more, so a write cut short (a full disk) can damage only the last line. [`parse`] drops a final
//! line with no newline and reports it in [`Parsed::truncated`] rather than rejecting the file.
//!
//! # What a trace does not see
//! Only calls into the [`Pager`]. A DRAM tier used through `HostPager::fill` (the dense host tier
//! under a GPU) decides most of its traffic outside the pager: a block it streams straight to the
//! caller because the arena is full, or has not missed before, makes no pager call at all, and a
//! resident one is served by a `pin_if_resident` probe. Such a pool's trace therefore omits most of
//! its disk reads; its `paging.stats` line (the pager's counters) still matches the replay, but
//! `reads` does not. The MoE host tier (`HostPager::pin`) and the CPU backend's pools are pager
//! calls throughout.

use super::{BlockId, Insert, Pager, PagerStats, Resolution};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// The first line of every trace file.
pub const MAGIC: &str = "# infr pager trace v1";

/// Bytes a pool buffers before taking the file lock to hand them over: large enough that the lock
/// is taken rarely on a busy pool, small enough not to matter per pool.
const POOL_BUFFER_BYTES: usize = 64 * 1024;

/// Which tier a traced pool caches for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// A GPU arena (the MoE expert pools, the dense streaming pools).
    Vram,
    /// A host arena (`crate::hostpager::HostPager`), under a GPU tier or under the CPU backend.
    Dram,
}

impl Tier {
    fn name(self) -> &'static str {
        match self {
            Tier::Vram => "vram",
            Tier::Dram => "dram",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "vram" => Some(Tier::Vram),
            "dram" => Some(Tier::Dram),
            _ => None,
        }
    }
}

/// What a trace header says about a pool beyond its slot count, which the pager supplies itself.
#[derive(Debug, Clone)]
pub struct PoolInfo {
    pub tier: Tier,
    /// Free text naming the pool in its run (role, size class, ...). Line breaks are replaced.
    pub label: String,
    /// How many distinct blocks the pool can be asked for.
    pub n_blocks: usize,
}

/// One pager entry point, as recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Call {
    BeginBatch,
    Touch(BlockId),
    TouchCold(BlockId),
    Schedule(BlockId),
    ResolveAndPin(BlockId, Insert),
    PinIfResident(BlockId),
    Repin(BlockId),
    Unpin(BlockId),
    Evict(BlockId),
}

/// What a [`Call`] returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// [`Call::BeginBatch`] and [`Call::Unpin`], which return nothing.
    Done,
    /// A resolution: every `touch` flavour, and [`Call::ResolveAndPin`] when it found a slot.
    Resolved(Resolution),
    /// [`Call::ResolveAndPin`] found every resident block protected.
    Refused,
    /// [`Call::PinIfResident`], [`Call::Repin`] and [`Call::Evict`]: the slot, or `None` when the
    /// block was not resident.
    Slot(Option<u32>),
}

impl Call {
    /// Serve this call on `p` — the replay half of the trace.
    pub fn apply(self, p: &mut Pager) -> Outcome {
        match self {
            Call::BeginBatch => {
                p.begin_batch();
                Outcome::Done
            }
            Call::Touch(id) => Outcome::Resolved(p.touch(id)),
            Call::TouchCold(id) => Outcome::Resolved(p.touch_cold(id)),
            Call::Schedule(id) => Outcome::Resolved(p.schedule(id)),
            Call::ResolveAndPin(id, insert) => match p.resolve_and_pin(id, insert) {
                Some(r) => Outcome::Resolved(r),
                None => Outcome::Refused,
            },
            Call::PinIfResident(id) => Outcome::Slot(p.pin_if_resident(id)),
            Call::Repin(id) => Outcome::Slot(p.repin(id)),
            Call::Unpin(id) => {
                p.unpin(id);
                Outcome::Done
            }
            Call::Evict(id) => Outcome::Slot(p.evict(id)),
        }
    }

    /// The block this call names, if it names one.
    pub fn block(self) -> Option<BlockId> {
        match self {
            Call::BeginBatch => None,
            Call::Touch(id)
            | Call::TouchCold(id)
            | Call::Schedule(id)
            | Call::ResolveAndPin(id, _)
            | Call::PinIfResident(id)
            | Call::Repin(id)
            | Call::Unpin(id)
            | Call::Evict(id) => Some(id),
        }
    }
}

/// Write one event line. Shared by the recorder and [`PoolTrace::write`], so the two cannot
/// disagree about the format the parser reads.
/// `outcome` is `None` for a call whose outcome was not recorded (written `?`).
fn write_event(out: &mut Vec<u8>, pool: u32, call: Call, outcome: Option<Outcome>) {
    // Writing into a `Vec` cannot fail.
    let w = &mut *out;
    let _ = match call {
        Call::BeginBatch => write!(w, "{pool} b"),
        Call::Touch(id) => write!(w, "{pool} t {id}"),
        Call::TouchCold(id) => write!(w, "{pool} c {id}"),
        Call::Schedule(id) => write!(w, "{pool} s {id}"),
        Call::ResolveAndPin(id, Insert::Mru) => write!(w, "{pool} r {id} m"),
        Call::ResolveAndPin(id, Insert::Cold) => write!(w, "{pool} r {id} c"),
        Call::PinIfResident(id) => write!(w, "{pool} q {id}"),
        Call::Repin(id) => write!(w, "{pool} p {id}"),
        Call::Unpin(id) => write!(w, "{pool} u {id}"),
        Call::Evict(id) => write!(w, "{pool} e {id}"),
    };
    let returns = !matches!(call, Call::BeginBatch | Call::Unpin(_));
    let _ = match outcome {
        None if returns => write!(w, " ?"),
        None | Some(Outcome::Done) => Ok(()),
        Some(Outcome::Resolved(Resolution::Hit { slot })) => write!(w, " h {slot}"),
        Some(Outcome::Resolved(Resolution::Miss {
            slot,
            evicted: Some(e),
        })) => write!(w, " m {slot} {e}"),
        Some(Outcome::Resolved(Resolution::Miss {
            slot,
            evicted: None,
        })) => write!(w, " m {slot} -"),
        Some(Outcome::Refused | Outcome::Slot(None)) => write!(w, " -"),
        Some(Outcome::Slot(Some(slot))) => write!(w, " {slot}"),
    };
    w.push(b'\n');
}

fn write_header(out: &mut Vec<u8>, pool: u32, info: &PoolInfo, n_slots: usize) {
    let label: String = info
        .label
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    let _ = writeln!(
        out,
        "pool {pool} {} {n_slots} {} {label}",
        info.tier.name(),
        info.n_blocks
    );
}

/// The file one run's pools write into. Opened through [`TraceFile::open`], which hands every pool
/// of the process naming the same path the same file.
pub struct TraceFile {
    out: Mutex<Sink>,
}

struct Sink {
    file: BufWriter<File>,
    /// Set by the first failed write or flush. Nothing is written after it, so a short write can
    /// only damage the last line of the file, which [`parse`] tolerates.
    failed: bool,
}

impl Sink {
    fn guarded(
        &mut self,
        op: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>,
    ) -> io::Result<()> {
        if self.failed {
            return Err(io::Error::other(
                "an earlier write to this pager trace failed; it takes no more",
            ));
        }
        let r = op(&mut self.file);
        self.failed = r.is_err();
        r
    }
}

/// Every trace file this process has opened, so pools built in different crates share one file
/// instead of truncating each other's. A path opened before and since dropped is reopened for
/// APPEND: a server that reloads a model keeps the earlier model's pools.
static OPEN: Mutex<Vec<(PathBuf, Weak<TraceFile>)>> = Mutex::new(Vec::new());

/// Pool numbers, unique per process so pools stay distinct across a reopened file.
static NEXT_POOL: AtomicU32 = AtomicU32::new(0);

impl TraceFile {
    /// The trace file at `path`, shared with every other pool of this process that opened it.
    /// The first open in a process truncates the file and writes [`MAGIC`].
    pub fn open(path: &Path) -> io::Result<Arc<TraceFile>> {
        let mut open = OPEN.lock().unwrap_or_else(|e| e.into_inner());
        let seen = open.iter().position(|(p, _)| p == path);
        if let Some(live) = seen.and_then(|i| open[i].1.upgrade()) {
            return Ok(live);
        }
        let file = match seen {
            Some(_) => OpenOptions::new().append(true).open(path)?,
            None => {
                let mut f = File::create(path)?;
                writeln!(f, "{MAGIC}")?;
                f
            }
        };
        let tf = Arc::new(TraceFile {
            out: Mutex::new(Sink {
                file: BufWriter::new(file),
                failed: false,
            }),
        });
        match seen {
            Some(i) => open[i].1 = Arc::downgrade(&tf),
            None => open.push((path.to_path_buf(), Arc::downgrade(&tf))),
        }
        Ok(tf)
    }

    /// [`Self::open`] for a `paging.trace` setting: `None` when it is unset, and an error that
    /// names the key and the path when the file cannot be opened.
    pub fn for_config(path: Option<&Path>) -> crate::Result<Option<Arc<TraceFile>>> {
        path.map(|p| {
            Self::open(p).map_err(|e| {
                crate::Error::backend(format!("paging.trace: cannot write {}: {e}", p.display()))
            })
        })
        .transpose()
    }

    fn write(&self, bytes: &[u8]) -> io::Result<()> {
        self.out
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .guarded(|f| f.write_all(bytes))
    }

    /// Push everything handed to this file so far to the OS.
    pub fn flush(&self) -> io::Result<()> {
        self.out
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .guarded(|f| f.flush())
    }
}

/// One pool's recording state, owned by its [`Pager`].
pub(crate) struct Recorder {
    file: Arc<TraceFile>,
    pool: u32,
    buf: Vec<u8>,
    /// The first write that failed. Recording stops there, since a trace with a hole in it would
    /// replay to different counts; [`Self::flush`] reports it.
    failed: Option<io::Error>,
}

impl Recorder {
    /// A recorder for a new pool of `n_slots` slots, its header already written to `file`.
    pub(crate) fn new(file: &Arc<TraceFile>, info: &PoolInfo, n_slots: usize) -> io::Result<Self> {
        let pool = NEXT_POOL.fetch_add(1, Ordering::Relaxed);
        let mut header = Vec::new();
        write_header(&mut header, pool, info, n_slots);
        file.write(&header)?;
        Ok(Self {
            file: Arc::clone(file),
            pool,
            buf: Vec::with_capacity(POOL_BUFFER_BYTES + 64),
            failed: None,
        })
    }

    /// Out of line and cold so the untraced pager's call sites carry only the `Option` test.
    #[cold]
    #[inline(never)]
    pub(crate) fn record(&mut self, call: Call, outcome: Outcome) {
        if self.failed.is_some() {
            return;
        }
        write_event(&mut self.buf, self.pool, call, Some(outcome));
        if self.buf.len() >= POOL_BUFFER_BYTES {
            self.spill();
        }
    }

    fn spill(&mut self) {
        if self.failed.is_none() && !self.buf.is_empty() {
            if let Err(e) = self.file.write(&self.buf) {
                self.failed = Some(e);
            }
        }
        self.buf.clear();
    }

    /// Hand the buffered events to the file and flush it to the OS.
    pub(crate) fn flush(&mut self) -> io::Result<()> {
        self.spill();
        if let Some(e) = &self.failed {
            return Err(io::Error::new(
                e.kind(),
                format!(
                    "pager trace: pool {} stopped recording after a failed write ({e}); the \
                     trace is incomplete and will not replay",
                    self.pool
                ),
            ));
        }
        self.file.flush()
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            tracing::warn!("{e}");
        }
    }
}

/// One pool of a parsed trace.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolTrace {
    pub pool: u32,
    pub tier: Tier,
    pub label: String,
    pub n_slots: usize,
    pub n_blocks: usize,
    pub calls: Vec<Call>,
    /// What each call returned when recorded, parallel to `calls`; empty for a trace that was
    /// generated rather than recorded.
    pub outcomes: Vec<Outcome>,
}

impl PoolTrace {
    /// A pool with no recorded outcomes, for generated traces.
    pub fn generated(label: impl Into<String>, n_slots: usize, n_blocks: usize) -> Self {
        Self {
            pool: 0,
            tier: Tier::Vram,
            label: label.into(),
            n_slots,
            n_blocks,
            calls: Vec::new(),
            outcomes: Vec::new(),
        }
    }

    /// This pool in the file format (header and events), for tests and tools that write traces
    /// out themselves. A pool without outcomes (a generated one) is written with `?` for each.
    pub fn write(&self, out: &mut Vec<u8>) {
        let info = PoolInfo {
            tier: self.tier,
            label: self.label.clone(),
            n_blocks: self.n_blocks,
        };
        write_header(out, self.pool, &info, self.n_slots);
        for (i, &call) in self.calls.iter().enumerate() {
            write_event(out, self.pool, call, self.outcomes.get(i).copied());
        }
    }
}

/// A trace line the parser could not read.
#[derive(Debug)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pager trace line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

/// A parsed trace file.
#[derive(Debug)]
pub struct Parsed {
    /// One per pool, in header order.
    pub pools: Vec<PoolTrace>,
    /// A final line with no newline, dropped: what a write cut short leaves. Its events are
    /// missing from `pools`, so a replay of the pool it belonged to ends early.
    pub truncated: Option<ParseError>,
}

/// Whether each call that returns something carried its outcome (see the module doc's `?`).
#[derive(Clone, Copy, PartialEq)]
enum Recorded {
    Unknown,
    Yes,
    No,
}

/// Read a trace written by the recorder, one [`PoolTrace`] per pool, in header order.
pub fn parse(mut input: impl BufRead) -> Result<Parsed, ParseError> {
    let mut pools: Vec<(PoolTrace, Recorded)> = Vec::new();
    let mut raw = Vec::new();
    let mut n = 0;
    let mut truncated = None;
    loop {
        raw.clear();
        let got = input.read_until(b'\n', &mut raw).map_err(|e| ParseError {
            line: n + 1,
            message: e.to_string(),
        })?;
        if got == 0 {
            break;
        }
        n += 1;
        let err = |message: String| ParseError { line: n, message };
        if raw.last() != Some(&b'\n') {
            truncated = Some(err(format!(
                "final line has no newline (a cut-short write), dropped: `{}`",
                String::from_utf8_lossy(&raw)
            )));
            break;
        }
        raw.pop();
        let line = std::str::from_utf8(&raw).map_err(|e| err(e.to_string()))?;
        if n == 1 && line != MAGIC {
            return Err(err(format!("expected `{MAGIC}`, found `{line}`")));
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("pool ") {
            let mut f = rest.splitn(5, ' ');
            let mut next = |what: &str| f.next().ok_or_else(|| err(format!("header: no {what}")));
            let pool = num(next("pool")?).map_err(&err)?;
            let tier = next("tier")?;
            let tier = Tier::parse(tier).ok_or_else(|| err(format!("unknown tier `{tier}`")))?;
            let n_slots = num(next("n_slots")?).map_err(&err)?;
            let n_blocks = num(next("n_blocks")?).map_err(&err)?;
            let label = next("label").unwrap_or_default().to_string();
            if pools.iter().any(|(p, _)| p.pool == pool) {
                return Err(err(format!("pool {pool} declared twice")));
            }
            let trace = PoolTrace {
                pool,
                tier,
                label,
                n_slots,
                n_blocks,
                calls: Vec::new(),
                outcomes: Vec::new(),
            };
            pools.push((trace, Recorded::Unknown));
            continue;
        }
        let (pool, call, outcome) = parse_event(line).map_err(&err)?;
        let (p, recorded) = pools
            .iter_mut()
            .find(|(p, _)| p.pool == pool)
            .ok_or_else(|| err(format!("event for undeclared pool {pool}")))?;
        if !matches!(call, Call::BeginBatch | Call::Unpin(_)) {
            let this = if outcome.is_some() {
                Recorded::Yes
            } else {
                Recorded::No
            };
            if *recorded != Recorded::Unknown && *recorded != this {
                return Err(err(format!("pool {pool} mixes recorded outcomes with `?`")));
            }
            *recorded = this;
        }
        p.calls.push(call);
        p.outcomes.push(outcome.unwrap_or(Outcome::Done));
    }
    let pools = pools
        .into_iter()
        .map(|(mut p, recorded)| {
            if recorded == Recorded::No {
                p.outcomes.clear();
            }
            p
        })
        .collect();
    Ok(Parsed { pools, truncated })
}

fn num<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("`{s}` is not a number"))
}

/// One event line; the outcome is `None` for `?` (not recorded).
fn parse_event(line: &str) -> Result<(u32, Call, Option<Outcome>), String> {
    let f: Vec<&str> = line.split(' ').collect();
    let field = |i: usize| {
        f.get(i)
            .copied()
            .ok_or_else(|| format!("truncated event `{line}`"))
    };
    let pool: u32 = num(field(0)?)?;
    let op = field(1)?;
    // Each op's outcome (`None` for `?`) and how many fields the line must have.
    let resolution = |at: usize| -> Result<(Option<Outcome>, usize), String> {
        match field(at)? {
            "?" => Ok((None, at + 1)),
            "h" => Ok((
                Some(Outcome::Resolved(Resolution::Hit {
                    slot: num(field(at + 1)?)?,
                })),
                at + 2,
            )),
            "m" => {
                let slot = num(field(at + 1)?)?;
                let evicted = match field(at + 2)? {
                    "-" => None,
                    e => Some(num(e)?),
                };
                Ok((
                    Some(Outcome::Resolved(Resolution::Miss { slot, evicted })),
                    at + 3,
                ))
            }
            "-" => Ok((Some(Outcome::Refused), at + 1)),
            o => Err(format!("unknown outcome `{o}`")),
        }
    };
    let slot = |at: usize| -> Result<(Option<Outcome>, usize), String> {
        match field(at)? {
            "?" => Ok((None, at + 1)),
            "-" => Ok((Some(Outcome::Slot(None)), at + 1)),
            s => Ok((Some(Outcome::Slot(Some(num(s)?))), at + 1)),
        }
    };
    let (call, (outcome, len)) = if op == "b" {
        (Call::BeginBatch, (Some(Outcome::Done), 2))
    } else {
        let id: BlockId = num(field(2)?)?;
        match op {
            "t" => (Call::Touch(id), resolution(3)?),
            "c" => (Call::TouchCold(id), resolution(3)?),
            "s" => (Call::Schedule(id), resolution(3)?),
            "r" => {
                let insert = match field(3)? {
                    "m" => Insert::Mru,
                    "c" => Insert::Cold,
                    i => return Err(format!("unknown insert `{i}`")),
                };
                (Call::ResolveAndPin(id, insert), resolution(4)?)
            }
            "q" => (Call::PinIfResident(id), slot(3)?),
            "p" => (Call::Repin(id), slot(3)?),
            "e" => (Call::Evict(id), slot(3)?),
            "u" => (Call::Unpin(id), (Some(Outcome::Done), 3)),
            o => return Err(format!("unknown op `{o}`")),
        }
    };
    if outcome == Some(Outcome::Refused) && !matches!(call, Call::ResolveAndPin(..)) {
        return Err(format!("`{op}` cannot be refused"));
    }
    if f.len() != len {
        return Err(format!("expected {len} fields in `{line}`"));
    }
    Ok((pool, call, outcome))
}

/// Where a replay first diverged from the recording.
#[derive(Debug, Clone, PartialEq)]
pub struct Divergence {
    /// Index into the pool's `calls`.
    pub at: usize,
    pub call: Call,
    pub recorded: Outcome,
    pub replayed: Outcome,
}

impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "call {} ({:?}) recorded {:?}, replayed {:?}",
            self.at, self.call, self.recorded, self.replayed
        )
    }
}

/// Replay `trace` into a fresh [`Pager`] of its slot count, checking every outcome against the
/// recorded one (when there are recorded outcomes), and return the pager's final counters.
pub fn replay(trace: &PoolTrace) -> Result<PagerStats, Divergence> {
    let mut p = Pager::new(trace.n_slots);
    for (at, &call) in trace.calls.iter().enumerate() {
        let replayed = call.apply(&mut p);
        if let Some(&recorded) = trace.outcomes.get(at) {
            if recorded != replayed {
                return Err(Divergence {
                    at,
                    call,
                    recorded,
                    replayed,
                });
            }
        }
    }
    Ok(p.stats())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every entry point, in a sequence that exercises each outcome it has: free-slot misses,
    /// LRU and cold-end evictions, batch protection, pins outliving a batch, a refused resolve, a
    /// probe hit and miss, repins, an explicit evict, and an evict of an absent block.
    const EVERY_ENTRY_POINT: &[Call] = &[
        Call::Touch(1),
        Call::Touch(2),
        Call::Touch(1),
        Call::Touch(3), // evicts 2
        Call::BeginBatch,
        Call::TouchCold(4),
        Call::TouchCold(5),
        Call::TouchCold(4),
        Call::BeginBatch,
        Call::Schedule(6),
        Call::Schedule(1),
        Call::BeginBatch,
        Call::ResolveAndPin(7, Insert::Mru),
        Call::ResolveAndPin(8, Insert::Cold),
        Call::ResolveAndPin(7, Insert::Mru),
        Call::ResolveAndPin(9, Insert::Cold),
        Call::BeginBatch,
        Call::ResolveAndPin(10, Insert::Mru), // refused: 7, 8, 9 pinned
        Call::PinIfResident(8),
        Call::PinIfResident(11),
        Call::Repin(9),
        Call::Repin(12),
        Call::Unpin(7),
        Call::Unpin(7),
        Call::Unpin(8),
        Call::Unpin(8),
        Call::Unpin(9),
        Call::Unpin(9),
        Call::Evict(9),
        Call::Evict(9),
        Call::ResolveAndPin(10, Insert::Mru),
        Call::Unpin(10),
        Call::BeginBatch,
        Call::Touch(1),
        Call::Touch(2),
    ];

    fn every_entry_point(p: &mut Pager) -> Vec<Outcome> {
        EVERY_ENTRY_POINT.iter().map(|c| c.apply(p)).collect()
    }

    fn temp_trace() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pager.trace");
        (dir, path)
    }

    /// The round trip the trace exists for: a recorded sequence replays into a fresh pager to the
    /// same outcome on every call and the same final counters, and recording changes nothing
    /// about what the traced pager itself returns.
    #[test]
    fn a_recorded_trace_replays_to_every_outcome_and_the_same_stats() {
        let (_dir, path) = temp_trace();
        let file = TraceFile::open(&path).expect("open");
        let mut traced = Pager::new(3);
        traced
            .trace_to(
                &file,
                PoolInfo {
                    tier: Tier::Dram,
                    label: "round trip".into(),
                    n_blocks: 13,
                },
            )
            .expect("trace");
        let mut plain = Pager::new(3);
        let got = every_entry_point(&mut traced);
        assert_eq!(
            got,
            every_entry_point(&mut plain),
            "tracing changed an outcome"
        );
        // The sequence must actually reach the outcomes it claims to cover.
        assert!(got.contains(&Outcome::Refused));
        assert!(got.iter().any(|o| matches!(
            o,
            Outcome::Resolved(Resolution::Miss {
                evicted: Some(_),
                ..
            })
        )));
        assert!(got.contains(&Outcome::Slot(None)));
        traced.flush_trace().expect("flush");

        let parsed = parse(io::BufReader::new(File::open(&path).expect("reopen"))).expect("parse");
        assert!(parsed.truncated.is_none());
        let pools = parsed.pools;
        assert_eq!(pools.len(), 1);
        let pool = &pools[0];
        assert_eq!(
            (pool.tier, pool.n_slots, pool.n_blocks),
            (Tier::Dram, 3, 13)
        );
        assert_eq!(pool.label, "round trip");
        assert_eq!(pool.outcomes, got);
        let stats = replay(pool).expect("replay matches the recording");
        let want = traced.stats();
        assert_eq!(
            (stats.hits, stats.misses, stats.evictions),
            (want.hits, want.misses, want.evictions)
        );
    }

    /// The comparison has teeth: a recording whose outcome was altered is reported at the call
    /// that diverged, not accepted.
    #[test]
    fn replay_reports_the_first_divergence() {
        let mut p = Pager::new(2);
        let mut trace = PoolTrace::generated("divergence", 2, 3);
        trace.calls = vec![Call::Touch(1), Call::Touch(2), Call::Touch(3)];
        trace.outcomes = trace.calls.iter().map(|c| c.apply(&mut p)).collect();
        replay(&trace).expect("an untouched recording replays");
        trace.outcomes[2] = Outcome::Resolved(Resolution::Miss {
            slot: 0,
            evicted: Some(2),
        });
        let d = replay(&trace).expect_err("an altered recording must not replay");
        assert_eq!(d.at, 2);
    }

    /// The writer and the parser agree on every op and outcome spelling.
    #[test]
    fn the_format_round_trips_through_text() {
        let mut trace = PoolTrace::generated("text with spaces", 3, 13);
        trace.pool = 7;
        trace.calls = EVERY_ENTRY_POINT.to_vec();
        trace.outcomes = every_entry_point(&mut Pager::new(3));
        let mut text = format!("{MAGIC}\n").into_bytes();
        trace.write(&mut text);
        let parsed = parse(&text[..]).expect("parse");
        assert_eq!(parsed.pools, vec![trace]);
    }

    /// A generated trace (no outcomes) is written with `?` and parses back to itself, outcomes
    /// still empty.
    #[test]
    fn a_generated_trace_round_trips() {
        let mut trace = PoolTrace::generated("generated", 3, 13);
        trace.calls = EVERY_ENTRY_POINT.to_vec();
        let mut text = format!("{MAGIC}\n").into_bytes();
        trace.write(&mut text);
        assert!(String::from_utf8_lossy(&text).contains("0 t 1 ?\n"));
        let parsed = parse(&text[..]).expect("a written generated trace must parse");
        assert_eq!(parsed.pools, vec![trace]);
        // A pool may not mix the two.
        let mixed = format!("{MAGIC}\npool 0 vram 2 2 x\n0 t 1 ?\n0 t 2 m 1 -\n");
        assert_eq!(parse(mixed.as_bytes()).expect_err("mixed").line, 4);
    }

    /// A write cut short leaves a final line with no newline: it is dropped and reported, and
    /// everything before it parses. A damaged line anywhere else is still an error.
    #[test]
    fn a_truncated_final_line_is_dropped_and_reported() {
        let body = format!("{MAGIC}\npool 0 vram 2 4 x\n0 t 1 m 0 -\n0 t 2 m 1 -\n0 t 1 h");
        let parsed = parse(body.as_bytes()).expect("only the tail is damaged");
        assert_eq!(parsed.pools[0].calls, [Call::Touch(1), Call::Touch(2)]);
        let t = parsed.truncated.expect("the cut line is reported");
        assert_eq!(t.line, 5);
        // A cut can leave a line that still parses (`0 t 1 h 3` cut from `0 t 1 h 31`), so a
        // final line without a newline is dropped even when it reads as valid.
        let looks_valid = format!("{MAGIC}\npool 0 vram 2 4 x\n0 t 1 m 0 -");
        let parsed = parse(looks_valid.as_bytes()).unwrap();
        assert!(parsed.pools[0].calls.is_empty() && parsed.truncated.is_some());
        let mid = format!("{MAGIC}\npool 0 vram 2 4 x\n0 t 1 h\n0 t 2 m 1 -\n");
        assert_eq!(parse(mid.as_bytes()).expect_err("mid-file damage").line, 3);
    }

    #[test]
    fn malformed_lines_are_rejected_with_their_line_number() {
        for (body, line) in [
            ("garbage\n", 1),
            ("# infr pager trace v1\n0 t 1 h 0\n", 2),
            ("# infr pager trace v1\npool 0 vram 2 2 x\n0 t 1 z 0\n", 3),
            ("# infr pager trace v1\npool 0 vram 2 2 x\n0 t 1 -\n", 3),
            ("# infr pager trace v1\npool 0 vram 2 2 x\n0 t 1 h 0 9\n", 3),
            ("# infr pager trace v1\npool 0 sram 2 2 x\n", 2),
        ] {
            let e = parse(body.as_bytes()).expect_err(body);
            assert_eq!(e.line, line, "{body:?}: {e}");
        }
    }

    /// Two pools opening the same path share one file, and a pool's events follow its header.
    #[test]
    fn pools_opening_one_path_share_the_file() {
        let (_dir, path) = temp_trace();
        let a = TraceFile::open(&path).expect("open a");
        let b = TraceFile::open(&path).expect("open b");
        assert!(Arc::ptr_eq(&a, &b));
        let info = |label: &str| PoolInfo {
            tier: Tier::Vram,
            label: label.into(),
            n_blocks: 4,
        };
        let mut p = Pager::new(2);
        let mut q = Pager::new(1);
        p.trace_to(&a, info("first")).unwrap();
        q.trace_to(&b, info("second")).unwrap();
        p.touch(1);
        q.touch(2);
        q.touch(3);
        p.flush_trace().unwrap();
        q.flush_trace().unwrap();
        let pools = parse(io::BufReader::new(File::open(&path).unwrap()))
            .unwrap()
            .pools;
        let labels: Vec<_> = pools
            .iter()
            .map(|p| (p.label.as_str(), p.calls.len()))
            .collect();
        assert_eq!(labels, [("first", 1), ("second", 2)]);
    }
}
