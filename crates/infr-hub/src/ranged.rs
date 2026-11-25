//! Intra-file parallelism: ONE object fetched over several connections at once, as a grid of
//! byte ranges, with [`crate::parts`]'s sidecar recording which cells have landed.
//!
//! The reason this exists is the same measurement that motivated fetching several shards at once —
//! a single connection to the HF CDN, not the link, is what caps a download — except that a model
//! shipped as one file (`unsloth/DeepSeek-V3.2-GGUF`'s 161 GB `UD-TQ1_0`) has no shards to spread
//! over connections. So the FILE is spread instead: worker `k` asks for `Range: bytes=A-B` and
//! `pwrite`s the reply at offset `A`, and the assembled file is byte-identical to what one stream
//! would have written.
//!
//! Three things are load-bearing, and all three are about not producing a plausible-sized file
//! assembled from two different uploads — the failure that once read as "this quant is broken" for
//! a day (memory `verify-gguf-downloads`):
//!
//!   * **Identity is checked before a resumed byte is written.** The sidecar records HF's LFS
//!     sha256 for the object it belongs to, and a run that finds a different one throws the partial
//!     away instead of continuing it. That is a content hash, so unlike an `ETag` it cannot differ
//!     merely because a different CDN edge answered.
//!   * **Every range request carries `If-Range` and must come back `206`** — one chunk, or a run of
//!     them on one request — with a `Content-Range` starting at exactly the byte asked for, out of
//!     a total that still matches the plan (it may end early; only whole chunks are kept). A `200`
//!     means the server declined the range and is offering the whole object from byte 0 — never
//!     something to write into the middle of a file.
//!   * **The sha256 gate is untouched.** Everything here is about where bytes land; whether the
//!     assembled file is the one HF advertises is still decided once, at the end, by
//!     [`crate::download`], and a mismatch still discards the temp and refuses to link it.

use crate::download::Object;
use crate::http::{api_client, download_client, token, Claim, ConnBudget, Permit};
use crate::parts::{self, Plan};
use indicatif::{MultiProgress, ProgressBar};
use infr_core::error::{Error, Result};
use infr_core::shutdown::shutdown_requested;
use infr_plat::fileio::write_all_at;
use reqwest::header::{
    ACCEPT_RANGES, CONTENT_LENGTH, CONTENT_RANGE, ETAG, IF_RANGE, LAST_MODIFIED, RANGE,
};
use reqwest::StatusCode;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tracing::{debug, warn};

/// What the probe `HEAD` learned about the object: how big it is, and the validator to present on
/// each range request.
pub(crate) struct Probe {
    pub(crate) total: u64,
    pub(crate) validator: Option<String>,
}

/// Ask the object how big it is and whether it can be fetched in pieces, with a `HEAD`.
///
/// `Ok(None)` is the graceful fallback and covers everything uncertain: a server that does not
/// advertise `Accept-Ranges: bytes`, one that will not say how long the object is, an error status.
/// The object is then downloaded as a single stream exactly as it was before ranges existed.
///
/// A `HEAD` rather than a one-byte ranged `GET` because it costs no body at all — a server that
/// declines the range would answer a ranged `GET` with the WHOLE object, and the client abandoning
/// that response has still made the origin start sending 161 GB down a socket it is about to close.
/// The cost of asking this way is that `Accept-Ranges` is an advertisement rather than a
/// demonstration; the demonstration is every range request, each of which must come back `206`
/// starting at the byte it asked for, and a server that advertised ranges and then declines one
/// falls back to the single stream from there.
pub(crate) fn probe(obj: &Object) -> Result<Option<Probe>> {
    let (url, label) = (obj.url, obj.label);
    let mut req = api_client()?.head(url);
    if let Some(t) = token() {
        req = req.bearer_auth(t);
    }
    let resp = req
        .send()
        .map_err(|e| Error::Other(format!("HEAD {url}: {e}")))?;
    if !resp.status().is_success() {
        debug!("{label}: range probe answered HTTP {}", resp.status());
        return Ok(None);
    }
    let ranges = resp
        .headers()
        .get(ACCEPT_RANGES)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|t| t.trim().eq_ignore_ascii_case("bytes")));
    if !ranges {
        debug!("{label}: server does not advertise byte ranges; not splitting");
        return Ok(None);
    }
    // The HEADER, not `Response::content_length()` — that reports the length of the body being
    // received, which for a `HEAD` is zero by definition and would read as "no length advertised".
    let Some(total) = resp
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|n| *n > 0)
    else {
        debug!("{label}: no length advertised; not splitting");
        return Ok(None);
    };
    check_total(label, total, obj.size)?;
    Ok(Some(Probe {
        total,
        validator: validator_of(resp.headers()),
    }))
}

/// Largest object a ranged download will plan — orders of magnitude above the largest single file
/// this repo has pulled (161 GB), and small enough that its plan is a modest table.
///
/// The total comes from a server's `Content-Length` and sizes the plan's per-cell table up front,
/// so an unchecked one near `u64::MAX` is an allocation that aborts the process before a byte is
/// fetched.
const MAX_OBJECT_BYTES: u64 = 1 << 44;

/// Refuse a probed size the plan must not be built from: past [`MAX_OBJECT_BYTES`], or different
/// from the LFS size (`lfs_size`) the hub's own `HEAD` reported for the same object.
fn check_total(label: &str, total: u64, lfs_size: Option<u64>) -> Result<()> {
    if total > MAX_OBJECT_BYTES {
        return Err(Error::Other(format!(
            "{label}: the server reports {total} bytes, more than the {MAX_OBJECT_BYTES}-byte \
             limit for one file; refusing to download it"
        )));
    }
    match lfs_size {
        Some(size) if size != total => Err(Error::Other(format!(
            "{label}: the server reports {total} bytes but the hub's LFS size is {size}; \
             refusing to download it"
        ))),
        _ => Ok(()),
    }
}

/// The `If-Range` validator a response offers: the `ETag` if present, else `Last-Modified`.
fn validator_of(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(ETAG)
        .or_else(|| headers.get(LAST_MODIFIED))
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .filter(|s| !s.is_empty())
}

/// `bytes 200-299/1000` → `(200, 299, 1000)`. `None` for any other shape, including the `*/N`
/// unsatisfied form and an unknown total (`.../*`), neither of which can plan a grid.
pub(crate) fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let rest = value.trim().strip_prefix("bytes ")?;
    let (span, total) = rest.split_once('/')?;
    let (first, last) = span.split_once('-')?;
    Some((
        first.trim().parse().ok()?,
        last.trim().parse().ok()?,
        total.trim().parse().ok()?,
    ))
}

/// `bytes */1000` → `1000`: the total a `416 Range Not Satisfiable` reports. `None` for any other
/// shape, including a satisfied span.
pub(crate) fn parse_unsatisfied_range(value: &str) -> Option<u64> {
    let total = value.trim().strip_prefix("bytes ")?.strip_prefix("*/")?;
    total.trim().parse().ok()
}

/// Why a ranged download stopped, split by what the caller must DO about it — the three outcomes
/// need three different treatments of the bytes already on disk, and collapsing them is how a
/// partial from a since-replaced upload gets continued.
pub(crate) enum RangedError {
    /// The object is not the one this partial belongs to (a re-upload, or a resize). The caller
    /// must DISCARD the partial and start over — continuing it is the splice.
    Changed(String),
    /// The server will not serve ranges after all. The caller must discard the partial — it has
    /// holes in it and no single-stream resume can use one — and fall back to a single stream.
    NoRanges(String),
    /// Transport or I/O. The partial and its sidecar are KEPT, so the next run resumes the cells
    /// that did land.
    Fatal(Error),
}

/// Fetch `obj` into `tmp` over several connections, resuming `existing` when it describes this same
/// object.
///
/// The partial is bound to [`Object::oid`] — HF's advertised LFS sha256 — which is what decides
/// whether `existing` may be continued at all. `claim` is this file's place in the pull's whole
/// connection allowance: this call starts on its share of what is spare
/// ([`Claim::take_share`]), takes more at chunk boundaries as other files give theirs back, and
/// returns them as its workers finish, so the total in flight across every file of the model stays
/// at `hub.pull_jobs`.
pub(crate) fn run(
    obj: &Object,
    tmp: &Path,
    existing: Option<Plan>,
    probe: &Probe,
    claim: &mut Claim<'_>,
    pb: &ProgressBar,
) -> std::result::Result<(), RangedError> {
    let label = obj.label;
    let budget = claim.budget();
    // Both failures below leave the partial for the CALLER to discard: `Changed` can also come out
    // of a worker mid-download, and one owner of "throw these bytes away" is one place to get it
    // right rather than two places to get it inconsistent.
    let plan = reconcile(existing, probe, obj.oid).map_err(RangedError::Changed)?;
    // The temp file first, then the sidecar that describes it — never a plan pointing at a file
    // that does not exist, which the next run would have to reason about.
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(tmp)
        .map_err(|e| RangedError::Fatal(Error::from(e)))?;
    // The workers write chunks in whatever order they finish. Not fatal when refused: the chunks
    // still land, a late one first just waits for the filesystem to fill the gap below it.
    if let Err(e) = infr_plat::fileio::allow_out_of_order_writes(&file) {
        debug!("{label}: could not prepare the partial for out-of-order writes: {e}");
    }
    parts::save(tmp, &plan).map_err(RangedError::Fatal)?;

    pb.set_position(plan.completed_bytes());
    // A resume jumps the bar forward instantly, and the rate estimator would read that jump as
    // transferred bytes; the numbers must describe THIS transfer.
    pb.reset_eta();

    // One connection per remaining chunk at most, and never more than this file's share of what is
    // spare. The calling thread is one of the workers, so it asks for one fewer than the
    // connections it wants.
    let permits = claim.take_share(plan.remaining().saturating_sub(1));
    let conns = permits.len() + 1;
    let job = Job {
        url: obj.url,
        label,
        tmp,
        file,
        size: plan.size,
        chunk: plan.chunk,
        chunks: plan.chunks(),
        validator: plan.validator.clone(),
        plan: Mutex::new(plan),
        cursor: AtomicUsize::new(0),
        // The calling thread; `spawn_worker` counts each of the others in.
        workers: AtomicUsize::new(1),
        stop: AtomicBool::new(false),
        err: Mutex::new(None),
        pb,
    };
    pb.suspend(|| {
        debug!(
            "{label}: {} of {} chunks left, {conns} connection(s)",
            job.plan.lock().unwrap().remaining(),
            job.chunks
        )
    });

    let job = &job;
    std::thread::scope(|scope| {
        for permit in permits {
            spawn_worker(scope, job, budget, permit);
        }
        worker(scope, job, budget);
    });

    if let Some(e) = job.err.lock().unwrap_or_else(|e| e.into_inner()).take() {
        return Err(e);
    }
    let plan = job.plan.lock().unwrap_or_else(|e| e.into_inner());
    if !plan.all_done() {
        // Both are `Fatal`, so the partial and its sidecar are kept either way; they differ only
        // in what the user is told. A Ctrl-C is not "chunks went missing".
        return Err(RangedError::Fatal(if shutdown_requested() {
            Error::Aborted
        } else {
            Error::Other(format!(
                "{label}: ranged download ended with {} chunk(s) missing",
                plan.remaining()
            ))
        }));
    }
    // The grid tiles the file exactly, so a complete plan means a complete file — assert it rather
    // than trust it, because the alternative is committing a blob with a hole in it.
    let got = fs::metadata(tmp)
        .map(|m| m.len())
        .map_err(|e| RangedError::Fatal(Error::from(e)))?;
    if got != plan.size {
        return Err(RangedError::Fatal(Error::Other(format!(
            "{label}: assembled {got} bytes but the plan says {}",
            plan.size
        ))));
    }
    Ok(())
}

/// Decide whether `existing` may be resumed against the object the probe just described, returning
/// the plan to run — or the reason the partial has to go.
///
/// The identity test prefers the LFS oid over the `ETag` deliberately: the oid is a hash of the
/// content, so it answers "are these the same bytes?" directly and identically from every edge,
/// whereas an `ETag` mismatch can only say "something the server tracks differs". The `ETag` is
/// still used when there is no oid (a non-LFS object), and is refreshed either way because
/// `If-Range` has to be presented in the server's own terms.
fn reconcile(
    existing: Option<Plan>,
    probe: &Probe,
    expected_sha: Option<&str>,
) -> std::result::Result<Plan, String> {
    let Some(mut plan) = existing else {
        return Ok(Plan::fresh(
            probe.total,
            parts::chunk_bytes(),
            expected_sha.map(str::to_string),
            probe.validator.clone(),
        ));
    };
    if plan.size != probe.total {
        return Err(format!(
            "the object is now {} bytes, the partial was planned for {}",
            probe.total, plan.size
        ));
    }
    match (plan.oid.as_deref(), expected_sha) {
        (Some(had), Some(now)) if !had.eq_ignore_ascii_case(now) => {
            return Err(format!("the LFS sha256 changed ({had} → {now})"));
        }
        (Some(_), Some(_)) => {}
        // No oid on one side or both, so the oids cannot be compared: fall back to the opaque
        // validator, which is weaker (an edge that reports a different one costs a restart) but is
        // all there is. Skipping it when only ONE side lacks an oid — a partial planned while the
        // HEAD worked, resumed in a run where it reports none — left the resume with no identity
        // check, and that run's missing oid means no sha256 gate at the end either.
        _ => match (plan.validator.as_deref(), probe.validator.as_deref()) {
            (Some(had), Some(now)) if had != now => {
                return Err(format!("the validator changed ({had} → {now})"));
            }
            _ => {}
        },
    }
    if plan.oid.is_none() {
        plan.oid = expected_sha.map(str::to_string);
    }
    plan.validator = probe.validator.clone();
    Ok(plan)
}

/// Everything the chunk workers share. Immutable but for the plan (behind its lock), the claim
/// cursor and the first error.
struct Job<'a> {
    url: &'a str,
    label: &'a str,
    tmp: &'a Path,
    file: fs::File,
    size: u64,
    chunk: u64,
    chunks: usize,
    validator: Option<String>,
    plan: Mutex<Plan>,
    cursor: AtomicUsize,
    /// Workers running now. One means a worker may fetch past its claimed cell on the same request
    /// (see [`span_end`]).
    workers: AtomicUsize,
    stop: AtomicBool,
    err: Mutex<Option<RangedError>>,
    pb: &'a ProgressBar,
}

impl Job<'_> {
    /// Chunk `i`'s `(start, len)`, without taking the plan's lock — the grid is fixed for the whole
    /// run, so the numbers cannot change under a worker.
    fn range(&self, i: usize) -> (u64, u64) {
        let start = (i as u64) * self.chunk;
        (start, self.chunk.min(self.size - start))
    }
}

/// Start one more chunk worker, holding `permit` for as long as it runs. The permit rides with the
/// thread so the budget gets it back the moment this worker runs out of chunks to claim, even if it
/// returns early or panics.
fn spawn_worker<'s, 'e>(
    scope: &'s std::thread::Scope<'s, 'e>,
    job: &'s Job<'s>,
    budget: &'s ConnBudget,
    permit: Permit<'s>,
) {
    // Counted before the thread exists, so no worker ever sees itself alone while another is
    // being started.
    job.workers.fetch_add(1, Ordering::AcqRel);
    scope.spawn(move || {
        let _permit = permit;
        worker(scope, job, budget);
    });
}

/// Start one more worker if the budget will give this download another connection and the grid
/// has a cell nobody has claimed yet.
///
/// Permits come free while a download runs — most often because another file of the model
/// finished and its worker gave one back — and a download that fixed its connection count when it
/// started would leave them idle. That is the tail of every shard set: once fewer files remain than
/// there are connections, the last few would otherwise crawl on one connection each while the rest
/// of the allowance sits unused. Growth is bounded by the same budget as everything else, so it
/// can only ever fill the allowance, never exceed it.
fn grow<'s>(scope: &'s std::thread::Scope<'s, '_>, job: &'s Job<'s>, budget: &'s ConnBudget) {
    if job.cursor.load(Ordering::Relaxed) < job.chunks {
        if let Some(permit) = budget.try_grow() {
            spawn_worker(scope, job, budget, permit);
        }
    }
}

/// Counts a worker out of [`Job::workers`] however it stops.
struct Leaving<'a>(&'a AtomicUsize);

impl Drop for Leaving<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Claim cells until the grid runs out or somebody fails. One shared cursor, so each chunk is
/// fetched by exactly one worker and the file is still filled roughly front to back.
///
/// Before each chunk a worker tries to [`grow`] the fan-out, which is how a download that started
/// on its own connection — every connection busy on other files — ends up with the connections
/// those files give back when they finish.
fn worker<'s>(scope: &'s std::thread::Scope<'s, '_>, job: &'s Job<'s>, budget: &'s ConnBudget) {
    let _leaving = Leaving(&job.workers);
    // A cell already claimed: one this worker took at a boundary of its own request and could not
    // continue into, because another worker had the cell between.
    let mut claimed = None;
    // `shutdown_requested()` here stops a worker CLAIMING another chunk, without touching
    // `job.stop`. A worker already inside a chunk stops there instead (`fetch_chunk`), which does
    // set `job.stop` — but it reports `Error::Aborted`, and the first error wins, so a Ctrl-C
    // still reads as a Ctrl-C rather than as a download failure.
    while !job.stop.load(Ordering::Relaxed) && !shutdown_requested() {
        let i = claimed
            .take()
            .unwrap_or_else(|| job.cursor.fetch_add(1, Ordering::Relaxed));
        if i >= job.chunks {
            return;
        }
        if job.plan.lock().unwrap().is_done(i) {
            continue; // landed in an earlier run
        }
        grow(scope, job, budget);
        match fetch_chunk(scope, job, budget, i) {
            Ok(next) => claimed = next,
            Err(e) => {
                job.stop.store(true, Ordering::Relaxed);
                let mut slot = job.err.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(e); // first failure wins; the rest are consequences of `stop`
                }
                return;
            }
        }
    }
}

/// The last cell the request for cell `i` asks for.
///
/// While other workers share the grid that is `i` alone. A worker on its own instead asks for `i`
/// and every cell after it up to the next one already on disk, in ONE request, and claims each of
/// them only as its body reaches it ([`fetch_chunk_once`]). That is the common case, not an edge:
/// a file that starts while every connection is busy on other files runs alone until one is freed,
/// and one request per chunk would cost it a round trip (two, through HF's redirect) every
/// [`parts::CHUNK_BYTES`]. When the worker grows, the new worker claims the next cell, and the
/// original abandons its request at that boundary and carries on with single cells.
fn span_end(job: &Job, i: usize) -> usize {
    if job.workers.load(Ordering::Acquire) > 1 {
        return i;
    }
    let plan = job.plan.lock().unwrap();
    (i + 1..job.chunks)
        .take_while(|c| !plan.is_done(*c))
        .last()
        .unwrap_or(i)
}

/// Tries one chunk gets before its failure ends the whole file. A transient failure — a dropped
/// connection, a short body, a `5xx` or `429` — usually clears on a fresh connection, and the
/// alternative is aborting a download that may be hours in for one bad request.
const CHUNK_ATTEMPTS: u32 = 3;

/// Wait before retry `n` is `n` times this. Short under test, where the local origin is the only
/// thing being waited on.
const CHUNK_RETRY_BACKOFF: Duration = if cfg!(test) {
    Duration::from_millis(10)
} else {
    Duration::from_secs(1)
};

/// One failed attempt at a chunk.
enum ChunkFailure {
    /// Worth another try on a fresh connection. Carries the message the file fails with if the
    /// tries run out, and the cell the attempt failed in: a request for several cells may have
    /// landed some of them first, and only the rest need asking for again.
    Transient { why: String, cell: usize },
    /// Retrying cannot help: the object moved, ranges stopped working, the disk failed, or the user
    /// asked to stop.
    Permanent(RangedError),
}

/// Fetch chunk `i` and record it as landed, retrying a transient failure in place up to
/// [`CHUNK_ATTEMPTS`] times before it fails the file. Another worker's failure or a Ctrl-C stops
/// the retrying, since the file is ending anyway.
///
/// A request that carried on past `i` ([`span_end`]) can fail in a later cell. The tries are then
/// counted for THAT cell, since every cell before it landed. Returns the cell this worker claimed
/// at a boundary and did not fetch, if any, for [`worker`] to fetch next.
fn fetch_chunk<'s>(
    scope: &'s std::thread::Scope<'s, '_>,
    job: &'s Job<'s>,
    budget: &'s ConnBudget,
    i: usize,
) -> std::result::Result<Option<usize>, RangedError> {
    let (mut cell, mut attempt) = (i, 1);
    loop {
        match fetch_chunk_once(scope, job, budget, cell) {
            Ok(next) => return Ok(next),
            Err(ChunkFailure::Permanent(e)) => return Err(e),
            Err(ChunkFailure::Transient { why, cell: failed }) => {
                if failed != cell {
                    (cell, attempt) = (failed, 1);
                }
                if attempt >= CHUNK_ATTEMPTS
                    || job.stop.load(Ordering::Relaxed)
                    || shutdown_requested()
                {
                    return Err(RangedError::Fatal(Error::Other(why)));
                }
                job.pb.suspend(|| {
                    warn!(
                        "{why}; retrying (attempt {} of {CHUNK_ATTEMPTS})",
                        attempt + 1
                    )
                });
                std::thread::sleep(CHUNK_RETRY_BACKOFF * attempt);
                attempt += 1;
            }
        }
    }
}

/// One request starting at chunk `i`: fetch each cell it covers ([`span_end`]) and record it as
/// landed — in that order, and only if every byte of it arrived.
///
/// A reply that ends short of the request keeps only the cells it holds whole; the worker then
/// asks for the rest from the next cell, as for any finished request.
///
/// Between two cells of the same request the worker claims the next cell. If the cursor hands it
/// exactly that one it reads on, and tries to [`grow`]; if another worker took it, this request's
/// remaining bytes are not this worker's to write, so the request is dropped there and the cell it
/// was handed instead is returned. A transient failure takes back the progress shown for the cell
/// it interrupted, because the retry counts those bytes again.
fn fetch_chunk_once<'s>(
    scope: &'s std::thread::Scope<'s, '_>,
    job: &'s Job<'s>,
    budget: &'s ConnBudget,
    i: usize,
) -> std::result::Result<Option<usize>, ChunkFailure> {
    use ChunkFailure::{Permanent, Transient};
    let mut last = span_end(job, i);
    let start = job.range(i).0;
    let end = {
        let (s, l) = job.range(last);
        s + l - 1
    };
    let what = if last == i {
        format!("chunk {i}")
    } else {
        format!("chunks {i}..={last}")
    };
    let mut req = download_client()
        .map_err(|e| Permanent(RangedError::Fatal(e)))?
        .get(job.url)
        .header(RANGE, format!("bytes={start}-{end}"));
    if let Some(v) = &job.validator {
        req = req.header(IF_RANGE, v.clone());
    }
    if let Some(t) = token() {
        req = req.bearer_auth(t);
    }
    let mut resp = req.send().map_err(|e| Transient {
        why: format!(
            "{}: {what} ({start}..={end}): {e} (partial kept for resume)",
            job.label
        ),
        cell: i,
    })?;
    if resp.status() == StatusCode::OK {
        // The range was declined and the whole object is on its way — writing that at `start` is
        // exactly the splice this design exists to prevent.
        return Err(Permanent(RangedError::NoRanges(format!(
            "{}: the server answered {what} with the whole object",
            job.label
        ))));
    }
    if resp.status().is_server_error() || resp.status() == StatusCode::TOO_MANY_REQUESTS {
        return Err(Transient {
            why: format!(
                "{}: {what}: HTTP {} (partial kept for resume)",
                job.label,
                resp.status()
            ),
            cell: i,
        });
    }
    if resp.status() != StatusCode::PARTIAL_CONTENT {
        return Err(Permanent(RangedError::Fatal(Error::Other(format!(
            "{}: {what}: HTTP {}",
            job.label,
            resp.status()
        )))));
    }
    // The reply must start where it was asked to, out of the size the plan was built on. It may
    // END early: a server or proxy is allowed to cap how much of a range it sends (RFC 9110
    // §14.2), and a lone worker asks for the whole remaining run. Only the cells that arrived
    // WHOLE are kept; the bytes of a cell cut short are not written, the request is dropped there,
    // and the worker asks again from that cell. A reply too short to hold even the first cell
    // cannot make progress by asking again, so the file falls back to one stream.
    let served_end = match resp
        .headers()
        .get(CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_content_range)
    {
        Some((a, b, total)) if a == start && a <= b && b <= end && total == job.size => b,
        Some((a, b, total)) => {
            return Err(Permanent(RangedError::Changed(format!(
                "{}: asked for bytes {start}-{end}/{} and was served {a}-{b}/{total}",
                job.label, job.size
            ))));
        }
        None => {
            return Err(Permanent(RangedError::NoRanges(format!(
                "{}: {what} came back 206 with no usable Content-Range",
                job.label
            ))));
        }
    };
    if let Some(n) = resp.content_length() {
        if n != served_end - start + 1 {
            return Err(Permanent(RangedError::Fatal(Error::Other(format!(
                "{}: {what} announced {n} bytes for the span {start}-{served_end}",
                job.label
            )))));
        }
    }
    if served_end < end {
        // The last cell that ends inside what was served.
        let whole = (i..=last).take_while(|c| {
            let (s, l) = job.range(*c);
            s + l - 1 <= served_end
        });
        last = match whole.last() {
            Some(c) => c,
            None => {
                return Err(Permanent(RangedError::NoRanges(format!(
                    "{}: asked for {what} and was served only bytes {start}-{served_end}, less \
                     than one chunk",
                    job.label
                ))))
            }
        };
    }

    let mut buf = vec![0u8; 1 << 16];
    let mut cell = i;
    loop {
        read_cell(job, &mut resp, &mut buf, cell)?;
        if cell == last {
            return Ok(None);
        }
        if job.stop.load(Ordering::Relaxed) || shutdown_requested() {
            return Ok(None);
        }
        let next = job.cursor.fetch_add(1, Ordering::Relaxed);
        if next != cell + 1 {
            return Ok(Some(next));
        }
        cell = next;
        grow(scope, job, budget);
    }
}

/// Read cell `cell`'s bytes off `resp` into the partial at their offset, then record the cell as
/// landed.
fn read_cell(
    job: &Job,
    resp: &mut reqwest::blocking::Response,
    buf: &mut [u8],
    cell: usize,
) -> std::result::Result<(), ChunkFailure> {
    use ChunkFailure::{Permanent, Transient};
    let (start, len) = job.range(cell);
    let mut at = start;
    let mut left = len;
    // A failure past this point has already advanced the bar by what landed of this attempt.
    let undo = |left: u64, why: String| {
        job.pb.dec(len - left);
        Transient { why, cell }
    };
    while left > 0 {
        // Between chunks is not often enough: a chunk is 64 MiB, so a worker that only checked
        // there would keep pulling for tens of seconds after a Ctrl-C. Abandoning mid-chunk costs
        // nothing — the bytes already written are at their absolute offsets and stay, the chunk
        // just never gets marked done, so a resume re-fetches this one chunk and no more.
        if shutdown_requested() {
            return Err(Permanent(RangedError::Fatal(Error::Aborted)));
        }
        let want = buf.len().min(left as usize);
        let n = match resp.read(&mut buf[..want]) {
            Ok(n) => n,
            Err(e) => {
                return Err(undo(
                    left,
                    format!(
                        "{}: chunk {cell} at {at}: {e} (partial kept for resume)",
                        job.label
                    ),
                ))
            }
        };
        if n == 0 {
            // A short body would otherwise leave a hole under a chunk marked done.
            return Err(undo(
                left,
                format!(
                    "{}: chunk {cell} ended {left} bytes early (partial kept for resume)",
                    job.label
                ),
            ));
        }
        write_all_at(&job.file, &buf[..n], at)
            .map_err(|e| Permanent(RangedError::Fatal(Error::from(e))))?;
        at += n as u64;
        left -= n as u64;
        job.pb.inc(n as u64);
    }
    // Data first, then the record of it: a sidecar that outlived its bytes across a crash would
    // claim a cell of zeros had landed, and only the sha256 gate would notice — after the whole
    // object had been downloaded again for nothing.
    job.file
        .sync_data()
        .map_err(|e| Permanent(RangedError::Fatal(Error::from(e))))?;
    let mut plan = job.plan.lock().unwrap();
    plan.mark_done(cell);
    parts::save(job.tmp, &plan).map_err(|e| Permanent(RangedError::Fatal(e)))?;
    Ok(())
}

/// Log a restart where the user will see it: a discarded partial is minutes or hours of transfer,
/// and the reason it happened is the difference between "the repo was updated" and "something is
/// wrong with this mirror".
pub(crate) fn warn_restart(progress: &MultiProgress, why: &str) {
    progress.suspend(|| warn!("{why}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_range_parses_the_forms_that_matter() {
        assert_eq!(parse_content_range("bytes 0-0/1000"), Some((0, 0, 1000)));
        assert_eq!(
            parse_content_range("bytes 200-299/173458321408"),
            Some((200, 299, 173458321408))
        );
        // Shapes that cannot plan a grid.
        for bad in [
            "bytes */1000", // unsatisfied
            "bytes 0-99/*", // unknown total
            "items 0-99/100",
            "0-99/100",
            "bytes 0-99",
            "",
        ] {
            assert!(parse_content_range(bad).is_none(), "accepted {bad:?}");
        }
    }

    #[test]
    fn unsatisfied_range_parses_only_the_star_form() {
        assert_eq!(parse_unsatisfied_range("bytes */1000"), Some(1000));
        assert_eq!(parse_unsatisfied_range(" bytes */0 "), Some(0));
        for bad in ["bytes 0-99/100", "bytes */*", "*/100", "bytes */x", ""] {
            assert!(parse_unsatisfied_range(bad).is_none(), "accepted {bad:?}");
        }
    }

    /// A server's `Content-Length` sizes the plan's table before a byte is fetched, so an absurd
    /// one — or one that disagrees with the hub's LFS size for the same object — is an error, not
    /// an allocation that aborts the process.
    #[test]
    fn an_absurd_or_mismatched_length_is_refused() {
        use crate::testhttp::TestHub;
        let hub = TestHub::start();
        let name = hub.add_shards("m-huge", 1, 4096).remove(0);
        let url = format!("{}/org/repo/resolve/main/{name}", hub.origin());
        let object = |size| Object {
            url: &url,
            label: &name,
            oid: None,
            size,
        };

        let honest = probe(&object(Some(4096)))
            .unwrap()
            .expect("ranges advertised");
        assert_eq!(honest.total, 4096);

        hub.lie_about_length(&name, u64::MAX);
        let err = probe(&object(None))
            .err()
            .expect("u64::MAX must be refused");
        assert!(err.to_string().contains("refusing"), "{err}");

        hub.lie_about_length(&name, 8192);
        let err = probe(&object(Some(4096)))
            .err()
            .expect("a size mismatch must be refused");
        assert!(err.to_string().contains("LFS size"), "{err}");
        // Without an LFS size to compare against, a plausible length is still accepted.
        assert_eq!(probe(&object(None)).unwrap().unwrap().total, 8192);
    }

    fn probe_of(total: u64, validator: &str) -> Probe {
        Probe {
            total,
            validator: Some(validator.to_string()),
        }
    }

    /// The re-upload rule, which is the whole reason the sidecar records an oid: same size, same
    /// server, different content — resume it and the blob is half of one upload and half of
    /// another, at exactly the right length, and only the sha256 gate stands between that and the
    /// user's model directory.
    #[test]
    fn a_changed_oid_refuses_the_partial() {
        let old = "a".repeat(64);
        let new = "b".repeat(64);
        let plan = Plan::fresh(1000, 100, Some(old.clone()), Some("\"v1\"".into()));

        let same = reconcile(Some(plan.clone()), &probe_of(1000, "\"v1\""), Some(&old))
            .expect("the same object must resume");
        assert_eq!(same.oid.as_deref(), Some(old.as_str()));

        let err = reconcile(Some(plan.clone()), &probe_of(1000, "\"v1\""), Some(&new))
            .expect_err("a re-upload must NOT resume");
        assert!(err.contains("LFS sha256 changed"), "{err}");

        // A resize is the same call for the same reason, and is caught even when no oid is known.
        let err = reconcile(Some(plan.clone()), &probe_of(999, "\"v1\""), Some(&old))
            .expect_err("a resized object must NOT resume");
        assert!(err.contains("999"), "{err}");
    }

    /// Without an oid the opaque validator is all there is, so it decides — and it is refreshed on
    /// every run, because `If-Range` has to be presented in the server's current terms.
    #[test]
    fn the_validator_decides_when_there_is_no_oid() {
        let plan = Plan::fresh(1000, 100, None, Some("\"v1\"".into()));
        let err = reconcile(Some(plan.clone()), &probe_of(1000, "\"v2\""), None)
            .expect_err("a changed validator must not resume an unhashed object");
        assert!(err.contains("validator changed"), "{err}");
        // …but an oid, once known, outranks it: same content, new edge, new ETag → resume.
        let sha = "c".repeat(64);
        let hashed = Plan::fresh(1000, 100, Some(sha.clone()), Some("\"v1\"".into()));
        let ok = reconcile(Some(hashed), &probe_of(1000, "\"v2\""), Some(&sha)).unwrap();
        assert_eq!(
            ok.validator.as_deref(),
            Some("\"v2\""),
            "validator refreshed"
        );
    }

    /// An oid on only one side cannot be compared, so the validator decides there too. A partial
    /// planned with an oid and resumed in a run that has none would otherwise continue with no
    /// identity check at all, and that run has no digest left to verify the blob against.
    #[test]
    fn the_validator_decides_when_one_side_has_no_oid() {
        let sha = "e".repeat(64);
        let hashed = Plan::fresh(1000, 100, Some(sha.clone()), Some("\"v1\"".into()));
        let err = reconcile(Some(hashed), &probe_of(1000, "\"v2\""), None)
            .expect_err("a changed validator must not resume when this run has no oid");
        assert!(err.contains("validator changed"), "{err}");

        let unhashed = Plan::fresh(1000, 100, None, Some("\"v1\"".into()));
        let err = reconcile(Some(unhashed), &probe_of(1000, "\"v2\""), Some(&sha))
            .expect_err("a changed validator must not resume a partial planned without an oid");
        assert!(err.contains("validator changed"), "{err}");
    }

    /// A fresh plan takes its whole shape from the probe.
    #[test]
    fn no_partial_plans_from_the_probe() {
        let sha = "d".repeat(64);
        let p = reconcile(None, &probe_of(4096, "\"v9\""), Some(&sha)).unwrap();
        assert_eq!(p.size, 4096);
        assert_eq!(p.chunk, parts::chunk_bytes());
        assert_eq!(p.oid.as_deref(), Some(sha.as_str()));
        assert_eq!(p.validator.as_deref(), Some("\"v9\""));
        assert_eq!(p.completed_bytes(), 0);
    }
}
