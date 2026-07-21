//! Model download from the HuggingFace hub over plain HTTP (reqwest) — no external CLI
//! (`huggingface-cli`) is ever invoked. Downloads stream into the **shared HF Hub cache**
//! (`~/.cache/huggingface/hub`, the same dir llama.cpp / `huggingface_hub` use) with **resume**
//! (HTTP Range) + a progress bar, and land in HF's `models--<org>--<repo>/{blobs,snapshots,refs}`
//! layout so the result is interchangeable with a `llama-cli -hf` download.

use crate::{model_ref::ModelRef, store::Store};
use indicatif::ProgressBar;
use infr_core::error::{Error, Result};
use infr_core::progress::{self, Unit};
use reqwest::blocking::{Client, Response};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    os::unix::fs::symlink,
    os::unix::io::AsRawFd,
    path::{Path, PathBuf},
};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Download a model into the HF Hub cache, returning the resolved GGUF path (a `snapshots/` symlink).
/// Idempotent: a model already cached is returned without re-downloading. Does NOT check for updates —
/// any cached snapshot satisfies it (the fast path for `infr run`). For an update check see
/// [`pull_latest`].
pub fn pull(r: &ModelRef) -> Result<PathBuf> {
    match r {
        ModelRef::Path(p) => Ok(p.clone()),
        ModelRef::Repo { repo, sel } => pull_repo(repo, sel.as_deref()),
    }
}

/// Like [`pull`] but ALWAYS queries HF for the repo's current `main` commit first and downloads when
/// the cached snapshot is missing or stale (the remote commit moved). A no-op when already up to date
/// (one cheap API call). On any network/API error, falls back to the cached copy if there is one
/// (offline-friendly). This is what `infr pull` runs so a re-pull actually picks up repo updates.
pub fn pull_latest(r: &ModelRef) -> Result<PathBuf> {
    match r {
        ModelRef::Path(p) => Ok(p.clone()),
        ModelRef::Repo { repo, sel } => pull_repo_latest(repo, sel.as_deref()),
    }
}

fn pull_repo_latest(repo: &str, sel: Option<&str>) -> Result<PathBuf> {
    let store = Store::discover()?;
    // Ask HF for the current commit + concrete gguf filename. If the API is unreachable (offline),
    // serve whatever is cached rather than failing.
    let (commit, filename, siblings) = match repo_info(repo, sel) {
        Ok(x) => x,
        Err(e) => {
            return match store.resolve_repo(repo, sel) {
                Some(p) => {
                    info!("hf:{repo}: update check failed ({e}); using cached copy");
                    Ok(p)
                }
                None => Err(e),
            };
        }
    };

    let repo_dir = store.repo_dir(repo);
    let blobs = repo_dir.join("blobs");
    let snap = repo_dir.join("snapshots").join(&commit);
    // A sharded GGUF needs its WHOLE `-NNNNN-of-MMMMM` set — one shard fails at load.
    let shards = crate::store::shard_set(&filename);
    let primary = snap.join(&shards[0]);
    // Up to date when THIS commit's snapshot already links every present shard blob.
    if shards.iter().all(|f| snap.join(f).exists()) {
        info!("hf:{repo}:{filename} already up to date ({commit})");
        // Still ensure companions — a snapshot pulled before this feature won't have them yet.
        fetch_companions(repo, &blobs, &snap, &siblings);
        return Ok(primary);
    }

    // Repoint refs/main + (re)build the snapshot. `fetch_and_link` content-addresses each shard, so a
    // commit that only moved a sibling (file bytes unchanged) relinks the cached blob — no re-download.
    info!("Updating hf:{repo}:{filename} → {commit}");
    write_text(&repo_dir.join("refs").join("main"), &commit)?;
    fs::create_dir_all(&snap).map_err(Error::from)?;
    for f in &shards {
        fetch_and_link(&blobs, &snap, repo, f)?;
    }
    fetch_companions(repo, &blobs, &snap, &siblings);
    Ok(primary)
}

/// Download one file (or reuse its content-addressed blob) into `snap` as a symlink into `blobs`.
/// Returns the snapshot symlink path. HEADs for the LFS sha256 so a present blob is relinked without
/// a download and a fresh download is verified against it.
fn fetch_and_link(blobs: &Path, snap: &Path, repo: &str, filename: &str) -> Result<PathBuf> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    let want = head_lfs_sha(repo, filename).ok().flatten();
    let hex = match &want {
        Some(sha) if blobs.join(sha).exists() => {
            debug!("hf:{repo}:{filename} blob already present; linking → {sha}");
            sha.clone()
        }
        _ => {
            download_to_blob(
                &http_client()?,
                &url,
                token().as_deref(),
                blobs,
                filename,
                want.as_deref(),
            )?
            .1
        }
    };
    let link = snap.join(filename);
    let _ = fs::remove_file(&link); // replace a stale/dangling link
    symlink(format!("../../blobs/{hex}"), &link).map_err(Error::from)?;
    debug!("linked {link:?} -> blobs/{hex}");
    Ok(link)
}

/// HEAD the resolve URL to read the file's LFS sha256 (HF's `X-Linked-Etag`) WITHOUT downloading the
/// body — so a commit bump that left the file unchanged can relink the cached blob, and so the
/// download can verify the body against it. Returns `Ok(Some(sha))` for an LFS file, `Ok(None)` for a
/// non-LFS file (no `X-Linked-Etag`), and `Err` only on a transport failure. The plain `ETag` is a
/// quoted md5, NOT the content sha256, so it is deliberately never used here — treating it as a sha
/// would defeat both the relink fast-path and integrity verification. Redirects are disabled because
/// the sha header is on huggingface.co's 302, not the CDN's final 200.
fn head_lfs_sha(repo: &str, filename: &str) -> Result<Option<String>> {
    let client = Client::builder()
        .user_agent("infr-hub/0.1")
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| Error::Other(format!("building HTTP client: {e}")))?;
    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    let mut req = client.head(&url);
    if let Some(t) = token() {
        req = req.bearer_auth(t);
    }
    let resp = req
        .send()
        .map_err(|e| Error::Other(format!("HEAD {url}: {e}")))?;
    let sha = resp
        .headers()
        .get("x-linked-etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"').to_ascii_lowercase())
        .filter(|s| is_sha256(s));
    Ok(sha)
}

/// True for a lowercase hex sha256 digest: exactly 64 hex digits.
fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn http_client() -> Result<Client> {
    Client::builder()
        .user_agent("infr-hub/0.1")
        .build()
        .map_err(|e| Error::Other(format!("building HTTP client: {e}")))
}

fn token() -> Option<String> {
    std::env::var("HF_TOKEN").ok()
}

// ---------------------------------------------------------------------------
// HuggingFace
// ---------------------------------------------------------------------------

fn pull_repo(repo: &str, sel: Option<&str>) -> Result<PathBuf> {
    let store = Store::discover()?;
    // Already cached (any matching snapshot)?
    if let Some(p) = store.resolve_repo(repo, sel) {
        debug!("hf:{repo} ({}) already cached", sel.unwrap_or("default"));
        return Ok(p);
    }

    // Resolve the repo's main commit + the concrete gguf filename for `sel` via the HF model API.
    let (commit, filename, siblings) = repo_info(repo, sel)?;
    info!("Pulling hf:{repo}:{filename}");

    let repo_dir = store.repo_dir(repo);
    let blobs = repo_dir.join("blobs");
    // Write the HF Hub pointers: refs/main = commit, snapshots/<commit>/<file> -> ../../blobs/<sha>.
    write_text(&repo_dir.join("refs").join("main"), &commit)?;
    let snap = repo_dir.join("snapshots").join(&commit);
    fs::create_dir_all(&snap).map_err(Error::from)?;
    // A sharded GGUF (`-NNNNN-of-MMMMM`) needs its WHOLE set downloaded/linked — a lone shard 1 fails
    // at load. A non-sharded file is a singleton set.
    let shards = crate::store::shard_set(&filename);
    if shards.len() > 1 {
        info!(
            "hf:{repo}:{filename} is a {}-shard split; fetching all",
            shards.len()
        );
    }
    for f in &shards {
        fetch_and_link(&blobs, &snap, repo, f)?;
    }
    fetch_companions(repo, &blobs, &snap, &siblings);
    Ok(snap.join(&shards[0]))
}

/// Query the HF model API for `repo`: return `(commit_sha, gguf_filename, sibling_filenames)` for
/// selector `sel`. The sibling list lets the caller fetch companion files (see [`fetch_companions`])
/// without a second API round-trip.
fn repo_info(repo: &str, sel: Option<&str>) -> Result<(String, String, Vec<String>)> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    debug!("GET {url}");
    let mut req = http_client()?.get(&url);
    if let Some(t) = token() {
        req = req.bearer_auth(t);
    }
    let resp = req
        .send()
        .map_err(|e| Error::Other(format!("HF API request: {e}")))?;
    if !resp.status().is_success() {
        return Err(Error::Other(format!(
            "HF API failed: HTTP {}",
            resp.status()
        )));
    }

    #[derive(serde::Deserialize)]
    struct Sibling {
        rfilename: String,
    }
    #[derive(serde::Deserialize)]
    struct ModelInfo {
        sha: String,
        siblings: Vec<Sibling>,
    }
    let info: ModelInfo = resp
        .json()
        .map_err(|e| Error::Other(format!("parsing HF API response: {e}")))?;

    let names: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();
    let file = crate::store::pick_gguf(&names, sel).ok_or_else(|| {
        Error::Other(match sel {
            Some(s) => format!("no .gguf matching '{s}' in {repo}"),
            None => format!("no .gguf files found in {repo}"),
        })
    })?;
    Ok((info.sha, file, names))
}

/// Small non-GGUF sibling files worth caching NEXT TO the GGUF. `generation_config.json` carries the
/// model's own recommended sampling (temperature/top_k/top_p) — the CLI reads it beside the model to
/// seed `infr run`/`serve` defaults (see infr-cli's `model_sampling_defaults`). Kept deliberately
/// tiny: only files the engine actually consumes belong here.
const COMPANIONS: &[&str] = &["generation_config.json"];

/// Download any [`COMPANIONS`] the repo lists into `snap` (the GGUF's snapshot dir, so they sit
/// beside it), content-addressed + symlinked exactly like the GGUF. Idempotent (skips a present
/// link) and STRICTLY NON-FATAL: a companion that's absent, unlisted, or fails to download never
/// fails the model pull — it's a convenience, not a requirement. `siblings` is the repo file list
/// already fetched by [`repo_info`], so an absent companion costs zero network calls.
fn fetch_companions(repo: &str, blobs: &Path, snap: &Path, siblings: &[String]) {
    for &name in COMPANIONS {
        if !siblings.iter().any(|s| s == name) {
            continue; // repo doesn't ship it
        }
        let link = snap.join(name);
        if link.exists() {
            continue; // already cached
        }
        let url = format!("https://huggingface.co/{repo}/resolve/main/{name}");
        // Companions are small (often non-LFS) convenience files; download best-effort, unverified.
        let dl = http_client()
            .and_then(|c| download_to_blob(&c, &url, token().as_deref(), blobs, name, None));
        match dl {
            Ok((_, hex, _)) => {
                let _ = fs::remove_file(&link);
                match symlink(format!("../../blobs/{hex}"), &link) {
                    Ok(()) => info!("hf:{repo}: cached companion {name}"),
                    Err(e) => debug!("hf:{repo}: companion {name} symlink failed: {e}"),
                }
            }
            Err(e) => debug!("hf:{repo}: companion {name} not cached ({e})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared streaming download (resume + progress + sha256)
// ---------------------------------------------------------------------------

/// Stream `url` into `blobs/<sha256>` (HF's content-addressed blob name), resuming a prior partial if
/// present. Returns `(blob_path, hex_digest, total_bytes)`. On a transport error the partial temp file
/// is KEPT so a later call resumes from where it stopped.
///
/// When `expected_sha` is `Some`, it is HF's advertised LFS sha256 and is used two ways: the download
/// is skipped entirely if that content-addressed blob is already on disk, and the downloaded bytes are
/// verified against it before the blob is committed — a mismatch discards the temp and errors (a
/// corrupt/truncated body, or a resume of a stale partial from a since-changed file, must never be
/// linked as the model). `None` (non-LFS file / no digest available) proceeds without verification.
fn download_to_blob(
    client: &Client,
    url: &str,
    bearer: Option<&str>,
    blobs: &Path,
    label: &str,
    expected_sha: Option<&str>,
) -> Result<(PathBuf, String, u64)> {
    fs::create_dir_all(blobs).map_err(Error::from)?;
    // Content-addressed short-circuit: if we already know the sha and hold that blob, we're done.
    if let Some(sha) = expected_sha {
        let blob = blobs.join(sha);
        if blob.exists() {
            let size = fs::metadata(&blob).map(|m| m.len()).unwrap_or(0);
            debug!("blob {sha} already present ({size} bytes); skipping download of {label}");
            return Ok((blob, sha.to_string(), size));
        }
    } else {
        debug!("no expected sha256 for {label}; download will not be integrity-checked");
    }
    let stem = sanitise(label);
    let tmp = blobs.join(format!(".dl-{stem}"));
    let meta = blobs.join(format!(".dl-{stem}.meta")); // stored If-Range validator for the partial

    // Serialize concurrent pulls of the SAME blob (auto-pull racing a manual `pull`, two `run`s).
    // An advisory `flock` on a per-blob lockfile is chosen over unique-per-process temp names ON
    // PURPOSE: it PRESERVES resume — the one shared temp keeps accumulating across processes instead
    // of each starting a fresh partial from byte 0. The lock releases when `_lock` drops.
    let _lock = FileLock::acquire(&blobs.join(format!(".dl-{stem}.lock")))?;
    // Re-check the content-addressed short-circuit now that we hold the lock: a racing process may
    // have finished this exact blob while we waited.
    if let Some(sha) = expected_sha {
        let blob = blobs.join(sha);
        if blob.exists() {
            let size = fs::metadata(&blob).map(|m| m.len()).unwrap_or(0);
            return Ok((blob, sha.to_string(), size));
        }
    }

    let have = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
    // Validator captured when THIS partial was first written; only meaningful with bytes on disk.
    let validator = (have > 0)
        .then(|| fs::read_to_string(&meta).ok())
        .flatten()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    debug!("GET {url}{}", if have > 0 { " (resume)" } else { "" });
    let mut req = client.get(url);
    if let Some(t) = bearer {
        req = req.bearer_auth(t);
    }
    // Resume with `If-Range`: if the object changed since the partial was written, the server
    // ignores the Range and sends a full 200 → we restart clean instead of splicing new bytes onto a
    // stale prefix (an undetectable corruption without the end-of-body sha check).
    if let Some((range, if_range)) = resume_headers(have, validator.as_deref()) {
        req = req.header(reqwest::header::RANGE, range);
        if let Some(v) = if_range {
            req = req.header(reqwest::header::IF_RANGE, v);
        }
    }
    let resp = req
        .send()
        .map_err(|e| Error::Other(format!("HTTP request: {e}")))?;
    if !resp.status().is_success() {
        return Err(Error::Other(format!(
            "download failed: HTTP {}",
            resp.status()
        )));
    }
    // The server honours the Range only with 206; on 200 it sends the whole file → restart clean.
    let resuming = have > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let remaining = resp.content_length();
    let total = remaining.map(|r| if resuming { have + r } else { r });

    // Persist the validator for a FUTURE resume — before streaming, so an interrupt mid-body still
    // leaves a usable `If-Range` for the next attempt. On a 206 the stored validator still matches
    // (the server accepted it), so only refresh it on a fresh/200 body.
    if !resuming {
        match response_validator(resp.headers()) {
            Some(v) => {
                let _ = fs::write(&meta, v);
            }
            None => {
                let _ = fs::remove_file(&meta);
            }
        }
    }

    let mut file = if resuming {
        info!("resuming {label} at {have} bytes");
        fs::OpenOptions::new()
            .append(true)
            .open(&tmp)
            .map_err(Error::from)?
    } else {
        fs::File::create(&tmp).map_err(Error::from)? // truncates any stale partial (changed object)
    };
    let start = if resuming { have } else { 0 };

    let pb = progress::bar(total, label, Unit::Bytes);
    pb.set_position(start);

    if let Err(e) = stream_into(resp, &mut file, &pb) {
        pb.abandon_with_message(format!("⚠ {label} interrupted (resumable)"));
        return Err(Error::Other(format!(
            "download failed (partial kept for resume): {e}"
        )));
    }
    drop(file); // flush + close before re-reading for the digest

    // Hash the COMPLETE file ONCE at the end. The old code folded the on-disk prefix into the digest
    // on every resume (O(K·size) over K flaky-link retries); since the whole body is sha-verified
    // here anyway, a single final pass is equivalent and cheaper.
    let mut hasher = Sha256::new();
    hash_file(&tmp, &mut hasher)?;
    let hex: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let size = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);

    // Integrity gate: the body MUST hash to HF's advertised LFS sha256. On mismatch discard the temp
    // (do NOT keep it for resume — a resumed corrupt prefix stays corrupt) and fail loudly.
    if let Err(e) = verify_sha(label, &hex, expected_sha) {
        pb.abandon_with_message(format!("⚠ {label} sha256 mismatch"));
        let _ = fs::remove_file(&tmp);
        let _ = fs::remove_file(&meta);
        return Err(e);
    }
    pb.finish_with_message(format!("✓ {label} ({} MiB)", size / (1024 * 1024)));

    let blob = blobs.join(&hex); // HF blob name = bare sha256 hex
    fs::rename(&tmp, &blob).map_err(Error::from)?;
    let _ = fs::remove_file(&meta); // partial committed; validator no longer needed
    info!("Saved blob: {blob:?}");
    Ok((blob, hex, size))
}

/// Build the resume request directives: the `Range` value and an optional `If-Range` value. Returns
/// `None` when there is nothing on disk to resume (`have == 0`). `If-Range` is omitted when no
/// validator was stored (a partial from before this feature) — the server may then splice, but the
/// end-of-download sha256 verification still catches it.
fn resume_headers(have: u64, validator: Option<&str>) -> Option<(String, Option<String>)> {
    if have == 0 {
        return None;
    }
    Some((format!("bytes={have}-"), validator.map(str::to_string)))
}

/// The value to persist as the `If-Range` validator for a partial: the strong `ETag` if present,
/// else `Last-Modified`. Either is an opaque object identity the server compares on the next resume.
fn response_validator(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(reqwest::header::ETAG)
        .or_else(|| headers.get(reqwest::header::LAST_MODIFIED))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
}

/// An advisory exclusive `flock` on a lockfile, released when dropped. Serializes concurrent
/// downloads of the same blob so two processes can't interleave writes into the shared temp.
struct FileLock {
    _file: fs::File,
}

impl FileLock {
    fn acquire(path: &Path) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
            .map_err(Error::from)?;
        // Blocks until any other holder releases. `flock` is process-associated and auto-releases if
        // the holder dies (crash-safe — no stale lock).
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        if rc != 0 {
            return Err(Error::Other(format!(
                "flock {path:?}: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(FileLock { _file: file })
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // Closing the fd releases the lock; the explicit unlock is belt-and-suspenders.
        unsafe { libc::flock(self._file.as_raw_fd(), libc::LOCK_UN) };
    }
}

/// Assert the downloaded digest `hex` matches the `expected` LFS sha256 (case-insensitive). `None`
/// means no digest was available (non-LFS file) → verification is skipped.
fn verify_sha(label: &str, hex: &str, expected: Option<&str>) -> Result<()> {
    match expected {
        Some(exp) if !hex.eq_ignore_ascii_case(exp) => Err(Error::Other(format!(
            "sha256 mismatch for {label}: expected {exp}, got {hex} — corrupt download discarded"
        ))),
        _ => Ok(()),
    }
}

/// Read an existing file fully through `hasher` (to continue a resumed digest).
fn hash_file(path: &Path, hasher: &mut Sha256) -> Result<()> {
    let mut f = fs::File::open(path).map_err(Error::from)?;
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf).map_err(Error::from)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(())
}

/// Stream the response body into `file`, advancing the progress bar. The digest is computed in a
/// single final pass over the completed file (see [`download_to_blob`]), not here.
fn stream_into(
    mut resp: Response,
    file: &mut fs::File,
    pb: &ProgressBar,
) -> std::result::Result<(), std::io::Error> {
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        pb.inc(n as u64);
    }
    file.flush()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write `text` to `path`, creating parent directories.
fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(Error::from)?;
    }
    fs::write(path, text).map_err(Error::from)?;
    Ok(())
}

/// Replace characters unsafe in a filename with `_`.
fn sanitise(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    #[test]
    fn is_sha256_shape() {
        assert!(is_sha256(SHA_A));
        assert!(is_sha256(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        ));
        assert!(!is_sha256(&"a".repeat(63))); // too short
        assert!(!is_sha256(&"a".repeat(65))); // too long
        assert!(!is_sha256(&"g".repeat(64))); // non-hex
        assert!(!is_sha256("d41d8cd98f00b204e9800998ecf8427e")); // md5 (32 chars)
        assert!(!is_sha256(""));
    }

    #[test]
    fn resume_headers_build() {
        // Nothing on disk → no resume directives.
        assert_eq!(resume_headers(0, Some("etag")), None);
        assert_eq!(resume_headers(0, None), None);
        // Bytes on disk with a stored validator → Range + If-Range.
        assert_eq!(
            resume_headers(100, Some("\"abc\"")),
            Some(("bytes=100-".to_string(), Some("\"abc\"".to_string())))
        );
        // Bytes but no validator (pre-feature partial) → Range only.
        assert_eq!(
            resume_headers(100, None),
            Some(("bytes=100-".to_string(), None))
        );
    }

    #[test]
    fn response_validator_prefers_etag() {
        use reqwest::header::{HeaderMap, HeaderValue, ETAG, LAST_MODIFIED};
        let mut h = HeaderMap::new();
        assert_eq!(response_validator(&h), None);
        h.insert(
            LAST_MODIFIED,
            HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
        );
        assert_eq!(
            response_validator(&h).as_deref(),
            Some("Wed, 21 Oct 2026 07:28:00 GMT")
        );
        h.insert(ETAG, HeaderValue::from_static("\"deadbeef\""));
        assert_eq!(response_validator(&h).as_deref(), Some("\"deadbeef\""));
    }

    #[test]
    fn file_lock_is_exclusive() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("blob.lock");
        let guard = FileLock::acquire(&path).unwrap();
        // A second exclusive lock on the same file (separate fd) must NOT be grantable while held.
        let other = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .unwrap();
        let rc = unsafe { libc::flock(other.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_ne!(rc, 0, "second flock should fail while the first is held");
        // After the first releases, the lock is grantable again.
        drop(guard);
        let rc = unsafe { libc::flock(other.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(rc, 0, "flock should succeed once the holder drops");
        unsafe { libc::flock(other.as_raw_fd(), libc::LOCK_UN) };
    }

    #[test]
    fn verify_sha_gate() {
        // Match (case-insensitive) passes.
        assert!(verify_sha("f", SHA_A, Some(SHA_A)).is_ok());
        assert!(verify_sha("f", SHA_A, Some(&SHA_A.to_ascii_uppercase())).is_ok());
        // No expected digest → skipped (non-LFS best-effort).
        assert!(verify_sha("f", SHA_A, None).is_ok());
        // Mismatch fails loudly.
        let other = "b".repeat(64);
        assert!(verify_sha("f", SHA_A, Some(&other)).is_err());
    }
}
