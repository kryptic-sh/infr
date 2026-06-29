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
    path::{Path, PathBuf},
};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Download a model into the HF Hub cache, returning the resolved GGUF path (a `snapshots/` symlink).
/// Idempotent: a model already cached is returned without re-downloading.
pub fn pull(r: &ModelRef) -> Result<PathBuf> {
    match r {
        ModelRef::Path(p) => Ok(p.clone()),
        ModelRef::Repo { repo, sel } => pull_repo(repo, sel.as_deref()),
    }
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
    let (commit, filename) = repo_info(repo, sel)?;
    info!("Pulling hf:{repo}:{filename}");

    let repo_dir = store.repo_dir(repo);
    let blobs = repo_dir.join("blobs");
    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    let (blob, hex, _size) =
        download_to_blob(&http_client()?, &url, token().as_deref(), &blobs, &filename)?;

    // Write the HF Hub pointers: refs/main = commit, snapshots/<commit>/<file> -> ../../blobs/<sha>.
    write_text(&repo_dir.join("refs").join("main"), &commit)?;
    let snap = repo_dir.join("snapshots").join(&commit);
    fs::create_dir_all(&snap).map_err(Error::from)?;
    let link = snap.join(&filename);
    let _ = fs::remove_file(&link); // replace a stale/dangling link
    symlink(format!("../../blobs/{hex}"), &link).map_err(Error::from)?;
    debug!("linked {link:?} -> blobs/{hex}");
    let _ = blob;
    Ok(link)
}

/// Query the HF model API for `repo`: return `(commit_sha, gguf_filename)` for selector `sel`.
fn repo_info(repo: &str, sel: Option<&str>) -> Result<(String, String)> {
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
    Ok((info.sha, file))
}

// ---------------------------------------------------------------------------
// Shared streaming download (resume + progress + sha256)
// ---------------------------------------------------------------------------

/// Stream `url` into `blobs/<sha256>` (HF's content-addressed blob name), resuming a prior partial if
/// present. Returns `(blob_path, hex_digest, total_bytes)`. On error the partial temp file is KEPT so
/// a later call resumes from where it stopped.
fn download_to_blob(
    client: &Client,
    url: &str,
    bearer: Option<&str>,
    blobs: &Path,
    label: &str,
) -> Result<(PathBuf, String, u64)> {
    fs::create_dir_all(blobs).map_err(Error::from)?;
    let tmp = blobs.join(format!(".dl-{}", sanitise(label)));
    let have = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);

    debug!("GET {url}{}", if have > 0 { " (resume)" } else { "" });
    let mut req = client.get(url);
    if let Some(t) = bearer {
        req = req.bearer_auth(t);
    }
    if have > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
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

    let mut hasher = Sha256::new();
    let mut file = if resuming {
        hash_file(&tmp, &mut hasher)?; // fold the bytes already on disk into the digest
        info!("resuming {label} at {have} bytes");
        fs::OpenOptions::new()
            .append(true)
            .open(&tmp)
            .map_err(Error::from)?
    } else {
        fs::File::create(&tmp).map_err(Error::from)? // truncates any stale partial
    };
    let start = if resuming { have } else { 0 };

    let pb = progress::bar(total, label, Unit::Bytes);
    pb.set_position(start);

    if let Err(e) = stream_into(resp, &mut file, &mut hasher, &pb) {
        pb.abandon_with_message(format!("⚠ {label} interrupted (resumable)"));
        return Err(Error::Other(format!(
            "download failed (partial kept for resume): {e}"
        )));
    }

    let hex: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let size = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
    pb.finish_with_message(format!("✓ {label} ({} MiB)", size / (1024 * 1024)));

    let blob = blobs.join(&hex); // HF blob name = bare sha256 hex
    fs::rename(&tmp, &blob).map_err(Error::from)?;
    info!("Saved blob: {blob:?}");
    Ok((blob, hex, size))
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

/// Stream the response body into `file`, updating the digest and progress bar.
fn stream_into(
    mut resp: Response,
    file: &mut fs::File,
    hasher: &mut Sha256,
    pb: &ProgressBar,
) -> std::result::Result<(), std::io::Error> {
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
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
