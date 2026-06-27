//! Model download: HuggingFace hub + Ollama registry. Both are standalone HTTP (reqwest) — no
//! external CLI (`ollama` / `huggingface-cli`) is ever invoked. Downloads stream into our own
//! content-addressed blob store with **resume** (HTTP Range) and a progress bar, and are
//! sha256-verified (against the registry digest for Ollama; by computed hash for HF).

use crate::{
    model_ref::ModelRef,
    store::{OllamaManifest, Store, OLLAMA_MODEL_MEDIA_TYPE},
};
use indicatif::ProgressBar;
use infr_core::error::{Error, Result};
use infr_core::progress::{self, Unit};
use reqwest::blocking::{Client, Response};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Download a model into our store, returning the GGUF blob path. Idempotent: a model already in
/// the store is returned without re-downloading.
///
/// - `Path` → returned as-is (no download).
/// - `Hf`   → streamed from HuggingFace.
/// - `Ollama` → manifest + blob streamed from the Ollama registry.
pub fn pull(r: &ModelRef) -> Result<PathBuf> {
    match r {
        ModelRef::Path(p) => Ok(p.clone()),
        ModelRef::Hf { repo, file } => pull_hf(repo, file.as_deref()),
        ModelRef::Ollama { name, tag } => pull_ollama(name, tag),
    }
}

fn http_client() -> Result<Client> {
    Client::builder()
        .user_agent("infr-hub/0.1")
        .build()
        .map_err(|e| Error::Other(format!("building HTTP client: {e}")))
}

// ---------------------------------------------------------------------------
// HuggingFace
// ---------------------------------------------------------------------------

fn pull_hf(repo: &str, file: Option<&str>) -> Result<PathBuf> {
    let store = Store::discover()?;
    let filename = match file {
        Some(f) => f.to_owned(),
        None => choose_hf_file(repo)?,
    };

    // Idempotent: already in the store?
    let cached = ModelRef::Hf {
        repo: repo.to_owned(),
        file: Some(filename.clone()),
    };
    if let Some(p) = store.resolve(&cached)? {
        debug!("hf:{repo}:{filename} already cached");
        return Ok(p);
    }

    info!("Pulling hf:{repo}:{filename}");
    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    let token = std::env::var("HF_TOKEN").ok();
    let (blob, hex, size) =
        download_to_blob(&http_client()?, &url, token.as_deref(), &store, &filename)?;

    write_hf_manifest(&store, repo, &filename, &format!("sha256:{hex}"), size)?;
    Ok(blob)
}

/// Query the HF model API and choose a `.gguf` file (prefer `Q4_K_M`).
fn choose_hf_file(repo: &str) -> Result<String> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    debug!("GET {url}");
    let mut req = http_client()?.get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        req = req.bearer_auth(token);
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
        siblings: Vec<Sibling>,
    }
    let info: ModelInfo = resp
        .json()
        .map_err(|e| Error::Other(format!("parsing HF API response: {e}")))?;

    let gguf: Vec<String> = info
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .filter(|f| f.ends_with(".gguf"))
        .collect();
    if gguf.is_empty() {
        return Err(Error::Other(format!("no .gguf files found in {repo}")));
    }
    if let Some(f) = gguf.iter().find(|f| f.to_lowercase().contains("q4_k_m")) {
        return Ok(f.clone());
    }
    Ok(gguf.into_iter().next().unwrap())
}

/// Construct a minimal Ollama-style manifest for an HF-downloaded blob so `resolve` can find it.
fn write_hf_manifest(
    store: &Store,
    repo: &str,
    filename: &str,
    digest: &str,
    size: u64,
) -> Result<()> {
    let manifest = serde_json::json!({
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": { "mediaType": "application/vnd.ollama.image.config", "digest": digest, "size": 0 },
        "layers": [ { "mediaType": OLLAMA_MODEL_MEDIA_TYPE, "digest": digest, "size": size } ]
    });
    write_text(
        &store.hf_manifest_path(repo, filename),
        &serde_json::to_string_pretty(&manifest).unwrap(),
    )
}

// ---------------------------------------------------------------------------
// Ollama registry
// ---------------------------------------------------------------------------

fn pull_ollama(name: &str, tag: &str) -> Result<PathBuf> {
    let store = Store::discover()?;
    let cached = ModelRef::Ollama {
        name: name.to_owned(),
        tag: tag.to_owned(),
    };
    if let Some(p) = store.resolve(&cached)? {
        debug!("ollama:{name}:{tag} already cached");
        return Ok(p);
    }

    let full = Store::ollama_full_name(name);
    info!("Pulling ollama:{name}:{tag}");
    let client = http_client()?;

    // 1. Manifest (the registry needs the docker manifest Accept header).
    let manifest_url = format!("https://registry.ollama.ai/v2/{full}/manifests/{tag}");
    debug!("GET {manifest_url}");
    let resp = client
        .get(&manifest_url)
        .header(
            reqwest::header::ACCEPT,
            "application/vnd.docker.distribution.manifest.v2+json",
        )
        .send()
        .map_err(|e| Error::Other(format!("ollama manifest request: {e}")))?;
    if !resp.status().is_success() {
        return Err(Error::Other(format!(
            "ollama manifest failed: HTTP {} for {name}:{tag}",
            resp.status()
        )));
    }
    let manifest_text = resp
        .text()
        .map_err(|e| Error::Other(format!("reading ollama manifest: {e}")))?;
    let manifest: OllamaManifest = serde_json::from_str(&manifest_text)
        .map_err(|e| Error::Other(format!("parsing ollama manifest: {e}")))?;
    let layer = manifest
        .layers
        .iter()
        .find(|l| l.media_type == OLLAMA_MODEL_MEDIA_TYPE)
        .ok_or_else(|| Error::Other(format!("ollama:{name}:{tag} has no model layer")))?;
    let digest = layer.digest.clone(); // "sha256:<hex>"
    let want_hex = digest.strip_prefix("sha256:").unwrap_or(&digest).to_owned();

    // 2. Blob (content-addressed → skip if another tag already pulled the same weights).
    let blob_path = store.blobs_dir().join(format!("sha256-{want_hex}"));
    if blob_path.exists() {
        debug!("ollama blob {want_hex} already present");
    } else {
        let blob_url = format!("https://registry.ollama.ai/v2/{full}/blobs/{digest}");
        let (blob, got_hex, _size) = download_to_blob(&client, &blob_url, None, &store, &want_hex)?;
        if got_hex != want_hex {
            let _ = fs::remove_file(&blob);
            return Err(Error::Other(format!(
                "ollama blob digest mismatch: got {got_hex}, want {want_hex}"
            )));
        }
    }

    // 3. Persist the manifest in our store so future runs resolve without a network call.
    write_text(&store.ollama_manifest_path(name, tag), &manifest_text)?;
    Ok(blob_path)
}

// ---------------------------------------------------------------------------
// Shared streaming download (resume + progress + sha256)
// ---------------------------------------------------------------------------

/// Stream `url` into the content-addressed blob store, resuming a prior partial if present.
/// Returns `(blob_path, hex_digest, total_bytes)`. `label` names the temp file + progress message.
/// On error the partial temp file is KEPT so a later call resumes from where it stopped.
fn download_to_blob(
    client: &Client,
    url: &str,
    bearer: Option<&str>,
    store: &Store,
    label: &str,
) -> Result<(PathBuf, String, u64)> {
    let blobs = store.blobs_dir();
    fs::create_dir_all(&blobs).map_err(Error::from)?;
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

    let blob = blobs.join(format!("sha256-{hex}"));
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
    debug!("wrote manifest {path:?}");
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
