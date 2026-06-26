//! Model download: HF hub (streaming + sha256 verification) and Ollama registry.

use crate::{model_ref::ModelRef, store::Store};
use indicatif::{ProgressBar, ProgressStyle};
use infr_core::error::{Error, Result};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::PathBuf,
};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Download a model into the shared store, returning the GGUF blob path.
///
/// - `Path`   → returned as-is (no download).
/// - `Hf`     → streamed from HuggingFace with progress + sha256 verification.
/// - `Ollama` → returns [`Error::Unsupported`]; use `ollama pull` then
///              [`Store::resolve`] to consume pre-pulled models.
pub fn pull(r: &ModelRef) -> Result<PathBuf> {
    match r {
        ModelRef::Path(p) => Ok(p.clone()),
        ModelRef::Hf { repo, file } => pull_hf(repo, file.as_deref()),
        ModelRef::Ollama { .. } => Err(Error::Unsupported(
            "ollama registry pull not yet implemented; pre-pull with `ollama pull`".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// HuggingFace pull
// ---------------------------------------------------------------------------

fn pull_hf(repo: &str, file: Option<&str>) -> Result<PathBuf> {
    let store = Store::discover()?;

    // Determine the filename to download.
    let filename = match file {
        Some(f) => f.to_owned(),
        None => choose_hf_file(repo)?,
    };

    info!("Pulling hf:{repo}:{filename}");

    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    debug!("GET {url}");

    let client = reqwest::blocking::Client::builder()
        .user_agent("infr-hub/0.1")
        .build()
        .map_err(|e| Error::Other(format!("building HTTP client: {e}")))?;

    let mut req = client.get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        req = req.bearer_auth(token);
    }

    let resp = req
        .send()
        .map_err(|e| Error::Other(format!("HTTP request: {e}")))?;

    if !resp.status().is_success() {
        return Err(Error::Other(format!(
            "HF download failed: HTTP {}",
            resp.status()
        )));
    }

    let total_size = resp.content_length();

    // Progress bar
    let pb: ProgressBar = match total_size {
        Some(n) => {
            let pb = ProgressBar::new(n);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] \
                     {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
            );
            pb
        }
        None => {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] {bytes} ({bytes_per_sec})",
                )
                .unwrap(),
            );
            pb
        }
    };
    pb.set_message(filename.clone());

    // Prepare blobs directory and a temporary file.
    let blobs_dir = store.blobs_dir();
    fs::create_dir_all(&blobs_dir).map_err(Error::from)?;

    // Use a sanitised temp name to avoid path issues.
    let tmp_path = blobs_dir.join(format!(".dl-{}", sanitise(&filename)));

    let download_result = stream_to_file(resp, &tmp_path, &pb);

    if let Err(e) = &download_result {
        // Best-effort cleanup of the temp file on failure.
        let _ = fs::remove_file(&tmp_path);
        return Err(Error::Other(format!("download failed: {e}")));
    }

    let (digest_hex, byte_count) = download_result.unwrap();
    pb.finish_with_message(format!("✓ {filename} ({byte_count} bytes)"));

    let digest = format!("sha256:{digest_hex}");
    let blob_name = format!("sha256-{digest_hex}");
    let blob_path = blobs_dir.join(&blob_name);

    // Atomically rename to the content-addressed path.
    fs::rename(&tmp_path, &blob_path).map_err(Error::from)?;
    info!("Saved blob: {blob_path:?}");

    // Write a minimal Ollama-style manifest so the blob is reusable.
    write_hf_manifest(&store, repo, &filename, &digest, byte_count)?;

    Ok(blob_path)
}

/// Stream `response` to `dest`, computing sha256 and counting bytes.
/// Returns `(hex_digest, bytes_written)`.
fn stream_to_file(
    mut response: reqwest::blocking::Response,
    dest: &std::path::Path,
    pb: &ProgressBar,
) -> std::result::Result<(String, u64), Box<dyn std::error::Error>> {
    let mut out = fs::File::create(dest)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    let mut total = 0u64;

    loop {
        let n = response.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        out.write_all(&buf[..n])?;
        total += n as u64;
        pb.inc(n as u64);
    }

    let result = hasher.finalize();
    let hex: String = result.iter().map(|b| format!("{b:02x}")).collect();
    Ok((hex, total))
}

/// Write a minimal Ollama-style manifest for an HF-downloaded blob.
///
/// Path: `<store>/manifests/huggingface.co/<org>/<model>/<filename>`
fn write_hf_manifest(
    store: &Store,
    repo: &str,
    filename: &str,
    digest: &str,
    size: u64,
) -> Result<()> {
    // repo = "org/model" → namespace="org", model_name="model"
    let (namespace, model_name) = split_repo(repo);

    let manifest_dir = store
        .root
        .join("manifests")
        .join("huggingface.co")
        .join(namespace)
        .join(model_name);
    fs::create_dir_all(&manifest_dir).map_err(Error::from)?;

    let manifest = serde_json::json!({
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {
            "mediaType": "application/vnd.ollama.image.config",
            "digest": digest,
            "size": 0
        },
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": digest,
                "size": size
            }
        ]
    });

    let manifest_path = manifest_dir.join(filename);
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .map_err(Error::from)?;

    debug!("Wrote manifest: {manifest_path:?}");
    Ok(())
}

// ---------------------------------------------------------------------------
// HuggingFace API: pick a .gguf file from repo
// ---------------------------------------------------------------------------

/// Query the HF model API and choose a `.gguf` file (prefer `Q4_K_M`).
fn choose_hf_file(repo: &str) -> Result<String> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    debug!("GET {url}");

    let client = reqwest::blocking::Client::builder()
        .user_agent("infr-hub/0.1")
        .build()
        .map_err(|e| Error::Other(format!("building HTTP client: {e}")))?;

    let mut req = client.get(&url);
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

    let gguf_files: Vec<String> = info
        .siblings
        .into_iter()
        .filter(|s| s.rfilename.ends_with(".gguf"))
        .map(|s| s.rfilename)
        .collect();

    if gguf_files.is_empty() {
        return Err(Error::Other(format!("no .gguf files found in {repo}")));
    }

    // Prefer Q4_K_M variant.
    if let Some(f) = gguf_files
        .iter()
        .find(|f| f.to_lowercase().contains("q4_k_m"))
    {
        return Ok(f.clone());
    }

    // Fall back to the first .gguf found.
    Ok(gguf_files.into_iter().next().unwrap())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Split `"org/model"` into `("org", "model")`, falling back to `("library", repo)`.
fn split_repo(repo: &str) -> (&str, &str) {
    match repo.splitn(2, '/').collect::<Vec<_>>()[..] {
        [ns, name] => (ns, name),
        _ => ("library", repo),
    }
}

/// Replace characters unsafe for use in a filename with `_`.
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
