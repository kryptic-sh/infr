//! The shared on-disk model store (Ollama-compatible layout).
//!
//! Layout inside `root`:
//! ```text
//! manifests/registry.ollama.ai/library/<name>/<tag>   (OCI-style JSON)
//! blobs/sha256-<hex>                                  (layer blobs; model layer == GGUF)
//! ```

use crate::model_ref::ModelRef;
use infr_core::error::{Error, Result};
use serde::Deserialize;
use std::{fs, path::PathBuf};

// ---------------------------------------------------------------------------
// Serde structs for Ollama manifests
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct OllamaManifest {
    layers: Vec<OllamaLayer>,
}

#[derive(Deserialize)]
struct OllamaLayer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// The shared on-disk model store (Ollama-compatible).
pub struct Store {
    pub root: PathBuf,
}

impl Store {
    /// Locate the store root: `$INFR_MODELS` → `$OLLAMA_MODELS` → `~/.ollama/models` (if it exists)
    /// → `/var/lib/ollama` (the systemd-service store, if it exists) → `~/.ollama/models`.
    ///
    /// The chosen directory is not required to exist (the final fallback may be absent).
    pub fn discover() -> Result<Self> {
        let root = if let Ok(p) = std::env::var("INFR_MODELS") {
            PathBuf::from(p)
        } else if let Ok(p) = std::env::var("OLLAMA_MODELS") {
            PathBuf::from(p)
        } else {
            let home = dirs::home_dir()
                .ok_or_else(|| Error::Other("cannot determine home directory".into()))?;
            let user_store = home.join(".ollama").join("models");
            let systemd_store = PathBuf::from("/var/lib/ollama");
            if user_store.exists() {
                user_store
            } else if systemd_store.exists() {
                systemd_store
            } else {
                user_store
            }
        };
        Ok(Store { root })
    }

    /// Return the blobs directory (`<root>/blobs`).
    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs")
    }

    /// If the referenced model already exists locally, return the GGUF blob path.
    ///
    /// - `Path(p)` → `Some(p)` if the file exists.
    /// - `Hf`     → `Ok(None)` (HF refs are resolved only by `pull`).
    /// - `Ollama` → read the Ollama manifest, find the model layer digest,
    ///              return the blob path if present.
    pub fn resolve(&self, r: &ModelRef) -> Result<Option<PathBuf>> {
        match r {
            ModelRef::Path(p) => {
                if p.exists() {
                    Ok(Some(p.clone()))
                } else {
                    Ok(None)
                }
            }
            ModelRef::Hf { .. } => Ok(None),
            ModelRef::Ollama { name, tag } => self.resolve_ollama(name, tag),
        }
    }

    fn resolve_ollama(&self, name: &str, tag: &str) -> Result<Option<PathBuf>> {
        // Primary path: manifests/registry.ollama.ai/library/<name>/<tag>
        let primary = self
            .root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join(name)
            .join(tag);

        // Secondary path (only when name already contains a namespace slash):
        // manifests/registry.ollama.ai/<name>/<tag>
        // e.g. name="library/qwen" → manifests/registry.ollama.ai/library/qwen/<tag>
        let secondary = if name.contains('/') {
            Some(
                self.root
                    .join("manifests")
                    .join("registry.ollama.ai")
                    .join(name)
                    .join(tag),
            )
        } else {
            None
        };

        // Pick whichever manifest file exists; prefer primary.
        let manifest_path = if primary.exists() {
            primary
        } else if let Some(sec) = secondary {
            if sec.exists() {
                sec
            } else {
                return Ok(None);
            }
        } else {
            return Ok(None);
        };

        self.blob_from_manifest(&manifest_path)
    }

    /// Parse a manifest file and return the GGUF blob path if it exists.
    fn blob_from_manifest(&self, manifest_path: &std::path::Path) -> Result<Option<PathBuf>> {
        let content = fs::read_to_string(manifest_path).map_err(|e| {
            Error::Other(format!("reading manifest {}: {e}", manifest_path.display()))
        })?;

        let manifest: OllamaManifest = serde_json::from_str(&content)
            .map_err(|e| Error::Other(format!("parsing manifest: {e}")))?;

        let model_layer = match manifest
            .layers
            .iter()
            .find(|l| l.media_type == "application/vnd.ollama.image.model")
        {
            Some(l) => l,
            None => return Ok(None),
        };

        // digest is "sha256:abc123…" → blob filename is "sha256-abc123…"
        let blob_name = model_layer.digest.replace(':', "-");
        let blob_path = self.root.join("blobs").join(blob_name);

        if blob_path.exists() {
            Ok(Some(blob_path))
        } else {
            Ok(None)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_manifest(digest: &str) -> String {
        serde_json::json!({
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
                    "size": 42
                }
            ]
        })
        .to_string()
    }

    /// Write a manifest + blob in a temp store and assert resolve finds the blob.
    #[test]
    fn resolve_ollama_simple_name() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();

        let digest = "sha256:aabbccddeeff001122334455667788990011223344556677889900aabbccddeeff";
        let blob_name = digest.replace(':', "-");

        // Manifest lives at: <root>/manifests/registry.ollama.ai/library/testmodel/latest
        // The tag ("latest") is the filename, not a directory.
        let manifest_parent = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join("testmodel");
        fs::create_dir_all(&manifest_parent).unwrap();
        fs::write(manifest_parent.join("latest"), fake_manifest(digest)).unwrap();

        // Write blob
        let blobs_dir = root.join("blobs");
        fs::create_dir_all(&blobs_dir).unwrap();
        fs::write(blobs_dir.join(&blob_name), b"fake gguf data").unwrap();

        let store = Store { root };
        let mr = ModelRef::Ollama {
            name: "testmodel".into(),
            tag: "latest".into(),
        };
        let result = store.resolve(&mr).unwrap();
        assert!(result.is_some(), "expected blob path, got None");
        assert!(result.unwrap().exists());
    }

    /// A namespaced Ollama ref should resolve via the secondary path.
    #[test]
    fn resolve_ollama_namespaced() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();

        let digest = "sha256:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";
        let blob_name = digest.replace(':', "-");

        // Write manifest at registry.ollama.ai/library/qwen2.5/latest
        // (secondary path for name="library/qwen2.5")
        let manifest_dir = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join("qwen2.5");
        fs::create_dir_all(&manifest_dir).unwrap();
        fs::write(manifest_dir.join("latest"), fake_manifest(digest)).unwrap();

        // Write blob
        let blobs_dir = root.join("blobs");
        fs::create_dir_all(&blobs_dir).unwrap();
        fs::write(blobs_dir.join(&blob_name), b"fake gguf").unwrap();

        let store = Store { root };
        let mr = ModelRef::Ollama {
            name: "library/qwen2.5".into(),
            tag: "latest".into(),
        };
        let result = store.resolve(&mr).unwrap();
        assert!(result.is_some(), "expected blob path, got None");
    }

    /// Missing model (no manifest file) should return Ok(None).
    #[test]
    fn resolve_missing_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store {
            root: tmp.path().to_path_buf(),
        };
        let mr = ModelRef::Ollama {
            name: "doesnotexist".into(),
            tag: "latest".into(),
        };
        assert_eq!(store.resolve(&mr).unwrap(), None);
    }

    /// Manifest present but blob missing → Ok(None).
    #[test]
    fn resolve_manifest_no_blob() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();

        let digest = "sha256:0000000000000000000000000000000000000000000000000000000000000001";
        let manifest_dir = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join("ghostmodel");
        fs::create_dir_all(&manifest_dir).unwrap();
        fs::write(manifest_dir.join("v1"), fake_manifest(digest)).unwrap();
        // intentionally do NOT create the blob

        let store = Store { root };
        let mr = ModelRef::Ollama {
            name: "ghostmodel".into(),
            tag: "v1".into(),
        };
        assert_eq!(store.resolve(&mr).unwrap(), None);
    }

    /// HF refs always resolve to None (network-only).
    #[test]
    fn resolve_hf_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store {
            root: tmp.path().to_path_buf(),
        };
        let mr = ModelRef::Hf {
            repo: "org/repo".into(),
            file: None,
        };
        assert_eq!(store.resolve(&mr).unwrap(), None);
    }

    /// Path variant returns Some when file exists.
    #[test]
    fn resolve_path_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf = tmp.path().join("model.gguf");
        fs::write(&gguf, b"fake").unwrap();
        let store = Store {
            root: tmp.path().to_path_buf(),
        };
        let mr = ModelRef::Path(gguf.clone());
        assert_eq!(store.resolve(&mr).unwrap(), Some(gguf));
    }

    /// Path variant returns None when file is absent.
    #[test]
    fn resolve_path_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store {
            root: tmp.path().to_path_buf(),
        };
        let mr = ModelRef::Path(tmp.path().join("nofile.gguf"));
        assert_eq!(store.resolve(&mr).unwrap(), None);
    }
}
