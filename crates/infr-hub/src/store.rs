//! infr's on-disk model store — the **standard HuggingFace Hub cache** (`~/.cache/huggingface/hub`),
//! shared with llama.cpp, `huggingface_hub`, and `transformers`. So `infr run hf:org/repo:Q4_K_M` and
//! `llama-cli -hf org/repo:Q4_K_M` hit the same files — one download, no duplication:
//!
//! ```text
//! <hub>/models--<org>--<repo>/
//!   refs/main                       -> <commit>
//!   blobs/<sha256>                     the file bytes (content-addressed; bare hex, no prefix)
//!   snapshots/<commit>/<file.gguf>  -> ../../blobs/<sha256>   (symlink with the real filename)
//! ```

use crate::model_ref::ModelRef;
use infr_core::error::{Error, Result};
use std::{fs, path::PathBuf};

/// Default quant when an `hf:` ref gives only `org/repo` (matches llama.cpp's `-hf`).
pub(crate) const DEFAULT_QUANT: &str = "Q4_K_M";

/// The HuggingFace Hub cache.
pub struct Store {
    pub hub: PathBuf,
}

impl Store {
    /// Locate the HF Hub cache: `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`.
    pub fn discover() -> Result<Self> {
        let hub = if let Ok(p) = std::env::var("HF_HUB_CACHE") {
            PathBuf::from(p)
        } else if let Ok(h) = std::env::var("HF_HOME") {
            PathBuf::from(h).join("hub")
        } else {
            dirs::cache_dir()
                .ok_or_else(|| Error::Other("cannot determine cache directory".into()))?
                .join("huggingface")
                .join("hub")
        };
        Ok(Store { hub })
    }

    /// `<hub>/models--<org>--<repo>` — the HF Hub repo dir (HF replaces `/` with `--`).
    pub fn repo_dir(&self, repo: &str) -> PathBuf {
        self.hub
            .join(format!("models--{}", repo.replace('/', "--")))
    }

    /// Resolve a cached GGUF for `repo` selecting `sel` (a quant like `Q4_K_M`, or an explicit
    /// `*.gguf` filename; `None` → [`DEFAULT_QUANT`]). Scans the repo's `snapshots/*/` dirs and
    /// returns the snapshot path (a symlink into `blobs/`) whose blob is present.
    pub fn resolve_repo(&self, repo: &str, sel: Option<&str>) -> Option<PathBuf> {
        let snaps = self.repo_dir(repo).join("snapshots");
        let sel = sel.unwrap_or(DEFAULT_QUANT);
        let mut fallback: Option<PathBuf> = None;
        for snap in fs::read_dir(&snaps).ok()?.flatten() {
            for f in fs::read_dir(snap.path()).into_iter().flatten().flatten() {
                let name = f.file_name().to_string_lossy().into_owned();
                if !name.to_lowercase().ends_with(".gguf") {
                    continue;
                }
                let p = f.path();
                if !p.exists() {
                    continue; // dangling symlink (blob garbage-collected)
                }
                match gguf_match(&name, sel) {
                    Match::Exact => return Some(p),
                    Match::Loose => fallback = fallback.or(Some(p)),
                    Match::No => {}
                }
            }
        }
        fallback
    }

    /// If the referenced model already exists locally, return its GGUF path.
    pub fn resolve(&self, r: &ModelRef) -> Result<Option<PathBuf>> {
        Ok(match r {
            ModelRef::Path(p) => p.exists().then(|| p.clone()),
            ModelRef::Repo { repo, sel } => self.resolve_repo(repo, sel.as_deref()),
        })
    }
}

/// How well a cached `.gguf` filename matches a selector.
enum Match {
    /// An explicit filename matched exactly, or the quant is the file's suffix (`…-Q4_K_M.gguf`).
    Exact,
    /// The quant appears somewhere in the name (weaker; e.g. an oddly-named or split file).
    Loose,
    No,
}

/// Pick the best `.gguf` from `names` for selector `sel` (quant or filename; `None` → default quant).
/// Exact match wins; else a loose (substring) match; else — only for the *default* quant (no explicit
/// selector) — the first `.gguf` (matches llama.cpp's "fall back to the first file" behavior).
pub(crate) fn pick_gguf(names: &[String], sel: Option<&str>) -> Option<String> {
    let want = sel.unwrap_or(DEFAULT_QUANT);
    let mut loose: Option<&String> = None;
    let mut first: Option<&String> = None;
    for n in names {
        if !n.to_lowercase().ends_with(".gguf") {
            continue;
        }
        first = first.or(Some(n));
        match gguf_match(n, want) {
            Match::Exact => return Some(n.clone()),
            Match::Loose => loose = loose.or(Some(n)),
            Match::No => {}
        }
    }
    loose.or(if sel.is_none() { first } else { None }).cloned()
}

/// Match a cached `.gguf` filename against a selector (an explicit `*.gguf` name, or a quant).
///
/// The quant must sit on **token boundaries** in the filename, else neighbouring formats collide:
/// `…-PQ2_0.gguf` / `…-TQ2_0.gguf` / `…-Q2_0_g64.gguf` are all DIFFERENT weight layouts from `Q2_0`
/// and must never satisfy a `Q2_0` selector. A token starts after `-`/`_`/`.` (or the name start) and
/// ends before `-`/`.` (or the stem end) — `_` does not end it, since quant names embed it (`Q4_K_M`).
fn gguf_match(fname: &str, sel: &str) -> Match {
    if sel.to_lowercase().ends_with(".gguf") {
        return if fname.eq_ignore_ascii_case(sel) {
            Match::Exact
        } else {
            Match::No
        };
    }
    let (f, q) = (fname.to_lowercase(), sel.to_lowercase());
    let Some(stem) = f.strip_suffix(".gguf") else {
        return Match::No;
    };
    let starts_token = |i: usize| i == 0 || matches!(stem.as_bytes()[i - 1], b'-' | b'_' | b'.');
    let ends_token = |i: usize| i == stem.len() || matches!(stem.as_bytes()[i], b'-' | b'.');

    let mut loose = false;
    for (i, _) in stem.match_indices(q.as_str()) {
        let end = i + q.len();
        if !starts_token(i) || !ends_token(end) {
            continue;
        }
        if end == stem.len() {
            return Match::Exact; // the quant IS the trailing token: `…-Q4_K_M.gguf`
        }
        loose = true; // a delimited hit elsewhere: split shards, `…-Q4_K_M-00001-of-00003.gguf`
    }
    if loose {
        Match::Loose
    } else {
        Match::No
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    /// Build a fake HF Hub repo dir: blobs/<sha> + snapshots/<commit>/<file> -> blob + refs/main.
    fn fake_hf(hub: &std::path::Path, repo: &str, commit: &str, file: &str, sha: &str) {
        let dir = hub.join(format!("models--{}", repo.replace('/', "--")));
        let blobs = dir.join("blobs");
        let snap = dir.join("snapshots").join(commit);
        fs::create_dir_all(&blobs).unwrap();
        fs::create_dir_all(&snap).unwrap();
        fs::write(blobs.join(sha), b"fake gguf bytes").unwrap();
        symlink(format!("../../blobs/{sha}"), snap.join(file)).unwrap();
        fs::create_dir_all(dir.join("refs")).unwrap();
        fs::write(dir.join("refs").join("main"), commit).unwrap();
    }

    fn store_at(hub: PathBuf) -> Store {
        Store { hub }
    }

    #[test]
    fn resolve_hf_default_quant() {
        let tmp = tempfile::tempdir().unwrap();
        fake_hf(
            tmp.path(),
            "unsloth/Qwen3-14B-GGUF",
            "abc123",
            "Qwen3-14B-Q4_K_M.gguf",
            "deadbeef",
        );
        let store = store_at(tmp.path().to_path_buf());
        let got = store.resolve_repo("unsloth/Qwen3-14B-GGUF", None).unwrap();
        assert!(got.ends_with("Qwen3-14B-Q4_K_M.gguf"));
    }

    #[test]
    fn resolve_hf_quant_selector() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path();
        fake_hf(hub, "u/r", "c", "model-Q4_K_M.gguf", "aa");
        fake_hf(hub, "u/r", "c", "model-Q8_0.gguf", "bb");
        let store = store_at(hub.to_path_buf());
        assert!(store
            .resolve_repo("u/r", Some("Q8_0"))
            .unwrap()
            .ends_with("model-Q8_0.gguf"));
        assert!(store
            .resolve_repo("u/r", Some("q4_k_m")) // case-insensitive
            .unwrap()
            .ends_with("model-Q4_K_M.gguf"));
    }

    /// prism-ml/Ternary-Bonsai-*-gguf ships Q2_0 next to PQ2_0 and Q2_0_g64 — all different layouts.
    /// A `Q2_0` selector must land on Q2_0 regardless of listing order, never on its neighbours.
    #[test]
    fn pick_gguf_quant_neighbours_never_collide() {
        let names: Vec<String> = [
            "Ternary-Bonsai-1.7B-F16.gguf",
            "Ternary-Bonsai-1.7B-PQ2_0.gguf",
            "Ternary-Bonsai-1.7B-Q2_0.gguf",
            "Ternary-Bonsai-1.7B-Q2_0_g64.gguf",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(
            pick_gguf(&names, Some("Q2_0")).as_deref(),
            Some("Ternary-Bonsai-1.7B-Q2_0.gguf")
        );
        assert_eq!(
            pick_gguf(&names, Some("PQ2_0")).as_deref(),
            Some("Ternary-Bonsai-1.7B-PQ2_0.gguf")
        );
        assert_eq!(
            pick_gguf(&names, Some("Q2_0_g64")).as_deref(),
            Some("Ternary-Bonsai-1.7B-Q2_0_g64.gguf")
        );
        // No Q2_0 in the repo at all → a PQ2_0/TQ2_0 sibling must NOT be served as a fallback.
        let only_p = vec!["Ternary-Bonsai-4B-TQ2_0.gguf".to_string()];
        assert_eq!(pick_gguf(&only_p, Some("Q2_0")), None);
    }

    #[test]
    fn pick_gguf_split_shards_are_loose() {
        let names: Vec<String> = ["m-Q4_K_M-00001-of-00002.gguf", "m-Q8_0.gguf"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            pick_gguf(&names, Some("Q4_K_M")).as_deref(),
            Some("m-Q4_K_M-00001-of-00002.gguf")
        );
        // A quant that is a strict prefix of another must not match it.
        assert_eq!(pick_gguf(&names, Some("Q4_K")), None);
    }

    #[test]
    fn resolve_hf_explicit_filename() {
        let tmp = tempfile::tempdir().unwrap();
        fake_hf(tmp.path(), "u/r", "c", "weird-name.gguf", "aa");
        let store = store_at(tmp.path().to_path_buf());
        assert!(store
            .resolve_repo("u/r", Some("weird-name.gguf"))
            .unwrap()
            .ends_with("weird-name.gguf"));
        assert_eq!(store.resolve_repo("u/r", Some("Q4_K_M")), None);
    }

    #[test]
    fn resolve_hf_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_at(tmp.path().to_path_buf());
        assert_eq!(store.resolve_repo("nope/missing", None), None);
    }

    #[test]
    fn resolve_path_variants() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf = tmp.path().join("model.gguf");
        fs::write(&gguf, b"x").unwrap();
        let store = store_at(tmp.path().to_path_buf());
        assert_eq!(
            store.resolve(&ModelRef::Path(gguf.clone())).unwrap(),
            Some(gguf)
        );
        assert_eq!(
            store
                .resolve(&ModelRef::Path(tmp.path().join("nope.gguf")))
                .unwrap(),
            None
        );
    }
}
