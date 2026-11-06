//! Listing and removing models in the HF Hub cache (`infr ls` / `infr rm`).

use crate::store::{pick_gguf, shard_set, snapshot_ggufs, Store};
use infr_core::error::{Error, Result};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::debug;

/// One cached GGUF (a whole shard set counts as one).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedModel {
    pub repo: String,
    /// Snapshot-relative name of the first shard (`UD-Q4_K_XL/m.gguf`).
    pub file: String,
    pub shards: usize,
    pub bytes: u64,
    pub path: PathBuf,
}

/// What [`Store::remove`] deleted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Removed {
    pub files: usize,
    pub bytes: u64,
}

impl Store {
    /// Every repo directory in the hub cache as `(org/repo, dir)`, sorted by name.
    fn repos(&self) -> Vec<(String, PathBuf)> {
        let mut out: Vec<_> = fs::read_dir(&self.hub)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let rest = name.strip_prefix("models--")?;
                let (org, repo) = rest.split_once("--")?;
                Some((format!("{org}/{repo}"), e.path()))
            })
            .collect();
        out.sort();
        out
    }

    /// Every cached GGUF whose files are all present, sorted by repo then file. Older snapshots a
    /// repo still carries are listed too, but a file name is reported once per repo.
    pub fn list(&self) -> Vec<CachedModel> {
        let mut out = Vec::new();
        for (repo, _) in self.repos() {
            let mut seen = HashSet::new();
            for snap in self.ordered_snapshots(&repo) {
                let mut names = snapshot_ggufs(&snap);
                names.sort();
                for name in names {
                    let set = shard_set(&name);
                    if set[0] != name || !seen.insert(name.clone()) {
                        continue;
                    }
                    let sizes: Option<Vec<u64>> = set
                        .iter()
                        .map(|f| fs::metadata(snap.join(f)).ok().map(|m| m.len()))
                        .collect();
                    let Some(sizes) = sizes else { continue };
                    out.push(CachedModel {
                        repo: repo.clone(),
                        path: snap.join(&name),
                        file: name,
                        shards: set.len(),
                        bytes: sizes.iter().sum(),
                    });
                }
            }
        }
        out
    }

    /// A copyable `org/repo:quant` reference, or an explicit filename when the quant is ambiguous.
    pub fn model_tag(&self, model: &CachedModel) -> String {
        let sel = quant_selector(&model.file)
            .filter(|sel| self.resolve_repo(&model.repo, Some(sel)).as_ref() == Some(&model.path))
            .unwrap_or_else(|| model.file.clone());
        format!("{}:{sel}", model.repo)
    }

    /// Remove cached models. `sel = None` deletes the whole repo directory; otherwise only the GGUF
    /// `sel` picks (a quant or filename) is unlinked from every snapshot and blobs nothing else
    /// references are deleted. Errors when nothing matches.
    pub fn remove(&self, repo: &str, sel: Option<&str>) -> Result<Removed> {
        let dir = self.repo_dir(repo);
        if !dir.is_dir() {
            return Err(Error::Other(format!("{repo} is not cached")));
        }
        let Some(sel) = sel else {
            let bytes = dir_size(&dir.join("blobs"));
            let files = self.list().iter().filter(|m| m.repo == repo).count();
            fs::remove_dir_all(&dir).map_err(Error::from)?;
            return Ok(Removed { files, bytes });
        };

        let mut doomed: Vec<PathBuf> = Vec::new();
        let mut matched = 0;
        for snap in self.ordered_snapshots(repo) {
            let Some(chosen) = pick_gguf(&snapshot_ggufs(&snap), Some(sel)) else {
                continue;
            };
            matched += 1;
            for f in shard_set(&chosen) {
                doomed.push(snap.join(f));
            }
        }
        if matched == 0 {
            return Err(Error::Other(format!(
                "no cached .gguf matching '{sel}' in {repo}"
            )));
        }

        let blobs_dir = dir.join("blobs");
        let mut blobs: HashSet<PathBuf> = HashSet::new();
        for link in &doomed {
            if let Some(b) = blob_of(link, &blobs_dir) {
                blobs.insert(b);
            }
            fs::remove_file(link).map_err(Error::from)?;
        }
        prune_empty_dirs(&dir.join("snapshots"));

        // Keep any blob a surviving snapshot link (other quant, companion file) still points at.
        let live = live_blobs(&dir.join("snapshots"), &blobs_dir);
        let mut out = Removed {
            files: matched,
            bytes: 0,
        };
        for b in blobs.difference(&live) {
            let len = fs::metadata(b).map(|m| m.len()).unwrap_or(0);
            match fs::remove_file(b) {
                Ok(()) => out.bytes += len,
                Err(e) => debug!("could not remove blob {b:?}: {e}"),
            }
        }
        if self.list().iter().all(|m| m.repo != repo) && live.is_empty() {
            let _ = fs::remove_dir_all(&dir);
        }
        Ok(out)
    }
}

fn quant_selector(file: &str) -> Option<String> {
    let name = file.rsplit('/').next()?;
    let upper = name.to_ascii_uppercase();
    if upper.starts_with("MMPROJ")
        || upper.starts_with("MTP")
        || file.split('/').any(|part| part.eq_ignore_ascii_case("mtp"))
    {
        return None;
    }
    let stem = infr_core::gguf_split::parse_shard(name)
        .map(|shard| shard.base.to_ascii_uppercase())
        .or_else(|| upper.strip_suffix(".GGUF").map(str::to_owned))?;
    for (i, _) in stem.char_indices().rev() {
        if i != 0 && !matches!(stem.as_bytes()[i - 1], b'-' | b'_' | b'.') {
            continue;
        }
        let token = stem[i..].split(['-', '.']).next()?;
        let is_quant = ["IQ", "TQ", "PQ", "Q"].iter().any(|prefix| {
            token.strip_prefix(prefix).is_some_and(|rest| {
                let (bits, variant) = rest.split_once('_').unwrap_or(("", ""));
                !bits.is_empty()
                    && bits.bytes().all(|b| b.is_ascii_digit())
                    && !variant.is_empty()
                    && variant
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_')
            })
        });
        if !is_quant && !matches!(token, "F16" | "F32" | "BF16" | "MXFP4" | "NVFP4") {
            continue;
        }
        let start = if stem[..i].ends_with("UD-")
            && (i == 3 || matches!(stem.as_bytes()[i - 4], b'-' | b'_' | b'.'))
        {
            i - 3
        } else {
            i
        };
        return Some(stem[start..i + token.len()].to_owned());
    }
    None
}

/// The `blobs/<sha>` file a snapshot symlink resolves to.
fn blob_of(link: &Path, blobs_dir: &Path) -> Option<PathBuf> {
    let target = fs::read_link(link).ok()?;
    Some(blobs_dir.join(target.file_name()?))
}

/// Every blob some symlink under `snapshots` points at.
fn live_blobs(snapshots: &Path, blobs_dir: &Path) -> HashSet<PathBuf> {
    let mut out = HashSet::new();
    let mut stack = vec![snapshots.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).into_iter().flatten().flatten() {
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                stack.push(e.path());
            } else if let Some(b) = blob_of(&e.path(), blobs_dir) {
                out.insert(b);
            }
        }
    }
    out
}

/// Remove directories under `root` that no longer hold any files (the root itself stays).
fn prune_empty_dirs(root: &Path) {
    for e in fs::read_dir(root).into_iter().flatten().flatten() {
        let p = e.path();
        if e.file_type().is_ok_and(|t| t.is_dir()) {
            prune_empty_dirs(&p);
            let _ = fs::remove_dir(&p);
        }
    }
}

fn dir_size(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_plat::link::link_blob;

    /// A hub with `org/repo` holding `files` (name, bytes) in one snapshot.
    fn hub(files: &[(&str, usize)]) -> (tempfile::TempDir, Store) {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store {
            hub: tmp.path().to_path_buf(),
        };
        let dir = store.repo_dir("org/repo");
        let snap = dir.join("snapshots").join("c1");
        fs::create_dir_all(dir.join("blobs")).unwrap();
        fs::create_dir_all(dir.join("refs")).unwrap();
        fs::write(dir.join("refs").join("main"), "c1").unwrap();
        for (i, (name, len)) in files.iter().enumerate() {
            let hex = format!("{i:064x}");
            fs::write(dir.join("blobs").join(&hex), vec![0u8; *len]).unwrap();
            let link = snap.join(name);
            fs::create_dir_all(link.parent().unwrap()).unwrap();
            link_blob(crate::pull::blob_link_target(name, &hex), &link).unwrap();
        }
        (tmp, store)
    }

    #[test]
    fn list_reports_files_with_sizes_and_groups_shards() {
        let (_t, s) = hub(&[
            ("m-Q4_K_M.gguf", 10),
            ("big-00001-of-00002.gguf", 5),
            ("big-00002-of-00002.gguf", 7),
            ("README.md", 1),
        ]);
        let got: Vec<_> = s
            .list()
            .into_iter()
            .map(|m| (m.file, m.shards, m.bytes))
            .collect();
        assert_eq!(
            got,
            vec![
                ("big-00001-of-00002.gguf".to_string(), 2, 12),
                ("m-Q4_K_M.gguf".to_string(), 1, 10),
            ]
        );
    }

    #[test]
    fn quant_selectors_preserve_variants_and_strip_shards() {
        for (file, sel) in [
            ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "Q4_K_M"),
            ("gemma-4-E4B_q4_0-it.gguf", "Q4_0"),
            ("gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
            ("UD-Q4_K_XL/m-UD-Q4_K_XL-00001-of-00002.gguf", "UD-Q4_K_XL"),
            ("model-iq4_xs.GGUF", "IQ4_XS"),
            ("model-PQ2_0.gguf", "PQ2_0"),
            ("model-TQ2_0.gguf", "TQ2_0"),
            ("model-Q2_0_g64.gguf", "Q2_0_G64"),
            ("model-bf16.gguf", "BF16"),
        ] {
            assert_eq!(quant_selector(file).as_deref(), Some(sel), "{file}");
        }
        for file in ["model.gguf", "mmproj-model-F16.gguf", "MTP/model-Q8_0.gguf"] {
            assert_eq!(quant_selector(file), None, "{file}");
        }
    }

    #[test]
    fn model_tags_resolve_to_the_listed_files() {
        let (_t, s) = hub(&[
            ("gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf", 10),
            ("m-Q4_K_M.gguf", 20),
            ("m-IQ4_XS-00001-of-00002.gguf", 3),
            ("m-IQ4_XS-00002-of-00002.gguf", 4),
            ("gemma-4-E4B_q4_0-it.gguf", 5),
            ("model.gguf", 6),
            ("mmproj-model-F16.gguf", 7),
            ("MTP/model-Q8_0.gguf", 8),
        ]);
        let models = s.list();
        let tags: Vec<_> = models.iter().map(|m| s.model_tag(m)).collect();
        assert_eq!(
            tags,
            [
                "org/repo:MTP/model-Q8_0.gguf",
                "org/repo:UD-Q4_K_XL",
                "org/repo:Q4_0",
                "org/repo:IQ4_XS",
                "org/repo:Q4_K_M",
                "org/repo:mmproj-model-F16.gguf",
                "org/repo:model.gguf",
            ]
        );
        for (model, tag) in models.iter().zip(tags) {
            let reference = crate::ModelRef::parse(&tag).unwrap();
            assert_eq!(
                s.resolve(&reference).unwrap(),
                Some(model.path.clone()),
                "{tag}"
            );
        }
    }

    #[test]
    fn unsloth_model_tag_is_copyable() {
        let (_t, s) = hub(&[("gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf", 10)]);
        let repo = "unsloth/gemma-4-E4B-it-qat-GGUF";
        fs::rename(s.repo_dir("org/repo"), s.repo_dir(repo)).unwrap();
        let model = s.list().pop().unwrap();
        let tag = s.model_tag(&model);
        assert_eq!(tag, "unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL");
        assert_eq!(
            s.resolve(&crate::ModelRef::parse(&tag).unwrap()).unwrap(),
            Some(model.path)
        );
    }

    #[test]
    fn model_tags_use_filenames_when_a_quant_selects_another_file() {
        let (_t, s) = hub(&[
            ("a-Q4_K_M.gguf", 10),
            ("b-Q4_K_M.gguf", 20),
            ("m-UD-Q4_K_XL.gguf", 30),
            ("m-Q4_K_XL.gguf", 40),
        ]);
        for model in s.list() {
            let tag = s.model_tag(&model);
            assert_eq!(
                s.resolve(&crate::ModelRef::parse(&tag).unwrap()).unwrap(),
                Some(model.path)
            );
        }
    }

    #[test]
    fn model_tags_keep_directory_only_quants_and_duplicate_basenames_explicit() {
        let (_t, s) = hub(&[
            ("Q4_K_M/model.gguf", 10),
            ("Q8_0/model.gguf", 20),
            ("model.gguf", 30),
        ]);
        for model in s.list() {
            let tag = s.model_tag(&model);
            assert_eq!(tag, format!("org/repo:{}", model.file));
            assert_eq!(
                s.resolve(&crate::ModelRef::parse(&tag).unwrap()).unwrap(),
                Some(model.path)
            );
        }
    }

    #[test]
    fn model_tags_keep_projectors_explicit_alongside_same_quant_weights() {
        let (_t, s) = hub(&[("mmproj-F16.gguf", 10), ("model-F16.gguf", 20)]);
        for model in s.list() {
            let tag = s.model_tag(&model);
            if model.file.starts_with("mmproj") {
                assert_eq!(tag, "org/repo:mmproj-F16.gguf");
            }
            assert_eq!(
                s.resolve(&crate::ModelRef::parse(&tag).unwrap()).unwrap(),
                Some(model.path)
            );
        }
    }

    #[test]
    fn remove_one_quant_keeps_the_others() {
        let (_t, s) = hub(&[("m-Q4_K_M.gguf", 10), ("m-Q8_0.gguf", 20)]);
        let r = s.remove("org/repo", Some("Q8_0")).unwrap();
        assert_eq!(
            r,
            Removed {
                files: 1,
                bytes: 20
            }
        );
        let left: Vec<_> = s.list().into_iter().map(|m| m.file).collect();
        assert_eq!(left, vec!["m-Q4_K_M.gguf"]);
    }

    #[test]
    fn remove_last_quant_drops_the_repo() {
        let (_t, s) = hub(&[("m-Q4_K_M.gguf", 10)]);
        s.remove("org/repo", Some("Q4_K_M")).unwrap();
        assert!(!s.repo_dir("org/repo").exists());
    }

    #[test]
    fn remove_whole_repo() {
        let (_t, s) = hub(&[("a-Q4_K_M.gguf", 10), ("a-Q8_0.gguf", 20)]);
        let r = s.remove("org/repo", None).unwrap();
        assert_eq!(
            r,
            Removed {
                files: 2,
                bytes: 30
            }
        );
        assert!(s.list().is_empty());
    }

    #[test]
    fn remove_unknown_errors() {
        let (_t, s) = hub(&[("m-Q4_K_M.gguf", 10)]);
        assert!(s.remove("org/other", None).is_err());
        assert!(s.remove("org/repo", Some("Q2_K")).is_err());
        assert_eq!(s.list().len(), 1);
    }
}
