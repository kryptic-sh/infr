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
