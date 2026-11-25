//! Detects a weight file changing underneath a live mapping.
//!
//! [`Gguf::open`](crate::Gguf::open) maps the model and hands out `&[u8]` slices into it for the
//! mapping's whole life, on an invariant it cannot enforce: nothing may write to or truncate that
//! file while the mapping lives. An in-place write mutates memory Rust believes is frozen; a
//! truncation turns a resident page into `SIGBUS` on next touch. Neither is preventable from
//! inside this process — an advisory lock binds only writers that ask for one, and the realistic
//! writer (`cp new.gguf live.gguf`, which opens the destination `O_TRUNC`) asks for nothing.
//!
//! So this does not prevent. It NOTICES, and turns a corrupted mapping from silent wrong output
//! into a named error. That is the whole claim — see `docs/backlog.md` B30 for why the preventing
//! options were each rejected.
//!
//! **It watches an inode, not a name.** [`WeightWatch::check`] re-`fstat`s a held descriptor
//! rather than re-`stat`ing the path, and the distinction is the design:
//!
//! * `infr pull` downloads to a temp and `rename`s into place. A rename swaps the directory entry
//!   while the old inode stays alive for anyone holding it — so the live mapping keeps reading the
//!   bytes it loaded, and is not corrupt. Statting the PATH would report that safe case as a
//!   change, which trains people to ignore the warning.
//! * An in-place write or truncate mutates the inode we are actually mapped onto. Statting the
//!   descriptor catches exactly that.
//!
//! **What it cannot see.** A same-length in-place write whose mtime is then restored (deliberate
//! `utimes`) is invisible here, and there is no cheap way to see it — the honest alternative is
//! hashing gigabytes on every check. Detection is best-effort by construction; it is worth having
//! because the accidental cases, which are the ones that actually happen, all move one of the two
//! fields.

use infr_core::blockio::FileStamp;
use infr_core::error::{Error, Result};
use std::{
    fs::File,
    path::{Path, PathBuf},
};

/// One watched file: its held descriptor, the path it was opened by, and its stamp at load.
struct Watched {
    /// Held open for the watch's whole life: this descriptor IS the identity being tracked, and
    /// `fstat` on it follows the inode even after the path stops naming it.
    file: File,
    path: PathBuf,
    at_load: FileStamp,
}

/// A model's weight files' identity as of load, re-checkable at any later point.
///
/// Hold one alongside a loaded model and [`check`](Self::check) it at whatever boundary suits the
/// command — per generated turn, per served request, once before reporting a benchmark. The check
/// is one `fstat` per file, so the boundary can be as tight as makes sense.
pub struct WeightWatch {
    files: Vec<Watched>,
}

impl WeightWatch {
    /// Stamp the single file `path` as it is now.
    ///
    /// Opens the path a second time, beside whatever mapping it guards, so a rename landing
    /// between the two opens leaves this watching the new inode while the mapping holds the old —
    /// a missed detection, never a false one. [`Self::for_model`] has no such window.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let at_load = FileStamp::of(&file)?;
        Ok(WeightWatch {
            files: vec![Watched {
                file,
                path: path.to_path_buf(),
                at_load,
            }],
        })
    }

    /// Watch the files `model` is mapped from — through its own descriptors, against the stamps it
    /// took when it mapped them ([`crate::Gguf::shard_handles`]). So this watches exactly the
    /// files the mapping reads, a `gguf-split` set whole, and a change that landed between the
    /// load and this call already shows on the first [`Self::check`].
    pub fn for_model(model: &crate::Gguf) -> Result<Self> {
        let files = model
            .shard_handles()
            .into_iter()
            .map(|h| {
                Ok(Watched {
                    file: h.file.try_clone()?,
                    path: h.path.to_path_buf(),
                    at_load: h.stamp,
                })
            })
            .collect::<Result<_>>()?;
        Ok(WeightWatch { files })
    }

    /// Stamp every file of the model `path` names: the whole `gguf-split` set when `path` is one
    /// shard of it, otherwise just `path`. The set comes from [`crate::Gguf::open`] itself, so it
    /// is exactly the files a mapping of that path reads, found the same way.
    ///
    /// That is a mapping of its OWN, not the one a model loaded elsewhere holds: a file replaced
    /// by rename between that load and this call leaves the two on different inodes. Where the
    /// model's [`crate::Gguf`] is at hand, [`Self::for_model`] watches its mapping instead.
    pub fn open_model(path: &Path) -> Result<Self> {
        Self::for_model(&crate::Gguf::open(path)?)
    }

    /// `Ok(())` while every watched file is byte-identical in the ways this can see; a described
    /// [`Error::Loader`] naming the first one that is not.
    ///
    /// The error names what moved and says the loaded weights are suspect, because by the time
    /// this fires the mapping may already be serving mutated bytes — the caller's job is to stop,
    /// not to carry on with a warning.
    pub fn check(&self) -> Result<()> {
        for w in &self.files {
            let now = FileStamp::of(&w.file)?;
            if now == w.at_load {
                continue;
            }
            let what = w.at_load.describe_change(&now);
            return Err(Error::Loader(format!(
                "weight file {:?} changed while mapped: {what}. The loaded weights are no longer \
                 trustworthy — any output produced since may be wrong, and a shrunk file can fault \
                 the process on next access. Restart against the current file.",
                w.path
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::SystemTime;

    /// A file written once and left alone must never trip the check, however often it is asked.
    #[test]
    fn an_untouched_file_stays_clean() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.gguf");
        std::fs::write(&path, vec![7u8; 4096]).unwrap();

        let watch = WeightWatch::open(&path).unwrap();
        for _ in 0..3 {
            assert!(watch.check().is_ok());
        }
    }

    /// Truncation is the case that ends in `SIGBUS`, and it moves the length.
    #[test]
    fn truncation_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.gguf");
        std::fs::write(&path, vec![7u8; 4096]).unwrap();

        let watch = WeightWatch::open(&path).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(512)
            .unwrap();

        let err = watch.check().expect_err("a shrunk file must be reported");
        let msg = err.to_string();
        assert!(msg.contains("4096 -> 512"), "{msg}");
    }

    /// A same-length rewrite moves only the mtime, so that half is load-bearing on its own. The
    /// timestamp is set explicitly rather than relied upon: a test writes fast enough to land in
    /// the same mtime tick on a coarse filesystem, which would make this pass for the wrong reason.
    #[test]
    fn a_same_length_rewrite_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.gguf");
        std::fs::write(&path, vec![7u8; 4096]).unwrap();

        let watch = WeightWatch::open(&path).unwrap();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        (&f).write_all(&vec![9u8; 4096]).unwrap();
        let bumped = SystemTime::now() + std::time::Duration::from_secs(60);
        f.set_times(std::fs::FileTimes::new().set_modified(bumped))
            .unwrap();
        drop(f);

        let err = watch
            .check()
            .expect_err("an in-place rewrite must be reported");
        assert!(
            err.to_string()
                .contains("modification time moved, size unchanged"),
            "{err}"
        );
    }

    /// The design decision, pinned: a rename INTO place is not a corruption. The old inode stays
    /// alive for whoever holds it, so a live mapping keeps reading exactly the bytes it loaded.
    /// Reporting this would be a false positive, and false positives are how a real warning gets
    /// ignored. This is also why `check` stats the descriptor and not the path.
    #[test]
    fn a_rename_into_place_is_not_a_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.gguf");
        std::fs::write(&path, vec![7u8; 4096]).unwrap();

        let watch = WeightWatch::open(&path).unwrap();

        let replacement = dir.path().join("new.gguf");
        std::fs::write(&replacement, vec![9u8; 8192]).unwrap();
        std::fs::rename(&replacement, &path).unwrap();

        assert!(
            watch.check().is_ok(),
            "a rename leaves the mapped inode intact; reporting it would be a false alarm"
        );
    }
}
