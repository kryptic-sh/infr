//! Pointing one path at another without copying its bytes.

use std::io;
use std::path::Path;

/// Create `link` pointing at `target`, for a content-addressed store.
///
/// `target` is a RELATIVE path (`../../blobs/<hex>`), deliberately: it keeps the store movable
/// and byte-identical to what `huggingface_hub` and llama.cpp write. It is interpreted relative
/// to `link`'s own directory, which is what a symlink target means and what the Windows fallback
/// below reconstructs by hand.
///
/// Fails if `link` already exists. Idempotency belongs to the caller, which knows whether an
/// existing entry is a resumable partial or a finished blob — this function must not guess and
/// unlink.
///
/// **Unix**: a symlink. **Windows**: a symlink too, falling back to a hard link when
/// `symlink_file` is refused — creating one there needs `SeCreateSymbolicLinkPrivilege`, which an
/// ordinary account does not have outside Developer Mode. The fallback is a real difference in
/// kind, not a spelling: a hard link shares an inode, so it survives the original being renamed
/// and does not track it, and the store's disk usage is the same either way only because both
/// point at bytes that already exist.
pub fn link_blob(target: impl AsRef<Path>, link: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(windows)]
    {
        let target = target.as_ref();
        match std::os::windows::fs::symlink_file(target, link) {
            Ok(()) => Ok(()),
            Err(symlink_err) => {
                let source = link.parent().unwrap_or_else(|| Path::new("")).join(target);
                std::fs::hard_link(&source, link).map_err(|hard_link_err| {
                    io::Error::new(
                        hard_link_err.kind(),
                        format!(
                            "symlink_file {link:?} -> {target:?} failed: {symlink_err}; hard_link fallback from {source:?} failed: {hard_link_err}"
                        ),
                    )
                })
            }
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (target, link);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "no link primitive on this platform",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Whatever the platform makes, reading through the link must yield the target's bytes, and
    /// the target must be resolved relative to the LINK's directory rather than the process's cwd
    /// — the store is entered from anywhere.
    #[test]
    fn a_relative_link_resolves_against_the_links_own_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::create_dir(dir.path().join("blobs")).expect("blobs/");
        fs::create_dir_all(dir.path().join("snapshots/abc")).expect("snapshots/");
        fs::write(dir.path().join("blobs/deadbeef"), b"payload").expect("blob");

        let link = dir.path().join("snapshots/abc/model.gguf");
        link_blob("../../blobs/deadbeef", &link).expect("link");

        assert_eq!(fs::read(&link).expect("read through link"), b"payload");
    }

    /// An existing link is an error, not a silent overwrite: the caller unlinks first because only
    /// it can tell a finished blob from a partial.
    #[test]
    fn linking_over_an_existing_path_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("target"), b"x").expect("target");
        let link = dir.path().join("link");
        fs::write(&link, b"in the way").expect("occupant");

        link_blob("target", &link).expect_err("must not overwrite");
        assert_eq!(fs::read(&link).expect("occupant survives"), b"in the way");
    }
}
