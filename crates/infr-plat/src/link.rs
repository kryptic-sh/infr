//! Pointing one path at another without copying its bytes.

use std::io;
use std::path::Path;

/// What [`link_blob`] managed to make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkKind {
    /// A symbolic link — what `huggingface_hub` and llama.cpp write, and the only kind that keeps
    /// the store byte-identical to theirs.
    Symlink,
    /// A hard link to the target: the filesystem refused symlinks (FAT/exFAT, many CIFS mounts,
    /// Windows without `SeCreateSymbolicLinkPrivilege`). Same bytes, no extra disk.
    HardLink,
    /// A full copy of the target: the filesystem refused both kinds of link. Correct, at the cost
    /// of the target's size again on disk.
    Copy,
}

/// Create `link` pointing at `target`, for a content-addressed store, and say which kind it is.
///
/// `target` is a RELATIVE path (`../../blobs/<hex>`), deliberately: it keeps the store movable
/// and byte-identical to what `huggingface_hub` and llama.cpp write. It is interpreted relative
/// to `link`'s own directory, which is what a symlink target means and what the fallbacks below
/// reconstruct by hand.
///
/// Fails if `link` already exists. Idempotency belongs to the caller, which knows whether an
/// existing entry is a resumable partial or a finished blob — this function must not guess and
/// unlink.
///
/// Tries, in order, the same chain `huggingface_hub` falls back through: a symlink, then a hard
/// link, then a copy ([`LinkKind`]). A filesystem without symlinks is otherwise a store that
/// downloads every blob and then fails to name it, on every run. The fallbacks are a real
/// difference in kind, not a spelling: a hard link or copy does not follow the blob if the blob
/// is later replaced, which is harmless here only because blobs are content-addressed and never
/// rewritten in place.
///
/// **Unix**: `symlink`. **Windows**: `symlink_file`, which needs `SeCreateSymbolicLinkPrivilege`
/// (Developer Mode or an elevated account), so the hard-link arm is the common case there.
/// **Other targets**: no symlink primitive, so the chain starts at the hard link.
pub fn link_blob(target: impl AsRef<Path>, link: &Path) -> io::Result<LinkKind> {
    link_with(target.as_ref(), link, symlink, |src, dst| {
        std::fs::hard_link(src, dst)
    })
}

/// Are `a` and `b` the same file on disk — one inode reached by two names, as a hard link is?
/// Compares identity (device + inode on unix, volume serial + file index on Windows), never
/// contents: two copies of the same bytes are NOT the same file. Follows symlinks on both sides.
pub fn is_same_file(a: &Path, b: &Path) -> io::Result<bool> {
    crate::lock::same_file(&std::fs::File::open(a)?, b)
}

/// [`link_blob`]'s chain with the two link primitives passed in, so a test can make either refuse.
fn link_with(
    target: &Path,
    link: &Path,
    symlink: impl FnOnce(&Path, &Path) -> io::Result<()>,
    hard_link: impl FnOnce(&Path, &Path) -> io::Result<()>,
) -> io::Result<LinkKind> {
    // The caller's target is `../../blobs/<hex>` with FORWARD slashes, which is what
    // `huggingface_hub` and llama.cpp write and what keeps the store portable. Windows accepts
    // those in an ordinary path, but NOT inside a symlink's reparse point: the link is created
    // successfully and then every open of it fails with ERROR_INVALID_NAME (123). Re-joining
    // through `components()` yields the same path with the platform's own separator.
    let native = target.components().collect::<std::path::PathBuf>();
    let symlink_err = match symlink(&native, link) {
        Ok(()) => return Ok(LinkKind::Symlink),
        // Something is already there: that is the caller's call, not a reason to fall back.
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => return Err(e),
        Err(e) => e,
    };
    let source = link.parent().unwrap_or_else(|| Path::new("")).join(&native);
    let hard_link_err = match hard_link(&source, link) {
        Ok(()) => return Ok(LinkKind::HardLink),
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => return Err(e),
        Err(e) => e,
    };
    match std::fs::copy(&source, link) {
        Ok(_) => Ok(LinkKind::Copy),
        Err(copy_err) => {
            // A failed copy can leave a partial file behind; it is not a link to anything.
            let _ = std::fs::remove_file(link);
            Err(io::Error::new(
                copy_err.kind(),
                format!(
                    "linking {link:?} -> {native:?}: symlink failed: {symlink_err}; hard link \
                     from {source:?} failed: {hard_link_err}; copy failed: {copy_err}"
                ),
            ))
        }
    }
}

#[cfg(unix)]
fn symlink(target: &Path, link: &Path) -> io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(windows)]
fn symlink(target: &Path, link: &Path) -> io::Result<()> {
    std::os::windows::fs::symlink_file(target, link)
}

#[cfg(not(any(unix, windows)))]
fn symlink(target: &Path, link: &Path) -> io::Result<()> {
    let _ = (target, link);
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "no symlink primitive on this platform",
    ))
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

    /// A filesystem that refuses symlinks gets a hard link, and one that refuses both gets a copy
    /// — each still yielding the blob's bytes through the link's own name, with the relative
    /// target resolved from the link's directory.
    #[test]
    fn a_refused_symlink_falls_back_to_a_hard_link_then_a_copy() {
        let refuse = |_: &Path, _: &Path| -> io::Result<()> {
            Err(io::Error::new(io::ErrorKind::Unsupported, "injected"))
        };
        let dir = tempfile::tempdir().expect("tempdir");
        fs::create_dir(dir.path().join("blobs")).expect("blobs/");
        fs::create_dir_all(dir.path().join("snapshots/abc")).expect("snapshots/");
        fs::write(dir.path().join("blobs/deadbeef"), b"payload").expect("blob");
        let target = Path::new("../../blobs/deadbeef");

        let hard = dir.path().join("snapshots/abc/hard.gguf");
        let kind = link_with(target, &hard, refuse, |s, d| fs::hard_link(s, d)).expect("hard");
        assert_eq!(kind, LinkKind::HardLink);
        assert_eq!(fs::read(&hard).expect("read hard link"), b"payload");
        assert!(!fs::symlink_metadata(&hard)
            .unwrap()
            .file_type()
            .is_symlink());
        let blob = dir.path().join("blobs/deadbeef");
        assert!(
            is_same_file(&hard, &blob).unwrap(),
            "a hard link is the blob itself"
        );

        let copy = dir.path().join("snapshots/abc/copy.gguf");
        assert_eq!(
            link_with(target, &copy, refuse, refuse).expect("copy"),
            LinkKind::Copy
        );
        assert_eq!(fs::read(&copy).expect("read copy"), b"payload");
        assert!(
            !is_same_file(&copy, &blob).unwrap(),
            "a copy is a different file"
        );

        // All three refusing is an error naming each, and leaves nothing at the link's name.
        let none = dir.path().join("snapshots/abc/none.gguf");
        let err = link_with(Path::new("../../blobs/missing"), &none, refuse, refuse)
            .expect_err("nothing can be linked to a missing blob");
        let msg = err.to_string();
        assert!(
            msg.contains("symlink failed") && msg.contains("copy failed"),
            "{msg}"
        );
        assert!(fs::symlink_metadata(&none).is_err());
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
