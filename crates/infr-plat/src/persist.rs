//! Making a finished file survive a crash under its final name.

use std::fs::OpenOptions;
use std::io;
use std::path::Path;

/// Move the fully written file `tmp` to `dest` (replacing any file there) so that, once this
/// returns, a crash or power loss leaves `dest` holding the complete bytes of `tmp` — never a
/// name that points at blocks the disk had not written yet.
///
/// Two things have to reach the disk, in order: the file's data, then the directory entry that
/// renames it. A plain `rename` promises neither, so a crash shortly after it can leave `dest`
/// present at its full length with unwritten (zero) blocks inside — a file that looks committed
/// and is not.
///
/// **Unix**: `fsync` the file, `rename`, then `fsync` the directory that holds `dest`, which is
/// what makes the new entry itself durable. **Windows**: `FlushFileBuffers` on the file, then
/// `MoveFileExW` with `MOVEFILE_WRITE_THROUGH`, which does not return until the move is flushed —
/// Windows cannot open a directory handle to flush the way unix does. **Other targets**: an
/// `Unsupported` error rather than an unsynced rename reported as durable.
///
/// Verified on Linux only; the Windows arm is compiled by CI's native clippy and test jobs.
pub fn commit_file(tmp: &Path, dest: &Path) -> io::Result<()> {
    // Write access, because Windows refuses `FlushFileBuffers` on a read-only handle.
    OpenOptions::new().write(true).open(tmp)?.sync_all()?;
    rename_durably(tmp, dest)
}

#[cfg(unix)]
fn rename_durably(tmp: &Path, dest: &Path) -> io::Result<()> {
    std::fs::rename(tmp, dest)?;
    let dir = match dest.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    match std::fs::File::open(dir)?.sync_all() {
        Err(e) if dir_sync_unsupported(&e) => Ok(()),
        other => other,
    }
}

/// Is this a directory `fsync` the filesystem cannot do at all, rather than one that failed?
///
/// Some FUSE filesystems, vboxsf and network mounts answer a directory `fsync` with `EINVAL`,
/// `ENOTSUP`/`EOPNOTSUPP` or `ENOSYS`. By then the rename has already happened, so failing the
/// commit would report a committed file as lost, on every call, on that filesystem. Such an answer
/// is taken as success — the same choice SQLite makes — with the durability the filesystem offers.
/// Any other error (`EIO` above all) still fails. The FILE's own sync is never excused this way.
#[cfg(unix)]
fn dir_sync_unsupported(e: &io::Error) -> bool {
    matches!(
        e.raw_os_error(),
        Some(code) if code == libc::EINVAL
            || code == libc::ENOTSUP
            || code == libc::EOPNOTSUPP
            || code == libc::ENOSYS
    )
}

#[cfg(windows)]
fn rename_durably(tmp: &Path, dest: &Path) -> io::Result<()> {
    use windows::core::HSTRING;
    use windows::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let (from, to) = (HSTRING::from(tmp), HSTRING::from(dest));
    // SAFETY: both arguments are NUL-terminated wide strings owned by this frame for the call.
    unsafe {
        MoveFileExW(
            &from,
            &to,
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    }
    .map_err(|_| io::Error::last_os_error())
}

#[cfg(not(any(unix, windows)))]
fn rename_durably(tmp: &Path, dest: &Path) -> io::Result<()> {
    let _ = (tmp, dest);
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "no durable rename on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// The bytes arrive under the final name, the temp name is gone, and an existing file there is
    /// replaced rather than refused. (Durability against power loss is not observable from a test;
    /// this pins the visible half of the contract.)
    #[test]
    fn a_commit_moves_the_bytes_and_replaces_the_old_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (tmp, dest) = (dir.path().join(".partial"), dir.path().join("blob"));
        fs::write(&dest, b"stale").expect("old file");
        fs::write(&tmp, b"fresh bytes").expect("temp");

        commit_file(&tmp, &dest).expect("commit");

        assert_eq!(fs::read(&dest).expect("read dest"), b"fresh bytes");
        assert!(!tmp.exists(), "the temp name must be gone");
    }

    /// The directory-sync answers that mean "this filesystem cannot", and only those, are excused.
    #[cfg(unix)]
    #[test]
    fn only_unsupported_directory_syncs_are_excused() {
        for code in [libc::EINVAL, libc::ENOTSUP, libc::EOPNOTSUPP, libc::ENOSYS] {
            assert!(
                dir_sync_unsupported(&io::Error::from_raw_os_error(code)),
                "{code}"
            );
        }
        for code in [libc::EIO, libc::ENOSPC, libc::EACCES, libc::EBADF] {
            assert!(
                !dir_sync_unsupported(&io::Error::from_raw_os_error(code)),
                "{code}"
            );
        }
        assert!(!dir_sync_unsupported(&io::Error::other("no os code")));
    }

    /// A missing source is an error, not a silent no-op that leaves the caller believing a blob
    /// was committed.
    #[test]
    fn committing_a_missing_file_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = commit_file(&dir.path().join("nope"), &dir.path().join("blob"))
            .expect_err("nothing to commit");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(!dir.path().join("blob").exists());
    }
}
