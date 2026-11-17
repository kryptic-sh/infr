//! An exclusive lock on a file, released when the guard drops or the process dies.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

/// An exclusive whole-file lock, released on drop **and** on process death.
///
/// Crash safety is the property the callers depend on: a process killed mid-download leaves no
/// stale lock behind, because both platforms release on last-handle-close and the kernel closes
/// every handle of a dead process.
///
/// # Strength: advisory
///
/// This type promises **advisory** exclusion — the weaker of what the two platforms give — and
/// callers must not rely on more.
///
/// - **Unix** (`flock`): advisory in the strict sense. A process that never calls [`acquire`] can
///   read and write the file freely; the lock only excludes other lockers.
/// - **Windows** (`LockFileEx`): mandatory for the locked range. Reads and writes through any
///   other handle are refused by the kernel whether or not that code asked for a lock.
///
/// Writing a caller against the mandatory behaviour would work on Windows and quietly do nothing
/// on the platform this is developed on, which is the wrong way round for a bug to surface.
///
/// [`acquire`]: FileLock::acquire
#[derive(Debug)]
pub struct FileLock {
    file: File,
}

impl FileLock {
    /// Take the lock on `path`, creating the file if needed, blocking until any other holder
    /// releases it.
    pub fn acquire(path: &Path) -> io::Result<Self> {
        let file = Self::open(path)?;
        lock(&file, Blocking::Wait)?;
        Ok(FileLock { file })
    }

    /// Like [`acquire`](Self::acquire), but the wait can be called off: `give_up` is polled while
    /// another holder keeps the lock, and once it answers `true` this returns an error of kind
    /// [`io::ErrorKind::Interrupted`] instead of waiting on.
    ///
    /// A blocking `acquire` cannot be reached by a shutdown request. On unix a signal interrupts
    /// `flock` only when it happens to be delivered to the waiting thread — any other thread takes
    /// it and the wait goes on — and a Windows console control event runs on a thread of its own,
    /// so `LockFileEx` is never interrupted at all. Polling a non-blocking attempt is the form that
    /// behaves the same on both, at the cost of noticing a release up to one poll interval late.
    pub fn acquire_unless(path: &Path, give_up: impl Fn() -> bool) -> io::Result<Self> {
        const POLL: std::time::Duration = std::time::Duration::from_millis(50);
        let file = Self::open(path)?;
        loop {
            match lock(&file, Blocking::Fail) {
                Ok(()) => return Ok(FileLock { file }),
                // Held elsewhere, or the attempt itself was interrupted by a signal: either way
                // nothing is held, so check whether to keep waiting and try again.
                Err(e) if would_block(&e) || e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
            if give_up() {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    format!("gave up waiting for the lock on {}", path.display()),
                ));
            }
            std::thread::sleep(POLL);
        }
    }

    /// Take the lock if it is free, or return `Ok(None)` immediately if someone else holds it.
    ///
    /// Distinguishes "held by another" (`Ok(None)`) from "could not be attempted" (`Err`) — a
    /// caller that collapsed the two would report a permissions failure as healthy contention.
    pub fn try_acquire(path: &Path) -> io::Result<Option<Self>> {
        Self::try_lock(Self::open(path)?)
    }

    /// [`try_acquire`](Self::try_acquire) for a file that must already exist: a missing one is an
    /// `Err` of kind `NotFound` rather than a new file. For a caller probing a file another
    /// process may delete — recreating it would leave an empty file the other process never made.
    pub fn try_acquire_existing(path: &Path) -> io::Result<Option<Self>> {
        let file = OpenOptions::new().write(true).open(path)?;
        Self::try_lock(file)
    }

    /// Does `path` still name the file this lock is held on? A file locked through one name can
    /// be unlinked and a new file created under that name, and acting on the name afterwards would
    /// act on the new file. An `Err` (the path is gone, or this platform cannot tell) is not a yes.
    pub fn is_at(&self, path: &Path) -> io::Result<bool> {
        same_file(&self.file, path)
    }

    fn try_lock(file: File) -> io::Result<Option<Self>> {
        match lock(&file, Blocking::Fail) {
            Ok(()) => Ok(Some(FileLock { file })),
            Err(e) if would_block(&e) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn open(path: &Path) -> io::Result<File> {
        OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // Closing the handle releases the lock; the explicit unlock is belt-and-braces, and its
        // failure has nowhere to go — the guard is being destroyed either way.
        let _ = unlock(&self.file);
    }
}

/// Whether a contended lock waits or gives up at once.
#[derive(Clone, Copy)]
enum Blocking {
    Wait,
    Fail,
}

/// Is this the "someone else holds it" error, as opposed to a real failure?
fn would_block(e: &io::Error) -> bool {
    // Unix reports EWOULDBLOCK/EAGAIN; Windows reports ERROR_LOCK_VIOLATION (33) as
    // `ErrorKind::Uncategorized`, so match on the raw code there rather than on the kind.
    e.kind() == io::ErrorKind::WouldBlock || e.raw_os_error() == Some(WOULD_BLOCK_OS_ERROR)
}

#[cfg(unix)]
const WOULD_BLOCK_OS_ERROR: i32 = libc::EWOULDBLOCK;
#[cfg(windows)]
const WOULD_BLOCK_OS_ERROR: i32 = 33; // ERROR_LOCK_VIOLATION
#[cfg(not(any(unix, windows)))]
const WOULD_BLOCK_OS_ERROR: i32 = 0;

/// Do `file` and `path` name the same file (device + inode on unix, volume serial + file index
/// on Windows)? Shared with [`crate::link::is_same_file`].
#[cfg(unix)]
pub(crate) fn same_file(file: &File, path: &Path) -> io::Result<bool> {
    use std::os::unix::fs::MetadataExt;

    let (held, named) = (file.metadata()?, std::fs::metadata(path)?);
    Ok(held.dev() == named.dev() && held.ino() == named.ino())
}

#[cfg(windows)]
pub(crate) fn same_file(file: &File, path: &Path) -> io::Result<bool> {
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };

    fn identity(file: &File) -> io::Result<(u32, u32, u32)> {
        let mut info = BY_HANDLE_FILE_INFORMATION::default();
        // SAFETY: the handle is valid for the call and `info` is a live out-parameter.
        unsafe { GetFileInformationByHandle(HANDLE(file.as_raw_handle()), &mut info) }
            .map_err(|_| io::Error::last_os_error())?;
        Ok((
            info.dwVolumeSerialNumber,
            info.nFileIndexHigh,
            info.nFileIndexLow,
        ))
    }
    // No access rights: only the handle's identity is read, and a locked range refuses reads.
    let named = OpenOptions::new().access_mode(0).open(path)?;
    Ok(identity(file)? == identity(&named)?)
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn same_file(_file: &File, _path: &Path) -> io::Result<bool> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "no file-identity probe on this platform",
    ))
}

#[cfg(unix)]
fn lock(file: &File, blocking: Blocking) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;

    let mut op = libc::LOCK_EX;
    if let Blocking::Fail = blocking {
        op |= libc::LOCK_NB;
    }
    // SAFETY: `file` owns a valid open fd for the duration of the call; `flock` takes no pointers
    // and mutates no memory.
    if unsafe { libc::flock(file.as_raw_fd(), op) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn unlock(file: &File) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;

    // SAFETY: as above.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// The whole file, as a 64-bit byte count split into the two 32-bit halves the API takes.
#[cfg(windows)]
const WHOLE_FILE: (u32, u32) = (u32::MAX, u32::MAX);

#[cfg(windows)]
fn lock(file: &File, blocking: Blocking) -> io::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Storage::FileSystem::{
        LockFileEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY,
    };
    use windows::Win32::System::IO::OVERLAPPED;

    let mut flags = LOCKFILE_EXCLUSIVE_LOCK;
    if let Blocking::Fail = blocking {
        flags |= LOCKFILE_FAIL_IMMEDIATELY;
    }
    let mut overlapped = OVERLAPPED::default();
    let (low, high) = WHOLE_FILE;
    // SAFETY: the handle is valid for the call, and `overlapped` is a zeroed OVERLAPPED that
    // outlives it — the synchronous form of the call, so nothing retains the pointer.
    unsafe {
        LockFileEx(
            HANDLE(file.as_raw_handle()),
            flags,
            0,
            low,
            high,
            &mut overlapped,
        )
    }
    .map_err(|_| io::Error::last_os_error())
}

#[cfg(windows)]
fn unlock(file: &File) -> io::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Storage::FileSystem::UnlockFileEx;
    use windows::Win32::System::IO::OVERLAPPED;

    let mut overlapped = OVERLAPPED::default();
    let (low, high) = WHOLE_FILE;
    // SAFETY: as above.
    unsafe { UnlockFileEx(HANDLE(file.as_raw_handle()), 0, low, high, &mut overlapped) }
        .map_err(|_| io::Error::last_os_error())
}

#[cfg(not(any(unix, windows)))]
fn lock(_file: &File, _blocking: Blocking) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "no file-locking primitive on this platform",
    ))
}

#[cfg(not(any(unix, windows)))]
fn unlock(_file: &File) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "no file-locking primitive on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the type, and portable by construction: both platforms lock per OPEN
    /// HANDLE, so a second attempt is refused even from the same process.
    ///
    /// This replaces a test that was `#[cfg(not(target_os = "windows"))]` and therefore left the
    /// `LockFileEx` path with no coverage on any platform.
    #[test]
    fn a_held_lock_excludes_a_second_holder() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("blob.lock");

        let held = FileLock::acquire(&path).expect("first acquire");
        assert!(
            FileLock::try_acquire(&path)
                .expect("contention is not an error")
                .is_none(),
            "a second lock must not be grantable while the first is held"
        );

        drop(held);
        assert!(
            FileLock::try_acquire(&path).expect("re-acquire").is_some(),
            "the lock must be grantable once the holder drops"
        );
    }

    /// The probe for an existing file never creates one, and otherwise answers like
    /// `try_acquire`.
    #[test]
    fn try_acquire_existing_does_not_create() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("marker");

        let err = FileLock::try_acquire_existing(&path).expect_err("no file to lock");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(!path.exists(), "the probe must not create the file");

        let held = FileLock::acquire(&path).expect("create and lock");
        assert!(FileLock::try_acquire_existing(&path)
            .expect("contention is not an error")
            .is_none());
        drop(held);
        assert!(FileLock::try_acquire_existing(&path)
            .expect("free once released")
            .is_some());
    }

    /// A lock knows which file it holds: its own path says yes, another file's says no, and so
    /// does a path that no longer exists.
    #[test]
    fn is_at_names_the_held_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (path, other) = (dir.path().join("marker"), dir.path().join("other"));
        std::fs::write(&other, b"").unwrap();
        let held = FileLock::acquire(&path).expect("create and lock");
        assert!(held.is_at(&path).expect("probe own path"));
        assert!(!held.is_at(&other).expect("probe another file"));
        assert!(held.is_at(&dir.path().join("missing")).is_err());
    }

    /// A wait that is called off returns `Interrupted` while the holder still holds the lock, and
    /// one that is not called off gets the lock once the holder lets go.
    #[test]
    fn an_abandoned_wait_returns_interrupted() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{mpsc, Arc};
        use std::time::Duration;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("blob.lock");
        let held = FileLock::acquire(&path).expect("first acquire");

        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let waiter = std::thread::spawn({
            let (path, stop) = (path.clone(), stop.clone());
            move || {
                let got = FileLock::acquire_unless(&path, || stop.load(Ordering::SeqCst));
                tx.send(got.map(drop)).expect("report");
            }
        });
        assert!(
            rx.recv_timeout(Duration::from_millis(300)).is_err(),
            "the wait ended while the lock was held and nobody gave up"
        );
        stop.store(true, Ordering::SeqCst);
        let err = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("giving up must end the wait")
            .expect_err("no lock was taken");
        assert_eq!(err.kind(), io::ErrorKind::Interrupted);
        waiter.join().expect("waiter thread panicked");

        drop(held);
        FileLock::acquire_unless(&path, || false).expect("a free lock is taken at once");
    }

    /// A blocking `acquire` must actually wait for the holder rather than sailing through — the
    /// failure mode that a `try_acquire`-only test cannot see.
    #[test]
    fn a_blocking_acquire_waits_for_the_holder() {
        use std::sync::mpsc;
        use std::time::Duration;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("blob.lock");
        let held = FileLock::acquire(&path).expect("first acquire");

        let (tx, rx) = mpsc::channel();
        let waiter = std::thread::spawn({
            let path = path.clone();
            move || {
                let lock = FileLock::acquire(&path).expect("second acquire");
                tx.send(()).expect("report acquisition");
                drop(lock);
            }
        });

        assert!(
            rx.recv_timeout(Duration::from_millis(300)).is_err(),
            "acquire returned while the lock was held"
        );
        drop(held);
        rx.recv_timeout(Duration::from_secs(10))
            .expect("acquire must complete once the holder releases");
        waiter.join().expect("waiter thread panicked");
    }
}
