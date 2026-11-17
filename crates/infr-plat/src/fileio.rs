//! Positioned reads and writes: I/O at an explicit offset, without a seek.
//!
//! The point of a positioned operation is that it carries no shared cursor, so any number of
//! threads may read one open file concurrently. Unix spells that `pread`/`pwrite`; Windows spells
//! it `ReadFile`/`WriteFile` with an `OVERLAPPED` offset, which std exposes as
//! `seek_read`/`seek_write`.

use std::fs::File;
use std::io;

/// Read into `buf` at `offset`, returning how many bytes arrived.
///
/// A positioned read may return fewer bytes than asked for even mid-file, so a caller that needs
/// the buffer filled must loop; [`read_exact_at`] is that loop.
pub fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        file.seek_read(buf, offset)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "positioned reads need a unix or windows FileExt",
        ))
    }
}

/// Read exactly `buf.len()` bytes at `offset`, looping over short reads.
///
/// Hitting EOF before the buffer is full is an error, not a short success: a partially-filled
/// buffer that reads as a successful load is the silent-wrong-output case.
pub fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let n = read_at(file, &mut buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "short read: wanted {} bytes at {offset}, got {done}",
                    buf.len()
                ),
            ));
        }
        done += n;
    }
    Ok(())
}

/// Write all of `buf` at `offset`.
///
/// The loop is not a platform difference dressed up as one: std's unix `write_all_at` already
/// loops internally over short writes and `EINTR`, and Windows has no such wrapper, so both arms
/// perform the same operation.
pub fn write_all_at(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;

        let mut buf = buf;
        let mut offset = offset;
        while !buf.is_empty() {
            let n = file.seek_write(buf, offset)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write the full chunk at its offset",
                ));
            }
            buf = &buf[n..];
            offset += n as u64;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "positioned writes need a unix or windows FileExt",
        ))
    }
}

/// Prepare `file` for writes that arrive in any order, as a download split into ranges makes them.
///
/// **Windows**: marks the file sparse (`FSCTL_SET_SPARSE`). Without it NTFS fills everything below
/// a write past the end of the valid data with zeros before the write returns, so the last chunk
/// of a large file landing first stalls its writer for the time it takes to write the whole file
/// once. **Unix**: nothing to do, a write past the end leaves a hole. **Other targets**: an
/// `Unsupported` error.
///
/// An error leaves the file usable: writes still land where they are told, only slower. A
/// filesystem without sparse files (FAT, exFAT) answers that way.
pub fn allow_out_of_order_writes(file: &File) -> io::Result<()> {
    #[cfg(unix)]
    {
        let _ = file;
        Ok(())
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::System::Ioctl::FSCTL_SET_SPARSE;
        use windows::Win32::System::IO::DeviceIoControl;

        let mut returned = 0u32;
        // SAFETY: the handle is open for the borrow of `file`; the call takes no input or output
        // buffer, and `returned` outlives it.
        unsafe {
            DeviceIoControl(
                HANDLE(file.as_raw_handle()),
                FSCTL_SET_SPARSE,
                None,
                0,
                None,
                0,
                Some(&mut returned),
                None,
            )
        }
        .map_err(|_| io::Error::last_os_error())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = file;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "out-of-order writes are only prepared for on unix and windows",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn scratch(contents: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("f.bin");
        let mut f = File::create(&path).expect("create");
        f.write_all(contents).expect("write");
        (dir, path)
    }

    #[test]
    fn a_positioned_read_starts_where_it_is_told() {
        let (_d, path) = scratch(b"0123456789");
        let f = File::open(&path).expect("open");
        let mut buf = [0u8; 4];
        read_exact_at(&f, &mut buf, 3).expect("read at 3");
        assert_eq!(&buf, b"3456");
        // ...and does not move a cursor: the same call twice gives the same bytes.
        let mut again = [0u8; 4];
        read_exact_at(&f, &mut again, 3).expect("read at 3 again");
        assert_eq!(buf, again);
    }

    /// Running off the end must be an error. A version that returned `Ok` with a short buffer
    /// would pass the test above unchanged.
    #[test]
    fn reading_past_the_end_is_an_error() {
        let (_d, path) = scratch(b"0123456789");
        let f = File::open(&path).expect("open");
        let mut buf = [0u8; 8];
        let err = read_exact_at(&f, &mut buf, 6).expect_err("must not succeed past EOF");
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn a_positioned_write_lands_at_its_offset() {
        let (_d, path) = scratch(b"0123456789");
        let f = File::options().write(true).open(&path).expect("open rw");
        write_all_at(&f, b"abc", 4).expect("write at 4");
        drop(f);
        assert_eq!(std::fs::read(&path).expect("read back"), b"0123abc789");
    }

    /// Writing beyond the current end extends the file rather than failing or wrapping.
    #[test]
    fn a_positioned_write_past_the_end_extends() {
        let (_d, path) = scratch(b"ab");
        let f = File::options().write(true).open(&path).expect("open rw");
        write_all_at(&f, b"z", 4).expect("write at 4");
        drop(f);
        let got = std::fs::read(&path).expect("read back");
        assert_eq!(got.len(), 5, "file must have grown: {got:?}");
        assert_eq!(&got[..2], b"ab");
        assert_eq!(got[4], b'z');
    }

    /// A write far past the end of a prepared file must not allocate the gap below it. Allocated
    /// size is the observable: without the preparation NTFS backs the whole gap with zeros.
    #[test]
    fn a_prepared_file_keeps_a_hole_below_a_late_write() {
        const GAP: u64 = 64 << 20;
        let (_d, path) = scratch(b"");
        let f = File::options().write(true).open(&path).expect("open rw");
        allow_out_of_order_writes(&f).expect("prepare");
        write_all_at(&f, b"tail", GAP).expect("write past the gap");
        f.sync_all().expect("sync");
        assert_eq!(f.metadata().expect("metadata").len(), GAP + 4);
        let on_disk = allocated_bytes(&path);
        assert!(
            on_disk < GAP / 2,
            "the gap was allocated: {on_disk} bytes on disk"
        );
    }

    #[cfg(unix)]
    fn allocated_bytes(path: &std::path::Path) -> u64 {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(path).expect("metadata").blocks() * 512
    }

    #[cfg(windows)]
    fn allocated_bytes(path: &std::path::Path) -> u64 {
        use windows::core::HSTRING;
        use windows::Win32::Storage::FileSystem::GetCompressedFileSizeW;

        let mut high = 0u32;
        // SAFETY: the path is a NUL-terminated wide string owned by this frame, and `high`
        // outlives the call.
        let low = unsafe { GetCompressedFileSizeW(&HSTRING::from(path), Some(&mut high)) };
        (u64::from(high) << 32) | u64::from(low)
    }
}
