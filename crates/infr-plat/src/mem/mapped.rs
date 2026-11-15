//! A file mapping's pages in this process's resident set: how many there are, and handing them
//! back to the OS's file cache without unmapping anything.
//!
//! Why this exists: a model uploaded to a discrete GPU is read once, through its mapping, and then
//! lives in VRAM. On Windows the mapped pages the upload touched stay in the process working set,
//! where they count as in use rather than as standby cache, so the host loses roughly the model's
//! size in available memory for as long as the model is loaded. [`release_mapped`] moves them back
//! to the cache; a later read of the same bytes faults them in again from there (or from disk).
//!
//! **Verified on:** Windows (`VirtualUnlock` / `QueryWorkingSetEx`) on the dev box, and Linux
//! (`madvise(MADV_DONTNEED)` / `/proc/self/pagemap`) by the same tests on CI. Every other platform,
//! macOS included, answers [`std::io::ErrorKind::Unsupported`]: macOS's `MADV_DONTNEED` is only a
//! priority hint and nothing here could observe whether it took, and its GPUs are unified memory,
//! where there is no second copy to release.

use std::io;

/// Remove the pages holding `bytes` from this process's resident set, keeping the mapping valid.
///
/// Every page containing at least one byte of `bytes` is released, so a page shared with a
/// neighbouring range goes too; reading it again faults it back in. Nothing is copied, unmapped or
/// freed: the bytes read the same before and after.
///
/// - **Windows:** `VirtualUnlock` on the range. On pages that are not locked it removes them from
///   the working set and reports `ERROR_NOT_LOCKED`, which is that call's success here. Clean
///   file-backed pages go to the standby list, which the system counts as available.
/// - **Linux:** `madvise(MADV_DONTNEED)`. On a shared file mapping this drops the page-table
///   entries and leaves the pages in the page cache.
/// - **Anything else:** `Err` with [`io::ErrorKind::Unsupported`]. Nothing is released.
///
/// An empty slice is `Ok` and does nothing.
///
/// # Safety
///
/// `bytes` must lie inside a shared, read-only, file-backed mapping (`memmap2::Mmap::map` over a
/// file, for example). On Linux `MADV_DONTNEED` discards a private or anonymous page outright, and
/// the next read of it returns zeros rather than the bytes the `&[u8]` promised; for a shared file
/// mapping it returns the file's bytes again, which a read-only mapping guarantees are the same.
pub unsafe fn release_mapped(bytes: &[u8]) -> io::Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    #[cfg(windows)]
    {
        use windows::Win32::Foundation::ERROR_NOT_LOCKED;
        use windows::Win32::System::Memory::VirtualUnlock;
        // SAFETY: the range is a live borrow, so every page it names is committed in this process;
        // `VirtualUnlock` changes residency only, never contents.
        match unsafe { VirtualUnlock(bytes.as_ptr().cast(), bytes.len()) } {
            // Returned when some page in the range WAS locked: it is now unlocked but still
            // resident. Nothing in this process locks pages, so this is not expected; it is not an
            // error either, just a release that did less.
            Ok(()) => Ok(()),
            Err(e) if e.code() == ERROR_NOT_LOCKED.to_hresult() => Ok(()),
            Err(e) => Err(io::Error::from_raw_os_error(e.code().0 & 0xFFFF)),
        }
    }
    #[cfg(target_os = "linux")]
    {
        let (start, len) = page_span(bytes, page_size());
        // SAFETY: `start..start + len` covers only pages holding bytes of `bytes`, which the caller
        // guarantees lie in a shared, read-only file mapping; `MADV_DONTNEED` there drops this
        // process's page-table entries and a later read refaults the file's bytes.
        let rc = unsafe { libc::madvise(start as *mut libc::c_void, len, libc::MADV_DONTNEED) };
        if rc == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        Err(unsupported())
    }
}

/// Bytes of the pages holding `bytes` that are in this process's resident set right now: whole
/// pages, so the figure can exceed `bytes.len()` by up to a page at each end.
///
/// - **Windows:** `QueryWorkingSetEx`, counting the pages whose `Valid` bit is set.
/// - **Linux:** `/proc/self/pagemap`, counting the pages whose "present" bit is set. That is this
///   process's page table, not the page cache: a page still cached but released by
///   [`release_mapped`] is not counted (which `mincore` would count).
/// - **Anything else:** `Err` with [`io::ErrorKind::Unsupported`].
pub fn resident_bytes(bytes: &[u8]) -> io::Result<u64> {
    if bytes.is_empty() {
        return Ok(0);
    }
    #[cfg(windows)]
    {
        windows_resident_pages(bytes).map(|n| n * page_size() as u64)
    }
    #[cfg(target_os = "linux")]
    {
        linux_resident_pages(bytes).map(|n| n * page_size() as u64)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        Err(unsupported())
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn unsupported() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "releasing mapped pages from the resident set is implemented on Windows and Linux only",
    )
}

/// The host's page size in bytes.
#[cfg(windows)]
fn page_size() -> usize {
    use windows::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    let mut info = SYSTEM_INFO::default();
    // SAFETY: an out-parameter the call fills completely; it cannot fail.
    unsafe { GetSystemInfo(&mut info) };
    info.dwPageSize as usize
}

/// The host's page size in bytes.
#[cfg(target_os = "linux")]
fn page_size() -> usize {
    // SAFETY: `sysconf` reads a constant; `_SC_PAGESIZE` is always defined on Linux.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// The page-aligned `(start, len)` covering every page that holds a byte of `bytes`.
#[cfg(any(windows, target_os = "linux"))]
fn page_span(bytes: &[u8], page: usize) -> (usize, usize) {
    let addr = bytes.as_ptr() as usize;
    let start = addr - addr % page;
    let end = (addr + bytes.len()).div_ceil(page) * page;
    (start, end - start)
}

#[cfg(windows)]
fn windows_resident_pages(bytes: &[u8]) -> io::Result<u64> {
    use windows::Win32::System::ProcessStatus::{
        QueryWorkingSetEx, PSAPI_WORKING_SET_EX_BLOCK, PSAPI_WORKING_SET_EX_INFORMATION,
    };
    use windows::Win32::System::Threading::GetCurrentProcess;

    /// Pages asked about per call, so a multi-GB range does not need one multi-MB buffer.
    const BATCH: usize = 1 << 14;
    let page = page_size();
    let (start, len) = page_span(bytes, page);
    let pages = len / page;
    let mut resident = 0u64;
    let mut buf = Vec::with_capacity(BATCH.min(pages));
    for first in (0..pages).step_by(BATCH) {
        buf.clear();
        buf.extend(
            (first..pages.min(first + BATCH)).map(|i| PSAPI_WORKING_SET_EX_INFORMATION {
                VirtualAddress: (start + i * page) as *mut core::ffi::c_void,
                VirtualAttributes: PSAPI_WORKING_SET_EX_BLOCK { Flags: 0 },
            }),
        );
        // SAFETY: `buf` holds `buf.len()` initialized entries and `cb` is exactly their size; the
        // call only writes each entry's attributes.
        unsafe {
            QueryWorkingSetEx(
                GetCurrentProcess(),
                buf.as_mut_ptr().cast(),
                std::mem::size_of_val(buf.as_slice()) as u32,
            )
        }
        .map_err(|e| io::Error::from_raw_os_error(e.code().0 & 0xFFFF))?;
        // Bit 0 of the attributes is `Valid`: the page is in this process's working set.
        // SAFETY: every bit pattern is a valid `usize`, and the call wrote the whole union.
        resident += buf
            .iter()
            .filter(|e| unsafe { e.VirtualAttributes.Flags } & 1 != 0)
            .count() as u64;
    }
    Ok(resident)
}

#[cfg(target_os = "linux")]
fn linux_resident_pages(bytes: &[u8]) -> io::Result<u64> {
    use std::os::unix::fs::FileExt;

    /// Pages read per `pread`, so a multi-GB range does not need one multi-MB buffer.
    const BATCH: usize = 1 << 14;
    /// `/proc/pid/pagemap`'s "page present" bit (Documentation/admin-guide/mm/pagemap.rst).
    const PRESENT: u64 = 1 << 63;
    let page = page_size();
    let (start, len) = page_span(bytes, page);
    let pages = len / page;
    let file = std::fs::File::open("/proc/self/pagemap")?;
    let mut raw = vec![0u8; BATCH.min(pages) * 8];
    let mut resident = 0u64;
    for first in (0..pages).step_by(BATCH) {
        let n = BATCH.min(pages - first);
        let buf = &mut raw[..n * 8];
        file.read_exact_at(buf, ((start / page + first) * 8) as u64)?;
        resident += buf
            .as_chunks::<8>()
            .0
            .iter()
            .filter(|&&e| u64::from_ne_bytes(e) & PRESENT != 0)
            .count() as u64;
    }
    Ok(resident)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Pages in the test file: enough that a release leaving a stray page or two behind still
    /// reads as released.
    const PAGES: usize = 256;

    /// A temporary file of `PAGES` pages of a pattern no two pages share, mapped read-only.
    fn mapped_file() -> (tempfile::NamedTempFile, memmap2::Mmap) {
        const PAGE: usize = 4096;
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let body: Vec<u8> = (0..PAGES * PAGE)
            .map(|i| (i / PAGE) as u8 ^ (i % 251) as u8)
            .collect();
        file.write_all(&body).expect("write");
        file.flush().expect("flush");
        // SAFETY: the file is private to this test and not written while mapped.
        let map = unsafe { memmap2::Mmap::map(file.as_file()) }.expect("map");
        (file, map)
    }

    /// Touch every page of a mapped file, release it, and see it leave the resident set while its
    /// bytes read back identical. Platforms without the calls skip visibly; the test still checks
    /// that they say so rather than claiming a release.
    #[test]
    fn a_released_mapping_leaves_the_resident_set_and_reads_back_the_same() {
        let (file, map) = mapped_file();
        let want = std::fs::read(file.path()).expect("read file");
        let sum: u64 = map.iter().map(|&b| b as u64).sum();
        assert_eq!(sum, want.iter().map(|&b| b as u64).sum::<u64>());

        // SAFETY: `map` is a read-only mapping of a file nobody writes.
        let released = unsafe { release_mapped(&map) };
        let touched = resident_bytes(&map);
        if cfg!(any(windows, target_os = "linux")) {
            released.expect("release");
        } else {
            let e = released.expect_err("an unimplemented platform must not claim a release");
            assert_eq!(e.kind(), io::ErrorKind::Unsupported);
            let e = touched.expect_err("an unimplemented platform must not invent a count");
            assert_eq!(e.kind(), io::ErrorKind::Unsupported);
            eprintln!("skip: mapped-page release is not implemented on this platform");
            return;
        }

        let after = touched.expect("residency after release");
        assert!(
            after <= map.len() as u64 / 16,
            "{after} of {} bytes still resident after the release",
            map.len()
        );

        // The mapping is still valid and still the file.
        assert_eq!(
            &map[..],
            &want[..],
            "a released mapping must read back the file"
        );
        let back = resident_bytes(&map).expect("residency after re-read");
        assert!(
            back >= map.len() as u64 / 2,
            "reading every page again should fault them back in, got {back} of {}",
            map.len()
        );
    }

    /// The residency probe can see a page come in: before any read nothing (or little) is
    /// resident, after a full read nearly all of it is. Without this the release test above could
    /// pass against a probe that always answers zero.
    #[cfg(any(windows, target_os = "linux"))]
    #[test]
    fn the_residency_probe_sees_pages_fault_in() {
        let (_file, map) = mapped_file();
        let before = resident_bytes(&map).expect("residency");
        let sum: u64 = map.iter().map(|&b| b as u64).sum();
        std::hint::black_box(sum);
        let after = resident_bytes(&map).expect("residency");
        assert!(
            after >= map.len() as u64 * 3 / 4,
            "a full read left only {after} of {} bytes resident",
            map.len()
        );
        assert!(
            after > before,
            "residency did not rise: {before} -> {after}"
        );
    }

    #[test]
    fn an_empty_range_is_a_no_op() {
        // SAFETY: an empty slice names no page at all.
        let released = unsafe { release_mapped(&[]) };
        assert!(released.is_ok());
        assert_eq!(resident_bytes(&[]).ok(), Some(0));
    }

    #[cfg(any(windows, target_os = "linux"))]
    #[test]
    fn page_span_covers_every_touched_page() {
        let buf = [0u8; 1];
        let p = 4096;
        let addr = buf.as_ptr() as usize;
        let (start, len) = page_span(&buf, p);
        assert_eq!(start % p, 0);
        assert_eq!(len, p);
        assert!(start <= addr && addr < start + len);
    }
}
