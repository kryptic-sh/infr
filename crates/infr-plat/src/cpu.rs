//! Facts about the host CPU that the standard library does not report.

/// The number of physical cores, or `None` where this platform has no probe here.
///
/// `std::thread::available_parallelism` counts LOGICAL processors, so with SMT it reports two per
/// core. That is the right answer for many things and the wrong one for a spin-waiting compute
/// pool, where two workers on one core's hyperthreads fight over the same execution units: on a
/// Ryzen 9 9950X3D under Windows, the CPU backend decoded Qwen3-0.6B at ~62 tok/s on 32 threads and
/// ~136 tok/s on 16 (measured 2026-09-23).
///
/// **Windows**: one `RelationProcessorCore` record per core from
/// `GetLogicalProcessorInformationEx`, which spans every processor group.
///
/// **Everywhere else**: `None`, deliberately — not a guess from `available_parallelism`. Callers
/// keep their logical-count default there, and a platform gains a probe here once someone has
/// measured that a physical-core default helps on it (Linux has not been measured; backlog B76).
#[cfg(windows)]
pub fn physical_cores() -> Option<usize> {
    use windows::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore,
    };

    let mut len = 0u32;
    // SAFETY: a size query — a null buffer with a zero length asks only for the length needed,
    // written to `len`. It "fails" with ERROR_INSUFFICIENT_BUFFER by design; `len` is the answer.
    let _ = unsafe { GetLogicalProcessorInformationEx(RelationProcessorCore, None, &mut len) };
    if len == 0 {
        return None;
    }
    // `u64` words, not bytes: the records embed pointer-sized fields, so the buffer must carry
    // their alignment rather than a byte vector's.
    let mut words = vec![0u64; (len as usize).div_ceil(8)];
    // SAFETY: `words` is a live, 8-aligned buffer of at least `len` bytes, and `len` tells the call
    // exactly that; it writes at most `len` bytes and updates `len` to what it wrote.
    unsafe {
        GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            Some(words.as_mut_ptr().cast()),
            &mut len,
        )
    }
    .ok()?;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    count_records(&bytes[..len as usize])
}

#[cfg(not(windows))]
pub fn physical_cores() -> Option<usize> {
    None
}

/// Count the variable-length records `GetLogicalProcessorInformationEx` returns, or `None` if the
/// buffer does not parse. Each record starts with two `u32`s — `Relationship`, then `Size`, the
/// record's whole length in bytes — so walking the `Size` chain visits every record.
#[cfg(any(windows, test))]
fn count_records(buf: &[u8]) -> Option<usize> {
    /// `Relationship` and `Size`, the fixed head every record starts with.
    const HEAD: usize = 8;
    let (mut at, mut n) = (0usize, 0usize);
    while at < buf.len() {
        let size_bytes: [u8; 4] = buf.get(at + 4..at + HEAD)?.try_into().ok()?;
        let size = u32::from_ne_bytes(size_bytes) as usize;
        if size < HEAD || at + size > buf.len() {
            return None;
        }
        n += 1;
        at += size;
    }
    (n > 0).then_some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(size: u32) -> Vec<u8> {
        let mut r = vec![0u8; (size as usize).max(8)]; // room for the head even when `size` lies
        r[4..8].copy_from_slice(&size.to_ne_bytes());
        r
    }

    #[test]
    fn counts_variable_length_records() {
        let buf = [record(48), record(80), record(48)].concat();
        assert_eq!(count_records(&buf), Some(3));
    }

    #[test]
    fn rejects_a_buffer_that_does_not_parse() {
        assert_eq!(count_records(&[]), None, "no records");
        assert_eq!(count_records(&record(4)), None, "size below the header");
        let mut overrun = record(48);
        overrun.truncate(40);
        assert_eq!(count_records(&overrun), None, "record runs past the end");
    }

    /// On Windows the probe must answer, and with a plausible number: at least one core, and no
    /// more cores than logical processors.
    #[cfg(windows)]
    #[test]
    fn physical_cores_is_plausible() {
        let cores = physical_cores().expect("Windows always reports its cores");
        let logical = std::thread::available_parallelism().unwrap().get();
        assert!(
            (1..=logical).contains(&cores),
            "{cores} cores vs {logical} logical"
        );
    }
}
