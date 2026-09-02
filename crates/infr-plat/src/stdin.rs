//! Reading standard input one byte at a time, so a caller can notice a signal between bytes.

use std::io;

/// Read one byte from file descriptor 0. `Ok(None)` is end of input.
///
/// Bytes, not chars: a multi-byte UTF-8 codepoint arrives one byte per read and is only a `char`
/// once it is whole. Assembling them is the caller's job.
///
/// **Unix**: a raw `read(2)`, deliberately unbuffered and NOT retried on `EINTR` — the error is
/// handed back as [`io::ErrorKind::Interrupted`], which is the whole point. A caller parked on
/// this at an idle prompt can then check a shutdown latch and give up, instead of sitting on the
/// read until the user presses Enter. Every other interrupted syscall in the process already
/// retries internally.
///
/// **Elsewhere**: `Stdin::read`. There are no POSIX signals, so this **cannot be interrupted** and
/// `Interrupted` is never returned; a caller polling a latch between bytes will not see it until
/// input arrives. That is a real difference in what the call promises, not a spelling — a console
/// read there returns when the line is submitted, so a Ctrl-C at an idle prompt is handled by the
/// console rather than by this loop.
pub fn read_byte() -> io::Result<Option<u8>> {
    let mut b = 0u8;
    #[cfg(unix)]
    {
        // SAFETY: a 1-byte read into a live stack byte.
        match unsafe { libc::read(0, std::ptr::addr_of_mut!(b).cast(), 1) } {
            0 => Ok(None),
            1 => Ok(Some(b)),
            _ => Err(io::Error::last_os_error()),
        }
    }
    #[cfg(not(unix))]
    {
        use io::Read;
        match io::stdin().lock().read(std::slice::from_mut(&mut b))? {
            0 => Ok(None),
            _ => Ok(Some(b)),
        }
    }
}
