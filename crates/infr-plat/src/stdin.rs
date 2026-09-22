//! Reading standard input one byte at a time, so a caller can notice a signal between bytes.

use std::io;

/// Read one byte from standard input. `Ok(None)` is end of input.
///
/// Bytes, not chars: a multi-byte UTF-8 codepoint arrives one byte per call and is only a `char`
/// once it is whole. Assembling them is the caller's job.
///
/// **Unix**: a raw `read(2)` on fd 0, deliberately unbuffered and NOT retried on `EINTR` — the
/// error is handed back as [`io::ErrorKind::Interrupted`], which is the whole point. A caller
/// parked on this at an idle prompt can then check a shutdown latch and give up, instead of
/// sitting on the read until the user presses Enter. Every other interrupted syscall in the
/// process already retries internally.
///
/// **Windows, console input**: `ReadConsoleW`, decoded to UTF-8 and handed out a byte at a time.
/// A Ctrl-C while the console is reading makes it return nothing with `ERROR_OPERATION_ABORTED`,
/// which comes back as [`io::ErrorKind::Interrupted`] — the same contract as unix. `Stdin::read`
/// cannot provide it: std retries that exact case internally, so the caller would stay parked
/// until Enter. A line starting with Ctrl-Z is end of input, the console's EOF convention (and
/// std's). Lines end `\r\n` here, as the console delivers them.
///
/// **Windows, redirected input** (a pipe or a file), **and every other target**: `Stdin::read`,
/// which cannot be interrupted, so `Interrupted` is never returned and a caller polling a latch
/// between bytes will not see it until input arrives.
pub fn read_byte() -> io::Result<Option<u8>> {
    #[cfg(unix)]
    {
        let mut b = 0u8;
        // SAFETY: a 1-byte read into a live stack byte.
        match unsafe { libc::read(0, std::ptr::addr_of_mut!(b).cast(), 1) } {
            0 => Ok(None),
            1 => Ok(Some(b)),
            _ => Err(io::Error::last_os_error()),
        }
    }
    #[cfg(windows)]
    {
        match console::read_byte()? {
            Some(result) => Ok(result),
            None => read_byte_std(),
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        read_byte_std()
    }
}

#[cfg(not(unix))]
fn read_byte_std() -> io::Result<Option<u8>> {
    use io::Read;
    let mut b = 0u8;
    match io::stdin().lock().read(std::slice::from_mut(&mut b))? {
        0 => Ok(None),
        _ => Ok(Some(b)),
    }
}

#[cfg(windows)]
mod console {
    use std::collections::VecDeque;
    use std::io;
    use std::sync::{Mutex, PoisonError};
    use windows::Win32::Foundation::{GetLastError, ERROR_OPERATION_ABORTED, HANDLE};
    use windows::Win32::System::Console::{
        GetConsoleMode, GetStdHandle, ReadConsoleW, CONSOLE_MODE, STD_INPUT_HANDLE,
    };

    /// UTF-16 units per `ReadConsoleW`. A cooked-mode read returns at most one line, so this only
    /// bounds how many calls a long line takes, never what can be typed.
    const READ_UNITS: usize = 512;

    /// The console's EOF marker when typed at the start of a line.
    const CTRL_Z: char = '\u{1a}';

    /// Bytes decoded from the console but not yet handed out.
    static PENDING: Mutex<VecDeque<u8>> = Mutex::new(VecDeque::new());

    /// One byte from the console, `Ok(None)` when standard input is not a console at all (so the
    /// caller falls back to `Stdin`), or `Ok(Some(None))` at end of input.
    pub(super) fn read_byte() -> io::Result<Option<Option<u8>>> {
        let mut pending = PENDING.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(b) = pending.pop_front() {
            return Ok(Some(Some(b)));
        }
        // SAFETY: querying this process's own standard input handle.
        let Ok(handle) = (unsafe { GetStdHandle(STD_INPUT_HANDLE) }) else {
            return Ok(None);
        };
        let mut mode = CONSOLE_MODE(0);
        // SAFETY: `mode` is a live out-parameter; failure just means "not a console".
        if unsafe { GetConsoleMode(handle, &mut mode) }.is_err() {
            return Ok(None);
        }

        let mut units = [0u16; READ_UNITS];
        // The last slot stays free for the low half of a surrogate pair the read may split.
        let mut len = read_units(handle, &mut units[..READ_UNITS - 1])?;
        if len == 0 {
            return Ok(Some(None));
        }
        // A read that stops between the halves of a surrogate pair would decode as two
        // replacement characters; fetch the low half before decoding.
        if (0xD800..0xDC00).contains(&units[len - 1]) {
            len += read_units(handle, &mut units[len..len + 1])?;
        }
        let text = String::from_utf16_lossy(&units[..len]);
        if text.starts_with(CTRL_Z) {
            return Ok(Some(None));
        }
        pending.extend(text.bytes());
        Ok(Some(pending.pop_front()))
    }

    /// `ReadConsoleW` into `buf`, returning the units read. A read cut short by Ctrl-C is
    /// [`io::ErrorKind::Interrupted`] however the console reports it: documented as success with
    /// nothing read and `ERROR_OPERATION_ABORTED` as the last error, but a failure carrying the
    /// same code means the same thing.
    fn read_units(handle: HANDLE, buf: &mut [u16]) -> io::Result<usize> {
        let mut read = 0u32;
        // SAFETY: `buf` is a live, writable slice of `buf.len()` UTF-16 units, and `read` a live
        // out-parameter. The unit count is bounded by `READ_UNITS`, so the cast cannot truncate.
        let result = unsafe {
            ReadConsoleW(
                handle,
                buf.as_mut_ptr().cast(),
                buf.len() as u32,
                &mut read,
                None,
            )
        };
        let aborted = match &result {
            // SAFETY: reads this thread's last-error value, set by the `ReadConsoleW` above.
            Ok(()) => read == 0 && unsafe { GetLastError() } == ERROR_OPERATION_ABORTED,
            Err(e) => e.code() == ERROR_OPERATION_ABORTED.to_hresult(),
        };
        if aborted {
            return Err(io::ErrorKind::Interrupted.into());
        }
        result?;
        Ok(read as usize)
    }
}
