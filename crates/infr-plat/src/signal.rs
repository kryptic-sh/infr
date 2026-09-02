//! Interrupt and termination signals, and what this process does about them.

use std::io;

/// Install a handler for `SIGINT` and `SIGTERM` (or the platform's nearest equivalent).
///
/// `on_signal` is called **from signal context**, so it must be async-signal-safe: no allocation,
/// no locking, no `println!` — which takes the stdout lock and deadlocks against an interrupted
/// `print!`. A lock-free atomic is the intended shape. It returns `true` when this was the FIRST
/// signal, meaning the process should latch a graceful shutdown and keep running; `false` means
/// the user has asked twice and is done waiting, and this function exits the process immediately
/// with `128 + signo` after writing a warning to fd 2.
///
/// Taking the latch as a `fn` pointer rather than calling into a shutdown module is what keeps
/// this crate a leaf. `write(2)` and `_exit(2)` are both on POSIX's async-signal-safe list, unlike
/// `exit`, which runs atexit handlers and flushes streams from a signal context.
///
/// Idempotent, with one caveat worth stating: installing twice re-arms the signal disposition,
/// but the FIRST latch wins — a second call with a *different* latch leaves the first in place.
/// That is a deliberate consequence of storing it somewhere a signal handler can read without
/// `unsafe`; nothing in this workspace installs two, and a caller that needs to swap one should
/// change its own latch rather than reinstall. Errors are returned rather than panicked on, since
/// a CLI can run without a handler.
///
/// **No `SA_RESTART`.** An interrupted blocking `read(2)` then returns `EINTR`, which is what lets
/// an idle chat prompt notice a Ctrl-C instead of sitting on the read until the user presses
/// Enter. Everything else in the process that can see an interrupted syscall already retries it.
///
/// **Not implemented off unix**, where this is a no-op returning `Ok(())`: the process keeps the
/// default disposition and dies on the first signal without draining the GPU. That is the
/// pre-existing behaviour, and it is a real gap rather than a spelling difference — a Windows
/// implementation needs `SetConsoleCtrlHandler`, whose handler runs on its own thread and so has
/// different safety rules than the async-signal-safe ones above.
#[cfg(unix)]
pub fn install_handlers(on_signal: fn(i32) -> bool) -> io::Result<()> {
    // Set BEFORE `sigaction`, so the handler cannot fire against an empty latch.
    let _ = LATCH.set(on_signal);

    // SAFETY: a zeroed `sigaction` with a valid handler pointer and an empty mask is exactly what
    // POSIX asks for; `trampoline` is async-signal-safe (see its own comment).
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = trampoline as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0;
        for signo in [libc::SIGINT, libc::SIGTERM] {
            if libc::sigaction(signo, &sa, std::ptr::null_mut()) != 0 {
                return Err(io::Error::last_os_error());
            }
        }
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn install_handlers(on_signal: fn(i32) -> bool) -> io::Result<()> {
    let _ = on_signal;
    Ok(())
}

/// The caller's latch.
///
/// A `OnceLock` because reading it from signal context must be async-signal-safe: `get` is one
/// atomic load and a reference, with no allocation and no locking. A `Mutex` or a `Box<dyn Fn>`
/// would be unusable here, and a raw `fn`-pointer-as-`usize` would need a `transmute` to get back
/// — this needs no `unsafe` at all.
#[cfg(unix)]
static LATCH: std::sync::OnceLock<fn(i32) -> bool> = std::sync::OnceLock::new();

/// The installed handler. **Async-signal-safe by construction**: one relaxed atomic load, one
/// indirect call into the caller's latch, and — only for a second signal — `write` and `_exit`.
#[cfg(unix)]
extern "C" fn trampoline(signo: libc::c_int) {
    if let Some(latch) = LATCH.get() {
        if latch(signo) {
            return; // first signal: let the caller wind down at its next safe point
        }
    }
    const MSG: &[u8] = b"\ninfr: second signal - exiting NOW without draining the GPU. \
        If a submit was in flight, the device may stay wedged until reboot.\n";
    // SAFETY: `write` and `_exit` are async-signal-safe; MSG is a 'static byte string.
    unsafe {
        libc::write(2, MSG.as_ptr().cast(), MSG.len());
        libc::_exit(128 + signo);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    // One slot per signal, not one shared cell: `install_handlers` keeps the FIRST latch, so both
    // tests below necessarily share this one, and they run concurrently in the same process.
    static SAW_INT: AtomicBool = AtomicBool::new(false);
    static SAW_TERM: AtomicBool = AtomicBool::new(false);

    fn latch(signo: i32) -> bool {
        match signo {
            libc::SIGINT => SAW_INT.store(true, Ordering::Relaxed),
            libc::SIGTERM => SAW_TERM.store(true, Ordering::Relaxed),
            other => panic!("unexpected signal {other}"),
        }
        true // always "first signal", so the handler returns instead of calling _exit
    }

    /// Installing must actually divert the signal to the caller's latch. Without this, an
    /// `install_handlers` that returned `Ok(())` and wired nothing up would look identical — and
    /// on the default disposition `SIGTERM` kills the test process, so a no-op fails loudly rather
    /// than passing quietly.
    #[test]
    fn an_installed_handler_receives_the_signal() {
        install_handlers(latch).expect("install");
        // SAFETY: raising SIGTERM in-process with a handler installed; the handler returns.
        unsafe { libc::raise(libc::SIGTERM) };
        assert!(
            SAW_TERM.load(Ordering::Relaxed),
            "the latch must have been called with the signal that was raised"
        );
    }

    /// Installing twice succeeds and leaves a working handler, rather than accumulating or
    /// failing. Raising afterwards is what proves the second call did not disarm the first.
    #[test]
    fn installing_twice_leaves_a_working_handler() {
        install_handlers(latch).expect("install once");
        install_handlers(latch).expect("install twice");
        // SAFETY: raising SIGINT in-process with a handler installed; the handler returns.
        unsafe { libc::raise(libc::SIGINT) };
        assert!(SAW_INT.load(Ordering::Relaxed));
    }
}

/// The documented no-op, pinned so it stays a deliberate stub rather than drifting into an
/// implementation nobody checked. A platform that grows a real handler replaces this test.
#[cfg(all(test, not(unix)))]
mod tests {
    use super::*;

    #[test]
    fn installing_is_a_no_op_that_reports_success() {
        fn latch(_signo: i32) -> bool {
            unreachable!("no handler is installed on this platform, so the latch cannot run")
        }
        install_handlers(latch).expect("the stub must report success, not an error");
    }
}
