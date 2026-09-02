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
/// Idempotent: installing twice replaces the previous handler. Errors are returned rather than
/// panicked on, since a CLI can run without one.
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
    LATCH.store(on_signal as usize, std::sync::atomic::Ordering::Relaxed);

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

/// The caller's latch, as a raw `fn` pointer so the handler can read it with one relaxed atomic
/// load. A `Mutex` or a `Box<dyn Fn>` would be unusable from signal context.
#[cfg(unix)]
static LATCH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The installed handler. **Async-signal-safe by construction**: one relaxed atomic load, one
/// indirect call into the caller's latch, and — only for a second signal — `write` and `_exit`.
#[cfg(unix)]
extern "C" fn trampoline(signo: libc::c_int) {
    let raw = LATCH.load(std::sync::atomic::Ordering::Relaxed);
    if raw != 0 {
        // SAFETY: `raw` is only ever written by `install_handlers` from a `fn(i32) -> bool`, and
        // this handler is only reachable after that write.
        let latch: fn(i32) -> bool = unsafe { std::mem::transmute::<usize, fn(i32) -> bool>(raw) };
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
    use std::sync::atomic::{AtomicI32, Ordering};

    static SEEN: AtomicI32 = AtomicI32::new(0);

    fn latch(signo: i32) -> bool {
        SEEN.store(signo, Ordering::Relaxed);
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
        assert_eq!(
            SEEN.load(Ordering::Relaxed),
            libc::SIGTERM,
            "the latch must have been called with the signal that was raised"
        );
    }

    /// Installing twice replaces rather than accumulating or failing.
    #[test]
    fn installing_is_idempotent() {
        install_handlers(latch).expect("install once");
        install_handlers(latch).expect("install twice");
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
