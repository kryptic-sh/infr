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
/// **Windows** has no signals; the nearest equivalent is the console control handler
/// (`SetConsoleCtrlHandler`). Ctrl-C and Ctrl-Break latch as `SIGINT`; closing the console window,
/// logging off and shutting down latch as `SIGTERM`. The handler runs on a thread the system
/// creates for it rather than interrupting one, so the async-signal-safe rules are stricter than
/// it needs — the latch is still one atomic, which is what it must be on unix anyway. Two
/// behaviours differ from unix and are stated here rather than hidden:
///
/// - For a close/logoff/shutdown event the system terminates the process as soon as the handler
///   RETURNS. So the handler parks its thread instead: the process then exits when `main` finishes
///   draining the GPU, or when the system's own close timeout expires, whichever is first.
/// - A second event force-exits through `TerminateProcess`, the `_exit(2)` analogue — no atexit
///   handlers, no DLL detach notifications into a GPU driver that may be mid-submit.
///
/// **Other targets** (neither unix nor Windows) get a no-op returning `Ok(())`: the process keeps
/// the default disposition and dies on the first signal without draining the GPU.
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

#[cfg(windows)]
pub fn install_handlers(on_signal: fn(i32) -> bool) -> io::Result<()> {
    use windows::Win32::System::Console::SetConsoleCtrlHandler;

    // Set BEFORE registering, so the handler cannot fire against an empty latch.
    let _ = LATCH.set(on_signal);
    // SAFETY: `console_handler` has the `PHANDLER_ROUTINE` signature and lives for the whole
    // process. Registering it twice is harmless: the system calls the most recently added routine
    // first, and this one returns TRUE (handled) for every event it latches.
    unsafe { SetConsoleCtrlHandler(Some(console_handler), true) }?;
    Ok(())
}

#[cfg(not(any(unix, windows)))]
pub fn install_handlers(on_signal: fn(i32) -> bool) -> io::Result<()> {
    let _ = on_signal;
    Ok(())
}

/// The C runtime's numbering (MSVC `<signal.h>`), which is also the POSIX one for these two. Using
/// it keeps `128 + signo` the same exit status on every platform: 130 for Ctrl-C, 143 for a close.
#[cfg(windows)]
const SIGINT: i32 = 2;
#[cfg(windows)]
const SIGTERM: i32 = 15;

/// Which signal a console control event stands for, or `None` for an event this process leaves to
/// the next handler in the chain (ultimately the system default).
#[cfg(windows)]
fn signal_for(ctrl_type: u32) -> Option<i32> {
    use windows::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT, CTRL_C_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT,
    };
    match ctrl_type {
        CTRL_C_EVENT | CTRL_BREAK_EVENT => Some(SIGINT),
        CTRL_CLOSE_EVENT | CTRL_LOGOFF_EVENT | CTRL_SHUTDOWN_EVENT => Some(SIGTERM),
        _ => None,
    }
}

/// The installed console control handler. Runs on its own system-created thread.
#[cfg(windows)]
unsafe extern "system" fn console_handler(ctrl_type: u32) -> windows::Win32::Foundation::BOOL {
    use windows::Win32::Foundation::{FALSE, TRUE};

    let Some(signo) = signal_for(ctrl_type) else {
        return FALSE;
    };
    if !LATCH.get().is_some_and(|latch| latch(signo)) {
        force_exit(signo);
    }
    if signo == SIGTERM {
        // Returning would let the system terminate the process right now, mid-submit. Hold this
        // thread instead; `main` ends the process once the GPU has drained (see the fn docs).
        loop {
            std::thread::park();
        }
    }
    TRUE
}

/// The second-signal path: warn on stderr and terminate without running any cleanup.
#[cfg(windows)]
fn force_exit(signo: i32) -> ! {
    use windows::Win32::Storage::FileSystem::WriteFile;
    use windows::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE};
    use windows::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};

    // SAFETY: plain Win32 calls on this process's own stderr handle and pseudo-handle. Both
    // results are deliberately dropped: the write is a courtesy that a closed stderr must not
    // block, and a failed terminate falls through to `abort` below.
    unsafe {
        if let Ok(stderr) = GetStdHandle(STD_ERROR_HANDLE) {
            let _ = WriteFile(stderr, Some(SECOND_SIGNAL_MSG), None, None);
        }
        let _ = TerminateProcess(GetCurrentProcess(), (128 + signo) as u32);
    }
    std::process::abort()
}

/// The caller's latch.
///
/// A `OnceLock` because reading it from signal context must be async-signal-safe: `get` is one
/// atomic load and a reference, with no allocation and no locking. A `Mutex` or a `Box<dyn Fn>`
/// would be unusable here, and a raw `fn`-pointer-as-`usize` would need a `transmute` to get back
/// — this needs no `unsafe` at all.
#[cfg(any(unix, windows))]
static LATCH: std::sync::OnceLock<fn(i32) -> bool> = std::sync::OnceLock::new();

/// What a second signal prints on its way out, on every platform that installs a handler.
#[cfg(any(unix, windows))]
const SECOND_SIGNAL_MSG: &[u8] = b"\ninfr: second signal - exiting NOW without draining the GPU. \
    If a submit was in flight, the device may stay wedged until reboot.\n";

/// The installed handler. **Async-signal-safe by construction**: one relaxed atomic load, one
/// indirect call into the caller's latch, and — only for a second signal — `write` and `_exit`.
#[cfg(unix)]
extern "C" fn trampoline(signo: libc::c_int) {
    if let Some(latch) = LATCH.get() {
        if latch(signo) {
            return; // first signal: let the caller wind down at its next safe point
        }
    }
    // SAFETY: `write` and `_exit` are async-signal-safe; the message is a 'static byte string.
    unsafe {
        libc::write(
            2,
            SECOND_SIGNAL_MSG.as_ptr().cast(),
            SECOND_SIGNAL_MSG.len(),
        );
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

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI32, Ordering};
    use windows::Win32::Foundation::{FALSE, TRUE};
    use windows::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT, CTRL_C_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT,
    };

    static SAW: AtomicI32 = AtomicI32::new(0);

    fn latch(signo: i32) -> bool {
        SAW.store(signo, Ordering::Relaxed);
        true // always "first signal", so the handler returns instead of terminating
    }

    #[test]
    fn console_events_map_to_the_posix_signal_numbers() {
        assert_eq!(signal_for(CTRL_C_EVENT), Some(SIGINT));
        assert_eq!(signal_for(CTRL_BREAK_EVENT), Some(SIGINT));
        assert_eq!(signal_for(CTRL_CLOSE_EVENT), Some(SIGTERM));
        assert_eq!(signal_for(CTRL_LOGOFF_EVENT), Some(SIGTERM));
        assert_eq!(signal_for(CTRL_SHUTDOWN_EVENT), Some(SIGTERM));
        assert_eq!(
            signal_for(3),
            None,
            "3 is unassigned; it must fall through to the default"
        );
    }

    /// Installing succeeds, and the routine it installs routes Ctrl-C into the latch and reports
    /// it handled. A real console event cannot be raised here without also hitting the test runner
    /// that shares the console, so the routine is called directly; the registration is the one
    /// `SetConsoleCtrlHandler` call whose error `install_handlers` returns.
    #[test]
    fn an_installed_handler_latches_ctrl_c() {
        install_handlers(latch).expect("install");
        install_handlers(latch).expect("installing twice is harmless");
        // SAFETY: called on this thread with a latch that returns `true`, so the routine neither
        // parks (Ctrl-C is not a close event) nor terminates.
        let handled = unsafe { console_handler(CTRL_C_EVENT) };
        assert_eq!(handled, TRUE);
        assert_eq!(SAW.load(Ordering::Relaxed), SIGINT);
        // SAFETY: as above; an unmapped event returns before touching the latch.
        assert_eq!(unsafe { console_handler(3) }, FALSE);
    }
}

/// The documented no-op, pinned so it stays a deliberate stub rather than drifting into an
/// implementation nobody checked. A platform that grows a real handler replaces this test.
#[cfg(all(test, not(any(unix, windows))))]
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
