//! Facts about processes: is one alive, and how this one reacts to a signal.

/// Is `pid` still running?
///
/// A caller uses this as a tripwire — "did the process that seeded this cache die holding it?" —
/// and guessing WRONG discards a live process's state, so every arm answers "dead" only when the
/// OS positively says so, and "alive" for anything it cannot tell.
///
/// **Unix**: `kill(pid, 0)` delivers no signal and only asks the kernel whether the process
/// exists. `Ok` = alive; `EPERM` = alive but not ours (a foreign process that reused the pid —
/// still alive, so still a reason not to reclaim whatever it owns); anything else (`ESRCH`) =
/// gone.
///
/// **Windows**: `OpenProcess` for the least privileged query right. `ERROR_INVALID_PARAMETER` is
/// the answer for a pid no process holds = gone; any other refusal (`ERROR_ACCESS_DENIED` for a
/// protected or foreign process) = alive. An opened process is alive while its exit code reads
/// `STILL_ACTIVE`; a process object that outlived its process (someone still holds a handle)
/// reports its real exit code and so reads as gone. The one misreport, a process that exited
/// with code 259 (`STILL_ACTIVE`'s value), errs on the alive side.
///
/// **Everywhere else**: `true`, unconditionally and on purpose — it loses the tripwire (the sweep
/// never fires) but never misbehaves. A deliberately conservative stub, not an implementation: a
/// platform added here should get a real probe rather than inherit this arm.
#[cfg(unix)]
pub fn pid_alive(pid: i32) -> bool {
    // SAFETY: `kill` with signal 0 performs only an existence/permission check — it delivers
    // nothing and mutates nothing.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(windows)]
pub fn pid_alive(pid: i32) -> bool {
    use windows::Win32::Foundation::{CloseHandle, ERROR_INVALID_PARAMETER, STILL_ACTIVE};
    use windows::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    // Windows pids are `u32`; a negative one names no process this code could have recorded, and
    // "cannot tell" answers alive.
    let Ok(pid) = u32::try_from(pid) else {
        return true;
    };
    // SAFETY: opening a query-only handle to a process by id; the handle is closed below.
    let handle = match unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) } {
        Ok(handle) => handle,
        Err(e) => return e.code() != ERROR_INVALID_PARAMETER.to_hresult(),
    };
    let mut code = 0u32;
    // SAFETY: `handle` was just opened with the query right `GetExitCodeProcess` needs, and
    // `code` is a live out-parameter. Closing cannot usefully fail for a handle we own.
    unsafe {
        let queried = GetExitCodeProcess(handle, &mut code);
        let _ = CloseHandle(handle);
        queried.is_err() || code == STILL_ACTIVE.0 as u32
    }
}

#[cfg(not(any(unix, windows)))]
pub fn pid_alive(_pid: i32) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a platform with a real probe, both answers must be reachable — otherwise a `pid_alive`
    /// that returns a constant passes every caller's test.
    #[cfg(any(unix, windows))]
    #[test]
    fn a_live_pid_and_a_dead_one_answer_differently() {
        let me = std::process::id() as i32;
        assert!(pid_alive(me), "this process must look alive to itself");

        // Reap a child so its pid is genuinely gone rather than a zombie, which `kill(0)` still
        // reports as existing.
        let mut cmd = if cfg!(windows) {
            let mut cmd = std::process::Command::new("cmd");
            cmd.args(["/C", "exit 0"]);
            cmd
        } else {
            let mut cmd = std::process::Command::new("sh");
            cmd.args(["-c", "exit 0"]);
            cmd
        };
        let mut child = cmd.spawn().expect("spawn a short-lived child");
        let dead = child.id() as i32;
        child.wait().expect("reap the child");
        assert!(!pid_alive(dead), "a reaped pid {dead} must not look alive");
    }

    /// The stub is a documented policy, so pin it: on a platform with no probe the answer is
    /// "alive", which loses the tripwire instead of discarding live state.
    #[cfg(not(any(unix, windows)))]
    #[test]
    fn without_a_probe_everything_looks_alive() {
        assert!(pid_alive(1));
        assert!(pid_alive(i32::MAX));
    }
}
