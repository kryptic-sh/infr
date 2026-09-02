//! Facts about processes: is one alive, and how this one reacts to a signal.

/// Is `pid` still running?
///
/// **Unix**: `kill(pid, 0)` delivers no signal and only asks the kernel whether the process
/// exists. `Ok` = alive; `EPERM` = alive but not ours (a foreign process that reused the pid —
/// still alive, so still a reason not to reclaim whatever it owns); anything else (`ESRCH`) =
/// gone.
///
/// **Everywhere else**: `true`, unconditionally and on purpose. A caller uses this as a tripwire
/// — "did the process that seeded this cache die holding it?" — and guessing WRONG discards a live
/// process's state. Answering "alive" loses the tripwire (the sweep never fires) but never
/// misbehaves. This is a deliberately conservative stub, not an implementation: a platform added
/// here should get a real probe rather than inherit this arm.
#[cfg(unix)]
pub fn pid_alive(pid: i32) -> bool {
    // SAFETY: `kill` with signal 0 performs only an existence/permission check — it delivers
    // nothing and mutates nothing.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
pub fn pid_alive(_pid: i32) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a platform with a real probe, both answers must be reachable — otherwise a `pid_alive`
    /// that returns a constant passes every caller's test.
    #[cfg(unix)]
    #[test]
    fn a_live_pid_and_a_dead_one_answer_differently() {
        let me = std::process::id() as i32;
        assert!(pid_alive(me), "this process must look alive to itself");

        // Reap a child so its pid is genuinely gone rather than a zombie, which `kill(0)` still
        // reports as existing.
        let mut child = std::process::Command::new("sh")
            .args(["-c", "exit 0"])
            .spawn()
            .expect("spawn a short-lived child");
        let dead = child.id() as i32;
        child.wait().expect("reap the child");
        assert!(!pid_alive(dead), "a reaped pid {dead} must not look alive");
    }

    /// The stub is a documented policy, so pin it: on a platform with no probe the answer is
    /// "alive", which loses the tripwire instead of discarding live state.
    #[cfg(not(unix))]
    #[test]
    fn without_a_probe_everything_looks_alive() {
        assert!(pid_alive(1));
        assert!(pid_alive(i32::MAX));
    }
}
