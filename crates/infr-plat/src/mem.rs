//! How much host memory could be committed right now — and, crucially, what that number does and
//! does not account for.

/// A host-memory figure together with where it came from.
///
/// The provenance is not decoration. The three platforms answer genuinely different questions, and
/// a bare `Option<u64>` hides that behind one type — which is how an arena sized inside a container
/// becomes an OOM kill. A caller doing arithmetic wants [`Self::bytes`]; a caller REPORTING the
/// figure, or deciding how much of it to trust, wants [`Self::source`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Available {
    /// Bytes that could be committed right now.
    pub bytes: u64,
    /// Which probe produced [`Self::bytes`], and therefore what it accounts for.
    pub source: Source,
}

/// Which probe produced a figure. Each variant answers a different question; see [`available`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Linux `MemAvailable`, with no cgroup limit binding this process. Host-wide.
    LinuxMemAvailable,
    /// Linux `MemAvailable` clamped by the tightest cgroup limit above this process — the figure
    /// an arena must be sized from inside a container.
    LinuxCgroupClamped,
    /// Windows `ullAvailPhys`. **Machine-wide and unclamped**: it knows nothing about a job
    /// object's `JOBOBJECT_EXTENDED_LIMIT_INFORMATION` commit limit, so inside a Windows container
    /// it can report far more than this process may actually take. The Linux arm has a clamp for
    /// exactly this reason and this one does not yet.
    WindowsAvailPhys,
}

/// Host memory that could be committed right now, or `None` where this platform has no probe.
///
/// `None` is a real answer: it means "do not auto-size", not "assume zero" and not "assume
/// plenty". A caller that cannot size stays on its conservative path and says so.
///
/// **Linux** reads `MemAvailable` from `/proc/meminfo` — the kernel's own estimate of what a new
/// allocation can have without swapping, already accounting for reclaimable page cache, so it is
/// the figure an arena wants and not something derivable from `MemTotal`. A cgroup memory limit
/// then overrides it: `/proc/meminfo` is host-wide and knows nothing about the cap a container or
/// a `systemd-run --scope -p MemoryMax=` puts on this process — measured on the dev box, an 8 GB
/// scope still reports 54.6 GB available — so the smaller of the two wins.
///
/// **Windows** reads `ullAvailPhys` from `GlobalMemoryStatusEx`, with no equivalent clamp.
///
/// **macOS and the BSDs have no probe at all** and answer `None`. That is a real gap rather than a
/// spelling difference: anything that sizes an arena from this figure is running unbudgeted there
/// today. Closing it needs `host_statistics64`'s free/inactive/purgeable split, which is its own
/// piece of work and not a translation of either arm above.
pub fn available() -> Option<Available> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let host = parse_mem_available(&text)?;
        Some(match cgroup_headroom() {
            Some(limited) if limited < host => Available {
                bytes: limited,
                source: Source::LinuxCgroupClamped,
            },
            _ => Available {
                bytes: host,
                source: Source::LinuxMemAvailable,
            },
        })
    }
    #[cfg(windows)]
    {
        Some(Available {
            bytes: windows_memory_status()?.ullAvailPhys,
            source: Source::WindowsAvailPhys,
        })
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

#[cfg(windows)]
fn windows_memory_status() -> Option<windows::Win32::System::SystemInformation::MEMORYSTATUSEX> {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    // `dwLength` is an in-parameter: `GlobalMemoryStatusEx` rejects a struct that does not
    // announce its own size, so it cannot be left at `Default`'s zero.
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    unsafe { GlobalMemoryStatusEx(&mut status).ok()? };
    Some(status)
}

/// Memory this process may still commit before its cgroup kills it, or `None` when no ancestor
/// limits it.
///
/// Walks from the process's own cgroup up to the root, because the binding limit is the TIGHTEST
/// of the ancestors and not necessarily the leaf's — a container's leaf is often unlimited while
/// the pod slice above it is capped. Both hierarchy versions are read: v2's `memory.max` /
/// `memory.current`, and v1's `memory.limit_in_bytes` / `memory.usage_in_bytes`, whose "no limit"
/// is a sentinel near `u64::MAX` rather than a word.
#[cfg(target_os = "linux")]
fn cgroup_headroom() -> Option<u64> {
    let own = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mut tightest: Option<u64> = None;
    for line in own.lines() {
        // v2: `0::/a/b`. v1: `N:memory:/a/b` (other controllers are not ours to read).
        let mut parts = line.splitn(3, ':');
        let hier = parts.next()?;
        let ctrl = parts.next()?;
        let path = parts.next()?;
        let (root, max_file, cur_file) = if hier == "0" && ctrl.is_empty() {
            ("/sys/fs/cgroup", "memory.max", "memory.current")
        } else if ctrl.split(',').any(|c| c == "memory") {
            (
                "/sys/fs/cgroup/memory",
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
            )
        } else {
            continue;
        };
        // From the leaf upward: `/a/b`, `/a`, `/`.
        let mut at = std::path::PathBuf::from(root);
        at.push(path.trim_start_matches('/'));
        loop {
            let max = read_u64(&at.join(max_file));
            let cur = read_u64(&at.join(cur_file));
            if let (Some(max), Some(cur)) = (max, cur) {
                // v1 spells "unlimited" as a huge number; treat anything past the host's plausible
                // range as no limit rather than as headroom nobody has.
                if max < u64::MAX / 2 {
                    let free = max.saturating_sub(cur);
                    tightest = Some(tightest.map_or(free, |t: u64| t.min(free)));
                }
            }
            if at.as_os_str().len() <= root.len() || !at.pop() {
                break;
            }
        }
    }
    tightest
}

/// One cgroup value file: a decimal, or `None` for the `max` sentinel, a missing file, or junk.
#[cfg(target_os = "linux")]
fn read_u64(path: &std::path::Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Pull `MemAvailable` (in kB, as `/proc/meminfo` always reports it) out of the file's text.
///
/// Split from the read so the parse is testable against a literal — the one machine this runs on
/// cannot produce a file with the field missing, which is the case worth checking.
#[cfg(any(target_os = "linux", test))]
fn parse_mem_available(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    // `MemAvailable:   12345678 kB`
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mem_available_in_kb() {
        let text = "MemTotal:       65780000 kB\nMemFree:         2000000 kB\n\
                    MemAvailable:   43000000 kB\nBuffers:          100000 kB\n";
        assert_eq!(parse_mem_available(text), Some(43_000_000 * 1024));
    }

    /// A kernel too old to report `MemAvailable` must produce `None`, not a figure derived from
    /// `MemTotal` — auto-sizing against total memory would commit the page cache's share too.
    #[test]
    fn a_file_without_the_field_is_unknown() {
        let text = "MemTotal:       65780000 kB\nMemFree:         2000000 kB\n";
        assert_eq!(parse_mem_available(text), None);
    }

    #[test]
    fn a_malformed_field_is_unknown() {
        assert_eq!(parse_mem_available("MemAvailable:   plenty kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable:\n"), None);
    }

    /// The probe must agree with itself on the machine running the tests: a plausible, non-zero
    /// figure no larger than what the same file reports as total.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_live_probe_is_plausible() {
        let avail = available()
            .map(|a| a.bytes)
            .expect("linux always has /proc/meminfo");
        let text = std::fs::read_to_string("/proc/meminfo").expect("meminfo");
        let total: u64 = text
            .lines()
            .find(|l| l.starts_with("MemTotal:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .expect("MemTotal")
            * 1024;
        assert!(avail > 0, "available must be non-zero");
        assert!(avail <= total, "available {avail} exceeds total {total}");
    }

    #[cfg(windows)]
    #[test]
    fn the_windows_live_probe_is_plausible() {
        let avail = available()
            .map(|a| a.bytes)
            .expect("windows GlobalMemoryStatusEx should answer");
        let status = windows_memory_status().expect("GlobalMemoryStatusEx");
        assert!(
            status.ullTotalPhys > 0,
            "total physical memory must be non-zero"
        );
        assert!(
            avail <= status.ullTotalPhys,
            "available {avail} exceeds total {}",
            status.ullTotalPhys
        );
    }

    /// The provenance must be a real answer rather than a constant: on a machine with no cgroup
    /// cap the figure is the unclamped host one, and it agrees with what the file says.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_source_matches_the_figure() {
        let got = available().expect("linux always has /proc/meminfo");
        let host = parse_mem_available(&std::fs::read_to_string("/proc/meminfo").expect("meminfo"))
            .expect("MemAvailable");
        match got.source {
            // Unclamped: the reported figure IS the host figure, not something else.
            Source::LinuxMemAvailable => assert_eq!(got.bytes, host),
            // Clamped: the clamp only ever binds downward.
            Source::LinuxCgroupClamped => assert!(
                got.bytes < host,
                "a clamp that did not reduce the figure must not be reported as one: \
                 {} vs host {host}",
                got.bytes
            ),
            other => panic!("linux must not report {other:?}"),
        }
    }

    /// Every platform without a probe says so, rather than inventing a number.
    #[cfg(not(any(target_os = "linux", windows)))]
    #[test]
    fn a_platform_without_a_probe_answers_none() {
        assert_eq!(available(), None);
    }
}
