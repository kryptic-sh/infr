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
    /// macOS `host_statistics64`'s VM info, converted to a reclaimable-without-swapping page count
    /// (`free_count - speculative_count + inactive_count + purgeable_count`) and then to bytes.
    /// Host-wide: macOS has no per-process commit limit analogous to a cgroup or Job Object.
    MacosVmStatistics64,
    /// Windows `ullAvailPhys`, with no Job Object limit binding this process. Machine-wide: it
    /// knows nothing about a job object's `JOBOBJECT_EXTENDED_LIMIT_INFORMATION` commit limit, so
    /// inside a Windows container it can report far more than this process may actually take.
    WindowsAvailPhys,
    /// Windows `ullAvailPhys` clamped by the tighter of the current process's Job Object
    /// `JobMemoryLimit` / `ProcessMemoryLimit` — the figure an arena must be sized from inside a
    /// Windows container.
    WindowsJobObjectClamped,
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
/// **Windows** reads `ullAvailPhys` from `GlobalMemoryStatusEx`, then clamps it the same way: a
/// process running inside a Job Object (as a Windows container does) has its `JobMemoryLimit` /
/// `ProcessMemoryLimit` read via `QueryInformationJobObject`, and the tighter of the two wins over
/// the machine-wide figure when it binds — the identical "smaller of the two, and say so" policy
/// as the Linux clamp above, sharing its implementation (`apply_limit_clamp`) rather than
/// duplicating it.
///
/// **macOS** reads `host_statistics64`'s VM info and reports what is reclaimable without swapping:
/// `free_count - speculative_count + inactive_count + purgeable_count`, converted to bytes by
/// `vm_page_size`. `free_count` already INCLUDES speculative pages, so it is subtracted back out
/// rather than added a second time. `compressor_page_count` and wired/active pages are deliberately
/// excluded: reclaiming the former costs CPU decompressing it, and the latter is in use.
///
/// **The BSDs still have no probe at all** and answer `None` — the same conservative "do not
/// auto-size" as before.
pub fn available() -> Option<Available> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        Some(apply_cgroup_clamp(
            parse_mem_available(&text)?,
            cgroup_headroom(),
        ))
    }
    #[cfg(target_os = "macos")]
    {
        let vm = macos_vm_statistics64()?;
        Some(Available {
            bytes: macos_available_bytes(
                vm.free_count as u64,
                vm.speculative_count as u64,
                vm.inactive_count as u64,
                vm.purgeable_count as u64,
                macos_page_size(),
            ),
            source: Source::MacosVmStatistics64,
        })
    }
    #[cfg(windows)]
    {
        Some(apply_job_object_clamp(
            windows_memory_status()?.ullAvailPhys,
            windows_job_memory_limit(),
        ))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        None
    }
}

/// Pick between a host figure and a container's headroom, and say which was taken.
///
/// Shared by the Linux cgroup clamp and the Windows Job Object clamp: both are the same policy —
/// "a limit below the host figure wins, and only a limit that actually binds counts as a clamp" —
/// and writing it twice is how the two copies drift. Split out and pure because the alternative is
/// comparing two separate reads of a live host figure (`MemAvailable` moves between reads), so an
/// equality assertion on it fails by a page at random. Here every case is reachable from a literal.
#[cfg(any(target_os = "linux", windows, test))]
fn apply_limit_clamp(
    host: u64,
    limit: Option<u64>,
    host_source: Source,
    limited_source: Source,
) -> Available {
    // Only a limit that actually BINDS is reported as one — otherwise the clamped source would
    // appear whenever a limit merely exists, which is always, and stop meaning anything.
    match limit {
        Some(limited) if limited < host => Available {
            bytes: limited,
            source: limited_source,
        },
        _ => Available {
            bytes: host,
            source: host_source,
        },
    }
}

/// The Linux cgroup clamp: [`apply_limit_clamp`] parameterised with the Linux [`Source`] pair.
#[cfg(any(target_os = "linux", test))]
fn apply_cgroup_clamp(host: u64, cgroup: Option<u64>) -> Available {
    apply_limit_clamp(
        host,
        cgroup,
        Source::LinuxMemAvailable,
        Source::LinuxCgroupClamped,
    )
}

/// The Windows Job Object clamp: [`apply_limit_clamp`] parameterised with the Windows [`Source`]
/// pair.
#[cfg(any(windows, test))]
fn apply_job_object_clamp(host: u64, job_limit: Option<u64>) -> Available {
    apply_limit_clamp(
        host,
        job_limit,
        Source::WindowsAvailPhys,
        Source::WindowsJobObjectClamped,
    )
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

/// The current process's Job Object memory limit in bytes, or `None` when it is not bound by one.
///
/// Passes a null job handle, which asks `QueryInformationJobObject` for the job the CURRENT
/// process belongs to rather than naming one — there is no other job this process could mean here.
/// The call itself fails (and this returns `None`) for a process not associated with any job,
/// which is the correct fallback: no limit binds it, so the unclamped host figure stands.
#[cfg(windows)]
fn windows_job_memory_limit() -> Option<u64> {
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::JobObjects::{
        JobObjectExtendedLimitInformation, QueryInformationJobObject,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    };

    let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    let len = std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32;
    // SAFETY: `info` is a valid, correctly-sized buffer for the call's duration; `len` matches its
    // actual size.
    unsafe {
        QueryInformationJobObject(
            HANDLE::default(),
            JobObjectExtendedLimitInformation,
            (&mut info as *mut JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            len,
            None,
        )
        .ok()?
    };
    job_object_limit_bytes(
        info.BasicLimitInformation.LimitFlags.0,
        info.JobMemoryLimit as u64,
        info.ProcessMemoryLimit as u64,
    )
}

/// Pick the binding memory limit out of a Job Object's extended limit info, or `None` if neither
/// applies.
///
/// A limit field is only meaningful when its bit is set in `LimitFlags` — reading it unconditionally
/// produces a garbage clamp from a zeroed-but-unset field. Both `JobMemoryLimit` and
/// `ProcessMemoryLimit` can be set at once (a process limit inside a looser job limit, say), so the
/// binding one is the TIGHTER of whichever are actually set — the same "tightest ancestor wins" rule
/// `cgroup_headroom` applies to the cgroup hierarchy. The two flag values (512, 256) are fixed by
/// the Win32 ABI as `JOB_OBJECT_LIMIT_JOB_MEMORY` / `JOB_OBJECT_LIMIT_PROCESS_MEMORY`; they are
/// re-declared here rather than imported because this function must also compile under `cfg(test)`
/// on non-Windows targets, where the `windows` crate is not a dependency.
#[cfg(any(windows, test))]
fn job_object_limit_bytes(
    limit_flags: u32,
    job_memory_limit: u64,
    process_memory_limit: u64,
) -> Option<u64> {
    const JOB_OBJECT_LIMIT_JOB_MEMORY: u32 = 512;
    const JOB_OBJECT_LIMIT_PROCESS_MEMORY: u32 = 256;

    let job = (limit_flags & JOB_OBJECT_LIMIT_JOB_MEMORY != 0).then_some(job_memory_limit);
    let process =
        (limit_flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY != 0).then_some(process_memory_limit);
    match (job, process) {
        (Some(j), Some(p)) => Some(j.min(p)),
        (Some(j), None) => Some(j),
        (None, Some(p)) => Some(p),
        (None, None) => None,
    }
}

/// Read macOS's VM statistics via `host_statistics64`, or `None` if the kernel call fails.
///
/// `mach_host_self()` returns a send right to the host port that is valid for the life of the
/// process, so there is nothing to release. `HOST_VM_INFO64_COUNT` is the field count
/// `host_statistics64` expects for the `HOST_VM_INFO64` flavor — passing anything else is how this
/// call fails or truncates.
#[cfg(target_os = "macos")]
fn macos_vm_statistics64() -> Option<libc::vm_statistics64> {
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let mut stats = std::mem::MaybeUninit::<libc::vm_statistics64>::uninit();
    // SAFETY: `stats` is sized for exactly `HOST_VM_INFO64_COUNT` `natural_t`s, which is what
    // `host_statistics64` writes for the `HOST_VM_INFO64` flavor; `count` is passed by pointer as
    // the API requires but this call does not resize the output on success.
    let kr = unsafe {
        libc::host_statistics64(
            libc::mach_host_self(),
            libc::HOST_VM_INFO64,
            stats.as_mut_ptr().cast(),
            &mut count,
        )
    };
    if kr != libc::KERN_SUCCESS {
        return None;
    }
    // SAFETY: `kr == KERN_SUCCESS` means the kernel filled in every field of `stats`.
    Some(unsafe { stats.assume_init() })
}

/// The host's page size in bytes.
#[cfg(target_os = "macos")]
fn macos_page_size() -> u64 {
    // SAFETY: `vm_page_size` is set once by the kernel before `main` runs and never changes for
    // the life of the process; reading it is not a data race with anything.
    (unsafe { libc::vm_page_size }) as u64
}

/// Reclaimable-without-swapping bytes from macOS VM statistics.
///
/// `free_count` on macOS already INCLUDES speculative pages (read ahead but not yet touched), so
/// adding `speculative_count` on top would double-count it — it is subtracted back out instead.
/// `inactive_count` (clean, evictable) and `purgeable_count` (the owner already said it may vanish)
/// are added because both can be reclaimed without writing anything back. Deliberately excluded:
/// `compressor_page_count` (reclaiming it costs CPU decompressing, not just freeing) and
/// wired/active pages (in use). Saturating throughout: a `natural_t` well below `u64::MAX` cannot
/// overflow the adds, but the final multiply by `page_size` can, and a probe should report "a lot"
/// rather than panic when it does.
#[cfg(any(target_os = "macos", test))]
fn macos_available_bytes(
    free_count: u64,
    speculative_count: u64,
    inactive_count: u64,
    purgeable_count: u64,
    page_size: u64,
) -> u64 {
    let pages = free_count
        .saturating_sub(speculative_count)
        .saturating_add(inactive_count)
        .saturating_add(purgeable_count);
    pages.saturating_mul(page_size)
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

    /// Every branch of the clamp, from literals — including the two that are easy to get wrong:
    /// a limit that does not bind must NOT be reported as a clamp, and an equal one is not a
    /// clamp either.
    #[test]
    fn a_cgroup_limit_is_reported_only_when_it_binds() {
        const HOST: u64 = 8 << 30;

        assert_eq!(
            apply_cgroup_clamp(HOST, None),
            Available {
                bytes: HOST,
                source: Source::LinuxMemAvailable
            }
        );
        assert_eq!(
            apply_cgroup_clamp(HOST, Some(2 << 30)),
            Available {
                bytes: 2 << 30,
                source: Source::LinuxCgroupClamped
            },
            "a binding limit wins, and says so"
        );
        assert_eq!(
            apply_cgroup_clamp(HOST, Some(64 << 30)),
            Available {
                bytes: HOST,
                source: Source::LinuxMemAvailable
            },
            "a limit above the host figure is not a clamp"
        );
        assert_eq!(
            apply_cgroup_clamp(HOST, Some(HOST)),
            Available {
                bytes: HOST,
                source: Source::LinuxMemAvailable
            },
            "an equal limit takes nothing away, so it is not a clamp either"
        );
    }

    /// The live probe reports one of the Linux sources — not, say, the Windows one — and its
    /// figure is the one its own source claims. Deliberately no equality against a second read of
    /// `/proc/meminfo`: that number moves between reads.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_live_probe_reports_a_linux_source() {
        let got = available().expect("linux always has /proc/meminfo");
        assert!(
            matches!(
                got.source,
                Source::LinuxMemAvailable | Source::LinuxCgroupClamped
            ),
            "linux must not report {:?}",
            got.source
        );
        assert!(got.bytes > 0);
    }

    /// Every platform without a probe says so, rather than inventing a number.
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    #[test]
    fn a_platform_without_a_probe_answers_none() {
        assert_eq!(available(), None);
    }

    /// The probe must agree with itself on the machine running the tests: a plausible, non-zero
    /// figure, reported through the macOS source.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_macos_live_probe_is_plausible() {
        let got = available().expect("macos host_statistics64 should answer");
        assert_eq!(got.source, Source::MacosVmStatistics64);
        assert!(got.bytes > 0, "available must be non-zero");
    }

    /// Every branch of the shared clamp, from literals, exercised through the Job Object wrapper —
    /// the arithmetic itself is [`a_cgroup_limit_is_reported_only_when_it_binds`]'s job; this just
    /// proves the Windows wrapper reports the Windows [`Source`] pair rather than the Linux one.
    #[test]
    fn a_job_object_limit_is_reported_only_when_it_binds() {
        const HOST: u64 = 8 << 30;

        assert_eq!(
            apply_job_object_clamp(HOST, None),
            Available {
                bytes: HOST,
                source: Source::WindowsAvailPhys
            }
        );
        assert_eq!(
            apply_job_object_clamp(HOST, Some(2 << 30)),
            Available {
                bytes: 2 << 30,
                source: Source::WindowsJobObjectClamped
            },
            "a binding job limit wins, and says so"
        );
        assert_eq!(
            apply_job_object_clamp(HOST, Some(64 << 30)),
            Available {
                bytes: HOST,
                source: Source::WindowsAvailPhys
            },
            "a limit above the host figure is not a clamp"
        );
    }

    /// `job_object_limit_bytes` re-declares the two flag bits so it can compile under `cfg(test)`
    /// where the `windows` crate is not a dependency — which means two copies of a value, and the
    /// copy this crate reads is not the one the ABI hands it. Pin them together where the real
    /// ones exist. Windows-only by necessity, so it is the Windows CI leg that enforces this.
    #[test]
    #[cfg(windows)]
    fn the_re_declared_flag_bits_match_the_windows_crate() {
        use windows::Win32::System::JobObjects::{
            JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
        };
        // Both limits set, `job` the tighter: it can only come back as `job` if the bit the
        // function tests is the bit the ABI actually sets.
        assert_eq!(
            job_object_limit_bytes(
                JOB_OBJECT_LIMIT_JOB_MEMORY.0 | JOB_OBJECT_LIMIT_PROCESS_MEMORY.0,
                1 << 30,
                2 << 30,
            ),
            Some(1 << 30),
        );
        // And each alone selects its own field, which a wrong bit value could not do.
        assert_eq!(
            job_object_limit_bytes(JOB_OBJECT_LIMIT_JOB_MEMORY.0, 1 << 30, 2 << 30),
            Some(1 << 30),
        );
        assert_eq!(
            job_object_limit_bytes(JOB_OBJECT_LIMIT_PROCESS_MEMORY.0, 1 << 30, 2 << 30),
            Some(2 << 30),
        );
    }

    /// `JOB_OBJECT_LIMIT_JOB_MEMORY` is bit 512, `JOB_OBJECT_LIMIT_PROCESS_MEMORY` is bit 256 —
    /// every combination of set/unset, and which one wins when both are set.
    #[test]
    fn job_object_limit_bytes_honors_only_set_flags_and_takes_the_tighter() {
        const JOB: u32 = 512;
        const PROCESS: u32 = 256;

        assert_eq!(
            job_object_limit_bytes(0, 1 << 30, 2 << 30),
            None,
            "neither flag set"
        );
        assert_eq!(
            job_object_limit_bytes(JOB, 1 << 30, 2 << 30),
            Some(1 << 30),
            "only the job flag set"
        );
        assert_eq!(
            job_object_limit_bytes(PROCESS, 1 << 30, 2 << 30),
            Some(2 << 30),
            "only the process flag set"
        );
        assert_eq!(
            job_object_limit_bytes(JOB | PROCESS, 1 << 30, 2 << 30),
            Some(1 << 30),
            "both set, job is tighter"
        );
        assert_eq!(
            job_object_limit_bytes(JOB | PROCESS, 4 << 30, 2 << 30),
            Some(2 << 30),
            "both set, process is tighter"
        );
    }

    /// `free_count` already includes speculative pages, so it is subtracted rather than added
    /// again; inactive and purgeable pages are added; the multiply saturates instead of wrapping.
    #[test]
    fn macos_available_bytes_matches_the_documented_formula() {
        let page_size = 16 << 10; // Apple Silicon's page size.

        assert_eq!(
            macos_available_bytes(1000, 200, 300, 50, page_size),
            (1000 - 200 + 300 + 50) * page_size,
            "free minus speculative plus inactive plus purgeable"
        );
        assert_eq!(
            macos_available_bytes(0, 0, 0, 0, page_size),
            0,
            "an idle-empty report is zero, not a panic"
        );
        assert_eq!(
            macos_available_bytes(u64::MAX, 0, u64::MAX, u64::MAX, u64::MAX),
            u64::MAX,
            "the page-count sum and the final multiply both saturate instead of wrapping"
        );
    }
}
