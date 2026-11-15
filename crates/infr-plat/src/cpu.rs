//! Facts about the host CPU that the standard library does not report, and where compute threads
//! should run on it.

/// The number of physical cores this process may run on, or `None` where this platform has no
/// probe here.
///
/// `std::thread::available_parallelism` counts LOGICAL processors, so with SMT it reports two per
/// core. That is the right answer for many things and the wrong one for a spin-waiting compute
/// pool, where two workers on one core's hyperthreads fight over the same execution units: on a
/// Ryzen 9 9950X3D under Windows, the CPU backend decoded Qwen3-0.6B at ~62 tok/s on 32 threads and
/// ~136 tok/s on 16 (measured 2026-09-23).
///
/// **Windows**: the cores (`GetSystemCpuSetInformation`, every processor group) with at least one
/// logical processor inside the process's affinity, so a process started under a restricted
/// affinity counts only the cores it was given.
///
/// **Everywhere else**: `None`, deliberately — not a guess from `available_parallelism`. Callers
/// keep their logical-count default there, and a platform gains a probe here once someone has
/// measured that a physical-core default helps on it (Linux has not been measured; backlog B76).
pub fn physical_cores() -> Option<usize> {
    imp::physical_cores()
}

/// A soft placement for a compute thread: one logical processor per core it may use (see
/// [`spread_placement`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    /// The OS's ids for the chosen logical processors (CPU set ids on Windows).
    ids: Vec<u32>,
}

/// The placement that keeps `threads` compute threads of the calling thread's process off each
/// other's SMT siblings: one logical processor (the first allowed one) of every core the calling
/// thread may run on, with the OS still free to place the threads among them. On a hybrid part,
/// the cores of the fastest efficiency class only, when there are at least `threads` of those.
///
/// It applies only when the threads fit one per core: `None` when `threads` exceeds the allowed
/// cores (then the OS keeps the whole allowed set, SMT siblings included, for them). Never wider
/// than what the thread is already allowed: the process affinity, and the thread's own affinity
/// when that is narrower than its processor group. Also `None` when there is nothing to narrow
/// (every allowed logical processor would be named anyway: no SMT, and no slower class left
/// out) or no way to say it: no probe on this platform, or a process with default CPU sets of
/// its own. Read at the call: an affinity change after it is not picked up by a placement
/// already handed out.
///
/// **Windows**: CPU sets. **Everywhere else**: `None`.
pub fn spread_placement(threads: usize) -> Option<Placement> {
    imp::spread_placement(threads)
}

/// Apply `p` to the calling thread. `Ok(false)` leaves the thread as it was because it already
/// has a placement of its own (someone else's, or an earlier call's that is still in force), so
/// the caller must only [`clear_placement`] after an `Ok(true)`.
///
/// **Windows**: `SetThreadSelectedCpuSets`. The CPU sets are a preference the scheduler applies
/// within the thread's affinity, never over it. **Everywhere else** no `Placement` exists to pass;
/// it reports `Unsupported`.
pub fn apply_placement(p: &Placement) -> std::io::Result<bool> {
    imp::apply_placement(p)
}

/// Remove the calling thread's placement, returning it to the process's default.
pub fn clear_placement() -> std::io::Result<()> {
    imp::clear_placement()
}

/// The calling thread's current placement, if it has one (Windows: its selected CPU sets);
/// `None` everywhere else.
pub fn current_placement() -> Option<Placement> {
    imp::current_placement()
}

/// One logical processor as the OS describes it.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Logical {
    /// The OS's id for it (a CPU set id on Windows).
    id: u32,
    group: u16,
    /// Its index within `group`.
    index: u8,
    /// Its core, group-relative.
    core: u8,
    /// Higher is faster (hybrid parts); all equal elsewhere.
    efficiency: u8,
}

/// The first allowed logical processor of every core that has one, whatever its efficiency
/// class: the cores [`physical_cores`] counts. Empty when nothing is allowed.
#[cfg(any(windows, test))]
fn cores_of(lps: &[Logical], allowed: impl Fn(&Logical) -> bool) -> Vec<Logical> {
    let mut firsts: Vec<Logical> = Vec::new();
    for lp in lps.iter().filter(|lp| allowed(lp)) {
        match firsts
            .iter_mut()
            .find(|f| f.group == lp.group && f.core == lp.core)
        {
            Some(f) if lp.index < f.index => *f = *lp,
            Some(_) => {}
            None => firsts.push(*lp),
        }
    }
    firsts
}

/// The logical processors [`spread_placement`] names for `threads` threads among the `allowed`
/// ones of `lps` (see there), or `None` when it names none.
#[cfg(any(windows, test))]
fn spread(
    lps: &[Logical],
    allowed: impl Fn(&Logical) -> bool,
    threads: usize,
) -> Option<Vec<Logical>> {
    let mut chosen = cores_of(lps, &allowed);
    if threads == 0 || threads > chosen.len() {
        return None;
    }
    let top = chosen.iter().map(|c| c.efficiency).max()?;
    if chosen.iter().filter(|c| c.efficiency == top).count() >= threads {
        chosen.retain(|c| c.efficiency == top);
    }
    let n_allowed = lps.iter().filter(|lp| allowed(lp)).count();
    (chosen.len() < n_allowed).then_some(chosen)
}

/// Parse the `SYSTEM_CPU_SET_INFORMATION` records `GetSystemCpuSetInformation` returns, or `None`
/// if the buffer does not parse. Each record starts with `Size` (`u32`, the whole record) and
/// `Type` (`u32`, 0 for a CPU set), then `Id` (`u32`), `Group` (`u16`), `LogicalProcessorIndex`,
/// `CoreIndex`, `LastLevelCacheIndex`, `NumaNodeIndex` and `EfficiencyClass` (`u8` each).
#[cfg(any(windows, test))]
fn parse_cpu_sets(buf: &[u8]) -> Option<Vec<Logical>> {
    /// The fields read here end at `EfficiencyClass`.
    const MIN: usize = 19;
    let (mut at, mut lps) = (0usize, Vec::new());
    while at < buf.len() {
        let size = u32::from_ne_bytes(buf.get(at..at + 4)?.try_into().ok()?) as usize;
        if size < MIN || at + size > buf.len() {
            return None;
        }
        let r = &buf[at..at + size];
        if u32::from_ne_bytes(r[4..8].try_into().ok()?) == 0 {
            lps.push(Logical {
                id: u32::from_ne_bytes(r[8..12].try_into().ok()?),
                group: u16::from_ne_bytes([r[12], r[13]]),
                index: r[14],
                core: r[15],
                efficiency: r[18],
            });
        }
        at += size;
    }
    (!lps.is_empty()).then_some(lps)
}

#[cfg(windows)]
mod imp {
    use super::{cores_of, parse_cpu_sets, spread, Logical, Placement};
    use windows::Win32::System::SystemInformation::{GetSystemCpuSetInformation, GROUP_AFFINITY};
    use windows::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentThread, GetProcessAffinityMask, GetProcessDefaultCpuSets,
        GetProcessGroupAffinity, GetThreadGroupAffinity, GetThreadSelectedCpuSets,
        SetThreadSelectedCpuSets,
    };

    /// Every logical processor, read once per process: the topology does not change under it.
    fn logical() -> Option<&'static [Logical]> {
        static LPS: std::sync::OnceLock<Option<Vec<Logical>>> = std::sync::OnceLock::new();
        LPS.get_or_init(|| {
            let mut len = 0u32;
            // SAFETY: a size query — no buffer, zero length; the call writes the length needed
            // to `len` and fails with ERROR_INSUFFICIENT_BUFFER by design.
            let _ =
                unsafe { GetSystemCpuSetInformation(None, 0, &mut len, GetCurrentProcess(), 0) };
            if len == 0 {
                return None;
            }
            // `u64` words: the records hold a `u64` field, so the buffer carries its alignment.
            let mut words = vec![0u64; (len as usize).div_ceil(8)];
            // SAFETY: `words` is a live, 8-aligned buffer of at least `len` bytes, and the call
            // is told exactly that; it writes at most `len` bytes and reports what it wrote.
            unsafe {
                GetSystemCpuSetInformation(
                    Some(words.as_mut_ptr().cast()),
                    len,
                    &mut len,
                    GetCurrentProcess(),
                    0,
                )
            }
            .ok()
            .ok()?;
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
            parse_cpu_sets(&bytes[..len as usize])
        })
        .as_deref()
    }

    /// The process's affinity restriction as `(group, mask)`, or `None` when it has none to read:
    /// a process whose threads span several groups reports zero masks (and has no single-group
    /// restriction).
    fn process_mask() -> Option<(u16, usize)> {
        let (mut pm, mut sm) = (0usize, 0usize);
        // SAFETY: the pseudo-handle is always valid; both out-params are live.
        unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut pm, &mut sm) }.ok()?;
        if pm == 0 {
            return None;
        }
        let (mut count, mut groups) = (1u16, [0u16; 1]);
        // SAFETY: `groups` has room for `count` entries; the call fails (and we give up the
        // restriction) if the process spans more.
        unsafe { GetProcessGroupAffinity(GetCurrentProcess(), &mut count, groups.as_mut_ptr()) }
            .ok()
            .ok()?;
        Some((groups[0], pm))
    }

    /// The calling thread's affinity as `(group, mask)`, when it is narrower than its group.
    fn thread_mask(lps: &[Logical]) -> Option<(u16, usize)> {
        let mut aff = GROUP_AFFINITY::default();
        // SAFETY: the pseudo-handle is valid for this thread; `aff` is a live out-param.
        unsafe { GetThreadGroupAffinity(GetCurrentThread(), &mut aff) }
            .ok()
            .ok()?;
        let full = lps
            .iter()
            .filter(|lp| lp.group == aff.Group)
            .fold(0usize, |m, lp| {
                m | 1usize.checked_shl(lp.index as u32).unwrap_or(0)
            });
        (aff.Mask & full != full).then_some((aff.Group, aff.Mask))
    }

    fn within(lp: &Logical, restriction: Option<(u16, usize)>) -> bool {
        restriction.is_none_or(|(g, m)| {
            lp.group == g && (lp.index as u32) < usize::BITS && m & (1usize << lp.index) != 0
        })
    }

    pub(super) fn physical_cores() -> Option<usize> {
        let lps = logical()?;
        let proc = process_mask();
        let n = cores_of(lps, |lp| within(lp, proc)).len();
        (n > 0).then_some(n)
    }

    pub(super) fn spread_placement(threads: usize) -> Option<Placement> {
        let lps = logical()?;
        let mut required = 0u32;
        // SAFETY: a count query on the current process's pseudo-handle.
        let _ = unsafe { GetProcessDefaultCpuSets(GetCurrentProcess(), None, &mut required) };
        if required > 0 {
            return None; // someone chose this process's CPU sets; do not second-guess them
        }
        let (proc, thread) = (process_mask(), thread_mask(lps));
        let chosen = spread(lps, |lp| within(lp, proc) && within(lp, thread), threads)?;
        let mut ids: Vec<u32> = chosen.iter().map(|lp| lp.id).collect();
        ids.sort_unstable();
        Some(Placement { ids })
    }

    pub(super) fn apply_placement(p: &Placement) -> std::io::Result<bool> {
        let mut required = 0u32;
        // SAFETY: a count query on this thread's pseudo-handle. It "fails" with
        // ERROR_INSUFFICIENT_BUFFER when the thread has a selection, which `required` reports.
        let _ = unsafe { GetThreadSelectedCpuSets(GetCurrentThread(), None, &mut required) };
        if required > 0 {
            return Ok(false);
        }
        // SAFETY: the pseudo-handle is valid; `ids` is a live slice of CPU set ids.
        unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), &p.ids) }
            .ok()
            .map_err(std::io::Error::from)?;
        Ok(true)
    }

    pub(super) fn clear_placement() -> std::io::Result<()> {
        // SAFETY: the pseudo-handle is valid; an empty id list clears the selection.
        unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), &[]) }
            .ok()
            .map_err(std::io::Error::from)
    }

    pub(super) fn current_placement() -> Option<Placement> {
        let mut required = 0u32;
        // SAFETY: as in `apply_placement`.
        let _ = unsafe { GetThreadSelectedCpuSets(GetCurrentThread(), None, &mut required) };
        if required == 0 {
            return None;
        }
        let mut ids = vec![0u32; required as usize];
        // SAFETY: `ids` has room for `required` ids; a selection changed by another thread in
        // between would make the call fail, and then there is nothing reliable to report.
        unsafe { GetThreadSelectedCpuSets(GetCurrentThread(), Some(&mut ids), &mut required) }
            .ok()
            .ok()?;
        ids.truncate(required as usize);
        ids.sort_unstable();
        Some(Placement { ids })
    }

    #[cfg(test)]
    pub(super) fn logical_for_tests() -> &'static [Logical] {
        logical().expect("Windows always reports its CPU sets")
    }
}

#[cfg(not(windows))]
mod imp {
    use super::Placement;

    pub(super) fn physical_cores() -> Option<usize> {
        None
    }

    pub(super) fn spread_placement(_threads: usize) -> Option<Placement> {
        None
    }

    pub(super) fn apply_placement(_p: &Placement) -> std::io::Result<bool> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "thread placement has no implementation on this platform",
        ))
    }

    pub(super) fn clear_placement() -> std::io::Result<()> {
        Ok(())
    }

    pub(super) fn current_placement() -> Option<Placement> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `SYSTEM_CPU_SET_INFORMATION` record of `size` bytes for one logical processor.
    fn record(size: u32, lp: Logical) -> Vec<u8> {
        let mut r = vec![0u8; (size as usize).max(32)]; // room for the fields even when `size` lies
        r[0..4].copy_from_slice(&size.to_ne_bytes());
        r[8..12].copy_from_slice(&lp.id.to_ne_bytes());
        r[12..14].copy_from_slice(&lp.group.to_ne_bytes());
        r[14] = lp.index;
        r[15] = lp.core;
        r[18] = lp.efficiency;
        r.truncate((size as usize).max(4));
        r
    }

    fn lp(id: u32, group: u16, index: u8, core: u8, efficiency: u8) -> Logical {
        Logical {
            id,
            group,
            index,
            core,
            efficiency,
        }
    }

    #[test]
    fn parses_variable_length_records() {
        let lps = [
            lp(256, 0, 0, 0, 1),
            lp(257, 0, 1, 0, 1),
            lp(300, 1, 0, 0, 0),
        ];
        let buf = [record(32, lps[0]), record(40, lps[1]), record(32, lps[2])].concat();
        assert_eq!(parse_cpu_sets(&buf), Some(lps.to_vec()));
    }

    #[test]
    fn rejects_a_buffer_that_does_not_parse() {
        let one = lp(1, 0, 0, 0, 0);
        assert_eq!(parse_cpu_sets(&[]), None, "no records");
        assert_eq!(
            parse_cpu_sets(&record(8, one)),
            None,
            "size below the fields"
        );
        let mut overrun = record(32, one);
        overrun.truncate(24);
        assert_eq!(parse_cpu_sets(&overrun), None, "record runs past the end");
    }

    /// Two SMT cores of the fast class, two single-thread cores of the slow one.
    fn hybrid() -> Vec<Logical> {
        vec![
            lp(10, 0, 0, 0, 1),
            lp(11, 0, 1, 0, 1),
            lp(12, 0, 2, 2, 1),
            lp(13, 0, 3, 2, 1),
            lp(14, 0, 4, 4, 0),
            lp(15, 0, 5, 5, 0),
        ]
    }

    fn ids(v: &[Logical]) -> Vec<u32> {
        v.iter().map(|l| l.id).collect()
    }

    /// Every allowed core counts, slow class included: the default thread count must not drop
    /// to the fast cores alone on a hybrid part.
    #[test]
    fn cores_of_counts_every_class() {
        assert_eq!(ids(&cores_of(&hybrid(), |_| true)), [10, 12, 14, 15]);
        // An affinity that leaves only a core's second sibling still gives that core.
        assert_eq!(
            ids(&cores_of(&hybrid(), |l| l.index != 0)),
            [11, 12, 14, 15]
        );
        assert!(cores_of(&hybrid(), |_| false).is_empty());
    }

    #[test]
    fn spread_names_one_logical_per_core_fastest_class_first() {
        let sp = |threads| spread(&hybrid(), |_| true, threads).map(|v| ids(&v));
        assert_eq!(sp(2), Some(vec![10, 12]), "fast cores suffice");
        assert_eq!(sp(3), Some(vec![10, 12, 14, 15]), "they do not");
        assert_eq!(sp(4), Some(vec![10, 12, 14, 15]), "one thread per core");
    }

    /// More threads than allowed cores get no placement, so they keep every allowed logical
    /// processor; so does a request for none.
    #[test]
    fn spread_steps_aside_when_threads_outnumber_cores() {
        assert_eq!(spread(&hybrid(), |_| true, 5), None);
        assert_eq!(
            spread(&hybrid(), |l| l.core != 0, 4),
            None,
            "three cores left"
        );
        assert_eq!(spread(&hybrid(), |_| true, 0), None);
    }

    /// Nothing to narrow: no SMT and every class named. A hybrid without SMT still narrows to
    /// its fast cores when they suffice.
    #[test]
    fn spread_names_nothing_it_cannot_narrow() {
        let no_smt = |l: &Logical| l.index != 1 && l.index != 3;
        assert_eq!(spread(&hybrid(), no_smt, 4), None);
        assert_eq!(
            spread(&hybrid(), no_smt, 2).map(|v| ids(&v)),
            Some(vec![10, 12])
        );
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

    /// A placement holds one logical processor per core; applying it reads back as the thread's
    /// selection, a second apply leaves it alone, and clearing removes it.
    #[cfg(windows)]
    #[test]
    fn placement_applies_and_clears() {
        let lps = imp::logical_for_tests();
        std::thread::spawn(move || {
            let Some(p) = spread_placement(2) else {
                eprintln!(
                    "SKIP placement_applies_and_clears: no placement for two threads on this \
                     machine (fewer than two allowed cores, nothing to narrow, or default CPU \
                     sets)"
                );
                return;
            };
            let chosen: Vec<&Logical> = lps.iter().filter(|l| p.ids.contains(&l.id)).collect();
            assert_eq!(chosen.len(), p.ids.len());
            for (i, a) in chosen.iter().enumerate() {
                for b in &chosen[i + 1..] {
                    assert!(
                        (a.group, a.core) != (b.group, b.core),
                        "{a:?} and {b:?} share a core"
                    );
                }
            }
            assert_eq!(current_placement(), None, "starts with no selection");
            assert!(apply_placement(&p).expect("apply"));
            assert_eq!(current_placement().as_ref(), Some(&p));
            assert!(
                !apply_placement(&p).expect("second apply"),
                "a selection is left alone"
            );
            clear_placement().expect("clear");
            assert_eq!(current_placement(), None, "clearing removes it");
        })
        .join()
        .unwrap();
    }

    /// Under a restricted thread affinity the placement names only processors inside it, one per
    /// core: it never widens a restriction. Two SMT cores of one group inside the process's
    /// affinity are kept; a machine without two such cores skips, saying why.
    #[cfg(windows)]
    #[test]
    fn placement_stays_inside_a_restricted_affinity() {
        use windows::Win32::System::SystemInformation::GROUP_AFFINITY;
        use windows::Win32::System::Threading::{
            GetCurrentProcess, GetCurrentThread, GetProcessAffinityMask, SetThreadGroupAffinity,
        };
        let lps = imp::logical_for_tests();
        let (mut pm, mut sm) = (0usize, 0usize);
        // SAFETY: the pseudo-handle is always valid; both out-params are live.
        let _ = unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut pm, &mut sm) };
        let in_process = |l: &Logical| pm == 0 || (l.index < 64 && pm & (1usize << l.index) != 0);
        // Cores of group 0 whose every logical processor the process may use, with at least two.
        let mut smt: Vec<u8> = Vec::new();
        for l in lps.iter().filter(|l| l.group == 0) {
            let sibs: Vec<&Logical> = lps
                .iter()
                .filter(|o| o.group == 0 && o.core == l.core)
                .collect();
            if sibs.len() >= 2 && sibs.iter().all(|o| in_process(o)) && !smt.contains(&l.core) {
                smt.push(l.core);
            }
        }
        if smt.len() < 2 {
            eprintln!(
                "SKIP placement_stays_inside_a_restricted_affinity: fewer than two SMT cores \
                 in group 0 inside this process's affinity"
            );
            return;
        }
        let keep = [smt[smt.len() - 2], smt[smt.len() - 1]];
        let mask = lps
            .iter()
            .filter(|l| l.group == 0 && keep.contains(&l.core))
            .fold(0usize, |m, l| m | 1usize << l.index);
        std::thread::spawn(move || {
            let aff = GROUP_AFFINITY {
                Mask: mask,
                Group: 0,
                Reserved: [0; 3],
            };
            // SAFETY: the pseudo-handle is valid; `aff` is a live GROUP_AFFINITY.
            unsafe { SetThreadGroupAffinity(GetCurrentThread(), &aff, None) }
                .ok()
                .expect("restrict the thread to cores inside the process's affinity");
            let Some(p) = spread_placement(2) else {
                eprintln!(
                    "SKIP placement_stays_inside_a_restricted_affinity: no placement (this \
                     process has default CPU sets)"
                );
                return;
            };
            assert_eq!(p.ids.len(), 2, "one per kept core");
            for id in &p.ids {
                let l = lps.iter().find(|l| l.id == *id).unwrap();
                assert!(
                    l.group == 0 && mask & (1usize << l.index) != 0,
                    "{l:?} is outside"
                );
            }
        })
        .join()
        .unwrap();
    }
}
