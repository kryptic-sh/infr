//! Windows job commit headroom. Failed probes are distinct from an absent limit.

// Win32 ABI flag values, also tested against the Windows bindings in the parent module.
const JOB_MEMORY: u32 = 512;
const PROCESS_MEMORY: u32 = 256;

pub(super) fn headroom(
    flags: u32,
    job_limit: u64,
    process_limit: u64,
    job_usage: u64,
    process_usage: u64,
) -> Option<u64> {
    let job = (flags & JOB_MEMORY != 0).then_some(job_limit.saturating_sub(job_usage));
    let process =
        (flags & PROCESS_MEMORY != 0).then_some(process_limit.saturating_sub(process_usage));
    match (job, process) {
        (Some(j), Some(p)) => Some(j.min(p)),
        (Some(j), None) => Some(j),
        (None, Some(p)) => Some(p),
        (None, None) => None,
    }
}

#[cfg(windows)]
pub(super) fn probe() -> windows::core::Result<Option<u64>> {
    use windows::Win32::Foundation::{BOOL, HANDLE};
    use windows::Win32::System::JobObjects::{
        IsProcessInJob, JobObjectExtendedLimitInformation, JobObjectReserved11Information,
        QueryInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    };
    use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    use windows::Win32::System::Threading::GetCurrentProcess;

    let mut in_job = BOOL::default();
    // SAFETY: the pseudo-handle is valid for this process, and the output is live and sized.
    unsafe { IsProcessInJob(GetCurrentProcess(), HANDLE::default(), &mut in_job)? };
    if !in_job.as_bool() {
        return Ok(None);
    }
    let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    // SAFETY: class, output type and byte length agree; the buffer lives through the call.
    unsafe {
        QueryInformationJobObject(
            HANDLE::default(),
            JobObjectExtendedLimitInformation,
            (&mut info as *mut JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            std::mem::size_of_val(&info) as u32,
            None,
        )?;
    }
    let flags = info.BasicLimitInformation.LimitFlags.0;
    let mut job_usage = 0;
    if flags & JOB_MEMORY != 0 {
        // JOBOBJECT_MEMORY_USAGE_INFORMATION from the installed ntapi bindings: ULONG64 fields.
        // The SDK/windows bindings expose its info class as JobObjectReserved11Information (28).
        #[repr(C)]
        #[derive(Default)]
        struct JobMemoryUsage {
            job_memory: u64,
            peak_job_memory_used: u64,
        }
        let mut usage = JobMemoryUsage::default();
        // SAFETY: repr(C) matches the native layout, and the exact buffer size is passed.
        // A rejected/unsupported query propagates unknown, never a raw-limit fallback.
        unsafe {
            QueryInformationJobObject(
                HANDLE::default(),
                JobObjectReserved11Information,
                (&mut usage as *mut JobMemoryUsage).cast(),
                std::mem::size_of_val(&usage) as u32,
                None,
            )?;
        }
        job_usage = usage.job_memory;
    }
    let mut process_usage = 0;
    if flags & PROCESS_MEMORY != 0 {
        let mut counters = PROCESS_MEMORY_COUNTERS_EX {
            cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
            ..Default::default()
        };
        // SAFETY: GetProcessMemoryInfo accepts the extended layout when cb is its full size.
        // The cast preserves alignment and the buffer remains live throughout the call.
        unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                (&mut counters as *mut PROCESS_MEMORY_COUNTERS_EX).cast(),
                counters.cb,
            )?;
        }
        process_usage = counters.PrivateUsage as u64;
    }
    Ok(headroom(
        flags,
        info.JobMemoryLimit as u64,
        info.ProcessMemoryLimit as u64,
        job_usage,
        process_usage,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaining_capacity_not_limit_order_decides_the_clamp() {
        assert_eq!(
            headroom(JOB_MEMORY | PROCESS_MEMORY, 100, 200, 10, 180),
            Some(20)
        );
        assert_eq!(
            headroom(JOB_MEMORY | PROCESS_MEMORY, 200, 100, 180, 10),
            Some(20)
        );
        assert_eq!(headroom(JOB_MEMORY, 100, 0, 0, u64::MAX), Some(100));
        assert_eq!(headroom(PROCESS_MEMORY, 0, 100, u64::MAX, 0), Some(100));
        assert_eq!(headroom(JOB_MEMORY, 100, 200, 100, 0), Some(0));
        assert_eq!(headroom(PROCESS_MEMORY, 100, 200, 0, 201), Some(0));
        assert_eq!(headroom(JOB_MEMORY, 100, 200, 101, 0), Some(0));
        assert_eq!(headroom(PROCESS_MEMORY, 100, 200, 0, 200), Some(0));
        assert_eq!(headroom(JOB_MEMORY, 0, 100, 0, 0), Some(0));
        assert_eq!(headroom(PROCESS_MEMORY, 100, 0, 0, 0), Some(0));
        assert_eq!(headroom(0, 0, 0, u64::MAX, u64::MAX), None);
    }
}

#[cfg(all(test, windows))]
mod native_tests;
