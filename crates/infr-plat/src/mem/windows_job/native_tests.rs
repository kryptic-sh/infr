use std::os::windows::io::{FromRawHandle, OwnedHandle};
use std::os::windows::process::CommandExt;
use std::process::Command;
use std::time::{Duration, Instant};
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT,
    JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
};
use windows::Win32::System::Threading::{GetCurrentProcess, CREATE_NO_WINDOW};

const CHILD_MODE: &str = "TEST_INFR_PLAT_MEMORY_JOB_CHILD";
const LIMIT: usize = 256 << 20;
const ALLOCATION: usize = 16 << 20;

#[test]
fn process_limit_tracks_retained_commit() {
    run_in_child(
        "process_limit_tracks_retained_commit",
        JOB_OBJECT_LIMIT_PROCESS_MEMORY,
    );
}

#[test]
fn job_limit_tracks_retained_commit() {
    run_in_child(
        "job_limit_tracks_retained_commit",
        JOB_OBJECT_LIMIT_JOB_MEMORY,
    );
}

fn run_in_child(test: &str, flags: JOB_OBJECT_LIMIT) {
    if std::env::var(CHILD_MODE).as_deref() == Ok(test) {
        // SAFETY: no attributes or name pointers; the returned owned handle is transferred
        // exactly once to OwnedHandle and kept alive while the job is configured and queried.
        let job = unsafe { CreateJobObjectW(None, windows::core::PCWSTR::null()) }.unwrap();
        let _owner = unsafe { OwnedHandle::from_raw_handle(job.0) };
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = flags;
        info.JobMemoryLimit = LIMIT;
        info.ProcessMemoryLimit = LIMIT;
        // SAFETY: handle is live, input type/size match the class. Only this isolated child
        // joins the job; the test runner's process and other tests are never assigned.
        unsafe {
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                std::mem::size_of_val(&info) as u32,
            )
            .unwrap();
            AssignProcessToJobObject(job, GetCurrentProcess()).unwrap();
        }
        let before = super::probe()
            .expect("job probe before allocation")
            .expect("job limit");
        let mut retained = vec![0_u8; ALLOCATION];
        // Touch every byte and retain the allocation across the second live probe.
        retained.fill(0xa5);
        std::hint::black_box(&retained);
        let after = super::probe()
            .expect("job probe after allocation")
            .expect("job limit");
        assert!(
            before.saturating_sub(after) >= ALLOCATION as u64,
            "available must fall by the retained commit: before={before:?}, after={after:?}"
        );
        assert_eq!(std::hint::black_box(retained)[ALLOCATION - 1], 0xa5);
        return;
    }

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("mem::windows_job::native_tests::{test}"),
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_MODE, test)
        .creation_flags(CREATE_NO_WINDOW.0)
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                assert!(status.success(), "job child failed: {status}");
                break;
            }
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            result => {
                child.kill().expect("terminate job child");
                child.wait().expect("reap job child");
                panic!("job child timed out or failed to wait: {result:?}");
            }
        }
    }
}
