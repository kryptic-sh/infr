//! Bounded host waits on GPU work, and the poisoned state a failed one leaves behind.
//!
//! Every place the backend blocks until the GPU has finished something goes through
//! [`VulkanShared::wait_fences`] or [`VulkanShared::wait_queue_idle`]. Both wait at most
//! `device.gpu_wait_timeout_ms`, and a wait that times out or meets a lost device POISONS the
//! backend: the error names the submit it was waiting on, how long it waited and what the device
//! said about itself, and from then on every wait and submit returns that same error at once
//! instead of touching the device.
//!
//! Why a timeout is fatal rather than retried: a fence that has not signalled after tens of
//! seconds belongs to work that will not finish, whether the GPU hung or a driver reset left it
//! unsignalled. (On the integrated GPU of plan 12 section 0c the reset DID signal the fence, and
//! the host then spun forever on a timestamp read instead; `recorder::account_segment` covers
//! that case, for forward segments only.) Whatever that submit was doing to its
//! buffers is unknown, so nothing it touched can be trusted, and nothing it USES may be destroyed
//! either: destroying a command buffer, a descriptor pool or memory still in use by a device that
//! is merely slow is undefined behaviour. So a poisoned backend LEAKS instead of freeing — every
//! drop path asks [`VulkanShared::poisoned`] first — and the process is expected to report the
//! error and exit, which hands the whole context back to the driver.
//!
//! [`vk::Queue`] waits (`vkQueueWaitIdle`, `vkDeviceWaitIdle`) take no timeout at all. A path
//! that submits work and waits for it ([`VulkanShared::submit_wait`]: a forward's last command
//! buffer, a decode replay, a one-shot copy) attaches a fence to that submit and waits on it; the
//! few that only need the queue drained ([`VulkanShared::wait_queue_idle`]: teardown,
//! `Backend::sync`, freeing a recording) submit no batches with that fence, which signals once
//! everything submitted before it has completed.

use ash::vk;
use infr_core::error::{backend as be, Error, Result};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::VulkanShared;

/// The default bound on one host wait for GPU work, in milliseconds (`device.gpu_wait_timeout_ms`).
///
/// It must sit far above the longest LEGITIMATE wait and still end a stuck one in human time. The
/// two mistakes are not alike: too low a bound poisons the backend on work that would have
/// completed, too high a one only delays an error in a rare case. One wait can cover several
/// segments (the pipelined window of `adapter::execute_static`), and a heavy chunk on a slow
/// device can run seconds per segment, so the bound is set well above the longest waits
/// measured, which `docs/config.md` records beside the key with their workloads.
pub const DEFAULT_GPU_WAIT_TIMEOUT_MS: u64 = 120_000;

/// The smallest bound `device.gpu_wait_timeout_ms` accepts (`0`, wait forever, aside). A guard
/// against a typo such as a value meant in seconds, NOT a safe value: legitimate waits longer than
/// this have been measured (see [`DEFAULT_GPU_WAIT_TIMEOUT_MS`]).
pub const MIN_GPU_WAIT_TIMEOUT_MS: u64 = 5_000;

/// The configured bound, checked: unset is [`DEFAULT_GPU_WAIT_TIMEOUT_MS`], `0` waits forever,
/// and anything else must be at least [`MIN_GPU_WAIT_TIMEOUT_MS`].
pub(crate) fn checked_timeout_ms(configured: Option<u64>) -> std::result::Result<u64, String> {
    match configured {
        None => Ok(DEFAULT_GPU_WAIT_TIMEOUT_MS),
        Some(ms) if ms == 0 || ms >= MIN_GPU_WAIT_TIMEOUT_MS => Ok(ms),
        Some(ms) => Err(format!(
            "device.gpu_wait_timeout_ms (INFR_GPU_WAIT_TIMEOUT_MS) = {ms} is below the {} ms \
             minimum; 0 waits forever",
            MIN_GPU_WAIT_TIMEOUT_MS
        )),
    }
}

/// The wait state one device owns: its bound, its poison, the fence queue-idle waits reuse, and
/// the `VK_EXT_device_fault` loader when the device offers it.
pub(crate) struct GpuWait {
    /// The bound in nanoseconds; `u64::MAX` = wait forever (`device.gpu_wait_timeout_ms = 0`).
    /// Fixed at creation except by the unit tests' hook (`VulkanShared::set_gpu_wait_timeout_ms`).
    timeout_ns: AtomicU64,
    /// The first fatal wait's message. Set once; every later wait and submit returns it.
    poison: OnceLock<String>,
    /// The fence [`VulkanShared::submit_wait`] attaches to a submit and waits on. Its lock is held
    /// from that submit to the fence's reset, so two waits never share it.
    idle_fence: Mutex<vk::Fence>,
    /// `vkGetDeviceFaultInfoEXT`, when the device offers `VK_EXT_device_fault` and the feature
    /// was enabled at creation. Only meaningful after the device has been lost.
    fault: Option<ash::ext::device_fault::Device>,
    /// The forward segments submitted last, oldest first, at most [`RECENT_SEGMENTS`]: what a
    /// failure reports beside its own submit (see [`RecentSegment`]).
    recent: Mutex<VecDeque<RecentSegment>>,
    /// The id the next remembered segment gets.
    next_segment: AtomicU64,
}

/// How many forward segments a failure's report can name.
const RECENT_SEGMENTS: usize = 3;

/// One forward segment as a later failure reports it. A driver can let a submit whose work it
/// cut short run to its end and signal its fence, and report the lost device only at the NEXT
/// submit (RADV after amdgpu's soft recovery, issue 94: a prefill forward was cancelled after
/// about 2.4 s and the error came from the one-shot copy after it). That error names the copy, so
/// it also lists these: the loss most likely happened in the newest of them.
#[derive(Clone, Copy, Debug)]
struct RecentSegment {
    id: u64,
    desc: SubmitDesc,
    submitted: Instant,
    /// The GPU time between its two timestamps, once it has completed and was timed.
    gpu_ns: Option<u64>,
}

/// The clause a failure appends about the forward segments submitted last, newest first: each
/// one's dispatches and kernels, how long before the failure it was submitted and, when it was
/// timed, how long it ran. A segment whose own wait failed is among them. Empty with no segment.
/// Pure, so the wording is testable without a GPU.
fn recent_report(newest_first: &[(SubmitDesc, Duration, Option<u64>)]) -> String {
    if newest_first.is_empty() {
        return String::new();
    }
    let each: Vec<String> = newest_first
        .iter()
        .map(|&(desc, ago, gpu_ns)| {
            let ran = match gpu_ns {
                Some(ns) => format!(
                    ", ran {} on the GPU",
                    fmt_duration(Duration::from_nanos(ns))
                ),
                None => ", not seen to complete".to_string(),
            };
            format!("{desc} submitted {} earlier{ran}", fmt_duration(ago))
        })
        .collect();
    format!(
        "; the last forward segments submitted, newest first (a lost device is most likely \
         the newest one's doing): {}",
        each.join("; ")
    )
}

impl GpuWait {
    /// `timeout_ms == 0` waits forever (the debugging escape hatch).
    pub(crate) fn new(
        device: &ash::Device,
        timeout_ms: u64,
        fault: Option<ash::ext::device_fault::Device>,
    ) -> Result<Self> {
        let idle_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| be(format!("create the queue-idle fence: {e}")))?;
        Ok(Self {
            timeout_ns: AtomicU64::new(timeout_ns(timeout_ms)),
            poison: OnceLock::new(),
            idle_fence: Mutex::new(idle_fence),
            fault,
            recent: Mutex::new(VecDeque::with_capacity(RECENT_SEGMENTS)),
            next_segment: AtomicU64::new(0),
        })
    }

    /// Destroy the queue-idle fence. Only on a device that is not poisoned (see the module doc).
    pub(crate) fn destroy(&self, device: &ash::Device) {
        let fence = *self
            .idle_fence
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        unsafe { device.destroy_fence(fence, None) };
    }
}

/// The configured milliseconds as the nanoseconds `vkWaitForFences` takes; `0` = forever.
fn timeout_ns(timeout_ms: u64) -> u64 {
    if timeout_ms == 0 {
        u64::MAX
    } else {
        timeout_ms.saturating_mul(1_000_000)
    }
}

/// What a wait was waiting on, for the error: the submit's dispatch count and its first and last
/// kernel. Built only when a wait fails.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SubmitDesc {
    /// Dispatches in the submit (or in every submit the wait covers).
    pub(crate) dispatches: usize,
    /// The first kernel recorded into it, `None` when it holds no dispatch.
    pub(crate) first: Option<&'static str>,
    /// The last kernel recorded into it.
    pub(crate) last: Option<&'static str>,
}

impl std::fmt::Display for SubmitDesc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = if self.dispatches == 1 { "" } else { "es" };
        write!(f, "{} dispatch{s}", self.dispatches)?;
        match (self.first, self.last) {
            (Some(a), Some(b)) if a == b && self.dispatches == 1 => write!(f, " (`{a}`)"),
            (Some(a), Some(b)) => write!(f, " (first `{a}`, last `{b}`)"),
            _ => Ok(()),
        }
    }
}

/// How a failure reads in the error: a wait on `what` that returned `result` after `waited`, or with
/// `None` the action `what` itself (a submit) returning it. Pure, so the wording of each case is
/// testable without a GPU.
fn failure_headline(
    result: vk::Result,
    waited: Option<Duration>,
    bound_ns: u64,
    what: &str,
) -> String {
    let Some(waited) = waited else {
        return match result {
            vk::Result::ERROR_DEVICE_LOST => {
                format!("the GPU device was lost: {what} failed")
            }
            other => format!("{what} failed: {other}"),
        };
    };
    let waited = fmt_duration(waited);
    match result {
        vk::Result::TIMEOUT => format!(
            "GPU work did not complete: waited {waited} for {what}, past the {} bound \
             (`device.gpu_wait_timeout_ms` / INFR_GPU_WAIT_TIMEOUT_MS; 0 waits forever)",
            fmt_duration(Duration::from_nanos(bound_ns)),
        ),
        vk::Result::ERROR_DEVICE_LOST => {
            format!("the GPU device was lost after {waited} waiting for {what}")
        }
        other => format!("waiting {waited} for {what} failed: {other}"),
    }
}

/// Seconds to one decimal, or milliseconds under a second.
fn fmt_duration(d: Duration) -> String {
    if d < Duration::from_secs(1) {
        format!("{:.1} ms", d.as_secs_f64() * 1e3)
    } else {
        format!("{:.1} s", d.as_secs_f64())
    }
}

/// The closing sentence every poisoned error carries.
const UNUSABLE: &str = "this Vulkan backend is unusable from here on and every later call returns \
                        this error; restart the process";

impl VulkanShared {
    /// Has a wait failed on this device? Every drop path that frees something the GPU might still
    /// be using asks this first and leaks instead (see the module doc).
    pub(crate) fn poisoned(&self) -> bool {
        self.gpu_wait.poison.get().is_some()
    }

    /// Change the bound after creation — the unit tests' way to force the timeout path on a real
    /// device, after the backend's own creation-time probes have run under the normal bound.
    #[cfg(test)]
    pub(crate) fn set_gpu_wait_timeout_ms(&self, ms: u64) {
        self.gpu_wait
            .timeout_ns
            .store(timeout_ns(ms), Ordering::Relaxed);
    }

    /// `Err` with the poisoning error once a wait has failed, `Ok` before. Every submit calls it,
    /// so a poisoned backend fails fast without touching the device.
    pub(crate) fn live(&self) -> Result<()> {
        match self.gpu_wait.poison.get() {
            Some(msg) => Err(be(msg)),
            None => Ok(()),
        }
    }

    /// Wait for every fence in `fences`, at most the configured bound. A timeout or any other
    /// failure poisons the backend; `what` names the work for the error and is only called then.
    pub(crate) fn wait_fences(
        &self,
        fences: &[vk::Fence],
        what: &dyn Fn() -> String,
    ) -> Result<()> {
        self.live()?;
        let t0 = Instant::now();
        let bound = self.gpu_wait.timeout_ns.load(Ordering::Relaxed);
        let waited = unsafe { self.device.wait_for_fences(fences, true, bound) };
        match waited {
            Ok(()) => Ok(()),
            Err(e) => Err(self.poison(e, Some(t0.elapsed()), fences.first().copied(), &what())),
        }
    }

    /// Wait until everything submitted to the queue so far has completed, at most the configured
    /// bound — the bounded replacement for `vkQueueWaitIdle` / `vkDeviceWaitIdle` (this backend
    /// owns one queue), for the paths that have no submit of their own to wait on (teardown,
    /// `Backend::sync`, freeing a recording). A submit with no batches still signals its fence,
    /// once all work submitted earlier to the queue has completed. See [`Self::wait_fences`] for
    /// the failure contract.
    pub(crate) fn wait_queue_idle(&self, what: &dyn Fn() -> String) -> Result<()> {
        self.submit_wait(&[], what)
    }

    /// Submit `submits` with the backend's wait fence and wait for it, at most the configured
    /// bound: the work's own submit signals the fence, so waiting costs no extra submit. With no
    /// batches it is a queue-idle wait ([`Self::wait_queue_idle`]).
    ///
    /// The fence signals when THESE batches complete, unlike `vkQueueWaitIdle`, which also
    /// covered everything submitted earlier. What makes that equivalent here, case by case:
    ///
    /// - a submit that opens with a global barrier, whose first scope is everything submitted
    ///   before it: a forward's continuation segment (`Recorder::seed_barrier` after a cut, a
    ///   staging-ring rotation or a split decode recording's later segments), a self-advancing
    ///   decode recording (its leading barrier), and every `one_shot`;
    /// - any other submit — a forward's FIRST segment, a single-segment replay — goes out only
    ///   after everything submitted before it has been waited: the weight staging ring is drained
    ///   when the load scope ends (`WeightProgress::drop`), a forward's pipelined segments are
    ///   waited before `execute_static` returns and in `sync_stream`, the tensor-parallel copies
    ///   are followed by a full queue-idle wait, and one-shots are synchronous.
    ///
    /// Nothing enforces the second rule; a new path that submits work without waiting for it must
    /// open the next submit with a barrier or wait for it itself.
    ///
    /// The fence's lock is held from the submit to the reset, so two callers never share it.
    pub(crate) fn submit_wait(
        &self,
        submits: &[vk::SubmitInfo],
        what: &dyn Fn() -> String,
    ) -> Result<()> {
        self.live()?;
        let guard = match self.gpu_wait.idle_fence.lock() {
            Ok(g) => g,
            // A panic between a submit and the fence's reset (in a wait, say) poisoned the lock:
            // the fence may still be in flight or signalled, so it cannot be reused.
            Err(_) => {
                return Err(self.poison_with(
                    "a panic interrupted an earlier GPU wait and left its fence in an unknown \
                     state"
                        .to_string(),
                    None,
                    false,
                ))
            }
        };
        let fence = *guard;
        if let Err(e) = unsafe { self.device.queue_submit(self.queue, submits, fence) } {
            return Err(self.submit_failed(e, &what()));
        }
        self.wait_fences(&[fence], what)?;
        // A fence that cannot be reset stays signalled, and the next submit with it would be a
        // usage error whose wait returns at once, before its work: poison instead.
        if let Err(e) = unsafe { self.device.reset_fences(&[fence]) } {
            return Err(self.poison_with(
                format!("resetting the GPU wait fence after {} failed: {e}", what()),
                None,
                e == vk::Result::ERROR_DEVICE_LOST,
            ));
        }
        drop(guard);
        Ok(())
    }

    /// The error for a `vkQueueSubmit` that failed submitting `what`. A lost device poisons the
    /// backend like a failed wait; any other failure is an ordinary error (nothing was queued).
    pub(crate) fn submit_failed(&self, e: vk::Result, what: &str) -> Error {
        if e == vk::Result::ERROR_DEVICE_LOST {
            return self.poison(e, None, None, &format!("submitting {what}"));
        }
        be(format!("queue_submit ({what}): {e}"))
    }

    /// The error for a lost device met by `what` (an action, not a wait). Poisons the backend.
    pub(crate) fn device_lost(&self, what: &str) -> Error {
        self.poison(vk::Result::ERROR_DEVICE_LOST, None, None, what)
    }

    /// The error for `what` whose fence signalled although its work never ran to its end — its
    /// closing timestamp is missing (`recorder::account_segment`). Poisons the backend.
    pub(crate) fn work_dropped(&self, what: &str) -> Error {
        self.poison_with(
            format!(
                "the GPU signalled {what} complete but never wrote its closing timestamp: the \
                 driver reset the GPU (a hang watchdog such as Windows TDR) and signalled the \
                 fence of work it discarded, so its results are missing"
            ),
            None,
            false,
        )
    }

    /// Remember a forward segment that was just submitted, for a later failure's report (see
    /// [`RecentSegment`]). Returns the id [`Self::note_segment_ran`] takes.
    pub(crate) fn note_segment_submitted(&self, desc: SubmitDesc) -> u64 {
        self.note_segment_submitted_at(desc, Instant::now())
    }

    /// [`Self::note_segment_submitted`] for a segment submitted at `submitted`: a submit that is
    /// waited in the same call is remembered once that call has returned.
    pub(crate) fn note_segment_submitted_at(&self, desc: SubmitDesc, submitted: Instant) -> u64 {
        let id = self.gpu_wait.next_segment.fetch_add(1, Ordering::Relaxed);
        let mut recent = self
            .gpu_wait
            .recent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if recent.len() == RECENT_SEGMENTS {
            recent.pop_front();
        }
        recent.push_back(RecentSegment {
            id,
            desc,
            submitted,
            gpu_ns: None,
        });
        id
    }

    /// The remembered segment `id` completed after `gpu_ns` between its timestamps.
    pub(crate) fn note_segment_ran(&self, id: u64, gpu_ns: u64) {
        let mut recent = self
            .gpu_wait
            .recent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(seg) = recent.iter_mut().find(|s| s.id == id) {
            seg.gpu_ns = Some(gpu_ns);
        }
    }

    /// [`recent_report`] over the remembered segments, as of now.
    fn recent_segments_report(&self) -> String {
        let recent = self
            .gpu_wait
            .recent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let newest_first: Vec<_> = recent
            .iter()
            .rev()
            .map(|s| (s.desc, s.submitted.elapsed(), s.gpu_ns))
            .collect();
        recent_report(&newest_first)
    }

    /// Record the first fatal failure and return the error every later call will see.
    fn poison(
        &self,
        result: vk::Result,
        waited: Option<Duration>,
        fence: Option<vk::Fence>,
        what: &str,
    ) -> Error {
        let bound = self.gpu_wait.timeout_ns.load(Ordering::Relaxed);
        let headline = failure_headline(result, waited, bound, what);
        self.poison_with(headline, fence, result == vk::Result::ERROR_DEVICE_LOST)
    }

    /// [`Self::poison`] with the headline written: adds the fence's state, the device-fault
    /// report and the closing sentence, and records the first failure.
    fn poison_with(&self, mut msg: String, fence: Option<vk::Fence>, lost: bool) -> Error {
        if let Some(f) = fence {
            let state = match unsafe { self.device.get_fence_status(f) } {
                Ok(true) => "signalled".to_string(),
                Ok(false) => "not signalled".to_string(),
                Err(e) => e.to_string(),
            };
            msg.push_str(&format!("; fence status now: {state}"));
        }
        let lost = lost
            || matches!(
                fence.map(|f| unsafe { self.device.get_fence_status(f) }),
                Some(Err(vk::Result::ERROR_DEVICE_LOST))
            );
        msg.push_str("; ");
        msg.push_str(&self.fault_report(lost));
        msg.push_str(&self.recent_segments_report());
        msg.push_str("; ");
        msg.push_str(UNUSABLE);
        // The FIRST failure is the cause; a later one (another thread's wait on the same dead
        // device) reports that cause rather than its own echo.
        let first = self.gpu_wait.poison.get_or_init(|| msg);
        tracing::error!("[infr] {first}");
        be(first)
    }

    /// What `VK_EXT_device_fault` says, when the device offers it. The query is only valid on a
    /// lost device (`lost`); a device that merely stopped signalling has nothing to report there.
    fn fault_report(&self, lost: bool) -> String {
        let Some(fault) = &self.gpu_wait.fault else {
            return "VK_EXT_device_fault: not offered by this device".to_string();
        };
        if !lost {
            return "the device has not reported itself lost, so VK_EXT_device_fault has no \
                    report"
                .to_string();
        }
        let get = fault.fp().get_device_fault_info_ext;
        let dev = self.device.handle();
        let mut counts = vk::DeviceFaultCountsEXT::default();
        let r = unsafe { get(dev, &mut counts, std::ptr::null_mut()) };
        if r != vk::Result::SUCCESS {
            return format!("VK_EXT_device_fault: count query failed: {r}");
        }
        let mut addrs =
            vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
        let mut vendors =
            vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
        // The vendor binary blob was not enabled (`deviceFaultVendorBinary`), so ask for none.
        counts.vendor_binary_size = 0;
        let mut info = vk::DeviceFaultInfoEXT {
            p_address_infos: addrs.as_mut_ptr(),
            p_vendor_infos: vendors.as_mut_ptr(),
            ..Default::default()
        };
        let r = unsafe { get(dev, &mut counts, &mut info) };
        if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
            return format!("VK_EXT_device_fault: info query failed: {r}");
        }
        let desc = info
            .description_as_c_str()
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut out = format!("VK_EXT_device_fault: \"{desc}\"");
        for a in &addrs[..counts.address_info_count as usize] {
            out.push_str(&format!(
                ", {:?} at {:#x} (precision {:#x})",
                a.address_type, a.reported_address, a.address_precision
            ));
        }
        for v in &vendors[..counts.vendor_info_count as usize] {
            let d = v
                .description_as_c_str()
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_default();
            out.push_str(&format!(
                ", vendor \"{d}\" code {:#x} data {:#x}",
                v.vendor_fault_code, v.vendor_fault_data
            ));
        }
        out
    }
}

/// Probe `VK_EXT_device_fault`'s `deviceFault` feature. `true` = the extension is offered and the
/// feature supported, so creation may enable both.
pub(crate) fn device_fault_supported(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    has_ext: &dyn Fn(&std::ffi::CStr) -> bool,
) -> bool {
    if !has_ext(ash::ext::device_fault::NAME) {
        return false;
    }
    let mut fault = vk::PhysicalDeviceFaultFeaturesEXT::default();
    let mut feat2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut fault);
    unsafe { instance.get_physical_device_features2(physical_device, &mut feat2) };
    fault.device_fault != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VulkanBackend;
    use infr_core::backend::{Backend, BufferUsage};
    use std::sync::Arc;

    /// Bytes each fill writes, and fills per submit: 16 GiB of writes, tens of milliseconds even on
    /// the RX 7900 XTX — far past the 1 ms bound the test sets.
    const FILL_BYTES: usize = 256 << 20;
    const FILLS: usize = 64;

    /// Record [`FILLS`] serial fills of `buf` into one command buffer.
    fn heavy_recorder<'a>(
        be: &'a VulkanBackend,
        buf: &dyn infr_core::backend::Buffer,
    ) -> crate::Recorder<'a> {
        let rec = be.recorder().expect("recorder");
        for _ in 0..FILLS {
            rec.zero(buf, FILL_BYTES / 4);
        }
        rec
    }

    /// The bound turns a submit that outlives it into an error that names the submit, poisons the
    /// backend so every later call fails at once with that same error, and lets the backend drop
    /// without waiting on the work it gave up on. Forced on a real device by setting a 1 ms bound
    /// AFTER creation (the creation-time probes run under the normal bound) and submitting work
    /// that takes far longer; the same work under no bound completes, so the bound is what failed
    /// it.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn gpu_wait_timeout_poisons_the_backend_and_drops_cleanly() {
        // Control: the same submit, waited for without a bound, completes.
        {
            let be = VulkanBackend::new_with(Arc::new(crate::test_config())).expect("backend");
            be.shared.set_gpu_wait_timeout_ms(0);
            let buf = be.alloc(FILL_BYTES, BufferUsage::Activations).unwrap();
            heavy_recorder(&be, buf.as_ref())
                .finish()
                .expect("unbounded wait");
            assert!(!be.shared.poisoned());
        }

        let be = VulkanBackend::new_with(Arc::new(crate::test_config())).expect("backend");
        let buf = be.alloc(FILL_BYTES, BufferUsage::Activations).unwrap();
        be.shared.set_gpu_wait_timeout_ms(1);
        // The pipelined path (`finish_nowait` + `PendingSegment::wait`) is where plan 12's
        // integrated-GPU stall spun forever.
        let seg = heavy_recorder(&be, buf.as_ref())
            .finish_nowait()
            .expect("submit");
        let err = seg
            .wait()
            .expect_err("a 1 ms bound must time out")
            .to_string();
        println!("{err}");
        assert!(
            err.contains("did not complete")
                && err.contains(&format!("{FILLS} dispatches"))
                && err.contains("`zero_fill`")
                && err.contains("restart the process"),
            "{err}"
        );
        assert!(be.shared.poisoned());

        // Every later call fails at once, with the same error, and touches no device.
        let t0 = Instant::now();
        let again = be.recorder().map(|_| ()).expect_err("poisoned").to_string();
        assert_eq!(again, err);
        let synced = be.sync().expect_err("poisoned").to_string();
        assert_eq!(synced, err);
        assert!(t0.elapsed() < Duration::from_millis(100));

        // Teardown leaks what the GPU may still be using instead of waiting for it.
        let t0 = Instant::now();
        drop(buf);
        drop(be);
        assert!(
            t0.elapsed() < Duration::from_secs(1),
            "drop waited {:?}",
            t0.elapsed()
        );
    }

    /// A failure on a one-shot also names the forward segments submitted before it: issue 94's
    /// RX 580 lost its device during a prefill forward, and the error came from the copy after
    /// it and named only the copy. Forced with the timeout hook: one forward completes, a second
    /// is submitted and left running, and a one-shot queued behind it meets a 1 ms bound.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn a_failed_one_shot_names_the_forward_segments_before_it() {
        let be = VulkanBackend::new_with(Arc::new(crate::test_config())).expect("backend");
        let buf = be.alloc(FILL_BYTES, BufferUsage::Activations).unwrap();
        let light = be.recorder().expect("recorder");
        light.zero(buf.as_ref(), FILL_BYTES / 4);
        light.finish().expect("the completed forward");
        let running = heavy_recorder(&be, buf.as_ref())
            .finish_nowait()
            .expect("submit");
        be.shared.set_gpu_wait_timeout_ms(1);
        let err = be
            .one_shot(|_| {})
            .expect_err("a one-shot behind running work must pass a 1 ms bound")
            .to_string();
        println!("{err}");
        let list = err
            .find("; the last forward segments")
            .expect("the segment list");
        assert!(
            err[..list].contains("did not complete") && err[..list].contains("one-shot"),
            "{err}"
        );
        assert!(
            !err[..list].contains("zero_fill"),
            "the failing submit must be the one-shot, not a forward: {err}"
        );
        // Newest first: the one still running, then the one that completed, with its GPU time.
        let newest = err[list..]
            .find(&format!(
                "{FILLS} dispatches (first `zero_fill`, last `zero_fill`) submitted "
            ))
            .expect("the running segment");
        let older = err[list..]
            .find("1 dispatch (`zero_fill`) submitted ")
            .expect("the completed segment");
        assert!(newest < older, "{err}");
        assert!(
            err[list + newest..list + older].contains("not seen to complete")
                && err[list + older..].contains(" earlier, ran ")
                && err[list + older..].contains(" on the GPU"),
            "{err}"
        );
        drop(running);
    }

    #[test]
    fn the_segment_list_reads_newest_first_with_age_and_gpu_time() {
        assert_eq!(recent_report(&[]), "");
        let seg = |n| SubmitDesc {
            dispatches: n,
            first: Some("a"),
            last: Some("b"),
        };
        let r = recent_report(&[
            (seg(757), Duration::from_millis(2400), None),
            (seg(64), Duration::from_millis(2900), Some(310_000_000)),
        ]);
        assert_eq!(
            r,
            "; the last forward segments submitted, newest first (a lost device is most \
             likely the newest one's doing): 757 dispatches (first `a`, last `b`) submitted \
             2.4 s earlier, not seen to complete; 64 dispatches (first `a`, last `b`) submitted \
             2.9 s earlier, ran 310.0 ms on the GPU"
        );
    }

    #[test]
    fn zero_means_wait_forever_and_ms_scale_to_ns() {
        assert_eq!(timeout_ns(0), u64::MAX);
        assert_eq!(timeout_ns(1), 1_000_000);
        assert_eq!(
            timeout_ns(DEFAULT_GPU_WAIT_TIMEOUT_MS),
            DEFAULT_GPU_WAIT_TIMEOUT_MS * 1_000_000
        );
        assert_eq!(timeout_ns(u64::MAX), u64::MAX, "saturates, never wraps");
    }

    #[test]
    fn a_bound_below_the_minimum_is_refused() {
        assert_eq!(checked_timeout_ms(None), Ok(DEFAULT_GPU_WAIT_TIMEOUT_MS));
        assert_eq!(checked_timeout_ms(Some(0)), Ok(0), "0 waits forever");
        assert_eq!(
            checked_timeout_ms(Some(MIN_GPU_WAIT_TIMEOUT_MS)),
            Ok(MIN_GPU_WAIT_TIMEOUT_MS)
        );
        let err = checked_timeout_ms(Some(1)).unwrap_err();
        assert!(
            err.contains("below") && err.contains("INFR_GPU_WAIT_TIMEOUT_MS"),
            "{err}"
        );
    }

    #[test]
    fn each_failure_names_the_work_and_the_wait() {
        let what = "a forward segment of 7 dispatches";
        let t = failure_headline(
            vk::Result::TIMEOUT,
            Some(Duration::from_millis(2500)),
            2_000_000_000,
            what,
        );
        assert!(
            t.contains("did not complete")
                && t.contains("2.5 s")
                && t.contains("2.0 s bound")
                && t.contains("INFR_GPU_WAIT_TIMEOUT_MS")
                && t.contains(what),
            "{t}"
        );
        let l = failure_headline(
            vk::Result::ERROR_DEVICE_LOST,
            Some(Duration::from_micros(1500)),
            1,
            what,
        );
        assert!(
            l.contains("device was lost") && l.contains("1.5 ms") && l.contains(what),
            "{l}"
        );
        let o = failure_headline(
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            Some(Duration::ZERO),
            1,
            what,
        );
        assert!(o.contains("failed") && o.contains(what), "{o}");
        // A lost device met at submit time says so, not "waited 0.0 ms".
        let submit = format!("submitting {what}");
        let s = failure_headline(vk::Result::ERROR_DEVICE_LOST, None, 1, &submit);
        assert_eq!(
            s,
            format!("the GPU device was lost: submitting {what} failed")
        );
    }

    #[test]
    fn submit_desc_names_first_and_last_kernel() {
        let d = SubmitDesc {
            dispatches: 3,
            first: Some("rmsnorm"),
            last: Some("native_gemm_x"),
        };
        assert_eq!(
            d.to_string(),
            "3 dispatches (first `rmsnorm`, last `native_gemm_x`)"
        );
        let one = SubmitDesc {
            dispatches: 1,
            first: Some("k"),
            last: Some("k"),
        };
        assert_eq!(one.to_string(), "1 dispatch (`k`)");
        assert_eq!(SubmitDesc::default().to_string(), "0 dispatches");
    }
}
