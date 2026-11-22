//! The work bound's part of the recorder (`crate::work`, plan 12 S2): what a tagged dispatch
//! costs, the cut of a heavy one into ranges of its grid, each in a submit of its own
//! ([`Recorder::rollover`]), the timestamps around the dispatches it times, and the accounting of
//! a completed segment into the work estimate.

use super::*;

/// Timestamp pairs around the dispatches a recorder times for the work estimate, and what each
/// pair covers: the kernel, its family and its FLOPs. The pool is created on the first timed
/// dispatch, so a recorder that times none (every one on a fast device at its usual batch)
/// creates nothing.
#[derive(Default)]
pub(crate) struct WatchStamps {
    pool: vk::QueryPool,
    covered: Vec<Watched>,
}

/// What one timed dispatch covers: its kernel, the kernel's family, and its FLOPs.
type Watched = (&'static str, crate::work::Family, u64);

impl WatchStamps {
    /// Timed dispatches one recorder can hold; past it a dispatch is simply not timed.
    const PAIRS: u32 = 64;

    pub(super) fn destroy(&mut self, device: &ash::Device) {
        if self.pool != vk::QueryPool::null() {
            unsafe { device.destroy_query_pool(self.pool, None) };
            self.pool = vk::QueryPool::null();
        }
        self.covered.clear();
    }
}

/// Feed a completed segment into the work estimate (`crate::work`): its timed dispatches first,
/// each an exact sample of one kernel, then its whole GPU time, `gpu_ns` (`None` when the
/// segment went untimed), set against the work recorded into it, for the kernels that have no
/// timed dispatch.
pub(super) fn account_work(
    shared: &crate::VulkanShared,
    watch: &mut WatchStamps,
    work: &crate::work::SegmentWork,
    gpu_ns: Option<u64>,
) {
    account_watched(shared, watch);
    if let Some(ns) = gpu_ns {
        shared.work.calibrate(work, ns);
    }
}

/// Feed a completed segment's timed dispatches into the work estimate, then release their pool.
/// Read without waiting: the fence has signalled, and a pair the GPU never wrote (dropped work,
/// which `account_segment` reports first) is simply not used.
fn account_watched(shared: &crate::VulkanShared, w: &mut WatchStamps) {
    if w.pool == vk::QueryPool::null() {
        return;
    }
    if let Some(period) = shared.segment_ts_period {
        let mut ticks = vec![[0u64; 2]; 2 * w.covered.len()];
        let read = unsafe {
            shared.device.get_query_pool_results(
                w.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };
        if matches!(read, Ok(()) | Err(vk::Result::NOT_READY)) {
            for (pair, &(kernel, family, flops)) in ticks.as_chunks::<2>().0.iter().zip(&w.covered)
            {
                if pair[0][1] != 0 && pair[1][1] != 0 {
                    let ns = (pair[1][0].wrapping_sub(pair[0][0]) as f64 * period as f64) as u64;
                    shared.work.record(kernel, family, flops, ns);
                }
            }
        }
    }
    w.destroy(&shared.device);
}

impl<'a> Recorder<'a> {
    /// Say what the NEXT dispatch costs, for the work bound (`crate::work`): its kernel `family`,
    /// its `flops`, and how its grid may be cut. Consumed by that dispatch, which is then split
    /// into ranges, timed and counted toward the submit's work as the bound decides; a dispatch
    /// nobody tagged is recorded as one, uncounted. Ops tag only work over more than one row: a
    /// decode step is never touched.
    pub(crate) fn work_next(
        &self,
        family: crate::work::Family,
        flops: u64,
        split: crate::work::Split,
    ) {
        if self.be.shared.work.enabled() {
            self.work_next.set(Some(crate::work::WorkTag {
                family,
                flops,
                split,
            }));
        }
    }

    /// [`Self::work_next`] for a GEMM of `m` rows over a `k` x `n` weight whose shader reads only
    /// `gl_WorkGroupID` (`work::Split::Flat`, checked against its SPIR-V when it is cut). A
    /// single-row GEMM is left untagged.
    pub(super) fn tag_gemm(&self, family: crate::work::Family, m: usize, k: usize, n: usize) {
        if m > 1 {
            self.work_next(
                family,
                crate::work::gemm_flops(m, k, n),
                crate::work::Split::Flat,
            );
        }
    }

    /// Estimated GPU time of the dispatches recorded into this command buffer so far, ns (only the
    /// tagged ones count; see [`Self::work_next`]).
    pub(crate) fn work_ns(&self) -> u64 {
        self.work_ns.get()
    }

    /// The `vkCmdDispatch` of [`Self::dispatch3`], under the work bound: an untagged dispatch is
    /// recorded as it always was. A tagged one is estimated; past the budget its grid is cut into
    /// ranges of its highest non-unit dimension, recorded through the kernel's `DISPATCH_BASE`
    /// twin (same module, layout and push constants: every workgroup runs as in the whole grid, so
    /// the output is bit-identical), each range in its own submit ([`Self::rollover`]). A
    /// watchdog counts a submit's time, so ranges sharing one submit would not help: a recording
    /// that cannot open submits of its own (a persistent one, replayed whole, or any under the
    /// splitter's kill switch `device.submit_dispatches = 0`) keeps the dispatch whole, as does
    /// a kernel whose grid cannot be cut the way its tag says. A grid has no more ranges than
    /// workgroups along the dimension that is cut, so a range can still be estimated past the
    /// budget.
    pub(super) fn record_dispatch(
        &self,
        k: ComputeKernel,
        buffers: &[vk::DescriptorBufferInfo],
        push: &[u8],
        grid: [u32; 3],
    ) {
        let device = &self.be.shared.device;
        let Some(tag) = self.work_next.take() else {
            unsafe { device.cmd_dispatch(self.cmd.get(), grid[0], grid[1], grid[2]) };
            return;
        };
        let model = &self.be.shared.work;
        let est = model.estimate(k.name, tag.family, tag.flops);
        let dim = if grid[2] > 1 {
            2
        } else if grid[1] > 1 {
            1
        } else {
            0
        };
        let extent = grid[dim];
        let own_submits = !self.persistent && !self.be.shared.submit_split_disabled;
        let cuttable = match tag.split {
            // The op's claim is checked against the shader it was built from: one that does read
            // `gl_NumWorkGroups` is a mistagged call site, and is left whole rather than cut
            // wrongly.
            crate::work::Split::Flat => {
                debug_assert!(
                    !k.reads_num_workgroups,
                    "kernel {:?} reads gl_NumWorkGroups, so its grid cannot be cut as Split::Flat",
                    k.name
                );
                !k.reads_num_workgroups
            }
            // The flat index is x + y * gl_NumWorkGroups.x: whole y rows, or x on a one-row grid.
            crate::work::Split::WideGrid => dim < 2,
        };
        let pieces = if own_submits && cuttable {
            model.pieces(est, extent)
        } else {
            1
        };
        self.be.note_work_estimated(est);
        if pieces <= 1 {
            self.count_work(k.name, tag.family, tag.flops, est);
            let watched = model.watch(est) && self.watch_stamp(None);
            unsafe { device.cmd_dispatch(self.cmd.get(), grid[0], grid[1], grid[2]) };
            if watched {
                self.watch_stamp(Some((k.name, tag.family, tag.flops)));
            }
            return;
        }
        let twin = self.be.dispatch_base_pipeline(&k);
        // Same layout as `k.pipeline`: the descriptors and push constants bound for it stay bound.
        unsafe { device.cmd_bind_pipeline(self.cmd.get(), vk::PipelineBindPoint::COMPUTE, twin) };
        for (i, (base, n)) in crate::work::ranges(extent, pieces).enumerate() {
            let share = |v: u64| (v as u128 * u128::from(n) / u128::from(extent)) as u64;
            let (piece_ns, piece_flops) = (share(est), share(tag.flops));
            // Each range opens a submit of its own, the first too unless the open one is empty
            // of estimated work, so no submit carries more than one range's worth of it.
            if i > 0 || self.work_ns.get() > 0 {
                if i == 0 {
                    // The open buffer goes out without this dispatch, which `stamp` already
                    // counted and named as its last.
                    self.dispatches.set(self.dispatches.get() - 1);
                    self.last_op.set(self.prev_last_op.get());
                }
                if let Err(e) = self.rollover(k, twin, buffers, push) {
                    self.fail(e);
                    return;
                }
            }
            self.count_work(k.name, tag.family, piece_flops, piece_ns);
            let watched = model.watch(piece_ns) && self.watch_stamp(None);
            let mut b = [0u32; 3];
            let mut c = grid;
            b[dim] = base;
            c[dim] = n;
            unsafe { device.cmd_dispatch_base(self.cmd.get(), b[0], b[1], b[2], c[0], c[1], c[2]) };
            if watched {
                self.watch_stamp(Some((k.name, tag.family, piece_flops)));
            }
        }
        self.be.note_work_split(pieces);
    }

    /// Count a tagged dispatch (or one range of it) into the open command buffer: `est_ns` toward
    /// the submit's work ([`Self::work_ns`]) and its FLOPs toward what the buffer's measured GPU
    /// time is set against.
    fn count_work(
        &self,
        kernel: &'static str,
        family: crate::work::Family,
        flops: u64,
        est_ns: u64,
    ) {
        self.work_ns.set(self.work_ns.get().saturating_add(est_ns));
        self.seg_work.borrow_mut().add(kernel, family, flops);
    }

    /// Submit the command buffer recorded so far, without waiting, and continue in a fresh one in
    /// which `k`'s `twin` pipeline, `buffers` and `push` are bound again, so the dispatch being
    /// cut can go on with its next range. The fresh buffer opens with [`Self::seed_barrier`],
    /// which orders it after everything submitted (hazard tracking does not cross a command
    /// buffer), and counts as a new segment: its own timestamps, dispatch count and estimated
    /// work. At most `MAX_FLUSHED` submitted buffers stay unwaited; the oldest is waited past
    /// that. Counted as a [`crate::SegmentCut::Work`] cut.
    ///
    /// The fresh buffer is opened BEFORE the old one is submitted, so on every error the recorder
    /// still holds a command buffer that is recording: the ops that follow keep recording into it
    /// until the forward finishes and returns the error ([`Self::fail`]).
    fn rollover(
        &self,
        k: ComputeKernel,
        twin: vk::Pipeline,
        buffers: &[vk::DescriptorBufferInfo],
        push: &[u8],
    ) -> Result<()> {
        /// Submitted-but-unwaited buffers one recorder keeps, as `execute_static`'s own window.
        const MAX_FLUSHED: usize = 2;
        let shared = &self.be.shared;
        let device = &shared.device;
        shared.live()?;
        // A fresh command buffer, as `new_inner` opens one.
        let cmd_pool = *shared.cmd_pool.lock().unwrap();
        let fresh = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| be(format!("alloc cmd buffer: {e}")))?[0];
        unsafe {
            device.begin_command_buffer(
                fresh,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|e| {
            unsafe { device.free_command_buffers(cmd_pool, &[fresh]) };
            be(format!("begin cmd buffer: {e}"))
        })?;
        // Close the old segment and hand the recorder the fresh one.
        let desc = self.submit_desc();
        self.stamp_segment_end();
        let old = self.cmd.replace(fresh);
        let old_stamps = self
            .seg_query_pool
            .replace(Self::open_segment_stamps(self.be, fresh));
        let mut old_watch = std::mem::take(&mut *self.watch.borrow_mut());
        let old_work = std::mem::take(&mut *self.seg_work.borrow_mut());
        self.dispatches.set(0);
        self.first_op.set(Some(k.name));
        self.last_op.set(Some(k.name));
        self.work_ns.set(0);
        let submitted = unsafe { device.end_command_buffer(old) }
            .map_err(|e| be(format!("end cmd: {e}")))
            .and_then(|()| {
                unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                    .map_err(|e| be(format!("create fence: {e}")))
            })
            .and_then(|fence| {
                let submit = unsafe {
                    device.queue_submit(
                        shared.queue,
                        &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&old))],
                        fence,
                    )
                };
                submit.map(|()| fence).map_err(|e| {
                    unsafe { device.destroy_fence(fence, None) };
                    shared.submit_failed(e, &format!("a forward segment of {desc}"))
                })
            });
        let fence = match submitted {
            Ok(fence) => fence,
            Err(e) => {
                // Never submitted, so nothing is using the old buffer or its stamps; a lost
                // device poisoned the backend, which then frees nothing (see `gpu_wait`).
                if !shared.poisoned() {
                    unsafe {
                        device.free_command_buffers(cmd_pool, &[old]);
                        if old_stamps != vk::QueryPool::null() {
                            device.destroy_query_pool(old_stamps, None);
                        }
                    }
                    old_watch.destroy(device);
                }
                return Err(e);
            }
        };
        self.flushed_dispatches
            .set(self.flushed_dispatches.get() + desc.dispatches);
        // The descriptor pools stay with the recorder (later ranges allocate from them too) and
        // are freed only after every flushed buffer has been waited for.
        self.flushed.borrow_mut().push_back(PendingSegment {
            shared: std::sync::Arc::clone(shared),
            fence: Some(fence),
            cmd: old,
            pools: Vec::new(),
            seg_query_pool: old_stamps,
            watch: old_watch,
            work: old_work,
            earlier: Vec::new(),
            earlier_dispatches: 0,
            desc,
            recent: shared.note_segment_submitted(desc),
        });
        self.be.note_segment_cut(crate::SegmentCut::Work);
        self.seed_barrier();
        self.bind_descriptors(k, buffers);
        unsafe {
            device.cmd_bind_pipeline(fresh, vk::PipelineBindPoint::COMPUTE, twin);
            if k.push_size > 0 {
                device.cmd_push_constants(
                    fresh,
                    k.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
        }
        // The range about to be recorded is this segment's first dispatch.
        self.dispatches.set(1);
        let oldest = {
            let mut flushed = self.flushed.borrow_mut();
            (flushed.len() > MAX_FLUSHED)
                .then(|| flushed.pop_front())
                .flatten()
        };
        match oldest {
            Some(old) => old.wait(),
            None => Ok(()),
        }
    }

    /// Keep the first error met while recording, for finishing to return (see `Recorder::failed`).
    fn fail(&self, e: infr_core::error::Error) {
        let mut f = self.failed.borrow_mut();
        if f.is_none() {
            *f = Some(e);
        }
    }

    /// The error recording met, if any (see `Recorder::failed`), after waiting for the buffers
    /// this recorder submitted itself; without an error they are left in flight, for
    /// [`Self::finish_nowait`] to hand on.
    pub(super) fn check_failed(&self) -> Result<()> {
        match self.failed.borrow_mut().take() {
            Some(e) => {
                // Settle what is in flight before the caller frees the recorder; its own error
                // is the one to report.
                let _drained = self.drain_flushed();
                Err(e)
            }
            None => Ok(()),
        }
    }

    /// Wait for every command buffer this recorder submitted itself ([`Self::rollover`]), oldest
    /// first; the first failure is returned once all have been tried.
    pub(super) fn drain_flushed(&self) -> Result<()> {
        let mut first = Ok(());
        let flushed: Vec<_> = self.flushed.borrow_mut().drain(..).collect();
        for seg in flushed {
            let r = seg.wait();
            if first.is_ok() {
                first = r;
            }
        }
        first
    }

    /// Write one timestamp of a timed dispatch's pair: the opening one (`None`; creates the pool
    /// on first use, and answers `false`, recording nothing, when the dispatch cannot be timed),
    /// or the closing one with what the pair covers.
    fn watch_stamp(&self, closing: Option<Watched>) -> bool {
        if self.persistent || self.be.shared.segment_ts_period.is_none() {
            return false;
        }
        let device = &self.be.shared.device;
        let mut w = self.watch.borrow_mut();
        if closing.is_none() {
            if w.covered.len() as u32 >= WatchStamps::PAIRS {
                return false;
            }
            if w.pool == vk::QueryPool::null() {
                let created = unsafe {
                    device.create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2 * WatchStamps::PAIRS),
                        None,
                    )
                };
                // Untimed is a supported state: the estimate keeps its prior.
                let Ok(qp) = created else { return false };
                Self::reset_queries(self.be, self.cmd.get(), qp, 2 * WatchStamps::PAIRS);
                w.pool = qp;
            }
        }
        let slot = 2 * w.covered.len() as u32 + u32::from(closing.is_some());
        unsafe {
            device.cmd_write_timestamp(
                self.cmd.get(),
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                w.pool,
                slot,
            );
        }
        if let Some(c) = closing {
            w.covered.push(c);
        }
        true
    }

    /// Before this recorder's command buffer is submitted or kept: wait for the buffers it already
    /// submitted itself, and return the error recording met, if any (see `Recorder::failed`). On
    /// an error the caller frees the recorder without submitting.
    pub(super) fn settle(&self) -> Result<()> {
        let drained = self.drain_flushed();
        match self.failed.borrow_mut().take() {
            Some(e) => Err(e),
            None => drained,
        }
    }
}
