//! Memory pressure on a device with room to spare: `kernels.vulkan.vram_limit` caps what the Vulkan
//! backend reports, so the context fit, the KV format rungs and the allocation guard decide against
//! a small device while the run itself uses a real GPU and a real model. Each test runs on the
//! device the test environment names (`INFR_DEV`), skips without a GPU or without its model in the
//! HF cache, and is ignored by default like every other model-gated GPU test.
//!
//! A limit is set as the memory already in use on the device plus the room a test wants, so what
//! other processes hold does not change what a test means.

// Wrapped in its own `#[cfg(test)]` module, like every in-crate test, so the bare-print guard
// (`infr-core/tests/no_bare_print.rs`) reads its per-run notices as test output.
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use infr_core::backend::Backend;
    use infr_core::{DType, SizeSpec};

    use super::super::{open_backend, DenseVulkanSession, SeamModel};
    use crate::seam::{
        dense_act_reserve_at, kv_bytes_estimate_fmt, kv_recurrent_alloc_bytes, kv_spill_reserve,
        kv_spill_split, KvAlign, SeamKv, MIN_SESSION_CTX,
    };
    use crate::EngineConfig;

    const MIB: u64 = 1 << 20;

    /// What the device's free memory may drop per KV buffer beyond the buffer's price
    /// (`Backend::kv_alloc_cost`): the residual measured on the integrated GPU, about 6 KB a
    /// buffer, which nothing prices (docs/backlog.md B13), with room to spare. The bound
    /// [`clamped_kv_opens_no_pooled_block`] holds each buffer to; a buffer placed in a shared
    /// block would open a whole block instead.
    const RESIDUAL_SLACK: u64 = 16 * 1024;

    /// The room the B12 tests leave Qwen3-0.6B: its trained window fits in neither cache format
    /// beside its weights there, and part of it fits in each.
    const B12_ROOM: u64 = 1880 * MIB;

    /// Rooms drawn per sweep between its tightest and loosest, besides those two.
    const SEEDED_ROOMS: u64 = 3;

    /// How far [`usable_room`] raises a room the session refused.
    const ROOM_STEP: u64 = 32;

    /// The test environment's configuration (the device it names and the GPU knobs every GPU test
    /// honours) with the device memory capped at `limit` bytes and, when given, an explicit context.
    fn limited_cfg(limit: u64, ctx: Option<usize>) -> Arc<EngineConfig> {
        limited_cfg_with(limit, ctx, |_| {})
    }

    /// [`limited_cfg`] with `edit` applied last.
    fn limited_cfg_with(
        limit: u64,
        ctx: Option<usize>,
        edit: impl FnOnce(&mut EngineConfig),
    ) -> Arc<EngineConfig> {
        let env = EngineConfig::load_from_layers(&[
            infr_core::config::ConfigLayer::env().expect("test environment config")
        ]);
        let mut cfg = EngineConfig::default();
        cfg.device.dev = env.device.dev;
        infr_testkit::honour_env(&mut cfg);
        cfg.kernels.vulkan.vram_limit = Some(SizeSpec::Bytes(limit));
        cfg.device.ctx = ctx.map(|c| SizeSpec::Bytes(c as u64));
        edit(&mut cfg);
        Arc::new(cfg)
    }

    /// The model at `path` loaded under [`limited_cfg`].
    fn load(path: &std::path::Path, limit: u64, ctx: Option<usize>) -> SeamModel {
        SeamModel::load_with(path, None, limited_cfg(limit, ctx)).expect("model load")
    }

    /// How far apart [`in_use`]'s readings are taken.
    const IN_USE_PERIOD: std::time::Duration = std::time::Duration::from_millis(100);

    /// Consecutive readings within [`IN_USE_SLACK`] of each other that [`in_use`] takes as settled.
    const IN_USE_STEADY: usize = 5;

    /// How far two readings may differ and still count as the same figure: what the desktop
    /// sharing the device moves by between readings.
    const IN_USE_SLACK: u64 = 16 * MIB;

    /// The longest [`in_use`] waits for the figure to settle before it fails.
    const IN_USE_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

    /// The backend this test process reads the device's memory in use with, or `None` (with the
    /// skip line printed) when no backend opens on the test environment's device. Opened once and
    /// never dropped, so every reading and every session counts it alike, and no reader's own
    /// release lands in a later reading.
    fn reader() -> Option<&'static infr_vulkan::VulkanBackend> {
        static READER: std::sync::OnceLock<Option<infr_vulkan::VulkanBackend>> =
            std::sync::OnceLock::new();
        READER
            .get_or_init(|| {
                infr_testkit::gpu_or_skip_noted(open_backend(None, limited_cfg(u64::MAX, None)))
            })
            .as_ref()
    }

    /// The device memory in use on the test environment's device, by this process and every other,
    /// as `vk` reads it once the figure has settled.
    ///
    /// Memory this process freed keeps counting in the driver's reported usage for a while after
    /// the free: measured on the RX 7900 XTX, the first reading after gemma-4-12b's session was
    /// dropped was 4 456 771 584 bytes and settled at 1 832 218 624 half a second later. A limit
    /// built on such a reading leaves far more room than the test meant, and a figure taken on
    /// one backend and used on the next misses what the first one freed. Settled is
    /// [`IN_USE_STEADY`] consecutive readings, [`IN_USE_PERIOD`] apart, within [`IN_USE_SLACK`]
    /// of each other; one that does not settle within [`IN_USE_WAIT`] fails the test, since no
    /// limit built on it would mean what the test says.
    fn settled_use(vk: &infr_vulkan::VulkanBackend) -> u64 {
        let used = || {
            let v = vk.vram();
            v.total - v.available
        };
        let start = std::time::Instant::now();
        let first = used();
        let mut window = vec![first];
        loop {
            let (lo, hi) = window
                .iter()
                .fold((u64::MAX, 0), |(lo, hi), &u| (lo.min(u), hi.max(u)));
            if window.len() >= IN_USE_STEADY && hi - lo <= IN_USE_SLACK {
                let settled = *window.last().expect("a reading");
                eprintln!(
                    "in use: {settled} bytes, settled after {} ms from a first reading of {first}",
                    start.elapsed().as_millis()
                );
                return settled;
            }
            assert!(
                start.elapsed() < IN_USE_WAIT,
                "the device's memory in use did not settle within {IN_USE_WAIT:?}: first {first}, \
                 last readings {window:?}"
            );
            std::thread::sleep(IN_USE_PERIOD);
            window.push(used());
            if window.len() > IN_USE_STEADY {
                window.remove(0);
            }
        }
    }

    /// [`settled_use`] on the process's [`reader`], or `None` (with the skip line printed) when no
    /// backend opens on the test environment's device. Every session a test opens is preceded by
    /// one, so what the session before it freed has landed.
    fn in_use() -> Option<u64> {
        Some(settled_use(reader()?))
    }

    /// How far the device's memory in use may move across a failed run before
    /// [`retry_if_moved`] runs it again. A test's limit is the memory in use plus its room, so
    /// memory another process takes after the reading comes out of the room; the tightest rooms
    /// here sit within a few MiB of where a session is refused, by construction
    /// ([`usable_room`] stops at the first room that runs). Observed on the RX 7900 XTX on
    /// 2026-11-25 with the desktop in use: the readings move in 4 MiB steps, and one such step
    /// across a run at the tightest Qwen3-0.6B room turned its 1080-token q8_0 window into a
    /// refusal; consecutive settled readings that did not step differed by under 100 KB.
    const IN_USE_MOVED: u64 = MIB;

    /// What marks a failure as a wrong result ([`wrong`]), which [`retry_if_moved`] never runs
    /// again.
    const WRONG_RESULT: &str = "wrong result: ";

    /// A failure no movement of the device's memory can explain — a logit or a token that differs
    /// from the reference — marked so [`retry_if_moved`] reports it on the first run.
    fn wrong(msg: impl std::fmt::Display) -> String {
        format!("{WRONG_RESULT}{msg}")
    }

    /// Whether a failed run may be run again when the device's memory moved across it: a session
    /// that did not open or an allocation that was refused may be the movement's doing; a wrong
    /// result ([`wrong`]) never is.
    trait MayRetry {
        fn may_retry(&self) -> bool;
    }

    impl MayRetry for String {
        fn may_retry(&self) -> bool {
            !self.starts_with(WRONG_RESULT)
        }
    }

    impl MayRetry for Vec<String> {
        fn may_retry(&self) -> bool {
            self.iter().all(MayRetry::may_retry)
        }
    }

    /// Run `run`, which opens a session on a limited device, between two [`in_use`] readings, and
    /// once more when it failed while the reading moved by more than [`IN_USE_MOVED`] across it,
    /// saying so; the second result is the one returned. A failure on a device that did not move,
    /// a wrong result ([`MayRetry`]), or a second failure, is returned as it is.
    fn retry_if_moved<T, E: MayRetry>(
        what: impl std::fmt::Display,
        run: impl FnMut() -> Result<T, E>,
    ) -> Result<T, E> {
        retry_if_moved_with(
            || in_use().expect("a backend opened for the test"),
            what,
            run,
        )
    }

    /// [`retry_if_moved`] with the device's memory in use read by `reading`.
    fn retry_if_moved_with<T, E: MayRetry>(
        mut reading: impl FnMut() -> u64,
        what: impl std::fmt::Display,
        mut run: impl FnMut() -> Result<T, E>,
    ) -> Result<T, E> {
        let before = reading();
        let first = run();
        match &first {
            Ok(_) => return first,
            Err(e) if !e.may_retry() => return first,
            Err(_) => {}
        }
        let moved = reading().abs_diff(before);
        if moved <= IN_USE_MOVED {
            return first;
        }
        eprintln!(
            "re-run: {what}: the device's memory in use moved {moved} bytes across a failed run, \
             so it runs once more"
        );
        run()
    }

    /// `Ok` when `holds`, else `Err` with `msg`: an assertion a [`retry_if_moved`] run reports
    /// instead of panicking, so that a run the device's movement failed can run again.
    fn ensure(holds: bool, msg: impl FnOnce() -> String) -> Result<(), String> {
        if holds {
            Ok(())
        } else {
            Err(msg())
        }
    }

    fn qwen3_06b() -> Option<std::path::PathBuf> {
        crate::util::test_qwen3_06b()
    }

    fn qwen35_08b() -> Option<std::path::PathBuf> {
        crate::util::test_hub_gguf("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf")
    }

    /// The rooms a sweep runs, in MiB, ascending: `lo`, `hi` and [`SEEDED_ROOMS`] drawn between.
    fn sweep(lo: u64, hi: u64, seed: u64) -> Vec<u64> {
        let mut rooms: Vec<u64> = (0..SEEDED_ROOMS)
            .map(|i| lo + infr_testkit::gguf::splitmix64(seed ^ i) % (hi - lo))
            .chain([lo, hi])
            .collect();
        rooms.sort_unstable();
        rooms.dedup();
        rooms
    }

    /// The session's one opened slot.
    fn slot(sess: &DenseVulkanSession) -> (usize, &SeamKv) {
        sess.pool
            .slots
            .iter()
            .enumerate()
            .find_map(|(i, s)| s.as_ref().map(|kv| (i, kv)))
            .expect("the generation opened a slot")
    }

    /// What one clamped run allocated and committed for its KV cache.
    struct ClampedRun {
        /// The model, to run it again.
        path: std::path::PathBuf,
        /// The room above what was in use that the limit left, in MiB.
        room_mib: u64,
        /// The context the slot was allocated for.
        slot_ctx: usize,
        /// The bytes the slot's KV buffers hold, and how many buffers.
        allocated: u64,
        buffers: u64,
        /// The same buffers each priced at what the backend says the device commits for it
        /// (`Backend::kv_alloc_cost`), as the estimate prices them.
        priced: u64,
        /// The recurrent-state snapshot's bytes: in the estimate, allocated only on first use.
        snapshot: u64,
        estimate: u64,
        /// What the KV allocation took from the device's budget; `None` on a device without a
        /// live budget (`VramInfo::live`), where the room the runner reads never moves.
        committed: Option<u64>,
    }

    /// Why a clamped run did not run.
    enum RunErr {
        /// The session refused to open because no usable cache fits the room.
        NoUsableCache(String),
        /// Anything else.
        Other(String),
    }

    impl MayRetry for RunErr {
        fn may_retry(&self) -> bool {
            true
        }
    }

    impl std::fmt::Display for RunErr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                RunErr::NoUsableCache(m) | RunErr::Other(m) => f.write_str(m),
            }
        }
    }

    /// Open the default (auto-sized) session of the model at `path` with the device limited to
    /// `room_mib` above the bytes in use as [`in_use`] reads them just before, check that the
    /// context was clamped below the
    /// trained window, generate a few tokens, and report what the slot allocated — or why the
    /// session refused to open or the generation failed. Run again when it failed while the
    /// device's memory in use moved ([`retry_if_moved`]).
    fn clamped_run(path: &std::path::Path, room_mib: u64) -> Result<ClampedRun, RunErr> {
        retry_if_moved(format!("{room_mib} MiB of room"), || {
            clamped_run_once(path, room_mib)
        })
    }

    /// One [`clamped_run`].
    fn clamped_run_once(path: &std::path::Path, room_mib: u64) -> Result<ClampedRun, RunErr> {
        let used = in_use().expect("a backend opened for the sweep");
        let model = load(path, used + room_mib * MIB, None);
        let mut sess = model.vulkan_session_default().map_err(|e| {
            let msg = format!("{room_mib} MiB of room: the default session did not open: {e}");
            // The refuse rung's own wording (`SeamModel::clamp_default_ctx`): the one open failure
            // that means the room is too small rather than that something else broke.
            if e.to_string().contains("no usable KV cache fits") {
                RunErr::NoUsableCache(msg)
            } else {
                RunErr::Other(msg)
            }
        })?;
        let opened_ctx = sess.max_ctx;
        assert!(
            opened_ctx < model.cfg.n_ctx_train,
            "{room_mib} MiB of room: the default context {opened_ctx} was not clamped below the \
             trained {}",
            model.cfg.n_ctx_train
        );
        if let Err(e) =
            model.generate_vulkan_session(&mut sess, "The capital of France is", 4, None, |_| {})
        {
            eprintln!("{room_mib} MiB of room: ctx {opened_ctx}, the run failed: {e}");
            return Err(RunErr::Other(format!("{room_mib} MiB of room: {e}")));
        }
        let live = sess.vram().live;
        let (_, kv) = slot(&sess);
        let sizes: Vec<u64> = kv
            .kbufs
            .iter()
            .chain(&kv.vbufs)
            .chain(kv.cbufs.iter().flatten())
            .map(|b| b.len_bytes() as u64)
            .collect();
        let align = KvAlign::of(&sess.be);
        let rounded = |b: usize| align.commit.committed(b as u64);
        let snapshot: u64 = (0..model.cfg.n_layer)
            .filter_map(|l| kv_recurrent_alloc_bytes(&model.cfg, l))
            .map(|(k, v)| rounded(k) + rounded(v))
            .sum();
        let estimate = kv_bytes_estimate_fmt(
            &model.cfg, kv.max_ctx, kv.kv_ring, kv.ubatch, align, kv.k_fmt, kv.v_fmt,
        );
        let run = ClampedRun {
            path: path.to_path_buf(),
            room_mib,
            slot_ctx: kv.max_ctx,
            allocated: sizes.iter().sum(),
            buffers: sizes.len() as u64,
            priced: sizes.iter().map(|&b| rounded(b as usize)).sum(),
            snapshot,
            estimate,
            committed: if live { kv.kv_committed } else { None },
        };
        eprintln!(
            "{room_mib} MiB of room: ctx {opened_ctx} -> {} ({:?}/{:?}), {} KV buffers holding {} \
             (priced {}) + snapshot {snapshot}, estimate {}, committed {}; {} device memory \
             allocations live of {}",
            run.slot_ctx,
            kv.k_fmt,
            kv.v_fmt,
            run.buffers,
            run.allocated,
            run.priced,
            run.estimate,
            match run.committed {
                Some(c) => format!(
                    "{c} ({:+} against the buffers, {:+} against their price)",
                    c as i64 - run.allocated as i64,
                    c as i64 - run.priced as i64
                ),
                None => "unmeasured (no live budget)".to_string(),
            },
            sess.be.device_memory_allocations(),
            sess.be.max_device_memory_allocations(),
        );
        Ok(run)
    }

    /// The smallest room, from `start` MiB up in [`ROOM_STEP`]s and below `hi`, where the model at
    /// `path` runs with a clamped context: below it the session refuses to open for want of a
    /// usable cache, and a run never keeps less than [`MIN_SESSION_CTX`] (which
    /// `clamped_default_ctx_allocates_its_estimate` asserts). Any other failure on the way up is a
    /// failure, not a room to step past. Probed rather than computed: where the refusal falls moves
    /// with what the device reports in use while the test runs.
    fn usable_room(path: &std::path::Path, start: u64, hi: u64) -> u64 {
        let mut room = start;
        while room < hi {
            match clamped_run(path, room) {
                Ok(_) => return room,
                Err(RunErr::NoUsableCache(_)) => {}
                Err(e @ RunErr::Other(_)) => panic!("{e}"),
            }
            room += ROOM_STEP;
        }
        panic!("no room from {start} to {hi} MiB gives a usable clamped context");
    }

    /// Every clamped run both clamped-context tests check: Qwen3-0.6B (attention only) and
    /// Qwen3.5-0.8B (recurrent layers beside attention, so a snapshot in the estimate), each over
    /// seeded rooms from its [`usable_room`] up. Qwen3-0.6B's loosest room is past the rooms where
    /// its KV buffers, when they were sub-allocated from shared blocks, outgrew the room and the
    /// guard refused one. `None` when no backend opens.
    fn clamped_sweeps() -> Option<Vec<Result<ClampedRun, RunErr>>> {
        reader()?;
        let mut runs = Vec::new();
        for (name, path, start, hi, seed) in [
            ("Qwen3-0.6B", qwen3_06b(), 1150, 2600, 0x5134),
            ("Qwen3.5-0.8B", qwen35_08b(), 1350, 3200, 0x5135),
        ] {
            let Some(path) = path else {
                eprintln!("skip: {name} is not in the HF cache");
                continue;
            };
            let lo = usable_room(&path, start, hi);
            runs.extend(
                sweep(lo, hi, seed)
                    .into_iter()
                    .map(|room| clamped_run(&path, room)),
            );
        }
        Some(runs)
    }

    /// A default context clamped to a limited device loads, generates, keeps a usable context and
    /// allocates exactly the KV bytes the fit priced it at, each buffer at what the backend's
    /// `kv_alloc_cost` says it commits (plan 13 S2's estimate, whose clamp could only be checked
    /// as arithmetic on a device the cached models fit whole).
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn clamped_default_ctx_allocates_its_estimate() {
        for run in clamped_sweeps().unwrap_or_default() {
            let run = run.unwrap_or_else(|e| panic!("{e}"));
            assert!(
                run.slot_ctx >= MIN_SESSION_CTX,
                "{} MiB of room: a clamped context of {} tokens",
                run.room_mib,
                run.slot_ctx
            );
            assert_eq!(
                run.priced + run.snapshot,
                run.estimate,
                "{} MiB of room: the clamped slot's KV bytes against the estimate it was fitted with",
                run.room_mib
            );
        }
    }

    /// What dedicated KV allocations produce: a clamped run's KV allocation takes from the device
    /// no more than its buffers' price (each buffer at what `Backend::kv_alloc_cost` says it
    /// commits, as the estimate prices it) plus [`RESIDUAL_SLACK`] each, at any room, so no shared
    /// block is opened for the cache. A KV buffer placed in one of gpu-allocator's shared blocks
    /// (as every buffer below the backend's dedicated threshold was, before KV buffers got their
    /// own allocations) commits the whole block when it opens one, which neither the fit nor the
    /// guard prices — every Qwen3.5-0.8B run (its recurrent states) and every Qwen3-0.6B room here
    /// did.
    ///
    /// The commitment is the device's free memory before the KV allocation less after it, a
    /// device-wide figure: on a unified-memory device, memory another process takes meanwhile
    /// lands in it. A run past the bound is run once more, saying so, and fails only if the
    /// second run is past it too. Skips the check, saying so, on a device without a live budget,
    /// where the commitment cannot be measured.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn clamped_kv_opens_no_pooled_block() {
        let past_bound = |run: &ClampedRun| -> Option<String> {
            let committed = run.committed?;
            let bound = run.priced + RESIDUAL_SLACK * run.buffers;
            (committed > bound).then(|| {
                format!(
                    "{} MiB of room: the KV allocation committed {committed} bytes for {} in {} \
                     buffers priced {}, {} past the bound",
                    run.room_mib,
                    run.allocated,
                    run.buffers,
                    run.priced,
                    committed - bound
                )
            })
        };
        let mut failed = Vec::new();
        for run in clamped_sweeps().unwrap_or_default() {
            let run = match run {
                Ok(run) => run,
                Err(e) => {
                    failed.push(e.to_string());
                    continue;
                }
            };
            if run.committed.is_none() {
                eprintln!(
                    "skip: {} MiB of room: no live budget, so the KV commitment is unmeasured",
                    run.room_mib
                );
                continue;
            }
            let Some(first) = past_bound(&run) else {
                continue;
            };
            eprintln!("{first}; running it once more, in case another process moved the figure");
            match clamped_run(&run.path, run.room_mib) {
                Ok(again) => failed.extend(past_bound(&again)),
                Err(e) => failed.push(e.to_string()),
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    /// How far the context a session opens with may exceed the one it runs with, priced as the KV
    /// cache and activation reserve the difference would have cost: what the device may lose to
    /// another process, or to memory the pre-load fit cannot price, between the session opening and
    /// the KV allocation. A whole arena block of weights the fit failed to price would be past it.
    const AGREEMENT_SLACK: u64 = 16 * MIB;

    /// How one auto-sized session on a limited device ended.
    enum AutoEnd {
        /// The session refused to open. Opening loads no weights, so none were.
        RefusedBeforeLoad(String),
        /// The session opened with `opened` tokens and its first generation failed once the
        /// weights had taken `weights_took` bytes of the device's room.
        RefusedAfterLoad {
            opened: usize,
            weights_took: u64,
            err: String,
        },
        /// The session ran: it opened with `opened` tokens and its slot holds `ctx`, `excess` being
        /// what the cache and activation reserve of the tokens it lost would have cost. `unused`
        /// is the other direction: what the fit before the load priced the weights and their
        /// upload staging at, less what the load took from the room — room the window was not
        /// given (negative when the load took more than was priced). `staging` is the ring's
        /// size as the backend reports it (`VulkanBackend::weight_load_cost`).
        Ran {
            opened: usize,
            ctx: usize,
            k_fmt: DType,
            v_fmt: DType,
            excess: u64,
            unused: i64,
            staging: u64,
        },
    }

    /// Open the default session of the model at `path` with the device limited to `room_mib` above
    /// the bytes in use as [`in_use`] reads them just before, and generate a few tokens, reporting
    /// where it stopped.
    fn auto_run(path: &std::path::Path, room_mib: u64) -> AutoEnd {
        let used = in_use().expect("a backend opened for the sweep");
        let model = load(path, used + room_mib * MIB, None);
        let mut sess = match model.vulkan_session_default() {
            Ok(sess) => sess,
            Err(e) => return AutoEnd::RefusedBeforeLoad(e.to_string()),
        };
        let opened = sess.max_ctx;
        let before = sess.be.alloc_room();
        let held = model.preload_held(&sess.be);
        if let Err(e) =
            model.generate_vulkan_session(&mut sess, "The capital of France is", 4, None, |_| {})
        {
            return AutoEnd::RefusedAfterLoad {
                opened,
                weights_took: before.saturating_sub(sess.be.alloc_room()),
                err: e.to_string(),
            };
        }
        let (_, kv) = slot(&sess);
        let align = KvAlign::of(&sess.be);
        let cost = |ctx: usize| {
            kv_bytes_estimate_fmt(
                &model.cfg, ctx, kv.kv_ring, kv.ubatch, align, kv.k_fmt, kv.v_fmt,
            ) + dense_act_reserve_at(&model.cfg, ctx, kv.ubatch)
        };
        let took = before.saturating_sub(kv.room_after_weights.expect("a Vulkan room"));
        AutoEnd::Ran {
            opened,
            ctx: kv.max_ctx,
            k_fmt: kv.k_fmt,
            v_fmt: kv.v_fmt,
            excess: cost(opened).saturating_sub(cost(kv.max_ctx)),
            unused: (held.weights + held.staging) as i64 - took as i64,
            // From the backend, not from what the fit priced, so a fit that over-prices the ring
            // cannot widen its own bound.
            staging: sess.be.weight_load_cost().staging,
        }
    }

    /// The context the default session opens with is the one it runs with, to within
    /// [`AGREEMENT_SLACK`], across rooms from where nothing fits to where the clamp barely binds,
    /// on both cached models — the rooms where the fit before the load, when it priced neither the
    /// weights' memory past their footprint nor the driver's reserve, promised a context the
    /// measurement after it refused or cut (measured with this test on 2026-11-24: Qwen3-0.6B
    /// refused after a full load at 1000 MiB, run with 524 tokens at 1100 and 980 at 1150;
    /// Qwen3.5-0.8B refused after a full load at 1100 and 1200 MiB; every run cut by about 395 MB
    /// and 413 MB of context). A room that cannot hold a usable context is refused before the
    /// weights load, and no run keeps less than [`MIN_SESSION_CTX`].
    ///
    /// The other direction is bounded too: the room the fit priced and the load did not take may
    /// not exceed the staging ring's size as the backend reports it, the most the fit can
    /// over-price on a device that charges the ring nothing (the integrated GPU charges about
    /// 12.4 MB less than its size, the RX 7900 XTX all of it; see docs/backlog.md
    /// B-PRELOAD-CLAMP). A fit that counted the ring twice is past it.
    ///
    /// A room that fails while the device's memory in use moved across its run (another process
    /// on the GPU: observed once on the RX 7900 XTX, 77 MB taken during one run and kept) is run
    /// once more ([`retry_if_moved`]), and fails the test only if it fails again.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn auto_ctx_before_the_load_is_the_ctx_after_it() {
        if reader().is_none() {
            return;
        }
        let mut failed = Vec::new();
        for (name, path, rooms) in [
            (
                "Qwen3-0.6B",
                qwen3_06b(),
                &[
                    1000, 1100, 1150, 1200, 1300, 1400, 1536, 1600, 1800, 2000, 2300, 2600,
                ][..],
            ),
            (
                "Qwen3.5-0.8B",
                qwen35_08b(),
                &[
                    1100, 1200, 1280, 1300, 1350, 1400, 1500, 1700, 2000, 2400, 2800, 3200,
                ][..],
            ),
        ] {
            let Some(path) = path else {
                eprintln!("skip: {name} is not in the HF cache");
                continue;
            };
            // One room's run judged: the row it prints and what it failed.
            let judge = |room: u64, end: AutoEnd| -> (String, Vec<String>) {
                let mut fails = Vec::new();
                let row = match end {
                    AutoEnd::RefusedBeforeLoad(e) => {
                        format!(
                            "refused before the load: {}",
                            e.lines().next().unwrap_or("")
                        )
                    }
                    AutoEnd::RefusedAfterLoad {
                        opened,
                        weights_took,
                        err,
                    } => {
                        fails.push(format!(
                            "{name} at {room} MiB: opened with {opened} tokens, refused after the \
                             weights took {weights_took} bytes: {err}"
                        ));
                        format!("opened {opened}, refused after the load: {err}")
                    }
                    AutoEnd::Ran {
                        opened,
                        ctx,
                        k_fmt,
                        v_fmt,
                        excess,
                        unused,
                        staging,
                    } => {
                        if ctx < MIN_SESSION_CTX {
                            fails.push(format!("{name} at {room} MiB: ran with {ctx} tokens"));
                        }
                        if excess > AGREEMENT_SLACK {
                            fails.push(format!(
                                "{name} at {room} MiB: opened with {opened} tokens, ran with \
                                 {ctx}: {excess} bytes promised and not held"
                            ));
                        }
                        if unused > staging as i64 {
                            fails.push(format!(
                                "{name} at {room} MiB: the fit priced {unused} bytes the load did \
                                 not take, past the staging ring's {staging}"
                            ));
                        }
                        format!(
                            "opened {opened}, ran {ctx} ({k_fmt:?}/{v_fmt:?}), excess {excess}, \
                             unused {unused} of a {staging}-byte ring"
                        )
                    }
                };
                (row, fails)
            };
            for &room in rooms {
                let ran = retry_if_moved(format!("{name} at {room} MiB"), || {
                    let (row, fails) = judge(room, auto_run(&path, room));
                    eprintln!("{name} at {room} MiB: {row}");
                    if fails.is_empty() {
                        Ok(())
                    } else {
                        Err(fails)
                    }
                });
                failed.extend(ran.err().unwrap_or_default());
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    /// What the resident weight arena really opens for a model's upload units, against the floor
    /// the fit before the load prices it at (`crate::seam::weight_arena_floor`): every unit is
    /// allocated as a weight on a fresh backend, in file order with the concatenated units last,
    /// and the blocks are counted. A unit of at least a block gets a block of its own size and the
    /// rest share blocks of `block_min`, so the count prices the commitment; it is never below
    /// the floor, and above it by no more than first-fit can strand — less than the largest shared
    /// unit in every shared block. The block sizes come from the placement rule, not a read of the
    /// device, so this checks how many blocks the arena opens, not what the driver commits for
    /// each.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn arena_commits_its_floor_for_a_model() {
        use infr_core::backend::BufferUsage;
        for (name, path) in [("Qwen3-0.6B", qwen3_06b()), ("Qwen3.5-0.8B", qwen35_08b())] {
            let Some(path) = path else {
                eprintln!("skip: {name} is not in the HF cache");
                continue;
            };
            let Some(vk) =
                infr_testkit::gpu_or_skip_noted(open_backend(None, limited_cfg(u64::MAX, None)))
            else {
                return;
            };
            let model = load(&path, u64::MAX, None);
            let units = crate::seam::weight_upload_units(
                &model.gguf,
                &model.cfg,
                &model.ecfg,
                vk.capabilities().combined_gu,
            );
            let cost = vk.weight_load_cost();
            let floor = crate::seam::weight_arena_floor(&units, cost);
            let round = |u: u64| u.max(1).next_multiple_of(cost.align);
            let (whole, shared): (Vec<u64>, Vec<u64>) = units
                .iter()
                .map(|&u| round(u))
                .partition(|&u| u >= cost.block_min);
            let before = vk.device_memory_allocations();
            let held: Vec<_> = units
                .iter()
                .map(|&u| {
                    vk.alloc(u as usize, BufferUsage::Weights)
                        .expect("a weight allocation")
                })
                .collect();
            let opened = vk.device_memory_allocations() - before;
            drop(held);
            let shared_blocks = opened - whole.len() as u64;
            let committed = whole.iter().sum::<u64>() + shared_blocks * cost.block_min;
            let largest_shared = shared.iter().copied().max().unwrap_or(0);
            let tail_bound = shared_blocks * largest_shared;
            eprintln!(
                "{name}: {} units ({} whole), {opened} blocks; floor {floor}, arena {committed} \
                 ({:+}), bound {tail_bound}; {cost:?}",
                units.len(),
                whole.len(),
                committed as i64 - floor as i64,
            );
            assert!(
                committed >= floor,
                "{name}: the arena committed {committed}, below the floor {floor}"
            );
            assert!(
                committed - floor <= tail_bound,
                "{name}: the arena committed {committed}, {} past the floor {floor}, more than \
                 first-fit can strand ({tail_bound})",
                committed - floor
            );
        }
    }

    /// The q8_0 rung for an explicit context (B12): on a limited device, a window between what f16 and
    /// what q8_0 can hold beside the weights is served as asked, with a q8_0 cache.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn explicit_ctx_f16_cannot_hold_is_served_in_q8() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B is not in the HF cache");
            return;
        };
        let Some(reader) = reader() else {
            return;
        };
        let run = || -> Result<(), String> {
            let used = settled_use(reader);
            let limit = used + B12_ROOM;
            let probe = load(&path, limit, None);
            let vk = open_backend(None, probe.ecfg.clone()).expect("backend");
            let _scope = crate::seam::enter_backend_scope(&vk);
            let f16 = probe
                .kv_fit_ctx_fmt(&vk, false)
                .expect("an attention model");
            let q8 = probe.kv_fit_ctx_fmt(&vk, true).expect("an attention model");
            drop(vk);
            // What the probe backend freed lands before the session's own backend measures the
            // room.
            settled_use(reader);
            let want = (f16 + q8) / 2;
            ensure(f16 < want && want < q8, || {
                format!("f16 {f16}, q8_0 {q8}: no window between them")
            })?;

            let model = load(&path, limit, Some(want));
            let mut sess = model
                .vulkan_session(want)
                .map_err(|e| format!("session: {e}"))?;
            // The explicit-context rung pins q8_0 when the session opens, before any weight loads;
            // the dense placement's own q8_0 rung would also catch this window later, so the
            // served format alone cannot say which rung decided.
            let scope = crate::seam::PlacementScope::enter(sess.pins().clone());
            let pinned = crate::seam::kv_auto_q8();
            drop(scope);
            ensure(pinned, || {
                "the session opened without the explicit-context q8_0 pin".to_string()
            })?;
            model
                .generate_vulkan_session(&mut sess, "The capital of France is", 4, None, |_| {})
                .map_err(|e| format!("the q8_0 window did not run: {e}"))?;
            let (_, kv) = slot(&sess);
            eprintln!(
                "f16 fits {f16}, q8_0 fits {q8}; asked {want}, served {} at {:?}/{:?}",
                kv.max_ctx, kv.k_fmt, kv.v_fmt
            );
            ensure(kv.max_ctx == want, || {
                format!(
                    "an explicit context is served as asked: asked {want}, served {}",
                    kv.max_ctx
                )
            })?;
            ensure((kv.k_fmt, kv.v_fmt) == (DType::Q8_0, DType::Q8_0), || {
                format!("served in {:?}/{:?}, not q8_0", kv.k_fmt, kv.v_fmt)
            })
        };
        retry_if_moved("the q8_0 window", run).unwrap_or_else(|e| panic!("{e}"));
    }

    /// The runner's refusal after the weights load (B12), and the window that used to reach it.
    /// The largest explicit f16 window the fit accepts before the load is now one the load leaves
    /// room for, so it runs as asked: the fit prices what the weights commit, the upload staging
    /// and the driver's reserve. What still reaches the refusal is a window the fit rejects and
    /// placement still takes resident, because placement prices the weights at their bare
    /// footprint: with the cache format set to f16, so neither q8_0 rung can take it, such a window
    /// loads and is refused between the weight upload and the KV allocation, with the sizes — not
    /// by the allocation guard on one layer's buffer (docs/backlog.md B-PRELOAD-CLAMP). Both
    /// windows are the largest at [`IN_USE_SLACK`] less room than the sessions get: each session
    /// reads the room again, and at the exact edge the desktop sharing the device moving a few
    /// MiB between the readings was observed to send the fit's window to q8_0 and placement's to
    /// streaming.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn explicit_ctx_short_of_the_room_after_the_weights_is_refused() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B is not in the HF cache");
            return;
        };
        let Some(reader) = reader() else {
            return;
        };
        let run = || -> Result<(), String> {
            let used = settled_use(reader);
            let limit = used + B12_ROOM;
            let probe = load(&path, limit, None);
            // Both windows are taken at the edge of a room `IN_USE_SLACK` smaller than the
            // sessions get, so the desktop moving between this reading and theirs cannot push
            // either one past its edge.
            let vk = open_backend(None, limited_cfg(limit - IN_USE_SLACK, None)).expect("backend");
            let _scope = crate::seam::enter_backend_scope(&vk);
            let fits = probe
                .kv_fit_ctx_fmt(&vk, false)
                .expect("an attention model");
            let placed = crate::seam::kv_fit_ctx_for(
                &probe.cfg,
                &probe.ecfg,
                probe.footprint().total(),
                &vk.vram(),
                KvAlign::of(&vk),
                DType::F16,
                DType::F16,
            )
            .expect("an attention model");
            drop(vk);
            // What the probe backend freed lands before the session's own backend measures the
            // room.
            settled_use(reader);
            ensure(fits < placed, || {
                format!("the fit ({fits}) must be below what the bare footprint leaves ({placed})")
            })?;

            let model = load(&path, limit, Some(fits));
            let mut sess = model.vulkan_session(fits).map_err(|e| {
                format!("the pre-load fit did not accept its own largest window {fits}: {e}")
            })?;
            model
                .generate_vulkan_session(&mut sess, "The capital of France is", 4, None, |_| {})
                .map_err(|e| format!("the largest window the fit accepts, {fits}, failed: {e}"))?;
            let (_, kv) = slot(&sess);
            ensure(kv.max_ctx == fits, || {
                format!(
                    "an explicit context is served as asked: asked {fits}, served {}",
                    kv.max_ctx
                )
            })?;
            ensure((kv.k_fmt, kv.v_fmt) == (DType::F16, DType::F16), || {
                format!("served in {:?}/{:?}, not f16", kv.k_fmt, kv.v_fmt)
            })?;
            drop(sess);
            // What the first session freed lands before the second one's placement measures the
            // room.
            settled_use(reader);

            let f16 = limited_cfg_with(limit, Some(placed), |c| {
                c.kv.type_k = Some(DType::F16);
                c.kv.type_k_specified = true;
                c.kv.type_v = Some(DType::F16);
                c.kv.type_v_specified = true;
            });
            let model = SeamModel::load_with(&path, None, f16).expect("model load");
            let mut sess = model.vulkan_session(placed).map_err(|e| {
                format!("the refusal before the load counts the cache alone, yet {placed}: {e}")
            })?;
            let Err(err) = model.generate_vulkan_session(
                &mut sess,
                "The capital of France is",
                4,
                None,
                |_| {},
            ) else {
                return Err(format!(
                    "the window {placed} ran: it must not once the weights' own memory is counted"
                ));
            };
            eprintln!("fit {fits}; asked {placed} in f16: {err}");
            let msg = err.to_string();
            ensure(
                msg.contains(&format!("context {placed} cannot be served"))
                    && msg.contains("after the weights"),
                || format!("the refusal after the load, not another failure: {msg}"),
            )
        };
        retry_if_moved("the windows beside the weights", run).unwrap_or_else(|e| panic!("{e}"));
    }

    /// The ladder's order under `kv.overflow` (B62, plan 13 S3) on a device whose room holds
    /// Qwen3-0.6B's trained window in neither cache format ([`B12_ROOM`]): the auto-sized window is
    /// clamped to what fits in VRAM before the load and runs at that, with its cache allocated
    /// whole in device memory. Before S3 the fit before the load kept the trained window under
    /// `kv.overflow` (dense placement was then decided for it) and left the re-clamp after the
    /// load to cut it.
    ///
    /// The backend placed none of the cache in system RAM, on either kind of device: on a discrete
    /// one a window the ladder serves whole is allocated as without the flag (B13's few KB a buffer
    /// past its price must not send its last buffers to system RAM), and on a unified-memory one
    /// nothing spills at all. Where the budget is live and not unified, the allocation's commitment
    /// says the same. The explicit window that does not fit, which spills, is
    /// [`overflow_spills_an_explicit_ctx_that_does_not_fit`]'s.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn overflow_clamps_an_auto_sized_ctx() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B is not in the HF cache");
            return;
        };
        let Some(reader) = reader() else {
            return;
        };
        let run = || -> Result<(), String> {
            let limit = settled_use(reader) + B12_ROOM;
            let cfg = limited_cfg_with(limit, None, |c| c.kv.overflow = true);
            let model = SeamModel::load_with(&path, None, cfg).expect("model load");
            let trained = model.cfg.n_ctx_train;
            let mut sess = model
                .vulkan_session_default()
                .map_err(|e| format!("the auto-sized session did not open: {e}"))?;
            let opened = sess.max_ctx;
            model
                .generate_vulkan_session(&mut sess, "The capital of France is", 4, None, |_| {})
                .map_err(|e| format!("the clamped window {opened} did not run: {e}"))?;
            let vram = sess.vram();
            let (_, kv) = slot(&sess);
            let allocated: u64 = kv
                .kbufs
                .iter()
                .chain(&kv.vbufs)
                .map(|b| b.len_bytes() as u64)
                .sum();
            let committed = kv.kv_committed.filter(|_| vram.live && !vram.uma);
            eprintln!(
                "trained {trained}, opened {opened}, served {} ({:?}/{:?}); KV buffers holding \
                 {allocated}, committed {:?}, unified memory {}",
                kv.max_ctx, kv.k_fmt, kv.v_fmt, kv.kv_committed, vram.uma
            );
            ensure(opened < trained, || {
                format!(
                    "the auto-sized window opened at {opened} of the trained {trained}: kept \
                     whole, not clamped, under kv.overflow"
                )
            })?;
            ensure((MIN_SESSION_CTX..=opened).contains(&kv.max_ctx), || {
                format!("opened with {opened} tokens, served {}", kv.max_ctx)
            })?;
            let placed = sess.be.kv_spill_counts();
            ensure(placed.host_bufs == 0, || {
                format!("a served cache placed buffers in system RAM: {placed:?}")
            })?;
            match committed {
                Some(c) => ensure(c >= allocated, || {
                    format!(
                        "the clamped cache's allocation took {c} bytes of device memory for \
                         {allocated} bytes of buffers: part of it spilled"
                    )
                }),
                None => {
                    eprintln!("skip: where the clamped cache landed is not read off this budget");
                    Ok(())
                }
            }
        };
        retry_if_moved("the auto-sized window", run).unwrap_or_else(|e| panic!("{e}"));
    }

    /// The prefill chunk the spill tests pin, so their prompts run several chunks on either device
    /// and the chunk cannot move between the runs one test compares.
    const SPILL_UBATCH: usize = 256;

    /// Words in the spill tests' prompt: more than two [`SPILL_UBATCH`] chunks of Qwen3-0.6B
    /// tokens and under [`MIN_SESSION_CTX`] with the decode, which the tests check.
    const SPILL_PROMPT_WORDS: usize = 640;

    /// Tokens the spill tests decode after the prompt.
    const SPILL_DECODE: usize = 8;

    /// [`limited_cfg_with`] under `kv.overflow` with the chunk pinned at [`SPILL_UBATCH`], and,
    /// when `kv` is given, both cache sides set to it.
    fn overflow_cfg(limit: u64, ctx: Option<usize>, kv: Option<DType>) -> Arc<EngineConfig> {
        limited_cfg_with(limit, ctx, |c| {
            c.kv.overflow = true;
            c.device.ubatch = Some(SPILL_UBATCH);
            c.device.ubatch_specified = true;
            if let Some(dt) = kv {
                c.kv.type_k = Some(dt);
                c.kv.type_k_specified = true;
                c.kv.type_v = Some(dt);
                c.kv.type_v_specified = true;
            }
        })
    }

    /// What one spilled run allocated, as the backend placed it.
    struct SpilledRun {
        ctx: usize,
        k_fmt: DType,
        v_fmt: DType,
        counts: infr_core::budget::SpillCounts,
        prompt_tokens: usize,
        n_gen: usize,
        /// Whether placement streamed the weights.
        streamed: bool,
    }

    /// Open `open`'s session on `model`, run the spill prompt of several chunks and a short decode
    /// on it, and report where the backend placed the cache. On a device with a live budget that
    /// is not unified memory, also check what the spill left: when any of the cache went to device
    /// memory, the device memory free after it is at least the spill reserve the run was priced
    /// with (`kv_spill_reserve`), short by no more than [`RESIDUAL_SLACK`] a buffer placed there
    /// (what the driver takes per buffer past its price, B13); and the bytes the backend placed in
    /// device memory are the split `kv_spill_split` priced from the same room, to within one
    /// buffer.
    fn spilled_run(
        model: &SeamModel,
        open: impl FnOnce(&SeamModel) -> anyhow::Result<DenseVulkanSession>,
    ) -> Result<SpilledRun, String> {
        let prompt = long_prompt(SPILL_PROMPT_WORDS);
        let prompt_tokens = model.encode(&prompt).expect("encode").len();
        ensure(
            prompt_tokens > 2 * SPILL_UBATCH && prompt_tokens + SPILL_DECODE < MIN_SESSION_CTX,
            || format!("the spill prompt is {prompt_tokens} tokens"),
        )?;
        let mut sess = open(model).map_err(|e| format!("the session did not open: {e}"))?;
        let stats = model
            .generate_vulkan_session(&mut sess, &prompt, SPILL_DECODE, None, |_| {})
            .map_err(|e| format!("the run failed: {e}"))?;
        let (_, kv) = slot(&sess);
        let run = SpilledRun {
            ctx: kv.max_ctx,
            k_fmt: kv.k_fmt,
            v_fmt: kv.v_fmt,
            counts: sess.be.kv_spill_counts(),
            prompt_tokens,
            n_gen: stats.n_gen,
            streamed: sess.be.dense_pager_totals().is_some(),
        };
        let vram = sess.vram();
        let reserve = kv_spill_reserve(
            &model.cfg,
            &model.ecfg,
            kv.max_ctx,
            kv.ubatch,
            sess.be.dense_paged(),
        );
        let measured = kv
            .room_after_weights
            .zip(kv.kv_committed)
            .filter(|_| vram.live && !vram.uma);
        if let Some((room, committed)) = measured {
            let align = KvAlign::of(&sess.be);
            let split = kv_spill_split(
                &model.cfg,
                kv.max_ctx,
                kv.kv_ring,
                kv.ubatch,
                align,
                kv.k_fmt,
                kv.v_fmt,
                1,
                room,
                reserve,
                crate::seam::kv_spill_cap(&model.ecfg),
            );
            let largest = kv
                .kbufs
                .iter()
                .chain(&kv.vbufs)
                .map(|b| align.commit.committed(b.len_bytes() as u64))
                .max()
                .unwrap_or(0);
            let left = room.saturating_sub(committed);
            eprintln!(
                "room after the weights {room}, cache committed {committed}, left {left}, \
                 reserve {reserve}; priced split {split:?}"
            );
            // What the cache itself may not take: a buffer placed in device memory leaves the
            // reserve free. Where the weights already left less than the reserve, the whole cache
            // goes to system RAM.
            ensure(
                run.counts.vram_bufs == 0
                    || left + RESIDUAL_SLACK * run.counts.vram_bufs >= reserve,
                || format!("the spill left {left} bytes free, short of its reserve {reserve}"),
            )?;
            ensure(
                run.counts.vram_bytes.abs_diff(split.vram) <= largest,
                || {
                    format!(
                        "the backend placed {} bytes in device memory, the split priced {}",
                        run.counts.vram_bytes, split.vram
                    )
                },
            )?;
        }
        eprintln!(
            "ctx {} ({:?}/{:?}), ubatch {}, prompt {} tokens, decoded {}, weights streamed {}; \
             {:?}",
            run.ctx,
            run.k_fmt,
            run.v_fmt,
            kv.ubatch,
            run.prompt_tokens,
            run.n_gen,
            run.streamed,
            run.counts
        );
        ensure(kv.ubatch == SPILL_UBATCH, || {
            format!("the pinned {SPILL_UBATCH}-row chunk ran at {}", kv.ubatch)
        })?;
        ensure(stats.n_gen > 0, || "nothing was decoded".to_string())?;
        Ok(run)
    }

    /// B62's first case: under `kv.overflow`, an explicit context neither cache format holds in
    /// device memory beside the weights (Qwen3-0.6B's trained window at [`B12_ROOM`]) opens as
    /// asked and runs a prefill of several chunks and a decode, with part of its cache in system
    /// RAM and nothing later refused ([`spilled_run`]'s checks). In the session's own f16 and with
    /// q8_0 set. On a unified-memory device, where a spill frees nothing, the window is refused
    /// when the session opens, before any weight loads, as it is without `kv.overflow`.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn overflow_spills_an_explicit_ctx_that_does_not_fit() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B is not in the HF cache");
            return;
        };
        let Some(reader) = reader() else {
            return;
        };
        let unified = reader.capabilities().unified_memory;
        for kv in [None, Some(DType::Q8_0)] {
            let run = || -> Result<(), String> {
                let limit = settled_use(reader) + B12_ROOM;
                let probe = SeamModel::load_with(&path, None, overflow_cfg(limit, None, kv))
                    .expect("model load");
                let want = probe.cfg.n_ctx_train;
                let vk = open_backend(None, probe.ecfg.clone()).expect("backend");
                let _scope = crate::seam::enter_backend_scope(&vk);
                let f16 = probe
                    .kv_fit_ctx_fmt(&vk, false)
                    .expect("an attention model");
                let q8 = probe.kv_fit_ctx_fmt(&vk, true).expect("an attention model");
                drop(vk);
                settled_use(reader);
                ensure(f16 < want && q8 < want, || {
                    format!("{want} tokens fit (f16 {f16}, q8_0 {q8}): nothing would spill")
                })?;
                let model = SeamModel::load_with(&path, None, overflow_cfg(limit, Some(want), kv))
                    .expect("model load");
                if unified {
                    let err = match model.vulkan_session(want) {
                        Ok(_) => return Err(format!("kv {kv:?}: {want} opened on unified memory")),
                        Err(e) => e.to_string(),
                    };
                    eprintln!("unified memory, kv {kv:?}: {err}");
                    return ensure(
                        err.contains(&format!("context {want} cannot be served on this device")),
                        || format!("kv {kv:?}: not the refusal before the load: {err}"),
                    );
                }
                let ran = spilled_run(&model, |m| m.vulkan_session(want))
                    .map_err(|e| format!("kv {kv:?}, ctx {want}: {e}"))?;
                ensure(ran.ctx == want, || {
                    format!("asked {want}, served {}", ran.ctx)
                })?;
                let fmt = kv.unwrap_or(DType::F16);
                ensure((ran.k_fmt, ran.v_fmt) == (fmt, fmt), || {
                    format!("served {:?}/{:?}, not {fmt:?}", ran.k_fmt, ran.v_fmt)
                })?;
                ensure(
                    ran.counts.host_bytes > 0 && ran.counts.vram_bytes > 0,
                    || format!("not a partial spill: {:?}", ran.counts),
                )?;
                ensure(!ran.streamed, || "the weights streamed".to_string())
            };
            retry_if_moved(format!("the explicit window, kv {kv:?}"), run)
                .unwrap_or_else(|e| panic!("{e}"));
        }
    }

    /// The room above what is in use, in MiB, at which Qwen3-0.6B's q8_0 cache holds about half the
    /// minimum session context beside its weights under `cfg_at`'s configuration: where the
    /// auto-sized window has nothing usable to clamp to, and a spill at the minimum keeps part of
    /// it in device memory.
    fn half_minimum_room(
        path: &std::path::Path,
        reader: &infr_vulkan::VulkanBackend,
        cfg_at: impl Fn(u64) -> Arc<EngineConfig>,
    ) -> u64 {
        let mut room = 700;
        loop {
            let limit = settled_use(reader) + room * MIB;
            let probe = SeamModel::load_with(path, None, cfg_at(limit)).expect("model load");
            let vk = open_backend(None, probe.ecfg.clone()).expect("backend");
            let _scope = crate::seam::enter_backend_scope(&vk);
            let q8 = probe.kv_fit_ctx_fmt(&vk, true).expect("an attention model");
            if q8 >= MIN_SESSION_CTX / 2 {
                eprintln!("{room} MiB of room: q8_0 holds {q8} tokens");
                return room;
            }
            room += 8;
            assert!(room < 2000, "no room up to 2000 MiB holds half the minimum");
        }
    }

    /// B62's second case: under `kv.overflow`, an auto-sized context for which not even the minimum
    /// fits in either cache format opens at the minimum and runs a prefill of several chunks and a
    /// decode, with part of its cache in system RAM and nothing later refused ([`spilled_run`]'s
    /// checks). On a unified-memory device it is refused when the session opens, as it is without
    /// `kv.overflow`.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn overflow_spills_the_minimum_when_nothing_fits() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B is not in the HF cache");
            return;
        };
        let Some(reader) = reader() else {
            return;
        };
        let unified = reader.capabilities().unified_memory;
        let room = half_minimum_room(&path, reader, |limit| overflow_cfg(limit, None, None));
        let run = || -> Result<(), String> {
            let limit = settled_use(reader) + room * MIB;
            let model = SeamModel::load_with(&path, None, overflow_cfg(limit, None, None))
                .expect("model load");
            if unified {
                let err = match model.vulkan_session_default() {
                    Ok(s) => return Err(format!("opened {} on unified memory", s.max_ctx)),
                    Err(e) => e.to_string(),
                };
                eprintln!(
                    "unified memory, {room} MiB of room: {}",
                    err.lines().next().unwrap_or("")
                );
                return ensure(err.contains("no usable KV cache fits"), || {
                    format!("not the refusal before the load: {err}")
                });
            }
            let ran = spilled_run(&model, |m| m.vulkan_session_default())
                .map_err(|e| format!("{room} MiB of room: {e}"))?;
            ensure(ran.ctx == MIN_SESSION_CTX, || {
                format!("{room} MiB of room: served {}, not the minimum", ran.ctx)
            })?;
            ensure(
                ran.counts.host_bytes > 0 && ran.counts.vram_bytes > 0,
                || format!("{room} MiB of room: not a partial spill: {:?}", ran.counts),
            )?;
            ensure(!ran.streamed, || {
                format!("{room} MiB of room: the weights streamed")
            })
        };
        retry_if_moved(format!("the minimum at {room} MiB"), run).unwrap_or_else(|e| panic!("{e}"));
    }

    /// `Ok` when a run's last prompt logits and decoded tokens (`got`) are the reference's
    /// (`want`) bit for bit, else a [`wrong`] result naming how many differ.
    fn same_result(
        name: &str,
        got: (&[f32], &[u32]),
        want: (&[f32], &[u32]),
    ) -> Result<(), String> {
        if got.1 != want.1 {
            return Err(wrong(format!(
                "{name}: decoded {:?}, resident {:?}",
                got.1, want.1
            )));
        }
        let differ = got
            .0
            .iter()
            .zip(want.0)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        if got.0.len() != want.0.len() || differ > 0 {
            return Err(wrong(format!(
                "{name}: {differ} of {} logits differ from the resident run's",
                want.0.len()
            )));
        }
        Ok(())
    }

    /// A wrong result fails on its first run even when the device's memory moved across it, where
    /// a failure to open is run once more: a planted logits mismatch, through the comparison the
    /// logits test makes, against readings that move by far more than [`IN_USE_MOVED`].
    #[test]
    fn a_wrong_result_is_never_run_again() {
        let moving = || {
            let mut r = 0u64;
            move || {
                r += 64 * MIB;
                r
            }
        };
        let want = (vec![1.0f32, 2.0, 3.0], vec![7u32, 8]);
        let planted = (vec![1.0f32, 2.0, 3.5], vec![7u32, 8]);
        let mut runs = 0;
        let got = retry_if_moved_with(moving(), "planted", || {
            runs += 1;
            same_result("planted", (&planted.0, &planted.1), (&want.0, &want.1))
        });
        assert_eq!(runs, 1, "a logits mismatch was run again: {got:?}");
        assert!(got.is_err_and(|e| e.contains("1 of 3 logits differ")));
        let mut runs = 0;
        let got = retry_if_moved_with(moving(), "planted", || {
            runs += 1;
            same_result("planted", (&want.0, &[7, 9]), (&want.0, &want.1))
        });
        assert_eq!(runs, 1, "a token mismatch was run again: {got:?}");
        let mut runs = 0;
        let _ = retry_if_moved_with(moving(), "open", || -> Result<(), String> {
            runs += 1;
            Err("the session did not open".to_string())
        });
        assert_eq!(
            runs, 2,
            "a failure to open while memory moved runs once more"
        );
        let mut runs = 0;
        let _ = retry_if_moved_with(
            || 0,
            "open",
            || -> Result<(), String> {
                runs += 1;
                Err("the session did not open".to_string())
            },
        );
        assert_eq!(
            runs, 1,
            "a failure on a device that did not move is not run again"
        );
    }

    /// A change to a test's configuration.
    type ConfigEdit = Box<dyn Fn(&mut EngineConfig)>;

    /// Qwen3-0.6B's trained window under `cfg`'s device, generated straight through the runner for
    /// the spill prompt with EOS ignored: the last prompt token's logits, the greedy tokens decoded
    /// after it, and where the backend placed the cache.
    fn spill_logits(
        path: &std::path::Path,
        cfg: Arc<EngineConfig>,
    ) -> Result<(Vec<f32>, Vec<u32>, infr_core::budget::SpillCounts), String> {
        let model = SeamModel::load_with(path, None, cfg).expect("model load");
        let ctx = model.cfg.n_ctx_train;
        let prompt = model
            .encode(&long_prompt(SPILL_PROMPT_WORDS))
            .expect("encode");
        let vk = open_backend(None, model.ecfg.clone()).expect("backend");
        let _scope = crate::seam::enter_backend_scope(&vk);
        let bind =
            crate::seam::vulkan_moe_binder(&vk, &model.gguf, &model.cfg, &model.ecfg, true, ctx)
                .map_err(|e| format!("placement: {e}"))?;
        let (mut logits, mut ids, mut state) = (Vec::new(), Vec::new(), None);
        crate::seam::generate_dense_backend(
            &vk,
            &*bind,
            &model.gguf,
            &model.cfg,
            &model.ecfg,
            model.embd(),
            model.per_layer_embd(),
            &prompt,
            SPILL_DECODE,
            |id| ids.push(id),
            &mut state,
            ctx,
            None,
            None,
            None,
            Some(&mut logits),
            None,
            None,
            None,
        )
        .map_err(|e| format!("the run failed: {e}"))?;
        let counts = vk.kv_spill_counts();
        eprintln!("decoded {ids:?}; {counts:?}");
        Ok((logits, ids, counts))
    }

    /// A spilled cache gives the logits a cache wholly in device memory gives: Qwen3-0.6B's trained
    /// window, the spill prompt prefilled in several chunks and a few tokens decoded, the last
    /// prompt token's logits and the decoded tokens compared bit for bit against the same run with
    /// the cache resident. The spilled layers read every row of the prompt from system RAM. Spilled
    /// once by `kv.overflow_vram_mb` capping the cache's device memory at half of it, on a device
    /// with room for all of it, and once by the room itself ([`B12_ROOM`]), through the placement
    /// and reserve the spill tests check. Not on a unified-memory device, where nothing spills.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn spilled_kv_gives_the_logits_of_a_resident_one() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B is not in the HF cache");
            return;
        };
        let Some(reader) = reader() else {
            return;
        };
        if reader.capabilities().unified_memory {
            eprintln!("skip: nothing spills on a unified-memory device");
            return;
        }
        let trained = SeamModel::load_with(&path, None, limited_cfg(u64::MAX, None))
            .expect("model load")
            .cfg
            .n_ctx_train;
        // Every leg: the trained window set explicitly (so no leg clamps it), the chunk pinned,
        // EOS ignored, and f16 set on both sides (so no leg's placement may pick q8_0).
        let leg = |limit: u64, edit: &dyn Fn(&mut EngineConfig)| {
            let cfg = limited_cfg_with(limit, Some(trained), |c| {
                c.device.ubatch = Some(SPILL_UBATCH);
                c.device.ubatch_specified = true;
                c.sampling.ignore_eos = true;
                c.kv.type_k = Some(DType::F16);
                c.kv.type_k_specified = true;
                c.kv.type_v = Some(DType::F16);
                c.kv.type_v_specified = true;
                edit(c);
            });
            settled_use(reader);
            spill_logits(&path, cfg)
        };
        let (want_logits, want_ids, resident) =
            leg(u64::MAX, &|_| {}).unwrap_or_else(|e| panic!("the resident run: {e}"));
        assert_eq!(
            resident.total_bufs(),
            0,
            "the resident run placed {resident:?}"
        );
        assert_eq!(
            want_ids.len(),
            SPILL_DECODE,
            "the resident run decoded {want_ids:?}"
        );
        let model = SeamModel::load_with(&path, None, limited_cfg(u64::MAX, None)).expect("load");
        let cache = kv_bytes_estimate_fmt(
            &model.cfg,
            trained,
            false,
            SPILL_UBATCH,
            KvAlign::NONE,
            DType::F16,
            DType::F16,
        );
        let half_mib = cache / 2 / MIB;
        let legs: [(&str, u64, ConfigEdit); 2] = [
            (
                "capped at half the cache",
                u64::MAX,
                Box::new(move |c: &mut EngineConfig| {
                    c.kv.overflow = true;
                    c.kv.overflow_vram_mb = Some(half_mib);
                }),
            ),
            (
                "spilled by the room",
                0,
                Box::new(|c: &mut EngineConfig| c.kv.overflow = true),
            ),
        ];
        for (name, limit, edit) in legs {
            let run = || -> Result<(), String> {
                let limit = if limit == 0 {
                    settled_use(reader) + B12_ROOM
                } else {
                    limit
                };
                let (logits, ids, counts) = leg(limit, &*edit)?;
                ensure(counts.host_bytes > 0 && counts.vram_bytes > 0, || {
                    format!("{name}: not a partial spill: {counts:?}")
                })?;
                same_result(name, (&logits, &ids), (&want_logits, &want_ids))
            };
            retry_if_moved(name, run).unwrap_or_else(|e| panic!("{e}"));
        }
    }

    /// What `kv.overflow` cannot serve is refused before a weight is uploaded, with the sizes: an
    /// explicit window on a device whose room, with the whole cache spilled, still leaves the
    /// streamed weights less than the streaming floor (Qwen3-0.6B at 500 MiB of room). On a
    /// unified-memory device, where nothing spills, the refusal is the one without the flag.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn overflow_refuses_what_cannot_stream_before_the_load() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B is not in the HF cache");
            return;
        };
        let Some(reader) = reader() else {
            return;
        };
        let unified = reader.capabilities().unified_memory;
        let want = 4096;
        let run = || -> Result<(), String> {
            let limit = settled_use(reader) + 500 * MIB;
            let model = SeamModel::load_with(&path, None, overflow_cfg(limit, Some(want), None))
                .expect("model load");
            let err = match model.vulkan_session(want) {
                Err(e) => e.to_string(),
                Ok(mut sess) => {
                    let before = sess.be.alloc_room();
                    let err = match model.generate_vulkan_session(&mut sess, "Hi", 1, None, |_| {})
                    {
                        Ok(_) => return Err(format!("{want} ran at 500 MiB of room")),
                        Err(e) => e.to_string(),
                    };
                    let after = sess.be.alloc_room();
                    ensure(after + IN_USE_SLACK >= before, || {
                        format!("the refusal came after {} bytes were taken", before - after)
                    })?;
                    err
                }
            };
            eprintln!("{err}");
            let wording = if unified {
                format!("context {want} cannot be served on this device")
            } else {
                "even with its whole KV cache".to_string()
            };
            ensure(
                err.contains(&wording) && (err.contains(" GB") || err.contains(" GiB")),
                || format!("not the refusal with the sizes: {err}"),
            )
        };
        retry_if_moved("the window that cannot stream", run).unwrap_or_else(|e| panic!("{e}"));
    }

    /// Two serve slots on one backend under `kv.overflow`, with an f16 window per slot that one
    /// cache holds beside the weights at [`B12_ROOM`] and two do not: the engine opens with at
    /// least a whole cache's buffers in device memory (the first slot's, which the ladder serves
    /// whole and allocates first, with no reserve; the backend's tally does not say which slot a
    /// buffer belongs to), the second slot's fork spills past the one spill reserve the backend
    /// holds, the room left after both is at least that reserve, and both slots then run the spill
    /// prompt at once, taking turns on the GPU. One reserve serves every slot because the slots
    /// take turns step by step (`sampling::StepGate`), and a step's activations are released with
    /// it.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn overflow_spills_a_second_slot_past_the_reserve() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B is not in the HF cache");
            return;
        };
        let Some(reader) = reader() else {
            return;
        };
        if reader.capabilities().unified_memory {
            eprintln!("skip: nothing spills on a unified-memory device");
            return;
        }
        let want = 4096;
        let run = || -> Result<(), String> {
            let limit = settled_use(reader) + B12_ROOM;
            let model =
                SeamModel::load_with(&path, None, overflow_cfg(limit, None, Some(DType::F16)))
                    .expect("model load");
            let vk = open_backend(None, model.ecfg.clone()).expect("backend");
            let _scope = crate::seam::enter_backend_scope(&vk);
            let fit = model
                .kv_fit_ctx_fmt(&vk, false)
                .expect("an attention model");
            drop(vk);
            settled_use(reader);
            ensure(want <= fit && fit < 2 * want, || {
                format!("{fit} tokens fit: not one {want}-token cache and short of two")
            })?;
            let model = SeamModel::load_with(
                &path,
                None,
                overflow_cfg(limit, Some(want), Some(DType::F16)),
            )
            .expect("model load");
            let cfg = model.cfg.clone();
            let ecfg = model.ecfg.clone();
            let engine =
                crate::parallel::ParallelSeam::new(model, 2, Some(SizeSpec::Bytes(want as u64)))
                    .map_err(|e| format!("the two-slot engine did not open: {e}"))?;
            let be = engine.backend();
            let placed = be.kv_spill_counts();
            let reserve = kv_spill_reserve(&cfg, &ecfg, want, SPILL_UBATCH, false);
            let left = be.alloc_room();
            eprintln!("two slots of {want}: {placed:?}; {left} left, reserve {reserve}");
            ensure(placed.host_bytes > 0, || {
                format!("nothing spilled: {placed:?}")
            })?;
            ensure(placed.vram_bufs >= 2 * cfg.n_layer as u64, || {
                format!("less than a whole cache in device memory: {placed:?}")
            })?;
            ensure(left + RESIDUAL_SLACK * placed.vram_bufs >= reserve, || {
                format!("the slots left {left} bytes free, short of the reserve {reserve}")
            })?;
            let greedy = crate::sampling::RequestSampling {
                temp: Some(0.0),
                ..Default::default()
            };
            let prompt = long_prompt(SPILL_PROMPT_WORDS);
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..2)
                    .map(|_| {
                        let req = engine.request_ctx(greedy.clone());
                        let (engine, prompt) = (&engine, &prompt);
                        s.spawn(move || engine.generate(prompt, SPILL_DECODE, None, &req, |_| {}))
                    })
                    .collect();
                handles.into_iter().try_for_each(|h| {
                    let stats = h
                        .join()
                        .expect("sequence thread")
                        .map_err(|e| format!("a slot's run failed: {e}"))?;
                    ensure(stats.n_gen > 0, || "a slot decoded nothing".to_string())
                })
            })
        };
        retry_if_moved("two slots", run).unwrap_or_else(|e| panic!("{e}"));
    }

    /// How far below the room where placement's spill rung keeps the weights resident the
    /// streamed-spill test runs. Measured on the RX 7900 XTX on 2026-11-25 with Qwen3-1.7B: the
    /// rung took the weights resident from about 1940 MiB of room, and the arenas held their floor
    /// with the whole cache spilled down to about 1810.
    const STREAM_SPILL_BELOW_RESIDENT: u64 = 64 * MIB;

    /// The room above what is in use, in MiB, at which placement's spill rung would keep the model
    /// at `path` resident with `want` tokens spilled, priced as the rung prices it: the weights at
    /// the least the arena commits for them and the upload staging (`resident_load_held`), and the
    /// spill reserve at the spilled chunk. Read off one limited backend: what its room holds less
    /// what the rung needs.
    fn spill_resident_room(
        path: &std::path::Path,
        reader: &infr_vulkan::VulkanBackend,
        want: usize,
    ) -> u64 {
        const PROBE_ROOM: u64 = 2048 * MIB;
        let limit = settled_use(reader) + PROBE_ROOM;
        let model = SeamModel::load_with(path, None, overflow_cfg(limit, Some(want), None))
            .expect("model load");
        let vk = open_backend(None, model.ecfg.clone()).expect("backend");
        let _scope = crate::seam::enter_backend_scope(&vk);
        let (held, staging) =
            crate::seam::resident_load_held(&vk, &model.gguf, &model.cfg, &model.ecfg);
        let ubatch = crate::seam::spill_ubatch(&model.ecfg);
        let need = held + staging + kv_spill_reserve(&model.cfg, &model.ecfg, want, ubatch, false);
        let room = vk.alloc_room();
        (PROBE_ROOM + need).saturating_sub(room) / MIB
    }

    /// A model whose weights do not fit the device even without its cache streams them, with the
    /// whole cache in system RAM, when the cache would leave the streaming arenas less than their
    /// floor: Qwen3-1.7B under `kv.overflow`, its trained window set explicitly, the spill prompt
    /// and a short decode, at [`STREAM_SPILL_BELOW_RESIDENT`] below the room where the weights
    /// would stay resident ([`spill_resident_room`]). Skips, saying so, where that room leaves the
    /// arenas short of their floor too (no streamed spill exists between the two on that device).
    /// Not on a unified-memory device, where nothing spills.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn overflow_streams_weights_beside_a_wholly_spilled_cache() {
        let Some(path) =
            crate::util::test_hub_gguf("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf")
        else {
            eprintln!("skip: Qwen3-1.7B is not in the HF cache");
            return;
        };
        let Some(reader) = reader() else {
            return;
        };
        if reader.capabilities().unified_memory {
            eprintln!("skip: nothing spills on a unified-memory device");
            return;
        }
        let want = SeamModel::load_with(&path, None, overflow_cfg(u64::MAX, None, None))
            .expect("model load")
            .cfg
            .n_ctx_train;
        let resident = spill_resident_room(&path, reader, want);
        let room = resident * MIB - STREAM_SPILL_BELOW_RESIDENT;
        eprintln!(
            "weights resident from {resident} MiB of room; running at {} MiB",
            room / MIB
        );
        let run = || -> Result<bool, String> {
            let limit = settled_use(reader) + room;
            let model = SeamModel::load_with(&path, None, overflow_cfg(limit, Some(want), None))
                .expect("model load");
            let ran = match spilled_run(&model, |m| m.vulkan_session(want)) {
                Err(e) if e.contains("below the streaming floor") => {
                    eprintln!("skip: no streamed spill below the resident room here: {e}");
                    return Ok(false);
                }
                ran => ran?,
            };
            ensure(ran.ctx == want, || {
                format!("asked {want}, served {}", ran.ctx)
            })?;
            ensure(ran.streamed, || "the weights did not stream".to_string())?;
            ensure(
                ran.counts.host_bytes > 0 && ran.counts.vram_bytes == 0,
                || format!("not a whole spill: {:?}", ran.counts),
            )?;
            Ok(true)
        };
        retry_if_moved("the streamed spill", run).unwrap_or_else(|e| panic!("{e}"));
    }

    /// Set in the child process [`a_streamed_spill_is_priced_before_an_automatic_host_tier`] runs
    /// itself in, under a host memory limit.
    const SPILL_BESIDE_TIER_CHILD: &str = "INFR_TEST_SPILL_BESIDE_TIER_CHILD";

    /// The line the child prints when it ran the whole check, which the parent requires.
    const SPILL_BESIDE_TIER_RAN: &str = "spill beside tier: the child ran the check";

    /// The prefix of the line the child prints, with the reason, when it skipped.
    const SPILL_BESIDE_TIER_SKIP: &str = "spill beside tier: the child skipped: ";

    /// The memory limit of the child's cgroup: what makes the host's spare memory small enough for
    /// a spilled Qwen3-1.7B window to need most of it, on a machine with far more free.
    const SPILL_BESIDE_TIER_HOST: u64 = 8 << 30;

    /// How much of what the host can spare the child leaves to the weights' host tier beside the
    /// spill: the automatic arena it should build, above `infr_core::hostmem`'s worth-building
    /// floor, and the drift in host memory between the child's reading and placement's it absorbs
    /// (half of a drop in available memory comes off the spare).
    const SPILL_BESIDE_TIER_ARENA: u64 = 512 * MIB;

    /// Plan 15 R0: a session whose weights stream with the whole cache in system RAM
    /// (`kv.overflow`) prices the spill before the weights' automatic host tier, so a spill that
    /// fits what the host can spare loads beside an arena sized from what it leaves. Qwen3-1.7B,
    /// as [`overflow_streams_weights_beside_a_wholly_spilled_cache`] runs it, at the window whose
    /// spill leaves [`SPILL_BESIDE_TIER_ARENA`] of the spare, where sizing the arena first takes the
    /// spill's room and refuses the load. The host's memory is made small by running the test
    /// again in a child process under a cgroup limit (`systemd-run --user --scope`,
    /// [`SPILL_BESIDE_TIER_HOST`]); skips, saying so, where that cannot start or the limit does
    /// not reach the host-memory probe. Not on a unified-memory device, where nothing spills. The
    /// parent passes only when the child says it ran the check ([`SPILL_BESIDE_TIER_RAN`]), and
    /// reports a child's skip as a skip.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn a_streamed_spill_is_priced_before_an_automatic_host_tier() {
        const NAME: &str = "seam::model::vram_limit_tests::tests::\
                            a_streamed_spill_is_priced_before_an_automatic_host_tier";
        let Some(path) =
            crate::util::test_hub_gguf("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf")
        else {
            eprintln!("skip: Qwen3-1.7B is not in the HF cache");
            return;
        };
        if std::env::var_os(SPILL_BESIDE_TIER_CHILD).is_none() {
            let exe = std::env::current_exe().expect("the test binary");
            let limit = format!("MemoryMax={SPILL_BESIDE_TIER_HOST}");
            let out = std::process::Command::new("systemd-run")
                .args(["--user", "--scope", "--quiet", "-p", &limit, "--"])
                .arg(exe)
                .args([NAME, "--exact", "--include-ignored", "--nocapture"])
                .args(["--test-threads", "1"])
                .env(SPILL_BESIDE_TIER_CHILD, "1")
                .stdin(std::process::Stdio::null())
                .output();
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    eprintln!("skip: systemd-run did not start ({e}), so no host limit");
                    return;
                }
            };
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!("child stdout:\n{stdout}\nchild stderr:\n{stderr}");
            assert!(
                !stdout.contains("running 0 tests"),
                "the child's filter {NAME} matched no test"
            );
            if let Some(reason) = stderr
                .lines()
                .find_map(|l| l.strip_prefix(SPILL_BESIDE_TIER_SKIP))
            {
                eprintln!("skip: the child under {limit} skipped: {reason}");
                return;
            }
            if stdout.contains("running 1 test") {
                assert!(
                    stdout.contains("test result: ok. 1 passed")
                        && stderr.lines().any(|l| l == SPILL_BESIDE_TIER_RAN),
                    "the child under {limit} failed, or passed without running the check: {}",
                    out.status
                );
                return;
            }
            eprintln!(
                "skip: systemd-run did not start the child under {limit}: {}",
                out.status
            );
            return;
        }
        let skip = |reason: &str| eprintln!("{SPILL_BESIDE_TIER_SKIP}{reason}");
        let Some(reader) = reader() else {
            skip("no Vulkan backend opened on the test environment's device");
            return;
        };
        if reader.capabilities().unified_memory {
            skip("nothing spills on a unified-memory device");
            return;
        }
        // `MemoryMax` is ignored where the user's systemd does not delegate the memory
        // controller; the probe then reads the whole host, and nothing here would mean anything.
        match infr_plat::mem::available() {
            Some(a)
                if a.source == infr_plat::mem::Source::LinuxCgroupClamped
                    && a.bytes <= SPILL_BESIDE_TIER_HOST => {}
            other => {
                skip(&format!(
                    "the cgroup limit did not reach the host-memory probe ({other:?}); is the \
                     memory controller delegated to the user's systemd?"
                ));
                return;
            }
        }
        let probe = SeamModel::load_with(&path, None, overflow_cfg(u64::MAX, None, None))
            .expect("model load");
        let trained = probe.cfg.n_ctx_train;
        let align = KvAlign::of(reader);
        let (k, v) = (DType::F16, DType::F16);
        let ring = crate::seam::placement_ring(&probe.cfg, &probe.ecfg, k, v);
        // The whole cache in system RAM, as placement prices a streamed spill.
        let host_part = |ctx: usize| {
            kv_spill_split(
                &probe.cfg,
                ctx,
                ring,
                SPILL_UBATCH,
                align,
                k,
                v,
                1,
                0,
                0,
                None,
            )
            .host
        };
        // The streamed weights, near enough to say whether an arena sized before the spill takes
        // its room: every layer tensor.
        let streamable: u64 = infr_core::WeightSource::tensors(&probe.gguf)
            .iter()
            .filter(|t| t.name.starts_with("blk."))
            .map(|t| t.nbytes as u64)
            .sum();
        let available = infr_core::hostmem::available_bytes().expect("a host-memory probe");
        let spare = infr_core::hostmem::spare_bytes(available, 0);
        let target = spare.saturating_sub(SPILL_BESIDE_TIER_ARENA);
        // The longest window whose spill is at most the target.
        let want = {
            let (mut lo, mut hi) = (MIN_SESSION_CTX, trained);
            assert!(
                host_part(lo) <= target && host_part(hi) > target,
                "{available} bytes available: no window between {lo} and {hi} spills {target}"
            );
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if host_part(mid) <= target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        let spill = host_part(want);
        let old_arena = spare.min(streamable);
        eprintln!(
            "{available} bytes available, {spare} spare; ctx {want} spills {spill}; an arena sized \
             first would take {old_arena} and leave {}",
            infr_core::hostmem::spare_bytes(available, old_arena)
        );
        assert!(
            infr_core::hostmem::spare_bytes(available, old_arena) < spill,
            "this window would load with the arena sized first: it tests nothing"
        );
        drop(probe);
        let resident = spill_resident_room(&path, reader, want);
        let room = resident * MIB - STREAM_SPILL_BELOW_RESIDENT;
        let run = || -> Result<(), String> {
            let limit = settled_use(reader) + room;
            let model = SeamModel::load_with(&path, None, overflow_cfg(limit, Some(want), None))
                .expect("model load");
            let ran = spilled_run(&model, |m| m.vulkan_session(want))?;
            ensure(ran.ctx == want, || {
                format!("asked {want}, served {}", ran.ctx)
            })?;
            ensure(ran.streamed, || "the weights did not stream".to_string())?;
            ensure(
                ran.counts.vram_bytes == 0 && ran.counts.host_bytes == spill,
                || {
                    format!(
                        "placement priced {spill} bytes in system RAM: {:?}",
                        ran.counts
                    )
                },
            )
        };
        retry_if_moved("the spill beside the host tier", run).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("{SPILL_BESIDE_TIER_RAN}");
    }

    /// gemma-4-12b-it Q4_K_M, the largest dense model the GPU tests use, or `None` (saying so)
    /// when it is not in the HF cache. Not required by `INFR_TEST_REQUIRE_MODELS`: the gate runs
    /// without it on a box that lacks it.
    fn gemma4_12b() -> Option<std::path::PathBuf> {
        let found = infr_hub::Store::discover().ok().and_then(|s| {
            s.snapshot_file("unsloth/gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf")
        });
        if found.is_none() {
            eprintln!("skip: unsloth/gemma-4-12b-it-GGUF is not in the HF cache");
        }
        found
    }

    /// The (room in MiB, pinned chunk) pairs `auto_ctx_prefills_full_chunks_on_a_large_model`
    /// requires to run: every room at the default chunk, and every room but the tightest at a
    /// 1024-row chunk, which ran with windows of 1 241 to 3 072 tokens across runs here, on either
    /// side of the prompt.
    const LARGE_MODEL_MUST_RUN: [(u64, Option<usize>); 7] = [
        (9800, None),
        (10500, None),
        (11500, None),
        (12800, None),
        (10500, Some(1024)),
        (11500, Some(1024)),
        (12800, Some(1024)),
    ];

    /// More tokens than [`long_prompt`]`(1600)` tokenizes to on gemma-4 (1607 measured), with the
    /// tokens decoded after it.
    const PROMPT_TOKENS_MAX: usize = 1700;

    /// A prompt of `words` words, several prefill chunks long at any chunk height the sessions
    /// below run at.
    fn long_prompt(words: usize) -> String {
        let vocab = [
            "river", "mountain", "cloud", "seven", "blue", "market", "window", "paper", "engine",
        ];
        let body: Vec<&str> = (0..words)
            .map(|i| vocab[(i * 7 + i / 3) % vocab.len()])
            .collect();
        format!("Summarize this in one sentence: {}", body.join(" "))
    }

    /// The activation reserve as estimated in bytes ([`crate::seam::dense_act_reserve_at`]) covers
    /// the allocator blocks the VRAM guard charges whole, on a run that opens several of them:
    /// gemma-4-12b's auto-sized session, on a limited device where the context clamps, prefills a
    /// prompt of many chunks and decodes, at the default chunk height and at a pinned 1024-row
    /// chunk (whose activations peaked at 1.12 GB here), with no allocation refused. The fit holds
    /// the reserve back in bytes and the guard charges a new shared block in full, so what covers
    /// the blocks' rounding is the estimate's own margin over the activations it predicts. Every
    /// room in [`LARGE_MODEL_MUST_RUN`] must load, prefill and decode at the chunk height it
    /// names; the tightest room at the pinned chunk may instead leave a window shorter than the
    /// prompt (before the load, or as the re-clamp after it keeps it), which is not run. That more
    /// than one shared block opened is not asserted: the blocks a prefill opens are freed with its
    /// scratch, and the backend keeps no count of the blocks opened.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn auto_ctx_prefills_full_chunks_on_a_large_model() {
        let Some(path) = gemma4_12b() else {
            return;
        };
        let Some(reader) = reader() else {
            return;
        };
        if reader.capabilities().integrated {
            // Measured: about seven minutes on the 2-CU integrated GPU here, past what the gate
            // can spend; the blocks the guard charges are the same on every device.
            eprintln!("skip: a 12B prefill sweep on an integrated GPU (run it on a discrete one)");
            return;
        }
        let prompt = long_prompt(1600);
        let mut ran = Vec::new();
        let mut failed = Vec::new();
        for ubatch in [None, Some(1024)] {
            for room in [9800, 10500, 11500, 12800] {
                // `Ok(true)` when the room ran the prompt, `Ok(false)` when its window is shorter
                // than the prompt (which is not run), `Err` when it failed.
                let run = || -> Result<bool, String> {
                    let used = settled_use(reader);
                    let cfg = limited_cfg_with(used + room * MIB, None, |c| {
                        c.device.ubatch = ubatch;
                        c.device.ubatch_specified = ubatch.is_some();
                    });
                    let model = SeamModel::load_with(&path, None, cfg).expect("model load");
                    let mut sess = match model.vulkan_session_default() {
                        Ok(sess) => sess,
                        Err(e) if e.to_string().contains("no usable KV cache fits") => {
                            eprintln!("{room} MiB, chunk {ubatch:?}: refused before the load");
                            return Ok(false);
                        }
                        Err(e) => {
                            return Err(format!("{room} MiB, chunk {ubatch:?}: did not open: {e}"))
                        }
                    };
                    if sess.max_ctx < PROMPT_TOKENS_MAX {
                        eprintln!(
                            "{room} MiB, chunk {ubatch:?}: a {}-token window, shorter than the \
                             prompt",
                            sess.max_ctx
                        );
                        return Ok(false);
                    }
                    match model.generate_vulkan_session(&mut sess, &prompt, 8, None, |_| {}) {
                        Ok(stats) => {
                            let (_, kv) = slot(&sess);
                            eprintln!(
                                "{room} MiB, chunk {ubatch:?}: ctx {} at a {}-row chunk, \
                                 prefilled {} tokens, decoded {}",
                                kv.max_ctx, kv.ubatch, stats.n_prompt, stats.n_gen
                            );
                            if let Some(pinned) = ubatch {
                                ensure(kv.ubatch == pinned, || {
                                    format!(
                                        "{room} MiB: the pinned chunk {pinned} did not take \
                                         effect ({} rows)",
                                        kv.ubatch
                                    )
                                })?;
                            }
                            Ok(true)
                        }
                        // The window the post-load re-clamp kept, shorter than the prompt: a
                        // context the room cannot hold, not an allocation the guard refused.
                        Err(e) if e.to_string().contains("exceeds the session KV capacity") => {
                            eprintln!("{room} MiB, chunk {ubatch:?}: {e}");
                            Ok(false)
                        }
                        Err(e) => Err(format!("{room} MiB, chunk {ubatch:?}: {e}")),
                    }
                };
                match retry_if_moved(format!("{room} MiB, chunk {ubatch:?}"), run) {
                    Ok(true) => ran.push((room, ubatch)),
                    Ok(false) => {}
                    Err(e) => failed.push(e),
                }
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
        let missing: Vec<_> = LARGE_MODEL_MUST_RUN
            .iter()
            .filter(|must| !ran.contains(must))
            .collect();
        assert!(
            missing.is_empty(),
            "rooms (MiB, pinned chunk) that must run did not: {missing:?}; ran {ran:?}"
        );
    }
}
