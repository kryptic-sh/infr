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

    use infr_core::{DType, SizeSpec};

    use super::super::{open_backend, DenseVulkanSession, SeamModel};
    use crate::seam::{
        dense_act_reserve_at, kv_bytes_estimate_fmt, kv_recurrent_alloc_bytes, KvAlign, SeamKv,
        MIN_SESSION_CTX,
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
    /// session refused to open or the generation failed.
    fn clamped_run(path: &std::path::Path, room_mib: u64) -> Result<ClampedRun, RunErr> {
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
    /// A room that fails while the device's memory in use moved by more than [`IN_USE_SLACK`]
    /// across its run (another process on the GPU: observed once on the RX 7900 XTX, 77 MB taken
    /// during one run and kept) is run once more, and fails the test only if it fails again.
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
                let before = in_use().expect("a backend opened for the sweep");
                let (mut row, mut fails) = judge(room, auto_run(&path, room));
                if !fails.is_empty() {
                    let moved = in_use().expect("a backend").abs_diff(before);
                    if moved > IN_USE_SLACK {
                        eprintln!(
                            "{name} at {room} MiB: {row}; the device's memory in use moved {moved} \
                             bytes across the run, so it runs once more"
                        );
                        (row, fails) = judge(room, auto_run(&path, room));
                    }
                }
                failed.extend(fails);
                eprintln!("{name} at {room} MiB: {row}");
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
        use infr_core::backend::{Backend, BufferUsage};
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
        let used = settled_use(reader);
        let limit = used + B12_ROOM;
        let probe = load(&path, limit, None);
        let vk = open_backend(None, probe.ecfg.clone()).expect("backend");
        let f16 = probe
            .kv_fit_ctx_fmt(&vk, false)
            .expect("an attention model");
        let q8 = probe.kv_fit_ctx_fmt(&vk, true).expect("an attention model");
        drop(vk);
        // What the probe backend freed lands before the session's own backend measures the room.
        settled_use(reader);
        let want = (f16 + q8) / 2;
        assert!(
            f16 < want && want < q8,
            "f16 {f16}, q8_0 {q8}: no window between them"
        );

        let model = load(&path, limit, Some(want));
        let mut sess = model.vulkan_session(want).expect("session");
        // The explicit-context rung pins q8_0 when the session opens, before any weight loads; the
        // dense placement's own q8_0 rung would also catch this window later, so the served format
        // alone cannot say which rung decided.
        let scope = crate::seam::PlacementScope::enter(sess.pins().clone());
        let pinned = crate::seam::kv_auto_q8();
        drop(scope);
        assert!(
            pinned,
            "the session opened without the explicit-context q8_0 pin"
        );
        model
            .generate_vulkan_session(&mut sess, "The capital of France is", 4, None, |_| {})
            .expect("the q8_0 window runs");
        let (_, kv) = slot(&sess);
        eprintln!(
            "f16 fits {f16}, q8_0 fits {q8}; asked {want}, served {} at {:?}/{:?}",
            kv.max_ctx, kv.k_fmt, kv.v_fmt
        );
        assert_eq!(kv.max_ctx, want, "an explicit context is served as asked");
        assert_eq!((kv.k_fmt, kv.v_fmt), (DType::Q8_0, DType::Q8_0));
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
        let used = settled_use(reader);
        let limit = used + B12_ROOM;
        let probe = load(&path, limit, None);
        // Both windows are taken at the edge of a room `IN_USE_SLACK` smaller than the sessions
        // get, so the desktop moving between this reading and theirs cannot push either one past
        // its edge.
        let vk = open_backend(None, limited_cfg(limit - IN_USE_SLACK, None)).expect("backend");
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
        // What the probe backend freed lands before the session's own backend measures the room.
        settled_use(reader);
        assert!(
            fits < placed,
            "the fit ({fits}) must be below what the bare footprint leaves ({placed})"
        );

        let model = load(&path, limit, Some(fits));
        let mut sess = model
            .vulkan_session(fits)
            .expect("the pre-load fit accepts its own largest window");
        model
            .generate_vulkan_session(&mut sess, "The capital of France is", 4, None, |_| {})
            .expect("the largest window the fit accepts runs");
        let (_, kv) = slot(&sess);
        assert_eq!(kv.max_ctx, fits, "an explicit context is served as asked");
        assert_eq!((kv.k_fmt, kv.v_fmt), (DType::F16, DType::F16));
        drop(sess);
        // What the first session freed lands before the second one's placement measures the room.
        settled_use(reader);

        let f16 = limited_cfg_with(limit, Some(placed), |c| {
            c.kv.type_k = Some(DType::F16);
            c.kv.type_k_specified = true;
            c.kv.type_v = Some(DType::F16);
            c.kv.type_v_specified = true;
        });
        let model = SeamModel::load_with(&path, None, f16).expect("model load");
        let mut sess = model
            .vulkan_session(placed)
            .expect("the refusal before the load counts the cache alone");
        let err = model
            .generate_vulkan_session(&mut sess, "The capital of France is", 4, None, |_| {})
            .expect_err("the window must not run once the weights' own memory is counted");
        eprintln!("fit {fits}; asked {placed} in f16: {err}");
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("context {placed} cannot be served"))
                && msg.contains("after the weights"),
            "the refusal after the load, not another failure: {msg}"
        );
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
        use infr_core::backend::Backend;
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
                        continue;
                    }
                    Err(e) => {
                        failed.push(format!("{room} MiB, chunk {ubatch:?}: did not open: {e}"));
                        continue;
                    }
                };
                if sess.max_ctx < PROMPT_TOKENS_MAX {
                    eprintln!(
                        "{room} MiB, chunk {ubatch:?}: a {}-token window, shorter than the prompt",
                        sess.max_ctx
                    );
                    continue;
                }
                match model.generate_vulkan_session(&mut sess, &prompt, 8, None, |_| {}) {
                    Ok(stats) => {
                        let (_, kv) = slot(&sess);
                        eprintln!(
                            "{room} MiB, chunk {ubatch:?}: ctx {} at a {}-row chunk, prefilled {} \
                             tokens, decoded {}",
                            kv.max_ctx, kv.ubatch, stats.n_prompt, stats.n_gen
                        );
                        if let Some(pinned) = ubatch {
                            assert_eq!(
                                kv.ubatch, pinned,
                                "{room} MiB: the pinned chunk did not take effect"
                            );
                        }
                        ran.push((room, ubatch));
                    }
                    // The window the post-load re-clamp kept, shorter than the prompt: a context
                    // the room cannot hold, not an allocation the guard refused.
                    Err(e) if e.to_string().contains("exceeds the session KV capacity") => {
                        eprintln!("{room} MiB, chunk {ubatch:?}: {e}");
                    }
                    Err(e) => failed.push(format!("{room} MiB, chunk {ubatch:?}: {e}")),
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
