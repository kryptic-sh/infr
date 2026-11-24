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
        kv_bytes_estimate_fmt, kv_recurrent_alloc_bytes, KvAlign, SeamKv, MIN_SESSION_CTX,
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

    /// How far [`usable_room`] raises a room that left too small a context.
    const ROOM_STEP: u64 = 32;

    /// The test environment's configuration (the device it names and the GPU knobs every GPU test
    /// honours) with the device memory capped at `limit` bytes and, when given, an explicit context.
    fn limited_cfg(limit: u64, ctx: Option<usize>) -> Arc<EngineConfig> {
        let env = EngineConfig::load_from_layers(&[
            infr_core::config::ConfigLayer::env().expect("test environment config")
        ]);
        let mut cfg = EngineConfig::default();
        cfg.device.dev = env.device.dev;
        infr_testkit::honour_env(&mut cfg);
        cfg.kernels.vulkan.vram_limit = Some(SizeSpec::Bytes(limit));
        cfg.device.ctx = ctx.map(|c| SizeSpec::Bytes(c as u64));
        Arc::new(cfg)
    }

    /// The model at `path` loaded under [`limited_cfg`].
    fn load(path: &std::path::Path, limit: u64, ctx: Option<usize>) -> SeamModel {
        SeamModel::load_with(path, None, limited_cfg(limit, ctx)).expect("model load")
    }

    /// The device memory in use on the test environment's device, by this process and every other,
    /// or `None` (with the skip line printed) when no backend opens there.
    fn in_use() -> Option<u64> {
        let vk = infr_testkit::gpu_or_skip_noted(open_backend(None, limited_cfg(u64::MAX, None)))?;
        let v = vk.vram();
        Some(v.total - v.available)
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
        /// The model and the bytes in use the run's limit was set above, to run it again.
        path: std::path::PathBuf,
        used: u64,
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

    /// Open the default (auto-sized) session of the model at `path` with the device limited to
    /// `room_mib` above the `used` bytes in use, check that the context was clamped below the
    /// trained window, generate a few tokens, and report what the slot allocated — or why the
    /// generation failed.
    fn clamped_run(path: &std::path::Path, used: u64, room_mib: u64) -> Result<ClampedRun, String> {
        let model = load(path, used + room_mib * MIB, None);
        let mut sess = model.vulkan_session_default().unwrap_or_else(|e| {
            panic!("{room_mib} MiB of room: the default session did not open: {e}")
        });
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
            return Err(format!("{room_mib} MiB of room: {e}"));
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
            used,
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
    /// `path` runs with a clamped context of at least [`MIN_SESSION_CTX`]. Measured rather than
    /// computed, because what the weights take beyond their footprint is only known once they are
    /// resident; below it the clamp either refuses or keeps an unusable context
    /// (docs/backlog.md B-PRELOAD-CLAMP).
    fn usable_room(path: &std::path::Path, used: u64, start: u64, hi: u64) -> u64 {
        let mut room = start;
        while room < hi {
            if clamped_run(path, used, room).is_ok_and(|r| r.slot_ctx >= MIN_SESSION_CTX) {
                return room;
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
    fn clamped_sweeps() -> Option<Vec<Result<ClampedRun, String>>> {
        let used = in_use()?;
        let mut runs = Vec::new();
        for (name, path, start, hi, seed) in [
            ("Qwen3-0.6B", qwen3_06b(), 1150, 2600, 0x5134),
            ("Qwen3.5-0.8B", qwen35_08b(), 1350, 3200, 0x5135),
        ] {
            let Some(path) = path else {
                eprintln!("skip: {name} is not in the HF cache");
                continue;
            };
            let lo = usable_room(&path, used, start, hi);
            runs.extend(
                sweep(lo, hi, seed)
                    .into_iter()
                    .map(|room| clamped_run(&path, used, room)),
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
                    failed.push(e);
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
            match clamped_run(&run.path, run.used, run.room_mib) {
                Ok(again) => failed.extend(past_bound(&again)),
                Err(e) => failed.push(e),
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
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
        let Some(used) = in_use() else {
            return;
        };
        let limit = used + B12_ROOM;
        let probe = load(&path, limit, None);
        let vk = open_backend(None, probe.ecfg.clone()).expect("backend");
        let f16 = probe
            .kv_fit_ctx_fmt(&vk, false)
            .expect("an attention model");
        let q8 = probe.kv_fit_ctx_fmt(&vk, true).expect("an attention model");
        drop(vk);
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

    /// The runner's refusal after the weights load (B12): the largest explicit f16 window the fit
    /// accepts before the load is the one the weights' unpriced memory (arena block tails, staging,
    /// the driver's own) pushes past what is left after it. Placement takes it as resident and the
    /// refusal before the load lets it through, so it is refused between the weight upload and the KV
    /// allocation, with the sizes — not by the allocation guard on one layer's buffer.
    #[test]
    #[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
    fn explicit_ctx_short_of_the_room_after_the_weights_is_refused() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B is not in the HF cache");
            return;
        };
        let Some(used) = in_use() else {
            return;
        };
        let limit = used + B12_ROOM;
        let probe = load(&path, limit, None);
        let vk = open_backend(None, probe.ecfg.clone()).expect("backend");
        let want = probe
            .kv_fit_ctx_fmt(&vk, false)
            .expect("an attention model");
        drop(vk);

        let model = load(&path, limit, Some(want));
        let mut sess = model
            .vulkan_session(want)
            .expect("the pre-load fit accepts its own largest window");
        let err = model
            .generate_vulkan_session(&mut sess, "The capital of France is", 4, None, |_| {})
            .expect_err("the window must not run once the weights' own memory is counted");
        eprintln!("asked {want}: {err}");
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("context {want} cannot be served"))
                && msg.contains("after the weights"),
            "the refusal after the load, not another failure: {msg}"
        );
    }
}
