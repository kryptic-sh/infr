# infr Codebase Audit

Full module-by-module audit for bugs, correctness, perf, DRY, and YAGNI.
Delegated per-module to Opus agents; every finding below has been independently
verified against the source by the coordinator.

- **Repo state at audit start:** `main` @ `5457dbe`
- **Audit date:** 2026-07-21
- **Severity:** 🔴 critical (correctness/crash/UB) · 🟠 major (perf/latent bug)
  · 🟡 minor (DRY/YAGNI/style/robustness)

Each finding: `file:line` · severity · summary · why · suggested fix. Findings
that a delegated agent raised but the coordinator could **not**
reproduce/confirm are dropped (not listed) to keep this a verified-only ledger.

---

## Summary

**Living ledger — findings are pruned from this file as their fix lands on
`main` (TDD, one module slice at a time).**

- **Original audit:** 157 findings across 24 module slices (1 🔴 critical, 33 🟠
  major, 123 🟡 minor).
- **Remaining open:** **123** — 0 🔴, 22 🟠, 101 🟡.

No finding was accepted on an agent's word — each was re-read against the source
by the coordinator; two agent-flagged "MAJOR"s (the Q5_1 clamp in the shader and
CPU quantizers) were **downgraded** to defensive-only after verifying the
overflow is unreachable, and one MTP off-by-one is marked **PLAUSIBLE** (real
code smell, could not fully confirm the position convention without running the
parked path).

### Resolved (landed on `main`)

- **`infr-hub` (all 6 findings)** — TDD, +10 tests. `1263bcc` verifies
  downloaded blobs against HF's expected LFS sha256 (the 🔴 critical); the slice
  adds `If-Range` resume (no stale-partial splice), an advisory `flock`
  serializing concurrent pulls, full split-shard (`-NNNNN-of-MMMMM`)
  download/relink, one shared `pick_gguf` selection for download+cache (kills
  the re-download loop) that excludes `mmproj`/float-master fallbacks,
  `refs/main` snapshot preference, trailing-colon ref parsing, and verify-once
  hashing.
- **`infr-llama` sampling + grammar (all 6 findings)** — TDD, +8 tests incl.
  byte-identity characterization guards on the default greedy/top-k paths.
  Repeat penalty is now per-distinct-token (was `repeat^K`); grammar
  `apply_mask` masks the padded-vocab tail to `-inf`; constrained decoding
  honors the `Sampler` (greedy at temp==0, stays inside the grammar mask);
  `seed|1` no longer collapses adjacent seeds; the `top_k==0` path uses a heap
  instead of a ~150K full sort; `truncated_dist`/`sample_logits` share one
  `truncated_softmax` helper. _Follow-up noted:_ `seed_rng()` (INFR_SEED path)
  has the same latent `|1` — deferred to avoid perturbing INFR_SEED determinism.
- **`infr-gguf` (all 5 findings)** — TDD, +9 tests. Corrupt/truncated GGUFs now
  return `Error::Loader` instead of panicking (`checked_add`/`checked_mul` in
  `ensure`/shape/offsets; `with_capacity` clamped to remaining bytes;
  non-pow2/zero `alignment` rejected). Host affine dequant is a single fused
  pass (no `sc`/`mn` materialization) — **bit-identical** (characterization
  tests on Q4_K/Q6_K assert `to_bits()` equality). `dequant_factored` returns
  `Result` (was `unreachable!`); `apply_signs`/`Gguf::resolve` de-duplicate the
  IQ sign loops and tensor-bounds lookup.
- **`infr-server` (all 6 findings)** — TDD, +13 tests. Streaming now emits a
  terminal SSE `error` frame on failure (never a fake `stop`) and a `DoneGuard`
  flushes `[DONE]` even on panic; a per-request `AtomicBool` cancel latch (set
  when the client's SSE `send` fails) is polled in `run_chat` → `req.abort()`,
  so a disconnected stream frees its GPU slot instead of running to
  `max_tokens`; `usage` carries real prompt/completion counts
  (`total=prompt+completion`); `make_id` appends a monotonic counter (no
  `--parallel` id collisions); optional `INFR_API_KEY` bearer auth (default
  open) + `max_tokens` clamp; `tools` passed as a borrowed `&Value` (no
  round-trip) and a malformed forced `tool_choice` now 400s instead of silently
  downgrading to auto. Trait `ChatGenerator::chat` signature updated with its
  two infr-cli impls. _Deferred:_ streaming `usage` chunk (needs
  `stream_options.include_usage` parsing) and the e2e disconnect→slot-release
  path (integration-only; the latch logic is unit-tested).
- **`infr-cli` (all 6 findings)** — TDD, +7 tests; dead-code `#![allow]`
  removed. `--dev` now `remove_var`s the sibling backend envs (an inherited
  `INFR_CPU` can't shadow it) via a pure `resolve_backend` + one unified
  `selected_backend()` reader (consistent METAL>CPU>Vulkan precedence); model
  `resolve` only treats an existing `.gguf` FILE as local (mistyped paths give a
  clear error, not a network pull); the sweep sort is NaN-safe (`total_cmp`);
  the spurious "`--parallel` ignored" note fires only when explicitly set
  (`Option<usize>`); the DG→Metal→CPU→Vulkan funnel is one `build_chat_model`
  (was duplicated in run+serve); dead
  `ResolvedDevice`/`print_run_stats`/`bench -b`/`tg64@d` branch deleted. `#2`
  forced-tool retry now `reset_kv`s the session first (new no-op default
  `ChatModel::reset_kv` + Vulkan/Metal overrides). _Deferred:_ parse-GGUF- once
  (a large cross-crate API change, out of proportion to a 🟡).
- **`infr-chat` (all 5 findings)** — TDD, +14 tests. Streaming holdback now
  stops the content region at the earliest tool-call opener across every dialect
  `finish()` parses (Hermes `<tool_call>`, pipe `<|tool_call>`, and a
  whitespace-leading bare-`{` confirmed via `parse_bare_json_call`), so
  non-Hermes calls no longer leak as content **and** duplicate as a ToolCall —
  while ordinary content with a stray brace is preserved. The pipe arg-parser
  now translates JSON escapes (incl. `\uXXXX` surrogate pairs), keeps
  `inf`/`NaN` as strings (not `0`), uses `from_utf8_lossy` for keys, and strips
  a dangling `<|tool_call>` opener. The minijinja `Environment` is cached per
  template source (was rebuilt every render); `bos_token_id` falls back to `""`
  not `2` (EOS on Llama); `emit` scans via resumable cursors (was O(n²));
  `remove_spans` de-duplicates the two parsers.

### Highest-priority (production default paths)

| #      | Sev | Location                              | Issue                                                                                                                            |
| ------ | --- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| ~~1~~  | ✅  | `infr-hub`                            | ~~Downloaded blob never sha256-verified~~ — **FIXED** (`1263bcc`, + full hub slice).                                             |
| 2      | 🟠  | `infr-llama chat/mod.rs:186`          | Generate error leaves an **orphaned user turn** → next turn has two consecutive user messages, permanent history corruption.     |
| ~~3~~  | ✅  | `infr-server lib.rs`                  | ~~Streaming swallows errors as `stop`~~ — **FIXED** (error frame + panic-safe `DoneGuard`).                                      |
| ~~4~~  | ✅  | `infr-server lib.rs`                  | ~~No per-request cancellation~~ — **FIXED** (cancel latch → `req.abort()` frees the slot).                                       |
| 5      | 🟠  | `infr-llama runner.rs:3743,3989`      | Prefix-cache records **KV rows never materialized** (`max_new==0` frontier; grammar-forced tokens) → next turn attends stale KV. |
| 6      | 🟠  | `infr-vulkan adapter.rs:2997`         | Static split-K attn bounds chunk _size_ not _count_ → `n_chunks>1024` **overruns `attn_combine` `wexp[1024]`** at huge ctx.      |
| 7      | 🟠  | `infr-vulkan ops.rs:229`              | Kernel-cache double-checked lock **double-compiles + leaks a pipeline** under concurrent first use.                              |
| ~~8~~  | ✅  | `infr-llama sampling.rs`              | ~~Repeat penalty per-occurrence~~ — **FIXED** (`70bbe4e`; now per-distinct-token).                                               |
| 9      | 🟠  | `infr-vulkan shaders dg_eb_sample:61` | argmax reduce **drops the lower-index tie-break** → diverges from host on ties (feeds diffusion goldens).                        |
| ~~10~~ | ✅  | `infr-gguf lib.rs`                    | ~~Corrupt GGUF `pos+n` overflow panic~~ — **FIXED** (`checked_add`/`checked_mul` → `Error::Loader`).                             |
| ~~11~~ | ✅  | `infr-cli main.rs`                    | ~~`--dev` can't override inherited backend env~~ — **FIXED** (clears siblings; unified precedence).                              |
| 12     | 🟠  | `infr-metal exec.rs:2836`             | `Op::Rope` snapshots positions on the replay tape → **frozen RoPE after token 0** (llama-family Metal decode).                   |

Other 🟠 majors span host-hot-path churn (recorder per-dispatch `env::var` +
`Vec` allocs; adapter MoE `counts` double-zero), prefill perf (`gemm_proj`
uncoalesced + non-saddr weight reads), resource leaks (`lib.rs` device/instance
on error paths), `seam/mod` (large-vocab lm_head rejected under TP/pipeline;
process-global model pins), and the gated multi-GPU / parked-MTP features (kept
lower-urgency, flagged as such). Full detail per module below.

### Cross-cutting themes

- **`partial_cmp(...).unwrap()` NaN panics** recur in ≥5 files (diffusion, cli
  sweep, prof-rt, sampling) — a shared `by_desc_f64`/`total_cmp` comparator
  kills the class.
- **`assert!`/`.expect()`/`panic!` on recoverable input** (unregistered pager
  buffers, `GpuPager::new`, `Op::Copy` src_off, `make_compute_kernel` OOM, GGUF
  parse, `Op::Sample` `top_k==0`) — should return `Err`.
- **First-match-not-longest-prefix slot pick** duplicated in `seam/model.rs` and
  `parallel.rs` (wasted prefill).
- **Name-table vs SPV-table drift guarded only at runtime by `.expect()`**
  across `gemm.rs`/`linear.rs` and Metal `exec.rs` — a missing dtype =
  mid-inference panic.
- **Per-render / per-warm-call recomputation** (jinja env rebuild,
  session-stable seam derivations, capabilities clone, GGUF re-parse ×5 in the
  CLI).

<!-- SLICES APPENDED BELOW AS THEY ARE VERIFIED -->

## infr-vulkan/src/recorder.rs

1. **🟠 `recorder.rs:2537,2545,2558` (helpers `119`,`153`,`194`) —
   `std::env::var` on the per-dispatch GEMV recording path, contradicting the
   struct's own "read once" note (`466`).** `linear_native` calls
   `native_sg_choice` (≤4 env lookups), then
   `INFR_NO_GEMV_REG`/`INFR_GEMV_VARIANT` (2), then `native_rm_choice` (≤4) —
   ~10 process-mutex-guarded env lookups per GEMV. `no_barrier`/`prof`/`prof2`
   are already cached to fields at construction; these routing knobs are not.
   Prefill records thousands of GEMVs/forward = thousands×10 needless host
   lookups in exactly the many-op regime the recorder optimizes. _Fix:_ resolve
   all `INFR_GEMV_*` knobs once into `self`/`OnceLock` at `new_inner`.
2. **🟠 `recorder.rs:957` (also `918`,`856`,`875`) — every dispatch
   heap-allocates ≥3 transient `Vec`s.** `dispatch3` builds
   `read_bufs`+`write_bufs` `Vec<vk::Buffer>` per call; `bind_descriptors` a
   `Vec<WriteDescriptorSet>`. Binding counts are statically ≤ ~9. A batched MoE
   prefill chunk records ~50k dispatches → ~150k tiny allocations of pure
   allocator churn on the host-bound path. _Fix:_ `SmallVec<[_; 12]>`/`ArrayVec`
   — counts are known-small.
3. **🟡 `recorder.rs:8627,8636,8639` — `finish` leaks cmd buffer + descriptor
   pools + query pool on the submit/wait error path.** `Recorder` has no `Drop`
   (by design); `end_command_buffer`/`queue_submit`/`queue_wait_idle` use `?`
   and early-return before the cleanup at 8651-8655. Fires on device-lost —
   exactly when live pools then get flagged at `vkDestroyDevice`.
   `finish_nowait` has the same asymmetry. _Fix:_ guard struct /
   free-then-propagate on both paths (as `discard` already does).
4. **🟡 `recorder.rs:7417 & 7696` — `matmul_mmq_experts` and
   `matmul_mmq_experts_paged` carry near-duplicate ~180-line dtype→kernel match
   tables** differing only by an `_xp`/`_xpg` suffix; the `unreachable!` arms
   must be hand-kept in sync. Drift hazard. _Fix:_ one dtype-keyed table
   returning stem + binding count, append suffixes programmatically.
5. **🟡 `recorder.rs` (5 `pub fn` sites) — parity-only `_at`/streamed entry
   points are `pub`**, each documented "Not wired into any production dispatch
   yet; exists so a parity test can exercise the `_streamed` SPV." They widen
   the crate's public contract and hide dead-in-prod code from dead-code lints.
   _Fix:_ `pub(crate)` + `#[cfg(any(test, feature="parity"))]`.
6. **🟡 `recorder.rs:643` — descriptor-pool tranche under-provisions descriptors
   vs sets.** `max_sets(4096)` but `descriptor_count: 16384`; dispatches binding
   ~8 buffers exhaust descriptors at ~2048 sets (half of `max_sets`), doubling
   pool-growth frequency. _Fix:_ size
   `descriptor_count ≈ max_sets × max_bindings` or lower `max_sets` to match.

_Clean:_ push-constant packing, `sync` RAW/WAR/WAW hazard logic + stage-mask
pairing, inert-bound-descriptor hazard convention, chunk-split math,
split-K/flash `n_splits`, `RecordedCmd`/`PendingSegment` ownership.

## infr-vulkan/src/adapter.rs

1. **🟠 `adapter.rs:3794` — batched-MoE `counts` is host-blocking calloc'd then
   re-zeroed on device.** `counts = al(n_expert_local)?` uses the zeroing
   `alloc` (one-shot submit + `queue_wait_idle` ≈27µs) — its own comment says
   "zeroed below" — then `rec.zero(counts,…)` at `3853` zeroes it again on-GPU
   before `moe_bucket_count`. Every sibling buffer uses non-blocking `alu`. One
   pointless full host sync per MoE layer ≈ ~1.3ms/forward on a 48-layer chunk.
   _Fix:_ allocate `counts` with `alu`; drop the single-use `al` closure.
2. **🟡 `adapter.rs:2997,3121` — static split-K attention bounds chunk _size_,
   not chunk _count_, so `n_chunks` can exceed `attn_combine.comp`'s
   `shared float wexp[1024]`.** `chunk=(span/32).clamp(64,512)`,
   `n_chunks=span.div_ceil(chunk)`; for `span>512*1024` (≈524k keys)
   `n_chunks>1024` → OOB shared-mem write (`wexp[c]=…`, combine shader L37). The
   Dynamic path (`2531`) scales chunk by `cap_rows.div_ceil(1024)` to prevent
   exactly this; the static path (all ineligible decode + prefill, reachable
   under `INFR_KV_OVERFLOW` huge ctx) doesn't. _Fix:_ bound count too, e.g.
   `chunk = (span/32).clamp(64,512).max(span.div_ceil(1024))`.
3. **🟡 `adapter.rs:2019` — `GatedAct::Silu` silently ignores `gate_stride`/
   `gate_block_width`.** It guards `up_off`/`up_stride` then calls contiguous
   `rec.silu_mul`, dropping the gate stride fields — the adjacent `Sigmoid` arm
   honors them via `mul_sigmoid`. A strided-gate `Silu` (shape-legal) computes
   silently wrong instead of erroring. _Fix:_ honor them, or add the same `Err`
   guard the `up_*` cases have.
4. **🟡 `adapter.rs:1933` — cross-dtype `Op::Copy` uses
   `assert_eq!(*src_off,0)`, panicking the process** on a structurally-valid IR
   (`store_f16` has no source offset) where every other unsupported-shape case
   in `lower_op` returns a recoverable `be(...)`. _Fix:_ replace the assert with
   an `if …{ return Err(be(…)) }`.
5. **🟡 `adapter.rs:2720 & 2813` — `INFR_FLASH_MIN_ROWS` read twice** into
   `flash_min_rows_env` (feeds `flash_geom`) and `flash_min_rows` (feeds
   `nc_fa_ok`); they must stay equal for the row-floor logic to agree — a latent
   divergence trap + redundant getenv per attention op. _Fix:_ read once, share.
6. **🟡 `adapter.rs:4959 & 1621/1676/5004` — `streamed_prefill_gemm`
   re-implements the resident GEMM tile-selection**; the identical
   `narrow_grid`/`splits` split-K block is copy-pasted three times, and the
   padded-dst temp+copy dance (`1715`/`1383`/`4977`) three more. The fn's own
   doc warns resident & streamed "must track exactly." _Fix:_ extract one
   tile-selection helper (param'd over a `w_addr: u64`) + a `with_padded_dst`
   helper; call from all tiers.

## infr-vulkan/src/lib.rs + pcache.rs

1. **🟠 `lib.rs:1740` (also `1344`,`1768`,`1893`) — `VkInstance`+`VkDevice`(+cmd
   pool) leaked on every `new_selected` error path after `create_device`
   (`1681`).** `ash::Instance`/`Device` have no `Drop` (only
   `VulkanShared::Drop` frees them). The subgroup-32 rejection at `1740` is a
   _recoverable_ path (seam catches `Err` → CPU fallback) yet a whole logical
   device + instance + pool leak for process life. _Fix:_ run the post-device
   body in a closure that destroys what it built on `Err` (as
   `enumerate_devices` already does), or move the subgroup/env validation before
   `create_device`.
2. **🟡 `lib.rs:2732` — device-local zero-init truncates to a 4-byte multiple**,
   leaving the trailing 1-3 bytes of a non-multiple-of-4 buffer holding recycled
   VRAM — violating `Backend::alloc`'s stated "zero-init so recycled VRAM can't
   leak" guarantee (`3055`). Latent today (all tensor sizes are 4-aligned) but
   the guarantee is unconditional. _Fix:_ round the fill up to cover the tail
   (backing allocation is aligned larger, so in-bounds) or memset the tail.
3. **🟡 `lib.rs:217/436/2348` — `uma_spilled` is write-only** (`fetch_add`/
   `fetch_sub` only; no `.load` anywhere). The doc at `221`/`583` claims
   "budgeting is `device_used`/`uma_spilled`" but every budget path reads only
   `device_used`. Dead accounting + inaccurate invariant. _Fix:_ drop the
   counter or actually consume it.
4. **🟡 `lib.rs:2440,2520` — redundant `get_buffer_device_address` re-query.**
   `alloc_arena_bda`/`bda_weight_alloc` call `make_buf_ex(device_address=true)`
   which already queries + stores `VkBuffer::own_addr` (`2702`), then
   immediately re-query the same handle. _Fix:_ read
   `buf.own_addr`/`buf.device_addr()`.
5. **🟡 `lib.rs:2612 & 2312` — UMA-spill runs the VRAM budget check (+ driver
   query) twice** (`make_buf_ex` then `alloc_vram_mapped(budget_check=true)`), a
   third memory-property round-trip on top of `device_local_room`'s. _Fix:_ pass
   `budget_check=false` from the spill branch.
6. **🟡 `pcache.rs:278` — `save()` temp file `tmp.{pid}` collides between
   concurrent saves in one process.** `persist_pipeline_cache` runs after every
   compile with no lock over the write; two threads compiling different kernels
   (concurrent `serve`) write+rename the same path → torn write / failed rename.
   Self-heals via checksum but is a real file race. _Fix:_ unique temp suffix or
   serialize `save`.
7. **🟡 `pcache.rs:153,325` — poison tripwire marker keyed by PID only.** When
   one process builds multiple backends on the same device (`infr bench` MTP
   loops, CPU/Vulkan parity), the first backend's clean drop `disarm()`s the
   shared marker while a sibling is still live; a later unclean death then
   leaves a poisoned blob uncaught — the exact case the tripwire exists for.
   _Fix:_ key the marker per backend instance (PID + per-`PcachePersist` nonce).

## infr-vulkan/src/ops.rs + pager.rs

1. **🟠 `ops.rs:229-244` (& `256-272`) — `kernel`/`kernel_sg` double-checked
   locking double-compiles and leaks a pipeline under concurrent first use.**
   `lock→get→unlock→compile→lock→insert`: two threads racing the first fetch of
   a `name` each run `make_compute_kernel` (full shader+pipeline+pool+layout),
   then both `insert` — the second overwrites the first, whose Vulkan objects
   are never destroyed (map holds one entry) → leak for process life, cache
   persisted twice. Reachable via lazy kernel fetch during parallel prefill /
   multi-stream serve. _Fix:_ build under one lock (`get_or_insert_with`) or a
   per-name `OnceCell`.
2. **🟠 `ops.rs:43,98-104,128 — `make_compute_kernel` panics on recoverable
   driver/OOM errors.** `create_shader_module`/`create_pipeline_layout`/
   `create_descriptor_pool` use `.expect()` and pipeline creation `panic!`s;
   `kernel`/`kernel_sg` return a bare `ComputeKernel`, so a late
   `OUT_OF_DEVICE_MEMORY` on kernel compile aborts the process instead of an
   `Err` the seam could handle. _Fix:_ thread `Result` through (keep the
   null-handle assert).
3. **🟡 `ops.rs:229` — kernel cache keyed on `name` alone; `spv`/`n_buf`/
   `push_size` ignored on hit.** A name reused with a different shader/layout
   silently returns a mismatched pipeline (VUID/UB); enforced only by manual
   discipline (see rope comment `396`). _Fix:_ debug-assert cached `n_buf`/
   `push_size` (+ spv hash) match the request.
4. **🟡 `pager.rs:97,850-879,711` — `assert!`/`.expect()` on recoverable
   input.** `GpuPager::new` (returns `Result`) asserts on `n_slots==0` /
   misaligned `slot_bytes` — reachable from a too-small seam VRAM budget;
   `pool_of`/`arena`/ `arena_addr`/`touch_all_hits`/`begin_batch` (all `pub`)
   `.expect()` on an unregistered buffer while siblings
   `touch_role`/`stage_role` return `Err`. _Fix:_ return `Err(be(…))`
   consistently.
5. **🟡 `pager.rs:199-209,273-282,317-327` — LUT-mirror evict+insert
   triplicated** across `touch_staged`/`schedule_staged`/`ensure_resident`
   (verified byte-for- byte identical) — the one place a wrong LUT entry becomes
   silent-zero MoE output. _Fix:_ extract `record_placement(id, slot, evicted)`.
6. **🟡 `pager.rs:646-678` — `touch_role` decode path allocates a `Vec` per call
   and does one synchronous `one_shot` submit per expert miss.** Steady-state
   demand path (per layer per token); the batched ring path exists to avoid
   exactly this serialized round-trip. _Fix:_ reuse a scratch `Vec`; batch the
   miss copies into one submission. (`DensePagerSession::stage` `1112` similarly
   allocates `seg_refs` per call.)
7. **🟡 `pager.rs:611/1038` — `MoePagerSession`/`DensePagerSession` duplicate
   the `INFR_PAGER_STATS` read, stats printer, and the subtle `&**arc as &dyn
   AsRef<[u8]>`deref dance** (guards against`Arc`'s own `AsRef`; a copy omitting it compiles but resolves wrong). *Fix:* shared `expert_bytes(arc)->&[u8]` +
   stats helpers; read env once.

## infr-vulkan/src/gemm.rs + matmul.rs + linear.rs

1. **🟠 `linear.rs:32` vs `gemm.rs:396` (and mmv/idm/embed pairs) — kernel-NAME
   gate and SPIR-V loader are hand-duplicated tables, often in different files,
   reconciled only at runtime via `.expect()`.** Recorder gates on
   `linear::native_id_kernel_name(dt).is_some()` (`recorder.rs:6816`) but loads
   via `gemm::native_id_build_spv(dt).expect(…)` (`8076`). Same for
   `native_idm_*`, `native_mmv*`, `embed_gather`. A dtype added to the name
   table but not the spv twin (or paged twin) turns a decode into a
   **mid-inference panic** instead of a compile error; only id/idm have a
   partial drift test (and it checks names, not spv). _Fix:_ make
   `*_kernel_name` delegate to `*_build_spv().map(|(n,_)|n)`, or generate both
   from one dtype→literal table.
2. **🟡 `gemm.rs:645` vs `819-908` — `Iq4Xs` is advertised int8-mrow-eligible
   (`native_mmv_mrow_kernel_name(Iq4Xs)=Some`) but has no `_res` variant arm**,
   so `linear_mmv_mrow_at(…,res).expect()` (`recorder.rs:2946`) would panic on a
   residual Iq4Xs decode. Unreachable only because `mmv_int8_decode_dtypes`
   excludes Iq4Xs; the adapter comment (`723`) "every dtype with a plain build
   has the o4/res twins" is false for Iq4Xs. _Fix:_ build the `_res` variants or
   make the res-legality predicate explicit + fix the comment.
3. **🟡 `matmul.rs:292` — `pub fn matmul_f32` unconditionally `println!`s
   per-call timing and rebuilds all Vulkan objects (shader/layout/pipeline/pool)
   every call** (none cached, unlike `kernel_sg`). Any repeat caller eats full
   pipeline creation + pollutes stdout. _Fix:_ gate the print behind a
   flag/delete; cache the pipeline or mark `#[doc(hidden)]` one-shot.
4. **🟡 `matmul.rs:169,287,57` — unchecked `usize` products (`m*n*4`, `m*k`,
   `k*n`) + `usize→u32` workgroup truncation in a `pub` GPU entry.** A
   large-but- plausible shape wraps silently (undersized alloc → OOB dispatch /
   truncated grid → wrong result). _Fix:_ `checked_mul`→`Err`, or document the
   ceiling.
5. **🟡 `matmul.rs:216`, `gemm.rs` `run_gemm` — push constants use `to_ne_bytes`
   while the comment promises "little-endian u32".** Latent (correct on LE
   targets only). _Fix:_ `to_le_bytes()` to match the GPU-side contract.
6. **🟡 `gemm.rs` (25 sites) — no-op `macro_rules! v { ($n:literal)=>{{$n}} }`
   wraps string literals in the name-only tables**, pure indirection existing
   only to make them resemble the spv tables. _Fix:_ return the literals
   directly. (`matmul.rs:24` similarly re-inlines `gemm::spv_words`'s byte→word
   decode — reuse the shared helper.)
7. **🟡 `linear.rs:413` — `moe_expert_floor_covers_dense_set` iterates a
   hand-written dtype list**, so a dtype added to `native_dense_supported` but
   not the list escapes the drift guard (the sibling test correctly derives from
   `MOE_MMQ_DTYPES`). _Fix:_ derive from `native_dense_supported`/canonical
   enum.

## infr-vulkan/src/{tp,tp_allreduce,tp_sem,pipeline,ep,p2p}.rs (gated multi-GPU)

_All default-OFF experimental features; correctness + resource-leak issues
weighted highest._

1. **🟠 `p2p.rs:168,284` — cross-device dma-buf buffers created
   `SharingMode::EXCLUSIVE` with no queue-family ownership transfer.** Written
   by device A's queue, read by device B's (different family/device); the spec
   requires an EXTERNAL/FOREIGN release+acquire for EXCLUSIVE cross-family
   sharing. The host-fence path hides it via `queue_wait_idle`, but the whole
   point of `tp_sem.rs` is to drop those fences — the transfer barrier is still
   missing → formally UB, driver-fragile. _Fix:_ `CONCURRENT` over the
   participating families, or emit `VK_QUEUE_FAMILY_EXTERNAL` buffer barriers
   around the publish/gather copies.
2. **🟠 `tp_allreduce.rs:444` — all-reduce hardwired to `DType::F32`**
   (`elems= bytes/4`), while `tp.rs:613`/`ep.rs:92` `dtype_bytes` + comments
   advertise "f32/f16" boundaries. An f16 boundary makes `numel != bytes/4` →
   every all-reduce hard-errors; if the guard were loosened it would sum f16
   bytes as f32 garbage. _Fix:_ carry boundary dtype into
   `AllReduce`/`build_reduce_graph`, or explicitly reject non-f32 + drop the
   misleading f16 generality.
3. **🟡 `tp.rs:477` — TP never resizes the KV-cache tensor _decl_, so
   `desc.numel()/row_stride` reports W× the real per-rank capacity.** Each rank
   gets a `bytes/W` buffer and strides are rewritten, but the decl numel stays
   full; any KV-capacity/overflow-to-host guard reading that value mis-fires
   (believes W× the room exists). Benign only because the runner never drives
   past true `ctx`. _Fix:_ shrink the KV decl to `numel/W` in the decl-shrink
   pass.
4. **🟡 `pipeline.rs:349` — residual handoff always copies the replica from
   `prev` (last segment's device), not the device that last wrote it.** Unsound
   for any replicated op-written tensor produced, skipped a segment, then read
   later — consumer gets stale bytes. Safe today only because `hidden` is
   touched every layer. _Fix:_ track per-cut-tensor last-writer device, hand off
   from it.
5. **🟡 `tp_sem.rs:166,226 + reduce loop — command buffers leak / GPU work
   abandoned on error paths.** `tp_record_copies`/`tp_submit_*` return `Err`
   without `free_command_buffers`; a mid-loop `?` in `reduce_p2p_semaphore`
   drops already-collected `pub_cmds`/`gat_cmds` unwaited. Shared long-lived
   pool. _Fix:_ RAII/explicit free in each error branch; free+await accumulated
   cmds before propagating.
6. **🟡
   `tp_allreduce.rs:399,416 — host-bounce reduce re-downloads each producer W−1×; the semaphore reduce still does serial per-rank `device_wait_idle`.**
   The former multiplies the dominant PCIe read by W−1 (download once into a
   per-producer host buffer instead); the latter serializes independent devices
   (submit all ranks, then wait once / single fence set).
7. **🟡 `p2p.rs:135` — `external_semaphore_supported` doc says the semaphore
   path is gated OFF ("v1 returns false"), but the body returns
   `external_semaphore_fd.is_some()`** — the untested GPU-ordering path
   activates automatically whenever the extension loads. Misleads a reviewer
   about whether the risky path runs. _Fix:_ correct the doc or actually gate
   it.
8. **🟡 `tp.rs:613 & ep.rs:92` — byte-identical `dtype_bytes` duplicated;
   `tp.rs:557` sizes one AllReduce to the `max` boundary but `reduce` requires
   `elems==self.elems` exactly** — holds only because both boundaries are
   `[tokens,n_embd]`; a model with differing row-parallel widths silently
   breaks. `p2p.rs:303` OpaqueFd import picks the lowest memory-type bit, not
   the exporter's (can spuriously reject). _Fix:_ hoist the helper; assert equal
   boundary sizes (or per-size transport); thread the exporter's memory-type
   index through `P2pExport`.

## infr-vulkan/shaders — attention / flash / KV / softmax

1. **🟠 `attn_flash.comp:71,116` (also `attn_flash_partial:183,270`,
   `attn_flash_warp:188,301`, `attn_flash_reg:119,169`) — direct `coopMatLoad`
   reads K/V rows past `kv_len` with no bounds guard.**
   `kvend=min(kv_len,qmax+1)` isn't rounded to `BN=64`, so the last 64-wide
   column tile over-reads global K/V. Numerically safe (scores masked at `87`),
   but a genuine OOB _global read_ unless K/V is capacity/`BN`-multiple
   allocated — and the `-DSTAGE` arms guard exactly this (`kv0+kvl<kv_len`)
   while the direct arms don't (author-acknowledged asymmetry). _Fix:_ add the
   `kv0+col<kv_len` clamp to the direct path, or document/enforce `BN`-multiple
   K/V allocation + rely on `robustBufferAccess2`.
2. **🟡 `attn_flash.comp:126` — output store unguarded on padding query rows.**
   `o[(qr0+r)*…]` runs for all `r<BM`; if `q`/`o` are sized to a
   non-`BM`-aligned `q_len`, tile rows `≥q_len` OOB-read `q` and OOB-write `o`.
   `attn_nc_fa.comp:228` guards the identical store with `if (gr<q_len)`. _Fix:_
   add `if (qr0+r<q_len)` or state the mpad precondition.
3. **🟡 `attn_pv_warp.comp:49` (`BN=128`) / `attn_pv.comp:43` (`BN=64`) —
   `tilesN=hd/BN` is a comment-only precondition → div-by-zero / dropped dims.**
   hd=64 gives `tilesN=0` → division by zero at `50`; a non-multiple hd
   (e.g. 80) truncates and never covers the top dims. _Fix:_ validate `hd%BN==0`
   at dispatch or `max(1u,hd/BN)` + dim tail.
4. **🟡 `attn_combine.comp:24` — output dims dropped when `ntile ∤ hd`.**
   `hdt= hd/ntile` truncates; the union across tiles covers only `ntile*hdt`,
   leaving the top `hd-ntile*hdt` output dims uninitialized. _Fix:_ assert
   `hd%ntile==0` or give the last tile the remainder. (Also: `n_chunks` is never
   bounded vs `wexp[1024]` at `37` — same root as the adapter static-`n_chunks`
   finding.)
5. **🟡 `attn_partial_mrows.comp:52` — `sc[RB*SC]` overruns if a chunk exceeds
   `SC`** (nothing clamps `pc.chunk≤SC`); the `SC_MAX=256` build lacks the guard
   `attn_partial` gets from `sc[1024]`+`chunk≤512`. Opt-in `INFR_MROWS_ATTN`
   path. _Fix:_ static-assert/document `chunk≤SC_MAX` or clamp the write index.
6. **🟡 `quant_kv.comp:128` (defensive) — `FMT_Q5_1` omits the `clamp(…,0,31)`
   the Q4_0/Q4_1/Q5_0 arms apply.** Unlike Q5*0's asymmetric
   `x*id+16.5`(which can genuinely round to 32.5), Q5_1's`(x-vmin)_id`is bounded by`vmax`
   so it can't reach 32 barring impossible fp error — so this is a
   robustness/consistency nit, not a live bug, but matching the siblings removes
   the latent trap. \_Fix:_ `clamp(int((x-vmin)*id+0.5),0,31)`.
7. **🟡 perf/DRY — 32-lane redundant recompute + copy-pasted softmax.**
   `attn_combine.comp:36,40` recompute `mm`/`l` over all `nch` identically in
   every one of 32 lanes (32× partial-array traffic; scales poorly as chunks
   grow). The causal-mask + online-softmax rescale block is copy-pasted across 5
   flash kernels (`attn_flash:80`, `_partial:193`, `_warp:203`, `_reg:130`,
   `nc_fa:162`) with subtle layout diffs — only `nc_fa:181` has the `-1e29`
   all-dead-block guard, so a numerics fix drifts. Ring `RROW`/`rcap` + Q8/f16
   vec4 readers likewise duplicated across decode kernels. _Fix:_
   subgroup-reduce `mm`/`l`; factor the mask/rescale + ring readers into a
   shared `.glsl` include so the all-dead guard stays consistent.
8. **🟡 `softmax.comp:28` — `shared float part[8]` assumes `gl_NumSubgroups≤8`
   (256/32) with no `requiredSubgroupSize`.** On a device enumerating 16-lane
   subgroups for this pipeline, `gl_SubgroupID` reaches 15 → overruns `part`.
   Safe on wave32/64 RADV but unpinned. _Fix:_ pin `requiredSubgroupSize=32` (as
   decode kernels do) or size `part[16]`. (`attn_flash_combine.comp:11` also has
   a dead `const uint MAXS` — delete.)

## infr-vulkan/shaders — GEMM / GEMV / MoE-expert matmul

1. **🟠 `gemm_proj.comp:61-63 — f16-A projection GEMM stages weights with N as
   the inner index → every warp's global loads stride by K (uncoalesced).**
   Consecutive lanes read `wf((wgCol+cc)*k + (k0+r))` `k` elements apart → 32
   cache lines/load. The large-warptile twin `gemm_proj_warp.comp:73`
   deliberately swaps to k-inner ("k contiguous → coalesced"); this 64×64 kernel
   never got the fix. _Fix:_ stage with k inner like the warp twin (adjust `Bs`
   layout + `coopMatLoad`).
2. **🟠 `gemm_proj.comp:28` (& `gemm_proj_warp.comp:39`) — arena weights read
   via `#define WQ(i) ArenaW(w_ptr).v[i]`, the exact divergent-index-in-64-bit
   deref that `native_weight_addr.glsl:35` documents as the ~2.2x streamed
   regressor** (per-load `v_add_co`/`v_add_co_ci`, defeats ACO saddr scalar-base
   `global_load`). Both f16 projection GEMMs use it instead of the
   `arena_word`/byte-offset idiom the whole native path was rewritten to. _Fix:_
   read through the `native_weight_addr.glsl` helpers so the loads select saddr
   like the native twins.
3. **🟡 `native_gemm.comp:87` — full 64-row tile stored unconditionally while A
   staging guards `gr<pc.m`**; rows `[m, tileEnd)` are OOB `coopMatStore` writes
   unless C is allocated `ceil(m/64)*64` rows (also the `qa`/`dact` reads at
   `native_gemm_mmq_q8_0.comp:137,174`). The `gemm_proj.comp:6` header states
   this padded-C contract; `native_gemm` relies on it silently. _Fix:_ document
   it or guard the store rows against `pc.m` (as the `EXPERT_GRID` mmq path
   does).
4. **🟡 `native_gemv_rm_v2.comp:70` — `shared float reg_part[2]` overflows under
   `-DVARIANT_REG -DVARIANT_WG128`** (128-thread wg = 4 wave32 subgroups →
   writes idx 2,3 OOB); plus dead `tot` load + barrier at `128`. Env-gated
   default-OFF tuning file. _Fix:_ size by `THREADS/minSubgroupSize` (`[8]`),
   delete dead `tot`.
5. **🟡 `native_mmv_mrow.comp:73` — `shared float part[…]` (512 B) declared
   unconditionally but unused in the `-DOUTS4` build** (which reduces via
   `part4`), needlessly cutting LDS-limited occupancy on the shape OUTS4
   targets. _Fix:_ move `part` inside the non-OUTS4 branch.
6. **🟡 DRY — int8 dp4a decode helpers copy-pasted across the mmv/mmq family.**
   `rb`/`ru16`/`f16tof32`/`k4` + per-format `dpsub`/`wdec` re-declared in
   `native_mmv.comp:43`, `native_mmv_mw.comp:62`, `native_mmv_id_q4k.comp:31`,
   `native_mmv_mrow.comp:86` and each `native_gemm_mmq_*`; the `KV_IQ4NL_W`
   table + `kv_iq4nl()` duplicated verbatim in ≥4 files. The dequant path
   already funnels through shared `native_decode.glsl` — the dp4a path didn't
   get the same treatment. _Fix:_ factor a shared `native_dp4a.glsl` (+ hoist
   the IQ4NL table).
7. **🟡 `moe_sample.comp:78` — top-k gather caps at `k` in nondeterministic
   atomic order, so a threshold-key tie can evict a strictly-greater logit.**
   Bit-exact `f2ui` ties give a strictly-greater logit `slot≥k` (discarded)
   while an equal-to-threshold value fills a low slot → not the true top-k.
   Rare + sampling-only, but nondeterministic. _Fix:_ min-replacement gather, or
   gather all `≥thresh` then select top-k by value.
8. **🟡 YAGNI — measurement/stub kernels shipped in the build set.**
   `gemm_dp4a.comp` (self-described "RAW dp4a GEMM ceiling probe … no scales")
   and `gemm_coopmat.comp` ("v1 … assumes M,N,K multiples of 16, partial-tile
   handling added next") are superseded by `native_gemm_mmq_*`/`gemm_*_warp` yet
   still compiled; likewise the whole default-OFF `native_gemv_rm_v2` variant
   matrix (`build.rs:820`). _Fix:_ confirm none are dispatched and drop / gate
   behind a dev-only build flag to shrink the pipeline set.

## infr-vulkan/shaders — norm / rope / activation / sampling / MoE-routing / misc

1. **🟠 `dg_eb_sample.comp:61 — argmax tree-reduce drops the lower-index
   tie-break, diverging from the host on exact ties.** The per-thread scan uses
   strict `>`, but the workgroup reduce is bare `sval[t+s]>sval[t]` with NO
   `sidx` tie-break — even though the comment at `50` claims it matches
   `argmax.comp`, whose reduce (`47`) carries
   `|| (sval[t+s]==sval[t] && sidx[t+s]<sidx[t])`. On equal logits at indices 5
   and 256 the host returns 5, this returns 256 — silently violating the
   "exactly like the host" invariant that feeds the entropy-bound scheduler +
   the diffusion goldens. _Fix:_ add the same tie-break compare.
2. **🟡 `rope.comp:57-77 — first `rope_dim` outputs written twice** (passthrough
   copy of `[0,hd)` then rotated overwrite of `[0,rope_dim)`). For
   `rope_dim==hd` the whole vector is stored twice, and under `KV_BDA` the
   redundant stores hit the KV cache (`kv_store_half`) — doubling write traffic
   on the path this kernel feeds. _Fix:_ restrict the passthrough to the tail
   `[rope_dim,hd)` (as `qk_norm_rope.comp:110`).
3. **🟡 `embed_gather.comp:33 — tail elements dropped when `ne % 32 != 0`.**
   `nsub=ne/32` and both branches iterate whole 32-sub-blocks, so the final
   `ne%32` outputs of every gathered row are never written → partially
   uninitialized embeddings, silently. _Fix:_ host-assert `ne%32==0` or mask a
   ragged tail block.
4. **🟡
   `softmax.comp:41 & dg_eb_sample.comp:52 — `-1.0/0.0`compile-time div-by-zero to synthesize`-inf`**
   (UB in GLSL; a stricter SPIR-V toolchain could fold to NaN / error and
   corrupt the max reduction). The rest of the tree uses the finite `-1e30`.
   _Fix:_ `-1e30` or `intBitsToFloat(0xFF800000)`.
5. **🟡
   `softmax.comp:28 & dg_eb_sample.comp:37 — `part[8]`cross-subgroup array assumes exactly 8 subgroups with no`requiredSubgroupSize`**
   — a 16-lane dispatch overruns write+read. `rmsnorm.comp:21` sizes `NSGMAX` +
   pins sg32 for exactly this. _Fix:_ size by `256/MIN_SG` and/or pin sg32.
6. **🟡 DRY — three duplicated shader pairs.** `qk_norm_rope.comp` vs
   `qk_norm_rope_interleaved.comp` are byte-identical except
   `in_base`/`src_stride`; `sample_topk.comp` vs `moe_sample.comp` carry
   independent copies of the radix select + gather + sort + softmax +
   inverse-CDF (subtle sampler math mirrored by hand). _Fix:_ fold each pair
   into one shader selected by a `-DINTERLEAVED` define / accessor-macro include
   (the `native_decode.glsl`/`kv_addr.glsl` pattern).
7. **🟡 perf — avoidable serialization.** `deltanet_gates.comp:34` runs the
   ≤32-entry prefix scan on lane 0 only (31 lanes idle) — replace with
   `subgroupInclusiveAdd`. `moe_bucket_scan.comp:5` fuses the
   embarrassingly-parallel `fill[]=0` reset into the 1-lane scan kernel, forcing
   the scatter pass to wait on the scan for a zero it doesn't depend on — split
   the reset out (parallel clear / `vkCmdFillBuffer`).

## infr-llama/src/seam/runner.rs

1. **🟠
   `runner.rs:3743,4084 — `max_new==0`records the un-materialized frontier token in`cached`.**
   The loop breaks once `pos+1>=prompt.len()`, so the frontier `pos=plen-1`'s KV
   row is never written (comment 3740-3742 confirms "frontier stays un-fed"),
   yet teardown does `*cached=prompt.to_vec()` (full prompt). The named "session
   cache warming" case (3736): next turn computes `common_prefix_len==plen`,
   sets `start=plen`, and attention over position `plen-1` reads an
   unwritten/stale KV row → corrupt output. _Fix:_ at `max_new==0` record
   `cached=prompt[..plen-1]`.
2. **🟠
   `runner.rs:3989,4086 — grammar-forced tokens pushed to `out`/`cur`then`break`
   are cached but their KV never written.** The constrained branch pushes each
   `step` token _before_ the stop check (plain path pushes _after_); KV is
   written lazily only when a token is later fed as `cur[pos]`. On an immediate
   break the just-pushed tokens are unfed, but teardown caches `out[..len-1]`
   including them → next-turn prefix reuse (serve tool-calling, multi-turn)
   attends stale/zero KV. _Fix:_ advance `cached` only by tokens whose KV was
   actually fed.
3. **🟡 `runner.rs:871 — `prompt.len()-1` underflows on an empty prompt** (debug
   panic / release wrap masked by `.min`). _Fix:_ `saturating_sub(1)` / early
   error.
4. **🟡 `runner.rs:187-262,3436 — session-stable derivations recomputed every
   warm call** (before the `if state.is_none()` gate): `has_wv` per-layer tensor
   scans, `out_scale`/`dec_out_scale` per-layer `load_tensor_dequant` (real
   dequant for gemma4/diffusion), `rope_freqs` dequant,
   `fuse_*_decision`/`moe_batched_ok` O(n*layer×n_tensors) `find`+`format!` —
   all pure in `(g,cfg,caps)` yet repeated per serve request. \_Fix:* compute
   once, stash in `SeamKv`.
5. **🟡
   `runner.rs:3848 — host-embed decode allocs a throwaway `Vec<f32>`per token even when`embed_scale==1.0`**
   (qwen3/llama): `.map(|x| x*scale).collect()` copies an already-f32 table
   slice. _Fix:_ upload the slice directly when scale==1.
6. **🟡 `runner.rs:3025,3274,3684,3929 — KV+weights bind loop duplicated 4×**
   (denoise/verify/record-once/per-token); a forgotten bind (e.g. `rope_freqs`,
   per the 3564 warning) is a live unbound-Input panic. Also `build`'s 11
   positional bool/Option params let a transposed pair type-check, and the
   `denoise` doc (911) "Never true for any caller" is contradicted by the
   denoise site passing `true` (2948). _Fix:_ `bind_layer_io` helper + options
   struct + fix the stale doc.
7. **🟡
   `runner.rs:2845,3203,3506,3845 — prompt/canvas token ids not range-checked before `tok
   as usize \* ne` embedding-table slicing** → OOB slice panic on an
   out-of-vocab id instead of a handled error. _Fix:_ validate `tok < vocab`
   once.

## infr-llama/src/seam/mod.rs

1. **🟠
   `mod.rs:1484,1237 — TP and pipeline binders apply the whole-tensor BDA element cap to `lm_head`/`token_embd`,
   which the resident + EP binders deliberately exempt.**
   `chunk_covered_dense_tensor` (1178) and `expert_parallel_binder` (1714) skip
   `check_bda_element_cap` for `output.weight`/`token_embd.weight` (read only by
   dispatch-chunked ops, #77, may exceed 2³² elems — "quantized 256k-vocab
   lm*head"), but `tensor_parallel_binder` (1484) / `pipeline_binder` (1237)
   call it unconditionally on the full `numel` before replication → a
   large-vocab model that runs single-device/EP is hard-rejected under
   `INFR_TENSOR_PARALLEL`/ `INFR_PIPELINE`. \_Fix:* mirror EP — exempt
   `chunk_covered_dense_tensor(name)`.
2. **🟠
   `mod.rs:327,345 — process-wide `PINNED_UBATCH`/`PINNED_KV_Q8` `OnceLock`s
   leak placement decisions across models in a multi-model process.** Set once
   per process on first-load placement; a second model's `.set()` is a silent
   no-op, so `ubatch_rows()`/`kv_auto_q8()` (+ runner KV sizing) apply model A's
   pinned chunk height / q8 decision to model B — may not fit B's VRAM or forces
   unwanted q8 KV. The "set once" invariant is per-model but the storage is
   global (the multi-model serve path is real). _Fix:_ move the pins into
   per-session state (`SeamKv`/ `SeamModel`), or key/reset per model load.
3. **🟡 `mod.rs:553 — MoE expert-placement KV estimate hard-codes f16 (`*2*2`)
   even when auto-q8 KV is pinned** (the dense path honors `kv_auto_q8` via
   `kv_total_at`, MoE doesn't) → over-reserves ~2×, potentially forcing
   avoidable expert paging. Safe direction, wastes residency. _Fix:_ route MoE
   `kv_bytes` through the `kv_auto_q8`-aware helper.
4. **🟡 `mod.rs:59 — `TokenEmbd::get` `.expect()`s on a truncated/corrupt GGUF
   at inference time** (returns `&[f32]`, can't surface an error) — a real
   non-programmer input aborts the process, lazily on first host gather. _Fix:_
   validate/dequant at load, or make `get` fallible.
5. **🟡
   `mod.rs:183 — warm `generate_dense_vulkan_session`still builds a full`vulkan_moe_binder`
   Box the runner never invokes** (weights already resident, 622-630) —
   per-warm-turn alloc + `cache_override`/env re-reads, pure waste. _Fix:_ skip
   binder construction when `state.is_some()`.
6. **🟡 `mod.rs:2174 — `#[cfg_attr(infr_profile,
   infr_prof::instrument)]`on an`impl Deref`block, not a`fn`** — a
   function-instrumentation macro on a trivially-hot `deref` (every weight
   bind): dead at best, a profiling-build hazard at worst. _Fix:_ remove it.
7. **🟡 DRY — `mod.rs:1206/1361/1630` three byte-identical `parse_*_devices`
   (`VulkanN` parse, differ only in env name/error/min-check);
   `mod.rs:1261/1531/ 1743` four multi-GPU `generate_*` wrappers duplicate the
   arch guards + backend construction + the 8-`None` `generate_dense_backend`
   tail; `mod.rs:95/1893` the CPU/Metal upload-binder closure copied across 4/3
   sites.** _Fix:_ `parse_device_list(env,min,label)`, `run_dense_oneshot(...)`,
   and `cpu_bind`/`metal_bind` constructors.

## infr-llama/src/seam/{model,weights,sc}.rs

1. **🟡 `model.rs:157 — `SlotPool::pick` extend-branch returns the FIRST
   matching slot, not the longest-prefix one** (the struct doc claims longest).
   If two slots are both prefixes of `prompt`, it hands out the shorter → the
   runner re-prefills more suffix than needed (`start=common_prefix_len`, runner
   `871`). Correctness fine, wasted prefill. _Fix:_ `max_by_key(prefix_score)`
   over the extend-satisfying slots.
2. **🟡
   `model.rs:1116 — `bench_vulkan` mutates process-global env (`remove_var`/`set_var`on`INFR_PROF2`)
   around a rayon-parallel forward.** Env is a process-wide table read by rayon
   workers at construction → a data race if anything else touches env
   concurrently (and `set_var` is `unsafe` under edition 2024). _Fix:_ thread a
   "suppress profiling" flag / `AtomicBool` instead of toggling the env var.
3. **🟡 `model.rs:1382 — `generate_metal_spec`clones the entire`committed`
   history every verify round** (`feed=committed.clone()`), O(n²) copied token
   ids over a long generation for no functional need. _Fix:_ persistent `feed`
   buffer + truncate after verify.
4. **🟡 `model.rs:432,455,511 — `weight_footprint(&gguf)` recomputed 2-3× per
   session-open** on the clamp path (`kv_fit_ctx` → `kv_fit_ctx_fmt` →
   `weight_footprint`, again for the log line) — each a full tensor-metadata
   scan. _Fix:_ compute once, pass it in / memoize on `SeamModel`.
5. **🟡
   `model.rs:1998 — `spec_accept_stochastic`degenerate fallback emits token id`0`**
   (`.map(...).unwrap_or(0)`) when the residual and `p_dists[i]` are both empty
   — a silent possibly-invalid token instead of an error. Unreachable today but
   the guard is written as if reachable. _Fix:_ error / `.expect()` it.
6. **🟡 DRY — tokenizer-encode boilerplate + upload closures duplicated.** The
   `tokenizer.encode(...).get_ids().to_vec()` pattern recurs ~9× across the
   `generate_*`/`prefill_*` entries though `encode` (`model.rs:639`) already
   does it; the DiffusionGemma Metal (`1812`/`1854`) and CPU (`1598`/`1649`)
   `prefill`/ `denoise` pairs hold byte-identical weight-upload closures. _Fix:_
   route through `self.encode`; hoist one `*_upload_bind()` per backend.

## infr-llama/src/mtp/{mod,backends}.rs (MTP spec-decode, parked/opt-in)

_`INFR_MTP` is opt-in and token-identity is VERIFY-guarded, so the correctness
items below are latent acceptance-rate/perf bugs, not output corruption._

1. **🟠 (PLAUSIBLE — validate)
   `mtp/mod.rs:2534 — cycle `catch_up`passes`start_pos = n_past + 1`, one
   position too high.** Draft appends at absolute position `n_past+s` (`1800`)
   and catch*up writes at `start_pos+s` (`947`); the committed tokens
   `t*{n*past..n_past+accepted}`should therefore land at head positions`n_past..`, i.e. `start_pos=n_past`. The `+1`stores`(t*{i-1},h\_{i-2})`at position`i`(wrong RoPE + stale`h`) and leaves the draft's stale row at `n_past`un-rewritten. Doesn't break token-identity (VERIFY only commits trunk-confirmed tokens) and is untested (the only multi-cycle test is`#[ignore]`d while MTP is parked) — I could not fully trace prime's convention to confirm, so **verify against `speculative.cpp`+ re-measure α** before changing. *Fix (if confirmed):* pass`n_past`, not `n_past+1`.
2. **🟠 `mtp/mod.rs:1867 — `catch_up` computes + downloads a full vocab-wide
   logits row it discards every cycle.** It calls `sess.forward()` and drops the
   result, but `forward` always builds the non-fused graph with the lm*head
   `Op::Linear [rows,vocab]` as an `Output` and downloads
   `rows*vocab`f32. For catch-up only the`WriteKv`ops matter — the`rows×n_embd×151936`GEMM + readback is pure waste per spec cycle. \_Fix:* a`want_logits:false`/KV-only
   forward variant that omits the lm_head Linear + its download.
3. **🟡 `mtp/mod.rs:2536 — `pending_h` handed to the next cycle's draft is one
   step stale** vs the init handoff (`h_{n_past+accepted-1}` for
   `id_last=t_{n_past+accepted}`), depressing α. _Fix:_ confirm vs
   `speculative.cpp`; obtain `next_tok`'s own hidden rather than reusing the
   prior row.
4. **🟡 `mtp/mod.rs:2470,2526 — unguarded `usize`underflow`base =
   m-(cand.len()+1)` in the non-leading branch** (the
   `debug_assert_eq!(m,cand.len())` only covers the leading branches); a state
   edge with `m<cand.len()+1` wraps huge → slice panic on an
   otherwise-unvalidated state. _Fix:_ `ensure!(m>=cand.len()+1,…)`.
5. **🟡 perf/DRY — per-call staging allocs + duplicated builders/glue.**
   `forward`/ `forward_draft`/`draft_chain` alloc all staging/readback buffers
   fresh per call (5 allocs/step × n*max/cycle) — pool on the session.
   `build_mtp_graph` (`409`) and `build_mtp_draft_chain_graph` (`864`) copy the
   ~150-line qwen35 layer op emission verbatim; `forward`/`forward_draft`
   duplicate the alloc/upload/bind/execute glue; the per-backend weight-bind
   closures are duplicated between the session constructors and the driver
   (`1372`, `backends.rs:132`). Also (`2489`) the leading-state flags
   (`leading_h`/`leading_id`/`leading_dist`) must stay present-or-absent in
   lock-step with no enforcement. \_Fix:* `emit_mtp_layer` helper, shared
   upload/bind helper, per-backend `mtp_bind_weight(be)`, and a single
   `Option<Leading{…}>` enum for the leading state.

## infr-llama/src/{diffusion,parallel,util}.rs + chat/\*

1. **🟠 `chat/mod.rs:186 — a failed `generate` leaves an orphaned user turn in
   history.** The user message is pushed at `158`; a _render_ failure rolls it
   back (`168`) but a `generate_with_step_hook(...)?` error at `188` propagates
   with the user turn still in `history` and no assistant reply — asymmetric.
   Any transient generate error (GPU fault, KV overflow, stop-abort) then makes
   the next `turn` push a second consecutive `user` message → the model
   re-renders with two user turns and no assistant between → permanently corrupt
   conversation state. _Fix:_ `history.pop()` on the generate `Err` too (or push
   the user turn only after success).
2. **🟠
   `parallel.rs:304 — `checkout`holds the pool`Mutex`across a device-side`seed_from`
   GPU submit** (`307-313`), directly against the module doc (`48-59`). Under
   `--parallel N` every other request's `checkout` and every `SlotGuard::drop`
   serializes behind a KV-copy submit → global chokepoint. _Fix:_ take
   `src`/`dst` out, drop the lock across `seed_from`, re-acquire to finish
   bookkeeping.
3. **🟡 `parallel.rs:276 — prefix-continuation picks the first matching free
   slot, not the longest-prefix one** (same bug shape as `seam/model.rs`
   `SlotPool::pick`) → re-prefills more suffix, undercutting the ~7x TTFT
   prefix-cache win. _Fix:_ `max_by_key(prefix_score)` over the qualifying free
   slots.
4. **🟡 `diffusion.rs:387 — `entropy[a].partial_cmp(&entropy[b]).unwrap()`
   panics on NaN.** The `DenoiseOutcome::Reduced` branch adopts `entropy`
   verbatim from GPU output; a single NaN row turns a numeric glitch into a hard
   mid-generation panic. _Fix:_ `total_cmp` / `unwrap_or(Equal)`.
5. **🟡 `diffusion.rs:323,426 + 390 — hot-loop waste in denoise.**
   `map_init(|| vec![0f32; vocab], …)` reallocates the ~1 MB (`vocab≈262144`)
   per-thread `escratch` every step (fresh `par_iter`/step) — hoist a persistent
   per-thread scratch; `prev_argmax=Some(argmax_canvas.clone())` (`426`) clones
   the whole canvas every step — `mem::swap` a reusable buffer; the acceptance
   loop (`390`) never `break`s though `cum_e-entropy[pos]` is monotonic once
   past the bound — add the `break`.
6. **🟡 `chat/diffusion.rs:302 / util.rs:59 — per-block stream detok redecodes
   the whole accumulator per token** (`decode(acc,true)` over growing `acc` per
   id) → O(total²) detok for a block path where the whole block's ids are
   already in hand. _Fix:_ decode the newly-committed span once (tracking the
   char-boundary holdback).
7. **🟡 DRY — per-backend chat boilerplate.** The `INFR_PROF2`-suppress +
   throwaway warmup dance (`chat/vulkan.rs:149`, `metal.rs:81`,
   `diffusion.rs:179`, `parallel.rs:154`) and the byte-identical `render` impl
   recur across all four `ChatModel`s; `wants_mtp` is duplicated near-verbatim
   (`vulkan.rs:71` vs `metal.rs:37`) and has already drifted (one warns, one
   doesn't). _Fix:_ `with_prof2_suppressed(||…)` helper, a provided
   `ChatModel::render` default, and one shared `should_use_mtp(cfg)` + loader.

## infr-metal/src/exec.rs (Metal backend — audited by reading; not runnable here)

1. **🟠 `exec.rs:2836 — plain `Op::Rope` captures a stale positions snapshot on
   the decode-replay tape → frozen RoPE after token 0.** The Rope arm binds
   positions via `ensure_device` (allocates a _new_ f32 buffer widened from the
   host mirror), so the replay `TapeEntry` retains the record-time value; the
   seam rewrites the live i32 positions buffer between tokens, but the recorded
   dispatch keeps reading token-0's angle. `Op::QkNormRope` (`2881`)
   deliberately binds the _live_ buffer with an explicit "replay tape stays
   valid" comment — the exact fix. `replay_shape` admits `Op::Rope` (`802`) and
   the position finder matches both, so a llama-family decode graph (plain
   Rope + f16 KV, hd 64/128/256) qualifies for replay and silently rotates every
   token by position 0. _Fix:_ bind the live i32 positions buffer in the Rope
   arm (i32 `rope_f32` variant, like QkNormRope), or exclude plain `Op::Rope`
   from `replay_shape`.
2. **🟡
   `exec.rs:743,3044 — `sample_split_shape`sizes scratch with`top_k.min(64)`but packs the raw`top_k`for`sample_f32_stage1`.**
   A `top_k>64` caller makes stage-1 write `top_k` candidates/group into a
   64-per-group buffer → OOB device write; nothing clamps `top_k` before here.
   _Fix:_ clamp once, use it for both sizing and the param (or assert `≤64`).
3. **🟡 `exec.rs:3105 — `Op::Softmax`launches`softmax_wide_f32` at tg=256 with
   no threadgroup-cap gate/fallback** (unlike RmsNorm/QkNormRope/attention
   tiers). `encode_tg_w` silently clamps `tgw=tg.min(cap)`, so on a
   capacity-capped device the wide cross-lane reduction runs with <256 lanes and
   reads uninitialized threadgroup slots → wrong result, no error (arm flagged
   UNVERIFIED on real HW). _Fix:_ gate on the cap, fall back to scalar softmax /
   error loudly.
4. **🟡 `exec.rs:3565,3629,2484 — per-op transient device buffers `new_buffer`'d
   fresh instead of pooled.** The KV f16 dequant scratch `ks`/`vs`, q-cast `qh`,
   and HGEMM `xh` allocate `kv_len*n_kv*hd*2`-byte buffers per layer per token
   on the decoupled-quant KV path, growing with depth, while the `scratch_buf`
   pool that exists to amortize this goes unused. _Fix:_ route through
   `scratch_buf` keyed by (size, tag).
5. **🟡 `exec.rs:2317 — pipeline-state cache looked up twice per dispatch**
   (cap-check `get(kern)` then encode `get(kern)`: cmm/hmm/rmsnorm/qknormrope +
   every attention `fits(...)`), each locking the cache mutex + hashing +
   retaining the PSO — redundant on the prefill hot path. _Fix:_ fetch once,
   read cap off it, reuse.
6. **🟡 DRY/YAGNI — `exec.rs:2267` ~7 parallel 15-arm `qw.kern → "…suffix"`
   match tables** (`_hmm`/`_cmm`/`_cmm_ks`/`_rt`/m==1 `_ks`/`_add`); a missed
   arm falls through a `_ => "linear_quik8_*"` default (wrong kernel, no error).
   And `2200` the Linear arm allocates `dev_dst` up front that the
   fused-residual peephole leaves dead. _Fix:_ one `(base → {suffix → kernel})`
   table so a miss is a registry error; defer `dev_dst` past the fused check.

## infr-cpu/src/kernels.rs

_(No CRITICAL/MAJOR: the integer quant math is bit-identity tested, and the
`abs`/`sign` maddubs trick is exact given the `[-127,127]` activation clamp —
agent verified and correctly ruled that out.)_

1. **🟡 `kernels.rs:2917 — `dot_bf16` uses a single serial accumulator**, unlike
   `dot`/`dot_f16` which use 8 independent lanes (doc `2926`) to avoid a
   latency-bound FMA chain → several× slower on the bf16-weight GEMM/attention
   hot path, no numerical reason. _Fix:_ mirror the 8-accumulator chunked
   structure.
2. **🟡 `kernels.rs:32,2933 — silent wrong-answer on shape mismatch.** Every
   256-block K-quant kernel computes `nb=in_f/256` and drops the tail with no
   assert if `in_f%256!=0`; `dot` truncates to `a.len().min(b.len())` masking
   unequal-length caller bugs. Both return a plausible-but-wrong scalar (wrong
   attention score / dot) with no signal — the worst inference failure mode.
   _Fix:_ `debug_assert_eq!(in_f%256,0)` / `debug_assert_eq!(a.len(),b.len())`.
3. **🟡 `kernels.rs:1819 — `vec_dot_q6k_batch_avx512bw` is misnamed** — it's
   `target_feature(avx512bw,avx512vnni)` and built on `_mm512_dpbusd_epi32`
   (VNNI), dispatched only when VNNI is present. An AVX512BW-without-VNNI CPU
   falls to the AVX2 path (256-bit) for Q6*K batch (unlike Q4_K/Q5_K/Q8_0). Name
   misleads dispatch reasoning. \_Fix:* rename `_vnni`; add a real avx512bw Q6_K
   path if that HW matters.
4. **🟡 `kernels.rs:835,1642 — DRY + per-call scratch allocs.** The
   144-byte-block decode/nibble-unpack sequence is copy-pasted ~10× across the
   Q4*K batch kernels (and Q5_K/Q6_K analogs); each
   `\_batch*`call heap-allocates fresh`d_arr`/`sc_arr`/`*\_flat`(+`ilv=vec![0u8;nb*2048]`) — churn inside the matmul row loop that dominates at small `m`(decode). \_Fix:*`#[inline]` `q4k_decode_row(...)` helper; caller-provided/thread-local reusable scratch (or route small-`m`
   to the single-token kernels).
5. **🟡
   `kernels.rs:312 (doc) — Q6_K maddubs pair-sum bound comment says `±8001`**
   but `maddubs` sums two adjacent products → true bound `2·63·127=16002`
   (`-16128`). No bug (still < i16 max) but the comment records half the real
   headroom, misleading anyone re-deriving the no-overflow guarantee (the Q4*K
   analog `100` is correct). \_Fix:* correct to `16002`/`-16128`.

## infr-cpu/src/{lib,pool,turbo,repack,kvquant,moe}.rs

1. **🟡 `lib.rs:1398 — `Op::Sample`panics when`top_k==0`.**
   `k=(top_k).min(len)`; with `top_k==0` (the common "disable top-k" sentinel)
   `k=0`, `if k<len` is taken, and `select_nth_unstable_by(k-1)` underflows →
   `usize::MAX` pivot → panic. _Fix:_ guard `k>=1 && k<len`, or treat `top_k==0`
   as no-truncation (`k=len`).
2. **🟡 `lib.rs:168,411 — `weight_cache` keyed by raw buffer address, never
   invalidated.** Key = `cpu_buf(buf) as usize`, entries live for the backend's
   lifetime. On a reused `CpuBackend` (serve model reload) a freed weight's
   address can be reallocated to a _different_ weight → the closure returns the
   previous model's dequant f32 with no length/content check; also grows
   unbounded across reloads. _Fix:_ key on `(addr,len,dtype)` and/or clear on
   binding change / scope to one `execute`.
3. **🟡
   `kvquant.rs:98 (defensive) — `q5_1_block`omits the`clamp(0,31)` its siblings (`q4_0`/`q4_1`/`q5_0`)
   apply.** If the code ever reached 32 the pack yields **0 instead of 31**
   (catastrophic full-scale error). Verified NOT reachable in practice —
   `d=(max-min)/31` makes the max element exactly 31, and reaching 32 needs
   ~1.6% error while f16 rounding of `d` gives ~0.05% — so this is a
   robustness/consistency nit, not a live bug (same as the shader Q5*1). Worth
   fixing because the failure mode is severe and every sibling clamps. \_Fix:*
   `.min(31)` before masking.
4. **🟡 `lib.rs:201 — `q4k_pack_for`/`q6k_pack_for` do a non-atomic
   check-then-insert.** The lock drops between `get` and re-lock `insert`, so
   two parallel-serve threads can both miss, both build the expensive repack,
   and both `guard.1 += bytes` — double repack + inflated byte accumulator
   drifting past `INFR_CPU_REPACK_MB` (no eviction). _Fix:_ re-check `get` under
   the insert lock; add bytes only when the key was absent.
5. **🟡 perf — hot-path clones/allocs in `lib.rs` op arms.** `Op::Rope` (`826`)
   clones the input twice (`xs.clone()` then `out=xs.clone()`); `Op::Copy`/
   `CopyStrided` (`1486`) clone the _entire_ source to copy a sub-slice;
   `Op::DeltaNet` (`2148`) clones `kf_raw`/`vf_raw` even on the strided path
   that only reads `qf_raw`; and the execute prologue (`395`) zero-fills every
   Internal/Output buffer (incl. the vocab×rows lm*head logits, per token)
   though most op arms immediately overwrite `vals[dst]` with a fresh `Vec`.
   \_Fix:* borrow `&vals[src]` when `src!=dst`; clone only the aliasing/strided
   cases; pre-zero only read-before-write tensors.
6. **🟡
   `lib.rs:989 (doc/parity) — `WriteKv`Q8_0 uses`round_ties_even`while the comment cites llama.cpp's`roundf`**
   (half-away-from-zero); halfway activations differ. Self-consistent with this
   backend's dequant, but if bit-identity to llama / a GPU kernel is expected,
   goldens diverge. _Fix:_ make code+comment agree on the intended rounding +
   parity target.

## infr-core/src/{graph,backend,tensor,loader,pager}.rs

1. **🟡
   `backend.rs:486 — `copy_buffer`default downloads the ENTIRE`src`to copy only`bytes`, and panics if `bytes>src.len_bytes()`.**
   This is the KV prefix-share primitive; the default (CPU/Metal, no override)
   zero-allocs + transfers the whole `max_ctx` KV cache to copy a small prefix,
   and `&tmp[..bytes]` panics on an oversize `bytes` instead of erroring. _Fix:_
   download only the first `bytes` (`vec![0u8;bytes]`), bounds-check
   `bytes<=src.len_bytes()`.
2. **🟡
   `loader.rs:23 — `MetaValue::as_u64`wraps a negative`I64`into a huge`u64`**
   (`I64(-1)→u64::MAX`); a count/size field (block/expert counts, ctx len) read
   this way drives a downstream alloc/loop into OOM/overflow instead of a clean
   "invalid field" rejection. _Fix:_ `u64::try_from(*v).ok()`.
3. **🟡
   `graph.rs:765 — `in_place_inputs`rebuilds a`HashSet`+ rescans all ops per call, and`execute`
   calls it per token** (`infr-cpu/lib.rs:392`, `infr-metal/exec.rs:1583`) — a
   per-token O(ops) scan + heap alloc of a graph-invariant. _Fix:_ compute once,
   cache in `GraphPlan`/memoize on `Graph`.
4. **🟡 `tensor.rs:184 — `MOE_MMQ_PAGED_DTYPES`"mirrors`MOE_MMQ_DTYPES` IN FULL"
   but the drift test only checks subset, not equality.** Adding a dtype to
   `MOE_MMQ_DTYPES` without the paged list keeps the test green while silently
   regressing that dtype's paged prefill to the slow id-GEMV path — the exact
   failure the doc forbids. _Fix:_ assert set-equality (or derive paged = full
   minus an explicit exclusion set).
5. **🟡 `pager.rs:98 — `epoch` map entries never removed on eviction.**
   `take_slot` drops the victim from `resident`/`lru` but not `epoch`, so
   `epoch` accumulates an entry per distinct `BlockId` ever touched (the one
   pager structure not sized to `n_slots`). Bounded by key space, not a true
   leak, but stale entries could in principle mask an id across `cur_epoch`
   wraparound. _Fix:_ drop the victim's `epoch` entry in `take_slot`.
6. **🟡 perf — `backend.rs:445` `capabilities()` returns an owned struct (heap
   `String name`) by value, cloned 8+×/build in `runner.rs`; `backend.rs:417`
   `Bindings` keys buffers by `HashMap<TensorId,…>` though `TensorId` is a dense
   `u32` index** (a `Vec<Option<&dyn Buffer>>` gives hash-free O(1), relevant
   since decode rebinds every step). _Fix:_ query capabilities once/build (or
   return `&Capabilities`); back `Bindings` with a `Vec` indexed by `id.0`.

## infr-core/iquant_grids.rs · infr-engine · infr-prof · infr-prof-rt

_`iquant_grids.rs` is **clean** — pure `static` arrays, compile-time-checked
lengths, no accessor/index/unsafe; callers index within the declared bounds.
`infr-engine` is **clean** — a pure re-export shim. `infr-prof-rt`
disabled-build overhead is correctly zero, the lock-free collector is sound, and
recursion/self-time accounting is correct (all verified). Findings are in the
prof crates only:_

1. **🟡
   `infr-prof/lib.rs:86 — `should_skip`substring-matches`"infr_prof"`+`"skip"`
   over the whole stringified attribute, incl. doc comments.** A fn whose doc
   says e.g. "see infr*prof to skip hot leaves" is silently un-instrumented;
   substring matching also can't tell `infr_prof::skip` from an unrelated
   `skip`. `visit_item_mod_mut` (`129`) similarly skips any module whose attr
   string `contains("test")` — wrongly dropping instrumentation from
   `#[cfg(not(test))]` and
   `feature="test-*"` modules. \_Fix:* match the attribute path/`cfg`meta structurally, exclude`doc`.
2. **🟡
   `infr-prof-rt/lib.rs:324 — GPU-report sort `partial_cmp(...).unwrap()`panics on a NaN`us`**
   (bad timestamp delta) inside the `atexit` reporter, aborting at process
   shutdown. _Fix:_ `total_cmp`. (Recurs across the codebase — see the
   `diffusion.rs`/`cli`/`prof` NaN-sort findings; a shared `by_desc_f64`
   comparator would kill the whole class.)
3. **🟡
   `infr-prof-rt/lib.rs:34,280 — the documented `[dropped]` bucket for over-`MAX_SITES`
   sites is not implemented** — such sites get `count==0` and are
   `retain`-filtered away, vanishing silently rather than being surfaced. _Fix:_
   emit a real count-only `[dropped]` aggregate, or correct the doc.
4. **🟡
   `infr-prof-rt/lib.rs:132 — per-thread `AccumTable`(192 KiB) is pushed to the global`tables`
   vec and never deregistered** → unbounded growth (+ reporter iteration cost)
   with transient threads. Acceptable if thread count stays bounded (typical
   here); otherwise prune dead `Weak` entries in `collect`.
