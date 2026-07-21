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
- **Remaining open:** **44** — 0 🔴, 6 🟠, 38 🟡. (4 findings are explicitly
  **deferred**, not open work: the 🟠 `make_compute_kernel` OOM→Result and three
  🟡 shader/dp4a DRY refactors — each risks the byte-identity gate or the
  recorded stream; see their sections.)

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
- **`infr-llama` diffusion/parallel/util/chat (all 7 findings)** — TDD, +7
  tests. A failed `generate` now pops the just-pushed user turn (no orphaned
  double-user-turn); `parallel.rs checkout` drops the pool lock across the
  `seed_from` GPU submit (reserving the slots) and picks the longest-prefix free
  slot (was first-match); diffusion entropy sort is NaN-safe (`total_cmp`), the
  per-step `escratch` (~1MB) and `prev_argmax` clone are hoisted/reused, and the
  acceptance loop `break`s at the (proven-monotonic) entropy cutoff; per-block
  detok decodes the committed span once (was O(total²)); and
  `with_prof2_suppressed` / a `ChatModel::render` default / one
  `should_use_mtp(cfg)` de-duplicate the four backends (fixing the `wants_mtp`
  drift).
- **`infr-llama` seam/runner (all 7 findings)** — TDD, +9 tests. A new
  `last_written` tracker (the highest KV-written sequence position) + a pure
  `resident_after_gen` helper make the prefix cache record **exactly** the
  tokens whose KV rows exist: the `max_new==0` frontier and any unfed
  grammar-forced tokens are now excluded (were cached as resident → stale-KV
  corruption next turn), while a normal run reproduces the old
  `prompt ++ out[..len-1]` result (byte-identical, no logits/golden path
  touched). Empty non-denoise prompt errors instead of underflow-panicking;
  prompt/canvas token ids are validated `< vocab` (`validate_token_ids`);
  session-stable derivations are cached on `SeamKv` (`SessionStable`) instead of
  recomputed per warm call; host-embed skips the per-token `Vec` when
  `embed_scale==1`; `bind_layer_io` de-duplicates all 5 KV+ weight bind sites.
  _Deferred:_ the `build` 11-positional-bool param-soup (a call-site-wide
  refactor with a transposed-arg risk, zero correctness gain).
- **`infr-llama` seam/mod (all 7 findings)** — TDD, +5 tests. TP/pipeline
  binders exempt `chunk_covered_dense_tensor` (`lm_head`/`token_embd`) from the
  BDA element cap (a large-vocab model no longer hard-rejects under
  `INFR_TENSOR_PARALLEL`/`INFR_PIPELINE`); the process-global `PINNED_UBATCH`/
  `PINNED_KV_Q8` `OnceLock`s become per-session `PlacementPins` via a
  thread-scoped RAII guard (multi-model `infr multi` no longer leaks model A's
  chunk-height/q8 decision to model B; single-model falls back to the old
  behavior byte-identically); MoE KV placement estimate honors auto-q8;
  `TokenEmbd::get` is fallible (corrupt GGUF → error, not `.expect()`); warm
  sessions skip the unused `vulkan_moe_binder`; a fn-instrument macro is removed
  from an `impl Deref`; and `parse_device_list` / `run_dense_oneshot` /
  `dense_multi_gpu_guard` / `cpu_upload_bind`/`metal_upload_bind` de-duplicate
  the multi-GPU + backend-bind boilerplate.
- **`infr-llama` seam/model+weights+sc (all 6 findings)** — TDD, +5 tests.
  `SlotPool::pick` now picks the longest-prefix slot (was first-match, via a
  pure `pick_continuation`); `bench_vulkan`'s unsafe `INFR_PROF2` env race is
  gone — suppression is a non-env `AtomicBool` in infr-prof-rt that the recorder
  reads (also fixes the warmup callers); `generate_metal_spec` reuses a
  persistent `feed` buffer (was O(n²) `committed.clone()` per round);
  `weight_footprint` is memoized on `SeamModel` (one scan/open);
  `spec_accept_stochastic` `.expect()`s the empty-distribution contract instead
  of emitting a bogus token 0; the ~8 `tokenizer.encode` sites and the
  DiffusionGemma upload closures route through shared helpers. No logits/golden
  path touched. **infr-llama's seam core is now fully cleared** (only the parked
  MTP slice remains in this crate).
- **`infr-vulkan` recorder (all 6 findings)** — TDD, +2 tests, **gpu_seam
  byte-identity verified on the RX 7900 XTX** (MoE-mmq resident+paged, GEMV
  row1/mrow/mw/id, and weight-addr goldens all match host). Per-dispatch
  `INFR_GEMV_*` env reads are resolved once into a `GemvKnobs` `OnceLock` (same
  routing); `dispatch3`/`bind_descriptors` use stack arrays (no per-dispatch
  heap `Vec`); `finish`/`finish_nowait` free the cmd buffer + pools on their
  error paths (was a leak); the two ~180-line `matmul_mmq_experts`/`_paged`
  tables collapse to one `moe_mmq_desc` table (a drift test asserts the same
  `(kernel,nbind)` for every `MOE_MMQ_DTYPES`); the descriptor pool is
  proportioned so sets+descriptors exhaust together. #5: the 2 genuinely
  parity-only `_at` fns are gated behind a new `parity` feature — the other 9
  carried a **stale** "not wired" doc but are in fact production-wired
  (correctly left `pub`).
- **`infr-vulkan` adapter (all 6 findings)** — TDD, +3 pure tests, **gpu_seam
  byte-identity verified** (GEMM prefill `nc_gemm`/`warp_gemm`, attention/KV
  `kv_addr` 15, MoE-mmq all match host). Batched-MoE `counts` drops the
  redundant host-blocking calloc (device-zeroed content unchanged, ~27µs/layer
  sync gone); the static split-K attention now bounds `n_chunks ≤ 1024` (was an
  OOB `attn_combine wexp[1024]` write above ~524k keys — the fix is proven
  identical to the old formula for all realistic spans); `GatedAct::Silu`
  rejects a strided gate instead of computing silently-wrong; cross-dtype
  `Op::Copy` with `src_off!=0` returns `Err` instead of panicking;
  `INFR_FLASH_MIN_ROWS` is read once; and
  `split_k_plan`/`native_warp_gemm`/`with_padded_dst` de-duplicate the three
  GEMM tiers (identical tile/split decisions).
- **`infr-vulkan` lib + pcache (all 7 findings)** — TDD, +5 tests, **gpu_seam
  byte-identity verified** (BDA weight-addr 26, attention/KV 15, MoE, GEMM). An
  `InstanceCleanup` RAII guard (disarmed before `Ok`) frees the device/instance/
  cmd-pool on every recoverable `new_selected` error path (was leaked for
  process life on the CPU-fallback path); device-local zero-init rounds the fill
  up (`fill_span`) so a non-4-multiple buffer's tail can't leak recycled VRAM;
  the write-only `uma_spilled` counter is removed;
  `alloc_arena_bda`/`bda_weight_alloc` reuse the already-stored `own_addr` (no
  re-query); the UMA-spill path skips its duplicate VRAM budget check; and
  `pcache` uses a unique temp suffix + a per-instance nonce on the
  poison-tripwire marker (no cross-thread/backend collisions). Success/alloc
  path byte-identical.
- **`infr-vulkan` ops + pager (6 of 7; #2 deferred)** — TDD, +7 tests,
  **gpu_seam byte-identity verified** (pager gemv/mmq/multi, MoE, attention).
  `kernel`/ `kernel_sg` build under one lock (`entry().or_insert_with`) — no
  double-compile/leaked-pipeline race; the cache hit `debug_assert`s the
  request's `n_buf`/`push_size`/spv-hash match (silent-mismatch guard);
  `GpuPager::new` + the `pub` accessors return `Err` on bad dims / unregistered
  buffers (were `assert!`/`.expect()`); the triplicated LUT evict+insert becomes
  one `record_placement` (silent-zero-safe, byte-identical);
  `touch_role`/`stage` reuse scratch buffers;
  `expert_bytes`/`pager_stats_enabled`/`stats_suffix` de-duplicate the two
  sessions. #2 (make_compute_kernel OOM→Result) deferred — 155 call sites across
  127 `()`-returning fns; panic messages improved instead.
- **`infr-vulkan` gemm + matmul + linear (all 7 findings)** — TDD, +4 tests,
  **gpu_seam byte-identity verified** (GEMV row1/mrow/mw, MoE id-gemv
  real-dims + new-formats all match host). Every `*_kernel_name` gate now
  DELEGATES to its `*_build_spv`/`*_spv` source (id/idm/mmv/mw/mrow/embed) so a
  name and its shader can't drift into a mid-inference panic — each table was
  diffed identical before delegating and a drift-guard test asserts equality
  across all dtypes; `Iq4Xs` gets an explicit `native_mmv_mrow_res_supported`
  predicate (no phantom residual panic) + the false adapter comment fixed;
  `matmul_f32` is `#[doc(hidden)]` with the per-call `println!` gone and shape
  math overflow-checked; push constants use `to_le_bytes` (matching the shader
  contract); the no-op `v!` macro and the re-inlined `spv_words` are removed;
  `moe_expert_floor_covers_dense_set` derives from `native_dense_dtypes()` (+ an
  exhaustive-`DType` guard).
- **`infr-vulkan` misc shaders (6 of 7; #6 DRY deferred)** — TDD, **gpu_seam
  verified** (`dg_eb_sample`, `chunked_delta_math`, `sample_topk`, MoE mmq/id +
  ragged-bucket `pager_mmq`/`weight_addr`, `add_bias` all pass). `dg_eb_sample`
  argmax now carries the lower-index tie-break (matches host/`argmax.comp` — the
  golden passes unchanged, no re-bless needed); `rope.comp` writes only the
  un-rotated tail (halves KV-store traffic, byte-identical); the `-1.0/0.0`
  `-inf` sentinel is `-1e30`; `part[8]` cross-subgroup arrays sized to
  `part[16]`; `embed_gather` gets a host `ne%32==0` guard; `deltanet_gates` uses
  `subgroupInclusiveAdd` (sg32-pinned, within tolerance); `moe_bucket_scan`
  drops the fused `fill[]=0` (zeroed by a separate overlapping dispatch — MoE
  goldens byte-identical).
- **`infr-vulkan` attention/flash shaders (7 of 8; #7 deferred)** — **gpu_seam
  verified** (flash stage/warp/coopmat/matches-cpu + kv*addr + kv_q8 goldens
  pass, run serially). The direct `coopMatLoad` K/V over-read past `kv_len` is
  made SAFE by tightening the host `flash_geom` gate to `kv_len.div_ceil(64)*64
  <=
  att*cap_rows`(a non-ring KV cache allocates non-64-aligned rows; flash reads 64-aligned tiles) — the masked over-read columns contribute nothing;`attn_flash.comp`documents the mpad precondition (keeps writing all BM rows to stay bit-identical with the staged/split-K paths — a store guard there was reverted after it broke the STAGE==direct golden); host debug-asserts guard`attn_pv` `tilesN`, `attn_combine` `ntile`, and `attn_partial_mrows` `chunk`; `quant_kv`Q5_1 gets the`clamp(0,31)`(never triggers); dead`MAXS`removed. \_Pre-existing note:* the flash parity LIB tests race on process-global`INFR_FLASH*\*`env under parallel cargo threads — pass with`--test-threads=1`.
- **`infr-vulkan` GEMM/GEMV shaders (7 of 8; #6 dp4a DRY deferred)** —
  **gpu_seam byte-identity verified** (`weight_addr`/`gemm_proj` 28,
  `sample_topk`, MoE-mmq, `nc_gemm` all match host). `gemm_proj.comp` now stages
  weights k-inner (coalesced global loads) and reads the arena through
  `native_weight_addr.glsl` so the compiler selects the scalar-base **saddr**
  `global_load` instead of the divergent-index 64-bit deref (the documented
  ~2.2× streamed regressor) — both bit-verified identical; the padded-C store
  contract is documented (all callers pad); `native_gemv_rm_v2` `reg_part` is
  sized `THREADS/16` (was OOB under WG128) + dead `tot` gone;
  `native_mmv_mrow`'s unused `part[]` is `#ifndef OUTS4`; `moe_sample` top-k
  gather is two-pass so a threshold tie can't evict a strictly-greater logit;
  and the superseded `gemm_dp4a`/`gemm_coopmat` probe kernels are removed
  (grep-proven never dispatched).

### Highest-priority (production default paths)

| #      | Sev | Location                           | Issue                                                                                                          |
| ------ | --- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| ~~1~~  | ✅  | `infr-hub`                         | ~~Downloaded blob never sha256-verified~~ — **FIXED** (`1263bcc`, + full hub slice).                           |
| ~~2~~  | ✅  | `infr-llama chat/mod.rs`           | ~~Generate error orphans the user turn~~ — **FIXED** (`Err` arm pops the user turn).                           |
| ~~3~~  | ✅  | `infr-server lib.rs`               | ~~Streaming swallows errors as `stop`~~ — **FIXED** (error frame + panic-safe `DoneGuard`).                    |
| ~~4~~  | ✅  | `infr-server lib.rs`               | ~~No per-request cancellation~~ — **FIXED** (cancel latch → `req.abort()` frees the slot).                     |
| ~~5~~  | ✅  | `infr-llama runner.rs`             | ~~Prefix-cache records unmaterialized KV rows~~ — **FIXED** (`last_written` tracker + `resident_after_gen`).   |
| ~~6~~  | ✅  | `infr-vulkan adapter.rs`           | ~~Static split-K `n_chunks>1024` overruns `wexp[1024]`~~ — **FIXED** (bounds `n_chunks ≤ 1024`).               |
| ~~7~~  | ✅  | `infr-vulkan ops.rs`               | ~~Kernel-cache double-checked lock double-compiles + leaks~~ — **FIXED** (single-lock `or_insert_with`).       |
| ~~8~~  | ✅  | `infr-llama sampling.rs`           | ~~Repeat penalty per-occurrence~~ — **FIXED** (`70bbe4e`; now per-distinct-token).                             |
| ~~9~~  | ✅  | `infr-vulkan shaders dg_eb_sample` | ~~argmax reduce drops the lower-index tie-break~~ — **FIXED** (matches host/`argmax.comp` on ties).            |
| ~~10~~ | ✅  | `infr-gguf lib.rs`                 | ~~Corrupt GGUF `pos+n` overflow panic~~ — **FIXED** (`checked_add`/`checked_mul` → `Error::Loader`).           |
| ~~11~~ | ✅  | `infr-cli main.rs`                 | ~~`--dev` can't override inherited backend env~~ — **FIXED** (clears siblings; unified precedence).            |
| 12     | 🟠  | `infr-metal exec.rs:2836`          | `Op::Rope` snapshots positions on the replay tape → **frozen RoPE after token 0** (llama-family Metal decode). |

The 6 remaining 🟠 majors: the metal `Op::Rope` replay-tape bug (#12, the last
open production major); the gated multi-GPU `p2p` `EXCLUSIVE` sharing + F32-only
all-reduce; the parked-MTP `catch_up` off-by-one + wasted-logits; and the
deferred `make_compute_kernel` OOM→Result. Everything else below is 🟡. Full
detail per module.

### Cross-cutting themes

- **`partial_cmp(...).unwrap()` NaN panics** recur in ≥5 files (diffusion, cli
  sweep, prof-rt, sampling) — a shared `by_desc_f64`/`total_cmp` comparator
  kills the class.
- **`assert!`/`.expect()`/`panic!` on recoverable input** (unregistered pager
  buffers, `GpuPager::new`, `Op::Copy` src_off, `make_compute_kernel` OOM, GGUF
  parse, `Op::Sample` `top_k==0`) — should return `Err`.
- **Name-table vs SPV-table drift guarded only at runtime by `.expect()`** —
  FIXED in vulkan `gemm.rs`/`linear.rs` (name gates now delegate to the SPV
  source); still open in Metal `exec.rs`.
- **Per-render / per-warm-call recomputation** (jinja env rebuild,
  session-stable seam derivations, capabilities clone, GGUF re-parse ×5 in the
  CLI).

<!-- SLICES APPENDED BELOW AS THEY ARE VERIFIED -->

## infr-vulkan/src/ops.rs + pager.rs

_6 of 7 findings fixed (see Resolved log); the one below is **DEFERRED**._

1. **🟠 `make_compute_kernel` panics on recoverable driver/OOM errors**
   (`.expect()`/ `panic!`); `kernel`/`kernel_sg` return a bare `ComputeKernel`,
   so a late `OUT_OF_DEVICE_MEMORY` on kernel compile aborts the process.
   _Deferred:_ threading `Result` ripples to **155 call sites across 127
   `()`-returning fns** (ops/gemm/recorder) — a byte-critical mass migration
   disproportionate to a rare OOM path. Interim: the panic messages now name the
   kernel + flag it as a recoverable alloc failure. A full `Result`-ification of
   the recorder dispatch surface is its own focused effort.

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

_7 of 8 findings fixed (see Resolved log); the one below is **DEFERRED**._

1. **🟡 perf/DRY** — `attn_combine.comp:36,40` recompute `mm`/`l` over all `nch`
   in every one of 32 lanes; the causal-mask + online-softmax rescale block is
   copy-pasted across 5 flash kernels; ring `RROW`/`rcap` + Q8/f16 vec4 readers
   are duplicated. _Deferred:_ `attn_combine` is dispatched WITHOUT a pinned
   subgroup size, so a `subgroupMax`/`subgroupAdd` reduce would be wrong on a
   16-lane device without adding `requiredSubgroupSize=32` (a host pipeline
   change with portability risk), and the `l` (weighted-sum) reduce is only
   within-tolerance, not byte-identical — both risk the byte-identity gate. The
   flash mask/rescale include-refactor would perturb the 5 kernels' numerics.

## infr-vulkan/shaders — GEMM / GEMV / MoE-expert matmul

_7 of 8 findings fixed (see Resolved log); the one below is **DEFERRED**._

1. **🟡 DRY — the int8 dp4a decode helpers (`rb`/`ru16`/`f16tof32`/`k4` +
   per-format `dpsub`/`wdec`) are copy-pasted across the `native_mmv*` +
   `native_gemm_mmq_*` shaders; the `KV_IQ4NL_W` table duplicated ≥4×.**
   _Deferred:_ the helper SETS differ per file (each activates exactly one
   `FMT_*`; `k4` only in Q4K/Q5K, odd-stride readers only in some), spread
   across ~15 mmq + 4 mmv shaders. A shared `native_dp4a.glsl` include would
   have to reconcile those conditional blocks + add build.rs include wiring, and
   any textual drift risks the compiled SPV on the byte-identity-critical
   GEMM/MoE goldens.

## infr-vulkan/shaders — norm / rope / activation / sampling / MoE-routing / misc

_6 of 7 findings fixed (see Resolved log); the one below is **DEFERRED**._

1. **🟡 DRY — two shader pairs (`qk_norm_rope` vs `_interleaved`, `sample_topk`
   vs `moe_sample`) look duplicated but aren't cleanly mergeable.** On
   inspection they differ by more than the audit assumed: `qk_norm_rope` carries
   an extra `kcap` push-constant + the SWA ring-modulo its interleaved twin
   lacks (different PC layout); `sample_topk` is two-stage (`PASS1`/`PASS2` + a
   `CHAIN` variant, buffer-sourced `u`) while `moe_sample` is single-stage with
   `u` in the push constant (different binding layouts). _Deferred:_ folding via
   `-DINTERLEAVED` would have to reconcile the PC/binding structs and the
   build.rs compile list, risking the recorded push-constant/descriptor stream
   for a 🟡 DRY gain.

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
