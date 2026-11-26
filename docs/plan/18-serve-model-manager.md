# 18 — A model manager: models that load, demote and come back

**Why.** The owner asked (2026-11-26) for a server that is configured with more
models than its GPUs hold at once: it keeps the hot ones in VRAM, unloads the
cold ones, and switches between them as requests arrive, integrated with the
tiered loading the streaming plans build. The owner then named a second client:
tests reload the same model's weights for every run (plan 17, slice 3), and the
same component should keep a model loaded across runs in one process. It is the
next priority after the slice in progress, and the part the tests need comes
first. The file keeps its name, although its first client is the test suite.

Nothing here is built. Statements marked **guess** are guesses; a cost marked
"not measured" names the measurement that would give it.

## Decided by the owner (2026-11-26)

- The manager is a **library component below the server**, in `infr-llama`
  beside `SeamModel` and `ParallelSeam`, usable from test support with no HTTP
  layer. The tolerance gate is its first client, `infr serve` its second.
- Build order: keep a loaded model alive across runs in one process, evict it
  when another model is requested, reset per-run state, and show the result
  bit-identical to an isolated run. The serve features (the budget across
  models, demotion, request queueing, config keys) follow as later slices on the
  same core.
- It joins the tiered-loading plans: 05 (streaming), 06 (eviction), 08 (streamed
  at resident speed), 14 (expert placement) and 15 (KV tiers). Decisions those
  plans took are inherited, not re-argued.

## What exists today

Read in the tree on 2026-11-26.

**Serving.** `infr serve` hosts **one** model (`cmd_serve` in `infr-cli`).
Several models are hosted by `infr multi` (`cmd_multi`): Vulkan only, each spec
`MODEL[@VulkanN]` loaded in turn at startup through `SeamModel::load_with` and
`ParallelSeam::new_on` on its own device index (round-robin when no suffix),
wrapped in a `ParallelGenerator`, and handed to `AppState::multi`. The listener
opens only after every model has loaded, the set never changes, and every model
gets the first model's sampling defaults (`apply_model_sampling_defaults` runs
for `i == 0` only) and the same `Config`.

- `AppState.models` is an `Arc<Vec<ModelEntry>>`. A `ModelEntry` holds `id`,
  `engine: Option<Arc<dyn ChatGenerator>>` (`None` only in `AppState::headless`,
  for tests and `/v1/models`), a per-model `slots` semaphore with `capacity`
  permits (`--parallel N`) and a `prepares` semaphore of the same size.
- `AppState::route` takes an exact id match, else the first entry: an unknown or
  empty `model` is served by the default model and never answers `404`.
- A request is prepared before it queues (`prepare_bounded` runs
  `ChatGenerator::prepare`: template render, tokenization, prompt count), then
  waits for a slot permit (`acquire_slot`, FIFO, counted by `QueuedGuard`), then
  generates on a blocking thread holding the permit inside a `Slot`
  (`run_unary`, `run_streamed`). A client that hangs up while queued drops the
  future and its place. `serve.request_timeout_secs` is armed only after the
  slot is taken: **the queue wait has no bound today**. A streamed request waits
  for its slot before the `200` is sent.
- `/health` is an unconditional `200`. `/v1/models` lists every entry's id.
  `/status` and `/metrics` report per model the backend, device, context per
  slot, slots and device memory (`GeneratorStatus`, `metrics.rs`).
- **Shutdown.** The CLI handles `SIGINT` and `SIGTERM` (`infr_plat::signal`) by
  setting the latch `infr_core::shutdown` keeps; `serve_state` stops accepting
  (`shutdown_latched`), every generation stops at its next abort poll
  (`sampling::abort_requested` ORs the latch in, also for the warmup generation,
  which passes no request), and the backends drop after `serve` returns. No
  other signal is handled: `SIGHUP` takes its default action, which ends the
  process without that drain (default disposition, not run).

**What one hosted model holds.** `ParallelGenerator` holds the engine, an
`OaiRenderer` (which opens the GGUF and loads the tokenizer itself:
`OaiRenderer::open_with_tokenizer`), a `WeightWatch` and the vocabulary size.
`ParallelSeam` holds the `SeamModel`, its own `VulkanBackend`, the slot pool
(`SeamKv` per slot), a `StepGate` when it has more than one slot, `max_ctx` and
its own `PlacementPins`. `SeamModel` holds the parsed and mapped `Gguf`, the
model `Config`, the `EngineConfig`, the tokenizer, a lazily built f32 token
table (host-gather paths only), the memoised footprint and, for the CPU, a kept
`CpuResident`.

- The weights are uploaded by the warmup generation in
  `ParallelSeam::init_slots`, which also compiles the pipelines and re-clamps
  the context against the room measured after the upload
  (`reclamp_ctx_to_live_room`). The other N-1 slots are forked eagerly from slot
  0 and share its `Arc<SeamWeights>`.
- Each `VulkanBackend` owns its own instance, logical device and allocator
  (`new_on_with`'s doc), its resident weight arena (`bda_weight_arena`), and at
  most one MoE pager session and one dense pager session (`MoePagerCell`,
  `DensePagerCell`), whose pool specs hold the weights' host tier
  (`Arc<HostPager>`). Two models on one device are two devices to the driver,
  with separate `StepGate`s: nothing arbitrates their submits.
- Dropping the backend frees all of it: `VulkanShared::drop` destroys the
  pipelines and kernels, saves the pipeline cache to its file and destroys the
  device (`backend_drop_frees_device_after_moe_pager` guards the Arc cycle that
  once kept a dropped device alive). What survives a drop is on disk: the
  `vkPipelineCache` file (`pcache.rs`, over `infr_core::kernel_cache`) and the
  driver's own shader cache. A backend whose GPU wait timed out or met a lost
  device is **poisoned** (`gpu_wait.rs`): every later call returns the same
  error, and its drop leaks the device and its memory instead of freeing them.

**Host copies.** A resident model's host copy is the file mapping, and by
default not even that: `paging.release_uploaded` (default `true`) releases a
resident weight's mapped pages once its bytes are in VRAM, on a discrete GPU.
Whether the OS keeps those file pages in its page cache after the release is
**to check**. The host tier (`HostPager` arenas) exists only for streamed models
and dies with the backend that owns the pager session.

**Budgets are read per model, at its load.** The VRAM guard
(`check_vram_commit`) admits each allocation against the live
`VK_EXT_memory_budget` reading; without a live budget it counts only this
backend's own bytes, so two backends on one device in one process would not see
each other. Placement and the fit read the room once per load
(`moe_expert_budget`, `kv_fit_ctx`, `vulkan_slot_ctx`, the re-clamp), and the
host tier is sized from `hostmem::available_bytes`, an OS probe, keeping half of
it free (`HEADROOM_FRACTION`). So a second model is sized against whatever the
first left, and an automatic context takes the device's whole fit.

**Tests reload everything for every run.** `SeamModel::forced_logits` copies the
engine configuration, applies the run's edit and calls `forced_run_vulkan`,
which builds `VulkanBackend::new_with(ec)` and binds every weight again;
`greedy_tail_vulkan` does the same, and `forced_run_cpu` builds a fresh
`CpuBackend` (`reference_with` or `new_with`). The gate's `evaluate_with` loops
prompts outside and configurations inside, so consecutive runs never share a
configuration. The precedent for keeping weights is already in the tree:
`CpuResident` keeps a CPU backend and its `BoundWeights` between generations
(`SeamModel::with_cpu`), and `runner::generate_dense_backend_bound` takes a
`prebound: Option<&BoundWeights>` that `forced_run` passes as `None`. Bound
weights are valid only on the backend that bound them.

## Which run settings can keep the weights

A `VulkanBackend` reads its configuration from the `Arc<Config>` it was built
with, for its whole life:

- **Capability masks** (`kernels.vulkan.coopmat`, `f16`, `i8_dot`) are applied
  when the backend is built (`config_masks_the_capability_probe`), and the
  weights' layout depends on the capabilities (`session_stable`:
  `fuse_gu_decision` and `fuse_qkv_decision` read `caps.combined_gu`).
- **Tuning parameters**, the queue family and the subgroup width are resolved
  once at creation (`tuning::resolve`, `VulkanBackend::tuning`).
- **Kernel-tier switches** (`mmq`, `mrow`, `mrow16`, `mmq_fallback`,
  `no_replay`, `i8_coopmat`, among others) are read at dispatch, but through
  `be_.cfg()`, the backend's own copy.
- **The prefill chunk and the KV formats** are read by the seam (placement pins,
  `kv_rows_at`, the runner); nothing in `infr-vulkan` reads `device.ubatch`.

So today **every edit forces a new backend and a new upload**. What each would
need to stay resident:

| Run setting                                                | Gate configurations                   | Weights can stay?                                                       |
| ---------------------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------- |
| None (same configuration, another prompt)                  | every prompt of every configuration   | yes, with no code change to the backend                                 |
| `device.ubatch`                                            | `*-ub32`                              | yes once the chunk is a per-run input (it is seam state already)        |
| `kv.type_k` / `kv.type_v`                                  | `gpu-kv-*`                            | yes: KV is allocated per run                                            |
| `no_replay`, `mmq`, `mrow`, `mmv_mw`, `i8cm_kq`, `dp4a_kq` | `*-static`, the float configurations  | yes once the backend's dispatch knobs are per run; audit each read site |
| `i8_dot`, `coopmat`, `f16` (capability masks)              | float (masks `i8_dot`), `*-nocoopmat` | **no**: another device instance with other capabilities                 |
| Tuning parameters, queue family, `device.sg_pref`          | `infr tune check`, tuning-mode runs   | **no**: resolved at creation                                            |
| `kernels.qkv_fuse`                                         | none                                  | **no**: changes the uploaded layout                                     |
| Another device (`device.dev`)                              | the same gate on the other GPU        | **no**: another device                                                  |
| CPU reference, CPU production                              | the reference, `Cpu` runs             | each its own backend; CPU weights are bound from the mapping            |

With per-run knobs split off, the gate's configurations (`GPU_CONFIGS`) fall
into three backend keys per device: default capabilities, `i8_dot` masked, and
`i8_dot` plus `coopmat` masked. Whether the kernel-tier switches change anything
that is built once per backend (a pipeline variant cached under a name that
ignores the switch) is **to check** in slice T2's audit.

## The core

One component, `infr_llama::residency` (name open), with three parts:

- **Resident:** a backend plus what was bound on it (`BoundWeights`, the pager
  sessions, and for serve the slot pool), keyed by the model instance, the
  device, and the construction-time part of the configuration.
- **Lease:** a guard a user holds while it runs on a resident. A leased resident
  is never demoted or parked. Serve's lease is the slot permit inside `Slot`; a
  test takes one per run.
- **Ledger per device:** each resident model's **share** of the device, what it
  was measured to take (the room before its load less the room after, the way
  the runner reads `kv_committed`), what loading another would cost (the fit's
  estimate, which must equal the allocation), and the order to demote in.

Per-run state a reused resident must not carry from one run to the next, from
the code:

- **KV and recurrent state:** `forced_run` starts from `state = None` and
  allocates its own; serve's slots carry KV on purpose (the prefix cache) and
  are reset or recycled by `checkout`.
- **Placement pins:** `forced_run_vulkan` enters `enter_model_scope`, which
  makes fresh `PlacementPins` when no scope is in force; a resident must keep
  that, since the pins latch (`kv_q8` is a `OnceLock`, the chunk is pinned
  once).
- **Backend state that accumulates:** `built_kernel_names` (a reused backend
  would report earlier runs' kernels in `ForcedLogits::kernels`, which tests
  assert on: needs a per-run delta), `decode_recordings`, the activation peak,
  the KV spill reserve (`set_kv_spill_reserve`), the submit splitter's measured
  cap, the pager sessions' residency, pooled activation buffers. Each is either
  reset at the start of a run or shown not to change results; the splitter
  changes where submits end, which is **guess**ed not to change any value.

## The test client

Built first, so in detail.

**Who gains, and in one process or not.** `cargo nextest` (what CI and
`AGENTS.md` run) starts a process per test; plain `cargo test` runs a binary's
tests on parallel threads in one process. Reuse therefore spans one test
function, and these are the ones that load once and run many times:

| Caller                                                                                                                          | Process                          | What reuse saves                                                                |
| ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------- |
| `measure_gpu_tolerance` (`tests/gpu_tolerance.rs`, through `tests/support/forced.rs`)                                           | one, every `INFR_TOL_MODELS` key | per model, every configuration on every prompt, and the greedy runs             |
| `infr tune check` (`tune_check::run`: `gate::check` over every configuration, then the greedy check)                            | one, one model                   | the same, on the shipped binary                                                 |
| The `gpu_seam_tolerance_*` tests                                                                                                | one per test                     | the forced runs of one model inside each                                        |
| The CPU tests in CI (`forced_run_reproduces_the_cpu_reference_on_every_path`, `tolerance_gate_goes_red_on_a_scaled_weight_cpu`) | one per test                     | little in time (synthetic models); they are where the identity check runs in CI |

**With plan 17's reference cache.** On a warm cache `Reference::cached` serves
the reference, and the CPU reference backend is built only for a miss and for
the canary (the shortest prompt recomputed). Residency adds little there: the
misses and the canary of one model share one `CpuReference` resident. The time
it saves is on the Vulkan side, which plan 17 does not touch. `infr tune check`
calls `gate::check`, which passes no reference cache, so its CPU reference is
computed every time (plan 17 slice 2's "remaining callers").

**The API, as plan 17 shaped its cache.** The library reads no environment and
takes an explicit object:

- `Residency::new(mode)` makes an empty set of residents; `mode` is `Off`, `On`
  or `Verify`. It is owned by the caller and dropped by it.
- `Residency::lease(&self, model: &SeamModel, on: ForcedOn, ec: &Arc<EngineConfig>) -> Result<Lease<'_>>`
  returns a resident for that model, device and construction-time configuration,
  building it when there is none and evicting another model's unleased resident
  on that device first. A `Lease` hands out the backend and its `BoundWeights`,
  and resets the per-run state when it ends.
- `SeamModel::forced_logits` and `greedy_tail_vulkan` gain a twin taking
  `Option<&Residency>`; `None` is today's fresh backend per run. The gate
  functions take it beside the cache, as `check_cached` takes
  `Option<&RefCache>`.
- `Residency::stats()` reports loads, reuses and evictions, for the timing line
  plan 17 slice 1 added.
- The model instance is the key's first part (a per-instance id given at
  `load_with`), so a model with a planted bug (`SeamModel::plant`) never shares
  a resident with the clean one.

**Off and verify.** Test support decides the mode from one variable,
`INFR_TOL_RESIDENT` (`off`, `on`, `verify`), in one place, the way `ref_cache`
reads `INFR_TOL_REFCACHE`. `off` is today's behaviour. `verify` runs every run
both ways, on the resident and on a fresh backend, and fails on any bit that
differs, so any contributor can check that reuse changes nothing. Recommended
defaults: `on` locally, `verify` when `CI` is set (the CPU tests are small, and
CI is then where the identity is held on every push).

**VRAM between tests.** Residents are not a process-wide static, unlike the
reference cache: under `cargo test` tests share the process and run at once, and
a resident kept for one would sit in another's device room. The test function
owns its `Residency` and drops it at its end. Tests that read the device's free
memory (`seam::model::vram_limit_tests`, the MoE automatic tier, which reads
`vram_info` once at load, plan 14's start-up wait) run with residency off or
clear it before their load. Plan 14 and backlog B-PP-SLOW-RUNS record a run
reading another process's memory as in use for about 150 ms after that process
exits; whether a backend dropped in the same process is out of the live budget
at once is **to check**, and T1's test asserts it (the room after a drop equals
the room before the load).

## What tests need, and what serve needs

| Need                                              | Tests | Serve |
| ------------------------------------------------- | ----- | ----- |
| Keep weights bound across runs of one model       | yes   | yes   |
| Evict one model's resident when another is loaded | yes   | yes   |
| Several backend configurations of one model       | yes   | no    |
| The CPU reference and CPU production backends     | yes   | no    |
| Exact isolation of each run                       | yes   | no    |
| KV kept across requests (prefix cache)            | no    | yes   |
| Several models resident at once, concurrently     | no    | yes   |
| A byte budget across models, demotion, queueing   | no    | yes   |
| Config keys, `/status`, `/metrics`                | no    | yes   |

The gate also needs its loop turned around (configurations outside, prompts
inside), or every run evicts the one before; the order changes no score when
each run is isolated, which `verify` shows.

## Demotion along the tiers, or park and load

The owner's requirement is that a cold model moves down the tiers the weights
already have, not only on and off. The tiers, top to bottom: resident; streamed
with a slot for every block; streamed with fewer slots, the host tier holding
the rest; streamed reading the mapping only; parked (no backend). Read from the
code:

- **Resident and streamed are fixed at bind time.** The binder that
  `vulkan_moe_binder` returns decides per tensor at the cold init: a group in
  `dense_plan` gets a placeholder registered with the dense session, any other
  is uploaded into the resident arena. `SeamWeights` is then shared immutably by
  every slot. A dense session exists only when placement chose streaming. A
  model loaded resident cannot become streamed without a rebuild.
- **Resident weights cannot be given back one by one.** The BDA weight arena
  bump-allocates tensors into blocks and its doc says sub-tensors are never
  freed individually; the arena keeps each block's handle until the backend
  drops.
- **The weights' layout is the same either way.** Plan 08 checked the fused QKV
  and gate/up decisions on both paths; `session_stable` does not read the
  placement. And on the default configuration a streamed model's logits are the
  resident ones bit for bit at the same prefill chunk (plan 08 step 4's
  `streamed_quant_takes_the_resident_kernels`; backlog B-PLACEMENT-TOKEN-FLIP's
  "weights streamed, KV resident" row). Streamed placement settles on a smaller
  chunk on its own, so the chunk must be pinned to the resident one.
- **A streamed model whose cache holds every block runs near resident speed.**
  Plan 08, RX 7900 XTX, tg128: Qwen3-1.7B 469 against 479 t/s, ERNIE-4.5-0.3B
  444 against 450; prefill within noise; on the integrated GPU, Qwen3-1.7B 27.54
  against 27.42. SmolLM3 does not replay (NoPE) and is unaffected.
- **A pool's slot count is fixed for the session.** `GpuPager::new` allocates
  one contiguous arena of `n_slots * slot_bytes` (`alloc_arena_bda`) per pool,
  and plan 14 records that it cannot grow in place. The pager can stop using
  slots (`evict` returns one to the free list) but the arena's memory is only
  freed with the pool. Nothing equivalent to plan 15 R3's shrinkable host arena
  exists for VRAM, and no plan builds one.
- **Promotion back is the pager's ordinary fill.** A dense pool fills a free
  slot on a miss (`DensePagerSession::stage`), and replay returns when every
  block holds a slot (`holds_every_block`, plan 08 step 3). For MoE, plan 14's
  slices A (alias map), P (promote a layer) and D (drop the pager) are the
  promotion path, and its fill reader already fills free slots in the
  background.
- **One device per model rules out one pager or one allocator across models.** A
  buffer of one `VkDevice` cannot be used by another, and each backend holds at
  most one session of each kind. A single budget across models is still
  possible, as host-side accounting in the manager's ledger over the live budget
  each device reports. Sharing one backend between models would need several
  sessions per backend and one `Config` for all of them; out of scope.
- **What streams is limited today.** Only llama-shaped projections stream (plan
  05 item A1: LFM3's convolution projections, about 13% of its parameters, stay
  resident; qwen35's DeltaNet projections and others are to confirm), and an MoE
  model pages its experts only, its dense part staying resident
  (`moe_expert_budget`'s doc). A fully held paged MoE model runs without decode
  replay (plan 06 step 5 is unbuilt: 177 against 186 t/s, quoted in plan 14),
  and plan 05 holds paged MoE to the resident path by a loose comparison, not
  bit for bit.

**What partial demotion buys.** Taking a cold model's slots away a few at a time
frees VRAM without a reload stall, and both models keep serving. What it costs
is decode speed once the model evicts: plan 08's evicting controls (RX 7900 XTX;
budgets 800M, 512M and 1G) decoded Qwen3-1.7B at 58.8 t/s against 479,
ERNIE-4.5-0.3B at 72.6 against 450, SmolLM3 at 21.6 against 228, with prefill
unchanged. A short reply to a long prompt hardly notices; a long reply may cost
more than reloading the model would have. Choosing between the two needs the
reload times T0 measures and the trace M4 records.

**Conclusion.** The core design is **shares**: the ledger gives each model a
byte share of its device, the model's placement fills its share by tiers, and
demoting or promoting a model is changing its share. Parking is a share of zero.
Whole-model park and load is built first, because every ladder ends in it (a
streamed model still holds its non-streaming groups, its pools' floors, its KV
and its pipelines), it works for every kind of model (MoE, CPU, Metal, models
whose projections do not stream), and it needs no new mechanism. Partial
demotion is the refinement after it, for dense Vulkan models first, because it
needs a VRAM pool that can give slots back (slice D2 here, no plan has it),
every projection streaming (plan 05 item A1) for full effect, and for MoE the
paged replay (plan 06 step 5), plan 14's A, P and D, and a bit-identity check of
paged against resident MoE. Since shares are the interface from M3 on, the D
slices change what a model does inside its share, not the manager.

## Life cycle (serve)

| State    | Holds                                                                                                             | Left by                                                         |
| -------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Declared | the spec; the file resolved (and pulled) at startup                                                               | open                                                            |
| Parked   | `SeamModel` (parsed, mapped), `OaiRenderer`, `WeightWatch`: enough to list, prepare and count a request           | load                                                            |
| Loading  | a backend being built, weights uploading, slots forking                                                           | success: Resident; error: Failed                                |
| Resident | the backend, weights within its share (resident or streamed), slots, pipelines, the host tier of a streamed model | demotion (share lowered), park (share zero); never while leased |
| Failed   | the load error, or a poisoned backend, kept for `/status` and every waiter                                        | retry after a back-off; a poisoned one only by restart          |

A model is busy while a slot permit is held (`ModelEntry::busy`); that is a view
of Resident, not a state of its own.

**A Parked model's host cost** is the `SeamModel`'s parsed metadata and
tokenizer, the `OaiRenderer`'s own `Gguf` and second tokenizer, and the
`WeightWatch`; the mapping itself is address space, its pages the OS's. The f32
token table is built only by a host-gather path (the CPU runner, MTP heads,
DiffusionGemma's soft embedding) and is `n_vocab * n_embd * 4` bytes once built
(`SeamModel::token_embd`'s doc: about 3.1 GiB for Qwen3-14B), so a parked model
that ever ran one must drop it. Per model: not measured. What bounds the number
of parked models is that host memory, charged in the host ledger, and the file
handles each `Gguf` keeps open (one per shard, **guess**).

What each transition costs, from the code:

| Transition        | Re-done                                                                                                                                                                      | Measured?                                                                                                                                                                                                                                                                                            |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Declared → Parked | `SeamModel::load_with` (parse, map, tokenizer), `OaiRenderer::open_with_tokenizer` (opens the GGUF and the tokenizer again)                                                  | not measured: time and host bytes per model                                                                                                                                                                                                                                                          |
| Parked → Resident | device creation, tuning resolve, the fit, weight upload (from the mapping, or through the host tier), pipeline creation (from the pipeline cache file when warm), slot forks | the startup log prints `warmup: pipelines compiled in` and `slots: … ready in`; plan 17 recorded ERNIE-21B's first GPU run at 6.4 s against 1.1 s after; `pcache.rs` records about 5 s of cold pipeline creation for a DiffusionGemma forward. Split by part, warm and cold page cache: not measured |
| Resident → Parked | queue idle, drop: frees weights, every slot's KV (the prefix cache is lost), the host tier, pipelines; saves the pipeline cache                                              | not measured: time to drop, and the room returned against the room taken                                                                                                                                                                                                                             |
| Demote / promote  | slots evicted or filled; no rebuild (slices D1 to D3)                                                                                                                        | decode at an evicting budget is plan 08's figures above; the time to give back and refill a share: not measured                                                                                                                                                                                      |

## One memory budget across models

**Device memory.** Within a model, placement stays the owner and the rule stays:
the estimate equals the allocation (plan 13 S2, B13). Across models the
manager's ledger is the single owner, through the shares. Placement sizes
against the share instead of the whole room (`alloc_room()` capped by it). A
model with no declared share gets the whole device when resident and is never
resident beside another model on that device; nothing changes for a single-model
server. `kernels.vulkan.vram_limit` is not a share: it subtracts every other use
from the limit, other models included.

**Host memory.** Plan 15 decided one `HostLedger` (slice R1) with a fixed order
of classes: live KV spill (mandatory), the weights' host tier (elastic), cached
KV blocks (elastic, lowest). This plan inherits the ledger and the order, and
extends them in two places:

- **Between models inside a class**, the elastic shares go to resident models by
  the manager's recency, the most recently used first, the same direction as
  plan 15's disk budget. A demoted model's evicted blocks land in its host tier
  under this rule.
- **Parked models' own host memory** (above) is a mandatory entry while they are
  declared; a manager-owned host copy of a parked model's weights, if ever built
  (decision 4), would be a new elastic class (decision 10).

Until R1 lands, each model's host tier is sized from the OS probe at its own
load, so the first model loaded takes half of what is available and keeps it:
the arenas cannot shrink until plan 15 R3. Serving several streamed models
therefore waits on R1 (pricing) and R3 (giving back).

**Where two plans would fight over the same bytes:**

| Bytes                                | Plans                                     | Owner                                                                                                                                                       |
| ------------------------------------ | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Free VRAM on a device                | plan 14 Q (promote experts into room)     | the manager: plan 14's policy promotes inside its model's share, or it fills the room the manager freed to load another model                               |
| "Held by others" in plan 14 W's wait | plan 14 W                                 | another model in this process is not an "other": the manager frees it. Whether one process's two devices count in each other's budget usage is **to check** |
| A spilled KV cache's host part       | plan 15 R0/R1, the spill                  | the ledger, mandatory class, per model                                                                                                                      |
| The weights' host tier               | plans 05, 06, 15 R1/R3                    | the ledger, elastic class, split by the manager's recency                                                                                                   |
| KV blocks on disk                    | plan 15 (`kv.disk_size`, LRU model first) | plan 15's store, unchanged: the manager does not own disk                                                                                                   |

Inherited and not re-argued: plan 14's decline of whole re-placement inside a
running model (the manager parks a model whole or changes its share; it never
re-places it in place), plan 14's "never demote" for experts, which holds until
the D slices give demotion a share to answer to, plan 15's same-setup-only key,
and plan 05 item A4 (`--parallel N` under a tight host budget, B35) applying per
model.

## Eviction

Plan 06 found that bookkeeping cost matters only with thousands of items, that
recency alone loses to a sweep or a burst of rare items, and that the policy is
chosen by replaying recorded traces against Belady's bound before building. A
server has a handful of models, so the bookkeeping is free and a miss costs a
reload of seconds; the method carries over, the data structure does not.

**Never demoted or parked:** a model with a lease (a slot permit held), a pinned
model, and a model with requests waiting for it while another has none.

**What "cold" means**, the candidates:

1. **Recency** (LRU across models): simple, and what plan 15 uses for disk.
   Loses when one heavy model's burst displaces a steady one.
2. **Recency weighted by reload cost** (GreedyDual style: value = priority plus
   the measured load time, aged): keeps the model that is expensive to bring
   back. Needs T0's load times.
3. **Frequency with aging**: protects the steady model; needs a window.

Recommended: 1 with pins and priority first, recording a trace (model, arrival,
duration, load, park and share events) from the first serve slice, and 2 or 3
only if the simulator shows a gap worth closing, as plan 06 did.

**Thrashing** when the requested models do not fit together:

- **Minimum residency:** a loaded model stays for `serve.min_resident_secs`
  before it may be parked.
- **Batching per model:** when a model becomes resident, its queued requests run
  before the next switch, up to `serve.switch_hold_secs` while other models
  wait.
- **Demote instead of park** (D slices): a smaller share keeps both models
  serving, slower, with no reload stall.
- **Refuse early:** with a queue timeout set, a request that cannot be served in
  time is answered `503` with `Retry-After` instead of waiting.

## The manager's own concurrency

- **One owner of the table:** a `std::sync::Mutex` over the state of every model
  and the per-device ledgers, held only to read or change a state, never across
  a load, a park or a GPU wait. Waiters for a model wait on a per-model
  notification (a `tokio::sync::Notify` or `watch`, **guess**), not on that
  lock.
- **Loads run on the blocking pool** (`spawn_blocking`, like preparation and
  generation), **one at a time per process** at first: `cmd_multi` loads its
  models in turn because two concurrent uploads spike host memory, and an
  integrated GPU's uploads come out of the same RAM. One at a time per discrete
  device is a later relaxation once measured.
- **A request for B while A loads on the same device** waits in B's
  wait-for-model queue. When A's load ends, the scheduler picks the next model
  by the oldest waiting request, after A's batching hold.
- **The lease and park invariant:** a model is parked or its share lowered only
  while the manager holds **every** permit of its `slots` semaphore
  (`acquire_many_owned(capacity)`), so no generation runs on it and none can
  start. Tokio's semaphore is FIFO, so requests already queued for a permit run
  first and later ones queue behind the manager; after the park they find the
  engine gone and move to the wait-for-model stage. A slot taken just before the
  manager's decision is a lease, and the park waits for it. The same invariant
  guards a share change in the D slices. `acquire_slot`, `Slot` and
  `QueuedGuard` are reused as they are; `StepGate` stays per engine.

## Failure paths

| Failure                                                    | The manager ends in                                                                                                                                                                                 | A client sees                                                                                                                                                                                                                               |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B's load fails after A was parked to make room             | B Failed (back-off); A stays Parked and is loaded again only on its own next request. A had no waiters, or it would not have been parked                                                            | B's waiters: `503` with the load error. A's next request: a load                                                                                                                                                                            |
| An allocation refused while a model is resident and leased | the model stays Resident; the request fails as today (the guard's error)                                                                                                                            | that request: the error it gets today; others unaffected                                                                                                                                                                                    |
| Device lost or a GPU wait timed out while resident         | the backend is poisoned and leaks its memory (`gpu_wait.rs`): the model is Failed, its charge stays in the ledger for the process's life, and the device is marked suspect, no new loads there      | that request and every later one for the model: the poisoned error, as `docs/config.md` says today; other models on that device: whatever their own next wait finds (a real loss likely reaches every logical device on the GPU, **guess**) |
| A park that hangs (the drop's queue-idle wait times out)   | the same as a poisoned backend: memory leaked, device suspect                                                                                                                                       | the waiters for the next model: `503` once their queue timeout ends, or they wait without one                                                                                                                                               |
| A park waiting on a lease that does not end                | the park waits; the next model waits with it                                                                                                                                                        | a request deadline bounds it where one is set; without one the wait is unbounded, as the slot queue is today                                                                                                                                |
| A load racing shutdown                                     | no new load starts once the latch is set; a load in progress stops at the warmup generation's abort poll, or ends, and drops before exit. Whether the upload itself polls the latch is **to check** | waiters: `503` "server shutting down", as `acquire_slot` answers on a closed semaphore                                                                                                                                                      |

## The request path

1. **Route:** an exact id among the declared models; an unknown or empty id
   keeps today's fallback to the default model and never causes a load.
2. **Prepare** without the device: the renderer, the vocabulary, the logprobs
   and constraint support are what the handler and `prepare` read before a slot,
   so they move to the Parked part (`ParallelGenerator` already holds its
   renderer apart from the engine). The context limit is known only after a load
   re-clamps it; a parked model is checked against its last loaded `max_ctx`, or
   its configured `ctx` before the first load, and checked again after the load.
3. **Wait for the model:** a new stage before `acquire_slot`, cancellable the
   same way (a dropped future leaves the queue), counted in `queued` and in a
   per-model gauge. The first waiter triggers the load; a load whose waiters all
   left completes anyway (decision 11).
4. **Slot and generation:** unchanged. The lease is the slot permit.

**What the client sees:** latency. A streamed request holds its `200` until the
slot, as today (decision 6). **Fairness between models:** the next model to load
is the one whose oldest request has waited longest, subject to the batching
hold. Within a model, the slot queue stays FIFO.

**Devices:** each device has its own ledger. A model is declared on a device or
placed on the one with the most room when it loads; moving it is a park and a
load elsewhere. Multi-GPU splits of one model are out of scope.
**`--parallel`:** per model; the slots are forked eagerly at load
(`init_slots`), so a model's charge includes its N caches and
`refuse_unhostable_forks` refuses at load, not mid-request.

**Reporting:**

- `/v1/models`: every declared model, whatever its state, with the standard
  fields only.
- `/health`: unchanged, `200` while the process serves.
- `/status`: per model `state`, `pinned`, `share_bytes`, `charged_bytes`,
  `streamed_slots` (held of total, for a demoted model), `last_used`, `loads`,
  `last_load_seconds`, `last_error`; per device `suspect`.
- `/metrics`: `infr_model_resident{model}`, `infr_model_loads_total{model}`,
  `infr_model_parks_total{model,reason}`,
  `infr_model_load_seconds_total{model}`, `infr_model_share_bytes{model}`,
  `infr_requests_waiting_model{model}`.

## KV state of a parked model

- **Before plan 15:** lost. Every slot's KV and the prefix cache go with the
  backend; the next request prefills. A demoted model keeps its KV: the D slices
  take weight slots, not KV.
- **Kept in RAM:** declined. It is plan 15's RAM block cache (S10) by another
  route, with a second format.
- **After plan 15 S5 and S6:** at park, the slots' full blocks not yet stored
  are captured through S5's path; a request after the wake finds them through
  S6's chain lookup like any other. No path of its own. Models plan 15 refuses
  (unrecoverable recurrent state until S9, the DeepSeek V4 compressed caches)
  lose their KV, logged once per park.

## Configuration

Following `docs/config.md`: positive field names, the size grammar, no `INFR_*`
twin for a per-model key (as `kernels.metal` has none). The keyed table has a
precedent in `[kernels.vulkan.tune]` (`infr_core::config::tune`); whether
`manifest.rs` and `--set` take a table keyed by model id is **to check** in M2.

| Key                           | Default                                     | Meaning                                                             |
| ----------------------------- | ------------------------------------------- | ------------------------------------------------------------------- |
| `serve.models.<id>.path`      | required                                    | the model, in `infr serve`'s grammar (a path or `org/repo[:quant]`) |
| `serve.models.<id>.dev`       | automatic                                   | the device, in `device.dev`'s grammar                               |
| `serve.models.<id>.ctx`       | `device.ctx`                                | context per slot                                                    |
| `serve.models.<id>.parallel`  | `--parallel`                                | slots                                                               |
| `serve.models.<id>.vram`      | unset: whole device, never co-resident      | the model's device share, size grammar                              |
| `serve.models.<id>.min_vram`  | unset: park rather than demote              | the smallest share it may be demoted to (D slices)                  |
| `serve.models.<id>.pin`       | `false`; `true` for a server with one model | never demoted or parked                                             |
| `serve.models.<id>.priority`  | `0`                                         | higher is demoted later                                             |
| `serve.models.<id>.idle_secs` | `0` (never parked for idleness)             | park after this long unused                                         |
| `serve.max_resident`          | unset (the byte budget decides)             | at most this many resident models per device                        |
| `serve.min_resident_secs`     | set by M4's thrash measurement              | minimum residency                                                   |
| `serve.switch_hold_secs`      | set by M4's thrash measurement              | the batching hold                                                   |
| `serve.queue_timeout_secs`    | `0` (no bound, as today)                    | longest wait for a model or a slot                                  |

On the command line `infr serve A B C` declares models by their file stems, as
`infr multi` names them today. A single model behaves exactly as now: pinned,
loaded at startup, the same banner. The model set is read at startup only; a
config reload or `SIGHUP` changes nothing (decision 13). Each change of keys
updates `docs/config.md`, `infr.example.toml` and the README's server section in
the same slice.

## Verification

Under the accuracy rule, every slice shows its output unchanged:

- **Bit identity:** on one device and configuration, a run on a reused resident
  equals a run on a fresh backend, bit for bit (plan 17's `same_bits`), for
  every gate configuration and for runs in both orders (`verify` mode). For
  serve, a model parked and loaded again answers a fixed prompt with the same
  logits row as one never parked, and a model demoted and promoted again the
  same as one never demoted. This holds only when the placement is the same:
  backlog B-PLACEMENT-TOKEN-FLIP records that the prefill chunk alone changes
  greedy tokens. So the first load's placement (context, chunk, KV formats) is
  recorded and kept through demotion and reload, a reload that cannot reproduce
  it is refused (decision 8), and the test asserts the placement first.
- **Stale state** (the project has shipped bugs of this class: the recurrent
  state inherited after a reset or warmup, rings recycled before a rewind,
  forked rings sized from the wrong chunk, Q8_0 seeds without scales): the first
  request after a wake equals a fresh process's on a recycled slot; a qwen35
  model and a sliding-window model reused across runs, across a park and across
  a demotion; `ForcedLogits::kernels` lists only the run's own kernels; a replay
  recorded before a demotion is not replayed after it (the placement epoch
  moves).
- **Budget:** after every transition (load, park, demote, promote, a model
  loaded beside another), the ledger's charge for each model equals what the
  device reports this process holding, read the way
  `kv_buffers_commit_no_more_than_their_price` reads heap usage; a park returns
  the room its load took and a demotion the slots' bytes (the leak
  `backend_drop_frees_device_after_moe_pager` once caught, repeated over many
  cycles). B13's unexplained remainder past the estimate is reported, not
  hidden.
- **Thrash:** the policy as a pure function on a scripted clock and request
  trace (CI); end to end, two small models that do not fit together under
  `INFR_VRAM_LIMIT`, alternating requests, with the load count held under the
  bound minimum residency sets.
- **Failure paths:** each row of the failure table driven once: a load made to
  fail after a park (a scripted fault), a refused allocation under
  `INFR_VRAM_LIMIT`, a poisoned backend (the existing GPU-wait tests' hook), a
  shutdown during a load.
- **Each test is seen red first:** a planted carry-over (a kernel list not
  reset, pins kept) fails the identity and stale tests.
- **Not testable without real load:** policy quality, latency and fairness under
  real traffic; contention between two resident models on one device (separate
  `StepGate`s, so nothing measured shares the GPU fairly); page cache warmth on
  reload; Metal, which reports no device room (B8 item 5) and has no hardware
  here.

## Slices

Each ships alone, through coder then reviewer, with a test that can fail. T0
runs alongside T1 or after it and blocks nothing: T1's acceptance is the bit
identity above plus a wall-time figure for the gate on one model, labelled with
the host's load.

| #   | Slice                                                                                                                                                                                                                                                                                               | Depends on                                                                                                | Unlocks                                                           |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| T0  | Measure only: per forced run, backend creation, weight upload, pipeline creation (warm and cold pipeline cache) and the run itself, gate models, both GPUs; with plan 17 slice 2's quiet window                                                                                                     | —                                                                                                         | the size of T2's win; reload costs for M4 and D3                  |
| T1  | `Residency` with one resident per device: a kept backend and `BoundWeights` keyed by model instance, device and the whole edited configuration; `forced_run` takes `prebound`; per-run kernel list; `INFR_TOL_RESIDENT` off, on, verify; the gate's loop turned around. CPU first (CI), then Vulkan | —                                                                                                         | reuse across prompts and greedy runs of one configuration         |
| T2  | Split the backend's construction-time configuration from per-run knobs (audit every `be_.cfg()` read); key on the first only                                                                                                                                                                        | T1                                                                                                        | three backends per device per model in the gate                   |
| T3  | Leases and the per-device ledger: loading one model evicts another model's unleased resident when the room is short; tests holding two models                                                                                                                                                       | T1                                                                                                        | the serve slices                                                  |
| M1  | Serve: split `ModelEntry` into the always-present part (renderer, prepare, vocabulary) and a resident engine behind the manager; every model pinned; no behaviour change                                                                                                                            | T3                                                                                                        | M2                                                                |
| M2  | Lazy load, idle park, the wait-for-model stage, the park invariant on the slot semaphore, `serve.models.<id>` keys, `infr serve A B`, per-model sampling defaults, `/status` and `/metrics` fields                                                                                                  | M1                                                                                                        | swapping models on one device                                     |
| M3  | Shares: placement sizes against a model's share; the charge checked against the measured take after every transition; recorded placement and refused reloads; the failure paths                                                                                                                     | M2                                                                                                        | co-residency; plan 14 Q and D read the share                      |
| M4  | Policy under contention: pins, priority, LRU, minimum residency, the batching hold, the queue timeout; the trace recorder and the thrash test                                                                                                                                                       | M3                                                                                                        | the policy measurement                                            |
| M5  | Streamed models priced in the host ledger, split by recency; give back on park                                                                                                                                                                                                                      | M3, plan 15 R1 and R3                                                                                     | several streamed models in one server                             |
| M6  | KV through the disk store at park                                                                                                                                                                                                                                                                   | M2, plan 15 S5 and S6                                                                                     | parked conversations survive                                      |
| D1  | Load a demotable dense model streamed with a slot for every block, its chunk pinned to the resident one; bit identity against resident and decode within plan 08's figures                                                                                                                          | M3, plan 08 (done; step 5's fusion for the last percent)                                                  | the top rung of the ladder                                        |
| D2  | VRAM pools that give slots back and take more: a pool's arena in several BDA buffers, a buffer freed once its slots are evicted, a buffer added to grow; the placement epoch moves on both                                                                                                          | D1                                                                                                        | shares that shrink and grow; also plan 14's bigger-arena fallback |
| D3  | The manager lowers and raises shares: demote by D2, the host tier takes the blocks, promotion by the pager's fill; the policy chooses demote or park from T0's reload costs and M4's trace                                                                                                          | D2, M4, M5; plan 05 item A1 for models beyond llama-shaped projections                                    | partial demotion for dense models                                 |
| D4  | The same for MoE expert pools                                                                                                                                                                                                                                                                       | D3, plan 06 step 5 (paged replay), plan 14 A, P and D, a bit-identity check of paged against resident MoE | partial demotion for MoE models                                   |

Before plans 14 and 15 land: T0 to T3, M1 to M4, D1 and D2. Plan 14's slices Q
and D are the ones that must read the share once M3 exists, whichever lands
second wiring it.

## Open decisions for the owner

1. **Config shape.** Tables keyed by id (`[serve.models.qwen3]`, reachable by
   `--set`) or an array of tables (more TOML-idiomatic, no `--set` path).
   Recommend keyed tables, for `--set` and env-free overrides.
2. **`infr multi`.** Keep it as an alias of `infr serve` with every model pinned
   to its device, or remove it. Recommend the alias: scripts keep working.
3. **Unknown model ids.** Keep the fallback to the default model, or answer
   `404` now that a model can be parked. Recommend keeping the fallback (OpenAI
   clients send names the server never heard of) and never loading for it.
4. **A host copy of a parked model's weights.** None beyond what the OS keeps
   (free, uncounted, gone under pressure, and `paging.release_uploaded` already
   drops the mapping's pages after a resident upload), or an arena the manager
   owns (counted, survives pressure, costs host RAM the KV tiers want). With the
   D slices the host tier under a demoted model is that copy while the model is
   resident. Recommend none for parked models, until T0 measures a cold reload
   against a warm one.
5. **Queue deadline.** A new `serve.queue_timeout_secs`, or
   `serve.request_timeout_secs` counted from arrival. The second changes what an
   existing key means. Recommend the new key, off by default.
6. **Streaming while a model loads.** Hold the `200` (today's semantics: an
   error stays a status code) or send it with keep-alive comments (no client
   read timeout, but a failed load becomes an error frame). Recommend holding,
   revisited with T0's load times.
7. **Shares of models on one device.** No share means the whole device and no
   co-residency (recommended, nothing changes for one model); or an equal split
   by default (co-resident out of the box, smaller windows for everyone).
8. **Placement on reload and demotion.** Reproduce the first load's placement or
   refuse (bit identity, may park more), or re-fit to the room (always loads,
   output may change at a chunk boundary). Recommend reproducing it.
9. **Sampling defaults per model.** Each model's own (recommended; it changes
   `infr multi`, which applies the first model's to all), or the first's.
10. **Order of a manager-owned host copy** in plan 15's ledger, only if decision
    4 builds one: below the weights' host tier (it saves one reload per wake, a
    per-burst value) and above cached KV blocks, recommended, or below them.
11. **A load whose waiters all left.** Finish it (recommended: whether the
    upload can stop part-way is to check, and the next request likely follows),
    or abort.
12. **The gate's loop order.** Turn it around (recommended; one resident per
    device is enough) or keep a resident per configuration (more VRAM, which an
    integrated GPU running a 21B model does not have).
13. **Config reload and `SIGHUP`.** Nothing by design: the model set is read at
    startup and changing it is a restart (recommended); or re-read the model
    table on `SIGHUP`, adding and parking models. Either way `SIGHUP` should
    drain like `SIGTERM` instead of ending the process without the GPU drain,
    which is a fix of its own, not part of this plan.
14. **Demotable models load streamed.** Under the D slices a model that may be
    demoted loads streamed from the start (within 2% of resident decode on plan
    08's figures, bit-identical at the same chunk), or loads resident and is
    parked whole when its share must shrink. Recommend streamed only for models
    with a `min_vram` set, so nothing changes for the rest.
15. **`INFR_TOL_RESIDENT`'s default in CI.** `verify` (recommended: CI holds the
    identity on every push, at the cost of running the small CPU tests twice) or
    `off` (CI as today).

## Out of scope

- Continuous batching, within or across models (`parallel.rs` explains why there
  is none).
- Multi-GPU splits of one model (pipeline, tensor and expert parallel) under the
  manager: they bypass the tiers too (plan 05 item A5); a declared model that
  asks for one is refused by name.
- One backend, pager or allocator shared by several models.
- Moving a resident model between devices without a park.
- Adding models at run time through the API; the set is declared.
- Rate limits per client (B5, declined), and sharing residents between
  processes.
- Metal beyond a model count: it reports no device room to budget against.
