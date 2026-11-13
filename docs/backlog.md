# Backlog

Known work that is deliberately not done, with enough context to pick it up
cold.

Everything here has been triaged: it is either blocked on something, scoped out
of the slice that surfaced it, or waiting on hardware. Items that were merely
_unfinished_ do not belong here — they get done. An item leaves this file when
it lands or when it is withdrawn (with the reason recorded, so it is not
rediscovered).

Provenance tags point at the finding that opened the item:

- `CR-*` — the whole-tree correctness reviews. Their report lived at
  `docs/code-review.md` and was **deleted on 2026-08-03, folded into this
  file**: the eight findings of the 2026-08-03 pass were re-verified against the
  code and became B19–B26, all eight of which have since been fixed and deleted
  from here (`git log -S'### B19' -- docs/backlog.md` finds them). That pass's
  cleared / hardening / coverage lists survive as B27–B29. The tags on B1–B5
  come from the earlier 2026-08-01 pass, whose text the file had already stopped
  carrying (`6ab8b1c` overwrote it with the later review). A `CR-*` tag is
  therefore a historical marker for where an item came from, not a link to
  anything.

---

## Open

### B1 — `kernels.vulkan.dn_chunk_scan` reads opposite to its name

**Tag:** CR-N10 · **Needs:** a decision (re-read 2026-11-12)

The env key is right: setting `INFR_DN_CHUNK_SCAN` forces the chunked scan. The
config field it fills is inverted — `dn_chunk_scan == true` selects the
sequential path — and its doc comment in `config/mod.rs` is wrong the other way
again (it says the key turns off the chunked path). The field cannot simply be
renamed: `kernels.vulkan.dn_chunk_scan` is a user-visible TOML key and `--set`
path, named in `docs/config.md`. Options: rename it and keep the old spelling as
an alias (the project's env policy is to drop old spellings, so this would be a
breaking rename); flip the polarity under the same key (silently changes the
meaning of an existing config file); or fix only the comment and live with the
name.

### B3 — the VNNI kernels' bounds assertions run on the dev box, not in CI

**Tag:** CR-U3 coverage gap · **Blocked on:** CI hardware

Corrected 2026-11-12: the dev machine (Ryzen 9 9950X3D) has `avx512_vnni`, so
`cargo test -p infr-cpu` there does run the `*_vnni` kernels and their
`debug_assert!`s; the earlier form of this entry said no machine did. What is
unknown is whether the GitHub runners have it — the SIMD tier tests cover only
the tiers the runner's CPU reports, so in CI the VNNI tier may be compiled and
skipped. To close it: log the detected tier in the test output and read one CI
log, or add a job that runs `cargo test -p infr-cpu` under an emulator (Intel
SDE).

### B4 — tensor-parallel has no automated coverage

**Tag:** CR-C10 · **Blocked on:** CI hardware

`TensorParallelBackend` needs ≥2 physical GPUs, so nothing in CI exercises it.
The C10 byte-accounting bug (KV checkpoints allocated at `full/W²` while the
copy moved `full/W`) went unnoticed because of this, and was only caught when an
unrelated bounds guard turned it into an error.

The fix landed with unit tests on the pure parts — `TpBuffer::len_bytes()` per
kind, `shard_bytes`' divisibility guard, and the
`shard_bytes(len_bytes()) == per-rank` round trip — but the real path is still
unrun.

**To do:** a two-GPU smoke run with tensor parallelism on — there is no
`--tensor-parallel` flag; it is `multi.tensor_parallel` (`INFR_TENSOR_PARALLEL`,
or `--set multi.tensor_parallel=…`). The KV-checkpoint path was exercised by
MTP, which is parked (`mtp_enabled()` returns `false`), so while MTP stays
parked a smoke run proves the sharded forward pass but NOT the checkpoint bytes;
that half needs another caller of the checkpoint path, or MTP revived. Worth
doing once by hand even without CI.

### B5 — `infr serve` has no rate limiting

**Tag:** CR-S7 (partial) · **Decision:** out of scope for this binary

The per-request wall-clock deadline landed (`serve.request_timeout_secs`), which
bounds how long one request can hold a `--parallel` slot. Rate limiting —
bounding how many requests one client may make — deliberately did not.

**Reasoning:** every operator exposing this to a network already has a reverse
proxy, and that is where connection limits, per-IP quotas and burst control
belong. Reimplementing them in a single-binary inference server means owning a
worse version of solved infrastructure.

**Revisit if:** someone wants to expose `infr serve` directly to untrusted
traffic with no proxy in front. That is a different product decision, not a
missing feature.

### B6 — a four-token bench column resolves nothing under ~10%

**Tag:** diagnosed + fixed 2026-08-02 · **Blocked on:** nothing; what is left is
a measurement floor nobody intends to chase

The prefill columns' 6.8–34.5% run-to-run spread was **not** tier
nondeterminism, which is what this entry originally claimed: `INFR_PROF_OPS=1`
over six back-to-back runs produced a byte-identical (op name, dispatch count)
signature while throughput moved 1029.5 → 1120.4 t/s, and nothing feeds a tier
decision live VRAM at bench scale. The cause was that `bench_vulkan`'s untimed
warmup was a hardcoded `(8, 2)` turn, so the first TIMED rep of any other shape
paid that shape's one-time costs — pipeline variants, first-touch scratch pools
— inside the measured window. Fixed by warming at the measured shape.
`pp4@d4096` peak-to-peak went 8.4% / 20.2% to 4.3% / 7.9% over four sweeps each
way, and the worst named row (Qwen3.6-35B-A3B UD-IQ3_S) 20.2% → **1.4%**.

**The residual, accepted.** Qwen3-0.6B and gemma-3-1b keep ~8% peak-to-peak on
`pp4@d4096`. That column times four tokens — ~3.7 ms wall per rep of which ~2.8
ms is device time — so roughly a quarter is host-side record/submit/fence and
its jitter is what is left. Options if it ever matters: raise `-r` for that
column only, report the median instead of the mean, or accept that a four-token
measurement resolves nothing under ~10%. `infr bench` prints the per-rep min-max
and spread on every line, so this is visible in the output rather than inferred.

**Methodology that outlived the diagnosis:** alternate A/B legs at least twice
and report every repeat, never a single value or an average that hides the
spread. A single-leg A/B on this box has been wrong by 9% — that outlier (one
leg in thirteen, gemma-3-12b `tg128 @ d8192`, 75.9 against a 83.0–83.3 cluster,
and it was the FIRST leg run against that model) was measured on the pre-fix
binary and never re-run, so it is a reason for the rule and not evidence about
the current tree.

### B7 — decode at depth: two designs declined, one landed, one still open

**Tag:** measured 2026-08-02 · **Blocked on:** nothing; slices 3a and 3b have
LANDED and are deleted from here — what remains open is (b), width by workgroup
count

The largest remaining gap to llama.cpp is decode at depth, not prefill.
Qwen3-30B-A3B Q4_K_M on a 7900 XTX against `llama-bench c629da5`, `tg128`
infr/llama: **138.7 / 165.1** @d8192, **106.5 / 140.3** @d16384, **66.9 /
112.1** @d32768 — 0.60× at depth while `pp512` holds parity. (Those are the
PRE-slice-3a baselines; `INFR_NO_ATTN_DECODE=1` reproduces them.)
`attn_partial_bda` is **59% of decode GPU time** at d32768 and scales exactly
linearly with KV.

**Two designs are DECLINED — do not re-try either as written.** Both reached
agreement with the reference (GQA bit-identically, the k-tile to 9.6e-7), so
they were correct, just slower:

- **GQA head-grouping**, one workgroup per (KV-head, chunk) covering all
  `g = nh/nkv` query heads: cuts K/V traffic 8× (537 → 67 MB per layer-token)
  and measures **329 µs against 177 µs — 1.87× SLOWER**. Grouping serializes 8
  cross-lane reductions into one wave that previously ran on 8 CUs; the re-read
  it eliminates was nearly free out of Infinity Cache. Not starvation either:
  re-run at matched parallelism it measured 359 µs, with `attn_combine` going
  24.5 → 146 µs on the 8× larger `pacc`. Fewer keys per workgroup also loses
  (chunk 256 on the ungrouped kernel: 314 µs). So neither traffic nor occupancy
  is the lever. Reverted; the tree is unchanged.
- **The LDS-staged K-tile** — the design `recorder.rs`'s
  `attention_kv_split_impl` comment names ("per-thread full dots, no cross-lane
  reductions, which is how llama.cpp wins that cell"). Best of four configs is
  **2.7× slower** at d32768 (382 µs against the shipped 184). Not an
  implementation miss: the ISA shows **0 cross-lane ops** against
  `attn_partial_bda`'s 54, so the reduction really is gone — the LDS transpose
  that buys its removal costs more than the reduction saved, because the K tile
  has ZERO data reuse (every byte written once, read once) and time is monotone
  in LDS budget (34 KB → 687 µs, 17 KB → 494, 9 KB → 382). Survives unwired as
  `tests/attn_ktile_probe.rs` + `shaders/attn_ktile.comp` because it is the
  measurement rig for the next attempt.

**What the oracle actually does**, read rather than guessed (`ggml-vulkan.cpp`,
`get_fa_tuning_params_scalar`, our shape on RDNA3): `path = FA_SCALAR` — coopmat
is **deliberately avoided at decode** ("scalar is faster than coopmat when
N==1"), which kills the matrix-core / `gqa_ratio` idea this entry once proposed;
`shmem_staging = 0` on AMD, independently corroborating the k-tile negative;
`block_rows = 1`, `block_cols = 64`, `workgroup_size = 128`; and
**`d_split = 8`** — the width of the group that cooperates on one key's dot, the
one parameter none of the experiments varied.

**The `d_split` sweep** (`shaders/attn_partial_dsplit.comp`,
`tests/attn_dsplit_probe.rs`) produced two SEPARABLE findings, and the `w=32`
control is what separates them:

1. **Specialization is free 6–9% at every depth with no algorithmic change** —
   `w=32` reproduces the shipped mapping exactly yet beats it, because a
   decode-only copy allocates **96 VGPRs against 120** with zero spills either
   way, so more waves fit per SIMD. This is what slices 3a/3b shipped.
2. **Narrow width wins only where workgroup parallelism is short**, monotone in
   workgroup COUNT rather than depth: 512 wg → best 2.01×, 1024 wg → 1.46×, 2048
   wg → every width loses. At depth the kernel is already at ~3.0 TB/s
   (Infinity-Cache rate) and splitting a wave's contiguous 512-byte K read into
   `32/w` segments costs more than the shallower reduction saves. So llama.cpp's
   `d_split = 8` is right for ITS configuration, not universally, and **the
   original B7 target (d32768) remains a negative**.

**Still open — (b) width by workgroup count.** Choose `w` from `nh * n_chunks`
against the device CU count rather than from depth. Needs shapes the probe never
covered — it only tested `nh=32 nkv=4 hd=128` — and its per-width numbers were
measured against the OLD reference, so the ratios overstate the remaining
headroom by the 4–8% slice 3a already took. Treat it as a mid-depth lever: it
does NOT close the headline gap, where the model is still ~0.64× at d32768.
Before leaning on that probe again, spend ten minutes on **the unexplained
9.6e-7 drift at w=32** — `attn_decode` itself is bit-identical, and neither
making every reduction `subgroupClusteredAdd(., 32u)` nor constant-folding
`sqrt(float(pc.hd))` reproduces it. Also note `dsplit_bench`'s "SHIPPED
reference" leg now measures `attn_decode`, so re-run it under
`INFR_NO_ATTN_DECODE=1` to compare against the old baseline.

**Still falling back to `attn_partial`** — each would need its own member of the
`attn_decode` family: planar-Q8 and mainline-inline quant KV (**this is what
gemma runs by DEFAULT at full context, so it is the biggest remaining coverage
hole**), the DiffusionGemma canvas mask, `rows > 1` (small-m spec-verify and
prefill), the bound-SSBO (non-BDA) dispatch, `chunk > 512` (only reachable above
~524k keys under `INFR_KV_OVERFLOW`), head dims other than 128/256/512, and a
RING cache on a `window == 0` layer (unreachable today — only SWA layers are
allocated as rings — and the static gate's row bound rejects it rather than
assuming).

**Two traps the shipped kernels carry, worth reading before touching them.**

- The hd 256/512 QK tail loop must keep `attn_partial`'s redundant
  `if (r < hd4)` guard AND read `hd4` from the push constant at runtime. Folded
  to a build-time literal the guard vanishes, ACO fuses the terms into one FMA
  chain, and the chunk score moves 1 ULP — which `attn_combine`'s `exp(m_c - M)`
  weight turns into 136/2048 non-identical outputs. The shader says so; do not
  "clean it up".
- `INFR_NO_ATTN_HD=1` is a **DIAGNOSTIC, not a bitwise A/B**. It deletes the
  specialized arm for the general runtime-`hd4` loop — a different summation
  order — and on Qwen3.5-9B the output stays coherent and correct while
  splitting from the shipped text at generated byte 302 of 396, where shipped
  and `INFR_NO_ATTN_DECODE` are byte-equal. Verified BY KERNEL NAME that both
  the static and replay gates honour it. Cost: −4.2% at d4096, −11.7% at d32768.

**Why the gains are model-shaped:** 4–7% on Qwen3-30B-A3B but ~0.5–1% on gemma
and a wash on Qwen3.5-9B, because attention SHARE differs (2.9% of decode GPU
time on qwen35 at d4096, 8.7% on gemma-3-12b), not because the kernel is worse —
at d32768 `attn_decode_hd256` beats `attn_partial_bda` 152.9 vs 167.0 µs on
qwen35. Gemma's 40 SWA layers are window-capped (18.4 µs flat from d4096 to
d32768), so most of its layers cannot grow with depth. The one place the
specialization LOSES is gemma above d16384 — see B15.

### B8 — what is still ESTIMATED after the measured-fit slice

**Tag:** re-measured 2026-08-04 · **Blocked on:** nothing; each item below is a
bounded piece of work, and none of them is currently causing a failure

The original entry claimed the reserve's runaway term was the non-flash score
tile, priced as "2 live pools at the final ctx" while the real pools "ACCUMULATE
across the ~150+ chunks of a deep prefill". **Both halves were wrong**, and the
measurement that settled it is now permanent machinery:
`Backend::activation_peak` (a high-water mark of live `Activations` bytes) and
the runner's `activation reserve too low` warning. Each chunk's `execute` drops
its pool before the next builds, and every layer of a chunk shares one `kv_len`,
so ONE tile is live — the model was over-reserving, by 3.5x at a 128-row chunk.

What actually broke the fit, on the reported gemma-4-31B UD-Q5_K_XL case
(reproduced at margin 1.0, `bench -p 0 -n 4 -d 19000`, which died on a 4 MiB
alloc **2 MiB** past the guard budget):

| term            | planner        | actual                         | delta   |
| --------------- | -------------- | ------------------------------ | ------- |
| weights         | 21 871 930 488 | 22 353 012 736 (217 arena blk) | +481 MB |
| KV              | 2 559 590 400  | ~2 504 MB                      | −56 MB  |
| activation peak | 911 343 616    | 262 MB                         | −649 MB |
| driver-side     | 0              | 187 MB after load → 368 at pk  | +368 MB |

So the fix was not a better activation model: it was to stop estimating the
things the device can be ASKED about. `reclamp_ctx_to_live_room` re-decides the
window between the weight upload and the KV allocation, against
`Backend::device_alloc_room`. What is still predicted at that point:

- **The activation reserve**, now re-fit to measured peaks with named MoE and
  DeltaNet terms and a 1.5x pad (`ACT_RESERVE_PAD`, sized by the worst arch
  measured — Qwen3.5-4B-MTP at 1.42x).
- **`POST_KV_DEVICE_RESERVE`** (256 MiB) — the pipelines/descriptors the driver
  builds while recording the first forwards, measured at 181 MiB on the largest
  model here.

**What is left, in the order it matters.**

1. **The per-arch activation algebra is whack-a-mole, and the structural fix is
   known.** Every term in `dense_act_reserve_at` re-derives, in the seam, what
   `runner.rs`'s `build` closure already declares exactly (`g.internal(...)`,
   each `batch * <width>`), plus what the Vulkan adapter's `pooled(...)` sites
   allocate. Fitting it per arch found three misses in one afternoon (MoE expert
   scratch, qwen35's DeltaNet mixer, qwen35's double-width `qg`/`gate_a` pair) —
   each caught by the new warning, none by a test. The fix is to SUM the graph's
   `Internal` tensors instead of modelling them, which needs the graph buildable
   for a shape before the KV cache exists; today `build` is defined after the
   cold-init block and closes over the KV handles. The pooled attention/MoE
   terms would still need the adapter's tier predicate (the old option 1).
2. **The chunk rung is still chosen pre-load, against the light weight
   estimate.** The re-clamp can LOWER it (`repin_ubatch_lower`) when that buys
   context, but placement's resident-vs-stream decision itself is unchanged and
   ~2% optimistic on weights. It self-corrects into a smaller window rather than
   a failure, so this is a context cost, not a correctness one. Whenever the
   reserve moves, re-check `docs/perf/results.md`'s placement box, which names
   the rung a model settles on: gemma-3-12b went 256 → 1024 rows at ctx 131072
   in this slice (780 t/s, was 760), while gemma-4-31B's documented 256 rows at
   d4096 and 1024 rows at `pp512` were re-measured and still hold.
3. **The context is ~1.5k tokens more conservative than the device strictly
   requires.** gemma-4-31B advertises 14 592 (the interim-margin build said 15
   872, and 19 021 was measured to miss by 2 MiB). The gap is
   `POST_KV_DEVICE_RESERVE` + the allocator's own 256 MiB `GUARD_HEADROOM` + the
   reserve's pad + `KV_SLOP_ROWS` (B13). Reclaiming it means measuring the
   driver-side growth on more than one GPU first — 181 MiB is one sample, on one
   RADV version.
4. **Every measurement here is from one 7900 XTX on RADV.** The arena-tail
   percentage, the driver-side growth and the activation peaks are all
   Mesa/RADV/discrete numbers. An Intel or NVIDIA driver could report its budget
   with different granularity, and an iGPU shares the heap with the host.
5. **Metal and CPU are untouched by design**: `device_alloc_room` and
   `activation_peak` default to `None`, so both keep exactly their previous
   behaviour and neither gets the measured clamp. Metal has a working-set query
   that could implement the first.
6. **`infr serve --parallel N` was exercised at N=2** (two 40960-token slots on
   Qwen3-1.7B, a request served) but not at a size where the re-clamp fires, so
   the interaction between a shrunk window and `vulkan_slot_ctx`'s divide-by-N
   is reasoned, not measured.

### B10a — the serve arrival line reports prompt CHARS, not prompt tokens

**Tag:** raised 2026-08-02 · **Blocked on:** a `ChatGenerator` trait change

B10's request/throughput logging landed, with one part of its spec unmet: the
`request start` line carries `prompt_chars`, not prompt tokens. The tokenizer
lives behind `ChatGenerator`, so at arrival the server genuinely cannot know the
token count — it only learns it from `ChatOutcome` when the generation ends,
which is where the real `prompt_tokens` is logged (`request done`).

Fixing it means a new `ChatGenerator` method
(`count_prompt_tokens(&[ChatMessage])` or similar) implemented by `infr-cli`'s
`SeamGenerator` / `ParallelGenerator`, which then have to render the chat
template a second time — the template render, not just the tokenization, is what
determines the count. Out of scope for the logging slice: it is a trait change
plus duplicated render work on every request, for a number the completion line
already reports accurately a moment later.

### B11 — an explicit `INFR_CACHE=<pct>` still resolves against raw free VRAM

**Tag:** narrowed 2026-08-02 · **Blocked on:** a decision about what the
percentage should MEAN

The placement budgets that infr derives for itself — `vulkan_moe_binder`'s dense
residency predicate, its streaming budget and its MoE expert budget — now take
the allocator's ceiling (`VramInfo::alloc_room`, free minus the VRAM guard's 256
MiB headroom), the same function the context-fit math uses, guarded by
`budgets_agree_with_the_allocator_ceiling` and
`fit_math_and_placement_pick_the_same_rung`.

Both `INFR_CACHE` tiers still resolve a percentage spec against `vram.available`
(`spec.resolve(..)` in the MoE and dense override arms). Left alone
deliberately: that value is the CALLER's budget, the grammar is documented as "a
percentage of the device's AVAILABLE VRAM", and the override exists to force a
placement the auto tiers would not choose — so failing loudly at the alloc guard
is defensible where silently handing back less than asked is not.
`INFR_CACHE=100%` is the case that would trip it. Decide whether the percentage
means "of free VRAM" (today) or "of what can actually be allocated" before
changing it; either way it is one `spec.resolve` argument per arm.

### B12 — what the explicit `--ctx N` fit ladder still does not cover

**Tag:** raised 2026-08-02 · **Blocked on:** nothing; the ladder shipped (q8
2026-10-27, refuse 2026-10-28), these are its gaps

**DECIDED 2026-08-12 (user):** an explicit `--ctx N` that does not fit must try
q8_0 KV first, then refuse — f16, then q8_0, then refuse.

**Shipped:** `SeamModel::auto_q8_for_explicit_ctx` pins q8_0 when f16 misses and
q8_0 holds `want * n_slots`. `SeamModel::refuse_explicit_ctx` then refuses,
before any weight loads, when `seam::kv_floor_alloc_bytes` (each side at the
cheapest of the requested format, f16 and q8_0, the ring wherever possible, the
shortest prefill chunk) exceeds `alloc_room()`. The runner repeats the check
with the exact buffers against the measured room between the weight upload and
the KV allocation. Both count only buffers of at least
`infr_vulkan::VRAM_GUARD_MIN_ALLOC`, and both skip under `INFR_KV_OVERFLOW` and
`INFR_NO_VRAM_GUARD`. Observed on the iGPU: Qwen3-0.6B at `INFR_CTX=1000000` is
refused in under a second with the sizes.

**Gaps:**

- **The refusal ignores the weights**, so it covers only "the KV cache alone
  cannot fit". A window whose KV fits but KV + weights does not stops at dense
  placement's streaming-floor error, which now names the context and the KV size
  (observed on the iGPU: Qwen3-0.6B, f16 KV, `INFR_CTX=500000`). Folding the
  weights into the early refusal is provable only for a model that cannot stream
  them.
- **Not observed in a real run:** the q8 pin, and the runner's post-load
  refusal. Both need an explicit window close to the whole device: on this box
  that means filling the iGPU's shared memory or using the dGPU, both
  off-limits. Only their decision functions are tested.
- The DEFAULT-path refuse rung stays as narrow as it was: only "no usable
  context at all" may refuse, or every ordinary long-context model breaks on a
  24 GiB card.

### B13 — the `+64` rows in every KV footprint estimate is slop, not padding

**Tag:** verified 2026-08-02 · **Blocked on:** nothing; left alone deliberately

`seam::kv_bytes_estimate_fmt` adds `KV_SLOP_ROWS = 64` rows per layer before
sizing each side's buffer, and the comment it inherited described this as
mirroring a pad `SeamKv` allegedly applies. It does not: both allocation sites
(`generate_dense_backend`'s KV loop and `SeamKv::fork`) allocate exactly
`kv_rows(..) * n_kv * head_dim` elements. The 64 rows are a deliberate
conservative margin and nothing more — the doc now says so.

Left in because every placement estimate shares this helper and removing it
would loosen all of them at once. That argument has weakened since: the window a
session actually gets is now re-decided against the device's measured free
memory (B8), so these rows are no longer a cushion against an over-optimistic
plan — they are context the fit hands back for nothing. Removing them is worth a
measurement now, not just a revisit.

### B14 — verification gaps from the 2026-08-02 decode-attention and KV-fit slices

**Tag:** raised 2026-08-02 · **Blocked on:** nothing; each is a measurement
someone has to run

Recorded as gaps rather than left implicit. Everything below shipped without the
check named, and in each case the check is cheap — the reason it is missing is
time or hardware, not difficulty.

- **Metal and CPU are unexercised for the KV-fit change.** Only the Vulkan path
  was measured. The Apple `#[cfg]`-gated code does not compile locally at all,
  so CI is the only thing that judges it — and CI does: the
  `cargo test (macOS / Metal)` and `cargo check (infr-metal, Apple target)` jobs
  are green. That settles compilation, not behaviour; nobody has run the fit
  math on an Apple device or on the CPU backend. The 2026-08-04 measured
  re-clamp does not change that either way: both backends return `None` from
  `Backend::device_alloc_room`, so they keep the estimate-only path unchanged
  (B8).
- **`infr serve --parallel N` is exercised at N=2 only, and not at a size where
  the window moves.** Two 40960-token slots on Qwen3-1.7B served a request
  (2026-08-04), which covers the fork path but not `vulkan_slot_ctx`'s
  divide-by-N against a window the post-load re-clamp shrank — the engine now
  reads its advertised window back from slot 0 after the warmup, and no run has
  put those two together.
- **The refuse rung's `Err` has never been printed by a real run.** No model on
  this box drives `max(fit_f16, fit_q8)` under `MIN_SESSION_CTX`, so the message
  text — the thing a stuck user actually reads — is untested against a human.
- **The iGPU chunk-ladder filtering is reasoned, not measured.** Filtering
  `ubatch_candidates` to heights below the current one also stops a placement
  sweep raising an integrated GPU's chunk above its watchdog-safe default. That
  argument was never run on an iGPU, and the watchdog is exactly the thing that
  punishes being wrong (see `docs/igpu.md`). `repin_ubatch_lower` (B8's measured
  re-clamp) refuses to RAISE a height for the same reason, and is equally
  unmeasured there.
- **The tightened placement budgets were only exercised on their RESIDENT
  branch** (the 2026-08-02 B11 slice). gemma-4-31B, gemma-3-12b and
  Qwen3-30B-A3B all stay resident on this box, so `dense_stream_budget_at` and
  `moe_expert_budget`'s `None` arm (dense weights + KV past the ceiling — the
  hard error) were verified only by unit test, never by a run that actually
  streams or pages. A model that does not fit this card would cover both;
  `INFR_CACHE=<size>` forces the streaming path but with the caller's budget,
  not the derived one.

### B15 — `attn_decode` crosses over and LOSES on gemma above d16384

**Tag:** measured 2026-08-02 · **Blocked on:** a decision — gate the
specialization, or accept 1.5% at deep context on gemma

`attn_decode` (B7 slices 3a/3b) is bit-identical to `attn_partial` and leaner on
paper (96 VGPRs / 3072 B LDS against 120 / 5120), and it wins at every depth
measured before this one. On **gemma-3-12b at d32768 it loses 1.5%** (tg128 70.8
/ 70.8 / 70.6 / 70.6 ON against 71.9 / 71.9 / 71.7 / 71.7 OFF).

**Where the time goes.** At d32768 the specialized family costs 128 × 335.7 µs +
640 × 18.1 µs = **54.6 ms** across the same 768 pass-1 dispatches that
`attn_partial_bda` serves in **51.0 ms** — 7.1% slower on attention, which lands
as +1.4% on the device total (229.8 vs 226.6 ms) and shows up as the 1.5%
throughput loss. `attn_combine` is untouched (9.3 ms both legs), so the whole
difference is pass 1.

**The global (non-SWA) layers are the regressor, and the argument needs no
modelling.** Because the SWA kernel is flat in depth, the d8192 → d32768 DELTA
belongs entirely to the 128 global dispatches: ON `attn_decode_hd256` goes 79.2
→ 335.7 µs (**+256.5**), OFF `attn_partial_bda`'s total goes 22.3 → 51.0 ms
(**+224.4 µs** per global dispatch). The specialized kernel scales **14% worse
with depth**. Taking the SWA cost as equal in both legs (18.4 / 18.4 / 18.5 /
18.1 µs, measured on the ON leg) the implied per-dispatch global cost is:

| depth | `attn_decode_hd256` | implied `attn_partial_bda` global | ratio |
| ----- | ------------------- | --------------------------------- | ----- |
| 4096  | 40.1                | ~43                               | 1.08× |
| 8192  | 79.2                | ~82                               | 1.04× |
| 16384 | 153.4               | ~163                              | 1.06× |
| 32768 | 335.7               | ~308                              | 0.92× |

**Crossover is between d16384 and d32768**, and end-to-end tg128 agrees: 1.008×
at d16384, 0.985× at d32768.

**Why gemma and not qwen35, at the same head dim and the same depth.** Both are
`nh=16 hd=256`, so both grids move the same 537 MB per layer-token — but gemma
has `nkv=8` where qwen35 has `nkv=4`, so gemma's UNIQUE KV footprint is 268 MB
against 134 MB and it gets half the re-read reuse out of the 96 MB Infinity
Cache. Achieved rate at d32768: qwen35 537 MB / 152.9 µs = **3.5 TB/s**
(cache-served, and the specialization wins 1.09× there); gemma 537 MB / 335.7 µs
= **1.6 TB/s**, far closer to DRAM. HYPOTHESIS, consistent with all the data but
NOT directly instrumented (no counter was read): the leaner kernel's extra
occupancy — 16 waves/SIMD against 12 — helps while the kernel is
occupancy/latency-limited and hurts once it is streaming-bound, because more
concurrent streams cost locality. Reading RGP/`RADV` memory counters on the two
builds at d32768 would settle it.

**The options.** (a) Leave it: 1.5% on one model at one depth, against 4–7% on
Qwen3-30B-A3B and 1.09× on the qwen35 kernel. (b) Gate `attn_decode` on
something that predicts the regime — the crossover tracks `kv_len * nkv * hd`
against cache size, not depth alone, so a depth threshold would be wrong on
qwen35, which is still gaining at d32768. (c) Chase the occupancy hypothesis and
fix the kernel rather than gate it. Nothing here is urgent; it is recorded so
the next profile does not rediscover it as a mystery.

**Related, not the same thing:** `docs/perf/results.md` used to say "the
decode-at-depth cells are now stale by 4–8%". The 2026-08-03 re-sweep settled
that: the `tg64@d4096` column moved **+0.021× averaged over all 35 rows**, with
the gain concentrated on Qwen and MoE rows (Qwen3-30B-A3B 0.91× → 0.96×,
Qwen3-14B Q4_K_M 0.94× → 1.00×) and nothing measurable on gemma (Gemma-3-12B
1.13× → 1.14×). The doc now says that instead of extrapolating. The d32768
regression this entry is about is beyond the table's depth and unaffected.

### B17 — the submit splitter arms for real on Qwen3.6-27B at d4096

**Tag:** measured 2026-08-03 (the `results.md` re-sweep) · **Blocked on:** a
decision — soften the trigger, or accept that one table row is measured with a
different submit structure

B6 recorded `VulkanBackend::observe_forward`'s submit splitter as a latent
hazard that had never fired. It fires. Benching **Qwen3.6-27B Q4_K_M** at
`-d 4096`, the untimed depth prime is a single 1633-dispatch forward that takes
**~1.01 s** — just past the 1 s `SUBMIT_DANGER_NS` threshold — so the cap
latches and every later forward in the process, **including the timed ones**,
splits every ~400 dispatches. Reproduced in **3 of 3** processes (caps 392 / 401
/ 403, read from the `submit_cap` field `infr bench --json` emits and the
matching WARN line). It also arms on **Qwen3-30B-A3B** on four of the nine
deep-context legs: cap 269 on `pp512`/`tg128` at `-d 32768`, and 342 / 222 on
`pg8192,512` at d16384 / d32768. It does not arm on the d16384 `pp512`/`tg128`
legs or anywhere at d8192, so the trigger tracks the single longest forward in
the process, not the depth. Also observed 2026-08-04 on **gemma-3-12b Q4_K_M at
`-p 131056`** (cap 133) — a whole-window prefill is long enough to latch it, so
any deep-prefill measurement on that model is split too.

**Why it matters.** The dispatched kernels stay byte-identical, so this is
invisible to `INFR_PROF_OPS` and to any golden; only the submit structure
changes. Two consequences:

- `results.md`'s Qwen3.6-27B `tg64@d4096` and `pp4@d4096` cells, and every
  deep-context row past d8192, are measured under a split submit while every
  other cell in the table is not. The doc flags this; the cells are not wrong,
  they are just not the same experiment.
- The trigger is a **wall-clock sample of one forward, at a threshold this model
  sits within 2% of**, so a slightly warmer or busier box would flip the row.
  Reproducing 3/3 here is the luck of that margin, not stability.

**What was NOT done:** no A/B of the split against `INFR_SUBMIT_DISPATCHES=0` on
this model, so the size of the effect on those two cells is **unknown** — it may
be nothing. That measurement is the obvious next step and is cheap.

**Options.** (a) Leave it. (b) Exclude the untimed depth-prime forward from
`observe_forward`'s sampling, which is arguably what the guard always meant —
the watchdog risk is about user-visible forwards, and the prime is one
deliberate bulk operation. (c) Raise the threshold. (b) looks right but changes
iGPU watchdog behaviour, so it is not a one-liner — see `docs/igpu.md`.

### B18 — three rows the 2026-08-03 re-sweep left unexplained

**Tag:** measured 2026-08-03 · **Blocked on:** nothing; each needs a profile

Recorded so the next `results.md` reader does not have to re-derive them. All
three are reproducible, not variance: `pp512` and both decode columns now repeat
to under 3.4% peak-to-peak over four full runs (see the doc's variance box).

- **Gemma-3-1B Q4_K_M `pp512` = 0.95×, and it is real.** Four independent runs
  gave 0.95 / 0.95 / 0.95 / 0.94. The doc used to dismiss the small-model
  `pp512` cluster as "prefill variance, not a real deficit"; that explanation
  died with the warm-rep fix. It is **dtype-specific, not architectural** — the
  same model's Q2_K reads 1.05× and its Q8_0 1.30× on the same column. Nobody
  has profiled the Q4_K prefill path on a 1B gemma.
- **Qwen3-14B Q8_0 `pp4@d4096` = 0.95×** — the table's only `pp4@d4096` loss.
  `results.md` footnote ⁷ used to imply this cell was 1.18×; that figure was the
  legacy-int8-GEMV slice's own before/after A/B, never a table cell, and the doc
  now says so. Why the small-m prefill path loses on Q8_0 specifically, when the
  same slice's +28.8% held on every other legacy format, is unexplained.
- **Llama-3.2-1B `tg64@d4096` = 0.88× (Q8_0) / 0.94× (Q4_K_M)** — the worst cell
  in the table, on the smallest model in it, while both of that model's other
  columns win. Reproducible (0.92–0.94× on Q4_K_M across four runs). An isolated
  small-model decode-at-depth deficit with no named cause.

**Coverage gaps in that sweep, stated plainly.** The Ternary-Bonsai Q2_0 table
(CPU oracle), the Llama-4-Scout pager figures and DiffusionGemma's `dg-e2e` were
**not** re-measured — of the DG pair only `dg-step` was. Metal was not measured
at all. The oracle was a cached `llama-cpp-vulkan-b9833` release build run
through an `LD_LIBRARY_PATH` shim, because the distro `llama-cpp` (b10182) links
ggml 0.17 against an installed `ggml-vulkan` 0.15.3 and **no system llama.cpp
binary runs on this box** (`undefined symbol: ggml_dsv4_hc_post`). The packaging
fix is not ours; the shim lived in session scratch and will not survive, so the
next sweep needs its own working oracle before it starts.

### B21a — the DG abort poll is wired but has never fired in a real run

**Tag:** CR-2026-08-03 M3 residual · **Blocked on:** nothing; it is a test that
needs a live DG serve request

`DiffusionGemmaChat` now threads `RequestCtx` into `diffusion_generate`, which
polls the abort latch at each BLOCK boundary, and the per-request seed is
resolved by `resolve_seed` (unit-tested). What is not tested is the poll
actually stopping anything: `diffusion_generate`'s loop needs a loaded
DiffusionGemma model, so no unit test can drive it, and the wiring was verified
by reading rather than by cancelling a real request.

Two things to check by hand against `infr serve` hosting a DG model: a client
disconnect mid-generation stops at the next block instead of running every
remaining block, and `serve.request_timeout_secs` does the same. Both latch the
same flag, so one test covers the mechanism.

Also still true, and by construction: a block is the finest granularity
available. `denoise_block` runs a whole canvas to completion, so a cancelled DG
turn stops at a block boundary, not immediately.

### B30 — the GGUF weight mmap trusts the file, and cannot enforce it

**Tag:** PR#90 · **Blocked on:** nothing outstanding; detection SHIPPED, the
preventing half is deliberately not attempted

`Gguf::open` maps the model file and hands out `&[u8]` slices into it for the
mapping's whole life. Nothing stops another process writing or truncating that
file: a write mutates memory Rust believes is frozen, and a truncation turns a
resident page into `SIGBUS` on next touch. That invariant is stated on `open`
and remains **UNENFORCED** — `infr_gguf::watch::WeightWatch` notices the breach,
it does not prevent it. Statting the HELD DESCRIPTOR rather than the path is the
load-bearing choice and `a_rename_into_place_is_not_a_change` pins it:
`infr pull` renames into place, which leaves a live mapping on the old inode
perfectly intact, so a path-stat would cry wolf on the one file-replacing
operation infr itself performs.

**Known boundaries of the detection, none worth closing on current evidence:**

- A same-length in-place write whose mtime is then restored is invisible. The
  only alternative is hashing gigabytes per check.
- `serve` checks per REQUEST, so a change landing mid-request is caught by the
  next one, after that response has already streamed. Post-checking would not
  un-send it.
- `WeightWatch::open` is a second `open` beside `Gguf::open`, so a rename
  landing exactly between them leaves the watch on the new inode and the mapping
  on the old — a missed detection, never a false one, two syscalls wide.

**Considered and rejected: copying the file into an anonymous mapping.** PR #90
did exactly that and it works, but it is not affordable. Measured on a 16 GiB
Qwen3.6-27B, warm page cache, two reps each: warm load 1.87 s → 10.5 s (5.6x,
re-opening almost exactly the gap `model-load-time` closed), and 14 GiB of
evictable page cache became 20.2 GiB of anonymous RSS. Anonymous pages are
swap-only, so a model larger than RAM goes from slow to unrunnable — the Llama-4
Scout blob on this host is 47.5 GiB against 60 GiB of RAM. Reverted in
`5ba6b3f`; the same PR's tensor byte-count overflow check was kept.

**Considered and rejected: an advisory `flock`.** `infr` cannot corrupt its own
mapping in the first place — `pull.rs` downloads to a temp and renames, nothing
anywhere opens a blob for writing — so a lock has no in-house writer to conflict
with. And it does not bind the writer that actually matters:
`cp new.gguf live.gguf` opens the destination `O_TRUNC` and takes no lock, nor
does any editor, nor llama.cpp. Reusable machinery exists (`FileLock` in
`pull.rs`) if this is ever revisited, and it should be taken SHARED — exclusive
would stop two `infr` processes sharing one model. `FileLock` is portable as of
PR #91: the `libc::flock` call is now a `cfg(unix)` arm with a `LockFileEx`
counterpart, so a revisit no longer has to solve Windows first, and CI tests it
on windows-2025 as well as ubuntu-26.04 and macos-15.

### B31a — what the weekly miri job does NOT cover

**Tag:** PR#90 review residual · **Blocked on:** nothing; recorded so the
coverage claim stays accurate

Miri runs weekly against `SpinPool` and `infr_core::hostpager` via
`.github/workflows/cron.yml` (which carries the reasoning for its flags — two
upstream workarounds are load-bearing and must not be "simplified" away). That
covers `collect`'s raw base pointer, its `set_len` over uninitialized slots,
`CollectGuard`'s `drop_in_place` during unwinding, the `Vec::from_raw_parts`
rebuild, and the host pager arena's per-slot raw-pointer slices across threads.
What it does not, and will not:

- **`kernels.rs`** — 168 of `infr-cpu`'s 191 `unsafe` uses are x86 SIMD, which
  miri cannot execute. That unsafe stays unchecked by anything but review.
- **Every FFI crate**, by construction: `infr-vulkan` dlopens `libvulkan`,
  `infr-metal` talks to a real GPU, `infr-gguf` maps a file, `infr-hub` takes an
  `flock`.
- **`infr-core` and `infr-chat` in FULL** — both probed crate-wide, neither
  finished inside the window it was given (10 and 50 minutes; `infr-chat` had
  completed 16 of 58 tests when stopped). Those bounds are "did not finish by",
  not measured durations. Little is lost: `infr-chat` contains no `unsafe`, and
  `infr-core`'s uses outside `hostpager` are three, one being a `libc::kill`
  miri cannot execute. The `hostpager::` filter added later runs in seconds, so
  the crate-wide cost was never the obstacle for the part that matters.

### B34 — no fuzz targets, and there is an obvious first one

**Tag:** cron slice 2026-08-04 · **Blocked on:** nothing; scoped out of the cron
slice that surfaced it

The sibling repos' `cron.yml` runs `cargo-fuzz`; this one does not, because the
tree has no fuzz targets. hjkl's job guards its absence with
`if [ -d .../fuzz ]`, which is a job that reports "skipping" forever — a check
that cannot fail — so it was left out rather than copied.

`infr_chat::tools::parse_tool_calls` is the target that has already earned it.
It parses model output, which on `infr serve` is steerable by whoever sent the
request, and it has produced one unbounded-allocation hang in that position
(`{a:[}]}`, fixed in `0aa0661`). `delimiter_soup_always_terminates` now covers
every 6-byte body of container punctuation exhaustively, which is a floor rather
than a ceiling — libFuzzer over the same entry point would reach the string,
escape and `\u` paths that the punctuation alphabet cannot.

Adding it means a `fuzz/` crate, a nightly job, and a decision about how long
per target per week.

### B35 — tiered weight paging: phase 4 unbuilt, phase 5 one lever in

**Tag:** design slice 2026-08-04 · **Blocked on:** phase 4 needs Apple hardware
this host does not have

`docs/disk-streaming-plan.md` carries the design and the per-phase verification.
**Phases 0-3 have LANDED** (baseline measured, core `blockio`/`hostpager`/pins,
CPU backend on the DRAM tier, and the Vulkan third tier under BOTH dense
streaming and the paged MoE cache — numbers in `docs/perf/results.md`), and the
tier now **beats mmap on both backends**: CPU 2.06x at a 1.5 GB cap, Vulkan
2.17x on decode at an 8 GB cap with a 7 GB arena (1.41x with a 3 GB one). What
is left:

- **Phase 4, Metal / UMA collapse.** Unbuilt and **unverifiable here**: no Apple
  hardware, and `infr-metal` does not compile on this box. Writing it blind
  would produce code whose only evidence is that it type-checks in CI. Its own
  precondition is the `qui_cache` gate below. The options and their trade-off
  are written out in the plan's §7 as an open question for the user — do not
  re-derive them.
- **Prefetch is deprioritized, and that reversal is the useful part.** It is
  still unbuilt on every backend (`HostPager::pin` reads synchronously, and on
  Vulkan under the dense/MoE session mutex). It was recorded here as "the
  leading suspect" for the GPU tier being slow. It was not: the run is I/O-bound
  by orders of magnitude — roughly 12.5 GB read per token against tens of
  milliseconds of GPU compute — so hiding a read behind compute has nearly
  nothing to hide it behind. The read was too SLOW, not too LATE, and the
  concurrent reader is what fixed it. Prefetch only becomes interesting once the
  arena is big enough that the tier stops being I/O-bound. Do not build it
  before then.
- **The reader's speedup is Linux/NVMe only.** `FileBlockIo` splits a block
  across `IO_FANOUT` concurrent positioned reads, measured 1.2-1.5 → 2.2 GB/s on
  this box's Samsung 980. Correctness is platform-independent (each read carries
  its own offset), but the SPEEDUP is not: on Windows `seek_read` issues
  `ReadFile` with an `OVERLAPPED` offset and a handle not opened
  `FILE_FLAG_OVERLAPPED` has concurrent operations serialized by the kernel, so
  the fanout may buy nothing until the file is opened for overlapped I/O.
  Untested on Windows and macOS. A rotational disk is also untested and is the
  one case where the concurrency could plausibly HURT (seek interleaving);
  nothing in the code adapts to device type.
- **The rest of phase 5**, still gated on measurements not taken: io_uring only
  if the reader proves queue-depth bound beyond what `IO_FANOUT` concurrent
  `pread`s reach (on this drive they already hit the device ceiling, so there
  may be nothing left), frequency-warmed DRAM for MoE-on-GPU, exclusive
  VRAM/DRAM placement for MoE, and multi-GPU/MTP coverage (TP/EP/pipeline
  binders and MTP's second weight set bypass the tier entirely).

Constraints the remaining phases must handle, recorded so they are not
rediscovered:

- `infr-metal`'s `qui_cache` factored arm copies the transformed weight out and
  retains it unboundedly, keyed by `MTLBuffer::id()`. Correct, but it is a
  second full copy of every touched weight in host RAM, so a paged Metal model
  must gate or budget it (plan §3.4).
- **Host paging is single-process-shaped.** `HostPager`'s exhaustion error names
  `paging.dram`, but nothing sizes the arena against `infr serve --parallel N`:
  the floor is N concurrent working sets and only 1 is priced. A tight budget
  under `--parallel` will surface as that error rather than a deadlock, which is
  the safe failure, but the sizing is still not done. The Vulkan tier is less
  exposed than the CPU one — `DensePagerSession::stage` drops its pin before
  returning, so its floor is one slot per pool regardless of N — while the CPU
  interpreter holds a whole op's pins across the op.
- **The per-pass file-change check is not wired — needs a decision.**
  `FileBlockIo::verify_unchanged` exists and is tested; no caller runs it yet
  (read), so a model rewritten mid-generation is read as whatever the new bytes
  are (the same exposure as B30, now reachable through explicit reads rather
  than the mapping). Options: check once per forward pass, once per request, or
  at each `HostPager::pin`; and on a mismatch either fail the request or reload.
  Each costs one `fstat` per check; which granularity is worth it is the
  decision.

### B36 — paging optimizations found by review, measured but not built

**Tag:** paging review 2026-08-04 · **Blocked on:** nothing; each is scoped out
of the slice that found it

A read of the whole paging path (DISK `blockio`, DRAM `hostpager`, VRAM
`infr-vulkan::pager`, the CPU `paged` pools and the seam placement) against the
counters `INFR_PAGER_STATS` reports. Three items LANDED from it — the concurrent
reader, the admission doorkeeper, and auto-sizing — and only the last leaves a
residue worth tracking; what follows is that plus what was deliberately left.

- **Auto-sizing LANDED on every platform; only Linux has been measured.**
  `infr_core::hostmem::available_bytes` sizes the arena from
  `infr_plat::mem::available`, which now has a Linux arm (`MemAvailable` floored
  by the tightest cgroup limit), a macOS arm (`host_statistics64`) and a Windows
  arm (`GlobalMemoryStatusEx`, clamped by a Job Object limit). The tier turns
  itself on whenever a model does not fit. Measured on Linux only: it picks 7.44
  GB under an 8 GB cap and reaches 0.38 t/s against the swept best of 0.39. The
  macOS and Windows arms are only checked for plausible figures (the
  `*_live_probe_is_plausible` tests, which pass on CI's macOS and Windows legs);
  neither has been benchmarked against a cap.
- **The unified-memory path's sizing is unverified.** An iGPU/APU streams
  `DISK → GPU-accessible RAM` with no host cache (`HostPager::stream_only`,
  selected by `DeviceCaps::unified_memory`). The MECHANISM is covered on a
  discrete GPU by `INFR_DRAM_BYPASS` — the dense leg in `dense_tier_parity`
  content-checks it and the MoE leg in
  `gpu_seam_paged_moe_host_tier_matches_resident` is token-identical. The
  SELECTION is seen on the Ryzen 9950X3D's Raphael iGPU (RADV): `unified_memory`
  is set there and the banner prints `UNIFIED MEMORY` (run). What is NOT covered
  is the sizing: nothing sizes `paging.cache` (which on such parts comes out of
  shared RAM) against HOST memory the way `paging.dram` is, and no streamed
  model has been run on the iGPU. A feature (a host-memory term in the arena
  sizing) plus a run of a streamed model on the iGPU. Metal has no pager until
  phase 4, so it inherits none of this yet.
- **Layer-major prefill LANDED; what is left is where it does not reach.** The
  chunk loop now runs inside the layer loop for a streamed model
  (`seam::layer_major_prefill`, the `spans`/`chunks` walk in
  `generate_dense_backend`), so a prompt sweeps the weight set once instead of
  once per chunk. Re-measured on the B36 shape — Qwen3-14B Q8_0 / RX 7900 XTX,
  `MemoryMax=8G`, `paging.cache=2g`, `paging.dram=6g`, P=4096, three rounds with
  the arm order permuted and a cold page cache before every run — at the
  1024-row default chunk: 25.27 → 6.31 GB read and 341.9 → 779.9 pp t/s, the
  read volume now exactly a single-chunk prefill's. The residue:
  - **The remaining gap to the single-chunk arm is HOST cost, not I/O.** Same
    6.31 GB, but 779.9 t/s against 1049.6: layer-major builds and compiles a
    graph per (layer, chunk) — 40 x 4 here — where the one-chunk arm builds one.
    `build` re-declares every weight handle for the whole model on each call and
    `alloc_scratch` re-allocates the whole Internal set per execute, so both
    scale with the dispatch count. A per-(layer, batch-shape) plan cache, or a
    span band wider than one layer, is the obvious next lever; neither was
    tried.
  - **The activation reserve is priced at the FULL context, not the prompt.**
    `layer_major_act_bytes` reserves `ctx * n_embd` f32 out of the streaming
    budget (`dense_stream_budget_at`) because those buffers are allocated
    mid-prefill and the arenas are sized before any prompt arrives. A session
    that never fills its window holds that back for nothing — at ctx 32k /
    n_embd 5120 it is 671 MB of arena. Sizing it against a per-request prompt
    length, or reserving in bands, needs the budget to be re-decidable after
    load, which it is not today.
  - **Only the dense Vulkan path.** MoE expert paging, MTP, the qwen35/DeltaNet
    bespoke path, Metal and the CPU backend all keep chunk-major: the gate is
    `Backend::dense_paged`, and the E2B arch is refused by
    `layer_major_prefill`'s `spannable` arm (its `per_layer_inp` is
    prologue-built and a later span cannot see it). That gate is load-bearing,
    not belt-and-braces: E2B DOES take the batched-prefill path, so while the
    only refusal was the `assert!` in `build`, a streamed E2B panicked on an
    ordinary `infr bench <e2b> --set paging.cache=200m`. Covered by
    `gpu_seam_streamed_e2b_stays_chunk_major`. A paged MoE prefill has the same
    `ceil(P/ubatch)` structure and was never swept; whether the expert cache's
    locality makes it the same win is unknown.
  - **Not measured: more than one prompt length, and any model but this one.**
    The sweep is P=4096 on one dense model, one drive, one GPU. The claim that
    the ratio grows with prompt length is arithmetic (`ceil(P/ubatch)` sweeps
    become one), not an observation.
  - **`Capabilities::graph_input_inplace` is answered, not tested, for Metal.**
    It is set true there on the strength of `infr_core::exec::writes_back` being
    shared with the CPU interpreter — read, not run. Nothing on Metal takes the
    layer-major path today (its backend hosts no dense pager), so the flag is
    inert there until something forces it on.

- **`Pager`'s eviction is O(n_slots) per miss.** A touch is O(1) now
  (`infr_core::pager::Pager` keeps a `recency` stamp per block in a `HashMap`),
  but choosing the victim (`Pager::coldest`) scans every resident block. That is
  a scan per MISS, not per touch, and at today's model sizes it is not worth
  changing. At DeepSeek-V4-Flash scale (256 experts × 43 layers per role) a
  miss-heavy decode would pay it often; an ordered index on the stamp is the
  fix. Not measured.
- **The dense session mutex spans the disk read.** `stage_dense_linear`
  (`infr-vulkan/src/adapter.rs`) holds `be_.dense_pager().lock()` across
  `DensePagerSession::stage`, which reaches `HostPager::fill` and blocks on I/O.
  Irrelevant to `bench` (one sequence), but under `infr serve --parallel N`
  every sequence serializes on every other sequence's disk reads. Related to the
  `--parallel` sizing gap recorded in B35 but a distinct problem: that one is
  about arena capacity, this one is about lock hold time.

### B37 — a macOS break in `infr-llama` still costs a macOS runner (2026-09-02)

**Tag:** ci coverage · **Blocked on:** nothing; what is left is latency, not
coverage, so it may never be worth doing

This was "the cheap guard does not guard the crate that keeps breaking", and the
coverage half of it is closed. `clippy` and `test` now run natively on
`macos-15` over the whole workspace, so a `WBytes`-style break in `infr-llama`'s
`#[cfg(target_os = "macos")]` arms — the failure that left `test-macos` red from
`e657a66d` to `588653b` — is caught on every commit instead of being discovered
by whoever next looked at the Metal job.

What remains is only how FAST it is caught. `metal-check` cross-lints from the
Linux runner in about a minute and covers the nine crates that cross-build;
`infr-llama`, `infr-hub`, `infr-chat`, `infr-engine`, `infr-server` and
`infr-cli` cannot join it because `ring` and `esaxx-rs`/`onig_sys` compile C for
the target and need an Apple SDK (locally both fail with
`unrecognized command-line option '-arch'`). So a break in those six is found by
a macOS runner in about three minutes rather than a Linux one in about one.

That is a small enough difference that paying for an SDK (osxcross or similar)
on the cross-lint job is probably not worth it. Recorded so the question is not
re-opened from scratch.

### B48 — the segment-abort error paths are wired but never fired (2026-08-10)

**Tag:** CR-2026-08-09 deepseek · **Blocked on:** a way to make these calls fail

Every `?` exit inside `execute_static`'s op loop that can drop a live recorder
now routes through `abort_segment`: both `lower_op` sites (since B45), plus
2026-10-27's `resolve` for the paged-MoE gate and the streamed `Linear` weight,
`execute_paged_moe`, and `stage_dense_linear`. `Recorder::finish_nowait` was
checked and needs nothing: it frees its own command buffer and pools on each of
its error exits.

Only the `lower_op` route has been observed. The B45 guard test
(`mla_oversized_dims_are_refused`) is clean under
`VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation` (re-run on the iGPU
2026-10-27), and printed `VUID-vkDestroyDevice-device-05137 … 2 leaked objects`
before its fix. The four newer sites fail only on a broken binding or a paging
I/O error, and no test produces either, so their cleanup is unverified.

### B49 — the full-softmax MoE weight has no regression test, and cannot have one here (2026-08-10)

**Tag:** CR-2026-08-09 deepseek · **Blocked on:** hardware — a Vulkan
implementation that preserves f32 subnormals

The defect this entry opened with is FIXED: `moe_topk.comp`'s
softmax-without-renormalization branch summed `exp(logit - mx)` over every
expert while `mx` was the max over the SELECTED ones, so a router bias or group
mask could leave the largest raw logit unselected, overflow the denominator to
`+inf` and zero every weight. It now computes its own max over all experts,
matching the CPU oracle's constant.

What stays open is that **nothing guards it**, and nothing can on this box. A
selected expert's weight is `1 / D` where `D` is the denominator the wrong shift
computes; the bug needs `D` past f32's max (3.4e38) to overflow, so the correct
answer the fixed kernel must produce is always below 2.9e-39 — inside the
subnormal range. Measured on an RX 7900 XTX (RADV, no denorm-preserve execution
mode): a weight of 1.8e-35 comes back exactly, a weight of 5.5e-42 comes back as
`0.0`. The fixed and the broken kernel return the same bytes there, so a test
would be a green light wired to nothing; one was written, measured, and deleted
rather than landed. lavapipe was tried as a denorm-preserving second
implementation and the backend refuses it (it needs a pinnable subgroup size of
32; lavapipe's range is [8, 8]).

To close this, either run it under an implementation that preserves subnormals,
or enable `VK_KHR_shader_float_controls`'s denorm-preserve on the pipeline and
re-run the deleted case. The finding and these numbers are also recorded in the
shader beside the fix.

Unrelated residual in the same branch: the extra serial `mx_all` scan runs on
thread 0 once per token per MoE layer, and V2-Lite does take this branch
(softmax gating, `norm_topk_prob = false`). The cost was not measured.

### B50 — Metal cannot run a DeepSeek MoE layer (2026-08-10)

**Tag:** CR-2026-08-09 deepseek · **Blocked on:** nothing; it is a missing
kernel feature, not a bug, and no one has asked for DeepSeek on Metal

`infr-metal`'s `Op::MoeFfn` arm implements softmax gating + top-k renorm +
output-weighting and asserts on anything else. V2-Lite ships
`norm_topk_prob = false` and V3 is sigmoid-gated, so both already fail that
assert — DeepSeek MoE layers are CPU + Vulkan only. MLA attention itself IS
implemented on Metal and is unaffected.

The arm also read neither `exp_probs_b` nor the group-routing fields: it
destructured them away with `..`, so softmax + renorm + `expert_group_count > 1`
— a legal `deepseek2` config — passed the assert and then routed with neither
the bias nor the group mask applied, picking the wrong experts with no error.
That combination now asserts too, so the gap is loud rather than silent, but the
underlying feature is still missing.

Closing it means teaching the Metal router the same two extensions
`moe_topk.comp` grew: select on `probs + exp_probs_b` while weighting from the
unbiased probs, and mask all but the top `n_expert_groups_used` groups scored by
their top-2 sum. `moe_topk.comp` is the working reference. Note there is no
Apple hardware on the dev box, so this can only be verified on the macOS CI job.

### B52 — weight shapes: which tensors the loader still does not check (2026-11-12)

`seam/wshape.rs` derives each weight's expected dims from `Config` and `wload`
refuses a mismatch by name with both shapes (trailing size-1 dims ignored, as in
llama.cpp). DeepSeek V4's mixer, compressor, indexer and hyper-connection
tensors are checked too (`dsv4_weight_shape`, transcribed from
`deepseek4.cpp::load_arch_tensors`; `attn_output_a` is compared with its dims
past the first folded, so llama.cpp's 3-D form loads), and a V4 file whose
tensors follow a different `attention.key_length_swa` is refused by name.
`synthetic_deepseek2.rs` grows each V4 tensor by one row and expects that
refusal. Not checked, because the expected shape could not be derived with
confidence or the loader does not go through `wload`:

- MLA `attn_k_b` / `attn_v_b` (the `Op::Mla` kernels read them in their own
  head-major layout), and the V3.2 lightning indexer.
- qwen35 DeltaNet tensors, the dual-FFN router / banks / extra norms, E2B
  per-layer embedding tensors, DiffusionGemma's self-conditioning MLP.
- A KV-sharing layer's `attn_k` / `attn_v` (loaded, never read).
- `token_embd` when only the host embedding path reads it (CPU backend, untied
  `output.weight`): that path does not call `check_weight_shape`, so a wrong
  `ne0` is read at the wrong stride. Applies to every arch, not only V4.
- The MTP head, whose loader does not go through `wload`.
- V4's `ffn_gate_tid2eid` is checked, but by `generate_dense_backend`'s own
  vocabulary check rather than by `wshape.rs`.
- A LEADING size-1 dim is not ignored; no GGUF writer is known to emit one.

### B53 — V4's KV geometry duplicates the V side (2026-08-10)

**Tag:** CR-2026-08-09 deepseek · **Blocked on:** a one-line change in
`crates/infr-cpu`, which the wiring slice did not own

`seam::kv_row_elems` now has a `deepseek4` branch and it returns
`(head_dim, head_dim)` — one MQA row per side per token. **The V side is a
DUPLICATE of the K side, written by a second `Op::WriteKv` from the same source
row**, and that is not what the arithmetic wants.

V4's raw attention is `build_attn_mha(q, k_all, k_all, …)`: K and V really are
the same rows, so `(head_dim, 0)` — MLA's aliasing, with `kv_side_elems`
supplying the placeholder and `Op::Attention` pointed at one buffer for both
sides — is the correct shape. It is not the shape this codebase can execute. The
CPU backend's `Op::Attention` arm takes `cpu_buf(kbuf).read()` and
`cpu_buf(vbuf).read()` as two simultaneously-live guards, and a KV buffer is
`CpuStore::Owned(Mutex<Vec<u8>>)` — a non-reentrant `std::sync::Mutex`. One id
bound to both sides therefore self-deadlocks on the first V4 attention op. (The
CPU MLA arm and `Op::LightningIndexer` both already take exactly ONE guard,
which is the shape to copy; Vulkan is fine with the aliasing — both bindings are
`readonly` and `Recorder::sync`'s dirty-set tracking de-dupes.)

**To close it:** make the CPU arm take one guard when `k_cache == v_cache` (or
take one guard and dequant twice from it), then flip the branch to
`(head_dim, 0)` and drop the second `Op::WriteKv` in the `MixerW::Dsv4` emit.
Add a CPU parity test that binds one id to both sides — there is none today.
Worth roughly half of a V4 session's KV bytes. Also worth a `k_cache == v_cache`
short-circuit in the Vulkan adapter's dequant prepass, which would otherwise
dequant the same cache twice into two pooled scratch buffers (`kvdeq_k` /
`kvdeq_v`); harmless for V4's f16 cache, live if a quantized V4 KV ever lands.

**The compressed caches and compressor states are still unmodelled**, and cannot
be modelled by this helper: they are per-layer (a ratio-0 layer has none) and
the three compressor states are fixed-size recurrent buffers rather than
per-token rows, so the precedent is `MixerW::DeltaNet`'s conv/S-state
allocation. Sizes are tabulated in `docs/deepseek.md` § Stage 4. Nothing reads
them: `generate_dense_backend` refuses a non-zero ratio before a graph is built.

### B58 — what the concurrent-pull slice did NOT verify (2026-08-10)

**Tag:** concurrent-pull slice coverage · **Blocked on:** nothing; each line is
a gap, stated so the coverage claim stays honest

`hub.pull_jobs` and `pull::fetch_all` are covered by six tests against a local
HTTP origin (`infr-hub/src/testhttp.rs`) — all files land, the bound is asserted
on the peak the SERVER observed, `jobs = 1` stays sequential, a planted partial
resumes under concurrency, a stale `If-Range` restarts rather than splices, a
sha256 mismatch is refused and unlinked, and a failure stops the fan-out — plus
one real 229 GB five-shard pull. Not covered:

- **A concurrency bound above 5 in anger.** SUPERSEDED in part by the ranged
  slice, which ran the default 8 against one object: aggregate 79.9 MB/s, 10.0
  MB/s per connection against 15.6 MB/s for a lone one. So the knee is somewhere
  between 5 and 8 and the AGGREGATE is flat across it (78.7 MB/s at five, 79.9
  at eight) — this host tops out near 80 MB/s whatever the count. What is still
  unmeasured is a bound above 8, and whether the ceiling is the link, the CDN or
  a per-client cap.
- **Two `infr` processes pulling the same model at once.** The per-blob `flock`
  that serialises them is unchanged and still unit-tested (`infr_plat::lock`'s
  `a_held_lock_excludes_a_second_holder`), but no test starts a second process,
  and the cross-process case is now N locks held at once instead of one.
- **A repo with more files than the bound.** `DeepSeek-V3.2-REAP`'s 236 shards
  is the case the bound exists for, and it has only been exercised at 10 files /
  3 workers against a local origin with the progress group HIDDEN. What that
  leaves unknown is the shape of the bar block over a long queue: `indicatif`
  reaps a finished bar's line only once every bar ahead of it has also finished
  (`BarState::drop` marks a zombie; `MultiState::draw` reaps consecutive zombies
  from the head), so files completing out of order could leave the redrawn block
  growing past `pull_jobs` lines. Five shards did not show it. If it turns out
  to matter, the lever is `MultiProgress::remove` / `finish_and_clear` on each
  completed bar, at the cost of the per-shard `✓` line.
- **`HF_TOKEN` on a gated repo.** The token now reaches `download_to_blob` from
  inside rather than as a parameter; unchanged in effect, but no gated repo was
  pulled.
- **The progress rendering was read, not eyeballed live.** The five-bar block
  was captured from a `script`-allocated pty and inspected as escape sequences
  (one line per shard, `MultiProgress::suspend` clearing the block around each
  log line), not watched on a real terminal.

### B59 — what the ranged-download slice did NOT verify (2026-08-11)

**Tag:** ranged-pull slice coverage · **Blocked on:** nothing; each line is a
gap, stated so the coverage claim stays honest

Intra-file ranged parallelism (`infr_hub::ranged`, `infr_hub::parts`) is covered
by tests against the local origin (`infr-hub/src/testhttp.rs`) plus unit tests
for the sidecar, the grid, the identity check and the connection budget — each
one shown red before green by breaking what it guards — and by one real pull of
`unsloth/DeepSeek-V3.2-GGUF:UD-TQ1_0`: 161 280 830 528 bytes over 2 404 ranges
through two deliberate `SIGKILL`s, 33.6 minutes at 80.0 MB/s end to end, sha256
equal to HF's `lfs.oid`. Not covered:

- **A re-upload of a REAL object mid-download.** The splice guard is exercised
  only against the local origin. In production the guard that actually fires is
  the plan-time comparison of HF's LFS oid (`x-linked-etag`) with the one in the
  sidecar; the per-chunk `If-Range` has never been seen to reject anything,
  because the CDN answered all 2 404 chunk requests `206`.
- **A connection freed mid-chunk.** A range worker only tries to grow the
  fan-out between chunks (`ranged::worker`), so a download already inside its
  last chunk cannot pick up a permit another file just released. Bounded by one
  chunk time (64 MiB, seconds), and not worth a wake-up mechanism until
  something measures it.
- **An `ETag` that differs between CDN edges.** If one ever did, `If-Range`
  would make that chunk come back `200` and the file would restart as a single
  stream — correct, but slow, and nothing would say why beyond one `warn!`.
  Unobserved in a 2 404-chunk pull, so the risk is bounded but not zero.
- **Chunk retry, beyond the local origin.** A transient failure (send or read
  error, short body, `5xx`, `429`) is retried in place up to `CHUNK_ATTEMPTS`
  times, tested with one dropped body and one `503`. Not tested: the progress
  bar giving back a failed attempt's bytes (`ProgressBar::dec`, read only), and
  a retry against the real CDN, which never failed a chunk in the 2 404-range
  pull.
- **The 64 MiB chunk size was not swept.** It was picked from the two costs it
  trades (a request per chunk against work lost on interruption) and the real
  pull confirms the request overhead is not visible at 80 MB/s, but 32 or 128
  MiB were never run.
- **`hub.pull_jobs` above 8.** See B58: aggregate throughput is flat between
  five and eight connections on this host (78.7 → 79.9 MB/s), so the interesting
  question is now whether anything is left above 8 — untried.
- **Two processes pulling the same big file.** The per-blob `flock` that
  serialises them is unchanged and still unit-tested, and both modes take it
  from the same `.dl-…lock` name so a mode difference cannot dodge it. No test
  starts a second process.
- **A gated repo over ranges.** The bearer token is attached to the probe and to
  every chunk request, but reqwest drops `Authorization` across the cross-origin
  redirect to the CDN (as it should — the CDN URL is pre-signed), and no gated
  repo was pulled.
- **The progress block was read, not eyeballed.** One bar per file is what the
  code does (chunk workers all `inc` the same bar); the real pull ran with
  stderr redirected to a file, where bars are hidden.

### B60 — three CPU-vs-Vulkan tests still pass or fail on a whole-vocab logit cosine (2026-08-11)

**Tag:** CR-2026-08-11 · **Blocked on:** the models (none is ≤1B, so none runs
on this box under the iGPU rule)

`cpu_prefill_matches_llama_debug_dump` showed a whole-vocab logit cosine cannot
separate a match from a mismatch: a **correct** V2-Lite row (both engines pick "
Paris", probability cosine 0.9969) scored **0.774**, while two **unrelated**
Qwen3-0.6B rows scored **0.851** (probability cosine 0.0164). The bias every row
of a model shares dominates it.

Swept 2026-10-28: the docs no longer quote an infr-vs-llama.cpp logit cosine as
evidence (`docs/deepseek.md`'s YaRN entry carries the correction). The
CPU-vs-Vulkan figures in `docs/deepseek.md` and the synthetic tests in
`tests/synthetic_deepseek2.rs` also assert the two backends' argmax is equal, so
the cosine there is a report, not the gate. What still gates on it, all in
`crates/infr-llama/tests/cpu_backend.rs`:

- `gpu_seam_matches_cpu_qwen35moe`: top-5 overlap plus `cos > 0.5`.
- `gpu_seam_matches_cpu_diffusion_gemma`: `overlap || cos > 0.78`, then
  `cos > 0.7`. On the B60 numbers an unrelated row clears both.
- `gpu_seam_matches_cpu_diffusion_gemma_denoise`: per row
  `overlap || mask_tie || cos > 0.8`, and a min-row floor of `0.75`.

Their thresholds were calibrated against measured healthy runs (the comments
record 0.74–0.89), which is exactly the range an unrelated pair also reaches.
Replace each with a cosine over softmax probabilities, re-measured on the model
it guards, and show a broken backend fails it. Needs Qwen3.5-MoE and
DiffusionGemma-26B, plus a GPU the user allows them on.

### B65 — compute grids that can still exceed `maxComputeWorkGroupCount[0]` (2026-11-12)

The spec guarantees 65,535 workgroups per dimension. The elementwise kernels,
the KV store / quantise / dequantise family, rope, `qk_norm_rope`,
`attention_kv`, `attention` and `linear_bf16` now go through `dispatch_wide`,
and their shaders rebuild the flat index from the 2-D grid; a grid at or below
the limit is dispatched exactly as before. A default prefill already takes the
split on every device (`quant_q8` at 512 rows and a 4096-wide layer is 65,536
groups). `elementwise_ops_split_wide_grids` forces a split for each converted
kernel and compares every byte, with sentinel bytes past each buffer's end.
Still on a flat or tiled grid, each needing real shader rework:

| Sites                                                                                                                      | Grid                                                        | Overflows when                                                                 |
| -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `matmul_proj_at`, native off / cm8 / f16a / f32a, split-K's main GEMM, mmq q4k / q6k / `mmq_at`, `fma_at`                  | tiles: ⌈m/64⌉·(n/64), or n/128 for the warp tiles, × splits | m = 1,024 with n ≥ 262,144 (only if the vocab head ran over every prefill row) |
| i8cm (three variants)                                                                                                      | ⌈m/128⌉·(n/16)                                              | m = 1,024 with n > 131,070                                                     |
| f8cm (two variants), bf16cm                                                                                                | tiles                                                       | not computed                                                                   |
| `attention_kv_split_impl`                                                                                                  | rows·nh·4, shared memory                                    | rows·nh > 16,383 (small-m decode path)                                         |
| `attention_kv_split_dyn_inner`                                                                                             | nh·n_chunks, shared memory                                  | nh·n_chunks > 65,535; the adapter's bound on n_chunks was not checked          |
| `attention_prefill_flash`, `flash_reg`, `nc_fa`                                                                            | ⌈mpad/bm⌉·nh, shared memory                                 | (mpad/64)·nh > 65,535                                                          |
| layernorm, softmax, the eager `ops.rs` rmsnorm, `embed_gather`, `dg_eb_sample`, `moe_topk`, `quant_f8_row`, `quant_q8_row` | one workgroup per row                                       | rows > 65,535                                                                  |
| DeltaNet norm and scan passes, `hyper_mix`, `conv1d_silu_at`, `moe_ep_band_remap`                                          | per the dispatcher                                          | not computed                                                                   |

Only the GEMM tiles are near today's shapes. `decode_gemv_bw` (a bench) stays on
the default device whatever `INFR_DEV` says: its single-submit loop reset the
iGPU when pointed at it.

### B-DSV4-WIRING — what the V4 graph slice still owes (2026-08-10)

**Tag:** CR-2026-08-09 deepseek · **Blocked on:** nothing

**Slice A (ratio 0) is DONE and generates** — see `docs/deepseek.md` § Stage 4,
"Slice A". What is left:

**Slice B — ratios 4 and 128.** Needs new ops, not just wiring:

- **The plan half of `dsv4_build_comp_plan` is ported (2026-08-30), the buffer
  half is not.** `build_dsv4_comp_plan` in
  `crates/infr-llama/src/seam/dsv4_plan.rs` is a pure Rust function — no graph
  emission, no backend calls — that computes the same
  `n_visible`/`n_kv`/`state_pos`/`state_read_idxs`/`state_write_idxs`/
  `state_write_pos`/`state_persist_{src,dst}_idxs` vectors the reference does,
  including the `[persistent | scratch | sentinel]` gather layout and the
  two-contiguous-halves read order for the overlapping compressors. Covered by
  literal-vector tests (not just lengths) and red/green fault injection on the
  floor-vs-ceil, `n_kv` padding, overlap-halves-order, persist-dedup and
  CSA-only-dummy traps. **Nothing calls it yet** — no persistent compressor
  state buffers, no `Op::CompressPool` wiring, not referenced from `runner.rs`.
  Narrower than the reference, deliberately:
  - **Single stream only.** Takes an explicit `n_seqs` and refuses above `1`,
    mirroring the reference's `n_stream <= 1 && ubatch.n_seqs_unq > 1` throw.
    `dsv4_stream_offset`'s arithmetic (`0` whenever `n_stream <= 1`) is dropped
    rather than carried as dead code.
  - **No rollback/`n_rs_seq` planes** (`state_restore_*`/`state_snapshot_*`):
    infr's V4 has no speculative-decode rollback consumer, so they are left off
    the plan struct entirely rather than carried as vectors nobody fills.
  - **No coupled-ubatch CSA branch** (`dsv4_ubatch_has_coupled`): only the
    non-coupled dummy-block path is ported, since a single sequence is never
    coupled.
  - **No contiguity guard, because it cannot fire here.** The reference pads to
    `ceil(max(1, ubatch.n_seq_tokens)/ratio)` blocks and refuses a sequence more
    than one block short of that. For a single-stream cache —
    `raw_per_seq == false && comp_per_seq == false`, infr's exact and only scope
    — `llama_kv_cache_dsv4::init_batch` always splits with `split_simple`, whose
    `ubatch_add(idxs, idxs.size(), false)` passes `n_seqs == n_tokens` and so
    makes `n_seq_tokens = n_tokens/n_seqs` unconditionally `1`, whatever the
    ubatch's token count. That block count is therefore the constant `1`: the
    padding rule collapses to "an ubatch that commits nothing gets exactly one
    dummy", and the refusal's `n_writes + 1 != n_blocks` is only reachable at
    `n_writes == 0`, where it holds trivially. The port implements the collapsed
    rule and omits the unreachable refusal. **A future multi-stream slice must
    revisit this**, since `split_equal` does produce a real per-sequence token
    count — that is the point at which `n_seq_tokens` has to become a parameter
    rather than a constant.
- **`Op::Attention::key_bias` landed on CPU/Vulkan/Metal (2026-08-30), but
  nothing calls it for CSA yet.** The op-level capability is done and
  parity-tested (`crates/infr-llama/tests/seam_op_parity.rs`'s
  `attention_key_bias_*` tests, `crates/infr-metal/tests/parity.rs`'s
  `attention_key_bias_*_parity` `#[ignore]`d tests) — same kernel family as
  `sinks` on every backend, combinable with it. What is still missing is the CSA
  graph itself: a builder in `runner.rs` that concatenates `[raw | compressed]`
  K/V, runs `Op::TopkMask` over it, and passes the result as `key_bias` on an
  ordinary `Op::Attention` alongside `mw.sinks` — none of which exists yet
  (deepseek4's only real `Op::Attention` sinks call site in `runner.rs` still
  passes `key_bias: None`). Needs the `[raw | compressed]` concatenation layout
  decided first, which is part of Slice B above.
- **`Op::LightningIndexer`'s contract is written for V3.2's per-token key
  cache.** V4's keys come out of the compressor, so `top_k` counts compressed
  blocks; re-read the op's `k_cache`/`kv_len` meaning before reusing it.
- The per-layer KV geometry those tiers need (§ B53) and the `wpush` handles for
  their tensors, which the ratio-0 arm deliberately does not declare — the
  `assert!` at the top of the build closure is what keeps that honest.

**What slice A left unverified, in its own right:**

- **`batch > 1` has never been EXECUTED for V4 on any backend.** The only V4
  fixture that exists (`crates/infr-llama/tests/synthetic_deepseek2.rs`'s
  `dsv4_model`) writes f32 expert banks, so `moe_batched_ok` is false and the
  chunked batched prefill is unreachable from it — and `generate_dense_backend`
  now excludes V4 from that path outright rather than leave an untested shape
  live. Every V4 graph the tests build is `batch == 1`. Re-enabling it means a
  quantized-expert fixture (or a real GGUF) plus the layer-span question below.
- **A partial layer span is refused** (the assert beside the compress-ratio
  one): the widened residual is `hc_mult` streams wide and a span hands over one
  `n_embd`-wide `hidden` buffer. Layer-major prefill therefore cannot serve V4
  as written; a span-carried widened stream would need `hidden` to be the wide
  buffer for V4 builds.
- **Metal has never been executed** (no Apple hardware). The V4 emit is
  backend-generic, so Metal will take it as soon as its own gaps close — but its
  DEVICE MoE path asserts `MoeGating::Softmax` and V4's gating is mandatory
  `SqrtSoftplus` (§ B-DSV4-HASH), so a V4 MoE layer aborts there first.
- **A real V4 GGUF HAS now been opened** (2026-08-12) —
  `ggml-org/DeepSeek-V4-Flash-0731-GGUF:Q2_K_S`, 43 layers, 1328 tensors. Every
  tensor name and every shape `dsv4_model`'s formulas derive, evaluated at the
  real file's hyperparameters, matches the file exactly: no missing tensor, no
  extra tensor, no shape mismatch. `Config::from_gguf` and `wload`'s `is_dsv4`
  arm both run clean and the model reaches the designed compressed-layer
  refusal. The formulas that were only inferred and are now MEASURED:
  `attn_output_a` = `[n_head·key_length/o_groups, o_lora_rank·o_groups]`,
  `attn_output_b` = `[o_groups·o_lora_rank, n_embd]`, `hc_*_fn` =
  `[hc_mult·n_embd, (2+hc_mult)·hc_mult]` with a `[3]` scale, the compressor's
  `coff = 2 iff ratio == 4` channel doubling, and `attn_compressor_norm` staying
  `[key_length]` in BOTH tiers while `_kv`/`_gate`/`_ape` widen with `coff`.
  What is still NOT verified against a real file: any NUMBER the graph computes
  — the refusal fires before a single forward pass, so shapes and names are all
  this proves.

### B-DSV4 — what the V4 attention primitives do NOT cover yet (2026-08-10)

**Tag:** deepseek · vulkan · metal · **Blocked on:** nothing; scoped out of the
op-level slice, which was explicitly "add the primitives, emit nothing"

Each of the three new capabilities lives in exactly ONE kernel per backend
rather than across the whole attention/norm/rope tier ladder. That was
deliberate — a tier that quietly ignored a sink or a sign would produce
plausible wrong numbers, and there is no caller yet to justify the fan-out — but
the V4 wiring slice inherits the list. Every refusal below is a loud error, not
a silent fallback.

- **`Op::Attention { sinks }` on Vulkan** runs only `attention_kv.comp`'s
  `-DSINKS` build: f16 K/V, bound descriptors, static recording, causal/SWA.
  Refused: a Q8_0 or any other non-f16 cache (the dequant→f16 prepass sits below
  the early return that routes to the sinks kernel), and `AttnMask::Canvas`.
  `decode_eligible` also returns false for any graph containing a sinks op, so a
  V4 decode loses the record-once replay tape. Nothing routes to flash,
  non-FA-coopmat, split-K or the mrows tier, so a sinks layer runs the scalar
  one-workgroup-per-(row, head) kernel at every depth — the thing `attn_partial`
  exists to avoid. A perf pass means teaching at least `attn_partial` +
  `attn_combine` the sink (it folds cleanly into the combine's final `(m, l)`,
  which is where the split-K partials already merge).
- **`Op::Attention { sinks }` on Metal** runs only `ATTN_SINKS_KERNEL`'s two
  instantiations (`attention_sinks_f32`, `attention_sinks_f16kv`). Refused: a
  decoupled or quantized K/V pair, `AttnMask::Canvas`, and the decode-replay
  tape. Same tier story as Vulkan.
- **`Op::QkNorm { weight: None }` on Vulkan** is `rmsnorm.comp`'s `-DNO_WEIGHT`
  f32 build only; an f16 `x` (llama4's post-rope L2-norm shape) errors. V4's Q
  norm reads the f32 `wq_b` output, so nothing needs the f16 twin yet, and a
  build nothing dispatches is a build nothing tests.
- **`Op::Rope { backward: true }` on Vulkan** is static f32 NORM only
  (`rope_back`, `rope_ff_back`). NEOX+backward, f16-out and the record-once
  `_dyn` path all error. V4's de-rope is NORM on an f32 scratch. Metal's rope is
  one runtime-parameterised kernel, so it carries `backward` on every path
  already.
- **A V4 graph cannot take the record-once decode replay tape at all** — four of
  its ops (`Attention{sinks}`, the three `HyperConnect*`, `Rope{backward}`,
  `QkNorm{weight:None}`) have no `_dyn` twin, so `decode_eligible` is false and
  the seam's mirror gate excludes `c.deepseek4` explicitly. Every V4 decode
  token therefore rebuilds + recompiles its graph. That is a real per-token host
  cost (the thing the tape exists to remove) and the first perf item for this
  arch.
- **`Op::Linear` with `w_off` on an F16 weight is still refused on Vulkan.** F32
  now rides a shifted `bufferDeviceAddress` base (the grouped output
  projection's caller); F16 would need the same at `matmul_proj` / `linear` /
  `linear_f16_noext`, plus a `w_off % 2 == 0` obligation on the two that read
  the weight as packed u32 words. Nothing produces an f16 `w_off` today.
- **gemma4's V-norm and llama4's L2-norm still pass a ones-vector weight** to
  `Op::QkNorm`. They could pass `None` and drop a per-graph `head_dim`-float
  allocation each; the numbers are bit-identical (`x * s * 1.0` is `x * s` in
  IEEE, which `qknorm_weightless_matches_a_ones_weight` asserts at
  `max_err == 0`). A cleanup, not a defect. Blocked for llama4 by the missing
  f16-x weightless kernel above (its L2 norm reads f16); for gemma4, no gemma-4
  model is cached here to confirm its goldens after the change.

### B-DSV4-HC — what the Sinkhorn hyper-connection ops do NOT cover yet (2026-08-10)

**Tag:** deepseek · vulkan · metal · **Blocked on:** nothing; scoped out of the
op-level slice, which was explicitly "add the ops, emit nothing"

`Op::HyperConnectMix` / `HyperConnectPre` / `HyperConnectPost` are implemented
and parity-tested on CPU + Vulkan (and typechecked on Metal). What is left:

- **The weightless RMSNorm over the flattened `hc*n_embd` row went the
  ones-vector way**, not the `Op::RmsNorm { weight: Option<_> }` way: one
  `hc_mult*n_embd`-wide vector of 1.0 is uploaded per V4 session, matching the
  three ones-vectors gemma4 / dual-MoE / llama4 already upload. Extending
  `Op::RmsNorm` the way `Op::QkNorm` was extended would drop that allocation for
  all four callers and is the tidier end state; it was out of scope here because
  it is a change to every backend.
- **Performance was not considered at all.** Every kernel is the naive shape:
  `hyper_mix` runs ONE THREAD per token with the `hc × hc` matrix in a private
  array (dynamic indexing into a private array is the classic scratch-memory
  spill), and `hyper_pre` / `hyper_post` are one thread per output element with
  a serial `hc`-term loop and no vectorisation. At `hc = 4` the mix op is
  `rows × 24` floats of work, so it is unlikely to matter; `hyper_post` writes
  `rows × hc × n_embd` and is the one to measure first. Nothing here is
  measured.
- **`hc_mult` is capped at `HYPER_CONNECT_MAX_MULT`** by a host check in each
  backend, and the kernels size a private array by their own copy of it.
  `adapter::tests::hyper_mix_bound_matches_host` fails if `hyper_mix.comp`'s
  `HC_MAX` drifts from the Rust constant; the MSL copy in
  `elementwise_norms.metal` has no such test (Metal is not built on the Linux
  box, so the test would have to live in `infr-metal` and run on the macOS CI
  job).
- **`n_iter` is not exercised above 40, and `eps` only at 1e-6 / 1e-2.** The
  real V4 values come from `{arch}.hyper_connection.sinkhorn_iterations` and
  `.epsilon`, neither of which has been recorded from a real GGUF (see § "Open
  questions" 1 in `docs/deepseek.md`). Blocked on the real model
  (`ggml-org/DeepSeek-V4-Flash-0731-GGUF`), which is not cached here.
- **The four eps sites are not all pinned at production eps.** At `eps = 1e-6`
  the over-dst site moves the answer by ~3e-7, below the 1e-5 tolerance the
  backend comparisons run at; only the synthetic `eps = 1e-2` case pins it for
  an f32 kernel. Same for the asymmetric iteration COUNT (~1e-11 at 1e-6). Both
  are pinned in exact arithmetic by `hyper_connect_details_are_load_bearing`. If
  a real V4 GGUF turns out to use a small eps, a backend that dropped the
  over-dst eps would pass every test here. Same blocker: the real file's eps.

### B-DSV4-POOL — what `Op::CompressPool` does NOT cover yet (2026-08-13)

**Tag:** deepseek · vulkan · metal · **Blocked on:** nothing; scoped out of the
op-level slice, which was explicitly "add the primitive, emit nothing"

`Op::CompressPool` is implemented and parity-tested on CPU + Vulkan (and
typechecked on Metal): the four ggml nodes both V4 compressor variants share,
fused, with the permutes folded into the indexing. What is left:

- **Nothing emits it.** `crates/infr-llama/src` was deliberately untouched. The
  op is only half of what slice B needs: the GATHER that produces its
  `[blocks, window, n_embd]` operands — the `[persistent | scratch | sentinel]`
  layout and the two-contiguous-halves read order for the overlapping
  compressors — is the other half and does not exist (see § B-DSV4-WIRING).
- **An all-`-inf` window returns `0.0`, deviating from ggml's `NaN`.** The
  reasoning is on `Op::CompressPool`'s doc and in `docs/deepseek.md` § "The
  compressed-KV state machine"; the ggml behaviour was read off
  `ggml_vec_soft_max_f32` in `ggml-cpu/vec.cpp`, not assumed. What is NOT
  established is whether a real V4 forward pass ever PRODUCES such a window —
  `dsv4_build_comp_plan` should only ever pool blocks with at least one real
  row, so the case may be unreachable in practice. If it turns out to be
  reachable, the choice of `0.0` becomes a numerical difference from llama.cpp
  worth re-arguing rather than a defensive default.
- **Scores are assumed finite-or-`-inf`.** A `+inf` or `NaN` score is out of
  contract and unchecked on every backend; it would produce `NaN` silently. The
  gather that will feed this op is what should make that impossible, and it does
  not exist yet.
- **Perf is unmeasured, and the kernel is the obvious shape, not a tuned one.**
  Vulkan and Metal run one thread per output element, each walking its own
  window with stride `n_embd`, and read `scores` twice (max pass, then exp
  pass). At HCA's `window = 128` that is 128 strided loads per lane; a
  workgroup-cooperative form (one group per block, lanes splitting the window, a
  subgroup max/sum reduction) is the natural next step but has nothing to
  justify it until there is a caller to profile. The CPU arm keeps two
  `n_embd`-wide scratch vectors per block, allocated inside the chunk closure —
  fine for a test, worth hoisting if it ever runs hot.

### B-DSV4-HASH — what hash-routed MoE / the SwiGLU clamp do NOT cover yet (2026-08-13)

**Tag:** deepseek · vulkan · metal · **Blocked on:** nothing

Hash routing now EMITS and generates on CPU + Vulkan: `Op::GatherI32`
(`gather_i32.comp`, and the CPU interpreter arm) reads the token's row of
`blk.N.ffn_gate_tid2eid` into the `[batch, n_expert_used]` selection
`Op::MoeFfn::expert_ids` consumes, and `generate_dense_backend` no longer
refuses a hash-routed layer. See `docs/deepseek.md` § Stage 4, "Slice A2". What
is left:

- **`Op::GatherI32` is REFUSED on Metal**, by name, in `run_op`. Implementing it
  alone would move the failure one op later without making anything work: the
  Metal DEVICE `Op::MoeFfn` path asserts `MoeGating::Softmax` and V4's gating is
  mandatory `SqrtSoftplus`, so a V4 MoE layer cannot run there either way (the
  same blocker this entry has recorded since 2026-08-10). Whoever lifts the
  sqrt-softplus assert owns the gather too — it is one small MSL kernel, or a
  host arm beside `Op::MoeFfn`'s existing host fallback, which already
  implements `expert_ids`.
- **The gather has NO `_dyn` twin and was not considered for the replay tape.**
  It falls through `decode_eligible`'s `_ => {}`, i.e. it is treated as
  replay-safe (it is: pos-independent, and its `ids` input is the same buffer
  `Op::EmbedGather` reads under the chained decode). Nothing exercises that,
  because V4 is excluded from the tape outright for four other reasons (§
  B-DSV4). Re-examine when V4 gets its tape.
- **The gather runs once per hash layer per step, un-fused and unmeasured.** All
  `hash_layer_count` layers gather the SAME token's ids at the same `n_used`
  width, so the whole selection could be one dispatch over a stacked table (or
  cached across layers) instead of one dispatch each. At `hash_layer_count = 3`
  on the shipped file that is 3 dispatches of 6 threads — expected to be noise
  next to the MoE itself, but nothing was profiled.
- **The table is read as ONE bound storage buffer**, not through the
  resident-BDA arena. Fine at `n_expert_used * n_vocab` dwords (3.1 MB on the
  shipped file, 8 MB at a 256k vocab and `n_used = 8`), and `Recorder::vkb`
  asserts the range against the device's real `maxStorageBufferRange` — but a
  future table that tripped that assert would need a `-DSTREAMED` twin, which
  does not exist.
- **The host-gather alternative was reconsidered and declined again.** The token
  ids ARE host-known on every V4 path today (V4 takes neither the record-once
  tape nor the chained decode, so every step goes through the per-token loop
  where the host has just fed `cur[pos]`), so a host gather would have worked
  with no new kernel. Declined because it needs one extra graph Input and one
  extra per-step upload PER HASH LAYER, a host dequant of every table, and
  because it would have to be undone the moment V4 gets a GPU-resident decode —
  where the sampled id never reaches the host. The device gather costs one small
  shader and reuses the ids Input that already exists.
- **Only `batch == 1` has been executed.** `Op::GatherI32` is written for any
  `rows` and the op-level parity test drives `rows = 2`, but the WHOLE-MODEL
  path is per-token: V4 is excluded from batched prefill (§ B-DSV4-WIRING), and
  `build` asserts a hash-routed span cannot be built without the ids Input,
  which the chunked prefill has nowhere to bind. Re-enabling batched V4 prefill
  means giving `PfChunk` an ids buffer that is separate from its host-embedded
  rows.

- **The Vulkan PAGED MoE path refuses both features.** `execute_paged_moe`
  returns an error for `expert_ids` and for a clamp with
  Sigmoid/`weight_before`. The pager exists for an expert bank too large for
  VRAM (Llama-4-Scout); a V4 hash layer is not that model, and a second untested
  hash path there would have had no caller. If a V4 checkpoint ever needs the
  pager this is the gap.
- **Metal's DEVICE MoE path is only reachable for softmax gating.** Its arm
  asserts `MoeGating::Softmax`, and V4's gating is mandatory `SqrtSoftplus`, so
  a real V4 MoE layer on Metal would abort at that assert even with the gather
  in hand. Hash routing and the clamp are implemented there anyway (`moe_topk`'s
  `hash` flag, `gatedact_f32`'s `do_clamp`/`limit`) and
  `moe_ffn_hash_routing_parity` covers the softmax shape; the sqrt-softplus
  refusal is the real blocker for V4 on Metal and is untouched. Lifting it is
  what makes the refused `Op::GatherI32` above worth writing.
- **Vulkan's strided/sigmoid gated forms refuse the clamp.** `gelu_mul_off`
  (gemma4's per-layer-embd gate) and `mul_sigmoid` push the clamp words as zero
  and the adapter returns an error if a clamp reaches them. V4 is SiLU, so
  nothing needs them; a future clamped GELU-with-strides would.
- **Performance was not considered.** The clamp adds a workgroup-uniform branch
  and one `min`/`clamp` per element to `silu_mul`/`silu_mul_fused` — expected to
  be free, not measured. `moe_topk`'s hash branch replaces the whole `n_used`
  reduction with an `n_used`-element copy, so it can only be faster; also not
  measured.

### B-DSV4-REAL — the shipped V4 file now needs ONLY the compressed tiers (2026-08-13)

**Tag:** deepseek · **Blocked on:** B-DSV4-WIRING slice B

Measured from `ggml-org/DeepSeek-V4-Flash-0731-GGUF:Q2_K_S` (`block_count = 43`,
`hash_layer_count = 3`, `compress_ratios` = `[0, 0, 4, 128, 4, 128, …, 4]` over
the 43 layers, then three surplus zeros): **no layer of the real model is both
ratio-0 and bias-routed.** Layers 0 and 1 are the only ratio-0 layers and both
are hash-routed; layers 2..42 are all compressed (alternating 4/128), and layer
2 is compressed AND hash-routed.

Hash routing landed on 2026-08-13 (§ B-DSV4-HASH), so layers 0 and 1 — 2 of the
43 — are now fully covered, and **the only thing between this checkpoint and a
forward pass is B-DSV4-WIRING slice B**, the compressed (ratio 4/128) tiers.
That is also the only refusal left in `generate_dense_backend` for this file:
`arch=deepseek4 … layer 2 has compress_ratio 4`.

What is NOT established by this entry: any NUMBER the graph computes on a real
file. The refusal still fires before the first forward pass, so what has been
verified against the real GGUF is names, shapes and the two hash-routing
prerequisites (an i32 `blk.{0,1,2}.ffn_gate_tid2eid.weight` of shape
`[6, 129280]` = `[n_expert_used, vocab]`, which is exactly what the new shape
check demands) — nothing about its logits. The hash-routing correctness evidence
is the synthetic fixture and the op-level parity tests, on CPU and Vulkan.

Verified by running `infr run` against the file on the CPU backend (2026-08-12):
config parses, `wload` resolves every tensor, and the refusal is reached. NOT
re-run after the hash slice — the compressed-layer refusal is ordered first and
fires at layer 2 regardless, so the observable would not have moved.

### B-GGUF-META-IGNORED — GGUF keys infr still does not apply (2026-11-12)

**Tag:** gguf

- **`general.sampling.*` beyond temp / top_k / top_p.** Those three are now
  applied (`arch::declared_sampling`), ordered arch table → sidecar
  `generation_config.json` → GGUF keys → anything the user set; llama.cpp reads
  only the GGUF. llama.cpp's `common_init_sampler_from_model` also applies
  `min_p`, `penalty_repeat`, `penalty_last_n`, `xtc_*`, `mirostat*` and
  `sequence`, and infr has no engine-wide setting for any of them, so a file
  that pairs a hot temperature with `min_p` or a repeat penalty gets the
  temperature without its counterweight. `unapplied_sampling_keys` names those
  keys in one load-time warning. llama.cpp also keeps a built-in `min_p` under a
  GGUF temperature; infr has no equivalent. Of the files to hand, only the
  DeepSeek-V4 one declares these keys (temp 1.0, top_p 1.0).
- **`{arch}.embedding_length_out`** = 16384 on V4 (= `hc_mult · n_embd`, the
  un-collapsed hyper-connection residual). In llama.cpp this feeds
  `llama_hparams::n_embd_out()`, which sizes the EMBEDDINGS output buffer only —
  never the logits path — so ignoring it costs nothing while infr has no
  embeddings endpoint. It becomes wrong the moment one exists for an arch whose
  output width differs from `embedding_length`.

<!-- ── hardware-capability audit, 2026-08-11 ─────────────────────────────────
     Three follow-up slices came out of one read-only audit of what infr detects
     about a GPU versus what llama.cpp detects. The prefixes are the slices:
       B-HWDET-*   detection and gating          (task #16)
       B-NVSHAPE-* NVIDIA kernel shapes          (task #17)
       B-DSHW-*    DeepSeek-specific fast paths  (task #18)
     HARDWARE REALITY FOR EVERY ENTRY BELOW: this box has one RX 7900 XTX
     (RDNA3, RADV) and no NVIDIA, Intel or Apple part. Each entry states its own
     evidence; anything about NVIDIA or Intel BEHAVIOUR is read off llama.cpp's
     source, never observed here. -->

### B-HWDET-VENDOR-RULES — the coopmat driver rules and the arch bucket exist now, and NONE of their non-RADV arms can be run here (2026-08-11)

**Tag:** vulkan · detection · **Blocked on:** NVIDIA / Intel / AMD-proprietary
hardware; nothing else

The capability-first design now has exactly one vendor-keyed exception, landed
as `caps::coopmat_trust` + `caps::device_architecture`
(`crates/infr-vulkan/src/caps.rs`), applied in `VulkanBackend::new` as a FILTER
over the enumerated coopmat shape list. Both rules are transcriptions of
llama.cpp's `ggml_vk_khr_cooperative_matrix_support` / `get_device_architecture`
(read at pin 030ebb5): AMD proprietary + AMDVLK are believed only on RDNA3+, and
Intel pre-Xe2 loses the 16x16x16 tier while keeping the 8x8x16 tile that already
sits behind `INFR_CM_8X8=1`.

**What is actually verified:** only the RADV/RDNA3 arm, on this box — the banner
now prints `AmdRdna3/MESA_RADV` and every coopmat tier stayed live
(`f16cm:y i8cm:y`), i.e. the filter is a no-op here. Everything else is
unit-tested over synthetic `DeviceProbe` values only (`caps::tests`), which
proves the RULE, not the PROBE that feeds it: no NVIDIA, Intel or
AMD-proprietary device has ever filled a `DeviceProbe` in this tree.

**The specific things a machine with one of those parts should check first:**

- That `probe_device_facts` actually fills `wavefronts_per_simd` /
  `warps_per_sm` / the packed-dot bits on that driver. A silently-zero field
  misfiles the device as `DeviceArch::Other`, which on an AMD proprietary driver
  means coopmat REFUSED (fail-safe) but on Intel means the 16x16x16 tier refused
  too.
- Whether the Intel `Tile8Only` verdict is the right shape for `infr` at all. It
  was chosen so the `_cm8` `native_gemm_warp` builds stay reachable on Alchemist
  (they are the only kernels at that shape, and the only reason the opt-in
  exists); the alternative — refusing Intel pre-Xe2 coopmat outright, as
  upstream does — would make those builds dead code on every Intel part.
- That an `AMD_PROPRIETARY` refusal is survivable: it drops the device to the
  non-coopmat ladder, which is a large prefill regression, not a failure. There
  is no force-on knob to override the deny-list; adding one is the obvious
  escape hatch if a real device is misjudged.

**Considered and declined:** generalizing the known-answer validation (option 3
of the retired B-HWDET-DRIVERID entry) to the f16/bf16/f8 coopmat tiers. The
int8 one was built because that kernel reads accumulator elements at an
implementation-defined mapping; every other coopmat kernel here moves data in
and out of fragments only through `coopMatLoad`/`coopMatStore` with an explicit
layout enum, so there is no per-element assumption to falsify — a known-answer
GEMM there would test the driver's arithmetic, not infr's assumption, and would
cost init latency on every run.

### B-HWDET-LIMITS — compute limits: what the workgroup-size guard does not cover (2026-08-11)

**Tag:** vulkan · detection · **Blocked on:** a perf measurement, and hardware
with limits below desktop drivers'

Read from the device and enforced now: `maxPushConstantsSize` and the compute
workgroup limits (`caps::KernelLimits`, checked per kernel by
`caps::check_kernel_workgroup` in `ops::try_make_compute_kernel` and in
`matmul_f32`, against the `LocalSize` `spirv::local_size` reads out of the
module), and `maxStorageBufferRange` (`Recorder::vkb`'s release assert). A
kernel the device cannot hold is refused by name, and the kernel cache releases
its lock before that panic, so later requests still build. `Recorder::rmsnorm`
asks the same predicate (`KernelLimits::admits`) before choosing the 1024-wide
`rmsnorm_wide`, so a smaller device decodes on the 256-thread `rmsnorm`. All of
this is tested through test-only overrides (see B144), not on a device that
actually has such limits. What is left:

- **The spec's subgroup-size rule (the VUID ending `-02756`): a pipeline built
  with a pinned `requiredSubgroupSize` may have at most
  `requiredSubgroupSize × maxComputeWorkgroupSubgroups` invocations.** Not
  checked; `maxComputeWorkgroupSubgroups`
  (`VkPhysicalDeviceSubgroupSizeControlProperties`) is not queried. Every kernel
  builds and the `infr-vulkan` suite passes on the RX 7900 XTX and the Raphael
  iGPU (run); whether either is near the limit was not checked, and a device
  with a smaller one would build an invalid pipeline. A `caps::KernelLimits`
  field plus a term in `check_workgroup_size`, testable through
  `TEST_KERNEL_LIMITS`.
- **A device under 256 invocations runs no rmsnorm at all.** The fallback is the
  256-thread build, so such a device is refused at the first norm. Vulkan only
  guarantees 128; desktop drivers report 1024 or more. A 128-thread twin would
  close it if such a device ever matters.
- **`maxComputeWorkGroupCount[0]`** is not read: `Recorder::MAX_GROUP_COUNT_X`
  stays at the spec minimum, so every dispatch wider than that takes the 2-D
  split (`dispatch_wide`) even on RDNA3, whose limit is far higher. Correct
  everywhere; whether the split costs anything measurable has never been
  profiled, and that measurement is the blocker. The flat grids that do not go
  through `dispatch_wide` at all are B65.
- **`minStorageBufferOffsetAlignment`** is not queried: 256, the spec's maximum
  for it, is used everywhere (`BDA_WEIGHT_ALIGN`, `vkb`'s ranges). Correct on
  every device; the over-alignment's cost is unmeasured.

### B-HWDET-FEATUREBITS — the bf16/f8 feature-bit code is written and cannot be exercised on this box (2026-08-11)

**Tag:** vulkan · detection · **Blocked on:** RDNA4-class hardware

`VK_KHR_shader_bfloat16` and `VK_EXT_shader_float8` are no longer gated on the
extension string alone. `crates/infr-vulkan/src/vkext.rs` supplies the two
feature structs ash 0.38 predates (layout and `sType` transcribed from the
system `vulkan_core.h`), `caps.bf16`/`caps.f8` now mean extension AND feature
bit, the coopmat tiers additionally require `shaderBFloat16CooperativeMatrix` /
`shaderFloat8CooperativeMatrix` (the bits the `-DBF16CM` and fp8 kernels'
operand types actually need), and both extension+feature are ENABLED on the
device when their tier is live — which they never were before, so those kernels'
SPIR-V would have violated its VUID on the first RDNA4 run.

**None of that has run.** RDNA3 advertises neither extension, so the query is
skipped and the enables are absent; this box's device-create chain is byte
identical to before. What IS exercised here is the raw `sType`/`pNext` chaining
those structs depend on: `vkext::tests::raw_chaining_matches_ash` hand-rolls
`VkPhysicalDeviceShaderFloat16Int8Features` and asserts the bits match ash's
typed query (passes on the 7900 XTX). So the mechanism is tested and the two
payloads are not.

**First thing to check on an RDNA4 part:** that `create_device` still succeeds
with `INFR_BF16_COOPMAT=1` — enabling a feature the device reports false is a
device create FAILURE, and `ShaderBfloat16Features::enable` only asks for bits
the query returned, which is exactly the code path nobody has run.

### B-HWDET-NO-ARCH-BUCKET — the per-op subgroup-width table is still not representable, and is still a guess (2026-08-11)

**Tag:** vulkan · detection · **Blocked on:** a measurement nobody can take here

The architecture bucket itself now exists (`caps::DeviceArch`, see
B-HWDET-VENDOR-RULES) and has exactly one consumer: coopmat trust. What the
audit listed as the thing a bucket would BUY is still absent and still
unjustified:

- **A per-architecture, per-pipeline subgroup-size table** (upstream's
  `gpu_pipeline_configs` / `rdna1_pipelines` / `rdna2_pipelines`). `infr` has
  two pinned widths — a global 32 and `sg_pref` for one curated decode family —
  so a per-op width cannot be expressed at all. Adding the table is a real
  change to the kernel-cache key, and the evidence for it is upstream's tuning
  of DIFFERENT kernels on hardware that is not here. Still a guess, deliberately
  not built.
- **Wave64-only hardware stays unsupported by rule**: the backend hard-refuses
  any device that cannot pin subgroup 32, before the bucket is ever consulted.
  Recorded as a decision, not a gap. `DeviceArch::AmdGcn` exists only so such a
  part is classified as known-unsupported rather than unknown.

### B-NVSHAPE-COOPMAT2 — `VK_NV_cooperative_matrix2` is gated and researched; the kernel path does not exist (2026-08-11)

**Tag:** vulkan · nvidia · **Blocked on:** NVIDIA hardware; nothing else

The adapter has no coopmat2 kernel — apart from the capability gate described
below, the only coopmat2 code anywhere in `crates/` is
`examples/coopmat2_test.rs`, a standalone probe with its own instance and
device. It is unusually well prepared: its header records the
`coopMatPerElementNV` signature (a `void` with an `out` result, not a
value-returning call), confirms via `spirv-dis` that it lowers to
`OpCooperativeMatrixPerElementOpNV` with no shared memory and no
`OpControlBarrier`, and it benches two epilogue strategies (single-tile and a
full 512×2048×2048 int8 GEMM) to answer whether per-element access removes the
int8 "rescale tax". **It has never been run** — RADV does not expose the
extension and there is no NVIDIA part here. So the question it was built to
answer is still open.

The direct motivation was the int8-coopmat fragment-layout question (formerly
B-HWDET-I8CM-FRAGLAYOUT, closed by `95fa3ce`, which verifies the empirically
derived mapping with an init-time probe rather than trusting it): coopmat2's
per-element callback addresses `(row, col)` **portably**, which would retire
that probe and the empirically derived fragment mapping entirely on hardware
that has it.

**The GATE now exists** (`caps::check_coopmat2_support` over `Coopmat2Probe`,
probed by `probe_coopmat2` in `lib.rs` and reported as `cm2:` in the device
banner). It is a transcription of llama.cpp's — all seven coopmat2 feature bits
plus `bufferDeviceAddress`, then fp16 A/B with fp16 AND fp32 accumulators at
BOTH 128- and 256-invocation workgroup sizes, then `maxDimension >= 512` — and
it fails closed on anything it cannot establish. What is still missing is the
kernel path: nothing consumes an `Ok`.

What has actually been exercised here: the refusal path on lavapipe, which
**does** advertise `VK_NV_cooperative_matrix2` on this box (device `Vulkan2`)
and is refused because `cooperativeMatrixWorkgroupScope` is clear — i.e. the
case that proves an extension-string check would have been wrong. No device here
has ever reached `Ok`, so the ACCEPT side of the gate rests on unit tests alone.
The optional bf16 half of upstream's gate was deliberately not copied: it only
feeds bf16 coopmat2 shaders, and there is no coopmat2 shader of any kind.

`examples/coopmat2_test.rs` deliberately does NOT go through this gate. It uses
only `coopMatPerElementNV`, not flexible dimensions / tensor addressing / block
loads, so the full gate would make it skip on hardware where its own experiment
would run.

### B-NVSHAPE-CORECOUNT — the shader-core count now exists on every vendor; nothing occupancy-driven consumes it (2026-08-11)

**Tag:** vulkan · nvidia · intel · **Blocked on:** NVIDIA/Intel hardware to
validate; a decision on whether the split-K constants should be device-derived
at all

`caps::shader_core_count` now sources the count the way llama.cpp does —
`VK_NV_shader_sm_builtins`' `shaderSMCount` on NVIDIA,
`VK_AMD_shader_core_properties2`' `activeComputeUnitCount` on AMD, and a PCI
device-id table on Intel transcribed verbatim from
`ggml_vk_intel_shader_core_count` — with the old `VK_AMD_shader_core_properties`
(v1) product kept as a fourth fallback so an AMD driver advertising only v1 does
not regress to 0. It lands in the existing `Capabilities::compute_units` (0 =
unknown), so the field name and every consumer are unchanged.

Verified on this box: both AMD devices report the same number through v2 as they
did through the v1 product (RX 7900 XTX 96, Raphael iGPU 2), so preferring v2 is
a no-op here. **The NVIDIA and Intel branches have never executed** — no such
part exists on this machine.

**Consequence nobody has watched run:** the only consumer is
`infr_core::integrated_ubatch_rows` via `infr_llama::seam::default_ubatch_rows`,
which scales the integrated prefill chunk from the count. An Intel iGPU in the
table (0xB080, Panther Lake Xe3 LPG, 12) previously got 0 → the 128-row floor
and would now get 768 rows. That is what the scaling function means, but it is a
6x larger per-submit chunk on a watchdog-sensitive part, unvalidated. If an
Intel iGPU ever hangs in prefill, this is the first thing to look at;
`INFR_UBATCH` overrides it.

**Deliberately NOT wired into split-K or tile selection.** The audit that
produced this entry left `infr`'s split-K selection untraced; it has now been
traced, and the answer is that a core count does not obviously improve it:

- Four independent sites carry a hardcoded "workgroups needed to fill this GPU"
  number — `split_k_plan` (`adapter.rs`, threshold 128 / target 256), the two
  flash split-K formulas and the PV split-K formula (`recorder.rs`, threshold
  1024 / target 2048), plus the `wide_grid >= 128` wide-tile gates and
  `ATTN_SPLIT.target_chunks = 32`.
- Their implied workgroups-per-core ratios are not the same number: 256/96 ≈ 2.7
  for the GEMM, 2048/96 ≈ 21 for attention. So the constants encode per-kernel
  occupancy (waves in flight, register/LDS pressure), not core count, and
  nothing in the probe knows that factor.
- Replacing them with `cores * K` therefore means inventing a `K` per site,
  chosen so that 96 reproduces today's empirically tuned constant, and then
  extrapolating it onto hardware nobody here can measure. That is a guess with a
  silent failure mode, so it was not built.
- Two extra costs if someone does: `split_k_plan` is pinned by a formula
  equivalence test in `adapter.rs`, and the attention chunk policy is
  **duplicated in GLSL** (`attn_live.comp` recomputes it on the GPU), so a
  device-derived target needs a new push constant.

The honest next step is a measurement, not a refactor: a second GPU with a
different core count, running the four sites' current constants against
core-derived ones.

### B-NVSHAPE-MMQ-PER-ARCH — upstream tunes quantized-GEMM tile shape per microarchitecture; infr has one shape (2026-08-11)

**Tag:** nvidia · perf-design · **Blocked on:** nothing to read; everything to
validate (no NVIDIA hardware)

**This is a CUDA finding, and `infr` has no CUDA backend** — it is recorded as a
design reference for how far per-shape tuning can go, not as a portable path.

llama.cpp's CUDA MMQ kernel selects its configuration from a
per-microarchitecture table:
`ggml/src/ggml-cuda/mmq-config-{pascal,ampere,blackwell,cdna,rdna2,rdna3,rdna3-5,rdna4}.cuh`.
Each entry is keyed on `(quant type, J, fallback)` and carries `nthreads`,
`occupancy`, `I`, `J`, an SRAM layout, `K_vram`, and a `stream_k` flag — the
`ggml_cuda_mmq_config` struct in `mmq.cuh`. Read directly. Two rows for the same
quant type show the spread: Ampere runs `nthreads=256, occupancy=1, I=128` with
**`stream_k = true`**, while RDNA3 runs
`nthreads=128/256, occupancy=2, I=64/128` with `stream_k = false`, and the two
enumerate different J granularities. `rows_per_warp()` also branches — 16 on AMD
MFMA/WMMA, 32 on NVIDIA when `J >= 48 && J % 16 == 0`.

Two things worth carrying over independent of CUDA:

- **Stream-K decomposition** is on for NVIDIA and off for RDNA in this table.
  `infr` does not have stream-K at all (it has split-K, which is a different
  decomposition). Whether it would win is unmeasured and unknown.
- The **tile shape is a function of the quant type and the N dimension**, not a
  constant. `infr`'s GEMM shape gates are coarse by comparison (`out_f % 256` /
  `in_f % 32` style divisibility tests choosing WIDE vs NARROW_N).

For the NVIDIA tensor-core shapes themselves: `mma.cuh` issues
`mma.sync.aligned.m16n8k16` and `m16n8k32` for `s8×s8→s32`, falling back to 2×
and 4× `m8n8k16` on Turing (`__CUDA_ARCH__ < GGML_CUDA_CC_AMPERE`); f16 uses
`m16n8k16` with a 2×/4× `m16n8k8` Turing fallback; there is a tf32 `m16n8k8` and
a Blackwell block-scaled `m16n8k64` mxf4/nvf4 path. Note `TURING_MMA_AVAILABLE`
is `!defined(GGML_USE_HIP) && __CUDA_ARCH__ >= 750`, so **none** of this is
reachable on AMD-via-HIP; AMD goes through `AMD_MFMA_AVAILABLE` /
`AMD_WMMA_AVAILABLE` instead. All read directly.

### B-NVSHAPE-AMD-NOT-HERE — CDNA MFMA and RDNA4 WMMA cannot be exercised on this box (2026-08-11)

**Tag:** amd · coverage-gap · **Blocked on:** hardware

Stated as a coverage gap, not a task. This machine's RX 7900 XTX is RDNA3
(gfx1100). From llama.cpp's own architecture constants, read directly in
`ggml/src/ggml-cuda/common.cuh`: RDNA3 is the **minimum for WMMA**, CDNA1 the
minimum for MFMA, RDNA2 the minimum for dp4a. So this box can exercise WMMA and
dp4a but **never** MFMA (CDNA/MI-series only) and never RDNA4's WMMA generation
(RX 9000).

Concretely, that means the following stay unvalidated here no matter how much
work is done: `infr`'s `coopmat_bf16` and `coopmat_f8` tiers, whose field docs
already say they are `None` on all pre-RDNA4 hardware and whose dispatches are
opt-in specifically because "this dev box has none" (read in `adapter.rs`'s
`bf16cm_ok` comment). Any claim that those tiers are correct is currently a
compile-time claim only.

### B-DSHW-MLA-SCALAR-TIER — DeepSeek attention has one subgroup tier now; the occupancy arm is still missing (2026-08-11, re-measured 2026-08-12)

**Tag:** deepseek · vulkan · perf · **Blocked on:** nothing

**Re-measured 2026-08-12** on the forward that adds attention to the residual
ONCE (7e778ee) — every number below is from this box today, RX 7900 XTX / RADV,
`JenniSD/DeepSeek-V2-Lite-Chat-Q4_K_M-GGUF`, oracle = `llama-bench` on
llama.cpp's **Vulkan** backend, same GGUF, `build: c629da5 (417)`. `infr` rows
are the means of three reps with the arms ALTERNATED (`INFR_NO_MLA_SG` toggled
between reps, not run in blocks):

| bench       | before (scalar `mla`) | after (`mla_sg`)       | gain  | llama.cpp Vulkan | ratio now |
| ----------- | --------------------- | ---------------------- | ----- | ---------------- | --------- |
| pp512       | 965.8 [963.3–969.7]   | 1035.5 [1030.2–1042.8] | 1.07× | 4470.77 ± 100.32 | 4.3×      |
| pp2048      | 671.3 [670.6–672.0]   | 851.1 [847.2–855.4]    | 1.27× | 4001.94 ± 13.56  | 4.7×      |
| tg64        | 111.6 [111.5–111.8]   | 128.7 [127.9–129.4]    | 1.15× | 225.73 ± 7.17    | 1.8×      |
| tg64 @ 2048 | 6.3 [6.3–6.3]         | 11.9 [11.9–11.9]       | 1.89× | 216.52 ± 2.95    | 18×       |

The pre-7e778ee numbers this entry used to carry (975.5 / 673.4 / 111.2 / 6.3)
are within noise of the "before" column, so the doubled-residual fix moved
throughput by nothing — it removed one `Op::Add` per layer against a kernel that
was 90%+ of the frame.

**What shipped: `mla.comp`'s `-DMLA_SG` builds** (`mla_sg`, `mla_sg_ff`,
`mla_sg_bias`, `mla_sg_ff_bias`), selected by `gemm::mla_sg_tier` and dispatched
through `kernel_sg(.., MLA_SG_WIDTH)`. Two structural changes to the key loop,
nothing else — same dispatch shape, same bindings, same push constants:

- **One pass, not two.** The scalar build walks the key range twice (once for
  the row max, once to exponentiate), computing every `sq · K[j]` dot twice. The
  tier carries a running max and running denominator and rescales `sv` by
  `exp(m_old − m_new)` on the keys that raise the max — the same online softmax
  `attn_flash_partial.comp` already uses.
- **`subgroupAdd` + a batch, not a 7-deep tree per key.** The scalar reduction
  is a 128-lane shared-memory tree costing 9 `barrier()`s per key per pass, 18
  per key in total. The tier scores `MLA_SG_KB` keys per barrier (a
  `subgroupAdd` each, then one ping-ponged cross-subgroup fixup), so it pays ONE
  barrier per 8 keys — and the batch's V contributions sum in a register,
  turning 8 read-modify-writes of `sv` into one.

`MLA_SG_KB` = 8 was picked by sweeping it, everything else held (3 reps each):

| `MLA_SG_KB` | 1     | 2     | 4     | 6     | 8      | 16     |
| ----------- | ----- | ----- | ----- | ----- | ------ | ------ |
| pp2048      | 839.8 | 904.9 | 936.1 | 947.8 | 852.7¹ | 834.9¹ |
| tg64 @ 2048 | 10.7  | 12.1  | 13.2  | —     | 11.8¹  | 11.6¹  |

¹ the 8/16 columns are the SHIPPED (re-reading) form; 1–6 were measured on the
held-K form below, so the two halves of that row are not directly comparable —
they are here for the shape, and the shipped column is the A/B table above.

**Still open, and now the whole remaining gap: occupancy at decode.**
`INFR_PROF_OPS` at `-d 2048` (8 tokens, `INFR_SEAM_NO_REPLAY=1`) puts
`mla_sg_ff` at **94.9%** of decode GPU time — 621.2 ms of 654.4 ms, 2.9 ms per
layer dispatch, down from 5.6 ms and 97.3% before. At prefill (`-p 2048`) it is
**86.8%**, 2.08 s of 2.40 s, 38.5 ms per dispatch, from 50.1 ms and 89.4%. So
the kernel is still the frame, and the shape that has NOT been touched is the
one the original entry named first: **one workgroup per (row, head)**, i.e.
`n_head` = 16 workgroups on a 96-CU part at decode, ~83% of the device idle.
`Op::Attention`'s split-K decomposition (`attn_partial` + `attn_combine`, chunk
policy in `infr_core::tier::adaptive_chunk`, partials pooled as
`split_pm`/`split_pl`/`split_pacc`) is the pattern to copy; `Op::Mla` has no arm
of it. Note prefill CANNOT be an occupancy story — 16384 workgroups there — so a
split-K arm buys the decode rows only.

**Measured and NOT shipped: holding the fetched K across the barrier.** V is the
leading `kv_lora_rank` columns of the same K row, so the V accumulation re-reads
columns the score loop already fetched. Holding them in registers
(`kk[MLA_SG_KB][MLA_SG_VSLOTS]`) was implemented and benched at `MLA_SG_KB` = 8:
pp512 857.2, pp2048 955.3, tg64 130.9, tg64 @ 2048 13.8 — i.e. **+12% pp2048 and
+17% tg64@2048 against the shipped form, but −18% pp512**, a regression against
the kernel it replaces on the short-prompt row. Shipped the form that regresses
nothing and recorded the trade here rather than defending a 12% prefill loss.

The mechanism is NOT understood, and the obvious hypothesis is RULED OUT:
`RADV_DEBUG=shaderstats` reports **96 VGPRs and 0 scratch for BOTH forms** (the
mla build is the largest compute shader in the dump — code size 18548 held vs
14692 re-reading), so occupancy per SIMD is identical and register pressure
cannot be the explanation. Trimming the provably-dead fifth held slot moved
pp512 by 3 t/s (857.2 vs 854), which also says registers are not the lever.
Whoever picks this up: the two forms differ only in the V loop, so a selection
on `rows`/`kv_len` (the way `flash_min_rows` and the `rows >= 64` gates already
work for `Op::Attention`) would take both wins — but find the mechanism first.

**Not tried, and the next cheap idea:** `kread` issues one dword load per column
and throws half of it away — the f16 cache packs two adjacent columns per
`uint`. Giving each lane a PAIR of adjacent columns instead of a stride-128
single would halve the load instruction count outright. `key_len` is already
required to be even (`kread` addresses the row as u32 pairs), so the shape
allows it.

- **`Op::Mla` and `Op::LightningIndexer` each disqualify a graph from
  record-once decode replay — and that costs almost nothing.** The
  decode-eligibility loop in `adapter.rs` returns false on either, for stated
  reasons: MLA uses a non-standard kernel, and the indexer bakes its causal
  bound from the `pos` push constant at record time with no params-driven `_dyn`
  twin, so a replayed tape would select keys for the position it was recorded
  at. So every DeepSeek decode token does rebuild and re-record its graph — but
  `INFR_PROF_STAGES` prices that rebuild at **0.039 ms/tok of an 8.9 ms/tok**
  decode step, 0.44%, both from the same run. A separate `INFR_PROF_OPS` run
  puts GPU at 8.2 ms/tok, which INFERS (across runs, so the profiler's own
  overhead is in it) that record+submit is the only other ~0.7 ms and a replay
  tape's whole ceiling is under a tenth of the step. Do the kernel tiers first;
  this is not the DeepSeek decode problem.

### B-DSV2-ACTIVATION-PRECISION — int8 activations are not a DeepSeek defect, and landing f16 for `cfg.deepseek2` was declined (2026-08-11, closed 2026-08-12)

**Tag:** deepseek · reference · **Blocked on:** nothing; this is a decision
record and a ruled-out list, not a defect

This entry used to be B-DSV2-DEGENERATE — "DeepSeek-V2-Lite output decays into
repetition on Vulkan". **That symptom is fixed** and its cause was not
activation precision: `MixerW::Mla` in `seam/runner.rs` pushed its own
`Op::Add{a: hidden, b: sub, dst: hidden}` AND fell through to the shared
post-mixer residual add, so the attention output entered the residual stream
TWICE on every DeepSeek2 layer, on both backends. Removed 2026-08-12; the
generation that named this entry now reads

```
The sky appears blue because of a phenomenon called Rayleigh scattering. When
sunlight enters the Earth's atmosphere, it collides with molecules and small
particles in the air. …
```

on Vulkan (`--temp 0 --max-new 60`, `Why is the sky blue?`), with the CPU run
differing in one word ("process" for "phenomenon") — a near-tie flip, not
degeneration. `gpu_seam_matches_cpu_deepseek2` measures the whole-vocab
CPU/Vulkan cosine at **0.99861** on the 55-token sky paragraph, against the
0.9385 recorded here at 56 tokens before the fix, and that prompt + a 0.99 floor
is now what the test carries.

What is worth keeping is everything that was ruled OUT on the way, each by
measurement, plus one decision.

#### The activation-precision lead is DEAD — do not re-open it

The reading that the Vulkan `Op::Linear` int8 activation prepass (`quant_q8`,
per-32-block symmetric) was the error rested on a layer-0 isolation showing the
int8 route at 6.423e-3 relative L2 against an f64 oracle where the
f32-activation GEMV is 9.081e-8. That isolation is correct about the MATMUL and
wrong about the CONCLUSION. Two independent measurements killed it, both taken
while the doubled residual was still in the tree:

**1. Taking every activation off int8 did not fix the output, and barely moved
the agreement.** Whole-vocab CPU-vs-Vulkan logits cosine on llama.cpp's own 56
token ids for the sky-scattering paragraph, one model load per row:

| activation route                                                     | cos(cpu, vk) | pp512     | tg64  |
| -------------------------------------------------------------------- | -----------: | --------- | ----- |
| int8 `quant_q8` + dp4a mmq (shipping default)                        |       0.9385 | 977.6 t/s | 111.5 |
| f16 (`INFR_NO_MMQ=1` → coopmat GEMM, f16 A staging)                  |       0.9503 | 968.0 t/s | 111.6 |
| f32 (`INFR_NO_COOPMAT=1 INFR_NO_MMQ_FALLBACK=1 INFR_MOE_SMALL_M=64`) |       0.9625 | 708.6 t/s | 111.2 |

All three still degenerated end to end. The perf column is the honest price of
each: f16 costs **1.0% of pp512**, f32 costs **27.5%** — the latter reproducing
the −28% the reverted `exact_linear_acts` flag measured, which is the
corroboration that this row really is the all-f32-activation route.

Coverage was checked rather than assumed. `mmq` off removes the only int8
dense-Linear arm a Q4_K prefill can take on this box (`nc_mmq` needs
`!f16_coopmat()`, which is false on RADV); `moe_small_m = 64` routes the whole
56-row MoE onto `linear_native_id_multi`'s native-block f32 GEMVs instead of
`quant_q8_gather` + `matmul_mmq_experts`; the m=1 decode GEMV was already
`linear_native` on AMD (`mmv_mw_choice` returns `None`, `mmv_decode` defaults
off), which is why tg64 is flat across all three rows. `gemm_warp` off on top of
the f16 row changed the output BIT-IDENTICALLY (cos 0.951823 twice), i.e. no
warp-tile shape was in play to begin with.

**2. llama.cpp's Vulkan backend uses the SAME int8 activation prepass here.**
Read at pin 030ebb5. `ggml_vk_mul_mat_q_f16`
(`ggml/src/ggml-vulkan/ggml-vulkan.cpp:9172`) sets `quantize_y` from
`device->integer_dot_product && src1 is contiguous f32 && (ne11*ne10)%4==0` and
then tries the mmq pipeline FIRST, with no perf heuristic — it is demoted only
if no q8_1 pipeline exists for the dtype, and Q4_K has one. `quantize_q8_1.comp`
is a per-32-block symmetric `d = amax/127` quantizer, and `mul_mmq_funcs.glsl`'s
Q4_K arm is `dotPacked4x8EXT` integer dot. The gate is
`VK_KHR_shader_integer_dot_product` +
`integerDotProduct4x8BitPackedSignedAccelerated`, on by default (kill switch:
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`), which RADV satisfies. Decode is the same
prepass behind a per-vendor `k` threshold (`ggml_vk_should_use_mmvq`: AMD uses
int8 once `k >= 2048`). The one architecture that never gets int8 activations at
prefill is coopmat2 (NVIDIA `VK_NV_cooperative_matrix2`), which force-converts
src1 to f16 at `ggml-vulkan.cpp:9163` and builds no q8_1 pipelines at all.

So **infr's int8-activation default is not a divergence from upstream**: the
coherent reference implementation reduces activations exactly the same way on
exactly this hardware.

#### Also ruled OUT by measurement — do not re-walk these

- **`mla.comp` is not a defect.** Two independent checks. (1) A DeepSeek-sized
  sweep (`n_head` 16, `kv_lora_rank` 512, `qk_nope_dim` 128, `qk_rope_dim` 64,
  `v_head_dim` 128) over `kv_len` 2…2048, with and without the `-DFREQ_FACTORS`
  build, against an f64 reference: max relative error 6e-7…4e-6. (2) The REAL
  layer-0 `Op::Mla` inputs (`q`, the f16 KV rows, `wk_b`, `wv_b`,
  `freq_factors`, `rows`=59, `kv_len`=59) captured out of the CPU backend's own
  arm and replayed through `Recorder::mla`: per-row cosine 1.00000000 against
  the CPU's output, max absolute error 1.8e-7.
- **Not the YaRN trunk scale, not the rope type, not the freq_factors ramp** —
  every constant read out of llama.cpp at the pinned revision and compared to
  what infr dispatches: `kq_scale` 0.11472137 on both,
  `llama_model_rope_type(LLM_ARCH_DEEPSEEK2)` is `LLAMA_ROPE_TYPE_NORM`
  (`neox: false`), and infr's `yarn_ff` divisors reproduce `rope_yarn_ramp`
  exactly (p=0 → 1.0, p=11 → 1.081081, p≥23 → 40.0).
- **Not the in-shader Q4_K WEIGHT dequant.** `native_decode.glsl`'s `dqblk` was
  read out of the device for EVERY element of the real
  `blk.0.attn_kv_a_mqa.weight` (Q4_K, 2048→576) by dispatching
  `native_gemv.comp` with one-hot activations: 1179648 elements, **0 not
  bit-identical** to `infr_gguf::dequant::dequant_block`.
- **Not `Op::EmbedGather`'s dequant of `token_embd`** (Q4_K, 2048×102400): 64
  rows spread across the vocabulary gathered on the GPU, 131072 elements, **0
  not bit-identical**.
- **Not `Op::RmsNorm` on `attn_norm`**: real weight, real embedding rows, eps
  from the file, against an f64 reference — max relative error 2.365e-7
  (256-thread prefill build) / 1.191e-7 (`-DWIDE` decode build).

All three dequant/norm results are guarded by
`crates/infr-vulkan/tests/weight_dequant_parity.rs`, and each assertion was
shown to go red before being trusted.

#### Considered and declined

**Landing the f16 activation route for `cfg.deepseek2`.** It is nearly free
(1.0% of pp512, measured above, vs 27.5% for f32) and its matmul really is more
accurate than int8. Declined: it did not fix the reported bug (the residual
did), its CPU-agreement gain was not even monotone across prompt lengths (worse
than int8 at len 16 and 32, better at 56), and upstream ships int8 on this exact
path. The lever already exists as `INFR_NO_MMQ` if a later slice wants to A/B it
again — nothing needs to be built.

#### Coverage gaps, stated plainly

- The f32-activation row above also drops coopmat from the ATTENTION kernels
  (`INFR_NO_COOPMAT` disables the device feature wholesale), so it is not a pure
  activation-precision A/B. Its pp512 matching the earlier `exact_linear_acts`
  measurement to within a point is what makes it trustworthy as one anyway.
- The Metal MLA path was not looked at at all, and the residual fix was verified
  on CPU and Vulkan only. Metal shares the same `seam/runner.rs` emit, so it
  gets the fix by construction, but nothing was RUN there.
- Whether V3/V3.2/V4 were affected is **untested**. They share the `MixerW::Mla`
  arm, so they were, but the 671B V3.2 has no llama.cpp oracle on this box (it
  cannot load there). `cpu_deepseek_v32_golden` was re-blessed afterwards (its
  comment records the 2026-08-12 re-bless and the text it now generates), but it
  locks what infr does, not what is correct — see B102.

### B-DSHW-FUSED-REF — upstream's fused DeepSeek kernels exist on CUDA and SYCL only (2026-08-11)

**Tag:** deepseek · reference · **Blocked on:** nothing; this is a design
reference, not a defect

Enumerated by listing each backend's source directory at the new pin: fused
DeepSeek kernels exist as `lightning-indexer` and `dsv4-hc` under
`ggml/src/ggml-cuda/` and `ggml/src/ggml-sycl/` (the SYCL pair is **new in this
pull**), and **nowhere else** — no Vulkan, no Metal, no CPU equivalent, and the
Vulkan shader directory has no indexer or hyper-connection shader at all.

Useful specifics for anyone porting, read directly from
`ggml_cuda_lightning_indexer_supported`: the fused path is refused unless the
indexer head dim is exactly 128 and the head count is 64 or 32, every non-quant
`q`/`k` stride is 16-byte aligned, and `k`'s type is one of F32, BF16, F16,
Q8_0, Q5_1, Q5_0, Q4_1, Q4_0. `dsv4-hc` exposes three entry points —
`ggml_cuda_op_dsv4_hc_comb`, `_pre`, `_post` — matching the
`build_hc_pre`/`build_hc_post` decomposition `infr` already mirrors.

Worth noting `infr`'s `Op::LightningIndexer` is architecturally **ahead** of
upstream on both backends that have it: its doc records that llama.cpp expands
the selected indices back into a `-inf` mask and runs dense attention,
"realising none of the FLOP saving", whereas `infr` emits the `top_k` indices
and leaves the gather open to the consumer. Nothing here suggests changing that.

### B-DSHW-PULL — the llama.cpp reference has moved past the pin the docs name (2026-08-11, updated 2026-11-12)

**Tag:** deepseek · reference · **Blocked on:** nothing

**The checkout is no longer at the pin below.** `~/Projects/mxaddict/llama.cpp`
is now at `b10688-1-g57291f2` (`57291f2644af8c9df0dd8d44395881c5bdcf0ecd`), 331
commits past `030ebb5`. Those commits were NOT re-diffed against
`docs/deepseek.md`; re-read at `57291f2` (read, not diffed) were only
`deepseek4.cpp`'s `load_arch_tensors` shapes — which still size V4 tensors with
`n_embd_head_k()`, layer 0's and so the sliding-window head width (see B52) —
and the hyper-connection stream seed in its graph prologue (`ggml_repeat_4d` of
the token embedding across the `hc_mult` streams, which is what infr does).
Everything below is the record of the earlier pull.

`~/Projects/mxaddict/llama.cpp` was pulled from `b10218-1-gc629da5`
(`c629da565c80b0b17fac6262acdca4d772e745d8`) to `b10356-2-g030ebb5`
(`030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`) — 139 commits, and all four files
`docs/deepseek.md` names as the source of its maths changed.
`docs/deepseek.md`'s pin line was updated to the new SHA in the same pass.

**Re-verified unchanged** (each by extracting the block from both revisions and
diffing a non-empty extraction, not by reading a diffstat):

- The trunk **YaRN pre-scale** in `deepseek2.cpp`'s `graph::graph` —
  `attn_factor_org`, `mscale`, `kq_scale` — byte-identical.
- The **lightning indexer** in `deepseek32.cpp` — every line of the indexer body
  is unchanged; the file's additions are all MTP.
- The **hyper-connection helpers** in `deepseek4.cpp` — `build_hc_pre`,
  `build_hc_post`, `build_hc_head` — 201 lines compared, identical.
- **`build_moe_ffn`'s router bias** in `llama-graph.cpp` — the bias is still
  added to `probs` (the gated/sigmoid output) and not to `logits`, which is the
  semantics `fa6a7a8` fixed on the Vulkan side.

**Changed, and worth knowing:**

- `build_moe_ffn` changed by **exactly one line**: the asymmetric-gate SwiGLU
  clamp arm, previously `arch == LLM_ARCH_DEEPSEEK4`, now also fires for
  `LLM_ARCH_DFLASH` when `dsv4_hc_mult > 0`. The clamp maths itself
  (`ggml_clamp(gate, -INFINITY, limit)` with a symmetric `[-limit, limit]` on
  `up`) is untouched.
- `deepseek4.cpp` now reads `swiglu_clamp_exp` / `swiglu_clamp_shexp` over
  `n_layer_all` rather than `n_layer()` — an array **length** change to cover
  MTP layers, not a change to the clamp.
- `deepseek32.cpp`'s layer-count→model-type switch moved from `case 62` to
  `case 61`, consistent with it now switching on `hparams.n_layer()` (trunk
  only) rather than the all-layers count.
- **A second copy of the YaRN pre-scale now exists** in `deepseek2.cpp`'s new
  `graph_mtp`, and it divides by `n_embd_head_k_mla` where the trunk copy
  divides by `n_embd_head_k`. Flagged because `docs/deepseek.md` treats the YaRN
  block as having one home; anyone re-deriving it from upstream will now find
  two, and should read the trunk one.
- MTP/NextN support landed for **V3.2 and V4** (`graph_mtp` in both
  `deepseek2.cpp` and `deepseek32.cpp`, plus `load_mtp`/`TENSOR_SKIP` loader
  plumbing). `infr`'s MTP work may want to track it; not assessed here.

**Not checked:** everything else in the 139 commits. The Vulkan backend moved
only three commits in that range (a submission-batching fix,
`GATED_LINEAR_ATTN`, and a `topk_moe` fusion extension), so the
capability-probing findings elsewhere in this group are current; no other
subsystem was diffed.

### W2 — GGUF tensor lookup is O(n) per call (CR-N2, perf half)

Re-read 2026-11-12 and declined as it stands: each lookup scans the tensor
names, so a load costs about n² name comparisons — around 10⁶ at n ≈ 10³,
milliseconds against seconds of load. Four more `tensors().iter().find` scans in
infr-llama mean indexing `resolve` alone would not remove the cost. Worth doing
only after measuring total name-lookup time on a model with about 2000 tensors.

### B62 — KV spill comes before the context clamp when overflow is on (2026-08-12, traced 2026-11-12)

**Tag:** vram · kv · **Needs:** a decision

User, 2026-08-12: _"Only spill K/V cache to system ram when absolutely
necessary."_ The intended ladder is SWA → q8 → clamp → stream → KV-overflow
last. Traced in `SeamModel::clamp_default_ctx`: auto-q8 is tried before the
spill, as intended, but with `kv.overflow` on the default window is KEPT and
spilled instead of clamped — so the clamp does not come before the spill.
Options: (a) clamp the auto-sized window and spill only for an explicit `--ctx`,
which matches the instruction; (b) keep today's spill-instead-of-clamp, on the
reading that turning overflow on is asking for the full window; (c) a separate
knob. Whichever is chosen wants a test of the order: spilling is invisible in
output and only shows as throughput.

### B63 — deep-context performance is unmeasured across the whole model matrix (2026-08-12)

**Tag:** perf · coverage-gap · **Blocked on:** nothing

Every performance number this repo records is short-context: pp512 / tg128, or
pp2048 at most (`perf-state-2026-07-20`, the cross-model sweep in
`bench-profile-commands`). **Nothing has ever been measured at agentic-workload
depth.**

User, 2026-08-12: _"benchmark deep context for some models and optimize deep
context, i.e. 35k context prefill + tg512 at depth 35k, so we can see which
parts of infr are slow at those deeper context … I want to be able to run Qwen
3.6 27b locally with larger context for agentic workflow"_, then: _"We should
probably have the same perf sweep for all our supported models."_

**The measurement**, per model: `pp35000`, `tg512 @ depth 35000`, and an
`INFR_PROF_OPS` per-op profile **at depth** — the profile is the deliverable,
not the throughput, because the question is WHICH parts degrade. Score against
llama.cpp's Vulkan backend on the same GGUF wherever it can load it. One size
per family that fits 24 GiB, and always name the quant.

**The matrix** is README's own supported-model table (that table is the
authority; do not re-derive it): `llama`, `llama4`, `qwen2`, `qwen3`,
`qwen3moe`, `gemma3`, `gemma4` (dense / E2B / MoE), `qwen35`, `qwen35moe`,
`diffusion-gemma`, `deepseek`, `deepseek2`/`deepseek32`, and the ternary
`bitnet` / Ternary-Bonsai models.

Two cells are worth more than the rest:

- **SWA models (`gemma3`, and anything with a sliding window) should be FLAT in
  depth by construction** — the window caps the key range, so cost per token
  must not grow with context. If they are not flat, that is a BUG, not a perf
  item, and it is the single most informative measurement in the sweep.
- **`qwen35` / `qwen35moe` are hybrid**: DeltaNet layers are linear in sequence
  length, the attention layers are not, so their depth curve should bend
  differently from every dense model. Nobody has measured it — and this is the
  family the user's Qwen3.6-27B target rides.

**Precedent that this will find something.** DeepSeek V2-Lite decode collapsed
111 → 6.3 t/s between short context and `-d 2048` (34× off llama.cpp), and one
kernel's inner loop was the entire cause — see `B-DSHW-MLA-SCALAR-TIER`, fixed
in fa9dd2c to 11.9 t/s. An identical collapse in any Qwen or Gemma model would
be invisible today, because no measurement in this repo would show it.

Methodology, learned the hard way this session: benches run ONE AT A TIME (a
concurrent GPU process skews everything), arms alternate across reps, and the
spread is reported rather than just the mean.

### B74 — LNK4098 on every Windows link: `esaxx-rs` forces the static CRT (2026-09-23)

**Tag:** Windows review · **Blocked on:** upstream (`tokenizers` /
`toktrie_hf_tokenizers`)

Every Windows link of `infr-cli` prints
`LINK : warning LNK4098: defaultlib 'LIBCMT' conflicts with use of other libs`.
The cause is `esaxx-rs`, whose `build.rs` compiles its C++ with
`.static_crt(true)` while everything else links the dynamic CRT. It arrives
through `tokenizers`' default `esaxx_fast` feature, which only the Unigram
_trainer_ uses — infr never trains a tokenizer. Turning it off in the workspace
manifest does not help: `toktrie_hf_tokenizers` depends on `tokenizers` 0.21
with default features, and feature unification turns `esaxx-rs/cpp` back on.
Harmless so far (no heap object crosses the CRT boundary on the inference path),
but it is a warning a `-D warnings` link policy would fail on. Options: ask
upstream for `default-features = false`, or patch the crate.

### B75 — Windows-specific leftovers from the 2026-09-23 review (2026-09-23)

**Tag:** Windows review · **Blocked on:** nothing; each is small and none is on
a hot path

- **The iGPU's first submit cap costs ~5% decode on small models.** On the Ryzen
  iGPU under Windows, Qwen3-0.6B decodes at 52.5 tok/s with the integrated
  default (`initial_submit_dispatch_cap`, 128 dispatches per submit) and 55.0
  with `INFR_SUBMIT_DISPATCHES=256` or `0`; prefill is unchanged. Left alone:
  the cap guards a large model's first forward against TDR before the adaptive
  splitter has measured anything, and the splitter only ever ratchets down. A
  per-model starting cap would need the same watchdog evidence the current one
  was set from (see [igpu.md](igpu.md)).
- **Declined: keeping an existing blob instead of renaming over it.** The review
  suspected that committing a download over an already-cached blob would fail on
  Windows while another infr has that blob memory-mapped. std's `fs::rename`
  succeeds there (POSIX semantics, and std opens files with
  `FILE_SHARE_DELETE`); `MoveFileExW`, which `persist::commit_file` adopted on
  2026-11-12, does not ("Access is denied", run 2026-11-13). `commit_file` now
  falls back to std's rename plus a flush of the renamed file when `MoveFileExW`
  refuses, and `a_commit_replaces_a_mapped_file` holds the case on every OS.
- **Which `link_blob` arm runs on Windows.** Observed on this dev box: the
  snapshot entry `infr pull` made on 2026-09-23 is a symlink, so the
  `symlink_file` arm ran (Developer Mode grants the privilege). The hard-link
  fallback is what a machine without that privilege takes; neither is logged.
  `windows-smoke` cannot answer it for CI: it restores `~/.cache/huggingface`
  from the Actions cache, so on a hit `link_blob` never runs there. Left
  unlogged on purpose — `infr-plat` has no logging dependency, and both arms
  yield a working entry that `link.rs`'s tests read through.
- **No read-ahead hint at model load.** `infr-gguf`'s `WillNeed` advice is
  `cfg(unix)`; the Windows equivalent is `PrefetchVirtualMemory`. Not measured
  whether load time on Windows suffers for it.
- **`K=turbo2` is far from the CPU on both GPUs.** A throwaway diagnostic
  (2026-11-13) compared `prefill_logits_vulkan` with `prefill_logits_cpu` on
  prefixes of `kv_check_prompt` (40, 70 and 80 tokens): `K=turbo2` with an f16 V
  scored a logit cosine of 0.5 to 0.74 on both GPUs, against 0.94 to 0.99 for
  `V=turbo2` with an f16 K. That is the "K needs more precision than V" rule,
  and `K=turbo2` is not a configuration any test runs; recorded so the number is
  not rediscovered as a GPU bug. (The 7900 XTX failure of the old text-asserting
  Vulkan KV test, which this bullet used to describe, went with B141's numeric
  check: `gpu_seam_kv_formats_track_cpu` passes on both GPUs.)
- **Remaining iGPU benchmark exclusions.** `decode_gemv_bw` still needs
  device-selection preservation, scaled cache-busting allocations, bounded
  repetition submissions and asserted parity. `bandwidth_probe` needs a separate
  small-workload/upload-cache design; `interconnect_probe` remains blocked by
  B69. Do not run these unchanged on a small Windows iGPU. The selected bounded
  command is in [windows.md](windows.md#bounded-igpu-benchmarks-2026-10-09).
- **Benchmark hardware coverage remains limited.** The bounded Windows run did
  not exercise discrete profiles, direct cooperative-matrix kernels,
  shared-memory-ineligible K-tile/DeltaNet variants, Linux GPUs or other
  vendors. One cold expert-GEMM sample exceeded the completed-batch budget and
  stopped without measured throughput; full timing coverage is not claimed.
- **Default-parallel local nextest exceeds available Windows commit capacity.**
  Validation hit allocation failures and `os error 1455` (paging file too
  small), plus a downloader concurrency-floor failure under contention
  (`pull::tests::a_finished_file_hands_over_its_connection` failed again on
  2026-11-13 beside a release build, and passed five times alone). The whole
  workspace passed with `--test-threads=1`; no assertions or system memory
  settings were changed. Revisit parallel resource scheduling separately.
- **Why Windows CPU goldens diverge is unconfirmed.** `cpu_golden_qwen3` and
  `cpu_golden_gemma3`'s 48-token cases produce different, coherent text on
  Windows than on Linux on the same Ryzen, with and without `target-cpu=native`,
  identically run after run; the short cases match bit-for-bit. They carry a
  Windows hash via `per_os`. The C runtime's `f32` transcendentals (UCRT vs
  glibc) are the suspect — confirming it means diffing per-op outputs between
  the two builds. Nothing in CI runs these on Windows (no model cache in the
  `test` job), so the Windows hashes are checked on dev boxes only.

### B76 — the physical-core thread default is Windows-only until Linux is measured (2026-09-23)

**Tag:** Windows optimisation · **Blocked on:** a Linux measurement

With no `-t`, infr now publishes one thread per physical core on Windows
(`infr_plat::cpu::physical_cores`, used by `default_thread_count` in
`infr-cli`), because the CPU backend's spin pool with a worker on both
hyperthreads of every core decoded Qwen3-0.6B at ~65 tok/s against ~127 on the
same Ryzen 9 9950X3D with one per core; prefill improved too, and a GPU run's
end-to-end time did not change. Linux keeps rayon's logical-count default
because nobody has measured it there — the contention mechanism is not
Windows-specific, so it probably applies, but that is a guess.

To close: run `infr bench --dev cpu -p 0 -n 64 -r 5` with `-t` at the physical
and logical counts on Linux. If physical wins, give `physical_cores` a Linux arm
(count unique `core_id`s per `physical_package_id` under
`/sys/devices/system/cpu`, restricted to the process's affinity mask) rather
than widening the Windows one. macOS would need the same measurement and
`sysctl hw.physicalcpu`.

### B77 — what the 2026-09-23 Windows session did not verify (2026-09-23)

**Tag:** Windows session residual · **Blocked on:** hardware or a way to trigger
it; nothing here is known broken

Coverage gaps, stated plainly:

- **The console close / logoff / shutdown path of the Windows Ctrl-C handler.**
  Ctrl-C was verified end to end (a real `CTRL_C_EVENT` sent to an idle
  `infr run`, a generating `infr run` and an `infr serve`: each exited 130 after
  draining). `CTRL_CLOSE_EVENT` cannot be generated programmatically, so the
  branch that parks the handler thread until `main` drains is covered only by
  the unit test's event-to-signal mapping. Closing a console window during a
  long generation, and checking the process exits 143, is the missing check.
- **The f16 rounding probe off AMD's Windows driver.**
  `VulkanBackend::arm_f16_rounding` armed on the RDNA2 iGPU and stayed off on
  the RDNA3 7900 XTX under Windows. On Linux RADV it stays off on the RDNA2 iGPU
  (no arm line in an `infr run --dev vulkan1`, run 2026-11-12), matching the
  function's doc. NVIDIA and Intel were not run, so which of those default to
  truncation is unknown — hardware this project does not have. The probe decides
  for itself, so a wrong guess costs nothing, but no patched module has ever
  been compiled by a non-AMD driver.
- **Broader Q4_K performance and portability coverage.** The packed-word
  `native_gemm_mmq_q4k.comp` optimization has only been measured on cached
  Qwen3-0.6B Q4_K_M with the Windows AMD RDNA2 iGPU. Other small models,
  drivers/vendors and Linux GPU execution remain unverified. The entire ignored
  GPU suite was deliberately not run because its dGPU-sized probes can exceed
  the iGPU watchdog; selected parity suites do not cover every tile variant.
  Decode results were mixed, so no decode improvement is established. See
  [windows.md](windows.md#packed-q4_k-follow-up-2026-10-08) for measurements.
- **Windows memory-budget scope.** `mem::windows_job::probe` queries the
  immediate job; tighter ancestor jobs remain unaccounted for. Nested-job
  budgeting needs a separate design and regression test. Whether host budgeting
  should also clamp `ullAvailPhys` by `ullAvailPageFile` remains unverified.
  Application Verifier required elevation and did not run; native job tests
  passed under AddressSanitizer instead, which does not validate every Win32
  handle/API contract.
- **Emoji through console input.** Typed input round-tripped `café`; an emoji
  came back as `�` in the screen scrape, but conhost's screen buffer cannot draw
  astral characters either, so whether the bytes the model received were right
  was not established.

A structural problem noticed, not addressed (out of scope for the session):
`crates/infr-vulkan/build.rs` (3,708 lines, most of it one shader build list)
and `crates/infr-vulkan/src/lib.rs` are monoliths, and `lib.rs` is still
growing. The build list would split naturally by kernel family into data files
or per-family modules; `lib.rs` along its existing sections (device creation,
probes, allocation).

### B70 — no MoE model above 256 experts has ever been run (2026-09-03)

**Tag:** Vulkan MoE coverage · **Blocked on:** nothing; a gap, stated so the B67
fix is not mistaken for more than it is

B67's out-of-bounds `ssel_adj` write is fixed and the ceiling is now enforced
(`MOE_TOPK_MAX_EXPERTS` in `infr-vulkan/src/recorder.rs`, refused loudly; the
shader sizes the array from its own scan headroom, and
`moe_topk_shader_ceiling_matches_host` fails if the two ever drift). What that
does NOT establish is that a >256-expert model routes CORRECTLY: the OOB was
read off the code and never observed, and no such model has been run here or
anywhere in this tree. The two Qwen3.8 MoE models (512 experts — see
[qwen38.md](qwen38.md)) are the first that would exercise it, and neither has
been pulled.

So the guard is verified and the fix is not. The check to run when one of those
models is available is expert-routing parity against the CPU router, which is
`Vec`-based and was never affected.

### B68 — the Metal backend is gated on the host OS, not on itself (2026-09-02)

**Tag:** infr-plat residual · **Blocked on:** a decision between the two options
below

`docs/infr-plat.md` called this Problem B and deliberately scoped it out: it is
a feature-gating question, not a platform-plumbing one, and moving the gates
into a crate would relocate them rather than remove them. The platform seam has
landed without touching it, so the analysis is recorded here.

The `#[cfg(target_os = "macos")]` sites — most of the remaining ones in the tree
— do not mean "spell this differently here". They mean "the Metal backend exists
here": conditional re-exports in `infr-llama/src/chat/mod.rs`, a
`metal_chat_model` constructor, and errors like `chat/cpu.rs`'s _"the Metal
backend is only available on macOS"_. The consequence is that on a Mac the
"Metal unavailable" path cannot be compiled at all, and everywhere else the
"available" path cannot — each arm is built by exactly one runner and neither by
both.

Two facts settle the design, both verified by experiment rather than recalled:

1. **A build script cannot set a Cargo feature.** Feature resolution finishes
   before any build script runs, so `cargo:rustc-cfg=metal` injects
   `--cfg metal` and never `--cfg feature="metal"`. The two predicates are
   disconnected.
2. **Cargo CAN enable a dependency's feature per target**, via
   `[target.'cfg(..)'.dependencies] dep = { features = [..] }`. What it cannot
   do is let a crate enable its OWN feature per target — which is what consumer
   code would need to keep today's "on by default on macOS" behaviour.

So:

- **Option A — one build-script `cfg(metal)`, used uniformly.** No Cargo
  feature. Every crate gating on Metal emits `cfg(metal)` from its `build.rs`
  when `CARGO_CFG_TARGET_OS == "macos"` unless an opt-out env var is set. This
  matches in-tree precedent: `infr-core`, `infr-cpu`, `infr-gguf` and
  `infr-llama` already carry byte-identical build scripts of that shape for
  `INFR_PROFILE` → `cfg(infr_profile)`. Cost is two new `build.rs` files.
- **Option B — a real Cargo feature end-to-end.** Discoverable via `--features`,
  no build script, but consumer code still cannot key on its own feature per
  target, so only `infr-metal` itself becomes feature-gated and about 55 gates
  stay exactly as they are. It does not solve the problem.

Either way, guard the confusing failure: today selecting Metal off macOS fails
cleanly at runtime with a sentence, and if a feature can be forced on for a
non-Apple target the build instead dies inside unresolved `metal`/`objc`
imports. What this buys is that both arms become compilable on macOS; what it
does not buy is cross-building Metal from Linux, since `metal`/`objc` link the
Objective-C runtime.

### B69 — cross-process Vulkan sharing is POSIX-fd-only, with a stub on Windows (2026-09-02)

**Tag:** infr-plat residual · **Blocked on:** nothing but the work itself

`infr-vulkan`'s `p2p.rs` and `tp_sem.rs` export device memory and semaphores
between processes by duplicating POSIX file descriptors — `vkGetMemoryFdKHR`,
`libc::dup`. It looks like it has a Windows arm and does not: `p2p.rs` declares
`#[cfg(target_os = "windows")] type RawFd = std::os::raw::c_int`, which exists
only to keep `P2pExport`'s field type-checking. There is no `HANDLE` and no
`VK_KHR_external_memory_win32` anywhere in the tree, so every call site is
POSIX-only and the multi-GPU sharing path does not work on Windows.

This is why `infr-vulkan` is the one crate allow-listed to depend on `libc` in
`infr-plat`'s `platform_seam` test. Hoisting the stub into the seam would
launder it into an abstraction; implementing `external_memory_win32` is feature
work, not a move. Either do that, or delete the Windows alias in favour of a
refusal that says so.

### B78 — release channels still open (2026-10-27)

**Tag:** delivery

`ci.yml` packages `infr` for x86_64/aarch64 Linux (zigbuild, glibc 2.28), x86_64
Windows and both macOS arches in the `dist` matrix on every `main` push. On a
`v*` tag, `publish-github-release` attaches the archives, `SHA256SUMS` and a
completions plus man page tarball, and then `publish-crates`, `aur-bin`,
`brew-tap` and `scoop-bucket` publish from `pkg/*.in` (decided 2026-10-29:
crates.io, AUR, Homebrew and Scoop). Their secrets are kryptic-sh organisation
secrets visible to every repo. The shipped binaries are built with
`CARGO_PROFILE_RELEASE_STRIP=debuginfo`, so profiling a release build with line
attribution means building it locally. What is NOT done:

- **Verified on `v0.2.0` (2026-10-29), the first tag to run the publish jobs:**
  all 14 crates are on crates.io at 0.2.0, `infr-bin` 0.2.0-1 is on the AUR, and
  the tap's `Formula/infr.rb` and the bucket's `bucket/infr.json` carry 0.2.0,
  each with the checksums in the release's `SHA256SUMS`. Not checked: an actual
  `brew install`, `yay -S infr-bin`, `scoop install infr` or
  `cargo install infr-cli` on a clean machine.
- **`cargo install infr-cli` needs `glslc`**, like any source build:
  `infr-vulkan`'s build script compiles the shaders and has no prebuilt SPIR-V
  fallback.
- **The tap and bucket READMEs** (`kryptic-sh/homebrew-tap`,
  `kryptic-sh/scoop-bucket`) don't list infr; they were already behind on hrdr
  and the others.
- **`.deb` / `.rpm` and Alpine.** hrdr builds them with `cargo-deb` /
  `cargo-generate-rpm` and an `APKBUILD`; infr has no `[package.metadata.deb]`
  or `generate-rpm` section in `crates/infr-cli/Cargo.toml`, and Alpine is musl
  (see the next point).
- **No musl targets, by design:** ash loads `libvulkan` with `dlopen`, which a
  static musl binary cannot do.
- **Not verified:** that the glibc-2.28 Linux binary actually starts on an old
  distro. The smoke step runs `--version` on ubuntu-26.04 only, and aarch64
  Linux is never executed at all (it is cross-built).

### B79 — a GUI for infr: declined for now, service first (2026-10-28)

**Tag:** product · **Decision:** the user's, 2026-10-28 — focus on infr as a
service

Considered and declined, recorded so it is not re-argued from scratch. Any chat
GUI that speaks the OpenAI API (Open WebUI, Jan, Chatbox, Msty) already works
against `infr serve` through `/v1/chat/completions` and `/v1/models`, so an infr
chat app mostly duplicates them. A native desktop app (Tauri, egui) would add a
new dependency tree, per-OS installers, code signing and notarization, and its
own CI, all competing with engine and backend work.

If it is reopened, the cheap shape is a status page compiled into the binary
(`include_str!`) and served at `/` by the existing axum router in
`crates/infr-server`, like llama-server's built-in UI, reading `GET /status` and
`GET /metrics` (built 2026-10-29 for operating the service; see B80). Model
management (pull with progress, cache listing, a fit estimate per quant) would
come after that, and a native wrapper only if there were demand for a tray app.

### B80 — what `/status` and `/metrics` do not report yet (2026-10-29)

**Tag:** serve · **Blocked on:** nothing; scoped out of the first slice

The two routes (`crates/infr-server/src/status.rs`, `metrics.rs`) shipped with
these gaps:

- **Only `ParallelGenerator` reports a device.** `SeamGenerator` (CPU, Metal,
  diffusion-gemma, and anything else served through `build_chat_model`) reports
  its backend label and `null` device, context and memory, because that
  information sits behind the `ChatModel` mutex a generation holds and
  `ChatModel` has no accessor for it. Fixing it means capturing the figures at
  construction (context, device name) and adding a lock-free memory query: host
  RAM through `infr_plat::mem::available()` for CPU, the Metal working-set
  budget for Metal.
- **No latency histograms.** vLLM and TGI export time-to-first-token and
  end-to-end latency histograms; `/metrics` has only the prefill and decode
  second totals, which give mean rates but no tail. `ReqRecord` already carries
  `prefill` and `total`, so a fixed-bucket histogram is a matter of atomics per
  bucket.
- **Counters are server-wide, not per model.** `ServeStats` is shared by every
  hosted model, so under `infr multi` the token and request counters cannot be
  split by model; only the slot, context and memory gauges carry a `model`
  label.
- **Prompt tokens mean "prefilled".** `infr_prompt_tokens_total` sums
  `ChatOutcome::prompt_tokens`, which on the Vulkan seam is the uncached suffix
  (`GenStats::n_prompt`), so a request that reuses a cached prefix adds only its
  new tokens. Seen live on the iGPU: a repeat of a 10-token prompt added 1. A
  separate counter for the whole prompt would need the generator to report it.
- **A failed request's partial tokens are not counted** in the lifetime counters
  (they have no authoritative `ChatOutcome`); the periodic log line does count
  them, live.
- **Not verified:** a real Prometheus scrape. The format was checked by the
  route tests and by reading a live `/metrics` body, not by a Prometheus or
  `promtool check metrics` run (neither is installed here).

## Whole-codebase correctness review 2026-08-30

### Decision — a deadline that fires during prefill returns HTTP 500

Since the prefill fix, a `serve.request_timeout_secs` deadline that fires before
the first token makes `run_chat` fail with
`aborted: request cancelled during prompt prefill`. The non-streaming handler
answers every generator error with `500 server_error`, and the streaming one
sends an error frame. A deadline hit AFTER output started is a `200` with
`finish_reason: "length"`. Verified on CPU with Qwen3-0.6B (2026-10-27).

Options: keep the 500 (honest: nothing was produced), or map
`deadline_hit && Err` to `504 Gateway Timeout` / a `timeout` error type so
clients can retry it differently from a crash. Needs the user's call on the
status code; the change itself is a few lines in `infr-server`'s two chat
handlers.

### Cleared

- Empty-logit sampling does not expose a reachable empty-vocabulary generation
  path; its public helper's token-zero result remains internal behavior.
- DSV4 compressor zero-dimension arithmetic is currently unreachable bring-up
  code, not a production defect.
- Ordinary-attention `key_bias` is retained because incompatible flash/dynamic
  paths are rejected and biased attention is routed through consuming kernels.
- Sampled GGUF shard and tensor offsets use checked arithmetic and validate
  alignment and mapped ranges before slicing.
- Examined stop matching, SSE terminal framing, and completion accounting retain
  their stated token-boundary invariants.
- Unknown model IDs intentionally route to the documented default hosted model.

### Hardening

- `download_to_blob` trusts an existing expected-hash blob because its
  content-addressed filename exists; rehashing would detect external cache
  corruption, but no repository-controlled invariant break was found.
- DSV4 compressor-plan constructors should reject zero ratio/state dimensions if
  that bring-up path becomes reachable.
- Keep the non-empty distribution contract explicit in sampling internals rather
  than relying on current callers.

### Coverage

Reviewed the clean working tree across workspace entry points, public APIs,
unchecked arithmetic/indexing, panic and error paths, cancellation, HTTP range
handling, unsafe/FFI boundaries, and sampled CPU, Vulkan, Metal, GGUF, server,
CLI, hub, chat, profiling, and llama paths.

**Gaps:** most numerical kernels and shader math were not reviewed line by line;
platform-only Metal and Windows behavior was not executed. No formatter, linter,
build, or test command was run for this report.

### B81 — minijinja 2.24 prints booleans as `True` / `False` (2026-10-29)

**Tag:** chat templates · **Needs:** a decision

Regenerating `Cargo.lock` for v0.2.0 moved `minijinja` and `minijinja-contrib`
from 2.21.0 to 2.24.0, and `{{ enable_thinking }}` started rendering as `True`
instead of `true`: `template_tests::no_think_config_drives_enable_thinking`
failed. 2.24 writes a bool the way Python's Jinja2 does (`value/mod.rs`,
`utils.rs`), so any chat template that prints a bool, rather than testing one,
changes its prompt text. Both crates are held at 2.21.0 in `Cargo.lock` only
(`cargo update -p minijinja --precise 2.21.0`); the manifest still says `"2"`,
so the next `cargo generate-lockfile` or `cargo update` brings 2.24 back.

To decide: take 2.24 (it is probably closer to HF transformers, which renders
these templates with Python's Jinja2; not verified against a real template that
prints a bool) and change the test's expectation, or pin `~2.21` in
`[workspace.dependencies]` until someone checks what HF and llama.cpp's minja
actually emit.

Checked 2026-11-12: llama.cpp's Jinja prints a bool as `True`
(`value_bool_t::as_string` in `common/jinja/value.h`, the `tostring` builtin in
`value.cpp`); `tojson` gives `true`. So 2.24 is the one that matches llama.cpp.
A formatter override on 2.21 (`Environment::set_formatter`) covers only `{{ }}`,
not a bool reaching text through `~` or `|string`, so it cannot fully match.
Still the owner's call: take 2.24 and change the test's expectation (changes the
prompt of any template that prints a bool), or pin `~2.21` to keep today's
prompts.

## Whole-codebase review 2026-11-08

Three read-only review passes over `main` @ `f2f89cb` (correctness, performance,
docs/backlog drift). Nothing below is fixed; nothing below was measured. Each
entry says what was traced in the code versus what is inferred. Baseline at the
time: `cargo clippy --workspace --all-targets -- -D warnings` and
`cargo test --workspace` both green on Linux.

### B82 — `infr serve`: what the 2026-11-12 fixes left open

Fixed that day: `finish_reason: "length"` on a context-cut reply, whole-prompt
`usage.prompt_tokens` with `prompt_tokens_details.cached_tokens`, the poisoned
model lock after a panic, cancel on client hang-up,
`400 context_length_exceeded` (and a typed SSE error frame on a stream), 413 /
415 for body rejections, `stream_options.include_usage`, the bearer check before
the body, `-1` for `seed` / `max_tokens`, and the 401 body. Left:

- **A streamed over-long prompt still answers 200.** The headers and role chunk
  go out before `engine.chat` runs, so the client gets a typed error frame, not
  a 400. Returning 400 means tokenizing before the stream opens, as llama.cpp's
  server does; that is the `count_prompt_tokens` method B10a describes, and it
  would also let an over-long prompt be refused before it takes a slot.
- **No body-size key.** The axum default (2 MiB) still applies, undocumented;
  there is no `serve.*` setting for it.
- **`ctx_full` and `natural_stop` are set only on the dense seam paths.** The
  MTP, diffusion and Metal speculative paths report `false` for both, so a
  context-cut reply there still says `stop` and an end token on the last allowed
  position still says `length`. The runner setting `natural_stop` was read, not
  run: it needs a model. `ran_out_of_context` and the CLI's `finish_reason` have
  unit tests.
- **Panic recovery resets session state only.** `lock_recovered` clears the
  poison and calls `reset_kv`; state inside a backend that the panic broke (a
  poisoned lock, a half-recorded submit) is not repaired, and requests reaching
  it keep failing. `DiffusionGemmaChat::reset_kv` drops its whole session and
  has no test (it needs a real GGUF). A non-streaming 500 body still includes
  the panic message.
- **DiffusionGemma under `serve` has no cap on the session it opens.** The
  untyped overflow in `diffusion_generate` is reachable only from `bench`;
  `DiffusionGemmaChat::ensure_sess` always sizes the session to at least
  prompt + blocks × canvas + 64, so an over-long prompt opens an arbitrarily
  large session instead of being refused. Needs a decision on the cap.
- `infr-cpu/src/pool.rs` still sets a process-wide panic hook in two tests.
- The hang-up tests use real sockets with 2 s bounds; a loaded Windows or macOS
  runner could fail them. A failure would be an assertion, not a hang.

### B88 — special-token text in user messages tokenizes as control tokens (2026-11-08)

**Needs:** a decision. A user message containing the text of a control token
(`<|im_start|>`, `<|eot_id|>`, …) is tokenized as that control token, because
the rendered prompt is encoded with special-token parsing on
(`SeamModel::encode`). Checked 2026-11-12: llama.cpp's server does the same —
`handle_completions_impl` tokenizes the rendered chat prompt with
`parse_special=true` (`tokenize_input_prompts` → `tokenize_mixed`) — so infr
already matches the reference, and a user can forge a turn boundary on both.

Options: (a) render with placeholders and tokenize the user segments with
specials off, which diverges from llama.cpp and breaks templates that transform
content (`.strip()`, splitting on `</think>`) by mangling the placeholders; (b)
keep parity and document it as a property of the chat endpoint.

### B90 — chat stream: what the 2026-11-12 rework changed and left (2026-11-12)

`ChatStream` streams a plain JSON answer as soon as its first object closes
(with tools on, a reply starting with `{` used to be held to the end), streams
content that precedes a late `<think>`, and settles every stretch it held at a
tool-call opener: calls fire, reasoning inside the stretch goes out as reasoning
where it sits, and the rest is content, in the order it was written. A lead-in
held at an opener before `<think>` is settled when the `<think>` arrives (its
end is known then), ahead of the reasoning; a stretch in the main content region
is settled at `finish`. Decided on the way, and worth knowing:

- **A tool-call block that does not parse is shown as content**, alone or beside
  a valid call, in both the Hermes and the pipe dialect. A block cut off by
  `max_tokens` appears in the reply. A reply with no parsed call is emitted byte
  for byte (`a_reply_with_no_parsed_call_is_byte_identical`).
- **Whitespace changes only at a removed call.** Where a removed call had
  whitespace on both sides, one gap survives: the side holding a line break (the
  longer if both do), else one space — so a call on its own line does not join
  the paragraphs or list items around it, and `A <call> B` reads `A B`. This is
  in `tools::remove_spans`, so the non-streaming parsers' `clean` text follows
  it too. In the stream, a call that opens a text span is judged against the
  content actually shown so far, and only the stretch's last text span loses a
  trailing gap, at the end of the reply.
- **Mixed dialects fire Hermes calls first, then pipe calls**, not in the order
  they appear in the text; every call in a stretch fires after the stretch's
  text.
- **The cut detection is textual** (`cut_edges` compares the span with the
  parser's output from both ends). A removed call whose neighbouring text
  happens to repeat the characters it shares with the output can be misjudged;
  only whitespace is affected. Not seen in a test.
- The "lead-in first seen during the final flush" path was made consistent with
  streaming but has no test: every `push` runs a non-final pass first, so the
  case cannot be produced today.

### B92 — `top_k == 0` and penalised requests fall off GPU sampling and chaining (2026-11-08)

**Tag:** perf · decode · **Verified:** gate read; cost from the in-code
measurement, not re-measured

The `gpu_sample` gate in `generate_dense_backend` requires
`(2..=Recorder::SAMPLE_KMAX).contains(&top_k)` and `!penalize`; `can_chain`
needs `gpu_argmax || gpu_sample`. So Llama-3's shipped defaults (temp 0.6, top_p
0.9, `top_k = 0`) and any request with repeat/presence/frequency penalties take
the per-token host path: a submit + wait, a full-vocab logits download and
`truncated_softmax` on the host (~0.5 ms at n = 128256 per the doc comment in
`sampling.rs`), no chaining. Estimated 5–10 % of tg on 8B-class, more on small
models.

Options: extend `Op::Sample` to full-vocab top-p (stage 1 emits per-workgroup
max / rescaled sum-exp partials so the denominator is full-vocab; fall back to
the host for the rare token whose top-`SAMPLE_KMAX` set does not reach `top_p`);
a device penalty kernel over a history buffer is a separate, larger slice.
Smaller host-side items in the same file: `Penalties::apply` rebuilds a
`HashSet` of the whole `recent` window per token (O(window) with SipHash); the
`top_k > SAMPLE_KMAX` arm of `truncated_softmax` allocates a vocab-sized
`Vec<usize>` per token for `select_nth_unstable_by`. Measure: `infr bench` with
`--temp 0.6 --top-p 0.9 --top-k 0` vs `--top-k 40`, 0.6B and 8B.

### B93 — serve callbacks run while holding the GPU baton (2026-11-08)

**Tag:** perf · serve `--parallel N` · **Verified:** scope read; cost NOT
measured

In `generate_dense_backend`'s chained-decode loop, `_gp = req.gate_pass()` lives
to the end of the iteration, and the `on_token` calls (detokenize,
`ChatStream::push`, `StopMatcher`, SSE JSON, channel send) run inside it after
`execute_chain` has already returned. Every other slot waits on that host work.
The per-token path has the same shape. Fix: drop the pass once the GPU calls
return, before the callbacks; teardown takes its own pass. Check first that no
callback touches the queue or a command pool (the pass's invariant). Measure:
serve load test with `--parallel 4`, aggregate t/s and inter-token latency.

### B94 — chain k+1 is not submitted until the host has processed chain k (2026-11-08)

**Tag:** perf · decode · **Verified:** PLAUSIBLE design lead

`RecordedCmd::replay_n` (`crates/infr-vulkan/src/recorder.rs`) does
`queue_submit` then `queue_wait_idle`; the runner then reads the id ring, runs
the callbacks and uploads before the next submit, so the GPU idles for that
window every chain (estimated 100–300 µs; 1–3 % of tg). The sampled id and
position already live on the device, so chain k+1 does not depend on the host
reading chain k. Lead: submit k+1 behind a fence before draining k, accept an
overshoot of at most `chain_n` tokens on EOS/stop (rows past `last_written` are
already ignored), and mind the iGPU submit splitter, abort polling and the
baton. Measure with `INFR_PROF_STAGES=1` on a 0.6B tg128.

### B95 — CPU spin-pool atomics share a cache line (2026-11-08)

**Tag:** perf · CPU backend · **Verified:** layout read; effect NOT measured

`Shared` in `crates/infr-cpu/src/pool.rs` packs `seq` (spun on by idle workers),
`cursor` and `done` (hit with `fetch_add` on every task claim / check-in),
`n_tasks`, `panicked` and `shutdown` next to each other with no alignment, so
every claim invalidates every spinner's copy of the line. Fix: put `seq` and the
write-hot counters on separate `#[repr(align(128))]` wrappers — scheduling is
unchanged, outputs stay bit-identical. Estimate 1–3 % of CPU decode on a
many-core box; measure with `samply` / `perf c2c` on `infr bench --ngl 0`.

Same backend, smaller: on the m = 1 path of CPU `Op::Linear`
(`crates/infr-cpu/src/lib.rs`), `quantize_q8` of the activation row runs
serially per projection, so Q/K/V/gate/up re-quantize the same normed input.
Estimated well under 1 %.

### Review coverage gaps (2026-11-08)

What those three passes left unread has since been read or is listed under
"Review coverage as of 2026-11-12" at the end of this file. Cleared by the
correctness pass (do not re-open without new evidence): GGUF header parsing
bounds, hub path validation and resume, server bearer comparison / auth gating /
`StopMatcher` / `[DONE]` guard, sampling seed and penalty indexing, KV
prefix-reuse rules, the template fuel bound, and the `unsafe` in `infr-plat`
lock/signal, `infr-core` `hostpager.rs`, and `infr-cpu` `pool.rs`. Cleared by
the perf pass: compiled-template caching, `ChatStream` / `StopMatcher` cursors
(not quadratic), mapped host-visible `upload`/`download`, lazy `token_embd`
dequant.

### B97 — docs still to condense or restructure (2026-11-08, updated 2026-11-10)

**Tag:** docs · **Needs:** a decision on the plan layout

The 2026-11-10 cleanup retired the shipped parts of the plan docs: `audit.md`
was deleted (its deferrals are B98), and the finished plan / status sections of
`diffusion-gemma.md`, `qwen35.md`, `igpu.md`, `perf/cpu.md`,
`disk-streaming-plan.md` §5 and `plan.md`'s milestone table were cut, their open
items moved to B35/B36 and B99–B101. Still open:

- **`docs/mtp.md`** — MTP is parked and the Phase 1–3 build plan is history, but
  code comments in `crates/infr-llama/src/mtp/` cite "mtp.md Phase 1/2/3" by
  name. Condensing it means rewording those comments in the same change.
- **Plan layout.** The global working rules put plans in `docs/plan/` with a
  `00-index.md` and one `NN-name.md` per phase; this repo has `docs/plan.md`
  plus free-standing plan docs (`deepseek.md`, `qwen38.md`, `train.md`,
  `synthetic-models.md`, `disk-streaming-plan.md`). Options: migrate to
  `docs/plan/` (one index line per campaign, campaign docs moved under it), or
  keep the current layout and say so in `docs/README.md`. Needs the user's call
  — it moves files that code comments and every other doc link to.
- **`docs/plan.md`'s "As built (2026-08-05)" table** is dated and was only
  patched for the DeepSeek line; re-date it when it is next reviewed whole.
- **Code comments not swept:** any comment citing an env var or doc section that
  has moved. (The `arch.rs` DeepSeek comments and `deny.toml`'s wildcard comment
  were corrected on 2026-11-12.)

### B98 — the codebase audit's four deferred findings (2026-07-21, moved 2026-11-10)

**Tag:** audit · **Blocked on:** each is a deliberate deferral, not a blocker

The 2026-07-21 module-by-module audit (`docs/audit.md`, deleted 2026-11-10 —
`git log -- docs/audit.md` has it) closed 153 of 157 findings. These four were
deferred, each because the fix risks the byte-identity goldens or is a mass
migration out of proportion to the defect:

- **`make_compute_kernel` panics on recoverable driver / OOM errors**
  (`crates/infr-vulkan/src/ops.rs`). `kernel` / `kernel_sg` return a bare
  `ComputeKernel`, so a late `OUT_OF_DEVICE_MEMORY` during a kernel compile
  aborts the process. Threading `Result` reaches every recorder dispatch helper
  that returns `()` (ops / gemm / recorder); the panic message names the kernel
  and points here. A full `Result`-ification of the recorder dispatch surface is
  its own effort.
- **`attn_combine.comp` and the flash kernels (perf / DRY).** `attn_combine`
  recomputes `mm` / `l` over all `nch` in every lane; the causal-mask +
  online-softmax rescale block is copy-pasted across the flash kernels; ring
  `RROW` / `rcap` and the Q8 / f16 vec4 readers are duplicated. Deferred because
  `attn_combine` is dispatched without a pinned subgroup size (a subgroup reduce
  would be wrong on a 16-lane device without `requiredSubgroupSize = 32`), the
  `l` reduce is only within-tolerance rather than byte-identical, and an
  include-refactor of the flash mask / rescale would perturb their numerics.
- **The int8 dp4a decode helpers** (`rb` / `ru16` / `f16tof32` / `k4` and the
  per-format `dpsub` / `wdec`) are copy-pasted across the `native_mmv*` and
  `native_gemm_mmq_*` shaders, and the `KV_IQ4NL_W` table is duplicated. A
  shared include has to reconcile per-file conditional helper sets and add
  `build.rs` include wiring; textual drift risks the compiled SPV on the
  byte-identity GEMM / MoE goldens.
- **`qk_norm_rope` vs `_interleaved`, `sample_topk` vs `moe_sample`** look
  duplicated but differ in push-constant and binding layouts (`kcap` + the SWA
  ring modulo; two-stage `PASS1` / `PASS2` / `CHAIN` with a buffer-sourced `u`
  vs single-stage with `u` in the push constant). Folding them means reconciling
  those structs and the `build.rs` compile list, for a DRY gain.

Re-read 2026-11-12: all four are still true and none is a clear pure win; each
needs GPU byte-identity runs. The audit's `partial_cmp(..).unwrap()` theme is
closed — the last production sites (CPU MoE routing, `truncated_softmax`,
`infr-cpu` `sample_token`) use `total_cmp`, so a NaN no longer panics the sort.
It still ranks first and the output is still NaN: nothing downstream handles a
NaN logit or router score.

### B99 — iGPU perf: beat llama.cpp's Vulkan backend on the iGPU (moved 2026-11-10)

**Tag:** iGPU · **Blocked on:** nothing but priority

Correctness on the iGPU is complete (`docs/igpu.md`); perf was never started. In
the order the iGPU campaign left them (not re-measured since):

1. **The no-coopmat prefill tier is the whole gap** — pp 11–200 tok/s against
   thousands on the dGPU. RDNA2 has no cooperative matrix, so prefill falls to
   the scalar / f16-warp ladder. The dominant lever.
2. **Re-audit the UMA placement for headroom.** The UMA slice (`4771cda`) made
   `unified_memory` real in the Vulkan caps, skipped the pointless host →
   staging → device copy, and let the VRAM guard treat GTT as system RAM; the
   campaign asked for a re-audit of the headroom it leaves, which was not done.
3. **Revisit `INFR_UBATCH` on the iGPU.** Rows barely affect submit time now
   that submits are duration-bounded, so the 128-row prefill cap is likely
   costing throughput — do a FLOP-aware budget (llama.cpp's shape) before
   raising it.
4. **The default-ctx clamp collapsing to 1024** on many models — recorded as a
   wart during the survey; not re-checked against the current fit ladder (B12).

Operational rule from `docs/igpu.md` applies to every item: never let a timeout
kill `infr` mid-submit on the iGPU, and run GPU work serially.

### B100 — DiffusionGemma on Metal: unvalidated on hardware, MoE on the host path (moved 2026-11-10)

**Tag:** metal · DiffusionGemma · **Blocked on:** Apple hardware with a DG model

Phase D (Metal denoise) is built and its kernel-level checks run in the macOS CI
job (`cargo test -p infr-metal -- --include-ignored`: MSL compile, the
`attention_canvas_split*` parity tests against the CPU reference,
`kernel_names`). What has never happened is a real DG model on an Apple GPU:

- `infr run --dev metal <dg-gguf>` producing coherent text, including across
  several self-conditioned steps — the first dispatch of `softmax_wide_f32` and
  `sc_embT`'s dequant-to-f32 `Linear` at the real `[canvas, vocab]` shape.
- `infr bench --dev metal <dg-gguf> -p 128 -n 256` giving a sane rate, ideally
  beside the Vulkan figure on the same machine.
- On a memory-constrained device, telling the `weight_buf` "would exceed the GPU
  working set" error (`sc_embT` cached as f32, ~2x its f16 size) apart from a
  correctness bug.

Separately, **DG's MoE runs on Metal's host path**: the device MoE arm in
`exec.rs` (`device_ok`) requires separate gate / up banks and no per-expert down
scale, and DG has a fused `gate_up_exps` plus `ffn_down_exps.scale`. A device
path for that layout was judged too large to write blind. Metal's canvas
attention also always takes the split-KV canvas kernel, never the flash / vec
tiers, so its denoise rate is a conservative lower bound until profiled.

### B101 — CPU backend perf: what the CPU worklist left open (moved 2026-11-10)

**Tag:** perf · CPU backend · **Blocked on:** measurement for item 2

From `docs/perf/cpu.md`'s worklist, whose quant-coverage slices all landed:

1. **Native AVX-512-FP16 / BF16 dots** for f16 / bf16 weights. They already read
   native 2-byte (bandwidth minimal) but accumulate in f32 after widening; a
   native dot is a compute-only, prefill-leaning win. Changes accumulation
   precision → tolerance parity and a re-bless if the f16 / bf16 goldens move.
2. **Prefill tile tuning to the X3D topology** — size the prefill GEMM tile so
   the activation tile stays in L1 / L2 while weights stream; the current 8-row
   tile is a fixed heuristic. Bit-identical. Gated on `perf stat` (LLC / dTLB /
   backend-stall) showing cache-miss slack; `perf` is not installed on the dev
   box.
3. **Op fusion** beyond `GatedActFused` / `RmsNormAdd`: norm → linear and
   residual chains, so intermediate activations never round-trip to DRAM.
   Bit-identical if each fused op keeps the same values and order; verify per
   fusion.
4. **A VNNI tier for IQ4_XS.** `vec_dot_iq4xs` and `vec_dot_iq4xs_batch` in
   `crates/infr-cpu/src/kernels.rs` stop at AVX-512BW / AVX2; the other int8
   formats have VNNI. (VNNI cannot be exercised on any dev or CI box — B3.)

Declined, kept so it is not re-proposed: **DeltaNet input-clone elimination**.
The DeltaNet arm clones `q/k/v` to split borrows; measured ~216 MB of memcpy per
512-token prefill on the 9B, about 0.1% of a ~4.7 s prefill, with no RSS cost
and nothing to clone at decode. Revisit only if a profile flags DeltaNet
allocation.

## Plan and code review 2026-11-11

Every plan doc (`plan.md`, `deepseek.md`, `qwen38.md`, `synthetic-models.md`,
`train.md`, `mtp.md`) was read against the tree and corrected in place where it
had drifted; the entries below are what that pass and the accompanying code
review left open. `disk-streaming-plan.md` was not re-read in this pass (it was
condensed on 2026-11-10).

### B102 — a real DeepSeek V3.2 file runs without the lightning indexer (2026-11-11)

**Tag:** deepseek · **Kind:** surprising, not a bug; plus the coverage it leaves
open

`deepseek.md` and the root `README.md` said no real V3.2 file could be obtained.
One can, and the tree already tests it: `cpu_deepseek_v32_config` and the
`#[ignore]`d `cpu_deepseek_v32_golden` in
`crates/infr-llama/tests/cpu_backend.rs` run against
`unsloth/DeepSeek-V3.2-GGUF`. But that file — like every published V3.2
conversion, per the test module's own comment — declares
`general.architecture = "deepseek2"` and carries no indexer tensors. So a real
V3.2 runs the stage-2 dense-MLA path, and the `deepseek32` arch (MLA plus the
indexer) is reachable only from the synthetic fixtures. Both docs now say so.

What that leaves unverified:

- **The original model attends sparsely; the GGUF attends densely.** V3.2's
  indexer keeps `attention.indexer.top_k` keys per query. While the context is
  no longer than that, dense and indexed attention are the same computation
  (`synthetic_deepseek32_full_top_k_matches_no_indexer_at_all` pins exactly
  this). Past it, a `deepseek2`-declared V3.2 file computes something the
  original model does not. infr is faithful to the file, and llama.cpp reading
  the same file does the same, so this is a converter property — but nobody has
  measured what it does to output quality at long context.
- **The golden is self-blessed.** `cpu_deepseek_v32_golden` cannot be compared
  against llama.cpp (the 671B model does not load there on this box), costs
  about 26 minutes per run, and is `#[ignore]`d. The correctness claim rests on
  the V2-Lite oracle over the same shared MLA graph.
- **Group-limited routing and the `exp_probs_b` bias have real-file coverage on
  CPU only.** Both tests that open the V3.2 file are CPU tests; on Vulkan those
  two mechanisms are covered by the synthetic fixture alone
  (`gpu_synthetic_deepseek2_routing_paths_execute`).
- **`deepseek-v3` pre-tokenizer ids are still unchecked** (`deepseek.md` open
  question 3). Not looked at in this pass: whether the cached V3.2 GGUF declares
  `tokenizer.ggml.pre = "deepseek-v3"`. If it does, the `llama-tokenize`
  comparison that was only possible for `deepseek-llm` can now be run for it.
- **Stage 1's `gpu_seam_matches_cpu_deepseek` is still not written** — recorded
  in `deepseek.md` § Stage 1, repeated here so it is not only in a plan.

### B103 — the released binaries' CPU baseline: stated now, still unchecked (2026-11-11)

**Tag:** release / CI · **Kind:** fragile, not currently broken

`.cargo/config.toml` sets `-C target-cpu=native` for `x86_64-unknown-linux-gnu`
and `x86_64-pc-windows-msvc`, and until 2026-11-12 the `dist` job in
`.github/workflows/ci.yml` built those triples with no flags of its own. Read
alone, that said the release archives are compiled for whatever CPU the runner
had, and would `SIGILL` on an older one.

They are not. Disassembling the v0.2.0 release assets shows a baseline build:
legacy-SSE moves outnumber VEX ones 168838 to 4877 in the Linux binary and
193129 to 6653 in the Windows one, which a `native` build on an AVX2 runner
would invert. The AVX2/AVX-512 code present is the runtime-dispatched kernels.

The reason is in the job log, not the repo:
`actions-rust-lang/setup-rust-toolchain` exports `RUSTFLAGS=-D warnings` by
default, and an environment `RUSTFLAGS` replaces `[target.*] rustflags`
wholesale. So:

- **The baseline is now stated, not inherited** (2026-11-12): the `dist` job
  passes `-D warnings` as `rustflags` to the toolchain action and as `RUSTFLAGS`
  on both build steps, with a comment saying the releases are baseline builds.
  Still missing: an artifact check that would catch a release built for the
  runner's CPU (a VEX / legacy-SSE ratio, or a `zmm` count outside the kernel
  modules), and a decision on whether the floor should be raised (for example
  `x86-64-v2`).
- **Every CI job that uses that action tests the baseline build, not the
  `native` one developers run.** The comment in `.cargo/config.toml` says the
  CPU backend leans on host-CPU autovectorisation; that code path as compiled
  locally is therefore never what CI's `cargo test` or the CPU goldens execute.
  `docs/ci-matrix.md` now says so.
- **Not checked:** the macOS x86_64 asset, which has no rustflags table at all
  and so compiles every `f32::mul_add` in the CPU backend without FMA; whether
  that is a libm call per element there was reported by the review and not
  measured.

### B104 — tokenizer: where infr still differs from llama.cpp (2026-11-12)

The 2026-11-12 slice made runs of spaces, the Llama-3 and Qwen3.5 pre-tokenizers
(and the aliases that share a regex infr has) and the end-of-generation set
follow `llama-vocab.cpp`, checked against the vocab fixtures in the llama.cpp
checkout (`models/ggml-vocab-*.gguf.inp/.out`): gemma-4 46/46 rows, llama-bpe
47/47, qwen35 51/51. What still differs:

- **SPM decode keeps a leading space llama.cpp's `detokenize` would drop.**
  `add_space_prefix` now defaults as the reference does, the prefix is added
  before every fragment that starts the text or follows a special token, and
  Phi-3 / jina / modern-bert whitespace stripping around special tokens is
  ported (`SpecialStrip`). The decoder never strips: every infr caller decodes
  generated ids, and llama.cpp's server keeps the space on a reply. Decoding a
  whole BOS-less sequence from position 0 therefore differs by one space; no
  path does that. Fixture round-trips are 46/46 only when that one space is
  allowed for.
- **Phi-3's `</s>` is not stripped around**: llama.cpp forces that token to
  CONTROL and infr does not (next bullet).
- **UNKNOWN-typed tokens are now special**, as in `cache_special_tokens`, so a
  literal `<unk>` in text becomes that token, and a generated one is dropped by
  the reply paths that decode with `skip_special = true` (streaming and the
  `generate_*` helpers) — llama.cpp's server does the same unless run with
  `--special`. Gemma 3's vocab was not checked for other type-2 tokens; the
  golden did not move.
- **End of generation is narrower than llama.cpp's on purpose.** llama.cpp
  matches its list by text and then forces the token to CONTROL, so it stops on
  Qwen's `</s>` (a NORMAL token in the qwen2 / qwen35 vocabs). `eog_ids` counts
  a text match only when `tokenizer.ggml.token_type` says CONTROL or
  USER_DEFINED, so a literal `</s>` in a reply still prints; ids the GGUF
  declares as eos / eot / eom always count. With no token-type array the set is
  the old three names. llama.cpp's FIM pad / rep / sep ids are not in infr's
  set. Not checked against vocabs outside the fixtures (gpt-oss, Kimi, MiniMax).
- **Pre-types with a regex infr does not have** (tekken, minicpm5, youtu,
  falcon, starcoder, refact, command-r and any unknown name) still get the GPT-2
  regex, now with a warning; llama.cpp throws on an unknown one. Erroring was
  not chosen because it would refuse files that load today.
- gemma4's newline-run split and whole-token special case in llama.cpp are not
  replicated; the fixture matches without them.
- The forced-tool grammar still receives only the model's single EOS, not the
  whole end-of-generation set (`OaiRenderer::tool_constraint`).
- Local Qwen3.5 golden hashes may have moved, since `qwen35` changed regex; they
  need the model to re-bless and it is not on the dev machine. The two CI
  goldens (Qwen3-0.6B, gemma-3-1b) were run and did not move.

### B108 — CPU attention converts the whole KV prefix per layer per token (2026-11-11)

**Tag:** CPU perf · **Kind:** perf, confirmed by reading, not measured

In the `Op::Attention` arm of `crates/infr-cpu/src/lib.rs`, the `deq` closure
builds a fresh `Vec<f32>` of the K cache and another of the V cache —
`min(kv_len, cap_rows) * n_kv * head_dim` elements each — before the per-head
loop, on one thread, on every call. For the default F16 KV that is a scalar
`f16 -> f32` per element; for F32 KV it is a `.to_vec()` of data that could be
borrowed; for a sliding-window layer on a non-ring cache it converts the whole
prefix when only the window is read. The cost grows linearly with context depth
and is paid once per layer per decoded token.

Belongs with B101. Options: borrow F32 directly; convert only the rows the mask
admits; do the conversion per head inside the pool; or keep an incrementally
extended f32 shadow. Measure first (`samply`, deep context) — the review's
estimate of the cost was arithmetic, not a profile.

### B110 — hub selection: cached-quant fallback and the mmproj ordering (2026-11-11)

**Tag:** hub · **Kind:** one decision, one bug

`infr_hub::ensure` asks the local store before the network, and `pick_gguf` in
`crates/infr-hub/src/store.rs`, given no explicit quant and no cached
`DEFAULT_QUANT`, returns `fallback_quant`: the first non-float GGUF in the order
`snapshot_ggufs` got from `read_dir`, which is not sorted.

- **Decision needed.** After `infr pull org/repo:Q8_0`, a bare
  `infr run org/repo` runs the cached Q8_0 rather than fetching the default
  quant. That is defensible (do not download gigabytes unasked) or surprising
  (the same command means different models on different machines). If kept, it
  should be deterministic — sort — and say which file it chose.
- **Bug.** `Match::Exact` returns before the `is_mmproj` exclusion, so
  `org/repo:F16` or `:Q8_0` can select an `mmproj-…-f16.gguf` projector as the
  weights when it comes first. Skip mmproj files before matching. `is_mmproj`
  also only recognises a leading `mmproj`, not a `name.mmproj-Q8_0.gguf` form.

### B112 — a foreign `tokenizer.json` beside the GGUF is only warned about (2026-11-12)

**Needs:** a decision. `load_tokenizer` now compares a sample of the sidecar's
ids against `tokenizer.ggml.tokens` and warns loudly on a mismatch, but still
uses the sidecar: falling back could break a model that only works because of
it. Options: keep the warning; fall back to the embedded tokenizer on a
mismatch; refuse to load. The check cannot see a vocabulary that spells every
sampled id the same (one that extends or truncates the model's own, since sizes
are not compared — padded GGUF vocabularies differ in size legitimately), or a
change only in the pre-tokenizer or merges. The serve renderer and grammar now
load the same resolved tokenizer as generation; that wiring is compile-checked
only, since a test needs a GGUF file. `DiffusionVisual` still opens its renderer
with the embedded tokenizer, for display only.

### B115 — malformed `INFR_*` numbers are dropped silently (2026-11-11)

**Needs:** a decision. `num`, `num_pos`, `opt_size`, `opt_mib` in
`crates/infr-core/src/config/env.rs` ignore a value that does not parse, which
`config.md` documents as frozen behaviour. The consequence for the serve safety
limits: a typo in a request-timeout or max-tokens-cap variable leaves the server
with no limit and no message. Options: warn for every key; error for the serve
safety keys only; keep the documented behaviour.

### Review coverage gaps (2026-11-11)

One reviewer, reading only — nothing was built or run by it. Each finding above
was then re-read at its call site before being recorded, and B103 was checked
against the released binaries and a CI job log; "plausible" entries say what was
not established.

The areas this pass left unreviewed were taken up afterwards; see "Review
coverage as of 2026-11-12" at the end of this file for what is still unread.

## Remaining-area review 2026-11-11

The areas the earlier 2026-11-11 pass listed as unreviewed, taken one at a time.
Same method: one reviewer reading only (nothing built or run), each finding then
re-read at its site before being recorded. What was confirmed by that re-read
and what is still the reviewer's inference is stated per entry. Coverage is at
the end of the section.

### B116 — what the Vulkan lowering fixes left untested (2026-11-12)

The prefill over-read of the dequantised KV scratch, both MoE placeholder
buffers, the pager claiming blocks from a discarded segment, and the two
error-path leaks are fixed. What is left is all test coverage:

- **The over-read test only bites in a debug build.**
  `attention_q8_kv_prefill_off_tile_matches_host` fails before the fix because
  of the `debug_assert!`s in `Recorder::attention_prefill_flash` /
  `attention_prefill_nonfa`. In release the old read is out of bounds with
  undefined contents; on RADV it returned zeros and the test stayed green. It
  also needs the f16 cooperative-matrix tier, so it skips on the iGPU.
- **"Destroyed while bound" has no test.** The validation layer reported no
  error for the old `hash_buf` even before the fix, presumably because the
  binding goes through push descriptors. A test needs a recorder hook that
  checks every bound buffer outlives the recording. The arena-growth assertion
  in `moe_ffn_graph_matches_host` covers the leak half, and not the replay-tape
  path.
- **Failed submits and failed `end_command_buffer` are fixed by reading only.**
  `submit_nowait` / `submit_wait` undo a recording's staged placements when its
  submit fails, and `Recorder::end_segment` / `discard` free their command
  buffer and pools on an error. Nothing in the repo can make a submit, an
  `end_command_buffer` or a memory bind fail (same blocker as B48), so none of
  these, nor the `make_buf_ex` bind-failure cleanup, has a test. The
  `record_decode_replay` leak does: `tests/replay_error_teardown.rs` makes the
  recording fail and reads the validation layer's log, and skips visibly without
  the layer.
- **The pager's staged-placement journal assumes one recording at a time per
  backend.** Under `serve --parallel N` that holds: every GPU step in
  `generate_dense_backend`, each prefill chunk included, takes the `StepGate`
  (read, not run). The diffusion-gemma denoise, MTP verify and the `mtp/`
  forwards execute without it; the parallel server cannot reach them today, and
  wiring any of them in would break the assumption.
- The dense-streaming `forget_unsubmitted` and the host-read-failure ordering
  have no direct test.

### B120 — Vulkan findings confirmed by reading and not fixed (2026-11-12)

The 2026-11-12 slice fixed the tall lm_head over-read, the narrow warp GEMM at
`in_f % 64 == 32` (every BK=64 route, not only `matmul_native_off`), the mmq
scratch rows, the `store_q8` subgroup pin, the unchecked device features, the
zero split override, the `tp_record_copies` extents and the unused descriptor
pool. These were confirmed and left:

- **`sample_topk` exact ties are resolved by scheduling.** When more than k
  logits tie exactly at the threshold, Phase B's `atomicAdd` slot claim decides
  which survive. Needs a decision: a deterministic tie-break has to match the
  host sampler, and the decode sampler needs a perf check afterwards.
- **UMA budget drift.** On a unified-memory device without
  `VK_EXT_memory_budget`, spilled bytes are never added to `device_used` while
  the total counts every heap (`vram_budget_fits`). Both GPUs here expose the
  extension, so no test can fail here.
- **The replay guard is silent.** A replay whose bound buffers differ from the
  recording now drops the tape and records again (`execute`) or declines
  (`execute_chain`). No path in the tree rebinds a replay-eligible plan, but a
  caller that alternated a binding would re-record every token with no log line;
  a one-time debug message would make that visible. A non-`VkBuffer` wrapper
  (multi-GPU) is identified by object address, so a freed address being reused
  can hide a rebinding.
- **The Q3_K decoders read one word past an odd last block** in source, into the
  unused `.w` lane. Not observable: a 610304-row tensor exactly filling its
  arena block ran under GPU-assisted validation with no report, the dead load
  being removed by the compiler. Left as is.
- **`cmd_pool`'s Mutex serialises nothing**; only its comments were corrected.
  The pool's external synchronisation relies on one thread driving a backend at
  a time (`StepGate` under `serve --parallel`).

Not verified, because no device here can show it:

- `store_q8` / `store_q8_dyn` now pin a 32-wide subgroup. Both RADV devices
  already have a minimum of 32, so no test fails without the pin; it needs a
  device whose default subgroup is narrower (Intel ANV at SIMD16).
- A device missing `shaderInt8` or `computeFullSubgroups` is refused at backend
  creation with the feature named, after the wave32 check. The check itself has
  a unit test; the call site has never run against a device that lacks either.
- An `Activations` or `KvCache` buffer larger than `maxStorageBufferRange` is
  refused at creation, since both are bound whole. On RADV no buffer can be
  created that large. On a driver with a lower limit, an over-size KV cache
  fails model load with a clear error and an over-size activation scratch fails
  mid-generation, and nothing in the fit ladder reads the limit. Binding the
  inert KV descriptor with an explicit capped range would remove the KV half.
- The B122 and mmq-scratch tests (`mmv_mrow_tall_head`, `mmq_scratch_rows`) can
  only fail through GPU-assisted validation: the over-read changes no output.
  Without the Khronos validation layer they check the numeric result only and
  print a warning saying so. They are `#[ignore]`, so CI never runs them.

Worth measuring in a quiet window: decode on a 262144-token head with a narrow
embedding (padding workgroups now exit at once, so it should be faster), and the
KV-store kernel on RADV after the subgroup pin.

### B125 — Metal: fixed by reading, verified only by CI (2026-11-12)

The 2026-11-12 Metal slice was written on Linux: the code was cross-linted for
`aarch64-apple-darwin` and first compiled and run by the macOS CI jobs. Fixed:
the sampler's uniform slot (the runner now writes each draw to slot 0 and to
`pos & 63`, so the Metal sampler reads a fresh draw whether or not it replays),
the scratch pool (one buffer per tag, growing by at least doubling, clamped to
the device maximum), DeltaNet key-head widths the kernel does not fit (host
path), replay admission (`RmsNormAdd`, the fused / scaled MoE, and
`no_decode_replay` are refused), the nil-buffer check (`new_shared_buffer`), the
reply cap on the Metal chat path, and a chain that ran one step reporting it.
Left:

- **No test samples a llama-family model on Metal** (the B142 gap): it needs a
  model file. `a_step_uniform_reaches_both_sampler_slots` covers the runner side
  on any OS.
- **Scratch buffers never shrink** while the backend lives; each tag keeps a
  buffer under twice its largest request.
- **A chain that returns fewer ids than asked** (reachable only when two threads
  share one backend and the tape changes under it) leaves `written_end`
  overstated, costing a qwen35 re-prefill, and has spent sampling draws for
  steps it did not run, so a seeded run diverges from a per-token one. Also,
  pre-existing: `execute_graph_chain` re-takes the tape lock and checks only
  that a tape exists, not that its fingerprint still matches.
- **`replay_fp` now hashes `MetalBuffer::uid`** instead of wrapper addresses; no
  test, because address reuse cannot be forced.
- **CPU single-row Q5_0 still takes the generic arm**, so a decode row and its
  prefill row differ in the last bits. Routing it through
  `vec_dot_q5_0_32_batch` fixes that and was tried, but it moves the
  `cpu_golden_gemma3` hash; it needs the owner's go-ahead to re-bless.
- **CPU single-row MoE dequantises to f32**
  (`int8_ok = rows >= 2 && !reference`): deliberate, documented in `moe.rs`, and
  changing it also means re-blessed goldens.
- **`x_stride`:** CPU `QkNorm` and `Rope` now assert it is zero, like Metal.
  Vulkan ignores it on those ops and when `freq_factors` is set; `QkNormRope`
  uses a `2*hd` head offset on CPU and Vulkan and `x_stride/n_head` on Metal,
  which agree for the only emitter.
- Building docs for the Apple target fails on a public doc in `shaders.rs`
  linking the private `crate::pcache`, and on an unresolved
  ``[`MtlBuffer::contents`]`` in `lib.rs`. CI does not run that build.
- B50 (a DeepSeek MoE layer on Metal) could be written blind, since a no-model
  CPU parity test would check it in the macOS job, but it is a router-shader
  feature and was not started.

### B129 — what the qwen35 state-reset and Q8 KV seed fixes left untested (2026-11-12)

Both bugs are fixed: `generate_dense_backend` zeroes the DeltaNet state whenever
`cached` is empty and keeps `cached` empty across a qwen35 prefill, and KV
prefix seeding goes through `Backend::copy_kv_prefix`, which Vulkan overrides
for its planar Q8_0 layout. What is left:

- The `cached.clear()` after `cached.truncate(start)` is now a backstop, not the
  guard: `written_end != cached.len()` already refuses the overrun state, so
  `qwen35_aborted_prefill_matches_a_fresh_slot` goes red only with both removed.
  The clear's one remaining effect is to skip the snapshot-resume arm after an
  aborted call, which costs a re-prefill from zero.
- The Vulkan Q8_0 seed is covered only by
  `seeded_kv_prefix_decodes_like_its_source_vulkan`, which is `#[ignore]` and
  returns early without a device, so CI never runs it (see B140). It was run red
  and green on the iGPU.
- The `copy_kv_prefix` overrides in `tp.rs`, `ep.rs` and `pipeline.rs` compile
  and were read, but were never run: they need a multi-GPU session.

### B131 — qwen35 resume and the ring guard: what is left (2026-11-12)

`SeamKv::written_end` records one past the highest position any execute may have
written, and replaces `cached.len()` as the high-water mark: qwen35 reuses its
recurrent state only when the two agree, and the sliding-window ring guard uses
it. When a qwen35 chain stops early, the next turn restores a snapshot of the
state taken at the previous turn's prefill boundary and prefills from there
instead of from zero. Left:

- **The snapshot and the fit ladder — needs a decision.** The snapshot is a
  second copy of every DeltaNet layer's state, per slot:
  `4 × ((ssm_d_conv − 1) × q35_conv_channels + q35_num_v_heads × q35_head_k_dim × q35_head_v_dim)`
  bytes per layer, about 19.3 MiB for Qwen3.5-0.8B. Re-read 2026-11-12: the fit
  does not count it, but `kv_bytes_estimate_fmt` already prices every DeltaNet
  layer as a full per-token KV cache, which exceeds live state plus snapshot
  above roughly 1.1k tokens at f16 — so an auto-sized context leaves room, and
  adding a snapshot term there would count it twice. The gap is an explicit
  `--ctx` that fills the device: the snapshot can then be refused, which is
  handled (one warning, no snapshot on that slot, the next mid-chain turn
  re-prefills from zero). The real fix is to price DeltaNet layers by
  `kv_layer_alloc_bytes` plus the snapshot (when a chain or MTP can run), which
  makes auto-sized qwen35 contexts LARGER — a visible change. Separately, the
  MTP path's snapshot failure still fails the request (`?`) where the decode
  loop degrades.
- **The MTP rollback can leave a half-restored state behind.**
  `mtp_restore_delta` (in `seam/weights.rs`, called from the MTP loop in
  `mtp/mod.rs` after a partial accept) copies the snapshot layer by layer and
  only then truncates `cached` and sets `written_end`. If a copy fails partway,
  the error propagates with `cached`/`written_end` unchanged and the snapshot
  still valid, while the live state is part snapshot, part live; a later turn
  that extends `cached` would take plain reuse on it. The decode loop's resume
  arm had the same hazard and now clears `cached` and `written_end` before the
  copy and invalidates the snapshot on failure
  (`qwen35_failed_resume_is_not_reused_vulkan`). The MTP side was left because
  its fix is the same reordering plus an invalidate inside `SeamKv`, not a
  one-line change, and no test injects a failing copy into an MTP run (the
  resume test's `RefuseKv` wrapper is the pattern to copy).
- **Temperature-0 output can depend on where a chain stopped.** A resumed turn
  re-prefills the previous reply as a batch, where plain reuse had fed it token
  by token; the batched DeltaNet kernels chunk from the dispatch's first row, so
  logits differ at about 1e-5..1e-4 and a near-tie can sample differently.
  Bit-identity across turn splits did not hold before either.
- **Untested:** the `seed_from` high-water change, and the snapshot under a real
  VRAM refusal (the test injects one through a wrapping backend, because no knob
  makes the guard refuse a chosen allocation). The qwen35 chain and snapshot
  tests are `#[ignore]` and need a device whose `max_decode_chain` is at least
  2; the iGPU's watchdog caps it at 1, so there they print a skip naming the
  reason (`with_q35_vulkan`), and they run only on the RX 7900 XTX.
- B94 leans on rows past `last_written` being ignored. That is no longer true
  for ring and qwen35 sessions, which now track them; re-read B94 with that in
  mind before building it.
- The DiffusionGemma "Paris" coherence tests in `tests/cpu_backend.rs` have not
  been re-run since the seeded RNG changed (every even seed, the default 42
  included, now draws a different stream); the model is not on the dev machine.

### B136 — pipeline-cache tripwire: accepted limits (2026-11-12)

Liveness is now an advisory lock each instance holds on its marker, with the pid
check kept where no lock can be taken. Accepted and documented in
`kernel_cache.rs`: a still-running pre-lock `infr` loses the cache once across
an upgrade; and when the rename into place fails, a sweep landing between
opening and locking the fallback marker discards the cache once and leaves that
run unprotected against its own hang. Not run: real NFS (where `flock` is
emulated per process — the in-process registry is there for it). Windows was run
on 2026-11-13: renaming a marker while its lock is held and unlinking one under
a lock both work there (`arming_renames_the_locked_marker_into_place` observes
the rename, which the fallback would otherwise hide). The cgroup headroom fix
(`cgroup_level_headroom` subtracts inactive file cache) was likewise not run
inside a real container, and does not subtract `active_file`.

### B133 — what the loader, host-pager and LRU fixes left open (2026-11-12)

The malformed-GGUF panics, the host tier's insertion-order LRU and the block
stranded as `Loading` by a panicking read are fixed, each with a test run red
first. What is left:

- **I2S on ARM-quantised files, suspected.** `Gguf::open` now requires an i2_s
  element count to be a multiple of `I2S_GROUP_ELEMS`, which the decoder always
  assumed. The reviewer recalls bitnet.cpp using a group of 64 on ARM builds;
  there is no reference checkout here to check. Such a file would have decoded
  wrongly before and is refused now if its count is not also a multiple of the
  x86 group.
- **Eviction now scans every resident block.** `Pager` ranks blocks by a recency
  stamp in a map, so a hit is one map write (it was a deque scan) and choosing a
  victim is a full scan (it was the first eligible entry, usually the front).
  The order is unchanged. Not measured: the reviewer estimates a few
  microseconds per miss at today's slot counts, next to an upload of a megabyte
  or more, but a miss-heavy sweep at DeepSeek V4 slot counts would be the place
  it shows (`deepseek.md` §0.3 names the LRU cost).
- **`Claim` in `hostpager.rs` cannot be tested for a panic in the slot
  arithmetic** without fault injection into `Arena`; it is built before that
  code, and only the panicking-read case has a test. A poisoned `HostPager`
  mutex after a successful read still panics the next caller, as before.
- **`bytes_to_f32`'s integer arms drop a trailing partial word** where the old
  cast panicked. The loader never produces a byte length that is not a multiple
  of 4.
- The KV-cache `cast_slice` calls in infr-cpu (`dequant_cache_prefix`, the
  attention `deq` closure) were left: they read buffers the backend allocates,
  not the GGUF mapping.

### B137 — `infr serve`: three decisions for the owner (2026-11-12)

- **`response_format` is accepted and ignored.** A client that asks for JSON
  output gets unconstrained text with a 200. Options: reject `json_object` /
  `json_schema` with a 400 (safe, but clients that get 200 today break);
  implement JSON mode on the existing llguidance grammar path (correct,
  feature-sized); keep ignoring it (silently off-contract).
- **No `Host` / `Origin` validation on the default keyless loopback server.** A
  web page that rebinds its hostname to `127.0.0.1` becomes same-origin and
  could drive completions and read `/status` and `/metrics`. Reported from
  reading `build_router`; the attack was not tried. Options: validate `Host`
  against a loopback allow-list (blocks rebinding, breaks custom hostnames and
  reverse proxies); validate only when a browser `Origin` header is present;
  document it and leave it. Distinct from B5.
- **No SSE keep-alive.** Nothing is sent during a long prefill or a forced tool
  call, so an idle-timeout proxy may cut the stream. axum's `KeepAlive` sends
  comment lines at a 15 s default, which conforming SSE parsers ignore. Options:
  add it at the default; make the interval a `serve.*` key; leave it out.
- **`max_tokens: -1` means the server default (`sampling.max_new`)**, chosen on
  2026-11-12; llama.cpp reads `n_predict: -1` as "until the context is full".
  Recorded so the difference is a decision and not an accident.

### B138 — smaller infr-core findings (2026-11-11)

- **`cpu_arena_plan` (`hostmem.rs`) has no headroom in its fit test.** It
  returns `Skip::Fits` when `pageable <= available`, a deliberate choice with a
  unit test pinning it, while the arena it would otherwise size keeps headroom
  free. A model that just fits stays on mmap and competes with KV, activations
  and the repack cache for the same RAM. Changing it needs a measurement (a
  model that just fits, mmap against arena).
- `fusion::dst_only_read_by_next` is still used by `plan_linear_add`, but its
  doc claims a backend post-filters with it and none does; making it private is
  a breaking change to the published crate.

### B139 — NaN-aware comparisons: what the sweep could have missed (2026-11-12)

`infr-testkit` exports `nan_max` / `nan_max64`, and every pass/fail maximum
found in the test suites now propagates NaN (an `f32::max` fold scores an
all-NaN output as error 0). The sites were found by grepping for `f32::max` /
`f64::max`, `fold(0…max`, `x = x.max(` and `if e > m`; a gate written another
way is not covered. `infr-llama` does not depend on `infr-testkit`, so
`seam_op_parity.rs`, `synthetic_deepseek2.rs`, `cpu_backend.rs` and the qwen35
reset test in `seam/runner.rs` carry local copies. The Metal sites compile for
the Apple target but were not run.

### B140 — tests that CI reports as passed without running them (2026-11-11)

- **Model-gated unit tests in infr-llama that still skip.** `cpu-goldens` now
  also runs the library tests, so the grammar tests and
  `user_text_special_tokens_are_literal` execute. Still skipping there:
  `embedded_tokenizer_matches_sidecar` (needs a `tokenizer.json` beside the
  GGUF, which an HF cache snapshot does not have) and `loads_and_dims` in
  `qwen35.rs` (needs Qwen3.5-0.8B, which the job does not download).
- **Metal model-level tests.** The `metal_*` tests in `cpu_backend.rs` are
  macOS-only and model-gated, and `test-macos` runs only `-p infr-metal` and
  downloads nothing (confirmed in `ci.yml`). Metal's runner, session, slot pool
  and sampling integration therefore never executes anywhere — no one on the
  project has a Metal device. `docs/ci-matrix.md` was corrected on this point.
- **Windows golden hashes** (`per_os(windows, …)` in `QWEN3_GOLDEN` and
  `GEMMA3_GOLDEN`) are never asserted in CI: `cpu-goldens` is Ubuntu-only. On
  the Windows dev box they still match as of 2026-11-13 (`cpu_golden_qwen3`,
  `cpu_golden_gemma3` and `cpu_golden_qwen3_quants` pass, after the 2026-11-12
  tokenizer fixes).
- **Tests that skip on an integrated GPU, and pass.** The qwen35 snapshot tests
  in `seam/runner.rs` (`with_q35_vulkan`) skip where `max_decode_chain` is 1,
  because no chain runs and no snapshot is taken. Four prefill-chunk helper
  tests in `seam/mod.rs` (`default_chunk_is_discrete`) skip once the process has
  opened an integrated GPU, because the default chunk they assert follows the
  process-global device class (see B145). Both print the reason on stderr even
  without `--nocapture`; a run on the iGPU alone therefore checks neither.

### B141 — assertions weaker than the tests' names (2026-11-11)

- **What the Vulkan KV-format check cannot see.** `gpu_seam_kv_q8_tracks_cpu`
  and `gpu_seam_kv_formats_track_cpu` score every format through
  `assert_vulkan_kv_formats_track_the_cpu`: the Vulkan seam's last-row logits on
  `KV_VULKAN_PREFIXES` prefixes of `kv_check_prompt` (each a batched prefill
  that writes the cache, then one decode step that reads it) must differ from
  the Vulkan f16 cache's, and must reach a per-bit-width cosine floor against
  the CPU reference backend running the same format. Measured 2026-11-13 on
  Qwen3-0.6B; each item names its device. What it leaves uncovered:
  - **An f32 cache is exempt from the "differs from f16" half, by design.**
    Vulkan's attention is f16. The f32 cache is written by
    `store_kv_f32_from_f16` for K (the f16 rope output) and `store_kv_f32` for V
    (`store_kv_dense_kernel`), and read back through the `store_f16` cast into
    the f16 scratch (bf16 goes through `dequant_kv_f16` instead). So the f32
    cache holds K at f16 precision and V's cast to f16 moves from the write to
    the read; its logits equal the f16 cache's bit for bit on every prefix,
    observed on both GPUs. `store_kv_dense.comp` calls this a storage choice
    bounded by the f16 query precision. An f32 KV cache on Vulkan therefore
    costs twice the f16 memory and changes no logit — not a bug, but worth
    knowing before recommending it.
  - **A uniform scale error of a few percent passes** (all on the integrated
    GPU). Inflating the GPU-stored block scale by 5% left the worst cosine at
    0.9690 for a q4_0 V cache (floor 0.95) and 0.9952 for coupled q8_0 (floor
    0.98). By 25% both fail: q4_0 at 0.9421 on the 31-token prefix, q8_0 at
    0.9656 on the 31-token prefix (the first prefix under each floor; the run
    stops there). The floors sit under what correct caches reach — turbo3 0.9722
    on the 7900 XTX, iq4_nl 0.9749 and turbo2 0.9581 on the integrated GPU — so
    a smaller error is inside one row's noise.
  - **The coopmat prefill attention tiers are never reached.**
    `verify_dense_vulkan` sizes the KV cache at the prompt plus two rows, and
    the flash and nonfa tiers each need the KV length rounded up to their tile
    to fit the cache, so on the 7900 XTX the prefixes long enough for those
    tiers (64 prefill rows or more) fall back to the split-K or scalar
    attention, while the integrated GPU (no coopmat) runs `attn_nc_fa` for them.
    The two GPUs therefore run different prefill attention in this test, and the
    flash prefill with its padded dequant scratch is exercised by no KV-format
    test. Established by reading the gates; which of split-K or scalar runs was
    not instrumented.
  - **Multi-step decode is covered only by text.** With `logits_out` set, the
    chained decode is off and each forward runs exactly one decode step, so the
    replay tape advancing over many steps, and decode reading rows written by
    earlier decode steps, are reached only by the q8 test's
    `generate_vulkan_session` text check.
  - **Every quantised format sits on V except q8_0.** No low-bit, iq4_nl or
    turbo K is quantised and no coupled low-bit K==V pair runs, so the
    `flash_dequant` path and the opt-in inline decode (`kv.inline_decode`) are
    never reached.
  - **The `kv.force_q8` alias runs only on the session text path**; the numeric
    check names q8_0 through `kv_types`.
  - **Only the last row of each prefix is scored**, so the batched prefill's
    attention is checked only through what it writes into later layers' cache,
    not row by row.
- **Reported, not re-read:** `metal_kv_scratch_paths_are_coherent` pins one hash
  for eight KV formats with no f16 control; `mtp_head_trunk_acceptance_rate`
  prints the rate and asserts nothing about it (its model is not cached); no
  Metal parity test names `Op::CopyStrided`, `MulVec`, `Softmax`, `TopkMask` or
  `RmsNormAdd`. Metal's own `mla_parity` was not checked for the
  identical-K-rows weakness the infr-llama one had.
- **Hygiene, reported:** `test_serial_lock` and `gpu_serial_lock` are in-process
  mutexes, which do nothing under nextest's process-per-test model if the
  ignored GPU tests are ever run that way; the SIMD tier tests cover only the
  tiers the runner's CPU has (see B3).
- The CPU KV check adds about a tenth to the `cpu_backend` golden run on the dev
  machine (60 s against 55 s, the later figure taken at load average 10–16, the
  earlier at 1.3 — not a clean comparison). Read the `cpu-goldens` job time
  after it lands.

### B142 — none of this review's bugs has a test that would catch it (2026-11-11)

The test reviewer was asked, for six of the findings above, whether an existing
test would fail. All six answers were no; B116, B129 and B130 have since been
fixed with tests, as have B122 and B123, and the remaining one is:

- B125 (Metal sampling on a llama-family model): the sampling test uses Qwen3
  and compares Metal with Metal; the Llama test is greedy; neither runs in CI.

This is the reviewer's answer from reading the tests, not re-checked. It is the
test to write first when the entry is picked up.

### B143 — the macOS `glslc` install is retried but not cached (2026-11-12)

`.github/actions/install-glslc` now retries its install on every platform, and
on Linux and Windows it also keeps what it installed in the Actions cache (the
`.deb` files; `glslc.exe` copied out of the Vulkan SDK), keyed on the runner
image version, so a warm run needs no package feed. That followed the 2026-11-11
Ubuntu mirror outage, which failed every Linux job of three runs in a row.

macOS has the retry only. `brew install shaderc` still fetches Homebrew's
formula index even when the bottle is on disk, so caching the bottle does not
make the install offline, and caching the installed keg (`shaderc` plus the
`glslang` and `spirv-tools` it depends on) and relinking it was not tried. A
Homebrew or GitHub Packages outage longer than the retry window still fails the
macOS jobs, and through them a release.

Not verified: the two fallback paths (feed down with an older cache entry
present; exact hit whose cached copy fails to install). Nothing in CI can take a
feed down, so they have only been read, not run.

### B144 — device-limit refusals are tested through overrides, not on such a device (2026-11-12)

The Vulkan refusals for a device with smaller limits than this box's are tested
by playing that device on one test thread: `TEST_KERNEL_LIMITS`
(`VulkanBackend::kernel_limits`, read by the kernel build and by
`Recorder::rmsnorm`'s wide-vs-narrow choice) and `VKB_RANGE_CAP` (the
`maxStorageBufferRange` `Recorder::vkb` holds a sub-tensor bind to). The tests —
`a_workgroup_past_the_device_limit_is_refused_by_name`,
`rmsnorm_falls_back_to_the_narrow_kernel_on_a_small_device`,
`a_refused_kernel_does_not_poison_the_kernel_cache`,
`vkb_refuses_a_sub_tensor_past_the_storage_range` — run on both GPUs here. What
they cannot show:

- That a real driver reporting those limits fills them the way the overrides do.
  No device with limits below desktop drivers' is available.
- Anything the overrides do not reach: the process-global
  `max_storage_buffer_range` used by `check_bindable_size` has its own
  `TEST_BIND_LIMIT`, and the subgroup-size rule in B-HWDET-LIMITS is not
  modelled at all.
- A kernel refused at build still panics (the kernel cache is infallible until
  B98 threads a `Result` through); only the lock is released first.

### B145 — the device class is process-global, so tests see whichever GPU opened first (2026-11-12)

`infr_vulkan::device_class()` is published once, by the first `VulkanBackend`
the process opens, and `seam::default_ubatch_rows` reads it. In a test binary
run with `--include-ignored` on `INFR_DEV=vulkan1`, the first GPU test opens the
iGPU and every later test sees the integrated default chunk, including pure unit
tests that never touch a GPU. Four of them asserted the discrete default and
failed; they now skip instead (`default_chunk_is_discrete`, B140). The coupling
itself is left: a test's outcome depends on which device the binary opened
before it. Options: make the chunk default a parameter the tests pass, or scope
the device class per backend instead of per process. A design decision about
`device_class()`'s one production reader, `default_ubatch_rows` (read); not
attempted.

### B146 — Vulkan lowering review, 2026-11-12: what was fixed, refused and left (2026-11-12)

A read of every `lower_op` arm outside attention and MoE, with the recorder
methods each calls and each shader's push block, bindings and indexing, found no
push-constant layout mismatch, no binding or write-tracking error and no
undersized scratch. Fixed from it: an F16 `Linear`+`Add` fused onto a float16
shader on a device without `shaderFloat16` (or under `INFR_NO_F16`); a panic
under `INFR_MMV_FUSE_QUANT=1` on any decode dtype but Q4_K / Q6_K and under
`INFR_NO_MMV_M4` on a fused decode (selection now asks the same table the
recorder dispatches from, `mmv_mrow_variant`); the rmsnorm family and
`deltanet_norm` on an unsplit grid; and a set of limits that were only
`debug_assert!`ed and are now errors from `lower_op` (DeltaNet key-head width,
`QkNormRope` head dim, `Sample` top-k range, Q8_0 `WriteKv` row width, the `Mla`
cache dtype). Left:

- **The non-coopmat GEMM (`matmul_mmq_at`, `matmul_fma_at`) refuses a grid past
  the limit instead of splitting it.** Its mmq builds already use the y
  dimension for the expert id, so the flat-index trick does not apply. No
  default shape reaches it (it needs an output width of 262,144 at a 1024-row
  chunk, and the prefill head runs on the last row only); a user-set
  `INFR_UBATCH` on a very wide FFN can, on a discrete GPU without cooperative
  matrices. Options if it matters: fall back to the per-row GEMV tier, or split
  by m-tile across several dispatches. The guard repeats the recorder's
  `ceil(m/64) * (n/64)` with its own literals; it should read the tile size from
  one place.
- **Refusals for shapes nothing emits today:** `QkNormRope` with both `x_stride`
  and `freq_factors`; a record-once `Rope` or interleaved `QkNormRope` writing K
  into a cache a sliding-window attention reads (those two kernels have no ring
  wrap); `Op::Copy` between dtypes other than identity and F32→F16. A same-dtype
  `Op::Copy` of a block-quantised dtype still sizes its bytes from
  `dense_bytes(1).unwrap_or(4)`, which is wrong; nothing emits it.
- `linear_mmv_mw_at` is unreachable while `unified_mmv_row1` is hard-coded true;
  if it is re-enabled, an `INFR_MMV_MW_WARPS` value with no build panics.
- Two tests hold by construction and prove little:
  `mmv_choice_offers_only_built_variants`' "offered ⇒ built" half, and the new
  gate-vs-table asserts for `native_mrow` / `native_gemm_fma` in `linear.rs`.
- **Not read by that pass:** the bodies of the native GEMV / mmv / GEMM kernels
  (dequant and indexing), `streamed_prefill_gemm`, `native_sg_choice` /
  `native_rm_choice`, the DeltaNet recurrence maths, the `mla.comp` body, and
  the `argmax_prob`, `quant_kv`, `store_kv_dense`, `quant_turbo` and
  `embed_gather` shader bodies.
- To measure in a quiet window: per-op decode time for `rmsnorm` /
  `rmsnorm_wide` and `deltanet_norm` after the added uniform branch.

### B147 — seam emitters against llama.cpp, 2026-11-12: fixed, capped, refused and left (2026-11-12)

Every graph emitter in `seam/runner.rs` was compared op by op with its llama.cpp
builder (checkout `57291f2`). DeltaNet, the DeepSeek MLA and V4 ratio-0 paths
and MoE routing for every MoE arch match. Fixed from it: `rope_freqs.weight` is
applied for the llama and llama4 archs on every rope path (it was gemma4-only,
so Llama 3.1+ and Llama 4 rotated a large share of their rotary pairs too fast);
the strided DeltaNet path (`deltanet_strided.comp` wrote its output at an input
offset, and the emitter enabled the mode on backends that do not implement it);
slot `fork` / `seed_from` reading the wrong prefill-chunk height;
tensor-parallel classifying q/k/v biases as weights. Capped or refused instead
of implemented:

- **Llama 4 is capped at `Config::faithful_ctx_limit()`** (one token short of
  its attention chunk). Past it the reference uses chunked attention
  (`LLAMA_SWA_TYPE_CHUNKED`) and a position-dependent temperature on the NoPE
  layers; the mask needs a new attention-mask kind in the CPU, Vulkan and Metal
  kernels. Mistral-Large-3 (`deepseek2.attention.temperature_scale`) is capped
  at its `temperature_length` the same way. Implementing either lifts the cap.
- **A `.scale` tensor the reference applies and infr does not is refused at
  load** (`applied_weight_scales`, one row per llama.cpp builder; scales the
  builder ignores load as before). This is the NVFP4 / ModelOpt case. Applying
  the scales would lift the refusal. The diffusion-gemma row copies gemma4's:
  the checkout has no builder for it.
- **An MLA model whose YaRN magnitude is not 1 is refused** (`mla_rope_mscale`):
  infr's `Op::Mla` / `Op::Rope` take frequency divisors only. Every DeepSeek V2
  / V3 / V3.2, Kimi K2 and Mistral-Large-3 file evaluates to exactly 1.
- **Needs a decision — deepseek32 with an unusual YaRN multiplier.** llama.cpp
  applies its `mscale_all_dims` special case to `deepseek2` only, so a
  deepseek32 file with stretching and a multiplier that is neither 0 nor 1 gets
  a different attention factor there than the deepseek2 rule infr uses. Real
  V3.2 declares 1 and is unaffected. The synthetic deepseek32 fixture uses
  V2-Lite's multiplier and IS in that case, so refusing it fails the fixture's
  locked-logit tests. Today such a file loads with one warning
  (`deepseek32_yarn_rules_agree`). Options: change the fixture to the real
  file's multiplier, re-bless its logits and refuse the case; or keep the
  warning.
- A qwen35 file whose `attention.recurrent_layers` is not the periodic layout is
  refused; no converter writes one.

Left, confirmed by reading and not changed:

- **qwen35 KV estimate.** `kv_bytes_estimate_fmt` prices every DeltaNet layer as
  a full attention cache (`kv_row_elems` has no qwen35 branch) while
  `kv_layer_alloc_bytes` allocates the fixed conv / S state — so the default
  context clamps to a fraction of what fits and `moe_expert_budget`
  over-reserves. Same decision as B131's snapshot bullet: fixing the estimate
  makes qwen35 contexts much larger.
- **`dsv4_plan.rs` is dead code and diverges from the reference** (no HCA dummy
  block when none completes; `n_kv` not padded to a multiple of 256). Part of
  the V4 compressed-tier feature.
- The runner's prefill chunk height still reads the live `ubatch_rows(ec)` where
  the ring uses the slot's recorded value; they agree on every entry found, and
  `.min(recorded)` would make the bound hold regardless.
- Numerically negligible differences: the DeltaNet L2 norm's epsilon form, and
  the MoE renormalisation clamp.
- For a llama4 file with `sliding_window == 0` llama.cpp makes the last layer
  NoPE and then asserts; infr ropes every layer. No known file.
- Not read by that pass: the DiffusionGemma emitters against an actual reference
  (the fork checkout is absent).

### B148 — multi-GPU wrappers and shader bodies, 2026-11-12: fixed, refused and left (2026-11-12)

`tp.rs`, `tp_allreduce.rs`, `tp_sem.rs`, `p2p.rs`, `ep.rs`, `pipeline.rs`,
`linear.rs`, `vkext.rs` and the non-test part of `caps.rs` were read in full for
the first time, and every in-shader weight decoder was compared with
`ggml-quants.c` (all match; the i-quant grids and sign tables were compared by
script). Fixed: the in-graph sampler could evict a value above the threshold
when keys tied at the k-th place (`sample_topk.comp` now takes the strictly
greater keys first and breaks ties by lowest index); `attn_partial.comp`'s
shared array overflowed for a DiffusionGemma canvas past a few thousand keys
(the chunk is capped at `ATTN_PARTIAL_MAX_CHUNK`, and the rows-batched kernel at
its own bound); `embed_gather` wrapped its u32 byte offsets before its element
count on very large tables; `moe_topk.comp`'s group arrays had no host bound;
the multi-GPU wrappers reported rank 0's capabilities, so an integrated GPU
behind a discrete one got the discrete prefill chunk; `p2p_import` did not check
the importer's size requirement. Refused or left:

- **Expert-parallel refuses gemma4moe (the dual-FFN MoE).** Its dense-branch
  output is written before the MoE cut and read after it as scratch, which each
  segment zero-fills, so it read as zero (seen on one GPU: the result differed
  from single-device by more than 1); and its per-expert down-scale table is
  indexed by global expert id while the band remap makes ids rank-local, wrong
  on every rank but the first (read, not run: a one-rank world has only rank 0).
  To support it: carry `d_out` across the cut and split the scale table by band.
  `multi::refuse_scratch_across_cuts` now refuses any such crossing in EP, TP
  and pipeline by name; it treats a partial write (`dst_off`) as a full one, so
  it can miss a refusal, never invent one. Not run against a real qwen3moe EP
  graph (the model is not cached); read to pass.
- **Expert-parallel refuses a graph ending in `MoeFfn`.** With several MoE
  layers and a trailing one the last partial was never all-reduced.
- **Needs a decision — a lost GPU hangs the semaphore all-reduce.**
  `tp_queue_wait_idle` has no timeout. A timed wait is small, but on timeout the
  healthy ranks' gather command buffers stay pending on a semaphore that will
  never be signalled, and teardown hangs anyway. Options: host-signal the
  imported timeline semaphores to release the waiters, or give up the
  zero-host-stall property.
- Pipeline refuses a second KV set (a `fork`, the MTP checkpoint) when the world
  has more than one device: an allocation cannot say which layer it belongs to.
- `INFR_F8_COOPMAT`'s GEMM casts dequantised weights to E4M3 unscaled, so most
  weights land in the format's subnormal range. It is opt-in, never validated on
  hardware, and now says so in a warning; a per-column scale undone in the
  epilogue would fix it.
- Eleven shader variants are built and never selected (`attn_partial_mrows`
  base, several `native_mmv_mw_*_sg16`), so subgroup-16 devices fall back for
  those shapes. Performance only.
- Suspected, read only: `tp_record_copies` has no barrier between the previous
  submit's compute write and the copy that reads it (it relies on the host-side
  wait); TP's `Op::Add` marks its output replicated even when an input is split.
- **Kernel families with no host-parity test:** the int8-activation arms of
  `native_mmv.comp` / `native_mmv_mrow.comp` for Q4_K, Q6_K and IQ4_XS; the
  `f8cm` GEMM; turbo quant / dequant on the GPU against `turbo.rs`; `quant_kv`
  against a host quantiser; `lightning_indexer`; `moe_topk` grouping.
- To measure in a quiet window: the sampler's added single-lane sort per stage-1
  workgroup (estimated at microseconds per token, not measured).
- `build.rs`'s list was checked by script (no duplicate output, no missing
  source, every referenced variant built), not read entry by entry.

### B149 — CLI, hub, chat templates and platform crate, 2026-11-12: fixed and left (2026-11-12)

`infr-cli`, `infr-hub`, `infr-chat` (outside `stream.rs` / `tools.rs`) and
`infr-plat` were read in full (production code; not their tests). Fixed: a
GGUF's chat template could panic the process (`strftime_now` with an unknown
specifier) or request an unbounded allocation through a helper's width argument;
tool-call ids never reached the template; a same-host redirect without a digest
header skipped sha256 verification; a download was renamed into place without
being synced, and an existing blob was trusted whatever its size; two concurrent
pulls of one model failed at the link step; the cache did not work without
symlinks; a backslash in a repo id escaped the cache on Windows;
`generation_config.json` was never found for a GGUF in a subdirectory;
`infr compare` did not pass its config to its children; `infr multi` ignored
`INFR_CTX`. Left:

- **A hostile chat template can still exhaust memory.** Widths passed to helpers
  are bounded (`TEMPLATE_WIDTH_MAX`; one `format` call's total padding by
  `FORMAT_NUMBER_MAX`), but string repetition (`"x" * n`), list repetition and
  repeated `~` / `+` doubling are evaluated by minijinja's own operators
  (`ops::mul`, `ops::string_concat`, `ops::add`), which have no size check, and
  minijinja 2.21 offers no hook: no operator override, no size or memory limit,
  and fuel charges one unit per instruction however much it allocates. A
  template is input from a downloaded file. Closing this needs an upstream
  feature or a different engine.
- **Client-minted tool-call ids are passed through unchanged.** infr now mints
  nine-character alphanumeric ids (the only shape Mistral's template accepts),
  but an id a client made itself (`call_…` from another provider) still fails
  that template; vLLM normalises those and infr does not.
- **A copy is recognised as a blob only by size.** On a filesystem with neither
  symlinks nor hard links the snapshot holds copies, which doubles disk use per
  model (logged once) and means an up-to-date pull whose HEAD cannot be answered
  fails rather than keeping the file.
- **Repo ids:** `a--b/c` and `a/b--c` map to the same cache directory, since the
  cache folds `/` to `--`. Whether Hugging Face forbids `--` in a name was not
  checked (`huggingface_hub` is not installed on the dev machine).
- Temp files left by OTHER processes that crashed between link and rename are
  not swept; with the copy fallback one is model-sized.
- A hostile hub that misreports a file's size can force a re-download on every
  pull (the replacement only lands after its sha256 checks out).
- **Run on Windows 11 on 2026-11-13** (and by CI's `windows-2025` test job): the
  `MoveFileExW` commit, the symlink → hard link → copy fallback (the refusals
  are injected, so the hard-link and copy arms run even where symlinks work) and
  file identity by volume serial and file index. The run found the commit
  refusing a memory-mapped destination (fixed; see B75). **Not run:** a real
  power loss, or a filesystem that refuses a directory sync.
- **Tests that did not test.** Two dozen model-gated tests in
  `tests/cpu_backend.rs` returned early and counted as passed in every CI job;
  they are `#[ignore = "needs …"]` now, so that job runs the tests whose model
  it has and reports the rest as ignored. The CPU `decode_parity` sweeps use the
  decoder the kernels share, so they prove the GEMV path and cannot see a decode
  bug: `infr-gguf` gained known-answer tests for Q8_0, Q3_K, Q4_K, Q5_K and Q6_K
  transcribed from `ggml-quants.c`. `cpu_golden_qwen3_quants` runs only its
  Q4_K_M row in CI. Still open from the test read: several Metal attention
  parity tests use a tolerance about as wide as one key's contribution, so they
  cannot catch an off-by-one mask; `pcache.rs`'s round-trip test returns before
  any assertion on a device without binary archives; the Metal `*_probe` and
  `*_wall` tests assert nothing and run in CI under `--include-ignored`.

### B150 — Metal shaders and tests, 2026-11-12: fixed on Linux, first run in CI (2026-11-12)

The Metal `.metal` sources and tests were read against their Vulkan and CPU
counterparts, and the fixes were written on Linux: cross-linted for
`aarch64-apple-darwin`, with the decision logic in `src/policy.rs` unit-tested
on any OS, and first compiled and run by the macOS CI jobs. Fixed: `moe_topk`
used an unclamped winner index when no router logit was finite; the flash
attention kernels read whole key blocks past the newest key with nothing
guaranteeing those rows exist (the kernel is now chosen only when they do —
`policy::select_attention` — and Metal KV caches are padded to
`policy::KV_ROW_ALIGN` rows through `Backend::kv_row_align`, so an ordinary
prefill still takes flash); the Q4_0 / Q5_0 KV writers chose the block scale
differently from the CPU on a tie; `argmax_f32` broke ties by lane; the two-row
Q4_K / Q6_K GEMV bodies read a weight row past the bank for an odd output width;
`pcache_blob::encode` wrote blobs its own `decode` rejects. Left:

- **Vulkan's `quant_kv.comp` still breaks a ±max tie toward `vmax`** for Q4_0
  and Q5_0, where the CPU, Metal and ggml's `quantize_row_q4_0_ref` take the
  first element of largest magnitude. Exact ties are rare in real activations.
- **The MTP head's KV cache is not padded** (`mtp/mod.rs` allocates it
  directly), so on Metal its multi-row attention near the cache end takes a
  slower exact kernel. MTP is parked.
- **No test observes which attention kernel a real device takes.** The decision
  function is tested on Linux; the device's choice end to end is not.
- `linear_q4k_mr` / `linear_q6k_mr` in `linear.metal` are dispatched by nothing
  (multi-row K-quants use the `_mrv` kernels); delete them.
- `device_token` is injective only up to ASCII case. Two device names differing
  by case share a cache file and rebuild it for each other; they never load each
  other's pipelines, because the stored key holds the name verbatim.
- The mask-sensitivity check `kvquant_newest_key_is_attended` relies on a margin
  that was estimated, not measured, for the lossiest format.
- **In the macOS test job the bandwidth benches are most of the test time**
  (`gemv_bw` about 48 s of about 81 s summed in run 43212953975). They assert
  nothing and run because the job passes `--include-ignored`. They also fail the
  job: on 2026-11-13 run 43369452255 failed once in `f16_native_probe`
  (`crates/infr-metal/tests/gemv_bw.rs`) with "dequant-cached f32 weight would
  exceed the GPU working set: 4.64 GiB allocated + 0.28 GiB … > 4.67 GiB
  budget", and passed on a re-run with no code change, so the probe's memory use
  depends on the runner it lands on. The allocation total suggests buffers
  accumulating across the probes in that test binary (not verified). The fix is
  to stop running assert-nothing probes in the CI test job, or to bound their
  allocations.
- Not read by any pass: `kv_cache.metal`'s first f16 / f32 writers, the
  `topk_mask` and lightning-indexer tail of `attention.metal`, the rest of
  `elementwise_norms.metal`, most of `tests/parity.rs`, and
  `tests/kernel_names.rs`.

### B151 — MTP as parked: what un-parking it would hit first (2026-11-12)

`mtp_enabled()` returns `false` and every production entry point goes through it
(read 2026-11-12); `tests/cpu_backend.rs` still calls the public API. Found by
reading `mtp/`:

- **The MTP branch drops the `RequestCtx`.** `chat/{cpu,vulkan,metal}.rs` call
  the speculative generators without `req`, and `generate_mtp_spec_core` has no
  parameter for it — so abort (the stop-sequence latch), cancel (deadline,
  client hang-up), the `StepGate` baton and per-request sampling would all be
  lost. First bug on re-enable.
- **It cannot give output identical to plain decode** beyond the documented int8
  noise: candidates are computed at `m > 1` where decode uses `m = 1`; a qwen35
  rollback re-prefills committed rows at another batch size; and at `temp > 0`
  `spec_accept_stochastic` preserves the distribution but consumes the RNG in
  another order, so a seed gives different text.
- It reports `natural_stop: false` always; a snapshot failure aborts the whole
  generation where the decode loop degrades; its context is sized with no
  `reply_budget` or trained-context clamp; and `mtp_restore_delta`'s callers
  leave `cached` / `written_end` unchanged when a restore fails part-way
  (harmless today: the state is a local dropped on the error).
- Not read: `build_mtp_graph`, `build_mtp_draft_chain_graph`,
  `upload_mtp_head_bufs`, `load_mtp_head`.

### B152 — smaller findings from the 2026-11-12 CPU and seam reads (2026-11-12)

- **Dense Q5_0 on the CPU computes decode and prefill differently** (`m == 1`
  takes an f32 dot, `m > 1` the int8 `vec_dot_q5_0_32_batch`). Every other
  32-block format uses one kernel for both. The fix is one arm, and it moves the
  `cpu_golden_gemma3` hash: needs the owner's go-ahead to re-bless (same item as
  B125's Q5_0 bullet).
- **Suspected:** in `vulkan_moe_binder` a paged MoE layer whose expert bank does
  not arrive as `WBytes::Mmap` falls through to a full resident upload after the
  pager already budgeted for it. Not established whether any architecture hands
  over a non-mmap bank.
- **Suspected, low:** `turbo.rs` `quantize_block` stores `grp_norm / recon_norm`
  as f16, which overflows to infinity for a group whose L2 norm exceeds the f16
  range; and `dequant_prefix_orig` zero-fills when the length is not a multiple
  of its group (only a `debug_assert` on the write side).
- `pool.rs`: a panic payload whose own `Drop` panics would kill a worker and
  leave `run` spinning; `seq`, `cursor`, `done` and `n_tasks` share a cache line
  (B95).
- CPU kernels not compared with a reference: `turbo.rs` (llama.cpp's checkout
  has no counterpart).

### B153 — cargo 1.100's unused-dependency warnings (2026-11-13)

Rust 1.100 (local on the Windows box on 2026-11-13; CI was still on 1.99.1 the
day before) turned on `cargo::unused_dependencies`, and its clippy found
`needless_range_loop` and `duplicate_attributes` sites, fixed the same day. The
two manifest warnings are left, since neither fails `-D warnings`:

- `tower-http` in `infr-server`: nothing under `crates/` names `tower_http`.
  Removing it is the fix.
- `infr-metal` in `infr-llama`: used only from macOS code, so it is unused on
  Windows and Linux. A `[target.'cfg(target_os = "macos")'.dependencies]` entry
  would say so, if the code that names it is gated the same way everywhere (not
  checked).
- `bytemuck` and `infr-prof` in `infr-core` are reported only by
  `cargo clippy -p infr-core`, not by a `--workspace` run, so their use
  presumably hangs on a feature another crate turns on (not checked).

### B154 — the 2026-11-13 Windows run: what executed, what did not (2026-11-13)

The 2026-11-12 fixes were written on Linux. They were run the next day on the
Windows 11 dev box (Ryzen 9 9950X3D, AMD proprietary driver) with
`INFR_DEV=Vulkan1`, the integrated RDNA2 GPU, single-threaded. All of it passed:

- `cargo nextest run --workspace` and `cargo clippy --workspace --all-targets`.
- `infr-vulkan`: the library tests and every test binary under `tests/` except
  the benches and probes (`attn_dsplit_probe`, `attn_ktile_probe`,
  `bandwidth_probe`, `decode_gemv_bw`, `gemm_bench`, `interconnect_probe`,
  `small_m_bench`, `moe_id_gemv_real_dims`), with `--include-ignored`.
- `infr-llama`: `seam_op_parity` and `synthetic_deepseek2` with
  `--include-ignored`, the library tests, and from `cpu_backend` the tests whose
  model is cached: `cpu_golden_qwen3`, `cpu_golden_gemma3`, `cpu_golden_qwen35`,
  `cpu_golden_qwen3_quants` (the rows whose file is cached), the
  `cpu_kv_*_coherent` tests, the CPU paging tests, `gpu_seam_golden_qwen3`,
  `gpu_seam_flash_matches_cpu`, `gpu_seam_kv_mainline_quants_coherent` and
  `gpu_seam_kv_q8_coherent` (since replaced by `gpu_seam_kv_formats_track_cpu`
  and `gpu_seam_kv_q8_tracks_cpu`), `gpu_seam_kv_reuse_matches_fresh`,
  `gpu_seam_matches_cpu_gemma3`, `gpu_seam_matches_cpu_qwen2`,
  `gpu_seam_multi_slot_prefix_sharing`, `h_tap_matches_lm_head`,
  `qwen35_trunk_unaffected`, `unified_qwen35_gpu_seam_matches_cpu` and
  `unified_qwen35_session_no_rewind`.

Notes and gaps:

- **The 7900 XTX was run afterwards, the same day** (`INFR_DEV=Vulkan0`): the
  same `infr-vulkan` selection passed (293 tests by the runners' own totals),
  and so did the `infr-llama` selection apart from
  `gpu_seam_kv_mainline_quants_coherent`, which was B75's known failure. The
  multi-GPU tests (`expert_parallel_*`, `tensor_parallel_*`,
  `pipeline_matches_single_device`, `two_models_two_devices_concurrent`) were
  included there and reported ok; whether each executed or returned early was
  not checked. The benches and probes were not run on it, and the parked `mtp_*`
  tests were not run on either GPU.
- Three `infr-vulkan` library tests open every enumerated device (`multi.rs`,
  `pipeline.rs` and `adapter.rs` call `VulkanBackend::new_on` per index), so a
  library-test run touches the discrete GPU whatever `INFR_DEV` says.
- **Every model-gated test whose model is not cached.** They print `skip:` and
  count as passed under `--include-ignored`. That includes
  `gpu_seam_matches_cpu_llama`: it wants the Q8_0 file and the box has
  Llama-3.2-1B in Q4_K_M only.
- No performance figure was taken; other work was running on the machine.

### B155 — Vulkan streamed prefill and the default decode GEMV, 2026-11-13 read (2026-11-13)

A read-only pass over the Vulkan code the coverage list had as unread
(`streamed_prefill_gemm`, `native_sg_choice`, `native_rm_choice`, the
`native_gemm_fma.comp` body, the mmq shader headers). Nothing was built or run
by it. The `native_gemv_rm_v2` finding was re-read at its site afterwards; the
rest is the reviewer's account.

- **Streamed prefill GEMM (`streamed_gemm_route`): coverage gaps.**
  `streamed_gemm_route` now sends a streamed `Linear` past
  `STREAMED_GEMM_MIN_ROWS` to `nc_prefill_gemm` (the same `nc_gemm_choice` the
  resident arm uses). On a coopmat device with int8 dot it sends Q4_K to the
  arena-addressed mmq build wherever the resident `warp_ok` is false (`out_f` a
  multiple of 64 and not of 128, or any `out_f` under `INFR_NO_GEMM_WARP`). A
  weight that breaches the u32 addressing cap, or a grid past the x limit, keeps
  the chunked per-row GEMV (`streamed_tiled_fits`) rather than failing. Verified
  on the Radeon iGPU (`Vulkan1`, no coopmat, int8 dot) and the 7900 XTX
  (`Vulkan0`) by `tests/streamed_nc_gemm.rs` (kernel names after the streamed
  run, streamed == resident bitwise, against the GEMV and the host oracle, and
  the breaching-weight fallback with `bda_chunk_elems` lowered) and the
  `gpu_seam_dense_stream_*` tests with Qwen3-1.7B-Q4_K_M. Not verified: Intel
  Arc; a device without int8 dot (the fma arm only); a whole streamed model with
  bf16 weights; the oversized-grid fallback on a GPU (only the host-side
  predicate test covers it); the `INFR_CM_8X8` opt-in, where the resident Linear
  takes `_cm8` but the streamed one takes this tier (no arena-addressed `_cm8`
  build exists). Known divergence, debug knob only: under `INFR_NO_GEMM_WARP` a
  resident non-Q4_K quant still runs `native_warp_gemm` while the streamed one
  falls back to the GEMV. Open choice: streamed Linears at or below
  `STREAMED_GEMM_MIN_ROWS` take the per-row GEMV although
  `linear_native_mrow_at` / `linear_mmv_mrow_at` would read the arena by address
  too. Not measured in a quiet window: one profiled `infr bench -p 128` of
  Qwen3-1.7B on `Vulkan1` with `paging.cache=200m` and `INFR_PROF_OPS=1`, on a
  busy machine, gave 10.0 t/s with the old route and 103.6 t/s with the new one.
- **Correctness, conditional, confirmed by re-reading:
  `native_gemv_rm_v2.comp`'s `VARIANT_REG` sizes `reg_part` for subgroups of at
  least 16** (`THREADS / 16u` slots, written by `gl_SubgroupID`), and
  `Recorder::linear_native_rm_v2_at` builds it with `VulkanBackend::kernel`,
  which pins no subgroup size. `GemvCfg::variant` defaults to `"reg"`, so this
  is the default single-row Q4_K decode GEMV. On a device whose default subgroup
  size is below 16 the write is out of bounds. No such device is known to pass
  backend init (which demands a pinnable 32), so this is a latent assumption,
  and the kernel also ignores `caps.sg_pref`. Building it with
  `kernel_sg(..., self.sgp())` changes the subgroup size the default decode
  kernel runs at on AMD, so it needs a decode measurement first.
- **Suspected, not established:** `native_rm_choice`'s doc says large GEMVs stay
  on RM=1, while `rm_maxout` defaults to `usize::MAX`; the RM kernel is
  described as bit-identical, so this is a performance question at most.
  `native_gemm_mmq_nvfp4.comp`'s `ue4m3` uses `exp2()`, which GLSL does not
  require to be exact.
- **Tests that cover no production code** (reviewer's account):
  `deltanet_shared_capacity_boundaries`, the `routing_rejects_*` tests and most
  of `integrated_routing_and_bank_layout` in `tests/gemm_bench.rs`;
  `selected_profile_invariants` and `required_timing_rejects_incomplete_samples`
  in `tests/moe_id_gemv_real_dims.rs`. Nothing unit-tests the decisions of
  `native_sg_choice` or `native_rm_choice`, and the recorder's ignored GPU tests
  that need f16 coopmat print `skip:` and pass on a device without it.
- Read with nothing found: `streamed_prefill_gemm`, `native_sg_choice`,
  `native_rm_choice`, the `native_gemm_fma.comp` body, and the header blocks of
  the mmq shaders (not diffed against a llama.cpp checkout). Not reached:
  `matmul_native_off`, `matmul_native_splitk`, `matmul_native_f16a`,
  `matmul_native_f32a`, `dispatch_gemv_chunked`, and the ignored GPU test bodies
  in `recorder.rs`.

### B156 — pager survey leftovers: MoE touch costs, test-only GPU pager surface, MoE DRAM policy (2026-11-13)

Findings of the 2026-11-13 pager survey (`docs/plan/06-pager-eviction.md`, "What
the survey found") that are not fixed, each re-read at its site on 2026-11-13;
what is inference is said.

- **Split-path ids are not deduplicated.** On a small-m `MoeFfn` with an absent
  expert, `execute_paged_moe` (`adapter.rs`) downloads the router's ids buffer
  and passes all `rows × n_used` entries to `stage_and_window` as they are. A
  repeated expert is a same-batch VRAM hit (`moe_tier_parity`'s
  `a_repeated_expert_is_staged_once` pins that it reaches the host tier once),
  but each repeat is still a `Pager::touch` and counts as a hit in
  `paging.stats`, so the hit rate on that path is inflated by repeats. Only rows
  above one can repeat an expert (one row's top-k is distinct — inferred from
  top-k, not checked in the router shader). Listed with the plan's small items.
- **`touch_all_hits` promotes every expert of a fully resident layer.** The
  inline path calls `all_resident` (one `slot_of` per expert) for each role,
  then `touch_all_hits` (one MRU `Pager::touch` per expert) for each role, per
  layer per decode token. The cost is `n_expert` touches per layer and role, and
  recency loses which experts were routed: after such a token every expert of
  the layer is equally hot. The plan's "fully resident decode" item owns the
  redesign.
- **`HostPager::fill` clones the block descriptor on every call.** The first
  thing its loop does under the lock is `inner.descs.get(&id).cloned()`, a heap
  copy of the extent list, before the resident check — so hits pay it too.
  `fill` serves the dense host tier and the arena-less (unified-memory) MoE
  tier. Listed with the plan's small items.
- **Test-only GPU pager surface: `GpuPager::ensure_resident`, `flush_lut`,
  `lut_buffer` and the device LUT.** No production path calls them (the MoE
  execution path reads the LUT tape), but they are how the paged-kernel parity
  binaries fill an arena: `grid_mmq_parity`, `moe_id_gemv_new_formats_parity`,
  `moe_mmq_fp4_parity`, `moe_mmq_q2_0_parity`, `nc_gemm_parity`,
  `pager_gemv_multi_parity`, `pager_gemv_parity`, `pager_mmq_parity`. Deleting
  them means porting each fixture to `touch_staged` under a recorder plus a tape
  window. Meanwhile every `GpuPager` allocates and seeds a device LUT of
  `n_blocks` words that production never reads, and `MoePagerSession::lut` has
  no caller at all.
- **The MoE DRAM tier sees only VRAM misses, and what that costs is not
  measured.** Experts that keep hitting in VRAM age out of DRAM. That frees DRAM
  for experts VRAM does not hold, but an expert hot for a long stretch and then
  evicted from VRAM may have left DRAM too, so its next miss reads the disk
  (`disk-streaming-plan.md` §3.6). Which misses less depends on the routing
  trace; the plan's two-tier simulator is the instrument. One data point, from
  `infr run` on Qwen3-30B-A3B-Q4_K_M (7900 XTX, `paging.cache=4g`,
  `paging.dram=1g`, greedy, 32 tokens): the host tier, smaller than VRAM there,
  had no hits at all. A DRAM tier smaller than the VRAM arena may buy nothing
  for MoE; whether auto-sizing should skip it in that case is open.
- **`disk-streaming-plan.md` §3.5 describes a fault-injecting in-memory
  `BlockIo` in `infr-testkit`; none exists.** The tests that inject read
  failures define their own (`hostpager.rs`'s `FakeIo`, `moe_tier_parity`'s
  `BankIo`, `dense_tier_parity`'s `FaultIo`). A shared one is worth building
  when a fourth copy would be needed.

### Review coverage as of 2026-11-12

Eight further read-only passes on 2026-11-12 went through what the earlier
reviews had listed as unread. Each pass reported its own coverage; none built or
ran anything. Their findings are B146–B152, each saying what was confirmed at
its site and what is a reviewer's inference. What has now been read in full,
beyond the 2026-11-11 list:

- **infr-vulkan:** every `lower_op` arm with the recorder methods it calls;
  `tp.rs`, `tp_allreduce.rs`, `tp_sem.rs`, `p2p.rs`, `ep.rs`, `pipeline.rs`,
  `linear.rs`, `vkext.rs`, the non-test part of `caps.rs`; the logic of
  `build.rs` (its several-hundred-entry build list was checked by script).
- **Shaders:** every in-shader weight decoder against `ggml-quants.c`; the mmq,
  i8cm / f8cm, MLA, lightning-indexer, MoE, DeltaNet, turbo, KV store / quant /
  dequant, sampler, softmax and layernorm bodies; the flash, partial, decode,
  nc_fa, ktile and `attention_kv` attention kernels.
- **infr-llama:** every graph emitter in `seam/runner.rs` against its llama.cpp
  builder; `seam/mod.rs`, `seam/model.rs`, `seam/sc.rs`, `seam/dsv4_plan.rs`,
  `qwen35.rs`; most of `mtp/`.
- **infr-cpu:** every kernel family in `kernels.rs` across its SIMD tiers
  against `ggml-quants.c`; `turbo.rs`; `pool.rs`.
- **infr-core:** `iquant_grids.rs` (compared with `ggml-common.h` by script,
  element by element) and `config/manifest.rs`.
- **infr-cli, infr-hub, infr-chat, infr-plat:** production code in full.
- **infr-metal:** `pcache_blob.rs`, `profile.rs`, `common.metal`, the weight
  decoders, the GEMV / GEMM bodies, the KV writers and dequant kernels, the
  attention kernels, the sampler stages, `moe.metal`, `rope_ffn.metal`,
  `embed_gather.metal`.

Still not read by any pass, stated as a gap:

- **Tests, mostly.** The test code was swept for specific patterns and partly
  read. Not read end to end: most of `infr-metal/tests/parity.rs` and
  `kernel_names.rs`; the non-ignored host tests in
  `infr-vulkan/tests/ gemm_bench.rs` and `moe_id_gemv_real_dims.rs`; most of
  `tests/synthetic_deepseek2.rs`; the bodies of the non-ignored `seam_op_parity`
  tests; the `#[cfg(test)]` modules of `seam/runner.rs`, `seam/weights.rs` and
  `recorder.rs`; `infr-hub/src/testhttp.rs`; `infr-plat`'s Windows-job native
  tests; the infr-server tests beyond a grep.
- **Vulkan:** `matmul_native_off`, `matmul_native_splitk`, `matmul_native_f16a`,
  `matmul_native_f32a` and `dispatch_gemv_chunked` (what this list named before
  was read on 2026-11-13; see B155).
- **Metal:** the first f16 / f32 writers in `kv_cache.metal`, the `topk_mask`
  and lightning-indexer tail of `attention.metal`, the rest of
  `elementwise_norms.metal`, the DeltaNet recurrence in `deltanet.metal`.
- **infr-llama:** the MTP graph builders and head loader; the DiffusionGemma
  emitters against an actual reference (the fork checkout is not on the dev
  machine).
- **References that could not be checked:** NVFP4, Q2_0 and the I2S layout in
  `infr-gguf` (no reference to hand); `turbo.rs` (llama.cpp has no counterpart);
  the 331 llama.cpp commits between the pin the docs name and the checkout
  (B-DSHW-PULL).
- **Nothing was run on:** an Intel or NVIDIA GPU, a multi-GPU world beyond
  opening one, or a Metal device outside CI. Windows was run on 2026-11-13; see
  B154 for what that covered.

### B157 — SmolLM3 decodes without the record-once replay (2026-11-13)

**Tag:** vulkan · metal · seam · performance

The replay gate in `generate_dense_backend`
(`crates/infr-llama/src/seam/runner.rs`) no longer carries correctness: after
compiling the position-0 decode plan the runner asks
`Backend::decode_replay_ready` and takes the per-token path on a "no", and the
graph is marked `Graph::replay_only`, which Vulkan, Metal and CPU refuse to run
in a way that would read its baked positions. The gate's `mirror_admits` terms
are now only a way to skip building a plan that would be refused. What is left:

- **SmolLM3 still decodes per token on Vulkan**, recording each token's graph
  statically (`gpu_seam_smollm3_decode_records_no_replay` asserts it records no
  replay). Seen on 2026-11-13 on a busy machine, not a measurement: 217 tok/s
  decode on Vulkan0 and 11 tok/s on Vulkan1 (`infr bench -p 0 -n 64`), against
  780 and 55 for Qwen3-0.6B. Closing it needs either a `_dyn` twin for the
  standalone f16→f16 `WriteKv` (the K write a NoPE layer makes) or a fuse that
  writes the cast K straight into the cache the way `plan_kv_write` does for
  rope → `WriteKv`. llama4's rope layers have the same need, for their L2-normed
  K. Then drop the `c.no_rope_step == 0` / `!c.llama4` gate terms.
- **Metal's answer is not run on Metal.** `decode_replay_ready` there is true
  when the tape replays the plan (`replay_capable`) or the static walk takes
  every position from the bound positions buffer
  (`policy::static_walk_live_positions` in `crates/infr-metal/src/policy.rs`, an
  exhaustive `match` over `Op` that is unit-tested on every OS through
  `crates/infr-metal/tests/policy.rs`). The rest (`static_walk_live`,
  `refuse_replay_only`, the `execute_graph` / `execute_graph_chain` checks) was
  cross-linted for `aarch64-apple-darwin` only; no Metal device has executed it.
  The predicate's per-op decisions were read off the walk's arms in `exec.rs`
  (`dynpos` overriding single-row `WriteKv` / `Attention`, `Rope` loading its
  positions per execute, `QkNormRope` reading the bound buffer), not observed on
  a device.

### B158 — chat-template renders: what still differs from llama.cpp (2026-11-13)

**Tag:** chat · correctness

`build_env` (`crates/infr-chat/src/template.rs`) turns on `trim_blocks` and
`lstrip_blocks`, as llama.cpp's lexer (`common/jinja/lexer.cpp`) does for every
chat template. Checked by rendering each cached GGUF's template (Qwen2.5-0.5B,
Qwen3-0.6B / 1.7B / 30B-A3B, Qwen3.5-0.8B, gemma-3-1b, Llama-3.2-1B, SmolLM3-3B)
for seven conversations (user; user without thinking; system + user; user /
assistant / user, with and without thinking; user with one tool; the SmolLM3
golden's) through infr and through llama.cpp's `test-chat-template --no-common`
with the same variables. Every render matched byte for byte except the tool
renders. Hugging Face transformers was not run (not installed here).

`tojson` now writes what llama.cpp's does (`crates/infr-chat/src/tojson.rs`).
Rechecked the same way on the same templates with four tools conversations (one
or two tool definitions with nested parameters and a description holding
non-ASCII, `<`, `&` and both quotes; an assistant tool call in history; a tool
result; one with scalar arguments): every render matches except the first item
below. Gemma 3's template raises "Conversation roles must alternate" on a tool
message in both engines. What is left:

- **Turning a non-string into text.** Most ways a template turns a bool, none,
  float or container into text differ between the engines (integers, and floats
  in fixed notation, match); rendered through both (llama.cpp
  `test-chat-template --no-common`, infr `render_template`):

  | construct                          | llama.cpp                                 | infr (minijinja)                      |
  | ---------------------------------- | ----------------------------------------- | ------------------------------------- |
  | `true \| string`, `none \| string` | `True`, `None`                            | `true`, `none`                        |
  | `'x' ~ true ~ none ~ 2.5`          | `xTrue2.5` (none prints nothing)          | `xtruenone2.5`                        |
  | `'{} {}'.format(true, none)`       | `True None`                               | `true none`                           |
  | `{{ true }}`, `{{ none }}`         | `True`, empty                             | `true`, `none`                        |
  | a float `1e-7` from the input      | `\| string`: `1e-07`; bare `{{ }}`: `0.0` | `0.0000001` both ways                 |
  | `{{ [1, 2] }}`, `{{ {'a': 1} }}`   | `12`, empty                               | `[1, 2]`, `{"a": 1}`                  |
  | a dict `\| string`                 | `{'a': {}, 'b': [1, 'x'], 'c': True}`     | `{"a": {}, "b": [1, "x"], "c": true}` |
  | `[true, none] \| join(',')`        | error (joins only strings and numbers)    | `true,none`                           |
  | `'%s %s' \| format(true, none)`    | `%s %s` (nothing substituted)             | `true none`                           |

  Inside a container `| string` also writes a float `1e-7` as `0.0` and `1e16`
  as `10000000000000000.0` in llama.cpp.

  llama.cpp's `string` and `~` follow Python's `str()` (`value_to_string_repr`
  in `common/jinja/value.cpp`: single quotes, a string holding `'` in double
  quotes, `True` / `None`), but its bare interpolation does not: it prints
  nothing for none, a dict or a mapping inside a list, concatenates a list's
  items, and prints a small float as `0.0`, none of which Jinja2 does. So exact
  parity is not a formatter override (`Environment::set_formatter`) plus a
  `string` filter: it would copy those interpolation quirks, which look like
  llama.cpp bugs. Not built; it needs its own decision (match `str()` where
  llama.cpp does, and leave bare interpolation alone, or copy all of it). Cached
  templates that reach it: Qwen3.5 prints each scalar tool-call argument with
  `args_value | string`, so a bool, none or exponent-range float argument
  renders differently (verified for `true`, `none` and `1e-7`); SmolLM3 prints
  `tool | string` for `xml_tools`, which infr never passes (B159). Qwen3.5's
  `'Picture ' ~ image_count.value` concatenates an integer, which matches. No
  cached template interpolates a non-string bare.

- **`tojson` divergences kept on purpose.** Integers outside the signed 64-bit
  range: llama.cpp wraps them (`[18446744073709551615, 9223372036854775808]`
  prints `[-1, -9223372036854775808]`), infr prints the true values. The indent
  bound: `tojson(indent=2000)` renders in llama.cpp and is refused by infr
  (`TEMPLATE_WIDTH_MAX`, the guard against a template-sized allocation).
- **Tools and messages are passed as the request wrote them.** llama-server
  rebuilds each tool as `{type, function: {name, description, parameters}}`
  (`common_chat_tools_parse_oaicompat` then
  `common_chat_tools_to_json_oaicompat` in `common/chat.cpp`): other keys
  dropped, `description` filled with `""`, keys in that order. infr hands the
  request's `tools` to the template as they came, so a request with other keys
  or another key order prints a different tool block. Messages are built as
  `role, content, reasoning_content, name, tool_call_id, tool_calls` there and
  `role, content, tool_calls, tool_call_id, name` in `message_to_json`, which
  shows only in a template that `tojson`s a whole message (none of the cached
  ones). Read in the code, not rendered. A tool-call `arguments` string that is
  not JSON stays a string in infr (`parse_oai_tool_calls` in `infr-server`);
  llama.cpp refuses it for templates that take object arguments.
- **Variables.** infr always passes `tools` (null without tools) and
  `add_generation_prompt`; llama.cpp leaves `tools` out when there are none and
  `add_generation_prompt` out when false. Only Llama 3.2's template asks
  `tools is defined`, and it then sets `tools = none`, so it renders the same;
  rechecked by rendering a tool-less conversation with `tools` left out on the
  reference side through every cached template: all match.
- **Lexer edge cases, none copied.** Each was read in both lexers
  (`common/jinja/lexer.cpp`; minijinja 2.21.0 `src/compiler/lexer.rs`) and
  rendered through both engines. None is expected in a real chat template; every
  cached template starts with a tag and has no `\r`. Copying the first two would
  mean copying llama.cpp bugs.
  - **First text before the first tag.** llama.cpp's `lstrip_blocks` walk drops
    the template's first text when it reaches offset 0 over spaces or tabs, or
    when that text is a single character: `A{% if true %}B{% endif %}` and
    `A  {% if true %}B{% endif %}` render `B` there, `AB` and `A  B` in
    minijinja; `\n{% if true %}A{% endif %}` renders `A` there, `\nA` in
    minijinja.
  - **Unicode whitespace before a line-start tag.** minijinja strips it
    (`char::is_whitespace`); llama.cpp does not (byte-wise `isspace`):
    `A\n　{% if true %}B{% endif %}` renders `A\nB` in minijinja and keeps the
    U+3000 in llama.cpp.
  - **A lone `\r` line break.** llama.cpp turns `\r\n` and `\r` in the source
    into `\n` before lexing; minijinja keeps them and does not treat a lone `\r`
    as a line start for `lstrip_blocks`: `A\r  {% if true %}B{% endif %}`
    renders `A\nB` in llama.cpp and `A\r  B` in minijinja.
  - **`{%+` / `+%}`.** minijinja reads them (keep the whitespace at that tag);
    llama.cpp's lexer has no such case, and `test-chat-template` exited with no
    output on `{% if true +%}\nA{% endif %}` (minijinja: `\nA`).

### B159 — SmolLM3's tool definitions are dropped silently (2026-11-13)

**Tag:** chat · tools

`render_template` (`crates/infr-chat/src/template.rs`) passes tool definitions
as `tools`. SmolLM3's template reads `xml_tools` and `python_tools` and never
`tools`, so a request with tools renders exactly as one without them, and the
model is never told they exist. Nothing errors.

llama.cpp does the same: nothing in `common/` names `xml_tools` or
`python_tools`, and no specialized handler matches SmolLM3, so its own render of
a tools request (checked through `test-chat-template` with the common path) also
has no tool block. Mapping `tools` to `xml_tools` would therefore depart from
the reference, and is more than a variable:

- The template prints each tool with `tool | string`, Python's `str(dict)` in HF
  (single quotes), which minijinja's `string` filter does not reproduce.
- The template has no `tool_calls` branch for assistant messages, so a replayed
  call is lost from history; tool results go through its `tool` role branch.
- Replies come as `<tool_call>{"name": …, "arguments": …}</tool_call>`, the
  Hermes shape `parse_hermes_tool_calls` reads; not run against SmolLM3 output.

Options: refuse tools for a template that names `xml_tools` but not `tools` (an
honest error), or build the mapping with those three pieces and a test against
an HF render.

### B160 — deepseek2 loader: what the reference-parity fixes did not reach (2026-11-13)

**Tag:** deepseek · config

`Config::from_gguf` now ignores `rope.scaling.yarn_beta_fast` / `_slow` for
`deepseek2` and `deepseek32`, picks sigmoid for an absent `expert_gating_func`
on GLM 4.7 Lite's shape, and refuses `deepseek32.expert_gating_func = 0`, as
llama.cpp's loaders do (for the last: llama.cpp keeps the 0 and `build_moe_ffn`
in `llama-graph.cpp` hits `GGML_ABORT` on it, so it cannot run such a file
either). Covered by config tests on synthetic fixtures only. No real `deepseek2`
GGUF is cached here, so none of it was run end to end. Left:

- **GLM 4.7 Lite files converted with MTP are refused at load.** llama.cpp's
  converter (`Glm4MoeLiteModel` in `conversion/glm.py`, arch `deepseek2`) writes
  `nextn_predict_layers` and counts those layers in `block_count` unless run
  with `--no-mtp` (or the HF config has no `num_nextn_predict_layers`). infr
  refuses `nextn_predict_layers` on `deepseek2` (only qwen35 and `deepseek32`
  handle the split), so only `--no-mtp` or older conversions reach the new
  gating rule. llama.cpp applies the rule to the trunk depth, loads the NextN
  layers with `TENSOR_SKIP` unless MTP is requested (`nextn_flags` in
  `llama-model.cpp`), and its text-generation graph runs only the trunk
  (`il < n_layer` in `deepseek2.cpp`). Supporting these files for plain
  generation therefore looks like what infr already does for `deepseek32`: split
  the trunk off and skip the NextN blocks. A small follow-up slice, not started;
  read in the code, not run.
- **YaRN betas in real files.** Converters do write them for `deepseek2` files:
  `set_mistral_config` (`conversion/mistral.py`, used by `MistralMoeModel` on
  arch `deepseek2`) writes the Mistral `yarn.beta` / `yarn.alpha`, and
  `TextModel.set_gguf_parameters` (`conversion/base.py`) writes HF
  `rope_scaling.beta_fast` / `beta_slow` when present. So files with non-default
  values plausibly exist, and infr now follows llama.cpp in ignoring them.
  Whether that matches the HF reference implementation of those models (which
  presumably uses the config's values) was not checked.

### B161 — the SmolLM3 golden runs nowhere by default, and its first case is a near tie (2026-11-13)

**Tag:** tests · ci

`cpu_golden_smollm3` (`crates/infr-llama/tests/cpu_backend.rs`) is `#[ignore]`d
because the `cpu-goldens` CI job downloads only Gemma 3 1B and Qwen3 0.6B. Its
two 32-token hashes were captured on Windows only; 32 is the length the other
goldens keep OS-stable, but no Linux run has confirmed it for this model. To run
it in CI: add `unsloth/SmolLM3-3B-GGUF` `SmolLM3-3B-Q4_K_M.gguf` (about 1.9 GB)
to that job's download list, bump its cache key, drop the `#[ignore]`, and wrap
a hash in `per_os` if the Linux run differs.

Against llama.cpp, measured 2026-11-13 on this Windows box, CPU, on identical
prompt ids (llama.cpp's own 39-token render, which infr's now matches byte for
byte): the ocean case matches for all 32 tokens. The France case matches for 7
tokens (`The capital of France is Paris.`); infr then ends the turn, and
`llama-completion` continues `Paris is a city known for …`. The 8th token is a
near tie between end-of-turn (128012) and ` Paris` (12366), and llama.cpp itself
picks differently depending on how it gets there:

| engine and path, over the same 46 ids                    | 128012 |  12366 | margin               |
| -------------------------------------------------------- | -----: | -----: | -------------------- |
| infr CPU prefill (raw logits)                            | 29.185 | 29.139 | end-of-turn by 0.047 |
| llama.cpp `llama-server` prefill (logprobs)              | −0.847 | −0.911 | end-of-turn by 0.064 |
| llama.cpp `llama-server` token-by-token decode (logprob) | −0.970 | −0.818 | ` Paris` by 0.152    |

A logprob difference is a logit difference, so the margins compare directly.
Third place is ` It` (1102) in every row, about 1 logit further down. infr's
decode path, the one the golden runs, was not dumped; it ends the turn. The infr
row came from `cpu_prefill_matches_llama_debug_dump` fed a hand-made token file
(`llama-debug` does not parse special tokens in `-p`) and an unrelated logits
file, so only its printed infr top-5 means anything, and that run fails the
test's comparison by construction. The golden is kept: the flip sits inside
llama.cpp's own prefill-versus-decode spread.

### B162 — `cpu_golden_qwen3moe` fails on this Windows box: suspected OS difference (2026-11-13)

**Tag:** tests · moe

`cpu_golden_qwen3moe` (`crates/infr-llama/tests/cpu_backend.rs`, `#[ignore]`d,
runs nowhere in CI) fails here: got `0xe1a4ed6cecc5eb14`, want
`0xbc4f22b22d3e3c1d`. What is established:

- This golden had never run on Windows before: the Qwen3-30B-A3B GGUF was first
  downloaded to this machine on 2026-11-13. The hash was set in `e252233a`
  (2026-07-05), recorded by review as a Linux capture.
- The chat-template change is not the cause: the same hash comes out with
  `trim_blocks` / `lstrip_blocks` off, and this model's template renders
  identically either way.
- The output is coherent. The test prints only today's text,
  `<think>\nOkay, the user is asking "The capital of France is". I need to provide the correct answer. Let`;
  the comment above `QWEN3MOE_GOLDEN` records the blessed one as
  `<think>\nOkay, the user is asking, "The capital of France is". …`. The first
  difference is the token after ` asking`: `,` when blessed, ` "` today.

Suspected cause: the Windows-versus-Linux greedy difference the doc on `per_os`
describes, which is why the other long goldens carry a separate Windows hash.
Not excluded: the GGUF on the Hub was re-uploaded after the hash was taken.
Likely fix: once a Linux run confirms `0xbc4f22b22d3e3c1d` still holds, wrap it
as `per_os(0xe1a4ed6cecc5eb14, 0xbc4f22b22d3e3c1d)`. Re-blessing on Windows
alone would break Linux.
