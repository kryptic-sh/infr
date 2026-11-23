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

### B-RELAND — what the upstream sync left behind (2026-11-17)

**Tag:** re-land · **Needs:** work, per item

`docs/plan/reland-fork-features.md` records the whole replay. These items did
not come back, each for its own reason:

1. ~~Forward paged attention in the seam (`kv.paged`)~~ — **landed**: `KvPool`
   + `PagedSlot` live in `seam/pool.rs`, the pool is built at the first fork
   (`SlotPool`) or at boot (`--parallel`), and every pooled cache carries a
   `PageMap` into its graph. Two bugs the port exposed are fixed with it: the
   `Op::Rope` peephole wrote a pooled slot's K rows at row 0 (it applied no
   row-base bias, unlike the `QkNormRope` arm), and the record-once V store
   baked an unbiased address. `tests/kv_paged_session.rs` pins a two-conversation
   session byte-for-byte against a contiguous one.
2. **Store kernels' params-based row base (`p_rb`).** `store_f16.comp` and
   `quant_q8.comp` derive the write row from `p_pos` alone; a paged slot's row
   base travels in the destination device address instead
   (`adapter::kv_row_base * row_bytes`). The test that pinned `p_pos + p_rb`
   (`a_store_row_base_puts_the_row_in_that_block`) is `#[ignore]`d with that
   reason, because the params word never landed in any store kernel.
3. ~~fp32 K-quant prefill GEMM~~ — **landed** (`native_gemm_kq_fma.comp` + the
   split-K twin, `NcGemm::KqFma`).
4. **Small per-kernel tuning the replay dropped**: mmv OUTS4 K-tail
   (`eea45032`), small-turn attention + K-quant linears (`05eb5b55`), compare
   benchmark overrides (`c79c4034`).
5. **Fused RmsNorm consumers.** `infr-core/src/fusion.rs` now plans only
   `Op`-level fusions; `plan_rmsnorm_linear` / `FusionPlan::rmsnorm_linear` (and
   `446eb403`'s input-unchanged fix) are gone with upstream's pattern. Restoring
   it needs a Vulkan (and Metal) planner again.

The GCN tiers that landed (`kq_fma_gemv`, `wave64_native`, `kq_fma` prefill)
still owe the RX 580 run `docs/legacy-gpus.md` describes — the first two have
been spot-checked there, the prefill GEMM has not.

### B-DSV2-NONYARN-SCALE — deepseek2 ignores a non-YaRN `rope.scaling.factor` (2026-11-16)

**Tag:** deepseek · **Kind:** suspected divergence, unverified

Found while reading the reference's rope-scaling defaults for `deepseek4` (plan
09, S1). llama.cpp's generic `llama_model::load_hparams` defaults
`rope.scaling.type` to `"linear"` and turns any `rope.scaling.factor` into
`freq_scale = 1/factor`; `llama-context.cpp` then gives a `"linear"` file
`ext_factor = 0`, which in ggml's `rope_yarn` is a uniform angle scale by
`freq_scale`, and a `"none"` file `freq_scale = 1`. infr's `deepseek2` parse
keys everything on `rope.scaling.type == "yarn"`: a `deepseek2` file that
declares a factor with type `"linear"` (or no type) gets no scaling at all in
`seam::runner`'s `yarn_ff`. `deepseek4` now handles all three types
(`Config::dsv4_compressed_rope_divisors`). The score scale would differ too:
`deepseek2.cpp` derives its attention factor and the squared magnitude scale in
`kq_scale` from `freq_scale` whatever `ext_factor` is, and nothing cancels them
when `ext_factor` is 0, while infr's non-YaRN `mla_scale` is the file's
`rope_attn_factor` alone.

Not verified: that `deepseek2.cpp`'s rope actually receives `cparams`'
`freq_scale` on that path (its rope calls were not read), and whether any
published `deepseek2` file declares a factor without YaRN — every one this
project has opened declares `"yarn"`. Closing it means reading those calls, then
either mirroring the `deepseek4` parse or refusing such a file.

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
- **The reader's concurrency, by platform.** `FileBlockIo` splits a block across
  `IO_FANOUT` concurrent positioned reads, measured 1.2-1.5 → 2.2 GB/s on Linux
  on the Samsung 980. On Windows a handle opened for synchronous I/O runs its
  requests one at a time, so since 2026-11-16 every shard is opened `IO_FANOUT`
  times and reads take the handles in turn: with random 2.75 MB reads of the 98
  GB V4 file, sixteen threads on one handle read 0.65 GB/s (one thread's speed)
  and four threads on a handle each 1.05 GB/s, eight or sixteen no more
  (standalone bench). Untested on macOS. A rotational disk is also untested and
  is the one case where the concurrency could plausibly HURT (seek
  interleaving); nothing in the code adapts to device type.
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
  neither has been benchmarked against a cap. **The rule changed on 2026-11-16**
  from leaving an eighth of available memory to leaving half (and, on the CPU,
  pricing the run's KV cache, unpaged weights, a 1 GiB activation reserve and
  the repack cache first): an eighth sized a 38.47 GB arena on the Windows box
  and starved it (§ B-DSV4-REAL). The Linux 7.44 GB / 0.38 t/s figure above was
  taken under the old rule and was not re-measured; under an 8 GB cap the new
  rule takes about half as much, for the Vulkan tiers too. The trade on the GPU
  side, measured on this box by forcing the two rules' budgets for a 9
  GB-available machine on Qwen3-30B-A3B Q4_K_M (RX 7900 XTX, `paging.cache` 2
  GiB): decode 7.96 → 7.53 tokens/s, host-tier hit rate 53-55% → 37-39%. Run
  automatically on this box both rules size the whole 17.55 GB of experts, so
  comparing against the old tree there would measure nothing.
- **The unified-memory path's sizing is unverified.** An iGPU/APU streams
  `DISK → GPU-accessible RAM` with no host cache (`HostPager::stream_only`,
  selected by `DeviceCaps::unified_memory`). The MECHANISM is covered on a
  discrete GPU by `INFR_DRAM_BYPASS` — the dense leg in `dense_tier_parity`
  content-checks it and the MoE leg in
  `gpu_seam_paged_moe_host_tier_matches_resident` is token-identical. The
  SELECTION is seen on the Ryzen 9950X3D's Raphael iGPU (RADV): `unified_memory`
  is set there and the banner prints `UNIFIED MEMORY` (run). What is NOT covered
  is the sizing: nothing sizes `paging.cache` (which on such parts comes out of
  shared RAM) against HOST memory the way `paging.dram` is. Streamed models DO
  run on the Windows box's iGPU (`Vulkan1`, 2026-11-15, the arena-less
  `stream_only` host tier selected automatically): ERNIE-4.5-0.3B and LFM3-1.7B
  F16 at `paging.cache=200m` are token-identical to resident
  (`gpu_seam_dense_stream_f16_matches_resident`), and the GPU gate filter
  (`test(gpu_seam) | test(stream) | test(tier) | test(paged)`, the Qwen3-1.7B
  `gpu_seam_dense_stream_*` tests among them) passes there; ERNIE-4.5-0.3B F16
  measured pp512 271.1 and tg32 11.9 t/s streamed against 295.7 and 37.4
  resident. Left: the feature (a host-memory term in the arena sizing). Metal
  has no pager until phase 4, so it inherits none of this yet.
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
  - **Only the dense Vulkan path, and DeepSeek V4 on the CPU's host arena**
    (since 2026-11-17, `Backend::host_streams`; § B-CPU-EXPERT-PAGING). Vulkan's
    MoE expert paging, MTP, the qwen35/DeltaNet bespoke path, Metal and every
    other family on the CPU keep chunk-major: the gate is `Backend::dense_paged`
    (or `host_streams` with `seam::host_layer_major`), and the E2B arch is
    refused by `layer_major_prefill`'s `spannable` arm (its `per_layer_inp` is
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

### B-SYNTH-GPU-DEVICE — whole-model synthetic GPU tests ignore `INFR_DEV` (2026-11-16)

**Tag:** test coverage · **Blocked on:** nothing

`crates/infr-llama/tests/synthetic_deepseek2.rs` loads every model through its
`load` helper with `EngineConfig::default()`, and
`SeamModel::prefill_logits_vulkan` builds its backend from that config, so
`INFR_DEV` is never read: the whole-model GPU tests there
(`gpu_synthetic_deepseek4_*`, `gpu_synthetic_deepseek2_*` and the rest) have
only ever run on the default device (the discrete GPU here), never on the
integrated GPU, whatever the run was told. The op-level tests in
`seam_op_parity.rs` do honour it (they build `VulkanBackend::new()`, which reads
the environment). Verified 2026-11-16: a capture run with `INFR_DEV=Vulkan1`
produced logits bit-identical to Vulkan0 through `load`, and different ones once
the config came from `EngineConfig::load_from_env()`. Fix: build the test config
with `EngineConfig::load_from_env()` (or copy just the device field from it) in
`load`, and check the other synthetic test files for the same helper.

**Also open:** S2's ratio-0 logits check was done once by hand (see
`docs/plan/09-deepseek-v4-compressed.md` § 3); a permanent V4 lock needs
llama.cpp to bless the synthetic file, per `docs/synthetic-models.md` rule 2.

### B-DSV4-WIRING — what the V4 graph slice still owes (2026-08-10)

**Tag:** CR-2026-08-09 deepseek · **Blocked on:** nothing

**Slice A (ratio 0) is DONE and generates** — see `docs/deepseek.md` § Stage 4,
"Slice A". What is left:

**Slice B — ratios 4 and 128.** Both compressed tiers are DONE on the CPU (plan
09 S5 to S7, 2026-11-16): the compressor emit (`seam/dsv4_emit/`), the CSA
tier's lightning indexer, the persistent caches and rings
(`seam::dsv4_comp_bufs`, `SeamKv::cbufs`), the `wpush` handles and the no-rewind
predicate (`Config::has_unrewindable_state`), matched to llama.cpp's logits on
the synthetic all-HCA, all-CSA and every-tier files (`docs/deepseek.md` § "The
HCA tier" and "The CSA tier", plan 09 § "S5 and S6" and § S7 for the numbers and
the breaks). What is left:

- **The shipped V4-Flash file's numbers.** It runs on the CPU since 2026-11-16
  and generated coherent text for two prompts (§ B-DSV4-REAL), but no oracle has
  checked a number it computes: llama.cpp cannot load it on this machine. Its
  geometry (43 layers: 2 ratio-0, 21 CSA, 20 HCA; head 512 with a 64-dim rope;
  indexer 64 heads × 128, `top_k` 512; window 128) is read off the first shard's
  metadata. The comparison is plan 09 S8.
- **The LID cache's width under the KV formats.** The compressed caches take the
  raw cache's format (`check_attention` demands one dtype across segments).
  `kv_row_align_ok` holds every compressed row to the 32-element blocks, and
  TurboQuant's 128-element groups are now held to every compressed row too
  (`kv_turbo_align_ok`). No quantized KV format has been run on a V4 file (§
  below on the rotation).
- **GPU backends.** A compressed layer on Vulkan or Metal is refused by name
  (`gpu_synthetic_deepseek4_refuses_a_compressed_layer_on_vulkan`). Nothing in
  the HCA graph has been run on a GPU: `Op::GatherRows`, `Op::CompressPool`,
  `Op::EmbedGather` over an f32 APE table, the ring persisted by `Op::Copy` into
  an f32 Input, and the two-segment attention all exist on Vulkan, but no
  GPU-vs-CPU test of the whole tier exists (plan 09 S9).
- **Batched prefill is the CPU's only.** Since 2026-11-16 the CPU prefills a V4
  prompt in chunks (`dsv4_emit::batched_prefill_ok`; plan 09 § 5 "Batched
  prefill"), bit-identical to one position per graph on every synthetic layout.
  A GPU backend still prefills one position per graph: the compressed tiers are
  refused there anyway, and a multi-row ratio-0 V4 graph (the hyper-connection
  ops, `Op::GatherI32`, the sinks attention at `rows > 1`) has never run on
  Vulkan. Lifting it there is part of plan 09 S9, with a GPU-vs-CPU test of the
  chunked graph.
- **Chunked and one-position prefill differ on int8-path expert banks.** The
  CPU's `Op::MoeFfn` sets `int8_ok = rows >= 2 && !reference`, so a chunk feeds
  Q4_K, Q5_K, Q6_K and Q8_0 expert banks int8-quantized activations
  (`expert_acts_kind`) where a one-row graph (one position per graph, and every
  decode step) feeds them f32; the difference reaches the routers, the indexer's
  selections at near-ties and the logits. Measured on the synthetic file with
  Q8_0 banks
  (`quantized_banks_chunked_prefill_is_close_and_reference_is_exact`): 1.71e-4
  at the prompt's end, 2.37e-4 at the first decoded position, 87 of 594
  selections flipped; bit-identical in reference mode. Every MoE model's CPU
  batched prefill makes this trade; it is recorded, not a defect. The shipped
  V4-Flash file's routed banks are all Q2_K (read from its tensor-info header),
  which stays on f32 activations at any row count, so it does not reach that
  file's experts; the synthetic file in the shipped dtypes is bit-identical
  chunked and one position per graph
  (`shipped_dtypes_chunked_prefill_equals_one_position_per_graph`). Not checked
  on the real file itself: a long prompt there at `INFR_UBATCH=1` against the
  default (§ B-DSV4-REAL: a 5-token prompt alone takes 16 s).
- **Quantized KV caches round differently from llama.cpp's.** At a quantized
  `type_k` with a head width divisible by 64 (q8_0, q4_0 on V4-Flash's 512),
  llama.cpp's `llama_kv_cache` rotates q, the raw row, the compressed row and
  the attention output by a Hadamard matrix (`attn_rot_k`); infr quantizes the
  unrotated rows. Exact at f16 (neither cache is rotated there), equal in exact
  arithmetic otherwise, but the quantization error differs. To measure: the
  `hca` oracle case at `INFR_KV_TYPE_K=q8_0` against llama.cpp at
  `--cache-type-k q8_0`, which needs a fixture head width divisible by 64 (the
  oracle fixture's is 48). Not attempted.
- **One sequence per session.** `build_dsv4_comp_plan` refuses a ubatch of
  several sequences (as the reference does for one stream) and ports none of the
  rollback (`n_rs_seq`) planes, so a V4 session cannot roll back a speculative
  draft; it re-prefills instead. Parallel serving forks one session per slot,
  which this covers. A multi-stream graph would need per-sequence ring and cache
  offsets and the per-sequence block padding.
- **The rings round-trip through the CPU interpreter's store every step.** A
  ring is an f32 Input mutated by `Op::Copy`, so the interpreter loads it whole
  and writes it back after each graph (`infr_core::exec::writes_back`): 512 KiB
  per HCA layer at V4-Flash's head width, both ways, per token. Not measured
  against the rest of a V4 decode step. A CSA layer has two rings (8 rows of
  1024 and of 256 f32 on V4-Flash, kv and score), a far smaller round trip.
- **Per-step inputs are allocated per step.** `upload_step_i32` allocates and
  uploads one small staging buffer per V4 compressor input per graph; a decode
  that kept them across steps would save the allocations. Not measured.
- **The reset does not zero the compressed caches**, where llama.cpp's
  `clear_compressed` does. Every row a graph reads was written since the last
  reset by the one contiguous sequence the session holds (an extending turn
  reads the rows earlier turns wrote; every other turn resets), a compressed row
  is read only once committed, and zeroing a CSA cache costs a context-sized
  upload per new prompt (`runner::zero_unrewindable_state` has the argument).
  Decided, not an oversight; revisit if a path ever reads a compressed row past
  a row's own `(pos+1)/ratio`, or resumes a session without that reset.

**What slice A left unverified, in its own right:**

- **`batch > 1` runs on the CPU only** (see "Batched prefill is the CPU's only"
  above). The fixture's f32 expert banks fail `moe_batched_ok`, which the V4
  gate does not consult on the CPU, whose `Op::MoeFfn` takes any dtype at any
  row count.
- **A partial layer span carries the widened stream since 2026-11-17**
  (`DecodeHandles::hc_resid`, `docs/deepseek.md` § "Layer-major prefill on the
  CPU"), so layer-major prefill serves V4 on the CPU. Not run on any GPU
  backend: V4's batched prefill is the CPU's only.
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
- **`Op::Attention { comp }` on Vulkan** (2026-11-16) is `attention_kv.comp`'s
  four `-DCOMP` builds (with and without sinks and the segment bias): f16 raw
  and compressed caches, head dim up to `ATTN_KV_MAX_HD`, static recording only
  (`decode_eligible` refuses it). It is the scalar one-workgroup-per-(row, head)
  kernel and attends EVERY visible compressed key, masking the unselected ones
  with the bias, so a CSA layer's cost grows with `n_visible` (ctx/4) where the
  reference's flash path compacts to the top-k (B-DSHW-FUSED-REF). The
  tensor-parallel lowering refuses it (the compressed cache has no per-rank
  split). **Metal refuses `comp`** outright: V4's head dim 512 is past its
  attention bound, so a kernel there would serve no model yet (plan 09 S10).
  Metal's indexer kernel takes `block_ratio` but has not run anywhere; the macOS
  CI job runs `crates/infr-metal/tests/parity.rs`, whose indexer case is
  ratio 1.
- **The CPU attention arm widens the compressed prefix to f32 on every call**
  (2026-11-16): `len` rows per op, the same pattern as its raw cache, so a CSA
  layer at long context converts `ctx/4` rows per decode step before attending
  any of them. Converting per key inside the dot (or an f32 shadow cache) would
  remove it; not measured yet (plan 09 § 5).
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

### B-DSV4-CPU-PERF — what the V4 compressed layers still cost on the CPU (2026-11-16)

**Tag:** deepseek · performance · **Blocked on:** nothing (plan 09 § 5)

The S7 review's two findings are fixed: `Op::Attention` attends only the
compressed keys whose top-k bias is finite and converts only the cache rows some
query keeps (`infr-cpu`'s `attn_kv::comp_rows`), and `Op::LightningIndexer`
scores one row's keys across the pool and keeps its top-k by a partial selection
(`infr-cpu`'s `indexer` module). Output is bit-identical (`docs/deepseek.md` §
"The CSA tier's CPU cost"). What remains, measured with
`crates/infr-llama/tests/dsv4_cpu_bench.rs` (one decode row at V4-Flash's
geometry over random f16 caches, 32 threads, a quiet Ryzen 9 9950X3D):

- **The attention re-reads the kept rows once per head.** At 64k context a CSA
  layer now costs about 1.05-1.15 ms per token (indexer 0.36-0.39 ms, attention
  0.71-0.77 ms) and an HCA layer 0.66-0.70 ms, so about 21 × 1.1 + 20 × 0.7 ≈ 37
  ms per token over the compressed layers. The attention is flat in the context
  (0.63-0.77 ms from 2k to 64k: it reads at most `top_k` compressed rows); about
  0.2 ms of it is the raw window and the op's fixed costs (the HCA layer at 2k,
  16 compressed rows, takes 0.23-0.24 ms). The rest is 64 heads each streaming
  the same 512 converted rows (1 MiB of f32) twice, for the dot and the weighted
  sum: one task is one head (`Op::Attention`'s dispatch), so V4's single KV head
  is read 64 times from L2/L3. Grouping the heads that share a KV head into one
  task, each kept row loaded once and dotted with every head's query, would cut
  that traffic by the group size. It changes the attention dispatch every model
  uses, so it needs its own measurement across models; the per-head arithmetic
  would not change, so `attention_blocks_match_one_pair_per_task_bitwise` and
  the two-segment concatenation test keep their bit-for-bit checks.
- **The bench pays a pool wake per op.** It executes each op as its own graph,
  so the pool's workers have parked between ops; in a decode they would still be
  spinning. The fixed 0.2 ms above is an upper bound for that reason.
- **A quantized compressed cache still converts a prefix.** At f16 (the default)
  and f32 the attention converts the kept rows only; at any other KV format it
  dequantizes the prefix up to the last kept row (`attn_kv::comp_rows`' fallback
  arm), which at 64k is most of `len`. Converting quantized rows one by one
  needs a per-row dequant for every KV format, including TurboQuant's rotated
  blocks.
- **Multi-row graphs: measured for the indexer alone, on DeepSeek V3.2's
  shape.** `Op::LightningIndexer` is also what V3.2 emits, with `rows: batch`
  and `block_ratio` 1, so its prefill and decode take the same new scoring and
  selection. A reviewer's scratch bench at 64 heads × 128, `top_k` 2048, f16
  keys, 32 threads, old against new with equal output digests: 512 rows × 4096
  keys 462-464 → 39.9 ms; 512 × 16384 2030-2037 → 149-151 ms; 32 × 16384 134-143
  → 9.5 ms; 1 × 16384 61 → 0.37 ms. Not in the tree's bench, and no V3.2 model
  was run. The score buffer is `rows × keys` f32 (32 MiB for 512 rows at 16k
  keys). This entry used to say "about 335 MiB at a 160k context": that is 512
  rows × 163 840 keys × 4 bytes, V3.2's block ratio 1 (one key per token), and
  is 335.5 MB, i.e. 320 MiB, not 335 MiB. On V4 the keys are blocks of 4, so 512
  rows at 160k take 80 MiB and a 1024-row chunk 160 MiB, plus a top-k bias of
  the same shape. V4's prefill now shrinks its chunk to keep each within 64 MiB
  (`dsv4_emit::prefill_chunk_rows`, `DSV4_PREFILL_SCORE_BYTES`: 1008 rows at
  64k, 408 at 160k, 63 at 1M) and the CPU executor recycles both buffers across
  CSA layers; V3.2's prefill has no such bound and is not run on this box. The
  attention converts the union of the rows' kept sets, at most
  `min(blocks, rows × top_k)` rows (on V4-Flash 32 MiB at 64k, 80 MiB at 160k),
  not bounded further. None of it was timed on a V4 chunk at V4-Flash's
  geometry; `dsv4_cpu_op_bench` times one decode row.
- **Batched prefill: measured on the synthetic file and on one expert layer, not
  on the real file.** `dsv4_cpu_fixture_bench` (synthetic, 512 tokens): 1589.1 /
  1620.2 ms one position per graph against 27.9 / 27.5 ms batched, nearly all
  per-graph fixed cost. `dsv4_cpu_moe_prefill_bench` (one V4-Flash routed-expert
  layer, Q2_K banks in RAM, `INFR_DSV4_MOE_BENCH=1`): 0.63 / 0.62 ms per row in
  a 1024-row chunk (the default height) and 0.81 in a 512-row one, against 5.73
  / 5.70 ms one row at a time; 2.8 at 64 rows, barely any gain at 16 (each row's
  experts are mostly distinct). The one-row baseline repeats one row, so its
  experts stay warm in L3: conservative. The real file streams its banks from
  disk, where a chunk reads each routed expert once for all its rows. On the
  real file (§ B-DSV4-REAL) a 5-token prompt prefills in 16.21 s at a 12 GB
  arena, most of it the first read of every dense weight; whether the default
  chunk height is right for a long prompt there was not measured.
- **The rings' round trip** (§ B-DSV4-WIRING) is per token but not O(context):
  it was not measured here.

- **The bench's absolute figures are best-case.** It repeats one op over one
  layer's caches (16 MiB compressed, 4 MiB indexer), which stay in the 9950X3D's
  L3 between runs; a real decode walks every compressed layer's caches. That
  flatters the old code more than the new, which reads only kept rows, so the
  ratio is not inflated. `dot_mn` keeps 16 eight-wide accumulators live, which
  likely spills on AVX2-only x86 and on NEON: speed only, not measured on
  either.

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
- **`batch > 1` runs on the CPU only.** The CPU's batched V4 prefill gathers
  each row's experts by its own token id, from the chunk's ids (`PfChunk::ids`
  when the embedding is gathered on the host, the chunk's input when it is
  gathered in the graph); `seam/dsv4_emit/tests/prefill.rs` holds both to one
  position per graph. On Vulkan V4 still prefills one position per graph (§
  B-DSV4-WIRING), so `gather_i32.comp` has run at `rows > 1` only in the
  op-level parity test.

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

### B-DSV4-REAL — the shipped V4 file runs on CPU at 1.28 s a token (12 GB cache); its numbers are unverified (2026-08-13, run 2026-11-16)

**Tag:** deepseek · streaming · **Blocked on:** nothing

**It runs.** `ggml-org/DeepSeek-V4-Flash-0731-GGUF:Q2_K_S` (98.6 GB) generates
coherent text on the CPU backend of the 64 GB Windows box, through the host
weight arena:
`INFR_DRAM_CACHE=12g infr run ggml-org/DeepSeek-V4-Flash-0731-GGUF:Q2_K_S "Hi" --dev cpu --temp 0 --max-new 9 --ctx 1024 --no-think`
printed `Hi! How can I help you today?`, and the default budget (no
`INFR_DRAM_CACHE`) counted from one to twenty-three in words for "Count from one
to forty in words." with `--max-new 48`. Those are the only two prompts run.

**Speed, before and after 2026-11-16's fixes** (CPU, Ryzen 9950X3D, 32 threads,
the box has a Samsung SSD 980 1TB and a KINGSTON SNVS1000G; which one holds the
HF cache was not checked):

| tree       | cache           | prefill (5 tok) | decode        |
| ---------- | --------------- | --------------- | ------------- |
| `879a8677` | 12g             | 147.09 s        | 123338 ms/tok |
| this fix   | 12g             | 16.21 s         | 1284 ms/tok   |
| this fix   | default 21.6 GB | 16.03 s         | 1199 ms/tok   |

The decode figure is the run's average over 9 to 10 tokens, the first decode
token included; across tokens 2-9 of the 12g run it was 1.15 s a token (the
`max-new 1` and `max-new 9` runs differenced). On the 48-token default run it
was 815 ms a token as the 21.6 GB arena warmed (72.8% expert hits).

**Where a decode token's time went.** Before: every expert bank was ONE block of
704.6 MB, so a routed expert dragged in its whole bank; 4 forward passes read
336.11 GB of banks (477 reads) plus 24.83 GB of dense weights, which the
proportional budget split had left few slots, at about 0.70 GB/s — about 90 GB
per token. After (12g, tokens 2-9): 292.5 expert reads of 2.75 MB a token (0.805
GB), 481.5 expert hits (62%), no dense reads at all (every whole weight
resident, 0 evictions), 0.84 s a token waiting on residency and about 0.31 s of
compute. The disk-bound estimate — 0.805 GB at the 1.05 GB/s four concurrent
reads reach on this drive (measured with a standalone positioned-read bench) —
is 0.77 s, plus compute 1.08 s against 1.15 s measured.

**Not established: any NUMBER the graph computes on this file.** The synthetic
oracle (§ B-DSV4-ORACLE) says the emit computes what llama.cpp computes on small
files at this file's hyper-connection count and YaRN keys; it says nothing about
the quantized weights, the 64-head indexer at `top_k` 512 past 2051 tokens, or
long-context top-k selection, and llama.cpp cannot load this file on this
machine (plan 09 § 4). Coherent English for two prompts is the only evidence.
Its tokenizer IS checked: `joyai-llm` now maps to the `deepseek-v3` regex list,
and `deepseek_v4_tokenizes_as_llama_cpp` holds twelve strings to
`llama-tokenize`'s ids (vocabulary only, first shard). The "Hi" prompt's ids
(`[0, 128803, 23166, 128804, 128822]`) did not change with the mapping. Vulkan
and Metal refuse the file's first compressed layer by name.

**What is left, ranked:**

- **Decode is disk-bound on expert misses** (0.805 GB a token at 12g). A bigger
  cache helps only as it warms. Levers not built: overlapping the next pin wave
  with compute (routing for layer l is only known after its attention, so this
  would be the down bank's reads behind the gate/up GEMMs, or a speculative
  prefetch from the previous token's routing); unbuffered reads
  (`FILE_FLAG_NO_BUFFERING`), which would also stop the double caching below.
- **Compute is about 0.31 s a token** and was not profiled per op on this file.
- **Prefill of a cold model reads every whole weight once** (7.6 GB of dense
  classes, 4.8 s of summed read time at 12g) before the experts; 16 s for five
  tokens. Not optimized.
- **A long prompt prefills layer-major since 2026-11-17** and reads each routed
  expert once (§ B-CPU-EXPERT-PAGING has the runs and what is left); a
  1443-token prompt in two default chunks prefilled in 195 to 207 s.
- **Buffered reads grow the system file cache's working set.** On the 48-token
  default run `\Memory\Cache Bytes` rose from 245 MiB to 3701 MiB, which counts
  as in use, not available; available memory bottomed at 19154 MiB against the
  21687 MiB (half of 43329 MiB) the budget rule leaves. The process itself
  stayed inside its budget (private bytes 20872 MiB for a 21.55 GB arena plus
  1.19 GB priced). The `RANDOM_ACCESS` and `SEQUENTIAL_SCAN` open flags did not
  prevent it in a standalone bench; unbuffered I/O would, and needs
  sector-aligned reads.

Measured from the file's metadata (`block_count = 43`, `hash_layer_count = 3`,
`compress_ratios` = `[0, 0, 4, 128, 4, 128, …, 4]` over the 43 layers, then
three surplus zeros): **no layer of the real model is both ratio-0 and
bias-routed.** Layers 0 and 1 are the only ratio-0 layers and both are
hash-routed; layers 2..42 are all compressed (alternating 4/128), and layer 2 is
compressed AND hash-routed. The geometry
`kv_footprint_of_deepseek4_flash_counts_its_compressed_tiers` prices: head 512
with a 64-dim rope, indexer 64 heads × 128 with `top_k` 512, window 128,
`compress_rope_freq_base` 160000, YaRN factor 16 over 65536.

### B-CPU-EXPERT-PAGING — what per-expert CPU paging left (2026-11-16)

**Tag:** streaming · cpu · **Blocked on:** nothing

`infr_cpu::paged` pages a MoE bank one block per expert and `Op::MoeFfn` pins
only its routed experts (`CpuBackend::moe_banks`). Left:

- **The expert pool needs one op's worst case to exist at all:** `n_expert`
  slots for each bank of that pool's slice size one layer has (three on
  V4-Flash's split banks: 768 slots, 2.1 GB), because a prefill chunk can route
  to every expert and the op pins all of a layer's banks together. Below that
  the banks stay mapped, as any unseatable class does. Pinning per bank, or per
  group of buckets, would lower the floor; not built, and no budget that small
  was tried on a real MoE file.
- **`infr serve --parallel N` on the CPU** holds up to N ops' pins at once; the
  floor prices one (the gap B35 records for the whole tier, now larger for
  experts). And each CPU resident sizes its arena when it opens: a second
  generation that starts while the model's own resident is out opens another
  (`SeamModel::with_cpu`), which measures available memory at that moment —
  after the first arena was taken — and takes half of what is left. Two
  concurrent generations can so hold three quarters of what was available at the
  first one's start. Behaviour read from the code, not run on a large model.
- **The repack cache is priced at its whole budget** (`cpu_repack_reserve`, 4096
  MiB by default) whenever the fused Q4_K expert path can fill it, though the
  packs a run actually keeps may be fewer; not measured on a paged Q4_K MoE
  model (qwen35moe, gemma4's MoE).
- **`cpu_committed_bytes` prices the KV cache at f16**, the CPU default; a
  user-named f32 KV is under-priced. It goes through `kv_bytes_estimate`, the
  function the GPU fit uses, so V4's compressed caches are priced at what the
  session allocates for them (`dsv4_comp_bytes`), and the raw cache is priced at
  the whole context without the sliding-window ring the CPU uses, which
  over-prices it (conservative). The 1 GiB activation reserve is a reserve, not
  a measurement per model.
- **A V4 prompt of several chunks now prefills layer-major and reads each routed
  expert once (2026-11-17); what that left.** `docs/deepseek.md` § "Layer-major
  prefill on the CPU" has the design, the tests and the runs. On the real file a
  1443-token prompt in two default chunks (1024 + 418) read 23 292 expert blocks
  (64.1 GB), each exactly once by the pager trace, and prefilled in 206.83 and
  195.16 s; chunk-major had spent 209.5 s on its first chunk alone when the run
  guard stopped it at 235 s. Left:
  - **Small chunks stay chunk-major, by measurement only.** In 32-row chunks a
    69-token prompt was slower layer-major (98.96 and 114.89 s against 91.59 s),
    so the default needs chunks of `seam::HOST_LAYER_MAJOR_MIN_ROWS` (256);
    nothing between 32 and 1024 rows was run, so the threshold is a choice
    between two points. The likely cause is that a small chunk's sweep (about 20
    GB) stays in the OS file cache, so chunk-major's re-reads are copies; that
    was inferred from timings, not measured per read (the re-read counters
    `HostPagerStats::rereads` now exist; no chunk-major run printed them).
    Unbuffered reads (B-DSV4-REAL) would remove the confound and the double
    caching both, and might let the order pay at any height.
  - **The chunk-major arm of the decisive comparison did not finish**: the run
    guard stopped it at 11.5 GB available (the box's buffered reads grow the
    system cache; another session held memory too). Its total is an estimate
    from the rate of its last 11 layers (about 310 s).
  - **Non-wait time of the 32-row runs** (16.0 s layer-major against 8.8 s) is
    unexplained: the synthetic bench shows no per-span cost (229.3 against 230.8
    ms for 16 graphs against 80), so it was not the span graphs.
  - **The decode warm start keeps only the last row's experts and the most
    routed of the last chunk** (`paged::done_keep`, 8 a bank): the first
    decode-loop pass hit 470 of 774 expert pins after a 1443-token prompt (729
    misses of 774 without it). The share now comes from the pool's room
    (`PagedWeights::done_keep_share`). Counting routed rows over all of a
    layer's chunks, not just its last, was not tried.
  - **Not measured, and so not on by default: chunks of 256 rows, and concurrent
    sessions.** `HOST_LAYER_MAJOR_MIN_ROWS` (256) admits first chunks no run had
    (only 32, slower, and 1024, faster, were measured); a request sharing the
    backend (`infr serve --parallel`, chunks of `device.ubatch_parallel`) keeps
    chunk-major whatever its height. To decide both: on V4-Flash at 12 GB, the
    same 1443-token prompt at `--ubatch 256` and `512`, layer-major against
    chunk-major (`INFR_LAYER_MAJOR`), with `INFR_PAGER_STATS` (prefill time,
    reads, re-reads and their time); then two concurrent
    `infr serve --parallel 2` requests, one decoding while the other prefills a
    long prompt, comparing the decoding one's ms a token and expert hits in both
    orders. On the CPU today two concurrent generations do not share an arena
    (each opens its own resident, `SeamModel::with_cpu`), so the question is
    their combined memory and disk bandwidth, not one evicting the other's
    experts; if they ever share one, the demotions must spare what the other
    session uses.
  - **The chunk-major run held about 8.5 GB more of the box's memory than the
    layer-major ones** (lowest free memory 11.5 GB against about 20 GB, run 6
    against 5 and 7), unexplained. The guess that a whole-model graph zero-fills
    every layer's intermediate buffers at once does not hold up in the code: the
    compressed tiers' scratch is declared once per tier and shared by every
    layer of it (`dsv4_emit::Dsv4Tiers`), and the other scratch once per graph,
    so a whole-model graph holds at most both tiers' scratch at the chunk height
    where a span graph holds one tier's — hundreds of MB at 1024 rows (the
    widened stream's two `[rows, hc_mult * n_embd]` f32 buffers are 128 MiB),
    not gigabytes. Chunk-major had read about 1.7 times the bytes by then (108
    GB predicted against 64), and buffered reads grow the system file cache's
    working set (§ B-DSV4-REAL), which is the more likely cause; not measured.
  - **An arena with no slots cannot stream** (`PagedWeights::streams` asks for
    slots): unreachable on the CPU, whose pools always have at least one
    (`HostPager::new` refuses none), kept as a guard.
  - **The pool still needs one layer's every expert** (768 slots, 2.1 GB on
    V4-Flash): layer-major reads each expert once only while a layer's union
    fits beside what is pinned, which the floor guarantees.
  - **Other families keep chunk-major on the CPU arena**
    (`seam::host_layer_major`): no other family was checked bit for bit
    layer-major on the CPU (gemma4's KV-sharing layers read another layer's
    cache, which needs its own argument). `paging.layer_major = true` forces it
    on any family.
- **The six-minute hang on a broken bank stride (2026-11-16) is explained as far
  as it can be: not the pool; a same-thread double lock is the one mechanism
  found, and it now fails loudly.** Breaking `CpuBackend::moe_banks` so every
  bank held whole takes the gate bank's stride fails
  `paged_experts_compute_what_whole_banks_compute_while_evicting` in about a
  second (checked with and without `RUST_BACKTRACE=1`): `SpinPool::run` already
  caught task panics and waited for its workers. Writing that same break as
  `bufs[0].len_bytes()` inside the `.map` that holds `b.read()` hangs instead,
  with no panic at all: for bank 0, `len_bytes` relocks the `Mutex` whose guard
  `read()` holds, and a `std` mutex is not reentrant. Whether the original break
  was spelled that way is unknown. The graph-level form of the same deadlock —
  one owned buffer bound to two operands an op holds locked at once — is now
  refused by `refuse_relock` in `Op::MoeFfn` (router against every bank,
  `down_scale` and `exp_probs_b`; each bank against the others) and `Op::Mla`
  (`k_cache` against `wk_b`/`wv_b`), pinned by
  `moe_ffn_refuses_one_owned_buffer_bound_to_two_banks` and
  `mla_refuses_one_owned_buffer_bound_to_the_cache_and_a_weight` (both hang to
  the 60 s `within` limit with the check disabled). `Op::LightningIndexer` holds
  its k-cache lock for the op body but locks no other owned buffer (its `q` and
  `weights` come from the interpreter's f32 values), so it needs no guard. No
  other arm of `CpuBackend::execute` holds two owned locks at once (read
  2026-11-17: `Op::Attention` takes its K, V and compressed-cache guards one
  statement at a time; `Op::Linear` and the gathers hold one). A code path
  outside a graph that relocks, like the `len_bytes` break above, is not
  guarded.
- **Locks above the CPU backend after a panicking forward: settled for
  `infr serve`.** A panicking generation becomes a failed request
  (`spawn_blocking`; non-streaming gets a 500 carrying the panic message,
  streaming an error frame from `DoneGuard`), and no backend state is reused:
  `SeamModel::with_cpu`'s `Lease` drops the resident (backend, caches, paged
  store, bound weights) when `std::thread::panicking()`, and its slot lock uses
  `into_inner` and is never held across the closure; `generate_dense_cpu_mode`
  allocates a KV cache per call; `lock_recovered` resets DiffusionGemma's
  session (`reset_kv` drops it); MTP on the CPU builds a `CpuBackend` per call;
  `ParallelGenerator` is Vulkan-only; and no production code catches a panic
  around `execute` and keeps going. So clearing poison in `lock_bytes` changes
  nothing a `serve` user sees; it matters to a library caller that catches the
  panic and reuses the backend, which must then discard its session state.
- **Not covered by a test on a real MoE file other than V4-Flash** (the
  synthetic V4 fixture and an op-level test cover the mechanism, the latter
  bitwise against whole banks while evicting for f32, Q8_0, Q4_K and Q2_K in
  both bank layouts; Qwen3-30B-A3B and ERNIE-4.5-21B-A3B on the CPU with
  `paging.dram` were not run).
- **`hunyuan-dense` and `hy_v4`** share llama.cpp's regex arm with `joyai-llm`
  and are still unmapped: no file with either pre-type to check ids against.

### B-DSV4-ORACLE — llama.cpp over the synthetic V4 file: what it covers and what it does not (2026-11-16)

**Tag:** deepseek · **Blocked on:** nothing (each gap names its own)

`crates/infr-llama/tests/synthetic_deepseek4_llama_cpp.rs` and
`docs/plan/09-deepseek-v4-compressed.md` section 4 have the design and the
numbers. Open items:

- **Two llama.cpp defects at `5676dfbe0`, not reported upstream.** A V4 file
  with no ratio-4 layer dies with an integer division by zero in
  `llama_kv_cache`'s constructor (the forced LID rotation loops `n_rot_k` to 0),
  and `build_hc_mixes` asserts `hc_mult == 4`. Neither affects the shipped file.
  Reporting them is the user's call; the oracle works around the first with a
  null ratio-4 tail layer (`OracleCase::null_tail`), which is exact only while
  `attn_output_b` multiplies a finite attention output.
- **Every case is scored in CI** since plan 09 S7 (`ratio0`, `hca`, `mixed`,
  `csa`; max |logit diff| at most 1.6e-4 at any stored length). `csa`'s first
  data (4 indexer heads) missed by 3.4e-3 at 131 tokens; the bisect found exact
  zero-score ties at the top-k boundary, broken differently by llama.cpp's
  `std::partial_sort` and infr's lower-index rule, which no weight noise can
  expose. `csa` now has 16 indexer heads and another prompt (`prompt_salt` 6).
  The robustness check still cannot see an exact tie; the 16 heads make one
  improbable (all 16 dots negative), and for the checked-in data infr's own
  scores showed none, measured once by hand with a temporary dump.
- **Skipping the Hadamard rotation of the LID keys flips 2 of 432 selections**
  on the `csa` run (positions 111 and 122, last layer, margins 1.2e-4 and 1.8e-4
  in infr's scores; a temporary rotation in infr made every selection agree).
  Neither reaches a stored logit. On V4-Flash, f16-scale near-ties will choose
  differently from llama.cpp at a comparable rate. **Decision for the user:**
  port the rotation (a 128 × 128 orthonormal multiply on each LID row and each
  indexer query head, 64 heads × 128² multiply-adds per token per CSA layer on
  V4-Flash) for selection-level agreement, or keep the unrotated keys, which are
  equally correct in exact arithmetic. Not done: S7 kept the plan's choice.
- **Selection agreement is checked in CI only on the stored runs**
  (`synthetic_dsv4_indexer_keeps_what_llama_cpp_keeps`: every row of the `csa`
  131-token and `mixed` 300-token one-token-per-step runs). The near-tie guard
  reads infr's scores, not llama.cpp's, because llama-debug prints only six
  values of a score row; where the engines keep the same blocks the two agree to
  f32 rounding, but a near-tie llama.cpp has and infr does not would pass it.
  `mixed`'s selection is `all` at every row by construction (its `top_k` is at
  least the padded width), so it checks only that infr keeps every block there.
- **Only the CPU backend is scored against llama.cpp.** Vulkan's ratio-0 emit is
  checked against infr's own CPU (`gpu_synthetic_deepseek4_ratio0_matches_cpu`,
  at `hc_mult = 2`), not against the oracle data; a `prefill_logits_vulkan` twin
  of `synthetic_dsv4_ratio0_cpu_matches_llama_cpp` behind the GPU `#[ignore]` is
  a few lines.
- **`hc_mult = 2` and `hc_mult = 1` cannot be oracle-checked**, because
  llama.cpp refuses them. The canonical fixture keeps `hc_mult = 2` for infr's
  own tests, so those paths stay checked only by infr's mechanism tests.
- **What the synthetic oracle cannot reach:** quantized weights (every tensor is
  f32 but the i32 routing table, so the BF16 routers and Q8_0/Q2_K paths of the
  real file are untouched), the `deepseek-v3` tokenizer, and decode past the
  prompt. infr's side of every comparison is now its batched CPU prefill; it
  gives the numbers the one-position runs gave, to the last digit printed, and
  is held bit-identical to them by `seam/dsv4_emit/tests/prefill.rs`.
- **The sums files are 52 to 66 KB each**, mostly the compressors' intermediate
  rows at 7 positions. Trimming to the final roped rows would halve them; kept
  whole because S6/S7 bisect on the intermediates.
- **The indexer's top-k is oracle-checked only up to 131 tokens, on one case
  (`csa`).** Longer prompts with a biting `top_k` always held a near-tie that
  ±1e-3 noise on the indexer weights flips (40 prompt sequences tried at 300
  tokens, 30 at 520; plan 09 section 4; at 4 indexer heads). Covering
  long-context selection needs a fixture whose indexer scores are decisive by
  construction (for example keys with a few strongly separated levels) rather
  than random; not attempted. The robustness check is by perturbation, not by
  reading the score gaps, which `llama-debug` prints only as the first and last
  three values of each row.

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

### B-WAVE64-GCN-UNVERIFIED — the wave64 path has never run on GCN hardware here (2026-11-16)

**Tag:** vulkan · amd · **Blocked on:** a GCN card (Polaris, Vega), issue #94

Wave64-only devices run the wave32 kernels through `shaders/wave64.glsl` (ported
from the KessokuNet fork, commit `b108ff42f` by korewaChino, then reworked
here): every pipeline pins subgroup 64 and sets the `W64` specialization
constant. A workgroup of at most 32 invocations runs as one subgroup (id 0, lane
= `gl_LocalInvocationIndex`, plain subgroup operations over exactly its own
lanes); every larger one is 64-aligned in x, asks for full subgroups, and runs
each 64-lane subgroup as two 32-lane halves with cluster-32 operations.
Cooperative matrix is off on that path (`caps::subgroup_mode`, the `!wave64`
filter on `coopmat_f16`).

- **Verified here (2026-11-16/17):** the same path on RDNA, forced with
  `kernels.vulkan.force_wave64` (`INFR_FORCE_WAVE64=1`), under AMD's Windows
  driver on the RX 7900 XTX and the RDNA2 iGPU: the GPU gate (op parity, the
  `infr-vulkan` lib, the whole-model GPU-vs-CPU and golden tests) and
  `mmv_mw_parity`; results are in the PR that landed it. Both drivers report a
  subgroup range of [32, 64]. `ops::tests::subgroup_kernels_are_wave64_safe`
  checks every built module statically.
- **Not verified by us:** GCN itself. The fork author reports two GPU-vs-CPU
  model tests passing on an RX 580 (RADV, Mesa 26.2.5) with the fork's ORIGINAL
  commit, before our changes (include placement, the one-subgroup mode for small
  workgroups, the scan rewrite, full subgroups, coopmat off). GCN under AMD's
  Windows driver has not been run by anyone we know of.
- **Still assumed, and checked at startup:** a workgroup of at most 32
  invocations on a 64-pinned pipeline is ONE subgroup (wherever its lanes sit in
  the wave), and its inclusive scan runs in `gl_LocalInvocationIndex` order —
  the second only `deltanet_gates.comp` needs. `wave64_probe.comp` measures both
  on every device that takes the wave64 path (`caps::check_wave64_layout`), and
  a driver that breaks either is refused. The probe is one 32-wide workgroup; a
  driver that treated some other pipeline differently would not be caught.
- **No ordering assumption otherwise:** the 64-aligned workgroups use virtual
  subgroup ids and lanes only relative to each other, and full subgroups make
  the halves dense. The earlier note that the wave32 kernels all rely on
  in-order lane packing was wrong; only `deltanet_gates` mixes the subgroup lane
  with the invocation index.
- **Unreachable builds:** the `_sg16` decode builds include `wave64.glsl` but
  are never dispatched on the wave64 path: `sg_pref` stays 32 there, since 16 is
  outside the [64, 64] range the path reports.
- **Not checked:** `requiredSubgroupSizeStages` containing COMPUTE, for the 64
  pin and for the existing 32 pin alike. Both drivers here report it.
- **The Qwen3-30B-A3B "KV capacity 0" failures were not this path (settled
  2026-11-17):** on a quiet box, `gpu_seam_paged_moe_matches_resident_and_cpu`
  passed 62 of 62 runs alternating `origin/main` and the branch (15+15 native,
  15+15 forced, 8+8 back to back after the test that precedes it in the gate),
  and the test's resident leg measured the same live room after its weights on
  `origin/main`, the branch and the branch forced, to the byte in the same round
  (3.19 GiB at its 1-row chunk). A capacity of 0 needs that room under the 0.30
  GiB a one-token context and the driver reserve take, so about 2.9 GiB more
  device memory held by another process between the expert-budget snapshot and
  the post-load measurement. The earlier failures all came within a few hours of
  each other, when other GPU work on the box is the likeliest cause; nothing
  recorded what held the memory then. Such a session now refuses at load with
  the measured numbers (`reclamp_ctx_to_live_room`), so the next one explains
  itself.
- **Integer inclusive scan:** `vw_half_inclusive_add` is float-only, so an
  integer `subgroupInclusiveAdd` under the header fails to compile (glslc: the
  macro's float arm cannot be assigned to a `uint`). No kernel has one; a kernel
  that needs one needs an integer twin added first.
- **Possible later optimization:** the fork's wave64-native decode GEMV
  (`a1f58563b` and follow-ups) uses all 64 lanes instead of two halves. A
  32-wide workgroup on this path leaves half of each wave idle.

### B-HWDET-NO-ARCH-BUCKET — values can now be tuned per architecture; a per-kernel subgroup width still cannot (2026-11-17)

**Tag:** vulkan · detection · tuning · **Blocked on:** plan 10's S3/S6 for
per-kernel pins; measurements for everything else

The architecture bucket (`caps::DeviceArch`, see B-HWDET-VENDOR-RULES) now has a
second consumer: the tuning table (`crates/infr-vulkan/src/tuning/`, plan 10
S1). A row can match `vendor`, `arch`, `driver`, `integrated`, the subgroup
range, a compute-unit range or PCI ids, and set any HOST or VARIANT parameter
for that class — split counts, size bands, rows per workgroup, which build runs.
`infr tune list` names every parameter; `infr tune show` prints which value a
device runs with and why. What a bucket was meant to buy is therefore
representable for those values. Still absent:

- **A per-kernel subgroup width.** The only pin parameter is `device.sg_pref`
  (one width for the curated decode family); every other subgroup kernel pins
  32, `MLA_SG_WIDTH` (reported as `mla.sg_width`, fixed) or the driver's choice.
  A per-kernel pin needs the spec-constant mechanism (plan 10 S3) and is the
  per-architecture table upstream has (`gpu_pipeline_configs`). Still not
  justified by a measurement here.
- **Thresholds not yet migrated** (S1 kept them inline because no device rule
  chooses them today): `adapter::MMV_MIN_ELEMS`, the `2 << 20` floor in
  `mmv_mw_choice`, `recorder`'s `WIDE_GRID_MIN` (an occupancy number like the
  split-K window), the `DENSE_SMALL_TILE_*` bands, `DP4A_KQ_MIN_ROWS`, and
  `split_k_plan`'s clamp of 8. Each becomes a parameter when a measurement on
  some device wants a different value.
- **Wave64-only hardware runs through an emulation, not a per-op table**: a
  device that cannot pin 32 but can pin 64 (AMD GCN) takes the wave64 path,
  which is decided from the subgroup range (`caps::subgroup_mode`), not from the
  bucket. See B-WAVE64-GCN-UNVERIFIED for what is and is not verified.

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

### B-NVSHAPE-CORECOUNT — the shader-core count exists on every vendor; of the occupancy constants, only the prefill GEMM split-K uses it (2026-11-17)

**Tag:** vulkan · nvidia · intel · **Blocked on:** NVIDIA/Intel hardware to
validate; a RADV re-measure of the GEMM split-K window; per-site measurements
for the attention split-K constants

`caps::shader_core_count` sources the count the way llama.cpp does —
`VK_NV_shader_sm_builtins`' `shaderSMCount` on NVIDIA,
`VK_AMD_shader_core_properties2`' `activeComputeUnitCount` on AMD, and a PCI
device-id table on Intel transcribed verbatim from
`ggml_vk_intel_shader_core_count` — with the old `VK_AMD_shader_core_properties`
(v1) product kept as a fourth fallback so an AMD driver advertising only v1 does
not regress to 0. It lands in `Capabilities::compute_units` (0 = unknown) and in
the tuning table's device facts (`cu=` in `infr tune show`'s match key).

Verified on this box: both AMD devices report the same number through v2 as they
did through the v1 product (RX 7900 XTX 96, Raphael iGPU 2), so preferring v2 is
a no-op here. **The NVIDIA and Intel branches have never executed** — no such
part exists on this machine.

**Consequence nobody has watched run:** besides the split-K window (below), the
consumer is `infr_core::integrated_ubatch_rows` via
`infr_llama::seam::default_ubatch_rows`, which scales the integrated prefill
chunk from the count. An Intel iGPU in the table (0xB080, Panther Lake Xe3
LPG, 12) previously got 0 → the 128-row floor and would now get 768 rows. That
is what the scaling function means, but it is a 6x larger per-submit chunk on a
watchdog-sensitive part, unvalidated. If an Intel iGPU ever hangs in prefill,
this is the first thing to look at; `INFR_UBATCH` overrides it.

**The split-K window scales with the count on ONE driver.** The coopmat prefill
GEMM splits K when its plain tile grid is under
`max(split_k.fixed_grid, split_k.fill_per_cu · CUs)` (128, and 4 per CU) with
`split_k.scale_by_cu` on, else under the fixed 128 alone. The generic default is
off; the built-in table's `amd-proprietary-split-k` row (vendor `amd`, driver
`AMD_PROPRIETARY`) turns it on, because that is the one driver it was measured
on: an RX 7900 XTX (96 CUs) under the AMD Windows driver, where the 128-wg pp512
projections the fixed window left unsplit got faster (Qwen3-30B-A3B
`attn_output` 458 → 302 us, SmolLM3-3B `ffn_down` 1392 → 962 us). Below 32 CUs
the floor is the larger term, so the integrated GPU behaves as before.

**Unmeasured, so still on the fixed window:** RADV (the platform the README /
`docs/perf/results.md` benchmarks ran on, and where the fixed window was tuned),
AMDVLK, NVIDIA and Intel. To extend it to one of them: run
`infr bench -p 512 -n 0` on a dense model (SmolLM3-3B) and an MoE one
(Qwen3-30B-A3B) with `INFR_PROF_OPS=1 INFR_PROF_OP_SHAPES=1`, once as is and
once with `INFR_TUNE=split_k.scale_by_cu=true`, interleaved, and compare the
`native_gemm_warp_*` rows at 128-384 workgroups; where they win, add a row for
that driver to `table.toml` with the numbers as its `source`.

The other occupancy constants are tuning parameters now but still constants, not
derived from the count:

- the flash and PV split-K policies (`attn_flash.*`, `attn_pv.*`: no split at
  1024+ workgroups, a target of 2048, at most 8). Their implied
  workgroups-per-core ratio (2048/96 ≈ 21 for attention) is a different
  per-kernel occupancy factor from the GEMM's, so each needs its own
  measurement, not the GEMM's constant; a row or `INFR_TUNE` can now try one
  without a rebuild.
- `gemm_i8cm_kq.l_fill_per_cu` (2) does scale with the count, on the one driver
  where that GEMM runs.
- the `wide_grid >= 128` wide-tile gates (`WIDE_GRID_MIN`, not migrated; see
  B-HWDET-NO-ARCH-BUCKET).
- The decode attention chunk policy (`attn_split.*`, reported `fixed`) is
  **duplicated in GLSL** (`attn_live.comp` recomputes it on the GPU), so a
  device-derived target there needs a new push constant first (plan 10 S11).

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

### B-DSHW-FUSED-REF — where upstream's fused DeepSeek kernels live (2026-08-11, updated 2026-11-16)

**Tag:** deepseek · reference · **Blocked on:** nothing; this is a design
reference, not a defect

At llama.cpp `5676dfbe0` the DeepSeek ops `GGML_OP_LIGHTNING_INDEXER`,
`GGML_OP_DSV4_HC_COMB` / `_PRE` / `_POST` and `GGML_OP_TOP_K_MASK` have a case
in the CPU, CUDA and Metal backends' op switches. Vulkan has the indexer and the
three hyper-connection ops (`lightning_indexer.comp`, `dsv4_hc_comb.comp`,
`dsv4_hc_pre.comp`, `dsv4_hc_post.comp` under
`ggml/src/ggml-vulkan/vulkan-shaders/`) but no `GGML_OP_TOP_K_MASK` case; SYCL
likewise has the four but not the mask. Listed by grepping each backend's op
switch and the Vulkan shader directory, not by reading the kernels. These are
the reference kernels to read before writing or tuning infr's own (B-DSV4-HC and
the indexer's Vulkan cost in plan 09 § 5).

The reference's CSA attention is sparse where flash attention is on.
`deepseek4.cpp`'s `build_top_k_mask` still expands the selected indices into a
`-inf` mask over every compressed row, but `build_csa_attention` passes
`n_kv_max` (the raw window plus the top-k count) through `build_attn_mha` to
`ggml_flash_attn_ext_set_n_kv_max`, and the CUDA and Vulkan backends compact the
mask and attend over at most that many columns (`fattn-common.cuh`'s
`use_sparse`; `flash_attn_sparse_compact.comp`); Metal has `n_kv_max` handling
in `fa_aux.metal`. Read on 2026-11-16, kernels not run. So infr's two-segment
attention (`Op::Attention::comp`), which masks rather than gathers, does more
work than the reference from position 2051 on, when a CSA row's visible blocks
first outnumber the top-k; plan 09 § 5 lists gathering the CSA top-k as its
answer. The OpenCL and Hexagon backends also have cases for these ops and were
not looked at.

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

A request with a `logit_bias` takes the same host path, and so does one with
`min_p > 0` when `Op::Sample` would otherwise run (the kernel has no min-p cut;
greedy keeps `Op::Argmax`, which min-p cannot change).

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
`unsloth/DeepSeek-V3.2-GGUF`. That file declares
`general.architecture = "deepseek2"` and carries no indexer tensors, so it runs
the stage-2 dense-MLA path.

Public files that DO declare `deepseek32` exist too, contrary to what the docs
and the V3.2 test module's comment in `cpu_backend.rs` said: the Hugging Face
API (`/api/models/sszymczyk/DeepSeek-V3.2-Exp-light-GGUF`, read 2026-11-16)
reports `architecture: deepseek32` for that repo's Q4_K_M split, about 404 GB in
all — V3.2-Exp weights, tensor list not read. It cannot run on this box, so the
`deepseek32` arch (MLA plus the indexer) is still exercised only by the
synthetic fixtures. All three now say so.

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
  creation with the feature named, after the subgroup check. The check itself
  has a unit test; the call site has never run against a device that lacks
  either.
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

### B137 — `infr serve`: what JSON mode, the Host check and prepare-once left (2026-11-21)

`response_format`, the `Host` check, SSE keep-alive, `serve.max_body_bytes` and
preparing a request before it takes a slot have shipped. Left, all read in code
unless marked:

- **JSON mode narrows numbers.** A number the schema leaves unbounded is limited
  (`NUMBER_LIMIT`, `NUMBER_STEP` in `infr-llama/src/grammar/caps.rs`) and
  exponent form is never produced, so `6.02e23` cannot be written. The fraction
  limit is coarser than llama.cpp's because llguidance's `multipleOf` matcher
  does its arithmetic in 32 bits. Strings, array lengths and nesting are not
  bounded, as in llama.cpp, so a reply can still run to `max_tokens` inside a
  string.
- **Numbers in intersected subschemas keep only the schema's own bounds**
  (`allOf`, applicators beside sibling keywords, a property matching a pattern
  property, definitions referenced from those places): adding bounds or the
  any-value definition there makes llguidance refuse valid schemas. A digit run
  there can still reach `max_tokens`.
- **The schema guard over-refuses by design** (`grammar/shape.rs`,
  `grammar/caps.rs`). It exists because llguidance compiles recursively and a
  stack overflow aborts the process: a flat `$ref` chain killed the server
  before it. Refused although some could compile: any `$ref` that is not a plain
  local pointer; `$id`, `$anchor` and the dynamic/recursive keywords even where
  inert; a draft-04 `$schema`; a 2019-09+ keyword under a draft-06/07 `$schema`
  even as an annotation; two `multipleOf` steps that would combine badly but
  never meet; a `multipleOf` coefficient above `MAX_STEP_COEFFICIENT`, which
  llguidance compiles slowly or fails on at the first mask. Fractional steps
  that llguidance's own float conversion mangles (`0.0007`) are its 400, not
  infr's.
- **The stack measurements behind `MAX_COMPILE_LEVELS` are debug-build ones**;
  intersection-heavy chains were not re-measured in release, where frames are
  smaller. `the_keyword_lists_match_the_installed_llguidance` notices an
  llguidance upgrade that refuses or implements a listed keyword, not one that
  starts implementing a keyword infr treats as an annotation. A grammar built
  through the public `Constraint::new` gets the compile stack but not the shape
  check.
- **A mask-time grammar error** is a 500, or a `server_error` frame after the
  200 on a stream; the session stays usable (read, not run).
- **Thinking is off under `response_format`.** llama.cpp lets the model reason
  first and constrains only the content, through a per-template reasoning parser
  infr does not have.
- **The prompt is still tokenized twice** — once to count it, once inside the
  backend, which takes text. Removing that needs a token-input path on every
  backend. The render and the grammar compile happen once.
- **CPU and MTP sessions report no context limit before generation**, so an
  over-long prompt there is still refused by the runner, as a typed error frame
  on a stream.
- **A keyless server bound to `0.0.0.0` is not Host-checked** unless
  `serve.allowed_hosts` is set, though it is reachable through `127.0.0.1` and
  so open to the same rebinding. A resolver that forwards `*.localhost` names to
  DNS (plain glibc without nss-myhostname) weakens that allowance.
- **Not compiled or run here:** the macOS-only `supports_constraint` and
  `max_prompt_tokens` overrides in `chat/metal.rs`; JSON mode on Metal; the
  DiffusionGemma reply trim on a real model (its sizing is tested as a pure
  function); the arrival log line's `prompt_tokens`, checked by eye only.
- **`crates/infr-cli/src/main.rs` keeps growing**: the serve adapter and its
  tests belong in their own module.
- **`max_tokens: -1` means the server default (`sampling.max_new`)**, chosen on
  2026-11-12; llama.cpp reads `n_predict: -1` as "until the context is full".
  Recorded so the difference is a decision and not an accident.

### B232 — `strftime_local` reads `TZ` unsynchronised with `std::env::set_var` (2026-11-23)

**Tag:** plat · chat · soundness · **Kind:** open, needs a decision

`infr_plat::localtime::strftime_local` (the chat templates' `strftime_now`)
calls the C library's `localtime_r` on Unix. The C library takes the zone from
the `TZ` environment variable, reading the process environment directly and not
under `std`'s environment lock, so a `std::env::set_var` or `remove_var` on
another thread during that read is a data race. Edition 2021 lets safe code call
`set_var`, and `std`'s lock is private, so the function cannot enforce the
condition; its doc comment states it as a limitation. This is the class of
RUSTSEC-2020-0159, which `chrono` avoided by reading the zone files itself.

When the read happens differs by C library, which decides both the width of the
window and whether a later change to `TZ` is seen:

- **glibc 2.44, run here** (a C program and a Rust one: `setenv` one zone,
  `localtime_r` and `strftime`, `setenv` another, convert again): the second
  zone is not seen, by `localtime_r` or by `strftime`, however many times they
  are called. A `localtime` or `tzset` call takes the new value. So the zone is
  read at the process's first local-time conversion and then fixed, and the
  window is that first read, unless other code in the process calls `tzset` or
  `localtime`. Nothing in `infr-plat` does; C libraries loaded into the process
  were not checked.
- **musl, run here** (the same Rust program built for
  `x86_64-unknown-linux-musl`, statically linked): the second zone is seen by
  the next `localtime_r`. Every call reads `TZ`, so every call is a window. Not
  a target the project ships.
- **Apple's libc: not checked.** The doc comment says to assume the musl-like
  window.
- **UCRT, from its source only** (`ucrt/time/tzset.cpp` in the Windows SDK):
  `__tzset` initialises the zone once under the runtime's time lock and returns
  early afterwards, reading `TZ` through `_wgetenv_s`. Not run, and how that
  read interacts with a concurrent `std::env::set_var` was not checked.

It is safe today only because the one `set_var` outside tests,
`publish_thread_count` in `infr-cli`'s `main.rs`, runs in `main` before any
other thread exists (found by a grep of the workspace's `.rs` files for
`set_var` and `remove_var` on 2026-11-23; the others are in
`infr-vulkan/tests/gpuav_support` and `infr-metal/tests/pcache.rs`, test
binaries that do not render templates).

What would close it, cheapest first:

- Move the workspace to edition 2024, where `set_var` is `unsafe` and the
  obligation sits on the writer, which is where it can be discharged. The
  migration point is already marked at `publish_thread_count`.
- Deliver the thread count to rayon without the environment (build the global
  pool with `ThreadPoolBuilder::num_threads`), leaving no environment write in
  production code at all, and guard that with a test over the sources as
  `platform_seam.rs` guards manifests.
- Stop calling the C library: resolve the local offset once at startup while
  single-threaded and format from it, or read the zone data directly. This is a
  date library's job, so it means a new dependency or a sizeable module.

### B233 — `%Z` in `strftime_now` on Windows: failure modes read from source, never run (2026-11-23)

**Tag:** plat · chat · windows · **Kind:** coverage gap, unverified on Windows

`infr_plat::localtime::strftime_local` formats `%Z` through the UCRT's
`strftime`, which writes the system's localized zone name and converts it from
wide characters through the time locale's code page; the startup "C" locale sets
that to `CP_ACP` (`ucrt/locale/nlsdata.cpp`, `ucrt/time/strftime.cpp`). Read
from the SDK's source, not run on a Windows machine:

- A zone name with non-ASCII characters, on a machine whose ANSI code page is
  not UTF-8, comes back as bytes that are not UTF-8, which `strftime_local`
  refuses as `StrftimeError::NotUtf8` so that no replacement character reaches a
  prompt.
- If the conversion back from wide characters fails, `_Strftime_l` returns 0.
  `field_text` cannot tell that from a full buffer and reports
  `StrftimeError::FieldTooLong`. When that conversion fails was not traced.

Either way a chat template whose `strftime_now` format contains `%Z` fails to
render on such a machine; formats without `%Z` are unaffected, and no cached
template is known to use it (not surveyed). What is tested: `field_text` on
Windows-1252 bytes, on any platform; CI's Windows leg runs `%Z` only under
`TZ=XXX-5:30`, an ASCII name. To verify: run `strftime_now('%Z')` on a Windows
machine with a non-English display language. If it matters, the fix is to take
the name from the wide API (`wcsftime` or `GetTimeZoneInformation`) and convert
it to UTF-8 in Rust.

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
- **CPU goldens on Windows** are never asserted in CI: `cpu-goldens` is
  Ubuntu-only. Since `infr_core::math` (plan 11, S2) every case has one hash for
  every platform, identical natively on Windows and under WSL on the dev box; a
  Windows leg for the goldens is plan 11's S3.
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
- `linear_mmv_mw_at` is unreachable while `unified_mmv_row1` is hard-coded true,
  so the `mmv_mw.warps` tuning parameter (alias `INFR_MMV_MW_WARPS`) reaches no
  dispatch. Its range is the UNION of the built warps counts (1, 2, 4, 8, 16),
  but `gemm::native_mmv_mw_spv` builds only some per format (Q4_K all five;
  Q6_K, Q5_K, Q2_K, Q3_K 4 and 8; the 32-block formats none), so if the path is
  re-enabled a per-format value with no build panics: validate per qualifier
  against that table first.
- Two tests hold by construction and prove little:
  `mmv_choice_offers_only_built_variants`' "offered ⇒ built" half, and the new
  gate-vs-table asserts for `native_mrow` / `native_gemm_fma` in `linear.rs`.
- **Not read by that pass:** the bodies of the native GEMV / mmv / GEMM kernels
  (dequant and indexing), `streamed_prefill_gemm` (since removed),
  `native_sg_choice` / `native_rm_choice`, the DeltaNet recurrence maths, the
  `mla.comp` body, and the `argmax_prob`, `quant_kv`, `store_kv_dense`,
  `quant_turbo` and `embed_gather` shader bodies.
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
(`streamed_prefill_gemm`, since folded into `adapter::quant_linear`,
`native_sg_choice`, `native_rm_choice`, the `native_gemm_fma.comp` body, the mmq
shader headers). Nothing was built or run by it. The `native_gemv_rm_v2` finding
was re-read at its site afterwards; the rest is the reviewer's account.

- **Streamed quantised Linear routing: what is still unverified.** Since
  2026-11-15 resident and streamed quantised Linears share `quant_linear` (plan
  08 step 4), so the earlier divergences (`INFR_NO_GEMM_WARP`, the per-row GEMV
  at small m) are gone; the opt-in tiers a streamed slot skips are in B193.
  Verified on `Vulkan0` and `Vulkan1` by `tests/streamed_nc_gemm.rs`. Not
  verified: Intel Arc; a device without int8 dot (the fma arm only); a whole
  streamed model with bf16 weights; the oversized-grid fallback on a GPU (only
  the host-side `tiled_gemm_fits` test covers the predicate, and nothing
  dispatches a grid past the x limit).
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
- Read with nothing found: `streamed_prefill_gemm` (since removed),
  `native_sg_choice`, `native_rm_choice`, the `native_gemm_fma.comp` body, and
  the header blocks of the mmq shaders (not diffed against a llama.cpp
  checkout). Not reached: `matmul_native_off`, `matmul_native_splitk`,
  `matmul_native_f16a`, `matmul_native_f32a`, `dispatch_gemv_chunked`, and the
  ignored GPU test bodies in `recorder.rs`.

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
- **A wholly resident layer's experts all rank as used, routed or not.** The
  cost is gone (`all_resident` and `touch_all_hits` are one call per layer and
  role: a block group in `Pager`, `pager::groups`), but the policy was kept on
  purpose: a group touch is defined as an MRU touch of every member in id order,
  and the recorded Qwen3-30B-A3B runs (18 GB and 4 GB/12 GB budgets) make
  identical eviction decisions before and after. So after a fast-path token
  every expert of the layer ranks above anything used earlier, in id order, and
  a miss that reaches such a layer takes its lowest-numbered expert, not its
  least-routed one. Ranking by routed use needs the routed ids on the host — the
  readback the fast path exists to avoid — or a usage record the GPU writes and
  the host reads later, off the critical path. Needs a decision, after a `sim`
  comparison on an almost-resident trace whether any other ranking misses less
  there; none was run.
- **The fast path still copies a LUT window per layer and role per token.**
  `lut_window` writes `n_expert` words into the tape (512 B per layer and role
  on Qwen3-30B-A3B) even when nothing in the layer moved. Considered and
  declined for now: reusing an earlier window would need a per-group residency
  generation plus knowing that the tape region still holds it — the tape cursor
  starts at zero for every `execute_static` call and resets at every
  `sync_stream`, so a window outlives neither. Not measured on its own.
- **`INFR_SUBMIT_DISPATCHES=0` does not stop the submit splitter arming, though
  its warning says it does.** The `ARMED` warning in `infr-vulkan`'s `lib.rs`
  tells the user to set it to 0 to disable splitting, but a cap of 0 is the
  never-split state the arming check starts from (`cur == 0` and a forward past
  `SUBMIT_DANGER_NS` arms it), so 0 changes nothing. Seen on 2026-11-13: a paged
  Qwen3-30B-A3B bench with the variable at 0 still reported `submit split/16`.
  Either the warning or the knob is wrong; no way to keep a dGPU from splitting
  exists today. Matters for measurements: whether the host's recording is on the
  critical path depends on it.
- **`GpuPager::ensure_resident` (test-only) does not roll back a failed
  upload.** `Pager::touch` admits the block before `vk.upload` and
  `copy_into_slot` run, and an error from either leaves it resident (and now
  counted in its group) with no bytes and no LUT entry. The production paths go
  through `fill_or_roll_back`; only the paged-kernel parity fixtures call this.
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

### B158 — chat-template renders: what still differs from Python Jinja2 and llama.cpp (2026-11-13)

**Tag:** chat · correctness

**The rule (decided 2026-11-13):** infr renders chat templates as Python's
Jinja2 does in the environment Hugging Face transformers builds for
`apply_chat_template` (`_cached_compile_jinja_template` in
`transformers/utils/chat_template_utils.py`: `ImmutableSandboxedEnvironment`,
`trim_blocks`, `lstrip_blocks`, `loopcontrols`, its own `tojson`), since that is
what a model was trained on. Where llama.cpp's engine departs from Python, infr
follows Python. The oracle is `crates/infr-chat/tests/fixtures/jinja2_text.py`
(Jinja2 3.1.6 on Python 3.15.2 here); its table of how each construct turns each
kind of value into text is checked row by row by `tests/jinja2_text.rs`, whose
`KNOWN_GAPS` holds every row infr does not match, each pinned to today's output.

**Real templates, checked 2026-11-13.** Each cached GGUF's template
(Qwen2.5-0.5B, Qwen3-0.6B / 1.7B / 30B-A3B, Qwen3.5-0.8B, gemma-3-1b,
Llama-3.2-1B, SmolLM3-3B, ERNIE-4.5-0.3B), nine conversations each (plain,
system, multi-turn with and without thinking, one and two tools, tool-call
history with none / bool / `1e-07` / `1e+16` / `1e+22` / nested arguments and
tool results), through the oracle and infr: all 81 match (three are refused by
both: Llama 3.2 with two tool calls, Gemma 3 on the tool role). The plain-chat
renders are byte-identical to infr 0.3.0's. The 74 templates in llama.cpp's
`models/templates` (at `5676dfbe0`), nine conversations with thinking on and off
(1332 renders, scratch-only: that directory's README gives provenance but no
licensing, so none is copied into this repo; re-run 2026-11-14 once the
`{% generation %}` tag was supported, `crates/infr-chat/src/generation.rs`):
1084 match, 178 are refused by both, 36 differ, 34 are refused by Python only,
none by infr only. Every non-match:

- **`~` on a float (26 of the 36 that differ):** GigaChat3 and GigaChat3.1
  (`' // default: ' ~ default`, `minimum` / `maximum`) and functionary v3.2
  (`' Default=' ~ default_value`, `Maximum=`, `Minimum=`) print a float default
  or bound with Rust's digits: Python `// default: 1e-06`, `// maximum: 1e+16`;
  infr `// default: 0.000001`, `// maximum: 10000000000000000.0`. A list default
  now matches (`crates/infr-chat/src/pyvalue.rs`).
- **functionary v3.1 and Markup (the other 10):** it writes
  `"Use the function '"|safe + name + …`. In Python a `Markup` (`|safe`) plus a
  plain string escapes the plain one even with autoescaping off, so the
  reference prompt reads `&lt;now&gt; &amp; &#39;later&#39; &#34;x&#34;` and the
  whole tool JSON is entity-escaped. minijinja concatenates a safe string
  plainly. Surprising, not fixed: no other checked template relies on it.
- **Python refuses, infr renders (34):** Cohere command-r-plus and Hermes 2 Pro
  / Hermes 3 tool-use iterate `tools` when it is none (Python raises; minijinja
  loops zero times, B170); llama.cpp's own `llama-cpp-deepseek-r1` passes a
  `map` generator to `tojson` (Python cannot serialize a generator; infr's
  `tojson` walks it).

What is left:

- **Known gaps against Python** (rows in `KNOWN_GAPS`):
  - `~` on a float: minijinja concatenates with its own `Display`
    (`ops::string_concat`), which has no hook, and a float cannot be wrapped the
    way the request's lists and dicts are, since it must stay a number. Outside
    `1e-4 <= |x| < 1e16` it comes out in fixed notation, `nan` as `NaN`, and a
    tie rounds up (Rust's `{}`). Reached by GigaChat3 / 3.1 and functionary v3.2
    (above). A list or dict the template builds itself (`'x' ~ [1, 'a']`) is
    also written JSON-style; no checked template does that.
  - minijinja has no tuple and no range type: `(1, 2)`, `dictsort`, `items` and
    `range(3)` print as lists.
  - A float in a `str.format` field with a spec but no type (`{:>20}`) gets
    minijinja's six-digit `%g`; only an argument that meets nothing but
    spec-less fields is converted first (`crates/infr-chat/src/pyformat.rs`).
  - `'%r' | format`, `'{!r}'` and `'{!s}'` are refused by minijinja's formatter.
  - An unassigned code point (Unicode category Cn) inside a container is written
    raw; Python escapes it (`is_printable` in `crates/infr-chat/src/pystr.rs`
    covers every other non-printable category, checked against every such
    character in Python's Unicode 17.0 table).
  - A list or dict the template builds (`tools.l | list`, a slice, `+`,
    `dict(...)`) is minijinja's own value: `~` writes it JSON-style
    (`["string", "null"]`), while the request's lists and dicts print as
    Python's. No checked template concatenates one.
  - `capitalize` upper-cases the first character where Python title-cases it:
    `'ß'` gives `SS` (Python `Ss`), `'ǆ'` gives `Ǆ` (Python `ǅ`). The standard
    library has no titlecase mapping, and a hand-written table was declined.
  - `is sequence`: minijinja's `map` gives a list (Python an iterator: false)
    and its slice a lazy iterable (Python a list: true), so those two answer the
    other way round. `reverse` matches.
  - minijinja's formatter applies some specs its own way: `'{:,}'.format(false)`
    gives `False` (Python formats the integer, `0`) and `'{:.3}'.format(5)`
    gives `5` (Python refuses a precision on an integer).
  - Deliberate limits: a value nested deeper than `tojson::NESTING_MAX` is
    refused where Python prints it, and minijinja refuses a string repeated past
    100,000,000 bytes (`MAX_REPEATED_STRING_LEN`).
- **llama.cpp departs from Python here, and infr follows Python** (from the same
  table, llama.cpp at `5676dfbe0`): bare `{{ none }}` and a list or dict printed
  bare come out empty or as concatenated items; `join` refuses non-strings;
  printf-style `| format` substitutes nothing (`'%s %s' | format(true, none)`
  prints `%s %s`); `none | default('d')` gives `d`; a float literal in exponent
  form (`1e16`) is refused; a small float prints as `0.0` in bare `{{ }}`,
  `join` and `str.format`; `1e16` prints `10000000000000000.0` bare; a string
  holding both quote kinds is repr'd in double quotes (Python: single quotes and
  `\'`), which shows in SmolLM3's `tool | string` for such a description.
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
  (single quotes). infr's `string` filter now writes that (B158), checked by
  rendering this template with `xml_tools` set to a tools list.
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
to that job's download list, bump its cache key and drop the `#[ignore]`; one
hash serves every platform (plan 11).

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

### B163 — whether `schedule` should close its small gap to Belady on a cyclic sweep (2026-11-13)

**Tag:** pager · policy · **Needs:** a decision, low priority

`Pager::schedule` keeps a stable resident prefix on a cyclic sweep:
`n_slots − 1` hits per pass. Belady's MIN rotates which blocks it keeps and does
a little better; in steady state the difference tends to
`(n_slots − 1)/(n_blocks − 1)` misses per pass (simulated with `pager::sim`:
0.3375 per pass at 10 blocks / 4 slots over 400 passes against 1/3, 0.1925 at 40
/ 8 against 0.1795; short runs carry an end effect of a few misses). The docs
and the test that used to call `schedule` Belady-optimal were corrected in the
pager-instruments slice (`schedule_cyclic_sweep_keeps_a_stable_prefix`).

Open: is the gap worth closing? On the recorded dense trace (Qwen3-1.7B,
`paging.cache=400m`, iGPU; `docs/plan/06-pager-eviction.md`, "Measurements")
Belady saves 0.5–5.8% of each pool's misses, the most on the pool with the most
slots (pool0: 2600 against 2449). A rotating policy would give up the stable
prefix's fixed slot assignment, which
`schedule_stable_prefix_keeps_its_slots_across_passes` pins, for at most that.
Not worth it unless a dense run is found where those uploads dominate.

### B165 — the slot-indexed pager: what step 2 left (2026-11-13)

**Tag:** pager · **Needs:** nothing; records

`infr_core::pager::Pager` now keeps its state by slot with recency as a linked
list (`docs/plan/06-pager-eviction.md`, step 2). Checked against the old
implementation, kept as `pager::reference`, by a differential test; what that
does not cover, and what was decided:

- **Behaviour that may surprise: the group-touch path's timing moves with code
  it does not run.** Resident decode with group touches measured 4.2–4.7 ns per
  group touch slower than `HEAD` with the first build of step 2 and 3–4 ns
  faster with the final one, which changed only the id hash's finish and a pin
  check on the miss path. Read a few nanoseconds per call on that path as code
  placement, not as a change, unless it holds across builds.
- **Considered and declined: a direct-index `id → slot` table.** Measured 1–4 ns
  per call faster than the FxHash map, but it sizes itself to the largest id
  named, and `BlockId` is opaque to the pager: a caller packing sparse ids would
  allocate up to 16 GiB. Worth revisiting only with a constructor that takes the
  id space (`trace::PoolInfo::n_blocks` already carries it for traced pools).
- **Considered and declined: batch-aware victim selection.** A batch of k
  cold-end misses walks past its own k earlier inserts, k²/2 steps; on the
  DeepSeek-scale generated prefill that is 66–127 ns per miss all in.
- **Behaviour that may surprise:** after a group touch the members' own list
  nodes stay where they were, stale, and are unlinked when a victim walk reaches
  them; a group's node is dropped once no member is left in its run. Neither
  changes an outcome (the differential test), but a debugger shows both.
- **Not verified:** the Vulkan pager parity tests and the paged MoE seam tests
  ran on the RX 7900 XTX only, not on the integrated GPU;
  `gpu_seam_paged_moe_matches_scout_oracle` passed in 0.018 s by returning at
  its `need_model!` check: no Llama-4-Scout file is in the local cache. The 18
  GB end-to-end configuration was not rerun. B164's anomaly (`infr-core`'s build
  of the old pager ran ~1.5× slower than the same source built alone) no longer
  reproduces with the new code; its cause was never identified.

### B166 — infr never adds a BOS outside the chat template, whatever `add_bos_token` says (2026-11-13)

**Tag:** chat · tokenizer · **Needs:** a decision on honouring `add_bos_token`

**The rule, read in the code:** nothing in infr reads
`tokenizer.ggml.add_bos_token`. `SeamModel::encode` calls
`Tokenizer::encode(text, false)`, which adds no special token, and `render_core`
(`crates/infr-chat/src/template.rs`) hands the BOS to the chat template only as
the `bos_token` variable. So a prompt starts with a BOS exactly when the
template writes one, for every model, and a raw prompt (no template) never gets
one. llama.cpp prepends the BOS whenever the file sets `add_bos_token`. There is
no ERNIE-specific rule in either engine; for ERNIE 4.5, infr matching the
reference is a consequence of that general policy, not of a check.

**What the reference wants for ERNIE 4.5:** Baidu's `tokenizer_config.json` for
ERNIE-4.5-0.3B-PT has `add_bos_token: false`, and the model card tokenizes the
chat render with `add_special_tokens=False`, so the reference input starts with
the template's own `<|begin_of_sentence|>` and no `<s>`. The unsloth 0.3B GGUF
sets `add_bos_token = true` (the 21B-A3B GGUF sets it `false`), so llama.cpp
prepends `<s>` on the 0.3B file: `llama-completion -sp` prints
`<s><|begin_of_sentence|> User: …`.

**It changes the output.** Measured 2026-11-13 on the CPU unless noted,
`--temp 0`, on the chat-rendered "What is the capital of France? Answer in one
paragraph.":

- llama.cpp F16 with its `<s>`: "The capital of France is Paris."
- llama.cpp F16 with `add_bos_token` overridden to false: "The capital of France
  is Paris. It is located in the heart of the country and is known for its rich
  history, cultural attractions, and international influence." infr F16 on the
  CPU and on Vulkan0 prints exactly this.
- llama.cpp Q4_K_M with its `<s>`: "France is the capital of France."
- llama.cpp Q4_K_M without it: "France is known for its rich history, culture,
  and political power. The capital of France is Paris. …"; infr Q4_K_M on the
  CPU and Vulkan0 starts the same and continues "… is Paris, which is located
  …".

So on this file infr follows the reference and llama.cpp does not. **The open
question is the general policy:** a model whose template omits `{{ bos_token }}`
and whose reference relies on the tokenizer adding it would lose its BOS in
infr. None of the cached families was checked for that. The options are to
honour `add_bos_token` (then this ERNIE file regresses to llama.cpp's behaviour
unless its flag is overridden), or to keep the template-only rule and record it
as deliberate.

**Related, not verified:** the same `tokenizer_config.json` has
`add_prefix_space: false`. The converter writes `add_space_prefix` from that key
(`Ernie4_5Model.set_vocab` in `conversion/ernie.py`), but the unsloth 0.3B GGUF
carries no `tokenizer.ggml.add_space_prefix`, so both llama.cpp and infr prefix
a space (the tokenizer test pins llama.cpp's ids). Whether the Hugging Face
tokenizer agrees was not checked.

Also seen, not a bug: `<|end_of_sentence|>`, which the template writes after
each assistant turn, is in neither engine's end-of-generation list. The 0.3B
model ends its turns with `</s>`, the declared EOS.

### B167 — the ERNIE 4.5 golden runs nowhere by default, and the ocean prompt is a near tie at Q4_K_M (2026-11-13)

**Tag:** tests · ci

`cpu_golden_ernie4_5` and `ernie4_5_tokenizes_as_llama_cpp`
(`crates/infr-llama/tests/cpu_backend.rs`) are `#[ignore]`d for the same reason
as B161: the `cpu-goldens` CI job downloads only Gemma 3 1B and Qwen3 0.6B. The
golden's two 32-token hashes were captured on Windows only; no Linux run has
confirmed them. To run them in CI: add `unsloth/ERNIE-4.5-0.3B-PT-GGUF`
`ERNIE-4.5-0.3B-PT-Q4_K_M.gguf` (240645600 bytes) to that job, drop the
`#[ignore]`s; one hash serves every platform (plan 11).

`cpu_golden_ernie4_5_moe` (the 21B-A3B MoE) is `#[ignore]`d too. Its two hashes
are the same on Windows and Linux (WSL, 2026-11-14), but its file is 13331015456
bytes, which is not a download for the `cpu-goldens` job.

`gpu_seam_matches_cpu_ernie4_5` (retired 2026-11-23 to
`gpu_seam_tolerance_ernie`, which holds the GPU's greedy tokens to the reference
only outside near ties; `docs/gpu-tolerance.md`) ran the ocean prompt on the F16
file (723955168 bytes) because on Q4_K_M it failed on both GPUs: after "The
ocean is a vast," the f32-activation CPU oracle (`generate_cpu_reference`) picks
" vast", while the Vulkan decode, infr's production CPU path and llama.cpp's CPU
pick " dynamic". Prefill logits at that token, measured 2026-11-13 (they
reproduce the review's figures):

| path                         | " dynamic" (11290) | " vast" (12791) | top-1 by |
| ---------------------------- | -----------------: | --------------: | -------- |
| infr production CPU (int8)   |          16.520996 |       16.190592 | 0.330    |
| Vulkan0 prefill (7900 XTX)   |           16.47949 |       16.536148 | 0.057    |
| Vulkan1 prefill (integrated) |          16.530031 |       16.559702 | 0.030    |

The reference backend's own value is not exposed (no reference-mode prefill).
The docs on `SeamModel::generate_cpu_reference`, `seam_vulkan_matches_cpu`,
`CpuBackend::reference` and `generate_dense_cpu_mode` said the int8 activation
error is invisible at Q4_K and above; they now say it can flip a near tie.
`gpu_seam_matches_cpu_ernie4_5_q4_k_m` (retired the same day to the same case)
covered the Q4_K_M kernels on a prompt chosen for its margins instead. Its
9-token answer was clear by at least 1.46 on every path, which also made it
blind to a Vulkan rope base off by 1000x (the 32-token F16 case caught that
one); it did catch a halved attention scale. Neither red path has been shown
again on `gpu_seam_tolerance_ernie`.

### B169 — the llama.cpp logit oracle passes an ERNIE 4.5 with permuted Q/K (2026-11-13)

**Tag:** tests · oracle

`cpu_prefill_matches_llama_debug_dump` scores infr against a `llama-debug` dump
on mutual top-5 containment and a probability cosine above 0.7, floors set for
DeepSeek's quantized noise. With `ernie4_5` mutated to permute its Q/K rows, the
F16 story-prompt comparison still passed (probability cosine 0.9992, shared
top-5 4/5) while the max absolute logit difference it now prints went from
0.0057 to 4.27. So a green run of this test does not show an arch is right; the
printed max difference does, read against the F16 figures in
`docs/plan/02-ernie4-5.md`. Making it a check needs per-file floors (an F16 dump
can be held to about 0.01, a Q4_K_M one cannot: llama.cpp's own CPU and Vulkan
differ by 1.2 on ERNIE 4.5 0.3B Q4_K_M).

### B170 — Jinja behaviour that differs from Python and is not text conversion (2026-11-13)

**Tag:** chat · correctness

Found by the Python oracle table (B158, `crates/infr-chat/tests/jinja2_text.rs`,
the last group of `KNOWN_GAPS`) and the llama.cpp template corpus; each was
rendered through Python Jinja2 3.1.6 and infr. `{% break %}` / `{% continue %}`
(minijinja's `loop_controls` feature) and `is iterable` / `is sequence` were
fixed; these were not, and none is reached by a cached template:

- **Unary minus binds tighter than attribute access**: minijinja parses `-v.i`
  as `(-v).i` and refuses it; Python gives `-(v.i)`. `-(v.i)` works.
- **`2 ** -1` is refused** (Python `0.5`); **`1 / 0` prints `inf`** (Python
  raises); **`1 < nan` is true** (Python false; `nan < 1` and `nan == nan` are
  false in both).
- **`'abc' | float` and `'abc' | int` are refused** (Jinja2 returns the default,
  `0.0` / `0`).
- **`round` rounds half away from zero** (`2.5 | round` is `3.0`; Python's
  `round` gives `2.0`) and takes no `method` argument (`round(0, 'floor')` is
  refused). `replace` takes no `count` argument.
- **`center`, `wordcount` and `striptags` do not exist** (minijinja-contrib has
  `wordcount`, `striptags`, `truncate` and `wordwrap`, but `build_env` does not
  add them); `true | indent(2)` renders `True` where Python raises.
- **Iterating none is an empty loop** in minijinja; Python raises. Cohere
  command-r-plus and Hermes 2 Pro / 3 tool-use iterate `tools` and so render in
  infr for a conversation without tools, where Hugging Face refuses.

### B187 — `{% generation %}` as `{% with %}` is looser than Jinja2 in a macro-only corner (2026-11-14)

**Tag:** chat · compatibility · **Needs:** nothing now

`rewrite_generation_tags` (`crates/infr-chat/src/generation.rs`) compiles the
tag as `{% with %}`. transformers parses the body into a `CallBlock`, a macro
body, so three things differ, none of which a template that renders on Hugging
Face can show:

- `{% break %}` / `{% continue %}` directly inside the tag inside a loop: Jinja2
  raises a syntax error ("outside loop"); minijinja's `parse_with_block` keeps
  the loop state, so infr accepts it.
- `caller`, `varargs` and `kwargs` read inside the tag are the body macro's own
  in Jinja2 (`{}` / `()` at top level, "No caller defined"); under `with` they
  are the enclosing macro's or undefined.
- A tag written with non-ASCII whitespace before `%}` (a no-break space) is
  accepted by Jinja2 and refused by infr with "takes no arguments"; minijinja
  without its `unicode` feature refuses such whitespace in any block.

Evidence: the Jinja2 side was run (3.1.6, the oracle's environment); the
minijinja side was read in its parser, not run. The unit tests' expected strings
in `generation.rs` are hand-written constants checked once against Jinja2, not
rows the oracle regenerates; moving them into the `jinja2_text.json` fixture
would stop them drifting. The corpus counts under B158 come from a scratch
harness over llama.cpp's templates that is not checked in.

### B171 — minijinja 3.x is out; infr is on 2.24 (2026-11-13)

**Tag:** deps

`Cargo.lock` holds `minijinja` and `minijinja-contrib` 2.24.0, the newest 2.x;
3.1.0 is published. A new major version is its own change: `build_env`
(`crates/infr-chat/src/template.rs`) wraps several of minijinja's filters by
their signatures (`indent`, `format`, `batch`, `slice`, the string filters), and
the Python text rules lean on `Value::is_kwargs`, `Kwargs::args` / `peek`, the
formatter hook (`pystr.rs`, `pyformat.rs`) and on wrapping a `DynObject` to
replace only its `render` (`pyvalue.rs`). Check whether 3.x gives `~` a hook or
tuples a type before porting: either would close a B158 gap. Not read. A way to
register a statement there would also let the source rewrite that compiles
`{% generation %}` as `{% with %}` (`crates/infr-chat/src/generation.rs`) go.

### B173 — what the CPU backend still leaves on dense prefill and decode (2026-11-14, re-measured 2026-11-15)

**Tag:** perf · cpu

On 2026-11-15 (CHANGELOG: pool threads kept off each other's SMT siblings, the
decode Q4_K/Q6_K dot rewrite, blocked prefill attention) Windows CPU pp512 is
level with llama.cpp on Qwen3-1.7B and LFM3-1.7B Q4_K_M and decode is within
about 6%. What is left, from `INFR_PROF_OPS=1` on one warm pp512 forward of the
final build, 16 threads:

- **Qwen3-1.7B, 435.4 ms**: the 2048x6144 Q4_K gate/up GEMMs 138.17 ms (2467.3
  us per call, 12.86 GFLOP each, about 5.2 TOPS int8), `Attention` 77.63 ms
  (2772.4 us per layer), the 2048x2048 Q4_K Linears 53.02 ms, the Q6_K
  `ffn_down` 43.68 ms, `GatedAct` 18.17 ms. LFM3-1.7B measured 399.4-444.8 ms
  over three forwards in the same window (the box was not quiet: other sessions
  share it), the 2048x7680 gate/up 148.70-165.23 ms of it.
- **Attention.** Single-threaded on a Qwen3 pp512 layer (34.49 ms): taking the
  `exp` out left 32.62 ms, taking the PV accumulation out as well left 20.37 ms,
  so the q·k dots and their bookkeeping are about 60% and PV about a third
  (measured with the PV chunking declined below in place, which changed
  nothing). A register-blocked q·k tile (several queries against several keys,
  each dot keeping `dot`'s eight-lane FMA order so the result stays
  bit-identical) is the untried next step; see "considered and declined" for
  what did not help.
- **Decode** (Qwen3-1.7B tg128 56.72-56.83 against llama.cpp's 59.88-60.24;
  single-threaded it is level: 31.66 against 31.82 tg32). The gap is in scaling
  past 8 threads; not profiled further. Candidates: the per-op handoff (about
  480 ops per token), the serial ops between the GEMVs, and the activation
  quantization the q/k/v and gate/up Linears each redo on the same input
  (`quantize_q8` measured 1.27 us per 2048-element row, so at most a fraction of
  a millisecond per token).
- **A fresh process's first forward pays the mmap page-in**: about 3 s on
  Qwen3-1.7B whatever the thread count (single-thread pp512 first rep 70.37 t/s
  against 118.05-118.16 warm; 16 threads, pp64 first rep 20.18 t/s against
  1076.55-1096.52), so the soft faults serialize. `infr bench` pp runs no longer
  show it (the warmup is the timed prompt itself; Qwen3-30B-A3B's first pp512
  rep 417.37 against 415.36-418.04), but a real first request does. The CPU
  bench's warmup is now a whole timed rep, so a MoE model's first tg rep no
  longer pays for experts the one-token warmup never routed to (Qwen3-30B-A3B
  tg64 reps 3.68, 26.24, 25.24 and 3.66, 27.58, 27.68 before; 26.53, 27.02,
  27.20 and 27.32, 27.43, 27.17 after). Windows gets no `WillNeed` hint (the
  `madvise` in `Gguf::map_and_parse` is `cfg(unix)`); whether
  `PrefetchVirtualMemory` maps pages in or only reads them into the standby list
  (the file is already cached on a re-run, so the second would not help) was not
  tested.
- **Thread placement is Windows only.** The spin pool's threads select CPU sets
  naming one logical processor per allowed core
  (`infr_plat::cpu::spread_placement`) when they fit one per core. Linux and
  macOS have no placement in `infr_plat::cpu`, so the OS places the pool there;
  whether Linux has the same SMT placement problem was not measured (B76). Under
  a restricted affinity the placement stays inside it: `start /affinity 5` ran
  Qwen3-1.7B on two threads, logical processors 0 and 2 at 98.46-100% and 1, 3
  and 30 at 0-5.97% (`typeperf`). Not measured or not verified:
  - A hybrid part. The fastest-class preference is unit-tested only, and whether
    the default thread count should be the P-cores alone there (it counts every
    allowed core, so an 8P+16E part gets 24 threads, of which the E-cores may
    slow a spin-waiting pool) is unmeasured.
  - More than 64 logical processors. On Windows 11 a process's threads span
    every group, yet `process_mask` (in `infr_plat::cpu`) reads
    `GetProcessAffinityMask` and `GetProcessGroupAffinity`; if the latter
    reports a single group (the primary one) and the former a non-zero mask for
    it, as the documentation suggests for a process not yet spread over groups,
    `physical_cores` and every placement would be confined to that one group. A
    suspicion from the documentation, untested.
  - Job objects with CPU-set limits, and CPU sets the OS flags Allocated or
    Parked (the flags byte of each record is not read). Untested; a placement
    the OS refuses fails safe (one warning, the OS's placement).
  - Work outside the spin pool: `SpinPool::run`'s rayon fallback (another job
    holds the pool, or `kernels.cpu.spinpool = false`) runs on rayon threads,
    which get no placement. The MoE expert fan-out runs on the spin pool
    (checked: that fallback is the only rayon use left in `infr-cpu` outside
    tests).
  - A process started with default CPU sets (left alone by design).
- **Ideas not tried**: quantizing a shared input once for the Q, K and V Linears
  (and gate/up); building MoE stage C's activations from tiles of the o-major
  gate/up block, which it now reads with stride `count`; a vectorized Q6_K pack
  expansion (`q6k_pack_group_into` is scalar bit-fiddling, paid per call per
  group); a 16-row by 4-token GEMM tile (one activation broadcast feeding two
  weight vectors) for the packed K-quant GEMM.

**Considered and declined, with what was measured (2026-11-15):**

- **Several 8-row groups per GEMM task with a 64-token block** (each task packs
  four groups into thread-local scratch and walks token blocks across them, so
  activations come from L2 rather than L3): LFM3 pp512 forward 389.3, 390.4 and
  398.7 ms against 396.4, 398.1 and 399.7 without, interleaved. Within the
  window's noise once the threads were pinned; not kept.
- **Attention's PV accumulated in 64-element register chunks across the keys**
  (the same FMA chain per element): 2797.1 to 2909.5 us per layer, no gain.
- **Eight keys per `dot_n` in attention instead of four**: 20.37 to 20.86 ms on
  the single-thread layer with `exp` and PV removed, no gain.
- **`SetThreadIdealProcessorEx` as the placement**: Qwen3-1.7B pp512 reps
  1074.94, 930.14, 748.02, 761.68, the same decay as no placement (1072.52,
  991.43, 765.08, 750.60), against 1096.27-1152.17 with one fixed core per
  thread.
- **One fixed core per thread (hard affinity), the first version of the
  placement**: no faster than the CPU-set placement that shipped (one window:
  Qwen3-1.7B pp512 1099.74-1159.29 against 1115.18-1147.24, LFM3-1.7B
  1212.47-1267.58 against 1212.27-1262.83), it overrode a restricted process
  affinity, and every pool took the same cores, so two concurrent `-t 4` benches
  ran at 218.77-225.92 t/s each against 392.25-412.41. The fallback the review
  proposed, a group-affinity mask of first siblings, was not built, since CPU
  sets held the rate.

### B174 — the loose MoE prefill comparison passes a Vulkan router that ignores the bias (2026-11-14)

**Tag:** tests · moe

`gpu_seam_matches_cpu_ernie4_5_moe` scores the Vulkan prefill against the CPU in
the recipe's loose MoE form (mutual top-5, a cosine floor). Run against the
`moe_topk.comp` from before the softmax-plus-bias fix — which, on this model,
routes as if the bias were absent — it still passed: cosine 0.9976271849367727
and max |Δlogit| 0.813328 against 0.9986429426009094 and 0.5761112 with the fix.
The token test on a margin-checked prompt
(`gpu_seam_matches_cpu_ernie4_5_moe_tokens`) is what failed ("one, two, three,
…" against "1. One\n2. Two\n…"). So for a MoE port the loose form shows the
graph is wired, and only a token comparison on a prompt with measured leads
shows the router picks the reference's experts. `docs/plan.md`'s recipe step 3
says only the loose form for MoE; adding the token test to it is the change. Not
done here because the recipe is shared text the next port should decide on.

### B175 — llama.cpp's ERNIE 4.5 MoE loader and graph disagree on layer kinds for an interleave step above 1 (2026-11-14)

**Tag:** moe · reference

`llama_model_ernie4_5::load_arch_tensors` (`src/models/ernie4-5.cpp`) creates
MoE tensors on every layer from `leading_dense_block_count` on, whatever
`interleave_moe_layer_step` says, while the graph (`ernie4-5-moe.cpp`) runs a
layer as MoE only when it is past the lead AND `(il + 1) % step == 0`. With a
step above 1 a file cannot satisfy both: the loader wants expert tensors on the
off-step layers, the graph then runs them dense with tensors it never created.
infr follows the graph (`Config::is_moe_layer`, and `ernie4_5_moe_tensors`
expects dense FFN tensors off the step and refuses MoE tensors there). Every
ERNIE 4.5 MoE file seen has step 1 (the 21B-A3B Q4_K_M), where the two agree.
Not checked: which layout Baidu's checkpoints or the converter would produce for
a step above 1, or what the Hugging Face modelling code does. Revisit if such a
file appears.

### B176 — the renormalized MoE weight sum is clamped at 6.1e-5 in llama.cpp and at 1e-20 in infr (2026-11-14)

**Tag:** moe · numerics

llama.cpp's `build_moe_ffn` (`src/llama-graph.cpp`, the `norm_w` branch) clamps
the selected weights' sum with
`ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY)` (the smallest normal
f16) before dividing (except `LLM_ARCH_DEEPSEEK41`, which adds 1e-20). infr's
CPU backend (`Op::MoeFfn`), the Metal host path, the Metal `moe_topk` kernel and
the parity harness's `moe_ref` use `max(sum, 1e-20)` on the selected
probabilities; Vulkan's `moe_topk.comp` renormalizes exponentials shifted by the
selection's largest logit, whose sum is at least 1, so its clamp never acts and
it divides by the true sum. The two rules differ only when the selected UNBIASED
probabilities sum below 6.1e-5, which a selection bias can in principle cause:
ERNIE 4.5 MoE is the first softmax model with one. On the 21B-A3B Q4_K_M file it
does not come close: `llama-eval-callback` on the chat-rendered "What is the
capital of France? Answer in one paragraph." (19 prompt tokens, one generated)
printed 314 values of `ffn_moe_weights_sum` across the 27 MoE layers (it elides
middle rows), between 0.1847 and 0.8639. No change made; matching the clamp
would only matter for a router whose bias routinely selects
near-zero-probability experts.

### B177 — the CPU f32 expert dot keeps hand-written Q4_K/Q5_K/Q6_K/Q8_0 decoders beside `factor_block` (2026-11-14)

**Tag:** perf · cpu · duplication

`dequant_row::decode_span` decodes Q4_K, Q5_K, Q6_K and Q8_0 through its own
`q4k_block`/`q5k_block`/`q6k_block`/`q8_0_block`, which repeat the bit layouts
`infr_gguf::dequant::factor_block` owns; the other six affine formats go through
`factor_block` + `expand_factored` (the decode `bytes_to_f32` uses). Routing the
four through the shared pair was tried and measured slower, interleaved, warm
`tg64` reps on the Windows box: ERNIE-4.5-21B-A3B 12.13-12.19 t/s against
24.75-25.68 for the hand-written decoders, with `factor_block` and
`expand_factored` `#[inline(always)]`, one call site per format with a literal
dtype, the block geometry a `const`, and the dot accumulating in registers; an
earlier shared build without the per-format call sites measured Qwen3-30B-A3B
10.33-10.89 against 26.20-26.88. A single-thread microbenchmark (since deleted)
put the shared Q4_K decode at 78-79 ns per block against 52-55 hand-written and
Q6_K at 79-82 against 105-108, so it does not account for the whole production
gap; the cause was not found. The duplication is held by
`decode_span_matches_bytes_to_f32_bitwise` and the two dot tests, which compare
every format bit for bit against `bytes_to_f32`. Closing it means finding what
the shared form costs in the multi-threaded expert GEMM (a profile of
`expert_gemm_range` on both builds) and making `factor_block`'s hot arms as
cheap as the hand-written ones, then deleting the four copies.

### B178 — codebook and fp4 expert banks still dequantize per row through `bytes_to_f32` on CPU decode (2026-11-14)

**Tag:** perf · cpu · moe

`dequant_row::dequant_dot_rows` decodes only the formats with a factored form.
Every other dtype — the IQ and TQ quants, MXFP4 (gpt-oss's expert banks), NVFP4,
Q2_0 — still takes `dot(&bytes_to_f32(row, dt), x)` per weight row per routed
expert on a single-token MoE step, the allocation-heavy path that made ERNIE and
Qwen3 decode 3.4-3.5× slower before the affine formats moved off it. Not
measured on a model in those formats (none is cached on the Windows box). The
fix is the same shape: a per-block f32 decode into a stack span from the
codebook decoders `dequant_codebook` already has, held bit for bit against
`bytes_to_f32`.

### B179 — what keeping one CPU backend per model left (2026-11-15)

**Tag:** perf · cpu · design

Since 2026-11-15 a `SeamModel` keeps a `seam::CpuResident` (backend, paged
store, bound weights) between CPU generations (`generate_cpu`,
`generate_cpu_constrained`, `generate_cpu_ids`, `bench`). Left as it was, or not
verified:

- **The spin pool is still built per generation.** A pool kept across requests
  prefilled ERNIE-4.5-21B-A3B Q4_K_M about 10% slower (three interleaved rounds,
  six warm requests each, prompt ms: kept 208.9-222.8, 197.4-212.7, 205.0-219.0;
  pool rebuilt per request 178.0-207.3, 183.5-200.9, 184.1-207.5; whole backend
  rebuilt 184.4-199.7, 182.7-200.1, 185.2-201.9), spread over every op in the
  per-op profile rather than one kernel. So each generation ends with
  `CpuBackend::release_pool`. Why a long-lived pool is slower was not found;
  Windows thread scheduling of long-parked workers is a guess. Fixing it would
  save the pool spawn per request, which was not separately measured.
- **The repack cache's gain is unmeasured.** It only serves models with a fused
  `ffn_gate_up_exps` (gemma4 26B-A4B, Diffusion-Gemma), none of which is on the
  Windows box. It now lives as long as the model, bounded by
  `kernels.cpu.repack_mb` (default 4096 MiB) and never evicted, so an idle model
  holds up to that much; before, it was freed at the end of each request.
- **Memory held while idle.** Everything the resident holds now stays allocated
  between requests: the paged store's arena (`paging.dram`), which is what makes
  it a cache; every weight the loader rewrote into owned memory instead of
  mapping it (BitNet i2_s dequantized to f16, qwen2's NEOX-permuted q/k,
  deepseek2's `attn_k_b`/`attn_v_b` expanded to f32, f16 or bf16 conv weights
  widened to f32); and the backend's `weight_cache` (the dequantized norm
  weights). Under automatic paging (`paging.dram` unset), both whether to page
  at all and the arena's size are decided once, from the host memory available
  at the model's first CPU generation.
- **No way to drop a model's resident short of dropping the model.** It matters
  in a mixed process: a model loaded with `paging.dram` that runs a CPU oracle
  through `generate_cpu_ids` beside its GPU sessions keeps that CPU arena for
  the model's life. Nothing in the tree does this outside tests, so no API was
  added; a `SeamModel` method that empties the slot (when no generation has the
  resident out) would be the fix, and is small.
- **A second generation while another holds the resident** opens its own and
  drops it at the end, never putting it back, since it was sized while the
  model's own was still allocated (its arena may be smaller, or absent).
- **`paging.stats` counts are cumulative** since the model's first CPU
  generation, not per request.
- **Still per call**: `generate_cpu_reference` (a reference-mode backend, the
  oracle path), the CPU MTP path (`mtp::backends`), the `verify_dense_cpu`
  family, and the overflow generation above (with its own paged arena if
  `paging.dram` asks for one, as every generation did before). `infr serve`
  serialises CPU requests behind one lock, so that last case does not arise
  there.
- **Pool placement** is computed on the thread that builds the pool, now the
  thread of each generation as before, so nothing changed there.

### B180 — ERNIE prefill was 1.5× faster with `kernels.cpu.repack_mb=0`; reproduced, gone, cause unproven (2026-11-14)

**Tag:** perf · cpu · moe

Reproduced on the old build (HEAD 62f85532): warm pp512 reps 78-89 t/s at the
default budget against 115-125 at `repack_mb=0`, `Op::MoeFfn` 155 ms per layer
against 82 ms (per-op profile). On the new build the two settings measure the
same (449-463 t/s both, two pairs interleaved). The only thing the budget
changed on ERNIE was whether the dense Q6_K packs were kept (the MoE split
gate/up never used a pack); the MoE GEMMs it slowed then allocated a few dozen
buffers per 8-row group per call (`vec_dot_q4k_batch8_ilv_vnni`, blocks of 20 KB
and more, above the heap's low-fragmentation sizes) and faulted fresh pages for
each stage buffer. Those allocations are gone and the Q6_K pack cache was
removed, so the plausible mechanism (heap behaviour under the retained packs)
can no longer be tested on the current code. Sampled with `typeperf` over each
whole run (model load included), the old build averaged 353% user CPU at the
default budget and 493% at 0, of the 1600% its 16 threads could use: it was
mostly waiting, not computing. Kept as a surprise, not a bug.

### B181 — the Diffusion-Gemma Q6_K lm_head now packs per call instead of from a session cache (2026-11-14)

**Tag:** perf · cpu · unverified

The dense Q6_K prefill path packs each 8-row group into a per-thread scratch
group per call (`repack::q6k_pack_and_gemm_group`); the session-long Q6_K pack
cache it replaced (`q6k_pack_for`, ~740 MB for gemma's tied 262k-row lm_head)
was removed, because at the time every other CPU path built its backend per
request and the cache cost a full rebuild per call. Since 2026-11-15 the plain
CPU generations keep one backend per model (`seam::CpuResident`, B179), so a
pack cache would now survive between requests there too, at its full size held
for the model's life. For the Diffusion-Gemma session, which keeps its backend,
the cache had been built once; now each denoise step pays one pack pass over the
lm_head (a pass the step's GEMM over all canvas rows shares). Not measured: no
Diffusion-Gemma model is on the Windows box. If a measurement shows a loss, a
per-backend pack keyed as before is the fix, bounded like `repack_cache`.

### B182 — the per-op profile drops a reference-mode prefill Linear (2026-11-14)

**Tag:** perf-tooling · cpu

The `Op::Linear` arm's reference-mode branch ends with `continue`, which skips
the per-op timing at the bottom of the interpreter loop, so `INFR_PROF_OPS=1`
with `kernels.cpu.reference` leaves those Linears out of the table. Noticed
while restructuring the prefill Linear; not changed.

### B183 — what CI exercises of the CPU prefill paths depends on the runner's CPU (2026-11-14)

**Tag:** tests · ci · cpu

The packed K-quant GEMM (`repack::q4k_pack_and_gemm_group` /
`q6k_pack_and_gemm_group`, behind `linear_packed_ok` and `expert_gemm_range`'s
VNNI arm) only runs on a CPU with AVX-512BW and AVX-512 VNNI; elsewhere the same
ops take the per-row batch kernels (`vec_dot_q*k_batch`, and
`vec_dot_q4k_batch8` for Q4_K 8-row tiles). The workflow does not pin a CPU, so
whether a run of `.github/workflows/ci.yml` exercises the packed path depends on
the hardware GitHub assigns, and nothing in the logs says which. What each job
covers:

- `cargo test (ubuntu/macos/windows)` (`cargo nextest run --workspace`):
  `linear_prefill_matches_the_single_token_kernels_bitwise` (lib.rs) and
  `expert_gemm_range_matches_the_single_token_kernels_bitwise` (moe.rs) run on
  every host and compare whatever route it takes against the single-token
  kernels, on both sides of `repack::packed_min_tokens`. The packed-only tests
  (`linear_packed_matches_the_single_token_kernels_bitwise`,
  `q4k_pack_gemm_bit_identical_to_scalar`,
  `q6k_pack_gemm_bit_identical_to_batch`) print a `SKIP` line and pass without
  VNNI. macOS (aarch64) compiles none of the packed code. Since 2026-11-17
  `tests/linear_batch_identity.rs` holds every weight dtype's prefill
  `Op::Linear` to its one-row bits on the host's own route, and
  `kernels::batch_tier_tests` calls the tier kernels directly for each
  one-row/batch pairing an x86_64 host can dispatch (scalar, AVX2, AVX-512BW
  without VNNI, VNNI), skipping the tiers the runner lacks: a runner with
  AVX-512 VNNI covers every pairing an AVX2-only or AVX-512-without-VNNI host
  would run, and any x86_64 runner covers the scalar and (with AVX2) AVX2 ones.
  The packed GEMM still needs a VNNI runner.
- `cargo test (CPU goldens, real models)` (ubuntu, release,
  `--test cpu_backend`): downloads gemma-3-1b and Qwen3-0.6B Q4_K_M only. Their
  multi-token prompts run prefill `Op::Linear` on Q4_K/Q6_K weights (packed with
  VNNI, per-row without), `QkNormRope`, `Attention` (the `dot_n` scoring) and
  the buffer recycler. No MoE model is downloaded, so `Op::MoeFfn`'s changed
  stages (the packed expert tiles, the down-stage scatter, the `dot_n` f32
  router) are not reached by any CI golden; the MoE goldens
  (`cpu_golden_qwen3moe`, `cpu_golden_ernie4_5_moe`) ran only on the Windows dev
  box. Plain `Op::Rope` (SmolLM3's path) is likewise exercised only there.

Closing the gap would mean a small MoE GGUF in the goldens job and a way to
force the per-row route on a VNNI host (or the packed route's absence) so both
are tested on one machine.

### B184 — what still separates Vulkan MoE prefill and decode from llama.cpp on the RX 7900 XTX (2026-11-14)

**Tag:** vulkan · perf · **Blocked on:** nothing; each item is its own slice

Measured on the RX 7900 XTX with the AMD Windows driver (not RADV), pp512 / tg,
`INFR_PROF_OPS=1 INFR_PROF_OP_SHAPES=1` against llama.cpp `5676dfbe0`'s
`GGML_VK_PERF_LOGGER=1`, after the Q6_K expert GEMM, split-K window, router
top-k and router GEMV fixes. ERNIE-4.5-21B-A3B Q4_K_M pp512 device time is 157.5
ms against llama.cpp's 129.4 ms; what is left, largest first:

- **Dense K-quant prefill now runs on int8 coopmat (2026-11-15, CHANGELOG); what
  that slice left:**
  - The GEMM is gated on the AMD proprietary driver (`i8cm_kq_ready`). RADV
    RDNA3/RDNA4 was not measured; the old Q8_0 int8 kernel lost to f16 there
    (`docs/perf/playbook.md`), but that kernel staged a 16-column tile per
    workgroup and is not evidence about this one. Measure there before widening.
  - Q8_0, Q4_0/Q5_0/Q4_1/Q5_1, IQ4_NL/IQ4_XS, MXFP4 and Q3_K prefill still
    dequantize into `native_gemm_warp_*`; llama.cpp runs all of them on the same
    int8 coopmat MMQ on RDNA3. Each is a weight decode in `prefetch` /
    `store_to_shared` of `native_gemm_i8cm_kq.comp` plus a build; the existing
    `native_gemm_i8cm_q8_0.comp` measurement kernel would then be redundant.
  - The MoE expert GEMMs (`native_gemm_mmq_q4k_xp` and siblings, dp4a) do not
    use it; the streamed dense path does since 2026-11-15 (plan 08 step 4).
    llama.cpp's `matmul_id_subgroup_q4_k_q8_1` is the expert analogue.
  - Tiles: two (`I8cmTile::L` 128x128, `M` 64x64) chosen by grid fill
    (`i8cm_kq_choice`). Forcing one tile everywhere on Qwen3-1.7B pp512: the
    fused gate/up 417.1 us (L) against 491.6 us (M); the 2048-wide projections
    103.6 against 101.5 us; `ffn_down` 347.6 / 304.7 against 354.7 / 298.8 us
    (Q6_K / Q4_K); the 1024-wide K/V 101.4 against 64.2 us. No split-K variant;
    no tile between the two was tried.
  - Precision: B192 has the measured cost on real text.
  - The gate is `DriverId::AMD_PROPRIETARY`, which Linux AMDGPU-PRO reports too;
    only the Windows driver was measured. On those devices the int8
    accumulator-layout probe (`verify_i8_coopmat_layout`) now runs at init by
    default, where it used to run only under `INFR_I8_COOPMAT=1`; a failure logs
    an error and leaves the int8 GEMMs off.
  - Q6_K reads `d` and its scale pair as the whole 32-bit word holding them, as
    `native_decode.glsl`'s `ru16` does: for a word-aligned superblock the `d`
    word runs 2 bytes past the superblock's end. That stays in bounds because
    every weight upload is padded to a 4-byte multiple
    (`linear::pad_to_u32_align`): only an odd superblock count ends a buffer on
    a word-aligned last superblock, and its 210-byte tail is then 2 bytes short
    of a word, which the pad fills. The `ql`/`qh` reads end inside the
    superblock.
- **Expert gate/up dp4a GEMM (`native_gemm_mmq_q4k_xp`):** 946.3 us vs 903.8 us
  per call (54 calls), ~2.3 ms per pp512. Small; the Q6_K down that was 3x
  behind is now at parity (1087.5 vs 1071.3 us).
- **Activation quant passes:** `quant_q8_gather` 97.9 us + `quant_q8` 73.2 us
  per MoE layer; llama.cpp's `MUL_MAT_ID` quantizes inside its own pipeline.
- **Decode:** ERNIE tg128 is 187 vs 213 t/s. Per-op the id-indexed expert GEMVs
  are close (`native_idm_q4k` 27.5 us vs llama.cpp 24.5 / 22.3 us,
  `native_idm_q6k_sg2` 39.5 vs 30.3 us); per-token device time (5.0 ms, no
  replay) is below the wall time (5.35 ms), so part of the rest is host/submit
  side and was not profiled. Qwen3-30B-A3B decode per-op is at or below
  llama.cpp's.

**Paged decode throughput** is no longer here: the early-submit cadence
(`paging.early_submit`) replaced the overlap the accidental watchdog split used
to give, and the measured breakdown of a paged decode step and the larger
options it points to are in `docs/plan/06-pager-eviction.md` ("A paged decode
step, host and GPU" and step 5). What that slice found and left is B185.

**Not re-measured:** the README / `docs/perf/results.md` table, which is a RADV
snapshot; nothing here was run on RADV.

### B185 — what the paged early-submit slice found and left (2026-11-14)

**Tag:** vulkan · perf · paging · **Blocked on:** nothing

Found while adding `paging.early_submit` (`adapter::EarlySubmit`); the
measurements are in `docs/plan/06-pager-eviction.md`, "A paged decode step, host
and GPU".

- **The splitter undercounts a paged forward's dispatches.** `execute_static`
  adds to `submitted_dispatches` only the segments its own cut submits and the
  last one; segments submitted by the paged MoE path (`rotate_stream`,
  `sync_stream`) and by dense staging rotations are not counted, while their GPU
  time is in `ForwardClock`'s sum. `submit_cap_from_measurement` therefore sees
  more GPU time per dispatch than there was and, once armed, picks a smaller cap
  than it should: the safe direction, but the cap on a paged iGPU forward is
  tighter than measured. Read in code, not measured. Fix: carry a dispatch count
  on `PagedStream` and add it.
- **The early cadence costs ERNIE-4.5-21B-A3B about 1.5% on the router-readback
  path with nothing evicting** (`paging.cache=20g`, `-p 0 -d 0`: 141.6, 141.5 →
  139.8, 139.1 t/s, interleaved). There every MoE layer drains the queue for its
  router ids, the segment between drains is about 28 dispatches, and a cut costs
  12-18 µs of host time (a submit, a fence, a timestamp query pool, a new
  command buffer) against little recording left to hide. Qwen3-30B-A3B gains on
  the same path (116.1, 117.5 → 120.5, 120.3). That configuration needs a decode
  with no prefill (a prefill of 48 or more tokens makes every layer resident at
  that budget and takes the inline path), so it was left; a cadence that does
  not restart after a drain is the obvious variant to measure.
- **`seed_barrier` at an early cut could not be shown to matter here.**
  Commenting it out left `paged_early_submit` and its bit-identity checks green
  on the RX 7900 XTX (three runs): the AMD Windows driver evidently serializes
  submits on the queue. The barrier stays (it is what the Vulkan spec requires),
  but the tests do not prove it on this hardware; RADV or another vendor might.
- **Considered and declined:** busy-polling the fence instead of
  `vkQueueWaitIdle` at the per-layer router drain (`sync_stream`) — 75.1-76.0
  against 80.9-81.7 t/s on Qwen3-30B-A3B at `paging.cache=4g paging.dram=12g`;
  and splitting an expert's staging memcpy across the rayon pool
  (`par_copy_to_mapped` at 256 KiB chunks instead of 4 MiB) — 62.2-62.5 against
  81.4-81.8 t/s. The single-thread copy into the ReBAR ring already runs near
  the PCIe link rate. Also declined: carrying the early cadence on across a
  staging-ring rotation (`rotate_stream` leaves the GPU busy, unlike a drain),
  which was 1-2% slower on the evicting Qwen3-30B-A3B decode and level on dense
  streaming; figures in plan 06 next to the fixed-count variant. Why starting
  over wins there was not measured.
- **ERNIE's paged decode after a prefill starts with a burst of misses.** At
  `paging.cache=4g paging.dram=12g -d 365`, a 64-forward window right after the
  prefill averaged 21.6 ms per forward (11.6 ms of it staging), later windows
  7.1-7.2 ms: the prefill sweep evicts the decode hot set and decode pays to
  refill it. `tg128` (88-90 t/s) averages over both. A policy question for step
  3 of the eviction plan.
- **Paged decode with no prefill at `paging.cache=4g` is far below the same
  budget after a prefill** (Qwen3-30B-A3B 50-52 t/s at `-d 0` against 82 at
  `-d 365`). Not investigated. A guess, unchecked: the host tier starts empty,
  so the decode reads its misses from disk until it fills.

### B186 — what the MoE fill slice found and left (2026-11-14)

**Tag:** vulkan · perf · paging · **Blocked on:** nothing

Found while adding fills (`MoePagerSession::request_fills`, `stage_ready_fills`,
the drain arm of `adapter::execute_paged_moe`); the measurements are in
`docs/plan/06-pager-eviction.md`, "Filling a cache that holds the model".

- **A short decode from no prefill still pays for the fill reads.** At
  `paging.cache=20g`, tg128 with no prefill is about 3% below the build without
  fills on Qwen3-30B-A3B and 2% on ERNIE (plan 06 has the interleaved figures;
  the first fill build was 6% and 5% below), and the A/B there pins it on the
  reader's file reads themselves: with the read replaced by a 2 ms sleep and
  everything else unchanged (staging, copies, partial residency) the forwards
  ran at the no-fill rate, and with `paging.stats` the decode made no demand
  miss in either timed rep, so the reads are not delaying routed misses on the
  shared handle either. (The sleep run filled garbage, so its tokens differed; a
  decode step's time depends on its tokens only through routing, and that run
  missed 25 experts in its second 128 forwards and none in its third, by a
  temporary print.) What in the reads slows the forward was not found: CPU or
  cache interference of the kernel's buffered-read copy, interrupts, or the file
  cache are all unchecked guesses. Two things to try, both outside the fill
  code: a file handle of the reader's own opened for overlapped or unbuffered
  I/O (`FileBlockIo` reads through one synchronous handle, B35), and reading a
  run of a bank's adjacent experts in one call (`HostPager::read` reads one
  block).
- **Fill throughput is the reader's, and it is slow.** One background thread
  reads one expert at a time; an expert no token routed to was never read, so it
  comes off the drive cold. Timed with a temporary print in the reader on
  Qwen3-30B-A3B (`paging.cache=20g`, default DRAM tier): 1.3-6.1 ms per 0.9 MB
  expert, with stalls of 48-200 ms. More reads in flight would shorten that,
  under the same handle caveat as above.
- **Only a cache that holds every expert fills.** A budget that holds most of
  the model (ERNIE at `paging.cache=10g` holds its routed set, not every expert)
  never fills and keeps the readbacks. Filling there would need a policy for
  which free slots a fill may take without starving a later routed miss; not
  designed. The GPU-side LUT with a miss flag and redo (plan 06, step 5) is the
  general answer for those budgets.
- **The demand gate is per layer and per forward.** A layer requests fills only
  when no expert was missed since its previous call; one miss anywhere cancels
  the queued reads. A decode that keeps missing now and then (a long, varied
  generation from a cold cache) therefore fills slowly. A stricter gate (eight
  whole forwards without a miss before any read) was measured and was not faster
  at tg128 (plan 06), so the softer direction (a miss rate) is the one left
  untried. Because any demand miss cancels every queued fill, a varied
  generation may not start filling until its routed set is nearly warm; the
  end-to-end test (`gpu_seam_paged_moe_fills_until_wholly_resident`) repeats one
  prompt, so it does not show this, and it is unmeasured.
- **Tried and declined**, each against interleaved runs of the build without
  fills (figures in plan 06): the forward copying each read expert from a host
  buffer into the staging ring (the first version of this slice); the reader
  reading straight into a ReBAR staging buffer (the forward's copy is gone, but
  the reader's writes cross the bus; no faster than host memory, and it takes
  VRAM outside the cache budget); submitting the fill copies on their own right
  after the drain (about 50 us of host time per submission on the critical path,
  measured with a temporary timer); copies without the arena barriers around
  them; one fill per drain; throttling the reader's writes to 64 KiB per 20 us.
  None closed the gap the reads leave. From the first version: fills read
  synchronously in the forward (bounded at 128 MiB per forward, `infr run`
  decode fell from 5.6 to 2.5 tok/s; at 2 ms, bench tg128 85.3-85.9 t/s); a
  reader with no demand gate (`infr run` decode 6.8 to 4.7 tok/s); gating per
  call rather than per layer's forward (5.6 tok/s). Single runs, CPU load not
  recorded for those three.
- **`infr run` at `paging.cache=20g` prefills a 9-token prompt in about 6.7 s**
  (Qwen3-30B-A3B, before and after this change alike). Not investigated; likely
  the same cold reads as B185's last item.
- **`infr-vulkan/src/pager.rs` grew by the fill machinery** (`FillReader`,
  `FillQueue`, `FillMemory`, `Fills`, and `MoePagerSession`'s `request_fills`,
  `stage_ready_fills`, `fills_copied`, `end_forward`), which is a seam of its
  own: a `pager::fill` child module could hold all of it, the session methods as
  an `impl` block there. Not split in this slice, which was scoped to the
  existing files.
- **Not verified:** a model whose expert pools mix slot sizes (UD quants), where
  one role's pool can hold every expert and another's cannot — such layers never
  fill, by `MoePagerSession::fills_layer`, but no test runs one. Throughput was
  measured on the RX 7900 XTX only; the integrated GPU ran the fill tests for
  correctness, not for speed.
- **Test gaps:** no test kills the reader inside a session or a forward (the
  dead-reader path is tested on `FillReader` alone, in `pager.rs`), and
  concurrent forwards on one backend were not reviewed against
  `MoePagerSession::fills_copied`, which frees every staging slot whose
  recording's serial is at most the one just waited. The case to check: a second
  forward on another thread holding a recording with a lower serial that is not
  yet submitted, whose slots that call would free while its copies have still to
  run. Not reviewed.
- **The reader is still joined under the `moe_pager` lock on the rare paths:**
  `end_forward` hands it back to be dropped outside the lock, but turning fills
  off (`MoePagerSession::fills_off`, on a failed fill or a dead reader) and the
  last staged slot of an off session (`release_staged_fills`) drop it in place,
  waiting for a read in progress. Once per session at most; left as is.

### B188 — LFM3 against llama.cpp: CPU closed, Vulkan0 attention and head gate left (2026-11-14, profiled 2026-11-15)

**Tag:** perf · lfm3 · **Blocked on:** nothing; what is left is the Vulkan0
attention and head gate, each its own slice (the CPU remainder is in B173)

`LFM3-1.7B-Q4_K_M.gguf`, `infr bench` against `llama-bench` (llama.cpp
`5676dfbe0`), measured on 2026-11-14 with the CPU 1–3% busy before each run
(`typeperf`); no other session was asked to pause:

| device                | infr pp512 | llama.cpp pp512 | infr tg128 | llama.cpp tg128 |
| --------------------- | ---------: | --------------: | ---------: | --------------: |
| Vulkan0 (RX 7900 XTX) |     7259.0 |        10103.09 |      488.6 |          379.54 |
| Vulkan1 (integrated)  |      138.3 |          195.94 |       23.6 |           25.11 |
| CPU (16 threads)      |      565.5 |         1082.25 |       51.3 |           60.43 |

Every infr CPU pp512 mean includes a cold first rep near 160 t/s (the mmap
page-in of B173). Profiled on 2026-11-15 with `INFR_PROF_OPS=1` (plus
`INFR_PROF_OP_SHAPES=1` on Vulkan) against llama.cpp's `GGML_VK_PERF_LOGGER=1`.

**What the profile pinned on LFM3, now fixed:** the CPU `Op::ShortConv` took
4130.8-4279.0 us per layer per pp512 chunk, 53.70-55.63 ms of a 661.6-665.1 ms
warm forward, in serial index arithmetic around a scalar conv; it now takes
389.8-435.3 us (CHANGELOG). The rest is not LFM3-specific: Qwen3-1.7B Q4_K_M, a
dense model of the same width, trails by about the same ratio on each device
(pp512: CPU 742.9 best warm rep against 993.53, Vulkan0 7597.2 against 9730.23,
Vulkan1 126.3 against 186.83; CPU tg128 49.6 against 56.64).

- **Vulkan0, the dense GEMM: closed by the int8 coopmat K-quant GEMM**
  (CHANGELOG; LFM3 pp512 7165.3 / 7045.8 to 12098.1 / 12075.9 t/s against
  llama.cpp's 10172.32). What is left on that device's pp512 profile
  (`INFR_PROF_OPS`, 2 reps): the head-gate GEMV (below) at 80.8 us per attention
  layer, attention (`attn_flash_warp_bm32` 275.5 us, `attn_pv_warp` 215.9 us),
  and the quant passes at 14.7 us each.
- **Vulkan1 (integrated): closed, prefill and decode now ahead of llama.cpp**
  (2026-11-15, CHANGELOG: the register-blocked `native_gemm_dp4a_kq` GEMM, the
  register-blocked `attn_nc_fa`, a four-output int8 decode GEMV at every width).
  `infr bench` three reps, interleaved with the previous build, against
  `llama-bench` in the same window: Qwen3-1.7B pp512 231.9-232.0 against 187.04,
  tg128 27.4-27.5 against 25.46; LFM3-1.7B pp512 236.7-236.8 against 195.95,
  tg128 26.7 against 25.27. What is left on the pp512 profile (one rep,
  `INFR_PROF_OPS`): Qwen3 2.20 s of device time (llama.cpp's perf logger
  2.72966e+06 us), of it the dp4a GEMMs 1.80 s, `attn_nc_fa_hd128` 254.8 ms
  (llama.cpp `FLASH_ATTN_EXT` 325380 us), `quant_q8` 84.1 ms over 448 passes;
  LFM3 2.16 s, of it the fused gate/up 971.0 ms (10.1 ms per call), `ffn_down`
  250.0 / 243.1 ms (Q6_K / Q4_K), attention 94.2 ms, the head-gate GEMV
  (`gemv_streamed:m128:2048x16`) 9.2 ms. Open on this device:
  - The dp4a GEMM reaches about 805 GFLOPS on the gate/up shape (2·128·2048·
    12288 in 8.0 ms); the device's dp4a peak was not measured. In the tile sweep
    below the larger per-thread tiles lost to register spills. Untried:
    nibble-packed Q4_K in shared memory (half the weight reads, llama.cpp's
    layout) and a wider k step with fewer staging registers.
  - Prefill chunks of 64 rows or fewer keep the `native_gemm_mmq_*` tile
    (`DP4A_KQ_MIN_ROWS`, measured on Qwen3 only: pp64 139.3 t/s dp4a against
    145.3 mmq, pp80 162.6 against 93.8). A 64-token dp4a tile was not built.
  - Decode device time (`INFR_SEAM_NO_REPLAY=1`, 32 tokens): LFM3 1.20 s, Qwen3
    1.18 s, against llama.cpp's 39666.6 and 40111.4 us for one token. The LFM3
    2048x512 k/v projections stay on the dequant GEMV (61.1 us `gemv_rm_v2`,
    Q4_K; 80.4 us plain, Q6_K) against llama.cpp's 22.672 and 30.52 us; see the
    declined item below for why. The Q6_K lm_head is behind too: 7.6 ms (LFM3)
    and 9.0 ms (Qwen3) against 7319.04 and 8460.76 us; llama.cpp runs Q6_K on
    its dequant GEMV on AMD, infr's scalar Q6_K GEMV is far slower there (19.5
    ms for the LFM3 lm_head with `kernels.vulkan.mmv_mw=false`).
  - Not measured: any other non-coopmat GPU. Intel Arc, a discrete RDNA1/RDNA2
    card or anything else without an f16 cooperative matrix now takes the dp4a
    GEMM and the new attention unmeasured; the four-output GEMV change is gated
    on `Capabilities::integrated` and reaches no discrete GPU. It does reach
    every other device that reports itself integrated: Intel iGPUs and Apple
    GPUs under MoltenVK now take the four-output GEMV at every width, measured
    on neither. Small-m prefill (rows 2-8) on an integrated GPU now takes the
    four-output GEMV at in_f >= 2048 too, unmeasured.
  - Verify batches past 8 rows take the two-output m16 GEMV build, so on an
    integrated GPU decode and such a verify no longer share a reduction at
    in_f >= 2048 (they already did not below 2048 on any device).
  - `kquant_linear_*_within_int8_bound` (`seam_op_parity`) runs through the
    adapter and so through the dp4a GEMM on this device, but asserts which GEMM
    ran only for the int8 coopmat one; a dp4a counter like `i8cm_kq_counts`
    would make it say so.
- **CPU: closed for prefill, LFM3 decode level** (2026-11-15, CHANGELOG: the
  spin pool's threads had been sharing cores, and the decode Q4_K/Q6_K dots took
  a horizontal sum per sub-block). Two interleaved rounds of four reps:
  LFM3-1.7B pp512 1226.86-1251.96 and 1069.51-1206.10 t/s against llama.cpp's
  1135.79-1193.56 and 1112.89-1159.61, tg128 59.36 and 59.62 against
  53.95-61.40; Qwen3-1.7B pp512 1078.38-1111.61 and 1026.68-1143.06 against
  1128.14-1141.88 and 1120.53-1151.13, tg128 56.83 and 56.72 against
  59.88-60.24. What is left on the CPU is in B173.

**Open, measured once:**

- **The int8 decode GEMV for the small k/v projections on an integrated GPU**
  (dropping `mmv_mw_choice`'s 2M-element floor when `Capabilities::integrated`).
  Measured on the RDNA2 iGPU, LFM3-1.7B: the 2048x512 k/v went from 64.4 / 80.0
  us (Q4_K / Q6_K, dequant GEMV) to 19.7 us on average, tg128 26.7 to 27.5 t/s.
  It was declined because it moves numerics and so flipped the text hash
  `gpu_seam_golden_qwen3` on that device (got `0xe91c03c96815974c`, text still
  coherent). That hash is retired since 2026-11-23 (plan 11's G4), so the reason
  is gone and the change is open. What it needs now: the `gpu_seam_tolerance_*`
  tests passing on the integrated GPU with the floor dropped (int8 activations
  on those GEMVs are Int8-class arithmetic, which the rows already admit; not
  run), and the speed measured again in a quiet window on more than LFM3, under
  both drivers (the one figure is from the Windows driver).

**Considered and declined, with what was measured:**

- **Tiles for the dp4a K-quant GEMM on the integrated GPU** (Qwen3-1.7B pp512,
  one rep each, before the attention change; per-call gate/up and Q6_K
  `ffn_down`): 64 tokens x 128 rows, 8x8 per thread, 128 threads: 183.1 t/s, 8.0
  / 4.9 ms; **128 x 64, 8x8, 128 threads (shipped): 187.4 t/s, 8.0 / 4.1 ms**;
  128 x 128, 8x8, 256 threads: 184.7 t/s; 128 x 128, 16x8 per thread: 94.9 t/s
  (17.7 ms gate/up, spills); 128 x 64, 8x4, 256 threads: 155.6 t/s; 128 x 64,
  16x4: 183.2 t/s; 64 x 64 with four k blocks per step: 104.3 t/s; 128 x 64 with
  one k block per step: 173.2 t/s. A required subgroup size of 32 measured
  identical (187.6 t/s): the driver already picks wave32.
- **Plain `dotPacked4x8EXT` plus an add instead of the saturating accumulate**
  in the old mmq kernel: the gate/up call went from 13.6 to 14.5 ms; the
  saturating form maps to the native instruction on this driver.
- **A larger ubatch on the integrated GPU** (pp512 is compute-bound there):
  126.5 t/s at 256 and 126.6 at 512 against 126.2 at the default 128, before
  this slice.
- **Fused QKV for LFM3 on Vulkan.** `fuse_qkv_decision` refuses it because the
  short-convolution layers have no `attn_q` and because 5 of the 11 attention
  layers carry a Q6_K `attn_v`. On Vulkan0 the separate q, k and v GEMMs cost
  152.8, 47.5 and 47.5-50.5 us plus their split-K reduces; fusing can save at
  most part of the k and v share, and a per-layer rule would fuse 6 layers. Not
  built or measured.
- **The head-gate projection (2048x16) at m=512 on Vulkan** runs as the per-row
  GEMV (`gemv_streamed`, one workgroup per output): 75.3 us per layer on Vulkan0
  against llama.cpp's 3.956 us, 1.25% of device time; 162.6 us per chunk on the
  integrated GPU. No prefill GEMM tiles n below 64, so a fix is a small-n
  kernel, or a padded weight plus a gate stride on `Op::HeadGate` on all three
  backends. Not done for that share.
- **`Op::ShortConv` on Vulkan** recomputes `B ⊙ X` per tap and runs as two
  dispatches, and is already ahead of llama.cpp: `shortconv_par` 19.7 us and
  `shortconv_shift` 1.9 us per layer on Vulkan0 against `SSM_CONV` 20.483 us
  plus a `CONCAT` and two `MUL`s; 278.9 us against `SSM_CONV` 1023.68 us on the
  integrated GPU. No change.

### B189 — what the LFM3 phase did not verify (2026-11-14)

**Tag:** tests · lfm3 · metal

- **Metal not run.** `Op::ShortConv` / `Op::HeadGate` have MSL kernels
  (`shortconv_f32`, `shortconv_par_f32`, `shortconv_shift_f32` in
  `deltanet.metal`, `head_gate_f32` in `elementwise_norms.metal`), exec arms
  with a host fallback (`host_causal_conv`), replay gating, and parity cases
  (`shortconv_parity`, `head_gate_parity` in `infr-metal/tests/parity.rs`). They
  compile under `cargo clippy -p infr-metal --target aarch64-apple-darwin`; the
  MSL itself is compiled only at backend init on a Mac, so a syntax error there
  would surface first in CI's macOS job, and take every Metal test with it.
  Phase 03 closes once that job has run them.
- **The goldens run nowhere by default.** `cpu_golden_lfm3` and every LFM3 GPU
  test are model-gated; the `cpu-goldens` CI job downloads Gemma 3 1B and Qwen3
  0.6B only. The Q4_K_M file is 1099627808 bytes. Hashes captured on Windows
  only. To run it in CI: add `LiquidAI/LFM3-1.7B-GGUF` `LFM3-1.7B-Q4_K_M.gguf`
  to that job, drop the `#[ignore]`; one hash serves every platform (plan 11).
  The synthetic tests (`synthetic_lfm3.rs`, `seam::runner`'s
  `lfm3_divergent_turn_matches_a_fresh_slot` /
  `lfm3_extending_turn_matches_a_fresh_prefill`) do run in CI.
- **Only the CPU is scored against llama.cpp in CI.**
  `synthetic_lfm3_cpu_matches_llama_cpp` holds the CPU to a committed llama.cpp
  fixture (`crates/infr-llama/tests/fixtures/synthetic_lfm3_llama_cpp.txt`,
  regenerated by the ignored `synthetic_lfm3_regenerate_llama_cpp_fixture`).
  Vulkan and Metal are compared with the CPU only, and those tests need a GPU.
  The generic oracle (`cpu_prefill_matches_llama_debug_dump`) keeps its
  model-agnostic floors, which pass with the head gate dropped (5.780733) or the
  window pattern ignored (5.19669); its LFM3 sibling
  (`cpu_prefill_matches_llama_debug_dump_lfm3`, bound 0.02) is the check on the
  real file, and runs only where a dump and the F16 file are given.
- **Session tests are text-level on the real model.** An 8-token answer does not
  move when the convolution history is stale or one token off (it reaches six
  tokens back): a no-rewind test like qwen35's passed with the zeroing removed,
  and was not added; `gpu_seam_lfm3_session_extends_past_the_window` (F16)
  catches a wrong history kernel but not a one-token re-feed. The exact-logit
  synthetic tests in `seam::runner` are the guard for the reuse rule.
- **Snapshot resume with a window ring is covered on a chaining GPU only.**
  `lfm3_chain_stopped_early_resumes_past_the_window_vulkan` and
  `lfm3_snapshot_behind_a_recycled_ring_is_not_resumed_vulkan` (`seam::runner`,
  synthetic LFM3) skip on a device whose decode chain is one step, which
  includes the integrated GPU here, and on CI, which has no GPU. The ring
  fallback after the recurrent-state arm (a `debug_assert` with a
  zero-and-refill path in release) is unreachable by construction and has no
  test. Multi-GPU (pipeline, tensor-, expert-parallel) refuses LFM3 by its
  recurrent state, as it refuses qwen35.
- **GPU parity tests pass silently without a device.** `seam_op_parity.rs`'s
  `gpu()` returns `None` when no Vulkan device opens and every case returns
  early as a pass, with nothing on stderr; `synthetic_lfm3_vulkan_matches_cpu`
  and the other `prefill_logits_vulkan` tests fail instead. A run on a GPU-less
  box therefore reports the parity cases green. Printing the skip on the
  harness's stderr (as `with_chain_vulkan` does) would at least show it; making
  an explicit `--run-ignored` run fail without a device is the stronger option.
- **Not supported:** the non-causal "d1" variant (`attention.causal = false`) is
  refused at load; `lfm2` / `lfm2moe` are separate archs and were not looked at
  beyond `lfm-base.cpp`, whose short convolution and dense FFN they share.
- **Near tie on the integrated GPU:** at Q4_K_M, the marble-boxes prompt of the
  window-extension test splits at its third token between the integrated GPU's
  fresh 1999-token prefill (" does") and every other path (" marbles": that
  device's own session, the 7900 XTX, CPU production and reference). The same
  device's F16 prefill is within 0.0053259134 of llama.cpp on the 1092-token
  prompt, so the test runs on F16.

### B192 — what int8 prefill activations cost in accuracy (2026-11-15)

**Tag:** vulkan · precision · **Needs:** a decision on the default (the numbers
for it are here)

The int8 coopmat K-quant prefill GEMM (CHANGELOG) quantizes activations to int8
per 32-element block, as llama.cpp's Vulkan backend does on the same card, where
the f16 coopmat GEMM rounded them to f16. Measured on the RX 7900 XTX with
`int8_prefill_accuracy` (`crates/infr-llama/tests/`, ignored, a measurement): 6
windows of 512 tokens of wikitext-2 `wiki.test.raw`, the second half of each
scored (1530 positions), every backend against infr's CPU reference
(`kernels.cpu.reference`, f32 activations), on the exact tokens
`llama-perplexity -c 512 --chunks 6` used (llama.cpp `5676dfbe0`):

| Qwen3-1.7B Q4_K_M | KLD mean | KLD max |  top-1 |     PPL | cos mean |  cos min | max abs Δlogit |
| ----------------- | -------: | ------: | -----: | ------: | -------: | -------: | -------------: |
| infr CPU ref      |        0 |       0 |      1 | 16.3087 |        1 |        1 |              0 |
| infr int8 off     | 0.000003 | 0.00013 | 0.9987 | 16.3126 | 0.999999 | 0.999752 |         0.4452 |
| infr int8 on      | 0.001722 | 0.09546 | 0.9765 | 16.3407 | 0.999644 | 0.955304 |         6.5250 |
| llama.cpp Vulkan  | 0.001792 | 0.26817 | 0.9824 | 16.2741 |          |          |                |
| llama.cpp CPU     | 0.008603 | 0.28568 | 0.9529 | 16.2613 |          |          |                |
| _as dumped:_      |          |         |        |         |          |          |                |
| infr CPU ref      | -0.00024 | 0.00004 |      1 |         |          |          |                |
| infr int8 off     | -0.00024 | 0.00011 | 0.9987 |         |          |          |                |
| infr int8 on      | 0.001479 | 0.09518 | 0.9765 |         |          |          |                |

| SmolLM3-3B Q4_K_M | KLD mean | KLD max |  top-1 |    PPL | cos mean |  cos min | max abs Δlogit |
| ----------------- | -------: | ------: | -----: | -----: | -------: | -------: | -------------: |
| infr CPU ref      |        0 |       0 |      1 | 8.2465 |        1 |        1 |              0 |
| infr int8 off     | 0.000002 | 0.00008 | 0.9987 | 8.2475 | 1.000000 | 0.999988 |         0.1723 |
| infr int8 on      | 0.000730 | 0.01062 | 0.9863 | 8.2575 | 0.999871 | 0.993740 |         1.7175 |
| llama.cpp Vulkan  | 0.000328 | 0.01412 | 0.9863 | 8.2638 |          |          |                |
| llama.cpp CPU     | 0.002466 | 0.04499 | 0.9739 | 8.2610 |          |          |                |
| _as dumped:_      |          |         |        |        |          |          |                |
| infr CPU ref      | -0.00039 | 0.00001 |      1 |        |          |          |                |
| infr int8 off     | -0.00039 | 0.00001 | 0.9987 |        |          |          |                |
| infr int8 on      | 0.000335 | 0.01035 | 0.9863 |        |          |          |                |

llama.cpp's rows come from its `--kl-divergence-base` dumps, which keep each
row's log-probabilities only down to 16 nats below the top, so they carry no
logit cosine, and their PPL is llama.cpp's own `Final estimate` (a perplexity
read back from the clamped dump is biased low); its CPU backend quantizes
activations too (`q8_K`), hence its larger KLD.

**Scored here as the candidate against infr's reference, a dump's clamp biases
KLD low too, so llama.cpp's KLD only compares with the "as dumped" rows.** Every
token below the floor reads back at the floor, above its true log-probability
(the dumped row no longer sums to one), and the reference's mass down there
enters the sum with a negative sign. This is a property of how this test scores
a dump, not of llama.cpp's own `--kl-divergence` run, where the dump is the base
and terms below the floor are dropped. The test now also scores infr's logits
put through the dump's own encoding (`as_dumped`, the encoding
`perplexity.cpp`'s `log_softmax` writes): the reference against its own dump is
the bias alone, -0.00024 on Qwen3-1.7B and -0.00039 on SmolLM3-3B, which has
more tail mass below the floor. On that common footing infr's int8 prefill is
closer to the f32 reference than llama.cpp's Vulkan backend on Qwen3-1.7B (KLD
0.001479 against 0.001792) and level with it on SmolLM3-3B (0.000335 against
0.000328, and llama.cpp's row also carries its f16 KV cache and flash
attention). The "SmolLM3's KLD is twice llama.cpp's" reading of the raw rows
(0.00073 against 0.00033) was this bias, not the kernel. Per a reading of
llama.cpp `5676dfbe0`, not a measurement, the two kernels do the same thing: it
quantizes activations for every Q4_K and Q6_K matmul in both files, lm_head
included (`ggml_vk_mul_mat_q_f16`: `can_quantize_y` with no size or type
exception beyond `GGML_PREC_*` source precision, which these files do not set);
`quantize_q8_1.comp` is infr's `quant_q8` (32-element blocks, d = amax/127
rounded to f16 after quantizing with the f32 d, round to nearest, f16 Σd·q);
`mul_mmq_cm1.comp` scales each block's int32 dot in f32 with the same d·sc /
dmin·m terms and f32 accumulators. Nothing was changed in the kernel. The f16 KV
cache and flash attention share of llama.cpp's row was not measured. Top-1 is
unaffected by the clamp (the top token is never clamped), and there infr trails
llama.cpp on Qwen3-1.7B, 0.9765 against 0.9824, and ties on SmolLM3-3B; not
looked into, and small next to the KLD ordering.

Against the f16 path, top-1 agreement with the reference falls from 0.9987 to
0.9765 / 0.9863 and perplexity rises by 0.17% / 0.12% (16.3126 to 16.3407,
8.2475 to 8.2575).

**Greedy continuations** (64 tokens after the first 256 tokens of each window,
the same decode GEMV either way; leading tokens shared, re-tokenized from text):
Qwen3-1.7B on/off 64, 12, 24, 10, 33, 33 (off against the CPU reference 20, 22,
56, 64, 45, 52); SmolLM3-3B on/off 8, 29, 64, 64, 13, 46 (off against the
reference 8, 7, 64, 25, 64, 50). Continuations split at near ties within a few
dozen tokens whichever prefill runs, the f16 path included; neither prefill
tracks the reference's greedy text better on these prompts.

**Tests that score it:** `gpu_seam_int8_prefill_tracks_cpu` (Qwen3-0.6B, f16 KV,
default path) holds the shipped prefill's logit cosine against the CPU reference
to 0.995 (measured 0.9984) and fails if the int8 GEMM did not run;
`gpu_seam_kv_formats_track_cpu` pins `kernels.vulkan.i8cm_kq = false` so its
floors score the cache format alone (with the int8 GEMM its turbo2 case measured
0.9213 against a 0.93 floor). `seam_op_parity`'s
`kquant_linear_*_within_int8_bound` hold each K-quant Linear to a worst-case
bound of the activation quantization (worst observed 0.087-0.205 of it), and
`i8cm_kq_matches_its_decomposition` holds the kernel to its own decomposition.

### B191 — MTP sessions are sized outside the context fit (2026-11-15)

**Tag:** correctness · vram fit · qwen35 · mtp

The first reading of this entry was that the MTP head's own K and V cache
(`MtpHeadSession::build` in `infr-llama/src/mtp/mod.rs`) should be added to the
context fit (`kv_bytes_estimate_fmt` in `seam/mod.rs`). Checked by reading, that
fix would price nothing real, so it was not made:

- MTP is parked: `mtp_enabled` returns `false`, so `should_use_mtp` never
  returns `true` and no chat path builds a head. The `infr bench` `mtp` arm and
  `infr compare`'s `mtp128` column check `mtp_enabled` too.
- With MTP on, the fit is not used for it. `generate_mtp_spec_*_timed` sizes
  both the head's cache and a fresh trunk session (`run_verify` into
  `generate_dense_backend`, which takes `max_ctx` as given) to
  `prompt tokens + max_new + DEFAULT_N_MAX + 8`, per call. `DenseSeamChat`'s MTP
  branch of `generate` never opens the fitted persistent session, and neither
  does its warmup, which goes through `generate`.
- When the head's allocation is refused, `build` returns the error and the
  request fails; there is no fallback to plain decode.

What is real, for whoever unparks MTP: `DenseSeamChat::generate_constrained`
always opens the fitted persistent session, which the default context sizes to
fill VRAM, and keeps it. A later MTP `generate` then allocates the trunk session
and the head's cache on a second `VulkanBackend` (`mtp_vk`) beside it, so a
server that answered one forced tool call first can have every MTP request
refused. The fix belongs to the MTP session lifetime (share the persistent
session, or drop it before an MTP turn), not to the fit. The head also uploads
its own copy of the embedding table (`build_embed_chain_buf`) on GPU backends
that run the fused draft chain, which nothing counts either.

The trunk estimate also counts one snapshot copy that a device with a one-step
decode chain (CPU, Metal) never allocates, which is a small over-count on the
safe side.

#### Also from the same review, not fixed

- infr refuses LFM3 files llama.cpp accepts, none of which exists yet: attention
  layers with different nonzero `attention.head_count_kv`, a scalar
  `head_count_kv` (llama.cpp reads it with `get_key_or_arr`), layers that
  disagree on carrying sandwich norms, and an `attention.sliding_window_pattern`
  shorter than the layer count. Each is a refusal at load, not a wrong result.
- The widening of qwen35's `ssm_conv1d` kernel to f32 at load has no qwen35
  test; the LFM3 one (`synthetic_lfm3_vulkan_reads_an_f16_conv_kernel`) covers
  the shared code.

### B190 — `seam/runner.rs` keeps growing with no seam (2026-11-14)

**Tag:** maintainability · seam

`crates/infr-llama/src/seam/runner.rs` is one file holding the whole
`generate_dense_backend` (one function spanning most of the file) and its test
module. Every new arch adds a mixer arm to the graph build and a rule to the
session start, so each port makes the function longer and every review of it
reads all of it. The seams it already has, each nameable and testable alone:

- **Session start** — the `start` computation: prefix reuse, the recurrent-state
  rules (extend, resume from the snapshot, zero), `ring_keeps` and the
  fallbacks. Pure over `cached`, `written_end`, the snapshot length and the ring
  geometry, so it could be a function with unit tests rather than a block
  reached only through whole generations.
- **Mixer emits** — the `MixerW::Attn` / `DeltaNet` / `ShortConv` / `Mla` /
  `Dsv4` arms of the per-layer build, one module each.
- **Weight loading** — the `wload` sequence and its `wpush` mirror, which must
  stay in the same order (one module would put the two side by side).
- **The decode loop** — chain, per-token and replay paths.
- **The tests** — `mod tests` as `seam/runner/tests.rs`.

Not started: a move-only split first, then behaviour changes in their own
commits.

### B193 — what float dense streaming left (2026-11-15)

**Tag:** streaming · plan 05 · **Blocked on:** nothing; each is its own slice

F16 and F32 dense projections stream on Vulkan since 2026-11-15: the planner
admits `infr_vulkan::linear::dense_stream_supported` dtypes, and a streamed
float `Op::Linear` runs `float_linear` (adapter) — the resident float dispatch —
at the arena slot's address. What that slice did not do:

- **A streamed f16 Linear+Add pair is unfused.** `execute_static` removes every
  streamed weight's `Linear→Add` fusion because the native fused kernels bake a
  zero offset, so a streamed f16 decode residual runs `linear_f16` plus a
  standalone Add where the resident one runs `linear_res`. Same arithmetic (the
  fused kernel adds the residual to the same reduction), one more dispatch per
  pair. `Recorder::linear_add_at` already takes an address, so keeping the f16
  pairs fused is a filter change plus a test. Not measured.
- **The resident footprint still misprices two load-time widenings.**
  `weights::tensor_resident_bytes` now counts every dtype at its file bytes
  except `I2S`, but `wload` also widens an f16/bf16 depthwise conv kernel
  (`ssm_conv1d`, `shortconv.conv`) and deepseek2's quantised `attn_k_b`/
  `attn_v_b` to f32. Small tensors; the estimate is low by their difference.
- **The streamed path costs throughput even when every block is resident.**
  Measured and split in `docs/plan/08-streamed-at-resident-speed.md` (step 0).
  Steps 1 and 2 there landed on 2026-11-15: layer-major prefill now runs only
  for a multi-chunk prompt on a cache that cannot hold every block, and the
  arena barriers only around a staging copy. ERNIE-4.5-0.3B F16 on the RX 7900
  XTX at `INFR_CACHE=1G` now prefills at the resident rate (pp512 22614 / 22707
  against 22763 / 22134 resident) and decodes at 385.3 / 383.4 t/s against 449.8
  / 449.6. Step 4 landed the same day: a streamed quantised Linear now takes the
  resident routing (`adapter::quant_linear`) at every row count, so Qwen3-1.7B
  Q4_K_M at `INFR_CACHE=2G` prefills at the resident rate (pp512 12122 / 12037
  against 11841 / 11798 resident) and decodes at 387 t/s (was 308; resident
  479), and on the integrated GPU (`Vulkan1`) its streamed decode went from 9.3
  to 26.9 t/s against 27.5 resident — that was the old streamed GEMV route, the
  suspect named here before. Step 3 landed the same day: a session whose cache
  holds every block now decodes through the record-once replay and the chain,
  re-recording when the dense pager's placement epoch moves, so streamed decode
  holding every block reads 469.4 / 469.1 t/s for Qwen3 (resident 479.0 / 478.8)
  and 444.3 / 444.2 for ERNIE (resident 449.7 / 449.9). What is left is
  Linear+Add fusion for streamed weights (step 5), the known difference.
- **Replayed streamed decode steps count no pager hits.** A replay makes no
  `DensePagerSession::stage` call, so `paging.stats` and `dense_pager_totals`
  cover static forwards and the staging pass only: a session holding every block
  shows its prefill's hits and nothing for its decode. Documented on
  `print_stats_if_enabled`; crediting hits per replay would need a counter-only
  entry point on `infr_core::pager::Pager`, not built.
- **Streamed quantised Linears skip four opt-in tiers.** `quant_linear` runs a
  streamed slot past the `_cm8` warptile (`INFR_CM_8X8`), the fp8 coopmat GEMM
  (`INFR_F8_COOPMAT`), the int8 Q8_0 coopmat GEMM (`INFR_I8_COOPMAT`) and the
  bf16 coopmat GEMM (`INFR_BF16_COOPMAT`), because their dispatches bind the
  resident weight buffer, and takes the next tier down instead. With one of
  those switches on, a streamed Linear therefore runs a different-precision
  kernel from its resident twin and the outputs are not bit-identical; on the
  default configuration no route differs. What enabling each would take: the
  `_cm8` and BF16CM warptiles already compile under `-DSTREAMED`
  (`native_gemm_warp`'s arena reads), and `build.rs` drops their twins only
  because nothing calls them (`dead_streamed_twin`) — emit the twins and add
  `_at` recorder methods. The int8 Q8_0 GEMM has an arena-addressed build and
  recorder method (`Recorder::matmul_i8cm_q8_0_at`), compiled only for its
  parity test (`parity` feature); its row-scale variant's `_streamed` SPV is
  built but has no `_at` method. The fp8 GEMM has no streamed build at all
  (`native_gemm_f8cm_q8_0.comp` reads a bound weight; only the prepack's
  `repack_q8_to_f8` has a `-DSTREAMED` twin), so it needs a shader change. None
  is measured or tested on a streamed weight.
- **SmolLM3-3B decodes faster streamed than resident** (RX 7900 XTX, tg128:
  251.3 t/s at `INFR_CACHE=2G` against 228.3 resident, 2026-11-15). Linear
  routing is identical at m = 1, and neither run replays (SmolLM3's NoPE layers
  keep it off the record-once replay, B157; resident with
  `INFR_SEAM_NO_REPLAY=1` is 228.4). The difference is early submit
  (`paging.early_submit`), which only a paged forward gets: the streamed run
  with `--set paging.early_submit=0` drops to 223.0, and resident with
  `INFR_NO_FUSE_ADD=1` (the Linear+Add fusion the streamed run lacks) is 224.8,
  so streamed without early submit sits just below resident for the missing
  fusion. Resident static decode records the whole forward before submitting,
  and the streamed one overlaps recording with the GPU. One `INFR_PROF_OPS=1`
  tg64 of each shows the same kernels except the fused `_res` GEMVs against
  plain GEMVs plus `add` (resident 40128 dispatches, streamed 44736), but
  profiling itself serializes the forward (resident with and without replay give
  the same 40128-dispatch report, 255.8 and 256.9 ms device total, so it cannot
  show the overlap) and the streamed profile's per-dispatch times came out about
  twice the resident ones (attention 43.8 against 24.1 us), which reads as a GPU
  clock difference under the profiler, not a kernel one. Suggests: early submit
  for resident static forwards (a non-replaying model such as SmolLM3) could be
  worth about 10% decode there. Not tried.
- **The paged MoE path still records arena barriers on a hit.** Plan 08 step 2
  made the dense pager record them only around a copy;
  `adapter::execute_paged_moe` still records one before and one after every
  paged `MoeFfn`'s staging, copy or not, so a wholly resident expert layer pays
  two full barriers per forward. Not measured; applying the same rule there is a
  possible later step.
- **The arena-barrier change is checked by output, not by a validator.** The
  streamed parity suites passed 5 times on `Vulkan0` and 3 on `Vulkan1` after
  barriers went to copies only. Synchronization validation was tried
  (`VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation` with
  `VK_LAYER_VALIDATE_SYNC=1`, SDK 1.4.357) and logged no hazard, but it logged
  none either with `INFR_NOBARRIER=1`, so it was not shown to be running and
  proves nothing; it also cannot see the arena's pointer reads at all. A working
  syncval recipe, checked against `INFR_NOBARRIER`, is still to find. The same
  holds for the streamed decode replay (plan 08 step 3): its ordering argument
  is written on `DensePagerSession::stage` and checked only by token identity,
  the streamed parity suites passing 5 times on `Vulkan0` and 3 on `Vulkan1`
  with no flake.
- **Streamed replay under `serve --parallel` rests on the step baton.** The
  replay's ordering assumes executes on one backend are serialised: two
  unserialised threads could have a replay read a block whose copy another
  thread recorded but has not submitted (the dense session's mutex does not
  prevent that; see `DensePagerSession::stage`). Every execute reachable from
  `ParallelSeam::generate` (`generate_dense_vulkan_session` into
  `generate_dense_backend_bound`) was traced to a `RequestCtx::gate_pass` scope
  on 2026-11-15: placement and the cold init, the prefix-seed KV copy, the
  recurrent-state restore and snapshot, the decode IO allocations, each prefill
  chunk or layer-major span, the record-once plan's build and its drop, and
  every decode-loop iteration (chained chunk, replayed step, per-token rebuild,
  and the logits, hidden-state and sampled-id readbacks). The verify and denoise
  branches are not reachable from it. The engine's warmup decodes a token before
  any request, which stages every block, so under `serve --parallel` a cache
  that holds every block is full before two sequences run and its epoch then
  moves only on a taken-back or rolled-back placement.
  `synthetic_streamed_parallel_sequences_rebuild_on_an_epoch_move` runs two
  sequences on one engine with a forced move between them. A new execute site
  added without the baton would break this silently; nothing checks it
  mechanically.
- **The placement epoch is global to the session.** Any placement moves it,
  including one of a block no current recording reads, which then costs every
  live recording a needless re-record (one per sequence). Holding every block,
  that happens only while blocks are first placed, so it was left; a per-block
  check (compare the recording's own slots) would remove it.
- **A session whose cache holds every block still reserves the layer-major
  activation.** `dense_stream_budget_at` holds `layer_major_act_bytes` back from
  the arenas for every streamed session, because the arenas are sized before it
  is known whether they will hold every block. A session that ends up holding
  them never runs layer-major, so that reserve (`ctx * n_embd * 4` bytes) is
  unused; at a borderline budget it can be what keeps the cache from holding
  every block. Safe direction, not measured.
- **Not verified:** a slot placed past 4 GiB of an arena (the base is a 64-bit
  host-computed address and the float kernels never add the slot position in
  u32, read in `DensePagerSession::stage`, but no test allocates such an arena);
  an f16 or f32 model larger than VRAM under AUTO placement (every run here
  forced the cap with `paging.cache`; the f32 pricing fix that placement depends
  on is unit-tested only); an f16 `w_off` (refused, as it is resident); the
  Metal backend (no pager, unchanged).

### B194 — the Qwen3-30B-A3B tests take host RAM low on the discrete GPU too (2026-11-15)

**Tag:** test suite · host memory · **Blocked on:** nothing

The integrated-GPU half is done: `cpu_backend`'s `resident_room` /
`need_resident_room!` skip a test that would hold a multi-GB model on a
unified-memory device unless `UNIFIED_RESIDENT_FACTOR` times the GGUF size plus
`UNIFIED_HOST_MARGIN` is available (`docs/windows.md` has the rule).

**The discrete-GPU cause was measured and fixed (2026-11-16).** A resident load
on `Vulkan0` kept every mapped page the upload read in the process working set
for as long as the model was loaded; on Windows those count as in use, not as
standby. Measured with `typeperf` on an `infr serve` process (load, an 800-token
prompt with 600 generated tokens, 8 s idle): working set minus its private part
was 1.01 to 1.04 times the file (LFM3-1.7B F16 3.47 GB: 3351 MiB; SmolLM3-3B
Q4_K_M 1.92 GB: 1886 MiB; Qwen3-1.7B Q4_K_M 1.11 GB: 1100 MiB), the private
working set only 0.22 to 0.26 GB, and it held through prefill, decode and idle.
`paging.release_uploaded` (default on) now releases each resident weight's
mapped pages after its upload (`TensorBytes::release_resident`,
`infr_plat::mem::release_mapped`), and the same runs keep about 50 MiB mapped:
available memory falls 0.33 GB instead of 3.6 GB for the F16 file. Re-run on
`Vulkan0` at `9bc928cb` (2026-11-16): the filter
`(test(gpu_seam) | test(paged)) & (test(moe) | test(ernie))` with
`--run-ignored all --test-threads 1` passed 15 of 15, the Qwen3-30B-A3B and
ERNIE-4.5-21B-A3B tests among them, and available memory sampled every 2 s over
the whole run never fell below 24821 MiB. Only the minimum over the run was
recorded, not a figure per test.

What remains:

- **Commit charge is still about the model's size.** `Private Bytes` of the same
  serve processes was 1.9 GB (Qwen3-1.7B), 2.7 GB (SmolLM3) and 4.1 GB (LFM3
  F16) with a private working set of about 0.24 GB in each, with or without the
  release, so it is committed but not resident: most likely the driver's commit
  for the device-local weight allocations. It does not lower available memory,
  but it counts against the commit limit, which is what B75's `os error 1455`
  (paging file too small) runs out of. Not investigated further.
- **Not released:** weights the loader rewrites (`WBytes::Owned`: the qwen2 q/k
  permute, the conv and MLA dequants, BitNet) leave their source pages resident;
  the multi-GPU binders (pipeline, tensor and expert parallel) and MTP's weight
  set do not release; a unified-memory device does not either, where the mapped
  pages are part of `UNIFIED_RESIDENT_FACTOR` and releasing them was not
  measured. macOS and the BSDs answer `Unsupported` (macOS's `MADV_DONTNEED` is
  a priority hint whose effect nothing here can observe).
- **Linux is only CI-verified** (`infr-plat`'s and `infr-gguf`'s unit tests:
  `madvise(MADV_DONTNEED)` and `/proc/self/pagemap`); no model has been run
  there with the release. Linux's `MemAvailable` already counts part of the page
  cache, mapped pages included, so the gain there may be smaller than on
  Windows; not measured.

- **Left from the review of the release (2026-11-16), none a wrong result:**
  `vulkan_moe_binder` builds a `PageReleaser`, and so spawns and joins a thread,
  on every call, including every Diffusion-Gemma denoise step and prefill, where
  it releases nothing (start the thread on the first release instead);
  `release_mapped_pages` warns once per weight on an error, which can be
  hundreds of identical lines per load (warn once per binder); every fresh
  binder releases again, so a CPU resident on the same model re-faults the file
  after each one-shot GPU call, and a re-upload after a state reset reads from
  standby or disk; a tied `token_embd.weight` read by the host gather is
  released and then faulted back whole once.

### B195 — sampling and constrained decoding: what a first full read found (2026-11-15)

**Tag:** sampling · serve · parity · perf · **Evidence:** read from the code and
compared with llama.cpp `5676dfbe0` (`common/common.h`, `src/llama-sampler.cpp`,
`tools/server/server-context.cpp`); nothing was built or run. Earlier passes had
cleared only seed handling, penalty indexing, `StopMatcher` and the GPU sampler
shader bodies.

- **`SpecMetalChat` drops the request's `RequestCtx`.**
  `SpecMetalChat::generate` (`chat/metal.rs`) takes `_req` and
  `SeamModel::generate_metal_spec` has no parameter for it and polls no latch.
  Under `infr serve` on macOS with `INFR_SPEC_DRAFT` set, a stop sequence, a
  deadline or a client hang-up does not stop decoding (the text is still cut by
  `StopMatcher`, the compute runs to `max_tokens` holding the model), and a
  request's `temperature`, `seed` and penalties are ignored without a word. B151
  records the same for the parked MTP path only; this one is live. Fix: thread
  `req` through, poll `abort_requested(req)` per draft/verify round, and reject
  or log a request with `temperature > 0`. Needs macOS to run.
- **Temperature is applied before top-p; llama.cpp applies it after.**
  `truncated_softmax` (`sampling.rs`) cuts the nucleus on tempered
  probabilities, and `infr-cpu`'s `sample_token`, `sample_topk.comp` and the
  Metal sampler copy that order. llama.cpp's default chain ends with temperature
  and its top-p takes the softmax of the raw logits. Logits `[2, 0, 0]` at temp
  0.6, top_p 0.9: llama.cpp keeps all three tokens (raw 0.787 / 0.107 / 0.107)
  and draws token 0 at 0.933; infr's tempered 0.933 already passes the cut, so
  it always draws token 0. Below temp 1 infr's nucleus is systematically
  smaller. infr's order is the one HF transformers uses, and `arch_sampling`
  defaults come from HF `generation_config`; nothing says which was intended.
  **Needs a decision:** keep the HF order and document the difference, or
  switch, which changes four implementations together and re-pins the GPU parity
  and `reference_full_sort` tests.
- **Penalties differ from llama.cpp.** `Penalties::resolve` starts empty and
  `observe` sees generated tokens only, where llama.cpp's server feeds every
  prompt token to the sampler first; presence and frequency count over the whole
  generation, where llama.cpp windows one count to `penalty_last_n` for all
  three. `RequestSampling::default`'s doc says a request setting only
  `repeat_penalty` "gets llama.cpp's behavior", which is false, and
  `repeat_last_n` is not a request field. **Needs a decision:** match llama.cpp,
  or keep the OpenAI reading for presence/frequency and seed only the repeat
  window from the prompt. Fix the doc either way.
- **What is left of `logit_bias`, `min_p`, `logprobs` and `n`.** `infr serve`
  applies `logit_bias` (in `Penalties::apply`, ahead of the penalties) and
  `min_p` (in `truncated_softmax`) on the dense seam decode loop, returns
  OpenAI log probabilities (the chosen token plus `top_logprobs` alternatives
  from the full temperature-scaled softmax, streaming and non-streaming, on chat
  and completions), and answers `n > 1` with a 400. Still open: DiffusionGemma's denoise sampler and `SpecMetalChat`
  apply neither `logit_bias` nor `min_p` and refuse a request carrying them
  (`refuse_seam_only_fields`, a 400) — applying them there is separate work, and
  both still ignore the penalties without a word; the parked MTP paths read only
  the process sampler; `infr run` has no `--min-p`, which needs a
  `sampling.min_p` config key, its env and manifest entries and the flag, and
  the GGUF's `general.sampling.min_p` is still ignored (see
  `unapplied_sampling_keys`). Logprob gaps: a grammar-forced tool-call span has
  no entries, streaming attaches a piece's entries to the first delta that
  splitter emits for it (a `</think>` boundary can merge reasoning and content
  tokens into one chunk), and a byte-fallback token that only completes in
  context decodes to a replacement character in `token`/`bytes`. Returning
  several choices is still open. `sample_from_dist` (MTP's draw over
  `truncated_dist`, id order when `top_p` is 1) keeps the zero-draw shape
  `sample_logits` was fixed for: a uniform of exactly 0 takes the lowest id with
  any mass. `top_k` is still typed `Option<i64>`, so `"top_k": 40.0` or a
  string is a body-parse 400 that names no parameter, where `n` and
  `top_logprobs` now name theirs. The `SpecMetalChat` refusal compiles only on
  macOS and has not been built or run. B137 records what JSON mode
  (`response_format`) left.
- **MTP's `residual_dist` is quadratic in the support.** `dist_prob` is a linear
  search and `residual_dist` (`seam/model.rs`) calls it per entry, so two
  distributions spanning the vocabulary cost about the square of its size per
  call. A `top_p` of 1 with no top-k always returns the whole support (the old
  sorted path usually did too). Only matters once MTP is un-parked with sampling
  on; index one side by token id.
- **Question, needs a run:** `constrained_step` (`grammar.rs`) resamples the
  whole row, with a new vocabulary-sized vector, each time `try_accept` rejects
  a candidate. The module doc implies the mask and `consume` agree, so
  rejections may never happen; count them on a forced tool call before changing
  anything.

Checked and found right: empty and all-`-inf` rows, a `-inf` token is never
drawn while a finite one exists, the top-p `>=` comparison, request parameter
ranges, and no sampler plan is cached across requests. Not read: MTP and
Diffusion-Gemma sampling, `moe_sample.comp` and the argmax shaders, the
recorder's sampling dispatch, CLI flag validation, and the tokenizer beyond the
streaming detokenizer.

### B196 — pull requests compile nothing for x86_64 macOS (2026-11-16)

**Tag:** ci coverage · **Blocked on:** a decision on runner cost

`dist` packages five targets but skips pull requests, and the table in
`docs/ci-matrix.md` (§ Release targets on pull requests) lists what compiles
each one there instead. `x86_64-apple-darwin` is the one with nothing: the
`macos-15` runners are Apple Silicon, so `clippy`, `test` and `test-macos` all
build `aarch64-apple-darwin`, and `metal-check` cross-lints that same triple
from Linux. A break confined to x86_64 macOS (an x86_64 SIMD arm gated on
`target_os = "macos"`, a `MACOSX_DEPLOYMENT_TARGET = 10.15` API floor) is found
by the packaging leg on `main` after merge, as issue #93 was on aarch64 Linux
before `linux-arm-check`.

Options, none tried:

- **`cargo check --target x86_64-apple-darwin` on a `macos-15` runner.** Xcode's
  clang builds both arches, so the C/C++ build scripts (`ring`, `esaxx-rs`,
  `onig_sys`) should work with no extra toolchain; this is a guess, not run.
  Costs a macOS runner, the highest billing multiplier, for one check step.
- **Add `x86_64-apple-darwin` to `metal-check`'s Linux cross-lint.** Cheap, and
  the nine crates it covers include `infr-cpu`, which holds most of the
  `target_arch` code, but it leaves out `infr-llama`, `infr-hub`, `infr-chat`,
  `infr-engine`, `infr-server` and `infr-cli`, whose C/C++ dependencies need an
  Apple SDK.
- **Leave it.** x86_64 macOS shares `c_char` signedness with x86_64 Linux, which
  `clippy` and `test` cover, and no `cfg` in `crates/` combines
  `target_os = "macos"` with `x86_64` today, so the class of break this misses
  is narrower than the aarch64 Linux one was.

Separately, `linux-arm-check` runs `check`, not `clippy`. Nothing is lost by
that today: `crates/` has no `target_arch = "aarch64"` cfg, so aarch64 Linux
compiles the same code paths that `clippy` lints on x86_64 Linux and aarch64
macOS. The first aarch64-only arm would be linted nowhere; switching the job to
`cargo-zigbuild clippy` would then cover it at about the same cost.

### B197 — Actions cache pressure: pull-request saves and per-run Zig entries (2026-11-16)

**Tag:** ci cost · **Blocked on:** a decision; nothing is failing

The repo's Actions cache was observed at about 10.43 GB in 35 entries, all on
`main`, against GitHub's default 10 GB limit, so new entries evict old ones.
`linux-arm-check` saves its Rust cache only on `main` (`save-if`) and runs
setup-zig with `use-cache: false`. The rest of `ci.yml` was left as it was:

- **Every other `Swatinem/rust-cache` step saves on pull requests too.** A pull
  request's entry is readable only by that pull request, so each one can evict a
  `main` entry that every later run would have used. The same
  `save-if: ${{ github.ref == 'refs/heads/main' }}` would stop that, at the cost
  of a cold second run within a long-lived pull request.
- **`dist`'s setup-zig saves a new Zig cache entry on every run,** about 21 MB
  each: the pinned action (v2.2.2) keys it on the run id and attempt. Turning it
  off with `use-cache: false` is safe for what a release contains, since the
  cache only speeds up zig's own builds, but would cost link time on both Linux
  legs, unmeasured. Left alone because it touches the release job.

### B198 — `unquantised_numel_is_never_rejected` failed once, cause unknown (2026-11-17)

**Tag:** tests · infr-gguf · **Kind:** observed once, not reproduced

`infr-gguf`'s `tests::unquantised_numel_is_never_rejected` failed once in a full
workspace `nextest` run (585 of 586 passed before the run stopped) while other
builds and GPU tests were running on the same machine from other worktrees. Its
failure message was not captured. It passed alone, in the full rerun, and in 40
repeats of the whole `infr-gguf` test binary at 8 threads afterwards. The test
writes a fixture to a uniquely named `tempfile::NamedTempFile` and opens it with
`Gguf::open`, so two runs colliding on one path is ruled out. Unchecked guesses:
something else on Windows holding the new file briefly (antivirus, the indexer)
when it is opened or mapped, or memory pressure during the mapping. If it shows
again, capture the message first.

### B199 — llama.cpp divides by zero on a `deepseek4` file with no ratio-4 layer: reported (2026-11-17)

**Tag:** reference · **Kind:** upstream bug, filed

Filed as ggml-org/llama.cpp issue 32424. In the `llama_kv_cache` constructor the
DeepSeek branch sets `attn_rot_k` without the `n_embd_head_k_all > 0` check the
general condition has; for a model with no ratio-4 layer the indexer cache has
no layers and a head width of 0, the
`while (n_embd_head_k_all % (2*n_rot_k) == 0)` loop doubles `n_rot_k` until it
wraps to 0, and the next test divides by zero. The synthetic oracle files keep
their null tail layer (B-DSV4-ORACLE) until a llama.cpp with the fix is the
pinned reference; then that layer can go.

### B200 — what the CPU `Op::Linear` row-count identity slice left (2026-11-17)

**Tag:** tests · cpu · **Kind:** coverage gaps and recorded behaviour

`tests/linear_batch_identity.rs` holds every weight dtype's prefill `Op::Linear`
(F32, F16, BF16 and every `WEIGHT_QUANTS` format) to its one-row bits and to an
f64 reference under a bound derived from the int8 activation quantizer;
`kernels::batch_tier_tests` does the same for each x86_64 tier pairing. Q5_0 was
the one dtype that differed until its one row moved onto the int8 kernel (B201).
What it left:

- **Paths no switch reaches.** The kernel tier is chosen by runtime feature
  detection; `Op::Linear` has no config knob for it, so the op-level test runs
  only the host's own tiers (the kernel test covers the other x86_64 pairings).
  There are no NEON kernels: aarch64 runs the scalar ones, which macOS CI runs
  through both tests. `kernels.cpu.repack_mb` does not reach `Op::Linear` (its
  packed GEMM packs each call into scratch; the repack cache serves `Op::MoeFfn`
  only), so it is not a route to force here. `Op::Linear` has no bias; a bias is
  a separate op and was not part of this check.
- **What the int8 activation quantizer does at the edges** (`quantize_q8`,
  `quantize_q8_32`), the same on every route, so not a row-count difference: a
  block whose largest magnitude is so small that `127 / max` overflows (around
  3.7e-37 and below) gives every nonzero value the full code, an error of up to
  that maximum where the bound elsewhere is half a step; a NaN is dropped to
  code 0 (the f32 dtypes propagate it); an infinity sets the block scale to
  infinity and its codes to 0, so the block contributes NaN. Neither NaN nor
  infinity has a defined result there, so the identity tests do not feed them.
- **Not reviewed:** `Op::MoeFfn`'s row-count behaviour beyond what the V4
  fixtures pin (its int8 switch is documented in `docs/deepseek.md`), and the
  paged-weights route into `Op::Linear`, which reads the same bytes through
  another path.

### B201 — Q5_0 `Op::Linear` at one row: int8 activations chosen over f32 (decided 2026-11-17)

**Tag:** cpu · precision · **Kind:** decision made, alternative declined

**Was.** A one-row `Op::Linear` on a Q5_0 weight dequantized the row and took an
f32 dot, while several rows quantized the activations to int8 per 32-element
block and ran `vec_dot_q5_0_32_batch`, so Q5_0 was the one weight dtype whose
prefill and decode were not bit-identical (2.6e-3 of the output's `Σ|w·x|` on
ordinary rows of `tests/linear_batch_identity.rs`).

**Decision (owner, 2026-11-17): int8 activations at one row**, for consistency
with every other quantized dtype and decode speed, over the small distance to
llama.cpp that f32 keeps. Landed the same day: one row runs the batch kernel on
its own int8 activation, and the identity test checks Q5_0 bit for bit like
every other dtype. The options as weighed:

- **int8 at one row (taken).** Prefill and decode bit-identical, as for every
  other dtype; one-row `Op::Linear` 2.2-8.8x faster on 4096×4096, 4096×14336 and
  14336×4096 (CHANGELOG). Slightly further from llama.cpp's CPU logits: the
  reviewer's measurement on gemma-3-1b Q4_K_M (50 prefix lengths, one row per
  graph) gave mean KL 8.64e-3 for int8 against 7.04e-3 for f32, top-1 agreement
  45/50 against 48/50, int8 closer on 17 of 50 prompts, max |logit difference|
  between the two 1.315.
- **f32 for batches too (declined).** Closest to the exact sum, and bit-identity
  either way, but Q5_0 prefill would lose the int8 kernel. Measured (release,
  the generic dequantize-then-`dot` arm infr would fall back to, medians, two
  interleaved rounds, other builds sharing the machine), int8 against f32: 64
  rows 4096×4096 3.85 / 19.7 ms, 4096×14336 12.2-12.3 / 72.6-78.3 ms, 14336×4096
  12.6 / 74.2-75.2 ms; 512 rows 4096×4096 30.1-30.6 / 189-205 ms, 4096×14336
  103-123 / 624-716 ms, 14336×4096 111-123 / 578-767 ms: 5 to 7 times slower. A
  purpose-built f32 GEMM would narrow that; none exists.

**Context for the next time this comes up.**

- llama.cpp runs Q5_0 against int8 activations too (`ggml-cpu.c`:
  `[GGML_TYPE_Q5_0]` has `vec_dot_type = GGML_TYPE_Q8_0`), with an fp16 block
  scale (`block_q8_0`'s `ggml_half d`) where infr keeps the activation scale in
  f32, so the two still round differently.
- Cached models with Q5_0 tensors (read from their tensor-info headers):
  gemma-3-1b-it Q4_K_M, 117 (every layer's `attn_q`, `attn_k`, `ffn_gate`,
  `ffn_up`, and 13 `attn_v`); Qwen2.5-0.5B-Instruct q4_k_m, 133 (every layer's
  `attn_q`, `attn_k`, `attn_output`, `ffn_gate`, `ffn_up`, 12 `attn_v`, and
  `token_embd`, which is also its tied output head).
- Batched prefill stops one token short: the prompt's last token always runs as
  a one-row graph, so the first sampled token's logits come from the one-row
  path. The choice therefore moves `SeamModel::prefill_logits_cpu` and the CPU
  side of `prefill_matches_llama_debug_dump`, not only the decode steps.
- `Op::MoeFfn`'s one-row Q5_0 expert path stays on f32 activations on purpose
  (`int8_ok = rows >= 2 && !reference`, `crates/infr-cpu/src/moe.rs`); this
  decision is about `Op::Linear` only.
- What int8 activations cost in accuracy in general is B192.

### B202 — what the `infr_core::math` migration (plan 11, S2) left (2026-11-17)

**Tag:** cpu · determinism · math

- **Decision for the owner: `silu(-inf)` and `gelu_tanh(-inf)` are NaN.** The
  helpers reproduce the expressions the call sites always wrote,
  `x / (1 + e^-x)` (`-inf/inf`) and `0.5 x (1 + tanh(...))` (`-inf * 0`), so the
  migration changed no special value. The mathematical limits are 0 and `-0`.
  Returning them would touch every SiLU/GELU activation (`act_slice`, the gated
  norms, the short convolution, Metal's `act_fn`, the Diffusion-Gemma
  self-conditioning GELU) and needs the special-value tables in `docs/math.md`
  and `helper_special_values` changed; an `-inf` activation input is already a
  broken forward, so the gain is only that it would not turn into NaN there. The
  GPU shaders compute their own and were not checked.
- **`powf` is correctly rounded on every input tested, not proven** (see
  `docs/math.md`). Its production uses are RoPE's frequencies (`rope_freqs`,
  once per op) and the pager's Zipf shape; a proof needs a worst-case search
  over the pairs RoPE uses at least.
- **aarch64 is compiled, never run or measured.** CI's macOS job runs the grid
  hashes; the speed of the two-lane NEON path (the baseline tier there) against
  Apple's libm is unknown: run `bench_against_std` on a Mac (`docs/math.md`).
- **Not migrated to slices (scalar `math` calls, not hot):** the MoE router's
  sqrt-softplus, the hyper-connection sigmoids, DeltaNet's per-head gates,
  `Op::HeadGate` and `MoeSharedExpertAdd` (one sigmoid per head or row), the MLA
  rope's sines and cosines (per query, head and pair) and the Metal backend's
  host fallbacks. (Sinkhorn's softmax does use `exp_sub_slice`.)
- **`docs/synthetic-models.md` still argues that exact-token goldens are not
  portable** (its "Failure mode 2"). Plan 11 found the cause of the case it
  cites (the AVX-512 Q8_0 kernels, fixed in S4) and the CPU output is now the
  same on every OS, tier, thread count and chunk height tested; the section's
  advice (logits with tolerance, wide-margin weights) still holds for synthetic
  models, but its premise should be rewritten once CI's three systems have run
  the goldens.
- **What the S2 review checked only by reading.** The reviewer compared the
  branch op hash for op hash against a control (`origin/main` with every
  platform call rewritten one for one to scalar `infr_core::math`) across seven
  model families and the synthetic oracles. These paths were read but no run
  exercised them: `Op::Softcap`, the `Op::Softmax` op, `ArgmaxProb` and MTP,
  `MoeSharedExpertAdd`, diffusion and self-conditioning. The Metal backend's
  host math was compiled for aarch64, never run.
- **A decode `Linear` kernel's per-call time moves between builds that do not
  touch it.** gemma-3-1b's `Linear m=1 1152x6912 Q5_0` took 249-259 µs per call
  in two builds and 280-282 µs in a third (`INFR_PROF_OPS=1`, tg64, two
  interleaved rounds each, 2026-11-17), whose source differed from the second in
  the RoPE frequency cache (the one change on gemma's decode path), the math
  tier switch and its config type, `deltanet_scan` and the pager's Zipf shape:
  enough to move gemma's whole decode about 5%. A review found nothing in those
  changes on the one-row Linear path itself. Two guesses, neither confirmed:
  code placement (alignment of the hot loop), or the cache removing one
  allocation and free per RoPE op, which changes the allocator state and with it
  the address and alignment of the recycled activation buffer the Q5_0 Linear
  reads. Comparing that input's address mod 64 and mod 4096 in the two builds
  would settle the second. It means a decode A/B on this machine can show a few
  percent that belongs to neither side; a per-op profile shows where it lands.
- **`deltanet_scan` gaps a review found, neither reachable today.** Every shape
  in `deltanet_scan_bit_identical_to_serial` has `kd == vd`, so a transposed
  key/value dimension in the indexing would pass (real Qwen3.5 is square too).
  And `rows == 0` or a zero head dimension now panics in `chunks_mut` where the
  old form returned an empty result; the only producer (`runner.rs`) passes the
  batch size, and no zero-batch path was found.
- **Worth understanding: the int8 path amplifies last-bit differences.** Old
  against new decode logits on Qwen3-0.6B differ by up to about 1.2 absolute (KL
  up to 6e-2), from RoPE values that changed only in the last bit. The presumed
  mechanism is the int8 re-quantization of activations before each `Op::Linear`
  (a last-bit change moves a value across a rounding boundary of the int8 grid,
  a jump of one int8 step), but that is not confirmed. If it is the mechanism,
  it says how numerically sensitive the production int8 path is compared with
  `CpuBackend::reference()`, and a test that measures the amplification (one
  last-bit perturbation, logits distance on both paths) would put a number on
  it. Plan 11's Q8_0 fix (a last-bit change in one kernel's summation order)
  gave the same picture on Qwen3.5-0.8B (measured by the S3/S4 review,
  2026-11-17): before either golden forks, the top logit already differs by 0.07
  to 0.79 between `origin/main` and the branch; France forks at step 13 with
  top-two margins 0.62 and 0.22 on the two sides, the knight story at step 31
  with 0.034 and 0.099. This bears directly on plan 11's section 6: a GPU scored
  against the CPU reference must be teacher-forced and judged by
  distribution-level metrics, not by logit differences, which move this much
  from a last-bit change on the CPU alone.

### B206 — CPU speed from narrower datatypes: what is not int8 yet (2026-11-18)

**Tag:** cpu · perf · **Kind:** queued work, in this order

Quantized weights already run through int8 kernels on x86 (AVX2, AVX-512, VNNI)
with 8-bit activations. Where that is not yet the case, as reported by plan 11's
S3/S4 work and not re-read for this entry:

1. **aarch64: most families have no NEON kernel yet.** The `dotprod` tier
   (`kernels/neon.rs`) covers Q4_K, Q6_K and Q8_0, one-row and batch; CI's macOS
   leg holds them to the scalar bits (`tier_tests`, `op_hashes`) and prints
   their speed against scalar (`kbench:` lines); the `cpu-bench-macos` job
   prints the whole model's (`mbench:`). Measured on shared runners on
   2026-11-19 (`docs/math.md`, aarch64, has the tables): Qwen3-0.6B Q4_K_M
   decodes about 8.5x faster than scalar and prefills only about 1.9x (pp512
   about 100 t/s). **Next for prefill: an aarch64 packed or tiled GEMM** for the
   batch path; that this is where prefill time still is, is an inference from
   the low prefill gain and speed, not a measured profile. Still scalar on
   aarch64, in rough order of use: Q5_K (`Q5_K_M`), the native-32-block batch
   family (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 below 256 wide, IQ4_NL, MXFP4, NVFP4),
   Q3_K, Q2_K, IQ4_XS, the grid and ternary formats, and the f32/f16/bf16 dots.
   Also left: a NEON form of the Q4_K 2- and 8-row tiles (they run the batch
   kernel row by row, so each activation load serves one weight row where the
   AVX-512 tile's serves eight) and of the packed GEMM; an `i8mm` tier (`smmla`
   multiplies 2x8 by 8x2 bytes, a natural fit for the batch tiles, but its
   intrinsics are not stable Rust as of 1.100). **Unresolved: what the batch
   tiles' scale hoist (`b0dbceea`) did to speed.** Across two CI runs on
   different runner sessions, whose scalar baselines differ by 20 to 70%, the
   Q4_K and Q6_K batch ratios rose (2.57 to 3.04, 4.61 to 5.08) and Q8_0's fell
   (1.68 to 1.27); two runs cannot attribute that to the hoist. If the Q8_0
   batch kernel matters, the owed check is a controlled before/after in one
   process: both variants timed side by side, rounds interleaved. Issue 100's
   Mac owner can give quiet-machine figures: the commands in `docs/math.md`,
   aarch64.
2. **One-row MoE**: done; one-row `Op::MoeFfn` takes the int8 expert kernels
   (CHANGELOG), and decodes about 2% faster than the f32 path did on
   Qwen3-30B-A3B and ERNIE-4.5-21B-A3B (`docs/cpu-activations.md`, MoE models).

### B204 — what plan 11's S3/S4 (tier, thread and row identity) left (2026-11-17)

**Tag:** cpu · tests · ci · **Kind:** coverage gaps and open items

- **The CI cache will exceed its limit.** The repository's Actions cache was at
  10.52 GB before this work (`gh api`, 2026-11-17), with the old `-v2` Linux
  models entry still live; the golden steps add about 2.3 GB once `-v2` is
  evicted. Eviction is least-recently-used, so rust-cache entries will churn.
  The Windows models entry also stores Qwen3-0.6B a second time, beside the
  smoke job's `hf-windows-smoke-qwen3-06b-q4km-v1`; sharing one entry would save
  about 370 MB (see B197).
- **No test stops `CpuBackend::execute` from always taking the uncapped copy.**
  Every tier gives the same bits, so `op_hashes` and the capped goldens would
  still pass. A review confirmed the routing by planting a bit flip in an
  AVX-512 kernel (seen at `avx512` and uncapped, not at `avx2` or `scalar`). The
  missing test: run `execute` under a per-thread tier on a one-thread pool and
  assert the recorded tier. Likewise the uncapped half of `tier_tests` records
  no tier, so an uncapped copy that fell to scalar would be a slowdown no test
  sees.
- **The two interpreter copies decide "AVX-512 is present" differently.** The
  uncapped dispatch checks only `avx512bw`, as before; `SimdTier::Avx512` and
  `capped()` require F, DQ, VL and BW. On a machine that hides DQ or VL (some
  VMs) a ceiling of `avx2` would leave `capped()` false and the uncapped copy
  would still run AVX-512 kernels. Bits are unaffected; using one feature set in
  both costs nothing on a native build.
- **Thread counts: one threshold case does not vary its split.** `op_hashes.rs`
  now has a case past each pool threshold, but the lightning-indexer case splits
  into the same tasks at every thread count (its comment says otherwise). By
  reading, every parallel op partitions outputs and never splits one reduction
  across tasks (`Argmax`, `ArgmaxProb` and the samplers run serially).
- **`mul_add_exact` tests double rounding only in its constructed cases.** A
  random input lands on a halfway trap with probability around 2^-28, so the
  random cases do not test it; infinity and NaN inputs are not tested.
- **The goldens are weak detectors on their own.** A one-bit error planted in a
  Q8_0 batch kernel left the Qwen3 and Qwen3.5 golden tokens unchanged; the
  op-hash suite and the row-identity test caught it.
- **`docs/math.md` groups the synthetic LFM3 row test with heights 512/7/3/1;**
  that test uses 64/5/3/1.
- **Not traced: gemma-3-1b decodes faster on native builds** (tg64 about 41 on
  `main` against 45-46 on this work), level on baseline builds. See
  `docs/math.md`.
- **The interpreter is compiled twice.** infr-cpu's release `.text` grew from
  952 KB to 1160 KB with the two copies (833 KB on `main` before the ceiling).
- **The ±0/NaN audit's harmless sites** (`docs/math.md`, "Identity across tiers,
  threads, row counts and architectures"): softmax and attention row maxima, the
  lightning indexer's `d.max(0.0)` ReLU, the SwiGLU clamp's `gate.min(limit)`,
  the int8 quantizers' `|x|` maxima. If one of them is ever used for more than
  an `exp(x - max)` offset or a sum term, revisit it.
- **No NEON kernels.** aarch64 runs every family's scalar kernel. A NEON tier
  must compute the scalar expression; `kernels/tier_tests.rs` will hold it to
  that once it is a tier `crate::simd` dispatches.

### B207 — what the GPU tolerance gate (plan 11, G1/G2) cannot see, and left (2026-11-18)

**Tag:** gpu · tests · tolerance · **Kind:** coverage gaps, findings, decisions

`docs/gpu-tolerance.md` describes the gate; `tolerance::table::TABLE` holds its
values. What it leaves:

- **The production CPU's first-token row is wrong on SmolLM3-3B, and the
  production CPU is far from the reference elsewhere on gemma-3 and ERNIE.** Not
  noise and not the harness: the reviewer of this slice forced the reference's
  tokens through the production CPU on SmolLM3-3B and found every large KL at
  position 0 (prompt "Write…": KL 4.88; reference top-1 `"system"` at p=0.996,
  production top-1 `"bdb"` at p=0.169, its logits collapsed to [-13.5, 13.8];
  the other two prompts 3.33 and 1.92 at position 0), every later position at
  most about 2e-2. The measurement here (`measure_gpu_tolerance`, RX 7900 XTX
  native, CPU production against the CPU reference, worst KL at the first two
  positions / worst from position 2 on, per prompt 0, 1, 2):

  | Model            | First rows                         | From position 2 on              |
  | ---------------- | ---------------------------------- | ------------------------------- |
  | SmolLM3-3B       | 1.92, 4.88, 3.33 (all position 0)  | at most 2.0e-2                  |
  | gemma-3-1b       | 7.9e-3, 4.6e-2, 1.88 (pos 1, 1, 0) | 2.2e-2; **1.35 (pos 13)**; 0.97 |
  | ERNIE-4.5-0.3B   | 0.60, 0.27, 0.84 (all position 0)  | 0.22 (decode, pos 36); 0.12     |
  | Qwen2.5-0.5B     | 2.2e-2, 0.10, 5.1e-2               | at most 1.6e-2                  |
  | LFM3-1.7B Q4_K_M | 5.2e-2, 0.10, 9.9e-2               | at most 4.5e-2                  |
  | Qwen3-0.6B       | 7.5e-2, 7.7e-2, 3.7e-2             | at most 6.1e-2                  |
  | Qwen3.5-0.8B     | 2.6e-2, 8.7e-3, 6.8e-2             | at most 3.4e-2                  |

  On the same rows the shipped int8 GPU stays within 0.33 (gemma-3 position 0)
  and 8e-3 (SmolLM3). **Cause proven and fixed (2026-11-18):** `quantize_q8`
  took one scale per 256 activations, so the first token's massive-activation
  dimensions rounded the rest of their blocks to 0 or ±1; it now splits such
  outliers into a sparse part, and position 0 is at its steady-state order on
  every model above that was measured (`docs/cpu-activations.md` has the
  evidence, the options and the before/after tables). gemma-3's large distances
  away from position 0 are not outliers: B209 owns them. The table above was
  taken before the fix; the tolerance table's CPU-side rows were not re-run.

- **The int8 class only catches gross bugs.** With the int8-activation kernels
  on (the shipped default for every quantized model) the GPU's own activation
  quantization is two to three orders of magnitude larger than its float
  rounding. On Qwen3-0.6B (RX 7900 XTX, static decode path) an RMSNorm epsilon
  ten times too large and a 1% scale error on every block of one `ffn_down`
  tensor left the int8 run's distance where the clean run's is (steady KL
  1.35e-2 and 1.46e-2 against the clean run's 1.47e-2), so no bound on that run
  can see them, even after the first rows got their own bound; the int8-off
  (`vulkan_float`) run caught both. A bug that lives only in an int8 kernel is
  that run's blind spot: it is left to the op-level bounds (`seam_op_parity`'s
  `*_within_int8_bound`, `i8cm_kq_matches_its_decomposition`). An op-level check
  for each int8 kernel against its own decomposition, for every family the
  tuning table can switch, is what would close it.
- **A single mis-scaled block is invisible** (one Q4_K block's scale times 1.01:
  every metric at half its bound or less in both classes). Only a weight-level
  check (the dequant parity tests) covers it.
- **What the mutation demonstrations (G3) do not cover.** They are committed
  (`infr-llama`'s `mutations` feature, `tests/planted/`) and ran on both GPUs of
  the Linux machine:
  - **The GPU demonstrations are in no CI job**: they need a GPU. CI runs the
    CPU one (`planted_bugs_fail_the_gate_on_the_cpu`) and clippy with the
    feature, on the Linux legs of the `test` and `clippy` jobs.
  - **Not planted:** a dropped bias (q/k/v bias, Qwen2.5), any bug with the q8
    KV cache, any bug in a real MoE model (the synthetic MoE takes the router
    bias and two in its routed expert banks; its batched and int8 expert kernels
    never run), and any model but Qwen3-0.6B, gemma-3-1b, ERNIE-4.5-0.3B F16 and
    the synthetic ones. Bugs (a) and (b) opt their graphs out of the replayed
    decode, so the replayed decode never runs with either of them planted; every
    other bug reaches it.
  - **Two catches on the synthetic llama are thin.** In the CPU demonstration
    (c) is caught at a worst KL of 1.19x its bound (and a logit deviation of
    4.43e-3 against 3.9e-3), and (d) only on mean KL, at 1.43x (worst KL 0.73x).
    They are the first to turn into misses if the synthetic model or its row
    changes.
- **Routed-expert (MoE) models are forced along the reference's routing; what
  that leaves** (2026-11-23). A router near tie used to throw the comparison
  off: on the synthetic MoE both RADV GPUs select another expert at position
  107, layer 0, where the CPU reference's margin is 2.14e-4 of selection
  probability, and every later row inherited it (worst KL 2.87e-3 scored, 0.371
  at the position itself). A forced run now takes the reference's expert
  selections as well as its tokens (`Op::MoeFfn::expert_ids`), reads back what
  each of its own routers would have selected (`Op::MoeRoute`), and the gate
  holds those selections to the row's `route_near_tie`
  (`tolerance::metrics::RouteScore`); the synthetic MoE has a row and a gate
  test (`gpu_seam_tolerance_synthetic_moe`), and `docs/gpu-tolerance.md` has the
  mechanism, the measurements and the planted bugs. Open:
  - **No real MoE model has a row or a gate test.** ERNIE-4.5-21B-A3B Q4_K_M was
    measured on both RADV GPUs (`measure_gpu_tolerance`, key `ernie-21b-moe`;
    the table in `docs/gpu-tolerance.md`, "Mutations"): forced along the
    reference's routing its int8-free runs measure a worst KL of 6.1e-6 on the
    RX 7900 XTX and 3.8e-6 on the integrated GPU (6.8e-3 unforced on the RX 7900
    XTX), with at most 18 of 8802 selections differing, at margins up to 2.7e-5;
    its shipped int8 runs measure 3.6e-3 and 1.3e-2, with up to 434 selections
    differing at margins up to 3.2e-2, where a uniform share among its 64
    experts is 1.6e-2. So an Int8 routing near tie set by the rule would hold
    almost nothing there: either the Int8 class of a real MoE model gets no
    routing check (its Float runs carry it), or the int8 activation error at the
    router is looked into first. Needs a decision before its rows are set.
    Qwen3-30B-A3B is cached and was not run. Adding a row is a measurement on
    both GPUs plus a `gpu_seam_tolerance_*` test; whether the gate should carry
    a 13 GB model is the owner's call (on the integrated GPU the test's memory
    rule needs about 39 GB of free host memory for it).
  - **The routing near tie has no rule for a model that never selects
    differently.** The synthetic row's is 1.5 times the one legitimate
    difference seen. A row whose legitimate runs show none would get 0 by that
    rule, and then fail on the first device that does differ at a real near tie.
    The top-1 near tie has a floor for this (0.25 nats, 1 nat); a floor in
    selection score would have to be argued per model, since a router's margins
    scale with its expert count. Not decided.
  - **Windows has run none of it**: the row, the gate test, the own-routing
    check and the planted bugs are RADV only.
  - **A paged MoE model cannot be forced.** `execute_paged_moe` (Vulkan) refuses
    `expert_ids`, so a forced run of a model whose experts are paged fails with
    that error instead of running. Implementing the supplied selection there is
    the change.
  - **`Op::MoeRoute` on Vulkan repeats the router dispatch of the `Op::MoeFfn`
    lowering** (the choice between `linear_native`, `linear_f32` and `linear` by
    the router's dtype) so that no existing line of the adapter moved.
    **[read]** That choice is now written out in
    `crates/infr-vulkan/src/adapter.rs` at each of: the `Op::MoeRoute` arm of
    `lower_op`, the batched branch and the small-m branch of its `Op::MoeFfn`
    arm, and `execute_paged_moe`. They must stay in step. The one drift guard,
    `gpu_seam_tolerance_own_routing_is_the_selection_it_runs`, runs the
    synthetic model one token at a time: it covers the small-m branch with an
    f32 router (`linear_f32`) against the `Op::MoeRoute` arm, and neither the
    batched branch, nor the paged path, nor a quantized or f16 router. The fix
    is one helper that every one of those sites calls. Not done in this slice:
    it moves existing lines of a file a Windows session is working in.
  - **Vulkan reads back the selection, not the scores** (`moe_topk` does not
    write them; `Op::MoeRoute::dst_scores` is refused there), so only a CPU run
    can be the reference of a routing comparison. Metal refuses `Op::MoeRoute`
    by name: a `ForcedOn::Metal` arm needs it implemented first.
  - **The dual dense-and-MoE layers** (gemma-4 MoE, DiffusionGemma) emit the
    same readback and take a supplied selection through the same helper
    (`emit_route`), and no such model is cached here: **not run**.
  - **A forced run does not run the path from a router's own pick into its
    experts.** Its expert kernels read the supplied ids. That the two paths
    compute the same bits is shown on the synthetic model only (the own-routing
    check above, replayed and static decode, both GPUs), not on a batched
    prefill and not on quantized expert banks.
  - **A 1% scale of one layer's routed down bank is missed** by the synthetic
    row (0.35 of the KL bound and 0.77 of the mean bound on the CPU reference;
    1.36 times the mean bound on both GPUs, where the device's own distance adds
    to it). The shared expert beside the bank is not scaled.
- **Gate and up swapped (e) gives NaN logits on the RX 7900 XTX and finite ones
  on the integrated GPU** (both RADV, Qwen3-0.6B, both classes; KL 620x and
  3.3e5x its bound on the integrated GPU). The gate fails either way; why the
  two devices handle the same overflow differently is not looked into.
- **gemma-3-1b's rows were re-set on RADV; what that leaves** (2026-11-23, Linux
  7.3.5-arch1-1, Mesa `mesa` and `vulkan-radeon` 26.3.1-2, headless, no
  compositor running). Its Float check is split in two
  (`PathClass::FloatNoCoopmat`, the fine row, scored with the cooperative-matrix
  tier off, and Float, now a smoke row for the f16 cooperative-matrix prefill
  path), its Int8 smoke row's near tie is raised, gemma-3 is a planted-bug
  target (`gpu_planted_gemma3`), and `gpu_seam_tolerance_gemma3` passes on both
  RADV GPUs with no environment knob (`docs/gpu-tolerance.md`, "The table" and
  "Mutations", has the values and both sides of each gap). Open:
  - **Windows has not run the new rows.** FloatNoCoopmat is set from RADV alone
    (worst KL 8.81e-4 on both GPUs): no Windows run of the
    `gpu-float-nocoopmat*` configurations exists. The one older Windows figure
    that may be of this path, the integrated GPU's Float worst of 1.6e-3 quoted
    under "Against Windows" below, has no printed run behind it, is from the
    code of 2026-11-18, and is of this path only by inference (that device took
    the non-coopmat GEMMs on Qwen3); it set nothing, and would sit inside the KL
    bound. Owed on the Windows box: `gpu_seam_tolerance_gemma3` on both GPUs,
    and `measure_gpu_tolerance` for gemma with the three `gpu-float-nocoopmat*`
    configurations, natively and with wave64 forced. A run outside the row there
    is a finding to investigate, not a reason to widen it.
  - **The fine row's gap to (d) is thin.** A 1% scale of one layer's FFN output
    (on gemma the post-FFN norm's gain, `blk.10.post_ffw_norm`: the norm absorbs
    a scale of the whole `ffn_down`) measures a steady worst KL of 2.6e-3 on the
    CPU reference against the clean reference and 2.3e-3 on both GPUs: 1.26
    times the row's KL bound. It also fails the first rows' mean KL (2.5 times
    its bound on both GPUs, where the legitimate value is 1.6e-6 against a bound
    of 1.2e-4) and the steady cosine (1.5 times its bound's distance from 1).
    (c), the epsilon, is 14 times over the KL bound.
  - **gemma's Int8 smoke row does not hold at moved prompt ends, so those runs
    set none of it.** With prompt 1 ended early (`INFR_TOL_SPLIT`), so that
    position 13 and its neighbours are int8 decode rows, the RX 7900 XTX (native
    lanes, `gpu`) measures a decode worst KL of 0.618 (ended after 10 tokens,
    position 13; mean 9.0e-2, logit deviation 0.165) and 0.70 (after 12,
    position 40; mean 8.8e-2, deviation 0.192, cosine 0.546): over the row's
    steady KL, mean and deviation bounds, with a top-1 miss at a 1.349-nat
    reference margin (after 10, position 22, where the GPU's choice is 1.96 nats
    under the reference's), wider than the near tie. On the integrated GPU the
    prefill mean goes over its bound with prompt 2 ended after 16 tokens
    (3.46e-2 over 13 rows) and after 4 (4.0e-2, one row), and with prompt 1
    ended after 16 (3.2e-2); its worst KL there is 0.478 (prompt 1 after 12,
    position 13), inside. No gate configuration ends a prompt there, but a
    different prompt set could. Position 13 of prompt 1 is the row the
    production CPU misses by 1.35 (the table at the top of this entry; B209).
    Not looked into: whether the int8 decode GEMVs' distance on that row is all
    activation quantization. Needs a decision: re-derive the whole row with the
    moved ends in (it then admits more than it does now), or find why those rows
    are that far first.
  - **The Int8 row's margins at the gated prompt ends are thin on the integrated
    GPU.** Configuration `gpu` (the RX 7900 XTX with wave64 forced gives the
    same digits): worst KL 0.310 at prompt 1, position 13, against the 0.26 the
    row was set from on Windows, 1.7 times under the bound; mean KL 1.65e-2,
    1.63 times under; logit deviation 0.115, 1.30 times under. The rule asks for
    2, 2 and 1.5. Only the steady near tie was re-set (1.5 times the 1.185-nat
    miss at prompt 2, position 179).
  - **The coopmat-free runs repeat the Float runs on a device without the
    tier.** Where the Float run builds no cooperative-matrix GEMM (the
    integrated GPU; any device with wave64 forced) the three
    `gpu-float-nocoopmat*` runs build the kernels the three `gpu-float*` runs
    built and measure the same values, and gemma's test went from 116 s to 202 s
    on the integrated GPU. The obvious option: skip them when the two kernel
    lists are equal, which
    `gpu_seam_tolerance_nocoopmat_runs_build_no_coopmat_gemm` already computes,
    and hold the Float runs to the fine row there instead. Not done in this
    slice.
  - **The f16 cooperative-matrix prefill path's accuracy cost is unfixed.** On
    the RX 7900 XTX under RADV gemma-3's Float configurations measure a worst KL
    of 1.02e-2 (prompt 1, position 13) where the f32-operand path measures
    7.14e-4, and no tuning row matches RADV (B231), so the card's Int8
    configurations prefill on the same f16 GEMMs. The product option: run the
    first positions of a prefill through f32-operand kernels. It would help
    every cooperative-matrix device, Windows included; but the simulation below
    says a handful of rows is not reliably enough (three held, four to eight did
    not, of this prompt's 32), so it is a kernel-selection change with a
    prefill-speed cost on short prompts, unmeasured on any GPU, in code another
    session owns. Its size is not known until the first-rows curve is taken on a
    GPU and on more than one prompt. The evidence, all from before the rows were
    re-set ("the bound" below is the Float row's of the time, 6.4e-3 worst KL):
    - _Which kernels **[ran]**: the cooperative-matrix f16 prefill GEMM path._
      With `kernels.vulkan.coopmat` off (`gpu-float-nocoopmat`, the whole
      cooperative-matrix tier) position 13 measures 7.14e-4 and position 40
      4.22e-4, what `Vulkan1`, which has no cooperative matrices, measures. The
      two runs build the same kernels except `native_gemm_warp_q4k_sk_ag`,
      `_q6k_sk_ag`, `_q8_0_sk_ag`, `_q8_0_n128_ag` (or its `_bm16` build),
      `_q5_0_n128`, `splitk_reduce` and `store_f16` on one side and `native_q4k`
      on the other; every decode and attention kernel is in both lists, so
      decode runs the same kernels in both. (Forced wave64 selects the same f32
      prefill kernels and is not a second sample.) Turning off the attention
      warp kernels, `nc_fa`, the small row tiles, or staging A directly, and
      prefill chunks of 8, 16 or 32 rows, change no digit.
    - _Where it enters **[ran]**: through the KV of the first few positions, not
      through row 13's own prefill._ `INFR_TOL_SPLIT` ends the prompt early, so
      later positions are decode rows on kernels both configurations share.
      Position 13 / position 40 with the cooperative-matrix GEMMs on, by the
      number of rows the batched prefill ran: 3 rows 8.73e-3 / 6.55e-3; 4 rows
      7.42e-3 / 6.37e-3; 5 rows 1.03e-2 / 6.91e-3; 7 rows 5.86e-3 / 4.67e-3; 9
      rows 7.09e-3 / 6.13e-3; 11 rows 6.30e-3 / 5.23e-3; 13 rows 5.87e-3 /
      5.12e-3; 15 rows 1.02e-2 / 8.13e-3; 19, 23 and 27 rows 1.02e-2 / 7.85e-3,
      7.45e-3, 7.69e-3; all 32 rows 1.02e-2 / 7.87e-3. With only positions 0 to
      2 through those GEMMs the distance is already most of the whole, and from
      15 rows on position 40 stays within a tenth of its final value. The same
      splits with the tier off: 4.11e-4 / 2.90e-4 (3 rows), 6.19e-4, 8.07e-5,
      3.27e-4, 4.27e-4, 8.81e-4, 7.36e-4, then 7.14e-4 from 15 rows on (position
      13), so the f32 path's own value at that row moves elevenfold with where
      the prompt ends.
    - _Mechanism: f16 operand rounding, inferred, with a simulation that
      supports it **[ran, CPU]**._ The swap above changes kernel, tiling,
      summation structure and operand type together, and no f32-operand build of
      the cooperative-matrix kernel exists to hold the rest fixed (its operand
      type is a shader define with f16 and bf16 values only; whether the device
      enumerates an f32 cooperative matrix was not checked). What was run
      instead holds the kernel and changes only the operands, on the CPU: the
      reference backend with a scratch hook (not committed) that rounds the
      activations and the decoded weights of every batched-prefill `Op::Linear`
      to f16, to nearest even, and still takes its exact f32 dot. Against the
      unmodified reference that run measures 3.44e-3 at position 13 and 2.04e-3
      at position 40 (weights alone 5.08e-3 / 4.13e-3, activations alone 2.17e-3
      / 1.79e-3). The GPU's Float run scored against it measures 1.82e-3 and
      1.89e-3, where against the unmodified reference it measures 1.02e-2 and
      7.87e-3. So operand rounding alone produces a distance of this order on
      these rows and accounts for most of the GPU's; it is not shown to be all
      of it.
    - _How few rows matter **[ran, the same simulation]**._ Rounding one
      position's rows only: position 0, 2.14e-3 / 1.11e-3; position 1, 2.25e-3 /
      1.41e-3; position 2, 3.3e-4 / 1.8e-4; positions 3, 5, 8 and 12, at most
      3.6e-4 / 1.6e-4; position 13 itself, 6.6e-4 / 4.9e-4; positions 14, 15, 16
      and 20, at most 2.9e-5 at position 40. Keeping the first N positions exact
      and rounding the rest: N=1 2.35e-3 / 1.60e-3; N=2 4.48e-3 / 3.06e-3; N=3
      1.84e-4 / 2.68e-4; N=4 1.27e-3 / 1.26e-3; N=6 3.09e-3 / 2.08e-3; N=8
      2.31e-3 / 1.90e-3; N=10 4.6e-5 / 5.5e-5; N=12 3.4e-4 / 2.8e-4; N=14, 15
      and 16 at most 6.5e-5 at position 40. Positions 0 and 1 each carry about
      as much as the whole, the effects do not add, and no N under 10 holds the
      distance down reliably (N=3 does, N=4 to 8 do not). One prompt of one
      model; on the GPU only the complement was run (the split curve above).
    - _What the driver computes, as far as it was probed **[ran]**._ One
      variant, `native_gemm_warp_q4k_n128_ag`, at one shape (64 x 512 x 256) on
      synthetic operands, against host models: weights rounded to f16 to nearest
      even, exact products, summed: relative L2 1.66e-6; weights truncated to
      f16: 3.6e-4; products rounded to f16: 6.6e-5. gemma's failing value comes
      through the `_sk_ag` builds with `splitk_reduce`, `q5_0_n128` and
      `q8_0_n128_ag`, which were not modelled. In the gate on this device
      `recorder::tests::matmul_native_splitk_f16_matches_cpu` (a split-K build
      within its own 5e-2 absolute bound) and
      `gemm::tests::warp_gemm_matches_cpu` (the generic `gemm_warp` kernel, not
      a native-block one) pass; neither bounds rounding.
    - _The value moves with the summation **[ran, one reordering]**._
      `INFR_TUNE=split_k.fixed_grid=0` turns split-K off (the `n128_ag` builds
      run in place of `sk_ag` and `splitk_reduce`): position 13 measures 4.18e-3
      and position 40 1.98e-3, inside the row, and under the shipped int8 decode
      position 62 goes from 2.29e-2 to 0.112 with three top-1 flips. That is one
      sample (the other `split_k` settings tried changed no digit), and it does
      not exclude that the split-K path is the less accurate of the two. Read
      from the shader, the two builds differ in the order of the f32 sums over
      the same f16 operands; that a difference of that size moves this row by
      2.4 times is an inference, supported by the split curve, where position 13
      ranges from 5.86e-3 to 1.03e-2 with split-K unchanged.
    - _Why int8 on and off agree._ No row of `tuning/table.toml` matches RADV
      (B231), so the int8 cooperative-matrix GEMM is off and the Int8
      configurations prefill on the same f16 GEMMs as Float: the identical
      1.022885e-2.
    - _Against Windows._ The same card and kernel family measured a Float worst
      of 3.2e-3 there (position 40), against 3.0e-4 with wave64 forced and
      1.6e-3 on the integrated GPU, and it set the bound. Why that run is lower
      than this one was not established.
  - **The dp4a K-quant prefill GEMM is the far int8 kernel on the integrated
    GPU.** Configuration `gpu`, prompt 2 (180 tokens), position 179, the first
    decode row, flips its top-1 at a 1.185-nat reference margin (KL there 0.18);
    the near tie now admits it. Every Float configuration measures 3.1e-6 at
    that position.
    - _It does not follow the end of the prompt **[ran]**._ Ended at 150 tokens
      (position 179 a decode row 30 past the boundary) it measures 0.200 and
      flips; at 170, 0.170 and does not; at 183 and 190 (position 179 a
      batched-prefill row) 0.166 and does not.
    - _It is the KV the dp4a K-quant prefill GEMM wrote **[ran]**._ With
      `kernels.vulkan.dp4a_kq` off (`gpu-no-dp4a-kq`: `native_gemm_dp4a_q4k` and
      `_q6k` are the only kernels that drop out, their weights then taking
      `native_gemm_mmq_q4k_streamed` / `_q6k_streamed`, still int8 activations,
      and the int8 decode GEMVs unchanged) position 179 measures 2.17e-2 and
      does not flip, the decode phase's worst KL goes from 0.180 to 2.17e-2 and
      the prefill's mean from 1.53e-2 to 5.9e-3. Positions 0 to 2 keep their
      values to every digit (0.68, 0.81, 0.18), which was not explained. This is
      a lead for the entry below on gemma's int8 distance on this GPU: the far
      kernel is the dp4a K-quant GEMM, not the `mmq` one.
    - Position 179 is the last token of the prompt, where the reference's top
      two candidates are 1.185 nats apart; a KL of 0.17 to 0.20 is enough to
      swap them in some runs.
  - **Should every model have a FloatNoCoopmat row?** Only gemma-3 has one,
    because only its Float row was outside its bound on a cooperative-matrix
    device. The other models' Float rows carry the same f16-operand cost inside
    their bounds (Qwen3-0.6B's clean Float run is at 0.31 of its KL bound on the
    RX 7900 XTX and 0.13 on the integrated GPU, the planted-bug table), so a
    coopmat-free row per model would be a tighter fine check for each. Not
    measured for any other model: it needs `measure_gpu_tolerance` with the
    `gpu-float-nocoopmat*` configurations per model on every device, and three
    more runs per model in the gate.
  - **Not done:** the first-rows curve on a GPU; the split-K builds against a
    host model; any planted bug that lives only inside a cooperative-matrix
    GEMM, which is what the Float smoke row and the op-level tests are left to
    catch.

- **What the greedy check leaves** (2026-11-23; `docs/gpu-tolerance.md`, "The
  greedy check"). A forced run records and executes a fresh decode plan per
  step, so it never replays one recording at advancing positions and never
  chains. Each gate case now also generates greedily three ways (chained, one
  replay per token, the forced driver on its own argmax), from its plain prompts
  and from one chat-templated prompt, requires the same stop, the same tokens
  and bit-identical logits one step later, and holds the tokens to the CPU
  reference outside near ties. Open:
  - **Lost with the retired tests:** a 48-token generation (the longest is now
    33), the text hash through detokenization (token ids are compared; the CPU
    goldens still hash text), and the red paths shown on the old tests (a Vulkan
    rope base cut 1000×, a halved attention scale, LFM3's history rebuild), none
    shown again on a gate case.
  - **Windows has not run it.** Bit-for-bit agreement of a chain, a replay and a
    fresh recording was measured under RADV only, on both GPUs. A difference
    under the Windows driver would be a finding about that driver's replays, and
    would show as a red gate case, not a silent pass.
  - **The integrated GPU chains nothing** (`Backend::max_decode_chain` is 1 on a
    device that splits its submits), so on the Linux machine the chained decode
    is compared on the RX 7900 XTX alone; the integrated GPU compares the
    replayed decode with the stepped one.
  - **A chained step's logits are never read.** The check sees a chain through
    its tokens and through the one row after it. An error in the head or the
    argmax of a chained step that flips no token is not seen. A forced chain
    would need the chain to read back every step's logits.
  - **Not exercised:** the chained sampler (`Op::Sample`'s uniform ring,
    temperature above 0), a chain length other than the shipped
    `spec.decode_chain`, a chain or a replay on a KV format other than f16 and
    coupled q8_0, on a session continued from an earlier turn, and on every
    model without a gate case.
  - **The token rule sees gross bugs only.** On Qwen3-0.6B, both GPUs
    (`gpu_planted_greedy_tokens`): gate and up swapped and the rope pairing put
    several tokens of a prompt past the near tie in both classes; the nibble map
    and a dropped attention residual one token each (the residual, in Int8, on
    the chat prompt alone); the epsilon and the 1% scale none. The logit rows of
    the forced runs are what catch those.
  - **The stop rule and the chained count were shown red by throwaway edits**,
    not by a committed test (the runner returning after its first chained
    submit; the chained generation ignoring end tokens). A planted bug for each
    would make that permanent.
  - **A routed-expert model's greedy tokens are not held to the reference.**
    Left to its own routers a run parts from the reference at a router near tie,
    which the token rule cannot tell from a wrong token; a rule that reads the
    routing along the GPU's own tokens would be needed. Its chained, replayed
    and stepped generations are compared with each other. Only the synthetic MoE
    is a gate case, and it does not chain.
  - **The synthetic models do not chain** (their token embedding is gathered on
    the host), so the chained decode is exercised on real model files only.
- **The `Vulkan1` table data predates the full prefill head there.** Until
  2026-11-22 the integrated GPU's int8-off runs scored the prefill on each
  chunk's last row only, under a submit cap of 4, because the full head (one
  vocabulary-wide `native_q6k` dispatch, 3.26 s at the default 128-row chunk)
  was reset by Windows. The backend now cuts that dispatch into ranges, each in
  its own submit (plan 12 S2, B226), the workaround is gone, and the tolerance
  tests pass there with every prefill row scored and no table row changed. What
  is left: the `Vulkan1` rows were SET from the workaround runs (native lanes,
  tuning rows on only; gemma-3's and Qwen3.5's full-head int8-off runs had not
  completed), so `measure_gpu_tolerance` has not been re-run there to see how
  much room the full head leaves under them.
- **gemma-3-1b's int8 distance is ten times larger on the integrated GPU than on
  the RX 7900 XTX, unexplained.** Worst KL from position 2 on, configuration
  `gpu`: 0.26 on `Vulkan1` (prompt 2, position 115; 0.87 at position 1), 2.6e-2
  on the 7900 natively (0.33 at position 0), 0.11 with wave64 forced, 1.2e-2
  with the generic tuning (rows off). The other models' int8 rows stay within
  about 3x across the same paths, apart from LFM3 Q4_K_M (9x, its rows-off run
  the low one). The 7900 natively runs the int8 coopmat GEMM
  (`native_gemm_i8cm_*`); with wave64 forced, and on the integrated GPU, the
  dp4a `native_gemm_mmq_*` runs instead (the int8-free check's kernel lists, on
  Qwen3 **[ran]**; on gemma **[I]**). Those are the two far paths, so the mmq
  GEMM on gemma's large activations was the first guess. A later run on
  `Vulkan1` under RADV points at the dp4a K-quant GEMM instead
  (`native_gemm_dp4a_q4k` / `_q6k`: with `kernels.vulkan.dp4a_kq` off, prompt
  2's decode worst KL goes from 0.180 to 2.17e-2 and its prefill mean from
  1.53e-2 to 5.9e-3; the entry above on RADV). Not looked into further; it is
  what makes gemma's Int8 row a smoke check.
- **The int8-free check's marker list is typed by hand** (`INT8_KERNEL_MARKERS`
  in `tests/gpu_tolerance.rs`): the Vulkan crate has no central list of
  int8-activation kernels, so an int8 kernel named outside the markers would
  pass it unseen. A per-kernel "reads int8 activations" fact beside
  `tuning/classes.rs`'s classification would close it.
- **The top-set cosine is nearly blind on high-entropy rows.** Its single
  rest-of-mass component dominates when the union holds little mass (two rows
  flat over different 1000-token sets of a 32000 vocabulary: 0.99993 at KL 30),
  and `1 - cos` grows with the square of the leaked mass. KL gates the same
  rows, so nothing passes on the cosine alone. A Bhattacharyya coefficient
  (`Σ √(p q)`) would have the right shape; not changed now, because it means
  re-deriving the table.
- **One unreproduced failure of
  `gpu_seam_tolerance_prefill_head_leaves_decode_unchanged` on `Vulkan1`**, in a
  full GPU gate run (2026-11-18, the tree at `ad3b4458`, after about 170 other
  GPU tests, failed after 7.8 s; the assertion message was not captured here).
  The same binary then passed five times alone and three times after a
  neighbouring test (`goes_red`, `lfm3`), and the reworked test (two hook-off
  runs compared first, then `All` and `LastOfChunk`, small chunks) passed six
  times. That gate overlapped this round's stall experiments on the same GPU
  (section 0b of plan 12: hung and lost-device runs), the likeliest cause, but
  not shown. If it recurs, the reworked test's message says whether two hook-off
  decodes already differ (nondeterminism) or the head changed them, with the
  first differing row and token.
- **One run starved the host.** A full-configuration measurement of LFM3-1.7B
  F16 on `Vulkan1` took available memory from 32.5 GB to 2.4 GB before the
  memory guard stopped it; a repeat with one configuration peaked near 11 GB
  used and returned it between runs. Another session's WSL VM was also growing
  at the time, so the cause is not settled. The tolerance tests now skip, with a
  printed line, on an integrated GPU whose free host memory is under twice the
  model plus 12 GB; that prices one resident load, so it would not stop growth
  that is the harness's own.
- **The q8_0 KV class sits two orders above the float class** (Qwen3-0.6B worst
  KL about 2e-2 against 4e-5): the GPU quantizes the f16 RoPE output into the
  cache, the CPU reference its f32 one. It catches what the int8 class catches.
  Below 8 bits it is worse: K=f16 V=turbo3 measured KL 0.34-0.37 against the CPU
  reference running turbo3, with top-1 flips at 1.3-1.6-nat margins (Qwen3 0.6B
  and 1.7B), against 0.12-0.17 for V=q4_0. Worth checking that the GPU and CPU
  turbo3 codecs agree; not looked into, and no low-bit KV row is in the table.
- **Qwen2.5-0.5B's float prefill is twenty times further than the other Qwens'**
  (worst KL 8.1e-4, one top-1 flip; decode is normal). It carries q/k/v biases,
  which the others do not. Not looked into; Qwen2.5 has no gate row.
- **On the integrated GPU the gate filter leaves out every test named `moe` or
  `ernie`** (`& not (test(moe) | test(ernie))`, `docs/gpu-tolerance.md`,
  "Running it"). Pre-existing, and the doc gives no reason beside it; the reason
  inferred from the doc's other sections is host memory, since a resident MoE
  model on a unified-memory device takes twice its file plus a margin and the
  real MoE files are ERNIE-4.5-21B and Qwen3-30B. The cost: `test(ernie)` also
  drops `gpu_seam_tolerance_ernie` and the other tests of the 240 MB
  ERNIE-4.5-0.3B, and `test(moe)` drops the synthetic MoE's tolerance case, so
  those run there only when someone runs them by hand (2026-11-23: the whole
  `gpu_seam_tolerance_*` set was, and passes). Every ERNIE-21B GPU test has
  `moe` in its name, so `test(moe)` alone would keep the real MoE models out and
  let the small ERNIE cases back in, and the two synthetic MoE tolerance tests
  could be named back in; not changed here because the filter is the record of
  what was run.
- **`gpu_seam_tolerance_float_runs_build_no_int8_kernel` fails at its own guard
  when `gemv.kq_fma` is forced on under RADV** (2026-11-23, RX 7900 XTX,
  `INFR_TUNE=gemv.kq_fma=true,attn_split.decode_min_chunk=16`): the shipped run
  then builds no int8 kernel at all (fp32 decode GEMVs, and no int8 prefill GEMM
  under RADV, B231), and the test says it cannot fail. It is right to refuse to
  pass, and wrong to count as a gate failure on a device whose shipped path is
  int8-free: such a device (an RX 580 class under RADV, by default) cannot run
  the tolerance set green. What the test should do instead: when the shipped run
  builds no int8 kernel, print that the Int8 and Float classes run the same
  kernels on this device and assert that the two kernel lists are equal, the way
  `gpu_seam_tolerance_nocoopmat_runs_build_no_coopmat_gemm` treats a device
  without the cooperative-matrix tier. Not done: it changes what a pass of that
  test means, which is the owner's call.
- **Not covered by the gate:** every real MoE model (the synthetic one is; see
  the routed-expert entry above), KV formats other than f16 and q8_0, models not
  cached here (B60's Qwen3.5-MoE and DiffusionGemma), Metal (no Apple hardware:
  the doc says how a Mac owner would add it), and every device but these two.
- **`infr tune check` sees what the gate sees and no more** (2026-11-23). It is
  the same evaluation on the same prompts, so every blind spot above is the
  command's too: a tuning parameter whose effect lives only in an int8 kernel is
  judged by the Int8 smoke row; one that acts only past the prompts' depth (the
  longest prompt is under 200 tokens and each run decodes 32 steps), in a
  context deeper than that, in a batch shape the prompts never make, in the
  chained sampler or at another chain length, is not exercised at all. A PASS
  says the device is inside the model's rows on these runs, not that the
  parameter is safe at every shape. B220 has what the command itself left out.

### B234 — tolerance rows are keyed by file name: key them by content, and name a hub file without fetching it (2026-11-23)

**Tag:** gpu · tolerance · cli · hub · **Kind:** design gap, needs a decision

`tolerance::table::TABLE` finds a model's rows by its GGUF file name
(`Entry::model`), which is all `infr tune check` and the `gpu_seam_tolerance_*`
tests have to go by. **[read]**

- **A renamed file, or another repository's or quantizer's file under the same
  name, is scored against rows measured on a different file.** What stands in
  the way today: the command prints the row it matched and what the file says of
  itself (`general.architecture`, `general.name`, quantized or float weights,
  tensor count and bytes), and `gate::check_file` refuses a file whose weights
  are float where the model has an Int8 row, or quantized where it has none.
  That is the one fact the rows carry about their file. Two Q4_K_M files of one
  model from two quantizers (different imatrix, different per-tensor formats)
  pass it, and their distances to the CPU reference are not the same.
- **What keying by content would take.** A column per model in the table (not
  per row: a model's rows share a file): the file's identity as measured, for
  example the hub's LFS sha256 of the blob (the cache already names blobs by it,
  so a cached model costs a `readlink`, and a plain path costs a hash of the
  whole file), or a cheaper digest of the tensor directory (names, formats,
  shapes, sizes), which tells quantizers and re-quantizations apart without
  reading the weights and misses a file with the same layout and other values.
  Options: refuse a mismatch; or score it and mark the verdict as "of another
  file". The table's existing rows would need their files' identities read on
  the machines that have them (both GPUs' rows were measured on files from the
  `unsloth`, `LiquidAI` and `Qwen` repositories of `MODELS` in
  `tests/support/forced.rs`; the hashes were not recorded with the
  measurements). Not decided: which identity, and refuse or mark.
- **`infr tune check org/repo:QUANT` downloads the model before it can say there
  is no row**, when the repository is not in the cache. The file name such a
  reference resolves to comes from the hub's file listing (`repo_info` and
  `pick_gguf` in `infr-hub`), which only `infr_hub::pull` reaches, and it
  downloads what it picked. A path, a cached model and an explicit `.gguf`
  selector are refused before any fetch (`name_without_fetching` in
  `crates/infr-cli/src/tune_check.rs`, tested without network). Closing the rest
  takes a public `infr-hub` call that returns the picked file name from the
  listing alone; `infr-hub` was outside the slice that built the command.

### B231 — RADV runs on the generic Vulkan tuning: no table row matches it (2026-11-23)

**Tag:** vulkan · tuning · linux · perf · tests · **Kind:** coverage gap,
measurement to run

Every row of `crates/infr-vulkan/src/tuning/table.toml` names
`driver = "AMD_PROPRIETARY"`, so under Mesa RADV neither AMD GPU matches one:
the RX 7900 XTX gets the generic defaults (`split_k.scale_by_cu` off,
`gemm_i8cm_kq.enabled` off) and the integrated GPU the generic `queue.family`.
**[read]** the table; **[ran]** the kernels a shipped run builds on the RX 7900
XTX under RADV 26.3.1 hold `native_gemm_warp_*` and no `native_gemm_i8cm_*`
(Qwen3-0.6B and gemma-3-1b, 2026-11-23).

- **Coverage.** The Linux GPU gate never runs `native_gemm_i8cm_kq` through a
  model: on the RX 7900 XTX under RADV the Int8 tolerance configurations prefill
  on the f16 cooperative-matrix GEMMs, as the Float ones do (gemma-3's prefill
  rows are identical in both, B207), so the table's Int8 rows are scored there
  on int8 decode only. The kernel's own tests in `infr-vulkan` still run.
- **Performance.** The rows were measured on the Windows driver; their evidence
  fields give `attn_output` 458 to 302 us for the split-K row and pp512 of
  Qwen3-1.7B from 7580.6 to 11901.5 t/s for the int8 K-quant row. Whether either
  pays under RADV is unmeasured, and the int8 row's source notes that int8
  cooperative matrices hung an older Mesa.
- **What a tuning pass on RADV needs:** a quiet window; `infr bench` pp512 and
  tg128 on the models the rows cite, each parameter forced both ways with
  `INFR_TUNE`, interleaved; `i8cm_kq_matches_its_decomposition` and the int8
  cooperative-matrix layout check passing on Mesa 26.3.1 before anything is
  timed; the GPU gate with the candidate values forced; then rows keyed on the
  RADV driver name with those figures as their source.

### B208 — what the FMA tier (plan 11 S6) left (2026-11-18)

**Tag:** cpu · perf · tests · **Kind:** coverage gaps and open items

- **A baseline build still trails a native one, outside the f32 dots.** On a
  quiet machine (2026-11-19, twelve paired rounds, `docs/math.md`, the FMA tier
  section) a baseline build of the tier ran Qwen3-1.7B Q4_K_M pp512 at 90% of
  the native build (1021.7-1044.8 against 1114.2-1158.2 tok/s), gemma-3-1b pp512
  at 99%, and decode at 92% (Qwen3-1.7B) and 90% (gemma-3-1b). An earlier per-op
  profile, taken with another agent's test spinning one core, put the prefill
  gap mostly in the int8 `Linear` kernels (2048x6144 Q4_K 2.63-2.65 ms per call
  against 2.47-2.53 natively), whose surrounding code the compiler
  autovectorizes with the build's own features. Not investigated further.
- **Native reference mode is slower than baseline in the weight decode.**
  Confirmed on the quiet machine: Qwen3-0.6B tg64 in reference mode ran 8.1%
  faster on the baseline build (median paired, 12 of 12 rounds), and the native
  build was as slow before the FMA tier. With reference-mode `Linear` timed in
  two halves (a scratch build), the f32 dot takes the same time on both builds
  and `bytes_to_f32` (the Q4_K / Q6_K row dequantization to f32 it runs per
  weight row, allocating each row) takes 8 to 17% longer natively: the 1024x3072
  Q4_K rows 2.0-2.1 ms against 1.8 ms per call summed over threads, the Q6_K
  output head 115-117 against 106-107 ms. Why the native code generation of
  those decoders is slower is not traced (`dequant_dot_rows`' decoders, below,
  go the other way). Reference mode is the test oracle, so this costs test time,
  not users.
- **The per-op profile does not time reference-mode `Linear`.** Its branch in
  `CpuBackend::execute_with` ends in `continue`, which skips the per-op timer
  (`__t0`) at the end of the loop body, so `INFR_PROF_OPS=1` on a reference run
  shows a device total of a few percent of the wall time and no `Linear` row.
  Recording the time before the `continue` fixes it; not done here.
- **f32 multiply-adds not on the tier.** `seam/sc.rs` (Diffusion-Gemma's host
  self-conditioning: the soft embedding's f16 `mul_add` and the down phase's) is
  in `infr-llama`, which has no SIMD dispatch; putting it on the tier means
  making `kernels::Dots` public from `infr-cpu`. Not measured.
- **`dequant_dot_rows`' FMA copy decodes at baseline.** Its span decoders are
  separate functions, compiled for the build's own target: the Q4_K row took
  0.179 ns per element on the baseline build's FMA tier against 0.115 natively
  (`bench_float_dots`). It serves expert banks without an int8 kernel (Q2_K,
  Q3_K, the codebook formats: the shipped V4-Flash file's banks) and reference
  mode.
- **Short calls still pay a call.** The FMA copies cannot inline into the
  baseline-compiled interpreter: a lone 128-element `dot` took 0.252 ns per
  element against 0.243 natively, `axpy` 0.127 against 0.036. Moving a whole
  op's loop into the FMA compilation would remove it; nothing measured it at the
  op level.
- **Verified only on this AVX-512 Ryzen.** The "AVX2 without FMA or F16C" branch
  (the dots then run the C runtime's `fmaf`) is checked by reading only. The f32
  dots' NaN payloads are not compared (`tier_tests` folds NaN; no input makes
  one); the f16 KV write's NaNs are, exhaustively, and match.
- **Not run: a whole f16-weight model.** No F16 GGUF is cached here; the f16
  dot's gain (native 0.321 to 0.084 ns per element) is from the microbench.

### B220 — next up: the rest of plan 11's G4; then the queues (2026-11-22)

**Tag:** handoff · **Kind:** queue

**Top priority, the first slice when work resumes: what is left of plan 11's
G4** (`docs/plan/11-cpu-bit-identical.md`, section 6). It can be done on the
Linux machine as well as on the Windows box.

- **Done, 2026-11-23:** every gate case runs the greedy check (the chained and
  the replayed decode against the per-token path bit for bit, from plain and
  chat-templated prompts, with the stop and the chain-to-replay hand-off held,
  the greedy tokens against the reference outside near ties), and
  `gpu_seam_golden_qwen3`, `gpu_seam_flash_matches_cpu` and the exact-token
  tests of ERNIE-4.5-0.3B, Qwen3.5-0.8B and LFM3 are retired to their models'
  `gpu_seam_tolerance_*` tests. `docs/gpu-tolerance.md` has the check ("The
  greedy check") and the record of every exact and cosine GPU test with what
  covers it ("The exact and cosine GPU tests"); B207 has what the check leaves.
- **Done, 2026-11-23: `infr tune check <model>`.** The gate's evaluation is
  library code now (`infr_llama::tolerance::gate` and `greedy`; the
  `gpu_seam_tolerance_*` tests and the command both call `gate::check`, and
  `gate::runs_for` reads a model's runs off its rows), and the command scores a
  Vulkan device under the tuning in effect against a model's rows
  (`docs/gpu-tolerance.md`, "The command", has the interface and the recorded
  runs). What it left:
  - **No measuring mode.** A model with no row is an error. Printing distances
    with no verdict (`--measure`) was left out: without a row there is no class
    to pick the runs by and no near-tie margin to score top-1 at, so it would be
    `measure_gpu_tolerance` rebuilt in the command (every configuration, margins
    at 0, the three-way comparison) rather than a flag. Worth doing when someone
    without a checkout has to set a row.
  - **No op-level families**, which plan 11's first wording gave it: they are
    test binaries of `infr-vulkan`. Plan 10 (section 3.5) still lists
    `check [family]` as each family's known-answer check at the resolved
    parameters, a different command under the same name; the two need
    reconciling when plan 10's S5 is taken up (a subcommand argument, or another
    name). Not decided.
  - **No watchdog and no host-memory check**, which the tests have. On an
    integrated GPU a model that does not fit beside the reference in host memory
    is not refused.
  - **Run by hand only on RADV** (both GPUs, Qwen3-0.6B Q4_K_M). No automated
    test runs the command against a GPU: the tests cover its arguments, its
    refusal of a model with no row and its report, and the evaluation under it
    is the gate's. Windows has not run it.
  - **The reference is loaded under the shipped defaults**, apart from the
    configuration under test, so a setting that changes the CPU reference's
    arithmetic (a KV format) shows as a distance. Each run reports the settings
    its own edit wrote over (`gate::overridden`), but a setting given at its
    default value is not listed (the resolved configuration does not say which
    layer set a value), and nothing lists the settings that differ from the
    defaults and were left alone; `tune_hash` and `infr tune show` are how to
    see the tuning.
  - **Rows are found by file name only, and a quant selector of an uncached
    repository is downloaded before it can be refused** (B234).
- **Tests still to convert, each needing a row first** (the record in
  `docs/gpu-tolerance.md` lists them one by one): `gpu_seam_golden_qwen3moe`,
  the last text hash, and the real-MoE token and cosine tests (the Int8 routing
  near tie of a real MoE model is undecided, B207); the exact-token tests of
  Qwen2.5, Llama-3.2-1B and SmolLM3 (measured on Windows, no row); the KV-format
  cosine tests (only the coupled q8_0 cache has a row; turbo3's distance is
  unexplained, B207); the models not cached on the Linux machine.
  `gpu_seam_matches_cpu_gemma3` goes once `gpu_seam_tolerance_gemma3` has passed
  on the Windows box against the rows re-set under RADV.
- **Owed on the Windows box:** the `gpu_seam_tolerance_*` tests on both GPUs
  with the greedy check in them, which has run under RADV only.
- **What the retired golden unblocks:** `attn_split.decode_min_chunk` on the RX
  7900 XTX, held at 64 by that hash where 16 measured about 4% faster decode at
  shallow depth. `infr tune check` with the floor at 16 alone passes there under
  RADV (B227 has the measurement, the verdict and what the change still needs).

Then:

- **Waiting on other people:** issue 94 (the RX 580 owner; a follow-up with
  commands for the work bound is drafted, not posted), issue 100 (Apple Silicon
  numbers for the CPU math), issue 101 (the reporter was invited to open a PR
  for an incremental shader cache), llama.cpp issue 32424.
- **Next in the queues, not started:** the review findings on the PR follow-ups
  (B229); the families still scalar on aarch64 and the packed GEMM there (B206);
  plan 10's tuning slices S2 onward; plan 12's kernel ports from the fork; the
  activation split's sparse dot on the grid formats (B223).

### B229 — review findings on the PR 105 to 107 follow-ups, not yet fixed (2026-11-22)

**Tag:** vulkan · review · **Kind:** findings, not fixed (the session ended)

A reviewer read the follow-up commits on PRs 105 to 107 (B227) and found nothing
that blocked landing them. These were left; each was read in the code by the
reviewer, none was run.

- **Verify batches on the slow class cut at floor 16, unmeasured.** PR 105
  applied 16 to one-row decode only; `uses_decode_floor` extends it to every MTP
  verify batch that reaches split-K, so an RX 580 verify batch under 2048 keys
  runs up to four times the workgroups it did as merged.
- **`attn_combine.comp` changed on every device with no timing.** PR 105's
  rewrite (shared arrays, an extra barrier, batched loads) is ungated. The sum
  order reads unchanged and the goldens pass; no old-against-new timing exists.
- **Streamed weights never take the kq_fma kernels.** The dispatch in
  `quant_linear` needs `w.buffer`, and the streamed arm passes none, so on a
  `gemv.kq_fma` device a paged model decodes on the int8 or dequant route while
  its resident twin runs fp32. `quant_linear`'s doc lists its exceptions and
  omits this one. Inherited from PR 107.
- **About 55 GPU tests still skip silently.** `gpu_or_skip` now fails when
  `INFR_DEV` names a device that does not open, but tests that call
  `VulkanBackend::new()` themselves and return on an error do not go through it
  (unit tests in `adapter.rs`, `lib.rs`, `pager.rs`, `kq_gemm_tests.rs`, and
  `infr-llama`'s `seam/runner.rs` and `seam/weights.rs`).
- **The AMD integrated `queue.family` row cannot be limited to Windows.** The
  tuning table's `Match` has no OS field, so an AMD integrated GPU on a Linux
  driver reporting the same driver id would get the compute-only family,
  unmeasured. The row also rests on one integrated GPU.
- **`the_chunks_cover_each_key_once` cannot fail.** It checks a helper defined
  in the test. The GPU test `attn_split_host` is the real cover check.
- **`precise` does not cover every add in the kq_fma kernels.** `hsum`, the
  accumulator adds, the cross-subgroup sum and `subgroupAdd` are undecorated. No
  multiply is uncovered. The per-device parity tests are the guard.
- **Smaller:** the two Q4_K kernels do not spell identical float operations (one
  multiplies by a power of two where the other shifts; equal except at subnormal
  products); `QueueOutcome::reason` can contradict the family printed beside it;
  `docs/config.md` says an untimed compute-only family is never picked, while
  under `graphics` nothing warns and it is the last resort; bare literals for
  the floor defaults and in several comments; `honour_env` carries `INFR_TUNE`
  but not the `INFR_KQ_FMA_GEMV` and `INFR_GFX_QUEUE` aliases; the changelog
  cites a depth-2048 figure as a floor effect where the chunk is the same at
  either floor.

### B209 — gemma-3-1b's mid-sequence CPU distance is int8 noise, not outliers (2026-11-18)

**Tag:** cpu · accuracy · **Kind:** finding, needs a decision

Found while fixing the first-token defect (`docs/cpu-activations.md`). Against
the CPU reference, teacher-forced, gemma-3-1b Q4_K_M's production CPU has single
positions at KL 0.40 to 0.67 well past position 0 (its prompts' last positions;
the GPU-tolerance review saw 1.35 at position 13 and 0.97 on another prompt),
while its position 0 is ordinary. Those rows are not outlier rows: they split
the fewest activation blocks of any (`cpu_act_quant`'s
`localise_the_position_0_error` with `INFR_AQ_ROWS`), every op's own error on
them is ordinary, and the distance builds up through the layers (final hidden
state 0.09 to 0.17 relative distance against 0.04 at a quieter row). Simulated
in the reference (`linear_tap::simulate`): int16 activations on the 256-block
weights alone take the worst row from 0.52 to 0.09, on the 32-block weights
alone (gemma's Q5_0 and Q8_0 projections at width 1152) to 0.32, on both to
1e-5. So it is the int8 activations' ordinary rounding on a model that amplifies
it, on both activation paths. Options: 16-element activation scales (simulated
0.14 worst on that prompt, a rewrite of every kernel's epilogue), int16 for the
32-block path only, or accept it as gemma's sensitivity and score gemma's CPU
with a looser tolerance. Not measured: what llama.cpp's CPU gives on the same
rows.

### B210 — the activation split's ratio: speed against margin (2026-11-18)

**Tag:** cpu · accuracy · perf · **Kind:** decision taken, alternatives recorded

`quant::HI_RATIO` is 16 (`docs/cpu-activations.md`). The figure that costs is
the per-row one on the FFN-down inputs, where the split fires all through a
sequence (measured by the review: 5.4% / 16.1% of rows on Qwen3-1.7B's Q4_K /
Q6_K down inputs on the bench prompt, 32% / 38% on gemma-3-1b; about 1,930 and
7,100 split blocks per forward), not the per-block rate this entry first quoted
(0 to 1.1% of all projection-input blocks past position 0 on the bench prompt).
As first landed, each split block cost a one-block kernel call per weight row
(about 70 to 90 ns, 15 to 20 times a block of the packed GEMM): 3.5% of
Qwen3-1.7B pp512 and 1.9% of gemma-3-1b's on a native build, 1.65% and 1.5% on a
baseline build, and 2.9% of ERNIE-4.5-21B-A3B's MoE prefill (quiet machine,
2026-11-19). The exact sparse dot (`kernels/hi.rs`, 2026-11-19) evaluates the
same expression from the codes' weight elements alone, with no output bit
changed, and brought every prefill back to within the noise of the build before
the split, native and baseline (`docs/cpu-activations.md`, "Speed with the exact
sparse dot": medians -0.69% to +1.43% against `1b16e66b`, against an A-against-A
noise of up to 1.22%). What is left: gemma-3-1b's FFN-down layers at about 5 ms
of an 850 ms forward in the per-op profile, and the grid formats (B223). At 8
the split is somewhat better (ERNIE-4.5-0.3B's position 1, which attends to
position 0's value row, reaches 0.029 against 0.13 at 16 and 0.19 before) but
measured 14 to 17% slower Qwen3-1.7B pp512 in three disturbed pairs with the
one-block kernel calls; it fires on several times as many blocks, and what that
costs with the exact sparse dot is not measured. At 32 Qwen3-1.7B pp512 was
level in this branch's rounds, but 32 is the last ratio that still fixes ERNIE's
and SmolLM3's first tokens (64 leaves them at 0.61 and 0.30), so it has no
margin. Do not raise the ratio; whether 8 is now affordable is open (a
measurement, then the owner's call: it changes output bits).

**One position the split made worse, looked at (2026-11-19).** Qwen3-0.6B
position 18 of `cpu_act_quant`'s first prompt (KL 0.109 before the split, 0.402
after) is the `assistant` token after `<|im_start|>`; the same token is the
worst row of the two other chat prompts (prompt 1's position 23: 0.033 to 0.203;
prompt 2's position 23: 0.097 to 0.087, better). "Before" here is a scratch
build with `HI_RATIO` infinite, which never splits and so quantizes exactly as
before; it reproduced the review's 0.109 and 0.033.
`localise_the_position_0_error` on rows 17 and 18, before against after, per
`Op::Linear`:

- Row 18's own blocks barely split: 0.1% of its blocks over all ops at ratio 16,
  0.6 to 0.7% at ratio 8 (so about six times as many sit at 8 to 16 times their
  next magnitude as split, the review's proportion, but on a tiny base). Each
  op's own error on the row (`local`) is the same before and after to within
  ±10% all the way down (ratios 0.85 to 1.15, apart from one op that is smaller
  after: layer 7's FFN down, 0.56), and identical through the first two layers.
- The first difference is at op 17, layer 2's attention output projection (its
  input mixes earlier positions' values), and there the row's drift from the
  reference is smaller after (0.038 against 0.043). Drift moves both ways (0.66
  to 1.34 times its old value) until layer 24, then grows through the last four
  layers: the attention output inputs 0.118 to 0.170 (layer 24) and 0.116 to
  0.167 (layer 27), the output head's input 0.093 to 0.127.
- Row 17 (control) ends lower after (output head input 0.082 to 0.079).

So the worse KL is not this row's own quantization near the 16x threshold; it
arrives through attention from other positions whose keys and values the split
changed (position 0 the likeliest; no per-position ablation was run), and grows
in the last layers on a row the model is likely very sensitive on. A hint, not a
proof: the reference with the old 256-block quantizer simulated
(`linear_tap::simulate`, `256:0:0:127`) scores the same position at 0.484, four
times the real old production's 0.109. That simulation runs in reference mode
(weights decoded to f32, f32 dots, the simulated quantizer applied to `Linear`
inputs only), so the two differ in more than activation rounding, and part of
the gap may be that difference rather than the row's sensitivity alone.
Controlled here: the same prompt, tokens and teacher forcing, and the same
production code with only the split switched off. Not controlled: which of the
other positions' changes reach row 18 (no per-position ablation was run). Input
for the ratio question above, not a defect to fix.

### B211 — the 32-element activation path is not split (2026-11-18)

**Tag:** cpu · accuracy · **Kind:** open

`quantize_q8_32` (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4, NVFP4, Q2_0, and Q8_0
at a width that is not a multiple of 256) has one scale per 32 elements and no
split. Its blocks are small enough that the first token's outliers cost less
(gemma-3-1b's position 0 is at KL 2e-4 to 7e-4), but its row-0 local error is
still about twice the other rows' (gemma's Q5_0 projections: 1.6e-2 against
8e-3). The same split would apply (a one-block `Q8x32` part per split block).
Not measured on a model whose 32-block weights meet a large first-token outlier.

### B212 — llama.cpp parity at position 0 now differs by design (2026-11-18)

**Tag:** cpu · parity · **Kind:** behaviour to know

llama.cpp's CPU quantizes K-quant activations to `Q8_K` (256 elements, one
scale), the scheme infr used before the split, and has the same first-token
error: SmolLM3-3B's position 0 is at KL 1.85 from infr's reference in llama.cpp
and was 1.92 in infr. After the split infr's position 0 is close to its
reference and therefore 9.5 from llama.cpp's (Qwen3-0.6B 0.075, ERNIE BOS 0.094;
the first draft of this entry quoted 9.2, 0.073 and 0.085, measured at ratio 8,
before the committed ratio of 16). A parity test against a llama.cpp dump that
reads position 0 (a one-token prompt) will now see that distance; later
positions agree as before.

### B223 — the activation split's sparse dot on the grid formats (2026-11-19)

**Tag:** cpu · perf · **Kind:** open, small

The sparse part's dot (`kernels/hi.rs`, `docs/cpu-activations.md`, "How a part
is dotted") reads only the weight elements at a part's codes for the K-quants,
IQ4_XS, the TQ formats and Q8_0, about 6 ns per part and weight row against 115
ns for the old one-block kernel call (`kernels::hi::tests::microbench`,
single-threaded). The grid formats (IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS,
IQ1_S, IQ1_M) still expand the part's whole 256-element block
(`*_expand_row(blk, 256)`, two allocations and 32 grid lookups) for each part
and weight row, then sum only the codes' groups: 97 to 181 ns against 252 to 292
ns for the old call, so 1.6 to 2.7 times faster, not 20. A per-group decoder for
each grid format (the grid index, signs and scale of the one 8-element group a
code sits in) would bring them in line; it duplicates each `*_expand_row`'s
per-group logic unless those are refactored to call it, which touches the IQ
batch kernels' hot path. Not measured on a model: no grid-format model is in the
bench set.

Also left as it was: a split block still costs `quant::split_off`'s sort of 256
indices in the quantizer (a first-token row with two split blocks, 2048 wide,
6.9 µs against 1.3 for an ordinary row), and the first-token row's parts in the
attention and FFN-in projections each read their weight elements in a pass of
their own on the o-major routes (the packed VNNI GEMM adds them in its group
tasks).

### B224 — measurements that drift with the date or the binary's file (2026-11-19)

**Tag:** cpu · tests · perf · **Kind:** open, protocol

- **Date-stamped chat templates.** SmolLM3's template writes
  `Today Date: {{ strftime_now("%d %B %Y") }}` into its default system block,
  and `infr-chat`'s `strftime_now` (`template.rs`) formats
  `std::time::SystemTime::now()` through `infr_plat::localtime::strftime_local`
  with no way to fix the clock: no parameter, environment variable or test hook.
  So `cpu_act_quant`'s chat prompts on SmolLM3 tokenize differently every day,
  and its per-position KL after the date (positions 0 to 240 of those prompts
  are the shared system preamble that follows it) is comparable only within one
  day: on 2026-11-18 the worst chat position was 229 at 0.33 (240 at 0.22); on
  2026-11-19, on `origin/main` and the exact sparse dot alike, 240 at 0.691 (229
  at 0.027, the review's run). Nothing regressed. Fix: a way to pin the date for
  tests and measurements (a render option or a test-only clock in `infr-chat`),
  then use it in `cpu_act_quant`; not done here, to add no production surface in
  a CPU slice. Goldens checked (2026-11-19): `cpu_golden_smollm3` renders
  through `smollm3_render`, whose `/system_override` system message replaces the
  dated block, and asserts the render has no `Today Date`; of the cached GGUFs
  the CPU goldens use (Qwen3, Qwen3.5, gemma-3-1b, ERNIE, LFM3, Qwen3-30B-A3B,
  ERNIE-21B), only SmolLM3's template contains `strftime_now`. gemma-4-E2B's
  template was not checked (not cached here; its golden skips).
- **The binary's file.** In the 2026-11-19 rounds (`docs/cpu-activations.md`,
  "Speed with the exact sparse dot") a copy of the native binary ran 1.22% ahead
  of the original in 12 of 12 Qwen3-1.7B prefill rounds although the order was
  reversed every other round: an offset tied to the file, not its position. Next
  time, rotate the physical copies between rounds (each build as several copies,
  a different one each round) or take an A-against-A pair per build; until then
  a paired difference under about 1.2% is unresolved however many rounds agree.

### B214 — CPU goldens the activation split could not re-bless or confirm (2026-11-18)

**Tag:** cpu · tests · goldens · **Kind:** verification gaps

- **`QWEN3_QUANT_GOLDEN`** (`cpu_golden_qwen3_quants`) skips a row whose file is
  not cached, and that is how three rows went stale unseen (all eight were
  re-checked on 2026-11-19, Windows and WSL agreeing). CI's golden job caches
  only Q4_K_M, so CI runs that one row; the other seven run only on a machine
  that has their files (3.8 GB in all). The test now prints which rows ran and
  which were skipped. Closing the gap means caching the seven files in CI (cache
  cost, B204) or accepting a local-only check.
- CI does not run the MoE goldens (`cpu_golden_qwen3moe`,
  `cpu_golden_ernie4_5_moe`); both give the same hash on Windows and under WSL
  (`qwen3moe` checked 2026-11-19).
- Not run under WSL: the MoE models' chunk-height identity
  (`cpu_rows_give_the_same_bits_in_any_chunks`, which loads ERNIE-4.5-21B-A3B
  and Qwen3-30B-A3B one after the other; Windows passed). WSL here has a 30 GB
  limit, and the one 18.6 GB `qwen3moe` run took Windows free memory from 31 GB
  to 14 GB through the VM's file reads; the two in one process were not
  attempted.

### B226 — the work bound (plan 12 S2): what it leaves out, and what was not verified (2026-11-22)

**Tag:** vulkan · submit · watchdog · **Kind:** coverage gaps, decisions

`crate::work` estimates the GPU time of the heavy prefill dispatches and cuts by
it (`submit.budget_ms`): a dispatch estimated past the budget is recorded as
ranges of its grid, each in its own submit, and a submit is closed at the next
op boundary once its estimated work reaches the budget. Plan 12 sections 0d and
0e have the measurements behind it. What it does not do:

- **The budget is where cutting starts, not a ceiling.** A submit is closed at
  the next op boundary after its estimate reaches the budget, so one that was
  just under it takes one more op and can hold about twice the budget; and a
  dispatch has no more ranges than workgroups along the dimension that is cut
  (`WorkModel::pieces`), so a range of a coarse grid can pass the budget on its
  own. The defaults are set for the first: twice the budget is a quarter of the
  2 s limit on the devices that get reset and half of it elsewhere (500 ms since
  the second fix round; it was 1000, which with estimates corrected to real time
  made a full submit the whole limit). The second was not seen to matter.
- **Whether a fast discrete card is reset at 2 s was not tested.** The 500 ms
  default assumes it: Windows' `TdrDelay` is a system setting, not a per-adapter
  one, and `infr_core::SUBMIT_DANGER_NS` already treats a 1 s forward as
  dangerous on any device. No submit of 2 s was run on the RX 7900 XTX to see it
  reset; the measured resets are the integrated GPU's (plan 12 section 0d) and
  the reporter's RX 580.
- **Kernels that carry no estimate count as free.** Only these say what they
  cost (`Recorder::work_next`), and only over more than one row: the GEMMs
  recorded by `matmul_native_off`, `matmul_native_cm8`, `matmul_native_f16a`,
  `matmul_native_f32a`, `matmul_native_splitk`, `matmul_i8cm_kq`,
  `matmul_dp4a_kq`, `matmul_mmq_q4k`, `matmul_mmq_q6k`, `matmul_mmq_at` and
  `matmul_fma_at` (`tag_gemm`'s call sites), the per-row GEMV
  (`linear_native_at`'s multi-row dispatch), the fork's multi-row
  `native_gemv_kq_fma_mrow`, and the non-coopmat flash prefill attention
  (`attention_prefill_nc_fa`). Everything else is bounded by the dispatch cap
  alone. That includes prefill GEMMs, which the docs first described as all
  tagged: the cooperative-matrix GEMM of f16 weights outside the split-K plan
  (`matmul_proj_at`), the BF16 one (`matmul_bf16cm`), the opt-in Q8_0
  cooperative-matrix tiers (`matmul_i8cm_q8_0*`, `matmul_f8cm_q8_0*`) and the
  batched MoE expert GEMM (`matmul_mmq_experts`, `matmul_mmq_experts_paged`).
  And the rest: the multi-row GEMVs of the 2-to-16 row band
  (`native_mmv_mrow*`), every other attention kernel (the cooperative-matrix
  flash prefill, `attention_kv*`, MLA, the split-K decode kernels),
  convolutions, the DeltaNet and hyper kernels, MoE routing and scatter,
  embedding gathers, norms and activations. None of the second group was seen
  heavy in the profiles of plan 12 section 0d (the heaviest untagged dispatch
  there is under 4 ms, `rope_f16`); the untagged GEMMs were not profiled on a
  slow device. A model whose prefill is dominated by one of them there gets no
  cut from the work bound. To add one: tag it where it is recorded, with its
  FLOPs and `work::Split` (`Flat` only if the shader does not read
  `gl_NumWorkGroups`, which `record_dispatch` checks against the SPIR-V), and
  add its kernel to `tests/work_split_parity.rs`.
- **A batched MoE layer is one op with most of its work untagged.** Checked
  because a submit is only closed between ops: one `Op::MoeFfn` records ONE
  tagged dispatch, the router GEMV (`linear_native`), and its expert GEMMs
  through `matmul_mmq_experts`, which carry no estimate. No op records several
  tagged dispatches (`quant_linear`, `float_linear` and `nc_prefill_gemm` record
  one GEMM per `Op::Linear`, read 2026-11-22), so no op can run a submit far
  past the budget by estimate; but a MoE layer's real work is outside the
  estimate altogether. On the RX 7900 XTX, Qwen3-30B-A3B Q4_K_M at
  `-u 1024 -p 1024` was estimated at 151 ms a forward against 270 to 288 ms of
  GPU time, with no work cut and nothing learned (a segment that runs slower
  than its estimate corrects nothing). Not run on a slow device.
- **Refusing an uncuttable dispatch was removed (2026-11-22).** The first
  version refused, before submit, a dispatch that could not be cut and had been
  measured far past the budget on a reset-prone device. No tagged kernel is
  uncuttable, so nothing reached it and it had no GPU test; it went, with
  `work::Split::No`. A kernel that cannot be cut the way its tag says (a `Flat`
  tag on a shader that reads `gl_NumWorkGroups`, a `WideGrid` kernel on a
  three-dimensional grid) is recorded whole. If such a kernel is ever tagged and
  heavy, a refusal needs a GPU test that produces it.
- **A recording that cannot open submits keeps heavy dispatches whole.** A
  persistent recording (the decode replay tape, replayed whole) and every
  recording under `device.submit_dispatches = 0` record a tagged dispatch as
  one, whatever its estimate. No decode recording carries a multi-row GEMM heavy
  enough to matter in what was measured; a batched decode over many sequences on
  a slow device is the case to look at.
- **The estimate follows timestamps around one dispatch, which other dispatches
  can overlap.** A timed dispatch is bracketed by two timestamps with no barrier
  before the first, so a dispatch that started while earlier work was still
  running can read short. The model keeps the slowest of a kernel's last four
  timed samples (`work::TIMED_WINDOW`), which the first dispatch of an
  overlapping group always is, and on the integrated GPU eight independent GEMV
  dispatches in one submit read the same throughput as one alone (29.2 to 29.9
  GFLOP/s over the four probe points). Not shown: a kernel whose every dispatch
  overlaps a heavier one of another kernel. What bounds the damage is the gain
  bound (`work::max_prior_gain`): no reading takes a kernel's throughput past
  it, and a reading of no time is dropped. A barrier before the opening stamp
  would make the reading exact at the cost of serialising those dispatches; not
  done, since it would change what a slow device runs.
- **A segment's time is shared among its UNTIMED kernels by their priors, and a
  segment of mostly timed work corrects nothing.** `WorkModel::calibrate` treats
  the kernels without a timed dispatch as if they had taken the whole segment,
  and scales them by one factor when even so they ran faster than estimated.
  That can only err slow for the group, and a timed kernel latched on a slow
  sample no longer pushes the others down (the first version set the measured
  time against the whole estimate, timed kernels included, and did). Subtracting
  the timed kernels' estimate from the measured time, as review suggested, was
  not done: their rate is the slowest of a few samples, so it overstates them,
  and the overstatement would be handed to the untimed kernels as speed (a timed
  kernel estimated at 100 ms that really took 50, beside an untimed one whose
  200 ms prior is exact: measured 250 less 100 leaves 150, and the untimed
  kernel would be scaled to three quarters of its true time). Two costs. Within
  the group one factor is shared, so a kernel much slower against its prior than
  its neighbours still comes out too fast, by at most the gain bound, and only a
  timed dispatch raises it; nothing was built to provoke that on a GPU. And
  where a timed kernel carries most of a segment, the untimed ones stay on their
  priors: on the RX 7900 XTX with SmolLM3-3B at `-u 4096` only
  `native_gemm_i8cm_q4k_l` is learned, and the forward is estimated at 1098 ms
  against about 850 ms of GPU time.
- **The gain bound holds estimates above real time wherever a kernel is more
  than that much faster than its prior.** It is 3 at the integrated GPU's 250 ms
  budget (it was a constant 8, which let a full submit at the floor run 4 s if
  the prior was exact) and 2 at 500 ms. On the integrated GPU the int8 GEMMs are
  held at 384 GFLOP/s (timed at 502 to 770 before the bound) and the flash
  prefill attention at 48 (170 before), so its estimates read about twice the
  GPU time: Qwen3-1.7B `pp512` 1018 to 1253 ms a forward against 510 to 594,
  SmolLM3-3B `-u 2048` 55.8 s against 27.4 to 27.6. More cuts by work, no
  failure and no throughput seen lost: SmolLM3 takes 216 work cuts a forward
  where it took 180, and Qwen3-1.7B is cut by work where the dispatch cap cut
  it, four cuts a forward either way (pp512 231.0 t/s). On the RX 7900 XTX the
  timed matrix GEMM is held at 24576 GFLOP/s (27100 to 34952 measured before the
  bound). Raising the int8 and attention priors toward what was measured would
  remove the clamp on this GPU, at the price of a less pessimistic first forward
  on devices nobody measured; not done.
- **The prior's margin is thin for one shape.** `submit.prior_gflops` is 16 on
  the 2-CU integrated GPU; its per-row GEMV measured 29.9 GFLOP/s on a 2048 x
  22016 weight and 18.8 on 22016 x 2048 (the slow-class test's shapes). The
  budget's eighth of 2 s is what absorbs a prior that is wrong.
- **No 14B-class dense model was run.** The review asked for Qwen3-14B at
  `-u 1024` on the RX 7900 XTX; it is not in the cache and is larger than the
  download allowance. SmolLM3-3B at `-u 4096` stood in (about the same FLOPs a
  forward, by the prior 1875 ms of estimated work): plan 12's S2 status has the
  cut counts and the A/B.
- **The dispatch-cap loop still judges a forward the work bound already cut.**
  `observe_forward` divides a forward's whole GPU time by its dispatch count
  (`submit_cap_from_measurement`), heavy tagged dispatches included, and lowers
  the cap to what would have put that average at 250 ms a submit. With the count
  now every range of every recorder (2026-11-22, second fix round), on the
  integrated GPU: SmolLM3-3B `-u 2048` reports 802 dispatches for 27.4 to 27.6
  s, 34 ms each, which asks for a cap of 7; `-u 512` with `INFR_NO_I8DOT=1`
  reports 874 for 100.4 s, 115 ms each, a cap of 2. Both land on the floor of 16
  (`submit split/16`), as they did with the short counts of before. So the cap
  goes as low as it can although the heavy dispatches are already held by the
  work bound, and the light ones (norms, activations, the quantisation passes)
  are then cut every 16 for the rest of the process: more submits than needed,
  no failure. It is not double counting in the count any more; it is one average
  over two kinds of dispatch. Feeding the loop the forward's time less its
  estimated work, over its untagged dispatches only, would fix it; not done,
  because the estimate reads about twice the real time on this GPU (the gain
  bound, above), so the remainder would be negative here.
- **The integrated GPU's first prefill forward takes three submits by work** at
  default settings where it took none before the work bound (the pessimistic
  prior): Qwen3-1.7B, `pp512`, counted from the `RUST_LOG=infr_vulkan=debug`
  forward lines. Since the second fix round its later forwards are cut by work
  too, in place of the dispatch cap's cuts and as many (four a forward).
- **SmolLM3-3B `-u 2048 -p 2048` on the integrated GPU reads in two states,
  about 76 and about 89 t/s, and which one a run gets is not explained.** The
  first S2 run recorded 91.3 to 91.9 t/s; on 2026-11-22, later, three runs of
  the final binary gave 72.3, 72.5 and 71.7 and one of the binary from before
  the fix round 73.6, with the same cuts per forward as the first run (72
  dispatches cut, 180 work cuts), so the same work took longer on the GPU (about
  28 s a forward). An interleaved A/B of the two binaries then (six rounds, one
  run each of two copies per round, `-r 1`) read 72.6 to 81.0 t/s in its first
  three rounds and mostly 87.0 to 90.3 from the fourth on, with single runs of
  either binary in the other state among them (one binary 76.7 then the other
  87.0 in consecutive runs; three runs at 89 to 90 then one at 76.5). Means:
  81.22 t/s after the fix round against 82.06 before, with the two copies of one
  binary 3.3% and 1.4% apart, so that set does not compare the binaries. Under
  one busy core of 32 throughout but for one round at 1.07 (Windows'
  `CompatTelRunner`). The second fix round's three runs read 73.8, 73.8 and 79.2
  t/s, the third with forwards of 23.2 to 26.9 s where the other two took 27.4
  to 28.4. Qwen3-1.7B `pp512` on the same GPU is steady (231.03 against 231.07
  t/s over six rounds). To separate the binaries on this workload the state has
  to be found first: memory bandwidth taken by another process is the first
  thing to rule out, since this GPU reads its weights from DRAM [I].
- **`device.submit_dispatches = N` moves where the cap starts, not where it
  ends** (issue 94: 128 and 256 both ended on 62). The docs now say so. A
  DECISION for the owner: whether an explicit N should instead pin the cap,
  which would let a user raise it past what the measured loop allows, at their
  own risk now that the work bound holds the heavy dispatches. Not changed.
- **The RX 580's limit on the compute ring is not known.** PR 106 moved a
  discrete card's submits to a compute-only queue family, and amdgpu's
  `lockup_timeout` gives compute jobs a longer default than the 2 s the reporter
  measured on the graphics ring. The budget still assumes 2 s. Asked on
  issue 94.
- **The ordering barrier of a cut dispatch's later submits is not covered by a
  test.** `Recorder::rollover` opens each fresh command buffer with
  `seed_barrier`; removing it left every parity test green on the RX 7900 XTX,
  whose queue runs submits in order anyway. It is kept because Vulkan requires
  it.
- **The dispatch count of a cut forward was fixed in two steps, and one path has
  no test.** The first fix round counted every buffer a recorder submitted
  itself for a segment finished without waiting (`PendingSegment::dispatches`)
  and said the forward's count was then whole. It was not: the recorder a
  forward ends with is finished by a blocking `Recorder::finish`, and for it
  `execute_static` still took the open buffer's count. The second round added
  `Recorder::total_dispatches` for that one. `tests/work_split_parity.rs` now
  asserts the count a forward reports (`VulkanBackend::last_forward_dispatches`)
  against the uncut forward's count and the ranges recorded; its forwards end in
  the recorder that made the cuts, so it covers the blocking path (it failed
  four of six with the old line: 99 reported where 104 were recorded). The
  pipelined path has no assertion: it was read off the debug line (SmolLM3-3B
  `-u 2048` on the integrated GPU, 766 dispatches a forward where the first
  branch reported 397, the same with `INFR_PROF_OPS`).
- **The paged paths leave their submits out of a forward's dispatch count.**
  `rotate_stream` and `sync_stream` in `adapter.rs` submit the open segment (a
  ring rotation, or the blocking drain before a router readback) without adding
  its dispatches to the count `execute_static` hands the dispatch-cap loop,
  while those segments' GPU time is in the forward's sum. So a paged MoE forward
  reports too few dispatches for its time, the loop sees too much time per
  dispatch, and the cap comes out lower than it should: more splitting, not
  less. It predates this branch (both functions are on `main` as they are) and
  was not changed.
- **The timed window's first-sample rule was not run on a GPU.** Since the third
  review the prior stands in for the samples a kernel does not have yet, so its
  rate comes down only after `work::TIMED_WINDOW` fast readings. Host tests
  cover it; the cut counts and throughputs recorded in plan 12 are from the
  binary before it.
- **`Recorder::free_transient`'s wait has no test.** It now waits for the
  buffers a cut dispatch already submitted before it destroys the profiling
  query pool they write to (a discarded recorder under `INFR_PROF_OPS`). It
  needs a shutdown during a cut forward under profiling and was only read.
- **`INFR_PROF_OPS` across a cut dispatch was checked once, by a run.**
  SmolLM3-3B `-u 2048 -r 1` on the integrated GPU, profiled: the per-op tables
  of the measured forward (73 of them, one a recorder) sum to 27311.6 ms against
  27266.0 ms of GPU time for that forward, no stamp was dropped, and a cut
  dispatch is counted once, with all its ranges under its one label (the tables
  count 694 dispatches where the forward reports 802; the difference fits the
  extra ranges, which were not counted separately). The warm-up forwards are not
  profiled. No test asserts any of it.
- **A timestamp counter that wraps is handled in the arithmetic only.**
  `work::ticks_between` takes an interval under the queue's
  `timestampValidBits`, with a unit test; no device here was seen to wrap, and
  the per-op profile's own differences (`Recorder::report_timestamps`) still
  subtract over 64 bits.
- **A `LiveKernelEvent` 0x141 entry does not always mean dropped work.** Two
  were logged during runs all of whose segments wrote their closing timestamps
  (plan 12 section 0d), so counting those entries over-reports.
- **Not run:** GCN itself (the reporter's card); `INFR_FORCE_WAVE64` on the
  integrated GPU beyond the work-split parity tests; a model with a
  cooperative-matrix prefill on a device slow enough to cut it (the RX 7900 XTX
  cuts those kernels only under the tests' forced budget of 1 ms).

### B227 — PRs 105 to 107 as tuning parameters: what was measured here, and what was not (2026-11-22)

**Tag:** vulkan · tuning · attention · gemv · submit · **Kind:** decisions,
coverage gaps

Three pull requests from the fork of plan 12, written and measured on an RX 580
under RADV, were merged as they stood and each changed a default on every
device. On top of them (branch `fix/pr-followups`) each choice is a tuning
parameter. `gemv.kq_fma` (PR 107: the fp32 decode GEMVs) and
`attn_split.decode_min_chunk` (PR 105: 16 against 64) default to the contributed
behaviour on that class of device only, so on the RX 7900 XTX and the integrated
GPU they resolve to what ran before the merges, and `gpu_seam_golden_qwen3`,
which PR 105's floor had moved, passes unchanged. `queue.family` (PR 106) is
`auto` by default on every discrete device and, by a table row, on an integrated
GPU under the AMD Windows driver, and `auto` takes a compute-only queue family:
the project owner's decision, for a responsive desktop.

**Measured here.** `infr bench` of Qwen3-1.7B Q4_K_M, the parameter forced each
way with `INFR_TUNE`, 6 interleaved rounds, one process per sample, two copies
of one binary rotated between the arms, context 8192. Each cell is the mean with
the lowest and highest sample; the load column is the machine's total CPU over 3
s before and after the row. `Vulkan0` is the RX 7900 XTX and `Vulkan1` the Ryzen
integrated GPU, both under the AMD Windows driver, with the Windows desktop
running on the machine.

| Device    | What, t/s                  | First value                               | Second value                             | Change | Load before / after |
| --------- | -------------------------- | ----------------------------------------- | ---------------------------------------- | ------ | ------------------- |
| `Vulkan0` | floor, tg128 at depth 16   | 64: 469.06 (465.46 to 471.96)             | 16: 488.36 (485.96 to 492.29)            | +4.1%  | 0.7% / 2.0%         |
| `Vulkan0` | floor, tg128 at depth 64   | 64: 464.17 (461.33 to 465.64)             | 16: 485.65 (479.86 to 489.07)            | +4.6%  | 1.7% / 1.1%         |
| `Vulkan0` | floor, tg128 at depth 256  | 64: 458.27 (456.26 to 459.51)             | 16: 474.52 (471.32 to 476.39)            | +3.5%  | 1.7% / 0.7%         |
| `Vulkan0` | floor, tg128 at depth 1024 | 64: 449.79 (447.41 to 451.51)             | 16: 449.33 (445.67 to 451.13)            | -0.1%  | 2.2% / 1.5%         |
| `Vulkan0` | floor, tg128 at depth 4096 | 64: 379.85 (379.24 to 380.52)             | 16: 379.20 (377.64 to 380.88)            | -0.2%  | 1.7% / 0.8%         |
| `Vulkan0` | `gemv.kq_fma`, tg128       | off: 473.20 (467.43 to 475.37)            | on: 428.54 (425.17 to 434.90)            | -9.4%  | 1.9% / 1.8%         |
| `Vulkan0` | `queue.family`, tg128      | graphics: 474.86 (474.34 to 475.66)       | compute: 454.79 (454.20 to 455.17)       | -4.2%  | 1.7% / 2.2%         |
| `Vulkan0` | `queue.family`, pp512      | graphics: 11611.42 (11508.54 to 11743.71) | compute: 11482.28 (11426.91 to 11527.87) | -1.1%  | 1.5% / 2.4%         |
| `Vulkan1` | floor, tg128 at depth 16   | 64: 27.07 (26.92 to 27.31)                | 16: 26.93 (26.80 to 27.02)               | -0.5%  | 1.2% / 1.7%         |
| `Vulkan1` | floor, tg128 at depth 64   | 64: 26.84 (26.76 to 26.92)                | 16: 26.72 (26.49 to 26.94)               | -0.4%  | 2.0% / 0.9%         |
| `Vulkan1` | floor, tg128 at depth 256  | 64: 26.30 (26.26 to 26.34)                | 16: 25.87 (25.76 to 25.94)               | -1.6%  | 1.2% / 1.4%         |
| `Vulkan1` | floor, tg128 at depth 1024 | 64: 24.44 (24.36 to 24.58)                | 16: 23.95 (23.92 to 23.97)               | -2.0%  | 0.6% / 2.7%         |
| `Vulkan1` | floor, tg128 at depth 4096 | 64: 19.08 (18.98 to 19.18)                | 16: 19.13 (19.05 to 19.19)               | +0.3%  | 1.6% / 2.0%         |
| `Vulkan1` | `gemv.kq_fma`, tg128       | off: 27.15 (27.06 to 27.36)               | on: 22.75 (22.64 to 22.80)               | -16.2% | 2.6% / 3.9%         |
| `Vulkan1` | `queue.family`, tg128      | graphics: 27.12 (26.98 to 27.22)          | compute: 27.12 (27.07 to 27.15)          | +0.0%  | 3.6% / 1.3%         |
| `Vulkan1` | `queue.family`, pp512      | graphics: 230.94 (230.67 to 231.16)       | compute: 230.84 (230.58 to 231.25)       | -0.0%  | 0.6% / 1.8%         |

**The compute-only family here.** Both devices have one (queue family 1) and it
reports 64 valid timestamp bits, read from the startup log. The `queue.family`
rows above compare `graphics` with `compute`; they were taken while the
parameter was still a switch, whose two positions are those two families. On the
compute-only family, on each device, `first_forward_split`, `work_split_parity`,
`gpu_seam_golden_qwen3` (the hash unchanged, so the queue does not change
generated output) and `gpu_seam_tolerance_qwen3` / `_qwen35` passed. Timestamps
work on it: a profiled `pp512` on `Vulkan0` (`INFR_PROF_OPS=1`) reported a
device total of 42.1 ms for a forward of 47.0 ms wall on the compute-only
family, against 41.3 ms of 46.7 ms on the universal one. The integrated GPU's
row in `table.toml` (`amd-proprietary-integrated-compute-queue`) rests on the
two `Vulkan1` rows above. Not measured: how long a submit may run on that family
before Windows resets the device (the `Vulkan1` figures of plan 12 section 0d
are from the universal family).

**Forced configurations, `Vulkan0`.** The slice's GPU test set (49 tests: the
golden, `gpu_seam_tolerance_*`, the split and parity binaries, the new kernel
tests) passed whole at the defaults on both devices and under `INFR_NO_I8DOT=1`,
`INFR_NO_COOPMAT=1`, both together and `INFR_FORCE_WAVE64=1`. The two capability
masks do not reach the whole-model tests of `infr-llama`
(`docs/gpu-tolerance.md`), so those runs exercise them in `infr-vulkan`'s tests
only. Under `INFR_TUNE=gemv.kq_fma=true`, which does reach them, 48 passed and
`gpu_seam_golden_qwen3` did not: the fp32 decode kernels give the continuation
`0x94f246b09d34db5d`, the same one the 16-key floor gives, where the blessed
hash is of the int8 decode path. Every tolerance test passed with it on, so the
fp32 path is inside the CPU-reference tolerance; the golden is a hash of one
device's default path and is not expected to hold under a forced kernel.

**Decisions that need a call:**

- **The decode floor on the RX 7900 XTX.** 16 measured faster there at depths
  16, 64 and 256 and the same at 1024 and 4096 (table above), so by speed alone
  it should be 16. It stays 64 because 16 changes decode's attention sums for
  every context under 2048 keys, and `gpu_seam_golden_qwen3` then generates a
  different continuation (hash `0x94f246b09d34db5d` where the blessed one is
  `0xcf56ba8c4bb5c455`, from the test's failure message; the text it printed
  reads as a normal answer). That golden is retired since 2026-11-23 (plan 11's
  G4: the chained and the replayed decode are held to the per-token path bit for
  bit instead, B207), so no hash holds the floor any more. Measured that day on
  the RX 7900 XTX under RADV with
  `INFR_TUNE=gemv.kq_fma=true,attn_split.decode_min_chunk=16` (both forced
  together; the built kernels show `native_gemv_kq_fma_*`): every model's
  `gpu_seam_tolerance_*` case passes, greedy check included. One test of the set
  fails under the two together,
  `gpu_seam_tolerance_float_runs_build_no_int8_kernel`, at its own guard: with
  `gemv.kq_fma` forced the shipped configuration builds no int8 kernel under
  RADV (the int8 prefill GEMM does not run there either, B231), so the check
  says it cannot fail. Each alone, the same day, device and driver, with
  `infr tune check` on Qwen3-0.6B Q4_K_M (every check of the model's six
  configurations and the greedy check; no default was changed):

  | Forced                                                     | Verdict | Closest to a bound                                                          | Decode worst KL, `gpu` (Int8) | Decode worst KL, `gpu-float` (Float, bound 8.5e-5) |
  | ---------------------------------------------------------- | ------- | --------------------------------------------------------------------------- | ----------------------------- | -------------------------------------------------- |
  | nothing                                                    | PASS    | 0.49 of it: `gpu-float` prefill mean KL 3.575e-6 against 7.3e-6             | 4.626e-3                      | 1.602e-5                                           |
  | `--set kernels.vulkan.tune.attn_split.decode_min_chunk=16` | PASS    | 0.53 of it: `gpu-float-ub32` decode logit deviation 1.014e-3 against 1.9e-3 | 7.588e-3                      | 1.411e-5                                           |
  | `INFR_TUNE=gemv.kq_fma=true`                               | PASS    | 0.57 of it: `gpu-float-ub32` decode logit deviation 1.091e-3 against 1.9e-3 | 1.564e-5                      | 1.564e-5                                           |

  Each override is seen in the values. Against the unforced run's table, 27
  decode lines, one prefill line (`gpu-ub32`'s logit deviation) and one greedy
  token count differ with the floor at 16; 23 decode lines, the same prefill
  line and one token count with `gemv.kq_fma`, where the shipped configuration's
  decode lands at the int8-free one's distance (its decode GEMVs are then the
  fp32 ones). So on this device and driver, for this model, each is inside the
  rows on its own. Still to do before the default changes: the whole GPU gate
  with the floor at 16 alone, on the Windows driver the speed was measured on;
  neither was run. To try it: `INFR_TUNE=attn_split.decode_min_chunk=16`, which
  reaches the tests too, and `infr tune check <model>` under it for one model at
  a time. If it is taken, a class row in `table.toml` for
  `vendor = "amd", driver = "AMD_PROPRIETARY", integrated = false` is the place,
  with these figures as its source.

- **The decode floor on the integrated GPU** stays 64: 16 measured slower there
  at depths 256 and 1024 and within the samples' spread elsewhere.
- **The compute-only family on the RX 7900 XTX** is the default
  (`queue.family = auto`) by the owner's decision and costs 4.2% of decode
  there; whether that buys a more responsive desktop on Windows was not measured
  (no frame-time or compositor statistic was taken while a forward ran). B228
  has the conditional follow-up.
- **`gemv.kq_fma` on devices with an accelerated dot** stays off: the table
  above has what on costs.

**Left inline, measured on Polaris only (PR 107):** the rows-per-workgroup
picker `Recorder::kq_fma_gemv_rows` (8 rows only when that fills the last pass
at least 1.1 times better and still leaves 4 workgroups per compute unit) and
`Recorder::linear_f32_at`'s threads-per-output choice for `linear_f32_mo` (64,
or 32 when a row has under 128 vec4s). Both are hardware-shaped numbers of the
kind plan 10 makes tuning parameters; neither was swept here, because on both
devices here the kernels they shape are off by default.
`Recorder::linear_native_routed_at`'s bf16 arm also still requires
`out_f <= MAX_GROUP_COUNT_X`, which `linear_bf16_at` no longer needs (it splits
its grid); a wider one-row bf16 weight takes the generic GEMV.

**Changed on top of the PRs without the RX 580 to check them:**

- **`precise` in `native_gemv_kq_fma.comp` and its `_mrow` twin.** It was needed
  for the verify twin to be bit-identical to the decode kernel on the RX 7900
  XTX. It can only cost speed where the compiler was contracting multiply-adds.
  `kq_fma_gemv_bw` on the RX 7900 XTX: one run before and three after, single
  shapes between 15% lower and 36% higher, which is that bench's own spread from
  run to run, so no cost was established and none excluded. Not measured on GCN,
  the class the kernels are on for.
- **Decode takes the fp32 K-quant kernel ahead of the int8 GEMV at the plain
  (non-fused) `Linear` site.** As merged, `mmv_mw_choice` ran first there, so on
  a device with an int8 dot a plain decode `Linear` of 2M elements or more kept
  the int8 kernel (the fused-residual site already had the fp32 one first). PR
  107's commit message describes the order now in the tree, and its figures were
  taken in the fork; whether the merged order was ever what ran on the RX 580 is
  not known here.
- **A device that reports no facts** (the `unknown` fixture: no matrix units, no
  accelerated dot, not integrated) gets all three contributed defaults:
  `gemv.kq_fma` on, floor 16, compute-only queue. That follows the class rule;
  nothing was run on such a device.

**Not verified:**

- **No whole-model MTP run with a parameter forced.** At the defaults, on RADV
  (2026-11-23, Linux, Mesa 26.3.1, headless, Qwen3.5-4B-MTP),
  `mtp_head_cpu_vulkan_parity` passes on the RX 7900 XTX and the integrated GPU.
  `mtp_spec_matches_target_only_greedy`, ignored as "MTP parked", passes on the
  RX 7900 XTX and fails on the integrated GPU, where the speculative stream
  differs from plain greedy from its first words; its doc comment expects a
  failure while MTP is parked, and it was not looked into. No term of the GPU
  gate's filter (`docs/gpu-tolerance.md`) selects that test, so a gate run never
  shows its state on either device. Neither was run with
  `attn_split.decode_min_chunk=16` or `gemv.kq_fma=true` forced, the two
  settings the PRs brought, nor on Windows.
- **The work bound's 2 s on the compute ring.** `crate::work` assumes a submit
  is reset after about 2 s (plan 12 section 0d), measured on the universal
  family of the integrated GPU under Windows. No reset was provoked on a
  compute-only family on any device or driver, so whether the limit there is the
  same, longer (amdgpu's documented compute-ring timeout) or tighter (what
  `docs/igpu.md` records for an integrated part under RADV) is unknown. The
  dispatch-budget probe (`tests/dispatch_budget_probe.rs`) with
  `INFR_TUNE=queue.family=compute` would measure it here; on Linux it needs the
  RX 580.
- **The compute-only family on other drivers.** NVIDIA, Intel, and RDNA under
  RADV were not run. `VulkanBackend::pick_queue_family` refuses a family with no
  valid timestamp bits; both devices here report 64 for theirs, so the refusal
  is covered by its unit test only.
- **A verify batch and decode agree bit for bit only where the chunk is the
  floor.** Past 32 times the floor the chunk is `span / 32`, and a batch's rows
  share the batch's span while each decode step has its own, so the two cut
  differently. That was so before the PRs (from 2048 keys); at floor 16 it
  starts at 512 keys.
- **Static and replay decode cut differently past `floor * 1024` rows of cache
  capacity.** The replay path bakes a floor that rises with the capacity
  (`attn_chunk::baked_min_chunk`), the static path (`INFR_SEAM_NO_REPLAY`, and
  every verify batch) does not. Before the PRs that began at 65537 rows; at
  floor 16 it begins at 16385, so a slow-class device with a context past 16384
  is in it. `replay_cuts_a_span_like_the_static_path` pins where the two agree.
- **`linear_f32_mo_covers_a_vocabulary_wide_output` does not fail without the
  grid split on the RX 7900 XTX**: with the kernel put back on a one-dimensional
  dispatch the driver ran 65536 workgroups anyway. The split is held by
  `elementwise_ops_split_wide_grids`, which lowers the split point; the
  vocabulary test fails when the kernel's flat-index recovery is removed.

### B228 — the compute-only queue on Linux: what to measure; display detection declined (2026-11-22)

**Tag:** vulkan · submit · tuning · linux · **Kind:** measurement to run,
decision

infr submits on a COMPUTE-only queue family by default where a device has one
(tuning parameter `queue.family`, default `auto`;
`VulkanBackend::pick_queue_family`). The reason is a responsive desktop: the
compositor renders on the universal family's ring, and a long forward on that
ring stalls its frames. Nothing about it has been measured on Linux by the
project. The owner has a Linux machine and will run this.

**What is unmeasured.** The cost or gain of the compute-only queue on Linux
under RADV (amdgpu), on whatever GPU that machine has, in two states:
**headless** (no compositor or display server running) and **with a desktop
session running**. The owner's hypothesis is that headless it costs nothing. The
only Linux figure is the contributor's, on an RX 580 under RADV (Polaris10), in
the description of pull request 106 under "Cost": qwen2.5-0.5b Q4_K_M, pp512
1311.6 to 1299.9 t/s and tg256 161.6 to 152.2 t/s, "about 6% decode", prefill
unchanged; and in the same description and the changelog, zeta-2.1 Q4_K_M,
prefill 153 to 166 tok/s and decode 31.2 to 30.7 tok/s while glxgears held 130
fps where it had dropped to 7. Whether a desktop was running during the qwen2.5
re-measurement is not stated (glxgears implies one for the zeta run).

**Measured on Windows** (this slice; AMD proprietary driver, desktop running,
Qwen3-1.7B Q4_K_M, 6 interleaved rounds, the universal family against the
compute-only one):

| Device                  | tg128, t/s       | pp512, t/s           |
| ----------------------- | ---------------- | -------------------- |
| RX 7900 XTX             | 474.86 to 454.79 | 11611.42 to 11482.28 |
| Ryzen integrated, RDNA2 | 27.12 to 27.12   | 230.94 to 230.84     |

So on Windows it costs 4.2% of decode and 1.1% of prefill on the discrete card
and nothing measurable on the integrated one (B227 has the samples' ranges and
the load). **Desktop responsiveness was not measured there**: no objective
method was found in that session (no frame-time or compositor statistic was
taken while a forward ran), so on Windows the benefit the default is for is
unmeasured, and with a desktop running on Linux it is too, beyond the
contributor's glxgears figure.

**A confound in the Windows table, not controlled.** Which GPU drove the desktop
during those runs was not recorded. The owner's reading is that the integrated
GPU showed no cost because it was not driving any display, while the RX 7900
XTX, which was, showed 4.2%. If that is right, the cost follows the display, not
the device: it is one more reason the default is the compute-only queue (below),
and it means the tuning row that sets `queue.family = auto` for AMD integrated
GPUs on the proprietary driver rests on a measurement of an integrated GPU with
no display attached. To settle it on Windows: repeat the A/B with the monitor on
the integrated GPU's output, and again with the RX 7900 XTX driving nothing.

**Commands.** Build `infr` in release. The model is `Qwen3-1.7B-Q4_K_M.gguf`
from the `unsloth/Qwen3-1.7B-GGUF` repository (`$M` below). Pick the device with
`infr devices` and put its name in `$DEV` (`Vulkan0`, ...).

```sh
infr devices
infr tune show --dev $DEV    # the queue.family line: value, source, family picked

# The same setting from each of its three places (use any one):
#   config file:   [kernels.vulkan.tune]
#                  "queue.family" = "graphics"     # auto | compute | graphics
#   environment:   INFR_TUNE=queue.family=graphics
#   command line:  --set kernels.vulkan.tune.queue.family=graphics
infr tune show --dev $DEV --set kernels.vulkan.tune.queue.family=compute  | grep queue.family
infr tune show --dev $DEV --set kernels.vulkan.tune.queue.family=graphics | grep queue.family

# Interleave the two arms, at least 6 rounds, one process per sample.
for r in 1 2 3 4 5 6; do
  for q in graphics compute; do
    echo "round $r queue.family=$q"
    INFR_TUNE=queue.family=$q infr bench $M --dev $DEV -p 512 -n 0 -r 1 --json
    INFR_TUNE=queue.family=$q infr bench $M --dev $DEV -p 0 -n 128 -r 1 --json
  done
done
```

Run the whole loop twice: once headless (stop the display manager, or boot to a
text console, and check that no compositor is running) and once inside a desktop
session. Alternate which arm goes first between rounds if the samples drift.
Record with the numbers: the GPU, the Mesa version and the kernel version
(`uname -r`), whether a display server was running and which, whether the
measured GPU is the one driving the display, and the machine's load
(`top -b -n 2 -d 3`) before and after. A figure taken beside another build or
benchmark is labelled with that load or retaken.

**What else to check there.**

- **The family in use.** The startup log line
  `[infr] vulkan: queue.family = auto, picked queue family N (compute-only, timestampValidBits B): ...`
  names it and says why, and the `queue.family` line of `infr tune show` carries
  the same text after the value's source. Use `compute` for the measured arm,
  not `auto`, so that a fallback to the universal family is a warning and not a
  silent second `graphics` arm. `B` must not be 0: a compute-only family with no
  valid timestamp bits is refused with a warning that says so, and the universal
  family is used.
- **Timestamps on it.**
  `INFR_PROF_OPS=1 infr bench $M --dev $DEV -p 512 -n 0 -r 1` prints a per-op
  device report; its `device total` should be a little under the forward's wall
  time (512 divided by the reported t/s). On the RX 7900 XTX under Windows it
  was 42.1 ms of 47.0 ms on the compute-only family and 41.3 ms of 46.7 ms on
  the universal one. A total far off the wall time, or zero, means the segment
  timing the submit splitter and the work bound (`crate::work`) rely on is wrong
  on that family.
- **The work bound's times.** `crates/infr-vulkan/tests/work_split_parity.rs`
  and `first_forward_split.rs` pass on that family
  (`INFR_DEV=$DEV cargo nextest run --release --run-ignored all -E 'binary(work_split_parity) | binary(first_forward_split)'`;
  the first builds its config from the environment without `INFR_TUNE`, so it
  follows the device's default, which is `auto` on a discrete card).
- **The watchdog that applies.** amdgpu's `lockup_timeout` module parameter
  takes per-ring values (graphics, compute, and others; `modinfo amdgpu`
  describes the order, and `/sys/module/amdgpu/parameters/lockup_timeout` holds
  what is set; neither was read in the session that wrote this). The work bound
  assumes a submit is reset after about 2 s (`submit.budget_ms`, plan 12 section
  0d), which was measured on a graphics ring. Record the value that applies to
  the compute ring on that kernel. If it is longer, the bound is merely
  conservative there; if a reset is ever seen on the compute ring, the time the
  submit had run is the missing number (a failure lists the last forward
  segments with their GPU times).

**Measured on Linux, headless (2026-11-23).** infr at `a66f3d06`, RADV (Mesa
26.3.1, kernel 7.3.5), no display server running (Hyprland closed; the monitor
is cabled to the RX 7900 XTX and nothing drew to it), Qwen3-1.7B Q4_K_M, context
8192, 6 interleaved rounds with the first arm alternating, one process per
sample, `graphics` against `compute`. Each cell is the mean with the lowest and
highest sample. The machine's CPU was between 89.8% and 96.8% idle over 3 s
before each sample; no build, test or GPU work of ours was running, and what
used the rest was not identified.

| Device                                    | What, t/s | `graphics`                      | `compute`                       | Change |
| ----------------------------------------- | --------- | ------------------------------- | ------------------------------- | ------ |
| RX 7900 XTX (RADV NAVI31)                 | tg128     | 476.53 (476.07 to 477.22)       | 466.97 (466.56 to 467.36)       | -2.0%  |
| RX 7900 XTX (RADV NAVI31)                 | pp512     | 15155.32 (15106.65 to 15308.38) | 15122.66 (14917.06 to 15278.53) | -0.2%  |
| Ryzen integrated (RADV RAPHAEL_MENDOCINO) | tg128     | 28.84 (28.68 to 28.88)          | 28.86 (28.85 to 28.87)          | +0.1%  |
| Ryzen integrated (RADV RAPHAEL_MENDOCINO) | pp512     | 240.44 (240.03 to 240.61)       | 240.28 (240.02 to 240.36)       | -0.1%  |

So the owner's hypothesis, that the compute-only queue costs nothing headless,
does not hold on the RX 7900 XTX: decode is 2.0% slower with samples that do not
overlap, so part of the cost is the ring and not contention with a compositor.
Prefill and the integrated GPU show nothing beyond the samples' spread. On both
devices `compute` picked queue family 1 (compute-only, `timestampValidBits` 64),
read from the startup line. A profiled pp512 (`INFR_PROF_OPS=1`) reported a
device total of 31.8 ms on the universal family and 33.1 ms on the compute-only
one on the RX 7900 XTX, and 2.12 s on both on the integrated GPU, so timestamps
work on the compute-only family under RADV. `lockup_timeout` reads empty in
`/sys/module/amdgpu/parameters`, and `modinfo amdgpu` gives its default as 2000
ms with the format `[GFX,Compute,SDMA,Video]`: on this kernel one default covers
every ring.

**Still not measured on Linux:** the same loop inside a desktop session (the
script and the pinned binary are in `~/.cache/infr-queue-ab` on the Linux
machine, `run.sh desktop`), any measure of desktop responsiveness, and the work
bound's tests on the compute-only family. llama.cpp makes the same default
choice (`ggml-vulkan.cpp`, `ggml_vk_find_queue_family_index` avoiding the
graphics bit) and has the same escape hatch, `GGML_VK_ALLOW_GRAPHICS_QUEUE`,
commented "it can increase performance on RADV"; it does nothing else about the
cost.

**Decided (owner, 2026-11-22): no display detection.** The compute-only queue is
the default, and a user who wants the last of the throughput at the desktop's
expense sets `queue.family = graphics` (the model runs faster, the desktop may
lag while it does). `docs/config.md` states that trade-off. Detecting whether a
GPU drives a display, and switching queues on it, was **considered and
declined**: the setting already gives the choice to the one person who knows
which they want, and detection would be per-platform code (Vulkan has no direct
query; `VK_KHR_display`, DRM connector status and DXGI outputs were the
candidates, none researched) for a queue family that is chosen once, at device
creation. The measurements above are still worth taking, as figures for the
docs, not to decide a mechanism.

**Left to do, code, not started:** the integrated-GPU rule. By rule an
integrated GPU still defaults to `graphics` (the contributor's reason: under
RADV the compute ring's hang budget measured tighter on one), with a table row
setting `auto` for the AMD proprietary driver. The decision above reads as
`auto` everywhere a usable compute-only family exists. Before changing the rule
(`queue.family`'s default in `tuning/families.rs`, the snapshots and
`queue_family_follows_the_device_facts`), check the contributor's hang-budget
finding on an integrated GPU under RADV, since the work bound assumes 2 s.

### B230 — a second queue for transfers, as llama.cpp has: a candidate for the streaming work (2026-11-23)

**Tag:** vulkan · streaming · pager · submit · **Kind:** candidate, not
researched past the reference

infr submits everything on one queue. llama.cpp's Vulkan backend (checkout at
`57291f2`) also creates a transfer queue: `ggml_vk_find_queue_family_index`
looks for a family with the transfer bit that has neither the compute nor the
graphics bit, falls back to any other family than the compute one, and its
copies can then go out on that queue (`async_use_transfer_queue`, on by default
only for AMD discrete, non-GCN devices when the graphics queue is avoided, else
by `GGML_VK_ASYNC_USE_TRANSFER_QUEUE`).

**Where it could matter here.** Not decode: infr's pipeline is GPU-resident and
a decode step moves almost nothing between host and device, so a transfer queue
would not touch the compute-only queue's decode cost (B228). The candidate is
streaming (plans 05, 06 and 08): the pager's weight uploads on a queue of their
own while a forward runs on the compute queue, where today they share one submit
stream.

**What is known.** Read with `vulkaninfo` on the Linux machine under RADV (Mesa
26.3.1): neither the RX 7900 XTX nor the Ryzen integrated GPU has a
transfer-only family. Each has a universal family (1 queue), a compute and
transfer family (4 queues), video decode, video encode, and a sparse-binding
family with no timestamp bits. So on RADV the choices are a second queue of the
compute-only family, or the universal family, which is where llama.cpp's
fallback lands there.

**Not known, needed before building anything:** whether the AMD Windows driver,
NVIDIA or Intel expose a transfer-only family (not read); whether uploads on a
second queue overlap a running forward at all on one device, or the hardware
serialises them; what the cross-queue synchronisation costs (a buffer written on
one queue and read on another needs a semaphore or a fence between the two); and
how it sits with the work bound, which times and cuts one queue's submits. The
measurement that decides it: a streamed model that does not fit, uploads on one
queue against two, tokens per second and the pager's upload time.

### B222 — what the bounded GPU waits did not verify (2026-11-18)

**Tag:** vulkan · tests · **Kind:** coverage gaps

- **Dropped work is detected on one kind of submit only.** After a watchdog
  reset AMD's Windows driver signals the fence of work it threw away (plan 12
  section 0c). The backend catches that by the segment's closing timestamp
  (`recorder::account_segment`), which exists only on a forward segment from a
  non-persistent recorder (`Recorder::finish`, `PendingSegment::wait`) on a
  queue that writes timestamps (`segment_ts_period`). NOT checked:
  `RecordedCmd::replay` and `replay_n` (every replayed decode token and the
  chained decode: persistent recorders carry no timestamps), `one_shot`
  (uploads, downloads through staging, fills, probes), the weight staging ring,
  the tensor-parallel copies, and every submit on a queue without timestamps.
  After such a wait returns, the decode reads its logits from mapped memory, so
  a reset during a replay can hand back wrong logits; the lost device is
  reported only by the next submit (in every harness run of section 0c the next
  submit did report it). That error now also lists the last forward segments
  submitted, with their GPU times (`gpu_wait::RecentSegment`), so the forward
  that was cut can be told from it; it still cannot say that a decode replay was
  the one. A general detector was considered and not built: a completion marker
  the last command of every submit writes (the host clears a mapped word before
  the submit and checks it after the fence) would cover every path, at a
  transfer write and a barrier per submit, and on the persistent replays the
  value must change per replay, which the recorded command buffer cannot do
  without re-recording or a host-side reset per token. Worth doing with a decode
  A/B to show its cost; until then this is the gap.
- **The bounded wait costs decode about 0.4%; accepted for now (2026-11-19).** A
  returned error is a better failure than a hang or a killed process.
  Interleaved A/B against `main` (`086ebff0`), Qwen3-1.7B Q4_K_M, six rounds
  each, no other session on the machine: tg128 476.87 against 478.59 t/s on the
  RX 7900 XTX (branch lower in all six rounds) and 27.20 against 27.31 on the
  integrated GPU (lower in five of six); pp512 11910 against 11970 and 231.38
  against 231.66 (lower in four and six of six). About 7.5 µs per decode token
  on the XTX and about 80 to 150 µs on the integrated GPU (two A/Bs). **The
  cause is the fence wait itself on AMD's Windows driver, on both GPUs:** with
  the fence still on the work's own submit but waited by `vkQueueWaitIdle`
  instead, decode matched `main` (XTX 478.52 against 478.28, five rounds;
  integrated GPU 27.31 against 27.30 with the branch at 27.24, six rounds). On
  the XTX, waiting on the fence with no timeout (475.64), spinning on
  `vkGetFenceStatus` for up to 5 ms first (476.45) and waiting on a timeline
  semaphore (`vkWaitSemaphores`, no fence reset; 475.90) all cost the same as
  the bounded wait, so it is how late the driver reports a fence or semaphore
  signalled, not the timeout, the reset or a wake-up; the device-fault extension
  made no difference (475.76 without it). Those four variants ran on the XTX
  only. Why the integrated GPU pays ten to twenty times more per token is not
  known. Alternatives for later:
  - `vkQueueWaitIdle` for the decode replay only, bounded from outside: a
    watchdog thread that, past the bound, reports the stuck submit and ends the
    process (the waiting thread cannot be released).
  - One watchdog for every wait: each wait stores a deadline in an atomic before
    it blocks in `vkQueueWaitIdle` and clears it after, and one monitor thread
    wakes about once a second and ends the process with the report when a
    deadline has passed. The same failure mode as the first, cheaper and simpler
    than two wait styles.
  - Not an option: bounding only every Nth token, because the others stay
    unbounded.

  The experiment code is not in the tree.

- **The missing-closing-timestamp check has no automated test.** It fired on
  real hardware in the three production runs of plan 12 section 0c (`infr bench`
  on `Vulkan1`), never in the harness runs, where the next submit reported the
  lost device first; nothing forces a fence to signal without its work. The
  queries are reset from the host at creation (`hostQueryReset`, enabled where
  offered: both GPUs here), so a dropped command buffer leaves them reliably
  unavailable; without that feature they are reset inside the command buffer,
  and a dropped buffer leaves them undefined. The timeout path has a forced test
  (`gpu_wait::tests::gpu_wait_timeout_poisons_the_backend_and_drops_cleanly`; on
  2026-11-18 it failed with the bound replaced by `u64::MAX`, "a 1 ms bound must
  time out", and passed with it restored); the lost-device path only the real
  reproduction.
- **`infr serve` with a poisoned backend was read, not run.** A request fails
  with the error (500, or the stream's error frame) and so does every later one
  on that model, as a lost device already did; the server keeps running, so
  other models it hosts stay usable and an operator restarts it. `/health` does
  not report the poisoned backend.
- **Teardown after a timeout leaks the device** (`VulkanShared::drop` skips
  every destroy once poisoned). Process exit was checked on Windows (the
  reproduction exits and `infr devices` lists both GPUs afterwards); a
  long-lived embedder that recreates backends after a failure would keep the
  leaked one.
- **Metal's waits were not reviewed** for the same unbounded-wait pattern.

### B235 — `kv.paged` can never build at the clamp it is sized for (2026-11-24)

**Tag:** kv · vram · **Plan:** `docs/plan/14-kv-paged-boot.md`

`ParallelSeam::init_slots` builds slot 0's linear KV, resets it, then builds the
page pool on top, so boot needs `n_slots + 1` caches while the clamp priced
`n_slots`. On kikuri (RX 580 8 GB, Qwen3.5-4B Q4_K_M, q8): `--parallel 2` clamps
to 103559/slot and the pool is refused (215.7 MiB requested + 7.62 GiB in use >
7.75 GiB); slots stay linear. One slot with q8 builds at `--ctx 81920` and is
refused at 98304 — the boundary is the transient, not the pool. Fix: allocate
the pool in the cold KV init and skip it entirely at `n_slots == 1`.
