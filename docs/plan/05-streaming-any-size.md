# Phase 05 — streaming for any model on any hardware

**Goal:** a model of any size loads and generates on any supported machine,
whatever its VRAM and RAM: disk → DRAM → VRAM, each tier holding what fits and
streaming the rest. Slow is acceptable where the physics say so, and the banner
says how slow before the wait. "Cannot load" is not acceptable.

The design, the measurements and what shipped are in
[`../disk-streaming-plan.md`](../disk-streaming-plan.md) (phases 0 to 3, the
concurrent reader, the admission rule, auto-sizing); open residue is in
`backlog.md` B35 and B36. This file is the list of what still stands between
that and the goal, in build order. Each item names where the gap was read; an
item marked **to confirm** was taken from a doc and not re-read in code on
2026-11-13.

## A. Gaps that make a model fail to load or run

These are the ones that break the goal outright, so they come first.

1. **Only llama-shaped projections stream.** The dense planner in `seam/mod.rs`
   (`dense_plan`) enumerates each layer's `attn_q/k/v`, `attn_output`,
   `ffn_gate`/`ffn_up` and `ffn_down`, whatever their dtype (F16 and F32
   included since 2026-11-15). Any other per-layer projection stays resident
   under any budget: LFM3's `shortconv.in_proj`/`out_proj` (read on LFM3-1.7B
   F16: 92 groups stream, the 13 conv layers' two projections, about 13% of its
   parameters, do not), and, **to confirm** by the same count, qwen35's DeltaNet
   projections, gemma4 E2B's per-layer projections and the MLA families'
   projections. Needs those groups in the enumeration (and in
   `fuse_*_decision`'s upload-order contract), each with its streamed dispatch
   checked to take an arena address.
2. **Weights rewritten at load cannot be re-read from the file:** the qwen2 NEOX
   q/k row permute and the BitNet `I2S` → f16 dequant (§3.2). Options: apply the
   rewrite per block on the read path (a `BlockIo` wrapper that transforms after
   reading), or move the permute into the kernel's indexing. The first keeps
   kernels untouched and is the default choice.
3. **One tensor larger than a tier's whole budget.** A block is a whole weight
   group, and the sizing floor is one batch of blocks. An lm_head or embedding
   larger than the arena, or than the staging ring, has no path today. Needs
   sub-block (row-range) streaming for the oversized tensors: the chunked GEMV
   dispatch that already exists for weights past the 32-bit element limit is the
   shape to extend. **To confirm** what happens today (error text, or a panic).
4. **`serve --parallel N` under a tight budget.** The host arena's floor is N
   concurrent working sets and one is priced (B35). Either price N at load and
   reject by name, or serialise paged forwards with one permit and say so in the
   banner. The CPU interpreter holds a whole op's pins across the op, so it is
   the exposed backend.
5. **Multi-GPU and MTP bypass the tiers** (`tensor_parallel_binder`,
   `expert_parallel_binder`, `pipeline_binder`; MTP's second weight set). A
   model that needs two GPUs and still does not fit has no path. Last in this
   group: it needs a per-device budget story and MTP is parked.
6. **Metal has no pager.** Phase 4 of the design, blocked on a decision
   (`disk-streaming-plan.md` §7 item 4): no Apple hardware is available, so it
   could only be written unverified. Not started until that is decided.
7. **The KV cache must also fit.** `INFR_KV_OVERFLOW` spills KV to host; what
   happens when KV plus the activation reserve exceed VRAM on a small card at a
   long context was not read. **To confirm**, and add to the placement ladder if
   it is a hard failure.

## B. Unified memory, which can now be verified

`disk-streaming-plan.md` §4.0 says the two-tier unified shape (disk →
GPU-visible RAM, `HostPager::stream_only`) was never run on unified hardware.
The Windows dev box's integrated GPU is one. In order:

1. Confirm `DeviceCaps::unified_memory` is set for it and that the arena is
   sized from shared RAM sensibly (`infr run --dev VulkanN` with
   `INFR_PAGER_STATS`, on a model larger than a capped `INFR_CACHE`).
2. Run `dense_tier_parity` and the `gpu_seam_dense_stream_*` tests there without
   `INFR_DRAM_BYPASS` (they need Qwen3-1.7B in the cache).
3. Streamed prefill on that device takes the per-row GEMV today (B155); the
   tiled non-coopmat route is in progress as its own slice.

## C. Throughput once it runs

Ordered by what the existing measurements say is left on the table; each needs a
measurement in a quiet window before and after.

1. **Prefetch.** Not built. Dense order is known for the whole pass; MoE on the
   CPU knows the next layer's experts at the router. B35 records why it was
   deferred (the tier was I/O-bound, not late); re-measure now that the
   concurrent reader landed.
2. **Windows read path.** `FileBlockIo` fans a block out over concurrent
   positioned reads, and on Windows a handle not opened for overlapped I/O has
   concurrent reads serialised by the kernel (B35). Open the file with
   `FILE_FLAG_OVERLAPPED` (through `infr-plat`) and measure against the current
   path on the Windows box's NVMe.
3. **Read-ahead at load on Windows** (`PrefetchVirtualMemory`, B75): only the
   mmap path; measure on a cold cache with a multi-GB model.
4. **Double caching** (`F_NOCACHE` / `posix_fadvise(DONTNEED)` / unbuffered
   reads on Windows): the page cache holds a second copy of what the arena
   holds, which halves effective DRAM on a model far larger than RAM.
5. **Eviction cost and policy:** its own phase,
   [06-pager-eviction.md](06-pager-eviction.md). That includes the per-token
   cost of a paged decode step (re-recording, per-layer router drains), measured
   there, with what is left as its step 5.
6. **Direct-to-ring reads** for blocks that are never admitted to DRAM: only
   meaningful once a planned partition exists (`TierPlan`, §3.6).
7. **Prefill order on the CPU's host arena.** DeepSeek V4 prefills layer-major
   there since 2026-11-17, reading each routed expert once per prompt (each once
   by the pager trace on the real file); every other family still sweeps the
   arena once per chunk. A win at the default chunk height (a 1443-token prompt:
   195 to 207 s against more than 235 s chunk-major); slower in 32-row chunks,
   which therefore keep chunk-major, likely because buffered reads let the OS
   file cache serve chunk-major's re-reads there (item 4 would settle it).
   `backlog.md` § B-CPU-EXPERT-PAGING.

## D. Safety

- **The file-change check runs on every block read.** `FileBlockIo::read_block`
  re-stats each shard it read, after the read, against the stamp taken when the
  model was mapped, and fails the read (and with it the request) by name when
  the length or modification time moved. It reads through descriptors reopened
  from the mapping's own, so a file renamed into place is not a change. Its cost
  and what it leaves are backlog B35.

## Verification for the whole phase

- An end-to-end test per gap in group A that fails to load before and passes
  after, on a small model with the budget capped (`INFR_CACHE`,
  `INFR_DRAM_CACHE`) so the tier engages on hardware that would otherwise hold
  the model.
- Token-identical output against the resident path for dense models, the loose
  comparison for MoE, as the existing `*_matches_resident` tests do.
- A matrix run at the end: one dense and one MoE model, each on the CPU, the
  integrated GPU and the discrete GPU, at three budgets (fits; DRAM only;
  neither). Every cell either generates or names the knob that would let it.
