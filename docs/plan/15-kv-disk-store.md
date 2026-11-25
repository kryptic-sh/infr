# 15 — KV cache tiers: our own disk store

**Why.** The KV cache has two tiers today, VRAM and system RAM (the spill), and
both hold only a live window. Nothing keeps rows that no request is reading: an
idle prefix or a parked conversation is recomputed by prefill. The owner decided
2026-11-25 to build infr's own disk store for those rows, with no database
dependency, designed for speed and for how infr already moves data. Backlog
B-KV-TIERS carries the tier picture this plan fills in; issue #109 is the
outside proposal it answers.

Nothing here is built or measured yet. Every figure marked **guess** is one.

## Decided by the owner (2026-11-25)

- **The goal is the full disk → RAM → VRAM tier for the KV cache**, the way
  weights already stream. Idle prefix blocks go to disk first. The live window
  stays VRAM then RAM, and only when both are full does its coldest part page to
  disk, so any context length runs on any hardware; that rung is slow by nature,
  is logged loudly, and is built last (slice S13).
- **One global disk budget**, least recently used model first: `kv.disk_size`
  bounds the whole cache directory, and when it is full the store of the model
  used longest ago shrinks or goes first (only stores no running process holds).
- **Same setup only.** A block is reused only by the backend, GPU model, driver
  version, kernel choices and KV format that wrote it. The key names the GPU
  model and driver, not the individual card, so identical cards in one machine
  share blocks; other hardware misses, predictably.
- **Off by default** until measured; the owner picks the default from the
  numbers.
- **System RAM is one ledger with a fixed order:** live KV spill, then the
  weights' host tier, then cached KV blocks ("Sharing system RAM" below). The
  first version of the disk store uses the page cache as its RAM step; **our own
  KV block cache is the end state and a committed slice** (S10), built after the
  pressure response (R3) and before the live-window disk rung (S13), which needs
  it.
- **"Idle" means whole conversations and dead window rows:** parked
  conversations, stored prefix blocks, and rows a sliding-window layer has moved
  past. Any part of an active full-attention window on disk is the slow last
  rung only.
- **Swapping active conversations under load** (pausing the one that has waited
  longest for its turn and giving its slot to a queued request) is deferred: it
  buys responsiveness, not throughput, and reuses capture and restore. Decide it
  from server measurements of queue wait against swap cost (S14).

## The tiers

| Tier            | Holds                                                                                                                                      | State                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| VRAM            | the live window of each slot                                                                                                               | shipped                                                         |
| System RAM      | the part of a live window that VRAM cannot hold (`kv_spill_split`, `kv_spill_reserve`, `refuse_unhostable_kv`, `reclamp_ctx_to_live_room`) | shipped; arms only with a live memory budget (`kv_spill_armed`) |
| Slot reuse      | a prefix already in another slot (`ParallelSeam::checkout`, `SlotPool::pick`, `SeamKv::seed_from`)                                         | shipped; device to device                                       |
| Disk            | prefix blocks nothing reads every token                                                                                                    | this plan                                                       |
| RAM block cache | disk blocks kept warm in host memory                                                                                                       | slice S10, committed                                            |
| Remote          | —                                                                                                                                          | not planned                                                     |

Rules that hold across them:

- **Disk holds idle rows first and a live window last.** Attention reads every
  row of the window every step; a store behind it is a bandwidth floor (the
  fork's own figures: 5 to 13 times slower). So the live window reaches disk
  only when VRAM and RAM are both full (S13). `kv_spill_split` and the spill
  reserve are unchanged until then, and the disk tier has its own byte budget.
- **One host budget** ("Sharing system RAM" below). Anything this plan keeps in
  host memory (staging, the later RAM block cache) is charged where
  `refuse_unhostable_kv` charges the weights' host tier.
- **A block is reused only at the position it was computed at.** No re-roping,
  so backlog B-MLA-KV-SHIFT does not apply.

## Sharing system RAM

Weights already stream disk → RAM → VRAM. With this plan system RAM serves three
consumers, and what a gigabyte is worth to each sets the order:

| Consumer                                  | What a gigabyte saves                                                                                                                                  | Class                                   |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| Live KV spill                             | without it the context is refused, clamped, or read from disk every token                                                                              | mandatory, sized at load, never shrinks |
| Weights' host tier (streamed models only) | about a gigabyte of disk reads every token for a dense model; measured on Qwen3-30B-A3B, `paging.dram=12g` took decode from 11.7 to 19.8 t/s (plan 06) | elastic                                 |
| Cached KV blocks                          | the gap between a disk read and a RAM read, once per request                                                                                           | elastic, lowest                         |

Per-token value against per-request value differ by about the tokens in a
request, so the order between classes is fixed, not measured at run time.

- **One ledger.** A `HostLedger` in `infr-core` replaces the separate arithmetic
  in `hostmem::spare_bytes`, `streaming_arena_plan`, `host_tier_bytes` and
  `refuse_unhostable_kv`: mandatory entries first, the elastic ones share what
  is left.
- **The page cache is the first RAM tier for KV blocks.** Restores read
  buffered, so a block read back stays in memory the OS can reclaim under
  pressure and that survives a restart; captures write uncached (direct I/O, or
  `RWF_DONTCACHE` where the kernel has it, else a buffered write and
  `posix_fadvise(DONTNEED)`) so they do not push the weight mapping out. A block
  cache of our own (S10) follows as the end state: slots we manage, direct reads
  into them, trunk-keeping eviction, and the same pool holds a live window's
  paged rows for S13.
- **That needs the weights' streamed reads off the page cache.** `HostPager`
  fills its arena through buffered reads today, so every block may be held twice
  (plan 05 item C4) and a streamed dense model pushes gigabytes per token
  through the page cache, which would flush the KV blocks (inferred, not
  measured). Direct reads for the weight tier fix both.
- **The weight tier gives memory back.** Its arena is one allocation that cannot
  shrink today. It becomes an anonymous mapping whose slots can be retired and
  released, and a probe at token boundaries (not more than about once a second)
  reads the OS pressure signal: PSI on Linux, the memory resource notification
  on Windows, the pressure level on macOS, all reachable with the dependencies
  already present. Give back in reverse priority; grow back only after a
  sustained quiet period.
- **Unified memory:** the weight tier stays stream-only and the spill stays
  unarmed there, as today; a restore reads straight into the mapped KV buffer.

**Probable bug found on the way, not confirmed by a run.** The seam sizes the
weights' host tier before pricing the KV spill, and the arena is untouched
memory the live probe does not see yet, so with an automatic arena the spill's
room comes out as nothing and a spilled KV cache is refused at load whenever the
weights stream. That ranks weights above the live window, the reverse of the
order above. Slice R0 confirms it with a run (a model forced to stream with
`kv.overflow` on) and fixes the order.

## What is stored

Prefix extents only: blocks of `kv.disk_block_tokens` tokens (default 128),
chained by content hash. A parked conversation is the chain of its full blocks;
its partial tail is prefilled again on resume. Whole-session blobs were
considered and declined: no sharing between conversations, and a second index.

- **One extent holds every layer**, as [layer, side, plane] segments in an order
  fixed by the model's geometry, so a block is one aligned read. A per-layer
  table in the header leaves room for partial reads later.
- **Sliding-window layers store logical rows**, never physical ring rows, so the
  ring geometry (`kv_rows_at`) may differ between capture and restore. A header
  flag says whether the block's window-layer rows were still resident at
  capture; a restore to position P needs it on every block covering the window
  before P and is cut back otherwise. After a restore `written_end` is P.
- **MLA rows** are per position and fit as they are.
- **KV-sharing layers** (`n_layer_kv_from_start`): only the layers that own rows
  are stored.
- **Out of scope at first, refused by name and never silently:** models with
  unrewindable state (`has_unrewindable_state`: DeltaNet and shortconv state,
  the DeepSeek V4 compressed caches), the diffusion denoise path, and the
  multi-GPU wrappers. The store reports `refused: <reason>` in the log and in
  `/status`; the run itself never fails because of the store. Recurrent state
  comes back in slice S9.
- **To check before S6:** whether an MTP draft head keeps KV buffers of its own.

## Identity

A namespace is one directory, `<kv.disk_dir or cache_dir()/kv>/<hash of key>/`.
The full key bytes are stored and compared verbatim, the way `KernelCache` does;
a mismatch wipes the namespace. The key holds:

1. **Store format:** magic, version, block size, payload layout (packed, or
   planar Q8_0 as Vulkan keeps it).
2. **Weight content**, per shard:
   - a file inside the Hugging Face cache is a blob named by the sha256 that
     `infr-hub` verified at download; use that with the shard's `FileStamp`;
   - any other file gets a tree hash (sha256 over large chunks in parallel, then
     over the digests), memoised by path, inode and `FileStamp`; the store stays
     off until it is done;
   - `kv.disk_identity = "stamp"` opts into metadata, tensor table and
     `FileStamp` only, documented as weak (backlog B30: a same-length rewrite
     with the mtime restored passes it).
3. **Config that changes K or V:** geometry, rope and YaRN fields, window
   layers, `n_layer_kv_from_start`.
4. **Numerics:** a new `Backend::kv_numerics_key`; `None` refuses the backend.
   Vulkan: device and driver identity, the shader set fingerprint, every
   resolved `kernels.*` path and tuning parameter. CPU: crate version and
   `kernels.cpu.*`. Metal: `None` until checked on Apple hardware. Serving one
   backend's or device's rows to another is off, because GPUs are only held
   within tolerance of the CPU (plan 11). The device part names the GPU model
   and driver, never the card instance. A model split across GPUs adds the split
   layout to the key, since it decides which kernels compute which rows (the
   multi-GPU wrappers are out of scope until then).
5. **KV formats** and each stored layer's row widths.

Not in the key: template and BOS (the chain is over token ids), context length
and ring size (rows are logical), ubatch.

**Chain id:** sha256 over the namespace digest, the parent's id and the block's
token ids, truncated to 128 bits. `sha2` is already a workspace dependency. A
crafted prompt must not be able to collide into another tenant's rows under
`infr serve`; the block's own tokens are stored in its header and compared too.

**Caveat to document.** Restored rows are the rows the capturing session
computed. On a GPU they are within tolerance of, not equal to, a fresh prefill
with other chunk boundaries; `seed_from` already has this property. On the CPU
they should be equal (decode and prefill rows must agree); S6 verifies it.

## On disk

One sparse slot file per namespace with fixed-size slots: every block of a
namespace has the same size, so allocation is O(1) with no fragmentation. One
file per block (inode churn, slow on Windows) and size classes (a problem that
does not exist here) were declined.

- **Files:** `key`, `slots.bin`, `lock`.
- **Slot:** a small header then the payload, both aligned for direct I/O. The
  header holds magic, slot generation, chain id, parent id, block index, the
  block's token ids, flags, payload length, one checksum per payload piece and a
  header checksum.
- **Checksum:** XXH64 written in tree against its published vectors (FNV-1a is
  byte at a time and weak against bit rot; sha256 over payloads would cost
  seconds on a large restore, **guess**). Per-piece checksums let the read
  threads verify in parallel. The threat on payloads is accident, not attack:
  whoever can write the cache directory can write the model file.
- **Index:** rebuilt at open from the slot headers; a block is valid when its
  header checks and its parent is resident. No journal. A snapshot at clean
  shutdown only if S2 measures the scan as slow.
- **Direct I/O** (new in `infr-plat`, std and libc/windows only): `O_DIRECT`
  with the alignment from `statx`, `F_NOCACHE` on macOS,
  `FILE_FLAG_NO_BUFFERING` on Windows, and a buffered fallback where the
  filesystem refuses. The point is that capture writes must not push the weight
  mapping out of the page cache.
- **Crash model:** it is a cache; nothing is fsynced. Payload first, header
  second; a torn write is a checksum miss for that block and its descendants.
- **Processes:** one process owns a namespace through `FileLock`; a second logs
  once and runs without the store. Sharing between processes is declined for
  now.

## Read path

1. Compute chain ids over the prompt, stop at the first miss, apply the window
   cut. Restore only when it saves at least `kv.disk_min_tokens` beyond what the
   slot already holds.
2. Reader threads (the pager's `FillReader` pattern, `IO_FANOUT` pieces per
   block) read with direct I/O **into mapped staging memory**, verify, and one
   command buffer copies each segment into the KV buffers; a ring layer splits
   at the wrap and planar Q8_0 is two regions. Reads of the next block overlap
   the copy of the current one.
3. Where the KV buffer is itself mapped (integrated GPUs, Metal, CPU, a spilled
   Vulkan buffer) the read goes **straight into the KV buffer**.
4. A checksum failure midway keeps the blocks already restored and frees the bad
   one with its descendants.

Goal: no host heap copy between the disk and the device.

## Write path

- **When:** at the end of a request, while the slot is still held. Rows of
  blocks not yet stored are copied to a readback staging ring under a fence; the
  slot returns to the pool carrying that fence, and `checkout`/`pick` waits on
  it before handing the slot out. Nothing is added to decode.
- **Writer thread:** checksum, payload, header, publish.
- **Backpressure:** with no room in the ring, keep the earliest blocks (the
  trunk is worth most) and count the rest as dropped. A capture never blocks a
  request.
- **Window layers:** rows already overwritten are captured as absent; S8 adds
  capture during prefill so early blocks keep them.

## Eviction

One intrusive LRU over slot indices (the list in `pager/lru.rs`, shared), every
operation O(1). Chain order is an invariant, not a scan: a touch links a chain
leaf first and root last, so a parent is never older than its children and plain
LRU takes leaves before trunks. A child count per slot guards it; pinned slots
are never taken. The global budget is divided among the open stores; a store's
slot count can shrink when another model needs room.

## Config and reporting

`kv.overflow*` and `kv.slots` are untouched.

| Key                    | Default                    | Env                |
| ---------------------- | -------------------------- | ------------------ |
| `kv.disk_size`         | off                        | `INFR_KV_DISK`     |
| `kv.disk_dir`          | `cache_dir()/kv`           | `INFR_KV_DISK_DIR` |
| `kv.disk_block_tokens` | 128                        | —                  |
| `kv.disk_min_tokens`   | 256 (**guess**, set by S6) | —                  |
| `kv.disk_staging`      | automatic                  | —                  |
| `kv.disk_identity`     | `content`                  | —                  |

`/status`, metrics and `infr tune show` report state (on, refused with the
reason, locked by which process), directory, used and budget bytes, hits and
misses by reason, restored tokens, restore time, captured and dropped blocks.

## The store interface

The store's own interface takes the destination memory from the caller (staging
or a mapped KV buffer) and fills it; it never returns owned host vectors. An
adapter to another interface that moves host bytes (the fork's `KvStore`) is
possible on top and is slice S12, not the design centre.

## Slices

Each ships alone through coder then reviewer.

| #   | Slice                                                                                                                                                                                         | Size | Gate                                                                                                  |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ----------------------------------------------------------------------------------------------------- |
| S0  | Measure only: direct I/O into Vulkan-mapped staging and which memory type; upload rate from it; sha256 and XXH64 throughput; prefill baseline at 2k, 8k, 32k tokens; Windows unbuffered reads | S    | figures in this file                                                                                  |
| S1  | `infr-plat` direct I/O, alignment query, aligned buffers, buffered fallback                                                                                                                   | S–M  | tests on all three CI systems                                                                         |
| S2  | `infr-core` store: format, slots, scan and index, chain LRU, lock, XXH64                                                                                                                      | M    | unit tests below                                                                                      |
| S3  | Identity: blob sha detection, tree hash with memo, stamp tier, `kv_numerics_key`, config hash                                                                                                 | M    | one weight byte changed is a miss                                                                     |
| S4  | Row mapping (linear, ring, planar Q8_0) and offset upload/download plus the mapped and staged primitives on CPU, Vulkan, Metal                                                                | M    | span tests; byte round trip per KV format                                                             |
| S5  | Capture path, writer thread, backpressure, refusals, config keys                                                                                                                              | M–L  | captured bytes equal device rows; decode t/s unchanged                                                |
| S6  | Restore path, window cut, partial restore, choosing between continuation, slot seed and disk                                                                                                  | L    | round trips; tolerance rows; restore against prefill table                                            |
| S7  | `/status`, metrics, `tune show`                                                                                                                                                               | S    | serialisation test                                                                                    |
| S8  | Capture during prefill for window models                                                                                                                                                      | M    | a shared gemma prefix restores                                                                        |
| S9  | Recurrent snapshot at a block boundary                                                                                                                                                        | M–L  | bit-identical continuation after restore                                                              |
| S10 | RAM block cache under the host budget                                                                                                                                                         | M    | after R3, before S13                                                                                  |
| S11 | Read-ahead while queued; restore overlapping the suffix prefill                                                                                                                               | L    | measure first                                                                                         |
| S13 | Live window paging to disk when VRAM and RAM are full: page map for the cold part of the window, eviction between tiers, loud log, refusals for layouts it cannot page                        | L    | a context past VRAM plus RAM runs and matches the in-memory run within tolerance; decode t/s recorded |
| S14 | Swap an active conversation out under load (optional)                                                                                                                                         | M    | decided from queue wait against swap cost on a loaded server                                          |
| S12 | Adapter to a host-bytes store interface                                                                                                                                                       | S    | only if one is adopted                                                                                |

### RAM slices

These touch the weight tier as much as the KV cache and run alongside the table
above.

| #   | Slice                                                                                       | Gate                                                          |
| --- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| R0  | Price the live KV spill before the weights' host tier; confirm the refusal with a run first | an automatic arena plus a spill that fits loads               |
| R1  | `HostLedger`: mandatory and elastic classes, replacing the separate calculations            | unit tests on the arithmetic                                  |
| R2  | Direct reads for the weight host tier (with S1)                                             | t/s and rereads against buffered at three `paging.dram` sizes |
| R3  | Pressure probe and a shrinkable weight arena                                                | time to give memory back under a memory hog; t/s after        |
| R4  | Split between the elastic pair from eviction counters, only if both are seen competing      | measure first                                                 |

Order: R0, R1, S0 to S3, R2, S4 to S7, R3, S10, then the rest, with S13 after
S10.

## Tests that can fail

- **Store (CI):** round trip; one flipped bit in a payload piece, the header or
  a token is rejected; a payload with no header is a free slot after reopen; a
  key mismatch wipes; orphans freed; a second open runs disabled; a property
  test that no block with resident children is evicted; XXH64 vectors.
- **CPU seam (CI, synthetic models):** capture then restore gives equal bytes
  for every KV format; next-token logits equal the capturing session's
  continuation bit for bit, and a fresh prefill's; a window model wrapped and
  not; an MLA model; one weight byte changed is another namespace; the refused
  families report `refused`.
- **GPU (ignored, both local GPUs, one at a time):** the round trip with planar
  Q8_0, f16 and turbo; restored logits equal the capturing session's on the same
  device and sit inside the tolerance table against a fresh prefill; a recycled
  slot resumed from disk.

## Measurements

Quiet window, alternated runs, at least a second between processes (backlog
B-PP-SLOW-RUNS):

- restore against prefill at 2k, 8k and 32k tokens, small and 8B-class models,
  f16 and q8_0 KV, one window model, both GPUs, cold and warm page cache;
- decode t/s and time to first token with capture on and off, and with one slot
  capturing while another decodes.

To confirm the RAM order: t/s against `paging.dram` for a dense and a MoE model;
the hit rate of KV blocks in the page cache while weights stream, buffered
against direct weight reads; whether capture writes shrink the weight mapping's
resident bytes; direct and buffered read rates on Windows.

## Open

- The shared usage record the global budget needs: where it lives, and how two
  processes on different namespaces update it (S2).
- Direct I/O into GPU-mapped memory and the memory type to use (S0).
- Whether staging must be priced into the device reserve on unified memory;
  `reclamp_ctx_to_live_room` does not price it today (S5).
- Whether the next forward's first barrier would cover the capture copy without
  the fence wait; the wait is kept until known (S5).
- CPU chunk invariance of KV rows (S6).
