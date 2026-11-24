# Plan 18: KV tiers behind traits

Status: shipped. P1–P5 built the trait, the tiers and parked slots; P6 replaced the fixed RAM/disk/Redis knobs
with a list of pluggable stores and taught the row copies Vulkan's planar Q8_0 layout.

## Problem

KV rows can live in four places today, and each move between them is hand-written:

- live slot rows in VRAM, laid out linear, as a sliding-window ring, or in pool pages;
- the host-RAM prefix store (`KvPrefixStore`, `kv.store_mb`);
- the cold tier (`KvColdTier`: journal file or Redis), which only mirrors the RAM store.

`fill_from_device`, `restore`, `read_rows`, `write_rows`, `write_ring_block`, `import_linear` and
`copy_tail` each re-derive where a logical row sits. The cold tier is not a spill tier: the RAM
LRU drops a block and the drop-time rewrite deletes it from Redis too, so Redis never holds more
than the RAM budget. An idle slot that loses its pages to a new request loses its reply rows, because
capture only keeps the prompt.

## Constraint

A running slot stays resident in VRAM. Attention reads every row of the window on every step, so
a row outside VRAM would be fetched once per token. Tiers below VRAM hold idle data: stored
prefixes and parked conversations.

## Design

One general trait sits at the bottom. Everything above it either implements it or wraps a
value that does, so the layers compose.

**Base trait.** `KvStore` holds extents: runs of rows for every layer, addressed by an extent
id. It has `get(id, rows, &mut KvRows)`, `put(id, first, &KvRows)`, `remove(id)` and
`contains(id)`. `KvRows` is per-layer K and V bytes plus an optional recurrent snapshot, which
is what `Block` holds today. A call moves every layer at once, so Redis pays one round trip per
extent.

**Leaf stores** implement `KvStore` directly:

- `DeviceKv`: GPU buffers, one extent per slot, with a `RowLayout` (linear, ring, page map)
  resolving rows to buffer offsets;
- `RedbKv`: one redb database, in a file (`kv.store_path`) or in redb's in-memory backend. A
  B-tree reuses the pages a deleted extent freed, which suits a cache that evicts all the time;
  commits use `Durability::None`, because a lost extent is only a cache miss, and `Drop` flushes
  once;
- `RedisKv`: one key per extent on Redis or Valkey;
- `NoStore`: the store behind a disabled cache, which holds nothing.

**Capability traits** extend it where a store can do more:

- `BudgetedStore: KvStore` reports bytes used and evicts the least-recently-used extent,
  which is what the tiering below needs.

A `DeviceStore`/`copy` pair was proposed for the same slot and is not built: `DeviceKv` is the only
device store, and the three prefix copies that could have used it keep their own because `seed_from`
must go through Vulkan's planar-aware `copy_kv_prefix`, while `import_linear` and `copy_tail` move
pool pages and have no `SeamKv` to wrap.

**Combinators** wrap stores and implement `KvStore` themselves:

- `Tiered<Fast, Slow>`: reads fall through and promote, evictions from `Fast` demote to `Slow`
  with their index entry, so a list of stores folds into one;
- `Bounded<S: BudgetedStore>`: the last tier, which drops its own LRU instead of demoting, so a
  cache that ends here stays inside its tier's `size`;
- `BlockCache<S: KvStore>`: the prefix cache, a token index over content-addressed extents
  (FNV chain id) in any store. The index entry (tokens, snapshot flag, bytes) is side metadata on
  the store itself (`KvStore::meta`), so startup lists extent ids and reads metas only.

**The store list is config.** `[[kv.tiers]]` is one entry per store, fastest first, each with a
`backend` and its own `size`; `redb` takes a `path` (a file) or none (memory), `redis` takes a
`url`. The builder folds the list innermost-first into `Tiered`, `Bounded` at the bottom, so a new
backend is one `KvStore` impl and one arm in `open_tier`. Every entry except the last has to be
able to evict, which is why a `redis` tier that is not last is dropped with a warning: the server
owns its own `maxmemory`.

`Paged<S>` was proposed here and is not built: its only possible `S` is the one page pool, whose
buffers the attention kernels bind directly, so the wrapper would have a single implementor and no
second consumer. `DeviceKv` already resolves a paged slot's rows through its page map, which is all
the store side needs of it.

Blocks stay 128 rows and pages 512. Extent length is per store, so neither unit changes. A block and
the recurrent snapshot at its end are one extent, so a tier evicts the rows and their state
together. Config picks the tiers at runtime, so they sit behind `Box<dyn KvStore>`, and the box
implements the trait too. Attention never goes through the trait: kernels keep binding the
device buffers and page table directly, so the trait costs nothing per token.

**Parked slots.** When the pool runs out of pages, the least-recently-used idle slot captures
all of its rows, prompt and reply, into the store and releases its pages. Its conversation
resumes through the ordinary prefix restore. This replaces the `kv_pool_exhausted` 400 for
every case except a single request larger than the pool.

**Ring windows.** A prefix longer than a sliding-window ring is captured per block, and the block
that straddles the window's start keeps the rows that are still resident. A restore refuses when
the blocks cannot fill the destination's window, so a ring never claims rows attention would read
from a hole.

## Phases

- **P1** (done, `959b0864`) `RowLayout` and the generic row copies; rewrite the store and pool callers. No
  behavior change. Row ops refuse a format whose rows do not pack back to back (`row_bytes`). Test: ring and page span splits, a CPU round trip through a wrapped ring, a
  linear-to-pages copy.
- **P2** (done, `f7ef217a`) `KvStore`, `KvRows`, `DeviceKv` and `RamKv`; `BlockCache` replaces
  `KvPrefixStore`. Fixed with it: a wrapped ring kept no rows for the block straddling its window,
  so a restore left the rows attention reads unwritten, and residency came from `cached.len()`
  rather than the rows a chain wrote past it. The `DeviceStore`/`copy` pair was dropped once
  `Paged<S>` was: it had no second implementor.
- **P3** (done, `7d471b44`) `BudgetedStore`, `Tiered`, `Bounded`, `DiskKv` (redb, `Durability::None`
  plus one flush on drop) and `RedisKv` replace `KvColdTier` and the journal file; the index moves
  into per-extent metas. Test: demote and promote through two stores; a block evicted from RAM
  restores from the disk store, also across a restart; a damaged extent is a miss, not a failed
  request.
- **P4** (done, `39d89c19`) capture the whole slot after a generation, not only the prompt. A
  recurrent model keeps its prompt only: it resumes from a snapshot, and prefill takes those at
  prompt boundaries.
- **P5** `Paged<S>` is dropped (see above). Parked slots ship as they are: the retry loop around
  restore and the runner, and `park_idle` capturing the victim before `reset`. Test: two
  conversations in a one-conversation pool both resume with `cached_tokens > 0`, plus the
  panic-safety of the park reservation.
- **P6** (done, `06246237`) the store list: `[[kv.tiers]]` entries replace `kv.store_mb` /
  `kv.store_path` / `kv.store_url`, `DiskKv` becomes `RedbKv` with a file or in-memory database, and
  `RamKv` goes — one store with a location instead of a RAM tier in front of a disk tier. Fixed
  with it: Vulkan's planar Q8_0 cache (codes, then the scales at the buffer's element capacity) is
  addressed by `SideLayout` in the row copies, so capture, restore, a fork's tail copy and page
  copy-on-write all work on it, and the page pool no longer refuses it. Also fixed: a redb `put`
  recorded an empty index meta when it had none, which a later demotion copied over the real one, so
  a restart dropped those blocks. Follow-ups with the review: a promotion now carries the meta up
  with the rows, an empty redb meta reads as absent, a tier's `url`/`path`/`size` are checked
  against its backend instead of ignored, credentials are stripped from a logged `url`, and a redb
  tier without a `size` keeps 1 GiB, which is what `--set kv.tiers=redb` builds.

## Measures

- Restore cost per tier, on the RX 580 with Qwen2.5-0.5B: 2k tokens take 43 ms from an in-memory
  redb, a redb file on NVMe or loopback Redis, against 856-864 ms to prefill them. All three land on
  the same number because the upload is the floor. The table lives in `docs/perf/kv-prefix.md`, and
  `crates/infr-llama/tests/kv_tier_latency.rs` regenerates it.
- The same comparison at long context: 4k 2095 → 77 ms, 31561 tokens 38.5 s → 593 ms. A parked
  conversation resumes through that restore; the parking test asserts both conversations come back
  with `cached_tokens > 0`.
