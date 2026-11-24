# Plan 18: KV tiers behind traits

Status: proposed.

## Problem

KV rows can live in four places today, and each move between them is hand-written:

- live slot rows in VRAM, laid out linear, as a sliding-window ring, or in pool pages;
- the host-RAM prefix store (`KvPrefixStore`, `kv.store_mb`);
- the cold tier (`KvColdTier`: journal file or Redis), which only mirrors the RAM store.

`fill_from_device`, `restore`, `read_rows`, `write_rows`, `write_ring_block`, `import_linear` and
`copy_tail` each re-derive where a logical row sits. The cold tier is not a spill tier: the RAM
LRU drops a block and the drop-time rewrite deletes it from Redis too, so Redis never holds more
than `store_mb`. An idle slot that loses its pages to a new request loses its reply rows, because
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
- `RamKv`: a hash map of extents in host memory;
- `DiskKv`: one redb table. A B-tree reuses the pages a deleted extent freed, which suits a
  budgeted cache that evicts all the time. Commits use `Durability::None`, because a lost
  extent is only a cache miss;
- `RedisKv`: one key per extent on Redis or Valkey.

**Capability traits** extend it where a store can do more:

- `DeviceStore: KvStore` exposes its backend and buffers, which gives `copy` a
  device-to-device path, so `seed_from`, `import_linear` and `copy_tail` keep their
  `copy_buffer_at` speed;
- `BudgetedStore: KvStore` reports bytes used and evicts the least-recently-used extent,
  which is what the tiering below needs.

**Combinators** wrap stores and implement `KvStore` themselves:

- `Tiered<Fast, Slow>`: reads fall through and promote, evictions from `Fast` demote to `Slow`,
  so `Tiered<RamKv, Tiered<DiskKv, RedisKv>>` is a single store;
- `Paged<S: DeviceStore>`: a page pool and page tables over a device store; spilling a page is
  a `copy` into a lower store and a release;
- `BlockCache<S: KvStore>`: the prefix cache, a token index over content-addressed extents
  (FNV chain id) in any store. Startup loads the index only, never the bytes.

Blocks stay 128 rows and pages 512. Extent length is per store, so neither unit changes.
Config picks the tiers at runtime, so they sit behind `Box<dyn KvStore>`, and the box
implements the trait too. Attention never goes through the trait: kernels keep binding the
device buffers and page table directly, so the trait costs nothing per token.

**Parked slots.** When the pool runs out of pages, the least-recently-used idle slot captures
   all of its rows, prompt and reply, into the store and releases its pages. Its conversation
   resumes through the ordinary prefix restore. This replaces the `kv_pool_exhausted` 400 for
   every case except a single request larger than the pool.

## Phases

- **P1** `RowLayout` and the generic row copies; rewrite the store and pool callers. No
  behavior change. Test: ring and page span splits, a CPU round trip through a wrapped ring, a
  linear-to-pages copy.
- **P2** `KvStore`, `KvRows`, `DeviceStore`, `copy`, `DeviceKv` and `RamKv`; `BlockCache`
  replaces `KvPrefixStore` over `RamKv`. No behavior change.
- **P3** `BudgetedStore`, `Tiered`, `DiskKv` and `RedisKv` replace `KvColdTier` and the
  journal file. Redis
  gets a per-block meta key (tokens, length, snapshot flag) so the index loads without the
  bytes. Test: demote and promote through two stores; a block evicted from RAM restores from
  the disk store.
- **P4** capture the whole slot after a generation, not only the prompt.
- **P5** `Paged<S>` replaces `KvPool`/`PagedSlot`, then parked slots on pool exhaustion. Test: two conversations in a one-conversation pool
  both resume with `cached_tokens > 0`.

## Measures

- Restore latency per 512-token block from RAM, file and local Valkey on the RX 580 host.
- TTFT of a resumed parked conversation versus a cold prefill, at 4k and 32k tokens.
