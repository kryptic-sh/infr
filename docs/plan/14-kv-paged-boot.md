# 14 — Allocate the `kv.paged` pool in the cold KV init

## Problem

`ParallelSeam::init_slots` runs the warmup (which allocates slot 0's linear KV),
resets it, then builds a `max(n_slots, POOL_BLOCKS)`-block page pool on top and
imports into it. The boot peak is therefore the linear slot **plus** the whole
pool, while the clamp and the cold re-clamp price `n_slots` caches.

At `n_slots >= 2` the pool needs `n_slots + 1` caches at boot, so the guard
refuses it; the server falls back to linear slots exactly where paging pays. At
`n_slots == 1` the pool costs two caches for a session that never forks.

Measured on kikuri (RX 580 8 GB, Qwen3.5-4B Q4_K_M, q8 KV):

- `--parallel 2`, automatic clamp 103559/slot: `kv.paged` refused, 215.7 MiB
  requested + 7.62 GiB in use > 7.75 GiB budget; slots stay linear.
- 1 slot, q8: `--ctx 81920` builds the pool; `--ctx 98304` is refused. The
  boundary is the transient linear cache, not the pool itself.

## Decision

- **No pool at `n_slots == 1`.** `ParallelSeam` never forks a single-slot
  engine, so there is no sharing target. `kv.paged` stays linear there with one
  INFO line. The `SlotPool`/`start_paging` fork path (run/bench/chat) is
  unchanged.
- **Build the pool in the cold KV init**, after the post-load re-clamp and
  before any KV allocation, with `blocks = max(n_slots, POOL_BLOCKS)`. Pooled
  layers take their K/V from the pool; only non-pooled layers (DeltaNet, SWA
  ring, MLA, diffusion) allocate linear buffers. No linear pooled cache ever
  exists, so the boot peak is the pool alone.
- The guard (`refuse_unallocatable_kv`) prices `blocks` contexts for pooled
  layers and one per slot for the rest, so it agrees with the allocation.
- **No clamp change.** For `n_slots >= 2`, `blocks == n_slots`, so the existing
  per-slot division already prices the pool. For `n_slots == 1` there is no
  pool.
- If `KvPool::new` fails, warn and allocate every layer linearly, as today.

## Plumbing

- `generate_dense_vulkan_session` and `generate_dense_backend_bound` gain
  `paged_blocks: Option<usize>`.
- `ParallelSeam::init_slots` passes
  `Some(n_slots.max(POOL_BLOCKS))` when `kv.paged` is on, `n_slots > 1` and the
  backend reports `kv_paging`; `None` otherwise. Other callers pass `None`.
- Remove the post-warmup `KvPool::new` + `attach_paged` block from
  `ParallelSeam::init_slots`; the cold init owns the pool.
- In the cold init: after `reclamp_ctx_to_live_room`, if `paged_blocks` is
  `Some` and the backend reports `kv_paging`, build the pool and use
  `pool.slot_rows() * blocks` rows for pooled layers in `kv_sizes`; allocate
  the remaining layers linearly. Set `paged: Some(PagedSlot::new(&pool, be)?)`
  in the constructed `SeamKv`. Pooled `kbufs`/`vbufs` are `Arc` clones of the
  pool's buffers.
- `max_ctx` stays the requested window (the pool rounds its rows up to a page;
  `PagedSlot` owns that mapping).

## Tests

- Pure: a sizing helper prices `blocks` contexts for pooled layers and
  `n_slots` for the rest, tested on a synthetic qwen35-style config (8 pooled,
  24 recurrent).
- Policy: `paged_blocks_for(n_slots, paged, caps)` is `None` for 1 slot, `Some(2)`
  for 2, `Some(4)` for 4.
- Existing `kv_paged_session` / `kv_paged_graph` must still pass.
- Hardware acceptance (kikuri, RX 580): q8 KV, `--parallel 2`, no `--ctx`:
  the log must show `kv.paged: ... in 2 pool blocks of 103936 rows` and no
  budget refusal.

## Out of scope

`start_paging`'s first-fork attach still holds the parent's linear cache while
it builds the pool. That path serves run/bench/chat; its transient is a separate
slice if it ever matters.
