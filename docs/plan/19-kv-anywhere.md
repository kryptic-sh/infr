# Plan 19: KV residency behind the same store

Status: scoped. 18 unified where idle blocks are kept; this makes where a slot's rows live an
explicit choice, puts VRAM in the tier list, and keeps a device-local slot as the default.

## Problem

After 18 a store can live anywhere except on the device. `[[kv.tiers]]` names `redb` and `redis`; a
live slot's rows are `SeamKv`'s device buffers; `DeviceKv` is a view over one slot rather than a
store that holds many extents. Two different axes are tangled in that arrangement:

- **placement** — which store keeps idle blocks, unified in 18 behind `KvStore`;
- **residency** — where a live slot's bytes physically sit, hard-coded to device-local memory.

Separating them is what "the KV cache lives anywhere" actually needs. Left fused, three things stay
impossible: an idle block cannot stay on the card, because `park_idle` captures it out to the first
tier and resets the slot, so resuming pays a device→host→device round trip; a slot cannot run out of
host memory instead of VRAM (`-nkvo`-style) for a model that barely fits; and nothing in the tree
records why a remote cache is reachable per request but never per step.

## Constraint

Attention reads every row of the live window on every step, so residency has a bandwidth floor of
bytes per token times the window. With F16 K/V:

| model                       | KV per token | 8k working set | per step over PCIe ~25 GB/s | per step over loopback Redis ~3 GB/s |
| --------------------------- | ------------ | -------------- | --------------------------- | ------------------------------------ |
| Qwen2.5-0.5B (24L, 2 KV, 64) | 12 KB        | 100 MB         | 4 ms                        | 33 ms                                |
| 8B GQA (32L, 8 KV, 128)     | 128 KB       | 1.07 GB        | 43 ms                        | 357 ms                               |

Fair attention over 1 GbE puts the 8B row into seconds per token. So: device-local slots stay the
default for every model; host-visible residency is an opt-in experiment with a measured cost; a
remote tier backs a restore, never a step. Rows cross the bus between requests, not during one.

## Design

**Residency is a property of the shelf.** A `Shelf` owns one slot's buffers and layout, built from a
`RowLayout` and a `Residency`: `Device` (today), `Host` (host-visible mapped), `Pooled` (pages from
the shared pool). `SeamKv` holds a `Shelf` instead of raw `kbufs`/`vbufs`, and kernels keep binding
the buffer the shelf hands out plus the row layout over it, so nothing above the seam changes shape.

**Zero-copy is the retention path, not an optimisation of a copy.** A paged slot's rows are offsets
into buffers the pool owns: `PagedSlot::map` is `SlotMap { Arc<KvPool>, Vec<u32> }`, physical pages
carry refcounts, and `SlotMap::share_prefix` already adopts another map's pages by cloning the table
and bumping the refs. So a parked conversation does not have to move at all: hand its `SlotMap` to a
resident entry keyed by the block chain it covers, keep the refs, and clear the slot's own map. A
restore that matches installs the entry with `share_prefix` and the rows are already where attention
reads them; the first write into a shared page pays one `cow_page`, which is a page copy, not a
prefix copy. Nothing crosses the bus and nothing is decoded.

That turns the resident map into an L1 cache made of pages, which is why it needs its own budget and
LRU rather than being unbounded: retained prefixes hold pool pages that live slots would otherwise
use. A page is retainable only when every block it covers matched (512 rows against 128-token
blocks), partial coverage falls through to the host tiers, and a retired entry is where the download
happens — capture to the first tier, then free the pages.

**Linear slots need a device store, not a host bounce.** `VramKv` holds extents in device memory,
with a byte budget and an LRU, so `backend = "vram"` is legal as the first tier. A transfer to or
from it is `Backend::copy_buffer_at` between two device buffers — the same primitive the recurrent
snapshot and the continuation seed already use — so `KvRows` never appears in the path. A vram tier
requires an explicit `size` and refuses `path`/`url`.

**The budget has to price either one.** A vram tier's bytes and the retained pages both come out of
the device budget `vulkan_slot_ctx` divides among slots: `kv_fit_ctx` subtracts configured vram bytes
first, the retention cap is a share of what remains, and a configuration that leaves no usable
window is refused at open with both numbers, not silently clamped.

**Host residency is a slot property, not a tier.** `kv.slot_residency = "device" | "host"`, default
`device`. It names what a live slot reads from; a tier names where idle blocks are kept. Host
allocates the KV buffers host-visible and mapped and leaves the kernels alone.

## Phases

- **P1** Resident page retention: a `SlotMap` handed to an idle entry on park, restored through
  `share_prefix`, a cap and LRU over the retained pages, and a retired entry captured to the first
  tier before its pages go back. Gate: a paged resume after a park performs no copy of any kind
  (counted, not inferred from timing), and produces the same logits as the capture path. Measure:
  resume latency against the in-memory redb path, and pool pages held by idle prefixes.
- **P2** `VramKv` for linear slots: device extents with `copy_buffer_at` transfers, the budget and
  LRU, `backend = "vram"` in `[[kv.tiers]]`, and the `kv_fit_ctx` subtraction. Gate: no host staging
  on demote or promote — the transfer is device→device — and a vram tier still demotes into a redb
  tier.
- **P3** Host residency: `kv.slot_residency = "host"`, host-visible mapped buffers, unchanged
  kernels. Gate: opt-in only, output bit-identical to device residency. Measure: decode tokens/s at
  2k and 8k on the RX 580 and the integrated GPU, next to the table above.
- **P4** Measure the remote case instead of building it: a window's bytes fetched from loopback
  Redis and from host RAM per step, timed in `crates/infr-llama/tests/kv_tier_latency.rs` and added
  as rows in `docs/perf/kv-prefix.md`. Expected to lose to the device floor; the number is what
  closes the question for good.

## Non-goals

- Serving full-attention attention from a `redb` or `redis` tier.
- Multi-device stores: a tier lives on one device, and the tensor-parallel wrappers stay as they are.
- Changing the default. No tier list and a device-local slot remain what a plain `infr run` builds.

## Verification

A seam test for park → resume that counts copies and asserts zero, with logits equal to the capture
path; unit tests for the retained-page cap/LRU and for `VramKv`'s budget and demotion into the next
tier; a SeamModel test that a vram tier shrinks the per-slot window by its `size` and that an
oversized one is refused; GPU benchmarks for P3 recorded in `docs/perf/kv-prefix.md`.
