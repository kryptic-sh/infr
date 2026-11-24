# Plan 19: KV cache as a store, anywhere

Status: P1 shipped (`258a6e5f`, review fixes `ada1ebe4`); P2–P4 scoped. 18 unified where idle blocks
are kept; this makes the memory a slot's rows live in a tier like any other, with a `vram` tier as
the default rather than a hard-coded device buffer.

## Problem

After 18 a store can live anywhere except on the device. `[[kv.tiers]]` names `redb` and `redis`; a
live slot's rows are `SeamKv`'s device buffers, allocated `GpuOnly` by `BufferUsage::KvCache` with no
way to say otherwise; `DeviceKv` is a view over one slot rather than a store that holds many
extents. Two axes are tangled in that arrangement:

- **placement** — which store keeps idle blocks, unified in 18 behind `KvStore`;
- **residency** — where a live slot's bytes physically sit, hard-coded to device-local memory.

Left fused, three things stay impossible: an idle block cannot stay on the card, because `park_idle`
captures it out to the first tier and resets the slot; a slot cannot live in host memory
(`-nkvo`-style) for a model that barely fits; and the one escape hatch that exists, `kv.overflow`, is
a placement decision expressed as a boolean, which cannot also hold idle blocks or chain into a
storage tier.

## Constraint

Attention reads every row of the live window on every step, so residency has a bandwidth floor of
bytes per token times the window. With F16 K/V:

| model                       | KV per token | 8k working set | per step over PCIe ~25 GB/s | per step over loopback Redis ~3 GB/s |
| --------------------------- | ------------ | -------------- | --------------------------- | ------------------------------------ |
| Qwen2.5-0.5B (24L, 2 KV, 64) | 12 KB        | 100 MB         | 4 ms                        | 33 ms                                |
| 8B GQA (32L, 8 KV, 128)     | 128 KB       | 1.07 GB        | 43 ms                       | 357 ms                               |

Fair attention over 1 GbE puts the 8B row into seconds per token. So the live window's memory must
be device-addressable: either device-local VRAM, or host memory the kernels can address. A `redb` or
`redis` tier backs a restore, never a step. Rows cross the bus between requests, not during one.

## Design

**A tier names the memory.** Backends split into addressable (`vram`, `host`) and storage (`redb`,
`redis`). The first entry of `[[kv.tiers]]` is where a live slot's rows sit; storage tiers below it
receive what the addressable tier demotes. A storage backend may also be named first, and then it is
the slot's backing: the slot is host-resident and its window is synchronised with that store every
step. That is the experiment P4 measures — technically the same code path as a `host` tier, with the
store doing the reading and writing instead of a local buffer — not a configuration to recommend.

**The default is one `vram` tier, and it is implicit.** No tier list means `backend = "vram"` sized
from the device budget after weights and activations — exactly what a plain `infr run` allocates
today, so nothing changes for anyone who never writes a list. Writing `[[kv.tiers]] backend = "vram"`
with a `size` is how the same thing is said out loud and capped.

**Zero-copy is the retention path, not an optimisation of a copy.** A paged slot's rows are offsets
into buffers the pool owns: `PagedSlot::map` is `SlotMap { Arc<KvPool>, Vec<u32> }`, physical pages
carry refcounts, and `SlotMap::share_prefix` adopts another map's pages by cloning the table and
bumping the refs. So a parked conversation does not have to move at all: hand its `SlotMap` to a
resident entry keyed by the block chain it covers, keep the refs, and clear the slot's own map. A
restore that matches installs the entry with `share_prefix` and the rows are already where attention
reads them; the first write into a shared page pays one `cow_page`, a page copy rather than a prefix
copy. Nothing crosses the bus and nothing is decoded. This is P1 and it is shipped.

That makes the owner of the pages an L1 cache in VRAM, which is why it needs a budget and an LRU:
retained prefixes hold pages live slots would otherwise use. A page is retainable only when every
block it covers matched (512 rows against 128-token blocks), partial coverage falls through to the
storage tiers, and a retired entry is where the download happens — capture to the next tier, then
free the pages.

**The `vram` tier owns one budget and pins live windows.** A slot that is generating cannot have its
rows evicted underneath the kernels, so the tier's bytes are reserved per busy slot and only idle
extents sit in the LRU. A demote is `Backend::copy_buffer_at` between two device buffers, the
primitive the recurrent snapshot and the continuation seed already use, so `KvRows` never appears in
that path. The tier's `size` is what `kv_fit_ctx` subtracts before dividing the remainder among
slots; a configuration that leaves no usable window is refused at open with both numbers, not
silently clamped.

**Host residency is `backend = "host"`, the same shape.** Host-allocated KV buffers the kernels
address directly (`alloc_kv_host`, `INFR_KV_OVERFLOW`'s existing path), which is the `-nkvo`
analogue and the thing to reach for when a model leaves no room for a window. `kv.overflow` and
`kv.overflow_vram_mb` become the legacy spelling of a `vram` tier with a `size` followed by a `host`
tier, and keep working through a config migration.

## Phases

- **P1** (done, `258a6e5f`, review fixes `ada1ebe4`) resident page retention: a `SlotMap` handed to
  an idle entry on park, restored through `share_prefix`, a page budget with LRU spill, and a retired
  entry captured to the store before its pages go back. Gate met: a paged resume after a park
  performs no upload (asserted with counters), produces identical rows, and the GPU park test
  resumes with no store restore.
- **P2** (scoped) the `vram` tier: `backend = "vram"` in `[[kv.tiers]]`, implicit as the default when
  the list is absent, a budget covering pinned live windows plus idle extents with LRU eviction,
  device→device transfers, and `kv_fit_ctx` priced from the tier `size`. Gate: demote and promote
  never stage through the host, a vram tier still demotes into a `redb` tier, a vram tier smaller
  than one window is refused at open, and the implicit tier allocates what today's default path
  allocates.
- **P3** (scoped) the `host` tier: `backend = "host"` as first entry, host-visible mapped buffers
  with unchanged kernels, working with paging (today paging and `kv.overflow` are mutually
  exclusive), and the legacy `overflow` keys migrating onto it. Gate: output bit-identical to a vram
  tier; a whole host tier stores and resumes without a device allocation for KV. Measure: decode
  tokens/s at 2k and 8k on the RX 580 and the integrated GPU, next to the table above.
- **P4** (scoped) the remote-primary experiment: `[[kv.tiers]]` with `redis` (or `redb`) as the
  first entry, host-resident slot buffers, and the live window synchronised with that store each
  step — the configuration that proves placement is a property of the list rather than of the
  engine. Gate: output bit-identical to a vram tier, bytes transferred per step recorded and equal
  to the window, and the cost published. Expect it to be slow where the table above says it must be
  (a 4k window at 128 KB/token is ~0.5 GB over loopback per step); the small-model short-context row
  is the one that stays usable, and both numbers are what close the question for good.

## Non-goals

- Serving full attention from a `redb` or `redis` tier as a recommended configuration; P4's
  remote-primary experiment exists to show the path works, not to make it the default.
- Multi-device stores: a tier lives on one device, and the tensor-parallel wrappers stay as they are.
- Changing what a plain `infr run` does: the implicit `vram` tier is the default, so the default
  stays device-local.

## Verification

A seam test for park → resume that counts copies and asserts zero, with rows equal to the capture
path; unit tests for the retained-page budget/LRU, for the vram tier's pinned-versus-evictable split
and its demotion into the next tier, and for the implicit tier matching today's allocation; a
SeamModel test that a vram tier shrinks the per-slot window by its `size` and that an oversized one is
refused; GPU benchmarks for P3 recorded in `docs/perf/kv-prefix.md`.
