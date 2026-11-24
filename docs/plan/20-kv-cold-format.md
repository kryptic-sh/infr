# Plan 20: cold KV in a compressed format, and an eviction that keeps the trunk

Status: scoped. 18 and 19 put idle blocks in tiers and made the live window a tier's memory; this
decides what a tier holds — a compressed copy of the rows rather than the live cache's dtype — and
which block a full tier gives up.

## Problem

A tier stores `KvRows` verbatim: `RedbKv::put(id, first, rows)` writes the bytes the live cache
produced, in the live cache's `k_fmt`/`v_fmt`. Three consequences, all of them avoidable:

- **Cold capacity is the live format.** An f16 cache cannot keep q8 cold blocks, so the touted
  2–4× of a compressed cold tier needs the whole cache quantized, which taxes every step instead of
  only the restore. `docs/perf/kv-prefix.md` measures a 32k restore at 593 ms and 385 MB, about half
  of it redb decode plus the host copy: those bytes are the tier's format, not the device's.
- **A format difference is a miss.** The store's formats blob is checked before a restore and a
  mismatch returns 0, so two tiers that disagree — or one tier written under a different `type_k` —
  cannot exchange a block at all.
- **Eviction drops the wrong end.** `KvStore::contains` never stamps the budgeted tier,
  `BlockCache::matched` walks the chain with `contains`, and `capture` skips blocks that are already
  present. A chain hit therefore leaves the trunk unstamped while fresh blocks carry recent stamps,
  so `take_lru` evicts the front of the chain first: the prefix every request needs is the first
  thing a squeeze throws away, and the next request misses at token 0. `Entry` also has no parent,
  so nothing can tell a leaf from a block other chains extend through.

## Design

**The format belongs to the tier.** `[[kv.tiers]]` grows `type_k`/`type_v` (`f16`, `q8_0`, `q4_0`;
absent means `source`, today's behaviour of storing what the live cache holds). Each tier already
owns a formats blob (`FORMATS_META_ID` is per store), so a tier records what it actually holds and
refuses to open when the blob and the configured format disagree, rather than reinterpreting old
bytes. `plus` for a `Tiered` chain: promotion converts in one pass instead of refusing.

**Conversion happens at the tier boundary.** `KvStore` grows `fn format(&self) -> (KvFmt, KvFmt)` and
`get` takes the format the caller wants; a tier whose own format already matches returns the bytes
untouched, the others convert. Rows are read through `RowLayout::spans()` into packed host bytes
and converted 32 elements at a time (q8_0: f16 scale plus 32 `i8`; q4_0 the same with nibbles), which
is a CPU pass with no shader and no device work. The Vulkan planar Q8_0 side layout stays a
device-side concern — host rows are packed, so a planar slot compressing to q8 is nearly free.

**Eviction keeps the trunk.** `Entry` gains `parent: Option<u64>`, known at insert from the chain the
capture walked. `contains` stamps, `capture` re-stamps a block it finds present, `matched` touches
the chain it matched, and the LRU ranks on those stamps — so the trunk of a chain in use is the most
recently used thing in the store instead of the least. On overflow a leaf-first pass runs: an entry
no stored entry names as parent goes before anything a chain continues through, and a mid-chain block
only goes when every leaf is newer. With compression multiplying what a tier holds, this policy is
what decides whether the extra capacity is worth having.

**Miss reasons become counters.** A restore that matches nothing, or stops at block k, records why
(empty chain, mid-chain, format, coverage, `max_ctx`) next to `restores`/`restored_tokens`, exposed
where `kv_store_stats()` already is. Today a miss is a missing debug line and nothing else.

## Phases

- **P1** (scoped) the eviction fix, format untouched: `parent` on `Entry`, stamping `contains` and
  `capture`, chain touch in `matched`, leaf-first overflow, and the miss-reason counters. Gate: a
  chain long enough to overflow a small tier keeps blocks 0..k while both its tail and a second
  chain's tail are dropped, and an entry reached only through `contains` is no longer the next
  victim.
- **P2** (scoped) the per-tier format: `type_k`/`type_v` per tier, `format()` and `get`'s wanted
  format, requant on `put`, dequant on `get`, formats blob per tier, `Tiered` promotion converting
  instead of refusing. Gate: an f16 live cache with a q8_0 tier round-trips rows equal to a q8_0
  live cache's, a tier reopened with a different configured format is refused rather than
  reinterpreted, and a 32k restore moves half the bytes it moves today.
- **P3** (scoped) `q4_0` cold behind the same knob, offered only with a greedy-output comparison
  against f16 and q8_0 on a fixed prompt set, the divergence recorded, and a warning at open.
- **P4** (scoped) measures in `docs/perf/kv-prefix.md`: 32k restore bytes and milliseconds per
  format on the RX 580, capacity per GiB, and the same for Redis over loopback, next to the numbers
  plan 18 recorded.

## Non-goals

- Reusing a block at a different position (chunked hashing, CacheBlend-style partial recompute):
  separate plan, tracked as B242.
- Peer-to-peer block sharing and any RDMA path: single node is the target, and a cluster we do not
  own cannot be verified.
- Disaggregated prefill and decode: there is no second GPU to test the split with, and the transport
  it would need is the store we already have.
- Lossless `q4_0`: it is a capacity trade, judged by the output comparison, not by bit equality.

## Verification

A unit test for each eviction rule (trunk survives, leaf goes first, `contains` stamps); a
conversion round-trip per format pair, comparing against rows produced by a live cache configured
with that format rather than against the source; a `Tiered` promotion test across two formats; a
SeamModel test that a tier's configured format survives a reopen and a mismatch is refused; and the
restore measurements for P4.
