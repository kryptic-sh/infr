# KV prefix reuse: the store, and the inversion it is waiting for

Two different problems get called "prefix caching". They share a key and a
block size, and nothing else.

1. **Reuse across requests.** A new conversation starts with text an old one
   already prefilled. The cache lives *outside* the slot and is copied back in.
2. **Reuse inside a slot set.** Two live sequences share a prefix, so their
   page tables point at the same physical rows and diverge by copy-on-write.

This tree ships (1) and is building the table for (2). When writing about them,
be blunt: (1) is **reverse PagedAttention** — paging turned inside out. Physical
rows stay contiguous in a slot; the blocks are the thing that travels, and the
"restore" is a bulk upload back into a linear buffer. It is a cache in the CPU
sense, not an allocator.

## Knobs

Two mechanisms, two knobs, and neither implies the other:

| Knob | Config / env | Direction | Status |
| --- | --- | --- | --- |
| `store_mb` | `kv.store_mb` / `INFR_KV_STORE_MB` | reverse (host block store) | shipped |
| `paged` | `kv.paged` / `INFR_KV_PAGED` | forward (page pool) | not wired |

`kv.paged` does not exist in the tree yet: the page-table read path, the
refcounted pool and the offset-capable copies are in, but nothing allocates
pages until the seam binds a slot's rows into the pool. Do not document it as a
working switch before that lands.

## Reverse PagedAttention: what `kv.store_mb` does

`crates/infr-llama/src/seam/kv_store.rs` keys captured KV in 128-token blocks
(`BLOCK_TOKENS`), chained by a hash of `(previous block id, tokens)`, and keeps
them in host RAM under an LRU byte budget. A request walks the chain from token
0 as far as the prompt agrees, stamps the matched blocks as recently used, and
uploads each layer's whole matching run in one contiguous copy.

Measured on the RX 580 with Gemma 4 E4B (1 slot, 8k ctx, 4 GB store):

| Request | Prompt tokens | Prefill | Wall |
| --- | ---: | ---: | ---: |
| first, no entry | 1068 | 1068 | 3677 ms |
| different tail | 1090 | 1024 restored + 29 prefilled | 705 ms |
| first again | 1068 | 1068 restored | 319 ms |

The two conversations cost 111.8 MB of store between them, because the shared
1053 tokens are stored once.

## Why it is the wrong way around for long context

The restore is a PCIe upload of the entire matched prefix. At the runtime's
~36 KB/token that is fine for chat and fatal for a coding run:

| Context | KV per slot | Restore upload | Re-prefill at 350 t/s |
| --- | ---: | ---: | ---: |
| 8k | 0.29 GB | ~0.6 s | ~23 s |
| 32k | 1.15 GB | ~2.4 s | ~94 s |
| 128k | 4.6 GB | ~9.6 s | ~375 s |

At 8k the store wins 40×. At 128k it still wins, but it costs 2.5% of the
request *and* holds a second full copy in RAM, so the win is being spent on
avoiding an upload the hardware could have skipped entirely. A coding agent
that forks a branch every few hundred tokens pays that upload on every request
even though only the tail changed.

The inversion fixes that: a page table maps logical 512-row pages to physical
pages in a shared pool, a fork points at its parent's pages, and the first write
into a shared page copies it. Restore becomes "fill the table", with uploads
only for pages that are not resident. The host store survives as the cold tier
under the pool instead of being the whole mechanism.

## What the inversion costs

- The read side is one table read per chunk and one compare per key in
  `attn_decode.comp` (`pc.page_rows == 0` is the identity), so a linear cache
  pays a compare and nothing else.
- The write side needs no shader change: `store_kv_dense.comp` and friends write
  at a raw element offset, so a chunk that straddles a page boundary is split
  into two dispatches that differ only in `off`. Decode writes one row and never
  straddles.
- The cost is the allocator: a physical page pool with refcounts, copy-on-write
  on the first write into a shared page (~5 MB per layer at hd 2560, ~0.2 ms
  across the 7 non-ring layers), and eviction into the host tier.
- Only the non-ring layers grow with context. On Gemma 4 E4B that is 7 of 42
  layers; the 35 sliding-window layers keep their fixed-size rings and their
  `% rcap` mapping, untouched.

So: keep the store for chat-shaped workloads, and finish the table so a slot can
stop being a contiguous buffer.
