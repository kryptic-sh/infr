# 16 — Dynamic KV: one page pool, allocated per request

## Goal

One KV pool, sized once to the VRAM budget, with pages handed to a request as
its sequence grows and returned when it finishes. A request's *window* is the
model's trained context; its *memory* is what it actually uses. A 4k request
holds eight 512-row pages, not a 4k slice of a preallocated block.

## What already exists

- `infr-vulkan/src/paged.rs`: `PagePool` (refcounts, `alloc`/`share`/`release`),
  `SlotPages` (`grow`, `truncate`, `clear`, `fork`, `ensure_writable`), and the
  table the attention kernels already read.
- `seam/pool.rs`: `KvPool`, `SlotMap` (one fixed block per slot, `row_base`),
  `PagedSlot` (table buffer, COW, prefix sharing).
- `seam/weights.rs`: `attach_paged`, `fork`, `paged_row_base_at`.
- `seam/kv_store.rs`: the host prefix store captures/restores rows through the
  same `row_base`.

The gaps are the fixed block (`KvPool::new` allocates `blocks * slot_rows`
up front, `SlotMap::take` maps every logical page to its block) and the **store
path, which writes at `row_base + pos` and never consults the table**.

## Design

1. **Shared pool.** `KvPool` holds `pages` (from `kv.pool_mb`, or auto-sized to
   the KV fit after weights) and no `blocks`/`slot_rows`. `SlotMap` starts
   empty and grows one logical page at a time; `clear`/`truncate` release pages
   at the end of a request. `--parallel N` still bounds concurrent sequences;
   the pool is shared across them.
2. **Writes resolve pages on the host.** For a store of logical rows
   `pos..pos+n`, look up `table[pos/PAGE_ROWS]`, call `ensure_writable` (COW)
   once per touched page, and record the existing `store_f16`/`store_q8`
   dispatch split at page boundaries with a physical row base per segment. A
   chunk is at most 512–1024 rows, so this is one to three dispatches per
   store. The shaders do not change in the first cut; the table-aware store is
   an optimization if the dispatch split ever measures.
   The decode params path (`USE_PARAMS`) resolves the one current row the same
   way before updating `p_pos`.
3. **Reads are unchanged.** Attention already reads through the table.
4. **Admission reserves pages.** At request start, reserve
   `ceil((prompt + max_new)/PAGE_ROWS)` pages (or `max_new` as the client
   sent it), so generation never fails mid-stream. A request that does not fit
   is a clear 400 ("the pool has M pages, this request needs N"). No
   preemption in the first cut; the alternative (grow lazily and preempt /
   recompute) is a later policy.
5. **Non-pooled layers stay per-slot linear.** SWA rings, recurrent/DeltaNet
   states, MLA and diffusion layers keep their small per-slot buffers. Only
   full-attention layers use the pool.
6. **Prefix sharing and COW** are the existing `share_prefix`/`ensure_writable`
   mechanics, now on demand: a matching prefix points at the parent's pages and
   only the tail is allocated.

## Phases

- **P1 — Host-resolved stores.** Remove the `row_base` assumption from the
  store path; a permuted page map must produce bit-identical logits to a
  linear cache. No shader change. This is the gate: nothing else starts until
  a scrambled map is exact.
- **P2 — Grow on demand inside a slot.** `SlotMap` starts empty, grows as
  `cached` advances, uploads the table once per forward, frees on `reset`.
  Test: two sequential requests reuse the same pages (`in_use` returns to 0).
- **P3 — Shared budget and admission.** `KvPool` pages from a budget;
  reservation at admission; `/status` reports pages used/free. Test: two
  concurrent requests share the budget, a small request holds few pages, an
  over-budget request is a 400 before prefill.
- **P4 — Prefix sharing under dynamic pages.** `share_prefix` + COW; the host
  prefix store (`kv.store_mb`) captures/restores logical pages instead of a
  row range. Test: two conversations over a long shared prefix, and the
  existing `kv_paged_session`/`kv_paged_graph`.
- **P5 — Fit and chunk decoupling.** `max_ctx` becomes the trained window
  (capped by the model), and the pool budget comes from the VRAM fit. The
  activation reserve no longer prices every slot at the full window, so a
  short request can use a larger prefill chunk. Measure TTFT and decode at 4k
  vs 112k on the RX 580, plus the 2-slot case.

## Knobs and compatibility

- `kv.paged` keeps turning the page path on. A new `kv.pool_mb`
  (`INFR_KV_POOL_MB`) sets the pool budget; unset means auto-size from the KV
  fit after weights. The fixed `blocks * slot_rows` mode is removed once P3
  lands (it is the same code with the pool sized by slots).
- `max_ctx` per request is still what `/status` reports; the model card gets
  `context_length` from plan 15.

## Risks / open decisions

- Reserving `prompt + max_new` can waste pages for a client that asks for a
  huge `max_new`. Start with the reservation; if it bites, reserve the prompt
  plus a growth window and reject only on real exhaustion.
- The prefix store must walk pages; if that is awkward, keep `store_mb` on the
  linear path for one release and document the interaction.
- A failed admission must not leave a partially mapped slot; free on every
  error path.
- COW copies cost about one page of K/V per pooled layer on the first write to
  a shared page; overlapped with prefill, but measure it at a long shared
  prefix.
- The store dispatch split adds one or two dispatches per store; measure
  before considering a table-aware shader.

## Acceptance

- A permuted page map is bit-identical to a linear cache (P1).
- One 8 GB RX 580 serves a 4k request and a 100k request from the same
  auto-sized pool without preallocating either window per slot.
- Two concurrent requests share a prefix; `in_use` returns to zero after both
  finish.
- TTFT at 4k does not regress against today's small-window server.
