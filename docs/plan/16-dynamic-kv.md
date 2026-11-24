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

The fixed block and the `row_base` store path are gone: `KvPool` is a budget of
physical pages, each slot grows its map on demand, and the store uploads and
downloads through that map.

## Design

1. **Shared pool.** `KvPool` holds `pages` (from `kv.pool_mb`, or auto-sized to
   the KV fit after weights) and no `blocks`/`slot_rows`. `SlotMap` starts
   empty and grows one logical page at a time; `clear`/`truncate` release pages
   at the end of a request. `--parallel N` still bounds concurrent sequences;
   the pool is shared across them.
2. **Writes resolve pages in the store kernels.** For a store of logical rows
   `pos..pos+n`, the host still calls `ensure_writable` (COW) once per touched
   page, but the destination row is resolved in the kernel: every store shader
   computes `phys = table[row / PAGE_ROWS] * PAGE_ROWS + row % PAGE_ROWS` from
   the table's device address and `page_rows` (push constants on the
   static/prefill path, params SSBO words 2-3 on the record-once decode path),
   so no host per-token updates and no dispatch splits at page boundaries are
   needed. `page_rows == 0` keeps the linear identity.
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
   mechanics, now on demand: at admission, a request whose prompt starts with a
   live slot's cached tokens points the new slot at those pages and prefills only
   the tail. This is the performance win, now on demand rather than at a block
   boundary. **Hybrid models are served by P3's snapshots.** A recurrent layer's
   state cannot be sliced out of a page, so it rides the page-boundary snapshot
   instead: a divergent prompt resumes from the newest snapshot at or before its
   common prefix, and the host store keeps the same snapshots. Page sharing
   therefore helps every full-attention layer, and gemma's SWA ring rows are
   still copied rather than shared.

## Phases

**Status (2026-11-24):** P1–P6 shipped. A paged session advertises the pool's token
fit as its window instead of `fit / slots`, the pool is sized from that fit (or
`kv.pool_mb`), and slots hold only the pages they use. The prefix store behind it is
the `[[kv.tiers]]` list ([plan 18](18-kv-tiers.md)): a redb file or memory tier and,
optionally, a shared Redis/Valkey tier, so a restart restores a thread's prefix and
prefills only its tail. The activation reserve still prices the whole window, so a short request preflights at the window's chunk height rather than a
taller one.

- **P1 — Table-resolved stores.** Remove the `row_base` assumption from the
  store path: every store kernel resolves the destination row through the
  slot's page table (`phys = table[row / PAGE_ROWS] * PAGE_ROWS + row %
  PAGE_ROWS`, the same formula the reads use) instead of landing at
  `row_base + pos`. The table's device address and `page_rows` reach the
  kernels as push constants on the static/prefill path and via the params SSBO
  (words 2-3, unchanged by `params_advance`) on the record-once decode path, so
  a permuted page map must produce bit-identical logits to a linear cache with
  no host per-token updates and no dispatch splits. `row_base` remains only the
  linear fallback/bias where the table is unavailable (never added on top of a
  resolved row). No change to reads. This is the gate for every non-contiguous
  map.
- **P2 — Runtime prefix borrow (the speed slice).** At admission, when the
  prompt shares a cached prefix with a live slot and the model has no
  unrewindable state, seed the new slot's pages from that slot
  (`share_prefix`), set its `cached` to the shared tokens, and prefill only the
  suffix. Measure TTFT for two concurrent conversations over a shared system
  prompt. SWA-ring rows are copied for gemma; qwen35 is refused until P3's
  snapshot exists.
- **P3 — Recurrent snapshots at prefix boundaries.** For qwen35 and other
  unrewindable models, capture the recurrent state at page boundaries (the
  store's per-block capture, extended) so a shared prefix can restore it. The
  pi session on LFM3-F16 is the case to fix: turn 1 prefilled 36,636 tokens,
  and the next two turns, from the same live slot two minutes later, prefilled
  36,994 and 38,680 in full for only 358 and 1,686 new tokens. The prompt
  diverged inside the previous turn (pi re-renders the assistant message and
  its tool call, and its MCP section changed), so the common prefix ends before
  the cached end. A snapshot at the prefill boundary cannot serve that point.
  Snapshot at every page boundary, take it on every turn rather than only when
  a decode chain can run, and restore the newest boundary at or before the
  common prefix. That turns those turns into a 358- and 1,686-token prefill.
  The same snapshot is what the store keeps, so a restart restores too.
- **P4 — Shared budget and admission.** `KvPool` pages from a budget;
  reservation at admission; `/status` reports pages used/free. Test: two
  concurrent requests share the budget, a small request holds few pages, an
  over-budget request is a 400 before prefill.
- **P5 — Grow and release on demand, and the store.** `SlotMap` grows as
  `cached` advances and frees on `reset`; `kv.store_mb` captures/restores
  logical pages instead of a row range.
- **P6 — Fit and chunk decoupling.** `max_ctx` becomes the trained window
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
- Two concurrent conversations over a shared system prompt prefill the second
  one's prefix once, on a full-attention model (P2).
- One 8 GB RX 580 serves a 4k request and a 100k request from the same
  auto-sized pool without preallocating either window per slot (P4/P5).
- Two concurrent requests share a prefix; `in_use` returns to zero after both
  finish.
- TTFT at 4k does not regress against today's small-window server.
- A divergent follow-up on an unrewindable model (an edited last turn)
  restores a snapshot and prefills from the divergence, not from token zero.
- A pi turn that re-renders the previous assistant message and tool call
  prefills only the tokens it added, not the whole prompt.
- After a restart, the same thread restores from the host store and prefills
  only its tail.
