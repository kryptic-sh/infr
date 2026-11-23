# 15 — `/v1/models` reports the served context window

## Problem

`GET /status` already exposes `ctx_per_slot`, but an OpenAI client that only
knows `/v1/models` cannot see how many tokens the server will accept. A model
card answers `id`, `object` and `owned_by` and nothing else.

## Decision

Add `context_length: Option<u32>` to `ModelCard`, read from the hosted
generator's `GeneratorStatus::ctx_per_slot` — the per-slot window after the
clamp and the post-load re-clamp, which is what the server actually admits.
`null` when the generator cannot report it (the honest "unknown", the same rule
`/status` follows).

`ParallelGenerator` already reports `ctx_per_slot = engine.max_ctx()`; the
single-session `SeamGenerator` reports `None`, which stays `null`. Multi-model
servers report each model's own window.

## Slice

- `ModelCard` gains `context_length: Option<u32>` and the `/v1/models` handler
  fills it from `m.engine.as_ref().and_then(|g| g.status().ctx_per_slot)`.
- Update the route-list doc comment and the two serialization tests; add a
  handler test where `EchoGen` (ctx 4096) yields `context_length: 4096` and a
  `None` generator yields `null`.

No per-model `GET /v1/models/{id}` route exists, so the list is the only change.
