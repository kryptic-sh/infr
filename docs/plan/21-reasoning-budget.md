# Plan 21: a reasoning budget that the engine enforces

Status: scoped. Today the three request switches that ask for thinking are flattened to a boolean, so
`low` and `high` are the same request; this gives them a token count the engine honours by closing the
think block when it runs out.

## Problem

`request_think` (`crates/infr-server/src/lib.rs:916`) resolves `chat_template_kwargs.enable_thinking`,
then `think`, then `reasoning_effort` into one `bool`. `effort_think` (`:949`) accepts
`none|minimal|low|medium|high` and rejects anything else, so every level above `none` produces the
identical request. Clients that send `reasoning_effort: "low"` expecting a short think get the model's
full default, and a model that thinks for two thousand tokens on a one-line question cannot be told
otherwise for a single request.

The engine has nothing to enforce it with, either. Whether generation starts inside a think block is a
property of the rendered prompt — LFM3 ends in ` thinking` with thinking on and ` thinking</think>` with it
off (`crates/infr-chat/tests/lfm3_template.rs:4-6`), SmolLM3 renders `/no_think` — and the block ends
when the model emits the family's close token. Nothing counts tokens from the block's start, and there
is no way to end the block early. `sampling.no_think` (`INFR_NO_THINK`) is the only global lever, and
it is all-or-nothing.

## Design

**The budget is a token count, and the effort words map onto it.** `sampling.think_budget` holds the
three levels — `low`, `medium`, `high` — with defaults 512, 2048 and 8192, plus
`sampling.think_budget_tokens` for a flat number. A request may override with an exact count (a
`think_max_tokens` field, validated as a positive integer with a 400 that names the field), while
`reasoning_effort` keeps its string shape and is resolved through the same table, so an existing
client gets the behaviour it expected without new spelling.

**The renderer already knows where the block is, so it hands that to the generator.** Whether
generation starts inside a block, and which token sequence closes it, are both known when the prompt is
rendered; carry them into the generation parameters as a small `ThinkBlock { open, close }`. A template
with no reasoning block, a prompt that ends outside one, or `reasoning_effort: "none"` all make the
budget inert rather than approximate.

**Forcing the close is a constrained step, not a prompt trick.** Count tokens emitted since the block
opened; when the count reaches the budget, admit only the close sequence's next token for the next
step (a multi-token close advances one token per step), then resume ordinary sampling for the answer.
If the model closes the block itself first, nothing changes — the budget only ever shortens.

**Accounting stays honest on the wire.** The forced close arrives as the tail of
`reasoning_content`, the answer follows in `content`, and both count in `usage.completion_tokens`. A
budget hit is logged with the level and the limit, so "this model overthinks" becomes a number rather
than an impression.

## Phases

- **P1** (scoped) the surface: `sampling.think_budget` / `think_budget_tokens`, the level table,
  `think_max_tokens` on the request with validation, `reasoning_effort` resolved through the same
  table, and `docs/config.md` entries. Gate: `low` and `high` reach the engine as different numbers, an
  unknown level and a zero budget are 400s, and `none` still means off.
- **P2** (scoped) enforcement for a single-token close: `ThinkBlock` from the renderer, the counter, the
  constrained step, and the resume. Gate: with a scripted token stream that would think for 100 tokens
  and a budget of 32, exactly 32 reasoning tokens are emitted, the close sequence follows, and the
  answer continues to the stop; a model that closes at 20 tokens is untouched.
- **P3** (scoped) multi-token closes and families whose close is not a single token, plus the streamed
  shape: `Delta::Reasoning` ends at the forced close, `content` begins after it, and a template without
  a think block leaves the budget inert. Gate: a two-token close lands in one piece in the streamed
  text, with no partial close sequence visible to the client.
- **P4** (scoped) measure on LFM3-1.7B Q4_K_M on the RX 580 at the three levels: reasoning tokens
  emitted, TTFT, and total ms for a fixed question set, recorded in `docs/perf/`. Sanity check that the
  answer quality does not collapse at `low` by comparing greedy output with an unbeaten budget.

## Non-goals

- Budgets inside a tool call or a JSON-schema reply: `response_format` already runs with thinking off,
  and a truncated tool call is worse than a long one.
- Adaptive budgets learned per request, or a per-model default table beyond the three levels.
- Models whose template has no reasoning block — the budget stays inert there rather than inventing a
  block.

## Verification

Unit tests over scripted token streams for the single- and multi-token closes and for a model that
closes early; a request-level test that the level table and `think_max_tokens` both reach the
generator and that bad values are 400s; a streaming test that the forced close never leaks a partial
sequence; and the P4 measurements next to the current unbudgeted numbers.
