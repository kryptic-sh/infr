# Plan 21: a reasoning budget that the engine enforces

Status: P1–P3 shipped, P4 open. The surface, the counter and the forced close are in: `think_max_tokens`
and `reasoning_effort` reach the decode loop as a number, `ThinkBlock`/`ThinkTracker` count the block
and substitute the template's own close, and `infr_chat::think_markers` derives that close by diffing
the two renderings. Three deltas from the plan as written: the table is three flat keys
(`sampling.think_budget_low` / `_medium` / `_high`) because `cfg_struct!` forbids a `think_budget`
section mixing subsections and leaves; the forced step is a plain override rather than a constrained
step, since a budget only exists on a free-form reply where no grammar owns the tokens; and the
interactive chat path gets no budget, because `Chat::turn` renders without a thinking toggle and so has
no no-think twin to diff the close from — the feature is a per-request one (`infr run <prompt>` and
`infr serve` both go through the same `prepare`/`run_prepared` pair). P4 still has to be measured.

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

- **P1** (shipped) the surface: `sampling.think_budget_low` / `_medium` / `_high` and
  `think_budget_tokens`, `think_max_tokens` on the request with validation, `reasoning_effort` resolved
  through the same table, and `docs/config.md` entries. Gate: `low` and `high` reach the engine as
  different numbers (`the_effort_levels_reach_the_generator_as_different_budgets`), an unknown level and
  a zero budget are 400s (`an_invalid_think_budget_is_a_400`), and `none` still means off.
- **P2** (shipped) enforcement for a single-token close: `ThinkBlock` from the renderer, the counter in
  the decode loop, and the forced close. Gate covered by the tracker's unit tests rather than by a
  scripted 100-token stream: `budget_forces_the_close_after_exactly_the_allowed_tokens`,
  `a_natural_close_is_left_alone`, `an_abandoned_close_counts_as_reasoning`.
- **P3** (shipped) multi-token closes and the streamed shape: `think_markers` accepts a two-token close
  and a mismatched partial close counts as reasoning, and the splitter already holds back a partial
  marker (`streams_channel_thought_then_final`, `streams_think_then_content`), so a forced close lands
  whole. A template without a think block yields `No` markers and an inert budget
  (`no_close_marker_means_no_block`).
- **P4** (partly measured, see [../perf/reasoning-budget.md](../perf/reasoning-budget.md)) measure on
  LFM3-1.7B Q4_K_M on the RX 580 at the three levels: reasoning tokens emitted, TTFT, and total ms for
  a fixed question set, recorded in `docs/perf/`. Sanity check that the answer quality does not collapse
  at `low` by comparing greedy output with an unbeaten budget. Done on Qwen3-1.7B Q4_K_M for the token
  and timing numbers (the LFM3 file is not cached here); the LFM3 row and the quality comparison are
  the open part.

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
