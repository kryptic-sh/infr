# Reasoning budgets

Plan 21's token budget for the think block (see
[../plan/21-reasoning-budget.md](../plan/21-reasoning-budget.md)): what it costs
and what it buys, measured end to end through `infr serve`.

The plan named LFM3-1.7B on the RX 580. That model is not in this machine's
cache and the budget is model-agnostic, so the numbers below are the cached
Qwen3-1.7B instead; the LFM3 row is still open in the plan.

## Setup

- AMD Radeon RX 580 8 GB (RADV POLARIS10, wave64 device running the wave32
  kernels), Vulkan0, `infr serve --ctx 8192`, one slot.
- `Qwen3-1.7B-Q4_K_M.gguf` (unsloth). Sampling from the arch-family table:
  `temp 0.6`, `top_k 20`, `top_p 0.95`.
- One fixed question that provokes long reasoning — `Find every integer solution
  to x^2 + y^2 = 2024. Work through it carefully.` — streamed, `max_tokens 1200`,
  two runs per configuration. "First answer token" is the arrival of the first
  `content` delta; reasoning deltas arrive from the first token.

## Results

| Request                  | Budget | Forced close at | Reasoning chars | First answer token | Total  | Finish  |
| ------------------------ | -----: | --------------: | --------------: | -----------------: | -----: | ------- |
| no thinking              |      — |               — |               0 |           73–122ms | 7.1–10.4s | stop/length |
| `reasoning_effort: minimal` |  512 |       512 (×2) |     1 477–1 629 |          4.13–4.15s | 7.4–10.2s | stop  |
| `reasoning_effort: low`     |  512 |       512 (×2) |     1 539–1 554 |          4.14–4.20s | 7.3–8.4s  | stop  |
| `reasoning_effort: medium`  | 2048 |      never |     3 720–3 915 |             never | 10.3s  | length |
| `reasoning_effort: high`    | 8192 |      never |     3 709–3 951 |             never | 10.2–10.7s | length |
| `think_max_tokens: 256`  |    256 |       256 (×2) |      770–859 |          2.05–2.08s | 7.9–10.2s | stop/length |
| `think: true`, no budget |      — |      never |          14 172 |             never | (5 000-token cap) | length |

Decode rate over these runs: 113–122 t/s with and without a budget, no visible
difference.

## What the numbers say

**The budget closes the block exactly at the limit.** Every budgeted run logged
`reasoning budget spent; forcing the think block's close` once, with
`reasoning_tokens` equal to its limit — 512 for `low` and `minimal`, 256 for the
exact count — and the answer followed the close in the same reply. The forced
ids are the template's own close sequence, not an approximation, so the model
resumes as if it had closed the block itself.

**The level table is the difference between an answer and no answer.** Without a
budget this question never terminates: the model spent all 5 000 tokens of a
larger window on reasoning (14 172 chars) and returned `finish_reason: length`
with empty content. At `low` it answers after ~4.2 s of reasoning; at 256 tokens
it starts answering after ~2.1 s. `medium` and `high` behaved like the unbudgeted
run here because 1 200 output tokens is not enough to reach their limits on this
question — the budget is a cap, not a target.

**It costs the pipelined decode chain, and at this size that is not measurable.**
A budgeted reply decodes on the ordinary per-token path because the counter has to
see each id (chained decode hands the host no such point). Decode over the runs
above stays at 113–122 t/s either way: a 1.7B Q4_K_M decode on this card is
DRAM-bandwidth bound, and the chain needs two or more dependent graphs to help at
all, so the loss hides inside run-to-run variation. Expect it to show on models
and depths where the chain does pay.

**The streamed shape is clean.** Reasoning deltas arrive first and stop at the
forced close; `content` deltas follow. Neither stream carries the ` thinking`
marker, and no partial close leaks — the splitter's existing hold-back covers a
close that lands across several tokens, which is what Qwen3's single-token close
becomes when the tokenizer splits it.

## Reproduce

```sh
infr serve Qwen3-1.7B-Q4_K_M.gguf --ctx 8192 &
curl -sN localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"q",
  "messages":[{"role":"user","content":"Find every integer solution to x^2 + y^2 = 2024. Work through it carefully."}],
  "reasoning_effort":"low",
  "max_tokens":1200,
  "stream":true
}'
# The close is forced: RUST_LOG=info shows
#   reasoning budget spent; forcing the think block's close level="low" limit=512 reasoning_tokens=512
```

`think_max_tokens` overrides the level for one request; `sampling.think_budget_low`
/ `_medium` / `_high` and `sampling.think_budget_tokens` set the numbers
process-wide.

## Still open

- LFM3-1.7B Q4_K_M on this card, as the plan specifies: its template differs
  (thinking is off by default for small templates), so the interesting row is
  `low` against its own unbudgeted default rather than against `none`.
- Answer quality at `low` versus an unbudgeted run that is allowed to finish:
  this question is a poor judge, since the unbudgeted model never finishes at any
  reasonable length.
