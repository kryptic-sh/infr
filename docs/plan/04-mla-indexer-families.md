# Phase 04 — the MLA + indexer families (Mistral Large 4 first)

**Why:** the newest large models in the reference share one skeleton, and infr
already has most of it from the DeepSeek line: multi-head latent attention
(`deepseek2`), the lightning indexer (`deepseek32`) and MoE with shared experts.
Listing each file's metadata keys and tensors (llama.cpp `5676dfbe0`) shows:

| Arch (`src/models/`)       | Added      | Lines | On top of MLA + indexer + shared-expert MoE                                                                          |
| -------------------------- | ---------- | ----- | -------------------------------------------------------------------------------------------------------------------- |
| `mistral-large4`           | 2026-10-29 | 508   | per-head attention scale and bias tensors (`attn_ssmax`, `indexer_ssmax`), a per-layer RoPE pattern, a KV source map |
| `glm5-next`                | 2026-09-30 | 1087  | hyper-connections, a pooled-key indexer (`kpool`), a KDA attention variant, SwiGLU clamps                            |
| `kimi-k31` (and `kimi-k3`) | 2026-11-06 | 701   | KDA, residual score tensors, a latent expert length, a custom activation (`situ`), recurrent memory                  |
| `deepseek41`               | 2026-10-25 | 933   | follows `deepseek4`: KV compressor, hyper-connections, n-gram embedding heads (`ple`, `engram`), attention sinks     |
| `qwen4exp`                 | 2026-08-27 | 1505  | recurrent (SSM) layers, hyper-connections, per-layer embeddings, compress ratios                                     |

**Only the key and tensor names were read**, not one forward pass. The right
column is what those names imply and nothing more.

## Order

1. **Finish `deepseek4`** — already the first candidate in `plan.md`, tracked as
   B-DSV4-WIRING and B-DSV4-REAL: the compressed-KV tiers. `deepseek41` and
   `qwen4exp` both read compress ratios, so this unblocks the most.
2. **`mistral-large4`** — the shortest file and the smallest delta by name. Read
   it in full first; the unknowns are what `ssmax` computes (the graph scales
   positions through `build_inp_attn_scale`), the KV source map (layers sharing
   another layer's KV), and the model's size, which the builder does not name
   (`type` is always unknown) — whether any published file fits the 7900 XTX
   resident or needs streaming is a step-0 question.
3. **Hyper-connections**, as a shared building block: they appear in
   `glm5-next`, `deepseek41` and `qwen4exp`. Build once, with a synthetic-model
   test in the style of `tests/synthetic_deepseek2.rs`, before any of the three.
4. **Then one of `glm5-next`, `kimi-k31`, `deepseek41`, `qwen4exp`**, chosen
   when step 3 lands by which has a file that can be run here and which the user
   wants most. Each gets its own phase file then; none is scoped yet.

## What makes this tractable

- The synthetic-model harness (`docs/synthetic-models.md`) exists because the
  DeepSeek files were too large to develop against; the same approach covers
  these.
- These are large models. With streaming (phase 05) they run on this hardware at
  whatever the physics allow, which is the reason phases 04 and 05 are worth
  doing together.

## Risks

- Every row above is a guess from names until the file is read. `mistral-large4`
  is read first precisely to find out how good such guesses are.
- No oracle for a model too large to run under llama.cpp on this machine; the
  DeepSeek work handled that with synthetic models and a reference logits dump
  taken elsewhere (`INFR_LLAMA_DUMP`).
