# Phase 07 — Mistral Large 4 (`mistral-large4`)

**What it is:** DeepSeek V3.2's design — multi-head latent attention (MLA), the
lightning indexer, MoE with shared experts — with five additions and three
removals. infr has the base from `deepseek2` / `deepseek32`. Only one addition
needs a new op.

**Where this comes from:** a read-only pass on 2026-11-13 over
`src/models/mistral-large4.cpp` (in full), `deepseek32.cpp` (in full),
`deepseek2.cpp`, the graph, KV-cache and converter code it uses in llama.cpp
`5676dfbe0`, against infr's config, runner and ops. The reader's line references
are not repeated here; **re-read each claim at its site before writing code
against it.** Not read: the MLA and indexer shaders and Metal kernels, the
`dflash` draft model, most of infr's handle plumbing.

## The delta

**Added over `deepseek32`:**

1. **SSMax: a per-head, position-dependent scale on the query.** For head `h` at
   absolute position `p` the factor is `s[h] · ln(p + 1) + b[h]`, with `s` and
   `b` learned per head (`attn_ssmax.weight` / `.bias`), and the query is
   multiplied by it before attention. The indexer's query gets its own
   (`indexer.ssmax`), applied before its ReLU — the factor can be negative, so
   the order matters. `n` is the count of causally visible keys by position, not
   the number of keys the top-k kept. **This is the new op** (CPU, Vulkan,
   Metal): infr has no attention temperature of any kind today, and the
   context-cap trick used for Mistral-Large-3 cannot work here because the
   factor differs from 1 at position 0.
2. **A per-layer RoPE pattern** (`attention.rope_pattern`, a bool array): NoPE
   layers use the whole head width without rotation. The reference keeps the
   cache layout uniform by zero-padding the rope tail, and mirroring that needs
   no kernel change.
3. **A KV source map** (`attention.kv_source_map`): a layer may read another
   layer's MLA cache and then writes none of its own. infr's only cross-layer KV
   read today is gemma4's prefix rule.
4. **Shared indexer layers** (`attention.indexer.types`): a "shared" layer
   reuses the most recent "full" layer's top-k selection. Graph assembly only.
5. **Learned RoPE frequencies** (`rope_freqs.weight`, required) combined with
   the YaRN ramp, and a **final logit soft-cap** (`final_logit_softcapping`).
   infr has both pieces (`rope_freqs` for Llama, `Op::Softcap` for Gemma), gated
   on other archs.

**Removed relative to `deepseek32`:** the query LoRA (the query comes from one
`attn_q`, which is the "lite" path infr already has), the router bias
(`exp_probs_b` is never created), and the YaRN magnitude boost on the attention
score — the score scale is plain `1/sqrt(key_length_mla)`.

**Differences that are parameters, not new work:** the indexer query comes from
a LayerNorm of the hidden state with a bias (not from the query LoRA), its
epsilon is read from `attention.layer_norm_epsilon` (infr hardcodes the
deepseek32 value), its RoPE is the normal form with the learned factors and is
skipped on NoPE layers; `rope.scaling.yarn_log_multiplier` is used raw, where
`deepseek2` divides it; `expert_gating_func` is required; `output.weight` is
required (no tied fallback).

## What is not known without a file

Every dimension, the actual patterns (does any released file share KV?), the
model's size and whether it fits the 7900 XTX resident, whether its MLA width
fits the Vulkan MLA kernel's cache-row cap, and the tokenizer: the converter
writes a `tekken` pre-tokenizer for Mistral vocabularies, which infr does not
map (backlog, "Pre-types with a regex infr does not have"). The reference
builder names no model size at all.

There is unlikely to be a file small enough to run under llama.cpp here, so
correctness rests on synthetic models plus a reference logits dump taken
elsewhere, as it did for DeepSeek.

## Stages

Each is testable alone through the synthetic harness
(`tests/synthetic_deepseek2.rs`). The harness writer first needs bool and
integer array metadata, which it cannot emit today; that is a small
`infr-testkit` change and part of stage 1.

1. **Config and load, refusing at graph build.** The arch const and flag
   (widening `deepseek2` the way `deepseek32` does, auditing every read of that
   flag), the new keys, per-layer tensor presence and shape. Tests: every tensor
   consumed, a misshaped one refused, an invalid KV source map refused, a shared
   indexer on layer 0 refused, bool arrays parsed.
2. **The SSMax op:** CPU, a parity case against a from-definition reference,
   then Vulkan and Metal. Tests: `s = 0, b = 1` is bit-identical to no op; a
   negative factor flips the indexer's selection.
3. **The uniform graph:** every layer rope, own KV, full indexer. Direct query,
   LayerNorm indexer query with normal RoPE, the combined frequency divisors,
   the plain score scale, the soft-cap. Tests: SSMax at identity equals the same
   model without it; CPU against Vulkan.
4. **NoPE layers** through zero-padded tails. Tripwire: an all-NoPE model's
   logits do not move with `rope.freq_base`; a rope layer's do.
5. **Shared indexer layers.** Tests: with `top_k` at least the context, the type
   is irrelevant; below it, full and shared differ.
6. **The KV source map:** per-layer cache rows including a layer with none, the
   generalised source layer, MLA reading the source's cache. A layer with no K
   side has never existed in infr, so every consumer of the cache shape
   (allocation, fork, seeding, the VRAM estimate, the Vulkan row arithmetic)
   gets checked. Mechanism test: zero the source layer's output projection so it
   can only matter through the shared cache, then perturb its KV weights.
7. **A real file:** the tekken pre-tokenizer, placement and streaming, a
   reference dump comparison.

## Risks

- Stage 2 puts a new input into the hottest kernels' neighbourhood. It should be
  a separate multiply on the query before `Op::Mla`, not a change inside the
  attention kernels; confirm that is equivalent (the factor is per head and the
  absorption and rotation are linear) before choosing.
- Stage 6's zero-width K side is the change most likely to break something
  unrelated.
- Metal cannot be run locally; its arms are first compiled by CI.
