# Phase 03 — Gemma 2 (`gemma2`)

**Why:** `gemma-2-2b` is small and widely used, and infr already runs Gemma 3
and 4, so most of the block exists.

## What the reference does

Read in `src/models/gemma2.cpp` (llama.cpp `5676dfbe0`) as a diff against the
`llama` builder:

- Token embeddings scaled by `sqrt(n_embd)`; output tied to the embedding.
- **Sandwich norms:** `attn_norm` before and `attn_post_norm` after attention,
  `ffn_norm` before and `ffn_post_norm` after the FFN.
- Parallel **GELU** FFN.
- Q scaled before attention by `f_attention_scale`, which is `1/sqrt(head_dim)`
  except for the 27B model, where it is `1/sqrt(n_embd / n_head)`.
- **Sliding-window attention** on a pattern of 2 (`load_swa_pattern(ml, 2)`),
  default window 4096, with a separate SWA RoPE base when the file gives one.
- **Attention logit soft-capping** (`attn_soft_cap`, key
  `attn_logit_softcapping`) and **final logit soft-capping**
  (`final_logit_softcapping`).
- 26 layers = 2B, 42 = 9B, 46 = 27B.

## What infr already has

From `Config`: sandwich norms, GeGLU, dual RoPE and SWA for `gemma3`, and
`final_softcap` (its doc names gemma2). **Not found by a grep for `softcap`:**
an attention-logit soft-cap. That is the one piece that may need a change inside
the attention kernels (CPU, Vulkan, Metal), and it is the cost driver.

## Steps

1. Dump `gemma-2-2b-it`; confirm the keys above and whether
   `attn_logit_softcapping` is non-zero in the shipped file.
2. Read how `build_attn` applies `attn_soft_cap` in llama.cpp
   (`src/llama-graph.cpp`) — where in the QK → softmax chain the `tanh` sits.
3. Decide the shape of the attention soft-cap: a parameter on the existing
   attention op (preferred, if every kernel can take it) or a refusal of the
   flash kernels for this family. Write the decision here before coding.
4. CPU first, parity cases in `seam_op_parity.rs`, then Vulkan and Metal.
5. Recipe verification, README row, CHANGELOG.

## Risks

- A soft-cap inside the fused / flash attention kernels touches the hottest code
  in the engine; it must be a no-op for every other family, proven by the
  existing parity suites and a decode timing in a quiet window.
- Metal cannot be run locally; its arm is verified by the macOS CI job only.
