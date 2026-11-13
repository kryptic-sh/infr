# Phase 04 — Phi-3 / Phi-4 and GLM-4 (`phi3`, `glm4`)

**Why together:** both use a **fused SwiGLU** FFN — one `ffn_up` of width
`2 * n_ff` whose output is split into gate and value — which no family infr runs
today appears to use. Building that once serves both, and `glm4` is the block
`glm4-moe` (GLM-4.7-Flash, in `plan.md`) builds on.

## What the reference does

Read in `src/models/phi3.cpp` and `src/models/glm4.cpp` (llama.cpp `5676dfbe0`)
as diffs against the `llama` builder:

- **Both:** `ffn_up` is `{n_embd, 2 * n_ff}` with no `ffn_gate`; `glm4` calls
  `build_ffn` with `LLM_FFN_SWIGLU, LLM_FFN_SEQ`.
- **`phi3`:** Q/K/V may be one fused `attn_qkv` tensor (`create_tensor_qkv` with
  `TENSOR_NOT_REQUIRED`); Q is scaled by `1/sqrt(head_dim)` before attention;
  optional LongRoPE factor tensors (`rope_factors_long` / `rope_factors_short`);
  norms carry an optional bias (`attn_norm_b`); sliding-window attention is read
  from the file and then **disabled** with a warning. 24 layers = 1B, 32 = 3B
  (Phi-3-mini / Phi-4-mini), 40 = 14B.
- **`glm4`:** post-norms (`attn_post_norm`, `ffn_post_norm`) as well as
  pre-norms; layers past `n_layer` are next-token-prediction heads loaded with
  `TENSOR_SKIP`; multimodal RoPE (`use_mrope`) for the vision variants, which a
  text-only port can refuse. 17 layers = GLM-OCR, 40 = 9B, 61 = 32B.
- **Not read:** which half of the fused `ffn_up` output is the gate (in the
  `LLM_FFN_SWIGLU` arm of `build_ffn`), and the body of either graph beyond the
  diff.

## What infr already has

Fused-QKV weight slices (the Vulkan adapter has `w_off` handling for them), RoPE
factor tensors for Llama (applied since 2026-11-12), and post-norms for Gemma. A
fused SwiGLU was not found by a grep of `config.rs`.

## Steps

1. Dump Phi-4-mini (3.8B, Q4_K_M about 2.5 GB) and a small `glm4` model.
2. Read the SwiGLU arm of `build_ffn`; write the split order here.
3. Fused SwiGLU: prefer composing existing ops (one `Linear` to `2 * n_ff`, then
   a gated activation reading two slices of one tensor) over a new op; add one
   only if a slice cannot be expressed. CPU first, then a parity case.
4. `phi3`: arch, config, fused QKV, pre-scaled Q, LongRoPE factors. Recipe
   verification on Phi-4-mini.
5. `glm4`: arch, config, post-norms, skip the next-token heads, refuse `mrope`
   files by name. Recipe verification.
6. README rows, CHANGELOG. Then plan `glm4-moe` as its own phase.

## Risks

- The tokenizer and chat template differ between Phi-3 and Phi-4; the
  end-of-turn marker must be covered (`add_chat_eos`).
- The LongRoPE factors switch between the long and short set by context length
  in the reference; read that rule before reusing the Llama path.
