# Phase 02 — ERNIE 4.5 (`ernie4_5`, then `ernie4_5-moe`)

**Why:** `plan.md` already names ERNIE 4.5 21B-A3B as a candidate. The dense
0.3B model shares the arch file, is tiny, and proves the block and tokenizer
before the MoE model is downloaded.

## What the reference does

Read in `src/models/ernie4-5.cpp` (llama.cpp `5676dfbe0`) as a diff against the
`llama` builder, not line by line:

- **Dense (`ernie4_5`, 18 layers = 0.3B):** RMS pre-norms, separate Q/K/V, RoPE,
  attention scale `1/sqrt(head_dim)`, parallel SiLU FFN. The only metadata
  beyond what `llama` reads is `rope.dimension_sections`, which the text path
  does not appear to use (it is the multimodal RoPE split) — **to confirm**.
- **MoE (`ernie4_5-moe`, 28 layers = 21B-A3B, 54 = 300B-A47B):** the first
  `leading_dense_block_count` layers are dense; later ones carry `ffn_gate_exps`
  / `ffn_up_exps` / `ffn_down_exps`, an optional `ffn_exp_probs_b` router bias,
  and a shared expert (`ffn_*_shexp`) when `expert_shared_feed_forward_length`
  is set. `expert_feed_forward_length` is read as a per-layer array and
  `interleave_moe_layer_step` is required.
- **Not read:** `src/models/ernie4-5-moe.cpp` (128 lines), which holds the MoE
  forward pass — router activation (softmax or sigmoid), whether weights are
  renormalised over the selected experts, and how the shared expert is summed.

## What infr already has

Batched-expert MoE with a shared expert, gated or plain (`Config::shexp_ff`,
`shexp_gated`), and leading dense layers for the DeepSeek line. Whether the
router-bias tensor and the interleave step have counterparts was not checked.

## Steps

1. Dense: dump the 0.3B GGUF, add the arch, run it through the recipe (CPU
   logits against llama.cpp, golden, GPU seam test). Expect recipe steps 3 and 6
   only.
2. Read `ernie4-5-moe.cpp` in full and write the MoE delta into this file before
   any code.
3. MoE: config and weight loading for the mixed dense / MoE stack; reuse the
   batched-expert path. Use the loose GPU comparison (top-5 overlap and a cosine
   floor), as the recipe says for MoE.
4. 21B-A3B at Q4_K_M is about 13 GB: it fits the 7900 XTX resident and needs
   streaming on the integrated GPU. Check disk space before pulling.

## Risks

- A router or shared-expert rule that matches none of the existing MoE variants
  would mean new graph work on three backends; step 2 decides that.
- The tokenizer `pre` value may be one infr does not map yet.
