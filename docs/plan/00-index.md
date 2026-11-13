# Plans

One line per phase, in build order within each track. A phase file says what the
phase builds and how it is verified; what was raised and not done lives in
[`../backlog.md`](../backlog.md). The whole-system description, the recipe for
adding an architecture and the larger candidates (DeepSeek V4, Nemotron-H) are
in [`../plan.md`](../plan.md).

## Model families

These phases add families whose smallest published model fits the development
machines, so each can be taken through the full recipe: CPU reference first, a
logit comparison with llama.cpp, a CPU golden, then the GPU seam test.

| Phase | What                                                          | Status      |
| ----- | ------------------------------------------------------------- | ----------- |
| 01    | [SmolLM3](01-smollm3.md) (`smollm3`)                          | not started |
| 02    | [ERNIE 4.5](02-ernie4-5.md) (`ernie4_5`, then `ernie4_5-moe`) | not started |
| 03    | [Gemma 2](03-gemma2.md) (`gemma2`)                            | not started |
| 04    | [Phi-3 / Phi-4 and GLM-4](04-phi3-glm4.md) (`phi3`, `glm4`)   | not started |

## Streaming and paging

The standing goal: any model size on any hardware, disk to DRAM to VRAM.

| Phase | What                                                                | Status      |
| ----- | ------------------------------------------------------------------- | ----------- |
| 05    | [Streaming for any model on any hardware](05-streaming-any-size.md) | not started |
| 06    | [Pager eviction: cost and policy](06-pager-eviction.md)             | not started |

## Notes on the model phases

Every phase was scoped on 2026-11-13 from the llama.cpp checkout at
`~/Projects/mxaddict/llama.cpp` (commit `5676dfbe0`), by diffing the family's
`src/models/<arch>.cpp` against `src/models/llama.cpp`. **No GGUF of any of
these families has been dumped yet**: step 0 of the recipe comes first in each
phase and may change its estimate.

Considered and left for later, with the reason:

- **`granite`** (Granite 3.x / 4.0): llama-shaped plus four scale factors
  (`logit_scale`, `residual_scale`, `embedding_scale`, `attention_scale`), and
  its builder now also carries a deepstack embedding mapping that was not read
  through. Cheap once the scales exist; no shared work with the phases above.
- **`exaone4`, `olmo2`**: post-norm blocks with QK-norm. infr has both pieces
  for other families, but neither family is a common request and each has its
  own RoPE parameter override (`get_rope_params`).
- **`lfm2`**: small models (350M to 1.2B), but the builder is three times the
  length of the others and adds convolution blocks. A phase of its own once the
  cheaper families are in.
- **`glm4-moe`** (GLM-4.7-Flash, named in `plan.md`): depends on the `glm4`
  block from phase 04; planned after it lands.
