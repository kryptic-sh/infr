# Plans

One line per phase, in build order within each track. A phase file says what the
phase builds and how it is verified; what was raised and not done lives in
[`../backlog.md`](../backlog.md). The whole-system description, the recipe for
adding an architecture and the larger candidates (DeepSeek V4, Nemotron-H) are
in [`../plan.md`](../plan.md).

## Model families

**What gets a phase:** recent families people will actually run, and older ones
only when they are genuinely little work. Each is taken through the full recipe:
CPU reference first, a logit comparison with llama.cpp, a CPU golden, then the
GPU seam test.

| Phase | What                                                                                                  | Status                |
| ----- | ----------------------------------------------------------------------------------------------------- | --------------------- |
| 01    | SmolLM3 (`smollm3`): near config-only                                                                 | done                  |
| 02    | [ERNIE 4.5](02-ernie4-5.md) (`ernie4_5`, then `ernie4_5-moe`)                                         | done; Metal awaits CI |
| 03    | [LFM3](03-lfm3.md) (`lfm3`): recent, 1.7B                                                             | done; Metal awaits CI |
| 04    | [The MLA + indexer families](04-mla-indexer-families.md), Mistral Large 4 first                       | not started           |
| 07    | [Mistral Large 4](07-mistral-large4.md) (`mistral-large4`): scoped, staged                            | scoped                |
| 09    | [DeepSeek V4 compressed tiers](09-deepseek-v4-compressed.md) (`deepseek4`), then DeepSeek performance | S0-S4b, S5a done      |

## Streaming and paging

The standing goal: any model size on any hardware, disk to DRAM to VRAM.

| Phase | What                                                                                  | Status         |
| ----- | ------------------------------------------------------------------------------------- | -------------- |
| 05    | [Streaming for any model on any hardware](05-streaming-any-size.md)                   | not started    |
| 06    | [Pager eviction: cost and policy](06-pager-eviction.md)                               | steps 1–2 done |
| 08    | [A streamed model that fits runs at resident speed](08-streamed-at-resident-speed.md) | steps 0–4 done |

## Numerics

Priority: phase 11 goes ahead of the rest of phase 10, which needs the fixed CPU
reference to judge tuned values against.

| Phase | What                                                                                               | Status                                                                                                                                                          |
| ----- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 11    | [CPU output bit-identical on every platform, GPUs within tolerance of it](11-cpu-bit-identical.md) | S0-S4 and S6 done (S5 in S2), G1-G3 done; G4 partly done (the Qwen3 golden retired, `infr tune check` done); **the tests of G4 that need a row first are next** |

## Vulkan backend

| Phase | What                                                                                            | Status                                                                                                                                              |
| ----- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 10    | [Kernels with per-device tunable parameters](10-tunable-kernels.md), one tuning table           | S1 done                                                                                                                                             |
| 12    | [Vulkan devices without matrix units or fast int8 dot (GCN class)](12-low-capability-vulkan.md) | S0 answered by the reporter; S1 and S2 done; the fork's PRs 105 to 107 (the subjects of S5, S3 and S7) merged, their review pending; S4 and S6 open |

## Caps, refusals and open decisions

| Phase | What                                                                                                    | Status                                                                                                                                      |
| ----- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 13    | [Replace the caps, refusals and open decisions with implementations](13-caps-refusals-and-decisions.md) | S1 to S4 and S9 done, S5 first half done, S11 block-IO items (B35, B30) done; S5 second half, S6 to S8, S10 and the rest of S11 not started |

## Notes on the model phases

Every phase was scoped on 2026-11-13 from the llama.cpp checkout at
`~/Projects/mxaddict/llama.cpp` (commit `5676dfbe0`). Phase 02 was diffed
against `src/models/llama.cpp`; 03 and 04 were scoped from the keys and tensors
their files name; 07 from a full read of its builder. Each phase file states
which. Phase 02's dense GGUF and its 21B-A3B MoE GGUF, and phase 03's LFM3 1.7B,
have been dumped and run; no GGUF of any other open phase's family has been yet:
step 0 of the recipe comes first in each phase and may change its estimate.

Considered and declined, with the reason, so they are not re-proposed:

- **`gemma2`, `phi3` (Phi-3 / Phi-4-mini), `glm4`:** older lines, and none is
  cheap. Gemma 2 needs a soft-cap inside the attention kernels on three
  backends; Phi-3 and GLM-4 need a fused SwiGLU FFN that nothing else here uses.
  Gemma 3 and 4 are already supported. Scoped on 2026-11-13 and dropped the same
  day on the rule above; the scoping is in `git log` for
  `docs/plan/03-gemma2.md` and `docs/plan/04-phi3-glm4.md`.
- **`exaone4`, `olmo2`:** post-norm blocks with QK-norm, each with its own RoPE
  override; not recent, not requested.
- **`granite`** (Granite 3.x / 4.0): llama-shaped plus four scale factors and a
  deepstack embedding mapping that was not read through. Revisit only if it
  turns out to be config-only once dumped.
- **`glm4-moe`** (GLM-4.7-Flash, named in `plan.md`): superseded as a target by
  `glm5-next`, which is in phase 04's table.
