# Phase 01 — SmolLM3 (`smollm3`)

**Why first:** the smallest delta of any family on the list, and a 3B model that
fits both GPUs on the Windows box (Q4_K_M is about 2 GB; check the repo before
pulling).

## What the reference does

Read in `src/models/smollm3.cpp` (llama.cpp `5676dfbe0`), in full:

- The block is the `llama` block: RMS pre-norms, separate Q/K/V, RoPE, a
  parallel SiLU FFN (`ffn_gate`, `ffn_up`, `ffn_down`), tied output embedding
  when `output.weight` is absent.
- The one difference: `n_no_rope_layer_step` is **hardcoded to 4**, not read
  from the file, and layer `il` applies RoPE only when `(il + 1) % 4 != 0`.
- `kq_scale` is `1/sqrt(head_dim)` unless `f_attention_scale` is set.

## What infr already has

`Config::no_rope_step` and the predicate beside it implement the same
`(il + 1) % step == 0` rule for `llama4`. **Not yet read:** whether that
predicate only skips RoPE or also switches the other things the NoPE layers of
`llama4` do (the comment near it mentions an attention temperature). If it does
more, SmolLM3 needs the skip separated from the rest.

## Steps

1. Pull a SmolLM3 GGUF and dump it (`cargo run -p infr-gguf --example dump`).
   Confirm the arch string, that no `no_rope_layer_step` key exists, the
   tokenizer model and `pre` value, and the chat template.
2. `arch.rs`: add `SMOLLM3`. `Config::from_gguf`: the arm, with
   `no_rope_step = 4` set from the arch (the reference hardcodes it).
3. Make the NoPE layers skip RoPE and nothing else. It is a graph-builder
   branch, not a new op, so every backend gets it at once.
4. Verify per the recipe: logits against `llama-cli` on the CPU backend, a CPU
   golden in `tests/cpu_backend.rs` (read the text before blessing), a
   `gpu_seam_matches_cpu_smollm3` test, then `infr run`.
5. README supported-models row, CHANGELOG entry.

## Risks

- The chat template uses thinking toggles (`/think`, `/no_think`); check that
  the `no_think` handling and `add_chat_eos` cover its end-of-turn marker.
- llama.cpp must be built on the Windows box first for the oracle
  (`build/bin/llama-cli`); it is a source checkout only as of 2026-11-13.
