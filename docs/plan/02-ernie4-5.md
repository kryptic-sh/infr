# Phase 02 — ERNIE 4.5 (`ernie4_5`, then `ernie4_5-moe`)

**Why:** `plan.md` names ERNIE 4.5 21B-A3B as a candidate. The dense 0.3B model
shares the arch file, is tiny, and proved the block and tokenizer before the MoE
model is downloaded.

**Status:** the dense `ernie4_5` arch is done (steps 1–2 below). The MoE
`ernie4_5-moe` arch is scoped, not started.

## Dense `ernie4_5`: done

### The file

`unsloth/ERNIE-4.5-0.3B-PT-GGUF`, `ERNIE-4.5-0.3B-PT-Q4_K_M.gguf`, 240645600
bytes, pulled with `infr pull` into the HF cache. The F16 file of the same repo
(723955168 bytes) was pulled too, for the oracle and the GPU seam test.

What `cargo run -p infr-gguf --example dump` shows, against the guesses this
plan made before:

- `general.architecture = "ernie4_5"`, 164 tensors, 18 blocks.
- Keys: `block_count`, `context_length` (131072), `embedding_length` (1024),
  `feed_forward_length` (3072), `attention.head_count` (16),
  `attention.head_count_kv` (2), `attention.key_length` / `value_length` (128),
  `attention.layer_norm_rms_epsilon`, `rope.freq_base` (500000). **No
  `rope.dimension_sections`** and no `rope.dimension_count`, so the plan's open
  question about the multimodal sections does not arise for this file; reading
  `build_rope_impl` confirms they are only used under MROPE, which `ernie4_5`
  (NORM rope) never takes.
- The head width is 128 while `n_embd / n_head` is 64: `attn_q` is
  `[1024, 2048]`. infr already reads `attention.key_length`.
- Tensors per block: `attn_norm`, `attn_q`/`attn_k`/`attn_v`, `attn_output`,
  `ffn_norm`, `ffn_gate`/`ffn_up`/`ffn_down`. No `attn_output.bias` (the
  reference loads one when present and never applies it), no `output.weight`
  (tied to `token_embd`).
- Tokenizer: `tokenizer.ggml.model = "llama"` (SentencePiece),
  `tokenizer.ggml.pre = "default"`, 103424 tokens, BOS `<s>` (1), EOS `</s>`
  (2), `add_bos_token = true`, no `add_space_prefix` key. The chat template uses
  `<|begin_of_sentence|>` (100273) and `<|end_of_sentence|>` (100272), both
  CONTROL tokens.

### What it took

The `llama` block verbatim, so config-only, as the plan expected:
`arch::ERNIE4_5`, the `qk_norm = false` arm in `Config::from_gguf`, its row in
`applied_weight_scales` (`llama_model_ernie4_5::graph` applies the Q/K/V,
`attn_output` and `output` scales, no FFN scale), and an `arch_sampling` row
(temp 0.8 / top_p 0.8 from the checkpoint's `generation_config.json`). The
converter writes Q/K without llama's permute and the reference ropes them NORM,
so no row permute applies. SentencePiece ignores `tokenizer.ggml.pre`; nothing
in the tokenizer had to change.

One change reached beyond ERNIE: the reference loads `attn_q/k/v.bias` as
optional for every arch built through `create_tensor_qkv` and adds whichever are
present, while infr loaded them only for `qwen2` and left them unread elsewhere.
`Config::qkv_bias` now follows the tensors for those archs
(`qkv_projection_biases`): all three on every layer turn the bias path on, a
partial set is refused naming the missing tensor, and qwen35/qwen35moe refuse
any. Of the cached GGUFs only the Qwen2.5 file carries them.

The BOS differs from llama.cpp on this file, and infr's choice matches the
reference (B166).

### Verified (CPU, Windows box, llama.cpp `5676dfbe0`)

- **Tokenizer:** seven strings (space prefix and whitespace runs, CJK, accented
  Latin, Japanese, Korean, an emoji the vocabulary holds whole, byte fallback
  for a skin-tone modifier and a ZWJ family emoji, the chat CONTROL tokens,
  `<mask:1>` and `<|LOC_5|>`) give llama.cpp's `llama-tokenize` ids exactly
  (`ernie4_5_tokenizes_as_llama_cpp`).
- **Chat template:** infr's render of a system + user + assistant + user
  conversation is byte-identical to llama.cpp's jinja render
  (`ernie4_5_render_matches_the_reference` in `infr-chat`). llama.cpp's jinja
  path is its default here; its built-in (`--no-jinja`) path has no rule in
  `llm_chat_detect_template` that matches this template (read, not run).
- **Logits** (`cpu_prefill_matches_llama_debug_dump` on `llama-debug` dumps; the
  test now also prints the max absolute logit difference and the shared top-5
  count):

  | file   | prompt                            | max \|Δlogit\| | shared top-5 | top-1 |
  | ------ | --------------------------------- | -------------: | -----------: | ----- |
  | F16    | `The capital of France is`        |   0.0037351847 |          5/5 | same  |
  | F16    | `北京是中国的首都，它的历史`      |    0.004545212 |          5/5 | same  |
  | F16    | `Once upon a time, … there lived` |   0.0057168007 |          5/5 | same  |
  | Q4_K_M | `The capital of France is`        |      1.1721492 |          3/5 | same  |
  | Q4_K_M | `北京是中国的首都，它的历史`      |      0.9078784 |          4/5 | same  |

  The Q4_K_M spread is the size of llama.cpp's own: its CPU and Vulkan (7900
  XTX) logits on the English Q4_K_M prompt differ by up to 1.2040534, and on F16
  the two engines agree to under 0.006.

- **Greedy:** on F16, infr's 40-token continuation of llama.cpp's own prompt ids
  is text-identical to `llama-completion --temp 0` for the story prompt and the
  Chinese prompt. On Q4_K_M with raw prompts (`The capital of France is` and the
  Chinese one), the two engines agree for 8 tokens and then split, both times on
  a new line that starts with `0`. On the chat-rendered ocean prompt (Q4_K_M,
  llama.cpp told not to add its BOS so the ids are the same), llama.cpp CPU,
  infr's production CPU path and the Vulkan seam on the 7900 XTX produce the
  same 32 tokens.
- **Mutation:** making `ernie4_5` permute Q/K rows (`permute_qk_neox`) turns
  `ernie4_5_parses_as_the_llama_block` and `cpu_golden_ernie4_5` red and moves
  the F16 max |Δlogit| from 0.0057 to 4.27 — but the generic oracle test still
  passes (B169).

### Tests added

All in `crates/infr-llama/tests/cpu_backend.rs` unless named otherwise; each was
shown to go red:

- `ernie4_5_parses_as_the_llama_block` (`config.rs` unit test): red when
  `ernie4_5` permutes Q/K.
- `ernie4_5` rows in `unapplied_weight_scales_are_refused`.
- `ernie4_5_render_matches_the_reference` (`infr-chat`): red when the render
  context defines `cls_token`.
- `ernie4_5_tokenizes_as_llama_cpp` (ignored, model-gated): red when an absent
  `add_space_prefix` key stops meaning true for SentencePiece.
- `cpu_golden_ernie4_5` (ignored, model-gated, Q4_K_M; hashes captured on
  Windows only, no Linux hash computed — B167): red when `ernie4_5` permutes
  Q/K.
- `gpu_seam_matches_cpu_ernie4_5` (ignored, Vulkan, F16 file, 32 tokens): green
  on Vulkan0 (RX 7900 XTX) and Vulkan1 (integrated). Red on the Q4_K_M file on
  both devices (a near tie, B167), and red on Vulkan0 with the Vulkan rope base
  cut 1000×.
- `gpu_seam_matches_cpu_ernie4_5_q4_k_m` (ignored, Vulkan, Q4_K_M, "What is the
  chemical symbol for gold?", 9 tokens): green on both GPUs. Chosen by a scan of
  seventeen chat prompts for top-1 margins along the answer; this one leads the
  runner-up by at least 1.630 on infr's production CPU prefill, 1.493 on Vulkan0
  and 1.461 on Vulkan1 at every one of its 9 tokens. Each of the others split
  between the CPU and a GPU, had a step under 0.6 on some path (down to 0.003),
  or answered in three tokens or fewer. Red on Vulkan0 with the Vulkan attention
  scale halved; not red with the rope base cut 1000×, which the F16 case
  catches.
- `qkv_biases_follow_the_tensors` (`config.rs`) and
  `qkv_biases_are_added_on_a_non_qwen2_arch` (`seam::runner`, synthetic
  `ernie4_5` model: zero biases leave the logits exact, nonzero ones move them):
  both red when only `qwen2` turns the bias path on.
- `gpu_seam_ernie4_5_decode_replays_from_one_recording` (ignored, Vulkan): one
  record-once recording for the whole decode on both GPUs, so ERNIE takes the
  replay path. Red when `ernie4_5` is given a NoPE step.

### End to end (Q4_K_M, `infr run --temp 0 --max-new 64`)

Taken without a quiet window on a shared machine, so the rates are indicative
only. "Write a short paragraph about the ocean." starts "The ocean is a vast,
dynamic body of water that stretches across the globe, covering vast expanses of
land and oceans. It is home to countless" on every device;
"请用中文介绍一下长城。" on Vulkan0 starts
"长城是中国古代军事防御体系的重要组成部分，其建造始于春秋".

| device                | prefill (15 tok) | decode (64 tok) |
| --------------------- | ---------------: | --------------: |
| CPU                   |         22 tok/s |     132.1 tok/s |
| Vulkan0 (RX 7900 XTX) |       2046 tok/s |    1071.4 tok/s |
| Vulkan1 (integrated)  |        187 tok/s |      99.4 tok/s |

### Not verified

- **Metal:** nothing here is arch-specific to a backend — no new op, no new
  tensor, no runner branch — so the Metal path runs the same graph it runs for
  `llama`. Not run (no Apple hardware here).
- **Linux:** the golden hashes were captured on Windows only, and the golden
  runs nowhere in CI (B167).

## MoE `ernie4_5-moe`: scoped, not started

Read in full: `src/models/ernie4-5-moe.cpp` (the graph), the MoE half of
`llama_model_ernie4_5::load_arch_hparams` / `load_arch_tensors`
(`src/models/ernie4-5.cpp`), `build_moe_ffn` (`src/llama-graph.cpp`) and the
converter (`conversion/ernie.py`). The 21B-A3B Q4_K_M file was **not**
downloaded; its metadata and tensor directory come from its first 8 MiB, fetched
with an HTTP range request.

### What the 21B-A3B file carries

`general.architecture = "ernie4_5-moe"`, 28 blocks, 389 tensors,
`embedding_length` 2560, 20 heads / 4 KV heads (head width 128, from
`n_embd / n_head`; no `key_length` key), `feed_forward_length` 12288 (the dense
layer), `expert_count` 64, `expert_used_count` 6, `expert_feed_forward_length`
1536 (a scalar), `leading_dense_block_count` 1, `interleave_moe_layer_step` 1,
`expert_shared_count` 2, `expert_shared_feed_forward_length` 3072,
`general.sampling.temp` / `top_p` 0.8, `add_bos_token = false` (the 0.3B file
has `true`), no `expert_gating_func`, no `expert_group_count`, no
`expert_weights_scale`. `blk.0` is dense (`ffn_gate`/`ffn_up`/`ffn_down`);
`blk.1`–`blk.27` carry `ffn_gate_inp` `[2560, 64]` F32, `exp_probs_b.bias`
`[64]` F32, `ffn_{gate,up,down}_exps`, and `ffn_{gate,up,down}_shexp`.

### What the reference does

- **Layer kind:** a layer is MoE iff `il >= leading_dense_block_count` **and**
  `(il + 1) % interleave_moe_layer_step == 0`. The loader creates MoE tensors
  for every `il >= leading_dense_block_count` regardless of the step, so a file
  with a step above 1 would not load there; the shipped file has step 1.
- **Router:** `build_moe_ffn` with SOFTMAX gating, `norm_w = true` (renormalize
  the selected weights), no weight scale (the key is not read; 0 means
  unscaled), no expert groups, weights applied after the experts. When
  `exp_probs_b` is present it is added to the **softmax probabilities** for
  selection only; the weights come from the unbiased probabilities.
- **Shared expert:** when `expert_shared_feed_forward_length > 0`, a plain
  SiLU-gated FFN on the same normed input, **summed with no gate**.
- **Attention / norms / output:** identical to the dense arch.

### What infr already has (symbols)

- `MoeConfig` with `gating = MoeGating::Softmax`, `norm_w = true`, `scale = 1.0`
  — the default arm of the `moe` parse in `Config::from_gguf`.
- Plain-summed shared expert: `Config::shexp_ff` with `shexp_gated = false` (the
  llama4 / DeepSeek path), loaded as `ffn_*_shexp` in `seam::runner`.
- Leading dense layers: `Config::n_layer_dense_lead` (DeepSeek), and the MoE
  interleave step `Config::moe_interleave_step` (llama4).
- Router bias: `Op::MoeFfn::exp_probs_b`, loaded from `exp_probs_b.bias` in
  `seam::runner` (DeepSeek V3), sized in `seam::wshape`.
- The CPU backend (`infr-cpu`, `Op::MoeFfn`) adds the bias to the GATED
  probabilities whatever the gating function, which is the reference rule.

### What is missing

1. **Config (small):** `ernie4_5-moe` arch const and arms; `MoeConfig` from the
   default softmax/renorm arm; `shexp_ff` from
   `expert_shared_feed_forward_length` (as qwen35moe reads it) with
   `shexp_gated = false`; `n_layer_dense_lead` and `moe_interleave_step` both
   read, and `Config::is_moe_layer` taught to apply **both** conditions (today
   it applies the dense lead for DeepSeek or the step for llama4, never both).
   `expert_feed_forward_length` as a per-layer array is not handled; the 21B
   file has a scalar, so refuse an array rather than read its first entry.
2. **Weight loading (small):** load `exp_probs_b.bias` for this arch (today the
   load is gated on `deepseek2` / `deepseek4`), optional per layer as the
   reference has it.
3. **Vulkan router (kernel work):** `moe_topk.comp`'s `sel_score` ranks a
   SOFTMAX-gated router by the raw logit, then adds the bias. Without a bias
   that is the same order as the probabilities; with one it is not, because the
   reference adds the bias to `softmax(logits)`, whose scale is set by the whole
   row. The shader's comment says no real `exp_probs_b` model uses softmax
   gating; ERNIE 4.5 MoE is one. The selection needs the full-row softmax (max
   and denominator over every expert, which the un-renormalized branch already
   computes for its weights) before the bias is added, and a parity case for
   softmax + bias in `crates/infr-llama/tests/seam_op_parity.rs`. No file infr
   runs today reaches the wrong rule: the `exp_probs_b` files it loads
   (deepseek2/32/4) are sigmoid or sqrt-softplus gated, and the softmax-gated
   Mistral Large 3 shard that was read carries no bias (B168). Whether the
   expert-parallel path (`ExpertParallelBackend`) carries the bias to each rank
   was not checked.
4. **Metal (kernel work or refusal):** the Metal `Op::MoeFfn` refuses any
   `exp_probs_b`, so the 21B file fails loudly there today. Implementing it is
   the same selection rule as item 3; until then the refusal stands.
5. **Verification:** the loose GPU comparison (top-5 overlap and a cosine floor)
   the recipe prescribes for MoE, against the CPU and against llama.cpp.

### Size and disk

`ERNIE-4.5-21B-A3B-PT-Q4_K_M.gguf` is 13331015456 bytes; the C: drive had
406018056192 bytes free when this was written. It fits the 7900 XTX resident and
needs streaming on the integrated GPU.

## Risks

- The router selection fix in item 3 changes a shader every MoE family's decode
  runs through; it must leave the no-bias path byte-identical, which the
  existing parity cases cover.
