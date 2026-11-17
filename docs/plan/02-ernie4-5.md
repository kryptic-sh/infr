# Phase 02 — ERNIE 4.5 (`ernie4_5`, then `ernie4_5-moe`)

**Why:** `plan.md` names ERNIE 4.5 21B-A3B as a candidate. The dense 0.3B model
shares the arch file, is tiny, and proved the block and tokenizer before the MoE
model is downloaded.

**Status:** the dense `ernie4_5` arch and the MoE `ernie4_5-moe` arch are both
done on the CPU and Vulkan. The MoE arch's Metal path is written and compiled
for the Apple target but has not run; the phase closes once CI's macOS job has
run its parity cases (see the end of this file).

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

## MoE `ernie4_5-moe`: done; Metal unverified

### The file

`unsloth/ERNIE-4.5-21B-A3B-PT-GGUF`, `ERNIE-4.5-21B-A3B-PT-Q4_K_M.gguf`,
13331015456 bytes, pulled with
`infr pull unsloth/ERNIE-4.5-21B-A3B-PT-GGUF:Q4_K_M` (snapshot `de45a831`). Its
metadata and tensor directory are what the range fetch below recorded: 28
blocks, 389 tensors, `blk.0` dense, `blk.1`–`blk.27` MoE with an
`exp_probs_b.bias` each, `interleave_moe_layer_step` 1,
`leading_dense_block_count` 1, `expert_feed_forward_length` 1536 (a scalar),
`expert_shared_feed_forward_length` 3072. The routed `ffn_down_exps` banks are
mixed: the pager splits the down role into a 3.2 MB-per-expert pool of 832
experts (13 layers) and a 2.2 MB one of 896 (14 layers). The repo has no F16
file; its `Q8_0` (23205355296 bytes) would be the one to pull for a tighter
oracle (B169's per-file floors). It was not downloaded.

### What it took

- **Config** (`crates/infr-llama/src/config.rs`): `arch::ERNIE4_5_MOE` and its
  arms (no q/k-norm, the `qkv_projection_biases` and applied-weight-scale rows
  of the dense arch). `Config::is_moe_layer` is now one rule for every arch,
  `il >= n_layer_dense_lead && (moe_interleave_step == 0 || (il + 1) % step == 0)`;
  each earlier arch sets at most one of the two, so their layer kinds are
  unchanged. The router is the default softmax / renormalized / unscaled
  `MoeConfig`; `expert_feed_forward_length` is a scalar or an array whose
  entries agree (`ernie4_5_moe_expert_ff`), `interleave_moe_layer_step` is
  required and positive, `shexp_ff` comes from
  `expert_shared_feed_forward_length` and is summed ungated.
  `Config::optional_router_bias` (deepseek2, ernie4_5-moe) gates the optional
  per-layer `exp_probs_b.bias` load in `seam::runner`, which was deepseek2-only.
  `ernie4_5_moe_tensors` refuses a missing tensor or an unexpected one by name,
  as llama.cpp refuses a file with tensors it did not create; `ffn_gate_exps` is
  required (the reference allows an ungated expert FFN infr has no path for).
- **Vulkan router** (`crates/infr-vulkan/shaders/moe_topk.comp`): under a
  softmax router, selection ranks the full-row probability
  (`exp(logit - max) / sum`, the CPU's arithmetic) whenever a bias or expert
  groups take part (`sel_by_prob`); without either it still ranks raw logits, so
  every other model's routing is unchanged. The same flaw affected group scores
  (sums of logits instead of probabilities), and the same change fixes it. Every
  Vulkan MoE path routes through this one kernel via `Recorder::moe_topk`: the
  resident `Op::MoeFfn` arms, `execute_paged_moe` (whose readback reads the
  kernel's ids), and expert-parallel, whose lowering keeps `exp_probs_b` on each
  rank's op with the router weights replicated (read, not run).
- **Metal** (`crates/infr-metal/shaders/moe.metal`, `src/exec.rs`): the device
  `moe_topk` kernel takes a bias buffer and a `has_bias` flag and ranks
  `exp(logit - max) / psum + bias`; the host fallback sorts on `probs + bias`.
  Group routing is still refused, and so is a bias next to hash routing. The
  `Op::MoeFfn::exp_probs_b` doc now says what the bias is added to (it said the
  logits).

### Verified (Windows box, llama.cpp `5676dfbe0`)

- **Logits, CPU** (`cpu_prefill_matches_llama_debug_dump` on `llama-debug`
  dumps):

  | prompt                       | max \|Δlogit\| | shared top-5 | probability cosine | top-1 |
  | ---------------------------- | -------------: | -----------: | -----------------: | ----- |
  | `The capital of France is`   |     0.49118963 |          5/5 |  0.999999982307742 | same  |
  | `北京是中国的首都，它的历史` |      0.4421668 |          5/5 |  0.998523170262377 | same  |

  llama.cpp's own CPU and Vulkan0 logits on the same two prompts differ by up to
  0.5907817 and 0.4977362.

- **Greedy, CPU.** Seven chat prompts, the chat render's ids fed to
  `llama-server` (CPU, `n_probs` 5, temperature 0) and to
  `SeamModel::generate_cpu_ids`; the prompt ids are the same on both sides (the
  21B file sets `add_bos_token = false`, so neither engine adds `<s>`, B166).
  Every split lands where llama.cpp's own top two are close:

  | prompt                                                      | agree | llama.cpp's smallest lead (first 32) |
  | ----------------------------------------------------------- | ----: | -----------------------------------: |
  | What is the capital of France? Answer in one paragraph.     | 48/48 |                      0.279 at step 7 |
  | Count from one to twenty in words.                          | 48/48 |                      0.614 at step 0 |
  | What is the chemical symbol for gold?                       | 36/48 |                     0.355 at step 11 |
  | 请用中文介绍一下长城。                                      | 28/48 |                     0.059 at step 28 |
  | Explain what a prime number is.                             | 16/48 |                     0.070 at step 16 |
  | What are the three primary colors?                          |  5/48 |                      0.004 at step 5 |
  | List the planets of the solar system in order from the sun. |  0/48 |    0.161 at step 0 (0.025 at step 6) |

  "Write a short paragraph about the ocean." agrees for 12 tokens and splits on
  `’` against `'`, a 0.0702 lead on llama.cpp's CPU.

- **End to end** (`infr run --temp 0 --max-new 64` against
  `llama-completion --temp 0` on the rendered prompt, "What is the capital of
  France? Answer in one paragraph."): infr's CPU text equals llama.cpp's CPU
  text, and infr's Vulkan0 text equals llama.cpp's Vulkan0 text. The CPU and
  Vulkan answers split from each other after "the Eiffel Tower," in both engines
  alike.
- **Golden:** `cpu_golden_ernie4_5_moe`'s two hashes are the same on Windows and
  on Linux (run in the WSL clone at `~/b162/infr`).
- **GPU** (`INFR_DEV`): on Vulkan0 (RX 7900 XTX) and Vulkan1 (integrated),
  `gpu_seam_matches_cpu_ernie4_5_moe` (chat-rendered ocean prompt prefill: top-5
  shared; max |Δlogit| 0.5761112 and cosine 0.9986429426009094 on Vulkan0,
  0.48740315 and 0.9988556443707916 on Vulkan1),
  `gpu_seam_matches_cpu_ernie4_5_moe_tokens`, the paged test and the replay test
  pass. The token test compares Vulkan against the f32 reference CPU backend;
  its prompt's top-1 leads along the 32 generated tokens, measured on that
  oracle path, are 0.5467 at the first token, 0.9894 at the third and at least
  4.88 elsewhere, and 0.5191 at the first token on Vulkan0's prefill, where the
  two paths' logits for the leading pair differ by 0.032 and 0.059. infr's
  production int8-activation CPU path leads there by only 0.2335.
- **Paged:** `gpu_seam_paged_ernie4_5_moe_matches_resident` decodes the same 16
  ids as the resident run with 4 GiB of VRAM cache over 12 GiB of DRAM, and with
  256 MiB over 512 MiB, on both GPUs. The routed experts total 12309626880 bytes
  (one layer's banks 489553920 or 424673280, by the down projection's quant). As
  the pager resolves the budgets on Vulkan0: at 4 GiB / 12 GiB, 1581 of the 5184
  expert blocks fit in VRAM and the DRAM tier holds all of them; at 256 MiB /
  512 MiB the VRAM arena is floored to 64 slots per pool, above the budget, and
  the DRAM tier splits into 36 / 75 / 75 / 39 slots across the down-Q6_K / gate
  / up / down-Q4_K pools, so experts are re-read from the file. That the paged
  runs really page is shown by
  `infr bench --set paging.cache=4g --set paging.dram=12g --set paging.stats=true`
  on Vulkan0: "28/28 expert layers PAGED" and per-pool evictions and disk reads
  in the `[moe pager]` lines.
- **The router fix on the real model:** with the old `moe_topk.comp`,
  `gpu_seam_matches_cpu_ernie4_5_moe_tokens` fails on Vulkan0 — the GPU writes
  "one, two, three, …", the text the CPU writes with the bias dropped — while
  the loose prefill comparison still passes (cosine 0.9976271849367727, max
  |Δlogit| 0.813328), so the loose form alone does not guard the router.

### Tests added, and what turns each red

- `ernie4_5_moe_layer_kind_needs_lead_and_step`,
  `ernie4_5_moe_tensors_are_checked_by_name`,
  `ernie4_5_moe_refuses_what_the_reference_cannot_load` (`config.rs`): all three
  red when `is_moe_layer` drops the step condition, and when it drops the
  dense-lead condition.
- `moe_softmax_bias_parity`, `moe_softmax_groups_parity` (`seam_op_parity.rs`,
  through the shared `moe_routing_parity` and `select_experts`): the Vulkan leg
  red on the old shader (max_err 8.5518897e-1 and 1.2022667e0 against the CPU),
  green after (2.3841858e-7, 2.0861626e-7); the CPU leg red when the CPU backend
  adds the bias to the logits.
- `cpu_golden_ernie4_5_moe` (ignored, model-gated): red when the CPU adds the
  bias to the logits, when `seam::runner` skips loading the bias, and when the
  shared expert is dropped (which also takes the llama.cpp oracle to 0/5 shared
  top-5 and a max |Δlogit| of 19.249985).
- `gpu_seam_matches_cpu_ernie4_5_moe`,
  `gpu_seam_matches_cpu_ernie4_5_moe_tokens`,
  `gpu_seam_paged_ernie4_5_moe_matches_resident`,
  `gpu_seam_ernie4_5_moe_decode_replays_from_one_recording` (ignored, Vulkan):
  the tokens test red on the old shader, as above.
- `moe_ffn_softmax_bias_host_parity`, `moe_ffn_softmax_bias_device_parity`
  (`crates/infr-metal/tests/parity.rs`, CI's macOS job): each asserts the CPU
  output moves with the bias by more than ten times the parity tolerance, in the
  parity check's own relative metric, so a Metal router that drops it or adds it
  to the logits fails. Not run on Metal here; the CPU half, reproduced in a
  scratch binary, gives 4.287163 (host case) and 3.6820276 (device case) against
  the 0.01 floor.

### Performance (Vulkan0, Q4_K_M, 3 reps)

No quiet window was arranged; the box was shared (Git Bash's load average read
between 2.67 and 8.47 across these runs), so the rates are indicative only.

| run                                            | pp512 (t/s) | tg128 (t/s) |
| ---------------------------------------------- | ----------: | ----------: |
| `infr bench`, resident                         |      2633.1 |       177.4 |
| `infr bench`, paged (4 GiB VRAM / 12 GiB DRAM) |      1186.9 |       122.5 |
| `llama-bench -ngl 99 -dev Vulkan0`             |     3964.86 |      212.88 |

Decode takes the record-once replay (one recording for the whole decode on both
GPUs). On the CPU, `infr run` prefilled 19 tokens at 1 tok/s and decoded at 4.6
tok/s where `llama-completion` did 43.78 and 24.87; Qwen3-30B-A3B shows the same
gap, so it is not this arch's (B173).

### Not verified

- **Metal:** the kernel and host-path change are compiled locally only through
  `cargo clippy --target aarch64-apple-darwin`; the MSL is compiled at run time,
  so nothing here has compiled it. The two new Metal parity cases run in CI's
  `test-macos` job.
- **Expert parallelism** with a router bias: read, not run.
- **Other quants:** only Q4_K_M was run.

### What remains before the phase closes

CI's macOS job passing the two Metal cases. Then close the phase out per the
planning rules; code comments (`cpu_backend.rs`) and B167 / B169 cite figures in
this file, so those move with it.
