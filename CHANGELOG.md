# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **`paging.early_submit`** (config only, default 8; `0` = off): on Vulkan, a
  paged forward (MoE experts or dense layers streamed through the VRAM cache) is
  cut into segments that the GPU starts on while the host records the rest. The
  first segment closes at this many dispatches and each later one at twice the
  last. It starts over each time a paged MoE layer drains the queue for its
  router ids and each time the staging ring rotates, so a forward takes a
  logarithmic number of cuts between two of those, not in all. Separate from the
  submit splitter (`device.submit_dispatches`), whose watchdog cap is still
  checked first, so it never lengthens a submit.
  `infr_vulkan::VulkanBackend::segment_cuts` counts that backend's watchdog
  cuts, early cuts and ring rotations.
- **ERNIE 4.5 MoE (`ernie4_5-moe`).** Baidu's ERNIE-4.5-21B-A3B runs on the CPU
  and Vulkan backends, with its experts resident or paged, and decodes on the
  record-once replay: the dense `ernie4_5` attention, a routed FFN (6 of 64
  experts) on every layer past the leading dense one and on the interleave step,
  a softmax router whose `exp_probs_b` bias picks the experts while the unbiased
  probabilities weight them, and a shared expert summed in. Its prefill logits
  share llama.cpp's top-5, and a chat answer's 64 greedy tokens are the same
  text as llama.cpp's, CPU against CPU and Vulkan against Vulkan. A file missing
  a tensor this layout needs, or carrying one it does not use, is refused naming
  it. The Metal backend applies the router bias too, on the same rule; that path
  has not been run on Apple hardware.
- **`infr_gguf::dequant::{factor_block, expand_factored, has_factored_form}`**:
  one affine quant block decoded into its factored parts, and their expansion to
  f32 — the two halves `dequant_factored` and `dequant_block` are now built
  from, for a caller that wants a block at a time.
- **`infr_llama::Config::optional_router_bias`**: whether the arch's reference
  loads `blk.N.exp_probs_b.bias` as optional on its MoE layers (deepseek2 and
  ernie4_5-moe).

### Changed

- **Paged MoE decode on Vulkan is up to 11% faster** where the expert cache
  holds every layer: a paged decode token is re-recorded each time, and the GPU
  used to wait for the whole recording before starting. On the RX 7900 XTX,
  interleaved with the previous build, tg128 after a 365-token prefill at
  `paging.cache=20g`: Qwen3-30B-A3B 160.4, 158.7 to 176.8, 177.1 t/s,
  ERNIE-4.5-21B-A3B 166.2, 165.8 to 177.9, 178.1 (resident: 186.4 and 184.4). At
  `paging.cache=4g paging.dram=12g` Qwen3-30B-A3B goes from 76.7, 79.8 to 82.5,
  82.1; ERNIE there is unchanged, and ERNIE decoding with no prefill at
  `paging.cache=20g` is about 1.5% slower (141.6, 141.5 to 139.8, 139.1).
  Streamed dense prefill on the integrated GPU gains too (Qwen3-1.7B pp128 at
  `paging.cache=200m`: 117.3, 117.2 to 130.5, 130.2). Resident runs take no
  early cut.
- **Prefill on the CPU backend is 2.6-6.1× faster, and ahead of llama.cpp's CPU
  at 512 tokens.** Warm pp512 on a Ryzen 9 9950X3D, 16 threads, interleaved with
  the previous build: SmolLM3-3B Q4_K_M from 175-184 to 622-636 tok/s
  (llama.cpp: 579), ERNIE-4.5-21B-A3B Q4_K_M from 73-81 to 374-449 (llama.cpp:
  246), Qwen3-30B-A3B Q4_K_M from 147-151 to 395-400 (llama.cpp: 231). Short
  prompts gain too: pp16 SmolLM3 54-68 to 292-309 (llama.cpp: 381), ERNIE 13-20
  to 101-121 (llama.cpp: 111), Qwen3-30B-A3B 54-63 to 155-160 (llama.cpp: 120).
  Every result is bit-identical to before and every CPU golden is unchanged. The
  causes, in order of size: each op wrote a fresh multi-megabyte buffer, so the
  forward spent much of its time faulting in zeroed pages (`execute` now reuses
  the buffers the previous layer freed); Q4_K, Q5_K and Q6_K prefill Linears and
  MoE expert GEMMs run a packed AVX-512 VNNI GEMM that carries four tokens per
  pass over an 8-row weight group (Q6_K from three tokens up, below which its
  per-row kernels are as fast), packs each group into per-thread scratch instead
  of allocating per call, and writes the row-major output directly instead of
  through a transposed temporary (Q5_K had no packed path at all; the dense Q6_K
  path rebuilt its whole-weight pack on every request); `Rope` ran on one thread
  and recomputed its angles per head; attention's q·k dots, and an f32 MoE
  router's, waited on one FMA chain at a time; and the MoE down stage
  un-transposed a whole layer's output with a cache line per element. The dense
  Q6_K pack cache is gone, so `kernels.cpu.repack_mb` now bounds only the
  fused-gate/up MoE Q4_K packs. Decode is unchanged.
- **MoE decode on the CPU backend is 3.4-3.5× faster.** A single-token MoE step
  runs its routed experts on f32 activations, and dequantized every expert
  weight row through the general host dequant first: four allocations and two
  passes per row. Expert rows in every affine quant (Q4_0, Q4_1, Q5_0, Q5_1,
  Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) are now decoded 256 elements at a time on
  the stack and fed straight into the dot, with the same bits as before, so
  every output is unchanged. The codebook and fp4 formats (the IQ and TQ quants,
  MXFP4 as gpt-oss's experts use, NVFP4, Q2_0) still take the old path. Warm
  decode on a Ryzen 9 9950X3D, 16 threads: ERNIE-4.5-21B-A3B Q4_K_M from 7.4 to
  25.2 tok/s (llama.cpp's CPU: 29.70), Qwen3-30B-A3B Q4_K_M from 7.5 to 26.1
  tok/s (llama.cpp: 31.01). Prefill is unchanged.
- **Vulkan MoE prefill and decode are faster, and so is dense prefill on the AMD
  Windows driver.** Interleaved with the previous build on an RX 7900 XTX (AMD
  Windows driver, idle machine), pp512 / tg128 tok/s: ERNIE-4.5-21B-A3B Q4_K_M
  2618-2620 → 3021-3043 and 178.1-178.2 → 187.5-187.6 (llama.cpp Vulkan: 4032 /
  213.3); Qwen3-30B-A3B Q4_K_M 2855-2858 → 3118-3131 and 173.8 → 191.7
  (llama.cpp: 3798 / 202.8); SmolLM3-3B Q4_K_M 3615-3640 → 4020-4022 prefill,
  decode unchanged (229.9-230.1 → 229.4-229.6; llama.cpp: 5462 / 256.6). The
  Q6_K int8 GEMM (most Q4_K_M MoE quants' `ffn_down_exps`, and dense Q6_K
  prefill on GPUs without f16 coopmat) stepped through k 16 elements per barrier
  pair and decoded every weight byte with its own load, running 3.7× slower than
  the Q4_K one on ERNIE's down shape; it now stages 32 elements per step from
  word loads (3309 → 1080 us per layer at pp512; on the Raphael iGPU
  Qwen3-0.6B's Q6_K GEMMs 152.5 → 113.9 ms per pp512, prefill 273.5-273.6 →
  280.5-280.6). Its outputs move at ulp level against the old kernel (float
  accumulation order; the integer sums are unchanged). On the AMD proprietary
  driver only, the prefill GEMM's narrow-grid split-K window also scales with
  the compute-unit count (never below the old 128 workgroups), so
  128-192-workgroup projections on a 96-CU GPU split too (`attn_output` 458 →
  302 us on Qwen3-30B-A3B); every other driver, RADV included, keeps the old
  window. The router top-k picks each expert with subgroup operations instead of
  a seven-step shared-memory tree and reads the row from shared memory (13.1 →
  8.8 us per layer on ERNIE decode), and the one-row f32 router GEMV streams
  vec4 (12.0 → 4.5 us). Router logits and the reordered split-K sums move
  results within the existing GPU-vs-CPU parity tolerances; the top-k picks
  experts by the same rule (highest score, then lowest id).
- **Chat templates turn non-strings into text as Python's Jinja2 does**, which
  is what Hugging Face's `apply_chat_template` renders with and so what models
  were trained on: `True`, `False`, `None`; floats as Python's `repr` (`1e-07`,
  `1e+16`, `2.0`, `0.30000000000000004`); lists and dicts as Python's `repr`
  (`{'a': [1, 'x', True]}`). This covers bare `{{ }}`, `| string`, `| join`,
  `upper` / `lower` / `capitalize` / `title` / `trim` / `replace`, `%s` and
  spec-less `{}` fields of `| format` and `str.format`, and `~` on a list or
  dict from the request. They used to print `true`, `none`, `0.0000001` and
  JSON-style containers. Prompts that change where a template prints such a
  value: Qwen3.5, StepFun 3.5, ByteDance Seed-OSS, MiniCPM5 and MiniMax-M3 in a
  replayed tool call's arguments (a bool, null or exponent-range float now reads
  `True`, `None`, `1e-07`); Apriel 1.5 and 1.6 in their tool block
  (`tool | string`); Gemma 4 for a parameter typed `["string", "null"]`
  (`['STRING', 'NULL']`); functionary v3.2 for a bool default (`False`) and
  GigaChat3 for a list default. Plain chat renders as before for every cached
  template. `~` on a float still writes Rust's digits (`0.000001`,
  `10000000000000000.0`), which GigaChat3 and functionary v3.2 print for a float
  default or bound.
- **`tojson` and every other float print break ties as Python does.** When two
  shortest digit strings are equally close, Python's `repr` and `json.dumps`
  keep the even one (`1125899906842624.25` prints `1125899906842624.2`); infr
  printed the odd one (`…4.3`). Other floats print as before.
- **`minijinja` and `minijinja-contrib` move to 2.24.0**, which also changes how
  templates run: `{% set %}` inside a `for` loop no longer leaks into the next
  iteration; `.items()`, `.get()` and `.keys()` on a mapping with a key of the
  same name call the method (2.21 refused "object is not callable" on any tool
  schema with an `items` key, which is every array parameter, wherever a
  template iterated it with `.items()`); a keyword argument takes an inline `if`
  (`tojson(indent=2 if x else none)`); a string repeated past 100,000,000 bytes
  is refused. GigaChat3 and 3.1 (`.items()` on a tool's parameters) and
  muse-glimmer and muse-spark (the inline `if`) were refused for every
  conversation with tools, and every conversation, respectively, and now render.

### Fixed

- **A paged model's first forward no longer cuts every later Vulkan forward into
  16-dispatch submits.** The submit splitter, which keeps each submit under the
  driver's GPU hang timeout, judged a forward by its wall time, so the seconds a
  cold paged forward spends reading weights in (8 s over 75 dispatches for
  ERNIE-4.5-21B-A3B with `paging.cache=4g`) armed it on a discrete GPU for the
  rest of the process. It now judges the GPU time each submit actually ran,
  timed with two timestamps per command buffer (wall time only where the queue
  writes none). Paged ERNIE-4.5-21B-A3B on the RX 7900 XTX: pp512 1169-1174 →
  1185-1189, tg128 123.2-123.5 → 126.6; paged Qwen3-30B-A3B: pp512 483-485 →
  482-487, but tg128 53.4-53.7 → 51.2-51.3, because its paged decode had been
  profiting from the accidental split (see docs/backlog.md B184).
  `INFR_SUBMIT_DISPATCHES=0` now disables the splitter for good, as documented;
  before, the first slow forward re-armed it.
- **Vulkan MoE routing ranks softmax probabilities, as llama.cpp does, when a
  router bias or expert groups take part.** `moe_topk.comp` ranked a
  softmax-gated router by raw logit and then added the `exp_probs_b` bias, and
  scored expert groups by summing logits; the reference adds the bias to, and
  sums, the full-row softmax probabilities, which can select different experts.
  The resident and paged paths both route through this kernel and were run with
  the fix; expert-parallel calls the same kernel by reading the code, but was
  not run. No file infr ran before reaches the changed rule: the bias-carrying
  `deepseek2` / `deepseek32` / `deepseek4` files it runs are sigmoid or
  sqrt-softplus gated, and softmax routing without a bias or groups is
  unchanged. A softmax-gated `deepseek2` file carrying `expert_group_count` > 1
  would have routed differently on Vulkan; none was run. ERNIE 4.5 MoE is the
  first model that needs the rule.
- **`{% break %}` and `{% continue %}` work in chat templates** (minijinja's
  `loop_controls` feature; Hugging Face loads `jinja2.ext.loopcontrols`).
  Cohere2MoE, Cohere's command-r7b tool-use template, Kimi-K3 and Reka-Edge were
  refused for every conversation.
- **`is iterable` and `is sequence` answer as Jinja2 does**: none is not
  iterable, and a string, a dict and an undefined value are sequences. Nemotron
  3 Nano and Qwen3-Coder ask `tools is iterable and tools | length > 0` and were
  refused for every conversation without tools.
- **`trim`, `title` and `capitalize` follow Python's string methods**: `trim`
  strips what `str.strip()` strips (U+001C–U+001F too), and a final `Σ` is
  lower-cased to `ς` (`'ΟΔΟΣ' | title` is `Οδος`). `is sequence` is false for
  `reverse`'s result, as for Python's iterators. `'%s' | format(a=0)` formats
  the whole mapping (`{'a': 0}`), and positional and keyword arguments together
  are refused, as Jinja2 refuses them.
- **A float in a request reaches the chat template exactly.** serde_json now
  parses with `float_roundtrip`; a replayed tool-call argument such as `1e-25`
  had been read one ulp off and printed back as `9.999999999999999e-26`.
- **`str.format` fields that look up an attribute or an index**
  (`'{a.b}'.format(a=…)`) get their argument unconverted, and a non-empty spec
  on none, a list or a dict is refused as Python refuses it. `| format` with
  more or fewer positional arguments than conversions is refused, and `%.3s`
  cuts Python's `str()` of a bool or a container.

## [0.3.0] - 2026-11-13

### Breaking

- **`infr_server::ChatOutcome` has a new field, `cached_tokens`.** An
  out-of-tree `ChatGenerator` that builds it with a struct literal must set it
  (0 when no KV prefix was reused). `UsageInfo`, `ChatCompletionChunk` and
  `ChatRequest` also gained fields (`prompt_tokens_details`, `usage`,
  `stream_options`), and `infr_llama::GenStats` gained `n_cached`, `ctx_full`
  and `natural_stop`; `GenStats` has a `Default`, the others do not.
- **`infr_chat::ToolCall` has a new field, `id`**; `infr_plat::link::link_blob`
  returns the kind of link it made; `infr_hub::Store::repo_dir` returns a
  `Result`.
- **`infr_core::fusion` loses two patterns no backend planned**:
  `RmsNormLinearCfg`, `MoeAddCfg`, `ExpertsOk`, and the `rmsnorm_linear` /
  `moe_add` fields of `FusionCfg` and `FusionPlan`.
- **`infr_vulkan::pager`: `GpuPager::touch_staged` takes the block's bytes as a
  fill closure that runs only on a miss**, and replaces `touch_staged_read`.
  `MoePagerSession::touch_role`, which nothing in the engine called, is gone;
  `stage_role` is the MoE session's entry point.
- **`infr_cpu::paged::PagedWeights::new` takes a third argument**, the
  `paging.trace` file its pools record into (`None` for no trace).
- **`infr_core::Backend` has a new method, `decode_replay_ready`, and `Graph` a
  new field, `replay_only`.** The seam runner now asks the backend, after
  compiling the position-0 decode plan, whether executing that one plan per
  token with the bound positions advanced decodes every position correctly (by a
  recorded replay, or by a static walk that takes every position from live
  state), instead of trusting its own copy of the backend's rules. It marks the
  plan `replay_only`, which a backend must refuse to run any other way. The
  method defaults to `false`: an out-of-tree backend that sets
  `Capabilities::decode_replay` must override it or its decode takes the
  per-token path, and must return an error for a `replay_only` graph it would
  run at the baked positions.
- **`infr_core::pager::trace::Call` has a new variant, `TouchGroup`, and
  `Outcome` a new variant, `Grouped`** (a pager's group touch, written `g` in a
  trace); an exhaustive match on either must handle them, and a trace reader
  built before this rejects traces that contain one.
- **`infr_vulkan::pager::MoePagerSession::register` now refuses a layer whose
  `layer_base` is not a multiple of its expert count, or whose expert count
  differs from another layer's in the same pool.** Each layer is one block group
  of its pool.

### Added

- **`paging.trace` (`INFR_PAGER_TRACE`): a pager touch trace.** Set it to a file
  path and every pager pool of the run (the VRAM expert and streaming arenas,
  the DRAM tier under them, the CPU backend's host pools) records each call it
  serves there: the entry point, the block, batch boundaries and the outcome
  (hit, or miss and the block it evicted). The format is documented in
  `infr_core::pager::trace`; a pool's trace replays to exactly the hit, miss and
  eviction counts `paging.stats` prints. Unset, it costs one test per pager
  call. For developers, `infr_core::pager::sim` replays a trace under other
  eviction policies (Belady's among them, a lower bound on traces without
  probes), `infr_core::pager::shapes` generates the access patterns real
  sessions make, and `infr-core`'s `pager_bench` test times the pager on both.
- **SmolLM3 (`smollm3`).** Hugging Face's SmolLM3-3B runs on the CPU and Vulkan
  backends: the llama block with rope skipped on every fourth layer, as
  llama.cpp hardcodes it. Its NoPE layers keep it off the Vulkan record-once
  decode replay, so decode re-records each token. The template's `/think` and
  `/no_think` modes follow `--think` / `--no-think`; its tool definitions
  (`xml_tools`) are not passed yet.
- **ERNIE 4.5 dense (`ernie4_5`).** Baidu's ERNIE-4.5-0.3B runs on the CPU and
  Vulkan backends, including the record-once decode replay: the llama block with
  heads wider than `n_embd / n_head`, its SentencePiece vocabulary and its chat
  template. Its logits match llama.cpp's on the F16 file. The MoE sibling
  `ernie4_5-moe` (21B-A3B) is not supported yet.
- **`VulkanBackend::decode_recordings` and
  `DenseVulkanSession::decode_recordings`**: how many record-once decode
  recordings the backend has made, which tells a decode replayed from one
  recording from one recorded per token.
- **`stream_options.include_usage` on `infr serve`.** A streaming request that
  sets it gets one more chunk before `[DONE]`, with an empty `choices` and the
  request's `usage`.
- **`usage.prompt_tokens_details.cached_tokens`**: the part of the prompt served
  from the KV prefix cache.
- **A GGUF's own sampling recommendation is honoured.** `infr run` and
  `infr serve` take `general.sampling.temp`, `top_k` and `top_p` from the file
  for any of those you did not set yourself, above the built-in per-family
  defaults and a `generation_config.json`, as llama.cpp does. The startup banner
  shows `+gguf` when they apply. Other `general.sampling.*` keys (`min_p`,
  repeat penalties, mirostat) are not applied, and are named in a warning at
  load. A `top_k` of `-1` in `generation_config.json` now means "keep all", as
  it does in the GGUF.
- **`seed: -1` and `max_tokens: -1` are accepted** and mean "not set", for
  clients written against llama.cpp. `max_tokens: -1` therefore takes the
  server's default budget; it does not mean "until the context is full".

### Changed

- **The pager no longer scans its whole cache to pick an eviction victim.**
  `infr_core::pager::Pager` keeps its state by slot, with recency as a linked
  list, so a miss costs the protected blocks it walks past instead of every
  resident block, and a hit costs about half what it did. Every eviction
  decision is unchanged (checked call by call against the old implementation).
  Measured on Qwen3-30B-A3B with `paging.cache=4g` and `paging.dram=12g` on an
  RX 7900 XTX: decode 75.1–76.1 → 82.5–82.7 tok/s.
- **Decoding a paged MoE model does far less pager bookkeeping for layers whose
  experts are all in VRAM.** Each such layer used to check and touch every one
  of its experts, for each of its roles, on every token; it is now one check and
  one touch per layer and role, through a new block-group mechanism in
  `infr_core::pager::Pager` (`group_by`, `group_is_resident`, `touch_group`).
  Eviction decisions are unchanged: on Qwen3-30B-A3B with `paging.cache=18g`,
  which keeps most layers wholly resident, the VRAM pools' pager calls per
  decode token fell from 16,905 to 522 with identical output and identical
  `paging.stats` counters. Decode throughput on an RX 7900 XTX did not change
  measurably.
- **Tokenization now follows llama.cpp for more models, so their prompt token
  ids change.** Llama 3 (`llama-bpe` and its aliases) and Qwen3.5 / Qwen3.6
  (`qwen35`) GGUFs were split with the GPT-2 regex; they now use their own, and
  Llama 3 also ignores merges for whole-vocabulary words. Runs of spaces
  (indented code) in a SentencePiece-style GGUF without a `tokenizer.json` now
  reach the multi-space tokens. Several more `tokenizer.ggml.pre` names are
  mapped, and an unknown one logs a warning instead of silently getting the
  GPT-2 regex. Qwen3 and Gemma 3 prompts are unchanged.
- **Llama 3.1, 3.2, 3.3 and Llama 4 output changes: their RoPE frequency factors
  are now applied.** These files carry a `rope_freqs.weight` tensor that
  llama.cpp divides each rotary angle by; infr applied it only for Gemma 4, so
  many rotary pairs turned too fast and the error grew with position. Llama 3.0
  has no such tensor and is unchanged.
- **Llama 4's context is capped at 8191 tokens, and Mistral-Large-3's at its
  temperature length.** Past those points the reference changes its attention
  (chunking, and a position-dependent temperature) in ways infr does not
  implement, so a longer context computed something else. A larger `--ctx` is
  lowered with a warning.
- **Models infr would compute wrongly are refused at load, by name**: a GGUF
  with a per-tensor `.scale` that the reference applies (NVFP4 / ModelOpt
  conversions); a DeepSeek-family file whose YaRN magnitude is not 1; a Qwen3.5
  file with a non-periodic recurrent-layer layout. A `.scale` the reference
  ignores still loads.
- **SentencePiece GGUFs without a `tokenizer.ggml.add_space_prefix` key — Llama
  2-style, Mistral v0.x, Phi-3 — now get the leading-space prefix, as in
  llama.cpp**, before the text and after every special token, and Phi-3 drops
  the whitespace that follows its special tokens. Their prompt token ids change.
  A file that set the key to `true` used to decode with every space removed; it
  now decodes correctly.
- **Generation stops on more end-of-turn tokens.** The stop set was the declared
  EOS plus three fixed names; it now follows llama.cpp's list (`<end_of_turn>`,
  `<|end|>`, `<|eom_id|>`, `<turn|>` and the rest) and the GGUF's `eot` / `eom`
  ids. A listed name counts only where the vocabulary marks it as a control or
  user-defined token, so a literal `</s>` in a Qwen reply still prints.
- **Tool-call ids on `infr serve` are nine alphanumeric characters** (they were
  `call_…`), and an id the client sends back with a tool result now reaches the
  chat template. Templates that pair calls with results by id, Mistral's among
  them, need both.
- **The model cache works on filesystems without symlinks**: a snapshot entry
  falls back to a hard link, then a copy. A copy doubles the disk space that
  model takes, and says so once.
- **`infr pull` refuses a model file it cannot verify.** If the request that
  fetches a file's sha256 keeps failing, the pull stops instead of downloading
  unchecked; a definitive answer (401, 403, 404) fails at once and says why. An
  empty `HF_TOKEN=` is treated as unset.
- **A `tokenizer.json` beside the GGUF that does not match the model's
  vocabulary now logs a warning.** It is still used.
- **A GGUF whose weight shapes disagree with its own metadata is refused at
  load**, with the tensor's name and both shapes. Such a file used to load and
  then compute garbage or read out of bounds. DeepSeek V4's tensors are covered;
  not every tensor of every architecture is yet.
- **Vulkan: a kernel whose workgroup is larger than the device allows is refused
  by name**, and the wide RMS-norm kernel is chosen only on a device that can
  run it; a smaller device uses the narrow one.
- **DiffusionGemma: even seeds no longer repeat the next odd seed.** The seeded
  RNG forced its state odd, so `--seed 2` and `--seed 3` gave the same output.
  Every even seed and seed 0 now draw a different stream — including the default
  seed, 42, so default DiffusionGemma output changes. Odd seeds are unchanged.
- **Memory headroom inside a cgroup no longer counts reclaimable file cache as
  used.** A container that had read a lot of files looked nearly full, and infr
  sized its host weight cache too small.
- **Vulkan: prefill of a streamed dense model uses the tiled GEMM on a device
  without f16 cooperative matrix** (integrated GPUs, Intel Arc). A weight
  streamed through `paging.cache` took the per-row GEMV for every prefill chunk,
  re-reading the whole weight once per prompt row; it now takes the same dp4a or
  fma GEMM the weight would take resident, and its output matches the resident
  run's. On a cooperative-matrix device with int8 dot, a streamed Q4_K weight
  now takes the dp4a GEMM wherever the resident one does — an output width that
  is a multiple of 64 but not of 128, or any width under `INFR_NO_GEMM_WARP` —
  instead of the GEMV. A weight too large for the tiled GEMM's addressing (a
  big-vocab output projection) keeps the chunked GEMV, as before.
- **The host (DRAM) weight tier evicts by last use.** A hit did not refresh a
  block's position, so eviction followed load order. Output is unchanged; MoE
  models paged from disk should re-read fewer blocks.

### Removed

- **`infr_llama::grammar::tool_constraint_for`.** `ToolConstrainer`, which
  `OaiRenderer` holds, does the same and builds the grammar tokenizer once per
  model instead of once per request.

### Fixed

- **Attention q/k/v projection biases are added on every arch that carries them,
  not only `qwen2`.** llama.cpp loads `attn_q.bias` / `attn_k.bias` /
  `attn_v.bias` as optional for the llama-shaped families (`llama`, `llama4`,
  `qwen2`, `qwen3`, `qwen3moe`, `gemma3`, `deepseek`, `smollm3`, `ernie4_5`) and
  adds whichever are present; infr read them for `qwen2` only and ran any other
  file that carried them without them. They are now added when every layer
  carries all three; a file with only some of them is refused naming the first
  missing one, and `qwen35` / `qwen35moe` refuse them. A `qwen2` file without
  biases now loads instead of failing.
- **Chat templates render their whitespace the way llama.cpp renders it.** The
  newline after a `{% %}` or `{# #}` tag and the indentation before one are now
  dropped (Jinja's `trim_blocks` and `lstrip_blocks`, which llama.cpp's template
  engine always applies). Of the templates checked, only SmolLM3's prompt
  changes: it no longer starts with a stray newline token, so SmolLM3's output
  can differ from earlier infr versions. Qwen2.5, Qwen3, Qwen3.5, Gemma 3 and
  Llama 3.2 render as before for the conversations checked. The prompt of any
  other model whose template leaves a newline after a tag, or indents a tag,
  changes too.
- **Tool definitions and replayed tool calls reach the model in llama.cpp's JSON
  layout.** A chat template's `tojson` now writes what llama.cpp's writes
  (Python's `json.dumps` with `ensure_ascii=False`): `", "` and `": "`
  separators where infr wrote compact JSON, floats as Python writes them
  (`1e-05`, `1e+16`), and the `ensure_ascii`, `separators` and positional
  arguments templates may pass, which used to be a render error. The tools
  prompt of Qwen2.5, Qwen3, Qwen3.5 and Llama 3.2 changes (for Llama 3.2, whose
  tool definitions were already indented, in a replayed tool call's arguments
  and in any float written in exponent form), so tool-calling output can differ
  from earlier infr versions; prompts without tools are unchanged.
- **DeepSeek V2/V3.2 GGUFs route and rope the way llama.cpp loads them.** A
  `deepseek2` file without `expert_gating_func` and GLM 4.7 Lite's shape (47 or
  48 layers, a 154880-token vocabulary) now routes its experts with sigmoid, as
  llama.cpp chooses, instead of softmax. `rope.scaling.yarn_beta_fast` / `_slow`
  are no longer read for `deepseek2` and `deepseek32`: llama.cpp ignores them on
  those architectures, so a file declaring other values now ropes as it does
  there. A `deepseek32` file with `expert_gating_func = 0` (no gating function)
  is refused at load instead of routing as softmax; llama.cpp aborts on it.
- **Paging an MoE model through the host RAM tier no longer reads experts the
  GPU already holds.** Every routed expert used to be looked up in the host tier
  before the GPU cache was checked, so an expert resident on the GPU but not in
  host RAM was read from disk and thrown away, possibly evicting one host RAM
  needed, and the host tier's hit and miss counts in `paging.stats` tracked GPU
  traffic rather than its own. The host tier is now asked only on a GPU cache
  miss, as it already was for dense models.
- **Streaming no longer re-decodes the whole reply for every token.** Output is
  byte-identical; the work per token no longer grows with the reply's length.
- **A tool call quoted inside a model's reasoning is no longer run as a call,
  and no longer hides the real call from the stream.**
- **`infr serve` with tools: a plain JSON answer streams as it is generated**
  (it was held until the end in case it was a tool call), text before a late
  `<think>` block is no longer dropped, and a tool call that follows such text
  now runs. A tool-call block that cannot be parsed, for instance one cut off by
  `max_tokens`, is shown as content instead of disappearing. Text and reasoning
  around a tool call arrive in the order the model wrote them, and removing a
  call no longer leaves stray blank lines or joins the paragraphs on either side
  of it.
- **Forced tool calls on SentencePiece models (Llama 2, Mistral, Phi-3, Gemma
  3): the grammar now sees spaces.** It saw the `▁` marker instead, so it could
  not accept a space before or inside the JSON.
- **`infr serve --parallel`: a panic while seeding a slot from another no longer
  leaves both slots busy forever.**
- **Qwen3.5 / Qwen3.6 on Vulkan: a multi-turn chat no longer continues from
  recurrent state that ran ahead of the conversation.** Chained decode feeds
  several tokens per submit, so a reply that ended part-way through a chain left
  the state past the stop; the next turn reused it. The engine now tracks how
  far the state was written and, when it is ahead, restores a snapshot taken at
  the previous turn's prompt and prefills from there. The snapshot is one more
  copy of the recurrent state per slot (about 19 MiB for Qwen3.5-0.8B).
- **Sliding-window models (Gemma 3 / 4): a turn that rewinds after a longer
  earlier one no longer reads window rows that have been recycled.**
- **A NaN logit or router score no longer panics the sampler or the CPU MoE
  routing.** The output is still NaN.
- **Qwen3.5 / Qwen3.6: a snapshot restore that fails part-way can no longer be
  followed by a turn that reuses the half-restored state.**
- **DeepSeek on the CPU with paged weights, and on multi-GPU: the MoE router
  bias is now pinned and bound like the other weights.** It was missing from the
  list of tensors an MoE layer reads.
- **The pipeline-cache crash tripwire no longer trusts a process id alone.** In
  a container every run can be pid 1, so a crashed run looked alive and its
  poisoned cache was kept. Each run now holds a file lock on its marker.
- **Metal: temperature sampling on llama-family models was close to greedy.**
  The sampler read a random draw that was refreshed only once every 64 tokens
  and started at zero.
- **Metal: memory no longer grows with the square of the reply length** for
  quantised KV formats. A scratch pool kept one buffer per size requested, and
  the size grew every token.
- **Metal: a failed buffer allocation is an error naming the size** instead of a
  crash, and a large `max_tokens` on a small context truncates the reply (and
  reports `finish_reason: "length"`) instead of failing the request.
- **Metal: a DeltaNet key-head width other than 32, 64, 128 or 256 no longer
  reads into the neighbouring head's state.**
- **Vulkan: compute grids no longer exceed the device's workgroup limit on wide
  layers.** At the default 1024-row prefill chunk, an elementwise kernel over a
  4096-wide layer needed one workgroup more than the 65,535 the spec guarantees,
  which RADV tolerated and other drivers need not. The elementwise, KV-store,
  rope and attention kernels now split such a grid in two dimensions.
- **Vulkan: three crashes behind configuration.** An F16-weight model on a GPU
  without `shaderFloat16` (or with `INFR_NO_F16`) ran a float16 shader for its
  fused residual add; `INFR_MMV_FUSE_QUANT=1` panicked on any quant but Q4_K and
  Q6_K; `INFR_NO_MMV_M4` panicked on decode. Each now takes a kernel that
  exists.
- **Vulkan: shapes a kernel cannot hold are refused by name** in release builds
  too (DeltaNet head width, QK-norm head size, sampler top-k range, a Q8 KV row
  that is not a whole number of blocks). They were debug-only assertions.
- **`INFR_DELTA_STRIDED=1` computed garbage** on Vulkan (the kernel wrote its
  output at the wrong offset) and on the CPU (which does not implement the mode
  and was handed its input layout anyway).
- **Sliding-window models under `serve --parallel`: a slot forked or seeded from
  another could size its KV ring from the wrong prefill chunk**, losing the
  extra slot or seeding stale rows.
- **Vulkan sampling: an exact tie at the top-k boundary could drop a token that
  belonged in the top-k**, and which tied tokens survived depended on thread
  scheduling. The strictly greater values are taken first now and ties go to the
  lowest token id. Output with no exact ties is unchanged.
- **Vulkan: attention over a very long span no longer overruns a kernel's shared
  memory** (a DiffusionGemma canvas past a few thousand keys; batched attention
  past 262,144 keys).
- **Multi-GPU: `INFR_EXPERT_PARALLEL` refuses gemma-4-26B-A4B**, which it
  computed wrongly (the dense FFN branch was dropped at each cut). A mixed
  integrated + discrete world now sizes prefill for the integrated GPU, and any
  graph that would read a scratch tensor across a device cut is refused by name.
- **A chat template inside a GGUF can no longer crash infr** through
  `strftime_now` with a bad format, or ask for a multi-gigabyte pad through a
  helper's width argument; both are template errors. (String repetition inside a
  template is still unbounded.)
- **`infr pull`: a moved or renamed repo is still verified**, a download is
  flushed to disk before it is renamed into the cache, a cached file whose size
  disagrees with the hub is fetched again, and two pulls of the same model at
  once both succeed.
- **`infr pull` on Windows no longer stalls when a late part of a large file
  arrives first.** NTFS filled everything below such a write with zeros before
  it returned, which for the last chunk of a model meant writing the whole file
  once just to place it. The partial download is now a sparse file.
- **A repo id with a backslash, `..` or a drive prefix is refused**; on Windows
  it could write outside the cache.
- **`generation_config.json` is found for a GGUF in a subdirectory** of its
  repo, so those models get their recommended sampling again.
- **`infr compare` passes `--config` and `--set` to the runs it starts**, and a
  failing run reports its error; `infr multi` honours `INFR_CTX` and
  `[device] ctx`.
- **Metal: out-of-bounds reads.** The flash attention kernels read whole blocks
  of keys past the newest one with nothing guaranteeing those rows existed; the
  two-row Q4_K / Q6_K kernels read a weight row past the end for an odd output
  width; and the MoE router used an invalid expert index when no router logit
  was finite. KV caches on Metal are now padded to the attention block, and a
  kernel is chosen only when the rows it reads exist.
- **Metal: ties.** `argmax` returns the lowest tied token id, and the Q4_0 /
  Q5_0 KV cache picks its block scale the way the CPU does when two values tie
  for largest.
- **`infr serve`: `usage.prompt_tokens` counts the whole prompt.** It reported
  only the tokens prefilled for this request, so a KV prefix cache hit made a
  long prompt look short. `/metrics` still counts prefilled tokens only.
- **`infr serve`: a reply cut off because the context window filled finishes
  with `finish_reason: "length"`**, not `"stop"`; and a reply whose end token
  lands exactly on the `max_tokens` limit now says `"stop"`, not `"length"`.
- **`infr serve`: an over-long prompt is `400 context_length_exceeded`**, with
  `type: "invalid_request_error"`, instead of a 500. On a streaming request the
  same error arrives as a typed error frame. An oversized body is a 413 and a
  missing JSON content type a 415; both were 400.
- **`infr serve`: a client that disconnects stops its generation**, during
  prefill too, and frees its `--parallel` slot. It used to run to completion.
- **`infr serve` on CPU, Metal or DiffusionGemma: one panicking generation no
  longer makes every later request fail.** The model lock stayed poisoned.
- **`infr serve`: the API key is checked before the request body is read**, and
  a 401 body now has `type: "invalid_request_error"` and
  `code: "invalid_api_key"` (it said `server_error`).

- **Qwen3.5 / Qwen3.6: the first prompt after a reset no longer inherits the
  previous generation's state.** The gated-DeltaNet recurrent state was only
  zeroed when the slot's token list was non-empty, so the prompt after a KV
  reset, or after the GPU warmup generation, ran on top of stale state. A
  prefill that fails or is cancelled part-way now also forces the state to be
  rebuilt on the next request.
- **Vulkan with a Q8_0 KV cache: a conversation seeded from another slot's
  shared prefix decoded that prefix wrongly.** The seed copied the int8 codes
  but not their scales, which Vulkan stores separately. This is the shared
  system prompt case under `infr serve --parallel` with `Q8_0` KV. The backend
  trait gains `Backend::copy_kv_prefix`, with a default that out-of-tree
  backends inherit.
- **Malformed GGUF files now fail with an error instead of a panic**: a tensor
  offset that is not a multiple of `general.alignment`, an F32 tensor that is
  not 4-byte aligned (legal with an alignment of 1 or 2, and now read
  correctly), an i2_s tensor with a partial code group, a zero
  `qwen35.ssm.conv_kernel`, and a Gemma 4 `shared_kv_layers` that leaves fewer
  than two layers with their own KV.
- **DeepSeek and Llama 4 files with `expert_weights_scale` absent or 0 no longer
  lose their routed experts.** infr multiplied by the raw value; llama.cpp
  treats 0 as "unscaled", and infr now does the same.
- **A panic while reading a paged weight block no longer hangs later requests
  for that block.** The block was left marked as loading for good; it is now
  released, and only the request that hit the panic fails.
- **Vulkan with a quantised KV cache: prefill attention read past its scratch
  buffer** when the context length was not a multiple of the kernel's tile. The
  scratch now covers the rows the selected kernel reads.
- **Vulkan MoE models without an expert bias no longer grow the weight arena on
  every graph lowering**, and no longer bind a buffer that is destroyed before
  the command buffer runs.
- **Vulkan weight paging: a request that fails part-way no longer leaves the
  pager believing weights were uploaded.** The next request used whatever the
  slot held before; it now re-uploads them.
- **Vulkan: models with a very large vocabulary and a narrow embedding (the
  small Gemma 3 files) no longer read past the output weight on every decoded
  token**, which also roughly doubled the work of the final projection.
- **Vulkan: a weight whose input width is an odd multiple of 32 gave wrong
  prefill results** on GPUs with the f16 cooperative-matrix tier, for the
  32-element-block quants (Q4_0, Q5_0, Q5_1, Q8_0, IQ4_NL) and BF16 / F16. Those
  shapes now use a kernel that handles the width.
- **Vulkan: a GPU missing `shaderInt8` or `computeFullSubgroups` is refused at
  startup with the feature named**, instead of an unnamed device-creation error.
- **`kernels.vulkan.pv_splits = 0` or `kernels.vulkan.flash_splits = 0` no
  longer panics with a division by zero**; zero now means the adaptive default.

## [0.2.0] - 2026-10-29

### Breaking

- **`infr_server::ChatGenerator` has a new required method, `status()`**,
  returning a `GeneratorStatus` (backend, device, context per slot, device
  memory) for `/status` and `/metrics`. An out-of-tree generator must implement
  it; it must not wait on a running generation.

### Added

- **`GET /metrics` on `infr serve`**, in the Prometheus text format: request
  counts by result, prompt tokens prefilled and tokens generated, time spent in
  prefill and decode, running and queued requests, and slots per model (total
  and busy). Counters only grow, so `rate()` gives throughput over any window.
  Like `/v1/models`, it needs the bearer token when `serve.api_key` is set.

- **`GET /status` on `infr serve`**: one JSON snapshot with the version, uptime,
  running and queued requests, and for each hosted model its backend, device,
  context per slot, slots (total and busy) and device memory (total and still
  allocatable, from the driver's live budget). `/metrics` carries the same
  context and memory figures as gauges. Only the Vulkan multi-slot engine (the
  default `infr serve` path) reports a device, context and memory; the CPU,
  Metal and diffusion paths report their backend and `null` for the rest. Gated
  by the API key like `/metrics`.

- **Releases now publish to crates.io, Homebrew, the AUR and Scoop.** A `v*` tag
  publishes every workspace crate except `infr-testkit` to crates.io, in
  dependency order; install the binary with `cargo install infr-cli` (the `infr`
  name there belongs to another project). It also pushes the `infr` formula to
  `kryptic-sh/homebrew-tap`, `infr-bin` to the AUR and the `infr` manifest to
  `kryptic-sh/scoop-bucket`, all from the release's own archives and checksums.
  Templates live in `pkg/`.

### Changed

- **`infr pull` retries a failed chunk of a ranged download** instead of failing
  the whole file. A dropped connection, a short body, or an HTTP `5xx` or `429`
  for one range is retried in place with a short backoff, a few times, before
  the file fails as before (keeping its partial for the next run). A re-uploaded
  object, a server that stopped serving ranges and local disk errors still fail
  at once.

- **An explicit `--ctx` / `INFR_CTX` whose KV cache cannot fit on the Vulkan
  device is refused before the weights load**, with the cache size, the room the
  device has and a window that would fit. It used to fail during placement with
  "dense weights exceed VRAM … streaming floor", which blamed the weights, or
  after the weights had loaded with the allocation guard's generic message. The
  check counts only the KV buffers the VRAM guard checks, at the cheapest format
  the cache could still use, so it never refuses a window that would have run.
  `INFR_KV_OVERFLOW=1` and `INFR_NO_VRAM_GUARD=1` skip it. A window whose cache
  fits on its own but not beside the weights still stops at placement, and that
  message now names the context and the KV cache size too.

## [0.1.0] - 2026-10-28

### Added

- **Prebuilt release binaries.** A `v*` tag now builds `infr` for
  `x86_64`/`aarch64-unknown-linux-gnu` (glibc 2.28 and up),
  `x86_64-pc-windows-msvc`, and `aarch64`/`x86_64-apple-darwin`. It attaches
  each one to a GitHub Release as a `.tar.gz` or `.zip` with a `.sha256`
  sidecar, together with a `SHA256SUMS` file and a shell-completions plus man
  page tarball. Installing no longer means building from source with `glslc`.

- **`infr-plat`, the platform seam.** Every direct use of an operating system —
  file locks, positioned reads and writes, the content-addressed store's link
  step, the host-memory probe, process liveness, signal handlers, config/cache
  directory resolution, an interruptible stdin read — now lives in one leaf
  crate with an arm per platform, and a test forbids any other crate from
  depending on `libc` or the `windows` crate (`infr-vulkan` is allow-listed with
  its reason). `infr-core`, `infr-hub` and `infr-cli` no longer name an OS
  library at all, and `infr-hub`'s hand-written `LockFileEx` bindings and
  `#[repr(C)] OVERLAPPED` are replaced by the `windows` crate's generated ones.
  `infr_plat::mem::available` reports where its figure came from, so the Linux
  cgroup clamp, the unclamped Windows figure, and macOS having no probe at all
  are distinguishable instead of hidden behind one `Option<u64>`.
- **CI runs clippy and the test suite on Linux, macOS and Windows.** Previously
  one platform wide, and every platform arm had something sitting in it.
  Bringing the matrix up found, in order: a constant that is dead on aarch64,
  two Windows-only lint failures, a profiler test asserting a wall-clock ceiling
  that a loaded runner blows past, the `link_blob` symlink bug above, a download
  test whose hub over-counted bodies in flight and could fail on two cores, and
  two tests that hard-coded a POSIX path separator or `$HOME` and so asserted
  the behaviour Windows never had. Only the symlink bug affected shipped code;
  the rest were checks that could not have run, or could not have passed, where
  nothing was running them.

- **`infr` builds and runs natively on Windows** (thanks to
  [@Headmaster218](https://github.com/Headmaster218), PR #91). `infr-hub` and
  `infr-core` previously called `libc::flock`,
  `std::os::unix::fs::FileExt::write_all_at` and `/proc/meminfo`
  unconditionally, so the crates did not compile for a Windows target at all.
  Each is now a `cfg(unix)` arm paired with a Windows one:
  `LockFileEx`/`UnlockFileEx` for the blob-store lock, a `seek_write` retry loop
  for ranged downloads (Windows' positional write can return short where
  `write_all_at` cannot), a hard-link fallback in `link_blob`, and
  `GlobalMemoryStatusEx` for available memory. Unix and macOS behaviour is
  unchanged. CI now builds, lints and runs the workspace suite on a Windows
  runner, and the file lock has a test that runs on every platform; what is
  still unexercised there is anything above unit-test level — no job downloads a
  model or runs a generation on Windows. See `docs/backlog.md` § B66.
- **`Op::CompressPool` — DeepSeek V4's compressor pooling — on CPU, Vulkan and
  Metal.** One op for the four ggml nodes both V4 compressor variants share
  (`build_hca_compressed_kv_from_state` and
  `build_overlap_compressed_kv_from_state` differ only in their gathers): a
  softmax-weighted average that collapses a `window` of cached KV rows into one
  compressed row. The softmax runs **per channel over the window axis**, which
  the reference buys with a pair of `ggml_permute` + `ggml_cont` around its
  `ggml_soft_max`; here both permutes are folded into the op's indexing, so
  nothing is materialised. The max-subtract form is required, not an
  optimisation: the overlapping compressor pads short windows with an
  `-INFINITY` sentinel row, and those lanes must weigh exactly zero. A window
  that is entirely `-inf` is `0/0` and every backend writes `0.0` — a deliberate
  deviation from ggml, which propagates `NaN` there (see `Op::CompressPool`'s
  doc for why). Nothing emits this op yet; it is the third V4 op-level slice,
  like the two before it. Its Vulkan dispatch splits a wide grid
  (`Recorder::dispatch_wide`), because one thread per output element puts
  `blocks * n_embd` past the guaranteed `maxComputeWorkGroupCount[0]` at a real
  V4 context length. Metal is parity-tested too: its kernel is `#[ignore]`d
  locally for want of Apple hardware, but the macOS CI job runs the ignored
  tests, so `compress_pool_f32` compiled and matched CPU on a real device.
- **`Op::Attention` gained an optional `key_bias` — an additive per-(query row,
  key) score mask, on CPU, Vulkan and Metal.** Ordinary attention could not
  previously carry the top-k mask `Op::TopkMask` produces; only `Op::Mla` had a
  `key_bias` field, and DeepSeek V4's CSA (compressed/selective attention)
  layers need the same mask on plain attention instead. The new field is the
  identical contract MLA's `key_bias` already has — added to the scaled score
  `q·K[j]*scale` BEFORE the softmax max, indexed by key POSITION (never the
  ring-cache row), `-inf` at unselected keys — and combines independently with
  `sinks` on the same op, since a V4 CSA layer carries both at once. Landed in
  the SAME kernel family sinks already uses on every backend (Vulkan's
  `attention_kv.comp` `-DBIAS`/`-DSINKS -DBIAS` builds, Metal's
  `ATTN_BIAS_KERNEL`/`ATTN_SINKS_BIAS_KERNEL` macros, one CPU interpreter arm)
  rather than a second kernel, so a layer using both never has one field
  silently dropped by whichever kernel happens to run. `key_bias: None` is
  byte-identical to before this field existed on every model in the tree. Note
  for anyone porting the op to a fourth backend: a `-inf` on the FIRST key of a
  row is the case that separates the three softmax formulations — a running max
  seeded at `-inf` evaluates `exp(-inf - -inf)` and NaNs the whole row, which
  the Metal kernels guard explicitly and the CPU (separate max pass) and Vulkan
  (finite tile-max seed) arms cannot reach. Not yet wired into the DeepSeek V4
  model builder — see `docs/backlog.md`.
- **DeepSeek V4's hash-routed MoE layers run, on CPU and Vulkan.** Such a layer
  takes its experts from an i32 `blk.N.ffn_gate_tid2eid`
  `[n_expert_used, n_vocab]` table indexed by TOKEN ID rather than from the
  router's top-k, and nothing in the tree could gather that selection, so `infr`
  refused the layer by name rather than silently running a different set of
  experts. A new `Op::GatherI32` (`ggml_get_rows` on an integer table — CPU
  interpreter arm and the `gather_i32.comp` Vulkan kernel) produces the
  `[batch, n_expert_used]` selection that `Op::MoeFfn::expert_ids` already
  consumed, and the seam emits it per hash-routed layer. `Op::EmbedGather` could
  not serve this: its kernel walks the table in whole 32-element sub-blocks, so
  a 6- or 8-wide row gathers nothing. The graph's token-id Input is now declared
  whenever a hash-routed layer needs it, not only when the token EMBEDDING is
  gathered on-device — a model whose `token_embd` dtype has no gather kernel is
  still hash-routed. A `ffn_gate_tid2eid` that is not
  `expert_used_count * vocab` entries is refused, because the gather indexes it
  by token id with no clamp. Metal refuses `Op::GatherI32` by name (its device
  MoE path already refuses V4's mandatory sqrt-softplus gating). This does NOT
  make the shipped `DeepSeek-V4-Flash-0731-GGUF` generate: every layer of that
  file past 1 is compressed (ratio 4/128), which is still unimplemented, and it
  refuses there.

- **GGUFs carrying `GGML_TYPE_I32` tensors open.** Integer tensors are lookup
  tables a kernel indexes with rather than weights, and `Gguf::open` refused
  them outright (`unsupported: ggml type 26`). The first shipped model to need
  one is DeepSeek-V4, whose hash-routed layers each carry an i32
  `blk.N.ffn_gate_tid2eid.weight` token-id → expert-id table: the reference
  conversion `ggml-org/DeepSeek-V4-Flash-0731-GGUF` would not open at all,
  failing before a single tensor shape was read. It now loads end to end — the
  config parses, every one of its 1328 tensors resolves, and the model reaches
  the existing designed refusal naming its compressed (ratio 4/128) layers,
  which are still unimplemented.

- **`infr pull` splits ONE file across connections too.** Shard-level
  concurrency did nothing for a model that ships as a single file, which most
  do: `unsloth/DeepSeek-V3.2-GGUF`'s `UD-TQ1_0` is one 161 GB GGUF and still
  went over one connection. It is now fetched as a grid of 64 MiB byte ranges,
  several at a time, and reassembled — measured on that object, in the same
  60-second window: 11.4 MB/s before, 80.5 MB/s after, and the whole 161 GB pull
  ran at 80.0 MB/s end to end (33.6 minutes against a projected 3.9 hours).
  `hub.pull_jobs` (default `8`) is now the TOTAL connection bound rather than a
  file count, so shards and ranges share one allowance and a 236-shard repo
  cannot turn into 64 sockets; `0` and `1` still mean strictly one connection,
  never split. An interrupted ranged download resumes from a per-range sidecar
  next to the partial, and a partial whose object was re-published upstream is
  discarded rather than continued — a resumed splice of two uploads is a
  plausible-sized corrupt file, and it is bound to HF's LFS sha256 rather than
  to an `ETag` so a different CDN edge cannot look like a different file.
  Servers that do not serve ranges (no `Accept-Ranges`, or a `200` where a `206`
  was asked for) fall back to the single stream, files of 64 MiB or less are
  never split, and the end-of-download sha256 gate is unchanged: one bar per
  file, one digest, one decision.

- **`infr pull` fetches a split model's shards concurrently.** A single
  connection to the HF CDN is what caps a download, not the link — measured
  against the same objects, one connection sustained 8.8 MB/s and five sustained
  78.7 MB/s — so the shards of a `-NNNNN-of-MMMMM` set are now fetched several
  at a time. A 245 GB five-shard `unsloth/DeepSeek-V3.2-GGUF:Q2_K` pull ran at
  80 MB/s end to end. Bounded by the new `hub.pull_jobs` (`INFR_PULL_JOBS`,
  `--set hub.pull_jobs=N`, default `8`), because shard counts come from the repo
  rather than from the user (`DeepSeek-V3.2-REAP` ships 236); `0` and `1` both
  keep the old strictly sequential behaviour. Resume, the LFS-sha256 gate and
  the refusal to link a mismatched blob are unchanged and per-file. Concurrent
  downloads each get their own progress line (`infr_core::progress::group`)
  instead of overwriting one another, and a failed file stops the others from
  claiming more work instead of pulling the rest of a model that cannot load.

- **Multi-shard (`gguf-split`) models load.** `Gguf::open` on any member of a
  `-NNNNN-of-MMMMM.gguf` set follows `split.count` to the whole set, maps every
  shard separately (each keeping the shared file mapping that makes a
  larger-than-RAM model runnable), takes the metadata from shard 1 and unions
  the tensor index across all of them — so a tensor resolves against the file
  that actually holds it instead of failing with `tensor not found`. Opening a
  LATER shard works too: it carries no model metadata, so the set is re-rooted
  at shard 1. The set is validated rather than trusted — the filename's
  `-of-MMMMM` must equal `split.count`, every shard's `split.no` must match the
  position its filename claims, all shards must agree on `split.count`, the
  assembled index must be exactly `split.tensors.count` entries with no repeated
  name, and a missing shard fails naming the file. Tested against a 5-shard 229
  GB DeepSeek-V3.2-Q2_K.
- `infr_core::gguf_split` — the `-NNNNN-of-MMMMM.gguf` naming convention, parsed
  in one place instead of two. `infr-hub` (is the cached set complete?) and
  `infr-gguf` (which files is this model?) now share it.
- `infr_core::blockio::FileBlockIo::open_shards` — the weight pager's disk tier
  reads a shard set as one address space, refusing at open any shard whose
  length is no longer the one the weights were loaded against (which would shift
  every offset in every later shard).

- **A DeepSeek V4 model whose `compress_ratios` are all zero now GENERATES**
  (stage 4, slice A). The `ratio == 0` tier — hyper-connections wrapping both
  sublayers, single-head MQA attention with a weightless per-head Q norm, a
  `[nope | rope]` tail rope, attention sinks, a de-roped grouped low-rank output
  projection, and sqrt-softplus MoE with the per-layer SwiGLU clamps — is
  emitted end to end and runs on CPU and Vulkan. Compressed layers (ratio 4
  or 128) are refused by name, the message saying which layer and what is
  missing, rather than approximated by the ratio-0 graph. Two supporting changes
  ship with it: Vulkan's `Op::Linear` now accepts `w_off` on an **F32** weight
  (by shifting the weight's `bufferDeviceAddress` base — the float GEMVs already
  address their weight by pointer, and `w_off` is row-aligned), which is what
  lets the grouped output projection run there at all; and V4 is excluded from
  the record-once decode replay tape, like `deepseek2`, because its ops have no
  dynamic twins. See `docs/deepseek.md` § Stage 4.
- **DeepSeek V4 hash-routed MoE and per-layer SwiGLU clamping** (stage 4, op
  level). `Op::MoeFfn` gains `expert_ids`: an optional pre-gathered
  `[rows, n_expert_used]` I32 handle (llama.cpp's `selected_experts_in`,
  gathered from `ffn_gate_tid2eid` by token id) that replaces the top-k
  selection on V4's first `hash_layer_count` layers. The router matmul and
  gating still run — the routing WEIGHTS remain the router's own probabilities
  at the hash-chosen experts, renormalised and scaled — while `argsort_top_k`,
  `exp_probs_b` and group-limited routing are skipped; the last two are refused
  alongside `expert_ids` rather than silently ignored. `Op::GatedAct`,
  `Op::GatedActFused` and `Op::MoeFfn` also gain `swiglu_clamp: Option<f32>`,
  built through the new `infr_core::graph::swiglu_clamp(limit)` which carries
  llama.cpp's `limit > 1e-6` disabled gate so no caller can pass a layer's `0.0`
  through as a real clamp. V4 clamps `up` symmetrically and the gate one-sided
  and **pre**-activation, where every other arch clamps post-SiLU. All of it
  runs on CPU, Vulkan and Metal; `None` leaves every existing model's numerics
  bit-identical. See `docs/deepseek.md` § Stage 4.
- **DeepSeek V4 Sinkhorn hyper-connections** (stage 4, op level): three new ops
  that replace `x = x + f(x)` with `hc_mult` parallel residual streams.
  `Op::HyperConnectMix` turns the mixing matmul's output into the `pre` collapse
  weights, the `post` output gates and the `comb` mixing matrix
  (Sinkhorn-normalised to approximately doubly stochastic);
  `Op::HyperConnectPre` collapses the streams to one vector for a sublayer; and
  `Op::HyperConnectPost` re-expands that sublayer's output back across the
  streams. `Op::HyperConnectMix`'s `gates` is `None` for the model head
  (llama.cpp's `build_hc_head`), which is the same arithmetic over a narrower
  `mixes`. All three run on CPU, Vulkan and Metal; `hc_mult` is accepted in
  `1..=infr_core::graph::HYPER_CONNECT_MAX_MULT` (8) and refused loudly on the
  host beyond that. See `docs/deepseek.md` § "Sinkhorn hyper-connections".
- **DeepSeek V4 attention primitives** (stage 4, op level): `Op::QkNorm`'s
  `weight` became optional, so a weightless per-head RMSNorm is expressible
  without a fake ones-vector operand; `Op::Attention` gained an optional
  per-head `sinks` input (one extra logit per head joining the softmax max and
  denominator, contributing no value); and `Op::Rope` gained a `backward` flag
  for de-roping (llama.cpp's `ggml_rope_ext_back` — the forward rotation with
  `sin` negated). All three run on CPU, Vulkan and Metal; `sinks: None` /
  `backward: false` / `weight: Some(..)` reproduce the previous code path
  exactly, so no existing model's numerics move. V4's grouped low-rank output
  projection needs no new op — it composes from `Op::CopyStrided` and
  `Op::Linear`'s existing `w_off`.
- **DeepSeek V2 architecture support** (stage 2): registered `deepseek2` arch
  string, parsed MLA hyperparameters (`q_lora_rank`, `kv_lora_rank`,
  `qk_rope_dim`, `head_k_mla`, `v_head_dim`, lite detection via tensor
  presence), configurable MoE gating (`expert_gating_func` → softmax / sigmoid /
  sqrt-softplus), `expert_weights_norm`, group-limited routing fields, and
  `rope_yarn_log_mul` (with the convert-script ÷0.1 fix). Added
  `MoeGating::SqrtSoftplus` variant and wired it in CPU + Vulkan backends. See
  `docs/deepseek.md` § Stage 2.

- **MLA attention kernels** (DeepSeek V2/V3 absorbed form, `Op::Mla`): Vulkan
  `mla.comp` and Metal `mla_f16kv` compute kernels implement the full per-head
  pipeline — `wk_b` absorption of q_nope, internal q_pe RoPE (NORM interleaved),
  two-pass SDPA over the unified f16 KV cache (one row per token, V aliased from
  the first `kv_lora_rank` columns of K), and the `wv_b` output projection.
  Ring-buffer, causal / sliding-window / canvas masks supported. CPU math
  covered by `mla_parity` in `seam_op_parity.rs`; Metal dispatch covered by
  `mla_parity`/`mla_ff_parity` in the Metal parity suite, executed on the macOS
  CI job (a real Metal device).

- **DeepSeek V3 MoE routing** (Vulkan): `moe_topk` now selects on
  `probs + exp_probs_b` while weighting from the unbiased probs (the noaux_tc
  router bias), supports sqrt-softplus gating (`gating=2`), and enforces
  group-limited routing (per-group top-2, top `n_expert_groups_used` groups,
  mask the rest). The `blk.%d.exp_probs_b.bias` tensor loads from V3 GGUFs and
  threads into `Op::MoeFfn`.

- **DeepSeek V2-Lite tests**: `cpu_deepseek2_config`,
  `cpu_deepseek2_prefill_finite`, `cpu_deepseek2_prefill_paris` (CPU oracle +
  finiteness over the vocab) and `gpu_seam_matches_cpu_deepseek2` (Vulkan vs
  CPU, `#[ignore]`d behind a GPU) — gated behind a V2-Lite GGUF in the HF cache.

- **DeepSeek V1 support** (`deepseek` architecture): plain MHA attention +
  softmax-gated MoE with ungated shared expert, following llama.cpp's
  `src/models/deepseek.cpp`. First `n_layer_dense_lead` layers are dense FFN,
  the rest are MoE. Tokenizer pre-processing added for `deepseek-llm`,
  `deepseek-coder`, and `deepseek-v3` pre-types (see `docs/deepseek.md` § Stage
  1). Works on CPU + Vulkan backend via the existing `FfnW::Moe` with `shexp`
  path (same as llama4's plain-summed shared expert).

- `infr run`, `infr bench` and `infr serve` now notice the model file being
  overwritten underneath the live weight mapping and fail with a named error
  instead of serving output from weights that no longer match the file. `run`
  checks at both ends of every turn, `bench` before reporting any numbers, and
  `serve` at the start of each request. New `infr_gguf::watch::WeightWatch`,
  re-exported as `infr_llama::WeightWatch`.

- **A model that does not fit now streams from disk on its own.** Weights that
  fit stay exactly as they were — resident on the GPU, or zero-copy mmap on the
  CPU backend, with no arena and no copies. Only when they do not fit does the
  engine page them `DISK → DRAM → VRAM` (`DISK → DRAM` on the CPU backend),
  sizing the DRAM arena from the host memory actually available rather than
  requiring a budget nobody could guess. Measured on Qwen3-14B Q8_0 under an 8
  GB cap, the automatic budget lands on the same 7.4 GB that measured **2.17x
  faster decode than the mmap path** it replaces.
  - The probe honours **cgroup memory limits** (v2 and v1, tightest ancestor),
    not just `/proc/meminfo` — inside an 8 GiB container the host file still
    reports the whole machine, and sizing an anonymous arena from that is an OOM
    kill. Linux only; other platforms report "unknown" and keep the mmap path
    unless a budget is set by hand.
  - **Unified-memory devices (iGPU, APU) stream `DISK → GPU-accessible RAM` with
    no host cache between.** Their streaming arena is already host RAM, so a
    cache beneath it would hold a second copy the GPU cannot read in place;
    instead its misses are served by block-granular positioned reads rather than
    through the GGUF mapping, whose page cache thrashes on a forward pass's
    cyclic sweep. That is what lets a model far larger than the machine run on
    those parts at all. **Untested on unified hardware** — none was available —
    but the mechanism is covered on a discrete GPU by `INFR_DRAM_BYPASS`
    (below). Metal has no pager at all yet and is unaffected.
- `INFR_DRAM_CACHE` / `paging.dram`: the host weight cache's budget. **Unset now
  means "size it automatically"**; a value pins the arena and wins over every
  automatic decision (including on a machine where the model would have fit,
  which is how the streaming path gets exercised at all); and **`0` turns it off
  entirely**, which is what an A/B against the mmap path needs. A budget too
  small to seat a weight class leaves that class mapped and says so.
  - **CPU backend**: every weight above 1 MiB. Measured on a memory-capped
    Llama-3.2-1B F16: decode 2.06x faster at a 1.5 GB cap with 210x fewer major
    faults, prefill 3-7.5% slower (`docs/perf/results.md`).
  - **Vulkan backend**: a third tier under both dense weight streaming and the
    paged MoE expert cache, so a VRAM miss resolves against the arena and
    reaches the file only when that misses too. MoE pages ONE EXPERT at a time
    rather than a whole bank. A block the arena has no room for is read straight
    into the staging ring instead of evicting one, so the streaming majority
    costs one copy rather than two. Measured on a memory-capped Qwen3-14B Q8_0
    under a forced 2 GB VRAM budget: **decode 2.17x faster than the mmap path it
    replaces** at an 8 GB cap with a 7 GB arena (1.41x with a 3 GB one), 38x
    fewer major faults, 232 → 110 GB read, and parity when memory is plentiful
    (`docs/perf/results.md`). The arena budget is the dominant factor — 3 GB → 7
    GB is worth 1.6x on its own — which is why it is now sized automatically
    rather than left to a guess. The measurement covers one GPU, one drive and
    Linux only.
  - `INFR_PAGER_STATS=1` reports hit rate, reads and bytes for each tier.
  - The host arena admits a block on its SECOND miss, not its first. A tier
    above only calls down on its own misses, so first-miss admission filled the
    arena with the prefix the VRAM pager was about to keep resident forever —
    blocks that then never call down again. Measured on Qwen3-14B: 4 of 9 slots
    per pool were dead, and the rule turns useful hits per pass from 5 into 9
    while cutting bytes read ~9% at the same budget.
  - One paged block is read with several concurrent positioned reads rather than
    one, which is what puts the tier ahead of the mapping it replaces: a drive
    delivers its bandwidth on queue depth (measured 1.2-1.5 GB/s for a single
    read against a 2.2 GB/s device ceiling), while the page cache gets its
    readahead issued in parallel by the kernel for free. Reads stay correct on
    every platform, but the speedup is measured on Linux/NVMe only — a Windows
    handle not opened for overlapped I/O serializes them.
- `INFR_DRAM_BYPASS` / `paging.dram_bypass`: read paged blocks straight from
  disk into GPU memory with no host cache — the shape a unified-memory device
  takes automatically. It exists as a flag so that behaviour can be exercised on
  a discrete GPU, which is the only hardware it can be tested on here, and is
  also the honest choice on a machine whose RAM is better spent elsewhere. No
  effect on the CPU backend, where that arena is the only tier there is.
- `INFR_LAYER_MAJOR` / `paging.layer_major`: force the prefill loop order. `1` =
  layer-major, `0` = chunk-major, unset = layer-major exactly when the weights
  stream (see the Changed entry below). Both overrides are for A/B; forcing it
  on is the only way to put a resident model on the layer-major path.

### Changed

- **An explicit `--ctx N` that an f16 KV cache cannot hold now gets q8_0.** The
  default context already tried a q8_0 cache before clamping. A window the user
  chose (`--ctx N` / `INFR_CTX=N`, including per slot under `--parallel`) was
  taken as given and failed at allocation if f16 did not fit. It now switches
  the cache to q8_0 when that holds the whole window, with a warning, and keeps
  the window as asked. A window that fits in neither format still fails at the
  VRAM guard, and setting `INFR_KV_TYPE_K/V` still wins.

- **Q4_K Vulkan MMQ unpacks weights with aligned word reads.**
  `native_gemm_mmq_q4k.comp` masks packed nibbles instead of reconstructing each
  word byte by byte, preserving scales, accumulation order and watchdog limits.
  Alternating cached Qwen3-0.6B benchmarks improved prefill on the Windows AMD
  RDNA2 iGPU; decode results were mixed. Bound-buffer, resident-BDA and paged
  reload parity are covered in `nc_gemm_parity`; measurements and scope are in
  [the Windows guide](docs/windows.md#packed-q4_k-follow-up-2026-10-08).

- **On Windows, `-t` defaults to one thread per physical core.** It defaulted to
  every logical processor, so with SMT the CPU backend's spin pool put two
  workers on each core's hyperthreads, where they contend instead of helping. On
  a Ryzen 9 9950X3D (16 cores, 32 threads), Qwen3-0.6B Q4_K_M on `--dev cpu`
  decodes at ~127 tok/s by default instead of ~65, and prefills at ~660 tok/s
  instead of ~580; GPU runs are unaffected. `-t` or `RAYON_NUM_THREADS` still
  set it explicitly. The core count comes from the new
  `infr_plat::cpu::physical_cores`; other platforms keep the logical-count
  default until the same measurement is made there.
- **Windows builds target the host CPU, like Linux builds.**
  `.cargo/config.toml` set `target-cpu=native` for Linux only, so on Windows the
  CPU backend's dequant and matvec loops never vectorized to AVX2/AVX-512. On a
  Ryzen 9 9950X3D, Qwen3-0.6B Q4_K_M on `--dev cpu` went from ~380 to ~580 tok/s
  prefill (pp256) and ~48 to ~61 tok/s decode (tg64). As on Linux, the binary is
  then specific to the CPU it was built on.
- **Shaders compile in parallel, and the build finds the Vulkan SDK's `glslc`.**
  `infr-vulkan`'s build script ran one `glslc` per shader variant sequentially;
  it now spreads them across cargo's job allowance (`NUM_JOBS`) — a release
  rebuild of `infr-vulkan` after a shader edit took 148 s with `-j 1` and 14 s
  at the default 32 jobs on a Ryzen 9 9950X3D. When `glslc` is not on `PATH` it
  falls back to `$VULKAN_SDK/Bin`, so a shell opened before the Vulkan SDK was
  installed still builds.
- **`infr serve --parallel` (`-n`, `--np`) defaults to 1 slot, was 4.** Each
  slot owns a full KV cache and, with `--ctx` unset, the per-slot window is the
  VRAM-fit context divided by N — so the old default handed every user a quarter
  of the context window whether or not they ever issued a concurrent request.
  Concurrency is now opt-in and a single request gets the whole window; pass
  `--parallel N` for the previous behaviour. `infr multi` already defaulted to
  1, so the two now agree. The engine is unchanged: the concurrent seam runs at
  one slot exactly as before.

- **DeepSeek MLA attention gained a subgroup kernel tier, and it is the whole
  DeepSeek frame.** `mla.comp` was the only MLA kernel at any depth, and
  `INFR_PROF_OPS` measured it at 97.3% of decode GPU time at `-d 2048` and 89.4%
  of prefill. Its `-DMLA_SG` builds now walk the key range ONCE with an online
  softmax instead of twice, and score eight keys per `barrier()` with
  `subgroupAdd` instead of running a 128-lane shared-memory tree costing 18
  barriers per key. On an RX 7900 XTX with
  `JenniSD/DeepSeek-V2-Lite-Chat-Q4_K_M-GGUF`, three reps with the arms
  alternated: pp512 965.8 → 1035.5, pp2048 671.3 → 851.1, tg64 111.6 → 128.7,
  and tg64 at depth 2048 **6.3 → 11.9** t/s. Output is not bit-identical to the
  kernel it replaces (the online softmax sums the same terms in a different
  order); the CPU/Vulkan whole-vocab cosine on the 55-token seam test is
  0.99872, against 0.99861 for the kernel it replaces. `INFR_NO_MLA_SG` selects
  the scalar builds, which are unchanged and are also what a device that cannot
  pin a 32-wide subgroup gets.

- `Gguf::path()` → `Gguf::shards()`, which returns `(path, length)` per shard in
  order. `TensorBytes::file_range` / `Gguf::tensor_file_range` now report
  offsets in the model's concatenated file address space; for a single-file
  model that is the file offset it always was.

- **A streamed model now prefills LAYER-MAJOR: the prompt sweeps the weight set
  once instead of once per prefill chunk.** Prefill runs in `device.ubatch`
  chunks and every chunk used to run the whole model, so a P-token prompt paid
  `ceil(P/ubatch)` complete weight sweeps — invisible when the weights are
  resident and the entire bill when they stream. The chunk loop now runs INSIDE
  the layer loop, which reads each weight once per prompt at the same
  chunk-sized dispatches. (Raising `device.ubatch` to the prompt length reaches
  the same I/O and is not a substitute: it bakes a single multi-second submit
  that trips the GPU hang watchdog.) Measured on a memory-capped Qwen3-14B Q8_0
  (`MemoryMax=8G`, `paging.cache=2g`, `paging.dram=6g`, P=4096, RX 7900 XTX) at
  the 1024-row default chunk: **prefill reads 25.27 → 6.31 GB from disk (4.00x)
  and runs 341.9 → 779.9 tok/s (2.28x)**, with the read volume now identical to
  a single-chunk prefill's. The cost is holding every chunk's residual stream at
  once (`ctx * n_embd` f32), which the streaming budget reserves for. Resident
  models keep the chunk-major order, where the reorder would buy nothing and
  only add activation residency, and so does gemma4-E2B on any backend — its
  per-layer token embeddings are built by the graph prologue, which a span
  starting past layer 0 cannot see.

- The Vulkan context window is now re-decided against the memory the device
  reports free once the weights are resident, instead of only against a pre-load
  estimate of them. That estimate is systematically light — the weight footprint
  prices tensor bytes while the resident-BDA arena commits them into ≥64 MiB
  blocks (measured +2.20% on gemma-4-31B, +2.43% on gemma-3-12b, +1.16% on
  Qwen3-14B), and no footprint has a term for the driver's own pipeline and
  descriptor memory. Sessions whose window used to be advertised and then fail
  mid-prefill on a `VRAM budget exceeded` now get a window they can fill. The
  clamp logs what it measured, only ever shrinks, and leaves a context set
  explicitly via `--ctx`/`INFR_CTX` alone.
- The activation reserve is re-fit to measured peaks and its interim 1.5x safety
  margin is gone, so gemma-3-12b now serves its full 131072-token f16 window at
  the default 1024-row prefill chunk (780 t/s, was 760 at the 256-row rung it
  used to be pushed onto). The reserve gained explicit terms for MoE expert
  scratch and for qwen35's DeltaNet mixer, both of which it previously
  under-counted.
- New `Backend::device_alloc_room` and `Backend::activation_peak`, both
  defaulting to `None` for backends that cannot report them (CPU, Metal — those
  keep their existing behaviour unchanged). The second is a high-water mark of
  live activation bytes that the runner compares against what it reserved,
  warning when a generation's real peak exceeds the prediction.

### Fixed

- **tokio 1.54.1** (from 1.52.3) for
  [RUSTSEC-2026-0386](https://rustsec.org/advisories/RUSTSEC-2026-0386)
  (`AsyncFd::new` and related constructors violated I/O safety), and **rustls
  0.23.46**, **rustls-webpki 0.103.16** and **webpki-roots 1.0.11** for
  RUSTSEC-2026-0346 to -0349 (TLS secrets in `Debug` output, data sent after an
  ECH rejection, unbounded Brotli certificate decompression, a ServerHello
  key-exchange downgrade) and RUSTSEC-2026-0381/-0382 (distrusted emSign
  intermediates accepted). tokio runs `infr serve` and every `infr pull`, and
  rustls is the TLS stack under `infr pull`.

- **The weight-file watch covers every shard of a split model.** `run`, `serve`
  and `bench` stamp the model's files at load and refuse to continue if one is
  truncated or rewritten in place. On a `gguf-split` set only the shard named on
  the command line was stamped, so shards 2..N could change unnoticed under the
  live mapping. `WeightWatch::open_model` now stamps every file `Gguf::open`
  maps.

- **Each GPU gets one `VulkanN` index, however many drivers claim it.** The
  Vulkan loader lists a device once per ICD that reports it, so a machine with
  two installs of the same vendor driver (one AMD package per GPU, say) listed
  every card twice and shifted every later index. On such a box `--dev Vulkan1`
  bound the discrete card instead of the iGPU. `infr devices` and device
  selection now keep the first entry for each `deviceUUID`.

- **A forced `tool_choice` never returns ordinary text instead of the call.**
  With `tool_choice: "required"` or a named function, a constrained decode that
  failed or produced no parseable call used to retry unconstrained and could
  answer with plain text and no tool call. It now fails the request, saying the
  call did not complete within `max_tokens` or did not parse. The CPU backend
  (`--dev cpu`) now supports the constrained decode, so forced tool calls work
  there instead of always taking that fallback. Diffusion and speculative Metal
  models still have no constrained path, and a forced `tool_choice` on them now
  fails rather than returning plain text.

- **`serve.request_timeout_secs` now stops a request during prompt prefill.**
  The server's cancel latch reached the generation only through the first
  generated piece, so a deadline shorter than the prefill let every chunk run
  (on CPU, a 3,512-token prompt with a 1 s deadline still took 31.8 s and held
  its slot throughout). `RequestCtx::with_cancel` now links the latch into the
  per-request abort that the prefill loop polls between chunks, so the request
  ends at the next chunk boundary. A deadline that fires before any output
  returns an `aborted: request cancelled during prompt prefill` error instead of
  the misleading `shutdown requested`. `ChatGenerator::chat` now takes
  `cancel: &Arc<AtomicBool>`.

- **A resumed single-stream download no longer gets stuck or splices the wrong
  bytes.** A partial that already held the whole file (interrupted after the
  last byte, before the commit) got `416 Range Not Satisfiable` on every retry
  and never finished. It now goes through the sha256 gate and is committed when
  the server's reported size matches, and is downloaded again otherwise. A `206`
  whose `Content-Range` is missing, starts somewhere other than the resume
  offset, or reports a different total is no longer appended. That matters most
  for non-LFS files such as `generation_config.json`, which have no digest to
  catch a splice.

- **Selected Vulkan benchmarks use bounded integrated-GPU profiles.**
  `small_m_bench`, `attn_dsplit_probe`, `attn_ktile_probe`, `gemm_bench` and
  `moe_id_gemv_real_dims` select small synthetic workloads before dispatch,
  enforce submission bounds, and report unavailable timings and unsupported
  kernels explicitly. Kernel overrides preserve `INFR_DEV`; Q6_K expert-bank
  sizing uses the format layout. MoE parity checks every exercised bank after
  timing, while discrete performance budgets remain required on their original
  paths. Production kernels, submission policy and Windows TDR settings are
  unchanged.

- **Windows memory budgets use remaining Job Object headroom.**
  `infr_plat::mem::available` subtracts current job commit usage and process
  private usage from their enabled limits before choosing the tighter budget.
  Failed probes return unknown instead of treating already committed memory as
  available. Native child-process regression tests cover both limit types.

- **Ctrl-C on Windows drains the GPU instead of killing infr mid-submit.**
  `infr_plat::signal::install_handlers` was a no-op off unix, so the first
  Ctrl-C terminated the process with a submit still in flight. It now registers
  a console control handler: Ctrl-C and Ctrl-Break latch the same graceful
  shutdown as `SIGINT`, closing the console window (or logoff/shutdown) latches
  it as `SIGTERM`, a second event force-exits, and the exit status is 130/143 as
  on unix. A Ctrl-C at an idle `infr run` prompt also takes effect immediately,
  where it used to wait for Enter: console input is now read with
  `ReadConsoleW`, which reports the aborted read that `Stdin` silently retries.
- **The GPU kernel-cache tripwire works on Windows.** A cached pipeline binary
  whose seeding run died (the signature of a GPU hang) is discarded on the next
  start — but `infr_plat::proc::pid_alive` answered "alive" for every pid on
  Windows, so the dead run was never noticed, the suspect blob was fed back to
  the driver on every launch, and `.seeded.*` markers piled up in the cache
  directory. It now probes with `OpenProcess` / `GetExitCodeProcess`.
- **f16 values round to nearest on AMD's proprietary driver.** SPIR-V leaves the
  rounding of a conversion to `float16_t` up to the driver unless the shader
  declares one, and AMD's Windows driver truncates on RDNA2 (a Ryzen iGPU) — so
  every f16 the shaders wrote, KV-cache entries and activation scales included,
  came out up to one ulp low, and that device's results drifted from every
  other's (`copy_f32_to_f16_cast_matches_host` and the expert-parallel test
  failed on it). A known-answer dispatch at device creation now checks how the
  driver rounds; where it truncates, every kernel with an f16 type declares
  `RoundingModeRTE` for 16-bit floats, patched into the module at pipeline
  creation (`infr_vulkan::spirv`). Devices that already round to nearest keep
  their modules unchanged — declaring the mode there is not neutral, and moved a
  GPU golden on an RDNA3 card.
- **`infr compare` finds `llama-diffusion-cli` on Windows.** The `PATH` and
  fork-build lookups probed for the file without `.exe`, so the diffusion
  comparison always reported the binary missing. Fork-build lookup also checks
  `bin/Release/` for multi-config CMake builds, retaining `bin/` precedence.
- **rustls 0.23.45** (from 0.23.41) for
  [RUSTSEC-2026-0285](https://rustsec.org/advisories/RUSTSEC-2026-0285): rustls
  accepted TLS 1.3 handshake messages sent at the wrong encryption level. It is
  the TLS stack under every `infr pull`; the weekly `cargo-deny` job had failed
  on it.
- **`infr` can find a global config file on Windows.** `config::file::discover`
  resolved `$XDG_CONFIG_HOME`, else `$HOME/.config`, else nothing — and Windows
  sets neither variable, so the third lookup step silently never found anything
  there. It now resolves through `dirs::home_dir()`, which reads the
  `FOLDERID_Profile` Known Folder. `--config <path>` and `./infr.toml` were
  unaffected and are unchanged. The layout stays `~/.config/infr/config.toml` on
  every platform, deliberately: adopting `dirs::config_dir()` would have moved
  the file to `~/Library/Application Support` for existing macOS users. An
  absolute `$XDG_CONFIG_HOME` still wins; a relative or empty one is now ignored
  rather than honoured, per the spec.
- **The Hugging Face cache is in the right place on macOS and Windows.**
  `Store::discover` fell back to `dirs::cache_dir()`, which is
  `~/Library/Caches` on macOS and `%LOCALAPPDATA%` on Windows —
  `huggingface_hub` uses `$XDG_CACHE_HOME`, else `~/.cache`, on every platform
  with no native-directory arm, so infr agreed with it only on Linux and would
  re-download models `hf download` had already fetched. The resolution now
  matches `huggingface_hub`'s exactly, including the `HUGGINGFACE_HUB_CACHE`
  legacy variable, which was not read at all, and treating an empty `HF_HOME=`
  as unset rather than resolving the cache to the relative path `hub`. A cache
  an older infr left in the OS-native directory is still used while the standard
  one does not exist, with a warning naming both, so upgrading does not silently
  re-fetch gigabytes.

- **macOS reports host memory, and Windows respects a container limit.**
  `infr_plat::mem::available` returned `None` on macOS, so anything sizing a
  host arena from it — the DRAM paging tier — stayed off on every Mac however
  much RAM it had. It now probes `host_statistics64` and reports what is
  reclaimable without swapping (`free - speculative + inactive + purgeable`;
  `free_count` already includes speculative pages, so adding them again would
  double-count). On Windows, `ullAvailPhys` is machine-wide and knew nothing
  about a Job Object's commit limit, so a process inside a Windows container
  could size an arena from far more memory than it was allowed — it is now
  clamped by `QueryInformationJobObject`'s binding limit, the same policy the
  Linux cgroup clamp already applied and now literally the same code. Both
  figures carry their provenance in `Available::source`.

- **The GPU MoE router no longer corrupts routing above 256 experts.**
  `moe_topk.comp` sized its per-expert scratch `ssel_adj` for 256 while the
  selection scan beside it was written for the chunked headroom above that, and
  filled and read the array across the full expert count unconditionally — so a
  model with more experts than the array holds wrote and read past it, which is
  undefined behaviour in GLSL (corrupt routing weights or a driver hang, not a
  clean failure). The array is now sized from the same constants the scan uses,
  and `Recorder::moe_topk` refuses an expert count above the ceiling with a
  message naming both numbers instead of dispatching. No supported model
  exceeded the old bound, so nothing shipped was affected; the Qwen3.8 MoE
  models, at 512 experts, are the first that would have been.

- **Ctrl-C stops a download.** `infr pull` installed the `SIGINT`/`SIGTERM`
  handlers but nothing on the download path read the latch, so the first Ctrl-C
  appeared to do nothing and only `SIGKILL` ended a multi-gigabyte pull. Every
  loop that moves bytes now polls it: `pull::fetch_all` stops claiming files,
  `ranged::worker` stops claiming chunks, and both `download::stream_into` and
  the ranged chunk reader stop mid-body per 64 KiB — the last one matters
  because a chunk is 64 MiB, so checking only between chunks left the process
  running for seconds after the signal. The partial and its resume sidecar are
  kept in every case, so the next `infr pull` continues where it stopped; the
  exit status is the conventional 130 / 143. An interrupted transfer reports
  `Error::Aborted` rather than a download failure.

- **Cached models are readable on Windows.** `link_blob` wrote its snapshot
  symlink with the forward-slash target `../../blobs/<hex>` that
  `huggingface_hub` and llama.cpp use. Windows accepts that in an ordinary path
  but not inside a symlink's reparse point: the link was created and every read
  through it failed with `ERROR_INVALID_NAME`. The target now uses the
  platform's own separator.

- **Every DeepSeek V2/V3 model added its attention output to the residual stream
  twice.** The MLA mixer arm in `seam/runner.rs` pushed its own `hidden += sub`
  and then fell through to the shared post-mixer residual add, so every layer of
  every `deepseek2`-family model (V2, V2-Lite, V3, V3.1, V3.2) ran
  `x + 2·Wo·attn` instead of `x + Wo·attn` — on the CPU, Vulkan and Metal
  backends alike, because the defect was in the shared graph rather than in a
  kernel. V4 is NOT affected: it builds `MixerW::Dsv4`, whose hyper-connection
  POST closes the residual wrap in place of the shared add, so it never emitted
  the second one. It never produced obviously broken text, which is why it
  survived: the only thing that catches it is an external oracle. Scored against
  llama.cpp 030ebb5's own logits over its own token ids, on
  `JenniSD/DeepSeek-V2-Lite-Chat-Q4_K_M-GGUF`, next-token probability cosine
  (CPU / Vulkan) goes 0.956/0.955 → 0.997/0.995 at 2 tokens, 0.334/0.316 →
  0.9998/0.99998 at 10, 0.862/0.862 → 0.997/0.999 at 27, and 0.980/0.920 →
  0.9999/0.9997 at 56. `infr`'s greedy continuation of "The capital of France
  is" is now llama.cpp's token for token, and the Vulkan repetition loop that
  DeepSeek-V2-Lite fell into after a dozen tokens is gone.

- **Every DeepSeek model prefilled one token per submit.** The batched-prefill
  eligibility scan (`moe_batched_ok` in `seam/runner.rs`) demanded a dp4a-mmq
  expert bank on EVERY layer, but DeepSeek's leading dense blocks ship a plain
  `ffn_gate/up/down` and no `ffn_*_exps` tensor at all — the missing lookup read
  as "unsupported dtype" and disqualified the whole model, so the entire prompt
  went through the per-token decode path. The scan now covers the layers that
  actually hold expert banks (`Config::is_moe_layer`), which also picks up
  llama4 checkpoints with a non-`1` `interleave_moe_layer_step`. Measured on an
  RX 7900 XTX: DeepSeek-V2-Lite-Chat Q4_K_M pp512 39.1 → 975.5 t/s and pp2048
  12.3 → 673.4 t/s; DeepSeek-MoE-16B-Chat Q4_K_M pp512 229.0 → 4185.0 t/s, level
  with llama.cpp's Vulkan backend on the same file. Decode is unchanged.
- **The int8 cooperative-matrix prefill GEMM assumed a fragment layout no device
  was ever asked about.** `native_gemm_i8cm_q8_0.comp` reads its accumulator
  elements at `(row, col) = (2*i + (lane>>4), lane&15)`, a mapping
  `KHR_cooperative_matrix` fixes per IMPLEMENTATION and which was derived
  empirically on RADV/RDNA3 — on a driver that lays the fragment out differently
  the kernel would have returned plausible wrong numbers with no error. The
  Vulkan backend now multiplies two known matrices on the device and reads the
  product back through that same mapping before arming the tier
  (`INFR_I8_COOPMAT=1` runs the check at init; a mismatch logs the offending
  element and leaves the tier off). Verified passing on an RX 7900 XTX, and
  verified to REFUSE when the mapping is perturbed.
- **Two drivers that misreport cooperative-matrix support are no longer
  believed.** AMD's proprietary and AMDVLK drivers advertise the unit on all
  GPUs, and Intel pre-Xe2 regresses on it (both documented upstream in
  llama.cpp). Cooperative matrix is now refused on `AMD_PROPRIETARY` /
  `AMD_OPEN_SOURCE` below RDNA3, and Intel pre-Xe2 keeps only the 8x8x16 tile
  that already required `INFR_CM_8X8=1`. The device is bucketed by probes (wave
  mode, wavefronts/SIMD, packed-dot bits, subgroup width, warps/SM), never by
  PCI device id, and the bucket plus driver id are printed in the GPU banner.
  Mesa RADV and NVIDIA are unaffected.
- **bf16 and fp8 were gated on an extension STRING with no feature bit, and the
  features were never enabled on the device.** `VK_KHR_shader_bfloat16` /
  `VK_EXT_shader_float8` now go through their feature structs like every other
  capability: `Capabilities::bf16` / `f8` mean the device will run it, the bf16
  and fp8 coopmat tiers additionally require `shaderBFloat16CooperativeMatrix` /
  `shaderFloat8CooperativeMatrix`, and both extension and feature are enabled on
  the logical device when a tier is live — without which those kernels' SPIR-V
  violated its VUID.
- **`maxPushConstantsSize` and `maxStorageBufferRange` are queried instead of
  assumed.** A kernel whose push block exceeds the device limit is refused by
  name at build time rather than failing a driver-side VUID, and the
  descriptor-range bind check now runs in RELEASE builds against the limit the
  device reported (it was a `debug_assert!` against an assumed 4 GiB).
- **DeepSeek `deepseek-llm` pre-tokenizer split on the wrong character
  classes**: `DEEPSEEK_LLM_PRE_RES[2]` opened its quote range with `'` (U+0027)
  instead of `‘` (U+2018), so the class covered U+0027–U+201F — most of the BMP,
  including the ASCII digits, Hebrew, Arabic and Devanagari — rather than the
  eight quote characters; and `DEEPSEEK_LLM_PRE_RES[1]` had `ℹ-ℿ` where
  llama.cpp has `ℹℼ-ℿ`, adding U+213A and U+213B to the letter class. Both moved
  chunk boundaries silently, with no error: on DeepSeek-V2-Lite-Chat the first
  one changed real token ids wherever a non-ASCII space (U+00A0, U+0085)
  preceded punctuation, e.g. `"a \u{00A0}. b"` tokenized as
  `[64, 30683, 13, 270]` where llama.cpp gives `[64, 207, 1202, 13, 270]`. All
  fourteen DeepSeek regexes now match `llama-vocab.cpp` codepoint for codepoint,
  `infr` agrees with `llama-tokenize` on every text tried, and
  `deepseek_pre_split_boundaries_match_the_reference_lists` pins the chunk
  boundaries of all three lists without needing a model file.
- **DeepSeek V3 router bias applied to the wrong scores (Vulkan)**: `moe_topk`
  added `exp_probs_b` to the raw router LOGITS, but llama.cpp (and the CPU
  interpreter) add it to the gated PROBS
  (`selection_probs = ggml_add(probs, exp_probs_b)`). `probs + bias` is not
  monotone in `logits + bias`, so a real V3/V3.1 model (sigmoid gated, noaux_tc
  bias) routed to different experts than the reference. The shader now selects
  on `gating(logit) + bias`; the no-bias paths are unchanged. Caught by the new
  `moe_groups_bias_parity` op test, whose data makes the two semantics disagree.
- **Metal `mla_f16kv_ff` (YaRN twin) bound its buffers at the wrong indices**:
  the kernel declared `MlaParams` at buffer(5) and `freq_factors` at buffer(6),
  but `exec.rs` binds `[q, k, wk_b, wv_b, dst, ff]` with params at
  `bufs.len() = 6` — the shader would have read the 4-byte divisor buffer as its
  params struct. Never executed until the new `mla_parity`/`mla_ff_parity` tests
  ran it on the macOS CI job; buffer declarations swapped to match the binding
  convention.

- **DeepSeek V2/V3 decode writes every token's K to row 0**: the seam's
  `dyn_replay` gate (which must mirror the Vulkan adapter's record-once-replay
  eligibility) had no `Op::Mla` exclusion, so a DeepSeek MLA model built the
  replay tape with `pos = 0` baked into every `WriteKv`. The adapter rejects the
  MLA graph as ineligible and falls back to the static path — executing that
  baked `pos = 0` tape for every decode token, so each token's K row landed in
  row 0 and rows 1.. never got populated. Attention then ran against a one-row
  cache: logits diverged wholesale (GPU top-1 was unrelated to the CPU's, cosine
  ~0.5). Added the `c.deepseek2` exclusion to the gate so MLA models take the
  per-token static decode path.
- **GPU MoE routing nondeterminism**: `Recorder::moe_topk` bound
  `[logits, ids, wts, bias]` with `n_out = 2`, so the hazard tracker treated
  `ids` as a read and `bias` as a write — but the shader writes `ids` and only
  reads `bias`. The `ids` write never reached the RAW check, so the per-expert
  GEMVs that index the weight bank by `ids[slot]` got no barrier before reading
  it, and the pooled ids buffer (reused across layers) fed stale/torn expert ids
  into the routing — DeepSeek V2/V3 GPU logits changed run-to-run. Reordered the
  dispatch to `[logits, bias, ids, wts]` (and the shader bindings to match) so
  the two written buffers sit in the trailing write slots.
- **DeepSeek MLA absorption transposition**: the CPU/Vulkan/Metal MLA kernels
  read the per-head `attn_k_b` weight transposed, computing `W @ q_nope` where
  the absorbed-form math needs `Wᵀ @ q_nope` — the file stores it as
  `[qk_nope_dim, kv_lora_rank]` per head with the qk_nope dim fastest. The
  regression entered with the `6adb2de` "transpose fix" (which applied its own
  diagnosis backwards: the diff flipped `i` from fast to slow) and was then
  pinned by parity-test references written to the same wrong layout. The `wv_b`
  output projection had the identical transpose (`[kv_lora, v_head_dim]` read
  with the v_head_dim dim fastest) — invisible to every test because the parity
  synthetic `wv_b` is an identity. Both are now read file-order (the fast dim
  contracted) in all three backends and both parity references; the A/B check on
  `wv_b` shows the transposed orientation collapses the pearson correlation with
  llama.cpp from 0.79 to 0.10 and turns generation into `"ARSARSARS…"` garbage.
  The `mla_matches_cpu_reference` parity test dispatches two attended keys with
  random K rows so the absorbed-query scores actually shape the output (at
  `kv_len=1` softmax is trivial and the old test could not detect a
  transposition).
- **DeepSeek V2/V3 YaRN frequency ramp missing**: the GGUF declares
  `rope.scaling.type = yarn`, `factor = 40`, `original_context_length = 4096`,
  `yarn_log_multiplier = 0.0707` — which makes llama.cpp set
  `yarn_ext_factor = 1.0` and run the full per-dimension frequency ramp at EVERY
  context length. infr roped with plain `θ^(−2p/d)`, so its q_pe/k_pe
  frequencies were up to 40× off: the model output was coherent-looking but
  wrong (`"Reply Collabor…"` garbage, top-1 unrelated to llama.cpp's). The seam
  now computes the corr_dims spectral ramp divisors and feeds them through
  `Op::Rope.freq_factors` (k_pe) and the MLA kernels' internal q_pe rope (Vulkan
  `mla_ff`/`rope_ff` builds, Metal twin); the YaRN mscale² is a constant folded
  into `mla_scale = mscale²/√(qk_nope + qk_rope)` (√192 for V2-Lite, not √576 —
  `qk_nope = head_k_mla = 128`), replacing the previous context-length-gated
  approximation. The rope vector mscale cancels to `rope_attn_factor` for
  deepseek2, so the kernels need no vector scaling. Greedy output now matches
  llama.cpp's continuation " Paris." in content.
- **Vulkan hazard tracker mis-scoped the YaRN freq_factors dispatch**: the
  `mla_ff`/`rope_ff` dispatches bound the read-only ff divisors AFTER the output
  buffer with `n_out = 1`, so the tracker marked ff as the write and left the
  real output (dst / y) untracked — no store→read barrier before the
  wo-projection consumed the attention output, making DeepSeek V2 GPU logits
  nondeterministic run-to-run. Reordered both dispatches to `[.., ff, dst]` /
  `[x, ff, y]` and the shader bindings to match, so the trailing write slot is
  the actual output.
- Reject GGUF tensors whose encoded byte count overflows `usize` and model
  metadata with zero attention heads.
- Stop malformed pipe-format tool arrays from entering a non-progress allocation
  loop.
- Treat model JSON as a tool call only when the request offers a non-empty tool
  list.
- Publish graceful-shutdown state and its signal number atomically so
  interrupted CLI commands retain the correct exit status.
- Drop completed CPU spin-pool results when a sibling task panics.
- The CPU backend's dequantized-weight and Q4_K/Q6_K repack caches now key on a
  never-reused buffer id instead of a memory address. A `CpuBackend` that
  outlives a model — `infr serve` reloading one — could otherwise return a
  cached weight built from the PREVIOUS model, because both the allocator and a
  fresh mmap hand out addresses that were just freed.

### Security

- Update `crossbeam-epoch` 0.9.18 → 0.9.20 for RUSTSEC-2026-0204 (invalid
  pointer dereference in the `fmt::Pointer` impl for `Atomic`/`Shared`). Reached
  through `rayon`, so it applies to every CPU-backend build.

[Unreleased]: https://github.com/kryptic-sh/infr/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/kryptic-sh/infr/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/kryptic-sh/infr/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kryptic-sh/infr/releases/tag/v0.1.0
