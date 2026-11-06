# infr

[![CI](https://github.com/kryptic-sh/infr/actions/workflows/ci.yml/badge.svg)](https://github.com/kryptic-sh/infr/actions/workflows/ci.yml)

Pure-Rust LLM inference engine. Vulkan-first, built to run on any mainstream
GPU.

> **This is the [KessokuNet](https://github.com/KessokuNet/infr) fork of
> [kryptic-sh/infr](https://github.com/kryptic-sh/infr).** It focuses on
> optimizing and backporting infr to older and lower-end Vulkan GPUs — AMD
> GCN / Polaris (RX 470/480/570/580) is the baseline target: wave64-only
> devices, no cooperative matrix, no hardware int8 dot. Fixes that are not
> specific to those GPUs are meant to go upstream. See
> [docs/legacy-gpus.md](docs/legacy-gpus.md) for what changed and the measured
> results.

> Early WIP. The only non-Rust parts are the GPU driver calls (Vulkan via `ash`)
> and the compute shaders (SPIR-V).

## What this fork adds

Upstream `main` is merged in; this list is what it does not have yet. Work here
that is not specific to older GPUs is meant to go upstream.

**Both directions of KV paging**, neither of which upstream has:

- **`kv.store_mb` (`INFR_KV_STORE_MB`) — reverse PagedAttention.** A prompt's
  K/V rows are captured in 128-token blocks, chained by a hash of
  `(parent id, tokens)`, and kept in host RAM under an LRU byte budget. The next
  request walks the chain from token 0 as far as its prompt agrees, then uploads
  the matched run in one contiguous copy instead of re-prefilling it. Gemma 4 E4B
  on the RX 580: a 1068-token prompt took 3677 ms cold, 319 ms on a repeat,
  705 ms for a different 1090-token tail — the shared 1053 tokens stored once,
  111.8 MB for both conversations.
- **`kv.paged` (`INFR_KV_PAGED`) — forward PagedAttention.** A session's slots
  share one pool of 512-row pages; a fork points at its parent's complete pages
  and copies a page before writing it. It costs the device bytes the linear slots
  already cost (one block per slot the session may fork) and is built at the
  first fork. Ring, recurrent, MLA and diffusion layers stay compact, so a
  full-attention model gains most. Memory, not speed: reads pay one table lookup
  per KV tile.

The two compose — the pool shares pages between live slots, the store survives
slot recycling. The DRAM KV spillover (`kv.overflow`, `kv.overflow_vram_mb`,
`kv.overflow_reserve_mb`) is upstream's. Mechanism, limits and measurements:
[`docs/perf/kv-prefix.md`](docs/perf/kv-prefix.md).

**Older GPUs** ([`docs/legacy-gpus.md`](docs/legacy-gpus.md)) — wave64-only GCN
support, an fp32 `kq_fma` prefill GEMM and decode/verify GEMV for devices without
hardware int8 dot, the first prefill split on slow discrete parts, a compute-only
queue family so a long forward does not hold up the compositor. RX 580 against
llama.cpp: 1.13× at `pp512` and `tg128` on Qwen2.5-Coder-3B, level on Gemma 4
E4B, 0.58–0.64× at `pp4@d4096` — the four-token turn where this fork is still
behind.

**Models and formats** — Gemma 4 image input (`serve.mmproj`,
`serve.image_max_tokens`), EXL3 safetensors checkpoints (Qwen2, Qwen3.5 text,
Gemma 4 dense; [`docs/exl3.md`](docs/exl3.md)), the Seed-Coder pre-tokenizer
(`tokenizer.ggml.pre = seed-coder`) that Zeta 2/2.1 and Seed-Coder GGUFs need,
and a dense Qwen4 reference path used for experiments
([`docs/qwen4-exp.md`](docs/qwen4-exp.md)).

**HTTP and CLI** — `/v1/completions` (see
[FIM and edit prediction](#fim-and-edit-prediction)), `infr ls`, `infr rm`,
`infr perplexity`, and a Chrome-trace timeline behind `INFR_PROF_TRACE`.

## Goal

A from-the-metal inference server that works across AMD / NVIDIA / Intel
(Vulkan) and Apple (native Metal), plus a CPU reference — three backends behind
one `Backend` trait.

## Status

Runs **Llama / Qwen2 / Qwen3** (dense), **Gemma 3** (dense, sliding-window
attention + QK-norm + GeGLU), and **Gemma 4** (per-layer heterogeneous head
dims, proportional RoPE, V-norm, per-layer output scale — including the **E2B**
variant: per-layer input embeddings, per-layer FFN widths, KV-layer sharing) on
the Vulkan backend, competitive with llama.cpp at long context (`infr compare`).
**Qwen3.5 / Qwen3.6** (`qwen35` — hybrid gated-DeltaNet + attention, a sibling
of Qwen3-Next) run on the same unified runner, CPU + Vulkan (`docs/qwen35.md`).
**DiffusionGemma** (the original target — block text-diffusion MoE on a Gemma-4
backbone, entropy-bound denoise decode) runs end-to-end on CPU + Vulkan
(`docs/diffusion-gemma.md`).

```bash
infr pull   <model-ref>        # org/repo[:quant] (HuggingFace) | path to a .gguf
infr run    <model-ref> [msg]  # terminal chat (auto-pulls)
infr serve  <model-ref>        # OpenAI-compatible HTTP API
infr bench / infr compare      # tok/s benchmarks vs llama.cpp
```

Model refs match llama.cpp's `-hf`: `org/repo[:quant]` (quant default `Q4_K_M`,
e.g. `infr run unsloth/Qwen3-14B-GGUF:Q4_K_M`). Models share the standard
**HuggingFace Hub cache** (`~/.cache/huggingface/hub`) with llama.cpp and
`huggingface_hub` — one download, used by both.

## Supported models

All run on the Vulkan GPU backend unless noted. The chat template (turn markers,
system prompt) is read from the GGUF's own `tokenizer.chat_template`.

| Family            | Arch (GGUF)       | Notes                                                   |
| ----------------- | ----------------- | ------------------------------------------------------- |
| Llama             | `llama`           | dense transformer                                       |
| Llama 4           | `llama4`          | sigmoid top-1 MoE + shared expert, iRoPE, paged experts |
| Qwen2 / Qwen2.5   | `qwen2`           | dense, QKV bias, NEOX rope                              |
| Qwen3             | `qwen3`           | dense, QK-norm                                          |
| Qwen3 MoE         | `qwen3moe`        | softmax router, top-_k_ experts, paged experts          |
| Gemma 3           | `gemma3`          | SWA + QK-norm + GeGLU, dual-RoPE                        |
| Gemma 4 (dense)   | `gemma4`          | per-layer head dims, proportional RoPE, V-norm          |
| Gemma 4 **E2B**   | `gemma4`          | + per-layer input embeddings / FFN, KV sharing          |
| Gemma 4 **MoE**   | `gemma4`          | 26B-A4B: dual FFN (dense GeGLU ∥ 8-of-128 routed), AR   |
| Qwen3.5 / Qwen3.6 | `qwen35`          | hybrid gated-DeltaNet + attention (NOT `qwen3next`)     |
| Qwen3.6 MoE       | `qwen35moe`       | `qwen35` skeleton + routed experts + shared expert      |
| DiffusionGemma    | `diffusion-gemma` | block text-diffusion MoE, entropy-bound denoise decode  |
| BitNet b1.58      | `bitnet`          | llama skeleton + SubLN, ternary TQ2_0 weights           |
| BitNet b1.58 (MS) | `bitnet-b1.58`    | same, for the `microsoft/bitnet-b1.58-*` GGUFs (i2_s)   |
| Zeta 2 / 2.1      | `llama`           | Zed edit prediction; Seed-Coder-8B, FIM/rewrite         |

Fine-tunes on any of these backbones run unchanged. **Ornith-1.0**
(DeepReinforce.AI agentic-coding) validated 2026-07-09 — the 9B rides `qwen35`
and the 35B rides `qwen35moe` with no code changes
(`infr run deepreinforce-ai/Ornith-1.0-9B-GGUF:Q4_K_M "..."`).
**Ternary-Bonsai** (Prism ML, weights trained to {-1, 0, +1}) validated
2026-07-12 — the 1.7B / 4B / 8B all ride `qwen3`, zero-code, both in the TQ2_0
repack (`superkaiii/Ternary-Bonsai-4B-GGUF`) and in llama.cpp's new **Q2_0**
weight dtype (2.25 bpw, GGML type 42 — native in-shader dequant + dp4a mmq, no
fork needed). infr is the **only engine that runs Q2_0 on a GPU** (llama.cpp
merged the dtype CPU-only) — numbers in
[`docs/perf/results.md`](docs/perf/results.md). Pull the `Q2_0_g64` files:
`infr run prism-ml/Ternary-Bonsai-8B-gguf:Q2_0_g64 "..."`.

```bash
# Qwen3 dense
infr run unsloth/Qwen3-1.7B-GGUF:Q4_K_M "What is the capital of France?"

# Qwen3 MoE (experts page through the VRAM LRU cache when they don't fit —
# see docs/config.md)
infr run unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M "Explain MoE routing."

# Llama 4 Scout (37 GB Q2_K) — paged expert cache runs it on a 24 GB card
infr run unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF:Q2_K "What is the capital of France?"

# Gemma 3
infr run unsloth/gemma-3-1b-it-GGUF:Q4_K_M "What is bash?"

# Gemma 4 — dense and the E2B variant
infr run unsloth/gemma-4-12b-it-GGUF:Q4_K_M  "What is the capital of France?"
infr run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M  "What is bash?"

# DiffusionGemma — block text-diffusion decode (entropy-bound denoise)
infr run unsloth/diffusiongemma-26B-A4B-it-GGUF:Q4_K_M  "What is the capital of France?"

# Pick a specific quant with the `:quant` suffix (default is Q4_K_M)
infr run unsloth/Qwen3-8B-GGUF:Q6_K       "Summarize the plot of Hamlet."
infr run unsloth/Qwen3-0.6B-GGUF:IQ4_XS   "Write a haiku about Rust."

# MTP speculative decoding is currently DISABLED (rationale in docs/mtp.md).
# INFR_MTP=1 is ignored with a warning; MTP-head models run the ordinary decode
# path (their `nextn` tensors are simply unused) and are otherwise fully supported.
infr run unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_XL "Explain how a hash map works."

# Sampling defaults to the model's own recommended values; override per run:
infr run unsloth/Qwen3-1.7B-GGUF:Q4_K_M "Tell me a story." \
  --temp 0.7 --top-k 40 --top-p 0.95
```

## Configuration

Everything the engine can be told — device, context, sampling, KV format, paging
budgets, every kernel-tier switch — is one typed value resolved once at startup
from **four layers, later wins**:

```
defaults  <  config file (TOML)  <  INFR_* environment  <  CLI flags / --set
```

The config file is the **first existing** of `--config <PATH>` (an error if that
path does not exist), `./infr.toml`, then `$XDG_CONFIG_HOME/infr/config.toml`
(else `~/.config/infr/config.toml`). First match wins — there is no merging
across files, and finding no file is a no-op.

```toml
# ./infr.toml — see infr.example.toml for a commented starting point
[device]
ctx = "32k"

[kv]
type_k = "q8_0"

[kernels.vulkan]
flash_splits = 2
gemm_warp = false     # the file speaks the POSITIVE field names
```

**Every documented `INFR_*` variable still works** — nothing was renamed; the
variables now feed the same resolved value the file and the flags do. Knobs
without a dedicated flag are reachable with `--set <config.path>=<value>`, which
takes the same paths as the file:

```bash
infr bench "$M" -p 512 -n 0 --set kernels.vulkan.flash_splits=2
```

Where a bespoke flag and a `--set` name the same field, the flag wins and says
so (`--ctx 4096 --set device.ctx=8192` runs at 4096 and prints a warning).

Full reference — the per-section walkthrough, `--set` semantics, the unknown-key
behaviour, and the handful of `INFR_*` keys that are deliberately not
configuration — is in [`docs/config.md`](docs/config.md).

### Serving

```bash
# OpenAI-compatible HTTP API (streaming). Reuses a persistent KV cache across
# requests (common-prefix diff) for fast TTFT on shared-prefix chats.
infr serve unsloth/Qwen3-14B-GGUF:Q4_K_M          # default: 127.0.0.1:8080

curl -s localhost:8080/v1/chat/completions -d '{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "stream": true
}'
```

Works as a drop-in backend for OpenAI-API clients (opencode, the Claude Code
CLI, etc.). Tool calling renders the model's own `tokenizer.chat_template`
(Qwen, Llama-3.x, Gemma tool dialects supported).

`GET /metrics` serves Prometheus counters and gauges (`infr_requests_total`,
`infr_generation_tokens_total`, `infr_decode_seconds_total`,
`infr_requests_running`, `infr_slots_busy`,
`infr_device_memory_available_bytes`, …). `GET /status` returns the same picture
as JSON — each model's backend, device, context per slot, slots and device
memory. Both sit behind the API key when one is set.

**Images (Gemma 4).** OpenAI `image_url` parts with a base64 `data:` URI (PNG/JPEG/GIF/WebP/BMP)
are accepted on the Vulkan backend when a Gemma 4 vision projector is available — the first
`mmproj*.gguf` beside the model (`infr pull unsloth/gemma-4-E4B-it-GGUF:mmproj-F16.gguf` puts it
there), or `serve.mmproj` / `INFR_MMPROJ`. The vision encoder runs on the CPU (~1–3 s per image on
an 8-core desktop, cached per image so follow-up turns don't pay it again); `serve.image_max_tokens`
/ `INFR_IMAGE_MAX_TOKENS` (default 280, range 70–1120) trades detail for encode time. Remote image
URLs are not fetched.

`--temp` / `--top-k` / `--top-p` set the SERVER defaults (`--temp 0` = greedy);
a per-request OpenAI `temperature`/`top_p` still overrides them. See
[Configuration](#configuration).

### FIM and edit prediction

`/v1/completions` tokenizes the prompt verbatim, with no chat template. FIM
tokens in it (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, Seed-Coder's
`<[fim-prefix]>` …) encode as their single special ids and the model continues
the middle, so any FIM or next-edit model on a supported backbone works — the
client builds the prompt. `suffix` returns an explicit error instead of being
ignored; pass the model's own FIM tokens in `prompt`.

Zeta 2.1, Zed's edit-prediction model on an 8B Seed-Coder backbone, runs at
5.1 GB Q4_K_M: resident at `--ctx 8k` on an 8 GB card, 24–34 tok/s decode.

```bash
infr serve mradermacher/zeta-2.1-GGUF:Q4_K_M --ctx 8k --temp 0.2
```

Zed — `settings.json`, provider: OpenAI-compatible:

```json
{
  "edit_predictions": {
    "provider": "open_ai_compatible_api",
    "open_ai_compatible_api": {
      "api_url": "http://127.0.0.1:8080/v1/completions",
      "model": "zeta-2.1.Q4_K_M",
      "prompt_format": "zeta2_1",
      "max_output_tokens": 512
    }
  }
}
```

Zed formats the context itself: related files and edit history as
`<[fim-suffix]>`/`<[fim-prefix]>` blocks, the editable region marked with
`<|marker_N|>`, and `<[end▁of▁sentence]>` as the stop. The model's reply is the
marker-tagged replacement span, which Zed splices into the buffer. The prompt
budget is 4k tokens and generation is capped at `max_output_tokens * 4`, so
`--ctx 8k` leaves headroom. Requests are non-streaming, one per typing pause,
and the prefix cache absorbs the context that repeats between them.

`prompt_format` selects the layout: `zeta2_1` for Zeta 2/2.1, `zeta` for Zeta 1,
`qwen` for Qwen2.5-Coder, or `code_llama` / `star_coder` / `deepseek_coder` /
`codestral` / `glm` / `sweep` / `infer`.

## Performance

Measured against llama.cpp on an **AMD Radeon RX 7900 XTX** (RDNA3, Vulkan /
RADV), every validated model × quant, both engines on matched flags. Headline:

- **Decode** — the reproducible half — wins **29 of 35** rows at `tg128` and
  **24 of 35** at `tg64@d4096`.
- **`pp4@d4096`** (multi-turn ingest, the shape a coding agent actually runs) is
  the strongest column, roughly **1.5–2×** on the small and mid models.
- Losses concentrate on **Qwen3-14B and the larger MoEs**, mostly at depth.

**The full table, the per-row footnotes, and an honest account of where infr
loses are in [`docs/perf/results.md`](docs/perf/results.md).** Two caveats live
there and both matter: ratios move with _both_ engines, so snapshots taken
against different `llama-bench` builds are not comparable; and infr's
**prefill** columns vary up to ~30% run-to-run on an identical binary (a known
tier/chunk nondeterminism), so quote prefill to one significant figure and
decode as written.

To reproduce or extend the numbers — `infr bench` / `infr compare --sweep`
flag-for-flag against `llama-bench`, plus per-op GPU profiling — see
[`docs/perf/benchmarking.md`](docs/perf/benchmarking.md). The optimization
method and the recorded dead ends are in
[`docs/perf/playbook.md`](docs/perf/playbook.md); everything performance-related
is indexed at [`docs/perf/`](docs/perf/README.md).

> MTP self-speculative decode is currently **parked** — `INFR_MTP=1` is ignored
> with a warning and MTP-head GGUFs run the ordinary decode path. Rationale in
> [`docs/mtp.md`](docs/mtp.md).

## Scope

- **Format:** GGUF
- **Models:** Llama, Qwen2/2.5, Qwen3 (dense + MoE), Gemma 3, Gemma 4 (dense +
  E2B + 26B-A4B MoE), Qwen3.5/3.6 (dense + MoE) — all on GPU **and** the CPU
  reference; DiffusionGemma (block text-diffusion, CPU + GPU); Llama 4 (Scout —
  GPU by default via the paged expert cache, 37 GB Q2_K on a 24 GB card; pure
  CPU under `--dev cpu`); BitNet b1.58 (`bitnet` and Microsoft's `bitnet-b1.58`,
  ternary weights, CPU + GPU)
- **GPU:** AMD / NVIDIA / Intel via Vulkan (cooperative-matrix matmul); Apple
  via a native **Metal backend** (`--dev metal`) covering every op the CPU
  reference does — dense, MoE (`qwen3moe`) and Qwen3.5 (`qwen35`). Dense is
  optimized (simdgroup-matrix GEMM + flash attention, raw-block quant decode;
  within ~1.3-1.5× of llama.cpp Metal on M3 Pro — architecture and numbers in
  [`docs/metal.md`](docs/metal.md))
- **Store:** the shared **HuggingFace Hub cache** — located via `$HF_HUB_CACHE`,
  else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`, in HF's own
  `models--<org>--<repo>/{blobs,snapshots,refs}` layout. A model pulled by
  `infr`, `llama.cpp`, or `huggingface_hub` is shared — downloaded once.
  `infr pull` fetches from `huggingface.co` over resumable HTTP Range with a
  progress bar; gated repos authenticate with `HF_TOKEN`.
- **API:** OpenAI-compatible HTTP (streaming) — works with opencode / Claude
  Code CLI / Zed edit prediction

## Architecture

```
server   axum + SSE  ->  OpenAI /v1
chat     ChatModel        (autoregressive dense/MoE/qwen35; DiffusionGemma's block-diffusion loop)
runtime  SeamModel        tensors, KV cache, command/descriptor management (the unified runner)
loader   WeightSource     (Gguf; safetensors later)
compute  Backend          (Vulkan via ash + SPIR-V; native Metal via MSL; CPU reference)
```

## Install

```bash
brew install kryptic-sh/tap/infr          # macOS
yay -S infr-bin                           # Arch Linux (AUR)
scoop bucket add kryptic-sh https://github.com/kryptic-sh/scoop-bucket
scoop install infr                        # Windows
cargo install infr-cli --locked           # from source; needs glslc, see below
```

Prebuilt archives for Linux (x86_64, aarch64), macOS and Windows are on the
[releases page](https://github.com/kryptic-sh/infr/releases). The crate is
`infr-cli` because `infr` on crates.io is another project; it installs the
`infr` binary.

## Building

```bash
cargo build --release -p infr-cli   # → target/release/infr
```

Needs a stable Rust toolchain and `glslc` (shaderc; ships with the Vulkan SDK),
which compiles the compute shaders at build time. At runtime only a GPU driver
with Vulkan is required. Windows setup, step by step:
[`docs/windows.md`](docs/windows.md).

## Documentation

Deeper design docs, backend architecture, and performance material live in
[`docs/`](docs/README.md) — start with that index. Highlights:
[`docs/perf/`](docs/perf/README.md) (all performance: results, benchmarking,
optimization playbook, kernel coverage), [`docs/config.md`](docs/config.md) (the
configuration reference), [`docs/metal.md`](docs/metal.md) and
[`docs/igpu.md`](docs/igpu.md) (backends).

## License

[MIT](LICENSE)
