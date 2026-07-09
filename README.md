# infr

[![CI](https://github.com/kryptic-sh/infr/actions/workflows/ci.yml/badge.svg)](https://github.com/kryptic-sh/infr/actions/workflows/ci.yml)

Pure-Rust LLM inference engine. Vulkan-first, built to run on any mainstream
GPU.

> Early WIP. The only non-Rust parts are the GPU driver calls (Vulkan via `ash`)
> and the compute shaders (SPIR-V).

## Goal

A from-the-metal inference server that works across AMD / NVIDIA / Intel
(Vulkan) and Apple (MoltenVK), with native backends addable later behind a
`Compute` trait.

## Status

Runs **Llama / Qwen2 / Qwen3** (dense), **Gemma 3** (dense, sliding-window
attention + QK-norm + GeGLU), and **Gemma 4** (per-layer heterogeneous head
dims, proportional RoPE, V-norm, per-layer output scale — including the **E2B**
variant: per-layer input embeddings, per-layer FFN widths, KV-layer sharing) on
the Vulkan backend, competitive with llama.cpp at long context (`infr compare`).
**Qwen3.5 / Qwen3.6** (`qwen35` — hybrid gated-DeltaNet + attention, a sibling
of Qwen3-Next) run on the same unified runner, CPU + Vulkan (`docs/QWEN35.md`).
**DiffusionGemma** (the original target — block text-diffusion MoE on a Gemma-4
backbone, entropy-bound denoise decode) runs end-to-end on CPU + Vulkan
(`docs/DIFFUSIONGEMMA.md`).

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

| Family            | Arch (GGUF)       | Notes                                                  |
| ----------------- | ----------------- | ------------------------------------------------------ |
| Llama             | `llama`           | dense transformer                                      |
| Qwen2 / Qwen2.5   | `qwen2`           | dense, QKV bias, NEOX rope                             |
| Qwen3             | `qwen3`           | dense, QK-norm                                         |
| Qwen3 MoE         | `qwen3moe`        | softmax router, top-_k_ experts (CPU offload)          |
| Gemma 3           | `gemma3`          | SWA + QK-norm + GeGLU, dual-RoPE                       |
| Gemma 4 (dense)   | `gemma4`          | per-layer head dims, proportional RoPE, V-norm         |
| Gemma 4 **E2B**   | `gemma4`          | + per-layer input embeddings / FFN, KV sharing         |
| Qwen3.5 / Qwen3.6 | `qwen35`          | hybrid gated-DeltaNet + attention (NOT `qwen3next`)    |
| Qwen3.6 MoE       | `qwen35moe`       | `qwen35` skeleton + routed experts + shared expert     |
| DiffusionGemma    | `diffusion-gemma` | block text-diffusion MoE, entropy-bound denoise decode |

Fine-tunes on any of these backbones run unchanged. **Ornith-1.0**
(DeepReinforce.AI agentic-coding) validated 2026-07-09 — the 9B rides `qwen35`
and the 35B rides `qwen35moe` with no code changes
(`infr run deepreinforce-ai/Ornith-1.0-9B-GGUF:Q4_K_M "..."`).

```bash
# Qwen3 dense
infr run unsloth/Qwen3-1.7B-GGUF:Q4_K_M "What is the capital of France?"

# Qwen3 MoE (expert CPU offload with INFR_NCMOE=N for tight VRAM)
infr run unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M "Explain MoE routing."

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

# Qwen3.5 speculative decoding (opt-in MTP head; output is token-identical to
# greedy — pure speedup). WIP: fastest on real, content-rich prompts.
INFR_MTP=1 infr run unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_XL "Explain how a hash map works."

# Sampling: greedy by default (INFR_TEMP=0). Temperature / top-k / top-p:
INFR_TEMP=0.7 INFR_TOP_K=40 INFR_TOP_P=0.95 \
  infr run unsloth/Qwen3-1.7B-GGUF:Q4_K_M "Tell me a story."
```

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

Sampling is greedy at `INFR_TEMP=0`; otherwise `INFR_TEMP` / `INFR_TOP_K` /
`INFR_TOP_P` control it (see
[Benchmarking & profiling](#benchmarking--profiling) for the full env list).

## Benchmarking & profiling

`infr bench` matches `llama-bench`'s `-p`/`-n`/`-d`/`-r` flags, so the two are
directly comparable. Pipelines are compiled and GPU state is first-touched at
model load (`Llama::warmup`), so timing measures compute, not one-time setup.
**Run benchmarks one at a time** — concurrent GPU work skews results.

```bash
M='unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M'   # MoE perf target

# Prefill (pp = n_prompt/time) and decode (tg = n_gen/time):
infr bench "$M" -p 2048 -n 0 -r 3       # prefill 2048 tokens
infr bench "$M" -p 8000 -n 0 -r 2       # prefill at depth
infr bench "$M" -p 0 -n 64 -r 3         # decode 64 tokens
infr bench "$M" -p 0 -n 64 -d 2048      # decode at context depth 2048 (-d warms, untimed)
```

**Profile** per-op GPU time (timestamp queries) with `INFR_PROF2=1`. Every
dispatch is timestamped and labeled **automatically with its kernel name** (plus
a few role overrides like `expert_gateup`/`expert_down`); no manual stamping. It
prints one block per submit and ONE aggregated `INFR_PROF2 GPU report` at
process exit (per-kernel totals, counts, avg, %GPU over all timed submits —
warmup runs unprofiled). Add `INFR_PROF2_SHAPES=1` for shape-itemized GEMV/GEMM
buckets (`mmvr:m4:1536x24576`). Decode's replay tape carries no timestamps —
profile decode with `INFR_SEAM_NO_REPLAY=1`. Details in
[`docs/PERF.md`](docs/PERF.md).

```bash
INFR_PROF2=1 infr bench "$M" -p 2048 -n 0 -r 1 2>&1 | tail -30   # exit aggregate
```

**Compare to llama.cpp** — `infr compare` shells out to `infr bench` and the
system `llama-bench` with matching flags on coding-agent-shaped workloads
(prefill, decode-at-depth, whole turns). `--ctx` is comma-delimited:

```bash
infr compare "$M" --ctx 8000,16000 --gen 256 --turn 2048,256 --reps 2
```

**DiffusionGemma** has no upstream-merged `llama-bench` support, so
`infr compare`/`infr compare --sweep` route `arch=diffusion-gemma` models to a
different oracle: the reference fork's `llama-diffusion-cli`
(`~/Projects/mxaddict/llama.cpp-dg`, resolved via `INFR_LLAMA_DIFFUSION_CLI` >
`PATH` > the fork's `build-vulkan`/`build` directories — see
`ModelBench::llama_diffusion_cli_path` for the exact precedence and its PATH
fallback caveat). It prints two rows instead of the usual pp/tg matrix:
`dg-step` (in-step-parallel tok/s ratio — the apples-to-apples number, since
both implementations run entropy-bound and take a different number of denoise
steps) and `dg-e2e` (informational end-to-end tok/s, each side's own step count
folded into the row so the mismatch is visible). Details in
[`docs/DIFFUSIONGEMMA.md`](docs/DIFFUSIONGEMMA.md).

Useful env: `INFR_TEMP` / `INFR_TOP_K` / `INFR_TOP_P` (sampling; `TEMP=0` →
greedy), `INFR_MAX_NEW`, `INFR_MAX_CTX`, `INFR_NCMOE` (MoE expert CPU offload),
`INFR_NO_FLASH`.

## Validated models & performance

Everything below is **validated on an AMD Radeon RX 7900 XTX** (RDNA3, 24 GB,
Vulkan / RADV): correctness is checked against the CPU reference implementation
(the `gpu_seam_matches_cpu_*` tests generate token-for-token on both and
compare) and throughput is measured against the system `llama.cpp` build with
`infr compare`.

**Throughput vs llama.cpp** — ratios are `infr / llama.cpp` (**>1.0 = infr is
faster**); `Q4_K_M`, r=3, 2026-07-09 snapshot. Hardware: **AMD Radeon RX 7900
XTX** (RDNA3, 24 GB, Vulkan / RADV, Mesa). `pp512` = 512-token prefill
throughput, `tg128` = 128-token decode throughput, `tg64@d4096` = decode at 4096
KV depth, `pp4@d4096` = short-turn prefill at 4096 KV depth (the multi-turn
serve shape).

| Model                  | pp512     | tg128     | tg64@d4096 | pp4@d4096 |
| ---------------------- | --------- | --------- | ---------- | --------- |
| Qwen3-0.6B             | **1.18×** | **1.22×** | **1.28×**  | **2.07×** |
| Gemma-3-1B             | **1.04×** | **1.17×** | **1.09×**  | **1.04×** |
| Qwen3-1.7B             | **1.12×** | **1.10×** | **1.13×**  | **1.60×** |
| Qwen3.5-4B             | **1.02×** | 0.91×     | 0.93×      | **1.36×** |
| Qwen3-8B               | **1.28×** | 0.95×     | 0.96×      | **1.23×** |
| Qwen3.5-9B             | **1.11×** | 0.95×     | 0.96×      | **1.35×** |
| Gemma-3-12B            | **1.24×** | **1.02×** | **1.03×**  | **1.47×** |
| Qwen3-14B              | **1.11×** | 0.90×     | 0.86×      | **1.04×** |
| Gemma-4-E2B            | **1.14×** | **1.09×** | **1.04×**  | **1.08×** |
| Qwen3.6-27B            | **1.08×** | 0.91×     | 0.91×      | **1.13×** |
| Qwen3-30B-A3B (MoE)    | 0.96×     | 0.95×     | 0.94×      | **1.17×** |
| Qwen3.6-35B-A3B (MoE)² | 0.93×     | 0.94×     | 0.95×      | **1.45×** |

¹ gemma-4-E2B `pp4@d4096` was the only metric below 1.0×. The original in-sweep
reading was 0.46× (261 vs 572 t/s) — a thermal artifact from the multi-model
sweep (see [PERF.md](docs/PERF.md#archiving-sweeps)). Three dispatch fusions
landed to close the gap:
1. `CopyStrided` eliminated via per-row source stride on `GatedAct` (35 dispatches)
2. inp_gate `Op::Linear` + `GatedAct` fused into `e2b_gate` kernel (35 dispatches)
3. proj `Op::RmsNorm` + `Op::Add` fused into `rmsnorm_add` kernel (35 dispatches)
Total: 70 dispatches eliminated. E2B now beats llama.cpp on every metric.

² Qwen3.6-35B-A3B is the UD (ultra-dense) variant — only the standard UD Q4_K_M
quant was available.

infr **wins prefill on every dense model** (1.02–1.28×); the two MoEs prefill at
0.93–0.96× — correct full-expert routing (batch spreads across 128 or 256
experts into smaller per-expert GEMMs) costs some batch efficiency vs
llama.cpp's own expert dispatch. Multi-turn ingest **dominates on every model**
(1.04–2.07× on all 12 models). Decode is at-or-above parity
on models up to ~4B, and slightly behind on larger models — dense 8B/9B/14B/27B
at 0.90–0.96×, MoE 30B/35B at 0.94–0.95×, all bounded by the memory-bandwidth
wall (decode GEMVs run at 77–88% of DRAM peak, matching llama.cpp's own
efficiency). The **Qwen3.5-4B MTP path** trails at 0.67× (185 vs 277 t/s) —
drafted-token throughput not yet matching the batched-speculative path in
llama.cpp. **DiffusionGemma** (`dg-step`, the in-step-parallel metric) is at
parity-or-better vs the reference fork.

**Also validated for correctness** (GPU seam vs CPU reference), beyond the perf
table: Qwen2-0.5B, Llama-3.2-1B, Gemma-4-12B (dense), and Qwen3-0.6B across
quant formats **Q4_K_M / Q5_K_M / Q6_K / Q4_0 / Q2_K / IQ4_XS / Q8_0 / BF16**
(each decoded on-device via hand-written SPIR-V, byte-identical to the CPU
dequant).

> Numbers are a snapshot and move with each perf slice; regenerate on your own
> hardware with `infr compare --sweep <model...>`. Results on other GPUs
> (NVIDIA, Intel Arc) and Apple Metal are wanted — please open an issue with
> your `infr bench` / `infr compare` output.

## Scope

- **Format:** GGUF
- **Models:** Llama, Qwen2/2.5, Qwen3 (dense + MoE), Gemma 3, Gemma 4 (dense +
  E2B), Qwen3.5/3.6 (dense + MoE) — all on GPU **and** the CPU reference;
  DiffusionGemma (block text-diffusion, CPU + GPU)
- **GPU:** AMD / NVIDIA / Intel via Vulkan (cooperative-matrix matmul); Apple
  via a native **Metal backend** (`INFR_METAL=1`) covering every op the CPU
  reference does — dense, MoE (`qwen3moe`) and Qwen3.5 (`qwen35`). Dense is
  optimized (simdgroup-matrix GEMM + flash attention, raw-block quant decode;
  within ~1.3-1.5× of llama.cpp Metal on M3 Pro — architecture and numbers in
  [`docs/METAL.md`](docs/METAL.md))
- **Store:** the shared **HuggingFace Hub cache** — located via `$HF_HUB_CACHE`,
  else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`, in HF's own
  `models--<org>--<repo>/{blobs,snapshots,refs}` layout. A model pulled by
  `infr`, `llama.cpp`, or `huggingface_hub` is shared — downloaded once.
  `infr pull` fetches from `huggingface.co` over resumable HTTP Range with a
  progress bar; gated repos authenticate with `HF_TOKEN`.
- **API:** OpenAI-compatible HTTP (streaming) — works with opencode / Claude
  Code CLI

## Architecture

```
server   axum + SSE  ->  OpenAI /v1
chat     ChatModel        (autoregressive dense/MoE/qwen35; DiffusionGemma's block-diffusion loop)
runtime  SeamModel        tensors, KV cache, command/descriptor management (the unified runner)
loader   WeightSource     (Gguf; safetensors later)
compute  Backend          (Vulkan via ash + SPIR-V; reference Metal via MSL; CUDA later)
```

## License

[MIT](LICENSE)
