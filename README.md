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
**Qwen3.5 / Qwen3.6** (`qwen35` / Qwen3-Next — hybrid gated-DeltaNet +
attention) run via a CPU reference (`docs/QWEN35.md`); a Vulkan/hybrid path is
planned. DiffusionGemma (the original target) is future work.

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

| Family               | Arch (GGUF) | Notes                                          |
| -------------------- | ----------- | ---------------------------------------------- |
| Llama / Qwen2        | `llama`     | dense transformer                              |
| Qwen3                | `qwen3`     | dense, QK-norm                                 |
| Qwen3 MoE            | `qwen3moe`  | softmax router, top-_k_ experts (CPU offload)  |
| Gemma 3              | `gemma3`    | SWA + QK-norm + GeGLU, dual-RoPE               |
| Gemma 4 (dense)      | `gemma4`    | per-layer head dims, proportional RoPE, V-norm |
| Gemma 4 **E2B**      | `gemma4`    | + per-layer input embeddings / FFN, KV sharing |
| Qwen3.5 / Qwen3-Next | `qwen3next` | hybrid gated-DeltaNet — **CPU reference** only |

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

# Serve any of them over an OpenAI-compatible API
infr serve unsloth/Qwen3-14B-GGUF:Q4_K_M
```

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

**Profile** per-op GPU time (timestamp queries) with `INFR_PROF2=1`. It prints
one block **per submit**, each tagged by op label (prefill: `expert_gateup`,
`expert_down`, `matmul_proj`, `attn_flash`, `quant_q8`; decode: `lm_head`,
`mmq_expert`, `expert_ffn`, `attention_kv`, `vocab`, …). `warmup` runs
unprofiled, so the blocks are the timed reps only — sum a label across all
blocks for its total:

```bash
INFR_PROF2=1 infr bench "$M" -p 2048 -n 0 -r 1 2>&1 \
  | grep '^\[prof2\]' \
  | awk '!/per-op/{for(i=1;i<=NF;i++)if($i~/us$/){l=$(i-1);v=$i;sub(/us/,"",v);t[l]+=v}}
         END{for(l in t)printf "%-16s %10.0f us\n",l,t[l]}' | sort -k2 -rn
```

**Compare to llama.cpp** — `infr compare` shells out to `infr bench` and the
system `llama-bench` with matching flags on coding-agent-shaped workloads
(prefill, decode-at-depth, whole turns). `--ctx` is comma-delimited:

```bash
infr compare "$M" --ctx 8000,16000 --gen 256 --turn 2048,256 --reps 2
```

Useful env: `INFR_TEMP` / `INFR_TOP_K` / `INFR_TOP_P` (sampling; `TEMP=0` →
greedy), `INFR_MAX_NEW`, `INFR_MAX_CTX`, `INFR_NCMOE` (MoE expert CPU offload),
`INFR_NO_FLASH`.

## Scope

- **Format:** GGUF
- **Models:** Llama / Qwen3 / Gemma 3 / Gemma 4 (dense + E2B) (GPU); Qwen3.5/3.6
  (CPU ref); DiffusionGemma (planned)
- **GPU:** AMD / NVIDIA / Intel via Vulkan (cooperative-matrix matmul); Apple via a
  correctness-first **reference Metal backend** (`INFR_METAL=1`) covering every op the CPU
  reference does — dense, MoE (`qwen3moe`) and Qwen3-Next (`qwen35`)
- **Store:** own cache at `$XDG_CACHE_HOME/infr/models` (standalone HF + Ollama
  HTTP pulls)
- **API:** OpenAI-compatible HTTP (streaming) — works with opencode / Claude
  Code CLI

## Architecture

```
server   axum + SSE  ->  OpenAI /v1
decode   DecodeStrategy   (AutoRegressive; DiffusionDenoise later)
model    Model            (Llama/Qwen3; Qwen3-Next CPU ref; DiffusionGemma later)
runtime  tensors, KV cache, command/descriptor management
loader   WeightSource     (Gguf; safetensors later)
compute  Compute          (Vulkan via ash + SPIR-V; reference Metal via MSL; CUDA later)
```

## License

[MIT](LICENSE)
