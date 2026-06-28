# infr

Pure-Rust LLM inference engine. Vulkan-first, built to run on any mainstream
GPU.

> Early WIP. The only non-Rust parts are the GPU driver calls (Vulkan via `ash`)
> and the compute shaders (SPIR-V).

## Goal

A from-the-metal inference server that works across AMD / NVIDIA / Intel
(Vulkan) and Apple (MoltenVK), with native backends addable later behind a
`Compute` trait.

## Status

Runs **Llama / Qwen2 / Qwen3** (dense) on the Vulkan backend, competitive with
llama.cpp at long context (`infr compare`). **Qwen3.5 / Qwen3.6** (`qwen35` /
Qwen3-Next — hybrid gated-DeltaNet + attention) run via a CPU reference
(`docs/QWEN35.md`); a Vulkan/hybrid path is planned. DiffusionGemma (the
original target) is future work.

```bash
infr pull   <model-ref>        # hf:org/repo[:file] | ollama:name[:tag] | path
infr run    <model-ref> [msg]  # terminal chat (auto-pulls)
infr serve  <model-ref>        # OpenAI-compatible HTTP API
infr bench / infr compare      # tok/s benchmarks vs llama.cpp
```

## Benchmarking & profiling

`infr bench` matches `llama-bench`'s `-p`/`-n`/`-d`/`-r` flags, so the two are
directly comparable. Pipelines are compiled and GPU state is first-touched at
model load (`Llama::warmup`), so timing measures compute, not one-time setup.
**Run benchmarks one at a time** — concurrent GPU work skews results.

```bash
M='hf:unsloth/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf'   # MoE perf target

# Prefill (pp = n_prompt/time) and decode (tg = n_gen/time):
infr bench "$M" -p 2048 -n 0 -r 3       # prefill 2048 tokens
infr bench "$M" -p 8000 -n 0 -r 2       # prefill at depth
infr bench "$M" -p 0 -n 64 -r 3         # decode 64 tokens
infr bench "$M" -p 0 -n 64 -d 2048      # decode at context depth 2048 (-d warms, untimed)
```

**Profile** per-op GPU time (timestamp queries) with `INFR_PROF2=1`; it prints,
per submit, time aggregated by op label (`matmul_proj`, `attn_flash`,
`mmq_expert`, `quant_q8`, …). Read the ratios — the aggregate includes warmup:

```bash
INFR_PROF2=1 infr bench "$M" -p 2048 -n 0 -r 1 2>&1 | grep prof2
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
- **Models:** Llama / Qwen3 dense (GPU); Qwen3.5/3.6 (CPU ref); DiffusionGemma
  (planned)
- **GPU:** AMD / NVIDIA / Intel via Vulkan (cooperative-matrix matmul)
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
compute  Compute          (Vulkan via ash + SPIR-V; Metal/CUDA later)
```

## License

[MIT](LICENSE)
