# infr

Pure-Rust LLM inference engine. Vulkan-first, built to run on any mainstream GPU.

> Early WIP. The only non-Rust parts are the GPU driver calls (Vulkan via `ash`) and
> the compute shaders (SPIR-V).

## Goal

A from-the-metal inference server that works across AMD / NVIDIA / Intel (Vulkan) and
Apple (MoltenVK), with native backends addable later behind a `Compute` trait.

## MVP scope

- **Format:** GGUF
- **Model:** DiffusionGemma (diffusion decode)
- **GPU:** AMD via Vulkan (cooperative-matrix matmul)
- **API:** OpenAI-compatible HTTP (streaming) — works with opencode / Claude Code CLI

## Architecture

```
server   axum + SSE  ->  OpenAI /v1
decode   DecodeStrategy   (DiffusionDenoise; AutoRegressive later)
model    Model            (DiffusionGemma; Llama/Qwen later)
runtime  tensors, KV cache, command/descriptor management
loader   WeightSource     (Gguf; safetensors later)
compute  Compute          (Vulkan via ash + SPIR-V; Metal/CUDA later)
```

## License

[MIT](LICENSE)
