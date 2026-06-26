# infr — Plan

Pure-Rust LLM inference engine. Vulkan-first, designed to run on any mainstream
GPU. The only non-Rust surface is the GPU driver (called through thin Rust FFI)
and the compute shaders (SPIR-V).

---

## Vision

A from-the-metal inference server where **the server and model code never know
which GPU API is running underneath**. We ship a Vulkan backend first (covers
AMD/NVIDIA/Intel, and Apple via MoltenVK), then add native CUDA / ROCm / Metal
backends later **without touching any layer above the backend**.

The whole architecture is organized around four pluggable seams so that "add a
GPU", "add a model", "add a format", or "add a decode style" each means
_implementing one trait_, never refactoring the stack.

---

## MVP scope

Deliberately narrow, to ship something real on the author's hardware first.

| Dimension    | MVP                                           | Later                                 |
| ------------ | --------------------------------------------- | ------------------------------------- |
| Format       | **GGUF**                                      | safetensors, MLX                      |
| Model source | **HuggingFace + Ollama pull**, or local path  | other registries, mirrors             |
| Model        | **DiffusionGemma** (diffusion decode)         | Llama / Qwen / Gemma (autoregressive) |
| GPU backend  | **Vulkan** (`ash` + SPIR-V) on **AMD (RADV)** | CUDA, ROCm, native Metal              |
| Decode       | **diffusion** (block denoise)                 | autoregressive (greedy / sampling)    |
| API          | **OpenAI-compatible HTTP** (streaming)        | embeddings, batching, multi-model     |

**MVP done = `curl`/opencode/Claude Code CLI can hold a streaming chat with
DiffusionGemma served from `infr` over Vulkan on the 7900 XTX.**

### Product surface (the `infr` CLI)

Three commands, all sharing the same engine + backend underneath (ollama-like
UX):

```bash
infr pull  <model-ref>        # download + cache a GGUF model (HF or Ollama)
infr run   <model-ref> [msg]  # interactive terminal chat (auto-pulls if missing)
infr serve <model-ref>        # start the OpenAI-compatible HTTP API
```

- **`infr pull`** — resolve a model reference and fetch the GGUF into the local
  cache. References:
  - `hf:org/repo[:file.gguf]` — HuggingFace (resolve repo, pick/verify the
    GGUF).
  - `ollama:name[:tag]` — Ollama registry pull (or reuse an already-pulled local
    model).
  - a plain filesystem path to a `.gguf`.
- **`infr run`** — load the model on the backend and drop into a simple REPL
  chat in the terminal (streams tokens, shows reasoning vs answer). Auto-pulls
  if not cached.
- **`infr serve`** — same load path, but exposes `/v1/chat/completions` +
  `/v1/models` (streaming) for opencode / Claude Code CLI. Auto-pulls if not
  cached.

`run` and `serve` are two front-ends over the **same** `infr-engine` (load →
backend → decode); `pull` is just the acquisition half on its own. Nothing in
`run`/`serve`/the engine knows which GPU backend is active.

### End-to-end flow

```
model-ref ──▶ infr-hub (resolve + download + cache)
          ──▶ infr-gguf (mmap + parse + upload tensors to backend buffers)
          ──▶ infr-model (build DiffusionGemma graph)  ──▶ Backend.compile()
          ──▶ infr-decode (diffusion loop)             ──▶ Backend.execute() per step
          ──▶ run: terminal REPL    |    serve: OpenAI HTTP/SSE
```

### Non-goals (MVP)

- No autoregressive models yet (diffusion only — but see note below).
- No training, no fine-tuning, no quantization _creation_ (we only _consume_
  GGUF quants).
- No multi-GPU, no tensor/pipeline parallelism.
- No non-Vulkan backends yet (the abstraction exists; impls come later).
- Perf is "correct and usable," not "beat llama.cpp." Tuning is a later phase.

> Note: "diffusion only" does **not** skip the transformer. DiffusionGemma's
> forward pass is a full transformer (30 layers, sliding-window attention on 25
> / full attention on 5, MoE with ~3.8B active params, RoPE, RMSNorm). The
> "diffusion" is the _decode strategy_ layered on top of that forward. So the
> MVP implements a complete transformer forward + a diffusion decode loop.

---

## Architecture

Bottom-up. Each named trait is the seam where future variants plug in.

```
┌────────────────────────────────────────────────────────────────────┐
│ server      axum + SSE  ->  OpenAI /v1/chat/completions, /v1/models  │  knows NOTHING about the GPU
├────────────────────────────────────────────────────────────────────┤
│ engine      session orchestration, chat templating, tool-call bridge │
├────────────────────────────────────────────────────────────────────┤
│ decode      trait DecodeStrategy   -> DiffusionDenoise  (AR later)   │
├────────────────────────────────────────────────────────────────────┤
│ model       trait Model            -> DiffusionGemma     (Llama …)   │  builds an abstract compute Graph
├────────────────────────────────────────────────────────────────────┤
│ runtime     Tensor handles, KV cache, Graph builder                  │
├────────────────────────────────────────────────────────────────────┤
│ loader      trait WeightSource     -> Gguf  (safetensors later)      │
├────────────────────────────────────────────────────────────────────┤
│ compute     trait Backend          -> Vulkan (CUDA / ROCm / Metal …) │  the ONLY GPU-aware layer
├────────────────────────────────────────────────────────────────────┤
│ shaders     SPIR-V (reused / ported from ggml-vulkan)                │  not Rust
└────────────────────────────────────────────────────────────────────┘
```

Dependency rule: **everything above `compute` is generic over the backend.** The
`server` depends on `engine`, which holds a backend as `Box<dyn Backend>` (or a
generic `B: Backend`) and otherwise treats it as opaque.

---

## The backend abstraction (the core design goal)

The hard requirement: _the server does not know what's running behind it, and a
new driver (CUDA, ROCm, native Metal) can be dropped in without changing
anything above `compute`._

### Why a semantic compute-graph, not raw dispatch

A naive abstraction would expose "allocate buffer + dispatch shader." That leaks
Vulkan: SPIR-V doesn't port to CUDA (PTX) or ROCm (GCN/HIP), and cuBLAS/rocBLAS
aren't shader dispatches at all. So the seam is drawn **higher**, at the level
of _semantic tensor ops_.

The model layer builds an **abstract compute graph** (a DAG of ops over tensor
handles). The backend **compiles and executes** that graph however it wants:

- **Vulkan**: ops → SPIR-V pipelines + descriptor sets, batched into command
  buffers.
- **CUDA** (later): ops → cuBLAS / custom kernels / CUDA graphs.
- **ROCm** (later): ops → rocBLAS / HIP kernels.
- **Metal** (later): ops → MSL compute pipelines.

This mirrors how ggml separates graph construction from backend execution, and
it's what makes the backends truly interchangeable.

### Trait sketch (illustrative, not final)

```rust
/// Opaque device memory handle owned by a backend.
pub trait Buffer: Send + Sync {}

/// What the layers above can rely on, regardless of GPU API.
pub trait Backend: Send + Sync {
    type Buffer: Buffer;
    type Plan;                                  // a compiled, ready-to-run graph

    fn capabilities(&self) -> Capabilities;     // f16, coop-matrix, max buffer, etc.

    // memory
    fn alloc(&self, bytes: usize, usage: Usage) -> Result<Self::Buffer>;
    fn upload(&self, dst: &Self::Buffer, src: &[u8]) -> Result<()>;
    fn download(&self, src: &Self::Buffer, dst: &mut [u8]) -> Result<()>;

    // execution: compile an abstract graph once, run it many times (per token/step)
    fn compile(&self, graph: &Graph) -> Result<Self::Plan>;
    fn execute(&self, plan: &Self::Plan, io: &mut Bindings) -> Result<()>;
    fn sync(&self) -> Result<()>;
}

/// Backend-agnostic op set the model graph is built from.
pub enum Op {
    MatMul { a: TensorId, b: TensorId, /* quant-aware */ },
    Dequant { src: TensorId, dtype: QuantType },
    RmsNorm { x: TensorId, w: TensorId, eps: f32 },
    Rope { x: TensorId, /* pos, theta */ },
    Attention { q: TensorId, k: TensorId, v: TensorId, mask: AttnMask /* full | swa(window) */ },
    MoeFfn { x: TensorId, router: TensorId, experts: ExpertSet, active_k: u32 },
    Softmax { x: TensorId },
    Add { a: TensorId, b: TensorId },
    Mul { a: TensorId, b: TensorId },
    // … grown as the model needs
}
```

Key properties:

- **Capabilities are queried, not assumed.** A backend advertises coop-matrix /
  f16 / max buffer size; the graph compiler picks fast vs fallback kernels
  accordingly.
- **Quantization is a backend concern.** The graph says "matmul with a Q4_K
  weight"; the backend owns how it dequantizes / fuses. Different backends can
  do it differently.
- **Compile-once / execute-many.** A transformer layer's graph is compiled once;
  each diffusion step / token reuses the plan. Vulkan loves this (pipelines +
  command buffers built up front); CUDA graphs map to it too.
- **No backend type escapes upward.** `engine`/`server` only ever see `Backend`,
  `Tensor`, `Graph` — never `VkDevice`, never SPIR-V.

### Adding a backend later = one impl

To add CUDA/ROCm/Metal: implement `Backend` + the `Op` set for that API,
register it in the backend factory, done. Selection is runtime
(`--backend vulkan|cuda|rocm|auto`), and the factory probes availability.
Nothing above `compute` changes.

---

## Crate layout (Cargo workspace)

```
infr/
├── crates/
│   ├── infr-core       # Tensor, dtypes/quant, Graph, Op, Backend trait, errors
│   ├── infr-vulkan     # Backend impl: ash + gpu-allocator + SPIR-V dispatch
│   ├── infr-hub        # model resolve + download + cache (HuggingFace, Ollama)
│   ├── infr-gguf       # WeightSource impl: GGUF parse + metadata + tensor mapping
│   ├── infr-model      # Model trait + DiffusionGemma graph builder
│   ├── infr-decode     # DecodeStrategy + DiffusionDenoise (entropy-bound)
│   ├── infr-engine     # load pipeline, session orchestration, chat template, tool-call bridge
│   └── infr-server     # axum OpenAI-compatible HTTP + SSE
├── shaders/            # GLSL/comp sources + build step -> SPIR-V (reuse ggml-vulkan)
├── src/
│   └── main.rs         # the `infr` CLI: pull / run / serve  (clap subcommands)
├── bin/
│   └── smoke           # dev: f16 coop-matrix matmul on the GPU vs CPU reference
├── PLAN.md
├── README.md
└── LICENSE
```

---

## Component notes

### compute / Vulkan (`infr-vulkan`)

- `ash` for raw Vulkan; `gpu-allocator` for VRAM suballocation.
- Reuse **ggml's Vulkan `.comp` shaders → SPIR-V** for the tuned kernels (quant
  matmul, dequant, attention) instead of re-deriving quant math; compile at
  build time.
- Use `VK_KHR_cooperative_matrix` (verified on RADV/gfx1100) for fast f16
  matmul; fall back to a scalar/subgroup path where unavailable
  (capability-gated).
- Async compute queue + command-buffer batching for the
  compile-once/execute-many model.

### fetch / model acquisition (`infr-hub`)

- Resolve a model reference to a concrete local GGUF path, downloading + caching
  if needed.
- **The Ollama store is infr's primary cache — same dir + same on-disk format.**
  So a user who already has Ollama models keeps using them with zero
  re-download, and anything `infr pull`s is also visible to `ollama` (and vice
  versa). One shared model library, not two.
  - Location: `$OLLAMA_MODELS` if set, else `~/.ollama/models` (override with
    `INFR_MODELS`). Layout: `manifests/<registry>/<ns>/<name>/<tag>` (OCI-style
    JSON) + `blobs/sha256-<digest>`.
  - Resolve a tag: read its manifest, find the layer with mediaType
    `application/vnd.ollama.image.model` → that blob **is** the GGUF; `mmap` it
    in place (no copy). Optionally read the `template` / `params` / `system`
    layers too (handy chat-template/defaults source).
  - `infr pull ollama:name[:tag]` fetches via the Ollama registry pull protocol
    (`registry.ollama.ai`) and writes blobs + manifest in this exact format, so
    the result is a normal Ollama model.
- **HuggingFace** (`hf:org/repo[:file]`): hub HTTP API — resolve repo, list
  siblings, pick/verify the `.gguf`, download via `resolve/main/...` with
  resume, honor `HF_TOKEN` for gated repos. Imported into the same store as a
  synthesized manifest + the GGUF blob (so HF and Ollama pulls live side by
  side).
- A plain filesystem path to a `.gguf` is used as-is (no copy into the store).
- Streaming download with progress; checksum/digest verification. `infr pull` is
  just this step; `run`/`serve` call it lazily.

### loader / GGUF (`infr-gguf`)

- Parse GGUF: header, metadata KVs (arch, hyperparams, tokenizer, chat
  template), tensor directory (name, dtype, offset, shape).
- Memory-map the file; upload tensors to backend buffers (quantized blocks stay
  quantized).
- Expose embedded tokenizer + jinja chat template to the engine.

### model / DiffusionGemma (`infr-model`)

- Build the transformer graph from GGUF weights: token embed → 30 × (RMSNorm,
  attention, RMSNorm, MoE-FFN) → final norm → output projection.
- Attention: GQA; **sliding-window** for 25 layers, **full** for 5 (per the
  head_count_kv pattern); RoPE; per-layer KV.
- MoE: router + top-k expert gather (~3.8B active of 26B).
- Reference: the patched llama.cpp `diffusion-gemma` graph.

### decode / diffusion (`infr-decode`)

- Block (canvas) diffusion: denoise a fixed-size canvas over N steps;
  entropy-bound early stop; self-conditioning; commit per block.
- Channels: split `<|channel>thought` / `<channel|>` into reasoning vs final
  answer.

### engine + server (`infr-engine`, `infr-server`)

- OpenAI `/v1/chat/completions` (streaming + non-streaming) and `/v1/models`.
- Stream `reasoning_content` (thought) separately from `content` (answer).
- Parse the model's native `<|tool_call>call:NAME{…}<tool_call|>` into OpenAI
  `tool_calls`; pass tool definitions/results through so opencode/Claude Code
  agentic loops work.
- `engine` is generic over the backend; `server` only talks to `engine`.

---

## Target hardware (verified)

AMD RX 7900 XTX (RADV NAVI31, gfx1100), Mesa 26.1, Vulkan 1.4:

- `VK_KHR_cooperative_matrix` rev 2, `cooperativeMatrix = true` → fast matmul
  path available.
- `shaderFloat16 = true`, `VK_KHR_16bit_storage` → native f16.
- subgroup extended types / rotate → reductions, softmax, attention.
- `maxStorageBufferRange = 4 GB` → large weight tensors fit per binding.
- `maxComputeSharedMemorySize = 64 KB`.

---

## Roadmap / milestones

1. **Compute smoke test** — `bin/smoke`: enumerate Vulkan device, alloc via
   gpu-allocator, dispatch a SPIR-V f16 **cooperative-matrix matmul**, verify vs
   CPU. _De-risks everything._
2. **Core trait + tensor/graph** — `infr-core`: Tensor, dtypes, Graph/Op,
   Backend trait.
3. **`infr pull`** — `infr-hub`: HF + Ollama resolve/download/cache; CLI command
   works standalone (independent of the GPU work, can land early).
4. **Vulkan backend MVP** — matmul, dequant (Q4_K/Q8_0), rmsnorm, rope, softmax,
   add/mul.
5. **GGUF loader** — parse + upload DiffusionGemma weights; expose tokenizer +
   template.
6. **Forward pass** — one transformer layer → full stack → correct logits vs
   reference.
7. **Attention + MoE** — GQA + SWA + full-attn layers; MoE routing/gather.
8. **Diffusion decode** — canvas denoise loop, entropy-bound, channels.
9. **`infr run`** — interactive terminal REPL over the engine (streams,
   reasoning split).
10. **`infr serve`** — axum OpenAI server: streaming chat, reasoning split,
    tool-call bridge; opencode / Claude Code CLI work end-to-end. **(MVP
    complete)**
11. **Perf pass** — coop-matrix tuning, command-buffer batching, KV cache
    layout.
12. **Second backend** — prove the abstraction by adding CUDA or ROCm behind
    `Backend`.

---

## Risks / open questions

- **Reusing ggml SPIR-V**: licensing (MIT — compatible) and binding layout/ABI
  must be matched, or we port the GLSL to our own descriptor conventions.
- **Quant matmul correctness** is the classic footgun — validate each quant type
  against a CPU dequant reference early.
- **Graph abstraction vs perf**: the compile-once/execute-many design must not
  force per-op sync; batch aggressively in the Vulkan backend.
- **Diffusion decode** is novel in Rust — no reference to copy; port carefully
  from the llama.cpp implementation (see References below).

---

## Reference implementations (read these first)

All paths are on the author's machine; treat them as the source of truth.

- **`~/Projects/llama.cpp`** — checked out on branch **`diffusiongemma`** (PR
  `ggml-org/llama.cpp#24423`), already built with HIP. Key files:
  - `src/llama-model.cpp` + `src/llama-arch.cpp` — the `diffusion-gemma`
    architecture graph (embeddings → 30 layers w/ SWA-vs-full attention
    selection, MoE FFN, RoPE, RMSNorm → output). **The canonical forward to
    port.**
  - `examples/diffusion/diffusion.{cpp,h}` — diffusion algorithms incl.
    `diffusion_generate_entropy_bound` and the algorithm enum. **Port for
    `infr-decode`.**
  - `examples/diffusion-gemma-server/diffusion-gemma-visual-server.cpp` — the
    patched persistent server (tools passthrough, FA-aware sizing, fixed ubatch,
    `CMOE`/`NCMOE` expert offload). Reference for diffusion params, context
    sizing, channel/tool handling.
  - `conversion/diffusion_gemma.py` — GGUF conversion: **authoritative tensor
    names + metadata keys.**
  - `ggml/src/ggml-vulkan/vulkan-shaders/*.comp` + `ggml/src/ggml-vulkan/` — the
    Vulkan compute shaders to reuse (`mul_mm`, per-quant dequant, etc.) and how
    descriptor sets / specialization constants are wired.
- **`~/Projects/scratch/dgemma-openai-server.py`** — the working OpenAI shim we
  built. Port its logic into `infr-engine`/`infr-server`: channel split
  (`reasoning_content` vs `content`), `<|tool_call>` → OpenAI `tool_calls`
  parsing, tools passthrough, SSE streaming.
- **Test model:**
  `~/Projects/models/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`
  (16 GB, Q4_K_M). Also `unsloth/diffusiongemma-26B-A4B-it-GGUF` on HF, or via
  the Ollama store.
- **Validation oracle:** `~/Projects/llama.cpp/build/bin/llama-diffusion-cli`
  produces reference outputs/logits to diff against.

---

## DiffusionGemma spec (verified from the test GGUF)

Read these from GGUF metadata at runtime — do not hardcode — but here they are
so the implementer knows the shape up front.

- **arch** `diffusion-gemma`; **vocab** 262144; **hidden** (`embedding_length`)
  2816; **layers** (`block_count`) 30; **train ctx** 262144.
- **Attention:** `head_count` 16; `head_count_kv` per layer = `[8,8,8,8,8,2]`
  repeating → layers **5, 11, 17, 23, 29 are full-attention** (kv_heads 2), the
  other 25 are **sliding-window** (kv_heads 8); `key_length`=`value_length`=512
  (full), `*_swa`=256. This SWA split is why 256K KV is only ~a few GB.
- **MoE:** ~3.8B active of 26B; expert tensors `blk.N.ffn_(gate|up|down)_exps`,
  router/gate, top-k routing.
- **Diffusion:** `canvas_length` 256 tokens/block; `mask_token` id **4**;
  block-autoregressive (commit blocks sequentially). Entropy-bound decoder
  defaults observed: algorithm 4 (CONFIDENCE_BASED), schedule 0, `max_steps≈48`,
  `temperature` 0.8, `eps` 0.001, `t∈[0.4,0.8]`, `entropy_bound` 0.1,
  `stability` 1, `confidence` 0.005, self-conditioning on.
- **Special tokens / wire format** (the GGUF embeds the jinja `chat_template`;
  drive it with `minijinja` rather than reimplementing):
  - turns: `<|turn>role\n … <turn|>`, leading `bos`.
  - channels: `<|channel>thought\n … <channel|> … final`. Reasoning = before
    `<channel|>`, answer = after.
  - tool decl:
    `<|tool>declaration:NAME{description:<|"|>…<|"|>,parameters:{…}}<tool|>`
  - tool call: `<|tool_call>call:NAME{key:value,…}<tool_call|>` (strings wrapped
    `<|"|>…<|"|>`).
  - tool response: `<|tool_response>response:NAME{…}<tool_response|>`
- **Quants available:** Q4_K_M (target), Q5_K_M, Q6_K, Q8_0, BF16.

---

## Dependencies & toolchain

- **Rust** 1.96+ (edition 2021; 2024 ok). Workspace with the crates listed
  above.
- **Crates (proposed):** `ash` + `gpu-allocator` (Vulkan); `half`, `bytemuck`
  (dtypes); `memmap2` (GGUF mmap); `serde`/`serde_json`; `reqwest` (rustls) +
  `sha2` + `indicatif` (hub download); `tokio` + `axum` + `tower-http` (server);
  `clap` (CLI); `minijinja` (chat templates); `tokenizers` (or implement the
  GGUF-embedded tokenizer); `thiserror`/`anyhow`; `tracing`.
- **Shaders:** GLSL `.comp` → SPIR-V via the `shaderc` crate in `build.rs`,
  **or** reuse precompiled `.spv` from the ggml build. Device features to
  request: `VK_KHR_cooperative_matrix`, `shaderFloat16`, `VK_KHR_16bit_storage`,
  `VK_KHR_shader_subgroup_extended_types`; one compute queue.

---

## Validation strategy

- **smoke:** f16 cooperative-matrix matmul vs a CPU naive matmul; assert
  relative error < 1e-2.
- **quant:** dequantize a known tensor (Rust/shader) and diff vs a CPU
  reference; validate each quant type (Q4_K_M, Q8_0) before trusting matmuls.
- **per-layer:** fixed prompt → compare hidden states / final logits (top-k
  token probabilities) against `llama-diffusion-cli` on the same model.
- **end-to-end:** same prompt + fixed seed → compare generated text/structure to
  the llama.cpp oracle.

---

## Milestone acceptance criteria

| #   | Milestone    | Done when                                                            |
| --- | ------------ | -------------------------------------------------------------------- |
| 1   | smoke        | f16 coop-matrix matmul on the GPU matches CPU within tolerance       |
| 2   | core         | `Tensor`/`Graph`/`Op`/`Backend` compile; a 2-op graph runs on Vulkan |
| 3   | `infr pull`  | `ollama:` reuses `~/.ollama` with no re-download; `hf:` downloads    |
| 4   | vk backend   | matmul/dequant/rmsnorm/rope/softmax each validated vs CPU            |
| 5   | gguf load    | DiffusionGemma weights upload; tokenizer + template exposed          |
| 6   | forward      | final logits match llama.cpp top-k on a fixed prompt                 |
| 7   | attn + moe   | full multi-layer forward matches reference (SWA + full + MoE)        |
| 8   | diffusion    | greedy/fixed-seed generation matches llama.cpp text                  |
| 9   | `infr run`   | interactive terminal chat streams a coherent answer                  |
| 10  | `infr serve` | opencode / Claude Code CLI complete an agentic tool turn             |
