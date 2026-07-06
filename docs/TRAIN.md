# infr-train — LLM training support (plan)

Plan as of 2026-07-06, from a code assessment of the seam/backends plus a survey
of the Rust training landscape. Nothing here is built yet; this doc is the
design record and the phase plan.

**Verdict up front:** add training as a new sibling crate in this workspace
(`infr-train`), with hand-written llm.c-style backward passes over the existing
`infr-core` seam — not a general tape autodiff, and not an external framework.
Entry point is **LoRA finetuning on the CPU backend**, then QLoRA on Vulkan.

---

## Why this shape

### What the code assessment found

infr is purely forward and decode-oriented. There is zero gradient / backward /
optimizer code anywhere, and every matmul computes only `Y = X·Wᵀ` — the
transposed variants backward needs (`dX = dY·W`, `dW = dYᵀ·X`) do not exist on
any backend.

The seam is nonetheless an unusually good training substrate:

- The IR (`infr-core::graph`) is an **ordered list of composite semantic ops**
  over typed tensor handles. A backward pass is a reverse walk emitting grad ops
  — exactly the structure llm.c hand-codes, but ours is already
  backend-agnostic.
- **In-place aliasing is legal and pervasive** (in-place RoPE, KV writes,
  scratch reuse; `Graph::in_place_inputs()` exists because of it). This rules
  out naive reverse-mode autodiff over the existing graphs — backward must be
  hand-emitted per op, with the training-graph builder avoiding aliasing where a
  saved activation is needed.
- The CPU backend is the **oracle**: it runs every supported arch and quant
  zero-copy from the GGUF mmap. It is the natural home for the reference
  backward implementations that GPU backends get validated against — the same
  pattern the inference side already uses.
- Reusable unchanged: `infr-gguf` (read side), `infr-hub`, the tokenizer,
  `Backend`/`Bindings`/`DType`, and every forward kernel as the twin of its
  future backward.

Why a sibling crate and not the alternatives:

- **Not in-tree in `infr-llama`:** the forward path is optimized for inference
  (decode-only n=1 graphs, quant-resident weights, aliasing, KV growth).
  Grafting training in would entangle two regimes and endanger the
  token-for-token CPU/GPU parity guarantees.
- **Not a separate repo:** training co-evolves the shared crates (new `Op`
  variants in `infr-core`, backward kernels in each backend, a GGUF writer in
  `infr-gguf`). Cross-repo path-dep churn would be constant friction.

### What the landscape survey found

- **Hand-rolled backward is proven.** llm.c reproduced GPT-2 1.6B on one node
  and was ~7% faster than PyTorch at the time; the whole trick is ~20
  hand-derived backward functions, fused kernels, and preallocated arenas — no
  tape. Rust ports (`llm.rs`, `llm.rust`) confirm the size (single-digit-K LOC)
  and that performance tracks C.
- **llama.cpp's trainer is the cautionary tale.** `finetune` /
  `train-text-from-scratch` were removed (PR #8669) after a graph refactor broke
  them: a general training graph bolted onto an inference engine, with no CI on
  backward ops, rots. The replacement (`ggml_opt`, PR #10544) remains primitive.
  Lesson: keep the training op surface tiny and gradcheck every op, or don't
  ship it.
- **No framework fits.** burn is the only Rust framework seriously pursuing LLM
  training (CubeCL Vulkan/Metal backends, coopmat matmul), but burn-lm training
  is unshipped alpha, and adopting it means a second tensor stack, duplicate
  loaders, and GGUF conversion. candle trains ~4× slower than PyTorch with
  quantized tensors outside autograd (no QLoRA path). tch-rs is a libtorch blob
  with no Vulkan. Track burn for tricks (SPIR-V cooperative-matrix, autotuned
  tiles); adopt nothing.
- **QLoRA is the memory-sane entry.** 7B ≈ 4-bit base ~3.6 GB + adapters ~30
  MB + AdamW state ~60 MB (r=8–16) + activations 2–4 GB with checkpointing →
  **~7–9 GB total**, comfortable on 12 GB consumer GPUs. The frozen quantized
  base gets **no gradients**, which sidesteps both the dequant→grad→requant
  problem and most of what killed llama.cpp's trainer.

---

## Net-new build (the honest bill)

1. **Backward ops** (~20, as new `Op` variants in `infr-core`): both
   transpose-side matmuls (`dX = dY·W`, `dW = dYᵀ·X`), rmsnorm-bwd, rope-bwd,
   attention-bwd (flash backward needs softmax stats the forward doesn't save
   today), swiglu/gated-act-bwd, fused softmax-cross-entropy, embedding
   scatter-add. Each with a CPU reference first.
2. **Batched full-sequence forward graphs.** The CPU seam is decode-only (n=1);
   training needs `[batch, seq]` shapes and **retained activations** — today
   `Internal` scratch is allocated inside `execute` and freed after. Training
   graphs declare activations as persistent outputs instead.
3. **Activation checkpointing.** The single most important non-inference
   feature; recompute per layer-block on the backward walk. Without it, 7B
   activations are hopeless; with it, 2–4 GB.
4. **Optimizer:** AdamW with f32 master weights + f32 m/v — for LoRA adapters
   only (~60 MB at r=16 on 7B; trivial). Grad-norm clipping.
5. **Checkpoint writer:** `infr-gguf` is read-only; add a writer (adapter export
   as GGUF LoRA and/or safetensors, resume state).
6. **Gradcheck harness — non-negotiable.** Per-op finite-difference tests plus
   an end-to-end PyTorch parity run (same data, same init, loss-curve match).
   Silent gradient bugs are THE failure mode of hand-rolled backward.

---

## Phase plan

| Phase  | Deliverable                                                                                                                                                                                                                                                         |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P0** | `infr-train` crate; batched full-seq forward on the CPU seam; backward ops for `Linear`/`RmsNorm`/`Attention`/`GatedAct`; finite-difference gradcheck harness.                                                                                                      |
| **P1** | End-to-end LoRA finetune of a small llama/qwen3 on CPU: loss demonstrably decreases, adapter exports and loads back into `infr run`, PyTorch parity on the loss curve.                                                                                              |
| **P2** | Vulkan backward kernels: coopmat matmuls with **f32 accumulate mandatory** (f16 accumulate diverges in training even where it's fine for inference); scatter-add via CAS or privatized reduction (float atomics are patchy cross-vendor). QLoRA 7B on a 12 GB card. |
| **P3** | (Optional) small-model pretraining: bf16 + loss scaling, fused softmax-xent. llm.c proved ~1.6B on a single node is realistic.                                                                                                                                      |

**Explicitly deferred:** a general tape/autodiff. Only add a thin recorder over
the same hand-written kernels if the arch families outgrow hand-emitted backward
— and only with per-op CI, per the llama.cpp lesson. Full finetuning (gradients
through all base weights, dequant→grad→requant, optimizer state at model scale)
is a separate later decision, not part of this plan.

---

## Anchors in the code

- `crates/infr-core/src/graph.rs` — extend `Op` with grad ops; training-graph
  builder lives beside it.
- `crates/infr-core/src/backend.rs` — `alloc`/`compile`/`execute` seam reused
  as-is; activations become persistent bindings.
- `crates/infr-cpu/src/lib.rs` — home of the reference backward kernels.
- `crates/infr-llama/src/cpu_backend.rs` — the forward decode-graph builder to
  mirror for a training-graph builder.
- `crates/infr-gguf/` — loader reused; writer is net-new.
- `crates/infr-hub/`, tokenizer — reused unchanged.

## Sources

- llm.c: <https://github.com/karpathy/llm.c> — GPT-2 1.6B repro:
  <https://github.com/karpathy/llm.c/discussions/677>
- Rust ports: <https://github.com/ToJen/llm.rs>,
  <https://github.com/Steboss/llm.rust>, <https://github.com/yijunyu/llm.rs>
- llama.cpp trainer removal: <https://github.com/ggml-org/llama.cpp/pull/8669>;
  `ggml_opt` training: <https://github.com/ggerganov/llama.cpp/pull/10544>,
  <https://github.com/ggml-org/llama.cpp/pull/13105>; checkpoint RFC:
  <https://github.com/ggml-org/llama.cpp/issues/15442>
- QLoRA: <https://arxiv.org/pdf/2305.14314>; 7B VRAM recipes:
  <https://kaitchup.substack.com/p/mistral-7b-recipes-for-fine-tuning>,
  <https://www.spheron.network/blog/gpu-vram-requirements-fine-tune-llm-2026/>
- burn / burn-lm: <https://burn.dev/blog/>,
  <https://burn.dev/blog/burn-lm-announcement/>,
  <https://github.com/tracel-ai/burn>
- candle training gaps: <https://github.com/huggingface/candle/issues/1383>,
  <https://github.com/huggingface/candle/issues/3052>; LoRA:
  <https://github.com/EricLBuehler/candle-lora>
- Vulkan-vs-CUDA training-adjacent perf:
  <https://github.com/ggml-org/llama.cpp/issues/17273>; WebGPU LLM perf:
  <https://arxiv.org/pdf/2605.20706>
