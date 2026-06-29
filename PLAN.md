# infr — Backend-agnostic compute refactor

The active plan for making `infr`'s compute backend swappable, so the same model
forward runs on **CPU / Vulkan / CUDA / ROCm / Metal / MLX** behind one seam.
Correctness first; performance (SIMD, fused kernels, batching) is a later pass.

> Broader/older project roadmap, crate layout, and the DiffusionGemma spec:
> [`docs/PLAN.md`](docs/PLAN.md). Live session state + perf numbers:
> [`CONTEXT.md`](CONTEXT.md). Related pending plans:
> [`docs/GPULOAD.md`](docs/GPULOAD.md) (in-shader GPU dequant),
> [`docs/STDTYPE.md`](docs/STDTYPE.md) (GGUF dtype coverage).

---

## Goal

Today every model forward in `infr-llama` is written imperatively against
`VulkanBackend` + `Recorder` — Vulkan is the only backend. The goal is a single
**semantic compute seam** that the model builds against, with multiple
interchangeable backends underneath. Nothing above the seam knows which device
is running.

Design priorities (from the user):

1. **Most agnostic + scalable** seam — must fit CPU and every GPU API, not leak
   Vulkan-isms.
2. **Correctness first**, performance later (SIMD / threading / fused kernels
   are a follow-on).
3. **Dtype-aware, not f32-everywhere** — tensors carry their real type (quant /
   f16 / f8 / f32); each backend does the math in that type. This is both the
   path to running real models without an 8× memory blow-up and what lets a GPU
   backend map straight onto its f16 / coopmat / mmq kernels.

---

## The seam (as built)

It lives in `infr-core` and is the **only** device-aware vocabulary.

### IR — an ordered op-list, not a pure DAG (`infr-core::graph`)

The real forward is imperative (scratch reuse, in-place RoPE, KV writes at a
running offset), so the IR is an **ordered list of ops over typed tensor
handles**, not an SSA DAG. Two ops may legally write the same handle (in-place /
scratch reuse) — order is significant, like a command buffer.

```
Graph { tensors: Vec<TensorDecl>, ops: Vec<Op>, inputs/weights/outputs }
TensorDecl { desc: TensorDesc (shape + DType), kind, label }
TensorKind = Input | Weight | Internal | Output
```

`Op` is **composite + semantic** (not scalar primitives), so a GPU backend maps
each op straight to a hand-fused kernel (no perf loss) while the CPU runs a
plain loop. Current op set:

- `RmsNorm`, `QkNorm` (per-head), `Linear` (dtype-dispatched matmul)
- `Rope` (split-half NEOX, optional `freq_factors`), `WriteKv`
- `Attention` (GQA + causal / sliding-window, per-call scale)
- `GatedAct` (SiLU / GELU, separate gate/up, `up_off` for E2B)
- `Add`, `Scale`, `Softcap`, `Copy`

Grow this set as models need; a future backend either implements the composites
or adds a lowering pass that decomposes them into primitives.

### Backend trait (`infr-core::backend`)

```rust
trait Backend {
    fn name() -> &str;  fn capabilities() -> Capabilities;
    fn alloc(bytes, usage) -> Box<dyn Buffer>;
    fn upload(dst, &[u8]);  fn download(src, &mut [u8]);
    fn compile(&Graph) -> Box<dyn Plan>;        // once per shape
    fn execute(&Plan, &Bindings);               // per token/step
    fn sync();
}
```

- The model owns long-lived buffers (weights, KV cache) and per-step IO buffers,
  and binds them to the graph's `Input`/`Weight`/`Output` handles via
  `Bindings<'a>` (borrows; rebinding each step is cheap, no recompile).
- The backend allocates only `Internal` scratch during `execute`.
- `Buffer`/`Plan` carry an `as_any()` downcast hook so a backend recovers its
  concrete type from a bound `&dyn`.
- **Device-specific kernel selection lives inside `compile`** (mpad padding,
  coopmat vs GEMV vs mmq, f16 buffers, flash variants) — invisible to the IR.

### Dtype-awareness

Tensor handles carry a real `DType`. Weights keep their **native GGUF dtype**
(quant / f16 / bf16); the backend dequants as part of its kernel, not as a
host-side materialization. The CPU backend already does this (raw upload + lazy
dequant). The Vulkan adapter will use it to dispatch `linear_q` / coopmat by
weight dtype and to keep q / KV in f16.

---

## Status (2026-06-29)

| Piece                                                                        | State              |
| ---------------------------------------------------------------------------- | ------------------ |
| IR + Backend/Bindings seam (`infr-core`)                                     | ✅ `b6ef784`       |
| CPU reference backend (`infr-llama::cpu_backend`)                            | ✅ `4f3ba7b`       |
| Qwen3 / Llama dense on CPU, validated vs GPU                                 | ✅                 |
| Gemma 3 on CPU (sandwich norms, GeGLU, dual-RoPE, SWA, embed scale, softcap) | ✅ `c4d9a78`       |
| CLI `INFR_CPU=1` one-shot                                                    | ✅ `39ada38`       |
| Dtype-aware native-dtype weights                                             | ✅ `9964451`       |
| Vulkan adapter (`compile`/`execute`)                                         | ⬜ still `todo!()` |
| gemma4 / E2B / MoE on the seam                                               | ⬜                 |

**Validation:** `Llama::generate_cpu` mirrors `generate` (same tokenize/decode);
the CPU output must match the GPU greedy path **token-for-token**. Gated tests
in `crates/infr-llama/tests/cpu_backend.rs` (`cpu_matches_gpu_greedy`,
`cpu_matches_gpu_gemma3`) — run with `INFR_TEMP=0`, `--ignored`. The CPU backend
is also the oracle that future GPU backends are checked against.

**Math conventions to mirror exactly** (from infr's own CPU oracles in
`infr-vulkan/src/recorder.rs` tests): RMSNorm `x·rsqrt(mean(x²)+eps)·w` (gemma
GGUFs bake the `+1` into `w`, so no `+1` on our side); RoPE is split-half pairs
`(p, p+rope_dim/2)`, `freq = θ^(-2p/rope_dim)`, `ang = pos·freq [/ ff[p]]`;
attention scale `1/√hd` (gemma4 uses `1.0`); fused gate‖up is
`gate = gu[base+i]`, `up = gu[base+nff+i]`.

---

## Decisions

- **Graph/IR over a Recorder-mirroring trait.** A trait that mirrors the Vulkan
  `Recorder` methods would leak Vulkan-isms and force per-op submit on GPU
  backends (no batching). The semantic op-list batches on every backend and maps
  to MLX/Metal's graph/lazy model.
- **Composite ops, not pure primitives.** Keeps Vulkan's hand-fused kernels;
  primitives/fusion can be added later as a lowering pass.
- **Dtype-aware, not f32-everywhere** (see above).
- **CPU backend lives in `infr-llama` for now** — next to `dequant_block` + the
  qwen35 oracle, to avoid a circular crate dep. It implements the agnostic
  `infr_core::Backend`, so it can be extracted to an `infr-cpu` crate later
  without touching callers.
- **CPU bring-up is decode-only.** One `n=1` decode graph drives both prompt
  ingestion (looped) and generation, so no GEMM/flash prefill kernels are needed
  on CPU; the KV cache grows one row per step.

---

## Remaining phases

1. **Bounded f32 weight cache / true quantized matvec.** The CPU backend's lazy
   dequant still caches the whole model in f32 (~29 GB for an E2B-size model,
   OOM for 12B / MoE-30B). Dequant per-block inside the dot product (no full f32
   materialization), or a size-bounded cache, removes the memory wall and lets
   real models run on CPU.
2. **f16 KV cache + activations on CPU.** Match the GPU's f16 representation —
   halves KV memory and tightens parity.
3. **Vulkan adapter (`compile`/`execute`).** Map the dtype-aware graph onto the
   existing fused Recorder/shader kernels (coopmat / mmq / flash, f16 q+KV) into
   one command buffer, reproducing today's perf. Then **convert the GPU dense
   forward in place** to emit the same `Graph` and retire the parallel path —
   one graph, both backends.
4. **gemma4 / E2B / MoE on the seam.** gemma4: per-layer head dims, weightless
   V-norm, proportional-RoPE `freq_factors`, per-layer output scale, V=K on full
   layers; then E2B's per-layer input embeddings + KV-layer sharing. MoE: a
   routed-expert op (router softmax → top-k → per-expert FFN → combine).
5. **Cleanup.** Route all models through the seam, delete dormant
   direct-Recorder entry points, refresh `docs/PLAN.md` + `CONTEXT.md`. Extract
   `infr-cpu` crate.

---

## File map

| Path                                     | Role                                                                  |
| ---------------------------------------- | --------------------------------------------------------------------- |
| `crates/infr-core/src/graph.rs`          | IR: `Graph`, `Op`, `TensorDecl`, `TensorKind`                         |
| `crates/infr-core/src/backend.rs`        | `Backend`, `Buffer`, `Plan`, `Bindings`, `Capabilities`               |
| `crates/infr-core/src/tensor.rs`         | `DType`, `TensorDesc`, `TensorId`                                     |
| `crates/infr-llama/src/cpu_backend.rs`   | CPU backend + dense decode-graph builder + runner                     |
| `crates/infr-llama/tests/cpu_backend.rs` | CPU-vs-GPU parity tests (gated)                                       |
| `crates/infr-vulkan/src/lib.rs`          | `Backend` impl (alloc/upload/… done; `compile`/`execute` = `todo!()`) |
| `crates/infr-vulkan/src/recorder.rs`     | the fused GPU kernels the Vulkan adapter will target                  |
