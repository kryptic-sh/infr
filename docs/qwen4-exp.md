# Qwen4-Exp: dense HF CPU reference (experimental)

Tracking: [issue #18](https://github.com/KessokuNet/infr/issues/18).

**Dense HF loading and CPU-reference text inference are implemented. GPU execution
is not implemented.** Qwen3.8-Flash-Next / Qwen4 Experimental is not an alias for
Qwen3.5: its HC residual streams, PLE n-gram embeddings, QSA indexer and GDN output
gate execute together in a dedicated scalar f32 implementation.

## What loads

- A local, single-file **dense** HF `.safetensors` checkpoint with adjacent
  `config.json`, declaring `qwen4_exp` (nested `text_config`) or
  `qwen4_exp_text` (standalone text config).
- `SeamModel::load_with` selects the architecture adapter using that config.
  It reuses `infr-safetensors`, the same mmap reader used by EXL3; EXL3 dispatch
  and native trellis handling remain unchanged.
- `tokenizer.json` alongside the checkpoint is used unless the caller supplies
  an explicit tokenizer path. The adjacent `tokenizer_config.json` supplies the
  chat template when present.
- BF16, F16 and F32 text weights, without copying, widening or stacking expert
  matrices. Tensor descriptors reverse HF dimensions into infr's
  fastest-dimension-first convention; their bytes still borrow the mmap.
- Individual MoE expert projections and packed `gate_up_proj` / `down_proj`
  tensors; shared experts; HC mixers; linear attention; indexed attention; PLE
  embeddings with either row shards or a consolidated table; tied output heads.
- Persistent PLE I64 multipliers, vocabulary sizes and offsets, retained as
  **host-only buffers**. They are not graph tensors. The generic reader's
  ordinary `Safetensors::open` remains strict; the architecture adapter opts
  into `open_with_i64_buffers` explicitly.

Every required text weight and its shape is checked at load time. Unknown text
weights, unexpected integer buffers, inconsistent geometry and quantized HF
configs fail explicitly. PLE buffers must match the configured seed, vocabulary
base and PLE ordinal, not merely have plausible shapes. Parsing is bounded to
4096 layers and 65536 prime-search candidates per PLE table to prevent unbounded
allocation or numeric work on malformed inputs. Vision tensors are ignored:
this is text inference, not multimodal support.

The infr architecture identifier is **`qwen4_exp`** (HF spelling). This does
**not** add support for GGUF's separate `qwen4exp` spelling, sharded HF index
files, remote HF model resolution, EXL3 quantization of this family, or MTP.
`arch::ALL` enumerates shared-runner/GGUF families; this dedicated CPU family is
not in that list.

## CPU inference

`SeamModel` CPU generation and last/all-row logit verification dispatch directly
to the reference before ordinary tensor binding, embedding materialization or
paging. They preserve request sampling and penalties, grammar constraints,
cancellation and effective runtime EOS IDs (including tokenizer-added chat
terminators). GPU/session, generic graph, speculative and diffusion paths still
refuse; they must not substitute Qwen3.5 math.

For explicit recurrent-cache use, `qwen4_exp::CpuModel::new(&source)` borrows a
validated Qwen4-Exp weight source. `step(token)` returns final hidden states and
logits, `prefill(tokens)` returns causal logit rows, and `reset()` resets
attention/GDN/PLE caches and position. Invalid token/context prefill requests are
rejected before changing state. Geometry comes from the validated source, not
a mutable caller configuration.

The implementation intentionally favors correctness over throughput:

- All arithmetic and recurrent state use f32. Configurations requesting F16 or
  BF16 **recurrent cache dtype** refuse, even though their weights can load.
- Model weights stay mmap-backed; PLE table rows are accessed lazily. There is
  no eager full-model float expansion.
- Persistent cache allocation is checked against **256 MiB**, including reserved
  outer row-vector capacities. Output requests have a separate **256 MiB** bound,
  including nested prefill row metadata. These are cache/output allocation bounds,
  not a total-process-memory guarantee; weights, activations and allocator overhead
  are additional. Indexed row headers reserve the configured full context, so
  overly large reference contexts can refuse before evaluation.
- QSA block and MoE expert top-k ties select the lowest ID deterministically.
  HF's `torch.topk` does not specify tied-boundary ordering; such ties can therefore
  differ from HF without representing a scalar-math error.
- PLE uses the original HF first EOS for n-gram hashing, independently of the
  generation stop-token list.

This is suitable for the tiny random checkpoint and reference/debugging work,
not a fast inference path for the 176.9B model. No GPU memory or performance
claim is made, and no 4070 run has been attempted.

## Verification

Network-free Rust tests reconstruct deterministic BF16 weights from the schema
of [`malaiwah/qwen4-exp-tiny-random-bf16`](https://huggingface.co/malaiwah/qwen4-exp-tiny-random-bf16).
The committed JSON oracle contains expected outputs, not model weights. The tiny
model has hidden size 64, four HC streams, three linear-attention layers and one
indexed-attention layer, four experts (top two), and PLE at one-indexed layer 2.

Tests compare 17 causal final-hidden/logit rows and GDN state/raw-convolution
snapshots against CPU Hugging Face Transformers, and compare chunked Rust prefill
with a whole-sequence HF pass, using absolute tolerance **2e-5**. They also cover
reset, context/invalid-input handling, allocation limits, packed/individual expert
and PLE layouts, tied heads, deterministic ties, last/all-row API dispatch,
seeded sampling with penalties, grammar, EOS and cancellation. Loader and existing
EXL3 regressions run alongside them.

```sh
cargo test -p infr-safetensors
cargo test -p infr-llama --lib
cargo clippy -p infr-llama -p infr-safetensors --all-targets -- -D warnings
```

The oracle generator is optional, network-free and CPU-only once its Python
dependencies are installed. It requires CPU PyTorch, NumPy, safetensors and
Transformers pinned to revision `cc1451df3b5b218295bc16803569256c53c9d6e2`
(verified with Python 3.12.16, PyTorch 2.15.0+cpu and NumPy 2.5.4). These are
not Rust/runtime dependencies. To regenerate the synthetic oracle:

```sh
python crates/infr-llama/tests/generate_qwen4_exp_reference.py \
  --work-dir /tmp/qwen4-oracle
```

The same hidden/logit/cache comparisons have also passed against the **real
cached tiny checkpoint**, not only synthetic weights. To reproduce locally,
keep `config.json` beside the cached `model.safetensors`, generate a local oracle
beside it, and run both ignored tests without downloading a model:

```sh
python crates/infr-llama/tests/generate_qwen4_exp_reference.py \
  --work-dir /tmp/qwen4-oracle --checkpoint /path/to/tiny-model-directory \
  --output /path/to/tiny-model-directory/cpu_reference.json
INFR_TEST_MODEL=/path/to/tiny-model-directory/model.safetensors \
  cargo test -p infr-llama --lib cached_tiny -- --ignored
```

Remaining work is native GPU kernels/graph integration and parity before any
GPU support claim. See [Stage C in the Qwen3.8 investigation](qwen38.md#stage-c--qwen38-flash-next-qwen4exp)
for the architectural differences and large-model memory constraints.
