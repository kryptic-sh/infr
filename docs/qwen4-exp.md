# Qwen4-Exp: HF checkpoint loading (experimental)

Tracking: [issue #18](https://github.com/KessokuNet/infr/issues/18).

**Loading only — inference is not implemented.** This is the first slice of
Qwen3.8-Flash-Next / Qwen4 Experimental architecture support. It is not an alias
for Qwen3.5: HC residual streams, PLE n-gram embeddings, the QSA indexer and the
GDN output gate need their own forward implementation.

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
  embeddings with either row shards or a consolidated table.
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
this is text loading, not multimodal support.

The infr architecture identifier is **`qwen4_exp`** (HF spelling). This does
**not** add support for GGUF's separate `qwen4exp` spelling, sharded HF index
files, remote HF model resolution, EXL3 quantization of this family, or MTP.
`arch::ALL` continues to list executable architectures and does not include this
loading-only family.

## Why generation refuses

There are no ordinary attention/FFN/final RMSNorm tensors in this architecture.
Replacing its HC mixers with such norms, or running the existing Qwen3.5 hybrid
forward, would produce incorrect results even though much of the geometry looks
similar. All dense generation backends refuse with:

```
arch=qwen4_exp: checkpoint loading is supported, but inference is not yet implemented (requires Qwen4-Exp HC, PLE, QSA and GDN output-gate semantics)
```

CPU generation refuses before embedding materialization or tensor binding.
Vulkan/Metal session creation refuses before device initialization or placement.
No GPU memory or performance claims are made by this slice.

## Verification

Network-free tests reproduce the tensor schema and integer buffers of
[`malaiwah/qwen4-exp-tiny-random-bf16`](https://huggingface.co/malaiwah/qwen4-exp-tiny-random-bf16),
with synthetic float data. The JSON fixtures contain no model weights. The tiny
model has hidden size 64, four HC streams, three linear-attention layers and one
indexed-attention layer, four experts (top two), and PLE at one-indexed layer 2.
Tests cover config semantics, mmap aliasing, complete text mapping, both expert
layouts, both PLE layouts, malformed checkpoints, HF tokenizer dispatch and
early CPU/GPU refusal. Existing EXL3 regression tests run alongside them.

```sh
cargo test -p infr-safetensors
cargo test -p infr-llama --lib
cargo clippy -p infr-llama -p infr-safetensors --all-targets -- -D warnings
```

A separate ignored test checks the real tiny checkpoint's schema without
network access. Set the path to a cached `model.safetensors` whose directory also
contains the original `config.json`:

```sh
INFR_QWEN4_EXP_TEST_MODEL=/path/to/model.safetensors \
  cargo test -p infr-llama --lib cached_tiny_checkpoint_schema -- --ignored
```

This checks loading, **not reference logits or token generation**. The remaining
forward slices need CPU reference parity first, then GPU support and parity. See
[Stage C in the Qwen3.8 investigation](qwen38.md#stage-c--qwen38-flash-next-qwen4exp)
for the architectural differences and large-model memory constraints.
