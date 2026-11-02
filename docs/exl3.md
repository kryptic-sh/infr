# EXL3 checkpoints

Infr loads local Qwen2, text-only Qwen3.5, and dense Gemma 4 EXL3
`model.safetensors` checkpoints with adjacent `config.json` and `tokenizer.json`. The chat template comes from
`tokenizer_config.json` or an adjacent `chat_template.jinja`.
Hugging Face auto-resolution still targets GGUF, so download the EXL3 branch first
and pass the local weight path.

```sh
INFR_DEV=cpu infr perplexity /path/to/model.safetensors -f corpus.txt -c 512 --chunks 2
RUST_LOG=info cargo run -p infr-llama --release --example exl3_smoke -- /path/to/model-directory
```

Test checkpoint: `turboderp/Qwen2.5-7B-Instruct-exl3`, branch `4.0bpw`
(default codebook; 4-bit projections, 6-bit LM head).

Supported: Qwen2, Qwen3.5 hybrid text, and Gemma 4 dense text architectures,
integer rates 1–8, the default `3inst` codebook (cb0), `mcg` with multiplier
`0xCBAC1FED`, and `mul1` with multiplier `0x83DCD12D`. Every released turboderp Qwen2.5-7B branch inspected (2/3/4 bpw)
is cb0; rate and codebook are independent.

Qwen3.5 loading translates the HF tensor names and configuration into the existing
`qwen35` runtime; it does not add another DeltaNet or attention implementation.
The adapter handles zero-centered norms, `A_log` decay conversion, BF16 convolution
parameters, and grouped-to-tiled value-head ordering. Vision and MTP tensors are
not exposed by the text-only source.

Gemma 4 loading reuses the existing `gemma4` runtime. It handles heterogeneous
sliding/full attention head dimensions and KV-head counts, KV sharing, per-layer
inputs, proportional RoPE, and BF16 norms without an added unit offset. The supplied
6-bit LM head stays independent even when the HF configuration marks embeddings tied.
Vision tensors are not exposed.

Rejected with an explicit error: custom codebook multipliers, fractional rates,
unsupported architecture-specific RoPE/attention configurations, Gemma 4 MoE or
double-width FFNs, sharded EXL3 checkpoints, and checkpoints requiring an absent
tied LM head.

## Execution

Projections stay compressed. `Exl3Source::open_native` exposes each linear as three
weights — the raw I16 trellis, F16 `suh`, F16 `svh` — and the seam rewrites the
matching `Op::Linear` into `Op::LinearExl3` after graph construction. Per row:

```text
xh = H128(gather(x) ⊙ suh)   input ordering + scale + orthonormal Hadamard
z  = xh · D                  D decoded tile by tile from the trellis; never stored
y  = scatter(H128(z) ⊙ svh)  output Hadamard + scale, then output ordering
```

Qwen2 Q/K projections carry `Exl3Layout::RopeInterleaved`, which reorders outputs
from HF rotate-half order to the interleaved order the existing `Op::Rope` expects;
biases are permuted by the existing binder. Qwen3.5 DeltaNet projections use
`Exl3Layout::GroupedValueHeads`: input gather precedes the original scale/Hadamard,
and output scatter follows the original Hadamard/scale. Compressed factors are not
permuted. Full-attention Q/gate interleaving remains unchanged.
Norm and bias vectors are widened to F32 at load. Qwen embeddings stay F16 mmap
views. Gemma 4 embeddings are gathered from the host mmap one requested row at a time:
BF16 dense rows are converted to F16, while ROW256 trellises decode the circular
LSB-first symbol stream, fixed `mul1` codebook, group gain, H256, and stored signs.
Neither token nor per-layer tables are expanded or uploaded in full. The main
embedding multiplier is BF16-rounded; main and per-layer products retain their
F16 rounding boundary before being uploaded as F32 graph inputs.

| backend | status |
|---|---|
| CPU | `infr_cpu::exl3_linear` streams 16×16 tiles into 8-row × 128-column accumulators; no expanded weight is allocated |
| Vulkan | three dispatches: `exl3_hadamard_in`, `exl3_gemv` (m = 1) or `exl3_prefill` (8-row tiles), `exl3_hadamard_out`; trellis and scales are read by buffer device address |
| Metal | not implemented; graphs containing `LinearExl3` are rejected |

The Vulkan kernels are portable correctness-first implementations: integer decoding
and FP32 projection/transform arithmetic, with no `shaderFloat16` requirement,
subgroup-size assumption, cooperative matrix, or integer dot-product requirement.
The RX 580 uses this same path automatically; no separate legacy-math switch is needed.
The codebook's half-precision sum is rounded to nearest-even with integer arithmetic
because `packHalf2x16` rounding is implementation-defined. The `mul1` byte-sum/FMA
path uses exact FP32 intermediates before the same rounding; all 65,536 windows
are checked against the CPU half-FMA oracle.

`Exl3Source::open` keeps the earlier folded-F16 reference path for tests: it expands
each linear to row-major F16 (about 14 GB for the 7B linears). Production loading no
longer uses it.

Disk paging and dense streaming need GGUF shard byte ranges and are unavailable for
EXL3. Resident footprint is the uploaded raw trellis, scale, norm, and bias bytes;
host-only row embeddings retain their packed/dense size in metadata but are not
charged against VRAM.
Large resident BDA tensors use near-exact-sized blocks; small tensors share arena
blocks. This avoids charging a full 64 MiB block to each half-block-sized projection.

## Performance status

The portable EXL3 decode kernel is not yet competitive with the optimized GGUF
path on the RX580. A warmed E4B QAT comparison (`infr bench -p 0 -n 64 -d 16 -r 3`,
context 256, microbatch 64) measured:

| Path | Decode tok/s |
| --- | ---: |
| Cached Google QAT Q4_0 GGUF | 41.48 |
| Initial EXL3 4.0bpw kernel | 3.89 |
| EXL3 with register-local accumulation and cheaper circular indexing | 5.21 |

Before the cleanup, GPU timestamps attributed 96.5% of device time to EXL3 GEMV;
both Hadamard passes together accounted for less than 1%. Removing the per-K-tile
shared-memory round-trip and barriers improves throughput, but packed-window
decoding and the narrow K-parallelism remain the next optimization targets. Lower
stored bpw does not translate into higher throughput until this kernel is efficient.
These timings compare the same architecture/QAT lineage, not bit-identical weights.

## Validation

```sh
cargo test -p infr-safetensors
cargo test -p infr-core --lib exl3
cargo test -p infr-cpu --lib exl3
cargo test -p infr-cpu --test exl3_parity
cargo test -p infr-llama --lib exl3_source
cargo test -p infr-llama --lib perplexity_window
cargo test --release -p infr-vulkan --test exl3_linear_parity -- --ignored   # needs a GPU
```

- Trellis bit extraction and CUDA fragment layouts are checked against the pinned
  exllamav3 source; the tile permutation is checked as a bijection.
- The CPU native projection matches a dense decode + Hadamard + scale reference at
  every integer rate, with partial-rotary output scatter and tail batches.
- Full synthetic Qwen2, Qwen3.5, and Gemma 4 logits from the native path match the
  folded reference, and batched logits match fresh causal-prefix forwards. Unequal
  key/value-head counts check the non-involutive value-head permutations.
- ROW256 lookup covers every integer rate, circular group tails, partial groups,
  repeated/unsorted/boundary token ids, nonfinite parameters, and explicit Walsh
  reconstruction. Compact first/last-group goldens from the pinned E4B checkpoint
  are generated independently by `crates/infr-cpu/tests/generate_exl3_embedding_reference.py`.
- The Vulkan projection matches the CPU kernel within `1e-4 × max|y|` for rates 1–8,
  identity, rotary, and grouped input/output layouts, m ∈ {1, 2, 7, 8, 9, 17},
  and Qwen-sized projections.
  `3inst`, `mcg`, and `mul1` are verified on an RX 580 (RADV Polaris), including
  4096×8192 and 4096×4096 at 2bpw and Gemma E4B-sized 4bpw projections. `3inst` is also verified on the Ryzen 7800X3D
  integrated GPU (RADV) and an RTX 4070 (NVIDIA 615.71.09).
- Full Qwen2.5-7B EXL3 4.0bpw chat generation runs on the RTX 4070 through Vulkan:
  `infr run /path/to/model.safetensors "What is the capital of France?" --dev Vulkan0
  --ctx 256 --ubatch 64 --temp 0 --max-new 16` produces
  `The capital of France is Paris.` Both prefill and token decoding execute on Vulkan.
  The same checkpoint also loads and generates on an 8 GB RX 580, with the VRAM
  guard enabled and the portable FP32 kernels.

- Full text generation on the 8 GB RX 580 also runs for
  `cortexist/gemma-4-E4B-it-qat-exl3-4.0bpw` and upstream
  `turboderp/gemma-4-12B-it-exl3` at `2.00bpw`. The former uses row-quantized token
  and per-layer embeddings; the latter has a dense BF16 token table. Both keep
  projections compressed and the VRAM guard enabled. These are loading/generation
  smoke tests, not quality or performance benchmarks.

The Qwen3.5-9B EXL3 `2.00bpw` checkpoint's end-to-end Vulkan generation has not yet
been verified.

Perplexity scores the second half of each context window, excluding its final
position: two 512-token windows score 510 targets. Windows keep their first corpus
token for models that do not prepend BOS. No perplexity result has been compared
against an upstream exllamav3 reference yet, so none is a quality claim.

## Weight-source API migration

`SeamModel::weights()` replaces the GGUF-specific accessor and returns
`&dyn WeightSource`. Format-specific paging and zero-copy GGUF views use
`weights.as_any().downcast_ref::<Gguf>()`; callers must handle non-GGUF sources.
Custom `WeightSource` implementations must implement `as_any()` by returning `self`.
`tensor_rows(name, ids)` is an optional bounded embedding hook with a default of
`Ok(None)`; row-only sources return selected raw F32 rows and reject whole-table
`tensor_bytes` access.
