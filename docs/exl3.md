# EXL3 checkpoints

Infr loads a local Qwen2 EXL3 `model.safetensors` with adjacent `config.json` and
`tokenizer.json`; `tokenizer_config.json`, if present, supplies the chat template.
Hugging Face auto-resolution still targets GGUF, so download the EXL3 branch first
and pass the local weight path.

```sh
INFR_DEV=cpu infr perplexity /path/to/model.safetensors -f corpus.txt -c 512 --chunks 2
RUST_LOG=info cargo run -p infr-llama --release --example exl3_smoke -- /path/to/model-directory
```

Test checkpoint: `turboderp/Qwen2.5-7B-Instruct-exl3`, branch `4.0bpw`
(default codebook; 4-bit projections, 6-bit LM head).

Supported: Qwen2 architecture, untied embeddings, integer rates 1–8, the default
`3inst` codebook (cb0). Rejected with an explicit error: `.mcg`/`.mul1` codebooks,
fractional rates, RoPE scaling, sliding-window attention, sharded checkpoints, and
tied embeddings. Every released turboderp Qwen2.5-7B branch inspected (2/3/4 bpw)
is cb0; rate and codebook are independent.

## Execution

Projections stay compressed. `Exl3Source::open_native` exposes each linear as three
weights — the raw I16 trellis, F16 `suh`, F16 `svh` — and the seam rewrites the
matching `Op::Linear` into `Op::LinearExl3` after graph construction. Per row:

```text
xh = H128(x ⊙ suh)            input scale + orthonormal 128-point Hadamard
z  = xh · D                   D decoded tile by tile from the trellis; never stored
y  = scatter(H128(z) ⊙ svh)   output Hadamard + scale, then the Q/K rotary reorder
```

Q/K projections carry `Exl3OutputLayout::RopeInterleaved`, which reorders outputs
from HF rotate-half order to the interleaved order the existing `Op::Rope` expects;
biases are permuted by the existing binder. Norm and bias vectors are widened to F32
at load; embeddings stay F16 mmap views.

| backend | status |
|---|---|
| CPU | `infr_cpu::exl3_linear` streams 16×16 tiles into 8-row × 128-column accumulators; no expanded weight is allocated |
| Vulkan | three dispatches: `exl3_hadamard_in`, `exl3_gemv` (m = 1) or `exl3_prefill` (8-row tiles), `exl3_hadamard_out`; trellis and scales are read by buffer device address |
| Metal | not implemented; graphs containing `LinearExl3` are rejected |

The Vulkan kernels are portable correctness-first implementations: no subgroup-size
assumption, no cooperative matrix, no integer dot product. The codebook's half-precision
sum is rounded to nearest-even with integer arithmetic because `packHalf2x16` rounding is
implementation-defined.

`Exl3Source::open` keeps the earlier folded-F16 reference path for tests: it expands
each linear to row-major F16 (about 14 GB for the 7B linears). Production loading no
longer uses it.

Disk paging and dense streaming need GGUF shard byte ranges and are unavailable for
EXL3. Resident footprint is the raw trellis, scale, embedding, norm, and bias bytes.

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
- Full synthetic Qwen2 logits from the native path match the folded reference, and
  batched logits match fresh causal-prefix forwards.
- The Vulkan projection matches the CPU kernel within `1e-4 × max|y|` for rates 1–8,
  identity and rotary layouts, m ∈ {1, 2, 7, 8, 9, 17}, and the Qwen2.5-7B projection
  shapes (3584×512, 3584×18944, 18944×3584, 3584×3584). Verified on an RX 580
  (RADV Polaris), the Ryzen 7800X3D integrated GPU (RADV), and an RTX 4070 (NVIDIA 615.71.09).
- Full Qwen2.5-7B EXL3 4.0bpw chat generation runs on the RTX 4070 through Vulkan:
  `infr run /path/to/model.safetensors "What is the capital of France?" --dev Vulkan0
  --ctx 256 --ubatch 64 --temp 0 --max-new 16` produces
  `The capital of France is Paris.` Both prefill and token decoding execute on Vulkan.

Perplexity scores the second half of each context window, excluding its final
position: two 512-token windows score 510 targets. Windows keep their first corpus
token for models that do not prepend BOS. No perplexity result has been compared
against an upstream exllamav3 reference yet, so none is a quality claim.

## Weight-source API migration

`SeamModel::weights()` replaces the GGUF-specific accessor and returns
`&dyn WeightSource`. Format-specific paging and zero-copy GGUF views use
`weights.as_any().downcast_ref::<Gguf>()`; callers must handle non-GGUF sources.
Custom `WeightSource` implementations must implement `as_any()` by returning `self`.
