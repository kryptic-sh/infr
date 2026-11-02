# EXL3 checkpoints

The Qwen2 CPU reference path accepts a local EXL3 `model.safetensors` with adjacent
`config.json` and `tokenizer.json`. If present, `tokenizer_config.json` supplies
the chat template. Hugging Face automatic model resolution still targets GGUF;
download the EXL3 branch before passing its local weight path.

```sh
INFR_DEV=cpu infr perplexity /path/to/model.safetensors -f corpus.txt -c 512 --chunks 2
RUST_LOG=info cargo run -p infr-llama --release --example exl3_smoke -- /path/to/model-directory
```

The initial checkpoint is `turboderp/Qwen2.5-7B-Instruct-exl3`, branch `4.0bpw`.
Supported files have Qwen2 architecture metadata, separate embeddings and LM
head, integer rates 1–8, and the default three-instruction codebook. `.mcg` and
`.mul1` codebooks, fractional rates, custom RoPE scaling, sliding-window
attention, and sharded checkpoints are rejected explicitly. Bit rate and
codebook are independent: the test checkpoint uses the default codebook at
4 bits for projections and 6 bits for the LM head.

## CPU reference representation

The loader reconstructs each linear into row-major F16, including both
orthonormal 128-element Hadamard transforms and input/output scales. HF Q/K row
ordering is converted by the existing Qwen2 binder. Norm and bias vectors are
widened to F32 to match all backend scalar-vector kernels; embeddings retain
their F16 mmap representation. Loader-owned folds bypass the source's borrowed
weight cache so resident CPU weights do not retain a second folded copy.

This path validates model integration, not compressed-memory performance. A 7B
model expands to approximately 14 GB of linear weights, in addition to
embeddings, KV cache, and loader scratch. Native compressed GPU execution is a
separate implementation, not provided by this reference representation.

## Validation

```sh
cargo test -p infr-safetensors
cargo test -p infr-cpu --test exl3_parity
cargo test -p infr-llama --lib exl3_source
cargo test -p infr-llama --lib perplexity_window
```

Synthetic tests cover config/tensor validation, signed nonuniform scales,
multiple 128-row output bands, all integer bit rates, folded versus unfused
Hadamard forward execution, and batched versus fresh causal-prefix logits.
The pre-existing trellis tests independently check bit extraction and CUDA
fragment layouts against the pinned exllamav3 source.

Perplexity scores the second half of each context window, excluding its final
position. Two 512-token windows therefore score 510 targets, not 1,024.
Windows retain their first corpus token for models that do not prepend BOS.
An early Alice-in-Wonderland smoke score is not a quality gate: its boilerplate
and famous opening may be memorized. Compare the same unfamiliar corpus,
tokenizer, and scoring scheme against an upstream reference before claiming
model-quality parity or an advantage over another quantization.

## Weight-source API migration

`SeamModel::weights()` replaces the GGUF-specific accessor and returns
`&dyn WeightSource`. Configuration, chat templates, and common tensor loading
accept this trait. Format-specific paging and zero-copy GGUF views use
`weights.as_any().downcast_ref::<Gguf>()`; callers must handle non-GGUF sources.
Custom `WeightSource` implementations must implement `as_any()` by returning
`self`. Paging that requires GGUF shard byte ranges is unavailable for EXL3.
