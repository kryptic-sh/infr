# Vulkan Q4_K packed metadata: first modern decode optimization

## Scope and implementation

Single-token dense decode on the modern integer-dot path uses
`quant_q8` + `native_mmv_mrow(rows=1)`: this is MMV, not the tiled prefill MMQ
kernel. The initial investigation targeted the reported RX 6600 XT (36 tok/s)
versus RX 580 FMA (41 tok/s) gap. Neither card was available locally.

Q4_K's 16-byte header contains both half scales and twelve packed scale/min
bytes. Load it with one `NW4`, unpack the two halves together, and extract the
same six-bit fields in registers instead of repeated byte/half loads. The
144-byte super-block stride preserves alignment. This changes **only weight
metadata loading/unpacking**, not the integer dots, floating-point arithmetic,
output/K lane assignment, or reduction tree.

Enable this only in Q4_K's four-output layout (`OUTS4`), including its MRV=4,
MRV=8 and single-row residual variants. Other dtypes, the two-output layout,
rows 9..16, fused quantization and tiled MMQ are unchanged. Retain scalar twins
for exact A/B checks. Default on; scalar optout:

```sh
INFR_NO_MMV_Q4K_META=1 infr bench "$M" -p 0 -n 16 -d 16 -r 3 --dev Vulkan1 --ctx 128
# Equivalent config/CLI override:
infr bench "$M" --set kernels.vulkan.mmv_q4k_meta=false
```

## End-to-end result on the available AMD integrated GPU

- AMD Ryzen 7 7800X3D integrated GPU, RADV RAPHAEL_MENDOCINO / Mesa 25.2.7.
- `unsloth/gemma-4-E4B-it-GGUF`, `gemma-4-E4B-it-Q4_K_M.gguf`, 4,977,171,584 bytes.
- Decode 16 tokens at depth 16, context 128; three repetitions per process.
- Scalar / packed / packed / scalar order, serial headless workloads.
- Native signed packed int8 dot enabled; cooperative matrices and KQ FMA disabled.
- Explicit empty config file to exclude machine-specific config overrides.
- Validation and profiling disabled during timing. Default decode replay retained.

| Path | First process (tok/s) | Second process (tok/s) | Mean (tok/s) |
| --- | ---: | ---: | ---: |
| Scalar Q4_K metadata | 6.14 | 5.86 | 6.00 |
| Packed Q4_K metadata | 7.66 | 7.31 | 7.49 |

**About 24.8% higher decode throughput** in this small integrated-GPU run.
This is not an RX 6600 XT measurement or a prediction for that card. The short
sequence is deliberately bounded for the integrated GPU; longer contexts,
discrete GPUs and other models still need direct measurements.

Reproduce with a local cached model, replacing the device as appropriate:

```sh
M=/path/to/gemma-4-E4B-it-Q4_K_M.gguf
printf '# empty benchmark config\n' > /tmp/infr-mmq-bench.toml
for variant in false true true false; do
  INFR_NO_COOPMAT=1 INFR_KQ_FMA_GEMV=0 INFR_KQ_FMA_TIER=0 \
    infr bench "$M" -p 0 -n 16 -d 16 -r 3 --dev Vulkan1 --ctx 128 \
      --config /tmp/infr-mmq-bench.toml \
      --set kernels.vulkan.mmv_q4k_meta=$variant --json
done
```

## RTX 4070 baseline and limitations

The requested common-feature baseline used RTX 4070 / NVIDIA 615.71.09 with
`INFR_NO_COOPMAT=1 INFR_SG=32 INFR_KQ_FMA_GEMV=0 INFR_KQ_FMA_TIER=0`.
This selects the shared signed-int8-dot path; it does **not** emulate RDNA2's
cache, memory bandwidth, wave64 or scheduling. RX 6600 XT's supplied feature
report has accelerated signed packed int8 dots but no GPU cooperative matrices.

Three initial runs of `kq_fma_gemv_bw` confirmed a weak shape: Q4_K 11008x2048
achieved 255/260/271 effective weight GB/s on int8 MMV versus 351/351/354 on
FMA NR4. This existing benchmark includes activation quantization, CPU command
recording, submission and waiting; it repeatedly reads one weight bank within
a batch and is not guaranteed cache-cold. It is neither tok/s nor pure streaming
bandwidth. Fused quantization was substantially slower and remains off.

Paired packed-metadata experiments improved this shape, but the later model
trials became highly variable. The GPU showed 50–98% utilization and roughly
4.3 GiB in use even after our benchmark processes exited. Consequently **no
reliable RTX 4070 end-to-end speedup is claimed**. Repeat the model A/B on an
idle GPU before drawing a conclusion or comparing it with the RX 6600 XT.

## Correctness gates and regression coverage

`tests/mmv_q4k_meta.rs::packed_metadata_matches_scalar_exact` checks:

- exact scalar/packed output bits and finite outputs;
- rows 1, 2, 3, 4, 5, 8, 9, 16 and both MRV=4/8 selections;
- output tails (1, 3, 5, 66), multiple K layouts and nonzero aligned weight offsets;
- zero, signed zero, subnormal and signed half scales; packed scale/min bit edges;
- fused residual addition; decode/verify row-0 identity within rows <=8;
- actual kernel cache names: all three packed variants must be built;
- scalar optout, `mmv_o4=false`, two-output and m16 fallbacks must stay scalar.

Exact parity passed on both the RTX 4070 and AMD integrated GPU with the
Khronos validation layer. A Gemma end-to-end generation also completed with
validation silence. GPU tests and their subprocesses must run serially and
headlessly. Example:

```sh
INFR_DEV=Vulkan1 VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation \
  cargo test -p infr-vulkan --release --test mmv_q4k_meta \
    packed_metadata_matches_scalar_exact -- --ignored --exact --nocapture --test-threads=1
```

`packed_metadata_bw` is a separate **bounded prequantized microbenchmark**:
record/submit/wait-inclusive times, one bank on integrated GPUs and four rotating
banks on discrete GPUs. Neither arrangement guarantees cold-cache bandwidth;
use it alongside, not instead of, end-to-end measurements.

Rejected experiments: changing all OUTS4 reductions to subgroup shuffles
regressed Q6_K; disabling OUTS4 globally hurt other projections. Neither change
is part of this patch. One-row specialization did not establish a reliable win
in the initial experiment and was not included.
