# CPU backend performance

Findings and results for the `infr-cpu` reference backend, from the CPU perf
review. The worklist it carried has landed; its open items are in `backlog.md` §
B101.

## Results snapshot

Landed so far (bit-identical unless noted; precision-flip slices
coherence-checked token-identical to the independent Vulkan int8 path before
acceptance):

| Slice                              | Model / quant     | Decode       | Prefill (pp512)   |
| ---------------------------------- | ----------------- | ------------ | ----------------- |
| conv1d parallel (`ac9c228`)        | Qwen3.5-9B Q4_K_M | —            | flat (GEMM-bound) |
| mmap madvise (`5ed932a`)           | Qwen3.5-9B Q4_K_M | neutral      | neutral           |
| DeltaNet head-parallel (`9595bf3`) | Qwen3.5-9B Q4_K_M | flat         | 67→110 t/s (+63%) |
| native int8 **Q4_0** (`6e7decd`)   | Qwen3-0.6B Q4_0   | +142% (2.4×) | +239% (3.4×)      |
| native int8 **IQ4_XS** (`304dd42`) | Qwen3-0.6B IQ4_XS | +156% (2.6×) | +320% (4.2×)      |
| native int8 **Q2_K** (`1e90613`)   | Qwen3-0.6B Q2_K   | +29%         | +71%              |
| native int8 **Q3_K** (`d559984`)   | Qwen3-0.6B Q3_K_M | +185% (2.9×) | +199% (3.0×)      |
| native int8 **Q4_1** (`f4738a1`)   | Qwen3-0.6B Q4_1   | +138% (2.4×) | +243% (3.4×)      |
| native int8 **IQ4_NL** (`0ef8366`) | Qwen3-0.6B IQ4_NL | +47%         | +253% (3.5×)      |

### Full CPU↔Vulkan native-quant parity (unit-test-verified — no models)

CPU now has a native int8 kernel for **every format Vulkan has a native mmq
kernel for**. The six below close the last of the gap; none has a small
supported-arch GGUF for an end-to-end coherence check (`IQ2_S`/`IQ3_S` are
big-model-only; `Q5_1`/`Q2_0` legacy; `MXFP4`=gpt-oss, whose architecture
infr-llama doesn't implement; `NVFP4` bleeding-edge). Gate for these =
**bit-identical SIMD↔ scalar + tolerance-parity to the exact `dequant_block`
(tight 1e-3 vs the quantized-activation dot)** — strong on the math, no
coherence generation.

| Format    | commit    | notes                             |
| --------- | --------- | --------------------------------- |
| `Q5_1`    | `e7465ed` | affine, Q5_0 5-bit + Q4_1 min     |
| `Q2_0`    | `3f7c79e` | Bonsai ternary, 64-block, 2×Q8x32 |
| `MXFP4`   | `29ee2e5` | IQ4_NL + E8M0 scale (gpt-oss)     |
| `NVFP4`   | `06cc0ef` | MXFP4 codebook + per-16 UE4M3     |
| `IQ2_S`   | `8e616c3` | grid-codebook, expand-row-once    |
| `IQ3_S`   | `34fd4f2` | grid-codebook, per-32 scale       |
| `IQ2_XXS` | `aa07a2f` | grid, KSIGNS lookup, coherence ✓  |
| `IQ3_XXS` | `e97a913` | grid, token-identical to GPU ✓    |
| `IQ2_XS`  | `7e0fb1d` | grid, 9-bit index (unit-test)     |
| `IQ1_S`   | `93e3225` | 1-bit, `dl·(iprod + delta·asum)`  |
| `IQ1_M`   | `c70111f` | 1-bit, d-in-scales, per-8 delta   |

The grid/codebook set (`IQ2_XXS`/`IQ2_XS`/`IQ2_S`, `IQ3_XXS`/`IQ3_S`,
`IQ1_S`/`IQ1_M`) uses the expand-row-once → signed-i8 → per-group int dot
pattern; each mirrors `dequant_codebook` and is gated on a **tight 1e-3**
tolerance vs the exact decode. Coherence: `IQ3_XXS` is token-identical to the
GPU native path; `IQ2_XXS` matches leading tokens; the 1-bit `IQ1_S`/`IQ1_M` are
chaotic at the noise floor (int8/f32/CPU/GPU all diverge — the 1e-3 math check
is the correctness gate).

**CPU now natively handles EVERY weight quant format** — including ternary
`TQ1_0` (`b7a4201`) / `TQ2_0` (`6336df3`), fold `(digit−1)` into signed i8 +
single-scale int dot. Full parity with Vulkan's native set.

What is still open is in `backlog.md` § B101.

## Cross-backend fast-kernel coverage

CPU, Vulkan, and Metal now each have a native fast kernel for **every** weight
quant format infr supports (24/24) — no format falls back to dequant→float on
any backend. The full matrix and per-backend decode strategy moved to its own
doc: **`kernels.md`**. (CPU closed its last gap with ternary `TQ1_0`
`6336df3`/`b7a4201`; TriLM-3.9B TQ2_0 prefill 16.5→131.9 t/s ≈ 8×.)

## Context: two regimes, two different bottlenecks

CPU inference splits hard by batch size, and the cache/bandwidth story is
different in each:

- **Decode (`m == 1`) is DRAM-bandwidth-bound.** A real model's weights are GBs
  (`Q4_K` 9B ≈ 5 GB; even a `Q2_K` 0.6B ≈ 180 MB) — all **≫ L3**. Every weight
  is read once per token, streamed contiguously from RAM. On a sequential stream
  the hardware prefetcher already saturates the memory controllers, so the only
  lever that scales decode is **fewer bytes streamed** (native quantization)
  plus **TLB** relief (hugepages). Software prefetch of weights is a wash here —
  the HW prefetcher already predicts the stream.
- **Prefill (`m > 1`) is compute + cache-reuse-bound.** Weights still stream,
  but each weight row is reused across `m` activation columns, so keeping the
  activation tile resident in L1/L2 is the lever. This is where blocking / tile
  sizing / fusion pay off.

### Reference hardware (dev box)

AMD Ryzen 9 9950X3D (Zen 5, 3D V-Cache): **128 MiB L3**, 16 MiB L2 (1 MB/core),
768 KiB L1d (48 KB/core), 16 cores / 32 threads, **1 NUMA node**. ISA: AVX-512
F/BW/VL/DQ/CD, **AVX-512-VNNI**, AVX-512-BF16, AVX-VNNI, F16C, 3DNow-prefetch.
The big X3D L3 helps prefill (a whole layer's activations + KV stay hot); it
does **not** rescue decode (weights still ≫ 128 MB).

## What is already done (do not redo)

- **Weights mmap'd native.** `Op::Linear` streams the row-major GGUF weight one
  row at a time straight from the mmap — no f32 materialization in RAM.
- **Int8-quantized-activation VNNI dots** for the common k-quants — **Q4_K,
  Q5_K, Q6_K, Q8_0, Q5_0** — with scalar→AVX2→AVX-512BW→VNNI kernels and up to
  **8-row cache-blocking tiles** (activation loaded once, reused across 8 weight
  rows). This is already the "native format, lossy-but-fast, cache- friendly"
  strategy; it is why a Q4_K_M model is already tight.
- **Prefill conv1d parallelized** over the virtual `[state‖x]` sequence
  (`ac9c228`). Bit-identical; isolated kernel ~7.3× but end-to-end flat (conv1d
  is <1% of GEMM-bound prefill).

## The bottleneck ranking (why the list is ordered as it is)

1. **Fewer bytes (native quant coverage)** — dominant decode lever.
2. **Hugepages / madvise on the weight mmap** — real TLB win on the GB stream.
3. **Op fusion** — cuts intermediate DRAM round-trips.
4. **Prefill tile tuning** to the X3D topology — real but measure-first.
5. **Software prefetch** — micro-opt, usually a wash. Not a strategy.

---

## Worklist

The worklist's quant-coverage, madvise and DeltaNet head-parallel slices all
landed (results snapshot above). What it left open — native AVX-512 f16/bf16
dots, prefill tile tuning to the X3D topology, broader op fusion, and the
deferred DeltaNet input-clone elimination — is in `backlog.md` § B101.

**Re-bless discipline** for any CPU slice that changes precision: a golden only
gets re-blessed after the new output is _verified correct/coherent_ — never
blind-accept a diff. For a precision flip that means confirming the model still
generates sane, coherent text (compare a short generation against the f32/GPU
path) AND that the CPU result matches the GPU int8 result within tolerance. A
golden diff that changes _which_ tokens are produced in a way that looks like
garbage is a bug, not a precision flip.

## Measurement (prerequisite for prefill tile tuning, useful throughout)

"Should help in theory" gets verified with counters, not intuition. `perf` is
**not installed** on the dev box; installing it (or an equivalent that reads
`LLC-load-misses`, `dTLB-load-misses`, `stalled-cycles-backend`) lets us
classify each stall as DRAM-bound (→ only fewer bytes helps), TLB-bound (→
madvise / hugepages), or cache-miss slack (→ prefill tile tuning). For hotspot
attribution use `samply` (never ad-hoc timers); for A/B throughput use
`infr bench --dev cpu` / `infr compare`.

## Software prefetch — explicitly deprioritized

Explicit `_mm_prefetch` of weights is a micro-opt, not a strategy: the HW
prefetcher already predicts the sequential weight stream, and a mistuned
prefetch distance evicts useful lines. Only revisit if `perf` shows
latency-bound (not bandwidth-bound) stalls on an _irregular_ access pattern.
