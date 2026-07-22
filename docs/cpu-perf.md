# cpu-perf.md — CPU backend performance roadmap

Findings and a prioritized worklist for the `infr-cpu` reference backend,
aggregated from the CPU perf review. Ordered **low → high implementation
difficulty** so we land the cheap, high-certainty wins first.

## Context: two regimes, two different bottlenecks

CPU inference splits hard by batch size, and the cache/bandwidth story is
different in each:

- **Decode (`m == 1`) is DRAM-bandwidth-bound.** A real model's weights are GBs
  (Q4*K 9B ≈ 5 GB; even a Q2_K 0.6B ≈ 180 MB) — all **≫ L3**. Every weight is
  read \_exactly once per token*, streamed contiguously from RAM. On a
  sequential stream the hardware prefetcher already saturates the memory
  controllers, so the only lever that scales decode is **fewer bytes streamed**
  (native quantization) plus **TLB** relief (hugepages). Software prefetch of
  weights is a wash here — the HW prefetcher already predicts the stream.
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

## Worklist (low → high difficulty)

Each slice: TDD, bit-identity where the math is unchanged (parallelization,
fusion of exact ops), tolerance-parity + a sanctioned golden re-bless where the
math changes (int8 activation quant is lossy). One slice at a time; validate
correctness before benching.

### 1. Weight mmap `madvise` + THP hint — _easy_

- **What:** on the weight mmap, advise the kernel: `MADV_HUGEPAGE` (2 MB pages
  cut dTLB page-walks on the multi-GB sequential read), `MADV_SEQUENTIAL` /
  `MADV_WILLNEED` (bias readahead the way we actually consume).
- **Why:** a >L3 sequential mmap read at 4 KB pages hammers the dTLB; the TLB is
  the one "prediction" structure with headroom in the decode stream. Hugepages
  is the closest thing to "help the CPU preload the next region" that survives
  the bandwidth reality.
- **Impact:** small–moderate decode + weight-load win; low risk.
- **Precision:** none (pure memory hint). Bit-identical.
- **Status:** TODO

### 2. DeltaNet head-parallelism — _easy–medium_

- **What:** `Op::DeltaNet` runs a serial single-thread scan. The outer `for t`
  is inherently sequential (state carries across tokens), but the inner
  `for h in 0..n_vhead` loop over value heads is **fully independent** — each
  head owns a disjoint `state[h*kd*vd..]` slice, its own out slice, and reads
  only shared inputs. Parallelize over heads: each head task runs its whole
  `t`-scan on its own state copy (`pool.collect`), then write state + out back.
- **Why:** DeltaNet is the linear-attention path for **~75% of Qwen3.5 layers**
  (full attention only every 4th) — a major CPU cost, unlike conv1d. 16 heads
  (9B) → up to 16-way parallelism on the dominant attention op.
- **Impact:** real prefill **and** decode win expected.
- **Precision:** bit-identical (same per-head float order; state rebuild is a
  copy).
- **Status:** TODO

### 3. DeltaNet input-clone elimination — _medium_

- **What:** the DeltaNet arm `.clone()`s the whole `q/k/v` buffers
  (`[rows, heads·dim]`) every op purely to dodge the borrow checker (state needs
  `&mut vals` while q/k/v need `&vals`). Introduce a disjoint-`vals` accessor
  (split one `&mut` index out, borrow the rest `&`) to drop the clones. The same
  pattern recurs in other ops (conv1d clones too), so the accessor is reusable.
- **Why:** at prefill those clones are ~1M floats × 3 per DeltaNet layer of pure
  allocation + copy traffic.
- **Impact:** moderate prefill win; removes allocator pressure.
- **Precision:** bit-identical.
- **Status:** TODO

### 4. Native int8 dot: **Q4_0** — _medium_

- **What:** Q4*0 currently falls to `bytes_to_f32` dequant + f32 dot (the slow
  `*
  =>`fallback). Add native int8-activation kernels (scalar/AVX2/AVX-512BW/VNNI + batch/batch8) and wire into both the`m==1`and`m>1`
  dispatch, mirroring Q8_0 and the GPU's native Q4_0 kernel.
- **Why:** Q4_0 is ubiquitous; the GPU already has a native kernel. First and
  simplest of the uncovered formats.
- **Impact:** large on Q4_0 models (decode + prefill); kills the f32 fan-out.
- **Precision:** int8 activation quant is lossy → this **changes the CPU
  reference output for Q4_0**. Tolerance-parity test vs the f32 reference to
  bound error; the Q4_0 gpu_seam golden is a sanctioned **precision-flip
  re-bless** (`--include-ignored`), and the new CPU path should match the GPU
  int8 result, not the old f32.
- **Status:** TODO

### 5. Native int8 dot: **IQ4_XS** — _medium_

- Same treatment as Q4_0, for the common small-model format IQ4_XS (local
  Qwen3-0.6B has one). GPU reference exists (quant-cliff-warp).
- **Precision:** precision-flip re-bless as in #4.
- **Status:** TODO

### 6. Native int8 dot: **Q2_K, Q3_K** — _medium–high_

- K-quant super-block formats with packed scales; more decode work than Q4_0 but
  same int8-activation regime. One slice each.
- **Precision:** precision-flip re-bless per dtype.
- **Status:** TODO

### 7. Native int8 dot: **IQ2/IQ3 family** (IQ4*NL, IQ2_XXS/XS/S, IQ3_XXS/S) — \_high (volume)*

- The remaining uncovered formats; codebook/grid decode is fiddlier. Land as a
  mini-campaign, one dtype per slice, only after #4–#6 prove the pattern.
- **Precision:** precision-flip re-bless per dtype.
- **Status:** TODO

### 8. f16 / bf16 native AVX-512-FP16/BF16 dot — _medium_

- **What:** f16/bf16 weights already read native 2-byte (bandwidth already
  minimal), but the dot accumulates in f32 after widening. Add a native
  AVX-512-FP16 / AVX-512-BF16 dot to cut the arithmetic.
- **Why:** compute-only win; the bandwidth is already optimal, so this is
  smaller than the quant slices — do it after the quant gap is closed.
- **Impact:** modest, prefill-leaning.
- **Precision:** changes accumulation precision → tolerance-parity + re-bless if
  the f16/bf16 goldens move.
- **Status:** TODO

### 9. Prefill tile-size tuning to the X3D topology — _medium–high (measure-first)_

- **What:** tune the prefill GEMM tile (rows × `m` block) so the activation tile
  stays resident in L1/L2 (48 KB / 1 MB) while weights stream; exploit the 128
  MB L3 for layer-resident activations + KV.
- **Why:** prefill is the cache-reuse regime; current tiling (8-row) is a fixed
  heuristic, not topology-aware.
- **Impact:** prefill win, hardware-dependent.
- **Precision:** bit-identical (scheduling/tiling only).
- **Gate:** needs `perf stat` (LLC / dTLB / backend-stall) to confirm we have
  cache-miss slack before investing. **`perf` is not installed on the dev box.**
- **Status:** TODO (blocked on measurement)

### 10. Op fusion (RMSNorm→Linear, gate/up, residual-add) — _high (structural)_

- **What:** fuse adjacent ops in the Graph/IR so intermediate activation vectors
  never round-trip to DRAM (stay in L1/registers). Some fusion exists
  (`GatedActFused`, `RmsNormAdd`); extend to norm→linear and residual chains.
- **Why:** the real "keep it in cache" lever in both regimes — cuts memory
  traffic, which is what helps when bandwidth-bound.
- **Impact:** moderate–large, broad.
- **Precision:** bit-identical if the fused ops compute the same values in the
  same order; verify per fusion.
- **Status:** TODO

---

## Measurement (prerequisite for #9, useful throughout)

"Should help in theory" gets verified with counters, not intuition. `perf` is
**not installed** on the dev box; installing it (or an equivalent that reads
`LLC-load-misses`, `dTLB-load-misses`, `stalled-cycles-backend`) lets us
classify each stall as DRAM-bound (→ only fewer bytes helps), TLB-bound (→ #1),
or cache-miss slack (→ #9). For hotspot attribution use `samply` (never ad-hoc
timers); for A/B throughput use `infr bench --dev cpu` / `infr compare`.

## Software prefetch — explicitly deprioritized

Explicit `_mm_prefetch` of weights is a micro-opt, not a strategy: the HW
prefetcher already predicts the sequential weight stream, and a mistuned
prefetch distance evicts useful lines. Only revisit if `perf` shows
latency-bound (not bandwidth-bound) stalls on an _irregular_ access pattern.
