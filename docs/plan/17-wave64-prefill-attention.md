# 17 — Wave64 prefill attention for GCN

## Problem

On the RX 580 (RADV Polaris, wave64, no cooperative matrix) every prefill
attention row runs through `attn_nc_fa.comp`, the shared-memory fallback
written for Intel Arc. At depth it is two-thirds of prefill: LFM3-1.7B Q4_K_M at
58,728 tokens spent 89.1 s of 132.3 s in it (B237).

After the PV unroll cap (80 VGPRs, no spills, two waves per SIMD) the kernel
runs about 1.35 TFLOP/s against a 6.2 TFLOP/s FP32 peak, and pp58728 is
539 t/s. The prefill GEMM (`native_gemm_kq_fma`, 128x128 tile, `As[k][m/4]`
layout, 8x8 microtile) runs near 4.4 TFLOP/s on the same card, so the gap is in
the attention kernel, not the hardware.

What still holds the kernel back, from RADV shader stats and the source:

- LDS caps occupancy: 32 KB per 256-thread workgroup, so two workgroups per CU
  (8 of 40 wave slots).
- About 3 B of LDS traffic per FMA. QK reads 6 vec4 per 32 FMAs; PV reads two
  scalars and one vec4 per 8 FMAs.
- 15 barriers per 64-key block at hd128 (27 at hd256): 8 for KC=32 chunking,
  1 to publish scores, 2 for the shared-memory softmax, 4 for V chunks.
- A 4-way bank conflict on the QK K reads (`QP4 = 9` vec4 pitch) and an 8-way
  conflict on the softmax reads (`SP = 68`).
- Q is restaged from global for every 64-key block and every KC chunk.
- The grid runs query tiles in ascending order, so the longest tiles start last.

## Goal

Prefill attention near GEMM efficiency on wave64 GCN: at least 2.5 TFLOP/s per
full-attention layer at depth, which puts pp58728 near 750 t/s, with 3.5 TFLOP/s
(about 860 t/s) as the stretch. 1,000 t/s at 58k also needs the GEMM near 80% of
peak and is the ceiling, not the target.

## Phases

- **P0 — Register spill (done in the slice before this plan).** Cap the PV key
  unroll at 8; 256 VGPRs with 62 spilled became 80 with none.
- **P1 — Cheap fixes inside `attn_nc_fa`.** Each one measured alone with
  `INFR_PROF_OPS=1` at pp1024@d16384 and pp1024@d32768:
  - Wave64 softmax: the 8 threads of a softmax row are 8 consecutive lanes, so
    the row max and sum become clustered subgroup reductions on a pinned
    64-lane build (`kernel_sg64_native`). Drops `redm`/`reds` (2 KB) and two
    barriers per block. The non-wave64 build keeps the shared-memory path.
  - Bank-conflict-free pitches for the K chunk and the score rows.
  - Reverse the query-tile order so the longest tiles start first.
  These keep the kernel's summation order, so `attention_prefill_nc_fa_*`
  stays exact against its CPU reference.
- **P2 — `attn_fa_w64`, a register-tiled flash kernel.** A new shader for
  wave64 devices without cooperative matrix, gated in the adapter ahead of
  `nc_fa_ok`, hd128 and hd256 builds (LFM3 and Qwen3.5's full-attention layers).
  Starting geometry, from the GEMM that already reaches 4.4 TFLOP/s here:
  - 256 threads (4 waves), BM=64 query rows, BN=64 keys per block.
  - QK: Q and K staged d-major (`Qs[d][row]`, `Ks[d][key]`) in KC-dim chunks,
    so one `ds_read_b128` gives four rows at one d. 4x4 score microtile per
    thread.
  - Softmax on the score registers: row max and sum across the threads that
    share a row with subgroup shuffles, never through shared memory.
  - P written key-major (`Ps[key][row]`, f32) once per block; PV with a 4x8
    output microtile, V staged in 64-dim chunks.
  - LDS target at most 32 KB per workgroup (two per CU); VGPRs at most 128.
  Correctness: a CPU-reference test on the existing `run_attn_nc_fa` shapes
  (causal, sliding window, ring cache, GQA, pad rows, hd128 and hd256), with a
  tolerance, because the summation order changes.
- **P3 — Tune on the RX 580.** BM 64 vs 128, microtile 4x4 vs 8x4, KC 16/32/64,
  f16 V in LDS, and packing the query heads that share a KV head into one tile
  (one K/V staging for four heads on LFM3). Add the winners as RADV/GCN rows in
  `tuning/table.toml` with their measurements, since Polaris has none (B231).

## Measurement

- Per kernel: `INFR_PROF_OPS=1 infr bench <LFM3 Q4_K_M> -p 1024 -n 0 -d D
  --ctx ... -u 1024` with `INFR_KV_TYPE_K/V=q8_0`, D in {0, 16384, 32768}. The
  per-layer cost per 1,024 keys gives TFLOP/s (8.59 GFLOP per layer per 1,024
  keys at nh=16, hd=128).
- Per build: `RADV_DEBUG=shaderstats INFR_NO_PIPELINE_CACHE=1` must show no
  spilled VGPRs and the intended waves per SIMD. Record both in the commit
  message's reply.
- End to end: pp58728 against 539 t/s; llama.cpp on the same card does
  pp1024@d32768 at 456 t/s.

## P1 result (RX 580, LFM3 Q4_K_M, q8 KV)

- Paired `INFR_WAVE64_NATIVE` A/B at pp1024, three pairs per depth: the twin
  did 773/700/697 t/s against the base build's 678/647/652 at d16384 (+7 to
  +14%), and 551/487/467 against 459/520/484 at d32768 (neutral; run variance
  reaches 20%, so only adjacent pairs compare).
- The K/V vec2 split removed the 4-way LDS bank conflict: attention at d16384
  fell from 606 ms (origin/main) to 552 ms median for the base build.
- The twin needs `subgroupShuffle`, which the backend does not require, so the
  `_sg64` build is chosen only when the device reports the shuffle capability;
  everywhere else the base build runs.

## Risks

- A wave64-only kernel needs `subgroup_size_control` with a pinned size of 64;
  the gate must check it, not assume it.
- The tile's summation order differs from `nc_fa`, so `nc_fa`'s bit-identical
  cut test does not transfer; the new kernel needs its own split-grid test.
- hd256 doubles the QK chunk count and the O microtile; it may need BM=32 to
  stay under 128 VGPRs.
- RADV and AMDVLK compile the same SPIR-V differently. Tune on RADV, which is
  what kikuri runs, and record the Mesa version with each table row.
