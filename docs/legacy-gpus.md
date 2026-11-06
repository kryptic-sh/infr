# Older and lower-end Vulkan GPUs

This fork ([KessokuNet/infr](https://github.com/KessokuNet/infr)) optimizes and
backports infr to Vulkan GPUs that upstream does not target: no cooperative
matrix, no hardware-accelerated int8 dot product, and wave64-only subgroups.
The baseline is **AMD GCN4 / Polaris** (RX 470/480/570/580), a common cheap
used GPU. Changes that are not specific to these GPUs are meant to go upstream
to [kryptic-sh/infr](https://github.com/kryptic-sh/infr).

## What is different on these GPUs

| Change | What it does |
|---|---|
| Wave64 support | Wave32 kernels run as 32-lane halves of a 64-lane wave (`shaders/wave64.glsl`, spec constant `W64`), so infr starts at all on wave64-only devices. |
| First-forward submit split | Slow discrete GPUs split even the first prefill into submits of 64 dispatches, so a long prompt can't trip the driver's hang watchdog. |
| `kq_fma` prefill GEMM | An fp32 FMA GEMM for Q4_K/Q6_K/Q4_0 (`native_gemm_kq_fma.comp`) replaces the dp4a MMQ kernels, which are emulated in ALU on GCN. |
| `kq_fma` decode GEMV | An fp32 Q4_K/Q6_K GEMV on f32 activations (`native_gemv_kq_fma.comp`) replaces `quant_q8` + the dp4a GEMV. |
| `kq_fma` verify GEMV | An fp32 multi-row Q4_K/Q6_K GEMV (`native_gemv_kq_fma_mrow.comp`) for MTP-verify batches — MR=4 rows share each superblock's weight-word load, bit-identical row-for-row to the decode GEMV. Replaces the int8 `mrow` on verify batches, whose quantized activations flip near-tie greedy tokens vs decode. |
| Float decode GEMVs | F32 weights use a multi-output GEMV (`linear_f32_mo.comp`); BF16 weights use the dedicated reduction GEMV. |

All of these switch on automatically when the device does not report int8 dot
as hardware-accelerated (`Capabilities::i8_dot_fast`), and are off on
GPUs that have it.

## Switches

| Variable | Config key | Effect |
|---|---|---|
| `INFR_KQ_FMA_TIER=0\|1` | `kernels.vulkan.kq_fma_tier` | Force the `kq_fma` prefill GEMM off/on. |
| `INFR_KQ_FMA_GEMV=0\|1` | `kernels.vulkan.kq_fma_gemv` | Force the `kq_fma` decode GEMV off/on. |
| `INFR_KQ_FMA_SPLITK=N` | `kernels.vulkan.kq_fma_splitk` | Force the `kq_fma` prefill GEMM's split-k count (1 = direct tile); unset picks it from the grid size. |
| `INFR_WAVE64_NATIVE=0\|1` | `kernels.vulkan.wave64_native` | Run the subgroup decode GEMV at its native 64-lane width instead of emulated 32-lane halves (default on; inert off wave64 hardware). |
| `INFR_WEIGHTS_AS_SSBO=0\|1` | `kernels.vulkan.weights_as_ssbo` | Bind `kq_fma` decode-GEMV weights as storage buffers instead of reading them by 64-bit device address (default off). May help GPUs without a scalar-base global load, e.g. GFX8/Polaris, where each device-address load builds its own 64-bit address. |
| `INFR_SUBMIT_DISPATCHES=N` | `device.submit_dispatches` | Dispatches per submit (`0` = never split). |
| `INFR_GFX_QUEUE=1` | `kernels.vulkan.gfx_queue` | Submit on the universal graphics family, the ring the compositor renders on. Off by default: a discrete GPU uses a compute-only family so prefill does not hold up the desktop. |

## Serving tips (8 GB cards)

- Pass `--ctx` (e.g. `--ctx 8192`). At the default 131k context, placement
  falls back to a 256-row prefill chunk, which slows prefill and breaks
  KV-prefix reuse between chat turns on Gemma's sliding-window layers.
- A larger `--ubatch` (e.g. `1024`) gives more room for prefix reuse after
  long replies, if it fits.

## Measured (RX 580 8 GB, RADV)

`infr compare --sweep` against llama.cpp `785322d` (its Vulkan backend), same GGUF,
same card, re-run back to back on 2026-11-06. Tokens/s, `-r 3`, F16 KV; the existing
power cap and undervolt were left unchanged. The server was stopped during all GPU
work. Ratios use the tool's unrounded averages.

| Model | metric | infr | llama.cpp | ratio |
|---|---|---|---|---|
| Qwen2.5-Coder-3B Q4_K_M | pp512 | 540 | 478 | 1.13x |
| | tg128 | 81 | 71 | 1.13x |
| | tg64 @ d4096 | 64 | 62 | 1.02x |
| | pp4 @ d4096 | 87 | 136 | **0.64x** |
| Gemma 4 E4B QAT UD-Q4_K_XL | pp512 | 380 | 364 | 1.04x |
| | tg128 | 48 | 48 | 0.99x |
| | tg64 @ d4096 | 44 | 44 | 1.02x |
| | pp4 @ d4096 | 67–69 | 114 | **0.58–0.60x** |

The Gemma range is two separate three-repetition run averages, not a confidence
interval: the first measured 67, the repeat 69. A repeat using general attention
measured 68 against 115. Its small-turn throughput is effectively unchanged.

### Four-token turns at 4k depth

This path had two routing gaps. Ordinary small prefill could use scalar-dequant
multi-row matmuls even though the fp32 K-quant multi-row kernel existed for MTP
verification. It now uses that kernel where the scalar fallback applied, while
keeping the eligible int8 prefill tier. Fixed-head split-K attention was restricted
to one query row; static batches of two to eight rows now use the same kernel
family with separate query offsets, causal ends and sliding-window floors. Replay
and one-row selection stay unchanged.

A controlled comparison disabling both paths with
`--set kernels.vulkan.kq_fma_gemv=false --set kernels.vulkan.attn_decode=false`
measured Qwen's `pp4@d4096` at 71 against 136. New defaults measured 87 against
136: about 22% more tokens/s, or 56 → 46 ms for the four-token suffix. Those switches
also affect decode, so their decode results are not an old-default baseline.

One controlled Qwen device profile measured scalar small projections at
8.4 → 4.0 ms, attention at 13.4 → 10.0 ms, and total GPU time at 46.4 → 38.0 ms.
Build and compile time were below the printed 0.1 ms precision. Large up/down
matmuls still account for about 54% of the new GPU time; attention another 26%.
Emulated int8 is not universally slower: direct microbenchmarks still favour it
on several eligible shapes, which is why that tier remains enabled.

The service's Gemma file stores all 168 Q/O/up/down projection tensors as Q4_0.
The new fp32 multi-row path supports Q4_K/Q6_K, not Q4_0. Its device profile still
spends about 45% on the large up/down matmuls and 22% on two scalar-dequant
projection shapes; attention is about 11%. Improving Q4_0 small-row matmuls is a
separate remaining task, not a page-table fix.

`pp4@d4096` remains behind llama.cpp. These are benchmark compute times, not
end-to-end TTFT: both exclude depth preparation, but infr omits its output head
and llama.cpp evaluates last-row logits and crops the last-layer FFN. Matched
models/settings therefore do not imply identical total work or an engine-wide
ranking. Bench uses one session, so the multi-slot page pool and host prefix store
are not providing a prefix-reuse speedup here.

`compare` now forwards explicit `--config` and ordered `--set` values to its infr
subprocesses. Earlier attempts to toggle these paths through `compare` silently
lost those arguments; only the default-run results from those attempts were valid.

Earlier campaign (same card, before the `kq_fma` tier and the seam work; no date was
recorded at the time, so read the llama.cpp columns as historical):

| Model | pp512 before | pp512 after | llama.cpp pp512 | tg128 after | llama.cpp tg128 |
|---|---|---|---|---|---|
| Qwen2.5-Coder-3B | 206 | 499 | 421 | 63.6 | 63.8 |
| Gemma 4 E2B | 285 | 616 | 602 | — | 65.7 |
| Gemma 4 E4B QAT Q4_0 | 146 | 340 | 343 | 37.2 | 42.6 |

Gemma 4 E4B decode at 4k context went 29.6 → 44.4 t/s across both campaigns.

MTP-verify shapes (RX 580, effective weight-stream GB/s, m = drafts+1): the verify
kernel ties the int8 `mrow` it replaces on big weights (down-proj Q4_K m=7: 39 vs 49)
and beats it on small ones (lm_head m=2: 89 vs 77), while crushing the f32-dequant
`mrow` that serves small weights (2–3.5×). The property that matters is exactness:
verify-flagged graphs score rel ~2e-7 vs the host oracle (pure f32 reassociation
noise), where the int8 path scores ~4e-3 — the noise that flipped greedy tokens
and collapsed draft acceptance.

## Status and plans

Tracked on the fork's project board. Next up: single-pass sliding-window attention for
decode, KV shifting for rolling chat windows, and speculative decoding on
Vulkan.
