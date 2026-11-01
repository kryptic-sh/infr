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
| `kq_fma` prefill GEMM | An fp32 FMA GEMM for Q4_K/Q6_K (`native_gemm_kq_fma.comp`) replaces the dp4a MMQ kernels, which are emulated in ALU on GCN. |
| `kq_fma` decode GEMV | An fp32 Q4_K/Q6_K GEMV on f32 activations (`native_gemv_kq_fma.comp`) replaces `quant_q8` + the dp4a GEMV. |
| Float decode GEMVs | F32 weights use a multi-output GEMV (`linear_f32_mo.comp`); BF16 weights use the dedicated reduction GEMV. |

All of these switch on automatically when the device does not report int8 dot
as hardware-accelerated (`Capabilities::i8_dot_fast`), and are off on
GPUs that have it.

## Switches

| Variable | Config key | Effect |
|---|---|---|
| `INFR_KQ_FMA_TIER=0\|1` | `kernels.vulkan.kq_fma_tier` | Force the `kq_fma` prefill GEMM off/on. |
| `INFR_KQ_FMA_GEMV=0\|1` | `kernels.vulkan.kq_fma_gemv` | Force the `kq_fma` decode GEMV off/on. |
| `INFR_WAVE64_NATIVE=0\|1` | `kernels.vulkan.wave64_native` | Run the subgroup decode GEMV at its native 64-lane width instead of emulated 32-lane halves (default on; inert off wave64 hardware). |
| `INFR_SUBMIT_DISPATCHES=N` | `device.submit_dispatches` | Dispatches per submit (`0` = never split). |

## Serving tips (8 GB cards)

- Pass `--ctx` (e.g. `--ctx 8192`). At the default 131k context, placement
  falls back to a 256-row prefill chunk, which slows prefill and breaks
  KV-prefix reuse between chat turns on Gemma's sliding-window layers.
- A larger `--ubatch` (e.g. `1024`) gives more room for prefix reuse after
  long replies, if it fits.

## Measured (RX 580 8 GB, RADV)

Q4_K_M, tokens/s. llama.cpp numbers are its Vulkan backend on the same card.

| Model | pp512 before | pp512 now | llama.cpp pp512 | tg128 now | llama.cpp tg128 |
|---|---|---|---|---|---|
| Qwen2.5-Coder-3B | 206 | 499 | 421 | 63.6 | 63.8 |
| Gemma 4 E2B | 285 | 616 | 602 | — | 65.7 |

Gemma 4 E4B decode at 4k context: 29.6 → 36.5 t/s.

## Status and plans

Tracked on the fork's project board. Next up: Q4_0 support in the `kq_fma`
kernels (Gemma QAT quants are Q4_0), single-pass sliding-window attention for
decode, KV shifting for rolling chat windows, and speculative decoding on
Vulkan.
