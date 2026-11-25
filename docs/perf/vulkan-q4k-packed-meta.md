# Vulkan Q4_K decode: packed header loads in the four-output GEMV

Single-token dense decode on the integer-dot path is `quant_q8` + `native_mmv_mrow` with `rows=1`.
Q4_K's 16-byte header holds two half scales and twelve packed scale/min bytes, and the scalar path
reads it with eight separate byte and half loads per 32-element sub-block: `d`, `dmin`, and then
three words re-read a byte at a time by `k4()`.

## Change

The 144-byte super-block stride keeps every header 16-byte aligned, so one `NW4` (a
`global_load_b128` with a scalar base on RADV) replaces all eight loads. The six-bit scale and min
fields are then extracted in registers with the same masks `k4()` applies, including the shared-byte
cases for sub-blocks 4..7.

Only the metadata load changes. The dp4a dots, the row and K lane assignment, the shared-memory
reduction and the write-out are untouched, so the output is bit-identical.

`-DQ4K_PACKED_META` is applied to Q4_K's four-output (`-DOUTS4`) builds only: `native_mmv_mrow_q4k_o4`,
`_o4_m4` and `_o4_m4_res`. Where that layout applies — a discrete GPU below `mmv_mrow.o4_max_in_f`
(2048 by default), an integrated GPU at every width through `o4_all_widths` — `mmv_q4k_meta = true`
(`INFR_MMV_Q4K_META=1`) selects the packed build instead of the scalar one.

It is **off by default**. The kernel is 7-10% faster at the shapes it reaches, but the decode GEMV is
bandwidth-bound on the discrete cards it has run on, so the only end-to-end gain measured is on an
integrated GPU; turn it on per device once that is confirmed for the device in question.

## Measurements

RX 580 (RADV POLARIS10, Mesa 26.2.7, `mesa-vulkan-drivers-26.2.7-1.fc44`), `--config /dev/null`,
Vulkan0. Driver string: `radv` / `Mesa 26.2.7`, apiVersion 1.4.354.

Prequantized microbench (`packed_metadata_bw`: record + submit + wait over 64 operations across four
rotating weight banks, so this is not a cold-cache number), means of the two interleaved runs:

| shape            | scalar metadata |      packed | delta |
| ---------------- | --------------: | ----------: | ----: |
| Q4_K 1536 × 2048 |      79.35 GB/s |  84.80 GB/s | +6.9% |
| Q4_K 1536 × 1024 |      49.10 GB/s |  53.95 GB/s | +9.9% |
| Q4_K 1024 × 8192 |     120.05 GB/s | 123.45 GB/s | +2.8% |

Command:

```sh
INFR_DEV=Vulkan0 cargo test -p infr-vulkan --release --test mmv_q4k_meta \
  packed_metadata_bw -- --ignored --exact --nocapture
```

### End to end on this card

The RX 580 has an accelerated signed int8 dot, but its decode takes `native_gemv_kq_fma` (the fp32
emulated-dot tier), so the int8 GEMV is not on the critical path by default and the knob changes
nothing there. Forcing the tier the change acts on is what actually compares the two kernels:

```sh
E="INFR_KQ_FMA_GEMV=0 INFR_KQ_FMA_TIER=0"
env $E INFR_MMV_Q4K_META=1 infr --config /dev/null bench "$M" -p 0 -n 128 -r 3 --dev Vulkan0
env $E infr --config /dev/null bench "$M" -p 0 -n 128 -r 3 --dev Vulkan0
# and the same pair with -d 4096
```

Both legs alternated (`on`, `off`, `off`, `on`) with 20s cooldowns and 3 repetitions per run, means
of the two runs:

| model, phase                           |    packed |    scalar | delta |
| -------------------------------------- | --------: | --------: | ----: |
| Qwen3-0.6B Q4_K_M, tg128               | 197.8 t/s | 197.8 t/s |  0.0% |
| Qwen3-0.6B Q4_K_M, tg128 @ d4096       | 121.4 t/s | 121.4 t/s |  0.0% |
| Qwen2.5-Coder-3B Q4_K_M, tg128         |  63.8 t/s |  63.8 t/s |  0.0% |
| Qwen3-0.6B Q4_K_M, tg128, default tier | 249.7 t/s | 249.6 t/s |  0.0% |

Per-run spreads were 0.5% or less. Identical results are the expected outcome for a bit-identical
kernel on a bandwidth-bound device: the header bytes are read either way. The last row is the
default decode tier, where the int8 GEMV is not reached at all.

That the packed kernel is what runs with the knob on, rather than a silent fallback, is visible in
the decode profile: `native_mmv_mrow_q4k_o4_m4_packed_meta`, 1792 dispatches at 20.5us, with
`INFR_PROF_OPS=1`.

### The integrated GPU where it was found

Measured earlier in this project's campaign, on the device where the header unpacking was on the
critical path — AMD Ryzen 7 7800X3D integrated GPU, RADV RAPHAEL_MENDOCINO, Mesa 25.2.7,
`unsloth/gemma-4-E4B-it-GGUF` `gemma-4-E4B-it-Q4_K_M.gguf` (4,977,171,584 bytes), decode 16 tokens at
depth 16, context 128, three repetitions per process in `scalar / packed / packed / scalar` order,
serial headless workloads, `INFR_NO_COOPMAT=1 INFR_KQ_FMA_GEMV=0 INFR_KQ_FMA_TIER=0`, an empty
config file: 6.14 and 5.86 tok/s on the scalar metadata (mean 6.00) against 7.66 and 7.31 on the
packed (mean 7.49), about 24.8% higher. That GPU is not available here, so the figure is quoted
rather than reproduced, and it covers one model at one depth.

## Correctness gates

`tests/mmv_q4k_meta.rs::packed_metadata_matches_scalar_exact` compares scalar and packed output bits
for rows 1, 2, 3, 4, 5, 8, 9 and 16, both row caps, with and without the fused residual, over output
tails (1, 3, 5, 66) and weight bases 0/256/512/768. The two-output leg covers `in_f` 256, 1536, 2048,
2560, 2816, 6144 and 11008; the four-output leg stays under 2048 so that a discrete device takes it
through `o4_max_in_f` and an integrated one through `o4_all_widths`, which keeps the expectation
device-independent. The weight bytes cover zero, signed zero, subnormal and signed half scales and
both edges of the packed scale/min fields.

Because `built_kernel_names` accumulates over a backend's life, the kernel-name check compares the
names each run _added_ to the names it started with: a run may add nothing (the variant is already
built) or exactly the kernel the naming scheme predicts, and the first case that needs a variant has
to build it. A row that silently fell back to the scalar build fails there even though the earlier
kernel is still in the list.
