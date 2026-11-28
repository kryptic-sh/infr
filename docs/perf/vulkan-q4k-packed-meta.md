# Vulkan Q4_K decode: packed header loads in the four-output GEMV

Single-token dense decode on the integer-dot path is `quant_q8` +
`native_mmv_mrow` with `rows=1`. Q4_K's 16-byte header holds two half scales and
twelve packed scale/min bytes, and the scalar path reads it with eight separate
byte and half loads per 32-element sub-block: `d`, `dmin`, and then three words
re-read a byte at a time by `k4()`.

## Change

The 144-byte super-block stride keeps every header 16-byte aligned, so one `NW4`
(a `global_load_b128` with a scalar base on RADV) replaces all eight loads. The
six-bit scale and min fields are then extracted in registers with the same masks
`k4()` applies, including the shared-byte cases for sub-blocks 4..7.

Only the metadata load changes. The dp4a dots, the row and K lane assignment,
the shared-memory reduction and the write-out are untouched, so the output is
bit-identical.

`-DQ4K_PACKED_META` is applied to Q4_K's four-output (`-DOUTS4`) builds only:
`native_mmv_mrow_q4k_o4`, `_o4_m4` and `_o4_m4_res`. Where that layout applies —
a discrete GPU below `mmv_mrow.o4_max_in_f` (2048 by default), an integrated GPU
at every width through `o4_all_widths` — `mmv_q4k_meta = true`
(`INFR_MMV_Q4K_META=1`) selects the packed build instead of the scalar one.

It is **off by default**. On the prequantized microbench the packed build is
+2.8% to +9.9% on an RX 580 under RADV and +9.3% to +12.4% on a Ryzen integrated
GPU under the AMD Windows driver, and it is within the run-to-run spread on an
RX 7900 XTX. End to end, decode on the integrated GPU is 5.7% to 20.7% faster
with it, while the discrete cards measured gain at most 0.4%: their decode GEMV
is bandwidth-bound, and a discrete GPU takes the four-output layout only below
`o4_max_in_f`, so a model 2048 wide or wider (Qwen3-1.7B) never reaches the
packed build there. The figures are below.

## Measurements

RX 580 (RADV POLARIS10, Mesa 26.2.7, `mesa-vulkan-drivers-26.2.7-1.fc44`),
`--config /dev/null`, Vulkan0. Driver string: `radv` / `Mesa 26.2.7`, apiVersion
1.4.354.

Prequantized microbench (`packed_metadata_bw`: record + submit + wait over 64
operations across four rotating weight banks, so this is not a cold-cache
number), means of the two interleaved runs:

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

The RX 580's driver does not report the packed signed int8 dot as accelerated
(GCN under RADV emulates it in ALU; `dot_accelerated = false` in the
`rx580-gcn-radv` tuning fixture), so the `gemv.kq_fma` tuning parameter defaults
to true there and its decode takes `native_gemv_kq_fma`, the fp32 GEMV written
for devices without an accelerated int8 dot. The int8 GEMV is not on the
critical path by default and the knob changes nothing there. Forcing the tier
the change acts on is what actually compares the two kernels:

```sh
E="INFR_KQ_FMA_GEMV=0 INFR_KQ_FMA_TIER=0"
env $E INFR_MMV_Q4K_META=1 infr --config /dev/null bench "$M" -p 0 -n 128 -r 3 --dev Vulkan0
env $E infr --config /dev/null bench "$M" -p 0 -n 128 -r 3 --dev Vulkan0
# and the same pair with -d 4096
```

Both legs alternated (`on`, `off`, `off`, `on`) with 20s cooldowns and 3
repetitions per run, means of the two runs:

| model, phase                           |    packed |    scalar | delta |
| -------------------------------------- | --------: | --------: | ----: |
| Qwen3-0.6B Q4_K_M, tg128               | 197.8 t/s | 197.8 t/s |  0.0% |
| Qwen3-0.6B Q4_K_M, tg128 @ d4096       | 121.4 t/s | 121.4 t/s |  0.0% |
| Qwen2.5-Coder-3B Q4_K_M, tg128         |  63.8 t/s |  63.8 t/s |  0.0% |
| Qwen3-0.6B Q4_K_M, tg128, default tier | 249.7 t/s | 249.6 t/s |  0.0% |

Per-run spreads were 0.5% or less. Identical results are the expected outcome
for a bit-identical kernel on a bandwidth-bound device: the header bytes are
read either way. The last row is the default decode tier, where the int8 GEMV is
not reached at all.

That the packed kernel is what runs with the knob on, rather than a silent
fallback, is visible in the decode profile:
`native_mmv_mrow_q4k_o4_m4_packed_meta`, 1792 dispatches at 20.5us, with
`INFR_PROF_OPS=1`.

### The integrated GPU where it was found

Measured earlier in this project's campaign, on the device where the header
unpacking was on the critical path — AMD Ryzen 7 7800X3D integrated GPU, RADV
RAPHAEL_MENDOCINO, Mesa 25.2.7, `unsloth/gemma-4-E4B-it-GGUF`
`gemma-4-E4B-it-Q4_K_M.gguf` (4,977,171,584 bytes), decode 16 tokens at depth
16, context 128, three repetitions per process in
`scalar / packed / packed / scalar` order, serial headless workloads,
`INFR_NO_COOPMAT=1 INFR_KQ_FMA_GEMV=0 INFR_KQ_FMA_TIER=0`, an empty config file:
6.14 and 5.86 tok/s on the scalar metadata (mean 6.00) against 7.66 and 7.31 on
the packed (mean 7.49), about 24.8% higher. That GPU is not available here, so
the figure is quoted rather than reproduced, and it covers one model at one
depth.

## Measurements on the Windows dev box

Both GPUs on one machine, AMD Windows driver 2.0.353 (`AMD_PROPRIETARY`, driver
version `0x800161`) on both, as `infr devices` and `infr tune show --json`
report them:

- `Vulkan0`: AMD Radeon RX 7900 XTX (RDNA3, discrete, 24 GiB).
- `Vulkan1`: AMD Radeon(TM) Graphics, the Ryzen integrated GPU (RDNA2; infr
  reports 2 cores).

Both report the packed int8 dot as accelerated, so `gemv.kq_fma` is false on
both and Q4_K decode takes the int8 GEMV (`native_mmv_mrow`) by default. Which
build it takes, read from the per-op profile of a Qwen3-0.6B Q4_K_M decode
(`INFR_PROF_OPS=1 INFR_SEAM_NO_REPLAY=1`, the replay tape carries no
timestamps), the first profiled forward of each run, microseconds per dispatch:

| device  | knob | Q4_K builds in decode                                                        |
| ------- | ---- | ---------------------------------------------------------------------------- |
| RX 7900 | off  | `native_mmv_mrow_q4k_o4_m4` 6.8 us, `native_mmv_mrow_q4k_m4_res` 4.8 us      |
| RX 7900 | on   | `native_mmv_mrow_q4k_o4_m4_packed_meta` 6.9 us, `_q4k_m4_res` 4.8 us         |
| iGPU    | off  | `native_mmv_mrow_q4k_o4_m4` 77.2 us, `native_mmv_mrow_q4k_o4_m4_res` 46.1 us |
| iGPU    | on   | `..._q4k_o4_m4_packed_meta` 56.6 us, `..._q4k_o4_m4_res_packed_meta` 33.7 us |

On the discrete card the residual projections are 2048 wide (Qwen3-0.6B's
`o_proj` and `down_proj` inputs are 2048 and 3072), so they take the two-output
build, which has no packed twin.

**Correctness.** `packed_metadata_matches_scalar_exact` passes on both GPUs. The
tolerance harness (`measure_gpu_tolerance`,
`INFR_TOL_MODELS=qwen3-0.6b,qwen3-1.7b INFR_TOL_ROWS=1`, reference cache `on`,
the default) prints the same 450 `TOL` rows and the same per-position `ROWS` KL
values, byte for byte, with the knob on and off, on each GPU. The harness does
not read `INFR_MMV_Q4K_META` (`infr_testkit::honour_env` carries only the tuning
parameters and capability masks), so the "on" runs were made from a local build
with the knob's default set to true; their `KERNELS` lines name
`native_mmv_mrow_q4k_o4_m4_packed_meta` (both models on the integrated GPU,
Qwen3-0.6B on the RX 7900 XTX) and `_o4_m4_res_packed_meta` (integrated GPU),
and the "off" runs name no packed build.

**Microbench**, `packed_metadata_bw` (the command above), two processes per
device, each running scalar, packed, packed, scalar; means of the four legs per
build, effective weight GB/s:

| device  | shape (in × out) | scalar | packed |  delta |
| ------- | ---------------- | -----: | -----: | -----: |
| RX 7900 | 1536 × 2048      |  201.4 |  221.9 | +10.2% |
| RX 7900 | 1024 × 8192      |  454.2 |  456.1 |  +0.4% |
| RX 7900 | 1536 × 1024      |  107.1 |  106.6 |  -0.4% |
| iGPU    | 1536 × 1024      |    9.7 |   10.6 |  +9.3% |
| iGPU    | 1024 × 2048      |   11.3 |   12.7 | +12.4% |

The RX 7900 XTX legs spread widely (173.3 to 225.5 GB/s for the scalar 1536 ×
2048 legs), so its rows are noise; the integrated GPU's legs do not overlap.

**Decode**, `infr --config <empty file> bench <model> -p 0 -n 128 -r 3` and the
same with `-d 4096`, `INFR_DEV` set, knob on (`INFR_MMV_Q4K_META=1`) and off.
Six rounds per device, the two modes alternated within each round and the order
flipped every round, the mode run from two byte-identical copies of the release
binary with the copy swapped every round. Mean of the six runs (each the mean of
its 3 repetitions), with the range of the six:

| device  | model, phase                     |                  on |                 off |  delta |
| ------- | -------------------------------- | ------------------: | ------------------: | -----: |
| RX 7900 | Qwen3-0.6B Q4_K_M, tg128         | 694.0 (693.2-694.4) | 691.4 (691.0-691.8) |  +0.4% |
| RX 7900 | Qwen3-0.6B Q4_K_M, tg128 @ d4096 | 515.1 (514.9-515.6) | 513.8 (513.4-514.2) |  +0.3% |
| RX 7900 | Qwen3-1.7B Q4_K_M, tg128         | 455.7 (455.3-456.2) | 455.5 (454.6-456.3) |   0.0% |
| RX 7900 | Qwen3-1.7B Q4_K_M, tg128 @ d4096 | 366.8 (366.5-367.4) | 367.0 (366.7-367.6) |   0.0% |
| iGPU    | Qwen3-0.6B Q4_K_M, tg128         |    61.7 (61.6-61.8) |    55.4 (55.2-55.7) | +11.5% |
| iGPU    | Qwen3-0.6B Q4_K_M, tg128 @ d4096 |    31.2 (31.1-31.4) |    29.5 (29.3-29.8) |  +5.7% |
| iGPU    | Qwen3-1.7B Q4_K_M, tg128         |    32.4 (32.3-32.6) |    26.9 (26.6-27.3) | +20.7% |
| iGPU    | Qwen3-1.7B Q4_K_M, tg128 @ d4096 |    21.4 (21.3-21.5) |    18.8 (18.7-19.0) | +13.6% |

Tokens per second. The machine was otherwise idle: total CPU, sampled for 1 s
before each run with `typeperf`, averaged 6.7% (RX 7900 XTX series) and 6.8%
(integrated GPU series) of 32 logical cores, at most 12.9%, and a per-process
sample before the window showed the System process and the antivirus at about a
third of one core each and nothing else above an eighth of one. The Qwen3-1.7B
rows on the RX 7900 XTX are the control: the packed build is not reached there,
and the two modes agree within their ranges. gemma-4 E4B was not cached here and
was not measured.

## Correctness gates

`tests/mmv_q4k_meta.rs::packed_metadata_matches_scalar_exact` compares scalar
and packed output bits for rows 1, 2, 3, 4, 5, 8, 9 and 16, both row caps, with
and without the fused residual, over output tails (1, 3, 5, 66) and weight bases
0/256/512/768. The two-output leg covers `in_f` 256, 1536, 2048, 2560, 2816,
6144 and 11008; the four-output leg stays under 2048 so that a discrete device
takes it through `o4_max_in_f` and an integrated one through `o4_all_widths`,
which keeps the expectation device-independent. The weight bytes cover zero,
signed zero, subnormal and signed half scales and both edges of the packed
scale/min fields.

Because `built_kernel_names` accumulates over a backend's life, the kernel-name
check compares the names each run _added_ to the names it started with: a run
may add nothing (the variant is already built) or exactly the kernel the naming
scheme predicts, and the first case that needs a variant has to build it. A row
that silently fell back to the scalar build fails there even though the earlier
kernel is still in the list.
