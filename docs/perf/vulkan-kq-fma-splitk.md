# Splitting the kq_fma prefill k-range inside one dispatch

`native_gemm_kq_fma.comp` is the fp32 FMA prefill GEMM for K-quant/Q4_0 weights on GPUs
without a fast int8 dot (see [legacy-gpus](../legacy-gpus.md)). Its tile is 128 rows x 128
(or 64) columns, one workgroup per tile, so a narrow grid underfills the device: a 512-token
prefill of `down_proj` (k=10240, n=2560) dispatches 80 workgroups onto 36 CUs.

## What the grid costs, measured in isolation

Timing the kernel alone with a fixed Q4_0 weight bank, m=512, tile 128 (`nc_gemm_parity`'s
shape scan), shows the deficit is workgroup count, not the long k:

| shape (k x n) | workgroups | us per workgroup-tile | TFLOP/s |
| --- | ---: | ---: | ---: |
| 2560 x 2560 | 80 | 32.6 | 2.58 |
| 10240 x 2560 | 80 | 110.2 | 3.04 |
| 2560 x 20480 | 640 | 18.1 | 4.63 |
| 10240 x 10240 | 320 | 96.5 | 3.48 |
| 10240 x 20480 | 640 | 95.7 | 3.51 |

A dense grid runs each tile up to 1.8x faster, so the lever is more workgroups.

## One dispatch per slice does not work

The first version dispatched `splits` copies of the tile, each writing its own plane of a
`splits x mpad x n` partial buffer, then ran `splitk_reduce`. Back-to-back dispatches do not
overlap on this driver, so each slice paid the same underfilled grid again and the reduce was
pure overhead: prefill went 338 -> 321 / 320 / 314 t/s at 2 / 4 / 8 splits. A slice cost
exactly `1/splits` of the unsplit row, which is the signature of no added parallelism.

## One dispatch, slices in the workgroup id

The slice now rides the top of `gl_WorkGroupID.x`, so `groups = row_tiles * column_tiles *
splits` is a single dispatch and every workgroup is co-resident. The tile, the k-block stride,
the staging and the epilogue are unchanged; `splits = 1` is still the direct tile writing `c4`
at offset 0, and the reduce pass is the same one.

Prefill throughput, `infr bench -p <n> -n 0 -r 3 --ctx 4096 --dev Vulkan0`, empty config,
gemma-4-E4B-it-qat-UD-Q4_K_XL, Polaris10 (RX 580):

| pp | direct tile | auto split | delta |
| ---: | ---: | ---: | ---: |
| 64 | 160.0 | 187.9 | +17.4% |
| 128 | 268.3 | 311.2 | +16.0% |
| 256 | 308.8 | 351.9 | +14.0% |
| 512 | 333.7 | 354.3 | +6.2% |
| 1024 | 335.5 | 345.6 | +3.0% |
| 2048 | 321.4 | 326.3 | +1.5% |

The gain falls with m because the direct grid grows with it: past ~8 workgroups per CU the
split stops paying and only the partials round-trip is left. The policy therefore targets
`8 * compute_units` workgroups, caps at 8 splits, and halves until `k` divides by `256 *
splits`. `down_proj` lands at 8 splits for a 64-token prompt and 4 at 512; the gate/up rows
already have 640 workgroups and never split.

Numerics are unchanged: slices sum in fixed plane order in `splitk_reduce`, and
`nc_kq_fma_splitk_matches_the_direct_tile` checks both against the host reference across
Q4_0/Q4_K/Q6_K and both tiles.

`INFR_KQ_FMA_SPLITK=<n>` (config `kernels.vulkan.kq_fma_splitk`) forces the count; `1`
disables the split.
