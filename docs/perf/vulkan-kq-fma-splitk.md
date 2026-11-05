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

## Tiles that were measured and rejected

The grid is the easy axis. The tile itself is not, and both wider shapes lost on this card:

| Variant | Shared (per workgroup) | Result |
| --- | ---: | --- |
| `BN=128`, `BK=32`, 8x8 per thread (shipped) | 32 KiB | pp512 360 t/s, pp128 318 |
| `BN=256` for Q4_0, 8x16 per thread | 48 KiB | pp512 331, pp128 299 (-8%) |
| `BK=16`, 16 KiB | 16 KiB | rejected: parity failure |

`BN=256` is numerically clean — a Q4_0 staging thread owns one whole 32-element block, so it
reads both nibble halves of the same bytes — but 48 KiB of shared memory drops residency from
two workgroups per CU to one, and half the residency costs more than the reduced load count
buys. `BK=16` would fit three or four workgroups, yet it cannot work in this kernel: the quant
block math assumes a 32-element-aligned `k0` (a 16-aligned tile start lands in the wrong Q4_0
block, and `part * EPT` offsets no longer start a run), so `nc_kq_fma_gemm_parity_and_determinism`
fails for Q4_0 at `BN=64` before any measurement.

Both are reverted. The shipped tile sits at a local optimum: 32 KiB exactly, two workgroups per
CU, 64 accumulators per thread. The distance to the fp32 peak (~4.1 of 6.17 TFLOP/s on the two
hot rows) is per-thread issue rate, not tile shape — closing it means packed f16 math, which
trades the exactness the parity test enforces.
