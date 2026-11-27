# Intel integrated GPUs: correct first, then tuned

The first Intel device infr has run on is a laptop's Raptor Lake-P UHD Graphics
(i7-1355U, PCI id `0xa721`, Xe-LP, Mesa ANV). Its first gate run is
`docs/backlog.md` B-INTEL-IGPU. This plan makes that device pass the GPU gate
and then gives it tuning rows in the built-in table
(`crates/infr-vulkan/src/tuning/table.toml`), so an Intel iGPU runs measured
values instead of the generic defaults tuned on AMD.

Accuracy comes first: no row lands for a device that still fails the gate, and
no row lands without `infr tune check` showing the device still within its
tolerance rows with the row's values.

## S1. The model gates run on a small host — done (`a9752d37`)

The unified-memory margin scales with total host memory (`unified_host_margin`),
so the tolerance gate runs small models on a 16 GB laptop instead of skipping
all of them.

## S2. RoPE within tolerance on Intel — done (`8e0a650a`, `3def4aec`)

Every RoPE shader (`rope.comp`, `qk_norm_rope.comp`,
`qk_norm_rope_interleaved.comp`, `mla.comp`) takes the GLSL built-in `sin`,
`cos` and `pow`. The CPU reference rounds `sin_cos` and `powf` correctly
(`infr_core::math`). First measure which built-in carries the error on ANV. Then
replace it in every RoPE shader, from one shared include, with an f32 evaluation
whose error bound is stated and tested:

- range reduction by a split `pi/2` (Cody-Waite), then minimax polynomials;
- or the frequencies computed on the host with `rope_freqs` and uploaded, if
  `pow` is the culprit.

Checks: `rope_matches_host` and `rope_freq_factors_match_ggml_at_long_positions`
pass on the iGPU, and a bit-identity or tolerance comparison against the
previous build on another device shows nothing moved away from the reference. No
other device is available here, so an AMD run is owed and recorded as such.

## S2b. The int8 path on gemma3

On the iGPU, `gpu_seam_tolerance_gemma3` fails its two int8 runs (`gpu`,
`gpu-ub32`): mean KL 0.062 over a 0.027 bound, and top-k cosine 0.60 at position
114 of the 180-token prompt. Every float run passes, and so does the greedy
check. The device reports DP4A as accelerated, so it takes the dp4a kernels. The
same symptom on the AMD iGPU is B207, owned by the Windows session. The split
agreed between the two sessions:

- **The Windows session owns** any change to the dp4a K-quant kernels
  (`native_gemm_dp4a_q4k` / `_q6k` and the rest of that family).
- **This plan narrows the cause down on Intel** without editing those kernels.
  It uses tuning switches and the gate to find which kernel and which stage
  carries the error (GEMM or GEMV, the activation quantizer), and hands over the
  numbers.
- **An Intel-only cause stays here**, for example a 16-wide subgroup in the
  quantizer.

## S2c. Win RoPE's speed back

S2 (`8e0a650a`, `3def4aec`) made RoPE accurate on every device at a cost:

- **Intel iGPU:** Qwen3-1.7B pp512 152.1 / 152.2 to 149.2 / 149.2 t/s.
- **RX 7900 XTX, AMD Windows driver:** tg128 453.43 to 440.48 t/s (-2.9%, six
  interleaved rounds).
- **AMD iGPU:** tg128 27.15 to 27.11, within its spread.

Decode runs one row, so the per-pair work is what shows: each thread derives its
frequency in double-float (`rope_freq`) and reduces its own angle. The
frequencies depend only on the op's shape, so compute them once per op on the
host with `infr_core::math::rope_freqs` (the CPU's own values, so bit-identical
by construction) and bind them the way `freq_factors` is bound. That leaves only
the sincos per pair. Hold the angle bit-identity and `rope_angle_matches_f64` as
they are, and measure tg128 on the XTX (through the Windows session) and
pp512/tg128 on the Intel iGPU, alternating builds.

## S3. The harness failures — the first two done; the pool guard is B-ANV-UMA-BUDGET-GUARD

- The multi-device tests open every enumerated device. They must skip a device
  the backend refuses (llvmpipe here), not panic.
- The slow-device tests force wave64 to make a device slow. Where 64 cannot be
  pinned they need another way to slow the device, or a skip that says why.
- The two pool-guard tests see room the filler should have taken under ANV's
  memory budget. Find out what `VK_EXT_memory_budget` reports before and after
  the filler, then fix the guard or the test, whichever is wrong.

## S4. Core count

`0xa721` is not in `intel_shader_core_count` (transcribed from llama.cpp), so
the device reports `cores:?` and every rule scaled by core count falls back. A
device id does not fix the EU count (the same id ships in 80 and 96 EU parts),
so read it from the driver if ANV exposes it. Otherwise leave it unknown and
tune the parameters that depend on it directly in S5.

## S5. Tuning rows for the Intel iGPU

There is no `infr tune sweep` yet (plan 10 section 3.5). Sweep by hand: for each
candidate parameter (`infr tune list`; first `device.sg_pref`, `gemv.variant`,
`gemv.kq_fma`, `gemv_rm.rows`, `gemv_sg.rows`, `mmv_mw.warps@*`, the prefill
tile and split-K parameters, and the chunk height and submit split the watchdog
rule sets), run

```text
INFR_DEV=Vulkan0 INFR_TUNE=<param>=<value> infr bench -p 512 -n 0 -r 5 <model>
INFR_DEV=Vulkan0 INFR_TUNE=<param>=<value> infr bench -p 0 -n 128 -r 5 <model>
```

on Qwen3-1.7B, Llama-3.2-1B and Gemma-3-1B Q4_K_M, alternating values in one
quiet window (AGENTS.md "Performance measurements"). Then run `infr tune check`
at the winning set. A value that wins and stays within tolerance goes into a
row:

- a class row (`vendor = "intel"`, `arch = "IntelXe1"`,
  `driver = "INTEL_OPEN_SOURCE_MESA"`, `integrated = true`) for what should hold
  across Xe-LP iGPUs;
- a device row (`device_id = 0xa721`) only for what is plainly specific to this
  part.

Each row cites its source (device, driver, date, command, evidence) the way the
AMD rows do. The snapshot test for the Intel fixture changes with it, and no
other fixture's snapshot may.

## Not in this plan

The NVIDIA MX550 in the same laptop. Its module did not load at the time (kernel
package update pending a reboot). It gets its own run and its own rows when it
is back.
