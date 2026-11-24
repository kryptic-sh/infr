# 10 — Vulkan kernels with per-device tunable parameters

Every hardware-shaped number in a Vulkan compute kernel (workgroup size,
subgroup width, rows per group, tile sizes, split counts) becomes a parameter
chosen per device at pipeline creation, so tuning for a card means choosing
values in one central table, not writing a shader. People with other hardware
can sweep the values on their card and send the result as a small data change.

Scoped on 2026-11-17 by reading every shader and its host dispatch, with a few
throwaway `glslc`/`spirv-opt` experiments (shaderc 2026.3). Tags: **[R]** read
or run, **[I]** inferred, **[U]** unknown until measured.

**S1 is done (2026-11-17):** the table, the resolver, `infr tune show/list`, the
aliases and the fixtures exist (`crates/infr-vulkan/src/tuning/`), and every
existing runtime value reads through them; §0 below describes the tree as it was
scoped, and S1's section in §5 says what it found that differs.

## 0. Where things stand

- **No shader uses a specialization constant for its shape [R].** The only one
  is `W64` (`constant_id = 1000`, `shaders/wave64.glsl`), set on the wave64 path
  by `ops::try_make_compute_kernel`. Every `local_size_x` is a literal or a
  `#define`.
- **The build list [R]:** every built `.comp` source expands to a base entry per
  `-D` set, the `SG16_SOURCES` expansion adds `_sg16` twins for
  `native_gemv_sg`, `native_gemv_id_multi_sg`, `native_mmv_mw` and
  `quant_q8_row`, and the STREAMED pass adds `_streamed` twins.
- **The kernel cache is keyed on the kernel's name alone [R]**
  (`shared.kernels`), with a debug assertion against one name with two SPIR-V
  bodies.
- **Two creation paths [R]:** `kernel()` does not pin a subgroup width on native
  hardware (the driver picks); `kernel_sg()` pins 32, `caps.sg_pref`, 16 or
  `MLA_SG_WIDTH`. On the wave64 path `subgroup_pin` pins 64 everywhere.
- **Device classification [R]:** `caps::DeviceArch`, whose only consumer is the
  cooperative-matrix trust rule; `Capabilities` and `DeviceProbe` carry the
  subgroup range, `sg_pref`, integrated, compute units, shared memory, driver
  id, device id and vendor id.
- **Tuning knobs are scattered [R]:** config keys (`flash_splits`, `pv_splits`,
  `flash_min_rows`, `flash_bm32`, `mmv_mw_warps`, `moe_small_m`, the `gemv.*`
  group, `device.subgroup_pref`) and inline host constants
  (`SPLIT_K_FIXED_GRID`, `SPLIT_K_FILL_PER_CU`, `ATTN_SPLIT`, which
  `attn_live.comp` duplicates in GLSL, `QUANT_Q8_BLOCKS_PER_WG`, `MLA_SG_WIDTH`,
  the per-dtype mmv WARPS default).
- **No issue or PR templates, no contributing or hardware doc [R].**

### What glslang and SPIR-V allow [R]

1. `layout(local_size_x_id = N)` emits `LocalSizeId`, not `LocalSize`, so
   `spirv::local_size()` returns `None`: `caps::check_kernel_workgroup`, the
   full-subgroups decision in `subgroup_pin` and
   `subgroup_kernels_are_wave64_safe` all need a `LocalSizeId` reader.
2. Writing only `local_size_x_id = N` gives a default of 1. The default must be
   written in both forms
   (`layout(local_size_x = 32) in; layout(local_size_x_id = N) in;`), and the
   host should always supply every declared constant.
3. A cluster size cannot be a spec constant (`subgroupClusteredAdd(x, SPEC)` is
   refused). A width-dependent cluster needs literal arms or a `-D`.
4. Spec constants do work for array sizes, `shared` arrays, loop bounds and
   cooperative-matrix dimensions (driver acceptance of the last is **[U]**).
5. `spirv-opt --freeze-spec-const -O --canonicalize-ids` on a spec-constant
   module, compared with the `-D` module of the same values, differs only in
   `LocalSizeId` against `LocalSize` and the order of two constants. This is the
   basis of the "defaults are today's kernels" proof.
6. Specialization data is not in the SPIR-V, so the shader fingerprint does not
   change with parameters; the driver's pipeline cache keys on the create info
   **[I]**, so the on-disk cache gains entries per parameter set.

## 1. Kernel classes

Many literal 32s are quant block sizes (Q8_0, Q4_K sub-blocks), not lane counts,
and must not be parameterized.

The authority is now `crates/infr-vulkan/src/tuning/classes.rs`: every `.comp`
source has a class there, and a test fails when one is missing or when an
assignment disagrees with what the source uses (class 4 exactly the
cooperative-matrix kernels, class 2 only kernels with subgroup operations, class
5 none with any). Corrections S1 made to the lists below: `attn_pv_reduce` and
`attn_flash_combine` use no cooperative matrix (class 5a, not 4: the `attn_pv*`
and `attn_flash*` globs were too wide; `attn_flash_combine` only enables the
subgroup extension); `native_gemm*` in class 4 means the coopmat builds only
(`native_gemm`, `native_gemm_warp`, `native_gemm_f8cm_q8_0`,
`native_gemm_i8cm_q8_0`, `native_gemm_i8cm_kq`), the mmq/dp4a/fma ones are 5b;
`attn_combine` was unclassified and is class 3 (its `wexp[1024]` is the host's
`ATTN_MAX_CHUNKS`); `attn_ktile` uses no subgroup operation at all (still class
1). `moe_topk`, `wave64_probe` and `native_gemm_i8cm_kq` use subgroup operations
without including `wave64.glsl`.

- **Class 1 — already parameterized or width-agnostic:** `native_gemv_sg`
  (`-DSG`, `-DNR`), `native_gemv_id_multi_sg`, `native_mmv_mw` (`WARPS*SG`),
  `quant_q8_row`, `moe_topk`, `wave64_probe`, and the probe-only
  `attn_partial_dsplit` and `attn_ktile`. Selection goes through `caps.sg_pref`.
- **Class 2 — width only in local size, strides or per-subgroup arrays
  (mechanical):** `rmsnorm` (incl. `_wide`), `layernorm`, `softmax`,
  `native_gemv_id`, `native_gemv_id_multi`, `native_gemv_mrow`,
  `native_gemv_rm_v2`, `native_mmv_mrow` (the default AMD decode GEMV; its
  `MAX_SG` sizes partials for a minimum 8-wide subgroup), `dg_eb_sample`,
  `quant_f8_row`, `quant_q8` (lane = element of a Q8 block; blocks per workgroup
  is the knob, with a host group count and a drift test), `store_q8`,
  `deltanet_norm`, `deltanet_seq` (register arrays sized `kd/32`), and
  `attn_nc_fa` (its base build is subgroup-free; the `_sg64` twin reduces the
  softmax rows with clustered ops). The unpinned
  ones already run at whatever width the driver picks; which width AMD's Windows
  driver picks is **[U]**.
- **Class 3 — shape baked into a layout shared with the host or other kernels:**
  `attn_decode`, `attn_partial`, `attn_partial_mrows` (64 = two subgroups of 32;
  head dim 128 = 32 lanes × vec4; `sc[512]`; the chunk policy duplicated in GLSL
  and host; a partials layout `attn_combine` consumes), `attn_live`,
  `deltanet_gates` and `deltanet_scan` (chunk 32 is an algorithm parameter that
  changes numerics; `deltanet_gates` needs a scan in invocation order), `mla`
  (`MLA_SG_WIDTH`).
- **Class 4 — cooperative matrix:** the `gemm_*`, `native_gemm*` (f16, bf16, i8,
  f8, `_cm8`), `attn_flash*`, `attn_qk*`, `attn_pv*` kernels, `deltanet_prep`
  and the `coopmat_i8_layout` probe. Shapes come from the driver's enumerated
  matrix shapes; all pin 32; all are off on the wave64 path.
- **Class 5 — no subgroup operations:** (a) element-wise, gather and copy
  kernels where local size is the only, arbitrary, knob; (b) shared-memory tree
  or tiled kernels whose workgroup or tile is a real knob independent of width
  (the `native_gemm_mmq_*` family, `native_gemm_dp4a_kq`, `native_gemm_fma`,
  the tree GEMVs, `linear_*`, `attention_kv`, `lightning_indexer`
  and others).

**Where GCN loses time today [I]:** workgroups of 64 or more already fill each
wave under the emulation. The waste is workgroups of at most 32 invocations,
which leave half of each wave idle: `native_gemv_sg`, `native_gemv_id_multi_sg`,
`deltanet_seq`, `deltanet_norm`, `deltanet_gates`, `store_q8`. The class-3
attention kernels are the opposite case: their 32 is the head dimension, and two
32-lane halves of a 64-lane wave already is the natural algorithm there.

## 2. Mechanism: spec constants for sizes, `-D` for code structure

- **Spec constants** for every size that only affects local size, loop bounds,
  strides, array sizes, rows or blocks per workgroup and split counts: one
  SPIR-V per (source, format, structural define), specialized at pipeline
  creation as `requiredSubgroupSize` and `W64` already are. Any value in the
  declared range can be swept without a rebuild, and the `_sg16` twins and probe
  width builds collapse.
- **`-D` variants stay** for quant format, `STREAMED`, `USE_RES`, `KV_BDA`,
  cooperative-matrix element types, any width-dependent clustered operation, and
  algorithm switches. Cooperative-matrix tile dimensions stay `-D`; the host's
  choice among those builds becomes an enum parameter in the table.
- A `-D` build per width instead would add roughly three quarters again to the
  module count **[I]**.

**Conventions.** Reserved `constant_id` ranges: 1000–1009 backend-global (1000
is `W64`); 1010–1019 standard shape parameters (`P_SG`, `P_WG_X/Y/Z`, `P_ROWS`,
`P_WARPS`, `P_BLOCKS`, `P_SPLITS`, `P_UNROLL`, `P_TILE`); 1020–1099 further
standard ones; 1100–1999 kernel-private. A shared header `shaders/params.glsl`,
included after `wave64.glsl`, declares the constants from per-kernel defaults,
writes the local size in the double form, derives dependent sizes, and offers a
clustered-add helper that switches over literal arms. The SPIR-V `SpecId` set is
the shader-side declaration; a unit test holds it equal to the Rust family
descriptor's, replacing the per-constant drift tests.

**Composition with the emulation.** A parameterized kernel specialized to 64
lanes runs with `W64` false, pinned 64, with plain subgroup operations. If its
resolved width is 32 on a wave64-only device, the host sets `W64` and today's
emulation runs. The emulation stays the fallback for that case and for every
unconverted kernel. Full subgroups are requested when the SPECIALIZED
`local_size_x` is a multiple of the pin.

## 3. Host side

### 3.1 One central place: `crates/infr-vulkan/src/tuning/`

- `families.rs` (Rust): one descriptor per kernel family with its parameters
  (name, kind: spec constant / host value / build variant / pin, declared range,
  qualifier axis such as dtype), the DERIVED generic default as a function of
  the device's facts, a validator, and the functions that turn parameters into
  local size, shared-memory bytes and group counts.
- `table.toml` (data, compiled in): every VALUE, as generic overrides and
  device-class and device rows with provenance. Contributors edit this file
  only. The same parser loads a user's file at run time, so a row can be tried
  without rebuilding.
- `resolve.rs` and `report.rs`.

The device's facts are gathered once at backend creation: vendor, `DeviceArch`,
driver id and version, device id, name, integrated, VRAM, compute units,
subgroup range, required-subgroup-size stages, the compute workgroup limits
(including `maxComputeWorkgroupSubgroups`, a new query), cooperative-matrix
shapes, wave64 mode.

### 3.2 Resolution: most specific wins, per parameter

1. Built-in generic default (derived from the device's facts).
2. Class rows: vendor plus architecture, driver, integrated, subgroup range or
   compute-unit range.
3. Device rows: vendor plus PCI device ids (or a name pattern, discouraged),
   optionally a driver and version range.
4. The user's tuning file (`kernels.vulkan.tuning.file` / `INFR_TUNE_FILE`).
5. Config (`[kernels.vulkan.tune]`).
6. Environment (`INFR_TUNE="family.param=value,…"`).
7. Command line (`--set kernels.vulkan.tune.…`).

A row sets only the parameters it names; the rest fall through. Device rows beat
class rows; within a level the row matching more key fields wins; two built-in
rows that tie on a fixture and disagree fail the table test. Existing knobs
(`flash_splits`, `pv_splits`, `mmv_mw_warps`, `gemv.*`, `moe_small_m`,
`flash_min_rows`, `flash_bm32`, `device.subgroup_pref`) become aliases of the
matching parameter. `kernels.vulkan.tuning.rows = false` uses defaults only, and
`kernels.vulkan.tuning.disable = [ids]` drops named rows. A row marked
`verified = "contributor"` logs once at startup that it is in use and how to
turn it off.

### 3.3 One source of truth

A typed per-family parameter struct is resolved once per backend. The recorder
hands the same struct to pipeline creation (which builds the specialization
info, merged with `W64`, and the pin) and to the family's group-count and
scratch-size functions, so the shader receives the number the host divides by.
The kernel cache key becomes (name, parameter key).

### 3.4 Validation at startup

Per family: value in range; specialized local size and invocation count within
the device limits; pin within the subgroup range and compute among the
required-subgroup-size stages; invocations within pin × subgroups per workgroup;
full subgroups only where the local size divides; shared memory within the
limit; family constraints. A ROW value that fails is dropped with a warning and
falls through, so a bad row cannot brick a device; a USER value that fails is a
hard error naming the key. No known-answer dispatch at every startup (init
latency); it is `infr tune check`, and mandatory inside a sweep.

### 3.5 Tools: `infr tune`

- `show [--dev] [--json]`: the device header, the match key the table uses, the
  matched rows, then each parameter with its value and the level it came from. A
  short hash of the set goes in the device banner.
- `list [family]`: each tunable parameter with its range, default rule and the
  value on this device.
- `check [family]`: each family's known-answer or parity check at the resolved
  parameters.
- `sweep <family.param> --values … | --range …`: per value, a correctness gate
  (a failing value prints FAIL and is never ranked), an op-level microbench with
  values interleaved in one process, optionally model-level `infr bench` child
  runs; output is a table, a paste-ready report and a candidate row with
  provenance filled in.

### 3.6 Table schema and tests

```toml
[[row]]
id = "amd-rdna3-amdwin"
match = { vendor = "amd", arch = "AmdRdna3", driver = "AMD_PROPRIETARY" }
verified = "maintainer"            # or "contributor"
source = { by = "…", device = "…", driver = "…", date = "…", command = "…", evidence = "…" }
params = { "gemv_sg.nr" = 2, "mmv_mw.warps@q4_k" = 1 }
```

Tests: schema; every parameter exists and is in range; every row names a vendor
and at least one more key field and never matches the `unknown` fixture; no
ambiguous ties; every row cites its source; each family's spec parameters equal
the `SpecId` set of every build; and a snapshot test that resolves the full set
for fixture devices (the two local GPUs under each driver, a GCN card, Intel,
NVIDIA, unknown) against checked-in files. A PR adding a row may only change the
snapshots of fixtures its key matches: "a row cannot change defaults for
hardware it does not name", made mechanical.

## 4. Verification

- **Defaults are today's kernels.** (a) Structural, per converted family:
  compile the old source and the new one, freeze the new module's constants at
  the default values with `spirv-opt`, and diff; expected empty apart from the
  `LocalSizeId` form. (b) Runtime, on both GPUs: the family's op-parity tests
  bit-identical to the old build at defaults, and the full GPU gate with its
  exact-token goldens, native and with `INFR_FORCE_WAVE64=1`. If a family drifts
  because the driver specializes differently **[U]**, keep a `-D` build for the
  default and use spec constants for non-default values. (c) Resolver: the
  snapshot test, and the per-op kernel-name list of a prefill and a decode
  identical before and after.
- **Across values:** bit-identical where the per-output summation order is
  unchanged, tolerance where a width change regroups a reduction; the
  decode-equals-verify contract (`mmv_row1_bit_identical`) holds under every
  pin.
- **Static test:** `subgroup_kernels_are_wave64_safe` learns the specialized
  local size and checks each allowed specialization; its width-agnostic list
  becomes a family attribute.
- **Performance guard:** defaults must not move pp512 or tg128 on either GPU
  beyond noise.

## 5. Slices

Each is green on Linux and Windows CI and needs no macOS verification.

| #   | Slice                                                                                                                                                                                                                | Verification                                                                                                                |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| S1  | **Done.** The central table, resolver and `infr tune show/list`; every EXISTING runtime value migrated verbatim as the default; config aliases; snapshot fixtures; every shader assigned a class. No kernel changes. | unit and snapshot tests; identical kernel-name lists, GPU gate and goldens on both GPUs                                     |
| S2  | Contributor doc v0 (`docs/kernel-tuning.md`) and an issue form for tuning reports; links from the README and perf docs                                                                                               | reads true against S1                                                                                                       |
| S3  | The spec-constant mechanism: `params.glsl`, `kernel_p`, `LocalSizeId` and `SpecId` readers, the new limit checks, a probe that reports the real subgroup and workgroup sizes per specialization                      | the probe across workgroup sizes and pins on both GPUs; GPU gate unchanged                                                  |
| S4  | First kernel end to end and the GCN win: `native_gemv_sg` on `P_SG`/`P_ROWS`, with a native 64-lane value (the community fork's commit `a1f58563b` is the reference)                                                 | equivalence at 32 and 16; on both GPUs native 64 against the CPU, and its speed against native 32 and the 32-lane emulation |
| S5  | `infr tune sweep` and `check`, the parity and microbench harness, the report and row generator; doc v1                                                                                                               | a sweep on both GPUs; a deliberately bad value shows FAIL and is never ranked                                               |
| S6  | Class 2 batch A: the decode GEMVs (`native_mmv_mrow`, `native_gemv_mrow`, `native_gemv_id*`, `native_gemv_rm_v2`, `native_mmv_mw`) with a pin parameter                                                              | equivalence; bit-identity at defaults; tolerance across pins; the decode-verify contract per pin; perf guard                |
| S7  | Class 2 batch B: norms, softmax, quantization (`quant_q8`'s blocks per workgroup as the one-source group-count exemplar)                                                                                             | as S6                                                                                                                       |
| S8  | Class 2 batch C: the small-workgroup DeltaNet kernels                                                                                                                                                                | Qwen3.5-0.8B; GCN gated                                                                                                     |
| S9  | Rows for the two local GPUs from sweeps of everything converted so far, native 32 against native 64 against unpinned included                                                                                        | the protocol below                                                                                                          |
| S10 | Split today's tuned values from the generic defaults (the only slice that changes behaviour, and only for hardware no row matches)                                                                                   | snapshots for the two local GPUs byte-unchanged; `infr tune show --json` and kernel-name lists identical on both            |
| S11 | Class 3: the attention split family (chunk policy as push constants read by `attn_live`, `SC_MAX`, workgroup size, the per-key reduction width)                                                                      | parity; the deep-context shapes of B15                                                                                      |
| S12 | Class 5b in batches, as Intel or NVIDIA numbers ask for it                                                                                                                                                           | parity; perf guard                                                                                                          |

Class 4 gets no slice: its tile shapes stay builds, and the host's choice among
them becomes a variant parameter in the table (S1, S10).

**What S1 found.** Values migrated (`tuning::families`, `infr tune list`):
`device.sg_pref` (the capability rule; alias `INFR_SG`); `gemv.variant`,
`gemv_rm.*`, `gemv_sg.*`, `gemv_id_sg.enabled` (the `[kernels.vulkan.gemv]`
group); `mmv_mw.warps@dtype` (per-format default, alias `mmv_mw_warps`);
`mmv_mrow.o4_max_in_f` / `o4_all_widths` (the four-outputs GEMV rule: below 2048
wide, or every width on an integrated GPU — not in the plan's list);
`attn_flash.*` and `attn_pv.*` (the fixed count, `flash_min_rows`, `flash_bm32`
and the split policy's five constants, one policy for the two flash sites and
the PV site, which compute the same function); `attn_canvas.chunks`;
`moe.small_m`; `split_k.fixed_grid` / `fill_per_cu` / `scale_by_cu`;
`gemm_i8cm_kq.enabled` and `l_fill_per_cu` (the large-tile fill, not in the
plan's list); `rmsnorm.wide` / `wide_min_dim`. Reported as `fixed`:
`attn_split.*` (`ATTN_SPLIT`, which `attn_live.comp` computes too),
`mla.sg_width`, `quant_q8.blocks_per_wg`. The two rules measured on one driver
(the split-K CU scaling and the int8 coopmat K-quant GEMM, both
`AMD_PROPRIETARY`) are ROWS with their measurements as provenance, their generic
defaults off — which is what S10 asks of them, reached here without a behaviour
change because the rows match exactly the devices the old driver check did.
Everything else is a derive function, because it follows from a capability
(`sg_pref`, the integrated-GPU `o4` rule) or is a plain constant.

Differences from the plan as written: `mmv_mw.warps` reaches no dispatch today
(`adapter::unified_mmv_row1` is always true, so the decode takes
`native_mmv_mrow` at rows=1 and `native_mmv_mw` is never dispatched; the plan
and the old docs said WARPS still affected Intel). The device facts S1 gathers
omit the required-subgroup-size stages, the compute workgroup limits,
`maxComputeWorkgroupSubgroups` and the cooperative-matrix shapes: no S1 rule
reads them, and S3's validation is where they are needed. Feature switches that
pick a tier on or off for A/B (`gemm_warp`, `mmv_o4`, `mrow`, `mla_sg`, …) stay
config switches, not parameters. Inline thresholds S1 did NOT migrate, because
no device rule chooses them today and each is a measurement away from mattering:
`MMV_MIN_ELEMS`, the `2 << 20` mmv_mw floor, `WIDE_GRID_MIN` (an occupancy
number like split-K's window), the `DENSE_SMALL_TILE_*` bands,
`DP4A_KQ_MIN_ROWS`, the split-K clamp of 8, and the integrated ubatch rows and
submit splitter (not kernel parameters); the flash tile's shared-memory pick is
a limit, not a choice. `docs/backlog.md` B-HWDET-NO-ARCH-BUCKET lists them.

**S9 protocol.** Quiet machine: run the baseline five times interleaved first
and abort if its spread exceeds 3%. Warm pipeline cache; discard the first
repetition; op microbenches interleave values in one process; model benches are
separate processes with a cooldown, candidates interleaved, never in blocks.
Decode GEMVs on tg128 at depth 0 and 4096 (Qwen3-1.7B Q4_K_M, SmolLM3-3B);
attention on gemma-3-1b tg128 at depth; DeltaNet on Qwen3.5-0.8B; prefill
parameters on pp512. Noise is the larger of the baseline's interquartile spread
and the difference between two independent baseline rounds; a value is better
only if its median improves by more than max(2 × noise, 1.5%) in two independent
rounds without regressing the other metric. A tuned value must also pass the
parity checks and the goldens on that device; a value that changes a golden is
rejected for existing devices.

## 6. Defaults for an unknown card

S1 migrates everything verbatim; S10 is where generic defaults move, after the
rows for the hardware they were tuned on hold the old values.

| Value today                                          | Tuned on                   | Generic default proposed                                              |
| ---------------------------------------------------- | -------------------------- | --------------------------------------------------------------------- |
| Pin 32 for subgroup kernels                          | capability                 | 32 if pinnable, else 64 with the emulation (today's rule)             |
| `sg_pref` (16 if the minimum is at most 16, else 32) | capability                 | keep                                                                  |
| `gemv_sg` Q6_K band and rows per group               | RDNA3 under RADV           | tier off; the RDNA3 row turns it on. Sweep first: rows, width         |
| `gemv.rm` and its band                               | RDNA3                      | the tree kernel **[I]**                                               |
| mmv WARPS (Q4_K 1, others 8)                         | RDNA3 for 1; Intel for 8   | a middle value; sweep first                                           |
| Flash and PV split formula; `ATTN_SPLIT`             | RDNA3's compute-unit count | derived from compute units, today's formula when the count is unknown |
| `split_k_plan` fill per compute unit                 | AMD's Windows driver       | the fixed window                                                      |
| Flash tile from shared memory                        | limit-derived              | keep                                                                  |
| `rmsnorm_wide`                                       | RDNA3, limit-gated         | the narrow build; the RDNA3 row widens it                             |
| `QUANT_Q8_BLOCKS_PER_WG`, `moe_small_m`              | RDNA3                      | keep (safe)                                                           |
| int8 cooperative-matrix K-quant prefill              | AMD's Windows driver       | off (already)                                                         |
| `attn_decode` on; per-key reduction width; workgroup | RDNA3                      | keep; sweep after S11                                                 |
| Integrated ubatch rows and the submit splitter       | integrated GPUs, watchdog  | keep (not kernel parameters; reported)                                |

Where the right value for an unknown card cannot be known, the conservative
choice is the one that is correctness-neutral and leaves RDNA3-specific fast
tiers off.

## 7. Risks, non-goals, what must not change

- **Must not change:** wave32 behaviour bit for bit on the RX 7900 XTX and the
  integrated GPU; the `wave64.glsl` emulation as the fallback; the
  `wave64_probe` startup check; decode equal to verify under any pin.
- **Non-goals:** cooperative-matrix shapes as free parameters; DeltaNet's chunk
  length (it changes numerics); a native 64-lane attention (the halves are the
  algorithm); spec-constant cluster sizes; converting class 5a without a
  measurement asking for it.
- **Risks:** a driver specializing to different machine code than the `-D`
  compile **[U]**; register-pressure cliffs at wider rows, which only the
  sweep's timing can see; the on-disk pipeline cache growing during sweeps;
  contributed rows nobody here can verify (marked, logged, and switchable off).
- **Gated on numbers from real GCN hardware (issue 94):** the GCN default for
  the decode GEMV width, native 64 against the emulation for `native_mmv_mrow`
  and the DeltaNet kernels, and whether GCN drivers list compute among the
  required-subgroup-size stages.
