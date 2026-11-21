# Configuring infr

infr has one configuration value, resolved once at startup from four layers and
then passed explicitly to the backends and sessions. Every knob is a typed field
on `infr_core::config::Config`; nothing reads the environment behind your back.

A commented starting point lives at [`infr.example.toml`](../infr.example.toml).

## Precedence

Four layers. **Later wins**, and a layer only overrides a field it actually
specifies — leaving `temp` out of your config file does not reset it to the
default, it just leaves the question to the layers below.

| Layer                  | Source                                 | Notes                                                |
| ---------------------- | -------------------------------------- | ---------------------------------------------------- |
| 1. Defaults (lowest)   | `impl Default for Config`              | The shipped behaviour.                               |
| 2. Config file         | TOML (see below)                       | Absent file = no-op. Malformed file = error.         |
| 3. Environment         | `INFR_*`                               | Same names as always — nothing was renamed.          |
| 4. CLI flags (highest) | `--dev`, `--ctx`, `--temp`, …, `--set` | A dedicated flag beats a `--set` for the same field. |

Two consequences worth knowing:

- **Every documented `INFR_*` variable still works.** The campaign that
  introduced `Config` changed _where the value goes_, not what you type. Your
  existing `INFR_PROF_OPS=1 infr bench …` scripts are unaffected.
- **A flag beats an inherited environment variable**, because the CLI layer sits
  above the env layer. `INFR_CTX=32k infr run … --ctx 8192` runs at 8192.

## The config file

### Lookup

The **first existing** file wins. There is no merging across files.

1. `--config <PATH>` — and it is an **error** if that path does not exist.
2. `./infr.toml`
3. `$XDG_CONFIG_HOME/infr/config.toml`, else `~/.config/infr/config.toml`

Finding no file at all is a no-op, never an error.

### Format

TOML. The section path is the struct path, so a key's full name is exactly what
`--set` takes: `[kernels.vulkan] flash_splits = 2` is
`--set kernels.vulkan.flash_splits=2`.

The file speaks the **positive** field names. Where the environment has an
`INFR_NO_*` disable-switch, the config has the thing being enabled, defaulting
to `true`:

```toml
[device]
dev = "Vulkan1"      # same grammar as --dev / INFR_DEV
ctx = "32k"          # the shared size grammar: 8192 / 256k / 50%

[kv]
type_k = "q8_0"
type_v = "q8_0"

[paging]
cache = "8g"         # force the paged expert cache with an 8 GiB budget

[kernels.vulkan]
flash_splits = 2
gemm_warp = false    # NOT `no_gemm_warp` — INFR_NO_GEMM_WARP's field, inverted

[multi]
pipeline = [0, 1]    # or ["Vulkan0", "Vulkan1"]

[serve]
max_tokens_cap = 8192
stats_interval_secs = 10  # throughput line every 10 s; 0 turns it off
```

Value grammars: booleans accept `true`/`false` (and `1`/`0`, `yes`/`no`,
`on`/`off` from `--set`); sizes take the shared `8192` / `256k` / `50%` grammar;
device lists take an array of indices or `VulkanN` strings; an `Option` field is
cleared with `""` or `"none"`.

### Unknown keys warn; wrong types fail

An unrecognized key — or a whole unrecognized section — is **warned about on
stderr and ignored**, with a did-you-mean:

```
[infr] config: unknown key `bogus` (ignored)
[infr] config: unknown key `kernels.vulkan.flash_split` (ignored) — did you mean `kernels.vulkan.flash_splits`?
```

That is deliberate: a config file written for a newer infr must not hard-fail on
an older binary, and removing a knob must not break everyone who has it in their
file. Typo protection comes from the warning line.

A **known** key given a value that does not parse into its type is a hard error:

```
Error: config `device.ctx`: expected a size/count like 8192, 256k or 50% (got "banana")
```

Note the asymmetry with the environment, which is frozen at today's behaviour:
`ctx = "banana"` in a file fails to load, while `INFR_CTX=banana` is silently
ignored and falls back. Seven environment keys do reject a bad value loudly, at
startup, on every subcommand: `INFR_SG`, `INFR_SUBMIT_DISPATCHES`,
`INFR_GPU_WAIT_TIMEOUT_MS`, `INFR_PIPELINE`, `INFR_TENSOR_PARALLEL`,
`INFR_EXPERT_PARALLEL`, and `INFR_TUNE` (see
[Vulkan kernel tuning](#vulkan-kernel-tuning)).

### Diagnostics announce themselves

If the **file** turns on a `prof.*` or `debug.*` knob, infr prints one line at
startup naming the file and the fields:

```
[infr] config: /home/you/.config/infr/config.toml enabled diagnostics: prof.ops
```

So "why is my server printing timings" is answerable from the command line, even
when the cause is a global config file you forgot about.

## `--set`

Most knobs have no dedicated flag. `--set <config.path>=<value>` reaches all of
them, using the same path grammar as the TOML file:

```bash
infr bench "$M" -p 512 -n 0 --set kernels.vulkan.flash_splits=2
infr run "$M" "hi" --set kv.type_k=q8_0 --set kv.type_v=q8_0
```

- The path is the **config path**, never the `INFR_*` name. They are not 1:1:
  `INFR_NO_GEMM_WARP` is `kernels.vulkan.gemm_warp=false`, `INFR_NO_GEMV_REG`
  and `INFR_GEMV_VARIANT` are both `kernels.vulkan.gemv.variant`, and
  `INFR_MMV_MW` is tri-state.
- An **unknown** path is a hard error with a did-you-mean (unlike the file layer
  — you typed it for this run, so silently ignoring it would give you a wrong
  answer with no second chance to notice):
  ```
  Error: unknown config path `kernels.vulkan.flash_split` — did you mean `kernels.vulkan.flash_splits`?
  ```
- The **same path twice** is an error, not a silent last-wins:
  ```
  Error: `--set kv.slots=` given more than once
  ```
- `--set` is **additive** to the dedicated flags, and loses to them. Passing
  both prints a warning naming the field:
  ```
  $ infr run "$M" "hi" --ctx 4096 --set device.ctx=8192
  [infr] config: `--set device.ctx=8192` ignored — the dedicated flag for `device.ctx` wins
  ```

### The dedicated flags

`--config` and `--set` are global. The rest are on `run` / `serve` (device flags
also on `bench`); `infr <cmd> --help` is the authority.

| Flag                     | Config path         | Env             |
| ------------------------ | ------------------- | --------------- |
| `--dev`                  | `device.dev`        | `INFR_DEV`      |
| `--ctx`                  | `device.ctx`        | `INFR_CTX`      |
| `-u` / `--ubatch`        | `device.ubatch`     | `INFR_UBATCH`   |
| `-t` / `--threads`       | `device.threads`    | —               |
| `--temp`                 | `sampling.temp`     | `INFR_TEMP`     |
| `--top-k`                | `sampling.top_k`    | `INFR_TOP_K`    |
| `--top-p`                | `sampling.top_p`    | `INFR_TOP_P`    |
| `--seed`                 | `sampling.seed`     | `INFR_SEED`     |
| `--max-new`              | `sampling.max_new`  | `INFR_MAX_NEW`  |
| `--no-think` / `--think` | `sampling.no_think` | `INFR_NO_THINK` |

`device.threads` has no `INFR_*` twin — it is published as `RAYON_NUM_THREADS`,
because rayon's global pool has no other input.

## What is tunable

The authority is
[`crates/infr-core/src/config/manifest.rs`](../crates/infr-core/src/config/manifest.rs):
every `INFR_*` key, the config path it lands on, and its value grammar, in one
table that the tests check against the tree. `Config::all_paths()` is the full
path list. What follows is the shape plus the knobs people actually reach for.

**`[device]`** — which GPU (`dev`), how much context (`ctx`), the prefill
micro-batch (`ubatch`, `ubatch_parallel`), CPU `threads`, and three low-level
device knobs (`submit_dispatches` for the iGPU submit splitter, `subgroup_pref`
to force subgroup 16 or 32, and `gpu_wait_timeout_ms`, below).

**`device.gpu_wait_timeout_ms`** (`INFR_GPU_WAIT_TIMEOUT_MS`, default 120000) is
the longest the Vulkan backend waits for one piece of submitted GPU work. A wait
that runs past it, or meets a lost device, is an error naming the submit it
waited on (its dispatch count and first and last kernel), how long it waited,
the fence's state and, on a lost device that offers `VK_EXT_device_fault`, the
driver's fault report. The backend is then unusable: every later call returns
the same error without touching the device, and teardown leaks rather than frees
what the GPU may still be using. `infr run` prints it and exits non-zero;
`infr serve` fails that request and every later one on that model with the same
error, so restart the server. `0` waits forever (for debugging). Any other value
below 5000 is refused when the backend is created: that floor is a guard against
a typo (a value meant in seconds), NOT a safe setting, since legitimate waits
far longer than 5 s have been measured. One wait can cover several pipelined
forward segments, so heavy chunks on a slow device make long waits. Measured
with a temporary log (not in the tree) of every successful wait:

| Device               | Workload                                                               | Longest wait         |
| -------------------- | ---------------------------------------------------------------------- | -------------------- |
| Ryzen integrated GPU | `infr bench` SmolLM3-3B Q4_K_M `-p 2048 -u 2048`, 2 runs               | 12.6 s (next 10.5 s) |
| Ryzen integrated GPU | the full GPU gate (light workloads), 1 run                             | 2.75 s               |
| RX 7900 XTX          | `infr bench` SmolLM3-3B Q4_K_M `-p 2048 -u 2048`, 2 runs               | 0.35 s               |
| RX 7900 XTX          | the gate's paged tests (`test(paged)`, 44 tests, paged MoE among them) | under 0.2 s          |
| RX 7900 XTX          | the full GPU gate, 1 run                                               | under 0.1 s          |

The integrated GPU's 12.6 s (2026-11-19) leaves the 120 s default about 9.5
times headroom; a larger model or chunk on that GPU waits longer in proportion,
so raise the bound for such runs rather than lower it (a bound too low fails
work that would complete; one too high only delays the error of a rare hang).
The gate figures are from 2026-11-18. Before this bound a submit whose work the
GPU dropped hung the process forever with one core spinning
(`docs/plan/12-low-capability-vulkan.md`). What the backend can and cannot
detect about dropped work is in `docs/backlog.md` B222.

**`[sampling]`** — `temp` (0 = greedy), `top_k`, `top_p`, `seed`, `max_new`,
`ignore_eos`, `no_think`. **Provenance matters here**: `infr run` / `infr serve`
fill `temp` / `top_k` / `top_p` / `max_new` from the model's own recommended
sampling (an arch-family table, then any `generation_config.json` beside the
model, then the GGUF's own `general.sampling.*` keys, the last one winning) only
for knobs that **no layer specified**. Putting `temp` in your config file pins
it and suppresses that fallback. The `Config` defaults, which library callers
get, are greedy: `temp = 0.0`, `top_k = 20`, `top_p = 0.95`, `max_new = 2048`.
On `serve` these are the server defaults — a per-request OpenAI
`temperature`/`top_p` still overrides them.

**`[kv]`** — cache element format (`type_k` / `type_v`, plus the legacy
`force_q8` alias), prefix-cache `slots`, the sliding-window `ring`, and the
host-overflow trio (`overflow`, `overflow_vram_mb`, `overflow_reserve_mb`) that
spills KV to system RAM when VRAM runs out.

**`[paging]`** — the MoE expert cache and dense layer streaming: `cache` sizes
the paged VRAM budget (and forces paging even when the weights would have fit),
`ring` overrides the upload staging ring, `stats` prints per-pool
hit/miss/eviction counts, and `trace` (`INFR_PAGER_TRACE`, a file path) records
every pool's pager calls to that file, one file for the whole run, in the format
`infr_core::pager::trace` documents. A trace replays to exactly the counts
`stats` prints; `infr-core`'s `pager_bench` test replays it, times it and
compares today's eviction policy with Belady's (a lower bound for that request
sequence when the trace has no probes). Unset, tracing costs nothing measurable.

**`[kernels]`** — two backend-independent graph-shape gates (`qkv_fuse`,
`gated_rmsnorm`) plus one sub-section per backend. Everything under them is a
kernel **tier** override: the engine picks the best tier the device supports,
and these exist to force one off when bisecting a correctness or perf problem.

- **`[kernels.vulkan]`** (the biggest section): capability masks (`coopmat`,
  `f16`, `i8_dot`, `coopmat_8x8`, `i8_coopmat`), the `force_wave64` testing knob
  (runs the wave64 path that AMD GCN takes on a device that can pin 32),
  GEMM/GEMV tiers (`gemm_warp`, `mmq`, `mmv`, `mrow`, `moe_small_m`, the
  `[kernels.vulkan.gemv]` sub-table), attention (`flash_warp`, `flash_splits`,
  `flash_min_rows`, `pv_splits`), DeltaNet (`dn_chunk_scan`, `dn_chunk`,
  `dn_split`), and the plumbing switches (`push_desc`, `pipeline_cache_disk`,
  `no_replay`, `no_vram_guard`, `gfx_queue` — stay on the compositor's graphics
  ring instead of a compute-only one — the BDA chunk caps). Note `f16 = false` also
  disables coopmat regardless of `coopmat`, matching `INFR_NO_F16`'s effect
  today. The hardware-shaped numbers (split counts, thresholds, rows per
  workgroup, subgroup pins) are per-device tuning parameters, set under
  `[kernels.vulkan.tune]`; see the next section.
- **`[kernels.metal]`** — the Apple backend's native/CMM/RT kernel families per
  dtype, plus `deltanet`, `moe` and `pipeline_cache` (persist the compiled
  `MTLComputePipelineState`s as an `MTLBinaryArchive` under `~/.cache/infr`, so
  a launch does not re-run the driver's AIR → GPU-ISA back end for every kernel;
  the MSL → AIR front end is not cacheable and still runs every launch). Like
  `kernels.cpu.reference` it has no `INFR_*` twin.
- **`[kernels.cpu]`** — `spin` (spin-pool idle ceiling), `spinpool`,
  `repack_mb`, `reference` (the bit-reference kernel path, which has no `INFR_*`
  twin and never had one), and `core_spread` (keep the spin pool's threads, and
  the thread running the graph, on one logical processor per allowed core so no
  two share a core's SMT siblings; only when the threads fit one per allowed
  core; Windows only, no `INFR_*` twin), and `max_tier` (`INFR_CPU_MAX_TIER`:
  the widest SIMD tier the CPU backend's kernels and `infr_core::math`'s slice
  functions may run, `scalar`; `avx2`, `avx512` or `avx512vnni` on x86-64;
  `dotprod` on aarch64, instead of the widest the CPU has; every tier gives the
  same bits, so it changes only speed. On a build without FMA (the released
  binaries), `avx2` and up also let the f32 dots run their FMA code where the
  CPU has FMA and F16C; `scalar` runs them through the C runtime's `fmaf`. A
  ceiling, not a demand: one above what the CPU has runs the CPU's widest, and
  so does another architecture's tier (every name loads on every architecture,
  so one file serves both; the backend logs a warning). It is a process-wide
  debug switch: every CPU backend applies it when built, unset meaning no
  ceiling, so the last backend built decides for the whole process. A misspelt
  name is a config error at load; see [`math.md`](math.md)).

### Vulkan kernel tuning

Every hardware-shaped number a Vulkan kernel choice depends on — a split count,
a size threshold, rows per workgroup, a pinned subgroup width — is a **tuning
parameter**, named `family.param` (and `family.param@qualifier` where it varies
by quant format, e.g. `mmv_mw.warps@q4_k`). `infr tune list` names every one,
with its range, its default rule and its value on your device; `infr tune show`
prints the device, the key the table matched it by, the matched table rows, and
each value with the level it came from:

```
$ infr tune show --dev Vulkan0
device:  AMD Radeon RX 7900 XTX (AmdRdna3, AMD_PROPRIETARY)
match:   vendor=amd arch=AmdRdna3 driver=AMD_PROPRIETARY integrated=false device_id=0x744c cu=96 subgroup=32-64 driver_version=0x800161
hash:    7751091e
rows:
  amd-proprietary-split-k [class row, maintainer] sets split_k.scale_by_cu
  amd-proprietary-i8cm-kq [class row, maintainer] sets gemm_i8cm_kq.enabled

device.sg_pref              32     default
gemv_sg.rows                2      default
...
```

`--json` prints the same as a document. The `hash` is also in the device banner
(`tune:7751091e`), so a log says whether two runs tuned alike.

One parameter is not a kernel choice: `submit.initial_dispatches`, the submit
splitter's cap before any forward has been measured (`0` = one command buffer
per forward). Its default is 128 on an integrated GPU, 64 on a discrete device
that is wave64-only or has neither 16x16x16 f16 cooperative matrices nor an
accelerated int8 dot (the device facts `matrix_units` and `dot_accelerated` in
`infr tune show --json`), and 0 otherwise. A device that reports neither fact,
an unknown card included, starts at 64: splitting a fast card's first forward
costs a few submits, while not splitting a slow one's lost the device (issue
94). The default spares single-token forwards on a discrete card; a value set at
any other level, or `device.submit_dispatches`, applies to every forward.

Each value is resolved once, when the backend is created, from these levels —
the last one that sets a parameter wins, per parameter:

1. the **generic default**, derived from the device's facts;
2. **class rows** of the built-in table
   (`crates/infr-vulkan/src/tuning/table.toml`) — a vendor plus an architecture,
   driver, integrated flag, subgroup range or compute-unit range;
3. **device rows** of that table (PCI device ids, or a name);
4. your **tuning file**, `kernels.vulkan.tuning.file` / `INFR_TUNE_FILE`: rows
   in the same schema, so a row can be tried without rebuilding;
5. the **config file**, `[kernels.vulkan.tune]`;
6. the **environment**, `INFR_TUNE="family.param=value,…"`;
7. the **command line**, `--set kernels.vulkan.tune.family.param=value`.

```toml
[kernels.vulkan.tune]
"gemv_sg.rows" = 4
"mmv_mw.warps@q4_k" = 2

[kernels.vulkan.tuning]
rows = false              # INFR_TUNE_NO_ROWS: generic defaults only, no table rows
disable = ["amd-proprietary-split-k"]   # INFR_TUNE_DISABLE: drop rows by id
```

A value you set that is not a parameter, is out of range, or targets a `fixed`
parameter (one that must equal a literal compiled into a shader) refuses the
device with an error naming where it came from. A range is what the code after
it can hold: the attention split counts stop at 8 because the partials scratch
holds 8, `gemv.variant` takes only the names that have a build (`infr tune list`
prints them). A built-in row value that fails validation is dropped with a
warning instead, so a bad row cannot brick a device. Within one level a
qualified setting (`@q4_k`) beats an unqualified one. A tuning-file row whose
`driver`, `arch` or `vendor` is not a known name is an error (a typo would
otherwise match nothing); `name` is a free substring and cannot be checked.

**`rows = false` means generic defaults only, everywhere.** It drops every
built-in row, including the two the AMD proprietary driver's GEMM choices come
from: on that driver it turns off the int8 cooperative-matrix K-quant prefill
GEMM and the compute-unit-scaled split-K window, so prefill takes the f16
coopmat GEMM — about a third slower pp512 on the RX 7900 XTX (11.9k against 7.5k
t/s on Qwen3-1.7B Q4_K_M when that GEMM landed). `infr tune show` says
`rows: off` when it is set. To drop just one row, use `disable` with its id.

**`tuning.file` paths.** A relative path in a config file is relative to that
config file's folder; from `INFR_TUNE_FILE` or `--set` it is relative to the
working directory.

**What the device can use wins over what was asked.** A `device.sg_pref` of 16
on a device that cannot pin a 16-wide subgroup runs at 32, as it always has;
`infr tune show` reports 32 with the reason, and the banner hash is of 32.

**The older knobs are aliases.** `device.subgroup_pref` (`INFR_SG`),
`kernels.vulkan.flash_splits`, `pv_splits`, `flash_min_rows`, `flash_bm32`,
`mmv_mw_warps`, `moe_small_m`, `canvas_chunk_n`, `i8cm_kq` (`INFR_NO_I8CM_KQ`)
and the whole `[kernels.vulkan.gemv]` group keep their names, grammars and
meanings — including their old clamps and fallbacks — and write the matching
parameter at the level that set them (`infr tune show` labels it, e.g.
`env alias INFR_FLASH_SPLITS`). At the same level the new `tune` key wins over
the alias; across levels the higher level wins whichever spelling it uses. An
alias set to a value that has no effect — its unset spelling
(`--set kernels.vulkan.mmv_mw_warps=none`), `i8cm_kq = true`, `no_rm = false`, a
`gemv.variant` name with no build — resets the parameter at its level to what
the defaults, rows and tuning file give, so it still beats a lower level's
`tune` key. Two changes to old spellings, both where the old value was unsafe or
meaningless: `INFR_FLASH_SPLITS` / `INFR_PV_SPLITS` above 8 are now refused
(they wrote split partials past the end of an 8-split scratch buffer); 0, an
unparseable value and one past 2³²−1 still mean "the policy", as before. And an
`INFR_GEMV_VARIANT` naming no build, the empty value included, still takes the
RM path, now shown as `none`. A `Config` built in code (no layers) cannot say
whether an alias field equal to its default was set, so such a field counts as
unset; through the file, environment or command line, setting a knob to its
default value counts as set.

**`[spec]`** — MTP / speculative decode: `mtp`, `k`, `decode_chain`, `draft` (a
draft-model path), the GPU-side sampling steps (`gpu_argmax`, `gpu_sample`,
`gpu_embed`, …). MTP is currently parked; see the README.

**`[multi]`** — multi-GPU splits: `pipeline`, `tensor_parallel` (both need ≥ 2
devices) and `expert_parallel` (≥ 1); too few devices is a hard error. The three
`*_p2p` flags choose GPU-to-GPU transport (`true`, the default) over staging
through host RAM.

**`[prof]`** — every key is `INFR_PROF_*`, and every one is settable three ways
(env, `--set prof.<name>=…`, or a `[prof]` section in the file):

| knob                | what it does                                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `ops`               | per-op device profiling on EVERY backend — vulkan, metal, cpu                                                           |
| `op_shapes`         | itemize per-op labels by shape rather than folding a kind into one row (vulkan)                                         |
| `stages`            | host-side stage timing: throughput, decode setup-vs-execute, prefill build/compile/execute, MTP verify, diffusion steps |
| `vram`              | log live VRAM in use after weight load                                                                                  |
| `out`               | ALSO write the exit report as JSON to this path                                                                         |
| `diffusion_trace`   | per-step schedule/entropy trace for diffusion models                                                                    |
| `metal_device_time` | `off` / `flush` / `counters` — how Metal obtains per-op device time                                                     |
| `metal_debug`       | extra Metal profiler output                                                                                             |

Two consolidations are worth knowing if you have old scripts. Per-op profiling
answered to a different knob on each backend (`INFR_PROF2` on vulkan,
`INFR_PROF_OPS` on cpu, `INFR_METAL_PROFILE` on metal) — it is all `ops` now.
And host-side stage timing was five knobs, one per pipeline (`INFR_PROF`,
`INFR_PROF_DEC`, `INFR_PROF_PF`, `INFR_MTP_TIME`, `INFR_DIFFUSION_TIME`) — it is
all `stages`. Old spellings were dropped cleanly and are simply no longer read.

**`[serve]`** — `api_key` (bearer token; an **empty** value means no auth — it
gates `/v1/chat/completions`, `/v1/models`, `/metrics` and `/status`, never
`/health`), `max_tokens_cap`, `request_timeout_secs` (per-request wall-clock
deadline in seconds; `0`, the default, means no deadline — a deadline truncates
a legitimate slow reply, so it is opt-in), and `stats_interval_secs`
(`INFR_SERVE_STATS_SECS` — how often the server logs its throughput line,
default `5`; `0` switches the line off). Per-request sampling is not here — it
stays on the request. Three more keys shape the HTTP surface:

| key                  | env                             | what it does                                                                                                                     |
| -------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `allowed_hosts`      | `INFR_SERVE_ALLOWED_HOSTS`      | host names a request's `Host` header may carry (comma-separated in the environment, a list in the file); `*` turns the check off |
| `sse_keepalive_secs` | `INFR_SERVE_SSE_KEEPALIVE_SECS` | how long a streamed reply may stay silent before the server sends an SSE comment line; `0` sends none                            |
| `max_body_bytes`     | `INFR_SERVE_MAX_BODY_BYTES`     | the largest request body accepted; a larger one is a `413`                                                                       |

**The `Host` check is DNS-rebinding protection.** A web page whose hostname
re-resolves to `127.0.0.1` is same-origin with a local server and could drive
it, but its requests still carry the page's own hostname. With `allowed_hosts`
unset, the check guards only a server with no API key bound to a loopback
address: it accepts `localhost`, any `*.localhost` name, `127.0.0.1`, `[::1]`
and the bind host, on any port (so an SSH or container port forward works), and
a request with no `Host` at all, which no browser sends. Anything else is a
`403` whose body names this key. A server with an API key, or one bound to a
non-loopback address, is not checked unless the key is set; once set, the listed
names are allowed in addition to the loopback ones and the check always applies,
which is the reverse-proxy and custom-hostname case. An entry that could never
match (a URL, a path, a `*.x` pattern) stops the server at startup. `/health` is
never checked.

**`response_format`** is enforced, not ignored: `json_object` constrains the
reply to a JSON object and `json_schema` to the given schema, streamed or not,
on every path that can force a tool call (Vulkan at any `--parallel`, CPU, a
Metal session). A path that cannot constrain (DiffusionGemma, a Metal
speculative draft) answers `400`, as does a schema the grammar engine rejects.
Thinking is switched off for such a request. Numbers the schema leaves unbounded
are limited in length so a reply cannot run digits to `max_tokens`, and exponent
form is not produced; strings and nesting are not bounded.

The throughput line is **activity-only**: an interval in which nothing happened
emits nothing, so an idle server leaves a clean log and there is no heartbeat to
mistake for load. Its `prefill_tps`/`decode_tps` are the whole server's tokens
divided by the WALL time of that one interval — not cumulative, and not the same
number as the per-request `prefill_tps`/`decode_tps` on the `request done` line,
which are that one request's own speeds (`prompt_tokens / TTFT` and
`gen_tokens / (total - TTFT)`). Request logging carries **counts only, never
prompt text**.

**`[hub]`** — model acquisition (`infr pull`, and the auto-pull `infr run` /
`infr serve` do when a model is missing). One knob: `pull_jobs`
(`INFR_PULL_JOBS`, default `8`) — how many CONNECTIONS one model's download may
use at once. It is the CONNECTION that is slow, not the link: measured against
the same CDN objects, one connection sustained 8.8 MB/s and five sustained 78.7
MB/s — 15.7 MB/s each, so the per-connection rate rose rather than fell.

What the connections are spent on depends on the model, and the setting does not
have to say: a split model is `-NNNNN-of-MMMMM` shards fetched one file per
connection, while a model shipped as one file is split into byte ranges instead
and reassembled (on `unsloth/DeepSeek-V3.2-GGUF`'s single 161 GB `UD-TQ1_0`,
same 60-second window: 11.4 MB/s at `1`, 80.5 MB/s at the default `8`; the whole
161 GB pull ran at 80.0 MB/s end to end). One number covers both because a bound
per axis would multiply — 8 files x 8 ranges is 64 sockets on a repo whose file
count is the publisher's choice (`DeepSeek-V3.2-REAP` ships 236).

Ranges need the server to support them; one that does not (no `Accept-Ranges`,
or a `200` where a `206` was asked for) falls back to a single stream, and a
file of 64 MiB or less is never split at all. `0` and `1` both mean strictly one
connection — the setting for a metered link, or for a proxy that objects to
several connections from one client.

**`[debug]`** — poison/barrier/dump switches: `coopmat` (print the enumerated
and chosen coopmat shapes — useful on Intel Arc), `bda_chunk`, `wide_dispatch`,
`chat`, `moe_counts`, `moe_counts_dump`, `poison_uninit`, `no_barrier`,
`full_barrier`, `cpu_op_hash` (`INFR_CPU_OP_HASH=<path>`: the CPU interpreter
appends a hash of every tensor each op reads and writes, for diffing two runs to
the first op that differs).

## The `INFR_*` keys that are deliberately NOT config

Four keys keep reading the environment directly, for reasons a runtime `Config`
cannot fix:

| Key                        | Why                                                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `INFR_PROFILE`             | **Build-time** input: read by `build.rs` in core/cpu/gguf/llama/vulkan to set `cfg(infr_profile)`. No runtime value exists when it is read. |
| `INFR_TEST_GGUF`           | Test fixture: points `infr-gguf`'s tests at a `.gguf` on disk.                                                                              |
| `INFR_TEST_MODEL`          | Test fixture: overrides the HF-cache lookup for the model-backed tests.                                                                     |
| `INFR_LLAMA_DIFFUSION_CLI` | Dev fixture: points `infr compare` at a `llama-diffusion-cli` binary on disk.                                                               |

Two related notes. `INFR_DIFFUSION_VISUAL` is also not a `Config` field — since
it only steers CLI presentation it became the plain flag
`infr run --diffusion-visual`, whose clap `env =` fallback keeps the old
spelling working. And `INFR_CPU` / `INFR_METAL` are **dead**: they were removed
as backend selectors with no aliases. Use `--dev cpu` / `--dev metal`, or
`INFR_DEV=cpu` / `INFR_DEV=metal`.

## Library callers

`Config` is a value, not a global. Build one and hand it out:

```rust
use infr_core::config::{Config, ConfigOverrides};
use std::sync::Arc;

// The full four-layer resolve (file + env + the CLI overrides you pass).
let cfg = Arc::new(Config::load(&ConfigOverrides::default())?);

// Or construct exactly what you want — no environment, no file, no ordering hazard.
let cfg = Arc::new(Config { kv: infr_core::config::KvCfg { slots: 8, ..Default::default() },
                            ..Default::default() });
```

Backends take it at construction (`VulkanBackend::new_with(cfg)`,
`Backend::new_with(device_id, cfg)`, `CpuBackend::new_with(cfg)`), and
`Config::load_from_env()` is the defaults-plus-environment fold for a caller
that wants the environment honoured but no config file.

The migration history — the layer machinery, the per-knob polarity tables, and
the slice-by-slice record — lived in the campaign's planning doc while the
migration was in flight; it was deleted at commit `3010e45` once the migration
landed, so that record now exists only in `git log`. What actually moved is
tracked today in
[`crates/infr-core/src/config/manifest.rs`](../crates/infr-core/src/config/manifest.rs).
