# infr-metal — Apple GPU backend architecture

State as of 2026-07-03 (PRs #4-#14 merged, #15 = native legacy quants + replay
concurrency + counter profiler). `crates/infr-metal` implements the same
`Backend` seam as `infr-cpu` and `infr-vulkan` on Apple's Metal API. It started
as a correctness-first reference (per-op command buffers, dequant-to-f32
everything) and was rebuilt profiling-first; this doc records the architecture
that emerged and the measured reasoning behind it.

## Numbers (M3 Pro 18-core, `infr compare --sweep --dev MTL0`, same GGUFs)

| model | pp512 | tg128 | tg64@d4096 |
| --- | --- | --- | --- |
| Qwen3-0.6B Q4_K_M | 0.89× | 0.98× | 1.00× |
| Qwen3-4B Q4_K_M | 0.93× | 1.00× | 0.96× |
| Qwen3-0.6B Q8_0 | 0.84× | 0.97× | 0.99× |
| Qwen3-MoE-2.4B Q4_K_M | 0.87× | 0.96× | 0.99× |
| gemma3-1b Q4_K_M | 0.84× | 0.96× | 0.92× |
| TinyLlama-1.1B Q4_0 | 0.85× | **1.02×** | **1.02×** |

(ratios = infr / llama.cpp; >1 = infr ahead.) **Decode is at parity.** The
prefill band decomposes by ablation (strip the dequant scale math: −2.4% of
GEMM time; strip ALL weight loads + decode: −9%) to a tile floor of ~6.9
TFLOPS — ~53% of the f16 `simdgroup_matrix` peak and about llama.cpp's own
effective rate — so what remains there is MMA/tile scheduling margin, not an
addressable component. Decode was weight-stream + launch-latency bound; the
levers that closed it are recorded below in the order the profiler surfaced
them.

From the naive reference implementation: decode ~44×, prefill ~250×+. Model
load is ~10× faster than the repack era and resident weight memory is the raw
mmap only (−3 GB on 4B).

## Execution model: device residency, one command buffer per forward

`exec::Resident` tracks, per graph tensor, whether the current value lives in
the host mirror (`vals`) or a device buffer (`dev`/`loc`). GPU ops encode into
one shared command buffer (lazily opened, Metal automatic hazard tracking
orders intra-batch dependencies) and the batch is committed+waited only at a
host-side op or the final write-back. A decode forward is exactly one
commit+wait; the naive per-op barrier version spent ~90% of wall in ~75k
command buffers per run.

Measured floors that shape everything else (probe tests in
`crates/infr-metal/tests/dispatch_overhead.rs`):

- dispatch + hazard barrier inside a batch: ~1.2 µs/kernel → op-count fusion is
  NOT a lever (a fused Add+RmsNorm measured nothing and was removed);
- device compute-read roofline: ~133 GB/s (spec 150);
- quantized GEMV achieved: ~90-105 GB/s (probe `tests/gemv_bw.rs`).

## Weight forms (quantized `Op::Linear`)

Chosen per dtype by `weight_qui`, cached per bound buffer for the backend's
single-generation lifetime:

- **Native GGUF blocks — Q4_K, Q6_K** (the formats real checkpoints ship): the
  bound weight buffer is used AS-IS. No host repack, no extra residency;
  kernels decode raw block bytes in-place (`DEC16_Q4K`/`DEC16_Q6K`, one
  16-element sub-block per call; `get_scale_min_k4` ported), and the 32-element legacy formats Q8_0 (34 B),
  Q5_0 (22 B) and Q4_0 (18 B) — every format real checkpoints ship. ~4.5-8.5 bpw
  on the wire. Two hard-won details: the Q6_K sub-block selector must be
  branch-free (a 4-way `if` serializes the simdgroup — measured 2.3× slower),
  and Q6_K blocks are 210 B (2-aligned) so only byte/ushort loads are legal,
  while Q4_K's 144 B blocks allow uint loads.
- **Factored form — every other affine quant** (`infr_gguf::dequant_factored`,
  the single decoder that `dequant_unified` now expands): bit-packed 4/6/8-bit
  codes + one i16 `(sc, m)` pair per 16 elements + one f16 `(d, dmin)` pair per
  quant block, so `weight = (d·sc)·code + (dmin·m)` reproduces the reference
  dequant bit-for-bit (same f32 products; the sign/power-of-two factors folded
  into `m` are exact scalings — Q6_K's `m = −32·sc` is why i16).
- Float weights (f16/bf16/f32) dequant to a cached f32 device buffer.

## Quantized Linear kernels: one decoder, three shapes

Per-format `DEC16_*` macros decode the 16-element block with global index `bi`
into `wk[16]`; three kernel shapes instantiate each format:

| shape | route | precision |
| --- | --- | --- |
| `linear_*` GEMV | m == 1 (decode) | exact f32 (bit-for-bit dequant, f32 dot) |
| `linear_*_ks` k-split GEMV | m == 1, row groups ≤ 4096 AND k deep enough to feed 4 simdgroups twice | exact f32, reassociated |
| `linear_*_rt` row-tiled | 1 < m < 16, or out_f % 16 ≠ 0 | exact f32 |
| `linear_*_cmm` cooperative GEMM | m ≥ 16 && out_f % 64 == 0 | f16 operands, f32 accumulate |
| `linear_*_hmm` per-simdgroup GEMM | m ≥ 16 && out_f % 16 == 0 (fallback) | f16 operands, f32 accumulate |

The cooperative GEMM is the mul_mm shape (ported from llama.cpp and adapted to
the DEC16 decoders): a 64-output × 32-token tile per 128-thread threadgroup,
NK=32. The load-bearing piece is the **threadgroup-memory layout**: staged
operands are contiguous 8×8 half tiles (weights written pre-transposed), so
every `simdgroup_load` is stride-8 and conflict-free — tile size, barrier
count, and decode width were each probed separately and none of them was the
gap. Two more mul_mm details each measured real wins later: the device reads +
dequant issue BEFORE the barrier that drains the previous iteration's MMA
(latency-hides behind compute, +5% 4B pp2048), and the DEC16 decoders use the
reference dequantize shapes — in-place high nibble with the /16 folded into
the scale (Q4_K), uint32-lane ql/qh combines (Q6_K) — bit-identical values,
~+4% more. Each thread dequantizes exactly one 16-block per K-step (128 threads = the
whole weight tile) and x is cast f32→f16 inline during staging. Each simdgroup
owns a 32×16 quadrant as 8 f32 accumulators; partial token tiles stage through
threadgroup memory with a row-guarded copy-out so every tile takes the MMA
path.

Precision policy: **decode is algebraically identical to `dequant_block`, FP
reassociation only** (the mul_mv-shape GEMV reconstructs the exact dequant
values but reorders the f32 dot's summation; parity holds at 1e-4). Prefill
GEMM rounds operands to f16 (~5e-4 relative, well under quantization error —
the llama.cpp trade); its parity tests compare against a reference that rounds
operands identically, keeping the tolerance at 1e-3 instead of testing nothing
with a loosened bound.

## Attention routing

All scalar variants share `AttnParams` and an online-softmax core; routing is
by launch shape because the scalar kernels are latency-bound on their serial
O(kv_len) chain:

- **`attnflash2_f16kv` (hd 64/128)** — the llama.cpp `kernel_flash_attn_ext`
  structure for wide launches: NSG=4 simdgroups cooperate on ONE (8-query
  tile, head) threadgroup over C=64 KV positions per iteration, each phase
  split along a different axis (QK^T by KV columns with K^T direct from the
  f16 cache; softmax by query rows, all 32 lanes on float2 scores, M/S stats
  pinned in the owning simdgroup's registers; P·V by output columns with O
  fragments held in registers). Analytic causal/window masking, zero-padded
  partial query tiles. Templated over compile-time head_dim (hd 64/128/256
  instantiations — the fully-unrolled loops were worth +38% alone; hd=256 is
  gemma, whose absence was a 3× prefill crater); NSG=8 benched slower.
- **`attnflash_f16kv`** — the single-simdgroup flash kernel, kept for
  hd % 8 == 0 shapes outside the hd 64/128 instantiations: one simdgroup per
  (8-query tile, head), K^T/V 8×8 fragments direct from the f16 cache, scalar
  masked online softmax per 16-position KV block, diagonal-MMA row rescale.
  An earlier all-f32 flash kernel lost to the scalar split (it staged K/V
  through 8 KB threadgroup tiles); zero staging is what makes these win. Tail
  blocks may read up to 7 rows past the causal limit — always inside the
  bound cache buffer — and are masked.
- **`attnvec_f16kv` (hd 64/128)** — the llama.cpp `kernel_flash_attn_ext_vec`
  structure for long-context decode (kv_len ≥ 128; hd 64/128 at NSG=32,
  hd=256 at NSG=16 — its per-simdgroup O partials are NSG·hd·4 B and 32
  simdgroups would blow the threadgroup budget): NSG simdgroups per
  (query, head), each owning interleaved 32-position KV blocks with a private
  online softmax — 4 KV rows × 8-lane dots per pass, shuffle-tree reduce, ONE
  simd_max/simd_sum per block (the split kernels below run that chain once
  per position and are latency-bound on it) — merged once at the end by a
  log2 tree. Q stays f32 (parity 1e-4); tail rows clamp into the cache and
  mask in the softmax.
- **`attnsplit32_*` / `attnsplit_*`** — narrow launches and long contexts on
  head sizes without a vec instantiation: NSG (32 or 8) simdgroups per
  (query, head), each owning a strided KV slice with a private online
  softmax, merged in threadgroup memory.
- **`attention_*`** — the lean one-simdgroup-per-(query, head) kernel for
  short-context leftovers.

**Per-pipeline thread-cap gating**: `maxTotalThreadsPerThreadgroup` is
per-PIPELINE (register pressure or a paravirtual device — GitHub's macOS CI —
can push it below a kernel's requirement) and `encode_tg` clamps silently,
which for these kernels means skipped KV slices and garbage merges. Every
wide-threadgroup route checks its own pipeline's cap and degrades to the
next-narrower kernel.

## Q8_0 KV cache (`INFR_KV_Q8=1`)

Opt-in: both caches stored as Q8_0 blocks (34 B / 32 elems) — **half the f16
footprint and bandwidth**, so 16k-context sessions fit machines the f16 cache
would swap on. Write quantization is byte-identical between the CPU reference
and `writekv_q8` (d = amax/127 as f16, q = rint(x/d)); reads dequantize
exactly. Prefill rides `attnflash2_q8kv` — the cooperative flash structure
with a dequant-staging stage (all 128 threads dequantize each 64-position KV
block into a 24 KB threadgroup tile: K pre-transposed [hd][64] for
conflict-free fragment loads, V staged into the same tile during the softmax
phase). Decode at depth rides `attnvec_q8kv`; the q8 decode replay tape
records like f16. Throughput is model-dependent — the q8 read path trades
bandwidth for ALU: 0.6B tg128@d16384 62.7 t/s (+26% over the f16 cache, +35%
over llama.cpp's), 4B ~-6% (already weight-stream-bound); pp8192 runs at ~68%
of the f16 flash. Enable it for the footprint, and for small models at depth.

## Decode latency: wide small-op kernels + the replay tape

The GPU-counter profiler's first reading found gemma decode spending 20% of
its GPU time in RmsNorm: 105 launches/token, each a single 32-lane simdgroup
serializing dim/32 dependent loads (~20 µs) with nothing else resident at
rows==1. The same disease showed in three more places, each fixed the same
way:

- `rmsnorm_wide_f32` / `qknormrope_wide_f32`: 8 simdgroups per row (per head
  for rope), partials folded through threadgroup memory, routed at rows ≤ 4.
  Fleet decode +8-19%.
- `rope_f32` ran ONE THREAD per (row, head) — serial pairs, trig included,
  plus a whole-head copy (19% of TinyLlama decode). Now one thread per
  rotation pair.
- The k-split GEMVs above (4 simdgroups per row group) — small/mid GEMVs
  underfill the GPU and serialize the whole k-dim.

The decode replay tape (`Capabilities::decode_replay`) records one token's
dispatches and re-encodes them per token with **no graph walk** — and replays
on a CONCURRENT encoder: each recorded dispatch carries a write mask
(`TapeEntry.wmask`), and barriers are placed exactly at RAW/WAW/WAR hazards,
so independent runs (q/k/v GEMVs, the two KV writes) overlap. Un-annotated
dispatches record writes-everything and replay serialized — never
under-fenced. MoE decode tapes (the device expert path resolves all data
dependence on-GPU), and so does the llama arch (`Op::Rope` reads the bound
positions buffer, replay-safe by construction).

## Profiling

`INFR_METAL_PROFILE=1`: per-op CPU-encode + GPU commit+wait wall, printed on
drop. `INFR_METAL_PROFILE=2`: additionally flushes after each op to attribute
GPU wall per op — costs the batching AND, because the replay tape re-executes
decode tokens without walking ops, only ever attributes the RECORDED token.
`INFR_METAL_PROFILE=3` is the honest mode: `MTLCounterSampleBuffer`
stage-boundary timestamps, one encoder per op inside ONE command buffer —
true in-context per-op GPU time, with the tape disabled so every token
attributes. (Apple silicon samples at stage boundaries only, not dispatch
boundaries; metal-rs 0.33's `resolve_counter_range` returns uninitialized
memory, so the resolve reads the NSData directly.)

## Tests

`tests/parity.rs` — 68 GPU parity tests vs the CPU reference: every op, and
for quantized Linear every (format × kernel shape) pair including partial
tiles; attention covers unsplit/split8/split32/flash/flash2/vec, sliding
windows at both tile and block granularity, partial query tiles, and the
retained hd=72/96 routes.
`tests/dispatch_overhead.rs`, `tests/gemv_bw.rs` — evidence probes (ignored),
kept so the floors above stay reproducible.

## Negative results (measured — don't re-try without new evidence)

- Op fusion / dispatch-count reduction: per-kernel overhead is ~1.2 µs; a fused
  Add+RmsNorm measured nothing at prefill. (The decode Linear→Add residual
  peephole — mirroring the Vulkan adapter — is the exception: +2.5% on 4B
  decode, from removing a dependency stage per sublayer, not dispatch count.)
- Row-tiled GEMV for prefill weight reuse: zero change — the SLC already
  absorbed the re-reads; the scalar pipeline's ~1:1 load:FMA ratio was the
  limit (hence MMA).
- f32 flash attention (three tuning rounds) and a chunked-SIMD scoring variant:
  both lost to the scalar 32-way split.
- Same-shape half-fragment GEMM swap: zero — the f16 win only cashes as bigger
  tiles.
- 64-row private GEMM tiles: register pressure regression. BK=32 in both the
  private and the first cooperative tile: nothing / worse.
- Linear threadgroup-width sweep (32→1024), uint4-wide GEMV loads, select-based
  `get_scale_min_k4`: all flat.
- NSG=2 mid-k GEMV split: wash-to-negative (the reduce tax eats the halved
  chain at gemma's ne=1152 shapes).
- Vectorized DEC16 staging loads in the GEMM path: no change (staging is
  hidden behind the MMAs).
- B-direct cooperative GEMM (activation fragments straight from a pre-cast
  f16 buffer, no B staging — the Vulkan A_GLOBAL idea): wash on M3; B staging
  was already free.
- Ungated k-split: −8% on 0.6B (in_f=1024 is four Q4_K superblocks; three of
  four simdgroups idle and the reduce is pure tax) — hence the per-format
  k-depth gates.

## Remaining levers

Decode is at parity; prefill sits at the measured tile floor. What's left
needs a new evidence class or new hardware:

- Prefill GEMM: ~53% of f16 simdgroup_matrix peak, same class as llama.cpp —
  Xcode's per-line shader profiler (not installed on the dev box) is the next
  instrument if the last ~10-15% ever matters.
- Metal-4 tensor ops: the hardware step change; `has tensor = false` on M3,
  likely M5-only.
- gemma decode at depth (0.92×): +0.58 ms/token of hd=256 vec attention on
  the global layers — 4% of one cell, measured and parked.
- The routing gates (k-split thresholds, wide-norm rows ≤ 4) were tuned on an
  M3 Pro; a boot-time micro-sweep would adapt them across M1-M4.
