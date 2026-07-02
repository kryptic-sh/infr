# infr-metal — Apple GPU backend architecture

State as of 2026-07-02 (PRs #4/#5/#6 merged, #10 = the flash_attn_ext arc).
`crates/infr-metal` implements the same
`Backend` seam as `infr-cpu` and `infr-vulkan` on Apple's Metal API. It started
as a correctness-first reference (per-op command buffers, dequant-to-f32
everything) and was rebuilt profiling-first; this doc records the architecture
that emerged and the measured reasoning behind it.

## Numbers (M3 Pro 18-core, Qwen3 Q4_K_M, `infr bench` vs `llama-bench`, same GGUF)

(One idle-machine run, both engines back-to-back, 2-3 reps each.)

| model | metric | infr-metal | llama.cpp | gap |
| --- | --- | --- | --- | --- |
| 0.6B | tg128 | 158 tok/s | 193 | 1.23× |
| 0.6B | tg128 @ d16384 | 49.7 tok/s | 46.6 | **infr ahead** |
| 0.6B | pp2048 | 3508 tok/s | 3751 | 1.07× |
| 0.6B | pp8192 | 2233 tok/s | 2342 | 1.05× |
| 4B | tg128 | 40.5 tok/s | 46.6 | 1.15× |
| 4B | tg128 @ d16384 | 21.8 tok/s | 24.4 | 1.12× |
| 4B | pp2048 | 569 tok/s | 603 | 1.06× |
| 4B | pp8192 | 455 tok/s | 489 | 1.07× |

Decode is weight-stream bound: the GEMV kernels run ~107 GB/s of a measured
~133 GB/s read roofline, and the per-token host cost is ~0.2 ms (the recorded
decode-replay tape re-encodes the whole forward with no graph walk). The
remaining gap vs llama.cpp is effective-bandwidth tail (dispatch ramp on
36-66 MB matrices + the serial small-op chain), not kernel structure — the
mul_mv port matches their N_R0/N_SG configuration exactly.

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
  16-element sub-block per call; `get_scale_min_k4` ported). ~4.5 / ~6.6 bpw
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
  partial query tiles. Templated over compile-time head_dim (hd=64/128
  instantiations — the fully-unrolled loops were worth +38% alone); NSG=8
  benched slower.
- **`attnflash_f16kv`** — the single-simdgroup flash kernel, kept for
  hd % 8 == 0 shapes outside the hd 64/128 instantiations: one simdgroup per
  (8-query tile, head), K^T/V 8×8 fragments direct from the f16 cache, scalar
  masked online softmax per 16-position KV block, diagonal-MMA row rescale.
  An earlier all-f32 flash kernel lost to the scalar split (it staged K/V
  through 8 KB threadgroup tiles); zero staging is what makes these win. Tail
  blocks may read up to 7 rows past the causal limit — always inside the
  bound cache buffer — and are masked.
- **`attnvec_f16kv` (hd 64/128)** — the llama.cpp `kernel_flash_attn_ext_vec`
  structure for long-context decode (kv_len ≥ 128): 32 simdgroups per
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

## Profiling

`INFR_METAL_PROFILE=1`: per-op CPU-encode + GPU commit+wait wall, printed on
drop. `INFR_METAL_PROFILE=2`: additionally flushes after each op to attribute
GPU wall per op — costs the batching (analysis mode, not the fast path).

## Tests

`tests/parity.rs` — 39 GPU parity tests vs the CPU reference: every op, and
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

## Remaining levers (est. ≤ 15% each on this model class)

Decode GEMV lane remap (llama.cpp's access pattern; the byte win of native
blocks was partly eaten by scatter), per-format dequant micro-tuning in the
GEMM staging, MoE path (`Op::MoeFfn` is still host-side here), decode graph
replay (CPU encode is 2.5 ms/forward and becomes the wall below ~5 ms GPU),
attention at small-model scale.
