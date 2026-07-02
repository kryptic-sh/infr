# infr-metal — Apple GPU backend architecture

State as of 2026-07-02 (PRs #4/#5/#6). `crates/infr-metal` implements the same
`Backend` seam as `infr-cpu` and `infr-vulkan` on Apple's Metal API. It started
as a correctness-first reference (per-op command buffers, dequant-to-f32
everything) and was rebuilt profiling-first; this doc records the architecture
that emerged and the measured reasoning behind it.

## Numbers (M3 Pro 18-core, Qwen3 Q4_K_M, greedy, 889-token prompt)

| model | metric | infr-metal | llama.cpp (same GGUF/machine) | gap |
| --- | --- | --- | --- | --- |
| 0.6B | decode 360 tok | ~132 tok/s | 191 | 1.45× |
| 0.6B | prefill 889 | ~2840 tok/s | 4191 | 1.48× |
| 4B | decode 430 tok | 37.5 tok/s | 46.7 | 1.25× |
| 4B | prefill 889 | ~498 tok/s | 630 | 1.27× |

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
gap. Each thread dequantizes exactly one 16-block per K-step (128 threads = the
whole weight tile) and x is cast f32→f16 inline during staging. Each simdgroup
owns a 32×16 quadrant as 8 f32 accumulators; partial token tiles stage through
threadgroup memory with a row-guarded copy-out so every tile takes the MMA
path.

Precision policy: **decode is bit-exact** against `dequant_block` (same f32
multiply sequence). Prefill GEMM rounds operands to f16 (~5e-4 relative, well
under quantization error — the llama.cpp trade); its parity tests compare
against a reference that rounds operands identically, keeping the tolerance at
1e-3 instead of testing nothing with a loosened bound.

## Attention routing

All scalar variants share `AttnParams` and an online-softmax core; routing is
by launch shape because the scalar kernels are latency-bound on their serial
O(kv_len) chain:

- **`attnflash_f16kv`** — wide launches on an f16 cache (rows·n_head ≥ 128,
  kv_len ≥ 64, hd ≤ 128, hd % 8 == 0): one simdgroup per (8-query tile, head).
  K^T/V 8×8 fragments load DIRECTLY from the f16 cache (strided, transposed for
  K), Q is cast once per op to f16, scores + output accumulate f32, the masked
  online softmax runs scalar f32 on an 8×8 tile per 8-position KV block with
  the row rescale as a diagonal f32 MMA. An earlier all-f32 flash kernel lost
  to the scalar split (it staged K/V through 8 KB threadgroup tiles); zero
  staging is what makes this one win. Tail blocks may read up to 7 rows past
  the causal limit — always inside the bound cache buffer — and are masked.
- **`attnsplit32_*` / `attnsplit_*`** — narrow launches (decode) and long
  contexts: NSG (32 or 8) simdgroups per (query, head), each owning a strided
  KV slice with a private online softmax, merged in threadgroup memory. The
  32-way split fires when kv_len ≥ 128 and hd ≤ 128 (its accumulator needs
  16 KB of threadgroup memory).
- **`attention_*`** — the lean one-simdgroup-per-(query, head) kernel for
  short-context leftovers.

**Per-pipeline thread-cap gating**: `maxTotalThreadsPerThreadgroup` is
per-PIPELINE (register pressure or a paravirtual device — GitHub's macOS CI —
can push it below a kernel's requirement) and `encode_tg` clamps silently,
which for these kernels means skipped KV slices and garbage merges. Every
wide-threadgroup route checks its own pipeline's cap and degrades to the
next-narrower kernel.

## Profiling

`INFR_METAL_PROFILE=1`: per-op CPU-encode + GPU commit+wait wall, printed on
drop. `INFR_METAL_PROFILE=2`: additionally flushes after each op to attribute
GPU wall per op — costs the batching (analysis mode, not the fast path).

## Tests

`tests/parity.rs` — 30+ GPU parity tests vs the CPU reference: every op, and
for quantized Linear every (format × kernel shape) pair including partial
tiles; attention covers unsplit/split8/split32/flash and sliding window.
`tests/dispatch_overhead.rs`, `tests/gemv_bw.rs` — evidence probes (ignored),
kept so the floors above stay reproducible.

## Negative results (measured — don't re-try without new evidence)

- Op fusion / dispatch-count reduction: per-kernel overhead is ~1.2 µs; a fused
  Add+RmsNorm measured nothing.
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
