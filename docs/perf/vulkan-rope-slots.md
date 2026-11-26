# Vulkan rope: a thread per pair, not per head

## The lever

`rope.comp` applied the whole rotation for one (token, head) in one thread: a
`rope_dim/2`-iteration loop — 64 pairs at hd 128 — each pair paying `pow`, `sin`
and `cos` before two dependent stores. Decode carries one token, so the dispatch
had exactly as many live threads as heads. Qwen2.5-Coder-3B's query rope has 16,
which leaves 48 of a 64-lane workgroup exited at the index bound; its K rotate
has dot `nkv` = 2, so two lanes walked 64 pairs each while the rest of the wave
was empty. The parallelism a GPU needs to hide a `sin`/`cos` chain is simply not
in that shape.

The grid was also sized in threads rather than in workgroups (`t * n_heads`
workgroups of 64 for `t * n_heads` threads), which idled 63 of 64 lanes on top of
that. That one is a plain bug and landed separately, as its own fix.

## Change

A slot is the unit of work now: `hd2 = ceil(hd/2)` of them per (token, head).
Slot `i < hf` rotates pair `i` (interleaved `2i, 2i+1`, or NEOX `i, i + hf`); a
slot at or past `hf` copies two elements of the un-rotated tail `[rope_dim, hd)`.
Every element is still written by exactly one thread, so an in-place rope stays
race-free, and the pair-level arithmetic is untouched — the backward build still
negates `sin`, the freq-factor build still divides the angle, and the fused
`qk_norm_rope` kernels are not involved at all.

The kernel grid-strides (`for (gi = first; gi < total; gi += stride)`), so a grid
that `dispatch_wide` cuts in 2-D still visits every slot exactly once, and the
launch is sized in workgroups by `rope_groups`: `ceil(t · n_heads · hd2 / 64)`.

## The numbers

RX 580 (RADV POLARIS10, wave64-only), `radv` / Mesa 26.2.7
(`mesa-vulkan-drivers-26.2.7-1.fc44`), `--config /dev/null`, Vulkan0, 3 reps,
alternating base/new with 20s cooldowns, base = `kryptic-sh/infr` main at
`4e93a740`:

```sh
infr --config /dev/null bench "$M" -p 0 -n 128 -r 3 --dev Vulkan0
infr --config /dev/null bench "$M" -p 512 -n 0 -r 3 --dev Vulkan0
```

| shape                                   |      base | slot threads |                        delta |
| --------------------------------------- | --------: | -----------: | ---------------------------: |
| Qwen2.5-0.5B-Instruct Q4_K_M, tg128     | 174.4 t/s |    181.0 t/s |                        +3.8% |
| Qwen2.5-Coder-3B-Instruct Q4_K_M, tg128 |  77.7 t/s |     81.9 t/s |                        +5.3% |
| Qwen2.5-Coder-3B-Instruct Q4_K_M, pp512 | 124.1 t/s |    124.4 t/s | +0.2%, inside the run spread |

Per dispatch (`INFR_PROF_OPS=1`, decode with `INFR_SEAM_NO_REPLAY=1` so the tape
carries timestamps, Coder-3B, 2304 dispatches each over 64 tokens):

| kernel                                     |   base | slot threads | share of GPU time |
| ------------------------------------------ | -----: | -----------: | ----------------: |
| `rope_f16_bda` (K into the cache, `nkv`=2) | 26.8us |        5.5us |       7.2% → 1.5% |
| `rope_f16` (Q scratch, `nh`=16)            |  603ns |        2.1us |     0.16% → 0.59% |

That is ~713us less rope per decoded token from the K rotate and ~54us more from
the query rope, against a 12.87ms token: the measured 660us between 77.7 and
81.9 t/s is the same difference seen end to end.

## What the query rope's regression is

The small dispatch gets slower: 16 workgroups of one pair each cost more to
launch and schedule than one workgroup walking 64 pairs per lane, so a kernel
that used to be at the timestamp floor is now measurable. It is a fixed cost per
dispatch that the pair mapping cannot amortise, and it is 14x smaller than what
the K rotate saves at the same count. A shape whose query rope already fills a
wave — a prefill ubatch — is flat, which is what the pp512 row shows.

## Correctness gates

- `ops::tests::rope_partial_matches_host` — new. A partial rope's tail slots
  against a host reference, which the rope tests here did not cover because they
  rope the full head. (`seam_op_parity::rope_neox_and_norm_parity` does rope a
  partial head, at hd 24 / rope_dim 12, so a partial rotation was not untested —
  the eager `rope()` helper's tail slots were.)
- `ops::tests::rope_matches_host`, `recorder::tests::qk_norm_rope_matches_cpu`
  (the fused kernel, unchanged), and the ring-cache/graph refusals in `adapter`.
- `recorder::tests::rope_groups_size_the_slot_grid` — new host test: coverage at a
  decode shape, a ragged tail copy, an odd head dim, and a prefill grid past the
  guaranteed group count.
- `recorder::tests::elementwise_ops_split_wide_grids` still drives all ten rope
  dispatches through the forced 2-D split and compares the bytes.
- An end-to-end generation under the Khronos validation layer is silent apart
  from `VUID-VkPipelineShaderStageCreateInfo-pNext-02755`, which the unmodified
  binary reports identically (a pre-existing `kernel_sg(…, 32)` subgroup pin on a
  device whose `requiredSubgroupSizeStages` is empty).
