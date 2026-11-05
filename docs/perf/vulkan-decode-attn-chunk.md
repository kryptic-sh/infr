# Vulkan decode attention: the split-K chunk floor

## The lever

Decode attention is split-K: each `(row, head)`'s key range is cut into chunks, one
workgroup per chunk, then `attn_combine` merges the partials. The chunk is
`max(span/32, floor)`, where the floor is the policy's `min_chunk` — and the glue
that lets a record-once tape serve every depth: the chunk is baked from the cache
*capacity*, and each token re-derives it from the live `kv_len`.

That floor was 64 keys for every row count. At `rows == 1` it is wrong: kv below
512 is a couple of chunks, so the split pass launched `nh * 2` workgroups — 16 on
an 8-head model — and each one walked a 64-key serial chain. On Polaris a decode
attention dispatch then cost ~45 us at kv 81 *and* at kv 529, i.e. the cost was
independent of how much work it had.

## Change

Decode rows use `ATTN_SPLIT_DECODE` (floor 16) instead of `ATTN_SPLIT` (floor 64):
110-ish keys is now ~7 chunks/head instead of 2. `span/32` still takes over above
512 keys, so deep-context decode is byte-identical, and the combine's shared
`wexp[1024]` cap still bounds the count.

The floor had to move out of the shaders too. `attn_live.comp`, `attn_decode.comp`
and `attn_partial.comp`'s SELF_CHUNK arms spelled the floor as a literal `64u`
inside `max(pc.chunk, min(max(span/32, 64u), 512u))`, which would have overridden
any host-side floor. They now derive it from the baked push constant
(`min(max(span/32, pc.chunk), 512u)`), which is what the host already bakes.

## Result

Measured on an RX 580 (RADV POLARIS10), `gemma-4-E4B-it-qat-UD-Q4_K_XL`, decode
24 tokens at depth, ctx 8192, three processes each, empty config:

| depth | before (tok/s) | after (tok/s) |
| ---: | ---: | ---: |
| 16 | 39.24 | 45.74 |
| 64 | 44.18 | 45.28 |
| 128 | 43.90 | 44.83 |
| 512 | 43.09 | 43.50 |
| 2048 | 40.73 | 41.45 |

Per-op, static path (`INFR_PROF_OPS=1 INFR_SEAM_NO_REPLAY=1`), attention only:

| depth | before | after |
| ---: | ---: | ---: |
| 16 | scalar `attention_kv_bda`, 112 us x 42 | 12.8 us x 35 + 15.2 us x 7 + 5.4 us x 42 |
| 64 | 42.9 us x 35 + 51.8 us x 7 | 14.5 us x 35 + 16.9 us x 7 |
| 128 | 43.5 us x 35 + 52.6 us x 7 | 14.6 us x 35 + 17.7 us x 7 |
| 512 | 42.0 us x 35 + 52.9 us x 7 | 17.3 us x 35 + 23.3 us x 7 |

The static path's attention drops ~3x, and at depth 16 it also stops falling back to
the scalar kernel: decode there goes 39.24 -> 45.74 tok/s (+16.6%). Shallow kv is
where the old floor hurt most, because below 64 keys the split path did not
qualify at all. Deeper than 512 keys nothing changes.

Correctness: `attn_decode_parity` still passes bitwise (`f32::to_bits`, no
tolerance) on both legs — static push-constant and record-once replay — across
its 8 shapes x 12 configs, including single-key chunks and a ring past its wrap.

## Reproduce

```sh
M=/path/to/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf
for d in 16 64 128 512 2048; do
  infr bench "$M" -p 0 -n 24 -d $d -r 1 --ctx 8192 --dev Vulkan0 \
    --config /tmp/infr-empty.toml --json
done
```

## What this did not fix

Per dispatch, decode attention is still ~14 us for kv < 200 — far more than the
bytes justify. The kernel's cost is flat in `kv` up to ~512 and only then starts
to track work, so the next lever there is per-dispatch fixed cost, not
parallelism. Everything below looks fine and was ruled out: workgroup count (16
vs 80 workgroups changed nothing), chunk size alone (tested at 64), and the
combine (5–10 us, tracked by its own label).
