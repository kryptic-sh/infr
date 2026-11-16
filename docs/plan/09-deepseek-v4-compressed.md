# 09 — DeepSeek V4: the compressed-KV tiers, then performance

Finishes `deepseek4`: layers with `compress_ratio` 4 and 128, which is every
layer but two of the shipped V4-Flash file. Scoped on 2026-11-16 by reading
llama.cpp at `5676dfbe0` (`src/models/deepseek4.cpp`,
`src/llama-kv-cache-dsv4.cpp`, `src/llama-graph.cpp`, `src/models/models.h`).
Statements marked **[I]** were inferred, not read; **[U]** are unknown. Nothing
here has been built or run yet, except slices S0 to S3 (marked below). Cite
functions, not lines: both reference files have grown since the notes in
`docs/deepseek.md` were written.

## 0. The reference moved past our records

Fixed in slice S0 (done 2026-11-16); kept as the record of what moved:

1. `dsv4_build_comp_plan` now ends with `n_kv = max(PAD(n_kv, 256), 256)`, so
   the compressed branch runs on every ubatch. `deepseek.md` boundary condition
   2 no longer holds. An all-masked compressed half adds nothing (every padded
   column is `-inf`-masked, and the LID top-k count `min(width, top_k)` keeps
   every visible row either way), so infr may still skip the segment when every
   `n_visible` is 0. `dsv4_plan.rs` carries only the unpadded width,
   `n_visible_max`.
2. HCA gets a dummy commit too (the `!pad_blocks && state_write_idxs.empty()`
   branch writes one dummy block to row `kv_size-1`). `deepseek.md` condition 4
   and the `DSV4_HCA_RATIO` doc in `dsv4_plan.rs` say otherwise.
3. `llama_kv_cache_dsv4::init_batch` now uses `split_equal`, so for one sequence
   `n_seq_tokens = n_tokens` and CSA pads its commits up to `ceil(n_tokens/4)`
   BLOCKS, which is at most one dummy, not `ceil(n_tokens/4)` dummies: the
   reference refuses a sequence two or more blocks short ("DSV4 CSA sequence
   positions are not contiguous"), and contiguous positions are never more than
   one short. (The first draft of this item said `ceil(n_tokens/4)` dummies; the
   code says otherwise.) The refusal was unreachable under `split_simple`; with
   `split_equal` the function can reach it, though `llama_batch_allocr::init`
   rejects a sequence with non-continuous positions before `init_batch` runs, so
   the port carries it as a guard, not as a path a real batch takes. Dummies
   only land in the masked row `kv_size-1`, so no number changes **[I]**.
4. The LID plan is built by its own call with identical parameters (its own
   state and cache, whose sizes equal the CSA ones on V4).
5. B-DSHW-FUSED-REF was stale: the reference's Vulkan backend now has
   `dsv4_hc_{comb,pre,post}.comp` and `lightning_indexer.comp` (no top-k-mask
   case), and its CPU and Metal backends implement the HC, indexer and
   top-k-mask ops.
6. `deepseek.md` says no public GGUF declares `deepseek32`. The HF API reports
   that architecture for `sszymczyk/DeepSeek-V3.2-Exp-light-GGUF` (Q4_K_M 404
   GB; V3.2-Exp weights; tensor list not read). It cannot run on this box.
   Corrected in `deepseek.md`, the README, B102 and the `cpu_backend.rs` V3.2
   comment.

## 1. What a compressed layer computes

Reference names: ratio 4 is **CSA** and comes with the **LID** lightning
indexer; ratio 128 is **HCA**; the plain sliding window is **raw**.

V4-Flash (`config.json` of `deepseek-ai/DeepSeek-V4-Flash`): hidden 4096, head
dim 512 of which the last 64 are rotated, 64 heads, `q_lora` 1024, window 128,
indexer 64 heads × 128, `index_topk` 512, `compress_rope_theta` 160000,
`rope_theta` 10000, YaRN factor 16 with original context 65536. CSA layers are
2, 4, …, 42; HCA layers 3, 5, …, 41 **[I]** from the ratio list.

### Persistent buffers per layer

From the `llama_kv_cache_dsv4` constructor and `llama_dsv4_comp_state`:

| buffer                     | dtype             | row width × rows                |
| -------------------------- | ----------------- | ------------------------------- |
| raw window (K only, V = K) | f16               | 512 × window                    |
| `kv_csa` (CSA layers)      | f16               | 512 × `PAD(ceil(ctx/4), 256)`   |
| `kv_lid` (CSA layers)      | f16               | 128 × `PAD(ceil(ctx/4), 256)`   |
| `kv_hca` (HCA layers)      | f16               | 512 × `PAD(ceil(ctx/128), 256)` |
| `csa_state`                | f32, kv and score | 1024 × 8                        |
| `lid_state`                | f32, kv and score | 256 × 8                         |
| `hca_state`                | f32, kv and score | 512 × 128                       |

All are zeroed at construction. A partial rewind (`seq_rm` with `p0 > 0`) is
only supported through rollback planes infr does not port; otherwise the cache
is cleared. infr needs no 256 padding.

### Shared prologue (`build_attention_impl`)

`cur = attn_norm(hc_pre(…))`.

- **RoPE on a compressed layer is not the plain one.** Every rope there (query,
  kv, the backward de-rope, compressed rows, indexer query) uses
  `freq_base = compress_rope_theta` with YaRN: `freq_scale = 1/factor`,
  `ext_factor` 1 (because the file's `rope.scaling.type` is `"yarn"`; a
  `"linear"` or absent type gives 0, a uniform `1/freq_scale` angle scale),
  `n_ctx_orig`, betas 32 and 1 (no V4 loader key reads the beta keys).
  `dsv4_rope_attn_factor` cancels the magnitude scale to 1. Ratio-0 layers keep
  `rope_theta` with no YaRN, which infr already does. All are NORM ropes over
  the tail of the head. `Config::dsv4_compressed_rope_divisors` (S1) builds the
  `freq_factors` vector, and its doc states the mapping onto `Op::Rope`.
- **Query:** `qr = rmsnorm_w(q_a_norm, wq_a·cur)`; `q = wq_b·qr` as
  `[512, heads, T]`, a weightless per-head RMS norm, tail roped at the token
  position.
- **Key/value:** `kv = rmsnorm_w(attn_kv_a_norm, wkv·cur)`, tail roped at the
  token position; this one row is both K and V of the raw cache.

### CSA compressor (`build_overlap_compressed_kv_from_state`)

- `skv = attn_compressor_kv·cur` (1024 wide);
  `ssc = attn_compressor_gate·cur + ape[pos % 4]`. The positional term is added
  before persisting, so persisted scores include it.
- Source rows: `[ring of 8 | this ubatch's T rows | sentinel]`; the sentinel is
  0 for kv and `-inf` for score.
- Per token at `pos`: `n_visible = (pos+1)/4` (floor). A block is committed only
  when `(pos+1) % 4 == 0`, to cache row `pos/4`, roped at its first position
  `pos+1-4`. A source position below 0 reads the sentinel, one inside this
  ubatch reads its scratch row, anything older reads ring row `pos % 8`.
- Each block reads its previous window (4 rows) and its current window (4 rows).
  The previous-window rows contribute channels `[0, 512)` of their 1024-wide
  state row, the current-window rows channels `[512, 1024)`. Swapping either
  pair still runs and is wrong.
- Pool: per channel, softmax over the 8 window slots of the scores, weighted sum
  of the values (`Op::CompressPool`, window 8); then
  `rmsnorm_w(attn_compressor_norm)`, rope the tail at the block's first
  position, write to `kv_csa`.
- **Commit before attention** (a token that completes a block sees it in the
  same step). **Persist after compressing** (the ring is written after the
  compress has read it).
- The LID compressor is the same at width 128 with `indexer_compressor_*`
  tensors, rotating the last 64 dims, writing `kv_lid`.

### Indexer and attention on a CSA layer

`build_lid_top_k`, `build_top_k_mask`, `build_csa_lid_attention`:

- `iq = indexer.attn_q_b·qr` as `[128, 64, T]`, no norm, tail roped at the token
  position with the compressed-layer rope.
- `w = (indexer.proj·cur) / sqrt(128·64)`;
  `score[t,j] = Σ_h w[t,h]·relu(iq[t,h]·lidK[j])`, with `j >= n_visible[t]`
  masked.
- Keep the top `min(rows, 512)`; a compressed key is visible only if it is in
  the top-k and `j < n_visible[t]`. Top-k only bites once `n_visible > 512`,
  beyond about 2051 tokens **[I]**.
- Attention: one softmax over the raw window, the visible compressed keys and
  the per-head sink, scale `1/sqrt(512)`, V = K; de-rope at the query position;
  grouped `wo_a`/`wo_b` as on ratio 0.

### HCA layer

`build_hca_compressed_kv_from_state`, `build_hca_attention`: `skv` and `ssc` are
512 wide, `ape` has 128 rows, the source is `[ring of 128 | T rows]` with no
sentinel, each block reads its own 128 rows, `CompressPool` with window 128,
norm, rope at the block's first position, write `kv_hca`, persist. Attention is
raw plus the visible prefix of `kv_hca`; no indexer.

Decode is the same graph with `T = 1`. The raw window `[p-127, p]` matches
`AttnMask::SlidingWindow`.

## 2. What infr is missing

Present: `dsv4_plan.rs` (no caller, re-synced in S0), `Op::CompressPool`,
`Op::TopkMask`, `Op::LightningIndexer` (V3.2 contract), `Op::Attention` with
`sinks` and `key_bias`, the ratio-0 `MixerW::Dsv4` emit, `wload`'s `is_dsv4` arm
(uploads the compressor and indexer tensors; `wpush` does not declare them),
`dsv4_weight_shape`, the config fields, the refusal in `generate_dense_backend`,
and the `dsv4_model` fixture.

1. **YaRN for compressed layers:** done in S1 —
   `Config::dsv4_compressed_rope_divisors`, built by the `yarn_freq_divisors`
   helper the deepseek2 `yarn_ff` now calls too. Not applied anywhere yet.
2. **A row gather:** done in S3 — `Op::GatherRows` on CPU and Vulkan, Metal
   written. Its doc spells out the HCA gather and the two overlap halves; any
   negative index writes the fill. **The sentinel is not `-1` in the
   reference:** `dsv4_append_zero_row` appends a row to the source and
   `state_read_idxs` names it by position (`state rows + tokens`), which
   `dsv4_plan.rs` reproduces. That position equals the gather's source row
   count, so the gather treats it as out of range: decided contract, a negative
   index is the fill and an index at or past the source rows is an error, which
   the CPU refuses and the GPU kernels (which cannot see index values when
   recording) write as the fill. `Dsv4CompPlan::gather_read_idxs` hands the emit
   the reads with the sentinel already `DSV4_GATHER_SENTINEL` (`-1`), so no
   translation is left to the emit. Declined: "index == source rows is the
   sentinel" everywhere, because a source declared with spare rows would then
   read a real row where the sentinel was meant, and the out-of-range check
   would lose the one value it can catch on the CPU. The op reads one source
   tensor, which must be an internal tensor (the CPU executor does not load KV
   cache operands into its store), so the emit assembles
   `[persistent state | this step's rows]` with copies, as the reference's own
   `ggml_concat` does.
3. **No scatter is needed [I]:** for one stream the committed rows are
   consecutive (one `WriteKv` per tier, real commits only, never the dummies),
   and the persisted ring rows are at most two contiguous runs (copies).
4. **Two-segment attention:** `Op::Attention` gains an optional compressed
   segment `{ cache, len, ratio, bias }`; key `j` is visible to row `t` iff
   `j < (pos+t+1)/ratio`. `None` must leave every backend on today's path.
   Declined: copying both segments into one scratch cache (the Vulkan sinks
   kernel refuses a canvas mask, and it copies about ctx/4 rows per layer per
   step).
5. **`Op::LightningIndexer` gains a block ratio** (1 keeps V3.2).
6. **Persistent buffers** per layer, threaded through allocation, graph inputs,
   reset, `fork`, `seed_from`, `kv_layer_alloc_bytes`, the footprint estimates;
   V4 becomes not rewindable (audit the callers of `has_recurrent_state()` that
   mean "cannot truncate").
7. **`wpush` handles** mirroring `wload`, then drop the build closure's assert.
8. **Per-step i32 inputs:** compressed-row positions, gather indices, APE rows.
9. **Fixtures:** all-HCA and all-CSA layouts, a one-layer `hc_mult = 1` variant
   for from-definition checks. (YaRN keys: a variant exists,
   `dsv4_model_with_scaling`, kept out of the canonical fixture.)

## 3. Slices

Every slice leaves the ratio-0, hash, `deepseek2` and `deepseek32` tests
unchanged.

| #   | Slice                                                                                                                   | Verification that can fail                                                                                                                                                                                                                    |
| --- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0  | **Done 2026-11-16.** Re-sync `dsv4_plan.rs` with section 0, add a count of real commits; fix the docs                   | literal-vector tests: an HCA decode step gets one dummy at the last row; a 9-token CSA ubatch with no commit is refused (one block short gets one dummy); the commit count excludes dummies                                                   |
| S1  | **Done 2026-11-16.** `deepseek4` YaRN config and the compressed-tier `ff` vector (unused yet)                           | `ff` against a closed form at chosen positions; a file with no scaling keys gives 1; a factor without a type is "linear", a uniform divisor                                                                                                   |
| S2  | **Done 2026-11-16.** B53: one cache as K and V (`kv_row_elems` → `(hd, 0)`, CPU attention takes one guard at a time)    | an op-parity case binding one id to both sides; ratio-0 logits bit-identical, checked ONCE by hand (no test in the tree locks them; see below)                                                                                                |
| S3  | **Done 2026-11-16.** `Op::GatherRows` on CPU and Vulkan; Metal arm written                                              | from-definition parity with fill, grouping and column offset; a swapped column offset goes red                                                                                                                                                |
| S4a | CPU two-segment attention and the indexer block ratio                                                                   | equals single-segment attention over a concatenated cache with an explicit bias; `None` and ratio 1 bit-identical to today                                                                                                                    |
| S4b | Vulkan: the same                                                                                                        | CPU-vs-Vulkan op parity at head dim 48 and 512                                                                                                                                                                                                |
| S5  | Persistent buffers, geometry, the no-rewind predicate, `wpush` handles; refusal kept                                    | hand-computed V4-Flash footprint; a V4 session refuses a partial rewind                                                                                                                                                                       |
| S6  | CPU HCA emit; lift the ratio-128 refusal                                                                                | perturbing `attn_compressor_kv` leaves positions 0..126 bit-identical and moves 127 on; a from-definition cache row on the one-layer fixture; a run split at 150 of 200 tokens equals the unsplit run                                         |
| S7  | CPU CSA emit: overlap, LID, indexer, top-k; lift the last refusal                                                       | commit timing at position 3; from-definition rows 0 and 1 (row 1 reads block 0 as its previous half); with `top_k` 5, indexer weights are inert up to position 22 and move logits from 23; `top_k >= n_visible` equals zeroed indexer weights |
| S8  | Real-file CPU oracle against llama.cpp                                                                                  | `cpu_prefill_matches_llama_debug_dump` on V4-Flash, one prompt under 100 tokens and one over 300 (HCA live); the `deepseek-v3` tokenizer question closed with `llama-tokenize`                                                                |
| S9  | Vulkan whole model; then paged MoE with hash routing and the SwiGLU clamp (both refused today; the model exceeds 24 GB) | synthetic GPU-vs-CPU; the real-file dump comparison on Vulkan                                                                                                                                                                                 |
| S10 | Metal: gather, two-segment attention, indexer field, sqrt-softplus router, `GatherI32`                                  | macOS CI only                                                                                                                                                                                                                                 |

S2's logits check was a one-off: a temporary test wrote the bit patterns of four
ratio-0 fixtures (`PROMPT`; windows 1 and 3 at a 40-token prompt; hash-routed)
on CPU, Vulkan0 and Vulkan1 from the pre-change tree, and the changed tree
reproduced all twelve exactly; the test was then deleted. No permanent lock was
added, because `docs/synthetic-models.md` rule 2 allows a synthetic golden only
when it is blessed against an external oracle, and these values would be
self-blessed. An oracle-blessed V4 lock (llama.cpp over the synthetic file) is
backlog work.

Prefill one-batch against token-by-token equality is the best single test of the
plan, state and sentinel logic; it arrives with batched V4 prefill (section 5).

## 4. Oracle

llama.cpp at this pin runs V4 on CPU; `llama-debug` is built under `build/` and
`build-vulkan/`. Harness: `read_llama_debug_dump` and
`prefill_matches_llama_debug_dump` in `cpu_backend.rs` (`INFR_LLAMA_DUMP`,
`INFR_LLAMA_DUMP_MODEL`, `INFR_LLAMA_DUMP_GEN`); add a `_dsv4` wrapper as LFM3
has. For bisecting, `llama-debug --verbose` prints sums of `hca_state_compress`,
`csa_state_compress`, `lid_top_k` and `attn_csa_lid`.

File: `ggml-org/DeepSeek-V4-Flash-0731-GGUF` Q2_K_S, 98.6 GB (the one infr has
opened). Smaller published files exist (`unsloth` UD-IQ1_S 82.5 GB; a 60.8 GB
"Fractal" repack) whose tensor types infr may not load **[U]**. On 64 GB of RAM
the file is read through the page cache: expect a per-token infr run of a
300-token prompt to take minutes **[I]**.

**The Q2_K_S file does not load in llama.cpp on this box (2026-11-16).** Three
`llama-debug --save-logits -ngl 0` runs on the 98.6 GB file with 61.6 GB of RAM,
two of them with `--no-warmup`: the process working set reached 45 to 48 GB
within about 70 s, available memory fell under 1 GB, and none printed a loader
line or wrote a dump (one ran for an hour, one for over 20 minutes). The default
load mode maps the file and prefetches it. Not tried: `--load-mode none` and
`dio` (both read the weights into allocated memory, which cannot hold this
file), a larger page file, or a smaller file. S8 therefore needs a file that
fits in RAM (the 60.8 GB repack, if both engines load its tensor types) or
another machine; until then the real file is checked by infr alone.

## 5. Performance, after correctness

Measure against `llama-bench` (CPU and Vulkan builds) on the same file.

V4, in order: batched prefill (per-token today, which multiplies expert I/O by
the prompt length; needs an ids buffer per chunk for hash layers); a
span-carried hyper-connection stream so layer-major and streamed prefill serve
V4; paged MoE with hash routing and the clamp, then B184; the sinks and
compressed attention tiers on Vulkan, gathering the CSA top-k at long context
instead of masking; the indexer's cost per CSA layer; `CompressPool` and
hyper-connection kernels (B-DSV4-POOL, B-DSV4-HC); decode replay, measured
first.

`deepseek2` / V3: an `Op::Mla` split-K decode arm and a prefill tier
(B-DSHW-MLA-SCALAR-TIER has the gaps); B108 on CPU at depth. Do not reopen
B-DSV2-ACTIVATION-PRECISION. No `deepseek2` file is cached on the Windows box;
V2-Lite has to be pulled first.

## 6. Not in this phase

`deepseek41` subclasses `deepseek4` with ratios 2 and 1, no overlap, index keys
stored beside the compressed row, cross-layer cache sharing, a two-level
candidate top-k and n-gram embedding lookups (`build_attention41` only skimmed).
Its published files are 233 GB and up, beyond this box. It reuses S3 to S7 and
needs its own phase.
