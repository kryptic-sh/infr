# 09 — DeepSeek V4: the compressed-KV tiers, then performance

Finishes `deepseek4`: layers with `compress_ratio` 4 and 128, which is every
layer but two of the shipped V4-Flash file. Scoped on 2026-11-16 by reading
llama.cpp at `5676dfbe0` (`src/models/deepseek4.cpp`,
`src/llama-kv-cache-dsv4.cpp`, `src/llama-graph.cpp`, `src/models/models.h`).
Statements marked **[I]** were inferred, not read; **[U]** are unknown. Nothing
here has been built or run yet, except slices S0 to S4b and S5a (marked below).
Cite functions, not lines: both reference files have grown since the notes in
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
2, 4, …, 42; HCA layers 3, 5, …, 41, read off the GGUF's `compress_ratios` (46
entries for 43 blocks: the three surplus are zeros, which infr's loader drops by
reading a prefix).

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
  the top-k and `j < n_visible[t]` (`build_top_k_mask` adds the 0/`-inf`
  selection mask to the compressed causal mask; when fewer than 512 rows are
  visible, the selected tail is masked rows and stays masked). Top-k only bites
  once `n_visible > 512`, from position 2051 on (`(2051 + 1) / 4 = 513`).
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
4. **Two-segment attention:** done in S4a/S4b — `Op::Attention::comp`, a
   `CompSegment { cache, len, ratio, bias }`; key `j < len` is visible to row
   `t` iff `j < (pos+t+1)/ratio`, the cache is its own V, and the optional bias
   is `[rows, len]`, which is `Op::TopkMask`'s output at `kv_len = len`. It
   requires `key_bias: None` and a causal or sliding-window mask
   (`check_attention`). CPU, and Vulkan through `-DCOMP` builds of
   `attention_kv.comp` (f16 caches, head dim up to 512, static recording only);
   Metal refuses it. `None` leaves every backend on today's path: the
   attention_kv builds without `-DCOMP` compile to byte-identical SPIR-V, and
   the CPU arm's raw-key arithmetic is unchanged. Declined: copying both
   segments into one scratch cache (the Vulkan sinks kernel refuses a canvas
   mask, and it copies about ctx/4 rows per layer per step).
5. **`Op::LightningIndexer` gains a block ratio:** done in S4a/S4b —
   `block_ratio`, key `j` eligible iff `j < (abs+1)/block_ratio`; 1 keeps V3.2
   bit for bit. The V4 emit passes the LID cache as `k_cache` (one 128-wide f16
   row per block), `kv_len = n_visible_max` and `top_k = min(kv_len, 512)`. The
   scoring is ggml's `GGML_OP_LIGHTNING_INDEXER` (`Σ_h w·relu(q·k)`, `w`
   pre-scaled by `1/sqrt(128·64)`), the same as V3.2's non-fused branch. The LID
   compressor and the indexer query are Hadamard-rotated in the reference
   (`attn_rot_k` is forced on for the LID cache); one orthogonal transform on
   both sides, so the emit skips it on both.
6. **Persistent buffers** per layer, threaded through allocation, graph inputs,
   reset, `fork`, `seed_from`, `kv_layer_alloc_bytes`, the footprint estimates;
   V4 becomes not rewindable (audit the callers of `has_recurrent_state()` that
   mean "cannot truncate").
7. **`wpush` handles** mirroring `wload`, then drop the build closure's assert.
8. **Per-step i32 inputs:** compressed-row positions, gather indices, APE rows.
9. **Fixtures:** the all-CSA, all-HCA and mixed layouts exist since S5a, as
   llama.cpp oracle cases (`synthetic_deepseek4_llama_cpp.rs`, section 4), all
   at `hc_mult = 4` and with YaRN keys. A one-layer `hc_mult = 1` variant for
   from-definition checks is still owed, and can only be checked against a hand
   computation: llama.cpp asserts `hc_mult == 4`, and a one-layer HCA file does
   not load there at all (no ratio-4 layer, section 4). The fixture builder
   moved to `tests/dsv4_fixture/mod.rs` so both test binaries build the same
   file.

## 3. Slices

Every slice leaves the ratio-0, hash, `deepseek2` and `deepseek32` tests
unchanged.

| #   | Slice                                                                                                                                                  | Verification that can fail                                                                                                                                                                                                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0  | **Done 2026-11-16.** Re-sync `dsv4_plan.rs` with section 0, add a count of real commits; fix the docs                                                  | literal-vector tests: an HCA decode step gets one dummy at the last row; a 9-token CSA ubatch with no commit is refused (one block short gets one dummy); the commit count excludes dummies                                                                                                                  |
| S1  | **Done 2026-11-16.** `deepseek4` YaRN config and the compressed-tier `ff` vector (unused yet)                                                          | `ff` against a closed form at chosen positions; a file with no scaling keys gives 1; a factor without a type is "linear", a uniform divisor                                                                                                                                                                  |
| S2  | **Done 2026-11-16.** B53: one cache as K and V (`kv_row_elems` → `(hd, 0)`, CPU attention takes one guard at a time)                                   | an op-parity case binding one id to both sides; ratio-0 logits bit-identical, checked ONCE by hand (no test in the tree locks them; see below)                                                                                                                                                               |
| S3  | **Done 2026-11-16.** `Op::GatherRows` on CPU and Vulkan; Metal arm written                                                                             | from-definition parity with fill, grouping and column offset; a swapped column offset goes red                                                                                                                                                                                                               |
| S4a | **Done 2026-11-16.** CPU two-segment attention and the indexer block ratio                                                                             | equals single-segment attention over a concatenated cache with an explicit bias; `None` and ratio 1 bit-identical to today                                                                                                                                                                                   |
| S4b | **Done 2026-11-16.** Vulkan: the same; Metal refuses the segment, its indexer kernel takes the ratio                                                   | CPU-vs-Vulkan op parity at head dim 48 and 512                                                                                                                                                                                                                                                               |
| S5a | **Done 2026-11-16.** llama.cpp loads the synthetic V4 file; ratio-0 compared in CI; compressed-case data checked in, every biting indexer point robust | ratio-0 comparison red under a window off by one and under dropped sinks; digest guard red under a one-value change; robustness check red on the near-ties it found; sums parse test red on a dropped line; YaRN ramp test red at original context 65536; compare-only regeneration red on one changed logit |
| S5  | Persistent buffers, geometry, the no-rewind predicate, `wpush` handles; refusal kept                                                                   | hand-computed V4-Flash footprint; a V4 session refuses a partial rewind                                                                                                                                                                                                                                      |
| S6  | CPU HCA emit; lift the ratio-128 refusal; score the `hca` oracle case                                                                                  | perturbing `attn_compressor_kv` leaves positions 0..126 bit-identical and moves 127 on; a from-definition cache row on the one-layer fixture; a run split at 150 of 200 tokens equals the unsplit run                                                                                                        |
| S7  | CPU CSA emit: overlap, LID, indexer, top-k; lift the last refusal; score `csa` and `mixed`                                                             | commit timing at position 3; from-definition rows 0 and 1 (row 1 reads block 0 as its previous half); with `top_k` 5, indexer weights are inert up to position 22 and move logits from 23; `top_k >= n_visible` equals zeroed indexer weights                                                                |
| S8  | Real-file CPU oracle against llama.cpp                                                                                                                 | `cpu_prefill_matches_llama_debug_dump` on V4-Flash, one prompt under 100 tokens and one over 300 (HCA live); the `deepseek-v3` tokenizer question closed with `llama-tokenize`                                                                                                                               |
| S9  | Vulkan whole model; then paged MoE with hash routing and the SwiGLU clamp (both refused today; the model exceeds 24 GB)                                | synthetic GPU-vs-CPU; the real-file dump comparison on Vulkan                                                                                                                                                                                                                                                |
| S10 | Metal: gather, two-segment attention, the indexer ratio (written in S4b), sqrt-softplus router, `GatherI32`                                            | macOS CI only                                                                                                                                                                                                                                                                                                |

S2's logits check was a one-off: a temporary test wrote the bit patterns of four
ratio-0 fixtures (`PROMPT`; windows 1 and 3 at a 40-token prompt; hash-routed)
on CPU, Vulkan0 and Vulkan1 from the pre-change tree, and the changed tree
reproduced all twelve exactly; the test was then deleted. No permanent lock was
added, because `docs/synthetic-models.md` rule 2 allows a synthetic golden only
when it is blessed against an external oracle, and these values would be
self-blessed. The oracle-blessed V4 lock came with S5a: llama.cpp's logits over
the synthetic file, scored in CI (section 4).

S4's "unchanged" check was the same kind of one-off: a temporary test wrote the
output bits of twelve `Op::Attention` shapes (causal, sliding window and ring,
canvas, sinks, `key_bias`, both, the V4 one-cache shape; head dims 48 to 512;
prefill and decode) and three `Op::LightningIndexer` shapes on CPU, Vulkan0 and
Vulkan1 from the pre-change tree; the changed tree, with `comp: None` and
`block_ratio: 1`, reproduced every one. Separately, the fifteen existing
`attention_kv.comp` builds compile to byte-identical SPIR-V before and after
(compiled with `build.rs`'s `glslc` flags and hashed).

### Notes for S6 and S7: before the first block

`build_attention_impl` still has a raw-only branch (`build_raw_attention`) for a
compressed layer, taken when the tier's compressed mask (`kq_mask`) is null.
`dsv4_build_comp_inputs` creates that mask only when the plan's `n_kv > 0`, and
`dsv4_build_comp_plan` ends with `n_kv = max(PAD(n_kv, 256), 256)`, so at this
pin the branch is unreachable for any ubatch with tokens: the compressed half
always runs, all `-inf` until a block completes (section 0 item 1). The two
statements agree; the fallback is dead code, not the reference's behaviour.
Numerically an all-masked compressed half adds nothing, so infr's emit is free
to skip it:

- **HCA:** when `n_visible_max` is 0, pass `comp: None` (a `len` 0 segment
  computes the same bits on CPU, `attention_comp_segment_*`, but there is
  nothing to bind it to).
- **CSA:** when `n_visible_max` is 0, skip `Op::LightningIndexer` and
  `Op::TopkMask` as well as the segment: both ops refuse `kv_len` or `top_k` 0
  on every backend (`lightning_indexer_and_topk_mask_refuse_zero_keys`), since a
  zero-length score or mask tensor has nothing to select. Otherwise
  `kv_len = len = n_visible_max` and `top_k = min(len, index_topk)`, which is at
  least 1.
- Rows inside a ubatch with no visible block while others have some are the
  segment's ordinary case (tested); their max is finite through the raw keys.

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
another machine; until then the real file is checked by infr alone. The 60.8 GB
"Fractal" repack is not an alternative: its repository ships patched llama.cpp
binaries, so stock llama.cpp presumably cannot read it (not tried).

**The second oracle, done in S5a (2026-11-16): llama.cpp on the synthetic V4
file.** `crates/infr-llama/tests/synthetic_deepseek4_llama_cpp.rs` builds four
cases from `dsv4_fixture`'s `dsv4_model` and checks in what llama.cpp computes
for them. All four are at `hc_mult = 4`, with an indexer key width of 128, hash
routing on the first layers, and YaRN keys (below):

| case     | `compress_ratios`               | indexer `top_k` | prompt lengths           |
| -------- | ------------------------------- | --------------- | ------------------------ |
| `ratio0` | `[0,0,0,0,0]` + null tail       | 512             | 7, 30, 70, 131, 300, 520 |
| `mixed`  | `[0,4,128,4,0]`                 | 512             | 7, 30, 70, 131, 300, 520 |
| `csa`    | `[4,4,4,4]`                     | 5               | 7, 30, 70, 131           |
| `hca`    | `[128,128,128,128]` + null tail | 512             | 7, 30, 70, 131, 300, 520 |

`tests/fixtures/synthetic_dsv4_<case>_llama_cpp.txt` holds the last-position
logits at each length (each prompt a prefix of one id sequence per case; 4.9 to
8.9 KB a file). For the three compressed cases, `..._llama_cpp_sums.txt` (52 to
66 KB) holds per-tensor sums at positions 3, 4, 22, 23, 127, 128 and the last of
a run fed one token per ubatch, 300 tokens long (131 for `csa`). Each header
names the llama.cpp build, the commands, the GGUF's byte length and FNV-1a
digest, and generated measurements: the logit range per length, the indexer
robustness per length, and for `ratio0` infr's own scores against the data.
`synthetic_dsv4_llama_cpp_fixtures_match_their_gguf` fails, naming the command
below, when the builder writes other bytes.

The regenerator compares by default and writes only into a directory
`INFR_LLAMA_FIXTURE_OUT` names, so the GPU gate (`--run-ignored all`) never
rewrites the tree, with or without `INFR_LLAMA_DEBUG`; the LFM3 regenerator now
works the same way (`infr_testkit::llama_cpp::write_or_compare_fixture`).
Without `INFR_LLAMA_DEBUG` both return at once rather than fail: the gate runs
every ignored test on machines with no llama.cpp build, and the checks that must
run everywhere are the non-ignored ones. Run twice, the regenerator reproduced
every file:

```text
INFR_LLAMA_DEBUG=~/Projects/mxaddict/llama.cpp/build/bin/Release/llama-debug.exe \
INFR_LLAMA_FIXTURE_OUT=crates/infr-llama/tests/fixtures \
  cargo test -p infr-llama --test synthetic_deepseek4_llama_cpp -- --ignored \
  synthetic_dsv4_regenerate_llama_cpp_fixtures
```

`synthetic_dsv4_write_oracle_ggufs` (`INFR_DSV4_GGUF_DIR`) writes both sides'
files for bisecting by hand with `llama-eval-callback` or `llama-debug`.

What llama.cpp at `5676dfbe0` demanded of the fixture, each a way the old
fixture differed from the real file:

1. `attention.value_length` (the real file has 512, its `key_length`):
   `build_attention_impl` asserts the two are equal, and the default is
   `n_embd / n_head`. infr reads no V width; added to the fixture.
2. `hyper_connection.count = 4` (the real value): `build_hc_mixes` asserts
   `hc == 4`. The canonical fixture keeps 2 for infr's own tests.
3. An indexer key width of at least 64 (the real file has 128): the LID cache
   forces a Hadamard rotation whose matrices exist only from 64 up, so 24 died
   in `llama_kv_cache::init_rotation` with "invalid unordered_map key". The
   oracle files use 128; the canonical fixture keeps 24, which no other
   canonical dimension equals (128 would equal its `hc_dim`), now through a
   per-dims `Dsv4Dims::indexer`.
4. **At least one ratio-4 layer.** Without one the LID cache has no layers, and
   `llama_kv_cache`'s constructor doubles `n_rot_k` from 64 until the `u32`
   wraps to 0, then takes a remainder by it: exit `0xC0000094` (integer division
   by zero) with no message. So no all-ratio-0 or all-HCA V4 file loads in
   llama.cpp at this pin; the real file has CSA layers and is unaffected. Worked
   around with a NULL tail layer: ratio 4 in llama.cpp's file, ratio 0 in
   infr's, `attn_output_b` all zero in both, so that layer's attention adds
   exactly zero on either side.
5. A vocabulary llama.cpp can type a prompt in: the printable ASCII characters
   from `!`, no merges, so a prompt string reads back as exactly its ids, and
   `add_bos_token = false`, as the real file declares. Prompts skip `"`, `\` and
   the EOS id.

Items 1 and 5 changed the canonical fixture. A one-off bit comparison (ratio-0
logits at two prompts, old metadata against new) showed no ratio-0 number moved:
infr reads none of those keys on that path.

**Indexer near-ties, and why `csa` stops at 131 tokens.** Where the top-k drops
blocks, a row whose k-th and (k+1)-th scores nearly tie flips its selection
under rounding, and a stored point that flips under rounding cannot judge S7.
The regenerator measures it by perturbation at every such point: ±1e-3 relative
noise on `indexer.attn_q_b` and `indexer_compressor_kv` (about twice the f16
rounding of the keys), three draws; a kept selection moves the logits by exactly
0, since the scores only choose blocks, and the bound is 1e-4. It fails the run
on a near-tie, and also unless ±0.5 noise moves the logits by more than 1e-2, so
the indexer stays load-bearing. With `top_k` 5 the near-ties are unavoidable on
long prompts: 4 CSA layers × hundreds of rows × a selection among dozens of
blocks. Measured with llama.cpp: none of 40 prompt sequences held at 300 tokens
(moves 1.3e-4 to 1.2e-1) and none of 30 at 520 (9.0e-4 to 7.1e-2), and the first
`mixed` sequence failed at 70 tokens (7.3e-3) when it used `top_k` 5. So only
`csa` lets the top-k bite, and only to 131 tokens; there the noise moves nothing
at 30, 70 and 131 tokens while ±0.5 moves 8.8e-2, 6.1e-1 and 6.5e-1. The other
cases use the real file's `top_k` 512, which keeps every visible block (the
reference keeps `min(width, top_k)` with the width padded to at least 256), so
they can go to 520 tokens with nothing to tie.

**What S7 must do about indexer precision.** llama.cpp scores the indexer on
keys rotated by an orthonormal Walsh-Hadamard matrix (`ggml_gen_hadamard`, H² =
I) and then stored as f16 in `kv_lid`, against a query rotated by the same
matrix. infr's plan (section 2 item 5) stores unrotated keys as f16 and skips
the query rotation. In exact arithmetic the two give the same scores; in f16
they do not, because rounding `Hk` and rounding `k` make different per-element
errors of up to 2^-11 relative. That is below the ±1e-3 noise every stored
biting point was checked against, so at those points the selections should agree
**[I]** and the logits should meet the ratio-0 bound (2e-3); llama.cpp's own
one-token-per-ubatch run differs from its batched run by at most 3.2e-4 over all
cases and lengths, which is the scale of what remains. A gap of order 1e-1 at a
biting point and not elsewhere means a flipped selection: compare `lid_top_k` in
the sums file. S7 must not add biting points beyond what the regenerator's
robustness check accepts, and needs no rotation of its own unless it wants
bit-closer agreement.

**Ratio-0 result: infr's CPU matches llama.cpp; no infr bug found.** From the
`ratio0` header, max |logit diff| at 7 / 30 / 70 / 131 / 300 / 520 tokens:
3.4570694e-6 / 5.376339e-5 / 2.771616e-5 / 8.420646e-5 / 4.518032e-5 /
1.5926361e-4; probability cosine at least 0.9999999980694658; top-1 agrees at
every length. The logits span -2.9464645 to 2.3770182 at 7 tokens and -4.006359
to 2.3026319 at 520 (`# logits at ... span` lines), so nothing is saturated.
This is the first time the ratio-0 emit has been compared with llama.cpp on any
file: hash and bias routing, four-stream hyper-connections, sinks, the grouped
output projection, the de-rope, the sliding window past 64 tokens, and ratio-0
layers ignoring the YaRN keys. `synthetic_dsv4_ratio0_cpu_matches_llama_cpp`
holds it at 2e-3 in CI. Shown to fail: a window one token too wide gives
2.0916581e-1 / 2.2315145e-1 / 1.2815297e-1 / 1.021879 from 70 tokens on (7 and
30 unchanged, inside the window); dropped sinks give 3.8182247e-1 / 1.7681241e-1
/ 7.642567e-2 / 5.416107e-2 / 5.1752746e-2 / 7.2669625e-2 at every length, with
top-1 flipping at 30 and 70. Probability cosine stays at or above
0.9320889705648777 and 0.9901684623372332 under those breaks, so max |diff| is
what is asserted.

**YaRN.** The oracle files declare type `yarn`, factor 16 and an original
context of 512, not the real file's 65536. With the fixture's 16-dim rope at
compress base 40000, 65536 leaves five of the eight pairs unscaled (divisors 1,
1, 1, 1, 1, 1.4545456, 2.666667, 16), so the rotation only differs from an
unscaled one slowly with position; 512 gives 1, 1.3061224, 1.882353, 3.368421,
16, 16, 16, 16 (`synthetic_dsv4_oracle_yarn_ramp_varies` prints and checks
this). Max |logit diff| that `--rope-scaling none` causes in llama.cpp, per
length:

| case     | original context 65536 (7 / 30 / 70 / 131 / 300 / 520)    | 512 (the files now)                                    |
| -------- | --------------------------------------------------------- | ------------------------------------------------------ |
| `ratio0` | 0 at every length                                         | 0 at every length                                      |
| `mixed`  | 9.11e-4 / 3.94e-3 / 9.02e-3 / 7.58e-3 / 1.64e-2 / 4.50e-2 | 7.48e-2 / 3.02e-1 / 4.79e-1 / 4.63e-1 / 6.01e-1 / 1.34 |
| `csa`    | 2.01e-3 / 3.44e-2 / 1.46e-2 / 8.55e-1 (to 131)            | 4.50e-1 / 1.73 / 1.16 / 2.67 (to 131)                  |
| `hca`    | 2.30e-3 / 3.22e-2 / 1.82e-2 / 1.65e-2 / 1.62e-2 / 3.33e-2 | 5.82e-1 / 2.29 / 1.02 / 2.46 / 3.24 / 3.83             |

At 65536 the effect at 7 tokens was 9.11e-4 to 2.30e-3, about the 2e-3 bound, so
only the longer prompts could have caught a YaRN error; at 512 it is at least
7.48e-2 at every length. (The 8.55e-1 for `csa` at 131 includes the indexer
choosing other blocks under the other rope.) `ratio0` moving by exactly 0
confirms that llama.cpp leaves ratio-0 layers unscaled.

`synthetic_dsv4_compressed_cases_are_still_refused` asserts the three compressed
cases still hit the ratio-0-only refusal; it goes red the moment S6 or S7 lets
one through, and the fix is to score that case with `cpu_matches_llama_cpp`.

The sums files keep every intermediate `build_overlap_compressed_kv_from_state`
and `build_hca_compressed_kv_from_state` name with `cb` (`csa_state_compress`,
`lid_state_compress` and its `_rot`, `hca_state_compress`: the gathered previous
and current windows, the pooled row, the normed row, the roped row), then
`lid_top_k`, `attn_csa_lid`, `attn_hca`, `attn_raw`, `attn_out` and `ffn_out`
per layer, and `result_norm`; the four names this section gave for bisecting are
the ones this commit prints. `# position: p` heads each position (the logits
files use `# tokens:` for a prompt's ids, a different line). A bisect must allow
for four things, each also said in the file headers: at `-ub 1` the reference
runs every compressor on every step, a real commit or the dummy one (section 0
items 2 and 3); `lid_top_k` is a sum of indices, meaningful only once more than
`top_k` blocks are visible; sums are printed `%f`, six absolute decimals, so
compare with an absolute tolerance; and `result_norm` has no rows (sum 0) except
at the last position. In `hca` the null tail's `csa_*`/`lid_*` tensors and its
zero `attn_out` exist only on llama.cpp's side.
`synthetic_dsv4_llama_cpp_sums_files_parse` reads the checked-in files in CI and
writes them back, and `synthetic_dsv4_sums_log_splits_into_positions` runs the
log parser on a hand-written log, so neither can rot.

## 5. Performance, after correctness

Measure against `llama-bench` (CPU and Vulkan builds) on the same file.

V4, in order: the CPU attention arm converts the whole committed compressed
prefix (`len` rows) to f32 on every call, as it already does for the raw cache,
which at long context is `ctx/4` rows per CSA layer per decode step (convert per
key inside the dot instead, or keep an f32 shadow); batched prefill (per-token
today, which multiplies expert I/O by the prompt length; needs an ids buffer per
chunk for hash layers); a span-carried hyper-connection stream so layer-major
and streamed prefill serve V4; paged MoE with hash routing and the clamp, then
B184; the sinks and compressed attention tiers on Vulkan, gathering the CSA
top-k at long context instead of masking; the indexer's cost per CSA layer;
`CompressPool` and hyper-connection kernels (B-DSV4-POOL, B-DSV4-HC); decode
replay, measured first.

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
