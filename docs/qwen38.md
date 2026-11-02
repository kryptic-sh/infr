# Qwen3.8 support

"Qwen3.8" is a release name, not an architecture. The three open-weight models
under it land on **three different `general.architecture` strings**, two of
which infr already runs:

| Model                     | HF `architectures`                 | GGUF arch   | Params  | infr today                  |
| ------------------------- | ---------------------------------- | ----------- | ------- | --------------------------- |
| `Qwen/Qwen3.8-27B`        | `Qwen3_5ForConditionalGeneration`  | `qwen35`    | 27.3 B  | **arch already supported**  |
| `Qwen/Qwen3.8-2.4T-A95B`  | `Qwen3_5MoeForCausalLM`            | `qwen35moe` | 2.446 T | arch supported, one GPU bug |
| `Qwen/Qwen3.8-Flash-Next` | `Qwen4ExpForConditionalGeneration` | `qwen4exp`  | 176.9 B | Dense HF CPU reference; no GPU |

So this is not one port. It is one config check, one shader fix, and one large
new family — and they are independent, so they are staged that way below.

Neither Qwen publishes GGUFs itself; every conversion below is third-party but
produced by llama.cpp's own `conversion/` package, so tensor names and metadata
keys are identical across converters. Both the official-weight lineage and the
unsloth lineage are therefore the same loading problem.

## The evidence base

Everything in this document was read from the shipped artifacts, not inferred:

- GGUF headers pulled by HTTP range request and parsed directly (metadata KVs +
  full tensor directory) for `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M`, its
  `mtp-Qwen3.8-27B-Q4_0.gguf` sidecar, and
  `unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ1_S` (shards 1 and 2).
- Diffed against the locally cached `unsloth/Qwen3.5-4B-MTP-GGUF:UD-Q4_K_XL`,
  which infr already runs.
- `llama.cpp` at `57291f2` — `src/models/qwen4exp.cpp`, `src/models/qwen35.cpp`,
  `src/llama-arch.cpp`, `conversion/qwen4exp.py`, `conversion/qwen.py`.
- HuggingFace `transformers@main` `modeling_qwen3_5.py` / `modular_qwen4_exp.py`
  where the C++ was ambiguous.

---

## Stage A — Qwen3.8-27B (`qwen35`), dense

**Expected cost: a config check and a bench run. Possibly zero code.**

This is the same dense gated-DeltaNet hybrid described in
[`qwen35.md`](qwen35.md), at 27 B instead of 0.8 B.

### It is the arch infr already loads

Metadata from the shipped `Qwen3.8-27B-Q4_K_M.gguf`:

```
general.architecture         = qwen35
qwen35.block_count           = 64        embedding_length     = 5120
qwen35.attention.head_count  = 24        head_count_kv        = 4
qwen35.attention.key_length  = 256       value_length         = 256
qwen35.feed_forward_length   = 17408     (dense — no expert_count key)
qwen35.ssm.conv_kernel       = 4         state_size           = 128
qwen35.ssm.group_count       = 16        time_step_rank       = 48
qwen35.ssm.inner_size        = 6144
qwen35.full_attention_interval = 4
qwen35.rope.dimension_count  = 64        dimension_sections   = [11,11,10,0]
qwen35.rope.freq_base        = 1e7       rms_eps              = 1e-6
```

Normalising block indices, its **tensor-name set is identical** to the cached
Qwen3.5-4B's, with exactly two differences:

- the 4B has `blk.N.nextn.{eh_proj,enorm,hnorm,shared_head_norm}` — its MTP head
  is inline; the 27B ships MTP as a separate file (see below);
- the 27B has `output.weight` — it does not tie its LM head, and
  `seam/runner.rs:483` already detects and loads the untied case.

The metadata-key sets differ only by `qwen35.nextn_predict_layers` and
provenance/imatrix keys. So this classifies as **"only metadata differs"** under
[`plan.md`](plan.md)'s recipe — the cheapest of the three categories.

### The shapes that look alarming are already exercised

- **`head_dim` 256 with 24 heads over hidden 5120** — 24×256 = 6144 ≠ 5120, so
  the q/k/v projections are not square. This is not a new hazard: the cached
  0.8B and 4B are the same shape class (16×256 = 4096 ≠ 2560). `key_length` is
  read as its own metadata key, never derived as `n_embd / n_head`, and weight
  shapes come from the GGUF tensor headers rather than from arithmetic.
- **48 value heads over 16 key heads (a 3× ratio)** — the DeltaNet broadcast is
  `h % nk` on every backend (`infr-cpu/src/lib.rs`, `deltanet.comp`,
  `deltanet_chunked.comp`), generic to any ratio. The cached 4B already runs a
  2× ratio, so this is a wider case of a live path, not an unimplemented one.
- **`output_gate_type: "swish"`** in the HF config is a **dead key**. It appears
  nowhere in llama.cpp (`src/`, `conversion/`, `gguf-py/`) and nowhere in
  HuggingFace's own `modeling_qwen3_5.py`, which hardcodes
  `attn_output = attn_output * torch.sigmoid(gate)` (line 775). infr's sigmoid
  gate matches the reference. **Do not "fix" this into a swish gate.**
- **262144 context, 248320 vocab, 64 layers** — no fixed-size array or cap
  found; vocab is taken from the `token_embd.weight` shape.

### Precedent at this exact size

Backlog **B17** benched **Qwen3.6-27B Q4_K_M** — same `qwen35` family, same
parameter scale, dense — end to end, including the Vulkan submit splitter. The
27 B-scale loading, streaming and kernel paths are not speculative here.

### What Stage A actually has to do

1. Pull `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M` (19.0 GB) or
   `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` (16.5 GB). Both fit the 7900 XTX's 25.75
   GB.
2. Run recipe step 7 in order: logits vs llama.cpp on CPU, then a CPU golden in
   `cpu_backend.rs`, then `gpu_seam_matches_cpu_*`, then `infr compare`.
3. Only if something disagrees does this become a code task. **Budget the
   verification, not the implementation.**

Add the golden as a self-skipping case keyed on the HF cache directory, matching
the existing `find_gguf(...)` helpers, so it runs where the model exists and
skips where it does not.

### Two gaps that are real but out of scope for "it runs"

- **MTP is a sidecar file.** `mtp-Qwen3.8-27B-Q4_0.gguf` is a standalone GGUF
  declaring `qwen35`, `block_count = 65`, `nextn_predict_layers = 1`, and
  carrying exactly `blk.64.{attn_*,ffn_*,nextn.*,post_attention_norm}` plus its
  own copies of `token_embd` / `output` / `output_norm`. The tensor names are
  the ones `mtp/mod.rs` already knows — but `load_mtp_head` takes a single
  `Gguf` handle, and nothing in `infr-hub`/`infr-cli` opens a second GGUF
  (`fetch_companions` is hardcoded to `generation_config.json`). So the head is
  loadable in principle and unreachable in practice. This is also moot until MTP
  is unparked — see [`mtp.md`](mtp.md). Base text inference does not need it.
- **Vision is ignored.** infr has no multimodal support at all, and the
  `mmproj-*.gguf` is a separate file infr never opens. The text GGUF still runs
  correctly text-only: nothing in the `qwen35` path reads or reserves image
  token ids. Worth stating in the README row so nobody expects images.

---

## Stage B — Qwen3.8-2.4T-A95B (`qwen35moe`)

**One real code defect; otherwise blocked on hardware, not on infr.**

`Qwen3_5MoeForCausalLM` maps to `qwen35moe` (`conversion/qwen.py:636`), which
infr already parses — `config.rs:624-625` widens `qwen35_moe` into the `qwen35`
bool so the whole DeltaNet/attention parse is shared and only the FFN shape
differs. Config: 92 layers, hidden 8192, 64 heads / 4 KV, head_dim 256, 512
experts with 10 active, `moe_intermediate_size` 2048, a shared expert, and a
128:16 value:key head ratio. It is text-only — no mrope sections, no vision —
and the sectioned-RoPE parse falls back cleanly when the key is absent.

### The defect: `moe_topk.comp` breaks above 256 experts

`crates/infr-vulkan/shaders/moe_topk.comp:57`:

```glsl
// 256 experts → 1 KB shared — well within limits.
shared float ssel_adj[256];
```

but the selection scan directly below it is sized for far more
(`#define MAX_CHUNKS 8u`, commented "128 lanes \* 8 chunks = 1024 experts"). The
no-extension branch — the one every qwen35moe layer takes — writes the array
unconditionally across the full expert count:

```glsl
} else {
    // No extensions: ssel_adj stays zero — the lane-level arithmetic below is unchanged.
    for (uint e = 0u; e < pc.n_expert; e++) { ssel_adj[e] = 0.0; }
}
```

and reads `ssel_adj[e]` for `e < pc.n_expert` in the selection loop. At
`n_expert = 512` that is an out-of-bounds shared-memory write and read:
undefined behaviour, so corrupted routing weights or a driver-dependent hang,
**not** a clean refusal. There is no Rust-side guard — `recorder.rs` passes
`n_expert` into the push constants unchecked.

Nothing currently supported exceeds 256 experts, which is why this has never
fired. **Both** Qwen3.8 MoE models have 512, so this blocks Stage B and Stage C
alike on the GPU path. The CPU MoE router is `Vec`-based and unaffected.

Fix: size the array to the same 1024 the selection scan already assumes (4 KB
shared, still comfortable), or index it in chunks like `taken[]`. Then add the
missing bound as a loud `bail!` rather than leaving the shader to decide.
Tracked as its own backlog entry because it is not Qwen3.8-specific.

### Otherwise: hardware, not code

The smallest published conversion is `unsloth/Qwen3.8-2.4T-A95B-GGUF:UD-Q1_0` at
**397.3 GB across 10 shards**; Q8_0 is 2.6 TB and BF16 is 4.9 TB. That exceeds
local disk, so this model is **not developable here** — the same shape of
constraint [`deepseek.md`](deepseek.md) is staged around. Treat Stage B as: fix
the shader bug, confirm the config parses, and stop. Do not claim support for a
model that has never been loaded.

---

## Stage C — Qwen3.8-Flash-Next (`qwen4exp`)

**Implementation status:** [Qwen4-Exp dense HF loading and scalar CPU-reference
inference](qwen4-exp.md) support the HF identifier `qwen4_exp`. Causal hidden
states, logits and recurrent caches match CPU Transformers on synthetic and
real tiny checkpoints. GGUF's `qwen4exp`, GPU and generic graph execution remain
unsupported and refuse before backend setup. The investigation below describes
native graph/kernel work and the production model's memory constraints.

**A genuinely new architecture — and the one that overlaps the DeepSeek V4 work
already in flight.**

Described upstream as "A Preview of the Qwen4 Architecture". It landed in
llama.cpp on **2026-08-27** (`6c84c7d`), with one follow-up (`6fe7498`, graph
splits) — the reference is days old and has an open "qwen4exp: follow up fixes"
issue plus several correctness reports. **The specification is still moving.**
Re-read it before implementing, and prefer the HuggingFace
`modular_qwen4_exp.py` where the two disagree.

### Shape

48 layers, hidden 2560, 24 heads / 2 KV, head_dim 256, vocab 248320, ctx 262144.
Layers cycle 3× `linear_attention` then 1× `full_attention`
(`full_attention_interval = 4`) — the same hybrid pattern as `qwen35`. MoE
everywhere: 512 experts, 10 active, `moe_intermediate_size` 640, plus a
sigmoid-gated shared expert of the same width.

### What is new relative to `qwen35`

**1. Hyper-connections (HC) replace every norm and every residual add.**
`hc_count = 4`, `hc_lowrank = 320`. The residual is not `[n_embd, T]` but
`[n_embd, 4, T]` — four parallel streams. There is no `attn_norm`, no
`post_attention_norm` and no `output_norm` tensor anywhere in the file; the
dumped `blk.3` confirms this directly. Instead each of the two sub-blocks per
layer is wrapped by a mixer:

```
xn     = rms_norm_per_stream(x) * w_norm        # w_norm is [4*n_embd]
lo     = silu((w_down @ xn) / hc)               # -> [320]
gate   = sigmoid(w_up @ lo)                     # -> [4*n_embd]
mixed  = mean_over_streams(xn * gate)           # -> [n_embd, T]  == sub-block input
inject = w_inject @ xn                          # -> [4, T]
...
w      = 2 * sigmoid(inject / hc)               # centred on 1
x      = x + broadcast(sub_block(mixed)) * w    # per-stream gated scatter
```

Tensors per layer: `hc_{attn,ffn}_{norm,down,up,inject}`; at the head,
`output_hc_{norm,down,up}` — the final mixer **is** the output norm.

**2. PLE — per-layer n-gram hash embeddings.** One tensor dominates the model:

```
per_layer_token_embd.weight   IQ4_NL   [160, 320001536]
```

320 million rows. At IQ4_NL's 18 bytes per 32 weights that is 51,200,245,760
weights → **≈28.8 GB in the smallest published quant**, roughly 40% of the 72.5
GB file.

It works by hashing the last _n_ token ids (`ngram_size = 3`, so n ∈ {2,3}) with
64-bit multipliers, taking `mixed % head_vocab_sizes[h] + head_offsets[h]` for
each of 16 hash heads, gathering those rows, and folding the result into the raw
HC residual through a sigmoid-gated key/query/value interaction plus a dilated
depthwise causal conv. `ple.layers = [1]` — **it runs at exactly one layer**,
and llama.cpp asserts that (`GGML_ASSERT(n_ple == 1)`).

Two consequences worth planning around:

- The hash needs **raw token ids, including predecessors**, read back from KV
  cell metadata. That is a real constraint on infr's GPU-resident pipeline,
  which uploads token ids and otherwise stays on-device — see
  `gpu-resident-pipeline` in the memory notes. It is a once-per-ubatch host-side
  computation, not per layer, so the cost is small; the _plumbing_ is the work.
- The 28.8 GB table is a pure gather of 16 rows × 160 values per token. It never
  needs to be resident. This is the natural first candidate for host residency
  or mmap, and it is what makes a 72.5 GB model plausibly runnable on a 25.75 GB
  card at all.

**3. QSA — block-sparse attention via a lightning indexer.** On full-attention
layers only. Indexer keys are mean-pooled into blocks of `compress_ratio`
tokens, normed and roped once per block, scored `relu(q·k)` summed over 4
indexer heads, and the top `indexer_budget / compress_ratio` **whole blocks**
are selected (plus the incomplete tail). Attention then runs dense over the
selected set via a mask. GGUF: `indexer.{q,k}_proj`, `indexer.{q,k}_norm`,
`attention.indexer.{head_count 4, key_length 128, top_k 2048}`.

**4. The GDN output gate is sigmoid, not silu.** Verified at
`qwen4exp.cpp:416-418`, which carries the comment "the one numerical difference
from Qwen3.5's GDN: sigmoid output gate, not silu", against `qwen35.cpp:253`'s
`ggml_silu`. A single wrong activation here produces plausible-looking garbage,
so it belongs in a parity test, not a comment.

**5. No MTP.** `conversion/qwen4exp.py` sets `supports_mtp_export = False` /
`no_mtp = True`, and transformers ignores `mtp.*` keys on load. The config's
`mtp` block describes a separate draft checkpoint.

### The overlap with DeepSeek V4 — the reason to sequence C after V4

The shipped GGUF carries:

```
qwen4exp.attention.compress_ratios = [0, 0, 0, 4, 0, 0, 0, 4, ...]   (48 entries)
```

and llama.cpp reads it into **`hparams.dsv4_compress_ratios`** — literally the
DeepSeek-V4 field (`qwen4exp.cpp:36,488,711`), alongside `dsv4_hc_mult` for the
HC count. The per-layer compressed-KV state machine, the lightning indexer and
the hyper-connection scaffolding infr is **already building for DeepSeek V4**
(`Op::CompressPool`, `Op::LightningIndexer`, `seam/dsv4_plan.rs`) are the same
machinery this arch needs.

They are not identical, and the difference matters: **DeepSeek V4's indexer
selects individual tokens; qwen4exp's QSA selects whole mean-pooled blocks.**
The plumbing is shared, the selection kernel is not. Likewise llama.cpp's V4 HC
is a full-rank Sinkhorn-normalised variant with a different tensor set, while
qwen4exp's is the low-rank form above — same name, different math, do not reuse
one for the other.

Still, doing Stage C **after** the V4 compressed-KV slice lands turns a large
part of it into reuse. Doing it before means building that machinery twice.

### Feasibility here

`unsloth/Qwen3.8-Flash-Next-GGUF` smallest quant is **UD-IQ1_S at 72.5 GB across
3 shards** (BF16 is 354.0 GB across 8). That fits local disk with room to spare
but not the 25.75 GB card, so Stage C needs the expert pager plus host residency
for the PLE table from the outset — and it inherits the `moe_topk` 512-expert
bug from Stage B. There is no small `qwen4exp` model to develop against, which
is exactly the situation [`deepseek.md`](deepseek.md) is staged around: build
and test each op against the CPU reference on synthetic shapes first, and treat
the full-model run as the last step rather than the development loop.

---

## Suggested order

1. **Stage A** — pull, verify, bench, add a golden. Independent of everything
   else and probably the cheapest supported-model row infr has ever added.
2. **`moe_topk.comp` 512-expert fix** — small, self-contained, and unblocks both
   MoE models. Worth doing even if Stage C never happens.
3. **Stage B config confirmation** — parse-only; do not claim support.
4. **Stage C** — after the DeepSeek V4 compressed-KV slice, to reuse rather than
   duplicate the indexer and compress-ratio machinery.

## Open questions

- **Indexer score scaling.** HF's `Qwen4ExpTextQSAIndexer.forward` divides the
  summed relu'd score by `sqrt(index_head_dim)`; no equivalent scale was found
  in `qwen4exp.cpp:576-585`. Since top-k is scale-invariant under a positive
  constant this may be deliberate, but it must be confirmed numerically against
  a reference logit rather than assumed either way.
- **`output_gate_type` in `qwen4exp`.** Unlike `qwen3_5`, HF's
  `Qwen4ExpTextConfig` genuinely reads this field, but the converter never
  writes it and the C++ hardcodes sigmoid. Correct for the one shipped
  checkpoint (`"sigmoid"`); a port should hardcode it the same way and say so,
  not generalise silently.
- **Recurrent-state rollback.** `qwen4exp` is absent from llama.cpp's
  `llm_arch_supports_rs_rollback`, unlike `qwen35`/`qwen35moe`, so `n_rs_seq` is
  silently clamped to 0 there. Whether infr's own DeltaNet state snapshotting
  has an equivalent limitation for this arch is unexamined.

## References

- llama.cpp `57291f2`: `src/models/qwen4exp.cpp`, `src/models/qwen35.cpp`,
  `src/llama-arch.cpp`, `conversion/qwen4exp.py`, `conversion/qwen.py`,
  `gguf-py/gguf/constants.py`. MIT.
- transformers `main`: `models/qwen3_5/modeling_qwen3_5.py`,
  `models/qwen4_exp/modular_qwen4_exp.py`. Apache-2.0 — **read for math,
  reimplement**; do not copy code into this MIT tree.
- Upstream PRs: `#27742` (add qwen4exp), `#27880` (graph splits), issue `#27941`
  ("qwen4exp: follow up fixes") — check these before implementing Stage C.
