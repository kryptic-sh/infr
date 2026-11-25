# DeepSeek support plan (V1 → V2/V3 → V3.2 → V4)

Status: **Stage 3 done.** A `deepseek32` (V3.2) model loads, builds a graph with
the lightning indexer wired in, and generates on CPU and Vulkan — see § Stage 3
for what that is verified against. The `deepseek32` path is exercised by
synthetic fixtures only. Public GGUFs that declare `deepseek32` do exist — the
Hugging Face API reports that architecture for
`sszymczyk/DeepSeek-V3.2-Exp-light-GGUF` (Q4_K_M, about 404 GB; tensor list not
read) — but none fits this box. The conversion the tree does test,
`unsloth/DeepSeek-V3.2-GGUF`, declares `deepseek2` and drops the indexer
tensors, so it runs the stage-2 dense-MLA path instead
(`cpu_deepseek_v32_config` and the `#[ignore]`d `cpu_deepseek_v32_golden`). See
`backlog.md` B102 for what that leaves unverified.

Stage 2: CPU path works end-to-end on V2-Lite; Vulkan and Metal MLA kernels are
implemented, wired, and executed on their real devices (Vulkan on the GPU box,
Metal via the macOS CI job's parity suite); Vulkan MoE is implemented;
exp_probs_b loads from V3 GGUFs; the GPU seam test passes (cosine 0.9955
CPU-vs-Vulkan, matching greedy output vs llama.cpp c629da5); the YaRN ramp is
verified numerically against llama.cpp at short AND long context (see the
checklist).

Stage 1 (`deepseek`) is accepted and CPU-tested only: `cpu_deepseek_config` and
`cpu_deepseek_prefill_paris` pass on DeepSeek-MoE-16B (dense DeepSeek-LLM-7B
GGUFs declare `llama`, so they cannot test this arch). There is no CPU-vs-Vulkan
seam test for it. V2-Lite is the development model for the rest.

The reference implementation is llama.cpp, checked out locally at
`~/Projects/mxaddict/llama.cpp`. Every claim about DeepSeek's maths in this
document was read out of that tree at `b10356-2-g030ebb5`
(`030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`), and every claim about what `infr`
already has was read out of this one. Where something was **not** verified it
says so — those lines are the ones to check first, not to trust.

**The checkout has since moved to `b10688-1-g57291f2`**
(`57291f2644af8c9df0dd8d44395881c5bdcf0ecd`), 331 commits later, and those
commits were **not** re-diffed against this document. Re-read at `57291f2` were
only `deepseek4.cpp`'s `load_arch_tensors` shapes and the hyper-connection
stream seed in its graph prologue; see `backlog.md` § B-DSHW-PULL.

The document was originally written against `b10218-1-gc629da5`
(`c629da565c80b0b17fac6262acdca4d772e745d8`). The 2026-08-11 pull advanced the
checkout by 139 commits and touched all four files this document draws from, so
each maths claim was re-checked at the new pin: the trunk YaRN pre-scale, the
lightning indexer, the V4 hyper-connection helpers and `build_moe_ffn`'s router
bias are all **unchanged**. What did move — a second YaRN copy in the new
`graph_mtp`, the SwiGLU-clamp arm widening to `LLM_ARCH_DFLASH`, and the V3.2
layer-count switch — is itemised in `backlog.md` § B-DSHW-PULL.

Re-verified against both trees on 2026-08-05. That pass corrected the stage-1
rope-type mapping (it prescribed a permute that would have corrupted output),
renamed the router fields to where they actually live, and closed the Metal
shared-expert and LayerNorm questions. Everything else below survived the check
unchanged, including the llama.cpp line counts and every GGUF key name.

The generic procedure for adding any architecture — dump, diff, register, load,
graph, verify — is
[plan.md § Adding a model architecture](plan.md#adding-a-model-architecture-the-recipe).
This document is only what is DeepSeek-specific on top of it.

Disk paging is what stages 3–4 run on at all; it landed (`69b6de0`, `588653b`).
[backlog.md](backlog.md) B36 holds the paging optimizations that were measured
but **not** built, including the one §0.3 below depends on.

## Why this order

llama.cpp keeps **five** DeepSeek architectures, not one. They are separate
model classes with separate builders (`llama-model.cpp`, the
`LLM_ARCH_DEEPSEEK*` arms of `llama_model::create`):

| GGUF `general.architecture` | models                         | llama.cpp builder             |       size |
| --------------------------- | ------------------------------ | ----------------------------- | ---------: |
| `deepseek`                  | DeepSeek-LLM 7B/67B, MoE-16B   | `src/models/deepseek.cpp`     |  194 lines |
| `deepseek2`                 | V2, V2-Lite, V3, V3-0324, V3.1 | `src/models/deepseek2.cpp`    |  438 lines |
| `deepseek2-ocr`             | DeepSeek-OCR                   | `src/models/deepseek2ocr.cpp` |          — |
| `deepseek32`                | V3.2                           | `src/models/deepseek32.cpp`   |  506 lines |
| `deepseek4`                 | V4-Flash, V4-Pro               | `src/models/deepseek4.cpp`    | 1203 lines |

The staging follows from one hard constraint: **only the first two stages have a
model small enough to develop against.**

| stage | arch         | development model                                | fits a 24 GB card?   |
| ----- | ------------ | ------------------------------------------------ | -------------------- |
| 1     | `deepseek`   | `deepseek-llm-7b-chat`, `deepseek-moe-16b-chat`  | yes (~4 / ~10 GB Q4) |
| 2     | `deepseek2`  | `DeepSeek-V2-Lite-Chat` (16B total, 2.4B active) | yes (~10 GB Q4)      |
| 3     | `deepseek32` | **none — V3.2 is 671B**                          | no                   |
| 4     | `deepseek4`  | V4-Flash Q2_K_S (98.6 GB), paged from disk       | yes, CPU only        |

GGUFs for the stage-1/2 models exist (TheBloke, mradermacher, legraphista).
Stages 3 and 4 can only ever be exercised through the disk pager at 80+ GB, at a
few tokens/sec, with no CPU oracle that finishes in reasonable time. **So stages
1–2 must leave behind MLA and MoE-routing pieces that are independently tested,
because from stage 3 on there is no cheap way to find a bug.**

`deepseek2-ocr` is out of scope; it is a vision model that reuses the arch id.

## What `infr` already has

Verified against this tree. There is **no plugin system** — an architecture is a
set of fields on one `Config`, branched on inside one graph builder and one
weight-load loop.

Already present and directly reusable:

Note on naming: there is no `MoeConfig` type. Everything the router is
configured by lives as **fields on the `Op::MoeFfn` variant** in
`infr-core/src/graph.rs` (`gating`, `norm_w`, `scale`, `n_expert`, `n_used`,
`n_ff_exp`, `down_scale`, `fused_gate_up`, `weight_before`, `ep_band`) — grep
for the variant, not for a struct.

- **Sigmoid MoE gating** — `MoeGating::Sigmoid` (`infr-core/src/graph.rs`),
  CPU + Vulkan. This is V3's `scoring_func`.
- **Gate-weight normalisation on/off** (`Op::MoeFfn`'s `norm_w`) — DeepSeek's
  `norm_topk_prob`.
- **Routed scaling factor** (`Op::MoeFfn`'s `scale`), read from
  `{arch}.expert_weights_scale` — the same GGUF key DeepSeek uses.
- **Shared experts, plain-summed** — `FfnW::Moe { shexp }` with tensor names
  `ffn_{gate,up,down}_shexp`. DeepSeek's shared expert is plain-summed, so the
  llama4 path fits.
- **Expert count headroom** — the Vulkan `moe_topk.comp` supports up to 1024
  experts; V3/V4's 256 fit.
- **Expert paging** — `infr-core/src/pager.rs` + `hostpager.rs` +
  `infr-vulkan/src/pager.rs`, keyed `(layer, role, expert_id)`.
- **Partial RoPE** — `Op::Rope` rotates the first `rope_dim` of each head and
  passes the rest through, which is exactly DeepSeek's decoupled rope shape.
- **Per-layer heterogeneity** — `Config::layer_head_dim`, `layer_n_kv`,
  `layer_rope_theta`, `is_swa_layer`, `is_moe_layer` and friends already exist.
- **Low-rank projections** need no new op: `Linear → RmsNorm → Linear` covers
  `wq_a → q_a_norm → wq_b`.
- **All the weight quants** V4 ships — every IQ1/IQ2/IQ3/IQ4 and k-quant has a
  native Vulkan dense kernel.

What this inventory once listed as missing — MLA attention, group-limited
routing, the router selection bias, YaRN, an indexer, and a first-N-dense
`is_moe_layer` — has all landed; the stage sections below say how, and what each
is verified against. What is still missing:

- **The V4 compressed-KV tiers** (`compress_ratio` 4 and 128) — see § Stage 4
  and `backlog.md` § B-DSV4-WIRING.
- **Metal cannot run a DeepSeek MoE layer at all.** Its `Op::MoeFfn` arm
  implements softmax gating + top-k renorm + output-weighting and asserts on
  anything else, so V2-Lite (`norm_topk_prob = false`) fails that assert, and V3
  (sigmoid) fails it too. It also reads neither `exp_probs_b` nor the
  group-routing fields — softmax + renorm + `expert_group_count > 1` is a legal
  `deepseek2` config that used to pass the assert and then route with neither
  applied, silently; that combination now asserts as well. DeepSeek is CPU +
  Vulkan for the MoE layers until a Metal router gains those inputs. MLA
  attention itself IS implemented on Metal.
- **Metal has no `MoeSharedExpertAdd` kernel** (`infr-metal/src/exec.rs` refuses
  it by name), but **this does not bite DeepSeek**: that op is only emitted for
  a per-token-sigmoid-gated shared expert (qwen35moe); an ungated one is summed
  in plain with `Op::Add`, which is the llama4 path and exists on all three
  backends — see `FfnW::Moe`'s `shexp` doc in `seam/weights.rs` and
  `Config::shexp_gated`.

### Places a new arch string must be registered

1. `infr-llama/src/arch.rs` — the `pub const`, plus `arch::TRANSFORMER` and
   `arch::ALL`. Neither list gates anything: `TRANSFORMER` is read once to
   render the rejection message (`config.rs`), and `ALL` currently has **no
   consumer at all**. They are documentation that happens to compile — adding to
   them does not make the arch load.
2. `infr-llama/src/config.rs`, the `match arch.as_str()` inside
   `Config::from_gguf` — **this is the load gate**; an unknown arch fails here
   and nowhere else.
3. `infr-cli/src/main.rs`, `arch_sampling` — recommended sampling defaults
   (optional; falls back to `(0.6, 20, 0.95)`).
4. `infr-llama/src/tokenizer.rs`, the `match pre` — see below, this one is
   required for DeepSeek.

The two commits worth reading as templates are `5b44ef9` (BitNet — the floor for
a config-only arch, 5 files) and `e24399d` (llama4 Scout — the shape of an arch
that needs new op semantics, 15 files including `graph.rs` and all backends).

## Stage 0 — prerequisites, before any architecture

These are independent of which DeepSeek you start with, and two of them are
things you would otherwise discover as silent wrongness.

### 0.1 Dump a real GGUF first

`crates/infr-gguf/examples/dump.rs` exists for this. Run it on a real DeepSeek
GGUF before planning around anything below. Specifically confirm: the ggml type
ids used (anything outside `ggml_type_to_dtype` in `infr-gguf/src/lib.rs` fails
at `Gguf::open`), the exact metadata keys, and the head layouts.

### 0.2 Tokenizer — a real gap, not a formality

llama.cpp maps three DeepSeek pre-tokenizers (`src/llama-vocab.cpp`):

| `tokenizer.ggml.pre` | patterns | `clean_spaces` |
| -------------------- | -------: | -------------- |
| `deepseek-llm`       |        6 | false          |
| `deepseek-coder`     |        5 | false          |
| `deepseek-v3`        |        3 | false          |

V3 uses `deepseek-v3`; V4 reuses it (there is no `deepseek-v4` pre-type).

llama.cpp applies these as an **ordered list of successive splits**. `infr`
implements this in `build_multi_split_seq` (`infr-llama/src/tokenizer.rs`): N
`Split` pre-tokenizers in `Isolated` mode followed by `ByteLevel`, with the
regex lists in `infr-llama/src/util.rs` (`DEEPSEEK_LLM_PRE_RES`,
`DEEPSEEK_CODER_PRE_RES`, `DEEPSEEK_V3_PRE_RES`).

**Collapsing the list into one alternation is not equivalent** — and the
successive form is no longer an assumption either. It was checked against
`llama-tokenize` token ids; see open question 3, which is now resolved.

All fourteen regexes were diffed codepoint by codepoint against
`llama-vocab.cpp` (2026-08-09) and now match it exactly. Two did not, and the
symptom of each is instructive: a wrong character inside a class compiles, runs,
raises nothing, and merely moves a chunk boundary.

- `DEEPSEEK_LLM_PRE_RES[2]` opened its quote range with `'` (U+0027) where the
  reference has `‘` (U+2018), so the class ran U+0027–U+201F instead of the
  eight quote characters — most of the BMP, including the ASCII digits, Hebrew,
  Arabic and Devanagari.
- `DEEPSEEK_LLM_PRE_RES[1]` had `ℹ-ℿ` where the reference has `ℹℼ-ℿ`, adding
  U+213A and U+213B to the letter class.

`clean_spaces = false`, which llama.cpp sets for all three DeepSeek pre-types,
**needs no equivalent here**. It is read in exactly one place —
`llama_vocab::impl::detokenize` in `llama-vocab.cpp` — where it drops the space
before `?!.,`, strips a lone apostrophe between spaces, and closes up
`'s`/`'m`/`'re`/`'ve`. It is a detokenizer post-process (HF's
`clean_up_tokenization_spaces`) and never runs during encoding, so it cannot
affect token ids. llama.cpp defaults the flag to `false`, turns it ON for every
BPE vocab, and the DeepSeek arms turn it back off. `infr` detokenizes through
`tokenizers::Tokenizer::decode`, and that crate contains no such pass at all —
so `infr` is unconditionally in the `clean_spaces = false` state DeepSeek wants.
(It is therefore also unconditionally in that state for BPE pre-types where
llama.cpp leaves the flag ON. That is a display-only divergence, not an id one,
and it is out of scope here.)

The lists are guarded by
`deepseek_pre_split_boundaries_match_the_reference_lists` in `tokenizer.rs`,
which pins the chunk boundaries all three produce and needs no model file.

Getting this wrong degrades output with **no error at all**, which is why it is
stage 0.

### 0.3 Pager LRU

Done: `Pager` keeps its state by slot with recency as a linked list, so a miss
walks from the cold end past the protected blocks only, instead of scanning
every resident block (`docs/plan/06-pager-eviction.md`, step 2, which has the
measurements at V4-Flash-scale pool sizes).

## Stage 1 — `deepseek` (V1)

**Smallest possible first step, and it buys a real model.** No MLA, no YaRN, no
indexer. Plain MHA plus DeepSeek-style MoE.

### Hyperparameters (`{arch}.` prefixed)

| key                                | maps to                              |
| ---------------------------------- | ------------------------------------ |
| `attention.layer_norm_rms_epsilon` | RMS eps                              |
| `leading_dense_block_count`        | `n_layer_dense_lead`                 |
| `expert_feed_forward_length`       | `n_ff_exp`                           |
| `expert_shared_count`              | `n_expert_shared`                    |
| `expert_weights_scale`             | routed scaling (absent/0 = unscaled) |

V1 has **no** `expert_gating_func` and **no** `expert_weights_norm` — it
hardcodes softmax scoring and no normalisation.

### Attention

Vanilla MHA. `n_embd_head_v == n_embd_head_k == n_rot` (llama.cpp asserts it).
Full-dim rope on Q and K, rope type **NORM** (interleaved consecutive pairs).
`kq_scale = 1/sqrt(n_embd_head)`.

**Do NOT set `Config::permute_qk_neox`.** All five DeepSeek arches return
`LLAMA_ROPE_TYPE_NORM` from `llama_model_rope_type` (`src/llama-model.cpp`) —
the same arm as `LLM_ARCH_LLAMA`, not the NEOX arm that `LLM_ARCH_QWEN2` sits
in. The permute exists to make infr's interleaved (NORM) `Op::Rope` reproduce
**NEOX** for an arch whose GGUF stayed in HF rotate-half order (qwen2, bitnet —
see the field's own doc in `config.rs`). DeepSeek is already NORM with
converter-permuted rows, exactly like llama, so permuting would rotate the wrong
pairs and produce fluent nonsense. This applies to stage 2's `q_pe`/`k_pe` as
well; the one NEOX rope in the family is the V3.2 indexer (stage 3), which is
hardcoded NEOX against a NORM main rope.

Nothing new is required in the IR.

### MoE

`softmax(logits)` → top-k → gather → **no** normalisation → `× scale` → SwiGLU
experts → add shared expert. All of this exists.

### What stage 1 had to add (landed)

- Arch registration (§ above).
- `is_moe_layer` has a **first-N-dense threshold** mode alongside the periodic
  one.
- Shared-expert width: llama.cpp allocates `ffn_*_shexp` as
  `{n_embd, n_ff_exp * n_expert_shared}` — one fused branch of `n_expert_shared`
  experts' width. `infr` models exactly one branch of width `shexp_ff`, so
  `shexp_ff = n_ff_exp * n_expert_shared` lines up — confirmed on V2-Lite
  (`n_shared_experts = 2`), whose next-token distribution matches llama.cpp (§
  Stage 2's checklist).

### Done when

- `cpu_deepseek_config` — opens the GGUF, asserts the arch string, asserts every
  gate boolean including that other arches' gates are false. (Pattern:
  `cpu_bitnet_config` in `infr-llama/tests/cpu_backend.rs`.)
- `cpu_deepseek_prefill_paris` — top-1 token after "The capital of France is".
  (Pattern: `cpu_bitnet_prefill_paris`.)
- `gpu_seam_matches_cpu_deepseek` — token-identical CPU vs Vulkan, `#[ignore]`d
  behind a GPU. Use the **strict** form for the 7B dense model and the **loose**
  form (top-5 overlap + `cosine > 0.5`) for MoE-16B, for the same reason
  `gpu_seam_matches_cpu_qwen35moe` does: routing near-ties legitimately flip.
  **Not written.** The first two exist and run against
  `mradermacher/deepseek-moe-16b-chat-GGUF` (the dense 7B declares `llama`, so
  the "strict form for the 7B" half no longer applies); this one is still owed.

**No CI golden.** The `cpu-goldens` job downloads two ~1B GGUFs; the smallest
DeepSeek is 7B. Commit `273f8d4` already removed qwen35 from that job because an
exact-token golden did not reproduce across machines.

## Stage 2 — `deepseek2` (V2, V2-Lite, V3, V3.1)

The big one, and the one with a 16 GB development model. Everything here is
reused by stage 3 almost verbatim.

### Hyperparameters

Adds to stage 1: `attention.q_lora_rank`, `attention.kv_lora_rank`,
`attention.key_length_mla`, `attention.value_length_mla`, `expert_weights_norm`,
`expert_gating_func` (1 softmax / 2 sigmoid / 3 softmax-on-weights / 4
sqrt-softplus), `rope.scaling.yarn_log_multiplier`, `expert_group_count`,
`expert_group_used_count`.

Derived: `key_length = kv_lora_rank + qk_rope_head_dim` (576 for V3),
`value_length = kv_lora_rank` (512), `head_count_kv = 1`,
`rope.dimension_count = qk_rope_head_dim` (64). **Read these from a GGUF rather
than trusting the numbers here** — they were derived from the conversion
script's formulas, not read out of a file.

Two loader details worth copying exactly:

- **`rope_yarn_log_mul /= 0.1`** on load. The convert script writes
  `0.1 * mscale_all_dim`; the loader divides it back out. Double-applying or
  double-cancelling this is a silent long-context quality bug.
- **"lite" detection.** llama.cpp has a heuristic on layer count and vocab size,
  but the graph actually decides on **tensor presence** (`wq` present ⇒ lite,
  else `wq_a`/`wq_b`). Port the tensor-presence test and drop the heuristic.

### Tensors

Per layer, beyond the stage-1 set:

| tensor            | GGUF name                      | shape                                       |
| ----------------- | ------------------------------ | ------------------------------------------- |
| `wq_a`            | `blk.%d.attn_q_a.weight`       | `{n_embd, q_lora_rank}`                     |
| `attn_q_a_norm`   | `blk.%d.attn_q_a_norm.weight`  | `{q_lora_rank}`                             |
| `wq_b`            | `blk.%d.attn_q_b.weight`       | `{q_lora_rank, n_head * head_k_mla}`        |
| `wq` (lite)       | `blk.%d.attn_q.weight`         | `{n_embd, n_head * head_k_mla}`             |
| `wkv_a_mqa`       | `blk.%d.attn_kv_a_mqa.weight`  | `{n_embd, kv_lora_rank + qk_rope}`          |
| `attn_kv_a_norm`  | `blk.%d.attn_kv_a_norm.weight` | `{kv_lora_rank}`                            |
| `wk_b`            | `blk.%d.attn_k_b.weight`       | `{qk_nope, kv_lora_rank, n_head}`           |
| `wv_b`            | `blk.%d.attn_v_b.weight`       | `{kv_lora_rank, v_head_dim, n_head}`        |
| `wo`              | `blk.%d.attn_output.weight`    | `{n_head * v_head_dim, n_embd}`             |
| `ffn_exp_probs_b` | `blk.%d.exp_probs_b.bias`      | `{n_expert}` — optional, V3's noaux_tc bias |

**`wk_b` is transposed relative to the HF weight and `wv_b` is not.** The
conversion script splits `kv_b` into `k_b`/`v_b` and calls `.transpose(1, 2)` on
`k_b` only. This is the classic MLA porting bug — getting it backwards produces
plausible-looking garbage.

### The attention, step by step

Head layout is **`[nope | rope]`, nope first**, on Q. The KV projection is
`[latent(512) | rope(64)]`.

```
q      = wq_b · RMSNorm_{q_a_norm}(wq_a · x)      # or wq · x when lite
q_nope = q[0 .. qk_nope]                          # per head
q_pe   = q[qk_nope .. qk_nope+qk_rope]

kv_cmpr_pe = wkv_a_mqa · x                        # {512+64, n_tokens}
kv_cmpr    = kv_cmpr_pe[0 .. 512]                 # the latent
k_pe       = kv_cmpr_pe[512 .. 576]               # ONE rope head, shared by all query heads

q_pe = rope(q_pe, n_rot=64, NORM)
k_pe = rope(k_pe, n_rot=64, NORM)
kv_cmpr = RMSNorm_{kv_a_norm}(kv_cmpr)            # AFTER the split, BEFORE absorption
```

`attn_kv_a_norm` applies **only** to the 512-wide latent, not to `k_pe`.
`attn_q_a_norm` sits **only** between `wq_a` and `wq_b`.

Then the absorbed form:

```
q_nope_absorbed = wk_b[h]ᵀ · q_nope[h]            # {128} -> {512}, per head
Q = concat(q_nope_absorbed, q_pe)                 # {576, n_head}
K = concat(kv_cmpr, k_pe)                         # {576, 1}
V = kv_cmpr                                       # {512, 1}  -- an ALIASED PREFIX VIEW of K
out = wv_b · attn(Q, K, V)                        # wv_b applied to the OUTPUT, {512}->{128} per head
```

Three things that will each silently corrupt output:

1. **Only one 576-wide row is cached per token per layer.** There is no separate
   V cache; V is the first 512 columns of K.
2. **`wv_b` is applied after the KQV product**, not to the cache.
3. **`kq_scale` divides by `head_k_mla` (192), not by 576** and not by the
   concatenated Q width.

There is also an "unabsorbed" legacy path for older GGUFs carrying `wkv_b`
instead of `wk_b`/`wv_b`: it up-projects to full MHA, broadcasts `k_pe` across
all heads, and uses ordinary K/V caches. **Skip it** unless a target GGUF needs
it — it is a much larger cache for identical output.

### YaRN

Two stages that must not double-apply:

```
attn_factor_org = attn_factor * (1 + 0.1·ln(1/freq_scale))
mscale          = attn_factor_org * (1 + 0.1·rope_yarn_log_mul·ln(1/freq_scale))
kq_scale        = mscale² / sqrt(head_k_mla)
```

The mscale² is folded into the **softmax scale**, not applied to Q, because
ggml's rope already applies `attn_factor` to the rotated slice and the first
line undoes it.

For `infr`, the frequency ramp itself should fold into the existing
`freq_factors` mechanism (`Op::Rope`'s optional per-pair divisor, already used
by gemma4's proportional rope): precompute the ramp on the host at load and bind
it like `rope_freqs.weight`. **This is an inference from the maths, not
something either codebase states — validate numerically against llama.cpp before
relying on it.**

### MoE

Order matters:

1. `logits = gate_inp · x`
2. score: softmax / sigmoid / sqrt-softplus per `expert_gating_func`
3. **`selection_probs = probs + exp_probs_b`** — bias affects _selection only_;
   the returned weights are read from the **unbiased** `probs`
4. group-limited routing when `n_expert_groups > 1`: per group take the **top
   2** scores and sum them for a group score, take the top `n_group_used`
   groups, mask the rest to `-inf`. The top-2-within-group is **hardcoded** in
   llama.cpp and matches V3, but is not the general `topk_group` formulation.
5. top-k over `selection_probs`
6. gather weights from `probs`
7. if `norm_w`: divide by `clamp(sum, 6.103515625e-5, inf)` — the clamp is the
   smallest normal f16
8. if `scale != 0 && != 1`: multiply
9. SwiGLU experts, weighted sum
10. add the shared expert; first `n_layer_dense_lead` layers instead run a plain
    dense SwiGLU with `n_ff` (not `n_ff_exp`) and no shared expert

### What stage 2 had to add (landed)

- **MLA in the IR**, as its own `Op::Mla` rather than asymmetric K/V dims on
  `Op::Attention`. This touches KV-cache sizing (`seam/mod.rs`,
  `seam/weights.rs`, `runner.rs` all compute `layer_n_kv * layer_head_dim`) and
  every attention kernel on CPU, Vulkan and Metal. It was the single biggest
  item in the plan, and was prototyped on CPU alone first.
- `MoeFfn` gains an `exp_probs_b` input and group-routing fields; the Vulkan
  `moe_topk.comp` must select on the biased score and weight from the unbiased.
- YaRN ramp precomputation at load.
- A new `MixerW::Mla(MlaW { .. })` variant plus a third branch in each of the
  three lockstep loops in `runner.rs` (`wload`, `wpush`, emit). The file's own
  comments say these MUST mirror; getting them out of step is silent corruption.

### Done when

- [x] Config + CPU-finite + CPU-top-token tests on V2-Lite
      (`cpu_deepseek2_config`, `cpu_deepseek2_prefill_finite`,
      `cpu_deepseek2_prefill_paris` — all added 2026-08-06, gated behind model
      file).
- [x] `gpu_seam_matches_cpu_deepseek2` — skeleton added 2026-08-06; passing on
      the GPU box 2026-08-07 (CPU-vs-Vulkan cosine 0.9955, matching top-5) after
      the YaRN ramp + wk_b/wv_b orientation fixes.
- [x] `cpu_deepseek2_golden` — hash-locked generation, blessed from the coherent
      post-fix output (2026-08-07).
- [x] **An op-level MLA parity test** in `infr-llama/tests/seam_op_parity.rs`
      against a hand-written CPU reference, following `deltanet_parity`. This is
      the one that matters: it is the only cheap check that survives into stages
      3–4.
- [x] **A numeric YaRN check against llama.cpp at a long context** — done
      2026-08-07 on the V2-Lite Q4_K GGUF vs llama.cpp `c629da5` (CPU
      reference):
  - 228-token prompt, infr CPU prefill vs llama.cpp last-row logits: cosine
    0.978, greedy token identical (185).
  - 4560-token prompt (positions past `n_ctx_orig`=4096, in the ramp region),
    infr Vulkan prefill vs llama.cpp: cosine 0.860, greedy token identical
    (549). The seam's ff divisors / mscale are context-independent (llama.cpp
    runs the full ramp at every context length), so the short-CPU and long-GPU
    runs exercise the same numbers; both greedy tokens match.
  - **Ignore those two cosine numbers; the greedy tokens are the result.** This
    entry used to say both sat "in the established deepseek2 infr-vs-llama.cpp
    range (~0.79–0.91)". That range means nothing. Measured 2026-08-11 by
    `cpu_prefill_matches_llama_debug_dump`, on the same V2-Lite GGUF and over
    llama.cpp's own token ids: a **correct** match — both engines picking token
    8913 (" Paris"), probability cosine 0.9969 — scores a logit cosine of only
    **0.774**, while two **unrelated** rows of Qwen3-0.6B score **0.851**. A
    whole-vocab logit cosine is dominated by the per-token bias every row of a
    model shares, so a correct row can score below an unrelated one and the
    metric cannot separate them. Score agreement on mutual top-5 containment and
    a cosine over softmax **probabilities** (0.9969 vs 0.0164 for that unrelated
    pair). The greedy-token identity above is real evidence and stands.
- [x] Metal MLA kernel — `mla_f16kv` in `attention.metal` + `exec.rs` dispatch
      (2026-08-06; ported from `mla.comp`, f16 KV cache), plus the YaRN
      `mla_f16kv_ff` twin. Executed for the first time by `mla_parity` /
      `mla_ff_parity` in the Metal parity suite on the macOS CI job (2026-08-07)
      — which also caught and fixed an ff/params buffer-index swap in the kernel
      declaration.
- [x] **The doubled MLA residual** — found 2026-08-12 with a per-op tap on the
      CPU interpreter compared against `llama-debug --verbose`'s per-tensor
      `sum =` lines. `MixerW::Mla` pushed its own `hidden += sub` and then fell
      through to the shared post-mixer residual add, so every layer computed
      `x + 2·Wo·attn`. At layer 0 of a ten-token prefill, llama.cpp's
      `ffn_inp-0` last row reads `[-0.0645, 0.0322, -0.1029, …]` and infr's
      residual read `[-0.0921, 0.0076, -0.2065, …]` =
      `[-0.0636, 0.0309, -0.1029, …]` plus the attention output a second time.
      Everything upstream of it matched llama.cpp to four printed decimals,
      `k_pe` rope included. Next-token probability cosine against llama.cpp at
      ten tokens: 0.3336 → 0.9998 (CPU), 0.3156 → 0.99998 (Vulkan). Guarded by
      `synthetic_deepseek2_attention_enters_the_residual_once`, which recomputes
      a one-token MLA layer by hand — the only kind of check that can see it, as
      a doubled residual is indistinguishable from a `Wo` scaled by 2 to any
      test that only varies weights. Two goldens moved with it and were
      re-blessed in the same change, both because the residual stream they were
      blessed off was wrong: `cpu_deepseek2_golden` (V2-Lite now generates
      `" Paris."`, llama.cpp's own greedy continuation of that prompt at pin
      030ebb5, token for token) and `GOLDEN_DS32_TOPK5` /
      `synthetic_deepseek32_indexer_selection_is_locked` (the indexer's queries
      and keys are both projections of the attn-normed residual, so on every
      layer but the first the blessed key set came off the doubled stream). The
      indexer's own mechanism tests —
      `synthetic_deepseek32_top_k_restricts_attention` and
      `synthetic_deepseek32_full_top_k_matches_no_indexer_at_all` — were green
      before and after, and are what still holds that arithmetic in place.
- [x] YaRN per-dimension frequency ramp in `Op::Rope` and MLA kernels — the
      `freq_factors` divisors (`ff[p] = 1/s(p)` from the corr_dims spectral
      ramp) + the constant `mla_scale = mscale²/√(qk_nope+qk_rope)` landed in
      `784704e` (2026-08-07); verified numerically above.

## Stage 3 — `deepseek32` (V3.2)

**Progress: complete — a `deepseek32` model loads, builds a graph and
generates.** The indexer is wired end to end on CPU and Vulkan (executed on an
RX 7900 XTX under the validation layer) and typechecks on Metal, whose first
real execution is the macOS CI job's parity suite. What landed, on top of the
already-tested `Op::LayerNorm` and `Op::LightningIndexer`:

- **`Op::Rope` gained `neox`.** The indexer's rope is NEOX where the MLA rope
  beside it is NORM; `Op::Rope` had only the interleaved NORM pairing (the NEOX
  one existed solely fused inside `Op::QkNormRope`). CPU branches on the field;
  Vulkan compiles two more `rope.comp` builds (`rope_neox`, `rope_ff_neox`) and
  REFUSES `neox` on the f16-out / record-once paths, which have no such build;
  Metal carries it as a `RopeParams` field.
- **`Op::Mla` gained `key_bias`**, an optional additive `[rows, kv_len]` score
  mask, and **`Op::TopkMask`** expands the indexer's indices into it. `None` for
  `deepseek2`, where every backend takes exactly the code it took before.
- **The indexer's second KV cache rides the V side MLA leaves empty** — see
  `seam::kv_row_elems`, which answers
  `(kv_lora_rank + qk_rope_dim, indexer_head_size)` for `deepseek32`. Every
  geometry site (allocation, graph declaration, `fork`, `seed_from`, both VRAM
  estimates) therefore sizes and copies it without any of them growing a private
  branch. The allocation asserts the cache does not ring, because
  `Op::LightningIndexer` masks causally only.
- **The refusal is gone**, replaced by an assertion on `MlaW::indexer`: a
  `deepseek32` model whose indexer weights were not captured fails at the emit
  instead of degrading to the deepseek2 graph.

### The cache and mask decisions, with their costs

**Second cache on the V side.** One `indexer_head_size`-wide f16 row per token
per layer: on V3.2 that is 128 halves = 256 B/token/layer against MLA's 576
halves = 1152 B, so the KV footprint grows ~22%. It cannot ring (see above), so
it is sized at the full context like any non-SWA side.

**Mask, not gather.** `Op::Mla` takes the `[rows, kv_len]` f32 mask; the inner
loop pays one extra f32 load per key and the graph pays `rows × kv_len` floats
of scratch (16 MiB at a 512-row ubatch and an 8192 context — the same order as
the indexer's own score scratch, which was already there). The alternative,
membership-testing the index list inside the MLA kernel, needs no scratch but
scans `top_k` (≈2048 on V3.2) per (row, head, key) — three orders of magnitude
worse on the inner loop. The mask is also what llama.cpp materialises, so the
numerics are comparable term by term. **The FLOP saving is still not realised**;
a gather remains a pure optimisation on top.

### What is verified, and what cannot be

All verification of the indexer is synthetic — `tests/synthetic_deepseek2.rs`
builds a `deepseek32` GGUF from the same description as the `deepseek2` one plus
an indexer. A real V3.2 file exists locally, but it declares `deepseek2` and
carries no indexer tensors, so it cannot exercise any of this; the public files
that do declare `deepseek32` are far larger than this box can run.

- `synthetic_deepseek32_cpu_prefill_is_finite` — the whole model runs.
- `gpu_synthetic_deepseek32_matches_cpu` — CPU vs Vulkan: cosine
  `0.9999999999998446`, same top token, on the GPU box under the Khronos
  validation layer with zero VUID lines. Everything the indexer adds executes
  there: `layernorm.comp`, `rope.comp`'s `-DNEOX` build, the second `WriteKv`,
  `lightning_indexer.comp`, `topk_mask.comp`, `mla.comp`'s `-DKEY_BIAS` build.
- `synthetic_deepseek32_full_top_k_matches_no_indexer_at_all` — with
  `top_k >= kv_len` the mask is all-zero and the CPU logits are
  **bit-identical** (`max|Δ| = 0`) to the same model with no indexer at all.
  This is the property that makes the mask faithful, and it is also what proves
  the indexer perturbs nothing else. Its Vulkan twin can only assert ~1 ULP
  (`9.536743e-7`): the masked build's score is `red[0]*scale + kbias[j]`, which
  the compiler fuses into one FMA where the unmasked build rounds the product
  separately, so the two pipelines differ by a rounding even at `kbias == 0`.
- `synthetic_deepseek32_top_k_restricts_attention` — at `top_k = 5` of 8 keys
  the output moves by `max|Δ| = 2.52` against a logit rms of `1.11`. A version
  of this suite without that case would pass with the mask never reaching the
  kernel.
- `rope_neox_and_norm_parity`, `topk_mask_parity`,
  `mla_key_bias_removes_the_masked_keys` in `seam_op_parity.rs` — op-level, CPU
  vs from-definition references and CPU vs Vulkan. The `key_bias` case checks
  the equivalence that matters: masking a key is exactly the key not being in
  the cache.
- `synthetic_deepseek32_indexer_selection_is_locked` — a tolerance-based lock on
  the restricted run's logits. It is a REGRESSION lock blessed from this
  implementation, not an independent oracle; what it buys is that the indexer's
  NEOX pairing and its `[rope | nope]` head order cannot change silently.

Each of the three traps was applied as a perturbation, seen to fail, and
reverted (2026-08-10, CPU, `--test synthetic_deepseek2`; the lock's own scale is
logit rms `1.0331706e0`):

| perturbation                                     | what went red                    |            max\|Δ\| |
| ------------------------------------------------ | -------------------------------- | ------------------: |
| indexer roped NORM instead of NEOX               | the selection lock               |       `8.557266e-2` |
| indexer head read as `[nope \| rope]`            | the selection lock               |       `1.0638273e0` |
| `Op::Mla::key_bias` passed `None` (mask dropped) | the lock AND `top_k_restricts_…` | `2.5244246e0` / `0` |

The third row is the important one: with the mask dropped, `top_k = 5` and
`top_k >= kv_len` become **indistinguishable** (`max|Δ| = 0`), which is exactly
the failure a suite without that case would have shipped.

**Not verified, and not verifiable here:** anything against llama.cpp's own
numbers. There is no V3.2 GGUF that carries the indexer and no CPU oracle, so
the indexer's absolute scores, the top-k it picks on a real model, and the
resulting text are all unchecked. The synthetic harness proves the pieces agree
with their own specifications and with each other across backends; it cannot
prove the specification matches DeepSeek.

**Metal has not executed any of this.**
`cargo check -p infr-metal --target x86_64-apple-darwin` typechecks the new arms
and the four `mla_f16kv*` entry points; the macOS CI parity job is the first
thing that runs them. (And a `deepseek32` MoE layer cannot run on Metal at all
yet, for the router reasons in "What `infr` already has".)

**~80% of this is stage 2 copied verbatim.** llama.cpp's `deepseek32.cpp` is
deepseek2's absorbed MLA path plus the lightning indexer. Non-MLA is rejected
outright. No small model exists; budget for slow iteration.

Adds: `attention.indexer.head_count`, `attention.indexer.key_length`,
`attention.indexer.top_k`, and `f_norm_eps` hardcoded to `1e-6`.
`expert_gating_func` is **mandatory** here (no fallback), and `q_lora_rank` is
mandatory (no lite variant).

### The lightning indexer

Per layer, unconditionally. It computes a scalar relevance score per (query
token, key token) and keeps the top-k keys for the real attention.

```
w[h, t]     = (indexer_proj · x)[h, t] / sqrt(index_head_dim · index_n_heads)
score[t, j] = Σ_h  w[h, t] · ReLU( q[h, t] · k[j] )  + causal_mask[t, j]
top_k       = argsort_top_k(score, min(n_kv, index_topk))
```

Note: **one key head shared by all indexer query heads** (MQA), the **ReLU is
inside the head-weighted sum**, and the `1/sqrt(d·H)` normaliser is pre-folded
into `w` to avoid scaling a huge score tensor.

New tensors: `indexer.k_norm.{weight,bias}`, `indexer.proj.weight`,
`indexer.attn_k.weight`, `indexer.attn_q_b.weight`.

Traps, each of which produces silent wrongness:

- **The indexer's rope type is NEOX, hardcoded**, while the main MLA rope is
  NORM. Same width, same frequencies, different pairing. **Done**: `Op::Rope`
  carries a `neox` flag on all three backends (§ Stage 3 progress above).
  `Config::permute_qk_neox` is NOT the tool — it is model-wide and keyed on
  `attn_q`/`attn_k` tensor names, so it cannot say "this one projection ropes
  NEOX while the model's main rope is NORM" (see § Stage 1).
- **The indexer head layout is `[rope | nope]`** — the _opposite_ of the MLA
  head. Worse, llama.cpp writes the nope view's offset as `row_size(nope)`
  rather than `row_size(rope)`; these coincide only because both are 64 for
  V3.2. **Ported as the layout says**: rope occupies each head's first `n_rot`
  dims, which is what makes both indexer ropes a plain `Op::Rope` with no offset
  and no split (that op rotates `[0, rope_dim)` of each head and passes the tail
  through). This port therefore does not depend on the coincidence at all, and
  the synthetic fixture runs at `head_size = 24, n_rot = 16` — nope 8 ≠ rope 16,
  a geometry where the two readings genuinely differ. Note the consequence: on a
  hypothetical V3.2-shaped model with nope ≠ rope, llama.cpp's view arithmetic
  and this port would disagree, and llama.cpp would be the one reading a
  misaligned nope.
- `indexer_k_norm` is a real **LayerNorm with bias** (mean-centred), the only
  non-RMS norm anywhere in the family — `graph.rs` carried `RmsNorm` and
  `RmsNormAdd` and nothing mean-centred, so it needed a new op rather than a
  config flag. **Done**: `Op::LayerNorm` landed on CPU, Vulkan
  (`layernorm.comp`) and Metal (`layernorm_f32`), with `layernorm_parity` in
  `seam_op_parity.rs` checking it against a from-definition f64 reference. The
  arithmetic follows `ggml_compute_forward_norm_f32`: the BIASED variance
  estimator, `eps` INSIDE the sqrt, then `* weight` then `+ bias`. The
  `deepseek32` indexer emit is its only caller.
- The indexer keeps a **second, independent KV cache**: one
  `index_head_dim`-wide row per token per layer, on top of the 576-wide MLA
  cache. **Done**: it rides the V side MLA leaves empty (`seam::kv_row_elems`),
  and it must never ring.
- A **Hadamard rotation** is applied to q and k. It is an orthogonal transform
  applied identically to both, so dot products are preserved: it exists for
  quantisation friendliness and **can be skipped entirely** in an unquantised
  port.

### How top-k feeds attention

llama.cpp does **not** gather or compact. It builds a `-inf` mask everywhere
except the selected indices, adds it to the ordinary causal mask, and runs dense
attention over the full `n_kv`. The FLOP saving is not realised — only the
numerics are faithful.

**This is the interesting decision for `infr`.** A port that wants the actual
speedup must gather, and the selected indices are per (query token, stream), not
per head. Doing the mask version first is the safe order: it is checkable
against llama.cpp token-for-token, and the gather can follow as a pure
optimisation.

**Done, as the mask.** `Op::LightningIndexer` emits indices, `Op::TopkMask`
expands them into an additive `[rows, kv_len]` f32 mask (0 on the selected keys,
`-inf` elsewhere), and `Op::Mla::key_bias` adds it to the score. Costs are in §
Stage 3's "cache and mask decisions". `-inf` is safe rather than a large finite
negative because at least one key is always selected — `top_k >= 1`, and the
indexer ranks every causally-eligible key above every ineligible one — so each
row's softmax max is finite and `exp(-inf - max)` is 0 exactly.

## Stage 4 — `deepseek4` (V4-Flash / V4-Pro)

**Progress: every V4 layer tier generates on the CPU backend, under either
routing.** The ratio-0 tier is emitted end to end and runs on CPU and Vulkan;
the ratio-128 (HCA) and ratio-4 (CSA) tiers run on the CPU and match llama.cpp
there on synthetic files (see "The HCA tier" and "The CSA tier" below); a
compressed layer on a GPU backend is refused by name. **What a V4 file needs to
run now:** the CPU backend (`--dev cpu`), RAM for its weights (the shipped
V4-Flash Q2_K_S is 98.6 GB, which this machine's 64 GB cannot hold; a run pages
it from disk through the host weight arena, one routed expert at a time). The
CPU prefills a V4 prompt in chunks of many positions per graph, as it does every
other model (see "Batched prefill" below); the GPU backends prefill one position
per graph. The shipped file runs on the CPU since 2026-11-16: with a 12 GB arena
it prefills a 5-token prompt in 16.2 s and decodes at 1.28 s a token, nearly all
of it expert reads from disk, and it answered two prompts coherently
(`docs/backlog.md` § B-DSV4-REAL has the measurements). No number it computes
has been checked by any oracle: llama.cpp cannot load it on this machine. The
compressed caches default to f16 (what the oracle checks); a quantized KV format
is an explicit opt-in, unverified against the reference, which Hadamard-rotates
those rows before quantizing them. See "Slice A — ratio 0" below for what that
covers, "Slice A2 — hash routing" for the `ffn_gate_tid2eid` gather, and
`docs/backlog.md` § B-DSV4-WIRING for what is left. Note that no layer of the
SHIPPED V4-Flash file is both ratio-0 and bias-routed — see `docs/backlog.md` §
B-DSV4-REAL.

The 2026-08-10 read slice that preceded it was a read, not a port:
`llama-kv-cache-dsv4.cpp` had never been read in full, which made this section's
account of the compressed caches the least trustworthy part of the document. It
has been, and "The compressed-KV state machine" below is rewritten off it — the
real inventory, the per-ubatch index plan, four boundary conditions, and two
corrections to what this document previously claimed about V4's indexer (its
rope type and its head order were both stated backwards).

### Slice A — ratio 0 end to end (2026-08-10)

`seam::runner`'s `MixerW::Dsv4` arm emits, per layer:

```text
hc wrap (attn) : rmsnorm(res, ones) -> Linear(hc_attn_fn) -> HyperConnectMix
                 -> HyperConnectPre -> RmsNorm(attn_norm)
attention      : wq_a -> q_a_norm -> wq_b -> QkNorm{weight:None}
                 -> rope the [nope|rope] TAIL -> Copy to f16 q
                 wkv -> attn_kv_a_norm -> rope the tail -> WriteKv (K only)
                 Attention{ k_cache = v_cache, n_kv:1, sinks, SlidingWindow(swa) }
                 -> rope BACKWARD (de-rope) -> per-group Linear(wo_a, w_off)
                 -> Linear(wo_b)
hc post (attn) : HyperConnectPost  res[0] -> res[1]
hc wrap (ffn)  : ... on res[1] -> RmsNorm(ffn_norm)
ffn            : [hash layers only] GatherI32(tok_ids, ffn_gate_tid2eid)
                 MoeFfn{ SqrtSoftplus, swiglu_clamp_exp[il],
                         exp_probs_b XOR expert_ids }
                 + shared expert (swiglu_clamp_shexp[il]), summed
hc post (ffn)  : HyperConnectPost  res[1] -> res[0]
```

then, once, `HyperConnectMix { gates: None }` + `HyperConnectPre` collapse the
streams back into `hidden` for `output_norm`. Four resolved design points:

- **The residual PING-PONGS between two `[batch, hc_mult, n_embd]` buffers.**
  `Op::HyperConnectPost` cannot run in place — every output element reads every
  `src` stream of `residual`. A layer wraps exactly two sublayers, so the parity
  returns to buffer 0 at every layer boundary and the pair is a fixed a→b→a.
- **The widened stream is seeded by REPLICATING the embedding across the
  `hc_mult` streams.** Read at llama.cpp `57291f2`: `deepseek4.cpp`'s graph
  prologue does the same (`ggml_repeat_4d` of the token embedding to
  `[n_embd, hc, n_tokens]`, named `hc_init`).
- **The `[nope | rope]` tail is sliced with ONE `Op::CopyStrided` each way**, at
  `rows = batch * n_head`, `src_stride = head_dim`, `src_off = nope` — the
  packed result is then a plain `n_head`-head rope row. Three sites: q, kv, and
  the backward de-rope of the attention output.
- **`Op::Attention` reads an f16 `q`** (the seam's producer→consumer dtype
  flow), so the normed+roped f32 query is `Op::Copy`d into `q16` exactly as
  llama4's NoPE layer casts its unroped one. The rope itself stays on the f32
  buffer.

Two things outside the emit had to move with it:

- **Vulkan's `Op::Linear` now takes `w_off` on an F32 weight**, by shifting the
  weight's `bufferDeviceAddress` base rather than adding a push field — every
  float GEMV here already addresses its weight by pointer, and `w_off` is
  row-aligned, so the slice IS a shifted pointer. F16 stays refused (four more
  call sites, an alignment obligation, and no caller). Without this the grouped
  output projection could not run on Vulkan at all with an f32 `wo_a`.
- **V4 is excluded from the record-once decode replay tape**, exactly as
  `deepseek2` already is. Its four new ops have no dyn twins, so the adapter's
  `decode_eligible` is false and the tape would have been executed STATICALLY
  with `pos = 0` baked into every `WriteKv` and `Attention` — every token
  writing its KV to row 0 and attending only itself. Measured before the fix:
  the CPU-vs-Vulkan cosine fell 1.0 → 0.95 → 0.67 over prompt lengths 1, 2, 3,
  while a `sliding_window = 1` model (where attending only your own row IS
  correct) stayed exact and hid it.

What ratio 0 does NOT cover: compressed layers. Ratio 128 has generated on the
CPU since plan 09 S6 ("The HCA tier" below), ratio 4 since S7 ("The CSA tier").

### Slice A2 — hash routing end to end (2026-08-13)

A layer below `hash_layer_count` takes its experts from its own
`blk.N.ffn_gate_tid2eid` — an i32 `{n_expert_used, n_vocab}` table indexed by
TOKEN ID — instead of the router's top-k. The op that consumes the selection
(`Op::MoeFfn::expert_ids`) already existed; what was missing was the gather that
produces it, so the layer was refused by name. It now emits:

- **`Op::GatherI32`** — `ggml_get_rows` on an integer table:
  `dst[r, :] = table[ids[r], :]`, values copied, no dequant and no scale. CPU
  interpreter arm plus `gather_i32.comp` on Vulkan; **Metal refuses it by name**
  (its device MoE path already refuses V4's mandatory sqrt-softplus gating, so a
  V4 MoE layer could not run there either way — `docs/backlog.md` §
  B-DSV4-HASH).
- **Not a mode of `Op::EmbedGather`.** That kernel walks the table in whole
  32-element sub-blocks (`nsub = ne / 32`), so a row of `n_expert_used` integers
  (6 on the shipped file) is narrower than one sub-block and gathers nothing —
  since `c5f542a` a hard host-side assert rather than silent zeros. It also
  dequantizes through `native_decode.glsl`, which has no I32 format, and scales
  the result. Three semantics that do not belong on an integer lookup.
- **The table is BOUND as a storage buffer**, not read through the resident-BDA
  arena the way `embed_gather` reads `token_embd`: it is
  `n_expert_used * n_vocab` dwords (3.1 MB on the shipped V4 file), far inside
  `maxStorageBufferRange`, and one addressing mode is cheaper to reason about
  than two.
- **The token-id Input is declared for the gather's sake too.** It used to exist
  only under `gpu_embed` (the on-device embed gather), which gates on a
  `token_embd` dtype the gather kernels cover — a model can fail that and still
  be hash-routed. `build` now declares the ids whenever a hash-routed layer is
  in the span, and `hidden` stays an Input unless the EMBEDDING really is
  gathered on-device. Both binds live in one place (`bind_step_input`) so the
  per-token and record-once paths cannot drift.
- **No host round trip is added.** V4 is excluded from the record-once replay
  tape and from the chained decode, so every V4 step already goes through the
  per-token loop, where the host has just fed `cur[pos]`; the gather's input is
  the same 4-byte id upload that path already does (or, under `gpu_embed`, the
  identical buffer the embed gather reads).
- **The table's shape is validated, not clamped.** `generate_dense_backend`
  refuses a `ffn_gate_tid2eid` that is not `expert_used_count * vocab` entries,
  because the gather indexes it by token id with no bounds clamp — the same
  contract `Op::EmbedGather` reads `token_embd` under.

What the routing weights are is unchanged and was already pinned at the op
level: `build_moe_ffn` takes them from `ggml_get_rows(probs, selected_experts)`,
i.e. the router's own sqrt-softplus probability at each hash-chosen expert, so
the router matmul still runs and only `argsort_top_k` (with `exp_probs_b` and
the group masking that feed only it) is skipped.

Slice 2 added the op-level pieces V4's attention needs, on CPU, Vulkan and
Metal, with op-level parity tests in `crates/infr-llama/tests/seam_op_parity.rs`
and `crates/infr-metal/tests/parity.rs`:

1. **Unweighted per-head RMS norm on Q** — `Op::QkNorm { weight: Option<_> }`.
   `None` is the bare `ggml_rms_norm` V4 calls after `wq_b`; a ones-vector
   weight would be a fake operand costing a real allocation.
2. **Attention sinks** — `Op::Attention { sinks: Option<_> }`, `[n_head]` f32.
   Read off `ggml_compute_forward_soft_max_f32`'s `src2` handling: the sink
   joins the softmax MAX **and** the DENOMINATOR, and never the numerator.
   `None` for every existing arch, so no current model's numerics move.
3. **Rope BACKWARD** — `Op::Rope { backward: bool }`. `ggml_rope_ext_back` is
   the same kernel with `sin_sign = -1`: `cos` untouched, `sin` negated, i.e.
   the TRANSPOSE. That is only the INVERSE when `mscale == 1` — ggml applies
   `attn_factor` to both `cos` and `sin` and `sin_sign` does not undo it, so a
   YaRN'd forward-then-back scales by `mscale²`. V4 sidesteps it exactly:
   `dsv4_rope_attn_factor` returns `1/(1 + 0.1·ln(1/freq_scale))` whenever
   `ext_factor != 0`, cancelling YaRN's correction to `mscale == 1` at every V4
   rope call site. `Op::Rope` carries no magnitude scale at all, so `backward`
   is an exact inverse there.
4. **The grouped low-rank output projection needs NO new op.** `wo_a`'s batch
   axis is the OUTERMOST axis of both operands in `ggml_mul_mat`, so group `g`
   is exactly `Op::Linear` over `wo_a` rows `[g·o_lora_rank, (g+1)·o_lora_rank)`
   — which `Op::Linear::w_off` already selects — over output columns
   `[g·o_group_dim, (g+1)·o_group_dim)`, which `Op::CopyStrided` already slices.
   The Vulkan caveat this listed — `w_off` refused on the f32/f16 fallbacks — is
   half resolved: F32 now rides a shifted device address (see "Slice A"), F16 is
   still refused.

The GPU coverage is deliberately narrow, because each new capability lives in
ONE kernel rather than across the whole tier ladder — see `docs/backlog.md` (§
B-DSV4) for exactly which shapes are refused and what a perf pass would need.

**The LOAD path is complete.** A `deepseek4` GGUF registers, parses into a
`Config` — including the per-layer `compress_ratios` array, the per-layer SwiGLU
clamps and the mandatory sqrt-softplus gating — and loads every tensor, with
each layer's set chosen by its ratio and its hash/bias routing.
`Config::deepseek2` is FALSE for V4 (unlike V3.2, which genuinely is V2 plus an
indexer); every reader of that flag was enumerated and is MLA-specific. `wpush`
declares every tensor `wload` uploads, the compressor's and the indexer's
included (`Dsv4W::comp`, after the hyper-connection triples, in upload order),
so the build closure's old ratio-0-only `assert!` is gone; a ratio-4 layer is
refused by name before the build. Hash-routed layers are no longer refused:
their `ffn_gate_tid2eid` occupies the SAME slot of the upload order that a
bias-routed layer's `exp_probs_b` does, and `wpush` pushes exactly one of the
two.

A genuinely different architecture, not an increment. Sharing with stage 2 is
limited to the MoE block, the FFN, norms, and generic rope/embedding plumbing.

**V4 is not MLA.** There is no `kv_lora_rank`, no `wk_b`/`wv_b`. Instead:

1. **Single-head MQA KV** — `wkv` is `{n_embd, n_embd_head}`, one KV head for
   all query heads. The Q path keeps its LoRA (`wq_a`/`q_a_norm`/`wq_b`) and
   adds an **unweighted per-head RMS-norm on Q** with no analogue in V2/V3.2.
2. **Low-rank grouped output projection** — `wo_a` + `wo_b` over
   `attention.output_group_count` groups.
3. **Attention sinks** — `attn_sinks {n_head}`.
4. **De-roping of the attention output** — the rope slice of the output is
   rotated _backwards_ by the query position before the output projection
   (`ggml_rope_ext_back`). Nothing else in the family does this.
5. **Hyper-connections** — `hc_mult` parallel residual streams with learned
   Sinkhorn-normalised mixing, replacing `x = x + f(x)` everywhere.
6. **Three-tier per-layer attention** keyed on
   `compress_ratios[il] ∈ {0, 4, 128}`.
7. **Compressor blocks** that softmax-pool blocks of tokens into single KV rows.
8. **Hash-routed MoE** on the first `hash_layer_count` layers. ✓ emitted (see
   "Slice A2").
9. **`sqrt(softplus)` gating**, mandatory (`MoeGating::SqrtSoftplus`, already
   there since stage 2).
10. **Per-layer SwiGLU clamping**, with V4 clamping the gate **pre-SiLU** where
    every other arch clamps post-SiLU. ✓ emitted.

No dense-lead layers, no NextN.

#### ✓ LANDED (2026-08-10 at the op level, 2026-08-13 emitted) — items 8 and 10

**Per-layer SwiGLU clamping** is `swiglu_clamp: Option<f32>` on `Op::GatedAct`,
`Op::GatedActFused` (the dense / shared-expert path, `swiglu_clamp_shexp[il]`)
and `Op::MoeFfn` (the routed experts, `swiglu_clamp_exp[il]`). One arithmetic,
read off `llm_graph_context::build_ffn` and `build_moe_ffn`:

```text
up   = clamp(up, -limit, +limit)      // symmetric
gate = clamp(gate, -INFINITY, limit)  // ONE-SIDED, upper bound only
out  = silu(gate) * up                // the gate clamp is BEFORE the activation
```

- The `limit > 1e-6` disabled gate lives in exactly one place,
  `infr_core::graph::swiglu_clamp(limit)`, which turns a raw per-layer array
  entry into the field. Passing a non-clamping layer's `0.0` straight through as
  `Some(0.0)` would clamp that whole FFN to zero — verified by injection, the
  output goes to all ±0.
- **Where the pre/post orders actually differ is narrower than it looks.** For a
  positive limit the two agree everywhere the gate is negative
  (`silu(g) < 0 < limit`, so the upper bound never bites either way), and the
  whole negative lobe is inert. They separate only where `gate > limit`, by
  `|silu(limit) − limit|` times the clamped `up` — the gap therefore grows with
  the LIMIT, not with how far the gate reaches below zero. `seam_op_parity.rs`'s
  `swiglu_clamp_orders_are_distinguishable` asserts both halves of that.
- Only the gated SiLU/GELU forms carry it (llama.cpp clamps in its
  `LLM_FFN_SILU` arm); a clamped `Activation::Sigmoid`, and a clamp alongside
  llama4's `weight_before` (which would clamp the already-weighted values), are
  refused on every backend rather than silently approximated.

**Hash-routed MoE** is `expert_ids: Option<TensorId>` on `Op::MoeFfn` — the
`[rows, n_expert_used]` I32 selection already gathered from `ffn_gate_tid2eid`
by TOKEN ID, i.e. llama.cpp's `selected_experts_in`. What supplying it changes,
read off `build_moe_ffn`, is **only the selection**:

- The router matmul still runs and the gating function still produces `probs`.
  The routing WEIGHTS stay `ggml_get_rows(probs, selected_experts)` — the
  router's own probability at each hash-chosen expert, then `norm_w` and
  `w_scale` exactly as on a top-k layer. **They are not uniform**; the uniform
  `1/n_used` reading is asserted to differ and shown to fail.
- `ggml_argsort_top_k` is skipped, and with it everything that only feeds
  `selection_probs`: `exp_probs_b` and the group masking. llama.cpp nulls
  `exp_probs_b` on a hash layer for the same reason, so the two are refused
  alongside `expert_ids` instead of being computed and discarded. Every backend
  branches around the selection rather than overwriting it.

**The gather is `Op::GatherI32`** — see "Slice A2 — hash routing end to end"
above for why it is a separate op rather than a mode of `Op::EmbedGather`, and
`docs/backlog.md` § B-DSV4-HASH for what it still does not cover (Metal, the
paged MoE path, perf).

### llama.cpp over the synthetic file (2026-11-16, plan 09 S5a)

The ratio-0 tier is now checked against llama.cpp itself, in CI, with no
download: `crates/infr-llama/tests/synthetic_deepseek4_llama_cpp.rs` scores
infr's CPU prefill of a synthetic V4 file against `llama-debug --save-logits`
over the same file at prompts of 7 to 520 tokens. The largest gap is
1.5926361e-4, at 520 tokens (the `ratio0` fixture header records every length),
so the ratio-0 emit had no bug for this oracle to find. llama.cpp's data for
three compressed layouts (mixed, all ratio 4, all ratio 128) is checked in
beside it for slices S6 and S7, with per-tensor sums of the compressors, the
indexer and each layer for bisecting; only the all-ratio-4 case lets the
indexer's top-k drop blocks, and only to 131 tokens, because longer prompts
always held a top-k near-tie that f16 rounding of the keys can flip. Getting
llama.cpp to load the file took four fixture changes and one workaround for a
llama.cpp crash on any V4 file without a ratio-4 layer;
`docs/plan/09-deepseek-v4-compressed.md` section 4 lists them with the numbers,
the indexer-precision note for S7 and the regeneration command. The fixture
builder is `crates/infr-llama/tests/dsv4_fixture/mod.rs`, shared by both test
binaries.

### The HCA tier (2026-11-16, plan 09 S5 and S6)

A ratio-128 layer now runs on the CPU, and the synthetic all-HCA file matches
llama.cpp at every stored length (max |logit diff| 1.85e-6 to 4.49e-5 from 7 to
520 tokens, top-1 agreeing everywhere; plan 09 § S6 has the table).
`synthetic_dsv4_hca_cpu_matches_llama_cpp` holds it at the ratio-0 bound in CI.

**The emit** (`seam/dsv4_emit/`, called by the runner's V4 arm between the raw
cache write and the attention) follows `build_attention_impl`:

- every rope of a compressed layer — query, raw kv row, the backward de-rope and
  each compressed row — runs at `compress_rope_theta` with the YaRN divisors
  (`Dsv4Rope::for_layer`; the divisors ride the `yarn_ff` Input, which V4 has no
  other use for), while ratio-0 layers keep the plain rope;
- `kv = attn_compressor_kv·x` and
  `score = attn_compressor_gate·x + ape[pos % 128]`, the APE row gathered with
  `Op::EmbedGather` by a per-step id input, so the scores carry it before
  anything reads or persists them;
- when the graph's positions complete a block: the source
  `[ring | this graph's rows]` is assembled with two copies per half (the ring
  is an f32 Input, and `Op::GatherRows` must read an op-produced tensor), each
  block's 128 rows are gathered with the plan port's `gather_read_idxs()`,
  pooled by `Op::CompressPool` (window 128), RMS-normed by
  `attn_compressor_norm`, roped at the block's FIRST position
  (`state_write_pos`) and written with ONE `WriteKv` (a contiguous sequence
  commits consecutive rows). Real commits only: a step that completes no block
  emits none of this, which changes no number, because the reference's dummy
  commit writes a row no token can see before a real commit overwrites it;
- the step's rows are then persisted into the ring (`state_persist_*`, in runs
  consecutive on both sides), AFTER the commit has read it;
- the attention gets `comp: Some { cache, len: n_visible_max, ratio: 128 }`, or
  `None` while no block is visible to any row.

The commit lands before the attention, so the token that completes a block
attends it. Per-step i32 inputs (APE ids, gather indices, block positions) are
declared by `Dsv4Tier` once per graph and uploaded at each bind site
(`upload_step_i32`); the record-once decode tape never takes a graph that has
any.

**Buffers** (`seam::dsv4_comp_bufs`, one answer for allocation, declaration,
fork, reset and every estimate): per HCA layer the compressed cache
(`ceil(ctx/128)` rows of `head_dim`, in the raw cache's format, which the CPU
attention requires) and the ring (128 rows of `head_dim`, f32, kv and score);
per CSA layer the CSA and LID caches and rings (8 rows of twice the width). They
live in `SeamKv::cbufs`, are forked with the slot, are priced in
`kv_bytes_estimate_fmt` (so the VRAM/RAM fits and the `--ctx` clamp see them)
and in the KV allocation guard, one entry each.

**No partial rewind.** `Config::has_unrewindable_state` is true for a V4 file
with a compressed layer, as for qwen35 and LFM3, and every caller that would
truncate a session reads it: the runner's prefix reuse (a turn reuses the
session only when its prompt extends exactly what the session holds, and
otherwise zeroes the rings and re-prefills from position 0), `SeamKv::seed_from`
(a no-op) and the SWA-ring fallback. The invariant this keeps: every ring row
and compressed row a graph reads was written since the last reset by the one
contiguous sequence the session holds. An extending turn does read rows earlier
turns wrote — a block that straddles two turns pools ring rows the first one
persisted — but every other turn resets first. Within the sequence a compressed
row is read only once its block is committed (the segment's `len` is the graph's
widest visible count, and the attention masks each row to its own
`(pos+1)/ratio`), a block pools only positions from its first to the one that
completes it, and a position below 0 is never read from a buffer (HCA reads
none; the CSA and LID compressors read block 0's previous window as the
sentinel, which the gather writes from its fill: value 0, score `-inf`). That is
why the reset zeroes the rings but not the compressed caches. Speculative
verify, layer-major prefill and the record-once tape never see V4 (MoE, or
excluded by name); batched prefill does since "Batched prefill" below, on the
CPU.

What the tests found when the code was broken on purpose (plan 09 § S6 has the
full table): ropes at position 0 are the identity, so a wrong rope on compressed
rows is invisible until block 1 (the oracle catches it from 300 tokens, the
compressor's from-definition test at once), and persisting before compressing is
invisible when every graph holds one token — only the compressor test that feeds
50- and 150-token graphs catches it.

### The CSA tier (2026-11-16, plan 09 S7)

A ratio-4 layer now runs on the CPU. On synthetic files llama.cpp runs, infr's
CPU logits match llama.cpp's at every stored length: the every-tier `mixed` file
from 7 to 520 tokens (at most 9.35e-5) and the all-CSA `csa` file from 7 to 131
tokens at `top_k` 5 (at most 7.95e-5), where the indexer's top-k drops blocks
from 24 tokens on. `synthetic_dsv4_csa_cpu_matches_llama_cpp` and
`synthetic_dsv4_mixed_cpu_matches_llama_cpp` hold both at the ratio-0 bound in
CI; plan 09 § S7 has the tables.

**The emit** (`seam/dsv4_emit/`) shares the HCA tier's compressor
(`Dsv4Tier::compress`) and runs it twice per CSA layer, with the overlapping
window:

- the CSA compressor at `head_dim` with `attn_compressor_*`, and the LID
  compressor at `indexer_head_size` with `indexer_compressor_*`, each into its
  own cache and ring, from one plan (the reference builds the two plans with
  identical parameters, and the caches and rings have equal row counts);
- a state row is twice the compressor's width: kv and score rows `kv·x` /
  `gate·x + ape[pos % 4]`, both `2·width` wide, the APE added before anything
  reads or persists them;
- each block pools a window of 8 slots: its previous 4 positions' FIRST halves
  (channels `[0, width)`), then its own 4 positions' SECOND halves (channels
  `[width, 2·width)`), gathered by two `Op::GatherRows` into one window; a
  position before the sequence start reads the sentinel (value 0, score `-inf`,
  the gather's fill for the `-1` that `gather_read_idxs()` writes), so block 0
  pools its own positions alone;
- `Op::CompressPool` at window 8, the norm, the rope of the row's last
  `rope_dim` dims at the block's first position (the reference's `n_rot()` is
  the model's rope width for every head, the LID row's included: 64 of 128 on
  V4-Flash, 16 of 128 on the oracle files), one `WriteKv` per cache;
- the ring of 8 rows persisted after the commit has read it.

Then, once some row of the graph sees a block (`n_visible_max > 0`):

- the indexer query `indexer.attn_q_b · qr` (`qr` the normed low-rank query), no
  norm, each head's tail roped at the token's position with the compressed-layer
  rope; the per-head weights `indexer.proj · x`, scaled by
  `1/sqrt(indexer_head_size · indexer_n_head)` inside `Op::LightningIndexer`;
- `Op::LightningIndexer` over the LID cache with `block_ratio` 4,
  `kv_len = n_visible_max` and `top_k = min(n_visible_max, index_topk)`, then
  `Op::TopkMask` into the `[rows, len]` bias of the segment
  `comp: Some { cache: CSA cache, len, ratio: 4, bias }`.

While no block is visible the indexer, the mask and the segment are skipped (an
all-masked compressed half adds nothing, and both ops refuse zero keys).

**The Hadamard rotation is not ported.** llama.cpp rotates the LID row before
storing it as f16, and the indexer query, by one orthonormal Walsh-Hadamard
matrix (`llama_kv_cache` forces it on the LID cache, whose head width is the
indexer's). In exact arithmetic it changes no score; it changes where the f16
rounding of the stored row lands. Measured on the `csa` file over 131 tokens:
with the rotation skipped, 2 of the 432 rows where the top-k drops blocks keep a
different block than llama.cpp (both in the last layer, at positions 111 and
122, with infr's kept-versus-dropped margins 1.2e-4 and 1.8e-4); a temporary
rotation in infr made all 432 agree. Neither flip reaches the stored logits (a
last-layer selection at an earlier position feeds nothing the last position
reads). Porting the rotation is in `docs/backlog.md` § B-DSV4-ORACLE. Every
row's selection is compared with llama.cpp's in CI
(`synthetic_dsv4_indexer_keeps_what_llama_cpp_keeps`, reading infr's through
`infr_cpu::indexer_tap`), those two rows listed as near-ties with their margins;
the regenerator refuses data with any other row whose kept and dropped scores
tie or come within 1e-3 relative.

**Exact-zero indexer scores are an implementation-defined tie.** A block scores
exactly 0 when every indexer head's dot with it is negative (the ReLU), which
with the canonical fixture's 4 indexer heads happens to one block in sixteen;
two such blocks competing for the last kept slot are a tie that llama.cpp breaks
by `std::partial_sort`'s order and infr toward the lower index. On the first
`csa` file this moved 9 selections in 131 tokens and, through them, the logits
by 3.4e-3. No weight noise can expose it (a negative dot stays negative), so the
regenerator's robustness check could not see it. The `csa` oracle file now has
16 indexer heads (the real file has 64), which left no exact zero among the
scores that decide a selection over the whole run.

**What breaking it showed** (plan 09 § S7 has the full table): every break of
the emit turned a test red except three that are equivalent to the correct code
by construction (the window's slot order, which the per-channel softmax ignores;
a nonzero sentinel value, which a `-inf` score weighs 0; the indexer weight
scale, which keeps the order). Two are visible only in graphs of several tokens,
which the runner never builds for V4 (`block_ratio` 1 in the indexer, and
persisting before compressing); the stand-alone tests in
`seam/dsv4_emit/tests/compressor.rs` hold them.

### The CSA tier's CPU cost (2026-11-16)

The CPU's compressed layers now cost what they use. `Op::Attention` attends only
the compressed keys whose top-k bias is finite — a key at `-inf` is skipped
before its dot, as llama.cpp's
`ggml_compute_forward_flash_attn_ext_f16_one_chunk` skips it — and converts to
f32 only the cache rows some query keeps (at most `top_k` per row at f16 or f32;
every visible row on an HCA layer, which has no bias). `Op::LightningIndexer`
splits one row's keys across the pool, dots four query heads against four keys
at a time (`dot_mn`), and keeps the top-k by a partial selection.

**Output bits do not change.** A skipped key's exponential is exactly 0, so the
max, the denominator (summed in key order) and the weighted sum are what they
were; each kept key's dot is the same `dot` (`dot_n` and `dot_mn` equal it bit
for bit); each indexer score sums the heads in the same order; and the selection
keeps the same keys in the same order, ties to the lower index, a row with a NaN
score handed to the old scan. The synthetic V4 file's logits after 64, 256 and
512 tokens have the same digest before and after. Two inputs can differ: a NaN
or infinity in a masked compressed row no longer reaches the output (it is never
read, as in llama.cpp), and a weighted sum that was `-0.0` can now be `+0.0`.

Measured with `crates/infr-llama/tests/dsv4_cpu_bench.rs` (one decode row of one
layer at V4-Flash's geometry — 64 heads of 512, raw window 128, indexer 64 heads
× 128, `top_k` 512 — over random f16 caches; median ms; 32 threads on a Ryzen 9
9950X3D, total CPU load 0-3% before and after, no other build or bench running).
Synthetic and op-level; the real file's per-token cost is disk-bound (§
B-DSV4-REAL) and was not broken down by op. Two interleaved rounds, the commit
before (`0be008dd`) then this change:

| context | CSA blocks | indexer before | indexer after | CSA attention before | CSA attention after | CSA layer before | CSA layer after |
| ------- | ---------- | -------------- | ------------- | -------------------- | ------------------- | ---------------- | --------------- |
| 2k      | 512        | 1.24 / 1.15    | 0.063 / 0.061 | 0.73 / 0.62          | 0.69 / 0.72         | 1.88 / 1.78      | 0.75 / 0.78     |
| 8k      | 2 048      | 4.82 / 4.86    | 0.079 / 0.080 | 2.53 / 2.28          | 0.69 / 0.68         | 7.34 / 7.12      | 0.77 / 0.76     |
| 32k     | 8 192      | 19.36 / 19.33  | 0.201 / 0.196 | 7.79 / 9.44          | 0.63 / 0.71         | 27.07 / 28.75    | 0.83 / 0.91     |
| 64k     | 16 384     | 38.78 / 38.61  | 0.389 / 0.355 | 18.85 / 18.82        | 0.77 / 0.71         | 57.51 / 57.21    | 1.15 / 1.05     |

The CSA layer is the indexer, its top-k mask and the attention; the mask is a
few microseconds at every size. The HCA layer's attention (16 to 512 compressed
rows, no bias, all kept) did not change: 0.23-0.82 ms before, 0.23-0.70 after.
At 64k context the compressed layers went from about 21 × 57 + 20 × 0.7 ≈ 1.2 s
per token to about 21 × 1.1 + 20 × 0.7 ≈ 37 ms. What dominates now is the
attention's 64 heads each reading the same kept rows (`docs/backlog.md` §
B-DSV4-CPU-PERF).

The synthetic file end to end (ratios `[0, 4, 128, 4, 0]`, `top_k` 5, prefill
token by token) is unchanged within noise: 512 tokens in 1724 / 1699 ms before
and 1683 / 1702 ms after. Its indexer took 16.4 ms of the run before and 0.8 ms
after (`INFR_PROF_OPS=1`); a first cut split its tiny keys over the pool and
took 223 ms, which is why a scoring task now has a minimum amount of work.

### Batched prefill (2026-11-16)

The CPU prefills a V4 prompt in chunks: every position but the last in graphs of
`INFR_UBATCH` rows (`crate::seam::ubatch_rows`, the height every model's batched
prefill uses, the default of the backend the session opened,
`PlacementPins::for_caps`), the last in the decode loop, as on every other
model. Two caps apply: no model's chunk is taller than the height its session's
sliding-window rings were sized for (below), and on a file with a CSA layer the
chunk shrinks as the context grows (`dsv4_emit::prefill_chunk_rows`, below).
Chunk boundaries need not align to blocks of either tier. `INFR_UBATCH=1` keeps
one position per graph, which is what the comparison tests use as the reference.
Decode stays one token per graph. The gate is `dsv4_emit::batched_prefill_ok`:
the CPU only, since a GPU backend refuses the compressed layers and a ratio-0 V4
graph of several rows has not run on one, and the CPU's `Op::MoeFfn` routes any
expert dtype at any row count (the other backends' batched expert path needs
`MOE_MMQ_DTYPES` banks, which the synthetic file's f32 banks are not). Since
2026-11-17 a prompt of several chunks prefills LAYER-MAJOR when the experts
stream from disk through the host arena (§ "Layer-major prefill on the CPU"
below); everywhere else the order is chunk-major as described here.

**Why a chunk's rows see exactly what one position per graph shows them.** The
emits were already written for any row count; the runner only fed one row.

- Compressed blocks: every compressor of the graph commits every block the chunk
  completes before the layer's attention runs, and the attention's segment shows
  row `t` only key `j < (pos + t + 1) / ratio` (`comp_visible`). A block
  completed by row `t` has index `(pos + t + 1) / ratio - 1`, so row `t` and
  every later row see it; an earlier row `u < t` sees at most
  `(pos + u + 1) / ratio` blocks, which excludes it. A block completed by a
  later row can therefore never be read, even though it is in the cache when the
  attention runs.
- Rings: the compressor's source is `[ring | this chunk's rows]`, so a block
  pools ring rows earlier graphs persisted and this chunk's own rows in one
  gather; the ring is written after the commit has read it, and when several of
  the chunk's rows land on one ring row the latest wins
  (`build_dsv4_comp_plan`'s persist dedup), which is what one position per graph
  leaves there.
- Indexer: `Op::LightningIndexer` scores each row over its own visible blocks
  only (`(pos + t + 1) / ratio`, the `block_ratio` S7 added) and keeps
  `top_k = min(n_visible_max, index_topk)` per row, filling a row that sees
  fewer blocks than that with indices past its visible count, which
  `Op::TopkMask` marks kept and the segment's own rule masks. A row that sees at
  most `top_k` blocks therefore keeps all of them, as one position per graph
  does with its own `min(visible, index_topk)`; a row that sees more keeps its
  own best `top_k`, the same count. The reference pads the width for the whole
  ubatch the same way (`build_lid_top_k`, `build_top_k_mask`). A row that sees
  no block yet in a chunk whose later rows do is scored over nothing and attends
  its raw window and the sink.
- Hash routing reads each row's own token id (`Op::GatherI32` over the chunk's
  ids). When the embedding is gathered on the host (a `token_embd` dtype with no
  gather kernel, such as the fixture's f32), the chunk uploads the ids into a
  buffer of their own (`PfChunk::ids`), since its main input then holds the
  embedded rows.
- The hyper-connection ops mix each row's streams alone; the model head runs on
  the last position only, in the decode loop.

**Verified** (`seam/dsv4_emit/tests/prefill.rs`), bit for bit, not to a
tolerance: on the four oracle layouts (`ratio0`, `hca`, `csa`, `mixed`, at four
streams and the oracle's indexer) and the canonical two-stream file, prompts of
70 and 300 tokens (and 520 with an HCA tier, 200 on `csa`, past where the top-k
drops blocks) prefilled at chunk heights 3, 37, 150 and the whole prompt give
the same last-position logits, the same kept blocks at every (position, layer)
row (`infr_cpu::indexer_tap`, which now numbers the ops so rows of one graph can
be told apart by layer), the same bytes in every compressed cache and ring, and
the same greedy decode after, with its logits and buffers, as one position per
graph; also with the embedding gathered in the graph (an f16 table) at
height 37. Every run also checks how it ran, through `infr_cpu::op_tap` (each
`Op::Attention`'s row count and each `Op::EmbedGather`'s table dtype): every
layer's attention at each expected graph height, and the f16 table gathered in
the graph exactly in the gathered case, so a run that fell back to one position
per graph, on any layout, cannot pass. Bit-exact because the CPU computes a row
the same way whatever the row count for every op this f32 file runs. One
position per graph through the batched prefill equals the decode loop fed one
turn per token. Session tests: an extending turn prefills only its suffix in
chunks, completing an HCA block from ring rows the turn before persisted; a turn
sharing nothing resets the rings; a prefill cancelled between chunks is not
reused, and the state the cancel leaves is asserted for each of the two
deliberately redundant guards (the session records no prompt while a turn runs;
`written_end` is raised before each chunk executes). The CPU's execute has no
abort check, so a cancel always lands between chunks; an execute that failed
part way would leave the same state.

**Quantized expert banks: chunked prefill is close, not equal.** `Op::MoeFfn` on
the CPU feeds a Q4_K, Q5_K, Q6_K or Q8_0 expert bank int8-quantized activations
once the op has two rows or more (`int8_ok = rows >= 2 && !reference`, then
`expert_acts_kind`), and f32 activations at one row; Q2_K, Q3_K and every other
bank dtype stay on f32 activations at any row count. So on a file with such
banks a chunked prefill and one position per graph differ by the int8 rounding
of the expert activations, which reaches everything downstream: the router of
the next layer, the indexer's scores (a selection at a near-tie can flip between
`INFR_UBATCH=1` and the default) and the logits. Every MoE model's CPU batched
prefill makes the same trade. `Op::Linear` is not part of it: it quantizes the
activations of a quantized weight at one row too, and its batched kernels equal
its one-row ones bit for bit on every weight dtype (below).
`quantized_banks_chunked_prefill_is_close_and_reference_is_exact` gives the
canonical file Q8_0 expert banks (`infr_testkit`'s new `Fill::Quant`) and pins
it at 300 tokens: chunked against one position per graph, max |logit diff|
1.71e-4 at the prompt's last position and 2.37e-4 at the first decoded one (held
at 1e-3), 87 of 594 indexer selections different (the canonical indexer's four
heads leave many near-ties), every chunk height bit-identical to every other
(each row is quantized alone); in reference mode (`kernels.cpu.reference`)
chunked and one-position prefill are bit-identical again, and its one-position
run equals the production one.

**The shipped file's dtypes** (`ggml-org/DeepSeek-V4-Flash-0731-GGUF` Q2_K_S,
read from the tensor-info header of its second shard, which holds every tensor;
the first holds metadata only): the routed expert banks are all Q2_K (43 each of
`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`), so they keep f32 activations
at every row count and the int8 trade above does not reach them; the shared
experts are Q8_0 (129), the routers BF16 (43; the three hash tables I32, the 40
`exp_probs_b` F32), attention 215 Q8_0 and 172 F32, the compressors 123 Q8_0
(the APE tables among them) and 41 F32, the indexers 105 Q8_0 and 21 F32, the
hyper-connections 86 Q8_0 and 172 F32, the norms F32, `token_embd` Q8_0,
`output` Q6_K, `output_hc_fn` Q8_0 and the other head tensors F32. Every Q8_0
tensor is an `Op::Linear` weight except `token_embd` and the APE tables, which
are gathered a row at a time and so cannot depend on the row count. Whether
chunked and one-position prefill are bit-identical on that file therefore rests
on Q8_0 `Op::Linear` (and the BF16 router GEMM inside `Op::MoeFfn`, which stays
on f32 activations).

**`Op::Linear` at several rows equals it at one, on every dtype (2026-11-17).**
`infr-cpu`'s `tests/linear_batch_identity.rs` runs every weight dtype (F32, F16,
BF16 and every GGUF quant infr loads) through one `Op::Linear` of R rows and R
of one row each, at R from 1 to 129 across the batched kernels' tile edges, at
widths of one block, of an odd block count off the 256 boundary and (for the
32-element formats) on it, at 1, 3, 15 and 16 outputs, with zero, one-spike,
all-negative, subnormal, wide-range and exact int8 rounding-tie rows, in the
production kernels and in reference mode: every value bit-identical, and every
one inside a bound derived from the int8 quantizer against an f64 reference.
`kernels::batch_tier_tests` checks each x86_64 tier pairing (scalar, AVX2,
AVX-512BW, VNNI) directly, so one machine covers what the others dispatch. Q5_0
was not identical: it took an f32-activation dot at one row and the int8 kernel
at several (2.6e-3 of the output's scale on ordinary rows); its one row now runs
the same int8 kernel, 2.2 to 8.8 times faster on the measured decode shapes and
slightly further from llama.cpp's logits (`docs/backlog.md` B201). On the
synthetic file stored in the shipped dtypes (every matmul above Q8_0, the head
Q6_K, the routers BF16, the banks Q2_K, widened so each row is whole blocks),
`shipped_dtypes_chunked_prefill_equals_one_position_per_graph` finds chunked
prefill at heights 3, 37 and the whole prompt bit-identical to one position per
graph over 200 tokens: logits, kept blocks, caches and rings, and the decode
after. Breaking a Q8_0 batch kernel's summation order, dropping its last weight
row, or rounding its activation ties to even turns that test red, as it does the
op-level one; the first two also turn `infr-cpu`'s older Q8_0 tests red, the tie
rounding only the new one (its tie rows), and an AVX2-only reorder only
`batch_tier_tests`.

**Memory per chunk.** A CSA layer's indexer scores every row of the chunk
against every block its last row sees, and its top-k mask is a bias of the same
shape: two `rows × (start + rows) / 4` f32 buffers per CSA layer per chunk. At a
fixed 1024-row chunk that is 64 MiB each at 64k context, 160 MiB at 160k and 1
GiB at 1M. `dsv4_emit::prefill_chunk_rows` shrinks the chunk until each fits
`DSV4_PREFILL_SCORE_BYTES` (64 MiB): on any V4 file with a CSA layer (V4-Flash
included; the bound depends only on the ratio) a chunk starting at 8k keeps 1024
rows, at 64k 1008, at 160k 408 and at 1M 63
(`prefill_chunk_rows_bound_the_csa_buffers` pins them, worked by hand). The CPU
executor now recycles both buffers from one CSA layer to the next within a chunk
(the indexer builds its scores in a buffer the executor's `Recycler` hands it,
and `Op::TopkMask` takes its output from the same pool) instead of mapping fresh
ones per layer. The attention's converted kept rows are at most
`min(blocks, rows × top_k)` rows of the head width in f32, twice the f16
compressed cache it reads at most (32 MiB at 64k on V4-Flash's 512-wide head, 80
MiB at 160k), not bounded further.

**The sliding-window ring caps every model's chunk.** A window layer's ring
holds `window + chunk` rows for the chunk height the session was made with
(`kv_rows_at`), but the prefill loop read `ubatch_rows` at call time, which
answers for the placement scope current then; a session made under a pinned
short chunk and extended outside that scope prefilled in taller chunks than its
ring holds, overwriting rows its own earlier rows still attend — on every model
with a window layer, before V4 too. The prefill chunk is now capped at the
session's own height when it has a window ring;
`prefill_chunk_never_outgrows_the_window_ring` (V4's window layers, a session
made at an 8-row pin and extended at the default height) is red without the cap.
The llama.cpp comparisons run on batched prefill and give the same numbers as
the one-position runs, to the last digit printed.

What breaking it on purpose showed (each alone, then reverted): a row seeing the
blocks of the chunk's last row, or only those of its first, turned the chunked
comparison, the extending-turn test, `split_prefill_equals_one_prefill`, the
selection test and the `csa`, `hca` and `mixed` oracle cases red; ring rows
persisted from the first row landing on them instead of the last turned the
plan's literal tests, both compressor definition tests, the split session test,
the chunked comparisons and the `mixed` oracle red; persisting before
compressing turned the compressor definition tests, the split session test and
the chunked comparisons red, but no oracle case (a prompt in one chunk never
reads a ring row it persisted); every row scoring the graph's widest visible
count turned the indexer definition test, the chunked comparisons, the split
session test, the selection test and the `csa` oracle red; hash routing by row
0's token and the hyper-connection pre-mix by row 0's weights turned every
chunked comparison and every oracle case red; binding the host-embedded rows as
the token ids failed every hash-routed run.

**Measured** (32 threads, Ryzen 9 9950X3D, CPU load about 1% before,
`crates/infr-llama/tests/dsv4_cpu_bench.rs`, two interleaved rounds, the same
binary with `INFR_UBATCH=1` and without): the synthetic canonical file
(`dsv4_cpu_fixture_bench`) prefills 64 / 256 / 512 tokens in 198.9 / 809.4 /
1589.1 ms and 192.5 / 824.4 / 1620.2 ms one position per graph, and in 19.0 /
22.7 / 27.9 ms and 17.6 / 21.8 / 27.5 ms batched (322 → 18 337 tokens/s at 512),
with the same logits digests. That gain is almost all per-graph fixed cost (a
build, a compile and an execute per position on a file whose weights are a few
megabytes); it predicts nothing about the real file (those prompts fit in one
chunk at the default height, so 512 or 1024 makes no difference to them). What
does is one V4-Flash routed-expert layer (`dsv4_cpu_moe_prefill_bench`, run with
`INFR_DSV4_MOE_BENCH=1`: `n_embd` 4096, 256 experts, 6 used, 2048 wide, Q2_K
banks held in RAM; two rounds): one row costs 5.73 / 5.70 ms, a chunk of 64 rows
2.81 / 2.82 ms per row, of 256 1.21 / 1.21, of 512 0.81 / 0.81 and of 1024, the
default height, 0.63 / 0.62, so 9.1 / 9.2 times less per position at the default
chunk. The one-row baseline repeats the same row, so its experts stay warm in
L3, which a real prompt's distinct tokens do not: the ratio is conservative. On
the real file the banks do not fit in RAM and stream from disk one routed expert
at a time, where a chunk reads each routed expert once for all the rows that
route to it instead of once per token; that saving was not measured on a long
prompt (a 5-token one prefills in 16.2 s with a 12 GB arena, § B-DSV4-REAL).

### Layer-major prefill on the CPU (2026-11-17)

**Why.** On V4-Flash a chunk of more than about 43 rows routes to most of every
layer's 256 experts, so a chunk-major prompt reads nearly all 90.9 GB of expert
banks once PER CHUNK through a 12 GB arena that keeps about 5 GB of them. The
fix is the order every streamed model already had on the GPU (plan 08): run
every chunk of the prompt through layer L before any chunk reaches L+1, so each
layer's experts are read once for the whole prompt.

**What it is.** The runner's existing layer-major walk (`spans` × `chunks` in
`generate_dense_backend`, one single-layer span graph per (layer, chunk)), now
also taken by V4 on the CPU:

- **The wide residual crosses spans.** A partial-span V4 build declares the
  widened `[rows, hc_mult * n_embd]` stream (`hcr[0]`, where it stands at every
  layer boundary) as a caller-owned Input, `DecodeHandles::hc_resid`; each chunk
  owns one (`PfChunk::wide`). The span that starts the stack widens `hidden`
  into it, every span advances it in place (the CPU writes a mutated f32 Input
  back), and no partial span runs the hyper-connection head. A whole-model build
  is unchanged: the stream stays graph scratch.
- **The compressed state needs nothing new.** Every piece of it — the raw window
  ring, both compressed caches, the rings, the indexer cache — belongs to one
  layer, and within a layer the chunks still run in ascending order, so each
  layer's state receives the same writes in the same order as chunk-major, and
  every op the same rows. A row's block visibility (`comp_visible`), the
  indexer's per-row top-k and the ring persists are per graph and unchanged.
- **When.** `seam::layer_major_prefill`'s automatic rule (on when the weights
  stream) now counts the CPU's host arena for V4 (`seam::host_layer_major`):
  when the run's own weights stream through an arena that cannot hold them and
  has slots (`Backend::host_streams`, asked about the model's bound buffers, not
  the backend), for a sole request (not one sharing the backend with others, as
  `infr serve --parallel` makes; nothing concurrent was measured), and when the
  prompt's FIRST chunk is at least `seam::HOST_LAYER_MAJOR_MIN_ROWS` rows (the
  measurements below are why). That first chunk's height comes from
  `dsv4_emit::prefill_chunk_rows` at the prefill's start position, so a prompt
  prefilled from position 0 at the default height qualifies at any length, and
  an extending turn starting deep in the context may not
  (`layer_major_gate_follows_the_first_chunk` pins the boundary).
  `layer_major_this_prompt` still runs it only for more than one chunk. Other
  families keep chunk-major on the host arena (none was checked bit for bit
  there); `paging.layer_major` forces either order on any family, as before
  (`INFR_LAYER_MAJOR=0` / `1`).
- **Bounded memory.** Layer-major holds every chunk's residual between layers:
  `seam::layer_major_row_bytes` per row, the four streams and the embedded input
  row in f32 plus id and position (81 928 bytes at V4-Flash's geometry). A
  prompt longer than `LAYER_MAJOR_CARRY_BYTES` (1 GiB) of that, 13 105 rows on
  V4-Flash, runs as consecutive SEGMENTS of whole chunks
  (`seam::layer_major_segments`), each layer-major, so the experts are read once
  per segment. `paging.layer_major_rows` (config only) sets the cap; other
  families carry the whole prompt as before. An automatic V4 arena too small for
  the weights gives the carry up (`seam::layer_major_carry_bytes`); a model that
  fits, or an explicit `paging.dram`, reserves nothing for it.
- **Residency hints.** A span graph carries `Graph::weight_reuse`: `Again` while
  more chunks of its layer follow, `Done` on the layer's last. On the CPU arena
  `Again`/`Done` insert what a graph pins at the hot end
  (`paged::expert_insert`, `paged::whole_insert`) — a cold-end insert, right for
  an ordinary chunk's sweep, would make a chunk's misses evict the experts the
  chunk before brought in — and after a `Done` graph every paged weight it read
  is demoted to the cold end (`HostPager::demote`, `Pager::demote`), so the next
  layer's misses take a finished layer before anything older — except, per bank,
  the experts the layer's last row routed to and then its most-routed ones, up
  to a share the pool has room for (`PagedWeights::done_keep_share`: its slots
  beyond one op's pins over its banks, at most `paged::DONE_KEEP_EXPERTS`; 8 on
  V4-Flash at 12 GB), which stay at the hot end for the decode that follows (its
  first token continues from that row). On the CPU two live generations never
  share a store (`SeamModel::with_cpu` opens a resident per concurrent
  generation), so the demotions cannot reach another session's working set
  today; a pinned block is never evicted whatever its place.

**Verified** (`seam/dsv4_emit/tests/prefill.rs`, bit for bit): on the five
layouts, at every prompt length of the chunked tests, layer-major in one segment
at chunk heights 3, 37 and the whole prompt (one chunk, still run as one span
per layer), in segments of two chunks (3 rows in 7, 37 in 80) and in segments
shorter than a chunk (150 in 40), with the embedding on the host and gathered in
the graph, equals one position per graph — logits, every row's kept blocks,
every compressed cache and ring, and the decode after — and every run's op tap
shows the layer-major order it names. An extending turn prefills its suffix
layer-major from the turn before's state; a non-extending turn resets the rings
(red without the reset); a cancel landing between layers 1 and 2 leaves layer
1's state advanced and layers 2 and 3's not (asserted), records no prompt,
raises `written_end` past the suffix and is not reused. With Q8_0 expert banks
layer-major equals chunk-major at the same height bit for bit (int8 activations
included), and in reference mode equals one position per graph. The llama.cpp
oracle cases now also prefill layer-major in chunks of 64, asserted bit-equal to
the chunk-major logits they score. Breaks that turned these red: widening the
stream in every span (every layer-major test and the paging test), sweep
(`Sweep`) hints in the layer-major walk (the paging test: 619 reads against a
bound of 504), the V4 host gate off (the paging test), both cancel guards off
(the cancel test).

**Reads, synthetic** (counters, asserted): `infr_cpu`'s
`layer_major_chunks_read_each_routed_expert_once` (six layers, top-4 of 32,
eight 8-row chunks, a pool of one layer's experts) reads 546 blocks layer-major,
exactly the blocks routed to, against 889 with sweep hints and 2177 chunk-major;
`a_finished_layer_is_evicted_before_older_residents` reads 354 instead of 546 on
a second prefill with the demotions. Through the runner
(`dsv4_emit/tests/paging.rs`: the canonical layout with 32 experts of 32 KiB a
bank, 480 expert blocks, a 600-token prompt in three chunks of 256): 456 reads
with an arena holding every block, 502 layer-major with 120 slots (the prompt's
reads plus at most the decode step's 60 misses), 1293 chunk-major with the same
120 slots, and in chunks of 37 the automatic rule keeps chunk-major (5622
reads); the logits are bit-identical.
`the_last_rows_experts_survive_a_layer_major_prefill` (`infr_cpu`) replays the
prompt's last row alone after a layer-major prefill and reads nothing (red with
nothing kept).

**Reads and time, real file** (V4-Flash Q2_K_S, `--dev cpu`,
`INFR_DRAM_CACHE=12g` — 1904 expert slots of 2.75 MB, 5.24 GB, of 33 024 expert
blocks — `INFR_PAGER_STATS`, `INFR_PAGER_TRACE`; 2026-11-17, the box shared with
another session's GPU tests). Per-block read counts come from the pager trace,
whose expert pool records every pin; a layer-major prefill ends at the pool's
last demotion, so its reads are separable from the decode's. Before the first
run the expected volume was at most about 87 GB (130 rows of 6-of-256 routing,
if routing were uniform: about 244 experts a layer), about 83 s at the 1.05 GB/s
this drive gave in 2026-11-16's bench.

| run | prompt, chunks         | order       | expert reads in the prefill                 | prefill  |
| --- | ---------------------- | ----------- | ------------------------------------------- | -------- |
| 1   | 131 tok, 80 + 50       | layer-major | 16 881 (46.5 GB), 1 each                    | 104.3 s  |
| 2   | 69 tok, 32 + 32 + 4    | layer-major | 13 122 (36.1 GB), 1 each                    | 98.96 s  |
| 3   | 69 tok, 32 + 32 + 4    | chunk-major | 20 342 (56.0 GB) with the last token's pass | 91.59 s  |
| 4   | 69 tok, as 3, after it | layer-major | 13 122 (36.1 GB), 1 each                    | 114.89 s |

Run 1's prefill time is the sum of its span executes: the watchdog stopped it in
the decode (available memory reached the 12 GB floor), so its counters were not
printed; the trace up to then holds the whole prefill. Routing is far from
uniform: 131 tokens touch about 131 of 256 experts a layer. Before run 3, the
per-(layer, chunk) expert sets in run 2's trace, replayed chunk-major through a
model of the pool, predicted 19 983 prefill reads (24 741 for run 1's prompt:
68.1 GB, unrun); run 3 read 20 342 including the last prompt position's pass
(774 pins), so the model holds: 1.55 times run 2's reads.

**But it was not faster on these short prompts.** Chunk-major took 91.6 s
against 99.0 and 114.9 s layer-major. Per read (2.75 MB, four lanes),
chunk-major spent 13.7 ms of read time and layer-major 22.2 ms (run 4: 307.5 s
for 13 851 reads; run 3: 278.5 s for 20 342). The likely reason, not
established: reads are buffered, and chunk-major's second and third chunks
re-read blocks it read seconds before, which the OS file cache (about 27 GB of
standby between runs) still held, while every layer-major read is a block's
first. Its non-wait time was also higher, 16.0 s against 8.8 s (129 span graphs
against 3), not profiled. At the default chunk height (1024) the order only runs
for a prompt of more than one chunk, which runs 5 to 7 below measured. The first
decode-loop pass after runs 2 to 4's layer-major prefill missed 729 of its 774
expert pins (run 4), before the last-row experts were kept (below). Decode after
run 2's prefill: 6 tokens in 11.08 s.

**The decisive case: a prompt of two default-height chunks** (same setup, with
the last-row keep; a fixed 66-token English passage repeated 21 times and a
question, 1443 tokens, so chunks of 1024 and 418; `--max-new 4 --ctx 2048`;
`INFR_LAYER_MAJOR=0` for chunk-major). Predicted before each run: layer-major 20
000 to 32 000 expert reads (55 to 88 GB) and 190 to 280 s, from runs 1 and 2's
routed share and per-read wall time; chunk-major, from run 5's trace replayed
chunk-major, 39 314 prefill reads (108.2 GB) and about 280 to 310 s.

| run | order                | expert reads in the prefill | prefill             | min available |
| --- | -------------------- | --------------------------- | ------------------- | ------------- |
| 5   | layer-major          | 23 292 (64.1 GB), 1 each    | 206.83 s            | 20 295 MB     |
| 6   | chunk-major          | 24 524 when stopped         | over 235 s, stopped | 11 501 MB     |
| 7   | layer-major, after 6 | 23 292 (64.1 GB), 1 each    | 195.16 s            | 20 153 MB     |

Run 6's watchdog stopped it at 11.5 GB available, 235 s in, with its first chunk
done (209.5 s, more than either layer-major run's whole prefill) and 11 of the
second chunk's 43 layers; at the rate of those 11 the rest would have taken
about 75 s more, about 310 s in all, as predicted. So layer-major is at least
12% faster on a measured lower bound, about 1.5 times by the estimate: the order
stays on for chunks this tall. Read time against waiting: run 5 waited 113.2 s
of 214.9 (prefill and decode), run 7 100.5 s of 207.6; the rest, about 100 s, is
compute, mostly the 1443 rows' expert GEMMs (`INFR_PROF_OPS`: `Op::MoeFfn` 81%
of a 1024-row span's time, its wave of pins included). The first decode-loop
pass hit 470 of its 774 expert pins in both (304 misses, against 729 before the
keep). Decode: 4 tokens in 8.09 s (run 5) and 12.45 s (run 7).

**What the slowness of the short prompts was.** Not the span graphs: the
synthetic file prefills 512 tokens in 229.3 ms chunk-major and 230.8 ms
layer-major in 32-row chunks (16 graphs against 80, `dsv4_cpu_fixture_bench`,
release), so a span's build, compile and execute setup cost about nothing, and
nothing was changed there; the 16.0 s against 8.8 s of non-wait time in runs 4
and 3 stays unexplained (the box was shared). The file-cache explanation was
tested only indirectly: the pager now counts re-reads (`HostPagerStats::rereads`
and `reread_ns`, printed by `INFR_PAGER_STATS`), but the only re-reads in runs 5
and 7 are the decode's (1205, at 10.8 and 18.2 ms each against 15.8 and 13.5 ms
for first reads: no signal), and run 6 was stopped before printing. At the
default height chunk-major's second chunk took about 2.3 s a layer for about 300
reads (run 6's trace holds its first 7 layers: 2069 reads of 3339 pins), about 5
ms of wall time per read once a layer's compute (0.8 s at 418 rows) is taken
off, against layer-major's 4.6 ms of waiting per read: a sweep of most of the
experts does not fit the file cache, so its re-reads cost disk reads, which the
32-row chunks' sweeps (about 20 GB each) did not need to be. Inferred from those
timings, not measured per read.

### `compress_ratio` is the master per-layer switch

`hparams.set_swa_pattern(0)` makes **every** layer sliding-window, so long-range
recall comes exclusively from the compressed caches.

| ratio | flavour                 | caches                                                |
| ----: | ----------------------- | ----------------------------------------------------- |
|     0 | pure sliding window     | raw SWA only                                          |
|     4 | CSA + lightning indexer | raw SWA + CSA(4:1) + LID(4:1) + two compressor states |
|   128 | HCA                     | raw SWA + HCA(128:1) + compressor state               |

The two ratio-4 compressor states are **overlapping** (`state_size == 2*ratio`,
so each committed row pools a 2×ratio window); the ratio-128 one is **not**
(`state_size == ratio`). That asymmetry is in the constructor, not in the graph
— see "The compressed-KV state machine" below.

Only `{0, 4, 128}` are accepted. Compressed layers use YaRN at
`compress_rope_theta`; ratio-0 layers use plain unscaled rope. `kq_scale` is
plain `1/sqrt(n_embd_head)` at all three call sites — none of stage 2's mscale²
games.

V4's indexer differs from V3.2's in three structural ways. **There is no
`indexer_attn_k` and no `indexer_k_norm`** — the indexer keys come from the
compressor, so `index_topk` counts _compressed blocks_, not tokens. And its rope
is **NORM** with a **`[nope | rope]`** head, both the opposite of V3.2: see "Two
corrections to what this document said about V4's indexer" below, which reads
them off the source.

### Sinkhorn hyper-connections

The residual stream is widened to `hc_mult` copies. Each sublayer is wrapped
`pre → sublayer → post`, where one matmul produces three chunks — `pre` (stream
collapse weights), `post` (per-stream output gates) and `comb` (an `hc × hc`
mixing matrix). `comb` is made approximately doubly-stochastic by Sinkhorn
iteration, so no stream's mass blows up or vanishes with depth.

```
comb = softmax(comb) + eps          # softmax over dst
norm_cols()                          # then n_iter column normalisations
for i in 1..n_iter: norm_rows(); norm_cols()
```

Then `out[i, dst] = x[i]·post[dst] + Σ_src residual[i, src]·comb[dst, src]`.

**Expect to get this wrong twice.** The index formula is
`logits[dst, src, t] = mixes[2·hc + dst + hc·src, t]·scale + base[...]`; the
loop is asymmetric (`n_iter` column normalisations, `n_iter − 1` row); eps is
added in three distinct places; and llama.cpp's own lambda names
`norm_rows`/`norm_cols` are **inverted** relative to its header's `dst`/`src`
vocabulary. Trust the index formula and the lambda bodies, not the names.

#### ✓ LANDED at the op level (2026-08-10), emitted by slice A

Three ops on CPU + Vulkan + Metal, mirroring llama.cpp's three fused nodes:
`Op::HyperConnectMix` (`ggml_dsv4_hc_comb` plus the `pre`/`post` gate arithmetic
the reference leaves as elementwise views), `Op::HyperConnectPre`
(`ggml_dsv4_hc_pre`) and `Op::HyperConnectPost` (`ggml_dsv4_hc_post`).
`build_hc_head` is NOT a fourth op: its `output_hc_fn` is `{hc_dim, hc}` so its
`mixes` is exactly the `pre` chunk, read at the same `scale[0]` / `base[0..hc]`
indices — it is `Op::HyperConnectMix { gates: None }`. `hc_mult` is accepted in
`1..=HYPER_CONNECT_MAX_MULT` (8) and refused on the host beyond that, because
every backend holds a token's whole `hc × hc` matrix in a fixed-size array.

Resolving the warnings above, verified in `seam_op_parity.rs`'s
`hyper_connect_*` against a from-definition f64 reference (each of the nine
deviations below was injected into the CPU arm and shown to go red):

- **`norm_cols` reduces over `src`; `norm_rows` reduces over `dst`.**
  llama.cpp's `norm_cols` permutes so `src` is `ne[0]` before `ggml_sum_rows`;
  its `norm_rows` sums the matrix as laid out, whose `ne[0]` is `dst`. Both
  names are the opposite of the axis they touch. The loop therefore ends on an
  over-`src` normalisation, which is why `Σ_src comb[dst, ·] = 1` comes out
  exact to eps level while `Σ_dst` is only as converged as the iteration got.
- **The doubly-stochastic property does NOT catch a transposed `comb` index.**
  Sinkhorn of a transposed matrix is Sinkhorn of another matrix: it still ends
  on `norm_src` and its sums are just as well behaved. What the sums DO catch is
  giving the extra normalisation to `dst` (i.e. trusting the lambda names). The
  transposed index is caught by value — it moves the output by ~1.
- **Sinkhorn does not converge on peaked logits.** At `hc = 4, n_iter = 3` with
  logits spread over ±8, `|Σ_dst − 1| ≈ 0.5`. That is the algorithm (its rate
  collapses as the matrix approaches a permutation), not a port bug; asserting
  convergence needs mild logits and tens of iterations.
- **The asymmetric COUNT is nearly inert, for a reason worth knowing.** The
  softmax already leaves every `src` column summing to `1 + hc·eps`, so the
  symmetric variant's extra leading `norm_dst` is a UNIFORM rescale that the
  following `norm_src` undoes. What survives is second order in eps: ~1e-11 at
  `eps = 1e-6`, ~9e-5 at `eps = 1e-2`. The count is pinned, but only the
  large-eps case pins it for an f32 backend.
- **At `hc = 1` almost every Sinkhorn detail is genuinely inert** (a 1×1 matrix
  is its own transpose, `norm_src` and `norm_dst` are the same operation, and
  the iteration's fixed point does not depend on the initial value). It is a
  kernel SHAPE case, not a semantics one.

### The compressed-KV state machine

This is the **largest single porting risk in the family**, and it is now
specified rather than guessed: `llama-kv-cache-dsv4.cpp` (1978 lines) has been
read in full, together with its header, `deepseek4.cpp`'s consumers and
`llama-graph.cpp`'s input setters. Everything below is read off those files;
where an earlier revision of this document disagreed, the file won and the
paragraph was rewritten.

#### The inventory, exactly

Not "seven cache structures" — the constructor
(`llama-kv-cache-dsv4.cpp:1013-1131`) builds **five caches and three compressor
states**, and only four of the caches are reachable from a V4 graph:

| structure   | kind                | rows                            | row width             |
| ----------- | ------------------- | ------------------------------- | --------------------- |
| `kv_raw`    | ISWA base half      | —                               | never read by V4      |
| `kv_raw`    | ISWA **SWA** half   | the sliding window              | `n_embd_head_k`       |
| `kv_csa`    | K-only block cache  | `PAD(ceil(kv_size/4), 256)`     | `n_embd_head_k`       |
| `kv_hca`    | K-only block cache  | `PAD(ceil(kv_size/128), 256)`   | `n_embd_head_k`       |
| `kv_lid`    | K-only block cache  | `PAD(ceil(kv_size/4), 256)`     | `indexer_head_size`   |
| `csa_state` | compressor state ×2 | `state_size = 2*4 = 8`          | `2*n_embd_head_k`     |
| `hca_state` | compressor state ×2 | `state_size = 128` (**not** 2×) | `n_embd_head_k`       |
| `lid_state` | compressor state ×2 | `state_size = 2*4 = 8`          | `2*indexer_head_size` |

"×2" is literal: each compressor state is **two** f32 tensors, `kv` and `score`,
both `[n_embd_state, state_size, n_stream]` and both always f32 regardless of
`type_k`. The raw base half is allocated for generic ISWA bookkeeping and the
header says so outright: "DSV4 raw attention only uses the SWA half of
`kv_raw`."

Two consequences the old prose hid. **`state_size == 2*ratio` is what makes a
compressor overlapping, and only CSA and LID are** — `hca_state` gets
`state_size == ratio`, so HCA pools a single non-overlapping block. And **the
LID plan is built from the same parameters as the CSA plan**: the
`llama_kv_cache_dsv4_context` constructor calls `dsv4_build_comp_plans` once per
tier, and the LID call reads its own state and cache, whose ratio (4), overlap,
state size (8) and row count equal the CSA ones on V4, with the same
`pad_blocks = true` — so the two plans are equal. Only the row widths differ.

#### The per-ubatch plan

`dsv4_build_comp_plan` is the whole state machine. Per token, at absolute
position `pos`:

```text
state_pos[i]  = pos % ratio                 // APE row id, gathers attn_comp_ape
n_visible[i]  = (pos + 1) / ratio           // FLOOR — completed blocks only
plan.n_kv     = max(GGML_PAD(max_i n_visible[i], 256), 256)
```

The compressor state is a **ring of `state_size` rows** indexed
`pos % state_size`, and `state_persist_{src,dst}_idxs` carry one entry per
distinct ring row the ubatch touches, keeping the highest `pos` when several
tokens collide (lines 496-505) and sorted by destination (line 576) so the write
order is deterministic.

A compressed row is committed **only on a block boundary**,
`(pos + 1) % ratio == 0` (line 507), to cache row `pos / ratio`, with
`state_write_pos = pos + 1 - ratio` — the block's FIRST position, which is what
the compressed row then ropes at.

`state_source_idx` (lines 461-476) is the join that makes this work. It
addresses a graph-local tensor laid out
`[persistent_state | current_ubatch_scratch | sentinel]`:

- `pos < 0` → `state_rows + n_tokens`, the appended zero/`-inf` sentinel row;
- `pos` present in this ubatch → `state_rows + i`, the scratch row;
- otherwise → `stream_off + pos % state_size`, the persistent ring row.

For the overlapping compressors the reads are collected into **two contiguous
halves** — every block's previous-window indices, then every block's
current-window indices (lines 565-572) — which is exactly how
`build_overlap_compressed_kv_from_state` slices them back apart
(`deepseek4.cpp:463-489`). Getting that concatenation order wrong swaps the two
halves of every pooling window and still runs.

The pooling itself (the same four lines in
`build_overlap_compressed_kv_from_state` and in the HCA variant) is
**per-channel softmax over the WINDOW axis** — the `ratio` (or `2*ratio`) cached
rows a block pools — not over the feature axis: values and scores are both
permuted so the window index becomes the fast axis, `soft_max` runs, and
`sum_rows` collapses it. Then RMS-norm by `attn_comp_norm`, rope the
`[nope | rope]` tail at `compress_rope_base`, and write.

#### ✓ LANDED at the op level (2026-08-13), emitted for HCA since plan 09 S6

`Op::CompressPool` on CPU + Vulkan + Metal, one op for the four ggml nodes both
compressor variants share once their gathers have diverged (the pair of
`ggml_permute`+`ggml_cont`s, the `soft_max`, the `mul` and the `sum_rows`).
`values`/`scores` arrive `[blocks, window, n_embd]` and the permutes are folded
into the op's indexing, so infr pays none of the reference's two `ggml_cont`s of
the full permuted tensor. `window` is the op's name for what is `DSV4_HCA_RATIO`
on an HCA layer and `2*ratio` on the overlapping CSA/LID one, which is why one
op serves both.

Verified in `seam_op_parity.rs`'s `compress_pool_*` against a from-definition
f64 reference over windows 4 / 8 / 128, `blocks` 1 and >1, and an `n_embd` that
is not a multiple of a workgroup. Four deviations were injected into the CPU arm
and shown to go red; two are worth carrying forward:

- **Softmaxing the feature axis runs and looks fine.** It stays finite, keeps
  the output shape and lands in the right order of magnitude — it moved the
  answer by ~14× the output scale, but nothing downstream would have said so.
  This is the whole reason the pooling is one op rather than four generic ones.
- **The `-inf` sentinel does NOT by itself expose a missing max-subtract.**
  `exp(-inf)` is exactly `0.0`, so the naive `exp(s)/Σexp(s)` is algebraically
  the same answer on a window with SOME sentinel lanes. It breaks in exactly two
  places, and both are now test cases: scores large enough for `exp` to overflow
  f32, and a window that is ENTIRELY `-inf`.

That last case is `0/0`. **infr writes `0.0`; ggml produces `NaN`** — its
`ggml_vec_soft_max_f32` computes `exp(-inf − -inf)` and scales by `1/NaN`,
caught only by an `assert(sum > 0.0)` that release builds compile out. The
deviation is deliberate: zero is the value the sentinel's own zero `values` make
meaningful, a NaN cannot be told from a real defect once it has spread, and
three backends can be tested to agree on `0.0` where `NaN != NaN` makes a parity
assertion vacuous.

#### The four boundary conditions

Each is a place where a plausible implementation runs and is wrong.

1. **A partial block at the end of a prefill is invisible, and is never
   committed.** `n_visible` floors, and the commit is gated on
   `(pos + 1) % ratio == 0`. Tokens in a trailing partial block are recalled
   through the raw sliding window alone; their compressor-state rows persist and
   the block completes on a later ubatch. An implementation that ceils
   `n_visible`, or that flushes the partial block at the end of a prefill,
   exposes a row built from a half-filled window.

2. **The compressed half is always in the graph, even before any block
   completes.** `dsv4_build_comp_plan` ends with
   `plan.n_kv = max(GGML_PAD(plan.n_kv, 256), 256)`, so `n_kv` is never 0 and
   the compressed mask, and with it the compressed branch, exist on every
   ubatch; a ubatch in which no token has completed a block — any prefill
   shorter than 128 tokens on an HCA layer, shorter than 4 on a CSA layer — gets
   an all-`-inf` compressed half. (Earlier pins skipped the branch when `n_kv`
   was 0; that is no longer the reference.) An all-masked half weighs exactly
   zero, so infr may skip the segment when every `n_visible` is 0 and compute
   the same numbers; `dsv4_plan.rs` reports the unpadded width for that reason.

3. **Padded `n_kv` versus per-token visible length.** `plan.n_kv` is padded to
   256 so the graph shape does not change at every block boundary, and the mask
   is a plain per-token prefix:
   `data[i*ne0 + j] = j < n_visible[i] ? 0 : -INFINITY`
   (`llama-graph.cpp:659-681`). Rows in `[n_visible[i], n_kv)` therefore cover
   committed-but-not-yet-visible data and never-written zeros alike, and both
   are masked identically. A token whose `n_visible` is 0 while others in the
   ubatch are larger gets an **all-`-inf` compressed half** and must survive on
   the raw half plus the attention sink — which is exactly why the sink joins
   the softmax max as well as the denominator.

4. **Dummy commits to the last row, on every tier.** A dummy commit writes
   `cache_off + kv_size - 1` — the cache's LAST row — at rope position 0,
   sourced from the first token's state row repeated, purely so the compress and
   write ops keep their place in the graph. It is garbage, safe only because
   `n_visible < kv_size` always keeps it masked. Two rules add one, selected by
   `dsv4_build_comp_plan`'s `pad_blocks` argument:
   - **CSA and LID (`pad_blocks = true`)** pad each sequence's commits up to the
     ubatch's block count `ceil(max(1, n_seq_tokens)/ratio)`. `init_batch`
     splits with `split_equal`, so for one sequence `n_seq_tokens` is the whole
     ubatch. Contiguous positions complete that many blocks or one fewer, so
     this adds at most one dummy; a sequence two or more blocks short is refused
     ("DSV4 CSA sequence positions are not contiguous").
   - **HCA (`pad_blocks = false`)** adds one dummy when the ubatch commits
     nothing at all (`!pad_blocks && state_write_idxs.empty()`).

   Dummies are appended after every real commit, in the write vectors and in
   each read half, so a consumer that writes only the real commits takes a
   prefix.

#### Two corrections to what this document said about V4's indexer

Both are contradicted by the source, and both would produce silent wrongness.

- **V4's indexer ropes NORM, not NEOX.** Stage 3 above makes much of V3.2's
  indexer being NEOX-hardcoded while the main rope is NORM. V4 does **not**
  inherit that: `build_lid_top_k` ropes `indexer_q_pe` with the graph's
  `rope_type` (`deepseek4.cpp:555-557`), and `llama_model::rope_type` puts
  `LLM_ARCH_DEEPSEEK4` in the NORM group (`llama-model.cpp:2530`). Every V4 rope
  call site — q, kv, both compressors, the indexer — is NORM. The
  `hparams_lid.rope_type = LLAMA_ROPE_TYPE_NEOX` at
  `llama-kv-cache-dsv4.cpp:1063` is dead: it feeds only the KV-shift path, and
  `llama_kv_cache_dsv4::get_can_shift()` returns `false`.
- **V4's indexer head layout is `[nope | rope]`** — the opposite of V3.2's, and
  the same order as V4's own q and kv heads. `indexer_q_nope` is the view at
  offset 0, `indexer_q_pe` the view at `row_size(nope)`, concatenated in that
  order (`deepseek4.cpp:546-560`).

#### What the Hadamard rotation actually gates

`attn_rot_k` is normally on only for a quantized KV cache, but
`llama-kv-cache.cpp:346-356` force-enables it when the arch is DEEPSEEK32,
DEEPSEEK4 or GLM_DSA **and** `n_embd_head_k_full == indexer_head_size` — which
is true for the LID cache by construction (`hparams_lid` sets both, lines
1059-1062) and generally false for raw/CSA/HCA. That matters because
`build_attention`'s CSA dispatch is guarded on `inp_dsv4->get_lid().k_rot` being
non-null (`deepseek4.cpp:1053`): the guard passes for the reason above, but it
is keyed on a quantisation artifact rather than on anything semantic. The
rotation itself is orthogonal and applied to both sides of every dot product, so
an unquantised port skips it entirely — as stage 3 already does.

**Re-read at `5676dfbe0` for the HCA tier (plan 09 S6).** The rule now lives in
`llama_kv_cache`'s constructor (the `attn_rot_k` / `n_rot_k` block before
`init_rotation`): K is rotated when rotation is not disabled
(`LLAMA_ATTN_ROT_DISABLE`, `llama_model_set_attn_rot`), the cache has heads,
`type_k` is quantized and `n_embd_head_k() % 64 == 0`; or, for these DeepSeek
arches, when the cache's `n_embd_head_k_full` equals `indexer_head_size`. Each
V4 tier is its own `llama_kv_cache` with its own hparams, and only the LID cache
sets its head width to the indexer's. So at f16 neither the raw nor the HCA
cache is rotated — V4-Flash's head is 512 against an indexer width of 128, the
oracle fixture's 48 against 128 — and infr's unrotated f16 caches compute the
reference's function exactly. At a quantized `type_k` whose head width is a
multiple of 64 (q8_0 or q4_0 on V4-Flash's 512; never the fixture's 48) the
reference rotates the query, the raw row, the compressed row and the attention
output, and infr does not: the same function in exact arithmetic, but the
quantization error lands on rotated values there and on raw ones here. That
precision difference is unmeasured (`docs/backlog.md` § B-DSV4-WIRING).

Budget stage 4 accordingly, and do not start it until stages 2–3 are solid.

## Open questions — check these before trusting the above

Ordered by how much damage a wrong assumption does.

1. **Head layouts and exact dims** — ✓ RESOLVED. The
   `192 / 576 / 512 / 64 / 128` figures came from conversion-script formulas;
   `cpu_deepseek_v32_config` now asserts the MLA dims off a real V3-scale GGUF,
   and `cpu_deepseek2_config` does the same for V2-Lite.
2. **ggml type ids in V4 GGUFs** — ✓ RESOLVED for the shipped V4-Flash file: it
   opens and every tensor loads (`backlog.md` § B-DSV4-REAL). A new V4 quant
   outside `ggml_type_to_dtype` would still fail at open and need a new `DType`,
   `block_spec` and `dequant_block` arm; the i2_s commit `dbc8431` is the
   template.
3. **Whether N successive splits reproduce llama.cpp's tokenizer** (§0.2) — ✓
   RESOLVED (2026-08-09), for `deepseek-llm` only. `infr`'s ids were compared
   against `llama-tokenize --ids --no-bos --no-escape` on
   `deepseek-v2-lite-chat-q4_k_m.gguf`. Note that this GGUF is
   `tokenizer.ggml.pre == "deepseek-llm"`, **not** `deepseek-v3` — so what it
   exercises is the six-regex V1 list, the one that carried both transcription
   slips. They agreed **exactly on all 31 texts**, covering digits, decimals and
   grouped numbers, CJK, Hangul, Greek, Hebrew, Arabic, Devanagari, punctuation
   runs, code, emoji, CRLF, smart quotes and non-ASCII whitespace. So N
   successive `Isolated` splits do reproduce `unicode_regex_split` here.

   The comparison was shown to be capable of failing: re-introducing the U+0027
   slip made `infr` disagree with llama.cpp on 8 of 11 texts in the
   NBSP-before-punctuation battery (e.g. `"a \u{00A0}. b"` → llama.cpp
   `[64, 207, 1202, 13, 270]`, broken `infr` `[64, 30683, 13, 270]`).

   **`deepseek-coder` and `deepseek-v3` have no token-id coverage.** Both GGUFs
   in the local HF cache are `deepseek-llm`, so neither of those lists was
   exercised against real ids. They are structurally identical (same
   `build_multi_split_seq`) and byte-identical to the reference, and their chunk
   boundaries are pinned by the unit test — but that is not the same as having
   been checked against llama.cpp. Re-open this for either list if a matching
   GGUF appears.

4. **Shared-expert width when `n_shared_experts > 1`** — ✓ RESOLVED on V2-Lite
   (2 shared experts): see § Stage 1.
5. **`rope_off`** — ✓ RESOLVED (2026-08-06). `Op::Rope` only rotates standalone
   k_pe slices (extracted via `CopyStrided`, no nope prefix). The q_pe rope is
   done inside the MLA kernel at offset `qk_nope_dim` — the offset lives in the
   kernel, not in `Op::Rope`. No `rope_off` field needed.
6. **YaRN** — RESOLVED (2026-08-07). The per-dimension frequency ramp IS
   implemented and the mscale² is a constant (both folded per `ggml_rope_yarn` +
   `deepseek2.cpp:162-172`). The earlier note claimed the ramp is "INERT for
   default deepseek2 GGUFs" because it assumed the convert script never writes
   `rope.scaling.factor`/`type` — **wrong**: the V2-Lite Q4_K GGUF declares
   `rope.scaling.type = yarn`, `factor = 40`, `original_context_length = 4096`,
   `yarn_log_multiplier = 0.0707`, which makes llama.cpp set
   `yarn_ext_factor = 1.0` (llama-context.cpp:189-191) and run the FULL ramp at
   every context length. Without it, infr's greedy output was
   `"Reply Collabor…"` garbage while llama.cpp produced coherent text. The ramp
   lives in `Op::Rope.freq_factors` (per-pair divisors, computed in the seam
   from the corr_dims spectral ramp) plus the MLA kernels' internal q_pe rope;
   the mscale² is folded into the MLA attention scale as a constant
   (`mscale = 1 + 0.1·log_mul·ln(factor)`, applied via
   `mla_scale = mscale²/√(qk_nope + qk_rope)` — note `qk_nope = head_k_mla` is
   128 for V2-Lite, so the denominator is √192, not √576). The rope vector
   mscale cancels to `rope_attn_factor` for deepseek2, so no vector scaling is
   needed in the kernels.
7. **DeepSeek's EOS** — `add_chat_eos` appends a fixed list that does not
   include `<｜end▁of▁sentence｜>`. It is normally the GGUF's declared
   `tokenizer.ggml.eos_token_id` and therefore already in `eos_ids`, but check
   whether the chat template ends turns on something else.
8. **`LLM_TENSOR_ATTN_KV_NORM` and `LLM_TENSOR_ATTN_KV_A_NORM` share the on-disk
   name** `blk.%d.attn_kv_a_norm`. Two enum values, one string — not
   distinguishable on disk.
9. **llama.cpp's V4 support is young.** Its model-type detection for V4 is a
   stub where both branches return `UNKNOWN`. Treat the reference as possibly
   buggy rather than authoritative. Seen at `5676dfbe0`: a V4 file with no
   ratio-4 layer kills the process with an integer division by zero while the
   lightning-indexer cache is built, and `hyper_connection.count` must be 4
   (plan 09 section 4).

## What was not covered

- `deepseek2-ocr` — out of scope.
- The **DSpark speculative module** (`dspark_block_size`,
  `dspark_target_layer_ids`, `dspark_markov_rank`). It is a separate head over
  V4's last three layers and does not appear in the graph builder at all. `infr`
  has MTP machinery (`docs/mtp.md`) that may host it; not investigated.
- V3.2's **NextN** tensors — loaded but skipped by llama.cpp.
- Performance. This plan is about correctness only. Nothing here is measured,
  and no throughput claim is made.
