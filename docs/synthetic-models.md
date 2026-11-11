# Synthetic model fixtures

A plan to give every supported architecture a fake, in-repo, structurally
complete stand-in GGUF — so the multi-GB real models can be deleted from the HF
cache without losing coverage, and so CI covers more than two architectures.

## The proposal, and the one amendment it needs

The idea as stated: build fake models with the same layouts as the real ones,
bless their outputs with the same code, and rely on this argument —

> If the same implementation generates _x_ on the fake while passing the goldens
> on the real models, the fake is a valid stand-in as long as its golden is
> tracked. If a logic change would move the fake's output, the code would also
> be broken on the real model.

The **contrapositive is sound and is the useful half**: _fake output moved ⟹
behaviour changed_. That is exactly what a regression test owes us, and it is
why this work is worth doing.

The converse does not hold, and two failure modes follow from that. Both are
already recorded in this repo, so neither is hypothetical.

### Failure mode 1 — a golden preserves behaviour, it does not establish correctness

From the `coherent-is-not-correct` incident: every `deepseek2`-family model
computed `x + 2·Wo·attn` for weeks, on every backend, because the defect was in
the shared graph. The text read as fluent English throughout. And:

> Goldens passing. **Two goldens were blessed off the doubled residual stream,
> so they locked the bug in and reported green forever after.**

A fake blessed from infr's own output does the same thing, permanently. Nothing
about the fake being synthetic makes this worse — but nothing makes it better
either, and a fake is _easier_ to bless carelessly because there is no readable
text to sanity-check. The same incident also kills the obvious fallback:

> CPU-vs-GPU agreement. Useless here: a shared-graph defect is present on both
> sides, so they agree with each other while both being wrong.

**Consequence:** a fake's golden may only be blessed against an **external
oracle**, never against infr's own output alone. That means the real model is
needed exactly once per architecture — at bless time — and is deletable
afterwards. The oracle is `llama-debug --save-logits` (verified present at
`examples/debug/debug.cpp`, flag registered at `common/arg.cpp:4536`), scored as
probability cosine at several prompt lengths, with a known-good control model
through the same harness.

### Failure mode 2 — exact-token goldens are not portable

This is the one that decides the assertion shape. Commit `273f8d4` (2026-07-22),
verified:

> `cpu_golden_qwen35` produces a COHERENT but different greedy trajectory on the
> GitHub x86 runner than on the dev box — its longer (n=48) gated-DeltaNet
> generation is FP-sensitive, so the exact-token FNV golden hash isn't
> reproducible across CPU microarchitectures (got `0x0a0d..21` vs want
> `0xbe06..78`; output was a valid brave-knight story). gemma3 and qwen3 (n=32,
> short/stable) DID reproduce bit-for-bit and passed.

Same OS, same ISA family, different microarchitecture, and the fix taken was to
**drop the model from CI** rather than change the assertion. `de987d7` is the
same mechanism from the other direction: an int8-tier flip moved a close-margin
greedy argmax, staled `gpu_seam_golden_qwen3`, and left `main` red for a day
while the answer stayed correct.

Random-weight fakes make this strictly worse. A trained model's argmax has a
wide margin because the model is confident; an untrained one's logits are
near-uniform, so essentially every token is a near-tie and any change in
reduction order flips it.

**Consequence:** do not freeze `fnv1a(decoded_text)` for synthetic models. Two
better options, and we should use both:

1. **Freeze logits, score with tolerance.** This is what the existing synthetic
   harness already does — `GOLDEN_DS32_TOPK5` is asserted at `d < 1e-4 * scale`,
   with a comment noting cross-machine float noise sits many orders below that
   bound.
2. **Engineer the weights for a wide margin.** We control the fake's weights, so
   we can make the output projection produce a decisive argmax. A synthetic
   model can be _more_ stable than a real one — which is the direct fix for the
   thing that got Qwen3.5-0.8B dropped from CI.

## Why this is worth doing anyway: the numbers

Measured from the last three successful `main` runs:

| job                                      | runs                            | duration                   |
| ---------------------------------------- | ------------------------------- | -------------------------- |
| `cargo test` (`nextest run --workspace`) | every commit, no downloads      | **3m38s**                  |
| `cargo test (CPU goldens, real models)`  | every commit, downloads ~1.7 GB | **24.4 / 36.3 / 35.7 min** |

The goldens job is the CI long pole by a factor of ~10, and it covers exactly
**two** architectures — gemma-3-1b and Qwen3-0.6B — because the workflow
deliberately excludes the multi-GB models. The synthetic DeepSeek harness
already runs inside the fast job, with no downloads, on every commit.

And the silent-skip problem is real, not theoretical. The workflow's own comment
records it:

> on a bare runner they always no-op — which is exactly how a real
> arch-correctness bug (the Q8_0 sub-256 truncation that changed gemma-3-1b's
> output) slipped past CI.

Measured on this machine today, `crates/infr-llama/tests/cpu_backend.rs` has
**84 `#[test]` functions**: 45 `#[ignore]`d, 6 `cfg(target_os = "macos")` (not
compiled on Linux at all), leaving **33 that run under plain `cargo test`**. Of
those 33, **6 self-skip** for a missing fixture and report as passed —
`cpu_golden_qwen3moe`, `cpu_qwen35moe_prefill_finite`, `cpu_llama4_config`,
`cpu_llama4_scout_greedy`, `cpu_diffusion_gemma_prefill_finite`,
`cpu_diffusion_gemma_denoise_step`. A seventh,
`cpu_prefill_matches_llama_debug_dump`, skips for a missing env var.

Worse, `cpu_golden_qwen3_quants` skips **inside a loop**: only the `Q4_K_M` file
is present locally, so 7 of its 8 quant cases are dark while the test reads
green. That is the "guard whose scope silently matches nothing" shape.

**Synthetic fixtures never self-skip.** That alone is a coverage upgrade
independent of disk space.

## What already exists — this is a generalization, not a new build

`crates/infr-llama/tests/synthetic_deepseek2.rs` already does exactly this for
`deepseek2`, `deepseek32` and `deepseek4`: it builds a GGUF in memory, writes it
to a temp file, and drives it through the real `Config::from_gguf` and the real
seam. Its module doc has an explicit "Adding an architecture" section.

**Reusable as-is** (everything above `mla_model`): `Meta` (the metadata value
writer), `TensorSpec` + `Fill` (name, ggml-order shape, deterministic values),
`SyntheticModel` and its byte writer, `TempGguf`.

The determinism contract is sound and verified: fills are seeded by
`fnv1a64(tensor_name)` through pure `u64` wrapping arithmetic, with the float
produced by dividing the top 24 hash bits by an exact power of two —
bit-identical on any IEEE-754 platform. Metadata and tensors are `Vec`s, never
`HashMap`s, and `mla_model` sorts its metadata explicitly. A test already pins
`to_gguf_bytes() == to_gguf_bytes()`.

The existing assertion mix is also the right model to copy: of its tests, only
**one** is a frozen numeric array (tolerance-scored), ~6 are exact structural
equality on bytes/config, ~10 are negative tests that damage the fixture and
demand a specific error, ~7 are CPU-vs-Vulkan cosine differentials, and the rest
are same-backend differential tripwires. **None freezes a token sequence.**

### What the harness cannot express yet

These are the concrete build items, not vague gaps:

- **Quantized weights.** `TensorSpec::ggml_type()` returns F32 for everything
  except an I32 routing table. Real files are mixed — the V4 fixture's own
  backlog note records the real file's dtype set as
  `{Q2_K: 129, Q8_0: 660, F32: 492, BF16: 43, Q6_K: 1, I32: 3}`, and that **the
  43 BF16 tensors are the per-layer `ffn_gate_inp.weight` routers, which no V4
  test has ever exercised.** Meanwhile `infr-testkit` already synthesizes valid
  blocks for all 24 weight quants via `synth_weight(dtype, n_elem, seed)` — but
  the two are entirely disconnected: `infr-llama` does not depend on
  `infr-testkit` at all. Wiring them needs a `Fill::Quant(DType)` that emits
  pre-encoded bytes rather than going through `values() -> Vec<f32>`, plus a
  `DType -> ggml_type` reverse table (only the forward direction exists today).
- **Metadata types.** The writer emits U32, F32, Bool, Str, and arrays of
  Str/I32/F32. No U64, I64, F64, or U32-arrays. The reader is fully general, so
  this is a writer-side limit only.
- **Multi-shard files.** `to_gguf_bytes()` writes a single file with no
  `split.*` keys. The reader's shard logic (`open_split`, `parse_shard`,
  `shard_set`) is implemented and reachable only from real multi-file GGUFs.
- **The decode path.** Synthetic tests run a single causal prefill via
  `verify_dense_cpu`/`verify_dense_vulkan`. They never touch the autoregressive
  decode loop, sampling, the tokenizer (the prompt is raw `u32` ids), or chat
  templating. `cpu_backend.rs`'s `cpu_gen` exercises all four.
- **Shape validation.** Backlog **B52** already records that the loader
  validates tensor _names_, not _shapes_ — so every "the loader consumes every
  tensor" test proves only that each name was requested. A generalized harness
  inherits that blindness unless B52's shared expected-dims check lands first.
  **Read B52 before extending the harness.**

## Where the coverage actually is today

Per-architecture, from `crates/infr-llama/src/arch.rs`'s 16 constants:

| arch              | real-model coverage today                     | status                                 |
| ----------------- | --------------------------------------------- | -------------------------------------- |
| `qwen3`           | CPU golden + GPU golden + Metal + quant sweep | strongest; keep a real fixture         |
| `gemma3`          | CPU golden + 2 GPU seam                       | in CI today                            |
| `gemma4`          | CPU golden (E2B only) + 3 GPU seam            | dense 12B has no CPU golden            |
| `qwen35`          | CPU golden + GPU seam + 8 MTP tests           | golden **not CI-portable**             |
| `deepseek2`       | CPU golden + config + 4 dequant-parity        | strong                                 |
| `deepseek`        | config + weak "non-empty output" prefill      | weakest assertion of any arch          |
| `bitnet`          | config + exact top-1 token + GPU seam         | good                                   |
| `llama`           | GPU/Metal only — **no CPU test at all**       | dark under plain `cargo test`          |
| `qwen2`           | one `#[ignore]`d GPU test — **nothing else**  | zero running coverage locally          |
| `qwen3moe`        | all 5 tests dark (fixture absent)             | **fully dark**                         |
| `qwen35moe`       | all fixture-backed tests dark                 | **fully dark**                         |
| `llama4`          | all 3 correctness tests dark                  | **fully dark**                         |
| `diffusion-gemma` | all 6 tests dark                              | **fully dark**                         |
| `deepseek32`      | synthetic only — no real GGUF declares it     | converter gap, not a download gap      |
| `deepseek4`       | synthetic only — smallest real quant ~82.5 GB | by design                              |
| `bitnet-b1.58`    | **nothing, at any tier**                      | never tested; not even a config assert |

Model-level quant coverage is narrower than it looks: of 24 weight quants swept
at the op level, **13 are never exercised through a model load at all** —
`Q4_1, Q5_1, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, TQ1_0, Q2_0, MXFP4, NVFP4`.

## Design rules

1. **Never freeze decoded text or a token hash for a synthetic model.** Freeze
   logits and score with a relative tolerance, following the existing
   `GOLDEN_DS32_TOPK5` pattern.
2. **Bless only against an external oracle.** `llama-debug --save-logits` over
   llama.cpp's own token ids, probability cosine at several prompt lengths, with
   a known-good control through the same harness. Record the oracle score in a
   comment next to the golden, so a future reader can tell a validated golden
   from a self-blessed one. A golden with no recorded oracle score is a golden
   nobody has checked.
3. **Synthetic tests must never self-skip.** No `need_model!`, no `#[ignore]`
   except for a genuine GPU-device requirement. If it cannot run, it must fail.
4. **Keep the negative tests.** The damaged-fixture tests — missing tensor,
   missing metadata key, misnamed tensor, wrong shape — are the part of the
   existing harness that a real model can never provide cheaply, and they are
   where most of its value sits.
5. **Make the fakes adversarial, not merely small.** This is the part that makes
   synthetic fixtures better than real ones rather than a cheap substitute. We
   choose the shapes, so choose the ones real models do not reach: expert counts
   above 256, a KV ring that wraps within a few tokens, sequence lengths that
   straddle a chunked-prefill boundary, dispatch sizes near the 65535 group
   limit, non-unit V/K head ratios, mixed dtypes per tensor. **A synthetic
   512-expert model would have caught B67** — the `moe_topk.comp` out-of-bounds
   shared-memory access, since fixed — and it is still the only way to run
   routing above 256 experts at all (`backlog.md` B70).
6. **One fixture per architecture, sized to run in the fast job.** The budget is
   the current `cargo test` job's 3m38s, not the goldens job's half hour.

## Staging

1. **Quantized fills.** Add `infr-testkit` as an `infr-llama` dev-dependency (an
   intra-workspace edge, but still a dependency decision to confirm), add
   `Fill::Quant(DType)` and the `DType -> ggml_type` reverse table. Nothing else
   in this plan is worth much without it: an all-F32 fixture cannot exercise the
   dequant path that most real bugs live in, and it cannot express a BF16
   router.
2. **Land B52's shape validation first,** so "consumes every tensor" means what
   it says.
3. **Extend the harness to the decode path** — decode loop, sampling, tokenizer,
   chat template — or state explicitly that synthetic fixtures cover load and
   prefill only and that decode coverage stays with a small number of real
   models. This is the biggest open design question in the plan and should be
   decided before building fixtures, not after.
4. **Fill in the fully dark archs first**, in this order, because they have no
   coverage at all today and need no oracle work to improve on nothing:
   `bitnet-b1.58`, `llama4`, `qwen3moe`, `qwen35moe`, `diffusion-gemma`.
5. **Then the archs whose real fixtures we would like to delete:** `qwen35moe`
   and `qwen3moe` are the expensive ones; `deepseek2` (9.7 GB) and
   `deepseek-moe-16b` (11 GB) are the next largest.
6. **Adversarial fixtures last** — once the per-arch shapes exist, adding a
   512-expert or ring-wrapping variant is a dims change, not new machinery.

## What this can never replace

- **Arch correctness.** Only the external oracle establishes that infr computes
  what llama.cpp computes. Keep `cpu_prefill_matches_llama_debug_dump` working,
  and keep a documented path to re-pull each real model — a repo-tracked list of
  `org/repo:quant` plus the expected sha256, so a future bless is a command, not
  an archaeology exercise.
- **Tokenizer and chat-template fidelity.** A fake's vocab is a stub. Real
  templates, real merges, real special tokens need a real file.
- **Scale-dependent behaviour.** Weight streaming, expert paging, KV overflow
  and VRAM pressure are properties of a 27B model, not a 3-layer one.
- **Trained-weight numerics.** Real weights have outlier channels; uniform
  random weights do not. Overflow, saturation and outlier-clamp paths may simply
  never be entered by a fake, which is why rule 5 says to engineer those cases
  in deliberately rather than hope they arise.

Keep at least one real model per _family_ for the oracle check, and delete the
rest. The goal is not zero real models — it is that no test silently vanishes
when one is missing.
