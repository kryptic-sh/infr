# infr

[![CI](https://github.com/kryptic-sh/infr/actions/workflows/ci.yml/badge.svg)](https://github.com/kryptic-sh/infr/actions/workflows/ci.yml)

Pure-Rust LLM inference engine. Vulkan-first, built to run on any mainstream
GPU.

> Early WIP. The only non-Rust parts are the GPU driver calls (Vulkan via `ash`)
> and the compute shaders (SPIR-V).

## Goal

A from-the-metal inference server that works across AMD / NVIDIA / Intel
(Vulkan) and Apple (MoltenVK), with native backends addable later behind a
`Compute` trait.

## Status

Runs **Llama / Qwen2 / Qwen3** (dense), **Gemma 3** (dense, sliding-window
attention + QK-norm + GeGLU), and **Gemma 4** (per-layer heterogeneous head
dims, proportional RoPE, V-norm, per-layer output scale — including the **E2B**
variant: per-layer input embeddings, per-layer FFN widths, KV-layer sharing) on
the Vulkan backend, competitive with llama.cpp at long context (`infr compare`).
**Qwen3.5 / Qwen3.6** (`qwen35` — hybrid gated-DeltaNet + attention, a sibling
of Qwen3-Next) run on the same unified runner, CPU + Vulkan (`docs/QWEN35.md`).
**DiffusionGemma** (the original target — block text-diffusion MoE on a Gemma-4
backbone, entropy-bound denoise decode) runs end-to-end on CPU + Vulkan
(`docs/DIFFUSIONGEMMA.md`).

```bash
infr pull   <model-ref>        # org/repo[:quant] (HuggingFace) | path to a .gguf
infr run    <model-ref> [msg]  # terminal chat (auto-pulls)
infr serve  <model-ref>        # OpenAI-compatible HTTP API
infr bench / infr compare      # tok/s benchmarks vs llama.cpp
```

Model refs match llama.cpp's `-hf`: `org/repo[:quant]` (quant default `Q4_K_M`,
e.g. `infr run unsloth/Qwen3-14B-GGUF:Q4_K_M`). Models share the standard
**HuggingFace Hub cache** (`~/.cache/huggingface/hub`) with llama.cpp and
`huggingface_hub` — one download, used by both.

## Supported models

All run on the Vulkan GPU backend unless noted. The chat template (turn markers,
system prompt) is read from the GGUF's own `tokenizer.chat_template`.

| Family            | Arch (GGUF)       | Notes                                                   |
| ----------------- | ----------------- | ------------------------------------------------------- |
| Llama             | `llama`           | dense transformer                                       |
| Llama 4           | `llama4`          | sigmoid top-1 MoE + shared expert, iRoPE, paged experts |
| Qwen2 / Qwen2.5   | `qwen2`           | dense, QKV bias, NEOX rope                              |
| Qwen3             | `qwen3`           | dense, QK-norm                                          |
| Qwen3 MoE         | `qwen3moe`        | softmax router, top-_k_ experts, paged experts          |
| Gemma 3           | `gemma3`          | SWA + QK-norm + GeGLU, dual-RoPE                        |
| Gemma 4 (dense)   | `gemma4`          | per-layer head dims, proportional RoPE, V-norm          |
| Gemma 4 **E2B**   | `gemma4`          | + per-layer input embeddings / FFN, KV sharing          |
| Gemma 4 **MoE**   | `gemma4`          | 26B-A4B: dual FFN (dense GeGLU ∥ 8-of-128 routed), AR   |
| Qwen3.5 / Qwen3.6 | `qwen35`          | hybrid gated-DeltaNet + attention (NOT `qwen3next`)     |
| Qwen3.6 MoE       | `qwen35moe`       | `qwen35` skeleton + routed experts + shared expert      |
| DiffusionGemma    | `diffusion-gemma` | block text-diffusion MoE, entropy-bound denoise decode  |

Fine-tunes on any of these backbones run unchanged. **Ornith-1.0**
(DeepReinforce.AI agentic-coding) validated 2026-07-09 — the 9B rides `qwen35`
and the 35B rides `qwen35moe` with no code changes
(`infr run deepreinforce-ai/Ornith-1.0-9B-GGUF:Q4_K_M "..."`).
**Ternary-Bonsai** (Prism ML, weights trained to {-1, 0, +1}) validated
2026-07-12 — the 1.7B / 4B / 8B all ride `qwen3`, zero-code, both in the TQ2_0
repack (`superkaiii/Ternary-Bonsai-4B-GGUF`) and in llama.cpp's new **Q2_0**
weight dtype (2.25 bpw, GGML type 42 — native in-shader dequant + dp4a mmq, no
fork needed). infr is the **only engine that runs Q2_0 on a GPU** (llama.cpp
merged the dtype CPU-only) — see the perf table below. Pull the `Q2_0_g64`
files: `infr run prism-ml/Ternary-Bonsai-8B-gguf:Q2_0_g64 "..."`.

```bash
# Qwen3 dense
infr run unsloth/Qwen3-1.7B-GGUF:Q4_K_M "What is the capital of France?"

# Qwen3 MoE (experts page through the VRAM LRU cache when they don't fit —
# see INFR_CACHE below)
infr run unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M "Explain MoE routing."

# Llama 4 Scout (37 GB Q2_K) — paged expert cache runs it on a 24 GB card
infr run unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF:Q2_K "What is the capital of France?"

# Gemma 3
infr run unsloth/gemma-3-1b-it-GGUF:Q4_K_M "What is bash?"

# Gemma 4 — dense and the E2B variant
infr run unsloth/gemma-4-12b-it-GGUF:Q4_K_M  "What is the capital of France?"
infr run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M  "What is bash?"

# DiffusionGemma — block text-diffusion decode (entropy-bound denoise)
infr run unsloth/diffusiongemma-26B-A4B-it-GGUF:Q4_K_M  "What is the capital of France?"

# Pick a specific quant with the `:quant` suffix (default is Q4_K_M)
infr run unsloth/Qwen3-8B-GGUF:Q6_K       "Summarize the plot of Hamlet."
infr run unsloth/Qwen3-0.6B-GGUF:IQ4_XS   "Write a haiku about Rust."

# Qwen3.5 speculative decoding (opt-in MTP head; output is token-identical to
# greedy — pure speedup). WIP: fastest on real, content-rich prompts.
INFR_MTP=1 infr run unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_XL "Explain how a hash map works."

# Sampling: greedy by default (INFR_TEMP=0). Temperature / top-k / top-p:
INFR_TEMP=0.7 INFR_TOP_K=40 INFR_TOP_P=0.95 \
  infr run unsloth/Qwen3-1.7B-GGUF:Q4_K_M "Tell me a story."
```

### Serving

```bash
# OpenAI-compatible HTTP API (streaming). Reuses a persistent KV cache across
# requests (common-prefix diff) for fast TTFT on shared-prefix chats.
infr serve unsloth/Qwen3-14B-GGUF:Q4_K_M          # default: 127.0.0.1:8080

curl -s localhost:8080/v1/chat/completions -d '{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "stream": true
}'
```

Works as a drop-in backend for OpenAI-API clients (opencode, the Claude Code
CLI, etc.). Tool calling renders the model's own `tokenizer.chat_template`
(Qwen, Llama-3.x, Gemma tool dialects supported).

Sampling is greedy at `INFR_TEMP=0`; otherwise `INFR_TEMP` / `INFR_TOP_K` /
`INFR_TOP_P` control it (see
[Benchmarking & profiling](#benchmarking--profiling) for the full env list).

## Benchmarking & profiling

`infr bench` matches `llama-bench`'s `-p`/`-n`/`-d`/`-r` flags, so the two are
directly comparable. Pipelines are compiled and GPU state is first-touched at
model load (`Llama::warmup`), so timing measures compute, not one-time setup.
**Run benchmarks one at a time** — concurrent GPU work skews results.

```bash
M='unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M'   # MoE perf target

# Prefill (pp = n_prompt/time) and decode (tg = n_gen/time):
infr bench "$M" -p 2048 -n 0 -r 3       # prefill 2048 tokens
infr bench "$M" -p 8000 -n 0 -r 2       # prefill at depth
infr bench "$M" -p 0 -n 64 -r 3         # decode 64 tokens
infr bench "$M" -p 0 -n 64 -d 2048      # decode at context depth 2048 (-d warms, untimed)
```

**Profile** per-op GPU time (timestamp queries) with `INFR_PROF2=1`. Every
dispatch is timestamped and labeled **automatically with its kernel name** (plus
a few role overrides like `expert_gateup`/`expert_down`); no manual stamping. It
prints one block per submit and ONE aggregated `INFR_PROF2 GPU report` at
process exit (per-kernel totals, counts, avg, %GPU over all timed submits —
warmup runs unprofiled). Add `INFR_PROF2_SHAPES=1` for shape-itemized GEMV/GEMM
buckets (`mmvr:m4:1536x24576`). Decode's replay tape carries no timestamps —
profile decode with `INFR_SEAM_NO_REPLAY=1`. Details in
[`docs/PERF.md`](docs/PERF.md).

```bash
INFR_PROF2=1 infr bench "$M" -p 2048 -n 0 -r 1 2>&1 | tail -30   # exit aggregate
```

**Validate Vulkan work** — any change touching `infr-vulkan` (kernels, recorder,
adapter, pager) must run its GPU tests and at least one end-to-end generation
under the Khronos validation layer, and fix every error AND warning it reports
before landing (validation silence is the bar, not "it produces the right
tokens" — robust-access reads, missing barriers, and binding-range overflows can
return plausible garbage instead of crashing):

```bash
VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation cargo test -p infr-vulkan -- --ignored
VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation infr run "$M" "smoke prompt"
```

The layer ships with the `vulkan-validation-layers` package. It slows GPU work
noticeably — use it for correctness passes, never inside timed benches.

**Compare to llama.cpp** — `infr compare` shells out to `infr bench` and the
system `llama-bench` with matching flags on coding-agent-shaped workloads
(prefill, decode-at-depth, whole turns). `--ctx` is comma-delimited:

```bash
infr compare "$M" --ctx 8000,16000 --gen 256 --turn 2048,256 --reps 2
```

**DiffusionGemma** has no upstream-merged `llama-bench` support, so
`infr compare`/`infr compare --sweep` route `arch=diffusion-gemma` models to a
different oracle: the reference fork's `llama-diffusion-cli`
(`~/Projects/mxaddict/llama.cpp-dg`, resolved via `INFR_LLAMA_DIFFUSION_CLI` >
`PATH` > the fork's `build-vulkan`/`build` directories — see
`ModelBench::llama_diffusion_cli_path` for the exact precedence and its PATH
fallback caveat). It prints two rows instead of the usual pp/tg matrix:
`dg-step` (in-step-parallel tok/s ratio — the apples-to-apples number, since
both implementations run entropy-bound and take a different number of denoise
steps) and `dg-e2e` (informational end-to-end tok/s, each side's own step count
folded into the row so the mismatch is visible). Details in
[`docs/DIFFUSIONGEMMA.md`](docs/DIFFUSIONGEMMA.md).

Useful env: `INFR_TEMP` / `INFR_TOP_K` / `INFR_TOP_P` (sampling; `TEMP=0` →
greedy), `INFR_MAX_NEW`, `INFR_CTX`, `INFR_NO_FLASH`.

**MoE expert placement**: resident when the expert banks fit VRAM (zero config,
zero change); otherwise every layer pages through a VRAM-resident LRU expert
cache (`infr_vulkan::pager`) sized to the remaining VRAM. `INFR_CACHE=<size>`
forces every layer through the pager with that budget regardless of fit (useful
for testing, or to free VRAM for a larger context). Every bank shape pages:
split gate/up (llama4/Qwen3-MoE/Qwen3.6-MoE), fused gate_up (DiffusionGemma,
Gemma-4 MoE — one double-width slot per expert), and mixed-dtype roles
(unsloth-dynamic quants bumping a subset of layers' banks to a wider K-quant —
one arena pool per (role, byte size)). `INFR_PAGER_STATS=1` prints each pool's
hit/miss/eviction counts.

**Dense layer streaming**: DENSE models bigger than VRAM stream their per-layer
projection weights (attn q/k/v/o + FFN gate/up/down, as the same fused
qkv/gate_up groups the loader uploads) through the same paged VRAM machinery —
but schedule-driven, not LRU: a dense forward visits layers in one fixed order,
so residency uses an exact cyclic-sweep policy (Belady-parity — a stable
resident prefix plus one churn slot per pool) and there are NO readbacks
anywhere (every "miss" is known in advance; misses ride recorded ring→arena
copies on the same pipelined fenced-half staging ring the MoE path uses, so CPU
memcpys for later layers overlap GPU execution of earlier ones). Streamed
dispatches are the ordinary dense kernels reading the pool arena at a slot
element offset (the `w_off` convention) — no kernel variants, so streamed output
is token-identical to the resident run. Embeddings, lm_head, norms and biases
stay resident (lm_head is read at every token edge — streaming it adds its full
bytes to every token's PCIe bill with zero locality to exploit). Placement is
automatic (resident when everything fits — zero change); `INFR_CACHE=<size>`
forces streaming with that budget. Honest expectations: prefill amortizes
uploads across the whole batch (Qwen3-14B Q8_0, ~15.7 GB, at `INFR_CACHE=8g`:
pp512 987 t/s vs 1505 resident = 0.66×); decode has no locality to exploit, so
it is capped at PCIe_bw ÷ overflow_bytes per token — physics, not a bug (same
setup: ~7.0 GB re-uploaded per token ÷ ~22 GB/s ≈ 3.1 t/s ceiling, measured 3.1
t/s; the CPU backend does 4.4 t/s at that ~45% overflow, so streaming only beats
CPU when the overflow is smaller — measured crossover on this box is around a
quarter of the model overflowing). An MoE model whose DENSE part also doesn't
fit is out of scope and errors clearly.

**Size grammar** — `INFR_CACHE` and `INFR_CTX` share one value grammar
(`infr_core::parse_size`): a plain number is the base unit (bytes for
`INFR_CACHE`, tokens for `INFR_CTX`), `k`/`m`/`g`/`t` suffixes scale by 1024
(`INFR_CACHE=19g`, `INFR_CTX=256k`), and `%` resolves against the
device-appropriate base — available VRAM for the expert cache, the free-VRAM KV
capacity for the Vulkan context (`INFR_CACHE=80%`, `INFR_CTX=50%`; on the
CPU/Metal chat paths a ctx-`%` resolves against the model's trained context).

## Validated models & performance

Everything below is **validated on an AMD Radeon RX 7900 XTX** (RDNA3, 24 GB,
Vulkan / RADV): correctness is checked against the CPU reference implementation
(the `gpu_seam_matches_cpu_*` tests generate token-for-token on both and
compare) and throughput is measured against the system `llama.cpp` build with
`infr compare`.

**Throughput vs llama.cpp** — ratios are `infr / llama.cpp` (**>1.0 = infr is
faster**); r=3, 2026-07-12 snapshot (commit `8513358`, every model×quant in the
local cache, oracle `llama-bench` **b9957** on every row). Hardware: **AMD
Radeon RX 7900 XTX** (RDNA3, 24 GB, Vulkan / RADV, Mesa). `pp512` = 512-token
prefill throughput, `tg128` = 128-token decode throughput, `tg64@d4096` = decode
at 4096 KV depth, `pp4@d4096` = short-turn prefill at 4096 KV depth (the
multi-turn serve shape).

| Model                 | Quant       | pp512      | tg128     | tg64@d4096 | pp4@d4096 |
| --------------------- | ----------- | ---------- | --------- | ---------- | --------- |
| Qwen3-0.6B            | Q2_K        | **1.30×**  | **1.46×** | **1.36×**  | **2.17×** |
| Qwen3-0.6B            | IQ4_XS      | **1.14×**  | **1.12×** | **1.19×**  | **2.01×** |
| Qwen3-0.6B            | Q4_0        | **1.23×**  | **1.32×** | **1.28×**  | **2.16×** |
| Qwen3-0.6B            | Q4_K_M      | **1.16×**  | **1.17×** | **1.23×**  | **2.01×** |
| Qwen3-0.6B            | Q5_K_M      | **1.08×**  | **1.19×** | **1.23×**  | **2.04×** |
| Qwen3-0.6B            | Q6_K        | **1.23×¹** | **1.10×** | **1.15×**  | **1.86×** |
| Qwen3-0.6B            | Q8_0        | **1.25×**  | **1.18×** | **1.20×**  | **2.09×** |
| Qwen3-0.6B            | BF16        | **1.08×**  | 0.87×     | 0.93×      | **1.83×** |
| Qwen3.5-0.8B          | Q4_K_M      | **1.02×**  | **1.12×** | **1.06×**  | **1.74×** |
| Gemma-3-1B            | Q2_K        | **1.16×**  | **1.09×** | **1.02×**  | **1.15×** |
| Gemma-3-1B            | Q4_K_M      | **1.04×**  | **1.18×** | **1.09×**  | **1.16×** |
| Gemma-3-1B            | Q8_0        | **1.44×**  | **1.25×** | **1.15×**  | **1.09×** |
| Llama-3.2-1B          | Q4_K_M      | 0.99×      | 0.96×     | 0.88×      | 0.96×     |
| Llama-3.2-1B          | Q8_0        | **1.05×**  | **1.05×** | 0.90×      | **1.01×** |
| Qwen3-1.7B            | Q4_K_M      | **1.11×**  | **1.09×** | **1.13×**  | **1.70×** |
| Qwen3.5-4B (MTP)²     | Q4_K_M      | 0.98×      | 0.93×     | 0.95×      | **1.64×** |
| Qwen3.5-4B (MTP)²     | UD-Q4_K_XL  | **1.03×**  | 0.94×     | 0.95×      | **1.51×** |
| Gemma-4-E2B           | Q4_K_M      | **1.14×**  | **1.06×** | 0.98×      | **1.01×** |
| Qwen3-8B              | Q4_K_M      | **1.29×**  | 0.94×     | 0.93×      | **1.40×** |
| Ornith-1.0-9B         | Q4_K_M      | **1.18×**  | 0.95×     | 0.96×      | **1.61×** |
| Qwen3.5-9B            | Q4_K_M      | **1.17×**  | 0.96×     | 0.97×      | **1.54×** |
| Qwen3.5-9B (MTP)²     | Q4_K_M      | **1.19×**  | 0.92×     | 0.93×      | **1.55×** |
| Qwen3.5-9B (MTP)²     | UD-Q4_K_XL  | **1.16×**  | 0.93×     | 0.94×      | **1.44×** |
| Gemma-3-12B           | Q4_K_M      | **1.25×**  | 1.00×     | **1.02×**  | **1.77×** |
| Gemma-4-12B           | Q4_K_M      | **1.27×**  | 1.00×     | 0.99×      | **1.73×** |
| Qwen3-14B             | Q2_K³       | **1.22×**  | 0.81×     | 0.78×      | **1.17×** |
| Qwen3-14B             | Q4_K_M      | **1.12×**  | 0.92×     | 0.87×      | **1.30×** |
| Qwen3-14B             | Q8_0        | **1.14×**  | **1.02×** | 0.97×      | 0.93×     |
| Gemma-4-26B-A4B (MoE) | UD-Q4_K_M   | **1.06×**  | **1.03×** | **1.04×**  | **1.52×** |
| Qwen3.6-27B           | Q4_K_M      | **1.09×**  | 0.94×     | 0.93×      | **1.21×** |
| Qwen3-30B-A3B (MoE)   | Q4_K_M      | 0.95×      | 0.94×     | 0.91×      | **1.17×** |
| Gemma-4-31B           | UD-Q5_K_XL⁴ | 0.98×      | 0.88×     | 0.90×      | 0.84×     |
| Ornith-1.0-35B        | Q4_K_M⁵     | 0.89×      | **1.02×** | **1.02×**  | **1.55×** |
| Qwen3.6-35B-A3B (MoE) | UD-IQ3_S⁶   | 0.90×      | 0.89×     | 0.90×      | **1.08×** |
| Qwen3.6-35B-A3B (MoE) | UD-Q4_K_M   | **1.02×**  | 0.98×     | 0.99×      | **1.51×** |

¹ The Q6_K `pp512` cell is a re-measure. The sweep run read 0.98× and an
immediate re-run of the same binary read 1.23× — infr's default prefill is
run-to-run nondeterministic in its tier/chunk choice (a known open issue), and
this small-model row has the widest spread of any in the table. Treat ±5% on
`pp512` as noise everywhere; this row's spread is larger than that.

² The MTP repos ship a `mtp-*.gguf` draft head. Their `mtp128`
speculative-decode ratio is **0.76–0.78× (4B) / 0.61–0.70× (9B)** — infr's one
consistent loss, see the prose below. Plain (non-MTP) metrics for the same
weights are the rows shown.

An optimization pass (`a9b5cae`) removed the wasted vocab-wide `lm_head` compute
on the reprime re-sync pass (it computed up to `MTP_REPRIME_MAX_M`−1 rows and
discarded all but the last); token-identity is preserved, but the gain does not
survive independent re-measurement against a fresh llama.cpp baseline, so the
ratios above are unchanged. **The verify pass (57–60% of MTP wall time) is still
the whole gap**, and the reason it is stuck is worth recording: the obvious
lever — an int8-quantized-activation `mrow` GEMV tier for the small-m verify
batch — was built for **Q5_K** and **broke token-identity**, because infr's
_plain_ decode still runs Q5_K through an f32-exact dequant GEMV. That asymmetry
between the spec and non-spec streams flips the occasional greedy token.
Q4_K/Q6_K don't have the problem: both streams already use int8 there. The fix
is therefore not an MTP fix at all — it is to make the int8-activation tier
**symmetric** across plain decode and verify (footnote ³), which is also what
closes the mid/large decode gap.

These four rows' `tg64@d4096` cells were a GPU device-lost in the raw sweep and
are re-measured post-`8513358`: 35821b6's capacity gate on the `nonfa` coopmat
prefill tier (which reads K in whole 256-row tiles, so it touches
`ceil(kv_len/256)*256` rows) had no catcher for a **non-SWA** model — `split_ok`
only covered the SWA `ring_past` case — so the op fell through to the scalar
`attention_kv` at 3591 rows × 3591 kv and hung the GPU. MTP's un-chunked
whole-prompt verify is the only shape that reliably lands `kv_len` within one
tile-pad of the cache's row capacity.

³ **Q2_K now decodes on the int8-activation tier by default on AMD**
(`43806da`): tg128 0.74× → **0.81×**, tg64@d4096 0.72× → **0.78×**, and
`pp4@d4096` 0.98× → **1.17×** (a loss turned into a win). This is the same trade
llama.cpp takes — their `ggml_vk_should_use_mmvq` returns true for every quant
type on AMD at `k >= 2048`, carving out only Q6_K (2-byte alignment, an
Intel-only win) and Q8_0 (GCN only) — so matching it is parity, not a quality
regression. infr's tier is **per-(dtype, vendor) with every entry measured on
infr's own kernels** rather than inherited from llama.cpp's table (the two
engines' kernels have different overheads, so a win on one does not imply a win
on the other). It is also **symmetric**: any dtype on the int8 decode tier takes
int8 in the MTP verify batch too — `mmv_int8_decode_dtypes` is the single source
of truth both gates read, and a unit guard asserts they agree across every
vendor × env combination. That symmetry is exactly the constraint footnote ²'s
Q5_K attempt violated.

**Q4_K is deliberately NOT on the tier on AMD** — the throughput half of this is
now FIXED, the correctness half is not, so it stays off by default. What was
fixed: the old **−9.6%** (14B Q4_K_M tg64 78.5 → 71.0 t/s) is root-caused and
closed. (1) infr's m=1 kernel unpacked Q4_K **byte-at-a-time**, where llama.cpp
(and infr's own `mrow` kernel) read **aligned u32s** and nibble-mask 4 weights
at a time; switching `native_mmv_mw.comp`'s Q4_K `dpsub` to the same
word-parallel load measured **+5.8%, bit-identical output** (`mmv_mw_parity`).
(2) A dispatch-shape sweep over `INFR_MMV_MW_WARPS` (rows/block) — llama.cpp
runs Q4_K mmvq on AMD non-GCN at `rm_kq_int=1` (one output row per workgroup,
single-subgroup reduce), infr had only ever tried {4, 8} warps/block — found
WARPS=1 the clear winner over the full {1, 2, 4, 8, 16} sweep: 14B Q4_K_M tg64
78.3 → **80.1 t/s (+2.3%)**, tg128 78.0 → 79.8, tg64@d4096 68.2 → 69.3. Int8
Q4_K now genuinely beats infr's own f32 path.

It still isn't shipped as the AMD default, because flipping it breaks
`mtp_spec_matches_target_only_greedy`: infr's `mrow` kernel (m≥3 verify batch)
has taken Q4_K int8 **unconditionally** since before this policy table existed
(see the wart noted below), so turning on the m=1 decode tier makes BOTH streams
int8 for the first time — and they still disagree on the occasional greedy
token. The two kernels are different code (warp-per-row `subgroupAdd` vs
row-tile accumulation); both quantize activations identically and dot the same
integers per sub-block, but the cross-sub-block **summation order** differs, the
same reassociation class `mmv_mw_parity` already tolerates at 5e-3 for
throughput purposes — apparently wide enough here to flip an argmax across
streams often enough to fail in 64 tokens. This is not a regression introduced
by the fixes above: it reproduces on the pre-fix code too via the pre-existing
`INFR_MMV_MW=1` escape, which nobody had run against the MTP symmetry test
before. The lesson: **"both streams int8" is necessary but not sufficient for
token-identity** — it also needs the two kernels bit-identical at the same
position, and `mmv_mw`/`mrow` aren't. Making them so (e.g. porting `mmv_mw` to
`mrow`'s row-tile accumulation, or vice versa) is a real kernel project, left
open. Until then the fix + WARPS=1 stay reachable via
`INFR_MMV_MW=1 INFR_MMV_MW_WARPS=1` for A/B measurement on non-MTP workloads,
where the win is real (14B Q4_K_M: tg128 0.92× → 0.94×, tg64@d4096 0.87× →
0.89×, both still short of llama.cpp parity but a real step, not a forced one).
Q6_K/Q3_K stay off on AMD as **unmeasured** (no suitable model in the validated
cache) — left off rather than assumed.

Known wart, recorded rather than hidden: Q4_K/Q6_K/IQ4_XS take the int8 `mrow`
kernel at m≥3 **unconditionally**, while their m=1 decode stays f32-exact on AMD
— the same decode/verify asymmetry described above, pre-dating the policy table
and live on every Q4_K MTP run today. It is untouched here (it is load-bearing
for the Gemma-4-E2B `pp4@d4096` numbers) and queued as its own fix. **Turns out
"fixing" it by making m=1 int8 too is not actually a fix** — see footnote ³:
`mmv_mw` and `mrow` are different kernels that don't agree bit-for-bit at the
same position, so making both streams int8 for Q4_K trades a known asymmetry for
a different, still-live disagreement (`mtp_spec_matches_target_only_greedy`
fails either way). The real fix needs the two kernels bit-identical, not just
same-precision.

⁴ Gemma-4-31B (21.9 GiB weights on the 24 GB card) runs **fully resident,
including at depth**, after two placement slices: try-resident-first dense
placement (`e2c0694` — honest activation reserve + a phantom +1.6 GiB accounting
fix) and **window-sized ring KV for sliding-window layers** (`35821b6`,
llama.cpp-parity: 50 of its 60 layers are SWA with a 1024 window, so their
caches are 2048-row rings instead of full-context — @8k that's 0.44 GiB instead
of 5.5). The d4096 row went 0.08× → 0.90× (28 vs 31 t/s). The same slice also
reuses empty KV slots instead of forking a duplicate (`f74556c` — was silently
wasting a full KV per session, 6.25 GiB on a 14B), and lifted the gemma-family
multi-turn rows (12B `pp4@d4096` 1.40× → 1.66×: less dead KV to re-scan).

⁵ Ornith-35B's `pp512` 0.89× is the DeltaNet **scan** kernel: 4.6× slower than
llama.cpp's fused GDN (31.3 vs 6.8 ms per 512 tokens), plus `expert_down` at ~13
ms against llama.cpp's `mmid` row packing. A BN=128 wide-N expert tile
(`50059c9`) already lifted it from 0.83×; the rest is a kernel project.

⁶ Grid i-quant (IQ1–IQ3) row: the grid-perf slice closed both structural gaps
`618cd3b` left behind (that commit fixed the device-lost TDR — dynamically
indexed GLSL `const` codebook tables lowered to ~1 MB of per-invocation scratch
by RADV/ACO — by staging the grids through `shared` memory): a grid-aware
`dqblk` amortizes the per-32-group scale/sign/qh decode and grid gathers that
the per-element `dq()` re-derived (decode 0.50× → 0.89×, tg128 75 → 134 t/s),
and IQ2_S/IQ3_S — this file's expert pair — got batched dp4a mmq expert GEMMs
(shared-LUT grid staging feeds the int8 dot; prefill 0.03× → 0.91×, pp512 75 →
2575 t/s). The other five grid formats keep the id-GEMV prefill fallback (no
shipped MoE GGUF uses them for expert banks — see `MOE_MMQ_DTYPES`'s exclusions
doc).

The MoE expert kernel floor (the id-indexed GEMV family every MoE model needs
for decode) now covers **every weight dtype the dense Vulkan path supports** —
all quants (Q\* incl. ternary Q2_0, K-quants, IQ\*, TQ\*, MXFP4/NVFP4, BF16)
plus F16/F32 float banks — so no expert-bank quant is rejected at load. On top
of that, the batched-MoE dp4a mmq prefill family covers Q4_0 / Q4_1 / Q5_0 /
Q5_1 / Q8_0 / Q2_0 / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / IQ4_NL / IQ4_XS / MXFP4
/ NVFP4 (`infr_core::tensor::MOE_MMQ_DTYPES` is the single source of truth both
the graph-build and adapter gates derive from; `moe_mmq_drift_test` guards the
kernel tables against drift, and its doc records the deliberate exclusions: grid
i-quants (IQ1–IQ3), ternary (TQ\*), and float banks prefill via the per-token
id-GEMV path).

**Where infr wins.** Prefill on **every dense family** at the mainstream quants
(1.04–1.44× at Q4_K_M/Q8_0), and it now leads the gemma-4 MoE too (1.06×).
Multi-turn ingest (`pp4@d4096`) wins on **32 of 35 rows** (1.01–2.17×) — this is
the shape a coding agent actually runs, and it is where infr is furthest ahead.
Decode is at-or-above parity up to ~1.7B, on the gemma-4 MoE (1.03×), and on the
35B-class DeltaNet models (Ornith-35B 1.02×).

**Where infr loses.** Four places, all understood:

- **MTP speculative decode** (`mtp128`, 0.60–0.78×) — the only consistent,
  material loss. Both engines decode the same un-templated content (α is
  content-sensitive, so cross-engine spec ratios are meaningless otherwise);
  llama.cpp's spec self-speedup simply survives low accept rates better because
  its batched verify amortizes further. Levers: wide-n small-m GEMM efficiency,
  and verifying the lm_head only over the rows whose logits are kept.
- **Mid/large dense + Qwen MoE decode** (8B–31B at 0.87–0.97×, Qwen3-30B/35B MoE
  at 0.91–0.99×) — these are all **Q4_K_M** rows. Footnote ³ named the cause
  (infr's int8 Q4_K GEMV kernel was slower than its own f32 path) and the kernel
  is now fixed — int8 Q4_K measures a real win in isolation — but it stays
  unshipped because it breaks MTP token-identity against the pre-existing
  always-int8 `mrow` verify tier (footnote ³ again), so these rows are
  unchanged: still on the f32 path, still behind llama.cpp's `q4_k × q8_1`. This
  was previously written off as the memory-bandwidth wall — decode GEMVs do run
  at 77–88% of DRAM peak — but that figure was measured on the f32-activation
  kernels, and llama.cpp beating us by 35% on 14B Q2_K decode proves those rows
  are partly **ALU-bound**, with more headroom than the wall story implied.
  Correct full-expert routing separately costs the Qwen MoEs some prefill batch
  efficiency (0.95× on Qwen3-30B).
- **Qwen3-14B Q2_K decode** (0.78–0.81×, was 0.72–0.74×) — improved by the
  int8-activation tier (footnote ³); the residual is infr's Q4_K/Q2_K GEMV
  kernel shape, not the precision policy.
- **Ornith-35B prefill** (0.89×) and **the IQ3_S MoE** (0.90×) — the DeltaNet
  scan kernel (footnote ⁵) and the grid i-quant path (footnote ⁶), both known
  kernel projects.

Two isolated rows also sit just under parity with no story beyond noise:
Llama-3.2-1B Q4_K_M (0.96–0.99× across the board) and Gemma-4-31B, whose
`pp4@d4096` 0.84× is the deepest multi-turn row in the table.

**DiffusionGemma** (`dg-step`) beats the reference fork at 1.18× (52.5 vs 46.5
t/s e2e at matched-ish 24/23 steps).

**Ternary-Bonsai (Q2_0) — infr is the only engine that runs these on a GPU.**
llama.cpp merged the **Q2_0** weight dtype (GGML type 42) but shipped **no GPU
kernels for it**: there is not a single `q2_0` reference in its `ggml-vulkan/`
or `ggml-cuda/` trees, so every backend but CPU refuses to load these files.
infr runs Q2_0 natively on Vulkan (in-shader dequant + dp4a mmq — `ad89fb4`), so
the comparison below is **absolute throughput on different devices, not a
like-for- like ratio**: infr on the RX 7900 XTX vs llama.cpp on a Ryzen 9
9950X3D (16 threads, Release + `GGML_NATIVE`). r=3, 2026-07-12.

| Model (Prism ML) | Size    | infr pp512 | infr tg128 | llama.cpp pp512 | llama.cpp tg128 |
| ---------------- | ------- | ---------- | ---------- | --------------- | --------------- |
| Bonsai-1.7B      | 462 MiB | **6365**   | **594**    | 108.7 (CPU)     | 78.3 (CPU)      |
| Bonsai-4B        | 1.05 GB | **2756**   | **303**    | 41.9 (CPU)      | 33.9 (CPU)      |
| Bonsai-8B        | 2.15 GB | **1647**   | **212**    | 22.1 (CPU)      | 18.6 (CPU)      |

Use the **`Q2_0_g64`** files — despite the name they are the layout upstream
merged (64-elem / 18 B blocks, 2.25 bpw). The repos' plain `*-Q2_0.gguf` /
`*-PQ2_0.gguf` uploads predate the merge and use 128-elem / 34 B blocks (2.125
bpw); llama.cpp master rejects them too. Same scheme otherwise — one f16 scale
per 128 weights instead of per 64 — so they could be supported by a lossless
load-time repack if the format sticks around.

```bash
infr run prism-ml/Ternary-Bonsai-8B-gguf:Q2_0_g64 "What is the capital of France?"
```

**Llama-4-Scout** (109B-A17B, Q2_K, 37 GB) is deliberately absent from the table
above (its per-token small-m dispatch shape isn't comparable to the batched
pp/tg columns) but runs end to end on a 24 GB card via the paged expert cache
(`infr_vulkan::pager`). Prefill runs the batched bucket-scatter dp4a mmq
expert-GEMM pipeline against the pager arena with NO host round-trip at all:
each layer pre-stages its full expert set through a pipelined staging ring
(recorded ring→arena copies, fenced half rotation — CPU expert memcpys overlap
GPU execution) under a scan-resistant eviction policy, and every paged dispatch
reads a frozen per-layer LUT window from a tape instead of a live LUT. Decode
keeps the id-indexed small-m GEMV with at most ONE mapped-readback sync per
non-resident layer (fully-resident layers record straight through). Greedy
output is oracle-locked against llama.cpp (`cpu_llama4_scout_greedy`) AND
against the paged Vulkan path itself
(`gpu_seam_paged_moe_matches_scout_oracle`), token-for-token identical. Measured
(all 48 expert layers paged, per-role LRU caches of 312/312/238 experts — each
role's arena is one SSBO, capped at the device's 4 GiB binding range): `pp512`
**404 t/s** warm (r=3; pre-rework host-orchestration baseline: 189; llama.cpp's
CPU-offload hybrid: 136 — and past the ~363 t/s-equivalent GPU-busy ceiling the
old per-layer submit→readback→upload cadence measured, since staging now
overlaps compute), warm decode `tg64@d128` **~17 t/s** (baseline 14.2; llama.cpp
hybrid: 6.55 — decode stays upload-bound: a 24 GB budget can't hold the ~37 GB
decode working set, so ~350 MB/token still pages in). `INFR_CACHE` sizes the
pager's budget (see the MoE placement paragraph above); `INFR_PAGER_RING`
overrides the staging-ring size (default: budget/8 clamped to [256 MiB, 2 GiB]);
pure CPU stays available under `INFR_CPU=1` / `-ngl 0`. Remaining follow-up:
splitting a role across several arena buffers to lift the 4 GiB per-role cache
cap.

**Also validated for correctness** (GPU seam vs CPU reference), beyond the perf
table: Qwen2-0.5B, Llama-3.2-1B, Gemma-4-12B (dense), and Qwen3-0.6B across
quant formats **Q4_K_M / Q5_K_M / Q6_K / Q4_0 / Q2_K / IQ4_XS / Q8_0 / BF16**
(each decoded on-device via hand-written SPIR-V, byte-identical to the CPU
dequant).

> Numbers are a snapshot and move with each perf slice; regenerate on your own
> hardware with `infr compare --sweep <model...>`. Results on other GPUs
> (NVIDIA, Intel Arc) and Apple Metal are wanted — please open an issue with
> your `infr bench` / `infr compare` output. Intel Arc testers: include one run
> with `INFR_DEBUG_COOPMAT=1` (the enumerated/chosen coopmat shapes), then A/B
> `INFR_CM_8X8=1` (opt-in 8x8x16 XMX prefill GEMM) against the default.

## Scope

- **Format:** GGUF
- **Models:** Llama, Qwen2/2.5, Qwen3 (dense + MoE), Gemma 3, Gemma 4 (dense +
  E2B + 26B-A4B MoE), Qwen3.5/3.6 (dense + MoE) — all on GPU **and** the CPU
  reference; DiffusionGemma (block text-diffusion, CPU + GPU); Llama 4 (Scout —
  GPU by default via the paged expert cache, 37 GB Q2_K on a 24 GB card; pure
  CPU under `INFR_CPU=1`)
- **GPU:** AMD / NVIDIA / Intel via Vulkan (cooperative-matrix matmul); Apple
  via a native **Metal backend** (`INFR_METAL=1`) covering every op the CPU
  reference does — dense, MoE (`qwen3moe`) and Qwen3.5 (`qwen35`). Dense is
  optimized (simdgroup-matrix GEMM + flash attention, raw-block quant decode;
  within ~1.3-1.5× of llama.cpp Metal on M3 Pro — architecture and numbers in
  [`docs/METAL.md`](docs/METAL.md))
- **Store:** the shared **HuggingFace Hub cache** — located via `$HF_HUB_CACHE`,
  else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`, in HF's own
  `models--<org>--<repo>/{blobs,snapshots,refs}` layout. A model pulled by
  `infr`, `llama.cpp`, or `huggingface_hub` is shared — downloaded once.
  `infr pull` fetches from `huggingface.co` over resumable HTTP Range with a
  progress bar; gated repos authenticate with `HF_TOKEN`.
- **API:** OpenAI-compatible HTTP (streaming) — works with opencode / Claude
  Code CLI

## Architecture

```
server   axum + SSE  ->  OpenAI /v1
chat     ChatModel        (autoregressive dense/MoE/qwen35; DiffusionGemma's block-diffusion loop)
runtime  SeamModel        tensors, KV cache, command/descriptor management (the unified runner)
loader   WeightSource     (Gguf; safetensors later)
compute  Backend          (Vulkan via ash + SPIR-V; reference Metal via MSL; CUDA later)
```

## License

[MIT](LICENSE)
