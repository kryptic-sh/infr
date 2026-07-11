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
faster**); r=3, 2026-07-11 snapshot (commit `a3e1e9a`, every model×quant in the
local cache). Hardware: **AMD Radeon RX 7900 XTX** (RDNA3, 24 GB, Vulkan / RADV,
Mesa). `pp512` = 512-token prefill throughput, `tg128` = 128-token decode
throughput, `tg64@d4096` = decode at 4096 KV depth, `pp4@d4096` = short-turn
prefill at 4096 KV depth (the multi-turn serve shape).

| Model                 | Quant       | pp512     | tg128     | tg64@d4096 | pp4@d4096 |
| --------------------- | ----------- | --------- | --------- | ---------- | --------- |
| Qwen3-0.6B            | Q2_K        | **1.28×** | **1.33×** | **1.34×**  | **2.29×** |
| Qwen3-0.6B            | IQ4_XS      | **1.15×** | **1.15×** | **1.22×**  | **2.01×** |
| Qwen3-0.6B            | Q4_0        | **1.14×** | **1.32×** | **1.32×**  | **2.16×** |
| Qwen3-0.6B            | Q4_K_M      | **1.21×** | **1.20×** | **1.25×**  | **2.17×** |
| Qwen3-0.6B            | Q5_K_M      | **1.16×** | **1.20×** | **1.24×**  | **2.17×** |
| Qwen3-0.6B            | Q6_K        | **1.24×** | **1.10×** | **1.16×**  | **1.95×** |
| Qwen3-0.6B            | Q8_0        | **1.12×** | **1.06×** | **1.13×**  | **2.12×** |
| Qwen3-0.6B            | BF16        | **1.09×** | 0.88×     | 0.94×      | **1.81×** |
| Qwen3.5-0.8B          | Q4_K_M      | **1.01×** | **1.10×** | **1.06×**  | **1.78×** |
| Gemma-3-1B            | Q2_K¹       | **1.21×** | **1.11×** | **1.04×**  | 0.83×     |
| Gemma-3-1B            | Q4_K_M      | **1.06×** | **1.13×** | **1.05×**  | **1.08×** |
| Gemma-3-1B            | Q8_0        | **1.18×** | **1.07×** | 1.00×      | **1.08×** |
| Llama-3.2-1B          | Q4_K_M      | 1.00×     | 0.96×     | 0.89×      | **1.04×** |
| Llama-3.2-1B          | Q8_0        | 0.88×     | 0.88×     | 0.82×      | **1.02×** |
| Qwen3-1.7B            | Q4_K_M      | **1.12×** | **1.09×** | **1.14×**  | **1.86×** |
| Qwen3.5-4B (MTP)²     | Q4_K_M      | **1.02×** | 0.93×     | 0.95×      | **1.17×** |
| Qwen3.5-4B (MTP)²     | UD-Q4_K_XL  | **1.01×** | 0.93×     | 0.95×      | **1.34×** |
| Gemma-4-E2B           | Q4_K_M      | **1.13×** | **1.07×** | 1.00×      | **1.08×** |
| Qwen3-8B              | Q4_K_M      | **1.28×** | 0.94×     | 0.95×      | **1.32×** |
| Ornith-1.0-9B         | Q4_K_M      | **1.17×** | 0.96×     | 0.97×      | **1.41×** |
| Qwen3.5-9B            | Q4_K_M      | **1.16×** | 0.96×     | 0.97×      | **1.32×** |
| Qwen3.5-9B (MTP)²     | Q4_K_M      | **1.18×** | 0.93×     | 0.93×      | **1.39×** |
| Qwen3.5-9B (MTP)²     | UD-Q4_K_XL  | **1.15×** | 0.93×     | 0.93×      | **1.22×** |
| Gemma-3-12B           | Q4_K_M      | **1.25×** | 1.00×     | **1.03×**  | **1.39×** |
| Gemma-4-12B           | Q4_K_M      | **1.27×** | **1.01×** | 1.00×      | **1.40×** |
| Qwen3-14B             | Q2_K¹       | **1.21×** | 0.74×     | 0.73×      | 0.96×     |
| Qwen3-14B             | Q4_K_M      | **1.12×** | 0.92×     | 0.88×      | **1.21×** |
| Qwen3-14B             | Q8_0¹       | **1.14×** | **1.02×** | 0.97×      | 0.93×     |
| Gemma-4-26B-A4B (MoE) | UD-Q4_K_M   | 0.99×     | 0.92×     | 0.95×      | **1.16×** |
| Qwen3.6-27B           | Q4_K_M      | **1.09×** | 0.94×     | 0.93×      | **1.14×** |
| Qwen3-30B-A3B (MoE)   | Q4_K_M      | 0.96×     | 0.95×     | 0.93×      | **1.14×** |
| Gemma-4-31B           | UD-Q5_K_XL³ | **0.98×** | 0.89×     | 0.08×      | 0.13×     |
| Ornith-1.0-35B        | Q4_K_M¹     | 0.89×     | **1.01×** | **1.03×**  | **1.48×** |
| Qwen3.6-35B-A3B (MoE) | UD-IQ3_S⁴   | 0.03×     | 0.50×     | 0.52×      | 0.35×     |
| Qwen3.6-35B-A3B (MoE) | UD-Q4_K_M   | **1.03×** | 0.98×     | 0.99×      | **1.53×** |

¹ Rows re-measured 2026-07-12 after the quant-extreme perf slice
(`e55d744..50059c9`): gemma-3-1b "Q2_K" (a mixed quant whose ffn/attn_q banks
are actually **IQ4_NL** — the last 4-bit codebook format with no warp-GEMM
family) went 0.37× → 1.21× pp512; Qwen3-14B **Q8_0** decode went 0.75× → 1.02×
(word-parallel dqblk, ~600 → ~850 GB/s); Qwen3-14B **Q2_K** decode improved
0.68× → 0.74× (packed bit-plane extraction) — the rest of that gap is
llama.cpp's dp4a **mmvq** int8-activation tier, a precision trade infr
deliberately doesn't take by default; Ornith-35B pp512 improved 0.83× → 0.89×
(BN=128 expert tile, also +4.5% on Qwen3.6-35B) — the residual is the DeltaNet
scan kernel (4.6× slower than llama.cpp's fused GDN, queued). Other Q8_0/Q2_K
rows in the table predate these fixes and can only have improved.

² The MTP repos ship a `mtp-*.gguf` draft head; their `mtp128`
speculative-decode ratio is 0.63–0.68× (4B) / 0.52–0.55× (9B) — see the MTP
paragraph below. Plain (non-MTP) metrics for the same weights are the rows
shown.

³ Gemma-4-31B (21.9 GiB weights on the 24 GB card) runs **resident at default
context** since `e2c0694` (try-resident-first dense placement with an honest
activation reserve — the old MoE-sized 2 GiB headroom plus a phantom +1.6 GiB in
the tied-lm-head accounting used to push it into streaming): pp512 0.98×, tg128
0.89×. The `@d4096` columns still **stream**: infr sizes every layer's KV at
full context while llama.cpp sizes sliding-window layers to the SWA window, and
full-ctx KV (4.3 GB) doesn't fit beside the weights. Window-sized SWA KV is
queued — it makes d4096 resident too.

⁴ Grid i-quant (IQ1–IQ3) expert banks are **correct but on the slow floor** (row
measured post-`618cd3b`, which fixed a device-lost TDR: dynamically indexed GLSL
`const` codebook tables were lowered to ~1 MB of per-invocation scratch by
RADV/ACO — they now stage through `shared` memory, ~400× faster). Remaining gaps
are structural and queued: grid formats have no batched dp4a mmq GEMM, so MoE
**prefill** rides the per-token id-GEMV floor (0.03×), and the grid decode still
pays a per-element `dq()` (a grid-aware `dqblk` is the decode lever, ~0.50×
today).

The MoE expert kernel floor (the id-indexed GEMV family every MoE model needs
for decode) now covers **every weight dtype the dense Vulkan path supports** —
all quants (Q\*, K-quants, IQ\*, TQ\*, MXFP4/NVFP4, BF16) plus F16/F32 float
banks — so no expert-bank quant is rejected at load. On top of that, the
batched-MoE dp4a mmq prefill family covers Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 /
Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / IQ4_NL / IQ4_XS / MXFP4 / NVFP4
(`infr_core::tensor::MOE_MMQ_DTYPES` is the single source of truth both the
graph-build and adapter gates derive from; `moe_mmq_drift_test` guards the
kernel tables against drift, and its doc records the deliberate exclusions: grid
i-quants (IQ1–IQ3), ternary (TQ\*), and float banks prefill via the per-token
id-GEMV path).

infr **wins prefill on nearly every dense model at the mainstream quants**
(1.01–1.28× at Q4_K_M across every family) and is at parity on the gemma-4 MoE
(0.99×); the two Qwen MoEs prefill at 0.94–0.96× — correct full-expert routing
costs some batch efficiency vs llama.cpp. Multi-turn ingest (`pp4@d4096`) **wins
on every working row but two** (1.02–2.29×; the exceptions are the footnoted
Q2_K/Q8_0 gap rows). Decode is at-or-above parity on models up to ~4B and on the
35B-class DeltaNet models (Ornith-35B 1.01–1.03×), slightly behind on mid/large
dense (8B–27B at 0.88–0.96×) and MoE (0.92–0.96×), all bounded by the
memory-bandwidth wall (decode GEMVs run at 77–88% of DRAM peak). The quant
spread is new in this sweep: Q4/Q5/Q6/IQ4_XS rows track the Q4_K_M story
everywhere, while the Q2_K/Q8_0 extremes expose the footnote-¹ gaps now queued.
The **MTP speculative path** trails at 0.52–0.68× (`mtp128`) — drafted-token
throughput not yet matching llama.cpp's batched-speculative path (verify-kernel
efficiency at m≈6–8 is the known lever). **DiffusionGemma** (`dg-step`) beats
the reference fork at 1.20× (53.1 vs 46.3 t/s e2e at matched-ish 24/23 steps).

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
> your `infr bench` / `infr compare` output.

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
