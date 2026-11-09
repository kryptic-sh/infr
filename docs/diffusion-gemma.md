# DiffusionGemma (`diffusion-gemma`) — design for the unified seam

## Status

Shipped on CPU and Vulkan, on the same unified transformer runner every other
arch uses (`SeamModel` + the per-backend `ChatModel` impls): `infr run`
(entropy-bound block-diffusion decode, multi-turn REPL), `infr serve` (streaming
and non-streaming, `reasoning_content` split on the
`<|channel>thought…<channel|>` markers), `infr bench` (DG-aware: pp tok/s,
generated tok/s, EB steps, in-step-parallel rate) and `infr compare` against the
reference fork's `llama-diffusion-cli` (see "Comparing against the fork
oracle"). The canvas-denoise graph is compiled once per shape and reused
(`DenoiseCache` in `seam/sc.rs`), self-conditioning runs in-graph (`gpu_sc`),
and `dg-step` beats the fork — current figure in `perf/results.md`.

Metal (Phase D) is built and its kernels are exercised by the macOS CI job's
`infr-metal` parity suite, but no real DG model has run on an Apple GPU, and
DG's fused MoE shape still takes Metal's host path. Both are open in
`backlog.md` § B100.

Block text-diffusion MoE on a **Gemma-4 backbone**. Reference implementation:
llama.cpp PR #24423 (checked out at `~/Projects/mxaddict/llama.cpp-dg`,
`src/models/diffusion-gemma.cpp` + `gemma4-common.h`); oracle binary
`build/bin/llama-diffusion-cli` (CPU build). Model:
`unsloth/diffusiongemma-26B-A4B-it-GGUF` (Q4_K_M ≈ 16 GB, fits the 7900 XTX).

This REPLACED the pre-seam scaffolding (`infr-model`/`infr-decode`/the
`infr-engine::Engine` stub, all dead code) — DiffusionGemma is built on the
unified transformer runner like every other arch; the scaffolding was deleted in
the unified runner's phase 4.

## Confirmed metadata (real GGUF dump, 2026-07-05)

- `general.architecture = diffusion-gemma`, 30 layers, n_embd=2816,
  vocab=262144, tied lm_head, `final_logit_softcapping=30`, rms_eps=1e-6,
  **attention scale = 1.0** (NO 1/√d — same as gemma4).
- SWA pattern 5:1 (`sliding_window_pattern` bool array; full-attn at layers
  5,11,17,23,29), window 1024.
- **Heterogeneous attn dims**: SWA layers hd=256, 16q/8kv, rope base 1e4, no
  freq_factors; FULL layers hd=512, 16q/2kv, rope base 1e6, `rope_freqs[256]`
  proportional-rope freq_factors. `attention.causal = False` in metadata (the
  forward builds its own region mask).
- **Global layers have NO `attn_v`** (missing at exactly 5,11,17,23,29): V = the
  raw k_proj output (pre-k-norm), reshaped, then the weightless V rms-norm.
- FFN per layer = **dense MLP ∥ 128-expert MoE, summed** (see wiring below):
  dense gate/up/down at n_ff=2112 (GELU-par), experts fused
  `ffn_gate_up_exps[2816,1408,128]` + `ffn_down_exps[704,2816,128]` with a
  per-expert scale `ffn_down_exps.scale[128]`, router `ffn_gate_inp[2816,128]` +
  elementwise input scale `ffn_gate_inp.scale[2816]`, 8 experts used, softmax
  gating.
- Per-layer scalars are `[1]` TENSORS: `blk.N.enc_layer_output_scale.weight`
  (encoder/prompt) and `blk.N.layer_output_scale.weight` (decoder/canvas).
- Top-level self-conditioning MLP: `self_cond_pre_norm[2816]`,
  `self_cond_gate/up [2816,2112]`, `self_cond_down [2112,2816]`.
- `diffusion.canvas_length = 256`, `tokenizer.ggml.mask_token_id = 4`.

## The architecture, per the PR source (verified)

- **Backbone identical to gemma4** (shared `gemma4-common.h`): SWA pattern,
  heterogeneous per-layer dims, sandwich norms, GeGLU/GELU, per-layer output
  scalars — machinery infr's gemma4 support already has, EXCEPT the dual FFN and
  the V=K reuse (new).
- **[prompt | canvas] regions**, split at `P = n_tokens − canvas_length`:
  1. input embeddings — prompt: `embed·√n_embd` (standard gemma); canvas:
     `rmsnorm_noscale(embed·√n_embd)` (weightless RMSNorm — infr has it as
     QkNorm-with-ones, the gemma4 V-norm trick);
  2. per-layer output scalar — prompt rows use the ENCODER scalar, canvas rows
     the DECODER scalar (two per-layer arrays in the GGUF);
  3. attention mask — prompt queries: causal over prompt only (SWA-clipped),
     NEVER see canvas; canvas queries: bidirectional over everything — global
     layers see all prompt + canvas, sliding layers the last `n_swa−1` prompt
     positions + all canvas.
- **Two-pass equivalence** (the key implementation fact, stated by the PR): a
  causal encoder prefill of the prompt + a bidirectional decoder denoise of the
  canvas (zero self-conditioning) reproduces the single no-cache forward. infr
  implements the two-pass form natively:
  - **Pass 1 (per block)**: standard causal prompt prefill through the existing
    gemma4 path, encoder scalars — KV cache rows `0..P` written once.
  - **Pass 2 (per denoise step)**: forward ONLY the C canvas rows with decoder
    scalars; `WriteKv` overwrites cache rows `P..P+C` each step;
    `Attention { rows: C, kv_len: P+C, mask: Canvas{..} }` — the one new seam
    piece.
  - **Self-conditioning is ON by default for canvas models**
    (`diffusion-cli.cpp:273 diff_params.self_conditioning = true`; the cli
    enables it before context creation). Step 0 gates it to zero (`sc_use=0`,
    reproducing the zero-SC exactness forward); every later step feeds the
    PREVIOUS step's raw canvas logits:
    `soft = softmax(logits·temp_inv) @ embdᵀ · √n_embd`, then `sc_pre_norm` →
    GELU-gated MLP (gate/up/down, n_ff=2112, `gelu(gate)·up`) →
    `canvas += sc_sig` → the weightless rms-norm. Oracle parity REQUIRES this
    block (Phase 2/3). llama.cpp pre-dequantizes/transposes `tok_embd` to f16
    `[n_vocab, n_embd]` once for the soft-embed matmul.
- **Block diffusion decode**: canvas of `diffusion.canvas_length` (256) mask
  tokens (`tokenizer.ggml.mask_token_id`); N denoise steps unmask progressively
  (entropy-bound sampler from the PR's `common/sampling`); a finished block
  commits (its rows become prompt: already in the KV cache — the next block's P
  grows by C) and the loop advances. `-n` derives the block count:
  `blocks = ceil(n_predict / canvas_length)`.

## FFN block wiring (exact, from `gemma4_build_ffn_moe`)

On the post-attention residual `attn_out` (already includes the attn residual
add):

```
dense = post_ffw_norm_1( gelu_par_ffn( ffn_norm(attn_out) ) )         # gate/up/down 2112, GELU
router_in  = rmsnorm_noscale(attn_out) · (1/√n_embd) · ffn_gate_inp_s  # UNNORMED residual!
logits     = ffn_gate_inp @ router_in                                  # [128]
moe_in     = pre_ffw_norm_2(attn_out)
moe = post_ffw_norm_2( moe_ffn(moe_in, softmax-top8(logits), GELU,
                               gate_up_exps fused, down_exps·down_exps_s) )
out = post_ffw_norm(dense + moe) + attn_out                            # final sandwich + residual
```

Notes: `build_moe_ffn(..., norm_w=true, ...)` — top-8 softmax weights are
renormalized to sum 1; the per-expert `ffn_down_exps.scale[128]` multiplies each
expert's down output. Both parallel branches read `attn_out`.

## Seam extensions required (small!)

1. **`AttnMask::Canvas { prompt_len }`** (name TBD) on `Op::Attention`: rows are
   all canvas queries; every row's valid range is `[lo, P+C)` where `lo = 0` on
   global layers and `max(0, P−(n_swa−1))` on SWA layers — i.e. NOT per-row
   causal; uniform bidirectional reach per layer. Backends: CPU interpreter
   arm + Vulkan scalar/split paths (per-row `qpos1` becomes `kv_len` for every
   row; `lo` from the mask instead of the causal window). Flash/nonfa can gate
   off initially (canvas C=256 rows at kv P+C — the split path handles it; perf
   pass later).
2. Per-phase per-layer scalars: the graph is built per phase, so the prefill
   graph bakes encoder scalars and the denoise graph decoder scalars — no op
   changes, just Config carrying both arrays.
3. Denoise-step graph variant in the runner: canvas embedding norm + decoder
   scalars + the Canvas mask + logits over ALL C rows (the spec-decode
   `logits_rows`-style multi-row LM head already exists on Metal; Vulkan's
   lm_head GEMV takes rows — confirm).

## Decode loop (engine level, backend-agnostic)

```
tokens = chat_prompt
for block in 0..blocks:
    P = len(tokens); canvas = [mask_id; C]
    prefill(tokens[cached..])            # causal, encoder scalars (suffix diff as usual)
    for step in 0..steps:
        logits = denoise_forward(canvas) # C rows, decoder scalars, Canvas mask
        unmask most-confident positions per the entropy-bound schedule
        if all unmasked: break
    tokens += canvas                     # block commits; KV rows P..P+C already valid?
                                         # (confirm: the PR re-prefills committed blocks
                                         #  causally or reuses the denoise KV — check)
    stop on EOS inside the block
```

Resolved from the oracle source (`diffusion-cli.cpp` run_turn, ~line 413+):

- **Block commit = causal re-processing.** Committed canvas tokens append to
  `prefix`, and the next block's pass treats them as prompt — in infr terms the
  committed block gets a causal suffix prefill (encoder scalars) through the
  normal session machinery before the next canvas denoise. The whole
  `[prefix | canvas]` must fit one ubatch (the oracle errors otherwise).
- **`trim_canvas`** cuts the block at an end token or a repetition loop; a
  partial cut (< canvas_length) ends the turn. Blocks derive from `-n`:
  `ceil(n_predict / canvas_length)`.
- **Sampler: entropy-bound is the DEFAULT for canvas models**
  (`use_eb = canvas_length > 0 && eb_mode != 2`), parameters metadata-driven
  with these fallbacks:
  `diffusion.eb_max_steps=48, eb_t_min=0.4, eb_t_max=0.8, eb_entropy_bound=0.1, eb_stability_threshold=1, eb_confidence_threshold=0.005`
  (CLI-overridable). Phase 3 ports `diffusion_generate_entropy_bound` from the
  PR.

## Live denoise view (`infr run`)

`infr run --diffusion-visual` turns on a live canvas view while it denoises a
block — a small opt-in port of the oracle's `--diffusion-visual` UX (see
`~/Projects/mxaddict/llama.cpp-dg/examples/diffusion/diffusion-cli.cpp`), not a
dependency on it. It is CLI presentation only, so it is a plain clap flag rather
than a `Config` field; the `INFR_DIFFUSION_VISUAL` spelling still works as the
flag's env fallback. Only active on a TTY stdout (piped/redirected output is
left alone); `--diffusion-visual=force` bypasses the TTY check for scripted
verification. Per step it decodes the block's current canvas fresh (a throwaway
tokenizer, ≤ `canvas_length` tokens, no GPU work), prints already-accepted runs
as normal text and still-renoising positions (this sampler has no literal mask
token — see `crate::diffusion`'s module doc) as a dim `·` placeholder, and
redraws a fixed-size scratch region in place (cursor-up + erase) so the terminal
never scrolls mid-block. The region is erased right before the finished reply
prints, so the ordinary transcript stays clean; the hook is a no-op with the
flag unset (byte-identical decode loop, zero GPU cost).

```bash
infr run <dg-gguf> --diffusion-visual
```

(The flag takes an optional value, so it must follow the model argument — put it
first and clap consumes the model as its value.)

## Validation ladder

1. Metadata parse matches the GGUF dump (Phase 1).
2. Prompt prefill: CPU==Vulkan, finite logits (Phase 1).
3. Denoise step: CPU==Vulkan; qualitative single-step unmask sanity (Phase 2).
4. End-to-end vs `llama-diffusion-cli` on the same GGUF, greedy/fixed-seed —
   token-identical or documented-equivalent (Phase 3).
5. Perf: each denoise step is a C=256-row prefill-shaped forward — the shape
   infr's batched machinery is strongest at (Phase 4 bench + compare rows;
   `infr compare`/`--sweep` measure this against the fork's
   `llama-diffusion-cli` oracle — see "Comparing against the fork oracle" below
   — since mainline `llama-bench` has no diffusion mode at all).

## Comparing against the fork oracle

`arch=diffusion-gemma` isn't merged into mainline llama.cpp, so `llama-bench`
can't run it — but the reference fork at `~/Projects/mxaddict/llama.cpp-dg`
builds `llama-diffusion-cli`, a real oracle for this arch. `infr compare` /
`infr compare --sweep` shell it out instead of bailing (the bail this doc used
to describe is gone).

**Binary resolution** (`ModelBench::llama_diffusion_cli_path`), in order:

1. `INFR_LLAMA_DIFFUSION_CLI` env var — explicit override.
2. `llama-diffusion-cli` on `PATH`.
3. The fork's own build directories: `build-vulkan/bin/llama-diffusion-cli`
   (GPU,
   `cmake -DGGML_VULKAN=ON -B build-vulkan && cmake --build build-vulkan --target llama-diffusion-cli`)
   or `build/bin/llama-diffusion-cli` (CPU-only) — picked by `-ngl 0` vs
   `-ngl >0`, matching every other arm.

**Caveat hit in practice**: mainline llama.cpp already ships its own generic
`llama-diffusion-cli` (LLaDA/Dream support), so tier 2 can resolve to a REAL
binary on `PATH` that still can't load `arch=diffusion-gemma` — it errors
`unknown model architecture: 'diffusion-gemma'` rather than failing to resolve.
`ModelBench::llama_diffusion` catches that specific failure signature on the
first rep and falls through to the fork build automatically (logging the
fallback), so a stock llama.cpp install on `PATH` doesn't silently break this
arm.

**Metric honesty**: DG is entropy-bound, so infr and the oracle run a DIFFERENT
NUMBER of denoise steps for the same `-n 256` request — end-to-end tok/s is NOT
apples-to-apples (whichever tool trims its canvas earlier looks faster for
free). The compare/sweep rows split accordingly:

- `dg-step` — per-step throughput (`canvas_length * steps / decode_secs`, infr's
  `DgBenchResult::parallel_ts` vs the oracle's own reported "in-step parallel N
  tok/s"). This is the real infr-vs-llama ratio and the only DG metric that
  feeds the sweep's ranked "BIGGEST GAPS" summary.
- `dg-e2e` — informational only: each side's own end-to-end tok/s with its own
  step count folded into the row text (e.g. `14.0t/s@48st` vs `46.8t/s@23st`),
  so the step-count mismatch is visible rather than hidden.

Example (Vulkan, `unsloth/diffusiongemma-26B-A4B-it-GGUF:Q4_K_M`):

```
diffusiongemma-26B-A4B-it-Q4_K_M.gguf dg-step    |       674 |      1077 |      0.63x
diffusiongemma-26B-A4B-it-Q4_K_M.gguf dg-e2e     |   14.0t/s@48st |   46.8t/s@23st | informational (steps differ)
```

## Metal implementation notes (Phase D)

Written BLIND on a Linux box: `infr-metal` compiles here
(`#![cfg(target_os = "macos")]` strips the whole crate off-macOS, so
`cargo check`/`clippy` only syntax-check it), but `.metal` shaders only actually
compile+run on a real Metal device, and macOS CI has no GPU-attached runner for
this arch's model files. Same situation as the KV-quant Metal work, which
shipped this way and landed CI-green.

- **Canvas attention** (`AttnMask::Canvas`,
  `crates/infr-metal/shaders/attention.metal`): every one of Metal's existing
  attention tiers (the plain kernel, the split-KV `ATTNSPLIT_KERNEL` family, the
  flash/flash2 kernels, the vector-flash `attnvec_*` kernels) computes its
  per-row bound from `pos + row_index` plus an optional sliding-window `lo` —
  none can express "every row shares the SAME `[lo, kv_len)` regardless of its
  own position". Repurposing an existing kernel's dead field with a sentinel
  (the way the Vulkan `attn_partial.comp` canvas branch reuses its `rows`
  push-constant slot) risked a genuine non-canvas dispatch and a canvas dispatch
  both landing on the same nonzero sentinel value in the SAME kernel, so canvas
  gets its own dedicated `ATTNSPLIT_CANVAS_KERNEL` macro (4 instantiations:
  `attention_canvas_f32`/ `attention_canvas_f16kv` at NSG=8/MAXHD=256,
  `attention_canvas32_f32`/ `attention_canvas32_f16kv` at NSG=32/MAXHD=128) that
  never serves a non-canvas caller, so its repurposed `p.pos`/`p.kv_len` fields
  (fixed `hi`/`lo` instead of their ordinary meaning) can't collide with
  anything. `exec.rs`'s `Op::Attention` arm forces every `AttnMask::Canvas` op
  straight to this kernel family (bypassing the flash/split/vec routing
  entirely), gated the same way the ordinary split kernels are (threadgroup-cap
  `fits()` checks, hd<=128 preferring the 32-wide variant). Q8_0 KV cache +
  canvas is explicitly rejected (`Error::Unsupported`) — Q8_0's attention family
  has no split-canvas sibling yet; every other KV dtype (native f16, or a
  block-quant that goes through the existing dequant-to-f16-scratch prepass) is
  covered.
- **Self-conditioning** (`gpu_sc`, Phase B): widened from Vulkan-only to
  `matches!(be.name(), "vulkan" | "metal")` — verified by reading, not running:
  `softmax_wide_f32` is a plain grid-stride loop over `(rows, dim)` with no
  shape restriction (handles the `[256, vocab]` shape unmodified), and
  `sc_embT`'s `DType::F16` weight already flows through Metal's existing
  non-quant `Linear` path (`weight_buf` + `linear_f32`) — functionally correct,
  though not a native f16 GEMV: `weight_buf` dequant-caches the whole
  `[n_embd, n_vocab]` weight to f32 once per session (~2x the f16 footprint).
- **Batched MoE**: NOT built — too large to write blind. Metal's device MoE path
  requires separate gate/up banks and no down scale (`device_ok` in `exec.rs`'s
  `MoeFfn` arm), so DG's fused `gate_up_exps` + `ffn_down_exps.scale` falls to
  the host path (B100).

The kernel-level checks (MSL compile, the `attention_canvas_split*` parity
tests, `kernel_names`) run in the macOS CI job
(`cargo test -p infr-metal -- --include-ignored`). The checks that need a real
DG model on an Apple GPU are in `backlog.md` § B100.

## Oracle reference outputs (CPU, greedy `-s 42 --temp 0`, captured 2026-07-05)

```
$ llama-diffusion-cli -m <Q4_K_M> -p "What is the capital of France?" -n 64 -s 42 --temp 0
<|channel>thought
The user is asking for the capital of France.
    *   Country: France.
    *   Capital: Paris.
Provide the direct answer clearly.<channel|>The capital of France is Paris.
(10 EB steps, 1 block; 13.2 tok/s CPU, in-step parallel 132 tok/s)

$ llama-diffusion-cli -m <Q4_K_M> -p "Write a haiku about the moon." -n 64 -s 42 --temp 0
<|channel>thought
*   Topic: The moon.
    *   Format: Haiku (5-7-5 syllables).

    *   Night
(19 EB steps, 1 block; trim cut mid-thought at the 64-token budget)
```

Thinking channel markers: `<|channel>thought … <channel|>` (from the GGUF's own
chat template — the jinja renderer handles it like every other model).
