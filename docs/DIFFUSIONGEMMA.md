# DiffusionGemma (`diffusion-gemma`) — design for the unified seam

Block text-diffusion MoE on a **Gemma-4 backbone**. Reference implementation:
llama.cpp PR #24423 (checked out at `~/Projects/mxaddict/llama.cpp-dg`,
`src/models/diffusion-gemma.cpp` + `gemma4-common.h`); oracle binary
`build/bin/llama-diffusion-cli` (CPU build). Model:
`unsloth/diffusiongemma-26B-A4B-it-GGUF` (Q4_K_M ≈ 16 GB, fits the 7900 XTX).

This REPLACES the pre-seam scaffolding (`infr-model`/`infr-decode`/the
`infr-engine::Engine` stub, all dead code) — DiffusionGemma is built on the
unified transformer runner like every other arch; the scaffolding gets deleted
in the final phase.

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

## Validation ladder

1. Metadata parse matches the GGUF dump (Phase 1).
2. Prompt prefill: CPU==Vulkan, finite logits (Phase 1).
3. Denoise step: CPU==Vulkan; qualitative single-step unmask sanity (Phase 2).
4. End-to-end vs `llama-diffusion-cli` on the same GGUF, greedy/fixed-seed —
   token-identical or documented-equivalent (Phase 3).
5. Perf: each denoise step is a C=256-row prefill-shaped forward — the shape
   infr's batched machinery is strongest at (Phase 4 bench + compare rows; note
   llama-bench has no diffusion mode, so compare needs a diffusion-aware
   scenario — measure steps/s and tokens/s end-to-end).

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
