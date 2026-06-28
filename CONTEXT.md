# infr — session context (resume file)

Compressed state for starting a fresh session. `infr` = pure-Rust, Vulkan-first LLM
inference engine. Repo: `~/Projects/kryptic-sh/infr`, remote
`git@github.com:kryptic-sh/infr.git`, branch `main`. GPU: AMD RX 7900 XTX (RADV/Vulkan,
24GB), wave32, coopmat. Compare target: llama.cpp (`llama-cli`/`llama-bench`, pacman
`llama-cpp-vulkan`, build b9827).

## Workflow rules (user)
- Conventional Commits (`type(scope): msg`). **No Claude attribution** in commits/PRs/comments.
- Prefix shell with `rtk` (token-saving proxy). Run `cargo clippy/fmt/test` after Rust changes.
- **Push after every verified win** (don't batch). Max 2 subagents.
- Caveman comms mode active (terse); code/commits normal prose. Nickname: "Jean Claude Van Dam".
- North star: **long-context speed** (coding-agent workload) — win 16k/32k+, deprioritize short prompt.

## Qwen3.5 / Qwen3.6 (`qwen35` = Qwen3-Next) — CPU reference WORKS ✅
Scoped to the CPU reference (NOT GPU — ggml/our Vulkan has no SSM kernels; even llama.cpp runs
qwen35 CPU-only). DONE: `infr run hf:unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf "..."`
produces correct output ("…France is" → "…Paris…"; "2+2=" → "4"). ~3 tok/s (naive single-thread
f32 matvec — optimization TODO: rayon-parallelize `matvec`, then quantized matvec). One-shot only
(no multi-turn REPL yet). **THE bug was the attention q/gate split: `attn_q` packs query+gate
INTERLEAVED PER HEAD `[h0 q|h0 gate|h1 q|h1 gate|…]`, not two contiguous blocks** (fixed in
commit 2f55bf4 / 3cacd4a). Code: `crates/infr-llama/src/qwen35.rs` (Cfg/Model/forward/generate_chat
/is_qwen35); dispatched in `cmd_run`. Debug env: `Q35_DBG`, `Q35_NOLIN`, `Q35_NOATTN`, `Q35_PROMPT`,
`Q35_N`. NOTE: set `TMPDIR=$HOME/.cache/tmp` for shell work — `/tmp` is a quota'd RAM tmpfs.

**Architecture (fully reverse-engineered, spec in `docs/QWEN35.md`):** hybrid of gated-DeltaNet
linear-attention + gated full-attention. Qwen3.5-0.8B: 24 layers, `full_attention_interval=4` →
6 attention layers (i where `(i+1)%4==0`: 3,7,11,15,19,23) + 18 linear (gated DeltaNet); dense
SwiGLU (larger Qwen3-Next are MoE — skip for ≤1B). Qwen3.6 has NO ≤1B (smallest 27B); only
Qwen3.5-0.8B fits "≤1B first". Both declare `general.architecture = qwen35`.
- Linear layer: `attn_qkv`[→6144]=q+k+v (16 k-heads×128 + 16×128 + 16 v-heads×128); `attn_gate`[→2048]=z;
  `ssm_alpha/beta`[→16]=a/b per v-head; `ssm_a`=−exp(A_log) (already negative in GGUF); `ssm_dt.bias`;
  `ssm_conv1d`[4,6144] depthwise causal conv; `ssm_norm`[128]; `ssm_out`[2048→1024]. Flow:
  rmsnorm→qkv & z → ggml_ssm_conv(k=4)→silu → split q/k/v → l2norm(q,k), q*=1/√128 →
  per v-head gated-delta recurrence (S[128×128]: `S*=exp(g)`; `kv=kᵀS`; `delta=(v−kv)·sigmoid(b)`;
  `S+=k⊗delta`; `out=qᵀS`), g=`ssm_a·softplus(a+dt)` → silu-gated rmsnorm(out, ssm_norm, gate=z) →
  ssm_out. Then residual; rmsnorm(post_attention_norm); SwiGLU FFN; residual.
- Attention layer: `attn_q`[→4096]=q(8×256)+out-gate(2048); `attn_k/v`[→512]=2 KV×256; q/k norm(256);
  **head_dim=256**; partial sectioned RoPE (rope_dim=64, sections [11,11,10,0] — text = standard RoPE
  on first 64 dims); sigmoid output gate; GQA 8q/2kv; then FFN.

**Code (committed):** `crates/infr-llama/src/qwen35.rs` — `Cfg::from_gguf`, `Model::load` (loads both
layer types as f32 via `crate::load_f32`, which dequants Q4_K/Q5_K/Q6_K/Q8_0/F16/F32), `Model::forward`
(full CPU forward + recurrent conv/SSM state + KV cache), `generate()`. Two ignored tests
(`loads_and_dims` passes; `greedy_generate`). Registered `pub mod qwen35;` in lib.rs.

**STATUS: forward compiles + runs end-to-end but output is GARBAGE.** Per-layer hidden RMS is healthy
and finite (0.04→0.6, no blow-up) → a SUBTLE layout/numeric bug, not instability. Linear path
re-verified on paper (matvec orientation, q/k/v split, recurrence indexing, gated norm all match
refs). Prime suspect: **conv tap order / `ggml_ssm_conv` weight convention** (corrupts all 18 linear
layers without blowing norms = exact symptom). UNCOMMITTED debug toggles in forward: `Q35_DBG`
(per-layer norms), `Q35_NOLIN`, `Q35_NOATTN` (zero a mixer type to bisect linear vs attention).

**Next steps to finish CPU reference:**
1. Read `ggml`'s CPU `ggml_ssm_conv` op + tail of llama.cpp `src/models/qwen3next.cpp` (lines ~470-593,
   the recurrence/norm/out wiring) — pin exact conv tap order + delta-net op sequence. Fix the bug.
2. Validate vs llama.cpp CPU. **llama.cpp runs qwen35 CPU-ONLY** (`-ngl 0`; Vulkan `-ngl 99` errors —
   ggml has NO Vulkan SSM kernels → a GPU qwen35 path is research-grade, beyond even llama.cpp, hence
   CPU-only scope). Oracle: `llama-cli -m <blob> -ngl 0 -t 16 --temp 0 -st --simple-io -p "..."` works
   (single-turn, applies chat template, exits). `-no-cnv` HANGS in this build. Qwen3.5 is a thinking model.
3. Once coherent: wire `infr run` to dispatch qwen35 vs the dense qwen3/llama path by `general.architecture`.

**Refs (read for math, reimplement — both MIT):** HF `transformers/models/qwen3_next/modular_qwen3_next.py`
(recurrence: `torch_chunk_gated_delta_rule` + the fused recurrent loop ~lines 300-340); llama.cpp
`src/models/qwen3next.cpp` (GGUF tensor→role, `build_qkvz`, `build_layer_attn_linear`).

**Model files (in our store):**
- Qwen3.5-0.8B: `~/.cache/infr/models/blobs/sha256-bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`
  (pull: `infr pull hf:unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf`)
- Qwen3-14B (dense, runs fine): `hf:unsloth/Qwen3-14B-GGUF:Qwen3-14B-Q4_K_M.gguf`
- qwen3-0.6b (ollama): `ol:qwen3:0.6b`; also `~/Projects/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf`

## Session accomplishments (committed, on main)
Perf (dense qwen3, the optimization target):
- `mmq` (dp4a int) is now DEFAULT for u4 prefill projections (`INFR_NOMMQ` to disable) — small-ubatch
  prefill was f16-dequant-bound; +26-40% at ub512.
- Full-occupancy V pass in flash-decode `attn_partial.comp` (both subgroups do V): decode +12-24%,
  grows with depth; attn_partial now ~93% of peak BW (bandwidth-bound, done).
- 32 KV chunks/head (was 64): halves attn_combine; `INFR_DECODE_NCHUNK`.
- prefill chunk budget 16M→32M: bigger chunks at depth; big-ingest turn +16-18%.
- 256-thread subgroup rmsnorm kernel (kept though end-to-end neutral here — helps slower/higher-latency GPUs).
- Shared auto-width progress bar `infr_core::progress::bar(total, label, Unit)` used by download + weight load.
- `infr bench` gained `-b/-ub` + `--pg P,G`; `infr compare` CLI command (replaced scripts/compare.sh,
  now deleted). Coding-agent scenarios: CONTEXT LOAD / REPLY@depth / SESSION TURN.

Standing vs llama.cpp (qwen3-0.6b-Q4, tool-default): prefill 0.65/0.73/0.80× (8k/16k/32k);
decode tg256 0.81/0.84/0.90×; turns 0.81-0.90×. **Matched ubatch=2048: we WIN long-ctx prefill
(16k 1.34×, 32k 1.38×).** Short-prompt prefill (0.37× @512) is weakest but lowest priority.

Infra:
- Pull/store refactor: own store `$INFR_MODELS` or `$XDG_CACHE_HOME/infr/models` (no more ~/.ollama,
  /var/lib/ollama). Standalone HTTP pulls for BOTH HuggingFace AND Ollama registry (no `ollama`/HF CLI).
  Resumable downloads (HTTP Range), idempotent (fixed HF re-pull-every-run bug), prefix aliases
  `hf:`/`huggingface:` and `ol:`/`ollama:`. Moved existing models into the new store.
- Model-load progress bar (per-layer, byte-based, TTY-gated).

## Negative results (DON'T re-try — measured)
- Decode dispatch-chain fusion (rmsnorm+GEMV, qkv fusion `INFR_FUSE`) = NEUTRAL: fused GEMV re-reads
  normw per output workgroup, cost ≈ saved dispatch. Decode is DISPATCH-LATENCY-bound, not op-compute.
- `NUM_ROWS=4` register blocking on decode GEMV = WORSE (−32%): n=1 needs many workgroups for occupancy.
- Conclusion: only BIG bandwidth-bound decode ops respond (attention done); small ops can't be sped up
  end-to-end. Next real decode lever would be KV quant (q8_0) — trades quality, not pursued.

## Persistent memories (live in ~/.claude/projects/-home-mxaddict-Projects-mxaddict-grbrd/memory/)
- `infr-optimization-priority` — north star long-ctx; standing vs llama; lossless wins; reproduce via `infr compare`.
- `infr-decode-kernel-catchup` — decode GEMV/attention kernel history; V-fix; 32-chunk; fusion/NUM_ROWS
  dead ends; decode is dispatch-latency-bound; KV quant = only big remaining decode lever.
- `infr-ollama-store-compat` — NOTE: now superseded by the own-store refactor (XDG `~/.cache/infr/models`).
- `ollama-gpu-verify-gotcha`, `rust-vulkan-llm-engine-plan`, `qwen3-chat-gotchas`, `push-after-progress`,
  `diffusiongemma-opencode-setup`.

## Handy commands
- Build: `rtk proxy cargo build -p infr-cli --release` → `./target/release/infr`
- Compare: `infr compare <model.gguf>` (also `-u 2048` matched, `--ctx`, `--turn P,G`)
- qwen35 CPU test: `cargo test -p infr-llama --release qwen35::tests::greedy -- --ignored --nocapture`
  (env: `Q35_DBG=1` per-layer norms; `Q35_NOLIN=1`/`Q35_NOATTN=1` bisect)
- Profiling: `INFR_PROF2=1` (per-op GPU timestamps), `INFR_PROF=1` (record vs submit+wait).
