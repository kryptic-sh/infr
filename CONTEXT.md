# infr — session context (resume file)

Compressed state for starting a fresh session. `infr` = pure-Rust, Vulkan-first
LLM inference engine. Repo: `~/Projects/kryptic-sh/infr`, remote
`git@github.com:kryptic-sh/infr.git`, branch `main`. GPU: AMD RX 7900 XTX
(RADV/Vulkan, 24GB), wave32, coopmat. Compare target: llama.cpp
(`llama-cli`/`llama-bench`, pacman `llama-cpp-vulkan`, build b9827).

## Workflow rules (user)

- Conventional Commits (`type(scope): msg`). **No Claude attribution** in
  commits/PRs/comments.
- Prefix shell with `rtk` (token-saving proxy). Run `cargo clippy/fmt/test`
  after Rust changes.
- **Push after every verified win** (don't batch). Max 2 subagents.
- Caveman comms mode active (terse); code/commits normal prose. Nickname: "Jean
  Claude Van Dam".
- North star: **long-context speed** (coding-agent workload) — win 16k/32k+,
  deprioritize short prompt.

## Qwen3.5 / Qwen3.6 (`qwen35` = Qwen3-Next) — CPU reference WORKS ✅

Scoped to a CPU reference first. GPU note (CORRECTED): ggml's Vulkan backend
doesn't _implement_ the SSM/gated-delta/ssm_conv ops, but ggml's scheduler
**falls back to CPU per-op** — so `llama-cli -ngl 99 -dev Vulkan0` RUNS qwen35
as a CPU+GPU hybrid (NOT an error; my earlier "Vulkan-errors/CPU-only" claim was
a misattribution of `~`-path/`-no-cnv` failures). Benchmark (Qwen3.5-0.8B-Q4,
7900 XTX, llama.cpp): pure CPU `-ngl 0` pp512=4208 tg64=33.5 t/s; Vulkan hybrid
`-ngl 99` pp512=18054 (4.3×) tg64=387 (11.5×). \*\*Implication: an `infr` HYBRID
path (SSM recurrence

- conv on CPU, all matmuls/FFN/attention on Vulkan) would capture ~most of the
  GPU win without writing SSM Vulkan kernels — the recommended next step over a
  full-GPU SSM-kernel build.** DONE:
  `infr run hf:unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf "..."`
  produces correct output ("…France is" → "…Paris…"; "2+2=" → "4"). ~3 tok/s
  (naive single-thread f32 matvec — optimization TODO: rayon-parallelize
  `matvec`, then quantized matvec). One-shot only (no multi-turn REPL yet).
  **THE bug was the attention q/gate split: `attn_q` packs query+gate
  INTERLEAVED PER HEAD `[h0 q|h0 gate|h1 q|h1 gate|…]`, not two contiguous
  blocks\*\* (fixed in commit 2f55bf4 / 3cacd4a). Code:
  `crates/infr-llama/src/qwen35.rs` (Cfg/Model/forward/generate_chat
  /is_qwen35); dispatched in `cmd_run`. Debug env: `Q35_DBG`, `Q35_NOLIN`,
  `Q35_NOATTN`, `Q35_PROMPT`, `Q35_N`. NOTE: set `TMPDIR=$HOME/.cache/tmp` for
  shell work — `/tmp` is a quota'd RAM tmpfs.

**Architecture (fully reverse-engineered, spec in `docs/QWEN35.md`):** hybrid of
gated-DeltaNet linear-attention + gated full-attention. Qwen3.5-0.8B: 24 layers,
`full_attention_interval=4` → 6 attention layers (i where `(i+1)%4==0`:
3,7,11,15,19,23) + 18 linear (gated DeltaNet); dense SwiGLU (larger Qwen3-Next
are MoE — skip for ≤1B). Qwen3.6 has NO ≤1B (smallest 27B); only Qwen3.5-0.8B
fits "≤1B first". Both declare `general.architecture = qwen35`.

- Linear layer: `attn_qkv`[→6144]=q+k+v (16 k-heads×128 + 16×128 + 16
  v-heads×128); `attn_gate`[→2048]=z; `ssm_alpha/beta`[→16]=a/b per v-head;
  `ssm_a`=−exp(A_log) (already negative in GGUF); `ssm_dt.bias`;
  `ssm_conv1d`[4,6144] depthwise causal conv; `ssm_norm`[128];
  `ssm_out`[2048→1024]. Flow: rmsnorm→qkv & z → ggml_ssm_conv(k=4)→silu → split
  q/k/v → l2norm(q,k), q*=1/√128 → per v-head gated-delta recurrence
  (S[128×128]:
  `S*=exp(g)`; `kv=kᵀS`; `delta=(v−kv)·sigmoid(b)`; `S+=k⊗delta`; `out=qᵀS`), g=`ssm_a·softplus(a+dt)`
  → silu-gated rmsnorm(out, ssm_norm, gate=z) → ssm_out. Then residual;
  rmsnorm(post_attention_norm); SwiGLU FFN; residual.
- Attention layer: `attn_q`[→4096]=q(8×256)+out-gate(2048); `attn_k/v`[→512]=2
  KV×256; q/k norm(256); **head_dim=256**; partial sectioned RoPE (rope_dim=64,
  sections [11,11,10,0] — text = standard RoPE on first 64 dims); sigmoid output
  gate; GQA 8q/2kv; then FFN.

**Code (committed):** `crates/infr-llama/src/qwen35.rs` — `Cfg::from_gguf`,
`Model::load` (loads both layer types as f32 via `crate::load_f32`, which
dequants Q4_K/Q5_K/Q6_K/Q8_0/F16/F32), `Model::forward` (full CPU forward +
recurrent conv/SSM state + KV cache), `generate()`. Two ignored tests
(`loads_and_dims` passes; `greedy_generate`). Registered `pub mod qwen35;` in
lib.rs.

**STATUS: CPU reference correct & wired into `infr run`** (see top section). The
bug that was fixed: attention q/gate interleaved-per-head split. Debug toggles
(committed, env-gated): `Q35_DBG` (per-layer norms), `Q35_NOLIN`/`Q35_NOATTN`
(zero a mixer to bisect — how the bug was localized).

**Next steps (CPU reference is DONE + wired into `infr run`):**

1. **Hybrid GPU path (recommended, high ROL):** run SSM recurrence + conv on
   CPU, push the matmuls/ FFN/attention through our Vulkan backend. Benchmark
   above shows ~11× decode from this alone — no SSM Vulkan kernels needed.
   (Full-GPU SSM kernels are the bigger, optional follow-on.)
2. **Speed up the CPU reference:** rayon-parallelize `matvec` (single biggest
   win at ~3 tok/s), then a quantized matvec (keep weights quantized instead of
   full f32 dequant at load).
3. Multi-turn REPL for qwen35 (currently one-shot in `cmd_run`). Oracle for
   validation:
   `llama-cli -m <blob> -ngl 0 -t 16 --temp 0 -st --simple-io -p "..."`
   (single-turn, applies chat template, exits; `-ngl 99` also works = hybrid).
   `-no-cnv` HANGS in this build. Qwen3.5 is a thinking model (emits
   `<think>…</think>`).

**Refs (read for math, reimplement — both MIT):** HF
`transformers/models/qwen3_next/modular_qwen3_next.py` (recurrence:
`torch_chunk_gated_delta_rule` + the fused recurrent loop ~lines 300-340);
llama.cpp `src/models/qwen3next.cpp` (GGUF tensor→role, `build_qkvz`,
`build_layer_attn_linear`).

**Model files (in our store):**

- Qwen3.5-0.8B:
  `~/.cache/infr/models/blobs/sha256-bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`
  (pull: `infr pull hf:unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf`)
- Qwen3-14B (dense, runs fine):
  `hf:unsloth/Qwen3-14B-GGUF:Qwen3-14B-Q4_K_M.gguf`
- qwen3-0.6b (ollama): `ol:qwen3:0.6b`; also
  `~/Projects/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf`

## Session accomplishments (committed, on main)

Perf (dense qwen3, the optimization target):

- `mmq` (dp4a int) is now DEFAULT for u4 prefill projections (`INFR_NOMMQ` to
  disable) — small-ubatch prefill was f16-dequant-bound; +26-40% at ub512.
- Full-occupancy V pass in flash-decode `attn_partial.comp` (both subgroups do
  V): decode +12-24%, grows with depth; attn_partial now ~93% of peak BW
  (bandwidth-bound, done).
- 32 KV chunks/head (was 64): halves attn_combine; `INFR_DECODE_NCHUNK`.
- prefill chunk budget 16M→32M: bigger chunks at depth; big-ingest turn +16-18%.
- 256-thread subgroup rmsnorm kernel (kept though end-to-end neutral here —
  helps slower/higher-latency GPUs).
- Shared auto-width progress bar `infr_core::progress::bar(total, label, Unit)`
  used by download + weight load.
- `infr bench` gained `-b/-ub` + `--pg P,G`; `infr compare` CLI command
  (replaced scripts/compare.sh, now deleted). Coding-agent scenarios: CONTEXT
  LOAD / REPLY@depth / SESSION TURN.

Standing vs llama.cpp (qwen3-0.6b-Q4, tool-default, NATIVE DEFAULT): prefill
0.67/0.75/0.83× (8k/16k/32k); decode tg256 0.94/0.94/0.97× (was 0.81/0.84/0.90×
pre-native — the native-default flip lifted decode to ~parity); turns 0.90-0.95×
EXCEPT pg8192,512@32000 = 0.74× (REPRODUCIBLE, not variance — confirmed 2×).
Root cause: large prefill chunk (8192 tok) on deep KV (32k) ⇒ ingest attends
over ~40k ctx; infr's DEEP-CONTEXT PREFILL ATTENTION is the bottleneck
(decode@32k fine 0.97×, pure pp32000 0.84×, small 2048 ingest 0.94× — only
big-chunk×deep-ctx dips). **Matched ubatch=2048: we WIN long-ctx prefill (16k
1.34×, 32k 1.38×).** Short-prompt prefill (0.37× @512) is weakest but lowest
priority.

DEEP-PREFILL-ATTENTION investigation (pg8192,512@32000, repro
`infr bench --pg 8192,512 -d 32000`): profiled INFR_PROF2 — `attn_flash` grows
506→1866 µs/op as the ingest deepens 32k→40k, reaching ~47% of chunk GPU time (≈
matmul_proj). Tuning swept (ALL solo — running 2 benches at once halves each via
GPU contention, invalidated a first pass): default `attn_flash_warp` 950 t/s
BEST; INFR_FLASH_REG (FA2 reg-O Br=128) 796, INFR_NO_FLASH (non-FA, materializes
S) 739, INFR_NO_FLASH_WARP slower. Chunk-budget near-flat (16M 922 / 32M 950 /
48M 957 / 64M 963 — +1.4% at 64M, not worth watchdog risk). n_splits heuristic
is adaptive+good (INFR_FLASH_SPLITS override also hits the 512-tok decode →
can't isolate). attn_flash_warp ALREADY uses coopmat (coopMatMulAdd QK+PV) — the
limit is BN=64 K-tile (S/P/O in shared) ⇒ ~625 iters over 40k KV. Real lever =
wider-KV-tile / better-pipelined flash kernel (substantial; FA2-reg attempt
already lost). Config is otherwise tuned; gap is the kernel, not a knob.

WATCHDOG IS REAL + orthogonal to native loading (it's per-SUBMIT GPU time; the
recorder submits a whole chunk's 28-layer forward as one cmd buffer +
queue_wait_idle). PROVEN: forcing the 8192 ingest into ONE chunk at d32000
(cap=8192) → `VK_ERROR_DEVICE_LOST` (TDR reset; recovers on fresh process). 4096
chunks complete (387 t/s), 8192 doesn't. So the 32M-budget taper (→~960 tok @
32k) is load-bearing, not paranoia. Native being faster only buys headroom, not
removal of chunking. ⇒ chunk size is NOT the deep-prefill lever (bigger barely
helps + eventually trips TDR); the flash kernel is.

VRAM pre-flight (load): weights upload incrementally per-tensor (gpu-allocator:
dedicated VkDeviceMemory ≥32MB, else 256MB pooled blocks) — NO arena. Added a
pre-flight sizing+fit check in `Model::load_opt`: `weight_footprint(gguf)` sums
each tensor's RESIDENT bytes (mirrors `upload_wt`: native raw-blocks / unified
repack / f16; excludes token_embd=host unless tied lm head), split dense vs
MoE-expert (`*_exps`); `VulkanBackend::vram()` (VK_EXT_memory_budget) gives
total+live-free; bail early if weights+384MB > free. Estimates verified vs
actual: Q4_K 0.57 / Q8_0 0.78 / BF16 1.19 GB. MoE-READY = expert bytes tracked
separately + all tensors enumerated (correct once an MoE arch is added).

WEIGHT ARENA (done): `VulkanBackend::reserve_weights(total)` commits the model's
whole weight VRAM as ONE contiguous device-memory block up front; `Weights`
allocs bump-allocate from it (never individually freed → one-shot free, no
fragmentation). `VkBuffer.backing: Backing::{Pooled(gpu-allocator) | Arena}`;
arena buffers own only their handle, the block owns memory; `mapped_ptr()`
replaces direct `.allocation` (arena bufs are device-local, staging-filled).
64MB overflow blocks cover footprint underestimates; reservation failure →
loader note

- per-tensor fallback. `load_opt` reserves after the fit check. Test
  `weight_arena_roundtrip` (reserved+overflow+coexistence). Perf unchanged.

EXPERT POOL (done): `infr_vulkan::ExpertPool` (src/expert_pool.rs) = host-backed
LRU VRAM cache for MoE experts — n_slots fixed slots; `resident(be, id, bytes)`
returns the slot buffer (hit = cached, miss = evict LRU + upload). Weight VRAM =
dense + n_slots·stride regardless of expert count. Backend-agnostic (public
Backend API). `WeightFootprint::streaming_total(n_slots, stride)` = streaming
plan vs `total()` (all-resident). Tested standalone
(residency/hit/LRU-evict/round- trip); the MoE FORWARD that consumes it is NOT
wired yet. MoE REMAINING (the real consumer): (1) arch support — infr-llama
bails on non-llama|qwen3; add an MoE arch (qwen3moe/gemma3moe) reading n_expert/
n_expert_used/`*_exps` tensors; (2) router = ffn_gate_inp → top-k
experts+weights; (3) expert-dispatch FFN forward: ExpertPool.resident per routed
expert → matmul vs the slot → weighted-accumulate; (4) loader picks all-resident
vs streaming via the footprint.

QWEN3MOE WIRED + VALIDATED (done): arch="qwen3moe" → `Config::moe` (MoeConfig:
n*expert/n_used/n_ff_exp; meta keys expert_count/expert_used_count/
expert_feed_forward_length). `FfnWt::{Dense|Moe}`; `load_moe` slices each expert
from stacked
`ffn*{gate,up,down}\_exps`(contiguous 1/n_expert block) via`upload_wt_bytes`(experts stay quantized).`forward_moe`= eager, NO KV: GPU matmuls via`gemv_wt`(one-submit linear, any Wt kind), host qk-norm/RoPE/causal- GQA/router-softmax-topk(renorm)/expert-combine.`generate_moe`+ cmd_run branch (one-shot). TEST MODEL:`hf:unsloth/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf`(~18.6GB). RESULT: loads at weights 18.35GB (dense 0.80 + experts 17.55, footprint split correct, arena reserves it, fits 24GB ALL-RESIDENT so ExpertPool not needed here); greedy output correct + coherent ("<think>\nOkay, the user asked..."). PERF: 38.4 tok/s decode (Qwen3-30B-A3B-Q4, greedy), from 0.3 originally (~128x). Wins: (1) host KV cache `MoeKv`+`forward_moe_chunk`+`attention_kv`+`rope_rows_at`→ decode 1 tok/step not O(n²): 0.3→8.6; (2)`gemv_wt_many`batches N independent GEMVs into ONE submit → per-token expert gate+up (1 submit) / down (1 submit) + per-layer QKV (1 submit): ~1400→~240 submits/tok, 8.6→38.4 (prefill 1879→278ms). MoE PERF FOLLOW-ON (open, ~240 submits/tok left): GPU-resident attention (drop qkv→attn→o readbacks); GPU top-k + gathered`mul_mat_id`;
batch tokens-by-expert in prefill. ExpertPool streaming only needed for MoE
models that DON'T fit VRAM (PCIe caveat: pays off only if working set fits the
pool / high LRU hit-rate).

Infra:

- Pull/store refactor: own store `$INFR_MODELS` or `$XDG_CACHE_HOME/infr/models`
  (no more ~/.ollama, /var/lib/ollama). Standalone HTTP pulls for BOTH
  HuggingFace AND Ollama registry (no `ollama`/HF CLI). Resumable downloads
  (HTTP Range), idempotent (fixed HF re-pull-every-run bug), prefix aliases
  `hf:`/`huggingface:` and `ol:`/`ollama:`. Moved existing models into the new
  store.
- Model-load progress bar (per-layer, byte-based, TTY-gated).

## Negative results (DON'T re-try — measured)

- Decode dispatch-chain fusion (rmsnorm+GEMV, qkv fusion `INFR_FUSE`) = NEUTRAL:
  fused GEMV re-reads normw per output workgroup, cost ≈ saved dispatch. Decode
  is DISPATCH-LATENCY-bound, not op-compute.
- `NUM_ROWS=4` register blocking on decode GEMV = WORSE (−32%): n=1 needs many
  workgroups for occupancy.
- Conclusion: only BIG bandwidth-bound decode ops respond (attention done);
  small ops can't be sped up end-to-end. Next real decode lever would be KV
  quant (q8_0) — trades quality, not pursued.

## Native perf (INFR_NATIVE=1) — decode AND prefill now beat unified

- Native = raw GGUF blocks on GPU, in-shader dequant. Unified (`Wt::Q`) =
  pre-repacked at load (u4/u8 idx + f16 scale/min). Native = smaller VRAM + fast
  load. Decode uses `native_gemv.comp`; prefill uses `native_gemm.comp` (NEW).
- Shared decode lib `native_decode.glsl`: per-format `dq(g)` (single elem) +
  amortized `dqblk(gstart, out v[32])` (decode block scale ONCE per
  32-sub-block, reused across the sub-block). Included by BOTH gemv + gemm. The
  includer declares the weight SSBO as `uint nw[]` (any binding).
- ROOT CAUSE native decode was slow: per-element `dq(g)` re-decoded the block's
  f16 scale (+ Q4_K 6-bit sub-scale, Q3_K 12-byte recon) EVERY element. Fix:
  sub-block-major via dqblk. DEAD END: subgroup reduction (64→32 lanes) WORSE —
  decode-bound, K parallelism beats dropping the 6 reduce barriers.
- ROOT CAUSE native prefill was 3x slow: per-row GEMV for the matmul. FIX:
  `matmul_native` = tiled coopmat GEMM (copy of gemm_proj) that dequants weights
  via dqblk during shared staging → decode-once per weight elem, reused across
  the 64-row tile. Dispatched when n%64 && k%32 (else GEMV fallback). GOTCHA
  fixed: `matmul_native(dtype, a, w, c)` — a=ACTIVATIONS, w=WEIGHTS; both call
  sites first passed (weight, activation) → swapped buffers → garbage/NaN.
- DECODE (qwen3-0.6b, ctx=128, t/s) native vs unified — native WINS all: Q8_0
  423/365, Q6_K 413/357, Q5_K 376/365, Q4_K 506/480 (was 259 pre-fix).
- PREFILL (pp512, t/s) native vs unified — native wins K-quants: Q8_0 8487/6283,
  Q6_K 7622/5437, Q5_K 8018/6320, Q4_K 10248/10744 (unified dp4a mmq still edges
  Q4_K by ~5%). Was native ~3400 (3x slower) pre-GEMM.
- ⇒ Native now faster on decode (all) + prefill (except Q4_K ~4%) AND smaller
  VRAM. NOW THE DEFAULT for optimized affine quants (Q8_0, Q4_0/1, Q5_0/1,
  Q2_K..Q6_K) via `is_native_default`/`use_native_for`. `INFR_NONATIVE=1` →
  unified/f16 (old behavior); `INFR_NATIVE=1` → native for ALL supported formats
  (incl. grid/codebook, which otherwise stay f16 since their native path is the
  slow per-element fallback). Q4_K prefill gap = unified's int8 dp4a mmq beats
  f16 coopmat ~5%/op (native coopmat itself is +45% vs unified coopmat); a
  native dp4a GEMM would close it but is poor ROI (Q4_K superblock scales
  hairy).
- Tests: 36 `*_native_matches_cpu` (GEMV decode) + 3
  `*_native_gemm_matches_gemv` (GEMM vs trusted GEMV, M spans row-tiles,
  col-varied weights). All pass.
- Codebook/grid i-quants (IQ\*, MXFP4, NVFP4, TQ) still use fallback `dqsub`
  (got contiguous layout, not decode amortization) — low priority, rarely used.

## Persistent memories (live in ~/.claude/projects/-home-mxaddict-Projects-mxaddict-grbrd/memory/)

- `infr-optimization-priority` — north star long-ctx; standing vs llama;
  lossless wins; reproduce via `infr compare`.
- `infr-decode-kernel-catchup` — decode GEMV/attention kernel history; V-fix;
  32-chunk; fusion/NUM_ROWS dead ends; decode is dispatch-latency-bound; KV
  quant = only big remaining decode lever.
- `infr-ollama-store-compat` — NOTE: now superseded by the own-store refactor
  (XDG `~/.cache/infr/models`).
- `ollama-gpu-verify-gotcha`, `rust-vulkan-llm-engine-plan`,
  `qwen3-chat-gotchas`, `push-after-progress`, `diffusiongemma-opencode-setup`.

## Handy commands

- Build: `rtk proxy cargo build -p infr-cli --release` → `./target/release/infr`
- Compare: `infr compare <model.gguf>` (also `-u 2048` matched, `--ctx`,
  `--turn P,G`)
- qwen35 CPU test:
  `cargo test -p infr-llama --release qwen35::tests::greedy -- --ignored --nocapture`
  (env: `Q35_DBG=1` per-layer norms; `Q35_NOLIN=1`/`Q35_NOATTN=1` bisect)
- Profiling: `INFR_PROF2=1` (per-op GPU timestamps), `INFR_PROF=1` (record vs
  submit+wait).
