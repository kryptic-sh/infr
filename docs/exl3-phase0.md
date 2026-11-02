# EXL3 Phase 0 — CPU-only quality gate (trellis decode port + perplexity)

Source of truth: issue #6, plan comment `6562674888` (supersedes the
bandwidth-only plan). Worktree `exl3-phase0-perplexity`, base = `main` @
`19c3d6a` (verified `HEAD == origin/main`; `git log` tail recorded below).
Hardware: RX 580 only — this phase is CPU-only by design. Nothing here touches
PR #12 or the `wave64*` experiment branches.

## Upstream reference (MIT, committed pin)

- Repo: `turboderp-org/exllamav3` (canonical; NOT `turboderp/exllamav3`), MIT.
- Pin: `bd1f519172406a9560bd5c90e61000691bb69a07` (v1.6.6, 2026-10-31).
- Files ported (full paths in-tree):
  - `exllamav3/exllamav3_ext/quant/exl3_dq.cuh` (329 lines)
  - `exllamav3/exllamav3_ext/quant/codebook.cuh` (175 lines)
  - `exllamav3/exllamav3_ext/rocm/quant/ptx_portable.cuh` (`exl3_lop3_6a` fallback)
- Consulted, NOT ported (Phase 1+ scope): `quant/reconstruct.cu`
  (`reconstruct_tile`: tensor-core shuffle + Hadamard), `quant/quantize.py`
  (`cb = 2 if mul1 else 1 if mcg else 0`; `codebook_mcg_mult = 0xCBAC1FED`,
  `codebook_mul1_mult = 0x83DCD12D`), `tests/testlib/exl3.py` (layout:
  `trellis (k//16, n//16, 16*K)` int16; markers: `.mcg`/`.mul1` or neither =
  3inst), `modules/quant/exl3.py` (`mcg/mul1 = marker is not None`).

## What was built (this worktree, CPU-only)

- `crates/infr-cpu/src/exl3.rs` — scalar trellis decoder. One 16-bit window →
  one f32 via `decode_3inst` for cb0 (`3inst`, 2 bpw), cb1 (`mcg`, 3 bpw),
  cb2 (`mul1`, 4 bpw); `decode_tile` (funnel-shift path mirroring the CUDA
  `dq*` extractors) + `decode_tile_oracle` (independent bit-slice path).
  Half arithmetic is bit-exact vs hardware (`__hadd`/`__hfma`, RN-even; the
  `half` 2.x `mul_add` gap is closed with a documented exact `f64` detour).
  Integer rates 1..=8; half-integer `dq8_half` tiling NOT ported (GPU-kernel
  detail, no Phase-0 consumer). Full-tensor layout (16x16 tiles, Hadamard +
  `suh`/`svh`) is Phase 1 scope — documented in the module header.
- `crates/infr-cpu/tests/exl3_parity.rs` — bit-parity in the `mmv_*_parity`
  style: funnel vs slice legs agree BIT-FOR-BIT over rates 1..=8 x all
  codebooks x 3 seeds; seeded hand-derivation spot-check; exhaustive
  0..=0xFFFF codebook totality (all finite).
- `crates/infr-cpu/examples/exl3_probe.rs` — real-checkpoint validation:
  parses any EXL3 `.safetensors`, auto-detects codebook from markers, decodes
  sampled tiles per `*.trellis` tensor, reports mean/std/min/max/finite%.

## Oracle results (CPU)

Raw commands (all ran on this box, CPU-only, worktree root):

```sh
git log --oneline -3            # base record: 19c3d6a == origin/main == main
cargo test -p infr-cpu --lib exl3          # 6/6 unit goldens
cargo test -p infr-cpu --test exl3_parity   # 3/3 parity
cargo run -p infr-cpu --example exl3_probe -- /tmp/exl3phase0/qwen05b-exl3.safetensors 2
cargo test -p infr-cpu          # full crate: 98 + 6 + 3 pass, clippy clean
cargo fmt -p infr-cpu --check   # clean
```

- `cargo test -p infr-cpu --lib exl3` — 6/6 pass (lop3 + hadd + cb2-zero
  goldens, funnel≡slice alignments, pair≡scalars, finiteness everywhere).
- `cargo test -p infr-cpu --test exl3_parity` — 3/3 pass.
- Real bytes: `Volko76/Qwen2.5-Coder-0.5B-Instruct-exl3` (`model.safetensors`,
  555 MB, K=4/cb0 linears + K=6/cb0 `lm_head`, no markers = 3inst):
  `cargo run -p infr-cpu --example exl3_probe -- model.safetensors 2` →
  **169/169 tensors, 338 tiles, 86,528 values, 0 non-finite**; per-tensor
  mean ≈ 0, std ≈ 1.0–1.2, range ±4 (sane pre-Hadamard coefficient spread).

## Perplexity (wikitext-2-raw test, ctx 512, stride 512)

| arm | model | harness | PPL |
|---|---|---|---|
| Q4_0-QAT (4.50 bpw) | `google/gemma-4-E4B-it-qat-q4_0-gguf` (`gemma-4-E4B_q4_0-it.gguf`, 5.15 GB, cached) | `llama-perplexity -c 512 -t 12`, first 300 lines (`wt2-subset-300.txt`, 43 chunks) | **28.10 ± 1.02** |
| Q4_0-QAT full-test | same | same, full test.raw (4358 lines) | *running — see raw log* |
| EXL3 cb2 (4.0 bpw) | `turboderp/gemma-3-27b-it-exl3` @ `3.0bpw`… | — | **BLOCKED (needs CUDA)** |
| EXL3 cb1 (3.0 bpw) | `turboderp/gemma-3-27b-it-exl3` @ `4.0bpw`… | — | **BLOCKED (needs CUDA)** |

Same-model triple note: the ONLY public triple sharing one base model is
Gemma-3-27B-IT (`google/gemma-3-27b-it-qat-q4_0-gguf` + turboderp EXL3
`3.0bpw`/`4.0bpw` branches). No 1B/4B/7B model has all three arms (small EXL3
quants don't exist; QAT GGUFs are Gemma-only). 27B × 3 arms is not runnable
CPU-only here (~38 GB downloads + hours of CPU PPL per arm), and EXL3
inference/conversion (`convert.py`, `eval/qbench.py`) requires a CUDA GPU —
`exllamav3` has no CPU path (its ROCm path needs gfx9+, not Polaris gfx803).

Repro commands (exact):
```sh
# Q4_0-QAT (ran here, RX 580 box, CPU-only):
/var/home/cappy/Projects/llama.cpp/build/bin/llama-perplexity \
  -m ~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-gguf/snapshots/4b4a2c1d584be7264f87aac328a1bc739ce81b6c/gemma-4-E4B_q4_0-it.gguf \
  -f /tmp/exl3phase0/wiki/wt2-subset-300.txt -c 512 -t 12 --no-perf
# EXL3 arms (needs a CUDA box; NOT run here):
git clone https://github.com/turboderp-org/exllamav3 && cd exllamav3
git checkout bd1f519172406a9560bd5c90e61000691bb69a07
huggingface-cli download turboderp/gemma-3-27b-it-exl3 --revision 3.0bpw --local-dir models/gemma3-27b-3bpw
huggingface-cli download turboderp/gemma-3-27b-it-exl3 --revision 4.0bpw --local-dir models/gemma3-27b-4bpw
python eval/qbench.py -m models/gemma3-27b-3bpw   # + same for 4bpw; compare vs QAT PPL above
```

## Gate verdict: NO-GO (STOP — no shader work)

The feature gate is Phase 0's perplexity: *"If cb1 does not clearly beat
Q4_0-QAT, stop — do not port anything."* No cb1-vs-QAT evidence can be produced
on this CPU-only box (EXL3 stack is CUDA-only), so the gate is NOT satisfied.
Per the plan, that is a STOP: **no GCN4 shader work started, none planned.**
The decode port + oracle above stand alone as the reusable Phase-1 input.

## Base record (`git log --oneline -3` at work time)

- `19c3d6a` feat(llama): support per-layer-missing attn_k/attn_k_norm (Gemma QAT GGUFs) [= origin/main = main]
- `fb1fc29` docs: describe the fork's focus on older Vulkan GPUs
- `af7df09` fix(core): keep the norm's input unchanged for fused RmsNorm consumers

Worktree diff vs base (new files only + 2 wiring lines): `crates/infr-cpu/src/exl3.rs`,
`crates/infr-cpu/tests/exl3_parity.rs`, `crates/infr-cpu/examples/exl3_probe.rs`,
`docs/exl3-phase0.md`; `src/lib.rs` (`pub mod exl3;`), `Cargo.toml` (dev-dep `serde_json`).
