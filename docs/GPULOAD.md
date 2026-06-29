# GPULOAD.md — native-block GPU weight loading

> **Status (2026-06-29): NOT started — still a valid pending plan.** The Vulkan
> loader continues to host-dequant + upload (f16 / repacked quant). This doc is
> the design for moving that in-kernel. Note the new CPU reference backend
> already keeps weights in their native GGUF dtype and dequants lazily on read
> ([root `PLAN.md`](../PLAN.md), "dtype-aware weights") — the same principle,
> applied to the GPU loader, is what this plan describes.

Plan to upload **raw GGUF quant blocks to VRAM and dequantize in-shader**,
instead of the current host-side dequant + repack. Eliminates host processing
for the GPU-consumed weights → faster model load (host dequant currently
dominates startup), native (smallest, bit-exact) VRAM footprint, and i-quants
that run in-kernel instead of being blown up to f16. Reference: llama.cpp's
`ggml-vulkan` per-quant dequant shaders at `~/Projects/mxaddict/llama.cpp` —
port their math (GLSL → our WGSL).

---

## 0. Goal & honest scope

**In scope:** the GPU matmul weights — the dense Llama `Wt` path (and optionally
the qwen35 `Lin` GPU path). Upload the raw block bytes; decode them inside the
GEMV/GEMM shader.

**Explicitly NOT "zero host" yet:** two consumers still need host f32 and are
out of scope here (each is its own follow-on):

- **Embedding gather** — runs host-side (`token_embd[tok·ne..]`). Needs a GPU
  gather kernel to go raw. (Optional Phase 8 below.)
- **qwen35 CPU SSM + the norms it applies on CPU** — need host f32 until the SSM
  itself is a GPU kernel (tracked separately).

So this plan makes **weights** fully GPU-loaded; true 100%-zero-host is
weights + GPU-gather + GPU-SSM.

---

## 1. Current state (what we're replacing)

`crates/infr-llama/src/lib.rs`:

- `upload_wt` (the dense loader): for affine quants it runs `dequant_unified`
  (host decode → `u8 index + scale + min`) → `pack_unified` (repack to a uniform
  GPU layout) → uploads **3** buffers as `Wt::Q { q, s, m, bits, blk_shift }`.
  Codebook quants are host-dequanted to f16 → `Wt::F16`. Floats → `Wt::F16`.
- `Wt` enum (l.~75): `F16(buf)` | `Q { q, s, m, bits, blk_shift }`.
- Forward dispatch (`forward_resident_kv`, l.~2087): `Wt::F16 → rec.linear`,
  `Wt::Q → rec.linear_q` / `linear_add_q` / `ffn_in_q`; prefill uses
  `matmul_proj_mmq` (dp4a int8 GEMM, **u4-only**).

`crates/infr-vulkan/src/`:

- `linear.rs::LINEAR_Q_WGSL` — the unified-affine dequant GEMV (reads the
  repacked `q`/`s`/`m` layout, NOT native blocks).
- `recorder.rs` — `linear_q` (l.434), `linear_add_q` (489), `ffn_in_q` (636),
  `matmul_proj_mmq` (367).

Floats are already near-raw: f16-source passes through (`f16_bytes`), bf16 is
stored native (`upload_weight_bf16`). The work here is the **quant** formats.

---

## 2. Target architecture (native-block path)

1. **`Wt::Native { buf, dtype }`** — a single VRAM buffer holding the tensor's
   raw GGUF block bytes, uploaded zero-copy from the mmap
   (`g.tensor_bytes(name)` → `be.upload_weight_bytes(bytes)`), tagged with its
   `DType`. No host decode, no repack.
2. **Per-format in-shader dequant.** A WGSL function `dequant_<fmt>(buf, elem)`
   that reads element `elem`'s value from its native block (locate block =
   `elem / blck_size`, decode the field bytes). Port from the matching
   `dequant_<fmt>.comp` / `dequant_funcs.glsl`.
3. **Format-specialized GEMV/GEMM.** Mirror `LINEAR_F16_WGSL`'s cooperative-
   over-K layout (one workgroup per output, 64 threads stride K, tree-reduce);
   replace the single weight-read line `f32(w_buf[wbase+i])` with
   `dequant_<fmt>(w_buf, wbase+i)`. One WGSL string per format (cache per format
   via the existing `kernel()` map). The buffer is bound as `array<u32>` and the
   shader does the byte/bit extraction.
4. **Grids as SSBOs (i-quants).** Upload `iq2xxs_grid` etc. **once** (shared,
   device-lifetime) as read-only storage buffers, bound as an extra binding to
   the i-quant shaders. NOTE: grids are `u64`/`u32`; **WGSL has no 64-bit int**
   — store each `u64` grid as two `u32` words and reconstruct the 8 signed bytes
   from the pair. This is the main complication for Phases 4–5.

The dispatch in `forward_resident_kv` gains a `Wt::Native` arm that picks the
format's GEMV. `LINEAR_Q_WGSL` / `pack_unified` / `dequant_unified` stay only as
the CPU-reference/oracle once `Native` is the default (Phase 7).

---

## 3. Reference map (llama.cpp, read these)

Under `~/Projects/mxaddict/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`:

- Per-format dequant shaders (native-block reads, the port source):
  `dequant_q8_0.comp`, `dequant_q4_0.comp`, `dequant_q4_1.comp`,
  `dequant_q5_0.comp`, `dequant_q5_1.comp`, `dequant_q2_k.comp`,
  `dequant_q3_k.comp`, `dequant_q4_k.comp`, `dequant_q5_k.comp`,
  `dequant_q6_k.comp`, `dequant_iq4_nl.comp`, `dequant_iq4_xs.comp`,
  `dequant_iq2_xxs.comp`, `dequant_iq2_xs.comp`, `dequant_iq2_s.comp`,
  `dequant_iq3_xxs.comp`, `dequant_iq3_s.comp`, `dequant_iq1_s.comp`,
  `dequant_iq1_m.comp`, `dequant_mxfp4.comp`, `dequant_nvfp4.comp`.
- Shared decode helpers + block field access: `dequant_funcs.glsl`,
  `dequant_head.glsl`. How blocks feed matmul: `mul_mm.comp`,
  `mul_mm_funcs.glsl`.
- Block byte layouts: `ggml/src/ggml-common.h` (block structs — same ones listed
  in `STDTYPE.md §3`). Block sizes: `ggml/src/ggml.c` `type_traits[]`.
- We already have correct **host** dequant for every format in
  `crates/infr-llama/src/lib.rs` (`dequant_unified`, `dequant_codebook`) and the
  verified grid tables in `iquant_grids.rs` — use these as the WGSL port's
  cross-check oracle (the GPU result must match the host dequant element-wise).

---

## 4. Phases (each: build + `cargo clippy` + `cargo fmt` + test, commit; do NOT push)

Sequential. **Phases 0–3 (affine quants) are the must-have, high-confidence,
fully-testable core.** Phases 4+ (i-quant/fp4 native, with the WGSL u64 grid
workaround) are higher-risk — only proceed if each validates.

- **Phase 0 — `Wt::Native` infrastructure.** Add the enum variant + raw upload
  in `upload_wt` (gate behind `INFR_NATIVE` env initially so the existing path
  stays default) + a `forward_resident_kv` dispatch arm. Prove the plumbing with
  ONE format end-to-end: **Q8_0** (simplest: `[f16 d][i8 qs[32]]`, `y=d·q`).
  Unit test: native GEMV vs host `dequant_unified` matvec, and vs the existing
  `Wt::Q` output — all three must match.
- **Phase 1 — legacy affine native.** Q4_0, Q4_1, Q5_0, Q5_1 in-shader dequant
  GEMV. Validate vs CPU + vs `Wt::Q`.
- **Phase 2 — k-quant native.** Q4_K, Q5_K, Q6_K, Q2_K, Q3_K. Port the
  `dequant_q*_k.comp` field decode (the 6-bit scale unpacking is the fiddly part
  — `dequant_q4_k.comp` / `dequant_q3_k.comp`). Validate vs CPU + `Wt::Q`.
- **Phase 3 — make affine native the default.** Flip `upload_wt` so all affine
  quants use `Wt::Native`; keep `dequant_unified`/`pack_unified` only as the CPU
  oracle. Re-run a dense model (`infr run` vs `llama-cli`) for at least Q4_K and
  Q6_K. **Benchmark load time** native vs old (expect a clear win). This is a
  shippable milestone on its own.
- **Phase 4 — i-quant native (grids as SSBOs).** Upload grids once as `u32`-pair
  SSBOs (WGSL u64 workaround — see §2.4). Start with IQ4_NL/IQ4_XS (16-entry
  codebook, no grid), then grid-based IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_S/M. Each:
  GPU vs host `dequant_codebook` element-wise. **Safety valve: if a grid/sign
  decode can't be validated against the host oracle, STOP — do not commit.**
- **Phase 5 — fp4 / ternary native.** MXFP4, NVFP4, TQ1_0, TQ2_0. Validate vs
  host.
- **Phase 6 — prefill GEMM + fused ops.** The above covers the decode GEMV. Add
  native variants for the prefill path: `linear_add`, `ffn_in`, and the prefill
  GEMM (`matmul_proj_mmq` is dp4a-int8, u4-only — decide: generalize it, or
  route prefill through the native GEMV and measure). Driven by perf, not
  correctness.
- **Phase 7 — retire the unified GPU path.** Remove `Wt::Q` / `pack_unified` /
  `LINEAR_Q_WGSL` from the live path once `Native` covers everything; keep host
  `dequant_unified` only inside `load_tensor_dequant` (CPU oracle / qwen35).
- **Phase 8 (optional, toward zero-host).** GPU embedding-gather kernel so
  `token_embd` uploads raw and gathers on-device. (qwen35 SSM kernel is tracked
  separately — both are needed for true 100%-zero-host.)

---

## 5. Validation strategy

1. **Per-format GPU-vs-host unit test (primary):** dequant a random/known weight
   block on the GPU GEMV and compare to the host `dequant_unified` /
   `dequant_codebook` result for the same bytes (tolerance: exact for the index
   math, f16-ish for the accumulate). Mirror
   `lib.rs::gpu_affine_tests::*_gpu_matches_cpu`. No model download needed.
2. **Native-vs-Wt::Q parity (affine):** the new native GEMV must produce the
   same logits as the current unified path for the same tensor — a strong
   regression gate during Phases 0–3.
3. **End-to-end vs llama.cpp:** for at least one real model per major format,
   run `infr run <blob> "..."` and diff greedy output against
   `llama-cli -m <blob> -ngl 99 --temp 0 -st -p "..."`.
4. **Load-time benchmark (Phase 3):** time model load native vs old; record the
   delta in this file.

Repo rules: `cargo clippy` / `cargo fmt` / `cargo test` after each phase;
`TMPDIR=$HOME/.cache/tmp` for shell work; build CLI with
`cargo build -p infr-cli --release`. Commit per phase, Conventional Commits, no
attribution, **do not push** (review first).

---

## 6. Key risks

- **WGSL has no 64-bit int** — the i-quant `u64` grids must be split into `u32`
  pairs (Phases 4–5). This is the main porting hazard; the affine phases (0–3)
  avoid it entirely.
- **Byte/endianness of native blocks** — GGUF is little-endian; reading
  `array<u32>` then masking must match the C struct field order exactly. The
  host `dequant_*` in `lib.rs` already encodes the correct field offsets — use
  them as the spec.
- **Buffer alignment** — block sizes aren't all multiples of 4 (e.g. Q6_K=210,
  Q3_K=110 bytes). Binding as `array<u32>` needs the upload padded to 4 bytes
  and the shader to index by byte within u32 words. Pad on upload.
- **Don't regress the working paths** — gate behind `INFR_NATIVE` until Phase 3;
  keep `Wt::Q` until Phase 7.

---

## 7. Definition of done

- Every GPU-supported quant loads via **raw block upload + in-shader dequant**
  (`Wt::Native`); host dequant remains only for the CPU/oracle path
  (`load_tensor_dequant`).
- GPU output matches the host dequant oracle per-format and matches the old
  `Wt::Q` path for affine quants; end-to-end output unchanged vs llama.cpp.
- Model **load is measurably faster** and quant weights occupy **native VRAM
  size**; i-quants run in-kernel (no host→f16 blow-up).
- `cargo clippy`/`fmt`/`test` clean; per-phase commits; this file updated with a
  short DONE note + the load-time benchmark per phase.
