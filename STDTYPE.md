# STDTYPE.md — GGUF dtype coverage (match llama.cpp)

Implementation plan to add the GGUF weight quant types that llama.cpp supports
but `infr` does not yet. **Scope: GGUF data formats only** (safetensors is
explicitly out of scope for this work). Reference implementation is the local
llama.cpp checkout at `~/Projects/mxaddict/llama.cpp` — port its dequant math,
do not re-derive it.

---

## 0. Current state vs target

`infr` today (authoritative: `crates/infr-gguf/src/lib.rs::ggml_type_to_dtype`):

- **Parsed + GPU in-kernel dequant:** `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`.
- **Parsed, host-dequant only (no GPU kernel):** `Q5_0`, `Q5_1` — _actually
  these currently parse but `load_f32` does NOT handle them, so they error on
  load. Fixing that is Phase 1._
- **Float:** `F32`, `F16`, `BF16` (all paths).

Target = the usable weight quants in llama.cpp's `enum ggml_type` (ggml.h):

| Group        | Types                                                                | Status                     |
| ------------ | -------------------------------------------------------------------- | -------------------------- |
| legacy round | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0                                         | Q8_0 ✅; rest ❌           |
| k-quant      | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K                                         | Q4/5/6_K ✅; Q2_K, Q3_K ❌ |
| i-quant      | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS | all ❌                     |
| ternary      | TQ1_0, TQ2_0                                                         | all ❌                     |
| fp4          | MXFP4, NVFP4                                                         | all ❌                     |

`Q8_1` / `Q8_K` are intermediate (activation/dot-scratch) types, not weight
storage — **out of scope**. `F64` / `I8/I16/I32/I64` — out of scope (not LLM
weights). `NVFP4` is newest/rarest — do it last, OK to defer if the block layout
is unclear in this checkout.

---

## 1. Architecture: the two consumption paths

`infr` consumes a quantized tensor in two ways. Every new type must pick a path.

### Path A — host dequant to f32 (`crates/infr-llama/src/lib.rs`)

`load_f32(g, name)` returns `Vec<f32>`. Used by **qwen35** (every projection)
and as the universal fallback. Its quant branch (line ~260) currently calls
`dequant_unified` for `is_quant(d)`. **Any** new quant must be dequantable here.

### Path B — GPU in-kernel dequant ("keep quantized in VRAM")

Only for **affine** quants that fit the unified form
`weight = scale·index + min` with `(scale, min)` constant per 16- or 32-element
block. Built by `dequant_unified` → `pack_unified` → `Wt::Q`, run by
`LINEAR_Q_WGSL` (in `crates/infr-vulkan/src/linear.rs`). Used by the **dense
Llama** path (`upload_wt`, line ~495).

**Decision rule for each new type:**

- **Affine** (Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K): fits the unified form →
  **both paths** (host f32 + GPU in-kernel). Best memory + speed.
- **Codebook / non-affine** (all IQ\*, TQ\*, MXFP4, NVFP4): does NOT fit
  `scale·index+min` (grid lookups, sign packing, fp4 codebooks) → **host dequant
  to f32 only**, then stored on GPU as **f16** (via the existing f16 weight
  path). Correct + runs; not VRAM-optimal. Native GPU kernels for these are an
  explicit follow-on (Phase 10), not required for llama.cpp _coverage_.

This split is the key design point: affine quants get the fast in-kernel path;
codebook quants get correctness via host→f16. qwen35 (which only uses Path A)
gets every new type for free the moment `load_f32` handles it.

---

## 2. Exact integration points (files + functions)

When adding a type, touch these in order:

1. `crates/infr-core/src/tensor.rs` — `enum DType`: add the variant. Update
   `is_quant()`, `dense_bytes()` (quants return `None`).
2. `crates/infr-gguf/src/lib.rs`:
   - `ggml_type_to_dtype()` — add the `ggml_type` code → `DType` arm.
   - `block_layout()` — add `(elements_per_block, bytes_per_block)`. **Take
     these from llama.cpp's `type_traits[]` in `ggml/src/ggml.c`** (`.blck_size`
     and `.type_size`) — do not guess.
3. `crates/infr-llama/src/lib.rs`:
   - Host dequant: extend `dequant_unified` (affine) OR add a codebook dequant
     fn; wire into `load_f32`'s match.
   - `is_quant()` (line ~444) — affine types that go GPU-in-kernel; keep this
     the **GPU-affine** set.
   - `quant_params()` (line ~451) — `(bits, scale/min block size)` for affine
     GPU types.
   - `dequant_unified` (line ~341) — add affine arm producing
     `(u8 index, scale, min)` per element.
   - `upload_wt` (line ~495) — route codebook quants to host-dequant→f16 (NOT
     `Wt::Q`). Add an `is_codebook_quant()` helper or invert: `Wt::Q` iff
     `is_gpu_affine_quant`, else if any-quant → dequant f32 →
     `upload_weight_f16`, else `f16_bytes` path.
4. `crates/infr-vulkan/src/linear.rs` — only if a type needs a NEW GPU kernel
   (Phase 10). Affine types reuse `LINEAR_Q_WGSL` unchanged.

**The unified affine form** (`weight = scale·index + min`, `(scale,min)` const
per block) for the new affine types — derive from the llama.cpp dequant fns:

- **Q4_0** (`d·(q-8)`, q∈0..15): scale=`d`, index=`q`, min=`-8d`; block 32; GPU
  4-bit index.
- **Q4_1** (`d·q + m`): scale=`d`, index=`q`, min=`m`; block 32; GPU 4-bit.
- **Q5_0** (`d·(q-16)`, q∈0..31): scale=`d`, min=`-16d`; block 32; GPU 8-bit
  (index>15).
- **Q5_1** (`d·q + m`, q∈0..31): scale=`d`, min=`m`; block 32; GPU 8-bit.
- **Q2_K** (per-16 sub-scale/min, q∈0..3): scale=`d·sc`, min=`-dmin·m`; block
  16; GPU 4-bit. (Same shape as Q4_K but 2-bit quants — mirror the Q4_K arm.)
- **Q3_K** (per-16 6-bit scale, q∈0..7, symmetric −4 offset): scale=`d·sc`,
  index=`q`, min=`-4·d·sc`; block 16; GPU 4-bit.

---

## 3. llama.cpp reference map (read these)

All paths under `~/Projects/mxaddict/llama.cpp`:

- **Block structs:** `ggml/src/ggml-common.h` — `block_q4_0` (l.188),
  `block_q4_1` (201), `block_mxfp4` (208), `block_q5_0` (224), `block_q5_1`
  (238), `block_q8_0` (245), `block_tq1_0` (270), `block_tq2_0` (277),
  `block_q2_K` (298), `block_q3_K` (310), `block_iq2_xxs` (374), `block_iq2_xs`
  (382), `block_iq2_s` (391), `block_iq3_xxs` (400), `block_iq3_s` (411),
  `block_iq1_s` (419), `block_iq1_m` (427), `block_iq4_nl` (441), `block_iq4_xs`
  (449).
- **Dequant math:** `ggml/src/ggml-quants.c` — `dequantize_row_q4_0` (l.401),
  `_q4_1` (421), `_q5_0` (442), `_q5_1` (468), `_q2_K` (903), `_q3_K` (1247),
  `_tq1_0` (2356), `_iq2_xxs` (2416), `_iq3_xxs` (2503), `_iq1_s` (2578),
  `_iq4_nl` (2653), `_iq4_xs` (2671). (Find the rest by
  `grep -n "dequantize_row_<type>" ggml/src/ggml-quants.c`.)
- **i-quant grids (copy verbatim):** `ggml/src/ggml-common.h` — `iq2xxs_grid`
  (l.550, 256×u64), `iq2xs_grid` (617, 512×u64), `iq3xxs_grid` (1007, 256×u32),
  `kvalues_iq4nl` (1110, 16×i8), `iq1s_grid` (1124). Port these as `const`
  arrays in a new `crates/infr-llama/src/iquant_grids.rs`.
- **Block sizes:** `ggml/src/ggml.c` `type_traits[]` — authoritative
  `.blck_size` / `.type_size` per type for `block_layout()`.

---

## 4. Phases (each: build + `cargo clippy` + `cargo fmt` + test, then commit)

Keep commits per-phase, Conventional Commits, no attribution. Phases are
**sequential** (all touch `DType` / `ggml_type_to_dtype` / `load_f32`) — do NOT
parallelize; merge conflicts otherwise.

- **Phase 0 — plumbing.** Add ALL target `DType` variants + `ggml_type_to_dtype`
  codes + `block_layout` sizes. No dequant yet — a model of a new type should
  now _parse_ (and error cleanly on load with "dtype X not yet dequantable").
  `cargo build` green. Commit
  `feat(gguf): parse all llama.cpp quant type codes`.
- **Phase 1 — legacy affine (Q4_0, Q4_1, Q5_0, Q5_1).** Host dequant in
  `load_f32` + `dequant_unified` affine arms + `quant_params` + `is_quant` + GPU
  path. Test: unit tests with hand-built blocks (known f32); if a real GGUF in
  one of these quants is available, `infr run` vs `llama-cli`.
- **Phase 2 — Q2_K, Q3_K.** Same dual path (mirror Q4_K/Q6_K arms).
- **Phase 3 — GPU-affine validation.** Confirm Phases 1–2 types run through
  `LINEAR_Q_WGSL` (extend the `recorder::tests::linear_q_matches_cpu` test to
  the new types). This is the "keep-quantized-in-VRAM" proof.
- **Phase 4 — IQ4_NL, IQ4_XS.** Simplest i-quants (16-entry `kvalues_iq4nl`
  codebook, no grid). Host dequant → f16. Add `is_codebook_quant`, route in
  `upload_wt`.
- **Phase 5 — IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S.** Grid-based. Port grids
  to `iquant_grids.rs`. Host dequant → f16.
- **Phase 6 — IQ1_S, IQ1_M.** 1-bit grid i-quants. Host dequant → f16.
- **Phase 7 — TQ1_0, TQ2_0.** Ternary. Host dequant → f16.
- **Phase 8 — MXFP4 (+ NVFP4 if layout clear).** fp4 e2m1 codebook + e8m0 scale.
  Host dequant → f16.
- **Phase 9 — dense `upload_wt` audit.** Ensure every quant either takes the
  GPU-affine path or the host→f16 path; no quant reaches `f16_bytes` (which only
  handles F16/F32/BF16). Add a fallthrough test that loads a tiny model of each
  available quant.
- **Phase 10 (optional, later) — native GPU kernels for codebook quants.** Only
  if VRAM/perf demands it. Not required for coverage. Out of scope unless asked.

---

## 5. Testing strategy

1. **Unit tests (primary, no model needed):** for each quant, hand-construct one
   block with known field values, dequant, assert exact f32. Mirror the existing
   `recorder::tests::linear_q_matches_cpu` style. This is the gold check and
   needs no downloads.
2. **Cross-check vs llama.cpp (when a real GGUF exists):** pull a small model in
   the target quant (`infr pull hf:...`), run `infr run <blob> "..."` and
   compare greedy output to `llama-cli -m <blob> -ngl 0 --temp 0 -st -p "..."`.
   The oracle. (qwen35 path = `Q35_CPU=1` for pure-CPU determinism.)
3. **Validate the i-quant grids** by dequanting a known block and matching
   llama.cpp's `dequantize_row_*` element-for-element (port a few expected
   values from a tiny C harness or from a real tensor).

Per the repo rules: run `cargo clippy`, `cargo fmt`, `cargo test` after each
phase. `TMPDIR=$HOME/.cache/tmp` for shell work (`/tmp` is a small RAM tmpfs).
Build CLI with `cargo build -p infr-cli --release`.

---

## 6. Definition of done

- Every type in the §0 target table parses, dequants to correct f32 (unit-tested
  against llama.cpp math), and loads via `infr run` (qwen35) and the dense Llama
  path without error.
- Affine quants (Q4_0/Q4_1/Q5_0/Q5_1/Q2_K/Q3_K) run through the GPU in-kernel
  `LINEAR_Q_WGSL` path (keep-quantized-in-VRAM), validated vs CPU.
- Codebook quants (IQ\*/TQ\*/MXFP4) load via host-dequant→f16, validated vs the
  llama.cpp dequant reference.
- `cargo clippy`/`fmt`/`test` clean. Each phase committed separately.
- This file updated with a short "DONE" note per phase as it lands.
