# Re-landing the fork's features

The 2026-11-17 sync (`092c6ff2`) made `main` byte-identical to `kryptic-sh/infr`,
which dropped every fork-only feature from the tree. They live in
`backup/pre-upstream-merge-1117` (`ae92380b`) and the fork's branches. This is
the checklist for putting them back, in slices, on top of upstream's code.

To rebuild the list of commits still to replay:

    git cherry HEAD backup/pre-upstream-merge-1117

A `+` line is a commit whose patch is not in `HEAD` by patch id. Resolutions
that differ from the original commit also show as `+`, so check the subject
before re-applying one.

## Landed again

- `fix(vulkan): size rope grids in workgroups, not threads`
- `feat(server): per-request thinking switch`
- `feat(vulkan): faster f32 and bf16 decode GEMVs` (`linear_f32_mo`)
- `feat(vulkan): fp32 K-quant decode GEMV for GPUs without fast int8 dot`
  (`native_gemv_kq_fma`, `kernels.vulkan.kq_fma_gemv`, `Capabilities::i8_dot_fast`)
- `feat(vulkan): wave64-native subgroup decode GEMV` (`SG=64` twins,
  `kernels.vulkan.wave64_native`, `kernel_sg64_native`)
- `docs: describe the fork's focus on older Vulkan GPUs`

## Skipped as superseded by upstream

Re-check each before dropping it permanently:

- `feat(vulkan): split the first forward on slow discrete GPUs` — upstream has
  its own decode/prefill submit caps and `EarlySubmit`.
- `refactor(vulkan): generic kq_fma names` / `style: cargo fmt the wave64-native
  SG64 twin generation` / `fix(core): keep the norm's input unchanged...` —
  the latter targets the rmsnorm fusion pattern upstream removed; it comes back
  only with the fusion slice below.
- `feat(llama): support per-layer-missing attn_k/attn_k_norm`,
  `fix(llama): don't hard-error on missing attn_k for KV-owning layers` —
  upstream loads `attn_k` behind `has_wk[l]`.
- `feat(vulkan): fp32 K-quant prefill GEMM`, `feat(vulkan): fp32 Q4_0 prefill
  GEMM` — the prefill half of the GCN tier; the decode half is back. The
  prefill tier needs `native_gemm_kq_fma.comp` ported onto upstream's
  `quant_linear`.

## Still to re-land

Each item names the original commits; replay them with `git cherry-pick -x`
and port where upstream rewrote the same code.

1. **`infr perplexity`** — `0be02d59`, `b8c11f39` (perplexity CLI + Readback
   verify logits + 128-row pieces).
2. **Gemma 4 MTP** — `e8fa1f30` (gemma4-assistant draft model on Vulkan).
3. **EXL3** — `b6340afd` (scalar trellis oracle), `b0751bee` (`DType::I16` +
   `infr-safetensors`), `d4a02f11`, `637027f4`, `17eea5b7`, `1b9668c9`,
   `9a8a1a28`, `bf085b70`, `2daba2e3`, `fda51777`, `ed6be7b5`, `e8b1797d`,
   `39adbfcf`, `b6000ca5`, `c3aca059`, `2f3b499d`, `5aa05850`, and the
   `exl3_*` shaders/tests. Requires the `WeightSource`-threaded `SeamModel`
   (`Box<dyn WeightSource>` over `Gguf`/`Exl3Source`/safetensors).
4. **Qwen4 reference path** — `a918a027`, `d941c4b3`, `035223f8`.
5. **`infr ls` / `infr rm`** — `f5efb0ee`, `a77a3eb3` (+ `infr-hub`'s
   `manage` module, deleted in the sync).
6. **`/v1/completions`** — `dbdd13b6`, `37f9ef35` (request-level logit_bias on
   the text endpoint).
7. **CLI fixes** — `75e62a07` (copyable quant tags), `c79c4034` (compare
   benchmark overrides).
8. **Chrome-trace timeline** — `5cdd7d36` (`infr-prof-rt/src/timeline.rs`).
9. **spv-hoard shader cache** — `f5081cbf` (build-time glslc output cache).
10. **KV state** — reverse paged attention (`kv.store_mb`: `473802cb`,
    `3ed93354`, `86081d6a`, `deae48d9`, `6c1b0fd8`) and forward paged
    attention (`kv.paged`: `b7626c9a`, `e65f1601`, `f9534310`, `d17b073a`,
    `fe3199ce`, `dacc0b94`, `8107896e`, `b9f3e6dd`, `a6e17ac3`, `c179fc2f`,
    `f610fd85`). Upstream's `SeamKv` now has `cbufs`/`written_end`; the fork's
    `Arc<dyn Buffer>` + `paged: Option<PagedSlot>` graft onto it.
11. **GCN attention/perf** — `4545166e` (GQA decode), `1b498965` (split-K
    combine), `53adff6e` (rope grid-stride), `eea45032` (mmv OUTS4),
    `102397eb` (paged prefill attention), `4708c789`, `05eb5b55` (small-turn
    attention, K-quant linears), `22a2c948` (split-K prefill grid),
    `97336091`, `d6b4807b`, `4bb53713`, `d6b4807b` (bandwidth `bw_scan` tests).
12. **`weights_as_ssbo`** — `16a11a3d`, `4312ef93`.
13. **Fused RmsNorm consumers** — the core pattern is `infr-core/src/fusion.rs`
    `plan_rmsnorm_linear` plus `FusionPlan::rmsnorm_linear`/`moe_add`; needs a
    Vulkan (and Metal) planner again, and `446eb403`'s input-unchanged fix.
14. **Gemma 4 vision** — `31041145` + the `vision.rs`/`media.rs` modules and
    `serve.mmproj` / `serve.image_max_tokens`.
15. **Systemd socket activation / idle exit** — `ae92380b` +
    `infr-plat/src/socket.rs`, `infr-server/src/idle.rs`.
16. **Seed-Coder pre-tokenizer** — `53072acd` (re-applied in the sync's first
    attempt, dropped with the reset).
17. **README fork section, `docs/legacy-gpus.md` measurements, `justfile`** —
    `a8af110d`, `e71e8236`, `c265a5a6`, `ac60a3d6`.

## Landed in this pass but not yet exercised on hardware

The GCN tiers (`kq_fma_gemv`, `wave64_native`) build and pass the host-side
tests; they still need the RX 580 run that `docs/legacy-gpus.md` documents.
