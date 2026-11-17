# Re-landing the fork's features

The 2026-11-17 sync (`092c6ff2`) made `main` byte-identical to `kryptic-sh/infr`,
which dropped every fork-only feature from the tree. They live in
`backup/pre-upstream-merge-1117` (`ae92380b`) and the fork's branches. This
records what came back, what upstream superseded, and the few things that
stayed out.

To see what is still missing, compare patches:

    git cherry HEAD backup/pre-upstream-merge-1117

A `+` line is a commit whose patch is not in `HEAD` by patch id. Resolutions
that differ from the original commit also show as `+`, so check the subject
before re-applying one.

## Landed

Replayed onto upstream's code, which meant re-expressing each one against
upstream's seams (`SeamModel` over `Box<dyn WeightSource>`, `Linear`/`Op`
enums, `Recorder` kernel tables, the BDA-only KV stores):

- **Server** — per-request thinking switch (`reasoning_effort`,
  `chat_template_kwargs`, `think`), `/v1/completions` with request-level
  `logit_bias`, idle exit and systemd socket activation, body limit sized for
  inline images, §auth on the headers.
- **EXL3** — the scalar trellis oracle, `DType::I16` + `infr-safetensors`,
  `Exl3Source` behind `WeightSource`, the native compressed projection paths
  (`Op::LinearExl3`), Qwen2/Qwen3.5/Qwen4-Exp/Gemma4 loaders, and the
  `exl3_*` shaders with the fp16 decode GEMV.
- **Vulkan** — `linear_f32_mo` + `float_gemv_bw`, `native_gemv_kq_fma` decode
  and its multi-row verify twin, wave64-native `SG=64` subgroup decode GEMV,
  the fp32 K-quant prefill GEMM (`NcGemm::KqFma`, split-K included),
  `weights_as_ssbo`, the spv-hoard build cache, the `bw_scan` bandwidth test,
  Chrome-trace timeline, GQA decode pass 1, KV paged reads (`kv.store_mb`),
  forward paged attention in the seam (`kv.paged`: `KvPool`/`PagedSlot`, a
  pool at the first fork or at boot, page-mapped caches in every graph).
- **Models** — Gemma 4 MTP draft, Gemma 4 text + row embeddings, Gemma 4
  vision (`serve.mmproj`, `serve.image_max_tokens`), Qwen4 reference path,
  Seed-Coder pre-tokenizer.
- **CLI / docs** — `infr perplexity`, `infr ls` / `infr rm`, copyable quant
  tags, `docs/legacy-gpus.md`, the README fork section, `justfile`.

## Skipped as superseded by upstream

- `feat(vulkan): split the first forward on slow discrete GPUs` — upstream has
  its own decode/prefill submit caps and `EarlySubmit`.
- `refactor(vulkan): generic kq_fma names` / `style: cargo fmt the wave64-native
  SG64 twin generation` / `fix(core): keep the norm's input unchanged...` —
  the latter targets the rmsnorm fusion pattern upstream removed.
- `feat(llama): support per-layer-missing attn_k/attn_k_norm`,
  `fix(llama): don't hard-error on missing attn_k for KV-owning layers` —
  upstream loads `attn_k` behind `has_wk[l]`.
- `feat(vulkan): fp32 Q4_0 prefill GEMM` — upstream's `quant_linear` covers it.

## Not re-landed

See `docs/backlog.md` (B-RELAND) for what each one needs:

- the store kernels' params-based row base (`p_rb`).
- split-K prefill attention (`22a2c948`), small-turn attention + K-quant
  linears (`05eb5b55`), mmv OUTS4 K-tail (`eea45032`), compare benchmark
  overrides (`c79c4034`), fused RmsNorm consumers (the `plan_rmsnorm_linear`
  planner no longer exists in core).

## Landed in this pass but not yet exercised on hardware

The GCN tiers (`kq_fma_gemv`, `wave64_native`) build and pass the host-side
tests; they still need the RX 580 run that `docs/legacy-gpus.md` documents.
