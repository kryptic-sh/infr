# 13 — Replace the caps, refusals and open decisions with implementations

The 2026-11-12 review passes capped or refused what infr could not compute
faithfully (B146, B147, B148) and left a set of decisions open (B11, B13, B62,
B131, B137, B145 and others). This phase implements them: each slice replaces a
cap, a refusal or a standing decision with the real behaviour and a test that
can fail.

Tags: **[R]** read in code, **[I]** inferred, **[U]** unknown until run. The
designs were read against the tree at `1ac2f19` and the llama.cpp checkout at
`57291f2` on 2026-11-13; `main` has moved since, so each slice re-verifies its
sites first. Nothing below has been built unless its slice says so.

## Decisions taken (owner delegated them, 2026-11-13)

- **`infr serve` (B137, B82, B10a):** implement `response_format` JSON mode on
  the llguidance path; validate `Host` on a keyless loopback server; SSE
  keep-alive on by default; count the prompt before a slot is taken so a
  streamed over-long prompt is a real 400; a body-size key; cap a DiffusionGemma
  session at the trained context. `max_tokens: -1` stays "server default" and an
  unparseable tool-call block stays content.
- **B62:** clamp an auto-sized context first; spill only for an explicit context
  that does not fit; with `kv.overflow` on and nothing fitting, spill at the
  minimum session context rather than refuse.
- **B131 / B13:** the KV estimate becomes the sum of the real allocations, with
  the qwen35 snapshot priced and `KV_SLOP_ROWS` removed.
- **B11:** an `INFR_CACHE` percentage resolves against allocatable room.
- **B145:** the device class becomes a property of the backend.
- **B147, deepseek32 YaRN:** implement the reference's arch-keyed rule exactly
  and re-bless the synthetic fixture, with an f64 host test of an MLA layer at a
  non-unit magnitude so the new values are checked and not just re-recorded.
- **B148, lost GPU:** wait on the publish fences on the host, then submit the
  gathers. Costs one host round trip per all-reduce; measure it in a quiet
  window afterwards.
- **B68:** declined (the build-script `cfg(metal)`). It only makes the "Metal
  unavailable" arm buildable on a Mac. What blocks type-checking `infr-llama`
  for macOS from Linux is C and C++ in `tokenizers`' default features, not
  Metal.
- **B115:** a malformed `INFR_*` number is a hard error. **B110:** a bare
  `infr run org/repo` keeps using a cached quant, chosen deterministically and
  named in the log; fix the mmproj selection. **B112:** a foreign sidecar
  tokenizer falls back to the GGUF's own, with a warning. **B1:** make
  `dn_chunk_scan` mean what its name says. **B88:** keep parity with llama.cpp
  and document it.
- **B152:** fix the CPU Q5_0 decode/prefill mismatch and re-bless the gemma3
  golden against llama.cpp.

## Slices, in order

Each is independently shippable. Sizes: S, M, L.

### S1 — `infr serve` decisions — done 2026-11-21

Shipped; what it left is backlog B137.

### S2 — KV estimate equals allocation — done 2026-11-24

Shipped; what it left is backlog B13. The estimate's inputs still come from
`ubatch_candidates`, `placement_ring` and `SeamModel::kv_fit_fmts`, which S3 and
S4 touch: a change there must keep them the values the runner allocates with.

Departs from the slice as planned in one place: `kv_floor_alloc_bytes` does NOT
gain the snapshot term. The refuse rung's floor is documented, and tested, as
never refusing a run that would have worked, and the snapshot breaks that: a
slot allocates it on first use, the decode loop carries on when it is refused,
and a slot that never snapshots never allocates it. So the floor is what the
runner's cold init must allocate (`kv_init_allocs`) and the estimate that sizes
a context is that plus the snapshot (`kv_slot_allocs`); the two functions' docs
say so. The estimate also prices each buffer at what the backend says a KV
allocation of that size commits (`Backend::kv_alloc_cost`), which the plan did
not foresee.

### S3 — ladder order and the cache percentage — done 2026-11-25

Shipped; what it left is backlog B62. The two places the ladder spills (an
explicit window that does not fit, and the minimum when nothing fits) did not
reach a working run on a dense model when S3 shipped: dense placement priced the
whole cache in VRAM and the spill left later allocations no room. A follow-up
slice (2026-11-25) made both run, with one split rule and one reserve shared by
the fit, placement, the re-clamp and the allocation; on a unified-memory device
the spill is not armed, since system RAM counts in the same budget there.

The follow-up departs from the ladder as decided in one place, as the
orchestrator's call and open to the owner's veto: under `kv.overflow`, when no
cache format fits beside the weights but the weights fit without it, placement
keeps the weights resident and spills the cache, where the decided order streams
the weights first. The reasoning: spilled rows are read over PCIe in proportion
to the depth a generation reaches, while streamed weights are read again every
token whatever the depth; and the fit before the load already spills such a
window on the assumption of resident weights, so streaming them would price the
session two ways. Not measured: no run compared the two; backlog B62 has the
measurement to take. Streaming still comes first when the weights do not fit
resident at all (with the whole cache then spilled when it would leave the
streaming arenas less than their floor).

Departs from the slice as planned in two places. The pre-load rule
(`seam::preload_ctx`) covers the explicit window too, so
`SeamModel::fit_explicit_ctx` replaced `auto_q8_for_explicit_ctx` and the pure
`explicit_ctx_rung` replaced `explicit_ctx_q8_fit`; and it shares its clamp with
`seam::post_load_ctx` (`clamp_rung`), whose result enum is now `CtxRung`. The
plan's account of the old behaviour was short of what a run showed: under
`kv.overflow` the kept trained window was not cut after the load, it failed at
dense placement's streaming floor first (Qwen3-0.6B, 1880 MiB of room, both
GPUs).

### S4 — the device class into the pins — done 2026-11-25

Shipped; what it left is backlog B-PINS-CLASS. The VRAM limit knob
(`INFR_VRAM_LIMIT`) shipped first, 2026-11-24; what it found is in backlog B12,
B13, B-GUARD-POOLED and B-PRELOAD-CLAMP.

Departs from the slice as planned in one place: not every site is held to an
active scope. The CPU and Metal paths still run on the process fallback pins,
which are made for their class, so the backend-generic runner checks that the
pins in force were made for its backend's class (`debug_assert_pins_for`) rather
than that a scope is active; the Vulkan-only functions (placement, a session's
generation, the `SeamModel` fit helpers) check both (`debug_assert_scope_for`).

### S5 — position temperature and the MLA magnitude (M) — first half done 2026-11-25

The first two bullets shipped; what they left is backlog B-ATTN-TEMP. The last
two (Mistral-Large-3's temperature and the deepseek32 rule) are still to do.

Departs from the slice as planned in three places. `Op::AttnTemp`'s `offset` is
a whole number, where the reference's is a float: every builder that sets one
passes 0 or 1, and it makes the floor an exact integer division. Metal's two
decode-loop gates refuse the op as Vulkan's `decode_eligible` does
(`replay_shape`, `static_walk_live_positions`), since its position is baked in.
And the magnitude reaches deepseek32's lightning indexer too: the reference
ropes the indexer's query and key with the same magnitude, so its query's rotary
columns take the same gain, or a deepseek32 file at a magnitude other than 1
would select different keys.

- New `Op::AttnTemp { x, dst, rows, n, pos, floor_scale, scale, offset }`:
  `dst[r,:] = x[r,:] * (ln(floor((pos+r+offset)/floor_scale)+1)*scale + 1)`
  (llama.cpp `llm_graph_input_attn_temp::set_input`). CPU loop, one Vulkan
  shader, one Metal kernel. Vulkan's `decode_eligible` must refuse it
  explicitly.
- MLA YaRN magnitude: multiply only the q rope columns by m² with the existing
  `Op::MulVec` before `Op::Mla` (rope is linear, so this is exact); skipped when
  m is 1, so V2, V3, V3.2 and Kimi do not move.
- Mistral-Large-3 temperature: `Op::AttnTemp` on `mla_q`; delete its context
  cap.
- deepseek32: compute `yarn_attn_factor` with the reference's arch-keyed rule,
  drop `deepseek32_yarn_rules_agree`, re-bless the synthetic fixture.

### S6 — chunked attention, lifting the Llama 4 cap (L, three slices)

- A chunked mask (a query at p attends keys in `[floor(p/N)*N, p]`) is a subset
  of a sliding window of N, and its lower bound is non-decreasing in p, so it
  reuses the sliding-window machinery: no new kernel variants **[R]**.
- `AttnMask::Chunked(n)` with `key_lo`, `reach` and `kernel_window` (the chunk
  size with a flag bit). One `attn_lo` helper replaces the repeated lower-bound
  expression in every Vulkan and Metal attention kernel.
- Hazard: several adapter sites match `SlidingWindow(w) => *w, _ => 0`, which
  the compiler will not flag for a new variant; an un-updated kernel would read
  the flagged value as a huge window and silently run causal attention. Every
  tier gets a test against an f64 reference with a small N.
- Flash stays causal-only, so chunked prefill takes the non-flash path: correct,
  slower; teaching flash a lower bound is a later perf item.
- The emitter sends `Causal` while the whole batch is inside the first chunk, so
  every context under the old cap keeps its numerics.
- Chunked layers can ring at N + ubatch rows. Then delete the llama4 arm of
  `faithful_ctx_limit`.
- Order: core + CPU + config; Vulkan; Metal mirror + lifting the cap.

### S7 — per-tensor `.scale` tensors (M–L, two slices)

- Dense: read each scalar at load and push `Op::Scale` after the scaled `Linear`
  exactly where llama.cpp's `build_lora_mm` multiplies. A scale of 1 emits
  nothing, so existing models keep identical graphs. Do not fold into weights
  (NVFP4 block scales would lose precision).
- MoE: the output side folds into the existing `down_scale` table; the
  pre-activation side is a new optional `MoeFfn.in_scale`. Metal refuses
  `in_scale` on device and takes the host path.
- Then remove `refuse_unapplied_weight_scales`. No perf folding unless a
  measurement shows it is needed.

### S8 — expert parallelism for the dual-FFN MoE (M)

- Carry every tensor written before the MoE cut and read after it (gemma4moe's
  dense branch) as a replicated per-rank buffer; shard the `down_scale` and
  `in_scale` tables like an expert bank. Needs S7's MoE slice.
- Test: a two-rank world on one device against single-device, which runs on the
  integrated GPU alone.

### S9 — the non-coopmat GEMM past the workgroup limit — done 2026-11-25

Shipped; what it left is backlog B-EXPERT-GRID-Z.

Departs from the slice as planned in two places. The dense builds did index by
`.x` alone, but that did not make `dispatch_wide` apply as they stood: its
paired shaders rebuild the flat index from the 2-D grid, so the dense arm of
every `native_gemm_mmq_*` shader and `native_gemm_fma.comp` had to learn to (the
`EXPERT_GRID` builds compile to the same SPIR-V as before). And the
`EXPERT_GRID` grid can also pass the limit, at a user-set prefill chunk far
above the default; it is refused by name, against the limit the device reports
(`caps::KernelLimits::workgroup_count_x`), rather than split, and the split is
the backlog entry. The resident Q4_K mmq GEMM (`matmul_mmq_q4k`, the Q4_K-mmq
arm on a coopmat device) shares those shaders and had no guard at all; it takes
the split too, as does its Q6_K twin `matmul_mmq_q6k`, which only tests call.

### S10 — bounded all-reduce wait (M)

- Publish fences waited on the host in slices, then the gathers submitted, so no
  cross-device wait is ever pending; a `poisoned` flag; an error naming the lost
  rank. Test with a hook that skips one rank's publish and fakes `DEVICE_LOST`,
  asserting a bounded error and that dropping the world finishes.

### S11 — small independent slices

- **B35:** `fstat` each touched shard inside `FileBlockIo::read_block` and fail
  the request on a mismatch — done 2026-11-25. The check runs after the read, so
  a change landing during it is caught too; what it left is backlog B35.
- **B30:** keep and share the descriptor that was mapped (`Gguf::shard_handles`,
  `FileBlockIo::from_handles`, `WeightWatch::for_model`) — done 2026-11-25.
  Every reader handle is reopened from the mapped descriptor
  (`infr_plat::fileio::reopen_for_reads`, `ReOpenFile` on Windows). The CLI's
  watches still open the model a second time (`WeightWatch::open_model`); that
  is backlog B30.
- **B103:** a `dist` artifact check counting VEX against legacy-SSE
  instructions, proven able to fail with a `target-cpu=native` build.
- **B-HWDET-VENDOR-RULES:** `INFR_COOPMAT_FORCE` next to `INFR_NO_COOPMAT`.
- **B115, B110, B112, B1, B88, B152** as decided above.
- `tokenizers` without its C++ default feature, only if the vocabulary fixtures
  hold their pass counts.

## What this phase cannot verify here

Intel, NVIDIA and Windows hardware; a real device loss; a real Llama 4, Mistral
Large 3, NVFP4 or gemma-4-26B-A4B run (none is cached: each feature is tested on
a synthetic fixture against a transcription of the reference); Metal, which
compiles and runs only in CI.
