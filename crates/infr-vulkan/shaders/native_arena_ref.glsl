// Buffer-device-address (BDA) access to the paged-MoE expert arena. Included ONLY by the `-DPAGED`
// builds of the expert kernels (native_gemv_id(_multi).comp and every native_gemm_mmq_*.comp).
//
// The arena is a device-local buffer created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS, addressed
// by its 64-bit VkDeviceAddress rather than a bound SSBO. A single SSBO binding caps at
// maxStorageBufferRange (~4 GiB on RADV), which limited a per-role expert pool to ~4 GiB; a raw
// pointer removes that cap, so a pool holds as many experts as VRAM allows (higher hit rate, fewer
// PCIe page-ins). The host passes the arena base address split into two u32 push-constant fields
// (`arena_lo`/`arena_hi` — a uvec2 avoids the 8-byte push-constant alignment a uint64_t member
// would force) plus the per-slot byte stride; the LUT carries each resident expert's SLOT INDEX.
//
// `nw_ptr` is set ONCE in main() to this slot's base byte address:
//     nw_ptr = arena_base(lo, hi) + uint64_t(lut_slot) * uint64_t(slot_bytes)
// The multiply is 64-bit, so no arena size overflows it (the u32 element-space multiply this
// replaces was the original coherent-but-wrong bug at slot ≥ ~102 on Scout). `arena_word(wi)` then
// reads word `wi` of the slot — the drop-in replacement for the old `nw[nw_base + wi]` SSBO read.
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ArenaWords { uint v[]; };

uint64_t nw_ptr = 0ul; // this expert's arena base byte address (set once in main from the LUT slot)

uint64_t arena_base(uint lo, uint hi) { return (uint64_t(hi) << 32) | uint64_t(lo); }
// Word read, shaped for the GLOBAL_LOAD saddr form: the byte offset is computed fully in 32-bit
// and added to the (wave-uniform) base pointer BEFORE the cast, and the deref index is the
// constant 0 — NIR then sees `iadd(uniform64, u2u64(divergent32))`, which ACO selects as a
// scalar-base global_load (64-bit base in SGPRs, one 32-bit VGPR offset). The former
// `ArenaWords(nw_ptr).v[wi]` promoted the index to 64-bit inside the deref, defeating saddr:
// every load materialized its own 64-bit VGPR address via a v_add_co/v_add_co_ci pair (measured
// ~1.5 carry-adds per load on the Q6K streamed warp GEMM — the 2.2x flag-on regression vs the
// descriptor-bound twin, which gets base+offset addressing for free from the SGPR descriptor).
// `nw_ptr` is wave-uniform by construction (push constants, or a LUT slot picked by workgroup id)
// but it lives in a mutable global, which NIR's divergence analysis treats as divergent — that
// alone blocks saddr selection. subgroupBroadcastFirst is the standard uniformity hint; ACO CSEs
// the readfirstlane across all inlined calls.
uint arena_word(uint wi) {
    return ArenaWords(nw_ptr + uint64_t(wi << 2u)).v[0];
}
