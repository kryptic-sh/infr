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
uint arena_word(uint wi) { return ArenaWords(nw_ptr).v[wi]; }
