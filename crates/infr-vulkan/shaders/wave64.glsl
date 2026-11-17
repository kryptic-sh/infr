// Virtual wave32 on wave64-only devices. The host pins every pipeline to subgroup 64 and sets W64;
// the macros below then give every kernel written for 32-lane subgroups the subgroup it expects:
//
// - A workgroup of at most 32 invocations is ONE 64-wide subgroup with only its own invocations
//   active, wherever the driver placed them in the wave. Its virtual subgroup is the whole
//   workgroup: id 0, lane = gl_LocalInvocationIndex, and the plain subgroup operations, which
//   span exactly the active lanes.
// - Any larger workgroup is a multiple of 64 wide in x (`ops::tests` checks every module), so the
//   host asks for full subgroups and each one holds 64 invocations. Its virtual subgroups are the
//   two 32-lane halves of each, and the operations are clustered at 32.
//
// Every subgroup operation a kernel may use under this header has a remap below; the reductions
// all keep their plain spelling as the wave32 twin, so `ops::tests::subgroup_kernels_are_wave64_safe`
// can pair them. To add one: define `subgroupFoo(x)` the way `subgroupAdd` is, and teach that test
// its opcode.
#ifndef WAVE64_GLSL
#define WAVE64_GLSL
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require

layout(constant_id = 1000) const bool W64 = false;

// The real builtins, read before the macros below rename them.
uint vw_hw_lane() { return gl_SubgroupInvocationID; }
uint vw_hw_sgid() { return gl_SubgroupID; }
bool vw_half_elect() {
    return subgroupClusteredMin(gl_SubgroupInvocationID, 32u) == gl_SubgroupInvocationID;
}
// Each half scans only its own lanes: the other half's are zeroed rather than subtracted back out
// afterwards, which would cancel catastrophically when the lower half's total dwarfs the upper's.
float vw_half_inclusive_add(float x) {
    bool hi = gl_SubgroupInvocationID >= 32u;
    float lo_scan = subgroupInclusiveAdd(hi ? 0.0 : x);
    float hi_scan = subgroupInclusiveAdd(hi ? x : 0.0);
    return hi ? hi_scan : lo_scan;
}

// Macros, not functions: gl_WorkGroupSize can only be read after the shader's local_size layout.
#define VW_WG_INVOCATIONS (gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z)
#define VW_WHOLE (W64 && VW_WG_INVOCATIONS <= 32u)
#define VW_HALVES (W64 && VW_WG_INVOCATIONS > 32u)

#define gl_SubgroupInvocationID \
    (VW_HALVES ? (vw_hw_lane() & 31u) : (VW_WHOLE ? gl_LocalInvocationIndex : vw_hw_lane()))
#define gl_SubgroupID \
    (VW_HALVES ? (vw_hw_sgid() * 2u + (vw_hw_lane() >> 5)) : (VW_WHOLE ? 0u : vw_hw_sgid()))
#define gl_NumSubgroups \
    (VW_HALVES ? (VW_WG_INVOCATIONS >> 5) : (VW_WHOLE ? 1u : gl_NumSubgroups))
#define gl_SubgroupSize (W64 ? 32u : gl_SubgroupSize)
#define subgroupElect() (VW_HALVES ? vw_half_elect() : subgroupElect())
#define subgroupInclusiveAdd(x) (VW_HALVES ? vw_half_inclusive_add(x) : subgroupInclusiveAdd(x))
#define subgroupAdd(x) (VW_HALVES ? subgroupClusteredAdd((x), 32u) : subgroupAdd(x))
#define subgroupMul(x) (VW_HALVES ? subgroupClusteredMul((x), 32u) : subgroupMul(x))
#define subgroupMin(x) (VW_HALVES ? subgroupClusteredMin((x), 32u) : subgroupMin(x))
#define subgroupMax(x) (VW_HALVES ? subgroupClusteredMax((x), 32u) : subgroupMax(x))
#define subgroupAnd(x) (VW_HALVES ? subgroupClusteredAnd((x), 32u) : subgroupAnd(x))
#define subgroupOr(x) (VW_HALVES ? subgroupClusteredOr((x), 32u) : subgroupOr(x))
#define subgroupXor(x) (VW_HALVES ? subgroupClusteredXor((x), 32u) : subgroupXor(x))
#endif
