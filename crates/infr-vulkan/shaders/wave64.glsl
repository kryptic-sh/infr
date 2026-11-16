// Virtual wave32 on wave64-only devices: the host sets W64 and every subgroup op below acts on 32-lane halves.
#ifndef WAVE64_GLSL
#define WAVE64_GLSL
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require

layout(constant_id = 1000) const bool W64 = false;

uint vw_lane() { return W64 ? (gl_SubgroupInvocationID & 31u) : gl_SubgroupInvocationID; }
uint vw_sgid() { return W64 ? (gl_SubgroupID * 2u + (gl_SubgroupInvocationID >> 5)) : gl_SubgroupID; }
uint vw_sgsize() { return W64 ? 32u : gl_SubgroupSize; }
bool vw_elect() {
    return W64 ? (subgroupClusteredMin(gl_SubgroupInvocationID, 32u) == gl_SubgroupInvocationID) : subgroupElect();
}
// Each half scans only its own lanes: the other half's are zeroed rather than subtracted back out
// afterwards, which would cancel catastrophically when the lower half's total dwarfs the upper's.
float vw_inclusive_add(float x) {
    if (!W64) {
        return subgroupInclusiveAdd(x);
    }
    bool hi = gl_SubgroupInvocationID >= 32u;
    float lo_scan = subgroupInclusiveAdd(hi ? 0.0 : x);
    float hi_scan = subgroupInclusiveAdd(hi ? x : 0.0);
    return hi ? hi_scan : lo_scan;
}

#define gl_SubgroupInvocationID vw_lane()
#define gl_SubgroupID vw_sgid()
// A macro, not a function: gl_WorkGroupSize must be read after the shader's local_size layout.
#define gl_NumSubgroups \
    (W64 ? ((gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z + 31u) >> 5) : gl_NumSubgroups)
#define gl_SubgroupSize vw_sgsize()
#define subgroupElect() vw_elect()
#define subgroupAdd(x) (W64 ? subgroupClusteredAdd((x), 32u) : subgroupAdd(x))
#define subgroupMax(x) (W64 ? subgroupClusteredMax((x), 32u) : subgroupMax(x))
#define subgroupMin(x) (W64 ? subgroupClusteredMin((x), 32u) : subgroupMin(x))
#define subgroupInclusiveAdd(x) vw_inclusive_add(x)
#endif
