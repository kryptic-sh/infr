// ---- RoPE (Op::Rope = the no-qk-norm llama-family rotation): INTERLEAVED pairs (2p, 2p+1) —
// llama.cpp's ROPE_TYPE_NORM, matching infr-cpu and the Vulkan `rope` kernel. (QkNormRope below
// is the NEOX split-half used by qwen/gemma; the styles are NOT interchangeable.) Rotates the
// first rope_dim of each head; dims beyond pass through. One thread per (row, head).
struct RopeParams { uint rows; uint n_head; uint head_dim; uint rope_dim; float theta; uint has_ff; };
// One thread per (row, head, rotation pair) — the previous one-thread-per-head form rotated
// rope_dim/2 pairs SERIALLY (trig included) and copied the whole head, which left a decode row
// on a single simdgroup (the counter profiler measured it at 19% of TinyLlama decode). Threads
// past the pairs copy the pass-through dims. Per-element float expressions are unchanged.
kernel void rope_f32(device const float* x   [[buffer(0)]],
                     device const float* pos [[buffer(1)]],
                     device const float* ff  [[buffer(2)]],
                     device float*       dst [[buffer(3)]],
                     constant RopeParams& p  [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
    uint hf = p.rope_dim / 2;
    uint per = hf + (p.head_dim - p.rope_dim);
    uint rh = gid / per;
    if (rh >= p.rows * p.n_head) return;
    uint k = gid % per;
    uint r = rh / p.n_head;
    uint base = rh * p.head_dim;
    if (k >= hf) {
        uint i = p.rope_dim + (k - hf);
        dst[base + i] = x[base + i];
        return;
    }
    uint i0 = 2 * k, i1 = 2 * k + 1;
    float p0 = pos[r];
    float ang = p0 * pow(p.theta, -2.0f * (float)k / (float)p.rope_dim);
    if (p.has_ff != 0) ang /= ff[k];
    float c = cos(ang), s = sin(ang);
    float a = x[base + i0], b = x[base + i1];
    dst[base + i0] = a * c - b * s;
    dst[base + i1] = a * s + b * c;
}

// ---- Fused per-head RMSNorm + RoPE (QkNormRope): rmsnorm (× weight) then rotate the normed head.
// One SIMD group per (row, head): `simd_sum` for the norm, then lanes split the rotation pairs. Each
// lane forms its normed values straight from `x` (× s × w), so no cross-lane read of `dst` — no
// barrier needed. Pass-through dims [rope_dim, head_dim) are written normed in the tail loop.
struct QkRopeParams { uint rows; uint n_head; uint head_dim; uint rope_dim; float theta; float eps; uint has_ff; };
kernel void qknormrope_f32(device const float* x   [[buffer(0)]],
                           device const float* w   [[buffer(1)]],
                           device const int*   pos [[buffer(2)]],
                           device const float* ff  [[buffer(3)]],
                           device float*       dst [[buffer(4)]],
                           constant QkRopeParams& p [[buffer(5)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint grp = gid / 32u;
    if (grp >= p.rows * p.n_head) return;
    uint r = grp / p.n_head;
    uint base = grp * p.head_dim;
    float ss = 0.0f;
    for (uint i = lane; i < p.head_dim; i += 32u) { float v = x[base + i]; ss += v * v; }
    ss = simd_sum(ss) / (float)p.head_dim;
    float s = 1.0f / sqrt(ss + p.eps);
    uint hf = p.rope_dim / 2;
    float p0 = (float)pos[r];  // bound i32 read directly; exact widening
    for (uint pp = lane; pp < hf; pp += 32u) {
        uint i0 = pp, i1 = pp + hf;
        float ang = p0 * pow(p.theta, -2.0f * (float)pp / (float)p.rope_dim);
        if (p.has_ff != 0) ang /= ff[pp];
        float c = cos(ang), sn = sin(ang);
        float a = x[base + i0] * s * w[i0];
        float b = x[base + i1] * s * w[i1];
        dst[base + i0] = a * c - b * sn;
        dst[base + i1] = a * sn + b * c;
    }
    for (uint i = p.rope_dim + lane; i < p.head_dim; i += 32u) dst[base + i] = x[base + i] * s * w[i];
}

// Wide fused QkNorm+RoPE for DECODE (rows == 1): 8 simdgroups (256 threads) per (row, head) —
// same latency story as `rmsnorm_wide_f32` (the 32-lane form serializes head_dim/32 loads and a
// decode row only launches n_head simdgroups; gemma has FOUR heads).
kernel void qknormrope_wide_f32(device const float* x   [[buffer(0)]],
                                device const float* w   [[buffer(1)]],
                                device const int*   pos [[buffer(2)]],
                                device const float* ff  [[buffer(3)]],
                                device float*       dst [[buffer(4)]],
                                constant QkRopeParams& p [[buffer(5)]],
                                uint tid  [[thread_position_in_threadgroup]],
                                uint grp  [[threadgroup_position_in_grid]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    if (grp >= p.rows * p.n_head) return;
    uint r = grp / p.n_head;
    uint base = grp * p.head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < p.head_dim; i += 256u) {
        float v = x[base + i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0u) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
    float s = 1.0f / sqrt(tot / (float)p.head_dim + p.eps);
    uint hf = p.rope_dim / 2;
    float p0 = (float)pos[r];
    for (uint pp = tid; pp < hf; pp += 256u) {
        uint i0 = pp, i1 = pp + hf;
        float ang = p0 * pow(p.theta, -2.0f * (float)pp / (float)p.rope_dim);
        if (p.has_ff != 0) ang /= ff[pp];
        float c = cos(ang), sn = sin(ang);
        float a = x[base + i0] * s * w[i0];
        float b = x[base + i1] * s * w[i1];
        dst[base + i0] = a * c - b * sn;
        dst[base + i1] = a * sn + b * c;
    }
    for (uint i = p.rope_dim + tid; i < p.head_dim; i += 256u) {
        dst[base + i] = x[base + i] * s * w[i];
    }
}

// ---- Gated FFN activation: dst[r,i] = act(gate[r,i]) * up[r, i + up_off]. act: 0=SiLU,1=GELU,2=Sigmoid
inline float gated_act(uint act, float g) {
    if (act == 0u) return g / (1.0f + exp(-g));                       // SiLU
    if (act == 2u) return 1.0f / (1.0f + exp(-g));                    // Sigmoid
    // GELU (gelu_pytorch_tanh)
    return 0.5f * g * (1.0f + tanh(0.7978845608f * (g + 0.044715f * g * g * g)));
}
struct GatedParams { uint rows; uint nff; uint act; uint up_off; };
// Fused-projection form (`combined_gu`): gate|up live in ONE [rows, 2*nff] buffer (gate half
// first), produced by a single Linear over the concatenated weights.
kernel void gatedactfused_f32(device const float* gu  [[buffer(0)]],
                              device float*       dst [[buffer(1)]],
                              constant GatedParams& p [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.nff) return;
    uint r = gid / p.nff;
    uint i = gid % p.nff;
    ulong gb = (ulong)r * 2u * p.nff;
    dst[gid] = gated_act(p.act, gu[gb + i]) * gu[gb + p.nff + i];
}
kernel void gatedact_f32(device const float* gate [[buffer(0)]],
                         device const float* up   [[buffer(1)]],
                         device float*       dst  [[buffer(2)]],
                         constant GatedParams& p  [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.nff) return;
    uint r = gid / p.nff;
    uint i = gid % p.nff;
    uint gb = r * p.nff + i;
    uint ub = r * p.nff + p.up_off + i;
    dst[gb] = gated_act(p.act, gate[gb]) * up[ub];
}
