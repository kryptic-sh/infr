// ---- elementwise ----
kernel void advance_position_i32(device int* position [[buffer(0)]]) {
    position[0] += 1;
}

kernel void add_f32(device const float* a   [[buffer(0)]],
                    device const float* b   [[buffer(1)]],
                    device float*       dst [[buffer(2)]],
                    constant uint&      n   [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) dst[gid] = a[gid] + b[gid];
}

// Broadcast bias add (Qwen2/2.5 q/k/v `Wx + b`): dst[i] = x[i] + bias[i % n] over `total = rows*n`
// elements. Params: n = bias length / row width, total = rows*n.
struct AddBiasParams { uint n; uint total; };
kernel void add_bias_f32(device const float* x    [[buffer(0)]],
                         device const float* bias [[buffer(1)]],
                         device float*       dst  [[buffer(2)]],
                         constant AddBiasParams& p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < p.total) dst[gid] = x[gid] + bias[gid % p.n];
}

// Broadcast multiply (diffusion-gemma router input scale): dst[i] = x[i] * vec[i % n] over
// `total = rows*n` elements. The multiplicative twin of `add_bias_f32`.
struct MulVecParams { uint n; uint total; };
kernel void mul_vec_f32(device const float* x    [[buffer(0)]],
                        device const float* vec_ [[buffer(1)]],
                        device float*       dst  [[buffer(2)]],
                        constant MulVecParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < p.total) dst[gid] = x[gid] * vec_[gid % p.n];
}

struct ScaleParams { float s; uint n; };
kernel void scale_f32(device const float* x   [[buffer(0)]],
                      device float*       dst [[buffer(1)]],
                      constant ScaleParams& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < p.n) dst[gid] = x[gid] * p.s;
}

struct SoftcapParams { float cap; uint n; };
kernel void softcap_f32(device const float* x   [[buffer(0)]],
                        device float*       dst [[buffer(1)]],
                        constant SoftcapParams& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < p.n) dst[gid] = p.cap * tanh(x[gid] / p.cap);
}

// ---- norms: one SIMD group (32 lanes) per normalized group. Lanes stride the group, `simd_sum`
// reduces the sum-of-squares, then all 32 write the scaled output in parallel. (Decode has rows=1,
// so the old one-thread-per-row kernel ran the whole reduction on a single thread — pathological.)
struct RmsParams { uint rows; uint dim; float eps; };
kernel void rmsnorm_f32(device const float* x   [[buffer(0)]],
                        device const float* w   [[buffer(1)]],
                        device float*       dst [[buffer(2)]],
                        constant RmsParams& p   [[buffer(3)]],
                        uint gid  [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32u;
    if (row >= p.rows) return;
    uint base = row * p.dim;
    float ss = 0.0f;
    for (uint i = lane; i < p.dim; i += 32u) { float v = x[base + i]; ss += v * v; }
    ss = simd_sum(ss) / (float)p.dim;
    float s = 1.0f / sqrt(ss + p.eps);
    for (uint i = lane; i < p.dim; i += 32u) dst[base + i] = x[base + i] * s * w[i];
}

// Wide RMSNorm for DECODE (rows == 1): 8 simdgroups (256 threads) cooperate on the one row.
// The 32-lane kernel is latency-bound on its dim/32 serial loads — ~20 us per launch at
// dim=1152 (the counter profiler's first catch: gemma decode fires 105 of these per token,
// 20% of its GPU time). Partials fold through threadgroup memory; every thread re-sums the 8
// partials to skip a second barrier.
kernel void rmsnorm_wide_f32(device const float* x   [[buffer(0)]],
                             device const float* w   [[buffer(1)]],
                             device float*       dst [[buffer(2)]],
                             constant RmsParams& p   [[buffer(3)]],
                             uint tid  [[thread_position_in_threadgroup]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    if (row >= p.rows) return;
    uint base = row * p.dim;
    float ss = 0.0f;
    for (uint i = tid; i < p.dim; i += 256u) {
        float v = x[base + i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0u) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
    float s = 1.0f / sqrt(tot / (float)p.dim + p.eps);
    for (uint i = tid; i < p.dim; i += 256u) dst[base + i] = x[base + i] * s * w[i];
}

// Row-wise softmax: dst[r,:] = softmax(x[r,:] * scale), one threadgroup (8 simdgroups) per row —
// diffusion-gemma's in-graph self-conditioning (see docs/DIFFUSIONGEMMA.md's Phase-B and the
// reference's `dg_canvas_embed`). Same wide-launch shape as `rmsnorm_wide_f32` since the row width
// (vocab) is large. NOTE: unlike the rest of this backend, this kernel is UNVERIFIED on real
// Metal hardware (added blind, following the CPU/Vulkan implementations — see infr-vulkan's
// `softmax.comp` for the sibling shader this mirrors).
struct SoftmaxParams { uint rows; uint dim; float scale; };
kernel void softmax_wide_f32(device const float* x   [[buffer(0)]],
                             device float*       dst [[buffer(1)]],
                             constant SoftmaxParams& p [[buffer(2)]],
                             uint tid  [[thread_position_in_threadgroup]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    if (row >= p.rows) return;
    uint base = row * p.dim;

    // row max (numerically stable exp)
    float m = -INFINITY;
    for (uint i = tid; i < p.dim; i += 256u) {
        m = max(m, x[base + i] * p.scale);
    }
    m = simd_max(m);
    if (lane == 0u) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m0 = max(max(max(red[0], red[1]), max(red[2], red[3])),
                   max(max(red[4], red[5]), max(red[6], red[7])));
    threadgroup_barrier(mem_flags::mem_threadgroup); // every thread read `red[]` before it's reused

    // row sum of exp(x*scale - m0)
    float s = 0.0f;
    for (uint i = tid; i < p.dim; i += 256u) {
        s += exp(x[base + i] * p.scale - m0);
    }
    s = simd_sum(s);
    if (lane == 0u) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
    float inv = 1.0f / tot;

    for (uint i = tid; i < p.dim; i += 256u) {
        dst[base + i] = exp(x[base + i] * p.scale - m0) * inv;
    }
}

// per-head RMSNorm: one SIMD group (32 lanes) per (row, head), weight indexed within head_dim.
struct QkNormParams { uint rows; uint n_head; uint head_dim; float eps; };
kernel void qknorm_f32(device const float* x   [[buffer(0)]],
                       device const float* w   [[buffer(1)]],
                       device float*       dst [[buffer(2)]],
                       constant QkNormParams& p [[buffer(3)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint grp = gid / 32u;
    if (grp >= p.rows * p.n_head) return;
    uint base = grp * p.head_dim;
    float ss = 0.0f;
    for (uint i = lane; i < p.head_dim; i += 32u) { float v = x[base + i]; ss += v * v; }
    ss = simd_sum(ss) / (float)p.head_dim;
    float s = 1.0f / sqrt(ss + p.eps);
    for (uint i = lane; i < p.head_dim; i += 32u) dst[base + i] = x[base + i] * s * w[i];
}

// Greedy argmax over `n` logits → token id (one 256-thread threadgroup, strided scan +
// threadgroup tree-reduce). Strict > keeps the lowest index on ties, matching the host argmax
// (same contract as the Vulkan argmax.comp). The id is written as a u32 bit-pattern into the
// f32 output slot — greedy decode reads back 4 bytes instead of the [vocab] logits.
struct ArgmaxParams { uint n; };
kernel void argmax_f32(device const float* logits [[buffer(0)]],
                       device uint*        out_id [[buffer(1)]],
                       constant ArgmaxParams& p   [[buffer(2)]],
                       uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sval[256];
    threadgroup uint  sidx[256];
    float best = -1e30f;
    uint bi = 0u;
    for (uint i = t; i < p.n; i += 256u) {
        if (logits[i] > best) { best = logits[i]; bi = i; }
    }
    sval[t] = best;
    sidx[t] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s /= 2u) {
        if (t < s && sval[t + s] > sval[t]) {
            sval[t] = sval[t + s];
            sidx[t] = sidx[t + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) { out_id[0] = sidx[0]; }
}

// GPU stochastic sampling over VOCAB-scale logits (Op::Sample): temperature + top-k + top-p,
// IDENTICAL order of operations to the host `sample_logits` (infr-cpu's Op::Sample arm) given the
// same uniform draw `u`, so the same `u` picks the same token (modulo exact-tie order — ties are
// legitimately unspecified, same caveat as the Vulkan sample_topk.comp reference).
//
// Correctness-first single-threadgroup version (this is the reference backend): `top_k` (bounded
// 2..=64 by the caller) selection is done via `top_k` sequential parallel-max reductions (each an
// argmax_f32-shaped strided-scan + threadgroup tree-reduce), skipping indices already selected in
// earlier rounds — descending order falls out for free since round `j` finds the (j+1)-th largest.
// This re-scans the `n` logits `top_k` times instead of Vulkan's one-pass radix select; fine for a
// per-token decode op where correctness, not throughput, is the bar. Phase 2 (single lane) mirrors
// the host: softmax(temp) over the selected set, nucleus (top-p) cutoff, inverse-CDF walk with `u`.
#define SAMPLE_KMAX 64u
struct SampleParams { uint n; uint top_k; float temp; float top_p; };
inline void sample_f32_impl(device const float* logits,
                            float uniform,
                            device uint* out_id,
                            constant SampleParams& p,
                            uint t,
                            threadgroup float* sval,
                            threadgroup uint* sidx,
                            threadgroup float* gval,
                            threadgroup uint* gidx) {
    // Clamp like the host (`k = top_k.min(logits.len())`) — defensive against a vocab smaller
    // than top_k; never triggers in practice (vocab >> 64).
    uint k = min(p.top_k, p.n);
    k = min(k, SAMPLE_KMAX);
    for (uint iter = 0u; iter < k; iter++) {
        float best = -1e30f;
        uint bi = 0u;
        for (uint i = t; i < p.n; i += 256u) {
            bool used = false;
            for (uint j = 0u; j < iter; j++) {
                if (gidx[j] == i) { used = true; break; }
            }
            if (!used && logits[i] > best) { best = logits[i]; bi = i; }
        }
        sval[t] = best;
        sidx[t] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s /= 2u) {
            if (t < s && sval[t + s] > sval[t]) {
                sval[t] = sval[t + s];
                sidx[t] = sidx[t + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t == 0u) {
            gval[iter] = sval[0];
            gidx[iter] = sidx[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Phase 2 (single lane): softmax(temp), nucleus cutoff, inverse-CDF sample — same two-pass
    // structure (cutoff scan, then an independent re-summed `total`) as the host reference.
    if (t == 0u) {
        float maxl = gval[0];
        float sum = 0.0f;
        for (uint j = 0u; j < k; j++) {
            float pr = exp((gval[j] - maxl) / p.temp);
            gval[j] = pr; // reuse gval to hold probabilities
            sum += pr;
        }
        for (uint j = 0u; j < k; j++) { gval[j] /= sum; }
        float cum = 0.0f;
        uint cutoff = k;
        for (uint j = 0u; j < k; j++) {
            cum += gval[j];
            if (cum >= p.top_p) { cutoff = j + 1u; break; }
        }
        float total = 0.0f;
        for (uint j = 0u; j < cutoff; j++) { total += gval[j]; }
        float r = uniform * total;
        uint tok = gidx[cutoff - 1u];
        float acc = 0.0f;
        for (uint j = 0u; j < cutoff; j++) {
            acc += gval[j];
            if (r <= acc) { tok = gidx[j]; break; }
        }
        out_id[0] = tok;
    }
}

kernel void sample_f32(device const float* logits [[buffer(0)]],
                       device const float* u_buf  [[buffer(1)]],
                       device uint*        out_id [[buffer(2)]],
                       constant SampleParams& p    [[buffer(3)]],
                       uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sval[256];
    threadgroup uint  sidx[256];
    threadgroup float gval[SAMPLE_KMAX];
    threadgroup uint  gidx[SAMPLE_KMAX];
    sample_f32_impl(logits, u_buf[0], out_id, p, t, sval, sidx, gval, gidx);
}

// Record-once decode variant: params are fixed in the tape, while the bound position and the
// runner's 64-slot uniform ring change per token.
kernel void sample_f32_dyn(device const float* logits    [[buffer(0)]],
                           device const float* u_buf     [[buffer(1)]],
                           device const int*   positions [[buffer(2)]],
                           device uint*        out_id    [[buffer(3)]],
                           constant SampleParams& p       [[buffer(4)]],
                           uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sval[256];
    threadgroup uint  sidx[256];
    threadgroup float gval[SAMPLE_KMAX];
    threadgroup uint  gidx[SAMPLE_KMAX];
    sample_f32_impl(
        logits, u_buf[(uint)positions[0] & 63u], out_id, p, t, sval, sidx, gval, gidx
    );
}
