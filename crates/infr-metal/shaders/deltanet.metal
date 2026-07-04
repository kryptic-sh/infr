// ---- Gated DeltaNet (qwen35/Qwen3.5 linear attention) + its depthwise conv, ON DEVICE. Both are
// sequential-over-rows recurrences; the parallelism is across CHANNELS (conv: each thread owns a
// channel and its state column, no cross-thread deps) and across (value-dim, head) (deltanet:
// one SIMDGROUP per state column S[:, d], the column register-resident across the chunk — see
// the kernel comment).
// State lives in the BOUND buffer and is updated in place (no host round-trip per layer).
struct ConvSiluParams { uint rows; uint channels; uint kwidth; };
kernel void conv1d_silu_f32(device const float* x     [[buffer(0)]],
                            device const float* w     [[buffer(1)]],
                            device float*       state [[buffer(2)]],
                            device float*       dst   [[buffer(3)]],
                            constant ConvSiluParams& p [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    uint ch = gid;
    if (ch >= p.channels) return;
    uint kk = p.kwidth;             // host gates kwidth <= 8
    float st[7];
    float wv[8];
    for (uint j = 0; j + 1 < kk; j++) st[j] = state[j * p.channels + ch];
    for (uint j = 0; j < kk; j++) wv[j] = w[ch * kk + j];
    for (uint t = 0; t < p.rows; t++) {
        float xt = x[t * p.channels + ch];
        float acc = xt * wv[kk - 1u];
        for (uint j = 0; j + 1 < kk; j++) acc += st[j] * wv[j];
        dst[t * p.channels + ch] = acc / (1.0f + exp(-acc));
        for (uint j = 0; j + 2 < kk; j++) st[j] = st[j + 1];
        if (kk >= 2u) st[kk - 2u] = xt;
    }
    for (uint j = 0; j + 1 < kk; j++) state[j * p.channels + ch] = st[j];
}

// One SIMDGROUP per (value-dim d, value-head h) — the llama.cpp kernel_gated_delta_net
// parallelization. The simdgroup's 32 lanes split the k-dim (KPL = kd/32 state entries per
// lane, register-resident ls[] for the WHOLE chunk — the old shape re-read the state column
// from device twice per token), so the token recurrence needs only simd_sums: no threadgroup
// memory, no barriers. DN_VPT simdgroups share a threadgroup purely for occupancy (grid =
// nv * vd/DN_VPT threadgroups of DN_VPT*32 threads — vd/DN_VPT times the old
// one-threadgroup-per-head grid, which left the GPU idle at qwen35's 32 heads). State
// reads/writes touch device memory once per chunk instead of 2*rows times.
struct DeltaNetParams { uint rows; uint nv; uint nk; uint kd; uint vd; float eps; };
#define DN_VPT 4u
// KPL (k entries per lane, = kd/32) is a COMPILE-TIME template parameter: with a runtime
// bound the ls[]/qv[]/kv[] arrays are runtime-indexed, the compiler cannot promote them to
// registers, and the "register-resident" state silently spills to thread-private memory —
// measured 1.7x SLOWER than the old threadgroup-staged shape (507 ms vs 302 per qwen35
// prefill). Unrolled at fixed KPL the arrays genuinely live in registers: 185 ms, 1.6x
// faster than the old shape.
template<uint KPL>
kernel void deltanet_f32_t(device const float* q       [[buffer(0)]],
                         device const float* k       [[buffer(1)]],
                         device const float* v       [[buffer(2)]],
                         device const float* b       [[buffer(3)]],
                         device const float* a       [[buffer(4)]],
                         device const float* a_coef  [[buffer(5)]],
                         device const float* dt_bias [[buffer(6)]],
                         device float*       state   [[buffer(7)]],
                         device float*       dst     [[buffer(8)]],
                         constant DeltaNetParams& p  [[buffer(9)]],
                         uint   tgpig [[threadgroup_position_in_grid]],
                         uint   lane  [[thread_index_in_simdgroup]],
                         uint   sgid  [[simdgroup_index_in_threadgroup]]) {
    uint dpg = p.vd / DN_VPT;                 // d-groups per head
    uint h = tgpig / dpg;
    uint d = (tgpig % dpg) * DN_VPT + sgid;   // this simdgroup's value dim
    uint kh_idx = h % p.nk;
    float qscale = rsqrt((float)p.kd);

    // Lane `lane` owns k indices lane*kpl .. lane*kpl+kpl-1 of state column S[:, d],
    // register-resident across the whole chunk.
    device float* S = state + (ulong)h * p.kd * p.vd;
    float ls[KPL];
#pragma unroll
    for (uint j = 0; j < KPL; j++) ls[j] = S[(ulong)(lane * KPL + j) * p.vd + d];

    for (uint t = 0; t < p.rows; t++) {
        // This row's q/k head slice: each lane loads its kpl entries; L2 norms via simd_sum
        // (q also x 1/sqrt(kd)) — same formulas as the CPU reference.
        ulong qb = (ulong)t * p.nk * p.kd + (ulong)kh_idx * p.kd + lane * KPL;
        float qv[KPL], kv[KPL];
        float sq = 0.0f, sk = 0.0f;
#pragma unroll
        for (uint j = 0; j < KPL; j++) {
            qv[j] = q[qb + j];
            kv[j] = k[qb + j];
            sq += qv[j] * qv[j];
            sk += kv[j] * kv[j];
        }
        float qn = sqrt(simd_sum(sq) + p.eps);
        float kn = sqrt(simd_sum(sk) + p.eps);
#pragma unroll
        for (uint j = 0; j < KPL; j++) {
            qv[j] = qv[j] / qn * qscale;
            kv[j] = kv[j] / kn;
        }
        // beta = sigmoid(b); decay = exp(a_coef * softplus(a + dt_bias)) — computed on every
        // lane (identical inputs, cheaper than a broadcast).
        float bv = b[t * p.nv + h];
        float beta = 1.0f / (1.0f + exp(-bv));
        float z = a[t * p.nv + h] + dt_bias[h];
        float sp = max(z, 0.0f) + log(1.0f + exp(-fabs(z)));
        float decay = exp(a_coef[h] * sp);

        // Delta rule on the decayed state, fused with the output accumulation.
        float s_k = 0.0f;
#pragma unroll
        for (uint j = 0; j < KPL; j++) {
            ls[j] *= decay;
            s_k += ls[j] * kv[j];
        }
        s_k = simd_sum(s_k);
        float delta = (v[(ulong)t * p.nv * p.vd + (ulong)h * p.vd + d] - s_k) * beta;
        float y = 0.0f;
#pragma unroll
        for (uint j = 0; j < KPL; j++) {
            ls[j] += kv[j] * delta;
            y += ls[j] * qv[j];
        }
        y = simd_sum(y);
        if (lane == 0u) {
            dst[(ulong)t * p.nv * p.vd + (ulong)h * p.vd + d] = y;
        }
    }

#pragma unroll
    for (uint j = 0; j < KPL; j++) S[(ulong)(lane * KPL + j) * p.vd + d] = ls[j];
}

typedef decltype(deltanet_f32_t<4>) deltanet_f32_k;
template [[host_name("deltanet_f32_k1")]] kernel deltanet_f32_k deltanet_f32_t<1>;
template [[host_name("deltanet_f32_k2")]] kernel deltanet_f32_k deltanet_f32_t<2>;
template [[host_name("deltanet_f32_k4")]] kernel deltanet_f32_k deltanet_f32_t<4>;
template [[host_name("deltanet_f32_k8")]] kernel deltanet_f32_k deltanet_f32_t<8>;
