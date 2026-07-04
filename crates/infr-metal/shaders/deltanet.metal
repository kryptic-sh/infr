// ---- Gated DeltaNet (Qwen3-Next linear attention) + its depthwise conv, ON DEVICE. Both are
// sequential-over-rows recurrences; the parallelism is across CHANNELS (conv: each thread owns a
// channel and its state column, no cross-thread deps) and across (head, value-dim) (deltanet:
// one threadgroup per value head, each lane owns state COLUMN S[:, d] — the delta-rule update
// touches only that column, so rows need no state barrier, just the shared-q/k staging one).
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

struct DeltaNetParams { uint rows; uint nv; uint nk; uint kd; uint vd; float eps; };
kernel void deltanet_f32(device const float* q       [[buffer(0)]],
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
                         uint   tid   [[thread_position_in_threadgroup]],
                         uint   lane  [[thread_index_in_simdgroup]],
                         uint   sgid  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float qh[256];       // kd <= 256 (host-gated)
    threadgroup float kh[256];
    threadgroup float red[16];       // cross-simdgroup reductions + (beta, decay) broadcast
    uint h = tgpig;
    uint kh_idx = h % p.nk;
    uint nsg = p.vd / 32u;           // threads == vd (host-gated: vd % 32 == 0, vd <= 1024)
    device float* S = state + (ulong)h * p.kd * p.vd;
    uint d = tid;

    for (uint t = 0; t < p.rows; t++) {
        // stage this row's q/k head and L2-normalize (q also x 1/sqrt(kd))
        ulong qb = (ulong)t * p.nk * p.kd + (ulong)kh_idx * p.kd;
        float sq = 0.0f, sk = 0.0f;
        for (uint i = tid; i < p.kd; i += p.vd) {
            float qv = q[qb + i];
            float kv = k[qb + i];
            qh[i] = qv;
            kh[i] = kv;
            sq += qv * qv;
            sk += kv * kv;
        }
        sq = simd_sum(sq);
        sk = simd_sum(sk);
        if (lane == 0u) {
            red[sgid] = sq;
            red[8u + sgid] = sk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tq = 0.0f, tk = 0.0f;
            for (uint i = 0; i < nsg; i++) {
                tq += red[i];
                tk += red[8u + i];
            }
            red[0] = sqrt(tq + p.eps);
            red[1] = sqrt(tk + p.eps);
            // beta = sigmoid(b); decay = exp(a_coef * softplus(a + dt_bias))
            float bv = b[t * p.nv + h];
            red[2] = 1.0f / (1.0f + exp(-bv));
            float z = a[t * p.nv + h] + dt_bias[h];
            float sp = max(z, 0.0f) + log(1.0f + exp(-fabs(z)));
            red[3] = exp(a_coef[h] * sp);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qn = red[0], kn = red[1], beta = red[2], decay = red[3];
        float qscale = rsqrt((float)p.kd);
        for (uint i = tid; i < p.kd; i += p.vd) {
            qh[i] = qh[i] / qn * qscale;
            kh[i] = kh[i] / kn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // lane d owns state column S[:, d]: kv_d over the decayed state, then the delta-rule
        // update fused with the output accumulation (one read + one write of the column)
        float kvd = 0.0f;
        for (uint kk = 0; kk < p.kd; kk++) kvd += kh[kk] * S[(ulong)kk * p.vd + d];
        kvd *= decay;
        float delta = (v[(ulong)t * p.nv * p.vd + (ulong)h * p.vd + d] - kvd) * beta;
        float od = 0.0f;
        for (uint kk = 0; kk < p.kd; kk++) {
            float sv = S[(ulong)kk * p.vd + d] * decay + kh[kk] * delta;
            S[(ulong)kk * p.vd + d] = sv;
            od += qh[kk] * sv;
        }
        dst[(ulong)t * p.nv * p.vd + (ulong)h * p.vd + d] = od;
        threadgroup_barrier(mem_flags::mem_threadgroup); // qh/kh restaged next row
    }
}
