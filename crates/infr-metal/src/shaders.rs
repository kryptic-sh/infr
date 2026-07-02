//! MSL compute kernels and lazy pipeline-state cache.
//!
//! All kernels operate on `float` (f32) buffers — quantized weights are dequantized to f32 on the
//! host before they reach a kernel, so the shaders stay format-agnostic and simple. The full MSL
//! source is compiled once at backend init; individual `MTLComputePipelineState`s are created on
//! first use and cached by function name.

use crate::be;
use infr_core::error::Result;
use metal::{ComputePipelineState, Device, Library};
use std::collections::HashMap;
use std::sync::Mutex;

pub struct Pipelines {
    device: Device,
    library: Library,
    cache: Mutex<HashMap<&'static str, ComputePipelineState>>,
}

unsafe impl Send for Pipelines {}
unsafe impl Sync for Pipelines {}

impl Pipelines {
    pub fn build(device: &Device) -> Result<Self> {
        let opts = metal::CompileOptions::new();
        // Reference backend: prefer accurate transcendentals (sin/cos/tanh) over fast intrinsics so
        // results stay in tight numeric parity with the CPU interpreter.
        opts.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(MSL_SRC, &opts)
            .map_err(|e| be(format!("compile MSL library: {e}")))?;
        Ok(Self {
            device: device.clone(),
            library,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Get (creating + caching on first use) the compute pipeline for an MSL kernel function.
    pub fn get(&self, name: &'static str) -> Result<ComputePipelineState> {
        if let Some(p) = self.cache.lock().unwrap().get(name) {
            return Ok(p.clone());
        }
        let func = self
            .library
            .get_function(name, None)
            .map_err(|e| be(format!("get MSL function {name}: {e}")))?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| be(format!("pipeline for {name}: {e}")))?;
        self.cache.lock().unwrap().insert(name, pso.clone());
        Ok(pso)
    }
}

/// Metal Shading Language source for every kernel. Kept in one string so it compiles as a single
/// library. Grows as ops are added.
const MSL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---- elementwise ----
kernel void add_f32(device const float* a   [[buffer(0)]],
                    device const float* b   [[buffer(1)]],
                    device float*       dst [[buffer(2)]],
                    constant uint&      n   [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) dst[gid] = a[gid] + b[gid];
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

// ---- Linear: dst[m, out_f] = x[m, in_f] · Wᵀ, W row-major [out_f, in_f], pre-dequantized to f32.
// One SIMD group (32 lanes) per output element: lanes stride the weight row contiguously — so
// consecutive lanes read consecutive weights (coalesced, full memory bandwidth, vs the strided
// one-thread-per-output naive matvec) — then `simd_sum` reduces. The lane-interleaved partial sums
// change the f32 summation order (still within parity tolerance; Linear was never bit-identical).
struct LinearParams { uint m; uint in_f; uint out_f; };
kernel void linear_f32(device const float* x   [[buffer(0)]],
                       device const float* w   [[buffer(1)]],
                       device float*       dst [[buffer(2)]],
                       constant LinearParams& p [[buffer(3)]],
                       uint gid [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;                       // one simdgroup per (row, output) pair
    if (sg >= p.m * p.out_f) return;
    uint r = sg / p.out_f;
    uint o = sg % p.out_f;
    device const float* xr = x + (ulong)r * p.in_f;
    device const float* wo = w + (ulong)o * p.in_f;
    float acc = 0.0f;
    for (uint i = lane; i < p.in_f; i += 32u) acc += xr[i] * wo[i];
    acc = simd_sum(acc);
    if (lane == 0u) dst[sg] = acc;
}

// ---- Linear over a NATIVE quantized weight in FACTORED form: `w = (d*sc)*code + (dmin*m)`, with
// bit-packed codes (4/6/8-bit, low bits first), one (sc, m) i16 pair per 16-element block, and one
// (d, dmin) f16 pair per 2^dshift elements. Same simdgroup GEMV as linear_f32, but each lane
// decodes one 16-element block per step: block codes + 4 bytes of (sc, m) + an amortized 4 bytes
// of (d, dmin) — ~6.1 bpw for Q4_K, ~8.1 for Q6_K, where the old u8-code + f32-scale form paid
// 8-12. Decode GEMV is bound on this stream, so bits are throughput. The two f32 multiplies
// reproduce the host dequant's scale/min bit-for-bit, keeping parity with the CPU exact.
struct QLinParams { uint m; uint in_f; uint out_f; uint dshift; };

// 4-bit codes: one uint2 = 8 bytes = one 16-element block, code k at bits 4k (LSB-first).
kernel void linear_quik4(device const float*  x     [[buffer(0)]],
                         device const uint2*  codes [[buffer(1)]],
                         device const short2* scm   [[buffer(2)]],
                         device const half2*  dd    [[buffer(3)]],
                         device float*        dst   [[buffer(4)]],
                         constant QLinParams& p     [[buffer(5)]],
                         uint gid  [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    if (sg >= p.m * p.out_f) return;
    uint r = sg / p.out_f;
    uint o = sg % p.out_f;
    uint nb = p.in_f >> 4;                        // 16-element blocks per weight row
    ulong row16 = (ulong)o * nb;
    ulong rowe = (ulong)o * p.in_f;
    device const float* xr = x + (ulong)r * p.in_f;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        uint2 cw = codes[row16 + b];
        short2 s = scm[row16 + b];
        half2 dv = dd[(rowe + ((ulong)b << 4)) >> p.dshift];
        float scale = (float)dv.x * (float)s.x;
        float mn = (float)dv.y * (float)s.y;
        device const float* xb = xr + (b << 4);
        for (uint k = 0; k < 8u; k++) {
            acc += xb[k]      * (scale * (float)((cw.x >> (4u * k)) & 15u) + mn);
            acc += xb[k + 8u] * (scale * (float)((cw.y >> (4u * k)) & 15u) + mn);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) dst[sg] = acc;
}

// 6-bit codes: three uints = 12 bytes = one 16-element block, code k at bits 6k (LSB-first).
kernel void linear_quik6(device const float*  x     [[buffer(0)]],
                         device const uint*   codes [[buffer(1)]],
                         device const short2* scm   [[buffer(2)]],
                         device const half2*  dd    [[buffer(3)]],
                         device float*        dst   [[buffer(4)]],
                         constant QLinParams& p     [[buffer(5)]],
                         uint gid  [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    if (sg >= p.m * p.out_f) return;
    uint r = sg / p.out_f;
    uint o = sg % p.out_f;
    uint nb = p.in_f >> 4;
    ulong row16 = (ulong)o * nb;
    ulong rowe = (ulong)o * p.in_f;
    device const float* xr = x + (ulong)r * p.in_f;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        ulong ci = (row16 + b) * 3ul;
        uint u0 = codes[ci], u1 = codes[ci + 1ul], u2 = codes[ci + 2ul];
        short2 s = scm[row16 + b];
        half2 dv = dd[(rowe + ((ulong)b << 4)) >> p.dshift];
        float scale = (float)dv.x * (float)s.x;
        float mn = (float)dv.y * (float)s.y;
        device const float* xb = xr + (b << 4);
        uint c[16];
        c[0] = u0 & 63u;                  c[1] = (u0 >> 6) & 63u;
        c[2] = (u0 >> 12) & 63u;          c[3] = (u0 >> 18) & 63u;
        c[4] = (u0 >> 24) & 63u;          c[5] = ((u0 >> 30) | (u1 << 2)) & 63u;
        c[6] = (u1 >> 4) & 63u;           c[7] = (u1 >> 10) & 63u;
        c[8] = (u1 >> 16) & 63u;          c[9] = (u1 >> 22) & 63u;
        c[10] = ((u1 >> 28) | (u2 << 4)) & 63u;
        c[11] = (u2 >> 2) & 63u;          c[12] = (u2 >> 8) & 63u;
        c[13] = (u2 >> 14) & 63u;         c[14] = (u2 >> 20) & 63u;
        c[15] = (u2 >> 26) & 63u;
        for (uint k = 0; k < 16u; k++) acc += xb[k] * (scale * (float)c[k] + mn);
    }
    acc = simd_sum(acc);
    if (lane == 0u) dst[sg] = acc;
}

// 8-bit codes: one uchar16 = one 16-element block.
kernel void linear_quik8(device const float*   x     [[buffer(0)]],
                         device const uchar4*  codes [[buffer(1)]],
                         device const short2*  scm   [[buffer(2)]],
                         device const half2*   dd    [[buffer(3)]],
                         device float*         dst   [[buffer(4)]],
                         constant QLinParams&  p     [[buffer(5)]],
                         uint gid  [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    if (sg >= p.m * p.out_f) return;
    uint r = sg / p.out_f;
    uint o = sg % p.out_f;
    uint nb = p.in_f >> 4;
    ulong row16 = (ulong)o * nb;
    ulong rowe = (ulong)o * p.in_f;
    device const float* xr = x + (ulong)r * p.in_f;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        ulong ci = (row16 + b) * 4ul;
        short2 s = scm[row16 + b];
        half2 dv = dd[(rowe + ((ulong)b << 4)) >> p.dshift];
        float scale = (float)dv.x * (float)s.x;
        float mn = (float)dv.y * (float)s.y;
        device const float* xb = xr + (b << 4);
        for (uint q = 0; q < 4u; q++) {
            uchar4 cq = codes[ci + q];
            acc += xb[q * 4u]      * (scale * (float)cq.x + mn);
            acc += xb[q * 4u + 1u] * (scale * (float)cq.y + mn);
            acc += xb[q * 4u + 2u] * (scale * (float)cq.z + mn);
            acc += xb[q * 4u + 3u] * (scale * (float)cq.w + mn);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) dst[sg] = acc;
}

// ---- Row-tiled quantized GEMM for prefill (m > 1): one simdgroup per (row-tile, output), RT=8
// rows per tile. Each lane decodes a weight block ONCE into registers and applies it to all 8
// rows — the GEMV kernels above re-stream the whole weight matrix once per row, which is what
// made prefill weight-traffic-bound (m=600+ re-reads). Per-row summation order matches the GEMV
// kernels exactly (same lane-strided blocks, same in-block order), so parity is unchanged.
// DEC(wk, scale, mn): decode the current 16-element block's weights into wk[16], per code width.
#define RT 8u
#define DEC_K4(wk, scale, mn)                                                                     \
    uint2 cw = ((device const uint2*)codes)[bi];                                           \
    for (uint k = 0; k < 8u; k++) {                                                               \
        wk[k]      = scale * (float)((cw.x >> (4u * k)) & 15u) + mn;                              \
        wk[k + 8u] = scale * (float)((cw.y >> (4u * k)) & 15u) + mn;                              \
    }
#define DEC_K6(wk, scale, mn)                                                                     \
    device const uint* cp = (device const uint*)codes + bi * 3ul;                        \
    uint u0 = cp[0], u1 = cp[1], u2 = cp[2];                                                      \
    wk[0]  = scale * (float)(u0 & 63u) + mn;                                                      \
    wk[1]  = scale * (float)((u0 >> 6) & 63u) + mn;                                               \
    wk[2]  = scale * (float)((u0 >> 12) & 63u) + mn;                                              \
    wk[3]  = scale * (float)((u0 >> 18) & 63u) + mn;                                              \
    wk[4]  = scale * (float)((u0 >> 24) & 63u) + mn;                                              \
    wk[5]  = scale * (float)(((u0 >> 30) | (u1 << 2)) & 63u) + mn;                                \
    wk[6]  = scale * (float)((u1 >> 4) & 63u) + mn;                                               \
    wk[7]  = scale * (float)((u1 >> 10) & 63u) + mn;                                              \
    wk[8]  = scale * (float)((u1 >> 16) & 63u) + mn;                                              \
    wk[9]  = scale * (float)((u1 >> 22) & 63u) + mn;                                              \
    wk[10] = scale * (float)(((u1 >> 28) | (u2 << 4)) & 63u) + mn;                                \
    wk[11] = scale * (float)((u2 >> 2) & 63u) + mn;                                               \
    wk[12] = scale * (float)((u2 >> 8) & 63u) + mn;                                               \
    wk[13] = scale * (float)((u2 >> 14) & 63u) + mn;                                              \
    wk[14] = scale * (float)((u2 >> 20) & 63u) + mn;                                              \
    wk[15] = scale * (float)((u2 >> 26) & 63u) + mn;
#define DEC_K8(wk, scale, mn)                                                                     \
    device const uchar4* cp = (device const uchar4*)codes + bi * 4ul;                    \
    for (uint q = 0; q < 4u; q++) {                                                               \
        uchar4 cq = cp[q];                                                                        \
        wk[q * 4u]      = scale * (float)cq.x + mn;                                               \
        wk[q * 4u + 1u] = scale * (float)cq.y + mn;                                               \
        wk[q * 4u + 2u] = scale * (float)cq.z + mn;                                               \
        wk[q * 4u + 3u] = scale * (float)cq.w + mn;                                               \
    }

#define RT_KERNEL(NAME, DEC)                                                                      \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const short2* scm   [[buffer(2)]],                                       \
                 device const half2*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 constant QLinParams& p     [[buffer(5)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]]) {                                      \
    uint sg = gid / 32u;                                                                          \
    uint ntile = (p.m + RT - 1u) / RT;                                                            \
    if (sg >= ntile * p.out_f) return;                                                            \
    uint t = sg / p.out_f;                                                                        \
    uint o = sg % p.out_f;                                                                        \
    uint r0 = t * RT;                                                                             \
    uint rm = min(RT, p.m - r0);                                                                  \
    uint nb = p.in_f >> 4;                                                                        \
    ulong row16 = (ulong)o * nb;                                                                  \
    ulong rowe = (ulong)o * p.in_f;                                                               \
    float acc[RT];                                                                                \
    for (uint rr = 0; rr < RT; rr++) acc[rr] = 0.0f;                                              \
    for (uint b = lane; b < nb; b += 32u) {                                                       \
        ulong bi = row16 + b;                                                                     \
        short2 s = scm[bi];                                                                       \
        half2 dv = dd[(rowe + ((ulong)b << 4)) >> p.dshift];                                      \
        float scale = (float)dv.x * (float)s.x;                                                   \
        float mn = (float)dv.y * (float)s.y;                                                      \
        float wk[16];                                                                             \
        DEC(wk, scale, mn)                                                                        \
        for (uint rr = 0; rr < rm; rr++) {                                                        \
            device const float* xb = x + (ulong)(r0 + rr) * p.in_f + (b << 4);                    \
            for (uint k = 0; k < 16u; k++) acc[rr] += xb[k] * wk[k];                              \
        }                                                                                         \
    }                                                                                             \
    for (uint rr = 0; rr < rm; rr++) {                                                            \
        float v = simd_sum(acc[rr]);                                                              \
        if (lane == 0u) dst[(ulong)(r0 + rr) * p.out_f + o] = v;                                  \
    }                                                                                             \
}

RT_KERNEL(linear_quik4_rt, DEC_K4)
RT_KERNEL(linear_quik6_rt, DEC_K6)
RT_KERNEL(linear_quik8_rt, DEC_K8)

// ---- simdgroup_matrix quantized GEMM for prefill (m >= 16): one simdgroup per 8x8 output tile
// (8 token rows x 8 outputs). Per 16-wide K block: lanes 0..7 each decode one output row's block
// into a per-simdgroup threadgroup tile (via the same DEC_* macros), then two 8x8x8 MMAs consume
// it. The matrix pipeline reuses every loaded operand 8x, where the scalar GEMV/row-tiled kernels
// pay one load per FMA — that ALU/load structure, not weight bandwidth, is what capped prefill.
// The last (partial) row tile falls back to a scalar dot per element; host only routes here when
// out_f % 8 == 0. Threadgroup = 128 threads (4 simdgroups), each with its own 8x16 f32 tile.
#define GEMM_KERNEL(NAME, DEC)                                                                    \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const short2* scm   [[buffer(2)]],                                       \
                 device const half2*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 constant QLinParams& p     [[buffer(5)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]],                                        \
                 uint sgid [[simdgroup_index_in_threadgroup]]) {                                 \
    uint sg = gid / 32u;                                                                          \
    uint ntm = (p.m + 7u) / 8u;                                                                   \
    uint nto = p.out_f / 16u;                                                                     \
    if (sg >= ntm * nto) return;                                                                  \
    uint tm = sg / nto;                                                                           \
    uint to = sg % nto;                                                                           \
    uint r0 = tm * 8u;                                                                            \
    uint o0 = to * 16u;                                                                           \
    uint nb = p.in_f >> 4;                                                                        \
    threadgroup float wt[4][16 * 16];                                                             \
    if (r0 + 8u <= p.m) {                                                                         \
        simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);                                       \
        simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);                                       \
        for (uint kb = 0; kb < nb; kb++) {                                                        \
            if (lane < 16u) {                                                                     \
                ulong bi = (ulong)(o0 + lane) * nb + kb;                                          \
                short2 s = scm[bi];                                                               \
                half2 dv = dd[((ulong)(o0 + lane) * p.in_f + ((ulong)kb << 4)) >> p.dshift];      \
                float scale = (float)dv.x * (float)s.x;                                           \
                float mn = (float)dv.y * (float)s.y;                                              \
                float wk[16];                                                                     \
                DEC(wk, scale, mn)                                                                \
                for (uint k2 = 0; k2 < 16u; k2++) wt[sgid][lane * 16u + k2] = wk[k2];             \
            }                                                                                     \
            simdgroup_barrier(mem_flags::mem_threadgroup);                                        \
            device const float* xp = x + (ulong)r0 * p.in_f + ((ulong)kb << 4);                   \
            simdgroup_float8x8 xa, wb;                                                            \
            for (uint kh = 0; kh < 2u; kh++) {                                                    \
                simdgroup_load(xa, xp + kh * 8u, p.in_f);                                         \
                simdgroup_load(wb, &wt[sgid][kh * 8u], 16, ulong2(0, 0), true);                   \
                simdgroup_multiply_accumulate(acc0, xa, wb, acc0);                                \
                simdgroup_load(wb, &wt[sgid][128u + kh * 8u], 16, ulong2(0, 0), true);            \
                simdgroup_multiply_accumulate(acc1, xa, wb, acc1);                                \
            }                                                                                     \
            simdgroup_barrier(mem_flags::mem_threadgroup);                                        \
        }                                                                                         \
        simdgroup_store(acc0, dst + (ulong)r0 * p.out_f + o0, p.out_f);                           \
        simdgroup_store(acc1, dst + (ulong)r0 * p.out_f + o0 + 8u, p.out_f);                      \
    } else {                                                                                      \
        /* partial row tile: scalar dot per (row, output) element, 4 per lane */                  \
        for (uint e = lane; e < 128u; e += 32u) {                                                 \
            uint rr = r0 + e / 16u;                                                               \
            uint o = o0 + (e % 16u);                                                              \
            if (rr >= p.m) continue;                                                              \
            float a2 = 0.0f;                                                                      \
            for (uint kb = 0; kb < nb; kb++) {                                                    \
                ulong bi = (ulong)o * nb + kb;                                                    \
                short2 s = scm[bi];                                                               \
                half2 dv = dd[((ulong)o * p.in_f + ((ulong)kb << 4)) >> p.dshift];                \
                float scale = (float)dv.x * (float)s.x;                                           \
                float mn = (float)dv.y * (float)s.y;                                              \
                float wk[16];                                                                     \
                DEC(wk, scale, mn)                                                                \
                device const float* xb = x + (ulong)rr * p.in_f + ((ulong)kb << 4);               \
                for (uint k2 = 0; k2 < 16u; k2++) a2 += xb[k2] * wk[k2];                          \
            }                                                                                     \
            dst[(ulong)rr * p.out_f + o] = a2;                                                    \
        }                                                                                         \
    }                                                                                             \
}

GEMM_KERNEL(linear_quik4_mm, DEC_K4)
GEMM_KERNEL(linear_quik6_mm, DEC_K6)
GEMM_KERNEL(linear_quik8_mm, DEC_K8)

// ---- RoPE (NEOX): rotate the first rope_dim of each head; dims beyond pass through. One thread
// per (row, head). `pos`/`ff` buffers are f32. `has_ff` selects the per-pair freq divisor.
struct RopeParams { uint rows; uint n_head; uint head_dim; uint rope_dim; float theta; uint has_ff; };
kernel void rope_f32(device const float* x   [[buffer(0)]],
                     device const float* pos [[buffer(1)]],
                     device const float* ff  [[buffer(2)]],
                     device float*       dst [[buffer(3)]],
                     constant RopeParams& p  [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.n_head) return;
    uint r = gid / p.n_head;
    uint base = gid * p.head_dim;
    for (uint i = 0; i < p.head_dim; i++) dst[base + i] = x[base + i]; // pass-through
    uint hf = p.rope_dim / 2;
    float p0 = pos[r];
    for (uint pp = 0; pp < hf; pp++) {
        uint i0 = pp, i1 = pp + hf;
        float ang = p0 * pow(p.theta, -2.0f * (float)pp / (float)p.rope_dim);
        if (p.has_ff != 0) ang /= ff[pp];
        float c = cos(ang), s = sin(ang);
        float a = x[base + i0], b = x[base + i1];
        dst[base + i0] = a * c - b * s;
        dst[base + i1] = a * s + b * c;
    }
}

// ---- Fused per-head RMSNorm + RoPE (QkNormRope): rmsnorm (× weight) then rotate the normed head.
// One SIMD group per (row, head): `simd_sum` for the norm, then lanes split the rotation pairs. Each
// lane forms its normed values straight from `x` (× s × w), so no cross-lane read of `dst` — no
// barrier needed. Pass-through dims [rope_dim, head_dim) are written normed in the tail loop.
struct QkRopeParams { uint rows; uint n_head; uint head_dim; uint rope_dim; float theta; float eps; uint has_ff; };
kernel void qknormrope_f32(device const float* x   [[buffer(0)]],
                           device const float* w   [[buffer(1)]],
                           device const float* pos [[buffer(2)]],
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
    float p0 = pos[r];
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

// ---- Gated FFN activation: dst[r,i] = act(gate[r,i]) * up[r, i + up_off]. act: 0=SiLU,1=GELU,2=Sigmoid
inline float gated_act(uint act, float g) {
    if (act == 0u) return g / (1.0f + exp(-g));                       // SiLU
    if (act == 2u) return 1.0f / (1.0f + exp(-g));                    // Sigmoid
    // GELU (gelu_pytorch_tanh)
    return 0.5f * g * (1.0f + tanh(0.7978845608f * (g + 0.044715f * g * g * g)));
}
struct GatedParams { uint rows; uint nff; uint act; uint up_off; };
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

// ---- Scaled-dot-product attention (GQA + causal/sliding-window). One SIMD group (32 lanes) per
// (query, head): lanes split head_dim — the q·k score is a lane-strided dot reduced by `simd_sum`,
// and each lane owns a head_dim/32 slice of the online-softmax `acc`. All lanes see the same score,
// so `m`/`l` stay in sync with no cross-lane state. Fixes the old one-thread-per-(query,head) kernel,
// where decode (1 query) ran each head's whole O(kv_len·head_dim) pass on a single thread.
constant constexpr uint MAX_HD = 256;
constant constexpr uint MAX_DPL = MAX_HD / 32u;   // head_dim slots per lane (head_dim ≤ MAX_HD)
struct AttnParams { uint rows; uint kv_len; uint n_head; uint n_kv; uint head_dim; float scale; uint window; uint pos; };
kernel void attention_f32(device const float* q   [[buffer(0)]],
                          device const float* k   [[buffer(1)]],
                          device const float* v   [[buffer(2)]],
                          device float*       dst [[buffer(3)]],
                          constant AttnParams& p  [[buffer(4)]],
                          uint gid  [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    if (sg >= p.rows * p.n_head) return;
    uint ti = sg / p.n_head;
    uint h = sg % p.n_head;
    uint group = p.n_head / p.n_kv;
    uint kvh = h / group;
    uint qb = sg * p.head_dim;                    // (ti*n_head + h) * head_dim
    uint abs = p.pos + ti;                         // absolute position of this query
    uint lo = (p.window > 0u && abs + 1u > p.window) ? (abs + 1u - p.window) : 0u;

    float acc[MAX_DPL];
    for (uint t = 0; t < MAX_DPL; t++) acc[t] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint j = lo; j <= abs; j++) {
        uint kb = (j * p.n_kv + kvh) * p.head_dim;
        float part = 0.0f;
        for (uint d = lane; d < p.head_dim; d += 32u) part += q[qb + d] * k[kb + d];
        float sc = simd_sum(part) * p.scale;
        float mnew = max(m, sc);
        float corr = exp(m - mnew);
        float pw = exp(sc - mnew);
        l = l * corr + pw;
        uint vb = (j * p.n_kv + kvh) * p.head_dim;
        uint t = 0;
        for (uint d = lane; d < p.head_dim; d += 32u) { acc[t] = acc[t] * corr + pw * v[vb + d]; t++; }
        m = mnew;
    }
    uint t = 0;
    for (uint d = lane; d < p.head_dim; d += 32u) { dst[qb + d] = acc[t] / l; t++; }
}

// Same as attention_f32, but reads the KV cache in its native f16 straight from the bound buffer
// (no host materialize-to-f32 round-trip). Values match the CPU's f16→f32 read exactly.
kernel void attention_f16kv(device const float* q   [[buffer(0)]],
                            device const half*  k   [[buffer(1)]],
                            device const half*  v   [[buffer(2)]],
                            device float*       dst [[buffer(3)]],
                            constant AttnParams& p  [[buffer(4)]],
                            uint gid  [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    if (sg >= p.rows * p.n_head) return;
    uint ti = sg / p.n_head;
    uint h = sg % p.n_head;
    uint group = p.n_head / p.n_kv;
    uint kvh = h / group;
    uint qb = sg * p.head_dim;
    uint abs = p.pos + ti;
    uint lo = (p.window > 0u && abs + 1u > p.window) ? (abs + 1u - p.window) : 0u;

    float acc[MAX_DPL];
    for (uint t = 0; t < MAX_DPL; t++) acc[t] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint j = lo; j <= abs; j++) {
        uint kb = (j * p.n_kv + kvh) * p.head_dim;
        float part = 0.0f;
        for (uint d = lane; d < p.head_dim; d += 32u) part += q[qb + d] * (float)k[kb + d];
        float sc = simd_sum(part) * p.scale;
        float mnew = max(m, sc);
        float corr = exp(m - mnew);
        float pw = exp(sc - mnew);
        l = l * corr + pw;
        uint vb = (j * p.n_kv + kvh) * p.head_dim;
        uint t = 0;
        for (uint d = lane; d < p.head_dim; d += 32u) { acc[t] = acc[t] * corr + pw * (float)v[vb + d]; t++; }
        m = mnew;
    }
    uint t = 0;
    for (uint d = lane; d < p.head_dim; d += 32u) { dst[qb + d] = acc[t] / l; t++; }
}

// ---- Split-KV ("flash-decode") attention: same math as attention_*, but NSG simdgroups per
// (query, head) threadgroup, each running a private online softmax over a strided slice of the KV
// positions, merged at the end through threadgroup memory (rescale each partial to the global max,
// sum, divide). Exists because decode has rows=1: the one-simdgroup kernel then launches only
// `n_head` simdgroups — far too few to occupy the GPU — and its runtime grows O(kv_len) on that
// fixed tiny width. Split kernels multiply decode parallelism by NSG; the host routes here only
// when rows*n_head is small, so prefill keeps the leaner kernel (this one's static ~8 KB of
// threadgroup memory would cap prefill occupancy).
// One macro instantiates each (KV type, split width) variant. NSG=8 covers short contexts and any
// head_dim; NSG=32 quarters the serial online-softmax chain per simdgroup (the kernel is
// latency-bound on that chain, ~kv_len/NSG dependent steps), but its threadgroup accumulator only
// fits head_dim <= 128 in the 32 KB threadgroup-memory budget, so the host routes to it only for
// long-context decode at hd <= 128.
#define ATTNSPLIT_KERNEL(NAME, KVT, NSG, MAXHD)                                                    \
kernel void NAME(device const float* q   [[buffer(0)]],                                           \
                 device const KVT*   k   [[buffer(1)]],                                           \
                 device const KVT*   v   [[buffer(2)]],                                           \
                 device float*       dst [[buffer(3)]],                                           \
                 constant AttnParams& p  [[buffer(4)]],                                           \
                 uint3 tgpig [[threadgroup_position_in_grid]],                                    \
                 uint sgid [[simdgroup_index_in_threadgroup]],                                    \
                 uint lane [[thread_index_in_simdgroup]]) {                                       \
    uint tg = tgpig.x;                                                                            \
    if (tg >= p.rows * p.n_head) return;   /* uniform per threadgroup — safe with the barrier */  \
    uint ti = tg / p.n_head;                                                                      \
    uint h = tg % p.n_head;                                                                       \
    uint group = p.n_head / p.n_kv;                                                               \
    uint kvh = h / group;                                                                         \
    uint qb = tg * p.head_dim;                                                                    \
    uint abs = p.pos + ti;                                                                        \
    uint lo = (p.window > 0u && abs + 1u > p.window) ? (abs + 1u - p.window) : 0u;                \
                                                                                                  \
    float acc[MAXHD / 32u];                                                                       \
    for (uint t = 0; t < MAXHD / 32u; t++) acc[t] = 0.0f;                                         \
    float m = -INFINITY, l = 0.0f;                                                                \
    for (uint j = lo + sgid; j <= abs; j += NSG) {                                                \
        uint kb = (j * p.n_kv + kvh) * p.head_dim;                                                \
        float part = 0.0f;                                                                        \
        for (uint d = lane; d < p.head_dim; d += 32u) part += q[qb + d] * (float)k[kb + d];       \
        float sc = simd_sum(part) * p.scale;                                                      \
        float mnew = max(m, sc);                                                                  \
        float corr = exp(m - mnew);                                                               \
        float pw = exp(sc - mnew);                                                                \
        l = l * corr + pw;                                                                        \
        uint t = 0;                                                                               \
        for (uint d = lane; d < p.head_dim; d += 32u) {                                           \
            acc[t] = acc[t] * corr + pw * (float)v[kb + d];                                       \
            t++;                                                                                  \
        }                                                                                         \
        m = mnew;                                                                                 \
    }                                                                                             \
    /* Merge the NSG partials. A simdgroup whose slice was empty has l==0 (skip; its m is -inf) */ \
    threadgroup float tg_m[NSG], tg_l[NSG], tg_acc[NSG * MAXHD];                                  \
    if (lane == 0u) { tg_m[sgid] = m; tg_l[sgid] = l; }                                           \
    uint t = 0;                                                                                   \
    for (uint d = lane; d < p.head_dim; d += 32u) {                                               \
        tg_acc[sgid * p.head_dim + d] = acc[t];                                                   \
        t++;                                                                                      \
    }                                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
    if (sgid == 0u) {                                                                             \
        float gm = -INFINITY;                                                                     \
        for (uint i = 0; i < NSG; i++) if (tg_l[i] > 0.0f) gm = max(gm, tg_m[i]);                 \
        float gl = 0.0f;                                                                          \
        float w[NSG];                                                                             \
        for (uint i = 0; i < NSG; i++) {                                                          \
            w[i] = (tg_l[i] > 0.0f) ? exp(tg_m[i] - gm) : 0.0f;                                   \
            gl += tg_l[i] * w[i];                                                                 \
        }                                                                                         \
        for (uint d = lane; d < p.head_dim; d += 32u) {                                           \
            float s = 0.0f;                                                                       \
            for (uint i = 0; i < NSG; i++) s += tg_acc[i * p.head_dim + d] * w[i];                \
            dst[qb + d] = s / gl;                                                                 \
        }                                                                                         \
    }                                                                                             \
}

ATTNSPLIT_KERNEL(attnsplit_f32, float, 8u, 256u)
ATTNSPLIT_KERNEL(attnsplit_f16kv, half, 8u, 256u)
ATTNSPLIT_KERNEL(attnsplit32_f32, float, 32u, 128u)
ATTNSPLIT_KERNEL(attnsplit32_f16kv, half, 32u, 128u)

// ---- WriteKv: cast-copy `n` f32 source elems into the bound KV cache at row offset `base`, on the
// GPU so it stays in the batch (no host round-trip that would force a per-layer flush). The (half)
// cast is IEEE round-to-nearest-even — byte-identical to the host `f16::from_f32` reference.
struct WriteKvParams { uint n; uint base; };
kernel void writekv_f16(device const float* src   [[buffer(0)]],
                        device half*        cache [[buffer(1)]],
                        constant WriteKvParams& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    cache[p.base + gid] = (half)src[gid];
}
kernel void writekv_f32(device const float* src   [[buffer(0)]],
                        device float*       cache [[buffer(1)]],
                        constant WriteKvParams& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    cache[p.base + gid] = src[gid];
}
"#;
