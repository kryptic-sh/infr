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

// ---- Linear over a NATIVE quantized weight. Two on-device forms:
//
// * FACTORED (`dequant_factored`): bit-packed 4/6/8-bit codes + one (sc, m) i16 pair per
//   16-element block + one (d, dmin) f16 pair per 2^dshift elements — for the affine formats
//   without a fast native decoder (legacy quants, Q2_K/Q3_K/Q5_K).
// * NATIVE GGUF blocks for Q4_K (144 B / 256 elems, ~4.5 bpw) and Q6_K (210 B, ~6.6 bpw): the
//   kernel decodes the raw block bytes — the bound weight buffer is used as-is, no host repack,
//   no extra residency, and the weight stream shrinks to the format's true size (the factored
//   form paid ~6.1/8.1 bpw). Decode GEMV is bound on this stream, so bits are throughput.
//
// Every decode reproduces the host dequant reference bit-for-bit: the same f32 products
// (f32(d)·f32(int)) and the same `scale*code + min` operation order.
//
// One DEC16_* macro per format decodes the 16-element block with global index `bi` into wk[16]
// (ambient: `codes`, `scm`, `dd`, `p`, `bi`). Three kernel shapes instantiate each format:
// GEMV (one simdgroup per output, decode), RT (row-tiled, small m), MM (simdgroup_matrix GEMM,
// prefill).
struct QLinParams { uint m; uint in_f; uint out_f; uint dshift; };

// factored, 4-bit codes: one uint2 = 8 bytes = one 16-element block, code k at bits 4k.
#define DEC16_K4(wk)                                                                              \
    short2 s = ((device const short2*)scm)[bi];                                                   \
    half2 dv = ((device const half2*)dd)[((ulong)bi << 4) >> p.dshift];                           \
    float scale = (float)dv.x * (float)s.x;                                                       \
    float mn = (float)dv.y * (float)s.y;                                                          \
    uint2 cw = ((device const uint2*)codes)[bi];                                                  \
    for (uint k = 0; k < 8u; k++) {                                                               \
        wk[k]      = scale * (float)((cw.x >> (4u * k)) & 15u) + mn;                              \
        wk[k + 8u] = scale * (float)((cw.y >> (4u * k)) & 15u) + mn;                              \
    }

// factored, 6-bit codes: three uints = 12 bytes = one 16-element block, code k at bits 6k.
#define DEC16_K6(wk)                                                                              \
    short2 s = ((device const short2*)scm)[bi];                                                   \
    half2 dv = ((device const half2*)dd)[((ulong)bi << 4) >> p.dshift];                           \
    float scale = (float)dv.x * (float)s.x;                                                       \
    float mn = (float)dv.y * (float)s.y;                                                         \
    device const uint* cp = (device const uint*)codes + bi * 3ul;                                 \
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

// factored, 8-bit codes: one uchar16 = one 16-element block.
#define DEC16_K8(wk)                                                                              \
    short2 s = ((device const short2*)scm)[bi];                                                   \
    half2 dv = ((device const half2*)dd)[((ulong)bi << 4) >> p.dshift];                           \
    float scale = (float)dv.x * (float)s.x;                                                       \
    float mn = (float)dv.y * (float)s.y;                                                         \
    device const uchar4* cp = (device const uchar4*)codes + bi * 4ul;                             \
    for (uint q = 0; q < 4u; q++) {                                                               \
        uchar4 cq = cp[q];                                                                        \
        wk[q * 4u]      = scale * (float)cq.x + mn;                                               \
        wk[q * 4u + 1u] = scale * (float)cq.y + mn;                                               \
        wk[q * 4u + 2u] = scale * (float)cq.z + mn;                                               \
        wk[q * 4u + 3u] = scale * (float)cq.w + mn;                                               \
    }

// NATIVE Q4_K block (144 B / 256 elems): [f16 d][f16 dmin][12 B 6-bit scales][128 B nibbles].
// 16-block `sub` within the 256-block: quarter j = sub/4 (qs bytes j*32..), low/high nibble half
// `hi`, and which 16 of the 32 l-values `l0`. Scale/min pair via get_scale_min_k4(2j + hi).
// scm/dd are unused (dummy bindings). 144 % 4 == 0, so uint loads stay aligned.
#define DEC16_Q4K(wk)                                                                             \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 144ul;                                   \
    uint sub = bi & 15u;                                                                          \
    uint j = sub >> 2;                                                                            \
    uint hi = (sub >> 1) & 1u;                                                                    \
    uint l0 = (sub & 1u) * 16u;                                                                   \
    uint dm = *(device const uint*)blk;                                                           \
    float d = (float)as_type<half>((ushort)(dm & 0xFFFFu));                                       \
    float dmin = (float)as_type<half>((ushort)(dm >> 16));                                        \
    device const uchar* scb = blk + 4u;                                                           \
    uint jj = 2u * j + hi;                                                                        \
    uint sc6, m6;                                                                                 \
    if (jj < 4u) {                                                                                \
        sc6 = scb[jj] & 63u;                                                                      \
        m6 = scb[jj + 4u] & 63u;                                                                  \
    } else {                                                                                      \
        sc6 = (scb[jj + 4u] & 0x0Fu) | ((scb[jj - 4u] >> 6) << 4);                                \
        m6 = (scb[jj + 4u] >> 4) | ((scb[jj] >> 6) << 4);                                         \
    }                                                                                             \
    float scale = d * (float)sc6;                                                                 \
    float mn = -(dmin * (float)m6);                                                               \
    device const uint* qw4 = (device const uint*)(blk + 16u + j * 32u + l0);                      \
    for (uint w = 0; w < 4u; w++) {                                                               \
        uint u = qw4[w];                                                                          \
        for (uint k2 = 0; k2 < 4u; k2++) {                                                        \
            uint byt = (u >> (8u * k2)) & 0xFFu;                                                  \
            uint q = hi ? (byt >> 4) : (byt & 0xFu);                                              \
            wk[w * 4u + k2] = scale * (float)q + mn;                                              \
        }                                                                                         \
    }

// NATIVE Q6_K block (210 B / 256 elems): [128 B ql][64 B qh][16 x i8 scales][f16 d].
// 16-block `sub`: half h6 = sub/8 (128 elems each), then group off = (sub%8 / 2)*32 with
// is = sub%2 selecting which 16 l-values; q built from ql/qh nibble+2-bit pieces, scale from the
// i8 scale at [h6*8 + off/16 + is], min = -32*scale (exact power-of-two scaling of the same f32
// product the reference computes). 210 % 4 != 0, so all loads are byte loads.
#define DEC16_Q6K(wk)                                                                             \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 210ul;                                   \
    uint sub = bi & 15u;                                                                          \
    uint h6 = sub >> 3;                                                                           \
    uint sub8 = sub & 7u;                                                                         \
    uint off = (sub8 >> 1) * 32u;                                                                 \
    uint is = sub8 & 1u;                                                                          \
    device const uchar* ql = blk + h6 * 64u;                                                      \
    device const uchar* qh = blk + 128u + h6 * 32u;                                               \
    device const uchar* scs = blk + 192u;                                                         \
    float d = (float)as_type<half>((ushort)(blk[208] | ((ushort)blk[209] << 8)));                 \
    float scale = d * (float)(char)scs[h6 * 8u + (off >> 4) + is];                                \
    float mn = -32.0f * scale;                                                                    \
    uint lb = is * 16u;                                                                           \
    /* branch-free (lanes hold different `off`s; a 4-way if would serialize the simdgroup): */    \
    /* ql byte at l (+32 for the odd 32-groups), low/high nibble by off>=64, qh bits at off/16 */ \
    uint qlo = (off & 32u);                                                                       \
    uint nsh = (off >= 64u) ? 4u : 0u;                                                            \
    uint qhs = off >> 4;                                                                          \
    for (uint k = 0; k < 16u; k++) {                                                              \
        uint l = lb + k;                                                                          \
        uint q = ((ql[l + qlo] >> nsh) & 0xFu) | (((qh[l] >> qhs) & 3u) << 4);                    \
        wk[k] = scale * (float)q + mn;                                                            \
    }

// GEMV: one simdgroup (32 lanes) per output element; each lane decodes one 16-element block per
// step, coalesced across lanes, `simd_sum` reduction. m=1 decode is bound on the weight stream.
#define GEMV_KERNEL(NAME, DEC)                                                                    \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 constant QLinParams& p     [[buffer(5)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]]) {                                      \
    uint sg = gid / 32u;                                                                          \
    if (sg >= p.m * p.out_f) return;                                                              \
    uint r = sg / p.out_f;                                                                        \
    uint o = sg % p.out_f;                                                                        \
    uint nb = p.in_f >> 4;                                                                        \
    ulong row16 = (ulong)o * nb;                                                                  \
    device const float* xr = x + (ulong)r * p.in_f;                                               \
    float acc = 0.0f;                                                                             \
    for (uint b = lane; b < nb; b += 32u) {                                                       \
        ulong bi = row16 + b;                                                                     \
        float wk[16];                                                                             \
        DEC(wk)                                                                                   \
        device const float* xb = xr + (b << 4);                                                   \
        for (uint k = 0; k < 16u; k++) acc += xb[k] * wk[k];                                      \
    }                                                                                             \
    acc = simd_sum(acc);                                                                          \
    if (lane == 0u) dst[sg] = acc;                                                                \
}

// Row-tiled (1 < m < 16, or shapes the GEMM kernel can't take): one simdgroup per (8-row tile,
// output); each lane decodes a block once and applies it to all rows.
#define RT_KERNEL(NAME, DEC)                                                                      \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 constant QLinParams& p     [[buffer(5)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]]) {                                      \
    uint sg = gid / 32u;                                                                          \
    uint ntile = (p.m + 7u) / 8u;                                                                 \
    if (sg >= ntile * p.out_f) return;                                                            \
    uint t = sg / p.out_f;                                                                        \
    uint o = sg % p.out_f;                                                                        \
    uint r0 = t * 8u;                                                                             \
    uint rm = min(8u, p.m - r0);                                                                  \
    uint nb = p.in_f >> 4;                                                                        \
    ulong row16 = (ulong)o * nb;                                                                  \
    float acc[8];                                                                                 \
    for (uint rr = 0; rr < 8u; rr++) acc[rr] = 0.0f;                                              \
    for (uint b = lane; b < nb; b += 32u) {                                                       \
        ulong bi = row16 + b;                                                                     \
        float wk[16];                                                                             \
        DEC(wk)                                                                                   \
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


// ---- Cast f32 -> f16 (prefill GEMM feeds on half activations; round-to-nearest-even).
kernel void cast_f32_f16(device const float* src [[buffer(0)]],
                         device half*        dst [[buffer(1)]],
                         constant uint&      n   [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < n) dst[gid] = (half)src[gid];
}

// Half-fragment GEMM: same 8x16-tile shape as GEMM_KERNEL, but x and the staged weight tile are
// f16 and the MMAs run half x half -> float (Apple's mixed-precision simdgroup_multiply_accumulate,
// double the f32 matrix rate and half the staging bytes). Accumulation stays f32. This is the one
// path that is NOT bit-exact against the dequant reference: weights and activations round to f16
// (~5e-4 relative) — the llama.cpp trade, well under quantization error. Decode (GEMV) stays f32.
#define HGEMM_KERNEL(NAME, DEC)                                                                   \
kernel void NAME(device const half*   x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 constant QLinParams& p     [[buffer(5)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]],                                        \
                 uint sgid [[simdgroup_index_in_threadgroup]]) {                                 \
    uint sg = gid / 32u;                                                                          \
    uint ntm = (p.m + 31u) / 32u;                                                                 \
    uint nto = p.out_f / 16u;                                                                     \
    if (sg >= ntm * nto) return;                                                                  \
    uint tm = sg / nto;                                                                           \
    uint to = sg % nto;                                                                           \
    uint r0 = tm * 32u;                                                                           \
    uint o0 = to * 16u;                                                                           \
    uint nb = p.in_f >> 4;                                                                        \
    threadgroup half wt[4][16 * 16];                                                              \
    if (r0 + 32u <= p.m) {                                                                        \
        simdgroup_float8x8 acc[4][2];                                                             \
        for (uint i = 0; i < 4u; i++)                                                             \
            for (uint jx = 0; jx < 2u; jx++) acc[i][jx] = simdgroup_float8x8(0.0f);               \
        for (uint kb = 0; kb < nb; kb++) {                                                        \
            if (lane < 16u) {                                                                     \
                ulong bi = (ulong)(o0 + lane) * nb + kb;                                          \
                float wk[16];                                                                     \
                DEC(wk)                                                                           \
                for (uint k2 = 0; k2 < 16u; k2++) wt[sgid][lane * 16u + k2] = (half)wk[k2];       \
            }                                                                                     \
            simdgroup_barrier(mem_flags::mem_threadgroup);                                        \
            for (uint kh = 0; kh < 2u; kh++) {                                                    \
                simdgroup_half8x8 wb0, wb1;                                                       \
                simdgroup_load(wb0, &wt[sgid][kh * 8u], 16, ulong2(0, 0), true);                  \
                simdgroup_load(wb1, &wt[sgid][128u + kh * 8u], 16, ulong2(0, 0), true);           \
                for (uint rh = 0; rh < 4u; rh++) {                                                \
                    simdgroup_half8x8 xa;                                                         \
                    device const half* xp =                                                       \
                        x + (ulong)(r0 + rh * 8u) * p.in_f + ((ulong)kb << 4) + kh * 8u;          \
                    simdgroup_load(xa, xp, p.in_f);                                               \
                    simdgroup_multiply_accumulate(acc[rh][0], xa, wb0, acc[rh][0]);               \
                    simdgroup_multiply_accumulate(acc[rh][1], xa, wb1, acc[rh][1]);               \
                }                                                                                 \
            }                                                                                     \
            simdgroup_barrier(mem_flags::mem_threadgroup);                                        \
        }                                                                                         \
        for (uint rh = 0; rh < 4u; rh++)                                                          \
            for (uint oh = 0; oh < 2u; oh++)                                                      \
                simdgroup_store(acc[rh][oh],                                                      \
                                dst + (ulong)(r0 + rh * 8u) * p.out_f + o0 + oh * 8u, p.out_f);   \
    } else {                                                                                      \
        /* partial row tile: scalar dot per (row, output) element */                              \
        for (uint e = lane; e < 512u; e += 32u) {                                                 \
            uint rr = r0 + e / 16u;                                                               \
            uint o = o0 + (e % 16u);                                                              \
            if (rr >= p.m) continue;                                                              \
            float a2 = 0.0f;                                                                      \
            for (uint kb = 0; kb < nb; kb++) {                                                    \
                ulong bi = (ulong)o * nb + kb;                                                    \
                float wk[16];                                                                     \
                DEC(wk)                                                                           \
                device const half* xb = x + (ulong)rr * p.in_f + ((ulong)kb << 4);                \
                for (uint k2 = 0; k2 < 16u; k2++) a2 += (float)xb[k2] * (float)((half)wk[k2]);    \
            }                                                                                     \
            dst[(ulong)rr * p.out_f + o] = a2;                                                    \
        }                                                                                         \
    }                                                                                             \
}

// Decode GEMV for the native K-quant formats, mul_mv shape (ported from llama.cpp's
// kernel_mul_mv_q4_K_f32 / q6_K and adapted to our buffers): each simdgroup computes TWO output
// rows; activations load once into registers and are reused across both rows, and the inner loop
// is decode-free — masked integer nibble ops with the block scales applied once per group (Q4_K
// splits the affine min out via pre-summed activations; the 1/256 and 1/16 factors are exact
// power-of-two corrections for the high-nibble/high-byte lanes). Algebraically identical to the
// reference dequant dot, floating-point reassociated. This access pattern (4 blocks in flight per
// simdgroup for Q4_K, contiguous 8-element runs per lane) is what the sub-block-scatter DEC16
// GEMV left on the table.
kernel void linear_q4k(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint first_row = (gid / 32u) * 2u;
    if (first_row >= p.out_f) return;
    uint nb = p.in_f >> 8;                 // 256-element blocks per row
    ulong row_b = (ulong)nb * 144ul;       // row stride in bytes
    device const uchar* xr = codes + first_row * row_b;

    const ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    uint ix = lane >> 3;                   // 0..3: which of 4 blocks in flight
    uint it = lane & 7u;
    uint iq = it >> 2;                     // 0/1: which 128-half
    uint ir = it & 3u;                     // 0..3: which 8-element run

    float yl[16], yh[16];
    float sumf[2] = {0.0f, 0.0f};
    device const float* y4 = x + ix * 256u + 64u * iq + 8u * ir;

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        float4 sumy = float4(0.0f);
        for (uint i = 0; i < 8u; i++) {
            yl[i]      = y4[i];        sumy[0] += yl[i];
            yl[i + 8u] = y4[i + 32u];  sumy[1] += yl[i + 8u];
            yh[i]      = y4[i + 128u]; sumy[2] += yh[i];
            yh[i + 8u] = y4[i + 160u]; sumy[3] += yh[i + 8u];
        }
        device const uchar* blk = xr + (ulong)ib * 144ul;
        device const ushort* sc = (device const ushort*)(blk + 4u) + iq;
        device const ushort* q1 = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
        device const half* dh = (device const half*)blk;

        for (uint row = 0; row < 2u; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);
            for (uint i = 0; i < 4u; i++) {
                acc1[0] += yl[2u * i]      * (float)(q1[i] & 0x000F);
                acc1[1] += yl[2u * i + 1u] * (float)(q1[i] & 0x0F00);
                acc1[2] += yl[2u * i + 8u] * (float)(q1[i] & 0x00F0);
                acc1[3] += yl[2u * i + 9u] * (float)(q1[i] & 0xF000);
                acc2[0] += yh[2u * i]      * (float)(q2[i] & 0x000F);
                acc2[1] += yh[2u * i + 1u] * (float)(q2[i] & 0x0F00);
                acc2[2] += yh[2u * i + 8u] * (float)(q2[i] & 0x00F0);
                acc2[3] += yh[2u * i + 9u] * (float)(q2[i] & 0xF000);
            }
            sumf[row] += (float)dh[0] * ((acc1[0] + 1.0f/256.0f * acc1[1]) * sc8[0] +
                                         (acc1[2] + 1.0f/256.0f * acc1[3]) * sc8[1] * 1.0f/16.0f +
                                         (acc2[0] + 1.0f/256.0f * acc2[1]) * sc8[4] +
                                         (acc2[2] + 1.0f/256.0f * acc2[3]) * sc8[5] * 1.0f/16.0f) -
                         (float)dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                         sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += row_b / 2u;
            sc += row_b / 2u;
            dh += row_b / 2u;
        }
        y4 += 4u * 256u;
    }
    for (uint row = 0; row < 2u && first_row + row < p.out_f; row++) {
        float s = simd_sum(sumf[row]);
        if (lane == 0u) dst[first_row + row] = s;
    }
}

kernel void linear_q6k(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint first_row = (gid / 32u) * 2u;
    if (first_row >= p.out_f) return;
    uint nb = p.in_f >> 8;
    ulong row_b = (ulong)nb * 210ul;
    device const uchar* xr = codes + first_row * row_b;

    const uchar kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    uint tid2 = lane >> 1;
    uint ix = lane & 1u;                   // 0/1: which of 2 blocks in flight
    uint ip = tid2 >> 3;                   // 0/1: which 128-half
    uint il = tid2 & 7u;
    uint l0 = 4u * il;
    uint is = 8u * ip + (l0 >> 4);
    uint y_off = 128u * ip + l0;
    uint ql_off = 64u * ip + l0;
    uint qh_off = 32u * ip + l0;

    float sumf[2] = {0.0f, 0.0f};
    float yl[16];

    for (uint i = ix; i < nb; i += 2u) {
        device const uchar* blk = xr + (ulong)i * 210ul;
        device const uchar* q1 = blk + ql_off;
        device const uchar* q2 = q1 + 32u;
        device const uchar* qh = blk + 128u + qh_off;
        device const char* sc = (device const char*)(blk + 192u) + is;
        device const half* dh = (device const half*)(blk + 208u);
        device const float* y = x + i * 256u + y_off;

        for (uint l = 0; l < 4u; l++) {
            yl[4u * l]      = y[l];
            yl[4u * l + 1u] = y[l + 32u];
            yl[4u * l + 2u] = y[l + 64u];
            yl[4u * l + 3u] = y[l + 96u];
        }
        for (uint row = 0; row < 2u; row++) {
            float4 sums = float4(0.0f);
            for (uint l = 0; l < 4u; l++) {
                sums[0] += yl[4u * l]      * (float)((char)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4u * l + 1u] * (float)((char)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4u * l + 2u] * (float)((char)((q1[l] >> 4)  | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4u * l + 3u] * (float)((char)((q2[l] >> 4)  | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += (float)dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                         sums[2] * sc[4] + sums[3] * sc[6]);
            q1 += row_b;
            q2 += row_b;
            qh += row_b;
            sc += row_b;
            dh += row_b / 2u;
        }
    }
    for (uint row = 0; row < 2u && first_row + row < p.out_f; row++) {
        float s = simd_sum(sumf[row]);
        if (lane == 0u) dst[first_row + row] = s;
    }
}

GEMV_KERNEL(linear_quik4, DEC16_K4)
GEMV_KERNEL(linear_quik6, DEC16_K6)
GEMV_KERNEL(linear_quik8, DEC16_K8)
RT_KERNEL(linear_quik4_rt, DEC16_K4)
RT_KERNEL(linear_quik6_rt, DEC16_K6)
RT_KERNEL(linear_quik8_rt, DEC16_K8)
RT_KERNEL(linear_q4k_rt, DEC16_Q4K)
RT_KERNEL(linear_q6k_rt, DEC16_Q6K)
// Cooperative-tile half-fragment GEMM, mul_mm-shape: one 64-output x 32-token tile per 128-thread
// threadgroup, NK=32 K-steps. What the simpler cooperative tile above lacked (each of its shapes
// measured and replaced): weights AND activations are staged into threadgroup memory as
// CONTIGUOUS 8x8 half tiles (stride-8, conflict-free simdgroup_load; the weight tile is written
// pre-transposed so no transposed loads), each thread dequantizes exactly one 16-element block
// per K-step (128 threads = the whole 64x32 weight tile), and x is cast f32->f16 inline during
// staging (no separate cast pass). Each simdgroup owns a 32-output x 16-token quadrant as 8 f32
// accumulators; a partial token tile stages through threadgroup memory and row-guards the copy,
// so every tile takes the MMA path. Requires out_f % 64 == 0 and in_f % 32 == 0; other shapes
// fall back to the per-simdgroup HGEMM.
#define CMM_KERNEL(NAME, DEC)                                                                     \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 constant QLinParams& p     [[buffer(5)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]],                                        \
                 uint sgid [[simdgroup_index_in_threadgroup]]) {                                 \
    uint tgix = gid / 128u;                                                                       \
    uint nto = p.out_f / 64u;                                                                     \
    uint ntm = (p.m + 31u) / 32u;                                                                 \
    if (tgix >= ntm * nto) return;                                                                \
    uint to = tgix % nto;                                                                         \
    uint tm = tgix / nto;                                                                         \
    uint ro = to * 64u;                                                                           \
    uint rt = tm * 32u;                                                                           \
    uint tid = sgid * 32u + lane;                                                                 \
    uint nr1 = min(32u, p.m - rt);                                                                \
                                                                                                  \
    threadgroup float shraw[2048];  /* 8 KB: sa(4K half) + sb(2K half); reused f32 for stores */  \
    threadgroup half* sa = (threadgroup half*)shraw;                                              \
    threadgroup half* sb = ((threadgroup half*)shraw) + 2048u;                                    \
                                                                                                  \
    uint lr0 = tid >> 1;                 /* weight (output) row 0..63 */                          \
    uint il0 = tid & 1u;                 /* which 16-element half of the 32-K step */             \
    uint lr1 = tid >> 2;                 /* token row 0..31 */                                    \
    uint iyk = (tid & 3u) * 8u;          /* k offset within the 32-K step */                      \
    uint lr1c = min(lr1, nr1 - 1u);      /* clamp token loads to the matrix edge */               \
                                                                                                  \
    simdgroup_half8x8 ma[4];                                                                      \
    simdgroup_half8x8 mb[2];                                                                      \
    simdgroup_float8x8 mc[8];                                                                     \
    for (uint i = 0; i < 8u; i++) mc[i] = simdgroup_float8x8(0.0f);                               \
                                                                                                  \
    uint nb = p.in_f >> 4;                                                                        \
    for (uint k0 = 0; k0 < p.in_f; k0 += 32u) {                                                   \
        {   /* stage A: one 16-block per thread, into pre-transposed 8x8 tiles */                 \
            ulong bi = (ulong)(ro + lr0) * nb + (ulong)(k0 >> 4) + il0;                           \
            float wk[16];                                                                         \
            DEC(wk)                                                                               \
            uint sy = lr0 >> 3;                                                                   \
            uint lx = lr0 & 7u;                                                                   \
            for (uint i = 0; i < 16u; i++) {                                                      \
                uint sx = 2u * il0 + (i >> 3);                                                    \
                sa[64u * (8u * sx + sy) + 8u * (i & 7u) + lx] = (half)wk[i];                      \
            }                                                                                     \
        }                                                                                         \
        {   /* stage B: 8 activations per thread, f32 -> f16 inline */                            \
            device const float* yy = x + (ulong)(rt + lr1c) * p.in_f + k0 + iyk;                  \
            uint ib = 4u * (tid & 3u) + (lr1 >> 3);                                               \
            uint ly = lr1 & 7u;                                                                   \
            for (uint i = 0; i < 8u; i++) sb[64u * ib + 8u * ly + i] = (half)yy[i];               \
        }                                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        threadgroup const half* lsma = sa + 4u * 64u * (sgid & 1u);                               \
        threadgroup const half* lsmb = sb + 2u * 64u * (sgid >> 1);                               \
        for (uint ik = 0; ik < 4u; ik++) {                                                        \
            for (uint i = 0; i < 4u; i++) simdgroup_load(ma[i], lsma + 64u * i, 8);               \
            for (uint i = 0; i < 2u; i++) simdgroup_load(mb[i], lsmb + 64u * i, 8);               \
            for (uint i = 0; i < 8u; i++)                                                         \
                simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);              \
            lsma += 8u * 64u;                                                                     \
            lsmb += 4u * 64u;                                                                     \
        }                                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
    }                                                                                             \
                                                                                                  \
    if (rt + 32u <= p.m) {                                                                        \
        device float* C = dst + (ro + 32u * (sgid & 1u)) +                                        \
                          (ulong)(rt + 16u * (sgid >> 1)) * p.out_f;                              \
        for (uint i = 0; i < 8u; i++)                                                             \
            simdgroup_store(mc[i], C + 8u * (i & 3u) + 8u * p.out_f * (i >> 2), p.out_f);         \
    } else {                                                                                      \
        /* partial token tile: stage through threadgroup memory, row-guard the copy-out */        \
        threadgroup float* tc = shraw + 32u * (sgid & 1u) + (16u * (sgid >> 1)) * 64u;            \
        for (uint i = 0; i < 8u; i++)                                                             \
            simdgroup_store(mc[i], tc + 8u * (i & 3u) + 8u * 64u * (i >> 2), 64u);                \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        if (sgid == 0u) {                                                                         \
            for (uint j = lane; j < nr1; j += 32u) {                                              \
                device float* d2 = dst + ro + (ulong)(rt + j) * p.out_f;                          \
                threadgroup const float* c2 = shraw + j * 64u;                                    \
                for (uint i = 0; i < 64u; i++) d2[i] = c2[i];                                     \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
}

CMM_KERNEL(linear_quik4_cmm, DEC16_K4)
CMM_KERNEL(linear_quik6_cmm, DEC16_K6)
CMM_KERNEL(linear_quik8_cmm, DEC16_K8)
CMM_KERNEL(linear_q4k_cmm, DEC16_Q4K)
CMM_KERNEL(linear_q6k_cmm, DEC16_Q6K)

HGEMM_KERNEL(linear_quik4_hmm, DEC16_K4)
HGEMM_KERNEL(linear_quik6_hmm, DEC16_K6)
HGEMM_KERNEL(linear_quik8_hmm, DEC16_K8)
HGEMM_KERNEL(linear_q4k_hmm, DEC16_Q4K)
HGEMM_KERNEL(linear_q6k_hmm, DEC16_Q6K)


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

// ---- Half-fragment flash attention for prefill (f16 KV cache, hd <= 128, hd % 8 == 0): one
// simdgroup per (8-query tile, head). Unlike the earlier f32 simdgroup_matrix attempt (built,
// benched, lost to the scalar split-KV kernel, removed), there is NO staging: K^T and V 8x8
// fragments load DIRECTLY from the f16 cache (strided, transposed for K), and Q is pre-cast once
// per op to f16 — the f32 version spent its time converting K/V through threadgroup memory and
// choked occupancy on the 8 KB tiles. Scores and the output tile accumulate in f32; the online
// softmax runs scalar in f32 on an 8x8 score tile per 8-position KV block, with the row-rescale
// applied as a diagonal f32 MMA. P rounds to f16 (same trade as the half-fragment GEMM).
// Tail KV blocks may read up to 7 rows past the causal limit — always inside the bound cache
// buffer (sized for the full context) — and those columns are masked in the softmax.
// A partial final query tile falls back to the serial per-query path.
kernel void attnflash_f16kv(device const half*  q   [[buffer(0)]],
                            device const half*  k   [[buffer(1)]],
                            device const half*  v   [[buffer(2)]],
                            device float*       dst [[buffer(3)]],
                            constant AttnParams& p  [[buffer(4)]],
                            uint gid  [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    uint ntq = (p.rows + 7u) / 8u;
    if (sg >= ntq * p.n_head) return;
    uint qt = sg / p.n_head;
    uint h = sg % p.n_head;
    uint group = p.n_head / p.n_kv;
    uint kvh = h / group;
    uint hd = p.head_dim;
    uint r0 = qt * 8u;

    if (r0 + 8u > p.rows) {
        // partial query tile: serial per-query fallback (lane-split dot, online softmax)
        for (uint ti = r0; ti < p.rows; ti++) {
            uint qb = (ti * p.n_head + h) * hd;
            uint abs = p.pos + ti;
            uint lo = (p.window > 0u && abs + 1u > p.window) ? (abs + 1u - p.window) : 0u;
            float acc[MAX_DPL];
            for (uint t = 0; t < MAX_DPL; t++) acc[t] = 0.0f;
            float m = -INFINITY, l = 0.0f;
            for (uint j = lo; j <= abs; j++) {
                ulong kb = ((ulong)j * p.n_kv + kvh) * hd;
                float part = 0.0f;
                for (uint d = lane; d < hd; d += 32u) part += (float)q[qb + d] * (float)k[kb + d];
                float sc = simd_sum(part) * p.scale;
                float mnew = max(m, sc);
                float corr = exp(m - mnew);
                float pw = exp(sc - mnew);
                l = l * corr + pw;
                uint t = 0;
                for (uint d = lane; d < hd; d += 32u) {
                    acc[t] = acc[t] * corr + pw * (float)v[kb + d];
                    t++;
                }
                m = mnew;
            }
            uint t = 0;
            for (uint d = lane; d < hd; d += 32u) { dst[qb + d] = acc[t] / l; t++; }
        }
        return;
    }

    threadgroup half tgP[64];
    threadgroup float tgD[64];
    threadgroup float tgM[8], tgL[8];
    uint abs0 = p.pos + r0;                 // row i sees positions <= abs0 + i
    uint abs_max = abs0 + 7u;
    uint lo_min = (p.window > 0u && abs0 + 1u > p.window) ? (abs0 + 1u - p.window) : 0u;
    if (lane < 8u) { tgM[lane] = -INFINITY; tgL[lane] = 0.0f; }

    device const half* qbase = q + ((ulong)r0 * p.n_head + h) * hd;
    ulong qstride = (ulong)p.n_head * hd;
    ulong kvstride = (ulong)p.n_kv * hd;

    simdgroup_float8x8 oa[16];
    uint nfrag = hd / 8u;
    for (uint i = 0; i < nfrag; i++) oa[i] = simdgroup_float8x8(0.0f);

    for (uint j0 = lo_min & ~7u; j0 <= abs_max; j0 += 8u) {
        device const half* kb = k + ((ulong)j0 * p.n_kv + kvh) * hd;
        simdgroup_float8x8 sf = simdgroup_float8x8(0.0f);
        for (uint e0 = 0; e0 < hd; e0 += 8u) {
            simdgroup_half8x8 qa, kt;
            simdgroup_load(qa, qbase + e0, qstride);
            simdgroup_load(kt, kb + e0, kvstride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sf, qa, kt, sf);
        }
        simdgroup_store(sf, tgD, 8);        // reuse tgD as the f32 score scratch
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 8u) {
            uint r = lane;
            uint absr = abs0 + r;
            uint lor = (p.window > 0u && absr + 1u > p.window) ? (absr + 1u - p.window) : 0u;
            float mr = tgM[r];
            float mnew = mr;
            float s[8];
            for (uint c = 0; c < 8u; c++) {
                uint j = j0 + c;
                bool valid = (j >= lor) && (j <= absr);
                s[c] = valid ? tgD[r * 8u + c] * p.scale : -INFINITY;
                mnew = max(mnew, s[c]);
            }
            float corr = (mr == mnew) ? 1.0f : exp(mr - mnew);
            float lsum = 0.0f;
            for (uint c = 0; c < 8u; c++) {
                float pw = (s[c] == -INFINITY) ? 0.0f : exp(s[c] - mnew);
                tgP[r * 8u + c] = (half)pw;
                lsum += pw;
            }
            tgL[r] = tgL[r] * corr + lsum;
            tgM[r] = mnew;
            for (uint c = 0; c < 8u; c++) tgD[r * 8u + c] = (c == r) ? corr : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 df;
        simdgroup_half8x8 pf;
        simdgroup_load(df, tgD, 8);
        simdgroup_load(pf, tgP, 8);
        device const half* vb = v + ((ulong)j0 * p.n_kv + kvh) * hd;
        for (uint i = 0; i < nfrag; i++) {
            simdgroup_float8x8 tmp;
            simdgroup_multiply(tmp, df, oa[i]);
            simdgroup_half8x8 vf;
            simdgroup_load(vf, vb + i * 8u, kvstride);
            simdgroup_multiply_accumulate(oa[i], pf, vf, tmp);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < 8u) {
        for (uint c = 0; c < 8u; c++) tgD[lane * 8u + c] = (c == lane) ? 1.0f / tgL[lane] : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 d2;
    simdgroup_load(d2, tgD, 8);
    ulong obase = ((ulong)r0 * p.n_head + h) * hd;
    ulong ostride = (ulong)p.n_head * hd;
    for (uint i = 0; i < nfrag; i++) {
        simdgroup_float8x8 tmp;
        simdgroup_multiply(tmp, d2, oa[i]);
        simdgroup_store(tmp, dst + obase + i * 8u, ostride);
    }
}

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
