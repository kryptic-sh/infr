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
    /* high nibble stays in place (values 16x) and the scale absorbs the /16 — no per-element  */ \
    /* shift/select, just a mask (the reference dequantize_q4_K trick)                         */ \
    float scale = (hi != 0u ? d * (1.0f / 16.0f) : d) * (float)sc6;                               \
    float mn = -(dmin * (float)m6);                                                               \
    uint nibmask = hi != 0u ? 0xF0F0F0F0u : 0x0F0F0F0Fu;                                          \
    device const uint* qw4 = (device const uint*)(blk + 16u + j * 32u + l0);                      \
    for (uint w = 0; w < 4u; w++) {                                                               \
        uint u = qw4[w] & nibmask;                                                                \
        for (uint k2 = 0; k2 < 4u; k2++) {                                                        \
            wk[w * 4u + k2] = scale * (float)((u >> (8u * k2)) & 0xFFu) + mn;                     \
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
    /* uint32-lane unpack (the reference dequantize_q6_K shape): four 32-bit combines cover the */ \
    /* 16 codes, nibble/crumb selection folded into masks and power-of-two scale variants — all */ \
    /* exact, so the value is bit-identical to the byte-at-a-time form this replaces            */ \
    uint qlo = (off & 32u);                                                                       \
    uint qhs = off >> 4;                                                                          \
    device const ushort* ql16 = (device const ushort*)ql + (qlo != 0u ? 16u : 0u) + 8u * is;      \
    device const ushort* qh16 = (device const ushort*)qh + 8u * is;                               \
    uint kmask1 = (off >= 64u) ? ((qhs > 4u) ? 0xC0C0C0C0u : 0x30303030u)                         \
                               : ((qhs > 0u) ? 0x0C0C0C0Cu : 0x03030303u);                        \
    uint kmask2 = (off >= 64u) ? 0xF0F0F0F0u : 0x0F0F0F0Fu;                                       \
    float dl0 = scale;                                                                            \
    float dl1 = dl0 * (1.0f / 256.0f);                                                            \
    float dl2 = dl1 * (1.0f / 256.0f);                                                            \
    float dl3 = dl2 * (1.0f / 256.0f);                                                            \
    uint shr_h = (qhs > 4u) ? 2u : 0u;                                                            \
    uint shl_h = (off >= 64u) ? 0u : ((qhs > 0u) ? 2u : 4u);                                      \
    uint shr_l = (off >= 64u) ? 4u : 0u;                                                          \
    for (uint i = 0; i < 4u; i++) {                                                               \
        uint low  = ((uint)ql16[2u * i] | ((uint)ql16[2u * i + 1u] << 16)) & kmask2;              \
        uint high = ((uint)qh16[2u * i] | ((uint)qh16[2u * i + 1u] << 16)) & kmask1;              \
        uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);                                     \
        wk[4u * i]      = dl0 * (float)(q & 0xFFu)       + mn;                                    \
        wk[4u * i + 1u] = dl1 * (float)(q & 0xFF00u)     + mn;                                    \
        wk[4u * i + 2u] = dl2 * (float)(q & 0xFF0000u)   + mn;                                    \
        wk[4u * i + 3u] = dl3 * (float)(q & 0xFF000000u) + mn;                                    \
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


// ---- Strided row copy (the fused-QKV prefill split: one wide GEMM output sliced into q/k/v).
// Pure data movement, but on-device — a host copy here would round-trip the [m, qkv] buffer and
// break the command buffer mid-forward.
struct CopyStridedParams {
    uint src_off;
    uint src_stride;
    uint dst_off;
    uint dst_stride;
    uint rows;
    uint n;
};
kernel void copy_strided_f32(device const float*         src [[buffer(0)]],
                             device float*               dst [[buffer(1)]],
                             constant CopyStridedParams& p   [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.n) return;
    uint r = gid / p.n, c = gid % p.n;
    dst[p.dst_off + r * p.dst_stride + c] = src[p.src_off + r * p.src_stride + c];
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
// Body shared by the plain GEMV and the fused-residual variant (`FADD`: dst = W·x + res, the
// decode o_proj/down_proj + Add peephole — one dispatch and no sublayer-output round-trip).
// EPI epilogue modes: 0 = dst = s; 1 = dst = s + res (fused residual Add); 2 = MoE accumulate,
// dst = (zeroacc ? 0 : dst) + wgt*s (the weighted expert sum, first expert zeroes).
template<int EPI, typename PT>
inline void linear_q4k_body(device const float*  x,
                            device const uchar*  codes,
                            device float*        dst,
                            device const float*  res,
                            constant PT& p,
                            float wgt, bool zeroacc,
                            uint gid, uint lane) {
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
        if (lane == 0u) {
            if (EPI == 2)      dst[first_row + row] = (zeroacc ? 0.0f : dst[first_row + row]) + wgt * s;
            else if (EPI == 1) dst[first_row + row] = s + res[first_row + row];
            else               dst[first_row + row] = s;
        }
    }
}

kernel void linear_q4k(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    linear_q4k_body<0>(x, codes, dst, x, p, 0.0f, false, gid, lane);
}
kernel void linear_q4k_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    linear_q4k_body<1>(x, codes, dst, res, p, 0.0f, false, gid, lane);
}

template<int EPI, typename PT>
inline void linear_q6k_body(device const float*  x,
                            device const uchar*  codes,
                            device float*        dst,
                            device const float*  res,
                            constant PT& p,
                            float wgt, bool zeroacc,
                            uint gid, uint lane) {
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
        if (lane == 0u) {
            if (EPI == 2)      dst[first_row + row] = (zeroacc ? 0.0f : dst[first_row + row]) + wgt * s;
            else if (EPI == 1) dst[first_row + row] = s + res[first_row + row];
            else               dst[first_row + row] = s;
        }
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
    linear_q6k_body<0>(x, codes, dst, x, p, 0.0f, false, gid, lane);
}
kernel void linear_q6k_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    linear_q6k_body<1>(x, codes, dst, res, p, 0.0f, false, gid, lane);
}

// ---- MoE expert GEMVs: the shared GEMV bodies, batched over the SELECTED experts — one
// dispatch covers all n_used experts (slot = high grid bits), each picking its weight slice
// from the device expert table (`moe_topk` below), so a whole MoE FFN is 7 dispatches with no
// host round-trip. Gate/up read the shared x and write per-slot rows of the [n_used, out_f]
// scratch; the down variant (EPI 2) reads its slot's activation row and writes w[slot]*y to its
// slot's output row — `moe_reduce` then folds the weighted expert sum (slot-ascending, the CPU
// reference's accumulation order).
struct MoeLinParams { uint m; uint in_f; uint out_f; uint dshift; uint n_used; uint row0; };
#define MOE_WRAP(NAME, BODY, EPI, ROWB)                                                           \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 device const uint*   tbl   [[buffer(5)]],                                       \
                 constant MoeLinParams& p   [[buffer(6)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]]) {                                      \
    uint sg = gid / 32u;                                                                          \
    uint per_out = p.out_f >> 1;         /* simdgroups per expert (2 weight rows each) */        \
    uint row = sg / (p.n_used * per_out);          /* token row within this chunk */             \
    uint rem = sg % (p.n_used * per_out);                                                         \
    uint slot = rem / per_out;                                                                    \
    uint e = tbl[(p.row0 + row) * 32u + slot];                                                    \
    float w = as_type<float>(tbl[(p.row0 + row) * 32u + 16u + slot]);                             \
    ulong row_b = (ulong)(p.in_f >> 8) * ROWB;                                                    \
    device const uchar* ec = codes + (ulong)e * p.out_f * row_b;                                  \
    /* gate/up read the token's row of x [rows, ne]; down reads its (row, slot) activation */     \
    device const float* xs = (EPI == 2) ? x + (ulong)(row * p.n_used + slot) * p.in_f             \
                                        : x + (ulong)(p.row0 + row) * p.in_f;                     \
    device float* ds = dst + (ulong)(row * p.n_used + slot) * p.out_f;                            \
    uint g2 = (rem % per_out) * 32u + lane;  /* body sees a per-(row, expert) grid */             \
    BODY<EPI>(xs, ec, ds, xs, p, w, true, g2, lane);                                              \
}
MOE_WRAP(linear_q4k_moe,     linear_q4k_body, 0, 144ul)
MOE_WRAP(linear_q4k_moe_acc, linear_q4k_body, 2, 144ul)
MOE_WRAP(linear_q6k_moe,     linear_q6k_body, 0, 210ul)
MOE_WRAP(linear_q6k_moe_acc, linear_q6k_body, 2, 210ul)
#undef MOE_WRAP

// ---- Expert-grouped GEMM for batched MoE prefill (the llama.cpp mul_mm_id shape). Rather than
// one GEMV per (token, slot) — no MMA, compute-bound at prefill widths — `moe_map` groups the
// chunk's (token, slot) pairs by EXPERT (one thread per expert scans the routing table, so each
// expert's list is token-ascending and deterministic), and the `*_cmm_id` kernels run the shared
// cooperative-GEMM tile over each expert's token group: grid = expert x token-tile x out-tile,
// tiles past an expert's count return before any barrier, activations stage through the same
// pre-transposed 8x8 layout with the token row resolved through the id list, and the output
// SCATTERS through threadgroup staging to each token's (row, slot) scratch row. The down variant
// folds the routing weight during the scatter. Ids are chunk-relative (`row0` rebases).
struct MoeMapParams { uint n_expert; uint n_used; uint rows; uint row0; uint cap; };
kernel void moe_map(device const uint* tbl [[buffer(0)]],
                    device uint*       ids [[buffer(1)]],
                    device uint*       tpe [[buffer(2)]],
                    constant MoeMapParams& p [[buffer(3)]],
                    uint tid [[thread_position_in_grid]]) {
    if (tid >= p.n_expert) return;
    uint n = 0;
    for (uint r = 0; r < p.rows; r++) {
        device const uint* rt = tbl + (ulong)(p.row0 + r) * 32u;
        for (uint u = 0; u < p.n_used; u++) {
            if (rt[u] == tid) {
                ids[tid * p.cap + n] = r * p.n_used + u;
                n++;
            }
        }
    }
    tpe[tid] = n;
}

struct MoeCmmParams { uint in_f; uint out_f; uint n_used; uint cap; uint row0; uint ntt; };
// DIVROW: gate/up read x[row0 + t/n_used]; down (DIVROW=0) reads its (row, slot) activation row.
// WSCALE: fold the routing weight tbl[(row0 + t/n_used)*32 + 16 + t%n_used] during the scatter.
#define MOE_CMM_KERNEL(NAME, DEC, DIVROW, WSCALE)                                                 \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 device const uint*   ids   [[buffer(5)]],                                       \
                 device const uint*   tpe   [[buffer(6)]],                                       \
                 device const uint*   tbl   [[buffer(7)]],                                       \
                 constant MoeCmmParams& p   [[buffer(8)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]],                                        \
                 uint sgid [[simdgroup_index_in_threadgroup]]) {                                 \
    uint tgix = gid / 128u;                                                                       \
    uint nto = p.out_f / 64u;                                                                     \
    uint e = tgix / (p.ntt * nto);                                                                \
    uint rem = tgix % (p.ntt * nto);                                                              \
    uint tt0 = (rem / nto) * 32u;                                                                 \
    uint ro = (rem % nto) * 64u;                                                                  \
    uint cnt = tpe[e];                                                                            \
    if (tt0 >= cnt) return;   /* uniform per threadgroup, before any barrier */                   \
    uint nr1 = min(32u, cnt - tt0);                                                               \
    uint tid = sgid * 32u + lane;                                                                 \
                                                                                                  \
    threadgroup float shraw[2048];                                                                \
    threadgroup half* sa = (threadgroup half*)shraw;                                              \
    threadgroup half* sb = ((threadgroup half*)shraw) + 2048u;                                    \
                                                                                                  \
    uint lr0 = tid >> 1;                                                                          \
    uint il0 = tid & 1u;                                                                          \
    uint lr1 = tid >> 2;                                                                          \
    uint iyk = (tid & 3u) * 8u;                                                                   \
    uint lr1c = min(lr1, nr1 - 1u);                                                               \
    uint tident = ids[e * p.cap + tt0 + lr1c];                                                    \
    uint xrow = DIVROW ? (p.row0 + tident / p.n_used) : tident;                                   \
                                                                                                  \
    simdgroup_half8x8 ma[4];                                                                      \
    simdgroup_half8x8 mb[2];                                                                      \
    simdgroup_float8x8 mc[8];                                                                     \
    for (uint i = 0; i < 8u; i++) mc[i] = simdgroup_float8x8(0.0f);                               \
                                                                                                  \
    uint nb = p.in_f >> 4;                                                                        \
    ulong ebase = (ulong)e * p.out_f * nb;   /* expert's first block index */                     \
    for (uint k0 = 0; k0 < p.in_f; k0 += 32u) {                                                   \
        ulong bi = ebase + (ulong)(ro + lr0) * nb + (ulong)(k0 >> 4) + il0;                       \
        float wk[16];                                                                             \
        DEC(wk)                                                                                   \
        device const float4* yy =                                                                 \
            (device const float4*)(x + (ulong)xrow * p.in_f + k0 + iyk);                          \
        float4 yv0 = yy[0];                                                                       \
        float4 yv1 = yy[1];                                                                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        {                                                                                         \
            uint sy = lr0 >> 3;                                                                   \
            uint lx = lr0 & 7u;                                                                   \
            for (uint i = 0; i < 16u; i++) {                                                      \
                uint sx = 2u * il0 + (i >> 3);                                                    \
                sa[64u * (8u * sx + sy) + 8u * (i & 7u) + lx] = (half)wk[i];                      \
            }                                                                                     \
        }                                                                                         \
        {                                                                                         \
            uint ib = 4u * (tid & 3u) + (lr1 >> 3);                                               \
            uint ly = lr1 & 7u;                                                                   \
            threadgroup half4* sb4 = (threadgroup half4*)(sb + 64u * ib + 8u * ly);               \
            sb4[0] = half4(yv0);                                                                  \
            sb4[1] = half4(yv1);                                                                  \
        }                                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        threadgroup const half* lsma = sa + 4u * 64u * (sgid & 1u);                               \
        threadgroup const half* lsmb = sb + 2u * 64u * (sgid >> 1);                               \
        for (uint ik = 0; ik < 4u; ik++) {                                                        \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            for (uint i = 0; i < 4u; i++) simdgroup_load(ma[i], lsma + 64u * i, 8);               \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            for (uint i = 0; i < 2u; i++) simdgroup_load(mb[i], lsmb + 64u * i, 8);               \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            for (uint i = 0; i < 8u; i++)                                                         \
                simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);              \
            lsma += 8u * 64u;                                                                     \
            lsmb += 4u * 64u;                                                                     \
        }                                                                                         \
    }                                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
                                                                                                  \
    /* scatter through threadgroup staging: token rows are non-contiguous scratch rows */         \
    threadgroup float* tc = shraw + 32u * (sgid & 1u) + (16u * (sgid >> 1)) * 64u;                \
    for (uint i = 0; i < 8u; i++)                                                                 \
        simdgroup_store(mc[i], tc + 8u * (i & 3u) + 8u * 64u * (i >> 2), 64u);                    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
    if (sgid == 0u) {                                                                             \
        for (uint j = lane; j < nr1; j += 32u) {                                                  \
            uint t = ids[e * p.cap + tt0 + j];                                                    \
            float w = WSCALE                                                                      \
                ? as_type<float>(tbl[(ulong)(p.row0 + t / p.n_used) * 32u + 16u + t % p.n_used])  \
                : 1.0f;                                                                           \
            device float* d2 = dst + (ulong)t * p.out_f + ro;                                     \
            threadgroup const float* c2 = shraw + j * 64u;                                        \
            for (uint i = 0; i < 64u; i++) d2[i] = c2[i] * w;                                     \
        }                                                                                         \
    }                                                                                             \
}
MOE_CMM_KERNEL(linear_q4k_cmm_id,   DEC16_Q4K, 1, 0)
MOE_CMM_KERNEL(linear_q4k_cmm_id_w, DEC16_Q4K, 0, 1)
MOE_CMM_KERNEL(linear_q6k_cmm_id,   DEC16_Q6K, 1, 0)
MOE_CMM_KERNEL(linear_q6k_cmm_id_w, DEC16_Q6K, 0, 1)
#undef MOE_CMM_KERNEL

// Weighted expert sum: dst[i] = sum_u y[u*ne + i] (weights already folded into y by the down
// GEMV's EPI-2 epilogue; slot-ascending order matches the CPU reference).
struct MoeReduceParams { uint ne; uint n_used; uint rows; uint row0; };
kernel void moe_reduce(device const float* y   [[buffer(0)]],
                       device float*       dst [[buffer(1)]],
                       constant MoeReduceParams& p [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.ne) return;
    uint row = gid / p.ne;
    uint i = gid % p.ne;
    float s = 0.0f;
    for (uint u = 0; u < p.n_used; u++) s += y[(row * p.n_used + u) * p.ne + i];
    dst[(p.row0 + row) * p.ne + i] = s;
}

struct MoeTopkParams { uint n_expert; uint n_used; float scale; };
kernel void moe_topk(device const float* logits_all [[buffer(0)]],
                     device uint*        tbl_all    [[buffer(1)]],
                     constant MoeTopkParams& p  [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    // one simdgroup (32 threads) per token row; each lane owns experts e = lane + 32j
    uint row = gid / 32u;
    uint lane = gid % 32u;
    device const float* logits = logits_all + (ulong)row * p.n_expert;
    device uint* tbl = tbl_all + row * 32u;
    float lmax = -MAXFLOAT;
    for (uint e = lane; e < p.n_expert; e += 32u) lmax = max(lmax, logits[e]);
    float maxl = simd_max(lmax);
    // psum in the reference's ascending order (exact bit-match), broadcast from lane 0
    float psum = 0.0f;
    if (lane == 0u) {
        for (uint e = 0; e < p.n_expert; e++) psum += exp(logits[e] - maxl);
    }
    psum = simd_broadcast_first(psum);
    // top-k selection, lane-parallel: each round every lane offers its best untaken expert
    // (ascending scan + strict > == lowest index per lane), simd_max picks the winning logit
    // and simd_min the lowest tied index — exactly the reference's stable-sort order
    uint taken = 0u;   // bitmask over this lane's stride slots j
    uint sel[16];
    for (uint u = 0; u < p.n_used; u++) {
        float bv = -MAXFLOAT;
        uint be = 0xFFFFFFFFu;
        uint j = 0u;
        for (uint e = lane; e < p.n_expert; e += 32u, j++) {
            if ((taken & (1u << j)) == 0u && logits[e] > bv) { bv = logits[e]; be = e; }
        }
        float m = simd_max(bv);
        uint pick = simd_min(bv == m ? be : 0xFFFFFFFFu);
        sel[u] = pick;
        if ((pick & 31u) == lane) taken |= 1u << (pick >> 5);
    }
    if (lane == 0u) {
        float wsum = 0.0f;
        float ws[16];
        for (uint u = 0; u < p.n_used; u++) {
            ws[u] = exp(logits[sel[u]] - maxl) / psum;
            wsum += ws[u];
        }
        wsum = max(wsum, 1e-20f);
        for (uint u = 0; u < p.n_used; u++) {
            tbl[u] = sel[u];
            tbl[16u + u] = as_type<uint>(ws[u] / wsum * p.scale);
        }
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
        /* device reads + dequant FIRST, into registers — issued while the previous       */     \
        /* iteration's MMA phase (other simdgroups) is still draining; the barrier below  */     \
        /* orders only the threadgroup-memory stores (mul_mm does exactly this)           */     \
        ulong bi = (ulong)(ro + lr0) * nb + (ulong)(k0 >> 4) + il0;                               \
        float wk[16];                                                                             \
        DEC(wk)                                                                                   \
        device const float4* yy =                                                                 \
            (device const float4*)(x + (ulong)(rt + lr1c) * p.in_f + k0 + iyk);                   \
        float4 yv0 = yy[0];                                                                       \
        float4 yv1 = yy[1];                                                                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        {   /* stage A: one 16-block per thread, into pre-transposed 8x8 tiles */                 \
            uint sy = lr0 >> 3;                                                                   \
            uint lx = lr0 & 7u;                                                                   \
            for (uint i = 0; i < 16u; i++) {                                                      \
                uint sx = 2u * il0 + (i >> 3);                                                    \
                sa[64u * (8u * sx + sy) + 8u * (i & 7u) + lx] = (half)wk[i];                      \
            }                                                                                     \
        }                                                                                         \
        {   /* stage B: 8 activations per thread, two vectorized f32->f16 half4 stores */         \
            uint ib = 4u * (tid & 3u) + (lr1 >> 3);                                               \
            uint ly = lr1 & 7u;                                                                   \
            threadgroup half4* sb4 = (threadgroup half4*)(sb + 64u * ib + 8u * ly);               \
            sb4[0] = half4(yv0);                                                                  \
            sb4[1] = half4(yv1);                                                                  \
        }                                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        threadgroup const half* lsma = sa + 4u * 64u * (sgid & 1u);                               \
        threadgroup const half* lsmb = sb + 2u * 64u * (sgid >> 1);                               \
        for (uint ik = 0; ik < 4u; ik++) {                                                        \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            for (uint i = 0; i < 4u; i++) simdgroup_load(ma[i], lsma + 64u * i, 8);               \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            for (uint i = 0; i < 2u; i++) simdgroup_load(mb[i], lsmb + 64u * i, 8);               \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            for (uint i = 0; i < 8u; i++)                                                         \
                simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);              \
            lsma += 8u * 64u;                                                                     \
            lsmb += 4u * 64u;                                                                     \
        }                                                                                         \
    }                                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
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


// ---- RoPE (Op::Rope = the no-qk-norm llama-family rotation): INTERLEAVED pairs (2p, 2p+1) —
// llama.cpp's ROPE_TYPE_NORM, matching infr-cpu and the Vulkan `rope` kernel. (QkNormRope below
// is the NEOX split-half used by qwen/gemma; the styles are NOT interchangeable.) Rotates the
// first rope_dim of each head; dims beyond pass through. One thread per (row, head).
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
        uint i0 = 2 * pp, i1 = 2 * pp + 1;
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
    /* same-head query tiles are ADJACENT simdgroups: concurrent tiles then stream the SAME
       head's KV region and hit the SLC, instead of 16 heads' regions at once (measured: the
       head-fastest order collapsed pp8k to ~1/3 of llama.cpp) */
    uint qt = sg % ntq;
    uint h = sg / ntq;
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

    threadgroup half tgP[128];
    threadgroup float tgS16[128];
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

    for (uint j0 = lo_min & ~15u; j0 <= abs_max; j0 += 16u) {
        /* two 8-position score fragments per iteration: one scalar softmax phase and one
           rescale per 16 KV positions instead of per 8 — the scalar phase and its barriers,
           not KV bandwidth, are what this kernel waits on */
        device const half* kb = k + ((ulong)j0 * p.n_kv + kvh) * hd;
        simdgroup_float8x8 sf0 = simdgroup_float8x8(0.0f);
        simdgroup_float8x8 sf1 = simdgroup_float8x8(0.0f);
        for (uint e0 = 0; e0 < hd; e0 += 8u) {
            simdgroup_half8x8 qa, kt;
            simdgroup_load(qa, qbase + e0, qstride);
            simdgroup_load(kt, kb + e0, kvstride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sf0, qa, kt, sf0);
            simdgroup_load(kt, kb + 8u * kvstride + e0, kvstride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sf1, qa, kt, sf1);
        }
        simdgroup_store(sf0, tgS16, 16);      // f32 score scratch, 8 rows x 16 cols
        simdgroup_store(sf1, tgS16 + 8u, 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 8u) {
            uint r = lane;
            uint absr = abs0 + r;
            uint lor = (p.window > 0u && absr + 1u > p.window) ? (absr + 1u - p.window) : 0u;
            float mr = tgM[r];
            float mnew = mr;
            float s[16];
            for (uint c = 0; c < 16u; c++) {
                uint j = j0 + c;
                bool valid = (j >= lor) && (j <= absr);
                s[c] = valid ? tgS16[r * 16u + c] * p.scale : -INFINITY;
                mnew = max(mnew, s[c]);
            }
            float corr = (mr == mnew) ? 1.0f : exp(mr - mnew);
            float lsum = 0.0f;
            for (uint c = 0; c < 16u; c++) {
                float pw = (s[c] == -INFINITY) ? 0.0f : exp(s[c] - mnew);
                tgP[r * 16u + c] = (half)pw;
                lsum += pw;
            }
            tgL[r] = tgL[r] * corr + lsum;
            tgM[r] = mnew;
            for (uint c = 0; c < 8u; c++) tgD[r * 8u + c] = (c == r) ? corr : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 df;
        simdgroup_half8x8 pf0, pf1;
        simdgroup_load(df, tgD, 8);
        simdgroup_load(pf0, tgP, 16);
        simdgroup_load(pf1, tgP + 8u, 16);
        device const half* vb = v + ((ulong)j0 * p.n_kv + kvh) * hd;
        for (uint i = 0; i < nfrag; i++) {
            simdgroup_float8x8 tmp;
            simdgroup_multiply(tmp, df, oa[i]);
            simdgroup_half8x8 vf;
            simdgroup_load(vf, vb + i * 8u, kvstride);
            simdgroup_multiply_accumulate(tmp, pf0, vf, tmp);
            simdgroup_load(vf, vb + 8u * kvstride + i * 8u, kvstride);
            simdgroup_multiply_accumulate(oa[i], pf1, vf, tmp);
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

// ---- Cooperative flash attention for prefill (f16 KV cache, hd 64 or 128 instantiations): the
// llama.cpp `kernel_flash_attn_ext` structure — NSG=4 simdgroups cooperate on ONE (8-query tile,
// head) threadgroup, processing C=64 KV positions per iteration. The phases split the work along
// different axes so every lane stays busy (the single-simdgroup attnflash_f16kv above stalls in
// its scalar softmax, 8 of 32 lanes active, one phase per 16 KV):
//   QK^T    — the 8 score fragments (64 KV cols x 8 queries) split across simdgroups, 2 each;
//             K^T fragments load DIRECTLY from the f16 cache (transposed, no staging).
//   softmax — split by query ROWS (2 rows per simdgroup); each row's 64 scores are one float2
//             per lane, so all 32 lanes work; the online max/sum (M/S) stats live in that
//             simdgroup's registers for the whole KV loop — no cross-simdgroup stat merges.
//   P*V     — split by output COLUMNS (hd/32 8x8 O fragments per simdgroup) held in registers
//             across the MMA, staged through threadgroup `so` only for the softmax rescale.
// Masking is analytic (causal + window per row) — no mask buffer, no -inf staging; masked lanes
// force pw=0, and M is floored at -MAXFLOAT/2 so an all-masked block leaves S/O untouched.
// A partial final query tile zero-pads Q rows in shared memory and skips their output store
// (the fallback serial path in attnflash_f16kv is not needed here). Score/O accumulation is f32;
// P rounds through f32 shared and enters the V MMA as an f32 fragment against half V fragments.
// Tail KV blocks read up to 7 rows past the causal limit (same in-buffer contract as above);
// 8-row blocks entirely past it are skipped, so reads never go further.
template<uint hd, uint NSG>   // compile-time head_dim + simdgroup count: fully unrolled, exact shared sizing
kernel void attnflash2_f16kv_t(device const half*  q   [[buffer(0)]],
                               device const half*  k   [[buffer(1)]],
                               device const half*  v   [[buffer(2)]],
                               device float*       dst [[buffer(3)]],
                               constant AttnParams& p  [[buffer(4)]],
                               uint3  tgpig [[threadgroup_position_in_grid]],
                               ushort sgitg [[simdgroup_index_in_threadgroup]],
                               ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint QT = 8, C = 64, NQ = QT / NSG, SH = C;
    threadgroup half  sq[QT * hd];    // Q tile (rows x hd, half)
    threadgroup float so[QT * hd];    // O accumulator (rows x hd, f32)
    threadgroup float ss[QT * SH];    // scores, then P, per KV block (rows x C, f32)

    uint ntq = (p.rows + QT - 1u) / QT;
    uint qt = tgpig.x % ntq;          // same-head tiles adjacent (SLC — see attnflash_f16kv)
    uint h  = tgpig.x / ntq;
    constexpr uint hd4 = hd / 4u;
    constexpr uint no = hd / (8u * NSG);   // O column fragments owned per simdgroup
    constexpr uint NC = (C / 8u) / NSG;    // score fragments owned per simdgroup
    uint kvh = h / (p.n_head / p.n_kv);
    uint r0 = qt * QT;
    uint abs0 = p.pos + r0;
    uint abs_max = p.pos + min(p.rows - 1u, r0 + QT - 1u);
    uint lo_min = (p.window > 0u && abs0 + 1u > p.window) ? (abs0 + 1u - p.window) : 0u;
    ulong qstride = (ulong)p.n_head * hd;
    ulong kvstride = (ulong)p.n_kv * hd;

    // stage Q rows to shared (zero rows past p.rows), zero the O accumulator
    for (uint jj = 0; jj < NQ; jj++) {
        uint j = jj * NSG + sgitg;
        bool live = r0 + j < p.rows;
        // clamp the dead-row pointer (a select may still speculate the load)
        device const half4* q4 =
            (device const half4*)(q + (ulong)min(r0 + j, p.rows - 1u) * qstride + (ulong)h * hd);
        threadgroup half4*  sq4 = (threadgroup half4*)sq + j * hd4;
        threadgroup float4* so4 = (threadgroup float4*)so + j * hd4;
        for (uint i = tiisg; i < hd4; i += 32u) {
            sq4[i] = live ? q4[i] : half4(0.0h);
            so4[i] = float4(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    for (uint jj = 0; jj < NQ; jj++) { S[jj] = 0.0f; M[jj] = -MAXFLOAT / 2; }

    for (uint ic = lo_min & ~(C - 1u); ic <= abs_max; ic += C) {
        // Q*K^T — 8 score fragments split across simdgroups (fragment f covers KV rows
        // ic+8f, columns interleaved so each simdgroup's two fragments are NSG apart)
        {
            device const half* pk = k + ((ulong)(ic + 8u * sgitg) * p.n_kv + kvh) * hd;
            threadgroup float* ps = ss + 8u * sgitg;
            for (uint cc = 0; cc < NC; cc++) {
                simdgroup_float8x8 mqk = simdgroup_float8x8(0.0f);
                if (ic + 8u * (sgitg + cc * NSG) <= abs_max) {
                    for (uint i = 0; i < hd; i += 16u) {
                        simdgroup_half8x8 mq, mk;
                        simdgroup_load(mq, sq + i, hd);
                        simdgroup_load(mk, pk + i, kvstride, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        simdgroup_load(mq, sq + i + 8u, hd);
                        simdgroup_load(mk, pk + i + 8u, kvstride, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                    }
                }
                simdgroup_store(mqk, ps, SH);
                pk += (ulong)(8u * NSG) * kvstride;
                ps += 8u * NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax — rows split across simdgroups, 2 scores (one float2) per lane
        for (uint jj = 0; jj < NQ; jj++) {
            uint j = jj * NSG + sgitg;
            uint absr = abs0 + j;   // rows past p.rows compute junk, never stored
            uint lor = (p.window > 0u && absr + 1u > p.window) ? (absr + 1u - p.window) : 0u;
            threadgroup float2* ss2 = (threadgroup float2*)(ss + j * SH);
            float2 s2 = ss2[tiisg] * p.scale;
            uint c0 = ic + 2u * tiisg;
            bool v0 = (c0 >= lor) && (c0 <= absr);
            bool v1 = (c0 + 1u >= lor) && (c0 + 1u <= absr);
            float m = M[jj];
            float mnew = simd_max(max(m, max(v0 ? s2.x : -MAXFLOAT / 2, v1 ? s2.y : -MAXFLOAT / 2)));
            float ms = exp(m - mnew);
            float pw0 = v0 ? exp(s2.x - mnew) : 0.0f;
            float pw1 = v1 ? exp(s2.y - mnew) : 0.0f;
            S[jj] = S[jj] * ms + simd_sum(pw0 + pw1);
            M[jj] = mnew;
            ss2[tiisg] = float2(pw0, pw1);
            threadgroup float4* so4 = (threadgroup float4*)so + j * hd4;
            for (uint i = tiisg; i < hd4; i += 32u) so4[i] *= ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O += P*V — O column fragments split across simdgroups, held in registers; V fragments
        // load directly from the f16 cache; fully-causal-masked 8-row KV blocks skipped (their
        // P is all zero, and skipping keeps reads within 7 rows of the limit)
        {
            simdgroup_float8x8 lo[4];
            threadgroup float* sot = so + 8u * sgitg;
            for (uint ii = 0; ii < no; ii++) simdgroup_load(lo[ii], sot + 8u * NSG * ii, hd);
            device const half* pv = v + ((ulong)ic * p.n_kv + kvh) * hd + 8u * sgitg;
            // only KV blocks up to the causal limit (P is zero past it, and skipping keeps
            // reads within 7 rows); paired blocks keep 2 P and 2*no V loads in flight
            uint nblk = min(C / 8u, (abs_max - ic) / 8u + 1u);
            for (uint cc = 0; cc + 1u < nblk; cc += 2u) {
                simdgroup_float8x8 vs[2];
                simdgroup_load(vs[0], ss + 8u * cc, SH);
                simdgroup_load(vs[1], ss + 8u * cc + 8u, SH);
                for (uint ii = 0; ii < no; ii++) {
                    simdgroup_half8x8 mv[2];
                    simdgroup_load(mv[0], pv + 8u * NSG * ii, kvstride);
                    simdgroup_load(mv[1], pv + 8u * NSG * ii + 8u * kvstride, kvstride);
                    simdgroup_multiply_accumulate(lo[ii], vs[0], mv[0], lo[ii]);
                    simdgroup_multiply_accumulate(lo[ii], vs[1], mv[1], lo[ii]);
                }
                pv += 16u * kvstride;
            }
            if (nblk & 1u) {
                simdgroup_float8x8 vs;
                simdgroup_load(vs, ss + 8u * (nblk - 1u), SH);
                for (uint ii = 0; ii < no; ii++) {
                    simdgroup_half8x8 mv;
                    simdgroup_load(mv, pv + 8u * NSG * ii, kvstride);
                    simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
                }
            }
            for (uint ii = 0; ii < no; ii++) simdgroup_store(lo[ii], sot + 8u * NSG * ii, hd);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O / S — same row split as the softmax (that simdgroup holds the row's S)
    for (uint jj = 0; jj < NQ; jj++) {
        uint j = jj * NSG + sgitg;
        if (r0 + j >= p.rows) continue;
        float sc = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        device float4* out = (device float4*)(dst + ((ulong)(r0 + j)) * qstride + (ulong)h * hd);
        threadgroup const float4* so4 = (threadgroup const float4*)so + j * hd4;
        for (uint i = tiisg; i < hd4; i += 32u) out[i] = so4[i] * sc;
    }
}

typedef decltype(attnflash2_f16kv_t<64, 4>) attnflash2_t;
template [[host_name("attnflash2_f16kv_hd64")]]  kernel attnflash2_t attnflash2_f16kv_t<64, 4>;
template [[host_name("attnflash2_f16kv_hd128")]] kernel attnflash2_t attnflash2_f16kv_t<128, 4>;

// ---- Vector flash attention for decode (f16 KV cache, hd 64 or 128, one query row per
// threadgroup): the llama.cpp `kernel_flash_attn_ext_vec` structure. NSG simdgroups each own
// interleaved C=32-position KV blocks with a PRIVATE online softmax, merged once at the end by a
// log2 tree — same split-KV idea as attnsplit32 above, but each simdgroup step handles 32
// positions instead of 1: lanes fold as (ty, tx) = 4 KV rows x 8-lane dots, a shuffle tree
// reduces the 8 partials per row, and ONE simd_max/simd_sum softmax pass covers the whole block.
// That cuts the serial chain per simdgroup from kv_len/NSG simd reductions to kv_len/(NSG*32)
// block passes — the attnsplit kernels are latency-bound on exactly that chain at long context.
// Q stays f32 in shared (no rounding: f32 dots over exactly-widened f16 K/V, same numeric class
// as attnsplit32, only reassociated). Tail positions clamp their row pointer to kv_len-1 and are
// masked in the softmax, so reads never leave the cache. O accumulates in shared per simdgroup
// (ty==0 lanes own hd/32 float4 columns each after the fold).
// The body is a plain inline function so the static kernel (baked pos/kv_len from AttnParams)
// and the DYNAMIC-POS kernel (pos read from the bound positions buffer — the decode-replay
// contract, where one recorded dispatch is replayed every token) share it exactly.
template<uint hd, uint NSG>
inline void attnvec_body(device const float* q,
                         device const half*  k,
                         device const half*  v,
                         device float*       dst,
                         constant AttnParams& p,
                         uint abs, uint kvl,
                         threadgroup float* sq,
                         threadgroup float* ssc,
                         threadgroup float* so,
                         uint3  tgpig,
                         ushort sgitg,
                         ushort tiisg) {
    constexpr uint C = 32, NE = 4, NL = 32u / NE;  // 4 KV rows x 8-lane dots per simdgroup pass
    constexpr uint hd4 = hd / 4u;
    constexpr uint NI = hd4 / NL;                  // float4s per lane per row (2 or 4)

    uint tg = tgpig.x;
    uint ti = tg / p.n_head;
    uint h  = tg % p.n_head;
    uint kvh = h / (p.n_head / p.n_kv);
    uint lo = (p.window > 0u && abs + 1u > p.window) ? (abs + 1u - p.window) : 0u;
    uint tx = tiisg % NL, ty = tiisg / NL;

    {
        device const float4* q4 = (device const float4*)(q + ((ulong)ti * p.n_head + h) * hd);
        threadgroup float4* sq4 = (threadgroup float4*)sq;
        for (uint i = sgitg * 32u + tiisg; i < hd4; i += NSG * 32u) sq4[i] = q4[i];
    }
    threadgroup float* ss = ssc + sgitg * C;
    threadgroup float4* so4 = (threadgroup float4*)so + sgitg * hd4;
    if (ty == 0) {
        for (uint ii = 0; ii < NI; ii++) so4[ii * NL + tx] = float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f, M = -MAXFLOAT / 2;
    device const half4* k4 = (device const half4*)k;
    device const half4* v4 = (device const half4*)v;
    threadgroup const float4* sq4 = (threadgroup const float4*)sq;

    for (uint ic = sgitg * C; ic <= abs; ic += NSG * C) {
        if (ic + C <= lo) continue;   // whole block below the window (uniform per simdgroup)

        // Q*K^T — each (ty, tx) fold: row ic + NE*cc + ty, 8-lane split of the hd dot
        {
            float mqk[C / NE];
            for (uint cc = 0; cc < C / NE; cc++) {
                // clamp tail rows into the cache; their scores are masked below
                uint rc = min(ic + NE * cc + ty, kvl - 1u);
                device const half4* pk = k4 + ((ulong)rc * p.n_kv + kvh) * hd4;
                float acc = 0.0f;
                for (uint ii = 0; ii < NI; ii++)
                    acc += dot(float4(pk[ii * NL + tx]), sq4[ii * NL + tx]);
                // fold the 8 tx-lane partials of each row
                acc += simd_shuffle_down(acc, 4);
                acc += simd_shuffle_down(acc, 2);
                acc += simd_shuffle_down(acc, 1);
                mqk[cc] = simd_shuffle(acc, NL * ty);  // broadcast row sum within the ty group
            }
            ss[NE * tx + ty] = mqk[tx];  // lane (tx, ty) stores score of row ic + NE*tx + ty
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax — one pass over the whole 32-score block
        {
            float s = ss[tiisg] * p.scale;
            uint jkv = ic + tiisg;
            bool valid = (jkv >= lo) && (jkv <= abs);
            float m = M;
            M = simd_max(max(M, valid ? s : -MAXFLOAT / 2));
            float ms = exp(m - M);
            float vs = valid ? exp(s - M) : 0.0f;
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            if (ty == 0) {
                for (uint ii = 0; ii < NI; ii++) so4[ii * NL + tx] *= ms;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O += P*V — same (ty, tx) fold as Q*K^T, accumulated into the ty==0 lanes' columns
        {
            float4 lov[NI];
            for (uint ii = 0; ii < NI; ii++) lov[ii] = float4(0.0f);
            for (uint cc = 0; cc < C / NE; cc++) {
                uint rc = min(ic + NE * cc + ty, kvl - 1u);
                device const half4* pv = v4 + ((ulong)rc * p.n_kv + kvh) * hd4;
                float pw = ss[NE * cc + ty];
                for (uint ii = 0; ii < NI; ii++)
                    lov[ii] += float4(pv[ii * NL + tx]) * pw;
            }
            for (uint ii = 0; ii < NI; ii++) {
                lov[ii] += simd_shuffle_down(lov[ii], 16);
                lov[ii] += simd_shuffle_down(lov[ii], 8);
            }
            if (ty == 0) {
                for (uint ii = 0; ii < NI; ii++) so4[ii * NL + tx] += lov[ii];
            }
        }
    }

    // publish (S, M) for the merge (scores are dead)
    if (tiisg == 0) { ss[0] = S; ss[1] = M; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // log2 merge of the per-simdgroup partials
    for (uint rr = NSG / 2u; rr > 0u; rr >>= 1u) {
        if (sgitg < rr) {
            float s0 = ss[0], s1 = ssc[(sgitg + rr) * C + 0];
            float m0 = ss[1], m1 = ssc[(sgitg + rr) * C + 1];
            float mm = max(m0, m1);
            float ms0 = exp(m0 - mm), ms1 = exp(m1 - mm);
            if (tiisg == 0) { ss[0] = s0 * ms0 + s1 * ms1; ss[1] = mm; }
            threadgroup float4* sob = (threadgroup float4*)so + (sgitg + rr) * hd4;
            for (uint i = tiisg; i < hd4; i += 32u) so4[i] = so4[i] * ms0 + sob[i] * ms1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0) {
        float sinv = ssc[0] == 0.0f ? 0.0f : 1.0f / ssc[0];
        device float4* out = (device float4*)(dst + ((ulong)ti * p.n_head + h) * hd);
        for (uint i = tiisg; i < hd4; i += 32u) out[i] = so4[i] * sinv;
    }
}

template<uint hd, uint NSG>
kernel void attnvec_f16kv_t(device const float* q   [[buffer(0)]],
                            device const half*  k   [[buffer(1)]],
                            device const half*  v   [[buffer(2)]],
                            device float*       dst [[buffer(3)]],
                            constant AttnParams& p  [[buffer(4)]],
                            uint3  tgpig [[threadgroup_position_in_grid]],
                            ushort sgitg [[simdgroup_index_in_threadgroup]],
                            ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float sq[hd];
    threadgroup float ssc[NSG * 32];               // per-simdgroup scores, then P; (S, M) at merge
    threadgroup float so[NSG * hd];                // per-simdgroup O partials
    uint abs = p.pos + tgpig.x / p.n_head;
    attnvec_body<hd, NSG>(q, k, v, dst, p, abs, p.kv_len, sq, ssc, so, tgpig, sgitg, tiisg);
}

// Dynamic-pos variant for decode replay: `pos` comes from the bound positions buffer (updated by
// the host every token) instead of the recorded AttnParams, whose baked pos/kv_len are stale by
// the second replay. rows is 1 on this path, so kv_len is exactly pos + 1.
template<uint hd, uint NSG>
kernel void attnvec_dyn_f16kv_t(device const float* q    [[buffer(0)]],
                                device const half*  k    [[buffer(1)]],
                                device const half*  v    [[buffer(2)]],
                                device float*       dst  [[buffer(3)]],
                                device const int*   posb [[buffer(4)]],
                                constant AttnParams& p   [[buffer(5)]],
                                uint3  tgpig [[threadgroup_position_in_grid]],
                                ushort sgitg [[simdgroup_index_in_threadgroup]],
                                ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float sq[hd];
    threadgroup float ssc[NSG * 32];
    threadgroup float so[NSG * hd];
    uint abs = (uint)posb[0];
    attnvec_body<hd, NSG>(q, k, v, dst, p, abs, abs + 1u, sq, ssc, so, tgpig, sgitg, tiisg);
}

typedef decltype(attnvec_f16kv_t<64, 32>) attnvec_t;
template [[host_name("attnvec_f16kv_hd64")]]  kernel attnvec_t attnvec_f16kv_t<64, 32>;
template [[host_name("attnvec_f16kv_hd128")]] kernel attnvec_t attnvec_f16kv_t<128, 32>;
typedef decltype(attnvec_dyn_f16kv_t<64, 32>) attnvec_dyn_t;
template [[host_name("attnvec_dyn_f16kv_hd64")]]  kernel attnvec_dyn_t attnvec_dyn_f16kv_t<64, 32>;
template [[host_name("attnvec_dyn_f16kv_hd128")]] kernel attnvec_dyn_t attnvec_dyn_f16kv_t<128, 32>;

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

// ---- WriteKv:// ---- WriteKv: cast-copy `n` f32 source elems into the bound KV cache at row offset `base`, on the
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
// Dynamic-pos WriteKv for decode replay: the row offset is pos*row_stride with pos read from the
// bound positions buffer per token (`base` in a recorded params blob would be stale). f16 cache
// only (the replay gate requires it).
struct WriteKvDynParams { uint n; uint row_stride; };
kernel void writekv_dyn_f16(device const float* src   [[buffer(0)]],
                            device half*        cache [[buffer(1)]],
                            device const int*   posb  [[buffer(2)]],
                            constant WriteKvDynParams& p [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    cache[(uint)posb[0] * p.row_stride + gid] = (half)src[gid];
}

// ---- Q8_0 KV cache (INFR_KV_Q8): 34-byte blocks of 32 codes — half the f16 footprint and
// bandwidth. Quantization matches the CPU reference bit-for-bit: d = amax/127 stored as f16,
// q = rint(x/d) (ties to even, matching Rust's round_ties_even). Rows are 32-aligned (the
// runner gates on it), so a written row never straddles a block.
inline float q8_at(device const uchar* c, ulong e) {
    device const uchar* blk = c + (e >> 5) * 34ul;
    float d = (float)*(device const half*)blk; /* 34*b is even — the f16 d is always aligned */
    return d * (float)(char)blk[2u + (e & 31u)];
}
inline float4 q8_float4(device const uchar* c, ulong e) { /* e % 4 == 0: never straddles */
    device const uchar* blk = c + (e >> 5) * 34ul;
    float d = (float)*(device const half*)blk;
    /* codes start at blk+2 (2-byte aligned): two ushort loads cover the 4 codes */
    device const ushort* q2 = (device const ushort*)(blk + 2u + (e & 31u));
    uint lo = q2[0];
    uint hi = q2[1];
    return d * float4((float)(char)(lo & 0xFFu), (float)(char)(lo >> 8),
                      (float)(char)(hi & 0xFFu), (float)(char)(hi >> 8));
}
kernel void writekv_q8(device const float* src   [[buffer(0)]],
                       device uchar*       cache [[buffer(1)]],
                       constant WriteKvParams& p [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 32u) return;   // one thread per 32-elem block; p.base is in elements
    device const float* s = src + gid * 32u;
    float amax = 0.0f;
    for (uint i = 0; i < 32u; i++) amax = max(amax, fabs(s[i]));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    device uchar* blk = cache + ((ulong)(p.base >> 5) + gid) * 34ul;
    *(device half*)blk = (half)d;
    for (uint i = 0; i < 32u; i++) blk[2u + i] = (uchar)(char)(int)rint(s[i] * id);
}
kernel void writekv_dyn_q8(device const float* src   [[buffer(0)]],
                           device uchar*       cache [[buffer(1)]],
                           device const int*   posb  [[buffer(2)]],
                           constant WriteKvDynParams& p [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 32u) return;
    device const float* s = src + gid * 32u;
    float amax = 0.0f;
    for (uint i = 0; i < 32u; i++) amax = max(amax, fabs(s[i]));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    ulong base_blk = (ulong)(uint)posb[0] * (p.row_stride >> 5);
    device uchar* blk = cache + (base_blk + gid) * 34ul;
    *(device half*)blk = (half)d;
    for (uint i = 0; i < 32u; i++) blk[2u + i] = (uchar)(char)(int)rint(s[i] * id);
}

// Scalar attention over a Q8_0 cache — the attention_f16kv shape with dequant-on-read. The
// catch-all q8 route (prefill + odd shapes); the vector kernel below covers rows==1 decode.
kernel void attention_q8kv(device const float* q   [[buffer(0)]],
                           device const uchar* k   [[buffer(1)]],
                           device const uchar* v   [[buffer(2)]],
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
        ulong kb = ((ulong)j * p.n_kv + kvh) * p.head_dim;
        float part = 0.0f;
        for (uint d = lane; d < p.head_dim; d += 32u) part += q[qb + d] * q8_at(k, kb + d);
        float sc = simd_sum(part) * p.scale;
        float mnew = max(m, sc);
        float corr = exp(m - mnew);
        float pw = exp(sc - mnew);
        l = l * corr + pw;
        uint t = 0;
        for (uint d = lane; d < p.head_dim; d += 32u) {
            acc[t] = acc[t] * corr + pw * q8_at(v, kb + d);
            t++;
        }
        m = mnew;
    }
    uint t = 0;
    for (uint d = lane; d < p.head_dim; d += 32u) { dst[qb + d] = acc[t] / l; t++; }
}

// Vector flash attention over a Q8_0 cache — the attnvec structure with dequant-on-read
// (see attnvec_body; kept as a sibling body because the KV accessor type differs). Same
// numeric class: f32 dots over exactly-dequantized q8 values, reassociation only.
template<uint hd, uint NSG>
inline void attnvec_q8_body(device const float* q,
                            device const uchar* k,
                            device const uchar* v,
                            device float*       dst,
                            constant AttnParams& p,
                            uint abs, uint kvl,
                            threadgroup float* sq,
                            threadgroup float* ssc,
                            threadgroup float* so,
                            uint3  tgpig,
                            ushort sgitg,
                            ushort tiisg) {
    constexpr uint C = 32, NE = 4, NL = 32u / NE;
    constexpr uint hd4 = hd / 4u;
    constexpr uint NI = hd4 / NL;

    uint tg = tgpig.x;
    uint ti = tg / p.n_head;
    uint h  = tg % p.n_head;
    uint kvh = h / (p.n_head / p.n_kv);
    uint lo = (p.window > 0u && abs + 1u > p.window) ? (abs + 1u - p.window) : 0u;
    uint tx = tiisg % NL, ty = tiisg / NL;

    {
        device const float4* q4 = (device const float4*)(q + ((ulong)ti * p.n_head + h) * hd);
        threadgroup float4* sq4 = (threadgroup float4*)sq;
        for (uint i = sgitg * 32u + tiisg; i < hd4; i += NSG * 32u) sq4[i] = q4[i];
    }
    threadgroup float* ss = ssc + sgitg * C;
    threadgroup float4* so4 = (threadgroup float4*)so + sgitg * hd4;
    if (ty == 0) {
        for (uint ii = 0; ii < NI; ii++) so4[ii * NL + tx] = float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f, M = -MAXFLOAT / 2;
    threadgroup const float4* sq4 = (threadgroup const float4*)sq;

    for (uint ic = sgitg * C; ic <= abs; ic += NSG * C) {
        if (ic + C <= lo) continue;
        {
            float mqk[C / NE];
            for (uint cc = 0; cc < C / NE; cc++) {
                uint rc = min(ic + NE * cc + ty, kvl - 1u);
                ulong eb = ((ulong)rc * p.n_kv + kvh) * hd;
                float acc = 0.0f;
                for (uint ii = 0; ii < NI; ii++)
                    acc += dot(q8_float4(k, eb + (ii * NL + tx) * 4u), sq4[ii * NL + tx]);
                acc += simd_shuffle_down(acc, 4);
                acc += simd_shuffle_down(acc, 2);
                acc += simd_shuffle_down(acc, 1);
                mqk[cc] = simd_shuffle(acc, NL * ty);
            }
            ss[NE * tx + ty] = mqk[tx];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            float sv = ss[tiisg] * p.scale;
            uint jkv = ic + tiisg;
            bool valid = (jkv >= lo) && (jkv <= abs);
            float m = M;
            M = simd_max(max(M, valid ? sv : -MAXFLOAT / 2));
            float ms = exp(m - M);
            float vs = valid ? exp(sv - M) : 0.0f;
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            if (ty == 0) {
                for (uint ii = 0; ii < NI; ii++) so4[ii * NL + tx] *= ms;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            float4 lov[NI];
            for (uint ii = 0; ii < NI; ii++) lov[ii] = float4(0.0f);
            for (uint cc = 0; cc < C / NE; cc++) {
                uint rc = min(ic + NE * cc + ty, kvl - 1u);
                ulong eb = ((ulong)rc * p.n_kv + kvh) * hd;
                float pw = ss[NE * cc + ty];
                for (uint ii = 0; ii < NI; ii++)
                    lov[ii] += q8_float4(v, eb + (ii * NL + tx) * 4u) * pw;
            }
            for (uint ii = 0; ii < NI; ii++) {
                lov[ii] += simd_shuffle_down(lov[ii], 16);
                lov[ii] += simd_shuffle_down(lov[ii], 8);
            }
            if (ty == 0) {
                for (uint ii = 0; ii < NI; ii++) so4[ii * NL + tx] += lov[ii];
            }
        }
    }

    if (tiisg == 0) { ss[0] = S; ss[1] = M; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint rr = NSG / 2u; rr > 0u; rr >>= 1u) {
        if (sgitg < rr) {
            float s0 = ss[0], s1 = ssc[(sgitg + rr) * C + 0];
            float m0 = ss[1], m1 = ssc[(sgitg + rr) * C + 1];
            float mm = max(m0, m1);
            float ms0 = exp(m0 - mm), ms1 = exp(m1 - mm);
            if (tiisg == 0) { ss[0] = s0 * ms0 + s1 * ms1; ss[1] = mm; }
            threadgroup float4* sob = (threadgroup float4*)so + (sgitg + rr) * hd4;
            for (uint i = tiisg; i < hd4; i += 32u) so4[i] = so4[i] * ms0 + sob[i] * ms1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0) {
        float sinv = ssc[0] == 0.0f ? 0.0f : 1.0f / ssc[0];
        device float4* out = (device float4*)(dst + ((ulong)ti * p.n_head + h) * hd);
        for (uint i = tiisg; i < hd4; i += 32u) out[i] = so4[i] * sinv;
    }
}

template<uint hd, uint NSG>
kernel void attnvec_q8kv_t(device const float* q   [[buffer(0)]],
                           device const uchar* k   [[buffer(1)]],
                           device const uchar* v   [[buffer(2)]],
                           device float*       dst [[buffer(3)]],
                           constant AttnParams& p  [[buffer(4)]],
                           uint3  tgpig [[threadgroup_position_in_grid]],
                           ushort sgitg [[simdgroup_index_in_threadgroup]],
                           ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float sq[hd];
    threadgroup float ssc[NSG * 32];
    threadgroup float so[NSG * hd];
    uint abs = p.pos + tgpig.x / p.n_head;
    attnvec_q8_body<hd, NSG>(q, k, v, dst, p, abs, p.kv_len, sq, ssc, so, tgpig, sgitg, tiisg);
}

template<uint hd, uint NSG>
kernel void attnvec_dyn_q8kv_t(device const float* q    [[buffer(0)]],
                               device const uchar* k    [[buffer(1)]],
                               device const uchar* v    [[buffer(2)]],
                               device float*       dst  [[buffer(3)]],
                               device const int*   posb [[buffer(4)]],
                               constant AttnParams& p   [[buffer(5)]],
                               uint3  tgpig [[threadgroup_position_in_grid]],
                               ushort sgitg [[simdgroup_index_in_threadgroup]],
                               ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float sq[hd];
    threadgroup float ssc[NSG * 32];
    threadgroup float so[NSG * hd];
    uint abs = (uint)posb[0];
    attnvec_q8_body<hd, NSG>(q, k, v, dst, p, abs, abs + 1u, sq, ssc, so, tgpig, sgitg, tiisg);
}

// Cooperative flash attention over a Q8_0 cache — the attnflash2 structure with a cooperative
// dequant-staging stage (the llama.cpp flash_attn_ext quantized-KV branch shape): K/V can't be
// simdgroup_load'ed from q8 blocks, so per 64-position KV block all 128 threads dequantize the
// tile into threadgroup memory — K PRE-TRANSPOSED [hd][64] so the QK fragments load
// non-transposed and conflict-free, V row-major [64][hd] staged into the SAME tile during the
// softmax phase (the K reads are done by then; one extra barrier per block). ~24 KB threadgroup
// at hd=128 (vs 8 KB for the f16 kernel — the occupancy cost of in-kernel dequant).
template<uint hd, uint NSG>
kernel void attnflash2_q8kv_t(device const half*  q   [[buffer(0)]],
                              device const uchar* k   [[buffer(1)]],
                              device const uchar* v   [[buffer(2)]],
                              device float*       dst [[buffer(3)]],
                              constant AttnParams& p  [[buffer(4)]],
                              uint3  tgpig [[threadgroup_position_in_grid]],
                              ushort sgitg [[simdgroup_index_in_threadgroup]],
                              ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint QT = 8, C = 64, NQ = QT / NSG, SH = C;
    constexpr uint NBR = hd / 32u;          // q8 blocks per KV row
    threadgroup half  sq[QT * hd];
    threadgroup float so[QT * hd];
    threadgroup float ss[QT * SH];
    threadgroup half  kvt[C * hd];          // K as [hd][C], then V as [C][hd]

    uint ntq = (p.rows + QT - 1u) / QT;
    uint qt = tgpig.x % ntq;
    uint h  = tgpig.x / ntq;
    constexpr uint hd4 = hd / 4u;
    constexpr uint no = hd / (8u * NSG);
    constexpr uint NC = (C / 8u) / NSG;
    uint kvh = h / (p.n_head / p.n_kv);
    uint r0 = qt * QT;
    uint abs0 = p.pos + r0;
    uint abs_max = p.pos + min(p.rows - 1u, r0 + QT - 1u);
    uint lo_min = (p.window > 0u && abs0 + 1u > p.window) ? (abs0 + 1u - p.window) : 0u;
    ulong qstride = (ulong)p.n_head * hd;
    uint tid = (uint)sgitg * 32u + tiisg;

    for (uint jj = 0; jj < NQ; jj++) {
        uint j = jj * NSG + sgitg;
        bool live = r0 + j < p.rows;
        device const half4* q4 =
            (device const half4*)(q + (ulong)min(r0 + j, p.rows - 1u) * qstride + (ulong)h * hd);
        threadgroup half4*  sq4 = (threadgroup half4*)sq + j * hd4;
        threadgroup float4* so4 = (threadgroup float4*)so + j * hd4;
        for (uint i = tiisg; i < hd4; i += 32u) {
            sq4[i] = live ? q4[i] : half4(0.0h);
            so4[i] = float4(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    for (uint jj = 0; jj < NQ; jj++) { S[jj] = 0.0f; M[jj] = -MAXFLOAT / 2; }

    for (uint ic = lo_min & ~(C - 1u); ic <= abs_max; ic += C) {
        // stage K [hd][C]: each thread dequantizes whole q8 blocks (clamped rows are masked
        // in the softmax, so their junk never contributes)
        for (uint b = tid; b < C * NBR; b += NSG * 32u) {
            uint rr = b / NBR;
            uint dsub = (b % NBR) * 32u;
            uint rc = min(ic + rr, p.kv_len - 1u);
            ulong eb = ((ulong)rc * p.n_kv + kvh) * hd + dsub;
            device const uchar* blk = k + (eb >> 5) * 34ul;
            float d = (float)*(device const half*)blk;
            for (uint i = 0; i < 32u; i++)
                kvt[(dsub + i) * C + rr] = (half)(d * (float)(char)blk[2u + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            threadgroup const half* pk = kvt + 8u * sgitg;
            threadgroup float* ps = ss + 8u * sgitg;
            for (uint cc = 0; cc < NC; cc++) {
                simdgroup_float8x8 mqk = simdgroup_float8x8(0.0f);
                if (ic + 8u * (sgitg + cc * NSG) <= abs_max) {
                    for (uint i = 0; i < hd; i += 16u) {
                        simdgroup_half8x8 mq, mk;
                        simdgroup_load(mq, sq + i, hd);
                        simdgroup_load(mk, pk + i * C, C);
                        simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        simdgroup_load(mq, sq + i + 8u, hd);
                        simdgroup_load(mk, pk + (i + 8u) * C, C);
                        simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                    }
                }
                simdgroup_store(mqk, ps, SH);
                pk += 8u * NSG;
                ps += 8u * NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // softmax (rows split across simdgroups) + V staging [C][hd] in the same phase —
        // the K reads are complete, the V reads haven't started
        for (uint jj = 0; jj < NQ; jj++) {
            uint j = jj * NSG + sgitg;
            uint absr = abs0 + j;
            uint lor = (p.window > 0u && absr + 1u > p.window) ? (absr + 1u - p.window) : 0u;
            threadgroup float2* ss2 = (threadgroup float2*)(ss + j * SH);
            float2 s2 = ss2[tiisg] * p.scale;
            uint c0 = ic + 2u * tiisg;
            bool v0 = (c0 >= lor) && (c0 <= absr);
            bool v1 = (c0 + 1u >= lor) && (c0 + 1u <= absr);
            float m = M[jj];
            float mnew =
                simd_max(max(m, max(v0 ? s2.x : -MAXFLOAT / 2, v1 ? s2.y : -MAXFLOAT / 2)));
            float ms = exp(m - mnew);
            float pw0 = v0 ? exp(s2.x - mnew) : 0.0f;
            float pw1 = v1 ? exp(s2.y - mnew) : 0.0f;
            S[jj] = S[jj] * ms + simd_sum(pw0 + pw1);
            M[jj] = mnew;
            ss2[tiisg] = float2(pw0, pw1);
            threadgroup float4* so4 = (threadgroup float4*)so + j * hd4;
            for (uint i = tiisg; i < hd4; i += 32u) so4[i] *= ms;
        }
        for (uint b = tid; b < C * NBR; b += NSG * 32u) {
            uint rr = b / NBR;
            uint dsub = (b % NBR) * 32u;
            uint rc = min(ic + rr, p.kv_len - 1u);
            ulong eb = ((ulong)rc * p.n_kv + kvh) * hd + dsub;
            device const uchar* blk = v + (eb >> 5) * 34ul;
            float d = (float)*(device const half*)blk;
            for (uint i = 0; i < 32u; i++)
                kvt[rr * hd + dsub + i] = (half)(d * (float)(char)blk[2u + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            simdgroup_float8x8 lo[4];
            threadgroup float* sot = so + 8u * sgitg;
            for (uint ii = 0; ii < no; ii++) simdgroup_load(lo[ii], sot + 8u * NSG * ii, hd);
            threadgroup const half* pv = kvt + 8u * sgitg;
            uint nblk = min(C / 8u, (abs_max - ic) / 8u + 1u);
            for (uint cc = 0; cc < nblk; cc++) {
                simdgroup_float8x8 vs;
                simdgroup_load(vs, ss + 8u * cc, SH);
                for (uint ii = 0; ii < no; ii++) {
                    simdgroup_half8x8 mv;
                    simdgroup_load(mv, pv + 8u * NSG * ii, hd);
                    simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
                }
                pv += 8u * hd;
            }
            for (uint ii = 0; ii < no; ii++) simdgroup_store(lo[ii], sot + 8u * NSG * ii, hd);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint jj = 0; jj < NQ; jj++) {
        uint j = jj * NSG + sgitg;
        if (r0 + j >= p.rows) continue;
        float sc = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        device float4* out = (device float4*)(dst + ((ulong)(r0 + j)) * qstride + (ulong)h * hd);
        threadgroup const float4* so4 = (threadgroup const float4*)so + j * hd4;
        for (uint i = tiisg; i < hd4; i += 32u) out[i] = so4[i] * sc;
    }
}

typedef decltype(attnflash2_q8kv_t<64, 4>) attnflash2_q8_t;
template [[host_name("attnflash2_q8kv_hd64")]]  kernel attnflash2_q8_t attnflash2_q8kv_t<64, 4>;
template [[host_name("attnflash2_q8kv_hd128")]] kernel attnflash2_q8_t attnflash2_q8kv_t<128, 4>;

typedef decltype(attnvec_q8kv_t<64, 32>) attnvec_q8_t;
template [[host_name("attnvec_q8kv_hd64")]]  kernel attnvec_q8_t attnvec_q8kv_t<64, 32>;
template [[host_name("attnvec_q8kv_hd128")]] kernel attnvec_q8_t attnvec_q8kv_t<128, 32>;
typedef decltype(attnvec_dyn_q8kv_t<64, 32>) attnvec_dyn_q8_t;
template [[host_name("attnvec_dyn_q8kv_hd64")]]  kernel attnvec_dyn_q8_t attnvec_dyn_q8kv_t<64, 32>;
template [[host_name("attnvec_dyn_q8kv_hd128")]] kernel attnvec_dyn_q8_t attnvec_dyn_q8kv_t<128, 32>;
"#;
