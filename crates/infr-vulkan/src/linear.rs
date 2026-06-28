//! Persistent-weight linear layer: `y = W · x` where `W` is stored `[out, in]` row-major
//! (the GGUF layout: data index `o*in + i`). The weight buffer is uploaded once
//! (`upload_weight`) and reused; the compute pipeline is built once (cached in
//! `VulkanShared.linear_kernel`) and reused across all calls — only the (small) activation
//! buffers are created per call.
//!
//! WGSL → SPIR-V via naga, same pattern as `matmul.rs`.

use std::ffi::CStr;
use std::sync::OnceLock;

use ash::vk;

use infr_core::{
    backend::{Buffer, BufferUsage},
    error::Result,
    Backend,
};

use super::{as_vk_buf, be, VulkanBackend};

pub(crate) const LINEAR_WGSL: &str = r#"
struct PushConstants { rows: u32, in_f: u32, out_f: u32 }
var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read>       w_buf: array<f32>; // [out, in]  (w[o*in+i])
@group(0) @binding(1) var<storage, read>       x_buf: array<f32>; // [rows, in]
@group(0) @binding(2) var<storage, read_write> y_buf: array<f32>; // [rows, out]

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = pc.rows * pc.out_f;
    if idx >= total { return; }
    let r = idx / pc.out_f;
    let o = idx % pc.out_f;
    let wbase = o * pc.in_f;
    let xbase = r * pc.in_f;
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < pc.in_f; i = i + 1u) {
        acc = acc + w_buf[wbase + i] * x_buf[xbase + i];
    }
    y_buf[r * pc.out_f + o] = acc;
}
"#;

/// Like `LINEAR_WGSL` but adds a residual: `y = residual + x·Wᵀ`. `r_buf` and `y_buf` may alias
/// (in-place residual): each invocation reads and writes only index `idx`, so it is safe.
pub(crate) const LINEAR_RES_WGSL: &str = r#"
enable f16;
struct PushConstants { rows: u32, in_f: u32, out_f: u32 }
var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read>       w_buf: array<f16>; // [out, in] f16
@group(0) @binding(1) var<storage, read>       x_buf: array<f32>; // [rows, in]
@group(0) @binding(2) var<storage, read>       r_buf: array<f32>; // [rows, out] residual
@group(0) @binding(3) var<storage, read_write> y_buf: array<f32>; // [rows, out]

var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let unit = wid.x;            // = r * out_f + o
    let o = unit % pc.out_f;
    let r = unit / pc.out_f;
    let t = lid.x;
    let wbase = o * pc.in_f;
    let xbase = r * pc.in_f;
    var acc: f32 = 0.0;
    for (var i: u32 = t; i < pc.in_f; i = i + 64u) {
        acc = acc + f32(w_buf[wbase + i]) * x_buf[xbase + i];
    }
    red[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if t < stride { red[t] = red[t] + red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if t == 0u { y_buf[unit] = r_buf[unit] + red[0]; }
}
"#;

/// f16-weight GEMV `y = x·Wᵀ` for the recorder (e.g. the LM head). ONE workgroup per output
/// element: its 64 threads stride the K dimension so consecutive lanes read consecutive weights
/// (coalesced), then a tree-reduce sums the partials. Dispatch `rows*out_f` workgroups. This is
/// far more bandwidth-efficient than thread-per-output (which read weight rows stride-K apart).
pub(crate) const LINEAR_F16_WGSL: &str = r#"
enable f16;
struct PushConstants { rows: u32, in_f: u32, out_f: u32 }
var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read>       w_buf: array<f16>; // [out, in] f16
@group(0) @binding(1) var<storage, read>       x_buf: array<f32>; // [rows, in]
@group(0) @binding(2) var<storage, read_write> y_buf: array<f32>; // [rows, out]

var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let unit = wid.x;            // = r * out_f + o
    let o = unit % pc.out_f;
    let r = unit / pc.out_f;
    let t = lid.x;
    let wbase = o * pc.in_f;
    let xbase = r * pc.in_f;
    var acc: f32 = 0.0;
    for (var i: u32 = t; i < pc.in_f; i = i + 64u) {
        acc = acc + f32(w_buf[wbase + i]) * x_buf[xbase + i];
    }
    red[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if t < stride { red[t] = red[t] + red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if t == 0u { y_buf[unit] = red[0]; }
}
"#;

/// bf16-weight GEMV `y = x·Wᵀ`. WGSL has no native bf16, so weights are stored as a flat u16 stream
/// packed 2-per-u32; each is unpacked losslessly to f32 by `bitcast(bf16_bits << 16)` (bf16 IS the
/// top 16 bits of an f32). Same cooperative-over-K layout as `LINEAR_F16_WGSL`; dispatch `rows*out_f`
/// workgroups. Element addressing is global (word = elem/2, half = elem&1) so rows need not be u32-
/// aligned even when `in_f` is odd.
pub(crate) const LINEAR_BF16_WGSL: &str = r#"
struct PushConstants { rows: u32, in_f: u32, out_f: u32 }
var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read>       w_buf: array<u32>; // [out, in] bf16 packed 2/u32
@group(0) @binding(1) var<storage, read>       x_buf: array<f32>; // [rows, in]
@group(0) @binding(2) var<storage, read_write> y_buf: array<f32>; // [rows, out]

var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let unit = wid.x;            // = r * out_f + o
    let o = unit % pc.out_f;
    let r = unit / pc.out_f;
    let t = lid.x;
    let wbase = o * pc.in_f;
    let xbase = r * pc.in_f;
    var acc: f32 = 0.0;
    for (var i: u32 = t; i < pc.in_f; i = i + 64u) {
        let gi = wbase + i;                                  // global element index
        let word = w_buf[gi >> 1u];
        var bits16: u32 = word & 0xffffu;
        if ((gi & 1u) == 1u) { bits16 = word >> 16u; }
        acc = acc + bitcast<f32>(bits16 << 16u) * x_buf[xbase + i];
    }
    red[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if t < stride { red[t] = red[t] + red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if t == 0u { y_buf[unit] = red[0]; }
}
"#;

/// Unified quantized-weight dequant GEMV `y = x·Wᵀ` (cooperative-over-K, like `LINEAR_F16_WGSL`).
/// ALL supported quants repack at load into one form: `quants` = index per element packed at
/// `pc.bits` (4 → 8/u32 for Q4, 8 → 4/u32 for Q5/Q6/Q8), `scales`/`mins` = one f16 each per
/// `1<<blk_shift`-element block; `dq(g) = scales·q + mins`. Dispatch `rows*out_f` workgroups.
pub(crate) const LINEAR_Q_WGSL: &str = r#"
enable f16;
struct PushConstants { rows: u32, in_f: u32, out_f: u32, bits: u32, blk_shift: u32 }
var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read>       quants: array<u32>;
@group(0) @binding(1) var<storage, read>       scales: array<f16>;
@group(0) @binding(2) var<storage, read>       mins: array<f16>;
@group(0) @binding(3) var<storage, read>       x_buf: array<f32>;  // [rows, in]
@group(0) @binding(4) var<storage, read_write> y_buf: array<f32>;  // [rows, out]

var<workgroup> red: array<f32, 64>;

fn dq(g: u32) -> f32 {
    var q: f32;
    if pc.bits == 4u {
        q = f32((quants[g >> 3u] >> ((g & 7u) * 4u)) & 0xFu);
    } else {
        q = f32((quants[g >> 2u] >> ((g & 3u) * 8u)) & 0xFFu);
    }
    let blk = g >> pc.blk_shift;
    return f32(scales[blk]) * q + f32(mins[blk]);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let unit = wid.x;
    let o = unit % pc.out_f;
    let r = unit / pc.out_f;
    let t = lid.x;
    let wbase = o * pc.in_f;
    let xbase = r * pc.in_f;
    var acc: f32 = 0.0;
    for (var i: u32 = t; i < pc.in_f; i = i + 64u) {
        acc = acc + dq(wbase + i) * x_buf[xbase + i];
    }
    red[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if t < stride { red[t] = red[t] + red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if t == 0u { y_buf[unit] = red[0]; }
}
"#;

/// Unified quant dequant GEMV with fused residual add: `y = residual + x·Wᵀ`.
pub(crate) const LINEAR_RES_Q_WGSL: &str = r#"
enable f16;
struct PushConstants { rows: u32, in_f: u32, out_f: u32, bits: u32, blk_shift: u32 }
var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read>       quants: array<u32>;
@group(0) @binding(1) var<storage, read>       scales: array<f16>;
@group(0) @binding(2) var<storage, read>       mins: array<f16>;
@group(0) @binding(3) var<storage, read>       x_buf: array<f32>;
@group(0) @binding(4) var<storage, read>       r_buf: array<f32>; // [rows, out] residual
@group(0) @binding(5) var<storage, read_write> y_buf: array<f32>;

var<workgroup> red: array<f32, 64>;

fn dq(g: u32) -> f32 {
    var q: f32;
    if pc.bits == 4u {
        q = f32((quants[g >> 3u] >> ((g & 7u) * 4u)) & 0xFu);
    } else {
        q = f32((quants[g >> 2u] >> ((g & 3u) * 8u)) & 0xFFu);
    }
    let blk = g >> pc.blk_shift;
    return f32(scales[blk]) * q + f32(mins[blk]);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let unit = wid.x;
    let o = unit % pc.out_f;
    let r = unit / pc.out_f;
    let t = lid.x;
    let wbase = o * pc.in_f;
    let xbase = r * pc.in_f;
    var acc: f32 = 0.0;
    for (var i: u32 = t; i < pc.in_f; i = i + 64u) {
        acc = acc + dq(wbase + i) * x_buf[xbase + i];
    }
    red[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if t < stride { red[t] = red[t] + red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if t == 0u { y_buf[unit] = r_buf[unit] + red[0]; }
}
"#;

// ─── Native-block dequant GEMV shaders (Phase 0-2) ─────────────────────────
//
// Each shader reads raw GGUF block bytes (uploaded padded to a u32-multiple)
// from `w_buf: array<u32>` and dequantizes elements in-shader. The outer GEMV
// cooperative-over-K structure matches LINEAR_F16_WGSL: one workgroup per
// output element, 64 threads stride K, tree-reduce.
//
// All WGSL is generated by `native_gemv_wgsl(dtype, residual)` at first call
// (the kernel is then compiled to SPIR-V and cached by name).

/// Common WGSL header for native-block shaders (no-residual variant, 3 bindings).
const NATIVE_HDR: &str = r#"
struct PushConstants { rows: u32, in_f: u32, out_f: u32 }
var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read>       w_buf: array<u32>;
@group(0) @binding(1) var<storage, read>       x_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> y_buf: array<f32>;

var<workgroup> red: array<f32, 64>;

fn rb(bo: u32) -> u32 { return (w_buf[bo >> 2u] >> ((bo & 3u) << 3u)) & 0xFFu; }
fn ru16(bo: u32) -> u32 { return rb(bo) | (rb(bo + 1u) << 8u); }
fn ru32b(bo: u32) -> u32 { return rb(bo) | (rb(bo+1u)<<8u) | (rb(bo+2u)<<16u) | (rb(bo+3u)<<24u); }
fn f16tof32(bits: u32) -> f32 {
    let s = (bits >> 15u) & 1u;
    let e = (bits >> 10u) & 0x1Fu;
    let m = bits & 0x3FFu;
    if e == 0u {
        if m == 0u { return bitcast<f32>(s << 31u); }
        // Subnormal f16: value = (-1)^s * m * 2^(-24). Convert via integer→float multiply.
        let v = f32(m) * bitcast<f32>(0x33800000u);
        return select(v, -v, s != 0u);
    }
    if e == 31u { return bitcast<f32>((s << 31u) | 0x7F800000u | (m << 13u)); }
    return bitcast<f32>((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}
"#;

/// Common WGSL header for native-block shaders (residual variant, 4 bindings).
const NATIVE_HDR_RES: &str = r#"
struct PushConstants { rows: u32, in_f: u32, out_f: u32 }
var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read>       w_buf: array<u32>;
@group(0) @binding(1) var<storage, read>       x_buf: array<f32>;
@group(0) @binding(2) var<storage, read>       r_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> y_buf: array<f32>;

var<workgroup> red: array<f32, 64>;

fn rb(bo: u32) -> u32 { return (w_buf[bo >> 2u] >> ((bo & 3u) << 3u)) & 0xFFu; }
fn ru16(bo: u32) -> u32 { return rb(bo) | (rb(bo + 1u) << 8u); }
fn ru32b(bo: u32) -> u32 { return rb(bo) | (rb(bo+1u)<<8u) | (rb(bo+2u)<<16u) | (rb(bo+3u)<<24u); }
fn f16tof32(bits: u32) -> f32 {
    let s = (bits >> 15u) & 1u;
    let e = (bits >> 10u) & 0x1Fu;
    let m = bits & 0x3FFu;
    if e == 0u {
        if m == 0u { return bitcast<f32>(s << 31u); }
        let v = f32(m) * bitcast<f32>(0x33800000u);
        return select(v, -v, s != 0u);
    }
    if e == 31u { return bitcast<f32>((s << 31u) | 0x7F800000u | (m << 13u)); }
    return bitcast<f32>((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}
"#;

/// GEMV main function body (no residual). `dq()` is defined by each format's snippet.
const NATIVE_BODY: &str = r#"
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let unit = wid.x;
    let o = unit % pc.out_f;
    let r = unit / pc.out_f;
    let t = lid.x;
    let wbase = o * pc.in_f;
    let xbase = r * pc.in_f;
    var acc: f32 = 0.0;
    for (var i: u32 = t; i < pc.in_f; i = i + 64u) {
        acc = acc + dq(wbase + i) * x_buf[xbase + i];
    }
    red[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if t < stride { red[t] = red[t] + red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if t == 0u { y_buf[unit] = red[0]; }
}
"#;

/// GEMV+residual main function body.
const NATIVE_BODY_RES: &str = r#"
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let unit = wid.x;
    let o = unit % pc.out_f;
    let r = unit / pc.out_f;
    let t = lid.x;
    let wbase = o * pc.in_f;
    let xbase = r * pc.in_f;
    var acc: f32 = 0.0;
    for (var i: u32 = t; i < pc.in_f; i = i + 64u) {
        acc = acc + dq(wbase + i) * x_buf[xbase + i];
    }
    red[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if t < stride { red[t] = red[t] + red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if t == 0u { y_buf[unit] = r_buf[unit] + red[0]; }
}
"#;

/// Per-format dequant function snippets. Each defines `fn dq(g: u32) -> f32`.

/// Q8_0: [f16 d][i8 qs[32]] = 34 bytes, 32 elements. y = d * qs[j] (signed).
const DQ_Q8_0: &str = r#"
fn dq(g: u32) -> f32 {
    let b  = g / 32u;
    let j  = g % 32u;
    let bd = b * 34u;
    let d  = f16tof32(ru16(bd));
    let raw = rb(bd + 2u + j);
    let q = i32(raw) - i32(select(0u, 256u, raw >= 128u));
    return d * f32(q);
}
"#;

/// Q4_0: [f16 d][u8 qs[16]] = 18 bytes, 32 elements. y = d * (nibble - 8).
const DQ_Q4_0: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 32u;
    let j = g % 32u;
    let bd = b * 18u;
    let d = f16tof32(ru16(bd));
    var nibble: u32;
    if j < 16u {
        nibble = rb(bd + 2u + j) & 0xFu;
    } else {
        nibble = rb(bd + 2u + j - 16u) >> 4u;
    }
    return d * (f32(nibble) - 8.0);
}
"#;

/// Q4_1: [f16 d][f16 m][u8 qs[16]] = 20 bytes, 32 elements. y = d*nibble + m.
const DQ_Q4_1: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 32u;
    let j = g % 32u;
    let bd = b * 20u;
    let d = f16tof32(ru16(bd));
    let m = f16tof32(ru16(bd + 2u));
    var nibble: u32;
    if j < 16u {
        nibble = rb(bd + 4u + j) & 0xFu;
    } else {
        nibble = rb(bd + 4u + j - 16u) >> 4u;
    }
    return d * f32(nibble) + m;
}
"#;

/// Q5_0: [f16 d][u8 qh[4]][u8 qs[16]] = 22 bytes, 32 elements. y = d*(q5-16).
const DQ_Q5_0: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 32u;
    let j = g % 32u;
    let bd = b * 22u;
    let d = f16tof32(ru16(bd));
    let qh = ru32b(bd + 2u);
    var val: u32;
    if j < 16u {
        let xh0 = ((qh >> j) << 4u) & 0x10u;
        val = (rb(bd + 6u + j) & 0xFu) | xh0;
    } else {
        let jj = j - 16u;
        let xh1 = (qh >> (jj + 12u)) & 0x10u;
        val = (rb(bd + 6u + jj) >> 4u) | xh1;
    }
    return d * (f32(val) - 16.0);
}
"#;

/// Q5_1: [f16 d][f16 m][u8 qh[4]][u8 qs[16]] = 24 bytes, 32 elements. y = d*q5 + m.
const DQ_Q5_1: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 32u;
    let j = g % 32u;
    let bd = b * 24u;
    let d = f16tof32(ru16(bd));
    let m = f16tof32(ru16(bd + 2u));
    let qh = ru32b(bd + 4u);
    var val: u32;
    if j < 16u {
        let xh0 = ((qh >> j) << 4u) & 0x10u;
        val = (rb(bd + 8u + j) & 0xFu) | xh0;
    } else {
        let jj = j - 16u;
        let xh1 = (qh >> (jj + 12u)) & 0x10u;
        val = (rb(bd + 8u + jj) >> 4u) | xh1;
    }
    return d * f32(val) + m;
}
"#;

/// Q2_K: [u8 scales[16]][u8 qs[64]][f16 d][f16 dmin] = 84 bytes, 256 elements.
/// y = d*(sc&0xF)*q2 - dmin*(sc>>4)
const DQ_Q2K: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 84u;
    let d    = f16tof32(ru16(bd + 80u));
    let dmin = f16tof32(ru16(bd + 82u));
    let sc_byte = rb(bd + p / 16u);
    let dl = d * f32(sc_byte & 0xFu);
    let ml = dmin * f32(sc_byte >> 4u);
    let n       = p / 128u;
    let p_half  = p % 128u;
    let j       = p_half / 32u;
    let p_j     = p_half % 32u;
    let qs_idx  = 32u * n + p_j;
    let shift   = 2u * j;
    let q2 = (rb(bd + 16u + qs_idx) >> shift) & 3u;
    return dl * f32(q2) - ml;
}
"#;

/// Q3_K: [u8 hmask[32]][u8 qs[64]][u8 scales_raw[12]][f16 d] = 110 bytes, 256 elements.
/// 6-bit sub-block scales; q3u ∈ 0..7; y = dl*(q3u - 4).
const DQ_Q3K: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 110u;
    let d_all = f16tof32(ru16(bd + 108u));
    // decode 6-bit scales (port of llama.cpp bit manipulation)
    let a0 = ru32b(bd + 96u);
    let a1 = ru32b(bd + 100u);
    let a2 = ru32b(bd + 104u);
    let k1: u32 = 0x03030303u;
    let k2: u32 = 0x0f0f0f0fu;
    let tmp = a2;
    var aux: array<u32, 4>;
    aux[2] = ((a0 >> 4u) & k2) | (((tmp >> 4u) & k1) << 4u);
    aux[3] = ((a1 >> 4u) & k2) | (((tmp >> 6u) & k1) << 4u);
    aux[0] = (a0 & k2) | (((tmp) & k1) << 4u);
    aux[1] = (a1 & k2) | (((tmp >> 2u) & k1) << 4u);
    // scale index = p/16
    let is = p / 16u;
    let sc_byte = (aux[is >> 2u] >> ((is & 3u) * 8u)) & 0xFFu;
    let sc = i32(sc_byte) - i32(select(0u, 256u, sc_byte >= 128u)) - 32;
    let dl = d_all * f32(sc);
    // element mapping
    let n       = p / 128u;
    let p_half  = p % 128u;
    let j       = p_half / 32u;
    let p_j     = p_half % 32u;
    let shift   = 2u * j;
    let jg      = 4u * n + j;         // global j index (0..7)
    let m       = 1u << jg;
    let qs_idx  = 32u * n + p_j;
    let hm_idx  = p_j;
    let low2 = (rb(bd + 32u + qs_idx) >> shift) & 3u;
    let high = select(0u, 1u, (rb(bd + hm_idx) & m) != 0u);
    let q3u = low2 | (high << 2u);
    return dl * (f32(q3u) - 4.0);
}
"#;

/// k4 helper used by Q4_K and Q5_K: decode 6-bit scale/min from 12-byte scales field.
const K4_FN: &str = r#"
fn k4(i: u32, sb: u32) -> vec2<u32> {
    if i < 4u {
        return vec2<u32>(rb(sb + i) & 63u, rb(sb + i + 4u) & 63u);
    } else {
        let sc = (rb(sb + i + 4u) & 0xFu) | ((rb(sb + i - 4u) >> 6u) << 4u);
        let mn = (rb(sb + i + 4u) >> 4u) | ((rb(sb + i) >> 6u) << 4u);
        return vec2<u32>(sc, mn);
    }
}
"#;

/// Q4_K: [f16 d][f16 dmin][u8 scales[12]][u8 qs[128]] = 144 bytes, 256 elements.
const DQ_Q4K: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 144u;
    let d    = f16tof32(ru16(bd));
    let dmin = f16tof32(ru16(bd + 2u));
    let sb   = bd + 4u;
    let j    = p / 64u;
    let p_j  = p % 64u;
    let l    = p_j % 32u;
    var k: vec2<u32>;
    if p_j < 32u { k = k4(2u * j, sb); } else { k = k4(2u * j + 1u, sb); }
    let dl = d * f32(k.x);
    let mm = dmin * f32(k.y);
    let qs_byte = rb(bd + 16u + j * 32u + l);
    var val: u32;
    if p_j < 32u { val = qs_byte & 0xFu; } else { val = qs_byte >> 4u; }
    return dl * f32(val) - mm;
}
"#;

/// Q5_K: [f16 d][f16 dmin][u8 scales[12]][u8 qh[32]][u8 qs[128]] = 176 bytes, 256 elements.
const DQ_Q5K: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 176u;
    let d    = f16tof32(ru16(bd));
    let dmin = f16tof32(ru16(bd + 2u));
    let sb   = bd + 4u;
    let j    = p / 64u;
    let p_j  = p % 64u;
    let l    = p_j % 32u;
    var k: vec2<u32>;
    if p_j < 32u { k = k4(2u * j, sb); } else { k = k4(2u * j + 1u, sb); }
    let dl = d * f32(k.x);
    let mm = dmin * f32(k.y);
    let qs_byte = rb(bd + 48u + j * 32u + l);
    let qh_byte = rb(bd + 16u + l);
    var val: u32;
    if p_j < 32u {
        val = (qs_byte & 0xFu) + select(0u, 16u, (qh_byte & (1u << (2u * j))) != 0u);
    } else {
        val = (qs_byte >> 4u) + select(0u, 16u, (qh_byte & (2u << (2u * j))) != 0u);
    }
    return dl * f32(val) - mm;
}
"#;

/// Q6_K: [u8 ql[128]][u8 qh[64]][i8 scales[16]][f16 d] = 210 bytes, 256 elements.
/// 6-bit index 0..63; y = d * sc_i8 * (q6 - 32).
const DQ_Q6K: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 210u;
    let d = f16tof32(ru16(bd + 208u));
    let half   = p / 128u;
    let p_half = p % 128u;
    let og     = p_half / 32u;
    let l      = p_half % 32u;
    let qlo    = half * 64u;
    let qho    = half * 32u;
    let qa = rb(bd + qlo + l);
    let qb = rb(bd + qlo + l + 32u);
    let qh = rb(bd + 128u + qho + l);
    var q: u32;
    if og == 0u      { q = (qa & 0xFu) | ((qh & 3u) << 4u); }
    else if og == 1u { q = (qb & 0xFu) | (((qh >> 2u) & 3u) << 4u); }
    else if og == 2u { q = (qa >> 4u)  | (((qh >> 4u) & 3u) << 4u); }
    else             { q = (qb >> 4u)  | (((qh >> 6u) & 3u) << 4u); }
    let sc_idx = half * 8u + l / 16u + 2u * og;
    let sc_raw = rb(bd + 192u + sc_idx);
    let sc = f32(i32(sc_raw) - i32(select(0u, 256u, sc_raw >= 128u)));
    return d * sc * (f32(q) - 32.0);
}
"#;

// ── Codebook-format helpers (shared const tables / decode fns, prepended per format) ──

/// IQ4_NL / IQ4_XS 16-entry signed codebook (kvalues_iq4nl).
const KV_IQ4NL_DEF: &str = r#"
const KV_IQ4NL = array<i32,16>(-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113);
"#;
/// MXFP4 / NVFP4 E2M1 4-bit codebook (kvalues_mxfp4).
const KV_MXFP4_DEF: &str = r#"
const KV_MXFP4 = array<i32,16>(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
"#;
/// TQ1_0 ternary powers of 3.
const POW3_TQ_DEF: &str = r#"
const POW3_TQ = array<u32,6>(1u,3u,9u,27u,81u,243u);
"#;
/// E8M0 exponent byte → fp32 (halved): `2^(x-128)`.
const E8M0_FN: &str = r#"
fn e8m0_half(x: u32) -> f32 {
    if x < 2u { return bitcast<f32>(0x00200000u << x); }
    return bitcast<f32>((x - 1u) << 23u);
}
"#;
/// UE4M3 byte → fp32 (halved): 4 exp bits (bias 7), 3 mantissa bits.
const UE4M3_FN: &str = r#"
fn ue4m3(x: u32) -> f32 {
    if x == 0u || x == 0x7Fu { return 0.0; }
    let e = (x >> 3u) & 0xFu;
    let man = f32(x & 7u);
    var raw: f32;
    if e == 0u { raw = man * exp2(-9.0); }
    else { raw = (1.0 + man / 8.0) * exp2(f32(i32(e) - 7)); }
    return raw * 0.5;
}
"#;

/// IQ4_NL: [f16 d][u8 qs[16]] = 18 bytes, 32 elements. y = d * kv[nibble].
const DQ_IQ4_NL: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 32u;
    let j = g % 32u;
    let bd = b * 18u;
    let d = f16tof32(ru16(bd));
    var idx: u32;
    if j < 16u { idx = rb(bd + 2u + j) & 0xFu; } else { idx = rb(bd + 2u + (j - 16u)) >> 4u; }
    return d * f32(KV_IQ4NL[idx]);
}
"#;

/// IQ4_XS: [f16 d][u16 scales_h][u8 scales_l[4]][u8 qs[128]] = 136 bytes, 256 elements.
/// 8 sub-blocks of 32; 6-bit per-sub-block scale `ls`; y = d*(ls-32) * kv[nibble].
const DQ_IQ4_XS: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 136u;
    let d = f16tof32(ru16(bd));
    let scales_h = ru16(bd + 2u);
    let ib = p / 32u;
    let within = p % 32u;
    let lo = (rb(bd + 4u + (ib / 2u)) >> (4u * (ib & 1u))) & 0xFu;
    let hi = (scales_h >> (2u * ib)) & 3u;
    let ls = lo | (hi << 4u);
    let dl = d * f32(i32(ls) - 32);
    let qoff = bd + 8u + 16u * ib;
    var idx: u32;
    if within < 16u { idx = rb(qoff + within) & 0xFu; } else { idx = rb(qoff + (within - 16u)) >> 4u; }
    return dl * f32(KV_IQ4NL[idx]);
}
"#;

/// MXFP4: [u8 e8m0][u8 qs[16]] = 17 bytes, 32 elements. y = kv_mxfp4[nibble] * 2^(e-128).
const DQ_MXFP4: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 32u;
    let j = g % 32u;
    let bd = b * 17u;
    let d = e8m0_half(rb(bd));
    var idx: u32;
    if j < 16u { idx = rb(bd + 1u + j) & 0xFu; } else { idx = rb(bd + 1u + (j - 16u)) >> 4u; }
    return f32(KV_MXFP4[idx]) * d;
}
"#;

/// NVFP4: [u8 scales[4]][u8 qs[32]] = 36 bytes, 64 elements. 4 sub-blocks of 16, UE4M3 scale each.
const DQ_NVFP4: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 64u;
    let p = g % 64u;
    let bd = b * 36u;
    let s = p / 16u;
    let within = p % 16u;
    let d = ue4m3(rb(bd + s));
    var idx: u32;
    if within < 8u { idx = rb(bd + 4u + s * 8u + within) & 0xFu; }
    else { idx = rb(bd + 4u + s * 8u + (within - 8u)) >> 4u; }
    return f32(KV_MXFP4[idx]) * d;
}
"#;

/// TQ1_0: [u8 qs[48]][u8 qh[4]][f16 d] = 54 bytes, 256 elements. Ternary via pow-3 digit extract.
const DQ_TQ1_0: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 54u;
    let d = f16tof32(ru16(bd + 52u));
    var src: u32;
    var n: u32;
    if p < 160u { n = p / 32u; src = rb(bd + (p % 32u)); }
    else if p < 240u { let pp = p - 160u; n = pp / 16u; src = rb(bd + 32u + (pp % 16u)); }
    else { let pp = p - 240u; n = pp / 4u; src = rb(bd + 48u + (pp % 4u)); }
    let q = (src * POW3_TQ[n]) & 0xFFu;
    let xi = (q * 3u) >> 8u;
    return f32(i32(xi) - 1) * d;
}
"#;

/// TQ2_0: [u8 qs[64]][f16 d] = 66 bytes, 256 elements. 2-bit ternary, y = (q-1)*d.
const DQ_TQ2_0: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 66u;
    let d = f16tof32(ru16(bd + 64u));
    let chunk = p / 128u;
    let rem = p % 128u;
    let l = rem / 32u;
    let m = rem % 32u;
    let q = (rb(bd + chunk * 32u + m) >> (l * 2u)) & 3u;
    return (f32(q) - 1.0) * d;
}
"#;

// ── Grid-based i-quant helpers ────────────────────────────────────────────────
// WGSL has no u64, so each u64 grid entry is emitted as two u32 words (lo, hi); the dq() snippet
// reads byte j from the lo word (j<4) or hi word (j>=4). Grids/ksigns are embedded as module
// consts (baked into the per-format SPIR-V, compiled once and cached by kernel name).

/// Emit a `u64` grid as a WGSL `const NAME = array<u32, 2N>(lo0, hi0, ...)`.
fn grid_u64_wgsl(name: &str, grid: &[u64]) -> String {
    let mut s = format!("const {name} = array<u32,{}>(", grid.len() * 2);
    for &v in grid {
        s.push_str(&format!("{}u,{}u,", v as u32, (v >> 32) as u32));
    }
    s.pop();
    s.push_str(");\n");
    s
}

/// Emit a `u32` grid (IQ3) as a WGSL `const NAME = array<u32, N>(...)`.
fn grid_u32_wgsl(name: &str, grid: &[u32]) -> String {
    let mut s = format!("const {name} = array<u32,{}>(", grid.len());
    for &v in grid {
        s.push_str(&format!("{v}u,"));
    }
    s.pop();
    s.push_str(");\n");
    s
}

/// The 128-entry `ksigns_iq2xs` sign-mask table as a WGSL const.
fn ksigns_wgsl() -> String {
    let mut s = String::from("const KSIGNS = array<u32,128>(");
    for &v in infr_core::iquant_grids::KSIGNS_IQ2XS.iter() {
        s.push_str(&format!("{v}u,"));
    }
    s.pop();
    s.push_str(");\n");
    s
}

/// IQ2_XXS: [f16 d][u16 qs[32]] = 66 bytes, 256 elements. 8-bit grid index + 7-bit sign + scale.
const DQ_IQ2XXS: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 66u;
    let d = f16tof32(ru16(bd));
    let ib32 = p / 32u;
    let l = (p % 32u) / 8u;
    let j = p % 8u;
    let off = bd + 2u + ib32 * 8u;
    let aux0 = ru32b(off);
    let aux1 = ru32b(off + 4u);
    let grid_idx = (aux0 >> (8u * l)) & 0xFFu;
    let sign_idx = (aux1 >> (7u * l)) & 127u;
    let db = d * (0.5 + f32(aux1 >> 28u)) * 0.25;
    var byte: u32;
    if j < 4u { byte = (G_IQ2XXS[2u * grid_idx] >> (8u * j)) & 0xFFu; }
    else { byte = (G_IQ2XXS[2u * grid_idx + 1u] >> (8u * (j - 4u))) & 0xFFu; }
    let gv = f32(i32(byte) - i32(select(0u, 256u, byte >= 128u)));
    let sign = select(1.0, -1.0, ((KSIGNS[sign_idx] >> j) & 1u) != 0u);
    return db * gv * sign;
}
"#;

/// IQ2_XS: [f16 d][u16 qs[32]][u8 scales[8]] = 74 bytes, 256 elements. 9-bit grid + 7-bit sign.
const DQ_IQ2XS: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 74u;
    let d = f16tof32(ru16(bd));
    let ib32 = p / 32u;
    let l = (p % 32u) / 8u;
    let j = p % 8u;
    let qs16 = ru16(bd + 2u + (ib32 * 4u + l) * 2u);
    let grid_idx = qs16 & 511u;
    let sign_idx = qs16 >> 9u;
    let sc = rb(bd + 66u + ib32);
    let db0 = d * (0.5 + f32(sc & 0xFu)) * 0.25;
    let db1 = d * (0.5 + f32(sc >> 4u)) * 0.25;
    let dl = select(db1, db0, l < 2u);
    var byte: u32;
    if j < 4u { byte = (G_IQ2XS[2u * grid_idx] >> (8u * j)) & 0xFFu; }
    else { byte = (G_IQ2XS[2u * grid_idx + 1u] >> (8u * (j - 4u))) & 0xFFu; }
    let gv = f32(i32(byte) - i32(select(0u, 256u, byte >= 128u)));
    let sign = select(1.0, -1.0, ((KSIGNS[sign_idx] >> j) & 1u) != 0u);
    return dl * gv * sign;
}
"#;

/// IQ2_S: [f16 d][u8 qs[64]][u8 qh[8]][u8 scales[8]] = 82 bytes, 256 elements. 10-bit grid; signs
/// are inline bytes (qs[32..64]), no ksigns table.
const DQ_IQ2S: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 82u;
    let d = f16tof32(ru16(bd));
    let ib32 = p / 32u;
    let l = (p % 32u) / 8u;
    let j = p % 8u;
    let qs_byte = rb(bd + 2u + ib32 * 4u + l);
    let sign_byte = rb(bd + 2u + 32u + ib32 * 4u + l);
    let qh_byte = rb(bd + 66u + ib32);
    let hi = (qh_byte << (8u - 2u * l)) & 0x300u;
    let grid_idx = qs_byte | hi;
    let sc = rb(bd + 74u + ib32);
    let db0 = d * (0.5 + f32(sc & 0xFu)) * 0.25;
    let db1 = d * (0.5 + f32(sc >> 4u)) * 0.25;
    let dl = select(db1, db0, l < 2u);
    var byte: u32;
    if j < 4u { byte = (G_IQ2S[2u * grid_idx] >> (8u * j)) & 0xFFu; }
    else { byte = (G_IQ2S[2u * grid_idx + 1u] >> (8u * (j - 4u))) & 0xFFu; }
    let gv = f32(i32(byte) - i32(select(0u, 256u, byte >= 128u)));
    let sign = select(1.0, -1.0, ((sign_byte >> j) & 1u) != 0u);
    return dl * gv * sign;
}
"#;

/// IQ3_XXS: [f16 d][u8 qs[64]][u8 sas[32]] = 98 bytes, 256 elements. u32 grid (256 entries).
const DQ_IQ3XXS: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 98u;
    let d = f16tof32(ru16(bd));
    let ib32 = p / 32u;
    let l = (p % 32u) / 8u;
    let j8 = p % 8u;
    let g1_idx = rb(bd + 2u + ib32 * 8u + 2u * l);
    let g2_idx = rb(bd + 2u + ib32 * 8u + 2u * l + 1u);
    let aux32 = ru32b(bd + 66u + 4u * ib32);
    let db = d * (0.5 + f32(aux32 >> 28u)) * 0.5;
    let signs = KSIGNS[(aux32 >> (7u * l)) & 127u];
    var gidx: u32;
    var bytej: u32;
    if j8 < 4u { gidx = g1_idx; bytej = j8; } else { gidx = g2_idx; bytej = j8 - 4u; }
    let byte = (G_IQ3XXS[gidx] >> (8u * bytej)) & 0xFFu;
    let gv = f32(i32(byte) - i32(select(0u, 256u, byte >= 128u)));
    let sign = select(1.0, -1.0, ((signs >> j8) & 1u) != 0u);
    return db * gv * sign;
}
"#;

/// IQ3_S: [f16 d][u8 qs[64]][u8 qh[8]][u8 signs[32]][u8 scales[4]] = 110 bytes, 256 elements.
const DQ_IQ3S: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 110u;
    let d = f16tof32(ru16(bd));
    let pair = p / 64u;
    let group = (p % 64u) / 32u;
    let within = p % 32u;
    let l = within / 8u;
    let j8 = within % 8u;
    let sb = rb(bd + 106u + pair);
    let db = select(d * (1.0 + 2.0 * f32(sb & 0xFu)), d * (1.0 + 2.0 * f32(sb >> 4u)), group == 1u);
    let qh = rb(bd + 66u + pair * 2u + group);
    let qs_base = bd + 2u + pair * 16u + group * 8u;
    let signs_byte = rb(bd + 74u + pair * 8u + group * 4u + l);
    var grididx: u32;
    var bytej: u32;
    if j8 < 4u {
        grididx = rb(qs_base + 2u * l) | ((qh << (8u - 2u * l)) & 256u);
        bytej = j8;
    } else {
        grididx = rb(qs_base + 2u * l + 1u) | ((qh << (7u - 2u * l)) & 256u);
        bytej = j8 - 4u;
    }
    let byte = (G_IQ3S[grididx] >> (8u * bytej)) & 0xFFu;
    let gv = f32(i32(byte) - i32(select(0u, 256u, byte >= 128u)));
    let sign = select(1.0, -1.0, ((signs_byte >> j8) & 1u) != 0u);
    return db * gv * sign;
}
"#;

/// IQ1_S: [f16 d][u8 qs[32]][u16 qh[8]] = 50 bytes, 256 elements. 11-bit grid index + delta.
const DQ_IQ1S: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 50u;
    let d = f16tof32(ru16(bd));
    let ib = p / 32u;
    let l = (p % 32u) / 8u;
    let j = p % 8u;
    let qh = ru16(bd + 34u + 2u * ib);
    let dl = d * (2.0 * f32((qh >> 12u) & 7u) + 1.0);
    let delta = select(0.125, -0.125, (qh & 0x8000u) != 0u);
    let grid_idx = rb(bd + 2u + ib * 4u + l) | (((qh >> (3u * l)) & 7u) << 8u);
    var byte: u32;
    if j < 4u { byte = (G_IQ1S[2u * grid_idx] >> (8u * j)) & 0xFFu; }
    else { byte = (G_IQ1S[2u * grid_idx + 1u] >> (8u * (j - 4u))) & 0xFFu; }
    let gv = f32(i32(byte) - i32(select(0u, 256u, byte >= 128u)));
    return dl * (gv + delta);
}
"#;

/// IQ1_M: [u8 qs[32]][u8 qh[16]][u8 scales[8]] = 56 bytes, 256 elements. d packed in scale nibbles;
/// reuses the IQ1S grid.
const DQ_IQ1M: &str = r#"
fn dq(g: u32) -> f32 {
    let b = g / 256u;
    let p = g % 256u;
    let bd = b * 56u;
    let sc0 = ru16(bd + 48u);
    let sc1 = ru16(bd + 50u);
    let sc2 = ru16(bd + 52u);
    let sc3 = ru16(bd + 54u);
    let d_bits = (sc0 >> 12u) | ((sc1 >> 8u) & 0xF0u) | ((sc2 >> 4u) & 0xF00u) | (sc3 & 0xF000u);
    let d = f16tof32(d_bits);
    let ib = p / 32u;
    let l = (p % 32u) / 8u;
    let j = p % 8u;
    var scw: u32;
    let scidx = ib / 2u;
    if scidx == 0u { scw = sc0; } else if scidx == 1u { scw = sc1; }
    else if scidx == 2u { scw = sc2; } else { scw = sc3; }
    let dl1 = d * (2.0 * f32((scw >> (6u * (ib & 1u))) & 7u) + 1.0);
    let dl2 = d * (2.0 * f32((scw >> (6u * (ib & 1u) + 3u)) & 7u) + 1.0);
    let dl = select(dl1, dl2, l >= 2u);
    let qh0 = rb(bd + 32u + ib * 2u);
    let qh1 = rb(bd + 32u + ib * 2u + 1u);
    var grididx: u32;
    var deltaneg: bool;
    if l == 0u { grididx = rb(bd + ib * 4u) | ((qh0 << 8u) & 0x700u); deltaneg = (qh0 & 0x08u) != 0u; }
    else if l == 1u { grididx = rb(bd + ib * 4u + 1u) | ((qh0 << 4u) & 0x700u); deltaneg = (qh0 & 0x80u) != 0u; }
    else if l == 2u { grididx = rb(bd + ib * 4u + 2u) | ((qh1 << 8u) & 0x700u); deltaneg = (qh1 & 0x08u) != 0u; }
    else { grididx = rb(bd + ib * 4u + 3u) | ((qh1 << 4u) & 0x700u); deltaneg = (qh1 & 0x80u) != 0u; }
    let delta = select(0.125, -0.125, deltaneg);
    var byte: u32;
    if j < 4u { byte = (G_IQ1S[2u * grididx] >> (8u * j)) & 0xFFu; }
    else { byte = (G_IQ1S[2u * grididx + 1u] >> (8u * (j - 4u))) & 0xFFu; }
    let gv = f32(i32(byte) - i32(select(0u, 256u, byte >= 128u)));
    return dl * (gv + delta);
}
"#;

/// Return the static kernel name for a native-block GEMV (Phase 0-2).
pub fn native_kernel_name(dtype: infr_core::DType, residual: bool) -> &'static str {
    use infr_core::DType::*;
    match (dtype, residual) {
        (Q8_0, false) => "native_q8_0",
        (Q8_0, true) => "native_q8_0_res",
        (Q4_0, false) => "native_q4_0",
        (Q4_0, true) => "native_q4_0_res",
        (Q4_1, false) => "native_q4_1",
        (Q4_1, true) => "native_q4_1_res",
        (Q5_0, false) => "native_q5_0",
        (Q5_0, true) => "native_q5_0_res",
        (Q5_1, false) => "native_q5_1",
        (Q5_1, true) => "native_q5_1_res",
        (Q2K, false) => "native_q2k",
        (Q2K, true) => "native_q2k_res",
        (Q3K, false) => "native_q3k",
        (Q3K, true) => "native_q3k_res",
        (Q4K, false) => "native_q4k",
        (Q4K, true) => "native_q4k_res",
        (Q5K, false) => "native_q5k",
        (Q5K, true) => "native_q5k_res",
        (Q6K, false) => "native_q6k",
        (Q6K, true) => "native_q6k_res",
        (Iq4Nl, false) => "native_iq4nl",
        (Iq4Nl, true) => "native_iq4nl_res",
        (Iq4Xs, false) => "native_iq4xs",
        (Iq4Xs, true) => "native_iq4xs_res",
        (Mxfp4, false) => "native_mxfp4",
        (Mxfp4, true) => "native_mxfp4_res",
        (Nvfp4, false) => "native_nvfp4",
        (Nvfp4, true) => "native_nvfp4_res",
        (Tq1_0, false) => "native_tq1_0",
        (Tq1_0, true) => "native_tq1_0_res",
        (Tq2_0, false) => "native_tq2_0",
        (Tq2_0, true) => "native_tq2_0_res",
        (Iq2Xxs, false) => "native_iq2xxs",
        (Iq2Xxs, true) => "native_iq2xxs_res",
        (Iq2Xs, false) => "native_iq2xs",
        (Iq2Xs, true) => "native_iq2xs_res",
        (Iq2S, false) => "native_iq2s",
        (Iq2S, true) => "native_iq2s_res",
        (Iq3Xxs, false) => "native_iq3xxs",
        (Iq3Xxs, true) => "native_iq3xxs_res",
        (Iq3S, false) => "native_iq3s",
        (Iq3S, true) => "native_iq3s_res",
        (Iq1S, false) => "native_iq1s",
        (Iq1S, true) => "native_iq1s_res",
        (Iq1M, false) => "native_iq1m",
        (Iq1M, true) => "native_iq1m_res",
        _ => panic!("no native GEMV for {:?}", dtype),
    }
}

/// Build a native-block GEMV WGSL program string for `dtype` (± residual add).
/// Called once per format the first time the kernel is needed; the compiled SPIR-V
/// is cached by name so this string is only used once.
pub fn native_gemv_wgsl(dtype: infr_core::DType, residual: bool) -> String {
    use infr_core::DType::*;
    let (hdr, body) = if residual {
        (NATIVE_HDR_RES, NATIVE_BODY_RES)
    } else {
        (NATIVE_HDR, NATIVE_BODY)
    };
    let dq = match dtype {
        Q8_0 => DQ_Q8_0,
        Q4_0 => DQ_Q4_0,
        Q4_1 => DQ_Q4_1,
        Q5_0 => DQ_Q5_0,
        Q5_1 => DQ_Q5_1,
        Q2K => DQ_Q2K,
        Q3K => DQ_Q3K,
        // Q4K and Q5K share the k4() helper function
        Q4K => return format!("{hdr}{K4_FN}{DQ_Q4K}{body}"),
        Q5K => return format!("{hdr}{K4_FN}{DQ_Q5K}{body}"),
        Q6K => DQ_Q6K,
        // Codebook formats: prepend their const table(s) / decode fn(s).
        Iq4Nl => return format!("{hdr}{KV_IQ4NL_DEF}{DQ_IQ4_NL}{body}"),
        Iq4Xs => return format!("{hdr}{KV_IQ4NL_DEF}{DQ_IQ4_XS}{body}"),
        Mxfp4 => return format!("{hdr}{KV_MXFP4_DEF}{E8M0_FN}{DQ_MXFP4}{body}"),
        Nvfp4 => return format!("{hdr}{KV_MXFP4_DEF}{UE4M3_FN}{DQ_NVFP4}{body}"),
        Tq1_0 => return format!("{hdr}{POW3_TQ_DEF}{DQ_TQ1_0}{body}"),
        Tq2_0 => return format!("{hdr}{DQ_TQ2_0}{body}"),
        Iq2Xxs => {
            let grid = grid_u64_wgsl("G_IQ2XXS", &infr_core::iquant_grids::IQ2XXS_GRID);
            let ks = ksigns_wgsl();
            return format!("{hdr}{grid}{ks}{DQ_IQ2XXS}{body}");
        }
        Iq2Xs => {
            let grid = grid_u64_wgsl("G_IQ2XS", &infr_core::iquant_grids::IQ2XS_GRID);
            let ks = ksigns_wgsl();
            return format!("{hdr}{grid}{ks}{DQ_IQ2XS}{body}");
        }
        Iq2S => {
            let grid = grid_u64_wgsl("G_IQ2S", &infr_core::iquant_grids::IQ2S_GRID);
            return format!("{hdr}{grid}{DQ_IQ2S}{body}");
        }
        Iq3Xxs => {
            let grid = grid_u32_wgsl("G_IQ3XXS", &infr_core::iquant_grids::IQ3XXS_GRID);
            let ks = ksigns_wgsl();
            return format!("{hdr}{grid}{ks}{DQ_IQ3XXS}{body}");
        }
        Iq3S => {
            let grid = grid_u32_wgsl("G_IQ3S", &infr_core::iquant_grids::IQ3S_GRID);
            return format!("{hdr}{grid}{DQ_IQ3S}{body}");
        }
        Iq1S => {
            let grid = grid_u64_wgsl("G_IQ1S", &infr_core::iquant_grids::IQ1S_GRID);
            return format!("{hdr}{grid}{DQ_IQ1S}{body}");
        }
        Iq1M => {
            let grid = grid_u64_wgsl("G_IQ1S", &infr_core::iquant_grids::IQ1S_GRID);
            return format!("{hdr}{grid}{DQ_IQ1M}{body}");
        }
        _ => panic!("no native GEMV for {dtype:?}"),
    };
    format!("{hdr}{dq}{body}")
}

/// Pad raw GGUF block bytes to the next multiple of 4 for upload as `array<u32>`.
/// Appends zero bytes; the final u32 word's padding bytes are never read (they
/// contain only out-of-block data which the shader never accesses for valid g).
pub fn pad_to_u32_align(bytes: &[u8]) -> Vec<u8> {
    let padded = (bytes.len() + 3) & !3;
    let mut v = bytes.to_vec();
    v.resize(padded, 0u8);
    v
}

static LINEAR_SPV: OnceLock<Vec<u32>> = OnceLock::new();

fn linear_spv() -> &'static [u32] {
    LINEAR_SPV.get_or_init(|| {
        use naga::back::spv;
        use naga::front::wgsl;
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let module = wgsl::parse_str(LINEAR_WGSL).expect("linear WGSL parse");
        let info = Validator::new(ValidationFlags::all(), Capabilities::IMMEDIATES)
            .validate(&module)
            .expect("linear WGSL validate");
        spv::write_vec(
            &module,
            &info,
            &spv::Options {
                lang_version: (1, 3),
                ..Default::default()
            },
            None,
        )
        .expect("linear SPIR-V write")
    })
}

/// Cached, reusable compute objects for the linear kernel (built once per device).
pub(crate) struct LinearKernel {
    pub shader: vk::ShaderModule,
    pub ds_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub desc_pool: vk::DescriptorPool,
}

pub(crate) fn create_linear_kernel(device: &ash::Device) -> LinearKernel {
    let spv = linear_spv();
    let shader = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spv), None)
    }
    .expect("create linear shader module");

    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let ds_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .expect("create linear ds layout");

    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(12);
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&ds_layout))
                .push_constant_ranges(std::slice::from_ref(&push_range)),
            None,
        )
    }
    .expect("create linear pipeline layout");

    let entry = CStr::from_bytes_with_nul(b"main\0").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(entry);
    let pipeline = unsafe {
        device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout)],
                None,
            )
            .expect("create linear pipeline")[0]
    };

    // Pool holds one set; we reset + reallocate it each call (single-stream gen).
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 3,
    }];
    let desc_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )
    }
    .expect("create linear desc pool");

    LinearKernel {
        shader,
        ds_layout,
        pipeline_layout,
        pipeline,
        desc_pool,
    }
}

pub(crate) fn destroy_linear_kernel(device: &ash::Device, k: &LinearKernel) {
    unsafe {
        device.destroy_descriptor_pool(k.desc_pool, None);
        device.destroy_pipeline(k.pipeline, None);
        device.destroy_pipeline_layout(k.pipeline_layout, None);
        device.destroy_descriptor_set_layout(k.ds_layout, None);
        device.destroy_shader_module(k.shader, None);
    }
}

impl VulkanBackend {
    fn linear_kernel(&self) -> &LinearKernel {
        self.shared
            .linear_kernel
            .get_or_init(|| create_linear_kernel(&self.shared.device))
    }

    /// Upload an `[out, in]` f32 weight to a persistent device buffer.
    pub fn upload_weight(&self, data: &[f32]) -> Result<Box<dyn Buffer>> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let buf = self.alloc(bytes.len(), BufferUsage::Weights)?;
        self.upload(buf.as_ref(), bytes)?;
        Ok(buf)
    }

    /// Upload an `[out, in]` weight as f16 (halves device bandwidth for the GEMV/matmul kernels
    /// that read weights). Source stays f32; converted on the host.
    pub fn upload_weight_f16(&self, data: &[f32]) -> Result<Box<dyn Buffer>> {
        let f16: Vec<u16> = data
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        self.upload_weight_bytes(bytemuck::cast_slice(&f16))
    }

    /// Upload an `[out, in]` weight as bf16 (truncate-round of f32; bf16 is the top 16 bits of f32).
    /// Read back losslessly to f32 in-shader by `LINEAR_BF16_WGSL`. Preserves f32's exponent range
    /// (unlike f16), so it's the correct GPU storage for bf16-source tensors that would overflow f16.
    pub fn upload_weight_bf16(&self, data: &[f32]) -> Result<Box<dyn Buffer>> {
        let bf16: Vec<u16> = data
            .iter()
            .map(|&x| {
                // round-to-nearest-even on the f32→bf16 truncation
                let bits = x.to_bits();
                let round = 0x7fffu32 + ((bits >> 16) & 1);
                ((bits.wrapping_add(round)) >> 16) as u16
            })
            .collect();
        self.upload_weight_bytes(bytemuck::cast_slice(&bf16))
    }

    /// Upload raw weight bytes (already in the target dtype) to a persistent device buffer.
    /// Use for f16 GGUF tensors to skip the f16→f32→f16 round-trip.
    pub fn upload_weight_bytes(&self, bytes: &[u8]) -> Result<Box<dyn Buffer>> {
        let buf = self.alloc(bytes.len(), BufferUsage::Weights)?;
        self.upload(buf.as_ref(), bytes)?;
        Ok(buf)
    }

    /// Compute `y[rows, out] = x[rows, in] · Wᵀ` where `w_buf` holds `W[out, in]`.
    /// Reuses the cached pipeline; only the per-call x/y buffers + descriptor set are fresh.
    pub fn linear(
        &self,
        w_buf: &dyn Buffer,
        x: &[f32],
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<Vec<f32>> {
        assert_eq!(x.len(), rows * in_f, "x must be rows*in");
        let device = self.shared.device.clone();
        let k = self.linear_kernel();

        // fresh descriptor set from the cached pool
        unsafe {
            device
                .reset_descriptor_pool(k.desc_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| be(format!("reset_descriptor_pool: {e}")))?;
        }
        let desc_set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(k.desc_pool)
                        .set_layouts(std::slice::from_ref(&k.ds_layout)),
                )
                .map_err(|e| be(format!("allocate_descriptor_sets: {e}")))?[0]
        };

        // Host-visible activation buffers: upload/download become direct memcpy (no extra
        // submit+wait), leaving the dispatch as the only GPU round-trip in this call.
        let x_bytes: &[u8] = bytemuck::cast_slice(x);
        let buf_x = self.alloc(x_bytes.len(), BufferUsage::Staging)?;
        let buf_y = self.alloc(rows * out_f * 4, BufferUsage::Readback)?;
        self.upload(buf_x.as_ref(), x_bytes)?;

        let vk_w = unsafe { as_vk_buf(w_buf) }.buffer;
        let vk_x = unsafe { as_vk_buf(buf_x.as_ref()) }.buffer;
        let vk_y = unsafe { as_vk_buf(buf_y.as_ref()) }.buffer;

        let infos = [
            vk::DescriptorBufferInfo {
                buffer: vk_w,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: vk_x,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: vk_y,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
        ];
        let writes: Vec<vk::WriteDescriptorSet> = (0..3)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i..i + 1])
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());

        let groups = ((rows * out_f) as u32).div_ceil(64);
        let shared = std::sync::Arc::clone(&self.shared);
        let (pipeline, pipeline_layout) = (k.pipeline, k.pipeline_layout);
        self.one_shot(move |cmd| unsafe {
            let barriers = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(vk_x)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            shared.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
            shared
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            shared.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            shared.device.cmd_push_constants(
                cmd,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            shared.device.cmd_dispatch(cmd, groups, 1, 1);
        })?;

        let mut y_bytes = vec![0u8; rows * out_f * 4];
        self.download(buf_y.as_ref(), &mut y_bytes)?;
        Ok(bytemuck::cast_slice(&y_bytes).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn linear_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (rows, in_f, out_f) = (3usize, 5usize, 4usize);
        let w: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32) * 0.01).collect();
        let x: Vec<f32> = (0..rows * in_f).map(|i| (i as f32) * 0.02).collect();
        let wbuf = be.upload_weight(&w).unwrap();
        // run twice to exercise the cached pipeline path
        let _ = be.linear(wbuf.as_ref(), &x, rows, in_f, out_f).unwrap();
        let y = be.linear(wbuf.as_ref(), &x, rows, in_f, out_f).unwrap();
        let mut want = vec![0.0f32; rows * out_f];
        for r in 0..rows {
            for o in 0..out_f {
                let mut acc = 0.0;
                for i in 0..in_f {
                    acc += x[r * in_f + i] * w[o * in_f + i];
                }
                want[r * out_f + o] = acc;
            }
        }
        for (g, w) in y.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-3, "{g} vs {w}");
        }
    }

    // CPU reference GEMV for the f16/bf16 eager-path tests (odd in_f exercises bf16 packing).
    fn cpu_gemv(w: &[f32], x: &[f32], rows: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * out_f];
        for r in 0..rows {
            for o in 0..out_f {
                let mut acc = 0.0;
                for i in 0..in_f {
                    acc += x[r * in_f + i] * w[o * in_f + i];
                }
                y[r * out_f + o] = acc;
            }
        }
        y
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn linear_f16_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (rows, in_f, out_f) = (2usize, 70usize, 5usize);
        let w: Vec<f32> = (0..out_f * in_f)
            .map(|i| (i as f32 % 9.0) * 0.05 - 0.2)
            .collect();
        let x: Vec<f32> = (0..rows * in_f).map(|i| (i as f32 % 7.0) * 0.03).collect();
        let wbuf = be.upload_weight_f16(&w).unwrap();
        let _ = be.linear_f16(wbuf.as_ref(), &x, rows, in_f, out_f).unwrap();
        let y = be.linear_f16(wbuf.as_ref(), &x, rows, in_f, out_f).unwrap();
        for (g, c) in y.iter().zip(cpu_gemv(&w, &x, rows, in_f, out_f).iter()) {
            assert!((g - c).abs() < 1e-2, "{g} vs {c}");
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn linear_bf16_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        // in_f odd → rows are NOT u32-aligned in the packed bf16 stream (exercises global addressing)
        let (rows, in_f, out_f) = (3usize, 65usize, 4usize);
        let w: Vec<f32> = (0..out_f * in_f)
            .map(|i| (i as f32 % 11.0) * 0.04 - 0.2)
            .collect();
        let x: Vec<f32> = (0..rows * in_f).map(|i| (i as f32 % 5.0) * 0.06).collect();
        let wbuf = be.upload_weight_bf16(&w).unwrap();
        let _ = be
            .linear_bf16(wbuf.as_ref(), &x, rows, in_f, out_f)
            .unwrap();
        let y = be
            .linear_bf16(wbuf.as_ref(), &x, rows, in_f, out_f)
            .unwrap();
        // bf16 has 8 mantissa bits → looser tolerance than f16
        for (g, c) in y.iter().zip(cpu_gemv(&w, &x, rows, in_f, out_f).iter()) {
            assert!((g - c).abs() < 5e-2, "{g} vs {c}");
        }
    }
}
