//! HIP kernel-source assembly and hiprtc compilation.
//!
//! Each kernel is a `__global__` function taking device pointers. All operate on f16 or f32
//! buffers — quantized weights are dequantized to f16 on the host BEFORE they reach a kernel
//! (see `exec.rs`'s dequant cache), so the kernels stay format-agnostic and simple.
//!
//! On first use each kernel name is fetched via `hipModuleGetFunction` and cached in a
//! `HashMap`. The module is compiled once at backend init via `hiprtcCompileProgram`.

use crate::be;
use crate::ffi;
use infr_core::error::Result;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, CString};
use std::sync::Mutex;

// ── Kernel source ────────────────────────────────────────────────────────────

/// Assemble the complete HIP source string from its parts.
pub fn hip_source() -> String {
    let mut s = String::with_capacity(128 * 1024);
    for part in HIP_PARTS {
        s.push_str(part);
    }
    s
}

/// The individual kernel source parts (one per kernel, so the hot-patching assembly is greppable).
const HIP_PARTS: &[&str] = &[
    RMSNORM,
    RMSNORM_ADD,
    SOFTMAX,
    LINEAR_F16,
    QK_NORM,
    ROPE,
    QK_NORM_ROPE,
    GATED_RMSNORM,
    GATED_ACT,
    ADD,
    ADD_BIAS,
    SCALE,
    MUL_VEC,
    SOFTCAP,
    COPY,
    COPY_STRIDED,
    EMBED_GATHER,
    ARGMAX,
    WRITE_KV,
    ATTENTION,
    MOE_FFN,
    CONV1D_SILU,
    DELTANET,
    MOE_SHARED_EXPERT_ADD,
];

const RMSNORM: &str = r#"
extern "C" __global__ void rmsnorm(
    const __half* __restrict__ x,    // [rows, dim]
    const __half* __restrict__ weight,// [dim]
    float* __restrict__ dst,         // [rows, dim]
    int rows,
    int dim,
    float eps
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float ss = 0.0f;
    const __half* xr = x + row * dim;
    for (int i = 0; i < dim; i++) {
        float v = __half2float(xr[i]);
        ss += v * v;
    }
    ss /= (float)dim;
    float rms = 1.0f / sqrtf(ss + eps);
    float* d = dst + row * dim;
    for (int i = 0; i < dim; i++) {
        d[i] = __half2float(xr[i]) * rms * __half2float(weight[i]);
    }
}
"#;

const RMSNORM_ADD: &str = r#"
extern "C" __global__ void rmsnorm_add(
    const __half* __restrict__ x,      // [rows, dim]
    const __half* __restrict__ weight, // [dim]
    float* __restrict__ dst,           // [rows, dim] read + write in-place
    int rows,
    int dim,
    float eps
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float ss = 0.0f;
    const __half* xr = x + row * dim;
    for (int i = 0; i < dim; i++) {
        float v = __half2float(xr[i]);
        ss += v * v;
    }
    ss /= (float)dim;
    float rms = 1.0f / sqrtf(ss + eps);
    float* d = dst + row * dim;
    for (int i = 0; i < dim; i++) {
        d[i] += __half2float(xr[i]) * rms * __half2float(weight[i]);
    }
}
"#;

const SOFTMAX: &str = r#"
extern "C" __global__ void softmax(
    const float* __restrict__ x, // [rows, dim]
    float* __restrict__ dst,     // [rows, dim]
    int rows,
    int dim,
    float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * dim;
    float* dr = dst + row * dim;
    // find max
    float m = xr[0] * scale;
    for (int i = 1; i < dim; i++) {
        float v = xr[i] * scale;
        if (v > m) m = v;
    }
    // exp sum
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float v = expf(xr[i] * scale - m);
        dr[i] = v;
        sum += v;
    }
    // normalize
    float inv = 1.0f / sum;
    for (int i = 0; i < dim; i++) {
        dr[i] *= inv;
    }
}
"#;

const LINEAR_F16: &str = r#"
extern "C" __global__ void linear_f16(
    const float* __restrict__ x,     // [m, in_f]
    const __half* __restrict__ w,    // [out_f, in_f] row-major
    float* __restrict__ dst,         // [m, out_f]
    int m,
    int in_f,
    int out_f
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    // Each thread handles 4 outputs via loop over out_f
    for (int o = tid; o < out_f; o += blockDim.x) {
        float acc = 0.0f;
        const float* xr = x + row * in_f;
        const __half* wr = w + o * in_f;
        for (int i = 0; i < in_f; i++) {
            acc += xr[i] * __half2float(wr[i]);
        }
        dst[row * out_f + o] = acc;
    }
}
"#;

const QK_NORM: &str = r#"
extern "C" __global__ void qk_norm(
    const float* __restrict__ x,       // [rows, n_head, head_dim] or strided
    const __half* __restrict__ weight, // [head_dim]
    float* __restrict__ dst,           // [rows, n_head, head_dim]
    int rows,
    int n_head,
    int head_dim,
    float eps,
    int x_stride       // per-row stride in elements; 0 = packed
) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = rows * n_head;
    if (head >= total_heads) return;
    int r = head / n_head;
    int h = head % n_head;
    int stride = (x_stride > 0) ? x_stride : (n_head * head_dim);
    int off = r * stride + h * head_dim;
    float ss = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float v = x[off + i];
        ss += v * v;
    }
    ss /= (float)head_dim;
    float rms = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < head_dim; i++) {
        dst[off + i] = x[off + i] * rms * __half2float(weight[i]);
    }
}
"#;

const ROPE: &str = r#"
extern "C" __global__ void rope(
    float* __restrict__ x,              // [rows, n_head, head_dim] — mutated in-place
    const int* __restrict__ positions,  // [rows]
    const float* __restrict__ freq_factors, // optional (null = unused)
    int rows,
    int n_head,
    int head_dim,
    float theta,
    int use_neox,
    float freq_base
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int pos = positions[row];
    float* xr = x + row * n_head * head_dim;
    int half = head_dim / 2;
    for (int h = 0; h < n_head; h++) {
        float* xh = xr + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
            if (freq_factors != nullptr) {
                freq *= freq_factors[i];
            }
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float x0 = xh[i];
            float x1 = xh[i + half];
            xh[i]       = x0 * c - x1 * s;
            xh[i + half] = x0 * s + x1 * c;
        }
    }
}
"#;

const QK_NORM_ROPE: &str = r#"
extern "C" __global__ void qk_norm_rope(
    float* __restrict__ x,              // [rows, n_head, head_dim] — mutated in-place
    const __half* __restrict__ weight,  // [head_dim]
    const int* __restrict__ positions,  // [rows]
    const float* __restrict__ freq_factors, // optional
    int rows,
    int n_head,
    int head_dim,
    float eps,
    float theta,
    int use_neox,
    float freq_base,
    int x_stride
) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = rows * n_head;
    if (head >= total_heads) return;
    int r = head / n_head;
    int h = head % n_head;
    int pos = positions[r];
    int stride = (x_stride > 0) ? x_stride : (n_head * head_dim);
    int off = r * stride + h * head_dim;
    // rmsnorm
    float ss = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float v = x[off + i];
        ss += v * v;
    }
    ss /= (float)head_dim;
    float rms = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < head_dim; i++) {
        x[off + i] = x[off + i] * rms * __half2float(weight[i]);
    }
    // rope
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
        if (freq_factors != nullptr) {
            freq *= freq_factors[i];
        }
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = x[off + i];
        float x1 = x[off + i + half];
        x[off + i]       = x0 * c - x1 * s;
        x[off + i + half] = x0 * s + x1 * c;
    }
}
"#;

const GATED_RMSNORM: &str = r#"
extern "C" __global__ void gated_rmsnorm(
    const float* __restrict__ x,        // [rows, n_head, head_dim]
    const __half* __restrict__ weight,  // [head_dim]
    const float* __restrict__ gate,     // [rows, n_head, head_dim]
    float* __restrict__ dst,            // [rows, n_head, head_dim]
    int rows,
    int n_head,
    int head_dim,
    float eps
) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = rows * n_head;
    if (head >= total_heads) return;
    int off = head * head_dim;
    // rmsnorm
    float ss = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float v = x[off + i];
        ss += v * v;
    }
    ss /= (float)head_dim;
    float rms = 1.0f / sqrtf(ss + eps);
    // gate (SiLU) multiply
    for (int i = 0; i < head_dim; i++) {
        float g = gate[off + i];
        float silu_g = g / (1.0f + expf(-g));
        dst[off + i] = x[off + i] * rms * __half2float(weight[i]) * silu_g;
    }
}
"#;

const GATED_ACT: &str = r#"
extern "C" __global__ void gated_act(
    const float* __restrict__ gate, // [rows, nff] or strided
    const float* __restrict__ up,   // [rows, nff] or strided
    float* __restrict__ dst,        // [rows, nff]
    int rows,
    int nff,
    int act_type,       // 0=SiLU, 1=GeLU(tanh), 2=Sigmoid
    int up_off,         // element offset into up
    int up_stride,      // 0 = packed
    int gate_stride,    // 0 = packed
    int gate_block_width // 0 = no interleave
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int effective_gate_stride = (gate_stride > 0) ? gate_stride : nff;
    int effective_up_stride = (up_stride > 0) ? up_stride : nff;
    int gate_off = row * effective_gate_stride;
    int up_off_base = up_off + row * effective_up_stride;
    for (int i = 0; i < nff; i++) {
        float g;
        if (gate_block_width > 0) {
            // Interleaved: block of (query then gate) repeating
            int block = i / gate_block_width;
            int offset_in_block = i % gate_block_width;
            int block_start = block * (gate_block_width * 2);
            g = gate[gate_off + block_start + gate_block_width + offset_in_block];
        } else {
            g = gate[gate_off + i];
        }
        float u = up[up_off_base + i];
        float a;
        if (act_type == 0) {
            // SiLU: x * sigmoid(x)
            a = g / (1.0f + expf(-g));
        } else if (act_type == 1) {
            // GeLU (tanh approx)
            float x3 = g * g * g;
            float c = 0.044715f;
            a = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + c * x3)));
        } else {
            // Sigmoid
            a = 1.0f / (1.0f + expf(-g));
        }
        dst[row * nff + i] = a * u;
    }
}
"#;

const ADD: &str = r#"
extern "C" __global__ void add(
    const float* __restrict__ a, // [n]
    const float* __restrict__ b, // [n]
    float* __restrict__ dst,     // [n]
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = a[i] + b[i];
}
"#;

const ADD_BIAS: &str = r#"
extern "C" __global__ void add_bias(
    const float* __restrict__ x,    // [rows, n]
    const float* __restrict__ bias, // [n]
    float* __restrict__ dst,        // [rows, n]
    int rows,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * n;
    float* dr = dst + row * n;
    for (int i = 0; i < n; i++) {
        dr[i] = xr[i] + bias[i];
    }
}
"#;

const SCALE: &str = r#"
extern "C" __global__ void scale(
    const float* __restrict__ x, // [n]
    float* __restrict__ dst,     // [n]
    float s,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = x[i] * s;
}
"#;

const MUL_VEC: &str = r#"
extern "C" __global__ void mul_vec(
    const float* __restrict__ x,   // [rows, n]
    const float* __restrict__ vec, // [n]
    float* __restrict__ dst,       // [rows, n]
    int rows,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * n;
    float* dr = dst + row * n;
    for (int i = 0; i < n; i++) {
        dr[i] = xr[i] * vec[i];
    }
}
"#;

const SOFTCAP: &str = r#"
extern "C" __global__ void softcap(
    const float* __restrict__ x, // [n]
    float* __restrict__ dst,     // [n]
    float cap,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = cap * tanhf(x[i] / cap);
}
"#;

const COPY: &str = r#"
extern "C" __global__ void copy(
    const float* __restrict__ src,
    int src_off,
    float* __restrict__ dst,
    int dst_off,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[dst_off + i] = src[src_off + i];
}
"#;

const COPY_STRIDED: &str = r#"
extern "C" __global__ void copy_strided(
    const float* __restrict__ src,
    int src_off,
    int src_stride,
    float* __restrict__ dst,
    int dst_off,
    int dst_stride,
    int rows,
    int n
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* sr = src + src_off + r * src_stride;
    float* dr = dst + dst_off + r * dst_stride;
    for (int i = 0; i < n; i++) {
        dr[i] = sr[i];
    }
}
"#;

const EMBED_GATHER: &str = r#"
extern "C" __global__ void embed_gather(
    const int* __restrict__ ids,     // [rows]
    const __half* __restrict__ table, // [vocab, dim]
    float* __restrict__ dst,          // [rows, dim]
    int rows,
    int dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int id = ids[row];
    const __half* tr = table + id * dim;
    float* dr = dst + row * dim;
    for (int i = 0; i < dim; i++) {
        dr[i] = __half2float(tr[i]);
    }
}
"#;

const ARGMAX: &str = r#"
extern "C" __global__ void argmax(
    const float* __restrict__ x, // [rows, n]
    float* __restrict__ dst,     // [rows] — u32 bit-pattern in f32 slot
    int rows,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * n;
    float best_val = xr[0];
    int best_idx = 0;
    for (int i = 1; i < n; i++) {
        if (xr[i] > best_val) {
            best_val = xr[i];
            best_idx = i;
        }
    }
    // Store u32 bit-pattern in an f32 slot (the runner reads as u32)
    dst[row] = __int_as_float(best_idx);
}
"#;

const WRITE_KV: &str = r#"
extern "C" __global__ void write_kv(
    const float* __restrict__ src,  // [rows, n_kv, head_dim] at row-stride
    __half* __restrict__ cache,     // [kv_len_max, n_kv, head_dim]
    int row_offset,                 // pos in cache to write to
    int rows,
    int n_kv,
    int head_dim,
    int src_stride                  // per-row stride in src (0 = packed)
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int effective_stride = (src_stride > 0) ? src_stride : (n_kv * head_dim);
    int cache_row = row_offset + row;
    const float* sr = src + row * effective_stride;
    __half* cr = cache + cache_row * n_kv * head_dim;
    for (int i = 0; i < n_kv * head_dim; i++) {
        cr[i] = __float2half(sr[i]);
    }
}
"#;

const ATTENTION: &str = r#"
extern "C" __global__ void attention(
    const float* __restrict__ q,       // [rows, n_head, head_dim]
    const __half* __restrict__ k_cache,// [kv_len, n_kv, head_dim]
    const __half* __restrict__ v_cache,// [kv_len, n_kv, head_dim]
    float* __restrict__ dst,           // [rows, n_head, head_dim]
    int rows,
    int kv_len,
    int n_head,
    int n_kv,
    int head_dim,
    float scale,
    int pos,            // absolute position of first query row
    int mask_type,      // 0=Causal, 1=SlidingWindow, 2=Canvas
    int swa_window      // window size for SlidingWindow
) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = rows * n_head;
    if (head >= total_heads) return;
    int r = head / n_head;
    int h = head % n_head;
    int kv_h = h * n_kv / n_head; // GQA head mapping
    int q_off = head * head_dim;

    // Two-pass online softmax: pass 1 finds max, pass 2 computes weighted sum
    float max_score = -1e30f;
    for (int j = 0; j < kv_len; j++) {
        const __half* kr = k_cache + j * n_kv * head_dim + kv_h * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            s += q[q_off + d] * __half2float(kr[d]);
        }
        s *= scale;
        // masking
        bool masked = false;
        if (mask_type == 0) {
            masked = (j > pos + r);
        } else if (mask_type == 1) {
            int q_pos = pos + r;
            masked = (j > q_pos || j < q_pos - swa_window + 1);
        } else if (mask_type == 2) {
            masked = (j < swa_window);
        }
        if (!masked && s > max_score) max_score = s;
    }
    // Pass 2: exp sum and weighted value sum
    float sum = 0.0f;
    float* dr = dst + q_off;
    for (int d = 0; d < head_dim; d++) { dr[d] = 0.0f; }
    for (int j = 0; j < kv_len; j++) {
        const __half* kr = k_cache + j * n_kv * head_dim + kv_h * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            s += q[q_off + d] * __half2float(kr[d]);
        }
        s *= scale;
        bool masked = false;
        if (mask_type == 0) {
            masked = (j > pos + r);
        } else if (mask_type == 1) {
            int q_pos = pos + r;
            masked = (j > q_pos || j < q_pos - swa_window + 1);
        } else if (mask_type == 2) {
            masked = (j < swa_window);
        }
        if (masked) continue;
        float w = expf(s - max_score);
        sum += w;
        const __half* vr = v_cache + j * n_kv * head_dim + kv_h * head_dim;
        for (int d = 0; d < head_dim; d++) { dr[d] += w * __half2float(vr[d]); }
    }
    float inv = 1.0f / sum;
    for (int d = 0; d < head_dim; d++) { dr[d] *= inv; }
}
"#;

const MOE_FFN: &str = r#"
// Host-side router for MoE — this kernel runs ONE expert's gated FFN on x.
// The router (softmax + top-k selection) is done on the HOST in the execute() walk.
extern "C" __global__ void moe_ffn_expert(
    const float* __restrict__ x,        // [ne] — input row
    const __half* __restrict__ gate_w,  // [n_ff_exp, ne] — expert's gate weight
    const __half* __restrict__ up_w,    // [n_ff_exp, ne] — expert's up weight
    const __half* __restrict__ down_w,  // [ne, n_ff_exp] — expert's down weight
    float* __restrict__ dst,            // [ne] — accumulated * weight
    int ne,
    int n_ff_exp,
    int act_type,   // 0=SiLU, 1=GeLU, 2=Sigmoid
    float weight,   // routing weight for this expert
    float down_scale // per-expert down-projection output scale (0 = no scale)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // gate: [n_ff_exp]
    if (i < (int)n_ff_exp) {
        // compute gate[i] and up[i]
        float g = 0.0f, u = 0.0f;
        for (int j = 0; j < (int)ne; j++) {
            g += x[j] * __half2float(gate_w[i * ne + j]);
            u += x[j] * __half2float(up_w[i * ne + j]);
        }
        // activation
        float a;
        if (act_type == 0) {
            a = g / (1.0f + expf(-g)); // SiLU
        } else if (act_type == 1) {
            float x3 = g * g * g;
            a = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * x3)));
        } else {
            a = 1.0f / (1.0f + expf(-g));
        }
        float h = a * u * weight;
        // down projection: accumulate into dst
        const __half* dr = down_w + i; // column i of down_w (row-major: down_w[*, i])
        // Actually down_w is [ne, n_ff_exp], so column i has stride n_ff_exp
        for (int d = 0; d < (int)ne; d++) {
            atomicAdd(&dst[d], h * __half2float(down_w[d * n_ff_exp + i]));
        }
    }
}
"#;

const CONV1D_SILU: &str = r#"
extern "C" __global__ void conv1d_silu(
    const float* __restrict__ x,       // [rows, channels]
    const __half* __restrict__ weight, // [channels, kernel]
    float* __restrict__ state,         // [(kernel-1), channels] — mutated in-place
    float* __restrict__ dst,           // [rows, channels]
    int rows,
    int channels,
    int kernel
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    for (int c = 0; c < channels; c++) {
        float acc = 0.0f;
        const __half* wc = weight + c * kernel;
        // history contributions
        for (int j = 0; j < (int)(kernel - 1); j++) {
            acc += state[j * channels + c] * __half2float(wc[j]);
        }
        // current token contribution
        acc += x[row * channels + c] * __half2float(wc[kernel - 1]);
        // SiLU
        float v = acc / (1.0f + expf(-acc));
        dst[row * channels + c] = v;
    }
}
// State update is done HOST-SIDE in execute() after the kernel: shift history,
// append current x row.
"#;

const DELTANET: &str = r#"
extern "C" __global__ void deltanet(
    const float* __restrict__ q,         // [rows, n_khead * head_k]
    const float* __restrict__ k,         // [rows, n_khead * head_k]
    const float* __restrict__ v,         // [rows, n_vhead * head_v]
    const float* __restrict__ b,         // [rows, n_vhead]
    const float* __restrict__ a,         // [rows, n_vhead]
    const __half* __restrict__ a_coef,   // [n_vhead]
    const __half* __restrict__ dt_bias,  // [n_vhead]
    float* __restrict__ state,           // [n_vhead, head_k, head_v] — mutated in-place
    float* __restrict__ dst,             // [rows, n_vhead * head_v]
    int rows,
    int n_khead,
    int n_vhead,
    int head_k,
    int head_v
) {
    // Process one value head at a time
    int vh = blockIdx.x * blockDim.x + threadIdx.x;
    if (vh >= (int)n_vhead) return;
    // GQA: value head vh uses q/k head vh / (n_vhead / n_khead)
    int kh = vh / (n_vhead / n_khead);
    float ac = __half2float(a_coef[vh]);
    float dtb = __half2float(dt_bias[vh]);
    for (int r = 0; r < rows; r++) {
        // q, k rows
        const float* qr = q + r * n_khead * head_k + kh * head_k;
        const float* kr = k + r * n_khead * head_k + kh * head_k;
        const float* vr = v + r * n_vhead * head_v + vh * head_v;
        float br = b[r * n_vhead + vh];
        float ar = a[r * n_vhead + vh];
        // L2 norm q, k
        float qn = 0.0f, kn = 0.0f;
        for (int i = 0; i < (int)head_k; i++) { qn += qr[i] * qr[i]; kn += kr[i] * kr[i]; }
        qn = 1.0f / sqrtf(qn + 1e-6f);
        kn = 1.0f / sqrtf(kn + 1e-6f);
        // beta, decay
        float beta = 1.0f / (1.0f + expf(-br));
        float decay = expf(ac * logf(1.0f + expf(ar + dtb)));
        // state *= decay (naive: O(head_k * head_v) per head per row)
        float* Sv = state + vh * head_k * head_v;
        for (int i = 0; i < (int)(head_k * head_v); i++) {
            Sv[i] *= decay;
        }
        // compute delta
        // S^T k: (head_v x head_k) * (head_k) = (head_v)
        float sk[256]; // max head_v (typically 128)
        for (int j = 0; j < (int)head_v; j++) {
            float s = 0.0f;
            for (int i = 0; i < (int)head_k; i++) {
                s += Sv[j * head_k + i] * kr[i] * kn;
            }
            sk[j] = s;
        }
        // delta = (v_norm - sk) * beta
        float delta[256];
        float vnorm = 0.0f;
        for (int j = 0; j < (int)head_v; j++) {
            delta[j] = (vr[j] - sk[j]) * beta;
        }
        // S += k_norm ⊗ delta: (head_k) * (head_v) outer product
        for (int i = 0; i < (int)head_k; i++) {
            float ki = kr[i] * kn;
            for (int j = 0; j < (int)head_v; j++) {
                Sv[j * head_k + i] += ki * delta[j];
            }
        }
        // dst = S^T q_norm
        float qscale = 1.0f / sqrtf((float)head_k);
        float* dr = dst + r * n_vhead * head_v + vh * head_v;
        for (int j = 0; j < (int)head_v; j++) {
            float s = 0.0f;
            for (int i = 0; i < (int)head_k; i++) {
                s += Sv[j * head_k + i] * qr[i] * qn * qscale;
            }
            dr[j] = s;
        }
    }
}
"#;

const MOE_SHARED_EXPERT_ADD: &str = r#"
extern "C" __global__ void moe_shared_expert_add(
    const float* __restrict__ moe,    // [rows, n]
    const float* __restrict__ shexp,   // [rows, n]
    const float* __restrict__ gate,     // [rows] — pre-sigmoid per-row gate
    float* __restrict__ dst,           // [rows, n]
    int rows,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float g = 1.0f / (1.0f + expf(-gate[row]));
    const float* mr = moe + row * n;
    const float* sr = shexp + row * n;
    float* dr = dst + row * n;
    for (int i = 0; i < n; i++) {
        dr[i] = mr[i] + g * sr[i];
    }
}
"#;

// ── Module cache ─────────────────────────────────────────────────────────────

/// Compiled HIP module + kernel-function cache.
pub struct Pipelines {
    module: ffi::hipModule_t,
    /// Kernel name → function handle (lazily fetched).
    cache: Mutex<HashMap<&'static str, ffi::hipFunction_t>>,
}

unsafe impl Send for Pipelines {}
unsafe impl Sync for Pipelines {}

impl Pipelines {
    /// Compile the assembled HIP source via hiprtc and load the resulting module.
    pub fn build(device: c_int) -> Result<Self> {
        let src = hip_source();
        let csrc = CString::new(src).map_err(|e| be(format!("kernel source NUL-byte: {e}")))?;
        let mut prog: ffi::hiprtcProgram = std::ptr::null_mut();
        let name_cstr = CString::new("infr_kernels").unwrap();
        let rc = unsafe {
            ffi::hiprtcCreateProgram(
                &mut prog,
                csrc.as_ptr(),
                name_cstr.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if rc != ffi::HIPRTC_SUCCESS {
            return Err(be(format!("hiprtcCreateProgram: rc={rc}")));
        }

        // Compile for the current device's arch. Default flags: f16 support + fast math.
        let gfx_arch = {
            let mut props: ffi::hipDeviceProp_t = unsafe { std::mem::zeroed() };
            unsafe { ffi::hipGetDeviceProperties(&mut props, device) };
            let name_bytes: Vec<u8> = props
                .gcn_arch_name
                .iter()
                .take_while(|b| **b != 0)
                .map(|b| *b as u8)
                .collect();
            String::from_utf8_lossy(&name_bytes).to_string()
        };
        let arch_flag = format!("--gpu-architecture={gfx_arch}");
        let arch_c = CString::new(arch_flag.as_str()).unwrap();
        let std_flag = CString::new("-std=c++17").unwrap();
        let opts: [*const c_char; 2] = [arch_c.as_ptr(), std_flag.as_ptr()];
        let rc = unsafe { ffi::hiprtcCompileProgram(prog, opts.len() as i32, opts.as_ptr()) };
        if rc != ffi::HIPRTC_SUCCESS {
            // Fetch the compile log for diagnostics
            let mut log_size: usize = 0;
            unsafe { ffi::hiprtcGetProgramLogSize(prog, &mut log_size) };
            let mut log_buf: Vec<u8> = vec![0u8; log_size];
            unsafe { ffi::hiprtcGetProgramLog(prog, log_buf.as_mut_ptr() as *mut c_char) };
            let log = String::from_utf8_lossy(&log_buf);
            unsafe { ffi::hiprtcDestroyProgram(&mut prog) };
            return Err(be(format!("hiprtcCompileProgram failed (rc={rc}):\n{log}")));
        }

        // Get compiled code
        let mut code_size: usize = 0;
        let rc = unsafe { ffi::hiprtcGetCodeSize(prog, &mut code_size) };
        if rc != ffi::HIPRTC_SUCCESS {
            unsafe { ffi::hiprtcDestroyProgram(&mut prog) };
            return Err(be(format!("hiprtcGetCodeSize: rc={rc}")));
        }
        let mut code: Vec<u8> = vec![0u8; code_size];
        let rc = unsafe { ffi::hiprtcGetCode(prog, code.as_mut_ptr() as *mut c_char) };
        if rc != ffi::HIPRTC_SUCCESS {
            unsafe { ffi::hiprtcDestroyProgram(&mut prog) };
            return Err(be(format!("hiprtcGetCode: rc={rc}")));
        }
        unsafe { ffi::hiprtcDestroyProgram(&mut prog) };

        // Load the code object into a module
        let mut module: ffi::hipModule_t = std::ptr::null_mut();
        let rc = unsafe {
            ffi::hipModuleLoadData(&mut module, code.as_ptr() as *const std::ffi::c_void)
        };
        if rc != ffi::HIP_SUCCESS {
            return Err(be(format!("hipModuleLoadData: rc={rc}")));
        }

        Ok(Self {
            module,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Get (creating + caching on first use) the kernel function for a given name.
    pub fn get(&self, name: &'static str) -> Result<ffi::hipFunction_t> {
        if let Some(f) = self.cache.lock().unwrap().get(name) {
            return Ok(*f);
        }
        let cname = CString::new(name).map_err(|e| be(format!("kernel name NUL-byte: {e}")))?;
        let mut func: ffi::hipFunction_t = std::ptr::null_mut();
        let rc = unsafe { ffi::hipModuleGetFunction(&mut func, self.module, cname.as_ptr()) };
        if rc != ffi::HIP_SUCCESS {
            return Err(be(format!("hipModuleGetFunction({name}): rc={rc}")));
        }
        self.cache.lock().unwrap().insert(name, func);
        Ok(func)
    }
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        // hipModuleDestroy doesn't exist in public API; the module leaks on drop.
        // This is fine for a single-backend-instance lifetime.
    }
}
