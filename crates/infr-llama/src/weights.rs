//! GPU weight-storage types: a projection weight (f16 / native-quant blocks), MoE expert
//! weights (GPU-resident or host-backed), and a layer FFN (dense gate||up + down, or a MoE
//! bank). Mechanically split out of `lib.rs` (no logic change).
use crate::{dequant_block, f16_bytes, f32_to_f16_sat};
use anyhow::{anyhow, Result};
use infr_core::backend::Buffer;
use infr_core::WeightSource;
use infr_gguf::Gguf;
use infr_vulkan::VulkanBackend;

/// A projection weight on the GPU: f16, unified repacked quant, or native raw-block quant.
///
/// - `F16`: f16 weight buffer (float or codebook-quant host-dequanted → f16)
/// - `Q`: unified repacked affine quant (q/s/m buffers, `dq = s·u8 + m`); fallback when native is
///   disabled (`INFR_NONATIVE=1`) or for grid/codebook quants under `INFR_NATIVE=1`.
/// - `Native`: raw GGUF block bytes, padded to u32 alignment, dequantized in-shader (decode-once
///   GEMV + tiled coopmat GEMM). The DEFAULT for optimized affine quants — faster decode + prefill
///   and smaller VRAM (see [`is_native_default`]); `INFR_NATIVE=1` extends it to all formats.
pub(crate) enum Wt {
    F16(Box<dyn Buffer>),
    /// Raw native-block bytes on the GPU; `dtype` identifies the dequant shader.
    Native {
        buf: Box<dyn Buffer>,
        dtype: infr_core::DType,
    },
}
impl Wt {
    /// The f16 buffer (panics if quantized — used by the llama fused path, which is f16-only).
    pub(crate) fn f16(&self) -> &dyn Buffer {
        match self {
            Wt::F16(b) => b.as_ref(),
            Wt::Native { .. } => {
                panic!("expected f16 weight, got native quant (llama fused path needs f16)")
            }
        }
    }
}

/// One expert weight: resident on the GPU (`Gpu`) or host-backed (`Cpu`) — for host-backed experts
/// the bytes are read on demand from the kept-alive GGUF mmap (no host-RAM copy), then computed on
/// the CPU or streamed to a VRAM pool (`INFR_NCMOE` / `INFR_MOE_STREAM`, cf. `--n-cpu-moe`).
pub(crate) enum ExpertW {
    Gpu(Wt),
    Cpu { dtype: infr_core::DType },
}
impl ExpertW {
    pub(crate) fn is_cpu(&self) -> bool {
        matches!(self, ExpertW::Cpu { .. })
    }
    /// The GPU weight (panics for host experts — callers branch on [`is_cpu`] first).
    pub(crate) fn gpu(&self) -> &Wt {
        match self {
            ExpertW::Gpu(w) => w,
            ExpertW::Cpu { .. } => panic!("expected GPU expert"),
        }
    }
}

/// One routed expert's SwiGLU weights (gate/up [n_embd→n_ff_exp], down [n_ff_exp→n_embd]).
pub(crate) struct ExpertWt {
    pub(crate) gate: ExpertW,
    pub(crate) up: ExpertW,
    pub(crate) down: ExpertW,
}

/// Stacked GPU expert bank: one Native buffer per role holding all experts contiguously, addressed
/// by element offset (`expert_id * stride`). Lets the GPU-resident decode/prefill dispatch every
/// expert from a single buffer (so an on-GPU router id can pick the expert — no host round-trip).
/// Built only for fully-GPU, native-quant, non-offloaded models; offloaded models keep `experts`.
pub(crate) struct MoeStacked {
    pub(crate) gate: Wt,
    pub(crate) up: Wt,
    pub(crate) down: Wt,
    pub(crate) stride: usize, // elements per expert (n_ff_exp * n_embd), identical for gate/up/down
}

/// A layer's FFN: dense fused gate‖up + down, or a routed MoE bank (router + per-expert weights).
pub(crate) enum FfnWt {
    Dense {
        wgateup: Wt,
        wdown: Wt,
    },
    Moe {
        gate_inp: Wt,
        experts: Vec<ExpertWt>, // empty when `stacked` is Some (per-expert buffers dropped)
        stacked: Option<MoeStacked>,
    },
}
pub(crate) fn is_native_default(d: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(
        d,
        Q8_0 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q2K | Q3K | Q4K | Q5K | Q6K
    )
}

/// Upload a projection weight, keeping quantized weights quantized in-VRAM (else convert to f16).
///
/// - Affine quants (Q4_K/Q5_K/Q6_K/Q8_0/Q4_0…) → `Wt::Native` (raw block bytes, in-shader
///   decode-once dequant — faster decode + prefill, smaller VRAM). These have the `native_id_*`
///   decode GEMV shaders ([`is_native_default`]).
/// - Codebook quants (IQ*/TQ*/fp4) and float types (F16/F32/BF16) → host dequant → f16 → `Wt::F16`.
///   The i-quants have no decode-GEMV shader yet, so they stay on f16 until those land.
pub(crate) fn upload_wt(be: &VulkanBackend, g: &Gguf, name: &str) -> Result<Wt> {
    let dtype = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor not found: {name}"))?
        .dtype;
    let bytes = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
    upload_wt_bytes(be, dtype, bytes)
}

/// Like [`upload_wt`] but from a raw byte slice + dtype — lets a stacked MoE expert tensor be sliced
/// per expert (each expert is a contiguous block of the `*_exps` tensor) and uploaded individually.
pub(crate) fn upload_wt_bytes(
    be: &VulkanBackend,
    dtype: infr_core::DType,
    bytes: &[u8],
) -> Result<Wt> {
    // Native-block path: raw upload + in-shader dequant — for every quant format with the dense
    // native pipeline (decode GEMV + prefill GEMM; see `native_dense_supported`). Only float types
    // (F16/F32/BF16, not quants) fall to the host dequant → f16 path.
    if infr_vulkan::linear::native_dense_supported(dtype) {
        let padded = infr_vulkan::linear::pad_to_u32_align(bytes);
        return Ok(Wt::Native {
            buf: be
                .upload_weight_bytes(&padded)
                .map_err(|e| anyhow!("native upload: {e}"))?,
            dtype,
        });
    }
    // Float types → host dequant to f32 → f16.
    let f16_bytes: Vec<u8> = dequant_block(dtype, bytes)?
        .iter()
        .flat_map(|&v| f32_to_f16_sat(v).to_bits().to_le_bytes())
        .collect();
    Ok(Wt::F16(be.upload_weight_bytes(&f16_bytes)?))
}

/// Build a dense layer's fused gate‖up weight (`[2*n_ff, n_embd]`, gate rows then up rows). Quant
/// stays quantized (raw native blocks); float → f16. `prefix` is e.g. `"blk.3."`.
pub(crate) fn build_wgateup(be: &VulkanBackend, g: &Gguf, prefix: &str) -> Result<Wt> {
    let gate_name = format!("{prefix}ffn_gate.weight");
    let up_name = format!("{prefix}ffn_up.weight");
    let gate_dtype = g
        .tensors()
        .iter()
        .find(|t| t.name == gate_name)
        .map(|t| t.dtype);
    let gb = g.tensor_bytes(&gate_name).map_err(|e| anyhow!("{e}"))?;
    let ub = g.tensor_bytes(&up_name).map_err(|e| anyhow!("{e}"))?;
    // Native path (every quant format): the fused `[2*n_ff, n_embd]` weight is just gate's rows
    // followed by up's rows, and each tensor's bytes are already row-contiguous native blocks — so
    // concatenating the raw bytes IS the fused weight (up's first block lands exactly at `gb.len()`,
    // no inter-tensor padding). Raw upload, in-shader dequant — no host dequant/repack.
    if let Some(dt) = gate_dtype {
        if infr_vulkan::linear::native_dense_supported(dt) {
            let mut fused = gb.to_vec();
            fused.extend_from_slice(ub);
            let padded = infr_vulkan::linear::pad_to_u32_align(&fused);
            return Ok(Wt::Native {
                buf: be
                    .upload_weight_bytes(&padded)
                    .map_err(|e| anyhow!("native gateup upload: {e}"))?,
                dtype: dt,
            });
        }
    }
    // Float gate/up → f16.
    let mut gateup = f16_bytes(g, &gate_name)?;
    gateup.extend_from_slice(&f16_bytes(g, &up_name)?);
    Ok(Wt::F16(be.upload_weight_bytes(&gateup)?))
}
/// Dispatch a [`Wt`] linear (`y = x·Wᵀ`) into a recorder, picking the f16 / quant / native op.
/// The (dtype, buffer) of a Native weight — for the stacked MoE expert dispatch (native-only).
pub(crate) fn native_parts(w: &Wt) -> (infr_core::DType, &dyn Buffer) {
    match w {
        Wt::Native { buf, dtype } => (*dtype, buf.as_ref()),
        _ => unreachable!("stacked MoE experts are native-only"),
    }
}

/// Prefill projection (`y = X·Wᵀ`, X = [m,in_f], m≥64): tiled coopmat GEMM for native-quant weights
/// (decode-once, reused across the 64-row tile) instead of the per-row GEMV that re-reads the weight
/// m times. `y` is allocated `ceil(m/64)*64` rows. Non-native weights (the small f16 router) fall
/// back to the GEMV.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rec_proj(
    rec: &infr_vulkan::Recorder,
    w: &Wt,
    x: &dyn Buffer,
    y: &dyn Buffer,
    m: usize,
    in_f: usize,
    out_f: usize,
) {
    match w {
        Wt::Native { buf, dtype } => rec.matmul_native(*dtype, x, buf.as_ref(), y, m, in_f, out_f),
        _ => rec_linear(rec, w, x, y, m, in_f, out_f),
    }
}

/// Dispatch a stacked MoE expert as a tiled coopmat GEMM (`y = X·W_eᵀ`, X = [m,in_f]): the weight is
/// `expert*stride` elements into the stacked Native buffer, decoded ONCE and reused across the 64-row
/// tile (vs the per-row GEMV re-read). `y` is allocated `ceil(m/64)*64` rows. Native-only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rec_gemm_expert(
    rec: &infr_vulkan::Recorder,
    w: &Wt,
    expert: usize,
    stride: usize,
    x: &dyn Buffer,
    y: &dyn Buffer,
    m: usize,
    in_f: usize,
    out_f: usize,
) {
    let (dtype, buf) = native_parts(w);
    rec.matmul_native_off(dtype, x, buf, expert * stride, y, m, in_f, out_f);
}

/// Dispatch a stacked MoE expert's linear (`y = x·W_eᵀ`): the weight is `expert * stride` elements
/// into the role's stacked Native buffer. Stacked experts are native-only (see [`load_moe`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rec_linear_expert(
    rec: &infr_vulkan::Recorder,
    w: &Wt,
    expert: usize,
    stride: usize,
    x: &dyn Buffer,
    y: &dyn Buffer,
    rows: usize,
    in_f: usize,
    out_f: usize,
) {
    match w {
        Wt::Native { buf, dtype } => rec.linear_native_off(
            *dtype,
            buf.as_ref(),
            expert * stride,
            x,
            y,
            rows,
            in_f,
            out_f,
        ),
        _ => unreachable!("stacked MoE experts are native-only"),
    }
}

pub(crate) fn rec_linear(
    rec: &infr_vulkan::Recorder,
    w: &Wt,
    x: &dyn Buffer,
    y: &dyn Buffer,
    rows: usize,
    in_f: usize,
    out_f: usize,
) {
    match w {
        Wt::F16(b) => rec.linear(b.as_ref(), x, y, rows, in_f, out_f),
        Wt::Native { buf, dtype } => {
            rec.linear_native(*dtype, buf.as_ref(), x, y, rows, in_f, out_f)
        }
    }
}

/// `y = x·Wᵀ + residual` (fused-residual GEMV), dispatching on how `W` is stored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rec_linear_add(
    rec: &infr_vulkan::Recorder,
    w: &Wt,
    x: &dyn Buffer,
    residual: &dyn Buffer,
    y: &dyn Buffer,
    rows: usize,
    in_f: usize,
    out_f: usize,
) {
    match w {
        Wt::F16(b) => rec.linear_add(b.as_ref(), x, residual, y, rows, in_f, out_f),
        Wt::Native { buf, dtype } => {
            rec.linear_add_native(*dtype, buf.as_ref(), x, residual, y, rows, in_f, out_f)
        }
    }
}
