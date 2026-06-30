//! GPU weight-storage types: a projection weight (f16 / native-quant blocks), MoE expert
//! weights (GPU-resident or host-backed), and a layer FFN (dense gate||up + down, or a MoE
//! bank). Mechanically split out of `lib.rs` (no logic change).
use infr_core::backend::Buffer;

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
