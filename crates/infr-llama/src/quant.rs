//! Host-side weight loading: GGUF tensor → dequantized f32 or f16 bytes.
//! Pure dequant kernels have moved to `infr_gguf::dequant`; re-exported here
//! so all existing `crate::dequant_block` / `crate::rdf16` call-sites continue to work.
use anyhow::{anyhow, bail, Context, Result};
use infr_core::WeightSource;
use infr_gguf::Gguf;

// Re-export from infr_gguf::dequant so existing callers inside infr-llama resolve via `crate::`.
// Non-test code uses dequant_block and f32_to_f16_sat; the rest are only needed in tests.
pub(crate) use infr_gguf::dequant::{dequant_block, f32_to_f16_sat};

// Test-only re-exports: transformer.rs `gpu_affine_tests` use these via `use crate::*;`.
#[cfg(test)]
pub(crate) use infr_gguf::dequant::{dequant_codebook, dequant_unified};

/// Load a named tensor and dequantize it to host f32, returning (data, shape in GGUF ne order
/// `[in, out]`). The host/CPU-side dequant path — it does NOT load the bulk projection weights for
/// the GPU (those upload quantized/f16 in-VRAM). It feeds: the host embedding gather, the CPU norm
/// and SSM recurrence math (qwen35), the `Q35_CPU=1` oracle, and serves as the f32 source we
/// convert into f16/bf16/quant GPU weights. Survives even with full GPU format coverage.
pub(crate) fn load_tensor_dequant(g: &Gguf, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let info = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor not found: {name}"))?
        .clone();
    let bytes = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
    let v = dequant_block(info.dtype, bytes).with_context(|| format!("tensor {name}"))?;
    Ok((v, info.shape))
}

/// Return a tensor's data as raw f16 bytes (little-endian u16). F16 tensors pass through with no
/// conversion (fast path for f16 GGUFs); F32 tensors are converted on the host.
pub(crate) fn f16_bytes(g: &Gguf, name: &str) -> Result<Vec<u8>> {
    let info = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor not found: {name}"))?
        .clone();
    let bytes = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
    match info.dtype {
        infr_core::DType::F16 => Ok(bytes.to_vec()),
        infr_core::DType::F32 => {
            let f16: Vec<u16> = bytemuck::cast_slice::<u8, f32>(bytes)
                .iter()
                .map(|&x| f32_to_f16_sat(x).to_bits())
                .collect();
            Ok(bytemuck::cast_slice(&f16).to_vec())
        }
        infr_core::DType::Bf16 => {
            // bf16 → f32 → f16 (bf16 is the top 16 bits of f32). f16 has MORE mantissa than bf16
            // (10 vs 7 bits) but a far smaller exponent range, so the only loss is overflow — which
            // saturates (see `f32_to_f16_sat`) instead of becoming inf. Values that fit (≈all) are
            // exact. A native-bf16 fused path (no clip at all) is a follow-on; the eager qwen35 path
            // already stores bf16 natively via `upload_weight_bf16`.
            let f16: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| {
                    let f = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                    f32_to_f16_sat(f).to_bits()
                })
                .collect();
            Ok(bytemuck::cast_slice(&f16).to_vec())
        }
        other => bail!("unsupported dtype {other:?} for {name} (bring-up wants F16/F32/BF16)"),
    }
}
