//! Reconstruct one full EXL3 weight matrix from a real checkpoint through the CPU
//! oracle: trellis tiles -> [`decode_matrix`] (with [`tile_perm`]) -> scales.
//!
//! This is the Phase-1 layout exercised on REAL quantized bytes: the tile values
//! land in matrix positions via the transliterated `reconstruct_tile` shuffle,
//! then `suh`/`svh` fold in. Every value must be finite with a sane spread, or
//! the permutation or the scale folding misreads the checkpoint layout.
//!
//! Run: `cargo run -p infr-cpu --example exl3_matrix -- <model.safetensors> <prefix>`
//! e.g. `... model.safetensors model.layers.0.self_attn.v_proj`

use infr_core::WeightSource;
use infr_cpu::exl3::{apply_scales, decode_matrix, Codebook};
use infr_safetensors::Safetensors;

fn f16le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| half::f16::from_le_bytes(*p).to_f32())
        .collect()
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: exl3_matrix <model.safetensors> <prefix>");
    let prefix = std::env::args()
        .nth(2)
        .expect("usage: exl3_matrix <model.safetensors> <prefix>");
    let st = Safetensors::open(std::path::Path::new(&path))?;

    let trellis_key = format!("{prefix}.trellis");
    let info = st
        .tensors()
        .iter()
        .find(|t| t.name == trellis_key)
        .unwrap_or_else(|| panic!("no tensor {trellis_key}"));
    assert_eq!(info.shape.len(), 3);
    let (kti, nt, wtile) = (info.shape[0], info.shape[1], info.shape[2]);
    let (k, n) = (kti * 16, nt * 16);
    // Integer rates only (Phase 0 scope); +8 half-integer mul1 tiles unsupported.
    assert_eq!(wtile % 16, 0, "fractional-K tiles are Phase-1 scope");
    let bits = wtile as u32 / 16;
    // Released checkpoints are all cb0 (no .mcg/.mul1 markers); refuse to guess.
    let has_mcg = st
        .tensors()
        .iter()
        .any(|t| t.name == format!("{prefix}.mcg"));
    let has_mul1 = st
        .tensors()
        .iter()
        .any(|t| t.name == format!("{prefix}.mul1"));
    assert!(
        !has_mcg && !has_mul1,
        "{prefix}: mcg/mul1 codebooks are later scope"
    );
    let cb = Codebook::Cb0;

    let trellis: Vec<u16> = st
        .tensor_bytes(&trellis_key)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| u16::from_le_bytes(*p))
        .collect();
    let suh = f16le_to_f32(st.tensor_bytes(&format!("{prefix}.suh"))?);
    let svh = f16le_to_f32(st.tensor_bytes(&format!("{prefix}.svh"))?);

    let mut mat = decode_matrix(cb, bits, &trellis, k, n);
    let (mut dmin, mut dmax, mut dsum) = (f32::INFINITY, f32::NEG_INFINITY, 0f64);
    let mut dn = 0u64;
    for v in &mat {
        if !v.is_finite() {
            anyhow::bail!("non-finite value in decoded D matrix for {prefix}");
        }
        dmin = dmin.min(*v);
        dmax = dmax.max(*v);
        dsum += *v as f64;
        dn += 1;
    }
    apply_scales(&mut mat, &suh, &svh, k, n);
    let (mut smin, mut smax, mut ssum) = (f32::INFINITY, f32::NEG_INFINITY, 0f64);
    for v in &mat {
        if !v.is_finite() {
            anyhow::bail!("non-finite value in scaled S matrix for {prefix}");
        }
        smin = smin.min(*v);
        smax = smax.max(*v);
        ssum += *v as f64;
    }
    println!("{prefix}: K={bits} cb=cb0 shape=({k},{n})");
    println!(
        "  D: mean={:.5} min={dmin:.4} max={dmax:.4} ({} values)",
        dsum / dn as f64,
        dn
    );
    println!(
        "  S: mean={:.5} min={smin:.4} max={smax:.4}",
        ssum / dn as f64
    );
    println!("  S[0][..8] = {:?}", &mat[..8]);
    Ok(())
}
