//! Bitwise parity for the Q4K warp GEMM A_GLOBAL pairs (n128_ag and sk_ag, resident vs
//! `-DSTREAMED` twin) — the prefill-hot tile variants weight_addr_parity.rs doesn't cover.
//! Born as the resident-BDA perf campaign's ISA probe and kept for the coverage; it still doubles
//! as the ISA-dump vehicle: RADV_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true <bin> --ignored
//! 2> isa.txt (move ~/.cache/infr/vk-pipeline-cache-*.bin aside first).
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

/// Relative L2 bound for a warp GEMM against the f64 host GEMM over the same dequantized weight.
/// The kernels round the weight to f16 and accumulate in a different order; a mis-decoded or
/// misaddressed tile lands at order one.
const HOST_REL_L2: f64 = 1e-3;

fn synth_bytes(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7;
            (h % 0x40) as u8
        })
        .collect()
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn warp_ag_isa_probe() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    if !be.capabilities().f16_coopmat() {
        eprintln!("skip: no 16x16x16 f16 coopmat on this device");
        return;
    }
    let dtype = DType::Q4K;
    let (m, k, n, splits) = (64usize, 512usize, 256usize, 2usize);
    let mpad = 64usize;
    let w_bytes = n * k / 256 * 144; // Q4K: 256 elems / 144 bytes

    let w = synth_bytes(w_bytes, 7);
    let a16 = synth_bytes(mpad * k * 2, 13); // f16 bits, high byte < 0x40 => finite

    let a_buf = be.alloc(a16.len(), BufferUsage::Activations).unwrap();
    be.upload(a_buf.as_ref(), &a16).unwrap();
    let c_buf = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();
    let part_buf = be
        .alloc(splits * mpad * n * 4, BufferUsage::Activations)
        .unwrap();

    // OFFSET-INVARIANCE (weights are u64 BDA-addressed only — no resident SSBO leg to compare):
    // leg A reads the weight at arena offset 0, leg B reads the SAME weight parked at a NON-ZERO
    // offset inside a bigger arena behind a garbage prefix. Both dispatch the `-DSTREAMED` twin and
    // read the weight by 64-bit address; a kernel that dropped the arena base offset would read the
    // garbage prefix and diverge — the mutation the offset leg exists to kill.
    let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
    be.upload(arena0.as_ref(), &w).unwrap();
    // Non-zero offset leg: a 256-aligned garbage prefix (arena blocks are 256-aligned, so the
    // weight base stays aligned for the kernel's uvec4 loads), then the identical weight bytes.
    let off = 256usize;
    let mut backing = synth_bytes(off, 99);
    backing.extend_from_slice(&w);
    let (arena1, base1) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena1.as_ref(), &backing).unwrap();
    let addr1 = base1 + off as u64;

    let mut outs: Vec<Vec<u8>> = Vec::new();
    // n128_ag @0, n128_ag @nonzero-offset, sk_ag @0, sk_ag @nonzero-offset
    for (sk, addr) in [(false, addr0), (false, addr1), (true, addr0), (true, addr1)] {
        let rec = be.recorder().unwrap();
        if sk {
            rec.matmul_native_splitk(
                dtype,
                a_buf.as_ref(),
                addr,
                0,
                part_buf.as_ref(),
                c_buf.as_ref(),
                m,
                k,
                n,
                splits,
                true,
            );
        } else {
            rec.matmul_native_f16a(dtype, a_buf.as_ref(), addr, 0, c_buf.as_ref(), m, k, n);
        }
        rec.finish().unwrap();
        let mut out = vec![0u8; m * n * 4];
        be.download(c_buf.as_ref(), &mut out).unwrap();
        assert!(
            out.iter().any(|&b| b != 0),
            "all-zero output (sk={sk} addr={addr:#x})"
        );
        outs.push(out);
    }
    assert_eq!(outs[0], outs[1], "n128_ag @nonzero-offset != @0 (bitwise)");
    assert_eq!(outs[2], outs[3], "sk_ag @nonzero-offset != @0 (bitwise)");

    // The legs above only compare the kernel with itself, so a kernel wrong the same way at every
    // offset passes them. Hold each tile variant to a host GEMM over the dequantized weight: the
    // kernels differ from it only by rounding the weight to f16 and the accumulation order.
    let w_host = infr_gguf::dequant::dequant_block(dtype, &w).unwrap();
    let a_host: Vec<f32> = a16
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::f16::from_le_bytes(*b).to_f32())
        .collect();
    let want: Vec<f64> = (0..m * n)
        .map(|i| {
            let (row, col) = (i / n, i % n);
            (0..k)
                .map(|j| f64::from(a_host[row * k + j]) * f64::from(w_host[col * k + j]))
                .sum()
        })
        .collect();
    for (name, out) in [("n128_ag", &outs[0]), ("sk_ag", &outs[2])] {
        let got: Vec<f32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let (num, den) = got
            .iter()
            .zip(&want)
            .fold((0f64, 0f64), |(err, mag), (&g, &r)| {
                (err + (f64::from(g) - r).powi(2), mag + r * r)
            });
        let rel = (num / den).sqrt();
        assert!(
            rel < HOST_REL_L2,
            "{name}: relative L2 error {rel:e} against the host GEMM"
        );
    }
    println!("ok: q4k n128_ag + sk_ag streamed offset-invariant (0 vs nonzero)");
}
