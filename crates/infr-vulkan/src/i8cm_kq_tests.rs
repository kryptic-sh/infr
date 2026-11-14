//! Kernel-level parity for the int8 cooperative-matrix K-quant prefill GEMM
//! (`native_gemm_i8cm_kq.comp`, [`crate::recorder::Recorder::matmul_i8cm_kq`]).
//!
//! The reference is the kernel's own decomposition evaluated in f64 on the host, from the GPU's
//! downloaded int8 activation codes and their f16 scales: per 32-element block,
//! `dA · Σ qa·(w − min) + sA · min`, with `w` from the production host dequant and `min` the same
//! dequant of the block with its quant bits zeroed. The only freedom left is f32 association, so
//! the tolerance is a small multiple of f32 epsilon over the absolute term sum; a wrong nibble,
//! scale, sign or row lands at the size of the terms themselves. What int8 activations cost
//! against f32 is bounded separately, in `seam_op_parity`'s `kquant_linear_*` tests.
// Wrapped in its own `#[cfg(test)]` module, like every in-crate test, so the bare-print guard
// (`infr-core/tests/no_bare_print.rs`) reads its skip notices as test output.
#[cfg(test)]
mod tests {
    use crate::gemm::I8cmTile;
    use crate::VulkanBackend;
    use infr_core::backend::{Backend, Buffer, BufferUsage};
    use infr_core::DType;

    /// Tokens: two large tiles, the second partial, and a partial medium tile.
    const M: usize = 200;
    /// Two superblocks, so a weight row spans more than one superblock.
    const K: usize = 512;
    /// One large tile, two medium ones.
    const N: usize = 128;
    /// One superblock of other weights ahead of the tested rows: exercises `w_base`, and puts the
    /// Q6_K superblocks (210 bytes) on 2-byte boundaries.
    const W_BASE: usize = 256;

    struct Rng(u64);
    impl Rng {
        fn byte(&mut self) -> u8 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u8
        }
    }

    /// A valid bank of `n_elems` weights: random quant and scale bits with small f16 super-scales.
    fn synth_bank(dt: DType, n_elems: usize, seed: u64) -> Vec<u8> {
        let (epb, bpb) = infr_gguf::block_layout(dt);
        let mut rng = Rng(seed);
        let mut out = Vec::with_capacity(n_elems / epb * bpb);
        for _ in 0..n_elems / epb {
            let mut b: Vec<u8> = (0..bpb).map(|_| rng.byte()).collect();
            let d = half::f16::from_f32(0.004 + (rng.byte() as f32) * 1e-5).to_le_bytes();
            let dmin = half::f16::from_f32(0.002 + (rng.byte() as f32) * 1e-5).to_le_bytes();
            match dt {
                DType::Q4K | DType::Q5K => {
                    b[0..2].copy_from_slice(&d);
                    b[2..4].copy_from_slice(&dmin);
                }
                DType::Q6K => b[208..210].copy_from_slice(&d),
                other => panic!("no synth for {other:?}"),
            }
            out.extend_from_slice(&b);
        }
        out
    }

    /// The per-block additive term of `bank` (`−dmin·mn` per 32 elements), or zeros for Q6_K, whose
    /// values are symmetric: the dequant of the same bank with every quant bit cleared.
    fn block_mins(dt: DType, bank: &[u8]) -> Vec<f32> {
        let (_, bpb) = infr_gguf::block_layout(dt);
        let quant_bytes = match dt {
            DType::Q4K => 16..144,
            DType::Q5K => 16..176,
            DType::Q6K => return vec![0.0; bank.len() / bpb * 8],
            other => panic!("no mins for {other:?}"),
        };
        let mut zeroed = bank.to_vec();
        for block in zeroed.chunks_mut(bpb) {
            block[quant_bytes.clone()].fill(0);
        }
        let w = infr_gguf::dequant::dequant_block(dt, &zeroed).unwrap();
        w.chunks(32).map(|c| c[0]).collect()
    }

    fn f16_at(bytes: &[u8], i: usize) -> f64 {
        half::f16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]).to_f64()
    }

    fn download_bytes(be: &VulkanBackend, b: &dyn Buffer) -> Vec<u8> {
        let mut v = vec![0u8; b.len_bytes()];
        be.download(b, &mut v).unwrap();
        v
    }

    /// Run every K-quant at both tiles; each output checked against the decomposition, both tiles
    /// bitwise equal (the k order is the same, so the tile choice can never move a result), rows past
    /// `M` untouched.
    #[test]
    #[ignore = "requires a Vulkan GPU with int8 cooperative matrix"]
    fn i8cm_kq_matches_its_decomposition() {
        let Ok(be) = VulkanBackend::new() else {
            eprintln!("i8cm_kq_matches_its_decomposition: no Vulkan device — skipped");
            return;
        };
        if !be.i8cm_kq_ready() {
            eprintln!(
                "i8cm_kq_matches_its_decomposition: int8 coopmat GEMM not armed here — skipped"
            );
            return;
        }
        let x: Vec<f32> = (0..M * K)
            .map(|i| (i as f32 * 0.37).sin() * (1.0 + (i % 97) as f32 * 0.05))
            .collect();
        let xbuf = be.alloc(M * K * 4, BufferUsage::Activations).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let nblk = K / 32;
        let qa = be.alloc(M * K, BufferUsage::Activations).unwrap();
        let dact = be.alloc(M * nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(M * nblk * 2, BufferUsage::Activations).unwrap();
        // Two extra rows of sentinel past M: a store past the last real row shows up there.
        let mpad = M + 2;
        let sentinel = vec![f32::NAN; mpad * N];

        for dt in [DType::Q4K, DType::Q5K, DType::Q6K] {
            let (epb, bpb) = infr_gguf::block_layout(dt);
            let mut bank = synth_bank(dt, W_BASE, 0xbad0 ^ dt as u64);
            let rows = synth_bank(dt, N * K, 0x5eed ^ dt as u64);
            bank.extend_from_slice(&rows);
            let w_host = infr_gguf::dequant::dequant_block(dt, &rows).unwrap();
            let mins = block_mins(dt, &rows);
            let wbuf = be.alloc(bank.len(), BufferUsage::Weights).unwrap();
            be.upload(wbuf.as_ref(), &bank).unwrap();
            let w_addr = wbuf.device_addr().expect("weights carry a device address");
            assert_eq!(bank.len(), (W_BASE + N * K) / epb * bpb);

            let mut outs = Vec::new();
            for tile in [I8cmTile::L, I8cmTile::M] {
                let out = be.alloc(mpad * N * 4, BufferUsage::Activations).unwrap();
                be.upload(out.as_ref(), bytemuck::cast_slice(&sentinel))
                    .unwrap();
                let rec = be.recorder().unwrap();
                rec.quant_q8(
                    xbuf.as_ref(),
                    qa.as_ref(),
                    dact.as_ref(),
                    sact.as_ref(),
                    M,
                    K,
                );
                rec.matmul_i8cm_kq(
                    dt,
                    tile,
                    qa.as_ref(),
                    dact.as_ref(),
                    sact.as_ref(),
                    w_addr,
                    W_BASE,
                    out.as_ref(),
                    M,
                    K,
                    N,
                );
                rec.finish().unwrap();
                let mut got = vec![0f32; mpad * N];
                be.download(out.as_ref(), bytemuck::cast_slice_mut(&mut got))
                    .unwrap();
                outs.push(got);
            }

            let qa_h = download_bytes(&be, qa.as_ref());
            let da_h = download_bytes(&be, dact.as_ref());
            let sa_h = download_bytes(&be, sact.as_ref());
            let got = &outs[0];
            let mut worst = 0f64;
            for r in 0..M {
                for o in 0..N {
                    let (mut want, mut mag) = (0f64, 0f64);
                    for b in 0..nblk {
                        let da = f16_at(&da_h, r * nblk + b);
                        let sa = f16_at(&sa_h, r * nblk + b);
                        let min = mins[o * nblk + b] as f64;
                        let mut dot = 0f64;
                        for i in b * 32..(b + 1) * 32 {
                            let q = qa_h[r * K + i] as i8 as f64;
                            let t = q * (w_host[o * K + i] as f64 - min);
                            dot += t;
                            mag += (t * da).abs();
                        }
                        want += da * dot + sa * min;
                        mag += (sa * min).abs();
                    }
                    let g = got[r * N + o] as f64;
                    let err = (g - want).abs();
                    worst = worst.max(err / mag);
                    assert!(
                        err <= 1e-5 * mag,
                        "{dt:?}: row {r} col {o}: got {g} want {want} (term sum {mag})"
                    );
                }
            }
            eprintln!("{dt:?}: worst error / term sum = {worst:e}");
            for (out, tile) in outs.iter().zip(["large", "medium"]) {
                assert!(
                    out[M * N..].iter().all(|v| v.is_nan()),
                    "{dt:?}: the {tile} tile wrote rows past m"
                );
            }
            let ndiff = outs[0][..M * N]
                .iter()
                .zip(&outs[1][..M * N])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                ndiff, 0,
                "{dt:?}: the two tiles disagree in {ndiff} outputs"
            );
        }
    }
}
