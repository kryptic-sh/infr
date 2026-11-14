//! GPU↔host parity + determinism for the resident Q6_K batched expert mmq GEMM
//! (`native_gemm_mmq_q6k`'s EXPERT_GRID builds) at every tile `matmul_mmq_experts` picks:
//! `_xp` (BM=64, BN=64), `_xp128` (BM=64, BN=128), `_xp32` (BM=32, BN=64) and `_xp32w`
//! (BM=32, BN=128). Q6_K is the down bank of most Q4_K_M MoE quants (ERNIE-4.5-21B-A3B,
//! Qwen3-30B-A3B), and before this test only the dense build had a host reference.
//!
//! Banks are APERIODIC pseudo-random 210-byte blocks with the f16 `d` patched small, so every
//! (half, og) nibble/high-bit lane decodes differently (a 64-periodic bank once hid a transposed
//! nibble map), and `k = 512`/`1280` puts every other superblock 2 bytes off a word boundary
//! (210 % 4 == 2) — the kernel's unaligned word-read arm. The host reference is the production
//! host dequant (`infr_gguf::dequant::dequant_block`) over the GPU's OWN quantized activation
//! codes (downloaded `qa`/`dact`), so a symmetric format matches to float-association noise.
//!
//! Determinism: the same GEMM dispatched 3x in one submit must be bitwise identical (the
//! restage-barrier race lesson from `mmq_wide_bn_determinism.rs`).
//!
//! Run: `cargo test -p infr-vulkan --release --test moe_mmq_q6k_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::linear::pad_to_u32_align;
use infr_vulkan::VulkanBackend;

/// Deterministic byte stream so failures reproduce.
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

/// Random Q6_K blocks (ql/qh/scales cover their full ranges) with `d` patched small.
fn synth_q6k(n_elems: usize, seed: u64) -> Vec<u8> {
    let (epb, bpb) = infr_gguf::block_layout(DType::Q6K);
    assert_eq!(n_elems % epb, 0, "bank must be superblock-aligned");
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n_elems / epb * bpb);
    for _ in 0..n_elems / epb {
        let mut b: Vec<u8> = (0..bpb).map(|_| rng.byte()).collect();
        let d = half::f16::from_f32(0.004 + (rng.byte() as f32) * 1e-5);
        b[208..210].copy_from_slice(&d.to_le_bytes());
        out.extend_from_slice(&b);
    }
    out
}

/// One tile configuration: `counts` per expert (n_used = 1, so avg rows/expert = rows / n_expert
/// picks BM) and the GEMM's `k`/`n` (n % 128 and k ≤ 1024 pick BN — see `matmul_mmq_experts`).
fn run(be: &VulkanBackend, label: &str, counts_host: &[u32], k: usize, n: usize) {
    let n_expert = counts_host.len();
    let stride = k * n;
    let banks: Vec<Vec<u8>> = (0..n_expert)
        .map(|e| synth_q6k(stride, 0x6b6b ^ ((e as u64) << 32) ^ k as u64))
        .collect();
    let host_w: Vec<Vec<f32>> = banks
        .iter()
        .map(|b| infr_gguf::dequant::dequant_block(DType::Q6K, b).unwrap())
        .collect();
    let flat = banks.concat();
    let padded = pad_to_u32_align(&flat);
    let w = be.alloc(padded.len(), BufferUsage::Weights).unwrap();
    be.upload(w.as_ref(), &padded).unwrap();

    let offsets_host: Vec<u32> = counts_host
        .iter()
        .scan(0u32, |acc, &c| {
            let o = *acc;
            *acc += c;
            Some(o)
        })
        .collect();
    let rows = counts_host.iter().sum::<u32>() as usize;
    let npad = rows.div_ceil(64) * 64 + 64; // GEMM As-stage overread padding

    let x: Vec<f32> = (0..rows * k)
        .map(|i| (i as f32 * 0.11).sin() * 0.15 + 0.02)
        .collect();
    let xbuf = be.alloc(rows * k * 4, BufferUsage::Activations).unwrap();
    be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let qa = be.alloc(npad * k, BufferUsage::Activations).unwrap();
    let qda = be
        .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let qsa = be
        .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let outs: Vec<_> = (0..3)
        .map(|_| be.alloc(npad * n * 4, BufferUsage::Activations).unwrap())
        .collect();
    let mk_u32 = |v: &[u32]| {
        let b = be.alloc(v.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
        b
    };
    let counts = mk_u32(counts_host);
    let offsets = mk_u32(&offsets_host);

    let rec = be.recorder().unwrap();
    rec.quant_q8(
        xbuf.as_ref(),
        qa.as_ref(),
        qda.as_ref(),
        qsa.as_ref(),
        rows,
        k,
    );
    for out in &outs {
        rec.matmul_mmq_experts(
            DType::Q6K,
            "expert_down",
            qa.as_ref(),
            qda.as_ref(),
            None, // symmetric — no sact
            w.as_ref(),
            0,
            stride,
            counts.as_ref(),
            offsets.as_ref(),
            out.as_ref(),
            rows,
            k,
            n,
            n_expert,
            1,
        );
    }
    rec.finish().unwrap();

    let dl = |b: &dyn infr_core::backend::Buffer| -> Vec<f32> {
        let mut v = vec![0f32; npad * n];
        be.download(b, bytemuck::cast_slice_mut(&mut v)).unwrap();
        v.truncate(rows * n);
        v
    };
    let got = dl(outs[0].as_ref());
    assert!(got.iter().all(|v| v.is_finite()), "{label}: non-finite");
    for (run, out) in outs.iter().enumerate().skip(1) {
        let other = dl(out.as_ref());
        let ndiff = got
            .iter()
            .zip(&other)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(ndiff, 0, "{label}: dispatch 0 vs {run}: {ndiff} differ");
    }

    // Exact activation operand: the GPU's int8 codes × its f16 per-32-block scale.
    let mut codes = vec![0u8; npad * k];
    be.download(qa.as_ref(), &mut codes).unwrap();
    let mut scales = vec![0u8; npad * (k / 32) * 2];
    be.download(qda.as_ref(), &mut scales).unwrap();
    let xq: Vec<f32> = (0..rows * k)
        .map(|i| {
            let (r, c) = (i / k, i % k);
            let si = (r * (k / 32) + c / 32) * 2;
            let d = half::f16::from_le_bytes([scales[si], scales[si + 1]]).to_f32();
            (codes[r * k + c] as i8 as f32) * d
        })
        .collect();
    let mut checked = 0usize;
    for (e, (&off, &cnt)) in offsets_host.iter().zip(counts_host).enumerate() {
        for r in off as usize..(off + cnt) as usize {
            let xr = &xq[r * k..(r + 1) * k];
            for o in 0..n {
                let want: f32 = host_w[e][o * k..(o + 1) * k]
                    .iter()
                    .zip(xr)
                    .map(|(a, b)| a * b)
                    .sum();
                let g = got[r * n + o];
                assert!(
                    (g - want).abs() < 1e-3 + 1e-3 * want.abs(),
                    "{label}: expert {e} row {r} col {o}: got {g} want {want}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, rows * n, "{label}: every routed output checked");
    println!("{label}: OK ({rows} rows, k={k}, n={n}, 3x bitwise identical)");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn q6k_expert_gemm_matches_host_at_every_tile() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // avg rows/expert 40 (> the BM=32 cutoff) with k > 1024 → BM=64, BN=64.
    let big = [50u32, 40, 45, 35, 30];
    run(&be, "q6k xp", &big, 1280, 128);
    // Same rows, shallow k and n % 128 == 0 → the BN=128 wide twin.
    run(&be, "q6k xp128", &big, 512, 256);
    // avg rows/expert ~2 → BM=32; n % 128 != 0 keeps BN=64, n % 128 == 0 takes BN=128.
    let small = [0u32, 0, 4, 3, 2];
    run(&be, "q6k xp32", &small, 512, 192);
    run(&be, "q6k xp32w", &small, 512, 256);
}
