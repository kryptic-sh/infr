//! Parity + determinism for the NON-COOPMAT prefill GEMM tier's kernels (adapter.rs
//! `nc_mmq`/`nc_fma` — the Intel Arc route, exercisable anywhere since these kernels use no
//! coopmat):
//!
//!  • the 12 NEWLY WIRED dense (non-expert-grid) dp4a mmq GEMM builds (`matmul_mmq` over every
//!    `MOE_MMQ_DTYPES` member — Q4_K/Q6_K's dense builds pre-existed but ride the same dispatch
//!    here, so all 16 run — incl. the IQ2_S/IQ3_S grid pair);
//!  • the 3 fma-warp float GEMMs (`matmul_fma`: f16/bf16/f32 weights, native_gemm_fma.comp).
//!
//! Each config dispatches the SAME GEMM three times in one submission — bitwise-identical
//! outputs required (the mmq barrier-race lesson: goldens can't catch intra-dispatch races; see
//! `mmq_wide_bn_determinism.rs`, the template) — and checks the first output against a host
//! reference: `infr_gguf::dequant::dequant_block` weights × the GPU's own downloaded int8
//! activation codes for mmq (the exact-reference trick from `moe_mmq_fp4_parity.rs`), plain f32
//! dot for fma (whose weights are bit-exact on both sides). Min-carrying mmq dtypes
//! (Q4_K/Q5_K/Q4_1/Q5_1 via the f16 `sact` Σx term, Q2_K via its self-computed one) keep the
//! coarser tolerance the other mmq parity tests carry; symmetric dtypes decompose exactly.
//!
//! Run: `cargo test -p infr-vulkan --release --test nc_gemm_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, Buffer, BufferUsage};
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

/// (elements per block, bytes per block) for every mmq dtype under test.
/// `(elements, bytes)` per block, from the shared decode spec (`infr_core::decode_spec` via
/// `infr_gguf::block_layout`) — not a hand-kept local table.
fn block_geom(dt: DType) -> (usize, usize) {
    infr_gguf::block_layout(dt)
}

/// Synthetic VALID bank: random payload bytes with the scale fields patched small (a valid
/// encoding for every format here — quant/code bits cover their full ranges), decodable by the
/// production host dequant (`dequant_block`).
fn synth_bank(dt: DType, n_elems: usize, seed: u64) -> Vec<u8> {
    let (epb, bpb) = block_geom(dt);
    assert_eq!(n_elems % epb, 0);
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n_elems / epb * bpb);
    for _ in 0..n_elems / epb {
        let mut b: Vec<u8> = (0..bpb).map(|_| rng.byte()).collect();
        let d16 = |r: &mut Rng| half::f16::from_f32(0.004 + (r.byte() as f32) * 1e-5).to_le_bytes();
        let m16 = |r: &mut Rng| half::f16::from_f32(0.002 + (r.byte() as f32) * 1e-5).to_le_bytes();
        match dt {
            DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Iq4Nl | DType::Q2_0 => {
                b[0..2].copy_from_slice(&d16(&mut rng))
            }
            DType::Q4_1 | DType::Q5_1 => {
                b[0..2].copy_from_slice(&d16(&mut rng));
                b[2..4].copy_from_slice(&m16(&mut rng));
            }
            DType::Q2K => {
                b[80..82].copy_from_slice(&d16(&mut rng));
                b[82..84].copy_from_slice(&m16(&mut rng));
            }
            DType::Q3K => b[108..110].copy_from_slice(&d16(&mut rng)),
            DType::Q4K | DType::Q5K => {
                b[0..2].copy_from_slice(&d16(&mut rng));
                b[2..4].copy_from_slice(&m16(&mut rng));
            }
            DType::Q6K => b[208..210].copy_from_slice(&d16(&mut rng)),
            DType::Iq4Xs => b[0..2].copy_from_slice(&d16(&mut rng)),
            // grid i-quants: leading f16 d; grid-index/sign bits cover their full table ranges
            DType::Iq2S | DType::Iq3S => b[0..2].copy_from_slice(&d16(&mut rng)),
            // e8m0 scale byte: 122..=125 → 2^-6..2^-3
            DType::Mxfp4 => b[0] = 122 + (rng.byte() & 3),
            // ue4m3 scale bytes: [0x18, 0x37] → small positive, never 0/0x7F
            DType::Nvfp4 => {
                for s in b.iter_mut().take(4) {
                    *s = 0x18 + (rng.byte() & 0x1F);
                }
            }
            other => panic!("no synth for {other:?}"),
        }
        out.extend_from_slice(&b);
    }
    out
}

/// The GEMM shape shared by every config: m NOT a tile multiple (exercises the padded row
/// range), k covers 2 superblocks, n two BN=64 column tiles.
const M: usize = 100;
const K: usize = 512;
const N: usize = 128;

fn download_rows(be: &VulkanBackend, b: &dyn Buffer, mpad: usize) -> Vec<f32> {
    let mut v = vec![0f32; mpad * N];
    be.download(b, bytemuck::cast_slice_mut(&mut v)).unwrap();
    v.truncate(M * N);
    v
}

fn assert_deterministic(be: &VulkanBackend, outs: &[Box<dyn Buffer>], mpad: usize, label: &str) {
    let a = download_rows(be, outs[0].as_ref(), mpad);
    assert!(a.iter().all(|v| v.is_finite()), "{label}: non-finite");
    for (run, out) in outs.iter().enumerate().skip(1) {
        let b = download_rows(be, out.as_ref(), mpad);
        let ndiff = a.iter().zip(&b).filter(|(p, q)| p != q).count();
        assert_eq!(
            ndiff,
            0,
            "{label}: nondeterministic — dispatch 0 vs {run}: {ndiff}/{} differ",
            a.len()
        );
    }
}

fn assert_parity(got: &[f32], want: &[f32], tol: f32, label: &str) {
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < tol + tol * w.abs(),
            "{label}: mismatch at row {} col {}: got {g} want {w}",
            i / N,
            i % N
        );
    }
}

fn run_mmq(be: &VulkanBackend, dt: DType) {
    let label = format!("{dt:?} dense mmq");
    let bank = synth_bank(dt, N * K, 0x5eed ^ dt as u64);
    let w_host = infr_gguf::dequant::dequant_block(dt, &bank).unwrap();
    let padded = pad_to_u32_align(&bank);
    let w = be.alloc(padded.len(), BufferUsage::Weights).unwrap();
    be.upload(w.as_ref(), &padded).unwrap();

    let x: Vec<f32> = (0..M * K)
        .map(|i| (i as f32 * 0.11).sin() * 0.15 + 0.02)
        .collect();
    let xbuf = be.alloc(M * K * 4, BufferUsage::Activations).unwrap();
    be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

    let mpad = M.div_ceil(64) * 64;
    // zero-init allocs: the GEMM's As stage reads rows M..mpad (zeros, results discarded).
    let qa = be.alloc(mpad * K, BufferUsage::Activations).unwrap();
    let dact = be
        .alloc(mpad * (K / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let sact = be
        .alloc(mpad * (K / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let outs: Vec<_> = (0..3)
        .map(|_| be.alloc(mpad * N * 4, BufferUsage::Activations).unwrap())
        .collect();

    let rec = be.recorder().unwrap();
    rec.quant_q8(
        xbuf.as_ref(),
        qa.as_ref(),
        dact.as_ref(),
        sact.as_ref(),
        M,
        K,
    );
    for out in &outs {
        rec.matmul_mmq(
            dt,
            qa.as_ref(),
            dact.as_ref(),
            sact.as_ref(),
            w.as_ref(),
            0,
            out.as_ref(),
            M,
            K,
            N,
        );
    }
    rec.finish().unwrap();

    assert_deterministic(be, &outs, mpad, &label);

    // Host reference over the GPU's ACTUAL quantized activations (download + dequantize the int8
    // codes) — the dp4a decomposition then matches to float-association noise for symmetric
    // dtypes; min-carrying ones add the f16 `sact` rounding, hence their coarser tolerance.
    let mut cb = vec![0u8; mpad * K];
    be.download(qa.as_ref(), &mut cb).unwrap();
    let mut sb = vec![0u8; mpad * (K / 32) * 2];
    be.download(dact.as_ref(), &mut sb).unwrap();
    let xq: Vec<f32> = (0..M * K)
        .map(|i| {
            let (r, c) = (i / K, i % K);
            let si = (r * (K / 32) + c / 32) * 2;
            let d = half::f16::from_le_bytes([sb[si], sb[si + 1]]).to_f32();
            (cb[r * K + c] as i8 as f32) * d
        })
        .collect();
    let mut want = vec![0f32; M * N];
    for r in 0..M {
        let xr = &xq[r * K..(r + 1) * K];
        for o in 0..N {
            want[r * N + o] = w_host[o * K..(o + 1) * K]
                .iter()
                .zip(xr)
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    let min_carrying = infr_core::tensor::moe_mmq_needs_sact(dt) || matches!(dt, DType::Q2K);
    let tol = if min_carrying { 5e-2 } else { 1e-3 };
    let got = download_rows(be, outs[0].as_ref(), mpad);
    assert_parity(&got, &want, tol, &label);
    if dt == DType::Q4K {
        let mut seen = [[0u16; 2]; 4];
        for block in bank.as_chunks::<144>().0 {
            for (i, &byte) in block[16..144].iter().enumerate() {
                seen[i % 4][0] |= 1 << (byte & 15);
                seen[i % 4][1] |= 1 << (byte >> 4);
            }
        }
        assert_eq!(seen, [[u16::MAX; 2]; 4], "Q4K nibble coverage");
    }
    // The three weight routes — bound SSBO at an element offset, BDA at a byte offset, and the
    // paged arena after an eviction moved the slot — must agree bitwise with the plain dispatch.
    // Q6_K rides along for its `rw` word reads through the STREAMED/PAGED `arena_word` path: its
    // 256-element prefix is one 210-byte block, so every offset route starts 2 bytes off a word.
    if matches!(dt, DType::Q4K | DType::Q6K) {
        let needs_sact = infr_core::tensor::moe_mmq_needs_sact(dt);
        let mut prefixed = synth_bank(dt, 256, 0x1234);
        prefixed.extend_from_slice(&bank);
        let bound = be.alloc(prefixed.len(), BufferUsage::Weights).unwrap();
        be.upload(bound.as_ref(), &prefixed).unwrap();
        let mut arena_bytes = vec![0xa5; 256];
        arena_bytes.extend_from_slice(&prefixed);
        let (arena, addr) = be.alloc_arena_bda(arena_bytes.len()).unwrap();
        be.upload(arena.as_ref(), &arena_bytes).unwrap();
        let route_outs: Vec<_> = (0..3)
            .map(|_| be.alloc(mpad * N * 4, BufferUsage::Activations).unwrap())
            .collect();

        let banks: Vec<_> = (0..5)
            .map(|e| {
                if e == 0 {
                    bank.clone()
                } else {
                    synth_bank(dt, N * K, 0x1234 + e)
                }
            })
            .collect();
        for (i, a) in banks.iter().enumerate() {
            for b in &banks[i + 1..] {
                assert_ne!(a, b, "pager banks must differ");
            }
        }
        let mut pager = infr_vulkan::pager::GpuPager::new(be, 5, 3, bank.len()).unwrap();
        let staging = be.alloc(bank.len(), BufferUsage::Staging).unwrap();
        let first_slot = pager
            .ensure_resident(be, staging.as_ref(), 0, &banks[0])
            .unwrap();
        let evictions = pager.stats().evictions;
        for e in [1u32, 2, 3, 4] {
            pager
                .ensure_resident(be, staging.as_ref(), e, &banks[e as usize])
                .unwrap();
        }
        assert!(!pager.is_resident(0), "expert zero must be evicted");
        let reloaded_slot = pager
            .ensure_resident(be, staging.as_ref(), 0, &banks[0])
            .unwrap();
        assert!(pager.stats().evictions > evictions, "pager must churn");
        assert_ne!(first_slot, reloaded_slot, "expert zero must move slots");
        pager.flush_lut(be).unwrap();
        let mut lut = [0u32; 5];
        be.download(pager.lut_buffer(), bytemuck::cast_slice_mut(&mut lut))
            .unwrap();
        assert_eq!(lut[0], reloaded_slot);
        let counts = be.alloc(5 * 4, BufferUsage::Activations).unwrap();
        let offsets = be.alloc(5 * 4, BufferUsage::Activations).unwrap();
        be.upload(
            counts.as_ref(),
            bytemuck::cast_slice(&[M as u32, 0, 0, 0, 0]),
        )
        .unwrap();
        be.upload(
            offsets.as_ref(),
            bytemuck::cast_slice(&[0u32, M as u32, M as u32, M as u32, M as u32]),
        )
        .unwrap();
        let rec = be.recorder().unwrap();
        // The bound-SSBO builds (`nw[]` reads, no arena address) — dtype-specific entry points.
        if dt == DType::Q4K {
            rec.matmul_mmq_q4k(
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                bound.as_ref(),
                256,
                route_outs[0].as_ref(),
                M,
                K,
                N,
            );
        } else {
            rec.matmul_mmq_q6k(
                qa.as_ref(),
                dact.as_ref(),
                bound.as_ref(),
                256,
                route_outs[0].as_ref(),
                M,
                K,
                N,
            );
        }
        rec.matmul_mmq_at(
            dt,
            qa.as_ref(),
            dact.as_ref(),
            sact.as_ref(),
            addr + 256,
            256,
            route_outs[1].as_ref(),
            M,
            K,
            N,
        );
        rec.matmul_mmq_experts_paged(
            dt,
            "expert_gateup",
            qa.as_ref(),
            dact.as_ref(),
            needs_sact.then(|| sact.as_ref() as &dyn Buffer),
            pager.arena_addr(),
            pager.slot_bytes() as u32,
            pager.lut_buffer(),
            0,
            counts.as_ref(),
            offsets.as_ref(),
            route_outs[2].as_ref(),
            M,
            K,
            N,
            5,
            1,
        );
        rec.finish().unwrap();
        assert_eq!(got.len(), M * N);
        assert!(got.iter().all(|v| v.is_finite()));
        for (route, out) in ["SSBO", "offset BDA", "paged reload"]
            .iter()
            .zip(&route_outs)
        {
            let actual = download_rows(be, out.as_ref(), mpad);
            assert_eq!(actual.len(), got.len());
            for (i, (a, b)) in actual.iter().zip(&got).enumerate() {
                assert!(a.is_finite(), "{route}: non-finite at {i}");
                assert_eq!(a.to_bits(), b.to_bits(), "{route}: mismatch at {i}");
            }
        }
        println!("{label}: SSBO / offset BDA / paged reload routes bitwise equal");
    }
    println!("{label}: OK");
}

fn run_fma(be: &VulkanBackend, dt: DType, w_base: usize) {
    let label = format!("{dt:?} fma-warp (w_base={w_base})");
    let mut rng = Rng(0xfab ^ dt as u64);
    // Weight values built directly in the storage format — bit-exact on both sides.
    let n_w = w_base + N * K;
    let (bytes, w_host): (Vec<u8>, Vec<f32>) = match dt {
        DType::F16 => {
            let mut by = Vec::with_capacity(n_w * 2);
            let mut ho = Vec::with_capacity(n_w);
            for _ in 0..n_w {
                // sign | exp 11..14 of 31 (2^-4..2^-1) | mantissa
                let bits: u16 = (((rng.byte() & 1) as u16) << 15)
                    | ((11 + (rng.byte() & 3) as u16) << 10)
                    | ((rng.byte() as u16) << 2 | (rng.byte() & 3) as u16);
                by.extend_from_slice(&bits.to_le_bytes());
                ho.push(half::f16::from_bits(bits).to_f32());
            }
            (by, ho)
        }
        DType::Bf16 => {
            let mut by = Vec::with_capacity(n_w * 2);
            let mut ho = Vec::with_capacity(n_w);
            for _ in 0..n_w {
                // sign | exp 0x7B..0x7E (2^-4..2^-1) | mantissa(7)
                let bits: u16 = (((rng.byte() & 1) as u16) << 15)
                    | (((0x7B + (rng.byte() & 3) as u16) & 0xFF) << 7)
                    | (rng.byte() & 0x7F) as u16;
                by.extend_from_slice(&bits.to_le_bytes());
                ho.push(f32::from_bits((bits as u32) << 16));
            }
            (by, ho)
        }
        DType::F32 => {
            let mut by = Vec::with_capacity(n_w * 4);
            let mut ho = Vec::with_capacity(n_w);
            for _ in 0..n_w {
                let v = ((rng.byte() as f32) - 127.5) * 0.004;
                by.extend_from_slice(&v.to_le_bytes());
                ho.push(v);
            }
            (by, ho)
        }
        other => panic!("no fma build for {other:?}"),
    };
    let w = be.alloc(bytes.len(), BufferUsage::Weights).unwrap();
    be.upload(w.as_ref(), &bytes).unwrap();

    let x: Vec<f32> = (0..M * K)
        .map(|i| (i as f32 * 0.13).sin() * 0.2 - 0.01)
        .collect();
    let xbuf = be.alloc(M * K * 4, BufferUsage::Activations).unwrap();
    be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

    let mpad = M.div_ceil(64) * 64;
    let outs: Vec<_> = (0..3)
        .map(|_| be.alloc(mpad * N * 4, BufferUsage::Activations).unwrap())
        .collect();

    let rec = be.recorder().unwrap();
    for out in &outs {
        rec.matmul_fma(dt, xbuf.as_ref(), w.as_ref(), w_base, out.as_ref(), M, K, N);
    }
    rec.finish().unwrap();

    assert_deterministic(be, &outs, mpad, &label);

    let mut want = vec![0f32; M * N];
    for r in 0..M {
        for o in 0..N {
            want[r * N + o] = w_host[w_base + o * K..w_base + (o + 1) * K]
                .iter()
                .zip(&x[r * K..(r + 1) * K])
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    let got = download_rows(be, outs[0].as_ref(), mpad);
    assert_parity(&got, &want, 1e-3, &label);
    println!("{label}: OK");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn nc_mmq_dense_gemm_parity_and_determinism() {
    let Some(be) = infr_testkit::gpu_or_skip(VulkanBackend::new()) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // NB: no i8_dot gate — production callers check `caps().i8_dot`; this targets dp4a dev boxes.
    for &dt in infr_core::tensor::MOE_MMQ_DTYPES {
        run_mmq(&be, dt);
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn nc_fma_gemm_parity_and_determinism() {
    let Some(be) = infr_testkit::gpu_or_skip(VulkanBackend::new()) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    run_fma(&be, DType::F16, 0);
    run_fma(&be, DType::Bf16, 0);
    run_fma(&be, DType::F32, 0);
    // Non-zero element base (a bf16 fused-QKV slice / streamed-slot offset rides `w_base`).
    run_fma(&be, DType::Bf16, K);
}

/// One fp32 K-quant prefill GEMM (`native_gemm_kq_fma.comp`) config: `splits` picks the split-K
/// twin over the direct tile. Three identical dispatches must agree bit for bit, the padded row
/// range must cover the whole write, and every element must match a host f32 dot of the dequantized
/// weight.
#[allow(clippy::too_many_arguments)]
fn run_kq_fma(
    be: &VulkanBackend,
    dt: DType,
    bn: usize,
    m: usize,
    k: usize,
    n: usize,
    w_base: usize,
    splits: Option<usize>,
) {
    let label = format!("{dt:?} kq_fma t{bn} m{m} k{k} n{n} w_base{w_base} splits{splits:?}");
    let bank = synth_bank(dt, w_base + n * k, 0x9c4 ^ dt as u64 ^ (k * n) as u64);
    let w_host = infr_gguf::dequant::dequant_block(dt, &bank).unwrap();
    let padded = pad_to_u32_align(&bank);
    let w = be.alloc(padded.len(), BufferUsage::Weights).unwrap();
    be.upload(w.as_ref(), &padded).unwrap();

    let x: Vec<f32> = (0..m * k)
        .map(|i| infr_core::math::sin(i as f32 * 0.13) * 0.2 - 0.01)
        .collect();
    let xbuf = be.alloc(m * k * 4, BufferUsage::Activations).unwrap();
    be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

    let mpad = m.div_ceil(64) * 64;
    // One spare row past the padded range catches an epilogue overrun.
    let outs: Vec<_> = (0..3)
        .map(|_| {
            let b = be
                .alloc((mpad + 1) * n * 4, BufferUsage::Activations)
                .unwrap();
            be.upload(b.as_ref(), &vec![0u8; (mpad + 1) * n * 4])
                .unwrap();
            b
        })
        .collect();
    let rec = be.recorder().unwrap();
    let parts = splits.map(|s| {
        be.alloc(s * mpad * n * 4, BufferUsage::Activations)
            .unwrap()
    });
    for out in &outs {
        match (&parts, splits) {
            (Some(p), Some(s)) => rec.matmul_kq_fma_splitk(
                dt,
                bn,
                xbuf.as_ref(),
                w.as_ref(),
                w_base,
                p.as_ref(),
                out.as_ref(),
                m,
                k,
                n,
                s,
            ),
            _ => rec.matmul_kq_fma(
                dt,
                bn,
                xbuf.as_ref(),
                w.as_ref(),
                w_base,
                out.as_ref(),
                m,
                k,
                n,
            ),
        }
    }
    rec.finish().unwrap();

    let download = |b: &dyn Buffer| {
        let mut v = vec![0f32; (mpad + 1) * n];
        be.download(b, bytemuck::cast_slice_mut(&mut v)).unwrap();
        v
    };
    let got = download(outs[0].as_ref());
    assert!(
        got[mpad * n..].iter().all(|v| *v == 0.0),
        "{label}: wrote past the padded row range"
    );
    for (run, out) in outs.iter().enumerate().skip(1) {
        let other = download(out.as_ref());
        let ndiff = got[..m * n]
            .iter()
            .zip(&other)
            .filter(|(p, q)| p != q)
            .count();
        assert_eq!(ndiff, 0, "{label}: nondeterministic — dispatch 0 vs {run}");
    }
    for r in 0..m {
        for o in 0..n {
            let want: f32 = w_host[w_base + o * k..w_base + (o + 1) * k]
                .iter()
                .zip(&x[r * k..(r + 1) * k])
                .map(|(a, b)| a * b)
                .sum();
            let g = got[r * n + o];
            assert!(
                g.is_finite() && (g - want).abs() < 1e-3 + 1e-3 * want.abs(),
                "{label}: mismatch at row {r} col {o}: got {g} want {want}"
            );
        }
    }
}

/// What [`kq_fma_and_gemv_rms_error`] measures on one shape.
struct KqFmaError {
    /// The tile's root-mean-square error against the f64 reference.
    tile: f64,
    /// The per-row GEMV's, on the same weights and activations.
    gemv: f64,
    /// The root mean square of the f64 reference's outputs: the scale both errors are read
    /// against.
    scale: f64,
}

/// The tile's RMS error allowed against the f64 reference, as a fraction of the RMS of the
/// reference's outputs, in units of `f32::EPSILON`. It is a second, independent requirement, not a
/// relaxation of the first: the comparison with the GEMV alone would pass a tile that got worse
/// together with the GEMV, so this bound holds the tile to the reference itself, and a case fails
/// when it crosses either. The GEMV comparison stays strict, so a change that makes `native_gemv`
/// more accurate can turn these tests red with no regression in the tile; then check the tile
/// against this bound and the previous figures, and either improve the tile or bring the change
/// to the owner, rather than loosening the comparison. The value sits above the compensated
/// tile's worst case with a margin and below what most direct cases give with the carry dropped,
/// so on the direct path the bound on its own catches a lost compensation; the figures are in
/// `docs/gpu-tolerance.md`, "The fp32 K-quant prefill GEMM in the float runs". That margin was
/// measured on these seeded inputs, one driver release and two GPUs whose shader compilers agree
/// to the printed digit, and on nothing else: not RADV in these units, not Intel. Before calling
/// a crossing on a new driver a regression, compare the tile against a host emulation of the same
/// sum (`f32::mul_add` partials of 32 plus the Kahan step) to see whether the driver fuses
/// `fma()`; if it does not and the figure is above this bound, raise the constant and write that
/// figure into the docs.
const KQ_FMA_REL_BOUND_EPS: f64 = 1.2;

impl KqFmaError {
    /// Why this shape fails, if it does: the tile further from the reference than the GEMV, past
    /// [`KQ_FMA_REL_BOUND_EPS`], or both.
    fn failure(&self) -> Option<String> {
        let mut why = Vec::new();
        if self.tile > self.gemv {
            why.push("above the per-row GEMV's error");
        }
        if self.tile > KQ_FMA_REL_BOUND_EPS * f32::EPSILON as f64 * self.scale {
            why.push("above the bound against the f64 reference");
        }
        (!why.is_empty()).then(|| why.join(" and "))
    }
}

impl std::fmt::Display for KqFmaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let eps = f32::EPSILON as f64 * self.scale;
        write!(
            f,
            "tile {:.3e} ({:.3} eps), gemv {:.3e} ({:.3} eps), outputs' RMS {:.3e}",
            self.tile,
            self.tile / eps,
            self.gemv,
            self.gemv / eps,
            self.scale
        )
    }
}

/// The fp32 K-quant tile's (`splits` picks split-K) and the per-row f32 dequant GEMV's
/// (`native_gemv.comp`, the kernel the tile replaces) root-mean-square error against an f64 dot
/// of the host-dequantized weight, on one shared input.
#[allow(clippy::too_many_arguments)]
fn kq_fma_and_gemv_rms_error(
    be: &VulkanBackend,
    dt: DType,
    bn: usize,
    m: usize,
    k: usize,
    n: usize,
    splits: Option<usize>,
) -> KqFmaError {
    let bank = synth_bank(dt, n * k, 0xacc ^ dt as u64 ^ (k * n) as u64);
    let w_host = infr_gguf::dequant::dequant_block(dt, &bank).unwrap();
    let padded = pad_to_u32_align(&bank);
    let w = be.alloc(padded.len(), BufferUsage::Weights).unwrap();
    be.upload(w.as_ref(), &padded).unwrap();
    // Signed activations around a positive mean, so each output's running sum drifts away from
    // zero along k the way a real layer's does.
    let mut rng = Rng(0x5a17 ^ k as u64);
    let x: Vec<f32> = (0..m * k)
        .map(|_| (rng.byte() as f32 / 255.0 - 0.3) * 2.0)
        .collect();
    let xbuf = be.alloc(m * k * 4, BufferUsage::Activations).unwrap();
    be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let mpad = m.div_ceil(64) * 64;
    let tile = be.alloc(mpad * n * 4, BufferUsage::Activations).unwrap();
    let gemv = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();
    let parts = splits.map(|s| {
        be.alloc(s * mpad * n * 4, BufferUsage::Activations)
            .unwrap()
    });
    let rec = be.recorder().unwrap();
    match (&parts, splits) {
        (Some(p), Some(s)) => rec.matmul_kq_fma_splitk(
            dt,
            bn,
            xbuf.as_ref(),
            w.as_ref(),
            0,
            p.as_ref(),
            tile.as_ref(),
            m,
            k,
            n,
            s,
        ),
        _ => rec.matmul_kq_fma(dt, bn, xbuf.as_ref(), w.as_ref(), 0, tile.as_ref(), m, k, n),
    }
    let addr = w
        .device_addr()
        .expect("a resident weight has a device address");
    rec.linear_native_at(dt, addr, 0, xbuf.as_ref(), gemv.as_ref(), m, k, n);
    rec.finish().unwrap();
    let want: Vec<f64> = (0..m * n)
        .map(|i| {
            let (r, o) = (i / n, i % n);
            w_host[o * k..(o + 1) * k]
                .iter()
                .zip(&x[r * k..(r + 1) * k])
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum()
        })
        .collect();
    let rms = |b: &dyn Buffer, rows: usize| {
        let mut v = vec![0f32; rows * n];
        be.download(b, bytemuck::cast_slice_mut(&mut v)).unwrap();
        let mut se = 0f64;
        for (i, want) in want.iter().enumerate() {
            let g = v[i];
            assert!(
                g.is_finite(),
                "{dt:?} non-finite at row {} col {}",
                i / n,
                i % n
            );
            se += (g as f64 - want).powi(2);
        }
        (se / (m * n) as f64).sqrt()
    };
    KqFmaError {
        tile: rms(tile.as_ref(), mpad),
        gemv: rms(gemv.as_ref(), m),
        scale: (want.iter().map(|w| w * w).sum::<f64>() / (m * n) as f64).sqrt(),
    }
}

/// The tile is held to the kernel it replaces and to the reference: on the same weights and
/// activations its RMS error against an f64 host reference is no larger than the per-row GEMV's,
/// and within [`KQ_FMA_REL_BOUND_EPS`] of the reference's own scale. A tile summing each output
/// in one f32 running sum over the whole k range fails this (the shapes include gemma-3-1b's
/// `ffn_down` and `attn_output`); its summation is what this guards.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn nc_kq_fma_gemm_is_as_accurate_as_the_gemv() {
    let Some(be) = infr_testkit::gpu_or_skip(VulkanBackend::new()) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let mut worse = Vec::new();
    for dt in [DType::Q4_0, DType::Q4K, DType::Q6K] {
        for bn in [64, 128] {
            for (m, k, n) in [(20, 6912, 1152), (33, 1024, 1152), (64, 4096, 256)] {
                let e = kq_fma_and_gemv_rms_error(&be, dt, bn, m, k, n, None);
                let line = format!("{dt:?} t{bn} m{m} k{k} n{n}: {e}");
                println!("{line}");
                if let Some(why) = e.failure() {
                    worse.push(format!("{line}: {why}"));
                }
            }
        }
    }
    assert!(
        worse.is_empty(),
        "the tile is not as accurate as required:\n  {}",
        worse.join("\n  ")
    );
}

/// The split-K twin under the same two bounds as the direct tile, at the slice counts the plan
/// picks for gemma-3-1b's `attn_output` (k = 1024) and for a deeper k.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn nc_kq_fma_splitk_is_as_accurate_as_the_gemv() {
    let Some(be) = infr_testkit::gpu_or_skip(VulkanBackend::new()) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let mut worse = Vec::new();
    for dt in [DType::Q4_0, DType::Q4K, DType::Q6K] {
        for bn in [64, 128] {
            for (m, k, n, s) in [(33, 1024, 1152, 4), (33, 1024, 1152, 2), (20, 8192, 256, 8)] {
                let e = kq_fma_and_gemv_rms_error(&be, dt, bn, m, k, n, Some(s));
                let line = format!("{dt:?} t{bn} m{m} k{k} n{n} splits{s}: {e}");
                println!("{line}");
                if let Some(why) = e.failure() {
                    worse.push(format!("{line}: {why}"));
                }
            }
        }
    }
    assert!(
        worse.is_empty(),
        "the split-K tile is not as accurate as required:\n  {}",
        worse.join("\n  ")
    );
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn nc_kq_fma_gemm_parity_and_determinism() {
    let Some(be) = infr_testkit::gpu_or_skip(VulkanBackend::new()) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    for dt in [DType::Q4_0, DType::Q4K, DType::Q6K] {
        for bn in [64, 128] {
            // k=768 gives 3 blocks per row, so Q6_K columns alternate 2-byte block alignment.
            for (m, k, n) in [
                (2, 256, 128),
                (100, 512, 128),
                (130, 768, 256),
                (512, 1024, 256),
            ] {
                run_kq_fma(&be, dt, bn, m, k, n, 0, None);
            }
            run_kq_fma(&be, dt, bn, 70, 768, 128, 256, None);
            // gemma-3-1b's ffn_down: an odd count of superblocks per row, too few to split.
            run_kq_fma(&be, dt, bn, 20, 6912, 1152, 0, None);
        }
    }
}

/// The split-K twin: same host-reference parity and run-to-run determinism as the direct tile,
/// with the k range cut into `splits` slices plus a `splitk_reduce_kahan` pass. Shapes cover a
/// single row tile, a partial row tile and several (m=512: 4 row tiles) across both column tiles.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn nc_kq_fma_splitk_matches_the_direct_tile() {
    let Some(be) = infr_testkit::gpu_or_skip(VulkanBackend::new()) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    for dt in [DType::Q4_0, DType::Q4K, DType::Q6K] {
        for bn in [64, 128] {
            for (m, k, n, splits) in [(2, 1024, 128, 4), (130, 1024, 256, 4), (512, 1024, 256, 2)] {
                run_kq_fma(&be, dt, bn, m, k, n, 0, Some(splits));
            }
        }
    }
}
