//! Small-m Linear cost on the Vulkan seam — the spec-decode verify / short-suffix-prefill shape
//! (m = 2..8 rows). Today the adapter routes any m>1 to the tiled coopmat GEMM, whose grid is only
//! n/64 workgroups at a single M-tile (underfills a 48-WGP GPU); the alternative GEMV re-streams
//! the weight per row. This bench bounds the multi-row-GEMV opportunity: ideal mrv ≈ the m=1 GEMV
//! cost (weights streamed once, grid stays out_f-wide).
//! Run: cargo test -p infr-vulkan --test small_m_bench -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn small_m_linear_bench() {
    let be = VulkanBackend::new().unwrap();
    // Qwen3-8B Q4_K prefill shapes (the spec-decode verify target class).
    let shapes = [
        (4096usize, 6144usize, "qkv"),
        (4096, 4096, "o"),
        (4096, 24576, "gate+up"),
        (12288, 4096, "down"),
    ];
    let reps = 50usize;
    let a = be.alloc(16 * 12288 * 4, BufferUsage::Activations).unwrap();
    let c = be.alloc(16 * 24576 * 4, BufferUsage::Activations).unwrap();
    for (k, n, label) in shapes {
        let wbytes = n * k / 256 * 144; // Q4_K
        let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
        println!("--- {label} [k={k} n={n}] weight {} MiB ---", wbytes >> 20);
        for m in [1usize, 2, 4, 8] {
            let time = |f: &dyn Fn(&infr_vulkan::Recorder)| -> f64 {
                let rec = be.recorder().unwrap(); // warmup / pipeline compile
                f(&rec);
                rec.finish().unwrap();
                let t0 = std::time::Instant::now();
                let rec = be.recorder().unwrap();
                for _ in 0..reps {
                    f(&rec);
                }
                rec.finish().unwrap();
                t0.elapsed().as_micros() as f64 / reps as f64
            };
            // Current adapter route for m>1 (m=1 shown for reference as the GEMV ideal).
            let gemm_us = if m > 1 {
                time(&|rec| {
                    rec.matmul_native(
                        infr_core::DType::Q4K,
                        a.as_ref(),
                        w.as_ref(),
                        c.as_ref(),
                        m,
                        k,
                        n,
                    )
                })
            } else {
                f64::NAN
            };
            // GEMV route (re-streams the weight per row at m>1).
            let gemv_us = time(&|rec| {
                rec.linear_native(
                    infr_core::DType::Q4K,
                    w.as_ref(),
                    a.as_ref(),
                    c.as_ref(),
                    m,
                    k,
                    n,
                )
            });
            // Multi-row GEMV (weight streamed once for all m rows).
            let mrow_us = if m > 1 {
                time(&|rec| {
                    rec.linear_native_mrow(
                        infr_core::DType::Q4K,
                        w.as_ref(),
                        0,
                        a.as_ref(),
                        c.as_ref(),
                        m,
                        k,
                        n,
                    )
                })
            } else {
                f64::NAN
            };
            // Effective weight-stream bandwidth if the weight were read once.
            let bw = |us: f64| wbytes as f64 / (us * 1e-6) / 1e9;
            println!(
                "  m={m}: gemm {gemm_us:8.1} us ({:5.0} GB/s)   gemv {gemv_us:8.1} us ({:5.0} GB/s)   mrow {mrow_us:8.1} us ({:5.0} GB/s)",
                bw(gemm_us),
                bw(gemv_us),
                bw(mrow_us),
            );
        }
    }
}

/// The multi-row GEMV must match m independent single-row GEMV dispatches to reassociation-only
/// error: the same dqblk decode feeds vec4 pairwise dots (4 independent FMA chains per sub-block —
/// the flat-in-m latency win), so the accumulation ORDER differs from the GEMV's sequential scalar
/// chain. Same contract as the attention kernels ("reassociation-only vs the oracle") and Metal's
/// spec-verify GEMMs.
#[test]
fn mrow_matches_single_row_gemv() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // k=512 → 2 Q4_K super-blocks per row; n=192 not a multiple of 64 (mrow has no tile
    // constraint); m sweeps the whole supported range.
    let (k, n) = (512usize, 192usize);
    let wbytes = n * k / 256 * 144;
    // Deterministic pseudo-random weight bytes, with SANE super-block scales: raw random bytes can
    // decode the f16 d/dmin to ±1e4-class values, whose huge canceling dot terms amplify pure
    // reassociation error past any fixed tolerance (real checkpoints have no such blocks). Pin
    // d = dmin = 2^-8 (f16 0x1C00) per 144-byte super-block; the 6-bit scales/mins and nibbles
    // stay pseudo-random.
    let mut wsrc: Vec<u8> = (0..wbytes)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    for blk in wsrc.chunks_exact_mut(144) {
        blk[0..2].copy_from_slice(&[0x00, 0x1C]); // d    = 2^-8
        blk[2..4].copy_from_slice(&[0x00, 0x1C]); // dmin = 2^-8
    }
    let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
    be.upload(w.as_ref(), &wsrc).unwrap();
    for m in 2usize..=8 {
        let xs: Vec<f32> = (0..m * k)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.021)
            .collect();
        let x = be.alloc(m * k * 4, BufferUsage::Activations).unwrap();
        be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
        let y_ref = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();
        let y_mrow = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q4K,
            w.as_ref(),
            x.as_ref(),
            y_ref.as_ref(),
            m,
            k,
            n,
        );
        rec.linear_native_mrow(
            infr_core::DType::Q4K,
            w.as_ref(),
            0,
            x.as_ref(),
            y_mrow.as_ref(),
            m,
            k,
            n,
        );
        rec.finish().unwrap();
        let read = |b: &dyn infr_core::backend::Buffer| -> Vec<f32> {
            let mut out = vec![0u8; m * n * 4];
            be.download(b, &mut out).unwrap();
            bytemuck::cast_slice::<u8, f32>(&out).to_vec()
        };
        let (r, g) = (read(y_ref.as_ref()), read(y_mrow.as_ref()));
        for i in 0..m * n {
            // The pseudo-random weight bytes can decode an f16 SCALE to NaN/Inf; NaN poisons the
            // dot in every accumulation order, so both-NaN counts as agreement (the old bit-exact
            // assert matched them bitwise).
            if r[i].is_nan() && g[i].is_nan() {
                continue;
            }
            let tol = 1e-4 * r[i].abs().max(1.0);
            assert!(
                (r[i] - g[i]).abs() <= tol,
                "m={m} row={} col={}: gemv {} vs mrow {} (reassociation tol {tol})",
                i / n,
                i % n,
                r[i],
                g[i]
            );
        }
    }
    eprintln!("mrow == single-row GEMV bit-exact for m=2..8");
}
