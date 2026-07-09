//! The multi-warp dp4a decode GEMV (`native_mmv_mw.comp`, wave32-native route) must match the
//! validated 64-thread `native_mmv` dp4a kernel at the real projection shapes — both quantize the
//! activation identically (quant_q8) and use the identical `dpsub` math, differing ONLY in the
//! reduction (warp-per-row subgroupAdd vs shared-tree) and lane→sub-block mapping, so the gap is
//! pure float-reassociation. Proves mmv_mw inherits native_mmv's correctness. Run:
//!   cargo test -p infr-vulkan --test mmv_mw_parity -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

fn blk_bytes(dt: DType) -> usize {
    match dt {
        DType::Q4K => 144,
        DType::Q6K => 210,
        _ => unreachable!(),
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_mw_matches_native_mmv() {
    let be = VulkanBackend::new().unwrap();
    if be.capabilities().subgroup_max != 32 {
        eprintln!("skip: device is not wave32-native (subgroup_max != 32)");
        return;
    }
    // (in_f, out_f, dtype) — the decode projection shapes mmv_mw serves.
    let shapes = [
        (2048usize, 2048usize, DType::Q4K), // q / o
        (2048, 6144, DType::Q4K),           // gate+up (fused)
        (6144, 2048, DType::Q4K),           // down
        (2048, 2048, DType::Q6K),
        (6144, 2048, DType::Q6K),   // down-q6
        (2048, 151936, DType::Q6K), // lm_head (odd out_f: 151936 % 8 != 0)
    ];
    let xmax = 6144;
    let xs: Vec<f32> = (0..xmax)
        .map(|i| ((i % 61) as f32 - 30.0) * 0.013)
        .collect();
    let x = be.alloc(xmax * 4, BufferUsage::Activations).unwrap();
    be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
    let read = |b: &dyn infr_core::backend::Buffer, n: usize| -> Vec<f32> {
        let mut out = vec![0u8; n * 4];
        be.download(b, &mut out).unwrap();
        bytemuck::cast_slice::<u8, f32>(&out).to_vec()
    };

    let mut worst = 0f32;
    for (in_f, out_f, dt) in shapes {
        let wbytes = in_f * out_f / 256 * blk_bytes(dt);
        let mut src: Vec<u8> = (0..wbytes)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        for blk in src.chunks_exact_mut(blk_bytes(dt)) {
            match dt {
                DType::Q4K => {
                    blk[0..2].copy_from_slice(&[0x00, 0x1C]); // d = 1.0 (f16)
                    blk[2..4].copy_from_slice(&[0x00, 0x18]); // dmin
                }
                DType::Q6K => blk[208..210].copy_from_slice(&[0x00, 0x1C]),
                _ => unreachable!(),
            }
        }
        let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
        be.upload(w.as_ref(), &src).unwrap();
        let nblk = in_f / 32;
        let qa = be.alloc(in_f, BufferUsage::Activations).unwrap();
        let dact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let y_ref = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        let y_mw = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();

        let rec = be.recorder().unwrap();
        rec.quant_q8(
            x.as_ref(),
            qa.as_ref(),
            dact.as_ref(),
            sact.as_ref(),
            1,
            in_f,
        );
        rec.linear_mmv(
            dt,
            w.as_ref(),
            0,
            qa.as_ref(),
            dact.as_ref(),
            sact.as_ref(),
            y_ref.as_ref(),
            in_f,
            out_f,
        );
        rec.finish().unwrap();
        let r = read(y_ref.as_ref(), out_f);
        for warps in [4u32, 8u32] {
            let rec = be.recorder().unwrap();
            rec.quant_q8(
                x.as_ref(),
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                1,
                in_f,
            );
            rec.linear_mmv_mw(
                dt,
                warps,
                w.as_ref(),
                0,
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                None,
                y_mw.as_ref(),
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            let g = read(y_mw.as_ref(), out_f);
            let mut mr = 0f32;
            for (a, b) in r.iter().zip(&g) {
                mr = mr.max((a - b).abs() / (a.abs().max(b.abs()) + 1e-6));
            }
            worst = worst.max(mr);
            println!("  {dt:?} in{in_f} out{out_f} W{warps}: max rel err {mr:.2e}");
            assert!(
                mr < 5e-3,
                "{dt:?} {in_f}x{out_f} W{warps}: mmv_mw diverges from native_mmv (rel {mr:.2e})"
            );
        }
    }
    println!("mmv_mw == native_mmv within reassociation tolerance (worst {worst:.2e})");
}
