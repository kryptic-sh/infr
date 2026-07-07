//! Cold-weight decode GEMV bandwidth at the real Qwen3-8B decode shapes. Rotates through DISTINCT
//! weight buffers so the aggregate working set exceeds the 96 MiB Infinity Cache — the reported
//! GB/s is TRUE DRAM bandwidth (the in-model INFR_PROF2 numbers for small tensors are cache-
//! contaminated). A/Bs the RM=1 grid vs the multi-output-row (RM=2/4) grid, and asserts the RM
//! path is BIT-IDENTICAL to RM=1 (per-row math is unchanged). Run:
//!   cargo test -p infr-vulkan --test decode_gemv_bw -- --ignored --nocapture
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
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn decode_gemv_bw() {
    let be = VulkanBackend::new().unwrap();
    // (label, in_f, out_f, dtype) — the Qwen3-8B decode GEMV shapes.
    let shapes = [
        ("q      ", 4096usize, 4096usize, DType::Q4K),
        ("k/v    ", 4096, 1024, DType::Q4K),
        ("qkv-fus", 4096, 6144, DType::Q4K),
        ("o      ", 4096, 4096, DType::Q4K),
        ("gate+up", 4096, 24576, DType::Q4K),
        ("down-q4", 12288, 4096, DType::Q4K),
        ("down-q6", 12288, 4096, DType::Q6K),
        ("lm_head", 4096, 151936, DType::Q6K),
    ];
    let xmax = 12288;
    let xs: Vec<f32> = (0..xmax).map(|i| ((i % 61) as f32 - 30.0) * 0.01).collect();
    let x = be.alloc(xmax * 4, BufferUsage::Activations).unwrap();
    be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();

    let read = |b: &dyn infr_core::backend::Buffer, n: usize| -> Vec<f32> {
        let mut out = vec![0u8; n * 4];
        be.download(b, &mut out).unwrap();
        bytemuck::cast_slice::<u8, f32>(&out).to_vec()
    };

    for (label, in_f, out_f, dt) in shapes {
        let wbytes = in_f * out_f / 256 * blk_bytes(dt);
        let n_w = (200usize << 20).div_ceil(wbytes).clamp(3, 24);
        let ws: Vec<_> = (0..n_w)
            .map(|s| {
                let mut src: Vec<u8> = (0..wbytes)
                    .map(|i| {
                        ((i as u32).wrapping_mul(2654435761).wrapping_add(s as u32) >> 24) as u8
                    })
                    .collect();
                for blk in src.chunks_exact_mut(blk_bytes(dt)) {
                    match dt {
                        DType::Q4K => {
                            blk[0..2].copy_from_slice(&[0x00, 0x1C]);
                            blk[2..4].copy_from_slice(&[0x00, 0x1C]);
                        }
                        DType::Q6K => blk[208..210].copy_from_slice(&[0x00, 0x1C]),
                        _ => unreachable!(),
                    }
                }
                let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
                be.upload(w.as_ref(), &src).unwrap();
                w
            })
            .collect();
        let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        let reps = (n_w * 4).max(40);

        // Bit-identity: RM=1 vs RM=2 vs RM=4 on the first weight buffer must match exactly.
        let run_once = |rm: &str| -> Vec<f32> {
            if rm == "1" {
                std::env::set_var("INFR_NO_GEMV_RM", "1");
            } else {
                std::env::remove_var("INFR_NO_GEMV_RM");
                std::env::set_var("INFR_GEMV_RM", rm);
            }
            std::env::set_var("INFR_GEMV_RM_MAXOUT", "999999");
            let rec = be.recorder().unwrap();
            rec.linear_native(dt, ws[0].as_ref(), x.as_ref(), y.as_ref(), 1, in_f, out_f);
            rec.finish().unwrap();
            read(y.as_ref(), out_f)
        };
        let base = run_once("1");
        for rm in ["2", "4"] {
            let got = run_once(rm);
            let mism = base
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(mism, 0, "{label} RM={rm}: {mism} bits differ from RM=1");
        }

        // Bandwidth A/B (cold, rotated).
        let bench = |rm: &str| -> f64 {
            if rm == "1" {
                std::env::set_var("INFR_NO_GEMV_RM", "1");
            } else {
                std::env::remove_var("INFR_NO_GEMV_RM");
                std::env::set_var("INFR_GEMV_RM", rm);
            }
            std::env::set_var("INFR_GEMV_RM_MAXOUT", "999999");
            let run = || {
                let rec = be.recorder().unwrap();
                for r in 0..reps {
                    rec.linear_native(
                        dt,
                        ws[r % n_w].as_ref(),
                        x.as_ref(),
                        y.as_ref(),
                        1,
                        in_f,
                        out_f,
                    );
                }
                rec.finish().unwrap();
            };
            run();
            let t = std::time::Instant::now();
            run();
            let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            wbytes as f64 / (us * 1e-6) / 1e9
        };
        let (g1, g2, g4) = (bench("1"), bench("2"), bench("4"));
        println!(
            "  {label} in{in_f} out{out_f} {dt:?} [{} MiB]:  RM1 {g1:5.0}  RM2 {g2:5.0}  RM4 {g4:5.0} GB/s  (best {:+.0}%)",
            wbytes >> 20,
            (g2.max(g4) / g1 - 1.0) * 100.0,
        );
    }
    std::env::remove_var("INFR_NO_GEMV_RM");
    std::env::remove_var("INFR_GEMV_RM");
    std::env::remove_var("INFR_GEMV_RM_MAXOUT");
}
