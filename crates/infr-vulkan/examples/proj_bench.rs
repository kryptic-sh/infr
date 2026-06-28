//! Time matmul_proj (q4) at the qwen3-0.6b projection shapes for small M (32k chunk) vs large M
//! (4k chunk), to see per-shape efficiency and where small-M prefill loses.
use infr_core::backend::BufferUsage;
use infr_core::Backend;
use infr_vulkan::VulkanBackend;

/// A unified quant weight on the GPU: (quants, scales, mins) buffers.
type QWeight = (
    Box<dyn infr_core::backend::Buffer>,
    Box<dyn infr_core::backend::Buffer>,
    Box<dyn infr_core::backend::Buffer>,
);

fn main() {
    let be = VulkanBackend::new().unwrap();
    // (label, k, n) — q/k/v/o/gate+up(fused)/down for qwen3-0.6b
    let shapes = [
        ("q   ", 1024usize, 2048usize),
        ("k   ", 1024, 1024),
        ("v   ", 1024, 1024),
        ("o   ", 2048, 1024),
        ("gu  ", 1024, 6144),
        ("down", 3072, 1024),
        ("qkv ", 1024, 4096), // hypothetical fused q+k+v
    ];
    // cold=true rotates through DISTINCT weight buffers (defeats L2/Infinity-Cache reuse, like real
    // prefill walking 16 layers of different weights); cold=false reuses one (cache-resident).
    for &cold in &[false, true] {
        for &m in &[512usize, 2048] {
            println!("=== m={m} cold_weights={cold} ===");
            let mpad = m.div_ceil(64) * 64;
            for &(label, k, n) in &shapes {
                let reps = 50;
                let n_w = if cold { reps } else { 1 };
                // q4 weights: 8 nibbles/u32, per-32 f16 scale/min
                let mk = |salt: u32| {
                    let packed = vec![0x33u32.wrapping_add(salt); n * k / 8];
                    let scales = vec![half::f16::from_f32(0.02).to_bits(); n * k / 32];
                    let mins = vec![half::f16::from_f32(-0.1).to_bits(); n * k / 32];
                    (
                        be.upload_weight_bytes(bytemuck::cast_slice(&packed))
                            .unwrap(),
                        be.upload_weight_bytes(bytemuck::cast_slice(&scales))
                            .unwrap(),
                        be.upload_weight_bytes(bytemuck::cast_slice(&mins)).unwrap(),
                    )
                };
                let ws: Vec<_> = (0..n_w).map(|i| mk(i as u32)).collect();
                let ba = be.alloc(mpad * k * 4, BufferUsage::Activations).unwrap();
                be.upload(ba.as_ref(), &vec![0u8; mpad * k * 4]).unwrap();
                let bc = be.alloc(mpad * n * 4, BufferUsage::Activations).unwrap();
                let run = || {
                    let rec = be.recorder().unwrap();
                    for r in 0..reps {
                        let (bwq, bs, bmn) = &ws[r % n_w];
                        rec.matmul_proj(
                            ba.as_ref(),
                            bwq.as_ref(),
                            bs.as_ref(),
                            bmn.as_ref(),
                            bc.as_ref(),
                            m,
                            k,
                            n,
                            4,
                            5,
                        );
                    }
                    rec.finish().unwrap();
                };
                run(); // warm
                let t = std::time::Instant::now();
                run();
                let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
                let gflops = 2.0 * m as f64 * k as f64 * n as f64 / (us * 1e3);
                println!("  {label} k{k} n{n}: {us:7.1} us/op  {gflops:6.1} GFLOP/s");
            }
        }
    }
    // Decisive test for qkv fusion: time q→k→v as a SEQUENCE into 3 different buffers (no WAW
    // barrier between them → they pipeline, as in real prefill) vs the single fused n=4096 GEMM.
    // If sequence ≈ fused, fusion is pointless (the microbench's "2×" was self-barrier serialization).
    println!("=== fused-qkv vs pipelined-separate (m=512) ===");
    {
        let (m, k, q_n, kv_n) = (512usize, 1024usize, 2048usize, 1024usize);
        let mpad = m.div_ceil(64) * 64;
        let mkw = |n: usize| {
            (
                be.upload_weight_bytes(bytemuck::cast_slice(&vec![0x33u32; n * k / 8]))
                    .unwrap(),
                be.upload_weight_bytes(bytemuck::cast_slice(&vec![
                    half::f16::from_f32(0.02)
                        .to_bits();
                    n * k / 32
                ]))
                .unwrap(),
                be.upload_weight_bytes(bytemuck::cast_slice(&vec![
                    half::f16::from_f32(-0.1)
                        .to_bits();
                    n * k / 32
                ]))
                .unwrap(),
            )
        };
        let (wq, wk, wv, wqkv) = (mkw(q_n), mkw(kv_n), mkw(kv_n), mkw(q_n + 2 * kv_n));
        let ba = be.alloc(mpad * k * 4, BufferUsage::Activations).unwrap();
        be.upload(ba.as_ref(), &vec![0u8; mpad * k * 4]).unwrap();
        let bq = be.alloc(mpad * q_n * 4, BufferUsage::Activations).unwrap();
        let bk = be.alloc(mpad * kv_n * 4, BufferUsage::Activations).unwrap();
        let bv = be.alloc(mpad * kv_n * 4, BufferUsage::Activations).unwrap();
        let bqkv = be
            .alloc(mpad * (q_n + 2 * kv_n) * 4, BufferUsage::Activations)
            .unwrap();
        let reps = 50;
        let pj = |rec: &infr_vulkan::Recorder,
                  w: &QWeight,
                  c: &dyn infr_core::backend::Buffer,
                  n: usize| {
            rec.matmul_proj(
                ba.as_ref(),
                w.0.as_ref(),
                w.1.as_ref(),
                w.2.as_ref(),
                c,
                m,
                k,
                n,
                4,
                5,
            );
        };
        for (name, f) in [("pipelined q+k+v", 0u8), ("fused qkv      ", 1u8)] {
            let run = || {
                let rec = be.recorder().unwrap();
                for _ in 0..reps {
                    if f == 0 {
                        pj(&rec, &wq, bq.as_ref(), q_n);
                        pj(&rec, &wk, bk.as_ref(), kv_n);
                        pj(&rec, &wv, bv.as_ref(), kv_n);
                    } else {
                        pj(&rec, &wqkv, bqkv.as_ref(), q_n + 2 * kv_n);
                    }
                }
                rec.finish().unwrap();
            };
            run();
            let t = std::time::Instant::now();
            run();
            let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            println!("  {name}: {us:7.1} us per q+k+v");
        }
    }
}
