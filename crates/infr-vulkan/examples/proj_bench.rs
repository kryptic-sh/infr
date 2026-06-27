//! Time matmul_proj (q4) at the qwen3-0.6b projection shapes for small M (32k chunk) vs large M
//! (4k chunk), to see per-shape efficiency and where small-M prefill loses.
use infr_core::backend::BufferUsage;
use infr_core::Backend;
use infr_vulkan::VulkanBackend;

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
    for &m in &[512usize, 2048] {
        println!("=== m={m} ===");
        let mpad = m.div_ceil(64) * 64;
        for &(label, k, n) in &shapes {
            // q4 weights: 8 nibbles/u32, per-32 f16 scale/min
            let packed = vec![0x33u32; n * k / 8];
            let scales = vec![half::f16::from_f32(0.02).to_bits(); n * k / 32];
            let mins = vec![half::f16::from_f32(-0.1).to_bits(); n * k / 32];
            let a = vec![0u8; mpad * k * 4];
            let bwq = be
                .upload_weight_bytes(bytemuck::cast_slice(&packed))
                .unwrap();
            let bs = be
                .upload_weight_bytes(bytemuck::cast_slice(&scales))
                .unwrap();
            let bmn = be.upload_weight_bytes(bytemuck::cast_slice(&mins)).unwrap();
            let ba = be.alloc(a.len(), BufferUsage::Activations).unwrap();
            be.upload(ba.as_ref(), &a).unwrap();
            let bc = be.alloc(mpad * n * 4, BufferUsage::Activations).unwrap();
            let reps = 50;
            // warm
            let rec = be.recorder().unwrap();
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
            rec.finish().unwrap();
            let t = std::time::Instant::now();
            let rec = be.recorder().unwrap();
            for _ in 0..reps {
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
            let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let gflops = 2.0 * m as f64 * k as f64 * n as f64 / (us * 1e3);
            println!("  {label} k{k} n{n}: {us:7.1} us/op  {gflops:6.1} GFLOP/s");
        }
    }
}
