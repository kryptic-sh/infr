//! Serialized micro-bench of the Q4_K prefill GEMM variants at qwen35-in-proj shape
//! ([512,1024]×[1024,6144]): mmq (dp4a int8), native 64×64 coopmat, native 8-warp warptile.
//! WAW hazards on the shared output buffer serialize the repeated dispatches, so wall/REPS is the
//! per-GEMM cost. Weights are zeros (perf only). Run: cargo test -p infr-vulkan --test gemm_bench -- --ignored --nocapture

use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn q4k_gemm_variants_bench() {
    let be = VulkanBackend::new().unwrap();
    let (m, k, n) = (512usize, 1024usize, 6144usize);
    let reps = 20usize;
    // Q4_K: 144 bytes / 256 elems
    let wbytes = n * k / 256 * 144;
    let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
    let a = be.alloc(m * k * 4, BufferUsage::Activations).unwrap();
    let c = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();
    // mmq activation quant buffers
    let nblk = k / 32;
    let qa = be.alloc(m * k, BufferUsage::Activations).unwrap();
    let dact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
    let sact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();

    let run = |name: &str, f: &dyn Fn(&infr_vulkan::Recorder)| {
        // warmup (pipeline compile)
        let rec = be.recorder().unwrap();
        f(&rec);
        rec.finish().unwrap();
        let t0 = std::time::Instant::now();
        let rec = be.recorder().unwrap();
        for _ in 0..reps {
            f(&rec);
        }
        rec.finish().unwrap();
        let us = t0.elapsed().as_micros() as f64 / reps as f64;
        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / (us * 1e3);
        println!("{name:>10}: {us:8.1} us/GEMM  ({gflops:.0} GFLOP/s)");
    };

    run("mmq", &|rec| {
        rec.quant_q8(a.as_ref(), qa.as_ref(), dact.as_ref(), sact.as_ref(), m, k);
        rec.matmul_mmq_q4k(
            qa.as_ref(),
            dact.as_ref(),
            sact.as_ref(),
            w.as_ref(),
            0,
            c.as_ref(),
            m,
            k,
            n,
        );
    });
    std::env::set_var("INFR_NO_GEMM_WARP", "1");
    run("native64", &|rec| {
        rec.matmul_native(
            infr_core::DType::Q4K,
            a.as_ref(),
            w.as_ref(),
            c.as_ref(),
            m,
            k,
            n,
        );
    });
    std::env::remove_var("INFR_NO_GEMM_WARP");
    run("warp", &|rec| {
        rec.matmul_native(
            infr_core::DType::Q4K,
            a.as_ref(),
            w.as_ref(),
            c.as_ref(),
            m,
            k,
            n,
        );
    });
}

/// Sum the real qwen35 prefill GEMM inventory (one 512-row chunk) on the warp kernel — ground
/// truth for how much of the chunk's execute time is genuinely GEMM.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn qwen35_gemm_inventory_bench() {
    let be = VulkanBackend::new().unwrap();
    let m = 512usize;
    // (k, n, count): DeltaNet-layer in-proj/out/FFN ×18, attention-layer qkv/out/FFN ×6
    let shapes = [
        (1024usize, 6144usize, 18usize), // qkvz in-proj
        (2048, 1024, 18),                // deltanet out-proj
        (1024, 4096, 24),                // FFN gate+up (both layer kinds)
        (2048, 1024, 24),                // FFN down
        (1024, 3072, 6),                 // attn qkv(+gate)
        (2048, 1024, 6),                 // attn out
    ];
    let a = be.alloc(m * 2048 * 4, BufferUsage::Activations).unwrap();
    let c = be.alloc(m * 6144 * 4, BufferUsage::Activations).unwrap();
    let mut total = 0f64;
    for (k, n, cnt) in shapes {
        let w = be.alloc(n * k / 256 * 144, BufferUsage::Weights).unwrap();
        // warmup
        let rec = be.recorder().unwrap();
        rec.matmul_native(
            infr_core::DType::Q4K,
            a.as_ref(),
            w.as_ref(),
            c.as_ref(),
            m,
            k,
            n,
        );
        rec.finish().unwrap();
        let reps = 10usize;
        let t0 = std::time::Instant::now();
        let rec = be.recorder().unwrap();
        for _ in 0..reps {
            rec.matmul_native(
                infr_core::DType::Q4K,
                a.as_ref(),
                w.as_ref(),
                c.as_ref(),
                m,
                k,
                n,
            );
        }
        rec.finish().unwrap();
        let us = t0.elapsed().as_micros() as f64 / reps as f64;
        println!(
            "[{k}x{n}] {us:8.1} us  ×{cnt} = {:.1} ms",
            us * cnt as f64 / 1e3
        );
        total += us * cnt as f64 / 1e3;
    }
    println!("qwen35 512-row chunk GEMM total: {total:.1} ms");
}
