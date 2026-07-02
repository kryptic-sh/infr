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

/// Per-op serialization floor: a chain of small hazard-dependent dispatches (each reads the
/// previous one's output → global barrier each). wall/ops ≈ the fixed bubble every seam op pays
/// on top of its kernel time — the number that says how much op-count reduction / overlap is worth.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn chained_op_bubble_bench() {
    let be = VulkanBackend::new().unwrap();
    let n = 512usize * 1024; // one chunk's hidden activations
    let a = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    let b = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    let w = be.alloc(1024 * 4, BufferUsage::Activations).unwrap();
    for ops in [50usize, 400] {
        // warmup
        let rec = be.recorder().unwrap();
        rec.rmsnorm(a.as_ref(), w.as_ref(), b.as_ref(), 512, 1024, 1e-6);
        rec.finish().unwrap();
        let t0 = std::time::Instant::now();
        let rec = be.recorder().unwrap();
        for i in 0..ops {
            // ping-pong a→b→a…: every dispatch RAW-depends on the previous one
            let (x, y) = if i % 2 == 0 { (&a, &b) } else { (&b, &a) };
            rec.rmsnorm(x.as_ref(), w.as_ref(), y.as_ref(), 512, 1024, 1e-6);
        }
        rec.finish().unwrap();
        let us = t0.elapsed().as_micros() as f64;
        println!(
            "{ops} chained rmsnorm(512x1024): {:.1} us total, {:.1} us/op",
            us,
            us / ops as f64
        );
    }
}

/// Real isolated cost of the two remaining big prefill ops at qwen35 shapes.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn qwen35_dn_attn_bench() {
    let be = VulkanBackend::new().unwrap();
    let rows = 512usize;
    let (nv, nk, kd, vd) = (16usize, 16usize, 128usize, 128usize);
    let q = be
        .alloc(rows * nk * kd * 4, BufferUsage::Activations)
        .unwrap();
    let k = be
        .alloc(rows * nk * kd * 4, BufferUsage::Activations)
        .unwrap();
    let v = be
        .alloc(rows * nv * vd * 4, BufferUsage::Activations)
        .unwrap();
    let b = be.alloc(rows * nv * 4, BufferUsage::Activations).unwrap();
    let al = be.alloc(rows * nv * 4, BufferUsage::Activations).unwrap();
    let ac = be.alloc(nv * 4, BufferUsage::Weights).unwrap();
    let dt = be.alloc(nv * 4, BufferUsage::Weights).unwrap();
    let st = be
        .alloc(nv * kd * vd * 4, BufferUsage::Activations)
        .unwrap();
    let o = be
        .alloc(rows * nv * vd * 4, BufferUsage::Activations)
        .unwrap();
    let reps = 10usize;
    let rec = be.recorder().unwrap();
    rec.deltanet_chunked(
        q.as_ref(),
        k.as_ref(),
        v.as_ref(),
        b.as_ref(),
        al.as_ref(),
        ac.as_ref(),
        dt.as_ref(),
        st.as_ref(),
        o.as_ref(),
        rows,
        nv,
        nk,
        kd,
        vd,
        1e-6,
    );
    rec.finish().unwrap();
    let t0 = std::time::Instant::now();
    let rec = be.recorder().unwrap();
    for _ in 0..reps {
        rec.deltanet_chunked(
            q.as_ref(),
            k.as_ref(),
            v.as_ref(),
            b.as_ref(),
            al.as_ref(),
            ac.as_ref(),
            dt.as_ref(),
            st.as_ref(),
            o.as_ref(),
            rows,
            nv,
            nk,
            kd,
            vd,
            1e-6,
        );
    }
    rec.finish().unwrap();
    let us = t0.elapsed().as_micros() as f64 / reps as f64;
    println!(
        "deltanet_chunked rows=512: {us:.1} us/op  ×18 = {:.1} ms/chunk",
        us * 18.0 / 1e3
    );

    // split variant (prep + gates + scan)
    let nchunk = rows.div_ceil(32);
    let kn = be
        .alloc(rows * nk * kd * 4, BufferUsage::Activations)
        .unwrap();
    let qn = be
        .alloc(rows * nk * kd * 4, BufferUsage::Activations)
        .unwrap();
    let dkb = be
        .alloc(nchunk * nk * 1024 * 4, BufferUsage::Activations)
        .unwrap();
    let dqb = be
        .alloc(nchunk * nk * 1024 * 4, BufferUsage::Activations)
        .unwrap();
    let bg = be
        .alloc(nchunk * nv * 32 * 4, BufferUsage::Activations)
        .unwrap();
    let gg = be
        .alloc(nchunk * nv * 32 * 4, BufferUsage::Activations)
        .unwrap();
    let split = |rec: &infr_vulkan::Recorder| {
        rec.deltanet_chunked_split(
            q.as_ref(),
            k.as_ref(),
            v.as_ref(),
            b.as_ref(),
            al.as_ref(),
            ac.as_ref(),
            dt.as_ref(),
            st.as_ref(),
            o.as_ref(),
            kn.as_ref(),
            qn.as_ref(),
            dkb.as_ref(),
            dqb.as_ref(),
            bg.as_ref(),
            gg.as_ref(),
            rows,
            nv,
            nk,
            kd,
            vd,
            1e-6,
        );
    };
    let rec = be.recorder().unwrap();
    split(&rec);
    rec.finish().unwrap();
    let t0 = std::time::Instant::now();
    let rec = be.recorder().unwrap();
    for _ in 0..reps {
        split(&rec);
    }
    rec.finish().unwrap();
    let us = t0.elapsed().as_micros() as f64 / reps as f64;
    println!(
        "deltanet_split   rows=512: {us:.1} us/op  ×18 = {:.1} ms/chunk",
        us * 18.0 / 1e3
    );

    // nonfa attention at qwen35 attn shape: rows=512, kv=822, nh=16, nkv=2, hd=256
    let (nh, nkv, hd, kv_len) = (16usize, 2usize, 256usize, 822usize);
    let mpad = 512usize;
    let kv_pad = kv_len.div_ceil(256) * 256;
    let qb = be
        .alloc(mpad * nh * hd * 2, BufferUsage::Activations)
        .unwrap();
    let kc = be
        .alloc(kv_len * nkv * hd * 2, BufferUsage::Activations)
        .unwrap();
    let vc = be
        .alloc(kv_len * nkv * hd * 2, BufferUsage::Activations)
        .unwrap();
    let at = be
        .alloc(mpad * nh * hd * 4, BufferUsage::Activations)
        .unwrap();
    let s = be
        .alloc(nh * mpad * kv_pad * 2, BufferUsage::Activations)
        .unwrap();
    let pv = be
        .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
        .unwrap();
    let rec = be.recorder().unwrap();
    rec.attention_prefill_nonfa(
        qb.as_ref(),
        kc.as_ref(),
        vc.as_ref(),
        at.as_ref(),
        s.as_ref(),
        pv.as_ref(),
        mpad,
        kv_len,
        nh,
        nkv,
        hd,
        310,
        0,
        0.0,
    );
    rec.finish().unwrap();
    let t0 = std::time::Instant::now();
    let rec = be.recorder().unwrap();
    for _ in 0..reps {
        rec.attention_prefill_nonfa(
            qb.as_ref(),
            kc.as_ref(),
            vc.as_ref(),
            at.as_ref(),
            s.as_ref(),
            pv.as_ref(),
            mpad,
            kv_len,
            nh,
            nkv,
            hd,
            310,
            0,
            0.0,
        );
    }
    rec.finish().unwrap();
    let us = t0.elapsed().as_micros() as f64 / reps as f64;
    println!(
        "nonfa attn rows=512 kv=822 hd=256: {us:.1} us/op  ×6 = {:.1} ms/chunk",
        us * 6.0 / 1e3
    );
}
