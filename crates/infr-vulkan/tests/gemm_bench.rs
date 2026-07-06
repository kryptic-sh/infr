//! Serialized micro-bench of the Q4_K prefill GEMM variants at qwen35-in-proj shape
//! ([512,1024]×[1024,6144]): mmq (dp4a int8), native 64×64 coopmat, native 8-warp warptile.
//! WAW hazards on the shared output buffer serialize the repeated dispatches, so wall/REPS is the
//! per-GEMM cost. Weights are zeros (perf only). Run: cargo test -p infr-vulkan --test gemm_bench -- --ignored --nocapture

use infr_core::backend::{Backend, Buffer, BufferUsage};
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

/// Sum the real qwen3-0.6B Q8_0 prefill GEMM inventory (m=512) per kernel variant — the pp512
/// sweep gap is 84% GEMM time, and the suspect is narrow-n occupancy on the warp tile (n=1024 →
/// 32 workgroups on a 96-CU part). Prints per-shape µs + effective TFLOPS for warp vs native64.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn qwen3_gemm_inventory_bench() {
    let be = VulkanBackend::new().unwrap();
    let m = 512usize;
    // (k, n, count/layer-set): q, k+v, o, gate+up (fused), down — 28 layers.
    let shapes = [
        (1024usize, 2048usize, 28usize), // q
        (1024, 1024, 56),                // k, v
        (2048, 1024, 28),                // o
        (1024, 6144, 28),                // gate+up (combined)
        (3072, 1024, 28),                // down
    ];
    let a = be.alloc(m * 3072 * 4, BufferUsage::Activations).unwrap();
    let c = be.alloc(m * 6144 * 4, BufferUsage::Activations).unwrap();
    for variant in ["warp", "native64"] {
        if variant == "native64" {
            std::env::set_var("INFR_NO_GEMM_WARP", "1");
        }
        let mut total = 0f64;
        for (k, n, cnt) in shapes {
            let w = be.alloc(n * k / 32 * 34, BufferUsage::Weights).unwrap();
            let rec = be.recorder().unwrap(); // warmup (pipeline compile)
            rec.matmul_native(
                infr_core::DType::Q8_0,
                a.as_ref(),
                w.as_ref(),
                c.as_ref(),
                m,
                k,
                n,
            );
            rec.finish().unwrap();
            let reps = 20usize;
            let t0 = std::time::Instant::now();
            let rec = be.recorder().unwrap();
            for _ in 0..reps {
                rec.matmul_native(
                    infr_core::DType::Q8_0,
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
            let tflops = (2.0 * m as f64 * k as f64 * n as f64) / us / 1e6;
            println!(
                "[{variant:>8}] [{k}x{n}] {us:8.1} us  {tflops:5.1} TF  ×{cnt} = {:.2} ms",
                us * cnt as f64 / 1e3
            );
            total += us * cnt as f64 / 1e3;
        }
        println!("[{variant:>8}] qwen3-0.6B m=512 GEMM total: {total:.1} ms\n");
        std::env::remove_var("INFR_NO_GEMM_WARP");
    }
}

/// The kernel-tuning harness: the real 8B Q4_K prefill shapes (m=512) on the warp kernel —
/// these run at the kernel's ceiling (~33-36 TF vs llama.cpp's ~45-50 on the same shapes), so
/// any micro-arch change shows here directly. Serialized reps = per-dispatch latency.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn qwen3_8b_gemm_shapes_bench() {
    let be = VulkanBackend::new().unwrap();
    let m = 512usize;
    let shapes = [
        (4096usize, 6144usize, "qkv"),
        (4096, 4096, "o"),
        (4096, 24576, "gate+up"),
        (12288, 4096, "down"),
    ];
    let a = be.alloc(m * 12288 * 4, BufferUsage::Activations).unwrap();
    let a16 = be.alloc(m * 12288 * 2, BufferUsage::Activations).unwrap();
    let c = be.alloc(m * 24576 * 4, BufferUsage::Activations).unwrap();
    for (k, n, label) in shapes {
        let w = be.alloc(n * k / 256 * 144, BufferUsage::Weights).unwrap();
        for f16a in [false, true] {
            let run = |reps: usize| {
                let rec = be.recorder().unwrap();
                for _ in 0..reps {
                    if f16a {
                        rec.store_f16(a.as_ref(), a16.as_ref(), m * k, 0);
                        rec.matmul_native_f16a(
                            infr_core::DType::Q4K,
                            a16.as_ref(),
                            w.as_ref(),
                            0,
                            c.as_ref(),
                            m,
                            k,
                            n,
                        );
                    } else {
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
                }
                rec.finish().unwrap();
            };
            run(1);
            let reps = 10usize;
            let t0 = std::time::Instant::now();
            run(reps);
            let us = t0.elapsed().as_micros() as f64 / reps as f64;
            let tflops = (2.0 * m as f64 * k as f64 * n as f64) / us / 1e6;
            let tag = if f16a { "f16a" } else { "f32 " };
            println!("[{label:>8}] [{k}x{n}] {tag} {us:8.1} us  {tflops:5.1} TF");
        }
    }
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

/// Isolated decode-attention kernel A/B at deep context (qwen3-0.6b dims, kv=8000): the push-const
/// split, the params-driven dyn split, and the self-chunking dynac variant. Hunts the seam-vs-
/// bespoke deep-decode gap.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn decode_attn_variants_bench() {
    let be = VulkanBackend::new().unwrap();
    let (nh, nkv, hd) = (16usize, 8usize, 128usize);
    let kv_len = 8000usize;
    let cap = 8065usize;
    let q = be.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
    let kc = be
        .alloc(cap * nkv * hd * 2, BufferUsage::Activations)
        .unwrap();
    let vc = be
        .alloc(cap * nkv * hd * 2, BufferUsage::Activations)
        .unwrap();
    let o = be.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
    let params = be.alloc(8, BufferUsage::Activations).unwrap();
    be.upload(
        params.as_ref(),
        bytemuck::cast_slice(&[kv_len as u32 - 1, kv_len as u32]),
    )
    .unwrap();
    let reps = 200usize;

    let run = |name: &str, f: &dyn Fn(&infr_vulkan::Recorder)| {
        let rec = be.recorder().unwrap();
        f(&rec);
        rec.finish().unwrap();
        let t0 = std::time::Instant::now();
        let rec = be.recorder().unwrap();
        for _ in 0..reps {
            f(&rec);
        }
        rec.finish().unwrap();
        println!(
            "{name:>22}: {:8.1} us/op",
            t0.elapsed().as_micros() as f64 / reps as f64
        );
    };

    // static split (bespoke-style): adaptive chunk for kv=8000
    let chunk = (kv_len / 32).clamp(64, 512);
    let n_chunks = kv_len.div_ceil(chunk);
    let pm = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc = be
        .alloc(nh * n_chunks * hd * 4, BufferUsage::Activations)
        .unwrap();
    run("static split c250", &|rec| {
        rec.attention_kv_split(
            q.as_ref(),
            kc.as_ref(),
            vc.as_ref(),
            o.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            1,          // rows (decode shape)
            kv_len - 1, // pos of the single query row
            kv_len,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            0.0,
            0,
            None,  // canvas_lo
            false, // k f16
            false, // v f16
            0,     // cap (unused for f16)
            false, // batched: decode shape stays on the per-row grid
        );
    });
    // dyn split, same chunks
    run("dyn split c250", &|rec| {
        rec.attention_kv_split_dyn(
            q.as_ref(),
            kc.as_ref(),
            vc.as_ref(),
            o.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            params.as_ref(),
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            0.0,
            0,
        );
    });
    // dynac: baked min chunk 64, capacity-sized scratch (the seam's config)
    let cap_chunks = cap.div_ceil(64);
    let pm2 = be
        .alloc(nh * cap_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl2 = be
        .alloc(nh * cap_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc2 = be
        .alloc(nh * cap_chunks * hd * 4, BufferUsage::Activations)
        .unwrap();
    let args = be.alloc(16, BufferUsage::Activations).unwrap();
    run("dynac cap126", &|rec| {
        rec.attn_live_prologue(params.as_ref(), args.as_ref(), nh, 64, 0);
        rec.attention_kv_split_dynac(
            q.as_ref(),
            kc.as_ref(),
            vc.as_ref(),
            o.as_ref(),
            pm2.as_ref(),
            pl2.as_ref(),
            pacc2.as_ref(),
            params.as_ref(),
            args.as_ref(),
            nh,
            nkv,
            hd,
            64,
            cap_chunks,
            0.0,
            0,
            false, // f16 KV cache
            0,     // cap (unused for f16)
        );
    });
    // dynac with a TIGHT bake (capacity == live): isolates the dead-workgroup/scan cost from the
    // SELF_CHUNK in-kernel logic cost.
    run("dynac tight c250", &|rec| {
        rec.attn_live_prologue(params.as_ref(), args.as_ref(), nh, chunk, 0);
        rec.attention_kv_split_dynac(
            q.as_ref(),
            kc.as_ref(),
            vc.as_ref(),
            o.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            params.as_ref(),
            args.as_ref(),
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            0.0,
            0,
            false, // f16 KV cache
            0,     // cap (unused for f16)
        );
    });
    // dyn split with chunk=64 all live (the earlier env-sweep shape)
    let n64 = kv_len.div_ceil(64);
    run("dyn split c64", &|rec| {
        rec.attention_kv_split_dyn(
            q.as_ref(),
            kc.as_ref(),
            vc.as_ref(),
            o.as_ref(),
            pm2.as_ref(),
            pl2.as_ref(),
            pacc2.as_ref(),
            params.as_ref(),
            nh,
            nkv,
            hd,
            64,
            n64,
            0.0,
            0,
        );
    });
}

/// DiffusionGemma slice 6: probe how much of `matmul_mmq_experts`'s cost at DG's shapes is
/// GENUINE compute vs the fixed worst-case grid (`rows` bound = canvas tokens = 256, so
/// `gx = ceil(rows/64)*(n/64)` is the SAME regardless of the real per-expert counts — early-exit
/// workgroups are supposed to be nearly free per docs/PERF.md's class-4 precedent). Varies ONLY
/// the `rows` bound argument (grid-sizing only, never read by the shader for real row ranges)
/// against a FIXED, realistic packed layout (128 experts, 2048 pairs, ~16 rows/expert average —
/// the canvas-256/n_used-8 arithmetic) to isolate "cost of the bound" from "cost of the real
/// work". If shrinking the bound doesn't shrink wall time, early-exit workgroups really are free
/// and an adaptive (indirect-dispatch) bound isn't worth building. If it does, the delta between
/// `bound=256` (today's production grid) and `bound=<realistic max>` is the ceiling an adaptive
/// bound could recover.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn moe_expert_grid_bound_bench() {
    let be = VulkanBackend::new().unwrap();
    let (ne, nff, n_expert, n_used, tokens) = (2816usize, 704usize, 128usize, 8usize, 256usize);
    let n_pairs = tokens * n_used; // 2048
    let npad = n_pairs.div_ceil(64) * 64 + 64;
    let reps = 30usize;

    // Packed activation buffers: gate_up reads k=ne=2816 (qa/dact/sact), down reads its OWN
    // k=nff=704 packed activations (dqa/dda) — separate buffers, correctly strided, matching the
    // real two-quantization-layer pipeline (`quant_q8_gather` then `quant_q8` on the FFN act).
    let qa = be.alloc(npad * ne, BufferUsage::Activations).unwrap();
    let dact = be
        .alloc(npad * (ne / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let sact = be
        .alloc(npad * (ne / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let dqa = be.alloc(npad * nff, BufferUsage::Activations).unwrap();
    let dda = be
        .alloc(npad * (nff / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let gu_c = be
        .alloc(npad * (2 * nff) * 4, BufferUsage::Activations)
        .unwrap();
    let down_c = be.alloc(npad * ne * 4, BufferUsage::Activations).unwrap();

    // Weight banks: Q4_K gate_up [n_expert, 2*nff, ne] (144B/256-elem block), Q5_0 down
    // [n_expert, ne, nff] (22B/32-elem block). Contents are zeros — perf only, no golden check.
    let gu_w = be
        .alloc(
            n_expert * (2 * nff) * (ne / 256) * 144,
            BufferUsage::Weights,
        )
        .unwrap();
    let down_w = be
        .alloc(n_expert * ne * (nff / 32) * 22, BufferUsage::Weights)
        .unwrap();

    // counts/offsets uploader for a given per-expert distribution (must sum to n_pairs and every
    // entry must be <= every `bound` this distribution is tested against, or the run under-covers
    // its own segment — fine for a perf-only probe, but keep the invariant so timings stay honest).
    let upload_dist = |counts_v: &[u32]| -> (
        Box<dyn infr_core::backend::Buffer>,
        Box<dyn infr_core::backend::Buffer>,
    ) {
        assert_eq!(counts_v.len(), n_expert);
        assert_eq!(counts_v.iter().sum::<u32>() as usize, n_pairs);
        let mut offsets_v = vec![0u32; n_expert];
        let mut acc = 0u32;
        for e in 0..n_expert {
            offsets_v[e] = acc;
            acc += counts_v[e];
        }
        let counts = be.alloc(n_expert * 4, BufferUsage::Activations).unwrap();
        let offsets = be.alloc(n_expert * 4, BufferUsage::Activations).unwrap();
        be.upload(counts.as_ref(), bytemuck::cast_slice(counts_v))
            .unwrap();
        be.upload(offsets.as_ref(), bytemuck::cast_slice(&offsets_v))
            .unwrap();
        (counts, offsets)
    };

    // 16 each, sum 2048.
    let balanced: Vec<u32> = vec![(n_pairs / n_expert) as u32; n_expert];
    // Skewed: one "hot" expert takes 150 of the 256 tokens' assignments (near the hard ceiling —
    // a top-k router picks each expert at most once per token, so any expert's count is bounded by
    // `tokens`=256, never `n_pairs`=2048; see matmul_mmq_experts' doc), the rest share the
    // remainder — stresses the same bound question under real imbalance instead of the average.
    let hot = 150u32;
    let mut skewed = vec![(n_pairs as u32 - hot) / (n_expert as u32 - 1); n_expert];
    skewed[0] = hot;
    {
        let sum: u32 = skewed.iter().sum();
        skewed[n_expert - 1] += n_pairs as u32 - sum; // fix up rounding remainder
    }

    // `gemm` takes (rec, bound, counts, offsets) — the caller supplies the distribution buffers
    // so this closure only knows the GEMM shape (gate_up vs down), not any specific distribution.
    #[allow(clippy::type_complexity)]
    let run =
        |label: &str, gemm: &dyn Fn(&infr_vulkan::Recorder, usize, &dyn Buffer, &dyn Buffer)| {
            for (dist_name, counts_v) in [("balanced~16", &balanced), ("skewed", &skewed)] {
                let (counts, offsets) = upload_dist(counts_v);
                let max_real = *counts_v.iter().max().unwrap();
                for &bound in &[64usize, 128, 192, 256] {
                    if (bound as u32) < max_real {
                        continue; // would silently truncate this distribution's hottest expert
                    }
                    let rec = be.recorder().unwrap();
                    gemm(&rec, bound, counts.as_ref(), offsets.as_ref()); // warmup (pipeline compile)
                    rec.finish().unwrap();
                    let t0 = std::time::Instant::now();
                    let rec = be.recorder().unwrap();
                    for _ in 0..reps {
                        gemm(&rec, bound, counts.as_ref(), offsets.as_ref());
                    }
                    rec.finish().unwrap();
                    let us = t0.elapsed().as_micros() as f64 / reps as f64;
                    println!(
                    "[{label:>11}] dist={dist_name:>11} (max={max_real:3}) bound={bound:3}: {us:7.1} us"
                );
                }
            }
        };

    run("gate_up", &|rec, bound, counts, offsets| {
        rec.matmul_mmq_experts(
            infr_core::DType::Q4K,
            "expert_gateup",
            qa.as_ref(),
            dact.as_ref(),
            Some(sact.as_ref()),
            gu_w.as_ref(),
            0,
            ne, // stride = k (per-expert weight stride, elements)
            counts,
            offsets,
            gu_c.as_ref(),
            bound,
            ne,
            2 * nff,
            n_expert,
        );
    });
    run("down", &|rec, bound, counts, offsets| {
        rec.matmul_mmq_experts(
            infr_core::DType::Q5_0,
            "expert_down",
            dqa.as_ref(),
            dda.as_ref(),
            None, // Q5_0 is symmetric — no min term
            down_w.as_ref(),
            0,
            nff, // stride = k
            counts,
            offsets,
            down_c.as_ref(),
            bound,
            nff,
            ne,
            n_expert,
        );
    });
    // A BN=128 tile (halving down's 44 N-tiles to 22, matching gate_up's grid granularity) was
    // probed here at both TN=8 (doubles the per-thread accumulator: 1.3-1.6ms, a clear LOSS vs
    // baseline's ~1.25-1.37ms — register-pressure/occupancy cost, the exact class-3 risk
    // docs/PERF.md warns "bigger tiles often lose" for) and TN=4/THREADS=512 (register-neutral:
    // ~1.3-1.35ms, a wash/marginal loss within noise). Neither improved on the baseline BN=64
    // tile, so down's lower TFLOPS-vs-gate_up efficiency (measured via INFR_PROF2: ~8.1 vs
    // ~10.2 TFLOPS at production counts) is a K-depth ceiling (k=nff=704 → only 22 BLK=32
    // iterations, less loop depth to amortize fixed per-iteration cost), not a tile/occupancy
    // config bug — reverted both variants per docs/PERF.md's "a measured wash gets reverted"
    // rule rather than landing a wash.
}
