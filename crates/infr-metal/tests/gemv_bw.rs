//! Probe: achieved memory bandwidth of the quantized GEMV kernels at decode shapes (m=1).
//! Temporary evidence for kernel-efficiency work — run with
//! `cargo test -p infr-metal --release --test gemv_bw -- --ignored --nocapture`.
#![cfg(target_os = "macos")]

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc};
use infr_metal::MetalBackend;

fn lcg_bytes(mut seed: u32, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as u8
        })
        .collect()
}

fn synth_q4k(n_elem: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 144];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());
        blk[2..4].copy_from_slice(&half::f16::from_f32(0.10).to_le_bytes());
        blk[4..144].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 140));
        out.extend_from_slice(&blk);
    }
    out
}

fn synth_q6k(n_elem: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 210];
        blk[0..208].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 208));
        blk[208..210].copy_from_slice(&half::f16::from_f32(0.03).to_le_bytes());
        out.extend_from_slice(&blk);
    }
    out
}

fn bench(dtype: DType, wbytes: Vec<u8>, in_f: usize, out_f: usize, bpw: f64, label: &str) {
    let be = MetalBackend::new().unwrap();
    let m = 1usize;
    let xs: Vec<f32> = (0..in_f).map(|i| (i % 7) as f32 * 0.01).collect();

    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], dtype));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
    });

    let xb = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
    let wb = be.alloc(wbytes.len(), BufferUsage::Weights).unwrap();
    be.upload(wb.as_ref(), &wbytes).unwrap();
    let ob = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();

    let mut binds = Bindings::new();
    binds.bind(x, xb.as_ref());
    binds.bind(w, wb.as_ref());
    binds.bind(dst, ob.as_ref());
    let plan = be.compile(&g).unwrap();

    // First execute builds + caches the factored weight; then timed reps.
    be.execute(plan.as_ref(), &binds).unwrap();
    let reps = 30;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        be.execute(plan.as_ref(), &binds).unwrap();
    }
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    let bytes = (out_f * in_f) as f64 * bpw / 8.0;
    println!(
        "{label}: {out_f}x{in_f} -> {:.3} ms/dispatch, {:.1} GB/s (stream {:.1} MB)",
        dt * 1e3,
        bytes / dt / 1e9,
        bytes / 1e6
    );
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn gemv_bandwidth() {
    // lm_head shape (Qwen3-0.6B): 151936x1024 Q6K -> quik6, 8.125 bpw
    bench(
        DType::Q6K,
        synth_q6k(151936 * 1024, 1),
        1024,
        151936,
        8.125,
        "quik6",
    );
    // Same shape as Q4K -> quik4, 6.125 bpw
    bench(
        DType::Q4K,
        synth_q4k(151936 * 1024, 2),
        1024,
        151936,
        6.125,
        "quik4",
    );
    // FFN shape: 3072x1024 Q4K (small stream: launch overhead visible)
    bench(
        DType::Q4K,
        synth_q4k(3072 * 1024, 3),
        1024,
        3072,
        6.125,
        "quik4-ffn",
    );
}
