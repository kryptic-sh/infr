//! Probe: device vs host MoeFfn path at Qwen3-30B-A3B decode shapes (one token, one layer).
//! Temporary evidence for the MoE-on-device work — run with
//! `INFR_METAL_NOMOE=1` to force the host path for the comparison:
//! `cargo test -p infr-metal --release --test moe_bw -- --ignored --nocapture`.
#![cfg(target_os = "macos")]

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
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

#[test]
#[ignore = "requires a Metal GPU (evidence probe)"]
fn moe_layer_wall() {
    // Qwen3-30B-A3B MoE layer: ne=2048, n_ff_exp=768, 128 experts, 8 used.
    let (ne, n_expert, n_used, nff) = (2048usize, 128usize, 8usize, 768usize);
    let be = MetalBackend::new().unwrap();

    // CHAIN of layers in one graph (dst feeds the next op's x), so the timing reports the
    // MARGINAL per-layer cost inside a batched forward, not the fixed per-execute overhead.
    let nlayers = 8usize;
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![ne], DType::F32));
    let router = g.weight(TensorDesc::new(vec![n_expert, ne], DType::F32));
    let gate = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
    let up = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
    let down = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q4K));
    let mut cur = x;
    let mut dst = x;
    for l in 0..nlayers {
        dst = if l + 1 == nlayers {
            g.output(TensorDesc::new(vec![ne], DType::F32))
        } else {
            g.internal(TensorDesc::new(vec![ne], DType::F32))
        };
        g.push(Op::MoeFfn {
            x: cur,
            router,
            gate_exps: gate,
            up_exps: up,
            down_exps: down,
            dst,
            ne: ne as u32,
            n_expert: n_expert as u32,
            n_used: n_used as u32,
            n_ff_exp: nff as u32,
            scale: 1.0,
            act: infr_core::graph::Activation::Silu,
        });
        cur = dst;
    }

    let xs: Vec<f32> = (0..ne).map(|i| (i % 13) as f32 * 0.01 - 0.06).collect();
    let rw: Vec<f32> = (0..n_expert * ne)
        .map(|i| (i % 17) as f32 * 0.001)
        .collect();
    let bound: Vec<(infr_core::tensor::TensorId, Vec<u8>)> = vec![
        (x, bytemuck::cast_slice(&xs).to_vec()),
        (router, bytemuck::cast_slice(&rw).to_vec()),
        (gate, synth_q4k(n_expert * nff * ne, 1)),
        (up, synth_q4k(n_expert * nff * ne, 2)),
        (down, synth_q4k(n_expert * ne * nff, 3)),
    ];

    let mut bufs: Vec<(infr_core::tensor::TensorId, Box<dyn Buffer>)> = Vec::new();
    for (id, bytes) in &bound {
        let b = be.alloc(bytes.len().max(4), BufferUsage::Weights).unwrap();
        be.upload(b.as_ref(), bytes).unwrap();
        bufs.push((*id, b));
    }
    let ob = be.alloc(ne * 4, BufferUsage::Activations).unwrap();
    bufs.push((dst, ob));
    let mut binds = Bindings::new();
    for (id, b) in &bufs {
        binds.bind(*id, b.as_ref());
    }
    let plan = be.compile(&g).unwrap();

    // Warmup (weight-cache build + pipeline compile), then timed reps.
    for _ in 0..3 {
        be.execute(plan.as_ref(), &binds).unwrap();
    }
    be.sync().unwrap();
    let reps = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        be.execute(plan.as_ref(), &binds).unwrap();
    }
    be.sync().unwrap();
    let per = t0.elapsed().as_secs_f64() / (reps * nlayers) as f64;
    let path = if std::env::var("INFR_METAL_NOMOE").is_ok() {
        "host"
    } else {
        "device"
    };
    // Active bytes per token: 3 matmuls x n_used experts (q4k = 4.5 bpw -> 0.5625 B/elem).
    let mb = (3 * n_used * nff * ne) as f64 * 0.5625 / 1e6;
    println!(
        "moe layer ({path}, marginal of {nlayers}-chain): {:.3} ms/op, active expert stream {mb:.1} MB -> {:.1} GB/s",
        per * 1e3,
        mb / 1e3 / per
    );
}
