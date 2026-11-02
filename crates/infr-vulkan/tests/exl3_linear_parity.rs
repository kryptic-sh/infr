//! Native EXL3 projection on Vulkan vs the CPU tile-streamed reference.

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::{DType, Exl3Codebook, Exl3OutputLayout, Exl3Spec, Graph, Op, TensorDesc};
use infr_vulkan::VulkanBackend;

fn upload(be: &VulkanBackend, bytes: &[u8], usage: BufferUsage) -> Box<dyn Buffer> {
    let b = be.alloc(bytes.len().max(4), usage).unwrap();
    be.upload(b.as_ref(), bytes).unwrap();
    b
}

fn run(be: &VulkanBackend, spec: Exl3Spec, m: usize, seed: usize) {
    let (k, n) = (spec.in_f as usize, spec.out_f as usize);
    let packed: Vec<u8> = (0..spec.trellis_bytes().unwrap())
        .map(|i| (i.wrapping_mul(131).wrapping_add(seed) ^ (i >> 7)) as u8)
        .collect();
    let half_bytes = |len: usize, mul: usize| -> (Vec<f32>, Vec<u8>) {
        let v: Vec<f32> = (0..len)
            .map(|i| {
                half::f16::from_f32((((i * mul + seed) % 17) as f32 - 8.0) * 0.013 + 0.004).to_f32()
            })
            .collect();
        let b = v
            .iter()
            .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
            .collect();
        (v, b)
    };
    let (suh, suh_b) = half_bytes(k, 5);
    let (svh, svh_b) = half_bytes(n, 3);
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i + seed) as f32 * 0.031).sin())
        .collect();
    let want = infr_cpu::exl3_linear::linear(&packed, &suh, &svh, &x, m, spec).unwrap();

    let mut g = Graph::new();
    let xi = g.input(TensorDesc::new(vec![m * k], DType::F32));
    let ti = g.weight(TensorDesc::new(vec![packed.len() / 2], DType::I16));
    let si = g.weight(TensorDesc::new(vec![k], DType::F16));
    let vi = g.weight(TensorDesc::new(vec![n], DType::F16));
    let yi = g.output(TensorDesc::new(vec![m * n], DType::F32));
    g.push(Op::LinearExl3 {
        x: xi,
        trellis: ti,
        suh: si,
        svh: vi,
        dst: yi,
        m: m as u32,
        spec,
    });
    let xb = upload(be, bytemuck::cast_slice(&x), BufferUsage::Activations);
    let tb = upload(be, &packed, BufferUsage::Weights);
    let sb = upload(be, &suh_b, BufferUsage::Weights);
    let vb = upload(be, &svh_b, BufferUsage::Weights);
    let yb = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();
    let mut bind = Bindings::new();
    bind.bind(xi, xb.as_ref())
        .bind(ti, tb.as_ref())
        .bind(si, sb.as_ref())
        .bind(vi, vb.as_ref())
        .bind(yi, yb.as_ref());
    let plan = be.compile(&g).unwrap();
    be.execute(plan.as_ref(), &bind).unwrap();
    let mut got = vec![0f32; m * n];
    be.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
        .unwrap();
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!(
            (a - b).abs() <= 1e-4 * scale + 1e-6,
            "{:?} m={m} idx={i}: vulkan {a} vs cpu {b} (scale {scale})",
            spec
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn exl3_linear_matches_cpu_all_rates_decode_and_prefill() {
    let be = VulkanBackend::new().unwrap();
    for bits in 1..=8 {
        for (k, n, layout) in [
            (256, 384, Exl3OutputLayout::Identity),
            (
                384,
                256,
                Exl3OutputLayout::RopeInterleaved {
                    heads: 4,
                    head_dim: 64,
                    rotary_dim: 48,
                },
            ),
        ] {
            let spec = Exl3Spec {
                in_f: k,
                out_f: n,
                bits,
                codebook: Exl3Codebook::Cb0,
                output_layout: layout,
            };
            for m in [1, 2, 7, 8, 9, 17] {
                run(&be, spec, m, (bits * 31 + m as u32) as usize);
            }
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn exl3_linear_matches_cpu_at_qwen25_7b_shapes() {
    let be = VulkanBackend::new().unwrap();
    for (k, n, bits) in [
        (3584, 512, 4),
        (3584, 18944, 4),
        (18944, 3584, 4),
        (3584, 3584, 3),
    ] {
        let spec = Exl3Spec {
            in_f: k,
            out_f: n,
            bits,
            codebook: Exl3Codebook::Cb0,
            output_layout: Exl3OutputLayout::Identity,
        };
        for m in [1, 5] {
            run(&be, spec, m, (k + n) as usize);
        }
    }
}
