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
fn exl3_projection_reads_weights_after_recorded_transfers() {
    let be = VulkanBackend::new().unwrap();
    let spec = Exl3Spec {
        in_f: 128,
        out_f: 128,
        bits: 4,
        codebook: Exl3Codebook::Cb0,
        input_layout: Exl3OutputLayout::Identity,
        output_layout: Exl3OutputLayout::Identity,
    };
    let packed: Vec<u8> = (0..spec.trellis_bytes().unwrap())
        .map(|i| (i * 131) as u8)
        .collect();
    let suh = vec![0.125f32; 128];
    let svh = vec![-0.125f32; 128];
    let scale_bytes = |v: &[f32]| {
        v.iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin()).collect();
    let expected = infr_cpu::exl3_linear::linear(&packed, &suh, &svh, &x, 1, spec).unwrap();
    let sources = [
        upload(&be, &packed, BufferUsage::Activations),
        upload(&be, &scale_bytes(&suh), BufferUsage::Activations),
        upload(&be, &scale_bytes(&svh), BufferUsage::Activations),
    ];
    let weights = [
        be.alloc(packed.len(), BufferUsage::Weights).unwrap(),
        be.alloc(256, BufferUsage::Weights).unwrap(),
        be.alloc(256, BufferUsage::Weights).unwrap(),
    ];
    let input = upload(&be, bytemuck::cast_slice(&x), BufferUsage::Activations);
    let xh = be.alloc(512, BufferUsage::Activations).unwrap();
    let z = be.alloc(512, BufferUsage::Activations).unwrap();
    let output = be.alloc(512, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    for (src, dst) in sources.iter().zip(&weights) {
        rec.copy(src.as_ref(), 0, dst.as_ref(), 0, src.len_bytes());
    }
    rec.linear_exl3(
        weights[0].as_ref(),
        weights[1].as_ref(),
        weights[2].as_ref(),
        input.as_ref(),
        xh.as_ref(),
        z.as_ref(),
        output.as_ref(),
        1,
        spec,
    )
    .unwrap();
    rec.finish().unwrap();
    let mut got = vec![0f32; 128];
    be.download(output.as_ref(), bytemuck::cast_slice_mut(&mut got))
        .unwrap();
    let scale = expected.iter().fold(0f32, |a, v| a.max(v.abs()));
    for (a, b) in got.iter().zip(expected) {
        assert!((a - b).abs() <= 1e-4 * scale + 1e-6, "GPU {a} vs CPU {b}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn exl3_linear_matches_cpu_all_rates_decode_and_prefill() {
    let be = VulkanBackend::new().unwrap();
    for (codebook, bits) in [Exl3Codebook::Cb0, Exl3Codebook::Mcg]
        .into_iter()
        .flat_map(|codebook| (1..=8).map(move |bits| (codebook, bits)))
    {
        for (k, n, input_layout, layout) in [
            (
                256,
                384,
                Exl3OutputLayout::Identity,
                Exl3OutputLayout::Identity,
            ),
            (
                384,
                256,
                Exl3OutputLayout::Identity,
                Exl3OutputLayout::RopeInterleaved {
                    heads: 4,
                    head_dim: 64,
                    rotary_dim: 48,
                },
            ),
            (
                384,
                512,
                Exl3OutputLayout::GroupedValueHeads {
                    prefix: 0,
                    key_heads: 3,
                    value_heads: 6,
                    head_dim: 64,
                },
                Exl3OutputLayout::GroupedValueHeads {
                    prefix: 128,
                    key_heads: 3,
                    value_heads: 6,
                    head_dim: 64,
                },
            ),
        ] {
            let spec = Exl3Spec {
                in_f: k,
                out_f: n,
                bits,
                codebook,
                input_layout,
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
    for (codebook, (k, n, bits)) in
        [Exl3Codebook::Cb0, Exl3Codebook::Mcg]
            .into_iter()
            .flat_map(|codebook| {
                [
                    (3584, 512, 4),
                    (3584, 18944, 4),
                    (18944, 3584, 4),
                    (3584, 3584, 3),
                    (4096, 8192, 2),
                    (4096, 4096, 2),
                ]
                .into_iter()
                .map(move |shape| (codebook, shape))
            })
    {
        let spec = Exl3Spec {
            in_f: k,
            out_f: n,
            bits,
            codebook,
            input_layout: Exl3OutputLayout::Identity,
            output_layout: Exl3OutputLayout::Identity,
        };
        for m in [1, 5] {
            run(&be, spec, m, (k + n) as usize);
        }
    }
}
