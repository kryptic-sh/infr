//! Exact scalar/packed Q4_K metadata parity, including OUTS4 tails and both row caps.
//! Run headlessly with INFR_DEV=VulkanN; GPU workloads must run serially.
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::DType;
use infr_vulkan::{Recorder, VulkanBackend};

#[allow(dead_code)]
mod bench_support;

fn weights(in_f: usize, out_f: usize, base: usize, seed: u32) -> Vec<u8> {
    let mut bytes: Vec<u8> = (0..(base + in_f * out_f) / 256 * 144)
        .map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 24) as u8)
        .collect();
    let scales = [0.0, -0.0, 0.000001, -0.000001, 0.004, -0.007, 0.25, 0.75];
    for (i, block) in bytes.chunks_exact_mut(144).enumerate() {
        block[0..2].copy_from_slice(&half::f16::from_f32(scales[i % scales.len()]).to_le_bytes());
        block[2..4]
            .copy_from_slice(&half::f16::from_f32(scales[(i + 3) % scales.len()]).to_le_bytes());
        // Exercise all bit positions of the six-bit scale/min fields, not just small scales.
        if i % 5 == 0 {
            block[4..16].fill(0xff);
        } else if i % 5 == 1 {
            block[4..16].fill(0);
        }
    }
    bytes
}

fn upload(be: &VulkanBackend, bytes: &[u8], usage: BufferUsage) -> Box<dyn Buffer> {
    let buf = be.alloc(bytes.len(), usage).unwrap();
    be.upload(buf.as_ref(), bytes).unwrap();
    buf
}

struct Inputs {
    qa: Box<dyn Buffer>,
    dact: Box<dyn Buffer>,
    sact: Box<dyn Buffer>,
    residual: Box<dyn Buffer>,
    y: Box<dyn Buffer>,
}
impl Inputs {
    fn new(be: &VulkanBackend, rows: usize, in_f: usize, out_f: usize) -> Self {
        // Identical prequantized operands isolate weight unpacking from the quantizer.
        let qa: Vec<i8> = (0..rows * in_f)
            .map(|i| ((i * 37 % 255) as i32 - 127) as i8)
            .collect();
        let dact: Vec<half::f16> = (0..rows * in_f / 32)
            .map(|i| half::f16::from_f32(0.003 + (i % 11) as f32 * 0.001))
            .collect();
        let sact: Vec<half::f16> = qa
            .chunks_exact(32)
            .zip(&dact)
            .map(|(q, d)| {
                half::f16::from_f32(q.iter().map(|&v| v as f32).sum::<f32>() * d.to_f32())
            })
            .collect();
        let dact: Vec<u16> = dact.iter().map(|v| v.to_bits()).collect();
        let sact: Vec<u16> = sact.iter().map(|v| v.to_bits()).collect();
        let residual: Vec<f32> = (0..out_f).map(|i| (i % 23) as f32 * 0.025 - 0.25).collect();
        Self {
            qa: upload(be, bytemuck::cast_slice(&qa), BufferUsage::Activations),
            dact: upload(be, bytemuck::cast_slice(&dact), BufferUsage::Activations),
            sact: upload(be, bytemuck::cast_slice(&sact), BufferUsage::Activations),
            residual: upload(
                be,
                bytemuck::cast_slice(&residual),
                BufferUsage::Activations,
            ),
            y: be
                .alloc(rows * out_f * 4, BufferUsage::Activations)
                .unwrap(),
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        rec: &Recorder,
        w: &dyn Buffer,
        base: usize,
        rows: usize,
        in_f: usize,
        out_f: usize,
        res: bool,
    ) {
        rec.linear_mmv_mrow(
            DType::Q4K,
            w,
            base,
            self.qa.as_ref(),
            self.dact.as_ref(),
            self.sact.as_ref(),
            res.then_some(self.residual.as_ref()),
            self.y.as_ref(),
            rows,
            in_f,
            out_f,
        );
    }
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        be: &VulkanBackend,
        w: &dyn Buffer,
        base: usize,
        rows: usize,
        in_f: usize,
        out_f: usize,
        res: bool,
    ) -> Vec<u32> {
        let rec = be.recorder().unwrap();
        self.record(&rec, w, base, rows, in_f, out_f, res);
        rec.finish().unwrap();
        let mut bits = vec![0u32; rows * out_f];
        be.download(self.y.as_ref(), bytemuck::cast_slice_mut(&mut bits))
            .unwrap();
        assert!(bits.iter().all(|&b| f32::from_bits(b).is_finite()));
        bits
    }
}

fn expected_kernel(
    in_f: usize,
    rows: usize,
    res: bool,
    m4: bool,
    o4: bool,
    packed: bool,
) -> String {
    let mut name = String::from("native_mmv_mrow_q4k");
    if rows > 8 {
        name.push_str("_m16");
    } else {
        let o4 = o4 && !(in_f / 32).is_multiple_of(64);
        if o4 {
            name.push_str("_o4");
        }
        if rows <= 4 && m4 {
            name.push_str("_m4");
        }
        if res {
            name.push_str("_res");
        }
        if packed && o4 {
            name.push_str("_packed_meta");
        }
    }
    name
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn packed_metadata_matches_scalar_exact() {
    for (m4, o4) in [(true, true), (false, true), (true, false)] {
        let old = bench_support::backend_with(|vk| {
            vk.mmv_q4k_meta = false;
            vk.mmv_m4 = m4;
            vk.mmv_o4 = o4;
        });
        let new = bench_support::backend_with(|vk| {
            vk.mmv_q4k_meta = true;
            vk.mmv_m4 = m4;
            vk.mmv_o4 = o4;
        });
        // Nonzero element offsets retain the Q4_K 256-element block alignment. k=2048/6144
        // exercises the unchanged two-output layout; the rest cover partial 64-lane K tiles.
        for (in_f, out_f, base) in [
            (256, 1, 0),
            (1536, 3, 256),
            (2560, 5, 768),
            (2816, 66, 256),
            (11008, 5, 768),
            (2048, 3, 256),
            (6144, 3, 0),
        ] {
            let bytes = weights(in_f, out_f, base, 17);
            let w_old = upload(&old, &bytes, BufferUsage::Weights);
            let w_new = upload(&new, &bytes, BufferUsage::Weights);
            let mut decode = None;
            for rows in [1, 2, 3, 4, 5, 8, 9, 16] {
                let x_old = Inputs::new(&old, rows, in_f, out_f);
                let x_new = Inputs::new(&new, rows, in_f, out_f);
                for res in [false, true] {
                    // Residual shaders are built only at MRV=4 and dispatched for rows=1.
                    if res && (rows != 1 || !m4) {
                        continue;
                    }
                    let want = x_old.run(&old, w_old.as_ref(), base, rows, in_f, out_f, res);
                    let got = x_new.run(&new, w_new.as_ref(), base, rows, in_f, out_f, res);
                    assert_eq!(
                        got, want,
                        "k={in_f} n={out_f} base={base} rows={rows} res={res} m4={m4} o4={o4}"
                    );
                    // Bit parity alone is vacuous if both configurations select scalar kernels.
                    for (be, packed) in [(&old, false), (&new, true)] {
                        let expected = expected_kernel(in_f, rows, res, m4, o4, packed);
                        assert!(
                            be.built_kernel_names().contains(&expected.as_str()),
                            "missing {expected}"
                        );
                    }
                    // rows>8 intentionally uses the existing two-output m16 reduction; the
                    // decode/verify bit-identity invariant applies within the OUTS4 row caps.
                    if !res && rows <= 8 {
                        if let Some(ref row0) = decode {
                            assert_eq!(&got[..out_f], row0);
                        } else {
                            decode = Some(got[..out_f].to_vec());
                        }
                    }
                }
            }
        }
        assert!(!old
            .built_kernel_names()
            .iter()
            .any(|n| n.ends_with("_packed_meta")));
        if o4 {
            for name in [
                "native_mmv_mrow_q4k_o4_packed_meta",
                "native_mmv_mrow_q4k_o4_m4_packed_meta",
                "native_mmv_mrow_q4k_o4_m4_res_packed_meta",
            ] {
                if m4 || !name.contains("_m4") {
                    assert!(new.built_kernel_names().contains(&name), "missing {name}");
                }
            }
        } else {
            assert!(!new
                .built_kernel_names()
                .iter()
                .any(|n| n.ends_with("_packed_meta")));
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU (bounded serial perf micro-bench)"]
fn packed_metadata_bw() {
    use std::cell::Cell;
    let old = bench_support::backend_with(|vk| {
        vk.mmv_q4k_meta = false;
        vk.mmv_o4 = true;
    });
    let new = bench_support::backend_with(|vk| {
        vk.mmv_q4k_meta = true;
        vk.mmv_o4 = true;
    });
    let shapes = if old.capabilities().integrated {
        vec![(2560, 1024), (11008, 256)]
    } else {
        vec![(2560, 20480), (11008, 2048), (2560, 2048)]
    };
    for (in_f, out_f) in shapes {
        let banks = if old.capabilities().integrated { 1 } else { 4 };
        let wbytes = in_f * out_f / 256 * 144;
        let source: Vec<_> = (0..banks)
            .map(|i| weights(in_f, out_f, 0, i as u32 * 123))
            .collect();
        let w_old: Vec<_> = source
            .iter()
            .map(|b| upload(&old, b, BufferUsage::Weights))
            .collect();
        let w_new: Vec<_> = source
            .iter()
            .map(|b| upload(&new, b, BufferUsage::Weights))
            .collect();
        let x_old = Inputs::new(&old, 1, in_f, out_f);
        let x_new = Inputs::new(&new, 1, in_f, out_f);
        // Interleave old/new/new/old in one process. Four discrete-device banks reduce L2
        // reuse but do not guarantee a cold cache; times include recording/submit/wait.
        for (label, be, w, x) in [
            ("scalar", &old, &w_old, &x_old),
            ("packed", &new, &w_new, &x_new),
            ("packed", &new, &w_new, &x_new),
            ("scalar", &old, &w_old, &x_old),
        ] {
            let next = Cell::new(0usize);
            let timing = bench_support::time(be, label, 64, |rec| {
                let i = next.get();
                next.set(i + 1);
                x.record(rec, w[i % banks].as_ref(), 0, 1, in_f, out_f, false);
            });
            let us = timing
                .expect("bounded submission")
                .mean_us()
                .expect("complete bounded samples");
            println!("Q4K {in_f}x{out_f} {label}: {us:.2} us/op, {:.1} effective weight GB/s (prequantized, banks={banks})", wbytes as f64 / us / 1000.0);
        }
    }
}
