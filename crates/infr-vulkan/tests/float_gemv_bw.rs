//! Decode (m=1) bandwidth of the float-weight GEMVs at Gemma 4 E4B's per-layer shapes.
//! Run: cargo test -p infr-vulkan --release --test float_gemv_bw -- --ignored --nocapture
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::DType;
use infr_vulkan::{Recorder, VulkanBackend};

const N_W: usize = 8;
const REPS: usize = 64;

fn bench(
    be: &VulkanBackend,
    wbytes: usize,
    ws: &[Box<dyn Buffer>],
    f: impl Fn(&Recorder<'_>, &dyn Buffer),
) -> f64 {
    let run = || {
        let rec = be.recorder().unwrap();
        for r in 0..REPS {
            f(&rec, ws[r % N_W].as_ref());
        }
        rec.finish().unwrap();
    };
    run();
    let mut best = 0f64;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        run();
        let s = t.elapsed().as_secs_f64() / REPS as f64;
        best = best.max(wbytes as f64 / s / 1e9);
    }
    best
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn float_gemv_bw() {
    let be = VulkanBackend::new().unwrap();
    for (label, dt, in_f, out_f) in [
        ("inp_gate", DType::F32, 2560, 256),
        ("proj", DType::F32, 256, 2560),
        ("per_layer_model_proj", DType::Bf16, 2560, 10752),
    ] {
        let eb = if dt == DType::F32 { 4 } else { 2 };
        let wbytes = in_f * out_f * eb;
        let ws: Vec<_> = (0..N_W)
            .map(|i| {
                let b = be.alloc(wbytes, BufferUsage::Weights).unwrap();
                let bytes: Vec<u8> = (0..wbytes).map(|j| ((j * 7 + i) % 61) as u8).collect();
                be.upload(b.as_ref(), &bytes).unwrap();
                b
            })
            .collect();
        let x = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
        let xv: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.01).sin()).collect();
        be.upload(x.as_ref(), bytemuck::cast_slice(&xv)).unwrap();
        let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        let mut line = format!("{label:22} {dt:?} {in_f}x{out_f} [{} KiB]:", wbytes >> 10);
        if dt == DType::F32 {
            let g = bench(&be, wbytes, &ws, |rec, w| {
                rec.linear_f32(w, x.as_ref(), y.as_ref(), 1, in_f, out_f)
            });
            line += &format!("  linear_f32 {g:5.0} GB/s |");
            let want = {
                let rec = be.recorder().unwrap();
                rec.linear_f32(ws[0].as_ref(), x.as_ref(), y.as_ref(), 1, in_f, out_f);
                rec.finish().unwrap();
                let mut v = vec![0f32; out_f];
                be.download(y.as_ref(), bytemuck::cast_slice_mut(&mut v))
                    .unwrap();
                v
            };
            for lg in 4..=8u32 {
                let addr = |w: &dyn Buffer| w.device_addr().unwrap();
                let rec = be.recorder().unwrap();
                rec.linear_f32_mo_at(
                    addr(ws[0].as_ref()),
                    x.as_ref(),
                    y.as_ref(),
                    in_f,
                    out_f,
                    lg,
                );
                rec.finish().unwrap();
                let mut got = vec![0f32; out_f];
                be.download(y.as_ref(), bytemuck::cast_slice_mut(&mut got))
                    .unwrap();
                let maxrel = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs() / (b.abs() + 1e-3))
                    .fold(0f32, f32::max);
                assert!(maxrel < 1e-3, "{label} tpo 2^{lg}: maxrel {maxrel}");
                let g = bench(&be, wbytes, &ws, |rec, w| {
                    rec.linear_f32_mo_at(addr(w), x.as_ref(), y.as_ref(), in_f, out_f, lg)
                });
                line += &format!("  mo{} {g:4.0}", 1 << lg);
            }
        } else {
            let g = bench(&be, wbytes, &ws, |rec, w| {
                rec.linear_native(dt, w, 0, x.as_ref(), y.as_ref(), 1, in_f, out_f)
            });
            line += &format!("  linear_native {g:5.0} GB/s");
            let g = bench(&be, wbytes, &ws, |rec, w| {
                rec.linear_bf16(w, x.as_ref(), y.as_ref(), 1, in_f, out_f)
            });
            line += &format!("  linear_bf16 {g:5.0} GB/s");
        }
        println!("{line}");
    }
}

fn host_gemv(w: &[f32], x: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
    (0..out_f)
        .map(|o| {
            w[o * in_f..(o + 1) * in_f]
                .iter()
                .zip(x)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn float_gemv_decode_parity() {
    let be = VulkanBackend::new().unwrap();
    let mut seed = 0x51u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1 << 24) as f32) - 0.5
    };
    // (dtype, in_f, out_f, w_base): Gemma's shapes, an in_f%4 fallback, a bf16 slice offset, and a
    // bf16 output count past the 1-D group limit.
    for (dt, in_f, out_f, w_base) in [
        (DType::F32, 2560, 256, 0),
        (DType::F32, 256, 2560, 0),
        (DType::F32, 18, 40, 0),
        (DType::Bf16, 2560, 10752, 0),
        (DType::Bf16, 256, 300, 512),
        (DType::Bf16, 64, 70_000, 0),
    ] {
        let n = w_base + in_f * out_f;
        let wf: Vec<f32> = (0..n).map(|_| rnd()).collect();
        let (bytes, wq): (Vec<u8>, Vec<f32>) = if dt == DType::F32 {
            (bytemuck::cast_slice(&wf).to_vec(), wf.clone())
        } else {
            let h: Vec<u16> = wf.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let q = h
                .iter()
                .map(|b| f32::from_bits((*b as u32) << 16))
                .collect();
            (bytemuck::cast_slice(&h).to_vec(), q)
        };
        let w = be.alloc(bytes.len(), BufferUsage::Weights).unwrap();
        be.upload(w.as_ref(), &bytes).unwrap();
        let xv: Vec<f32> = (0..in_f).map(|_| rnd()).collect();
        let x = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
        be.upload(x.as_ref(), bytemuck::cast_slice(&xv)).unwrap();
        let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        if dt == DType::F32 {
            rec.linear_f32(w.as_ref(), x.as_ref(), y.as_ref(), 1, in_f, out_f);
        } else {
            rec.linear_native(
                dt,
                w.as_ref(),
                w_base,
                x.as_ref(),
                y.as_ref(),
                1,
                in_f,
                out_f,
            );
        }
        rec.finish().unwrap();
        let mut got = vec![0f32; out_f];
        be.download(y.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        let want = host_gemv(&wq[w_base..], &xv, in_f, out_f);
        for (o, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - e).abs() <= 1e-4 + 1e-4 * e.abs(),
                "{dt:?} {in_f}x{out_f} w_base {w_base}: out {o} got {g} want {e}"
            );
        }
        println!("{dt:?} {in_f}x{out_f} w_base {w_base}: OK");
    }
}
