//! Parity and timing for the final-logit tail kernels (softcap, two-stage argmax).
//! Run: cargo test -p infr-vulkan --release --test logits_tail -- --ignored --nocapture
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_vulkan::VulkanBackend;

fn logits(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * 60.0
        })
        .collect()
}

fn upload(be: &VulkanBackend, v: &[f32]) -> Box<dyn Buffer> {
    let b = be.alloc(v.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
    b
}

fn host_argmax(row: &[f32]) -> u32 {
    let mut bi = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[bi] {
            bi = i;
        }
    }
    bi as u32
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn argmax_matches_host_with_ties() {
    let be = VulkanBackend::new().unwrap();
    for n in [1000usize, 1024, 4096, 151936, 262144, 262145] {
        for rows in [1usize, 3] {
            let mut v = logits(n * rows, 7 + n as u64);
            for r in 0..rows {
                // duplicate the maximum at a lower and a higher index
                let row = &mut v[r * n..(r + 1) * n];
                let m = row.iter().cloned().fold(f32::MIN, f32::max) + 1.0;
                row[n / 3 + r] = m;
                row[n - 1 - r] = m;
                row[(n / 3 + r) / 2] = m;
            }
            let x = upload(&be, &v);
            let part = be.alloc(512 * 4, BufferUsage::Activations).unwrap();
            let out = be.alloc(rows * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.argmax(x.as_ref(), part.as_ref(), out.as_ref(), n, rows);
            rec.finish().unwrap();
            let mut got = vec![0u32; rows];
            be.download(out.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for r in 0..rows {
                assert_eq!(got[r], host_argmax(&v[r * n..(r + 1) * n]), "n {n} row {r}");
            }
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn softcap_matches_host() {
    let be = VulkanBackend::new().unwrap();
    for n in [1000usize, 262144, 262145] {
        let v = logits(n, 3);
        let x = upload(&be, &v);
        let y = be.alloc(n * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.softcap(x.as_ref(), y.as_ref(), 30.0, n);
        rec.finish().unwrap();
        let mut got = vec![0f32; n];
        be.download(y.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..n {
            let want = 30.0 * (v[i] / 30.0).tanh();
            assert!((got[i] - want).abs() <= 1e-4 * want.abs().max(1.0), "n {n} i {i}");
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn logits_tail_time() {
    let be = VulkanBackend::new().unwrap();
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(262144);
    let v = logits(n, 5);
    let x = upload(&be, &v);
    let y = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    let part = be.alloc(512 * 4, BufferUsage::Activations).unwrap();
    let out = be.alloc(4, BufferUsage::Activations).unwrap();
    const REPS: usize = 64;
    let time = |f: &dyn Fn(&infr_vulkan::Recorder<'_>)| -> f64 {
        let run = || {
            let rec = be.recorder().unwrap();
            for _ in 0..REPS {
                f(&rec);
            }
            rec.finish().unwrap();
        };
        run();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            run();
            best = best.min(t.elapsed().as_secs_f64() / REPS as f64 * 1e6);
        }
        best
    };
    let sc = time(&|r| r.softcap(x.as_ref(), y.as_ref(), 30.0, n));
    let am = time(&|r| r.argmax(y.as_ref(), part.as_ref(), out.as_ref(), n, 1));
    println!("softcap {sc:.1} us   argmax {am:.1} us   (n {n}, {REPS} chained reps)");
}
