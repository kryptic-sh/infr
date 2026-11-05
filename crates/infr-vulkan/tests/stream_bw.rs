//! Device-local streaming bandwidth: one plain element copy over a 1 GiB buffer pair, no matmul.
//!
//! The decode-GEMV probe measures a kernel that reads weights and dequantizes them; if that kernel
//! is the limiter rather than DRAM, its 71-93 GB/s says nothing about the card. This probe reads and
//! writes DRAM and nothing else, so its number IS the memory ceiling under the clocks the card is
//! actually running. Clocks are sampled while it runs, because Polaris drops to a low state when it
//! thinks the load is light.
//!
//! Run: `cargo test -p infr-vulkan --release --test stream_bw -- --ignored --nocapture`

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::TensorDesc;
use infr_core::DType;

fn f32d(n: usize) -> TensorDesc {
    TensorDesc::new(vec![n], DType::F32)
}

/// Current clock state line (the one the driver marks with `*`) for `pp_dpm_<name>`.
fn clock_state(name: &str) -> Option<String> {
    for card in std::fs::read_dir("/sys/class/drm").ok()? {
        let p = card
            .ok()?
            .path()
            .join("device")
            .join(format!("pp_dpm_{name}"));
        let Ok(s) = std::fs::read_to_string(&p) else {
            continue;
        };
        if let Some(l) = s.lines().find(|l| l.trim_end().ends_with('*')) {
            return Some(l.trim().trim_end_matches('*').trim().to_string());
        }
    }
    None
}

#[test]
#[ignore]
fn stream_bandwidth() {
    let be = infr_vulkan::VulkanBackend::new().expect("vulkan device");
    // 1 GiB per buffer, in f32 ELEMENTS (the IR counts elements, the recorder counts bytes).
    let elems = 256 * 1024 * 1024;
    let bytes = elems * 4;
    let src = be.alloc(bytes, BufferUsage::KvCache).expect("alloc src");
    let dst = be.alloc(bytes, BufferUsage::KvCache).expect("alloc dst");

    let mut g = Graph::new();
    let s = g.input(f32d(elems));
    let d = g.input(f32d(elems));
    g.push(Op::Copy {
        src: s,
        src_off: 0,
        dst: d,
        dst_off: 0,
        n: elems as u32,
    });
    let plan = be.compile(&g).expect("compile stream copy");
    let mut b = Bindings::new();
    b.bind(s, src.as_ref());
    b.bind(d, dst.as_ref());

    be.execute(plan.as_ref(), &b).expect("warmup");

    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut mclk = Vec::new();
            let mut sclk = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                if let Some(c) = clock_state("mclk") {
                    if mclk.last() != Some(&c) {
                        mclk.push(c);
                    }
                }
                if let Some(c) = clock_state("sclk") {
                    if sclk.last() != Some(&c) {
                        sclk.push(c);
                    }
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            (mclk, sclk)
        })
    };

    let iters = 8;
    let t0 = Instant::now();
    for _ in 0..iters {
        be.execute(plan.as_ref(), &b).expect("copy");
    }
    let dt = t0.elapsed();
    stop.store(true, Ordering::Relaxed);
    let (mclk, sclk) = sampler.join().expect("clock sampler");

    let gbs = (iters as f64 * 2.0 * bytes as f64) / dt.as_secs_f64() / 1e9;
    println!(
        "stream copy: {iters} x 1 GiB, {:.1} ms/pass, {:.1} GB/s (read+write)",
        dt.as_secs_f64() * 1000.0 / iters as f64,
        gbs
    );
    println!("  mclk states seen: {mclk:?}");
    println!("  sclk states seen: {sclk:?}");
}
