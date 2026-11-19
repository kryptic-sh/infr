//! How fast can this device read DRAM when nothing else is in the way?
//!
//! `bw_scan.comp` streams a buffer with every thread reading consecutive `vec4` words, so a warp
//! covers 512B and DRAM sees one linear stream. That number is the ceiling the decode GEMV
//! (`native_mmv_mrow` / `narrow`, 177-189 GB/s in-model) is measured against; the gap between the
//! two is what a staging rewrite could win, and a small gap means decode is already at the wall.
//!
//! Run: `cargo test -p infr-vulkan --release --test bw_scan -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;
use std::time::Instant;

const WORDS: usize = 128 * 1024 * 1024; // 2 GiB of vec4

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn bw_scan() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let src = be.alloc(WORDS * 16, BufferUsage::Weights).unwrap();
    be.upload(src.as_ref(), &vec![0xA5u8; WORDS * 16]).unwrap();
    let bytes = (WORDS * 16) as f64;

    let threads = 256usize;
    println!("source: {:.2} GiB", bytes / 1e9);
    for groups in [36usize, 72, 144, 288, 576, 1152, 2304, 4608] {
        let dst = be
            .alloc(groups * threads * 16, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.bw_scan_at(src.as_ref(), dst.as_ref(), WORDS, groups);
        let start = Instant::now();
        rec.finish().unwrap();
        be.download(dst.as_ref(), &mut [0u8; 16]).unwrap();
        let secs = start.elapsed().as_secs_f64();
        println!("  {groups:5} wgs: {:7.1} GB/s", bytes / secs / 1e9);
    }
}
