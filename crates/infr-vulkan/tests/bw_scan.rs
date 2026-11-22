//! How fast can this device read DRAM when nothing else is in the way?
//!
//! `bw_scan.comp` streams a buffer with every thread reading consecutive `vec4` words, so a warp
//! covers 512B and DRAM sees one linear stream. That number is the ceiling the decode GEMV
//! (`native_mmv_mrow` / `narrow`, 177-189 GB/s in-model) is measured against; the gap between the
//! two is what a staging rewrite could win, and a small gap means decode is already at the wall.
//!
//! The source buffer is `BW_SCAN_MIB` MiB ([`DEFAULT_MIB`] when unset), held once on the host
//! and once on the device. The published figures were taken at 2048, which is past every cache
//! level; a smaller buffer still streams, but check the size printed beside a number before
//! comparing it with those.
//!
//! Run: `BW_SCAN_MIB=2048 cargo test -p infr-vulkan --release --test bw_scan -- --ignored
//! --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;
use std::time::Instant;

/// Source size when `BW_SCAN_MIB` is unset: small enough to run beside other work on a shared
/// machine and on an integrated GPU, whose device memory is host memory.
const DEFAULT_MIB: usize = 256;

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn bw_scan() {
    let Some(be) = infr_testkit::gpu_or_skip(VulkanBackend::new()) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let mib = match std::env::var("BW_SCAN_MIB") {
        Ok(s) => s
            .parse::<usize>()
            .ok()
            .filter(|&m| m > 0)
            .unwrap_or_else(|| panic!("BW_SCAN_MIB: expected a size in MiB above 0 (got {s:?})")),
        Err(_) => DEFAULT_MIB,
    };
    let bytes = mib << 20;
    let words = bytes / 16;
    let src = be.alloc(bytes, BufferUsage::Weights).unwrap();
    be.upload(src.as_ref(), &vec![0xA5u8; bytes]).unwrap();

    let threads = 256usize;
    println!("source: {mib} MiB");
    for groups in [36usize, 72, 144, 288, 576, 1152, 2304, 4608] {
        let dst = be
            .alloc(groups * threads * 16, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.bw_scan_at(src.as_ref(), dst.as_ref(), words, groups);
        let start = Instant::now();
        rec.finish().unwrap();
        be.download(dst.as_ref(), &mut [0u8; 16]).unwrap();
        let secs = start.elapsed().as_secs_f64();
        println!("  {groups:5} wgs: {:7.1} GB/s", bytes as f64 / secs / 1e9);
    }
}
