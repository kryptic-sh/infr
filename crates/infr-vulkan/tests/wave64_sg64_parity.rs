//! Wave64-native (`-DSG=64`, plain full-wave subgroup ops) vs emulated (`-DSG=32` workgroups
//! as 32-lane halves via `subgroupClusteredAdd`) decode GEMV parity for the Q6_K projection band
//! (`native_gemv_sg.comp`).
//!
//! The two builds reorder the float accumulation differently (32- vs 64-lane partitioning), so
//! this asserts each leg against a host f64 reference with a reassociation-tolerant epsilon —
//! exactly like `kq_fma_gemv`'s parity — rather than bit-identity between the legs. It ALSO
//! asserts the selector actually diverged: on a wave64 device the native leg must have built a
//! `*_sg64` kernel and the emulated leg must not have (a parity test where both legs quietly
//! take the same kernel is vacuous). On non-wave64 devices the SG=64 builds are never selected
//! and the divergence assert is skipped (both legs still prove correctness vs the host).
//!
//! Run: `cargo test -p infr-vulkan --test wave64_sg64_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

struct Rng(u64);
impl Rng {
    fn byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
    fn f32(&mut self) -> f32 {
        self.byte() as f32 / 128.0 - 1.0
    }
}

/// Valid Q6_K blocks: random payload with a small positive scale field.
fn synth(n_elems: usize, seed: u64) -> Vec<u8> {
    let (epb, bpb) = infr_gguf::block_layout(DType::Q6K);
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n_elems / epb * bpb);
    for _ in 0..n_elems / epb {
        let mut b: Vec<u8> = (0..bpb).map(|_| rng.byte()).collect();
        let d = half::f16::from_f32(0.004 + rng.byte() as f32 * 1e-5).to_le_bytes();
        b[208..210].copy_from_slice(&d);
        out.extend_from_slice(&b);
    }
    out
}

fn backend(wave64_native: bool) -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.kernels.vulkan.wave64_native = wave64_native;
    VulkanBackend::new_with(std::sync::Arc::new(cfg)).unwrap()
}

fn run_leg(be: &VulkanBackend, w_bytes: &[u8], xv: &[f32], in_f: usize, out_f: usize, nr: u32) -> Vec<f32> {
    let (arena, addr) = be.alloc_arena_bda(w_bytes.len()).unwrap();
    be.upload(arena.as_ref(), w_bytes).unwrap();
    let x = be.alloc(xv.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(x.as_ref(), bytemuck::cast_slice(xv)).unwrap();
    let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.linear_native_sg_at(DType::Q6K, addr, 0, x.as_ref(), y.as_ref(), in_f, out_f, nr);
    rec.finish().unwrap();
    let mut got = vec![0f32; out_f];
    be.download(y.as_ref(), bytemuck::cast_slice_mut(&mut got)).unwrap();
    got
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn wave64_sg64_parity() {
    let native = backend(true);
    let emulated = backend(false);
    // Real decode shapes in the SG band (Q6_K down/o projections).
    for (in_f, out_f) in [(4096usize, 4096usize), (4096, 2048), (12288, 4096)] {
        let bank = synth(in_f * out_f, 0x51 ^ (in_f * out_f) as u64);
        let wq = infr_gguf::dequant::dequant_block(DType::Q6K, &bank).unwrap();
        let mut rng = Rng(11 + in_f as u64);
        let xv: Vec<f32> = (0..in_f).map(|_| rng.f32()).collect();
        for nr in [2u32, 4, 8] {
            let got_nat = run_leg(&native, &bank, &xv, in_f, out_f, nr);
            let got_emu = run_leg(&emulated, &bank, &xv, in_f, out_f, nr);
            let mut max_cross = 0f32;
            for o in 0..out_f {
                let row = &wq[o * in_f..(o + 1) * in_f];
                let want: f64 = row.iter().zip(&xv).map(|(a, b)| *a as f64 * *b as f64).sum();
                let want = want as f32;
                let tol = 1e-4 + 1e-4 * want.abs();
                assert!(
                    (got_nat[o] - want).abs() <= tol,
                    "{in_f}x{out_f} nr {nr}: NATIVE out {o} got {} want {want}",
                    got_nat[o]
                );
                assert!(
                    (got_emu[o] - want).abs() <= tol,
                    "{in_f}x{out_f} nr {nr}: EMULATED out {o} got {} want {want}",
                    got_emu[o]
                );
                max_cross = max_cross.max((got_nat[o] - got_emu[o]).abs());
            }
            println!("Q6_K {in_f}x{out_f} nr {nr}: native+emulated vs host OK, max cross-leg diff {max_cross:.3e}");
        }
    }
    // The divergence proof: on a wave64 device the legs must have built different kernels.
    let nat_names = native.built_kernel_names();
    let emu_names = emulated.built_kernel_names();
    let nat_sg64: Vec<&&str> = nat_names.iter().filter(|n| n.contains("_sg64")).collect();
    if nat_sg64.is_empty() {
        println!("no _sg64 kernel built — not a wave64 device (or knob off): parity-only, skipping divergence assert");
    } else {
        assert!(
            emu_names.iter().all(|n| !n.contains("_sg64")),
            "emulated leg built an _sg64 kernel — the A/B did not diverge: {emu_names:?}"
        );
        println!("divergence proven: native leg built {nat_sg64:?}");
    }
}

/// Cold-weight SG bandwidth A/B: native (`-DSG=64`) vs emulated (`-DSG=32` + W64) at Q6_K
/// decode shapes, working set sized past the 96 MiB Infinity Cache so the number is TRUE DRAM
/// bandwidth (same discipline as `decode_gemv_bw`). Dispatches `linear_native_sg_at` directly
/// so exactly the kernel family under test fires; `finish()` is submit+wait-idle, so the
/// host-side timer is true GPU time.
///
/// Run: `cargo test -p infr-vulkan --test wave64_sg64_parity wave64_sg64_bw -- --ignored --nocapture`
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn wave64_sg64_bw() {
    let native = backend(true);
    let emulated = backend(false);
    for (label, in_f, out_f) in [("o      ", 4096usize, 4096usize), ("down-q6", 12288, 4096)] {
        let (epb, bpb) = infr_gguf::block_layout(DType::Q6K);
        let wbytes = in_f * out_f / epb * bpb;
        let n_w = (200usize << 20).div_ceil(wbytes).clamp(3, 24);
        let src: Vec<Vec<u8>> = (0..n_w)
            .map(|s| {
                (0..wbytes)
                    .map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(s as u32) >> 24) as u8)
                    .collect()
            })
            .collect();
        // Same synthetic weights on both legs (per-backend upload: a buffer belongs to the
        // device that allocated it).
        let prep = |be: &VulkanBackend| {
            let ws: Vec<_> = src
                .iter()
                .map(|s| {
                    let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
                    be.upload(w.as_ref(), s).unwrap();
                    w
                })
                .collect();
            let mut rng = Rng(3 + in_f as u64);
            let xv: Vec<f32> = (0..in_f).map(|_| rng.f32()).collect();
            let x = be.alloc(xv.len() * 4, BufferUsage::Activations).unwrap();
            be.upload(x.as_ref(), bytemuck::cast_slice(&xv)).unwrap();
            let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
            (ws, x, y)
        };
        let (nws, nx, ny) = prep(&native);
        let (ews, ex, ey) = prep(&emulated);
        let reps = (n_w * 4).max(40);
        let bench = |be: &VulkanBackend, ws: &[Box<dyn infr_core::backend::Buffer>], x: &Box<dyn infr_core::backend::Buffer>, y: &Box<dyn infr_core::backend::Buffer>, nr: u32| -> f64 {
            let run = || {
                let rec = be.recorder().unwrap();
                for r in 0..reps {
                    rec.linear_native_sg_at(DType::Q6K, ws[r % n_w].device_addr().unwrap(), 0, x.as_ref(), y.as_ref(), in_f, out_f, nr);
                }
                rec.finish().unwrap();
            };
            run();
            let t = std::time::Instant::now();
            run();
            let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            wbytes as f64 / (us * 1e-6) / 1e9
        };
        for nr in [2u32, 8] {
            let mut nat_best = 0f64;
            let mut emu_best = 0f64;
            for _ in 0..3 {
                emu_best = emu_best.max(bench(&emulated, &ews, &ex, &ey, nr));
                nat_best = nat_best.max(bench(&native, &nws, &nx, &ny, nr));
            }
            println!(
                "  {label} in{in_f} out{out_f} Q6_K NR={nr} [{} MiB]:  SG64-native {nat_best:5.0}  SG32-emulated {emu_best:5.0} GB/s  (native {:+.1}%)",
                wbytes >> 20,
                (nat_best / emu_best - 1.0) * 100.0,
            );
        }
    }
}
