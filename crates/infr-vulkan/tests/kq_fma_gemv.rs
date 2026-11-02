//! Parity and decode bandwidth for the fp32 K-quant GEMV (`native_gemv_kq_fma.comp`) and its
//! multi-row verify twin (`native_gemv_kq_fma_mrow.comp`).
//! Run: cargo test -p infr-vulkan --release --test kq_fma_gemv -- --ignored --nocapture
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::DType;
use infr_vulkan::linear::pad_to_u32_align;
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

/// Valid Q4_K/Q6_K blocks: random payload with small positive scale fields.
fn synth(dt: DType, n_elems: usize, seed: u64) -> Vec<u8> {
    let (epb, bpb) = infr_gguf::block_layout(dt);
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n_elems / epb * bpb);
    for _ in 0..n_elems / epb {
        let mut b: Vec<u8> = (0..bpb).map(|_| rng.byte()).collect();
        let d = half::f16::from_f32(0.004 + rng.byte() as f32 * 1e-5).to_le_bytes();
        let m = half::f16::from_f32(0.002 + rng.byte() as f32 * 1e-5).to_le_bytes();
        match dt {
            DType::Q4K => {
                b[0..2].copy_from_slice(&d);
                b[2..4].copy_from_slice(&m);
            }
            DType::Q6K => b[208..210].copy_from_slice(&d),
            _ => unreachable!(),
        }
        out.extend_from_slice(&b);
    }
    out
}

fn upload(be: &VulkanBackend, bytes: &[u8], usage: BufferUsage) -> Box<dyn Buffer> {
    let b = be.alloc(bytes.len(), usage).unwrap();
    be.upload(b.as_ref(), bytes).unwrap();
    b
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn kq_fma_gemv_parity() {
    let be = VulkanBackend::new().unwrap();
    // (in_f, out_f, w_base): an out_f not divisible by NR, an odd block count per row (Q6_K blocks
    // alternate 2-byte alignment), a non-zero weight base, and a multi-superblock row.
    for dt in [DType::Q4K, DType::Q6K] {
        for (in_f, out_f, w_base) in [
            (256, 64, 0),
            (768, 30, 0),
            (2560, 301, 0),
            (768, 40, 768 * 3),
        ] {
            for (res, nr) in [(false, 4), (true, 4), (false, 8), (true, 8)] {
                let n = w_base + in_f * out_f;
                let bank = synth(dt, n, 0x6e ^ (in_f * out_f) as u64 ^ dt as u64);
                let wq = infr_gguf::dequant::dequant_block(dt, &bank).unwrap();
                let w = upload(&be, &pad_to_u32_align(&bank), BufferUsage::Weights);
                let mut rng = Rng(7 + in_f as u64);
                let xv: Vec<f32> = (0..in_f).map(|_| rng.f32()).collect();
                let rv: Vec<f32> = (0..out_f).map(|_| rng.f32()).collect();
                let x = upload(&be, bytemuck::cast_slice(&xv), BufferUsage::Activations);
                let r = upload(&be, bytemuck::cast_slice(&rv), BufferUsage::Activations);
                let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
                let rec = be.recorder().unwrap();
                let rr = res.then_some(r.as_ref());
                rec.linear_kq_fma_nr(
                    dt,
                    w.as_ref(),
                    w_base,
                    x.as_ref(),
                    rr,
                    y.as_ref(),
                    in_f,
                    out_f,
                    nr,
                );
                rec.finish().unwrap();
                let mut got = vec![0f32; out_f];
                be.download(y.as_ref(), bytemuck::cast_slice_mut(&mut got))
                    .unwrap();
                for o in 0..out_f {
                    let row = &wq[w_base + o * in_f..w_base + (o + 1) * in_f];
                    let mut want: f64 = row
                        .iter()
                        .zip(&xv)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum();
                    if res {
                        want += rv[o] as f64;
                    }
                    let want = want as f32;
                    assert!(
                        (got[o] - want).abs() <= 1e-4 + 1e-4 * want.abs(),
                        "{dt:?} {in_f}x{out_f} w_base {w_base} res {res} nr {nr}: out {o} got {} want {want}",
                        got[o]
                    );
                }
                println!("{dt:?} {in_f}x{out_f} w_base {w_base} res {res} nr {nr}: OK");
            }
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn kq_fma_gemv_bw() {
    let be = VulkanBackend::new().unwrap();
    const N_W: usize = 4;
    const REPS: usize = 32;
    for (label, dt, in_f, out_f) in [
        ("gate+up", DType::Q4K, 2560, 20480),
        ("down q4", DType::Q4K, 10240, 2560),
        ("down q6", DType::Q6K, 10240, 2560),
        ("q", DType::Q4K, 2560, 2048),
        ("o", DType::Q4K, 2048, 2560),
        ("lm_head", DType::Q4K, 2560, 262144),
    ] {
        let bank = synth(dt, in_f * out_f, 3);
        let wbytes = bank.len();
        let n_w = if wbytes > 100 << 20 { 1 } else { N_W };
        let ws: Vec<_> = (0..n_w)
            .map(|_| upload(&be, &pad_to_u32_align(&bank), BufferUsage::Weights))
            .collect();
        let xv: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.01).sin()).collect();
        let x = upload(&be, bytemuck::cast_slice(&xv), BufferUsage::Activations);
        let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        let nblk = in_f / 32;
        let qa = be.alloc(in_f, BufferUsage::Activations).unwrap();
        let dact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let time = |f: &dyn Fn(&infr_vulkan::Recorder<'_>, &dyn Buffer)| -> f64 {
            let run = || {
                let rec = be.recorder().unwrap();
                for i in 0..REPS {
                    f(&rec, ws[i % n_w].as_ref());
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
        };
        let mrow = time(&|rec, w| {
            rec.quant_q8(
                x.as_ref(),
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                1,
                in_f,
            );
            rec.linear_mmv_mrow(
                dt,
                w,
                0,
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                None,
                y.as_ref(),
                1,
                in_f,
                out_f,
            );
        });
        let fuseq = time(&|rec, w| {
            rec.linear_mmv_mrow_fuseq(dt, w, 0, x.as_ref(), None, y.as_ref(), 1, in_f, out_f)
        });
        let kq = |nr: usize| {
            time(&|rec, w| {
                rec.linear_kq_fma_nr(dt, w, 0, x.as_ref(), None, y.as_ref(), in_f, out_f, nr)
            })
        };
        let (k4, k8) = (kq(4), kq(8));
        let pick = be.recorder().unwrap().kq_fma_gemv_rows(out_f);
        let best = if pick == 4 { k4 } else { k8 };
        println!(
            "{label:8} {dt:?} {in_f}x{out_f} [{} MiB]: quant_q8+mrow {mrow:4.0}  mrow_fuseq {fuseq:4.0}  kq_fma nr4 {k4:4.0} nr8 {k8:4.0} GB/s  picked nr{pick} ({:+.0}%)",
            wbytes >> 20,
            (best / mrow - 1.0) * 100.0
        );
    }
}

/// Host reference: dequantized-weight × f32-activation dot, summed in f64.
fn host_dot(wq: &[f32], xv: &[f32]) -> f32 {
    wq.iter()
        .zip(xv)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum::<f64>() as f32
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn kq_fma_mrow_parity() {
    let be = VulkanBackend::new().unwrap();
    // rows spans MR-grouping edges (1..4 single group, 5..8 two groups, 9..16 the second band);
    // shapes mirror the m=1 test (non-NR-multiple out_f, odd Q6_K block parity, nonzero w_base,
    // multi-superblock rows).
    for dt in [DType::Q4K, DType::Q6K] {
        for (in_f, out_f, w_base) in [
            (256, 64, 0),
            (768, 30, 0),
            (2560, 301, 0),
            (768, 40, 768 * 3),
        ] {
            for rows in [1usize, 2, 3, 4, 5, 7, 8, 9, 13, 16] {
                let n = w_base + in_f * out_f;
                let bank = synth(dt, n, 0x6e ^ (in_f * out_f) as u64 ^ dt as u64);
                let wq = infr_gguf::dequant::dequant_block(dt, &bank).unwrap();
                let w = upload(&be, &pad_to_u32_align(&bank), BufferUsage::Weights);
                let mut rng = Rng(7 + in_f as u64 + rows as u64 * 131);
                let xv: Vec<f32> = (0..rows * in_f).map(|_| rng.f32()).collect();
                let x = upload(&be, bytemuck::cast_slice(&xv), BufferUsage::Activations);
                let y = be
                    .alloc(rows * out_f * 4, BufferUsage::Activations)
                    .unwrap();
                let rec = be.recorder().unwrap();
                rec.linear_kq_fma_mrow(
                    dt,
                    w.as_ref(),
                    w_base,
                    x.as_ref(),
                    y.as_ref(),
                    rows,
                    in_f,
                    out_f,
                );
                rec.finish().unwrap();
                let mut got = vec![0f32; rows * out_f];
                be.download(y.as_ref(), bytemuck::cast_slice_mut(&mut got))
                    .unwrap();
                // Against the host f64 reference (the m=1 test's tolerance).
                for r in 0..rows {
                    for o in 0..out_f {
                        let row = &wq[w_base + o * in_f..w_base + (o + 1) * in_f];
                        let want = host_dot(row, &xv[r * in_f..(r + 1) * in_f]);
                        let g = got[r * out_f + o];
                        assert!(
                            (g - want).abs() <= 1e-4 + 1e-4 * want.abs(),
                            "{dt:?} {in_f}x{out_f} w_base {w_base} rows {rows}: ({r},{o}) got {g} want {want}"
                        );
                    }
                }
                // BIT-EXACT against one m=1 dispatch per row: the verify-vs-decode identity
                // contract. The m=1 kernel reads a single row, so stage each row separately.
                let nr = be.recorder().unwrap().kq_fma_gemv_rows(out_f);
                for r in 0..rows {
                    let xr = upload(
                        &be,
                        bytemuck::cast_slice(&xv[r * in_f..(r + 1) * in_f]),
                        BufferUsage::Activations,
                    );
                    let yr = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
                    let rec = be.recorder().unwrap();
                    rec.linear_kq_fma_nr(
                        dt,
                        w.as_ref(),
                        w_base,
                        xr.as_ref(),
                        None,
                        yr.as_ref(),
                        in_f,
                        out_f,
                        nr,
                    );
                    rec.finish().unwrap();
                    let mut want = vec![0f32; out_f];
                    be.download(yr.as_ref(), bytemuck::cast_slice_mut(&mut want))
                        .unwrap();
                    for o in 0..out_f {
                        assert!(
                            got[r * out_f + o] == want[o],
                            "{dt:?} {in_f}x{out_f} w_base {w_base} rows {rows} nr {nr}: ({r},{o}) mrow {} != m1 {}",
                            got[r * out_f + o],
                            want[o]
                        );
                    }
                }
                println!(
                    "{dt:?} {in_f}x{out_f} w_base {w_base} rows {rows}: OK (host + m1-bit-exact)"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn kq_fma_mrow_bw() {
    let be = VulkanBackend::new().unwrap();
    const N_W: usize = 4;
    const REPS: usize = 32;
    // Verify-shaped (m = drafts+1): the MTP draft window is <= 7 (spec.k default 6); 13/16 probe
    // the second mrow band.
    for (label, dt, in_f, out_f) in [
        ("down q4", DType::Q4K, 10240, 2560),
        ("down q6", DType::Q6K, 10240, 2560),
        ("q", DType::Q4K, 2560, 2048),
        ("o", DType::Q4K, 2048, 2560),
        ("lm_head", DType::Q4K, 2560, 262144),
    ] {
        let bank = synth(dt, in_f * out_f, 3);
        let wbytes = bank.len();
        let n_w = if wbytes > 100 << 20 { 1 } else { N_W };
        let ws: Vec<_> = (0..n_w)
            .map(|_| upload(&be, &pad_to_u32_align(&bank), BufferUsage::Weights))
            .collect();
        for m in [2usize, 4, 7, 8, 13, 16] {
            if m > 8 && out_f > 8192 {
                continue; // outside MROW16_BAND's wide-n cutoff, like the adapter
            }
            let xv: Vec<f32> = (0..m * in_f).map(|i| (i as f32 * 0.01).sin()).collect();
            let x = upload(&be, bytemuck::cast_slice(&xv), BufferUsage::Activations);
            let y = be.alloc(m * out_f * 4, BufferUsage::Activations).unwrap();
            let nblk = in_f / 32;
            let qa = be.alloc(m * in_f, BufferUsage::Activations).unwrap();
            let dact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
            let sact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
            let time = |f: &dyn Fn(&infr_vulkan::Recorder<'_>, &dyn Buffer)| -> f64 {
                let run = || {
                    let rec = be.recorder().unwrap();
                    for i in 0..REPS {
                        f(&rec, ws[i % n_w].as_ref());
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
            };
            let int8 = time(&|rec, w| {
                rec.quant_q8(
                    x.as_ref(),
                    qa.as_ref(),
                    dact.as_ref(),
                    sact.as_ref(),
                    m,
                    in_f,
                );
                rec.linear_mmv_mrow(
                    dt,
                    w,
                    0,
                    qa.as_ref(),
                    dact.as_ref(),
                    sact.as_ref(),
                    None,
                    y.as_ref(),
                    m,
                    in_f,
                    out_f,
                );
            });
            // f32-dequant mrow: today's path wherever the int8 gate fails (small weights) —
            // and the exact-but-slow fallback the kq tier must beat, not just the int8 one.
            let deq = time(&|rec, w| {
                rec.linear_native_mrow(dt, w, 0, x.as_ref(), y.as_ref(), m, in_f, out_f);
            });
            let kq = time(&|rec, w| {
                rec.linear_kq_fma_mrow(dt, w, 0, x.as_ref(), y.as_ref(), m, in_f, out_f);
            });
            // Tiled fp32 GEMM reference where the shape allows it (n%bn, the padded-dst
            // convention): what the adapter falls through to past the mrow bands.
            let gemm = if out_f % 64 == 0 {
                let c = be.alloc(64 * out_f * 4, BufferUsage::Activations).unwrap();
                Some(time(&|rec, w| {
                    rec.matmul_kq_fma(dt, 64, x.as_ref(), w, 0, c.as_ref(), m, in_f, out_f);
                }))
            } else {
                None
            };
            println!(
                "{label:8} {dt:?} m={m:<2} [{} MiB]: int8-mrow {int8:5.1}  deq-mrow {deq:5.1}  kq_fma-mrow {kq:5.1} GB/s  kq/int8 {mul:+.0}%{gemm}",
                wbytes >> 20,
                mul = (kq / int8 - 1.0) * 100.0,
                gemm = gemm.map(|g| format!("  kq_fma-gemm {g:5.1} GB/s")).unwrap_or_default(),
            );
        }
    }
}

/// End-to-end through the adapter: an MTP-verify-flagged graph (`Graph::mtp_verify`, the
/// speculative-verify batched forward, m = drafts+1 rows) must take the fp32 kq_fma mrow kernel.
/// Scored TIGHT (rel 1e-5): the int8 mrow this replaces carries activation-quant noise ~1e-3, so
/// it cannot pass — the tolerance IS the proof the fp32 arm fired. Shapes sit above the int8
/// gate's 8M-element floor (where the old verify path was int8, not dequant) and span both mrow
/// bands, with an NR-tail out_f and multi-group row counts.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn kq_fma_mtp_verify_tight() {
    use infr_core::graph::{Graph, Op};
    use infr_core::tensor::TensorDesc;
    let be = VulkanBackend::new().unwrap();
    for dt in [DType::Q4K, DType::Q6K] {
        for (m, in_f, out_f) in [
            (2usize, 2048usize, 4096usize),
            (7, 2048, 4096),
            (8, 2048, 4101),
            (13, 1024, 8192),
        ] {
            let case = infr_testkit::LinearCase::new(dt, m, in_f, out_f).with_seed(0x9e7);
            let (w_bytes, x) = infr_testkit::case_inputs(case);
            let w_ref = infr_testkit::dequant_oracle(dt, &w_bytes);
            let want = infr_testkit::ref_linear(&x, &w_ref, m, in_f, out_f);
            let mut g = Graph::new();
            g.mtp_verify = true;
            let xid = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
            let wid = g.weight(TensorDesc::new(vec![out_f, in_f], dt));
            let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
            g.push(Op::Linear {
                x: xid,
                weight: wid,
                dst,
                m: m as u32,
                in_f: in_f as u32,
                out_f: out_f as u32,
                w_off: 0,
            });
            let got = infr_testkit::run_graph(
                &be,
                &g,
                &[(xid, infr_testkit::f32_bytes(&x)), (wid, w_bytes)],
                dst,
                m * out_f,
            );
            let abs = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let ref_mag = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let rel = abs / ref_mag.max(1e-3);
            assert!(
                ref_mag > 1e-3 && rel < 1e-5,
                "{dt:?} verify m={m} {in_f}x{out_f}: rel={rel:.3e} abs={abs:.3e} — not fp32-exact"
            );
            println!("{dt:?} verify m={m} {in_f}x{out_f}: rel={rel:.3e} OK (fp32 arm)");
        }
    }
}
