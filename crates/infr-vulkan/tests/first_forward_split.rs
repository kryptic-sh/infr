//! The submit splitter protects a slow device's FIRST forward (issue 94): on a device in the
//! slow class (`tuning::DeviceFacts::slow_class`: wave64-only, or neither matrix units nor an
//! accelerated int8 dot) a prefill-shaped forward is cut into submits from its first run, without
//! changing a bit of its output, while a decode-shaped (single-token) forward stays one submit.
//! On every other discrete device nothing is cut.
//!
//! The slow class is forced on the hardware here with `kernels.vulkan.force_wave64`
//! (`INFR_FORCE_WAVE64`); the reference for the output is the same forced backend with the
//! splitter off (`device.submit_dispatches = 0`), since the wave64 path runs other kernels than
//! the native one.
//!
//! Observable: `VulkanBackend::segment_cuts().watchdog`, the submits a forward cut before its
//! recording ended because the splitter's cap was reached.
//!
//! Run: `cargo test -p infr-vulkan --release --test first_forward_split -- --ignored --nocapture`
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::config::{Config, ConfigLayer};
use infr_core::{DType, Graph, Op, TensorDesc};
use infr_vulkan::VulkanBackend;
use std::sync::Arc;

const DT: DType = DType::Q8_0;
const DIM: usize = 256;
/// Linear + RmsNorm pairs: 2 * LAYERS dispatches or more, past both initial caps (64, 128).
const LAYERS: usize = 80;
/// Rows of the prefill-shaped forward.
const ROWS: usize = 8;

/// A backend on the `INFR_DEV` device; `wave64` forces the slow class.
fn backend(wave64: bool, submit_dispatches: Option<usize>) -> VulkanBackend {
    let mut cfg = Config::default();
    let env = ConfigLayer::env().expect("test environment config");
    cfg.device.dev = Config::load_from_layers(&[env]).device.dev;
    cfg.device.submit_dispatches = submit_dispatches;
    infr_testkit::honour_env(&mut cfg);
    // After `honour_env`, which copies the environment's `force_wave64` over the config's.
    cfg.kernels.vulkan.force_wave64 |= wave64;
    VulkanBackend::new_with(Arc::new(cfg)).expect("Vulkan device")
}

/// One forward of `rows` rows through the chain; returns its output and the watchdog cuts it took.
fn run(be: &VulkanBackend, rows: usize) -> (Vec<f32>, u64) {
    let ws: Vec<Vec<u8>> = (0..LAYERS)
        .map(|l| infr_testkit::synth_weight(DT, DIM * DIM, 300 + l as u32))
        .collect();
    let x = infr_testkit::gen_f32(rows * DIM, 11);
    let mut g = Graph::new();
    let xi = g.input(TensorDesc::new(vec![rows, DIM], DType::F32));
    let norm_w = g.weight(TensorDesc::new(vec![DIM], DType::F32));
    let wis: Vec<_> = (0..LAYERS)
        .map(|_| g.weight(TensorDesc::new(vec![DIM, DIM], DT)))
        .collect();
    let mut h = xi;
    let mut out = xi;
    for (l, &wi) in wis.iter().enumerate() {
        let t = g.internal(TensorDesc::new(vec![rows, DIM], DType::F32));
        g.push(Op::Linear {
            x: h,
            weight: wi,
            dst: t,
            m: rows as u32,
            in_f: DIM as u32,
            out_f: DIM as u32,
            w_off: 0,
        });
        let next = if l + 1 == LAYERS {
            out = g.output(TensorDesc::new(vec![rows, DIM], DType::F32));
            out
        } else {
            g.internal(TensorDesc::new(vec![rows, DIM], DType::F32))
        };
        g.push(Op::RmsNorm {
            x: t,
            weight: norm_w,
            dst: next,
            rows: rows as u32,
            dim: DIM as u32,
            eps: 1e-6,
        });
        h = next;
    }
    let plan = be.compile(&g).expect("compile");
    let wbufs: Vec<Box<dyn Buffer>> = ws
        .iter()
        .map(|wb| {
            let b = be.alloc(wb.len(), BufferUsage::Weights).unwrap();
            be.upload(b.as_ref(), wb).unwrap();
            b
        })
        .collect();
    let xb = be.alloc(rows * DIM * 4, BufferUsage::Activations).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let nb = be.alloc(DIM * 4, BufferUsage::Weights).unwrap();
    be.upload(nb.as_ref(), bytemuck::cast_slice(&vec![1f32; DIM]))
        .unwrap();
    let yb = be.alloc(rows * DIM * 4, BufferUsage::Readback).unwrap();
    let mut binds = Bindings::new();
    binds.bind(xi, xb.as_ref());
    binds.bind(norm_w, nb.as_ref());
    for (&wi, w) in wis.iter().zip(&wbufs) {
        binds.bind(wi, w.as_ref());
    }
    binds.bind(out, yb.as_ref());
    let before = be.segment_cuts().watchdog;
    be.execute(plan.as_ref(), &binds).expect("execute");
    let cuts = be.segment_cuts().watchdog - before;
    let mut y = vec![0f32; rows * DIM];
    be.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut y))
        .unwrap();
    assert!(y.iter().all(|v| v.is_finite()), "output is not finite");
    (y, cuts)
}

fn assert_bitwise(label: &str, got: &[f32], want: &[f32]) {
    let mismatch = got
        .iter()
        .zip(want)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(mismatch, None, "{label}: differs at element {mismatch:?}");
}

/// In the slow class the first prefill-shaped forward is split, bit-identical to the unsplit
/// one, and a single-token forward is not split at all.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn gpu_seam_first_forward_split_on_a_slow_device() {
    let (want, unsplit_cuts) = run(&backend(true, Some(0)), ROWS);
    assert_eq!(unsplit_cuts, 0);
    let be = backend(true, None);
    println!("forced wave64: initial cap {}", be.submit_cap_now());
    let (got, cuts) = run(&be, ROWS);
    println!("forced wave64: first prefill forward cut {cuts} time(s)");
    assert!(
        cuts > 0,
        "the first forward of a slow-class device was not split"
    );
    assert_bitwise("split first forward", &got, &want);
    // On a discrete device a single-token forward ignores the slow-class seed (a decode step is
    // short); an integrated device's own cap applies to every forward.
    if !be.capabilities().integrated {
        let (_, decode_cuts) = run(&be, 1);
        assert_eq!(decode_cuts, 0, "a single-token forward was split");
    }
}

/// On the device as it is, a discrete device outside the slow class cuts nothing; an integrated
/// one may (its own cap), so only the discrete case is asserted.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn gpu_seam_first_forward_unsplit_on_a_fast_discrete_device() {
    let be = backend(false, None);
    let integrated = be.capabilities().integrated;
    let (_, cuts) = run(&be, ROWS);
    println!("native: first prefill forward cut {cuts} time(s) (integrated: {integrated})");
    if !integrated {
        assert_eq!(cuts, 0, "a fast discrete device split its first forward");
    }
}
