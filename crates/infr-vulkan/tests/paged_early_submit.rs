//! The early-submit cadence of a PAGED forward (`paging.early_submit`) cuts the forward into
//! segments the GPU starts on while the host is still recording, starts over after a staging-ring
//! rotation, is off when configured off and on every resident forward,
//! and never stops the submit splitter's watchdog cut.
//!
//! The forward is a chain of streamed `Op::Linear`s, each followed by an `Op::RmsNorm` that keeps
//! the activations in range for the next one. Every Linear's weight is a block of one dense
//! streaming pool with fewer slots than blocks, so the pass stages and evicts between the cuts.
//! With the upload ring at its smallest (`Ring::Rotating`, one slot per half) nearly every miss
//! also rotates the ring, so ring halves are rewritten while copies that read them sit in segments
//! the cadence submitted early.
//!
//! Observables: `VulkanBackend::segment_cuts`, which counts the segments a forward submitted
//! before its recording ended by cause (watchdog cut, early cut, ring rotation), and the chain's
//! output against the same chain over resident weights on a backend with no pager at all.
//!
//! What this does NOT prove: that `Recorder::seed_barrier` orders a segment after the ones before
//! it. Each layer reads what an earlier segment wrote, but the bit-identity checks stayed green
//! with the barrier removed on the RX 7900 XTX under the AMD Windows driver (`docs/backlog.md`,
//! B185). On that driver they show that the cadence ran (the cut counts) and left the output
//! unchanged, not that the barrier is what keeps it so.
//!
//! Run: `cargo test -p infr-vulkan --release --test paged_early_submit -- --ignored --nocapture`
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::config::{Config, ConfigLayer};
use infr_core::{DType, Graph, Op, TensorDesc};
use infr_vulkan::pager::{
    buffer_identity, DenseBytes, DensePagerLayout, DensePoolSpec, DenseSource,
};
use infr_vulkan::{SegmentCuts, VulkanBackend};
use std::sync::Arc;

const DT: DType = DType::Q8_0;
const DIM: usize = 256;
/// Linear layers in the chain, one streamed block each.
const LAYERS: usize = 8;
/// Arena slots: fewer than the layers, so every pass misses and evicts.
const SLOTS: usize = 3;

/// A backend on the `INFR_DEV` device with the given early-submit cadence and, optionally, a
/// forced submit-splitter cap.
fn backend(early_submit: usize, submit_dispatches: Option<usize>) -> VulkanBackend {
    let mut cfg = Config::default();
    let env = ConfigLayer::env().expect("test environment config");
    cfg.device.dev = Config::load_from_layers(&[env]).device.dev;
    cfg.paging.early_submit = early_submit;
    cfg.device.submit_dispatches = submit_dispatches;
    VulkanBackend::new_with(Arc::new(cfg)).expect("Vulkan device")
}

fn weights() -> Vec<Vec<u8>> {
    (0..LAYERS)
        .map(|l| infr_testkit::synth_weight(DT, DIM * DIM, 100 + l as u32))
        .collect()
}

/// Which upload ring a streamed leg stages through.
#[derive(Clone, Copy)]
enum Ring {
    /// The default ring (`ring_bytes: 0`), far larger than the whole chain: it never rotates.
    Default,
    /// The smallest ring, one slot per half: staging rotates at nearly every miss.
    Rotating,
}

/// Install a dense streaming session holding every layer's weight as one block of one pool, and
/// return the placeholder buffers that stand for them in the graph's bindings.
fn stream_weights(be: &VulkanBackend, ws: &[Vec<u8>], ring: Ring) -> Vec<Box<dyn Buffer>> {
    let (_, blk_bytes) = infr_gguf::block_layout(DT);
    let stride = ws[0].len().next_multiple_of(blk_bytes * 4);
    be.init_dense_pager(DensePagerLayout {
        pools: vec![DensePoolSpec {
            slot_bytes: stride,
            n_slots: SLOTS,
            n_blocks: LAYERS,
            host: None,
        }],
        ring_bytes: match ring {
            Ring::Default => 0,
            // `DensePagerSession::new` floors each half at the largest slot, so this is the
            // smallest ring there is: one slot per half.
            Ring::Rotating => 2 * stride,
        },
    })
    .unwrap();
    ws.iter()
        .enumerate()
        .map(|(l, wb)| {
            let placeholder = be.alloc_uninit(4, BufferUsage::Weights).unwrap();
            be.register_dense_stream(
                0,
                buffer_identity(placeholder.as_ref()),
                DenseSource {
                    bytes: DenseBytes::Mmap(vec![
                        Arc::new(wb.clone()) as Arc<dyn AsRef<[u8]> + Send + Sync>
                    ]),
                    block_id: l as u32,
                },
            )
            .unwrap();
            placeholder
        })
        .collect()
}

fn resident_weights(be: &VulkanBackend, ws: &[Vec<u8>]) -> Vec<Box<dyn Buffer>> {
    ws.iter()
        .map(|wb| {
            let b = be.alloc(wb.len(), BufferUsage::Weights).unwrap();
            be.upload(b.as_ref(), wb).unwrap();
            b
        })
        .collect()
}

/// One decode-shaped pass (m = 1) of the chain, the weights bound to `ws`.
fn run(be: &VulkanBackend, ws: &[Box<dyn Buffer>], x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new();
    let xi = g.input(TensorDesc::new(vec![1, DIM], DType::F32));
    let norm_w = g.weight(TensorDesc::new(vec![DIM], DType::F32));
    let wis: Vec<_> = (0..LAYERS)
        .map(|_| g.weight(TensorDesc::new(vec![DIM, DIM], DT)))
        .collect();
    let mut h = xi;
    let mut out = xi;
    for (l, &wi) in wis.iter().enumerate() {
        let t = g.internal(TensorDesc::new(vec![1, DIM], DType::F32));
        g.push(Op::Linear {
            x: h,
            weight: wi,
            dst: t,
            m: 1,
            in_f: DIM as u32,
            out_f: DIM as u32,
            w_off: 0,
        });
        let next = if l + 1 == LAYERS {
            out = g.output(TensorDesc::new(vec![1, DIM], DType::F32));
            out
        } else {
            g.internal(TensorDesc::new(vec![1, DIM], DType::F32))
        };
        g.push(Op::RmsNorm {
            x: t,
            weight: norm_w,
            dst: next,
            rows: 1,
            dim: DIM as u32,
            eps: 1e-6,
        });
        h = next;
    }
    let plan = be.compile(&g).expect("compile");
    let xb = be.alloc(DIM * 4, BufferUsage::Activations).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let nb = be.alloc(DIM * 4, BufferUsage::Weights).unwrap();
    be.upload(nb.as_ref(), bytemuck::cast_slice(&vec![1f32; DIM]))
        .unwrap();
    let yb = be.alloc(DIM * 4, BufferUsage::Readback).unwrap();
    let mut binds = Bindings::new();
    binds.bind(xi, xb.as_ref());
    binds.bind(norm_w, nb.as_ref());
    for (&wi, w) in wis.iter().zip(ws) {
        binds.bind(wi, w.as_ref());
    }
    binds.bind(out, yb.as_ref());
    be.execute(plan.as_ref(), &binds).expect("execute");
    let mut y = vec![0f32; DIM];
    be.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut y))
        .unwrap();
    y
}

/// The chain over resident weights on a backend with no pager: the reference every paged leg must
/// reproduce bit for bit.
fn reference(ws: &[Vec<u8>], x: &[f32]) -> Vec<f32> {
    let be = backend(0, None);
    let y = run(&be, &resident_weights(&be, ws), x);
    assert!(y.iter().all(|v| v.is_finite()), "reference is not finite");
    y
}

/// Two passes of the streamed chain on `be`, the cuts they made, and their outputs.
fn streamed(
    be: &VulkanBackend,
    ws: &[Vec<u8>],
    x: &[f32],
    ring: Ring,
) -> (SegmentCuts, Vec<Vec<f32>>) {
    let placeholders = stream_weights(be, ws, ring);
    let before = be.segment_cuts();
    let ys = (0..2).map(|_| run(be, &placeholders, x)).collect();
    (cuts_since(be, before), ys)
}

/// The cuts `be` made since `before` (an earlier [`VulkanBackend::segment_cuts`]).
fn cuts_since(be: &VulkanBackend, before: SegmentCuts) -> SegmentCuts {
    let after = be.segment_cuts();
    SegmentCuts {
        watchdog: after.watchdog - before.watchdog,
        early: after.early - before.early,
        rotation: after.rotation - before.rotation,
        drain: after.drain - before.drain,
    }
}

fn assert_bitwise(label: &str, ys: &[Vec<f32>], want: &[f32]) {
    for (pass, y) in ys.iter().enumerate() {
        let mismatch = y
            .iter()
            .zip(want)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert_eq!(
            mismatch, None,
            "{label}, pass {pass}: differs from the resident reference at element {mismatch:?}"
        );
    }
}

/// With the cadence at one dispatch, the forward is cut before its recording ends (an early cut is
/// counted), and the output is the resident one bit for bit.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn early_submit_cuts_a_paged_forward_and_keeps_its_output() {
    let ws = weights();
    let x = infr_testkit::gen_f32(DIM, 7);
    let want = reference(&ws, &x);
    // No forced cap: on a discrete GPU the splitter never cuts here, on a splitting device it may.
    let be = backend(1, None);
    let (cuts, ys) = streamed(&be, &ws, &x, Ring::Default);
    println!("early_submit=1, default ring: {cuts:?}");
    assert!(cuts.early > 0, "no segment was submitted early: {cuts:?}");
    assert_eq!(cuts.rotation, 0, "the default ring rotated: {cuts:?}");
    assert_bitwise("early_submit=1", &ys, &want);
}

/// The same chain through a ring that rotates at nearly every miss: ring halves are rewritten
/// while early-cut segments are in flight, and the output is still the resident one bit for bit.
/// The cadence starts over at every rotation (`adapter::EarlySubmit`), so the rotating run takes
/// more early cuts than the run that never rotates; a cadence that carried on across rotations
/// would take no more, since a rotation only shortens the recorder the cadence counts in.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn early_submit_starts_over_after_a_ring_rotation() {
    let ws = weights();
    let x = infr_testkit::gen_f32(DIM, 7);
    let want = reference(&ws, &x);
    let (plain, _) = streamed(&backend(1, None), &ws, &x, Ring::Default);
    let be = backend(1, None);
    let (cuts, ys) = streamed(&be, &ws, &x, Ring::Rotating);
    println!("early_submit=1: default ring {plain:?}, rotating ring {cuts:?}");
    assert!(cuts.rotation > 0, "the ring never rotated: {cuts:?}");
    assert!(
        cuts.early > plain.early,
        "ring rotations did not restart the early cadence: {} early cuts rotating, {} without",
        cuts.early,
        plain.early
    );
    assert_bitwise("early_submit=1, rotating ring", &ys, &want);
}

/// A resident forward takes no early cut with the cadence at its default: the cadence is for paged
/// forwards only, and a backend with no pager session never runs one.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_resident_forward_takes_no_early_cut() {
    let ws = weights();
    let x = infr_testkit::gen_f32(DIM, 7);
    let be = backend(Config::default().paging.early_submit, None);
    let before = be.segment_cuts();
    let y = run(&be, &resident_weights(&be, &ws), &x);
    let cuts = cuts_since(&be, before);
    println!("resident, default early_submit: {cuts:?}");
    assert!(
        y.iter().all(|v| v.is_finite()),
        "resident output is not finite"
    );
    assert_eq!(cuts.early, 0, "a resident forward was cut early: {cuts:?}");
}

/// `paging.early_submit = 0` turns the cadence off: no early cut, same output.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn early_submit_off_makes_no_early_cut() {
    let ws = weights();
    let x = infr_testkit::gen_f32(DIM, 7);
    let want = reference(&ws, &x);
    let be = backend(0, None);
    let (cuts, ys) = streamed(&be, &ws, &x, Ring::Rotating);
    println!("early_submit=0: {cuts:?}");
    assert_eq!(cuts.early, 0, "early cuts with the cadence off: {cuts:?}");
    assert_bitwise("early_submit=0", &ys, &want);
}

/// The splitter's cap still cuts with the cadence on: with the cadence starting AT the cap, every
/// cut is the watchdog's (it is checked first, and the cadence only grows from there), and the
/// forward does not loosen the cap.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn the_watchdog_cut_survives_early_submit() {
    let ws = weights();
    let x = infr_testkit::gen_f32(DIM, 7);
    let want = reference(&ws, &x);
    const CAP: usize = 2;
    let be = backend(CAP, Some(CAP));
    let (cuts, ys) = streamed(&be, &ws, &x, Ring::Rotating);
    println!("early_submit={CAP}, submit_dispatches={CAP}: {cuts:?}");
    assert!(cuts.watchdog > 0, "the splitter never cut: {cuts:?}");
    assert_eq!(
        cuts.early, 0,
        "a cut at the splitter's cap was attributed to the cadence: {cuts:?}"
    );
    assert!(
        be.submit_cap_now() <= CAP && be.submit_cap_now() > 0,
        "the splitter's cap moved off {CAP}: {}",
        be.submit_cap_now()
    );
    assert_bitwise("early_submit=2, cap=2", &ys, &want);
}
