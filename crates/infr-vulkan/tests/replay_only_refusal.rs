//! A graph built for record-once decode replay (`Graph::replay_only`) has its position baked in
//! and is right only when replayed. When the adapter will not replay its plan, running it
//! statically would decode every later token at the baked position, which is wrong output with no
//! error; `execute` and `execute_chain` must refuse it instead, and `decode_replay_ready` must say
//! no first so a caller that asks never gets that far.
//!
//! `kernels.vulkan.no_replay` is the switch that makes the same single-token decode graph
//! ineligible, so the two halves differ in nothing but the adapter's decision.
//!
//! Run: cargo test -p infr-vulkan --test replay_only_refusal -- --ignored --nocapture

use std::sync::Arc;

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::config::{Config, ConfigLayer};
use infr_core::{AttnMask, DType, Graph, Op, TensorDesc};
use infr_vulkan::VulkanBackend;

/// A backend on the environment's device with record-once replay allowed or forced off.
fn backend(no_replay: bool) -> Option<VulkanBackend> {
    let mut cfg = Config::load_from_layers(&[ConfigLayer::env().expect("environment config")]);
    cfg.kernels.vulkan.no_replay = no_replay;
    match VulkanBackend::new_with(Arc::new(cfg)) {
        Ok(be) => Some(be),
        Err(e) => {
            eprintln!("skip: no Vulkan device ({e})");
            None
        }
    }
}

/// The single-token decode shape the replay path takes (rope + one attention row over an f16
/// KV cache), built for replay when `replay_only`, and run once at position 0. Returns what
/// `decode_replay_ready`, `execute` and `execute_chain` said.
fn run(
    be: &VulkanBackend,
    replay_only: bool,
) -> (
    bool,
    infr_core::Result<()>,
    infr_core::Result<Option<Vec<u32>>>,
) {
    let (nh, nkv, hd) = (2usize, 1usize, 128usize);
    let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
    let mut g = Graph::new();
    g.replay_only = replay_only;
    let x = g.input(f32d(nh * hd));
    let qw = g.weight(f32d(hd));
    let pos = g.input(TensorDesc::new(vec![1], DType::I32));
    let kc = g.input(TensorDesc::new(vec![nkv * hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![nkv * hd], DType::F16));
    let qa = g.internal(f32d(nh * hd));
    let dst = g.output(f32d(nh * hd));
    g.push(Op::QkNormRope {
        x,
        weight: qw,
        positions: pos,
        dst: qa,
        rows: 1,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: hd as u32,
        theta: 1e6,
        eps: 1e-6,
        freq_factors: None,
        x_stride: 0,
    });
    g.push(Op::Attention {
        q: qa,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: 1,
        kv_len: 1,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: AttnMask::Causal,
        pos: 0,
        sinks: None,
        key_bias: None,
    });
    let buf = |bytes: usize, usage| be.alloc(bytes, usage).unwrap();
    let xb = buf(nh * hd * 4, BufferUsage::Activations);
    let wb = buf(hd * 4, BufferUsage::Weights);
    be.upload(wb.as_ref(), bytemuck::cast_slice(&vec![1f32; hd]))
        .unwrap();
    let pb = buf(4, BufferUsage::Staging);
    be.upload(pb.as_ref(), bytemuck::cast_slice(&[0i32]))
        .unwrap();
    let kb = buf(nkv * hd * 2, BufferUsage::KvCache);
    let vb = buf(nkv * hd * 2, BufferUsage::KvCache);
    let ob = buf(nh * hd * 4, BufferUsage::Activations);
    let mut bind = Bindings::new();
    bind.bind(x, xb.as_ref())
        .bind(qw, wb.as_ref())
        .bind(pos, pb.as_ref())
        .bind(kc, kb.as_ref())
        .bind(vc, vb.as_ref())
        .bind(dst, ob.as_ref());
    let plan = be.compile(&g).unwrap();
    let ready = be.decode_replay_ready(plan.as_ref(), &bind);
    let chained = be.execute_chain(plan.as_ref(), &bind, 2);
    let executed = be.execute(plan.as_ref(), &bind);
    be.sync().unwrap();
    (ready, executed, chained)
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_replay_only_plan_the_adapter_will_not_replay_is_refused() {
    let Some(be) = backend(true) else {
        return;
    };
    eprintln!("vulkan device: {}", be.capabilities().name);
    let (ready, executed, chained) = run(&be, true);
    assert!(
        !ready,
        "no_replay is set, yet the plan reports it will replay"
    );
    let err = executed
        .expect_err("a replay-only plan ran statically")
        .to_string();
    eprintln!("execute refused as intended: {err}");
    assert!(
        err.contains("Graph::replay_only") && err.contains("no_replay"),
        "the refusal must name the flag and why the plan does not replay: {err}"
    );
    let err = chained
        .expect_err("execute_chain declined a replay-only plan instead of refusing it")
        .to_string();
    assert!(err.contains("Graph::replay_only"), "{err}");

    // The same graph, not built for replay, is an ordinary static decode at position 0: the
    // refusal is about the flag, not a fixture the static path cannot run.
    let (ready, executed, chained) = run(&be, false);
    assert!(!ready);
    executed.expect("an ordinary graph must still run statically");
    assert!(chained
        .expect("an ordinary graph declines a chain")
        .is_none());
    assert_eq!(
        be.decode_recordings(),
        0,
        "an ineligible plan was recorded for replay"
    );
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_replay_only_plan_the_adapter_replays_runs() {
    let Some(be) = backend(false) else {
        return;
    };
    eprintln!("vulkan device: {}", be.capabilities().name);
    let (ready, executed, chained) = run(&be, true);
    assert!(
        ready,
        "an eligible single-token decode must report it will replay"
    );
    executed.expect("an eligible replay-only plan must run");
    // No sampler in this graph, so there is nothing to chain; the chain declines, it does not fail.
    assert!(chained.expect("execute_chain").is_none());
    assert_eq!(
        be.decode_recordings(),
        1,
        "the plan reported ready but the adapter did not record one replay for it"
    );
}
