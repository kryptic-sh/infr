//! A record-once decode replay that fails part-way through recording must free the command buffer
//! and descriptor pools it already holds. `Recorder` has no `Drop`, so an error path that skips the
//! teardown leaks them silently: the decode just falls over with its error, and the only witness
//! is the Khronos layer reporting the leaked objects when the device is destroyed (see
//! `gpuav_support`).
//!
//! The failure is an unbound KV cache: the rope op records first, then the attention op cannot
//! resolve its key cache, so the recording holds live dispatches when it errors. Push descriptors
//! are turned off: with them a recording allocates no descriptor pool, and a dropped command
//! buffer goes with its command pool at device teardown, so the leak would have nothing to show.
//!
//! Run: cargo test -p infr-vulkan --test replay_error_teardown -- --ignored --nocapture
mod gpuav_support;

use std::io::Write;
use std::sync::Arc;

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::config::{Config, ConfigLayer};
use infr_core::{AttnMask, DType, Graph, Op, TensorDesc};
use infr_vulkan::VulkanBackend;

#[test]
#[ignore = "requires a Vulkan GPU"]
fn failed_replay_recording_frees_its_segment() {
    let log = gpuav_support::enable("replay-error");
    let mut cfg = Config::load_from_layers(&[ConfigLayer::env().expect("environment config")]);
    cfg.kernels.vulkan.push_desc = false;
    // The leak under test is the replay path's; `INFR_SEAM_NO_REPLAY` in the environment would
    // quietly route the execute through the static path instead.
    cfg.kernels.vulkan.no_replay = false;
    let Ok(be) = VulkanBackend::new_with(Arc::new(cfg)) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // The single-token decode shape the replay path takes (rope + one attention row, f16 KV).
    let (nh, nkv, hd) = (2usize, 1usize, 128usize);
    let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
    let mut g = Graph::new();
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
    {
        let buf = |bytes: usize, usage| be.alloc(bytes, usage).unwrap();
        let xb = buf(nh * hd * 4, BufferUsage::Activations);
        let wb = buf(hd * 4, BufferUsage::Weights);
        let pb = buf(4, BufferUsage::Staging);
        be.upload(pb.as_ref(), bytemuck::cast_slice(&[0i32]))
            .unwrap();
        let vb = buf(nkv * hd * 2, BufferUsage::KvCache);
        let ob = buf(nh * hd * 4, BufferUsage::Activations);
        let mut bind = Bindings::new();
        bind.bind(x, xb.as_ref())
            .bind(qw, wb.as_ref())
            .bind(pos, pb.as_ref())
            .bind(vc, vb.as_ref())
            .bind(dst, ob.as_ref());
        let plan = be.compile(&g).unwrap();
        let err = be
            .execute(plan.as_ref(), &bind)
            .expect_err("an unbound key cache must fail the recording");
        eprintln!("recording failed as intended: {err}");
    }
    drop(be);
    // The leak is visible only to the layer, so without it this run checked nothing.
    if !gpuav_support::assert_clean(
        &log,
        "failed_replay_recording_frees_its_segment",
        "the segment teardown",
    ) {
        let _ = writeln!(
            std::io::stderr().lock(),
            "skip: failed_replay_recording_frees_its_segment needs the Khronos validation layer"
        );
    }
}
