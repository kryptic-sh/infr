//! Measurement probe, not a test of correctness: how long ONE heavy dispatch, or one submit of
//! several, may run on the `INFR_DEV` device before the driver's watchdog drops the work (plan 12
//! section 0c). Every iteration executes one graph, one submit unless the work bound cuts it, and
//! reports its wall time and whether it completed.
//!
//! The heavy op is a Q4_K `Op::Linear` of `PROBE_ROWS` rows over a 2048 x 22016 weight (SmolLM3-3B's
//! fused gate/up shape). With the int8 kernels off (`INFR_NO_I8DOT=1`) on a device without matrix
//! units it lowers to the per-row GEMV, whose time grows linearly with the rows: the duration knob.
//! `PROBE_PIECES=K` records the same rows as K ops of `PROBE_ROWS / K` rows each (K dispatches,
//! one submit), to tell a limit on one dispatch from a limit on one submit.
//!
//! Knobs: `PROBE_ROWS` (default 64), `PROBE_PIECES` (1), `PROBE_ITERS` (10), and `PROBE_CHAIN`
//! (set: every piece writes the same output, so a barrier separates each from the next). The rest
//! of the configuration comes from the environment as usual: `INFR_SUBMIT_DISPATCHES=0` keeps every
//! iteration one submit (the splitter's kill switch, under which the work bound cuts nothing
//! either), `INFR_TUNE=submit.budget_ms=0` keeps the work bound from cutting the probed dispatch
//! while the dispatch-count splitter stays on, and with neither the probe shows what the work
//! bound does with it.
//!
//! Run: `INFR_DEV=Vulkan1 INFR_NO_I8DOT=1 INFR_SUBMIT_DISPATCHES=0 INFR_TUNE=submit.budget_ms=0
//! PROBE_ROWS=640 cargo test -p infr-vulkan --release --test dispatch_budget_probe -- --ignored
//! --nocapture`
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::config::{Config, ConfigLayer};
use infr_core::{DType, Graph, Op, TensorDesc};
use infr_vulkan::VulkanBackend;
use std::sync::Arc;

const DT: DType = DType::Q4K;
const IN_F: usize = 2048;
const OUT_F: usize = 22016;

fn knob(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{name}: not a count: {v}"))
        })
        .unwrap_or(default)
}

#[test]
#[ignore = "measurement probe: requires a Vulkan GPU and can trip the GPU watchdog"]
fn probe_heavy_dispatch_duration() {
    let rows = knob("PROBE_ROWS", 64);
    let pieces = knob("PROBE_PIECES", 1).max(1);
    let iters = knob("PROBE_ITERS", 10);
    assert!(
        rows.is_multiple_of(pieces),
        "PROBE_ROWS must divide by PROBE_PIECES"
    );
    let env = ConfigLayer::env().expect("environment config");
    let cfg = Config::load_from_layers(&[env]);
    let be = VulkanBackend::new_with(Arc::new(cfg)).expect("Vulkan device");
    let part = rows / pieces;

    let mut g = Graph::new();
    let wi = g.weight(TensorDesc::new(vec![OUT_F, IN_F], DT));
    let xi = g.input(TensorDesc::new(vec![part, IN_F], DType::F32));
    let chain = std::env::var_os("PROBE_CHAIN").is_some();
    let mut outs = Vec::new();
    for i in 0..pieces {
        // Chained: every piece writes the same output, so the recorder orders each after the
        // one before it with a barrier; otherwise the pieces are independent and may overlap.
        if chain && i > 0 {
            g.push(Op::Linear {
                x: xi,
                weight: wi,
                dst: outs[0],
                m: part as u32,
                in_f: IN_F as u32,
                out_f: OUT_F as u32,
                w_off: 0,
            });
            continue;
        }
        let o = g.output(TensorDesc::new(vec![part, OUT_F], DType::F32));
        g.push(Op::Linear {
            x: xi,
            weight: wi,
            dst: o,
            m: part as u32,
            in_f: IN_F as u32,
            out_f: OUT_F as u32,
            w_off: 0,
        });
        outs.push(o);
    }
    let plan = be.compile(&g).expect("compile");
    let w = infr_testkit::synth_weight(DT, OUT_F * IN_F, 7);
    let wb = be.alloc(w.len(), BufferUsage::Weights).unwrap();
    be.upload(wb.as_ref(), &w).unwrap();
    let x = infr_testkit::gen_f32(part * IN_F, 3);
    let xb = be.alloc(part * IN_F * 4, BufferUsage::Activations).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let ybs: Vec<Box<dyn Buffer>> = (0..outs.len())
        .map(|_| {
            be.alloc(part * OUT_F * 4, BufferUsage::Activations)
                .unwrap()
        })
        .collect();
    let mut binds = Bindings::new();
    binds.bind(wi, wb.as_ref());
    binds.bind(xi, xb.as_ref());
    for (&o, yb) in outs.iter().zip(&ybs) {
        binds.bind(o, yb.as_ref());
    }
    println!(
        "probe: {rows} rows as {pieces} dispatch(es) of {part} rows (chained: {chain}), {iters} iterations"
    );
    let mut failed = 0;
    for i in 0..iters {
        let t = std::time::Instant::now();
        let r = be.execute(plan.as_ref(), &binds);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        match r {
            Ok(()) => println!("probe: iter {i}: completed in {ms:.0} ms"),
            Err(e) => {
                println!("probe: iter {i}: FAILED after {ms:.0} ms: {e}");
                failed += 1;
                break;
            }
        }
    }
    println!(
        "probe: rows={rows} pieces={pieces}: {failed} failure(s); cuts {:?}; measured GFLOP/s {:?}",
        be.segment_cuts(),
        be.work_measured_gflops()
    );
}
