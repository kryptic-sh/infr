//! The work bound (plan 12 S2, `crate::work`) cuts a heavy prefill GEMM or GEMV into ranges of
//! its workgroup grid, each in its own submit, without changing a bit of the output.
//!
//! Each case runs a two-layer chain of `Op::Linear` once with the bound off
//! (`submit.budget_ms = 0`) and once with a budget so small, and a prior so slow
//! (`submit.prior_gflops = 1`), that every tagged dispatch is cut, on fresh backends, and compares
//! the outputs bit for bit. The forced kernel masks route the same graph to each kernel family that
//! can be cut: the per-row GEMV on a two-dimensional and on a one-row grid (int8 and matrix kernels
//! off), the int8 GEMMs (matrix kernels off) and the device's default tier.
//!
//! Observables: `VulkanBackend::work_splits` (dispatches cut), `segment_cuts().work` (submits the
//! cuts opened) and `split_kernels` (which kernels were cut), so a case that cut nothing fails
//! instead of comparing two identical unsplit runs.
//!
//! A case skips, with a printed line, only on a machine with no Vulkan device and no `INFR_DEV`
//! request. Where a device is present or was asked for, a backend that does not open fails the
//! test: every backend here is opened with tuning keys of the work bound, and a key the backend
//! rejects must not turn into a pass that compared nothing. "No device" includes an enumeration
//! that itself fails (no loader, or a broken one): with `INFR_DEV` unset that machine skips too,
//! since nothing here can tell a broken loader from an absent one.
//!
//! Run: `INFR_DEV=Vulkan0 cargo test -p infr-vulkan --release --test work_split_parity --
//! --ignored --nocapture` (also under `INFR_FORCE_WAVE64=1`).
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::config::Config;
use infr_core::{DType, Graph, Op, TensorDesc};
use infr_vulkan::VulkanBackend;
use std::sync::Arc;

mod slow_class_support;
use slow_class_support::SlowBy;

/// A backend on the `INFR_DEV` device with the environment's settings, then `extra` over them.
/// `None` only when the machine has no Vulkan device and none was requested (see the module doc);
/// any other failure to open panics with the backend's error.
fn backend(extra: &[(&str, &str)]) -> Option<VulkanBackend> {
    let mut pairs: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| k.starts_with("INFR_") && k != "INFR_TUNE")
        .collect();
    for (k, v) in extra {
        pairs.retain(|(n, _)| n != k);
        pairs.push((k.to_string(), v.to_string()));
    }
    let get = |k: &str| pairs.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
    let layer = infr_core::config::env::parse(&get).expect("environment config");
    let cfg = Arc::new(Config::load_from_layers(&[layer]));
    let opened = VulkanBackend::new_with(Arc::clone(&cfg));
    if let Err(e) = &opened {
        let present = VulkanBackend::enumerate_devices(&cfg).is_ok_and(|devs| !devs.is_empty());
        assert!(
            !present && get("INFR_DEV").is_none(),
            "the Vulkan backend did not open with {extra:?} (a Vulkan device is present: \
             {present}; INFR_DEV: {:?}): {e}",
            get("INFR_DEV")
        );
    }
    infr_testkit::gpu_env::gpu_or_skip(opened)
}

/// What one run produced: the output, and what the work bound did.
struct Run {
    y: Vec<f32>,
    splits: u64,
    /// The ranges the cut dispatches were recorded as.
    ranges: u64,
    work_cuts: u64,
    /// The dispatch count the forward reported to the submit splitter.
    dispatches: u64,
    kernels: Vec<&'static str>,
}

/// `x[rows, dim] -> Linear -> Linear -> y[rows, dim]` with `dt` weights, on `be`.
fn run(be: &VulkanBackend, dt: DType, rows: usize, dim: usize) -> Run {
    run_chain(be, dt, rows, &[dim, dim, dim])
}

/// `x[rows, widths[0]]` through one `Op::Linear` per following width, with `dt` weights, on `be`.
fn run_chain(be: &VulkanBackend, dt: DType, rows: usize, widths: &[usize]) -> Run {
    let weight = |n: usize, seed: u32| match dt {
        // Random bit patterns are not finite f16 values: scaled noise instead.
        DType::F16 => infr_testkit::f16_bytes(
            &infr_testkit::gen_f32(n, seed as usize)
                .iter()
                .map(|v| v * 0.05)
                .collect::<Vec<_>>(),
        ),
        _ => infr_testkit::synth_weight(dt, n, seed),
    };
    let dim = *widths.last().expect("a width");
    let x = infr_testkit::gen_f32(rows * widths[0], 5);
    let mut g = Graph::new();
    let xi = g.input(TensorDesc::new(vec![rows, widths[0]], DType::F32));
    let y = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
    let upload = |bytes: &[u8], usage| {
        let b = be.alloc(bytes.len(), usage).unwrap();
        be.upload(b.as_ref(), bytes).unwrap();
        b
    };
    let mut weights = Vec::new();
    let mut src = xi;
    for (l, io) in widths.windows(2).enumerate() {
        let (in_f, out_f) = (io[0], io[1]);
        let wi = g.weight(TensorDesc::new(vec![out_f, in_f], dt));
        let dst = if l + 2 == widths.len() {
            y
        } else {
            g.internal(TensorDesc::new(vec![rows, out_f], DType::F32))
        };
        g.push(Op::Linear {
            x: src,
            weight: wi,
            dst,
            m: rows as u32,
            in_f: in_f as u32,
            out_f: out_f as u32,
            w_off: 0,
        });
        src = dst;
        let bytes = weight(in_f * out_f, 41 + l as u32);
        weights.push((wi, upload(&bytes, BufferUsage::Weights)));
    }
    let plan = be.compile(&g).expect("compile");
    let xb = upload(bytemuck::cast_slice(&x), BufferUsage::Activations);
    let yb: Box<dyn Buffer> = be.alloc(rows * dim * 4, BufferUsage::Readback).unwrap();
    let mut binds = Bindings::new();
    binds.bind(xi, xb.as_ref());
    for (wi, wb) in &weights {
        binds.bind(*wi, wb.as_ref());
    }
    binds.bind(y, yb.as_ref());
    let (splits0, cuts0) = (be.work_splits(), be.segment_cuts().work);
    let ranges0 = be.work_split_ranges();
    be.execute(plan.as_ref(), &binds).expect("execute");
    let mut out = vec![0f32; rows * dim];
    be.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut out))
        .unwrap();
    assert!(out.iter().all(|v| v.is_finite()), "output is not finite");
    Run {
        y: out,
        splits: be.work_splits() - splits0,
        ranges: be.work_split_ranges() - ranges0,
        work_cuts: be.segment_cuts().work - cuts0,
        dispatches: be.last_forward_dispatches(),
        kernels: be.split_kernels(),
    }
}

const OFF: (&str, &str) = ("INFR_TUNE", "submit.budget_ms=0");
const TINY: (&str, &str) = ("INFR_TUNE", "submit.budget_ms=1,submit.prior_gflops=1");

/// One case: the bound off and the bound tiny, same graph, same masks; bit-identical, and cut.
fn case(label: &str, masks: &[(&str, &str)], dt: DType, rows: usize, dim: usize) {
    let with = |tune| {
        let mut v = masks.to_vec();
        v.push(tune);
        backend(&v)
    };
    let (Some(off), Some(tiny)) = (with(OFF), with(TINY)) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let want = run(&off, dt, rows, dim);
    assert_eq!(
        want.splits, 0,
        "{label}: the bound off still cut a dispatch"
    );
    let got = run(&tiny, dt, rows, dim);
    println!(
        "{label}: {} dispatch(es) cut, {} submit(s) opened by cuts, kernels {:?}",
        got.splits, got.work_cuts, got.kernels
    );
    assert!(got.splits > 0, "{label}: the tiny budget cut nothing");
    assert!(
        got.work_cuts > 0,
        "{label}: the ranges were not given submits of their own"
    );
    // Every range is a dispatch, whichever of the recorder's command buffers it went into: the
    // forward reports the uncut forward's dispatches plus the ranges, less the dispatches the
    // ranges replace. These forwards end in the recorder that made the cuts, so a count taken
    // from its open buffer alone comes out short.
    assert!(want.dispatches > 0, "{label}: no dispatch count reported");
    assert_eq!(
        got.dispatches,
        want.dispatches + got.ranges - got.splits,
        "{label}: the forward reported {} dispatches for {} uncut, {} cut into {} ranges",
        got.dispatches,
        want.dispatches,
        got.splits,
        got.ranges
    );
    let mismatch = got
        .y
        .iter()
        .zip(&want.y)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(
        mismatch, None,
        "{label}: cut output differs at element {mismatch:?}"
    );
}

/// Matrix kernels and int8 kernels off: the per-row GEMV over a grid of more than 65535
/// workgroups, which `dispatch_wide` lays out in two dimensions (cut by rows of the grid).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn gpu_seam_work_split_gemv_two_dimensional_grid() {
    let masks = [("INFR_NO_COOPMAT", "1"), ("INFR_NO_I8DOT", "1")];
    case("gemv 2-D", &masks, DType::Q4K, 64, 2048);
}

/// The same GEMV on a grid small enough for one dimension (cut along x). Rows past the
/// multi-row GEMV's eight, so the per-row GEMV runs.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn gpu_seam_work_split_gemv_one_row_grid() {
    let masks = [("INFR_NO_COOPMAT", "1"), ("INFR_NO_I8DOT", "1")];
    case("gemv 1-D", &masks, DType::Q4K, 16, 2048);
}

/// Matrix kernels off: the int8 GEMMs (the dp4a K-quant GEMM for Q4_K, the dense mmq GEMM for
/// Q8_0) where the device has an int8 dot.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn gpu_seam_work_split_int8_gemm() {
    let masks = [("INFR_NO_COOPMAT", "1")];
    case("int8 q4_k", &masks, DType::Q4K, 192, 2048);
    case("int8 q8_0", &masks, DType::Q8_0, 192, 2048);
}

/// The splitter's kill switch (`device.submit_dispatches = 0`) is one submit per forward: under
/// it the work bound neither cuts a dispatch nor opens a submit, however small the budget, and
/// the output is the unsplit one.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn gpu_seam_work_split_kill_switch_keeps_one_submit() {
    let masks = [("INFR_NO_COOPMAT", "1"), ("INFR_NO_I8DOT", "1")];
    let with = |extra: &[(&str, &str)]| {
        let mut v = masks.to_vec();
        v.extend_from_slice(extra);
        backend(&v)
    };
    let (Some(off), Some(killed)) = (with(&[OFF]), with(&[TINY, ("INFR_SUBMIT_DISPATCHES", "0")]))
    else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let want = run(&off, DType::Q4K, 64, 2048);
    let got = run(&killed, DType::Q4K, 64, 2048);
    assert_eq!(
        (got.splits, got.work_cuts),
        (0, 0),
        "the kill switch did not keep the forward in one submit"
    );
    assert_eq!(killed.segment_cuts(), Default::default());
    assert!(got
        .y
        .iter()
        .zip(&want.y)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

/// Issue 94's class, simulated: wave64 only, no matrix kernels, no int8 dot, and on a discrete
/// card the work estimate that class's own tuning gives the RX 580 (`submit.prior_gflops` 288,
/// its 36 compute units), under the DEFAULT budget. A prefill-sized forward over SmolLM3-3B's
/// feed-forward shapes then has its heaviest GEMVs cut into ranges and its submits closed by
/// work, from the first forward, and its output is bit-identical to the same forward in one
/// submit (the kill switch).
///
/// An integrated GPU is slow for real: its own default tuning is used, and the forward is
/// smaller, because the one-submit reference of the large one is exactly what its watchdog
/// resets (four dispatches of about 3 s each on the Ryzen integrated GPU).
///
/// Wave64 is forced only where the device can pin 64; one that cannot is run as it is when that
/// is slow already, and skipped otherwise (`slow_class_support`).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn gpu_seam_work_split_slow_class_forward_under_the_default_budget() {
    const MASKS: [(&str, &str); 2] = [("INFR_NO_COOPMAT", "1"), ("INFR_NO_I8DOT", "1")];
    let Some(facts) = backend(&MASKS).map(|be| be.tuning_report().device.clone()) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let Some(by) = slow_class_support::slow_by(&facts) else {
        return;
    };
    let with = |extra: &[(&str, &str)]| {
        let mut v = MASKS.to_vec();
        if by == SlowBy::Wave64 {
            v.push(("INFR_FORCE_WAVE64", "1"));
        }
        v.extend_from_slice(extra);
        backend(&v)
    };
    let (rows, tuning): (usize, &[(&str, &str)]) = if facts.integrated {
        (64, &[])
    } else {
        (1024, &[("INFR_TUNE", "submit.prior_gflops=288")])
    };
    let (Some(whole), Some(bounded)) = (with(&[("INFR_SUBMIT_DISPATCHES", "0")]), with(tuning))
    else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let widths = [2048, 22016, 2048, 22016, 2048];
    let want = run_chain(&whole, DType::Q4K, rows, &widths);
    assert_eq!((want.splits, want.work_cuts), (0, 0));
    let t = std::time::Instant::now();
    let got = run_chain(&bounded, DType::Q4K, rows, &widths);
    println!(
        "slow by {by:?}, {rows} rows: first forward {:.0} ms, {} dispatch(es) cut, {} submit(s) \
         opened by the work bound, kernels {:?}, measured GFLOP/s {:?}",
        t.elapsed().as_secs_f64() * 1e3,
        got.splits,
        got.work_cuts,
        got.kernels,
        bounded.work_measured_gflops()
    );
    assert!(got.splits > 0, "no heavy dispatch was cut into ranges");
    assert!(
        got.work_cuts > got.splits,
        "the work bound opened no submit beyond one per cut dispatch"
    );
    let mismatch = got
        .y
        .iter()
        .zip(&want.y)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(mismatch, None, "bounded output differs at {mismatch:?}");
}

/// The device's own prefill tier, unmasked: the matrix GEMMs on the RX 7900 XTX, the int8 GEMMs
/// on the integrated GPU.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn gpu_seam_work_split_default_tier() {
    case("default q4_k", &[], DType::Q4K, 192, 2048);
    case("default q8_0", &[], DType::Q8_0, 192, 2048);
    case("default f16", &[], DType::F16, 192, 2048);
}
