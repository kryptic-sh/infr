//! F16 and F32 dense weights stream: an `Op::Linear` whose weight is a registered
//! `DensePagerSession` block of a float dtype runs the resident float dispatch (`float_linear` in
//! the adapter) with the arena slot's address in place of the resident buffer's.
//!
//! Each case drives a graph of several `Op::Linear`s, one per weight, through `Backend::execute`
//! on one backend — once with the weights streamed through a pool with fewer slots than blocks,
//! once uploaded resident — and checks:
//!
//!  • the weights really streamed, from slots past the arena's start: the session's counters show
//!    misses, and with fewer slots than blocks some block must sit in a slot other than the first
//!    (a slot offset that was dropped would serve every block the first slot's bytes);
//!  • the ROUTE: the streamed run built the kernel the resident float dispatch picks for that shape
//!    on this device, named per case below;
//!  • streamed == resident, bit for bit (the same kernel over the same bytes, relocated);
//!  • both against the host oracle (`infr_testkit::ref_linear`, f64), within the kernel's own
//!    precision. Every block is filled differently, so a block served from a neighbour's slot, or a
//!    row slice read from the wrong offset, is order 1 off.
//!
//! The F32 cases also read a row slice past the block's first rows through `Op::Linear::w_off`
//! (the shifted-address path `float_linear` takes for f32); F16 refuses a `w_off`, as it does
//! resident.
//!
//! Gap, stated rather than tested: no case places a slot past 4 GiB of the arena. The slot base
//! is a 64-bit byte address computed on the host (`DensePagerSession::stage`), and the float
//! kernels only ever add their in-weight element index to it, so the slot's position never enters
//! u32 arithmetic — but proving that on hardware needs an arena larger than 4 GiB.
//!
//! Runs on any Vulkan device (select it with `INFR_DEV`); the route each case expects follows the
//! device's capabilities (f16 coopmat, shaderFloat16), and assumes the default kernel config.
//!
//! Run: `cargo test -p infr-vulkan --release --test dense_float_stream_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::{DType, Graph, Op, TensorDesc};
use infr_vulkan::pager::{
    buffer_identity, DenseBytes, DensePagerLayout, DensePoolSpec, DenseSource,
};
use infr_vulkan::VulkanBackend;
use std::sync::Arc;

/// Weights in the sweep: more than [`SLOTS`], so blocks share and recycle slots.
const BLOCKS: usize = 3;
const SLOTS: usize = 2;
/// Graph executions: the second one stages every block again into recycled slots.
const PASSES: usize = 2;

struct Case {
    dt: DType,
    m: usize,
    in_f: usize,
    out_f: usize,
    /// Weight rows ABOVE the `out_f` the op reads, skipped by `w_off` (f32 only).
    skip_rows: usize,
}

impl Case {
    fn label(&self) -> String {
        format!(
            "{:?} m={} in_f={} out_f={} w_off_rows={}",
            self.dt, self.m, self.in_f, self.out_f, self.skip_rows
        )
    }

    fn rows(&self) -> usize {
        self.skip_rows + self.out_f
    }

    /// The kernel the resident float dispatch picks for this shape on `be` — what the streamed run
    /// must build too.
    fn kernel(&self, be: &VulkanBackend) -> &'static str {
        let caps = be.capabilities();
        // `float_linear`'s GEMM gate.
        let gemm = self.m > 1 && self.out_f.is_multiple_of(64) && self.in_f.is_multiple_of(32);
        match self.dt {
            DType::F16 if gemm && caps.f16_coopmat() => {
                // The split-K warptile where the planner splits (and its 64-wide k stage fits),
                // else `gemm_proj`. Assumes the default `kernels.vulkan.gemm_warp`.
                if be.split_k_splits(self.m, self.in_f, self.out_f) > 1
                    && self.in_f.is_multiple_of(64)
                {
                    "native_gemm_warp_f16_sk"
                } else {
                    "gemm_proj"
                }
            }
            DType::F16 | DType::F32 if gemm => {
                if self.dt == DType::F16 {
                    "native_gemm_fma_f16"
                } else if caps.f16_coopmat() {
                    // A coopmat device has no f32 GEMM: the row-tiled f32 GEMV.
                    f32_gemv_kernel(be, self.m, self.in_f)
                } else {
                    "native_gemm_fma_f32"
                }
            }
            DType::F16 if caps.f16 => "linear_f16",
            DType::F16 => "linear_f16_noext",
            DType::F32 => f32_gemv_kernel(be, self.m, self.in_f),
            other => panic!("not a float case: {other:?}"),
        }
    }
}

/// The `linear_f32r` build the f32 GEMV dispatches for `rows` x `in_f` on `be`.
fn f32_gemv_kernel(be: &VulkanBackend, rows: usize, in_f: usize) -> &'static str {
    let rec = be.recorder().expect("recorder");
    let name = rec.linear_f32_kernel(rows, in_f);
    rec.finish().expect("finish the empty recording");
    name
}

/// Block `b`'s weight bytes and the values they hold.
fn weight(c: &Case, b: usize) -> (Vec<u8>, Vec<f32>) {
    let vals = infr_testkit::gen_f32(c.rows() * c.in_f, 7 + 5 * b);
    let bytes: Vec<u8> = match c.dt {
        DType::F16 => vals
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect(),
        DType::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        other => panic!("not a float case: {other:?}"),
    };
    // What the GPU reads back from those bytes: f16 rounds, f32 is exact.
    let read = infr_testkit::dequant_oracle(c.dt, &bytes);
    (bytes, read)
}

/// One graph, one `Op::Linear` per weight in `ws`, all over the same `x`. Returns each op's output.
fn run(be: &VulkanBackend, c: &Case, ws: &[&dyn Buffer], x: &[f32]) -> Vec<Vec<f32>> {
    let mut g = Graph::new();
    let xi = g.input(TensorDesc::new(vec![c.m, c.in_f], DType::F32));
    let mut ids = Vec::new();
    for _ in ws {
        let wi = g.weight(TensorDesc::new(vec![c.rows(), c.in_f], c.dt));
        let yi = g.output(TensorDesc::new(vec![c.m, c.out_f], DType::F32));
        g.push(Op::Linear {
            x: xi,
            weight: wi,
            dst: yi,
            m: c.m as u32,
            in_f: c.in_f as u32,
            out_f: c.out_f as u32,
            w_off: (c.skip_rows * c.in_f) as u32,
        });
        ids.push((wi, yi));
    }
    let plan = be.compile(&g).expect("compile");
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let ybs: Vec<_> = ws
        .iter()
        .map(|_| be.alloc(c.m * c.out_f * 4, BufferUsage::Readback).unwrap())
        .collect();
    let mut binds = Bindings::new();
    binds.bind(xi, xb.as_ref());
    for ((wi, yi), (w, yb)) in ids.iter().zip(ws.iter().zip(&ybs)) {
        binds.bind(*wi, *w);
        binds.bind(*yi, yb.as_ref());
    }
    be.execute(plan.as_ref(), &binds).expect("execute");
    ybs.iter()
        .map(|yb| {
            let mut y = vec![0f32; c.m * c.out_f];
            be.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut y))
                .unwrap();
            y
        })
        .collect()
}

/// Install a dense streaming session on `be` holding `blocks` in one pool of [`SLOTS`] slots, and
/// return the placeholders that stand for them in a graph binding.
fn stream_weights(be: &VulkanBackend, dt: DType, blocks: &[Vec<u8>]) -> Vec<Box<dyn Buffer>> {
    // The seam planner's stride: whole blocks, at the dtype's slot alignment.
    let stride = infr_vulkan::linear::dense_stream_slot_stride(dt, blocks[0].len());
    be.init_dense_pager(DensePagerLayout {
        pools: vec![DensePoolSpec {
            dtype: dt,
            slot_bytes: stride,
            n_slots: SLOTS,
            n_blocks: blocks.len(),
            host: None,
        }],
        ring_bytes: 0,
    })
    .unwrap();
    blocks
        .iter()
        .enumerate()
        .map(|(b, wb)| {
            let placeholder = be.alloc_uninit(4, BufferUsage::Weights).unwrap();
            be.register_dense_stream(
                0,
                buffer_identity(placeholder.as_ref()),
                DenseSource {
                    bytes: DenseBytes::Mmap(vec![
                        Arc::new(wb.clone()) as Arc<dyn AsRef<[u8]> + Send + Sync>
                    ]),
                    block_id: b as u32,
                },
            )
            .unwrap();
            placeholder
        })
        .collect()
}

/// Worst absolute difference as a fraction of `want`'s RMS.
fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    let worst = got.iter().zip(want).fold(0f64, |a, (g, w)| {
        infr_testkit::nan_max64(a, (*g as f64 - *w as f64).abs())
    });
    worst / rms
}

fn check(c: &Case) {
    let be = VulkanBackend::new().expect("Vulkan device");
    let label = c.label();
    let (blocks, vals): (Vec<_>, Vec<_>) = (0..BLOCKS).map(|b| weight(c, b)).unzip();
    let x = infr_testkit::gen_f32(c.m * c.in_f, 11);
    let oracle: Vec<Vec<f32>> = vals
        .iter()
        .map(|w| infr_testkit::ref_linear(&x, &w[c.skip_rows * c.in_f..], c.m, c.in_f, c.out_f))
        .collect();
    for b in 1..BLOCKS {
        assert!(
            rel_err(&oracle[b], &oracle[0]) > 0.1,
            "{label}: blocks 0 and {b} give the same output — a mis-served slot would be invisible"
        );
    }

    let placeholders = stream_weights(&be, c.dt, &blocks);
    let ws: Vec<&dyn Buffer> = placeholders.iter().map(|p| p.as_ref()).collect();
    let mut streamed = Vec::new();
    for _ in 0..PASSES {
        streamed = run(&be, c, &ws, &x);
    }
    let want_kernel = c.kernel(&be);
    let built = be.built_kernel_names();
    println!("{label}: streamed run built {built:?}");
    assert!(
        built.contains(&want_kernel),
        "{label}: the streamed Linear did not take `{want_kernel}`"
    );
    let totals = be
        .dense_pager_totals()
        .expect("a dense streaming session is installed");
    println!("{label}: {totals:?}");
    assert_eq!(totals.groups, BLOCKS, "{label}: blocks registered");
    let stats = totals.vram;
    // The first pass misses every block; with fewer slots than blocks the second has to stage
    // some of them again, into recycled slots.
    assert!(
        stats.misses > BLOCKS as u64,
        "{label}: the weights did not stream through recycled slots ({stats:?})"
    );

    let resident_bufs: Vec<_> = blocks
        .iter()
        .map(|wb| {
            let w = be.alloc(wb.len(), BufferUsage::Weights).unwrap();
            be.upload(w.as_ref(), wb).unwrap();
            w
        })
        .collect();
    let ws: Vec<&dyn Buffer> = resident_bufs.iter().map(|w| w.as_ref()).collect();
    let resident = run(&be, c, &ws, &x);

    // The coopmat GEMMs round the activations to f16 for the matrix cores; the GEMVs, the fma
    // warptile and every f32 path accumulate f32 over the exact weight values.
    let half_activations = c.dt == DType::F16 && c.m > 1;
    let tol = if half_activations { 5e-3 } else { 1e-5 };
    for b in 0..BLOCKS {
        let (s, r) = (&streamed[b], &resident[b]);
        assert!(
            s.iter().all(|v| v.is_finite()),
            "{label}: block {b} non-finite"
        );
        let mismatch = s
            .iter()
            .zip(r)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert_eq!(
            mismatch, None,
            "{label}: block {b} streamed differs from resident at element {mismatch:?}"
        );
        let err = rel_err(s, &oracle[b]);
        println!("{label}: block {b} is {err:.2e} of the RMS from the oracle");
        assert!(
            err < tol,
            "{label}: block {b} diverges from the host oracle ({err:.2e})"
        );
    }
}

/// Decode: the f16 GEMV and the f32 GEMV, f32 reading a row slice through `w_off`.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn streamed_float_decode_matches_resident() {
    if VulkanBackend::new().is_err() {
        eprintln!("skip: no Vulkan device");
        return;
    }
    for c in [
        Case {
            dt: DType::F16,
            m: 1,
            in_f: 256,
            out_f: 96,
            skip_rows: 0,
        },
        Case {
            dt: DType::F32,
            m: 1,
            in_f: 256,
            out_f: 96,
            skip_rows: 32,
        },
    ] {
        check(&c);
    }
}

/// Prefill: on a coopmat device the f16 coopmat GEMM, the f16 split-K warptile (a deep k and a
/// narrow n) and the row-tiled f32 GEMV; without coopmat the fma warptile for both dtypes.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn streamed_float_prefill_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // The deep case below is the split-K one on a coopmat device; if the planner stopped splitting
    // it, the case would quietly cover `gemm_proj` twice.
    if be.capabilities().f16_coopmat() {
        assert!(
            be.split_k_splits(40, 1024, 128) > 1,
            "the deep case no longer splits"
        );
    }
    drop(be);
    for c in [
        Case {
            dt: DType::F16,
            m: 40,
            in_f: 256,
            out_f: 192,
            skip_rows: 0,
        },
        Case {
            dt: DType::F16,
            m: 40,
            in_f: 1024,
            out_f: 128,
            skip_rows: 0,
        },
        Case {
            dt: DType::F32,
            m: 40,
            in_f: 256,
            out_f: 128,
            skip_rows: 64,
        },
    ] {
        check(&c);
    }
}
