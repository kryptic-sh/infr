//! A STREAMED dense prefill `Op::Linear` (its weight a registered `DensePagerSession` block, read
//! from the pool arena by 64-bit address) takes the same tiled GEMM the resident Linear does, not
//! the per-row streamed GEMV that re-reads the whole weight once per row.
//!
//! Each case drives `Op::Linear` through `Backend::execute` twice on one backend — once with the
//! weight streamed, once uploaded resident — and checks three things:
//!
//!  • the ROUTE: right after the streamed run, `built_kernel_names()` holds the tiled GEMM's kernel
//!    and not the streamed GEMV's. The two agree numerically to within the int8-activation error,
//!    so the output alone cannot say which one ran;
//!  • streamed == resident, bit for bit (the same arithmetic over the same bytes, relocated);
//!  • the result against the per-row streamed GEMV (`Recorder::linear_native_at` on the same
//!    bytes — the route this replaces) and the host oracle, within the mmq parity tolerance.
//!
//! A third test covers the fallback: a weight the tiled GEMM cannot read (one that breaches the u32
//! addressing cap, forced here by lowering `kernels.vulkan.bda_chunk_elems`) must keep the chunked
//! per-row GEMV and run, not fail.
//!
//! Which device each test needs (select it with `INFR_DEV`; on any other device a test prints
//! `skip:` and passes):
//!
//!  • `streamed_prefill_takes_the_non_coopmat_gemm`: a device WITHOUT f16 coopmat (an iGPU, Arc).
//!    The bf16 fma case runs on any such device; the Q8_0 and Q4_K mmq cases need int8 dot too.
//!  • `streamed_narrow_q4k_takes_mmq_on_coopmat`: a device WITH f16 coopmat AND int8 dot.
//!  • `streamed_breaching_weight_keeps_the_gemv`: any device except one with f16 coopmat and no
//!    int8 dot (there no streamed shape routes to a non-warp tiled GEMM).
//!
//! All assume the default kernel config otherwise (no `INFR_NO_MMQ*` / `INFR_NO_GEMM_WARP`).
//!
//! Run: `cargo test -p infr-vulkan --release --test streamed_nc_gemm -- --ignored --nocapture`
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::config::{Config, ConfigLayer};
use infr_core::{DType, Graph, Op, TensorDesc};
use infr_vulkan::pager::{
    buffer_identity, DenseBytes, DensePagerLayout, DensePoolSpec, DenseSource,
};
use infr_vulkan::VulkanBackend;
use std::sync::Arc;

/// Rows of the prefill chunk: past the multi-row GEMV bands, and not a whole 64-row tile, so the
/// padded-dst copy is exercised too.
const M: usize = 40;
const IN_F: usize = 256;

struct Case {
    dt: DType,
    out_f: usize,
    /// Weight rows ABOVE the `out_f` the op reads, skipped by `w_off` — a fused-QKV-style slice
    /// that checks the within-slot element offset reaches the tiled kernel.
    skip_rows: usize,
    /// The kernel the streamed route must build.
    tiled: &'static str,
    /// The streamed GEMV kernel it must NOT build.
    gemv: &'static str,
}

fn weight_bytes(dt: DType, rows: usize) -> Vec<u8> {
    let n = rows * IN_F;
    if dt == DType::Bf16 {
        infr_testkit::gen_f32(n, 7)
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect()
    } else {
        infr_testkit::synth_weight(dt, n, 7)
    }
}

/// `y[M, out_f] = x · W[w_off..]ᵀ` through the graph, the weight bound to `w`.
fn run(be: &VulkanBackend, c: &Case, w: &dyn Buffer, x: &[f32]) -> Vec<f32> {
    let rows = c.skip_rows + c.out_f;
    let mut g = Graph::new();
    let xi = g.input(TensorDesc::new(vec![M, IN_F], DType::F32));
    let wi = g.weight(TensorDesc::new(vec![rows, IN_F], c.dt));
    let yi = g.output(TensorDesc::new(vec![M, c.out_f], DType::F32));
    g.push(Op::Linear {
        x: xi,
        weight: wi,
        dst: yi,
        m: M as u32,
        in_f: IN_F as u32,
        out_f: c.out_f as u32,
        w_off: (c.skip_rows * IN_F) as u32,
    });
    let plan = be.compile(&g).expect("compile");
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let yb = be.alloc(M * c.out_f * 4, BufferUsage::Readback).unwrap();
    let mut binds = Bindings::new();
    binds.bind(xi, xb.as_ref());
    binds.bind(wi, w);
    binds.bind(yi, yb.as_ref());
    be.execute(plan.as_ref(), &binds).expect("execute");
    let mut y = vec![0f32; M * c.out_f];
    be.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut y))
        .unwrap();
    y
}

/// Worst absolute difference as a fraction of `want`'s RMS.
fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    let worst = got.iter().zip(want).fold(0f64, |a, (g, w)| {
        infr_testkit::nan_max64(a, (*g as f64 - *w as f64).abs())
    });
    worst / rms
}

/// Install a one-block dense streaming session on `be` holding `wb`, and return the placeholder
/// buffer that stands for it in a graph binding.
fn stream_weight(be: &VulkanBackend, dt: DType, wb: &[u8]) -> Box<dyn Buffer> {
    // One pool, one slot, one block: the slot stride is the block's bytes rounded to whole quant
    // blocks AND u32 words, which is what the seam's planner computes.
    let (_, blk_bytes) = infr_gguf::block_layout(dt);
    let stride = wb.len().next_multiple_of(blk_bytes * 4);
    be.init_dense_pager(DensePagerLayout {
        pools: vec![DensePoolSpec {
            slot_bytes: stride,
            n_slots: 1,
            n_blocks: 1,
            host: None,
        }],
        ring_bytes: 0,
    })
    .unwrap();
    let placeholder = be.alloc_uninit(4, BufferUsage::Weights).unwrap();
    be.register_dense_stream(
        0,
        buffer_identity(placeholder.as_ref()),
        DenseSource {
            bytes: DenseBytes::Mmap(vec![
                Arc::new(wb.to_vec()) as Arc<dyn AsRef<[u8]> + Send + Sync>
            ]),
            block_id: 0,
        },
    )
    .unwrap();
    placeholder
}

/// The per-row streamed GEMV (`Recorder::linear_native_at`) over `wb` in a plain arena — the route
/// the tiled GEMM replaces, and the one a streamed Linear falls back to.
fn gemv_at(be: &VulkanBackend, c: &Case, wb: &[u8], x: &[f32]) -> Vec<f32> {
    let (arena, addr) = be.alloc_arena_bda(wb.len()).unwrap();
    be.upload(arena.as_ref(), wb).unwrap();
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let yb = be.alloc(M * c.out_f * 4, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.linear_native_at(
        c.dt,
        addr,
        c.skip_rows * IN_F,
        xb.as_ref(),
        yb.as_ref(),
        M,
        IN_F,
        c.out_f,
    );
    rec.finish().unwrap();
    let mut gemv = vec![0f32; M * c.out_f];
    be.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut gemv))
        .unwrap();
    gemv
}

fn label(c: &Case) -> String {
    format!(
        "{:?} m={M} in_f={IN_F} out_f={} w_off_rows={}",
        c.dt, c.out_f, c.skip_rows
    )
}

fn check(c: &Case) {
    let be = VulkanBackend::new().expect("Vulkan device");
    let rows = c.skip_rows + c.out_f;
    let wb = weight_bytes(c.dt, rows);
    let x = infr_testkit::gen_f32(M * IN_F, 11);
    let label = label(c);
    let placeholder = stream_weight(&be, c.dt, &wb);

    let streamed = run(&be, c, placeholder.as_ref(), &x);
    let built = be.built_kernel_names();
    println!("{label}: streamed run built {built:?}");
    assert!(
        built.contains(&c.tiled),
        "{label}: the streamed prefill did not take the tiled GEMM `{}`",
        c.tiled
    );
    assert!(
        !built.contains(&c.gemv),
        "{label}: the streamed prefill ran the per-row GEMV `{}`",
        c.gemv
    );

    let resident_w = be.alloc(wb.len(), BufferUsage::Weights).unwrap();
    be.upload(resident_w.as_ref(), &wb).unwrap();
    let resident = run(&be, c, resident_w.as_ref(), &x);
    assert!(
        streamed.iter().all(|v| v.is_finite()),
        "{label}: non-finite"
    );
    let mismatch = streamed
        .iter()
        .zip(&resident)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(
        mismatch, None,
        "{label}: streamed differs from resident at element {mismatch:?}"
    );

    let gemv = gemv_at(&be, c, &wb, &x);

    let w_all = infr_testkit::dequant_oracle(c.dt, &wb);
    let oracle = infr_testkit::ref_linear(&x, &w_all[c.skip_rows * IN_F..], M, IN_F, c.out_f);
    let (vs_gemv, vs_oracle) = (rel_err(&streamed, &gemv), rel_err(&streamed, &oracle));
    println!("{label}: {vs_gemv:.2e} of the RMS from the GEMV, {vs_oracle:.2e} from the oracle");
    // mmq quantizes the activations to int8: ~1e-2 of the RMS (`mmq_scratch_rows.rs`). The fma
    // GEMM is f32 throughout, so only accumulation order separates it from the GEMV. A wrong row,
    // block or slice offset is order 1 either way.
    let tol = if c.dt == DType::Bf16 { 1e-3 } else { 5e-2 };
    assert!(
        vs_gemv < tol,
        "{label}: streamed GEMM diverges from the GEMV"
    );
    assert!(
        vs_oracle < tol,
        "{label}: streamed GEMM diverges from the host oracle"
    );
}

#[test]
#[ignore = "requires a Vulkan GPU without f16 cooperative matrix"]
fn streamed_prefill_takes_the_non_coopmat_gemm() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let caps = be.capabilities();
    if caps.f16_coopmat() {
        eprintln!("skip: the non-coopmat GEMM tier runs only without f16 coopmat");
        return;
    }
    drop(be);
    let mut cases = vec![Case {
        dt: DType::Bf16,
        out_f: 128,
        skip_rows: 64,
        tiled: "native_gemm_fma_bf16",
        gemv: "native_bf16",
    }];
    if caps.i8_dot {
        cases.extend([
            Case {
                dt: DType::Q8_0,
                out_f: 128,
                skip_rows: 128,
                tiled: "native_gemm_mmq_q8_0",
                gemv: "native_q8_0",
            },
            // Min-carrying, so the `sact` binding is in play.
            Case {
                dt: DType::Q4K,
                out_f: 192,
                skip_rows: 0,
                tiled: "native_gemm_mmq_q4k_streamed",
                gemv: "native_q4k",
            },
        ]);
    } else {
        eprintln!("no int8 dot: only the fma arm is checked");
    }
    for c in &cases {
        check(c);
    }
}

/// The coopmat tier's one dp4a pick: Q4_K with no warp tile for its `out_f` (a multiple of 64 and
/// not of 128). The streamed route takes the arena-addressed build of that same mmq shader.
#[test]
#[ignore = "requires a Vulkan GPU with f16 cooperative matrix and int8 dot"]
fn streamed_narrow_q4k_takes_mmq_on_coopmat() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let caps = be.capabilities();
    if !caps.f16_coopmat() || !caps.i8_dot {
        eprintln!("skip: needs f16 coopmat and int8 dot");
        return;
    }
    drop(be);
    check(&Case {
        dt: DType::Q4K,
        out_f: 192,
        skip_rows: 64,
        tiled: "native_gemm_mmq_q4k_streamed",
        gemv: "native_q4k",
    });
}

/// A weight that breaches the u32 addressing cap cannot take a tiled GEMM (it cannot be output-row
/// chunked), but the per-row GEMV can chunk it, so a streamed Linear whose route is a non-warp
/// tiled GEMM must fall back to that GEMV and run — as it did before the tiled route existed (a
/// big-vocab lm_head at m > 16: all-position logits, speculative verify). The cap is forced down
/// to this small weight's size through the config value `INFR_BDA_CHUNK_ELEMS` also sets.
#[test]
#[ignore = "requires a Vulkan GPU: without f16 coopmat, or with f16 coopmat and int8 dot"]
fn streamed_breaching_weight_keeps_the_gemv() {
    let Ok(probe) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let caps = probe.capabilities();
    drop(probe);
    // A shape whose route on this device is the tiled GEMM the tests above observe.
    let c = match (caps.f16_coopmat(), caps.i8_dot) {
        (false, true) => Case {
            dt: DType::Q8_0,
            out_f: 128,
            skip_rows: 0,
            tiled: "native_gemm_mmq_q8_0",
            gemv: "native_q8_0",
        },
        (false, false) => Case {
            dt: DType::Bf16,
            out_f: 128,
            skip_rows: 0,
            tiled: "native_gemm_fma_bf16",
            gemv: "native_bf16",
        },
        (true, true) => Case {
            dt: DType::Q4K,
            out_f: 192,
            skip_rows: 0,
            tiled: "native_gemm_mmq_q4k_streamed",
            gemv: "native_q4k",
        },
        (true, false) => {
            eprintln!("skip: with coopmat and no int8 dot no streamed shape takes a non-warp GEMM");
            return;
        }
    };
    let mut cfg = Config::default();
    let env = ConfigLayer::env().expect("test environment config");
    cfg.device.dev = Config::load_from_layers(&[env]).device.dev;
    // The whole weight is exactly the cap, so it breaches; a 64-row chunk of it does not.
    cfg.kernels.vulkan.bda_chunk_elems = Some((c.out_f * IN_F) as u64);
    let be = VulkanBackend::new_with(Arc::new(cfg)).expect("Vulkan device");
    let label = label(&c);
    let wb = weight_bytes(c.dt, c.out_f);
    let x = infr_testkit::gen_f32(M * IN_F, 11);
    let placeholder = stream_weight(&be, c.dt, &wb);

    let streamed = run(&be, &c, placeholder.as_ref(), &x);
    let built = be.built_kernel_names();
    println!("{label}: breaching streamed run built {built:?}");
    assert!(
        built.contains(&c.gemv) && !built.contains(&c.tiled),
        "{label}: a breaching streamed weight must take the GEMV `{}`, not `{}`",
        c.gemv,
        c.tiled
    );
    // The reference runs on the same backend, so it chunks under the same lowered cap.
    let gemv = gemv_at(&be, &c, &wb, &x);
    assert!(
        streamed.iter().all(|v| v.is_finite()),
        "{label}: non-finite"
    );
    let mismatch = streamed
        .iter()
        .zip(&gemv)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(
        mismatch, None,
        "{label}: streamed fallback differs from the chunked GEMV at element {mismatch:?}"
    );
    let w_all = infr_testkit::dequant_oracle(c.dt, &wb);
    let oracle = infr_testkit::ref_linear(&x, &w_all, M, IN_F, c.out_f);
    let vs_oracle = rel_err(&streamed, &oracle);
    println!("{label}: {vs_oracle:.2e} of the RMS from the oracle");
    // f32 dequant and f32 accumulation: only summation order separates it from the oracle.
    assert!(
        vs_oracle < 1e-3,
        "{label}: the chunked GEMV diverges from the host oracle"
    );
}
