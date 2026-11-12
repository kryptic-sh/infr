//! The dp4a (mmq) prefill GEMMs stage whole row tiles of the quantized activations, so a row count
//! that is not a multiple of the tile reads `qa`/`dact`/`sact` rows past `m`. Those rows only feed
//! output rows nobody reads, so the result cannot show it; GPU-assisted validation reports the
//! out-of-bounds descriptor reads when the adapter sizes that scratch at exactly `m` rows (see
//! `gpuav_support`). The reference comparison guards that the real rows stay right.
//!
//! The non-coopmat tier (`nc_mmq`) is the route that reaches the dense mmq GEMM, so the test needs
//! a device without the f16 cooperative-matrix tier and with packed int8 dot (select it with
//! `INFR_DEV`), and says when it skipped. Without the validation layer installed it verifies only
//! the numeric result, not the bound.
//!
//! Run: cargo test -p infr-vulkan --test mmq_scratch_rows -- --ignored --nocapture
mod gpuav_support;

use infr_core::backend::Backend;
use infr_core::DType;
use infr_core::{Graph, Op, TensorDesc};
use infr_vulkan::VulkanBackend;

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmq_gemm_reads_no_activation_rows_past_m() {
    let log = gpuav_support::enable("mmq-scratch");
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let caps = be.capabilities();
    if caps.f16_coopmat() || !caps.i8_dot {
        eprintln!("skip: the dense mmq GEMM is reached only without f16 coopmat and with int8 dot");
        return;
    }
    // m past the GEMV tiers and not a multiple of the 64-row tile.
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    let dt = DType::Q8_0;
    let (blk_elems, blk_bytes) = infr_gguf::block_layout(dt);
    // 0x00..=0x3F keeps every f16 scale finite and small.
    let wb: Vec<u8> = (0..out_f * in_f / blk_elems * blk_bytes)
        .map(|i| ((i.wrapping_mul(2654435761) >> 7) % 0x40) as u8)
        .collect();
    let x: Vec<f32> = (0..m * in_f)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.021)
        .collect();
    let want = infr_testkit::ref_linear(&x, &infr_testkit::dequant_oracle(dt, &wb), m, in_f, out_f);
    let mut g = Graph::new();
    let xi = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let wi = g.weight(TensorDesc::new(vec![out_f, in_f], dt));
    let yi = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x: xi,
        weight: wi,
        dst: yi,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let got = infr_testkit::run_graph(
        &be,
        &g,
        &[(xi, infr_testkit::f32_bytes(&x)), (wi, wb)],
        yi,
        m * out_f,
    );
    let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    let worst = got.iter().zip(&want).fold(0f64, |a, (g, w)| {
        infr_testkit::nan_max64(a, (*g as f64 - *w as f64).abs())
    });
    println!(
        "Q8_0 m={m} in_f={in_f} out_f={out_f}: {:.2e} of the output RMS",
        worst / rms
    );
    // int8 activations: ~1e-2 of the RMS; a wrong row or block is order 1.
    assert!(
        worst / rms < 5e-2,
        "mmq GEMM diverges from the host reference"
    );

    drop(be);
    gpuav_support::assert_clean(
        &log,
        "mmq_gemm_reads_no_activation_rows_past_m",
        "the out-of-bounds bound (only the numeric result was checked)",
    );
}
