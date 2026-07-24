//! GPU-gated parity tests for the ROCm backend — the correctness gate for Part A of
//! `docs/rocm-plan.md`. Every test is `#[ignore]`d: they require a real ROCm device
//! (the RX 7900 XTX dev box). Run on the dev box with:
//!
//!   cargo test -p infr-rocm --features rocm -- --include-ignored
//!
//! What is validated:
//!   * `alloc` honours the calloc contract (returns ZEROED VRAM),
//!   * `upload`→`download` is byte-identical,
//!   * a naive `Op::Linear` (dequant→f16 GEMV) matches the CPU reference
//!     (`infr_gguf::dequant::dequant_block` + f32 matmul, i.e. the `infr-cpu`
//!     backend running the same one-op graph) for F16 and a k-quant (Q4_K).
//!
//! The single-op agnostic-`Graph` pattern mirrors `infr-llama/tests/seam_op_parity.rs`.

#![cfg(all(target_os = "linux", feature = "rocm"))]

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::graph::{Activation, Graph, MoeGating, Op};
use infr_core::tensor::TensorDesc;
use infr_core::DType;
use infr_rocm::RocmBackend;

/// Construct the ROCm backend on device 0, or `None` if no ROCm device is present
/// (keeps the ignored tests a no-op on a machine without the hardware).
fn rocm() -> Option<RocmBackend> {
    RocmBackend::new(0).ok()
}

fn f32d(n: usize) -> TensorDesc {
    TensorDesc::new(vec![n], DType::F32)
}

/// Deterministic small-magnitude pseudo-random f32 stream (same shape as the seam
/// op-parity generator — keeps values well inside f16 range).
fn gen(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
        .collect()
}

fn maxerr(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn maxabs(a: &[f32]) -> f32 {
    a.iter().map(|x| x.abs()).fold(0.0, f32::max)
}

// ── alloc / upload / download ────────────────────────────────────────────────

/// `alloc` must return zero-initialized VRAM (the calloc contract every backend obeys).
#[test]
#[ignore = "requires a ROCm GPU"]
fn alloc_returns_zeroed() {
    let Some(be) = rocm() else {
        return;
    };
    let bytes = 4096usize;
    let buf = be.alloc(bytes, BufferUsage::Activations).expect("alloc");
    // Poison the host buffer so an all-zero readback can only come from the device.
    let mut host = vec![0xABu8; bytes];
    be.download(buf.as_ref(), &mut host).expect("download");
    assert!(
        host.iter().all(|&b| b == 0),
        "alloc did not zero-initialize VRAM (calloc contract violated)"
    );
}

/// `upload` then `download` round-trips byte-for-byte.
#[test]
#[ignore = "requires a ROCm GPU"]
fn upload_download_roundtrip() {
    let Some(be) = rocm() else {
        return;
    };
    let data: Vec<u8> = (0..8192u32).map(|i| ((i * 31 + 7) & 0xFF) as u8).collect();
    let buf = be
        .alloc(data.len(), BufferUsage::Activations)
        .expect("alloc");
    be.upload(buf.as_ref(), &data).expect("upload");
    let mut back = vec![0u8; data.len()];
    be.download(buf.as_ref(), &mut back).expect("download");
    assert_eq!(data, back, "upload→download is not byte-identical");
}

// ── Linear (dequant→f16 GEMV) vs the CPU reference ───────────────────────────

/// Run a single-`Op::Linear` graph on `be`: `dst[m, out_f] = x[m, in_f] · w[out_f, in_f]ᵀ`,
/// with `w` uploaded as its raw native `w_dtype` bytes (dequantized on first touch by the
/// backend). Returns the downloaded f32 output.
fn run_linear(
    be: &dyn Backend,
    x: &[f32],
    w_bytes: &[u8],
    w_dtype: DType,
    m: usize,
    in_f: usize,
    out_f: usize,
) -> Vec<f32> {
    let mut g = Graph::new();
    let xid = g.input(f32d(m * in_f));
    let wid = g.weight(TensorDesc::new(vec![out_f * in_f], w_dtype));
    let dst = g.output(f32d(m * out_f));
    g.push(Op::Linear {
        x: xid,
        weight: wid,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let plan = be.compile(&g).expect("compile");
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let wb = be.alloc(w_bytes.len(), BufferUsage::Weights).expect("w");
    be.upload(wb.as_ref(), w_bytes).unwrap();
    let ob = be.alloc(m * out_f * 4, BufferUsage::Readback).expect("out");
    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(wid, wb.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; m * out_f];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// F16 weight: the CPU reference dequants f16→f32 exactly, ROCm reads f16 as-is, so parity
/// is near bit-exact.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_f16_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (m, in_f, out_f) = (3usize, 256usize, 8usize);
    let x = gen(m * in_f, 4);
    // f16 weight bytes (little-endian half per element).
    let wf32 = gen(out_f * in_f, 7);
    let w_bytes: Vec<u8> = wf32
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let c = run_linear(&cpu, &x, &w_bytes, DType::F16, m, in_f, out_f);
    let r = run_linear(&be, &x, &w_bytes, DType::F16, m, in_f, out_f);
    let e = maxerr(&c, &r);
    println!("Linear F16 max_err={e:e} max|ref|={:e}", maxabs(&c));
    assert!(e < 1e-3, "Linear F16 diverges from CPU reference: {e:e}");
}

/// Q4_K weight: exercises the host block-dequant path. The CPU reference decodes the same
/// bytes with `dequant_block` + f32 matmul; ROCm decodes to f16 then GEMVs, so the tolerance
/// absorbs the f16 weight rounding.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_q4k_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    // Q4_K super-block = 256 elems / 144 bytes. in_f must be a multiple of 256.
    let (m, in_f, out_f) = (2usize, 256usize, 4usize);
    let blocks = (out_f * in_f) / 256; // one block per output row here
                                       // Build valid Q4_K blocks: patterned bytes, but the two f16 scale slots (d, dmin) at the
                                       // block head overwritten with finite small values so codes span a sane range and never
                                       // decode to Inf/NaN (mirrors infr-gguf's `affine_single_pass_bit_identical_q4k`).
    let mut w_bytes = vec![0u8; blocks * 144];
    for (i, byte) in w_bytes.iter_mut().enumerate() {
        *byte = ((i * 37 + 11) & 0xFF) as u8;
    }
    for blk in 0..blocks {
        let base = blk * 144;
        w_bytes[base..base + 2].copy_from_slice(&half::f16::from_f32(0.375).to_le_bytes());
        w_bytes[base + 2..base + 4].copy_from_slice(&half::f16::from_f32(-0.125).to_le_bytes());
    }
    let x = gen(m * in_f, 5);
    let c = run_linear(&cpu, &x, &w_bytes, DType::Q4K, m, in_f, out_f);
    let r = run_linear(&be, &x, &w_bytes, DType::Q4K, m, in_f, out_f);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "Linear Q4_K max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        e / ref_mag < 2e-2,
        "Linear Q4_K diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
    // Guard against a silently-zero output masquerading as agreement.
    assert!(
        ref_mag > 1e-3,
        "Q4_K reference is all-zero — test is vacuous"
    );
}

// ── MoeFfn (router GEMV → gating → top-k → expert FFN → weighted sum) vs CPU ──

/// f16 little-endian bytes for an f32 slice (expert weight banks upload as raw f16).
fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

/// Run a single-`Op::MoeFfn` graph on `be` and return the downloaded f32 output `[rows, ne]`.
/// `router` is F32 `[n_expert, ne]`; the gate/up/down expert banks are F16 (gate/up are
/// `[n_expert, n_ff_exp, ne]`, down is `[n_expert, ne, n_ff_exp]`, row-major). `router_x` is
/// bound to the SAME handle as `x` (the qwen3moe convention).
#[allow(clippy::too_many_arguments)]
fn run_moe(
    be: &dyn Backend,
    x: &[f32],
    router_f32: &[f32],
    gate_f16: &[u8],
    up_f16: &[u8],
    down_f16: &[u8],
    rows: usize,
    ne: usize,
    n_expert: usize,
    n_used: usize,
    n_ff_exp: usize,
    gating: MoeGating,
    norm_w: bool,
) -> Vec<f32> {
    let mut g = Graph::new();
    let xid = g.input(f32d(rows * ne));
    let rid = g.weight(TensorDesc::new(vec![n_expert * ne], DType::F32));
    let gid = g.weight(TensorDesc::new(vec![n_expert * n_ff_exp * ne], DType::F16));
    let uid = g.weight(TensorDesc::new(vec![n_expert * n_ff_exp * ne], DType::F16));
    let did = g.weight(TensorDesc::new(vec![n_expert * ne * n_ff_exp], DType::F16));
    let dst = g.output(f32d(rows * ne));
    g.push(Op::MoeFfn {
        x: xid,
        router_x: xid,
        router: rid,
        gate_exps: gid,
        up_exps: uid,
        down_exps: did,
        down_scale: None,
        dst,
        ne: ne as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
        n_ff_exp: n_ff_exp as u32,
        scale: 1.0,
        act: Activation::Silu,
        gating,
        norm_w,
        weight_before: false,
        fused_gate_up: false,
        ep_band: None,
    });
    let plan = be.compile(&g).expect("compile");

    let up = |desc_bytes: &[u8], usage| {
        let b = be.alloc(desc_bytes.len(), usage).expect("alloc");
        be.upload(b.as_ref(), desc_bytes).unwrap();
        b
    };
    let xb = up(bytemuck::cast_slice(x), BufferUsage::Activations);
    let rb = up(bytemuck::cast_slice(router_f32), BufferUsage::Weights);
    let gb = up(gate_f16, BufferUsage::Weights);
    let ub = up(up_f16, BufferUsage::Weights);
    let db = up(down_f16, BufferUsage::Weights);
    let ob = be.alloc(rows * ne * 4, BufferUsage::Readback).expect("out");

    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(rid, rb.as_ref());
    b.bind(gid, gb.as_ref());
    b.bind(uid, ub.as_ref());
    b.bind(did, db.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");

    let mut o = vec![0f32; rows * ne];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// Small synthetic MoE (F32 router + F16 experts): the ROCm router GEMV → gating → top-k →
/// renorm → per-expert gated FFN → weighted-sum path must match the CPU reference. Exercises
/// the softmax+renorm (qwen3moe) path and the sigmoid+no-renorm gating path.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, ne, n_expert, n_used, n_ff_exp) = (2usize, 128usize, 4usize, 2usize, 64usize);

    let x = gen(rows * ne, 3);
    // Distinct salts per bank so router logits are well-separated (deterministic top-k).
    let router = gen(n_expert * ne, 9);
    let gate = f16_bytes(&gen(n_expert * n_ff_exp * ne, 11));
    let up = f16_bytes(&gen(n_expert * n_ff_exp * ne, 17));
    let down = f16_bytes(&gen(n_expert * ne * n_ff_exp, 23));

    for (gating, norm_w, label) in [
        (MoeGating::Softmax, true, "softmax+renorm"),
        (MoeGating::Sigmoid, false, "sigmoid+no-renorm"),
    ] {
        let c = run_moe(
            &cpu, &x, &router, &gate, &up, &down, rows, ne, n_expert, n_used, n_ff_exp, gating,
            norm_w,
        );
        let r = run_moe(
            &be, &x, &router, &gate, &up, &down, rows, ne, n_expert, n_used, n_ff_exp, gating,
            norm_w,
        );
        let e = maxerr(&c, &r);
        let ref_mag = maxabs(&c).max(1e-6);
        println!(
            "MoeFfn [{label}] max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
            e / ref_mag
        );
        // Guard against a silently-zero output masquerading as agreement (the pre-fix bug
        // produced garbage/zeros because the router weight was never applied).
        assert!(
            ref_mag > 1e-3,
            "MoeFfn [{label}] reference is all-zero — test is vacuous"
        );
        assert!(
            e / ref_mag < 2e-2,
            "MoeFfn [{label}] diverges from CPU reference: abs={e:e} rel={:e}",
            e / ref_mag
        );
    }
}
