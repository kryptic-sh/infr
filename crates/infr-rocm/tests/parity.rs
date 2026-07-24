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

// ── EmbedGather (gather + dequant embedding rows, ×scale) vs CPU ─────────────

/// Run a single-`Op::EmbedGather` graph on `be`: `dst[r, :] = dequant(table[ids[r], :]) * scale`.
/// `ids` is an I32 input (token ids); `table` uploads as its raw native `table_dtype` bytes.
/// Returns the downloaded f32 output `[rows, ne]`.
fn run_embed_gather(
    be: &dyn Backend,
    ids: &[i32],
    table_bytes: &[u8],
    table_dtype: DType,
    vocab: usize,
    ne: usize,
    scale: f32,
) -> Vec<f32> {
    let rows = ids.len();
    let mut g = Graph::new();
    let ids_id = g.input(TensorDesc::new(vec![rows], DType::I32));
    let tbl = g.weight(TensorDesc::new(vec![vocab * ne], table_dtype));
    let dst = g.output(f32d(rows * ne));
    g.push(Op::EmbedGather {
        ids: ids_id,
        table: tbl,
        dst,
        rows: rows as u32,
        ne: ne as u32,
        scale,
    });
    let plan = be.compile(&g).expect("compile");
    let ids_bytes: &[u8] = bytemuck::cast_slice(ids);
    let ib = be
        .alloc(ids_bytes.len(), BufferUsage::Activations)
        .expect("ids");
    be.upload(ib.as_ref(), ids_bytes).unwrap();
    let tb = be
        .alloc(table_bytes.len(), BufferUsage::Weights)
        .expect("table");
    be.upload(tb.as_ref(), table_bytes).unwrap();
    let ob = be.alloc(rows * ne * 4, BufferUsage::Readback).expect("out");
    let mut b = Bindings::new();
    b.bind(ids_id, ib.as_ref());
    b.bind(tbl, tb.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; rows * ne];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// EmbedGather with a NON-1.0 scale (Gemma's sqrt(n_embd)): the scale must be applied on-device.
/// The pre-fix bug dropped the scale entirely (the HIP kernel had no `scale` param), so a Gemma
/// model's token embeddings came out unscaled — this test would fail loudly against the CPU
/// reference (`v * scale`). Covers an F16 table and a Q4_K table.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_matches_cpu() {
    if rocm().is_none() {
        return;
    }
    let cpu = infr_cpu::CpuBackend::new();
    let ids = [0i32, 3, 5, 1, 5, 2];

    // ── F16 table ──
    // Fresh backend per case: `dequant_weight_or_cache` keys the dequantized-weight cache by the
    // table's raw device pointer, and a table buffer freed at the end of one case can have its VRAM
    // address recycled by the next case's table — a stale cache hit would then feed the wrong
    // dequantized rows. Real models never hit this (weights are long-lived), but back-to-back
    // single-op test cases do; a per-case backend gives each an empty cache.
    {
        let be = rocm().unwrap();
        let (vocab, ne) = (6usize, 8usize);
        let scale = (ne as f32).sqrt(); // non-1.0, mirrors Gemma's embed scaling
        let tf32 = gen(vocab * ne, 41);
        let t_bytes: Vec<u8> = tf32
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let c = run_embed_gather(&cpu, &ids, &t_bytes, DType::F16, vocab, ne, scale);
        let r = run_embed_gather(&be, &ids, &t_bytes, DType::F16, vocab, ne, scale);
        let e = maxerr(&c, &r);
        let ref_mag = maxabs(&c).max(1e-6);
        println!(
            "EmbedGather F16 scale={scale:e} max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
            e / ref_mag
        );
        assert!(
            ref_mag > 1e-3,
            "EmbedGather F16 reference is all-zero — test is vacuous"
        );
        assert!(
            e / ref_mag < 1e-3,
            "EmbedGather F16 diverges from CPU reference: abs={e:e} rel={:e}",
            e / ref_mag
        );
    }

    // ── Q4_K table (ne must be a multiple of 256 = one super-block per row) ──
    {
        let be = rocm().unwrap();
        let (vocab, ne) = (6usize, 256usize); // vocab > max(ids)=5
        let scale = (ne as f32).sqrt();
        let blocks = (vocab * ne) / 256; // one block per vocab row
        let mut t_bytes = vec![0u8; blocks * 144];
        for (i, byte) in t_bytes.iter_mut().enumerate() {
            *byte = ((i * 37 + 11) & 0xFF) as u8;
        }
        // Q4_K super-block = d(2) + dmin(2) + scales(12) + qs(128). Set the f16 d/dmin slots to
        // finite small values, and the 12 packed 6-bit sub-block scale/min bytes to a benign
        // constant. Random bytes in those scale nibbles hit adversarial corners where the two
        // independent Q4_K decoders (infr-cpu ref vs infr-gguf device dequant) diverge on a
        // handful of raw elements — a dot product (the linear test) averages that away, but a raw
        // per-element gather exposes it. A benign, in-range sub-scale keeps both decoders in lock-
        // step so the comparison isolates the embed gather + on-device SCALE, not quant corners.
        for blk in 0..blocks {
            let base = blk * 144;
            t_bytes[base..base + 2].copy_from_slice(&half::f16::from_f32(0.375).to_le_bytes());
            t_bytes[base + 2..base + 4].copy_from_slice(&half::f16::from_f32(-0.125).to_le_bytes());
            for b in t_bytes[base + 4..base + 16].iter_mut() {
                *b = 0x11;
            }
        }
        let c = run_embed_gather(&cpu, &ids, &t_bytes, DType::Q4K, vocab, ne, scale);
        let r = run_embed_gather(&be, &ids, &t_bytes, DType::Q4K, vocab, ne, scale);
        let e = maxerr(&c, &r);
        let ref_mag = maxabs(&c).max(1e-3);
        println!(
            "EmbedGather Q4_K scale={scale:e} max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
            e / ref_mag
        );
        assert!(
            ref_mag > 1e-3,
            "EmbedGather Q4_K reference is all-zero — test is vacuous"
        );
        assert!(
            e / ref_mag < 2e-2,
            "EmbedGather Q4_K diverges from CPU reference: abs={e:e} rel={:e}",
            e / ref_mag
        );
    }
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

// ── Rope (ggml NORM interleaved RoPE, packed + strided) vs CPU ────────────────

/// Run a single-`Op::Rope` graph on `be` and return the FULL output buffer (length = `x.len()`).
/// `x` is the raw input: packed `[rows, n_head, head_dim]` when `x_stride == 0`, else a wider
/// `[rows, x_stride]` row buffer whose per-row `n_head*head_dim` query slice packs at the row
/// start. `positions` is an I32 tensor. `dst != x`, so the backend copies the (possibly strided)
/// source and rotates in place — the rotated query lands at `row*x_stride + h*head_dim`.
#[allow(clippy::too_many_arguments)]
fn run_rope(
    be: &dyn Backend,
    x: &[f32],
    positions: &[i32],
    rows: usize,
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    x_stride: usize,
) -> Vec<f32> {
    let mut g = Graph::new();
    let xid = g.input(f32d(x.len()));
    let pid = g.input(TensorDesc::new(vec![positions.len()], DType::I32));
    let dst = g.output(f32d(x.len()));
    g.push(Op::Rope {
        x: xid,
        positions: pid,
        dst,
        rows: rows as u32,
        n_head: n_head as u32,
        head_dim: head_dim as u32,
        rope_dim: rope_dim as u32,
        theta,
        freq_factors: None,
        x_stride: x_stride as u32,
    });
    let plan = be.compile(&g).expect("compile");
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let pbytes: &[u8] = bytemuck::cast_slice(positions);
    let pb = be
        .alloc(pbytes.len(), BufferUsage::Activations)
        .expect("pos");
    be.upload(pb.as_ref(), pbytes).unwrap();
    let ob = be.alloc(x.len() * 4, BufferUsage::Readback).expect("out");
    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(pid, pb.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; x.len()];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// `Op::Rope` (the no-qk-norm llama-family INTERLEAVED rotation) must match the CPU reference for
/// BOTH a packed input and a NON-trivial `x_stride`. The pre-fix kernel had three defects any of
/// which this catches: (1) split-half (NEOX) pairing instead of interleaved (2p, 2p+1), (2) the
/// dropped `x_stride` — a strided view read the wrong elements — plus a `dst != x` copy that
/// grabbed a packed prefix regardless of stride, and (3) `freq *= freq_factors` (the wrong
/// direction). The strided case is the qwen35 q+g shape: the rotated query is a slice inside a
/// wider row buffer; a stride-blind kernel rotates the poison tail as extra heads and diverges.
#[test]
#[ignore = "requires a ROCm GPU"]
fn rope_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, n_head, head_dim, rope_dim) = (3usize, 2usize, 8usize, 8usize);
    let theta = 10000.0f32;
    let positions: Vec<i32> = vec![1, 7, 4]; // non-zero + distinct per row so RoPE actually rotates
    let hw = n_head * head_dim; // packed per-row width
    let np = rows * hw;
    let packed = gen(np, 6); // logical query, packed [rows, n_head, head_dim]

    // ── (a) packed input (x_stride = 0 / natural) ──
    let c = run_rope(
        &cpu, &packed, &positions, rows, n_head, head_dim, rope_dim, theta, 0,
    );
    let r = run_rope(
        &be, &packed, &positions, rows, n_head, head_dim, rope_dim, theta, 0,
    );
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-6);
    println!(
        "Rope packed max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "Rope packed reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 1e-3,
        "Rope packed diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );

    // ── (b) NON-trivial x_stride: query slice inside a wider row buffer (qwen35 q+g shape) ──
    // Each row is `stride` wide; the query packs at the row start, the tail is POISON the kernel
    // must never touch. The CPU reference is the SAME query packed (CPU Op::Rope is packed-only),
    // so parity holds on the logical query values regardless of the wider ROCm layout.
    let stride = hw * 2; // double-width row, like the interleaved q+g buffer
    let mut wide = vec![0f32; rows * stride];
    for row in 0..rows {
        for i in 0..hw {
            wide[row * stride + i] = packed[row * hw + i];
        }
        for j in hw..stride {
            wide[row * stride + j] = 1000.0 + (row * stride + j) as f32; // large, distinctive poison
        }
    }
    let rs = run_rope(
        &be, &wide, &positions, rows, n_head, head_dim, rope_dim, theta, stride,
    );
    // Extract the packed roped query out of each strided row.
    let mut rs_packed = vec![0f32; np];
    for row in 0..rows {
        for i in 0..hw {
            rs_packed[row * hw + i] = rs[row * stride + i];
        }
    }
    let e2 = maxerr(&c, &rs_packed);
    let ref_mag2 = maxabs(&c).max(1e-6);
    println!(
        "Rope strided(stride={stride}) max_err={e2:e} max|ref|={ref_mag2:e} rel={:e}",
        e2 / ref_mag2
    );
    assert!(
        ref_mag2 > 1e-3,
        "Rope strided reference is all-zero — test is vacuous"
    );
    assert!(
        e2 / ref_mag2 < 1e-3,
        "Rope strided diverges from CPU reference (x_stride dropped?): abs={e2:e} rel={:e}",
        e2 / ref_mag2
    );
    // The poison tail must survive untouched: a stride-correct kernel only rotates the query slice.
    for row in 0..rows {
        for j in hw..stride {
            let idx = row * stride + j;
            assert!(
                (rs[idx] - wide[idx]).abs() < 1e-6,
                "rope touched the strided-row tail at {idx} — kernel read/wrote outside the query slice"
            );
        }
    }
}

// ── QkNormRope (fused per-head RMSNorm + NEOX split-half RoPE, strided q+g) vs CPU ──

/// Run a single-`Op::QkNormRope` graph on `be` and return the downloaded PACKED f32 output
/// `[rows, n_head, head_dim]`. `x` is the raw input: packed `[rows, n_head, head_dim]` when
/// `x_stride == 0`, else a wider `[rows, x_stride]` row buffer whose per-head query slice packs at
/// `row*x_stride + h*(x_stride/n_head)` (the qwen35 interleaved q+g layout). `weight` is the F16
/// per-head RMSNorm weight `[head_dim]`; `positions` is an I32 tensor.
#[allow(clippy::too_many_arguments)]
fn run_qk_norm_rope(
    be: &dyn Backend,
    x: &[f32],
    weight_f16: &[u8],
    positions: &[i32],
    rows: usize,
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    eps: f32,
    x_stride: usize,
) -> Vec<f32> {
    let mut g = Graph::new();
    let xid = g.input(f32d(x.len()));
    let wid = g.weight(TensorDesc::new(vec![head_dim], DType::F16));
    let pid = g.input(TensorDesc::new(vec![positions.len()], DType::I32));
    let dst = g.output(f32d(rows * n_head * head_dim));
    g.push(Op::QkNormRope {
        x: xid,
        weight: wid,
        positions: pid,
        dst,
        rows: rows as u32,
        n_head: n_head as u32,
        head_dim: head_dim as u32,
        rope_dim: rope_dim as u32,
        theta,
        eps,
        freq_factors: None,
        x_stride: x_stride as u32,
    });
    let plan = be.compile(&g).expect("compile");
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let wb = be.alloc(weight_f16.len(), BufferUsage::Weights).expect("w");
    be.upload(wb.as_ref(), weight_f16).unwrap();
    let pbytes: &[u8] = bytemuck::cast_slice(positions);
    let pb = be
        .alloc(pbytes.len(), BufferUsage::Activations)
        .expect("pos");
    be.upload(pb.as_ref(), pbytes).unwrap();
    let ob = be
        .alloc(rows * n_head * head_dim * 4, BufferUsage::Readback)
        .expect("out");
    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(wid, wb.as_ref());
    b.bind(pid, pb.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; rows * n_head * head_dim];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// `Op::QkNormRope` (fused per-head RMSNorm + NEOX split-half RoPE) must match the CPU reference for
/// a MULTI-ROW (prefill) input with a NON-trivial `x_stride` — the qwen35 interleaved q+g layout
/// where each attention head is a strided slice of a wider `[q | gate]` row buffer. The pre-fix
/// kernel indexed the per-head base as `r*x_stride + h*head_dim` (packed head stride) instead of
/// `h*(x_stride/n_head)`, AND wrote the rotation in place into a packed-size buffer while indexing
/// it with the strided stride — an out-of-bounds read/write past the buffer on rows > 1 that MAFFs
/// on-device (qwen35 prefill op 67). It also divided the RoPE angle by the wrong `freq_factors`
/// direction. This test runs the SAME graph on `RocmBackend` and `infr_cpu::CpuBackend` and compares
/// the packed outputs; it fails loudly without the fix.
#[test]
#[ignore = "requires a ROCm GPU"]
fn qk_norm_rope_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, n_head, head_dim, rope_dim) = (4usize, 3usize, 8usize, 8usize);
    let theta = 10000.0f32;
    let eps = 1e-6f32;
    let positions: Vec<i32> = vec![0, 1, 5, 9]; // distinct per row so RoPE actually rotates
    let hw = n_head * head_dim; // packed per-row query width

    // Per-head RMSNorm weight [head_dim], F16 (non-trivial so the norm scale is observable).
    let wf32 = gen(head_dim, 31);
    let weight_f16: Vec<u8> = wf32
        .iter()
        .flat_map(|&v| half::f16::from_f32(1.0 + v).to_bits().to_le_bytes())
        .collect();

    // Interleaved q+g row: stride = n_head * 2 * head_dim, head h at `h*2*head_dim`, query = first
    // head_dim of the head block, the trailing head_dim is POISON (the gate half) the kernel must
    // never read. Mirrors qwen35's attn q+g buffer (x_stride = nh*2*hd).
    let head_stride = 2 * head_dim;
    let stride = n_head * head_stride;
    let qpacked = gen(rows * hw, 6); // logical per-head queries, packed [rows, n_head, head_dim]
    let mut wide = vec![0f32; rows * stride];
    for row in 0..rows {
        for h in 0..n_head {
            for i in 0..head_dim {
                wide[row * stride + h * head_stride + i] =
                    qpacked[(row * n_head + h) * head_dim + i];
            }
            // poison the gate half of each head block
            for i in head_dim..head_stride {
                wide[row * stride + h * head_stride + i] = 1000.0 + (row * stride + h) as f32;
            }
        }
    }

    let c = run_qk_norm_rope(
        &cpu,
        &wide,
        &weight_f16,
        &positions,
        rows,
        n_head,
        head_dim,
        rope_dim,
        theta,
        eps,
        stride,
    );
    let r = run_qk_norm_rope(
        &be,
        &wide,
        &weight_f16,
        &positions,
        rows,
        n_head,
        head_dim,
        rope_dim,
        theta,
        eps,
        stride,
    );
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-6);
    println!(
        "QkNormRope strided(stride={stride}) max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    // Guard against a silently-zero output masquerading as agreement.
    assert!(
        ref_mag > 1e-3,
        "QkNormRope reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 2e-3,
        "QkNormRope strided diverges from CPU reference (OOB head stride / packed-vs-strided?): abs={e:e} rel={:e}",
        e / ref_mag
    );
}

// ── Conv1dSilu (depthwise causal conv + SiLU, rolling state) vs CPU ───────────

/// Run a single-`Op::Conv1dSilu` graph on `be` and return BOTH the downloaded output
/// `[rows, channels]` AND the updated `state` `[(kernel-1), channels]`. `state` is bound as an
/// F32 Input so the op mutates it in place; the backend must write the trailing `kernel-1`
/// columns of the virtual `[state ‖ x]` sequence back to that buffer (verified by downloading it
/// after execute — the same in-place-state-persistence contract as `seam_op_parity`'s state test).
/// `weight` uploads as raw F16 bytes (dequantized on first touch), so CPU (f16→f32) and ROCm
/// (f16 as-is) see identical kernel taps.
fn run_conv1d_silu(
    be: &dyn Backend,
    x: &[f32],
    weight_f16: &[u8],
    state_init: &[f32],
    rows: usize,
    channels: usize,
    kernel: usize,
) -> (Vec<f32>, Vec<f32>) {
    let km1 = kernel - 1;
    let mut g = Graph::new();
    let xid = g.input(f32d(rows * channels));
    let wid = g.weight(TensorDesc::new(vec![channels * kernel], DType::F16));
    let sid = g.input(f32d(km1 * channels)); // F32 Input → mutated in place, read back after
    let dst = g.output(f32d(rows * channels));
    g.push(Op::Conv1dSilu {
        x: xid,
        weight: wid,
        state: sid,
        dst,
        rows: rows as u32,
        channels: channels as u32,
        kernel: kernel as u32,
    });
    let plan = be.compile(&g).expect("compile");
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let wb = be.alloc(weight_f16.len(), BufferUsage::Weights).expect("w");
    be.upload(wb.as_ref(), weight_f16).unwrap();
    let sb = be
        .alloc(state_init.len() * 4, BufferUsage::Activations)
        .expect("state");
    be.upload(sb.as_ref(), bytemuck::cast_slice(state_init))
        .unwrap();
    let ob = be
        .alloc(rows * channels * 4, BufferUsage::Readback)
        .expect("out");
    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(wid, wb.as_ref());
    b.bind(sid, sb.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut out = vec![0f32; rows * channels];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut out))
        .unwrap();
    let mut ns = vec![0f32; km1 * channels];
    be.download(sb.as_ref(), bytemuck::cast_slice_mut(&mut ns))
        .unwrap();
    (out, ns)
}

/// `Op::Conv1dSilu` (qwen35's depthwise causal 1-D conv + SiLU, rolling `state`) must match the CPU
/// reference for a MULTI-ROW (prefill) input with a NON-trivial initial `state` — BOTH the output
/// AND the updated state. The pre-fix ROCm kernel applied the SAME unchanged `state` to every one of
/// the `rows` output rows (no per-row window advance) and the host shift chained from the ORIGINAL
/// `x` for each row, so for `rows > 1` both the conv outputs and the returned state were wrong (only
/// `rows == 1` decode was correct) — one of the two bugs making qwen35 prefill incoherent. This runs
/// the SAME single-op graph on `RocmBackend` and `infr_cpu::CpuBackend`; it fails loudly without the
/// fix. Correct semantics: convolve the virtual sequence `[state ‖ x]` per (row, channel), and the
/// returned state is that sequence's trailing `kernel-1` columns.
#[test]
#[ignore = "requires a ROCm GPU"]
fn conv1d_silu_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, channels, kernel) = (6usize, 32usize, 4usize); // rows > 1 (prefill), rows > kernel-1
    let km1 = kernel - 1;

    let x = gen(rows * channels, 6);
    // Per-channel kernel [channels, kernel], F16 bytes (CPU dequants f16→f32, ROCm reads f16 as-is).
    let wf32 = gen(channels * kernel, 7);
    let w_bytes: Vec<u8> = wf32
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    // NON-trivial initial state (exercises the cross-row warmup carry — a zeroed state would hide
    // the "state applied to every row unchanged" bug on the first km1 rows).
    let state_init = gen(km1 * channels, 13);

    let (c_out, c_state) = run_conv1d_silu(&cpu, &x, &w_bytes, &state_init, rows, channels, kernel);
    let (r_out, r_state) = run_conv1d_silu(&be, &x, &w_bytes, &state_init, rows, channels, kernel);

    let eo = maxerr(&c_out, &r_out);
    let out_mag = maxabs(&c_out).max(1e-6);
    let es = maxerr(&c_state, &r_state);
    let st_mag = maxabs(&c_state).max(1e-6);
    println!(
        "Conv1dSilu multirow(rows={rows}) out max_err={eo:e} max|ref|={out_mag:e} rel={:e} | state max_err={es:e} max|ref|={st_mag:e} rel={:e}",
        eo / out_mag,
        es / st_mag
    );
    // Guard against a silently-zero output/state masquerading as agreement.
    assert!(
        out_mag > 1e-3,
        "Conv1dSilu output reference is all-zero — test is vacuous"
    );
    assert!(
        st_mag > 1e-3,
        "Conv1dSilu state reference is all-zero — test is vacuous"
    );
    assert!(
        eo / out_mag < 2e-3,
        "Conv1dSilu multirow output diverges from CPU reference (per-row window not advanced?): abs={eo:e} rel={:e}",
        eo / out_mag
    );
    // The updated state is a pure gather from `[state ‖ x]` (no arithmetic), so f16 weight rounding
    // does not touch it — the returned state must match the CPU reference near-exactly.
    assert!(
        es / st_mag < 1e-5,
        "Conv1dSilu multirow updated state diverges from CPU reference (host chain from original x?): abs={es:e} rel={:e}",
        es / st_mag
    );
}

// ── DeltaNet (gated linear-attention recurrence, persistent S state) vs CPU ──

/// Run a single-`Op::DeltaNet` graph on `be` and return BOTH the downloaded output
/// `[rows, n_vhead*head_v]` AND the mutated recurrent state `[n_vhead, head_k, head_v]`. `state` is
/// bound as an F32 Input the op mutates IN PLACE (read back after execute — the persistent-state
/// contract: qwen35's DeltaNet-S survives across `execute` calls). `a_coef`/`dt_bias` upload as raw
/// F16 bytes (CPU dequants f16→f32, ROCm reads f16 as-is, so both see identical per-head scalars).
#[allow(clippy::too_many_arguments)]
fn run_deltanet(
    be: &dyn Backend,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bcoef: &[f32],
    acoef_in: &[f32],
    a_coef_f16: &[u8],
    dt_bias_f16: &[u8],
    state_init: &[f32],
    rows: usize,
    n_vhead: usize,
    n_khead: usize,
    head_k: usize,
    head_v: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut g = Graph::new();
    let qid = g.input(f32d(rows * n_khead * head_k));
    let kid = g.input(f32d(rows * n_khead * head_k));
    let vid = g.input(f32d(rows * n_vhead * head_v));
    let bid = g.input(f32d(rows * n_vhead));
    let aid = g.input(f32d(rows * n_vhead));
    let acid = g.weight(TensorDesc::new(vec![n_vhead], DType::F16));
    let dtid = g.weight(TensorDesc::new(vec![n_vhead], DType::F16));
    let sid = g.input(f32d(n_vhead * head_k * head_v)); // F32 Input → mutated in place, read back
    let dst = g.output(f32d(rows * n_vhead * head_v));
    g.push(Op::DeltaNet {
        q: qid,
        k: kid,
        v: vid,
        b: bid,
        a: aid,
        a_coef: acid,
        dt_bias: dtid,
        state: sid,
        dst,
        rows: rows as u32,
        n_vhead: n_vhead as u32,
        n_khead: n_khead as u32,
        head_k: head_k as u32,
        head_v: head_v as u32,
        eps,
        src_stride: 0,
    });
    let plan = be.compile(&g).expect("compile");
    let up_f32 = |data: &[f32], usage| {
        let b = be.alloc(data.len() * 4, usage).expect("alloc f32");
        be.upload(b.as_ref(), bytemuck::cast_slice(data)).unwrap();
        b
    };
    let up_bytes = |data: &[u8], usage| {
        let b = be.alloc(data.len(), usage).expect("alloc bytes");
        be.upload(b.as_ref(), data).unwrap();
        b
    };
    let qb = up_f32(q, BufferUsage::Activations);
    let kb = up_f32(k, BufferUsage::Activations);
    let vb = up_f32(v, BufferUsage::Activations);
    let bb = up_f32(bcoef, BufferUsage::Activations);
    let ab = up_f32(acoef_in, BufferUsage::Activations);
    let acb = up_bytes(a_coef_f16, BufferUsage::Weights);
    let dtb = up_bytes(dt_bias_f16, BufferUsage::Weights);
    let sb = up_f32(state_init, BufferUsage::Activations);
    let ob = be
        .alloc(rows * n_vhead * head_v * 4, BufferUsage::Readback)
        .expect("out");
    let mut bnd = Bindings::new();
    bnd.bind(qid, qb.as_ref());
    bnd.bind(kid, kb.as_ref());
    bnd.bind(vid, vb.as_ref());
    bnd.bind(bid, bb.as_ref());
    bnd.bind(aid, ab.as_ref());
    bnd.bind(acid, acb.as_ref());
    bnd.bind(dtid, dtb.as_ref());
    bnd.bind(sid, sb.as_ref());
    bnd.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &bnd).expect("execute");
    let mut out = vec![0f32; rows * n_vhead * head_v];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut out))
        .unwrap();
    let mut ns = vec![0f32; n_vhead * head_k * head_v];
    be.download(sb.as_ref(), bytemuck::cast_slice_mut(&mut ns))
        .unwrap();
    (out, ns)
}

/// `Op::DeltaNet` (qwen35's gated-DeltaNet linear-attention recurrence) must match the CPU reference
/// (`infr_cpu` `deltanet_scan`) for BOTH the output AND the mutated persistent `S` state, on a
/// MULTI-ROW (prefill) input with a NON-trivial initial `S` — the token recurrence carries `S`
/// sequentially across rows, so a per-row-independent or mis-sequenced kernel is wrong for rows>1.
/// Uses GQA (`n_khead < n_vhead`) so the value→key head mapping is exercised, and injects large
/// `a` values so the decay's softplus is pushed into its overflow regime.
///
/// The pre-fix ROCm kernel had four divergences any of which this catches: (1) the state was stored
/// TRANSPOSED (`S[d*head_k+k]`) so the mutated-state readback disagreed with the CPU `[head_k,
/// head_v]` layout; (2) GQA used `vh/(n_vhead/n_khead)` (grouped) instead of the CPU/qwen35
/// INTERLEAVED `vh % n_khead`, so every value head past the first group read the wrong q/k; (3) the
/// decay used the naive `log(1+exp(z))` softplus, which overflows to +inf for large z and (with
/// a_coef<0) collapses decay to 0, silently wiping the state every token; (4) `eps` was hardcoded.
/// It also runs a decode (rows==1) case. Fails loudly without the fix; guarded against vacuous
/// all-zero agreement.
#[test]
#[ignore = "requires a ROCm GPU"]
fn deltanet_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let eps = 1e-6f32;
    // Two shapes: a small GQA case (nk=2 < nv=4 — exercises the interleaved value→key head map),
    // and the REAL qwen35-0.8B DeltaNet shape (nv=nk=16, head_k=head_v=128) to rule out any
    // large-dim / long-reduction divergence at the size the model actually runs.
    for &(n_vhead, n_khead, head_k, head_v) in
        &[(4usize, 2usize, 8usize, 8usize), (16, 16, 128, 128)]
    {
        // Per-head scalars: a_coef modestly NEGATIVE (the sign that makes an overflowing softplus
        // collapse decay to zero), dt_bias small. F16 like the seam's dequant path.
        let acoef_f32: Vec<f32> = (0..n_vhead).map(|h| -0.02 * (1.0 + h as f32)).collect();
        let dtbias_f32: Vec<f32> = gen(n_vhead, 71);
        let a_coef_f16: Vec<u8> = acoef_f32
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let dt_bias_f16: Vec<u8> = dtbias_f32
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();

        for &rows in &[5usize, 1usize] {
            let q = gen(rows * n_khead * head_k, 6);
            let k = gen(rows * n_khead * head_k, 9);
            let v = gen(rows * n_vhead * head_v, 12);
            let bcoef = gen(rows * n_vhead, 15);
            // `a`: mostly small, but a couple large-z entries so softplus(z) overflows the naive form
            // (z ~ 100 → exp(z) is +inf) while stable softplus stays finite (sp ≈ z, decay ≈ exp(ac·z)).
            let mut acoef_in = gen(rows * n_vhead, 18);
            acoef_in[0] = 100.0;
            if rows * n_vhead > n_vhead {
                acoef_in[n_vhead] = 100.0;
            }
            // NON-trivial initial S state (a zeroed S would hide the transposed-layout bug entirely).
            let state_init = gen(n_vhead * head_k * head_v, 21);

            let (c_out, c_state) = {
                // The CPU backend mutates `state` in place too — run it through the SAME single-op graph.
                run_deltanet(
                    &cpu,
                    &q,
                    &k,
                    &v,
                    &bcoef,
                    &acoef_in,
                    &a_coef_f16,
                    &dt_bias_f16,
                    &state_init,
                    rows,
                    n_vhead,
                    n_khead,
                    head_k,
                    head_v,
                    eps,
                )
            };
            let (r_out, r_state) = run_deltanet(
                &be,
                &q,
                &k,
                &v,
                &bcoef,
                &acoef_in,
                &a_coef_f16,
                &dt_bias_f16,
                &state_init,
                rows,
                n_vhead,
                n_khead,
                head_k,
                head_v,
                eps,
            );

            let eo = maxerr(&c_out, &r_out);
            let out_mag = maxabs(&c_out).max(1e-6);
            let es = maxerr(&c_state, &r_state);
            let st_mag = maxabs(&c_state).max(1e-6);
            println!(
            "DeltaNet nv={n_vhead} nk={n_khead} kd={head_k} vd={head_v} rows={rows} out max_err={eo:e} max|ref|={out_mag:e} rel={:e} | state max_err={es:e} max|ref|={st_mag:e} rel={:e}",
            eo / out_mag,
            es / st_mag
        );
            // Guard against a silently-zero output/state masquerading as agreement.
            assert!(
                out_mag > 1e-3,
                "DeltaNet rows={rows} output reference is all-zero — test is vacuous"
            );
            assert!(
                st_mag > 1e-3,
                "DeltaNet rows={rows} state reference is all-zero — test is vacuous"
            );
            assert!(
            eo / out_mag < 2e-2,
            "DeltaNet rows={rows} output diverges from CPU reference (GQA map / softplus / sequencing?): abs={eo:e} rel={:e}",
            eo / out_mag
        );
            assert!(
            es / st_mag < 2e-2,
            "DeltaNet rows={rows} mutated state diverges from CPU reference (transposed layout / decay?): abs={es:e} rel={:e}",
            es / st_mag
        );
        }
    }
}
