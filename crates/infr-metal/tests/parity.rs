//! Numeric-parity tests: run the SAME graph op on `infr-cpu` (the trusted reference interpreter)
//! and on `infr-metal`, assert the outputs match. This is the contract a backend must satisfy.
//!
//! macOS-only (the backend is), and each test is `#[ignore]`d — it needs a real Metal device, like
//! the Vulkan GPU tests. Run them with `cargo test -p infr-metal -- --include-ignored`.
#![cfg(target_os = "macos")]

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_cpu::CpuBackend;
use infr_metal::MetalBackend;

// ---- deterministic test data (LCG, no rng dependency) ----
fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}
fn rand_f32(n: usize, mut seed: u64) -> Vec<f32> {
    (0..n)
        .map(|_| ((lcg(&mut seed) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
        .collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}
fn i32_bytes(v: &[i32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

/// Bind raw byte buffers to graph handles, run one graph on `be`, and read back `out` as f32.
fn run(
    be: &dyn Backend,
    g: &Graph,
    bound: &[(TensorId, Vec<u8>)],
    out: TensorId,
    out_n: usize,
) -> Vec<f32> {
    let mut bufs: Vec<(TensorId, Box<dyn Buffer>)> = Vec::new();
    for (id, bytes) in bound {
        let b = be
            .alloc(bytes.len().max(4), BufferUsage::Activations)
            .unwrap();
        be.upload(b.as_ref(), bytes).unwrap();
        bufs.push((*id, b));
    }
    let ob = be.alloc(out_n * 4, BufferUsage::Activations).unwrap();
    bufs.push((out, ob));

    let mut binds = Bindings::new();
    for (id, b) in &bufs {
        binds.bind(*id, b.as_ref());
    }
    let plan = be.compile(g).unwrap();
    be.execute(plan.as_ref(), &binds).unwrap();
    be.sync().unwrap();

    let ob = &bufs.iter().find(|(i, _)| *i == out).unwrap().1;
    let mut bytes = vec![0u8; out_n * 4];
    be.download(ob.as_ref(), &mut bytes).unwrap();
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

/// Run one graph on `be`, then download raw bytes of the bound tensor `readback` (used for
/// stateful ops like WriteKv that mutate a bound buffer instead of producing an Output).
fn run_readback(
    be: &dyn Backend,
    g: &Graph,
    bound: &[(TensorId, Vec<u8>)],
    readback: TensorId,
    byte_len: usize,
) -> Vec<u8> {
    let mut bufs: Vec<(TensorId, Box<dyn Buffer>)> = Vec::new();
    for (id, bytes) in bound {
        let b = be
            .alloc(bytes.len().max(4), BufferUsage::Activations)
            .unwrap();
        be.upload(b.as_ref(), bytes).unwrap();
        bufs.push((*id, b));
    }
    let mut binds = Bindings::new();
    for (id, b) in &bufs {
        binds.bind(*id, b.as_ref());
    }
    let plan = be.compile(g).unwrap();
    be.execute(plan.as_ref(), &binds).unwrap();
    be.sync().unwrap();
    let rb = &bufs.iter().find(|(i, _)| *i == readback).unwrap().1;
    let mut bytes = vec![0u8; byte_len];
    be.download(rb.as_ref(), &mut bytes).unwrap();
    bytes
}

/// Run one graph on `be` and read back several tensors as f32 (Outputs, or mutated f32 Inputs like
/// recurrent state). Any `read` id not present in `bound` is allocated zeroed and bound.
fn run_multi(
    be: &dyn Backend,
    g: &Graph,
    bound: &[(TensorId, Vec<u8>)],
    reads: &[(TensorId, usize)],
) -> Vec<Vec<f32>> {
    let mut bufs: Vec<(TensorId, Box<dyn Buffer>)> = Vec::new();
    for (id, bytes) in bound {
        let b = be
            .alloc(bytes.len().max(4), BufferUsage::Activations)
            .unwrap();
        be.upload(b.as_ref(), bytes).unwrap();
        bufs.push((*id, b));
    }
    for (id, n) in reads {
        if !bound.iter().any(|(bid, _)| bid == id) {
            let b = be.alloc(n * 4, BufferUsage::Activations).unwrap();
            bufs.push((*id, b));
        }
    }
    let mut binds = Bindings::new();
    for (id, b) in &bufs {
        binds.bind(*id, b.as_ref());
    }
    let plan = be.compile(g).unwrap();
    be.execute(plan.as_ref(), &binds).unwrap();
    be.sync().unwrap();
    reads
        .iter()
        .map(|(id, n)| {
            let b = &bufs.iter().find(|(i, _)| i == id).unwrap().1;
            let mut bytes = vec![0u8; n * 4];
            be.download(b.as_ref(), &mut bytes).unwrap();
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        })
        .collect()
}

fn assert_close(cpu: &[f32], mtl: &[f32], tol: f32, what: &str) {
    assert_eq!(cpu.len(), mtl.len(), "{what}: length");
    for (i, (c, m)) in cpu.iter().zip(mtl.iter()).enumerate() {
        let err = (c - m).abs() / c.abs().max(1.0);
        assert!(
            err <= tol,
            "{what} elem {i}: cpu={c} metal={m} err={err} > {tol}"
        );
    }
}

/// Run on both backends and assert close.
fn assert_parity(g: &Graph, bound: &[(TensorId, Vec<u8>)], out: TensorId, out_n: usize, tol: f32) {
    let cpu = run(&CpuBackend::new(), g, bound, out, out_n);
    let mtl = run(
        &MetalBackend::new().expect("metal backend"),
        g,
        bound,
        out,
        out_n,
    );
    assert_eq!(cpu.len(), mtl.len());
    for (i, (c, m)) in cpu.iter().zip(mtl.iter()).enumerate() {
        let err = (c - m).abs() / c.abs().max(1.0);
        assert!(
            err <= tol,
            "elem {i}: cpu={c} metal={m} rel_err={err} > tol={tol}"
        );
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn add_parity() {
    let n = 4096usize;
    let mut g = Graph::new();
    let a = g.input(TensorDesc::new(vec![n], DType::F32));
    let b = g.input(TensorDesc::new(vec![n], DType::F32));
    let dst = g.output(TensorDesc::new(vec![n], DType::F32));
    g.push(Op::Add {
        a,
        b,
        dst,
        n: n as u32,
    });
    let bound = vec![
        (a, f32_bytes(&rand_f32(n, 1))),
        (b, f32_bytes(&rand_f32(n, 2))),
    ];
    assert_parity(&g, &bound, dst, n, 0.0);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn scale_parity() {
    let n = 4096usize;
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![n], DType::F32));
    let dst = g.output(TensorDesc::new(vec![n], DType::F32));
    g.push(Op::Scale {
        x,
        dst,
        s: 0.125,
        n: n as u32,
    });
    let bound = vec![(x, f32_bytes(&rand_f32(n, 3)))];
    assert_parity(&g, &bound, dst, n, 0.0);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn softcap_parity() {
    let n = 4096usize;
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![n], DType::F32));
    let dst = g.output(TensorDesc::new(vec![n], DType::F32));
    g.push(Op::Softcap {
        x,
        dst,
        cap: 30.0,
        n: n as u32,
    });
    // scale inputs up so tanh saturation is exercised
    let xs: Vec<f32> = rand_f32(n, 4).iter().map(|v| v * 60.0).collect();
    let bound = vec![(x, f32_bytes(&xs))];
    assert_parity(&g, &bound, dst, n, 1e-5);
}

// naive reference matmul: dst[r,o] = sum_i x[r,i] * w[o,i]   (w row-major [out_f, in_f])
fn ref_linear(x: &[f32], w: &[f32], m: usize, in_f: usize, out_f: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * out_f];
    for r in 0..m {
        for o in 0..out_f {
            let mut acc = 0f32;
            for i in 0..in_f {
                acc += x[r * in_f + i] * w[o * in_f + i];
            }
            out[r * out_f + o] = acc;
        }
    }
    out
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
        .collect()
}

// Quantize a whole row-major weight to GGUF Q8_0 (32-elem blocks: f16 scale + 32×i8).
fn quantize_q8_0(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in w.chunks(32) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for &v in blk {
            let q = if d > 0.0 {
                (v / d).round().clamp(-127.0, 127.0) as i8
            } else {
                0
            };
            out.push(q as u8);
        }
    }
    out
}

// A deterministic LCG byte stream — arbitrary but reproducible payload for the k-quant nibble
// fields (which decode to finite values for *any* byte pattern).
fn lcg_bytes(mut seed: u32, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as u8
        })
        .collect()
}

// Synthesize a well-formed GGUF Q4_K weight: 144-byte / 256-elem blocks laid out as
// [f16 d][f16 dmin][12B scales][128B qs]. We only need *valid* bytes with finite f16 scales — not a
// faithful quantization of any target — because the parity test compares Metal's dequant against a
// reference dequant of these SAME bytes. Scale/nibble fields take an arbitrary reproducible pattern.
fn synth_q4k(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0, "Q4_K blocks are 256 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 144];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes()); // d
        blk[2..4].copy_from_slice(&half::f16::from_f32(0.10).to_le_bytes()); // dmin
        blk[4..144].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 140)); // scales + qs
        out.extend_from_slice(&blk);
    }
    out
}

// Synthesize a well-formed GGUF Q6_K weight: 210-byte / 256-elem blocks laid out as
// [128B ql][64B qh][16×i8 scales][f16 d]. Same rationale as `synth_q4k`.
fn synth_q6k(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0, "Q6_K blocks are 256 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 210];
        blk[0..208].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 208)); // ql + qh + scales
        blk[208..210].copy_from_slice(&half::f16::from_f32(0.03).to_le_bytes()); // d
        out.extend_from_slice(&blk);
    }
    out
}

// Shared quant-Linear parity check: Metal dequants `wbytes` (via infr_gguf) and matmuls; compare to
// a reference that dequants the SAME bytes and matmuls — isolates Metal's quant-weight path.
fn check_quant_linear_parity(dtype: DType, wbytes: Vec<u8>, m: usize, in_f: usize, out_f: usize) {
    use infr_gguf::dequant::dequant_block;
    let xs = rand_f32(m * in_f, 24);
    let wref = dequant_block(dtype, &wbytes).unwrap();
    let reference = ref_linear(&xs, &wref, m, in_f, out_f);

    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], dtype));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
    });
    let bound = vec![(x, f32_bytes(&xs)), (w, wbytes)];
    let mtl = run(
        &MetalBackend::new().expect("metal backend"),
        &g,
        &bound,
        dst,
        m * out_f,
    );
    for (i, (r, mm)) in reference.iter().zip(mtl.iter()).enumerate() {
        let err = (r - mm).abs() / r.abs().max(1.0);
        assert!(
            err <= 1e-3,
            "{dtype:?} elem {i}: ref={r} metal={mm} err={err}"
        );
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_f32_parity() {
    let (m, in_f, out_f) = (3usize, 512usize, 200usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], DType::F32));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(m * in_f, 20))),
        (w, f32_bytes(&rand_f32(out_f * in_f, 21))),
    ];
    assert_parity(&g, &bound, dst, m * out_f, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_f16_parity() {
    let (m, in_f, out_f) = (2usize, 256usize, 128usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], DType::F16));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(m * in_f, 22))),
        (w, f16_bytes(&rand_f32(out_f * in_f, 23))),
    ];
    assert_parity(&g, &bound, dst, m * out_f, 1e-3);
}

// Quantized Linear: Metal dequants the weight to f32 (via infr_gguf) and matmuls. Compare to a
// reference that dequants the SAME bytes and matmuls — isolates Metal's quant-weight path.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q8_0_matches_dequant_reference() {
    use infr_gguf::dequant::dequant_block;
    let (m, in_f, out_f) = (2usize, 256usize, 96usize);
    let xs = rand_f32(m * in_f, 24);
    let wf = rand_f32(out_f * in_f, 25);
    let wbytes = quantize_q8_0(&wf);
    let wref = dequant_block(DType::Q8_0, &wbytes).unwrap();
    let reference = ref_linear(&xs, &wref, m, in_f, out_f);

    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], DType::Q8_0));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
    });
    let bound = vec![(x, f32_bytes(&xs)), (w, wbytes)];
    let mtl = run(
        &MetalBackend::new().expect("metal backend"),
        &g,
        &bound,
        dst,
        m * out_f,
    );
    for (i, (r, mm)) in reference.iter().zip(mtl.iter()).enumerate() {
        let err = (r - mm).abs() / r.abs().max(1.0);
        assert!(err <= 1e-3, "elem {i}: ref={r} metal={mm} err={err}");
    }
}

// K-quants are the formats real checkpoints actually ship. Exercise the Metal dequant path
// (`weight_buf` → `dequant_block`) for Q4_K and Q6_K, same dequant-reference comparison as Q8_0.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4k_matches_dequant_reference() {
    let (m, in_f, out_f) = (2usize, 256usize, 96usize);
    check_quant_linear_parity(DType::Q4K, synth_q4k(out_f * in_f, 26), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q6k_matches_dequant_reference() {
    let (m, in_f, out_f) = (2usize, 256usize, 96usize);
    check_quant_linear_parity(DType::Q6K, synth_q6k(out_f * in_f, 27), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn rmsnorm_parity() {
    let (rows, dim) = (7usize, 512usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
    let w = g.weight(TensorDesc::new(vec![dim], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
    g.push(Op::RmsNorm {
        x,
        weight: w,
        dst,
        rows: rows as u32,
        dim: dim as u32,
        eps: 1e-6,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * dim, 10))),
        (w, f32_bytes(&rand_f32(dim, 11))),
    ];
    assert_parity(&g, &bound, dst, rows * dim, 1e-5);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn qknorm_parity() {
    let (rows, nh, hd) = (5usize, 8usize, 128usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let w = g.weight(TensorDesc::new(vec![hd], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::QkNorm {
        x,
        weight: w,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        eps: 1e-6,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * nh * hd, 12))),
        (w, f32_bytes(&rand_f32(hd, 13))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-5);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn rope_parity() {
    let (rows, nh, hd, rd) = (4usize, 6usize, 128usize, 128usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Rope {
        x,
        positions: pos,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rd as u32,
        theta: 10000.0,
        freq_factors: None,
    });
    let positions: Vec<i32> = (0..rows as i32).map(|i| i + 3).collect();
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * nh * hd, 30))),
        (pos, i32_bytes(&positions)),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn rope_partial_with_freq_factors_parity() {
    // rope_dim < head_dim (dims beyond rope_dim pass through) + per-pair freq_factors divisor
    let (rows, nh, hd, rd) = (3usize, 4usize, 128usize, 64usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let ff = g.input(TensorDesc::new(vec![rd / 2], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Rope {
        x,
        positions: pos,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rd as u32,
        theta: 1000000.0,
        freq_factors: Some(ff),
    });
    let positions: Vec<i32> = (0..rows as i32).map(|i| i * 2 + 1).collect();
    let ffv: Vec<f32> = (0..rd / 2).map(|i| 1.0 + i as f32 * 0.1).collect();
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * nh * hd, 31))),
        (pos, i32_bytes(&positions)),
        (ff, f32_bytes(&ffv)),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn qknormrope_parity() {
    let (rows, nh, hd, rd) = (4usize, 8usize, 128usize, 128usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let w = g.weight(TensorDesc::new(vec![hd], DType::F32));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::QkNormRope {
        x,
        weight: w,
        positions: pos,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rd as u32,
        theta: 10000.0,
        eps: 1e-6,
        freq_factors: None,
    });
    let positions: Vec<i32> = (0..rows as i32).map(|i| i + 1).collect();
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * nh * hd, 32))),
        (w, f32_bytes(&rand_f32(hd, 33))),
        (pos, i32_bytes(&positions)),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn writekv_f16_parity() {
    // WriteKv casts f32 rows into an f16 cache at row `pos`. Both backends must produce identical
    // f16 bytes.
    let (rows, row_stride, max_ctx, pos) = (2usize, 256usize, 8usize, 3usize);
    let cache_elems = max_ctx * row_stride;
    let mut g = Graph::new();
    let src = g.input(TensorDesc::new(vec![rows, row_stride], DType::F32));
    let cache = g.input(TensorDesc::new(vec![cache_elems], DType::F16));
    g.push(Op::WriteKv {
        src,
        cache,
        rows: rows as u32,
        row_stride: row_stride as u32,
        pos: pos as u32,
    });
    let bound = vec![
        (src, f32_bytes(&rand_f32(rows * row_stride, 40))),
        (cache, vec![0u8; cache_elems * 2]),
    ];
    let cpu = run_readback(&CpuBackend::new(), &g, &bound, cache, cache_elems * 2);
    let mtl = run_readback(
        &MetalBackend::new().expect("metal backend"),
        &g,
        &bound,
        cache,
        cache_elems * 2,
    );
    assert_eq!(cpu, mtl, "WriteKv f16 cache bytes must be identical");
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_gqa_causal_parity() {
    let (rows, kv_len, nh, nkv, hd, pos) = (3usize, 6usize, 8usize, 2usize, 64usize, 0usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::Causal,
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 41))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 42))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 43))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_sliding_window_parity() {
    let (rows, kv_len, nh, nkv, hd, pos, win) =
        (4usize, 10usize, 4usize, 4usize, 64usize, 2usize, 3usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::SlidingWindow(win),
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 44))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 45))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 46))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

fn gated_test(act: infr_core::graph::Activation, up_off: usize, seed: u64) {
    let (rows, nff) = (3usize, 512usize);
    let up_len = rows * nff + up_off;
    let mut g = Graph::new();
    let gate = g.input(TensorDesc::new(vec![rows, nff], DType::F32));
    let up = g.input(TensorDesc::new(vec![up_len], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nff], DType::F32));
    g.push(Op::GatedAct {
        gate,
        up,
        dst,
        rows: rows as u32,
        nff: nff as u32,
        act,
        up_off: up_off as u32,
    });
    let bound = vec![
        (gate, f32_bytes(&rand_f32(rows * nff, seed))),
        (up, f32_bytes(&rand_f32(up_len, seed + 1))),
    ];
    assert_parity(&g, &bound, dst, rows * nff, 1e-5);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn gatedact_silu_parity() {
    gated_test(infr_core::graph::Activation::Silu, 0, 50);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn gatedact_gelu_parity() {
    gated_test(infr_core::graph::Activation::Gelu, 0, 52);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn gatedact_upoff_parity() {
    gated_test(infr_core::graph::Activation::Silu, 128, 54);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn moe_ffn_parity() {
    let (ne, n_expert, n_used, nff) = (64usize, 8usize, 2usize, 128usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![ne], DType::F32));
    let router = g.weight(TensorDesc::new(vec![n_expert, ne], DType::F32));
    let gate = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::F32));
    let up = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::F32));
    let down = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::F32));
    let dst = g.output(TensorDesc::new(vec![ne], DType::F32));
    g.push(Op::MoeFfn {
        x,
        router,
        gate_exps: gate,
        up_exps: up,
        down_exps: down,
        dst,
        ne: ne as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
        n_ff_exp: nff as u32,
        scale: 1.0,
        act: infr_core::graph::Activation::Silu,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(ne, 60))),
        (router, f32_bytes(&rand_f32(n_expert * ne, 61))),
        (gate, f32_bytes(&rand_f32(n_expert * nff * ne, 62))),
        (up, f32_bytes(&rand_f32(n_expert * nff * ne, 63))),
        (down, f32_bytes(&rand_f32(n_expert * ne * nff, 64))),
    ];
    assert_parity(&g, &bound, dst, ne, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn conv1d_silu_parity() {
    let (cc, kk) = (256usize, 4usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![cc], DType::F32));
    let w = g.weight(TensorDesc::new(vec![cc, kk], DType::F32));
    let state = g.input(TensorDesc::new(vec![kk - 1, cc], DType::F32));
    let dst = g.output(TensorDesc::new(vec![cc], DType::F32));
    g.push(Op::Conv1dSilu {
        x,
        weight: w,
        state,
        dst,
        channels: cc as u32,
        kernel: kk as u32,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(cc, 70))),
        (w, f32_bytes(&rand_f32(cc * kk, 71))),
        (state, f32_bytes(&rand_f32((kk - 1) * cc, 72))),
    ];
    let reads = [(dst, cc), (state, (kk - 1) * cc)];
    let cpu = run_multi(&CpuBackend::new(), &g, &bound, &reads);
    let mtl = run_multi(&MetalBackend::new().expect("metal"), &g, &bound, &reads);
    assert_close(&cpu[0], &mtl[0], 1e-5, "conv1d dst");
    assert_close(&cpu[1], &mtl[1], 0.0, "conv1d state"); // shift is exact
}

#[test]
#[ignore = "requires a Metal GPU"]
fn deltanet_parity() {
    let (nv, nk, kd, vd) = (4usize, 2usize, 64usize, 64usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![nk * kd], DType::F32));
    let k = g.input(TensorDesc::new(vec![nk * kd], DType::F32));
    let v = g.input(TensorDesc::new(vec![nv * vd], DType::F32));
    let b = g.input(TensorDesc::new(vec![nv], DType::F32));
    let a = g.input(TensorDesc::new(vec![nv], DType::F32));
    let a_coef = g.weight(TensorDesc::new(vec![nv], DType::F32));
    let dt_bias = g.weight(TensorDesc::new(vec![nv], DType::F32));
    let state = g.input(TensorDesc::new(vec![nv * kd * vd], DType::F32));
    let dst = g.output(TensorDesc::new(vec![nv * vd], DType::F32));
    g.push(Op::DeltaNet {
        q,
        k,
        v,
        b,
        a,
        a_coef,
        dt_bias,
        state,
        dst,
        n_vhead: nv as u32,
        n_khead: nk as u32,
        head_k: kd as u32,
        head_v: vd as u32,
        eps: 1e-6,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(nk * kd, 80))),
        (k, f32_bytes(&rand_f32(nk * kd, 81))),
        (v, f32_bytes(&rand_f32(nv * vd, 82))),
        (b, f32_bytes(&rand_f32(nv, 83))),
        (a, f32_bytes(&rand_f32(nv, 84))),
        (a_coef, f32_bytes(&rand_f32(nv, 85))),
        (dt_bias, f32_bytes(&rand_f32(nv, 86))),
        (state, f32_bytes(&rand_f32(nv * kd * vd, 87))),
    ];
    let reads = [(dst, nv * vd), (state, nv * kd * vd)];
    let cpu = run_multi(&CpuBackend::new(), &g, &bound, &reads);
    let mtl = run_multi(&MetalBackend::new().expect("metal"), &g, &bound, &reads);
    assert_close(&cpu[0], &mtl[0], 1e-4, "deltanet dst");
    assert_close(&cpu[1], &mtl[1], 1e-4, "deltanet state");
}

#[test]
#[ignore = "requires a Metal GPU"]
fn copy_parity() {
    let n = 4096usize;
    let (src_off, dst_off, cnt) = (1000usize, 64usize, 2048usize);
    let mut g = Graph::new();
    let src = g.input(TensorDesc::new(vec![n], DType::F32));
    let dst = g.output(TensorDesc::new(vec![n], DType::F32));
    g.push(Op::Copy {
        src,
        src_off: src_off as u32,
        dst,
        dst_off: dst_off as u32,
        n: cnt as u32,
    });
    let bound = vec![(src, f32_bytes(&rand_f32(n, 5)))];
    assert_parity(&g, &bound, dst, n, 0.0);
}
