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

// Well-formed Q5_0 blocks (22 B / 32 elems: [f16 d][4 B qh][16 B nibbles]) — like the k-quant
// synths, any nibble/bit payload decodes to finite values; only d must be a sane f16.
fn synth_q5_0(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 32, 0, "Q5_0 blocks are 32 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 32) {
        let mut blk = vec![0u8; 22];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.04).to_le_bytes());
        blk[2..22].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 20));
        out.extend_from_slice(&blk);
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

// Linear (m=1, K-quant) immediately followed by a residual Add: the backend's peephole fuses the
// pair into `linear_*_add` (one dispatch, Add's dst written directly). Compare against the CPU
// reference running the UNFUSED pair — the fusion must be invisible.
fn check_linear_add_fusion(dtype: DType, wbytes: Vec<u8>, in_f: usize, out_f: usize) {
    let xs = rand_f32(in_f, 91);
    let res = rand_f32(out_f, 92);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![1, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], dtype));
    let rt = g.input(TensorDesc::new(vec![out_f], DType::F32));
    let mid = g.internal(TensorDesc::new(vec![1, out_f], DType::F32));
    let dst = g.output(TensorDesc::new(vec![out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst: mid,
        m: 1,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    g.push(Op::Add {
        a: mid,
        b: rt,
        dst,
        n: out_f as u32,
    });
    let bound = vec![
        (x, f32_bytes(&xs)),
        (w, wbytes.clone()),
        (rt, f32_bytes(&res)),
    ];
    // Reference: dequant the SAME bytes + f32 matmul + add (the CPU backend Q8-quantizes the
    // activation for quant Linear, so it is not the oracle here — same as the other quant tests).
    let wref = infr_gguf::dequant::dequant_block(dtype, &wbytes).unwrap();
    let mut reference = ref_linear(&xs, &wref, 1, in_f, out_f);
    for (o, r) in reference.iter_mut().zip(res.iter()) {
        *o += r;
    }
    let mtl = run(
        &MetalBackend::new().expect("metal backend"),
        &g,
        &bound,
        dst,
        out_f,
    );
    assert_close(&reference, &mtl, 1e-3, "linear+add fusion");
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q4k_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Q4K, synth_q4k(out_f * in_f, 93), in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q8_0_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    let wf = rand_f32(out_f * in_f, 95);
    check_linear_add_fusion(DType::Q8_0, quantize_q8_0(&wf), in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q6k_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Q6K, synth_q6k(out_f * in_f, 94), in_f, out_f);
}

// Shared quant-Linear parity check: Metal dequants `wbytes` (via infr_gguf) and matmuls; compare to
// a reference that dequants the SAME bytes and matmuls — isolates Metal's quant-weight path.
fn check_quant_linear_parity(dtype: DType, wbytes: Vec<u8>, m: usize, in_f: usize, out_f: usize) {
    check_quant_linear_parity_tol(dtype, wbytes, m, in_f, out_f, 1e-3);
}

fn check_quant_linear_parity_tol(
    dtype: DType,
    wbytes: Vec<u8>,
    m: usize,
    in_f: usize,
    out_f: usize,
    tol: f32,
) {
    check_quant_linear_parity_impl(dtype, wbytes, m, in_f, out_f, tol, false);
}

fn check_quant_linear_parity_impl(
    dtype: DType,
    wbytes: Vec<u8>,
    m: usize,
    in_f: usize,
    out_f: usize,
    tol: f32,
    half_ops: bool,
) {
    use infr_gguf::dequant::dequant_block;
    let mut xs = rand_f32(m * in_f, 24);
    let mut wref = dequant_block(dtype, &wbytes).unwrap();
    // Half-fragment GEMM path (m >= 16): the kernel rounds weights and activations to f16, so
    // the reference mirrors that rounding — the comparison then checks the kernel, not f16.
    if half_ops {
        for v in xs.iter_mut().chain(wref.iter_mut()) {
            *v = half::f16::from_f32(*v).to_f32();
        }
    }
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
        w_off: 0,
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
            err <= tol,
            "{dtype:?} elem {i}: ref={r} metal={mm} err={err} > {tol}"
        );
    }
}

// Fused-QKV slices: several Linear ops share ONE concatenated [Σslices, in_f] weight, each
// reading its rows at `w_off` (the runner's combined-QKV shape — `Op::Linear.w_off`). Every
// slice must match a reference matmul over the dequant of just that slice's rows; the
// non-zero offsets exercise the byte-offset binds into the codes/scm/dd streams.
fn check_linear_woff(
    dtype: DType,
    wbytes: Vec<u8>,
    m: usize,
    in_f: usize,
    slices: &[usize],
    half_ops: bool,
    tol: f32,
) {
    use infr_gguf::dequant::dequant_block;
    let rows_total: usize = slices.iter().sum();
    let mut xs = rand_f32(m * in_f, 34);
    let mut wref = dequant_block(dtype, &wbytes).unwrap();
    if half_ops {
        for v in xs.iter_mut().chain(wref.iter_mut()) {
            *v = half::f16::from_f32(*v).to_f32();
        }
    }
    let be = MetalBackend::new().expect("metal backend");
    let mut row0 = 0usize;
    for &out_f in slices {
        let wslice = &wref[row0 * in_f..(row0 + out_f) * in_f];
        let reference = ref_linear(&xs, wslice, m, in_f, out_f);
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
        let w = g.weight(TensorDesc::new(vec![rows_total, in_f], dtype));
        let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
        g.push(Op::Linear {
            x,
            weight: w,
            dst,
            m: m as u32,
            in_f: in_f as u32,
            out_f: out_f as u32,
            w_off: (row0 * in_f) as u32,
        });
        let bound = vec![(x, f32_bytes(&xs)), (w, wbytes.clone())];
        let mtl = run(&be, &g, &bound, dst, m * out_f);
        for (i, (r, mm)) in reference.iter().zip(mtl.iter()).enumerate() {
            let err = (r - mm).abs() / r.abs().max(1.0);
            assert!(
                err <= tol,
                "{dtype:?} slice@{row0} elem {i}: ref={r} metal={mm} err={err} > {tol}"
            );
        }
        row0 += out_f;
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_q8_0_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 35);
    check_linear_woff(
        DType::Q8_0,
        quantize_q8_0(&wf),
        1,
        in_f,
        &slices,
        false,
        1e-3,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_q8_0_coop_gemm() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 36);
    check_linear_woff(
        DType::Q8_0,
        quantize_q8_0(&wf),
        40,
        in_f,
        &slices,
        true,
        1e-3,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_q4k_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    check_linear_woff(
        DType::Q4K,
        synth_q4k(256 * in_f, 37),
        1,
        in_f,
        &slices,
        false,
        1e-3,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_q4k_coop_gemm() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    check_linear_woff(
        DType::Q4K,
        synth_q4k(256 * in_f, 38),
        40,
        in_f,
        &slices,
        true,
        1e-3,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_q6k_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    check_linear_woff(
        DType::Q6K,
        synth_q6k(256 * in_f, 39),
        1,
        in_f,
        &slices,
        false,
        1e-3,
    );
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
        w_off: 0,
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
        w_off: 0,
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
        w_off: 0,
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

// Native Q5_0: GEMV (four rows per simdgroup, out_f=94 exercises clamped tail rows), HGEMM,
// coop-GEMM, and the Linear+Add fusion — the gemma-family dominant weight format.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5_0_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    check_quant_linear_parity(DType::Q5_0, synth_q5_0(out_f * in_f, 98), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5_0_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    check_quant_linear_parity_impl(
        DType::Q5_0,
        synth_q5_0(out_f * in_f, 99),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5_0_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Q5_0,
        synth_q5_0(out_f * in_f, 100),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q5_0_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Q5_0, synth_q5_0(out_f * in_f, 101), in_f, out_f);
}

// Native Q4_0: quantizer + GEMV/HGEMM/coop-GEMM/add-fusion (TinyLlama-class checkpoints ship
// this format; it rode the factored path at ~6.1 bpw vs the native 4.5).
fn quantize_q4_0(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in w.chunks(32) {
        let amax = blk
            .iter()
            .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
        let d = amax / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for j in 0..16 {
            let q = |v: f32| ((v * id + 8.5) as u8).min(15);
            out.push(q(blk[j]) | (q(blk[j + 16]) << 4));
        }
    }
    out
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4_0_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    let wf = rand_f32(out_f * in_f, 102);
    check_quant_linear_parity(DType::Q4_0, quantize_q4_0(&wf), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4_0_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    let wf = rand_f32(out_f * in_f, 103);
    check_quant_linear_parity_impl(DType::Q4_0, quantize_q4_0(&wf), m, in_f, out_f, 1e-3, true);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4_0_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    let wf = rand_f32(out_f * in_f, 104);
    check_quant_linear_parity_impl(DType::Q4_0, quantize_q4_0(&wf), m, in_f, out_f, 1e-3, true);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q4_0_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    let wf = rand_f32(out_f * in_f, 105);
    check_linear_add_fusion(DType::Q4_0, quantize_q4_0(&wf), in_f, out_f);
}

// Native Q8_0 half-fragment GEMM (m=18 → the hmm route; out_f % 64 != 0 keeps cmm out).
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q8_0_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    let wf = rand_f32(out_f * in_f, 96);
    check_quant_linear_parity_impl(DType::Q8_0, quantize_q8_0(&wf), m, in_f, out_f, 1e-3, true);
}

// Native Q8_0 GEMV (m=1, the mul_mv_q8_0 shape: FOUR rows per simdgroup; out_f=94 exercises the
// clamped tail rows of a partial 4-row group).
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q8_0_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    let wf = rand_f32(out_f * in_f, 97);
    check_quant_linear_parity(DType::Q8_0, quantize_q8_0(&wf), m, in_f, out_f);
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

// m = 2..8 K-quants route to the MULTI-ROW mul_mv GEMV (weight registers reused across 4
// token rows); m=5 exercises the partial token block, out_f=94 the partial row pair.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4k_multirow_matches_dequant_reference() {
    let (m, in_f, out_f) = (5usize, 512usize, 94usize);
    check_quant_linear_parity(DType::Q4K, synth_q4k(out_f * in_f, 110), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q6k_multirow_matches_dequant_reference() {
    let (m, in_f, out_f) = (3usize, 512usize, 96usize);
    check_quant_linear_parity(DType::Q6K, synth_q6k(out_f * in_f, 111), m, in_f, out_f);
}

// m=1 routes to the GEMV kernels — decode's path, distinct from the m=2 row-tiled and m>=16 GEMM
// routes the tests above/below take.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4k_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 96usize);
    check_quant_linear_parity(DType::Q4K, synth_q4k(out_f * in_f, 30), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q6k_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 96usize);
    check_quant_linear_parity(DType::Q6K, synth_q6k(out_f * in_f, 31), m, in_f, out_f);
}

// m >= 16 routes to the simdgroup_matrix GEMM kernels (`linear_quik*_mm`); m=18 also covers the
// partial row tile's scalar fallback (18 = 2 full 8-row tiles + 2 remainder rows).
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4k_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    // m >= 16 runs the half-fragment GEMM (f16 operands, f32 accumulate — the llama.cpp trade,
    // well under quantization error); the reference below rounds its operands the same way.
    check_quant_linear_parity_impl(
        DType::Q4K,
        synth_q4k(out_f * in_f, 28),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

// out_f % 64 == 0 routes to the cooperative-tile GEMM (`linear_*_cmm`); m=40 covers one full
// 32-row tile plus a partial one. Same f16-operand reference as the other GEMM tests.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4k_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Q4K,
        synth_q4k(out_f * in_f, 32),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q6k_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Q6K,
        synth_q6k(out_f * in_f, 33),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q6k_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    // Half-fragment GEMM path — see the Q4K GEMM test for the f16-operand rationale.
    check_quant_linear_parity_impl(
        DType::Q6K,
        synth_q6k(out_f * in_f, 29),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
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

// Q8_0 KV cache (INFR_KV_Q8): the quantization on write must be BYTE-identical between the CPU
// reference and the Metal kernel (d = amax/127 as f16, q = rint(x/d)).
#[test]
#[ignore = "requires a Metal GPU"]
fn writekv_q8_parity() {
    let (rows, row_stride, max_ctx, pos) = (2usize, 256usize, 8usize, 3usize);
    let cache_bytes = max_ctx * row_stride / 32 * 34;
    let mut g = Graph::new();
    let src = g.input(TensorDesc::new(vec![rows, row_stride], DType::F32));
    let cache = g.input(TensorDesc::new(vec![max_ctx * row_stride], DType::Q8_0));
    g.push(Op::WriteKv {
        src,
        cache,
        rows: rows as u32,
        row_stride: row_stride as u32,
        pos: pos as u32,
    });
    let bound = vec![
        (src, f32_bytes(&rand_f32(rows * row_stride, 44))),
        (cache, vec![0u8; cache_bytes]),
    ];
    let cpu = run_readback(&CpuBackend::new(), &g, &bound, cache, cache_bytes);
    let mtl = run_readback(
        &MetalBackend::new().expect("metal backend"),
        &g,
        &bound,
        cache,
        cache_bytes,
    );
    assert_eq!(cpu, mtl, "WriteKv q8 cache bytes must be identical");
}

// Attention over a Q8_0 cache: WriteKv quantizes, Attention dequantizes on read. Both routes:
// the scalar fallback (prefill shape) and the rows==1 vector kernel (decode at depth).
fn q8_attention_test(rows: usize, kv_len: usize, hd: usize, pos: usize, tol: f32, seed: u64) {
    let (nh, nkv) = (8usize, 2usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::Q8_0));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::Q8_0));
    let ksrc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F32));
    let vsrc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let row = (nkv * hd) as u32;
    g.push(Op::WriteKv {
        src: ksrc,
        cache: kc,
        rows: kv_len as u32,
        row_stride: row,
        pos: 0,
    });
    g.push(Op::WriteKv {
        src: vsrc,
        cache: vc,
        rows: kv_len as u32,
        row_stride: row,
        pos: 0,
    });
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
    let nkv_elems = kv_len * nkv * hd;
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, seed))),
        (kc, vec![0u8; nkv_elems / 32 * 34]),
        (vc, vec![0u8; nkv_elems / 32 * 34]),
        (ksrc, f32_bytes(&rand_f32(nkv_elems, seed + 1))),
        (vsrc, f32_bytes(&rand_f32(nkv_elems, seed + 2))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, tol);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_q8_scalar_parity() {
    q8_attention_test(3, 6, 64, 3, 1e-4, 240);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_q8_vec_parity() {
    q8_attention_test(1, 200, 128, 199, 1e-4, 250);
}

// Wide q8 launch: routes to the cooperative q8 flash (dequant-staged KV tiles). Q rounds to f16
// on this path (the flash trade), hence the flash-class tolerance.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_q8_flash_parity() {
    q8_attention_test(17, 136, 128, 119, 5e-3, 260);
}

// hd=256 q8 decode (gemma + INFR_KV_Q8): the NSG=16 q8 vector instantiation.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_q8_vec_hd256_parity() {
    q8_attention_test(1, 200, 256, 199, 1e-4, 270);
}

// ── Decoupled quant KV (mainline block quants q4_0/q4_1/q5_0/q5_1/iq4_nl, dense bf16, TurboQuant
// turbo2/3/4, dense f32). WriteKv quantizes into the compact cache; Attention expands each
// quantized/bf16 side into a transient f16 scratch (f32 reads natively via attention_f32) and runs
// the standard f16 attention over it — the ported Vulkan dequant→f16 prepass. Parity is against the
// CPU oracle, which dequants to f32 and runs f32 SDPA; the tolerances cover the extra f16-scratch
// attention rounding (looser than the q8 native-read path, which accumulates in float). Each
// quantize/dequant kernel is a bit-for-bit port of the CPU reference so only the attention precision
// differs, not the stored quant values. K stays f16 in the common decoupled shape (high-precision K,
// quantized V — llama's guidance); coupled quant/quant is also covered.

fn kv_bytes(dt: DType, elems: usize) -> usize {
    match dt {
        DType::Q4_0 | DType::Iq4Nl => elems / 32 * 18,
        DType::Q4_1 => elems / 32 * 20,
        DType::Q5_0 => elems / 32 * 22,
        DType::Q5_1 => elems / 32 * 24,
        DType::Turbo2 => elems / 128 * 34,
        DType::Turbo3 => elems / 128 * 50,
        DType::Turbo4 => elems / 128 * 66,
        DType::F16 | DType::Bf16 => elems * 2,
        _ => elems * 4, // F32
    }
}

#[allow(clippy::too_many_arguments)]
fn kvquant_attention_test(
    kdt: DType,
    vdt: DType,
    rows: usize,
    kv_len: usize,
    hd: usize,
    pos: usize,
    tol: f32,
    seed: u64,
) {
    let (nh, nkv) = (8usize, 2usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], kdt));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], vdt));
    let ksrc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F32));
    let vsrc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let row = (nkv * hd) as u32;
    g.push(Op::WriteKv {
        src: ksrc,
        cache: kc,
        rows: kv_len as u32,
        row_stride: row,
        pos: 0,
    });
    g.push(Op::WriteKv {
        src: vsrc,
        cache: vc,
        rows: kv_len as u32,
        row_stride: row,
        pos: 0,
    });
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
    let nkv_elems = kv_len * nkv * hd;
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, seed))),
        (kc, vec![0u8; kv_bytes(kdt, nkv_elems)]),
        (vc, vec![0u8; kv_bytes(vdt, nkv_elems)]),
        (ksrc, f32_bytes(&rand_f32(nkv_elems, seed + 1))),
        (vsrc, f32_bytes(&rand_f32(nkv_elems, seed + 2))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, tol);
}

// Block quants, decoupled (K=f16 native, V=quant prepassed) at the rows==1 vector-flash decode
// shape, and coupled (quant/quant) at the scalar prefill shape. Both routes read the f16 scratch.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_q4_0_parity() {
    kvquant_attention_test(DType::F16, DType::Q4_0, 1, 200, 128, 199, 6e-3, 300);
    kvquant_attention_test(DType::Q4_0, DType::Q4_0, 3, 6, 64, 3, 6e-3, 305);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_q4_1_parity() {
    kvquant_attention_test(DType::F16, DType::Q4_1, 1, 200, 128, 199, 6e-3, 310);
    kvquant_attention_test(DType::Q4_1, DType::Q4_1, 3, 6, 64, 3, 6e-3, 315);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_q5_0_parity() {
    kvquant_attention_test(DType::F16, DType::Q5_0, 1, 200, 128, 199, 6e-3, 320);
    kvquant_attention_test(DType::Q5_0, DType::Q5_0, 3, 6, 64, 3, 6e-3, 325);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_q5_1_parity() {
    kvquant_attention_test(DType::F16, DType::Q5_1, 1, 200, 128, 199, 6e-3, 330);
    kvquant_attention_test(DType::Q5_1, DType::Q5_1, 3, 6, 64, 3, 6e-3, 335);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_iq4_nl_parity() {
    kvquant_attention_test(DType::F16, DType::Iq4Nl, 1, 200, 128, 199, 6e-3, 340);
    kvquant_attention_test(DType::Iq4Nl, DType::Iq4Nl, 3, 6, 64, 3, 6e-3, 345);
}

// Dense bf16 (near-lossless top-16-bits store, dequant <<16 → f16). Decoupled + coupled.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_bf16_parity() {
    kvquant_attention_test(DType::F16, DType::Bf16, 1, 200, 128, 199, 5e-3, 350);
    kvquant_attention_test(DType::Bf16, DType::Bf16, 3, 6, 64, 3, 5e-3, 355);
}

// Dense f32: the native f32 attention path (no prepass) — coupled f32/f32 (the Metal clamp forbids
// a mixed f32/other request). Both backends run f32 SDPA, so a tight tolerance.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_f32_parity() {
    kvquant_attention_test(DType::F32, DType::F32, 1, 200, 128, 199, 1e-3, 360);
    kvquant_attention_test(DType::F32, DType::F32, 3, 6, 64, 3, 1e-3, 365);
}

// TurboQuant (WHT-rotated, 128-elem blocks = head_dim slices, so hd must be a multiple of 128).
// Coupled turbo/turbo at the vector shape + K=f16/V=turbo at the scalar shape. The inverse-WHT
// dequant plus f16-scratch storage widen the tolerance vs the block quants.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_turbo2_parity() {
    kvquant_attention_test(DType::Turbo2, DType::Turbo2, 1, 200, 128, 199, 1.2e-2, 370);
    kvquant_attention_test(DType::F16, DType::Turbo2, 3, 6, 128, 3, 1.2e-2, 375);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_turbo3_parity() {
    kvquant_attention_test(DType::Turbo3, DType::Turbo3, 1, 200, 128, 199, 1.2e-2, 380);
    kvquant_attention_test(DType::F16, DType::Turbo3, 3, 6, 128, 3, 1.2e-2, 385);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_turbo4_parity() {
    kvquant_attention_test(DType::Turbo4, DType::Turbo4, 1, 200, 128, 199, 1.2e-2, 390);
    kvquant_attention_test(DType::F16, DType::Turbo4, 3, 6, 128, 3, 1.2e-2, 395);
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

// Wide launch, short context (rows*n_head >= 128, kv_len < 128): routes to the lean unsplit
// kernel (`attention_*`), which the small-shape tests above never reach at this width.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_prefill_wide_parity() {
    let (rows, kv_len, nh, nkv, hd, pos) = (17usize, 24usize, 8usize, 2usize, 64usize, 7usize);
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
        (q, f32_bytes(&rand_f32(rows * nh * hd, 61))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 62))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 63))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

// Wide launch, long f16 context: routes to the half-fragment flash kernel (`attnflash_f16kv`).
// Q and P round to f16 in that path (accumulation stays f32), hence the wider tolerance than the
// exact-f32 attention kernels. kv_len is a multiple of 8 so the kernel's tail-block reads stay
// inside these exact-sized test buffers (the runtime cache is sized for the full context).
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_flash_matches_reference() {
    let (rows, kv_len, nh, nkv, hd, pos) = (17usize, 136usize, 8usize, 2usize, 64usize, 119usize);
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
        (q, f32_bytes(&rand_f32(rows * nh * hd, 71))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 72))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 73))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

// Same flash shape at hd = 72 (% 8 but not % 32): routes to the single-simdgroup flash kernel
// (`attnflash_f16kv`), which the hd % 32 == 0 tests above no longer reach (those take the
// cooperative `attnflash2_f16kv`).
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_flash_hd72_matches_reference() {
    let (rows, kv_len, nh, nkv, hd, pos) = (17usize, 136usize, 8usize, 2usize, 72usize, 119usize);
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
        (q, f32_bytes(&rand_f32(rows * nh * hd, 171))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 172))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 173))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

// The cooperative flash kernel (`attnflash2_f16kv`) at the real model head size (hd = 128), with
// a partial final query tile (rows = 17) and a KV length that lands mid-block (the kernel's
// causal-skip keeps tail reads within 7 rows of the limit).
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_flash2_hd128_matches_reference() {
    let (rows, kv_len, nh, nkv, hd, pos) = (17usize, 136usize, 8usize, 2usize, 128usize, 119usize);
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
        (q, f32_bytes(&rand_f32(rows * nh * hd, 181))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 182))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 183))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

// hd=256 (gemma): the cooperative flash instantiation with 8 O fragments per simdgroup.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_flash2_hd256_matches_reference() {
    let (rows, kv_len, nh, nkv, hd, pos) = (17usize, 136usize, 8usize, 2usize, 256usize, 119usize);
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
        mask: infr_core::graph::AttnMask::SlidingWindow(64),
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 401))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 402))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 403))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

// hd=256 decode (gemma): the NSG=16 vector flash instantiation, sliding window active (gemma's
// local layers decode with window clipping — the shape the sweep found on the split fallback).
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_vec_hd256_sliding_window_parity() {
    let (rows, kv_len, nh, nkv, hd, pos) = (1usize, 200usize, 4usize, 1usize, 256usize, 199usize);
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
        mask: infr_core::graph::AttnMask::SlidingWindow(96),
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 404))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 405))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 406))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

// Sliding-window masking through the cooperative flash kernel: the analytic per-row window
// lower bound must match the CPU reference (whole leading KV blocks fall below some rows'
// windows but not others').
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_flash2_sliding_window_parity() {
    let (rows, kv_len, nh, nkv, hd, pos) = (17usize, 136usize, 8usize, 2usize, 128usize, 119usize);
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
        mask: infr_core::graph::AttnMask::SlidingWindow(64),
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 191))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 192))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 193))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

// Long-context decode shape (rows=1, kv_len >= 128, hd=128): routes to the VECTOR flash kernel
// (`attnvec_f16kv_hd128`) — 32 simdgroups, 32 KV positions per simdgroup step, log2 merge. The
// kv_len=200 tail lands mid-block, exercising the clamped+masked tail rows.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_long_context_split32_parity() {
    let (rows, kv_len, nh, nkv, hd, pos) = (1usize, 200usize, 8usize, 2usize, 128usize, 199usize);
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
        (q, f32_bytes(&rand_f32(rows * nh * hd, 51))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 52))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 53))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

// The 32-way split-KV kernel (`attnsplit32_*`) retained for head sizes without a vec-kernel
// instantiation (hd=96 here): same long-context decode shape as above.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_long_context_split32_hd96_parity() {
    let (rows, kv_len, nh, nkv, hd, pos) = (1usize, 200usize, 8usize, 2usize, 96usize, 199usize);
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
        (q, f32_bytes(&rand_f32(rows * nh * hd, 201))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 202))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 203))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

// Sliding-window decode at depth through the vector flash kernel: whole leading KV blocks fall
// below the window (the kernel's block-skip), and the window edge lands mid-block.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_vec_sliding_window_parity() {
    let (rows, kv_len, nh, nkv, hd, pos, win) = (
        1usize, 300usize, 8usize, 2usize, 64usize, 299usize, 100usize,
    );
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
        (q, f32_bytes(&rand_f32(rows * nh * hd, 211))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 212))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 213))),
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
fn gatedactfused_parity() {
    let (rows, nff) = (3usize, 256usize);
    let mut g = Graph::new();
    let gu = g.input(TensorDesc::new(vec![rows, 2 * nff], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nff], DType::F32));
    g.push(Op::GatedActFused {
        gu,
        dst,
        rows: rows as u32,
        nff: nff as u32,
        act: infr_core::graph::Activation::Silu,
    });
    let bound = vec![(gu, f32_bytes(&rand_f32(rows * 2 * nff, 270)))];
    assert_parity(&g, &bound, dst, rows * nff, 1e-5);
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

// Quantized experts route to the DEVICE MoE path (router GEMV + on-device top-k + expert-table
// GEMVs); the f32 test above keeps the host fallback covered. CPU is the oracle here (its MoE
// matvec dequants the same bytes and dots in f32 — no Q8 activation quantization).
fn moe_quant_test(dtype: DType, synth: fn(usize, u32) -> Vec<u8>, seed: u32) {
    let (ne, n_expert, n_used, nff) = (256usize, 8usize, 3usize, 256usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![ne], DType::F32));
    let router = g.weight(TensorDesc::new(vec![n_expert, ne], DType::F32));
    let gate = g.weight(TensorDesc::new(vec![n_expert, nff, ne], dtype));
    let up = g.weight(TensorDesc::new(vec![n_expert, nff, ne], dtype));
    let down = g.weight(TensorDesc::new(vec![n_expert, ne, nff], dtype));
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
        (x, f32_bytes(&rand_f32(ne, seed as u64))),
        (router, f32_bytes(&rand_f32(n_expert * ne, seed as u64 + 1))),
        (gate, synth(n_expert * nff * ne, seed + 2)),
        (up, synth(n_expert * nff * ne, seed + 3)),
        (down, synth(n_expert * ne * nff, seed + 4)),
    ];
    assert_parity(&g, &bound, dst, ne, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn moe_ffn_q4k_device_parity() {
    moe_quant_test(DType::Q4K, synth_q4k, 80);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn moe_ffn_q6k_device_parity() {
    moe_quant_test(DType::Q6K, synth_q6k, 90);
}

// Batched rows through the device MoE path (rows spanning two 256-row chunks): every row routes
// independently; parity vs the CPU reference's per-row loop.
#[test]
#[ignore = "requires a Metal GPU"]
fn moe_ffn_batched_rows_parity() {
    let (rows, ne, n_expert, n_used, nff) = (300usize, 256usize, 8usize, 3usize, 256usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
    let router = g.weight(TensorDesc::new(vec![n_expert, ne], DType::F32));
    let gate = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
    let up = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
    let down = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q4K));
    let dst = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
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
    // x scaled down: the ~50x-real synthetic weights would push gate/up activations past f16
    // range (the kernels' operand precision) with unit-scale inputs — real hidden states don't.
    let xs_small: Vec<f32> = rand_f32(rows * ne, 95).iter().map(|v| v * 0.02).collect();
    let bound = vec![
        (x, f32_bytes(&xs_small)),
        (router, f32_bytes(&rand_f32(n_expert * ne, 96))),
        (gate, synth_q4k(n_expert * nff * ne, 97)),
        (up, synth_q4k(n_expert * nff * ne, 98)),
        (down, synth_q4k(n_expert * ne * nff, 99)),
    ];
    // Reference mirrors the grouped-GEMM path's numerics (same policy as the dense GEMM parity
    // tests): expert weights and stage inputs round to f16 (the kernels' operand precision, f32
    // accumulate), router/top-k stay f32. Residual tolerance covers reassociation over the
    // ~50x-real-magnitude synthetic weights' cancellation tail.
    let r16 =
        |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect() };
    let xs: Vec<f32> = {
        let (_, b) = &bound[0];
        bytemuck::cast_slice::<u8, f32>(b).to_vec()
    };
    let rw: Vec<f32> = {
        let (_, b) = &bound[1];
        bytemuck::cast_slice::<u8, f32>(b).to_vec()
    };
    use infr_gguf::dequant::dequant_block;
    let gw = r16(&dequant_block(DType::Q4K, &bound[2].1).unwrap());
    let uw = r16(&dequant_block(DType::Q4K, &bound[3].1).unwrap());
    let dw = r16(&dequant_block(DType::Q4K, &bound[4].1).unwrap());
    let mut reference = vec![0f32; rows * ne];
    for row in 0..rows {
        let x = &xs[row * ne..(row + 1) * ne];
        let logits: Vec<f32> = (0..n_expert)
            .map(|e| (0..ne).map(|i| rw[e * ne + i] * x[i]).sum::<f32>())
            .collect();
        let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
        let psum: f32 = probs.iter().sum();
        let mut idx: Vec<usize> = (0..n_expert).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        idx.truncate(n_used);
        let wsum: f32 = idx.iter().map(|&e| probs[e] / psum).sum::<f32>().max(1e-20);
        let x16 = r16(x);
        for &e in &idx {
            let gs = &gw[e * nff * ne..(e + 1) * nff * ne];
            let us = &uw[e * nff * ne..(e + 1) * nff * ne];
            let ds = &dw[e * ne * nff..(e + 1) * ne * nff];
            let gate: Vec<f32> = (0..nff)
                .map(|o| (0..ne).map(|i| gs[o * ne + i] * x16[i]).sum::<f32>())
                .collect();
            let up: Vec<f32> = (0..nff)
                .map(|o| (0..ne).map(|i| us[o * ne + i] * x16[i]).sum::<f32>())
                .collect();
            let act: Vec<f32> = (0..nff)
                .map(|i| {
                    let g = gate[i];
                    (g / (1.0 + (-g).exp())) * up[i]
                })
                .collect();
            let a16 = r16(&act);
            let w_e = (probs[e] / psum) / wsum;
            for o in 0..ne {
                let y: f32 = (0..nff).map(|i| ds[o * nff + i] * a16[i]).sum();
                reference[row * ne + o] += w_e * y;
            }
        }
    }
    let mtl = run(
        &MetalBackend::new().expect("metal backend"),
        &g,
        &bound,
        dst,
        rows * ne,
    );
    assert_close(&reference, &mtl, 5e-3, "moe batched grouped");
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
        rows: 1,
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
        rows: 1,
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

// Multi-row scan at the qwen3-next head shape: the state must carry across rows exactly (the
// device kernel loops rows with each lane owning its state column).
#[test]
#[ignore = "requires a Metal GPU"]
fn deltanet_multirow_parity() {
    let (rows, nv, nk, kd, vd) = (5usize, 4usize, 2usize, 128usize, 128usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nk * kd], DType::F32));
    let k = g.input(TensorDesc::new(vec![rows, nk * kd], DType::F32));
    let v = g.input(TensorDesc::new(vec![rows, nv * vd], DType::F32));
    let b = g.input(TensorDesc::new(vec![rows, nv], DType::F32));
    let a = g.input(TensorDesc::new(vec![rows, nv], DType::F32));
    let a_coef = g.weight(TensorDesc::new(vec![nv], DType::F32));
    let dt_bias = g.weight(TensorDesc::new(vec![nv], DType::F32));
    let state = g.input(TensorDesc::new(vec![nv * kd * vd], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nv * vd], DType::F32));
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
        rows: rows as u32,
        n_vhead: nv as u32,
        n_khead: nk as u32,
        head_k: kd as u32,
        head_v: vd as u32,
        eps: 1e-6,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nk * kd, 300))),
        (k, f32_bytes(&rand_f32(rows * nk * kd, 301))),
        (v, f32_bytes(&rand_f32(rows * nv * vd, 302))),
        (b, f32_bytes(&rand_f32(rows * nv, 303))),
        (a, f32_bytes(&rand_f32(rows * nv, 304))),
        (a_coef, f32_bytes(&rand_f32(nv, 305))),
        (dt_bias, f32_bytes(&rand_f32(nv, 306))),
        (state, f32_bytes(&rand_f32(nv * kd * vd, 307))),
    ];
    let reads = [(dst, rows * nv * vd), (state, nv * kd * vd)];
    let cpu = run_multi(&CpuBackend::new(), &g, &bound, &reads);
    let mtl = run_multi(&MetalBackend::new().expect("metal"), &g, &bound, &reads);
    assert_close(&cpu[0], &mtl[0], 1e-4, "deltanet multirow dst");
    assert_close(&cpu[1], &mtl[1], 1e-4, "deltanet multirow state");
}

// Multi-row conv: the rolling state shifts once per row and survives to the next.
#[test]
#[ignore = "requires a Metal GPU"]
fn conv1d_silu_multirow_parity() {
    let (rows, cc, kk) = (5usize, 256usize, 4usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, cc], DType::F32));
    let w = g.weight(TensorDesc::new(vec![cc, kk], DType::F32));
    let state = g.input(TensorDesc::new(vec![kk - 1, cc], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, cc], DType::F32));
    g.push(Op::Conv1dSilu {
        x,
        weight: w,
        state,
        dst,
        rows: rows as u32,
        channels: cc as u32,
        kernel: kk as u32,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * cc, 310))),
        (w, f32_bytes(&rand_f32(cc * kk, 311))),
        (state, f32_bytes(&rand_f32((kk - 1) * cc, 312))),
    ];
    let reads = [(dst, rows * cc), (state, (kk - 1) * cc)];
    let cpu = run_multi(&CpuBackend::new(), &g, &bound, &reads);
    let mtl = run_multi(&MetalBackend::new().expect("metal"), &g, &bound, &reads);
    assert_close(&cpu[0], &mtl[0], 1e-4, "conv multirow dst");
    assert_close(&cpu[1], &mtl[1], 0.0, "conv multirow state");
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
