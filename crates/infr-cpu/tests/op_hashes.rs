//! Op-level output hashes: one checked-in value per case, the same at every SIMD tier and every
//! thread count, on every system.
//!
//! Each case is a small graph around one op (every `Op` variant has at least one, and
//! [`every_op_variant_has_a_case`] fails when a new variant arrives without one), built from
//! integer-generated inputs: weights from `infr_testkit::synth_weight`'s LCG payloads, activations
//! from integer formulas scaled by powers of two, so every input is the same bit pattern on any
//! machine. The case runs on a fresh CPU backend at each tier this CPU has (`kernels.cpu.max_tier`)
//! inside a pool of each of [`THREADS`] (plus this machine's parallelism), and the bits of every
//! tensor it writes — its outputs and the state it updates in place — are hashed (FNV-1a over the
//! little-endian bytes, every f32 NaN folded to `0x7fc00000` first) against ONE checked-in value.
//!
//! So a case passes only if its result does not depend on the SIMD tier, the thread count or the
//! way the pool chunks the work, and is the value Windows and Linux computed when it was blessed;
//! CI runs it on Linux, Windows and macOS (aarch64, the scalar tier only), which makes the macOS
//! run the architecture check. Print fresh values with `INFR_OP_HASH_BLESS=1 cargo test -p
//! infr-cpu --test op_hashes -- --nocapture`, and bless them only when every system prints the
//! same ones.
//!
//! The ceiling is process-wide, so the cases run one after another in one test.

use std::collections::BTreeSet;
use std::sync::Arc;

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::config::Config;
use infr_core::graph::{
    Activation, AttnMask, CompSegment, Graph, HyperGates, MoeGating, Op, TensorKind,
};
use infr_core::simd::{self, SimdTier};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_cpu::CpuBackend;

/// Pool sizes every case runs at: one thread, two, three, a prime, and more threads than any
/// case has rows or heads. This machine's parallelism is added to them.
const THREADS: [usize; 5] = [1, 2, 3, 7, 33];

// ─── inputs ────────────────────────────────────────────────────────────────────────────────────

/// Element `i` of input stream `seed`: a multiple of 2^-9 in about [-2, 2), from integers only.
fn val(seed: u64, i: usize) -> f32 {
    let h = (i as u64 + 1)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(seed.wrapping_mul(0xbf58_476d_1ce4_e5b9));
    let h = (h ^ (h >> 29)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((h >> 40) % 2048) as f32 / 512.0 - 2.0
}

fn vals(seed: u64, n: usize) -> Vec<f32> {
    (0..n).map(|i| val(seed, i)).collect()
}

/// [`val`] made inexact in every reduced-precision format: scaled by `1 + 2^-10 + 2^-20` and
/// offset by a second stream's `val / 4096`, so most elements need all 24 bits of an f32 and a
/// conversion to f16 or bf16 must round (a `val` alone, a multiple of 2^-9 in [-2, 2), fits f16
/// exactly and hides how the KV write rounds). Plain binary32 arithmetic on integer-generated
/// values: the same bits on every machine.
fn inexact_vals(seed: u64, n: usize) -> Vec<f32> {
    const SCALE: f32 = 1.0 + 1.0 / 1024.0 + 1.0 / 1_048_576.0;
    (0..n)
        .map(|i| val(seed, i) * SCALE + val(seed + 1000, i) / 4096.0)
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
        .collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// One case's graph, the bytes bound to its inputs and weights, and the tensors whose bits are
/// hashed after it ran (outputs, and inputs an op updates in place).
struct Case {
    name: String,
    g: Graph,
    bound: Vec<(TensorId, Vec<u8>)>,
    hashed: Vec<TensorId>,
    reference: bool,
}

impl Case {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            g: Graph::new(),
            bound: Vec::new(),
            hashed: Vec::new(),
            reference: false,
        }
    }

    fn input(&mut self, shape: Vec<usize>, dt: DType, bytes: Vec<u8>) -> TensorId {
        let id = self.g.input(TensorDesc::new(shape, dt));
        self.bound.push((id, bytes));
        id
    }

    fn input_f32(&mut self, shape: Vec<usize>, seed: u64) -> TensorId {
        let n = shape.iter().product();
        self.input(shape, DType::F32, f32_bytes(&vals(seed, n)))
    }

    fn input_i32(&mut self, v: &[i32]) -> TensorId {
        self.input(vec![v.len()], DType::I32, i32_bytes(v))
    }

    /// An input an op updates in place: bound, and hashed after the run.
    fn state(&mut self, shape: Vec<usize>, dt: DType, bytes: Vec<u8>) -> TensorId {
        let id = self.input(shape, dt, bytes);
        self.hashed.push(id);
        id
    }

    fn weight(&mut self, shape: Vec<usize>, dt: DType, bytes: Vec<u8>) -> TensorId {
        let id = self.g.weight(TensorDesc::new(shape, dt));
        self.bound.push((id, bytes));
        id
    }

    fn weight_f32(&mut self, shape: Vec<usize>, seed: u64) -> TensorId {
        let n = shape.iter().product();
        self.weight(shape, DType::F32, f32_bytes(&vals(seed, n)))
    }

    /// A `[rows, cols]` weight in `dt` (`synth_weight`'s payloads for a quantized format).
    fn weight_q(&mut self, dt: DType, rows: usize, cols: usize, seed: u32) -> TensorId {
        let bytes = match dt {
            DType::F32 => f32_bytes(&vals(seed as u64, rows * cols)),
            DType::F16 => f16_bytes(&vals(seed as u64, rows * cols)),
            DType::Bf16 => vals(seed as u64, rows * cols)
                .iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect(),
            _ => infr_testkit::synth_weight(dt, rows * cols, seed),
        };
        self.weight(vec![rows, cols], dt, bytes)
    }

    fn internal(&mut self, shape: Vec<usize>, dt: DType) -> TensorId {
        self.g.internal(TensorDesc::new(shape, dt))
    }

    fn output(&mut self, shape: Vec<usize>, dt: DType) -> TensorId {
        let id = self.g.output(TensorDesc::new(shape, dt));
        self.hashed.push(id);
        id
    }

    fn push(mut self, op: Op) -> Self {
        self.g.push(op);
        self
    }
}

// ─── running a case ────────────────────────────────────────────────────────────────────────────

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Run `case` on a fresh backend capped at `tier`, inside a pool of `threads`; return the hash
/// of every hashed tensor's bits, whether any of them held a non-zero value, and whether the
/// run built the backend's thread pool.
fn run(case: &Case, tier: SimdTier, threads: usize) -> (u64, bool, bool) {
    let mut cfg = Config::default();
    cfg.kernels.cpu.max_tier = Some(tier);
    cfg.kernels.cpu.reference = case.reference;
    let be = CpuBackend::new_with(Arc::new(cfg));
    assert_eq!(
        simd::active(),
        tier,
        "{}: the ceiling did not apply",
        case.name
    );
    let plan = be.compile(&case.g).expect("compile");
    let mut binds = Bindings::new();
    let mut bufs = Vec::new();
    for (id, bytes) in &case.bound {
        let usage = match case.g.tensors[id.0 as usize].kind {
            TensorKind::Weight => BufferUsage::Weights,
            _ => BufferUsage::Activations,
        };
        let b = be.alloc(bytes.len().max(4), usage).expect("alloc");
        be.upload(b.as_ref(), bytes).expect("upload");
        bufs.push((*id, b));
    }
    for &id in &case.hashed {
        if !bufs.iter().any(|(b, _)| *b == id) {
            let d = case.g.desc(id);
            let bytes = infr_gguf::nbytes(d.dtype, d.numel());
            bufs.push((
                id,
                be.alloc(bytes.max(4), BufferUsage::Readback)
                    .expect("alloc"),
            ));
        }
    }
    for (id, b) in &bufs {
        binds.bind(*id, b.as_ref());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");
    pool.install(|| be.execute(plan.as_ref(), &binds).expect("execute"));
    let pooled = be.pool_threads().is_some();
    if let Some(n) = be.pool_threads() {
        assert_eq!(
            n, threads,
            "{}: the backend's pool has another size",
            case.name
        );
    }
    let mut h = FNV_OFFSET;
    let mut nonzero = false;
    for &id in &case.hashed {
        let desc = case.g.desc(id);
        let (_, b) = bufs.iter().find(|(b, _)| *b == id).unwrap();
        let mut bytes = vec![0u8; infr_gguf::nbytes(desc.dtype, desc.numel())];
        be.download(b.as_ref(), &mut bytes).expect("download");
        if desc.dtype == DType::F32 {
            for c in bytes.as_chunks_mut::<4>().0 {
                if f32::from_le_bytes(*c).is_nan() {
                    c.copy_from_slice(&0x7fc0_0000u32.to_le_bytes());
                }
            }
        }
        nonzero |= bytes.iter().any(|&b| b != 0);
        h = fnv1a(h, &bytes);
    }
    (h, nonzero, pooled)
}

// ─── the cases ─────────────────────────────────────────────────────────────────────────────────

fn linear(dt: DType, m: usize, in_f: usize, out_f: usize, reference: bool) -> Case {
    let mode = if reference { " reference" } else { "" };
    let mut c = Case::new(format!("Linear {dt:?} m{m} {in_f}x{out_f}{mode}"));
    c.reference = reference;
    let x = c.input_f32(vec![m, in_f], 11);
    let w = c.weight_q(dt, out_f, in_f, 0x5eed ^ in_f as u32);
    let dst = c.output(vec![m, out_f], DType::F32);
    c.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    })
}

fn linear_cases() -> Vec<Case> {
    let mut v = Vec::new();
    let mut dts = vec![DType::F32, DType::F16, DType::Bf16];
    dts.extend(
        infr_core::decode_spec::WEIGHT_QUANTS
            .iter()
            .copied()
            .filter(|d| !matches!(d, DType::Turbo2 | DType::Turbo3 | DType::Turbo4)),
    );
    for dt in dts {
        for m in [1usize, 5, 19] {
            v.push(linear(dt, m, 512, 20, false));
        }
    }
    // The native-32-block Q8_0 path (a width that is not a whole super-block).
    for m in [1usize, 19] {
        v.push(linear(DType::Q8_0, m, 96, 20, false));
    }
    // Reference mode: f32 dots over the decoded weight.
    for dt in [DType::Q4K, DType::Q8_0] {
        v.push(linear(dt, 5, 512, 20, true));
    }
    v
}

fn norm_cases() -> Vec<Case> {
    let (rows, dim) = (5usize, 67usize);
    let mut v = Vec::new();

    let mut c = Case::new("RmsNorm");
    let x = c.input_f32(vec![rows, dim], 1);
    let w = c.weight_f32(vec![dim], 2);
    let dst = c.output(vec![rows, dim], DType::F32);
    v.push(c.push(Op::RmsNorm {
        x,
        weight: w,
        dst,
        rows: rows as u32,
        dim: dim as u32,
        eps: 1e-6,
    }));

    let mut c = Case::new("RmsNormAdd");
    let x = c.input_f32(vec![rows, dim], 3);
    let x2 = c.input_f32(vec![rows, dim], 4);
    let w = c.weight_f32(vec![dim], 5);
    let dst = c.output(vec![rows, dim], DType::F32);
    let c = c.push(Op::RmsNorm {
        x,
        weight: w,
        dst,
        rows: rows as u32,
        dim: dim as u32,
        eps: 1e-6,
    });
    v.push(c.push(Op::RmsNormAdd {
        x: x2,
        weight: w,
        dst,
        rows: rows as u32,
        dim: dim as u32,
        eps: 1e-5,
    }));

    let mut c = Case::new("LayerNorm");
    let x = c.input_f32(vec![rows, dim], 6);
    let w = c.weight_f32(vec![dim], 7);
    let b = c.weight_f32(vec![dim], 8);
    let dst = c.output(vec![rows, dim], DType::F32);
    v.push(c.push(Op::LayerNorm {
        x,
        weight: w,
        bias: b,
        dst,
        rows: rows as u32,
        dim: dim as u32,
        eps: 1e-5,
    }));

    for (name, sb) in [("Softmax", false), ("Softmax scale_buf", true)] {
        let mut c = Case::new(name);
        let x = c.input_f32(vec![3, 131], 9);
        let scale_buf = sb.then(|| c.input(vec![1], DType::F32, f32_bytes(&[0.375])));
        let dst = c.output(vec![3, 131], DType::F32);
        v.push(c.push(Op::Softmax {
            x,
            dst,
            rows: 3,
            dim: 131,
            scale: 0.75,
            scale_buf,
        }));
    }

    for (name, weighted) in [("QkNorm", true), ("QkNorm weightless", false)] {
        let mut c = Case::new(name);
        let x = c.input_f32(vec![3, 4 * 32], 10);
        let weight = weighted.then(|| c.weight_f32(vec![32], 11));
        let dst = c.output(vec![3, 4 * 32], DType::F32);
        v.push(c.push(Op::QkNorm {
            x,
            weight,
            dst,
            rows: 3,
            n_head: 4,
            head_dim: 32,
            eps: 1e-6,
            x_stride: 0,
        }));
    }

    let mut c = Case::new("GatedRmsNorm");
    let x = c.input_f32(vec![3, 4 * 32], 12);
    let gate = c.input_f32(vec![3, 4 * 32], 13);
    let w = c.weight_f32(vec![32], 14);
    let dst = c.output(vec![3, 4 * 32], DType::F32);
    v.push(c.push(Op::GatedRmsNorm {
        x,
        weight: w,
        gate,
        dst,
        rows: 3,
        n_head: 4,
        head_dim: 32,
        eps: 1e-6,
    }));
    v
}

fn rope_cases() -> Vec<Case> {
    let (rows, nh, hd) = (4usize, 3usize, 32usize);
    let positions = [0i32, 7, 300, 40961];
    let mut v = Vec::new();
    for (name, neox, backward, ff, rope_dim) in [
        ("Rope norm", false, false, false, hd),
        ("Rope neox partial", true, false, false, 24),
        ("Rope freq_factors backward", false, true, true, hd),
    ] {
        let mut c = Case::new(name);
        let x = c.input_f32(vec![rows, nh * hd], 20);
        let p = c.input_i32(&positions);
        let freq_factors = ff.then(|| {
            let f: Vec<f32> = (0..rope_dim / 2).map(|i| 1.0 + i as f32 / 4.0).collect();
            c.input(vec![rope_dim / 2], DType::F32, f32_bytes(&f))
        });
        let dst = c.output(vec![rows, nh * hd], DType::F32);
        v.push(c.push(Op::Rope {
            x,
            positions: p,
            dst,
            rows: rows as u32,
            n_head: nh as u32,
            head_dim: hd as u32,
            rope_dim: rope_dim as u32,
            theta: 1_000_000.0,
            freq_factors,
            x_stride: 0,
            neox,
            backward,
        }));
    }
    let mut c = Case::new("QkNormRope");
    let x = c.input_f32(vec![rows, nh * hd], 21);
    let w = c.weight_f32(vec![hd], 22);
    let p = c.input_i32(&positions);
    let dst = c.output(vec![rows, nh * hd], DType::F32);
    v.push(c.push(Op::QkNormRope {
        x,
        weight: w,
        positions: p,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: 24,
        theta: 10_000.0,
        eps: 1e-6,
        freq_factors: None,
        x_stride: 0,
    }));
    v
}

/// A KV cache of `cap` rows of `width` at `dt`, the first `filled` rows holding values.
fn kv_bytes(dt: DType, cap: usize, width: usize, filled: usize, seed: u64) -> Vec<u8> {
    let mut v = vals(seed, cap * width);
    v[filled * width..].fill(0.0);
    match dt {
        DType::F32 => f32_bytes(&v),
        DType::F16 => f16_bytes(&v),
        _ => {
            let mut out = vec![0u8; infr_gguf::nbytes(dt, cap * width)];
            infr_cpu::kvquant::quantize_row(dt, &v, &mut out);
            out
        }
    }
}

/// Bytes of a KV cache of `n` elements at `dt`.
fn kv_cache_len(dt: DType, n: usize) -> usize {
    match dt {
        DType::Turbo2 | DType::Turbo3 | DType::Turbo4 => {
            n / infr_cpu::turbo::QK * infr_cpu::turbo::block_bytes(dt)
        }
        _ => infr_gguf::nbytes(dt, n),
    }
}

/// Every KV format the CPU backend writes and reads: a prompt's K and V rows written into an
/// empty cache (`Op::WriteKv`, which quantizes them), then attended (`Op::Attention`, which
/// dequantizes the prefix). Row 0 of each source is zeros of both signs, row 1 one value
/// throughout (a block whose least and greatest values tie), so the quantizers' `min`/`max` meet
/// equal values; the cache bytes are hashed with the output.
fn kv_format_cases() -> Vec<Case> {
    let (rows, nh, nkv, hd, cap) = (5usize, 4usize, 2usize, 64usize, 8usize);
    let width = nkv * hd;
    // The float formats get values they cannot hold exactly, so the write's rounding shows; the
    // quantized ones round every value anyway.
    let src = |seed: u64, dt: DType| -> Vec<f32> {
        let body = if matches!(dt, DType::F16 | DType::Bf16) {
            inexact_vals(seed, rows * width)
        } else {
            vals(seed, rows * width)
        };
        (0..rows * width)
            .map(|i| match i / width {
                0 => {
                    if i % 3 == 0 {
                        -0.0
                    } else {
                        0.0
                    }
                }
                1 => 0.75,
                _ => body[i],
            })
            .collect()
    };
    let mut v = Vec::new();
    for dt in [
        DType::F32,
        DType::F16,
        DType::Bf16,
        DType::Q8_0,
        DType::Q4_0,
        DType::Q4_1,
        DType::Q5_0,
        DType::Q5_1,
        DType::Iq4Nl,
        DType::Turbo2,
        DType::Turbo3,
        DType::Turbo4,
    ] {
        let mut c = Case::new(format!("KV {dt:?} write and attend"));
        let ks = c.input(vec![rows, width], DType::F32, f32_bytes(&src(130, dt)));
        let vs = c.input(vec![rows, width], DType::F32, f32_bytes(&src(131, dt)));
        let bytes = vec![0u8; kv_cache_len(dt, cap * width)];
        let kc = c.state(vec![cap * width], dt, bytes.clone());
        let vc = c.state(vec![cap * width], dt, bytes);
        let q = c.input_f32(vec![rows, nh * hd], 132);
        let dst = c.output(vec![rows, nh * hd], DType::F32);
        let w = width as u32;
        let c = c
            .push(Op::WriteKv {
                src: ks,
                cache: kc,
                rows: rows as u32,
                row_stride: w,
                pos: 0,
            })
            .push(Op::WriteKv {
                src: vs,
                cache: vc,
                rows: rows as u32,
                row_stride: w,
                pos: 0,
            });
        v.push(c.push(Op::Attention {
            q,
            k_cache: kc,
            v_cache: vc,
            dst,
            rows: rows as u32,
            kv_len: rows as u32,
            n_head: nh as u32,
            n_kv: nkv as u32,
            head_dim: hd as u32,
            scale: 0.125,
            mask: AttnMask::Causal,
            pos: 0,
            sinks: None,
            key_bias: None,
            comp: None,
        }));
    }
    v
}

/// Shapes past the pool's size thresholds, where the work splits differently with the thread
/// count: a decode `Op::Linear` whose rows per task range over `DECODE_ROWS_MIN..=MAX`, a decode
/// attention whose f16 prefix widens in several `KV_WIDEN_CHUNK` pieces, and a lightning indexer
/// with several score tasks (`SCORE_TASK_MACS_MIN`).
fn threshold_cases() -> Vec<Case> {
    let mut v = vec![linear(DType::Q4K, 1, 512, 1024, false)];
    let (nkv, hd, kv_len, nh) = (2usize, 128usize, 160usize, 4usize);
    let mut c = Case::new("Attention decode f16, prefix in several widen chunks");
    let q = c.input_f32(vec![1, nh * hd], 140);
    let k = c.input(
        vec![kv_len * nkv * hd],
        DType::F16,
        kv_bytes(DType::F16, kv_len, nkv * hd, kv_len, 141),
    );
    let vv = c.input(
        vec![kv_len * nkv * hd],
        DType::F16,
        kv_bytes(DType::F16, kv_len, nkv * hd, kv_len, 142),
    );
    let dst = c.output(vec![1, nh * hd], DType::F32);
    v.push(c.push(Op::Attention {
        q,
        k_cache: k,
        v_cache: vv,
        dst,
        rows: 1,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 0.0884,
        mask: AttnMask::Causal,
        pos: (kv_len - 1) as u32,
        sinks: None,
        key_bias: None,
        comp: None,
    }));
    let (rows, kv_len, nh, hd, top_k, pos) = (8usize, 256usize, 4usize, 64usize, 16usize, 248usize);
    let mut c = Case::new("LightningIndexer, several score tasks");
    let q = c.input_f32(vec![rows, nh * hd], 143);
    let k = c.input_f32(vec![kv_len, hd], 144);
    let w = c.input_f32(vec![rows, nh], 145);
    let idx = c.output(vec![rows, top_k], DType::I32);
    v.push(c.push(Op::LightningIndexer {
        q,
        k_cache: k,
        weights: w,
        dst: idx,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        top_k: top_k as u32,
        scale: 1.0 / ((hd * nh) as f32).sqrt(),
        pos: pos as u32,
        block_ratio: 1,
    }));
    v
}

fn attention_cases() -> Vec<Case> {
    let mut v = Vec::new();
    let (nh, nkv, hd, cap) = (4usize, 2usize, 32usize, 64usize);
    for (name, dt, rows, pos, mask, extras) in [
        (
            "Attention causal f16",
            DType::F16,
            37usize,
            5usize,
            AttnMask::Causal,
            false,
        ),
        (
            "Attention decode f32",
            DType::F32,
            1,
            41,
            AttnMask::Causal,
            false,
        ),
        (
            "Attention window q4_0 kv",
            DType::Q4_0,
            9,
            20,
            AttnMask::SlidingWindow(6),
            false,
        ),
        (
            "Attention sinks key_bias",
            DType::F16,
            6,
            3,
            AttnMask::Causal,
            true,
        ),
    ] {
        let kv_len = pos + rows;
        let mut c = Case::new(name);
        let q = c.input_f32(vec![rows, nh * hd], 30);
        let k = c.input(
            vec![cap * nkv * hd],
            dt,
            kv_bytes(dt, cap, nkv * hd, kv_len, 31),
        );
        let vv = c.input(
            vec![cap * nkv * hd],
            dt,
            kv_bytes(dt, cap, nkv * hd, kv_len, 32),
        );
        let sinks = extras.then(|| c.input_f32(vec![nh], 33));
        let key_bias = extras.then(|| {
            let b: Vec<f32> = (0..rows * kv_len)
                .map(|i| {
                    if i % 5 == 2 {
                        f32::NEG_INFINITY
                    } else {
                        val(34, i)
                    }
                })
                .collect();
            c.input(vec![rows, kv_len], DType::F32, f32_bytes(&b))
        });
        let dst = c.output(vec![rows, nh * hd], DType::F32);
        v.push(c.push(Op::Attention {
            q,
            k_cache: k,
            v_cache: vv,
            dst,
            rows: rows as u32,
            kv_len: kv_len as u32,
            n_head: nh as u32,
            n_kv: nkv as u32,
            head_dim: hd as u32,
            scale: 1.0 / (hd as f32).sqrt(),
            mask,
            pos: pos as u32,
            sinks,
            key_bias,
            comp: None,
        }));
    }

    // DeepSeek V4's compressed segment after the raw keys.
    let (rows, pos, ratio, blocks) = (6usize, 10usize, 4usize, 4usize);
    let kv_len = pos + rows;
    let mut c = Case::new("Attention comp segment");
    let q = c.input_f32(vec![rows, nh * hd], 35);
    let k = c.input(
        vec![cap * hd],
        DType::F16,
        kv_bytes(DType::F16, cap, hd, kv_len, 36),
    );
    let comp_cache = c.input(
        vec![blocks * hd],
        DType::F16,
        kv_bytes(DType::F16, blocks, hd, blocks, 37),
    );
    let dst = c.output(vec![rows, nh * hd], DType::F32);
    v.push(c.push(Op::Attention {
        q,
        k_cache: k,
        v_cache: k,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: 1,
        head_dim: hd as u32,
        scale: 0.125,
        mask: AttnMask::Causal,
        pos: pos as u32,
        sinks: None,
        key_bias: None,
        comp: Some(CompSegment {
            cache: comp_cache,
            len: blocks as u32,
            ratio: ratio as u32,
            bias: None,
        }),
    }));

    // F16 on inexact values, at a width of 64 and at 61, where the 8-wide narrowing leaves a
    // tail in every row.
    for (dt, w, name) in [
        (DType::F16, 64usize, "WriteKv F16"),
        (DType::F16, 61, "WriteKv F16 width 61"),
        (DType::Q4_0, 64, "WriteKv Q4_0"),
    ] {
        let (rows, cap, pos) = (3usize, 16usize, 5usize);
        let mut c = Case::new(name);
        let x = if dt == DType::F16 {
            inexact_vals(38, rows * w)
        } else {
            vals(38, rows * w)
        };
        let src = c.input(vec![rows, w], DType::F32, f32_bytes(&x));
        let cache = c.state(vec![cap * w], dt, kv_bytes(dt, cap, w, pos, 39));
        v.push(c.push(Op::WriteKv {
            src,
            cache,
            rows: rows as u32,
            row_stride: w as u32,
            pos: pos as u32,
        }));
    }

    // Multi-head latent attention.
    let (rows, kv_len, nh, kvr, nope, rope, vhd) =
        (3usize, 8usize, 2usize, 16usize, 8usize, 4usize, 8usize);
    let qhd = nope + rope;
    let mut c = Case::new("Mla");
    let q = c.input_f32(vec![rows, nh * qhd], 40);
    let k = c.input(
        vec![16 * (kvr + rope)],
        DType::F16,
        kv_bytes(DType::F16, 16, kvr + rope, kv_len, 41),
    );
    let wk = c.weight_f32(vec![nh, kvr, nope], 42);
    let wv = c.weight_f32(vec![nh, kvr, vhd], 43);
    let dst = c.output(vec![rows, nh * vhd], DType::F32);
    v.push(c.push(Op::Mla {
        q,
        k_cache: k,
        wk_b: wk,
        wv_b: wv,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        q_head_dim: qhd as u32,
        kv_lora_rank: kvr as u32,
        qk_nope_dim: nope as u32,
        qk_rope_dim: rope as u32,
        v_head_dim: vhd as u32,
        scale: 0.25,
        mask: AttnMask::Causal,
        pos: (kv_len - rows) as u32,
        theta: 10_000.0,
        freq_factors: None,
        key_bias: None,
    }));

    // The lightning indexer, then its top-k expanded into a mask.
    let (rows, kv_len, nh, hd, top_k, pos) = (4usize, 10usize, 3usize, 8usize, 4usize, 6usize);
    let mut c = Case::new("LightningIndexer TopkMask");
    let q = c.input_f32(vec![rows, nh * hd], 44);
    let k = c.input_f32(vec![kv_len, hd], 45);
    let w = c.input_f32(vec![rows, nh], 46);
    let idx = c.output(vec![rows, top_k], DType::I32);
    let mask = c.output(vec![rows, kv_len], DType::F32);
    let c = c.push(Op::LightningIndexer {
        q,
        k_cache: k,
        weights: w,
        dst: idx,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        top_k: top_k as u32,
        scale: 1.0 / ((hd * nh) as f32).sqrt(),
        pos: pos as u32,
        block_ratio: 1,
    });
    v.push(c.push(Op::TopkMask {
        idx,
        dst: mask,
        rows: rows as u32,
        kv_len: kv_len as u32,
        top_k: top_k as u32,
    }));
    v
}

fn hyper_cases() -> Vec<Case> {
    let (rows, hc, ne) = (3usize, 4usize, 24usize);
    let mut v = Vec::new();
    for gated in [true, false] {
        let mut c = Case::new(if gated {
            "HyperConnectMix"
        } else {
            "HyperConnectMix head"
        });
        let width = if gated { (2 + hc) * hc } else { hc };
        let mixes = c.input_f32(vec![rows, width], 50);
        let scale = c.input_f32(vec![if gated { 3 } else { 1 }], 51);
        let base = c.input_f32(vec![width], 52);
        let pre = c.output(vec![rows, hc], DType::F32);
        let gates = gated.then(|| HyperGates {
            post: c.output(vec![rows, hc], DType::F32),
            comb: c.output(vec![rows, hc, hc], DType::F32),
        });
        v.push(c.push(Op::HyperConnectMix {
            mixes,
            scale,
            base,
            pre,
            gates,
            rows: rows as u32,
            hc: hc as u32,
            eps: 1e-6,
            n_iter: 20,
        }));
    }
    let mut c = Case::new("HyperConnectPre");
    let x = c.input_f32(vec![rows, hc, ne], 53);
    let w = c.input_f32(vec![rows, hc], 54);
    let dst = c.output(vec![rows, ne], DType::F32);
    v.push(c.push(Op::HyperConnectPre {
        x,
        weights: w,
        dst,
        rows: rows as u32,
        hc: hc as u32,
        n_embd: ne as u32,
    }));

    let mut c = Case::new("HyperConnectPost");
    let x = c.input_f32(vec![rows, ne], 55);
    let residual = c.input_f32(vec![rows, hc, ne], 56);
    let post = c.input_f32(vec![rows, hc], 57);
    let comb = c.input_f32(vec![rows, hc, hc], 58);
    let dst = c.output(vec![rows, hc, ne], DType::F32);
    v.push(c.push(Op::HyperConnectPost {
        x,
        residual,
        post,
        comb,
        dst,
        rows: rows as u32,
        hc: hc as u32,
        n_embd: ne as u32,
    }));

    let (blocks, window, ne) = (3usize, 4usize, 8usize);
    let mut c = Case::new("CompressPool");
    let values = c.input_f32(vec![blocks, window, ne], 59);
    let s: Vec<f32> = (0..blocks * window * ne)
        .map(|i| match i % 11 {
            3 => f32::NEG_INFINITY,
            _ if i < ne * window && i % ne == 1 => f32::NEG_INFINITY,
            _ => val(60, i),
        })
        .collect();
    let scores = c.input(vec![blocks, window, ne], DType::F32, f32_bytes(&s));
    let dst = c.output(vec![blocks, ne], DType::F32);
    v.push(c.push(Op::CompressPool {
        values,
        scores,
        dst,
        blocks: blocks as u32,
        window: window as u32,
        n_embd: ne as u32,
    }));

    let mut c = Case::new("GatherRows");
    let src = c.input_f32(vec![6, 2 * ne], 61);
    let idx = c.input_i32(&[3, -1, 0, 5, 2, -1]);
    let dst = c.output(vec![6, ne], DType::F32);
    let c = c.push(Op::GatherRows {
        src,
        idx,
        dst,
        rows: 3,
        idx_off: 0,
        src_stride: 2 * ne as u32,
        src_col: 0,
        n: ne as u32,
        group: 3,
        dst_group_stride: 6,
        dst_row_off: 0,
        fill: 0.0,
    });
    v.push(c.push(Op::GatherRows {
        src,
        idx,
        dst,
        rows: 3,
        idx_off: 3,
        src_stride: 2 * ne as u32,
        src_col: ne as u32,
        n: ne as u32,
        group: 3,
        dst_group_stride: 6,
        dst_row_off: 3,
        fill: f32::NEG_INFINITY,
    }));
    v
}

fn elementwise_cases() -> Vec<Case> {
    let (rows, n) = (3usize, 70usize);
    let mut v = Vec::new();
    for (act, clamp) in [
        (Activation::Silu, None),
        (Activation::Gelu, None),
        (Activation::Sigmoid, None),
        (Activation::Silu, Some(1.5f32)),
    ] {
        let mut c = Case::new(format!("GatedAct {act:?} clamp {clamp:?}"));
        let gate = c.input_f32(vec![rows, n], 70);
        let up = c.input_f32(vec![rows, n], 71);
        let dst = c.output(vec![rows, n], DType::F32);
        v.push(c.push(Op::GatedAct {
            gate,
            up,
            dst,
            rows: rows as u32,
            nff: n as u32,
            act,
            up_off: 0,
            up_stride: 0,
            gate_stride: 0,
            gate_block_width: 0,
            swiglu_clamp: clamp,
        }));
        let mut c = Case::new(format!("GatedActFused {act:?} clamp {clamp:?}"));
        let gu = c.input_f32(vec![rows, 2 * n], 72);
        let dst = c.output(vec![rows, n], DType::F32);
        v.push(c.push(Op::GatedActFused {
            gu,
            dst,
            rows: rows as u32,
            nff: n as u32,
            act,
            swiglu_clamp: clamp,
        }));
    }
    let mut c = Case::new("Add AddBias Scale MulVec Softcap");
    let a = c.input_f32(vec![rows, n], 73);
    let b = c.input_f32(vec![rows, n], 74);
    let bias = c.weight_f32(vec![n], 75);
    let mv = c.weight_f32(vec![n], 76);
    let t0 = c.internal(vec![rows, n], DType::F32);
    let t1 = c.internal(vec![rows, n], DType::F32);
    let t2 = c.internal(vec![rows, n], DType::F32);
    let t3 = c.internal(vec![rows, n], DType::F32);
    let dst = c.output(vec![rows, n], DType::F32);
    let nn = (rows * n) as u32;
    let c = c
        .push(Op::Add {
            a,
            b,
            dst: t0,
            n: nn,
        })
        .push(Op::AddBias {
            x: t0,
            bias,
            dst: t1,
            rows: rows as u32,
            n: n as u32,
        })
        .push(Op::Scale {
            x: t1,
            dst: t2,
            s: 0.3,
            n: nn,
        })
        .push(Op::MulVec {
            x: t2,
            vec: mv,
            dst: t3,
            rows: rows as u32,
            n: n as u32,
        });
    v.push(c.push(Op::Softcap {
        x: t3,
        dst,
        cap: 1.25,
        n: nn,
    }));

    let mut c = Case::new("HeadGate MoeSharedExpertAdd");
    let x = c.input_f32(vec![rows, 4 * 16], 77);
    let gate = c.input_f32(vec![rows, 4], 78);
    let moe = c.input_f32(vec![rows, 64], 79);
    let sgate = c.input_f32(vec![rows], 80);
    let t = c.internal(vec![rows, 64], DType::F32);
    let dst = c.output(vec![rows, 64], DType::F32);
    let c = c.push(Op::HeadGate {
        x,
        gate,
        dst: t,
        rows: rows as u32,
        n_head: 4,
        head_dim: 16,
    });
    v.push(c.push(Op::MoeSharedExpertAdd {
        moe,
        shexp: t,
        gate: sgate,
        dst,
        rows: rows as u32,
        n: 64,
    }));

    let vocab = 1000usize;
    let mut c = Case::new("Argmax ArgmaxProb Sample");
    let x = c.input_f32(vec![3, vocab], 81);
    let u = c.input(vec![1], DType::F32, f32_bytes(&[0.6875]));
    let am = c.output(vec![3], DType::F32);
    let id = c.output(vec![1], DType::F32);
    let prob = c.output(vec![1], DType::F32);
    let s = c.output(vec![1], DType::F32);
    let c = c
        .push(Op::Argmax {
            x,
            dst: am,
            n: vocab as u32,
            rows: 3,
        })
        .push(Op::ArgmaxProb {
            x,
            dst_id: id,
            dst_prob: prob,
            n: vocab as u32,
        });
    v.push(c.push(Op::Sample {
        x,
        u,
        dst: s,
        n: vocab as u32,
        top_k: 40,
        temp: 0.8,
        top_p: 0.9,
    }));

    for dt in [DType::F32, DType::Q4K, DType::Q8_0] {
        let ne = 256usize;
        let mut c = Case::new(format!("EmbedGather {dt:?}"));
        let ids = c.input_i32(&[5, 0, 9, 5]);
        let table = c.weight_q(dt, 10, ne, 90);
        let dst = c.output(vec![4, ne], DType::F32);
        v.push(c.push(Op::EmbedGather {
            ids,
            table,
            dst,
            rows: 4,
            ne: ne as u32,
            scale: 16.0,
        }));
    }

    let mut c = Case::new("GatherI32 Copy CopyStrided");
    let ids = c.input_i32(&[2, 0, 3]);
    let table: Vec<i32> = (0..4 * 6).map(|i| i * 7 - 50).collect();
    let table = c.weight(vec![4, 6], DType::I32, i32_bytes(&table));
    let gi = c.output(vec![3, 6], DType::I32);
    let src = c.input_f32(vec![5, 12], 91);
    let cp = c.output(vec![20], DType::F32);
    let cs = c.output(vec![5, 4], DType::F32);
    let c = c
        .push(Op::GatherI32 {
            ids,
            table,
            dst: gi,
            rows: 3,
            ne: 6,
        })
        .push(Op::Copy {
            src,
            src_off: 7,
            dst: cp,
            dst_off: 2,
            n: 15,
        });
    v.push(c.push(Op::CopyStrided {
        src,
        src_off: 3,
        src_stride: 12,
        dst: cs,
        dst_off: 0,
        dst_stride: 4,
        rows: 5,
        n: 4,
    }));
    v
}

fn recurrent_cases() -> Vec<Case> {
    let mut v = Vec::new();
    let (rows, ch, k) = (5usize, 48usize, 4usize);
    let mut c = Case::new("Conv1dSilu");
    let x = c.input_f32(vec![rows, ch], 100);
    let w = c.weight_f32(vec![ch, k], 101);
    let st = c.state(
        vec![k - 1, ch],
        DType::F32,
        f32_bytes(&vals(102, (k - 1) * ch)),
    );
    let dst = c.output(vec![rows, ch], DType::F32);
    v.push(c.push(Op::Conv1dSilu {
        x,
        weight: w,
        state: st,
        dst,
        rows: rows as u32,
        channels: ch as u32,
        kernel: k as u32,
    }));

    let mut c = Case::new("ShortConv");
    let bcx = c.input_f32(vec![rows, 3 * ch], 103);
    let w = c.weight_f32(vec![ch, k], 104);
    let st = c.state(
        vec![k - 1, ch],
        DType::F32,
        f32_bytes(&vals(105, (k - 1) * ch)),
    );
    let dst = c.output(vec![rows, ch], DType::F32);
    v.push(c.push(Op::ShortConv {
        bcx,
        weight: w,
        state: st,
        dst,
        rows: rows as u32,
        channels: ch as u32,
        kernel: k as u32,
    }));

    let (rows, nvh, nkh, hk, hv) = (4usize, 4usize, 2usize, 16usize, 8usize);
    let mut c = Case::new("DeltaNet");
    let q = c.input_f32(vec![rows, nkh * hk], 106);
    let kk = c.input_f32(vec![rows, nkh * hk], 107);
    let vv = c.input_f32(vec![rows, nvh * hv], 108);
    let b = c.input_f32(vec![rows, nvh], 109);
    let a = c.input_f32(vec![rows, nvh], 110);
    let a_coef = c.weight(
        vec![nvh],
        DType::F32,
        f32_bytes(&[-0.5, -1.25, -0.0625, -2.0]),
    );
    let dt_bias = c.weight_f32(vec![nvh], 111);
    let st = c.state(
        vec![nvh * hk * hv],
        DType::F32,
        f32_bytes(&vals(112, nvh * hk * hv)),
    );
    let dst = c.output(vec![rows, nvh * hv], DType::F32);
    v.push(c.push(Op::DeltaNet {
        q,
        k: kk,
        v: vv,
        b,
        a,
        a_coef,
        dt_bias,
        state: st,
        dst,
        rows: rows as u32,
        n_vhead: nvh as u32,
        n_khead: nkh as u32,
        head_k: hk as u32,
        head_v: hv as u32,
        eps: 1e-6,
        src_stride: 0,
    }));
    v
}

/// A MoE layer of `rows` tokens over `n_expert` experts of `dt`.
fn moe(name: &str, dt: DType, rows: usize, gating: MoeGating, variant: u8) -> Case {
    let (ne, n_expert, n_used, nff) = (256usize, 8usize, 3usize, 256usize);
    let mut c = Case::new(format!("MoeFfn {name} {dt:?} rows {rows}"));
    let x = c.input_f32(vec![rows, ne], 120);
    let router = c.weight_f32(vec![n_expert, ne], 121);
    let gate = c.weight_q(dt, n_expert * nff, ne, 122);
    let up = c.weight_q(dt, n_expert * nff, ne, 123);
    let down = c.weight_q(dt, n_expert * ne, nff, 124);
    let exp_probs_b = (variant == 1).then(|| c.weight_f32(vec![n_expert], 125));
    let expert_ids = (variant == 2).then(|| {
        let ids: Vec<i32> = (0..rows * n_used)
            .map(|i| ((i * 5 + 1) % n_expert) as i32)
            .collect();
        c.input(vec![rows, n_used], DType::I32, i32_bytes(&ids))
    });
    let dst = c.output(vec![rows, ne], DType::F32);
    c.push(Op::MoeFfn {
        x,
        router_x: x,
        router,
        gate_exps: gate,
        up_exps: up,
        down_exps: down,
        down_scale: None,
        fused_gate_up: false,
        dst,
        ne: ne as u32,
        n_expert: n_expert as u32,
        n_used: if variant == 3 { 1 } else { n_used as u32 },
        n_ff_exp: nff as u32,
        scale: 1.5,
        act: Activation::Silu,
        swiglu_clamp: None,
        gating,
        norm_w: variant != 3,
        weight_before: variant == 3,
        ep_band: None,
        exp_probs_b,
        n_expert_groups: if variant == 1 { 4 } else { 0 },
        n_expert_groups_used: if variant == 1 { 2 } else { 0 },
        expert_ids,
    })
}

fn moe_cases() -> Vec<Case> {
    let mut v = Vec::new();
    for dt in [DType::Q4K, DType::Q6K, DType::Q8_0, DType::Q5_0, DType::F16] {
        for rows in [1usize, 9] {
            v.push(moe("softmax", dt, rows, MoeGating::Softmax, 0));
        }
    }
    v.push(moe("grouped biased", DType::Q4K, 5, MoeGating::Sigmoid, 1));
    v.push(moe(
        "hash routed",
        DType::Q4K,
        5,
        MoeGating::SqrtSoftplus,
        2,
    ));
    v.push(moe("weight before", DType::Q4K, 5, MoeGating::Sigmoid, 3));
    v
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    v.extend(linear_cases());
    v.extend(norm_cases());
    v.extend(rope_cases());
    v.extend(attention_cases());
    v.extend(kv_format_cases());
    v.extend(threshold_cases());
    v.extend(hyper_cases());
    v.extend(elementwise_cases());
    v.extend(recurrent_cases());
    v.extend(moe_cases());
    v
}

/// Every `Op` variant's name. The match is exhaustive, so a new variant does not compile until
/// it is named here, and [`every_op_variant_has_a_case`] then fails until a case runs it.
fn op_name(op: &Op) -> &'static str {
    match op {
        Op::RmsNorm { .. } => "RmsNorm",
        Op::RmsNormAdd { .. } => "RmsNormAdd",
        Op::LayerNorm { .. } => "LayerNorm",
        Op::Linear { .. } => "Linear",
        Op::Softmax { .. } => "Softmax",
        Op::QkNorm { .. } => "QkNorm",
        Op::GatedRmsNorm { .. } => "GatedRmsNorm",
        Op::Rope { .. } => "Rope",
        Op::QkNormRope { .. } => "QkNormRope",
        Op::WriteKv { .. } => "WriteKv",
        Op::Attention { .. } => "Attention",
        Op::Mla { .. } => "Mla",
        Op::TopkMask { .. } => "TopkMask",
        Op::HyperConnectMix { .. } => "HyperConnectMix",
        Op::HyperConnectPre { .. } => "HyperConnectPre",
        Op::HyperConnectPost { .. } => "HyperConnectPost",
        Op::CompressPool { .. } => "CompressPool",
        Op::LightningIndexer { .. } => "LightningIndexer",
        Op::GatedAct { .. } => "GatedAct",
        Op::GatedActFused { .. } => "GatedActFused",
        Op::Add { .. } => "Add",
        Op::AddBias { .. } => "AddBias",
        Op::Scale { .. } => "Scale",
        Op::MulVec { .. } => "MulVec",
        Op::Softcap { .. } => "Softcap",
        Op::Argmax { .. } => "Argmax",
        Op::ArgmaxProb { .. } => "ArgmaxProb",
        Op::Sample { .. } => "Sample",
        Op::EmbedGather { .. } => "EmbedGather",
        Op::GatherI32 { .. } => "GatherI32",
        Op::GatherRows { .. } => "GatherRows",
        Op::Copy { .. } => "Copy",
        Op::CopyStrided { .. } => "CopyStrided",
        Op::MoeFfn { .. } => "MoeFfn",
        Op::Conv1dSilu { .. } => "Conv1dSilu",
        Op::ShortConv { .. } => "ShortConv",
        Op::HeadGate { .. } => "HeadGate",
        Op::DeltaNet { .. } => "DeltaNet",
        Op::MoeSharedExpertAdd { .. } => "MoeSharedExpertAdd",
    }
}

/// The number of `Op` variants [`op_name`] names (keep in step with it: a variant named there
/// but not counted here fails [`every_op_variant_has_a_case`]).
const OP_VARIANTS: usize = 39;

#[test]
fn every_op_variant_has_a_case() {
    let covered: BTreeSet<&str> = cases()
        .iter()
        .flat_map(|c| c.g.ops.iter().map(op_name).collect::<Vec<_>>())
        .collect();
    assert_eq!(
        covered.len(),
        OP_VARIANTS,
        "ops with a case: {covered:?}; every Op variant needs one"
    );
}

// ─── the expected values ───────────────────────────────────────────────────────────────────────

/// One hash per case, the same on every system (see the module doc for how to bless).
const EXPECTED: &[(&str, u64)] = &[
    ("Linear F32 m1 512x20", 0x9bc620ce57a46553),
    ("Linear F32 m5 512x20", 0xf1b6e36149436862),
    ("Linear F32 m19 512x20", 0x9b04ff36b51e4db3),
    ("Linear F16 m1 512x20", 0x9bc620ce57a46553),
    ("Linear F16 m5 512x20", 0xf1b6e36149436862),
    ("Linear F16 m19 512x20", 0x9b04ff36b51e4db3),
    ("Linear Bf16 m1 512x20", 0xa8bf39953ab73edc),
    ("Linear Bf16 m5 512x20", 0xe4f54f8117bc677f),
    ("Linear Bf16 m19 512x20", 0xcaca98e08bc93da8),
    ("Linear Q4_0 m1 512x20", 0xaa9f0a7d10821c9f),
    ("Linear Q4_0 m5 512x20", 0xa3c847d48987e746),
    ("Linear Q4_0 m19 512x20", 0x3fd2d25993b6dc59),
    ("Linear Q4_1 m1 512x20", 0x5e836b3f81dfa063),
    ("Linear Q4_1 m5 512x20", 0x5dc50dc116f70ea2),
    ("Linear Q4_1 m19 512x20", 0xf9367a4f0b44a639),
    ("Linear Q5_0 m1 512x20", 0x1d17eae010dbd8ef),
    ("Linear Q5_0 m5 512x20", 0x61963be281a50e90),
    ("Linear Q5_0 m19 512x20", 0xa450c643bc4dc7cf),
    ("Linear Q5_1 m1 512x20", 0x1312972c947eab02),
    ("Linear Q5_1 m5 512x20", 0x2e110ea6f98042cd),
    ("Linear Q5_1 m19 512x20", 0x66ea04da73e73eaf),
    ("Linear Q8_0 m1 512x20", 0x5852cad945b1f7be),
    ("Linear Q8_0 m5 512x20", 0xa97911fb71e0fec7),
    ("Linear Q8_0 m19 512x20", 0xf90cfcadfec6d8fb),
    ("Linear Q2K m1 512x20", 0x95f353407d1da31d),
    ("Linear Q2K m5 512x20", 0xf3c06cb571b53190),
    ("Linear Q2K m19 512x20", 0xd5e96513490969a9),
    ("Linear Q3K m1 512x20", 0xc7fcb54c71979266),
    ("Linear Q3K m5 512x20", 0xb0ec286e62073852),
    ("Linear Q3K m19 512x20", 0xcf439f7b63762188),
    ("Linear Q4K m1 512x20", 0xf18c010a28093ef5),
    ("Linear Q4K m5 512x20", 0x04b15bc606080d59),
    ("Linear Q4K m19 512x20", 0xbbf5ce68650ba354),
    ("Linear Q5K m1 512x20", 0x6f645eb5eefbb5e2),
    ("Linear Q5K m5 512x20", 0x425f81db401559bc),
    ("Linear Q5K m19 512x20", 0xf73288e2011a180a),
    ("Linear Q6K m1 512x20", 0x01060ffbc70a307c),
    ("Linear Q6K m5 512x20", 0xfc80821f5a47d6a6),
    ("Linear Q6K m19 512x20", 0x9cff473aa5b8c2c9),
    ("Linear Iq1S m1 512x20", 0x37cfa83bd91943d7),
    ("Linear Iq1S m5 512x20", 0xa1c31f3a735f8a3c),
    ("Linear Iq1S m19 512x20", 0x50f021f3aa7b2fb7),
    ("Linear Iq1M m1 512x20", 0xd201886b8658ef23),
    ("Linear Iq1M m5 512x20", 0x72b8777fcefb8e4f),
    ("Linear Iq1M m19 512x20", 0x6c73a1edb8b6edd0),
    ("Linear Iq2Xxs m1 512x20", 0x574451d3a3484add),
    ("Linear Iq2Xxs m5 512x20", 0x1c5037c8f20ed514),
    ("Linear Iq2Xxs m19 512x20", 0xa69a781a823c27db),
    ("Linear Iq2Xs m1 512x20", 0x8a9b174d7196c67f),
    ("Linear Iq2Xs m5 512x20", 0x742edbc64c18ed84),
    ("Linear Iq2Xs m19 512x20", 0x961a868748a1508f),
    ("Linear Iq2S m1 512x20", 0x0d18f7eaaf5bfe74),
    ("Linear Iq2S m5 512x20", 0x53378e603f96e54b),
    ("Linear Iq2S m19 512x20", 0x6df993dcf2030c53),
    ("Linear Iq3Xxs m1 512x20", 0xb76e5837282c39f7),
    ("Linear Iq3Xxs m5 512x20", 0x6ebbd89675dbf8ec),
    ("Linear Iq3Xxs m19 512x20", 0xa0c8ea53a427415b),
    ("Linear Iq3S m1 512x20", 0x0d437a9d8e25e57d),
    ("Linear Iq3S m5 512x20", 0x5b7f1131a5600cdc),
    ("Linear Iq3S m19 512x20", 0x9347d8f38b62107c),
    ("Linear Iq4Nl m1 512x20", 0xcb80e89281791958),
    ("Linear Iq4Nl m5 512x20", 0x4886cf06352cf5ce),
    ("Linear Iq4Nl m19 512x20", 0x301b59661990184c),
    ("Linear Iq4Xs m1 512x20", 0xadd7b5a51147745d),
    ("Linear Iq4Xs m5 512x20", 0xa534bcd0904a9890),
    ("Linear Iq4Xs m19 512x20", 0xd653f47cebfba801),
    ("Linear Tq1_0 m1 512x20", 0x00a608b358a4e8e5),
    ("Linear Tq1_0 m5 512x20", 0x8bc465f51234ed32),
    ("Linear Tq1_0 m19 512x20", 0x94008c290a176cf1),
    ("Linear Tq2_0 m1 512x20", 0x93e548e9678e9779),
    ("Linear Tq2_0 m5 512x20", 0xf4ee3e6ae1636620),
    ("Linear Tq2_0 m19 512x20", 0x6545134e48179edd),
    ("Linear Q2_0 m1 512x20", 0x8a1d67ced568fda6),
    ("Linear Q2_0 m5 512x20", 0xbe2e8108aec4edef),
    ("Linear Q2_0 m19 512x20", 0xdc71b213ec2ea86f),
    ("Linear Mxfp4 m1 512x20", 0x9873f84287a3a8da),
    ("Linear Mxfp4 m5 512x20", 0x8d1c47e5ba21b683),
    ("Linear Mxfp4 m19 512x20", 0x72dffb5f3abf5dd1),
    ("Linear Nvfp4 m1 512x20", 0x6115a1b7df8013a2),
    ("Linear Nvfp4 m5 512x20", 0x235bdd8141f9659b),
    ("Linear Nvfp4 m19 512x20", 0x484aa5514624f005),
    ("Linear Q8_0 m1 96x20", 0xe0899134ede65230),
    ("Linear Q8_0 m19 96x20", 0xdd970b1827d0ea00),
    ("Linear Q4K m5 512x20 reference", 0x9b5f3f511bad7f45),
    ("Linear Q8_0 m5 512x20 reference", 0x5d2958f17c07a5f1),
    ("RmsNorm", 0x27121e3fae28e214),
    ("RmsNormAdd", 0x9bca04ac6e03aa27),
    ("LayerNorm", 0xefe36c726570d899),
    ("Softmax", 0xea3fd54682dce4c2),
    ("Softmax scale_buf", 0xaa76483c15872190),
    ("QkNorm", 0xd257c6a2316b8ea6),
    ("QkNorm weightless", 0x4aefce10428cf55d),
    ("GatedRmsNorm", 0xee6286db5a123b76),
    ("Rope norm", 0x2f416992ea273de9),
    ("Rope neox partial", 0xdbe59d464b5c63cb),
    ("Rope freq_factors backward", 0x7535a997b2dbaaa8),
    ("QkNormRope", 0x40608ad480df62e1),
    ("Attention causal f16", 0x2ddf4eac99b6bdf8),
    ("Attention decode f32", 0xd8e66c5574bc0e1d),
    ("Attention window q4_0 kv", 0x573a8ad55fac6f70),
    ("Attention sinks key_bias", 0x99db266e7d4a5969),
    ("Attention comp segment", 0xd06093c9b9126d06),
    ("WriteKv F16", 0x9251f37837f52d3f),
    ("WriteKv F16 width 61", 0xdc0f7d67cec92a2d),
    ("WriteKv Q4_0", 0x4f536590c43efedc),
    ("Mla", 0x555b78feb583f984),
    ("LightningIndexer TopkMask", 0x0ec80ae09928fb74),
    ("KV F32 write and attend", 0x094d8912d2db3f8d),
    ("KV F16 write and attend", 0xea88366b6189a0a9),
    ("KV Bf16 write and attend", 0x880d37e4101d5a1b),
    ("KV Q8_0 write and attend", 0x93ad98142cd777ee),
    ("KV Q4_0 write and attend", 0xc65665a1b22b10fa),
    ("KV Q4_1 write and attend", 0x3af6531a77ec0857),
    ("KV Q5_0 write and attend", 0x74825214e6d58771),
    ("KV Q5_1 write and attend", 0x20053a0dd59b5f6c),
    ("KV Iq4Nl write and attend", 0x358daa018bdbe1c5),
    ("KV Turbo2 write and attend", 0xacec32ac5e7e7c93),
    ("KV Turbo3 write and attend", 0xc3c637601e8e40d1),
    ("KV Turbo4 write and attend", 0xd50dc492f7158232),
    ("Linear Q4K m1 512x1024", 0x6574d4b7dcc95f0a),
    (
        "Attention decode f16, prefix in several widen chunks",
        0x86f55bcb60383105,
    ),
    ("LightningIndexer, several score tasks", 0x187bcf7e94afe7b1),
    ("HyperConnectMix", 0xffbe691ade250168),
    ("HyperConnectMix head", 0xebc31bdf6121f4a3),
    ("HyperConnectPre", 0x408fe1ed03183ef2),
    ("HyperConnectPost", 0x80b00d97e9ab8a99),
    ("CompressPool", 0x02de0a1248a5cb3b),
    ("GatherRows", 0xd292f25f2bdda8ee),
    ("GatedAct Silu clamp None", 0x67bb2a472166fa61),
    ("GatedActFused Silu clamp None", 0x8d679af4a925e4f1),
    ("GatedAct Gelu clamp None", 0xbd3f026033bd8826),
    ("GatedActFused Gelu clamp None", 0x537a5b2f6d653d31),
    ("GatedAct Sigmoid clamp None", 0xa752fbcd42b5ece6),
    ("GatedActFused Sigmoid clamp None", 0x69a6198ddd0fea5a),
    ("GatedAct Silu clamp Some(1.5)", 0xef63310dc5ea2c83),
    ("GatedActFused Silu clamp Some(1.5)", 0x50676ff33695f14a),
    ("Add AddBias Scale MulVec Softcap", 0xdf94bc9ec1d50176),
    ("HeadGate MoeSharedExpertAdd", 0xe7739c400283a04b),
    ("Argmax ArgmaxProb Sample", 0xd1d986129922ec54),
    ("EmbedGather F32", 0x19fc8dbe6d6a31f0),
    ("EmbedGather Q4K", 0x92c4163a7ea78169),
    ("EmbedGather Q8_0", 0xa84e5c4eb092a6b9),
    ("GatherI32 Copy CopyStrided", 0xd1399384859a0a91),
    ("Conv1dSilu", 0x0352fc2a7484d21a),
    ("ShortConv", 0x67a6bd345ba46fa4),
    ("DeltaNet", 0x2f0a40475d4b4085),
    ("MoeFfn softmax Q4K rows 1", 0x1d2a2edb4078ed10),
    ("MoeFfn softmax Q4K rows 9", 0x7dbc267c9f2ef97c),
    ("MoeFfn softmax Q6K rows 1", 0xfd869ffe8d226e93),
    ("MoeFfn softmax Q6K rows 9", 0x0804efa239fe3838),
    ("MoeFfn softmax Q8_0 rows 1", 0xa2ea513858997419),
    ("MoeFfn softmax Q8_0 rows 9", 0xf18bdb7f12b96ab3),
    ("MoeFfn softmax Q5_0 rows 1", 0xa6222d3ef4e12f54),
    ("MoeFfn softmax Q5_0 rows 9", 0xf274033123784d70),
    ("MoeFfn softmax F16 rows 1", 0x7ada78c4c40cc59d),
    ("MoeFfn softmax F16 rows 9", 0x975c267fd160643e),
    ("MoeFfn grouped biased Q4K rows 5", 0x10423bc2d8e420c5),
    ("MoeFfn hash routed Q4K rows 5", 0x187b3a67fc5c9f74),
    ("MoeFfn weight before Q4K rows 5", 0x900aebc15c5f130a),
];

/// Every case gives its one checked-in hash at every tier this CPU has and at every thread
/// count. Turns red when a tier, a thread count or the pool's chunking changes a bit, or when
/// a case's value moved.
#[test]
fn every_op_gives_its_one_hash_at_every_tier_and_thread_count() {
    let bless = std::env::var_os("INFR_OP_HASH_BLESS").is_some();
    let tiers: Vec<SimdTier> = SimdTier::ALL
        .into_iter()
        .filter(|t| t.supported())
        .collect();
    let lacking: Vec<&str> = SimdTier::ALL
        .into_iter()
        .filter(|t| !t.supported())
        .map(|t| t.name())
        .collect();
    let mut threads = THREADS.to_vec();
    let par = std::thread::available_parallelism().map_or(1, |n| n.get());
    if !threads.contains(&par) {
        threads.push(par);
    }
    let cases = cases();
    let mut names = BTreeSet::new();
    let mut failures = Vec::new();
    let mut table = String::new();
    let mut pooled_cases = 0;
    for case in &cases {
        assert!(
            names.insert(case.name.clone()),
            "two cases named {}",
            case.name
        );
        let mut seen: Option<u64> = None;
        let mut pooled = false;
        for &t in &tiers {
            for &n in &threads {
                let (h, nonzero, p) = run(case, t, n);
                pooled |= p;
                assert!(
                    nonzero,
                    "{}: every hashed output is zero, the case is vacuous",
                    case.name
                );
                match seen {
                    None => seen = Some(h),
                    Some(s) if s != h => failures.push(format!(
                        "{}: {:#018x} at {} with {n} threads, {s:#018x} at the first run",
                        case.name,
                        h,
                        t.name()
                    )),
                    Some(_) => {}
                }
            }
        }
        pooled_cases += pooled as usize;
        let h = seen.unwrap();
        table.push_str(&format!("    ({:?}, {h:#018x}),\n", case.name));
        if !bless {
            match EXPECTED.iter().find(|(n, _)| *n == case.name) {
                Some(&(_, want)) if want == h => {}
                Some(&(_, want)) => {
                    failures.push(format!("{}: {h:#018x}, expected {want:#018x}", case.name))
                }
                None => failures.push(format!("{}: {h:#018x}, no expected value", case.name)),
            }
        }
    }
    simd::set_max(None);
    // The thread counts reached the ops: most cases ran on a pool of the requested size.
    assert!(
        pooled_cases * 2 > cases.len(),
        "only {pooled_cases} cases built a pool"
    );
    if bless {
        println!("tiers {tiers:?}, threads {threads:?}; lacking {lacking:?}\n{table}");
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        EXPECTED.iter().all(|(n, _)| names.contains(*n)),
        "an expected value names no case"
    );
}
