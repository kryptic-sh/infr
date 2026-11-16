//! Perf-only measurements of DeepSeek V4's compressed layers on the CPU backend; no parity claim
//! (the parity tests are `seam_op_parity.rs` and `synthetic_deepseek4_llama_cpp.rs`).
//!
//! [`dsv4_cpu_op_bench`] runs one decode row of each compressed tier's per-token op sequence at
//! the real model's geometry ([`V4_GEOMETRY`]) over random f16 caches, at several context lengths:
//! the CSA layer's lightning indexer, its top-k mask and its two-segment attention, and the HCA
//! layer's attention. [`dsv4_cpu_fixture_bench`] times the synthetic V4 file end to end, a few
//! hundred tokens of prefill, so that the small case is watched too, and prints a digest of the
//! logits, so that a change meant to keep every bit can be checked against the commit before it.
//!
//! ```text
//! cargo test -p infr-llama --release --test dsv4_cpu_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Nothing else should be running: every op here spreads over the whole CPU pool.

mod dsv4_fixture;

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{AttnMask, CompSegment, Graph, Op};
use infr_core::tensor::TensorDesc;
use infr_core::{DType, TensorId};
use std::time::Instant;

/// The dimensions of one V4-Flash compressed layer that the ops below see.
struct Geometry {
    n_head: usize,
    head_dim: usize,
    /// The raw sliding window, which is also the raw ring's row count.
    window: usize,
    csa_ratio: usize,
    hca_ratio: usize,
    indexer_heads: usize,
    indexer_width: usize,
    top_k: usize,
}

const V4_GEOMETRY: Geometry = Geometry {
    n_head: 64,
    head_dim: 512,
    window: 128,
    csa_ratio: 4,
    hca_ratio: 128,
    indexer_heads: 64,
    indexer_width: 128,
    top_k: 512,
};

/// Context lengths the decode row sits at.
const CONTEXTS: [usize; 4] = [2048, 8192, 32768, 65536];

fn f32d(n: usize) -> TensorDesc {
    TensorDesc::new(vec![n], DType::F32)
}

/// Values in `[-1, 1)` from a multiplicative hash, so no two rows of a cache repeat each other and
/// the indexer's scores do not tie.
fn noise(n: usize, salt: u64) -> Vec<f32> {
    (0..n as u64)
        .map(|i| {
            let x = (i ^ (salt << 40)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            ((x >> 40) as f32) / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

/// A compiled graph with every binding uploaded once, executed repeatedly.
struct Run {
    plan: Box<dyn infr_core::backend::Plan>,
    bufs: Vec<(TensorId, Box<dyn Buffer>)>,
    out: (TensorId, Box<dyn Buffer>, usize),
}

impl Run {
    fn new(
        be: &dyn Backend,
        g: &Graph,
        binds: Vec<(TensorId, Vec<u8>, BufferUsage)>,
        out: TensorId,
        out_bytes: usize,
    ) -> Self {
        let plan = be.compile(g).expect("compile");
        let bufs = binds
            .into_iter()
            .map(|(id, bytes, usage)| {
                let b = be.alloc(bytes.len(), usage).expect("alloc");
                be.upload(b.as_ref(), &bytes).expect("upload");
                (id, b)
            })
            .collect();
        let ob = be.alloc(out_bytes, BufferUsage::Readback).expect("alloc");
        Run {
            plan,
            bufs,
            out: (out, ob, out_bytes),
        }
    }

    fn exec(&self, be: &dyn Backend) {
        let mut b = Bindings::new();
        for (id, buf) in &self.bufs {
            b.bind(*id, buf.as_ref());
        }
        b.bind(self.out.0, self.out.1.as_ref());
        be.execute(self.plan.as_ref(), &b).expect("execute");
    }

    fn download(&self, be: &dyn Backend) -> Vec<u8> {
        let mut o = vec![0u8; self.out.2];
        be.download(self.out.1.as_ref(), &mut o).expect("download");
        o
    }

    /// Median wall time of one execute in ms, after warm-up runs.
    fn median_ms(&self, be: &dyn Backend, reps: usize) -> f64 {
        for _ in 0..2 {
            self.exec(be);
        }
        let mut t: Vec<f64> = (0..reps)
            .map(|_| {
                let t0 = Instant::now();
                self.exec(be);
                t0.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        t.sort_by(f64::total_cmp);
        t[reps / 2]
    }
}

/// The CSA layer's indexer over a `len`-row LID cache for one decode row at `pos`, optionally
/// followed by its top-k mask; the output is the indices, or the mask.
fn csa_indexer(be: &dyn Backend, len: usize, pos: usize, with_mask: bool) -> Run {
    let geo = &V4_GEOMETRY;
    let (nh, w, top_k) = (geo.indexer_heads, geo.indexer_width, geo.top_k.min(len));
    let mut g = Graph::new();
    let q = g.input(f32d(nh * w));
    let k_cache = g.input(TensorDesc::new(vec![len * w], DType::F16));
    let weights = g.input(f32d(nh));
    let idx_desc = TensorDesc::new(vec![top_k], DType::I32);
    let (idx, out, out_bytes) = if with_mask {
        let idx = g.internal(idx_desc);
        let mask = g.output(f32d(len));
        (idx, mask, len * 4)
    } else {
        let idx = g.output(idx_desc);
        (idx, idx, top_k * 4)
    };
    g.push(Op::LightningIndexer {
        q,
        k_cache,
        weights,
        dst: idx,
        rows: 1,
        kv_len: len as u32,
        n_head: nh as u32,
        head_dim: w as u32,
        top_k: top_k as u32,
        scale: 1.0 / ((w * nh) as f32).sqrt(),
        pos: pos as u32,
        block_ratio: geo.csa_ratio as u32,
    });
    if with_mask {
        g.push(Op::TopkMask {
            idx,
            dst: out,
            rows: 1,
            kv_len: len as u32,
            top_k: top_k as u32,
        });
    }
    let binds = vec![
        (q, f32_bytes(&noise(nh * w, 1)), BufferUsage::Activations),
        (k_cache, f16_bytes(&noise(len * w, 2)), BufferUsage::KvCache),
        (weights, f32_bytes(&noise(nh, 3)), BufferUsage::Activations),
    ];
    Run::new(be, &g, binds, out, out_bytes)
}

/// One decode row of a compressed layer's attention at `pos`: the raw ring (K = V, as on V4),
/// the sinks, and a `len`-row compressed segment at `ratio` with an optional score bias.
fn comp_attention(
    be: &dyn Backend,
    pos: usize,
    len: usize,
    ratio: usize,
    bias: Option<Vec<f32>>,
) -> Run {
    let geo = &V4_GEOMETRY;
    let (nh, hd, win) = (geo.n_head, geo.head_dim, geo.window);
    let mut g = Graph::new();
    let q = g.input(f32d(nh * hd));
    let kc = g.input(TensorDesc::new(vec![win * hd], DType::F16));
    let cache = g.input(TensorDesc::new(vec![len * hd], DType::F16));
    let sinks = g.weight(f32d(nh));
    let mut binds = vec![
        (q, f32_bytes(&noise(nh * hd, 4)), BufferUsage::Activations),
        (kc, f16_bytes(&noise(win * hd, 5)), BufferUsage::KvCache),
        (cache, f16_bytes(&noise(len * hd, 6)), BufferUsage::KvCache),
        (sinks, f32_bytes(&noise(nh, 7)), BufferUsage::Weights),
    ];
    let bias = bias.map(|b| {
        let id = g.input(f32d(len));
        binds.push((id, f32_bytes(&b), BufferUsage::Activations));
        id
    });
    let dst = g.output(f32d(nh * hd));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: kc,
        dst,
        rows: 1,
        kv_len: (pos + 1) as u32,
        n_head: nh as u32,
        n_kv: 1,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: AttnMask::SlidingWindow(win),
        pos: pos as u32,
        sinks: Some(sinks),
        key_bias: None,
        comp: Some(CompSegment {
            cache,
            len: len as u32,
            ratio: ratio as u32,
            bias,
        }),
    });
    Run::new(be, &g, binds, dst, nh * hd * 4)
}

#[test]
#[ignore = "perf measurement; run in release with --nocapture on a quiet machine"]
fn dsv4_cpu_op_bench() {
    let be = infr_cpu::CpuBackend::new();
    let geo = &V4_GEOMETRY;
    println!(
        "dsv4 cpu op bench: {} threads, one decode row, ms per op (median)",
        rayon::current_num_threads()
    );
    println!(
        "{:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>6} {:>10}",
        "ctx", "csa", "indexer", "topkmask", "csa attn", "csa total", "hca", "hca attn"
    );
    for ctx in CONTEXTS {
        let pos = ctx - 1;
        let len = ctx / geo.csa_ratio;
        let reps = if len >= 8192 { 7 } else { 15 };
        let ix = csa_indexer(&be, len, pos, false);
        let ix_ms = ix.median_ms(&be, reps);
        let chain = csa_indexer(&be, len, pos, true);
        let chain_ms = chain.median_ms(&be, reps);
        let mask: Vec<f32> = bytemuck::cast_slice(&chain.download(&be)).to_vec();
        assert_eq!(
            mask.iter().filter(|m| m.is_finite()).count(),
            geo.top_k.min(len),
            "the mask keeps top_k blocks"
        );
        let csa = comp_attention(&be, pos, len, geo.csa_ratio, Some(mask));
        let csa_ms = csa.median_ms(&be, reps);
        let hca_len = ctx / geo.hca_ratio;
        let hca = comp_attention(&be, pos, hca_len, geo.hca_ratio, None);
        let hca_ms = hca.median_ms(&be, reps);
        println!(
            "{ctx:>6} {len:>6} {ix_ms:>10.3} {:>10.3} {csa_ms:>10.3} {:>10.3} {hca_len:>6} \
             {hca_ms:>10.3}",
            chain_ms - ix_ms,
            chain_ms + csa_ms,
        );
    }
}

#[test]
#[ignore = "perf measurement; run in release with --nocapture on a quiet machine"]
fn dsv4_cpu_fixture_bench() {
    let dims = dsv4_fixture::dsv4_dims();
    let tmp =
        infr_testkit::gguf::TempGguf::write("dsv4-cpu-bench", &dsv4_fixture::dsv4_model(&dims));
    let model = infr_llama::SeamModel::load_with(
        tmp.path(),
        None,
        // From the environment, so that `INFR_PROF_OPS=1` profiles the run per op.
        std::sync::Arc::new(infr_llama::EngineConfig::load_from_env()),
    )
    .expect("synthetic model load");
    let vocab = model.config().vocab as u32;
    println!(
        "dsv4 cpu fixture bench: {} threads, ratios {:?}, indexer top_k {}",
        rayon::current_num_threads(),
        dims.compress_ratios,
        dims.indexer.top_k
    );
    for n in [64usize, 256, 512] {
        let tokens: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % vocab).collect();
        let logits = model.prefill_logits_cpu(&tokens).expect("warm-up");
        let digest = infr_testkit::gguf::fnv1a64(bytemuck::cast_slice(&logits));
        let mut t: Vec<f64> = (0..3)
            .map(|_| {
                let t0 = Instant::now();
                model.prefill_logits_cpu(&tokens).expect("cpu prefill");
                t0.elapsed().as_secs_f64()
            })
            .collect();
        t.sort_by(f64::total_cmp);
        println!(
            "{n:>4} tokens: {:.1} ms, {:.0} tokens/s (median of 3), logits digest {digest:016x}",
            t[1] * 1e3,
            n as f64 / t[1]
        );
    }
}
