//! Perf-only measurements of DeepSeek V4's compressed layers on the CPU backend; no parity claim
//! (the parity tests are `seam_op_parity.rs` and `synthetic_deepseek4_llama_cpp.rs`).
//!
//! [`dsv4_cpu_op_bench`] runs one decode row of each compressed tier's per-token op sequence at
//! the real model's geometry ([`V4_GEOMETRY`]) over random f16 caches, at several context lengths:
//! the CSA layer's lightning indexer, its top-k mask and its two-segment attention, and the HCA
//! layer's attention. [`dsv4_cpu_fixture_bench`] times the synthetic V4 file end to end, a few
//! hundred tokens of prefill, so that the small case is watched too, and prints a digest of the
//! logits, so that a change meant to keep every bit can be checked against the commit before it;
//! `INFR_UBATCH=1` there prefills one position per graph, for comparison with the batched default.
//! [`dsv4_cpu_moe_prefill_bench`] runs one V4-Flash routed-expert layer over a prefill chunk
//! against the same rows one at a time: what batched prefill saves where the real model spends
//! most of its bytes; it allocates real-size banks, so it runs only with `INFR_DSV4_MOE_BENCH`
//! set.
//!
//! ```text
//! [INFR_DSV4_MOE_BENCH=1] cargo test -p infr-llama --release --test dsv4_cpu_bench -- --ignored \
//!   --nocapture --test-threads=1
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

/// One routed-expert layer of V4-Flash as `Op::MoeFfn` sees it: the model width, the expert count,
/// the experts a token uses and an expert's width, with sqrt-softplus gating. The banks are Q2_K,
/// as in the shipped `Q2_K_S` file, with random codes and fixed block scales; the router is f32
/// (the file's is BF16).
const MOE_NE: usize = 4096;
const MOE_EXPERTS: usize = 256;
const MOE_USED: usize = 6;
const MOE_FF: usize = 2048;

/// The row counts the expert layer runs at: one decode row, and prefill chunks up to the default
/// prefill chunk height on a discrete GPU's host or a CPU-only one (`seam::ubatch_rows`'s
/// fallback; an integrated GPU's default is lower).
const MOE_ROWS: [usize; 6] = [1, 16, 64, 256, 512, 1024];

/// Set to run [`dsv4_cpu_moe_prefill_bench`]; without it the bench returns at once, so a run of
/// every ignored test (the GPU gate's `--run-ignored all`) does not allocate its banks.
const MOE_BENCH_ENV: &str = "INFR_DSV4_MOE_BENCH";

/// A Q2_K bank of `elems` values: every block's scale and code bytes from a multiplicative hash of
/// `salt`, its `d` and `dmin` fixed, so every value is finite.
fn q2k_bank(elems: usize, salt: u64) -> Vec<u8> {
    // ggml's `block_q2_K`: the scale and code bytes of its values, then `d` and `dmin` as f16.
    const VALUES: usize = 256;
    const CODES: usize = VALUES / 16 + VALUES / 4;
    const BYTES: usize = CODES + 4;
    let mut out = vec![0u8; elems / VALUES * BYTES];
    let d = half::f16::from_f32(0.01).to_le_bytes();
    let dmin = half::f16::from_f32(0.005).to_le_bytes();
    for (b, blk) in out.as_chunks_mut::<BYTES>().0.iter_mut().enumerate() {
        let (codes, scales) = blk.split_at_mut(CODES);
        for (i, w) in codes.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let x = ((b * CODES / 8 + i) as u64 ^ (salt << 48)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            *w = x.to_le_bytes();
        }
        scales[..2].copy_from_slice(&d);
        scales[2..].copy_from_slice(&dmin);
    }
    out
}

/// **What a prefill chunk saves on V4's routed experts**: one V4-Flash expert layer
/// ([`MOE_NE`] and the rest) executed over `rows` positions in one op, against `rows` one-row
/// executes, which is what a prefill of one position per graph pays. Each one-row execute reads the
/// banks of the [`MOE_USED`] experts its token routes to; a chunk reads each expert any of its rows
/// routes to once, for all of those rows. The banks stay in RAM between executes, so this measures
/// the arithmetic and memory traffic a chunk saves, not the file reads a streamed model saves. The
/// one-row baseline executes the same row again and again, so its experts stay warm in L3 between
/// repetitions, which no prefill of distinct tokens enjoys: the ratio it prints is conservative.
///
/// Three real-size banks: about 2.1 GB resident, plus one bank's bytes while each is generated.
/// Runs only with [`MOE_BENCH_ENV`] set.
#[test]
#[ignore = "perf measurement; run in release with --nocapture and INFR_DSV4_MOE_BENCH=1 on a quiet \
            machine"]
fn dsv4_cpu_moe_prefill_bench() {
    use infr_core::graph::{Activation, MoeGating};
    if std::env::var_os(MOE_BENCH_ENV).is_none() {
        eprintln!("{MOE_BENCH_ENV} is not set: skipping the expert-layer bench");
        return;
    }
    let be = infr_cpu::CpuBackend::new();
    let upload = |bytes: &[u8]| {
        let b = be.alloc(bytes.len(), BufferUsage::Weights).expect("alloc");
        be.upload(b.as_ref(), bytes).expect("upload");
        b
    };
    let bank = |salt| upload(&q2k_bank(MOE_EXPERTS * MOE_FF * MOE_NE, salt));
    let (gate, up, down) = (bank(1), bank(2), bank(3));
    let router = upload(&f32_bytes(&noise(MOE_EXPERTS * MOE_NE, 4)));
    println!(
        "dsv4 cpu moe prefill bench: {} threads, n_embd {MOE_NE}, {MOE_EXPERTS} experts, \
         {MOE_USED} used, {MOE_FF} wide, Q2_K banks; ms (median)",
        rayon::current_num_threads()
    );
    println!(
        "{:>5} {:>12} {:>14} {:>14} {:>8}",
        "rows", "chunk ms", "chunk ms/row", "1-row ms/row", "ratio"
    );
    let mut one_row_ms = 0.0;
    for rows in MOE_ROWS {
        let mut g = Graph::new();
        let x = g.input(f32d(rows * MOE_NE));
        let wr = g.weight(TensorDesc::new(vec![MOE_EXPERTS, MOE_NE], DType::F32));
        let wd = |g: &mut Graph, shape: Vec<usize>| g.weight(TensorDesc::new(shape, DType::Q2K));
        let wg = wd(&mut g, vec![MOE_EXPERTS, MOE_FF, MOE_NE]);
        let wu = wd(&mut g, vec![MOE_EXPERTS, MOE_FF, MOE_NE]);
        let wdn = wd(&mut g, vec![MOE_EXPERTS, MOE_NE, MOE_FF]);
        let dst = g.output(f32d(rows * MOE_NE));
        g.push(Op::MoeFfn {
            x,
            router_x: x,
            router: wr,
            gate_exps: wg,
            up_exps: wu,
            down_exps: wdn,
            down_scale: None,
            fused_gate_up: false,
            dst,
            ne: MOE_NE as u32,
            n_expert: MOE_EXPERTS as u32,
            n_used: MOE_USED as u32,
            n_ff_exp: MOE_FF as u32,
            scale: 1.0,
            act: Activation::Silu,
            gating: MoeGating::SqrtSoftplus,
            norm_w: true,
            weight_before: false,
            ep_band: None,
            exp_probs_b: None,
            n_expert_groups: 1,
            n_expert_groups_used: 0,
            swiglu_clamp: None,
            expert_ids: None,
        });
        let plan = be.compile(&g).expect("compile");
        let xb = upload(&f32_bytes(&noise(rows * MOE_NE, 5)));
        let ob = be
            .alloc(rows * MOE_NE * 4, BufferUsage::Readback)
            .expect("alloc");
        let mut b = Bindings::new();
        b.bind(x, xb.as_ref());
        b.bind(wr, router.as_ref());
        b.bind(wg, gate.as_ref());
        b.bind(wu, up.as_ref());
        b.bind(wdn, down.as_ref());
        b.bind(dst, ob.as_ref());
        let exec = || be.execute(plan.as_ref(), &b).expect("execute");
        exec();
        let reps = 5;
        let mut t: Vec<f64> = (0..reps)
            .map(|_| {
                let t0 = Instant::now();
                exec();
                t0.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        t.sort_by(f64::total_cmp);
        let ms = t[reps / 2];
        if rows == 1 {
            one_row_ms = ms;
        }
        println!(
            "{rows:>5} {ms:>12.2} {:>14.3} {one_row_ms:>14.3} {:>8.1}",
            ms / rows as f64,
            one_row_ms * rows as f64 / ms
        );
    }
}
