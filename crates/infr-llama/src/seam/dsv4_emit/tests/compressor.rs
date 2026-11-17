//! One compressed layer's ops alone, on the CPU interpreter, against a host computation of their
//! definition: each compressor's committed rows (`build_hca_compressed_kv_from_state`,
//! `build_overlap_compressed_kv_from_state`) and the CSA tier's top-k selection
//! (`build_lid_top_k`, `build_top_k_mask`), fed token by token and in graphs of several tokens.

use super::super::{Dsv4LayerIn, Dsv4Rope, Dsv4Tier};
use super::{max_diff, rand_vals, read_f16, read_f32, upload};
use crate::seam::dsv4_plan::{DSV4_CSA_RATIO, DSV4_HCA_RATIO};
use crate::seam::weights::{Dsv4CompressorW, Dsv4IndexerW};
use crate::Config;
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_cpu::CpuBackend;

/// The stand-alone layer's dimensions: a small head with a real `[nope | rope]` split, and an
/// indexer whose head width differs from the attention head's, so a compressor run at the wrong
/// width, or an LID row read as a CSA row, has nowhere to hide.
const NE: usize = 12;
const W: usize = 16;
const WI: usize = 24;
const RD: usize = 8;
const QL: usize = 10;
const IH: usize = 3;
/// The indexer's `top_k`: small, so the selection drops blocks from position `4 * TOPK + 3` on.
const TOPK: usize = 3;
const THETA: f32 = 40000.0;
const EPS: f32 = 1e-5;
/// The context the buffers are sized for.
const CTX: usize = 384;

const HCA: usize = DSV4_HCA_RATIO as usize;
const CSA: usize = DSV4_CSA_RATIO as usize;

/// One compressor's four tensors, as host values, for a compressor `width` wide at `ratio`.
struct CompW {
    width: usize,
    ratio: usize,
    overlap: bool,
    kv: Vec<f32>,
    gate: Vec<f32>,
    ape: Vec<f32>,
    norm: Vec<f32>,
}

impl CompW {
    fn new(width: usize, ratio: usize, seed: u64) -> Self {
        let overlap = ratio == CSA;
        let sw = if overlap { 2 * width } else { width };
        CompW {
            width,
            ratio,
            overlap,
            kv: rand_vals(sw * NE, seed, 0.5),
            gate: rand_vals(sw * NE, seed + 1, 0.5),
            ape: rand_vals(ratio * sw, seed + 2, 0.5),
            norm: rand_vals(width, seed + 3, 0.25)
                .iter()
                .map(|v| 1.0 + v)
                .collect(),
        }
    }

    fn host(&self) -> [&[f32]; 4] {
        [&self.kv, &self.gate, &self.ape, &self.norm]
    }
}

/// One compressed layer's weights: its compressors (HCA: one; CSA: its own, then the LID one),
/// the CSA tier's indexer projections, and the compressed-layer rope divisors.
struct LayerW {
    comps: Vec<CompW>,
    /// `indexer.proj` `[IH, NE]` and `indexer.attn_q_b` `[IH * WI, QL]`, row-major by output.
    proj: Vec<f32>,
    q_b: Vec<f32>,
    ff: Vec<f32>,
}

impl LayerW {
    fn hca() -> Self {
        LayerW {
            comps: vec![CompW::new(W, HCA, 1)],
            proj: Vec::new(),
            q_b: Vec::new(),
            ff: divisors(),
        }
    }

    fn csa() -> Self {
        LayerW {
            comps: vec![CompW::new(W, CSA, 11), CompW::new(WI, CSA, 21)],
            proj: rand_vals(IH * NE, 31, 0.5),
            q_b: rand_vals(IH * WI * QL, 32, 0.5),
            ff: divisors(),
        }
    }

    fn ratio(&self) -> usize {
        self.comps[0].ratio
    }
}

/// Uneven per-pair divisors, so a rope without them is a different rope.
fn divisors() -> Vec<f32> {
    (0..RD / 2).map(|p| 1.0 + p as f32).collect()
}

fn cfg(ratio: usize) -> Config {
    Config {
        n_layer: 1,
        n_embd: NE,
        head_dim: W,
        rope_dim: RD,
        q_lora_rank: QL,
        indexer_n_head: IH,
        indexer_head_size: WI,
        indexer_top_k: TOPK,
        deepseek4: true,
        compress_ratios: vec![ratio],
        ..Default::default()
    }
}

/// The layer's session-held buffers, in `crate::seam::dsv4_comp_bufs` order.
fn layer_bufs(be: &CpuBackend, ratio: usize) -> Vec<Box<dyn Buffer>> {
    crate::seam::dsv4_comp_bufs(&cfg(ratio), 0, CTX, DType::F16)
        .iter()
        .map(|b| be.alloc(b.bytes(), BufferUsage::KvCache).expect("alloc"))
        .collect()
}

/// What one graph of a run selected: its first position, its rows, the blocks its indexer scored,
/// and its `[rows, len]` top-k bias.
struct Selection {
    start: usize,
    rows: usize,
    len: usize,
    bias: Vec<f32>,
}

/// Feed `x` (`[n, NE]`) and `qr` (`[n, QL]`) through one compressed layer's emit in graphs of
/// `chunks` tokens, in order, on `bufs`; `after` sees the buffers after each graph, with the
/// number of positions run so far. Returns every graph's top-k bias (CSA graphs where some row
/// sees a block).
fn run_layer(
    be: &CpuBackend,
    w: &LayerW,
    x: &[f32],
    qr: &[f32],
    chunks: &[usize],
    bufs: &[Box<dyn Buffer>],
    mut after: impl FnMut(usize, &[Box<dyn Buffer>]),
) -> Vec<Selection> {
    let ratio = w.ratio();
    let c = cfg(ratio);
    let geometry = crate::seam::dsv4_comp_bufs(&c, 0, CTX, DType::F16);
    let mut host: Vec<&[f32]> = w.comps.iter().flat_map(|cw| cw.host()).collect();
    if ratio == CSA {
        host.extend([&w.proj[..], &w.q_b[..]]);
    }
    let wb: Vec<Box<dyn Buffer>> = host
        .iter()
        .map(|v| upload(be, v, BufferUsage::Weights))
        .collect();
    let ffb = upload(be, &w.ff, BufferUsage::Staging);
    let mut selections = Vec::new();
    let mut start = 0;
    for &t in chunks {
        let mut g = Graph::new();
        let xin = g.input(TensorDesc::new(vec![t * NE], DType::F32));
        let qrin = g.input(TensorDesc::new(vec![t * QL], DType::F32));
        let positions = g.input(TensorDesc::new(vec![t], DType::I32));
        let ff = g.input(TensorDesc::new(vec![RD / 2], DType::F32));
        let ins: Vec<TensorId> = geometry
            .iter()
            .map(|b| g.input(TensorDesc::new(vec![b.elems], b.dtype)))
            .collect();
        let wid: Vec<TensorId> = host
            .iter()
            .map(|v| g.weight(TensorDesc::new(vec![v.len()], DType::F32)))
            .collect();
        let comp = |k: usize| Dsv4CompressorW {
            kv: wid[4 * k],
            gate: wid[4 * k + 1],
            ape: wid[4 * k + 2],
            norm: wid[4 * k + 3],
        };
        let mut step = Vec::new();
        let tier = Dsv4Tier::new(&mut g, &c, start, t, ratio as u32, &geometry, &mut step);
        let rope = Dsv4Rope {
            head_dim: W as u32,
            rope_dim: RD as u32,
            theta: THETA,
            freq_factors: Some(ff),
        };
        let seg = if ratio == HCA {
            tier.emit_hca(&mut g, &comp(0), xin, NE as u32, &ins, rope, EPS)
        } else {
            let ix = Dsv4IndexerW {
                proj: wid[8],
                attn_q_b: wid[9],
                comp: comp(1),
            };
            let io = Dsv4LayerIn {
                x: xin,
                qr: qrin,
                positions,
            };
            tier.emit_csa(&mut g, &c, &comp(0), &ix, io, &ins, rope, EPS)
        };
        let out = seg
            .and_then(|s| s.bias.map(|b| (b, s.len)))
            .map(|(b, len)| {
                let out = g.output(TensorDesc::new(vec![t * len as usize], DType::F32));
                g.push(Op::Copy {
                    src: b,
                    src_off: 0,
                    dst: out,
                    dst_off: 0,
                    n: t as u32 * len,
                });
                (out, len as usize)
            });
        let plan = be.compile(&g).expect("compile");
        let pos: Vec<i32> = (start..start + t).map(|p| p as i32).collect();
        let xb = upload(be, &x[start * NE..(start + t) * NE], BufferUsage::Staging);
        let qrb = upload(be, &qr[start * QL..(start + t) * QL], BufferUsage::Staging);
        let pb = upload(be, &pos, BufferUsage::Staging);
        let step_bufs: Vec<Box<dyn Buffer>> = step
            .iter()
            .map(|(_, v)| upload(be, v, BufferUsage::Staging))
            .collect();
        let ob = out.map(|(o, len)| {
            (
                o,
                len,
                be.alloc(t * len * 4, BufferUsage::Activations)
                    .expect("alloc"),
            )
        });
        let mut b = Bindings::new();
        b.bind(xin, xb.as_ref())
            .bind(qrin, qrb.as_ref())
            .bind(positions, pb.as_ref())
            .bind(ff, ffb.as_ref());
        for (id, buf) in ins.iter().zip(bufs) {
            b.bind(*id, buf.as_ref());
        }
        for (id, buf) in wid.iter().zip(&wb) {
            b.bind(*id, buf.as_ref());
        }
        for ((id, _), buf) in step.iter().zip(&step_bufs) {
            b.bind(*id, buf.as_ref());
        }
        if let Some((o, _, buf)) = &ob {
            b.bind(*o, buf.as_ref());
        }
        be.execute(plan.as_ref(), &b).expect("execute");
        if let Some((_, len, buf)) = &ob {
            selections.push(Selection {
                start,
                rows: t,
                len: *len,
                bias: read_f32(be, buf.as_ref()),
            });
        }
        start += t;
        after(start, bufs);
    }
    selections
}

/// `m · v` in f64 for a row-major `[out, inn]` matrix.
fn matvec(m: &[f32], v: &[f32], out: usize) -> Vec<f64> {
    let inn = v.len();
    (0..out)
        .map(|o| {
            (0..inn)
                .map(|i| f64::from(m[o * inn + i]) * f64::from(v[i]))
                .sum()
        })
        .collect()
}

/// The NORM rope of `row`'s last [`RD`] dims at `pos`, each pair's angle divided by its divisor.
fn rope_tail(row: &mut [f64], pos: usize, ff: &[f32]) {
    let nope = row.len() - RD;
    for (p, &d) in ff.iter().enumerate() {
        let ang = pos as f64 * f64::from(THETA).powf(-2.0 * p as f64 / RD as f64) / f64::from(d);
        let (s, c) = ang.sin_cos();
        let (a, b) = (row[nope + 2 * p], row[nope + 2 * p + 1]);
        row[nope + 2 * p] = a * c - b * s;
        row[nope + 2 * p + 1] = a * s + b * c;
    }
}

/// Block `blk`'s compressed row from its definition. Every position `p` has a state row
/// `kv·x[p]` and a score row `gate·x[p] + ape[p % ratio]`. The block's window is its own `ratio`
/// positions; the overlapping compressor puts its previous `ratio` positions first, reading the
/// FIRST half of their state rows (a position below 0 has value 0 and score `-inf`), and its own
/// positions' SECOND halves after. Per channel, the softmax of the window's scores weights its
/// values; then the RMSNorm with its gain; then the rope of the row's tail at the block's FIRST
/// position.
fn block_row(cw: &CompW, ff: &[f32], x: &[f32], blk: usize) -> Vec<f64> {
    let (wd, r) = (cw.width, cw.ratio);
    let sw = if cw.overlap { 2 * wd } else { wd };
    let state = |p: usize| {
        let xp = &x[p * NE..(p + 1) * NE];
        let mut sc = matvec(&cw.gate, xp, sw);
        for (s, a) in sc.iter_mut().zip(&cw.ape[(p % r) * sw..]) {
            *s += f64::from(*a);
        }
        (matvec(&cw.kv, xp, sw), sc)
    };
    // (values, scores) per window slot, `wd` channels each.
    let mut slots: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    if cw.overlap {
        for i in 0..r {
            match (blk * r + i).checked_sub(r) {
                Some(p) => {
                    let (kv, sc) = state(p);
                    slots.push((kv[..wd].to_vec(), sc[..wd].to_vec()));
                }
                None => slots.push((vec![0.0; wd], vec![f64::NEG_INFINITY; wd])),
            }
        }
    }
    for i in 0..r {
        let (kv, sc) = state(blk * r + i);
        let off = if cw.overlap { wd } else { 0 };
        slots.push((kv[off..off + wd].to_vec(), sc[off..off + wd].to_vec()));
    }
    let mut row: Vec<f64> = (0..wd)
        .map(|ch| {
            let m = slots
                .iter()
                .map(|s| s.1[ch])
                .fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = slots.iter().map(|s| (s.1[ch] - m).exp()).collect();
            let den: f64 = e.iter().sum();
            e.iter().zip(&slots).map(|(e, s)| e * s.0[ch]).sum::<f64>() / den
        })
        .collect();
    let ms = row.iter().map(|v| v * v).sum::<f64>() / wd as f64;
    let s = 1.0 / (ms + f64::from(EPS)).sqrt();
    for (v, g) in row.iter_mut().zip(&cw.norm) {
        *v *= s * f64::from(*g);
    }
    rope_tail(&mut row, blk * r, ff);
    row
}

/// Assert every block a run of `n` positions completes sits in compressor `k`'s cache at its
/// definition ([`block_row`]).
fn assert_rows(
    be: &CpuBackend,
    w: &LayerW,
    x: &[f32],
    bufs: &[Box<dyn Buffer>],
    n: usize,
    run: &str,
) {
    for (k, cw) in w.comps.iter().enumerate() {
        let cache = read_f16(be, bufs[3 * k].as_ref());
        for blk in 0..n / cw.ratio {
            let want = block_row(cw, &w.ff, x, blk);
            let got = &cache[blk * cw.width..(blk + 1) * cw.width];
            for (ch, (g, w)) in got.iter().zip(&want).enumerate() {
                let g = f64::from(*g);
                assert!(
                    (g - w).abs() <= 1e-3 * w.abs() + 1e-4,
                    "{run}: compressor {k} block {blk} channel {ch} holds {g}, its definition \
                     gives {w}"
                );
            }
        }
    }
}

/// Assert two runs left the same rings, up to the rounding of a different graph split.
fn assert_same_rings(be: &CpuBackend, a: &[Box<dyn Buffer>], b: &[Box<dyn Buffer>], ratio: usize) {
    let geometry = crate::seam::dsv4_comp_bufs(&cfg(ratio), 0, CTX, DType::F16);
    for (i, _) in geometry.iter().enumerate().filter(|(_, g)| g.ring) {
        let d = max_diff(&read_f32(be, a[i].as_ref()), &read_f32(be, b[i].as_ref()));
        assert!(
            d <= 1e-5,
            "ring buffer {i}: token by token and in larger graphs differ by {d}"
        );
    }
}

/// **The HCA compressor writes, for every completed block, the row its definition gives, and
/// does so whatever graphs the positions arrive in.** 300 positions token by token and again in
/// three graphs that each end mid-block (100, 50 and 150 tokens): the second and third read the
/// ring for positions earlier graphs wrote, and the third completes block 1 while its own rows are
/// about to be persisted over the ring rows that block reads. Both runs must leave the two
/// committed rows at [`block_row`] (block 1 ropes at position 128, so a row roped anywhere else
/// misses) and must agree with each other on the ring.
#[test]
fn hca_compressor_rows_match_their_definition_in_any_graphs() {
    let n = 300;
    let be = CpuBackend::new();
    let w = LayerW::hca();
    let x = rand_vals(n * NE, 5, 1.0);
    let qr = vec![0.0; n * QL];
    let one = layer_bufs(&be, HCA);
    run_layer(&be, &w, &x, &qr, &vec![1; n], &one, |_, _| {});
    let chunked = layer_bufs(&be, HCA);
    run_layer(&be, &w, &x, &qr, &[100, 50, 150], &chunked, |_, _| {});
    assert_rows(&be, &w, &x, &one, n, "token by token");
    assert_rows(&be, &w, &x, &chunked, n, "in three graphs");
    assert_same_rings(&be, &one, &chunked, HCA);
}

/// The CSA stand-alone run's length: thirty blocks.
const CSA_POS: usize = 120;

/// The CSA tier's graph splits: the first ends mid-block 2, so the second starts with block 2's
/// first half in the ring and its previous window (block 1) wholly there; the second ends
/// mid-block 8; the third completes the rest, many blocks per graph.
const CSA_CHUNKS: [usize; 3] = [10, 23, 87];

/// **Both CSA-layer compressors write, for every completed block, the row their definition
/// gives, in any graphs, and at the step that completes it.** The CSA compressor at `head_dim`
/// and the LID one at the indexer's width, 120 positions token by token and in three graphs that
/// end mid-block ([`CSA_CHUNKS`]). Every committed row of both caches must sit at [`block_row`]:
/// row 0, whose previous window is all before the sequence start (value 0, score `-inf`); row 1,
/// whose previous window is block 0 read through the FIRST half of its state rows while its own
/// positions give their SECOND half, so a swap of the windows or of the halves misses; and every
/// later one, some read wholly from the ring. Token by token, row 0 of each cache must still be
/// untouched after position 2 and written after position 3, the step that completes block 0. The
/// two runs must leave the same rings.
#[test]
fn csa_compressor_rows_match_their_definition_in_any_graphs() {
    let be = CpuBackend::new();
    let w = LayerW::csa();
    let x = rand_vals(CSA_POS * NE, 6, 1.0);
    let qr = rand_vals(CSA_POS * QL, 7, 1.0);
    let one = layer_bufs(&be, CSA);
    let mut checked = 0;
    run_layer(&be, &w, &x, &qr, &[1; CSA_POS], &one, |n, bufs| {
        if n > CSA {
            return;
        }
        for (k, cw) in w.comps.iter().enumerate() {
            let row0 = &read_f16(&be, bufs[3 * k].as_ref())[..cw.width];
            if n < CSA {
                assert!(
                    row0.iter().all(|&v| v == 0.0),
                    "compressor {k} wrote block 0 after only {n} positions"
                );
            } else {
                assert!(
                    row0.iter().any(|&v| v != 0.0),
                    "compressor {k} had not written block 0 after the position that completes it"
                );
            }
        }
        checked += 1;
    });
    assert_eq!(
        checked, CSA,
        "the commit timing was checked at every early step"
    );
    let chunked = layer_bufs(&be, CSA);
    run_layer(&be, &w, &x, &qr, &CSA_CHUNKS, &chunked, |_, _| {});
    assert_rows(&be, &w, &x, &one, CSA_POS, "token by token");
    assert_rows(&be, &w, &x, &chunked, CSA_POS, "in three graphs");
    assert_same_rings(&be, &one, &chunked, CSA);
}

/// The selection the indexer's definition makes for the query row at position `pos`, against the
/// LID cache's committed rows (`lid`, as stored), over the `len` blocks its graph scores; `None`
/// when two eligible blocks score too close to the boundary for an f32 run to be held to the f64
/// order. A block `j` is eligible once completed, `j < (pos + 1) / 4`; it scores
/// `Σ_h w_h / sqrt(WI * IH) · relu(q_h · lid[j])`, where `q = attn_q_b · qr` roped on each head's
/// tail at `pos` and `w = proj · x`; the top `min(len, TOPK)` are kept by (eligible first, score
/// descending, index ascending), and the bias row is `0` on them and `-inf` elsewhere.
fn expected_bias(
    w: &LayerW,
    x: &[f32],
    qr: &[f32],
    lid: &[f32],
    pos: usize,
    len: usize,
) -> Option<Vec<f32>> {
    let mut q = matvec(&w.q_b, &qr[pos * QL..(pos + 1) * QL], IH * WI);
    for head in q.chunks_mut(WI) {
        rope_tail(head, pos, &w.ff);
    }
    let wts = matvec(&w.proj, &x[pos * NE..(pos + 1) * NE], IH);
    let scale = 1.0 / ((WI * IH) as f64).sqrt();
    let visible = ((pos + 1) / CSA).min(len);
    let mut scored: Vec<(f64, usize)> = (0..visible)
        .map(|j| {
            let k = &lid[j * WI..(j + 1) * WI];
            let s = (0..IH)
                .map(|h| {
                    let d: f64 = q[h * WI..(h + 1) * WI]
                        .iter()
                        .zip(k)
                        .map(|(q, k)| q * f64::from(*k))
                        .sum();
                    wts[h] * scale * d.max(0.0)
                })
                .sum();
            (s, j)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let top_k = len.min(TOPK);
    if scored.len() > top_k {
        let (kept, dropped) = (scored[top_k - 1].0, scored[top_k].0);
        if kept - dropped <= 1e-4 * kept.abs().max(dropped.abs()) + 1e-6 {
            return None;
        }
    }
    let mut sel: Vec<usize> = scored.iter().take(top_k).map(|s| s.1).collect();
    sel.extend((visible..len).take(top_k - sel.len()));
    let mut bias = vec![f32::NEG_INFINITY; len];
    for j in sel {
        bias[j] = 0.0;
    }
    Some(bias)
}

/// **The indexer keeps, for every query row, the blocks its definition picks, in any graphs.**
/// The CSA stand-alone run of [`csa_compressor_rows_match_their_definition_in_any_graphs`] at
/// `top_k` 3, so the selection drops blocks from position 15 on: each graph's top-k bias, row by
/// row, against [`expected_bias`] computed from the LID rows the run stored. The query ropes at
/// the token's own position with the compressed-layer rope, the weights are the scaled projection
/// of the layer input, a block becomes eligible at the token that completes it, and rows of one
/// graph see different numbers of blocks (the three-graph run). Rows on a near-tie are skipped,
/// and most rows must be held to it.
#[test]
fn csa_indexer_selects_by_its_definition_in_any_graphs() {
    let be = CpuBackend::new();
    let w = LayerW::csa();
    let x = rand_vals(CSA_POS * NE, 6, 1.0);
    let qr = rand_vals(CSA_POS * QL, 7, 1.0);
    for (run, chunks) in [
        ("token by token", vec![1; CSA_POS]),
        ("in three graphs", CSA_CHUNKS.to_vec()),
    ] {
        let bufs = layer_bufs(&be, CSA);
        let selections = run_layer(&be, &w, &x, &qr, &chunks, &bufs, |_, _| {});
        let lid = read_f16(&be, bufs[3].as_ref());
        let (mut rows, mut held, mut biting) = (0, 0, 0);
        for s in &selections {
            for t in 0..s.rows {
                let pos = s.start + t;
                rows += 1;
                if (pos + 1) / CSA > TOPK {
                    biting += 1;
                }
                let Some(want) = expected_bias(&w, &x, &qr, &lid, pos, s.len) else {
                    continue;
                };
                held += 1;
                assert_eq!(
                    &s.bias[t * s.len..(t + 1) * s.len],
                    &want[..],
                    "{run}: the row at position {pos} kept other blocks than its definition"
                );
            }
        }
        // Every row of every graph whose last row sees a block.
        let mut end = 0;
        let want_rows: usize = chunks
            .iter()
            .map(|&t| {
                end += t;
                if end / CSA > 0 {
                    t
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(rows, want_rows, "{run}: a graph's rows went unscored");
        assert!(
            biting > 80 && held * 10 >= rows * 9,
            "{run}: {held} of {rows} rows held to the definition, {biting} of them dropping blocks"
        );
    }
}
