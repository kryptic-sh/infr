//! DeepSeek V4's compressed-layer emit: the rope a V4 layer applies to its heads' tails, and the
//! HCA (ratio 128) compressor that turns each completed block of 128 positions into one row of the
//! layer's compressed cache. Transcribes `build_attention_impl`'s compressor half and
//! `build_hca_compressed_kv_from_state` from llama.cpp's `src/models/deepseek4.cpp` (the pin
//! `docs/plan/09-deepseek-v4-compressed.md` names); the index vectors come from
//! [`super::dsv4_plan`], the port of `dsv4_build_comp_plan`.
//!
//! The runner's V4 arm owns the shared prologue (query, raw kv row, raw cache write) and the
//! attention; it calls [`Dsv4Tier::emit_hca`] between the raw cache write and the attention, and
//! hands the [`CompSegment`] it returns to `Op::Attention::comp`.
//!
//! The CSA tier (ratio 4: the overlapping compressor, the lightning indexer and its LID cache) is
//! not emitted yet: `generate_dense_backend` refuses a ratio-4 layer before any graph is built.

use super::dsv4_plan::{build_dsv4_comp_plan, Dsv4CompPlan, DSV4_HCA_RATIO};
use super::weights::Dsv4CompressorW;
use crate::seam::{Dsv4CompBuf, DSV4_TIER_BUFS};
use crate::Config;
use infr_core::graph::{CompSegment, Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};

/// How one V4 layer ropes the `[nope | rope]` tail of its heads.
///
/// `build_attention_impl` picks the parameters per layer: a ratio-0 layer ropes plain at
/// `rope_theta` (`freq_scale` 1, `ext_factor` 0, betas 0), and EVERY rope of a compressed layer —
/// the query, the raw kv row, the attention output's de-rope and each compressed row — runs at
/// `compress_rope_theta` with the file's YaRN scaling, which `Op::Rope` takes as per-pair
/// divisors (`Config::dsv4_compressed_rope_divisors` maps one onto the other).
#[derive(Clone, Copy, Debug)]
pub(super) struct Dsv4Rope {
    pub(super) head_dim: u32,
    pub(super) rope_dim: u32,
    pub(super) theta: f32,
    pub(super) freq_factors: Option<TensorId>,
}

impl Dsv4Rope {
    /// Layer `l`'s rope. `compressed_ff` is the graph Input holding the compressed-layer divisors
    /// (`DecodeHandles::yarn_ff` on a V4 model with a compressed layer).
    pub(super) fn for_layer(c: &Config, l: usize, compressed_ff: Option<TensorId>) -> Self {
        let (theta, freq_factors) = if c.layer_compress_ratio(l) == 0 {
            (c.rope_theta, None)
        } else {
            (
                c.compress_rope_theta,
                Some(compressed_ff.expect(
                    "a V4 model with a compressed layer declares the compressed-rope divisors",
                )),
            )
        };
        Dsv4Rope {
            head_dim: c.head_dim as u32,
            rope_dim: c.rope_dim as u32,
            theta,
            freq_factors,
        }
    }

    /// Rope the tail of each of the `heads` heads of `tokens` rows of `x` in place, at
    /// `positions` (one i32 per row): `Op::Rope` rotates `[0, rope_dim)` of a packed row, so the
    /// tails are gathered into `pack` (`[tokens * heads, rope_dim]`), rotated and written back,
    /// one `CopyStrided` each way. `backward` is `ggml_rope_ext_back`, the de-rope of the attention
    /// output.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn tail(
        &self,
        g: &mut Graph,
        x: TensorId,
        pack: TensorId,
        tokens: u32,
        heads: u32,
        positions: TensorId,
        backward: bool,
    ) {
        let (hd, rd) = (self.head_dim, self.rope_dim);
        if rd == 0 {
            return;
        }
        let nope = hd - rd;
        g.push(Op::CopyStrided {
            src: x,
            src_off: nope,
            src_stride: hd,
            dst: pack,
            dst_off: 0,
            dst_stride: rd,
            rows: tokens * heads,
            n: rd,
        });
        g.push(Op::Rope {
            x: pack,
            positions,
            dst: pack,
            rows: tokens,
            n_head: heads,
            head_dim: rd,
            rope_dim: rd,
            theta: self.theta,
            freq_factors: self.freq_factors,
            x_stride: 0,
            // NORM (interleaved pairs) — `llama_model::rope_type` puts DEEPSEEK4 in the NORM
            // group, and V4's indexer does NOT inherit V3.2's NEOX override (docs/deepseek.md §
            // Stage 4, "Two corrections").
            neox: false,
            backward,
        });
        g.push(Op::CopyStrided {
            src: pack,
            src_off: 0,
            src_stride: rd,
            dst: x,
            dst_off: nope,
            dst_stride: hd,
            rows: tokens * heads,
            n: rd,
        });
    }
}

/// One compressor tier's per-graph half: the plan for this graph's positions, the per-step i32
/// inputs it declares, and the scratch every layer of the tier shares. Built once per graph; each
/// layer of the tier then emits through it with its own weights and buffers.
pub(super) struct Dsv4Tier {
    ratio: u32,
    /// The compressor's head width: one compressed row, one ring row.
    width: u32,
    /// Ring rows: the source's persistent half.
    state_rows: u32,
    tokens: u32,
    plan: Dsv4CompPlan,
    /// `[tokens]` i32: each token's APE row, `pos % ratio` (the plan's `state_pos`).
    ape_ids: TensorId,
    /// `[tokens, width]` scratch: the compressor's kv rows, its score rows, the APE rows.
    kv: TensorId,
    score: TensorId,
    ape_rows: TensorId,
    /// `None` when this graph's positions complete no block.
    commit: Option<Dsv4Commit>,
}

/// What a graph that completes blocks adds to its tier.
struct Dsv4Commit {
    /// Blocks completed, and the compressed-cache row of the first: the real commits are
    /// consecutive rows for one contiguous sequence.
    blocks: u32,
    first_row: u32,
    /// `[blocks * ratio]` i32: each block's source rows (`Dsv4CompPlan::gather_read_idxs`).
    read_idx: TensorId,
    /// `[blocks]` i32: each block's FIRST position, which its row ropes at.
    block_pos: TensorId,
    /// `[(state_rows + tokens), width]` scratch: `[ring | this graph's rows]`, kv and score.
    src_kv: TensorId,
    src_score: TensorId,
    /// `[blocks, ratio, width]` scratch: each block's pooling window, kv and score.
    win_kv: TensorId,
    win_score: TensorId,
    /// `[blocks, width]` scratch: the pooled rows, normed and roped in place.
    pooled: TensorId,
    /// `[blocks, rope_dim]` scratch for their rope.
    pack: TensorId,
}

impl Dsv4Tier {
    /// The HCA tier of a graph over positions `start_pos..start_pos + tokens` of one sequence.
    /// `bufs` is an HCA layer's [`crate::seam::dsv4_comp_bufs`] geometry, which every HCA layer
    /// shares; the cache's row count is read off it so the plan addresses the cache the session
    /// allocated. The per-step inputs are declared in `g` and their values pushed onto
    /// `step_i32`.
    pub(super) fn hca(
        g: &mut Graph,
        c: &Config,
        start_pos: usize,
        tokens: usize,
        bufs: &[Dsv4CompBuf],
        step_i32: &mut Vec<(TensorId, Vec<i32>)>,
    ) -> Self {
        let width = c.head_dim;
        let ratio = DSV4_HCA_RATIO;
        let [cache, ring, _] = bufs else {
            panic!("an HCA layer owns {DSV4_TIER_BUFS} compressed-tier buffers, got {bufs:?}");
        };
        let state_rows = ring.elems / width;
        let kv_size = cache.elems / width;
        let positions: Vec<i32> = (start_pos..start_pos + tokens).map(|p| p as i32).collect();
        let plan = build_dsv4_comp_plan(
            &positions,
            1,
            ratio,
            false,
            false,
            state_rows as u32,
            kv_size as u32,
        )
        .expect("the HCA plan of one contiguous sequence has no failure case");
        let f32d = |n: usize| TensorDesc::new(vec![n.max(1)], DType::F32);
        let mut input_i32 = |g: &mut Graph, v: Vec<i32>| {
            let id = g.input(TensorDesc::new(vec![v.len()], DType::I32));
            step_i32.push((id, v));
            id
        };
        let ape_ids = input_i32(g, plan.state_pos.clone());
        let blocks = plan.n_real_commits;
        let commit = (blocks > 0).then(|| {
            // Real commits only: a dummy commit (`build_dsv4_comp_plan`'s "HCA: a ubatch that
            // completes no block still commits one dummy") writes a row no token can see before a
            // real commit overwrites it, so it changes no number and is not emitted.
            let rows = &plan.state_write_idxs[..blocks];
            let first_row = rows[0];
            assert!(
                rows.iter().zip(first_row..).all(|(&r, want)| r == want),
                "a contiguous sequence commits consecutive cache rows, got {rows:?}"
            );
            let reads = plan.gather_read_idxs()[..blocks * ratio as usize].to_vec();
            Dsv4Commit {
                blocks: blocks as u32,
                first_row: first_row as u32,
                read_idx: input_i32(g, reads),
                block_pos: input_i32(g, plan.state_write_pos[..blocks].to_vec()),
                src_kv: g.internal(f32d((state_rows + tokens) * width)),
                src_score: g.internal(f32d((state_rows + tokens) * width)),
                win_kv: g.internal(f32d(blocks * ratio as usize * width)),
                win_score: g.internal(f32d(blocks * ratio as usize * width)),
                pooled: g.internal(f32d(blocks * width)),
                pack: g.internal(f32d(blocks * c.rope_dim)),
            }
        });
        Dsv4Tier {
            ratio,
            width: width as u32,
            state_rows: state_rows as u32,
            tokens: tokens as u32,
            ape_ids,
            kv: g.internal(f32d(tokens * width)),
            score: g.internal(f32d(tokens * width)),
            ape_rows: g.internal(f32d(tokens * width)),
            commit,
            plan,
        }
    }

    /// Emit one HCA layer's compressor over `x` (the attention-normed layer input,
    /// `[tokens, n_embd]`), committing every block this graph completes into the layer's
    /// compressed cache, then persisting this graph's rows into its ring. Returns the compressed
    /// segment the layer's attention reads, or `None` while no block is visible to any row (an
    /// all-masked segment adds nothing; plan 09 "Notes for S6 and S7").
    ///
    /// `bufs` are the layer's compressed-tier Inputs (cache, ring kv, ring score). Every op lands
    /// before the caller's `Op::Attention`, so a token that completes a block sees it (commit
    /// before attention), and the ring is written only after the commit has read it (persist
    /// after compress), as in `build_attention_impl`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_hca(
        &self,
        g: &mut Graph,
        w: &Dsv4CompressorW,
        x: TensorId,
        n_embd: u32,
        bufs: &[TensorId],
        rope: Dsv4Rope,
        eps: f32,
    ) -> Option<CompSegment> {
        let &[cache, ring_kv, ring_score] = bufs else {
            panic!("an HCA layer binds {DSV4_TIER_BUFS} compressed-tier Inputs, got {bufs:?}");
        };
        let (t, wd, s) = (self.tokens, self.width, self.state_rows);
        // `hca_state_kv`, `hca_state_score` and `hca_state_score_ape`: the APE row is added
        // BEFORE anything reads or persists the scores, so the ring holds scores with it.
        g.push(Op::Linear {
            x,
            weight: w.kv,
            dst: self.kv,
            m: t,
            in_f: n_embd,
            out_f: wd,
            w_off: 0,
        });
        g.push(Op::Linear {
            x,
            weight: w.gate,
            dst: self.score,
            m: t,
            in_f: n_embd,
            out_f: wd,
            w_off: 0,
        });
        g.push(Op::EmbedGather {
            ids: self.ape_ids,
            table: w.ape,
            dst: self.ape_rows,
            rows: t,
            ne: wd,
            scale: 1.0,
        });
        g.push(Op::Add {
            a: self.score,
            b: self.ape_rows,
            dst: self.score,
            n: t * wd,
        });

        if let Some(cm) = &self.commit {
            // The source `ggml_concat(ring, this ubatch's rows)`, assembled once for values and
            // once for scores. `Op::GatherRows` reads one op-produced tensor, never the ring
            // Input itself, so the ring is copied in.
            for (ring, rows, src) in [
                (ring_kv, self.kv, cm.src_kv),
                (ring_score, self.score, cm.src_score),
            ] {
                g.push(Op::Copy {
                    src: ring,
                    src_off: 0,
                    dst: src,
                    dst_off: 0,
                    n: s * wd,
                });
                g.push(Op::Copy {
                    src: rows,
                    src_off: 0,
                    dst: src,
                    dst_off: s * wd,
                    n: t * wd,
                });
            }
            // Each block's own `ratio` source rows, in index order (`Op::GatherRows`' HCA form).
            // No HCA read is before the sequence start, so the fill is never written.
            for (src, win, fill) in [
                (cm.src_kv, cm.win_kv, 0.0),
                (cm.src_score, cm.win_score, f32::NEG_INFINITY),
            ] {
                g.push(Op::GatherRows {
                    src,
                    idx: cm.read_idx,
                    dst: win,
                    rows: cm.blocks * self.ratio,
                    idx_off: 0,
                    src_stride: wd,
                    src_col: 0,
                    n: wd,
                    group: 1,
                    dst_group_stride: 1,
                    dst_row_off: 0,
                    fill,
                });
            }
            // Per channel: softmax of the scores over the block's rows, weighting its values.
            g.push(Op::CompressPool {
                values: cm.win_kv,
                scores: cm.win_score,
                dst: cm.pooled,
                blocks: cm.blocks,
                window: self.ratio,
                n_embd: wd,
            });
            g.push(Op::RmsNorm {
                x: cm.pooled,
                weight: w.norm,
                dst: cm.pooled,
                rows: cm.blocks,
                dim: wd,
                eps,
            });
            // Roped at the block's FIRST position (`state_write_pos`), with the compressed-layer
            // rope.
            rope.tail(g, cm.pooled, cm.pack, cm.blocks, 1, cm.block_pos, false);
            g.push(Op::WriteKv {
                src: cm.pooled,
                cache,
                rows: cm.blocks,
                row_stride: wd,
                pos: cm.first_row,
            });
        }

        // Persist this graph's rows into the ring, after the commit above has read it: ring row
        // `dst` takes the row of the latest token landing on it (`state_persist_*`), in runs that
        // are consecutive on both sides.
        let pairs: Vec<(u32, u32)> = self
            .plan
            .state_persist_src_idxs
            .iter()
            .zip(&self.plan.state_persist_dst_idxs)
            .map(|(&src, &dst)| (src as u32, dst as u32))
            .collect();
        let mut i = 0;
        while i < pairs.len() {
            let (src0, dst0) = pairs[i];
            let mut n = 1;
            while i + n < pairs.len() && pairs[i + n] == (src0 + n as u32, dst0 + n as u32) {
                n += 1;
            }
            for (rows, ring) in [(self.kv, ring_kv), (self.score, ring_score)] {
                g.push(Op::Copy {
                    src: rows,
                    src_off: src0 * wd,
                    dst: ring,
                    dst_off: dst0 * wd,
                    n: n as u32 * wd,
                });
            }
            i += n;
        }

        (self.plan.n_visible_max > 0).then_some(CompSegment {
            cache,
            len: self.plan.n_visible_max as u32,
            ratio: self.ratio,
            bias: None,
        })
    }
}

// The synthetic V4 GGUF the integration tests build (`tests/dsv4_fixture`), for the session tests
// below, which drive `SeamKv` through the crate-private runner.
#[cfg(test)]
#[path = "../../tests/dsv4_fixture/mod.rs"]
mod dsv4_fixture;

#[cfg(test)]
mod tests {
    use super::dsv4_fixture::{dsv4_dims, dsv4_model};
    use super::{Dsv4Rope, Dsv4Tier};
    use crate::seam::weights::{Dsv4CompressorW, SeamKv};
    use crate::seam::TokenEmbd;
    use crate::{Config, EngineConfig};
    use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
    use infr_core::graph::Graph;
    use infr_core::tensor::{DType, TensorDesc, TensorId};
    use infr_cpu::CpuBackend;
    use infr_gguf::Gguf;
    use infr_testkit::gguf::{splitmix64, Fill, Meta, SyntheticModel, TempGguf};

    const HCA: usize = super::DSV4_HCA_RATIO as usize;

    // ── the compressor on its own: one HCA layer's ops on the CPU interpreter ──────────────────

    /// The stand-alone compressor's dimensions: a small head with a real `[nope | rope]` split.
    const NE: usize = 12;
    const W: usize = 16;
    const RD: usize = 8;
    const THETA: f32 = 40000.0;
    const EPS: f32 = 1e-5;
    /// Positions run: two completed blocks and a partial third.
    const N_POS: usize = 300;
    /// The context its buffers are sized for.
    const CTX: usize = 384;

    fn rand_vals(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = splitmix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                ((h >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// One HCA layer's weights and the rope divisors, as host values.
    struct Weights {
        kv: Vec<f32>,
        gate: Vec<f32>,
        ape: Vec<f32>,
        norm: Vec<f32>,
        ff: Vec<f32>,
    }

    fn weights() -> Weights {
        Weights {
            kv: rand_vals(W * NE, 1, 0.5),
            gate: rand_vals(W * NE, 2, 0.5),
            ape: rand_vals(HCA * W, 3, 0.5),
            norm: rand_vals(W, 4, 0.25).iter().map(|v| 1.0 + v).collect(),
            ff: (0..RD / 2).map(|p| 1.0 + p as f32).collect(),
        }
    }

    /// The session-held buffers of one HCA layer: its compressed cache and its ring.
    struct LayerBufs {
        cache: Box<dyn Buffer>,
        ring_kv: Box<dyn Buffer>,
        ring_score: Box<dyn Buffer>,
    }

    fn cfg() -> Config {
        Config {
            n_layer: 1,
            head_dim: W,
            rope_dim: RD,
            deepseek4: true,
            compress_ratios: vec![HCA],
            ..Default::default()
        }
    }

    fn upload(be: &CpuBackend, v: &[f32], usage: BufferUsage) -> Box<dyn Buffer> {
        let b = be.alloc(v.len() * 4, usage).expect("alloc");
        be.upload(b.as_ref(), bytemuck::cast_slice(v))
            .expect("upload");
        b
    }

    fn layer_bufs(be: &CpuBackend) -> LayerBufs {
        let geometry = crate::seam::dsv4_comp_bufs(&cfg(), 0, CTX, DType::F16);
        let alloc = |i: usize| {
            be.alloc(geometry[i].bytes(), BufferUsage::KvCache)
                .expect("alloc")
        };
        LayerBufs {
            cache: alloc(0),
            ring_kv: alloc(1),
            ring_score: alloc(2),
        }
    }

    /// Feed `x` (`[N_POS, NE]`) through one HCA layer's compressor in graphs of `chunks` tokens,
    /// in order, on `bufs`.
    fn run_compressor(be: &CpuBackend, w: &Weights, x: &[f32], chunks: &[usize], bufs: &LayerBufs) {
        let c = cfg();
        let geometry = crate::seam::dsv4_comp_bufs(&c, 0, CTX, DType::F16);
        let host = [&w.kv, &w.gate, &w.ape, &w.norm];
        let wb: Vec<Box<dyn Buffer>> = host
            .iter()
            .map(|v| upload(be, v, BufferUsage::Weights))
            .collect();
        let ffb = upload(be, &w.ff, BufferUsage::Staging);
        let mut start = 0;
        for &t in chunks {
            let mut g = Graph::new();
            let xin = g.input(TensorDesc::new(vec![t * NE], DType::F32));
            let ff = g.input(TensorDesc::new(vec![RD / 2], DType::F32));
            let ins: Vec<TensorId> = geometry
                .iter()
                .map(|b| g.input(TensorDesc::new(vec![b.elems], b.dtype)))
                .collect();
            let wid: Vec<TensorId> = host
                .iter()
                .map(|v| g.weight(TensorDesc::new(vec![v.len()], DType::F32)))
                .collect();
            let cw = Dsv4CompressorW {
                kv: wid[0],
                gate: wid[1],
                ape: wid[2],
                norm: wid[3],
            };
            let mut step = Vec::new();
            let tier = Dsv4Tier::hca(&mut g, &c, start, t, &geometry, &mut step);
            let rope = Dsv4Rope {
                head_dim: W as u32,
                rope_dim: RD as u32,
                theta: THETA,
                freq_factors: Some(ff),
            };
            tier.emit_hca(&mut g, &cw, xin, NE as u32, &ins, rope, EPS);
            let plan = be.compile(&g).expect("compile");
            let xb = upload(be, &x[start * NE..(start + t) * NE], BufferUsage::Staging);
            let step_bufs: Vec<Box<dyn Buffer>> = step
                .iter()
                .map(|(_, v)| {
                    let b = be.alloc(v.len() * 4, BufferUsage::Staging).expect("alloc");
                    be.upload(b.as_ref(), bytemuck::cast_slice(v))
                        .expect("upload");
                    b
                })
                .collect();
            let mut b = Bindings::new();
            b.bind(xin, xb.as_ref()).bind(ff, ffb.as_ref());
            for (id, buf) in ins
                .iter()
                .zip([&bufs.cache, &bufs.ring_kv, &bufs.ring_score])
            {
                b.bind(*id, buf.as_ref());
            }
            for (id, buf) in wid.iter().zip(&wb) {
                b.bind(*id, buf.as_ref());
            }
            for ((id, _), buf) in step.iter().zip(&step_bufs) {
                b.bind(*id, buf.as_ref());
            }
            be.execute(plan.as_ref(), &b).expect("execute");
            start += t;
        }
        assert_eq!(start, N_POS, "the chunks cover the run");
    }

    fn read_f32(be: &CpuBackend, b: &dyn Buffer) -> Vec<f32> {
        let mut v = vec![0f32; b.len_bytes() / 4];
        be.download(b, bytemuck::cast_slice_mut(&mut v))
            .expect("download");
        v
    }

    fn read_f16(be: &CpuBackend, b: &dyn Buffer) -> Vec<f32> {
        let mut v = vec![0u16; b.len_bytes() / 2];
        be.download(b, bytemuck::cast_slice_mut(&mut v))
            .expect("download");
        v.iter()
            .map(|&h| half::f16::from_bits(h).to_f32())
            .collect()
    }

    /// Block `blk`'s compressed row from its definition (`build_hca_compressed_kv_from_state`):
    /// per channel, the softmax over the block's positions of `gate·x + ape[pos % ratio]`
    /// weighting `kv·x`; then the RMSNorm with its gain; then the NORM rope of the row's tail at
    /// the block's FIRST position, each pair's angle divided by its divisor.
    fn block_row(w: &Weights, x: &[f32], blk: usize) -> Vec<f32> {
        let proj = |m: &[f32], p: usize| -> Vec<f32> {
            (0..W)
                .map(|o| (0..NE).map(|i| m[o * NE + i] * x[p * NE + i]).sum())
                .collect()
        };
        let positions = blk * HCA..(blk + 1) * HCA;
        let kv: Vec<Vec<f32>> = positions.clone().map(|p| proj(&w.kv, p)).collect();
        let sc: Vec<Vec<f32>> = positions
            .map(|p| {
                let mut s = proj(&w.gate, p);
                for (s, a) in s.iter_mut().zip(&w.ape[(p % HCA) * W..]) {
                    *s += a;
                }
                s
            })
            .collect();
        let mut row: Vec<f32> = (0..W)
            .map(|ch| {
                let m = sc.iter().map(|r| r[ch]).fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = sc.iter().map(|r| (r[ch] - m).exp()).collect();
                let den: f32 = e.iter().sum();
                e.iter().zip(&kv).map(|(e, v)| e * v[ch]).sum::<f32>() / den
            })
            .collect();
        let ms = row.iter().map(|v| v * v).sum::<f32>() / W as f32;
        let s = 1.0 / (ms + EPS).sqrt();
        for (v, g) in row.iter_mut().zip(&w.norm) {
            *v *= s * g;
        }
        let pos = (blk * HCA) as f32;
        let nope = W - RD;
        for p in 0..RD / 2 {
            let ang = pos * THETA.powf(-2.0 * p as f32 / RD as f32) / w.ff[p];
            let (s, c) = ang.sin_cos();
            let (a, b) = (row[nope + 2 * p], row[nope + 2 * p + 1]);
            row[nope + 2 * p] = a * c - b * s;
            row[nope + 2 * p + 1] = a * s + b * c;
        }
        row
    }

    /// **The HCA compressor writes, for every completed block, the row its definition gives, and
    /// does so whatever graphs the positions arrive in.** One HCA layer's ops alone, on the CPU
    /// interpreter, fed 300 positions token by token and again in three graphs that each end
    /// mid-block (100, 50 and 150 tokens): the second and third read the ring for positions
    /// earlier graphs wrote, and the third completes block 1 while its own rows are about to be
    /// persisted over the ring rows that block reads. Both runs must leave the two committed rows
    /// at [`block_row`] (block 1 ropes at position 128, so a row roped anywhere else misses) and
    /// must agree with each other on the cache and the ring.
    #[test]
    fn hca_compressor_rows_match_their_definition_in_any_graphs() {
        let be = CpuBackend::new();
        let w = weights();
        let x = rand_vals(N_POS * NE, 5, 1.0);
        let one = layer_bufs(&be);
        run_compressor(&be, &w, &x, &[1; N_POS], &one);
        let chunked = layer_bufs(&be);
        run_compressor(&be, &w, &x, &[100, 50, 150], &chunked);

        for (run, bufs) in [("token by token", &one), ("in three graphs", &chunked)] {
            let cache = read_f16(&be, bufs.cache.as_ref());
            for blk in 0..N_POS / HCA {
                let want = block_row(&w, &x, blk);
                let got = &cache[blk * W..(blk + 1) * W];
                for (ch, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-3 * w.abs() + 1e-4,
                        "{run}: block {blk} channel {ch} holds {g}, its definition gives {w}"
                    );
                }
            }
        }
        let max_diff = |a: Vec<f32>, b: Vec<f32>| {
            a.iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        };
        for (what, d) in [
            (
                "ring kv",
                max_diff(
                    read_f32(&be, one.ring_kv.as_ref()),
                    read_f32(&be, chunked.ring_kv.as_ref()),
                ),
            ),
            (
                "ring score",
                max_diff(
                    read_f32(&be, one.ring_score.as_ref()),
                    read_f32(&be, chunked.ring_score.as_ref()),
                ),
            ),
        ] {
            assert!(
                d <= 1e-5,
                "{what}: token by token and in three graphs differ by {d}"
            );
        }
    }

    // ── sessions: the synthetic V4 file through the runner ────────────────────────────────────

    /// The session tests' context: room for every prompt below.
    const SESSION_CTX: usize = 640;

    /// A two-layer V4 file: an HCA layer, then a ratio-0 one, with the oracle files' YaRN keys
    /// so the compressed layer's rope differs from the plain one. `scale_comp_kv` multiplies every
    /// `attn_compressor_kv` weight.
    fn hca_model(scale_comp_kv: f32) -> SyntheticModel {
        let mut d = dsv4_dims();
        d.compress_ratios = vec![HCA, 0];
        d.hash_layer_count = 1;
        d.swiglu_clamp_exp = vec![7.0, 7.5];
        let mut m = dsv4_model(&d);
        for (k, v) in m.meta.iter_mut() {
            if k == "deepseek4.context_length" {
                *v = Meta::U32(SESSION_CTX as u32);
            }
        }
        m.meta.extend([
            (
                "deepseek4.rope.scaling.type".to_string(),
                Meta::Str("yarn".into()),
            ),
            ("deepseek4.rope.scaling.factor".to_string(), Meta::F32(16.0)),
            (
                "deepseek4.rope.scaling.original_context_length".to_string(),
                Meta::U32(512),
            ),
        ]);
        m.meta.sort_by(|a, b| a.0.cmp(&b.0));
        let mut n = 0;
        for t in m.tensors.iter_mut() {
            if t.name.ends_with("attn_compressor_kv.weight") {
                let v: Vec<f32> = t.values().iter().map(|x| x * scale_comp_kv).collect();
                t.fill = Fill::Exact(v);
                n += 1;
            }
        }
        assert_eq!(n, 1, "the model has one HCA layer");
        m
    }

    /// One loaded synthetic V4 file on the CPU backend.
    struct Model {
        g: Gguf,
        cfg: Config,
        embd: std::sync::OnceLock<Vec<f32>>,
        be: CpuBackend,
        ec: EngineConfig,
    }

    impl Model {
        fn open(tag: &str, m: &SyntheticModel) -> Self {
            let tmp = TempGguf::write(tag, m);
            let g = Gguf::open(tmp.path()).expect("open synthetic GGUF");
            let cfg = Config::from_gguf(&g).expect("synthetic config");
            assert!(cfg.has_unrewindable_state());
            Model {
                g,
                cfg,
                embd: std::sync::OnceLock::new(),
                be: CpuBackend::new(),
                ec: EngineConfig::default(),
            }
        }

        /// The last position's logits of `prompt` on `state` (a fresh session when it is `None`).
        fn logits(&self, prompt: &[u32], state: &mut Option<SeamKv>) -> Vec<f32> {
            self.turn(prompt, state).0
        }

        /// [`Self::logits`] and how many of `prompt`'s positions the turn took from the session
        /// instead of prefilling them (`GenStats::n_cached`): what tells a reuse from a re-prefill.
        fn turn(&self, prompt: &[u32], state: &mut Option<SeamKv>) -> (Vec<f32>, usize) {
            let mut out = Vec::new();
            let (_, stats) = crate::seam::generate_dense_backend(
                &self.be,
                &crate::seam::cpu_upload_bind(&self.be),
                &self.g,
                &self.cfg,
                &self.ec,
                TokenEmbd::new(&self.embd, &self.g),
                None,
                prompt,
                1,
                |_| {},
                state,
                SESSION_CTX,
                None,
                None,
                None,
                Some(&mut out),
                None,
                None,
                None,
            )
            .expect("synthetic V4 generation");
            assert_eq!(out.len(), self.cfg.vocab);
            (out, stats.n_cached)
        }

        fn fresh(&self, prompt: &[u32]) -> Vec<f32> {
            self.logits(prompt, &mut None)
        }

        /// The bytes of every compressor ring a session holds.
        fn rings(&self, state: &Option<SeamKv>) -> Vec<Vec<u8>> {
            let kv = state.as_ref().expect("a session");
            let mut out = Vec::new();
            for l in 0..self.cfg.n_layer {
                let geometry = crate::seam::dsv4_comp_bufs(&self.cfg, l, kv.max_ctx, kv.k_fmt);
                for (b, buf) in geometry.iter().zip(&kv.cbufs[l]) {
                    if b.ring {
                        let mut v = vec![0u8; buf.len_bytes()];
                        self.be.download(buf.as_ref(), &mut v).expect("download");
                        out.push(v);
                    }
                }
            }
            assert!(!out.is_empty(), "the session holds no ring");
            out
        }
    }

    /// `n` prompt ids drawn from `salt`.
    fn ids(n: usize, salt: u64) -> Vec<u32> {
        let vocab = dsv4_dims().vocab as u64;
        (0..n as u64)
            .map(|i| (splitmix64(i ^ (salt << 32)) % vocab) as u32)
            .collect()
    }

    /// **The compressor's weights are inert until the first block completes, and load-bearing from
    /// the token that completes it.** Scaling `attn_compressor_kv` changes only the rows the
    /// compressor pools; before position 127 none is visible, so every logit up to 127 tokens is
    /// bit-identical, and at 128 tokens (position 127 completes block 0 and attends it in the same
    /// step) and after, the logits move. One session per file, extended a token at a time, so
    /// each length is also a reuse of the session before it.
    #[test]
    fn hca_compressor_weights_are_inert_until_the_first_block() {
        let base = Model::open("dsv4-hca-inert-base", &hca_model(1.0));
        let scaled = Model::open("dsv4-hca-inert-scaled", &hca_model(1.5));
        let prompt = ids(140, 1);
        let (mut a, mut b) = (None, None);
        for len in 1..=prompt.len() {
            let (la, reused) = base.turn(&prompt[..len], &mut a);
            assert_eq!(
                reused,
                len - 1,
                "the turn to {len} tokens did not extend the session"
            );
            let lb = scaled.logits(&prompt[..len], &mut b);
            if len < HCA {
                assert_eq!(
                    la, lb,
                    "the compressor's kv weights moved the logits at {len} tokens"
                );
            } else {
                assert_ne!(
                    la, lb,
                    "the compressor's kv weights left the logits at {len} tokens unmoved"
                );
            }
        }
    }

    /// **A prompt prefilled in two calls ends where one call does, bit for bit.** The first call
    /// stops at 150 tokens, part way into block 1, so the second call — which must take those 150
    /// positions from the session and prefill only the rest — completes that block from ring rows
    /// 0..=21 the first call persisted, beside the block-0 row it committed.
    #[test]
    fn hca_split_prefill_equals_one_prefill() {
        let m = Model::open("dsv4-hca-split", &hca_model(1.0));
        let prompt = ids(270, 2);
        let mut s = None;
        assert_eq!(m.turn(&prompt[..150], &mut s).1, 0);
        let (got, reused) = m.turn(&prompt, &mut s);
        assert_eq!(
            reused, 150,
            "the extending turn re-prefilled instead of reusing the session"
        );
        assert_eq!(got, m.fresh(&prompt));
    }

    /// **A new prompt after a long one starts from a reset state.** The session first runs 200
    /// tokens (one block committed, the whole ring written); a 50-token prompt with nothing in
    /// common then gives a fresh session's logits AND leaves the session's rings byte-equal to a
    /// fresh session's: ring rows past 50 hold the first prompt's rows unless the reset zeroed
    /// them.
    #[test]
    fn hca_new_prompt_after_a_long_one_starts_from_a_reset_state() {
        let m = Model::open("dsv4-hca-reset", &hca_model(1.0));
        let (long, short) = (ids(200, 3), ids(50, 4));
        assert_ne!(long[0], short[0], "the prompts share no prefix");
        let mut s = None;
        m.logits(&long, &mut s);
        let (got, reused) = m.turn(&short, &mut s);
        assert_eq!(
            reused, 0,
            "a turn sharing nothing reused {reused} positions"
        );
        let mut fresh = None;
        let want = m.logits(&short, &mut fresh);
        assert_eq!(got, want);
        assert!(
            m.rings(&s) == m.rings(&fresh),
            "the reset left a ring row dirty"
        );
    }

    /// **A prompt that shares a prefix with a longer earlier one re-prefills instead of
    /// rewinding.** The session first runs 300 tokens, so its ring rows for positions 128..=149
    /// now hold positions 256..=277. The second prompt keeps the first 150 tokens and runs past
    /// position 255, so block 1 pools positions 128..=255: a rewind to the shared prefix would
    /// read those rows from the ring and pool the first prompt's later positions instead. The turn
    /// must re-prefill from position 0, not merely end on the right logits.
    #[test]
    fn hca_shared_prefix_does_not_rewind_into_a_stale_ring() {
        let m = Model::open("dsv4-hca-rewind", &hca_model(1.0));
        let first = ids(300, 5);
        let second = [&first[..150], &ids(110, 6)[..]].concat();
        let mut s = None;
        m.logits(&first, &mut s);
        let (got, reused) = m.turn(&second, &mut s);
        assert_eq!(
            reused, 0,
            "a non-extending turn rewound to position {reused}"
        );
        assert_eq!(got, m.fresh(&second));
    }
}
