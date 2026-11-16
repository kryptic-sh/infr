//! DeepSeek V4's compressed-layer emit: the rope a V4 layer applies to its heads' tails, the
//! compressors that turn each completed block of positions into one row of a compressed cache,
//! and on a CSA layer the lightning indexer that picks which of those rows the attention sees.
//! Transcribes `build_attention_impl`'s compressor half, `build_hca_compressed_kv_from_state`,
//! `build_overlap_compressed_kv_from_state`, `build_lid_top_k`, `build_top_k_mask` and the
//! compressed half of `build_csa_lid_attention` from llama.cpp's `src/models/deepseek4.cpp` (the
//! pin `docs/plan/09-deepseek-v4-compressed.md` names); the index vectors come from
//! [`super::dsv4_plan`], the port of `dsv4_build_comp_plan`.
//!
//! The runner's V4 arm owns the shared prologue (query, raw kv row, raw cache write) and the
//! attention; it calls [`Dsv4Tiers::emit`] between the raw cache write and the attention, and
//! hands the [`CompSegment`] it returns to `Op::Attention::comp`.
//!
//! Both tiers share one compressor emit ([`Dsv4Tier`]): an HCA layer (ratio 128) runs it once at
//! `head_dim`, non-overlapping; a CSA layer (ratio 4) runs it twice with the overlapping window,
//! once for its own cache at `head_dim` and once for the indexer's LID cache at
//! `indexer_head_size`, then scores the LID cache with the indexer ([`csa`]).

mod csa;

use super::dsv4_plan::{build_dsv4_comp_plan, Dsv4CompPlan, DSV4_CSA_RATIO, DSV4_HCA_RATIO};
use super::weights::{Dsv4CompW, Dsv4CompressorW};
use crate::seam::{Dsv4CompBuf, DSV4_TIER_BUFS};
use crate::Config;
use infr_core::graph::{CompSegment, Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};

/// How one V4 layer ropes the `[nope | rope]` tail of its heads.
///
/// `build_attention_impl` picks the parameters per layer: a ratio-0 layer ropes plain at
/// `rope_theta` (`freq_scale` 1, `ext_factor` 0, betas 0), and EVERY rope of a compressed layer —
/// the query, the raw kv row, the attention output's de-rope, each compressed row and the
/// indexer query — runs at `compress_rope_theta` with the file's YaRN scaling, which `Op::Rope`
/// takes as per-pair divisors (`Config::dsv4_compressed_rope_divisors` maps one onto the other).
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

    /// The same rope over heads `head_dim` wide: every rope of a layer rotates the last
    /// `rope_dim` dims (the reference's `n_rot()` at every site, with `ggml_rope_set_offset` at
    /// the head's `nope` width), whatever the head's own width — a compressed row, an LID row and
    /// an indexer query head alike.
    pub(super) fn with_head_dim(self, head_dim: u32) -> Self {
        assert!(
            head_dim >= self.rope_dim,
            "a {head_dim}-wide head cannot carry a {}-dim rope tail",
            self.rope_dim
        );
        Dsv4Rope { head_dim, ..self }
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

/// What one V4 layer's compressed half reads from the runner's prologue.
#[derive(Clone, Copy, Debug)]
pub(super) struct Dsv4LayerIn {
    /// The attention-normed layer input `[tokens, n_embd]` (`cur` in `build_attention_impl`):
    /// what every compressor and the indexer's per-head weights project.
    pub(super) x: TensorId,
    /// The normed low-rank query `[tokens, q_lora_rank]` (`qr`): what the indexer query projects.
    pub(super) qr: TensorId,
    /// `[tokens]` i32: each token's position, which the indexer query ropes at.
    pub(super) positions: TensorId,
}

/// The compressed tiers of one graph: the HCA tier when the graph has a ratio-128 layer, the CSA
/// tier when it has a ratio-4 one. Every layer of a tier shares the positions, so it shares the
/// plan, the per-step inputs and the scratch.
pub(super) struct Dsv4Tiers {
    hca: Option<Dsv4Tier>,
    csa: Option<Dsv4Tier>,
}

impl Dsv4Tiers {
    /// The tiers `layers` need for a graph over positions `start_pos..start_pos + tokens` of one
    /// sequence, against the buffers a `max_ctx`-token session allocated with raw cache format
    /// `k_fmt`. The per-step inputs are declared in `g` and their values pushed onto `step_i32`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        g: &mut Graph,
        c: &Config,
        layers: std::ops::Range<usize>,
        start_pos: usize,
        tokens: usize,
        max_ctx: usize,
        k_fmt: DType,
        step_i32: &mut Vec<(TensorId, Vec<i32>)>,
    ) -> Self {
        let mut tier = |ratio: u32| {
            layers
                .clone()
                .find(|&l| c.layer_compress_ratio(l) == ratio as usize)
                .map(|l| {
                    let bufs = crate::seam::dsv4_comp_bufs(c, l, max_ctx, k_fmt);
                    Dsv4Tier::new(g, c, start_pos, tokens, ratio, &bufs, step_i32)
                })
        };
        let hca = tier(DSV4_HCA_RATIO);
        let csa = tier(DSV4_CSA_RATIO);
        Dsv4Tiers { hca, csa }
    }

    /// Emit compressed layer `w`'s half: its compressors, and on a CSA layer the indexer. Returns
    /// the segment the layer's attention reads, or `None` while no block is visible to any row.
    /// `bufs` are the layer's compressed-tier Inputs, in `crate::seam::dsv4_comp_bufs` order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit(
        &self,
        g: &mut Graph,
        c: &Config,
        w: &Dsv4CompW,
        io: Dsv4LayerIn,
        bufs: &[TensorId],
        rope: Dsv4Rope,
        eps: f32,
    ) -> Option<CompSegment> {
        match &w.indexer {
            None => self
                .hca
                .as_ref()
                .expect("an HCA layer's graph declares the HCA tier")
                .emit_hca(g, &w.attn, io.x, c.n_embd as u32, bufs, rope, eps),
            Some(ix) => self
                .csa
                .as_ref()
                .expect("a CSA layer's graph declares the CSA tier")
                .emit_csa(g, c, &w.attn, ix, io, bufs, rope, eps),
        }
    }
}

/// One compressor tier's per-graph half: the plan for this graph's positions, the per-step i32
/// inputs it declares, and the scratch every layer of the tier shares. Built once per graph; each
/// layer of the tier then emits through it with its own weights and buffers.
///
/// A tier runs one compressor per entry of `comps`: the HCA tier one, at `head_dim`; the CSA tier
/// two, its own at `head_dim` and the LID one at `indexer_head_size`. The reference builds the
/// CSA and LID plans by two calls with identical parameters (its own state and cache, whose
/// sizes are equal), so one plan serves both here, and [`Dsv4Tier::new`] checks the sizes agree.
pub(super) struct Dsv4Tier {
    ratio: u32,
    /// The CSA/LID form: each block pools its previous window beside its own, and every ring row
    /// holds two halves (`build_overlap_compressed_kv_from_state`).
    overlap: bool,
    /// Ring rows: the source's persistent half.
    state_rows: u32,
    tokens: u32,
    start_pos: u32,
    plan: Dsv4CompPlan,
    /// `[tokens]` i32: each token's APE row, `pos % ratio` (the plan's `state_pos`).
    ape_ids: TensorId,
    /// `None` when this graph's positions complete no block.
    commit: Option<Dsv4Commit>,
    comps: Vec<Dsv4Scratch>,
    /// The CSA tier's indexer scratch; `None` on HCA, and while no block is visible to any row.
    indexer: Option<csa::Dsv4IndexerScratch>,
}

/// What a graph that completes blocks adds to its tier, shared by the tier's compressors.
struct Dsv4Commit {
    /// Blocks completed, and the compressed-cache row of the first: the real commits are
    /// consecutive rows for one contiguous sequence.
    blocks: u32,
    first_row: u32,
    /// i32: each block's source rows (`Dsv4CompPlan::gather_read_idxs`), real blocks only. On
    /// the overlapping tier, every block's `ratio` previous-window rows, then every block's
    /// `ratio` current-window rows (`blocks * ratio` each).
    read_idx: TensorId,
    /// `[blocks]` i32: each block's FIRST position, which its row ropes at.
    block_pos: TensorId,
}

/// One compressor's per-graph scratch.
struct Dsv4Scratch {
    /// The compressor's head width: one compressed row. A ring row and a projected row are
    /// `coff * width` wide (`coff` 2 on the overlapping tier, 1 on HCA).
    width: u32,
    /// `[tokens, coff * width]`: the compressor's kv rows, its score rows, the APE rows.
    kv: TensorId,
    score: TensorId,
    ape_rows: TensorId,
    commit: Option<Dsv4CommitScratch>,
}

/// One compressor's scratch for a graph that completes blocks.
struct Dsv4CommitScratch {
    /// `[(state_rows + tokens), coff * width]`: `[ring | this graph's rows]`, kv and score.
    src_kv: TensorId,
    src_score: TensorId,
    /// `[blocks, window, width]`: each block's pooling window, kv and score.
    win_kv: TensorId,
    win_score: TensorId,
    /// `[blocks, width]`: the pooled rows, normed and roped in place.
    pooled: TensorId,
    /// `[blocks, rope_dim]` scratch for their rope.
    pack: TensorId,
}

impl Dsv4Tier {
    /// The tier at `ratio` (`DSV4_HCA_RATIO` or `DSV4_CSA_RATIO`) of a graph over positions
    /// `start_pos..start_pos + tokens` of one sequence. `bufs` is a layer of the tier's
    /// [`crate::seam::dsv4_comp_bufs`] geometry, which every layer of the tier shares; the cache
    /// and ring row counts are read off it, so the plan addresses the buffers the session
    /// allocated. The per-step inputs are declared in `g` and their values pushed onto
    /// `step_i32`.
    pub(super) fn new(
        g: &mut Graph,
        c: &Config,
        start_pos: usize,
        tokens: usize,
        ratio: u32,
        bufs: &[Dsv4CompBuf],
        step_i32: &mut Vec<(TensorId, Vec<i32>)>,
    ) -> Self {
        let overlap = match ratio {
            DSV4_HCA_RATIO => false,
            DSV4_CSA_RATIO => true,
            _ => panic!("V4 compresses at ratio {DSV4_HCA_RATIO} or {DSV4_CSA_RATIO}, not {ratio}"),
        };
        let coff = if overlap { 2 } else { 1 };
        let widths: &[usize] = if overlap {
            &[c.head_dim, c.indexer_head_size]
        } else {
            &[c.head_dim]
        };
        assert_eq!(
            bufs.len(),
            widths.len() * DSV4_TIER_BUFS,
            "a ratio-{ratio} layer owns {DSV4_TIER_BUFS} buffers per compressor, got {bufs:?}"
        );
        // (cache rows, ring rows) per compressor: one plan addresses them all.
        let rows: Vec<(usize, usize)> = widths
            .iter()
            .zip(bufs.as_chunks::<DSV4_TIER_BUFS>().0)
            .map(|(&w, b)| (b[0].elems / w, b[1].elems / (coff * w)))
            .collect();
        let (kv_size, state_rows) = rows[0];
        assert!(
            rows.iter().all(|&r| r == (kv_size, state_rows)),
            "the compressors of one tier share a plan, so their cache and ring rows must agree: \
             {rows:?}"
        );
        let positions: Vec<i32> = (start_pos..start_pos + tokens).map(|p| p as i32).collect();
        // `pad_blocks` is the reference's own choice per tier (`dsv4_build_comp_plans`): the
        // CSA and LID plans pad, the HCA plan does not.
        let plan = build_dsv4_comp_plan(
            &positions,
            1,
            ratio,
            overlap,
            overlap,
            state_rows as u32,
            kv_size as u32,
        )
        .expect("the plan of one contiguous sequence has no failure case");
        let f32d = |n: usize| TensorDesc::new(vec![n.max(1)], DType::F32);
        let mut input_i32 = |g: &mut Graph, v: Vec<i32>| {
            let id = g.input(TensorDesc::new(vec![v.len()], DType::I32));
            step_i32.push((id, v));
            id
        };
        let ape_ids = input_i32(g, plan.state_pos.clone());
        let blocks = plan.n_real_commits;
        let n_read = blocks * ratio as usize;
        let window = coff * ratio as usize;
        let commit = (blocks > 0).then(|| {
            // Real commits only: a dummy commit (`build_dsv4_comp_plan`'s padding rules) writes
            // the cache's last row, which no token can see before a real commit overwrites it, so
            // it changes no number and is not emitted. Real commits and their reads come first
            // in every vector (`Dsv4CompPlan`'s doc), in each read half alike.
            let rows = &plan.state_write_idxs[..blocks];
            let first_row = rows[0];
            assert!(
                rows.iter().zip(first_row..).all(|(&r, want)| r == want),
                "a contiguous sequence commits consecutive cache rows, got {rows:?}"
            );
            let all = plan.gather_read_idxs();
            let reads = if overlap {
                let half = all.len() / 2;
                [&all[..n_read], &all[half..half + n_read]].concat()
            } else {
                all[..n_read].to_vec()
            };
            Dsv4Commit {
                blocks: blocks as u32,
                first_row: first_row as u32,
                read_idx: input_i32(g, reads),
                block_pos: input_i32(g, plan.state_write_pos[..blocks].to_vec()),
            }
        });
        let comps = widths
            .iter()
            .map(|&width| {
                let sw = coff * width;
                Dsv4Scratch {
                    width: width as u32,
                    kv: g.internal(f32d(tokens * sw)),
                    score: g.internal(f32d(tokens * sw)),
                    ape_rows: g.internal(f32d(tokens * sw)),
                    commit: (blocks > 0).then(|| Dsv4CommitScratch {
                        src_kv: g.internal(f32d((state_rows + tokens) * sw)),
                        src_score: g.internal(f32d((state_rows + tokens) * sw)),
                        win_kv: g.internal(f32d(blocks * window * width)),
                        win_score: g.internal(f32d(blocks * window * width)),
                        pooled: g.internal(f32d(blocks * width)),
                        pack: g.internal(f32d(blocks * c.rope_dim)),
                    }),
                }
            })
            .collect();
        let indexer = (overlap && plan.n_visible_max > 0)
            .then(|| csa::Dsv4IndexerScratch::new(g, c, tokens, plan.n_visible_max));
        Dsv4Tier {
            ratio,
            overlap,
            state_rows: state_rows as u32,
            tokens: tokens as u32,
            start_pos: start_pos as u32,
            plan,
            ape_ids,
            commit,
            comps,
            indexer,
        }
    }

    /// Emit one HCA layer's compressor over `x` (the attention-normed layer input,
    /// `[tokens, n_embd]`) and return the compressed segment the layer's attention reads, or
    /// `None` while no block is visible to any row (an all-masked segment adds nothing; plan 09
    /// "Notes for S6 and S7"). `bufs` are the layer's compressed-tier Inputs (cache, ring kv, ring
    /// score).
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
        assert!(!self.overlap, "emit_hca on the ratio-{} tier", self.ratio);
        self.compress(g, 0, w, x, n_embd, bufs, rope, eps);
        self.segment(bufs[0], None)
    }

    /// The segment over `cache` the layer's attention reads: every row any token of this graph
    /// can see, or `None` when none can.
    fn segment(&self, cache: TensorId, bias: Option<TensorId>) -> Option<CompSegment> {
        (self.plan.n_visible_max > 0).then_some(CompSegment {
            cache,
            len: self.plan.n_visible_max as u32,
            ratio: self.ratio,
            bias,
        })
    }

    /// Emit compressor `k` of the tier over `x` (`[tokens, n_embd]`), committing every block this
    /// graph completes into its compressed cache, then persisting this graph's rows into its
    /// ring. `bufs` are that compressor's Inputs: cache, ring kv, ring score.
    ///
    /// Every op lands before the caller's `Op::Attention` (and, on CSA, the indexer), so a token
    /// that completes a block sees it (commit before attention), and the ring is written only
    /// after the commit has read it (persist after compress), as in `build_attention_impl`.
    #[allow(clippy::too_many_arguments)]
    fn compress(
        &self,
        g: &mut Graph,
        k: usize,
        w: &Dsv4CompressorW,
        x: TensorId,
        n_embd: u32,
        bufs: &[TensorId],
        rope: Dsv4Rope,
        eps: f32,
    ) {
        let &[cache, ring_kv, ring_score] = bufs else {
            panic!("a compressor binds {DSV4_TIER_BUFS} Inputs, got {bufs:?}");
        };
        let sc = &self.comps[k];
        let (t, wd, s, r) = (self.tokens, sc.width, self.state_rows, self.ratio);
        // A state row: the overlapping compressor's holds the previous-window half `[0, wd)` and
        // the current-window half `[wd, 2*wd)`.
        let sw = if self.overlap { 2 * wd } else { wd };
        // `*_state_kv`, `*_state_score` and `*_state_score_ape`: the APE row is added BEFORE
        // anything reads or persists the scores, so the ring holds scores with it.
        for (weight, dst) in [(w.kv, sc.kv), (w.gate, sc.score)] {
            g.push(Op::Linear {
                x,
                weight,
                dst,
                m: t,
                in_f: n_embd,
                out_f: sw,
                w_off: 0,
            });
        }
        g.push(Op::EmbedGather {
            ids: self.ape_ids,
            table: w.ape,
            dst: sc.ape_rows,
            rows: t,
            ne: sw,
            scale: 1.0,
        });
        g.push(Op::Add {
            a: sc.score,
            b: sc.ape_rows,
            dst: sc.score,
            n: t * sw,
        });

        if let (Some(cm), Some(cs)) = (&self.commit, &sc.commit) {
            // The source `ggml_concat(ring, this ubatch's rows)`, assembled once for values and
            // once for scores. `Op::GatherRows` reads one op-produced tensor, never the ring
            // Input itself, so the ring is copied in.
            for (ring, rows, src) in [
                (ring_kv, sc.kv, cs.src_kv),
                (ring_score, sc.score, cs.src_score),
            ] {
                g.push(Op::Copy {
                    src: ring,
                    src_off: 0,
                    dst: src,
                    dst_off: 0,
                    n: s * sw,
                });
                g.push(Op::Copy {
                    src: rows,
                    src_off: 0,
                    dst: src,
                    dst_off: s * sw,
                    n: t * sw,
                });
            }
            let n_read = cm.blocks * r;
            for (src, win, fill) in [
                (cs.src_kv, cs.win_kv, 0.0),
                (cs.src_score, cs.win_score, f32::NEG_INFINITY),
            ] {
                let gather =
                    |idx_off, src_col, group, dst_group_stride, dst_row_off| Op::GatherRows {
                        src,
                        idx: cm.read_idx,
                        dst: win,
                        rows: n_read,
                        idx_off,
                        src_stride: sw,
                        src_col,
                        n: wd,
                        group,
                        dst_group_stride,
                        dst_row_off,
                        fill,
                    };
                if self.overlap {
                    // Each block's window is its `ratio` previous-window rows (their first half,
                    // and the sentinel — zero value, `-inf` score — before the sequence start),
                    // then its `ratio` current-window rows (their second half):
                    // `Op::GatherRows`' overlap form.
                    g.push(gather(0, 0, r, 2 * r, 0));
                    g.push(gather(n_read, wd, r, 2 * r, r));
                } else {
                    // Each block's own `ratio` source rows, in index order (`Op::GatherRows`' HCA
                    // form). No HCA read is before the sequence start, so the fill is unused.
                    g.push(gather(0, 0, 1, 1, 0));
                }
            }
            // Per channel: softmax of the scores over the block's window, weighting its values.
            g.push(Op::CompressPool {
                values: cs.win_kv,
                scores: cs.win_score,
                dst: cs.pooled,
                blocks: cm.blocks,
                window: if self.overlap { 2 * r } else { r },
                n_embd: wd,
            });
            g.push(Op::RmsNorm {
                x: cs.pooled,
                weight: w.norm,
                dst: cs.pooled,
                rows: cm.blocks,
                dim: wd,
                eps,
            });
            // Roped at the block's FIRST position (`state_write_pos`), with the compressed-layer
            // rope, on the row's own tail.
            rope.with_head_dim(wd)
                .tail(g, cs.pooled, cs.pack, cm.blocks, 1, cm.block_pos, false);
            g.push(Op::WriteKv {
                src: cs.pooled,
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
            for (rows, ring) in [(sc.kv, ring_kv), (sc.score, ring_score)] {
                g.push(Op::Copy {
                    src: rows,
                    src_off: src0 * sw,
                    dst: ring,
                    dst_off: dst0 * sw,
                    n: n as u32 * sw,
                });
            }
            i += n;
        }
    }
}

// The synthetic V4 GGUF the integration tests build (`tests/dsv4_fixture`), for the session tests,
// which drive `SeamKv` through the crate-private runner.
#[cfg(test)]
#[path = "../../../tests/dsv4_fixture/mod.rs"]
mod dsv4_fixture;

#[cfg(test)]
mod tests;
