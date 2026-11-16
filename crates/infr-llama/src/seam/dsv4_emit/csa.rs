//! The CSA tier's layer emit: its two overlapping compressors (the attention's own cache and the
//! lightning indexer's LID cache), then the indexer that scores the LID cache and keeps the top-k
//! blocks, and the segment that hands the kept CSA rows to the attention.
//!
//! **The Hadamard rotation is not ported.** `build_lid_top_k` multiplies the indexer query, and
//! `build_attention_impl` the LID compressed row (`lid_state_compress_rot`), by one orthonormal
//! Walsh-Hadamard matrix (`llama_kv_cache` forces the rotation on the LID cache because its head
//! width is the indexer's). An orthogonal transform applied to both sides of every dot product
//! changes no score in exact arithmetic; what differs is where the f16 rounding of the stored LID
//! row lands, which moves a score only at the scale of that rounding (plan 09 § 4, "What S7 must do
//! about indexer precision").

use super::{Dsv4LayerIn, Dsv4Rope, Dsv4Tier};
use crate::seam::weights::{Dsv4CompressorW, Dsv4IndexerW};
use crate::seam::DSV4_TIER_BUFS;
use crate::Config;
use infr_core::graph::{CompSegment, Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};

/// The CSA tier's per-graph indexer scratch, declared when some row of the graph sees a block.
pub(super) struct Dsv4IndexerScratch {
    /// `[tokens, indexer_n_head * indexer_head_size]`: the indexer queries, roped in place.
    q: TensorId,
    /// `[tokens * indexer_n_head, rope_dim]` scratch for their rope.
    pack: TensorId,
    /// `[tokens, indexer_n_head]`: the per-head weights, unscaled (`Op::LightningIndexer` scales).
    weights: TensorId,
    /// `[tokens, top_k]` i32: the selected block indices.
    top_k_idx: TensorId,
    /// `[tokens, len]`: the selection as an additive `0` / `-inf` score bias.
    bias: TensorId,
    /// Blocks the indexer scores: the graph's widest visible count, the segment's `len`.
    len: u32,
    /// Blocks it keeps per row: `min(len, indexer_top_k)`, the reference's
    /// `min(indexer_score->ne[0], indexer_top_k)` — whose padded width keeps every visible block
    /// either way, which `len` does too.
    top_k: u32,
}

impl Dsv4IndexerScratch {
    /// The scratch of a `tokens`-row graph whose rows see at most `len` blocks.
    pub(super) fn new(g: &mut Graph, c: &Config, tokens: usize, len: usize) -> Self {
        assert!(len > 0, "the indexer scores at least one block");
        let top_k = len.min(c.indexer_top_k);
        assert!(
            top_k > 0,
            "attention.indexer.top_k is 0: the indexer would keep no block"
        );
        let f32d = |n: usize| TensorDesc::new(vec![n.max(1)], DType::F32);
        let nh = c.indexer_n_head;
        Dsv4IndexerScratch {
            q: g.internal(f32d(tokens * nh * c.indexer_head_size)),
            pack: g.internal(f32d(tokens * nh * c.rope_dim)),
            weights: g.internal(f32d(tokens * nh)),
            top_k_idx: g.internal(TensorDesc::new(vec![tokens * top_k], DType::I32)),
            bias: g.internal(f32d(tokens * len)),
            len: len as u32,
            top_k: top_k as u32,
        }
    }
}

impl Dsv4Tier {
    /// Emit one CSA layer's compressed half: the attention compressor `attn` into its CSA cache,
    /// the indexer's own compressor into the LID cache, then — once some row sees a block — the
    /// indexer's top-k over the LID cache, expanded into the bias of the returned CSA segment.
    /// `bufs` are the layer's six compressed-tier Inputs (CSA cache and ring, then LID cache and
    /// ring). Returns `None` while no block is visible to any row: the reference's all-masked
    /// compressed half adds nothing, and the indexer and mask ops refuse zero keys (plan 09 "Notes
    /// for S6 and S7").
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_csa(
        &self,
        g: &mut Graph,
        c: &Config,
        attn: &Dsv4CompressorW,
        ix: &Dsv4IndexerW,
        io: Dsv4LayerIn,
        bufs: &[TensorId],
        rope: Dsv4Rope,
        eps: f32,
    ) -> Option<CompSegment> {
        assert!(self.overlap, "emit_csa on the ratio-{} tier", self.ratio);
        let (csa, lid) = bufs.split_at(DSV4_TIER_BUFS);
        let n_embd = c.n_embd as u32;
        // Both compressors commit before the indexer reads the LID cache and the attention the
        // CSA cache, so a token that completes a block can select and attend it.
        self.compress(g, 0, attn, io.x, n_embd, csa, rope, eps);
        self.compress(g, 1, &ix.comp, io.x, n_embd, lid, rope, eps);
        let is = self.indexer.as_ref()?;

        let (t, nh, ihs) = (
            self.tokens,
            c.indexer_n_head as u32,
            c.indexer_head_size as u32,
        );
        // `build_lid_top_k`: the query is `indexer.attn_q_b · qr` with no norm, its tail roped at
        // the token's position with the compressed-layer rope.
        g.push(Op::Linear {
            x: io.qr,
            weight: ix.attn_q_b,
            dst: is.q,
            m: t,
            in_f: c.q_lora_rank as u32,
            out_f: nh * ihs,
            w_off: 0,
        });
        rope.with_head_dim(ihs)
            .tail(g, is.q, is.pack, t, nh, io.positions, false);
        g.push(Op::Linear {
            x: io.x,
            weight: ix.proj,
            dst: is.weights,
            m: t,
            in_f: n_embd,
            out_f: nh,
            w_off: 0,
        });
        g.push(Op::LightningIndexer {
            q: is.q,
            k_cache: lid[0],
            weights: is.weights,
            dst: is.top_k_idx,
            rows: t,
            kv_len: is.len,
            n_head: nh,
            head_dim: ihs,
            top_k: is.top_k,
            // `indexer_weights` scaled by `1/sqrt(indexer_head_size * indexer_n_head)`.
            scale: 1.0 / ((ihs * nh) as f32).sqrt(),
            pos: self.start_pos,
            block_ratio: self.ratio,
        });
        // `build_top_k_mask`: `0` on the kept blocks, `-inf` elsewhere; the segment's own causal
        // rule then masks a kept block the row has not completed, as the reference's sum with the
        // compressed mask does.
        g.push(Op::TopkMask {
            idx: is.top_k_idx,
            dst: is.bias,
            rows: t,
            kv_len: is.len,
            top_k: is.top_k,
        });
        self.segment(csa[0], Some(is.bias))
    }
}
