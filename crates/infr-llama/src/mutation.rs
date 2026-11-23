//! Bugs a test plants in ONE loaded model, to show that the GPU tolerance gate fails on them
//! (`docs/gpu-tolerance.md`, "Mutations"). Compiled only with the `mutations` feature, so a normal
//! build has no trace of them.
//!
//! A bug is planted with [`crate::SeamModel::plant`] and lives in that model's [`crate::Config`]:
//! every graph the model builds, on any backend, carries it, and no other model does. The tests
//! load the model twice, plant the bug in one copy, run that copy on the GPU and score it against
//! the clean copy's CPU reference. Bugs that live in the weights (a mis-scaled block, a wrong
//! nibble map, swapped projections, a RoPE pairing error, a router bias ignored) are not here: the
//! tests write them into a copy of the model file, where the backend under test runs its
//! production code on wrong weights. Some are of neither kind: [`Planted::ReplayFeedsStaleToken`]
//! and [`Planted::ReplayRewritesLastRow`] are in how the runner drives a decode recording, for
//! the check that holds the replayed and the chained decode to the per-token path
//! (`docs/gpu-tolerance.md`, "The greedy check").

use crate::Config;
use infr_core::graph::Graph;
use infr_core::tensor::{DType, TensorDesc, TensorId};

/// A bug in the graph a model builds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Planted {
    /// The query is rotated at its position + 1 while every key keeps its own position, so each
    /// attention score sees the query one position late. Reaches the standard attention block's
    /// query rope (the fused per-head norm + rope of qwen3 and gemma, the plain rope of llama and
    /// ERNIE).
    RopeQueryAhead,
    /// Each query row's own key, the last one it may see, is left out of its attention, in every
    /// graph that does not start at position 0: every decode step and every prefill chunk after
    /// the first. The first chunk is untouched.
    MaskDropsLastKey,
    /// The RMSNorm epsilon of every norm the model's `rms_eps` reaches, times this factor.
    RmsEpsScaled(f32),
    /// The attention sublayer's output of this layer is never added to the residual stream.
    AttnResidualDropped { layer: usize },
    /// Not a bug of the graph but of how one decode recording is driven: every execution of it
    /// after the first within a generation is fed the token before the one it should be (a
    /// replayed step's, and the first step's of every chained submit after the first). A plan
    /// executed once, which is all a teacher-forced run ever does with one, cannot show it.
    ReplayFeedsStaleToken,
    /// Also in the driving: a generation that executed its decode recording more than once
    /// leaves the first layer's V row of the last position it wrote with every value's sign
    /// flipped. Every sampled token is already out and none changes; what the generation leaves
    /// in the KV cache does.
    ReplayRewritesLastRow,
}

impl Planted {
    /// A graph carrying the bug computes what the position it was built at says, so a recording
    /// replayed at other positions would run without it: such graphs opt out of the replayed
    /// decode, and every decode step is built at its own position.
    fn position_baked(self) -> bool {
        matches!(self, Planted::RopeQueryAhead | Planted::MaskDropsLastKey)
    }
}

/// Write `bug` into `c` (see [`crate::SeamModel::plant`]). An epsilon bug changes the
/// configuration's `rms_eps` straight away; one bug per model.
pub(crate) fn plant(c: &mut Config, bug: Planted) {
    assert!(c.planted.is_none(), "one bug per model: {:?}", c.planted);
    if let Planted::RmsEpsScaled(f) = bug {
        c.rms_eps *= f;
    }
    c.planted = Some(bug);
}

/// The positions the query rope reads: `positions` itself, or for [`Planted::RopeQueryAhead`] an
/// input of the graph's own holding each row's position + 1, its values pushed onto `step_i32`
/// for the driver to upload and bind.
pub(crate) fn query_positions(
    c: &Config,
    g: &mut Graph,
    positions: TensorId,
    start_pos: usize,
    batch: usize,
    step_i32: &mut Vec<(TensorId, Vec<i32>)>,
) -> TensorId {
    if c.planted != Some(Planted::RopeQueryAhead) {
        return positions;
    }
    let ahead = g.input(TensorDesc::new(vec![batch], DType::I32));
    let first = start_pos as i32 + 1;
    step_i32.push((ahead, (first..first + batch as i32).collect()));
    ahead
}

/// The `(pos, kv_len)` an attention over `batch` rows from `start_pos` is built with: the true
/// ones, or for [`Planted::MaskDropsLastKey`] past position 0 both one short, so every row's
/// causal range ends one key early and the cache is read one row short.
pub(crate) fn attention_span(c: &Config, start_pos: usize, batch: usize) -> (u32, u32) {
    let drop = usize::from(c.planted == Some(Planted::MaskDropsLastKey) && start_pos > 0);
    ((start_pos - drop) as u32, (start_pos + batch - drop) as u32)
}

/// Whether layer `l`'s attention output is added to the residual stream.
pub(crate) fn keeps_attn_residual(c: &Config, l: usize) -> bool {
    c.planted != Some(Planted::AttnResidualDropped { layer: l })
}

/// The token the decode step at `pos` feeds: `cur[pos]`, or for
/// [`Planted::ReplayFeedsStaleToken`], when the generation's one recording has `executed` before,
/// the token before it.
pub(crate) fn fed_token(c: &Config, cur: &[u32], pos: usize, executed: bool) -> u32 {
    let stale = c.planted == Some(Planted::ReplayFeedsStaleToken) && executed && pos > 0;
    cur[pos - usize::from(stale)]
}

/// Whether a generation whose one recording executed more than once (`replayed`) leaves its
/// last written V row negated ([`Planted::ReplayRewritesLastRow`]).
pub(crate) fn negates_last_row(c: &Config, replayed: bool) -> bool {
    c.planted == Some(Planted::ReplayRewritesLastRow) && replayed
}

/// Flip the sign of every f16 value in `row` (little-endian bytes).
pub(crate) fn negate_f16(row: &mut [u8]) {
    for v in row.as_chunks_mut::<2>().0 {
        v[1] ^= 0x80;
    }
}

/// Opt `g` out of the replayed decode when its bug is baked at its build position.
pub(crate) fn opt_out_of_replay(c: &Config, g: &mut Graph) {
    if c.planted.is_some_and(Planted::position_baked) {
        g.no_decode_replay = true;
    }
}
